// Per-shape microbench of the Direction B NVFP4 decode GEMV (roadmap P3).
//
// Times qwen36_decode_nvfp4_gemv on every production decode shape and
// reports achieved GB/s vs the 1.79 TB/s 5090 peak, plus the CTA/wave
// arithmetic. This is the paired-microbench instrument for the SMEM-paging
// prototype's kill-gate (<+20% kernel-level BW on M=5120 -> abandon).
//
// Build (matches smoke):
//   nvcc -std=c++17 -O2 -arch=sm_120a -I kernels-cuda/include \
//     kernels-cuda/tools/gemv_shape_bench.cu -L target/cuda \
//     -lqwen36_fp4_kernels -o target/cuda/gemv_shape_bench

#include "qwen36_fp4.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t err__ = (x);                                                   \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),   \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static qwen36_device_ptr_t dev_alloc_bytes(size_t bytes) {
  qwen36_device_allocation_t alloc{};
  if (qwen36_cuda_malloc(&alloc, bytes) != QWEN36_STATUS_SUCCESS) {
    fprintf(stderr, "alloc %zu bytes failed\n", bytes);
    exit(1);
  }
  return alloc.ptr;
}

struct Shape {
  const char *name;
  size_t m, k;
  int per_token; // launches per decode token
};

int main() {
  // Production decode shapes (dims from config.json; counts per token).
  const Shape shapes[] = {
      {"mlp.gate_up (fused)", 34816, 5120, 64},
      {"mlp.down", 5120, 17408, 64},
      {"deltanet.in_proj (fused)", 16640, 5120, 48},
      {"deltanet.out_proj", 5120, 3584, 48},
      {"attn.q_proj", 12288, 5120, 16},
      {"attn.kv_proj", 1024, 5120, 32},
      {"attn.o_proj", 5120, 6144, 16},
  };
  const double kPeak = 1.79e12;

  int dev_sms = 0;
  cudaDeviceProp props{};
  CK(cudaGetDeviceProperties(&props, 0));
  dev_sms = props.multiProcessorCount;
  printf("device: %s, %d SMs\n\n", props.name, dev_sms);
  printf("| shape | M | K | CTAs | GB | us | GB/s | %%peak | cuBLASLt us |\n");
  printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");

  double total_ms_token = 0.0, total_gb_token = 0.0;

  for (const Shape &s : shapes) {
    const size_t a_fp4_bytes = s.m * s.k / 2;
    const size_t a_scale_bytes =
        ((s.m + 127) / 128) * ((s.k / 16 + 3) / 4) * 512;
    const size_t b_fp4_bytes = s.k / 2;
    const size_t b_scale_bytes = ((s.k / 16 + 3) / 4) * 512;

    // Rotate across enough A-copies that successive reps read cold data —
    // a single buffer would go L2-resident (96 MB L2) and measure L2, not
    // DRAM, bandwidth (first version of this bench reported 2.4 TB/s).
    const size_t kColdBytes = 800ull << 20;
    const int n_copies =
        static_cast<int>(kColdBytes / (a_fp4_bytes + a_scale_bytes)) + 1;
    std::vector<qwen36_device_ptr_t> a_fp4s(n_copies), a_scales(n_copies);
    for (int c = 0; c < n_copies; ++c) {
      a_fp4s[c] = dev_alloc_bytes(a_fp4_bytes);
      a_scales[c] = dev_alloc_bytes(a_scale_bytes);
      CK(cudaMemset(reinterpret_cast<void *>(a_fp4s[c].ptr), 0x35,
                    a_fp4_bytes));
      CK(cudaMemset(reinterpret_cast<void *>(a_scales[c].ptr), 0x38,
                    a_scale_bytes));
    }
    qwen36_nvfp4_gemm_spec_t spec{};
    spec.m = s.m;
    spec.n = 1;
    spec.k = s.k;
    spec.a_fp4 = a_fp4s[0];
    spec.a_scale = a_scales[0];
    spec.b_fp4 = dev_alloc_bytes(b_fp4_bytes);
    spec.b_scale = dev_alloc_bytes(b_scale_bytes);
    spec.c_bf16 = dev_alloc_bytes(s.m * 2);
    spec.alpha = 1.0f;
    CK(cudaMemset(reinterpret_cast<void *>(spec.b_fp4.ptr), 0x35,
                  b_fp4_bytes));
    CK(cudaMemset(reinterpret_cast<void *>(spec.b_scale.ptr), 0x38,
                  b_scale_bytes));

    int rc = qwen36_decode_nvfp4_gemv(&spec);
    if (rc != QWEN36_STATUS_SUCCESS) {
      printf("| %s | %zu | %zu | shape rejected (rc=%d) | | | | |\n", s.name,
             s.m, s.k, rc);
      continue;
    }
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int reps = 200;
    // warmup
    for (int i = 0; i < 20; ++i) {
      qwen36_decode_nvfp4_gemv(&spec);
    }
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; ++i) {
      spec.a_fp4 = a_fp4s[i % n_copies];
      spec.a_scale = a_scales[i % n_copies];
      qwen36_decode_nvfp4_gemv(&spec);
    }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double us = 1000.0 * ms / reps;
    const double gb =
        (a_fp4_bytes + a_scale_bytes + b_fp4_bytes + b_scale_bytes) / 1e9;
    const double gbps = gb / (us / 1e6);
    // Same shape through the cuBLASLt path for comparison.
    const size_t kWsBytes = 256u << 20;
    static qwen36_device_ptr_t ws = dev_alloc_bytes(kWsBytes);
    spec.workspace = ws;
    spec.workspace_bytes = kWsBytes;
    double us_lt = -1.0;
    if (qwen36_nvfp4_gemm(&spec) == QWEN36_STATUS_SUCCESS) {
      CK(cudaDeviceSynchronize());
      for (int i = 0; i < 20; ++i) {
        qwen36_nvfp4_gemm(&spec);
      }
      CK(cudaEventRecord(e0));
      for (int i = 0; i < reps; ++i) {
        spec.a_fp4 = a_fp4s[i % n_copies];
        spec.a_scale = a_scales[i % n_copies];
        qwen36_nvfp4_gemm(&spec);
      }
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      float ms_lt = 0.0f;
      CK(cudaEventElapsedTime(&ms_lt, e0, e1));
      us_lt = 1000.0 * ms_lt / reps;
    }
    spec.workspace = qwen36_device_ptr_t{};
    spec.workspace_bytes = 0;
    const size_t ctas = (s.m + 15) / 16;
    printf("| %s | %zu | %zu | %zu(x%d) | %.4f | %.1f | %.0f | %.0f%% | %.1f |\n",
           s.name, s.m, s.k, ctas, n_copies, gb, us, gbps,
           100.0 * gbps * 1e9 / kPeak, us_lt);
    total_ms_token += us / 1000.0 * s.per_token;
    total_gb_token += gb * s.per_token;
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
  }
  printf("\nper-token (all launches): %.2f ms, %.2f GB -> %.0f GB/s = %.0f%% "
         "peak (isolated-kernel upper bound)\n",
         total_ms_token, total_gb_token,
         total_gb_token / (total_ms_token / 1000.0),
         100.0 * total_gb_token * 1e9 / (total_ms_token / 1000.0) / kPeak);
  return 0;
}
