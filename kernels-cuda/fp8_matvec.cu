// FP8 (e4m3) lm_head path (perf-roadmap, 2026-06-10).
//
// Two kernels:
//  - fp8_quantize_rows_kernel: one-time load-side quantization of a BF16
//    matrix to e4m3 with a per-row f32 scale (scale = row_amax / 448,
//    hardware round-to-nearest-even via __nv_cvt_float_to_fp8). Runs once
//    at engine load; never on the hot path.
//  - fp8_matvec_kernel: logits = (decode(W_e4m3) * row_scale) @ x. One warp
//    per output row, 8 rows per CTA, B vector staged in SMEM as f32.
//    Reads 1 byte/weight instead of BF16's 2 — the lm_head read was
//    1.65 ms/token at 86% of DRAM peak, this halves the bytes.
//
// Numerics gate: per-row scales are applied on the output (exact), e4m3
// decode is the IEEE one (hardware cvt). The offline probe measured 0/27
// argmax flips vs BF16 on real final_normed vectors; the smoke case below
// (smoke.cu) gates quantize+matvec against a host reference.

#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQuantThreads = 256;

__global__ void fp8_quantize_rows_kernel(const __nv_bfloat16 *__restrict__ w,
                                         uint8_t *__restrict__ out,
                                         float *__restrict__ row_scale,
                                         size_t K) {
  const size_t m = blockIdx.x;
  const __nv_bfloat16 *row = w + m * K;
  __shared__ float sm_amax[kQuantThreads / 32];

  float amax = 0.0f;
  for (size_t k = threadIdx.x; k < K; k += blockDim.x) {
    amax = fmaxf(amax, fabsf(__bfloat162float(row[k])));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  }
  if ((threadIdx.x & 31) == 0) {
    sm_amax[threadIdx.x >> 5] = amax;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_amax = sm_amax[0];
#pragma unroll
    for (int i = 1; i < kQuantThreads / 32; ++i) {
      block_amax = fmaxf(block_amax, sm_amax[i]);
    }
    sm_amax[0] = block_amax;
  }
  __syncthreads();
  const float scale = (sm_amax[0] > 0.0f) ? (sm_amax[0] / 448.0f) : 1.0f;
  const float inv_scale = 1.0f / scale;
  if (threadIdx.x == 0) {
    row_scale[m] = scale;
  }
  for (size_t k = threadIdx.x; k < K; k += blockDim.x) {
    const float v = __bfloat162float(row[k]) * inv_scale;
    out[m * K + k] = static_cast<uint8_t>(
        __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
  }
}

constexpr int kMvRowsPerCta = 8; // one warp per row
constexpr int kMvThreads = kMvRowsPerCta * 32;

// Up to this many input rows are staged together and share ONE pass over
// the weight bytes (grid.y batching would re-read the 1.27 GB matrix per
// row — measured 4x slower on the 5-row MTP verify). Matches the MTP
// verify maximum (8 drafts + 1, depth-8 unlock).
constexpr int kMvMaxRows = 10;

__global__ void __launch_bounds__(kMvThreads)
    fp8_matvec_kernel(const uint8_t *__restrict__ w,
                      const float *__restrict__ row_scale,
                      const __nv_bfloat16 *__restrict__ input,
                      __nv_bfloat16 *__restrict__ output, size_t M, size_t K,
                      size_t input_stride, int rows) {
  // BF16 staging: rows x K x 4B of f32 would blow the 99 KB SMEM cap at
  // rows=5 (K=5120); BF16 keeps rows=8 at 80 KB, cvt happens at the FMA.
  extern __shared__ __nv_bfloat16 sm_b[]; // [rows, K]
  for (int r = 0; r < rows; ++r) {
    const __nv_bfloat16 *b = input + static_cast<size_t>(r) * input_stride;
    for (size_t k = threadIdx.x; k < K; k += blockDim.x) {
      sm_b[r * K + k] = b[k];
    }
  }
  __syncthreads();

  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane = threadIdx.x & 31;
  const size_t m = static_cast<size_t>(blockIdx.x) * kMvRowsPerCta + warp_id;
  if (m >= M) {
    return;
  }
  const uint4 *row16 = reinterpret_cast<const uint4 *>(w + m * K);
  const size_t vecs = K / 16;
  float acc[kMvMaxRows];
#pragma unroll
  for (int r = 0; r < kMvMaxRows; ++r) {
    acc[r] = 0.0f;
  }
  for (size_t i = lane; i < vecs; i += 32) {
    const uint4 v = row16[i];
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
    float wdec[16];
#pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(
            static_cast<__nv_fp8x2_storage_t>(
                (words[wi] >> (16 * p)) & 0xffffu),
            __NV_E4M3);
        const float2 f2 =
            __half22float2(*reinterpret_cast<const __half2 *>(&h2));
        wdec[wi * 4 + p * 2] = f2.x;
        wdec[wi * 4 + p * 2 + 1] = f2.y;
      }
    }
    for (int r = 0; r < rows; ++r) {
      const __nv_bfloat16 *bk = sm_b + r * K + i * 16;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        acc[r] = fmaf(wdec[j], __bfloat162float(bk[j]), acc[r]);
      }
    }
  }
  const float scale = row_scale[m];
  for (int r = 0; r < rows; ++r) {
    float a = acc[r];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      a += __shfl_xor_sync(0xffffffff, a, off);
    }
    if (lane == 0) {
      output[static_cast<size_t>(r) * M + m] = __float2bfloat16(a * scale);
    }
  }
}

} // namespace

extern "C" int
qwen36_fp8_quantize_rows(const qwen36_fp8_quantize_rows_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->out_features == 0 || spec->in_features == 0 ||
      spec->weight_bf16.ptr == 0 || spec->weight_e4m3.ptr == 0 ||
      spec->row_scale_f32.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  fp8_quantize_rows_kernel<<<static_cast<unsigned>(spec->out_features),
                             kQuantThreads, 0,
                             qwen36_internal_active_stream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(spec->weight_bf16.ptr)),
      reinterpret_cast<uint8_t *>(
          static_cast<uintptr_t>(spec->weight_e4m3.ptr)),
      reinterpret_cast<float *>(
          static_cast<uintptr_t>(spec->row_scale_f32.ptr)),
      spec->in_features);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_fp8_matvec(const qwen36_fp8_matvec_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->out_features == 0 || spec->in_features == 0 || spec->rows == 0 ||
      spec->rows > static_cast<size_t>(kMvMaxRows) ||
      spec->weight_e4m3.ptr == 0 || spec->row_scale_f32.ptr == 0 ||
      spec->input_bf16.ptr == 0 || spec->output_bf16.ptr == 0 ||
      (spec->in_features % 16) != 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const size_t smem_bytes =
      spec->rows * spec->in_features * sizeof(__nv_bfloat16);
  cudaFuncSetAttribute(fp8_matvec_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem_bytes));
  const dim3 grid(static_cast<unsigned>(
      (spec->out_features + kMvRowsPerCta - 1) / kMvRowsPerCta));
  fp8_matvec_kernel<<<grid, kMvThreads, smem_bytes,
                      qwen36_internal_active_stream()>>>(
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(spec->weight_e4m3.ptr)),
      reinterpret_cast<const float *>(
          static_cast<uintptr_t>(spec->row_scale_f32.ptr)),
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(spec->input_bf16.ptr)),
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(spec->output_bf16.ptr)),
      spec->out_features, spec->in_features, spec->input_stride,
      static_cast<int>(spec->rows));
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
