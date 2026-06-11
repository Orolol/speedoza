// lm_head FP8 e4m3 (W8A16): per-row-quantized weight + BF16 activations,
// FP32 accumulation. Replaces the BF16 lm_head GEMV (2.54 GiB read/token,
// ~6.3 sequential full-vocab GEMVs per MTP=4 verify cycle) with a 1.27 GiB
// e4m3 copy — ~2x the bandwidth-bound throughput AND −1.27 GiB resident
// once the BF16 original is dropped.
//
// Quantization contract = the offline probe that opened this lane
// (scripts/lmhead_fp8_probe.py, 2026-06-11, 0/28 argmax flips): per-row
// scale = amax(row)/448, saturating float->e4m3 cast, dequant w = q * scale
// applied once per output element after the FP32 dot.
//
// GEMV layout contract (matches the cuBLAS paths it replaces):
//   input  X [n, cols]  row-major BF16 (n <= QWEN36_LM_HEAD_FP8_MAX_N)
//   output Y [n, rows]  row-major BF16 (sample_rows_argmax reads
//                       logits + row * vocab)

#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQuantThreads = 256;
constexpr int kGemvThreads = 256; // 8 warps, one output row per warp
constexpr int kGemvWarpsPerCta = kGemvThreads / 32;
constexpr int kGemvKTile = 1024; // SMEM X tile per K iteration
constexpr int kMaxN = QWEN36_LM_HEAD_FP8_MAX_N;

template <typename T> T *ptr(qwen36_device_ptr_t p) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(p.ptr));
}


__global__ void lm_head_fp8_quantize_kernel(const __nv_bfloat16 *weight,
                                            uint8_t *weight_e4m3,
                                            float *row_scales, size_t rows,
                                            size_t cols) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const __nv_bfloat16 *w = weight + row * cols;

  __shared__ float warp_amax[kQuantThreads / 32];
  float amax = 0.0f;
  for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
    amax = fmaxf(amax, fabsf(__bfloat162float(w[c])));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
  }
  if ((threadIdx.x & 31) == 0) {
    warp_amax[threadIdx.x >> 5] = amax;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_amax = 0.0f;
    for (int i = 0; i < kQuantThreads / 32; ++i) {
      block_amax = fmaxf(block_amax, warp_amax[i]);
    }
    // amax == 0 would make the scale 0 and the dequant NaN-free but the
    // quantized row all-zero anyway; keep scale 1 so dequant stays exact.
    warp_amax[0] = block_amax > 0.0f ? block_amax / 448.0f : 1.0f;
  }
  __syncthreads();
  const float scale = warp_amax[0];
  if (threadIdx.x == 0) {
    row_scales[row] = scale;
  }
  uint8_t *out = weight_e4m3 + row * cols;
  const float inv_scale = 1.0f / scale;
  for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
    const __nv_fp8_e4m3 q(__bfloat162float(w[c]) * inv_scale);
    out[c] = reinterpret_cast<const uint8_t &>(q);
  }
}

// Pure streaming GEMV, register-blocked over output rows: each warp walks
// R consecutive weight rows with 16-byte loads, so each x value (read via
// __ldg from the L1/L2-resident X, 10-130 KB total) is reused R times and
// R independent DRAM streams stay in flight per warp. No shared memory,
// no barriers. R shrinks as N grows to keep acc[R][N] in registers.
template <int N> __host__ __device__ constexpr int lm_head_fp8_rows_per_warp() {
  // N=1 measured FASTER without row blocking (790 vs 883 us — x reuse is
  // worthless at one RHS and the extra registers cost occupancy); blocking
  // pays once several x loads amortize per weight byte.
  return N == 1 ? 1 : (N <= 8 ? 2 : 1);
}

template <int N>
__global__ void lm_head_fp8_gemv_kernel(const uint8_t *__restrict__ weight,
                                        const float *__restrict__ row_scales,
                                        const __nv_bfloat16 *__restrict__ x,
                                        __nv_bfloat16 *__restrict__ y,
                                        size_t rows, size_t cols) {
  constexpr int R = lm_head_fp8_rows_per_warp<N>();
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const size_t row0 =
      (static_cast<size_t>(blockIdx.x) * kGemvWarpsPerCta + warp_id) * R;
  if (row0 >= rows) {
    return;
  }
  const int r_count =
      static_cast<int>(min(static_cast<size_t>(R), rows - row0));

  float acc[R][N];
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      acc[r][i] = 0.0f;
    }
  }

  const uint8_t *w0 = weight + row0 * cols;
  if (r_count == R) {
    for (size_t kk = static_cast<size_t>(lane_id) * 16; kk + 15 < cols;
         kk += 32 * 16) {
      uint4 packed[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        packed[r] = *reinterpret_cast<const uint4 *>(w0 + r * cols + kk);
      }
#pragma unroll
      for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const size_t k = kk + wi * 4 + b;
          float xv[N];
#pragma unroll
          for (int i = 0; i < N; ++i) {
            xv[i] = __bfloat162float(__ldg(x + i * cols + k));
          }
#pragma unroll
          for (int r = 0; r < R; ++r) {
            const uint32_t word =
                wi == 0 ? packed[r].x
                        : (wi == 1 ? packed[r].y
                                   : (wi == 2 ? packed[r].z : packed[r].w));
            const uint8_t byte = static_cast<uint8_t>(word >> (8 * b));
            const float wv =
                float(reinterpret_cast<const __nv_fp8_e4m3 &>(byte));
#pragma unroll
            for (int i = 0; i < N; ++i) {
              acc[r][i] += wv * xv[i];
            }
          }
        }
      }
    }
  } else {
    // Ragged final warp: scalar row loop, same math.
    for (int r = 0; r < r_count; ++r) {
      const uint8_t *w_row = w0 + r * cols;
      for (size_t kk = static_cast<size_t>(lane_id) * 16; kk + 15 < cols;
           kk += 32 * 16) {
        const uint4 packed = *reinterpret_cast<const uint4 *>(w_row + kk);
        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const uint8_t byte = static_cast<uint8_t>(words[wi] >> (8 * b));
            const float wv =
                float(reinterpret_cast<const __nv_fp8_e4m3 &>(byte));
            const size_t k = kk + wi * 4 + b;
#pragma unroll
            for (int i = 0; i < N; ++i) {
              acc[r][i] += wv * __bfloat162float(__ldg(x + i * cols + k));
            }
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (r >= r_count) {
      break;
    }
    const float scale = row_scales[row0 + r];
#pragma unroll
    for (int i = 0; i < N; ++i) {
      float v = acc[r][i];
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
      }
      if (lane_id == 0) {
        y[static_cast<size_t>(i) * rows + row0 + r] =
            __float2bfloat16(v * scale);
      }
    }
  }
}

template <int N>
void launch_lm_head_fp8_gemv(const qwen36_lm_head_fp8_gemv_spec_t *spec) {
  constexpr int rows_per_cta = lm_head_fp8_rows_per_warp<N>() * kGemvWarpsPerCta;
  const unsigned int grid = static_cast<unsigned int>(
      (spec->rows + rows_per_cta - 1) / rows_per_cta);
  lm_head_fp8_gemv_kernel<N><<<grid, kGemvThreads, 0,
                               qwen36_internal_active_stream()>>>(
      ptr<const uint8_t>(spec->weight_e4m3),
      ptr<const float>(spec->row_scales_f32),
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->rows, spec->cols);
}

} // namespace

extern "C" int
qwen36_lm_head_fp8_quantize(const qwen36_lm_head_fp8_quantize_spec_t *spec) {
  if (spec == nullptr || spec->rows == 0 || spec->cols == 0 ||
      spec->weight_bf16.ptr == 0 || spec->weight_e4m3.ptr == 0 ||
      spec->row_scales_f32.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  lm_head_fp8_quantize_kernel<<<static_cast<unsigned int>(spec->rows),
                                kQuantThreads, 0,
                                qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->weight_bf16),
      ptr<uint8_t>(spec->weight_e4m3), ptr<float>(spec->row_scales_f32),
      spec->rows, spec->cols);
  return cudaGetLastError() == cudaSuccess ? QWEN36_STATUS_SUCCESS
                                           : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_lm_head_fp8_gemv(const qwen36_lm_head_fp8_gemv_spec_t *spec) {
  if (spec == nullptr || spec->rows == 0 || spec->cols == 0 || spec->n == 0 ||
      spec->n > static_cast<size_t>(kMaxN) || (spec->cols & 15) != 0 ||
      spec->weight_e4m3.ptr == 0 || spec->row_scales_f32.ptr == 0 ||
      spec->input_bf16.ptr == 0 || spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  switch (spec->n) {
  case 1: launch_lm_head_fp8_gemv<1>(spec); break;
  case 2: launch_lm_head_fp8_gemv<2>(spec); break;
  case 3: launch_lm_head_fp8_gemv<3>(spec); break;
  case 4: launch_lm_head_fp8_gemv<4>(spec); break;
  case 5: launch_lm_head_fp8_gemv<5>(spec); break;
  case 6: launch_lm_head_fp8_gemv<6>(spec); break;
  case 7: launch_lm_head_fp8_gemv<7>(spec); break;
  case 8: launch_lm_head_fp8_gemv<8>(spec); break;
  case 9: launch_lm_head_fp8_gemv<9>(spec); break;
  case 10: launch_lm_head_fp8_gemv<10>(spec); break;
  case 11: launch_lm_head_fp8_gemv<11>(spec); break;
  case 12: launch_lm_head_fp8_gemv<12>(spec); break;
  case 13: launch_lm_head_fp8_gemv<13>(spec); break;
  case 14: launch_lm_head_fp8_gemv<14>(spec); break;
  case 15: launch_lm_head_fp8_gemv<15>(spec); break;
  case 16: launch_lm_head_fp8_gemv<16>(spec); break;
  default: return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  return cudaGetLastError() == cudaSuccess ? QWEN36_STATUS_SUCCESS
                                           : QWEN36_STATUS_CUDA_ERROR;
}
