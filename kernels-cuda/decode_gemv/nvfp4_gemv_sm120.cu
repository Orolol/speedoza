// Direction B NVFP4 gemv kernel for Blackwell SM_120 — Phase B2 (Option C).
//
// Hand-rolled CUDA kernel for the NVFP4 weight × NVFP4 activation → BF16
// gemv that dominates the decode hot path. CUTLASS is intentionally NOT
// used here — the SM120 BlockScaled cooperative scheduler statically
// rejects a narrow N tile and the TMA descriptor builder fails at N=1
// (see docs/superpowers/notes/2026-05-04-direction-b-cutlass-blockers.md
// for the full breakdown). Going hand-rolled lets us pick the natural
// N=1 layout directly.
//
// B2 baseline: scalar dequant via the FP4 LUT + per-group e4m3 scales
// folded into a register accumulator. Mirrors `nvfp4_matvec_kernel`
// (kernels-cuda/ops.cu:733) — one warp per output row, 8 rows per CTA,
// 32 lanes cooperatively walking K. The differences vs that kernel:
//   - Activation is FP4 (not BF16); we apply the activation's per-group
//     e4m3 scale in addition to the weight's.
//   - We multiply by both tensor scales (a_scale_2, b_scale_2) and
//     `alpha` in the epilogue.
//
// B3+ will move to MMA atoms (`mma.kind::mxf4`), persistent grid, and
// TMA multicast for the activation broadcast.
//
// On any unsupported shape the kernel returns
// QWEN36_STATUS_NOT_IMPLEMENTED (5) and the Rust dispatcher routes back
// to cuBLASLt. Active env var: `QWEN36_DECODE_GEMV=1`.

#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

// FP4 (E2M1) decode LUT. Indices 0..7 = positive magnitudes,
// 8..15 = the same magnitudes with the sign bit set.
__device__ __constant__ float kGemvFp4Lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ float decode_e4m3_local(uint8_t code) {
  const int sign = (code & 0x80) != 0 ? -1 : 1;
  const int exponent = (code >> 3) & 0x0f;
  const int mantissa = code & 0x07;
  if (exponent == 0) {
    if (mantissa == 0) {
      return 0.0f;
    }
    return sign * ldexpf(static_cast<float>(mantissa) / 8.0f, -6);
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return sign * 448.0f;
  }
  return sign *
         ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

__host__ __device__ size_t gemv_div_ceil(size_t a, size_t b) {
  return (a + b - 1) / b;
}

__host__ __device__ size_t gemv_round_up(size_t v, size_t m) {
  return gemv_div_ceil(v, m) * m;
}

// cuBLASLt vec16 scale-swizzle layout. Mirrors `vec16_scale_offset` in
// kernels-cuda/ops.cu so we read the same bytes the existing GEMM
// produces. For the activation (N=1 column) the caller passes outer=0
// and the formula collapses to a small contiguous range plus a 512-byte
// stride between every 4th group.
__host__ __device__ size_t gemv_vec16_scale_offset(size_t inner, size_t outer,
                                                   size_t sf_inner_dim) {
  const size_t block_inner = (inner / 4) * 4;
  const size_t block_outer = outer / 128;
  const size_t block_offset = (block_inner + block_outer * sf_inner_dim) * 128;
  const size_t tile_outer = outer % 128;
  const size_t tile_inner = inner % 4;
  return block_offset + (tile_outer % 32) * 16 + (tile_outer / 32) * 4 +
         tile_inner;
}

constexpr int kRowsPerBlock = 8;
constexpr int kLanesPerRow = 32;
constexpr int kThreadsPerBlock = kRowsPerBlock * kLanesPerRow;

__global__ void __launch_bounds__(kThreadsPerBlock)
nvfp4_gemv_kernel(const uint8_t *__restrict__ a_fp4,
                  const uint8_t *__restrict__ a_scale,
                  const float *__restrict__ a_tensor_scale,
                  const uint8_t *__restrict__ b_fp4,
                  const uint8_t *__restrict__ b_scale,
                  const float *__restrict__ b_tensor_scale,
                  float alpha, __nv_bfloat16 *__restrict__ output, size_t M,
                  size_t K) {
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane = threadIdx.x & 31;
  const size_t row =
      static_cast<size_t>(blockIdx.x) * kRowsPerBlock + warp_id;
  if (row >= M) {
    return;
  }

  const size_t packed_cols = K / 2;
  const size_t scale_cols = K / 16;
  const size_t sf_inner_dim = gemv_round_up(scale_cols, 4);
  const uint8_t *row_weight = a_fp4 + row * packed_cols;

  const float a_ts =
      (a_tensor_scale == nullptr) ? 1.0f : __ldg(a_tensor_scale);
  const float b_ts =
      (b_tensor_scale == nullptr) ? 1.0f : __ldg(b_tensor_scale);

  float sum = 0.0f;

  for (size_t g = lane; g < scale_cols; g += kLanesPerRow) {
    const size_t weight_byte_off = g * 8;  // 16 elements / 2 packed
    const size_t scale_off_a =
        gemv_vec16_scale_offset(g, row, sf_inner_dim);
    const size_t scale_off_b = gemv_vec16_scale_offset(g, 0, sf_inner_dim);

    const float s_a = decode_e4m3_local(__ldg(a_scale + scale_off_a));
    const float s_b = decode_e4m3_local(__ldg(b_scale + scale_off_b));

    // Load 8 packed FP4 weight bytes (16 weight values).
    uint64_t packed_w;
    if ((weight_byte_off & 7u) == 0) {
      packed_w =
          *reinterpret_cast<const uint64_t *>(row_weight + weight_byte_off);
    } else {
      packed_w = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        packed_w |= static_cast<uint64_t>(row_weight[weight_byte_off + i])
                    << (i * 8);
      }
    }

    // Load 8 packed FP4 activation bytes.
    uint64_t packed_b;
    if ((weight_byte_off & 7u) == 0) {
      packed_b = *reinterpret_cast<const uint64_t *>(b_fp4 + weight_byte_off);
    } else {
      packed_b = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        packed_b |= static_cast<uint64_t>(b_fp4[weight_byte_off + i])
                    << (i * 8);
      }
    }

    float local = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint8_t wb = (packed_w >> (i * 8)) & 0xffu;
      const uint8_t bb = (packed_b >> (i * 8)) & 0xffu;
      const float w0 = kGemvFp4Lut[wb & 0xf];
      const float w1 = kGemvFp4Lut[(wb >> 4) & 0xf];
      const float a0 = kGemvFp4Lut[bb & 0xf];
      const float a1 = kGemvFp4Lut[(bb >> 4) & 0xf];
      local += w0 * a0 + w1 * a1;
    }
    sum += local * s_a * s_b;
  }

  // Single warp-shuffle reduction.
  for (int off = 16; off > 0; off >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, off);
  }
  if (lane == 0) {
    output[row] = __float2bfloat16(sum * alpha * a_ts * b_ts);
  }
}

template <typename T> T *as_device_ptr(qwen36_device_ptr_t p) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(p.ptr));
}

}  // namespace

extern "C" int qwen36_decode_nvfp4_gemv(
    const qwen36_nvfp4_gemm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->m == 0 || spec->n == 0 || spec->k == 0 ||
      spec->a_fp4.ptr == 0 || spec->a_scale.ptr == 0 ||
      spec->b_fp4.ptr == 0 || spec->b_scale.ptr == 0 ||
      spec->c_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  // Supported regime: gemv at N=1, both M and K aligned to the FP4 group
  // size (16). The Direction B spec target shapes (M up to ~35K, K=5120 /
  // 17408) all satisfy this.
  if (spec->n != 1 || (spec->m % 16) != 0 || (spec->k % 16) != 0) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  const size_t M = spec->m;
  const size_t K = spec->k;
  const dim3 block(kThreadsPerBlock, 1, 1);
  const dim3 grid(static_cast<unsigned>(gemv_div_ceil(M, kRowsPerBlock)), 1,
                  1);

  cudaStream_t stream = qwen36_internal_active_stream();
  nvfp4_gemv_kernel<<<grid, block, 0, stream>>>(
      as_device_ptr<const uint8_t>(spec->a_fp4),
      as_device_ptr<const uint8_t>(spec->a_scale),
      as_device_ptr<const float>(spec->a_scale_2),
      as_device_ptr<const uint8_t>(spec->b_fp4),
      as_device_ptr<const uint8_t>(spec->b_scale),
      as_device_ptr<const float>(spec->b_scale_2), spec->alpha,
      as_device_ptr<__nv_bfloat16>(spec->c_bf16), M, K);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return QWEN36_STATUS_CUDA_ERROR;
  }
  return QWEN36_STATUS_SUCCESS;
}
