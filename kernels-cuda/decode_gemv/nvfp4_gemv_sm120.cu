// Direction B NVFP4 gemv kernel for Blackwell SM_120a — Phase B3.
//
// Hand-rolled tensor-core gemv that drives the SM120 NVFP4 block-scaled MMA
// atom (mma.sync.aligned.kind::mxf4nvf4.scale_vec::4X.m16n8k64.f32.e2m1.e2m1
// .f32.ue4m3) directly via inline PTX from cute. CUTLASS's high-level
// CollectiveBuilder rejects narrow-N tiles at N=1 (see Phase B2 notes), so
// we drop down to the atom and stage register tiles by hand.
//
// CTA layout
//   - 4 warps / CTA, 128 threads. Each warp owns one m16 MMA tile (16 rows).
//   - blockDim.x = 128, gridDim.x = ceil(M / 64). Each block emits 64 rows.
//   - For k_chunk in [0, K) step 64, every warp issues one MMA accumulating
//     into a per-warp 4-register float accumulator, then writes the n=0
//     column to gmem in the epilogue.
//
// Operand staging (lane L ∈ [0,32))
//   t0 = L & 3, t1 = L >> 2.
//   A[r] for r = v1 + 2*v2, v1,v2 ∈ {0,1}: 8 packed fp4 from row (t1 + 8*v2)
//   at fp4 offset (8*t0 + 32*v1) — load as one uint32 from the row's
//   packed-fp4 buffer.
//   B[r] for r = v1 ∈ {0,1}: 8 packed fp4 from the single activation column
//   at fp4 offset (8*t0 + 32*v1).
//   SFA: lane decomposition is (t0_sf=L&1, t2_sf=L>>2) → m_row_sf =
//   8*(L&1) + (L>>2). 4 packed e4m3 bytes (k_group 0..3 in low..high
//   nibbles).
//   SFB: n_col_sf = L>>2 (broadcast). At N=1 outer is always 0, so the
//   scale-byte address only varies with k_group.
//   D[r] for r = v0 + 2*v1: row (t1 + 8*v1), col (2*t0 + v0). For the
//   n=0 output we keep lanes with t0==0 → D[0] writes row t1, D[2] writes
//   row t1+8.
//
// Scale layout: identical vec16_scale_offset swizzle as the cuBLASLt and
// CUTLASS paths so we can read SFA/SFB straight from the same buffers.
//
// Soft regime: n==1 && m%16==0 && k%64==0. For shapes the MMA cannot
// service we return QWEN36_STATUS_NOT_IMPLEMENTED so the dispatcher routes
// to cuBLASLt. Active env var: QWEN36_DECODE_GEMV=1.

#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// We deliberately AVOID including <cutlass/...> or <cute/...> here even
// though that header defines the MMA atom we want. Pulling cute into a
// translation unit that contains a `__global__` kernel triggers a host
// stub-generator bug where it tries to register cuda::std::__cpo entries
// (begin/end/cbegin/cend etc.) as device variables, but `::cuda::std`
// isn't declared in the host compilation unit, so g++ chokes with
// "'::cuda' has not been declared". Mirror the exact PTX from the cute
// atom (cute/arch/mma_sm120.hpp:3215, kind::mxf4nvf4.scale_vec::4X
// .m16n8k64 with e2m1 × e2m1 and ue4m3 scales) inline below. This keeps
// the TU self-contained — no cute, no cuda::std.
//
// The `kind::mxf4nvf4` opcode is sm_120a-only. Building with
// `-arch=sm_120a` emits BOTH a compute_120 (PTX-forward) image AND a
// compute_120a image; the compute_120 image must softly fall through, so
// we gate the asm on __CUDA_ARCH_FEAT_SM120_ALL.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200) &&                       \
    defined(__CUDA_ARCH_FEAT_SM120_ALL)
#define QWEN36_DECODE_GEMV_MMA_DEVICE 1
#else
#define QWEN36_DECODE_GEMV_MMA_DEVICE 0
#endif

// Host-side guard: enabled whenever the CUDA toolchain can target sm_120a
// (driver / nvcc 12.8+). Nothing on the host depends on the device-side
// macro. We always compile the kernel; the device-side body is
// conditionally a no-op for the compute_120 fallback PTX image.
#define QWEN36_DECODE_GEMV_MMA 1

namespace {

__host__ __device__ size_t gemv_div_ceil(size_t a, size_t b) {
  return (a + b - 1) / b;
}

__host__ __device__ size_t gemv_round_up(size_t v, size_t m) {
  return gemv_div_ceil(v, m) * m;
}

// cuBLASLt vec16 scale-swizzle layout. See ops.cu:vec16_scale_offset.
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

// MMA-driven kernel. Each warp owns 16 contiguous rows; each CTA owns 4
// warps == 64 rows. Inner loop walks K in chunks of 64.
constexpr int kWarpsPerBlock = 4;
constexpr int kRowsPerWarp = 16;
constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;  // 64
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;          // 128
constexpr int kKPerMma = 64;

// Inline-PTX wrapper for the SM120 mxf4nvf4 scale_vec::4X m16n8k64 atom.
// Direct mirror of cute/arch/mma_sm120.hpp:3215. Lives at file scope so
// the kernel body stays readable.
__device__ __forceinline__ void mma_mxf4nvf4_4x_m16n8k64(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint32_t sfa0, uint32_t sfb0) {
#if QWEN36_DECODE_GEMV_MMA_DEVICE
  constexpr uint16_t bid = 0;
  constexpr uint16_t tid = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(sfa0), "h"(bid), "h"(tid),
        "r"(sfb0), "h"(bid), "h"(tid));
#else
  d0 = c0; d1 = c1; d2 = c2; d3 = c3;
  (void)a0; (void)a1; (void)a2; (void)a3;
  (void)b0; (void)b1; (void)sfa0; (void)sfb0;
#endif
}

// Kernel is always declared (host needs the symbol for the launch stub).
// On compute_120 the body is a no-op — the host wrapper guards the launch
// behind QWEN36_DECODE_GEMV_MMA so the kernel is never invoked at runtime.
__global__ void __launch_bounds__(kThreadsPerBlock)
nvfp4_gemv_mma_kernel(const uint8_t *__restrict__ a_fp4,
                      const uint8_t *__restrict__ a_scale,
                      const float *__restrict__ a_tensor_scale,
                      const uint8_t *__restrict__ b_fp4,
                      const uint8_t *__restrict__ b_scale,
                      const float *__restrict__ b_tensor_scale,
                      float alpha, __nv_bfloat16 *__restrict__ output,
                      size_t M, size_t K) {
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane = threadIdx.x & 31;
  const size_t m_base =
      static_cast<size_t>(blockIdx.x) * kRowsPerBlock + warp_id * kRowsPerWarp;
  if (m_base >= M) {
    return;
  }

  const size_t packed_cols = K / 2;       // bytes per fp4 row
  const size_t scale_cols = K / 16;       // scale groups per row
  const size_t sf_inner_dim = gemv_round_up(scale_cols, 4);

  // Lane decomposition for the operand layouts (canonical m16n8k* form).
  const unsigned t0 = lane & 3u;
  const unsigned t1 = lane >> 2;

  // SFA decomposition: m_row_sf = 8*(L&1) + (L>>2).
  const unsigned t0_sf_a = lane & 1u;
  const unsigned t2_sf_a = lane >> 2;
  const unsigned m_row_sf = 8u * t0_sf_a + t2_sf_a;
  const size_t a_row_for_sf = m_base + m_row_sf;

  // SFB at N=1: outer is always 0, no n-decomposition needed.

  const float a_ts =
      (a_tensor_scale == nullptr) ? 1.0f : __ldg(a_tensor_scale);
  const float b_ts =
      (b_tensor_scale == nullptr) ? 1.0f : __ldg(b_tensor_scale);

  // Row pointers for the two A sub-tiles owned by this lane.
  const size_t a_row0 = m_base + t1;        // for A[0], A[2]
  const size_t a_row1 = m_base + t1 + 8u;   // for A[1], A[3]
  const uint8_t *a_row0_ptr = a_fp4 + a_row0 * packed_cols;
  const uint8_t *a_row1_ptr = a_fp4 + a_row1 * packed_cols;

  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

  // K chunk index stride is fixed: 4 scale groups per chunk.
  const size_t k_chunks = K / kKPerMma;

  for (size_t kc = 0; kc < k_chunks; ++kc) {
    const size_t k_byte_base = kc * (kKPerMma / 2);  // 32 bytes per chunk
    const size_t k_group_base = kc * 4;              // 4 scale groups / chunk

    // ---- A operand: 4 uint32, each = 4 bytes = 8 fp4 elements. ----
    const size_t a_byte_off_v0 = k_byte_base + 4u * t0;        // v1=0
    const size_t a_byte_off_v1 = k_byte_base + 4u * t0 + 16u;  // v1=1
    uint32_t a0 =
        *reinterpret_cast<const uint32_t *>(a_row0_ptr + a_byte_off_v0);
    uint32_t a1 =
        *reinterpret_cast<const uint32_t *>(a_row1_ptr + a_byte_off_v0);
    uint32_t a2 =
        *reinterpret_cast<const uint32_t *>(a_row0_ptr + a_byte_off_v1);
    uint32_t a3 =
        *reinterpret_cast<const uint32_t *>(a_row1_ptr + a_byte_off_v1);

    // ---- B operand: 2 uint32. Single activation column at N=1. ----
    uint32_t b0 = *reinterpret_cast<const uint32_t *>(b_fp4 + a_byte_off_v0);
    uint32_t b1 = *reinterpret_cast<const uint32_t *>(b_fp4 + a_byte_off_v1);

    // ---- SFA: 4 e4m3 bytes packed into a uint32, k_group 0..3. ----
    uint32_t sfa = 0;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const size_t off = gemv_vec16_scale_offset(
          k_group_base + static_cast<size_t>(g), a_row_for_sf, sf_inner_dim);
      const uint8_t b = a_scale[off];
      sfa |= static_cast<uint32_t>(b) << (g * 8);
    }

    // ---- SFB: same packing; outer = 0 at N=1. ----
    uint32_t sfb = 0;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const size_t off = gemv_vec16_scale_offset(
          k_group_base + static_cast<size_t>(g), 0, sf_inner_dim);
      const uint8_t b = b_scale[off];
      sfb |= static_cast<uint32_t>(b) << (g * 8);
    }

    mma_mxf4nvf4_4x_m16n8k64(acc0, acc1, acc2, acc3,
                             a0, a1, a2, a3,
                             b0, b1,
                             acc0, acc1, acc2, acc3,
                             sfa, sfb);
  }

  // Epilogue: lanes with t0==0 hold the n=0 column. D[0] is row t1,
  // D[2] is row t1+8. Apply tensor scales + alpha and store as bf16.
  if (t0 == 0u) {
    const float scale = alpha * a_ts * b_ts;
    const size_t row_lo = m_base + t1;
    const size_t row_hi = m_base + t1 + 8u;
    if (row_lo < M) {
      output[row_lo] = __float2bfloat16(acc0 * scale);
    }
    if (row_hi < M) {
      output[row_hi] = __float2bfloat16(acc2 * scale);
    }
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

#if QWEN36_DECODE_GEMV_MMA
  // MMA regime: N=1, M aligned to the m16 MMA tile (matches the warp's row
  // stride), K aligned to the k64 inner-loop chunk. K%64 is tighter than
  // the previous K%16 fallback — anything outside this returns
  // NOT_IMPLEMENTED so cuBLASLt picks it up.
  if (spec->n != 1 || (spec->m % 16) != 0 || (spec->k % 64) != 0) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  const size_t M = spec->m;
  const size_t K = spec->k;
  const dim3 block(kThreadsPerBlock, 1, 1);
  const dim3 grid(static_cast<unsigned>(gemv_div_ceil(M, kRowsPerBlock)), 1,
                  1);

  cudaStream_t stream = qwen36_internal_active_stream();
  nvfp4_gemv_mma_kernel<<<grid, block, 0, stream>>>(
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
#else
  (void)spec;
  return QWEN36_STATUS_NOT_IMPLEMENTED;
#endif
}
