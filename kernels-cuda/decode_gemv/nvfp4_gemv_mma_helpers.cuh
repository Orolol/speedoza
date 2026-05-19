// Shared device-side helpers for the Direction B NVFP4 GEMV kernel
// (nvfp4_gemv_sm120.cu) — extracted into a header so the per-block
// megakernel (kernels-cuda/megakernel/full_attn_block_sm120.cu) can call
// the same MMA atom and SF helpers without duplicating the inline PTX.
//
// All definitions live in the `qwen36_gemv` namespace; including TUs can
// `using namespace qwen36_gemv;` at file scope to keep the existing call
// sites unchanged. The macros are kept at file scope because they gate
// inline PTX selection per device arch (sm_120 vs sm_120a) and must be
// evaluated in the including TU's __CUDA_ARCH__ context.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// sm_120a-only MMA atom gate. The compute_120 forward image must softly
// fall through, so the asm bodies are gated on __CUDA_ARCH_FEAT_SM120_ALL.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200) &&                       \
    defined(__CUDA_ARCH_FEAT_SM120_ALL)
#define QWEN36_DECODE_GEMV_MMA_DEVICE 1
#else
#define QWEN36_DECODE_GEMV_MMA_DEVICE 0
#endif

// Host-side guard: enabled whenever the CUDA toolchain can target sm_120a.
#define QWEN36_DECODE_GEMV_MMA 1

// When enabled, stage SFA + SFB into SMEM at kernel entry instead of
// re-loading them from gmem on every K-iter. Matches the layout used by
// the standalone GEMV kernel.
#define QWEN36_DECODE_GEMV_SF_SMEM 1

namespace qwen36_gemv {

__host__ __device__ inline size_t gemv_div_ceil(size_t a, size_t b) {
  return (a + b - 1) / b;
}

__host__ __device__ inline size_t gemv_round_up(size_t v, size_t m) {
  return gemv_div_ceil(v, m) * m;
}

// cuBLASLt vec16 scale-swizzle layout. See ops.cu:vec16_scale_offset.
__host__ __device__ inline size_t
gemv_vec16_scale_offset(size_t inner, size_t outer, size_t sf_inner_dim) {
  const size_t block_inner = (inner / 4) * 4;
  const size_t block_outer = outer / 128;
  const size_t block_offset = (block_inner + block_outer * sf_inner_dim) * 128;
  const size_t tile_outer = outer % 128;
  const size_t tile_inner = inner % 4;
  return block_offset + (tile_outer % 32) * 16 + (tile_outer / 32) * 4 +
         tile_inner;
}

// File-scope constants that are independent of the warp count.
constexpr int kRowsPerWarp = 16;
constexpr int kRowsPerBlock = kRowsPerWarp; // 16 — one m16 MMA tile / CTA
constexpr int kKPerMma = 64;
constexpr unsigned kATilePerWarpBytes =
    static_cast<unsigned>(kRowsPerBlock) * 32u; // 512

// ---- cp.async helpers (sm_80+; available on sm_120). ----
// `cg` (cache global, bypass L1) is the right choice for streaming the A
// weight tile — we don't reuse it after consumption. Predicated form uses
// the trailing src-size operand: when src_bytes==0, the destination smem
// is filled with 16 zeros (PTX cp.async semantics).
__device__ __forceinline__ void cp_async_16_pred(unsigned smem_addr,
                                                 const void *gmem_ptr,
                                                 bool valid) {
#if QWEN36_DECODE_GEMV_MMA_DEVICE
  const unsigned src_bytes = valid ? 16u : 0u;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes));
#else
  (void)smem_addr;
  (void)gmem_ptr;
  (void)valid;
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if QWEN36_DECODE_GEMV_MMA_DEVICE
  asm volatile("cp.async.commit_group;\n");
#endif
}

template <int N> __device__ __forceinline__ void cp_async_wait_group() {
#if QWEN36_DECODE_GEMV_MMA_DEVICE
  asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
#endif
}

// Inline-PTX wrapper for the SM120 mxf4nvf4 scale_vec::4X m16n8k64 atom.
// Direct mirror of cute/arch/mma_sm120.hpp:3215.
__device__ __forceinline__ void
mma_mxf4nvf4_4x_m16n8k64(float &d0, float &d1, float &d2, float &d3,
                         uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                         uint32_t b0, uint32_t b1, float c0, float c1,
                         float c2, float c3, uint32_t sfa0, uint32_t sfb0) {
#if QWEN36_DECODE_GEMV_MMA_DEVICE
  constexpr uint16_t bid = 0;
  constexpr uint16_t tid = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row."
      "col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
        "f"(c2), "f"(c3), "r"(sfa0), "h"(bid), "h"(tid), "r"(sfb0), "h"(bid),
        "h"(tid));
#else
  d0 = c0;
  d1 = c1;
  d2 = c2;
  d3 = c3;
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)b0;
  (void)b1;
  (void)sfa0;
  (void)sfb0;
#endif
}

} // namespace qwen36_gemv
