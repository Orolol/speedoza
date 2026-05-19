// Direction B NVFP4 gemv kernel for Blackwell SM_120a — Phase B3.
//
// Hand-rolled tensor-core gemv that drives the SM120 NVFP4 block-scaled MMA
// atom (mma.sync.aligned.kind::mxf4nvf4.scale_vec::4X.m16n8k64.f32.e2m1.e2m1
// .f32.ue4m3) directly via inline PTX from cute. CUTLASS's high-level
// CollectiveBuilder rejects narrow-N tiles at N=1 (see Phase B2 notes), so
// we drop down to the atom and stage register tiles by hand.
//
// B3.7: kernel is now templated on kWarpsPerBlockTpl with two compiled
// specializations:
//   - 16 warps / CTA, 512 threads, K%1024==0 — preferred path. Higher
//     SM-resident warp count, best latency hiding for "fat" decode shapes
//     (e.g. K=5120 for q_proj, K=17408 for o_proj).
//   - 8 warps / CTA, 256 threads, K%512==0 — fallback for shapes whose K
//     is a multiple of 512 but not 1024 (e.g. K=3584 for linear-attention
//     out_proj where value_dim=3584). Same architecture as B3.5; bench
//     showed +5-15% over cuBLASLt at MTP=0/4.
// The entry point picks the largest specialization the K dimension will
// admit; shapes that don't satisfy K%512==0 fall through to cuBLASLt.
//
// CTA layout (B3.7: intra-CTA split-K, kWarpsPerBlockTpl warps)
//   - Each CTA owns ONE m16 MMA tile (16 rows); all warps cooperate on
//     the SAME 16 output rows but DIFFERENT K shards. This multiplies
//     parallelism for low-M shapes by Nwarps× compared to the original
//     "1 warp = 1 m16 tile" layout.
//   - blockDim.x = kWarpsPerBlockTpl * 32, gridDim.x = ceil(M / 16).
//   - For k_chunk in [warp_id*K_per_warp/64, (warp_id+1)*K_per_warp/64),
//     every warp issues one MMA accumulating into its private 4-register
//     float accumulator, then a final cross-warp reduction in smem sums
//     the partial dot products and writes the n=0 column to gmem.
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
// Soft regime: n==1 && m%16==0 && k%512==0 (with k%1024==0 routing to the
// 16-warp specialization). The k-alignment constraint comes from split-K:
// K is sharded into Nwarps equal pieces (one per warp), each of which must
// be a multiple of the kKPerMma=64 inner-loop chunk. For shapes the MMA
// cannot service we return QWEN36_STATUS_NOT_IMPLEMENTED so the dispatcher
// routes to cuBLASLt. Active env var: QWEN36_DECODE_GEMV=1.

#include "qwen36_fp4.h"
#include "active_stream.h"
#include "nvfp4_gemv_mma_helpers.cuh"
#include "nvfp4_gemv_mma_kernel.cuh"

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
// atom in nvfp4_gemv_mma_helpers.cuh instead.
//
// The cp.async / MMA / SF helpers + constants (kRowsPerBlock, kKPerMma,
// kATilePerWarpBytes, etc.) and the QWEN36_DECODE_GEMV_* macros all live
// in that header so the per-block megakernel
// (kernels-cuda/megakernel/full_attn_block_sm120.cu) can reuse them
// without duplicating the inline PTX. Bring them into the unqualified
// namespace below for the existing kernel body's call sites.

using namespace qwen36_gemv;

// Templated kernel. Two specializations are emitted (kWarpsPerBlockTpl =
// 16 and 8). The kernel is declared at namespace scope so explicit
// instantiation can force the symbols into the .so.
// Tensor scales (a_scale_2, b_scale_2) intentionally NOT in the kernel
// signature: the runtime caller (engine.rs) already folds them into
// `spec->alpha` before invoking the kernel, mirroring the cuBLASLt
// contract (nvfp4_gemm.cu only passes `alpha`, never dereferences
// a_scale_2/b_scale_2). Multiplying by them inside the kernel would
// square the tensor-scale factor and produce silent gibberish on real
// model weights.
//
// The body itself lives in nvfp4_gemv_mma_kernel.cuh so the per-block
// megakernel can call the same __device__ template without duplicating
// the inline PTX. The __global__ wrapper here is a thin shim that picks
// up __launch_bounds__ for the standalone-dispatcher launch path.
template <int kWarpsPerBlockTpl>
__global__ void __launch_bounds__(kWarpsPerBlockTpl * 32)
nvfp4_gemv_mma_kernel_tpl(const uint8_t *__restrict__ a_fp4,
                          const uint8_t *__restrict__ a_scale,
                          const uint8_t *__restrict__ b_fp4,
                          const uint8_t *__restrict__ b_scale, float alpha,
                          __nv_bfloat16 *__restrict__ output, size_t M,
                          size_t K) {
  qwen36_gemv::nvfp4_gemv_mma_body<kWarpsPerBlockTpl>(
      a_fp4, a_scale, b_fp4, b_scale, alpha, output, M, K);
}

// Explicit instantiations — force both specialization symbols into the .so.
template __global__ void
nvfp4_gemv_mma_kernel_tpl<16>(const uint8_t *, const uint8_t *, const uint8_t *,
                              const uint8_t *, float, __nv_bfloat16 *, size_t,
                              size_t);
template __global__ void
nvfp4_gemv_mma_kernel_tpl<8>(const uint8_t *, const uint8_t *, const uint8_t *,
                             const uint8_t *, float, __nv_bfloat16 *, size_t,
                             size_t);

namespace {

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
  // MMA regime: N=1, M aligned to the m16 MMA tile, K aligned to the
  // split-K chunk. We pick the largest-warp specialization the K dimension
  // admits:
  //   - kAlign16 = 16 * 64 = 1024  → 16 warps / CTA (preferred).
  //   - kAlign8  =  8 * 64 =  512  → 8 warps / CTA  (fallback).
  // K not divisible by 512 returns NOT_IMPLEMENTED so cuBLASLt picks it up.
  constexpr size_t kAlign16 = 16u * static_cast<size_t>(kKPerMma);  // 1024
  constexpr size_t kAlign8 = 8u * static_cast<size_t>(kKPerMma);    // 512

  if (spec->n != 1 || (spec->m % 16) != 0) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  int chosen_warps = 0;
  if ((spec->k % kAlign16) == 0) {
    chosen_warps = 16;
  } else if ((spec->k % kAlign8) == 0) {
    chosen_warps = 8;
  } else {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  const size_t M = spec->m;
  const size_t K = spec->k;
  const dim3 grid(static_cast<unsigned>(gemv_div_ceil(M, kRowsPerBlock)), 1, 1);
  cudaStream_t stream = qwen36_internal_active_stream();

  // Smem footprint (parameterized on chosen_warps):
  //   - K/2 bytes activation column (B operand, CTA-shared).
  //   - chosen_warps * 2 * 512 bytes per-warp double-buffered A tile.
  //   - 2 * 16 * chosen_warps * 4 bytes cross-warp reduction scratch.
  //   - (SF SMEM staging only) 16 * (K/16) bytes SFA + (K/16) bytes SFB.
  // E.g. K=5120, chosen_warps=16, SF staging on:
  //        2560 + 16384 + 2048 + 5120 + 320 = 26432 B (~26 KiB).
  //      K=34816, chosen_warps=16, SF staging on:
  //       17408 + 16384 + 2048 + 34816 + 2176 = 72832 B (~71 KiB).
  // Both under the 100 KiB sm_120 dynamic per-block max.
  const size_t a_tile_bytes = static_cast<size_t>(chosen_warps) * 2u *
                              static_cast<size_t>(kATilePerWarpBytes);
  const size_t reduction_bytes =
      2u * static_cast<size_t>(kRowsPerBlock) *
      static_cast<size_t>(chosen_warps) * sizeof(float);
#if QWEN36_DECODE_GEMV_SF_SMEM
  const size_t sf_scale_cols = K / 16;
  const size_t sf_staging_bytes =
      static_cast<size_t>(kRowsPerBlock) * sf_scale_cols + sf_scale_cols;
#else
  const size_t sf_staging_bytes = 0;
#endif
  const size_t smem_bytes =
      K / 2 + a_tile_bytes + reduction_bytes + sf_staging_bytes;

  if (chosen_warps == 16) {
    const dim3 block(16u * 32u, 1, 1);
    nvfp4_gemv_mma_kernel_tpl<16><<<grid, block, smem_bytes, stream>>>(
        as_device_ptr<const uint8_t>(spec->a_fp4),
        as_device_ptr<const uint8_t>(spec->a_scale),
        as_device_ptr<const uint8_t>(spec->b_fp4),
        as_device_ptr<const uint8_t>(spec->b_scale), spec->alpha,
        as_device_ptr<__nv_bfloat16>(spec->c_bf16), M, K);
  } else {
    const dim3 block(8u * 32u, 1, 1);
    nvfp4_gemv_mma_kernel_tpl<8><<<grid, block, smem_bytes, stream>>>(
        as_device_ptr<const uint8_t>(spec->a_fp4),
        as_device_ptr<const uint8_t>(spec->a_scale),
        as_device_ptr<const uint8_t>(spec->b_fp4),
        as_device_ptr<const uint8_t>(spec->b_scale), spec->alpha,
        as_device_ptr<__nv_bfloat16>(spec->c_bf16), M, K);
  }

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
