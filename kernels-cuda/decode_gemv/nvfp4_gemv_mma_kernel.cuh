// Body of the Direction B NVFP4 GEMV kernel, extracted as a `__device__`
// template function so other fused callers (e.g. the decode interpreter's
// GEMV opcode in kernels-cuda/interpreter/opcodes/nvfp4_gemv.cuh) can call
// the same MMA loop without duplicating ~330 lines of inline PTX + cp.async
// pipeline + cross-warp reduction logic.
//
// The body is mechanically lifted from the original __global__ template
// in nvfp4_gemv_sm120.cu — same SMEM layout, same lane decomposition,
// same split-K pipeline. Behavior is therefore byte-equivalent to the
// pre-refactor path (validated via the chat parity gate after extract).
//
// Contract for callers:
//   - Launch (or call) with blockDim.x = kWarpsPerBlockTpl * 32 threads.
//   - Allocate dynamic SMEM of at least:
//         K/2                              (B operand staging)
//       + kWarpsPerBlockTpl * 2 * 512      (per-warp double-buffered A)
//       + 2 * kRowsPerBlock * kWarps * 4   (cross-warp reduction)
//       + kRowsPerBlock * (K/16) + (K/16)  (SFA + SFB staging, when on)
//     and start it at the buffer base (the body reinterprets `extern
//     __shared__ uint8_t smem[]`).
//   - Pass `m_tile_idx ∈ [0, ceil(M/16))` — the row-tile this invocation
//     owns. The non-persistent grid-shaped use ships `blockIdx.x`; a
//     persistent caller can call the body in a loop with an atomic
//     work counter.
//   - Caller is responsible for the M, K alignment gates the dispatcher
//     enforces (M % 16 == 0; K % (kWarpsPerBlockTpl * 64) == 0).

#pragma once

#include "nvfp4_gemv_mma_helpers.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace qwen36_gemv {

template <int kWarpsPerBlockTpl>
__host__ __device__ inline size_t nvfp4_gemv_mma_smem_bytes(size_t K) {
  constexpr int kWarpsPerBlock = kWarpsPerBlockTpl;
  const size_t a_tile_bytes = static_cast<size_t>(kWarpsPerBlock) * 2u *
                              static_cast<size_t>(kATilePerWarpBytes);
  const size_t reduction_bytes =
      2u * static_cast<size_t>(kRowsPerBlock) *
      static_cast<size_t>(kWarpsPerBlock) * sizeof(float);
#if QWEN36_DECODE_GEMV_SF_SMEM
  const size_t sf_scale_cols = K / 16;
  const size_t sf_staging_bytes =
      static_cast<size_t>(kRowsPerBlock) * sf_scale_cols + sf_scale_cols;
#else
  const size_t sf_staging_bytes = 0;
#endif
  return K / 2 + a_tile_bytes + reduction_bytes + sf_staging_bytes;
}

template <int kWarpsPerBlockTpl>
__device__ inline void nvfp4_gemv_mma_body_with_smem(
    unsigned m_tile_idx, const uint8_t *__restrict__ a_fp4,
    const uint8_t *__restrict__ a_scale, const uint8_t *__restrict__ b_fp4,
    const uint8_t *__restrict__ b_scale, float alpha,
    __nv_bfloat16 *__restrict__ output, size_t M, size_t K, uint8_t *smem) {
  constexpr int kWarpsPerBlock = kWarpsPerBlockTpl;
  constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
  constexpr int kKShardChunkAlign = kWarpsPerBlock * kKPerMma;
  (void)kKShardChunkAlign;

  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane = threadIdx.x & 31;
  const size_t m_base = static_cast<size_t>(m_tile_idx) * kRowsPerBlock;

  const size_t packed_cols = K / 2;
  const size_t scale_cols = K / 16;
  const size_t sf_inner_dim = gemv_round_up(scale_cols, 4);

  constexpr unsigned kATileBufBytes =
      static_cast<unsigned>(kWarpsPerBlock) * 2u * kATilePerWarpBytes;
  constexpr unsigned kReductionBytes =
      2u * static_cast<unsigned>(kRowsPerBlock) *
      static_cast<unsigned>(kWarpsPerBlock) * sizeof(float);
  uint8_t *smem_b_fp4 = smem;
  uint8_t *smem_a_base = smem + K / 2;
  uint8_t *const smem_a_buf_w0 =
      smem_a_base + warp_id * (2u * kATilePerWarpBytes);
  uint8_t *smem_a_buf[2] = {smem_a_buf_w0,
                            smem_a_buf_w0 + kATilePerWarpBytes};
  float *smem_reduction =
      reinterpret_cast<float *>(smem_a_base + kATileBufBytes);
#if QWEN36_DECODE_GEMV_SF_SMEM
  uint8_t *const smem_sfa_base =
      reinterpret_cast<uint8_t *>(smem_reduction) + kReductionBytes;
  uint8_t *const smem_sfb_base =
      smem_sfa_base + static_cast<size_t>(kRowsPerBlock) * scale_cols;
#endif
  {
    const size_t b_bytes = packed_cols;
    const size_t b_vecs = b_bytes / 16;
    const uint4 *b_fp4_vec = reinterpret_cast<const uint4 *>(b_fp4);
    uint4 *smem_b_vec = reinterpret_cast<uint4 *>(smem_b_fp4);
    for (size_t i = threadIdx.x; i < b_vecs;
         i += static_cast<size_t>(kThreadsPerBlock)) {
      smem_b_vec[i] = b_fp4_vec[i];
    }
  }
#if QWEN36_DECODE_GEMV_SF_SMEM
  {
    const size_t per_row_chunks = scale_cols / 4;
    const size_t total_sfa_u32 =
        static_cast<size_t>(kRowsPerBlock) * per_row_chunks;
    uint32_t *const smem_sfa_u32 = reinterpret_cast<uint32_t *>(smem_sfa_base);
    for (size_t t = threadIdx.x; t < total_sfa_u32;
         t += static_cast<size_t>(kThreadsPerBlock)) {
      const size_t row_in_tile = t & (static_cast<size_t>(kRowsPerBlock) - 1);
      const size_t k_chunk_idx = t >> 4;
      const size_t k_group = k_chunk_idx * 4;
      const size_t a_row_raw = m_base + row_in_tile;
      const size_t a_row_safe = a_row_raw < M ? a_row_raw : (M - 1);
      const size_t gmem_off =
          gemv_vec16_scale_offset(k_group, a_row_safe, sf_inner_dim);
      smem_sfa_u32[t] =
          *reinterpret_cast<const uint32_t *>(a_scale + gmem_off);
    }
    const size_t total_sfb_u32 = per_row_chunks;
    uint32_t *const smem_sfb_u32 = reinterpret_cast<uint32_t *>(smem_sfb_base);
    for (size_t t = threadIdx.x; t < total_sfb_u32;
         t += static_cast<size_t>(kThreadsPerBlock)) {
      const size_t k_group = t * 4;
      const size_t gmem_off = gemv_vec16_scale_offset(k_group, 0, sf_inner_dim);
      smem_sfb_u32[t] =
          *reinterpret_cast<const uint32_t *>(b_scale + gmem_off);
    }
  }
#endif
  __syncthreads();

  const unsigned t0 = lane & 3u;
  const unsigned t1 = lane >> 2;

  const unsigned t0_sf_a = lane & 1u;
  const unsigned t2_sf_a = lane >> 2;
  const unsigned m_row_sf = 8u * t0_sf_a + t2_sf_a;
#if !QWEN36_DECODE_GEMV_SF_SMEM
  const size_t a_row_for_sf_raw = m_base + m_row_sf;
  const size_t a_row_for_sf =
      a_row_for_sf_raw < M ? a_row_for_sf_raw : (M - 1);
#endif

  const unsigned a_row0_byte_off = t1 * 32u;
  const unsigned a_row1_byte_off = (t1 + 8u) * 32u;
  const unsigned a_off_v0 = 4u * t0;
  const unsigned a_off_v1 = 4u * t0 + 16u;

  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

  const size_t k_chunks_total = K / kKPerMma;
  const size_t k_chunks_per_warp =
      k_chunks_total / static_cast<size_t>(kWarpsPerBlock);
  const size_t kc_warp_start =
      static_cast<size_t>(warp_id) * k_chunks_per_warp;

  const unsigned a_load_row_in_tile = lane >> 1;
  const unsigned a_load_byte_off = (lane & 1u) << 4;
  const size_t a_load_global_row =
      static_cast<size_t>(m_tile_idx) * kRowsPerBlock + a_load_row_in_tile;
  const bool a_load_row_valid = a_load_global_row < M;
  const uint8_t *a_load_row_ptr =
      a_load_row_valid ? (a_fp4 + a_load_global_row * packed_cols) : nullptr;

  auto issue_chunk_async = [&](size_t kc_global, uint8_t *dst_buf) {
    const size_t k_byte_base = kc_global * (kKPerMma / 2);
    const void *src = a_load_row_valid
                          ? static_cast<const void *>(
                                a_load_row_ptr + k_byte_base + a_load_byte_off)
                          : static_cast<const void *>(a_fp4);
    const unsigned smem_addr =
        __cvta_generic_to_shared(dst_buf + lane * 16u);
    cp_async_16_pred(smem_addr, src, a_load_row_valid);
  };

  if (k_chunks_per_warp > 0) {
    issue_chunk_async(kc_warp_start, smem_a_buf[0]);
    cp_async_commit();
  }

  for (size_t kc_local = 0; kc_local < k_chunks_per_warp; ++kc_local) {
    const size_t kc_global = kc_warp_start + kc_local;
    const size_t k_byte_base = kc_global * (kKPerMma / 2);
#if !QWEN36_DECODE_GEMV_SF_SMEM
    const size_t k_group_base = kc_global * 4;
#endif
    const unsigned cur_buf = static_cast<unsigned>(kc_local) & 1u;

    const bool has_next = (kc_local + 1) < k_chunks_per_warp;
    if (has_next) {
      issue_chunk_async(kc_global + 1, smem_a_buf[cur_buf ^ 1u]);
      cp_async_commit();
    }

    if (has_next) {
      cp_async_wait_group<1>();
    } else {
      cp_async_wait_group<0>();
    }
    __syncwarp();

    uint8_t *smem_a_tile_cur = smem_a_buf[cur_buf];

    uint32_t a0 = *reinterpret_cast<const uint32_t *>(
        smem_a_tile_cur + a_row0_byte_off + a_off_v0);
    uint32_t a1 = *reinterpret_cast<const uint32_t *>(
        smem_a_tile_cur + a_row1_byte_off + a_off_v0);
    uint32_t a2 = *reinterpret_cast<const uint32_t *>(
        smem_a_tile_cur + a_row0_byte_off + a_off_v1);
    uint32_t a3 = *reinterpret_cast<const uint32_t *>(
        smem_a_tile_cur + a_row1_byte_off + a_off_v1);

    const size_t b_byte_off_v0 = k_byte_base + 4u * t0;
    const size_t b_byte_off_v1 = k_byte_base + 4u * t0 + 16u;
    uint32_t b0 =
        *reinterpret_cast<const uint32_t *>(smem_b_fp4 + b_byte_off_v0);
    uint32_t b1 =
        *reinterpret_cast<const uint32_t *>(smem_b_fp4 + b_byte_off_v1);

#if QWEN36_DECODE_GEMV_SF_SMEM
    const size_t k_chunk_idx = kc_global;
    const uint32_t sfa = reinterpret_cast<const uint32_t *>(
        smem_sfa_base)[k_chunk_idx * static_cast<size_t>(kRowsPerBlock) +
                       m_row_sf];
    const uint32_t sfb =
        reinterpret_cast<const uint32_t *>(smem_sfb_base)[k_chunk_idx];
#else
    const size_t sfa_off =
        gemv_vec16_scale_offset(k_group_base, a_row_for_sf, sf_inner_dim);
    const uint32_t sfa =
        *reinterpret_cast<const uint32_t *>(a_scale + sfa_off);
    const size_t sfb_off =
        gemv_vec16_scale_offset(k_group_base, 0, sf_inner_dim);
    const uint32_t sfb =
        *reinterpret_cast<const uint32_t *>(b_scale + sfb_off);
#endif

    mma_mxf4nvf4_4x_m16n8k64(acc0, acc1, acc2, acc3, a0, a1, a2, a3, b0, b1,
                             acc0, acc1, acc2, acc3, sfa, sfb);
  }

  constexpr unsigned kRedRowStride = static_cast<unsigned>(kWarpsPerBlock);
  constexpr unsigned kRedHalfStride = 16u * kRedRowStride;
  if (t0 == 0u) {
    smem_reduction[0u * kRedHalfStride + t1 * kRedRowStride + warp_id] = acc0;
    smem_reduction[1u * kRedHalfStride + t1 * kRedRowStride + warp_id] = acc2;
  }
  __syncthreads();

  if (warp_id == 0u && t0 == 0u) {
    float sum_lo = 0.f;
    float sum_hi = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      sum_lo += smem_reduction[0u * kRedHalfStride + t1 * kRedRowStride +
                               static_cast<unsigned>(w)];
      sum_hi += smem_reduction[1u * kRedHalfStride + t1 * kRedRowStride +
                               static_cast<unsigned>(w)];
    }
    const size_t row_lo = m_base + t1;
    const size_t row_hi = m_base + t1 + 8u;
    if (row_lo < M) {
      output[row_lo] = __float2bfloat16(sum_lo * alpha);
    }
    if (row_hi < M) {
      output[row_hi] = __float2bfloat16(sum_hi * alpha);
    }
  }
}

// ---------------------------------------------------------------------------
// Multi-N chunk variant (2 <= n <= 8): same MMA loop, but the atom's N
// dimension carries REAL activation columns instead of replicating column 0.
// Used by the MTP verify chunk (n = drafts+1, typically 5) where cuBLASLt at
// skinny N runs at ~34% of peak while this path re-reads no extra weight
// bytes versus the N=1 decode GEMV.
//
// Differences from the N=1 body (which stays byte-identical for the decode
// graph):
//   - B staging: n columns in SMEM, column stride padded by 16 bytes so the
//     8 t1-lanes hit distinct banks (K/2 is a multiple of 128 for all
//     supported shapes).
//   - Per-lane B/SFB column = min(t1, n-1): real columns for t1 < n,
//     harmless replication above (their outputs are discarded).
//   - Scale-factor loads go straight to gmem (no SF SMEM staging): the
//     chunk path is not in the captured decode graph and the scales are
//     L2-hot.
//   - Cross-warp reduction holds 16 rows x 8 cols x warps per half; the
//     first 256 threads then own one (half, row, col) cell each and write
//     output[col * M + row] for col < n (column-major [M, n] = the cuBLASLt
//     contract the verify consumers already expect).
constexpr unsigned kChunkColPadBytes = 16;

template <int kWarpsPerBlockTpl>
__host__ __device__ inline size_t nvfp4_gemm_chunk_smem_bytes(size_t K,
                                                              size_t n) {
  const size_t col_stride = K / 2 + kChunkColPadBytes;
  const size_t a_tile_bytes = static_cast<size_t>(kWarpsPerBlockTpl) * 2u *
                              static_cast<size_t>(kATilePerWarpBytes);
  const size_t reduction_bytes = 2u * 8u * 8u *
                                 static_cast<size_t>(kWarpsPerBlockTpl) *
                                 sizeof(float);
  return n * col_stride + a_tile_bytes + reduction_bytes;
}

template <int kWarpsPerBlockTpl>
__device__ inline void nvfp4_gemm_chunk_body(
    unsigned m_tile_idx, const uint8_t *__restrict__ a_fp4,
    const uint8_t *__restrict__ a_scale, const uint8_t *__restrict__ b_fp4,
    const uint8_t *__restrict__ b_scale, float alpha,
    __nv_bfloat16 *__restrict__ output, size_t M, size_t K, unsigned n) {
  extern __shared__ uint8_t smem[];
  constexpr int kWarpsPerBlock = kWarpsPerBlockTpl;
  constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;

  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane = threadIdx.x & 31;
  const size_t m_base = static_cast<size_t>(m_tile_idx) * kRowsPerBlock;

  const size_t packed_cols = K / 2;
  const size_t scale_cols = K / 16;
  const size_t sf_inner_dim = gemv_round_up(scale_cols, 4);
  const size_t col_stride = packed_cols + kChunkColPadBytes;

  constexpr unsigned kATileBufBytes =
      static_cast<unsigned>(kWarpsPerBlock) * 2u * kATilePerWarpBytes;
  uint8_t *smem_b_fp4 = smem;
  uint8_t *smem_a_base = smem + n * col_stride;
  uint8_t *const smem_a_buf_w0 =
      smem_a_base + warp_id * (2u * kATilePerWarpBytes);
  uint8_t *smem_a_buf[2] = {smem_a_buf_w0,
                            smem_a_buf_w0 + kATilePerWarpBytes};
  float *smem_reduction =
      reinterpret_cast<float *>(smem_a_base + kATileBufBytes);

  // Stage the n activation columns (16-byte vectors; both strides are
  // 16-aligned).
  {
    const size_t b_vecs_per_col = packed_cols / 16;
    for (unsigned c = 0; c < n; ++c) {
      const uint4 *src =
          reinterpret_cast<const uint4 *>(b_fp4 + c * packed_cols);
      uint4 *dst = reinterpret_cast<uint4 *>(smem_b_fp4 + c * col_stride);
      for (size_t i = threadIdx.x; i < b_vecs_per_col;
           i += static_cast<size_t>(kThreadsPerBlock)) {
        dst[i] = src[i];
      }
    }
  }
  __syncthreads();

  const unsigned t0 = lane & 3u;
  const unsigned t1 = lane >> 2;
  const unsigned col_b = t1 < n ? t1 : (n - 1);
  const uint8_t *smem_b_col = smem_b_fp4 + col_b * col_stride;

  const unsigned t0_sf_a = lane & 1u;
  const unsigned t2_sf_a = lane >> 2;
  const unsigned m_row_sf = 8u * t0_sf_a + t2_sf_a;
  const size_t a_row_for_sf_raw = m_base + m_row_sf;
  const size_t a_row_for_sf =
      a_row_for_sf_raw < M ? a_row_for_sf_raw : (M - 1);

  const unsigned a_row0_byte_off = t1 * 32u;
  const unsigned a_row1_byte_off = (t1 + 8u) * 32u;
  const unsigned a_off_v0 = 4u * t0;
  const unsigned a_off_v1 = 4u * t0 + 16u;

  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

  const size_t k_chunks_total = K / kKPerMma;
  const size_t k_chunks_per_warp =
      k_chunks_total / static_cast<size_t>(kWarpsPerBlock);
  const size_t kc_warp_start =
      static_cast<size_t>(warp_id) * k_chunks_per_warp;

  const unsigned a_load_row_in_tile = lane >> 1;
  const unsigned a_load_byte_off = (lane & 1u) << 4;
  const size_t a_load_global_row =
      static_cast<size_t>(m_tile_idx) * kRowsPerBlock + a_load_row_in_tile;
  const bool a_load_row_valid = a_load_global_row < M;
  const uint8_t *a_load_row_ptr =
      a_load_row_valid ? (a_fp4 + a_load_global_row * packed_cols) : nullptr;

  auto issue_chunk_async = [&](size_t kc_global, uint8_t *dst_buf) {
    const size_t k_byte_base = kc_global * (kKPerMma / 2);
    const void *src = a_load_row_valid
                          ? static_cast<const void *>(
                                a_load_row_ptr + k_byte_base + a_load_byte_off)
                          : static_cast<const void *>(a_fp4);
    const unsigned smem_addr =
        __cvta_generic_to_shared(dst_buf + lane * 16u);
    cp_async_16_pred(smem_addr, src, a_load_row_valid);
  };

  if (k_chunks_per_warp > 0) {
    issue_chunk_async(kc_warp_start, smem_a_buf[0]);
    cp_async_commit();
  }

  for (size_t kc_local = 0; kc_local < k_chunks_per_warp; ++kc_local) {
    const size_t kc_global = kc_warp_start + kc_local;
    const size_t k_byte_base = kc_global * (kKPerMma / 2);
    const size_t k_group_base = kc_global * 4;
    const unsigned cur_buf = static_cast<unsigned>(kc_local) & 1u;

    const bool has_next = (kc_local + 1) < k_chunks_per_warp;
    if (has_next) {
      issue_chunk_async(kc_global + 1, smem_a_buf[cur_buf ^ 1u]);
      cp_async_commit();
    }

    if (has_next) {
      cp_async_wait_group<1>();
    } else {
      cp_async_wait_group<0>();
    }
    __syncwarp();

    uint8_t *smem_a_tile_cur = smem_a_buf[cur_buf];

    uint32_t a0 = *reinterpret_cast<const uint32_t *>(
        smem_a_tile_cur + a_row0_byte_off + a_off_v0);
    uint32_t a1 = *reinterpret_cast<const uint32_t *>(
        smem_a_tile_cur + a_row1_byte_off + a_off_v0);
    uint32_t a2 = *reinterpret_cast<const uint32_t *>(
        smem_a_tile_cur + a_row0_byte_off + a_off_v1);
    uint32_t a3 = *reinterpret_cast<const uint32_t *>(
        smem_a_tile_cur + a_row1_byte_off + a_off_v1);

    const size_t b_byte_off_v0 = k_byte_base + 4u * t0;
    const size_t b_byte_off_v1 = k_byte_base + 4u * t0 + 16u;
    uint32_t b0 =
        *reinterpret_cast<const uint32_t *>(smem_b_col + b_byte_off_v0);
    uint32_t b1 =
        *reinterpret_cast<const uint32_t *>(smem_b_col + b_byte_off_v1);

    const size_t sfa_off =
        gemv_vec16_scale_offset(k_group_base, a_row_for_sf, sf_inner_dim);
    const uint32_t sfa =
        *reinterpret_cast<const uint32_t *>(a_scale + sfa_off);
    const size_t sfb_off =
        gemv_vec16_scale_offset(k_group_base, col_b, sf_inner_dim);
    const uint32_t sfb =
        *reinterpret_cast<const uint32_t *>(b_scale + sfb_off);

    mma_mxf4nvf4_4x_m16n8k64(acc0, acc1, acc2, acc3, a0, a1, a2, a3, b0, b1,
                             acc0, acc1, acc2, acc3, sfa, sfb);
  }

  // Reduction: every lane owns (row t1 / t1+8, cols 2*t0 and 2*t0+1).
  // Layout: red[half][row16][col8][warp].
  // Layout: red[half][row8 = t1][col8][warp] — t1 spans 8 rows, the half
  // index carries the +8 (acc2/acc3 are rows t1+8 in the atom).
  constexpr unsigned kRedW = static_cast<unsigned>(kWarpsPerBlock);
  constexpr unsigned kRedColStride = kRedW;
  constexpr unsigned kRedRowStride = 8u * kRedColStride;
  constexpr unsigned kRedHalfStride = 8u * kRedRowStride;
  smem_reduction[0u * kRedHalfStride + t1 * kRedRowStride +
                 (2u * t0 + 0u) * kRedColStride + warp_id] = acc0;
  smem_reduction[0u * kRedHalfStride + t1 * kRedRowStride +
                 (2u * t0 + 1u) * kRedColStride + warp_id] = acc1;
  smem_reduction[1u * kRedHalfStride + t1 * kRedRowStride +
                 (2u * t0 + 0u) * kRedColStride + warp_id] = acc2;
  smem_reduction[1u * kRedHalfStride + t1 * kRedRowStride +
                 (2u * t0 + 1u) * kRedColStride + warp_id] = acc3;
  __syncthreads();

  // 128 cells = 2 halves x 8 rows x 8 cols; thread tid < 128 owns one.
  if (threadIdx.x < 128u) {
    const unsigned half = threadIdx.x >> 6;
    const unsigned row8 = (threadIdx.x >> 3) & 7u;
    const unsigned col = threadIdx.x & 7u;
    if (col < n) {
      float sum = 0.f;
#pragma unroll
      for (int w = 0; w < kWarpsPerBlock; ++w) {
        sum += smem_reduction[half * kRedHalfStride + row8 * kRedRowStride +
                              col * kRedColStride + static_cast<unsigned>(w)];
      }
      const size_t row = m_base + row8 + 8u * half;
      if (row < M) {
        output[static_cast<size_t>(col) * M + row] =
            __float2bfloat16(sum * alpha);
      }
    }
  }
}

template <int kWarpsPerBlockTpl>
__device__ inline void
nvfp4_gemv_mma_body(unsigned m_tile_idx,
                    const uint8_t *__restrict__ a_fp4,
                    const uint8_t *__restrict__ a_scale,
                    const uint8_t *__restrict__ b_fp4,
                    const uint8_t *__restrict__ b_scale, float alpha,
                    __nv_bfloat16 *__restrict__ output, size_t M, size_t K) {
  extern __shared__ uint8_t smem[];
  nvfp4_gemv_mma_body_with_smem<kWarpsPerBlockTpl>(
      m_tile_idx, a_fp4, a_scale, b_fp4, b_scale, alpha, output, M, K, smem);
}

} // namespace qwen36_gemv
