// Flash-attention prefill kernel for Qwen3.6 (head_dim=256, GQA 24:4).
//
// Implementation: 4 warps cooperating on a M=32, N=64, D=256 tile.  wmma BF16
// (m16n16k16) for both S=Q@K^T and O += P @ V, accumulating to FP32.
// Online-softmax row state (m, l, α) lives in SMEM since rows are owned by
// different warps after the warp split.
//
// Warp split:
//   warp 0: M-tile 0 (rows 0-15), N halves 0,1 (cols 0-31)
//   warp 1: M-tile 0,             N halves 2,3 (cols 32-63)
//   warp 2: M-tile 1 (rows 16-31), N halves 0,1
//   warp 3: M-tile 1,             N halves 2,3
//
// For the second matmul O += P @ V, the same warps own the same M-tile but
// split D-output instead of N (each warp owns 8 of the 16 D-fragments).
// Specialised for the Qwen3.6 hot path:
//   tokens >= 16, no tree-mask, BF16 or FP8 E4M3 KV cache, q_heads=24,
//   kv_heads=4.  The dispatcher in attention.cu falls back to scalar kernels
//   for any other shape, and for TurboQuant cache dtypes.

#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

namespace {

constexpr int kFlashM = 32;       // Q rows per block
constexpr int kFlashN = 64;       // K rows per K-iter
constexpr int kFlashD = 256;      // head_dim
constexpr int kFlashWarps = 4;
constexpr int kFlashThreads = 32 * kFlashWarps;
constexpr int kFlashMTiles = kFlashM / 16; // 2
constexpr int kFlashNTiles = kFlashN / 16; // 4
constexpr int kFlashDTiles = kFlashD / 16; // 16

constexpr int kKvCacheBf16 = 0;
constexpr int kKvCacheFp8 = 1;

__device__ __forceinline__ float flash_decode_e4m3(uint8_t code) {
  // Same E4M3 decode the rest of the codebase uses (attention.cu:89).
  const float sign = (code & 0x80) ? -1.0f : 1.0f;
  const int exponent = (code >> 3) & 0x0f;
  const int mantissa = code & 0x07;
  if (exponent == 0) {
    if (mantissa == 0) {
      return sign * 0.0f;
    }
    return sign * ldexpf(static_cast<float>(mantissa) / 8.0f, -6);
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return sign * 448.0f;
  }
  return sign * ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f,
                       exponent - 7);
}

__device__ __forceinline__ __nv_bfloat16
flash_load_kv_bf16(const void *cache, int kv_cache_dtype, size_t index) {
  if (kv_cache_dtype == kKvCacheFp8) {
    return __float2bfloat16(flash_decode_e4m3(
        reinterpret_cast<const uint8_t *>(cache)[index]));
  }
  return reinterpret_cast<const __nv_bfloat16 *>(cache)[index];
}

template <typename SrcLoad>
__device__ __forceinline__ void
cooperative_load_bf16(__nv_bfloat16 *dst, size_t count, SrcLoad src_load) {
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    dst[i] = src_load(i);
  }
}

__global__ void
attention_flash_prefill_kernel(const __nv_bfloat16 *q, const void *cache_k,
                               const void *cache_v, int kv_cache_dtype,
                               __nv_bfloat16 *output, size_t start_position,
                               const int32_t *start_position_device,
                               size_t tokens,
                               qwen36_attention_shape_t shape) {
  // grid.x = kv_heads, grid.y = ceil(tokens / kFlashM)
  const size_t kvh = blockIdx.x;
  const size_t q_tile_idx = blockIdx.y;
  const size_t token_base = q_tile_idx * kFlashM;
  if (token_base >= tokens) {
    return;
  }
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t head_dim = shape.head_dim;
  const float qk_scale = rsqrtf(static_cast<float>(head_dim));
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int my_m_tile = warp_id / 2;      // 0 or 1
  const int my_n_half = warp_id & 1;      // 0 → cols 0-31, 1 → cols 32-63
  const int my_m_row_base = my_m_tile * 16;
  const int my_n_col_base = my_n_half * 32;
  // PV: D-output split: warp w owns D-tiles [my_n_half*8, my_n_half*8+8).
  const int my_d_tile_base = my_n_half * 8; // 0 or 8

  __shared__ size_t shared_start_position;
  if (threadIdx.x == 0) {
    shared_start_position = (start_position_device != nullptr)
                                ? static_cast<size_t>(*start_position_device)
                                : start_position;
  }
  __syncthreads();
  const size_t start_pos = shared_start_position;
  const size_t kv_total = start_pos + tokens;
  const size_t rows_in_tile =
      (token_base + kFlashM <= tokens) ? kFlashM : (tokens - token_base);
  const size_t max_kv_visible = start_pos + token_base + rows_in_tile - 1;

  extern __shared__ unsigned char smem_raw[];
  __nv_bfloat16 *sm_Q = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  __nv_bfloat16 *sm_K = sm_Q + kFlashM * kFlashD;
  __nv_bfloat16 *sm_V = sm_K + kFlashN * kFlashD;
  float *sm_S = reinterpret_cast<float *>(sm_V + kFlashN * kFlashD);
  __nv_bfloat16 *sm_P =
      reinterpret_cast<__nv_bfloat16 *>(sm_S + kFlashM * kFlashN);
  float *sm_m = reinterpret_cast<float *>(sm_P + kFlashM * kFlashN);
  float *sm_l = sm_m + kFlashM;
  float *sm_alpha = sm_l + kFlashM;

  // Loop over q_per_kv q_heads sharing the same KV head.
  for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
    const size_t qh = kvh * q_per_kv + qh_local;

    // ---- load Q tile [M × D] for this q_head into sm_Q ----
    cooperative_load_bf16(
        sm_Q, kFlashM * kFlashD, [&](size_t i) -> __nv_bfloat16 {
          const size_t row = i / kFlashD;
          const size_t d = i % kFlashD;
          if (row >= rows_in_tile) {
            return __float2bfloat16(0.0f);
          }
          return q[((token_base + row) * shape.q_heads + qh) * head_dim + d];
        });

    // ---- init row state ----
    if (threadIdx.x < kFlashM) {
      sm_m[threadIdx.x] = -INFINITY;
      sm_l[threadIdx.x] = 0.0f;
    }

    // O accumulator lives in registers per warp.  Each warp owns 8 D-tiles
    // (my_d_tile_base..my_d_tile_base+7) for one M-tile (rows
    // my_m_row_base..my_m_row_base+15).  We keep these as 8 wmma accumulator
    // fragments per warp, zeroed at the start of each q_head loop.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frags[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      wmma::fill_fragment(o_frags[i], 0.0f);
    }
    __syncthreads();

    // ---- K-iter loop ----
    const size_t k_iters = (max_kv_visible + kFlashN) / kFlashN;
    for (size_t k_iter = 0; k_iter < k_iters; ++k_iter) {
      const size_t k_base = k_iter * kFlashN;
      if (k_base > max_kv_visible) {
        break;
      }

      cooperative_load_bf16(
          sm_K, kFlashN * kFlashD, [&](size_t i) -> __nv_bfloat16 {
            const size_t n = i / kFlashD;
            const size_t d = i % kFlashD;
            const size_t kv_idx = k_base + n;
            if (kv_idx >= kv_total) {
              return __float2bfloat16(0.0f);
            }
            const size_t cache_index =
                (kv_idx * shape.kv_heads + kvh) * head_dim + d;
            return flash_load_kv_bf16(cache_k, kv_cache_dtype, cache_index);
          });
      cooperative_load_bf16(
          sm_V, kFlashN * kFlashD, [&](size_t i) -> __nv_bfloat16 {
            const size_t n = i / kFlashD;
            const size_t d = i % kFlashD;
            const size_t kv_idx = k_base + n;
            if (kv_idx >= kv_total) {
              return __float2bfloat16(0.0f);
            }
            const size_t cache_index =
                (kv_idx * shape.kv_heads + kvh) * head_dim + d;
            return flash_load_kv_bf16(cache_v, kv_cache_dtype, cache_index);
          });
      __syncthreads();

      // -------------------- S = Q @ K^T --------------------
      // Each warp computes 2 N-tiles of 16x16 for its M-tile.
      // n_in_half ∈ {0, 1} selects which 16-col tile within the warp's half.
#pragma unroll
      for (int n_in_half = 0; n_in_half < 2; ++n_in_half) {
        const int n_col = my_n_col_base + n_in_half * 16;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major>
            q_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                       wmma::col_major>
            k_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
        wmma::fill_fragment(s_frag, 0.0f);
#pragma unroll
        for (int d = 0; d < kFlashD; d += 16) {
          wmma::load_matrix_sync(q_frag, sm_Q + my_m_row_base * kFlashD + d,
                                 kFlashD);
          wmma::load_matrix_sync(k_frag, sm_K + n_col * kFlashD + d, kFlashD);
          wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
        }
        wmma::store_matrix_sync(sm_S + my_m_row_base * kFlashN + n_col, s_frag,
                                kFlashN, wmma::mem_row_major);
      }
      __syncthreads();

      // -------------------- causal mask + scale --------------------
      for (size_t i = threadIdx.x; i < kFlashM * kFlashN; i += blockDim.x) {
        const size_t r = i / kFlashN;
        const size_t c = i % kFlashN;
        const size_t row_pos = start_pos + token_base + r;
        const size_t col_pos = k_base + c;
        if (r >= rows_in_tile || col_pos > row_pos || col_pos >= kv_total) {
          sm_S[i] = -INFINITY;
        } else {
          sm_S[i] = sm_S[i] * qk_scale;
        }
      }
      __syncthreads();

      // -------------------- online softmax row update --------------------
      // One thread per Q-row.  Stores alpha[r] (rescale factor for old O),
      // updates m, l, and writes P bf16 row.
      if (threadIdx.x < rows_in_tile) {
        const int r = threadIdx.x;
        float row_max = -INFINITY;
#pragma unroll
        for (int c = 0; c < kFlashN; ++c) {
          row_max = fmaxf(row_max, sm_S[r * kFlashN + c]);
        }
        const float m_old = sm_m[r];
        const float m_new = fmaxf(m_old, row_max);
        const float alpha =
            (isinf(m_old) && m_old < 0.0f) ? 0.0f : expf(m_old - m_new);
        const bool row_all_masked = isinf(m_new) && m_new < 0.0f;
        float row_sum_p = 0.0f;
#pragma unroll
        for (int c = 0; c < kFlashN; ++c) {
          float p = 0.0f;
          if (!row_all_masked) {
            p = expf(sm_S[r * kFlashN + c] - m_new);
          }
          sm_P[r * kFlashN + c] = __float2bfloat16(p);
          row_sum_p += p;
        }
        sm_alpha[r] = alpha;
        sm_l[r] = alpha * sm_l[r] + row_sum_p;
        sm_m[r] = m_new;
      } else if (threadIdx.x < kFlashM) {
        // Padded rows: zero out their P row and set alpha = 0.
        sm_alpha[threadIdx.x] = 0.0f;
#pragma unroll
        for (int c = 0; c < kFlashN; ++c) {
          sm_P[threadIdx.x * kFlashN + c] = __float2bfloat16(0.0f);
        }
      }
      __syncthreads();

      // -------------------- rescale o_frags by alpha[row] --------------------
      // wmma 16x16 accumulator fragment per-thread layout (m16n16k16,
      // f32 acc): each lane holds 8 elements arranged as 4 pairs of cols.
      //   x[0..1] → row = my_m_row_base + lane/4,           cols c..c+1
      //   x[2..3] → row = my_m_row_base + lane/4 + 8,       cols c..c+1
      //   x[4..5] → row = my_m_row_base + lane/4,           cols (c+8)..(c+9)
      //   x[6..7] → row = my_m_row_base + lane/4 + 8,       cols (c+8)..(c+9)
      // The col is per-fragment so the row-rescale is identical across all
      // 8 D-fragments owned by this warp.  Loading alpha for the two rows
      // touched by this lane is enough.
      const int row_lo = my_m_row_base + (lane_id >> 2);
      const int row_hi = row_lo + 8;
      const float alpha_lo = (row_lo < kFlashM) ? sm_alpha[row_lo] : 0.0f;
      const float alpha_hi = (row_hi < kFlashM) ? sm_alpha[row_hi] : 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        o_frags[i].x[0] *= alpha_lo;
        o_frags[i].x[1] *= alpha_lo;
        o_frags[i].x[2] *= alpha_hi;
        o_frags[i].x[3] *= alpha_hi;
        o_frags[i].x[4] *= alpha_lo;
        o_frags[i].x[5] *= alpha_lo;
        o_frags[i].x[6] *= alpha_hi;
        o_frags[i].x[7] *= alpha_hi;
      }

      // -------------------- O += P @ V --------------------
      // P is matrix_a (row_major, ld=N), V is matrix_b (row_major, ld=D).
      // Each warp owns 8 D-output fragments and iterates 4 N-fragments
      // (N=64) per output cell.
#pragma unroll
      for (int d_local = 0; d_local < 8; ++d_local) {
        const int d_tile = my_d_tile_base + d_local;
        const int d_col = d_tile * 16;
#pragma unroll
        for (int n_frag = 0; n_frag < kFlashNTiles; ++n_frag) {
          const int n_offset = n_frag * 16;
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                         wmma::row_major>
              p_frag;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                         wmma::row_major>
              v_frag;
          wmma::load_matrix_sync(
              p_frag, sm_P + my_m_row_base * kFlashN + n_offset, kFlashN);
          wmma::load_matrix_sync(v_frag, sm_V + n_offset * kFlashD + d_col,
                                 kFlashD);
          wmma::mma_sync(o_frags[d_local], p_frag, v_frag, o_frags[d_local]);
        }
      }
      __syncthreads();
    }

    // ---- write output for this q_head ----
    // Reuse sm_S as scratch to store one warp's O-tile so we can divide by l
    // and write back as bf16 cooperatively.
    float *sm_O_tile = sm_S; // 16 * 256 floats = 16 KB, sm_S has 32*64*4 = 8 KB
    // sm_S only has 8KB so it isn't big enough for a 16x256 f32 tile.  Use
    // sm_S + sm_P + sm_m + sm_l + sm_alpha contiguous region (8 + 4 + 0.4 KB,
    // total ~13 KB) — also not enough.  Just write 16 D-frags at a time
    // through sm_S (8 KB) for the warp's M-tile rows.
    //
    // Concretely: sm_S can hold 16 rows × 128 cols of f32 (8 KB).  Each
    // warp owns 16 rows × 128 D (the 8 D-tiles of width 16 from
    // my_d_tile_base*16 to (my_d_tile_base+8)*16).  That fits.
    //
    // Two warps share the same M-tile; we serialise them so each gets to use
    // sm_S in turn.  Warps owning M-tile 1 wait, then their pair flips.
    // Simpler: each pair of warps owns DIFFERENT D ranges, so they can write
    // into DIFFERENT sm_S regions concurrently.  Use sm_S [16 × 256] f32?
    // We only have 8 KB → 2048 floats → 8 rows of 256.  Not enough.
    //
    // Cleanest fix: write directly to global output using the wmma store
    // primitive followed by a thread-wide scaling pass.  The wmma
    // store_matrix_sync writes into a contiguous SMEM/global region of
    // 16x16 floats — too big for our remaining SMEM, so we store to global
    // f32 scratch?  We don't have a global f32 scratch.
    //
    // Reuse the sm_K and sm_V regions which we no longer need at this point.
    // sm_K + sm_V = 64 KB of bf16 = 16 KB if we view as f32 (we have plenty
    // of capacity to view it as f32 — 16384 floats = enough for 16x256x4
    // (16 rows × 256 cols f32) per M-tile, so 8 KB per M-tile, two M-tiles
    // fit in 16 KB).  Use sm_K as the O scratch, treating it as
    // f32[2 × 16 × 256] (one block per M-tile).
    float *sm_O_scratch = reinterpret_cast<float *>(sm_K);
    // sm_K capacity is 64*256*2 = 32768 bytes = 8192 floats.  Need
    // 2 (M-tiles) × 16 × 256 = 8192 floats.  Exactly fits.
    __syncthreads();
#pragma unroll
    for (int d_local = 0; d_local < 8; ++d_local) {
      const int d_tile = my_d_tile_base + d_local;
      const int d_col = d_tile * 16;
      // M-tile 0 lives at sm_O_scratch[0..4095], tile 1 at
      // sm_O_scratch[4096..8191].  Each tile is 16 × 256 row-major.
      float *tile_base = sm_O_scratch + my_m_tile * 16 * kFlashD;
      wmma::store_matrix_sync(tile_base + d_col, o_frags[d_local], kFlashD,
                              wmma::mem_row_major);
    }
    __syncthreads();

    // Divide by l[r] and write bf16 to global.
    for (size_t i = threadIdx.x; i < kFlashM * kFlashD; i += blockDim.x) {
      const size_t r = i / kFlashD;
      const size_t d = i % kFlashD;
      if (r >= rows_in_tile) {
        continue;
      }
      const float l = sm_l[r];
      const float val = (l > 0.0f) ? (sm_O_scratch[i] / l) : 0.0f;
      output[((token_base + r) * shape.q_heads + qh) * head_dim + d] =
          __float2bfloat16(val);
    }
    __syncthreads();
  }
}

} // namespace

extern "C" int
qwen36_attention_flash_prefill(const qwen36_attention_prefill_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->tokens == 0 || spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0 ||
      spec->v_bf16.ptr == 0 || spec->kv_cache_k.ptr == 0 ||
      spec->kv_cache_v.ptr == 0 || spec->output_bf16.ptr == 0 ||
      spec->shape.head_dim != kFlashD || spec->shape.q_heads == 0 ||
      spec->shape.kv_heads == 0 ||
      spec->shape.q_heads % spec->shape.kv_heads != 0 ||
      (spec->kv_cache_dtype != kKvCacheBf16 &&
       spec->kv_cache_dtype != kKvCacheFp8)) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  // SMEM = Q + K + V (bf16) + S (f32) + P (bf16) + 3*M (f32).
  // Output scratch reuses sm_K so it doesn't add to the SMEM budget.
  const size_t smem_bytes =
      (kFlashM + 2 * kFlashN) * kFlashD * sizeof(__nv_bfloat16) +
      kFlashM * kFlashN * sizeof(float) +
      kFlashM * kFlashN * sizeof(__nv_bfloat16) +
      3 * kFlashM * sizeof(float);
  cudaFuncSetAttribute(attention_flash_prefill_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem_bytes));

  const dim3 grid(
      static_cast<unsigned int>(spec->shape.kv_heads),
      static_cast<unsigned int>((spec->tokens + kFlashM - 1) / kFlashM));
  attention_flash_prefill_kernel<<<grid, kFlashThreads, smem_bytes,
                                   qwen36_internal_active_stream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(spec->q_bf16.ptr)),
      reinterpret_cast<const void *>(
          static_cast<uintptr_t>(spec->kv_cache_k.ptr)),
      reinterpret_cast<const void *>(
          static_cast<uintptr_t>(spec->kv_cache_v.ptr)),
      spec->kv_cache_dtype,
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(spec->output_bf16.ptr)),
      spec->start_position,
      spec->start_position_device_i32.ptr != 0
          ? reinterpret_cast<const int32_t *>(
                static_cast<uintptr_t>(spec->start_position_device_i32.ptr))
          : nullptr,
      spec->tokens, spec->shape);

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
