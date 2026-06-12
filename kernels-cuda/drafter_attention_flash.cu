// DFlash drafter attention — FlashAttention-2-tiled BF16 kernel.
//
// Replaces the naive `O(q_len × kv_seq_len)` inner-loop kernel from
// `kernels-cuda/drafter_attention.cu` for the supported shape
// (head_dim=128, q_len=16, non-causal). The naive kernel stays in tree
// as `drafter_attention_v1_kernel` for parity reference and as a soft
// fallback for unsupported shapes (q_len != 16, head_dim != 128, or
// `QWEN36_DRAFTER_ATTENTION_DISABLE_FLASH=1`).
//
// Design and parity contract are in
// `docs/superpowers/specs/2026-06-09-dflash-fa-drafter-attention.md`.
//
// Algorithm: FlashAttention-2 forward
//   * Grid:    (q_heads, 1, 1) — one CTA per query head
//   * Block:   4 warps × 32 = 128 threads
//   * Q tile:  M=16 rows kept SMEM-resident for all KV tiles
//   * KV tile: N=64 rows streamed via cp.async-free coop loads
//   * MMA:     wmma m16n16k16, BF16 inputs, FP32 accumulators
//   * Softmax: online (max+sum+rescale) per row in SMEM
//
// Warp split (1 m-tile only since q_len=16):
//   QK^T S=[M×N]: 4 warps × 1 n-tile each (N=64 / 16 = 4 tiles)
//   PV   O+=PV:   4 warps × 2 d-tiles each (D=128 / 16 = 8 tiles, /4 warps)

#include "include/qwen36_fp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>
#include <stdint.h>

namespace wmma = nvcuda::wmma;

namespace {

constexpr int kFlashM = 16;       // q_len, drafter block_size
constexpr int kFlashN = 64;       // KV rows per outer K-iter
constexpr int kFlashD = 128;      // head_dim
constexpr int kFlashWarps = 4;
constexpr int kFlashThreads = kFlashWarps * 32;

// Per-warp D-tile fanout for PV: head_dim / 16 / warps = 8/4 = 2.
constexpr int kPvDFragsPerWarp = (kFlashD / 16) / kFlashWarps;
// QK^T: N=64 cols / 16 / warps = 4/4 = 1 n-tile per warp.
constexpr int kQkNFragsPerWarp = (kFlashN / 16) / kFlashWarps;

template <typename T> T *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

template <typename SrcLoad>
__device__ __forceinline__ void
flash_coop_load_bf16(__nv_bfloat16 *dst, size_t count, SrcLoad src_load) {
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    dst[i] = src_load(i);
  }
}

__global__ void __launch_bounds__(kFlashThreads, 2)
    drafter_attention_flash_kernel(
        const __nv_bfloat16 *__restrict__ q,
        const __nv_bfloat16 *__restrict__ k,
        const __nv_bfloat16 *__restrict__ v,
        __nv_bfloat16 *__restrict__ output,
        int kv_seq_len, int q_len, int q_heads, int kv_heads,
        int sliding_window, float qk_scale) {
  const int q_head = blockIdx.x;
  if (q_head >= q_heads) {
    return;
  }
  const int kv_head = (q_head * kv_heads) / q_heads;
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  // For QK^T: warp w owns 1 n-tile starting at col w*16.
  const int my_n_col_base = warp_id * 16;
  // For PV: warp w owns d-tiles [w*2, w*2+1].
  const int my_d_tile_base = warp_id * kPvDFragsPerWarp;

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

  const int rows_in_tile = (kFlashM < q_len) ? kFlashM : q_len;

  // ---- Load Q[:, q_head, :] into sm_Q once ----
  flash_coop_load_bf16(
      sm_Q, static_cast<size_t>(kFlashM) * kFlashD,
      [&](size_t i) -> __nv_bfloat16 {
        const size_t row = i / kFlashD;
        const size_t d = i % kFlashD;
        if (static_cast<int>(row) >= rows_in_tile) {
          return __float2bfloat16(0.0f);
        }
        return q[(row * q_heads + q_head) * kFlashD + d];
      });

  // Online softmax row state.
  if (threadIdx.x < kFlashM) {
    sm_m[threadIdx.x] = -INFINITY;
    sm_l[threadIdx.x] = 0.0f;
  }

  // O accumulator: per warp owns kPvDFragsPerWarp wmma 16x16 fragments.
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>
      o_frags[kPvDFragsPerWarp];
#pragma unroll
  for (int i = 0; i < kPvDFragsPerWarp; ++i) {
    wmma::fill_fragment(o_frags[i], 0.0f);
  }
  __syncthreads();

  const int k_iters = (kv_seq_len + kFlashN - 1) / kFlashN;
  // Drafter convention: Q absolute position = (kv_seq_len - q_len) + q_pos.
  const int q_abs_base = kv_seq_len - q_len;
  const bool has_swa = (sliding_window > 0);

  for (int k_iter = 0; k_iter < k_iters; ++k_iter) {
    const int k_base = k_iter * kFlashN;

    flash_coop_load_bf16(
        sm_K, static_cast<size_t>(kFlashN) * kFlashD,
        [&](size_t i) -> __nv_bfloat16 {
          const size_t n = i / kFlashD;
          const size_t d = i % kFlashD;
          const int kv_idx = k_base + static_cast<int>(n);
          if (kv_idx >= kv_seq_len) {
            return __float2bfloat16(0.0f);
          }
          return k[(static_cast<size_t>(kv_idx) * kv_heads + kv_head) *
                       kFlashD +
                   d];
        });
    flash_coop_load_bf16(
        sm_V, static_cast<size_t>(kFlashN) * kFlashD,
        [&](size_t i) -> __nv_bfloat16 {
          const size_t n = i / kFlashD;
          const size_t d = i % kFlashD;
          const int kv_idx = k_base + static_cast<int>(n);
          if (kv_idx >= kv_seq_len) {
            return __float2bfloat16(0.0f);
          }
          return v[(static_cast<size_t>(kv_idx) * kv_heads + kv_head) *
                       kFlashD +
                   d];
        });
    __syncthreads();

    // -------------------- S = Q @ K^T  (M=16, N=16 per warp, D=128) --------------------
#pragma unroll
    for (int n_in_warp = 0; n_in_warp < kQkNFragsPerWarp; ++n_in_warp) {
      const int n_col = my_n_col_base + n_in_warp * 16;
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
        wmma::load_matrix_sync(q_frag, sm_Q + d, kFlashD);
        wmma::load_matrix_sync(k_frag, sm_K + n_col * kFlashD + d, kFlashD);
        wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
      }
      wmma::store_matrix_sync(sm_S + n_col, s_frag, kFlashN,
                              wmma::mem_row_major);
    }
    __syncthreads();

    // -------------------- mask + scale --------------------
    // Drafter is non-causal. Apply scale + optional symmetric SWA mask.
    for (size_t i = threadIdx.x; i < kFlashM * kFlashN; i += blockDim.x) {
      const size_t r = i / kFlashN;
      const size_t c = i % kFlashN;
      const int kv_idx = k_base + static_cast<int>(c);
      const bool row_oob = static_cast<int>(r) >= rows_in_tile;
      const bool col_oob = kv_idx >= kv_seq_len;
      bool swa_masked = false;
      if (has_swa && !row_oob && !col_oob) {
        const int q_abs = q_abs_base + static_cast<int>(r);
        const int delta = q_abs - kv_idx;
        const int abs_delta = delta < 0 ? -delta : delta;
        swa_masked = (abs_delta > sliding_window);
      }
      if (row_oob || col_oob || swa_masked) {
        sm_S[i] = -INFINITY;
      } else {
        sm_S[i] = sm_S[i] * qk_scale;
      }
    }
    __syncthreads();

    // -------------------- online softmax row update --------------------
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
      // Padded row: zero P, alpha = 0.
      sm_alpha[threadIdx.x] = 0.0f;
#pragma unroll
      for (int c = 0; c < kFlashN; ++c) {
        sm_P[threadIdx.x * kFlashN + c] = __float2bfloat16(0.0f);
      }
    }
    __syncthreads();

    // -------------------- rescale o_frags by alpha[row] --------------------
    // wmma 16x16 FP32 accumulator layout (m16n16k16): each lane holds 8
    // floats arranged as 4 pairs of cols. The row owned by each pair is:
    //   x[0..1] → row = lane/4         (cols c..c+1)
    //   x[2..3] → row = lane/4 + 8     (cols c..c+1)
    //   x[4..5] → row = lane/4         (cols c+8..c+9)
    //   x[6..7] → row = lane/4 + 8     (cols c+8..c+9)
    // With kFlashM=16 and 1 m-tile, row_lo ∈ [0,8) and row_hi ∈ [8,16).
    const int row_lo = (lane_id >> 2);
    const int row_hi = row_lo + 8;
    const float alpha_lo = (row_lo < kFlashM) ? sm_alpha[row_lo] : 0.0f;
    const float alpha_hi = (row_hi < kFlashM) ? sm_alpha[row_hi] : 0.0f;
#pragma unroll
    for (int i = 0; i < kPvDFragsPerWarp; ++i) {
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
    // P is matrix_a row_major [M=16, K=64], V is matrix_b row_major
    // [K=64, N=128]. Each warp owns kPvDFragsPerWarp output D-tiles and
    // iterates kFlashN/16 = 4 K-fragments.
#pragma unroll
    for (int d_local = 0; d_local < kPvDFragsPerWarp; ++d_local) {
      const int d_tile = my_d_tile_base + d_local;
      const int d_col = d_tile * 16;
#pragma unroll
      for (int n_frag = 0; n_frag < (kFlashN / 16); ++n_frag) {
        const int n_offset = n_frag * 16;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major>
            p_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major>
            v_frag;
        wmma::load_matrix_sync(p_frag, sm_P + n_offset, kFlashN);
        wmma::load_matrix_sync(v_frag, sm_V + n_offset * kFlashD + d_col,
                               kFlashD);
        wmma::mma_sync(o_frags[d_local], p_frag, v_frag, o_frags[d_local]);
      }
    }
    __syncthreads();
  }

  // ---- normalise and write output for this q_head ----
  // Reuse sm_K (16384 bytes = 4096 floats) as FP32 O scratch — sized
  // exactly for kFlashM × kFlashD = 16 × 128 = 2048 floats with margin.
  float *sm_O_scratch = reinterpret_cast<float *>(sm_K);
  __syncthreads();
  // Each warp stores its kPvDFragsPerWarp d-tiles into the scratch.
#pragma unroll
  for (int d_local = 0; d_local < kPvDFragsPerWarp; ++d_local) {
    const int d_tile = my_d_tile_base + d_local;
    const int d_col = d_tile * 16;
    wmma::store_matrix_sync(sm_O_scratch + d_col, o_frags[d_local], kFlashD,
                            wmma::mem_row_major);
  }
  __syncthreads();

  // Divide by sm_l[row] and write BF16 output cooperatively.
  for (size_t i = threadIdx.x; i < kFlashM * kFlashD; i += blockDim.x) {
    const size_t row = i / kFlashD;
    if (static_cast<int>(row) >= rows_in_tile) {
      continue;
    }
    const size_t d = i % kFlashD;
    const float l = sm_l[row];
    const float o = sm_O_scratch[i];
    const float out = (l > 0.0f) ? (o / l) : 0.0f;
    output[(row * q_heads + q_head) * kFlashD + d] = __float2bfloat16(out);
  }
}

size_t compute_flash_smem_bytes() {
  size_t bytes = 0;
  bytes += static_cast<size_t>(kFlashM) * kFlashD * sizeof(__nv_bfloat16);
  bytes += static_cast<size_t>(kFlashN) * kFlashD * sizeof(__nv_bfloat16);
  bytes += static_cast<size_t>(kFlashN) * kFlashD * sizeof(__nv_bfloat16);
  bytes += static_cast<size_t>(kFlashM) * kFlashN * sizeof(float);
  bytes += static_cast<size_t>(kFlashM) * kFlashN * sizeof(__nv_bfloat16);
  bytes += 3 * static_cast<size_t>(kFlashM) * sizeof(float); // m, l, alpha
  return bytes;
}

bool flash_kernel_enabled_via_env() {
  // OPT-IN by default. The FA-tiled kernel measures at per-iter parity
  // with the v1 naive kernel on the drafter shape mix (q_len=16,
  // q_heads=32, head_dim=128) — the drafter forward at long context is
  // not compute-bound, so the wmma tiling doesn't win, and a
  // numerical-drift bug at medium kv_seq_len degrades DFlash AL. Until
  // that's diagnosed, fall back to v1 unless explicitly enabled.
  const char *val = std::getenv("QWEN36_DRAFTER_ATTENTION_FLASH");
  if (val == nullptr) {
    return false;
  }
  return val[0] == '1' || val[0] == 't' || val[0] == 'T' || val[0] == 'y' ||
         val[0] == 'Y';
}

} // namespace

extern "C" int qwen36_drafter_attention_block_flash_bf16(
    const qwen36_drafter_attention_block_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0 ||
      spec->v_bf16.ptr == 0 || spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->head_dim != kFlashD || spec->q_len != kFlashM) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }
  if (spec->q_heads == 0 || spec->kv_heads == 0 || spec->kv_seq_len == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (spec->q_heads % spec->kv_heads != 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (!flash_kernel_enabled_via_env()) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  const float qk_scale = rsqrtf(static_cast<float>(spec->head_dim));

  const auto *q = ptr<const __nv_bfloat16>(spec->q_bf16);
  const auto *k = ptr<const __nv_bfloat16>(spec->k_bf16);
  const auto *v = ptr<const __nv_bfloat16>(spec->v_bf16);
  auto *output = ptr<__nv_bfloat16>(spec->output_bf16);

  const size_t smem_bytes = compute_flash_smem_bytes();
  cudaError_t err = cudaFuncSetAttribute(
      drafter_attention_flash_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  if (err != cudaSuccess) {
    return QWEN36_STATUS_CUDA_ERROR;
  }

  dim3 grid(static_cast<unsigned>(spec->q_heads), 1, 1);
  dim3 block(kFlashThreads, 1, 1);

  drafter_attention_flash_kernel<<<grid, block, smem_bytes, 0>>>(
      q, k, v, output, static_cast<int>(spec->kv_seq_len),
      static_cast<int>(spec->q_len), static_cast<int>(spec->q_heads),
      static_cast<int>(spec->kv_heads),
      static_cast<int>(spec->sliding_window), qk_scale);

  err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
