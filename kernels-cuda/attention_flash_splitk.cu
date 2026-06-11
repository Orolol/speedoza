// Flash-attention split-K prefill for small-q / long-KV (DFlash verify).
//
// The DFlash verify pass runs a q=16 prefill chunk over the full target
// stack. The 16 full-attention layers are 89% of the verify wall time at
// 7K ctx (AGENT.md 2026-06-09 kill-gate) because q=16 is below
// kPrefillFlashMinTokens=1024: the normal flash prefill kernel grids only
// (kv_heads=4 × ceil(16/32)=1) = 4 CTAs (~2% of 170 SMs), so the dispatch
// falls back to the scalar GQA kernel that re-reads the full KV per token.
//
// This kernel adds a Flash-Decoding split-K dimension to the proven
// attention_flash_prefill tile: the KV sequence is partitioned across
// `n_splits` CTAs (grid.z), each computing a partial online-softmax over
// its KV chunk. A second reduce pass merges the per-split (m, l, O)
// partials via log-sum-exp. grid.y enumerates (q_tile, qh_local) pairs —
// one CTA per q_head (2026-06-11; previously a serial in-CTA q_head loop,
// which left the verify shape at (4 × 1 × n_splits) CTAs ≈ 6% occupancy at
// short ctx). With n_splits ~48 the grid is (4 × 6 × 48) = 1152 CTAs,
// saturating the GPU while staying numerically identical (FP32
// accumulators) to the full-KV tile — so it does NOT carry the
// precision/AL drift of the scalar split-GQA path.
//
// Tile: M=32 Q-rows (q<=32; verify uses 16, upper rows padded), N=64 KV
// rows per K-iter, D=256 head_dim, 4 warps, wmma m16n16k16 BF16 + FP32
// accum. Mechanically lifted from attention_flash_prefill.cu — same warp
// split, same online-softmax, same causal mask — with two changes:
//   1. the K-iter loop is bounded to this CTA's split range, and
//   2. the epilogue writes the UNNORMALIZED O + (m, l) to token-indexed
//      partials instead of dividing by l and writing the final output.
//
// Parity contract: cos >= 0.998 vs the scalar GQA / CPU FP32 reference at
// every (q<=32, ctx, n_splits) in the smoke gate. See
// docs/superpowers/specs/2026-06-09-dflash-mk-phase4-verify-megakernel.md.

#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

namespace {

constexpr int kSkM = 32;       // Q rows per block (q<=32; verify q=16)
constexpr int kSkN = 64;       // KV rows per K-iter
constexpr int kSkD = 256;      // head_dim
constexpr int kSkWarps = 4;
constexpr int kSkThreads = 32 * kSkWarps;
constexpr int kSkNTiles = kSkN / 16; // 4

constexpr int kSkKvCacheBf16 = 0;
constexpr int kSkKvCacheFp8 = 1;

__device__ __forceinline__ float sk_decode_e4m3(uint8_t code) {
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
sk_load_kv_bf16(const void *cache, int kv_cache_dtype, size_t index) {
  if (kv_cache_dtype == kSkKvCacheFp8) {
    return __float2bfloat16(
        sk_decode_e4m3(reinterpret_cast<const uint8_t *>(cache)[index]));
  }
  return reinterpret_cast<const __nv_bfloat16 *>(cache)[index];
}

template <typename SrcLoad>
__device__ __forceinline__ void
sk_coop_load_bf16(__nv_bfloat16 *dst, size_t count, SrcLoad src_load) {
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    dst[i] = src_load(i);
  }
}

// One CTA per (kv_head, q_tile, split). Writes per (token, q_head, split)
// partials: unnormalized O [.. * head_dim], running max m, running denom l.
__global__ void attention_flash_splitk_kernel(
    const __nv_bfloat16 *q, const void *cache_k, const void *cache_v,
    int kv_cache_dtype, float *partial_acc, float *partial_max,
    float *partial_denom, size_t start_position,
    const int32_t *start_position_device, size_t tokens,
    qwen36_attention_shape_t shape, int n_splits) {
  const size_t kvh = blockIdx.x;
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  // grid.y enumerates (q_tile, qh_local) pairs: one CTA per q_head instead
  // of a serial in-CTA q_head loop. At the verify shape (1 q-tile) this is
  // the difference between 12 and 72 CTAs on a 192-SM part — the kernel was
  // measured at 6% occupancy (grid (4,1,3), 550 us at ctx 128, 2026-06-11).
  // Per-(qh, split) arithmetic is unchanged: every qh iteration already
  // reloaded its own Q/K/V tiles and reset its softmax state, so the
  // partials are bit-identical to the looped version.
  const size_t q_tile_idx = blockIdx.y / q_per_kv;
  const size_t qh_local = blockIdx.y % q_per_kv;
  const size_t split = blockIdx.z;
  const size_t token_base = q_tile_idx * kSkM;
  if (token_base >= tokens) {
    return;
  }
  const size_t head_dim = shape.head_dim;
  const float qk_scale = rsqrtf(static_cast<float>(head_dim));
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int my_m_tile = warp_id / 2;       // 0 or 1
  const int my_n_half = warp_id & 1;       // 0 -> cols 0-31, 1 -> cols 32-63
  const int my_m_row_base = my_m_tile * 16;
  const int my_n_col_base = my_n_half * 32;
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
      (token_base + kSkM <= tokens) ? kSkM : (tokens - token_base);
  const size_t max_kv_visible = start_pos + token_base + rows_in_tile - 1;

  // K-iter range owned by this split. Total K-iters partitioned evenly.
  const size_t total_k_iters = (max_kv_visible + kSkN) / kSkN;
  const size_t iters_per_split =
      (total_k_iters + static_cast<size_t>(n_splits) - 1) /
      static_cast<size_t>(n_splits);
  const size_t k_iter_start = split * iters_per_split;
  const size_t k_iter_end =
      min(k_iter_start + iters_per_split, total_k_iters);

  extern __shared__ unsigned char smem_raw[];
  __nv_bfloat16 *sm_Q = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  __nv_bfloat16 *sm_K = sm_Q + kSkM * kSkD;
  __nv_bfloat16 *sm_V = sm_K + kSkN * kSkD;
  float *sm_S = reinterpret_cast<float *>(sm_V + kSkN * kSkD);
  __nv_bfloat16 *sm_P =
      reinterpret_cast<__nv_bfloat16 *>(sm_S + kSkM * kSkN);
  float *sm_m = reinterpret_cast<float *>(sm_P + kSkM * kSkN);
  float *sm_l = sm_m + kSkM;
  float *sm_alpha = sm_l + kSkM;

  {
    const size_t qh = kvh * q_per_kv + qh_local;

    // ---- load Q tile [M × D] for this q_head ----
    sk_coop_load_bf16(sm_Q, kSkM * kSkD, [&](size_t i) -> __nv_bfloat16 {
      const size_t row = i / kSkD;
      const size_t d = i % kSkD;
      if (row >= rows_in_tile) {
        return __float2bfloat16(0.0f);
      }
      return q[((token_base + row) * shape.q_heads + qh) * head_dim + d];
    });

    if (threadIdx.x < kSkM) {
      sm_m[threadIdx.x] = -INFINITY;
      sm_l[threadIdx.x] = 0.0f;
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frags[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      wmma::fill_fragment(o_frags[i], 0.0f);
    }
    __syncthreads();

    // ---- K-iter loop bounded to this split's range ----
    for (size_t k_iter = k_iter_start; k_iter < k_iter_end; ++k_iter) {
      const size_t k_base = k_iter * kSkN;
      if (k_base > max_kv_visible) {
        break;
      }
      sk_coop_load_bf16(sm_K, kSkN * kSkD, [&](size_t i) -> __nv_bfloat16 {
        const size_t n = i / kSkD;
        const size_t d = i % kSkD;
        const size_t kv_idx = k_base + n;
        if (kv_idx >= kv_total) {
          return __float2bfloat16(0.0f);
        }
        return sk_load_kv_bf16(cache_k, kv_cache_dtype,
                               (kv_idx * shape.kv_heads + kvh) * head_dim + d);
      });
      sk_coop_load_bf16(sm_V, kSkN * kSkD, [&](size_t i) -> __nv_bfloat16 {
        const size_t n = i / kSkD;
        const size_t d = i % kSkD;
        const size_t kv_idx = k_base + n;
        if (kv_idx >= kv_total) {
          return __float2bfloat16(0.0f);
        }
        return sk_load_kv_bf16(cache_v, kv_cache_dtype,
                               (kv_idx * shape.kv_heads + kvh) * head_dim + d);
      });
      __syncthreads();

      // -------------------- S = Q @ K^T --------------------
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
        for (int d = 0; d < kSkD; d += 16) {
          wmma::load_matrix_sync(q_frag, sm_Q + my_m_row_base * kSkD + d,
                                 kSkD);
          wmma::load_matrix_sync(k_frag, sm_K + n_col * kSkD + d, kSkD);
          wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
        }
        wmma::store_matrix_sync(sm_S + my_m_row_base * kSkN + n_col, s_frag,
                                kSkN, wmma::mem_row_major);
      }
      __syncthreads();

      // -------------------- causal mask + scale --------------------
      for (size_t i = threadIdx.x; i < kSkM * kSkN; i += blockDim.x) {
        const size_t r = i / kSkN;
        const size_t c = i % kSkN;
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
      if (threadIdx.x < rows_in_tile) {
        const int r = threadIdx.x;
        float row_max = -INFINITY;
#pragma unroll
        for (int c = 0; c < kSkN; ++c) {
          row_max = fmaxf(row_max, sm_S[r * kSkN + c]);
        }
        const float m_old = sm_m[r];
        const float m_new = fmaxf(m_old, row_max);
        const float alpha =
            (isinf(m_old) && m_old < 0.0f) ? 0.0f : expf(m_old - m_new);
        const bool row_all_masked = isinf(m_new) && m_new < 0.0f;
        float row_sum_p = 0.0f;
#pragma unroll
        for (int c = 0; c < kSkN; ++c) {
          float p = 0.0f;
          if (!row_all_masked) {
            p = expf(sm_S[r * kSkN + c] - m_new);
          }
          sm_P[r * kSkN + c] = __float2bfloat16(p);
          row_sum_p += p;
        }
        sm_alpha[r] = alpha;
        sm_l[r] = alpha * sm_l[r] + row_sum_p;
        sm_m[r] = m_new;
      } else if (threadIdx.x < kSkM) {
        sm_alpha[threadIdx.x] = 0.0f;
#pragma unroll
        for (int c = 0; c < kSkN; ++c) {
          sm_P[threadIdx.x * kSkN + c] = __float2bfloat16(0.0f);
        }
      }
      __syncthreads();

      // -------------------- rescale o_frags by alpha[row] --------------------
      const int row_lo = my_m_row_base + (lane_id >> 2);
      const int row_hi = row_lo + 8;
      const float alpha_lo = (row_lo < kSkM) ? sm_alpha[row_lo] : 0.0f;
      const float alpha_hi = (row_hi < kSkM) ? sm_alpha[row_hi] : 0.0f;
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
#pragma unroll
      for (int d_local = 0; d_local < 8; ++d_local) {
        const int d_tile = my_d_tile_base + d_local;
        const int d_col = d_tile * 16;
#pragma unroll
        for (int n_frag = 0; n_frag < kSkNTiles; ++n_frag) {
          const int n_offset = n_frag * 16;
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                         wmma::row_major>
              p_frag;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                         wmma::row_major>
              v_frag;
          wmma::load_matrix_sync(
              p_frag, sm_P + my_m_row_base * kSkN + n_offset, kSkN);
          wmma::load_matrix_sync(v_frag, sm_V + n_offset * kSkD + d_col,
                                 kSkD);
          wmma::mma_sync(o_frags[d_local], p_frag, v_frag, o_frags[d_local]);
        }
      }
      __syncthreads();
    }

    // ---- write UNNORMALIZED O + (m, l) to token-indexed partials ----
    // Reuse sm_K as FP32 O scratch (64*256 bf16 = 32 KB = 8K floats; need
    // kSkM*kSkD = 8K floats — exact fit).
    float *sm_O_scratch = reinterpret_cast<float *>(sm_K);
    __syncthreads();
#pragma unroll
    for (int d_local = 0; d_local < 8; ++d_local) {
      const int d_tile = my_d_tile_base + d_local;
      const int d_col = d_tile * 16;
      float *tile_base = sm_O_scratch + my_m_tile * 16 * kSkD;
      wmma::store_matrix_sync(tile_base + d_col, o_frags[d_local], kSkD,
                              wmma::mem_row_major);
    }
    __syncthreads();

    // partials layout:
    //   acc[((token * q_heads + qh) * n_splits + split) * head_dim + d]
    //   m/l[ (token * q_heads + qh) * n_splits + split ]
    for (size_t i = threadIdx.x; i < kSkM * kSkD; i += blockDim.x) {
      const size_t r = i / kSkD;
      const size_t d = i % kSkD;
      if (r >= rows_in_tile) {
        continue;
      }
      const size_t token = token_base + r;
      const size_t pidx =
          ((token * shape.q_heads + qh) * n_splits + split) * head_dim + d;
      partial_acc[pidx] = sm_O_scratch[i];
    }
    if (threadIdx.x < rows_in_tile) {
      const size_t token = token_base + threadIdx.x;
      const size_t sidx = (token * shape.q_heads + qh) * n_splits + split;
      partial_max[sidx] = sm_m[threadIdx.x];
      partial_denom[sidx] = sm_l[threadIdx.x];
    }
    __syncthreads();
  }
}

// Reduce the per-split partials into the final attention output. One CTA
// per (token, q_head); head_dim threads. Standard Flash-Decoding
// log-sum-exp merge in FP32.
__global__ void attention_flash_splitk_reduce_kernel(
    const float *partial_acc, const float *partial_max,
    const float *partial_denom, __nv_bfloat16 *output,
    qwen36_attention_shape_t shape, int n_splits, size_t tokens) {
  __shared__ float gmax;
  __shared__ float gdenom;

  const size_t flat = blockIdx.x; // token * q_heads + qh
  const size_t token = flat / shape.q_heads;
  if (token >= tokens) {
    return;
  }
  const size_t head_dim = shape.head_dim;
  const size_t d = threadIdx.x;

  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
      m = fmaxf(m, partial_max[flat * n_splits + s]);
    }
    float dn = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
      const float pm = partial_max[flat * n_splits + s];
      const float pd = partial_denom[flat * n_splits + s];
      const float scale = isinf(pm) && pm < 0.0f ? 0.0f : expf(pm - m);
      dn += pd * scale;
    }
    gmax = m;
    gdenom = dn;
  }
  __syncthreads();
  if (d >= head_dim) {
    return;
  }
  const float m = gmax;
  const float dn = gdenom;
  float acc_total = 0.0f;
  for (int s = 0; s < n_splits; ++s) {
    const float pm = partial_max[flat * n_splits + s];
    const float pa = partial_acc[(flat * n_splits + s) * head_dim + d];
    const float scale = isinf(pm) && pm < 0.0f ? 0.0f : expf(pm - m);
    acc_total += pa * scale;
  }
  output[flat * head_dim + d] =
      __float2bfloat16(dn > 0.0f ? acc_total / dn : 0.0f);
}

} // namespace

// Persistent grow-on-demand scratch for the split-K partials. Avoids a
// cudaMalloc/cudaFree (each Free implicitly syncs) on every attention
// call — there are 16 full-attn layers × many speculative iters per
// generation, so per-call alloc dominated. The scratch is reused on the
// stream-ordered active stream: within one entry call the split kernel
// writes the partials and the reduce consumes them before returning, and
// the next entry call (next layer) is ordered after on the same stream,
// so reuse is safe. Bounded by tokens(≤32) × q_heads × n_splits(≤48) ×
// (head_dim+2) ≈ 75 MB worst case. Freed at process exit (leaked — the
// process owns the device for its lifetime).
namespace {
struct SplitKScratch {
  float *acc = nullptr;
  float *max = nullptr;
  float *denom = nullptr;
  size_t acc_floats = 0;   // capacity in floats
  size_t scalar_floats = 0;
};
SplitKScratch g_splitk_scratch;

// True if the active stream is currently capturing a CUDA graph. cudaMalloc/
// cudaFree are illegal during capture and would invalidate it, so a grow
// that would fire mid-capture must fail loudly instead of corrupting the
// graph. After the engine pre-warms the scratch to its worst-case size
// (eager, outside capture) the grow branch is unreachable and this guard
// never trips; it is the belt-and-suspenders the adversarial review
// (wf_a36ff789-8b8) required for default-on.
bool splitk_active_stream_capturing() {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(qwen36_internal_active_stream(), &status) !=
      cudaSuccess) {
    return false;
  }
  return status != cudaStreamCaptureStatusNone;
}

enum SplitKReserve { kReserveOk = 0, kReserveCaptureBlocked = 1, kReserveOom = 2 };

// Ensure the scratch holds at least `partial_count` (scalar) and
// `partial_count * head_dim` (acc) floats. Returns kReserveCaptureBlocked
// if a grow would be required during an active graph capture (the caller
// then falls back to the capture-safe scalar GQA path), kReserveOom on
// allocation failure.
int splitk_scratch_reserve(size_t partial_count, size_t head_dim) {
  const size_t want_acc = partial_count * head_dim;
  const bool needs_grow = want_acc > g_splitk_scratch.acc_floats ||
                          partial_count > g_splitk_scratch.scalar_floats;
  if (needs_grow && splitk_active_stream_capturing()) {
    return kReserveCaptureBlocked;
  }
  if (want_acc > g_splitk_scratch.acc_floats) {
    cudaFree(g_splitk_scratch.acc);
    g_splitk_scratch.acc = nullptr;
    if (cudaMalloc(&g_splitk_scratch.acc, want_acc * sizeof(float)) !=
        cudaSuccess) {
      g_splitk_scratch.acc_floats = 0;
      return kReserveOom;
    }
    g_splitk_scratch.acc_floats = want_acc;
  }
  if (partial_count > g_splitk_scratch.scalar_floats) {
    cudaFree(g_splitk_scratch.max);
    cudaFree(g_splitk_scratch.denom);
    g_splitk_scratch.max = nullptr;
    g_splitk_scratch.denom = nullptr;
    if (cudaMalloc(&g_splitk_scratch.max, partial_count * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(&g_splitk_scratch.denom, partial_count * sizeof(float)) !=
            cudaSuccess) {
      g_splitk_scratch.scalar_floats = 0;
      return kReserveOom;
    }
    g_splitk_scratch.scalar_floats = partial_count;
  }
  return kReserveOk;
}
} // namespace

// Entry point: runs the split-K kernel + reduce using a persistent scratch.
// Numerically faithful to the scalar GQA prefill (smoke parity cos>=0.998).
extern "C" int qwen36_attention_flash_splitk_prefill_bf16(
    const qwen36_attention_prefill_spec_t *spec, int n_splits) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->tokens == 0 || spec->tokens > static_cast<size_t>(kSkM) ||
      spec->q_bf16.ptr == 0 || spec->kv_cache_k.ptr == 0 ||
      spec->kv_cache_v.ptr == 0 || spec->output_bf16.ptr == 0 ||
      spec->shape.head_dim != kSkD || spec->shape.q_heads == 0 ||
      spec->shape.kv_heads == 0 ||
      spec->shape.q_heads % spec->shape.kv_heads != 0 ||
      (spec->kv_cache_dtype != kSkKvCacheBf16 &&
       spec->kv_cache_dtype != kSkKvCacheFp8) ||
      n_splits < 1) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  const size_t tokens = spec->tokens;
  const size_t q_heads = spec->shape.q_heads;
  const size_t head_dim = spec->shape.head_dim;
  const size_t partial_count =
      tokens * q_heads * static_cast<size_t>(n_splits);

  // Prefer engine-owned partials passed via the prefill spec — pre-reserved
  // once at engine init to the verify worst case (#55). Capture-safe: no
  // cudaMalloc in the hot path, so a captured [9,32] chunk can use split-K.
  // Fall back to the process-global grow-on-demand scratch only when the
  // caller supplies none (the kernel-vs-kernel smoke uses spec{} = NULL).
  float *partial_acc;
  float *partial_max;
  float *partial_denom;
  if (spec->partial_acc_f32.ptr != 0 && spec->partial_max_f32.ptr != 0 &&
      spec->partial_denom_f32.ptr != 0) {
    partial_acc = reinterpret_cast<float *>(
        static_cast<uintptr_t>(spec->partial_acc_f32.ptr));
    partial_max = reinterpret_cast<float *>(
        static_cast<uintptr_t>(spec->partial_max_f32.ptr));
    partial_denom = reinterpret_cast<float *>(
        static_cast<uintptr_t>(spec->partial_denom_f32.ptr));
  } else {
    const int reserve = splitk_scratch_reserve(partial_count, head_dim);
    if (reserve == kReserveCaptureBlocked) {
      // Would need to cudaMalloc inside a graph capture — fall back to the
      // capture-safe scalar path (dispatch treats NOT_IMPLEMENTED as a
      // fall-through). Unreachable once engine-owned partials are wired.
      return QWEN36_STATUS_NOT_IMPLEMENTED;
    }
    if (reserve == kReserveOom) {
      return QWEN36_STATUS_CUDA_ERROR;
    }
    partial_acc = g_splitk_scratch.acc;
    partial_max = g_splitk_scratch.max;
    partial_denom = g_splitk_scratch.denom;
  }

  const size_t smem_bytes =
      (kSkM + 2 * kSkN) * kSkD * sizeof(__nv_bfloat16) +
      kSkM * kSkN * sizeof(float) + kSkM * kSkN * sizeof(__nv_bfloat16) +
      3 * kSkM * sizeof(float);
  cudaFuncSetAttribute(attention_flash_splitk_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem_bytes));

  const size_t q_per_kv = spec->shape.q_heads / spec->shape.kv_heads;
  const dim3 grid(
      static_cast<unsigned int>(spec->shape.kv_heads),
      static_cast<unsigned int>(((tokens + kSkM - 1) / kSkM) * q_per_kv),
      static_cast<unsigned int>(n_splits));
  attention_flash_splitk_kernel<<<grid, kSkThreads, smem_bytes,
                                  qwen36_internal_active_stream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(spec->q_bf16.ptr)),
      reinterpret_cast<const void *>(
          static_cast<uintptr_t>(spec->kv_cache_k.ptr)),
      reinterpret_cast<const void *>(
          static_cast<uintptr_t>(spec->kv_cache_v.ptr)),
      spec->kv_cache_dtype, partial_acc, partial_max, partial_denom,
      spec->start_position,
      spec->start_position_device_i32.ptr != 0
          ? reinterpret_cast<const int32_t *>(
                static_cast<uintptr_t>(spec->start_position_device_i32.ptr))
          : nullptr,
      tokens, spec->shape, n_splits);

  unsigned int reduce_threads = static_cast<unsigned int>(head_dim);
  attention_flash_splitk_reduce_kernel<<<
      static_cast<unsigned int>(tokens * q_heads), reduce_threads, 0,
      qwen36_internal_active_stream()>>>(
      partial_acc, partial_max, partial_denom,
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(spec->output_bf16.ptr)),
      spec->shape, n_splits, tokens);

  // Scratch is persistent — do NOT free; reused by the next call.
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
