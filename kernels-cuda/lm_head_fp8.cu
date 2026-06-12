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

// ---------------------------------------------------------------------------
// Two-stage exact argmax, stage 1: per-row top-2 + margin verdict over the
// FP8-path logits. Two passes so the scan is grid-wide (the historical
// sample_argmax runs ONE block over the whole vocab — ~170 us for a 0.5 MB
// read; this pair lands ~10-15 us).
//
// top-2 merge discipline: every thread tracks ITS OWN top-2 (a block top-2
// built from per-thread top-1s would be wrong — the two global leaders can
// land in the same thread's stride). Merging two top-2 pairs keeps the
// best (v, i) and the better of the two seconds. Ties break toward the
// smaller index for determinism; exactness never rests on ties (a tie
// means margin 0 -> stage 2 rescores in BF16 anyway).

struct Top2 {
  float v1;
  float v2;
  uint32_t i1;
};

__device__ inline void top2_consider(Top2 &t, float v, uint32_t i) {
  if (v > t.v1 || (v == t.v1 && i < t.i1)) {
    t.v2 = t.v1;
    t.v1 = v;
    t.i1 = i;
  } else if (v > t.v2) {
    t.v2 = v;
  }
}

__device__ inline void top2_merge(Top2 &t, const Top2 &o) {
  if (o.v1 > t.v1 || (o.v1 == t.v1 && o.i1 < t.i1)) {
    t.v2 = fmaxf(t.v1, o.v2);
    t.v1 = o.v1;
    t.i1 = o.i1;
  } else {
    t.v2 = fmaxf(t.v2, o.v1);
  }
}

constexpr int kTop2Threads = 256;
constexpr int kTop2Blocks = QWEN36_LM_HEAD_TOP2_BLOCKS;

// Workspace entry layout: 16 bytes per (row, block).
struct Top2WsEntry {
  float v1;
  float v2;
  uint32_t i1;
  uint32_t pad;
};

__device__ inline Top2 top2_block_reduce(Top2 mine, Top2 *warp_smem) {
  const unsigned lane = threadIdx.x & 31;
  const unsigned warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    Top2 other;
    other.v1 = __shfl_down_sync(0xffffffffu, mine.v1, offset);
    other.v2 = __shfl_down_sync(0xffffffffu, mine.v2, offset);
    other.i1 = __shfl_down_sync(0xffffffffu, mine.i1, offset);
    top2_merge(mine, other);
  }
  if (lane == 0) {
    warp_smem[warp] = mine;
  }
  __syncthreads();
  if (warp == 0) {
    const unsigned n_warps = (blockDim.x + 31) >> 5;
    mine = lane < n_warps
               ? warp_smem[lane]
               : Top2{-INFINITY, -INFINITY, 0xffffffffu};
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      Top2 other;
      other.v1 = __shfl_down_sync(0xffffffffu, mine.v1, offset);
      other.v2 = __shfl_down_sync(0xffffffffu, mine.v2, offset);
      other.i1 = __shfl_down_sync(0xffffffffu, mine.i1, offset);
      top2_merge(mine, other);
    }
  }
  return mine;
}

__global__ void __launch_bounds__(kTop2Threads)
lm_head_top2_pass1_kernel(const __nv_bfloat16 *__restrict__ logits,
                          Top2WsEntry *__restrict__ ws, size_t rows,
                          size_t vocab) {
  __shared__ Top2 warp_smem[kTop2Threads / 32];
  const size_t row = blockIdx.y;
  if (row >= rows) {
    return;
  }
  const __nv_bfloat16 *row_logits = logits + row * vocab;
  Top2 mine{-INFINITY, -INFINITY, 0xffffffffu};
  for (size_t i = static_cast<size_t>(blockIdx.x) * kTop2Threads + threadIdx.x;
       i < vocab; i += static_cast<size_t>(gridDim.x) * kTop2Threads) {
    top2_consider(mine, __bfloat162float(row_logits[i]),
                  static_cast<uint32_t>(i));
  }
  mine = top2_block_reduce(mine, warp_smem);
  if (threadIdx.x == 0) {
    ws[row * gridDim.x + blockIdx.x] = Top2WsEntry{mine.v1, mine.v2, mine.i1, 0u};
  }
}

__global__ void __launch_bounds__(kTop2Threads)
lm_head_top2_pass2_kernel(const Top2WsEntry *__restrict__ ws,
                          uint32_t *__restrict__ tokens,
                          uint32_t *__restrict__ flags,
                          uint32_t *__restrict__ mirror_last,
                          uint32_t *__restrict__ fallback_count, size_t rows,
                          unsigned blocks_per_row, float eps) {
  __shared__ Top2 warp_smem[kTop2Threads / 32];
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  Top2 mine{-INFINITY, -INFINITY, 0xffffffffu};
  for (unsigned b = threadIdx.x; b < blocks_per_row; b += kTop2Threads) {
    const Top2WsEntry e = ws[row * blocks_per_row + b];
    top2_merge(mine, Top2{e.v1, e.v2, e.i1});
  }
  mine = top2_block_reduce(mine, warp_smem);
  if (threadIdx.x == 0) {
    const bool guard_ok = (mine.v1 - mine.v2) >= eps;
    tokens[row] = mine.i1;
    flags[row] = guard_ok ? 1u : 0u;
    if (mirror_last != nullptr && row + 1 == rows) {
      *mirror_last = mine.i1;
    }
    if (!guard_ok && fallback_count != nullptr) {
      atomicAdd(fallback_count, 1u);
    }
  }
}

// ---------------------------------------------------------------------------
// v2 stage-1 verdict: top-8 candidate rescore. One CTA per row, after the
// shared block-top-2 pass. Phases (256 threads = 8 warps):
//   1. load the 240 block winners (v1, i1) into SMEM; block-reduce the max
//      of all block v2s (any non-winner is bounded by it).
//   2. warp 0 runs 9 max-extraction passes over the winners: passes 0..7
//      pick the candidates, pass 8 yields the 9th-best winner. Guard bound
//      B = max(9th winner, max block v2) bounds EVERY non-candidate.
//   3. warp w rescores candidate w: <input_row, weight[cand_w]> with FP64
//      lane partials + FP64 shuffle reduce — order-stable to ~1e-12, so
//      near-tie flips vs an exact dot are out of the picture.
//   4. thread 0: best rescored candidate -> token; guard_ok iff
//      best >= B + eps (every non-candidate bf16 logit <= fp8 + e_max).
constexpr int kTop8Candidates = 8;
constexpr int kTop8Threads = 256;

__global__ void __launch_bounds__(kTop8Threads)
lm_head_top8_rescore_kernel(const Top2WsEntry *__restrict__ ws,
                            const __nv_bfloat16 *__restrict__ weight,
                            const __nv_bfloat16 *__restrict__ input,
                            uint32_t *__restrict__ tokens,
                            uint32_t *__restrict__ flags,
                            uint32_t *__restrict__ mirror_last,
                            uint32_t *__restrict__ fallback_count, size_t rows,
                            size_t vocab, size_t cols,
                            unsigned blocks_per_row, float eps) {
  __shared__ float winner_v[kTop2Blocks];
  __shared__ uint32_t winner_i[kTop2Blocks];
  __shared__ float warp_max_v2[kTop8Threads / 32];
  __shared__ float cand_v[kTop8Candidates];
  __shared__ uint32_t cand_i[kTop8Candidates];
  __shared__ float ninth_v;
  __shared__ double rescored[kTop8Candidates];

  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const unsigned warp = threadIdx.x >> 5;
  const unsigned lane = threadIdx.x & 31;

  float my_max_v2 = -INFINITY;
  for (unsigned b = threadIdx.x; b < blocks_per_row; b += kTop8Threads) {
    const Top2WsEntry e = ws[row * blocks_per_row + b];
    winner_v[b] = e.v1;
    winner_i[b] = e.i1;
    my_max_v2 = fmaxf(my_max_v2, e.v2);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    my_max_v2 = fmaxf(my_max_v2, __shfl_down_sync(0xffffffffu, my_max_v2, offset));
  }
  if (lane == 0) {
    warp_max_v2[warp] = my_max_v2;
  }
  __syncthreads();

  if (warp == 0) {
    // 9 extraction passes over <=240 winners, lane-strided + shuffle
    // reduce; the winning slot is cleared between passes.
    for (int k = 0; k <= kTop8Candidates; ++k) {
      float best = -INFINITY;
      unsigned best_slot = 0xffffffffu;
      for (unsigned b = lane; b < blocks_per_row; b += 32u) {
        const float v = winner_v[b];
        // Deterministic tie-break toward the smaller token id.
        if (v > best ||
            (v == best && best_slot != 0xffffffffu &&
             winner_i[b] < winner_i[best_slot])) {
          best = v;
          best_slot = b;
        }
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, best, offset);
        const unsigned os = __shfl_down_sync(0xffffffffu, best_slot, offset);
        if (ov > best ||
            (ov == best && os != 0xffffffffu &&
             (best_slot == 0xffffffffu || winner_i[os] < winner_i[best_slot]))) {
          best = ov;
          best_slot = os;
        }
      }
      best_slot = __shfl_sync(0xffffffffu, best_slot, 0);
      best = __shfl_sync(0xffffffffu, best, 0);
      if (lane == 0) {
        if (k < kTop8Candidates) {
          cand_v[k] = best;
          cand_i[k] = best_slot == 0xffffffffu ? 0xffffffffu
                                               : winner_i[best_slot];
        } else {
          ninth_v = best;
        }
      }
      if (lane == 0 && best_slot != 0xffffffffu && k < kTop8Candidates) {
        winner_v[best_slot] = -INFINITY;
      }
      __syncwarp();
    }
  }
  __syncthreads();

  // Phase 3: warp w rescores candidate w (skip replicated/empty slots).
  if (warp < kTop8Candidates) {
    const uint32_t cand = cand_i[warp];
    double sum = 0.0;
    if (cand != 0xffffffffu && static_cast<size_t>(cand) < vocab) {
      const __nv_bfloat16 *w_row = weight + static_cast<size_t>(cand) * cols;
      const __nv_bfloat16 *x_row = input + row * cols;
      for (size_t c = lane; c < cols; c += 32u) {
        sum += static_cast<double>(__bfloat162float(w_row[c])) *
               static_cast<double>(__bfloat162float(x_row[c]));
      }
    } else {
      sum = -INFINITY;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
      rescored[warp] = cand == 0xffffffffu ? -INFINITY : sum;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    double best = -INFINITY;
    uint32_t best_token = 0;
    for (int k = 0; k < kTop8Candidates; ++k) {
      const double v = rescored[k];
      const uint32_t tok = cand_i[k];
      if (tok != 0xffffffffu &&
          (v > best || (v == best && tok < best_token))) {
        best = v;
        best_token = tok;
      }
    }
    float max_v2_all = -INFINITY;
    const unsigned n_warps = kTop8Threads / 32;
    for (unsigned w = 0; w < n_warps; ++w) {
      max_v2_all = fmaxf(max_v2_all, warp_max_v2[w]);
    }
    const float bound = fmaxf(ninth_v, max_v2_all);
    const bool guard_ok =
        best >= static_cast<double>(bound) + static_cast<double>(eps);
    tokens[row] = best_token;
    flags[row] = guard_ok ? 1u : 0u;
    if (mirror_last != nullptr && row + 1 == rows) {
      *mirror_last = best_token;
    }
    if (!guard_ok && fallback_count != nullptr) {
      atomicAdd(fallback_count, 1u);
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

extern "C" int
qwen36_lm_head_top2_margin(const qwen36_lm_head_top2_margin_spec_t *spec) {
  if (spec == nullptr || spec->rows == 0 || spec->vocab == 0 ||
      spec->logits_bf16.ptr == 0 || spec->tokens_u32.ptr == 0 ||
      spec->flags_u32.ptr == 0 || spec->workspace.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const size_t ws_needed = spec->rows * kTop2Blocks * sizeof(Top2WsEntry);
  if (spec->workspace_bytes < ws_needed) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = qwen36_internal_active_stream();
  const dim3 grid1(kTop2Blocks, static_cast<unsigned>(spec->rows));
  lm_head_top2_pass1_kernel<<<grid1, kTop2Threads, 0, stream>>>(
      ptr<const __nv_bfloat16>(spec->logits_bf16),
      ptr<Top2WsEntry>(spec->workspace), spec->rows, spec->vocab);
  if (cudaGetLastError() != cudaSuccess) {
    return QWEN36_STATUS_CUDA_ERROR;
  }
  lm_head_top2_pass2_kernel<<<static_cast<unsigned>(spec->rows), kTop2Threads,
                              0, stream>>>(
      ptr<const Top2WsEntry>(spec->workspace), ptr<uint32_t>(spec->tokens_u32),
      ptr<uint32_t>(spec->flags_u32),
      ptr<uint32_t>(spec->mirror_last_token_u32),
      ptr<uint32_t>(spec->fallback_count_u32), spec->rows, kTop2Blocks,
      spec->eps);
  return cudaGetLastError() == cudaSuccess ? QWEN36_STATUS_SUCCESS
                                           : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int
qwen36_lm_head_top8_rescore(const qwen36_lm_head_top8_rescore_spec_t *spec) {
  if (spec == nullptr || spec->rows == 0 || spec->vocab == 0 ||
      spec->cols == 0 || spec->logits_bf16.ptr == 0 ||
      spec->weight_bf16.ptr == 0 || spec->input_bf16.ptr == 0 ||
      spec->tokens_u32.ptr == 0 || spec->flags_u32.ptr == 0 ||
      spec->workspace.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const size_t ws_needed = spec->rows * kTop2Blocks * sizeof(Top2WsEntry);
  if (spec->workspace_bytes < ws_needed) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = qwen36_internal_active_stream();
  const dim3 grid1(kTop2Blocks, static_cast<unsigned>(spec->rows));
  lm_head_top2_pass1_kernel<<<grid1, kTop2Threads, 0, stream>>>(
      ptr<const __nv_bfloat16>(spec->logits_bf16),
      ptr<Top2WsEntry>(spec->workspace), spec->rows, spec->vocab);
  if (cudaGetLastError() != cudaSuccess) {
    return QWEN36_STATUS_CUDA_ERROR;
  }
  lm_head_top8_rescore_kernel<<<static_cast<unsigned>(spec->rows),
                                kTop8Threads, 0, stream>>>(
      ptr<const Top2WsEntry>(spec->workspace),
      ptr<const __nv_bfloat16>(spec->weight_bf16),
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<uint32_t>(spec->tokens_u32), ptr<uint32_t>(spec->flags_u32),
      ptr<uint32_t>(spec->mirror_last_token_u32),
      ptr<uint32_t>(spec->fallback_count_u32), spec->rows, spec->vocab,
      spec->cols, kTop2Blocks, spec->eps);
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
