// Register-tiled decode attention (q=1, split-KV) — the long-context fix.
//
// Replaces the serial inner loop of `attention_decode_split_gqa_kernel`
// (attention.cu:1225) for the BF16/FP8 KV, head_dim=256 hot shape. The v1
// kernel walks its split's timesteps one by one: per KV position it does a
// 1-byte-per-thread load, q_per_kv block-wide shuffle reductions, a
// block-wide online-softmax update and ~3 __syncthreads — measured 28×
// off the KV bandwidth floor at 24K ctx, and the entire source of the
// MTP=0 long-context slide (49.7 → 32.3 tok/s from 128 → 24K ctx; see
// docs/superpowers/notes/2026-06-09-decode-longctx-investigation.md).
//
// v2 structure (this file):
//   * one warp per timestep: 8 warps process a tile of 8 KV positions
//     concurrently instead of serially;
//   * vectorized loads: each lane owns 8 contiguous dims of the 256-dim
//     vector — one 8-byte load (FP8) or one 16-byte load (BF16) per
//     vector instead of one byte/element;
//   * LUT FP8 decode: a 256-entry SMEM table replaces the branchy
//     ldexpf-based decode_e4m3 per element;
//   * tile-batched online softmax: ONE accumulator rescale per 8
//     timesteps (standard FlashAttention tile reformulation) instead of
//     one per timestep;
//   * 2 __syncthreads per 8 timesteps instead of ~24.
//
// Everything outside the inner loop is kept bit-compatible with v1: the
// (kv_heads × n_splits) grid, the partials layout consumed by
// attention_decode_reduce_kernel, the device-side position read (graph
// capture), the k_new/v_new handling at t == position, and the cache
// append side effect (the owning split stores the current token's K/V
// into the cache, FP8-encoded with the byte-identical encoder).
//
// Out of scope (falls back to v1 in the dispatch): TurboQuant cache
// dtypes, head_dim != 256, q_per_kv > 8.

#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kTdHeadDim = 256;
constexpr int kTdThreads = 256;
constexpr int kTdTileT = 8; // timesteps per tile == warps per block (256/32)
constexpr int kTdMaxQPerKv = 8;
constexpr int kTdDimsPerLane = kTdHeadDim / 32; // 8

constexpr int kTdKvCacheBf16 = 0;
constexpr int kTdKvCacheFp8 = 1;

// Byte-identical to attention.cu decode_e4m3 (the LUT is built from this).
__device__ float td_decode_e4m3(uint8_t code) {
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

// Byte-identical to attention.cu encode_e4m3 (cache-append side effect).
__device__ uint8_t td_encode_e4m3(float value) {
  if (value == 0.0f || !isfinite(value)) {
    return 0;
  }
  const bool negative = value < 0.0f;
  float abs_value = fabsf(value);
  if (abs_value >= 448.0f) {
    return static_cast<uint8_t>((negative ? 0x80 : 0x00) | 0x7e);
  }

  constexpr float kMinNormal = 0x1p-6f;
  constexpr float kSubnormalStep = 0x1p-9f;
  constexpr float kNormalBoundary = (7.0f * kSubnormalStep + kMinNormal) * 0.5f;
  uint8_t code = 0;
  if (abs_value < kMinNormal) {
    if (abs_value >= kNormalBoundary) {
      code = 0x08;
    } else {
      int mantissa = static_cast<int>(floorf(abs_value / kSubnormalStep + 0.5f));
      if (mantissa <= 0) {
        code = 0;
      } else {
        code = static_cast<uint8_t>(mantissa);
      }
    }
  } else {
    int exponent = 0;
    float frac = frexpf(abs_value, &exponent);
    const int exponent_field = exponent + 6;
    float mantissa_f = (frac * 2.0f - 1.0f) * 8.0f;
    int mantissa = static_cast<int>(floorf(mantissa_f + 0.5f));
    int adjusted_exponent = exponent_field;
    if (mantissa >= 8) {
      mantissa = 0;
      adjusted_exponent += 1;
    }
    if (adjusted_exponent >= 15) {
      adjusted_exponent = 15;
      mantissa = min(mantissa, 6);
    }
    code = static_cast<uint8_t>((adjusted_exponent << 3) | mantissa);
  }
  return static_cast<uint8_t>((negative ? 0x80 : 0x00) | code);
}

// Load this lane's 8 contiguous dims of one K/V vector into `out[8]`.
// `t == position` reads the not-yet-cached current token from new_bf16.
__device__ __forceinline__ void
td_load_lane_chunk(float out[kTdDimsPerLane], const void *cache,
                   const __nv_bfloat16 *new_bf16, int kv_cache_dtype,
                   const float *fp8_lut, size_t t, size_t position,
                   size_t kvh, size_t kv_heads, unsigned lane) {
  const size_t dim0 = static_cast<size_t>(lane) * kTdDimsPerLane;
  if (t == position) {
    // 8 BF16 = 16 bytes, base offset kvh*256 elems = 512B-aligned.
    const uint4 raw = *reinterpret_cast<const uint4 *>(
        new_bf16 + kvh * kTdHeadDim + dim0);
    const __nv_bfloat16 *h = reinterpret_cast<const __nv_bfloat16 *>(&raw);
#pragma unroll
    for (int j = 0; j < kTdDimsPerLane; ++j) {
      out[j] = __bfloat162float(h[j]);
    }
    return;
  }
  const size_t vec = t * kv_heads + kvh;
  if (kv_cache_dtype == kTdKvCacheFp8) {
    // 8 FP8 codes = 8 bytes, base offset vec*256 bytes = 256B-aligned.
    const uint2 raw = *reinterpret_cast<const uint2 *>(
        reinterpret_cast<const uint8_t *>(cache) + vec * kTdHeadDim + dim0);
    const uint8_t *b = reinterpret_cast<const uint8_t *>(&raw);
#pragma unroll
    for (int j = 0; j < kTdDimsPerLane; ++j) {
      out[j] = fp8_lut[b[j]];
    }
    return;
  }
  const uint4 raw = *reinterpret_cast<const uint4 *>(
      reinterpret_cast<const __nv_bfloat16 *>(cache) +
      (vec * kTdHeadDim + dim0));
  const __nv_bfloat16 *h = reinterpret_cast<const __nv_bfloat16 *>(&raw);
#pragma unroll
  for (int j = 0; j < kTdDimsPerLane; ++j) {
    out[j] = __bfloat162float(h[j]);
  }
}

__global__ void __launch_bounds__(kTdThreads)
    attention_decode_split_tiled_kernel(
        const __nv_bfloat16 *__restrict__ q,
        const __nv_bfloat16 *__restrict__ k_new,
        const __nv_bfloat16 *__restrict__ v_new, void *cache_k, void *cache_v,
        int kv_cache_dtype, float *__restrict__ partial_acc,
        float *__restrict__ partial_max, float *__restrict__ partial_denom,
        size_t position_scalar, const int32_t *__restrict__ position_device,
        qwen36_attention_shape_t shape, int n_splits,
        int split_timesteps_per_block) {
  __shared__ float q_smem[kTdMaxQPerKv * kTdHeadDim];
  __shared__ float v_tile[kTdTileT][kTdHeadDim];
  __shared__ float scores[kTdTileT][kTdMaxQPerKv];
  __shared__ float p_tile[kTdTileT][kTdMaxQPerKv];
  __shared__ float fp8_lut[256];
  __shared__ float m_state[kTdMaxQPerKv];
  __shared__ float l_state[kTdMaxQPerKv];
  __shared__ float tile_scale[kTdMaxQPerKv];
  __shared__ size_t shared_position;

  const size_t kvh = blockIdx.x;
  const size_t split = blockIdx.y;
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const float qk_scale = rsqrtf(static_cast<float>(kTdHeadDim));
  const unsigned tid = threadIdx.x;
  const unsigned warp = tid >> 5;
  const unsigned lane = tid & 31;

  if (tid == 0) {
    shared_position = position_device != nullptr
                          ? static_cast<size_t>(*position_device)
                          : position_scalar;
  }
  // LUT covers both K and V decodes; build unconditionally (1 op/thread).
  fp8_lut[tid] = td_decode_e4m3(static_cast<uint8_t>(tid));
  if (tid < q_per_kv) {
    m_state[tid] = -INFINITY;
    l_state[tid] = 0.0f;
  }
  __syncthreads();

  const size_t position = shared_position;
  const size_t split_block = static_cast<size_t>(split_timesteps_per_block);
  const size_t t_start = split * split_block;
  size_t t_end = t_start + split_block;
  if (t_end > position + 1) {
    t_end = position + 1;
  }

  if (t_start >= position + 1) {
    // Empty split: publish neutral partials (reduce skips via scale=0).
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const size_t qh = kvh * q_per_kv + qh_local;
      partial_acc[(qh * n_splits + split) * kTdHeadDim + tid] = 0.0f;
    }
    if (tid < q_per_kv) {
      const size_t qh = kvh * q_per_kv + tid;
      partial_max[qh * n_splits + split] = -INFINITY;
      partial_denom[qh * n_splits + split] = 0.0f;
    }
    return;
  }

  // Q for this kv-head's q-head group, staged once.
  for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
    const size_t qh = kvh * q_per_kv + qh_local;
    q_smem[qh_local * kTdHeadDim + tid] =
        __bfloat162float(q[qh * kTdHeadDim + tid]);
  }
  __syncthreads();

  float acc[kTdMaxQPerKv];
#pragma unroll
  for (int i = 0; i < kTdMaxQPerKv; ++i) {
    acc[i] = 0.0f;
  }

  for (size_t tile_base = t_start; tile_base < t_end;
       tile_base += kTdTileT) {
    const size_t t = tile_base + warp;
    const bool active = t < t_end;

    // ---- per-warp: K dot for this warp's timestep + stage V ----
    if (active) {
      float k_chunk[kTdDimsPerLane];
      td_load_lane_chunk(k_chunk, cache_k, k_new, kv_cache_dtype, fp8_lut, t,
                         position, kvh, shape.kv_heads, lane);
      const float *qbase = q_smem + static_cast<size_t>(lane) * kTdDimsPerLane;
      for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
        const float *qv = qbase + qh_local * kTdHeadDim;
        float partial = 0.0f;
#pragma unroll
        for (int j = 0; j < kTdDimsPerLane; ++j) {
          partial += qv[j] * k_chunk[j];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          partial += __shfl_xor_sync(0xffffffffu, partial, offset);
        }
        if (lane == 0) {
          scores[warp][qh_local] = partial * qk_scale;
        }
      }
      float v_chunk[kTdDimsPerLane];
      td_load_lane_chunk(v_chunk, cache_v, v_new, kv_cache_dtype, fp8_lut, t,
                         position, kvh, shape.kv_heads, lane);
#pragma unroll
      for (int j = 0; j < kTdDimsPerLane; ++j) {
        v_tile[warp][lane * kTdDimsPerLane + j] = v_chunk[j];
      }
    } else {
      // Keep masked rows finite: p == 0 must yield exactly 0 contribution.
      if (lane < q_per_kv) {
        scores[warp][lane] = -INFINITY;
      }
#pragma unroll
      for (int j = 0; j < kTdDimsPerLane; ++j) {
        v_tile[warp][lane * kTdDimsPerLane + j] = 0.0f;
      }
    }
    __syncthreads();

    // ---- tile-batched online softmax (one thread per q-head) ----
    if (tid < q_per_kv) {
      float tile_max = -INFINITY;
#pragma unroll
      for (int t8 = 0; t8 < kTdTileT; ++t8) {
        tile_max = fmaxf(tile_max, scores[t8][tid]);
      }
      const float m_old = m_state[tid];
      const float m_new = fmaxf(m_old, tile_max);
      const bool all_masked = isinf(m_new) && m_new < 0.0f;
      const float scale =
          (isinf(m_old) && m_old < 0.0f) ? 0.0f : expf(m_old - m_new);
      float sum = 0.0f;
#pragma unroll
      for (int t8 = 0; t8 < kTdTileT; ++t8) {
        const float s = scores[t8][tid];
        const float p =
            (all_masked || (isinf(s) && s < 0.0f)) ? 0.0f : expf(s - m_new);
        p_tile[t8][tid] = p;
        sum += p;
      }
      tile_scale[tid] = scale;
      l_state[tid] = l_state[tid] * scale + sum;
      m_state[tid] = m_new;
    }
    __syncthreads();

    // ---- accumulate (each thread owns dim d == tid) ----
#pragma unroll
    for (int qh_local = 0; qh_local < kTdMaxQPerKv; ++qh_local) {
      if (qh_local >= static_cast<int>(q_per_kv)) {
        break;
      }
      acc[qh_local] *= tile_scale[qh_local];
    }
#pragma unroll
    for (int t8 = 0; t8 < kTdTileT; ++t8) {
      const float v = v_tile[t8][tid];
#pragma unroll
      for (int qh_local = 0; qh_local < kTdMaxQPerKv; ++qh_local) {
        if (qh_local >= static_cast<int>(q_per_kv)) {
          break;
        }
        acc[qh_local] += p_tile[t8][qh_local] * v;
      }
    }
    __syncthreads();
  }

  // ---- publish partials (same layout as v1 / the shared reduce) ----
  for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
    const size_t qh = kvh * q_per_kv + qh_local;
    partial_acc[(qh * n_splits + split) * kTdHeadDim + tid] = acc[qh_local];
  }
  if (tid < q_per_kv) {
    const size_t qh = kvh * q_per_kv + tid;
    partial_max[qh * n_splits + split] = m_state[tid];
    partial_denom[qh * n_splits + split] = l_state[tid];
  }

  // ---- cache append side effect (owning split only), mirrors v1 ----
  if (t_start <= position && position < t_end) {
    const size_t cache_off =
        (position * shape.kv_heads + kvh) * kTdHeadDim + tid;
    const __nv_bfloat16 kv = k_new[kvh * kTdHeadDim + tid];
    const __nv_bfloat16 vv = v_new[kvh * kTdHeadDim + tid];
    if (kv_cache_dtype == kTdKvCacheFp8) {
      reinterpret_cast<uint8_t *>(cache_k)[cache_off] =
          td_encode_e4m3(__bfloat162float(kv));
      reinterpret_cast<uint8_t *>(cache_v)[cache_off] =
          td_encode_e4m3(__bfloat162float(vv));
    } else {
      reinterpret_cast<__nv_bfloat16 *>(cache_k)[cache_off] = kv;
      reinterpret_cast<__nv_bfloat16 *>(cache_v)[cache_off] = vv;
    }
  }
}

template <typename T> T *td_ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

} // namespace

// Launches the tiled split kernel with the same grid/partials contract as
// attention_decode_split_gqa_kernel. The caller (attention.cu dispatch)
// guarantees: BF16|FP8 cache, head_dim==256, 1 < q_per_kv <= 8, n_splits
// from the engine, split_timesteps_per_block % 8 == 0. The shared
// attention_decode_reduce_kernel consumes the partials afterwards.
extern "C" int qwen36_attention_decode_split_tiled(
    const qwen36_attention_decode_spec_t *spec, int n_splits,
    int split_timesteps_per_block) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->shape.head_dim != kTdHeadDim ||
      (spec->kv_cache_dtype != kTdKvCacheBf16 &&
       spec->kv_cache_dtype != kTdKvCacheFp8) ||
      spec->shape.kv_heads == 0 ||
      spec->shape.q_heads % spec->shape.kv_heads != 0 ||
      spec->shape.q_heads / spec->shape.kv_heads > kTdMaxQPerKv ||
      spec->shape.q_heads / spec->shape.kv_heads < 2 || n_splits < 1 ||
      split_timesteps_per_block < kTdTileT ||
      split_timesteps_per_block % kTdTileT != 0) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  const dim3 grid(static_cast<unsigned int>(spec->shape.kv_heads),
                  static_cast<unsigned int>(n_splits));
  attention_decode_split_tiled_kernel<<<grid, kTdThreads, 0,
                                        qwen36_internal_active_stream()>>>(
      td_ptr<const __nv_bfloat16>(spec->q_bf16),
      td_ptr<const __nv_bfloat16>(spec->k_bf16),
      td_ptr<const __nv_bfloat16>(spec->v_bf16),
      td_ptr<void>(spec->kv_cache_k), td_ptr<void>(spec->kv_cache_v),
      spec->kv_cache_dtype, td_ptr<float>(spec->partial_acc_f32),
      td_ptr<float>(spec->partial_max_f32),
      td_ptr<float>(spec->partial_denom_f32), spec->position,
      td_ptr<const int32_t>(spec->position_device_i32), spec->shape, n_splits,
      split_timesteps_per_block);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
