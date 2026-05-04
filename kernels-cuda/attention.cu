#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

template <typename T> T *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

constexpr int kKvCacheBf16 = 0;
constexpr int kKvCacheFp8 = 1;

__device__ float decode_e4m3(uint8_t code) {
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

__device__ uint8_t encode_e4m3(float value) {
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
    // frexp returns abs_value = frac * 2^exponent with frac in [0.5, 1).
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

__device__ __forceinline__ float load_cache_value(const void *cache,
                                                  int kv_cache_dtype,
                                                  size_t index) {
  if (kv_cache_dtype == kKvCacheFp8) {
    return decode_e4m3(reinterpret_cast<const uint8_t *>(cache)[index]);
  }
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(cache)[index]);
}

__device__ __forceinline__ void store_cache_value(void *cache,
                                                  int kv_cache_dtype,
                                                  size_t index,
                                                  __nv_bfloat16 value) {
  if (kv_cache_dtype == kKvCacheFp8) {
    reinterpret_cast<uint8_t *>(cache)[index] =
        encode_e4m3(__bfloat162float(value));
  } else {
    reinterpret_cast<__nv_bfloat16 *>(cache)[index] = value;
  }
}

__global__ void copy_kv_prefill_kernel(
    const __nv_bfloat16 *k, const __nv_bfloat16 *v, void *cache_k,
    void *cache_v, int kv_cache_dtype, size_t start_position, size_t tokens,
    const int32_t *start_position_device, size_t kv_heads, size_t head_dim) {
  __shared__ size_t shared_start_position;
  const size_t kvh = blockIdx.x;
  const size_t token = blockIdx.y;
  if (token >= tokens) {
    return;
  }
  if (threadIdx.x == 0) {
    shared_start_position = start_position_device != nullptr
                                ? static_cast<size_t>(*start_position_device)
                                : start_position;
  }
  __syncthreads();
  for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const size_t src = (token * kv_heads + kvh) * head_dim + d;
    const size_t dst =
        ((shared_start_position + token) * kv_heads + kvh) * head_dim + d;
    store_cache_value(cache_k, kv_cache_dtype, dst, k[src]);
    store_cache_value(cache_v, kv_cache_dtype, dst, v[src]);
  }
}

// Decode-time attention with online softmax.
// Optimisations vs. the naive reference: Q is preloaded once into a register,
// the per-timestep QK reduction uses warp shuffles + a single shared-memory
// fan-in across at most 8 warps, and the new-token K/V write into the cache is
// fused into this kernel (one block per kv-group performs the store), letting
// callers skip the separate copy_kv launch. The current decode position is
// read from `position_device` when non-null so a CUDA-Graph capture can be
// reused across decode steps without re-recording.
__global__ void attention_decode_kernel(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k_new,
    const __nv_bfloat16 *v_new, void *cache_k, void *cache_v,
    int kv_cache_dtype,
    __nv_bfloat16 *output, size_t position_scalar,
    const int32_t *position_device, qwen36_attention_shape_t shape) {
  __shared__ float warp_sums[8];
  __shared__ float score_share;
  __shared__ size_t shared_position;

  const size_t qh = blockIdx.x;
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t kvh = qh / q_per_kv;
  const float scale = rsqrtf(static_cast<float>(shape.head_dim));
  const size_t d = threadIdx.x;
  const bool active = d < shape.head_dim;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31) >> 5;

  if (threadIdx.x == 0) {
    shared_position = position_device != nullptr
                          ? static_cast<size_t>(*position_device)
                          : position_scalar;
  }
  __syncthreads();
  const size_t position = shared_position;

  const float q_val =
      active ? __bfloat162float(q[qh * shape.head_dim + d]) : 0.0f;

  // The new K/V for t == position are not yet in the cache; read them from the
  // input pointers and let block 0 of each kv-group write them back at the end
  // of the kernel.
  const float k_new_val =
      active ? __bfloat162float(k_new[kvh * shape.head_dim + d]) : 0.0f;
  const float v_new_val =
      active ? __bfloat162float(v_new[kvh * shape.head_dim + d]) : 0.0f;

  float acc = 0.0f;
  float max_score = -INFINITY;
  float denom = 0.0f;

  for (size_t t = 0; t <= position; ++t) {
    const bool is_new = (t == position);
    float local = 0.0f;
    if (active) {
      const float kv = is_new ? k_new_val
                              : load_cache_value(
                                    cache_k, kv_cache_dtype,
                                    (t * shape.kv_heads + kvh) *
                                            shape.head_dim +
                                        d);
      local = q_val * kv;
    }

    // Warp-level reduction.
    for (int offset = 16; offset > 0; offset >>= 1) {
      local += __shfl_xor_sync(0xffffffff, local, offset);
    }
    if (lane_id == 0) {
      warp_sums[warp_id] = local;
    }
    __syncthreads();
    if (warp_id == 0) {
      float total = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffff, total, offset);
      }
      if (lane_id == 0) {
        score_share = total * scale;
      }
    }
    __syncthreads();
    const float score = score_share;

    const float new_max = fmaxf(max_score, score);
    const float old_scale =
        isinf(max_score) && max_score < 0.0f ? 0.0f : expf(max_score - new_max);
    const float score_scale = expf(score - new_max);
    if (active) {
      const float vv = is_new ? v_new_val
                              : load_cache_value(
                                    cache_v, kv_cache_dtype,
                                    (t * shape.kv_heads + kvh) *
                                            shape.head_dim +
                                        d);
      acc = acc * old_scale + score_scale * vv;
    }
    denom = denom * old_scale + score_scale;
    max_score = new_max;
  }

  if (active) {
    output[qh * shape.head_dim + d] = __float2bfloat16(acc / denom);
    // Block 0 of each kv-group writes the new K/V back to the cache.
    if (qh % q_per_kv == 0) {
      const size_t cache_off =
          (position * shape.kv_heads + kvh) * shape.head_dim + d;
      store_cache_value(cache_k, kv_cache_dtype, cache_off,
                        k_new[kvh * shape.head_dim + d]);
      store_cache_value(cache_v, kv_cache_dtype, cache_off,
                        v_new[kvh * shape.head_dim + d]);
    }
  }
}

// Shared-memory bounds for the GQA-aware prefill kernel below. The current
// Qwen3.6 config (q_heads=24, kv_heads=4 -> q_per_kv=6, head_dim=256) sits
// comfortably under both bounds.
constexpr int kGqaMaxQPerKv = 8;
constexpr int kGqaMaxHeadDim = 256;
constexpr int kGqaMaxWarps = 8;

// Split-KV (FlashDecoding-style) kernels for batch=1 decode attention.
//
// On long contexts the per-q-head decode kernel runs 24 blocks sequentially
// across the timestep loop, leaving most of the 170 SMs on Blackwell idle.
// The split kernel partitions [0, position] into chunks of
// `kSplitTimestepsPerBlock` and assigns one block per (q_head, chunk),
// computing partial online-softmax outputs to scratch global memory. A
// follow-up reduction kernel combines the partials per q-head using the
// log-sum-exp identity.
//
// Runtime scratch is sized on the Rust side for `kMinSplitTimestepsPerBlock`;
// individual calls can pass a larger tile size to reduce launch/reduce
// overhead at medium contexts.
constexpr int kDefaultSplitTimestepsPerBlock = 512;
constexpr int kMinSplitTimestepsPerBlock = 64;

__global__ void attention_decode_split_kernel(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k_new,
    const __nv_bfloat16 *v_new, void *cache_k, void *cache_v,
    int kv_cache_dtype,
    float *partial_acc, float *partial_max, float *partial_denom,
    size_t position_scalar, const int32_t *position_device,
    qwen36_attention_shape_t shape, int n_splits,
    int split_timesteps_per_block) {
  __shared__ float warp_sums[8];
  __shared__ float score_share;
  __shared__ size_t shared_position;

  const size_t qh = blockIdx.x;
  const size_t split = blockIdx.y;
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t kvh = qh / q_per_kv;
  const size_t head_dim = shape.head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const size_t d = threadIdx.x;
  const bool active = d < head_dim;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31u) >> 5;

  if (threadIdx.x == 0) {
    shared_position = position_device != nullptr
                          ? static_cast<size_t>(*position_device)
                          : position_scalar;
  }
  __syncthreads();
  const size_t position = shared_position;
  const size_t split_block =
      static_cast<size_t>(split_timesteps_per_block);
  const size_t t_start = split * split_block;
  size_t t_end = t_start + split_block;
  if (t_end > position + 1) {
    t_end = position + 1;
  }

  if (t_start >= position + 1) {
    // Empty split (caller pads n_splits to the worst case). Write softmax
    // identity values so the reduce kernel can multiply through unconditionally.
    if (active) {
      partial_acc[(qh * n_splits + split) * head_dim + d] = 0.0f;
    }
    if (threadIdx.x == 0) {
      partial_max[qh * n_splits + split] = -INFINITY;
      partial_denom[qh * n_splits + split] = 0.0f;
    }
    return;
  }

  const float q_val =
      active ? __bfloat162float(q[qh * head_dim + d]) : 0.0f;
  const float k_new_val =
      active ? __bfloat162float(k_new[kvh * head_dim + d]) : 0.0f;
  const float v_new_val =
      active ? __bfloat162float(v_new[kvh * head_dim + d]) : 0.0f;

  float acc = 0.0f;
  float max_score = -INFINITY;
  float denom = 0.0f;

  for (size_t t = t_start; t < t_end; ++t) {
    const bool is_new = (t == position);
    float local = 0.0f;
    if (active) {
      const float kv = is_new ? k_new_val
                              : load_cache_value(
                                    cache_k, kv_cache_dtype,
                                    (t * shape.kv_heads + kvh) *
                                            shape.head_dim +
                                        d);
      local = q_val * kv;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      local += __shfl_xor_sync(0xffffffff, local, offset);
    }
    if (lane_id == 0) {
      warp_sums[warp_id] = local;
    }
    __syncthreads();
    if (warp_id == 0) {
      float total = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffff, total, offset);
      }
      if (lane_id == 0) {
        score_share = total * scale;
      }
    }
    __syncthreads();
    const float score = score_share;

    const float new_max = fmaxf(max_score, score);
    const float old_scale =
        isinf(max_score) && max_score < 0.0f ? 0.0f : expf(max_score - new_max);
    const float score_scale = expf(score - new_max);
    if (active) {
      const float vv = is_new ? v_new_val
                              : load_cache_value(
                                    cache_v, kv_cache_dtype,
                                    (t * shape.kv_heads + kvh) *
                                            shape.head_dim +
                                        d);
      acc = acc * old_scale + score_scale * vv;
    }
    denom = denom * old_scale + score_scale;
    max_score = new_max;
  }

  if (active) {
    partial_acc[(qh * n_splits + split) * head_dim + d] = acc;
  }
  if (threadIdx.x == 0) {
    partial_max[qh * n_splits + split] = max_score;
    partial_denom[qh * n_splits + split] = denom;
  }

  // The split that owns `position` writes the new K/V to the cache. Gate to
  // the first q-head of each kv-group so each cache row is written exactly
  // once even though every q-head in the group hits this branch.
  if (t_start <= position && position < t_end && active &&
      (qh % q_per_kv == 0)) {
    const size_t cache_off =
        (position * shape.kv_heads + kvh) * shape.head_dim + d;
    store_cache_value(cache_k, kv_cache_dtype, cache_off,
                      k_new[kvh * head_dim + d]);
    store_cache_value(cache_v, kv_cache_dtype, cache_off,
                      v_new[kvh * head_dim + d]);
  }
}

__global__ void attention_decode_split_gqa_kernel(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k_new,
    const __nv_bfloat16 *v_new, void *cache_k,
    void *cache_v, int kv_cache_dtype, float *partial_acc, float *partial_max,
    float *partial_denom, size_t position_scalar,
    const int32_t *position_device, qwen36_attention_shape_t shape,
    int n_splits, int split_timesteps_per_block) {
  __shared__ size_t shared_position;
  __shared__ float kv_sram[kGqaMaxHeadDim];
  __shared__ float warp_partials[kGqaMaxWarps][kGqaMaxQPerKv];
  __shared__ float max_score_sram[kGqaMaxQPerKv];
  __shared__ float denom_sram[kGqaMaxQPerKv];
  __shared__ float scale_old[kGqaMaxQPerKv];
  __shared__ float scale_new[kGqaMaxQPerKv];

  const size_t kvh = blockIdx.x;
  const size_t split = blockIdx.y;
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t head_dim = shape.head_dim;
  const float qk_scale = rsqrtf(static_cast<float>(head_dim));
  const bool tile_active = threadIdx.x < head_dim;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31u) >> 5;

  if (threadIdx.x == 0) {
    shared_position = position_device != nullptr
                          ? static_cast<size_t>(*position_device)
                          : position_scalar;
  }
  if (threadIdx.x < q_per_kv) {
    max_score_sram[threadIdx.x] = -INFINITY;
    denom_sram[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  const size_t position = shared_position;
  const size_t split_block =
      static_cast<size_t>(split_timesteps_per_block);
  const size_t t_start = split * split_block;
  size_t t_end = t_start + split_block;
  if (t_end > position + 1) {
    t_end = position + 1;
  }

  if (t_start >= position + 1) {
    if (tile_active) {
      for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
        const size_t qh = kvh * q_per_kv + qh_local;
        partial_acc[(qh * n_splits + split) * head_dim + threadIdx.x] = 0.0f;
      }
    }
    if (threadIdx.x < q_per_kv) {
      const size_t qh = kvh * q_per_kv + threadIdx.x;
      partial_max[qh * n_splits + split] = -INFINITY;
      partial_denom[qh * n_splits + split] = 0.0f;
    }
    return;
  }

  float q_local[kGqaMaxQPerKv];
#pragma unroll
  for (int i = 0; i < kGqaMaxQPerKv; ++i) {
    q_local[i] = 0.0f;
  }
  if (tile_active) {
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const size_t qh = kvh * q_per_kv + qh_local;
      q_local[qh_local] =
          __bfloat162float(q[qh * head_dim + threadIdx.x]);
    }
  }

  float acc[kGqaMaxQPerKv];
#pragma unroll
  for (int i = 0; i < kGqaMaxQPerKv; ++i) {
    acc[i] = 0.0f;
  }

  const float k_new_val =
      tile_active ? __bfloat162float(k_new[kvh * head_dim + threadIdx.x])
                  : 0.0f;
  const float v_new_val =
      tile_active ? __bfloat162float(v_new[kvh * head_dim + threadIdx.x])
                  : 0.0f;

  for (size_t t = t_start; t < t_end; ++t) {
    const bool is_new = (t == position);
    if (tile_active) {
      kv_sram[threadIdx.x] =
          is_new ? k_new_val
                 : load_cache_value(
                       cache_k, kv_cache_dtype,
                       (t * shape.kv_heads + kvh) * head_dim + threadIdx.x);
    }
    __syncthreads();

    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      float local =
          tile_active ? q_local[qh_local] * kv_sram[threadIdx.x] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_xor_sync(0xffffffff, local, offset);
      }
      if (lane_id == 0) {
        warp_partials[warp_id][qh_local] = local;
      }
    }
    __syncthreads();

    if (threadIdx.x < q_per_kv) {
      float total = 0.0f;
      for (unsigned w = 0; w < n_warps; ++w) {
        total += warp_partials[w][threadIdx.x];
      }
      const float score = total * qk_scale;
      const float old_max = max_score_sram[threadIdx.x];
      const float new_max = fmaxf(old_max, score);
      const float so =
          isinf(old_max) && old_max < 0.0f ? 0.0f : expf(old_max - new_max);
      const float sn = expf(score - new_max);
      scale_old[threadIdx.x] = so;
      scale_new[threadIdx.x] = sn;
      denom_sram[threadIdx.x] = denom_sram[threadIdx.x] * so + sn;
      max_score_sram[threadIdx.x] = new_max;
    }
    __syncthreads();

    if (tile_active) {
      kv_sram[threadIdx.x] =
          is_new ? v_new_val
                 : load_cache_value(
                       cache_v, kv_cache_dtype,
                       (t * shape.kv_heads + kvh) * head_dim + threadIdx.x);
    }
    __syncthreads();

    if (tile_active) {
      const float v_val = kv_sram[threadIdx.x];
#pragma unroll
      for (int qh_local = 0; qh_local < kGqaMaxQPerKv; ++qh_local) {
        if (qh_local >= static_cast<int>(q_per_kv)) {
          break;
        }
        acc[qh_local] = acc[qh_local] * scale_old[qh_local] +
                        scale_new[qh_local] * v_val;
      }
    }
    __syncthreads();
  }

  if (tile_active) {
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const size_t qh = kvh * q_per_kv + qh_local;
      partial_acc[(qh * n_splits + split) * head_dim + threadIdx.x] =
          acc[qh_local];
    }
  }
  if (threadIdx.x < q_per_kv) {
    const size_t qh = kvh * q_per_kv + threadIdx.x;
    partial_max[qh * n_splits + split] = max_score_sram[threadIdx.x];
    partial_denom[qh * n_splits + split] = denom_sram[threadIdx.x];
  }

  if (t_start <= position && position < t_end && tile_active) {
    const size_t cache_off =
        (position * shape.kv_heads + kvh) * head_dim + threadIdx.x;
    store_cache_value(cache_k, kv_cache_dtype, cache_off,
                      k_new[kvh * head_dim + threadIdx.x]);
    store_cache_value(cache_v, kv_cache_dtype, cache_off,
                      v_new[kvh * head_dim + threadIdx.x]);
  }
}

__global__ void attention_decode_reduce_kernel(
    const float *partial_acc, const float *partial_max,
    const float *partial_denom, __nv_bfloat16 *output,
    qwen36_attention_shape_t shape, int n_splits) {
  __shared__ float gmax;
  __shared__ float gdenom;

  const size_t qh = blockIdx.x;
  const size_t head_dim = shape.head_dim;
  const size_t d = threadIdx.x;

  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
      m = fmaxf(m, partial_max[qh * n_splits + s]);
    }
    float dn = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
      const float pm = partial_max[qh * n_splits + s];
      const float pd = partial_denom[qh * n_splits + s];
      const float scale =
          isinf(pm) && pm < 0.0f ? 0.0f : expf(pm - m);
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
    const float pm = partial_max[qh * n_splits + s];
    const float pa = partial_acc[(qh * n_splits + s) * head_dim + d];
    const float scale = isinf(pm) && pm < 0.0f ? 0.0f : expf(pm - m);
    acc_total += pa * scale;
  }
  output[qh * head_dim + d] = __float2bfloat16(acc_total / dn);
}

__global__ void attention_prefill_split_kernel(
    const __nv_bfloat16 *q, const void *cache_k,
    const void *cache_v, int kv_cache_dtype, float *partial_acc, float *partial_max,
    float *partial_denom, size_t start_position_scalar,
    const int32_t *start_position_device, size_t token,
    qwen36_attention_shape_t shape, int n_splits,
    int split_timesteps_per_block) {
  __shared__ float warp_sums[8];
  __shared__ float score_share;
  __shared__ size_t shared_start_position;

  const size_t qh = blockIdx.x;
  const size_t split = blockIdx.y;
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t kvh = qh / q_per_kv;
  const size_t head_dim = shape.head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const size_t d = threadIdx.x;
  const bool active = d < head_dim;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31u) >> 5;

  if (threadIdx.x == 0) {
    shared_start_position =
        start_position_device != nullptr
            ? static_cast<size_t>(*start_position_device)
            : start_position_scalar;
  }
  __syncthreads();
  const size_t position = shared_start_position + token;
  const size_t split_block =
      static_cast<size_t>(split_timesteps_per_block);
  const size_t t_start = split * split_block;
  size_t t_end = t_start + split_block;
  if (t_end > position + 1) {
    t_end = position + 1;
  }

  if (t_start >= position + 1) {
    if (active) {
      partial_acc[(qh * n_splits + split) * head_dim + d] = 0.0f;
    }
    if (threadIdx.x == 0) {
      partial_max[qh * n_splits + split] = -INFINITY;
      partial_denom[qh * n_splits + split] = 0.0f;
    }
    return;
  }

  const float q_val =
      active ? __bfloat162float(
                   q[(token * shape.q_heads + qh) * head_dim + d])
             : 0.0f;
  float acc = 0.0f;
  float max_score = -INFINITY;
  float denom = 0.0f;

  for (size_t t = t_start; t < t_end; ++t) {
    float local = 0.0f;
    if (active) {
      const float kv = load_cache_value(
          cache_k, kv_cache_dtype, (t * shape.kv_heads + kvh) * head_dim + d);
      local = q_val * kv;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      local += __shfl_xor_sync(0xffffffff, local, offset);
    }
    if (lane_id == 0) {
      warp_sums[warp_id] = local;
    }
    __syncthreads();
    if (warp_id == 0) {
      float total = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffff, total, offset);
      }
      if (lane_id == 0) {
        score_share = total * scale;
      }
    }
    __syncthreads();

    const float score = score_share;
    const float new_max = fmaxf(max_score, score);
    const float old_scale =
        isinf(max_score) && max_score < 0.0f ? 0.0f : expf(max_score - new_max);
    const float score_scale = expf(score - new_max);
    if (active) {
      const float vv = load_cache_value(
          cache_v, kv_cache_dtype, (t * shape.kv_heads + kvh) * head_dim + d);
      acc = acc * old_scale + score_scale * vv;
    }
    denom = denom * old_scale + score_scale;
    max_score = new_max;
  }

  if (active) {
    partial_acc[(qh * n_splits + split) * head_dim + d] = acc;
  }
  if (threadIdx.x == 0) {
    partial_max[qh * n_splits + split] = max_score;
    partial_denom[qh * n_splits + split] = denom;
  }
}

// GQA-aware prefill kernel. Same online-softmax structure as the per-q-head
// kernel below, but lays the grid out as (kv_heads, tokens) so the q_per_kv
// queries that share each kv-head also share the K/V cache loads. With
// q_per_kv = 6 on Qwen3.6 this cuts the BF16 cache traffic for the Q*K and
// V matmuls by ~6x (the L2 used to absorb the redundancy via the per-q-head
// kernel; this version eliminates it outright). The per-kv-head launch
// configuration only makes sense when the block count stays large — i.e.
// during prefill, where `tokens` is in the thousands. The decode call site
// keeps using the per-q-head kernel because batch=1 decode only has 4
// kv-heads to dispatch and would starve the GPU.
__global__ void attention_prefill_gqa_kernel(
    const __nv_bfloat16 *q, const void *cache_k,
    const void *cache_v, int kv_cache_dtype, __nv_bfloat16 *output,
    size_t start_position, const int32_t *start_position_device, size_t tokens,
    qwen36_attention_shape_t shape) {
  __shared__ size_t shared_start_position;
  __shared__ float kv_sram[kGqaMaxHeadDim];
  __shared__ float warp_partials[kGqaMaxWarps][kGqaMaxQPerKv];
  // Online-softmax state lives in shared memory so dim-threads can read it
  // when applying the per-step rescale to their accumulators. Only the
  // q-head's owning thread (threadIdx.x == qh_local) writes these.
  __shared__ float max_score_sram[kGqaMaxQPerKv];
  __shared__ float denom_sram[kGqaMaxQPerKv];
  __shared__ float scale_old[kGqaMaxQPerKv];
  __shared__ float scale_new[kGqaMaxQPerKv];

  const size_t kvh = blockIdx.x;
  const size_t token = blockIdx.y;
  if (token >= tokens) {
    return;
  }
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t head_dim = shape.head_dim;
  const float qk_scale = rsqrtf(static_cast<float>(head_dim));
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31u) >> 5;
  const bool tile_active = threadIdx.x < head_dim;

  if (threadIdx.x == 0) {
    shared_start_position = start_position_device != nullptr
                                ? static_cast<size_t>(*start_position_device)
                                : start_position;
  }
  if (threadIdx.x < q_per_kv) {
    max_score_sram[threadIdx.x] = -INFINITY;
    denom_sram[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  const size_t position = shared_start_position + token;

  float q_local[kGqaMaxQPerKv];
#pragma unroll
  for (int i = 0; i < kGqaMaxQPerKv; ++i) {
    q_local[i] = 0.0f;
  }
  if (tile_active) {
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const size_t qh = kvh * q_per_kv + qh_local;
      q_local[qh_local] = __bfloat162float(
          q[(token * shape.q_heads + qh) * head_dim + threadIdx.x]);
    }
  }

  float acc[kGqaMaxQPerKv];
#pragma unroll
  for (int i = 0; i < kGqaMaxQPerKv; ++i) {
    acc[i] = 0.0f;
  }

  for (size_t t = 0; t <= position; ++t) {
    if (tile_active) {
      kv_sram[threadIdx.x] = load_cache_value(
          cache_k, kv_cache_dtype,
          (t * shape.kv_heads + kvh) * head_dim + threadIdx.x);
    }
    __syncthreads();

    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      float local =
          tile_active ? q_local[qh_local] * kv_sram[threadIdx.x] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_xor_sync(0xffffffff, local, offset);
      }
      if (lane_id == 0) {
        warp_partials[warp_id][qh_local] = local;
      }
    }
    __syncthreads();

    if (threadIdx.x < q_per_kv) {
      float total = 0.0f;
      for (unsigned w = 0; w < n_warps; ++w) {
        total += warp_partials[w][threadIdx.x];
      }
      const float score = total * qk_scale;
      const float old_max = max_score_sram[threadIdx.x];
      const float new_max = fmaxf(old_max, score);
      const float so =
          isinf(old_max) && old_max < 0.0f ? 0.0f : expf(old_max - new_max);
      const float sn = expf(score - new_max);
      scale_old[threadIdx.x] = so;
      scale_new[threadIdx.x] = sn;
      denom_sram[threadIdx.x] = denom_sram[threadIdx.x] * so + sn;
      max_score_sram[threadIdx.x] = new_max;
    }
    __syncthreads();

    if (tile_active) {
      kv_sram[threadIdx.x] = load_cache_value(
          cache_v, kv_cache_dtype,
          (t * shape.kv_heads + kvh) * head_dim + threadIdx.x);
    }
    __syncthreads();

    if (tile_active) {
      const float v_val = kv_sram[threadIdx.x];
#pragma unroll
      for (int qh_local = 0; qh_local < kGqaMaxQPerKv; ++qh_local) {
        if (qh_local >= static_cast<int>(q_per_kv)) {
          break;
        }
        acc[qh_local] = acc[qh_local] * scale_old[qh_local] +
                        scale_new[qh_local] * v_val;
      }
    }
    __syncthreads();
  }

  if (tile_active) {
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const float dn = denom_sram[qh_local];
      const size_t qh = kvh * q_per_kv + qh_local;
      output[(token * shape.q_heads + qh) * head_dim + threadIdx.x] =
          __float2bfloat16(acc[qh_local] / dn);
    }
  }
}

__global__ void attention_prefill_kernel(
    const __nv_bfloat16 *q, const void *cache_k,
    const void *cache_v, int kv_cache_dtype, __nv_bfloat16 *output,
    size_t start_position, const int32_t *start_position_device, size_t tokens,
    qwen36_attention_shape_t shape,
    const uint64_t *tree_ancestor_bitmap_u64,    // NULL = causal
    size_t verify_chunk_rows                      // 0 = causal
) {
  __shared__ float warp_sums[8];
  __shared__ float score_share;
  __shared__ size_t shared_start_position;

  const size_t qh = blockIdx.x;
  const size_t token = blockIdx.y;
  if (token >= tokens) {
    return;
  }
  if (threadIdx.x == 0) {
    shared_start_position = start_position_device != nullptr
                                ? static_cast<size_t>(*start_position_device)
                                : start_position;
  }
  __syncthreads();
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t kvh = qh / q_per_kv;
  const size_t position = shared_start_position + token;
  const float scale = rsqrtf(static_cast<float>(shape.head_dim));
  const __nv_bfloat16 *q_tok =
      q + (token * shape.q_heads + qh) * shape.head_dim;
  const size_t d = threadIdx.x;
  const bool active = d < shape.head_dim;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31) >> 5;
  const float q_val = active ? __bfloat162float(q_tok[d]) : 0.0f;
  float acc = 0.0f;
  float max_score = -INFINITY;
  float denom = 0.0f;

  for (size_t t = 0; t <= position; ++t) {
    // Tree mask: restrict intra-chunk visibility via the per-row ancestor
    // bitmap. Positions before the chunk (cache prefix) remain fully visible.
    // The gate is uniform across all threads in the block (t and token are
    // block-uniform), so `continue` skips all threads identically — no
    // __syncthreads() hazard.
    if (tree_ancestor_bitmap_u64 != nullptr && verify_chunk_rows > 0) {
      const size_t chunk_base = shared_start_position;
      if (t >= chunk_base && t < chunk_base + verify_chunk_rows) {
        const uint32_t row = static_cast<uint32_t>(token);
        const uint32_t col = static_cast<uint32_t>(t - chunk_base);
        if (row < verify_chunk_rows) {
          const uint64_t mask = tree_ancestor_bitmap_u64[row];
          if (!(mask & (1ULL << col))) {
            continue;
          }
        }
      }
    }
    float local = active
                      ? q_val *
                            load_cache_value(cache_k, kv_cache_dtype,
                                             (t * shape.kv_heads + kvh) *
                                                     shape.head_dim +
                                                 d)
                      : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      local += __shfl_xor_sync(0xffffffff, local, offset);
    }
    if (lane_id == 0) {
      warp_sums[warp_id] = local;
    }
    __syncthreads();
    if (warp_id == 0) {
      float total = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffff, total, offset);
      }
      if (lane_id == 0) {
        score_share = total * scale;
      }
    }
    __syncthreads();

    const float score = score_share;
    const float new_max = fmaxf(max_score, score);
    const float old_scale =
        isinf(max_score) && max_score < 0.0f ? 0.0f : expf(max_score - new_max);
    const float score_scale = expf(score - new_max);
    if (active) {
      const float vv = load_cache_value(
          cache_v, kv_cache_dtype,
          (t * shape.kv_heads + kvh) * shape.head_dim + d);
      acc = acc * old_scale + score_scale * vv;
    }
    denom = denom * old_scale + score_scale;
    max_score = new_max;
  }
  if (active) {
    output[(token * shape.q_heads + qh) * shape.head_dim + d] =
        __float2bfloat16(acc / denom);
  }
}

} // namespace

extern "C" int
qwen36_attention_prefill(const qwen36_attention_prefill_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->tokens == 0 || spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0 ||
      spec->v_bf16.ptr == 0 || spec->kv_cache_k.ptr == 0 ||
      spec->kv_cache_v.ptr == 0 || spec->output_bf16.ptr == 0 ||
      spec->shape.q_heads == 0 || spec->shape.kv_heads == 0 ||
      spec->shape.head_dim == 0 || spec->shape.head_dim > 256 ||
      spec->shape.q_heads % spec->shape.kv_heads != 0 ||
      (spec->kv_cache_dtype != kKvCacheBf16 &&
       spec->kv_cache_dtype != kKvCacheFp8)) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  // Tree mask is only supported by attention_prefill_kernel (the basic path).
  // Routing to split or GQA kernels with a tree mask would silently ignore
  // the mask, so guard against it here as a hard contract.
  const bool tree_mask_present = spec->tree_ancestor_bitmap_u64.ptr != 0
                               && spec->verify_chunk_rows > 0;

  const int threads = 256;
  const dim3 copy_grid(static_cast<unsigned int>(spec->shape.kv_heads),
                       static_cast<unsigned int>(spec->tokens));
  copy_kv_prefill_kernel<<<copy_grid, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->k_bf16),
      ptr<const __nv_bfloat16>(spec->v_bf16),
      ptr<void>(spec->kv_cache_k), ptr<void>(spec->kv_cache_v),
      spec->kv_cache_dtype, spec->start_position, spec->tokens,
      ptr<const int32_t>(spec->start_position_device_i32), spec->shape.kv_heads,
      spec->shape.head_dim);

  const bool partials_present = spec->partial_acc_f32.ptr != 0 &&
                                spec->partial_max_f32.ptr != 0 &&
                                spec->partial_denom_f32.ptr != 0;
  if (partials_present && spec->prefill_n_splits >= 2 && spec->tokens <= 2) {
    if (tree_mask_present) {
      return QWEN36_STATUS_NOT_IMPLEMENTED;
    }
    const int split_timesteps_per_block =
        spec->split_timesteps_per_block == 0
            ? kDefaultSplitTimestepsPerBlock
            : static_cast<int>(spec->split_timesteps_per_block);
    if (split_timesteps_per_block < kMinSplitTimestepsPerBlock) {
      return QWEN36_STATUS_INVALID_ARGUMENT;
    }
    unsigned int split_threads = static_cast<unsigned int>(spec->shape.head_dim);
    split_threads = (split_threads + 31u) & ~31u;
    if (split_threads == 0) {
      split_threads = 32u;
    } else if (split_threads > 256u) {
      split_threads = 256u;
    }
    const int n_splits = static_cast<int>(spec->prefill_n_splits);
    const dim3 split_grid(static_cast<unsigned int>(spec->shape.q_heads),
                          static_cast<unsigned int>(n_splits));
    for (size_t token = 0; token < spec->tokens; ++token) {
      attention_prefill_split_kernel<<<split_grid, split_threads, 0,
                                       qwen36_internal_active_stream()>>>(
          ptr<const __nv_bfloat16>(spec->q_bf16),
          ptr<const void>(spec->kv_cache_k),
          ptr<const void>(spec->kv_cache_v), spec->kv_cache_dtype,
          ptr<float>(spec->partial_acc_f32),
          ptr<float>(spec->partial_max_f32),
          ptr<float>(spec->partial_denom_f32), spec->start_position,
          ptr<const int32_t>(spec->start_position_device_i32), token,
          spec->shape, n_splits, split_timesteps_per_block);
      cudaError_t split_err = cudaGetLastError();
      if (split_err != cudaSuccess) {
        return QWEN36_STATUS_CUDA_ERROR;
      }
      __nv_bfloat16 *token_output =
          ptr<__nv_bfloat16>(spec->output_bf16) +
          token * spec->shape.q_heads * spec->shape.head_dim;
      attention_decode_reduce_kernel<<<
          static_cast<unsigned int>(spec->shape.q_heads), split_threads, 0,
          qwen36_internal_active_stream()>>>(
          ptr<const float>(spec->partial_acc_f32),
          ptr<const float>(spec->partial_max_f32),
          ptr<const float>(spec->partial_denom_f32), token_output,
          spec->shape, n_splits);
      cudaError_t reduce_err = cudaGetLastError();
      if (reduce_err != cudaSuccess) {
        return QWEN36_STATUS_CUDA_ERROR;
      }
    }
    return QWEN36_STATUS_SUCCESS;
  }

  // Prefer the GQA-aware kernel for the common Qwen3.6 shape: it lays out
  // the grid as (kv_heads × tokens) instead of (q_heads × tokens) and
  // shares each cache row across the q_per_kv queries that consume it,
  // eliminating the (q_per_kv − 1)× redundant cache reads the per-q-head
  // kernel relied on the L2 cache to absorb. With prefill `tokens` in the
  // hundreds-to-thousands the grid stays large enough to saturate the GPU
  // even at q_per_kv = 6.
  // Threshold below which the per-q-head kernel still wins. The GQA kernel
  // does 6x the per-block work and only `kv_heads` blocks per token, so for
  // very short chunks (notably 2-token MTP verify chunks) it under-utilizes
  // the GPU even at q_per_kv = 6. Empirically `tokens >= 16` is where the
  // crossover lands on Qwen3.6 / Blackwell.
  constexpr size_t kPrefillGqaMinTokens = 16;

  const size_t q_per_kv = spec->shape.q_heads / spec->shape.kv_heads;
  const bool gqa_eligible =
      spec->shape.head_dim <= static_cast<size_t>(kGqaMaxHeadDim) &&
      q_per_kv <= static_cast<size_t>(kGqaMaxQPerKv) && q_per_kv > 1 &&
      spec->tokens >= kPrefillGqaMinTokens;
  if (gqa_eligible) {
    if (tree_mask_present) {
      return QWEN36_STATUS_NOT_IMPLEMENTED;
    }
    unsigned int gqa_threads = static_cast<unsigned int>(spec->shape.head_dim);
    gqa_threads = (gqa_threads + 31u) & ~31u;
    if (gqa_threads == 0) {
      gqa_threads = 32u;
    } else if (gqa_threads > 256u) {
      gqa_threads = 256u;
    }
    const dim3 gqa_grid(static_cast<unsigned int>(spec->shape.kv_heads),
                        static_cast<unsigned int>(spec->tokens));
    attention_prefill_gqa_kernel<<<gqa_grid, gqa_threads, 0,
                                   qwen36_internal_active_stream()>>>(
        ptr<const __nv_bfloat16>(spec->q_bf16),
        ptr<const void>(spec->kv_cache_k), ptr<const void>(spec->kv_cache_v),
        spec->kv_cache_dtype, ptr<__nv_bfloat16>(spec->output_bf16), spec->start_position,
        ptr<const int32_t>(spec->start_position_device_i32), spec->tokens,
        spec->shape);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
  }

  const dim3 attn_grid(static_cast<unsigned int>(spec->shape.q_heads),
                       static_cast<unsigned int>(spec->tokens));
  attention_prefill_kernel<<<attn_grid, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->q_bf16),
      ptr<const void>(spec->kv_cache_k), ptr<const void>(spec->kv_cache_v),
      spec->kv_cache_dtype, ptr<__nv_bfloat16>(spec->output_bf16), spec->start_position,
      ptr<const int32_t>(spec->start_position_device_i32), spec->tokens,
      spec->shape,
      spec->tree_ancestor_bitmap_u64.ptr != 0
          ? ptr<const uint64_t>(spec->tree_ancestor_bitmap_u64)
          : nullptr,
      spec->verify_chunk_rows);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int
qwen36_attention_decode(const qwen36_attention_decode_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0 ||
      spec->v_bf16.ptr == 0 || spec->kv_cache_k.ptr == 0 ||
      spec->kv_cache_v.ptr == 0 || spec->output_bf16.ptr == 0 ||
      spec->shape.q_heads == 0 || spec->shape.kv_heads == 0 ||
      spec->shape.head_dim == 0 || spec->shape.head_dim > 256 ||
      spec->shape.q_heads % spec->shape.kv_heads != 0 ||
      (spec->kv_cache_dtype != kKvCacheBf16 &&
       spec->kv_cache_dtype != kKvCacheFp8)) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  // Split-KV / FlashDecoding path: when `decode_n_splits >= 2` the engine
  // has decided this attention call is on a long enough context to benefit
  // from T-axis parallelism. We launch a fixed grid of (q_heads × n_splits)
  // blocks and a small follow-up reduce — the value is derived from
  // `max_context`, not the current position, so the same launch shape is
  // valid for both fresh kernel calls AND graph replays where the position
  // grows after capture. Empty splits early-exit cheaply via the
  // `t_start >= position + 1` guard inside the kernel.
  const bool partials_present = spec->partial_acc_f32.ptr != 0 &&
                                spec->partial_max_f32.ptr != 0 &&
                                spec->partial_denom_f32.ptr != 0;
  if (partials_present && spec->decode_n_splits >= 2 &&
      spec->shape.head_dim <= 256) {
    const int split_timesteps_per_block =
        spec->split_timesteps_per_block == 0
            ? kDefaultSplitTimestepsPerBlock
            : static_cast<int>(spec->split_timesteps_per_block);
    if (split_timesteps_per_block < kMinSplitTimestepsPerBlock) {
      return QWEN36_STATUS_INVALID_ARGUMENT;
    }
    unsigned int split_threads = static_cast<unsigned int>(spec->shape.head_dim);
    split_threads = (split_threads + 31u) & ~31u;
    if (split_threads == 0) {
      split_threads = 32u;
    } else if (split_threads > 256u) {
      split_threads = 256u;
    }
    const int n_splits = static_cast<int>(spec->decode_n_splits);
    const size_t q_per_kv = spec->shape.q_heads / spec->shape.kv_heads;
    const bool gqa_split_eligible =
        spec->shape.head_dim <= static_cast<size_t>(kGqaMaxHeadDim) &&
        q_per_kv <= static_cast<size_t>(kGqaMaxQPerKv) && q_per_kv > 1 &&
        n_splits >= 32;
    if (gqa_split_eligible) {
      const dim3 split_grid(static_cast<unsigned int>(spec->shape.kv_heads),
                            static_cast<unsigned int>(n_splits));
      attention_decode_split_gqa_kernel<<<split_grid, split_threads, 0,
                                          qwen36_internal_active_stream()>>>(
          ptr<const __nv_bfloat16>(spec->q_bf16),
          ptr<const __nv_bfloat16>(spec->k_bf16),
          ptr<const __nv_bfloat16>(spec->v_bf16),
          ptr<void>(spec->kv_cache_k), ptr<void>(spec->kv_cache_v),
          spec->kv_cache_dtype,
          ptr<float>(spec->partial_acc_f32),
          ptr<float>(spec->partial_max_f32),
          ptr<float>(spec->partial_denom_f32), spec->position,
          ptr<const int32_t>(spec->position_device_i32), spec->shape, n_splits,
          split_timesteps_per_block);
    } else {
      const dim3 split_grid(static_cast<unsigned int>(spec->shape.q_heads),
                            static_cast<unsigned int>(n_splits));
      attention_decode_split_kernel<<<split_grid, split_threads, 0,
                                      qwen36_internal_active_stream()>>>(
          ptr<const __nv_bfloat16>(spec->q_bf16),
          ptr<const __nv_bfloat16>(spec->k_bf16),
          ptr<const __nv_bfloat16>(spec->v_bf16),
          ptr<void>(spec->kv_cache_k), ptr<void>(spec->kv_cache_v),
          spec->kv_cache_dtype,
          ptr<float>(spec->partial_acc_f32),
          ptr<float>(spec->partial_max_f32),
          ptr<float>(spec->partial_denom_f32), spec->position,
          ptr<const int32_t>(spec->position_device_i32), spec->shape, n_splits,
          split_timesteps_per_block);
    }
    attention_decode_reduce_kernel<<<
        static_cast<unsigned int>(spec->shape.q_heads), split_threads, 0,
        qwen36_internal_active_stream()>>>(
        ptr<const float>(spec->partial_acc_f32),
        ptr<const float>(spec->partial_max_f32),
        ptr<const float>(spec->partial_denom_f32),
        ptr<__nv_bfloat16>(spec->output_bf16), spec->shape, n_splits);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
  }

  // Round threads up to the next multiple of 32 so warp-shuffle reductions are
  // well-defined; cap at 256 (max head_dim) so the block fits within 8 warps,
  // matching the size of the warp_sums staging array inside the kernel.
  unsigned int threads = static_cast<unsigned int>(spec->shape.head_dim);
  threads = (threads + 31u) & ~31u;
  if (threads == 0) {
    threads = 32u;
  } else if (threads > 256u) {
    threads = 256u;
  }
  attention_decode_kernel<<<static_cast<unsigned int>(spec->shape.q_heads),
                            threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->q_bf16),
      ptr<const __nv_bfloat16>(spec->k_bf16),
      ptr<const __nv_bfloat16>(spec->v_bf16),
      ptr<void>(spec->kv_cache_k), ptr<void>(spec->kv_cache_v),
      spec->kv_cache_dtype, ptr<__nv_bfloat16>(spec->output_bf16), spec->position,
      ptr<const int32_t>(spec->position_device_i32), spec->shape);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
