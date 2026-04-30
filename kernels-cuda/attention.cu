#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

template <typename T> T *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

__global__ void copy_kv_prefill_kernel(
    const __nv_bfloat16 *k, const __nv_bfloat16 *v, __nv_bfloat16 *cache_k,
    __nv_bfloat16 *cache_v, size_t start_position, size_t tokens,
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
    cache_k[dst] = k[src];
    cache_v[dst] = v[src];
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
    const __nv_bfloat16 *v_new, __nv_bfloat16 *cache_k, __nv_bfloat16 *cache_v,
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
                              : __bfloat162float(cache_k[(t * shape.kv_heads +
                                                          kvh) *
                                                             shape.head_dim +
                                                         d]);
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
                              : __bfloat162float(cache_v[(t * shape.kv_heads +
                                                          kvh) *
                                                             shape.head_dim +
                                                         d]);
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
      cache_k[cache_off] = k_new[kvh * shape.head_dim + d];
      cache_v[cache_off] = v_new[kvh * shape.head_dim + d];
    }
  }
}

__global__ void attention_prefill_kernel(
    const __nv_bfloat16 *q, const __nv_bfloat16 *cache_k,
    const __nv_bfloat16 *cache_v, __nv_bfloat16 *output,
    size_t start_position, const int32_t *start_position_device, size_t tokens,
    qwen36_attention_shape_t shape) {
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
    float local = active
                      ? q_val *
                            __bfloat162float(
                                cache_k[(t * shape.kv_heads + kvh) *
                                            shape.head_dim +
                                        d])
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
      const float vv = __bfloat162float(
          cache_v[(t * shape.kv_heads + kvh) * shape.head_dim + d]);
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
      spec->shape.q_heads % spec->shape.kv_heads != 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  const int threads = 256;
  const dim3 copy_grid(static_cast<unsigned int>(spec->shape.kv_heads),
                       static_cast<unsigned int>(spec->tokens));
  copy_kv_prefill_kernel<<<copy_grid, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->k_bf16),
      ptr<const __nv_bfloat16>(spec->v_bf16),
      ptr<__nv_bfloat16>(spec->kv_cache_k),
      ptr<__nv_bfloat16>(spec->kv_cache_v), spec->start_position,
      spec->tokens, ptr<const int32_t>(spec->start_position_device_i32),
      spec->shape.kv_heads, spec->shape.head_dim);
  const dim3 attn_grid(static_cast<unsigned int>(spec->shape.q_heads),
                       static_cast<unsigned int>(spec->tokens));
  attention_prefill_kernel<<<attn_grid, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->q_bf16),
      ptr<const __nv_bfloat16>(spec->kv_cache_k),
      ptr<const __nv_bfloat16>(spec->kv_cache_v),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->start_position,
      ptr<const int32_t>(spec->start_position_device_i32), spec->tokens,
      spec->shape);
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
      spec->shape.q_heads % spec->shape.kv_heads != 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
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
      ptr<__nv_bfloat16>(spec->kv_cache_k),
      ptr<__nv_bfloat16>(spec->kv_cache_v),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->position,
      ptr<const int32_t>(spec->position_device_i32), spec->shape);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
