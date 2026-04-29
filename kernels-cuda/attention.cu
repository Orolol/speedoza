#include "qwen36_fp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

template <typename T> T *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

__global__ void copy_kv_kernel(const __nv_bfloat16 *k, const __nv_bfloat16 *v,
                               __nv_bfloat16 *cache_k,
                               __nv_bfloat16 *cache_v, size_t position,
                               size_t kv_heads, size_t head_dim) {
  const size_t kvh = blockIdx.x;
  for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const size_t src = kvh * head_dim + d;
    const size_t dst = (position * kv_heads + kvh) * head_dim + d;
    cache_k[dst] = k[src];
    cache_v[dst] = v[src];
  }
}

__global__ void copy_kv_prefill_kernel(
    const __nv_bfloat16 *k, const __nv_bfloat16 *v, __nv_bfloat16 *cache_k,
    __nv_bfloat16 *cache_v, size_t start_position, size_t tokens,
    size_t kv_heads, size_t head_dim) {
  const size_t kvh = blockIdx.x;
  const size_t token = blockIdx.y;
  if (token >= tokens) {
    return;
  }
  for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const size_t src = (token * kv_heads + kvh) * head_dim + d;
    const size_t dst =
        ((start_position + token) * kv_heads + kvh) * head_dim + d;
    cache_k[dst] = k[src];
    cache_v[dst] = v[src];
  }
}

__global__ void attention_decode_kernel(
    const __nv_bfloat16 *q, const __nv_bfloat16 *cache_k,
    const __nv_bfloat16 *cache_v, __nv_bfloat16 *output, size_t position,
    qwen36_attention_shape_t shape) {
  extern __shared__ float scratch[];

  const size_t qh = blockIdx.x;
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t kvh = qh / q_per_kv;
  const float scale = rsqrtf(static_cast<float>(shape.head_dim));
  const size_t d = threadIdx.x;
  float acc = 0.0f;
  float max_score = -INFINITY;
  float denom = 0.0f;

  for (size_t t = 0; t <= position; ++t) {
    float local = 0.0f;
    if (d < shape.head_dim) {
      const float qv = __bfloat162float(q[qh * shape.head_dim + d]);
      const float kv = __bfloat162float(
          cache_k[(t * shape.kv_heads + kvh) * shape.head_dim + d]);
      local = qv * kv;
    }
    scratch[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        scratch[threadIdx.x] += scratch[threadIdx.x + stride];
      }
      __syncthreads();
    }

    const float score = scratch[0] * scale;
    const float new_max = fmaxf(max_score, score);
    const float old_scale =
        isinf(max_score) && max_score < 0.0f ? 0.0f : expf(max_score - new_max);
    const float score_scale = expf(score - new_max);
    if (d < shape.head_dim) {
      const float vv = __bfloat162float(
          cache_v[(t * shape.kv_heads + kvh) * shape.head_dim + d]);
      acc = acc * old_scale + score_scale * vv;
    }
    denom = denom * old_scale + score_scale;
    max_score = new_max;
    __syncthreads();
  }
  if (d < shape.head_dim) {
    output[qh * shape.head_dim + d] = __float2bfloat16(acc / denom);
  }
}

__global__ void attention_prefill_kernel(
    const __nv_bfloat16 *q, const __nv_bfloat16 *cache_k,
    const __nv_bfloat16 *cache_v, __nv_bfloat16 *output,
    size_t start_position, size_t tokens, qwen36_attention_shape_t shape) {
  extern __shared__ float scratch[];

  const size_t qh = blockIdx.x;
  const size_t token = blockIdx.y;
  if (token >= tokens) {
    return;
  }
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t kvh = qh / q_per_kv;
  const size_t position = start_position + token;
  const float scale = rsqrtf(static_cast<float>(shape.head_dim));
  const __nv_bfloat16 *q_tok =
      q + (token * shape.q_heads + qh) * shape.head_dim;
  const size_t d = threadIdx.x;
  float acc = 0.0f;
  float max_score = -INFINITY;
  float denom = 0.0f;

  for (size_t t = 0; t <= position; ++t) {
    float local = 0.0f;
    if (d < shape.head_dim) {
      const float qv = __bfloat162float(q_tok[d]);
      const float kv = __bfloat162float(
          cache_k[(t * shape.kv_heads + kvh) * shape.head_dim + d]);
      local = qv * kv;
    }
    scratch[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        scratch[threadIdx.x] += scratch[threadIdx.x + stride];
      }
      __syncthreads();
    }

    const float score = scratch[0] * scale;
    const float new_max = fmaxf(max_score, score);
    const float old_scale =
        isinf(max_score) && max_score < 0.0f ? 0.0f : expf(max_score - new_max);
    const float score_scale = expf(score - new_max);
    if (d < shape.head_dim) {
      const float vv = __bfloat162float(
          cache_v[(t * shape.kv_heads + kvh) * shape.head_dim + d]);
      acc = acc * old_scale + score_scale * vv;
    }
    denom = denom * old_scale + score_scale;
    max_score = new_max;
    __syncthreads();
  }
  if (d < shape.head_dim) {
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
  copy_kv_prefill_kernel<<<copy_grid, threads>>>(
      ptr<const __nv_bfloat16>(spec->k_bf16),
      ptr<const __nv_bfloat16>(spec->v_bf16),
      ptr<__nv_bfloat16>(spec->kv_cache_k),
      ptr<__nv_bfloat16>(spec->kv_cache_v), spec->start_position,
      spec->tokens, spec->shape.kv_heads, spec->shape.head_dim);
  const dim3 attn_grid(static_cast<unsigned int>(spec->shape.q_heads),
                       static_cast<unsigned int>(spec->tokens));
  attention_prefill_kernel<<<attn_grid, threads, threads * sizeof(float)>>>(
      ptr<const __nv_bfloat16>(spec->q_bf16),
      ptr<const __nv_bfloat16>(spec->kv_cache_k),
      ptr<const __nv_bfloat16>(spec->kv_cache_v),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->start_position,
      spec->tokens, spec->shape);
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

  const int threads = 256;
  copy_kv_kernel<<<static_cast<unsigned int>(spec->shape.kv_heads), threads>>>(
      ptr<const __nv_bfloat16>(spec->k_bf16),
      ptr<const __nv_bfloat16>(spec->v_bf16),
      ptr<__nv_bfloat16>(spec->kv_cache_k),
      ptr<__nv_bfloat16>(spec->kv_cache_v), spec->position,
      spec->shape.kv_heads, spec->shape.head_dim);
  attention_decode_kernel<<<static_cast<unsigned int>(spec->shape.q_heads),
                            threads, threads * sizeof(float)>>>(
      ptr<const __nv_bfloat16>(spec->q_bf16),
      ptr<const __nv_bfloat16>(spec->kv_cache_k),
      ptr<const __nv_bfloat16>(spec->kv_cache_v),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->position, spec->shape);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
