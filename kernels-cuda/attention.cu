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

__global__ void attention_decode_kernel(
    const __nv_bfloat16 *q, const __nv_bfloat16 *cache_k,
    const __nv_bfloat16 *cache_v, __nv_bfloat16 *output, size_t position,
    qwen36_attention_shape_t shape) {
  extern __shared__ float shared[];
  float *shared_max = shared;
  float *shared_sum = shared + 1;

  const size_t qh = blockIdx.x;
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t kvh = qh / q_per_kv;
  const float scale = rsqrtf(static_cast<float>(shape.head_dim));

  if (threadIdx.x == 0) {
    float max_score = -INFINITY;
    for (size_t t = 0; t <= position; ++t) {
      float dot = 0.0f;
      for (size_t d = 0; d < shape.head_dim; ++d) {
        const float qv = __bfloat162float(q[qh * shape.head_dim + d]);
        const float kv = __bfloat162float(
            cache_k[(t * shape.kv_heads + kvh) * shape.head_dim + d]);
        dot += qv * kv;
      }
      max_score = fmaxf(max_score, dot * scale);
    }
    float denom = 0.0f;
    for (size_t t = 0; t <= position; ++t) {
      float dot = 0.0f;
      for (size_t d = 0; d < shape.head_dim; ++d) {
        const float qv = __bfloat162float(q[qh * shape.head_dim + d]);
        const float kv = __bfloat162float(
            cache_k[(t * shape.kv_heads + kvh) * shape.head_dim + d]);
        dot += qv * kv;
      }
      denom += expf(dot * scale - max_score);
    }
    *shared_max = max_score;
    *shared_sum = denom;
  }
  __syncthreads();

  for (size_t d = threadIdx.x; d < shape.head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (size_t t = 0; t <= position; ++t) {
      float dot = 0.0f;
      for (size_t kd = 0; kd < shape.head_dim; ++kd) {
        const float qv = __bfloat162float(q[qh * shape.head_dim + kd]);
        const float kv = __bfloat162float(
            cache_k[(t * shape.kv_heads + kvh) * shape.head_dim + kd]);
        dot += qv * kv;
      }
      const float weight = expf(dot * scale - *shared_max) / *shared_sum;
      const float vv = __bfloat162float(
          cache_v[(t * shape.kv_heads + kvh) * shape.head_dim + d]);
      acc += weight * vv;
    }
    output[qh * shape.head_dim + d] = __float2bfloat16(acc);
  }
}

} // namespace

extern "C" int
qwen36_attention_decode(const qwen36_attention_decode_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0 ||
      spec->v_bf16.ptr == 0 || spec->kv_cache_k.ptr == 0 ||
      spec->kv_cache_v.ptr == 0 || spec->output_bf16.ptr == 0 ||
      spec->shape.q_heads == 0 || spec->shape.kv_heads == 0 ||
      spec->shape.head_dim == 0 ||
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
                            threads, 2 * sizeof(float)>>>(
      ptr<const __nv_bfloat16>(spec->q_bf16),
      ptr<const __nv_bfloat16>(spec->kv_cache_k),
      ptr<const __nv_bfloat16>(spec->kv_cache_v),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->position, spec->shape);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
