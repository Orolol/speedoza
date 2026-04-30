#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

template <typename T> T *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

__device__ int8_t quantize_s8(float value, float scale) {
  const float scaled = scale == 0.0f ? 0.0f : value / scale;
  const int rounded = static_cast<int>(lrintf(scaled));
  const int high_clamped = rounded > 127 ? 127 : rounded;
  const int clamped = high_clamped < -127 ? -127 : high_clamped;
  return static_cast<int8_t>(clamped);
}

__global__ void encode_kv_kernel(const __nv_bfloat16 *k,
                                 const __nv_bfloat16 *v, int8_t *kq,
                                 int8_t *vq, float *metadata,
                                 size_t position,
                                 qwen36_attention_shape_t shape) {
  extern __shared__ float scratch[];
  float *max_k = scratch;
  float *max_v = scratch + blockDim.x;
  const size_t kvh = blockIdx.x;

  float local_k = 0.0f;
  float local_v = 0.0f;
  for (size_t d = threadIdx.x; d < shape.head_dim; d += blockDim.x) {
    local_k = fmaxf(local_k,
                    fabsf(__bfloat162float(k[kvh * shape.head_dim + d])));
    local_v = fmaxf(local_v,
                    fabsf(__bfloat162float(v[kvh * shape.head_dim + d])));
  }
  max_k[threadIdx.x] = local_k;
  max_v[threadIdx.x] = local_v;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      max_k[threadIdx.x] = fmaxf(max_k[threadIdx.x], max_k[threadIdx.x + stride]);
      max_v[threadIdx.x] = fmaxf(max_v[threadIdx.x], max_v[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const float k_scale = max_k[0] / 127.0f;
  const float v_scale = max_v[0] / 127.0f;
  if (threadIdx.x == 0) {
    const size_t meta = (position * shape.kv_heads + kvh) * 2;
    metadata[meta] = k_scale;
    metadata[meta + 1] = v_scale;
  }

  for (size_t d = threadIdx.x; d < shape.head_dim; d += blockDim.x) {
    const size_t src = kvh * shape.head_dim + d;
    const size_t dst = (position * shape.kv_heads + kvh) * shape.head_dim + d;
    kq[dst] = quantize_s8(__bfloat162float(k[src]), k_scale);
    vq[dst] = quantize_s8(__bfloat162float(v[src]), v_scale);
  }
}

__global__ void turboquant_attention_kernel(
    const __nv_bfloat16 *q, const int8_t *kq, const int8_t *vq,
    const float *metadata, __nv_bfloat16 *output, size_t position,
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
      const float k_scale = metadata[(t * shape.kv_heads + kvh) * 2];
      float dot = 0.0f;
      for (size_t d = 0; d < shape.head_dim; ++d) {
        const float qv = __bfloat162float(q[qh * shape.head_dim + d]);
        const float kv = static_cast<float>(
                             kq[(t * shape.kv_heads + kvh) * shape.head_dim + d]) *
                         k_scale;
        dot += qv * kv;
      }
      max_score = fmaxf(max_score, dot * scale);
    }
    float denom = 0.0f;
    for (size_t t = 0; t <= position; ++t) {
      const float k_scale = metadata[(t * shape.kv_heads + kvh) * 2];
      float dot = 0.0f;
      for (size_t d = 0; d < shape.head_dim; ++d) {
        const float qv = __bfloat162float(q[qh * shape.head_dim + d]);
        const float kv = static_cast<float>(
                             kq[(t * shape.kv_heads + kvh) * shape.head_dim + d]) *
                         k_scale;
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
      const float k_scale = metadata[(t * shape.kv_heads + kvh) * 2];
      const float v_scale = metadata[(t * shape.kv_heads + kvh) * 2 + 1];
      float dot = 0.0f;
      for (size_t kd = 0; kd < shape.head_dim; ++kd) {
        const float qv = __bfloat162float(q[qh * shape.head_dim + kd]);
        const float kv = static_cast<float>(
                             kq[(t * shape.kv_heads + kvh) * shape.head_dim + kd]) *
                         k_scale;
        dot += qv * kv;
      }
      const float weight = expf(dot * scale - *shared_max) / *shared_sum;
      const float vv = static_cast<float>(
                           vq[(t * shape.kv_heads + kvh) * shape.head_dim + d]) *
                       v_scale;
      acc += weight * vv;
    }
    output[qh * shape.head_dim + d] = __float2bfloat16(acc);
  }
}

} // namespace

extern "C" int
qwen36_turboquant_encode_kv(const qwen36_turboquant_encode_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->k_bf16.ptr == 0 || spec->v_bf16.ptr == 0 ||
      spec->k_quantized_i8.ptr == 0 || spec->v_quantized_i8.ptr == 0 ||
      spec->metadata_f32.ptr == 0 || spec->shape.kv_heads == 0 ||
      spec->shape.head_dim == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  encode_kv_kernel<<<static_cast<unsigned int>(spec->shape.kv_heads), threads,
                     2 * threads * sizeof(float), qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->k_bf16),
      ptr<const __nv_bfloat16>(spec->v_bf16),
      ptr<int8_t>(spec->k_quantized_i8), ptr<int8_t>(spec->v_quantized_i8),
      ptr<float>(spec->metadata_f32), spec->position, spec->shape);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_turboquant_attention(
    const qwen36_turboquant_attention_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->q_bf16.ptr == 0 || spec->k_quantized_i8.ptr == 0 ||
      spec->v_quantized_i8.ptr == 0 || spec->metadata_f32.ptr == 0 ||
      spec->output_bf16.ptr == 0 || spec->shape.q_heads == 0 ||
      spec->shape.kv_heads == 0 || spec->shape.head_dim == 0 ||
      spec->shape.q_heads % spec->shape.kv_heads != 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  turboquant_attention_kernel<<<static_cast<unsigned int>(spec->shape.q_heads),
                                threads, 2 * sizeof(float), qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->q_bf16),
      ptr<const int8_t>(spec->k_quantized_i8),
      ptr<const int8_t>(spec->v_quantized_i8),
      ptr<const float>(spec->metadata_f32),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->position, spec->shape);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
