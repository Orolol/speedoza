#include "qwen36_fp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

template <typename T> T *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

__global__ void deltanet_decode_kernel(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
    const float *gate, const float *beta, __nv_bfloat16 *state,
    __nv_bfloat16 *output, qwen36_deltanet_shape_t shape, size_t tokens,
    size_t q_token_stride, size_t k_token_stride, size_t v_token_stride,
    float state_decay, float update_scale, bool qk_l2norm) {
  __shared__ float shared_decay;
  __shared__ float shared_beta;
  __shared__ float shared_q_norm;
  __shared__ float shared_k_norm;
  __shared__ float shared_q_squares[128];
  __shared__ float shared_k_squares[128];
  const size_t vd = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t vh = blockIdx.y;
  if (vh >= shape.v_heads) {
    return;
  }
  const bool active = vd < shape.value_dim;

  const size_t q_repeat = shape.v_heads / shape.qk_heads;
  const size_t qh = vh / q_repeat;
  const size_t q_stride =
      q_token_stride == 0 ? shape.qk_heads * shape.key_dim : q_token_stride;
  const size_t k_stride =
      k_token_stride == 0 ? shape.qk_heads * shape.key_dim : k_token_stride;
  const size_t v_stride =
      v_token_stride == 0 ? shape.v_heads * shape.value_dim : v_token_stride;
  __nv_bfloat16 *state_row =
      active ? state + (vh * shape.value_dim + vd) * shape.key_dim : nullptr;

  for (size_t tok = 0; tok < tokens; ++tok) {
    const __nv_bfloat16 *q_tok = q + tok * q_stride + qh * shape.key_dim;
    const __nv_bfloat16 *k_tok = k + tok * k_stride + qh * shape.key_dim;
    const float v_value =
        active ? __bfloat162float(v[tok * v_stride + vh * shape.value_dim + vd])
               : 0.0f;

    if (gate != nullptr && beta != nullptr) {
      if (threadIdx.x == 0) {
        shared_decay = expf(gate[tok * shape.v_heads + vh]);
        shared_beta = beta[tok * shape.v_heads + vh];
      }
      float q_squares = 0.0f;
      float k_squares = 0.0f;
      for (size_t kd = threadIdx.x; kd < shape.key_dim; kd += blockDim.x) {
        const float q_value = __bfloat162float(q_tok[kd]);
        const float k_value = __bfloat162float(k_tok[kd]);
        q_squares += q_value * q_value;
        k_squares += k_value * k_value;
      }
      shared_q_squares[threadIdx.x] = q_squares;
      shared_k_squares[threadIdx.x] = k_squares;
      __syncthreads();
      for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          shared_q_squares[threadIdx.x] +=
              shared_q_squares[threadIdx.x + stride];
          shared_k_squares[threadIdx.x] +=
              shared_k_squares[threadIdx.x + stride];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        if (qk_l2norm) {
          shared_q_norm = rsqrtf(shared_q_squares[0] + 1.0e-6f) *
                          rsqrtf(static_cast<float>(shape.key_dim));
          shared_k_norm = rsqrtf(shared_k_squares[0] + 1.0e-6f);
        } else {
          shared_q_norm = rsqrtf(static_cast<float>(shape.key_dim));
          shared_k_norm = 1.0f;
        }
      }
      __syncthreads();
      if (!active) {
        __syncthreads();
        continue;
      }
      const float q_norm = shared_q_norm;
      const float k_norm = shared_k_norm;

      const float decay = shared_decay;
      const float beta_value = shared_beta;
      float kv_mem = 0.0f;
      for (size_t kd = 0; kd < shape.key_dim; ++kd) {
        const float key = __bfloat162float(k_tok[kd]) * k_norm;
        const float decayed = __bfloat162float(state_row[kd]) * decay;
        state_row[kd] = __float2bfloat16(decayed);
        kv_mem += decayed * key;
      }

      const float delta = (v_value - kv_mem) * beta_value;
      float acc = 0.0f;
      for (size_t kd = 0; kd < shape.key_dim; ++kd) {
        const float key = __bfloat162float(k_tok[kd]) * k_norm;
        const float query = __bfloat162float(q_tok[kd]) * q_norm;
        const float updated = __bfloat162float(state_row[kd]) + key * delta;
        state_row[kd] = __float2bfloat16(updated);
        acc += updated * query;
      }
      output[(tok * shape.v_heads + vh) * shape.value_dim + vd] =
          __float2bfloat16(acc);
      __syncthreads();
      continue;
    }

    if (!active) {
      continue;
    }
    float acc = 0.0f;
    for (size_t kd = 0; kd < shape.key_dim; ++kd) {
      const float previous = __bfloat162float(state_row[kd]);
      const float key = __bfloat162float(k_tok[kd]);
      const float updated = previous * state_decay + update_scale * v_value * key;
      state_row[kd] = __float2bfloat16(updated);
      acc += updated * __bfloat162float(q_tok[kd]);
    }
    output[(tok * shape.v_heads + vh) * shape.value_dim + vd] =
        __float2bfloat16(acc);
  }
}

} // namespace

extern "C" int
qwen36_deltanet_decode(const qwen36_deltanet_decode_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0 ||
      spec->v_bf16.ptr == 0 || spec->state_bf16.ptr == 0 ||
      spec->output_bf16.ptr == 0 || spec->shape.qk_heads == 0 ||
      spec->shape.v_heads == 0 || spec->shape.key_dim == 0 ||
      spec->shape.value_dim == 0 || spec->tokens_in_persistent_loop == 0 ||
      spec->shape.v_heads % spec->shape.qk_heads != 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if ((spec->gate_f32.ptr == 0) != (spec->beta_f32.ptr == 0)) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  const int threads = 128;
  const dim3 grid(
      static_cast<unsigned int>((spec->shape.value_dim + threads - 1) / threads),
      static_cast<unsigned int>(spec->shape.v_heads));
  const float state_decay = spec->state_decay == 0.0f ? 1.0f : spec->state_decay;
  const float update_scale =
      spec->update_scale == 0.0f ? 1.0f : spec->update_scale;
  deltanet_decode_kernel<<<grid, threads>>>(
      ptr<const __nv_bfloat16>(spec->q_bf16),
      ptr<const __nv_bfloat16>(spec->k_bf16),
      ptr<const __nv_bfloat16>(spec->v_bf16),
      ptr<const float>(spec->gate_f32), ptr<const float>(spec->beta_f32),
      ptr<__nv_bfloat16>(spec->state_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->shape,
      spec->tokens_in_persistent_loop, spec->q_token_stride,
      spec->k_token_stride, spec->v_token_stride, state_decay, update_scale,
      spec->qk_l2norm != 0);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
