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
    float state_decay, float update_scale, bool qk_l2norm) {
  const size_t vd = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t vh = blockIdx.y;
  if (vd >= shape.value_dim || vh >= shape.v_heads) {
    return;
  }

  const size_t q_repeat = shape.v_heads / shape.qk_heads;
  const size_t qh = vh / q_repeat;
  __nv_bfloat16 *state_row =
      state + (vh * shape.value_dim + vd) * shape.key_dim;

  for (size_t tok = 0; tok < tokens; ++tok) {
    const __nv_bfloat16 *q_tok =
        q + (tok * shape.qk_heads + qh) * shape.key_dim;
    const __nv_bfloat16 *k_tok =
        k + (tok * shape.qk_heads + qh) * shape.key_dim;
    const float v_value =
        __bfloat162float(v[(tok * shape.v_heads + vh) * shape.value_dim + vd]);

    if (gate != nullptr && beta != nullptr) {
      float q_norm = rsqrtf(static_cast<float>(shape.key_dim));
      float k_norm = 1.0f;
      if (qk_l2norm) {
        float q_squares = 0.0f;
        float k_squares = 0.0f;
        for (size_t kd = 0; kd < shape.key_dim; ++kd) {
          const float q_value = __bfloat162float(q_tok[kd]);
          const float k_value = __bfloat162float(k_tok[kd]);
          q_squares += q_value * q_value;
          k_squares += k_value * k_value;
        }
        q_norm = rsqrtf(q_squares + 1.0e-6f) *
                 rsqrtf(static_cast<float>(shape.key_dim));
        k_norm = rsqrtf(k_squares + 1.0e-6f);
      }

      const float decay = expf(gate[tok * shape.v_heads + vh]);
      const float beta_value = beta[tok * shape.v_heads + vh];
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
      spec->tokens_in_persistent_loop, state_decay, update_scale,
      spec->qk_l2norm != 0);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
