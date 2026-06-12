#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"

#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace qwen36_interpreter {

constexpr int kDeltaNetLogicalThreads = 128;
constexpr int kDeltaNetMaxPhysicalThreads = 512;

struct DeltaNetScratch {
  float shared_q[256];
  float shared_k[256];
  float reduction_q[kDeltaNetMaxPhysicalThreads];
  float reduction_k[kDeltaNetMaxPhysicalThreads];
  float shared_decay;
  float shared_beta;
  float shared_q_norm;
  float shared_k_norm;
};

__device__ inline size_t deltanet_div_ceil_size(size_t value,
                                                size_t divisor) {
  return (value + divisor - 1) / divisor;
}

__device__ inline void deltanet_decode_body(
    size_t value_tile, size_t vh, const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k, const __nv_bfloat16 *__restrict__ v,
    const float *__restrict__ gate, const float *__restrict__ beta,
    __nv_bfloat16 *__restrict__ state, __nv_bfloat16 *__restrict__ output,
    qwen36_deltanet_shape_t shape, size_t tokens, size_t q_token_stride,
    size_t k_token_stride, size_t v_token_stride, float state_decay,
    float update_scale, bool qk_l2norm, DeltaNetScratch &scratch) {
  if (vh >= shape.v_heads) {
    return;
  }

  const bool logical_thread = threadIdx.x < kDeltaNetLogicalThreads;
  const size_t vd = value_tile * static_cast<size_t>(kDeltaNetLogicalThreads) +
                    threadIdx.x;
  const bool active = logical_thread && vd < shape.value_dim;
  const bool gated = (gate != nullptr) && (beta != nullptr);

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

  const size_t key_dim = shape.key_dim;
  const bool vector_state =
      active && (key_dim % 8 == 0) &&
      ((reinterpret_cast<uintptr_t>(state_row) & 15u) == 0);

  for (size_t tok = 0; tok < tokens; ++tok) {
    const __nv_bfloat16 *q_tok = q + tok * q_stride + qh * shape.key_dim;
    const __nv_bfloat16 *k_tok = k + tok * k_stride + qh * shape.key_dim;
    const float v_value =
        active ? __bfloat162float(v[tok * v_stride + vh * shape.value_dim + vd])
               : 0.0f;

    float q_sq = 0.0f;
    float k_sq = 0.0f;
    for (size_t kd = threadIdx.x; kd < key_dim; kd += blockDim.x) {
      const float qv = __bfloat162float(q_tok[kd]);
      const float kv = __bfloat162float(k_tok[kd]);
      scratch.shared_q[kd] = qv;
      scratch.shared_k[kd] = kv;
      q_sq += qv * qv;
      k_sq += kv * kv;
    }
    if (gated && threadIdx.x == 0) {
      scratch.shared_decay = expf(gate[tok * shape.v_heads + vh]);
      scratch.shared_beta = beta[tok * shape.v_heads + vh];
    }

    if (gated) {
      scratch.reduction_q[threadIdx.x] = q_sq;
      scratch.reduction_k[threadIdx.x] = k_sq;
      __syncthreads();
      for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
          scratch.reduction_q[threadIdx.x] += scratch.reduction_q[threadIdx.x + s];
          scratch.reduction_k[threadIdx.x] += scratch.reduction_k[threadIdx.x + s];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        const float q_squares = scratch.reduction_q[0];
        const float k_squares = scratch.reduction_k[0];
        if (qk_l2norm) {
          scratch.shared_q_norm = rsqrtf(q_squares + 1.0e-6f) *
                                  rsqrtf(static_cast<float>(key_dim));
          scratch.shared_k_norm = rsqrtf(k_squares + 1.0e-6f);
        } else {
          scratch.shared_q_norm = rsqrtf(static_cast<float>(key_dim));
          scratch.shared_k_norm = 1.0f;
        }
      }
      __syncthreads();
      if (!active) {
        if (tok + 1 < tokens) {
          __syncthreads();
        }
        continue;
      }

      const float decay = scratch.shared_decay;
      const float beta_value = scratch.shared_beta;
      const float q_norm = scratch.shared_q_norm;
      const float k_norm = scratch.shared_k_norm;

      float kv_mem = 0.0f;
      float s_q = 0.0f;
      float k_q = 0.0f;
      if (vector_state) {
        const uint4 *state_vec_in =
            reinterpret_cast<const uint4 *>(state_row);
        const size_t vec_count = key_dim / 8;
#pragma unroll 1
        for (size_t i = 0; i < vec_count; ++i) {
          const uint4 chunk = state_vec_in[i];
          const __nv_bfloat162 *as_pairs =
              reinterpret_cast<const __nv_bfloat162 *>(&chunk);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float s0 = __bfloat162float(as_pairs[j].x);
            const float s1 = __bfloat162float(as_pairs[j].y);
            const size_t kd0 = i * 8 + 2 * j;
            const float decayed0 = s0 * decay;
            const float decayed1 = s1 * decay;
            const float k0 = scratch.shared_k[kd0] * k_norm;
            const float k1 = scratch.shared_k[kd0 + 1] * k_norm;
            const float q0 = scratch.shared_q[kd0] * q_norm;
            const float q1 = scratch.shared_q[kd0 + 1] * q_norm;
            kv_mem += decayed0 * k0 + decayed1 * k1;
            s_q += decayed0 * q0 + decayed1 * q1;
            k_q += k0 * q0 + k1 * q1;
          }
        }
      } else {
        for (size_t kd = 0; kd < key_dim; ++kd) {
          const float decayed = __bfloat162float(state_row[kd]) * decay;
          const float key = scratch.shared_k[kd] * k_norm;
          const float query = scratch.shared_q[kd] * q_norm;
          kv_mem += decayed * key;
          s_q += decayed * query;
          k_q += key * query;
        }
      }

      const float delta = (v_value - kv_mem) * beta_value;
      const float acc = s_q + delta * k_q;
      output[(tok * shape.v_heads + vh) * shape.value_dim + vd] =
          __float2bfloat16(acc);

      if (vector_state) {
        const uint4 *state_vec_in =
            reinterpret_cast<const uint4 *>(state_row);
        uint4 *state_vec_out = reinterpret_cast<uint4 *>(state_row);
        const size_t vec_count = key_dim / 8;
#pragma unroll 1
        for (size_t i = 0; i < vec_count; ++i) {
          const uint4 chunk = state_vec_in[i];
          const __nv_bfloat162 *as_pairs =
              reinterpret_cast<const __nv_bfloat162 *>(&chunk);
          __nv_bfloat162 out_pairs[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float s0 = __bfloat162float(as_pairs[j].x);
            const float s1 = __bfloat162float(as_pairs[j].y);
            const size_t kd0 = i * 8 + 2 * j;
            const float k0 = scratch.shared_k[kd0] * k_norm;
            const float k1 = scratch.shared_k[kd0 + 1] * k_norm;
            const float new0 = s0 * decay + k0 * delta;
            const float new1 = s1 * decay + k1 * delta;
            out_pairs[j] = __halves2bfloat162(__float2bfloat16(new0),
                                               __float2bfloat16(new1));
          }
          uint4 packed;
          memcpy(&packed, out_pairs, sizeof(packed));
          state_vec_out[i] = packed;
        }
      } else {
        for (size_t kd = 0; kd < key_dim; ++kd) {
          const float decayed = __bfloat162float(state_row[kd]) * decay;
          const float key = scratch.shared_k[kd] * k_norm;
          state_row[kd] = __float2bfloat16(decayed + key * delta);
        }
      }

      if (tok + 1 < tokens) {
        __syncthreads();
      }
      continue;
    }

    __syncthreads();
    if (!active) {
      if (tok + 1 < tokens) {
        __syncthreads();
      }
      continue;
    }
    float acc = 0.0f;
    for (size_t kd = 0; kd < key_dim; ++kd) {
      const float previous = __bfloat162float(state_row[kd]);
      const float key = scratch.shared_k[kd];
      const float updated =
          previous * state_decay + update_scale * v_value * key;
      state_row[kd] = __float2bfloat16(updated);
      acc += updated * scratch.shared_q[kd];
    }
    output[(tok * shape.v_heads + vh) * shape.value_dim + vd] =
        __float2bfloat16(acc);
    if (tok + 1 < tokens) {
      __syncthreads();
    }
  }
}

__device__ inline bool
deltanet_spec_valid(const qwen36_deltanet_decode_spec_t &spec) {
  return spec.q_bf16.ptr != 0 && spec.k_bf16.ptr != 0 &&
         spec.v_bf16.ptr != 0 && spec.state_bf16.ptr != 0 &&
         spec.output_bf16.ptr != 0 && spec.shape.qk_heads != 0 &&
         spec.shape.v_heads != 0 && spec.shape.key_dim != 0 &&
         spec.shape.key_dim <= 256 && spec.shape.value_dim != 0 &&
         spec.tokens_in_persistent_loop != 0 &&
         spec.shape.v_heads % spec.shape.qk_heads == 0 &&
         ((spec.gate_f32.ptr == 0) == (spec.beta_f32.ptr == 0));
}

__device__ inline void
exec_deltanet_recur(const qwen36_interpreter_instruction_t &insn,
                    PageAllocator &, DeltaNetScratch &scratch) {
  const qwen36_deltanet_decode_spec_t *spec_ptr =
      payload_ptr<const qwen36_deltanet_decode_spec_t>(insn.payload[0]);
  if (spec_ptr == nullptr) {
    return;
  }

  const qwen36_deltanet_decode_spec_t spec = *spec_ptr;
  if (!deltanet_spec_valid(spec)) {
    return;
  }

  const size_t v_tiles =
      deltanet_div_ceil_size(spec.shape.value_dim, kDeltaNetLogicalThreads);
  const size_t total_tiles = v_tiles * spec.shape.v_heads;
  const float state_decay = spec.state_decay == 0.0f ? 1.0f : spec.state_decay;
  const float update_scale =
      spec.update_scale == 0.0f ? 1.0f : spec.update_scale;

  for (size_t linear = blockIdx.x; linear < total_tiles; linear += gridDim.x) {
    const size_t vh = linear / v_tiles;
    const size_t value_tile = linear - vh * v_tiles;
    deltanet_decode_body(
        value_tile, vh, payload_ptr<const __nv_bfloat16>(spec.q_bf16.ptr),
        payload_ptr<const __nv_bfloat16>(spec.k_bf16.ptr),
        payload_ptr<const __nv_bfloat16>(spec.v_bf16.ptr),
        payload_ptr<const float>(spec.gate_f32.ptr),
        payload_ptr<const float>(spec.beta_f32.ptr),
        payload_ptr<__nv_bfloat16>(spec.state_bf16.ptr),
        payload_ptr<__nv_bfloat16>(spec.output_bf16.ptr), spec.shape,
        spec.tokens_in_persistent_loop, spec.q_token_stride,
        spec.k_token_stride, spec.v_token_stride, state_decay, update_scale,
        spec.qk_l2norm != 0, scratch);
    __syncthreads();
  }
}

} // namespace qwen36_interpreter
