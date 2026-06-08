#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace qwen36_interpreter {

constexpr int kAttentionDecodeMaxWarps = 16;

struct AttentionDecodeScratch {
  float warp_sums[kAttentionDecodeMaxWarps];
  float score_share;
  size_t shared_position;
};

__device__ inline void attention_decode_bf16_body(
    size_t qh, const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k_new,
    const __nv_bfloat16 *__restrict__ v_new,
    __nv_bfloat16 *__restrict__ cache_k,
    __nv_bfloat16 *__restrict__ cache_v, __nv_bfloat16 *__restrict__ output,
    size_t position_scalar, const int32_t *__restrict__ position_device,
    qwen36_attention_shape_t shape, AttentionDecodeScratch &scratch) {
  if (qh >= shape.q_heads || shape.kv_heads == 0 || shape.head_dim == 0) {
    return;
  }

  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t kvh = qh / q_per_kv;
  const float scale = rsqrtf(static_cast<float>(shape.head_dim));
  const size_t d = threadIdx.x;
  const bool active = d < shape.head_dim;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31u;
  const unsigned n_warps = (blockDim.x + 31u) >> 5;

  if (threadIdx.x == 0) {
    scratch.shared_position =
        position_device != nullptr ? static_cast<size_t>(*position_device)
                                   : position_scalar;
  }
  __syncthreads();
  const size_t position = scratch.shared_position;

  const float q_val =
      active ? __bfloat162float(q[qh * shape.head_dim + d]) : 0.0f;
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
      const float kv =
          is_new ? k_new_val
                 : __bfloat162float(cache_k[(t * shape.kv_heads + kvh) *
                                                shape.head_dim +
                                            d]);
      local = q_val * kv;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
      local += __shfl_xor_sync(0xffffffff, local, offset);
    }
    if (lane_id == 0) {
      scratch.warp_sums[warp_id] = local;
    }
    __syncthreads();
    if (warp_id == 0) {
      float total = (lane_id < n_warps) ? scratch.warp_sums[lane_id] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffff, total, offset);
      }
      if (lane_id == 0) {
        scratch.score_share = total * scale;
      }
    }
    __syncthreads();
    const float score = scratch.score_share;

    const float new_max = fmaxf(max_score, score);
    const float old_scale =
        isinf(max_score) && max_score < 0.0f ? 0.0f : expf(max_score - new_max);
    const float score_scale = expf(score - new_max);
    if (active) {
      const float vv =
          is_new ? v_new_val
                 : __bfloat162float(cache_v[(t * shape.kv_heads + kvh) *
                                                shape.head_dim +
                                            d]);
      acc = acc * old_scale + score_scale * vv;
    }
    denom = denom * old_scale + score_scale;
    max_score = new_max;
  }

  if (active) {
    output[qh * shape.head_dim + d] = __float2bfloat16(acc / denom);
    if (qh % q_per_kv == 0) {
      const size_t cache_off =
          (position * shape.kv_heads + kvh) * shape.head_dim + d;
      cache_k[cache_off] = k_new[kvh * shape.head_dim + d];
      cache_v[cache_off] = v_new[kvh * shape.head_dim + d];
    }
  }
}

__device__ inline bool
attention_decode_bf16_spec_valid(const qwen36_attention_decode_spec_t &spec) {
  return spec.q_bf16.ptr != 0 && spec.k_bf16.ptr != 0 &&
         spec.v_bf16.ptr != 0 && spec.kv_cache_k.ptr != 0 &&
         spec.kv_cache_v.ptr != 0 && spec.output_bf16.ptr != 0 &&
         spec.shape.q_heads != 0 && spec.shape.kv_heads != 0 &&
         spec.shape.head_dim != 0 && spec.shape.head_dim <= 256 &&
         spec.shape.q_heads % spec.shape.kv_heads == 0 &&
         spec.kv_cache_dtype == 0 && spec.partial_acc_f32.ptr == 0 &&
         spec.partial_max_f32.ptr == 0 && spec.partial_denom_f32.ptr == 0 &&
         spec.decode_n_splits <= 1;
}

__device__ inline void
exec_attn_decode_full(const qwen36_interpreter_instruction_t &insn,
                      PageAllocator &, AttentionDecodeScratch &scratch) {
  const qwen36_attention_decode_spec_t *spec_ptr =
      payload_ptr<const qwen36_attention_decode_spec_t>(insn.payload[0]);
  if (spec_ptr == nullptr) {
    return;
  }
  const qwen36_attention_decode_spec_t spec = *spec_ptr;
  if (!attention_decode_bf16_spec_valid(spec)) {
    return;
  }

  for (size_t qh = blockIdx.x; qh < spec.shape.q_heads; qh += gridDim.x) {
    attention_decode_bf16_body(
        qh, payload_ptr<const __nv_bfloat16>(spec.q_bf16.ptr),
        payload_ptr<const __nv_bfloat16>(spec.k_bf16.ptr),
        payload_ptr<const __nv_bfloat16>(spec.v_bf16.ptr),
        payload_ptr<__nv_bfloat16>(spec.kv_cache_k.ptr),
        payload_ptr<__nv_bfloat16>(spec.kv_cache_v.ptr),
        payload_ptr<__nv_bfloat16>(spec.output_bf16.ptr), spec.position,
        payload_ptr<const int32_t>(spec.position_device_i32.ptr), spec.shape,
        scratch);
    __syncthreads();
  }
}

} // namespace qwen36_interpreter
