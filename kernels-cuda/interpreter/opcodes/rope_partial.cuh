#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace qwen36_interpreter {

__device__ inline float unpack_payload_f32(uint64_t raw) {
  return __uint_as_float(static_cast<uint32_t>(raw & 0xffffffffu));
}

__device__ inline int32_t unpack_payload_i32_low(uint64_t raw) {
  return static_cast<int32_t>(static_cast<uint32_t>(raw & 0xffffffffu));
}

__device__ inline int32_t unpack_payload_i32_high(uint64_t raw) {
  return static_cast<int32_t>(static_cast<uint32_t>(raw >> 32));
}

__device__ inline void apply_rope_half_pair_body(__nv_bfloat16 *values,
                                                 size_t offset,
                                                 size_t half_dim,
                                                 size_t pair, float cosv,
                                                 float sinv) {
  const size_t first = offset + pair;
  const size_t second = offset + half_dim + pair;
  const float x0 = __bfloat162float(values[first]);
  const float x1 = __bfloat162float(values[second]);
  values[first] = __float2bfloat16(x0 * cosv - x1 * sinv);
  values[second] = __float2bfloat16(x1 * cosv + x0 * sinv);
}

__device__ inline void partial_rope_pair_body(
    size_t token, size_t pair, const int32_t *__restrict__ positions,
    int32_t scalar_position, int use_scalar_position,
    const int32_t *__restrict__ scalar_position_device, __nv_bfloat16 *q,
    __nv_bfloat16 *k, size_t q_heads, size_t kv_heads, size_t head_dim,
    size_t rope_dims, float base_theta) {
  const size_t half_dim = rope_dims / 2;
  const int32_t resolved_scalar =
      scalar_position_device != nullptr ? *scalar_position_device
                                        : scalar_position;
  const int32_t token_position =
      use_scalar_position != 0 ? resolved_scalar : positions[token];
  const float position = static_cast<float>(token_position);
  const float inv_freq = powf(
      base_theta, -static_cast<float>(2 * pair) / static_cast<float>(rope_dims));
  const float angle = position * inv_freq;
  const float cosv = cosf(angle);
  const float sinv = sinf(angle);

  for (size_t head = threadIdx.x; head < q_heads + kv_heads;
       head += blockDim.x) {
    if (head < q_heads) {
      const size_t head_offset = (token * q_heads + head) * head_dim;
      apply_rope_half_pair_body(q, head_offset, half_dim, pair, cosv, sinv);
    } else {
      const size_t kv_head = head - q_heads;
      const size_t head_offset = (token * kv_heads + kv_head) * head_dim;
      apply_rope_half_pair_body(k, head_offset, half_dim, pair, cosv, sinv);
    }
  }
}

__device__ inline void
exec_rope_partial(const qwen36_interpreter_instruction_t &insn,
                  PageAllocator &) {
  const size_t tokens = static_cast<size_t>(insn.payload[0]);
  const size_t q_heads = static_cast<size_t>(insn.payload[1]);
  const size_t kv_heads = static_cast<size_t>(insn.payload[2]);
  const size_t head_dim = static_cast<size_t>(insn.payload[3]);
  const size_t rope_dims = static_cast<size_t>(insn.payload[4]);
  const float raw_base_theta = unpack_payload_f32(insn.payload[5]);
  const float base_theta =
      raw_base_theta > 0.0f ? raw_base_theta : 10000.0f;
  const int32_t scalar_position = unpack_payload_i32_low(insn.payload[6]);
  const int use_scalar_position = unpack_payload_i32_high(insn.payload[6]);
  const int32_t *positions = payload_ptr<const int32_t>(insn.payload[7]);
  __nv_bfloat16 *q = payload_ptr<__nv_bfloat16>(insn.payload[8]);
  __nv_bfloat16 *k = payload_ptr<__nv_bfloat16>(insn.payload[9]);
  const int32_t *scalar_position_device =
      payload_ptr<const int32_t>(insn.payload[10]);

  if (tokens == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 ||
      rope_dims == 0 || rope_dims > head_dim || (rope_dims % 2) != 0 ||
      (positions == nullptr && use_scalar_position == 0) || q == nullptr ||
      k == nullptr) {
    return;
  }

  const size_t half_dim = rope_dims / 2;
  const size_t total_pairs = tokens * half_dim;
  for (size_t linear = blockIdx.x; linear < total_pairs; linear += gridDim.x) {
    const size_t token = linear / half_dim;
    const size_t pair = linear - token * half_dim;
    partial_rope_pair_body(token, pair, positions, scalar_position,
                           use_scalar_position, scalar_position_device, q, k,
                           q_heads, kv_heads, head_dim, rope_dims, base_theta);
  }
}

} // namespace qwen36_interpreter
