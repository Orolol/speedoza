#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace qwen36_interpreter {

__device__ inline void
exec_q_proj_deinterleave(const qwen36_interpreter_instruction_t &insn,
                         PageAllocator &) {
  const size_t rows = static_cast<size_t>(insn.payload[0]);
  const size_t heads = static_cast<size_t>(insn.payload[1]);
  const size_t head_dim = static_cast<size_t>(insn.payload[2]);
  const __nv_bfloat16 *input =
      payload_ptr<const __nv_bfloat16>(insn.payload[3]);
  __nv_bfloat16 *output = payload_ptr<__nv_bfloat16>(insn.payload[4]);

  if (rows == 0 || heads == 0 || head_dim == 0 || input == nullptr ||
      output == nullptr) {
    return;
  }

  const size_t q_values = heads * head_dim;
  const size_t row_stride = q_values * 2;
  const size_t total = rows * q_values;
  const size_t tid =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t idx = tid; idx < total; idx += stride) {
    const size_t row = idx / q_values;
    const size_t col = idx - row * q_values;
    const size_t head = col / head_dim;
    const size_t dim = col - head * head_dim;
    output[idx] = input[row * row_stride + head * head_dim * 2 + dim];
  }
}

__device__ inline void
exec_q_proj_sigmoid_gate(const qwen36_interpreter_instruction_t &insn,
                         PageAllocator &) {
  const size_t rows = static_cast<size_t>(insn.payload[0]);
  const size_t heads = static_cast<size_t>(insn.payload[1]);
  const size_t head_dim = static_cast<size_t>(insn.payload[2]);
  const __nv_bfloat16 *gate =
      payload_ptr<const __nv_bfloat16>(insn.payload[3]);
  const __nv_bfloat16 *input =
      payload_ptr<const __nv_bfloat16>(insn.payload[4]);
  __nv_bfloat16 *output = payload_ptr<__nv_bfloat16>(insn.payload[5]);

  if (rows == 0 || heads == 0 || head_dim == 0 || gate == nullptr ||
      input == nullptr || output == nullptr) {
    return;
  }

  const size_t q_values = heads * head_dim;
  const size_t row_stride = q_values * 2;
  const size_t total = rows * q_values;
  const size_t tid =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t idx = tid; idx < total; idx += stride) {
    const size_t row = idx / q_values;
    const size_t col = idx - row * q_values;
    const size_t head = col / head_dim;
    const size_t dim = col - head * head_dim;
    const float gate_value = __bfloat162float(
        gate[row * row_stride + head * head_dim * 2 + head_dim + dim]);
    const float input_value = __bfloat162float(input[idx]);
    output[idx] = __float2bfloat16(input_value / (1.0f + expf(-gate_value)));
  }
}

} // namespace qwen36_interpreter
