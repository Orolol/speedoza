#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace qwen36_interpreter {

__device__ inline void
exec_swiglu_bf16(const qwen36_interpreter_instruction_t &insn,
                 PageAllocator &) {
  const size_t rows = static_cast<size_t>(insn.payload[0]);
  const size_t intermediate = static_cast<size_t>(insn.payload[1]);
  const __nv_bfloat16 *gate =
      payload_ptr<const __nv_bfloat16>(insn.payload[2]);
  const __nv_bfloat16 *up = payload_ptr<const __nv_bfloat16>(insn.payload[3]);
  __nv_bfloat16 *output = payload_ptr<__nv_bfloat16>(insn.payload[4]);

  if (rows == 0 || intermediate == 0 || gate == nullptr || up == nullptr ||
      output == nullptr) {
    return;
  }

  const size_t total = rows * intermediate;
  const size_t tid =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t idx = tid; idx < total; idx += stride) {
    const float gate_value = __bfloat162float(gate[idx]);
    const float up_value = __bfloat162float(up[idx]);
    const float silu = gate_value / (1.0f + expf(-gate_value));
    output[idx] = __float2bfloat16(silu * up_value);
  }
}

} // namespace qwen36_interpreter
