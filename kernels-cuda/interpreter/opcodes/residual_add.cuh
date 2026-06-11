#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"

#include <cuda_bf16.h>
#include <stdint.h>

namespace qwen36_interpreter {

__device__ inline void
exec_residual_add(const qwen36_interpreter_instruction_t &insn,
                  PageAllocator &) {
  const uint64_t values = insn.payload[0];
  const __nv_bfloat16 *__restrict__ input =
      payload_ptr<const __nv_bfloat16>(insn.payload[1]);
  const __nv_bfloat16 *__restrict__ residual =
      payload_ptr<const __nv_bfloat16>(insn.payload[2]);
  __nv_bfloat16 *__restrict__ output =
      payload_ptr<__nv_bfloat16>(insn.payload[3]);

  if (values == 0 || input == nullptr || residual == nullptr) {
    return;
  }
  if (output == nullptr) {
    output = const_cast<__nv_bfloat16 *>(residual);
  }

  const uint64_t tid =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t idx = tid; idx < values; idx += stride) {
    const float a = __bfloat162float(input[idx]);
    const float r = __bfloat162float(residual[idx]);
    output[idx] = __float2bfloat16(a + r);
  }
}

} // namespace qwen36_interpreter
