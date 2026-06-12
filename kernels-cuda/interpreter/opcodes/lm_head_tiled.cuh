#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"

#include <cuda_bf16.h>
#include <stdint.h>

namespace qwen36_interpreter {

constexpr unsigned kLmHeadLogicalThreads = 256u;

__device__ inline void bf16_matvec_row_body(
    size_t row, const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ weight, __nv_bfloat16 *__restrict__ output,
    size_t in_features, float *scratch,
    unsigned logical_threads = kLmHeadLogicalThreads) {
  const unsigned tid = threadIdx.x;
  float sum = 0.0f;
  const __nv_bfloat16 *row_weight = weight + row * in_features;
  if (tid < logical_threads) {
    for (size_t col = tid; col < in_features; col += logical_threads) {
      sum += __bfloat162float(input[col]) * __bfloat162float(row_weight[col]);
    }
    scratch[tid] = sum;
  }
  __syncthreads();

  for (unsigned int stride = logical_threads / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[row] = __float2bfloat16(scratch[0]);
  }
}

__device__ inline void
exec_lm_head_tiled(const qwen36_interpreter_instruction_t &insn,
                   PageAllocator &, float *scratch) {
  const size_t out_features = static_cast<size_t>(insn.payload[0]);
  const size_t in_features = static_cast<size_t>(insn.payload[1]);
  const __nv_bfloat16 *input =
      payload_ptr<const __nv_bfloat16>(insn.payload[2]);
  const __nv_bfloat16 *weight =
      payload_ptr<const __nv_bfloat16>(insn.payload[3]);
  __nv_bfloat16 *output = payload_ptr<__nv_bfloat16>(insn.payload[4]);

  if (out_features == 0 || in_features == 0 || input == nullptr ||
      weight == nullptr || output == nullptr ||
      blockDim.x < kLmHeadLogicalThreads) {
    return;
  }

  for (size_t row = blockIdx.x; row < out_features; row += gridDim.x) {
    bf16_matvec_row_body(row, input, weight, output, in_features, scratch,
                         kLmHeadLogicalThreads);
  }
}

} // namespace qwen36_interpreter
