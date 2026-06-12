#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"
#include "rmsnorm_nvfp4_quant.cuh"

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace qwen36_interpreter {

__device__ inline uint32_t unpack_high_u32(uint64_t raw) {
  return static_cast<uint32_t>(raw >> 32);
}

__device__ inline void rmsnorm_bf16_row_body(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ output, size_t hidden, float eps,
    int direct_weight, float *scratch, unsigned int logical_threads) {
  const unsigned int logical =
      (logical_threads == 0 || logical_threads > blockDim.x)
          ? static_cast<unsigned int>(blockDim.x)
          : logical_threads;
  const bool active = threadIdx.x < logical;
  float local_sum = 0.0f;
  const size_t pairs = hidden / 2;
  const __nv_bfloat162 *input2 =
      reinterpret_cast<const __nv_bfloat162 *>(input);
  const __nv_bfloat162 *residual2 =
      residual != nullptr ? reinterpret_cast<const __nv_bfloat162 *>(residual)
                          : nullptr;

  if (active) {
    for (size_t p = threadIdx.x; p < pairs; p += logical) {
      const __nv_bfloat162 vp = input2[p];
      float a = __low2float(vp);
      float b = __high2float(vp);
      if (residual2 != nullptr) {
        const __nv_bfloat162 rp = residual2[p];
        a += __low2float(rp);
        b += __high2float(rp);
      }
      local_sum += a * a + b * b;
    }
  }
  if ((hidden & 1u) != 0u && threadIdx.x == 0u) {
    const size_t d = hidden - 1;
    float value = __bfloat162float(input[d]);
    if (residual != nullptr) {
      value += __bfloat162float(residual[d]);
    }
    local_sum += value * value;
  }

  if (active) {
    scratch[threadIdx.x] = local_sum;
  }
  __syncthreads();
  for (unsigned int stride = logical / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float scale = rsqrtf(scratch[0] / static_cast<float>(hidden) + eps);
  __nv_bfloat162 *output2 = reinterpret_cast<__nv_bfloat162 *>(output);
  __nv_bfloat162 *residual_out2 =
      residual_out != nullptr
          ? reinterpret_cast<__nv_bfloat162 *>(residual_out)
          : nullptr;
  const __nv_bfloat162 *weight2 =
      reinterpret_cast<const __nv_bfloat162 *>(weight);

  if (active) {
    for (size_t p = threadIdx.x; p < pairs; p += logical) {
      const __nv_bfloat162 vp = input2[p];
      float a = __low2float(vp);
      float b = __high2float(vp);
      if (residual2 != nullptr) {
        const __nv_bfloat162 rp = residual2[p];
        a += __low2float(rp);
        b += __high2float(rp);
      }
      if (residual_out2 != nullptr) {
        residual_out2[p] = __floats2bfloat162_rn(a, b);
      }
      const __nv_bfloat162 wp = weight2[p];
      const float wa = __low2float(wp);
      const float wb = __high2float(wp);
      const float scale_a = direct_weight != 0 ? wa : (1.0f + wa);
      const float scale_b = direct_weight != 0 ? wb : (1.0f + wb);
      output2[p] =
          __floats2bfloat162_rn(a * scale * scale_a, b * scale * scale_b);
    }
  }
  if ((hidden & 1u) != 0u && threadIdx.x == 0u) {
    const size_t d = hidden - 1;
    float value = __bfloat162float(input[d]);
    if (residual != nullptr) {
      value += __bfloat162float(residual[d]);
    }
    if (residual_out != nullptr) {
      residual_out[d] = __float2bfloat16(value);
    }
    const float w = __bfloat162float(weight[d]);
    const float weighted =
        value * scale * (direct_weight != 0 ? w : (1.0f + w));
    output[d] = __float2bfloat16(weighted);
  }
}

__device__ inline void
exec_rmsnorm_bf16(const qwen36_interpreter_instruction_t &insn,
                  PageAllocator &, float *scratch) {
  const size_t rows = static_cast<size_t>(insn.payload[0]);
  const size_t hidden = static_cast<size_t>(insn.payload[1]);
  const __nv_bfloat16 *input =
      payload_ptr<const __nv_bfloat16>(insn.payload[2]);
  const __nv_bfloat16 *weight =
      payload_ptr<const __nv_bfloat16>(insn.payload[3]);
  const __nv_bfloat16 *residual =
      payload_ptr<const __nv_bfloat16>(insn.payload[4]);
  __nv_bfloat16 *residual_out =
      payload_ptr<__nv_bfloat16>(insn.payload[5]);
  __nv_bfloat16 *output = payload_ptr<__nv_bfloat16>(insn.payload[6]);
  const float eps = unpack_low_f32(insn.payload[7]);
  const int direct_weight = static_cast<int>(unpack_high_u32(insn.payload[7]));

  if (rows == 0 || hidden == 0 || input == nullptr || weight == nullptr ||
      output == nullptr) {
    return;
  }

  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const size_t offset = row * hidden;
    rmsnorm_bf16_row_body(
        input + offset, weight,
        residual != nullptr ? residual + offset : nullptr,
        residual_out != nullptr ? residual_out + offset : nullptr,
        output + offset, hidden, eps > 0.0f ? eps : 1.0e-6f, direct_weight,
        scratch, kRmsNormLogicalThreads);
    __syncthreads();
  }
}

} // namespace qwen36_interpreter
