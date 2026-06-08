#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"
#include "rmsnorm_nvfp4_quant.cuh"

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace qwen36_interpreter {

__device__ inline void swiglu_nvfp4_quantize_group_body(
    size_t group, const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up, uint8_t *__restrict__ output_fp4,
    uint8_t *__restrict__ output_scale, float *__restrict__ tensor_scale,
    size_t values, float input_tensor_scale, float *scratch,
    float *decoded_scale, float *staged) {
  const float global_scale =
      input_tensor_scale > 0.0f ? input_tensor_scale : 1.0f;
  const size_t scale_inner_dim = round_up_size(div_ceil_size(values, 16), 4);
  const size_t start = group * 16;

  float local_amax = 0.0f;
  if (threadIdx.x < 16) {
    const size_t idx = start + threadIdx.x;
    if (idx < values) {
      const float g = __bfloat162float(gate[idx]);
      const float u = __bfloat162float(up[idx]);
      const float silu = g / (1.0f + expf(-g));
      const float y = silu * u;
      staged[threadIdx.x] = y;
      local_amax = fabsf(y);
    } else {
      staged[threadIdx.x] = 0.0f;
    }
  }

  scratch[threadIdx.x] = local_amax;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] =
          fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const float scale_value =
        scratch[0] > 0.0f
            ? fmaxf(scratch[0] / (6.0f * global_scale), 1.0e-8f)
            : 1.0f;
    const uint8_t scale_code = encode_e4m3_positive(scale_value);
    output_scale[vec16_scale_offset(group, 0, scale_inner_dim)] = scale_code;
    *decoded_scale = fmaxf(decode_e4m3(scale_code) * global_scale, 1.0e-8f);
    if (group == 0 && tensor_scale != nullptr) {
      *tensor_scale = global_scale;
    }
  }
  __syncthreads();

  if (threadIdx.x < 8) {
    const size_t col = start + threadIdx.x * 2;
    if (col < values) {
      const float v0 = staged[threadIdx.x * 2] / *decoded_scale;
      uint8_t packed = encode_e2m1(v0);
      if (col + 1 < values) {
        const float v1 = staged[threadIdx.x * 2 + 1] / *decoded_scale;
        packed |= static_cast<uint8_t>(encode_e2m1(v1) << 4);
      }
      output_fp4[col / 2] = packed;
    }
  }
  __syncthreads();
}

__device__ inline void
exec_swiglu_nvfp4_quant(const qwen36_interpreter_instruction_t &insn,
                        PageAllocator &, float *scratch,
                        float *decoded_scale, float *staged) {
  const size_t values = static_cast<size_t>(insn.payload[0]);
  const __nv_bfloat16 *gate =
      payload_ptr<const __nv_bfloat16>(insn.payload[1]);
  const __nv_bfloat16 *up = payload_ptr<const __nv_bfloat16>(insn.payload[2]);
  uint8_t *output_fp4 = payload_ptr<uint8_t>(insn.payload[3]);
  uint8_t *output_scale = payload_ptr<uint8_t>(insn.payload[4]);
  float *tensor_scale = payload_ptr<float>(insn.payload[5]);
  const float input_tensor_scale = unpack_low_f32(insn.payload[6]);

  if (values == 0 || gate == nullptr || up == nullptr ||
      output_fp4 == nullptr || output_scale == nullptr) {
    return;
  }

  const size_t groups = div_ceil_size(values, 16);
  for (size_t group = blockIdx.x; group < groups; group += gridDim.x) {
    swiglu_nvfp4_quantize_group_body(group, gate, up, output_fp4, output_scale,
                                     tensor_scale, values, input_tensor_scale,
                                     scratch, decoded_scale, staged);
  }
}

} // namespace qwen36_interpreter
