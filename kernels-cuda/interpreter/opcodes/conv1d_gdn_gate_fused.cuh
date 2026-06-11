#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace qwen36_interpreter {

__device__ inline void exec_conv1d_gdn_gate_fused(
    const qwen36_interpreter_instruction_t &insn, PageAllocator &) {
  const size_t channels = static_cast<size_t>(insn.payload[0]);
  const size_t kernel_size =
      static_cast<size_t>(insn.payload[1] & 0xffffffffu);
  const size_t heads = static_cast<size_t>(insn.payload[1] >> 32);
  const __nv_bfloat16 *conv_input =
      payload_ptr<const __nv_bfloat16>(insn.payload[2]);
  __nv_bfloat16 *history = payload_ptr<__nv_bfloat16>(insn.payload[3]);
  const __nv_bfloat16 *conv_weight =
      payload_ptr<const __nv_bfloat16>(insn.payload[4]);
  __nv_bfloat16 *conv_output =
      payload_ptr<__nv_bfloat16>(insn.payload[5]);
  const __nv_bfloat16 *gdn_a =
      payload_ptr<const __nv_bfloat16>(insn.payload[6]);
  const __nv_bfloat16 *gdn_b =
      payload_ptr<const __nv_bfloat16>(insn.payload[7]);
  const __nv_bfloat16 *gdn_a_log =
      payload_ptr<const __nv_bfloat16>(insn.payload[8]);
  const __nv_bfloat16 *gdn_dt_bias =
      payload_ptr<const __nv_bfloat16>(insn.payload[9]);
  float *gdn_gate = payload_ptr<float>(insn.payload[10]);
  float *gdn_beta = payload_ptr<float>(insn.payload[11]);

  if (channels == 0 || kernel_size < 2 || heads == 0 ||
      conv_input == nullptr || history == nullptr || conv_weight == nullptr ||
      conv_output == nullptr || gdn_a == nullptr || gdn_b == nullptr ||
      gdn_a_log == nullptr || gdn_dt_bias == nullptr || gdn_gate == nullptr ||
      gdn_beta == nullptr) {
    return;
  }

  const size_t tid =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t channel = tid; channel < channels; channel += stride) {
    __nv_bfloat16 *channel_history =
        history + channel * (kernel_size - 1);
    const __nv_bfloat16 *channel_weight =
        conv_weight + channel * kernel_size;
    float sum = __bfloat162float(conv_input[channel]) *
                __bfloat162float(channel_weight[kernel_size - 1]);
    for (size_t k = 0; k + 1 < kernel_size; ++k) {
      sum += __bfloat162float(channel_history[k]) *
             __bfloat162float(channel_weight[k]);
    }
    for (size_t k = 0; k + 2 < kernel_size; ++k) {
      channel_history[k] = channel_history[k + 1];
    }
    channel_history[kernel_size - 2] = conv_input[channel];
    const float silu = sum / (1.0f + expf(-sum));
    conv_output[channel] = __float2bfloat16(silu);
  }

  if (blockIdx.x == 0) {
    for (size_t idx = threadIdx.x; idx < heads; idx += blockDim.x) {
      const float x =
          __bfloat162float(gdn_a[idx]) + __bfloat162float(gdn_dt_bias[idx]);
      const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;
      gdn_gate[idx] = -expf(__bfloat162float(gdn_a_log[idx])) * softplus;
      const float b_value = __bfloat162float(gdn_b[idx]);
      gdn_beta[idx] = 1.0f / (1.0f + expf(-b_value));
    }
  }
}

} // namespace qwen36_interpreter
