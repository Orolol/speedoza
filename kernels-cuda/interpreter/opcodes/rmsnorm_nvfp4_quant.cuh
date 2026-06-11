#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace qwen36_interpreter {

constexpr unsigned int kRmsNormLogicalThreads = 256;

__host__ __device__ inline size_t div_ceil_size(size_t value, size_t divisor) {
  return (value + divisor - 1) / divisor;
}

__host__ __device__ inline size_t round_up_size(size_t value,
                                                size_t multiple) {
  return div_ceil_size(value, multiple) * multiple;
}

__host__ __device__ inline size_t vec16_scale_offset(size_t inner,
                                                     size_t outer,
                                                     size_t sf_inner_dim) {
  const size_t block_inner = (inner / 4) * 4;
  const size_t block_outer = outer / 128;
  const size_t block_offset = (block_inner + block_outer * sf_inner_dim) * 128;
  const size_t tile_outer = outer % 128;
  const size_t tile_inner = inner % 4;
  return block_offset + (tile_outer % 32) * 16 + (tile_outer / 32) * 4 +
         tile_inner;
}

__device__ inline float decode_e4m3(uint8_t code) {
  const int sign = (code & 0x80) != 0 ? -1 : 1;
  const int exponent = (code >> 3) & 0x0f;
  const int mantissa = code & 0x07;
  if (exponent == 0) {
    if (mantissa == 0) {
      return 0.0f;
    }
    return sign * ldexpf(static_cast<float>(mantissa) / 8.0f, -6);
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return sign * 448.0f;
  }
  return sign * ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f,
                       exponent - 7);
}

__device__ inline uint8_t encode_e4m3_positive(float value) {
  if (!(value > 0.0f)) {
    return 0;
  }
  if (value >= 448.0f) {
    return 0x7e;
  }

  constexpr float min_normal = 0x1p-6f;
  constexpr float subnormal_step = 0x1p-9f;
  constexpr float normal_boundary = (7.0f * subnormal_step + min_normal) * 0.5f;
  if (value < min_normal) {
    if (value >= normal_boundary) {
      return 0x08;
    }
    int mantissa =
        static_cast<int>(floorf(value / subnormal_step + 0.49999994f));
    if (mantissa <= 0) {
      return 0;
    }
    return static_cast<uint8_t>(mantissa);
  }

  const uint32_t bits = __float_as_uint(value);
  int exponent_field = static_cast<int>((bits >> 23) & 0xff) - 120;
  uint32_t mantissa = ((bits & 0x007fffffU) + 0x0007ffffU) >> 20;
  if (mantissa >= 8) {
    mantissa = 0;
    ++exponent_field;
  }
  if (exponent_field >= 15) {
    exponent_field = 15;
    if (mantissa > 6) {
      mantissa = 6;
    }
  }
  return static_cast<uint8_t>((exponent_field << 3) | mantissa);
}

__device__ inline uint8_t encode_e2m1(float value) {
  const bool negative = value < 0.0f;
  const float magnitude = fminf(fabsf(value), 6.0f);
  uint8_t best_index = 7;
  if (magnitude <= 0.25f) {
    best_index = 0;
  } else if (magnitude <= 0.75f) {
    best_index = 1;
  } else if (magnitude <= 1.25f) {
    best_index = 2;
  } else if (magnitude <= 1.75f) {
    best_index = 3;
  } else if (magnitude <= 2.5f) {
    best_index = 4;
  } else if (magnitude <= 3.5f) {
    best_index = 5;
  } else if (magnitude <= 5.0f) {
    best_index = 6;
  }
  return static_cast<uint8_t>((negative ? 0x08 : 0x00) | best_index);
}

__device__ inline void rmsnorm_nvfp4_quantize_body(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ residual,
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ output_bf16, uint8_t *__restrict__ output_fp4,
    uint8_t *__restrict__ output_scale, float *__restrict__ tensor_scale,
    size_t hidden, float eps, float input_tensor_scale, float *scratch,
    unsigned int logical_threads = blockDim.x) {
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

  const float norm_scale =
      rsqrtf(scratch[0] / static_cast<float>(hidden) + eps);
  const size_t groups = div_ceil_size(hidden, 16);
  const size_t scale_inner_dim = round_up_size(groups, 4);
  const float global_scale =
      input_tensor_scale > 0.0f ? input_tensor_scale : 1.0f;

  for (size_t group = threadIdx.x; active && group < groups;
       group += logical) {
    const size_t start = group * 16;
    float amax = 0.0f;
    float residual_values[16];
    float weighted_values[16];
    const size_t group_end = (start + 16 <= hidden) ? 16 : (hidden - start);
    if (group_end == 16) {
      const __nv_bfloat162 *input_pair =
          reinterpret_cast<const __nv_bfloat162 *>(input + start);
      const __nv_bfloat162 *residual_pair =
          residual != nullptr
              ? reinterpret_cast<const __nv_bfloat162 *>(residual + start)
              : nullptr;
      const __nv_bfloat162 *weight_pair =
          reinterpret_cast<const __nv_bfloat162 *>(weight + start);
      __nv_bfloat162 *output_pair =
          output_bf16 != nullptr
              ? reinterpret_cast<__nv_bfloat162 *>(output_bf16 + start)
              : nullptr;
#pragma unroll
      for (size_t p = 0; p < 8; ++p) {
        const __nv_bfloat162 ip = input_pair[p];
        float a = __low2float(ip);
        float b = __high2float(ip);
        if (residual_pair != nullptr) {
          const __nv_bfloat162 rp = residual_pair[p];
          a += __low2float(rp);
          b += __high2float(rp);
        }
        const __nv_bfloat162 wp = weight_pair[p];
        const float w0 = __low2float(wp);
        const float w1 = __high2float(wp);
        const float weighted0 = a * norm_scale * (1.0f + w0);
        const float weighted1 = b * norm_scale * (1.0f + w1);
        residual_values[p * 2] = a;
        residual_values[p * 2 + 1] = b;
        weighted_values[p * 2] = weighted0;
        weighted_values[p * 2 + 1] = weighted1;
        if (output_pair != nullptr) {
          output_pair[p] = __floats2bfloat162_rn(weighted0, weighted1);
        }
        amax = fmaxf(amax, fmaxf(fabsf(weighted0), fabsf(weighted1)));
      }
    } else {
      for (size_t offset = 0; offset < group_end; ++offset) {
        const size_t d = start + offset;
        float value = __bfloat162float(input[d]);
        if (residual != nullptr) {
          value += __bfloat162float(residual[d]);
        }
        const float weighted =
            value * norm_scale * (1.0f + __bfloat162float(weight[d]));
        residual_values[offset] = value;
        weighted_values[offset] = weighted;
        if (output_bf16 != nullptr) {
          output_bf16[d] = __float2bfloat16(weighted);
        }
        amax = fmaxf(amax, fabsf(weighted));
      }
    }

    const float scale_value =
        amax > 0.0f ? fmaxf(amax / (6.0f * global_scale), 1.0e-8f)
                    : 1.0f;
    const uint8_t scale_code = encode_e4m3_positive(scale_value);
    output_scale[vec16_scale_offset(group, 0, scale_inner_dim)] = scale_code;
    const float decoded_scale =
        fmaxf(decode_e4m3(scale_code) * global_scale, 1.0e-8f);

    for (size_t offset = 0; offset < 16 && start + offset < hidden;
         offset += 2) {
      const size_t d = start + offset;
      uint8_t packed = encode_e2m1(weighted_values[offset] / decoded_scale);
      if (d + 1 < hidden) {
        packed |= static_cast<uint8_t>(
            encode_e2m1(weighted_values[offset + 1] / decoded_scale) << 4);
      }
      output_fp4[d / 2] = packed;
    }
    if (residual_out != nullptr) {
      if (group_end == 16) {
        __nv_bfloat162 *resout_pair =
            reinterpret_cast<__nv_bfloat162 *>(residual_out + start);
#pragma unroll
        for (size_t p = 0; p < 8; ++p) {
          resout_pair[p] = __floats2bfloat162_rn(residual_values[p * 2],
                                                 residual_values[p * 2 + 1]);
        }
      } else {
        for (size_t offset = 0; offset < group_end; ++offset) {
          residual_out[start + offset] =
              __float2bfloat16(residual_values[offset]);
        }
      }
    }
  }

  if (threadIdx.x == 0 && tensor_scale != nullptr) {
    *tensor_scale = global_scale;
  }
}

__device__ inline float unpack_low_f32(uint64_t raw) {
  return __uint_as_float(static_cast<uint32_t>(raw & 0xffffffffu));
}

__device__ inline float unpack_high_f32(uint64_t raw) {
  return __uint_as_float(static_cast<uint32_t>(raw >> 32));
}

__device__ inline void
exec_rmsnorm_nvfp4_quant(const qwen36_interpreter_instruction_t &insn,
                         PageAllocator &, float *scratch) {
  if (blockIdx.x != 0) {
    return;
  }
  const size_t hidden = static_cast<size_t>(insn.payload[0]);
  const __nv_bfloat16 *input =
      payload_ptr<const __nv_bfloat16>(insn.payload[1]);
  const __nv_bfloat16 *weight =
      payload_ptr<const __nv_bfloat16>(insn.payload[2]);
  const __nv_bfloat16 *residual =
      payload_ptr<const __nv_bfloat16>(insn.payload[3]);
  __nv_bfloat16 *residual_out =
      payload_ptr<__nv_bfloat16>(insn.payload[4]);
  __nv_bfloat16 *output_bf16 =
      payload_ptr<__nv_bfloat16>(insn.payload[5]);
  uint8_t *output_fp4 = payload_ptr<uint8_t>(insn.payload[6]);
  uint8_t *output_scale = payload_ptr<uint8_t>(insn.payload[7]);
  float *tensor_scale = payload_ptr<float>(insn.payload[8]);
  const float eps = unpack_low_f32(insn.payload[9]);
  const float input_tensor_scale = unpack_high_f32(insn.payload[9]);

  if (hidden == 0 || input == nullptr || weight == nullptr ||
      output_fp4 == nullptr || output_scale == nullptr) {
    return;
  }
  rmsnorm_nvfp4_quantize_body(input, weight, residual, residual_out,
                              output_bf16, output_fp4, output_scale,
                              tensor_scale, hidden, eps, input_tensor_scale,
                              scratch, kRmsNormLogicalThreads);
}

} // namespace qwen36_interpreter
