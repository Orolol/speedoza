#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

template <typename T> T *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

__device__ bool better_score(float candidate, uint32_t candidate_idx,
                             float current, uint32_t current_idx) {
  return candidate > current ||
         (candidate == current && candidate_idx < current_idx);
}

__device__ float decode_e2m1(uint8_t packed, bool high_nibble) {
  const uint8_t code = high_nibble ? (packed >> 4) : (packed & 0x0f);
  const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                           2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = values[code & 0x07];
  return (code & 0x08) != 0 ? -magnitude : magnitude;
}

__device__ float decode_e4m3(uint8_t code) {
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

__host__ __device__ size_t div_ceil_size(size_t value, size_t divisor) {
  return (value + divisor - 1) / divisor;
}

__host__ __device__ size_t round_up_size(size_t value, size_t multiple) {
  return div_ceil_size(value, multiple) * multiple;
}

__host__ __device__ size_t vec16_scale_offset(size_t inner, size_t outer,
                                              size_t sf_inner_dim) {
  const size_t block_inner = (inner / 4) * 4;
  const size_t block_outer = outer / 128;
  const size_t block_offset = (block_inner + block_outer * sf_inner_dim) * 128;
  const size_t tile_outer = outer % 128;
  const size_t tile_inner = inner % 4;
  return block_offset + (tile_outer % 32) * 16 + (tile_outer / 32) * 4 +
         tile_inner;
}

__host__ __device__ size_t vec16_scale_bytes(size_t inner_groups,
                                             size_t outer_values) {
  const size_t sf_inner_dim = round_up_size(inner_groups, 4);
  return div_ceil_size(outer_values, 128) * (sf_inner_dim / 4) * 512;
}

__device__ uint8_t encode_e4m3_positive(float value) {
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
    int mantissa = static_cast<int>(floorf(value / subnormal_step + 0.49999994f));
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

__device__ uint8_t encode_e2m1(float value) {
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

__global__ void rmsnorm_kernel(const __nv_bfloat16 *input,
                               const __nv_bfloat16 *weight,
                               const __nv_bfloat16 *residual,
                               __nv_bfloat16 *residual_out,
                               __nv_bfloat16 *output, size_t hidden,
                               float eps, int direct_weight) {
  extern __shared__ float scratch[];
  const size_t row = blockIdx.x;
  float local_sum = 0.0f;

  for (size_t d = threadIdx.x; d < hidden; d += blockDim.x) {
    const size_t offset = row * hidden + d;
    float value = __bfloat162float(input[offset]);
    if (residual != nullptr) {
      value += __bfloat162float(residual[offset]);
    }
    local_sum += value * value;
  }

  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float scale = rsqrtf(scratch[0] / static_cast<float>(hidden) + eps);
  for (size_t d = threadIdx.x; d < hidden; d += blockDim.x) {
    const size_t offset = row * hidden + d;
    float value = __bfloat162float(input[offset]);
    if (residual != nullptr) {
      value += __bfloat162float(residual[offset]);
    }
    if (residual_out != nullptr) {
      residual_out[offset] = __float2bfloat16(value);
    }
    const float w = __bfloat162float(weight[d]);
    const float weighted = value * scale * (direct_weight != 0 ? w : (1.0f + w));
    output[offset] = __float2bfloat16(weighted);
  }
}

__global__ void rmsnorm_nvfp4_quantize_kernel(
    const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
    const __nv_bfloat16 *residual, __nv_bfloat16 *residual_out,
    __nv_bfloat16 *output_bf16, uint8_t *output_fp4,
    uint8_t *output_scale, float *tensor_scale, size_t hidden, float eps,
    float input_tensor_scale) {
  extern __shared__ float scratch[];
  float local_sum = 0.0f;

  for (size_t d = threadIdx.x; d < hidden; d += blockDim.x) {
    float value = __bfloat162float(input[d]);
    if (residual != nullptr) {
      value += __bfloat162float(residual[d]);
    }
    local_sum += value * value;
  }

  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
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

  for (size_t group = threadIdx.x; group < groups; group += blockDim.x) {
    const size_t start = group * 16;
    float amax = 0.0f;
    float residual_values[16];
    float weighted_values[16];
    for (size_t offset = 0; offset < 16 && start + offset < hidden; ++offset) {
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
    for (size_t offset = 0; offset < 16 && start + offset < hidden; ++offset) {
      if (residual_out != nullptr) {
        residual_out[start + offset] = __float2bfloat16(residual_values[offset]);
      }
    }
  }

  if (threadIdx.x == 0 && tensor_scale != nullptr) {
    *tensor_scale = global_scale;
  }
}

__device__ void apply_rope_half_pair(__nv_bfloat16 *values, size_t offset,
                                     size_t half_dim, size_t pair, float cosv,
                                     float sinv) {
  const size_t first = offset + pair;
  const size_t second = offset + half_dim + pair;
  const float x0 = __bfloat162float(values[first]);
  const float x1 = __bfloat162float(values[second]);
  values[first] = __float2bfloat16(x0 * cosv - x1 * sinv);
  values[second] = __float2bfloat16(x1 * cosv + x0 * sinv);
}

__global__ void partial_rope_kernel(int32_t const *positions,
                                    int32_t scalar_position,
                                    int use_scalar_position,
                                    const int32_t *scalar_position_device,
                                    __nv_bfloat16 *q, __nv_bfloat16 *k,
                                    size_t q_heads, size_t kv_heads,
                                    size_t head_dim, size_t rope_dims,
                                    float base_theta) {
  const size_t half_dim = rope_dims / 2;
  const size_t token = blockIdx.y;
  const size_t p = blockIdx.x;
  // Resolve the scalar position from device memory when available so a
  // captured CUDA graph can step through positions without re-recording.
  const int32_t resolved_scalar = scalar_position_device != nullptr
                                      ? *scalar_position_device
                                      : scalar_position;
  const int32_t token_position =
      use_scalar_position != 0 ? resolved_scalar : positions[token];
  const float position = static_cast<float>(token_position);
  const float inv_freq = powf(
      base_theta, -static_cast<float>(2 * p) / static_cast<float>(rope_dims));
  const float angle = position * inv_freq;
  const float cosv = cosf(angle);
  const float sinv = sinf(angle);
  for (size_t head = threadIdx.x; head < q_heads + kv_heads;
       head += blockDim.x) {
    if (head < q_heads) {
      const size_t q_head = head;
      const size_t head_offset = (token * q_heads + q_head) * head_dim;
      apply_rope_half_pair(q, head_offset, half_dim, p, cosv, sinv);
    } else {
      const size_t kv_head = head - q_heads;
      const size_t head_offset = (token * kv_heads + kv_head) * head_dim;
      apply_rope_half_pair(k, head_offset, half_dim, p, cosv, sinv);
    }
  }
}

__global__ void swiglu_kernel(const __nv_bfloat16 *gate,
                              const __nv_bfloat16 *up,
                              __nv_bfloat16 *output, size_t total) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    const float gate_value = __bfloat162float(gate[idx]);
    const float up_value = __bfloat162float(up[idx]);
    const float silu = gate_value / (1.0f + expf(-gate_value));
    output[idx] = __float2bfloat16(silu * up_value);
  }
}

__global__ void sample_argmax_kernel(const __nv_bfloat16 *logits,
                                     uint32_t *output_token,
                                     size_t vocab_size, float temperature) {
  extern __shared__ unsigned char shared_bytes[];
  float *scores = reinterpret_cast<float *>(shared_bytes);
  uint32_t *indices = reinterpret_cast<uint32_t *>(scores + blockDim.x);

  float best_score = -INFINITY;
  uint32_t best_idx = 0;
  const float temp = temperature > 0.0f ? temperature : 1.0f;
  for (size_t idx = threadIdx.x; idx < vocab_size; idx += blockDim.x) {
    const float score = __bfloat162float(logits[idx]) / temp;
    const uint32_t token = static_cast<uint32_t>(idx);
    if (better_score(score, token, best_score, best_idx)) {
      best_score = score;
      best_idx = token;
    }
  }

  scores[threadIdx.x] = best_score;
  indices[threadIdx.x] = best_idx;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride &&
        better_score(scores[threadIdx.x + stride],
                     indices[threadIdx.x + stride], scores[threadIdx.x],
                     indices[threadIdx.x])) {
      scores[threadIdx.x] = scores[threadIdx.x + stride];
      indices[threadIdx.x] = indices[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *output_token = indices[0];
  }
}

__global__ void embedding_lookup_kernel(const uint32_t *token_ids,
                                        const __nv_bfloat16 *embedding,
                                        __nv_bfloat16 *output, size_t tokens,
                                        size_t hidden, size_t vocab_size) {
  const size_t token_index = blockIdx.y;
  const uint32_t token = token_ids[token_index];
  if (token >= vocab_size) {
    return;
  }
  for (size_t d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden;
       d += blockDim.x * gridDim.x) {
    output[token_index * hidden + d] = embedding[static_cast<size_t>(token) *
                                                 hidden + d];
  }
}

__global__ void bf16_matvec_kernel(const __nv_bfloat16 *input,
                                   const __nv_bfloat16 *weight,
                                   __nv_bfloat16 *output, size_t in_features) {
  extern __shared__ float scratch[];
  const size_t row = blockIdx.x;
  float sum = 0.0f;
  const __nv_bfloat16 *row_weight = weight + row * in_features;
  for (size_t col = threadIdx.x; col < in_features; col += blockDim.x) {
    sum += __bfloat162float(input[col]) * __bfloat162float(row_weight[col]);
  }
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[row] = __float2bfloat16(scratch[0]);
  }
}

// Lookup table for E2M1 (FP4) decode. Index 0..7 = positive magnitudes,
// 8..15 = the same magnitudes negated. Lives in constant memory so every
// thread shares one cache line.
__device__ __constant__ float kFp4Lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Memory-bandwidth-bound NVFP4 N=1 matvec.
//
// Layout: each block owns ROWS_PER_BLOCK output rows; each row is processed
// by exactly one warp (32 lanes), so the 32 lanes cooperatively walk the
// in_features axis and finish with a single warp-shuffle reduction (no
// __syncthreads needed inside the row).
//
// Block packs 8 rows × 32 lanes = 256 threads. With 192 SMs on Blackwell,
// 17408 / 8 = 2176 blocks fits in roughly 3 waves at 4-block-per-SM
// occupancy — a much better fit than one-block-per-row, which serialised
// at ~6 waves for the same matvec.
//
// Per-group work (16 input dims):
// - 8 packed FP4 bytes pulled as a single 8-byte word (when aligned).
// - 16 BF16 inputs pulled as a single 32-byte uint4 (when aligned).
// - One block scale (e4m3) decoded via a constant-memory LUT.
// - The 16 multiplies fold into a register accumulator.
constexpr int kFp4MatvecRowsPerBlock = 8;
constexpr int kFp4MatvecLanesPerRow = 32;
constexpr int kFp4MatvecThreadsPerBlock =
    kFp4MatvecRowsPerBlock * kFp4MatvecLanesPerRow;

__global__ void __launch_bounds__(kFp4MatvecThreadsPerBlock)
nvfp4_matvec_kernel(const __nv_bfloat16 *__restrict__ input,
                    const uint8_t *__restrict__ weight,
                    const uint8_t *__restrict__ block_scale,
                    const float *__restrict__ tensor_scale,
                    __nv_bfloat16 *__restrict__ output,
                    size_t in_features, size_t out_features) {
  const unsigned warp_id = threadIdx.x >> 5;       // 0..ROWS_PER_BLOCK-1
  const unsigned lane = threadIdx.x & 31;          // 0..31
  const size_t row = static_cast<size_t>(blockIdx.x) *
                         kFp4MatvecRowsPerBlock +
                     warp_id;
  if (row >= out_features) {
    return;
  }

  const size_t packed_cols = (in_features + 1) / 2;
  const size_t scale_cols = (in_features + 15) / 16;
  const size_t scale_inner_dim = round_up_size(scale_cols, 4);
  const uint8_t *row_weight = weight + row * packed_cols;
  const float global_scale =
      tensor_scale == nullptr ? 1.0f : __ldg(tensor_scale);

  float sum = 0.0f;

  for (size_t g = lane; g < scale_cols; g += kFp4MatvecLanesPerRow) {
    const size_t col0 = g * 16;
    const size_t scale_off = vec16_scale_offset(g, row, scale_inner_dim);
    const float scale =
        decode_e4m3(__ldg(block_scale + scale_off)) * global_scale;

    // Vector load: 8 packed FP4 bytes -> 16 weight values.
    const size_t weight_byte_off = col0 / 2;
    uint64_t packed_weights;
    const bool fast = col0 + 16 <= in_features && (weight_byte_off & 7u) == 0;
    if (fast) {
      packed_weights =
          *reinterpret_cast<const uint64_t *>(row_weight + weight_byte_off);
    } else {
      packed_weights = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (col0 + 2 * i < in_features) {
          packed_weights |=
              static_cast<uint64_t>(row_weight[weight_byte_off + i])
              << (i * 8);
        }
      }
    }

    __nv_bfloat162 input_pairs[8];
    if (fast && (reinterpret_cast<uintptr_t>(input + col0) & 15u) == 0) {
      const uint4 packed_input = *reinterpret_cast<const uint4 *>(input + col0);
      const __nv_bfloat162 *as_pairs =
          reinterpret_cast<const __nv_bfloat162 *>(&packed_input);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        input_pairs[i] = as_pairs[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const size_t col = col0 + 2 * i;
        const __nv_bfloat16 a =
            (col < in_features) ? input[col] : __float2bfloat16(0.0f);
        const __nv_bfloat16 b = (col + 1 < in_features)
                                    ? input[col + 1]
                                    : __float2bfloat16(0.0f);
        input_pairs[i] = __halves2bfloat162(a, b);
      }
    }

    float local = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint8_t byte = (packed_weights >> (i * 8)) & 0xffu;
      const float w0 = kFp4Lut[byte & 0x0fu];
      const float w1 = kFp4Lut[(byte >> 4) & 0x0fu];
      const float a0 = __bfloat162float(input_pairs[i].x);
      const float a1 = __bfloat162float(input_pairs[i].y);
      local += a0 * w0 + a1 * w1;
    }
    sum += local * scale;
  }

  // Single warp-shuffle reduction for this row's 32 lanes.
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  if (lane == 0) {
    output[row] = __float2bfloat16(sum);
  }
}

__global__ void nvfp4_quantize_fused_kernel(const __nv_bfloat16 *input,
                                            uint8_t *output, uint8_t *scale,
                                            float *tensor_scale,
                                            size_t values,
                                            float input_tensor_scale) {
  __shared__ float scratch[32];
  __shared__ float decoded_scale;
  const float global_scale =
      input_tensor_scale > 0.0f ? input_tensor_scale : 1.0f;
  const size_t group = blockIdx.x;
  const size_t scale_inner_dim = round_up_size(div_ceil_size(values, 16), 4);
  const size_t start = group * 16;
  float local_amax = 0.0f;
  for (size_t offset = threadIdx.x; offset < 16 && start + offset < values;
       offset += blockDim.x) {
    local_amax =
        fmaxf(local_amax, fabsf(__bfloat162float(input[start + offset])));
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
    scale[vec16_scale_offset(group, 0, scale_inner_dim)] = scale_code;
    decoded_scale = fmaxf(decode_e4m3(scale_code) * global_scale, 1.0e-8f);
    if (group == 0 && tensor_scale != nullptr) {
      *tensor_scale = global_scale;
    }
  }
  __syncthreads();

  if (threadIdx.x < 8) {
    const size_t col = start + threadIdx.x * 2;
    if (col < values) {
      const float value0 = __bfloat162float(input[col]) / decoded_scale;
      uint8_t packed = encode_e2m1(value0);
      if (col + 1 < values) {
        const float value1 = __bfloat162float(input[col + 1]) / decoded_scale;
        packed |= static_cast<uint8_t>(encode_e2m1(value1) << 4);
      }
      output[col / 2] = packed;
    }
  }
}

__global__ void nvfp4_quantize_rows_kernel(const __nv_bfloat16 *input,
                                           uint8_t *output, uint8_t *scale,
                                           float *tensor_scale, size_t rows,
                                           size_t values,
                                           float input_tensor_scale) {
  __shared__ float scratch[32];
  __shared__ float decoded_scale;
  const float global_scale =
      input_tensor_scale > 0.0f ? input_tensor_scale : 1.0f;
  const size_t group = blockIdx.x;
  const size_t row = blockIdx.y;
  const size_t scale_inner_dim = round_up_size(div_ceil_size(values, 16), 4);
  const size_t start = group * 16;
  const __nv_bfloat16 *row_input = input + row * values;
  uint8_t *row_output = output + row * div_ceil_size(values, 2);
  float local_amax = 0.0f;
  for (size_t offset = threadIdx.x; offset < 16 && start + offset < values;
       offset += blockDim.x) {
    local_amax =
        fmaxf(local_amax, fabsf(__bfloat162float(row_input[start + offset])));
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
    scale[vec16_scale_offset(group, row, scale_inner_dim)] = scale_code;
    decoded_scale = fmaxf(decode_e4m3(scale_code) * global_scale, 1.0e-8f);
    if (row == 0 && group == 0 && tensor_scale != nullptr) {
      *tensor_scale = global_scale;
    }
  }
  __syncthreads();

  if (threadIdx.x < 8) {
    const size_t col = start + threadIdx.x * 2;
    if (col < values) {
      const float value0 = __bfloat162float(row_input[col]) / decoded_scale;
      uint8_t packed = encode_e2m1(value0);
      if (col + 1 < values) {
        const float value1 =
            __bfloat162float(row_input[col + 1]) / decoded_scale;
        packed |= static_cast<uint8_t>(encode_e2m1(value1) << 4);
      }
      row_output[col / 2] = packed;
    }
  }
}

__global__ void nvfp4_retile_scales_kernel(const uint8_t *input,
                                           uint8_t *output, size_t rows,
                                           size_t inner_groups) {
  const size_t total = rows * inner_groups;
  const size_t sf_inner_dim = round_up_size(inner_groups, 4);
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    const size_t row = idx / inner_groups;
    const size_t inner = idx % inner_groups;
    output[vec16_scale_offset(inner, row, sf_inner_dim)] = input[idx];
  }
}

__global__ void conv1d_update_kernel(const __nv_bfloat16 *input,
                                     __nv_bfloat16 *history,
                                     const __nv_bfloat16 *weight,
                                     __nv_bfloat16 *output, size_t channels,
                                     size_t kernel_size) {
  for (size_t channel = blockIdx.x * blockDim.x + threadIdx.x; channel < channels;
       channel += blockDim.x * gridDim.x) {
    __nv_bfloat16 *channel_history = history + channel * (kernel_size - 1);
    const __nv_bfloat16 *channel_weight = weight + channel * kernel_size;
    float sum = __bfloat162float(input[channel]) *
                __bfloat162float(channel_weight[kernel_size - 1]);
    for (size_t k = 0; k + 1 < kernel_size; ++k) {
      sum += __bfloat162float(channel_history[k]) *
             __bfloat162float(channel_weight[k]);
    }
    for (size_t k = 0; k + 2 < kernel_size; ++k) {
      channel_history[k] = channel_history[k + 1];
    }
    channel_history[kernel_size - 2] = input[channel];
    const float silu = sum / (1.0f + expf(-sum));
    output[channel] = __float2bfloat16(silu);
  }
}

__global__ void conv1d_prefill_kernel(const __nv_bfloat16 *input,
                                      __nv_bfloat16 *history,
                                      const __nv_bfloat16 *weight,
                                      __nv_bfloat16 *output, size_t tokens,
                                      size_t channels, size_t kernel_size) {
  for (size_t channel = blockIdx.x * blockDim.x + threadIdx.x;
       channel < channels; channel += blockDim.x * gridDim.x) {
    __nv_bfloat16 *channel_history = history + channel * (kernel_size - 1);
    const __nv_bfloat16 *channel_weight = weight + channel * kernel_size;
    for (size_t tok = 0; tok < tokens; ++tok) {
      float sum = __bfloat162float(input[tok * channels + channel]) *
                  __bfloat162float(channel_weight[kernel_size - 1]);
      for (size_t k = 0; k + 1 < kernel_size; ++k) {
        const size_t lag = kernel_size - 1 - k;
        float hist_value = 0.0f;
        if (tok >= lag) {
          hist_value =
              __bfloat162float(input[(tok - lag) * channels + channel]);
        } else {
          hist_value = __bfloat162float(channel_history[k + tok]);
        }
        sum += hist_value * __bfloat162float(channel_weight[k]);
      }
      const float silu = sum / (1.0f + expf(-sum));
      output[tok * channels + channel] = __float2bfloat16(silu);
    }
    for (size_t k = 0; k + 1 < kernel_size; ++k) {
      const size_t lag = kernel_size - 2 - k;
      if (tokens > lag) {
        channel_history[k] =
            input[(tokens - 1 - lag) * channels + channel];
      } else {
        channel_history[k] = channel_history[k + tokens];
      }
    }
  }
}

__global__ void gdn_gate_kernel(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                                const __nv_bfloat16 *a_log,
                                const __nv_bfloat16 *dt_bias, float *gate,
                                float *beta, size_t rows, size_t heads) {
  const size_t total = rows * heads;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    const size_t head = idx % heads;
    const float x =
        __bfloat162float(a[idx]) + __bfloat162float(dt_bias[head]);
    const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;
    gate[idx] = -expf(__bfloat162float(a_log[head])) * softplus;
    const float b_value = __bfloat162float(b[idx]);
    beta[idx] = 1.0f / (1.0f + expf(-b_value));
  }
}

__global__ void sigmoid_gate_kernel(const __nv_bfloat16 *gate,
                                    const __nv_bfloat16 *input,
                                    __nv_bfloat16 *output, size_t elements) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements;
       idx += blockDim.x * gridDim.x) {
    const float gate_value = __bfloat162float(gate[idx]);
    const float input_value = __bfloat162float(input[idx]);
    output[idx] = __float2bfloat16(input_value /
                                   (1.0f + expf(-gate_value)));
  }
}

__global__ void sigmoid_gate_strided_kernel(
    const __nv_bfloat16 *gate, const __nv_bfloat16 *input,
    __nv_bfloat16 *output, size_t rows, size_t elements_per_row,
    size_t gate_stride, size_t input_stride, size_t output_stride) {
  const size_t total = rows * elements_per_row;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    const size_t row = idx / elements_per_row;
    const size_t col = idx % elements_per_row;
    const float gate_value =
        __bfloat162float(gate[row * gate_stride + col]);
    const float input_value =
        __bfloat162float(input[row * input_stride + col]);
    output[row * output_stride + col] =
        __float2bfloat16(input_value / (1.0f + expf(-gate_value)));
  }
}

__global__ void q_proj_deinterleave_kernel(const __nv_bfloat16 *input,
                                           __nv_bfloat16 *output,
                                           size_t rows, size_t heads,
                                           size_t head_dim) {
  const size_t q_values = heads * head_dim;
  const size_t row_stride = q_values * 2;
  const size_t total = rows * q_values;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    const size_t row = idx / q_values;
    const size_t col = idx % q_values;
    const size_t head = col / head_dim;
    const size_t dim = col % head_dim;
    output[idx] = input[row * row_stride + head * head_dim * 2 + dim];
  }
}

__global__ void q_proj_sigmoid_gate_kernel(
    const __nv_bfloat16 *gate, const __nv_bfloat16 *input,
    __nv_bfloat16 *output, size_t rows, size_t heads, size_t head_dim) {
  const size_t q_values = heads * head_dim;
  const size_t row_stride = q_values * 2;
  const size_t total = rows * q_values;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    const size_t row = idx / q_values;
    const size_t col = idx % q_values;
    const size_t head = col / head_dim;
    const size_t dim = col % head_dim;
    const float gate_value = __bfloat162float(
        gate[row * row_stride + head * head_dim * 2 + head_dim + dim]);
    const float input_value = __bfloat162float(input[idx]);
    output[idx] = __float2bfloat16(input_value / (1.0f + expf(-gate_value)));
  }
}

__global__ void copy_strided_rows_kernel(const __nv_bfloat16 *input,
                                         __nv_bfloat16 *output, size_t rows,
                                         size_t values, size_t input_stride,
                                         size_t output_stride) {
  const size_t total = rows * values;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    const size_t row = idx / values;
    const size_t col = idx % values;
    output[row * output_stride + col] = input[row * input_stride + col];
  }
}

} // namespace

extern "C" int qwen36_rmsnorm(const qwen36_rmsnorm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->rows == 0 || spec->hidden == 0 || spec->input_bf16.ptr == 0 ||
      spec->weight_bf16.ptr == 0 || spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  const int threads = 256;
  const float eps = spec->eps == 0.0f ? 1.0e-6f : spec->eps;
  rmsnorm_kernel<<<static_cast<unsigned int>(spec->rows), threads,
                   threads * sizeof(float), qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<const __nv_bfloat16>(spec->weight_bf16),
      ptr<const __nv_bfloat16>(spec->residual_bf16),
      ptr<__nv_bfloat16>(spec->residual_out_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->hidden, eps,
      spec->direct_weight);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_rmsnorm_nvfp4_quantize(
    const qwen36_rmsnorm_nvfp4_quantize_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->hidden == 0 || spec->input_bf16.ptr == 0 ||
      spec->weight_bf16.ptr == 0 || spec->output_fp4.ptr == 0 ||
      spec->output_scale_e4m3.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  const int threads = 256;
  const float eps = spec->eps == 0.0f ? 1.0e-6f : spec->eps;
  rmsnorm_nvfp4_quantize_kernel<<<1, threads, threads * sizeof(float), qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<const __nv_bfloat16>(spec->weight_bf16),
      ptr<const __nv_bfloat16>(spec->residual_bf16),
      ptr<__nv_bfloat16>(spec->residual_out_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16),
      ptr<uint8_t>(spec->output_fp4),
      ptr<uint8_t>(spec->output_scale_e4m3),
      ptr<float>(spec->output_tensor_scale_f32), spec->hidden, eps,
      spec->input_tensor_scale_f32);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_partial_rope(const qwen36_partial_rope_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->tokens == 0 || spec->q_heads == 0 || spec->kv_heads == 0 ||
      spec->head_dim == 0 || spec->rope_dims == 0 ||
      spec->rope_dims > spec->head_dim || (spec->rope_dims % 2) != 0 ||
      (spec->positions_i32.ptr == 0 && spec->use_scalar_position == 0) ||
      spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  const int threads = 32;
  const dim3 grid(static_cast<unsigned int>(spec->rope_dims / 2),
                  static_cast<unsigned int>(spec->tokens));
  const float base_theta =
      spec->base_theta > 0.0 ? static_cast<float>(spec->base_theta) : 10000.0f;
  partial_rope_kernel<<<grid, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const int32_t>(spec->positions_i32), spec->position_i32,
      spec->use_scalar_position,
      ptr<const int32_t>(spec->scalar_position_device_i32),
      ptr<__nv_bfloat16>(spec->q_bf16), ptr<__nv_bfloat16>(spec->k_bf16),
      spec->q_heads, spec->kv_heads, spec->head_dim, spec->rope_dims,
      base_theta);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int
qwen36_embedding_lookup(const qwen36_embedding_lookup_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->tokens == 0 || spec->hidden == 0 || spec->vocab_size == 0 ||
      spec->token_ids_u32.ptr == 0 || spec->embedding_bf16.ptr == 0 ||
      spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((spec->hidden + threads - 1) / threads);
  const dim3 grid(blocks, static_cast<unsigned int>(spec->tokens));
  embedding_lookup_kernel<<<grid, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const uint32_t>(spec->token_ids_u32),
      ptr<const __nv_bfloat16>(spec->embedding_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->tokens, spec->hidden,
      spec->vocab_size);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_bf16_matvec(const qwen36_bf16_matvec_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->out_features == 0 || spec->in_features == 0 ||
      spec->input_bf16.ptr == 0 || spec->weight_bf16.ptr == 0 ||
      spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  bf16_matvec_kernel<<<static_cast<unsigned int>(spec->out_features), threads,
                       threads * sizeof(float), qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<const __nv_bfloat16>(spec->weight_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->in_features);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_nvfp4_matvec(const qwen36_nvfp4_matvec_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->out_features == 0 || spec->in_features == 0 ||
      spec->input_bf16.ptr == 0 || spec->weight_u8.ptr == 0 ||
      spec->block_scale_e4m3.ptr == 0 || spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  // ROWS_PER_BLOCK rows are handled per block (8 warps, 256 threads). This
  // keeps the live block count down to a few hundred for the wide MLP rows
  // while leaving enough work per warp to hide HBM latency.
  const unsigned int blocks =
      static_cast<unsigned int>((spec->out_features + kFp4MatvecRowsPerBlock - 1) /
                                kFp4MatvecRowsPerBlock);
  nvfp4_matvec_kernel<<<blocks, kFp4MatvecThreadsPerBlock, 0,
                        qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<const uint8_t>(spec->weight_u8),
      ptr<const uint8_t>(spec->block_scale_e4m3),
      ptr<const float>(spec->tensor_scale_f32),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->in_features,
      spec->out_features);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int
qwen36_nvfp4_quantize_bf16(const qwen36_nvfp4_quantize_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->values == 0 || spec->input_bf16.ptr == 0 ||
      spec->output_fp4.ptr == 0 || spec->output_scale_e4m3.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  const unsigned int scale_blocks =
      static_cast<unsigned int>((spec->values + 15) / 16);
  nvfp4_quantize_fused_kernel<<<scale_blocks, 32, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<uint8_t>(spec->output_fp4), ptr<uint8_t>(spec->output_scale_e4m3),
      ptr<float>(spec->output_tensor_scale_f32), spec->values,
      spec->input_tensor_scale_f32);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int
qwen36_nvfp4_quantize_rows(const qwen36_nvfp4_quantize_rows_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->rows == 0 || spec->values == 0 || spec->input_bf16.ptr == 0 ||
      spec->output_fp4.ptr == 0 || spec->output_scale_e4m3.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  const unsigned int scale_blocks =
      static_cast<unsigned int>((spec->values + 15) / 16);
  const dim3 grid(scale_blocks, static_cast<unsigned int>(spec->rows));
  nvfp4_quantize_rows_kernel<<<grid, 32, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<uint8_t>(spec->output_fp4), ptr<uint8_t>(spec->output_scale_e4m3),
      ptr<float>(spec->output_tensor_scale_f32), spec->rows, spec->values,
      spec->input_tensor_scale_f32);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int
qwen36_nvfp4_retile_scales(const qwen36_nvfp4_retile_scales_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->rows == 0 || spec->inner_groups == 0 ||
      spec->input_row_major_u8.ptr == 0 || spec->output_tiled_u8.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  cudaError_t err =
      cudaMemset(ptr<uint8_t>(spec->output_tiled_u8), 0,
                 vec16_scale_bytes(spec->inner_groups, spec->rows));
  if (err != cudaSuccess) {
    return QWEN36_STATUS_CUDA_ERROR;
  }

  const int threads = 256;
  const size_t total = spec->rows * spec->inner_groups;
  const unsigned int blocks =
      static_cast<unsigned int>((total + threads - 1) / threads);
  nvfp4_retile_scales_kernel<<<blocks, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const uint8_t>(spec->input_row_major_u8),
      ptr<uint8_t>(spec->output_tiled_u8), spec->rows, spec->inner_groups);
  err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_conv1d_update(const qwen36_conv1d_update_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->channels == 0 || spec->kernel_size < 2 || spec->input_bf16.ptr == 0 ||
      spec->conv_history_bf16.ptr == 0 || spec->weight_bf16.ptr == 0 ||
      spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((spec->channels + threads - 1) / threads);
  conv1d_update_kernel<<<blocks, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<__nv_bfloat16>(spec->conv_history_bf16),
      ptr<const __nv_bfloat16>(spec->weight_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->channels,
      spec->kernel_size);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_conv1d_prefill(const qwen36_conv1d_prefill_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->tokens == 0 || spec->channels == 0 || spec->kernel_size < 2 ||
      spec->input_bf16.ptr == 0 || spec->conv_history_bf16.ptr == 0 ||
      spec->weight_bf16.ptr == 0 || spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((spec->channels + threads - 1) / threads);
  conv1d_prefill_kernel<<<blocks, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<__nv_bfloat16>(spec->conv_history_bf16),
      ptr<const __nv_bfloat16>(spec->weight_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->tokens, spec->channels,
      spec->kernel_size);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_gdn_gate(const qwen36_gdn_gate_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->rows == 0 || spec->heads == 0 || spec->a_bf16.ptr == 0 ||
      spec->b_bf16.ptr == 0 || spec->a_log_bf16.ptr == 0 ||
      spec->dt_bias_bf16.ptr == 0 || spec->gate_f32.ptr == 0 ||
      spec->beta_f32.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 128;
  const size_t total = spec->rows * spec->heads;
  const unsigned int blocks =
      static_cast<unsigned int>((total + threads - 1) / threads);
  gdn_gate_kernel<<<blocks, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->a_bf16),
      ptr<const __nv_bfloat16>(spec->b_bf16),
      ptr<const __nv_bfloat16>(spec->a_log_bf16),
      ptr<const __nv_bfloat16>(spec->dt_bias_bf16),
      ptr<float>(spec->gate_f32), ptr<float>(spec->beta_f32), spec->rows,
      spec->heads);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_sigmoid_gate(const qwen36_sigmoid_gate_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->elements == 0 || spec->gate_bf16.ptr == 0 ||
      spec->input_bf16.ptr == 0 || spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((spec->elements + threads - 1) / threads);
  sigmoid_gate_kernel<<<blocks, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->gate_bf16),
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->elements);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int
qwen36_sigmoid_gate_strided(const qwen36_sigmoid_gate_strided_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->rows == 0 || spec->elements_per_row == 0 ||
      spec->gate_stride < spec->elements_per_row ||
      spec->input_stride < spec->elements_per_row ||
      spec->output_stride < spec->elements_per_row ||
      spec->gate_bf16.ptr == 0 || spec->input_bf16.ptr == 0 ||
      spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  const size_t total = spec->rows * spec->elements_per_row;
  const unsigned int blocks =
      static_cast<unsigned int>((total + threads - 1) / threads);
  sigmoid_gate_strided_kernel<<<blocks, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->gate_bf16),
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->rows,
      spec->elements_per_row, spec->gate_stride, spec->input_stride,
      spec->output_stride);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int
qwen36_q_proj_deinterleave(const qwen36_q_proj_deinterleave_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->rows == 0 || spec->heads == 0 || spec->head_dim == 0 ||
      spec->input_bf16.ptr == 0 || spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  const size_t total = spec->rows * spec->heads * spec->head_dim;
  const unsigned int blocks =
      static_cast<unsigned int>((total + threads - 1) / threads);
  q_proj_deinterleave_kernel<<<blocks, threads, 0,
                               qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->rows, spec->heads,
      spec->head_dim);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int
qwen36_q_proj_sigmoid_gate(const qwen36_q_proj_sigmoid_gate_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->rows == 0 || spec->heads == 0 || spec->head_dim == 0 ||
      spec->gate_bf16.ptr == 0 || spec->input_bf16.ptr == 0 ||
      spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  const size_t total = spec->rows * spec->heads * spec->head_dim;
  const unsigned int blocks =
      static_cast<unsigned int>((total + threads - 1) / threads);
  q_proj_sigmoid_gate_kernel<<<blocks, threads, 0,
                               qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->gate_bf16),
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->rows, spec->heads,
      spec->head_dim);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int
qwen36_copy_strided_rows(const qwen36_copy_strided_rows_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->rows == 0 || spec->values == 0 ||
      spec->input_stride < spec->values ||
      spec->output_stride < spec->values || spec->input_bf16.ptr == 0 ||
      spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const int threads = 256;
  const size_t total = spec->rows * spec->values;
  const unsigned int blocks =
      static_cast<unsigned int>((total + threads - 1) / threads);
  copy_strided_rows_kernel<<<blocks, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->input_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->rows, spec->values,
      spec->input_stride, spec->output_stride);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_swiglu(const qwen36_swiglu_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->rows == 0 || spec->intermediate == 0 ||
      spec->gate_bf16.ptr == 0 || spec->up_bf16.ptr == 0 ||
      spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  const int threads = 256;
  const size_t total = spec->rows * spec->intermediate;
  const unsigned int blocks =
      static_cast<unsigned int>((total + threads - 1) / threads);
  swiglu_kernel<<<blocks, threads, 0, qwen36_internal_active_stream()>>>(ptr<const __nv_bfloat16>(spec->gate_bf16),
                                     ptr<const __nv_bfloat16>(spec->up_bf16),
                                     ptr<__nv_bfloat16>(spec->output_bf16),
                                     total);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_sample(const qwen36_sampling_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->vocab_size == 0 || spec->logits_bf16.ptr == 0 ||
      spec->output_token_u32.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  const int threads = 256;
  const size_t shared_bytes = threads * (sizeof(float) + sizeof(uint32_t));
  sample_argmax_kernel<<<1, threads, shared_bytes, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->logits_bf16),
      ptr<uint32_t>(spec->output_token_u32), spec->vocab_size,
      spec->temperature);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
