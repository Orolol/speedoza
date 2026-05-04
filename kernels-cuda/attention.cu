#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

template <typename T> T *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

constexpr int kKvCacheBf16 = 0;
constexpr int kKvCacheFp8 = 1;
constexpr int kKvCacheTurboQuant3 = 2;
constexpr int kKvCacheTurboQuant35 = 3;
constexpr int kTqMetadataFloats = 4;
constexpr int kTqKeyNormSlot = 0;
constexpr int kTqKeyResidualNormSlot = 1;
constexpr int kTqValueNormSlot = 2;
constexpr int kTqMaxHeadDim = 256;

constexpr uint32_t kTqKeyRotationSalt = 0x2c1b3c6du;
constexpr uint32_t kTqKeyQjlSalt = 0x91e10da5u;
constexpr uint32_t kTqValueRotationSalt = 0x6d2b79f5u;

static __constant__ float kTqCentroids2[4] = {
    -0.1330401982533685f, -0.039990945215356365f,
    0.039990945215356365f, 0.1330401982533685f};
static __constant__ float kTqBoundaries2[3] = {
    -0.08651557173436243f, 0.0f, 0.08651557173436243f};
static __constant__ float kTqCentroids3[8] = {
    -0.188390613802078f, -0.11813298369899362f,
    -0.06658059531595685f, -0.021602468667239208f,
    0.021602468667239208f, 0.06658059531595685f,
    0.11813298369899362f, 0.188390613802078f};
static __constant__ float kTqBoundaries3[7] = {
    -0.1532617987505358f, -0.09235678950747524f,
    -0.04409153199159803f, 0.0f, 0.04409153199159803f,
    0.09235678950747524f, 0.1532617987505358f};
static __constant__ float kTqCentroids4[16] = {
    -0.23762718673095357f, -0.18079372947217434f,
    -0.1417616542967331f, -0.11024706538276363f,
    -0.08279256667309579f, -0.057744535605257094f,
    -0.034134028231120876f, -0.011296498142743928f,
    0.011296498142743841f, 0.034134028231120786f,
    0.05774453560525705f, 0.08279256667309574f,
    0.11024706538276359f, 0.14176165429673304f,
    0.18079372947217426f, 0.23762718673095345f};
static __constant__ float kTqBoundaries4[15] = {
    -0.20921045810156397f, -0.16127769188445373f,
    -0.12600435983974836f, -0.09651981602792971f,
    -0.07026855113917643f, -0.04593928191818898f,
    -0.022715263186932403f, 0.0f, 0.022715263186932313f,
    0.04593928191818892f, 0.0702685511391764f,
    0.09651981602792967f, 0.1260043598397483f,
    0.16127769188445365f, 0.20921045810156386f};

__host__ __device__ inline bool is_tq_cache_dtype(int kv_cache_dtype) {
  return kv_cache_dtype == kKvCacheTurboQuant3 ||
         kv_cache_dtype == kKvCacheTurboQuant35;
}

inline bool tq_supported_head_dim(size_t head_dim) {
  return head_dim >= 8 && head_dim <= kTqMaxHeadDim &&
         (head_dim & (head_dim - 1)) == 0;
}

__device__ float decode_e4m3(uint8_t code) {
  const float sign = (code & 0x80) ? -1.0f : 1.0f;
  const int exponent = (code >> 3) & 0x0f;
  const int mantissa = code & 0x07;
  if (exponent == 0) {
    if (mantissa == 0) {
      return sign * 0.0f;
    }
    return sign * ldexpf(static_cast<float>(mantissa) / 8.0f, -6);
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return sign * 448.0f;
  }
  return sign * ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f,
                       exponent - 7);
}

__device__ uint8_t encode_e4m3(float value) {
  if (value == 0.0f || !isfinite(value)) {
    return 0;
  }
  const bool negative = value < 0.0f;
  float abs_value = fabsf(value);
  if (abs_value >= 448.0f) {
    return static_cast<uint8_t>((negative ? 0x80 : 0x00) | 0x7e);
  }

  constexpr float kMinNormal = 0x1p-6f;
  constexpr float kSubnormalStep = 0x1p-9f;
  constexpr float kNormalBoundary = (7.0f * kSubnormalStep + kMinNormal) * 0.5f;
  uint8_t code = 0;
  if (abs_value < kMinNormal) {
    if (abs_value >= kNormalBoundary) {
      code = 0x08;
    } else {
      int mantissa = static_cast<int>(floorf(abs_value / kSubnormalStep + 0.5f));
      if (mantissa <= 0) {
        code = 0;
      } else {
        code = static_cast<uint8_t>(mantissa);
      }
    }
  } else {
    int exponent = 0;
    float frac = frexpf(abs_value, &exponent);
    // frexp returns abs_value = frac * 2^exponent with frac in [0.5, 1).
    const int exponent_field = exponent + 6;
    float mantissa_f = (frac * 2.0f - 1.0f) * 8.0f;
    int mantissa = static_cast<int>(floorf(mantissa_f + 0.5f));
    int adjusted_exponent = exponent_field;
    if (mantissa >= 8) {
      mantissa = 0;
      adjusted_exponent += 1;
    }
    if (adjusted_exponent >= 15) {
      adjusted_exponent = 15;
      mantissa = min(mantissa, 6);
    }
    code = static_cast<uint8_t>((adjusted_exponent << 3) | mantissa);
  }
  return static_cast<uint8_t>((negative ? 0x80 : 0x00) | code);
}

__device__ __forceinline__ int tq_value_bits(int kv_cache_dtype) {
  return kv_cache_dtype == kKvCacheTurboQuant35 ? 4 : 3;
}

__device__ __forceinline__ size_t tq_bytes_for_bits(size_t head_dim,
                                                    int bits) {
  return (head_dim * static_cast<size_t>(bits) + 7) >> 3;
}

__device__ __forceinline__ size_t tq_key_mse_bytes(size_t head_dim) {
  return tq_bytes_for_bits(head_dim, 2);
}

__device__ __forceinline__ size_t tq_key_vector_bytes(size_t head_dim) {
  return tq_key_mse_bytes(head_dim) + tq_bytes_for_bits(head_dim, 1);
}

__device__ __forceinline__ size_t tq_value_vector_bytes(size_t head_dim,
                                                        int kv_cache_dtype) {
  return tq_bytes_for_bits(head_dim, tq_value_bits(kv_cache_dtype));
}

__device__ __forceinline__ uint32_t tq_hash(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ float tq_rht_sign(size_t dim, uint32_t salt,
                                             uint32_t pass) {
  const uint32_t x = tq_hash(static_cast<uint32_t>(dim) * 0x9e3779b9u ^
                             salt ^ (pass * 0x85ebca6bu));
  return (x & 1u) ? -1.0f : 1.0f;
}

__device__ __forceinline__ float tq_hadamard_entry(size_t row, size_t col) {
  const unsigned int bits =
      static_cast<unsigned int>(row) & static_cast<unsigned int>(col);
  return (__popc(bits) & 1) ? -1.0f : 1.0f;
}

__device__ __forceinline__ float tq_dim_scale(size_t head_dim) {
  return sqrtf(128.0f / static_cast<float>(head_dim));
}

__device__ __forceinline__ float tq_centroid(int bits, int code,
                                             size_t head_dim) {
  const float scale = tq_dim_scale(head_dim);
  if (bits == 2) {
    return kTqCentroids2[code] * scale;
  }
  if (bits == 3) {
    return kTqCentroids3[code] * scale;
  }
  return kTqCentroids4[code] * scale;
}

__device__ __forceinline__ float tq_centroid_scaled(int bits, int code,
                                                    float dim_scale) {
  if (bits == 2) {
    return kTqCentroids2[code] * dim_scale;
  }
  if (bits == 3) {
    return kTqCentroids3[code] * dim_scale;
  }
  return kTqCentroids4[code] * dim_scale;
}

__device__ __forceinline__ float tq_boundary(int bits, int code,
                                             size_t head_dim) {
  const float scale = tq_dim_scale(head_dim);
  if (bits == 2) {
    return kTqBoundaries2[code] * scale;
  }
  if (bits == 3) {
    return kTqBoundaries3[code] * scale;
  }
  return kTqBoundaries4[code] * scale;
}

__device__ __forceinline__ uint8_t tq_quantize_code(float value, int bits,
                                                    size_t head_dim) {
  if (!isfinite(value)) {
    return static_cast<uint8_t>((1 << bits) / 2);
  }
  int code = 0;
  const int n_codes = 1 << bits;
  while (code < n_codes - 1 && value > tq_boundary(bits, code, head_dim)) {
    ++code;
  }
  return static_cast<uint8_t>(code);
}

__device__ __forceinline__ uint8_t tq_load_packed_code(const uint8_t *cache,
                                                       size_t dim, int bits) {
  const size_t bit = dim * static_cast<size_t>(bits);
  const size_t byte = bit >> 3;
  const unsigned int shift = static_cast<unsigned int>(bit & 7);
  unsigned int word = cache[byte];
  if (shift + static_cast<unsigned int>(bits) > 8) {
    word |= static_cast<unsigned int>(cache[byte + 1]) << 8;
  }
  return static_cast<uint8_t>((word >> shift) & ((1u << bits) - 1u));
}

__device__ __forceinline__ void tq_store_packed_code(uint8_t *cache,
                                                     size_t dim, int bits,
                                                     uint8_t code) {
  const size_t bit = dim * static_cast<size_t>(bits);
  const size_t byte = bit >> 3;
  const unsigned int shift = static_cast<unsigned int>(bit & 7);
  const unsigned int mask = (1u << bits) - 1u;
  const unsigned int value = static_cast<unsigned int>(code & mask) << shift;
  cache[byte] = static_cast<uint8_t>(cache[byte] | (value & 0xff));
  if (shift + static_cast<unsigned int>(bits) > 8) {
    cache[byte + 1] =
        static_cast<uint8_t>(cache[byte + 1] | ((value >> 8) & 0xff));
  }
}

__device__ __forceinline__ void tq_store_sign_bit(uint8_t *cache, size_t dim,
                                                  bool positive) {
  if (positive) {
    cache[dim >> 3] =
        static_cast<uint8_t>(cache[dim >> 3] | (1u << (dim & 7)));
  }
}

__device__ __forceinline__ float tq_load_sign_bit(const uint8_t *cache,
                                                  size_t dim) {
  return (cache[dim >> 3] & (1u << (dim & 7))) ? 1.0f : -1.0f;
}

__device__ void tq_rotate_forward_local(float *x, size_t head_dim,
                                        uint32_t salt) {
  for (size_t d = 0; d < head_dim; ++d) {
    x[d] *= tq_rht_sign(d, salt, 0);
  }
  for (size_t h = 1; h < head_dim; h <<= 1) {
    for (size_t i = 0; i < head_dim; i += h << 1) {
      for (size_t j = i; j < i + h; ++j) {
        const float a = x[j];
        const float b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
    }
  }
  const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));
  for (size_t d = 0; d < head_dim; ++d) {
    x[d] *= inv_sqrt * tq_rht_sign(d, salt, 1);
  }
}

__device__ void tq_rotate_forward_shared(float *x, size_t head_dim,
                                         uint32_t salt) {
  const size_t d = threadIdx.x;
  if (d < head_dim) {
    x[d] *= tq_rht_sign(d, salt, 0);
  }
  __syncthreads();
  for (size_t h = 1; h < head_dim; h <<= 1) {
    if (d < head_dim && (d & h) == 0 && d + h < head_dim) {
      const float a = x[d];
      const float b = x[d + h];
      x[d] = a + b;
      x[d + h] = a - b;
    }
    __syncthreads();
  }
  if (d < head_dim) {
    x[d] *= rsqrtf(static_cast<float>(head_dim)) * tq_rht_sign(d, salt, 1);
  }
  __syncthreads();
}

__device__ void tq_rotate_inverse_shared(float *x, size_t head_dim,
                                         uint32_t salt) {
  const size_t d = threadIdx.x;
  if (d < head_dim) {
    x[d] *= tq_rht_sign(d, salt, 1);
  }
  __syncthreads();
  for (size_t h = 1; h < head_dim; h <<= 1) {
    if (d < head_dim && (d & h) == 0 && d + h < head_dim) {
      const float a = x[d];
      const float b = x[d + h];
      x[d] = a + b;
      x[d + h] = a - b;
    }
    __syncthreads();
  }
  if (d < head_dim) {
    x[d] *= rsqrtf(static_cast<float>(head_dim)) * tq_rht_sign(d, salt, 0);
  }
  __syncthreads();
}

__device__ void tq_zero_bytes(uint8_t *dst, size_t bytes) {
  for (size_t i = 0; i < bytes; ++i) {
    dst[i] = 0;
  }
}

__device__ float tq_block_sum(float local, float *warp_sums) {
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31u) >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_xor_sync(0xffffffff, local, offset);
  }
  if (lane_id == 0) {
    warp_sums[warp_id] = local;
  }
  __syncthreads();
  if (warp_id == 0) {
    float total = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_xor_sync(0xffffffff, total, offset);
    }
    if (lane_id == 0) {
      warp_sums[0] = total;
    }
  }
  __syncthreads();
  return warp_sums[0];
}

__device__ void tq_zero_bytes_parallel(uint8_t *dst, size_t bytes) {
  for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
    dst[i] = 0;
  }
  __syncthreads();
}

__device__ __forceinline__ uint8_t tq_pack_code_byte(const uint8_t *codes,
                                                     size_t head_dim, int bits,
                                                     size_t byte_index) {
  uint8_t out = 0;
  if (bits == 1) {
    const size_t base_dim = byte_index << 3;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const size_t dim = base_dim + static_cast<size_t>(j);
      if (dim < head_dim) {
        out |= static_cast<uint8_t>((codes[dim] & 0x01u) << j);
      }
    }
    return out;
  }
  if (bits == 2) {
    const size_t base_dim = byte_index << 2;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const size_t dim = base_dim + static_cast<size_t>(j);
      if (dim < head_dim) {
        out |= static_cast<uint8_t>((codes[dim] & 0x03u) << (j << 1));
      }
    }
    return out;
  }
  if (bits == 4) {
    const size_t dim = byte_index << 1;
    if (dim < head_dim) {
      out = static_cast<uint8_t>(codes[dim] & 0x0fu);
    }
    if (dim + 1 < head_dim) {
      out |= static_cast<uint8_t>((codes[dim + 1] & 0x0fu) << 4);
    }
    return out;
  }

  const size_t byte_bit = byte_index << 3;
  const size_t first_dim = byte_bit / static_cast<size_t>(bits);
  size_t last_dim = (byte_bit + 7) / static_cast<size_t>(bits);
  if (last_dim >= head_dim) {
    last_dim = head_dim - 1;
  }
  for (size_t dim = first_dim; dim <= last_dim; ++dim) {
    const unsigned int code = codes[dim];
    const size_t code_bit = dim * static_cast<size_t>(bits);
    for (int bit = 0; bit < bits; ++bit) {
      const size_t packed_bit = code_bit + static_cast<size_t>(bit);
      if (packed_bit >= byte_bit && packed_bit < byte_bit + 8 &&
          (code & (1u << bit)) != 0) {
        out |= static_cast<uint8_t>(1u << (packed_bit - byte_bit));
      }
    }
  }
  return out;
}

__device__ void tq_pack_codes_parallel(const uint8_t *codes, uint8_t *dst,
                                       size_t head_dim, int bits,
                                       size_t bytes) {
  __syncthreads();
  for (size_t byte = threadIdx.x; byte < bytes; byte += blockDim.x) {
    dst[byte] = tq_pack_code_byte(codes, head_dim, bits, byte);
  }
  __syncthreads();
}

__device__ void encode_tq_key_vector(const __nv_bfloat16 *src, uint8_t *cache,
                                     float *metadata, size_t vector_index,
                                     size_t head_dim) {
  float x[kTqMaxHeadDim];
  float residual[kTqMaxHeadDim];
  float norm_sq = 0.0f;
  for (size_t d = 0; d < head_dim; ++d) {
    const float value = __bfloat162float(src[d]);
    x[d] = value;
    norm_sq += value * value;
  }
  const float norm = sqrtf(norm_sq);
  metadata[vector_index * kTqMetadataFloats + kTqKeyNormSlot] = norm;

  const size_t mse_bytes = tq_key_mse_bytes(head_dim);
  const size_t sign_bytes = tq_bytes_for_bits(head_dim, 1);
  uint8_t *dst = cache + vector_index * (mse_bytes + sign_bytes);
  tq_zero_bytes(dst, mse_bytes + sign_bytes);

  if (norm <= 1.0e-20f || !isfinite(norm)) {
    metadata[vector_index * kTqMetadataFloats + kTqKeyResidualNormSlot] = 0.0f;
    return;
  }
  const float inv_norm = 1.0f / norm;
  for (size_t d = 0; d < head_dim; ++d) {
    x[d] *= inv_norm;
  }
  tq_rotate_forward_local(x, head_dim, kTqKeyRotationSalt);

  float residual_unit_sq = 0.0f;
  for (size_t d = 0; d < head_dim; ++d) {
    const uint8_t code = tq_quantize_code(x[d], 2, head_dim);
    tq_store_packed_code(dst, d, 2, code);
    const float error = x[d] - tq_centroid(2, code, head_dim);
    residual[d] = error;
    residual_unit_sq += error * error;
  }

  const float residual_unit_norm = sqrtf(residual_unit_sq);
  const float residual_norm = norm * residual_unit_norm;
  metadata[vector_index * kTqMetadataFloats + kTqKeyResidualNormSlot] =
      residual_norm;
  if (residual_unit_norm <= 1.0e-20f || !isfinite(residual_unit_norm)) {
    return;
  }

  const float inv_residual_norm = 1.0f / residual_unit_norm;
  for (size_t d = 0; d < head_dim; ++d) {
    residual[d] *= inv_residual_norm;
  }
  tq_rotate_forward_local(residual, head_dim, kTqKeyQjlSalt);
  uint8_t *sign_dst = dst + mse_bytes;
  for (size_t d = 0; d < head_dim; ++d) {
    tq_store_sign_bit(sign_dst, d, residual[d] >= 0.0f);
  }
}

__device__ void encode_tq_value_vector(const __nv_bfloat16 *src, uint8_t *cache,
                                       float *metadata, size_t vector_index,
                                       size_t head_dim, int kv_cache_dtype) {
  const int bits = tq_value_bits(kv_cache_dtype);
  float x[kTqMaxHeadDim];
  float norm_sq = 0.0f;
  for (size_t d = 0; d < head_dim; ++d) {
    const float value = __bfloat162float(src[d]);
    x[d] = value;
    norm_sq += value * value;
  }
  const float norm = sqrtf(norm_sq);
  metadata[vector_index * kTqMetadataFloats + kTqValueNormSlot] = norm;
  metadata[vector_index * kTqMetadataFloats + 3] = 0.0f;

  const size_t vector_bytes = tq_value_vector_bytes(head_dim, kv_cache_dtype);
  uint8_t *dst = cache + vector_index * vector_bytes;
  tq_zero_bytes(dst, vector_bytes);
  if (norm <= 1.0e-20f || !isfinite(norm)) {
    return;
  }
  const float inv_norm = 1.0f / norm;
  for (size_t d = 0; d < head_dim; ++d) {
    x[d] *= inv_norm;
  }
  tq_rotate_forward_local(x, head_dim, kTqValueRotationSalt);
  for (size_t d = 0; d < head_dim; ++d) {
    tq_store_packed_code(dst, d, bits,
                         tq_quantize_code(x[d], bits, head_dim));
  }
}

__device__ void encode_tq_key_vector_parallel(
    const __nv_bfloat16 *src, uint8_t *cache, float *metadata,
    size_t vector_index, size_t head_dim, float *x, float *residual,
    uint8_t *codes, float *warp_sums) {
  const size_t d = threadIdx.x;
  const bool active = d < head_dim;
  float local_sq = 0.0f;
  if (active) {
    const float value = __bfloat162float(src[d]);
    x[d] = value;
    local_sq = value * value;
  }
  const float norm = sqrtf(tq_block_sum(local_sq, warp_sums));
  if (threadIdx.x == 0) {
    metadata[vector_index * kTqMetadataFloats + kTqKeyNormSlot] = norm;
  }

  const size_t mse_bytes = tq_key_mse_bytes(head_dim);
  const size_t sign_bytes = tq_bytes_for_bits(head_dim, 1);
  uint8_t *dst = cache + vector_index * (mse_bytes + sign_bytes);
  tq_zero_bytes_parallel(dst, mse_bytes + sign_bytes);

  if (norm <= 1.0e-20f || !isfinite(norm)) {
    if (threadIdx.x == 0) {
      metadata[vector_index * kTqMetadataFloats + kTqKeyResidualNormSlot] =
          0.0f;
    }
    return;
  }

  const float inv_norm = 1.0f / norm;
  if (active) {
    x[d] *= inv_norm;
  }
  tq_rotate_forward_shared(x, head_dim, kTqKeyRotationSalt);

  float residual_sq = 0.0f;
  if (active) {
    const uint8_t code = tq_quantize_code(x[d], 2, head_dim);
    codes[d] = code;
    const float error = x[d] - tq_centroid(2, code, head_dim);
    residual[d] = error;
    residual_sq = error * error;
  }
  const float residual_unit_norm = sqrtf(tq_block_sum(residual_sq, warp_sums));
  tq_pack_codes_parallel(codes, dst, head_dim, 2, mse_bytes);

  const float residual_norm = norm * residual_unit_norm;
  if (threadIdx.x == 0) {
    metadata[vector_index * kTqMetadataFloats + kTqKeyResidualNormSlot] =
        residual_norm;
  }
  if (residual_unit_norm <= 1.0e-20f || !isfinite(residual_unit_norm)) {
    return;
  }

  const float inv_residual_norm = 1.0f / residual_unit_norm;
  if (active) {
    residual[d] *= inv_residual_norm;
  }
  tq_rotate_forward_shared(residual, head_dim, kTqKeyQjlSalt);
  if (active) {
    codes[d] = residual[d] >= 0.0f ? 1u : 0u;
  }
  tq_pack_codes_parallel(codes, dst + mse_bytes, head_dim, 1, sign_bytes);
}

__device__ void encode_tq_value_vector_parallel(
    const __nv_bfloat16 *src, uint8_t *cache, float *metadata,
    size_t vector_index, size_t head_dim, int kv_cache_dtype, float *x,
    uint8_t *codes, float *warp_sums) {
  const int bits = tq_value_bits(kv_cache_dtype);
  const size_t d = threadIdx.x;
  const bool active = d < head_dim;
  float local_sq = 0.0f;
  if (active) {
    const float value = __bfloat162float(src[d]);
    x[d] = value;
    local_sq = value * value;
  }
  const float norm = sqrtf(tq_block_sum(local_sq, warp_sums));
  if (threadIdx.x == 0) {
    metadata[vector_index * kTqMetadataFloats + kTqValueNormSlot] = norm;
    metadata[vector_index * kTqMetadataFloats + 3] = 0.0f;
  }

  const size_t vector_bytes = tq_value_vector_bytes(head_dim, kv_cache_dtype);
  uint8_t *dst = cache + vector_index * vector_bytes;
  tq_zero_bytes_parallel(dst, vector_bytes);
  if (norm <= 1.0e-20f || !isfinite(norm)) {
    return;
  }

  const float inv_norm = 1.0f / norm;
  if (active) {
    x[d] *= inv_norm;
  }
  tq_rotate_forward_shared(x, head_dim, kTqValueRotationSalt);
  if (active) {
    codes[d] = tq_quantize_code(x[d], bits, head_dim);
  }
  tq_pack_codes_parallel(codes, dst, head_dim, bits, vector_bytes);
}

__device__ __forceinline__ float load_tq_key_score_component(
    const void *cache, const float *metadata, size_t vector_index, size_t dim,
    size_t head_dim, float q_rot, float q_sketch, float dim_scale) {
  const size_t mse_bytes = tq_key_mse_bytes(head_dim);
  const uint8_t *base = reinterpret_cast<const uint8_t *>(cache) +
                        vector_index * tq_key_vector_bytes(head_dim);
  const uint8_t code = tq_load_packed_code(base, dim, 2);
  const float key_norm =
      metadata[vector_index * kTqMetadataFloats + kTqKeyNormSlot];
  float local = q_rot * kTqCentroids2[code] * dim_scale * key_norm;

  const float residual_norm =
      metadata[vector_index * kTqMetadataFloats + kTqKeyResidualNormSlot];
  if (residual_norm != 0.0f) {
    constexpr float kQjlScaleNumerator = 1.2533141373155001f;
    const float sign = tq_load_sign_bit(base + mse_bytes, dim);
    local += q_sketch * sign * residual_norm *
             (kQjlScaleNumerator / static_cast<float>(head_dim));
  }
  return local;
}

__device__ __forceinline__ float load_tq_rotated_value_component(
    const void *cache, const float *metadata, int kv_cache_dtype,
    size_t vector_index, size_t dim, size_t head_dim, float dim_scale) {
  const int bits = tq_value_bits(kv_cache_dtype);
  const uint8_t *base = reinterpret_cast<const uint8_t *>(cache) +
                        vector_index *
                            tq_value_vector_bytes(head_dim, kv_cache_dtype);
  const uint8_t code = tq_load_packed_code(base, dim, bits);
  const float norm =
      metadata[vector_index * kTqMetadataFloats + kTqValueNormSlot];
  return tq_centroid_scaled(bits, code, dim_scale) * norm;
}

__device__ void prepare_tq_query_shared(const __nv_bfloat16 *q_vec,
                                        size_t head_dim, float *q_rot,
                                        float *q_sketch) {
  if (threadIdx.x < head_dim) {
    q_rot[threadIdx.x] = __bfloat162float(q_vec[threadIdx.x]);
  }
  __syncthreads();
  tq_rotate_forward_shared(q_rot, head_dim, kTqKeyRotationSalt);
  if (threadIdx.x < head_dim) {
    q_sketch[threadIdx.x] = q_rot[threadIdx.x];
  }
  __syncthreads();
  tq_rotate_forward_shared(q_sketch, head_dim, kTqKeyQjlSalt);
}

__device__ float load_tq_mse_original_component(const void *cache,
                                                const float *metadata,
                                                int kv_cache_dtype,
                                                size_t vector_index,
                                                size_t dim,
                                                size_t head_dim,
                                                int metadata_slot) {
  const bool key = metadata_slot == kTqKeyNormSlot;
  const int bits = key ? 2 : tq_value_bits(kv_cache_dtype);
  const uint8_t *base = reinterpret_cast<const uint8_t *>(cache) +
                        vector_index *
                            (key ? tq_key_vector_bytes(head_dim)
                                 : tq_value_vector_bytes(head_dim,
                                                         kv_cache_dtype));
  const float norm =
      metadata[vector_index * kTqMetadataFloats + metadata_slot];
  const uint32_t salt = key ? kTqKeyRotationSalt : kTqValueRotationSalt;
  float sum = 0.0f;
  for (size_t l = 0; l < head_dim; ++l) {
    const uint8_t code = tq_load_packed_code(base, l, bits);
    sum += tq_hadamard_entry(dim, l) * tq_rht_sign(l, salt, 1) *
           tq_centroid(bits, code, head_dim) * norm;
  }
  return tq_rht_sign(dim, salt, 0) * rsqrtf(static_cast<float>(head_dim)) *
         sum;
}

__device__ __forceinline__ float load_cache_value(
    const void *cache, const float *metadata, int kv_cache_dtype, size_t index,
    size_t head_dim, int metadata_slot) {
  if (is_tq_cache_dtype(kv_cache_dtype)) {
    const size_t vector_index = index / head_dim;
    const size_t dim = index % head_dim;
    return load_tq_mse_original_component(
        cache, metadata, kv_cache_dtype, vector_index, dim, head_dim,
        metadata_slot == 0 ? kTqKeyNormSlot : kTqValueNormSlot);
  }
  if (kv_cache_dtype == kKvCacheFp8) {
    return decode_e4m3(reinterpret_cast<const uint8_t *>(cache)[index]);
  }
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(cache)[index]);
}

__device__ __forceinline__ void store_cache_value(void *cache,
                                                  int kv_cache_dtype,
                                                  size_t index,
                                                  __nv_bfloat16 value) {
  if (kv_cache_dtype == kKvCacheFp8) {
    reinterpret_cast<uint8_t *>(cache)[index] =
        encode_e4m3(__bfloat162float(value));
  } else {
    reinterpret_cast<__nv_bfloat16 *>(cache)[index] = value;
  }
}

__global__ void copy_kv_prefill_kernel(
    const __nv_bfloat16 *k, const __nv_bfloat16 *v, void *cache_k,
    void *cache_v, float *cache_metadata, int kv_cache_dtype,
    size_t start_position, size_t tokens, const int32_t *start_position_device,
    size_t kv_heads, size_t head_dim) {
  __shared__ size_t shared_start_position;
  __shared__ float tq_x[kTqMaxHeadDim];
  __shared__ float tq_residual[kTqMaxHeadDim];
  __shared__ uint8_t tq_codes[kTqMaxHeadDim];
  __shared__ float tq_warp_sums[8];
  const size_t kvh = blockIdx.x;
  const size_t token = blockIdx.y;
  if (token >= tokens) {
    return;
  }
  if (threadIdx.x == 0) {
    shared_start_position = start_position_device != nullptr
                                ? static_cast<size_t>(*start_position_device)
                                : start_position;
  }
  __syncthreads();
  if (is_tq_cache_dtype(kv_cache_dtype)) {
    const size_t src = (token * kv_heads + kvh) * head_dim;
    const size_t vector_index =
        (shared_start_position + token) * kv_heads + kvh;
    encode_tq_key_vector_parallel(k + src, reinterpret_cast<uint8_t *>(cache_k),
                                  cache_metadata, vector_index, head_dim, tq_x,
                                  tq_residual, tq_codes, tq_warp_sums);
    encode_tq_value_vector_parallel(v + src,
                                    reinterpret_cast<uint8_t *>(cache_v),
                                    cache_metadata, vector_index, head_dim,
                                    kv_cache_dtype, tq_x, tq_codes,
                                    tq_warp_sums);
    return;
  }
  for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const size_t src = (token * kv_heads + kvh) * head_dim + d;
    const size_t dst =
        ((shared_start_position + token) * kv_heads + kvh) * head_dim + d;
    store_cache_value(cache_k, kv_cache_dtype, dst, k[src]);
    store_cache_value(cache_v, kv_cache_dtype, dst, v[src]);
  }
}

// Decode-time attention with online softmax.
// Optimisations vs. the naive reference: Q is preloaded once into a register,
// the per-timestep QK reduction uses warp shuffles + a single shared-memory
// fan-in across at most 8 warps, and the new-token K/V write into the cache is
// fused into this kernel (one block per kv-group performs the store), letting
// callers skip the separate copy_kv launch. The current decode position is
// read from `position_device` when non-null so a CUDA-Graph capture can be
// reused across decode steps without re-recording.
__global__ void attention_decode_kernel(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k_new,
    const __nv_bfloat16 *v_new, void *cache_k, void *cache_v,
    float *cache_metadata, int kv_cache_dtype,
    __nv_bfloat16 *output, size_t position_scalar,
    const int32_t *position_device, qwen36_attention_shape_t shape) {
  __shared__ float warp_sums[8];
  __shared__ float score_share;
  __shared__ size_t shared_position;
  __shared__ float tq_q_rot[kTqMaxHeadDim];
  __shared__ float tq_q_sketch[kTqMaxHeadDim];
  __shared__ float tq_v_sram[kTqMaxHeadDim];

  const size_t qh = blockIdx.x;
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t kvh = qh / q_per_kv;
  const float scale = rsqrtf(static_cast<float>(shape.head_dim));
  const size_t d = threadIdx.x;
  const bool active = d < shape.head_dim;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31) >> 5;

  if (threadIdx.x == 0) {
    shared_position = position_device != nullptr
                          ? static_cast<size_t>(*position_device)
                          : position_scalar;
  }
  __syncthreads();
  const size_t position = shared_position;
  const bool tq_cache = is_tq_cache_dtype(kv_cache_dtype);
  const float tq_codebook_scale =
      tq_cache ? tq_dim_scale(shape.head_dim) : 1.0f;

  const float q_val =
      active ? __bfloat162float(q[qh * shape.head_dim + d]) : 0.0f;
  if (tq_cache) {
    prepare_tq_query_shared(q + qh * shape.head_dim, shape.head_dim, tq_q_rot,
                            tq_q_sketch);
  }

  // The new K/V for t == position are not yet in the cache; read them from the
  // input pointers and let block 0 of each kv-group write them back at the end
  // of the kernel.
  const float k_new_val =
      active ? __bfloat162float(k_new[kvh * shape.head_dim + d]) : 0.0f;
  const float v_new_val =
      active ? __bfloat162float(v_new[kvh * shape.head_dim + d]) : 0.0f;
  float tq_v_new_rot = v_new_val;
  if (tq_cache) {
    if (active) {
      tq_v_sram[d] = v_new_val;
    }
    __syncthreads();
    tq_rotate_forward_shared(tq_v_sram, shape.head_dim, kTqValueRotationSalt);
    if (active) {
      tq_v_new_rot = tq_v_sram[d];
    }
  }

  float acc = 0.0f;
  float max_score = -INFINITY;
  float denom = 0.0f;

  for (size_t t = 0; t <= position; ++t) {
    const bool is_new = (t == position);
    float local = 0.0f;
    if (active) {
      if (tq_cache && !is_new) {
        local = load_tq_key_score_component(
            cache_k, cache_metadata, t * shape.kv_heads + kvh, d,
            shape.head_dim, tq_q_rot[d], tq_q_sketch[d], tq_codebook_scale);
      } else {
        const float kv = is_new ? k_new_val
                                : load_cache_value(
                                      cache_k, cache_metadata, kv_cache_dtype,
                                      (t * shape.kv_heads + kvh) *
                                              shape.head_dim +
                                          d,
                                      shape.head_dim, 0);
        local = q_val * kv;
      }
    }

    // Warp-level reduction.
    for (int offset = 16; offset > 0; offset >>= 1) {
      local += __shfl_xor_sync(0xffffffff, local, offset);
    }
    if (lane_id == 0) {
      warp_sums[warp_id] = local;
    }
    __syncthreads();
    if (warp_id == 0) {
      float total = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffff, total, offset);
      }
      if (lane_id == 0) {
        score_share = total * scale;
      }
    }
    __syncthreads();
    const float score = score_share;

    const float new_max = fmaxf(max_score, score);
    const float old_scale =
        isinf(max_score) && max_score < 0.0f ? 0.0f : expf(max_score - new_max);
    const float score_scale = expf(score - new_max);
    if (active) {
      float vv = v_new_val;
      if (tq_cache) {
        vv = is_new ? tq_v_new_rot
                    : load_tq_rotated_value_component(
                          cache_v, cache_metadata, kv_cache_dtype,
                          t * shape.kv_heads + kvh, d, shape.head_dim,
                          tq_codebook_scale);
      } else if (!is_new) {
        vv = load_cache_value(cache_v, cache_metadata, kv_cache_dtype,
                              (t * shape.kv_heads + kvh) * shape.head_dim + d,
                              shape.head_dim, 1);
      }
      acc = acc * old_scale + score_scale * vv;
    }
    denom = denom * old_scale + score_scale;
    max_score = new_max;
  }

  if (tq_cache) {
    if (active) {
      tq_v_sram[d] = acc / denom;
    }
    __syncthreads();
    tq_rotate_inverse_shared(tq_v_sram, shape.head_dim, kTqValueRotationSalt);
  }
  if (active) {
    const float out_val = tq_cache ? tq_v_sram[d] : acc / denom;
    output[qh * shape.head_dim + d] = __float2bfloat16(out_val);
    // Block 0 of each kv-group writes the new K/V back to the cache.
    if (qh % q_per_kv == 0 && tq_cache) {
      if (threadIdx.x == 0) {
        const size_t vector_index = position * shape.kv_heads + kvh;
        encode_tq_key_vector(k_new + kvh * shape.head_dim,
                             reinterpret_cast<uint8_t *>(cache_k),
                             cache_metadata, vector_index, shape.head_dim);
        encode_tq_value_vector(v_new + kvh * shape.head_dim,
                               reinterpret_cast<uint8_t *>(cache_v),
                               cache_metadata, vector_index, shape.head_dim,
                               kv_cache_dtype);
      }
    } else if (qh % q_per_kv == 0) {
      const size_t cache_off =
          (position * shape.kv_heads + kvh) * shape.head_dim + d;
      store_cache_value(cache_k, kv_cache_dtype, cache_off,
                        k_new[kvh * shape.head_dim + d]);
      store_cache_value(cache_v, kv_cache_dtype, cache_off,
                        v_new[kvh * shape.head_dim + d]);
    }
  }
}

// Shared-memory bounds for the GQA-aware prefill kernel below. The current
// Qwen3.6 config (q_heads=24, kv_heads=4 -> q_per_kv=6, head_dim=256) sits
// comfortably under both bounds.
constexpr int kGqaMaxQPerKv = 8;
constexpr int kGqaMaxHeadDim = 256;
constexpr int kGqaMaxWarps = 8;

// Split-KV (FlashDecoding-style) kernels for batch=1 decode attention.
//
// On long contexts the per-q-head decode kernel runs 24 blocks sequentially
// across the timestep loop, leaving most of the 170 SMs on Blackwell idle.
// The split kernel partitions [0, position] into chunks of
// `kSplitTimestepsPerBlock` and assigns one block per (q_head, chunk),
// computing partial online-softmax outputs to scratch global memory. A
// follow-up reduction kernel combines the partials per q-head using the
// log-sum-exp identity.
//
// Runtime scratch is sized on the Rust side for `kMinSplitTimestepsPerBlock`;
// individual calls can pass a larger tile size to reduce launch/reduce
// overhead at medium contexts.
constexpr int kDefaultSplitTimestepsPerBlock = 512;
constexpr int kMinSplitTimestepsPerBlock = 64;

__global__ void attention_decode_split_kernel(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k_new,
    const __nv_bfloat16 *v_new, void *cache_k, void *cache_v,
    float *cache_metadata, int kv_cache_dtype,
    float *partial_acc, float *partial_max, float *partial_denom,
    size_t position_scalar, const int32_t *position_device,
    qwen36_attention_shape_t shape, int n_splits,
    int split_timesteps_per_block) {
  __shared__ float warp_sums[8];
  __shared__ float score_share;
  __shared__ size_t shared_position;
  __shared__ float tq_q_rot[kTqMaxHeadDim];
  __shared__ float tq_q_sketch[kTqMaxHeadDim];
  __shared__ float tq_v_sram[kTqMaxHeadDim];

  const size_t qh = blockIdx.x;
  const size_t split = blockIdx.y;
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t kvh = qh / q_per_kv;
  const size_t head_dim = shape.head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const size_t d = threadIdx.x;
  const bool active = d < head_dim;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31u) >> 5;

  if (threadIdx.x == 0) {
    shared_position = position_device != nullptr
                          ? static_cast<size_t>(*position_device)
                          : position_scalar;
  }
  __syncthreads();
  const size_t position = shared_position;
  const bool tq_cache = is_tq_cache_dtype(kv_cache_dtype);
  const float tq_codebook_scale = tq_cache ? tq_dim_scale(head_dim) : 1.0f;
  const size_t split_block =
      static_cast<size_t>(split_timesteps_per_block);
  const size_t t_start = split * split_block;
  size_t t_end = t_start + split_block;
  if (t_end > position + 1) {
    t_end = position + 1;
  }

  if (t_start >= position + 1) {
    // Empty split (caller pads n_splits to the worst case). Write softmax
    // identity values so the reduce kernel can multiply through unconditionally.
    if (active) {
      partial_acc[(qh * n_splits + split) * head_dim + d] = 0.0f;
    }
    if (threadIdx.x == 0) {
      partial_max[qh * n_splits + split] = -INFINITY;
      partial_denom[qh * n_splits + split] = 0.0f;
    }
    return;
  }

  const float q_val =
      active ? __bfloat162float(q[qh * head_dim + d]) : 0.0f;
  if (tq_cache) {
    prepare_tq_query_shared(q + qh * head_dim, head_dim, tq_q_rot,
                            tq_q_sketch);
  }
  const float k_new_val =
      active ? __bfloat162float(k_new[kvh * head_dim + d]) : 0.0f;
  const float v_new_val =
      active ? __bfloat162float(v_new[kvh * head_dim + d]) : 0.0f;
  float tq_v_new_rot = v_new_val;
  if (tq_cache) {
    if (active) {
      tq_v_sram[d] = v_new_val;
    }
    __syncthreads();
    tq_rotate_forward_shared(tq_v_sram, head_dim, kTqValueRotationSalt);
    if (active) {
      tq_v_new_rot = tq_v_sram[d];
    }
  }

  float acc = 0.0f;
  float max_score = -INFINITY;
  float denom = 0.0f;

  for (size_t t = t_start; t < t_end; ++t) {
    const bool is_new = (t == position);
    float local = 0.0f;
    if (active) {
      if (tq_cache && !is_new) {
        local = load_tq_key_score_component(
            cache_k, cache_metadata, t * shape.kv_heads + kvh, d, head_dim,
            tq_q_rot[d], tq_q_sketch[d], tq_codebook_scale);
      } else {
        const float kv = is_new ? k_new_val
                                : load_cache_value(
                                      cache_k, cache_metadata, kv_cache_dtype,
                                      (t * shape.kv_heads + kvh) *
                                              shape.head_dim +
                                          d,
                                      shape.head_dim, 0);
        local = q_val * kv;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      local += __shfl_xor_sync(0xffffffff, local, offset);
    }
    if (lane_id == 0) {
      warp_sums[warp_id] = local;
    }
    __syncthreads();
    if (warp_id == 0) {
      float total = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffff, total, offset);
      }
      if (lane_id == 0) {
        score_share = total * scale;
      }
    }
    __syncthreads();
    const float score = score_share;

    const float new_max = fmaxf(max_score, score);
    const float old_scale =
        isinf(max_score) && max_score < 0.0f ? 0.0f : expf(max_score - new_max);
    const float score_scale = expf(score - new_max);
    if (active) {
      float vv = v_new_val;
      if (tq_cache) {
        vv = is_new ? tq_v_new_rot
                    : load_tq_rotated_value_component(
                          cache_v, cache_metadata, kv_cache_dtype,
                          t * shape.kv_heads + kvh, d, head_dim,
                          tq_codebook_scale);
      } else if (!is_new) {
        vv = load_cache_value(cache_v, cache_metadata, kv_cache_dtype,
                              (t * shape.kv_heads + kvh) * shape.head_dim + d,
                              shape.head_dim, 1);
      }
      acc = acc * old_scale + score_scale * vv;
    }
    denom = denom * old_scale + score_scale;
    max_score = new_max;
  }

  if (tq_cache) {
    if (active) {
      tq_v_sram[d] = acc;
    }
    __syncthreads();
    tq_rotate_inverse_shared(tq_v_sram, head_dim, kTqValueRotationSalt);
  }
  if (active) {
    partial_acc[(qh * n_splits + split) * head_dim + d] =
        tq_cache ? tq_v_sram[d] : acc;
  }
  if (threadIdx.x == 0) {
    partial_max[qh * n_splits + split] = max_score;
    partial_denom[qh * n_splits + split] = denom;
  }

  // The split that owns `position` writes the new K/V to the cache. Gate to
  // the first q-head of each kv-group so each cache row is written exactly
  // once even though every q-head in the group hits this branch.
  if (t_start <= position && position < t_end &&
      (qh % q_per_kv == 0) && tq_cache) {
    if (threadIdx.x == 0) {
      const size_t vector_index = position * shape.kv_heads + kvh;
      encode_tq_key_vector(k_new + kvh * head_dim,
                           reinterpret_cast<uint8_t *>(cache_k),
                           cache_metadata, vector_index, head_dim);
      encode_tq_value_vector(v_new + kvh * head_dim,
                             reinterpret_cast<uint8_t *>(cache_v),
                             cache_metadata, vector_index, head_dim,
                             kv_cache_dtype);
    }
  } else if (t_start <= position && position < t_end && active &&
             (qh % q_per_kv == 0)) {
    const size_t cache_off =
        (position * shape.kv_heads + kvh) * shape.head_dim + d;
    store_cache_value(cache_k, kv_cache_dtype, cache_off,
                      k_new[kvh * head_dim + d]);
    store_cache_value(cache_v, kv_cache_dtype, cache_off,
                      v_new[kvh * head_dim + d]);
  }
}

__global__ void attention_decode_split_gqa_kernel(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k_new,
    const __nv_bfloat16 *v_new, void *cache_k,
    void *cache_v, float *cache_metadata, int kv_cache_dtype, float *partial_acc, float *partial_max,
    float *partial_denom, size_t position_scalar,
    const int32_t *position_device, qwen36_attention_shape_t shape,
    int n_splits, int split_timesteps_per_block) {
  __shared__ size_t shared_position;
  __shared__ float kv_sram[kGqaMaxHeadDim];
  __shared__ float warp_partials[kGqaMaxWarps][kGqaMaxQPerKv];
  __shared__ float max_score_sram[kGqaMaxQPerKv];
  __shared__ float denom_sram[kGqaMaxQPerKv];
  __shared__ float scale_old[kGqaMaxQPerKv];
  __shared__ float scale_new[kGqaMaxQPerKv];
  __shared__ float tq_q_rot[kGqaMaxQPerKv][kTqMaxHeadDim];
  __shared__ float tq_q_sketch[kGqaMaxQPerKv][kTqMaxHeadDim];

  const size_t kvh = blockIdx.x;
  const size_t split = blockIdx.y;
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t head_dim = shape.head_dim;
  const float qk_scale = rsqrtf(static_cast<float>(head_dim));
  const bool tile_active = threadIdx.x < head_dim;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31u) >> 5;

  if (threadIdx.x == 0) {
    shared_position = position_device != nullptr
                          ? static_cast<size_t>(*position_device)
                          : position_scalar;
  }
  if (threadIdx.x < q_per_kv) {
    max_score_sram[threadIdx.x] = -INFINITY;
    denom_sram[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  const size_t position = shared_position;
  const bool tq_cache = is_tq_cache_dtype(kv_cache_dtype);
  const float tq_codebook_scale = tq_cache ? tq_dim_scale(head_dim) : 1.0f;
  const size_t split_block =
      static_cast<size_t>(split_timesteps_per_block);
  const size_t t_start = split * split_block;
  size_t t_end = t_start + split_block;
  if (t_end > position + 1) {
    t_end = position + 1;
  }

  if (t_start >= position + 1) {
    if (tile_active) {
      for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
        const size_t qh = kvh * q_per_kv + qh_local;
        partial_acc[(qh * n_splits + split) * head_dim + threadIdx.x] = 0.0f;
      }
    }
    if (threadIdx.x < q_per_kv) {
      const size_t qh = kvh * q_per_kv + threadIdx.x;
      partial_max[qh * n_splits + split] = -INFINITY;
      partial_denom[qh * n_splits + split] = 0.0f;
    }
    return;
  }

  float q_local[kGqaMaxQPerKv];
#pragma unroll
  for (int i = 0; i < kGqaMaxQPerKv; ++i) {
    q_local[i] = 0.0f;
  }
  if (tile_active) {
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const size_t qh = kvh * q_per_kv + qh_local;
      q_local[qh_local] =
          __bfloat162float(q[qh * head_dim + threadIdx.x]);
    }
  }
  if (tq_cache) {
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const size_t qh = kvh * q_per_kv + qh_local;
      prepare_tq_query_shared(q + qh * head_dim, head_dim,
                              tq_q_rot[qh_local], tq_q_sketch[qh_local]);
    }
  }

  float acc[kGqaMaxQPerKv];
#pragma unroll
  for (int i = 0; i < kGqaMaxQPerKv; ++i) {
    acc[i] = 0.0f;
  }

  const float k_new_val =
      tile_active ? __bfloat162float(k_new[kvh * head_dim + threadIdx.x])
                  : 0.0f;
  const float v_new_val =
      tile_active ? __bfloat162float(v_new[kvh * head_dim + threadIdx.x])
                  : 0.0f;
  float tq_v_new_rot = v_new_val;
  if (tq_cache) {
    if (tile_active) {
      kv_sram[threadIdx.x] = v_new_val;
    }
    __syncthreads();
    tq_rotate_forward_shared(kv_sram, head_dim, kTqValueRotationSalt);
    if (tile_active) {
      tq_v_new_rot = kv_sram[threadIdx.x];
    }
    __syncthreads();
  }

  for (size_t t = t_start; t < t_end; ++t) {
    const bool is_new = (t == position);
    if (!tq_cache || is_new) {
      if (tile_active) {
        kv_sram[threadIdx.x] =
            is_new ? k_new_val
                   : load_cache_value(
                         cache_k, cache_metadata, kv_cache_dtype,
                         (t * shape.kv_heads + kvh) * head_dim + threadIdx.x,
                         head_dim, 0);
      }
      __syncthreads();
    }

    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      float local = 0.0f;
      if (tile_active) {
        local = tq_cache && !is_new
                    ? load_tq_key_score_component(
                          cache_k, cache_metadata, t * shape.kv_heads + kvh,
                          threadIdx.x, head_dim, tq_q_rot[qh_local][threadIdx.x],
                          tq_q_sketch[qh_local][threadIdx.x],
                          tq_codebook_scale)
                    : q_local[qh_local] * kv_sram[threadIdx.x];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_xor_sync(0xffffffff, local, offset);
      }
      if (lane_id == 0) {
        warp_partials[warp_id][qh_local] = local;
      }
    }
    __syncthreads();

    if (threadIdx.x < q_per_kv) {
      float total = 0.0f;
      for (unsigned w = 0; w < n_warps; ++w) {
        total += warp_partials[w][threadIdx.x];
      }
      const float score = total * qk_scale;
      const float old_max = max_score_sram[threadIdx.x];
      const float new_max = fmaxf(old_max, score);
      const float so =
          isinf(old_max) && old_max < 0.0f ? 0.0f : expf(old_max - new_max);
      const float sn = expf(score - new_max);
      scale_old[threadIdx.x] = so;
      scale_new[threadIdx.x] = sn;
      denom_sram[threadIdx.x] = denom_sram[threadIdx.x] * so + sn;
      max_score_sram[threadIdx.x] = new_max;
    }
    __syncthreads();

    if (tq_cache) {
      if (tile_active) {
        kv_sram[threadIdx.x] =
            is_new ? tq_v_new_rot
                   : load_tq_rotated_value_component(
                         cache_v, cache_metadata, kv_cache_dtype,
                         t * shape.kv_heads + kvh, threadIdx.x, head_dim,
                         tq_codebook_scale);
      }
    } else if (tile_active) {
      kv_sram[threadIdx.x] =
          is_new ? v_new_val
                 : load_cache_value(
                       cache_v, cache_metadata, kv_cache_dtype,
                       (t * shape.kv_heads + kvh) * head_dim + threadIdx.x,
                       head_dim, 1);
    }
    __syncthreads();

    if (tile_active) {
      const float v_val = kv_sram[threadIdx.x];
#pragma unroll
      for (int qh_local = 0; qh_local < kGqaMaxQPerKv; ++qh_local) {
        if (qh_local >= static_cast<int>(q_per_kv)) {
          break;
        }
        acc[qh_local] = acc[qh_local] * scale_old[qh_local] +
                        scale_new[qh_local] * v_val;
      }
    }
    __syncthreads();
  }

  if (tq_cache) {
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const size_t qh = kvh * q_per_kv + qh_local;
      if (tile_active) {
        kv_sram[threadIdx.x] = acc[qh_local];
      }
      __syncthreads();
      tq_rotate_inverse_shared(kv_sram, head_dim, kTqValueRotationSalt);
      if (tile_active) {
        partial_acc[(qh * n_splits + split) * head_dim + threadIdx.x] =
            kv_sram[threadIdx.x];
      }
      __syncthreads();
    }
  } else if (tile_active) {
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const size_t qh = kvh * q_per_kv + qh_local;
      partial_acc[(qh * n_splits + split) * head_dim + threadIdx.x] =
          acc[qh_local];
    }
  }
  if (threadIdx.x < q_per_kv) {
    const size_t qh = kvh * q_per_kv + threadIdx.x;
    partial_max[qh * n_splits + split] = max_score_sram[threadIdx.x];
    partial_denom[qh * n_splits + split] = denom_sram[threadIdx.x];
  }

  if (t_start <= position && position < t_end && tq_cache) {
    if (threadIdx.x == 0) {
      const size_t vector_index = position * shape.kv_heads + kvh;
      encode_tq_key_vector(k_new + kvh * head_dim,
                           reinterpret_cast<uint8_t *>(cache_k),
                           cache_metadata, vector_index, head_dim);
      encode_tq_value_vector(v_new + kvh * head_dim,
                             reinterpret_cast<uint8_t *>(cache_v),
                             cache_metadata, vector_index, head_dim,
                             kv_cache_dtype);
    }
  } else if (t_start <= position && position < t_end && tile_active) {
    const size_t cache_off =
        (position * shape.kv_heads + kvh) * head_dim + threadIdx.x;
    store_cache_value(cache_k, kv_cache_dtype, cache_off,
                      k_new[kvh * head_dim + threadIdx.x]);
    store_cache_value(cache_v, kv_cache_dtype, cache_off,
                      v_new[kvh * head_dim + threadIdx.x]);
  }
}

__global__ void attention_decode_reduce_kernel(
    const float *partial_acc, const float *partial_max,
    const float *partial_denom, __nv_bfloat16 *output,
    qwen36_attention_shape_t shape, int n_splits) {
  __shared__ float gmax;
  __shared__ float gdenom;

  const size_t qh = blockIdx.x;
  const size_t head_dim = shape.head_dim;
  const size_t d = threadIdx.x;

  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
      m = fmaxf(m, partial_max[qh * n_splits + s]);
    }
    float dn = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
      const float pm = partial_max[qh * n_splits + s];
      const float pd = partial_denom[qh * n_splits + s];
      const float scale =
          isinf(pm) && pm < 0.0f ? 0.0f : expf(pm - m);
      dn += pd * scale;
    }
    gmax = m;
    gdenom = dn;
  }
  __syncthreads();
  if (d >= head_dim) {
    return;
  }
  const float m = gmax;
  const float dn = gdenom;
  float acc_total = 0.0f;
  for (int s = 0; s < n_splits; ++s) {
    const float pm = partial_max[qh * n_splits + s];
    const float pa = partial_acc[(qh * n_splits + s) * head_dim + d];
    const float scale = isinf(pm) && pm < 0.0f ? 0.0f : expf(pm - m);
    acc_total += pa * scale;
  }
  output[qh * head_dim + d] = __float2bfloat16(acc_total / dn);
}

__global__ void attention_prefill_split_kernel(
    const __nv_bfloat16 *q, const void *cache_k,
    const void *cache_v, const float *cache_metadata, int kv_cache_dtype, float *partial_acc, float *partial_max,
    float *partial_denom, size_t start_position_scalar,
    const int32_t *start_position_device, size_t token,
    qwen36_attention_shape_t shape, int n_splits,
    int split_timesteps_per_block) {
  __shared__ float warp_sums[8];
  __shared__ float score_share;
  __shared__ size_t shared_start_position;
  __shared__ float tq_q_rot[kTqMaxHeadDim];
  __shared__ float tq_q_sketch[kTqMaxHeadDim];
  __shared__ float tq_v_sram[kTqMaxHeadDim];

  const size_t qh = blockIdx.x;
  const size_t split = blockIdx.y;
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t kvh = qh / q_per_kv;
  const size_t head_dim = shape.head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const size_t d = threadIdx.x;
  const bool active = d < head_dim;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31u) >> 5;

  if (threadIdx.x == 0) {
    shared_start_position =
        start_position_device != nullptr
            ? static_cast<size_t>(*start_position_device)
            : start_position_scalar;
  }
  __syncthreads();
  const size_t position = shared_start_position + token;
  const bool tq_cache = is_tq_cache_dtype(kv_cache_dtype);
  const float tq_codebook_scale = tq_cache ? tq_dim_scale(head_dim) : 1.0f;
  const size_t split_block =
      static_cast<size_t>(split_timesteps_per_block);
  const size_t t_start = split * split_block;
  size_t t_end = t_start + split_block;
  if (t_end > position + 1) {
    t_end = position + 1;
  }

  if (t_start >= position + 1) {
    if (active) {
      partial_acc[(qh * n_splits + split) * head_dim + d] = 0.0f;
    }
    if (threadIdx.x == 0) {
      partial_max[qh * n_splits + split] = -INFINITY;
      partial_denom[qh * n_splits + split] = 0.0f;
    }
    return;
  }

  const float q_val =
      active ? __bfloat162float(
                   q[(token * shape.q_heads + qh) * head_dim + d])
             : 0.0f;
  if (tq_cache) {
    prepare_tq_query_shared(q + (token * shape.q_heads + qh) * head_dim,
                            head_dim, tq_q_rot, tq_q_sketch);
  }
  float acc = 0.0f;
  float max_score = -INFINITY;
  float denom = 0.0f;

  for (size_t t = t_start; t < t_end; ++t) {
    float local = 0.0f;
    if (active) {
      if (tq_cache) {
        local = load_tq_key_score_component(
            cache_k, cache_metadata, t * shape.kv_heads + kvh, d, head_dim,
            tq_q_rot[d], tq_q_sketch[d], tq_codebook_scale);
      } else {
        const float kv = load_cache_value(
            cache_k, cache_metadata, kv_cache_dtype,
            (t * shape.kv_heads + kvh) * head_dim + d, head_dim, 0);
        local = q_val * kv;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      local += __shfl_xor_sync(0xffffffff, local, offset);
    }
    if (lane_id == 0) {
      warp_sums[warp_id] = local;
    }
    __syncthreads();
    if (warp_id == 0) {
      float total = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffff, total, offset);
      }
      if (lane_id == 0) {
        score_share = total * scale;
      }
    }
    __syncthreads();

    const float score = score_share;
    const float new_max = fmaxf(max_score, score);
    const float old_scale =
        isinf(max_score) && max_score < 0.0f ? 0.0f : expf(max_score - new_max);
    const float score_scale = expf(score - new_max);
    if (active) {
      const float vv =
          tq_cache
              ? load_tq_rotated_value_component(
                    cache_v, cache_metadata, kv_cache_dtype,
                    t * shape.kv_heads + kvh, d, head_dim, tq_codebook_scale)
              : load_cache_value(cache_v, cache_metadata, kv_cache_dtype,
                                 (t * shape.kv_heads + kvh) * head_dim + d,
                                 head_dim, 1);
      acc = acc * old_scale + score_scale * vv;
    }
    denom = denom * old_scale + score_scale;
    max_score = new_max;
  }

  if (tq_cache) {
    if (active) {
      tq_v_sram[d] = acc;
    }
    __syncthreads();
    tq_rotate_inverse_shared(tq_v_sram, head_dim, kTqValueRotationSalt);
  }
  if (active) {
    partial_acc[(qh * n_splits + split) * head_dim + d] =
        tq_cache ? tq_v_sram[d] : acc;
  }
  if (threadIdx.x == 0) {
    partial_max[qh * n_splits + split] = max_score;
    partial_denom[qh * n_splits + split] = denom;
  }
}

// GQA-aware prefill kernel. Same online-softmax structure as the per-q-head
// kernel below, but lays the grid out as (kv_heads, tokens) so the q_per_kv
// queries that share each kv-head also share the K/V cache loads. With
// q_per_kv = 6 on Qwen3.6 this cuts the BF16 cache traffic for the Q*K and
// V matmuls by ~6x (the L2 used to absorb the redundancy via the per-q-head
// kernel; this version eliminates it outright). The per-kv-head launch
// configuration only makes sense when the block count stays large — i.e.
// during prefill, where `tokens` is in the thousands. The decode call site
// keeps using the per-q-head kernel because batch=1 decode only has 4
// kv-heads to dispatch and would starve the GPU.
__global__ void attention_prefill_gqa_kernel(
    const __nv_bfloat16 *q, const void *cache_k,
    const void *cache_v, const float *cache_metadata, int kv_cache_dtype, __nv_bfloat16 *output,
    size_t start_position, const int32_t *start_position_device, size_t tokens,
    qwen36_attention_shape_t shape) {
  __shared__ size_t shared_start_position;
  __shared__ float kv_sram[kGqaMaxHeadDim];
  __shared__ float warp_partials[kGqaMaxWarps][kGqaMaxQPerKv];
  // Online-softmax state lives in shared memory so dim-threads can read it
  // when applying the per-step rescale to their accumulators. Only the
  // q-head's owning thread (threadIdx.x == qh_local) writes these.
  __shared__ float max_score_sram[kGqaMaxQPerKv];
  __shared__ float denom_sram[kGqaMaxQPerKv];
  __shared__ float scale_old[kGqaMaxQPerKv];
  __shared__ float scale_new[kGqaMaxQPerKv];
  __shared__ float tq_q_rot[kGqaMaxQPerKv][kTqMaxHeadDim];
  __shared__ float tq_q_sketch[kGqaMaxQPerKv][kTqMaxHeadDim];

  const size_t kvh = blockIdx.x;
  const size_t token = blockIdx.y;
  if (token >= tokens) {
    return;
  }
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t head_dim = shape.head_dim;
  const float qk_scale = rsqrtf(static_cast<float>(head_dim));
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31u) >> 5;
  const bool tile_active = threadIdx.x < head_dim;

  if (threadIdx.x == 0) {
    shared_start_position = start_position_device != nullptr
                                ? static_cast<size_t>(*start_position_device)
                                : start_position;
  }
  if (threadIdx.x < q_per_kv) {
    max_score_sram[threadIdx.x] = -INFINITY;
    denom_sram[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  const size_t position = shared_start_position + token;
  const bool tq_cache = is_tq_cache_dtype(kv_cache_dtype);
  const float tq_codebook_scale = tq_cache ? tq_dim_scale(head_dim) : 1.0f;

  float q_local[kGqaMaxQPerKv];
#pragma unroll
  for (int i = 0; i < kGqaMaxQPerKv; ++i) {
    q_local[i] = 0.0f;
  }
  if (tile_active) {
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const size_t qh = kvh * q_per_kv + qh_local;
      q_local[qh_local] = __bfloat162float(
          q[(token * shape.q_heads + qh) * head_dim + threadIdx.x]);
    }
  }
  if (tq_cache) {
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const size_t qh = kvh * q_per_kv + qh_local;
      prepare_tq_query_shared(
          q + (token * shape.q_heads + qh) * head_dim, head_dim,
          tq_q_rot[qh_local], tq_q_sketch[qh_local]);
    }
  }

  float acc[kGqaMaxQPerKv];
#pragma unroll
  for (int i = 0; i < kGqaMaxQPerKv; ++i) {
    acc[i] = 0.0f;
  }

  for (size_t t = 0; t <= position; ++t) {
    if (!tq_cache) {
      if (tile_active) {
        kv_sram[threadIdx.x] = load_cache_value(
            cache_k, cache_metadata, kv_cache_dtype,
            (t * shape.kv_heads + kvh) * head_dim + threadIdx.x, head_dim, 0);
      }
      __syncthreads();
    }

    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      float local = 0.0f;
      if (tile_active) {
        local = tq_cache
                    ? load_tq_key_score_component(
                          cache_k, cache_metadata, t * shape.kv_heads + kvh,
                          threadIdx.x, head_dim, tq_q_rot[qh_local][threadIdx.x],
                          tq_q_sketch[qh_local][threadIdx.x],
                          tq_codebook_scale)
                    : q_local[qh_local] * kv_sram[threadIdx.x];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_xor_sync(0xffffffff, local, offset);
      }
      if (lane_id == 0) {
        warp_partials[warp_id][qh_local] = local;
      }
    }
    __syncthreads();

    if (threadIdx.x < q_per_kv) {
      float total = 0.0f;
      for (unsigned w = 0; w < n_warps; ++w) {
        total += warp_partials[w][threadIdx.x];
      }
      const float score = total * qk_scale;
      const float old_max = max_score_sram[threadIdx.x];
      const float new_max = fmaxf(old_max, score);
      const float so =
          isinf(old_max) && old_max < 0.0f ? 0.0f : expf(old_max - new_max);
      const float sn = expf(score - new_max);
      scale_old[threadIdx.x] = so;
      scale_new[threadIdx.x] = sn;
      denom_sram[threadIdx.x] = denom_sram[threadIdx.x] * so + sn;
      max_score_sram[threadIdx.x] = new_max;
    }
    __syncthreads();

    if (tq_cache) {
      if (tile_active) {
        kv_sram[threadIdx.x] = load_tq_rotated_value_component(
            cache_v, cache_metadata, kv_cache_dtype, t * shape.kv_heads + kvh,
            threadIdx.x, head_dim, tq_codebook_scale);
      }
    } else if (tile_active) {
      kv_sram[threadIdx.x] = load_cache_value(
          cache_v, cache_metadata, kv_cache_dtype,
          (t * shape.kv_heads + kvh) * head_dim + threadIdx.x, head_dim, 1);
    }
    __syncthreads();

    if (tile_active) {
      const float v_val = kv_sram[threadIdx.x];
#pragma unroll
      for (int qh_local = 0; qh_local < kGqaMaxQPerKv; ++qh_local) {
        if (qh_local >= static_cast<int>(q_per_kv)) {
          break;
        }
        acc[qh_local] = acc[qh_local] * scale_old[qh_local] +
                        scale_new[qh_local] * v_val;
      }
    }
    __syncthreads();
  }

  if (tq_cache) {
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const float dn = denom_sram[qh_local];
      const size_t qh = kvh * q_per_kv + qh_local;
      if (tile_active) {
        kv_sram[threadIdx.x] = acc[qh_local] / dn;
      }
      __syncthreads();
      tq_rotate_inverse_shared(kv_sram, head_dim, kTqValueRotationSalt);
      if (tile_active) {
        output[(token * shape.q_heads + qh) * head_dim + threadIdx.x] =
            __float2bfloat16(kv_sram[threadIdx.x]);
      }
      __syncthreads();
    }
  } else if (tile_active) {
    for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
      const float dn = denom_sram[qh_local];
      const size_t qh = kvh * q_per_kv + qh_local;
      output[(token * shape.q_heads + qh) * head_dim + threadIdx.x] =
          __float2bfloat16(acc[qh_local] / dn);
    }
  }
}

__global__ void attention_prefill_kernel(
    const __nv_bfloat16 *q, const void *cache_k,
    const void *cache_v, const float *cache_metadata, int kv_cache_dtype, __nv_bfloat16 *output,
    size_t start_position, const int32_t *start_position_device, size_t tokens,
    qwen36_attention_shape_t shape) {
  __shared__ float warp_sums[8];
  __shared__ float score_share;
  __shared__ size_t shared_start_position;
  __shared__ float tq_q_rot[kTqMaxHeadDim];
  __shared__ float tq_q_sketch[kTqMaxHeadDim];
  __shared__ float tq_v_sram[kTqMaxHeadDim];

  const size_t qh = blockIdx.x;
  const size_t token = blockIdx.y;
  if (token >= tokens) {
    return;
  }
  if (threadIdx.x == 0) {
    shared_start_position = start_position_device != nullptr
                                ? static_cast<size_t>(*start_position_device)
                                : start_position;
  }
  __syncthreads();
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t kvh = qh / q_per_kv;
  const size_t position = shared_start_position + token;
  const float scale = rsqrtf(static_cast<float>(shape.head_dim));
  const __nv_bfloat16 *q_tok =
      q + (token * shape.q_heads + qh) * shape.head_dim;
  const size_t d = threadIdx.x;
  const bool active = d < shape.head_dim;
  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane_id = threadIdx.x & 31;
  const unsigned n_warps = (blockDim.x + 31) >> 5;
  const bool tq_cache = is_tq_cache_dtype(kv_cache_dtype);
  const float tq_codebook_scale =
      tq_cache ? tq_dim_scale(shape.head_dim) : 1.0f;
  const float q_val = active ? __bfloat162float(q_tok[d]) : 0.0f;
  if (tq_cache) {
    prepare_tq_query_shared(q_tok, shape.head_dim, tq_q_rot, tq_q_sketch);
  }
  float acc = 0.0f;
  float max_score = -INFINITY;
  float denom = 0.0f;

  for (size_t t = 0; t <= position; ++t) {
    float local = 0.0f;
    if (active) {
      local = tq_cache
                  ? load_tq_key_score_component(
                        cache_k, cache_metadata, t * shape.kv_heads + kvh, d,
                        shape.head_dim, tq_q_rot[d], tq_q_sketch[d],
                        tq_codebook_scale)
                  : q_val *
                        load_cache_value(cache_k, cache_metadata, kv_cache_dtype,
                                         (t * shape.kv_heads + kvh) *
                                                 shape.head_dim +
                                             d,
                                         shape.head_dim, 0);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      local += __shfl_xor_sync(0xffffffff, local, offset);
    }
    if (lane_id == 0) {
      warp_sums[warp_id] = local;
    }
    __syncthreads();
    if (warp_id == 0) {
      float total = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffff, total, offset);
      }
      if (lane_id == 0) {
        score_share = total * scale;
      }
    }
    __syncthreads();

    const float score = score_share;
    const float new_max = fmaxf(max_score, score);
    const float old_scale =
        isinf(max_score) && max_score < 0.0f ? 0.0f : expf(max_score - new_max);
    const float score_scale = expf(score - new_max);
    if (active) {
      const float vv =
          tq_cache
              ? load_tq_rotated_value_component(
                    cache_v, cache_metadata, kv_cache_dtype,
                    t * shape.kv_heads + kvh, d, shape.head_dim,
                    tq_codebook_scale)
              : load_cache_value(cache_v, cache_metadata, kv_cache_dtype,
                                 (t * shape.kv_heads + kvh) * shape.head_dim + d,
                                 shape.head_dim, 1);
      acc = acc * old_scale + score_scale * vv;
    }
    denom = denom * old_scale + score_scale;
    max_score = new_max;
  }
  if (tq_cache) {
    if (active) {
      tq_v_sram[d] = acc / denom;
    }
    __syncthreads();
    tq_rotate_inverse_shared(tq_v_sram, shape.head_dim, kTqValueRotationSalt);
  }
  if (active) {
    const float out_val = tq_cache ? tq_v_sram[d] : acc / denom;
    output[(token * shape.q_heads + qh) * shape.head_dim + d] =
        __float2bfloat16(out_val);
  }
}

} // namespace

extern "C" int
qwen36_attention_prefill(const qwen36_attention_prefill_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->tokens == 0 || spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0 ||
      spec->v_bf16.ptr == 0 || spec->kv_cache_k.ptr == 0 ||
      spec->kv_cache_v.ptr == 0 || spec->output_bf16.ptr == 0 ||
      spec->shape.q_heads == 0 || spec->shape.kv_heads == 0 ||
      spec->shape.head_dim == 0 || spec->shape.head_dim > 256 ||
      spec->shape.q_heads % spec->shape.kv_heads != 0 ||
      (spec->kv_cache_dtype != kKvCacheBf16 &&
       spec->kv_cache_dtype != kKvCacheFp8 &&
       spec->kv_cache_dtype != kKvCacheTurboQuant3 &&
       spec->kv_cache_dtype != kKvCacheTurboQuant35) ||
      (is_tq_cache_dtype(spec->kv_cache_dtype) &&
       (spec->kv_cache_metadata.ptr == 0 ||
        !tq_supported_head_dim(spec->shape.head_dim)))) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  const int threads = 256;
  const dim3 copy_grid(static_cast<unsigned int>(spec->shape.kv_heads),
                       static_cast<unsigned int>(spec->tokens));
  copy_kv_prefill_kernel<<<copy_grid, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->k_bf16),
      ptr<const __nv_bfloat16>(spec->v_bf16),
      ptr<void>(spec->kv_cache_k), ptr<void>(spec->kv_cache_v),
      ptr<float>(spec->kv_cache_metadata),
      spec->kv_cache_dtype, spec->start_position, spec->tokens,
      ptr<const int32_t>(spec->start_position_device_i32), spec->shape.kv_heads,
      spec->shape.head_dim);

  const bool partials_present = spec->partial_acc_f32.ptr != 0 &&
                                spec->partial_max_f32.ptr != 0 &&
                                spec->partial_denom_f32.ptr != 0;
  if (partials_present && spec->prefill_n_splits >= 2 && spec->tokens <= 2) {
    const int split_timesteps_per_block =
        spec->split_timesteps_per_block == 0
            ? kDefaultSplitTimestepsPerBlock
            : static_cast<int>(spec->split_timesteps_per_block);
    if (split_timesteps_per_block < kMinSplitTimestepsPerBlock) {
      return QWEN36_STATUS_INVALID_ARGUMENT;
    }
    unsigned int split_threads = static_cast<unsigned int>(spec->shape.head_dim);
    split_threads = (split_threads + 31u) & ~31u;
    if (split_threads == 0) {
      split_threads = 32u;
    } else if (split_threads > 256u) {
      split_threads = 256u;
    }
    const int n_splits = static_cast<int>(spec->prefill_n_splits);
    const dim3 split_grid(static_cast<unsigned int>(spec->shape.q_heads),
                          static_cast<unsigned int>(n_splits));
    for (size_t token = 0; token < spec->tokens; ++token) {
      attention_prefill_split_kernel<<<split_grid, split_threads, 0,
                                       qwen36_internal_active_stream()>>>(
          ptr<const __nv_bfloat16>(spec->q_bf16),
          ptr<const void>(spec->kv_cache_k),
          ptr<const void>(spec->kv_cache_v),
          ptr<const float>(spec->kv_cache_metadata), spec->kv_cache_dtype,
          ptr<float>(spec->partial_acc_f32),
          ptr<float>(spec->partial_max_f32),
          ptr<float>(spec->partial_denom_f32), spec->start_position,
          ptr<const int32_t>(spec->start_position_device_i32), token,
          spec->shape, n_splits, split_timesteps_per_block);
      cudaError_t split_err = cudaGetLastError();
      if (split_err != cudaSuccess) {
        return QWEN36_STATUS_CUDA_ERROR;
      }
      __nv_bfloat16 *token_output =
          ptr<__nv_bfloat16>(spec->output_bf16) +
          token * spec->shape.q_heads * spec->shape.head_dim;
      attention_decode_reduce_kernel<<<
          static_cast<unsigned int>(spec->shape.q_heads), split_threads, 0,
          qwen36_internal_active_stream()>>>(
          ptr<const float>(spec->partial_acc_f32),
          ptr<const float>(spec->partial_max_f32),
          ptr<const float>(spec->partial_denom_f32), token_output,
          spec->shape, n_splits);
      cudaError_t reduce_err = cudaGetLastError();
      if (reduce_err != cudaSuccess) {
        return QWEN36_STATUS_CUDA_ERROR;
      }
    }
    return QWEN36_STATUS_SUCCESS;
  }

  // Prefer the GQA-aware kernel for the common Qwen3.6 shape: it lays out
  // the grid as (kv_heads × tokens) instead of (q_heads × tokens) and
  // shares each cache row across the q_per_kv queries that consume it,
  // eliminating the (q_per_kv − 1)× redundant cache reads the per-q-head
  // kernel relied on the L2 cache to absorb. With prefill `tokens` in the
  // hundreds-to-thousands the grid stays large enough to saturate the GPU
  // even at q_per_kv = 6.
  // Threshold below which the per-q-head kernel still wins. The GQA kernel
  // does 6x the per-block work and only `kv_heads` blocks per token, so for
  // very short chunks (notably 2-token MTP verify chunks) it under-utilizes
  // the GPU even at q_per_kv = 6. Empirically `tokens >= 16` is where the
  // crossover lands on Qwen3.6 / Blackwell.
  constexpr size_t kPrefillGqaMinTokens = 16;

  const size_t q_per_kv = spec->shape.q_heads / spec->shape.kv_heads;
  const bool gqa_eligible =
      spec->shape.head_dim <= static_cast<size_t>(kGqaMaxHeadDim) &&
      q_per_kv <= static_cast<size_t>(kGqaMaxQPerKv) && q_per_kv > 1 &&
      spec->tokens >= kPrefillGqaMinTokens;
  if (gqa_eligible) {
    unsigned int gqa_threads = static_cast<unsigned int>(spec->shape.head_dim);
    gqa_threads = (gqa_threads + 31u) & ~31u;
    if (gqa_threads == 0) {
      gqa_threads = 32u;
    } else if (gqa_threads > 256u) {
      gqa_threads = 256u;
    }
    const dim3 gqa_grid(static_cast<unsigned int>(spec->shape.kv_heads),
                        static_cast<unsigned int>(spec->tokens));
    attention_prefill_gqa_kernel<<<gqa_grid, gqa_threads, 0,
                                   qwen36_internal_active_stream()>>>(
        ptr<const __nv_bfloat16>(spec->q_bf16),
        ptr<const void>(spec->kv_cache_k), ptr<const void>(spec->kv_cache_v),
        ptr<const float>(spec->kv_cache_metadata), spec->kv_cache_dtype,
        ptr<__nv_bfloat16>(spec->output_bf16), spec->start_position,
        ptr<const int32_t>(spec->start_position_device_i32), spec->tokens,
        spec->shape);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
  }

  const dim3 attn_grid(static_cast<unsigned int>(spec->shape.q_heads),
                       static_cast<unsigned int>(spec->tokens));
  attention_prefill_kernel<<<attn_grid, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->q_bf16),
      ptr<const void>(spec->kv_cache_k), ptr<const void>(spec->kv_cache_v),
      ptr<const float>(spec->kv_cache_metadata), spec->kv_cache_dtype,
      ptr<__nv_bfloat16>(spec->output_bf16), spec->start_position,
      ptr<const int32_t>(spec->start_position_device_i32), spec->tokens,
      spec->shape);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int
qwen36_attention_decode(const qwen36_attention_decode_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0 ||
      spec->v_bf16.ptr == 0 || spec->kv_cache_k.ptr == 0 ||
      spec->kv_cache_v.ptr == 0 || spec->output_bf16.ptr == 0 ||
      spec->shape.q_heads == 0 || spec->shape.kv_heads == 0 ||
      spec->shape.head_dim == 0 || spec->shape.head_dim > 256 ||
      spec->shape.q_heads % spec->shape.kv_heads != 0 ||
      (spec->kv_cache_dtype != kKvCacheBf16 &&
       spec->kv_cache_dtype != kKvCacheFp8 &&
       spec->kv_cache_dtype != kKvCacheTurboQuant3 &&
       spec->kv_cache_dtype != kKvCacheTurboQuant35) ||
      (is_tq_cache_dtype(spec->kv_cache_dtype) &&
       (spec->kv_cache_metadata.ptr == 0 ||
        !tq_supported_head_dim(spec->shape.head_dim)))) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  // Split-KV / FlashDecoding path: when `decode_n_splits >= 2` the engine
  // has decided this attention call is on a long enough context to benefit
  // from T-axis parallelism. We launch a fixed grid of (q_heads × n_splits)
  // blocks and a small follow-up reduce — the value is derived from
  // `max_context`, not the current position, so the same launch shape is
  // valid for both fresh kernel calls AND graph replays where the position
  // grows after capture. Empty splits early-exit cheaply via the
  // `t_start >= position + 1` guard inside the kernel.
  const bool partials_present = spec->partial_acc_f32.ptr != 0 &&
                                spec->partial_max_f32.ptr != 0 &&
                                spec->partial_denom_f32.ptr != 0;
  if (partials_present && spec->decode_n_splits >= 2 &&
      spec->shape.head_dim <= 256) {
    const int split_timesteps_per_block =
        spec->split_timesteps_per_block == 0
            ? kDefaultSplitTimestepsPerBlock
            : static_cast<int>(spec->split_timesteps_per_block);
    if (split_timesteps_per_block < kMinSplitTimestepsPerBlock) {
      return QWEN36_STATUS_INVALID_ARGUMENT;
    }
    unsigned int split_threads = static_cast<unsigned int>(spec->shape.head_dim);
    split_threads = (split_threads + 31u) & ~31u;
    if (split_threads == 0) {
      split_threads = 32u;
    } else if (split_threads > 256u) {
      split_threads = 256u;
    }
    const int n_splits = static_cast<int>(spec->decode_n_splits);
    const size_t q_per_kv = spec->shape.q_heads / spec->shape.kv_heads;
    const bool gqa_split_eligible =
        spec->shape.head_dim <= static_cast<size_t>(kGqaMaxHeadDim) &&
        q_per_kv <= static_cast<size_t>(kGqaMaxQPerKv) && q_per_kv > 1 &&
        n_splits >= 32;
    if (gqa_split_eligible) {
      const dim3 split_grid(static_cast<unsigned int>(spec->shape.kv_heads),
                            static_cast<unsigned int>(n_splits));
      attention_decode_split_gqa_kernel<<<split_grid, split_threads, 0,
                                          qwen36_internal_active_stream()>>>(
          ptr<const __nv_bfloat16>(spec->q_bf16),
          ptr<const __nv_bfloat16>(spec->k_bf16),
          ptr<const __nv_bfloat16>(spec->v_bf16),
          ptr<void>(spec->kv_cache_k), ptr<void>(spec->kv_cache_v),
          ptr<float>(spec->kv_cache_metadata),
          spec->kv_cache_dtype,
          ptr<float>(spec->partial_acc_f32),
          ptr<float>(spec->partial_max_f32),
          ptr<float>(spec->partial_denom_f32), spec->position,
          ptr<const int32_t>(spec->position_device_i32), spec->shape, n_splits,
          split_timesteps_per_block);
    } else {
      const dim3 split_grid(static_cast<unsigned int>(spec->shape.q_heads),
                            static_cast<unsigned int>(n_splits));
      attention_decode_split_kernel<<<split_grid, split_threads, 0,
                                      qwen36_internal_active_stream()>>>(
          ptr<const __nv_bfloat16>(spec->q_bf16),
          ptr<const __nv_bfloat16>(spec->k_bf16),
          ptr<const __nv_bfloat16>(spec->v_bf16),
          ptr<void>(spec->kv_cache_k), ptr<void>(spec->kv_cache_v),
          ptr<float>(spec->kv_cache_metadata),
          spec->kv_cache_dtype,
          ptr<float>(spec->partial_acc_f32),
          ptr<float>(spec->partial_max_f32),
          ptr<float>(spec->partial_denom_f32), spec->position,
          ptr<const int32_t>(spec->position_device_i32), spec->shape, n_splits,
          split_timesteps_per_block);
    }
    attention_decode_reduce_kernel<<<
        static_cast<unsigned int>(spec->shape.q_heads), split_threads, 0,
        qwen36_internal_active_stream()>>>(
        ptr<const float>(spec->partial_acc_f32),
        ptr<const float>(spec->partial_max_f32),
        ptr<const float>(spec->partial_denom_f32),
        ptr<__nv_bfloat16>(spec->output_bf16), spec->shape, n_splits);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
  }

  // Round threads up to the next multiple of 32 so warp-shuffle reductions are
  // well-defined; cap at 256 (max head_dim) so the block fits within 8 warps,
  // matching the size of the warp_sums staging array inside the kernel.
  unsigned int threads = static_cast<unsigned int>(spec->shape.head_dim);
  threads = (threads + 31u) & ~31u;
  if (threads == 0) {
    threads = 32u;
  } else if (threads > 256u) {
    threads = 256u;
  }
  attention_decode_kernel<<<static_cast<unsigned int>(spec->shape.q_heads),
                            threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->q_bf16),
      ptr<const __nv_bfloat16>(spec->k_bf16),
      ptr<const __nv_bfloat16>(spec->v_bf16),
      ptr<void>(spec->kv_cache_k), ptr<void>(spec->kv_cache_v),
      ptr<float>(spec->kv_cache_metadata),
      spec->kv_cache_dtype, ptr<__nv_bfloat16>(spec->output_bf16), spec->position,
      ptr<const int32_t>(spec->position_device_i32), spec->shape);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
