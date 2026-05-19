// Per-block megakernel for the Qwen3.6 full-attn decode path (SM_120).
//
// Phase 2 of the megakernel roadmap. The goal is to fuse RMSNorm → Q/K/V →
// RoPE → attention → o_proj → residual → RMSNorm → MLP (gate+up → SwiGLU →
// down) → residual into a single persistent kernel launch, replacing the ~7
// launches the current host-driven path emits per full-attn layer. The 48
// DeltaNet layers stay on the existing path.
//
// **This is Stage A of the incremental build (the skeleton).**
// It implements only the persistent-grid + atomic-barrier infrastructure
// and an identity copy from `hidden_in` to `hidden_out` through a single
// barrier. Every later stage (B…F) plugs computation into the same
// scaffolding behind its own env-var gate (`QWEN36_MEGAKERNEL_FULL_ATTN_STAGE`).
//
// Barrier discipline: each phase consumes one slot in `barrier_state`. The
// caller must zero the entire `barrier_state` buffer before every launch
// (the kernel does not reset it on exit). One slot per phase × launch is
// enough; we use `gridDim.x` as the per-phase target count.

#include "qwen36_fp4.h"
#include "../decode_gemv/nvfp4_gemv_mma_helpers.cuh"
#include "../decode_gemv/nvfp4_gemv_mma_kernel.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS
#define QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS 256
#endif

#ifndef QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_CTAS
// Sized to cover the widest phase the megakernel will host (MLP gate+up:
// M = 2 * intermediate = 34816 with M-tile=128 → 272 CTAs). Stage A only
// needs one CTA per chunk of hidden, but we launch the full grid so the
// barrier and scheduling characteristics match the eventual fully-fused
// kernel.
#define QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_CTAS 272
#endif

namespace {

// NVFP4 codec helpers — duplicated from kernels-cuda/ops.cu so the
// megakernel can inline them without exporting the anonymous-namespace
// definitions. They are pure (small, stable) so the duplication is
// acceptable; parity smokes catch any future divergence.

__host__ __device__ inline size_t div_ceil_size(size_t value, size_t divisor) {
  return (value + divisor - 1) / divisor;
}

__host__ __device__ inline size_t round_up_size(size_t value, size_t multiple) {
  return div_ceil_size(value, multiple) * multiple;
}

__host__ __device__ inline size_t vec16_scale_offset(size_t inner, size_t outer,
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

// Atomic phase barrier — Alpindale "monotonic counter" pattern. Every CTA
// calls this exactly once per phase; on entry the kernel has reached the
// barrier with all threads visible (via the leading `__syncthreads`), then
// thread 0 increments the per-phase slot and spins until every CTA has
// done the same. Trailing `__syncthreads` republishes the post-barrier
// state to every warp in the CTA.
//
// `slot` must point to a 4-byte device memory location pre-initialised to
// zero before the kernel launch. `expected` is the total CTA count.
__device__ inline void phase_barrier(uint32_t *slot, uint32_t expected) {
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(slot, 1u);
    uint32_t cur;
    do {
      asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(cur) : "l"(slot));
    } while (cur < expected);
  }
  __syncthreads();
}

// Stage A kernel: cooperative identity copy `hidden_out = hidden_in` with
// one phase_barrier separating the read and the write. Every CTA processes
// a stride-equal share of the elements; the barrier ensures every CTA has
// completed its reads before any CTA starts its writes — overkill for an
// identity but the right shape for what comes in Stages B-F.
__global__ void __launch_bounds__(QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS)
qwen36_full_attn_block_stage_a_kernel(const __nv_bfloat16 *__restrict__ hidden_in,
                                      __nv_bfloat16 *__restrict__ hidden_out,
                                      uint32_t *__restrict__ barrier_state,
                                      uint32_t hidden_size) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;

  // Per-thread scratch keeps the load live so the barrier separates the
  // read and the write rather than letting the compiler fuse them away.
  for (uint32_t i = tid; i < hidden_size; i += stride) {
    __nv_bfloat16 v = hidden_in[i];
    asm volatile("" ::"h"(*reinterpret_cast<unsigned short *>(&v)));
    hidden_out[i] = v;
  }

  // Single phase barrier — Stage A only exercises one. Stages B-F add more
  // slots in the same `barrier_state` buffer.
  phase_barrier(&barrier_state[0], gridDim.x);
}

// Stage B.1 RMSNorm phase. Single CTA (CTA 0) cooperates on one row of
// length `hidden`; the rest of the 272-CTA grid idles at the barrier. Byte-
// exact match with `rmsnorm_kernel` (kernels-cuda/ops.cu:131) when invoked
// with the same (1 + weight) parameterization and no residual: same
// vectorized bfloat162 I/O, same reduction tree, same epsilon math, same
// reinterpret_cast layout. The dynamic SMEM tile (`scratch`) holds the
// per-thread sum-of-squares and is reduced inside the function.
__device__ inline void
qwen36_full_attn_block_rmsnorm_phase(const __nv_bfloat16 *__restrict__ input,
                                     const __nv_bfloat16 *__restrict__ weight,
                                     __nv_bfloat16 *__restrict__ output,
                                     uint32_t hidden, float eps) {
  extern __shared__ float scratch[];
  float local_sum = 0.0f;

  const uint32_t pairs = hidden >> 1;
  const __nv_bfloat162 *input2 =
      reinterpret_cast<const __nv_bfloat162 *>(input);

  for (uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
    const __nv_bfloat162 vp = input2[p];
    const float a = __low2float(vp);
    const float b = __high2float(vp);
    local_sum += a * a + b * b;
  }
  if ((hidden & 1u) != 0u && threadIdx.x == 0u) {
    const float value = __bfloat162float(input[hidden - 1u]);
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

  const float rms_scale =
      rsqrtf(scratch[0] / static_cast<float>(hidden) + eps);

  __nv_bfloat162 *output2 = reinterpret_cast<__nv_bfloat162 *>(output);
  const __nv_bfloat162 *weight2 =
      reinterpret_cast<const __nv_bfloat162 *>(weight);

  for (uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
    const __nv_bfloat162 vp = input2[p];
    const __nv_bfloat162 wp = weight2[p];
    const float a = __low2float(vp);
    const float b = __high2float(vp);
    const float wa = __low2float(wp);
    const float wb = __high2float(wp);
    // (1 + weight) parameterization — matches Qwen base layer norms.
    output2[p] = __floats2bfloat162_rn(a * rms_scale * (1.0f + wa),
                                        b * rms_scale * (1.0f + wb));
  }
  if ((hidden & 1u) != 0u && threadIdx.x == 0u) {
    const uint32_t d = hidden - 1u;
    const float value = __bfloat162float(input[d]);
    const float w = __bfloat162float(weight[d]);
    output[d] = __float2bfloat16(value * rms_scale * (1.0f + w));
  }
}

// Stage B.2 fused RMSNorm + NVFP4 quantize phase. Single CTA cooperates
// on one row of `hidden` length; output is FP4-packed bytes + e4m3 block
// scales matching the vec16_scale_offset tile layout (so the downstream
// NVFP4 GEMV reads the same memory the existing rmsnorm_nvfp4_quantize
// kernel would write). Designed to be byte-exact with that kernel for
// the no-residual / direct_weight=0 / even-hidden path.
__device__ inline void
qwen36_full_attn_block_rmsnorm_quantize_phase(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ output_bf16, uint8_t *__restrict__ output_fp4,
    uint8_t *__restrict__ output_scale, float *__restrict__ tensor_scale,
    uint32_t hidden, float eps, float input_tensor_scale) {
  extern __shared__ float scratch[];
  float local_sum = 0.0f;

  const uint32_t pairs = hidden >> 1;
  const __nv_bfloat162 *input2 =
      reinterpret_cast<const __nv_bfloat162 *>(input);

  for (uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
    const __nv_bfloat162 vp = input2[p];
    const float a = __low2float(vp);
    const float b = __high2float(vp);
    local_sum += a * a + b * b;
  }
  if ((hidden & 1u) != 0u && threadIdx.x == 0u) {
    const float value = __bfloat162float(input[hidden - 1u]);
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

  // Each thread owns a stride-equal subset of groups (16-element blocks).
  for (size_t group = threadIdx.x; group < groups; group += blockDim.x) {
    const size_t start = group * 16;
    const size_t group_end = (start + 16 <= hidden) ? 16 : (hidden - start);
    float amax = 0.0f;
    float weighted_values[16];

    if (group_end == 16) {
      const __nv_bfloat162 *input_pair =
          reinterpret_cast<const __nv_bfloat162 *>(input + start);
      const __nv_bfloat162 *weight_pair =
          reinterpret_cast<const __nv_bfloat162 *>(weight + start);
      __nv_bfloat162 *output_pair =
          output_bf16 != nullptr
              ? reinterpret_cast<__nv_bfloat162 *>(output_bf16 + start)
              : nullptr;
#pragma unroll
      for (size_t p = 0; p < 8; ++p) {
        const __nv_bfloat162 ip = input_pair[p];
        const __nv_bfloat162 wp = weight_pair[p];
        const float a = __low2float(ip);
        const float b = __high2float(ip);
        const float w0 = __low2float(wp);
        const float w1 = __high2float(wp);
        const float weighted0 = a * norm_scale * (1.0f + w0);
        const float weighted1 = b * norm_scale * (1.0f + w1);
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
        const float value = __bfloat162float(input[d]);
        const float weighted =
            value * norm_scale * (1.0f + __bfloat162float(weight[d]));
        weighted_values[offset] = weighted;
        if (output_bf16 != nullptr) {
          output_bf16[d] = __float2bfloat16(weighted);
        }
        amax = fmaxf(amax, fabsf(weighted));
      }
    }

    const float scale_value =
        amax > 0.0f ? fmaxf(amax / (6.0f * global_scale), 1.0e-8f) : 1.0f;
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
  }

  if (threadIdx.x == 0 && tensor_scale != nullptr) {
    *tensor_scale = global_scale;
  }
}

__global__ void __launch_bounds__(QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS)
qwen36_full_attn_block_stage_b2_kernel(
    const __nv_bfloat16 *__restrict__ hidden_in,
    const __nv_bfloat16 *__restrict__ input_norm_weight,
    __nv_bfloat16 *__restrict__ hidden_normed_out_bf16,
    uint8_t *__restrict__ output_fp4, uint8_t *__restrict__ output_scale_e4m3,
    float *__restrict__ output_tensor_scale,
    uint32_t *__restrict__ barrier_state, uint32_t hidden_size, float eps,
    float input_tensor_scale) {
  if (blockIdx.x == 0) {
    qwen36_full_attn_block_rmsnorm_quantize_phase(
        hidden_in, input_norm_weight, hidden_normed_out_bf16, output_fp4,
        output_scale_e4m3, output_tensor_scale, hidden_size, eps,
        input_tensor_scale);
  }
  phase_barrier(&barrier_state[0], gridDim.x);
}

// Stage B.3 kernel: fuses Stage B.2 (RMSNorm + NVFP4 quantize) with the Q
// projection NVFP4 GEMV in a single launch. Phase layout:
//   - Phase 0 [CTA 0 only]: RMSNorm + quantize hidden → quantized_fp4 +
//                            quantized_scale (vec16 tile layout).
//   - Barrier 0.
//   - Phase 1 [every CTA]: NVFP4 GEMV body — call into the shared
//                          __device__ template the standalone GEMV uses.
//                          Each CTA owns one m16 row tile; grid =
//                          ceil(q_features / 16).
//   - Barrier 1 (clean exit semantics).
//
// The grid must be sized to `ceil(q_features / 16)` CTAs at launch time;
// using `QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_CTAS` would under-provision Q
// proj (M=6144 → 384 CTAs > 272). The barrier slot count is gridDim.x at
// runtime, so the smaller-CTA Stage B.1/B.2 launches stay correct.
// Partial RoPE phase (Qwen3.6 split-half rotation). Each CTA owns one
// pair index p ∈ [0, rope_dims/2); threads parallelize over the
// (q_heads + kv_heads) heads. Math mirrors `partial_rope_kernel` in
// kernels-cuda/ops.cu:406 and uses the same `apply_rope_half_pair` math
// (split-then-rotate, not adjacent-pair).
__device__ inline void
apply_rope_half_pair_inline(__nv_bfloat16 *values, size_t offset,
                            size_t half_dim, size_t pair, float cosv,
                            float sinv) {
  const size_t first = offset + pair;
  const size_t second = offset + half_dim + pair;
  const float x0 = __bfloat162float(values[first]);
  const float x1 = __bfloat162float(values[second]);
  values[first] = __float2bfloat16(x0 * cosv - x1 * sinv);
  values[second] = __float2bfloat16(x1 * cosv + x0 * sinv);
}

// SwiGLU + NVFP4 quantize, processed by one warp per 16-element group.
// Mirrors `swiglu_nvfp4_quantize_kernel` in ops.cu exactly (same amax
// reduction, same e4m3 scale rounding, same e2m1 nibble packing, same
// vec16_scale_offset layout) but reorganised so a single warp owns
// one group via shuffle reductions instead of a 32-thread CTA via
// shared memory. Lane 0 publishes the e4m3 scale; lanes 0..7 emit
// the packed FP4 nibbles. Byte-exact equivalence to the standalone
// kernel is the property we test in the megakernel parity smoke.
__device__ inline void
qwen36_full_attn_block_swiglu_quantize_warp_one_group(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up, uint8_t *__restrict__ output_fp4,
    uint8_t *__restrict__ output_scale, uint32_t group_idx,
    uint32_t intermediate, uint32_t lane, float global_scale_in) {
  const uint32_t start = group_idx * 16u;
  if (start >= intermediate) {
    return;
  }
  const float global_scale = global_scale_in > 0.0f ? global_scale_in : 1.0f;
  const uint32_t scale_blocks = (intermediate + 15u) / 16u;
  const uint32_t scale_inner_dim = (scale_blocks + 3u) & ~3u;

  float y = 0.0f;
  if (lane < 16u) {
    const uint32_t idx = start + lane;
    if (idx < intermediate) {
      const float g = __bfloat162float(gate[idx]);
      const float u = __bfloat162float(up[idx]);
      const float silu = g / (1.0f + expf(-g));
      y = silu * u;
    }
  }

  float amax = fabsf(y);
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 16));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));

  float decoded_scale_at_lane0 = 0.0f;
  if (lane == 0u) {
    const float scale_value =
        amax > 0.0f ? fmaxf(amax / (6.0f * global_scale), 1.0e-8f) : 1.0f;
    const uint8_t scale_code = encode_e4m3_positive(scale_value);
    output_scale[vec16_scale_offset(group_idx, 0, scale_inner_dim)] =
        scale_code;
    decoded_scale_at_lane0 =
        fmaxf(decode_e4m3(scale_code) * global_scale, 1.0e-8f);
  }
  const float decoded_scale =
      __shfl_sync(0xffffffffu, decoded_scale_at_lane0, 0);

  if (lane < 8u) {
    const uint32_t col = start + lane * 2u;
    if (col < intermediate) {
      const float y_low = __shfl_sync(0xffffffffu, y, lane * 2u);
      uint8_t packed = encode_e2m1(y_low / decoded_scale);
      if (col + 1u < intermediate) {
        const float y_high = __shfl_sync(0xffffffffu, y, lane * 2u + 1u);
        packed |= static_cast<uint8_t>(
            encode_e2m1(y_high / decoded_scale) << 4);
      }
      output_fp4[col / 2u] = packed;
    }
  }
}

__device__ inline void qwen36_full_attn_block_partial_rope_phase(
    __nv_bfloat16 *__restrict__ q, __nv_bfloat16 *__restrict__ k,
    int32_t position, uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t rope_dims, float base_theta) {
  const uint32_t half_dim = rope_dims / 2;
  if (blockIdx.x >= half_dim) {
    return; // tail CTAs sit out RoPE; the barrier still gates everyone.
  }
  const uint32_t p = blockIdx.x;
  const float pos = static_cast<float>(position);
  const float inv_freq =
      powf(base_theta,
           -static_cast<float>(2 * p) / static_cast<float>(rope_dims));
  const float angle = pos * inv_freq;
  const float cosv = cosf(angle);
  const float sinv = sinf(angle);
  for (uint32_t head = threadIdx.x; head < q_heads + kv_heads;
       head += blockDim.x) {
    if (head < q_heads) {
      apply_rope_half_pair_inline(q, static_cast<size_t>(head) * head_dim,
                                  half_dim, p, cosv, sinv);
    } else {
      const uint32_t kv_head = head - q_heads;
      apply_rope_half_pair_inline(k, static_cast<size_t>(kv_head) * head_dim,
                                  half_dim, p, cosv, sinv);
    }
  }
}

// Pure NVFP4 quantize phase (no RMSNorm). Same per-group amax → e4m3
// scale → e2m1 pack math as the rmsnorm_quantize_phase, just without the
// rsqrt/(1+weight) preamble. Used in Stage E to quantize the attention
// output before the o_proj NVFP4 GEMV.
__device__ inline void qwen36_full_attn_block_quantize_phase(
    const __nv_bfloat16 *__restrict__ input,
    uint8_t *__restrict__ output_fp4, uint8_t *__restrict__ output_scale,
    float *__restrict__ tensor_scale, uint32_t hidden,
    float input_tensor_scale) {
  const size_t groups = div_ceil_size(hidden, 16);
  const size_t scale_inner_dim = round_up_size(groups, 4);
  const float global_scale =
      input_tensor_scale > 0.0f ? input_tensor_scale : 1.0f;

  for (size_t group = threadIdx.x; group < groups; group += blockDim.x) {
    const size_t start = group * 16;
    const size_t group_end = (start + 16 <= hidden) ? 16 : (hidden - start);
    float amax = 0.0f;
    float values[16];
    if (group_end == 16) {
      const __nv_bfloat162 *input_pair =
          reinterpret_cast<const __nv_bfloat162 *>(input + start);
#pragma unroll
      for (size_t p = 0; p < 8; ++p) {
        const __nv_bfloat162 ip = input_pair[p];
        const float a = __low2float(ip);
        const float b = __high2float(ip);
        values[p * 2] = a;
        values[p * 2 + 1] = b;
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
      }
    } else {
      for (size_t offset = 0; offset < group_end; ++offset) {
        const size_t d = start + offset;
        const float v = __bfloat162float(input[d]);
        values[offset] = v;
        amax = fmaxf(amax, fabsf(v));
      }
    }

    const float scale_value =
        amax > 0.0f ? fmaxf(amax / (6.0f * global_scale), 1.0e-8f) : 1.0f;
    const uint8_t scale_code = encode_e4m3_positive(scale_value);
    output_scale[vec16_scale_offset(group, 0, scale_inner_dim)] = scale_code;
    const float decoded_scale =
        fmaxf(decode_e4m3(scale_code) * global_scale, 1.0e-8f);

    for (size_t offset = 0; offset < 16 && start + offset < hidden;
         offset += 2) {
      const size_t d = start + offset;
      uint8_t packed = encode_e2m1(values[offset] / decoded_scale);
      if (d + 1 < hidden) {
        packed |= static_cast<uint8_t>(
            encode_e2m1(values[offset + 1] / decoded_scale) << 4);
      }
      output_fp4[d / 2] = packed;
    }
  }

  if (threadIdx.x == 0 && tensor_scale != nullptr) {
    *tensor_scale = global_scale;
  }
}

// Fused residual + RMSNorm + NVFP4 quantize phase. Math mirrors
// `rmsnorm_nvfp4_quantize_kernel` in kernels-cuda/ops.cu:231 exactly —
// folds (input + residual) before the sum-of-squares so we don't pay a
// BF16 round-trip on the residual sum between phases. Used by Stage E
// where the o_proj output and the pre-attention residual must be added
// before the post-attn norm.
__device__ inline void
qwen36_full_attn_block_residual_rmsnorm_quantize_phase(
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ output_bf16, uint8_t *__restrict__ output_fp4,
    uint8_t *__restrict__ output_scale, float *__restrict__ tensor_scale,
    uint32_t hidden, float eps, float input_tensor_scale) {
  extern __shared__ float scratch[];
  float local_sum = 0.0f;

  const uint32_t pairs = hidden >> 1;
  const __nv_bfloat162 *input2 =
      reinterpret_cast<const __nv_bfloat162 *>(input);
  const __nv_bfloat162 *residual2 =
      reinterpret_cast<const __nv_bfloat162 *>(residual);

  for (uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
    const __nv_bfloat162 ip = input2[p];
    const __nv_bfloat162 rp = residual2[p];
    const float a = __low2float(ip) + __low2float(rp);
    const float b = __high2float(ip) + __high2float(rp);
    local_sum += a * a + b * b;
  }
  if ((hidden & 1u) != 0u && threadIdx.x == 0u) {
    const float v = __bfloat162float(input[hidden - 1u]) +
                    __bfloat162float(residual[hidden - 1u]);
    local_sum += v * v;
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
    const size_t group_end = (start + 16 <= hidden) ? 16 : (hidden - start);
    float amax = 0.0f;
    float residual_values[16];
    float weighted_values[16];

    if (group_end == 16) {
      const __nv_bfloat162 *input_pair =
          reinterpret_cast<const __nv_bfloat162 *>(input + start);
      const __nv_bfloat162 *residual_pair =
          reinterpret_cast<const __nv_bfloat162 *>(residual + start);
      const __nv_bfloat162 *weight_pair =
          reinterpret_cast<const __nv_bfloat162 *>(weight + start);
      __nv_bfloat162 *output_pair =
          output_bf16 != nullptr
              ? reinterpret_cast<__nv_bfloat162 *>(output_bf16 + start)
              : nullptr;
#pragma unroll
      for (size_t p = 0; p < 8; ++p) {
        const __nv_bfloat162 ip = input_pair[p];
        const __nv_bfloat162 rp = residual_pair[p];
        const __nv_bfloat162 wp = weight_pair[p];
        const float a = __low2float(ip) + __low2float(rp);
        const float b = __high2float(ip) + __high2float(rp);
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
        const float v =
            __bfloat162float(input[d]) + __bfloat162float(residual[d]);
        const float w = __bfloat162float(weight[d]);
        const float weighted = v * norm_scale * (1.0f + w);
        residual_values[offset] = v;
        weighted_values[offset] = weighted;
        if (output_bf16 != nullptr) {
          output_bf16[d] = __float2bfloat16(weighted);
        }
        amax = fmaxf(amax, fabsf(weighted));
      }
    }

    const float scale_value =
        amax > 0.0f ? fmaxf(amax / (6.0f * global_scale), 1.0e-8f) : 1.0f;
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

// Stage E kernel: attention output → quantize → o_proj GEMV → fused
// (residual + post-attn RMSNorm + NVFP4 quantize). Phase layout:
//   0 [CTA 0]      Quantize attention output (BF16 → FP4 + scales).
//   barrier
//   1 [all CTAs]   o_proj GEMV  (M = hidden_size, K = q_features).
//   barrier
//   2 [CTA 0]      Fused residual + post-attn RMSNorm + NVFP4 quantize.
//                  Matches the reference fused kernel exactly (no BF16
//                  round-trip on the residual sum), so the post-norm
//                  outputs are byte-identical to qwen36_rmsnorm_nvfp4
//                  _quantize invoked with `residual_bf16 = residual_in`.
//   barrier (clean exit)
//
// Grid sized to the widest phase (o_proj). Use the same 384-CTA shape as
// Stage C so engine integration can share the grid across the layer.
__global__ void __launch_bounds__(QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS)
qwen36_full_attn_block_stage_e_kernel(
    const __nv_bfloat16 *__restrict__ attention_out,
    const __nv_bfloat16 *__restrict__ residual_in,
    const uint8_t *__restrict__ o_proj_fp4,
    const uint8_t *__restrict__ o_proj_scale, float o_alpha,
    const __nv_bfloat16 *__restrict__ post_norm_weight,
    uint8_t *__restrict__ attention_quantized_fp4,
    uint8_t *__restrict__ attention_quantized_scale,
    __nv_bfloat16 *__restrict__ o_proj_out,
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ post_normed_out,
    uint8_t *__restrict__ post_quantized_fp4,
    uint8_t *__restrict__ post_quantized_scale,
    uint32_t *__restrict__ barrier_state, uint32_t q_features,
    uint32_t hidden_size, float eps, float post_input_tensor_scale,
    float attention_output_tensor_scale) {
  // Phase 0 — Quantize attention output.
  if (blockIdx.x == 0) {
    qwen36_full_attn_block_quantize_phase(
        attention_out, attention_quantized_fp4, attention_quantized_scale,
        /*tensor_scale=*/nullptr, q_features, attention_output_tensor_scale);
  }
  phase_barrier(&barrier_state[0], gridDim.x);

  // Phase 1 — o_proj GEMV. M = hidden_size (5120), K = q_features (6144).
  qwen36_gemv::nvfp4_gemv_mma_body<8>(
      blockIdx.x, o_proj_fp4, o_proj_scale, attention_quantized_fp4,
      attention_quantized_scale, o_alpha, o_proj_out,
      static_cast<size_t>(hidden_size), static_cast<size_t>(q_features));
  phase_barrier(&barrier_state[1], gridDim.x);

  // Phase 2 — Fused residual + post-attn RMSNorm + NVFP4 quantize.
  if (blockIdx.x == 0) {
    qwen36_full_attn_block_residual_rmsnorm_quantize_phase(
        o_proj_out, residual_in, post_norm_weight, residual_out,
        post_normed_out, post_quantized_fp4, post_quantized_scale,
        /*tensor_scale=*/nullptr, hidden_size, eps,
        post_input_tensor_scale);
  }
  phase_barrier(&barrier_state[2], gridDim.x);
}

// Stage C kernel: Stage B.3 + K projection + V projection + partial RoPE
// on Q/K. Phase layout:
//   0  [CTA 0]            RMSNorm + NVFP4 quantize hidden
//   barrier
//   1  [all CTAs]         Q proj GEMV   (M = q_features)
//   barrier
//   2  [CTAs < kv/16]     K proj GEMV   (M = kv_features; tail CTAs idle
//                                          via body's row-bounds check)
//   barrier
//   3  [CTAs < kv/16]     V proj GEMV   (M = kv_features)
//   barrier
//   4  [CTAs < rope/2]    partial RoPE on Q + K (split-half pairs)
//   barrier (clean exit)
//
// Grid sized to the widest phase (Q proj, ceil(q_features/16)). The body's
// bounds checks make the smaller K/V phases correct with the same grid.
__global__ void __launch_bounds__(QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS)
qwen36_full_attn_block_stage_c_kernel(
    const __nv_bfloat16 *__restrict__ hidden_in,
    const __nv_bfloat16 *__restrict__ input_norm_weight,
    const uint8_t *__restrict__ q_weight_fp4,
    const uint8_t *__restrict__ q_weight_scale, float q_alpha,
    const uint8_t *__restrict__ k_weight_fp4,
    const uint8_t *__restrict__ k_weight_scale, float k_alpha,
    const uint8_t *__restrict__ v_weight_fp4,
    const uint8_t *__restrict__ v_weight_scale, float v_alpha,
    __nv_bfloat16 *__restrict__ hidden_normed_out_bf16,
    uint8_t *__restrict__ quantized_fp4,
    uint8_t *__restrict__ quantized_scale_e4m3,
    __nv_bfloat16 *__restrict__ q_out, __nv_bfloat16 *__restrict__ k_out,
    __nv_bfloat16 *__restrict__ v_out, uint32_t *__restrict__ barrier_state,
    uint32_t hidden_size, uint32_t q_features, uint32_t kv_features,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t rope_dims,
    int32_t position, float base_theta, float eps,
    float input_tensor_scale) {
  // Phase 0 — RMSNorm + quantize.
  if (blockIdx.x == 0) {
    qwen36_full_attn_block_rmsnorm_quantize_phase(
        hidden_in, input_norm_weight, hidden_normed_out_bf16, quantized_fp4,
        quantized_scale_e4m3, /*tensor_scale=*/nullptr, hidden_size, eps,
        input_tensor_scale);
  }
  phase_barrier(&barrier_state[0], gridDim.x);

  // Phase 1 — Q proj.
  qwen36_gemv::nvfp4_gemv_mma_body<8>(
      blockIdx.x, q_weight_fp4, q_weight_scale, quantized_fp4,
      quantized_scale_e4m3, q_alpha, q_out, static_cast<size_t>(q_features),
      static_cast<size_t>(hidden_size));
  phase_barrier(&barrier_state[1], gridDim.x);

  // Phase 2 — K proj. The GEMV body's bounds checks skip rows ≥ M, so all
  // CTAs run the cooperative B load and the MMA loop but only CTAs whose
  // m_base < kv_features write output. Wasted compute on tail CTAs is the
  // tradeoff for keeping the grid uniform across QKV phases.
  qwen36_gemv::nvfp4_gemv_mma_body<8>(
      blockIdx.x, k_weight_fp4, k_weight_scale, quantized_fp4,
      quantized_scale_e4m3, k_alpha, k_out, static_cast<size_t>(kv_features),
      static_cast<size_t>(hidden_size));
  phase_barrier(&barrier_state[2], gridDim.x);

  // Phase 3 — V proj (same shape as K).
  qwen36_gemv::nvfp4_gemv_mma_body<8>(
      blockIdx.x, v_weight_fp4, v_weight_scale, quantized_fp4,
      quantized_scale_e4m3, v_alpha, v_out, static_cast<size_t>(kv_features),
      static_cast<size_t>(hidden_size));
  phase_barrier(&barrier_state[3], gridDim.x);

  // Phase 4 — partial RoPE on Q + K in place.
  qwen36_full_attn_block_partial_rope_phase(q_out, k_out, position, q_heads,
                                             kv_heads, head_dim, rope_dims,
                                             base_theta);
  phase_barrier(&barrier_state[4], gridDim.x);
}

__global__ void __launch_bounds__(QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS)
qwen36_full_attn_block_stage_b3_kernel(
    const __nv_bfloat16 *__restrict__ hidden_in,
    const __nv_bfloat16 *__restrict__ input_norm_weight,
    const uint8_t *__restrict__ q_weight_fp4,
    const uint8_t *__restrict__ q_weight_scale, float q_alpha,
    __nv_bfloat16 *__restrict__ hidden_normed_out_bf16,
    uint8_t *__restrict__ quantized_fp4,
    uint8_t *__restrict__ quantized_scale_e4m3,
    __nv_bfloat16 *__restrict__ q_out, uint32_t *__restrict__ barrier_state,
    uint32_t hidden_size, uint32_t q_features, float eps,
    float input_tensor_scale) {
  // Phase 0 — RMSNorm + quantize on CTA 0.
  if (blockIdx.x == 0) {
    qwen36_full_attn_block_rmsnorm_quantize_phase(
        hidden_in, input_norm_weight, hidden_normed_out_bf16, quantized_fp4,
        quantized_scale_e4m3, /*tensor_scale=*/nullptr, hidden_size, eps,
        input_tensor_scale);
  }
  phase_barrier(&barrier_state[0], gridDim.x);

  // Phase 1 — Q projection NVFP4 GEMV. Every CTA participates; each owns
  // one m16 output tile (blockIdx.x ∈ [0, q_features/16)).
  qwen36_gemv::nvfp4_gemv_mma_body<8>(
      blockIdx.x, q_weight_fp4, q_weight_scale, quantized_fp4,
      quantized_scale_e4m3, q_alpha, q_out, static_cast<size_t>(q_features),
      static_cast<size_t>(hidden_size));

  phase_barrier(&barrier_state[1], gridDim.x);
}

// Stage F.1 kernel: MLP gate+up NVFP4 GEMV (persistent grid).
//
// The MLP m-dimension is 2*intermediate = 34816 → 2176 m-tiles, which
// exceeds the grid size that can be concurrently resident on the 5090
// (~340 CTAs given the 17 KB smem footprint of the GEMV body). A
// gridDim-sized atomic barrier across 2176 CTAs would deadlock —
// late-batch CTAs cannot make progress while the first batch spins.
// Stage F therefore uses the persistent + work-stealing pattern:
// launch ≤ concurrent CTA capacity, each CTA loops `atomicAdd(work)`
// to grab the next m-tile until the counter exceeds the tile count.
//
// `work_counter` must be pre-zeroed by the caller (same discipline as
// the barrier slots). No phase barrier is needed because F.1 is the
// only phase right now — kernel completion is the synchronization
// point. F.2/F.3/F.4 will append later phases and re-introduce
// barriers (safe here because the persistent grid size matches
// concurrent capacity).
__global__ void __launch_bounds__(QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS)
qwen36_full_attn_block_stage_f1_kernel(
    const uint8_t *__restrict__ hidden_quantized_fp4,
    const uint8_t *__restrict__ hidden_quantized_scale,
    const uint8_t *__restrict__ mlp_gate_up_fp4,
    const uint8_t *__restrict__ mlp_gate_up_scale, float gate_up_alpha,
    __nv_bfloat16 *__restrict__ gate_up_out,
    uint32_t *__restrict__ work_counter, uint32_t total_m_tiles,
    uint32_t two_intermediate, uint32_t hidden_size) {
  __shared__ uint32_t my_tile_shared;
  for (;;) {
    if (threadIdx.x == 0) {
      my_tile_shared = atomicAdd(work_counter, 1u);
    }
    __syncthreads();
    const uint32_t m_tile_idx = my_tile_shared;
    if (m_tile_idx >= total_m_tiles) {
      break;
    }
    qwen36_gemv::nvfp4_gemv_mma_body<8>(
        m_tile_idx, mlp_gate_up_fp4, mlp_gate_up_scale,
        hidden_quantized_fp4, hidden_quantized_scale, gate_up_alpha,
        gate_up_out, static_cast<size_t>(two_intermediate),
        static_cast<size_t>(hidden_size));
    __syncthreads(); // republish smem before next iteration overwrites it
  }
}

// Stage F.2 kernel: Stage F.1 + SwiGLU + NVFP4 quantize. Phase layout
// on the same persistent grid (256 CTAs):
//   0 [persistent, m-tile work-steal]    gate+up GEMV.
//   barrier
//   1 [persistent, group work-steal]      SwiGLU + NVFP4 quantize.
//                                          1 warp per 16-element group;
//                                          each CTA claims 8 groups
//                                          per atomicAdd batch.
//   barrier (clean exit)
//
// `barrier_state` must hold ≥ 4 u32 slots zeroed by the caller:
//   slot 0 — gate+up m-tile work counter
//   slot 1 — phase 0/1 spinlock
//   slot 2 — swiglu group work counter
//   slot 3 — phase 1/exit spinlock
//
// Co-residency caveat: the inter-phase spinlock requires every CTA in
// the grid to be concurrently scheduled. With a dedicated GPU and a
// 256-CTA grid the 5090 satisfies this (5 CTAs/SM × 170 SMs = 850 ≫
// 256). Under heavy contention from other processes (e.g. our dev
// box's training jobs holding 30+ GB and 100% util) the time-slice
// scheduler can defer some CTAs, causing the spinlock to wait
// indefinitely. The decode hot path runs on a dedicated GPU so this
// is not a deployment concern; for shared-GPU validation, either
// quiesce the contending processes or switch to
// `cudaLaunchCooperativeKernel` (a TODO for later if needed).
__global__ void __launch_bounds__(QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS)
qwen36_full_attn_block_stage_f2_kernel(
    const uint8_t *__restrict__ hidden_quantized_fp4,
    const uint8_t *__restrict__ hidden_quantized_scale,
    const uint8_t *__restrict__ mlp_gate_up_fp4,
    const uint8_t *__restrict__ mlp_gate_up_scale, float gate_up_alpha,
    __nv_bfloat16 *__restrict__ gate_up_out,
    uint8_t *__restrict__ swiglu_fp4, uint8_t *__restrict__ swiglu_scale,
    uint32_t *__restrict__ barrier_state, uint32_t total_m_tiles,
    uint32_t total_groups, uint32_t two_intermediate, uint32_t intermediate,
    uint32_t hidden_size, float down_input_tensor_scale) {
  __shared__ uint32_t shared_work;

  // Phase 0 — gate+up GEMV via work-stealing m-tile counter.
  for (;;) {
    if (threadIdx.x == 0) {
      shared_work = atomicAdd(&barrier_state[0], 1u);
    }
    __syncthreads();
    const uint32_t m_tile_idx = shared_work;
    if (m_tile_idx >= total_m_tiles) {
      break;
    }
    qwen36_gemv::nvfp4_gemv_mma_body<8>(
        m_tile_idx, mlp_gate_up_fp4, mlp_gate_up_scale,
        hidden_quantized_fp4, hidden_quantized_scale, gate_up_alpha,
        gate_up_out, static_cast<size_t>(two_intermediate),
        static_cast<size_t>(hidden_size));
    __syncthreads();
  }
  phase_barrier(&barrier_state[1], gridDim.x);

  // Phase 1 — SwiGLU + NVFP4 quantize. Read gate from gate_up_out[0..),
  // up from gate_up_out[intermediate..). 1 warp per group; CTA claims
  // 8 groups per atomicAdd. Tail warps with group_idx ≥ total_groups
  // exit inside the warp helper (early return on `start >= intermediate`).
  const __nv_bfloat16 *gate_ptr = gate_up_out;
  const __nv_bfloat16 *up_ptr = gate_up_out + intermediate;
  const uint32_t warp_id = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31u;
  for (;;) {
    if (threadIdx.x == 0) {
      shared_work = atomicAdd(&barrier_state[2], 8u);
    }
    __syncthreads();
    const uint32_t cta_group_base = shared_work;
    if (cta_group_base >= total_groups) {
      break;
    }
    const uint32_t group_idx = cta_group_base + warp_id;
    qwen36_full_attn_block_swiglu_quantize_warp_one_group(
        gate_ptr, up_ptr, swiglu_fp4, swiglu_scale, group_idx,
        intermediate, lane, down_input_tensor_scale);
    __syncthreads();
  }
  phase_barrier(&barrier_state[3], gridDim.x);
}

// Stage F.4 kernel: complete MLP block fused into one persistent launch.
// Phase layout (same persistent 256-CTA grid; same co-residency caveat
// as F.2 — fine on dedicated GPU, fragile under heavy contention):
//   0 [persistent, m-tile work-steal]    gate+up GEMV (M=2*intermediate)
//   barrier (slot 1)
//   1 [persistent, group work-steal]      SwiGLU + NVFP4 quantize (16-el
//                                          groups; 1 warp/group)
//   barrier (slot 3)
//   2 [persistent, m-tile work-steal]    down GEMV (M=hidden_size,
//                                          K=intermediate)
//   barrier (slot 5)
//   3 [persistent, element work-steal]   residual add in-place
//                                          (hidden_size BF16 adds)
//   barrier (slot 7, clean exit)
//
// `barrier_state` must hold ≥ 8 zeroed u32 slots (4 work counters
// interleaved with 4 spinlocks). `down_alpha` is the pre-folded
// per-tensor product for the down GEMV.
__global__ void __launch_bounds__(QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS)
qwen36_full_attn_block_stage_f4_kernel(
    const uint8_t *__restrict__ hidden_quantized_fp4,
    const uint8_t *__restrict__ hidden_quantized_scale,
    const uint8_t *__restrict__ mlp_gate_up_fp4,
    const uint8_t *__restrict__ mlp_gate_up_scale, float gate_up_alpha,
    const uint8_t *__restrict__ mlp_down_fp4,
    const uint8_t *__restrict__ mlp_down_scale, float down_alpha,
    __nv_bfloat16 *__restrict__ gate_up_out,
    uint8_t *__restrict__ swiglu_fp4, uint8_t *__restrict__ swiglu_scale,
    __nv_bfloat16 *__restrict__ down_out,
    __nv_bfloat16 *__restrict__ residual, uint32_t *__restrict__ barrier_state,
    uint32_t gate_up_m_tiles, uint32_t swiglu_groups, uint32_t down_m_tiles,
    uint32_t hidden_size, uint32_t intermediate, uint32_t two_intermediate,
    float down_input_tensor_scale) {
  __shared__ uint32_t shared_work;

  // Phase 0 — gate+up GEMV.
  for (;;) {
    if (threadIdx.x == 0) {
      shared_work = atomicAdd(&barrier_state[0], 1u);
    }
    __syncthreads();
    const uint32_t m_tile_idx = shared_work;
    if (m_tile_idx >= gate_up_m_tiles) {
      break;
    }
    qwen36_gemv::nvfp4_gemv_mma_body<8>(
        m_tile_idx, mlp_gate_up_fp4, mlp_gate_up_scale,
        hidden_quantized_fp4, hidden_quantized_scale, gate_up_alpha,
        gate_up_out, static_cast<size_t>(two_intermediate),
        static_cast<size_t>(hidden_size));
    __syncthreads();
  }
  phase_barrier(&barrier_state[1], gridDim.x);

  // Phase 1 — SwiGLU + NVFP4 quantize.
  {
    const __nv_bfloat16 *gate_ptr = gate_up_out;
    const __nv_bfloat16 *up_ptr = gate_up_out + intermediate;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    for (;;) {
      if (threadIdx.x == 0) {
        shared_work = atomicAdd(&barrier_state[2], 8u);
      }
      __syncthreads();
      const uint32_t cta_group_base = shared_work;
      if (cta_group_base >= swiglu_groups) {
        break;
      }
      const uint32_t group_idx = cta_group_base + warp_id;
      qwen36_full_attn_block_swiglu_quantize_warp_one_group(
          gate_ptr, up_ptr, swiglu_fp4, swiglu_scale, group_idx,
          intermediate, lane, down_input_tensor_scale);
      __syncthreads();
    }
  }
  phase_barrier(&barrier_state[3], gridDim.x);

  // Phase 2 — down GEMV.
  for (;;) {
    if (threadIdx.x == 0) {
      shared_work = atomicAdd(&barrier_state[4], 1u);
    }
    __syncthreads();
    const uint32_t m_tile_idx = shared_work;
    if (m_tile_idx >= down_m_tiles) {
      break;
    }
    qwen36_gemv::nvfp4_gemv_mma_body<8>(
        m_tile_idx, mlp_down_fp4, mlp_down_scale, swiglu_fp4, swiglu_scale,
        down_alpha, down_out, static_cast<size_t>(hidden_size),
        static_cast<size_t>(intermediate));
    __syncthreads();
  }
  phase_barrier(&barrier_state[5], gridDim.x);

  // Phase 3 — residual add in place. Element work-steal so each CTA
  // grabs a stride; the BF16 round-trip matches the standalone
  // residual_add path (which is just a BF16-precision add).
  {
    const uint32_t total_elems = hidden_size;
    const uint32_t per_cta = QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS;
    for (;;) {
      if (threadIdx.x == 0) {
        shared_work = atomicAdd(&barrier_state[6], per_cta);
      }
      __syncthreads();
      const uint32_t base = shared_work;
      if (base >= total_elems) {
        break;
      }
      const uint32_t idx = base + threadIdx.x;
      if (idx < total_elems) {
        const float r = __bfloat162float(residual[idx]);
        const float d = __bfloat162float(down_out[idx]);
        residual[idx] = __float2bfloat16(r + d);
      }
      __syncthreads();
    }
  }
  phase_barrier(&barrier_state[7], gridDim.x);
}

__global__ void __launch_bounds__(QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS)
qwen36_full_attn_block_stage_b1_kernel(
    const __nv_bfloat16 *__restrict__ hidden_in,
    const __nv_bfloat16 *__restrict__ input_norm_weight,
    __nv_bfloat16 *__restrict__ hidden_normed_out,
    uint32_t *__restrict__ barrier_state, uint32_t hidden_size, float eps) {
  // Stage B.1 only fuses RMSNorm. Decode is N=1 → one row of work → one
  // CTA active; the rest of the persistent grid waits at the barrier. The
  // unused SMs are intentional — later stages plug in computation that
  // covers them (Q proj ~24 CTAs, MLP ~272 CTAs).
  if (blockIdx.x == 0) {
    qwen36_full_attn_block_rmsnorm_phase(hidden_in, input_norm_weight,
                                          hidden_normed_out, hidden_size, eps);
  }
  phase_barrier(&barrier_state[0], gridDim.x);
}

} // namespace

extern "C" int qwen36_full_attn_block_stage_a(qwen36_device_ptr_t hidden_in,
                                              qwen36_device_ptr_t hidden_out,
                                              qwen36_device_ptr_t barrier_state,
                                              size_t hidden_size) {
  if (hidden_in.ptr == 0 || hidden_out.ptr == 0 || barrier_state.ptr == 0 ||
      hidden_size == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (hidden_size > 0xFFFFFFFFu) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  // The caller (engine) is responsible for zero-initialising `barrier_state`
  // before each launch (cudaMemsetAsync on the active stream). We don't do
  // it here so the kernel stays graph-captureable without an embedded
  // memset that depends on the call ordering.
  cudaStream_t stream = nullptr;
  // qwen36_internal_active_stream is declared in active_stream.h but we
  // pull it via the same extern pattern used by other kernels here, to
  // avoid a header dependency loop with active_stream.h.
  extern cudaStream_t qwen36_internal_active_stream();
  stream = qwen36_internal_active_stream();

  qwen36_full_attn_block_stage_a_kernel<<<
      QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_CTAS,
      QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(hidden_in.ptr)),
      reinterpret_cast<__nv_bfloat16 *>(static_cast<uintptr_t>(hidden_out.ptr)),
      reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(barrier_state.ptr)),
      static_cast<uint32_t>(hidden_size));

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_full_attn_block_stage_b_rmsnorm(
    qwen36_device_ptr_t hidden_in, qwen36_device_ptr_t input_norm_weight,
    qwen36_device_ptr_t hidden_normed_out, qwen36_device_ptr_t barrier_state,
    size_t hidden_size, float eps) {
  if (hidden_in.ptr == 0 || input_norm_weight.ptr == 0 ||
      hidden_normed_out.ptr == 0 || barrier_state.ptr == 0 ||
      hidden_size == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (hidden_size > 0xFFFFFFFFu) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  extern cudaStream_t qwen36_internal_active_stream();
  cudaStream_t stream = qwen36_internal_active_stream();

  const size_t smem_bytes =
      QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS * sizeof(float);

  qwen36_full_attn_block_stage_b1_kernel<<<
      QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_CTAS,
      QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS, smem_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(hidden_in.ptr)),
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(input_norm_weight.ptr)),
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(hidden_normed_out.ptr)),
      reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(barrier_state.ptr)),
      static_cast<uint32_t>(hidden_size), eps);

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_full_attn_block_stage_e_o_proj_residual_norm(
    qwen36_device_ptr_t attention_out, qwen36_device_ptr_t residual_in,
    qwen36_device_ptr_t o_proj_fp4, qwen36_device_ptr_t o_proj_scale,
    float o_alpha, qwen36_device_ptr_t post_norm_weight,
    qwen36_device_ptr_t attention_quantized_fp4,
    qwen36_device_ptr_t attention_quantized_scale,
    qwen36_device_ptr_t o_proj_out, qwen36_device_ptr_t residual_out,
    qwen36_device_ptr_t post_normed_out,
    qwen36_device_ptr_t post_quantized_fp4,
    qwen36_device_ptr_t post_quantized_scale,
    qwen36_device_ptr_t barrier_state, size_t q_features, size_t hidden_size,
    float eps, float post_input_tensor_scale,
    float attention_output_tensor_scale) {
  if (attention_out.ptr == 0 || residual_in.ptr == 0 ||
      o_proj_fp4.ptr == 0 || o_proj_scale.ptr == 0 ||
      post_norm_weight.ptr == 0 || attention_quantized_fp4.ptr == 0 ||
      attention_quantized_scale.ptr == 0 || o_proj_out.ptr == 0 ||
      residual_out.ptr == 0 || post_quantized_fp4.ptr == 0 ||
      post_quantized_scale.ptr == 0 || barrier_state.ptr == 0 ||
      q_features == 0 || hidden_size == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (q_features > 0xFFFFFFFFu || hidden_size > 0xFFFFFFFFu) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  // o_proj GEMV: M = hidden_size aligned to 16, K = q_features aligned to 512.
  if ((hidden_size & 15u) != 0u || (q_features & 511u) != 0u) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  extern cudaStream_t qwen36_internal_active_stream();
  cudaStream_t stream = qwen36_internal_active_stream();

  // Grid: max of o_proj m-tiles (hidden_size/16) and the Stage C shape
  // (q_features/16) so engine integration can use the same grid for both
  // megakernel launches in a full-attn layer.
  const unsigned o_proj_tiles =
      static_cast<unsigned>((hidden_size + 15u) / 16u);
  const unsigned q_proj_tiles =
      static_cast<unsigned>((q_features + 15u) / 16u);
  const unsigned grid_ctas =
      o_proj_tiles > q_proj_tiles ? o_proj_tiles : q_proj_tiles;

  constexpr unsigned kWarps = 8;
  const size_t k_over_2 = q_features / 2; // K of the o_proj GEMV
  const size_t a_tile_bytes =
      static_cast<size_t>(kWarps) * 2u * qwen36_gemv::kATilePerWarpBytes;
  const size_t reduction_bytes =
      2u * static_cast<size_t>(qwen36_gemv::kRowsPerBlock) *
      static_cast<size_t>(kWarps) * sizeof(float);
  const size_t sf_scale_cols = q_features / 16;
  const size_t sf_staging_bytes =
      static_cast<size_t>(qwen36_gemv::kRowsPerBlock) * sf_scale_cols +
      sf_scale_cols;
  const size_t gemv_smem =
      k_over_2 + a_tile_bytes + reduction_bytes + sf_staging_bytes;
  const size_t rmsnorm_smem =
      QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS * sizeof(float);
  const size_t smem_bytes =
      gemv_smem > rmsnorm_smem ? gemv_smem : rmsnorm_smem;

  qwen36_full_attn_block_stage_e_kernel<<<
      grid_ctas, QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS, smem_bytes,
      stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(attention_out.ptr)),
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(residual_in.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(o_proj_fp4.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(o_proj_scale.ptr)),
      o_alpha,
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(post_norm_weight.ptr)),
      reinterpret_cast<uint8_t *>(
          static_cast<uintptr_t>(attention_quantized_fp4.ptr)),
      reinterpret_cast<uint8_t *>(
          static_cast<uintptr_t>(attention_quantized_scale.ptr)),
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(o_proj_out.ptr)),
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(residual_out.ptr)),
      post_normed_out.ptr != 0
          ? reinterpret_cast<__nv_bfloat16 *>(
                static_cast<uintptr_t>(post_normed_out.ptr))
          : nullptr,
      reinterpret_cast<uint8_t *>(
          static_cast<uintptr_t>(post_quantized_fp4.ptr)),
      reinterpret_cast<uint8_t *>(
          static_cast<uintptr_t>(post_quantized_scale.ptr)),
      reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(barrier_state.ptr)),
      static_cast<uint32_t>(q_features),
      static_cast<uint32_t>(hidden_size), eps, post_input_tensor_scale,
      attention_output_tensor_scale);

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_full_attn_block_stage_c_qkv_rope(
    qwen36_device_ptr_t hidden_in, qwen36_device_ptr_t input_norm_weight,
    qwen36_device_ptr_t q_weight_fp4, qwen36_device_ptr_t q_weight_scale,
    float q_alpha, qwen36_device_ptr_t k_weight_fp4,
    qwen36_device_ptr_t k_weight_scale, float k_alpha,
    qwen36_device_ptr_t v_weight_fp4, qwen36_device_ptr_t v_weight_scale,
    float v_alpha, qwen36_device_ptr_t hidden_normed_out_bf16,
    qwen36_device_ptr_t quantized_fp4,
    qwen36_device_ptr_t quantized_scale_e4m3, qwen36_device_ptr_t q_out,
    qwen36_device_ptr_t k_out, qwen36_device_ptr_t v_out,
    qwen36_device_ptr_t barrier_state, size_t hidden_size, size_t q_features,
    size_t kv_features, size_t q_heads, size_t kv_heads, size_t head_dim,
    size_t rope_dims, int32_t position, float base_theta, float eps,
    float input_tensor_scale) {
  if (hidden_in.ptr == 0 || input_norm_weight.ptr == 0 ||
      q_weight_fp4.ptr == 0 || q_weight_scale.ptr == 0 ||
      k_weight_fp4.ptr == 0 || k_weight_scale.ptr == 0 ||
      v_weight_fp4.ptr == 0 || v_weight_scale.ptr == 0 ||
      quantized_fp4.ptr == 0 || quantized_scale_e4m3.ptr == 0 ||
      q_out.ptr == 0 || k_out.ptr == 0 || v_out.ptr == 0 ||
      barrier_state.ptr == 0 || hidden_size == 0 || q_features == 0 ||
      kv_features == 0 || head_dim == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (hidden_size > 0xFFFFFFFFu || q_features > 0xFFFFFFFFu ||
      kv_features > 0xFFFFFFFFu) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if ((q_features & 15u) != 0u || (kv_features & 15u) != 0u ||
      (hidden_size & 511u) != 0u) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  extern cudaStream_t qwen36_internal_active_stream();
  cudaStream_t stream = qwen36_internal_active_stream();

  const unsigned grid_ctas =
      static_cast<unsigned>((q_features + 15u) / 16u);
  constexpr unsigned kWarps = 8;
  const size_t k_over_2 = hidden_size / 2;
  const size_t a_tile_bytes =
      static_cast<size_t>(kWarps) * 2u * qwen36_gemv::kATilePerWarpBytes;
  const size_t reduction_bytes =
      2u * static_cast<size_t>(qwen36_gemv::kRowsPerBlock) *
      static_cast<size_t>(kWarps) * sizeof(float);
  const size_t sf_scale_cols = hidden_size / 16;
  const size_t sf_staging_bytes =
      static_cast<size_t>(qwen36_gemv::kRowsPerBlock) * sf_scale_cols +
      sf_scale_cols;
  const size_t gemv_smem =
      k_over_2 + a_tile_bytes + reduction_bytes + sf_staging_bytes;
  const size_t rmsnorm_smem =
      QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS * sizeof(float);
  const size_t smem_bytes =
      gemv_smem > rmsnorm_smem ? gemv_smem : rmsnorm_smem;

  qwen36_full_attn_block_stage_c_kernel<<<
      grid_ctas, QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS, smem_bytes,
      stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(hidden_in.ptr)),
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(input_norm_weight.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(q_weight_fp4.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(q_weight_scale.ptr)),
      q_alpha,
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(k_weight_fp4.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(k_weight_scale.ptr)),
      k_alpha,
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(v_weight_fp4.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(v_weight_scale.ptr)),
      v_alpha,
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(hidden_normed_out_bf16.ptr)),
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(quantized_fp4.ptr)),
      reinterpret_cast<uint8_t *>(
          static_cast<uintptr_t>(quantized_scale_e4m3.ptr)),
      reinterpret_cast<__nv_bfloat16 *>(static_cast<uintptr_t>(q_out.ptr)),
      reinterpret_cast<__nv_bfloat16 *>(static_cast<uintptr_t>(k_out.ptr)),
      reinterpret_cast<__nv_bfloat16 *>(static_cast<uintptr_t>(v_out.ptr)),
      reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(barrier_state.ptr)),
      static_cast<uint32_t>(hidden_size),
      static_cast<uint32_t>(q_features),
      static_cast<uint32_t>(kv_features),
      static_cast<uint32_t>(q_heads),
      static_cast<uint32_t>(kv_heads),
      static_cast<uint32_t>(head_dim),
      static_cast<uint32_t>(rope_dims), position, base_theta, eps,
      input_tensor_scale);

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_full_attn_block_stage_b_q_proj(
    qwen36_device_ptr_t hidden_in, qwen36_device_ptr_t input_norm_weight,
    qwen36_device_ptr_t q_weight_fp4, qwen36_device_ptr_t q_weight_scale,
    float q_alpha, qwen36_device_ptr_t hidden_normed_out_bf16,
    qwen36_device_ptr_t quantized_fp4,
    qwen36_device_ptr_t quantized_scale_e4m3, qwen36_device_ptr_t q_out,
    qwen36_device_ptr_t barrier_state, size_t hidden_size, size_t q_features,
    float eps, float input_tensor_scale) {
  if (hidden_in.ptr == 0 || input_norm_weight.ptr == 0 ||
      q_weight_fp4.ptr == 0 || q_weight_scale.ptr == 0 ||
      quantized_fp4.ptr == 0 || quantized_scale_e4m3.ptr == 0 ||
      q_out.ptr == 0 || barrier_state.ptr == 0 || hidden_size == 0 ||
      q_features == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (hidden_size > 0xFFFFFFFFu || q_features > 0xFFFFFFFFu) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  // Stage B.3 grid must cover Q projection. The GEMV body needs M aligned
  // to 16 rows and K aligned to (8 warps * 64) = 512.
  if ((q_features & 15u) != 0u || (hidden_size & 511u) != 0u) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  extern cudaStream_t qwen36_internal_active_stream();
  cudaStream_t stream = qwen36_internal_active_stream();

  const unsigned grid_ctas =
      static_cast<unsigned>((q_features + 15u) / 16u);
  // SMEM = max of RMSNorm scratch (256 floats = 1024 B) and the GEMV tile
  // footprint at K=hidden_size, 8 warps. The two phases share the same
  // dynamic SMEM block and never overlap in time (barrier between them).
  // GEMV layout matches the standalone dispatcher exactly.
  constexpr unsigned kWarps = 8;
  const size_t k_over_2 = hidden_size / 2;
  const size_t a_tile_bytes =
      static_cast<size_t>(kWarps) * 2u * qwen36_gemv::kATilePerWarpBytes;
  const size_t reduction_bytes =
      2u * static_cast<size_t>(qwen36_gemv::kRowsPerBlock) *
      static_cast<size_t>(kWarps) * sizeof(float);
  const size_t sf_scale_cols = hidden_size / 16;
  const size_t sf_staging_bytes =
      static_cast<size_t>(qwen36_gemv::kRowsPerBlock) * sf_scale_cols +
      sf_scale_cols;
  const size_t gemv_smem =
      k_over_2 + a_tile_bytes + reduction_bytes + sf_staging_bytes;
  const size_t rmsnorm_smem =
      QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS * sizeof(float);
  const size_t smem_bytes =
      gemv_smem > rmsnorm_smem ? gemv_smem : rmsnorm_smem;

  qwen36_full_attn_block_stage_b3_kernel<<<
      grid_ctas, QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS, smem_bytes,
      stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(hidden_in.ptr)),
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(input_norm_weight.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(q_weight_fp4.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(q_weight_scale.ptr)),
      q_alpha,
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(hidden_normed_out_bf16.ptr)),
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(quantized_fp4.ptr)),
      reinterpret_cast<uint8_t *>(
          static_cast<uintptr_t>(quantized_scale_e4m3.ptr)),
      reinterpret_cast<__nv_bfloat16 *>(static_cast<uintptr_t>(q_out.ptr)),
      reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(barrier_state.ptr)),
      static_cast<uint32_t>(hidden_size),
      static_cast<uint32_t>(q_features), eps, input_tensor_scale);

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_full_attn_block_stage_b_rmsnorm_quantize(
    qwen36_device_ptr_t hidden_in, qwen36_device_ptr_t input_norm_weight,
    qwen36_device_ptr_t hidden_normed_out_bf16,
    qwen36_device_ptr_t output_fp4, qwen36_device_ptr_t output_scale_e4m3,
    qwen36_device_ptr_t output_tensor_scale_f32,
    qwen36_device_ptr_t barrier_state, size_t hidden_size, float eps,
    float input_tensor_scale) {
  if (hidden_in.ptr == 0 || input_norm_weight.ptr == 0 ||
      output_fp4.ptr == 0 || output_scale_e4m3.ptr == 0 ||
      barrier_state.ptr == 0 || hidden_size == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (hidden_size > 0xFFFFFFFFu) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  extern cudaStream_t qwen36_internal_active_stream();
  cudaStream_t stream = qwen36_internal_active_stream();
  const size_t smem_bytes =
      QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS * sizeof(float);

  qwen36_full_attn_block_stage_b2_kernel<<<
      QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_CTAS,
      QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS, smem_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(hidden_in.ptr)),
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(input_norm_weight.ptr)),
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(hidden_normed_out_bf16.ptr)),
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(output_fp4.ptr)),
      reinterpret_cast<uint8_t *>(
          static_cast<uintptr_t>(output_scale_e4m3.ptr)),
      output_tensor_scale_f32.ptr != 0
          ? reinterpret_cast<float *>(
                static_cast<uintptr_t>(output_tensor_scale_f32.ptr))
          : nullptr,
      reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(barrier_state.ptr)),
      static_cast<uint32_t>(hidden_size), eps, input_tensor_scale);

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_full_attn_block_stage_f1_gate_up(
    qwen36_device_ptr_t hidden_quantized_fp4,
    qwen36_device_ptr_t hidden_quantized_scale,
    qwen36_device_ptr_t mlp_gate_up_fp4,
    qwen36_device_ptr_t mlp_gate_up_scale, float gate_up_alpha,
    qwen36_device_ptr_t gate_up_out, qwen36_device_ptr_t barrier_state,
    size_t intermediate, size_t hidden_size) {
  if (hidden_quantized_fp4.ptr == 0 || hidden_quantized_scale.ptr == 0 ||
      mlp_gate_up_fp4.ptr == 0 || mlp_gate_up_scale.ptr == 0 ||
      gate_up_out.ptr == 0 || barrier_state.ptr == 0 || intermediate == 0 ||
      hidden_size == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  // Alignment gates: M = 2*intermediate must be a multiple of the m-tile
  // (16), K = hidden_size must be a multiple of the GEMV K-shard
  // (kWarpsPerBlock=8 × kKPerMma=64 = 512).
  const size_t two_intermediate = 2u * intermediate;
  if ((two_intermediate & 15u) != 0u || (hidden_size & 511u) != 0u) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (two_intermediate > 0xFFFFFFFFu || hidden_size > 0xFFFFFFFFu) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  extern cudaStream_t qwen36_internal_active_stream();
  cudaStream_t stream = qwen36_internal_active_stream();

  // Persistent grid sized to match concurrent CTA capacity. The 5090 has
  // 170 SMs; with the GEMV body's ~17 KB smem + 256 threads + ~60-reg
  // pressure we typically fit 1-2 CTAs/SM (well under what shared smem
  // alone permits — register pressure is the limit). 256 CTAs is a safe
  // upper bound that keeps every CTA concurrent regardless of the wave
  // size NVCC ultimately picks, while still saturating the SM count.
  // The work-stealing loop inside the kernel iterates over m-tiles, so
  // there is no correctness dependency on the chosen grid size — only
  // on it being ≤ true concurrent capacity (no inter-CTA spinlock).
  constexpr unsigned kPersistentGridCtas = 256u;
  const unsigned total_m_tiles =
      static_cast<unsigned>(two_intermediate / qwen36_gemv::kRowsPerBlock);
  const unsigned grid_ctas =
      kPersistentGridCtas < total_m_tiles ? kPersistentGridCtas : total_m_tiles;

  constexpr unsigned kWarps = 8;
  const size_t k_over_2 = hidden_size / 2; // K of the gate+up GEMV
  const size_t a_tile_bytes =
      static_cast<size_t>(kWarps) * 2u * qwen36_gemv::kATilePerWarpBytes;
  const size_t reduction_bytes =
      2u * static_cast<size_t>(qwen36_gemv::kRowsPerBlock) *
      static_cast<size_t>(kWarps) * sizeof(float);
  const size_t sf_scale_cols = hidden_size / 16;
  const size_t sf_staging_bytes =
      static_cast<size_t>(qwen36_gemv::kRowsPerBlock) * sf_scale_cols +
      sf_scale_cols;
  const size_t smem_bytes =
      k_over_2 + a_tile_bytes + reduction_bytes + sf_staging_bytes;

  qwen36_full_attn_block_stage_f1_kernel<<<
      grid_ctas, QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS, smem_bytes,
      stream>>>(
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(hidden_quantized_fp4.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(hidden_quantized_scale.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(mlp_gate_up_fp4.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(mlp_gate_up_scale.ptr)),
      gate_up_alpha,
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(gate_up_out.ptr)),
      reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(barrier_state.ptr)),
      total_m_tiles, static_cast<uint32_t>(two_intermediate),
      static_cast<uint32_t>(hidden_size));

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_full_attn_block_stage_f2_gate_up_swiglu(
    qwen36_device_ptr_t hidden_quantized_fp4,
    qwen36_device_ptr_t hidden_quantized_scale,
    qwen36_device_ptr_t mlp_gate_up_fp4,
    qwen36_device_ptr_t mlp_gate_up_scale, float gate_up_alpha,
    qwen36_device_ptr_t gate_up_out, qwen36_device_ptr_t swiglu_fp4,
    qwen36_device_ptr_t swiglu_scale, qwen36_device_ptr_t barrier_state,
    size_t intermediate, size_t hidden_size,
    float down_input_tensor_scale) {
  if (hidden_quantized_fp4.ptr == 0 || hidden_quantized_scale.ptr == 0 ||
      mlp_gate_up_fp4.ptr == 0 || mlp_gate_up_scale.ptr == 0 ||
      gate_up_out.ptr == 0 || swiglu_fp4.ptr == 0 || swiglu_scale.ptr == 0 ||
      barrier_state.ptr == 0 || intermediate == 0 || hidden_size == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const size_t two_intermediate = 2u * intermediate;
  if ((two_intermediate & 15u) != 0u || (hidden_size & 511u) != 0u) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (two_intermediate > 0xFFFFFFFFu || hidden_size > 0xFFFFFFFFu ||
      intermediate > 0xFFFFFFFFu) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  extern cudaStream_t qwen36_internal_active_stream();
  cudaStream_t stream = qwen36_internal_active_stream();

  // Persistent grid same as F.1 (256 CTAs ≤ concurrent capacity).
  // total_m_tiles drives phase 0 work-stealing; total_groups drives
  // phase 1. Choose grid_ctas to cover the wider of the two so a
  // single CTA can always make progress on either phase (the per-phase
  // loop's early-exit handles tail CTAs when the phase is narrower).
  constexpr unsigned kPersistentGridCtas = 256u;
  const unsigned total_m_tiles =
      static_cast<unsigned>(two_intermediate / qwen36_gemv::kRowsPerBlock);
  const unsigned total_groups =
      static_cast<unsigned>((intermediate + 15u) / 16u);
  const unsigned widest_work =
      total_m_tiles > total_groups ? total_m_tiles : total_groups;
  const unsigned grid_ctas =
      kPersistentGridCtas < widest_work ? kPersistentGridCtas : widest_work;

  constexpr unsigned kWarps = 8;
  const size_t k_over_2 = hidden_size / 2;
  const size_t a_tile_bytes =
      static_cast<size_t>(kWarps) * 2u * qwen36_gemv::kATilePerWarpBytes;
  const size_t reduction_bytes =
      2u * static_cast<size_t>(qwen36_gemv::kRowsPerBlock) *
      static_cast<size_t>(kWarps) * sizeof(float);
  const size_t sf_scale_cols = hidden_size / 16;
  const size_t sf_staging_bytes =
      static_cast<size_t>(qwen36_gemv::kRowsPerBlock) * sf_scale_cols +
      sf_scale_cols;
  const size_t smem_bytes =
      k_over_2 + a_tile_bytes + reduction_bytes + sf_staging_bytes;

  qwen36_full_attn_block_stage_f2_kernel<<<
      grid_ctas, QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS, smem_bytes,
      stream>>>(
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(hidden_quantized_fp4.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(hidden_quantized_scale.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(mlp_gate_up_fp4.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(mlp_gate_up_scale.ptr)),
      gate_up_alpha,
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(gate_up_out.ptr)),
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(swiglu_fp4.ptr)),
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(swiglu_scale.ptr)),
      reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(barrier_state.ptr)),
      total_m_tiles, total_groups,
      static_cast<uint32_t>(two_intermediate),
      static_cast<uint32_t>(intermediate),
      static_cast<uint32_t>(hidden_size), down_input_tensor_scale);

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_full_attn_block_stage_f4_mlp_block(
    qwen36_device_ptr_t hidden_quantized_fp4,
    qwen36_device_ptr_t hidden_quantized_scale,
    qwen36_device_ptr_t mlp_gate_up_fp4,
    qwen36_device_ptr_t mlp_gate_up_scale, float gate_up_alpha,
    qwen36_device_ptr_t mlp_down_fp4, qwen36_device_ptr_t mlp_down_scale,
    float down_alpha, qwen36_device_ptr_t gate_up_out,
    qwen36_device_ptr_t swiglu_fp4, qwen36_device_ptr_t swiglu_scale,
    qwen36_device_ptr_t down_out, qwen36_device_ptr_t residual,
    qwen36_device_ptr_t barrier_state, size_t intermediate,
    size_t hidden_size, float down_input_tensor_scale) {
  if (hidden_quantized_fp4.ptr == 0 || hidden_quantized_scale.ptr == 0 ||
      mlp_gate_up_fp4.ptr == 0 || mlp_gate_up_scale.ptr == 0 ||
      mlp_down_fp4.ptr == 0 || mlp_down_scale.ptr == 0 ||
      gate_up_out.ptr == 0 || swiglu_fp4.ptr == 0 || swiglu_scale.ptr == 0 ||
      down_out.ptr == 0 || residual.ptr == 0 || barrier_state.ptr == 0 ||
      intermediate == 0 || hidden_size == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const size_t two_intermediate = 2u * intermediate;
  // Down GEMV K=intermediate must align to GEMV K-shard (512).
  // Gate+up GEMV K=hidden_size also aligns to 512.
  if ((two_intermediate & 15u) != 0u || (hidden_size & 15u) != 0u ||
      (hidden_size & 511u) != 0u || (intermediate & 511u) != 0u) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (two_intermediate > 0xFFFFFFFFu || hidden_size > 0xFFFFFFFFu ||
      intermediate > 0xFFFFFFFFu) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  extern cudaStream_t qwen36_internal_active_stream();
  cudaStream_t stream = qwen36_internal_active_stream();

  constexpr unsigned kPersistentGridCtas = 256u;
  const unsigned gate_up_m_tiles =
      static_cast<unsigned>(two_intermediate / qwen36_gemv::kRowsPerBlock);
  const unsigned swiglu_groups =
      static_cast<unsigned>((intermediate + 15u) / 16u);
  const unsigned down_m_tiles =
      static_cast<unsigned>(hidden_size / qwen36_gemv::kRowsPerBlock);
  unsigned widest = gate_up_m_tiles;
  if (swiglu_groups > widest) {
    widest = swiglu_groups;
  }
  if (down_m_tiles > widest) {
    widest = down_m_tiles;
  }
  const unsigned grid_ctas =
      kPersistentGridCtas < widest ? kPersistentGridCtas : widest;

  // SMEM sized to the widest phase. The gate+up GEMV needs B-staging
  // for K=hidden_size; the down GEMV needs B-staging for K=intermediate
  // (which is 17408 for Qwen3.6, much larger than hidden_size=5120).
  // Allocate the down-GEMV layout (max over phases).
  constexpr unsigned kWarps = 8;
  const size_t k_for_smem =
      intermediate > hidden_size ? intermediate : hidden_size;
  const size_t k_over_2 = k_for_smem / 2;
  const size_t a_tile_bytes =
      static_cast<size_t>(kWarps) * 2u * qwen36_gemv::kATilePerWarpBytes;
  const size_t reduction_bytes =
      2u * static_cast<size_t>(qwen36_gemv::kRowsPerBlock) *
      static_cast<size_t>(kWarps) * sizeof(float);
  const size_t sf_scale_cols = k_for_smem / 16;
  const size_t sf_staging_bytes =
      static_cast<size_t>(qwen36_gemv::kRowsPerBlock) * sf_scale_cols +
      sf_scale_cols;
  const size_t smem_bytes =
      k_over_2 + a_tile_bytes + reduction_bytes + sf_staging_bytes;

  qwen36_full_attn_block_stage_f4_kernel<<<
      grid_ctas, QWEN36_MEGAKERNEL_FULL_ATTN_BLOCK_THREADS, smem_bytes,
      stream>>>(
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(hidden_quantized_fp4.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(hidden_quantized_scale.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(mlp_gate_up_fp4.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(mlp_gate_up_scale.ptr)),
      gate_up_alpha,
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(mlp_down_fp4.ptr)),
      reinterpret_cast<const uint8_t *>(
          static_cast<uintptr_t>(mlp_down_scale.ptr)),
      down_alpha,
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(gate_up_out.ptr)),
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(swiglu_fp4.ptr)),
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(swiglu_scale.ptr)),
      reinterpret_cast<__nv_bfloat16 *>(static_cast<uintptr_t>(down_out.ptr)),
      reinterpret_cast<__nv_bfloat16 *>(static_cast<uintptr_t>(residual.ptr)),
      reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(barrier_state.ptr)),
      gate_up_m_tiles, swiglu_groups, down_m_tiles,
      static_cast<uint32_t>(hidden_size),
      static_cast<uint32_t>(intermediate),
      static_cast<uint32_t>(two_intermediate), down_input_tensor_scale);

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
