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
      q_weight_fp4, q_weight_scale, quantized_fp4, quantized_scale_e4m3,
      q_alpha, q_out, static_cast<size_t>(q_features),
      static_cast<size_t>(hidden_size));

  phase_barrier(&barrier_state[1], gridDim.x);
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
