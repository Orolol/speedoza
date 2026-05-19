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
