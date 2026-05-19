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
