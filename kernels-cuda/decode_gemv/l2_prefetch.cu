// Productive-spin L2 warmup kernel. Read-only walk over a global-memory
// region that pulls every byte into L2 without performing any computation.
// Used by the decode hot path to overlap MLP-weight prefetch with the small
// (24-CTA) full-attn decode kernel on the 5090's idle SMs.
//
// Dispatch contract: launched on the registered prefetch stream
// (`qwen36_internal_prefetch_stream()`), not on the active stream. Callers
// synchronize via `qwen36_cuda_event_record` / `qwen36_cuda_stream_wait_event`
// so the whole productive-spin sequence remains capturable in the decode
// CUDA graph. See `Phase 1` in the productive-spin plan.

#include "../active_stream.h"
#include "qwen36_fp4.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// One thread = one 16B uint4 load per stride. The act of issuing the load
// brings the surrounding L2 line (128B on Blackwell) into cache; we use a
// noop asm block to keep the read live so the compiler can't drop it.
__global__ void __launch_bounds__(128)
qwen36_l2_prefetch_kernel(const uint8_t *__restrict__ base, size_t bytes) {
  const size_t total_threads =
      static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
  const size_t thread_idx =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
      threadIdx.x;
  const size_t stride = total_threads * sizeof(uint4);

  for (size_t off = thread_idx * sizeof(uint4); off + sizeof(uint4) <= bytes;
       off += stride) {
    uint4 v = *reinterpret_cast<const uint4 *>(base + off);
    asm volatile("" : : "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
  }
}

} // namespace

extern "C" int qwen36_l2_prefetch(qwen36_device_ptr_t base, size_t bytes,
                                  int target_cta_count) {
  if (base.ptr == 0 || bytes == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (target_cta_count <= 0) {
    target_cta_count = 128;
  }
  cudaStream_t stream = qwen36_internal_prefetch_stream();
  if (stream == nullptr) {
    // The engine must register a prefetch stream before calling this.
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const uint8_t *ptr =
      reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(base.ptr));
  qwen36_l2_prefetch_kernel<<<static_cast<unsigned int>(target_cta_count), 128,
                              0, stream>>>(ptr, bytes);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
