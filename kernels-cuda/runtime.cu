#include "qwen36_fp4.h"

#include <atomic>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

void *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(value.ptr));
}

int status(cudaError_t value) {
  return value == cudaSuccess ? QWEN36_STATUS_SUCCESS
                              : QWEN36_STATUS_CUDA_ERROR;
}

// Single-process ambient stream. Defaults to the legacy default stream (0)
// so existing callers keep working; setting a non-default stream lets the
// engine route every kernel through it (e.g. for CUDA Graph capture).
std::atomic<cudaStream_t> g_active_stream{nullptr};

} // namespace

extern "C" cudaStream_t qwen36_internal_active_stream() {
  return g_active_stream.load(std::memory_order_acquire);
}

extern "C" qwen36_cuda_stream_t qwen36_get_active_stream(void) {
  return reinterpret_cast<qwen36_cuda_stream_t>(qwen36_internal_active_stream());
}

extern "C" void qwen36_set_active_stream(qwen36_cuda_stream_t stream) {
  g_active_stream.store(reinterpret_cast<cudaStream_t>(stream),
                        std::memory_order_release);
}

extern "C" int qwen36_cuda_stream_create(qwen36_cuda_stream_t *out) {
  if (out == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  cudaStream_t stream = nullptr;
  cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    return status(err);
  }
  *out = reinterpret_cast<qwen36_cuda_stream_t>(stream);
  return QWEN36_STATUS_SUCCESS;
}

extern "C" int qwen36_cuda_stream_destroy(qwen36_cuda_stream_t stream) {
  if (stream == nullptr) {
    return QWEN36_STATUS_SUCCESS;
  }
  return status(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}

extern "C" int qwen36_cuda_stream_synchronize(qwen36_cuda_stream_t stream) {
  return status(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

extern "C" int
qwen36_cuda_stream_begin_capture(qwen36_cuda_stream_t stream) {
  return status(cudaStreamBeginCapture(reinterpret_cast<cudaStream_t>(stream),
                                       cudaStreamCaptureModeGlobal));
}

extern "C" int qwen36_cuda_stream_end_capture(qwen36_cuda_stream_t stream,
                                              qwen36_cuda_graph_t *out) {
  if (out == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  cudaGraph_t graph = nullptr;
  cudaError_t err =
      cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(stream), &graph);
  if (err != cudaSuccess) {
    return status(err);
  }
  *out = reinterpret_cast<qwen36_cuda_graph_t>(graph);
  return QWEN36_STATUS_SUCCESS;
}

extern "C" int qwen36_cuda_graph_instantiate(qwen36_cuda_graph_t graph,
                                             qwen36_cuda_graph_exec_t *out) {
  if (out == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  cudaGraphExec_t exec = nullptr;
  cudaError_t err = cudaGraphInstantiate(
      &exec, reinterpret_cast<cudaGraph_t>(graph), nullptr, nullptr, 0);
  if (err != cudaSuccess) {
    return status(err);
  }
  *out = reinterpret_cast<qwen36_cuda_graph_exec_t>(exec);
  return QWEN36_STATUS_SUCCESS;
}

extern "C" int qwen36_cuda_graph_destroy(qwen36_cuda_graph_t graph) {
  if (graph == nullptr) {
    return QWEN36_STATUS_SUCCESS;
  }
  return status(cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(graph)));
}

extern "C" int qwen36_cuda_graph_exec_destroy(qwen36_cuda_graph_exec_t exec) {
  if (exec == nullptr) {
    return QWEN36_STATUS_SUCCESS;
  }
  return status(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(exec)));
}

extern "C" int qwen36_cuda_graph_launch(qwen36_cuda_graph_exec_t exec,
                                        qwen36_cuda_stream_t stream) {
  return status(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(exec),
                                reinterpret_cast<cudaStream_t>(stream)));
}

namespace {

__global__ void increment_i32_kernel(int32_t *target) { *target += 1; }

} // namespace

extern "C" int qwen36_increment_i32(qwen36_device_ptr_t target_i32) {
  if (target_i32.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  increment_i32_kernel<<<1, 1, 0, qwen36_internal_active_stream()>>>(
      reinterpret_cast<int32_t *>(static_cast<uintptr_t>(target_i32.ptr)));
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}

extern "C" int qwen36_cuda_malloc(qwen36_device_allocation_t *out,
                                  size_t bytes) {
  if (out == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  out->ptr.ptr = 0;
  out->bytes = 0;
  if (bytes == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  void *device_ptr = nullptr;
  cudaError_t err = cudaMalloc(&device_ptr, bytes);
  if (err != cudaSuccess) {
    if (getenv("QWEN36_DEBUG_CUDA_ALLOC") != nullptr) {
      fprintf(stderr, "qwen36_cuda_malloc failed: bytes=%zu error=%d (%s)\n",
              bytes, static_cast<int>(err), cudaGetErrorString(err));
    }
    return status(err);
  }
  if (getenv("QWEN36_DEBUG_CUDA_ALLOC") != nullptr) {
    fprintf(stderr, "qwen36_cuda_malloc: bytes=%zu ptr=%p\n", bytes,
            device_ptr);
  }
  out->ptr.ptr = reinterpret_cast<uint64_t>(device_ptr);
  out->bytes = bytes;
  return QWEN36_STATUS_SUCCESS;
}

extern "C" int qwen36_cuda_free(qwen36_device_ptr_t device_ptr) {
  if (device_ptr.ptr == 0) {
    return QWEN36_STATUS_SUCCESS;
  }
  return status(cudaFree(ptr(device_ptr)));
}

extern "C" int qwen36_cuda_memcpy_h2d(qwen36_device_ptr_t dst,
                                      const void *src, size_t bytes) {
  if (dst.ptr == 0 || src == nullptr || bytes == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  return status(cudaMemcpy(ptr(dst), src, bytes, cudaMemcpyHostToDevice));
}

extern "C" int qwen36_cuda_memcpy_d2h(void *dst, qwen36_device_ptr_t src,
                                      size_t bytes) {
  if (dst == nullptr || src.ptr == 0 || bytes == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  return status(cudaMemcpy(dst, ptr(src), bytes, cudaMemcpyDeviceToHost));
}

extern "C" int qwen36_cuda_memcpy_d2d(qwen36_device_ptr_t dst,
                                      qwen36_device_ptr_t src, size_t bytes) {
  if (dst.ptr == 0 || src.ptr == 0 || bytes == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  return status(cudaMemcpy(ptr(dst), ptr(src), bytes, cudaMemcpyDeviceToDevice));
}

extern "C" int qwen36_cuda_memset(qwen36_device_ptr_t dst, int value,
                                  size_t bytes) {
  if (dst.ptr == 0 || bytes == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  return status(cudaMemset(ptr(dst), value, bytes));
}

extern "C" int qwen36_cuda_synchronize(void) {
  return status(cudaDeviceSynchronize());
}
