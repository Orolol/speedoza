#include "qwen36_fp4.h"

#include <cuda_runtime.h>

namespace {

void *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(value.ptr));
}

int status(cudaError_t value) {
  return value == cudaSuccess ? QWEN36_STATUS_SUCCESS
                              : QWEN36_STATUS_CUDA_ERROR;
}

} // namespace

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
    return status(err);
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
