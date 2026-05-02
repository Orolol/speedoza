#include "qwen36_fp4.h"

#include <atomic>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

std::atomic<unsigned long long> g_malloc_calls{0};
std::atomic<unsigned long long> g_free_calls{0};
std::atomic<unsigned long long> g_h2d_calls{0};
std::atomic<unsigned long long> g_h2d_bytes{0};
std::atomic<unsigned long long> g_d2h_calls{0};
std::atomic<unsigned long long> g_d2h_bytes{0};
std::atomic<unsigned long long> g_d2d_calls{0};
std::atomic<unsigned long long> g_d2d_bytes{0};
std::atomic<unsigned long long> g_d2d_async_calls{0};
std::atomic<unsigned long long> g_d2d_async_bytes{0};
std::atomic<unsigned long long> g_memset_calls{0};
std::atomic<unsigned long long> g_memset_bytes{0};
std::atomic<unsigned long long> g_synchronize_calls{0};
std::atomic<unsigned long long> g_stream_synchronize_calls{0};
std::atomic<unsigned long long> g_graph_launch_calls{0};

void add_counter(std::atomic<unsigned long long> &counter,
                 unsigned long long value = 1) {
  counter.fetch_add(value, std::memory_order_relaxed);
}

void copy_cstr(char *dst, size_t dst_len, const char *src) {
  if (dst_len == 0) {
    return;
  }
  if (src == nullptr) {
    dst[0] = '\0';
    return;
  }
  strncpy(dst, src, dst_len - 1);
  dst[dst_len - 1] = '\0';
}

void fill_loaded_library_path(const char *soname, const char *symbol,
                              char *dst, size_t dst_len) {
  if (dst_len == 0) {
    return;
  }
  dst[0] = '\0';
  void *handle = dlopen(soname, RTLD_LAZY | RTLD_NOLOAD);
  if (handle == nullptr) {
    handle = dlopen(soname, RTLD_LAZY);
  }
  if (handle == nullptr) {
    return;
  }
  void *sym = dlsym(handle, symbol);
  Dl_info info{};
  if (sym != nullptr && dladdr(sym, &info) != 0 && info.dli_fname != nullptr) {
    copy_cstr(dst, dst_len, info.dli_fname);
  }
}

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
  add_counter(g_stream_synchronize_calls);
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
  add_counter(g_graph_launch_calls);
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
  add_counter(g_malloc_calls);
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
  add_counter(g_free_calls);
  return status(cudaFree(ptr(device_ptr)));
}

extern "C" int qwen36_cuda_memcpy_h2d(qwen36_device_ptr_t dst,
                                      const void *src, size_t bytes) {
  if (dst.ptr == 0 || src == nullptr || bytes == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  add_counter(g_h2d_calls);
  add_counter(g_h2d_bytes, bytes);
  return status(cudaMemcpy(ptr(dst), src, bytes, cudaMemcpyHostToDevice));
}

extern "C" int qwen36_cuda_memcpy_d2h(void *dst, qwen36_device_ptr_t src,
                                      size_t bytes) {
  if (dst == nullptr || src.ptr == 0 || bytes == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  add_counter(g_d2h_calls);
  add_counter(g_d2h_bytes, bytes);
  cudaStream_t stream = qwen36_internal_active_stream();
  if (stream != nullptr) {
    cudaError_t err =
        cudaMemcpyAsync(dst, ptr(src), bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
      return status(err);
    }
    add_counter(g_stream_synchronize_calls);
    return status(cudaStreamSynchronize(stream));
  }
  return status(cudaMemcpy(dst, ptr(src), bytes, cudaMemcpyDeviceToHost));
}

extern "C" int qwen36_cuda_memcpy_d2d(qwen36_device_ptr_t dst,
                                      qwen36_device_ptr_t src, size_t bytes) {
  if (dst.ptr == 0 || src.ptr == 0 || bytes == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  add_counter(g_d2d_calls);
  add_counter(g_d2d_bytes, bytes);
  return status(cudaMemcpy(ptr(dst), ptr(src), bytes, cudaMemcpyDeviceToDevice));
}

extern "C" int qwen36_cuda_memcpy_d2d_async(qwen36_device_ptr_t dst,
                                            qwen36_device_ptr_t src,
                                            size_t bytes) {
  if (dst.ptr == 0 || src.ptr == 0 || bytes == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = qwen36_internal_active_stream();
  if (stream == nullptr) {
    add_counter(g_d2d_calls);
    add_counter(g_d2d_bytes, bytes);
    return status(cudaMemcpy(ptr(dst), ptr(src), bytes,
                             cudaMemcpyDeviceToDevice));
  }
  add_counter(g_d2d_async_calls);
  add_counter(g_d2d_async_bytes, bytes);
  return status(cudaMemcpyAsync(ptr(dst), ptr(src), bytes,
                                cudaMemcpyDeviceToDevice, stream));
}

extern "C" int qwen36_cuda_memset(qwen36_device_ptr_t dst, int value,
                                  size_t bytes) {
  if (dst.ptr == 0 || bytes == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  add_counter(g_memset_calls);
  add_counter(g_memset_bytes, bytes);
  return status(cudaMemset(ptr(dst), value, bytes));
}

extern "C" int qwen36_cuda_synchronize(void) {
  add_counter(g_synchronize_calls);
  return status(cudaDeviceSynchronize());
}

extern "C" int
qwen36_cuda_get_diagnostics(qwen36_cuda_diagnostics_t *out) {
  if (out == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  memset(out, 0, sizeof(*out));

  cudaError_t err = cudaDriverGetVersion(&out->driver_version);
  if (err != cudaSuccess) {
    out->last_cuda_error = static_cast<int>(err);
    copy_cstr(out->last_cuda_error_name, sizeof(out->last_cuda_error_name),
              cudaGetErrorName(err));
    copy_cstr(out->last_cuda_error_string,
              sizeof(out->last_cuda_error_string), cudaGetErrorString(err));
  } else {
    err = cudaRuntimeGetVersion(&out->runtime_version);
    if (err != cudaSuccess) {
      out->last_cuda_error = static_cast<int>(err);
      copy_cstr(out->last_cuda_error_name, sizeof(out->last_cuda_error_name),
                cudaGetErrorName(err));
      copy_cstr(out->last_cuda_error_string,
                sizeof(out->last_cuda_error_string), cudaGetErrorString(err));
    }
  }
  err = cudaGetDeviceCount(&out->device_count);
  if (err == cudaSuccess && out->device_count > 0) {
    cudaGetDevice(&out->active_device);
    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, out->active_device);
    if (err == cudaSuccess) {
      out->sm_major = prop.major;
      out->sm_minor = prop.minor;
      out->multiprocessor_count = prop.multiProcessorCount;
      out->total_global_mem = prop.totalGlobalMem;
      copy_cstr(out->device_name, sizeof(out->device_name), prop.name);
    }
  }

  fill_loaded_library_path("libcuda.so.1", "cuInit", out->libcuda_path,
                           sizeof(out->libcuda_path));
  Dl_info cudart_info{};
  if (dladdr(reinterpret_cast<const void *>(&cudaRuntimeGetVersion),
             &cudart_info) != 0 &&
      cudart_info.dli_fname != nullptr) {
    copy_cstr(out->cudart_path, sizeof(out->cudart_path),
              cudart_info.dli_fname);
  }

  cudaError_t last = cudaPeekAtLastError();
  if (last != cudaSuccess || out->last_cuda_error == 0) {
    out->last_cuda_error = static_cast<int>(last);
    copy_cstr(out->last_cuda_error_name, sizeof(out->last_cuda_error_name),
              cudaGetErrorName(last));
    copy_cstr(out->last_cuda_error_string, sizeof(out->last_cuda_error_string),
              cudaGetErrorString(last));
  }
  return QWEN36_STATUS_SUCCESS;
}

extern "C" int qwen36_cuda_counters_reset(void) {
  g_malloc_calls.store(0, std::memory_order_relaxed);
  g_free_calls.store(0, std::memory_order_relaxed);
  g_h2d_calls.store(0, std::memory_order_relaxed);
  g_h2d_bytes.store(0, std::memory_order_relaxed);
  g_d2h_calls.store(0, std::memory_order_relaxed);
  g_d2h_bytes.store(0, std::memory_order_relaxed);
  g_d2d_calls.store(0, std::memory_order_relaxed);
  g_d2d_bytes.store(0, std::memory_order_relaxed);
  g_d2d_async_calls.store(0, std::memory_order_relaxed);
  g_d2d_async_bytes.store(0, std::memory_order_relaxed);
  g_memset_calls.store(0, std::memory_order_relaxed);
  g_memset_bytes.store(0, std::memory_order_relaxed);
  g_synchronize_calls.store(0, std::memory_order_relaxed);
  g_stream_synchronize_calls.store(0, std::memory_order_relaxed);
  g_graph_launch_calls.store(0, std::memory_order_relaxed);
  return QWEN36_STATUS_SUCCESS;
}

extern "C" int qwen36_cuda_counters_read(qwen36_cuda_counters_t *out) {
  if (out == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  out->malloc_calls = g_malloc_calls.load(std::memory_order_relaxed);
  out->free_calls = g_free_calls.load(std::memory_order_relaxed);
  out->h2d_calls = g_h2d_calls.load(std::memory_order_relaxed);
  out->h2d_bytes = g_h2d_bytes.load(std::memory_order_relaxed);
  out->d2h_calls = g_d2h_calls.load(std::memory_order_relaxed);
  out->d2h_bytes = g_d2h_bytes.load(std::memory_order_relaxed);
  out->d2d_calls = g_d2d_calls.load(std::memory_order_relaxed);
  out->d2d_bytes = g_d2d_bytes.load(std::memory_order_relaxed);
  out->d2d_async_calls = g_d2d_async_calls.load(std::memory_order_relaxed);
  out->d2d_async_bytes = g_d2d_async_bytes.load(std::memory_order_relaxed);
  out->memset_calls = g_memset_calls.load(std::memory_order_relaxed);
  out->memset_bytes = g_memset_bytes.load(std::memory_order_relaxed);
  out->synchronize_calls = g_synchronize_calls.load(std::memory_order_relaxed);
  out->stream_synchronize_calls =
      g_stream_synchronize_calls.load(std::memory_order_relaxed);
  out->graph_launch_calls = g_graph_launch_calls.load(std::memory_order_relaxed);
  return QWEN36_STATUS_SUCCESS;
}

// Pin a memory window to the L2 cache via the active stream's access policy
// window. Useful for hot data structures that fit in L2 (e.g. the DeltaNet
// recurrent state), so every recurrent read is an L2 hit instead of an HBM
// fetch. `hit_ratio` ∈ [0, 1] is the fraction of accesses to keep cached;
// 1.0 means "cache the whole window" (still best-effort under L2 pressure).
// Setting on the active stream applies to all kernels submitted to it,
// including those captured into a CUDA graph.
extern "C" int qwen36_cuda_set_l2_access_window(qwen36_device_ptr_t base,
                                                size_t bytes,
                                                float hit_ratio) {
  if (base.ptr == 0 || bytes == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (!(hit_ratio >= 0.0f) || hit_ratio > 1.0f) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = qwen36_internal_active_stream();
  if (stream == nullptr) {
    // No-op on the legacy default stream — its attributes are not settable.
    return QWEN36_STATUS_SUCCESS;
  }
  cudaStreamAttrValue attrs = {};
  attrs.accessPolicyWindow.base_ptr = ptr(base);
  attrs.accessPolicyWindow.num_bytes = bytes;
  attrs.accessPolicyWindow.hitRatio = hit_ratio;
  attrs.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attrs.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaError_t err = cudaStreamSetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow, &attrs);
  return status(err);
}

extern "C" int qwen36_cuda_clear_l2_access_window(void) {
  cudaStream_t stream = qwen36_internal_active_stream();
  if (stream == nullptr) {
    return QWEN36_STATUS_SUCCESS;
  }
  cudaStreamAttrValue attrs = {};
  cudaError_t err = cudaStreamSetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow, &attrs);
  return status(err);
}
