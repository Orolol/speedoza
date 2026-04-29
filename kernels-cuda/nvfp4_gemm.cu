#include "qwen36_fp4.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#include <mutex>
#include <unordered_map>

namespace {

void *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(value.ptr));
}

int status(cublasStatus_t value) {
  return value == CUBLAS_STATUS_SUCCESS ? QWEN36_STATUS_SUCCESS
                                        : QWEN36_STATUS_CUBLAS_ERROR;
}

int fail(cublasStatus_t value, const char *where) {
  if (value != CUBLAS_STATUS_SUCCESS &&
      getenv("QWEN36_DEBUG_CUBLASLT") != nullptr) {
    fprintf(stderr, "%s: cuBLASLt status %d\n", where,
            static_cast<int>(value));
  }
  return status(value);
}

struct GemmKey {
  size_t m;
  size_t n;
  size_t k;
  size_t workspace_bytes;

  bool operator==(const GemmKey &other) const {
    return m == other.m && n == other.n && k == other.k &&
           workspace_bytes == other.workspace_bytes;
  }
};

struct GemmKeyHash {
  size_t operator()(const GemmKey &key) const {
    size_t hash = key.m;
    hash ^= key.n + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    hash ^= key.k + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    hash ^= key.workspace_bytes + 0x9e3779b97f4a7c15ULL + (hash << 6) +
            (hash >> 2);
    return hash;
  }
};

std::mutex &cublas_mutex() {
  static std::mutex mutex;
  return mutex;
}

cublasLtHandle_t &shared_handle() {
  static cublasLtHandle_t handle = nullptr;
  return handle;
}

std::unordered_map<GemmKey, cublasLtMatmulHeuristicResult_t, GemmKeyHash>
    &algo_cache() {
  static std::unordered_map<GemmKey, cublasLtMatmulHeuristicResult_t,
                            GemmKeyHash>
      cache;
  return cache;
}

} // namespace

extern "C" int qwen36_nvfp4_gemm(const qwen36_nvfp4_gemm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->m == 0 || spec->n == 0 || spec->k == 0 || spec->a_fp4.ptr == 0 ||
      spec->b_fp4.ptr == 0 || spec->c_bf16.ptr == 0 || spec->a_scale.ptr == 0 ||
      spec->b_scale.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;

  auto cleanup = [&]() {
    if (pref != nullptr) {
      cublasLtMatmulPreferenceDestroy(pref);
    }
    if (d_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(d_desc);
    }
    if (c_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(c_desc);
    }
    if (b_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(b_desc);
    }
    if (a_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(a_desc);
    }
    if (op_desc != nullptr) {
      cublasLtMatmulDescDestroy(op_desc);
    }
  };

  cublasStatus_t rc = CUBLAS_STATUS_SUCCESS;
  {
    std::lock_guard<std::mutex> lock(cublas_mutex());
    if (shared_handle() == nullptr) {
      rc = cublasLtCreate(&shared_handle());
      if (rc != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return fail(rc, "cublasLtCreate");
      }
    }
    handle = shared_handle();
  }

  rc = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "cublasLtMatmulDescCreate");
  }

  cublasOperation_t trans_a = CUBLAS_OP_T;
  cublasOperation_t trans_b = CUBLAS_OP_N;
  rc = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "CUBLASLT_MATMUL_DESC_TRANSA");
  }
  rc = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "CUBLASLT_MATMUL_DESC_TRANSB");
  }

  cublasLtMatmulMatrixScale_t scale_mode =
      CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  rc = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode,
      sizeof(scale_mode));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "CUBLASLT_MATMUL_DESC_A_SCALE_MODE");
  }
  rc = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode,
      sizeof(scale_mode));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "CUBLASLT_MATMUL_DESC_B_SCALE_MODE");
  }

  void *a_scale = ptr(spec->a_scale);
  void *b_scale = ptr(spec->b_scale);
  rc = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale,
      sizeof(a_scale));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "CUBLASLT_MATMUL_DESC_A_SCALE_POINTER");
  }
  rc = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale,
      sizeof(b_scale));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "CUBLASLT_MATMUL_DESC_B_SCALE_POINTER");
  }

  rc = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, spec->k, spec->m,
                                  spec->k);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "cublasLtMatrixLayoutCreate A");
  }
  rc = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, spec->k, spec->n,
                                  spec->k);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "cublasLtMatrixLayoutCreate B");
  }
  rc = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, spec->m, spec->n,
                                  spec->m);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "cublasLtMatrixLayoutCreate C");
  }
  rc = cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, spec->m, spec->n,
                                  spec->m);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "cublasLtMatrixLayoutCreate D");
  }

  rc = cublasLtMatmulPreferenceCreate(&pref);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "cublasLtMatmulPreferenceCreate");
  }
  size_t workspace_bytes = spec->workspace_bytes;
  rc = cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");
  }
  const GemmKey key{spec->m, spec->n, spec->k, spec->workspace_bytes};
  {
    std::lock_guard<std::mutex> lock(cublas_mutex());
    auto found = algo_cache().find(key);
    if (found != algo_cache().end()) {
      heuristic = found->second;
      returned_results = 1;
    } else {
      rc = cublasLtMatmulAlgoGetHeuristic(handle, op_desc, a_desc, b_desc,
                                          c_desc, d_desc, pref, 1, &heuristic,
                                          &returned_results);
      if (rc == CUBLAS_STATUS_SUCCESS && returned_results != 0) {
        algo_cache().emplace(key, heuristic);
      }
    }
  }
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "cublasLtMatmulAlgoGetHeuristic");
  }
  if (returned_results == 0) {
    if (getenv("QWEN36_DEBUG_CUBLASLT") != nullptr) {
      fprintf(stderr, "cublasLtMatmulAlgoGetHeuristic: no results\n");
    }
    cleanup();
    return QWEN36_STATUS_CUBLAS_ERROR;
  }

  float alpha = spec->alpha == 0.0f ? 1.0f : spec->alpha;
  float beta = 0.0f;
  {
    std::lock_guard<std::mutex> lock(cublas_mutex());
    rc = cublasLtMatmul(handle, op_desc, &alpha, ptr(spec->a_fp4), a_desc,
                        ptr(spec->b_fp4), b_desc, &beta, ptr(spec->c_bf16),
                        c_desc, ptr(spec->c_bf16), d_desc, &heuristic.algo,
                        ptr(spec->workspace), spec->workspace_bytes, 0);
  }
  if (rc != CUBLAS_STATUS_SUCCESS &&
      getenv("QWEN36_DEBUG_CUBLASLT") != nullptr) {
    fprintf(stderr, "cublasLtMatmul dims: m=%zu n=%zu k=%zu workspace=%zu alpha=%g\n",
            spec->m, spec->n, spec->k, spec->workspace_bytes, alpha);
  }
  if (rc == CUBLAS_STATUS_SUCCESS &&
      getenv("QWEN36_DEBUG_CUBLASLT_SYNC") != nullptr) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      if (getenv("QWEN36_DEBUG_CUBLASLT") != nullptr) {
        fprintf(stderr, "cublasLtMatmul sync: CUDA error %d (%s), m=%zu n=%zu k=%zu\n",
                static_cast<int>(err), cudaGetErrorString(err), spec->m,
                spec->n, spec->k);
      }
      cleanup();
      return QWEN36_STATUS_CUDA_ERROR;
    }
  }
  cleanup();
  return fail(rc, "cublasLtMatmul");
}
