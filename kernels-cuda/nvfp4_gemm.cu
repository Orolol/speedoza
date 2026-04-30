#include "qwen36_fp4.h"
#include "active_stream.h"

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

struct GemmPlan {
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
};

std::unordered_map<GemmKey, GemmPlan *, GemmKeyHash> &plan_cache() {
  static std::unordered_map<GemmKey, GemmPlan *, GemmKeyHash> cache;
  return cache;
}

std::unordered_map<GemmKey, GemmPlan *, GemmKeyHash> &bf16_plan_cache() {
  static std::unordered_map<GemmKey, GemmPlan *, GemmKeyHash> cache;
  return cache;
}

void destroy_plan(GemmPlan *plan) {
  if (plan == nullptr) {
    return;
  }
  if (plan->d_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(plan->d_desc);
  }
  if (plan->c_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(plan->c_desc);
  }
  if (plan->b_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(plan->b_desc);
  }
  if (plan->a_desc != nullptr) {
    cublasLtMatrixLayoutDestroy(plan->a_desc);
  }
  if (plan->op_desc != nullptr) {
    cublasLtMatmulDescDestroy(plan->op_desc);
  }
  delete plan;
}

int set_scale_pointers(GemmPlan *plan, const qwen36_nvfp4_gemm_spec_t *spec) {
  void *a_scale = ptr(spec->a_scale);
  void *b_scale = ptr(spec->b_scale);
  cublasStatus_t rc = cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale,
      sizeof(a_scale));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    return fail(rc, "CUBLASLT_MATMUL_DESC_A_SCALE_POINTER");
  }
  rc = cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale,
      sizeof(b_scale));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    return fail(rc, "CUBLASLT_MATMUL_DESC_B_SCALE_POINTER");
  }
  return QWEN36_STATUS_SUCCESS;
}

int create_plan(cublasLtHandle_t handle, const qwen36_nvfp4_gemm_spec_t *spec,
                GemmPlan **out) {
  auto *plan = new GemmPlan();
  cublasLtMatmulPreference_t pref = nullptr;

  auto cleanup = [&]() {
    if (pref != nullptr) {
      cublasLtMatmulPreferenceDestroy(pref);
    }
    destroy_plan(plan);
  };

  cublasStatus_t rc =
      cublasLtMatmulDescCreate(&plan->op_desc, CUBLAS_COMPUTE_32F,
                               CUDA_R_32F);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "cublasLtMatmulDescCreate");
  }

  cublasOperation_t trans_a = CUBLAS_OP_T;
  cublasOperation_t trans_b = CUBLAS_OP_N;
  rc = cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "CUBLASLT_MATMUL_DESC_TRANSA");
  }
  rc = cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "CUBLASLT_MATMUL_DESC_TRANSB");
  }

  cublasLtMatmulMatrixScale_t scale_mode =
      CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  rc = cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode,
      sizeof(scale_mode));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "CUBLASLT_MATMUL_DESC_A_SCALE_MODE");
  }
  rc = cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode,
      sizeof(scale_mode));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "CUBLASLT_MATMUL_DESC_B_SCALE_MODE");
  }

  int scale_status = set_scale_pointers(plan, spec);
  if (scale_status != QWEN36_STATUS_SUCCESS) {
    cleanup();
    return scale_status;
  }

  rc = cublasLtMatrixLayoutCreate(&plan->a_desc, CUDA_R_4F_E2M1, spec->k,
                                  spec->m, spec->k);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "cublasLtMatrixLayoutCreate A");
  }
  rc = cublasLtMatrixLayoutCreate(&plan->b_desc, CUDA_R_4F_E2M1, spec->k,
                                  spec->n, spec->k);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "cublasLtMatrixLayoutCreate B");
  }
  rc = cublasLtMatrixLayoutCreate(&plan->c_desc, CUDA_R_16BF, spec->m, spec->n,
                                  spec->m);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "cublasLtMatrixLayoutCreate C");
  }
  rc = cublasLtMatrixLayoutCreate(&plan->d_desc, CUDA_R_16BF, spec->m, spec->n,
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

  int returned_results = 0;
  rc = cublasLtMatmulAlgoGetHeuristic(
      handle, plan->op_desc, plan->a_desc, plan->b_desc, plan->c_desc,
      plan->d_desc, pref, 1, &plan->heuristic, &returned_results);
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

  cublasLtMatmulPreferenceDestroy(pref);
  *out = plan;
  return QWEN36_STATUS_SUCCESS;
}

int create_bf16_plan(cublasLtHandle_t handle,
                     const qwen36_bf16_gemm_spec_t *spec, GemmPlan **out) {
  auto *plan = new GemmPlan();
  cublasLtMatmulPreference_t pref = nullptr;

  auto cleanup = [&]() {
    if (pref != nullptr) {
      cublasLtMatmulPreferenceDestroy(pref);
    }
    destroy_plan(plan);
  };

  cublasStatus_t rc =
      cublasLtMatmulDescCreate(&plan->op_desc, CUBLAS_COMPUTE_32F,
                               CUDA_R_32F);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "bf16 cublasLtMatmulDescCreate");
  }

  cublasOperation_t trans_a = CUBLAS_OP_T;
  cublasOperation_t trans_b = CUBLAS_OP_N;
  rc = cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "bf16 CUBLASLT_MATMUL_DESC_TRANSA");
  }
  rc = cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "bf16 CUBLASLT_MATMUL_DESC_TRANSB");
  }

  rc = cublasLtMatrixLayoutCreate(&plan->a_desc, CUDA_R_16BF, spec->k,
                                  spec->m, spec->k);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "bf16 cublasLtMatrixLayoutCreate A");
  }
  rc = cublasLtMatrixLayoutCreate(&plan->b_desc, CUDA_R_16BF, spec->k,
                                  spec->n, spec->k);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "bf16 cublasLtMatrixLayoutCreate B");
  }
  rc = cublasLtMatrixLayoutCreate(&plan->c_desc, CUDA_R_16BF, spec->m, spec->n,
                                  spec->m);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "bf16 cublasLtMatrixLayoutCreate C");
  }
  rc = cublasLtMatrixLayoutCreate(&plan->d_desc, CUDA_R_16BF, spec->m, spec->n,
                                  spec->m);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "bf16 cublasLtMatrixLayoutCreate D");
  }

  rc = cublasLtMatmulPreferenceCreate(&pref);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "bf16 cublasLtMatmulPreferenceCreate");
  }
  size_t workspace_bytes = spec->workspace_bytes;
  rc = cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes));
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "bf16 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");
  }

  int returned_results = 0;
  rc = cublasLtMatmulAlgoGetHeuristic(
      handle, plan->op_desc, plan->a_desc, plan->b_desc, plan->c_desc,
      plan->d_desc, pref, 1, &plan->heuristic, &returned_results);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return fail(rc, "bf16 cublasLtMatmulAlgoGetHeuristic");
  }
  if (returned_results == 0) {
    if (getenv("QWEN36_DEBUG_CUBLASLT") != nullptr) {
      fprintf(stderr, "bf16 cublasLtMatmulAlgoGetHeuristic: no results\n");
    }
    cleanup();
    return QWEN36_STATUS_CUBLAS_ERROR;
  }

  cublasLtMatmulPreferenceDestroy(pref);
  *out = plan;
  return QWEN36_STATUS_SUCCESS;
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

  cublasStatus_t rc = CUBLAS_STATUS_SUCCESS;
  cublasLtHandle_t handle = nullptr;
  GemmPlan *plan = nullptr;
  const GemmKey key{spec->m, spec->n, spec->k, spec->workspace_bytes};
  {
    std::lock_guard<std::mutex> lock(cublas_mutex());
    if (shared_handle() == nullptr) {
      rc = cublasLtCreate(&shared_handle());
      if (rc != CUBLAS_STATUS_SUCCESS) {
        return fail(rc, "cublasLtCreate");
      }
    }
    handle = shared_handle();
    auto found = plan_cache().find(key);
    if (found != plan_cache().end()) {
      plan = found->second;
    } else {
      int plan_status = create_plan(handle, spec, &plan);
      if (plan_status != QWEN36_STATUS_SUCCESS) {
        return plan_status;
      }
      plan_cache().emplace(key, plan);
    }
    int scale_status = set_scale_pointers(plan, spec);
    if (scale_status != QWEN36_STATUS_SUCCESS) {
      return scale_status;
    }
  }

  float alpha = spec->alpha == 0.0f ? 1.0f : spec->alpha;
  float beta = 0.0f;
  {
    std::lock_guard<std::mutex> lock(cublas_mutex());
    rc = cublasLtMatmul(
        handle, plan->op_desc, &alpha, ptr(spec->a_fp4), plan->a_desc,
        ptr(spec->b_fp4), plan->b_desc, &beta, ptr(spec->c_bf16),
        plan->c_desc, ptr(spec->c_bf16), plan->d_desc, &plan->heuristic.algo,
        ptr(spec->workspace), spec->workspace_bytes,
        qwen36_internal_active_stream());
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
      return QWEN36_STATUS_CUDA_ERROR;
    }
  }
  return fail(rc, "cublasLtMatmul");
}

extern "C" int qwen36_bf16_gemm(const qwen36_bf16_gemm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->m == 0 || spec->n == 0 || spec->k == 0 || spec->a_bf16.ptr == 0 ||
      spec->b_bf16.ptr == 0 || spec->c_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  cublasStatus_t rc = CUBLAS_STATUS_SUCCESS;
  cublasLtHandle_t handle = nullptr;
  GemmPlan *plan = nullptr;
  const GemmKey key{spec->m, spec->n, spec->k, spec->workspace_bytes};
  {
    std::lock_guard<std::mutex> lock(cublas_mutex());
    if (shared_handle() == nullptr) {
      rc = cublasLtCreate(&shared_handle());
      if (rc != CUBLAS_STATUS_SUCCESS) {
        return fail(rc, "cublasLtCreate");
      }
    }
    handle = shared_handle();
    auto found = bf16_plan_cache().find(key);
    if (found != bf16_plan_cache().end()) {
      plan = found->second;
    } else {
      int plan_status = create_bf16_plan(handle, spec, &plan);
      if (plan_status != QWEN36_STATUS_SUCCESS) {
        return plan_status;
      }
      bf16_plan_cache().emplace(key, plan);
    }
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  {
    std::lock_guard<std::mutex> lock(cublas_mutex());
    rc = cublasLtMatmul(
        handle, plan->op_desc, &alpha, ptr(spec->a_bf16), plan->a_desc,
        ptr(spec->b_bf16), plan->b_desc, &beta, ptr(spec->c_bf16),
        plan->c_desc, ptr(spec->c_bf16), plan->d_desc, &plan->heuristic.algo,
        ptr(spec->workspace), spec->workspace_bytes,
        qwen36_internal_active_stream());
  }
  if (rc != CUBLAS_STATUS_SUCCESS &&
      getenv("QWEN36_DEBUG_CUBLASLT") != nullptr) {
    fprintf(stderr,
            "bf16 cublasLtMatmul dims: m=%zu n=%zu k=%zu workspace=%zu\n",
            spec->m, spec->n, spec->k, spec->workspace_bytes);
  }
  if (rc == CUBLAS_STATUS_SUCCESS &&
      getenv("QWEN36_DEBUG_CUBLASLT_SYNC") != nullptr) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      if (getenv("QWEN36_DEBUG_CUBLASLT") != nullptr) {
        fprintf(stderr,
                "bf16 cublasLtMatmul sync: CUDA error %d (%s), m=%zu n=%zu k=%zu\n",
                static_cast<int>(err), cudaGetErrorString(err), spec->m,
                spec->n, spec->k);
      }
      return QWEN36_STATUS_CUDA_ERROR;
    }
  }
  return fail(rc, "bf16 cublasLtMatmul");
}
