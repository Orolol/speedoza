#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
OUT_DIR="${OUT_DIR:-target/cuda}"
SM="${QWEN36_FP4_SM:-120a}"

mkdir -p "${OUT_DIR}"

# decode_gemv is pure CUDA, always compile the real one.
EXTRA_SRC=(kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu)
EXTRA_SRC+=(kernels-cuda/decode_gemv/l2_prefetch.cu)
# Stage-0 decode interpreter substrate — pure CUDA, no CUTLASS dep.
EXTRA_SRC+=(kernels-cuda/interpreter/interpreter_sm120.cu)
# DFlash drafter attention (Phase C v1 + Phase 1 FA-tiled) — pure CUDA.
EXTRA_SRC+=(kernels-cuda/drafter_attention.cu)
EXTRA_SRC+=(kernels-cuda/drafter_attention_flash.cu)

"${NVCC}" \
  -std=c++17 \
  -O3 \
  --compiler-options=-fPIC \
  -shared \
  -arch="sm_${SM}" \
  -I kernels-cuda/include \
  -I kernels-cuda \
  kernels-cuda/nvfp4_gemm.cu \
  kernels-cuda/deltanet.cu \
  kernels-cuda/deltanet_prefill.cu \
  kernels-cuda/attention.cu \
  kernels-cuda/attention_flash_prefill.cu \
  kernels-cuda/attention_flash_splitk.cu \
  kernels-cuda/attention_decode_tiled.cu \
  kernels-cuda/attention_sage_prefill.cu \
  kernels-cuda/turboquant.cu \
  kernels-cuda/ops.cu \
  kernels-cuda/lm_head_fp8.cu \
  kernels-cuda/runtime.cu \
  "${EXTRA_SRC[@]}" \
  -lcublasLt \
  -ldl \
  -o "${OUT_DIR}/libqwen36_fp4_kernels.so"

echo "${OUT_DIR}/libqwen36_fp4_kernels.so"
