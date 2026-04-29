#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
OUT_DIR="${OUT_DIR:-target/cuda}"
SM="${QWEN36_FP4_SM:-120}"

mkdir -p "${OUT_DIR}"

"${NVCC}" \
  -std=c++17 \
  -O3 \
  --compiler-options=-fPIC \
  -shared \
  -arch="sm_${SM}" \
  -I kernels-cuda/include \
  kernels-cuda/nvfp4_gemm.cu \
  kernels-cuda/deltanet.cu \
  kernels-cuda/attention.cu \
  kernels-cuda/turboquant.cu \
  kernels-cuda/ops.cu \
  kernels-cuda/runtime.cu \
  -lcublasLt \
  -o "${OUT_DIR}/libqwen36_fp4_kernels.so"

echo "${OUT_DIR}/libqwen36_fp4_kernels.so"
