#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
OUT_DIR="${OUT_DIR:-target/cuda}"
SM="${QWEN36_FP4_SM:-120}"

"${NVCC}" \
  -std=c++17 \
  -O2 \
  -arch="sm_${SM}" \
  -I kernels-cuda/include \
  kernels-cuda/smoke.cu \
  -L "${OUT_DIR}" \
  -lqwen36_fp4_kernels \
  -o "${OUT_DIR}/qwen36_cuda_smoke"

LD_LIBRARY_PATH="${OUT_DIR}:${LD_LIBRARY_PATH:-}" "${OUT_DIR}/qwen36_cuda_smoke"

