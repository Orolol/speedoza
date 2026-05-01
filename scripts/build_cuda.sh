#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
OUT_DIR="${OUT_DIR:-target/cuda}"
SM="${QWEN36_FP4_SM:-120}"

mkdir -p "${OUT_DIR}"

CUTLASS_DIR="${CUTLASS_DIR:-kernels-cuda/cutlass}"

CUTLASS_FLAGS=()
if [ -d "${CUTLASS_DIR}/include" ]; then
  CUTLASS_FLAGS+=(
    -I "${CUTLASS_DIR}/include"
    -I "${CUTLASS_DIR}/tools/util/include"
    --expt-relaxed-constexpr
    --extended-lambda
  )
  EXTRA_SRC=(kernels-cuda/megakernel/nvfp4_matvec_sm120.cu)
else
  echo "warn: ${CUTLASS_DIR} not found; building without Mirage megakernel" >&2
  EXTRA_SRC=()
fi

"${NVCC}" \
  -std=c++17 \
  -O3 \
  --compiler-options=-fPIC \
  -shared \
  -arch="sm_${SM}" \
  -I kernels-cuda/include \
  -I kernels-cuda \
  "${CUTLASS_FLAGS[@]}" \
  kernels-cuda/nvfp4_gemm.cu \
  kernels-cuda/deltanet.cu \
  kernels-cuda/attention.cu \
  kernels-cuda/turboquant.cu \
  kernels-cuda/ops.cu \
  kernels-cuda/runtime.cu \
  "${EXTRA_SRC[@]}" \
  -lcublasLt \
  -o "${OUT_DIR}/libqwen36_fp4_kernels.so"

echo "${OUT_DIR}/libqwen36_fp4_kernels.so"
