#!/usr/bin/env bash
# Wrap nsys around the release bench binary for kernel-level profiling.
#
# Usage:
#   scripts/profile_bench.sh [PROMPT_TOKENS] [MAX_NEW_TOKENS] [MTP_DRAFTS]
# Defaults: 1024 128 3
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${QWEN36_MODEL_DIR:-$HOME/models/Qwen3.6-27B-Text-NVFP4-MTP}"
PROMPT_TOKENS="${1:-1024}"
MAX_NEW_TOKENS="${2:-128}"
MTP_DRAFTS="${3:-3}"
WSL_CUDA_LIB_DIR="${WSL_CUDA_LIB_DIR:-/usr/lib/wsl/lib}"

OUT_DIR="${ROOT_DIR}/target/profile"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_BASE="${OUT_DIR}/bench-p${PROMPT_TOKENS}-n${MAX_NEW_TOKENS}-mtp${MTP_DRAFTS}-${STAMP}"

export QWEN36_FP4_KERNEL_LIB_DIR="${ROOT_DIR}/target/cuda"
if [ -d "${WSL_CUDA_LIB_DIR}" ]; then
  export LD_LIBRARY_PATH="${WSL_CUDA_LIB_DIR}:${QWEN36_FP4_KERNEL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${QWEN36_FP4_KERNEL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

cargo build --release -p qwen36-fp4 --features cuda

nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --output="${OUT_BASE}" \
  -- "${ROOT_DIR}/target/release/qwen36" bench \
    --model-dir "${MODEL_DIR}" \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --mtp-speculative-tokens "${MTP_DRAFTS}"

echo "Profile saved to ${OUT_BASE}.nsys-rep"
