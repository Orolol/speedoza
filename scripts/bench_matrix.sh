#!/usr/bin/env bash
# Capture a bench matrix across (prompt_tokens, mtp_drafts) combinations
# into a JSONL file for regression tracking.
#
# Usage:
#   scripts/bench_matrix.sh [OUTPUT_FILE]
# Default OUTPUT_FILE: target/bench-$(date +%Y%m%d-%H%M%S).jsonl
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${QWEN36_MODEL_DIR:-$HOME/models/Qwen3.6-27B-Text-NVFP4-MTP}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${1:-target/bench-${STAMP}.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

mkdir -p "$(dirname "$OUT")"

export QWEN36_FP4_KERNEL_LIB_DIR="${ROOT_DIR}/target/cuda"
export LD_LIBRARY_PATH="${QWEN36_FP4_KERNEL_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# Build once up-front so each iteration only runs the binary.
cargo build --release -p qwen36-fp4 --features cuda

BIN="${ROOT_DIR}/target/release/qwen36"

: > "$OUT"
for ctx in 256 1024 2048 4096; do
  for mtp in 0 1; do
    echo "=== ctx=${ctx} mtp=${mtp} ===" >&2
    "$BIN" bench \
      --model-dir "${MODEL_DIR}" \
      --prompt-tokens "${ctx}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --mtp-speculative-tokens "${mtp}" \
      | jq -c --argjson ctx "$ctx" --argjson mtp "$mtp" \
            '. + {ctx_label: $ctx, mtp_label: $mtp}' \
      >> "$OUT"
  done
done

echo "Wrote ${OUT}"
