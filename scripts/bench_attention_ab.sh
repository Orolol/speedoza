#!/usr/bin/env bash
# A/B bench the three attention prefill paths against each other:
#   scalar  — QWEN36_ATTENTION_FLASH_PREFILL=0 + QWEN36_ATTENTION_SAGE_PREFILL=0
#   flash   — Phase A wmma BF16 (sage off)
#   sage    — Phase B INT8 Q·K (flash on as fallback, sage on by default)
#
# Long-context mode is on by default so attention isn't masked by FFN.
# Output: a one-line-per-(ctx, mode) summary plus prefill_tokens_per_second
# and decode_tokens_per_second pulled from the JSON the binary emits.
#
# Usage:
#   scripts/bench_attention_ab.sh [CTX_LIST...]
# Default CTX_LIST: 1024 2048 4096 8192
set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${QWEN36_MODEL_DIR:-$HOME/models/Qwen3.6-27B-Text-NVFP4-MTP}"
MTP="${MTP:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
LONG_CTX="${QWEN36_LONG_CONTEXT_MODE:-1}"
CTX_LIST=("$@")
if [ "${#CTX_LIST[@]}" -eq 0 ]; then
  CTX_LIST=(1024 2048 4096 8192)
fi

export QWEN36_FP4_KERNEL_LIB_DIR="${ROOT_DIR}/target/cuda"
export LD_LIBRARY_PATH="${QWEN36_FP4_KERNEL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export QWEN36_LONG_CONTEXT_MODE="$LONG_CTX"

cargo build --release -p qwen36-fp4 --features cuda >&2

BIN="${ROOT_DIR}/target/release/qwen36"

set_env_for() {
  case "$1" in
    scalar) export QWEN36_ATTENTION_FLASH_PREFILL=0 QWEN36_ATTENTION_SAGE_PREFILL=0 ;;
    flash)  unset QWEN36_ATTENTION_FLASH_PREFILL; export QWEN36_ATTENTION_SAGE_PREFILL=0 ;;
    sage)   unset QWEN36_ATTENTION_FLASH_PREFILL QWEN36_ATTENTION_SAGE_PREFILL ;;
    *) echo "unknown mode $1" >&2; exit 1 ;;
  esac
}

printf "%-5s %-7s %12s %12s\n" ctx mode prefill_t/s decode_t/s
printf "%-5s %-7s %12s %12s\n" ----- ------- ------------ ------------
for ctx in "${CTX_LIST[@]}"; do
  for mode in scalar flash sage; do
    set_env_for "$mode"
    out=$("$BIN" bench --model-dir "$MODEL_DIR" --prompt-tokens "$ctx" \
      --max-new-tokens "$MAX_NEW_TOKENS" --mtp-speculative-tokens "$MTP" 2>/dev/null || true)
    pf=$(echo "$out" | jq -r '.prefill_tokens_per_second // "FAIL"')
    dec=$(echo "$out" | jq -r '.decode_tokens_per_second // "n/a"')
    printf "%-5d %-7s %12s %12s\n" "$ctx" "$mode" "$pf" "$dec"
  done
done
