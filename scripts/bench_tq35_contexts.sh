#!/usr/bin/env bash
# Bench TQ35 across short, long, and full-context prompts.
#
# Defaults:
#   - KV dtype: tq35
#   - MTP drafts: 3
#   - generated tokens per point: 1024
#   - prompt points: 1024, 32768, and FULL_CONTEXT_TOKENS - MAX_NEW_TOKENS
#
# Environment overrides:
#   QWEN36_MODEL_DIR
#   KV_DTYPE=tq35|tq3|fp8|bf16
#   MTP_DRAFTS=3
#   MAX_NEW_TOKENS=1024
#   FULL_CONTEXT_TOKENS=262144
#   PROMPT_TOKENS_LIST="1024 32768 261120"
#   BENCH_OUT_DIR=target/bench-tq35-contexts-...
#   GPU_POLL_SECONDS=60
#   DRY_RUN=1
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${QWEN36_MODEL_DIR:-$HOME/models/Qwen3.6-27B-Text-NVFP4-MTP}"
KV_DTYPE="${KV_DTYPE:-tq35}"
MTP_DRAFTS="${MTP_DRAFTS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
FULL_CONTEXT_TOKENS="${FULL_CONTEXT_TOKENS:-262144}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BENCH_OUT_DIR="${BENCH_OUT_DIR:-target/bench-tq35-contexts-${STAMP}}"
WSL_CUDA_LIB_DIR="${WSL_CUDA_LIB_DIR:-/usr/lib/wsl/lib}"

if (( FULL_CONTEXT_TOKENS <= MAX_NEW_TOKENS )); then
  echo "FULL_CONTEXT_TOKENS must be greater than MAX_NEW_TOKENS" >&2
  exit 2
fi

FULL_PROMPT_TOKENS=$((FULL_CONTEXT_TOKENS - MAX_NEW_TOKENS))
PROMPT_TOKENS_LIST="${PROMPT_TOKENS_LIST:-1024 32768 ${FULL_PROMPT_TOKENS}}"

mkdir -p "$BENCH_OUT_DIR"
SUMMARY_JSONL="${BENCH_OUT_DIR}/summary.jsonl"
: > "$SUMMARY_JSONL"

export QWEN36_FP4_KERNEL_LIB_DIR="${ROOT_DIR}/target/cuda"
if [ -d "${WSL_CUDA_LIB_DIR}" ]; then
  export LD_LIBRARY_PATH="${WSL_CUDA_LIB_DIR}:${QWEN36_FP4_KERNEL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${QWEN36_FP4_KERNEL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
export QWEN36_KV_CACHE_DTYPE="${KV_DTYPE}"

gpu_busy() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null | awk 'NF { found=1 } END { exit found ? 0 : 1 }'
}

wait_for_gpu() {
  while gpu_busy; do
    echo "GPU busy; waiting ${GPU_POLL_SECONDS}s..." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits >&2 || true
    sleep "$GPU_POLL_SECONDS"
  done
}

run_point() {
  local prompt_tokens="$1"
  local label="$2"
  local out_json="${BENCH_OUT_DIR}/${label}.json"
  echo "=== ${label}: prompt=${prompt_tokens}, gen=${MAX_NEW_TOKENS}, mtp=${MTP_DRAFTS}, kv=${KV_DTYPE} ===" >&2
  wait_for_gpu
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN: target/release/qwen36 bench --model-dir ${MODEL_DIR} --prompt-tokens ${prompt_tokens} --max-new-tokens ${MAX_NEW_TOKENS} --mtp-speculative-tokens ${MTP_DRAFTS}" >&2
    return
  fi
  target/release/qwen36 bench \
    --model-dir "${MODEL_DIR}" \
    --prompt-tokens "${prompt_tokens}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --mtp-speculative-tokens "${MTP_DRAFTS}" \
    | tee "${out_json}"
  python3 - "$out_json" "$SUMMARY_JSONL" "$label" "$KV_DTYPE" "$FULL_CONTEXT_TOKENS" <<'PY'
import json
import sys

out_json, summary_jsonl, label, kv_dtype, full_context_tokens = sys.argv[1:]
with open(out_json, "r", encoding="utf-8") as f:
    data = json.load(f)
data["label"] = label
data["kv_cache_dtype"] = kv_dtype
data["configured_full_context_tokens"] = int(full_context_tokens)
with open(summary_jsonl, "a", encoding="utf-8") as f:
    f.write(json.dumps(data, separators=(",", ":")) + "\n")
PY
}

if [ "${DRY_RUN:-0}" != "1" ]; then
  env -u OUT_DIR scripts/build_cuda.sh
  env -u OUT_DIR cargo build --release -p qwen36-fp4 --features cuda
fi

idx=0
for prompt_tokens in ${PROMPT_TOKENS_LIST}; do
  if (( prompt_tokens == FULL_PROMPT_TOKENS )); then
    label="fullctx-p${prompt_tokens}-n${MAX_NEW_TOKENS}-mtp${MTP_DRAFTS}-${KV_DTYPE}"
  else
    label="p${prompt_tokens}-n${MAX_NEW_TOKENS}-mtp${MTP_DRAFTS}-${KV_DTYPE}"
  fi
  run_point "$prompt_tokens" "$label"
  idx=$((idx + 1))
done

echo "Wrote ${BENCH_OUT_DIR}"
echo "Summary: ${SUMMARY_JSONL}"
