#!/usr/bin/env bash
# Quality smoke checks for TQ35 against BF16 KV.
#
# This compares full BF16 logits dumps for post-prefill and one decode step
# across a small prompt set. It reports cosine, absolute-error stats, top-k
# overlap, and greedy next-token equality.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${QWEN36_MODEL_DIR:-$HOME/models/Qwen3.6-27B-Text-NVFP4-MTP}"
TOP_K="${TOP_K:-20}"
DECODE_TOKEN_ID="${DECODE_TOKEN_ID:-11}"
MAX_CONTEXT="${MAX_CONTEXT:-256}"
STAMP="$(date +%Y%m%d-%H%M%S)"
QUALITY_OUT_DIR="${QUALITY_OUT_DIR:-target/quality-tq35-${STAMP}}"
WSL_CUDA_LIB_DIR="${WSL_CUDA_LIB_DIR:-/usr/lib/wsl/lib}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

mkdir -p "$QUALITY_OUT_DIR"
SUMMARY_JSONL="${QUALITY_OUT_DIR}/summary.jsonl"
: > "$SUMMARY_JSONL"

export QWEN36_FP4_KERNEL_LIB_DIR="${ROOT_DIR}/target/cuda"
if [ -d "${WSL_CUDA_LIB_DIR}" ]; then
  export LD_LIBRARY_PATH="${WSL_CUDA_LIB_DIR}:${QWEN36_FP4_KERNEL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${QWEN36_FP4_KERNEL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

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

compare_logits() {
  local label="$1"
  local mode="$2"
  local prompt="$3"
  local bf16_bin="${QUALITY_OUT_DIR}/${label}-${mode}-bf16.bin"
  local tq35_bin="${QUALITY_OUT_DIR}/${label}-${mode}-tq35.bin"

  wait_for_gpu
  if [ "$mode" = "prefill" ]; then
    QWEN36_KV_CACHE_DTYPE=bf16 target/release/qwen36 dump-logits \
      --model-dir "$MODEL_DIR" --prompt "$prompt" --top-k "$TOP_K" \
      --max-context "$MAX_CONTEXT" --out "$bf16_bin" \
      > "${bf16_bin%.bin}.txt"
    QWEN36_KV_CACHE_DTYPE=tq35 target/release/qwen36 dump-logits \
      --model-dir "$MODEL_DIR" --prompt "$prompt" --top-k "$TOP_K" \
      --max-context "$MAX_CONTEXT" --out "$tq35_bin" \
      > "${tq35_bin%.bin}.txt"
  else
    QWEN36_KV_CACHE_DTYPE=bf16 target/release/qwen36 dump-decode \
      --model-dir "$MODEL_DIR" --prompt "$prompt" --decode-token-id "$DECODE_TOKEN_ID" \
      --top-k "$TOP_K" --max-context "$MAX_CONTEXT" --out "$bf16_bin" \
      > "${bf16_bin%.bin}.txt"
    QWEN36_KV_CACHE_DTYPE=tq35 target/release/qwen36 dump-decode \
      --model-dir "$MODEL_DIR" --prompt "$prompt" --decode-token-id "$DECODE_TOKEN_ID" \
      --top-k "$TOP_K" --max-context "$MAX_CONTEXT" --out "$tq35_bin" \
      > "${tq35_bin%.bin}.txt"
  fi

  python3 - "$bf16_bin" "$tq35_bin" "$SUMMARY_JSONL" "$label" "$mode" "$prompt" "$TOP_K" <<'PY'
import json
import math
import struct
import sys
from pathlib import Path

bf_path, tq_path, summary_jsonl, label, mode, prompt, top_k_s = sys.argv[1:]
top_k = int(top_k_s)

def bf16_to_float(x: int) -> float:
    return struct.unpack("<f", struct.pack("<I", x << 16))[0]

def load(path: str) -> list[float]:
    data = Path(path).read_bytes()
    return [bf16_to_float(x[0]) for x in struct.iter_unpack("<H", data)]

bf = load(bf_path)
tq = load(tq_path)
if len(bf) != len(tq):
    raise SystemExit(f"length mismatch: {len(bf)} vs {len(tq)}")

dot = sum(a * b for a, b in zip(bf, tq))
nb = math.sqrt(sum(a * a for a in bf))
nt = math.sqrt(sum(b * b for b in tq))
diffs = [abs(a - b) for a, b in zip(bf, tq)]
rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
bf_top = sorted(range(len(bf)), key=lambda i: bf[i], reverse=True)[:top_k]
tq_top = sorted(range(len(tq)), key=lambda i: tq[i], reverse=True)[:top_k]
row = {
    "label": label,
    "mode": mode,
    "prompt": prompt,
    "top_k": top_k,
    "cosine": dot / (nb * nt),
    "max_abs_diff": max(diffs),
    "mean_abs_diff": sum(diffs) / len(diffs),
    "rmse": rmse,
    "bf16_argmax": bf_top[0],
    "tq35_argmax": tq_top[0],
    "argmax_match": bf_top[0] == tq_top[0],
    "top_k_overlap": len(set(bf_top) & set(tq_top)),
    "bf16_top": bf_top,
    "tq35_top": tq_top,
}
print(json.dumps(row, indent=2))
with open(summary_jsonl, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, separators=(",", ":")) + "\n")
PY
}

env -u OUT_DIR scripts/build_cuda.sh
env -u OUT_DIR cargo build --release -p qwen36-fp4 --features cuda

compare_logits short prefill "hello world from speedoza"
compare_logits short decode "hello world from speedoza"
compare_logits reasoning prefill "Explain why quantizing the KV cache can change long-context attention quality in one paragraph."
compare_logits reasoning decode "Explain why quantizing the KV cache can change long-context attention quality in one paragraph."

echo "Wrote ${QUALITY_OUT_DIR}"
echo "Summary: ${SUMMARY_JSONL}"
