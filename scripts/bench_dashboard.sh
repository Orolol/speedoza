#!/usr/bin/env bash
# THE bench dashboard (perf-roadmap P0): one fixed grid, one markdown table.
#
# Every perf item in docs/perf-roadmap.md cites before/after numbers from THIS
# script and nothing else. Grid:
#   - MTP {0,4} x ctx {128, 3072, 8192, 24576}  (bench, real-text corpus)
#   - DFlash 2 cells (3K + 7K frozen snapshot prompts, drafter-chat-smoke)
#
# Prompts come from frozen files under benches/data/ — never from live docs.
# Output: a markdown table on stdout, ready to paste into DAILY.md, plus the
# raw JSONL at target/dashboard-<stamp>.jsonl.
#
# Usage:
#   scripts/bench_dashboard.sh            # full grid (~10 model loads)
#   scripts/bench_dashboard.sh --quick    # MTP {0,4} x ctx {128, 3072} only
#
# Env: MODEL_DIR, DRAFTER_DIR override model paths. DFlash cells are skipped
# with a warning when DRAFTER_DIR is absent.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.6-27B-Text-NVFP4-MTP}"
DRAFTER_DIR="${DRAFTER_DIR:-$HOME/models/Qwen3.6-27B-DFlash}"
BIN="$ROOT/target/release/qwen36"
CORPUS="benches/data/bench_corpus_91k.txt"
SNAPSHOT="benches/data/agent_md_snapshot_2026-06-09.txt"
STAMP="$(date +%Y%m%d-%H%M%S)"
JSONL="target/dashboard-${STAMP}.jsonl"
MAX_NEW="${MAX_NEW_TOKENS:-128}"

export QWEN36_FP4_KERNEL_LIB_DIR="$ROOT/target/cuda"
# Dashboard cells measure the PURE MTP path (kernel-work before/afters);
# the production default since 2026-06-11 is the online auto-fallback
# (QWEN36_MTP_AUTO_FALLBACK, drops to plain decode below 0.55 acceptance).
export QWEN36_MTP_AUTO_FALLBACK=0
export LD_LIBRARY_PATH="$ROOT/target/cuda:${LD_LIBRARY_PATH:-}"

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

[ -x "$BIN" ] || { echo "missing $BIN — build first" >&2; exit 1; }
[ -f "$CORPUS" ] || { echo "missing $CORPUS" >&2; exit 1; }

# --- GPU contention check ---
read -r used free util < <(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits | tr ',' ' ')
echo "GPU: used=${used}MiB free=${free}MiB util=${util}% | commit $(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty')"
if [ "${free:-0}" -lt 20000 ]; then
  echo "WARN: <20GB free — another process may pollute the numbers." >&2
fi

mkdir -p target
: > "$JSONL"

jnum() { grep -oE "\"$1\": ?[0-9.]+" | grep -oE '[0-9.]+' | head -1; }

CTXS="128 3072 8192 24576"
[ "$QUICK" = "1" ] && CTXS="128 3072"

echo
echo "| cell | prefill tok/s | decode tok/s | AL / acc |"
echo "|---|---:|---:|---:|"

for ctx in $CTXS; do
  for mtp in 0 4; do
    out=$("$BIN" bench --model-dir "$MODEL_DIR" --prompt-file "$CORPUS" \
      --prompt-tokens "$ctx" --max-new-tokens "$MAX_NEW" \
      --mtp-speculative-tokens "$mtp" 2>/dev/null) || { echo "| MTP=$mtp ctx=$ctx | ERROR | ERROR | |"; continue; }
    # bench prints pretty-printed JSON (multi-line); compact it for the JSONL.
    echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); d['cell']='mtp${mtp}_ctx${ctx}'; print(json.dumps(d))" >> "$JSONL" 2>/dev/null || true
    # MTP cells: accepted/proposed draft acceptance (per-position detail in the JSONL).
    acc=""
    if [ "$mtp" != "0" ]; then
      acc="$(echo "$out" | jnum mtp_draft_acceptance_rate)"
    fi
    printf "| MTP=%s ctx=%-5s | %s | %s | %s |\n" "$mtp" "$ctx" \
      "$(echo "$out" | jnum prefill_tokens_per_second)" \
      "$(echo "$out" | jnum decode_tokens_per_second)" \
      "$acc"
  done
done

if [ "$QUICK" = "0" ]; then
  if [ -f "$DRAFTER_DIR/config.json" ]; then
    P3K=$(mktemp); P7K=$(mktemp)
    trap 'rm -f "$P3K" "$P7K"' EXIT
    head -150 "$SNAPSHOT" > "$P3K"
    head -300 "$SNAPSHOT" > "$P7K"
    for cell in "dflash_3k:$P3K" "dflash_7k:$P7K"; do
      name="${cell%%:*}"; pf="${cell##*:}"
      out=$(QWEN36_LONG_CONTEXT_MODE=1 "$BIN" drafter-chat-smoke \
        --model-dir "$MODEL_DIR" --drafter-dir "$DRAFTER_DIR" \
        --prompt "$(cat "$pf")" --max-new-tokens 160 2>&1) || { echo "| $name | ERROR | ERROR | |"; continue; }
      echo "{\"cell\":\"$name\",\"tokens_per_second\":$(echo "$out" | jnum tokens_per_second),\"acceptance_length\":$(echo "$out" | jnum acceptance_length)}" >> "$JSONL"
      printf "| %s (ctx=%s) | | %s | %s |\n" "$name" \
        "$(echo "$out" | jnum prompt_tokens)" \
        "$(echo "$out" | jnum tokens_per_second)" \
        "$(echo "$out" | jnum acceptance_length)"
    done
  else
    echo "| dflash_3k / dflash_7k | SKIPPED (no drafter at $DRAFTER_DIR) | | |"
  fi
fi

echo
echo "raw: $JSONL"
