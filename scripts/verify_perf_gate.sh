#!/usr/bin/env bash
# DFlash verify + MTP perf/regression gate.
#
# Shared guardrail for the two parallel megakernel lanes (split-K verify /
# Claude, interpreter decode / Codex): run before and after a change to
# confirm neither the DFlash split-K path nor the MTP graph path regressed.
# Prints a table of decode tok/s (+ AL for DFlash). GPU is serial — this
# runs the cells sequentially; check the GPU is free first (see
# CLAUDE.md GPU-sharing rule).
#
# Usage:
#   scripts/verify_perf_gate.sh                 # full gate
#   scripts/verify_perf_gate.sh --quick         # 3K DFlash + MTP=4 only
#
# Env: MODEL_DIR, DRAFTER_DIR override the default model paths.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.6-27B-Text-NVFP4-MTP}"
DRAFTER_DIR="${DRAFTER_DIR:-$HOME/models/Qwen3.6-27B-DFlash}"
BIN="$ROOT/target/release/qwen36"
export QWEN36_FP4_KERNEL_LIB_DIR="$ROOT/target/cuda"
export LD_LIBRARY_PATH="$ROOT/target/cuda:${LD_LIBRARY_PATH:-}"
export QWEN36_LONG_CONTEXT_MODE=1

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

# --- GPU availability check (CLAUDE.md rule) ---
read -r used free util < <(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits | tr ',' ' ')
echo "GPU: used=${used}MiB free=${free}MiB util=${util}%"
if [ "${free:-0}" -lt 20000 ]; then
  echo "WARN: <20GB free — another process may be using the GPU. Proceed with care." >&2
fi
echo

# --- prompt fixtures ---
# Prompts come from a frozen snapshot (benches/data/), NOT the live AGENT.md:
# the recorded gate baselines were measured on the 2026-06-09 text, and
# AGENT.md was split into instructions + DAILY.md on 2026-06-10. Editing docs
# must never silently change the gate prompts.
SNAPSHOT=benches/data/agent_md_snapshot_2026-06-09.txt
PROMPT_3K=$(mktemp); PROMPT_5K=$(mktemp); PROMPT_7K=$(mktemp)
trap 'rm -f "$PROMPT_3K" "$PROMPT_5K" "$PROMPT_7K"' EXIT
head -150 "$SNAPSHOT" > "$PROMPT_3K"
cat doc.md docs/development.md docs/research.md > "$PROMPT_5K" 2>/dev/null || cp "$PROMPT_3K" "$PROMPT_5K"
head -300 "$SNAPSHOT" > "$PROMPT_7K"

jnum() { grep -oE "\"$1\": [0-9.]+" | grep -oE '[0-9.]+' | head -1; }

dflash_cell() { # $1=prompt $2=label $3=env
  local out
  out=$(env $3 "$BIN" drafter-chat-smoke --model-dir "$MODEL_DIR" \
    --drafter-dir "$DRAFTER_DIR" --prompt "$(cat "$1")" --max-new-tokens 160 2>&1) || {
      echo "  $2: ERROR"; return; }
  printf "  %-34s tok/s=%-8s AL=%-6s ctx=%s\n" "$2" \
    "$(echo "$out" | jnum tokens_per_second)" \
    "$(echo "$out" | jnum acceptance_length)" \
    "$(echo "$out" | jnum prompt_tokens)"
}

mtp_cell() { # $1=mtp $2=label $3=env
  local out
  out=$(env ${3:-} "$BIN" bench --model-dir "$MODEL_DIR" --prompt-tokens 128 \
    --max-new-tokens 32 --mtp-speculative-tokens "$1" 2>&1) || {
      echo "  $2: ERROR"; return; }
  printf "  %-34s decode=%s tok/s\n" "$2" \
    "$(echo "$out" | jnum decode_tokens_per_second)"
}

echo "=== DFlash verify (split-K default vs forced-off) ==="
dflash_cell "$PROMPT_3K" "DFlash 3K (split-K default)" ""
dflash_cell "$PROMPT_3K" "DFlash 3K (split-K OFF)" "QWEN36_VERIFY_FLASH_SPLITK=0"
if [ "$QUICK" = "0" ]; then
  dflash_cell "$PROMPT_5K" "DFlash 5.5K (split-K default)" ""
  dflash_cell "$PROMPT_5K" "DFlash 5.5K (split-K OFF)" "QWEN36_VERIFY_FLASH_SPLITK=0"
  dflash_cell "$PROMPT_7K" "DFlash 7K (split-K default)" ""
  dflash_cell "$PROMPT_7K" "DFlash 7K (split-K OFF)" "QWEN36_VERIFY_FLASH_SPLITK=0"
fi
echo
echo "=== MTP graph path (must not regress / no capture error) ==="
mtp_cell 0 "MTP=0 (auto interpreter off)" ""
mtp_cell 4 "MTP=4 (auto interpreter)" ""
mtp_cell 4 "MTP=4 (interpreter OFF)" "QWEN36_INTERPRETER_DECODE=0"
echo
echo "Gate done. Compare against the DAILY.md baselines before merging."
