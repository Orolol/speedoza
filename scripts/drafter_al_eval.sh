#!/usr/bin/env bash
# DFlash long-context AL evaluation battery.
#
# Single-prompt AL measurements at long context are NOISE: identical
# content reordered swings AL between 2.3 and 6.8 (measured 2026-06-09).
# Any drafter-quality knob (window size, capture layers, block size,
# fine-tune) must be evaluated against this battery and judged on the
# geomean, never on one prompt.
#
# Usage:
#   scripts/drafter_al_eval.sh                      # baseline only
#   scripts/drafter_al_eval.sh "ENV1=V1 ENV2=V2"    # one config (env string)
#
# Prints per-prompt tok/s + AL and the geomean AL for the config.
# GPU-serial; ~5-8 min per config at max-new=160.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.6-27B-Text-NVFP4-MTP}"
DRAFTER_DIR="${DRAFTER_DIR:-$HOME/models/Qwen3.6-27B-DFlash}"
BIN="$ROOT/target/release/qwen36"
export QWEN36_FP4_KERNEL_LIB_DIR="$ROOT/target/cuda"
export LD_LIBRARY_PATH="$ROOT/target/cuda:${LD_LIBRARY_PATH:-}"
export QWEN36_LONG_CONTEXT_MODE=1

CFG_ENV="${1:-}"

# --- battery: 6 long (~6-8K token) prompts from real repo text, with
# deliberate content-order variation to expose the chaotic sensitivity ---
TMPD=$(mktemp -d); trap 'rm -rf "$TMPD"' EXIT
cat doc.md docs/development.md docs/research.md docs/roadmap.md \
    docs/kernel-validation.md docs/troubleshooting.md > "$TMPD/p1_docs_a.txt"
cat docs/troubleshooting.md docs/kernel-validation.md docs/roadmap.md \
    docs/research.md docs/development.md doc.md > "$TMPD/p2_docs_b.txt"
# Frozen snapshot (NOT the live AGENT.md): the AL geomean baseline (5.10)
# was measured on the 2026-06-09 text; AGENT.md was split on 2026-06-10.
SNAPSHOT=benches/data/agent_md_snapshot_2026-06-09.txt
head -300 "$SNAPSHOT" > "$TMPD/p3_agent7k.txt"
{ head -250 "$SNAPSHOT"; cat doc.md; } > "$TMPD/p4_agent_doc.txt"
{ cat doc.md; head -250 "$SNAPSHOT"; } > "$TMPD/p5_doc_agent.txt"
cat docs/research.md docs/roadmap.md doc.md docs/development.md \
    docs/troubleshooting.md docs/kernel-validation.md > "$TMPD/p6_docs_c.txt"

jnum() { grep -oE "\"$1\": [0-9.]+" | grep -oE '[0-9.]+' | head -1; }

echo "config: ${CFG_ENV:-<baseline>}"
echo
als=""
for p in "$TMPD"/p*.txt; do
  name=$(basename "$p" .txt)
  out=$(env $CFG_ENV "$BIN" drafter-chat-smoke --model-dir "$MODEL_DIR" \
    --drafter-dir "$DRAFTER_DIR" --prompt "$(cat "$p")" \
    --max-new-tokens 160 2>&1) || { echo "  $name: ERROR"; continue; }
  tps=$(echo "$out" | jnum tokens_per_second)
  al=$(echo "$out" | jnum acceptance_length)
  ctx=$(echo "$out" | jnum prompt_tokens)
  echo "  $name ctx=$ctx tok/s=$tps AL=$al"
  als="$als $al"
done
echo
python3 -c "
import math, sys
vals = [float(x) for x in '''$als'''.split()]
if vals:
    geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
    print(f'geomean AL = {geo:.3f}  (n={len(vals)}, min={min(vals):.2f}, max={max(vals):.2f})')
"
