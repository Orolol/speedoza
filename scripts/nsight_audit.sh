#!/usr/bin/env bash
# Nsight Compute bandwidth audit of the decode hot path (perf-roadmap P0).
#
# Measures % of peak DRAM bandwidth + duration per decode kernel at a given
# context. Output: a CSV at target/ncu-decode-<ctx>.csv + a digest table on
# stdout (paste into DAILY.md). This sizes the P3 persistent-pipeline lane —
# per the roadmap, the GEMV SMEM-paging prototype's kill-gate number comes
# from here.
#
# Usage: scripts/nsight_audit.sh [ctx_tokens]   (default 3072)
#
# Notes:
# - ncu serializes and replays kernels; tok/s during the run is meaningless.
# - launch-skip skips prefill + the first decode steps so we sample
#   steady-state decode launches (the captured graph's kernel nodes).
# - Needs GPU perf-counter permission; on locked-down hosts ncu fails with
#   ERR_NVGPUCTRPERM (nothing to do in-container — document and move on).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CTX="${1:-3072}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.6-27B-Text-NVFP4-MTP}"
BIN="$ROOT/target/release/qwen36"
OUT="target/ncu-decode-${CTX}.csv"

export QWEN36_FP4_KERNEL_LIB_DIR="$ROOT/target/cuda"
export LD_LIBRARY_PATH="$ROOT/target/cuda:${LD_LIBRARY_PATH:-}"

# All kernels that appear in a decode step; cuBLASLt lm_head GEMM kernels are
# matched by the broad 'gemm|matvec' alternation.
KREGEX='regex:nvfp4_gemv_mma|deltanet_decode|attention_decode|bf16_matvec|gemm|rmsnorm|swiglu|conv1d_update|sample_argmax|partial_rope|q_proj|gdn_gate|nvfp4_quantize'

ncu --target-processes all \
    --kernel-name "$KREGEX" \
    --launch-skip 600 --launch-count 200 \
    --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv --page raw \
    "$BIN" bench --model-dir "$MODEL_DIR" \
      --prompt-file benches/data/bench_corpus_91k.txt \
      --prompt-tokens "$CTX" --max-new-tokens 8 \
      --mtp-speculative-tokens 0 \
    > "$OUT" 2> >(grep -vE '^\s*$' | tail -5 >&2)

python3 - "$OUT" <<'EOF'
import csv, sys, collections
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("Kernel Name")]
agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
for r in rows:
    name = r["Kernel Name"].split("(")[0]
    def f(k):
        v = r.get(k, "")
        try: return float(v.replace(",", ""))
        except Exception: return 0.0
    a = agg[name]
    a[0] += 1
    a[1] += f("gpu__time_duration.sum")
    a[2] += f("dram__throughput.avg.pct_of_peak_sustained_elapsed")
    a[3] += f("dram__bytes.sum")
    a[4] += f("sm__throughput.avg.pct_of_peak_sustained_elapsed")
total_t = sum(a[1] for a in agg.values()) or 1.0
print("| kernel | n | Σt (µs) | %decode | DRAM %peak (avg) | SM %peak | ΣDRAM MB |")
print("|---|---:|---:|---:|---:|---:|---:|")
for name, a in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    n, t, bw, by, sm = a
    print(f"| {name[:46]} | {n} | {t:.0f} | {100*t/total_t:.1f}% | {bw/n:.1f}% | {sm/n:.1f}% | {by/1e6:.0f} |")
EOF
echo "raw: $OUT"
