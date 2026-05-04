# Decode parity harness — `QWEN36_DECODE_GEMV` passthrough

**Date:** 2026-05-04
**Plan:** `docs/superpowers/plans/2026-05-04-direction-b-nvfp4-gemv.md`, Task 10
**Outcome:** Case B — no code change needed.

## Investigation summary

`scripts/decode_parity.py` is a **pure reference checker**, not an engine
launcher. End-to-end review:

- It reads engine-produced BF16 dumps from `QWEN36_PARITY_DUMP`
  (default `/tmp/qwen36_decode_dump`) and diffs them against a PyTorch
  reference computed in-process.
- It never spawns the engine: no `subprocess`, no `Popen`, no `env=` kwarg
  anywhere in the file.
- The only `os.environ` accesses (lines 14–19, 589, 621, 715) read
  parity-side knobs (`QWEN36_PARITY_*`, `QWEN36_DECODE_LOCAL`,
  `QWEN36_DECODE_STOP_ON_FAIL`, `QWEN36_DECODE_COS_FLOOR`). None are
  forwarded to a child process.
- There is no existing analogue for `QWEN36_USE_MEGAKERNEL_GEMM` inside
  the harness — `grep` for that string returns zero hits in
  `scripts/decode_parity.py`. The megakernel toggle is documented in
  `AGENT.md` (line 278) and `docs/mirage-megakernel.md` (line 125) as a
  flag the user sets on the **engine** command line
  (`./target/release/qwen36 dump-decode ...`), not on the Python harness.

The dump-producing step lives in the Rust binary (`qwen36 dump-decode`),
documented in `AGENT.md` lines 378–397. Any kernel-selection env var
(`QWEN36_DECODE_GEMV=1`, `QWEN36_USE_MEGAKERNEL_GEMM=1`, ...) must be
exported in the shell that invokes that binary; the Python checker then
runs against the resulting dump directory unchanged.

**Conclusion:** this is Case B from the plan. The harness inherits no
engine env vars because it does not launch the engine — instead the
*engine* inherits its env from the parent shell, and the harness only
reads the dumps. No edits to `scripts/decode_parity.py` are required for
the gemv parity flow.

## Exact invocation for a gemv parity check

Two-step flow, mirroring the existing megakernel pattern in `AGENT.md`:

```bash
# 1) Produce dumps with the gemv kernel forced on.
rm -rf /tmp/qwen36_decode_dump_gemv
mkdir -p /tmp/qwen36_decode_dump_gemv
QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda" \
LD_LIBRARY_PATH="$PWD/target/cuda:${LD_LIBRARY_PATH:-}" \
QWEN36_DEBUG_DUMP_DIR=/tmp/qwen36_decode_dump_gemv \
QWEN36_DEBUG_DUMP_DECODE=1 \
QWEN36_DEBUG_DUMP_ALL_LAYERS=1 \
QWEN36_DECODE_GEMV=1 \
  ./target/release/qwen36 dump-decode \
    --model-dir ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --prompt hello --decode-token-id 11 --top-k 5 \
    --out /tmp/qwen36_decode_logits_gemv.bf16

# 2) Diff each layer dump against the PyTorch NVFP4 reference.
QWEN36_PARITY_DUMP=/tmp/qwen36_decode_dump_gemv \
QWEN36_PARITY_PROMPT=hello \
QWEN36_PARITY_DECODE_TOKEN=11 \
QWEN36_DECODE_LOCAL=1 \
  python3 -u scripts/decode_parity.py
```

For an A/B comparison against the cuBLASLt baseline, repeat step 1 with
`QWEN36_DECODE_GEMV=0` (or unset) into a sibling directory
(`/tmp/qwen36_decode_dump_baseline`), then either:

- Run step 2 a second time pointing at the baseline dir, and compare
  the two `cos=...` series side-by-side, or
- Diff the raw BF16 buffers directly with `cmp -l` / a small numpy
  script — the reference computation in step 2 is identical for both
  runs, so the only thing changing is the engine kernel selection.

## Why no `--decode-gemv` CLI flag was added

Adding `--decode-gemv` to `scripts/decode_parity.py` would do nothing
useful: the harness has no child process to forward the env var to. The
flag would set `os.environ["QWEN36_DECODE_GEMV"] = "1"` only inside the
Python interpreter, which is never read by the Rust engine (already
finished by the time the Python script runs). Mirroring the existing
megakernel pattern exactly therefore means **leaving the harness
untouched** and exporting the env var on the `dump-decode` command line.
