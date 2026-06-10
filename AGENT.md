# AGENT.md

This file provides guidance to AI coding agents (Claude Code, Codex, Cursor, etc.) when working with code in this repository.

## Project

Single-stream inference engine for `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`, targeting **RTX 5090 / Blackwell SM120 only**. Rust 1.85 (edition 2024) Cargo workspace + a CUDA shared library built out-of-band by shell scripts. The repo is performance- and hardware-specific; **prefer explicit failure over silent fallback**.

Full design intent: `doc.md`. Operational docs: `docs/`.

The documentation contract, in reading order:

1. **`docs/code-inventory.md`** — the current state: every component (active / opt-in /
   archived-negative / dead), the default dispatch paths, every `QWEN36_*` env var.
   **Read it before building anything new**; update it in the same commit when you
   change a default, a dispatch condition, or add/remove a flag.
2. **`DAILY.md`** — the chronological lab journal (dated experiment write-ups, bench
   numbers, verdicts). **Append an entry every working session**; never delete entries.
3. **This file (AGENT.md)** — instructions only: build loops, rules, contracts,
   guardrails. No bench tables or dated narratives belong here.
4. `doc.md` — the original design spec (design intent).

Several past sessions rebuilt code that already existed or re-ran experiments that had
already failed — the inventory and the journal exist to prevent exactly that.

## Workspace map

- `crates/core` — topology, dtype, tensor classification, memory budgets
- `crates/loader` — `config.json` + safetensors mmap; emits `model_layout.json`
- `crates/tokenizer` — HF tokenizer wrapper + Qwen chat rendering
- `crates/kernels` — Rust kernel specs, CUDA FFI, `NoCudaBackend` / `CudaBackend`
- `crates/runtime` — KV cache + DeltaNet state planning, GPU weight upload, engine shell
- `crates/mtp` — speculative decode controller (snapshot / restore / replay)
- `crates/cli` — `qwen36-fp4` binary: `discover`, `inspect-config`, `budget`, `tokenize`, `validate-weights`, `gpu-load`
- `kernels-cuda/` — CUDA sources and the public C ABI in `include/qwen36_fp4.h`

## Build and test

CPU-only loop (always works, no GPU required):

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

CUDA loop (run before any change touching kernels, ABI, or runtime GPU paths):

```bash
./scripts/build_cuda.sh
./scripts/smoke_cuda.sh
export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$QWEN36_FP4_KERNEL_LIB_DIR:${LD_LIBRARY_PATH:-}"
cargo test  --workspace --features qwen36-fp4-kernels/cuda
cargo clippy --workspace --features qwen36-fp4-kernels/cuda -- -D warnings
```

Single test: `cargo test -p <crate> <name>` (e.g. `cargo test -p qwen36-fp4-mtp rollback`).
End-to-end smoke against a real checkpoint: `cargo run -p qwen36-fp4 --features cuda -- gpu-load --model-dir <path> --max-context 2256`.

Bench against a real-text prompt (preferred for MTP-acceptance gates): `bench --prompt-file benches/data/long_prompt_4k.txt --prompt-tokens 4096 --max-new-tokens 64 --mtp-speculative-tokens 4`. The synthetic single-token-repeat default produces adversarial MTP acceptance — see `DAILY.md` § 2026-05-15 — Anomaly diagnostics.
Measure MTP acceptance on real text: `chat --prompt "$(cat real_prompt.txt)" --max-new-tokens 64 --mtp-speculative-tokens 4` with `QWEN36_MTP_STATS=1` — prints `mtp.stats accepted=… acceptance_rate=…`.

## Build environment

- `QWEN36_FP4_CUDA_MIN_VERSION=13.0` and `QWEN36_FP4_SM=120` are set in `.cargo/config.toml`.
- CUDA 13.0+ required, but **avoid CUDA 13.2** (known Blackwell bugs).
- Kernel build links `-lcublasLt` and outputs `target/cuda/libqwen36_fp4_kernels.so`.
- Build overrides: `CUDA_HOME`, `NVCC`, `OUT_DIR`, `QWEN36_FP4_SM`.

## Non-obvious rules (read before editing)

**ABI sync.** Any change to `kernels-cuda/include/qwen36_fp4.h` MUST be mirrored in:
- `crates/kernels/src/backend.rs`
- the relevant typed spec module under `crates/kernels/src/`
- `kernels-cuda/smoke.cu` if the new field is required

While the ABI is still evolving, append new fields at the end of structs.

**Hybrid TurboQuant policy.** The model has 64 layers but only 16 are full attention. The vLLM heuristic ("skip first and last 2 layers") is wrong here. **Skip the first and last *full-attention* layers** — global indices `{3, 63}`. DeltaNet layers are not affected.

**Model topology.** Qwen3.6-27B is hybrid: 48 Gated DeltaNet layers (linear attention, conv1d, bf16 recurrent state, no KV cache) + 16 full Gated Attention layers (GQA, partial RoPE on 64 of 256 dims, FP8/FP4 KV via TurboQuant) + 1 MTP head. Only full-attention layers carry a KV cache.

**No fake inference paths.** Never return generated tokens from an incomplete path. Surface `UnsupportedNoCuda` (or an explicit error) instead of mock output.

**Checkpoint assumptions live in `qwen36 discover` output**, not in code comments. `model_layout.json` is the first debugging artifact for any model mismatch.

## Kernel development loop

1. Edit the C spec in `kernels-cuda/include/qwen36_fp4.h`.
2. Mirror it in `crates/kernels/src/backend.rs` and the typed Rust spec.
3. Implement the CUDA kernel under `kernels-cuda/`.
4. Add coverage in `kernels-cuda/smoke.cu`.
5. Compare against a CPU / PyTorch reference **before** optimizing.
6. Profile with Nsight Compute on the 5090.

## Numerical parity status

The engine is now **internally numerically validated** against a PyTorch decomposition of the shipped `modelopt` NVFP4 checkpoint for both prefill and one-token decode local boundaries. This covers:

- single-token prompt `"hello"`
- two-token prompt `"hello world"`
- all 64 layer local boundaries
- all full-attention layers
- multi-token DeltaNet recurrence within prefill
- one-token decode after prefill with carried KV/DeltaNet/conv state
- final RMSNorm + `lm_head`

The user-visible generation path no longer produces the earlier blank/`+` nonsense after the `q_proj` layout and decode fused-quantization fixes. Greedy chat still emits the model's thinking text and can truncate/repeat if `max_new_tokens` is too short; that is sampling/stop-policy behavior, not the previous numerical corruption.

Remaining gap:

- **External reference logits** from official Transformers/vLLM/modelopt are still not proven end-to-end. The local PyTorch decomposition now follows the official architecture semantics we can inspect, but do not claim external parity until a separate implementation's logits are compared.

## Key runtime contracts (do not break)

Distilled from hard-won debugging sessions (full stories in `DAILY.md`):

- **Hand-rolled GEMM/GEMV alpha contract.** Kernels must NOT apply `a_scale_2` /
  `b_scale_2` (the per-tensor scales): the runtime pre-folds them into `spec->alpha`,
  mirroring the cuBLASLt contract (`kernels-cuda/nvfp4_gemm.cu` only passes `alpha` and
  never dereferences the per-tensor scales). A kernel that multiplies them again on top
  of `alpha` produces gibberish on real weights despite uniform-data smoke passing —
  this shipped once and was caught only by end-to-end chat parity.
- **MTP parity floor.** `chat --prompt "hello" / "hello world" --max-new-tokens 12` must
  produce identical token streams for `--mtp-speculative-tokens` ∈ {0..4}. These are the
  gated prompts; do not weaken them. (MTP≥1 chunked verify is known not bit-equal to
  MTP=0 on borderline-argmax prompts — that documented divergence is the accepted
  baseline, not licence for new ones.)
- **MTP graph rule.** MTP=2/3 full-accept verification graph-captures the main verify
  chunk, per-draft greedy samples, and the next current-token sample; next-draft
  generation intentionally stays on the validated host launch path. Do not move
  recursive MTP draft generation into the graph unless the parity floor above still
  passes exactly afterwards.
- **Stream/graph invariant.** When a captured graph leaves a non-default active stream
  installed, later host-launched kernels also run on that stream. Any host read of
  `token_u32` / `sampled_token_u32` must synchronize the active stream before the D2H
  copy or the host reads stale tokens and drifts. `Engine::read_current_token` and
  `Engine::read_sampled_token` enforce this — keep it that way.
- **Attention dispatch lives in CUDA.** Which attention kernel runs is decided inside
  `kernels-cuda/attention.cu` entry points (sage → flash → split-K verify → scalar GQA;
  tiled vs v1 split decode), not in Rust. Grep there first.

## Process guardrails (staying on the rails)

The project goal is a **correct, fast-enough, maintainable single-model local engine**
for exactly one checkpoint on one RTX 5090 — not a kernel research playground. Rules:

1. **Correctness outranks throughput.** A known correctness bug (e.g. the
   decode-vs-prefill logits divergence, `qwen36 decode-vs-prefill-check`) takes priority
   over any perf work. Never trade parity for tok/s.
2. **Measure before building.** Any optimization idea gets a cheap falsification probe
   or paired microbench FIRST, with an explicit kill-gate ("≥ +N tok/s on path X or
   revert") written down in `DAILY.md` before the work starts. This repo's history:
   five model-based hypotheses falsified by cheap probes in one sprint, three
   megakernel variants built before measuring — the probes were always right.
3. **The 15% bar.** Do not start kernel work projected below ~15% end-to-end gain on a
   production path without explicit user sign-off. Diminishing-returns work (the
   MTP=0 decode is within ~15% of its context-flat ceiling) needs a stated reason.
4. **Perf gates around every perf change.** `scripts/verify_perf_gate.sh` before/after;
   op-level parity (cos ≥ 0.998) at the affected op; the MTP parity floor; smoke suite.
   Bench MTP acceptance only with real text (`--prompt-file` or `chat` +
   `QWEN36_MTP_STATS=1`); drafter-quality changes only via the geomean battery
   (`scripts/drafter_al_eval.sh`) — single-prompt AL deltas are noise.
5. **Failed experiments get DELETED, not archived in tree.** Git history is the
   archive. Write the negative result up in `DAILY.md` (numbers, root cause, what
   would have to change to retry), record it in `docs/code-inventory.md` §2.5, then
   remove the code. Do not re-attempt anything in inventory §2.5 without first
   addressing the recorded blocker.
6. **Complexity budget.** Every new env var, CLI flag, or dispatch branch needs a
   stated reason and an inventory entry in the same commit. Prefer changing a default
   over adding a knob; prefer deleting a knob over keeping "just in case".
7. **CI + GPU checklist before merge.** The CPU CI (fmt, clippy ±cuda feature, tests)
   must be green. Anything touching kernels, the ABI, or the engine hot path
   additionally requires on-target: `./scripts/build_cuda.sh && ./scripts/smoke_cuda.sh`,
   the perf gate, and the parity floor. One concern per branch.
8. **Keep the docs contract.** Inventory updated in the same commit as behavior
   changes; a dated `DAILY.md` entry per session with an explicit verdict
   (SHIPPED / NEGATIVE / FALSIFIED / WIP / DECISION). An experiment that isn't
   written up will be re-run by the next agent — that costs more than the write-up.

## Parity reference and hard rules

### Validated against PyTorch reference (matching to within FP4 quantization noise)

Cosine similarity floor is `0.998` unless noted.

| Op | Cos sim | Notes |
|--|--|--|
| Embedding lookup | 1.000000 | bit-exact |
| Layer-0 input RMSNorm | 0.999999 | uses `(1 + weight)` parameterization |
| Layer-0 NVFP4 GEMMs (`in_proj_qkv`, `_b`, `_a`, `_z`) | 0.998–0.9997 | within FP4 noise |
| Layer-0 conv1d **with SiLU** | 0.999999 | confirms `conv1d_update` applies SiLU implicitly |
| Layer-0 gate (log) and beta | 1.000000 | softplus + log-decay correct |
| Layer-0 DeltaNet recurrence (zero state) | 0.999999 | `kv_mem`, `s_q`, `k_q` math is correct |
| Layer-0 final attn_out | 0.999904 | post out_proj |
| Layer-0 post-attn RMSNorm + MLP | 0.998 | gate/up/swiglu/down |
| Final RMSNorm + lm_head | 0.999999 | output stage is OK |
| Layer-3 full attention | 0.999999–1.000000 | q/k/v, official q/gate deinterleave, q/k norm, RoPE, causal attention, sigmoid gate, o_proj |
| All 64 local prefill boundaries (`"hello"`) | worst 0.998745 | no below-floor results after official q/gate layout fix |
| All 64 local prefill boundaries (`"hello world"`) | worst 0.999687 | no below-floor results |
| All 64 local decode boundaries (`"hello"` then forced token id `11`, `","`) | worst 0.999279 | validates carried KV cache, DeltaNet state, and conv history |
| Final logits (`"hello"`, `"hello world"`) | 0.999986 | `final_normed @ lm_head.T` vs emitted BF16 logits; same top-5 IDs |

### Remaining validation gaps

- **External reference logits** from vLLM or official Transformers/modelopt. Current local checks prove the CUDA engine matches the Python decomposition, not that both match an independent implementation.
- **Full sequential decode against a pure Python rollout**. Local decode boundaries pass; an unconstrained 64-layer sequential Python rollout can drift below the local floor because tiny BF16/FP4 differences compound. Treat local-boundary parity as the optimization regression gate unless comparing against a true external implementation.
- **MTP head numerical decomposition**. User-visible MTP 2/3 token parity is checked against non-MTP greedy chat for a short deterministic prompt, but the MTP head itself has not yet been decomposed against PyTorch layer-by-layer.
- **Chunked-verify parity beyond the gated prompts**. Because the verify-chunk forward pass is not bit-equal to per-token decode, MTP 1/2/3 produce 1–2 token argmax flips on borderline prompts. A parity harness on the verify chunk (multi-position attention + RMSNorm) would be needed to close this — it is not on the current optimization track.

### Parity harness (use this for any future kernel change)

The engine emits intermediate BF16 buffers when `QWEN36_DEBUG_DUMP_DIR=<path>` is set; pair with the `dump-logits` CLI command and the Python script at `/tmp/parity_check.py`:

```bash
mkdir -p /tmp/qwen36_dump
export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$QWEN36_FP4_KERNEL_LIB_DIR:${LD_LIBRARY_PATH:-}"
QWEN36_DEBUG_DUMP_DIR=/tmp/qwen36_dump \
  ./target/release/qwen36 dump-logits \
    --model-dir ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --prompt "hello" --top-k 5
python3 /tmp/parity_check.py            # diffs each dump vs PyTorch reference
```

The all-layer local-boundary scan lives at `/tmp/stack_parity.py`. Use:

```bash
QWEN36_DEBUG_DUMP_ALL_LAYERS=1 QWEN36_DEBUG_DUMP_DIR=/tmp/qwen36_dump \
  ./target/release/qwen36 dump-logits \
    --model-dir ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --prompt "hello" --top-k 5
PYTHONPATH=/tmp QWEN36_PARITY_DUMP=/tmp/qwen36_dump \
  QWEN36_PARITY_PROMPT="hello" QWEN36_STACK_LOCAL=1 \
  QWEN36_STACK_STOP_ON_FAIL=0 python3 /tmp/stack_parity.py
```

The decode local-boundary scan lives at `scripts/decode_parity.py`. Use:

```bash
rm -rf /tmp/qwen36_decode_dump
mkdir -p /tmp/qwen36_decode_dump
QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda" \
LD_LIBRARY_PATH="$PWD/target/cuda:${LD_LIBRARY_PATH:-}" \
QWEN36_DEBUG_DUMP_DIR=/tmp/qwen36_decode_dump \
QWEN36_DEBUG_DUMP_DECODE=1 \
QWEN36_DEBUG_DUMP_ALL_LAYERS=1 \
  ./target/release/qwen36 dump-decode \
    --model-dir ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --prompt hello --decode-token-id 11 --top-k 5 \
    --out /tmp/qwen36_decode_logits.bf16
QWEN36_PARITY_DUMP=/tmp/qwen36_decode_dump \
QWEN36_PARITY_PROMPT=hello \
QWEN36_PARITY_DECODE_TOKEN=11 \
QWEN36_DECODE_LOCAL=1 \
  python3 -u scripts/decode_parity.py
```

`QWEN36_DEBUG_LAYER_TRACE=1` also prints per-layer min/max/mean-abs through the trace helper. Both env vars are no-ops when unset; do **not** ship code paths that need them.

Adding a new parity checkpoint takes ~5 lines:
1. In `crates/runtime/src/engine.rs`, call `self.dump_buffer_to_disk(&dir, "<name>.bf16", <ptr>, <count>)` at the layer/step you want to capture (gated on `if let Some(dir) = &dump_dir`).
2. In `/tmp/parity_check.py`, load the dump with `load_bf16(...)`, compute the reference via the manual `dequant_nvfp4` helper, and feed both into `cmp(label, ours, ref)`.

### Hard rules for kernel changes from now on

- **No CUDA-kernel optimization lands without a parity check** at the affected op. The harness above is the contract.
- "Cos similarity ≥ 0.998" is the floor for an op-level claim of "matches reference"; below that, treat as a regression even if smoke and bench pass.
- Smoke tests use tiny shapes (e.g. `key_dim=4`) that often miss bugs in vectorized code paths. Smoke passing is necessary, not sufficient.
- The engine's per-iteration recurrent flow (`prefill.residual` accumulates `embed + Σ attn_i + Σ mlp_i` *after the next layer's input RMSNorm reads it*) is non-obvious — don't refactor the residual write order without re-running parity.
- `rmsnorm_nvfp4_quantize` must be safe when `residual_bf16 == residual_out_bf16`; decode post-attention RMSNorm uses this aliasing. Do not reintroduce a second read from the aliased residual after writing `residual_out`.
- RMSNorm weight semantics are split intentionally: base Qwen layer norms use the model's `(1 + weight)` parameterization (`direct_weight = 0` via `rmsnorm`), while per-head Q/K norms and DeltaNet value-head norms use the weight directly (`direct_weight = 1` via `rmsnorm_direct_weight`).
- For CUDA Graph work, remember that capture is record-only unless explicitly launched. Host `RuntimeState::position` must advance exactly when the graph actually runs, and any fallback kernels launched after capture inherit the active graph stream until `disable_decode_graph` drops it.

## References

- `docs/code-inventory.md` — current state of every component, flag, and dispatch path
- `DAILY.md` — chronological lab journal (experiments, benches, verdicts)
- `doc.md` — full implementation specification (start here for design intent)
- `CONTRIBUTING.md` — baseline workflow and the rules above
- `docs/development.md` — kernel dev loop and PR checklist
- `docs/repo-layout.md` — crate responsibilities and the ABI rule
- `docs/kernel-validation.md`, `docs/troubleshooting.md`, `docs/roadmap.md`

