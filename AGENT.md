# AGENT.md

This file provides guidance to AI coding agents (Claude Code, Codex, Cursor, etc.) when working with code in this repository.

## Project

Single-stream inference engine for `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`, targeting **RTX 5090 / Blackwell SM120 only**. Rust 1.85 (edition 2024) Cargo workspace + a CUDA shared library built out-of-band by shell scripts. The repo is performance- and hardware-specific; **prefer explicit failure over silent fallback**.

Full design intent: `doc.md`. Operational docs: `docs/`.

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
- **MTP head**. The base language-model prefill/logits path is validated; speculative MTP parity is separate work.

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

## References

- `doc.md` — full implementation specification (start here for design intent)
- `CONTRIBUTING.md` — baseline workflow and the rules above
- `docs/development.md` — kernel dev loop and PR checklist
- `docs/repo-layout.md` — crate responsibilities and the ABI rule
- `docs/kernel-validation.md`, `docs/troubleshooting.md`, `docs/roadmap.md`
