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

## Current optimization status

The active optimization track is single-GPU RTX 5090 throughput for exactly `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` and the shipped NVFP4/MTP quantization. No generic-model fallback is required; prefer explicit guards and hard errors when an assumption is model-specific.

### MTP speculative decoding

- Runtime and CLI support `--mtp-speculative-tokens 0..=3`.
- MTP is automatically disabled for prompts longer than `QWEN36_MTP_MAX_PROMPT_TOKENS` (default `1_000_000`) to avoid pathological long-context regressions while tuning.
- MTP=1 keeps the optimized two-token verify graph path.
- MTP=2/3 is functional in chat and bench paths. It snapshots/restores DeltaNet recurrent state, conv history, main full-attention KV slices, and MTP KV slices for the exact verification token count. Rejection recovery commits only the accepted prefix instead of resetting and re-prefilling the full prompt.
- MTP=2/3 full-accept verification currently graph-captures the main verify chunk, per-draft greedy samples, and the next current-token sample. Next-draft generation intentionally stays on the validated host launch path. Do not move recursive MTP draft generation into the graph unless `chat --prompt "hello" --max-new-tokens 12` matches exactly for MTP 0, 2, and 3 afterward.
- MTP=1/2/3 share the chunked-verify forward pass, which is **not bit-equal** to the per-token decode path. For the gated prompts (`hello`, `hello world`) parity holds, but on prompts with a borderline argmax (e.g. `Write a short poem about cats.`, `Count from 1 to 5.`, `Write Python hello world`) all three speculative modes produce a self-consistent token stream that diverges from MTP=0 by 1–2 tokens. The divergence is independent of `QWEN36_MTP_MULTI_GRAPH_DISABLE`, so it is a chunked-verify numerical-noise issue, not a graph-capture bug. Treat the gated prompts as the parity floor; do not weaken them.
- `QWEN36_MTP_MULTI_GRAPH_DISABLE=1` disables the MTP=2/3 graph fast path and forces the host launch path. Keep this env var while bisecting MTP numerical issues.
- `QWEN36_MTP_TRACE=1` prints MTP verify windows, sampled verification tokens, next tokens, and next drafts.

Important invariant: when a captured graph leaves a non-default active stream installed, later host-launched kernels also run on that stream. Any host read of `token_u32` or `sampled_token_u32` must synchronize the active stream before D2H copy, otherwise the host can read stale tokens and drift. `Engine::read_current_token` and `Engine::read_sampled_token` currently enforce this.

Latest local checks before resuming optimization:

- `cargo fmt --all`
- `cargo check -p qwen36-fp4 --features cuda`
- `cargo build --release -p qwen36-fp4 --features cuda`
- Exact chat-token parity for `chat --prompt "hello" --max-new-tokens 12` with MTP 0, 2, and 3.

Bench reference (RTX 5090, `--prompt-tokens 128 --max-new-tokens 32`, full-accept regime, median of 5 runs — re-measure before drawing further conclusions):

| MTP | decode tok/s | speedup | accepted / decode steps |
|--|--|--|--|
| 0 | 45.5  | 1.00× | n/a |
| 1 | 62.5  | 1.37× | 16/16  (acceptance 1.00) |
| 2 | 74.2  | 1.63× | 21/22  (acceptance 1.00) |
| 3 | 86.3  | 1.90× | 24/24  (acceptance 1.00) |

The current numbers reflect four decode-side optimisations that landed in this branch:

1. **Combined gate + up FP4 GEMM** (`MlpFusedStore` in `crates/runtime/src/gpu.rs`). Pre-concatenates the gate_proj and up_proj NVFP4 weights along the output dim once at engine init; the decode path emits a single `(M=2·intermediate, N=1)` cuBLASLt FP4 GEMM instead of two. Only valid when every layer's gate/up share `weight_scale_2` and `input_scale` (validated at build time and confirmed for every layer of the shipped Qwen3.6 NVFP4 checkpoint).
2. **Vectorised BF16 I/O in `rmsnorm_kernel` and the first pass of `rmsnorm_nvfp4_quantize_kernel`** (`kernels-cuda/ops.cu`): switches the per-element scalar BF16 reads/writes to `__nv_bfloat162` pairs so each thread handles two elements per memory transaction. The per-group quantisation pass is left scalar.
3. **`swiglu_nvfp4_quantize` fused kernel** (new entry `qwen36_swiglu_nvfp4_quantize`, `kernels-cuda/ops.cu`): reads `[gate || up]` BF16 from the combined GEMM output, applies SwiGLU, and writes the down_proj input directly as NVFP4 (FP4 packed + e4m3 scales + tensor scale). Replaces a SwiGLU BF16 kernel + separate `nvfp4_quantize_bf16` kernel with one launch and removes a BF16 round-trip through `aux3`.
4. **DeltaNet 4-way in_proj fusion** (`LinearAttnInProjFusedStore` in `crates/runtime/src/gpu.rs`). Pre-concatenates `in_proj_qkv`, `_b`, `_a`, `_z` NVFP4 weights along the output dim into a single FP4 weight + block_scale per DeltaNet layer. Only valid when all four projections share `weight_scale_2` and `input_scale` (verified for every layer of the shipped checkpoint). `_b` and `_a` (48 rows each) are zero-padded to 128 rows to keep the FP4 block_scale outer-block alignment; the GEMM emits zeros for the padding rows, which the engine never reads. The decode path emits one combined `(M=16640, N=1)` GEMM instead of four (qkv: 10240, b: 48, a: 48, z: 6144). Downstream consumers (`conv1d_update`, `gdn_gate`, swiglu before `out_proj`) read their slices via pointer offsets into `forward.qkv`. **This is the biggest single decode win in the branch: profile `linear_attn` bucket drops ~30 % (13.6 → 9.3 ms host-launch), MTP=0 bench gains ~9 %.**

`QWEN36_PROFILE_DECODE_LAYERS=1` (host-launch path, instrumented via `cudaSynchronize` per block) shows the per-token decode breakdown shifting from ~38 ms (`linear_attn` ~14.4 ms, `mlp` ~14.7 ms) at branch start to ~31.5 ms (`linear_attn` ~9.3 ms, `mlp` ~13.8 ms) after the four optims. The graph-captured bench gain is smaller than the profile gain because the captured graph already amortises most per-launch overhead. The instrumentation env var bypasses the decode CUDA graph (see `crates/cli/src/main.rs` bench loop).

Memory cost of the fused stores: ~5.7 GB for `MlpFusedStore` + ~2.3 GB for `LinearAttnInProjFusedStore` ≈ 8 GB on top of the original weight upload. Comfortable on the 32 GB 5090; if memory ever becomes tight the originals could be dropped after the fused stores are built (no other code path consumes them on the decode side, but the prefill path still does, so this is a future trade-off).

**Two fusions tried and reverted in this branch** — keep in mind before re-attempting:

- *Full-attn Q/K/V fusion* (q_proj + k_proj + v_proj → one combined GEMM, M=14336). Validated parity but pushed peak engine VRAM to ~31 / 32 GB during init, which destabilised cuBLASLt plan caching: MTP=2/3 throughput regressed by 10–20 % with high run-to-run variance. The Q/K/V GEMMs are also small enough at decode that the launch-overhead saving is < 1 % bench. Re-enabling will require dropping the original q/k/v weights post-fuse (which means also fusing the prefill path), or finding another way to relieve VRAM pressure.
- *Single-block GQA `attention_decode_gqa_kernel`* (replacing the per-q-head `attention_decode_kernel` for the non-split path). Trades 24 q-head blocks for 4 kv-head blocks, each 6× heavier; at the bench `max_context = prompt + new_tokens` the split path doesn't fire so the redundant KV reads are absorbed by L2, and the wider per-block work on fewer SMs lost more than the broadcast saved (~5 % regression). The split-GQA kernel that already runs at long context (`n_splits >= 32`) covers the case where the broadcast wins.

**L2 access-policy primitive** — `cuda_set_l2_access_window` / `cuda_clear_l2_access_window` (`crates/kernels/src/memory.rs`, FFI to `qwen36_cuda_set_l2_access_window` in `kernels-cuda/runtime.cu`) are wired but **not currently called**. Pinning the DeltaNet state matrix (~50 MB, fits in the 5090's 232 MB L2) was tried and gave no measurable bench gain — the captured-graph replay locality already keeps the state hot in L2. The primitive is left in place for future experiments (e.g. pinning specific weight tiles).

**Realistic path beyond ~46 tok/s on MTP=0 / ~87 tok/s on MTP=3** — every easy fusion has now been done. The next steps require non-trivial work:
- *Prefill-path MLP / DeltaNet fusion*: same combined-GEMM idea but the multi-row cuBLASLt FP4 output is column-major `(M=combined, N=tokens)`, so `swiglu` / `conv1d_update` / `gdn_gate` would need stride-aware reads (or a deinterleave scatter kernel). Would help MTP verify forwards (chunked prefill) for a small additional gain on MTP=2/3.
- *MTP head BF16 → NVFP4*: the MTP attention + MLP run 24× per 32-token MTP=3 cycle and currently use BF16 weights. Quantising would speed it up significantly but needs a proper parity harness against the BF16 path; risks breaking the MTP parity gate.
- *Persistent / specialised DeltaNet recurrent kernel*: the existing decode kernel is rated 5/5, but a multi-layer persistent kernel that holds state in TMEM and avoids per-layer launches could shave more. Major rewrite.

Reproduce with:

```bash
QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda" \
LD_LIBRARY_PATH="$PWD/target/cuda:${LD_LIBRARY_PATH:-}" \
  ./target/release/qwen36 bench \
    --model-dir ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --prompt-tokens 128 --max-new-tokens 32 \
    --mtp-speculative-tokens <0|1|2|3>
```

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

- `doc.md` — full implementation specification (start here for design intent)
- `CONTRIBUTING.md` — baseline workflow and the rules above
- `docs/development.md` — kernel dev loop and PR checklist
- `docs/repo-layout.md` — crate responsibilities and the ABI rule
- `docs/kernel-validation.md`, `docs/troubleshooting.md`, `docs/roadmap.md`
