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

Bench against a real-text prompt (preferred for MTP-acceptance gates): `bench --prompt-file benches/data/long_prompt_4k.txt --prompt-tokens 4096 --max-new-tokens 64 --mtp-speculative-tokens 4`. The synthetic single-token-repeat default produces adversarial MTP acceptance — see `### 2026-05-15 — Anomaly diagnostics` below.
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

## Current optimization status

The active optimization track is single-GPU RTX 5090 throughput for exactly `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` and the shipped NVFP4/MTP quantization. No generic-model fallback is required; prefer explicit guards and hard errors when an assumption is model-specific.

### Direction B decode_gemv (NVFP4 N=1 hand-rolled gemv) — **default ON**

Decode-time NVFP4 GEMMs at gemv shape are routed through a hand-rolled tensor-core kernel built on the SM_120a `mma.kind::mxf4nvf4.scale_vec::4X.m16n8k64` atom (`kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu`). Soft-fallback to cuBLASLt for any unsupported shape via the existing dispatch in `crates/kernels/src/backend.rs`. Build script must use `-arch=sm_120a` (mandatory for the FP4 block-scaled MMA PTX).

**Enabled by default.** Two opt-out env vars (kill switches, either disables):
- `QWEN36_DECODE_GEMV_DISABLE=1` (preferred name)
- `QWEN36_DECODE_GEMV=0` (back-compat with the original opt-in flag)

Supported regime: `n==1 && m%16==0 && (k%1024==0 OR k%512==0)`. The entry point picks 16 warps/CTA when K%1024==0 (preferred path, ~47% occupancy at M=5120) or 8 warps/CTA when K%512==0 only (fallback for K=3584 out_proj on linear-attention layers). Anything outside that returns NOT_IMPLEMENTED. Two template instantiations compiled into the .so.

Bench (`bench --prompt-tokens 128 --max-new-tokens 128`, average of 3 warm runs, 2026-05-05):

| Mode  | cuBLASLt | gemv  | Δ |
|-------|----------|-------|---|
| MTP=0 | 43.5 tok/s | **49.85 tok/s** | **+14.5%** |
| MTP=4 | 95.4 tok/s | **99.3 tok/s**  | **+4.1%**  |

Hard parity gate: `chat --prompt "hello" / "hello world" --max-new-tokens 12 --mtp-speculative-tokens {0..4}` matches the cuBLASLt baseline byte-for-byte for all 10 combinations.

**Important runtime contract:** the kernel does NOT apply `a_scale_2`/`b_scale_2` (the per-tensor scales). The runtime caller in `crates/runtime/src/engine.rs` pre-folds them into `spec->alpha`, mirroring the cuBLASLt contract (`kernels-cuda/nvfp4_gemm.cu` only passes `alpha` and never dereferences `a_scale_2`/`b_scale_2`). A kernel that multiplies the per-tensor scales again on top of `alpha` produces gibberish on real model weights despite uniform-data smoke passing — this happened during B3.1 development and was caught only by end-to-end chat parity. Treat any future hand-rolled GEMM kernel the same way.

**Notes on what was tried and didn't ship:**
- Sub-byte LDSM (`SM100_SU4_DU8x16_x4_LDSM_N`, `b4x16_p64`) is a blocker for the k64 mxf4nvf4 atom — CUTLASS only binds it to the k32 f8f6f4 path. Plain `SM75_U32x4_LDSM_N` would consolidate 4 `ld.shared.u32` → 1 `ldmatrix` but offers limited gain (smem reads aren't the bottleneck). See `docs/superpowers/notes/2026-05-04-direction-b-cutlass-blockers.md` for the full investigation.
- TMA multicast (B4 in original spec) projected at <2% gain after setup overhead — activation reads are tiny (~5% of gmem traffic).
- Persistent grid doesn't help when each gemv call is independent; launch overhead is already amortized.

Full optimization history in commit log: `git log -- kernels-cuda/decode_gemv/`. Archived plans at `docs/superpowers/plans/archive/`.

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

### 2026-05-02 — Re-bench after PR #2 merge (MTP4 + batched lm_head + sample_rows)

WSL2, RTX 5090, model `Qwen3.6-27B-Text-NVFP4-MTP`, run on the
`feat/perf-tree-mtp-stack` branch (no tree-MTP code wired up — these
numbers are pure post-PR-#2 baseline). Median of 5 runs each.

`--prompt-tokens 128 --max-new-tokens 128`:

| MTP | decode tok/s | speedup | accepted_drafts / total_drafts |
|--|--|--|--|
| 0 | 41.6  | 1.00× | n/a |
| 1 | 55.6  | 1.34× | 64 |
| 2 | 73.4  | 1.77× | 85 |
| 3 | 79.9  | 1.92× | 96 |
| 4 | 98.3  | 2.36× | 102 |

`--prompt-tokens 128 --max-new-tokens 32` (matches PR #2's reported setup):

| MTP | decode tok/s | speedup | accepted_drafts |
|--|--|--|--|
| 0 | 34.3  | 1.00× | n/a |
| 1 | 42.0  | 1.22× | 16 |
| 2 | 41.0  | 1.20× | 21 |
| 3 | 46.7  | 1.36× | 24 |
| 4 | 55.1  | 1.61× | 25 |

PR #2 reported MTP4 = 117-120 tok/s on the same `--prompt-tokens 128
--max-new-tokens 32` setup; the numbers here are ~2x lower, attributed
to WSL2 launch latency (`docs/AGENT.md` notes 1-3 µs per kernel launch
penalty) and high run-to-run variance. The n=128 numbers are more
amortised and therefore higher.

`mtp_accepted_draft_tokens` is the cumulative count of accepted draft
tokens across the run (not per-cycle). At MTP=4 / n=128: 102 accepted
across ~25 cycles ≈ 4 drafts/cycle accepted = full-accept regime.

### 2026-05-03 — Re-bench after tree-MTP infra (no behavioural change)

Hard parity gate: `chat --prompt "hello" --max-new-tokens 12` and
`chat --prompt "hello world" --max-new-tokens 12` produce identical
output strings for `--mtp-speculative-tokens` ∈ {0, 1, 2, 3, 4}. ✅
Confirms the 18 tree-MTP infra commits (top-K kernel, tree-mask
attention path, walk_tree_acceptance, leaf buffers, per-leaf
snapshots, verify_mtp_tree_draft orchestrator) introduce no
regression on the chain MTP path.

`--prompt-tokens 128 --max-new-tokens 128`, median of 3 runs (excluded
isolated spikes attributable to a concurrent game running on the
same GPU):

| MTP | decode tok/s | speedup vs MTP=3 | accepted_drafts |
|--|--|--|--|
| 0 | 42.8  | 0.44× | n/a |
| 1 | 61.3  | 0.63× | 64 |
| 2 | 83.1  | 0.86× | 85 |
| 3 | 96.7  | 1.00× | 96 |
| 4 | 107.2 | 1.11× | 102 |

These numbers are ~5-15 % above the 2026-05-02 re-bench (MTP3 79.9 →
96.7, MTP4 98.3 → 107.2) but the delta is within run-to-run variance
caused by GPU contention (one MTP=4 run dropped to 75 tok/s). No
performance-relevant code landed between the two benches; treat both
as snapshots of the post-PR-#2 baseline.

### 2026-05-04 — Tree-MTP α (P1.I) bench: NEGATIVE result

Tree-MTP K>1 dispatch is fully wired: chat parity gate ✅ for K ∈ {1, 2,
4} on `hello` / `hello world` (identical token streams to chain MTP=3).
But the bench shows tree-MTP K>1 is dramatically slower than chain MTP
on this hardware/architecture:

| MTP | K | tok/s | leaf_accept_rate |
|--|--|--|--|
| 3 | 1 (chain fallback) | 110 | n/a |
| 3 | 2 | 41 | 0.00 |
| 3 | 4 | 27 | 0.00 |
| 4 | 1 (chain fallback) | 123 | n/a |
| 4 | 2 | 49 | 0.00 |

**Root cause:** tree-MTP processes K leaves via single-token
`forward_token_cuda` calls (one per leaf, full 64-layer forward each).
At ~25 ms per single-token decode in WSL2, K=2 adds ~50 ms / cycle and
K=4 adds ~100 ms / cycle. Chain MTP cycle is ~10 ms total. Per-cycle
overhead dominates the +1 token gain by ~10×.

**leaf_accept_rate = 0** is partly an artefact of the bench prompt (a
synthetic "x" repeated 128 tokens — MTP head's top-K disagrees with
the base model's argmax for the next position). Real prompts would
show non-zero leaf accept, but the per-cycle overhead would still
swamp the gain at this architecture.

**Path to make tree-MTP profitable** — Phase 2 work:

1. Batched leaf forward: process all K leaves through the model in ONE
   chunk pass. Requires (a) tree-mask attention (already implemented in
   `attention_prefill_kernel`, P1.D) PROPERLY USED in a custom
   `prefill_cuda_chunk_tree` variant; (b) batched DeltaNet kernel that
   fans K (state, output) pairs from one input state in a single launch.
   This collapses the K × 25 ms cost to ~1 × 30 ms (~3 ms / leaf vs
   25 ms / leaf today).
2. Or: drop tree-MTP for this hardware and pivot to a different
   optimisation track (NVFP4 gemv kernel for batch=1, PDL chains, etc.)

**Phase 1 outcome:** infrastructure complete (top-K kernel, tree-mask
attention path, walk_tree_acceptance v3, leaf buffers, per-leaf
snapshots, verify_mtp_tree_draft α with MTP head KV advance + next-
cycle pre-compute) and parity-validated. The infra stays useful for
Phase 2; the perf gain is deferred to that scope.

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
- *Native Linux re-bench*: WSL2 adds 1–3 µs per kernel launch; with ~30 cudaGraphLaunches/sec of decode + sync points, expected delta is ~5–10 % free. Zero engineering effort and the highest-ROI experiment to run before any further CUDA work. **Done 2026-05-15 — see entry below.**

### 2026-05-15 — Native Linux re-bench + max-new=1024 sweep

First full bench on native Linux (Ubuntu 26.04, glibc 2.43, CUDA 13.1.115, g++-15 host with rsqrt-noexcept patch on CUDA `crt/math_functions.h`, RTX 5090; post-PR #4 codebase). All numbers `--prompt-tokens 128 --max-new-tokens 1024` unless noted.

**Phase 1 — median of 5 runs at prompt=128**:

| MTP | decode tok/s (median) | decode min–max | prefill tok/s |
|-----|----------------------:|----------------|--------------:|
| 0   | 48.56 | 48.13–49.64 (±0.3 %) | 1117 |
| 3   | 96.41 | 90.43–97.22 (±3 %)   | 1118 |
| 4   | 112.08 | 102.23–117.46 (±7 %) | 1075 |

Vs WSL2 best (2026-05-03, same `prompt-tokens 128 max-new-tokens 128`): MTP3 96.7 → 96.4, MTP4 107.2 → 112.1. The native-Linux gain is smaller than the 5–10 % predicted; most of the cudaGraph launches in decode are already amortised by graph replay, so the µs/launch saving had limited room. MTP=4 variance grew with max-new (±7 % over 5 runs at 1024 vs ±1.5 % at 128) — same hardware running hotter.

**Phase 2 — context sweep, max-new=1024 (single run each, exploratory)**:

| prompt | prefill tok/s | decode MTP=0 | decode MTP=4 | acc MTP=4 |
|-------:|--------------:|-------------:|-------------:|----------:|
|   512  | 1849 | 51.6  | 101.0 | 100 %     |
|  1024  | 1712 | 52.1  | 84.0  | 100 %     |
|  2048  | 1389 | 51.1  | 63.6  | 99.9 %    |
|  4096  | 1095 | 50.1  | 62.8  | **83.9 %** ⚠️ |
|  8192  |  742 | 43.5  | **105.5** ⚠️ | 99.9 %    |

**Three anomalies to investigate before further optimisation**:

1. **Prefill degrades 2.5× from 512 to 8192** (1849 → 742 tok/s). PR #4 introduced a long-context prefill mode at `QWEN36_LONG_CONTEXT_AUTO_MIN_CONTEXT=8192` that disables `MlpFusedStore` / `LinearAttnInProjFusedStore` to save VRAM. Confirm with `QWEN36_PROFILE_PREFILL_CHUNKS=1` which phase regresses, and whether the VRAM safety margin still requires disabling fusions on the 32 GB native-Linux setup.

2. **MTP=4 acceptance drops to 83.9 % at prompt=4096** while staying ~100 % at 2048 and 8192. Suggests a dispatch transition between 2k and 4k that the MTP head doesn't follow correctly. Fixing this returns ~30 tok/s on that band.

3. **MTP=4 decode is faster at 8192 (105.5 tok/s) than at 2048–4096 (~63 tok/s)** — counter-intuitive given KV cache is larger. Likely the long-context mode swaps in a more efficient attention path (split-GQA `n_splits ≥ 32`?). Worth understanding what makes 8192 fast and porting that to the shorter-context paths.

**Post-anomaly roadmap** (external research, prioritised by ROI on this single-stream FP4 5090 target):

| # | Item | Source | Est. gain | Effort |
|---|------|--------|-----------|--------|
| B1 | NVFP4 KV cache (4-bit storage, FP8 dequant for attention) | NVIDIA dev blog 2026 — <1 % accuracy loss on Ruler-64K/LiveCodeBench | ~2× decode at long context | 1–2 wk |
| B2 | EAGLE-3 head replacing chain MTP-4 | arxiv 2503.01840 (NeurIPS'25) — 4.5–5 accepted/cycle vs our 3.92 | +15–25 % vs MTP-4 | 2–3 wk |
| B3 | Quest query-aware page sparsity | arxiv 2406.10774 (ICML'24) — stacks on B1 | ~3–5× extra at ≥ 4k ctx | 1–2 wk |
| B4 | Sage2++ FP8 attention with FP16 accumulator | arxiv 2505.11594 — rescaling trick | ~+10–20 % attention | 3–5 d |
| B5 | Split-K decode attention + SMEM page-index prefetch (FlashInfer pattern) | `flashinfer/csrc/single_decode.cu` | better SM utilisation at N=1 | 1 wk |
| C1 | FlashDecoding++ async-softmax unified-max | arxiv 2311.01282 | ~+10–15 % attention | 3–5 d |
| C2 | Prefix caching block-hash (multi-turn) | vLLM design doc | skip prefill on shared prefix | 1 wk |
| C3 | L2 persisting window on RoPE + MTP weights | already wired (`memory.rs`) | +1–3 % | 1–2 d |

**Explicit non-targets** (gain-negative or non-portable to SM_120):
- FlashAttention-4 / SM_100 trtllm-gen FMHA cubins — depend on `tcgen05` / TMEM, **absent on SM_120**. Reference SM_120 attention impl ≈ 94 % SOL with plain `mma.sync` (gau-nernst blog).
- Tree-MTP K>1 without a tree-aware attention kernel — already a NEGATIVE result on 2026-05-04. Re-attempt only with DeFT (ICLR'25) / FastTree style kernel.
- MagicPIG / StreamingLLM / PagedAttention / ring attention — none applicable to single-GPU single-stream.

### 2026-05-15 — Anomaly diagnostics + `PREFILL_CAPACITY` sweep

Four diagnostic experiments to isolate the three Phase-2-anomalies above. Raw data in `/tmp/{diag,e1,e2,e3,e4}*`.

**E1 — `QWEN36_PREFILL_CAPACITY` sweep at prompt=8192, max-new=1024** (medians of 2 reps each):

| cap | MTP=0 prefill | MTP=0 decode | MTP=4 prefill | MTP=4 decode | peak VRAM |
|----:|--------------:|-------------:|--------------:|-------------:|----------:|
| 512 (default) | 733 | 43.3 | 712  | 105.6 | ~30 GB* |
| 1024 | 833 | 44.4 | 807  | 107.3 | |
| **2048** | **875** | **45.1** | **841** | 106.2 | |
| 4096 | 865 | 44.3 | 847  | 104.3 | |
| 8192 | 869 | 44.3 | 871  | 103.4 | |

\* peak VRAM is inflated by concurrent multi-agent GPU sharing during the sweep; uncontended runs sit ≈22 GB at prompt=8192.

**Conclusion**: `QWEN36_PREFILL_CAPACITY=2048` is the optimum on 32 GB native Linux — **+19 % prefill** (MTP=0) and **+18 %** (MTP=4) vs the default 512. The default was tuned for tighter VRAM; on the 5090 native-Linux setup the chunk-launch overhead dominates and bigger chunks pay. Above 2048 there's no further prefill gain and decode regresses slightly. Recommended new default is **2048**, with the existing env var preserved as an override. The default lives at `crates/runtime/src/engine.rs:99-103` (`max_context.min(512)`).

**E2 — Bracketing the MTP-acceptance dip** (median of 3 reps, MTP=4, max-new=1024):

| prompt | n_splits | acc | decode tok/s |
|-------:|---------:|----:|-------------:|
| 2048 | 33 | 0.999 | ~63 |
| 2304 | 37 | 0.978 | 56.1 |
| 2560 | 41 | 0.998 | 56.6 |
| 3072 | 49 | **1.000** | **121.5** |
| 3584 | 57 | **0.913** | 82.5 |
| **4096** | 65 | **0.839** | 64.7 |
| 4608 | 73 | **0.800** | 56.1 |
| 5120 | 81 | 0.999 | 117.2 |
| 6144 | 97 | 0.998 | 114.4 |
| 7168 | 113 | **0.926** | 78.7 |
| 8192 | 129 | 0.999 | ~105 |

The dip is **not a single sharp threshold**: there are two valleys (3584–4608 with acc 0.80–0.91 and 7168 with acc 0.93) separated by a recovery band (5120–6144 acc ~1.0). MTP-depth doesn't matter — at prompt=4096, MTP={2, 3, 4} all show acc ~0.82. The initial hypothesis "split-GQA n_splits ≥ 64 dispatch is broken" was **falsified**: `QWEN36_ATTENTION_SPLIT_DISABLE=1` and `QWEN36_PREFILL_SPLIT_MIN_SPLITS=256` both leave acc at ~0.85, and prompts 5120/6144 (n_splits=81/97) pass the same dispatch path with acc ~1.0.

**E3 — Same prompts, real prose** (chat path with `QWEN36_MTP_STATS=1`, ~4000 tokens of natural English from `doc.md` + `AGENT.md`): acc = **0.980** at 4k, 0.942 at 2k, 0.925 at 8k. **The 4k dip does NOT reproduce on natural text.** It only manifests with synthetic single-token prompts (`bench` default) AND with short looping seeds (`--token-text "the quick brown fox..."` 27-token seed repeated to 4096). So:
- E2 bench at 4k with 27-token-loop: acc 0.84 (bug present)
- E3 chat at 4k with natural prose: acc 0.98 (bug absent)

**Verdict on the 4096 acceptance anomaly**: it is a **synthetic-prompt artefact**, not a production regression. The MTP draft head has a known weakness on low-entropy periodic inputs; on natural text it behaves normally. The `bench` MTP acceptance number at long prompts should be treated as an adversarial stress test, not a production forecast. Cheap follow-up: add `--prompt-file <path>` to `bench` and ship a small natural-text corpus for CI.

**E4 — `QWEN36_PROFILE_DECODE_LAYERS=1` at prompts {2048, 4096, 8192}**:

| bucket | p=2048 | p=4096 | p=8192 |
|--------|-------:|-------:|-------:|
| embed       |  0.18 |  0.19 |  0.18 |
| linear_attn |  4.13 |  4.21 |  4.45 |
| full_attn   |  3.44 |  3.74 |  5.87 |
| mlp         | 10.24 | 10.36 | 10.68 |
| lm_head     |  1.65 |  1.65 |  1.65 |
| **total**   | 19.64 | 20.15 | 22.82 |

**Every per-block bucket is monotone in context length** — no bucket regresses at 4k. The 64 → 105 tok/s gap between prompt=4k and 8k (MTP=4) is **entirely explained by acceptance**:
- At 4k, acc=0.754 → 20 main_steps + 82 mtp_steps = **102 forward calls** to emit 64 tokens.
- At 8k, acc=0.98 → 14 main_steps + 56 mtp_steps = **70 forward calls** for the same 64 tokens.

Per-call cost is *lower* at 8k than 2k (9.96 vs 14.41 ms). The decode is fine; the draft acceptance is the lever.

**Net takeaways from the four experiments**:
1. **Anomaly #1 (prefill 1849 → 742)** — root cause is chunk capacity, not the long-context fusion-disable. **Action**: set default `QWEN36_PREFILL_CAPACITY=2048` (free +18 % prefill at 8k).
2. **Anomaly #2 (MTP=4 acc 0.839 at 4096)** — synthetic-prompt artefact; does not affect real text. **Action**: add `--prompt-file` to `bench`, do not chase the kernel-side hypothesis.
3. **Anomaly #3 (MTP=4 decode @ 8k faster than @ 4k)** — explained by anomaly #2 downstream; same draft acceptance everywhere on real text, so the "fast at 8k" is just the absence of the synthetic-input pathology. No backport needed.

The Phase-2 single-run sweep was thus dominated by synthetic-prompt MTP pathology in the 2k-6k range. The real prefill-capacity win is independent and lands at **+18 %** for prompt=8192 just by changing one default.

#### Shipped 2026-05-15 (P0 batch)

- **`QWEN36_PREFILL_CAPACITY` default raised from `max_context.min(512)` to `max_context.min(2048)`** in `crates/runtime/src/engine.rs` (was `:99-103`). Free **+18 % prefill** at prompt=8192 (MTP=4: 712 → 841 tok/s; MTP=0: 733 → 875). Empirical plateau at 2048 — caps of 4096 / 8192 yield no further prefill gain and regress decode ~1 % (see E1 table). Env var still overrides the new default for explicit control; the old 512 value is reachable via `QWEN36_PREFILL_CAPACITY=512`.

- **`bench --prompt-file <path>` option added** to `crates/cli/src/main.rs`. Reads the file, tokenises to up to `--prompt-tokens` tokens, and runs the standard prefill+decode loop on that real input. Corpus shipped at `benches/data/long_prompt_{4k,8k}.txt` (natural English prose ≈ 4 k / 8 k tokens). Synthetic single-token-repeat path remains the default for back-compat and microbenchmarking. **For any MTP-acceptance CI gate, always use `--prompt-file` or the `chat` path** — synthetic single-token-repeat prompts produce adversarial acceptance numbers (E2: bench acc 0.84 at 4 k vs E3: chat acc 0.98 on the same length of real text) that do not reflect production behaviour.

- **Reference path for measuring real MTP acceptance**: `cargo run --release -p qwen36-fp4 --features cuda -- chat --prompt "$(cat real_prompt.txt)" --max-new-tokens 64 --mtp-speculative-tokens 4` with `QWEN36_MTP_STATS=1`. The chat path prints `mtp.stats accepted=N rejected=M acceptance_rate=R` from `run_chat_mtp_multi` in `crates/cli/src/main.rs`. Use this — not `bench` synthetic prompts — for any "is the MTP head healthy?" gate. E3 verified real-text acc=0.98 vs synthetic-bench acc=0.84 at the same 4 k prompt length, confirming the dip diagnosed in E2 is a synthetic-prompt artefact only.

### 2026-05-19 — Productive spin / L2 prefetch (Phase 1) — **NEGATIVE result, disabled by default**

Phase 1 of the megakernel roadmap built an idle-SM L2-prefetch path (à la
AlpinDale's RTX 5090 megakernel post) that overlaps the small-CTA full-attn
decode kernel with a read-only walk of the upcoming MLP combined weight
(`gate+up`, ~89 MB) on a secondary CUDA stream. The 5090's 16 full-attn
layers leave ~146 of 170 SMs idle during the 24-CTA attention kernel, so the
mechanism is sound; it just **doesn't move the needle on this engine** because
the decode CUDA graph already keeps the MLP weights L2-resident across
iterations (same code path as the L2 access-window experiment on DeltaNet
state — both lose to graph-replay locality).

**Bench (RTX 5090 native Linux, median of 5)**:

| Config | Baseline | `QWEN36_PRODUCTIVE_SPIN=1` (128 CTAs) | Δ |
|---|---:|---:|---:|
| `prompt=128 mtp=0` | 54.08 | 54.15 | +0.13 % |
| `prompt=128 mtp=3` | 112.02 | 112.61 | +0.53 % |
| `prompt=128 mtp=4` | 118.06 | 118.13 | +0.06 % |
| `prompt=4096 mtp=0` | 53.64 | 53.35 | −0.54 % |

All deltas within run-to-run noise. The plan's +5 % go/no-go threshold is not
met; further CTA-count tuning would not cross it given the mechanism itself
doesn't help here.

**Status — kept opt-in, disabled by default**:

- `QWEN36_PRODUCTIVE_SPIN=1` activates the prefetch fork in the full-attn
  decode path. Off by default. Parity gate validated bit-equal output on
  `chat --prompt hello/hello world --max-new-tokens 12` for MTP {0..4}.
- `QWEN36_PRODUCTIVE_SPIN_CTAS=N` (default 128, range 1..=1024) — number of
  CTAs the prefetch kernel launches on the secondary stream.
- The supporting C ABI (`qwen36_l2_prefetch`,
  `qwen36_internal_prefetch_stream`, `qwen36_cuda_event_*`) and the Rust
  helpers (`DecodeAuxStreams`, `fork_productive_spin` /
  `join_productive_spin`) **stay shipped** — they are the reusable cross-
  stream sync infrastructure that Phase 2 (per-block megakernel) needs for
  any future fork/join pattern inside the decode graph. The l2_prefetch
  kernel itself lives at `kernels-cuda/decode_gemv/l2_prefetch.cu` and is
  covered by a direct smoke test in `kernels-cuda/smoke.cu`.

**Lesson for Phase 2**: stop targeting bandwidth that the captured graph
already amortises. The remaining decode wins must come from collapsing
kernel launches and keeping activations in registers/SMEM between sub-ops
(per-block megakernel), not from prefetch tricks adjacent to the existing
flow.

### Mirage megakernel branch (`feat/mirage-megakernel`) — **dead code, kept for reference**

> **WARNING:** the file `kernels-cuda/megakernel/nvfp4_matvec_sm120.cu` does not actually run its CUTLASS path. It guards the SM120 body with `#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)` but never includes `<cutlass/arch/config.h>` (the header that defines the macro). So the `#if` evaluates to false at preprocessing and the function returns `NOT_IMPLEMENTED` for every shape, falling back to cuBLASLt silently. The parity claims below were never actually testing the megakernel — they were testing cuBLASLt twice. Discovered 2026-05-04 during Direction B development; documented in `docs/superpowers/notes/2026-05-04-direction-b-cutlass-blockers.md`. Direction B uses a hand-rolled gemv kernel (`kernels-cuda/decode_gemv/`) with verified parity instead. The megakernel scaffolding is left in place because the existing dispatch wiring + CUTLASS dependency are reusable for any future CUTLASS-based experiment, but **do not trust the "validated parity" claim below without re-running the gate.**

Long-running side branch that lays a CUTLASS-based substrate for the
NVFP4 GEMMs without committing to perf gains on `codex/numerical-parity-guardrails`.
Contents (six commits, branch is opt-in via env var, perf-neutral on
the default path):

* CUTLASS 4.4.2 vendored as a shallow clone under `kernels-cuda/cutlass/`
  (gitignored, ~200 MB). Build pipeline (`scripts/build_cuda.sh`) detects
  the directory and compiles `kernels-cuda/megakernel/nvfp4_matvec_sm120.cu`
  with `--expt-relaxed-constexpr --extended-lambda`.
* `qwen36_megakernel_nvfp4_gemm` (`kernels-cuda/megakernel/`) — live
  CUTLASS `GemmUniversalAdapter` for SM120 NVFP4 → BF16 with
  `ThreadBlockShape = <128, 8, 128>` (matches the SM120 FP4 MMA atom
  m16n8k64). Validated parity for every NVFP4 GEMM in the decode hot
  path under `QWEN36_USE_MEGAKERNEL_GEMM=1`.
* Rust-side dispatch (`crates/kernels/src/backend.rs`) routes through
  the megakernel when the env var is set; `QWEN36_STATUS_NOT_IMPLEMENTED`
  (= 5) is treated as a soft fallback so the cuBLASLt path still runs
  on shapes the kernel does not specialise.
* `cuda_set_l2_access_window` / `cuda_clear_l2_access_window` plumbing
  on `crates/kernels/src/memory.rs` (already in `codex/numerical-parity-guardrails`,
  ported here for completeness).

**Empirical finding from the branch:** scale-factor layout is
identical between cuBLASLt's `vec16_scale_offset` and CUTLASS's
`Sm1xxBlkScaledConfig::SfKMajorAtom` for `SFVecSize=16`. No re-tile
pass was needed. CUTLASS at `<128, 8, 128>` lands at perf parity with
cuBLASLt for our N=1 decode shapes; the wider `<128, 128, 128>`
default tile is ~5 % slower. The published "1.78× over cuBLASLt at
BS=1" reference does not reproduce on this exact shape mix with
the auto-selected schedule.

**Phase 2 (epilogue fusion) is mathematically blocked for SwiGLU on
our `MlpFusedStore` layout** — gate and up are stacked along M with
intermediate=17408 = 136×128, so every 128-row epilogue tile is
*entirely* gate or *entirely* up. SwiGLU needs to pair `gate[i]` with
`up[i + intermediate]`, which live in separate tiles, and CUTLASS
epilogues do not share state across tiles. Restructuring to batched
GEMM with L=2 just moves the pairing problem. Full deep-dive in
`docs/mirage-megakernel.md`.

The only tractable Phase 2 win remaining is `LinCombBlockScaleFactor`
on down_proj (NVFP4 output) + a new RMSNorm-from-NVFP4 kernel
variant — estimated 3–6 % MTP=0, ~1–2 days of focused kernel + parity
work. Recommended only after the WSL2 vs native Linux experiment
above is complete, since it gates whether 100 tok/s is reachable
without further CUTLASS work.

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
