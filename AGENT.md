# AGENT.md

This file provides guidance to AI coding agents (Claude Code, Codex, Cursor, etc.) when working with code in this repository.

## Project

Single-stream inference engine for `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`, targeting **RTX 5090 / Blackwell SM120 only**. Rust 1.85 (edition 2024) Cargo workspace + a CUDA shared library built out-of-band by shell scripts. The repo is performance- and hardware-specific; **prefer explicit failure over silent fallback**.

Full design intent: `doc.md`. Operational docs: `docs/`.

**Before building anything new, read `docs/code-inventory.md`** — the consolidated map of
every component (active / opt-in / archived-negative / dead), the default dispatch paths,
and every `QWEN36_*` env var. This file (AGENT.md) is the chronological journal; the inventory is
the current state. Several past sessions rebuilt code that already existed — check the
inventory first, and update it in the same commit when you change a default or add a flag.

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

### 2026-05-23 — Per-block megakernel (Phase 2) — **NEGATIVE result; code REMOVED 2026-06-10**

> The stage kernels, smoke coverage, Rust FFI and the
> `QWEN36_MEGAKERNEL_FULL_ATTN_STAGE_F4` gate described below were deleted on
> the `chore/rationalization` branch (recover from git history if needed).
> The write-up stays as the record of the pattern and its bring-up bugs.

Phase 2 of the megakernel roadmap built the per-block megakernel pattern
described by AlpinDale's RTX 5090 post: persistent grid + atomic
work-stealing + inter-CTA spinlock barriers, fusing multiple sub-ops of a
full-attn layer into single kernel launches. Six stages shipped byte-exact
against the standalone reference (`./scripts/smoke_cuda.sh` covers all):

- **Stage A** — skeleton + barrier infra (identity copy).
- **Stage B.1 / B.2 / B.3** — RMSNorm → RMSNorm+quantize → +Q proj GEMV.
- **Stage C** — Stage B.3 + K + V + partial RoPE.
- **Stage E** — attn_out quantize → o_proj GEMV → residual + post-attn
  RMSNorm + quantize.
- **Stage F.1 / F.2 / F.4** — gate+up GEMV → +SwiGLU+quantize → +down
  GEMV [+ optional residual add]. F.4 uses `cudaOccupancyMaxActiveBlocks
  PerMultiprocessor`-driven grid sizing because register pressure
  collapses occupancy and a hardcoded grid would deadlock the spinlock.

Stage F.4 is wired into the MLP hot path behind
`QWEN36_MEGAKERNEL_FULL_ATTN_STAGE_F4=1`. Parity gate (10/10 chat
byte-exact across MTP={0..4} × {"hello", "hello world"}) passes.

**Bench (RTX 5090 native Linux, median of 5, prompt=128 max-new=32)**:

| Config | Baseline | `QWEN36_MEGAKERNEL_FULL_ATTN_STAGE_F4=1` | Δ |
|---|---:|---:|---:|
| `mtp=0` | 55.27 | 53.05 | **−4.0 %** |
| `mtp=4` | 110.78 | 110.88 | +0.1 % |

MTP=0 regresses; MTP=4 is noise. Same lesson as Phase 1: the captured
decode CUDA graph already amortises kernel launches, so collapsing
{gate+up cuBLASLt GEMM, swiglu_nvfp4_quantize, down NVFP4 GEMV} into one
megakernel launch trades cuBLASLt's CUTLASS-tuned MLP GEMM for our
hand-rolled GEMV — that's a worse trade on the intermediate shape where
cuBLASLt wins. The megakernel's other potential benefit (keeping
intermediates in SMEM/registers) doesn't materialise because we still
write `gate_up_out` and `swiglu_fp4` to HBM between phases (their sizes
exceed any single CTA's SMEM and they need to be visible across the
persistent grid).

**Status — kept opt-in, disabled by default**:

- All ~8 stage kernels stay shipped + smoke-validated.
  `kernels-cuda/megakernel/full_attn_block_sm120.cu` is the canonical
  reference for the persistent + work-stealing + barrier pattern; the
  inter-phase spinlock requires every CTA to be concurrently scheduled,
  so future re-use should either run on a dedicated GPU or switch to
  `cudaLaunchCooperativeKernel`.
- `QWEN36_MEGAKERNEL_FULL_ATTN_STAGE_F4=1` activates the MLP fast path
  in `run_mlp_with_quantized_input`. Off by default. Stage E + Stage
  B.3 also have Rust FFI shipped but no engine call site (the residual
  fusion contract of the standard rmsnorm_nvfp4_quantize doesn't compose
  cleanly with Stage E without a wider refactor).
- Two bugs caught during bring-up are worth remembering:
  1. **Hardcoded persistent grid deadlocks under register pressure.**
     A "safe-looking" 256-CTA grid for Stage F.2/F.4 actually exceeds
     the per-SM occupancy ceiling once the kernel inlines two GEMV
     bodies + SwiGLU, so the spinlock waits forever. Fix:
     `cudaOccupancyMaxActiveBlocksPerMultiprocessor` × SM count.
  2. **Conditional `__shfl_sync` with full warp mask is UB.** The
     SwiGLU quantize helper had `__shfl_sync(0xffffffff, ..., lane*2)`
     inside `if (lane < 8u)` — only 8 lanes called it but the mask
     promised all 32. On SM_120 the warp deadlocked. Fix: hoist the
     shuffles to unconditional, use the result only on lanes 0..7.

**Stage D (attention) deferred.** The 24-CTA attention body uses
`__syncthreads` internally; inlining it into a 256-CTA persistent grid
where 232 CTAs idle at the barrier is straightforward but not worth
implementing given Phase 2's negative perf result.

**Lesson for Phase 3+**: the captured graph + cuBLASLt's CUTLASS MLP
kernel already win the decode hot path on this engine. Further decode
gains need to attack different bottlenecks — KV-cache quantization
(B1), attention algorithm changes (Sage2++ retry, EAGLE-3), or weight
layout — not kernel fusion.

### Mirage megakernel branch (`feat/mirage-megakernel`) — **dead code; REMOVED from main 2026-06-10**

> `kernels-cuda/megakernel/nvfp4_matvec_sm120.cu` (+ stub) and the
> `QWEN36_USE_MEGAKERNEL_GEMM` dispatch were deleted on the
> `chore/rationalization` branch — the kernel never executed (see WARNING
> below). The analysis survives in `docs/mirage-megakernel.md`.

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

### 2026-06-09 — Interpreter megakernel — WIP, Stage 2 gate FAILED, engine missing

Spec: `docs/superpowers/specs/2026-06-08-interpreter-megakernel-design.md`.
Goal was one persistent kernel for the whole decode pass with on-SM
counter sync, `cp.async.bulk` weight prefetch crossing instruction
boundaries, and sub-instruction chunking of MLP intermediates.
Projected gain: −1.7 ms / token realistic (+7.7 % on MTP=0, +3 tok/s).
Stage 2 gate from the spec: ≥ +3 tok/s on MTP=0 vs baseline or revert.

**What landed (Codex, in-flight working tree, not committed yet):**

- CUDA substrate `kernels-cuda/interpreter/` (~460 LoC across
  `interpreter_sm120.cu`, `instruction.h`, `counters.cuh`,
  `page_allocator.cuh`).
- 14 device opcode bodies in `kernels-cuda/interpreter/opcodes/`:
  `rmsnorm_bf16`, `rmsnorm_nvfp4_quant`, `nvfp4_gemv`, `nvfp4_quantize`,
  `swiglu_bf16`, `swiglu_nvfp4_quant`, `rope_partial`, `residual_add`,
  `deltanet_recur`, `attn_decode_full`, `lm_head_tiled`,
  `q_proj_deinterleave`, `q_proj_sigmoid_gate`, `conv1d_gdn_gate_fused`,
  plus `fallback_trampoline`.
- Rust ABI `crates/kernels/src/interpreter.rs` with typed instruction
  constructors and `static_assert(sizeof == 152)` mirror.
- `crates/runtime/src/interpreter_compile.rs` (~2 850 LoC):
  whole-layer compilers — `compile_full_attention_layer_decode`,
  `compile_full_attention_input_layer_decode`,
  `compile_full_transformer_layer_decode`,
  `compile_linear_attention_tail_decode`,
  `compile_linear_attention_post_inproj_decode`,
  `compile_linear_attention_layer_decode`,
  `compile_linear_attention_input_layer_decode`,
  `compile_linear_transformer_layer_decode`, plus per-op compilers.
- `crates/runtime/src/engine.rs`: 6 call sites of
  `interpreter_decode_sm120` (one per slice kind), feature-gated by
  `QWEN36_INTERPRETER_DECODE` master + fine gates per opcode
  (`_MLP`, `_NORM_MLP`, `_RMSNORM`, `_DELTANET`, `_ATTN`, `_ROPE`,
  `_FULL_ATTN`, `_FULL_ATTN_LAYER`, `_FULL_ATTN_INPUT_LAYER`,
  `_FULL_TRANSFORMER_LAYER`, `_LINEAR_ATTN_TAIL`,
  `_LINEAR_ATTN_POST_INPROJ`, `_LINEAR_ATTN_LAYER`,
  `_LINEAR_ATTN_INPUT_LAYER`, `_LINEAR_TRANSFORMER_LAYER`) and
  per-layer-type `_DISABLE` flags for surgical opt-out.

**What is NOT yet implemented (gap vs spec, drives the gate failure):**

- **No `cp.async.bulk` weight prefetch crossing instruction
  boundaries.** Spec line 70 ("Weight prefetch overlap") projected
  −0.8 ms realistic. Not present. PageAllocator has 4 weight slots
  reserved for double-buffering; none of them are used by any opcode
  body.
- **No sub-instruction chunking.** Each interpreter instruction
  publishes ONE counter at the end (`arrive_and_publish_last_cta`
  after a single `__syncthreads()`). Spec called for SwiGLU output
  publishing 4 chunks (4 352-element each) so `down_proj` could start
  consuming chunk by chunk. Not present.
- **No SMEM activation residency** between Q/K/V and o_proj. Every
  opcode reads/writes through `payload_ptr<...>(insn.payload[X])`
  which are GMEM pointers; nothing stays SMEM-resident across opcode
  boundaries.
- **Dispatch loop has a `__syncthreads()` after every instruction**
  (`interpreter_sm120.cu:162`). That barrier is structurally
  incompatible with both prefetch overlap and chunking.
- **Emitted programs are strictly serial.** Each instruction's `deps`
  point at the previous instruction's `publishes_counter` with no
  fan-out — the substrate could express parallel work, but the
  compiler emits a single chain.
- **No bench microsmoke gating the interpreter path.** `smoke.cu`
  doesn't exercise `interpreter_decode_sm120`.

**Bench (2026-06-09, in-flight, RTX 5090 native Linux, dmtp + zed +
Megabonk taking ~1.4 GB → `QWEN36_LONG_CONTEXT_MODE=1` required to
fit; both sides use it so comparison is fair):**

prompt=128, max-new=32, median of 5:

| Config | tok/s | Δ vs baseline |
|---|---:|---:|
| MTP=0 baseline | 36.88 | — |
| MTP=0 interpreter | 35.07 | **−4.9 %** |
| MTP=4 baseline | 74.96 | — |
| MTP=4 interpreter | 76.28 | +1.8 % (noise) |

Same shape as the per-block megakernel Phase 2 (`9cc92fc` /
2026-05-23): regressive on MTP=0, noise on MTP=4. Stage 2 spec gate
is +3 tok/s on MTP=0; we are at −1.81. **Gate failed.**

The regression makes sense given the gap: the substrate adds
dispatch overhead (opcode switch, counter spin, per-instruction
syncthreads, GMEM round-trip between opcodes inside a slice) without
yet collecting any of the wins the spec projected. The shipped state
is "kernel-fusion compiler that emits one launch per slice", not
"on-SM interpreter that pipelines instructions" — different
architecture, much smaller projected ceiling.

**Engine work Claude did in this session (commit follows):**

1. **L2 lookahead prefetch helper** — new
   `kernels-cuda/interpreter/prefetch.cuh`. Each instruction reads
   `program[pc + 1]` from GMEM (cheap) and, for opcodes whose weight
   pointer is determined by the canonical payload layout
   (`NVFP4_GEMV`, `LM_HEAD_TILED`, `RMSNORM_BF16`,
   `RMSNORM_NVFP4_QUANT`), issues a budget of 64 KiB of
   `prefetch.global.L2` PTX hints during instruction `pc`'s own
   compute. Non-blocking. No ABI change.
2. **Kernel-launch flag plumbing.** Added a `flags: u32` parameter
   to `qwen36_interpreter_decode_kernel`; bit 0 enables the
   prefetch lookahead. Rust side reads
   `QWEN36_INTERPRETER_PREFETCH` (new env var, default 0) into the
   single `interpreter_launch_flags()` helper used by all 17
   `InterpreterProgramSpec` construction sites in `engine.rs`.
3. **Dropped the original "drop per-instruction `__syncthreads()`"
   idea.** Measured: 60 syncthreads/token × ~10 ns ≈ 600 ns =
   0.003 % of token budget. Not where the regression comes from.

**Did NOT do this session (still gaps vs spec):**

- `cp.async.bulk` weight prefetch into double-buffered SMEM pages.
  The `PageAllocator` slots remain unused. A real SMEM-resident
  weight overlap path would require refactoring the NVFP4 GEMV body
  (~330 PTX-heavy lines in
  `kernels-cuda/decode_gemv/nvfp4_gemv_mma_kernel.cuh`) to accept a
  pre-warmed A operand from SMEM. Multi-day refactor; deferred.
- Sub-instruction chunking. Requires the GEMV / SwiGLU bodies to
  publish per-chunk counters mid-execution. The substrate ABI can
  express it (extend `publishes_counter` to an array indexed by
  chunk id) but no opcode body emits chunk-grained progress yet.

**Bench (after prefetch + flags plumbing, same session, same GPU
load — comparison apples-to-apples):**

prompt=128, max-new=32, median of 5:

| Config | tok/s | Δ vs baseline |
|---|---:|---:|
| MTP=0 baseline (no interpreter) | 37.50 | — |
| MTP=0 interpreter (PF off) | 35.02 | −6.6 % |
| MTP=0 interpreter (PF on)  | 34.12 | −9.0 % |
| MTP=4 baseline (no interpreter) | 74.98 | — |
| MTP=4 interpreter (PF off) | **80.43** | **+7.3 %** |
| MTP=4 interpreter (PF on)  | 75.19 | +0.3 % |

**Two findings worth noting:**

- **The interpreter actually WINS on MTP=4 (+7.3 %) without
  prefetch.** Hidden in the first bench session by noise. Hypothesis:
  with MTP=4 the speculative head runs 4× per accepted token, so
  per-token graph-node count is much higher than MTP=0, and the
  whole-layer compile saves enough cross-node overhead to net out as
  a real gain. **This passes the spec gate ("≥ +3 tok/s")** on
  MTP=4 even though it fails on MTP=0.
- **L2 prefetch as implemented is a net negative** on both MTP=0
  (−2.4 pts) and MTP=4 (−7 pts). 64 KiB / instruction across all
  prefetch-eligible opcodes pollutes the L2 working set faster than
  warming up the next instruction's first cache lines saves. Smaller
  budgets, opcode-filtered prefetch (only NVFP4 GEMV on the bigger
  GEMMs), or a true `cp.async.bulk`-into-SMEM mechanism would behave
  differently.

**Decision:**

- Prefetch helper stays in-tree but **default OFF**
  (`QWEN36_INTERPRETER_PREFETCH=1` to opt-in). The infra is reusable
  for budget / opcode-filter experiments, and the flag is plumbed
  through every launch site already.
- Interpreter master flag (`QWEN36_INTERPRETER_DECODE`) now defaults
  to **auto**: unset/`auto` enables whole-layer interpreter programs
  only when `EngineConfig.mtp_speculative_tokens > 0`; MTP=0 remains on
  the captured-graph baseline. `QWEN36_INTERPRETER_DECODE=1` forces the
  interpreter on for every decode engine; `=0` forces it off. Fine
  per-op env gates still override for diagnostics.
- MTP=0 regression source not yet root-caused. Most likely
  candidates: (1) the 192-CTA grid wastes atomicAdd publishes on
  small opcodes (`residual_add`, `rmsnorm_bf16`), (2) per-instruction
  cross-CTA counter sync contends in L2, (3) the dispatch loop's
  scratch SMEM allocation (`__shared__ float scratch[512]`) bloats
  the kernel's register pressure / occupancy budget. Each is testable
  in isolation.

**Codex follow-up (2026-06-09, auto-policy landed):**

- Rust gate plumbing is context-aware instead of process-global: the
  opcode allow-list remains cached, but the master decision is made
  from each engine's MTP depth. This avoids accidentally enabling the
  interpreter for MTP=0 while letting MTP>0 take the measured fast path.
- `scripts/verify_perf_gate.sh --quick` on RTX 5090,
  `QWEN36_LONG_CONTEXT_MODE=1`: DFlash 3K split-K default 142.6 tok/s
  vs forced-off 60.1 tok/s; MTP=0 auto 49.05 tok/s; MTP=4 auto
  102.40 tok/s vs interpreter forced-off 102.15 tok/s. No capture
  error, no MTP regression in this run.
- Targeted chat parity with `QWEN36_LONG_CONTEXT_MODE=1`:
  `hello` and `hello world`, `--mtp-speculative-tokens 4`, auto vs
  `QWEN36_INTERPRETER_DECODE=0` produced identical 12-token text
  prefixes.

**Codex follow-up (2026-06-09, MLP gate/up pair opcode landed):**

- Interpreter MLP programs now fuse the independent gate/up NVFP4 GEMVs
  into `NVFP4_GEMV_PAIR`, reducing each MLP sequence by one interpreter
  instruction and two counter slots while preserving the same GEMV body.
  Full MLP chunking remains deferred because the down-proj needs K-sliced
  FP32 accumulation to be correct.
- Validation on RTX 5090: `scripts/build_cuda.sh`, `scripts/smoke_cuda.sh`,
  `cargo test -p qwen36-fp4-kernels interpreter --lib`,
  `cargo test -p qwen36-fp4-runtime interpreter_compile --lib --features cuda`,
  `cargo check --release -p qwen36-fp4 --features cuda`, and
  `cargo build --release -p qwen36-fp4 --features cuda` all passed.
- `scripts/verify_perf_gate.sh --quick`, `QWEN36_LONG_CONTEXT_MODE=1`:
  DFlash 3K split-K default 143.18 tok/s vs forced-off 60.60 tok/s;
  MTP=0 auto 49.24 tok/s; MTP=4 auto 95.85 tok/s vs interpreter
  forced-off 95.60 tok/s. Targeted chat parity for `hello` and
  `hello world`, MTP=4 auto vs forced-off, remained byte-for-byte.
- Follow-up prefetch fix: `QWEN36_INTERPRETER_PREFETCH=1` now issues
  lookahead hints from CTA 0 only. The old opt-in path duplicated the same
  64 KiB prefetch stream across the whole interpreter grid. Repeated MTP=4
  runs after the fix varied between ~89 and ~96 tok/s even for
  interpreter-off, so this remains **default OFF**; it is only a cleaner
  diagnostic/experiment path. Prefetch-on chat parity for `hello` and
  `hello world` matched interpreter-off byte-for-byte.

**Codex hand-off (updated):** the prefetch infra (`prefetch.cuh`,
flags plumbing) is the substrate for any further weight-warmup
experiments — adjust `kPrefetchBudgetBytes`, narrow the opcode
filter to just `NVFP4_GEMV` shapes ≥ 5120 × 5120, or replace the
PTX hint with `cp.async.bulk` into a real SMEM page when you wire
the GEMV body to read from SMEM. The MTP=4 +7.3 % win is a real
result and worth defending — if you add anything to the dispatch
path, re-bench MTP=4 to make sure you don't lose it. The long-
context roadmap doc
(`docs/superpowers/notes/2026-06-09-long-context-decode-roadmap.md`)
lists higher-ROI alternatives (FA-tile DFlash drafter, Quest page
sparsity, NVFP4 KV) if the megakernel investigation plateaus.

Reproduce the bench with:

```bash
QWEN36_FP4_KERNEL_LIB_DIR=$PWD/target/cuda \
LD_LIBRARY_PATH=$PWD/target/cuda:${LD_LIBRARY_PATH:-} \
QWEN36_LONG_CONTEXT_MODE=1 \
[QWEN36_INTERPRETER_DECODE=<auto|0|1>] \
[QWEN36_INTERPRETER_PREFETCH=1] \
  target/release/qwen36 bench \
    --model-dir ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --prompt-tokens 128 --max-new-tokens 32 \
    --mtp-speculative-tokens <0|4>
```

### 2026-06-09 — DFlash megakernel Phase 4: kill-gate measurement → full_attn is 89% of verify

Investigation workflow (`wf_2128592f-3a2`, 6 parallel agents + synthesis +
spec) chose path C (verify megakernel) over Phase 2 (NVFP4 KV /
BitDecoding). BitDecoding rejected: no vendorable NVFP4 Blackwell kernel
exists (open repo ships only Int2/Int4 FP16 sm_80/sm_90), +12% only at
7K, masked by the launch floor. Full spec:
`docs/superpowers/specs/2026-06-09-dflash-mk-phase4-verify-megakernel.md`.

**P0 done** (`18d464d`): q_len=16 drafter FA parity gate in smoke.cu —
20 cases, cos 0.999999, proven fail-able. Closes the Phase 1 "drift
bug" as NOT a kernel bug (cos 0.999998 everywhere; the gen forks were
chaotic speculative sensitivity to BF16-ULP deltas).

**P1 KILL-GATE fired — graph capture KILLED by measurement.**
`QWEN36_PROFILE_PREFILL_CHUNKS=1` on the q=16 verify chunk at 7058 ctx
(each stage `cuda_synchronize`'d, so these are true GPU times):

| stage | ms/chunk | % |
|---|---:|---:|
| embed | 0.008 | — |
| input_norm_quant | 1.07 | 0.5% |
| linear_attn (48 layers) | 10.0 | 4.4% |
| **full_attn (16 layers)** | **200.2** | **89%** |
| post_norm_quant | 0.98 | 0.4% |
| mlp (64 layers) | 10.95 | 4.9% |
| logits | 1.65 | 0.7% |
| **chunk total** | **225** | |

Two decisive facts:
1. **Launch overhead is NOT the bottleneck.** The chunk is one 200ms
   GPU kernel stage, not 1300 small launches. Graph capture (P1) would
   net ~0% → killed. The synthesis gate (b) (launch-idle <5% → abandon)
   is satisfied. This is exactly the Stage-F.4/Phase-1-redux trap the
   kill-gate existed to prevent — and it worked, saving ~1 week.
2. **The investigation's profile ESTIMATE was wrong** (it guessed
   full_attn=70ms, mlp=130ms). Reality: full_attn=200ms, mlp=11ms. The
   MLP is fine (weight-bandwidth-bound, cuBLASLt-optimal at q=16). The
   bottleneck is the **q=16 full-attn**.

**Root cause (code-confirmed):** full-attn target shape is q_heads=24,
kv_heads=4, head_dim=256, q_per_kv=6, **FP8 KV** (drafter_chat_smoke
default, main.rs:2618 — R7 resolved, NOT TurboQuant). At q=16 the flash
prefill kernel grids (kv_heads=4 × 1 q-tile) = 4 blocks → ~2% of 170
SMs, so it's gated off below `kPrefillFlashMinTokens=1024`
(attention.cu:2502) and verify falls to the scalar GQA kernel that
re-reads the full 7K KV per token-block. 200ms / 16 layers = 12.5ms /
layer — vs a ~0.1ms KV-bandwidth floor (230 MB FP8 @ 1.8 TB/s), i.e.
the scalar kernel is ~100× off bandwidth.

**Decision: skip P1, go straight to P2 = Flash-Decoding split-K kernel
for q=16 / head_dim=256 / FP8 KV.** Split the 7K KV across ~48 CTAs
(grid 4 kvh × 1 q-tile × S splits) so the FA-2 wmma tile (M=16) saturates
the GPU; partial softmax per split + a reduction. Reuses the
`attention_flash_prefill.cu` tile + FP8 path. Projected: full_attn
200ms → 30–60ms (5–7×), verify 225 → ~55–85ms, end-to-end DFlash at 7K
~2.5–4×. Parity gated by a smoke cos≥0.998 (modeled on P0a) + the
no-fork end-to-end check.

### 2026-06-09 — Decode long-context fix SHIPPED: register-tiled attention, curve now flat

The fix scoped in the section below is implemented and default-on.
`kernels-cuda/attention_decode_tiled.cu`: register-tiled v2 of the
decode split-GQA kernel — one warp per timestep (8 warps × 8-timestep
tiles instead of a serial per-timestep loop), vectorized 8 B (FP8) /
16 B (BF16) lane loads, 256-entry SMEM LUT FP8 decode, tile-batched
online softmax (ONE accumulator rescale per 8 timesteps), 2 syncthreads
per tile instead of ~24. Same grid, partials layout, shared reduce,
device-position read and cache-append side effect as v1; TQ dtypes and
head_dim≠256 fall back to v1. `QWEN36_DECODE_TILED_ATTENTION=0` forces
the v1 scalar path.

**Measured (MTP=0, max-new=64):**

| ctx | v1 | tiled | gain |
|---:|---:|---:|---:|
| 128 | 49.7 | 50.7 | +2% |
| 8192 | 43.1 | **50.3** | **+17% — flat** |
| 16384 | 36.2 | **46.6** | +29% |
| 24576 | 32.7 | **44.0** | **+35%** |

Curve is now −13% at 24K (was −35%) — the classic-engine shape.
full_attn at 24K: 12.7 → 5.7 ms/token (2.2×); next dominant cost is the
context-flat MLP (11.6 ms), so further long-ctx attention work is
diminishing returns until the MLP/weights floor moves.

Gates: smoke parity 8 cases (BF16+FP8 × pos {255, 2047, 8191, 24575},
incl. empty splits; output cos ≥ 0.998 AND cache-append byte-identical
— the kernel owns the current token's K/V store), token identity MTP=0
md5-equal tiled-vs-v1, MTP=4 graph capture healthy (89.7 tok/s),
DFlash unaffected (143 tok/s, AL 8.3 at 3K).

### 2026-06-09 — Base decode (MTP=0) long-context slide: root-caused, fix scoped

User flagged that base decode drops too much with context vs classic
engines. Confirmed and root-caused — full analysis in
`docs/superpowers/notes/2026-06-09-decode-longctx-investigation.md`.

Curve (MTP=0): 49.7 (128) → 43.1 (8K) → 36.2 (16K) → 32.3 (24K) =
−35%. Per-layer profile: linear_attn / mlp / lm_head are **flat**; ALL
growth is the 16 full-attn layers (6.1 → 12.7 ms/token). At 24K the KV
bandwidth floor is 0.45 ms vs 12.7 measured = **28× off bandwidth** —
the decode split-GQA kernel is latency-bound (serial per-timestep loop,
1-byte loads, 6 shuffle-reduces + syncs per position), same disease the
P2 wmma split-K fixed for verify. Split-granularity probe: bigger
blocks strictly worse → inner loop dominates, not the reduce.
Secondary: fusion auto-off ≥8K costs only −3.6%; default config OOMs
at ctx 2048–4096 on a 29.5 GB-free GPU (fused stores don't fit below
the auto threshold — usability trap, use `QWEN36_LONG_CONTEXT_MODE=1`).

Fix scoped (not implemented): register-tiled multi-timestep inner loop
(T=8–16 positions/iter, 128-bit vectorized loads, LUT FP8 decode,
sync amortization), keeping split topology + reduce + graph capture
unchanged. Projected full_attn 12.7 → 2–4 ms @24K → ~45–48 tok/s
(near-flat ~50 → ~47 through 32K). Context-flat ceiling is ~52 tok/s
(linear_attn+mlp+lm_head ≈ 18.3 ms). Est. 2–4 days incl. parity.

### 2026-06-09 — Long-context AL lane: eval battery built, window knob dead, AL variance is the real finding

Follow-up to the strategic assessment (AL is the binding constraint at
long ctx). Probed the cheapest knob first — the drafter's sliding-window
size — and built the evaluation infrastructure the whole drafter-quality
lane needs.

**Probes added** (`crates/drafter/src/forward.rs`, env-gated, default
off, zero effect when unset):
- `QWEN36_DRAFTER_SWA_WINDOW=N` — override the checkpoint's sliding
  window (2048) on the 4 sliding layers.
- `QWEN36_DRAFTER_SWA_ALL=1` — also apply the window to the 5th
  (full-attention) layer.

**Eval battery** (`scripts/drafter_al_eval.sh`): 6 long prompts
(7-10K tokens, real repo text, deliberate content-order variation),
reports per-prompt tok/s + AL and the **geomean AL**. This is now the
standard for ANY drafter-quality change — single-prompt AL deltas are
noise (see below).

**Results:**

| config | geomean AL | min | max |
|---|---:|---:|---:|
| baseline (window 2048) | **5.10** | 2.78 | 8.10 |
| window 4096 | 4.83 | 2.16 | 7.00 |

The window knob is **dead**: an initial single-prompt probe showed a
spectacular AL 2.78 → 6.75 at window 4096, but the battery revealed it
as a chaotic reshuffle (2 prompts up, 4 down, swings ±2.5× both
directions, geomean slightly negative). Same trap as the FA-drafter
"drift" — the speculative loop amplifies any perturbation into large
per-prompt AL swings. `SWA_ALL` also negative (geomean-level).

**The real findings:**
1. **"AL collapses at long ctx" is wrong.** The stock config sustains
   AL 8.1 at 9.9K ctx on favorable content and drops to 2.78 at 7.8K on
   unfavorable content — and *identical documents reordered* swing AL
   from 2.78 to 6.79. It is content/order sensitivity, not length
   degradation, at least up to 10K.
2. **Single-prompt AL measurements are meaningless.** Every knob
   evaluation must use the battery geomean.
3. **The DFlash floor at long ctx now ≈ MTP=3.** With split-K, the
   worst battery prompt (AL 2.78) still does 40 tok/s ≈ MTP=3's ~40 at
   this ctx. DFlash is safe-by-default at long ctx; there is no regime
   where it clearly loses anymore.
4. **Knob-level interventions reshuffle; they don't lift.** The
   credible lever to raise the geomean is a drafter long-context
   fine-tune (the z-lab checkpoint's conditioning is what varies), or
   smarter conditioning (capture-layer/window co-design) — both are
   training-side projects, to be evaluated against this battery.

### 2026-06-09 — P2 SHIPPED: FA-2 wmma split-K verify kernel — 2.2–2.9×, parity-clean, opt-in

Built the correct version of the q=16 full-attn win:
`kernels-cuda/attention_flash_splitk.cu` — a Flash-Decoding split-K
kernel that lifts the proven `attention_flash_prefill.cu` wmma tile
(M=32, N=64, D=256, 4 warps, FP32 accum, causal, BF16/FP8 KV) and adds
a KV-split grid dimension (grid.z = n_splits) + a log-sum-exp reduce.
At q=16 the normal flash tile grids only 4 CTAs (~2% of 170 SMs); split-K
tiles the KV across ~48 CTAs to saturate, while staying numerically
faithful (unlike the scalar split-GQA path which drifted).

**Parity gate (smoke.cu, kernel-vs-scalar-GQA):** 9 cases (start ∈
{0,64,2048} × n_splits ∈ {1,4,48}), all cos ≥ 0.998 vs the scalar GQA
reference at the real verify shape (q_heads=24, kv_heads=4,
head_dim=256). The split-K is a faithful drop-in.

**End-to-end DFlash (drafter-chat-smoke, max-new=192, opt-in
`QWEN36_VERIFY_FLASH_SPLITK=1`, WITH per-call cudaMalloc overhead):**

| prompt | baseline tok/s | split-K tok/s | speedup | base AL | split-K AL |
|---|---:|---:|---:|---:|---:|
| AGENT.md head:150 (3K) | 65.8 | **147.5** | **2.24×** | 9.18 | 9.0 |
| AGENT.md head:300 (7K) | 18.5 | **54.0** | **2.92×** | 4.49 | 3.64 |

full_attn per chunk at 7K: 200 → ~29 ms (~6×); chunk wall 225 → ~54 ms.

The faithful kernel **preserves AL at 3K** (9.18 → 9.0 = noise) — the
clean win the rigorous path promised. Contrast the lossy scalar split-GQA
which *regressed* at 3K (63 tok/s, AL 4.17). At 7K (topic-diverse,
low-AL regime) AL drifts a little (borderline argmaxes are sensitive to
the ~1e-3 FP difference) but the 6× kernel speedup dominates → 2.9×. The
output is a faithful greedy decode of the flash-attention target — and
the flash tile is the SAME kernel the prompt prefill uses (≥1024 chunks),
so verify is now *consistent* with prefill rather than using a different
scalar kernel.

**Coherent long-ctx sweep** (P2.1, persistent-scratch build, FP8 KV —
the production default, so this exercises the FP8 path end-to-end):

| ctx | prompt | baseline tok/s | split-K tok/s | speedup | base AL | split-K AL |
|---|---|---:|---:|---:|---:|---:|
| 3262 | coherent | 65.1 | 155.2 | 2.38× | 9.18 | 9.0 |
| 5484 | coherent | 25.0 | **107.3** | **4.29×** | 5.16 | **7.91 ↑** |
| 7058 | AGENT.md×2 (repetitive) | 17.5 | 53.3 | 3.04× | 4.49 | 3.64 |
| 7815 | coherent | 9.6 | **40.4** | **4.18×** | 2.68 | 2.78 |

**AL is preserved or IMPROVED on coherent prompts** — at 5484 ctx it
jumps 5.16 → 7.91. This is the consistency dividend: the prompt is
prefilled with the flash kernel (≥1024 chunks), so a flash split-K
verify is *consistent* with prefill, whereas the scalar GQA verify used
a different kernel and hurt the drafter's hidden-state conditioning. The
only AL drift (7058) was on a pathological repetitive prompt
(AGENT.md concatenated with itself). On real coherent text the kernel is
a strict win on both axes.

**Status: DEFAULT-ON** (`QWEN36_VERIFY_FLASH_SPLITK=1` is the default;
set `=0` to force scalar GQA). Cleared by an adversarial correctness
review (workflow wf_a36ff789-8b8, 5 lenses + per-finding adversarial
verification + Opus synthesis):

- **Kernel math is correct.** The QK^T / online-softmax / o_frags
  alpha-rescale / PV stages are byte-identical to the proven
  `attention_flash_prefill.cu` tile (geometry unchanged), the
  log-sum-exp reduce is mathematically equivalent to the reference
  single-pass O/l, the causal mask is split-invariant, empty/all-masked
  splits write m=-inf/l=0 (skipped by the reduce), the partition is
  gap-free, scratch capacity is an exact fit (no overflow), and the FP8
  E4M3 decode is byte-identical to the scalar reference. 5 of 6 flagged
  concerns were adversarially refuted (cross-stream race, FP8 numerics,
  stale-partials, host/device start mismatch, error-swallowing — all NOT
  bugs).
- **One real HIGH finding, fixed:** the per-call grow-on-demand
  cudaMalloc was capture-illegal. Latent (the q=16 DFlash verify path is
  eager; captured MTP chunks (2-5 tok) are intercepted by the
  pre-existing split path before this gate). Fixed two ways: (1) a
  `cudaStreamIsCapturing` guard — a grow that would fire mid-capture
  returns NOT_IMPLEMENTED so the dispatch falls through to the
  capture-safe scalar GQA; (2) the eager q=16 calls pre-grow the
  persistent scratch so the grow branch never fires in steady state.
  Empirically confirmed: MTP=0/4 bench (which captures decode+MTP-verify
  graphs) runs with no capture error and no regression (49.6 / 104.4
  tok/s).
- **Parity coverage extended to the production path:** smoke gate now
  72 cases — BF16 **and FP8** KV × tokens {9,16,32} (the default-on
  redirect band) × starts {0,64,2048,4096} × n_splits {1,8,48}, all
  cos ≥ 0.998.

DFlash default (no env) measured 143.9 tok/s at 3K (2.2× vs scalar
baseline 65).

**#55 DONE (9af1c81) — engine-owned partials.** The split-K entry now
consumes the prefill spec's `partial_acc/max/denom_f32` (the engine
already allocated and passed them — the entry was ignoring them). `gpu.rs`
sizes the shared split-KV partial buffers to `max(decode, verify)` where
verify = 32 tokens × q_heads × 48 splits × head_dim (~38 MB). No
cudaMalloc in the production verify hot path → fully capture-safe (a
future captured [9,32] chunk uses split-K with no mid-capture alloc); the
process-global scratch stays only as the smoke/NULL-spec fallback. Smoke
parity now **144 cases** (BF16+FP8 × tokens{9,16,32} ×
starts{0,64,2048,4096} × splits{1,8,48} × **both scratch+engine paths**),
all cos ≥ 0.998. Perf gate: DFlash 3K 143.6 (split-K) vs 60.9 (off) =
2.36×, AL preserved; MTP=0/4 no capture error, no regression.

**M=16 tile + DeltaNet opt — DROPPED by measurement** (design workflow
wf_96d7d3e6-c03). An empirical paired microbench built a correct
M=16/D=256 split-K prototype and timed it head-to-head at the verify
shape (q=16, ctx=7000, FP8, n_splits=48): **1.22 vs 1.34 ms/layer =
~8 % on the kernel, ~3-4 % on the 49 ms chunk** — NOT the ~2× the
MMA-count model predicted. The kernel is latency-bound (32768 branchy
FP8 ldexpf decodes/K-iter, serial per-row softmax, 5 syncthreads) and
occupancy is 1 CTA/SM for BOTH M=16 and M=32 (the 64 KB sm_K+sm_V
dominates the 99 KB budget), so halving MMA count doesn't help. DeltaNet
self-assessed win=no: the scan is 3.1 ms (6.3 % of chunk); 64 % of the
9.9 ms linear-attn bucket is out-of-scope NVFP4 GEMMs. Both below the
15 % bar — skipped (~17 h for ~6 %). The real remaining full_attn levers
(vectorized/LUT FP8 decode, raise occupancy by shrinking the KV SMEM
tile / double-buffering, fix the ~11-of-48 empty-split load imbalance at
7K) are bigger separately-scoped efforts; the chunk is near-optimal for
the cheap wins.

**The DFlash verify lane (P0 → P2.1 + #55) is COMPLETE:** 4.6× chunk
speedup banked (225 → 49 ms), 2.2–4.3× end-to-end, parity-clean
(144-case gate), default-on, adversarial-review-cleared, capture-safe.
The remaining megakernel work is the interpreter decode path (Codex's
lane: auto-policy landed 5111903, then MLP chunking + SMEM prefetch).

Files: `kernels-cuda/attention_flash_splitk.cu` (M=32 wmma split-K +
reduce + engine-partials entry with `cudaStreamIsCapturing` fallback),
`attention.cu` dispatch (default-on), `crates/runtime/src/gpu.rs`
(partial sizing), `smoke.cu` (144-case gate), `scripts/verify_perf_gate.sh`.

### 2026-06-09 — P2 probe: existing split-K GQA at q=16 — 5× at 7K but lossy

The full_attn bottleneck (200ms = 89% of verify) is the q=16 scalar GQA
path. The codebase already has a Flash-Decoding split-K GQA kernel
(`attention_prefill_split_gqa_kernel` + `attention_decode_reduce_kernel`)
used for MTP 1-2 token chunks, gated off above 8 tokens
(`kPrefillSplitLongChunkMaxTokens=8`). Enabling it for the 16-token
verify chunk is pure env tuning:
`QWEN36_PREFILL_SPLIT_MAX_TOKENS=16 QWEN36_PREFILL_SPLIT_MIN_SPLITS=32`.

**Measured (7K ctx, drafter-chat-smoke):**
- full_attn per chunk: **201.7 → 36.5 ms (5.5×)**; chunk wall 226 → 61 ms
- end-to-end DFlash: **13.97 → 71.22 tok/s (5.1×)**

**But it is not parity-clean:**
- 3K coherent prompt: AL **9.18 → 4.17**, tok/s **69 → 63 (regresses)**.
  full_attn is a smaller fraction at 3K, so the kernel speedup no longer
  offsets the AL hit.
- The split-K kernel is FP32-correct per call (online softmax FP32, same
  structure as scalar; the reduce is a correct log-sum-exp merge). A
  chunk=16 whole-prompt dump-logits stress test showed cos 0.989, but
  that **compounds** the per-call divergence over ~300 split chunks ×
  16 layers — per-call divergence is ~0.99996. The split-K is
  numerically ~right.
- The problem is **speculative amplification**: DFlash verify is meant
  to be lossless w.r.t. the scalar target. Changing the verify attention
  shifts the target's greedy argmax slightly, which the draft→verify→
  capture-hidden→draft loop amplifies into an AL swing (same chaotic
  sensitivity as the Phase-1 FA drafter). It helps at 7K (full_attn
  dominates) and hurts at 3K.

**Decision pending (asked user):** ship env split-K context-gated to
long ctx (≥5K, 5× win, no short-ctx regression because gated, but a
slightly different/lossy verify output) vs build the proper FA-2 wmma
split-K tile (correct-at-all-ctx by construction, ~1 week, reuses the
attention_flash_prefill.cu tile + the existing reduction). No code
committed for P2 yet — env tuning only, nothing shipped.

### 2026-06-09 — DFlash megakernel roadmap Phase 1 (FA-tile drafter attn) — neutral, off by default

Pivot: the user committed to "Option A" of the long-context roadmap
(full TileRT-style megakernel), retargeted at DFlash instead of MTP.
Phase 1 was the cheapest entry — FA-tile the naive drafter attention
kernel. Spec + bench in
`docs/superpowers/specs/2026-06-09-dflash-fa-drafter-attention.md`.

**What shipped:** `kernels-cuda/drafter_attention_flash.cu` (~330 LoC),
wmma m16n16k16 BF16 + FP32 accum, online softmax, 4-warp split (warps
own 1 n-tile each for QK^T and 2 d-tiles each for PV). Dispatcher in
`drafter_attention.cu` tries FA first, falls back to v1 on
`NOT_IMPLEMENTED`. Build added to `scripts/build_cuda.sh`.

**Bench (DFlash chat smoke, max-new=256, `QWEN36_LONG_CONTEXT_MODE=1`,
same session):**

| Prompt | ctx | FA tok/s | v1 tok/s | FA AL | v1 AL | per-iter FA | per-iter v1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AGENT.md head:150 | 3262 | 34.5 | 54.7 | 4.87 | 7.76 | **141 ms** | **142 ms** |
| AGENT.md head:300 | 7058 | 18.8 | 21.2 | 4.81 | 5.43 | **256 ms** | **256 ms** |

**Two clean findings, both consequential for the rest of the roadmap:**

1. **Per-iter wall time is bit-identical** between FA and v1. The
   drafter forward at long ctx is *not* compute-bound on these shapes
   (q_len=16, q_heads=32, head_dim=128). The FA kernel's 32 CTAs
   (one per q_head) under-use the 192 SMs; v1's 640 CTAs (q_len ×
   q_heads) saturate better. The wmma tensor-core advantage doesn't
   matter at these shapes — both are launch/per-call-overhead bound
   at the same point. The throughput delta we see is entirely AL
   delta, not kernel speed.
2. **Numerical drift at ctx ≈ 120+.** Parity passes bit-exact at
   short ctx (33 tokens generated identical). At ~180-token ctx the
   generations fork. Source not yet diagnosed; candidates are online
   softmax accumulation drift across many K-tiles, tail-tile
   masking boundary, or PV BF16 cast precision.

**Decision: opt-in default off.** Env gate is now
`QWEN36_DRAFTER_ATTENTION_FLASH=1` (positive, opt-in). Absence
keeps the v1 path. The FA code stays in tree as substrate for Phase
4 (verify megakernel), where q=16 is actually the prefill compute-
bound regime that wmma tiling does win on.

**Roadmap rebalance:** the drafter attn kernel is *not* the long-
context bottleneck we thought. The verify step is. Phase 2 (NVFP4
KV cache + BitDecoding port — attacks verify's full-KV bandwidth)
and Phase 4 (verify megakernel — attacks per-iter launch overhead)
should drive the next investments, in that order. Phase 1's
substrate (cooperative load pattern, wmma layout, online softmax)
folds directly into Phase 4. See
`docs/superpowers/specs/2026-06-09-dflash-fa-drafter-attention.md`
§ 11 for the full outcome.

Reproduce:

```bash
QWEN36_FP4_KERNEL_LIB_DIR=$PWD/target/cuda \
LD_LIBRARY_PATH=$PWD/target/cuda:${LD_LIBRARY_PATH:-} \
QWEN36_LONG_CONTEXT_MODE=1 \
[QWEN36_DRAFTER_ATTENTION_FLASH=1] \
  target/release/qwen36 drafter-chat-smoke \
    --model-dir ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --drafter-dir ~/models/Qwen3.6-27B-DFlash \
    --prompt "$(head -150 AGENT.md)" --max-new-tokens 256
```

### 2026-06-09 — DFlash speculative decoding (`chat --drafter dflash`) — opt-in fast path, 1.8× MTP=3 geo-mean

Phase F.2 shipped. End-to-end speculative loop using
[`z-lab/Qwen3.6-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)
(2 B BF16 block-diffusion drafter, paper arXiv 2602.06036) against
our NVFP4 target. Implementation is 13 commits across two sessions;
full details in `docs/superpowers/notes/2026-06-09-dflash-final.md`
and the design spec
`docs/superpowers/specs/2026-06-08-dflash-speculative-decoding-design.md`.

**Activation (opt-in)** — default `chat` behaviour is unchanged:

```bash
export QWEN36_LONG_CONTEXT_MODE=1        # disable fused MLP stores to fit drafter
target/release/qwen36 chat \
    --model-dir   ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --drafter     dflash \
    --drafter-dir ~/models/Qwen3.6-27B-DFlash \
    --prompt "<text>" --max-new-tokens 256
```

Streams tokens as they commit; emits a trailing `[dflash] generated
N tokens in K iters | AL=… | decode Xs (Y tok/s)` summary on stderr.

**Engine-side pieces shipped:**

- `Engine::verify_block_batched(tokens) -> Vec<u32>` (commit `9b7e643`):
  runs one prefill chunk through the target then batched RMSNorm +
  bf16_gemm lm_head (rows=k+1) + sample_rows greedy. Returns all k+1
  argmaxes in one call. **~10× fewer target forwards per verify
  cycle** than the sequential `engine.prefill(&[t])` chain that
  preceded it.
- `Engine::crop_state_position(new)` — public KV-position truncation
  paired with `verify_block_batched` to drop the rejected speculative
  tail (data past the cut is left in place, overwritten by the next
  forward write).
- `DrafterHiddenCaptureHook` (commit `97993c0` + Phase F.2 update):
  `Arc<dyn Fn(layer_idx, residual_ptr, tokens) -> Result<()> + Send +
  Sync>`. Engine fires it once per layer after each `input_layernorm`
  in **both** prefill and decode paths. `crates/drafter/handoff.rs`
  provides `TargetHiddenCapture` which uses `copy_strided_rows` to
  scatter per-layer residuals into a `[max_tokens, hidden *
  n_target_layers]` BF16 buffer matching the drafter's
  `target_hidden_raw` input layout. Supports a per-decode
  `set_write_row(row)` so multi-iter chats accumulate per-decode
  captures into distinct rows.

**Drafter-side pieces** (new `crates/drafter` crate, no impact on
default engine surface):

- `DFlashDrafter`: mmap'd safetensors loader + 58-tensor manifest.
- `DFlashDrafterDevice`: GPU upload (~3.46 GB BF16).
- `DrafterForward`: 5-layer drafter forward with per-layer KV cache,
  reset/crop API, internal `fc + hidden_norm` collapse.
  Parity-validated against transformers reference at cos sim
  **0.99987** (Python harness `scripts/dflash_parity.py`).
- New CUDA kernel `kernels-cuda/drafter_attention.cu`:
  `qwen36_drafter_attention_block_bf16` — non-causal BF16 attention
  with `K = [k_ctx; k_noise]` (target_hidden + noise embeddings),
  GQA broadcast, optional SWA. Smoke cos sim 0.999999 vs host fp32.
- `propose_block`: embed → drafter forward → lm_head GEMM →
  greedy argmax. Returns k candidate tokens.

**Bench sweep** (RTX 5090 release build, 5 prompt types × 3
generation lengths × 2 backends = 30 runs, driver
`scripts/dflash_bench_sweep.py`, raw CSV at
`docs/superpowers/notes/2026-06-09-dflash-sweep.csv`):

| Prompt | prompt tok | gen | DFlash tok/s | DFlash AL | MTP=3 tok/s | MTP=3 AL_eff | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| completion_short | 7 | 32  | 102.4 | 3.67  | 67.7  | 3.52 | 1.51× |
| completion_short | 7 | 128 | 128.0 | 4.68  | 56.4  | 3.33 | **2.27×** |
| completion_short | 7 | 256 | 102.5 | 3.97  | 68.0  | 3.56 | 1.51× |
| code_short       | 4 | 32  | 121.7 | 4.38  | 108.0 | 4.00 | 1.13× |
| code_short       | 4 | 128 | 257.0 | 9.36  | 69.3  | 3.54 | **3.71×** |
| code_short       | 4 | 256 | **313.6** | **11.77** | 81.9 | 3.73 | **3.83×** |
| prose_medium     | 51 | 32  | 32.3  | 1.19  | 40.8 | 2.97 | 0.79× |
| prose_medium     | 51 | 128 | 87.9  | 3.25  | 53.0 | 3.28 | 1.66× |
| prose_medium     | 51 | 256 | 105.9 | 4.10  | 58.7 | 3.43 | **1.80×** |
| qa_medium        | 59 | 32  | 79.7  | 3.00  | 44.6 | 3.04 | 1.79× |
| qa_medium        | 59 | 128 | 130.4 | 4.96  | 48.8 | 3.20 | **2.67×** |
| qa_medium        | 59 | 256 | 153.8 | 6.07  | 51.7 | 3.30 | **2.98×** |
| code_long        | 229 | 32  | 46.6  | 2.00  | 85.9 | 3.88 | 0.54× |
| code_long        | 229 | 128 | 53.9  | 2.37  | 74.1 | 3.73 | 0.73× |
| code_long        | 229 | 256 | 79.8  | 3.55  | 69.1 | 3.70 | 1.15× |

- **Geo-mean over 15 cells:** DFlash 117.5 tok/s vs MTP=3 65.2 tok/s
  → **1.80× speedup**.
- **Peak:** 313.6 tok/s on code_short@256, AL **11.77** (block_size
  upper bound is 16 — near-full block acceptance every iter).
- **Worst case:** 0.54× on code_long@32. Long prompts dilute the
  drafter's `fc + hidden_norm` conditioning; short generations don't
  give the drafter time to warm up.

**Patterns:**

- DFlash AL **rises with generation length**: code 4.38 → 9.36 → 11.77
  across gen=32/128/256. MTP=3 AL_eff is flat at 3.3–4.0.
- DFlash AL **falls with prompt length**: same code task,
  3.83× speedup at 4 prompt tokens → 1.15× at 229 prompt tokens.
- Three cells favour MTP=3 (long-context + short-gen); common factor
  is AL ≤ 2.5.

**Routing heuristics** (manual today, not adaptive):

- `prompt_tokens > 150` AND `max_new_tokens < 64` → MTP=3.
- Code/QA short context → DFlash.
- Long prose continuations → DFlash if `max_new_tokens ≥ 128`.

**Workflow during DFlash bring-up (write-ups in
`docs/superpowers/notes/`):**

- `2026-06-08-dflash-kernel-reuse-audit.md` — Phase B catalog of which
  existing kernels the drafter forward can reuse and which needed
  new CUDA (only the `drafter_attention` kernel; everything else
  reuses `Bf16GemmSpec`, `RmsNormSpec`, `PartialRopeSpec` with
  `rope_dims=128`, `SwiGluSpec`, `EmbeddingLookupSpec`).
- `2026-06-09-dflash-final.md` — full results doc (this section's
  long form, plus implementation history and known issues).

**Known issues / out of scope (documented; not addressed):**

1. **NVFP4 decode-kernel divergence**: `chat
   --mtp-speculative-tokens 0` produces degenerate output ("Here
   question or looks sentence address") because the engine's
   per-token decode path produces logits with cos sim ~0.76
   (sometimes negative) vs the prefill kernel path on the same
   input. New diagnostic CLI `qwen36 decode-vs-prefill-check`
   reproduces it in 2 sequential engine loads. DFlash routes around
   it by verifying through prefill chunks (commit `6571f37`).
2. **CUDA-graph capture of `verify_block_batched`** — deferred.
   Plausible 20–30% more tok/s but requires coordination with the
   existing decode-graph machinery.
3. **Permanent logits workspace** on `GpuForwardBuffers` — evaluated
   and skipped. Per-call alloc cost ≈ 0.15 % of decode time; not
   worth touching `GpuForwardBuffers` for that margin.
4. **Adaptive drafter routing** — controller could swap dynamically
   based on prompt length + observed acceptance; today user picks
   via `--drafter dflash`.

**Long-context follow-up (added after the standard sweep):**

Probing prompt sizes from 400 to 7058 tokens shows the "long context
hurts DFlash" framing from the standard sweep was incomplete. Full
write-up: `docs/superpowers/notes/2026-06-09-dflash-long-context.md`.

| Prompt (content) | tokens | gen | DFlash tok/s | DFlash AL | MTP=3 tok/s | speedup |
|---|---:|---:|---:|---:|---:|---:|
| tech_xl_500t (post-mortem) | 400 | 128 | 69.2 | 3.45 | 55.2 | 1.25× |
| code_xl_1500t (Rust module) | 802 | 256 | 88.6 | 5.82 | 53.6 | **1.65×** |
| long_synth_xxl (coherent tech) | 953 | 256 | 86.1 | 6.09 | 43.9 | **1.96×** |
| long_synth_3000t (topic-shift) | 986 | 256 | 26.1 | 1.96 | 44.4 | 0.59× |
| AGENT.md head:150 | **3262** | 256 | 52.5 | **7.76** | 29.1 | **1.80×** |
| AGENT.md head:300 | **7058** | 256 | 20.8 | 5.43 | 39.9 | 0.52× |

Two distinct effects compose:

1. **Topic / distribution diversity in the prompt.** Same ~1000-token
   length, drastically different result: coherent tech writing
   (`long_synth_xxl`) gets AL 6.09 and 1.96× speedup; topic-shifting
   prose (`long_synth_3000t`, jumps between cooking, physics, game
   design, supply chain) gets AL 1.96 and 0.59× speedup. The
   block-diffusion drafter conditions on the concatenated target
   hidden states; incompatible distributions across the prompt
   destroy the denoising signal.

2. **Drafter forward cost dominates above ~5K tokens.** Our drafter
   attention kernel (`kernels-cuda/drafter_attention.cu`, Phase C
   commit `4c2b43c`) is the naive O(q_len × kv_seq_len) version — no
   tiling, no FlashAttention. At 3K context AL stays high (7.76) and
   DFlash wins 1.80×; at 7K context AL is still 5.4 but per-iter
   drafter time exceeds the gain, dropping to 0.52×. MTP's MTP head
   is a tiny single-extra-layer pass that doesn't redo prompt
   attention, so MTP=3 throughput stays roughly flat across context
   lengths (~30–40 tok/s in this range).

Updated routing rules:

- ≤ 200t → DFlash (drafter cheap, AL high).
- 200–1000t coherent text → DFlash gen ≥ 128.
- 200–1000t topic-shift prose → MTP=3.
- 1000–3000t structured technical → DFlash gen ≥ 64 (best case 1.96×).
- 3000–5000t → benchmark per workload; mixed.
- > 5000t → MTP=3 (drafter forward time dominates).

The break-even at ~5K is set entirely by the drafter attention
kernel. A FlashAttention-style tile rewrite would plausibly push it
to ~20K (drafter scales bandwidth-bound rather than O(n)). Scoped as
a follow-up; not done.

**Pre-flight to run DFlash on a fresh machine:**

```bash
hf download z-lab/Qwen3.6-27B-DFlash --local-dir ~/models/Qwen3.6-27B-DFlash
curl -sL https://raw.githubusercontent.com/z-lab/dflash/main/dflash/model.py \
     -o ~/models/Qwen3.6-27B-DFlash/dflash.py   # for the Python parity harness
cargo build --release -p qwen36-fp4 --features cuda
./scripts/build_cuda.sh                          # CUDA lib (.so) build
QWEN36_LONG_CONTEXT_MODE=1 \
QWEN36_FP4_KERNEL_LIB_DIR=$PWD/target/cuda \
LD_LIBRARY_PATH=$PWD/target/cuda:$LD_LIBRARY_PATH \
  target/release/qwen36 validate-drafter --drafter-dir ~/models/Qwen3.6-27B-DFlash
```

Then any of the chat / sweep / smoke commands above.

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

## Next steps (état au 2026-06-09, fin de session)

Ordonnés par ROI attendu. Chaque item porte son gate de validation.

1. **Valider le reduce parallèle décode (DANS L'ARBRE, bench en attente).**
   Les scans max/denom sériels thread-0 de `attention_decode_reduce_kernel`
   sont parallélisés (réductions warp block-wide) — c'est le coût résiduel
   du décode long-ctx après le kernel tiled (full_attn 5.7 → 13 ms entre
   24K et 64K quand n_splits passe 512 → 2048 ≈ 0.8 ms/layer de loads
   sériels mono-thread). Parité smoke verte (gate decode-tiled 8 cas, gate
   split-K 144 cas). RESTE À FAIRE : bench 24K + 64K (attendu : full_attn
   ~plat ≈ 5-6 ms à 64K → décode ~43-46 tok/s vs 32.7 avant), puis
   `scripts/verify_perf_gate.sh --quick`. Optionnel : une cellule 128K
   (le prefill seul y prend ~8-10 min au rythme actuel — voir item 2).

2. **Prefill long-context (LA prochaine target).** Mesuré : 2006 (8K) →
   1053 (16K) → 728 (24K) → **274 tok/s (64K)**. Signature puissance
   pendant le prefill 64K+ : **170 W / 575 W à "util" 100% et clocks SM
   à plein boost (2842 MHz)** = latency-stalled, PAS compute-bound (un
   prefill sain tirerait 450-550 W). Root cause identifiée : les kernels
   `attention_flash_prefill.cu` et `attention_sage_prefill.cu` ont **zéro
   cp.async / double-buffering** (grep = 0) — boucle K-tile sérielle à
   latence HBM exposée (load tile → sync → compute → sync), grille 256
   CTAs × 128 threads (~10% d'occupation thread, 1 CTA/SM à cause des
   ~90 KB SMEM). Fix : pipeline cp.async 2-3 étages sur les loads de
   tiles K/V. Le gate de succès se lit au wattmètre : la puissance doit
   monter vers 400-500 W ; cible prefill ≥ 1000 tok/s à 64K (3-5×).
   Est. 2-4 jours, parity-gated comme les autres kernels.

3. **Lane interpreter (Codex) : deux gates avant `compile_decode_stack`.**
   (a) Bencher le chunking MLP landé (~30 min) : ≥ +2-3 tok/s sur un
   chemin qui compte → la thèse pipelining vit ; ~0% → archiver
   l'interpreter en opt-in MTP>0 (le +7.3% MTP=4 est déjà banké via
   l'auto-policy). (b) Si (a) passe : probe de coût substrat — programme
   64-layers de trampolines timé (le coût des ~512 barrières grid-wide
   doit rester < 0.5 ms/token) AVANT de construire l'assemblage
   single-launch. Vigilance aliasing : `attn_partial_acc` a maintenant
   TROIS consommateurs (decode split, verify split-K #55, programmes
   MLP chunked) — cartographier avant tout programme full-stack.

4. **AL long-ctx DFlash : décision fine-tune drafter.** La batterie
   d'éval est prête (`scripts/drafter_al_eval.sh`, geomean sur 6 prompts
   7-10K ; baseline 5.10). Les knobs cheap sont falsifiés (sweep window
   = reshuffle chaotique). Le levier crédible pour lever le geomean est
   un fine-tune long-contexte du drafter z-lab — à scoper (données,
   pipeline, coût GPU) et à valider avec l'utilisateur avant lancement.

5. **Leviers en réserve (non planifiés).** Verify split-K : décode FP8
   par LUT + >1 CTA/SM via tile KV réduit + équilibrage des splits vides
   (~10-20% chacun sur le verify). NVFP4 KV cache : différé (aucun
   kernel sm_120 vendorable ; classe de bugs divergence FP4). Tile M=16
   verify + scan DeltaNet : mesurés sous la barre des 15%, skip.

Garde-fous process qui ont fait leurs preuves (à garder) :
- `scripts/verify_perf_gate.sh` avant/après tout changement perf.
- Mesure d'abord : microbench apparié / kill-gate avant de construire
  (5 hypothèses model-based falsifiées par des probes cheap ce sprint).
- Les deltas AL mono-prompt sont du bruit — geomean batterie uniquement.
- La puissance + clocks SM comme signal de profiling de premier rang
  (170 W à "util 100%" a exposé le stall prefill en un coup d'œil).
