# Code inventory — what exists, what runs, what is dead

> **Read this before building anything new.** This repo has been developed largely by AI
> agents, and the same code has been rediscovered or rebuilt more than once. This file is
> the authoritative map of (1) everything that has been developed, (2) what the default
> hot path actually executes, (3) what is opt-in, archived-negative, or dead, and (4) every
> option/env var that toggles a mode. `DAILY.md` is the chronological lab journal (`AGENT.md` holds instructions only); this
> file is the consolidated state. If you change a default, a dispatch condition, or add an
> env var, **update this file in the same commit.**
>
> Last full audit: 2026-06-10 (HEAD `0f91793`); dead-code purge + CPU CI added
> the same day on `chore/rationalization` (see §2.4 for what was removed).

## 1. Main use case

Single-stream inference of `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` on one RTX 5090
(SM120), via the `qwen36` binary:

```bash
qwen36 chat  --model-dir <m> --prompt "<text>" --max-new-tokens 256 [--mtp-speculative-tokens 0..4]
qwen36 bench --model-dir <m> --prompt-tokens N --max-new-tokens M [--prompt-file <f>] [--mtp-speculative-tokens 0..4]
```

Three production decode modes, in order of typical throughput:

| Mode | Activation | Status |
|---|---|---|
| Base decode (MTP=0) | default | captured CUDA graph, ~50 tok/s, near-flat to 24K ctx |
| Chain MTP 1–4 | `--mtp-speculative-tokens N` | ~2.3× at MTP=4 full-accept; interpreter auto-enables |
| DFlash speculative | `--drafter dflash --drafter-dir <d>` + `QWEN36_LONG_CONTEXT_MODE=1` | 1.8× geo-mean vs MTP=3, up to 3.8× on code; loses >5K-token prompts with short gen |

Everything else in the repo is support (loading, validation, parity, profiling) or
archived experiments.

## 2. Component status map

Legend: **DEFAULT** = runs on the main path with no env vars. **OPT-IN** = functional,
activated by flag/env. **NEG** = built, benchmarked negative/neutral, kept in tree default-off.
**DEAD** = unreachable / no call site. **INFRA** = no direct value, kept as substrate.

### 2.1 Default hot path

| Component | Files | Notes |
|---|---|---|
| Engine (prefill chunked + decode CUDA graph) | `crates/runtime/src/engine.rs` | the core; graph captured at first decode, replayed after |
| Fused weight stores (gate+up MLP; DeltaNet 4-way in_proj) | `crates/runtime/src/gpu.rs` (`MlpFusedStore`, `LinearAttnInProjFusedStore`) | ON only when `max_context < 8192` (auto long-context mode disables them to save ~8 GB VRAM) |
| NVFP4 GEMM via cuBLASLt | `kernels-cuda/nvfp4_gemm.cu` | prefill GEMMs + decode fallback shapes |
| Hand-rolled NVFP4 decode GEMV ("Direction B") | `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` | **default ON** for n==1, m%16==0, k%512==0 shapes; +14.5% MTP=0. Kill: `QWEN36_DECODE_GEMV_DISABLE=1` |
| Sage INT8 prefill attention | `kernels-cuda/attention_sage_prefill.cu` | **default ON** for chunks ≥1024 tokens (see §3) |
| Flash BF16 wmma prefill attention | `kernels-cuda/attention_flash_prefill.cu` | next in dispatch chain; also the tile reused by split-K verify |
| FA-2 split-K verify attention | `kernels-cuda/attention_flash_splitk.cu` | **default ON** for 2–32-token chunks (MTP/DFlash verify); 2.2–4.3× end-to-end DFlash, 144-case parity gate |
| Register-tiled split decode attention (v2) | `kernels-cuda/attention_decode_tiled.cu` | **default ON** on the split-KV decode path; long-ctx curve now flat (+35% at 24K). Kill: `QWEN36_DECODE_TILED_ATTENTION=0` |
| Scalar GQA / per-q-head attention (v1) | `kernels-cuda/attention.cu` | fallback for short chunks, TQ dtypes, head_dim≠256 |
| DeltaNet decode + chunked prefill | `kernels-cuda/deltanet.cu`, `deltanet_prefill.cu` | 48 linear-attention layers; chunked prefill default ON |
| Elementwise/fused ops | `kernels-cuda/ops.cu` | rmsnorm (+nvfp4-quantize), swiglu (+nvfp4-quantize), conv1d (+gdn-gate fused), sampling, embedding, q_proj deinterleave/gate |
| Partial RoPE | in `ops.cu` / `crates/kernels/src/rope.rs` | 64 of 256 dims |
| FP8 KV cache | engine + attention kernels | default KV dtype (`EngineConfig::default`, `engine.rs:602`) |
| MTP chain controller + verify graphs | `crates/mtp/src/lib.rs`, engine | opt-in by CLI flag but fully on the supported path; MTP=1 two-token graph, MTP=2/3/4 chunked verify + multi-graph |
| Decode interpreter (whole-layer fused launches) | `kernels-cuda/interpreter/`, `crates/kernels/src/interpreter.rs`, `crates/runtime/src/interpreter_compile.rs` | **AUTO**: enabled iff `mtp_speculative_tokens > 0` (+7.3% MTP=4), disabled at MTP=0 (would regress −5%). `QWEN36_INTERPRETER_DECODE=0|1|auto` |

### 2.2 Opt-in, functional

| Component | Activation | Files | Notes |
|---|---|---|---|
| DFlash drafter (block-diffusion speculative decoding) | `chat --drafter dflash --drafter-dir <d>`, requires `QWEN36_LONG_CONTEXT_MODE=1` (VRAM) | `crates/drafter/*`, `kernels-cuda/drafter_attention.cu` | z-lab 2B BF16 drafter; verify goes through prefill chunks (split-K kernel). AL is content-sensitive, not length-sensitive (eval battery: `scripts/drafter_al_eval.sh`, geomean baseline 5.10) |
| TurboQuant 3/3.5 KV cache | `QWEN36_KV_CACHE_DTYPE=tq3|tq35` | `kernels-cuda/turboquant.cu` | int-quantized KV; TQ dtypes exclude flash/sage/tiled kernels (scalar paths only) |
| Tree-MTP (K>1 leaves) | `--mtp-tree-leaves K` | `crates/mtp`, engine, top-K kernel, tree-mask attention | **functional but 2.7–4× SLOWER than chain** (leaves go through single-token forwards). Phase-2 (batched leaf forward) never built. Default K=1 = chain. |
| GQA tiled prefill (2/4-token tiles) | `QWEN36_PREFILL_GQA_TILE2=1` / `QWEN36_PREFILL_GQA_TILE_TOKENS=2\|4` | `attention.cu` | experimental first step toward tiled prefill; superseded in practice by flash/sage |
| MTP device-side chain | `QWEN36_MTP_DEVICE_CHAIN=1` (+`_BATCH`, default 2) | engine | experimental alternative to host-launched draft chain |
| Bench real-text prompts | `bench --prompt-file benches/data/long_prompt_{4k,8k}.txt` | CLI | **use this for MTP-acceptance numbers** — the synthetic single-token default is adversarial for the MTP head |

### 2.3 Archived-negative (in tree, default OFF — do not re-enable without a new idea)

| Component | Opt-in | Files | Measured result |
|---|---|---|---|
| Productive spin / L2 prefetch on idle SMs | `QWEN36_PRODUCTIVE_SPIN=1` (+`_CTAS`, default 128) | `kernels-cuda/decode_gemv/l2_prefetch.cu`, `DecodeAuxStreams` in engine | ≤+0.5% = noise (2026-05-19). Decode graph already keeps weights L2-resident |
| Interpreter L2 prefetch lookahead | `QWEN36_INTERPRETER_PREFETCH=1` | `kernels-cuda/interpreter/prefetch.cuh` | negative on both MTP=0 and MTP=4 (2026-06-09) |
| FA-tiled drafter attention | `QWEN36_DRAFTER_ATTENTION_FLASH=1` | `kernels-cuda/drafter_attention_flash.cu` | per-iter parity with v1 (not compute-bound) + numerical drift at ctx≳120 degrading AL (2026-06-09) |
| Drafter sliding-window knobs | `QWEN36_DRAFTER_SWA_WINDOW=N`, `QWEN36_DRAFTER_SWA_ALL=1` | `crates/drafter/src/forward.rs` | dead lever: geomean-negative chaotic reshuffle (2026-06-09) |

### 2.4 Dead code

**Removed 2026-06-10** (branch `chore/rationalization` — recover from git history if ever needed):

- **Mirage CUTLASS megakernel GEMM** (`kernels-cuda/megakernel/nvfp4_matvec_sm120.cu` + stub, `QWEN36_USE_MEGAKERNEL_GEMM`): never executed — the SM120 body was guarded by `CUTLASS_ARCH_MMA_SM120_SUPPORTED` without including the header defining it, so every call silently fell back to cuBLASLt. Its "validated parity" claims tested cuBLASLt against itself. Superseded by the Direction B gemv. Analysis kept in `docs/mirage-megakernel.md`.
- **Per-block megakernel, all stages** (`kernels-cuda/megakernel/full_attn_block_sm120.cu`, `QWEN36_MEGAKERNEL_FULL_ATTN_STAGE_F4`): only Stage F.4 was wired and it benched **−4.0% MTP=0**; the other stages had no engine call site. The persistent-grid + work-stealing + spinlock-barrier pattern (and its two bring-up bugs) is documented in `DAILY.md` § 2026-05-23.
- **Triton AOT placeholder** (`scripts/triton_aot.py`): never implemented.

Still in tree:

| Component | Files | Why dead | Recommendation |
|---|---|---|---|
| L2 access-window primitives | `crates/kernels/src/memory.rs` (`cuda_set/clear_l2_access_window`), `kernels-cuda/runtime.cu` | wired end-to-end, **zero call sites**; pinning DeltaNet state gave no gain (graph replay already keeps it hot) | keep as primitive; don't rediscover |

### 2.5 Tried and reverted / falsified — do NOT rebuild without addressing the recorded blocker

Full details in `DAILY.md` (dated sections):

- **Full-attn Q/K/V fusion** — parity OK but VRAM pressure destabilized cuBLASLt plan caching (−10–20% MTP=2/3). Needs dropping originals post-fuse first.
- **Single-block GQA decode kernel** (non-split path) — −5%; the split-GQA kernel already covers the win regime.
- **Tree-MTP K>1 as shipped** — −10× per-cycle overhead; profitable only with a batched leaf forward (tree-mask attention + batched DeltaNet), never built.
- **TMA multicast, persistent grid for gemv, sub-byte LDSM** — investigated, blocked or <2% (see `docs/superpowers/notes/2026-05-04-direction-b-cutlass-blockers.md`).
- **M=16 verify tile, DeltaNet scan optimization, BitDecoding/NVFP4-KV port** — measured/assessed below the 15% bar (2026-06-09).
- **Lossy env-tuned split-K verify** (`QWEN36_PREFILL_SPLIT_MAX_TOKENS=16`) — 5× at 7K but AL-destroying at 3K; superseded by the faithful `attention_flash_splitk.cu`.
- **KVarN (huawei-csl) KV-cache quantization** — evaluated 2026-06-10, NOT integrated: it is a vLLM fork (Python/Triton), KV is not the bottleneck at ≤24K (already FP8 + in-house TQ3/TQ35), and sub-4-bit V risks the known speculative-loop AL amplification. If the B1 lane (aggressive KV quant for 64K–262K) is reopened, its recipe (asymmetric K4/V2 RTN + Hadamard rotation, 128-token tiles, calibration-free) is the reference to port — see `DAILY.md` § 2026-06-10.
- **Synthetic-prompt MTP acceptance dips** (acc 0.84 at 4K) — artefact of single-token-repeat bench prompts; falsified as a kernel bug. Use `--prompt-file` or `chat`+`QWEN36_MTP_STATS=1`.

### 2.6 Known open issues

- **Per-token decode vs prefill logits divergence** (cos ~0.76 on some inputs) — diagnostic: `qwen36 decode-vs-prefill-check`. DFlash routes around it by verifying through prefill chunks.
- **MTP≥1 chunked verify is not bit-equal to MTP=0** — 1–2 token flips on borderline-argmax prompts. Parity floor = `hello` / `hello world` gates.
- **Long-context prefill is latency-stalled** (274 tok/s at 64K, 170 W) — flash/sage prefill kernels have no cp.async pipelining. This is the current top optimization target (`DAILY.md` § Next steps).
- **Default config OOM trap**: `EngineConfig::default()` reserves 262K context; fused stores don't fit at intermediate `max_context` (2–4K) on a busy GPU → use `QWEN36_LONG_CONTEXT_MODE=1`.

## 3. Attention dispatch reference

The attention kernel choice is made **inside the CUDA entry points** (`kernels-cuda/attention.cu`),
not in Rust — grep there, not in `engine.rs`, when wondering which kernel runs.

**Prefill / verify-chunk entry `qwen36_attention_prefill` (attention.cu ~2400+), in priority order:**

1. Engine-requested split-KV prefill (n_splits set by engine; used for short MTP verify chunks; bounded by `kPrefillSplitLongChunkMaxTokens=8` unless env-overridden).
2. **Sage INT8 prefill** — default ON; needs tokens ≥ 1024, head_dim 256, no tree mask, non-TQ KV. (`QWEN36_ATTENTION_SAGE_PREFILL=0` → next.)
3. **Flash BF16 wmma prefill** — default ON, same eligibility. (`QWEN36_ATTENTION_FLASH_PREFILL=0` → next.)
4. GQA tile2/4 kernel — only if `QWEN36_PREFILL_GQA_TILE2/…_TOKENS` opt-in.
5. **FA-2 split-K verify** — default ON for 2 ≤ tokens ≤ 32, head_dim 256, non-TQ (the MTP / DFlash q=16 verify shape). (`QWEN36_VERIFY_FLASH_SPLITK=0` → next.)
6. Scalar GQA kernel (tokens ≥ 16).
7. Per-q-head scalar kernel (final fallback; also the TQ and tree-mask path).

**Decode entry `qwen36_attention_decode` (attention.cu:2716):**

1. Split-KV path when the engine passes `decode_n_splits ≥ 2` (sized from the context bucket, min 8192 — `QWEN36_DECODE_ATTENTION_BUCKET_MIN_CONTEXT`):
   - **register-tiled v2** (`attention_decode_tiled.cu`) when n_splits ≥ 32, head_dim 256, BF16/FP8 — default ON;
   - v1 scalar split-GQA kernel otherwise / on `QWEN36_DECODE_TILED_ATTENTION=0`;
   - followed by `attention_decode_reduce_kernel` (scans parallelized 2026-06-09, bench pending).
2. Non-split per-q-head decode kernel for short contexts / TQ dtypes.

**Drafter attention** (`drafter_attention.cu`): naive v1 by default; FA-tiled variant opt-in (negative, §2.3).

## 4. Environment variable reference (complete, 88 vars)

Defaults verified in code 2026-06-10. "bool" vars accept `1/true/yes/on`.

### 4.1 Kernel-path toggles (read in CUDA code via `getenv`)

| Var | Default | Effect |
|---|---|---|
| `QWEN36_ATTENTION_SAGE_PREFILL` | **1** | `0` falls back to flash BF16 prefill (`attention.cu:2565`) |
| `QWEN36_ATTENTION_FLASH_PREFILL` | **1** | `0` falls back to scalar GQA prefill (`attention.cu:2578`) |
| `QWEN36_VERIFY_FLASH_SPLITK` | **1** | `0` forces scalar GQA for 2–32-token verify chunks (`attention.cu:2655`) |
| `QWEN36_DECODE_TILED_ATTENTION` | **1** | `0` forces v1 scalar split decode kernel (`attention.cu:2777`) |
| `QWEN36_PREFILL_GQA_TILE2` | 0 | opt-in 2-token GQA prefill tile (`attention.cu:2606`) |
| `QWEN36_PREFILL_GQA_TILE_TOKENS` | unset | `2\|4` selects the tile size directly |
| `QWEN36_DRAFTER_ATTENTION_FLASH` | 0 | opt-in FA-tiled drafter attention — known drift, neutral perf (`drafter_attention_flash.cu:335`) |

### 4.2 GEMM/GEMV selection (Rust, `crates/kernels/src/backend.rs`)

| Var | Default | Effect |
|---|---|---|
| `QWEN36_DECODE_GEMV_DISABLE` | off | kill-switch for the hand-rolled NVFP4 decode gemv → cuBLASLt |
| `QWEN36_DECODE_GEMV` | on | back-compat: `=0` same as DISABLE |

### 4.3 Decode interpreter (Rust, `engine.rs:99–440`, `interpreter.rs:677`)

| Var | Default | Effect |
|---|---|---|
| `QWEN36_INTERPRETER_DECODE` | **auto** | master: `auto` = ON iff MTP>0; `1` force on; `0` force off |
| `QWEN36_INTERPRETER_OPCODES_ENABLED` | all | CSV allow-list of opcodes (bring-up/diagnostics) |
| `QWEN36_INTERPRETER_PREFETCH` | 0 | L2 lookahead prefetch — measured negative |
| Fine-grained enable gates (force a slice on even when master off): `QWEN36_INTERPRETER_{LOGITS, MLP, MLP_CHUNKED, NORM_MLP, RMSNORM, DELTANET, ATTN, ROPE, FULL_ATTN, FULL_ATTN_LAYER, FULL_ATTN_INPUT_LAYER, FULL_TRANSFORMER_LAYER, LINEAR_ATTN_TAIL, LINEAR_ATTN_POST_INPROJ, LINEAR_ATTN_LAYER, LINEAR_ATTN_INPUT_LAYER, LINEAR_TRANSFORMER_LAYER}` | 0 | diagnostics / bisection |
| Per-layer-type kill switches: `QWEN36_INTERPRETER_{FULL_ATTN_INPUT_LAYER, FULL_TRANSFORMER_LAYER, LINEAR_ATTN_LAYER, LINEAR_ATTN_INPUT_LAYER, LINEAR_TRANSFORMER_LAYER}_DISABLE` | off | surgical opt-out while master is on |

### 4.4 MTP

| Var | Default | Effect |
|---|---|---|
| `QWEN36_MTP_MAX_PROMPT_TOKENS` | 1,000,000 | auto-disable MTP above this prompt length (`cli/main.rs:48`) |
| `QWEN36_MTP_STATS` | off | print `mtp.stats accepted=… acceptance_rate=…` (the reference way to measure acceptance) |
| `QWEN36_MTP_TRACE` | off | verbose verify-window trace |
| `QWEN36_MTP_MULTI_GRAPH_DISABLE` | off | force host-launch path for MTP=2/3 (bisection aid) |
| `QWEN36_MTP_TREE_DISABLE` | off | kill tree-MTP dispatch (force chain) |
| `QWEN36_MTP_BATCH_LM_HEAD_DISABLE` | off | kill batched lm_head in verify |
| `QWEN36_MTP_SNAPSHOT_RECURRENT` | **on** | `=0` skips recurrent-state snapshots (unsafe unless full-accept) |
| `QWEN36_MTP_SNAPSHOT_KV` | path-dependent | KV snapshot debug toggle |
| `QWEN36_MTP_ASSUME_ACCEPT` | off | skip per-draft verification + snapshots (only when rejection impossible/acceptable) |
| `QWEN36_MTP_DEVICE_CHAIN` / `_BATCH` | off / 2 | experimental device-side draft chain |

### 4.5 Memory, context, prefill strategy (Rust, `engine.rs:505–587`)

| Var | Default | Effect |
|---|---|---|
| `QWEN36_KV_CACHE_DTYPE` | fp8 | `bf16\|fp8\|tq3\|tq35` (read in CLI) |
| `QWEN36_LONG_CONTEXT_MODE` | auto | auto = ON iff `max_context ≥ 8192`; ON disables both fused weight stores (saves ~8 GB) |
| `QWEN36_LONG_CONTEXT_AUTO_MIN_CONTEXT` | 8192 | the auto threshold |
| `QWEN36_PREFILL_CAPACITY` | 8192 (ctx ≤ 32K) / 2048 (above) | prefill chunk size |
| `QWEN36_DECODE_ATTENTION_BUCKET_MIN_CONTEXT` | 8192 | min bucket sizing decode split launches (so 262K-reserved runs don't launch 262K-sized splits) |
| `QWEN36_DECODE_ATTENTION_BUCKET_DISABLE` | off | size splits from configured `max_context` (old behavior) |
| `QWEN36_CUDA_WORKSPACE_BYTES` / `_MIB` | 256 MiB | GPU workspace |
| `QWEN36_DISABLE_MLP_FUSED` / `QWEN36_DISABLE_LINEAR_ATTN_FUSED` | off | kill fused stores individually |
| `QWEN36_PREFILL_FUSED_MLP` | 0 | opt-in fused-MLP on the prefill path |
| `QWEN36_PREFILL_FUSED_LINEAR_ATTN_DISABLE` | off | kill fused in_proj on the prefill path |
| `QWEN36_DELTANET_CHUNKED_PREFILL` | **on** | `=0` disables chunked DeltaNet prefill |

### 4.6 Split-KV tuning (diagnostics; defaults are tuned — don't ship overrides)

| Var | Effect |
|---|---|
| `QWEN36_ATTENTION_SPLIT_DISABLE` | disable split-KV attention entirely (`engine.rs:10432`) |
| `QWEN36_ATTENTION_SPLIT_TIMESTEPS` | per-block timestep count override |
| `QWEN36_DECODE_ATTENTION_N_SPLITS` / `QWEN36_PREFILL_ATTENTION_N_SPLITS` | force split counts |
| `QWEN36_PREFILL_SPLIT_MAX_TOKENS` / `QWEN36_PREFILL_SPLIT_MIN_SPLITS` | widen split-prefill eligibility (the lossy 5×-at-7K experiment — superseded by split-K verify) |

### 4.7 Drafter diagnostics

| Var | Default | Effect |
|---|---|---|
| `QWEN36_DRAFTER_SWA_WINDOW` | checkpoint (2048) | override sliding window — **proven dead lever** |
| `QWEN36_DRAFTER_SWA_ALL` | 0 | apply window to the full-attention layer too |

### 4.8 Debug / profiling / parity

| Var | Effect |
|---|---|
| `QWEN36_DEBUG_DUMP_DIR`, `QWEN36_DEBUG_DUMP_ALL_LAYERS`, `QWEN36_DEBUG_DUMP_DECODE` | dump intermediate BF16 buffers (parity harness, see AGENT.md) |
| `QWEN36_DEBUG_LAYER_TRACE` | per-layer min/max/mean-abs |
| `QWEN36_PROFILE_PREFILL_CHUNKS` / `QWEN36_PROFILE_DECODE_LAYERS` | synchronized per-bucket timings (bypasses decode graph — never for throughput numbers) |
| `QWEN36_DEBUG_CUBLASLT`, `QWEN36_DEBUG_CUBLASLT_SYNC` | cuBLASLt call logging / sync (`nvfp4_gemm.cu`) |
| `QWEN36_DEBUG_CUDA_ALLOC` | allocation logging (`runtime.cu`) |
| `QWEN36_PARITY_MODEL/_DUMP/_PROMPT/_DECODE_TOKEN/_DEVICE`, `QWEN36_DECODE_LOCAL`, `QWEN36_DECODE_COS_FLOOR` (0.998), `QWEN36_DECODE_STOP_ON_FAIL` (1) | `scripts/decode_parity.py` inputs |

### 4.9 Build

| Var | Effect |
|---|---|
| `QWEN36_FP4_KERNEL_LIB_DIR` | where `libqwen36_fp4_kernels.so` lives (build.rs + runtime) |
| `CUDA_HOME`, `NVCC`, `OUT_DIR`, `QWEN36_FP4_SM` (=120), `QWEN36_FP4_CUDA_MIN_VERSION` (=13.0) | `scripts/build_cuda.sh` / `.cargo/config.toml` |

## 5. CLI reference (binary `qwen36`, `crates/cli/src/main.rs`)

| Command | Purpose | CUDA? |
|---|---|---|
| `discover` | mmap safetensors → `model_layout.json` | no |
| `inspect-config` | parse + print `config.json` topology | no |
| `budget --ctx N --kv <dtype>` | VRAM estimate | no |
| `tokenize` | tokenizer check | no |
| `validate-weights` | manifest validation against checkpoint | no |
| `cuda-diag` | device diagnostics | yes |
| `gpu-load --max-context N` | upload weights, report VRAM | yes |
| `chat` | generation; `--mtp-speculative-tokens 0..4`, `--mtp-tree-leaves K`, `--drafter none\|dflash` | yes |
| `bench` | throughput; `--prompt-tokens`, `--prompt-file`, `--token-text`, same MTP flags | yes |
| `dump-logits` / `dump-decode` | parity dumps (with `QWEN36_DEBUG_DUMP_*`) | yes |
| `decode-vs-prefill-check` | reproduces the decode/prefill logits divergence | yes |
| `validate-drafter`, `drafter-load`, `drafter-handoff-smoke`, `drafter-step-smoke`, `drafter-iter-smoke`, `drafter-chat-smoke`, `drafter-forward-smoke` | DFlash bring-up phases E/F (smoke + fixture parity) | yes |

Cargo feature: `cuda` on `kernels` → propagated by `runtime`, `drafter`, `cli`. Without it,
GPU commands fail explicitly (`UnsupportedNoCuda`); never any mock output.

## 6. Crate map

```
cli → {core, loader, tokenizer, kernels, runtime, mtp, drafter}
runtime → {core, kernels, loader(opt), mtp}     drafter → {core, kernels, loader}
mtp → {core, kernels}    kernels → {core}    loader → {core}    tokenizer → {}
```

| Crate | Role | Key files |
|---|---|---|
| `core` | topology, dtypes, layouts, memory budgets | `config.rs`, `layout.rs`, `dtype.rs`, `budget.rs` |
| `loader` | safetensors mmap + `model_layout.json` | `lib.rs` |
| `tokenizer` | HF tokenizer + Qwen chat template | `lib.rs` |
| `kernels` | typed kernel specs, C ABI mirror, `CudaBackend`/`NoCudaBackend` | `backend.rs` (dispatch + gemv routing), `interpreter.rs`, `memory.rs`, `graph.rs` |
| `runtime` | the engine: weight upload, KV/DeltaNet state, prefill/decode, CUDA graphs, MTP orchestration | `engine.rs` (11K LoC), `gpu.rs`, `weights.rs`, `interpreter_compile.rs` |
| `mtp` | speculative controller (snapshot/restore/replay contracts) | `lib.rs` |
| `drafter` | DFlash loader/forward/propose/handoff | `dflash.rs`, `forward.rs`, `propose.rs`, `handoff.rs` |
| `cli` | `qwen36` binary | `main.rs` |

C ABI: `kernels-cuda/include/qwen36_fp4.h` (~77 entry points). **Any change must be mirrored
in `crates/kernels/src/backend.rs` + the typed spec module + `smoke.cu`** (see AGENT.md "ABI sync").

## 7. Scripts & docs

| Script | Role |
|---|---|
| `build_cuda.sh` / `smoke_cuda.sh` | build `.so` (sm_120a) / run kernel smoke suite |
| `verify_perf_gate.sh` | **run before/after any perf change** — DFlash split-K + MTP interpreter gates vs baselines |
| `bench_matrix.sh`, `bench_tq35_contexts.sh`, `bench_attention_ab.sh`, `profile_bench.sh` | bench sweeps / A-B / nsys profiling |
| `quality_tq35.sh` | TQ35 vs BF16 KV logits quality |
| `decode_parity.py`, `dflash_parity.py` | PyTorch parity (64-layer decode boundaries / drafter fixtures) |
| `drafter_al_eval.sh` | 6-prompt AL geomean battery — **the only valid way to judge drafter-quality changes** |
| `dflash_bench_sweep.py` | DFlash vs MTP=3 sweep driver |

CI: `.github/workflows/ci.yml` runs the CPU loop on every push/PR — fmt,
clippy with and without the cuda feature (the cuda-feature clippy compiles all
GPU-gated Rust without linking), and the workspace tests. The GPU gates
(`build_cuda.sh`, `smoke_cuda.sh`, `verify_perf_gate.sh`) remain local and
mandatory before merging anything that touches kernels or the engine hot path.

Docs: `doc.md` = design spec (start here for intent). `AGENT.md` = agent instructions,
contracts and guardrails. `DAILY.md` = dated experiment journal (the source of the
verdicts summarized in §2). `docs/*.md` = operational
guides. `docs/superpowers/{specs,notes,plans}` = per-experiment design docs and write-ups;
`plans/archive/` is historical. `benches/data/` holds the real-text bench prompts.
