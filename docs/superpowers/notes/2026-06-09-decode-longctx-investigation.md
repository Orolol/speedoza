# Base decode (MTP=0) long-context slowdown — investigation

**Date:** 2026-06-09
**Trigger:** user observation — base token generation (no DFlash/MTP)
drops too much with context; a classic inference engine stays near-flat
to 32K and never below ~50%. Investigate.
**Verdict: confirmed, root-caused, fix scoped.** The slide is real
(−35% at 24K, extrapolating ~−43% at 32K) and essentially 100% of it is
the **decode full-attention kernel running ~28× off memory bandwidth**
(latency-bound scalar inner loop). Everything else in the engine is
already flat, as the user expects from a classic engine.

## 1. Measured curve (MTP=0, max-new=64, default config, FP8 KV)

| prompt ctx | decode tok/s | vs ctx=128 |
|---:|---:|---:|
| 128 | 49.7 | — |
| 2048 | **OOM** (see §4) | — |
| 4096 | **OOM** (see §4) | — |
| 8192 | 43.1 | −13% |
| 12288 | 39.2 | −21% |
| 16384 | 36.2 | −27% |
| 24576 | 32.3 | −35% |

Linear extrapolation puts 32K at ~28-29 tok/s (−43%), i.e. right at the
edge of the user's "not below 50%" bar — and it keeps sliding linearly
beyond.

## 2. Per-layer decode profile (`QWEN36_PROFILE_DECODE_LAYERS=1`)

| bucket | 8K (ms/tok) | 24K (ms/tok) | behavior |
|---|---:|---:|---|
| embed | 0.18 | 0.17 | flat |
| linear_attn (48 DeltaNet layers) | 5.6 | 5.3 | **flat** ✓ (state-resident, O(1) by design) |
| **full_attn (16 layers)** | **6.1** | **12.7** | **all of the growth** |
| mlp (64 layers) | 11.7 | 11.1 | flat |
| lm_head | 1.8 | 1.65 | flat |
| total | ~25.4 | ~31.0 | |

The architecture does its job: 75% of the network is context-flat.
The entire slide is the 16 full-attn layers.

## 3. Root cause: decode attention is 28× off bandwidth

KV traffic per token at 24K = 16 layers × 2 (K+V) × 24576 pos × 4
kv_heads × 256 head_dim × 1 B (FP8) ≈ **805 MB → 0.45 ms** at 1.8 TB/s.
Measured: **12.7 ms = 28× off the bandwidth floor** (40× at 8K). A
classic engine (FlashDecoding/FlashInfer-class) runs decode attention
within ~2-4× of bandwidth, which is why their curves stay flat — their
attention cost is invisible next to the flat MLP/weights cost.

The kernel (`attention_decode_split_gqa_kernel`, attention.cu:1225) is
the same scalar family as the verify bottleneck we already replaced
with the wmma split-K (P2): grid (kv_heads=4 × n_splits), 256 threads,
and a **serial per-timestep inner loop** — per KV position: a 1-byte-
per-thread K load, 6 (q_per_kv) block-wide shuffle reductions, a
block-wide online-softmax update with `__syncthreads`, then the V load
+ FMA. Per-timestep latency ~1 µs × 64 serial timesteps per CTA, and
CTA-count (waves) grows linearly with context.

**Structural probe** (`QWEN36_ATTENTION_SPLIT_TIMESTEPS` sweep at 24K):

| timesteps/block | n_splits | decode tok/s |
|---:|---:|---:|
| 64 (default) | 512 | **33.3** |
| 128 | 256 | 31.0 |
| 256 | 128 | 27.9 |
| 512 | 64 | 26.0 |

Bigger blocks = strictly worse → the serial inner loop dominates; the
split/reduce overhead is NOT the issue (the default 64 is already
optimal). The fix must attack per-timestep cost, not split topology.

## 4. Secondary findings

- **Fusion auto-disable is a minor effect.** `max_context ≥ 8192`
  auto-disables MlpFusedStore + LinearAttnInProjFused (VRAM saving).
  Isolated at ctx=128: 49.7 → 47.9 tok/s = **−3.6%** only.
- **OOM at ctx 2048–4096 with the default config** on a 29.5 GB-free
  GPU: below the 8192 auto threshold the fused stores stay ON and the
  total build no longer fits (fails on a 40 MB malloc). So default
  `bench` at 2–6K context *fails* while 128 and ≥8K work — a usability
  trap. Mitigations: `QWEN36_LONG_CONTEXT_MODE=1`, or make the auto
  switch free-VRAM-aware instead of max_context-based.
- Decode is CUDA-graph-captured with power-of-two context buckets
  (re-capture on bucket cross). A kernel swap keeps the same launch
  shape → graph-compatible.

## 5. Fix scope (not yet implemented)

Rewrite the inner loop of `attention_decode_split_gqa_kernel`
(decode, q=1) the way P2 fixed verify — but with register tiling
instead of wmma (tensor cores are the wrong tool at q=1):

1. **Multi-timestep register tiling**: process T=8–16 KV positions per
   iteration; one 128-bit vectorized load brings 16 FP8 values/thread;
   compute T dot-products in registers before any block-wide
   reduction/sync (amortizes the ~3 syncs + 6 shuffle-reduces that
   currently fire per single timestep).
2. **LUT-based FP8→float decode** (256-entry SMEM table) instead of
   the branchy `ldexpf` decode per element — also flagged by the M=16
   headroom analysis as the dominant latency term in the verify tile.
3. Keep the split-K topology, partials layout, and reduce kernel
   unchanged (they are proven and graph-captured).

Projected: per-timestep effective cost ÷4–8 → full_attn 12.7 →
~2–4 ms at 24K, total ~21–23 ms → **~45–48 tok/s at 24K** (vs 32.3),
and a near-flat curve ~50 → ~47 through 32K — the classic-engine
shape. Same parity discipline as P2: kernel-vs-kernel smoke gate
(cos ≥ 0.998 vs the current scalar at several ctx × dtypes) + token
identity on short prompts + `verify_perf_gate.sh`.

Estimated effort: 2–4 days including parity bisection (the P2
experience says budget 2× the kernel-writing time).

Note the ceiling: with linear_attn 5.3 + mlp 11.1 + lm_head 1.7 +
embed 0.2 ≈ 18.3 ms of context-flat cost, perfect attention gives
~52 tok/s max at any context. The fix brings 24K from 62% to ~90% of
that ceiling.

## 6. Repro

```bash
# curve
QWEN36_FP4_KERNEL_LIB_DIR=$PWD/target/cuda LD_LIBRARY_PATH=$PWD/target/cuda \
  target/release/qwen36 bench --model-dir ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
  --prompt-tokens {128|8192|16384|24576} --max-new-tokens 64 --mtp-speculative-tokens 0
# per-layer profile
QWEN36_PROFILE_DECODE_LAYERS=1 ... bench --prompt-tokens 24576 --max-new-tokens 16 ...
# split granularity probe
QWEN36_ATTENTION_SPLIT_TIMESTEPS={64|128|256|512} ... bench --prompt-tokens 24576 ...
```
