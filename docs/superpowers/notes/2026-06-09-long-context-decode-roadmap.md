# Long-context decode throughput — techniques & roadmap

**Date:** 2026-06-09
**Author:** Claude (research synthesis)
**Scope:** boost decode (token-gen) throughput at long context
(target: 3K → 20K+ tokens) on RTX 5090 / Qwen3.6-27B-NVFP4.
**Not in scope:** prefill throughput, batching, multi-GPU.

This note consolidates what we already know internally about long-context
decode performance, and surveys the external state of the art. It exists
to give us a menu of techniques ranked by expected ROI for our specific
stack (NVFP4 target, BF16 DFlash drafter, RTX 5090 Blackwell, hybrid
linear+full attention topology). It's a working roadmap, not a plan —
each entry has enough info to estimate whether it's worth doing.

---

## 1. Where we are

### Current observed bottleneck (from internal benches)

- **Per-token cost at 8K tokens** (`QWEN36_PROFILE_DECODE_LAYERS=1`,
  AGENT.md L386–401):
  - linear_attn (×48): 4.45 ms total
  - full_attn (×16): 5.87 ms total
  - mlp: 10.68 ms
  - **total: ~22.8 ms/token (~44 tok/s)**.

- **DFlash long-context cliff** (`docs/superpowers/notes/2026-06-09-dflash-long-context.md`):
  - 3K ctx, coherent prompt: AL 7.76, **1.80× vs MTP=3**
  - 7K ctx: AL still 5.43 but **0.52× vs MTP=3** — drafter forward
    dominates (naive O(q×kv) attention kernel, 5 layers).

- **MTP=3 holds throughput flat** across 512–8K tokens (~30–40 tok/s
  here): MTP head is one extra layer that doesn't redo prompt attention.

### What's already done / shipped

| Lever | Status | File / commit |
|---|---|---|
| Chunked prefill (2048 chunk default) | shipped | `engine.rs` |
| `LONG_CONTEXT_MODE` (drop fused stores ≥ 8K) | shipped | `engine.rs:357,374,393` |
| NVFP4 weights + decode_gemv (N=1) | shipped | AGENT.md L104–130 |
| Int8 TurboQuant KV on full-attn layers | shipped | AGENT.md L67 |
| GQA + KV splits (`n_splits ≥ 32` at long ctx) | shipped | AGENT.md L283 |
| DeltaNet 4-way in_proj fusion (−30% linear_attn) | shipped | AGENT.md L275 |
| MTP=3 chain spec | shipped | engine + cli |
| DFlash spec (block diffusion, k=16) | shipped 2026-06-09 | `crates/drafter/`, AGENT.md "DFlash" section |
| Combined gate+up FP4 GEMM fusion | shipped | AGENT.md L275 |

### Pending in-flight

- **Codex interpreter megakernel** — `docs/superpowers/specs/2026-06-08-interpreter-megakernel-design.md`.
  Persistent counter-sync kernel for the whole decode step. Projected
  +7.7 % realistic on MTP=0 (decision gate ≥ +3 tok/s). Multi-week.
  Not specifically long-context, but interacts with anything that
  changes the per-token call pattern. **We're blocked on this finishing
  before fanning out kernel-level work.**

### Negative results to remember

- **Per-block megakernel** (Phase 2, 2026-05-23): −4 % MTP=0. Reverted,
  kept opt-in. Lesson: the CUDA graph already amortises launch overhead,
  and cuBLASLt beats hand-fused MLP shapes for us.
- **Tree-MTP K > 1** without batched leaf forward: −60 to −75 % vs chain.
  Deferred until we batch leaves.
- **Productive-spin L2 prefetch**: noise (+0.13 %).
- **Full-attn Q/K/V fusion**: ≤ 1 % gain, not worth VRAM pressure.

The pattern: anything that fights the CUDA graph or cuBLASLt on shapes
they're already good at tends to lose. Wins come from changing the
*algorithm* (KV layout, sparsity, drafter), not micro-optimising what's
already graph-captured.

---

## 2. External techniques, grouped by what they target

I've organised by which of our bottlenecks each technique attacks, so we
can match them to where we're actually losing time, not pick by paper
novelty.

### A. Attention kernel time (per-step compute)

For us this is:
- 5 drafter layers × naive O(q × kv) attention (the DFlash bottleneck),
- 16 target full-attn layers × split-GQA decode kernel.

#### A.1 FlashAttention-3 / 4 — kernel state of the art on Blackwell

FA3 (Hopper) introduced warp specialisation + asynchrony + FP8.
**FA4 (March 2026)** is the Blackwell-targeted rewrite: redesigned
pipelines using Blackwell's fully-async MMA and tensor memory,
software-emulated exponentials to bypass the MUFU bottleneck, and
conditional softmax rescaling. On Blackwell, "the bottleneck for
attention has shifted away from matmul toward shared memory traffic and
non-matmul ops like softmax" — FA4 explicitly co-designs against that.

**ROI for us:** the DFlash drafter attention kernel
(`kernels-cuda/drafter_attention.cu`) is naive — replacing it with an
FA-tiled kernel is the documented next step (3–5× speedup at 7K ctx,
pushes break-even from ~5K to ~20K, "few days of focused work").
Reusing FA4 directly is non-trivial because it targets the standard
attention layout, not our drafter's
`K = [k_ctx; k_noise], V = [v_ctx; v_noise]` block.

#### A.2 Flash-Decoding — split-K along the KV-seq dimension

Standard FA parallelises across heads/batch — wastes SMs when batch=1.
**Flash-Decoding adds parallelism along the keys/values sequence axis**,
splitting KV into chunks decoded in parallel then reduced. Fully utilises
the GPU at batch=1 if context is long enough.

**ROI for us:** directly applicable to the **target's full-attn layers
at long ctx**, batch=1 is exactly our case. Could pair with FlashInfer's
split-K decode primitive. Likely ~1.3–1.8× on full_attn cost at 8K+.

#### A.3 SageAttention 2++ — FP8 matmul + FP16 accum

3.9× over FlashAttention while matching SageAttention2 accuracy. FP8
matmul accumulated in FP16 (vs FP8 accum in S2): 2× the matmul speed.

**ROI for us:** we already tried Sage B.2 INT8 P·V and regressed
(memory `project_sage_b2_blocked.md`: future Sage 2++ retry needs
cheaper V quant). Worth a second look now that the FP8-accum-FP16
variant is published — it's a different operating point than what we
tested. **Estimated +10–20 % on attn** if it lands cleanly on sm_120
without our prior accuracy issue.

#### A.4 BitDecoding (HPCA 2026) — tensor cores on low-bit KV

Up to **8.6× speedup on Blackwell with NVFP4 KV cache** over FP16
FlashDecoding-v2. Solves the historical problem that low-bit KV decode
struggled to use tensor cores because of de/quant overhead — BitDecoding
co-schedules CUDA + tensor cores. On LLaMA-3.1-8B with 128K context: 3×
single-batch decode latency reduction.

**ROI for us:** this is the *direct* implementation of the "NVFP4 KV
cache" roadmap item B1 (AGENT.md L330). 4-bit KV storage + FP8 dequant
during attn was already in our roadmap; BitDecoding gives us a published
kernel pattern + open-source implementation
(`github.com/OpenBitSys/BitDecoding`). Adapting it to our Int8
TurboQuant layout would need a layout migration, but the kernel pattern
is reusable.

### B. KV cache memory bandwidth

Decode at long ctx is memory-bound on KV loads. Two main families:
quantisation (smaller bytes per token) and architectural compression
(fewer K/V state vectors per token).

#### B.1 FP8 KV cache (production default)

vLLM's FP8 KV-cache halves memory and, with FA3 backend, performs attn
in the quantised domain. "FP8 is the practical default for production
deployments, halving KV cache memory with accuracy impact so small that
most benchmarks cannot distinguish it from FP16."

**ROI for us:** we already store Int8 on full-attn layers. Moving to
**FP8 KV** unlocks BitDecoding-style attn in the quantised domain — the
gain isn't the storage (Int8 ≈ FP8 for memory) but the *kernel ergonomics*
on Blackwell, which has native FP8 MMA.

#### B.2 NVFP4 KV cache (Blackwell-native)

NVFP4 (4-bit) gets another 2× over FP8. Direct prereq for BitDecoding's
8.6× speedup. AGENT.md roadmap item B1, est. ~2× decode at long ctx,
1–2 weeks.

**Notable caveat for us:** we already burned weeks finding the NVFP4
**decode-kernel divergence bug** (DFlash AL=1.4 mystery → diagnosed
via `decode-vs-prefill-check`, cos sim 0.81→−0.19 between decode and
prefill). NVFP4 KV cache means extending FP4 codepaths into attn — we'd
want to re-run the parity diagnostic at every step.

#### B.3 DeepSeek MLA — Multi-head Latent Attention

Compresses K/V into a single low-rank latent, reducing KV cache 7–14×
(some sources cite 57×). Supports 128K contexts on standard hardware.
2.5–3× decode tok/s in batch.

**ROI for us:** MLA is *architectural* — it requires the model to be
trained with MLA. We're using Qwen3.6, not DeepSeek. There's recent
work on "MHA → MLA retrofit" (arxiv 2502.14837, *TransMLA*) showing
you can convert pre-trained MHA models to MLA, but it requires post-
training and validation. **Not in scope short-term**, but worth noting:
if we ever fine-tune Qwen3.6, MLA retrofit would be a one-shot win.

### C. KV cache access pattern — sparse attention

Instead of attending to every cached token, attend to a query-relevant
subset. Reduces both memory traffic and compute proportionally to the
sparsity.

#### C.1 Quest — query-aware page sparsity

KV is paged; each page tracks min/max K. At decode, query × (min, max)
estimates page criticality → keep only Top-K pages → sparse attn.
**Up to 7.03× self-attention speedup, 2.23× total latency reduction**,
negligible accuracy loss on long-dep tasks.

**ROI for us:** AGENT.md B3, est. 3–5× extra at ≥ 4K, 1–2 weeks. This
is the cleanest published method that doesn't require retraining and
keeps the full KV cache around for correctness fallback. **Likely the
highest ROI single item on the roadmap for long context.** Pairs well
with paged-KV (which we don't have yet — would need to land paged-KV
first or fold them in together).

#### C.2 StreamingLLM (attention sinks) + SnapKV (heavy hitters)

- **StreamingLLM** keeps first-N "attention sink" tokens + sliding
  window. Bounded KV regardless of context length. Good for unbounded
  streams; loses information outside the window so unsuitable as
  default for chat/reasoning.
- **SnapKV** uses a small observation window at end of prompt to vote
  for important *prefix* positions; evicts non-heavy-hitters. Prefill-
  time decision, fixed at decode start.
- **H2O** is similar idea, online: tracks attention scores to evict.
  29× throughput over HF Accelerate at 20 % heavy-hitter retention on
  OPT-30B.

**ROI for us:** these trade accuracy for throughput. For a coding/
agent workload where we sometimes need to recall a function defined
3K tokens ago, sliding window will silently drop it. **Useful as an
opt-in mode** for streaming workloads, not as default. Lower priority
than Quest (which keeps full KV).

#### C.3 RocketKV — two-stage compression (permanent eviction + dynamic
selection)

3× end-to-end decode speedup, 31 % peak memory reduction on H100.
Combines permanent eviction (SnapKV-style) with dynamic selection
(Quest-style). State of the art on long-context KV compression as of
early 2026.

**ROI for us:** strictly more complex than Quest alone for similar
gains. Worth knowing exists; not the first thing to try.

#### C.4 DeepSeek Sparse Attention (DSA) — natively trained sparse

Newer line: train the model with sparse attention from the start. Not
retrofittable.

### D. Speculative decoding for long context

Our current spec stack: DFlash (block diffusion, k=16) and chain MTP=3.
DFlash collapses past 5K ctx because of drafter forward cost — the
acceptance rate stays decent (5.4 at 7K) but per-iter time explodes.

#### D.1 MagicDec — self-speculation with sparse-KV drafter

Key insight: as ctx grows, the KV memory bandwidth becomes the bottleneck
for *both* target and draft. MagicDec uses the **target model itself**
as the draft, but with **StreamingLLM-style sliding-window KV** (much
smaller, fits in faster memory). This makes the draft cheap enough at
long context that spec decoding stays worth it.

**ROI for us:** *exactly* solves the problem we have (DFlash drafter
forward cost dominating at 7K). Two ways to apply:
- **DFlash drafter with sparse KV**: extend our drafter forward to do
  sliding-window or Quest-style attn over the captured target hidden
  states. Cheap to prototype since the drafter is only 5 layers.
- **MTP head with sparse KV**: similarly. MTP head currently re-attends
  to full context; bounded KV would keep its cost flat.

This composes with Quest/StreamingLLM — same kernels, just plugged into
the drafter/MTP rather than the target.

#### D.2 EAGLE-3 / EAGLE 3.1

3–6.5× over vanilla AR, 20–40 % over EAGLE-2. **Long-context: EAGLE 3.1
achieves 2× longer acceptance length than EAGLE 3** in long-ctx workloads.
EAGLE-3 acceptance rate stays ~70–80 % flat across positions vs EAGLE's
drop-off.

**ROI for us:** roadmap B2 (15–25 % vs MTP-4, 2–3 wk). EAGLE-3 needs
its own head training — non-trivial. Less aligned with our DFlash bet
since DFlash already gives us a single-pass k=16 block. **EAGLE-3 makes
more sense as a fallback/complement when DFlash is suboptimal (multi-
topic prose, long ctx)** rather than a replacement.

#### D.3 FlashAttention drafter rewrite (in-house)

Documented in `2026-06-09-dflash-long-context.md` as the next step:
re-implement `drafter_attention.cu` with tiling. Expected 3–5× at 7K,
pushes break-even to ~20K. "A few days of focused work."

**This is the highest-ROI item we already understand.** Should happen
before any external tech adoption.

### E. Architectural / topology

Less actionable short-term but worth tracking:

- **Hybrid linear+full** (what Qwen3.6 already is): 75 % of layers are
  linear-attn with no KV — long-context-friendly by design.
- **Mamba / SSM** layers: similar idea, more aggressive.
- **MLA retrofit**: noted in B.3.
- **Native sparse attention** (DSA, NSA, etc.): trained from scratch.

We don't pick the architecture — we run Qwen3.6. Useful context for
why MTP=3 stays flat (it doesn't redo the long part), and why FA-tiling
the *drafter* matters more than micro-optimising the target's full-attn
(which already only runs on 25 % of layers).

---

## 3. Ranked recommendations for our stack

Filtered by: cost × probability × gain × strategic fit. Assumes
interpreter megakernel lands first (interacts with everything else
at the kernel level).

### Tier 1 — high ROI, low risk, in-flight prerequisites already met

1. **FA-tile the DFlash drafter attention kernel**
   (`kernels-cuda/drafter_attention.cu`)
   - Gain: 3–5× at 7K ctx, pushes break-even from ~5K to ~20K.
   - Cost: few days. Parity-checked against existing host fp32 ref.
   - No new model artefacts, no retraining.
   - **Direct fix for the bottleneck we just measured.**

2. **Flash-Decoding split-K on target full-attn**
   - Gain: ~1.3–1.8× on the 5.87 ms full_attn cost at 8K, i.e. ~6 %
     of total per-token at 8K.
   - Cost: ~1 week. Adapt FlashInfer's split-K pattern to our NVFP4
     decode path.
   - Composes with everything else.

### Tier 2 — bigger gain, more work, well-scoped

3. **Quest-style query-aware page sparsity** (AGENT.md B3)
   - Gain: 3–5× extra on full-attn at ≥ 4K. With our 16 full-attn
     layers contributing ~25 % of total cost, this lands closer to
     +15–20 % overall but stacks with #1, #2.
   - Cost: 1–2 weeks. Needs paged-KV layout (which we'd want anyway).
   - No accuracy loss claimed on long-dep tasks; verify on our coding
     benches.

4. **NVFP4 KV cache + BitDecoding-style attn** (AGENT.md B1)
   - Gain: ~2× decode at long ctx.
   - Cost: 1–2 weeks for KV layout migration. **Re-run NVFP4 parity
     diagnostic at every step** (we lost weeks on the decode-vs-prefill
     divergence bug; extending FP4 into attn is asking for the same
     class of issue).
   - Strong synergy with Quest (less data per page → more pages fit
     hot).

### Tier 3 — promising, more uncertainty

5. **MagicDec-style sparse-KV drafter** (extend DFlash drafter)
   - Gain: keeps DFlash speedup viable at 10K+ ctx (currently 0.52×
     at 7K — would aim for ≥ 1.5× at 10K+).
   - Cost: 1 week prototype on the existing drafter. Risk: AL may
     drop if the drafter loses information from outside the window.
   - Worth a small experiment after #1: if FA-tiling alone fixes
     long-ctx, this becomes less urgent.

6. **Sage 2++ FP8-MMA-FP16-accum retry** (AGENT.md B4)
   - Gain: +10–20 % on attn.
   - Cost: 3–5 days. Re-evaluate the Sage B.2 blocker
     (`project_sage_b2_blocked.md`) under the new FP16-accum operating
     point.

### Tier 4 — strategic, not short-term

7. **EAGLE-3 head as MTP replacement** (AGENT.md B2): 2–3 weeks, needs
   training pipeline. Most useful as a complement to DFlash, not
   replacement.

8. **MLA retrofit for Qwen3.6**: only if we ever do our own fine-tune.
   Massive long-ctx win if it works, but requires a training run.

9. **StreamingLLM/SnapKV as opt-in streaming mode**: low priority, only
   if we have a use case where unbounded streams matter and lossy
   context is acceptable.

### Skip / not now

- **Per-block megakernel revival** without algorithmic change: already
  proven −4 %.
- **Tree-MTP K>1** without batched leaf forward: blocked, defer.
- **RocketKV over Quest**: marginal complexity-vs-gain.

---

## 4. Decision matrix — what to pick next

| If we have | … then do |
|---|---|
| Codex megakernel landed, no breaking changes | start **#1 (FA-drafter)** immediately — single file, well-scoped, fixes the measured bottleneck |
| FA-drafter shipped, want next decode boost | **#2 (Flash-Decoding target)** + start scoping #3 (Quest) in parallel |
| > 1 wk to invest on a single big win | **#3 (Quest)** — biggest single decode lever for full-attn cost |
| Want to push effective context to 32K+ | **#4 (NVFP4 KV + BitDecoding)** — only path to that regime |
| DFlash collapses past FA-drafter fix | **#5 (sparse-KV drafter)** |

---

## 5. Open questions for the next session

- Does the Codex interpreter megakernel constrain attention kernel
  layout in a way that affects #1 / #2 / #3? Need to re-read its
  spec once landed.
- Do we want a paged-KV refactor as standalone work, or fold it into
  Quest (#3)? Standalone is cleaner but no immediate user-visible win.
- BitDecoding kernel licence and reuse: `github.com/OpenBitSys/BitDecoding`
  — needs a read-through to see how much we'd port vs. rewrite.
- Re-baseline expected gains on our actual prompt mix once #1 lands —
  the long-context bench is currently dominated by drafter cost, so
  fixing it will reshuffle the relative cost of everything downstream.

---

## Sources

Internal:
- `docs/superpowers/notes/2026-06-09-dflash-long-context.md`
- `docs/superpowers/notes/2026-06-09-dflash-final.md`
- `docs/superpowers/specs/2026-06-08-interpreter-megakernel-design.md`
- `docs/superpowers/specs/2026-06-08-dflash-speculative-decoding-design.md`
- `AGENT.md` (long-context section L310–401, roadmap L330–337)

External (papers / blog posts):
- FlashAttention-3 (Hopper FP8, warp specialisation): arxiv 2407.08608
- FlashAttention-4 (Blackwell, March 2026): lambda.ai blog,
  iclr-blogposts 2026
- Flash-Decoding (split-K along seq): pytorch.org/blog/flash-decoding
- SageAttention 2++ (FP8 matmul + FP16 accum): arxiv 2505.21136
- BitDecoding (HPCA 2026, NVFP4 + tensor cores): arxiv 2503.18773,
  github.com/OpenBitSys/BitDecoding
- Quest (query-aware page sparsity, ICML 2024): arxiv 2406.10774,
  github.com/mit-han-lab/Quest
- StreamingLLM (attention sinks): arxiv 2309.17453
- SnapKV / H2O: marktechpost.com 2026-04-29 top-10 survey
- RocketKV: arxiv 2502.14051
- MagicDec (long-ctx spec decoding): arxiv 2408.11049,
  infini-ai-lab.github.io/MagicDec
- EAGLE-3 / EAGLE 3.1: arxiv 2503.01840, vllm.ai/blog 2026-05-26
- DeepSeek MLA: arxiv 2405.04434
- MLA retrofit (TransMLA): arxiv 2502.14837
- NVFP4 KV in vLLM (state of FP8 KV-cache & attn quantisation):
  vllm.ai/blog 2026-04-22
- KV cache survey (2026 engineering guide): digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide
