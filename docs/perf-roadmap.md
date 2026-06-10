# Performance roadmap — extracting every last token/s

> Goal: the fastest possible single-stream engine for exactly ONE combo —
> Qwen3.6-27B NVFP4-MTP on one RTX 5090. Researched 2026-06-10 against the
> current state of the art (Xiaomi MiMo/TileRT 1T@1000+ tok/s, Hazy Research
> low-latency megakernel, EAGLE-3.1). Every item carries an expected gain and
> a kill-gate per the AGENT.md guardrails. Update checkboxes + DAILY.md as
> items land or die.

## North star

Per decode token, every weight byte must cross HBM once: ≈ 13.9 GB
(MLP 8.6 + DeltaNet 2.5 + full-attn 0.8 + lm_head BF16 1.55 + scales) →
**~7.7 ms/token floor @ 1.79 TB/s ⇒ ~129 tok/s** for non-speculative decode.
Measured today: ~19.6 ms (~50 tok/s) = **~40% bandwidth efficiency** (Hazy's
megakernel reaches 78% on the same problem shape; typical engines ~50%).

Speculation multiplies that floor by accepted-tokens-per-forward: at AL≈6
(MiMo's DFlash gets 6.3 on code with block=8; ours peaks 11.8, geomean 5.1
long-ctx) the budget allows **~300–500 tok/s sustained on favorable
workloads**. Today's best: 313 tok/s peak / ~143 typical @3K.

Targets: **MTP=0 ≥ 80 tok/s** (kernel lane), **DFlash ≥ 250 tok/s typical on
code/QA ≤8K** (speculation lane), **prefill ≥ 1000 tok/s @64K** (already
scoped). Stretch: 400+ tok/s peak.

The MiMo recipe (FP4 experts + DFlash block speculation + TileRT persistent
runtime) maps 1:1 onto this repo's existing lanes — we already have the
first two; the third failed here twice for reasons the Hazy write-up
explains precisely (see P3).

---

## P0 — Correctness & measurement (prerequisites, no perf work before)

- [ ] **Fix the decode-vs-prefill logits divergence** (cos ~0.76, degenerate
      MTP=0 output in some configs; repro: `qwen36 decode-vs-prefill-check`).
      Blocks trusting every MTP=0 number below. DFlash routes around it —
      a fix may also lift DFlash AL (verify/decode consistency dividend was
      already proven worth +2.7 AL once, DAILY § P2.1).
- [ ] **Merge `chore/rationalization`** after on-target GPU validation
      (`build_cuda.sh && smoke_cuda.sh`, perf gate, parity floor).
- [ ] **One bench dashboard**: a single script (fixed corpus from
      `benches/data/`, fixed configs: MTP {0,4} × ctx {128, 3K, 8K, 24K} +
      DFlash 2 cells) that emits one table to paste into DAILY.md. Every
      perf item below cites before/after from this script, nothing else.
- [ ] **Nsight bandwidth audit of the decode hot path** (cheap, sizes P3):
      % peak DRAM BW per kernel (gemv shapes M={5120,16640,34816},
      deltanet, tiled attention, lm_head). Output: a table in DAILY.md
      naming the worst offenders. This decides where P3 starts.

## P1 — Bank the pending wins (days, already scoped or in-tree)

- [ ] **Bench the parallel split-reduce** (in tree, gate pending).
      Expected: full_attn flat ≈5–6 ms at 64K → decode ~43–46 tok/s @64K
      (vs 32.7). Gate: `verify_perf_gate.sh` + 24K/64K cells.
- [ ] **Prefill cp.async pipeline** (top scoped item; DAILY § Next steps 2).
      flash/sage prefill have zero cp.async/double-buffering — 170 W
      latency-stalled at 64K. 2–3 stage pipelined K/V tile loads.
      Expected: 3–5× prefill at 64K (274 → ≥1000 tok/s). Gate: wattmeter
      400–500 W + parity (cos ≥ 0.998) + no short-ctx regression.
- [ ] **CUDA-graph the DFlash verify** (`verify_block_batched`): documented
      estimate +20–30% DFlash tok/s. #55 made the split-K path capture-safe
      already. Gate: byte-identical DFlash output on the AL battery prompts.
- [ ] **Interpreter MLP-chunking bench** (Codex lane gate (a)): ≥+2–3 tok/s
      on a counting path → the pipelining thesis lives and feeds P3;
      ~0% → freeze the interpreter as MTP>0 auto-only (the +7.3% is banked)
      and P3 proceeds on the Hazy design instead.

## P2 — Tokens per forward: speculation quality (MiMo's main lever)

MiMo's 1000 tok/s is ~85% speculation: block-diffusion drafter (AL 6.3,
block=8, SWA-only drafter ⇒ constant per-prediction cost) on top of an
efficient verify. Our DFlash AL is the single highest-leverage number in
the whole project: every +1 AL ≈ +15–20% end-to-end.

- [ ] **Drafter long-context fine-tune** (DAILY § AL lane: knobs falsified,
      training is the credible lever). Scope data + pipeline + GPU cost
      first; validate ONLY via `drafter_al_eval.sh` geomean (baseline 5.10).
      Target: geomean ≥ 7. Kill-gate: <5.5 after first training round.
      Note MiMo's drafter is SWA-only — evaluate retraining ours with
      SWA-on-all-layers for constant-cost drafting at long ctx (replaces
      the dead window-knob idea with a training-side version).
- [ ] **Adaptive MTP↔DFlash routing**: the routing table exists (DAILY §
      DFlash, 6 rules on prompt-length/content/gen-length) but is manual.
      Auto-route on (prompt_tokens, max_new_tokens) + online AL fallback
      (switch to MTP=4 if measured AL < 2.5 after N cycles). Expected:
      eliminates every <1× cell from the sweep (worst 0.52×). Gate: no
      cell of the standard sweep below its best-of-both baseline.
- [ ] **Block size sweep for DFlash** (block=16 today; MiMo uses 8):
      smaller blocks = cheaper drafts + higher accept density at low-AL
      regimes. Pure config sweep on the AL battery. Cheap, do with the
      fine-tune eval.
- [ ] **MTP head → NVFP4** (documented candidate): the BF16 MTP head runs
      4×/cycle at MTP=4. Needs a parity harness vs BF16 head first
      (MTP parity floor is the gate). Expected: +5–10% on MTP modes
      (which remain the long-prompt fallback even with DFlash routing).
- [ ] **Evaluate EAGLE-3.1-style head** (vLLM blog 2026-05: up to 2× AL at
      long ctx vs EAGLE-3; ~4.5–5 AL typical): would directly attack our
      weak regime (long-ctx low-AL). Heavy: requires training a head on
      Qwen3.6 hidden states. Decision item only — score vs drafter
      fine-tune on (cost, expected AL, risk); pick ONE training-side bet.
- [ ] ~~Tree/multi-leaf speculation~~ — blocked until a batched-leaf
      forward exists (inventory §2.5); re-open only if the routing +
      fine-tune lane plateaus.

## P3 — ms per forward: the persistent-pipeline lane, done right this time

Three attempts here failed (productive-spin, per-block megakernel,
interpreter v1) and all three died the same death: **they fused launches
without pipelining weights**. The captured graph already amortizes launch
cost, so fusion alone wins nothing. The Hazy megakernel (78% BW on
Llama-1B, 2.5× vs vLLM on H100) and TileRT both get their win elsewhere —
that's the part we never built:

1. **SMEM paging + cross-instruction weight prefetch.** Hazy splits SMEM
   into 13×16 KiB pages; an instruction releases pages as it drains and
   the interpreter hands them to the NEXT instruction, which starts
   `cp.async.bulk`-ing its weights while the current one still computes.
   Our interpreter's PageAllocator has 4 reserved slots — **unused**
   (DAILY § interpreter gaps). This is THE missing mechanism: the GPU
   should never stop reading weights.
2. **Counter-based fine-grained chunking.** Hazy's MLP produces/consumes
   in 4 chunks with separate counters so down-proj starts before gate/up
   finishes. Our substrate ABI can express it; no opcode emits it.
3. **Parallel instruction emission.** Our compiler emits a serial chain
   (deps = previous instruction); the substrate supports fan-out.

TODO, strictly gated:

- [ ] **Substrate cost probe** (gate (b), ~1 day): a 64-layer trampoline
      program; the ~512 grid-wide counter barriers must cost <0.5 ms/token
      or the single-launch path is dead on SM120 and we stop here.
- [ ] **GEMV SMEM-paging prototype on ONE shape** (M=5120 decode gemv):
      refactor the NVFP4 GEMV body to read its A-operand from a pre-warmed
      SMEM page (the scoped multi-day refactor of
      `nvfp4_gemv_mma_kernel.cuh`). Paired microbench vs current GEMV.
      Kill-gate: <+20% kernel-level BW on that shape → abandon the lane
      (the Nsight audit from P0 predicts this number — trust it).
- [ ] **MLP chunked pipeline** (gate+up → SwiGLU → down in 4 chunks with
      counters, K-sliced FP32 accum for down). Expected from Hazy's
      numbers: MLP bucket 10.7 → ~7 ms.
- [ ] **Full-layer program with prefetch overlap**, then whole-decode
      single launch. End gate: **MTP=0 ≥ +15%** vs captured-graph baseline
      (i.e. ≥ 58 tok/s) at the first milestone, ≥ 75 tok/s at lane end,
      MTP parity floor intact, MTP=4 not regressed (defend the +7.3%).
- [ ] **lm_head NVFP4** (small, independent): 1.55 GB BF16 read = 0.87 ms
      floor of its measured 1.65 ms; FP4 → ~0.4 ms. ~+1 ms/token ≈ +5%
      MTP=0. Needs argmax-parity gate on the bench corpus (top-1 flips =
      fail, it feeds sampling directly).

Explicit non-goals for this lane (already falsified here): naive op fusion
without prefetch, L2-pinning tricks against graph replay, persistent grids
without occupancy-derived sizing (deadlock class, DAILY § 2026-05-23).

## P4 — Long context (only after P1 prefill ships)

- [ ] **Quest-style query-aware page sparsity** (B3): skip cold KV pages at
      decode; stacks with everything above. Relevant only ≥16K. Expected
      2–3× decode attention at 32K+. Gate: long-ctx parity battery + AL
      battery (sparsity perturbs verify numerics — same risk class as KV
      quant; the speculative loops are the canary).
- [ ] **Fused-store VRAM rework**: drop original gate/up + in_proj weights
      after building fused stores (prefill must consume the fused layout —
      stride-aware swiglu/conv reads, scoped in DAILY). Frees ~8 GB ⇒
      fusions stay ON at long context AND the 2–4K OOM trap disappears.
- [ ] **KV quant revisit (B1)** — only if 64K–262K becomes a real workload:
      port the KVarN recipe (K4/V2 + Hadamard, see inventory §2.5) into the
      tiled-attention decode path with LUT dequant. Prereq: fast-kernel
      support for quantized KV (same work TQ35 already needs).

## P5 — System floor (cheap, anytime)

- [ ] Lock GPU clocks (`nvidia-smi -lgc`) + persistence mode for benches;
      document the variance delta in DAILY (game/desktop contention has
      polluted runs repeatedly).
- [ ] Audit host sync points per decode cycle (each costs ~5–20 µs; the
      stream-invariant syncs are mandatory, find any that aren't).
- [ ] Pin host memory for the per-token D2H token reads.
- [ ] Sampling path: greedy argmax is graph-captured, but stop-condition
      checking is host-side per token — batch it with the token read.

## Sequencing

```
P0 (gates everything)
 ├─ P1 items in parallel (independent, all scoped)
 ├─ P2 fine-tune scoping  ──► the ONE training bet ──► routing
 └─ P0 Nsight audit ──► P3 cost probe ──► GEMV prototype ──► (gate) ──► full lane
P4 after P1-prefill; P5 opportunistic.
```

Rough potential if every gate passes (NOT additive promises):
MTP=0 50→~85 tok/s (P3+P5), DFlash typical 143→~300 tok/s (P2 AL≥7 +
graphed verify + P3 floor), prefill@64K 274→~1000+ (P1), decode@64K
33→~55 (P1+P4). Re-derive after each phase; kill fast, per the guardrails.

## Sources

- Xiaomi MiMo × TileRT, 1T @ 1000+ tok/s: https://mimo.xiaomi.com/blog/mimo-tilert-1000tps
  (FP4 experts via QAT + DFlash block=8 AL 6.3 + persistent-pipeline runtime)
- TileRT (tile-level persistent runtime, compiler-driven): https://github.com/tile-ai/TileRT
- Hazy Research low-latency megakernel ("no bubbles", Llama-1B, 78% BW):
  https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
  (SMEM paging, cross-instruction prefetch, counter-based chunked sync —
  the exact mechanisms our interpreter lane lacks)
- EAGLE-3.1 (vLLM blog 2026-05-26, 2× AL at long ctx vs EAGLE-3):
  https://vllm.ai/blog/2026-05-26-eagle-3-1
- Prior in-repo evidence: DAILY.md (tiled attention, split-K verify,
  interpreter gap analysis, AL battery), docs/code-inventory.md §2.5.
