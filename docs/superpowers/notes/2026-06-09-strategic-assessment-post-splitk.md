# Strategic assessment — post split-K verify lane

**Date:** 2026-06-09 (end of the megakernel sprint, after `05c1d28`)
**Author:** Claude (session assessment, written at user request)
**Scope:** where the speedoza perf effort actually stands, what worked,
what the risks are, and where the next real gains live.

## 1. What actually happened

The sprint started from the TileRT/MiMo 1000-TPS inspiration with a
"megakernel, 4–6 months" plan. Measurement rewrote that plan at every
step. The factual outcome:

- **The main win came from a ~150-line kernel** (`attention_flash_splitk.cu`,
  Flash-Decoding split-K for the q=16 verify shape) found by *profiling*,
  not from the thousands of lines of interpreter substrate.
- Verify chunk at 7K ctx: **225 → 49 ms (4.6×)**. DFlash end-to-end:
  **2.2–4.3×** depending on context. Default-on, parity-clean (144-case
  smoke gate, BF16+FP8), adversarial-review-cleared, capture-safe.
- Every plausible *model-based* hypothesis was **falsified by measurement**,
  four times in a row:
  1. "Verify is launch-overhead-bound" → killed by the P1 profile
     (full_attn = 89% of the chunk, one 200 ms GPU stage, not 1300
     launches).
  2. "FA-tiling the drafter attention gives 3–5×" → per-iter parity
     (141 vs 142 ms); the drafter shape is latency-bound.
  3. "The existing scalar split-GQA is a free win" → 5× at 7K but lossy
     (AL 9.18 → 4.17 at 3K, regressed end-to-end).
  4. "M=16 tile ≈ 2× on full_attn" → paired microbench measured ~8% on
     the kernel, ~3-4% on the chunk (latency-bound, 1 CTA/SM for both
     tile sizes).

**The central lesson: the kill-gate / paired-microbench discipline is
what saved weeks — not the architecture.** A measurement-first probe
costs hours; a model-first build costs weeks and has been wrong every
time it was tested here.

## 2. Current state of the two lanes

### DFlash verify (Claude) — COMPLETE

P0 → P2.1 + #55 all shipped and pushed. 4.6× chunk, 2.2–4.3× e2e,
default-on, 144-case parity gate, engine-owned partials, capture-safe.
`scripts/verify_perf_gate.sh` protects both lanes from regressions.

Remaining kernel levers are known but deliberately NOT taken
(~10-20% each for 1–2 weeks each, diminishing returns):
- vectorized / LUT FP8 decode in the split-K inner loop
- shrink the 64 KB KV SMEM tile (or double-buffer) to get >1 CTA/SM
- fix the ~11-of-48 empty-split load imbalance at 7K

### Interpreter decode (Codex) — IN FLIGHT, carries sunk-cost risk

~5500 lines of substrate. Measured outcome so far: **MTP=4 +7%, MTP=0
negative**. Auto-policy for MTP landed (`5111903`); MLP gate/up pair
opcode + CTA-0-only prefetch landed since. The remaining promised
mechanisms (MLP sub-instruction chunking, SMEM weight prefetch) are
exactly the same class of "the model says it should help" claims that
failed four times elsewhere.

**Recommendation:** hold the lane to the same standard — one measured
gate per phase. MLP chunking is the last untested hypothesis of the
original spec and deserves its test; **if it does not show ≥ +3 tok/s,
archive the interpreter as MTP>0 opt-in and stop investing.** Codex's
own criterion ("default-on that wins or at least doesn't regress") is
correct; it just has to be enforced.

## 3. The strategic shift: AL is now the binding constraint at long ctx

The numbers that matter (coherent prompts, FP8 KV, split-K default-on):

| ctx | tok/s | AL | what limits it |
|---:|---:|---:|---|
| 3262 | 144–155 | 8.3–9.0 | ~balanced |
| 5484 | 107 | 7.9 | ~balanced |
| 7815 | **40** | **2.78** | **AL, not kernel time** |

At 7815 tokens the verify chunk costs 49 ms and is near its cheap-win
floor. What bounds throughput there is the **acceptance length**: at
AL 2.78 each ~60 ms iteration commits ~3.8 tokens. If AL were 8, the
same kernels would deliver ~100+ tok/s **with zero additional kernel
work**. The whole sprint optimized the denominator (cost/iter); at long
context the lever is now the numerator (tokens/iter).

This is work of a different nature — drafter conditioning, not CUDA:
- which target layers feed the hidden-state capture
  (`target_layer_ids = [1,16,31,46,61]` today)
- drafter context handling at long ctx (the drafter re-attends over the
  full captured context with its 5 layers)
- block size (16 today) — possibly adaptive by observed AL
- ultimately a drafter fine-tune on long-context data if the cheap
  knobs plateau

## 4. Hardware reality check

1000 TPS will not happen on a 5090 with a 27B model. The bandwidth
ceiling is ~200 tok/s for pure decode; DFlash already measured 313
tok/s peak at short ctx (earlier sweep) and ~150 at 3K. We are at
roughly 50–70% of the realistic speculative ceiling at short/mid ctx.
The remaining headroom at short ctx is modest; the *large* headroom is
at long ctx, and it is AL-shaped, not kernel-shaped.

## 5. Recommended order of work

1. **Codex finishes MLP chunking with a hard gate** — last untested
   spec hypothesis; then a definitive verdict on the interpreter.
2. **Open the long-context AL lane** (next session's focus): cheap
   experiments first — capture-layer choice, block size, drafter
   context window — before considering a fine-tune.
3. Keep NVFP4 KV and the big kernel levers in reserve — high cost,
   modest gain, and the FP4 divergence bug class that already cost
   weeks (Sage B.2, decode-vs-prefill).

## 6. Process notes

- Two agents in one working tree works: interleaved commits stayed
  clean because each lane stages by path and shared files (engine.rs,
  gpu.rs) were touched in disjoint regions. Keep committing frequently.
- `scripts/verify_perf_gate.sh` is the shared regression gate — run it
  before and after any change in either lane.
- The danger is no longer technical, it is strategic: continuing to
  dig kernels out of inertia while the constraint has moved to drafter
  quality.
