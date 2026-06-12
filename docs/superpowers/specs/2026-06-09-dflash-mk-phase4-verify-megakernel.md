# DFlash megakernel Phase 4 — verify-pass launch-collapse + q=16 full-attn fusion

**Date:** 2026-06-09
**Status:** PROPOSED — gated, fail-fast, opt-in default off
**Chosen path:** `phase4-verify-megakernel` (path C, scoped tight)
**Depends on:** DFlash Phase 1 FA-tiled drafter substrate (`f32f4c6`); interpreter Stage 0-3 substrate (`91f7c77`)
**Estimated effort:** ~5 effort-weeks, but P1 is a hard kill gate that can end the project in ~9 days
**Hardware:** RTX 5090, sm_120a (GB202), 99 KiB SMEM/SM, 128 MiB L2, 32 GiB HBM3 @ 1.8 TB/s, no TMA multicast, no TMEM/tcgen05/wgmma, 128-bit max vector load
**Target model:** Qwen3.6-27B-NVFP4-Text-MTP (48 linear-attn + 16 full-attn + 1 MTP head; NVFP4 weights, Int8 TurboQuant KV)

---

## 1. Goal & motivation

Make the DFlash **verify pass** faster at 3K–7K context by removing the eager
kernel-launch overhead floor, then — *only if measurement justifies it* —
fusing the `q=16` full-attn flash tile that currently falls through to the
scalar GQA prefill kernel.

Phase 1 already proved the thing this phase deliberately does **not** chase.
The FA drafter kernel landed at **per-iter parity** with the v1 scalar loop —
141 ms vs 142 ms at ctx 3262, 256 ms vs 256 ms at ctx 7058 (Phase 1 spec
§11.1) — because the drafter shape (q_len=16, q_heads=32, head_dim=128) is
**launch- and bandwidth-bound, not compute-bound**, and the 32-CTA FA grid
under-uses the 192 SMs. Phase 1 §11.3 explicitly defers the real win to
Phase 4: "the verify step is [the bottleneck]... q=16 there is the prefill
shape that's actually compute-bound."

The bottleneck profile (upstream verify-profile finding) at 7K ctx:

| Stage | est. ms @ 7K | kernel calls | why |
|---|---:|---:|---|
| embedding (16 rows) | 0.2 | 1 | trivial gather |
| 48 linear-attn layers | 9 | 480 | state-resident DeltaNet; NVFP4 in/out proj streaming |
| **16 full-attn layers** | **70** | **192** | q=16 < `kPrefillFlashMinTokens=1024` (attention.cu:2502) → scalar GQA re-reads full 7K KV ~16× |
| 64 MLP blocks | 130 | 320 | NVFP4 gate/up/down weight streaming — largest bucket |
| final norm + lm_head + sample | 5 | 3 | weight-bound over vocab=248320 |

`verify_block_batched` (engine.rs:974) calls `self.prefill(tokens)`
**eagerly** — confirmed no graph capture exists for verify (graph plumbing in
engine.rs is wired only for `DecodeGraphKind::{Decode, MtpVerifyOne,
MtpVerifyMulti}`). At ~20 kernels/layer × 65 layers + tail that is ~1300
eager launches per verify pass. Verify is the **only forward path still
eager**; decode is already graph-captured. That is exactly where launch
overhead lives.

Two buckets are explicitly **left alone**: the MLP (the k=16 cuBLASLt-winning
GEMM shape that beat the per-block megakernel by 4% — fusing it sank a prior
attempt) and the final-norm+lm_head tail (lm_head is M=vocab N=16, cuBLASLt
wins). The attack surface is **launch overhead** (P1) and the **scalar-GQA
q=16 full-attn re-read** (P2).

## 2. Hardware target & KV layout

- sm_120a. No TMA multicast / TMEM / wgmma — we use `cp.async` + wmma
  `m16n16k16` (or `mma.sync m16n8k16`), the same pattern that
  `attention_flash_prefill.cu` already runs on this hardware.
- 99 KiB SMEM/SM cap. Per-CTA budgets below stay well under it.
- 128-bit max vector load (no 256-bit) — constrains the head_dim=256 staging
  path; reuse the proven `attention_flash_prefill.cu` cooperative-load
  pattern, which already respects this limit.
- **KV dtype reality (load-bearing):** the *production* full-attn KV dtype is
  **Int8 TurboQuant** (`is_tq_cache_dtype`, attention.cu:91). The FA prefill
  tile only handles `head_dim==256 && !is_tq_cache_dtype` (BF16/FP8). This
  must be resolved **before P2 starts** (see §6, R7): P2 either (a) dequants
  TQ→BF16 in SMEM inside the opcode (extra compute + parity risk per Sage
  B.2), or (b) scopes to the BF16/FP8-KV verify config only. Default
  assumption: settle in the P1 profile step by reading which dtype the
  full-attn layers actually use in the DFlash chat path.

## 3. Quality contract

- **Parity oracle (P0):** per-position verify-logits cos-sim vs the current
  eager `verify_block_batched`, swept over `k ∈ {2,4,8,16}` × `ctx ∈ {128,
  1024, 4096, 7168}`. Pass = cos ≥ 0.998 with **no accepted-token fork** at
  long ctx (the no-fork criterion is what the short-ctx-only FA-drafter smoke
  lacked — it forked at ctx~120 uncaught, Phase 1 §11.2).
- **The harness must be able to FAIL** before any kernel work proceeds:
  prove it by injecting a 1e-3 perturbation at ctx 7168 and confirming RED.
- **NVFP4 gate:** any FP4 codepath entering verify re-runs the
  decode-vs-prefill parity diagnostic (the cos 0.81→−0.19 bug class, Sage
  B.2 / AGENT.md) **in addition** to the cos-sim smoke.
- **Soft fallback:** the unchanged eager `prefill()` path stays the default.
  All new behaviour is behind env gates; absence falls back to eager.
- **Env vars:**
  - `QWEN36_VERIFY_GRAPH=1` — enable P1 graph capture of the eager verify loop
  - `QWEN36_VERIFY_MEGAKERNEL=1` — enable P2+ in-kernel opcode path
  - `QWEN36_PROFILE_PREFILL_CHUNKS=1` — extended with launch-count +
    CPU-launch-gap columns (P1)
  - all default **off**.

## 4. Tile / kernel design

### P1 — graph capture (no new kernel)
Capture the unchanged 65-layer eager prefill chunk loop
(`prefill_cuda_chunk`, engine.rs:3727) as one CUDA graph via the existing
`graph::{begin_capture, end_capture, instantiate, launch}` (crates/kernels/src/graph.rs:199-251).
Add `VerifyForward` to `DecodeGraphKind` (engine.rs:836) and a
`VerifyGraphState` mirroring `DecodeGraphState`. Re-capture when ctx crosses
a bucket boundary (reuse `graph_attention_context_limit`). Zero math change —
graph replay is bit-exact by construction.

### P2 — `ATTN_PREFILL_FLASH_Q16` opcode
Lift the head_dim=256 GQA q=16 wmma flash tile from
`attention_flash_prefill.cu` (kFlashM=32 with only 16 of 32 rows live,
kFlashN=64, kFlashD=256, kFlashWarps=4 → 128 threads) into an
interpreter-callable `__device__` body reading an `AttnPrefillScratch`.

Per-CTA SMEM (mirrors the proven prefill tile):

| Buffer | Bytes |
|---|---:|
| `sm_Q[16 × 256]` BF16 | 8 192 |
| `sm_K[64 × 256]` BF16 | 32 768 |
| `sm_V[64 × 256]` BF16 | 32 768 |
| `sm_S[16 × 64]` FP32 | 4 096 |
| `sm_P[16 × 64]` BF16 | 2 048 |
| `sm_m,l,alpha[16]` FP32 | 192 |
| **Total** | **~80 KiB** |

Under the 99 KiB cap; leaves no room to also hold decode scratch, so the
verify program uses a **separate kernel entry point**
(`qwen36_interpreter_verify_sm120`) with its own occupancy tuning — **do not
overload the decode interpreter** (decode is N=1 GEMV-shaped; verify is M=16
GEMM-shaped; the decode interpreter already regresses ~6.6% with its 86 KiB
page-allocator SMEM cap — R5).

- **Grid:** one CTA per (kv_head × single M-tile). 16 full-attn layers ×
  4 kv_heads = small grid per layer; CTA mapping must reach **≥70% SM
  occupancy** in nsight-compute or it is the 32-CTA FA-drafter anti-pattern —
  STOP and rethink (split along KV with a second reduction pass if needed).
- **MMA atom:** wmma `m16n16k16` BF16 inputs / FP32 accum (proven on this
  hardware). Online softmax in SMEM, `o ← o·α + P·V`, `l ← l·α + sum(P)`.
- Route **only the 16 full-attn layers' attention** through the opcode; rest
  of verify stays on the P1 captured graph.
- KV: BF16/FP8 first (matches the FA tile's `!is_tq_cache_dtype` gate); Int8
  TurboQuant dequant→BF16-in-SMEM deferred to a later substage if the
  production path needs it (R7).

### P3 (conditional) — chain attn+norm+residual
Only if P2's nsight-compute shows GMEM round-trip cost between the attn opcode
and adjacent RMSNorm/residual. Keep the 16×hidden activation resident in
SMEM/L2 across `attn → RMSNorm → residual` within a layer; sync the per-layer
dependency chain with an **explicit barrier opcode** (`NOOP_BARRIER`), not the
FALLBACK_TRAMPOLINE. **MLP stays on cuBLASLt** (`run_mlp_fused_combined_gemm_rows`).

## 5. Phased delivery (each phase has a revert-triggering gate)

### P0 — Verify parity harness + q_len=16 drafter parity gate (4 days)
Build the FAIL-able test **first**. Add the per-position verify-logits cos-sim
smoke (k × ctx sweep) wired into `smoke_cuda.sh`. Bundle the cheap missing
**q_len=16 drafter FA parity gate** (Phase 1 task #49, never implemented
because smoke.cu:4010 uses `q_len=2` so the FA gate at
drafter_attention_flash.cu returns NOT_IMPLEMENTED and the path is never
exercised). Lift the existing CPU reference loop (smoke.cu:4032-4062) to
q_len=16, add the symmetric SWA mask from drafter_attention.cu:96-104.
- **Gate:** harness reproduces ≥0.9999 eager-vs-eager (sanity) AND turns RED
  on an injected 1e-3 perturbation at ctx 7168. Drafter q_len=16 gate passes
  cos≥0.998 vs v1 at kv ∈ {16,64,128,1024,4096} and SWA ∈ {0,2048}. **No perf
  claim.** This permanently closes the "drift bug" as not-a-kernel-bug
  (upstream fa-drift-diagnosis: cos 0.999998 everywhere, maxabs = 1 BF16 ULP).

### P1 — KILL-OR-CONTINUE MVP: graph-capture the eager verify loop (5 days)
Add launch-count + CPU-launch-gap instrumentation to
`QWEN36_PROFILE_PREFILL_CHUNKS`. Capture the 65-layer eager prefill loop in
`verify_block_batched` as one CUDA graph (`VerifyForward` kind). NO new
kernels.
- **HARD GATE.** Captured graph bit-exact vs eager (cos 1.0). Then:
  - **(a) ≥+3% end-to-end DFlash chat tok/s at 3K** → consider shipping the
    graph and STOP, unless launch-idle profile still shows >15% host-launch
    idle (then proceed to P2).
  - **(b) <+1% tok/s AND launch-idle <5%** → verify is compute-bound:
    **ABANDON the rest of Phase 4**, write the negative result, pivot to the
    deferred NVFP4-KV phase. (This is the cheap probe that avoids the
    Phase-1-redux trap — R-second.)
  - Proceed to P2 only if launch overhead is real **and** the graph leaves a
    measurable residual.

### P2 — `ATTN_PREFILL_FLASH_Q16` opcode (8 days)
Fuse only the q=16 full-attn flash tile (§4). Run the NVFP4 parity diagnostic
+ the P0 harness on every change.
- **Gate:** cos ≥0.998 at every k×ctx in the P0 harness with **no long-ctx
  fork**; nsight-compute shows the opcode at ≥70% SM occupancy (<50% → STOP,
  CTA mapping wrong); per-iter verify ms at 7K strictly < the P1
  captured-graph baseline; end-to-end DFlash tok/s at 7K beats the v1
  baseline. Revert opcode if not.

### P3 (conditional, gated by P2 profile) — chain attn+norm+residual (8 days)
Only start if P2 shows residual GMEM round-trip cost. Keep MLP on cuBLASLt.
- **Gate:** cos ≥0.998 at P0 sweep AND end-to-end DFlash tok/s strictly > the
  P1 graph baseline by ≥+3% at 3K. If it merely matches the graph baseline,
  this is Stage-F.4 redux: keep opt-in default OFF, write the negative result,
  STOP (do not roll into NVFP4-KV inside this phase).

**Global gate (all phases):** every phase must move (or be killed by) a
concrete DFlash chat tok/s number at 3K **and** 7K ctx. No
parity-green-perf-neutral substrate ships default-on.

## 6. Risks & exit ramps

| # | Risk | Exit ramp |
|---|---|---|
| R1 (MOST LIKELY) | P1 graph alone recovers the whole launch win, making P2+ perf-neutral. | This is a **success** — ship the graph. The P1 gate catches it before any kernel is written. |
| R2 (SECOND) | Verify at k=16 is already compute/cuBLASLt-bound per-layer → megakernel lands at exact parity, zero win (FA-drafter outcome, 141 vs 142 ms). | P1's launch-idle profile is the cheap probe; gate (b) abandons before kernel work. |
| R3 | Effort blowout from parity bisection (FA-drafter was "3-5 days", shipped same-day but with undiagnosed drift). | Budget ~2× the kernel-writing time in P2/P3 for parity debugging; P0 harness exists first so bisection is cheap. |
| R4 | wmma online-softmax accumulation across many K-tiles silently changes **accepted** verify tokens (corrupts output, not just speed). | P0 long-ctx harness with **no-fork** criterion gates every P2 change. Short-ctx smoke alone is what let FA-drafter fork uncaught. |
| R5 | Persistent single-PC interpreter under-occupies for M=16 GEMMs (decode interpreter already −6.6% with 86 KiB page-allocator cap). | **Separate verify entry point** with own occupancy tuning, not reuse of decode kernel. |
| R6 | NVFP4-into-attention re-opens the decode-vs-prefill divergence (cos 0.81→−0.19). | NVFP4 parity diagnostic re-run at every FP4 substage, not just cos-sim. |
| R7 | Int8 TurboQuant is the production KV dtype but the FA tile supports only BF16/FP8 (`head_dim==256 && !is_tq_cache_dtype`). Misjudging scopes out the production path. | **Settle before P2 starts:** read the actual full-attn KV dtype in the DFlash chat path during P1; choose dequant-in-SMEM (parity risk) vs BF16/FP8-only scope explicitly. |

## 7. File layout

**Create:**
- `kernels-cuda/interpreter/opcodes/attn_prefill_flash_q16.cuh` — q=16 head_dim=256 GQA wmma flash opcode, lifted from `attention_flash_prefill.cu`
- `kernels-cuda/interpreter/verify_program_sm120.cu` — verify-specialized program builder + separate `qwen36_interpreter_verify_sm120` entry point

**Modify:**
- `kernels-cuda/smoke.cu` — verify-pass per-position cos-sim smoke (k×ctx sweep) + the q_len=16 drafter FA parity block (P0)
- `scripts/smoke_cuda.sh` — wire the new smoke cases
- `kernels-cuda/drafter_attention_flash.cu` — (P0) optional BF16-sum/denom consistency tweak (~2.4e-4 effect, removes the one real flash-vs-v1 structural asymmetry; correctness-neutral)
- `kernels-cuda/interpreter/instruction.h` — add `ATTN_PREFILL_FLASH_Q16 = 16` (+ `NOOP_BARRIER` for P3), bump the opcode-known ceiling (`is_opcode_known`), keep the 152-byte `static_assert`
- `crates/kernels/src/interpreter.rs` — mirror the new opcode enum + sizeof static_assert; typed instruction constructor
- `crates/runtime/src/engine.rs` — `verify_block_batched`: P1 graph-capture path behind `QWEN36_VERIFY_GRAPH`; P2 interpreter-verify dispatch behind `QWEN36_VERIFY_MEGAKERNEL`; extend `QWEN36_PROFILE_PREFILL_CHUNKS` with launch-count + CPU-gap columns; add `DecodeGraphKind::VerifyForward` + `VerifyGraphState`
- `crates/runtime/src/cuda_graph.rs` — `VerifyForward` graph-kind plumbing alongside the existing buckets
- `scripts/build_cuda.sh` — compile `verify_program_sm120.cu` with `-arch=sm_120a`; add the new smoke case to the test list

## 8. Out of scope

- **Fusing the k=16 MLP** — cuBLASLt-winning shape (`run_mlp_fused_combined_gemm_rows`); sank the per-block megakernel −4%.
- **NVFP4 KV / BitDecoding** — deferred to a later phase, gated on Phase 4 first removing the launch floor (its own recommendation field says "after-verify-megakernel"; no vendorable sm_120a FP4 kernel exists — the open repo incl. blackwell branch ships only Int2/Int4 FP16 sm_80/sm_90).
- **Chasing the FA-drafter drift as a kernel bug** — proven non-bug at cos 0.999998 (fa-drift-diagnosis). Closed by the P0 permanent gate.
- **final-norm + lm_head tail fusion** — lm_head M=vocab N=16 favors cuBLASLt.
- **TMA / FA4** — FA4 is sm_100/sm_103 only (needs TMEM/tcgen05/wgmma GB202 lacks). FA2-style register-resident mma.sync is the sm_120 ceiling.

## 9. References

- `docs/superpowers/specs/2026-06-09-dflash-fa-drafter-attention.md` §11 — Phase 1 outcome (per-iter parity 141/142 ms, 256/256 ms; §11.3 defers to Phase 4) — **critical reading**
- `kernels-cuda/attention_flash_prefill.cu` — kFlashM=32/kFlashN=64/kFlashD=256 wmma tile to lift (P2)
- `kernels-cuda/attention.cu:2502-2556` — `kPrefillFlashMinTokens=1024` / `kPrefillGqaMinTokens=16` dispatch (the q=16 fall-through this phase attacks)
- `crates/runtime/src/engine.rs:974` (`verify_block_batched`), `:3727` (`prefill_cuda_chunk`), `:836` (`DecodeGraphKind`)
- `crates/kernels/src/graph.rs:199-251` — capture/replay API to reuse
- `kernels-cuda/interpreter/instruction.h` — opcode enum (next free = 16)
- BitDecoding (HPCA 2026, arXiv 2503.18773) — deferred NVFP4-KV phase reference only
- `docs/superpowers/notes/2026-06-09-long-context-decode-roadmap.md` — technique survey (note: FA4 not on GB202)

---

### First concrete action
Start with **P0** by adding the FAIL-able parity harness. The first edit is to
`kernels-cuda/smoke.cu`: add a `q_len=16` drafter FA parity block (lifting the
existing CPU reference loop at smoke.cu:4032-4062 to q_len=16, with the
symmetric SWA mask from drafter_attention.cu:96-104) and assert cos≥0.998
flash-vs-v1 and flash-vs-fp32-ref. This both closes Phase 1 task #49 and
establishes the oracle pattern the verify-logits smoke (P0 part 2) will reuse.