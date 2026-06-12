# Interpreter Megakernel for Qwen3.6-27B decode (SM_120)

**Date:** 2026-06-08
**Status:** Design — pending user approval
**Depends on:** main HEAD `9cc92fc` (post Phase-2 per-block megakernel archive)
**Estimated effort:** 4–6 weeks of focused CUDA work; 6 sub-deliverables
**Supersedes / alternative to:** Mirage Phase 4 (cross-layer CUTLASS persistent kernel)

## 1. Goal and motivation

Land a single persistent kernel that hosts the **entire decode forward pass**
(embed → 64 layers → final RMSNorm → lm_head → sample) as a sequence of
**short, counter-synchronised instructions executed by an on-SM interpreter**,
rather than as N separate CUDA kernels (graph-captured or not).

This is a deliberately different architecture from the two megakernel
attempts already on the books:

| Effort | Model | Status |
|---|---|---|
| `docs/mirage-megakernel.md` Phase 1 | CUTLASS NVFP4 GEMM swap for cuBLASLt | parity, not faster (`AGENT.md` 2026-05-04) |
| `docs/mirage-megakernel.md` Phase 2 | CUTLASS epilogue fusion (SwiGLU/RMSNorm) | mathematically blocked on `MlpFusedStore` |
| Per-block megakernel Phase 2 (Stage A→F.4) | Persistent grid + atomic barrier + work-stealing, fuse one full-attn layer | **−4 % MTP=0** (`AGENT.md` 2026-05-23), kept opt-in |
| **This spec** | On-SM interpreter, counter-based GMEM deps, sub-instruction chunking, SMEM paging — whole decode in one launch | proposed |

The two prior efforts fused operators that cuBLASLt was already winning at
batch=1. The bench loss came from replacing CUTLASS-tuned MLP GEMMs with
hand-rolled GEMV. **The interpreter design does not try to beat cuBLASLt at
GEMM**: it calls the same matvec body the standalone path uses, then attacks
the orthogonal bottleneck — the activation store/load + counter sync time
*between* matvecs, which the captured graph cannot pipeline.

### Why this is worth retrying after two negative megakernel results

The two prior negative results both attacked **one layer**: launches saved
per token were small (~7 → 1 for full-attn, ~4 → 1 for MLP), and the captured
CUDA graph already amortises those host-side launches.

The interpreter attacks a different bottleneck: **inter-operator data
movement on the GPU**. In the Hazy Research Llama-1B megakernel
([blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles))
this slice was **250 µs of a 600 µs B200 token** — 41 %. On our token budget
of 22 ms it is plausibly **1–4 ms / token** (4–18 %) depending on how much
of the inter-op write/read hits L2 vs HBM. Bound below by the irreducible
weight-load and matmul time, the realistic gain is capped at this slice
plus the inter-CTA sync overhead our current path pays at each graph node
boundary.

Concretely, three mechanisms move from "kernel-boundary-blocked" to
"on-the-fly-pipelined" inside one kernel:

1. **Weight prefetch of instruction N+1 starts while instruction N is still
   storing.** Today each cuBLASLt / GEMV kernel exits before the next one
   begins reading weights, so the next launch's first warp must stall on
   HBM for its first weight tile.
2. **Sub-instruction chunking** of the MLP intermediate (17408 = 4 × 4352).
   Down-proj can start consuming the first chunk of SwiGLU output the
   moment that chunk's counter is incremented; today it waits for the full
   17408 vector to be written.
3. **Counter-based deps in GMEM are finer than CUDA Graph node deps.** A
   CUDA Graph node depends on the previous node finishing entirely. The
   counter array lets a single SM start the next instruction as soon as
   *its* prerequisite counters are met, even if other SMs are still
   finishing the predecessor.

### Bench projection (decode MTP=0, RTX 5090 native Linux, baseline 55.27 tok/s)

| Source of gain | Best case | Realistic |
|---|---:|---:|
| Eliminate ~64 layer × 8 graph node = 512 inter-node sync points (~0.3 µs each on graph replay) | +0.15 ms | +0.1 ms |
| Weight prefetch overlap (instruction N+1 weights begin loading during instruction N store) | −2.0 ms | −0.8 ms |
| Sub-instruction MLP chunking (down-proj starts consuming SwiGLU chunk-by-chunk) | −1.5 ms | −0.6 ms |
| Activation residency in SMEM/registers between Q/K/V and o_proj (when sizes permit) | −0.4 ms | −0.2 ms |
| **Total** | **−4.0 ms (−18 %)** | **−1.7 ms (−7.7 %)** |

Realistic translates to **~60 tok/s MTP=0** (from 55.27). Best case is
~65 tok/s. **Multiplicative with MTP** so MTP=4 best case is ~130 tok/s
(from 110.78).

Decision gate: if Stage 2 below does not show **≥ +3 tok/s on MTP=0** the
whole branch is reverted to the captured graph. We do not ship a 4-week
neutral change.

## 2. Quality contract

**Hard parity (regression gate, blocks merge):**
- `chat --prompt "hello" --max-new-tokens 12` produces identical token
  streams for `--mtp-speculative-tokens` ∈ {0..4} with the interpreter
  enabled vs. the captured-graph path. (Same gate every stage of the
  per-block megakernel passed.)
- `chat --prompt "hello world" --max-new-tokens 12` same.

**Soft parity:** cos sim ≥ 0.998 against PyTorch reference at every layer
boundary (`scripts/decode_parity.py` + `QWEN36_DEBUG_DUMP_DIR`). The
borderline-prompt 1–2 token drift envelope must not widen.

**Op-level parity gate:** every instruction in the interpreter ships with
a smoke that calls **only that instruction** through the interpreter's
dispatch loop and diffs against the standalone kernel for the same
inputs. Failure to diff bit-exact blocks the instruction from being
enabled in the interpreter schedule (it remains in the per-op fallback).

**Soft fallback:** the env var
`QWEN36_INTERPRETER_OPCODES_ENABLED=<comma-separated>` selects which
instruction opcodes the interpreter executes; anything not in the list
falls back to the existing per-op kernel call. This lets us ship the
substrate first (interpreter dispatches every opcode to a fallback
trampoline) and migrate ops one at a time, just like Stage A→F.4 did.

## 3. Hardware target

- RTX 5090, Blackwell SM_120, 99 KB SMEM per SM (vs 228 KB on B200).
- 192 SMs total. Interpreter grid sized via
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (same trick that
  unblocked Stage F.4, `AGENT.md` 2026-05-23 bug #1).
- 32 GB HBM3, 128 MB L2.
- No TMEM. **No TMA multicast.** Counter polls use plain
  `atomicAdd` / `__threadfence_system` on GMEM ints.
- FP4 MMA atom `mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32` from
  the existing `decode_gemv/nvfp4_gemv_mma_kernel.cuh` body.

The 99 KB SMEM budget is the design's hard constraint. The Hazy Llama-1B
design uses 13 × 16 KiB pages = 208 KiB on H100. We have ~46 % of that
budget. Initial page plan:

- 4 × 16 KiB pages for weight tiles (one currently in use, one prefetching,
  two staging for next instruction)
- 2 × 8 KiB pages for activations (current input + next chunk)
- 1 × 4 KiB scratchpad
- 1 × 2 KiB counter mirror + opcode table
- Total: 90 KiB. Fits with margin for register spill.

If the page math collides with realistic instruction working sets,
we cut to 3 weight pages and accept the prefetch overlap loss; bench
delta gates whether the design still wins.

## 4. Existing baseline to beat

`AGENT.md` 2026-05-23, RTX 5090 native Linux, median of 5,
prompt=128 max-new=32:

| Config | tok/s |
|---|---:|
| `mtp=0` baseline | **55.27** |
| `mtp=0` Stage F.4 megakernel | 53.05 (−4 %) |
| `mtp=4` baseline | **110.78** |
| `mtp=4` Stage F.4 megakernel | 110.88 (noise) |

Token budget at 55.27 tok/s ≈ 18.1 ms/token. WSL2 → native Linux delta is
already captured in this number. Per `QWEN36_PROFILE_DECODE_LAYERS=1`,
the breakdown today is roughly: 9 ms linear-attention layers + 5 ms MLP
+ 1.5 ms full-attention + 2.5 ms other (embed, RMSNorm, lm_head,
sample, sync). The interpreter targets the **inter-op slice within each
of those buckets**, not the buckets themselves.

## 5. Proposed kernel design

### 5.1 Interpreter loop

One kernel: `qwen36_interpreter_decode<SM120>`. Grid sized to fill 192
SMs at the chosen occupancy (TBD by `cudaOccupancyMaxActiveBlocksPer
Multiprocessor` × 192, expected 1–2 CTAs/SM). Per CTA:

```cuda
__shared__ InterpreterState st;
__shared__ PageAllocator pages;

if (threadIdx.x == 0) {
    st.pc = 0;
    pages.init();
}
__syncthreads();

while (true) {
    Instruction insn = program[st.pc];     // GMEM, ~16 B per instruction
    if (insn.opcode == OPCODE_EXIT) break;

    // Wait for counters this instruction depends on (per-SM, not global)
    for (auto& dep : insn.deps) {
        while (counters[dep.id] < dep.target) {
            __nanosleep(8);                // tiny back-off
        }
    }

    // Dispatch
    switch (insn.opcode) {
        case OPCODE_RMSNORM_NVFP4_QUANT: exec_rmsnorm_quant(insn, pages); break;
        case OPCODE_NVFP4_GEMV:          exec_nvfp4_gemv(insn, pages);    break;
        case OPCODE_SWIGLU_QUANT:        exec_swiglu_quant(insn, pages);  break;
        case OPCODE_ROPE_PARTIAL:        exec_rope_partial(insn, pages);  break;
        case OPCODE_ATTN_DECODE:         exec_attn_decode(insn, pages);   break;
        case OPCODE_DELTANET_RECUR:      exec_deltanet_recur(insn, pages);break;
        case OPCODE_RESIDUAL_ADD:        exec_residual_add(insn, pages);  break;
        case OPCODE_FALLBACK_TRAMPOLINE: exec_fallback(insn);             break;
        // ...
    }

    // Publish this CTA's contribution
    if (this_cta_finishes_phase) {
        atomicAdd(&counters[insn.publishes], 1);
    }
    st.pc++;
}
```

The instruction stream is **precompiled host-side** by Rust into a
`Program` struct uploaded once at engine init. Each layer contributes a
fixed sequence of opcodes — `LinearAttnLayer` emits a different sequence
than `FullAttnLayer`, but both are static at compile time.

### 5.2 Counter array

Pre-allocate ~4 KiB of `int32` counters in GMEM, zeroed every decode
step. Each instruction declares:
- `publishes: counter_id` — incremented by the **last CTA to finish** the
  phase (the others use a per-instruction `cooperative_arrival_token`).
- `deps: [(counter_id, target_value)]` — counters to spin on before
  starting.

Sub-instruction chunking: an instruction operating on a 17408-element
intermediate has 4 sub-counters (one per 4352 chunk). The consumer
instruction declares 4 deps and consumes chunk-by-chunk.

Per-CTA arrival count uses the same `arrive_inc` / `wait_eq` pattern
already in `kernels-cuda/megakernel/full_attn_block_sm120.cu` Stage A.

### 5.3 Shared-memory paging

Page table in SMEM `PageAllocator` with explicit acquire/release calls:

```cuda
PageHandle h = pages.acquire(PAGE_KIND_WEIGHT, 16 * 1024);
cp_async_bulk(pages.ptr(h), weight_gmem_ptr, 16 * 1024);
cp_async_bulk_commit_group();
// ... do other work ...
cp_async_bulk_wait_group<0>();
// ... use weights from pages.ptr(h) ...
pages.release(h);
```

`cp.async.bulk` on SM120 fills the role TMA played in Hazy's H100 design.
Released pages immediately become acquirable by the next instruction's
prefetch.

### 5.4 Instruction set (initial)

Bring-up subset (Stage 0–2 below):

| Opcode | Backing kernel | Notes |
|---|---|---|
| `RMSNORM_NVFP4_QUANT` | `ops.cu::rmsnorm_nvfp4_quantize` | Aliasing rule re. residual still applies. |
| `NVFP4_GEMV` | `decode_gemv/nvfp4_gemv_mma_kernel.cuh` body | Already extracted into a shared `__device__` body (commit `3aa58de`). |
| `SWIGLU_NVFP4_QUANT` | `ops.cu::swiglu_nvfp4_quantize` | Bug from `AGENT.md` 2026-05-23 #2 — warp-wide mask. |
| `ROPE_PARTIAL` | `ops.cu::rope_partial` | factor 0.25 hardcoded for Qwen3.6. |
| `RESIDUAL_ADD` | inline | one warp per chunk. |
| `FALLBACK_TRAMPOLINE` | host-launched kernel | escape hatch; not faster than today, just keeps the schedule running. |

Stage 3+ adds:

| Opcode | Backing kernel | Notes |
|---|---|---|
| `ATTN_DECODE_FULL` | `attention.cu` | 24-CTA body uses `__syncthreads`; same deferred Stage D constraint as Phase 2. |
| `DELTANET_RECUR` | `deltanet.cu::deltanet_recurrence_one_step` | Per-layer state must stay in registers across chunks. |
| `ATTN_RESIDUAL_FUSED` | inline post-attn RMSNorm + quant | Reuses Stage E semantics. |
| `LM_HEAD_TILED` | new tiled lm_head | M=248320 × K=5120 is enormous; tile across SMs, reduce via counters. |

Embed and sample stay host-side (one call each, negligible).

### 5.5 File / module layout

```
kernels-cuda/
  interpreter/                       <- new directory
    interpreter_sm120.cu             <- the single megakernel
    instruction.h                    <- Opcode enum, Instruction struct, ABI
    page_allocator.cuh               <- SMEM paging primitives
    counters.cuh                     <- arrive/wait helpers
    opcodes/                         <- one .cuh per opcode body
      rmsnorm_nvfp4_quant.cuh
      nvfp4_gemv.cuh
      swiglu_nvfp4_quant.cuh
      rope_partial.cuh
      residual_add.cuh
      attn_decode_full.cuh           <- Stage 3+
      deltanet_recur.cuh             <- Stage 3+
      lm_head_tiled.cuh              <- Stage 3+
crates/kernels/src/interpreter.rs    <- Rust ABI for Instruction/Program
crates/runtime/src/interpreter_compile.rs <- Program builder per topology
```

The interpreter body **must not duplicate** opcode logic. Each
`opcodes/*.cuh` exports a `__device__` body with the same numerical
contract as the standalone kernel; the standalone wrapper in `ops.cu` /
`decode_gemv/` is rewritten to call that body. This is the lesson from
`full_attn_block_sm120.cu` duplicating NVFP4 codec helpers — the
parity smoke caught nothing for ages.

ABI sync rule (`AGENT.md`): every change to `instruction.h` mirrors into
`crates/kernels/src/interpreter.rs`.

## 6. Phased delivery

Each stage gates on parity + a measurable bench delta. The Stage F.4
lesson stands: a parity-green megakernel that regresses by 4 % is **not
shippable as default**.

### Stage 0 — Substrate (1 week)

- Interpreter shell that dispatches every opcode to
  `FALLBACK_TRAMPOLINE`. Counters, pages, opcode table all wired but
  every body is a host-launched kernel call inside the trampoline. This
  isolates the dispatch overhead.
- Rust `Program` builder that emits the static instruction stream for
  one decode step.
- Smoke: interpreter executes a 1-layer LinearAttn schedule, output
  bit-equal to host-launched path.
- Gate: ≤ 5 % bench regression vs baseline at MTP=0. Anything worse
  means the dispatch overhead itself is the problem; revisit.

### Stage 1 — Two cheapest opcodes inline (1 week)

- Inline `RMSNORM_NVFP4_QUANT` and `RESIDUAL_ADD` bodies into the
  interpreter. Everything else still trampolines.
- Op-level parity smoke per opcode.
- Gate: MTP=0 parity-green; bench parity ± 1 %. (Just establishing
  the inlining path works.)

### Stage 2 — GEMV + SwiGLU inline + weight prefetch (2 weeks)

- Inline `NVFP4_GEMV` and `SWIGLU_NVFP4_QUANT`. Wire
  `cp.async.bulk`-based weight prefetch from the GMEM weight pointer to
  the next page in the page table.
- Sub-instruction chunking on MLP intermediate (4352-element chunks,
  4 sub-counters).
- **This is the bench-decision stage.** Gate:
  - Parity green on full chat MTP=0..4.
  - MTP=0 bench **≥ +3 tok/s vs baseline** (58.3 tok/s+).
  - If the gain is below this floor, the branch is reverted and the
    captured graph is restored as default. Stage 0–2 work stays in
    repo as `QWEN36_INTERPRETER_DECODE=1` opt-in for diagnostics.

### Stage 3 — Attention + DeltaNet recurrence (1.5 weeks)

- Inline `ATTN_DECODE_FULL` (resolving the 24-CTA `__syncthreads`
  constraint Phase 2 deferred — likely via `cudaLaunchCooperativeKernel`
  for the whole interpreter).
- Inline `DELTANET_RECUR` keeping per-layer recurrent state in
  registers across the next layer's input RMSNorm reads.
- Gate: parity green; MTP=0 bench **≥ +5 tok/s** (60.3+).

### Stage 4 — lm_head tiling + final RMSNorm (0.5 week)

- Inline `LM_HEAD_TILED`. M=248320 × K=5120 split across grid; reduce
  via the same counter pattern.
- Gate: parity green; MTP=0 bench **≥ +6 tok/s** (61+).

### Stage 5 — Default-on flip (after a soak week)

- Soak: continuous bench loop on `hello`, `hello world`, and a 1k-token
  prompt for 24 hours. No parity drift, no perf regression.
- Flip default to `QWEN36_INTERPRETER_DECODE=1`. Captured-graph path
  stays available via `=0`.
- Update `AGENT.md` with bench table.

## 7. Risks and exit ramps

| Risk | Severity | Exit ramp |
|---|---|---|
| Dispatch overhead alone tanks Stage 0 by > 5 % | Medium | The opcode table + counter polls are too heavy. Move to **template-specialised interpreter** (one specialised interpreter per `LayerKind`, no opcode switch) before continuing. |
| Page allocator fragments under realistic instruction mix | Low | Drop to **statically allocated** pages, one slot per opcode kind. Loses some flexibility, keeps determinism. |
| Stage 2 hits the bench floor for the wrong reason (e.g. cuBLASLt MLP GEMM wins again) | High — same as Phase 2 | **The Stage 2 instruction set deliberately excludes cuBLASLt MLP GEMM.** MLP GEMV stays as our own kernel (already at parity with cuBLASLt per `AGENT.md` 2026-05-23). If even the GEMV path loses, the gain hypothesis is falsified. |
| Cooperative launch needed at Stage 3 cannot coexist with CUDA Graph | Medium | The interpreter does not need to be inside a graph — its whole point is to absorb everything one graph would do. The graph wrapping it just records `cudaLaunchCooperativeKernel` as one node. |
| Per-CTA counter polls thrash L2 | Medium | Move to per-cluster `mbarrier` once SM120 supports cluster launches in our toolchain. Until then, exponential `__nanosleep` back-off (already in the Stage A barrier body). |
| Total interpreter SMEM exceeds 99 KiB | Medium | Cut to 3 weight pages, drop activation paging, accept the prefetch overlap loss. Re-test gate. |

## 8. Out of scope

- **Multi-GPU.** Interpreter runs on one device. TP is a separate effort.
- **Multi-batch decode.** N=1 only, same as today.
- **Prefill.** Prefill stays on the existing path. The interpreter is
  decode-only; the launch-overhead argument doesn't apply at prefill
  shapes.
- **MTP integration changes.** The interpreter executes one main forward
  per call. MTP wraps the interpreter at the host level the same way it
  wraps the captured graph today. Tree-MTP, DFlash, or any other
  drafter improvements are orthogonal (see
  `2026-06-08-dflash-speculative-decoding-design.md`).
- **lm_head NVFP4 quantisation.** That is its own parity-sensitive
  experiment (Mirage doc Risk list); add it on top of the interpreter
  later if it ships.

## 9. References

- Hazy Research — *Look Ma, No Bubbles! Designing a Low-Latency
  Megakernel for Llama-1B*, 2025-05-27.
  <https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>
- AMD MI300X single-kernel inference engine — Kog AI, sentinel-based
  sync pattern reference.
  <https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/>
- Mirage Persistent Kernel — compiler oracle for what fusions are
  reachable on Qwen3-class models.
  <https://github.com/mirage-project/mirage>
- `AGENT.md` §"Per-block megakernel (Phase 2) — NEGATIVE result" — what
  not to repeat.
- `docs/mirage-megakernel.md` — CUTLASS-side history of the megakernel
  effort.
- `kernels-cuda/megakernel/full_attn_block_sm120.cu` — the existing
  persistent + arrive/wait scaffolding to lift into the interpreter.
