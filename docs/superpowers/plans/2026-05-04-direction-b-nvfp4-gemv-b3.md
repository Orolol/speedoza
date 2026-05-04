# Direction B Phase B3 — Tensor-core MMA gemv

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Replace the B2 scalar-dequant gemv body with a Blackwell tensor-core implementation built on the FP4 block-scaled MMA atom (`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`), keeping op-level parity with the cuBLASLt baseline.

**Predecessor:** Phase B2 (commit `0e81174`) — hand-rolled scalar kernel, smoke-validated, 1 warp per output row, 8 rows per CTA. The dispatch surface (`QWEN36_DECODE_GEMV=1`, `qwen36_decode_nvfp4_gemv` ABI) is already wired and stays unchanged through B3.

**Why this is a separate plan:** The FP4 MMA register layouts (per-thread A/B/C/SF mappings for the m16n8k64 atom) are non-trivial and there's no in-tree example to crib from. A single end-to-end B3 plan would be too long to land safely in one session. The plan is decomposed into four mergeable sub-phases (B3.1–B3.4) so each can be parity-gated independently and reverted in isolation.

---

## What B3 inherits from B2

- `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` — entry point, ABI handshake, shape-check guards.
- The `nvfp4_gemv_kernel<<<grid, block>>>` launch shape (8 rows per CTA, 32 lanes per warp) is the **starting point** but B3.1 redesigns the per-CTA tiling to match the m16n8 MMA atom.
- The smoke test (`kernels-cuda/smoke.cu`) at `M=K=128, N=1` stays as the floor parity gate.
- The scale-layout helpers (`gemv_vec16_scale_offset`, `decode_e4m3_local`, `kGemvFp4Lut`) stay; `kGemvFp4Lut` becomes a fallback for any unsupported edge case while the MMA path handles the hot loop.

## What B3 produces

- A working FP4 tensor-core gemv kernel for the same supported regime as B2 (`n=1, m%16==0, k%16==0`).
- An op-level parity gate against B2's scalar kernel **and** against cuBLASLt, both at smoke-shape (`M=K=128`) and at production decode shapes (TBD via `decode_parity.py` — out of scope for B3.1, lands in B3.2).
- Updated `docs/superpowers/notes/` with the actual measured numbers vs. cuBLASLt at the supported shapes. (No bench numbers in B3.1 itself; user has the right to gate that.)

## Out of scope for the entire B3 plan

- `lm_head` (BF16 weight today; spec §14 marks this as a separate quality decision).
- Prefill path optimisation (spec §14).
- MTP head migration (spec §14, deferred to B5+ in the original numbering).

---

## Sub-phase decomposition

Each sub-phase is a separate PR / merge candidate.

### B3.1 — Single-CTA FP4 MMA, no warp specialization

**Scope:** Replace the scalar inner loop with calls to `cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t, 16>::fma(...)`. Single warp drives the MMA; no producer/consumer split, no persistent grid, no TMA.

**Tile geometry:** 1 CTA owns BLOCK_M output rows. Each warp handles 1 m16 stripe → 16 rows. K is consumed in chunks of 64. Per chunk, per warp: 1 MMA call. The m16n8 atom computes 16 rows × 8 columns; we use column 0 only and discard 7/8 of the FP4 MMA throughput on the unused N columns. This matches the spec §5 "pad N to 8, mask to 1" contract.

**Open design questions** (to resolve at the start of B3.1):
1. **Register packing for A/B/SF.** The MMA expects A in `uint32_t[4]` and B in `uint32_t[2]` per lane. We need to map the gmem-layout FP4 weight bytes (row-major, 8 packed FP4 per byte) to that register layout. Options: (a) use `ldmatrix.x4.trans` to load shared-memory tiles directly into the MMA register layout; (b) load to shared, then bit-cast packed bytes per lane manually; (c) load to gmem-coalesced thread-local registers and shuffle into MMA layout. Option (a) is fastest but requires understanding `ldmatrix`'s thread-to-data mapping for the m16n8k64 row-col layout; (c) is simplest but burns shuffles. Pick (a) if the cute traits give us the right `Copy_Atom`; otherwise default to (b).
2. **SF register layout.** With VS=16 and `RegTypeSF=uint32`, each lane holds 1 uint32 = 4 packed e4m3 scales. The PTX expects SFA to cover 16 rows × 4 scale-groups = 64 scales total, distributed across 32 lanes = 2 scales per lane. But the signature is `RegTypeSF[1]` per lane (4 scales packed). Verify via the `MMA_Traits<...>::SLayoutA` accessor in `kernels-cuda/cutlass/include/cute/atom/mma_traits_sm120.hpp`. Mirror the layout exactly.
3. **Activation broadcast.** N=1 means all 8 N-columns of B carry the same activation column 0. Either load once and broadcast across the 8 cols (cleaner; matches the spec's "wasted N=1..7" framing) or zero-fill cols 1..7 (also correct since the output cols are masked at write time). Broadcast is preferable because the MMA's SFB layout assumes a per-N-group scale.
4. **Where to load activation.** With one CTA per BLOCK_M slice (no clusters yet in B3.1), each CTA redundantly loads the same activation vector from gmem. For K=5120, that's 2560 bytes × ~273 CTAs = ~700 KB of redundant gmem reads. L2 will absorb this on warm runs. Postpone the cluster + multicast optimisation to B3.4.

**Parity gate:** the existing smoke at `M=K=128` stays the floor. Add a second smoke probe at `M=128, K=256` to exercise the multi-K-chunk loop. Op-level parity (cosine similarity ≥ 0.998) against the B2 scalar kernel is implicit because both feed the same dispatch — toggling `QWEN36_DECODE_GEMV=1` vs unset is the A/B switch.

**Estimated effort:** 1–2 sessions of focused CUDA + debugging.

**Files touched:**
- `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` — replace inner loop, keep entry point
- `kernels-cuda/smoke.cu` — add `M=128, K=256` probe
- (No Rust-side changes; ABI is unchanged)

---

### B3.2 — Op-level parity sweep at production shapes

**Scope:** Run `decode_parity.py` (per the existing passthrough flow in `docs/superpowers/notes/2026-05-04-decode-parity-gemv-passthrough.md`) with `QWEN36_DECODE_GEMV=1` and validate cosine similarity ≥ 0.998 at every NVFP4 GEMM shape exercised by the decode hot path.

**No code changes** — this is a gate, not an implementation step. If any shape fails, write a minimal repro into `kernels-cuda/smoke.cu` and bisect. If a shape needs to fall back to cuBLASLt, widen the NOT_IMPLEMENTED guard in `qwen36_decode_nvfp4_gemv` rather than silently producing wrong output.

**Files touched:**
- Possibly `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` (if a shape needs to be excluded)
- `kernels-cuda/smoke.cu` (per-shape repros if any fail)
- `docs/superpowers/notes/2026-05-04-decode-gemv-shape-coverage.md` (new): record the supported / unsupported shapes after the sweep.

**Estimated effort:** 1 session, mostly waiting on the model loader.

---

### B3.3 — Persistent grid + warp specialization

**Scope:** Drop the "1 CTA per BLOCK_M" dispatch in favour of a persistent kernel. Launch ~192 CTAs (one per SM); each CTA loops over multiple BLOCK_M slices to amortise the launch and the activation broadcast.

Inside each CTA, split warps:
- Producer warps (2 warps, 64 threads): drive the next-tile loads.
- Consumer warps (4 warps, 128 threads): drive MMAs on the current tile.
- Synchronise via mbarriers (`bar.sync` or `cp.async.mbarrier.arrive`).
- Pipeline depth = 2 (double-buffered shared memory).

**Why this matters:** at decode-time the kernel launch overhead on WSL2 is ~3 µs (per `AGENT.md` 2026-05-04). For ~14 NVFP4 GEMMs per layer × 64 layers = ~900 launches per token, that's ~2.7 ms of pure launch overhead. A persistent kernel collapses this to a single launch per call.

**Estimated effort:** 1–2 sessions.

**Files touched:**
- `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` (heavy rewrite of the CTA structure)

---

### B3.4 — TMA multicast for activation broadcast

**Scope:** Use `cp.async.bulk.tensor.*.multicast` from cluster CTAs to fan a single activation TMA load to all CTAs in a cluster. Cluster size 4 (4 × 128 = 512 output rows per cluster).

**Why this matters at our scale:** activation vector is 2560 bytes for K=5120; with 192 SMs the redundant gmem loads are ~480 KB. L2 absorbs this on warm runs but the TMA path frees up gmem bandwidth that the weight loads need.

**Estimated effort:** 1 session.

**Files touched:**
- `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` (cluster launch + TMA multicast wiring)

---

## Decision points before starting B3.1

1. **Validate that `CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED` is actually set** in the build (gated by CUDA 12.8+ AND `__CUDA_ARCH__ == 1200`). A 5-line probe (`#ifdef CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED #pragma message "MMA_ENABLED" #endif`) added to `nvfp4_gemv_sm120.cu` confirms before touching the body.
2. **Pick the load strategy** per "Open design questions" #1 above.
3. **Decide whether to take the cute MMA atom as a black box** (call `SM120_16x8x64_TN_VS::fma(...)` directly with hand-packed register operands) **or** integrate via the cute `TiledMMA` infrastructure (more cute boilerplate but composes with `Copy_Atom<SM75_U32x4_LDSM_N>` for `ldmatrix`-based loads). The first is shorter and matches the existing kernel's flat structure; the second is the CUTLASS-idiomatic pattern.

The recommendation is: **call the atom directly + use raw `cp.async` for loads + hand-pack registers via `ldmatrix.x4.trans`** at the start. Migrate to cute infrastructure only if the hand-rolled version hits maintenance issues.

---

## Risk register

| Risk | Mitigation |
|---|---|
| MMA register layout off by one → wrong outputs | Op-level parity gate against B2 scalar kernel. Both run via the same dispatch; A/B by env var. Bisect by zeroing out the inner loop and checking the MMA register layout in isolation. |
| `mma.kind::mxf4nvf4` PTX emission requires `-arch=sm_120a` (not just `sm_120`) | The build script currently uses `sm_120`. If the MMA fails to assemble, switch to `sm_120a`. This may break SM_121 fallback compatibility — re-check `AGENT.md` constraints. |
| `ldmatrix` load doesn't match the m16n8k64 row-col layout for FP4 | Fall back to manual register packing via `__shfl_sync` / shared memory bit-cast. Slower but correct. |
| At N=1 the wasted 7/8 N-columns of the MMA negate any tensor-core win | Already known — accept the trade-off for B3.1; B3.3's persistent grid + B3.4's multicast amortise it. The structural win is decoupling math from memory bandwidth, which holds even at low N efficiency. |

---

## What stays unchanged through all of B3

- The C ABI (`qwen36_decode_nvfp4_gemv` signature in `kernels-cuda/include/qwen36_fp4.h`).
- The Rust dispatch (`crates/kernels/src/backend.rs`) and env-var gate (`QWEN36_DECODE_GEMV`).
- The B2 scalar fallback semantics: any shape outside the supported regime returns NOT_IMPLEMENTED, dispatcher falls back to cuBLASLt.
- The smoke harness contract: B1 unsupported-shape probe stays as the regression gate for the soft-fallback path.
