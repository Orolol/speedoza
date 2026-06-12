# Archived plans

Plans here describe approaches that were proposed but were either superseded by what was actually implemented, or kept as historical record after the corresponding feature shipped.

- `2026-05-04-direction-b-nvfp4-gemv.md` — Original Direction B Phase B1+B2 plan. B1 (ABI + dispatch + stub) shipped as written. B2 was redirected from "naive CUTLASS-based gemv" to "hand-rolled scalar" (Option C) after CUTLASS blockers were uncovered (see `docs/superpowers/notes/2026-05-04-direction-b-cutlass-blockers.md`). Eventually replaced entirely by the B3 hand-rolled MMA kernel.
- `2026-05-04-direction-b-nvfp4-gemv-b3.md` — Original Direction B Phase B3 plan, prescribed sub-phases B3.1 (single-CTA FP4 MMA), B3.2 (op-level parity sweep), B3.3 (persistent grid + warp specialization), B3.4 (TMA multicast). What actually shipped: B3.1 (MMA atom) → B3.3.0–B3.3.2 (smem activation cache, coalesced SF loads, smem weight staging, cp.async double-buffering) → B3.4–B3.7 (intra-CTA split-K + warp widening + adaptive 8/16-warp dispatch). Persistent grid and TMA multicast were investigated and skipped: persistent grid doesn't help when each gemv call is independent, and TMA multicast was projected to give <2% gain after setup overhead. Final state in `AGENT.md` "Direction B decode_gemv" section; all decisions in commit history `git log -- kernels-cuda/decode_gemv/`.

These files are kept for reference but should NOT be used as the source of truth for current behavior.
