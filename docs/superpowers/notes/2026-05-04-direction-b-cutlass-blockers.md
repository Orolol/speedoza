# Direction B — CUTLASS SM120 BlockScaled blockers for N=1

**Date:** 2026-05-04
**Plan:** `docs/superpowers/plans/2026-05-04-direction-b-nvfp4-gemv.md`
**Status:** B2 kernel body soft-disabled until the schedule is reworked (see "Path forward" below).

## TL;DR

The Direction B Phase B2 plan called for a "naive CUTLASS-based gemv" mirroring the Mirage megakernel. Bringing it up against the smoke harness exposed three issues in roughly increasing severity:

1. **Latent megakernel bug.** Both `kernels-cuda/megakernel/nvfp4_matvec_sm120.cu` and our new `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` guard the SM120 path with `#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)`. Neither file includes `<cutlass/arch/config.h>` (the header that *defines* the macro), and `<cutlass/cutlass.h>` does not pull it in transitively. So the `#if` silently evaluates to false at preprocessing, the SM120 path is never compiled, and the function returns `NOT_IMPLEMENTED` for every shape. The megakernel has therefore been falling back to cuBLASLt every call since it landed; the env-var toggle (`QWEN36_USE_MEGAKERNEL_GEMM=1`) has been a no-op at runtime, masked by the cuBLASLt fallback producing identical outputs.
2. **CUTLASS rejects the spec's narrow N tile.** The Direction B spec §5.1 calls for `BLOCK_M=128, BLOCK_N=8, BLOCK_K=128`. When the SM120 path is actually compiled (after fixing #1), the SM120 BlockScaled cooperative `CollectiveBuilder` rejects `N=8` at *template instantiation* with two static asserts: `"Invalid tile shape N."` and `"EPI_TILE_N must divide CTA_N"`. The smallest tile N this scheduler will accept appears to be 128.
3. **TMA stride alignment fails at N=1 (the actual decode shape).** Even with the safer `<128, 128, 128>` baseline tile, calling the kernel at runtime with `N=1, M=128, K=128` (the smoke harness's planted-data shape) crashes inside the CUTLASS TMA descriptor builder: `Assertion '(gmem_prob_stride[1] & 0b1111) == 0' failed.` (in `cute/atom/copy_traits_sm90_tma.hpp:977`). The output tensor C/D has stride[1] = N elements = 1 BF16 element = 2 bytes, but TMA requires 16-byte alignment on that stride. So the cooperative TMA-warpspecialised schedule cannot serve `N=1` regardless of tile shape.

`can_implement()` on the SM120 BlockScaled GemmUniversalAdapter does *not* surface either condition. It returns `Status::kInvalid (=11)` for our shape — the "unspecified" sentinel, not a real diagnostic. Skipping the `can_implement` check and going straight to `gemm.run()` produces the TMA assertion above. So even a careful caller cannot detect the failure ahead of time via the public CUTLASS surface.

## What is committed

- `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` keeps the full CUTLASS scaffolding (templates, types, dispatch boilerplate) but the function body returns `NOT_IMPLEMENTED` unconditionally with an inline comment pointing here. The Rust dispatch (`crates/kernels/src/backend.rs`) and ABI (`kernels-cuda/include/qwen36_fp4.h`) are unchanged from B1.
- The B2 planted-data smoke probe was reverted: it asserted the kernel produced the same output as cuBLASLt at `n=1, m=128, k=128`, which is the very shape that triggers the TMA assertion. The B1 NOT_IMPLEMENTED probe still runs.
- The `<cutlass/arch/config.h>` include was kept in `nvfp4_gemv_sm120.cu` so any future kernel body actually goes through the SM120 codegen path.
- The same include fix has **not yet been applied** to `kernels-cuda/megakernel/nvfp4_matvec_sm120.cu`. Doing so without changing its tile (`<128, 8, 128>`) will fail to compile (issue #2). A separate decision is required: (a) apply the include and bump tile to `<128, 128, 128>` to match what cuBLASLt is doing today, accepting that the megakernel is then a duplicate of cuBLASLt; or (b) leave it alone for now and flag the file as dead code.

## Path forward

Three options for re-attempting B2; pick before writing the next plan:

1. **Switch CUTLASS schedule.** Instead of `KernelTmaWarpSpecializedCooperative`, try a non-TMA scheduler that supports N=1. `KernelMultistage` / `KernelCpAsyncWarpSpecialized` family on Hopper used to admit small N. Need to verify CUTLASS 4.x SM120 still ships these for the BlockScaled FP4 path.
2. **Pad the output buffer.** Allocate a temporary `(M, 8)` or `(M, 16)` BF16 buffer, run CUTLASS at N=padded, copy column 0 back to the user's `(M, 1)` output. Costs an extra alloc + copy per call — likely kills the perf gain we were after.
3. **Hand-rolled CUDA kernel** (the spec's actual long-term goal — Marlin-style persistent + warp-specialised). Skips CUTLASS entirely, lets us choose tile = `<BLOCK_M, 1, BLOCK_K>` directly. Highest engineering cost but no CUTLASS landmines.

Until one of these is decided + implemented, `QWEN36_DECODE_GEMV=1` is safe to set — it enables the dispatch path which always returns `NOT_IMPLEMENTED` and falls back to cuBLASLt. The env var is effectively a no-op for now.

## How this was found

`./scripts/smoke_cuda.sh` after committing B2 reported `decode_gemv b2 returned status 5`. The investigation chain was:

1. Added stderr diagnostics around the `#if SM120_SUPPORTED` block → discovered the function fell into the `#else` branch (issue #1).
2. Added `#include <cutlass/arch/config.h>` → SM120 path activated → 27 template-instantiation errors centred on `"Invalid tile shape N."` / `"EPI_TILE_N must divide CTA_N"` (issue #2).
3. Bumped tile to `<128, 128, 128>` → built clean → smoke ran → `can_implement` returned 11 (`kInvalid`).
4. Skipped `can_implement` and called `gemm.run()` directly → TMA descriptor assertion fired (issue #3).
