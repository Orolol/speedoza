# Direction B NVFP4 gemv — Phase B1 + B2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land Phase B1 (ABI + Rust FFI + dispatch wiring + stub kernel returning `NOT_IMPLEMENTED`) and Phase B2 (naive NVFP4 gemv kernel for one shape, op-level parity-gated against cuBLASLt) of the Direction B design at `docs/superpowers/specs/2026-05-04-direction-b-nvfp4-gemv-design.md`.

**Architecture:** Mirror the existing Mirage megakernel pattern (`kernels-cuda/megakernel/`, `qwen36_megakernel_nvfp4_gemm`). Add a new C entry point `qwen36_decode_nvfp4_gemv` reusing `qwen36_nvfp4_gemm_spec_t`, gated by `QWEN36_DECODE_GEMV=1`. B1 ships a stub that always returns `NOT_IMPLEMENTED` (5) so the dispatcher transparently falls through to the megakernel/cuBLASLt path. B2 fills in a single-shape CUTLASS-based kernel for `M%128==0, K%128==0, N==1`, with a CPU-reference smoke test and an op-level parity extension.

**Tech Stack:** CUDA 13.1 / SM_120, CUTLASS 4.x (NVFP4 BlockScaledTensorOp on Sm120), Rust 1.x w/ `cuda` feature, existing `qwen36_fp4_kernels` shared library, `scripts/build_cuda.sh`, `scripts/smoke_cuda.sh`, `scripts/decode_parity.py`.

**Out of scope (deferred to follow-up plans):**
- Phase B3 — Persistent grid + warp specialization + TMA double-buffering
- Phase B4 — TMA multicast / cluster tuning
- Phase B5 — Bench matrix on RTX 5090 (user has explicitly asked NOT to run perf benches in this session — gate this on a future plan)
- Phase B6 — Default-on switch for the shipped checkpoint
- Bench scripts, perf measurement, profile collection. Functional + parity work only here.

---

## File Structure

**New files (Phase B1):**
- `kernels-cuda/decode_gemv/nvfp4_gemv_stub.cu` — fallback compilation unit when CUTLASS is absent. Always returns `NOT_IMPLEMENTED`. ~10 lines.
- `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` — real entry point. In B1 this returns `NOT_IMPLEMENTED` for every shape. In B2 it implements the kernel for the supported shape regime.

**Modified files (Phase B1):**
- `kernels-cuda/include/qwen36_fp4.h` — add `int qwen36_decode_nvfp4_gemv(const qwen36_nvfp4_gemm_spec_t *spec);` declaration with doc comment.
- `crates/kernels/src/backend.rs` — mirror FFI extern, add `decode_gemv_enabled()` env-var helper, wire dispatch in `CudaBackend::nvfp4_gemm` before the existing megakernel branch.
- `scripts/build_cuda.sh` — compile `decode_gemv/nvfp4_gemv_sm120.cu` when CUTLASS is available, else `decode_gemv/nvfp4_gemv_stub.cu`.
- `kernels-cuda/smoke.cu` — add a small test calling `qwen36_decode_nvfp4_gemv` with a deliberately unsupported shape and asserting return code 5 (B1).

**Modified files (Phase B2):**
- `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` — replace the stub body with a working CUTLASS-based gemv for `M%128==0, K%128==0, N==1`.
- `kernels-cuda/smoke.cu` — extend the smoke test with planted-data correctness against a CPU reference for `M=128, K=128, N=1`.
- `scripts/decode_parity.py` — add a `QWEN36_PARITY_GEMV_LAYER` env-var path that forces the gemv kernel for one global layer index and emits cos sim vs. the cuBLASLt-only run.

Each task below is self-contained; tasks within a phase should be executed in order.

---

# Phase B1 — ABI + dispatch + stub kernel

### Task 1: Add C ABI declaration for `qwen36_decode_nvfp4_gemv`

**Files:**
- Modify: `kernels-cuda/include/qwen36_fp4.h` (insert immediately after the existing `qwen36_megakernel_nvfp4_gemm` declaration, around line 490)

- [ ] **Step 1: Edit the header**

In `kernels-cuda/include/qwen36_fp4.h`, find the block:

```c
int qwen36_megakernel_nvfp4_gemm(const qwen36_nvfp4_gemm_spec_t *spec);
int qwen36_bf16_gemm(const qwen36_bf16_gemm_spec_t *spec);
```

Insert between them:

```c
// Direction B decode-time NVFP4 gemv: hand-written kernel optimised for the
// (M, N=1, K) shapes that dominate decode. Reuses `qwen36_nvfp4_gemm_spec_t`.
// Returns QWEN36_STATUS_NOT_IMPLEMENTED (5) for shapes outside the supported
// set (M%128==0, K%128==0, N==1); the Rust dispatcher falls back to the
// existing megakernel/cuBLASLt path on that code, mirroring the Mirage
// pattern. Gated by `QWEN36_DECODE_GEMV=1`. See
// `docs/superpowers/specs/2026-05-04-direction-b-nvfp4-gemv-design.md`.
int qwen36_decode_nvfp4_gemv(const qwen36_nvfp4_gemm_spec_t *spec);
```

- [ ] **Step 2: Verify the header still compiles standalone**

Run:

```bash
cc -fsyntax-only -x c -I kernels-cuda/include kernels-cuda/include/qwen36_fp4.h
```

Expected: no output (success). If you see a syntax error, fix the inserted text.

- [ ] **Step 3: Commit**

```bash
git add kernels-cuda/include/qwen36_fp4.h
git commit -m "feat(abi): declare qwen36_decode_nvfp4_gemv entry point"
```

---

### Task 2: Add CUTLASS-less stub compilation unit

**Files:**
- Create: `kernels-cuda/decode_gemv/nvfp4_gemv_stub.cu`

- [ ] **Step 1: Create the directory and stub file**

```bash
mkdir -p kernels-cuda/decode_gemv
```

Write `kernels-cuda/decode_gemv/nvfp4_gemv_stub.cu`:

```c
#include "qwen36_fp4.h"

extern "C" int qwen36_decode_nvfp4_gemv(
    const qwen36_nvfp4_gemm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  return QWEN36_STATUS_NOT_IMPLEMENTED;
}
```

- [ ] **Step 2: Commit**

```bash
git add kernels-cuda/decode_gemv/nvfp4_gemv_stub.cu
git commit -m "feat(kernels): add decode_gemv stub returning NOT_IMPLEMENTED"
```

---

### Task 3: Add SM120 entry-point file (still NOT_IMPLEMENTED in B1)

**Files:**
- Create: `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu`

- [ ] **Step 1: Write the file**

This file mirrors `kernels-cuda/megakernel/nvfp4_matvec_sm120.cu` structurally but contains no CUTLASS adapter yet. B2 fills in the body. Until then it must compile against CUTLASS headers and return `NOT_IMPLEMENTED` for every shape so the dispatch pipeline can be validated end-to-end.

Write `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu`:

```c
// Direction B NVFP4 gemv kernel for Blackwell SM_120.
//
// Phase B1: declares the entry point and validates the dispatch wiring;
// every shape returns QWEN36_STATUS_NOT_IMPLEMENTED so the Rust dispatcher
// falls back to the cuBLASLt path. Phase B2 fills in a CUTLASS-based gemv
// for the (M%128==0, K%128==0, N==1) regime.
//
// See `docs/superpowers/specs/2026-05-04-direction-b-nvfp4-gemv-design.md`.
#include "qwen36_fp4.h"

#include <cstdint>

extern "C" int qwen36_decode_nvfp4_gemv(
    const qwen36_nvfp4_gemm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->m == 0 || spec->n == 0 || spec->k == 0 ||
      spec->a_fp4.ptr == 0 || spec->a_scale.ptr == 0 ||
      spec->b_fp4.ptr == 0 || spec->b_scale.ptr == 0 ||
      spec->c_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  // Phase B1: kernel body not yet implemented; soft-fall back so the
  // Rust dispatcher routes to the existing megakernel/cuBLASLt path.
  return QWEN36_STATUS_NOT_IMPLEMENTED;
}
```

- [ ] **Step 2: Commit**

```bash
git add kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu
git commit -m "feat(kernels): scaffold decode_gemv SM120 entry point (B1 stub body)"
```

---

### Task 4: Wire `decode_gemv` source into the build script

**Files:**
- Modify: `scripts/build_cuda.sh`

- [ ] **Step 1: Edit the build script**

In `scripts/build_cuda.sh`, find:

```bash
if [ -d "${CUTLASS_DIR}/include" ]; then
  CUTLASS_FLAGS+=(
    -I "${CUTLASS_DIR}/include"
    -I "${CUTLASS_DIR}/tools/util/include"
    --expt-relaxed-constexpr
    --extended-lambda
  )
  EXTRA_SRC=(kernels-cuda/megakernel/nvfp4_matvec_sm120.cu)
else
  echo "warn: ${CUTLASS_DIR} not found; building without Mirage megakernel" >&2
  EXTRA_SRC=(kernels-cuda/megakernel/nvfp4_matvec_stub.cu)
fi
```

Replace with:

```bash
if [ -d "${CUTLASS_DIR}/include" ]; then
  CUTLASS_FLAGS+=(
    -I "${CUTLASS_DIR}/include"
    -I "${CUTLASS_DIR}/tools/util/include"
    --expt-relaxed-constexpr
    --extended-lambda
  )
  EXTRA_SRC=(
    kernels-cuda/megakernel/nvfp4_matvec_sm120.cu
    kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu
  )
else
  echo "warn: ${CUTLASS_DIR} not found; building without Mirage megakernel and decode_gemv" >&2
  EXTRA_SRC=(
    kernels-cuda/megakernel/nvfp4_matvec_stub.cu
    kernels-cuda/decode_gemv/nvfp4_gemv_stub.cu
  )
fi
```

- [ ] **Step 2: Build the shared library**

Run:

```bash
./scripts/build_cuda.sh
```

Expected: prints `target/cuda/libqwen36_fp4_kernels.so`. If a CUTLASS-related error appears, it likely indicates a header path issue — confirm `kernels-cuda/cutlass/include` exists.

- [ ] **Step 3: Confirm the new symbol is exported**

Run:

```bash
nm -D target/cuda/libqwen36_fp4_kernels.so | grep qwen36_decode_nvfp4_gemv
```

Expected: a single line ending with ` T qwen36_decode_nvfp4_gemv`. If empty, the source file did not get linked — check `EXTRA_SRC`.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_cuda.sh
git commit -m "build(cuda): compile decode_gemv source alongside megakernel"
```

---

### Task 5: Add Rust FFI extern + env-var gate + dispatch wiring

**Files:**
- Modify: `crates/kernels/src/backend.rs`

This is three coordinated edits in the same file: the FFI block, a new env-var helper, and the dispatch site.

- [ ] **Step 1: Add the FFI extern declaration**

In `crates/kernels/src/backend.rs`, find the block (around line 1308):

```rust
        pub fn qwen36_megakernel_nvfp4_gemm(spec: *const Nvfp4GemmSpec) -> i32;
```

Insert immediately after it:

```rust
        pub fn qwen36_decode_nvfp4_gemm(spec: *const Nvfp4GemmSpec) -> i32; // forward
        pub fn qwen36_decode_nvfp4_gemv(spec: *const Nvfp4GemmSpec) -> i32;
```

Then **delete** the `qwen36_decode_nvfp4_gemm` line — it was a typo guard against confusing the new symbol with `_gemm`. The final inserted line is just:

```rust
        pub fn qwen36_decode_nvfp4_gemv(spec: *const Nvfp4GemmSpec) -> i32;
```

(In other words: only `qwen36_decode_nvfp4_gemv` should remain after this step.)

- [ ] **Step 2: Add `decode_gemv_enabled()` helper**

In the same file, find `fn megakernel_enabled()` (around line 422). Below its closing `}`, insert:

```rust
/// Cached env-var lookup gating the Direction B decode-time NVFP4 gemv path.
/// Set `QWEN36_DECODE_GEMV=1` to opt in; the default (unset / 0) keeps the
/// existing megakernel/cuBLASLt route active. Cached so the dispatch hot
/// path does not re-parse the environment per GEMM call.
#[cfg(feature = "cuda")]
fn decode_gemv_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN36_DECODE_GEMV").ok().as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    })
}
```

- [ ] **Step 3: Wire dispatch in `nvfp4_gemm`**

Find the existing dispatch body in `CudaBackend::nvfp4_gemm` (around line 174-194). Replace it with:

```rust
    fn nvfp4_gemm(&self, spec: &Nvfp4GemmSpec) -> Result<()> {
        let ffi_spec = ffi::Nvfp4GemmSpec::from(spec);
        // Direction B decode-time gemv path. When enabled and the GEMM is
        // gemv-shaped (n == 1) we try the hand-written kernel first; on
        // QWEN36_STATUS_NOT_IMPLEMENTED we fall through to the existing
        // megakernel / cuBLASLt routing. See the Direction B spec under
        // `docs/superpowers/specs/2026-05-04-direction-b-nvfp4-gemv-design.md`.
        if decode_gemv_enabled() && spec.n == 1 {
            let code = unsafe { ffi::qwen36_decode_nvfp4_gemv(&ffi_spec) };
            if code != 5 {
                return check("qwen36_decode_nvfp4_gemv", code);
            }
        }
        // When the Mirage megakernel path is enabled (env var) we try the
        // CUTLASS-templated NVFP4 GEMM next. The kernel is being built up
        // shape-by-shape (`docs/mirage-megakernel.md`); on any unsupported
        // shape it returns QWEN36_STATUS_NOT_IMPLEMENTED, in which case we
        // transparently fall back to the cuBLASLt path. This keeps the
        // engine perf-neutral while individual shapes get migrated.
        if megakernel_enabled() {
            let code = unsafe { ffi::qwen36_megakernel_nvfp4_gemm(&ffi_spec) };
            // 5 == QWEN36_STATUS_NOT_IMPLEMENTED. Any other non-zero is a
            // real failure and surfaces through `check()` like the rest of
            // the FFI surface.
            if code != 5 {
                return check("qwen36_megakernel_nvfp4_gemm", code);
            }
        }
        check("qwen36_nvfp4_gemm", unsafe {
            ffi::qwen36_nvfp4_gemm(&ffi_spec)
        })
    }
```

- [ ] **Step 4: Verify the crate builds (no-CUDA feature)**

Run:

```bash
cargo build -p qwen36-fp4-kernels
```

Expected: success. The `decode_gemv_enabled` and FFI declarations are gated on `feature = "cuda"`, so a no-CUDA build still works.

- [ ] **Step 5: Verify the crate builds (CUDA feature)**

Run:

```bash
export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$QWEN36_FP4_KERNEL_LIB_DIR:${LD_LIBRARY_PATH:-}"
cargo build --workspace --features qwen36-fp4-kernels/cuda
```

Expected: success. If the linker complains about an undefined `qwen36_decode_nvfp4_gemv`, it means Task 4's library build was skipped — re-run `./scripts/build_cuda.sh`.

- [ ] **Step 6: Run clippy with the cuda feature**

Run:

```bash
cargo clippy --workspace --features qwen36-fp4-kernels/cuda -- -D warnings
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add crates/kernels/src/backend.rs
git commit -m "feat(kernels): wire decode_gemv soft-fallback dispatch in CudaBackend"
```

---

### Task 6: Add a smoke test that exercises the dispatch surface (B1)

**Files:**
- Modify: `kernels-cuda/smoke.cu`

The B1 smoke test does not need real planted weights — it only proves the symbol is wired and returns `NOT_IMPLEMENTED` cleanly so the Rust fallback is exercised.

- [ ] **Step 1: Add the smoke section**

Open `kernels-cuda/smoke.cu`. Find the existing `qwen36_nvfp4_gemm` test block (search for `qwen36_nvfp4_gemm_spec_t gemm_spec{}`) — it allocates real planted weights and validates GEMM output. Below the `expect_close(gemm_values[gemm_m - 1], 132.0f, 4.0f, "nvfp4 gemm[last]")` line (around line 599), insert:

```c
  // Direction B Phase B1 smoke: the decode_gemv entry point must exist,
  // accept a well-formed spec, and return NOT_IMPLEMENTED (5) so the Rust
  // dispatcher falls back. When the kernel body lands (Phase B2) the test
  // for the supported shape regime moves below; this case stays as the
  // unsupported-shape probe.
  qwen36_nvfp4_gemm_spec_t gemv_b1_spec = gemm_spec;
  gemv_b1_spec.m = gemm_m + 1;  // deliberately not multiple of 128
  gemv_b1_spec.n = 1;
  int gemv_b1_code = qwen36_decode_nvfp4_gemv(&gemv_b1_spec);
  if (gemv_b1_code != QWEN36_STATUS_NOT_IMPLEMENTED) {
    fprintf(stderr,
            "decode_gemv B1 expected NOT_IMPLEMENTED (5) for unsupported "
            "shape, got %d\n",
            gemv_b1_code);
    return 1;
  }
```

- [ ] **Step 2: Rebuild and run smoke**

Run:

```bash
./scripts/build_cuda.sh
./scripts/smoke_cuda.sh
```

Expected: smoke binary exits 0 (no `decode_gemv B1 expected ...` line on stderr). If the CUDA tests can't be run because the GPU is busy with the user's other workload, mark this step as deferred — the dispatch wiring is verified by Task 5 step 5.

- [ ] **Step 3: Commit**

```bash
git add kernels-cuda/smoke.cu
git commit -m "test(cuda): smoke-test decode_gemv NOT_IMPLEMENTED fallback path"
```

---

### Task 7: Add Rust integration test for the env-var gate

**Files:**
- Modify or create as appropriate: an integration test for `qwen36-fp4-kernels`.

Investigate first: run `ls crates/kernels/tests/ 2>/dev/null` and `grep -rn "megakernel_enabled\|QWEN36_USE_MEGAKERNEL_GEMM" crates/kernels/` to find any existing tests for `megakernel_enabled`. If none exist, this task is a no-op — note that and skip to Task 8. The env-var gate is private; a runtime smoke (Task 6) plus the dispatch-site unit tests already exercise the surface.

- [ ] **Step 1: Inspect the existing test surface**

Run:

```bash
ls crates/kernels/tests/ 2>/dev/null
grep -rn "megakernel_enabled\|QWEN36_USE_MEGAKERNEL_GEMM" crates/kernels/
```

If no existing parallel test for `megakernel_enabled` exists, **mark this task as no-op and move on**. Document in the commit log:

```bash
git commit --allow-empty -m "chore(kernels): no Rust unit test for decode_gemv_enabled (parity with megakernel_enabled)"
```

If a test exists for `megakernel_enabled`, mirror its structure for `decode_gemv_enabled`.

---

# Phase B2 — Naive single-shape kernel + parity gate

The supported shape regime in B2 is `M % 128 == 0`, `K % 128 == 0`, `N == 1`. The first concrete decode shape this exercises is `down_proj` at `(M=5120, N=1, K=17408)`. Bring the kernel up against the smallest viable case (`M=128, K=128, N=1`) first, then verify it scales to the real decode shapes via the parity harness.

### Task 8: Implement the CUTLASS NVFP4 gemv body

**Files:**
- Modify: `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` (replace the stub body produced in Task 3)

**Approach.** The simplest correct implementation is a near-clone of `kernels-cuda/megakernel/nvfp4_matvec_sm120.cu` with a tile shape skewed for `N=1` decode: `ThreadBlockShape = Shape<_128, _8, _128>` (the smallest N supported by the FP4 MMA atom is 8; we accept the padding cost). Cluster shape stays `<1,1,1>` in B2 — TMA multicast and clusters ship in B4. This deliberately gives up the headline Marlin-style speedup; B3 / B4 / B5 reclaim it. B2's job is *correctness* and *ABI parity*, not performance.

- [ ] **Step 1: Replace the file body**

Overwrite `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` with:

```c
// Direction B NVFP4 gemv kernel for Blackwell SM_120 — Phase B2.
//
// Naive CUTLASS-based implementation specialised for the gemv shape
// (M%128==0, K%128==0, N==1). Mirrors the layout/typing of the existing
// Mirage megakernel (`kernels-cuda/megakernel/nvfp4_matvec_sm120.cu`)
// but uses the smallest valid N tile (8) instead of 128 to avoid wasting
// SM occupancy on padding columns at decode-time N=1. Cluster shape
// stays <1,1,1>; persistent grid + TMA multicast land in Phase B3/B4.
//
// On any unsupported shape we return QWEN36_STATUS_NOT_IMPLEMENTED (5)
// and the Rust dispatcher routes back to the cuBLASLt path. Active env
// var: `QWEN36_DECODE_GEMV=1`.
//
// See `docs/superpowers/specs/2026-05-04-direction-b-nvfp4-gemv-design.md`.
#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_runtime.h>
#include <cstdint>

#include <cutlass/cutlass.h>
#include <cutlass/version.h>

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) ||                                \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#endif

namespace {

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) ||                                \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;    // weight matrix [M, K]
using LayoutBTag = cutlass::layout::ColumnMajor; // activation [K, N]
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;

using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Phase B2 tile shape: 128×8×128. N=8 is the smallest supported by the
// SM120 FP4 MMA atom (m16n8k64); we mask the unused N=1..7 in the
// epilogue. This minimises padding waste vs. the megakernel's 128×128×128
// (which is ~16× over-allocated on N at N=1). Persistent grid + warp
// specialisation land in Phase B3.
using ThreadBlockShape = cute::Shape<cute::_128, cute::_8, cute::_128>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
        LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
        LayoutBTag, AlignmentB, ElementAccumulator, ThreadBlockShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

template <typename T> T *as_device_ptr(qwen36_device_ptr_t p) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(p.ptr));
}

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED || CUTLASS_ARCH_MMA_SM121_SUPPORTED

} // namespace

extern "C" int qwen36_decode_nvfp4_gemv(
    const qwen36_nvfp4_gemm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  static_assert(CUTLASS_MAJOR >= 4,
                "decode_gemv requires CUTLASS 4.x (Blackwell FP4 path)");

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) ||                                \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (spec->m == 0 || spec->n == 0 || spec->k == 0 ||
      spec->a_fp4.ptr == 0 || spec->a_scale.ptr == 0 ||
      spec->b_fp4.ptr == 0 || spec->b_scale.ptr == 0 ||
      spec->c_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  // Supported regime: gemv-shaped, both M and K aligned to the tile.
  if (spec->n != 1 || (spec->m % 128) != 0 || (spec->k % 128) != 0) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  using namespace cute;

  const int M = static_cast<int>(spec->m);
  const int N = static_cast<int>(spec->n);
  const int K = static_cast<int>(spec->k);

  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          as_device_ptr<const typename ElementA::DataType>(spec->a_fp4),
          stride_A,
          as_device_ptr<const typename ElementB::DataType>(spec->b_fp4),
          stride_B,
          as_device_ptr<const typename ElementA::ScaleFactorType>(
              spec->a_scale),
          layout_SFA,
          as_device_ptr<const typename ElementB::ScaleFactorType>(
              spec->b_scale),
          layout_SFB,
      },
      {
          {spec->alpha, 0.0f},
          as_device_ptr<ElementD>(spec->c_bf16),
          stride_C,
          as_device_ptr<ElementD>(spec->c_bf16),
          stride_D,
      },
  };

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  size_t workspace_bytes = Gemm::get_workspace_size(arguments);
  void *workspace_ptr = nullptr;
  bool owns_workspace = false;
  if (workspace_bytes > 0) {
    if (spec->workspace.ptr != 0 && spec->workspace_bytes >= workspace_bytes) {
      workspace_ptr =
          reinterpret_cast<void *>(static_cast<uintptr_t>(spec->workspace.ptr));
    } else {
      cudaError_t alloc_err = cudaMalloc(&workspace_ptr, workspace_bytes);
      if (alloc_err != cudaSuccess) {
        return QWEN36_STATUS_CUDA_ERROR;
      }
      owns_workspace = true;
    }
  }

  cudaStream_t stream = qwen36_internal_active_stream();
  auto status = gemm.initialize(arguments, workspace_ptr, stream);
  if (status == cutlass::Status::kSuccess) {
    status = gemm.run(stream);
  }

  if (owns_workspace) {
    cudaFree(workspace_ptr);
  }

  if (status != cutlass::Status::kSuccess) {
    return QWEN36_STATUS_CUDA_ERROR;
  }
  return QWEN36_STATUS_SUCCESS;
#else
  (void)spec;
  return QWEN36_STATUS_NOT_IMPLEMENTED;
#endif
}
```

- [ ] **Step 2: Rebuild and verify the symbol still exports**

Run:

```bash
./scripts/build_cuda.sh
nm -D target/cuda/libqwen36_fp4_kernels.so | grep qwen36_decode_nvfp4_gemv
```

Expected: still resolves to a `T` symbol. If `nvcc` errors on a CUTLASS template, copy the diagnostic into the commit message and bail to investigation — common cause is a stale `cutlass/` submodule.

- [ ] **Step 3: Commit**

```bash
git add kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu
git commit -m "feat(kernels): naive NVFP4 gemv for SM120 (M%128, K%128, N=1)"
```

---

### Task 9: Extend the smoke test with a planted-data correctness probe

**Files:**
- Modify: `kernels-cuda/smoke.cu`

The B1 smoke probed `NOT_IMPLEMENTED`. For B2, plant a small `(M=128, K=128, N=1)` problem with known weights and expect the kernel to produce the same result the existing `qwen36_nvfp4_gemm` test produces for the same inputs.

- [ ] **Step 1: Inspect the existing nvfp4_gemm planted-data test**

Read `kernels-cuda/smoke.cu` lines 583–600 (the existing `gemm_spec` block). Note `gemm_m`, the planted-tensor allocation, and the expected output value(s).

- [ ] **Step 2: Reuse the existing planted tensors in a gemv probe**

In `kernels-cuda/smoke.cu`, locate the B1 gemv probe added in Task 6 (search for `gemv_b1_spec`). Immediately above that block, insert a planted-data B2 probe that reuses the existing `gemm_spec` tensors but constrains `n=1` and verifies the first element of the output matches the same expected value:

```c
  // Direction B Phase B2 smoke: gemv-shaped (n=1) call against the same
  // planted-weight tensors used by the qwen36_nvfp4_gemm probe above.
  // The expected per-row output is the same value (132.0) since the
  // activation column is identical to the n=0 column of the GEMM input.
  qwen36_nvfp4_gemm_spec_t gemv_b2_spec = gemm_spec;
  gemv_b2_spec.n = 1;
  must_status(qwen36_decode_nvfp4_gemv(&gemv_b2_spec), "decode_gemv b2");
  // The planted output tensor was overwritten by the gemv call; re-read
  // and validate the first / last rows.
  expect_close(read_bf16(gemm_spec.c_bf16.ptr, 1)[0], 132.0f, 4.0f,
               "decode_gemv b2[0]");
  expect_close(read_bf16(gemm_spec.c_bf16.ptr, gemm_m)[gemm_m - 1], 132.0f,
               4.0f, "decode_gemv b2[last]");
```

If `read_bf16` does not exist with that signature, follow the pattern used by the existing `nvfp4 gemm` test exactly — copy the surrounding allocation/read helpers verbatim. Do not invent helper signatures.

- [ ] **Step 3: Rebuild and run smoke**

Run:

```bash
./scripts/build_cuda.sh
./scripts/smoke_cuda.sh
```

Expected: smoke binary exits 0 with no `decode_gemv b2` failure messages on stderr.

If the user's GPU is busy, mark this step deferred but **do not** mark the task complete — the parity gate in Task 11 also requires GPU access, and we should batch the validation when the GPU frees up. Note this in the task comment.

- [ ] **Step 4: Commit**

```bash
git add kernels-cuda/smoke.cu
git commit -m "test(cuda): planted-data smoke for decode_gemv at M=K=128"
```

---

### Task 10: Add `QWEN36_PARITY_GEMV_LAYER` env-var path to the parity harness

**Files:**
- Modify: `scripts/decode_parity.py`

This is the op-level parity gate that validates the gemv kernel against cuBLASLt across the **real decode shapes** layer by layer. The harness already supports per-layer dumps via `QWEN36_DEBUG_DUMP_DIR`; we add an env var that tells the engine to force the gemv path for one specific global layer index.

**Investigate first.** Read `scripts/decode_parity.py` to understand its current API. The exact insertion points depend on the script's structure, which has not been pre-verified for this plan.

- [ ] **Step 1: Inspect the harness**

```bash
grep -n "QWEN36_DEBUG_DUMP_DIR\|QWEN36_USE_MEGAKERNEL_GEMM\|env" scripts/decode_parity.py | head -40
```

Identify:
1. Where the script sets engine env vars (look for an `os.environ` or a subprocess `env=` argument).
2. Where it dumps and diffs per-layer outputs (look for cosine-similarity or `cos_sim`).
3. Whether the existing megakernel parity path uses a similar per-layer toggle.

If the harness has no per-layer dump hook for GEMM outputs, this task expands into "add the dump hook *and* the env-var path" — a new sub-plan should be drafted before proceeding. Stop here, write the discovery up in `docs/superpowers/notes/2026-05-04-decode-parity-shape.md`, and ask the user how to proceed.

- [ ] **Step 2: If the harness already supports per-layer GEMM dumps, add the env-var pass-through**

In the engine-env-setup block of `scripts/decode_parity.py`, add a CLI flag or env-var passthrough that exports `QWEN36_DECODE_GEMV=1` together with a new `QWEN36_DECODE_GEMV_LAYER=<idx>` (mentioned in the spec section 9). The Rust side does not yet read `_LAYER`; for B2 we treat that env var as documented-but-unused, and apply the gemv globally (every NVFP4 GEMM, gemv-shaped).

- [ ] **Step 3: Run the parity check (only if GPU is free — user has explicitly asked NOT to bench)**

If GPU is available:

```bash
QWEN36_DECODE_GEMV=1 python scripts/decode_parity.py --prompt "hello" --max-new-tokens 4
```

Expected: every per-layer cosine similarity ≥ 0.998 against a baseline run with `QWEN36_DECODE_GEMV=0`. If any layer drops below, capture the layer index and shape into `docs/superpowers/notes/2026-05-04-gemv-parity.md` and **do not** mark Task 11 (commit) complete.

If GPU is NOT free, document the deferral in a TaskUpdate comment on Task 12 and skip Step 3.

- [ ] **Step 4: Commit**

```bash
git add scripts/decode_parity.py
git commit -m "test(parity): add QWEN36_DECODE_GEMV passthrough to decode_parity harness"
```

---

### Task 11: Documentation handoff

**Files:**
- Modify: `AGENT.md` (only if Task 10 step 3 confirmed parity)

- [ ] **Step 1: Update the optimization-status section**

If parity passed in Task 10, add a 1-paragraph note to `AGENT.md`'s "Current optimization status" section recording:

- The new env var `QWEN36_DECODE_GEMV=1`.
- Which decode shapes are supported (M%128==0, K%128==0, N=1).
- That benchmark numbers will be collected in a follow-up plan once the GPU is free.

If parity has NOT been validated yet (because GPU was busy), **do not** update AGENT.md — leave it for a follow-up so we don't claim a feature is shippable that has not been verified.

- [ ] **Step 2: Commit**

```bash
git add AGENT.md
git commit -m "docs(agent): note Direction B QWEN36_DECODE_GEMV opt-in (B2 parity passed)"
```

If skipped due to deferred parity, omit this commit entirely.

---

# Self-review checklist (run before handing off)

1. **Spec coverage.** Walk through spec sections 6.1–6.5 and 7.1: every modified file in those sections appears in the File Structure block above. Sections 6.4 (smoke) and 6.5 (parity harness) are Tasks 6/9 and 10. Section 7.2 ("no engine-level changes") is honored by routing entirely inside `CudaBackend::nvfp4_gemm`.
2. **Out-of-scope items** (B3–B6, perf benches) are explicitly listed at the top.
3. **Soft-fallback contract** (spec §5.6): both the C-side `NOT_IMPLEMENTED` returns (Task 8 step 1) and the Rust-side fallthrough (Task 5 step 3) are wired.
4. **ABI rule** (AGENT.md): header + Rust FFI changes ship in a single phase (Tasks 1, 5).
5. **No placeholder code:** every code block is concrete. Where investigation is required (Task 7, Task 10), the task explicitly says "investigate first" and routes to an alternative path if the assumed scaffolding is missing.
6. **Type consistency:** `qwen36_decode_nvfp4_gemv` reuses `qwen36_nvfp4_gemm_spec_t` end-to-end. The Rust extern signature matches the C declaration. No new typed Rust spec is introduced — `Nvfp4GemmSpec` already covers it.

# Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-04-direction-b-nvfp4-gemv.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
