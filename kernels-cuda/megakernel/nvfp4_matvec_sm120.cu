// Mirage-style megakernel for Qwen3.6 NVFP4 decode.
//
// Phase 1: a CUTLASS-templated NVFP4 GEMM (M, 1, K) for the hot decode
// shapes, replacing cuBLASLt's heuristic-picked kernel which loses ~1.78×
// at batch=1 on Blackwell SM120 vs a hand-tuned PingPong 64×128 schedule.
//
// This file is intentionally minimal at first commit: it just verifies the
// CUTLASS include path and gives us a dispatchable extern "C" entry point
// that returns NOT_IMPLEMENTED until the kernel proper lands.
#include "qwen36_fp4.h"

#include <cuda_runtime.h>
// CUTLASS sanity include — if the build fails here the include flags are
// not yet wired correctly. We only need the version macro to verify the
// header is reachable; the actual kernel templates land in the next step.
#include <cutlass/cutlass.h>
#include <cutlass/version.h>

extern "C" int qwen36_megakernel_nvfp4_gemm(
    const qwen36_nvfp4_gemm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  // CUTLASS_MAJOR is the explicit major-version macro from cutlass/version.h.
  // The Blackwell FP4 epilogues we need only exist in CUTLASS 4.x.
  static_assert(CUTLASS_MAJOR >= 4,
                "Mirage megakernel requires CUTLASS 4.x (Blackwell FP4 path)");
  // Placeholder: the actual templated kernel comes in the next commit.
  // Falling back lets the engine call this stub for parity testing without
  // breaking the existing cuBLASLt path.
  return QWEN36_STATUS_NOT_IMPLEMENTED;
}
