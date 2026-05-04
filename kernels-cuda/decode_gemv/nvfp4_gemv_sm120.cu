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
