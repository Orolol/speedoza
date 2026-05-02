#include "qwen36_fp4.h"

extern "C" int qwen36_megakernel_nvfp4_gemm(
    const qwen36_nvfp4_gemm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  return QWEN36_STATUS_NOT_IMPLEMENTED;
}
