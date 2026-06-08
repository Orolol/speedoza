#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"
#include "../../decode_gemv/nvfp4_gemv_mma_kernel.cuh"

#include <cuda_bf16.h>
#include <stdint.h>

namespace qwen36_interpreter {

__device__ inline float unpack_gemv_alpha_f32(uint64_t raw) {
  return __uint_as_float(static_cast<uint32_t>(raw & 0xffffffffu));
}

__device__ inline void
exec_nvfp4_gemv(const qwen36_interpreter_instruction_t &insn,
                PageAllocator &pages) {
  const size_t m = static_cast<size_t>(insn.payload[0]);
  const size_t k = static_cast<size_t>(insn.payload[1]);
  const uint8_t *a_fp4 = payload_ptr<const uint8_t>(insn.payload[2]);
  const uint8_t *a_scale = payload_ptr<const uint8_t>(insn.payload[3]);
  const uint8_t *b_fp4 = payload_ptr<const uint8_t>(insn.payload[4]);
  const uint8_t *b_scale = payload_ptr<const uint8_t>(insn.payload[5]);
  __nv_bfloat16 *output = payload_ptr<__nv_bfloat16>(insn.payload[6]);
  const float alpha = unpack_gemv_alpha_f32(insn.payload[7]);

  constexpr size_t kAlign16 =
      16u * static_cast<size_t>(qwen36_gemv::kKPerMma);
  if (m == 0 || k == 0 || (m % qwen36_gemv::kRowsPerBlock) != 0 ||
      (k % kAlign16) != 0 || a_fp4 == nullptr || a_scale == nullptr ||
      b_fp4 == nullptr || b_scale == nullptr || output == nullptr) {
    return;
  }

  const size_t smem_needed = qwen36_gemv::nvfp4_gemv_mma_smem_bytes<16>(k);
  if (smem_needed > pages.total_bytes || pages.base == nullptr) {
    return;
  }

  const size_t tiles = qwen36_gemv::gemv_div_ceil(
      m, static_cast<size_t>(qwen36_gemv::kRowsPerBlock));
  for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    qwen36_gemv::nvfp4_gemv_mma_body_with_smem<16>(
        static_cast<unsigned>(tile), a_fp4, a_scale, b_fp4, b_scale, alpha,
        output, m, k, pages.base);
    __syncthreads();
  }
}

} // namespace qwen36_interpreter
