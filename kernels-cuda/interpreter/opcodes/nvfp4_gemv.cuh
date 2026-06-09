#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"
#include "../../decode_gemv/nvfp4_gemv_mma_kernel.cuh"
#include "rmsnorm_nvfp4_quant.cuh"

#include <cuda_bf16.h>
#include <stdint.h>

namespace qwen36_interpreter {

__device__ inline float unpack_gemv_alpha_f32(uint64_t raw) {
  return __uint_as_float(static_cast<uint32_t>(raw & 0xffffffffu));
}

__device__ inline float decode_e2m1_for_gemv(uint8_t code) {
  float value = 0.0f;
  switch (code & 0x07u) {
  case 1:
    value = 0.5f;
    break;
  case 2:
    value = 1.0f;
    break;
  case 3:
    value = 1.5f;
    break;
  case 4:
    value = 2.0f;
    break;
  case 5:
    value = 3.0f;
    break;
  case 6:
    value = 4.0f;
    break;
  case 7:
    value = 6.0f;
    break;
  default:
    break;
  }
  return (code & 0x08u) != 0 ? -value : value;
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

__device__ inline void
exec_nvfp4_gemv_chunk_accum(const qwen36_interpreter_instruction_t &insn,
                            PageAllocator &) {
  const size_t m = static_cast<size_t>(insn.payload[0]);
  const size_t full_k = static_cast<size_t>(insn.payload[1]);
  const size_t chunk_start = static_cast<size_t>(insn.payload[2]);
  const size_t chunk_k = static_cast<size_t>(insn.payload[3]);
  const uint8_t *a_fp4 = payload_ptr<const uint8_t>(insn.payload[4]);
  const uint8_t *a_scale = payload_ptr<const uint8_t>(insn.payload[5]);
  const uint8_t *b_fp4 = payload_ptr<const uint8_t>(insn.payload[6]);
  const uint8_t *b_scale = payload_ptr<const uint8_t>(insn.payload[7]);
  float *accum = payload_ptr<float>(insn.payload[8]);
  __nv_bfloat16 *output = payload_ptr<__nv_bfloat16>(insn.payload[9]);
  const float alpha = unpack_gemv_alpha_f32(insn.payload[10]);
  const uint32_t flags = static_cast<uint32_t>(insn.payload[11]);

  constexpr uint32_t kResetAccum = 1u;
  constexpr uint32_t kFinalizeOutput = 2u;
  constexpr unsigned kRowsPerCta = 16u;
  constexpr unsigned kLanesPerRow = 32u;
  const bool reset_accum = (flags & kResetAccum) != 0u;
  const bool finalize_output = (flags & kFinalizeOutput) != 0u;

  if (m == 0 || full_k == 0 || chunk_k == 0 || (full_k % 16) != 0 ||
      (chunk_start % 16) != 0 || (chunk_k % 16) != 0 ||
      (chunk_start + chunk_k) > full_k || a_fp4 == nullptr ||
      a_scale == nullptr || b_fp4 == nullptr || b_scale == nullptr ||
      accum == nullptr || (finalize_output && output == nullptr)) {
    return;
  }

  const unsigned warp_id = static_cast<unsigned>(threadIdx.x) >> 5;
  const unsigned lane = static_cast<unsigned>(threadIdx.x) & 31u;
  if (warp_id >= kRowsPerCta) {
    return;
  }

  const size_t packed_cols = full_k / 2;
  const size_t scale_cols = full_k / 16;
  const size_t scale_inner_dim = round_up_size(scale_cols, 4);
  const size_t first_group = chunk_start / 16;
  const size_t chunk_groups = chunk_k / 16;

  for (size_t row = static_cast<size_t>(blockIdx.x) * kRowsPerCta + warp_id;
       row < m; row += static_cast<size_t>(gridDim.x) * kRowsPerCta) {
    const uint8_t *row_a = a_fp4 + row * packed_cols;
    float sum = 0.0f;

    for (size_t local_group = lane; local_group < chunk_groups;
         local_group += kLanesPerRow) {
      const size_t group = first_group + local_group;
      const size_t col0 = group * 16;
      const size_t scale_off_a =
          vec16_scale_offset(group, row, scale_inner_dim);
      const size_t scale_off_b = vec16_scale_offset(group, 0, scale_inner_dim);
      const float scale = decode_e4m3(__ldg(a_scale + scale_off_a)) *
                          decode_e4m3(__ldg(b_scale + scale_off_b));
      const size_t byte_off = col0 / 2;
      float local = 0.0f;
#pragma unroll
      for (int packed = 0; packed < 8; ++packed) {
        const uint8_t a = __ldg(row_a + byte_off + packed);
        const uint8_t b = __ldg(b_fp4 + byte_off + packed);
        local += decode_e2m1_for_gemv(a & 0x0fu) *
                 decode_e2m1_for_gemv(b & 0x0fu);
        local += decode_e2m1_for_gemv((a >> 4) & 0x0fu) *
                 decode_e2m1_for_gemv((b >> 4) & 0x0fu);
      }
      sum += local * scale;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }

    if (lane == 0) {
      const float previous = reset_accum ? 0.0f : accum[row];
      const float total = previous + sum;
      accum[row] = total;
      if (finalize_output) {
        output[row] = __float2bfloat16(total * alpha);
      }
    }
  }
}

__device__ inline void
exec_nvfp4_gemv_pair(const qwen36_interpreter_instruction_t &insn,
                     PageAllocator &pages) {
  const size_t m = static_cast<size_t>(insn.payload[0]);
  const size_t k = static_cast<size_t>(insn.payload[1]);
  const uint8_t *gate_a_fp4 = payload_ptr<const uint8_t>(insn.payload[2]);
  const uint8_t *gate_a_scale = payload_ptr<const uint8_t>(insn.payload[3]);
  const uint8_t *up_a_fp4 = payload_ptr<const uint8_t>(insn.payload[4]);
  const uint8_t *up_a_scale = payload_ptr<const uint8_t>(insn.payload[5]);
  const uint8_t *b_fp4 = payload_ptr<const uint8_t>(insn.payload[6]);
  const uint8_t *b_scale = payload_ptr<const uint8_t>(insn.payload[7]);
  __nv_bfloat16 *gate_output =
      payload_ptr<__nv_bfloat16>(insn.payload[8]);
  __nv_bfloat16 *up_output = payload_ptr<__nv_bfloat16>(insn.payload[9]);
  const float gate_alpha = unpack_gemv_alpha_f32(insn.payload[10]);
  const float up_alpha = unpack_gemv_alpha_f32(insn.payload[11]);

  constexpr size_t kAlign16 =
      16u * static_cast<size_t>(qwen36_gemv::kKPerMma);
  if (m == 0 || k == 0 || (m % qwen36_gemv::kRowsPerBlock) != 0 ||
      (k % kAlign16) != 0 || gate_a_fp4 == nullptr ||
      gate_a_scale == nullptr || up_a_fp4 == nullptr ||
      up_a_scale == nullptr || b_fp4 == nullptr || b_scale == nullptr ||
      gate_output == nullptr || up_output == nullptr) {
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
        static_cast<unsigned>(tile), gate_a_fp4, gate_a_scale, b_fp4,
        b_scale, gate_alpha, gate_output, m, k, pages.base);
    __syncthreads();
    qwen36_gemv::nvfp4_gemv_mma_body_with_smem<16>(
        static_cast<unsigned>(tile), up_a_fp4, up_a_scale, b_fp4, b_scale,
        up_alpha, up_output, m, k, pages.base);
    __syncthreads();
  }
}

} // namespace qwen36_interpreter
