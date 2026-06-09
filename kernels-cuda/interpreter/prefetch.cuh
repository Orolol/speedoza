#pragma once

// Lookahead L2 prefetch for the interpreter dispatch loop.
//
// Between the counter-wait and the dispatch of instruction `pc`, this helper
// inspects `program[pc + 1]` and, for opcodes whose weight matrix is known
// from the payload, issues a small budget of `prefetch.global.L2` hints so
// the next instruction's first cache lines land warm. The helper never
// blocks and never writes shared memory; it is safe to call from every
// thread of the dispatch CTA.
//
// Mechanism:
//   * Each call walks at most `kPrefetchBytes` bytes (64 KiB by default),
//     one 128-byte cache line per thread, with `__ldg(...)`-free prefetch
//     PTX. With 512 threads in the dispatch CTA this prefetches all 64 KiB
//     in a single PTX issue per thread, which the SM can pipeline with
//     instruction `pc`'s compute. The choice of 64 KiB is small enough to
//     survive in L2 (128 MiB on Blackwell) through instruction `pc`'s own
//     loads.
//   * Pointers and sizes are derived purely from the canonical instruction
//     payload layout in `instruction.h` / `interpreter.rs`; this file does
//     NOT add new payload fields, so the ABI is unchanged.
//   * Unknown opcodes are a no-op.

#include "instruction.h"

#include <stdint.h>

namespace qwen36_interpreter {

constexpr uint32_t kPrefetchCacheLineBytes = 128u;
constexpr uint32_t kPrefetchBudgetBytes = 64u * 1024u;

__device__ inline void prefetch_l2_one_line(const void *addr) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(addr));
}

__device__ inline void prefetch_range_l2(const uint8_t *base, uint32_t bytes) {
  if (base == nullptr || bytes == 0) {
    return;
  }
  const uint32_t budget =
      bytes < kPrefetchBudgetBytes ? bytes : kPrefetchBudgetBytes;
  const uint32_t lines =
      (budget + kPrefetchCacheLineBytes - 1) / kPrefetchCacheLineBytes;
  const uint32_t threads = blockDim.x;
  for (uint32_t line = threadIdx.x; line < lines; line += threads) {
    prefetch_l2_one_line(base + static_cast<size_t>(line) *
                                    static_cast<size_t>(kPrefetchCacheLineBytes));
  }
}

__device__ inline void
prefetch_next_instruction_weights(const qwen36_interpreter_instruction_t &next) {
  switch (next.opcode) {
  case QWEN36_INTERPRETER_OPCODE_NVFP4_GEMV: {
    // payload[0] = m, payload[1] = k, payload[2] = a_fp4 (weights),
    // payload[3] = a_scale_e4m3. Weight bytes = m * k / 2 (FP4 = 4 bits).
    const auto m = static_cast<uint64_t>(next.payload[0]);
    const auto k = static_cast<uint64_t>(next.payload[1]);
    const uint64_t weight_bytes = (m * k) / 2u;
    const auto *base =
        reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(next.payload[2]));
    const uint32_t to_fetch =
        weight_bytes > kPrefetchBudgetBytes
            ? kPrefetchBudgetBytes
            : static_cast<uint32_t>(weight_bytes);
    prefetch_range_l2(base, to_fetch);
    break;
  }
  case QWEN36_INTERPRETER_OPCODE_NVFP4_GEMV_PAIR: {
    // payload[2]/[4] are the two FP4 weight matrices. Split the small
    // lookahead budget so both first tiles can land warm when prefetch is
    // explicitly enabled.
    const auto m = static_cast<uint64_t>(next.payload[0]);
    const auto k = static_cast<uint64_t>(next.payload[1]);
    const uint64_t weight_bytes = (m * k) / 2u;
    const uint32_t half_budget = kPrefetchBudgetBytes / 2u;
    const uint32_t to_fetch =
        weight_bytes > half_budget ? half_budget
                                   : static_cast<uint32_t>(weight_bytes);
    const auto *gate_base =
        reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(next.payload[2]));
    const auto *up_base =
        reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(next.payload[4]));
    prefetch_range_l2(gate_base, to_fetch);
    prefetch_range_l2(up_base, to_fetch);
    break;
  }
  case QWEN36_INTERPRETER_OPCODE_LM_HEAD_TILED: {
    // payload[0] = out_features, payload[1] = in_features,
    // payload[3] = weight_bf16. We only prime the first slice — the head is
    // ~12 GiB total, way beyond any L2 budget, but the first BF16 rows seed
    // the cp.async pipeline of the first matvec body.
    const auto *base =
        reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(next.payload[3]));
    prefetch_range_l2(base, kPrefetchBudgetBytes);
    break;
  }
  case QWEN36_INTERPRETER_OPCODE_RMSNORM_BF16: {
    // payload[1] = hidden, payload[3] = weight_bf16 (hidden * 2 bytes).
    const auto hidden = static_cast<uint64_t>(next.payload[1]);
    const uint64_t weight_bytes = hidden * sizeof(uint16_t);
    const auto *base =
        reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(next.payload[3]));
    const uint32_t to_fetch =
        weight_bytes > kPrefetchBudgetBytes
            ? kPrefetchBudgetBytes
            : static_cast<uint32_t>(weight_bytes);
    prefetch_range_l2(base, to_fetch);
    break;
  }
  case QWEN36_INTERPRETER_OPCODE_RMSNORM_NVFP4_QUANT: {
    // payload[0] = hidden, payload[2] = weight_bf16 (hidden * 2 bytes).
    const auto hidden = static_cast<uint64_t>(next.payload[0]);
    const uint64_t weight_bytes = hidden * sizeof(uint16_t);
    const auto *base =
        reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(next.payload[2]));
    const uint32_t to_fetch =
        weight_bytes > kPrefetchBudgetBytes
            ? kPrefetchBudgetBytes
            : static_cast<uint32_t>(weight_bytes);
    prefetch_range_l2(base, to_fetch);
    break;
  }
  default:
    break;
  }
}

} // namespace qwen36_interpreter
