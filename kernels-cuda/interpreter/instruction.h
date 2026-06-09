#pragma once

#include "../include/qwen36_fp4.h"

#include <stdint.h>

// Keep these constants mirrored with crates/kernels/src/interpreter.rs.
#ifndef QWEN36_INTERPRETER_MAX_DEPS
#define QWEN36_INTERPRETER_MAX_DEPS 4
#endif
#ifndef QWEN36_INTERPRETER_PAYLOAD_U64S
#define QWEN36_INTERPRETER_PAYLOAD_U64S 12
#endif

enum qwen36_interpreter_opcode : uint16_t {
  QWEN36_INTERPRETER_OPCODE_EXIT = 0,
  QWEN36_INTERPRETER_OPCODE_FALLBACK_TRAMPOLINE = 1,
  QWEN36_INTERPRETER_OPCODE_RMSNORM_NVFP4_QUANT = 2,
  QWEN36_INTERPRETER_OPCODE_NVFP4_GEMV = 3,
  QWEN36_INTERPRETER_OPCODE_SWIGLU_NVFP4_QUANT = 4,
  QWEN36_INTERPRETER_OPCODE_ROPE_PARTIAL = 5,
  QWEN36_INTERPRETER_OPCODE_ATTN_DECODE_FULL = 6,
  QWEN36_INTERPRETER_OPCODE_DELTANET_RECUR = 7,
  QWEN36_INTERPRETER_OPCODE_RESIDUAL_ADD = 8,
  QWEN36_INTERPRETER_OPCODE_LM_HEAD_TILED = 9,
  QWEN36_INTERPRETER_OPCODE_RMSNORM_BF16 = 10,
  QWEN36_INTERPRETER_OPCODE_Q_PROJ_DEINTERLEAVE = 11,
  QWEN36_INTERPRETER_OPCODE_Q_PROJ_SIGMOID_GATE = 12,
  QWEN36_INTERPRETER_OPCODE_NVFP4_QUANTIZE = 13,
  QWEN36_INTERPRETER_OPCODE_SWIGLU_BF16 = 14,
  QWEN36_INTERPRETER_OPCODE_CONV1D_GDN_GATE_FUSED = 15,
};

static_assert(sizeof(qwen36_interpreter_dep_t) == 8,
              "qwen36_interpreter_dep_t ABI drift");
static_assert(sizeof(qwen36_interpreter_instruction_t) == 152,
              "qwen36_interpreter_instruction_t ABI drift");
static_assert(sizeof(qwen36_interpreter_program_t) == 40,
              "qwen36_interpreter_program_t ABI drift");

__host__ __device__ inline bool
qwen36_interpreter_opcode_known(uint16_t opcode) {
  return opcode <= QWEN36_INTERPRETER_OPCODE_CONV1D_GDN_GATE_FUSED;
}

namespace qwen36_interpreter {

template <typename T> __host__ __device__ inline T *payload_ptr(uint64_t raw) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(raw));
}

} // namespace qwen36_interpreter
