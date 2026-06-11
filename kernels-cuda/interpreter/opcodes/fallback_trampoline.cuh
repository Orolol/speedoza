#pragma once

#include "../instruction.h"
#include "../page_allocator.cuh"

namespace qwen36_interpreter {

__device__ inline void
exec_fallback_trampoline(const qwen36_interpreter_instruction_t &,
                         PageAllocator &) {
  // Stage 0 deliberately keeps non-EXIT opcodes as device-side no-ops. The
  // host path remains responsible for real kernels until an opcode body is
  // ported into the interpreter.
}

} // namespace qwen36_interpreter
