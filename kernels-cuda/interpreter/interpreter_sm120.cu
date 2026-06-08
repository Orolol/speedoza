#include "instruction.h"
#include "counters.cuh"
#include "opcodes/fallback_trampoline.cuh"
#include "opcodes/nvfp4_gemv.cuh"
#include "opcodes/residual_add.cuh"
#include "opcodes/rmsnorm_nvfp4_quant.cuh"
#include "opcodes/rope_partial.cuh"
#include "opcodes/swiglu_nvfp4_quant.cuh"
#include "page_allocator.cuh"
#include "../active_stream.h"

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef QWEN36_INTERPRETER_THREADS
#define QWEN36_INTERPRETER_THREADS 512
#endif

namespace {

int status(cudaError_t value) {
  return value == cudaSuccess ? QWEN36_STATUS_SUCCESS
                              : QWEN36_STATUS_CUDA_ERROR;
}

template <typename T> T *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

struct InterpreterState {
  uint32_t pc;
  uint32_t error_count;
};

__device__ inline bool
valid_dependency_count(const qwen36_interpreter_instruction_t &insn) {
  return insn.dep_count <= QWEN36_INTERPRETER_MAX_DEPS;
}

__device__ inline void
dispatch_instruction(const qwen36_interpreter_instruction_t &insn,
                     qwen36_interpreter::PageAllocator &pages,
                     InterpreterState &state, float *scratch,
                     float *swiglu_decoded_scale, float *swiglu_staged) {
  if (!qwen36_interpreter_opcode_known(insn.opcode)) {
    if (threadIdx.x == 0) {
      atomicAdd(&state.error_count, 1u);
    }
    return;
  }

  switch (insn.opcode) {
  case QWEN36_INTERPRETER_OPCODE_FALLBACK_TRAMPOLINE:
  case QWEN36_INTERPRETER_OPCODE_ATTN_DECODE_FULL:
  case QWEN36_INTERPRETER_OPCODE_DELTANET_RECUR:
  case QWEN36_INTERPRETER_OPCODE_LM_HEAD_TILED:
    qwen36_interpreter::exec_fallback_trampoline(insn, pages);
    break;
  case QWEN36_INTERPRETER_OPCODE_RMSNORM_NVFP4_QUANT:
    qwen36_interpreter::exec_rmsnorm_nvfp4_quant(insn, pages, scratch);
    break;
  case QWEN36_INTERPRETER_OPCODE_NVFP4_GEMV:
    qwen36_interpreter::exec_nvfp4_gemv(insn, pages);
    break;
  case QWEN36_INTERPRETER_OPCODE_SWIGLU_NVFP4_QUANT:
    qwen36_interpreter::exec_swiglu_nvfp4_quant(
        insn, pages, scratch, swiglu_decoded_scale, swiglu_staged);
    break;
  case QWEN36_INTERPRETER_OPCODE_ROPE_PARTIAL:
    qwen36_interpreter::exec_rope_partial(insn, pages);
    break;
  case QWEN36_INTERPRETER_OPCODE_RESIDUAL_ADD:
    qwen36_interpreter::exec_residual_add(insn, pages);
    break;
  default:
    break;
  }
}

__global__ void __launch_bounds__(QWEN36_INTERPRETER_THREADS)
    qwen36_interpreter_decode_kernel(
        const qwen36_interpreter_instruction_t *__restrict__ instructions,
        uint32_t instruction_count, int32_t *__restrict__ counters,
        uint32_t counter_count) {
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ InterpreterState state;
  __shared__ qwen36_interpreter::PageAllocator pages;
  __shared__ float scratch[QWEN36_INTERPRETER_THREADS];
  __shared__ float swiglu_decoded_scale;
  __shared__ float swiglu_staged[16];

  if (threadIdx.x == 0) {
    state.pc = 0;
    state.error_count = 0;
    pages.init(smem, qwen36_interpreter::PageAllocator::kTotalBytes);
  }
  __syncthreads();

  while (true) {
    const uint32_t pc = state.pc;
    if (pc >= instruction_count) {
      break;
    }

    const qwen36_interpreter_instruction_t insn = instructions[pc];
    if (insn.opcode == QWEN36_INTERPRETER_OPCODE_EXIT) {
      break;
    }

    if (!valid_dependency_count(insn)) {
      if (threadIdx.x == 0) {
        atomicAdd(&state.error_count, 1u);
      }
      break;
    }

    for (uint32_t dep_idx = 0; dep_idx < insn.dep_count; ++dep_idx) {
      qwen36_interpreter::wait_for_counter(counters, counter_count,
                                           insn.deps[dep_idx]);
    }

    dispatch_instruction(insn, pages, state, scratch, &swiglu_decoded_scale,
                         swiglu_staged);
    __syncthreads();

    if (threadIdx.x == 0) {
      qwen36_interpreter::arrive_and_publish_last_cta(
          counters, counter_count, insn.arrival_counter,
          insn.publishes_counter, insn.publish_value, gridDim.x);
      state.pc = pc + 1;
    }
    __syncthreads();
  }
}

uint32_t derive_cta_count(uint32_t requested_ctas) {
  if (requested_ctas != 0) {
    return requested_ctas;
  }

  int active_blocks_per_sm = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, qwen36_interpreter_decode_kernel,
      QWEN36_INTERPRETER_THREADS,
      qwen36_interpreter::PageAllocator::kTotalBytes);
  if (err != cudaSuccess || active_blocks_per_sm <= 0) {
    return 1;
  }

  int device = 0;
  err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return static_cast<uint32_t>(active_blocks_per_sm);
  }

  cudaDeviceProp props{};
  err = cudaGetDeviceProperties(&props, device);
  if (err != cudaSuccess || props.multiProcessorCount <= 0) {
    return static_cast<uint32_t>(active_blocks_per_sm);
  }

  return static_cast<uint32_t>(active_blocks_per_sm *
                               props.multiProcessorCount);
}

bool host_validate_program(const qwen36_interpreter_program_t *program) {
  if (program == nullptr) {
    return false;
  }
  if (program->instructions.ptr == 0 || program->instruction_count == 0) {
    return false;
  }
  if (program->instruction_count > UINT32_MAX) {
    return false;
  }
  if (program->counters_i32.ptr == 0 || program->counter_count == 0) {
    return false;
  }
  if (program->counter_count > UINT32_MAX) {
    return false;
  }
  return true;
}

} // namespace

extern "C" int
qwen36_interpreter_decode_sm120(const qwen36_interpreter_program_t *program) {
  if (program == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (!host_validate_program(program)) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  constexpr size_t smem_bytes =
      qwen36_interpreter::PageAllocator::kTotalBytes;
  cudaError_t err = cudaFuncSetAttribute(
      qwen36_interpreter_decode_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
  if (err != cudaSuccess) {
    return status(err);
  }
  err = cudaFuncSetCacheConfig(qwen36_interpreter_decode_kernel,
                               cudaFuncCachePreferShared);
  if (err != cudaSuccess) {
    return status(err);
  }

  const uint32_t cta_count = derive_cta_count(program->cta_count);
  qwen36_interpreter_decode_kernel<<<cta_count, QWEN36_INTERPRETER_THREADS,
                                     smem_bytes,
                                     qwen36_internal_active_stream()>>>(
      ptr<const qwen36_interpreter_instruction_t>(program->instructions),
      static_cast<uint32_t>(program->instruction_count),
      ptr<int32_t>(program->counters_i32),
      static_cast<uint32_t>(program->counter_count));

  err = cudaGetLastError();
  return status(err);
}
