#pragma once

#include "instruction.h"

#include <stdint.h>

namespace qwen36_interpreter {

__device__ inline int32_t load_counter_acquire(const int32_t *ptr) {
  int32_t value;
  asm volatile("ld.acquire.gpu.s32 %0, [%1];" : "=r"(value) : "l"(ptr));
  return value;
}

__device__ inline void wait_for_counter(const int32_t *counters,
                                        uint32_t counter_count,
                                        qwen36_interpreter_dep_t dep) {
  if (dep.counter_id >= counter_count) {
    return;
  }
  const int32_t *slot = counters + dep.counter_id;
  uint32_t sleep_ns = 8;
  while (load_counter_acquire(slot) < static_cast<int32_t>(dep.target)) {
    __nanosleep(sleep_ns);
    sleep_ns = sleep_ns < 256 ? sleep_ns << 1 : sleep_ns;
  }
}

__device__ inline void publish_counter(int32_t *counters, uint32_t counter_count,
                                       uint32_t counter_id,
                                       uint32_t publish_value) {
  if (counter_id >= counter_count || publish_value == 0) {
    return;
  }
  __threadfence_system();
  atomicAdd(counters + counter_id, static_cast<int32_t>(publish_value));
}

__device__ inline void arrive_and_publish_last_cta(
    int32_t *counters, uint32_t counter_count, uint32_t arrival_counter_id,
    uint32_t publishes_counter_id, uint32_t publish_value,
    uint32_t expected_ctas) {
  if (arrival_counter_id >= counter_count) {
    return;
  }
  __threadfence_system();
  const int32_t prior =
      atomicAdd(counters + arrival_counter_id, static_cast<int32_t>(1));
  if (static_cast<uint32_t>(prior + 1) == expected_ctas) {
    publish_counter(counters, counter_count, publishes_counter_id,
                    publish_value);
  }
}

} // namespace qwen36_interpreter
