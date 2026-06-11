#pragma once

#include <stdint.h>

namespace qwen36_interpreter {

enum PageKind : uint32_t {
  PAGE_KIND_WEIGHT = 1,
  PAGE_KIND_ACTIVATION = 2,
  PAGE_KIND_SCRATCH = 3,
  PAGE_KIND_METADATA = 4,
};

struct PageHandle {
  int32_t slot;
};

struct PageSlot {
  uint32_t offset;
  uint32_t bytes;
  uint32_t kind;
  uint32_t state;
};

struct PageAllocator {
  static constexpr uint32_t kWeightPageBytes = 16u * 1024u;
  static constexpr uint32_t kActivationPageBytes = 8u * 1024u;
  static constexpr uint32_t kScratchBytes = 4u * 1024u;
  static constexpr uint32_t kMetadataBytes = 2u * 1024u;
  static constexpr uint32_t kPageCount = 8u;
  static constexpr uint32_t kTotalBytes = 4u * kWeightPageBytes +
                                          2u * kActivationPageBytes +
                                          kScratchBytes + kMetadataBytes;

  unsigned char *base;
  uint32_t total_bytes;
  PageSlot slots[kPageCount];

  __device__ void init(void *smem, uint32_t smem_bytes) {
    base = reinterpret_cast<unsigned char *>(smem);
    total_bytes = smem_bytes;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < 4; ++i) {
      slots[i] = {offset, kWeightPageBytes, PAGE_KIND_WEIGHT, 0};
      offset += kWeightPageBytes;
    }
    for (uint32_t i = 0; i < 2; ++i) {
      slots[4 + i] = {offset, kActivationPageBytes, PAGE_KIND_ACTIVATION, 0};
      offset += kActivationPageBytes;
    }
    slots[6] = {offset, kScratchBytes, PAGE_KIND_SCRATCH, 0};
    offset += kScratchBytes;
    slots[7] = {offset, kMetadataBytes, PAGE_KIND_METADATA, 0};
  }

  __device__ PageHandle acquire(PageKind kind, uint32_t bytes) {
    for (uint32_t i = 0; i < kPageCount; ++i) {
      PageSlot &slot = slots[i];
      if (slot.kind == static_cast<uint32_t>(kind) && slot.bytes >= bytes &&
          slot.offset + bytes <= total_bytes &&
          atomicCAS(reinterpret_cast<unsigned int *>(&slot.state), 0u, 1u) ==
              0u) {
        return {static_cast<int32_t>(i)};
      }
    }
    return {-1};
  }

  __device__ void release(PageHandle handle) {
    if (handle.slot < 0 || handle.slot >= static_cast<int32_t>(kPageCount)) {
      return;
    }
    __threadfence_block();
    slots[handle.slot].state = 0;
  }

  __device__ void *ptr(PageHandle handle) {
    if (handle.slot < 0 || handle.slot >= static_cast<int32_t>(kPageCount)) {
      return nullptr;
    }
    PageSlot &slot = slots[handle.slot];
    if (slot.offset >= total_bytes) {
      return nullptr;
    }
    return base + slot.offset;
  }
};

static_assert(PageAllocator::kTotalBytes == 86u * 1024u,
              "interpreter SMEM page budget changed unexpectedly");

} // namespace qwen36_interpreter
