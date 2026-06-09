// DFlash drafter attention — Phase C v1.
//
// First-light non-causal BF16 attention for the DFlash drafter forward.
// The caller pre-concatenates K = [k_ctx; k_noise] and V = [v_ctx;
// v_noise] into a single contiguous buffer of length `kv_seq_len`. The
// kernel does **not** manage the KV cache; the speculative controller
// is responsible for cache append + DynamicCache-style `crop()` between
// drafter forwards.
//
// Layout convention:
//   Q       : [q_len, q_heads, head_dim]    BF16, post-RoPE, post-q_norm
//   K, V    : [kv_seq_len, kv_heads, head_dim]  BF16, post-RoPE,
//                                              post-k_norm
//   Output  : [q_len, q_heads, head_dim]    BF16
//
// Grid: (q_len, q_heads, 1); 1 CTA per (q_pos, q_head).
// Block: head_dim threads. Currently specialised to head_dim == 128
// (DFlash drafter's only head_dim). Other shapes return
// QWEN36_STATUS_NOT_IMPLEMENTED so the controller can fall back.
//
// Algorithm: per-CTA online softmax (FlashAttention-style accumulation).
// Each thread holds one element of the Q vector and one element of the
// running output accumulator; scalar reductions (`score`, `running_max`,
// `running_sum`) are computed once per key step via warp + block
// reductions, then broadcast through shared memory.
//
// Performance: this version reads K/V straight from HBM per key step
// (no tiling, no double-buffering). Adequate for block_size=16,
// kv_seq_len ~< 2k on a chat workload — measured at ~3 % of one decode
// token's budget. Optimisation paths (FlashAttention-2 tiling, GQA
// K-broadcast, TMA loads) are deferred to Phase D once parity is
// established.

#include "include/qwen36_fp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int WARP_SIZE = 32;

template <typename T> T *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

// One CTA per (q_pos, q_head). `HEAD_DIM` threads per CTA, one element
// per thread.
template <int HEAD_DIM>
__global__ void drafter_attention_block_v1_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    __nv_bfloat16 *__restrict__ output,
    int kv_seq_len,
    int q_heads,
    int kv_heads,
    int sliding_window,
    float scale) {
  static_assert(HEAD_DIM % WARP_SIZE == 0,
                "HEAD_DIM must be a multiple of warp size (32)");
  constexpr int N_WARPS = HEAD_DIM / WARP_SIZE;

  const int q_pos = blockIdx.x;
  const int q_head = blockIdx.y;
  const int q_len = gridDim.x;
  const int tid = threadIdx.x;

  // Integer GQA broadcast: each q_head maps to one kv_head.
  const int kv_head = (q_head * kv_heads) / q_heads;

  __shared__ float q_smem[HEAD_DIM];
  __shared__ float warp_partials[N_WARPS];
  __shared__ float score_broadcast;

  // Stage Q[q_pos, q_head, :] into SMEM once.
  const int q_offset = (q_pos * q_heads + q_head) * HEAD_DIM + tid;
  q_smem[tid] = __bfloat162float(q[q_offset]);
  __syncthreads();

  // Online softmax state. `o_acc` is per-thread (each thread owns one
  // element of the head-dim output vector); `running_max` and
  // `running_sum` are scalar but kept per-thread (every thread reaches
  // the same value because `score` is broadcast from SMEM).
  float o_acc = 0.0f;
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  // Q's absolute position in the cache: the q_len noise tokens come
  // after `kv_seq_len - q_len` context tokens in the KV layout.
  const int q_abs = kv_seq_len - q_len + q_pos;

  for (int j = 0; j < kv_seq_len; ++j) {
    // Sliding-window mask (symmetric, non-causal): skip keys outside
    // `[q_abs - sliding_window, q_abs + sliding_window]`.
    if (sliding_window > 0) {
      const int delta = q_abs - j;
      const int abs_delta = delta < 0 ? -delta : delta;
      if (abs_delta > sliding_window) {
        continue;
      }
    }

    const int kv_offset = (j * kv_heads + kv_head) * HEAD_DIM + tid;
    const float k_val = __bfloat162float(k[kv_offset]);
    const float v_val = __bfloat162float(v[kv_offset]);

    // dot(q, k_j) — block-wide reduction via two warp shuffles.
    float local = q_smem[tid] * k_val;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
      local += __shfl_xor_sync(0xffffffffu, local, off);
    }
    if ((tid & (WARP_SIZE - 1)) == 0) {
      warp_partials[tid / WARP_SIZE] = local;
    }
    __syncthreads();

    // Reduce N_WARPS partials inside a single warp's worth of lanes.
    if (tid < N_WARPS) {
      float w = warp_partials[tid];
      const unsigned mask = (1u << N_WARPS) - 1u;
      for (int off = N_WARPS / 2; off > 0; off >>= 1) {
        w += __shfl_xor_sync(mask, w, off);
      }
      if (tid == 0) {
        score_broadcast = w * scale;
      }
    }
    __syncthreads();
    const float score = score_broadcast;

    // Online softmax update (FlashAttention-1).
    const float new_max = fmaxf(running_max, score);
    const float alpha =
        (running_max == -INFINITY) ? 0.0f : expf(running_max - new_max);
    const float beta = expf(score - new_max);
    running_sum = running_sum * alpha + beta;
    o_acc = o_acc * alpha + beta * v_val;
    running_max = new_max;
  }

  // Normalize. `running_sum == 0` only if every key was masked out
  // (degenerate SWA configuration); write zeros to keep the output
  // defined.
  const float out_val = (running_sum > 0.0f) ? (o_acc / running_sum) : 0.0f;
  const int out_offset = (q_pos * q_heads + q_head) * HEAD_DIM + tid;
  output[out_offset] = __float2bfloat16(out_val);
}

} // namespace

// Forward declaration: FA-tiled kernel entry from drafter_attention_flash.cu.
// Returns QWEN36_STATUS_NOT_IMPLEMENTED for shapes outside its specialised
// (head_dim=128, q_len=16) gate, in which case we fall back to the v1
// per-key inner-loop kernel below.
extern "C" int qwen36_drafter_attention_block_flash_bf16(
    const qwen36_drafter_attention_block_spec_t *spec);

extern "C" int qwen36_drafter_attention_block_bf16(
    const qwen36_drafter_attention_block_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0 || spec->v_bf16.ptr == 0 ||
      spec->output_bf16.ptr == 0) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->q_len == 0 || spec->kv_seq_len == 0 || spec->q_heads == 0 ||
      spec->kv_heads == 0 || spec->head_dim == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (spec->q_heads % spec->kv_heads != 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if (spec->head_dim != 128) {
    // Phase C v1 only specialises head_dim == 128 (the DFlash drafter's
    // only head_dim). Other values fall back via the dispatcher.
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  // Try the FA-tiled kernel first for the common shape (q_len=16). It
  // returns NOT_IMPLEMENTED on shape mismatch or when
  // QWEN36_DRAFTER_ATTENTION_DISABLE_FLASH=1; either way we fall back to
  // the v1 reference kernel that handles every shape.
  const int flash_status = qwen36_drafter_attention_block_flash_bf16(spec);
  if (flash_status == QWEN36_STATUS_SUCCESS ||
      flash_status == QWEN36_STATUS_CUDA_ERROR ||
      flash_status == QWEN36_STATUS_NULL_POINTER ||
      flash_status == QWEN36_STATUS_INVALID_ARGUMENT) {
    return flash_status;
  }
  // flash_status == QWEN36_STATUS_NOT_IMPLEMENTED → fall back.

  const float scale = 1.0f / sqrtf(static_cast<float>(spec->head_dim));

  const auto *q = ptr<const __nv_bfloat16>(spec->q_bf16);
  const auto *k = ptr<const __nv_bfloat16>(spec->k_bf16);
  const auto *v = ptr<const __nv_bfloat16>(spec->v_bf16);
  auto *output = ptr<__nv_bfloat16>(spec->output_bf16);

  dim3 grid(static_cast<unsigned>(spec->q_len),
            static_cast<unsigned>(spec->q_heads),
            1);
  dim3 block(static_cast<unsigned>(spec->head_dim), 1, 1);

  drafter_attention_block_v1_kernel<128><<<grid, block, 0, 0>>>(
      q, k, v, output,
      static_cast<int>(spec->kv_seq_len),
      static_cast<int>(spec->q_heads),
      static_cast<int>(spec->kv_heads),
      static_cast<int>(spec->sliding_window),
      scale);

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
