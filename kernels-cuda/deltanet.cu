#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

template <typename T> T *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

// FlashQLA-inspired Gated DeltaNet decode kernel.
//
// Numerically equivalent to the reference scalar implementation but rewritten
// to halve the BF16 memory traffic per step:
//
// - q_tok / k_tok are pulled into shared memory via a single cooperative load
//   per token, so the inner kd-loops below access them at L1 latency instead
//   of re-reading the same global addresses 256 times per thread.
// - The compute over the recurrent state collapses the original two-pass
//   `kv_mem` sweep + `delta`-update sweep into one pass that accumulates three
//   partial sums in registers (kv_mem, sum(decayed*query), sum(key*query)).
//   The final output is then `s_q + delta * k_q` — one FMA — and the state is
//   written to global memory ONCE (the original kernel wrote it twice: once
//   with the decayed value, once with the post-update value).
// - When `key_dim` is a multiple of 8 and the row is uint4-aligned, both the
//   read and write of the state row issue uint4 loads/stores, packing 8 BF16
//   values per transaction instead of one per loop iteration.
//
// Block layout (unchanged): one block per v-head, threadIdx.x = vd. With
// value_dim ≤ 128 and 128 threads per block, every thread owns exactly one
// (vh, vd) recurrent state row.
constexpr int kDeltaNetBlockThreads = 128;

__global__ void __launch_bounds__(kDeltaNetBlockThreads)
deltanet_decode_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    const float *__restrict__ gate,
    const float *__restrict__ beta,
    __nv_bfloat16 *__restrict__ state,
    __nv_bfloat16 *__restrict__ output, qwen36_deltanet_shape_t shape,
    size_t tokens, size_t q_token_stride, size_t k_token_stride,
    size_t v_token_stride, float state_decay, float update_scale,
    bool qk_l2norm) {
  __shared__ float shared_q[256];
  __shared__ float shared_k[256];
  // Two scratch slots so q^2 and k^2 can be reduced in a single tree pass —
  // halves the __syncthreads count vs reducing them sequentially.
  __shared__ float reduction_q[kDeltaNetBlockThreads];
  __shared__ float reduction_k[kDeltaNetBlockThreads];
  __shared__ float shared_decay;
  __shared__ float shared_beta;
  __shared__ float shared_q_norm;
  __shared__ float shared_k_norm;

  const size_t vd = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t vh = blockIdx.y;
  if (vh >= shape.v_heads) {
    return;
  }
  const bool active = vd < shape.value_dim;
  const bool gated = (gate != nullptr) && (beta != nullptr);

  const size_t q_repeat = shape.v_heads / shape.qk_heads;
  const size_t qh = vh / q_repeat;
  const size_t q_stride =
      q_token_stride == 0 ? shape.qk_heads * shape.key_dim : q_token_stride;
  const size_t k_stride =
      k_token_stride == 0 ? shape.qk_heads * shape.key_dim : k_token_stride;
  const size_t v_stride =
      v_token_stride == 0 ? shape.v_heads * shape.value_dim : v_token_stride;
  __nv_bfloat16 *state_row =
      active ? state + (vh * shape.value_dim + vd) * shape.key_dim : nullptr;

  const size_t key_dim = shape.key_dim;
  const bool vector_state = active && (key_dim % 8 == 0) &&
                            ((reinterpret_cast<uintptr_t>(state_row) & 15u) == 0);

  for (size_t tok = 0; tok < tokens; ++tok) {
    const __nv_bfloat16 *q_tok = q + tok * q_stride + qh * shape.key_dim;
    const __nv_bfloat16 *k_tok = k + tok * k_stride + qh * shape.key_dim;
    const float v_value =
        active ? __bfloat162float(v[tok * v_stride + vh * shape.value_dim + vd])
               : 0.0f;

    // Cooperative load fused with the q^2 / k^2 partials: each thread loads
    // its kd-stride into shared memory AND accumulates the squared sums in
    // the same pass. For the gated path the upcoming tree reduction provides
    // the visibility barrier; the non-gated path adds an explicit sync.
    float q_sq = 0.0f, k_sq = 0.0f;
    for (size_t kd = threadIdx.x; kd < key_dim; kd += blockDim.x) {
      const float qv = __bfloat162float(q_tok[kd]);
      const float kv = __bfloat162float(k_tok[kd]);
      shared_q[kd] = qv;
      shared_k[kd] = kv;
      q_sq += qv * qv;
      k_sq += kv * kv;
    }
    if (gated && threadIdx.x == 0) {
      shared_decay = expf(gate[tok * shape.v_heads + vh]);
      shared_beta = beta[tok * shape.v_heads + vh];
    }

    if (gated) {
      reduction_q[threadIdx.x] = q_sq;
      reduction_k[threadIdx.x] = k_sq;
      // First barrier of the reduction also publishes the cooperative load
      // and the shared_decay / shared_beta writes above.
      __syncthreads();
      for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
          reduction_q[threadIdx.x] += reduction_q[threadIdx.x + s];
          reduction_k[threadIdx.x] += reduction_k[threadIdx.x + s];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        const float q_squares = reduction_q[0];
        const float k_squares = reduction_k[0];
        if (qk_l2norm) {
          shared_q_norm = rsqrtf(q_squares + 1.0e-6f) *
                          rsqrtf(static_cast<float>(key_dim));
          shared_k_norm = rsqrtf(k_squares + 1.0e-6f);
        } else {
          shared_q_norm = rsqrtf(static_cast<float>(key_dim));
          shared_k_norm = 1.0f;
        }
      }
      __syncthreads();
      if (!active) {
        if (tok + 1 < tokens) {
          __syncthreads();
        }
        continue;
      }

      const float decay = shared_decay;
      const float beta_value = shared_beta;
      const float q_norm = shared_q_norm;
      const float k_norm = shared_k_norm;

      // Single-pass: kv_mem = sum(decayed*key), s_q = sum(decayed*query),
      // k_q = sum(key*query). The final acc collapses to s_q + delta*k_q,
      // and `state_row` is left untouched until the second loop writes the
      // post-update value (avoids the redundant decayed-only write).
      float kv_mem = 0.0f;
      float s_q = 0.0f;
      float k_q = 0.0f;
      if (vector_state) {
        const uint4 *state_vec_in =
            reinterpret_cast<const uint4 *>(state_row);
        const size_t vec_count = key_dim / 8;
#pragma unroll 1
        for (size_t i = 0; i < vec_count; ++i) {
          const uint4 chunk = state_vec_in[i];
          const __nv_bfloat162 *as_pairs =
              reinterpret_cast<const __nv_bfloat162 *>(&chunk);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float s0 = __bfloat162float(as_pairs[j].x);
            const float s1 = __bfloat162float(as_pairs[j].y);
            const size_t kd0 = i * 8 + 2 * j;
            const float decayed0 = s0 * decay;
            const float decayed1 = s1 * decay;
            const float k0 = shared_k[kd0] * k_norm;
            const float k1 = shared_k[kd0 + 1] * k_norm;
            const float q0 = shared_q[kd0] * q_norm;
            const float q1 = shared_q[kd0 + 1] * q_norm;
            kv_mem += decayed0 * k0 + decayed1 * k1;
            s_q += decayed0 * q0 + decayed1 * q1;
            k_q += k0 * q0 + k1 * q1;
          }
        }
      } else {
        for (size_t kd = 0; kd < key_dim; ++kd) {
          const float decayed = __bfloat162float(state_row[kd]) * decay;
          const float key = shared_k[kd] * k_norm;
          const float query = shared_q[kd] * q_norm;
          kv_mem += decayed * key;
          s_q += decayed * query;
          k_q += key * query;
        }
      }

      const float delta = (v_value - kv_mem) * beta_value;
      const float acc = s_q + delta * k_q;
      output[(tok * shape.v_heads + vh) * shape.value_dim + vd] =
          __float2bfloat16(acc);

      // Apply the recurrence write-back. Reread the original state (stays in
      // L1 from the first pass), apply decay, add key * delta, store. One
      // global write per element instead of two.
      if (vector_state) {
        const uint4 *state_vec_in =
            reinterpret_cast<const uint4 *>(state_row);
        uint4 *state_vec_out = reinterpret_cast<uint4 *>(state_row);
        const size_t vec_count = key_dim / 8;
#pragma unroll 1
        for (size_t i = 0; i < vec_count; ++i) {
          const uint4 chunk = state_vec_in[i];
          const __nv_bfloat162 *as_pairs =
              reinterpret_cast<const __nv_bfloat162 *>(&chunk);
          __nv_bfloat162 out_pairs[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float s0 = __bfloat162float(as_pairs[j].x);
            const float s1 = __bfloat162float(as_pairs[j].y);
            const size_t kd0 = i * 8 + 2 * j;
            const float k0 = shared_k[kd0] * k_norm;
            const float k1 = shared_k[kd0 + 1] * k_norm;
            const float new0 = s0 * decay + k0 * delta;
            const float new1 = s1 * decay + k1 * delta;
            out_pairs[j] = __halves2bfloat162(__float2bfloat16(new0),
                                               __float2bfloat16(new1));
          }
          uint4 packed;
          memcpy(&packed, out_pairs, sizeof(packed));
          state_vec_out[i] = packed;
        }
      } else {
        for (size_t kd = 0; kd < key_dim; ++kd) {
          const float decayed = __bfloat162float(state_row[kd]) * decay;
          const float key = shared_k[kd] * k_norm;
          state_row[kd] = __float2bfloat16(decayed + key * delta);
        }
      }

      if (tok + 1 < tokens) {
        __syncthreads();
      }
      continue;
    }

    // Non-gated path (used by smoke tests / reference). Same shared q/k
    // cache; correctness mirrors the reference. We need an explicit sync
    // here because the gated path borrowed the reduction's first barrier to
    // publish the cooperative load.
    __syncthreads();
    if (!active) {
      if (tok + 1 < tokens) {
        __syncthreads();
      }
      continue;
    }
    float acc = 0.0f;
    for (size_t kd = 0; kd < key_dim; ++kd) {
      const float previous = __bfloat162float(state_row[kd]);
      const float key = shared_k[kd];
      const float updated =
          previous * state_decay + update_scale * v_value * key;
      state_row[kd] = __float2bfloat16(updated);
      acc += updated * shared_q[kd];
    }
    output[(tok * shape.v_heads + vh) * shape.value_dim + vd] =
        __float2bfloat16(acc);
    if (tok + 1 < tokens) {
      __syncthreads();
    }
  }
}

} // namespace

extern "C" int
qwen36_deltanet_decode(const qwen36_deltanet_decode_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0 ||
      spec->v_bf16.ptr == 0 || spec->state_bf16.ptr == 0 ||
      spec->output_bf16.ptr == 0 || spec->shape.qk_heads == 0 ||
      spec->shape.v_heads == 0 || spec->shape.key_dim == 0 ||
      spec->shape.value_dim == 0 || spec->tokens_in_persistent_loop == 0 ||
      spec->shape.v_heads % spec->shape.qk_heads != 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  if ((spec->gate_f32.ptr == 0) != (spec->beta_f32.ptr == 0)) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  const int threads = 128;
  const dim3 grid(
      static_cast<unsigned int>((spec->shape.value_dim + threads - 1) / threads),
      static_cast<unsigned int>(spec->shape.v_heads));
  const float state_decay = spec->state_decay == 0.0f ? 1.0f : spec->state_decay;
  const float update_scale =
      spec->update_scale == 0.0f ? 1.0f : spec->update_scale;
  deltanet_decode_kernel<<<grid, threads, 0, qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->q_bf16),
      ptr<const __nv_bfloat16>(spec->k_bf16),
      ptr<const __nv_bfloat16>(spec->v_bf16),
      ptr<const float>(spec->gate_f32), ptr<const float>(spec->beta_f32),
      ptr<__nv_bfloat16>(spec->state_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16), spec->shape,
      spec->tokens_in_persistent_loop, spec->q_token_stride,
      spec->k_token_stride, spec->v_token_stride, state_decay, update_scale,
      spec->qk_l2norm != 0);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
