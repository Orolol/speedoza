// SageAttention-style prefill kernel for Qwen3.6 (head_dim=256, GQA 24:4).
//
// Phase B.0: per-token INT8 quantisation of Q and K, INT8 wmma m16n16k16 for
// the S = Q @ K^T matmul, FP32 dequant via the saved per-row scales.
// P · V remains BF16 wmma m16n16k16 (same as the Phase A flash kernel) and
// will switch to FP8 e4m3 with f16 accumulator (the "2++" delta) in B.2.
// Smooth-K (subtract per-channel K mean before quant, add bias back to S) is
// scheduled for B.1.
//
// Why m16n16k16 INT8 (not m16n8k32 inline PTX yet): the wmma C++ API gives
// ~2× the per-mma throughput of BF16 m16n8k16 on sm_120a (838 vs 419 TOPS at
// boost) without the inline-PTX per-thread-fragment fragility.  The full ~4×
// requires m16n8k32 and is a follow-up.
//
// Specialised for the Qwen3.6 hot path:
//   tokens >= 16, no tree-mask, BF16 or FP8 E4M3 KV cache, head_dim=256,
//   q_heads % kv_heads == 0.  The dispatcher in attention.cu falls back to
//   the Phase A flash kernel (or scalar kernels) for any other shape, and
//   for TurboQuant cache dtypes.
//
// Warp split (same as Phase A): warp_id/2 → M-tile (2 tiles of 16 rows),
//                               warp_id%2 → N-half (2 halves of 32 cols).
// For PV, the same warps own the same M-tile but split the D-output instead.

#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

namespace {

constexpr int kSageM = 32;
constexpr int kSageN = 64;
constexpr int kSageD = 256;
constexpr int kSageWarps = 4;
constexpr int kSageThreads = 32 * kSageWarps;
constexpr int kSageQRowsPerWarp = kSageM / kSageWarps; // 8
constexpr int kSageKRowsPerWarp = kSageN / kSageWarps; // 16

constexpr int kKvCacheBf16 = 0;
constexpr int kKvCacheFp8 = 1;

__device__ __forceinline__ float sage_decode_e4m3(uint8_t code) {
  const float sign = (code & 0x80) ? -1.0f : 1.0f;
  const int exponent = (code >> 3) & 0x0f;
  const int mantissa = code & 0x07;
  if (exponent == 0) {
    if (mantissa == 0) {
      return sign * 0.0f;
    }
    return sign * ldexpf(static_cast<float>(mantissa) / 8.0f, -6);
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return sign * 448.0f;
  }
  return sign * ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f,
                       exponent - 7);
}

__device__ __forceinline__ float sage_load_kv_f32(const void *cache,
                                                  int kv_cache_dtype,
                                                  size_t index) {
  if (kv_cache_dtype == kKvCacheFp8) {
    return sage_decode_e4m3(reinterpret_cast<const uint8_t *>(cache)[index]);
  }
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(cache)[index]);
}

// Per-row INT8 quantisation: each warp owns `rows_per_warp` rows.  For each
// row, the 32 lanes scan D=256 cols (8 elements per lane), compute the row
// abs-max via shfl_xor reduction, write the scale to `sm_scale[row]`, then
// write 8 int8 elements per lane back to the destination buffer.
template <int kRowsPerWarp, typename SrcLoad>
__device__ __forceinline__ void
sage_per_row_quantize(int8_t *dst, float *sm_scale, SrcLoad src_load) {
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
#pragma unroll
  for (int row_local = 0; row_local < kRowsPerWarp; ++row_local) {
    const int row = warp_id * kRowsPerWarp + row_local;
    float vals[8];
    float row_max = 0.0f;
#pragma unroll
    for (int d_local = 0; d_local < 8; ++d_local) {
      const int d = lane_id * 8 + d_local;
      vals[d_local] = src_load(row, d);
      row_max = fmaxf(row_max, fabsf(vals[d_local]));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, offset));
    }
    const float scale = row_max / 127.0f;
    const float inv_scale = (row_max > 0.0f) ? (127.0f / row_max) : 0.0f;
    if (lane_id == 0) {
      sm_scale[row] = scale;
    }
#pragma unroll
    for (int d_local = 0; d_local < 8; ++d_local) {
      const int d = lane_id * 8 + d_local;
      const float q = vals[d_local] * inv_scale;
      const int q_int = __float2int_rn(fmaxf(-127.0f, fminf(127.0f, q)));
      dst[row * kSageD + d] = static_cast<int8_t>(q_int);
    }
  }
}

__global__ void
attention_sage_prefill_kernel(const __nv_bfloat16 *q, const void *cache_k,
                              const void *cache_v, int kv_cache_dtype,
                              __nv_bfloat16 *output, size_t start_position,
                              const int32_t *start_position_device,
                              size_t tokens, qwen36_attention_shape_t shape) {
  const size_t kvh = blockIdx.x;
  const size_t q_tile_idx = blockIdx.y;
  const size_t token_base = q_tile_idx * kSageM;
  if (token_base >= tokens) {
    return;
  }
  const size_t q_per_kv = shape.q_heads / shape.kv_heads;
  const size_t head_dim = shape.head_dim;
  const float qk_scale = rsqrtf(static_cast<float>(head_dim));
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int my_m_tile = warp_id / 2;
  const int my_n_half = warp_id & 1;
  const int my_m_row_base = my_m_tile * 16;
  const int my_n_col_base = my_n_half * 32;
  const int my_d_tile_base = my_n_half * 8;

  __shared__ size_t shared_start_position;
  if (threadIdx.x == 0) {
    shared_start_position = (start_position_device != nullptr)
                                ? static_cast<size_t>(*start_position_device)
                                : start_position;
  }
  __syncthreads();
  const size_t start_pos = shared_start_position;
  const size_t kv_total = start_pos + tokens;
  const size_t rows_in_tile =
      (token_base + kSageM <= tokens) ? kSageM : (tokens - token_base);
  const size_t max_kv_visible = start_pos + token_base + rows_in_tile - 1;

  extern __shared__ unsigned char smem_raw[];
  __nv_bfloat16 *sm_V = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  int8_t *sm_Q_int8 = reinterpret_cast<int8_t *>(sm_V + kSageN * kSageD);
  int8_t *sm_K_int8 = sm_Q_int8 + kSageM * kSageD;
  float *sm_S = reinterpret_cast<float *>(sm_K_int8 + kSageN * kSageD);
  __nv_bfloat16 *sm_P =
      reinterpret_cast<__nv_bfloat16 *>(sm_S + kSageM * kSageN);
  float *sm_qscale = reinterpret_cast<float *>(sm_P + kSageM * kSageN);
  float *sm_kscale = sm_qscale + kSageM;
  float *sm_m = sm_kscale + kSageN;
  float *sm_l = sm_m + kSageM;
  float *sm_alpha = sm_l + kSageM;
  // After the K-iter loop we reuse the sm_V region (32 KB) as f32 O scratch.

  for (size_t qh_local = 0; qh_local < q_per_kv; ++qh_local) {
    const size_t qh = kvh * q_per_kv + qh_local;

    // ---- quantize Q to int8 with per-row scale ----
    sage_per_row_quantize<kSageQRowsPerWarp>(
        sm_Q_int8, sm_qscale, [&](int row, int d) -> float {
          if (row >= static_cast<int>(rows_in_tile)) {
            return 0.0f;
          }
          return __bfloat162float(
              q[((token_base + row) * shape.q_heads + qh) * head_dim + d]);
        });

    // ---- init row state ----
    if (threadIdx.x < kSageM) {
      sm_m[threadIdx.x] = -INFINITY;
      sm_l[threadIdx.x] = 0.0f;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frags[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      wmma::fill_fragment(o_frags[i], 0.0f);
    }
    __syncthreads();

    const size_t k_iters = (max_kv_visible + kSageN) / kSageN;
    for (size_t k_iter = 0; k_iter < k_iters; ++k_iter) {
      const size_t k_base = k_iter * kSageN;
      if (k_base > max_kv_visible) {
        break;
      }

      // Load V bf16 (BF16 PV path; switches to FP8 in B.2).
      for (size_t i = threadIdx.x; i < kSageN * kSageD; i += blockDim.x) {
        const size_t n = i / kSageD;
        const size_t d = i % kSageD;
        const size_t kv_idx = k_base + n;
        if (kv_idx >= kv_total) {
          sm_V[i] = __float2bfloat16(0.0f);
        } else {
          const size_t cache_index =
              (kv_idx * shape.kv_heads + kvh) * head_dim + d;
          sm_V[i] = __float2bfloat16(
              sage_load_kv_f32(cache_v, kv_cache_dtype, cache_index));
        }
      }
      // Quantize K to int8 directly from cache (avoid an intermediate bf16
      // tile — keeps the SMEM budget under sm_120a's 100 KB cap).
      sage_per_row_quantize<kSageKRowsPerWarp>(
          sm_K_int8, sm_kscale, [&](int row, int d) -> float {
            const size_t kv_idx = k_base + row;
            if (kv_idx >= kv_total) {
              return 0.0f;
            }
            const size_t cache_index =
                (kv_idx * shape.kv_heads + kvh) * head_dim + d;
            return sage_load_kv_f32(cache_k, kv_cache_dtype, cache_index);
          });
      __syncthreads();

      // -------------------- S = Q @ K^T (INT8 wmma m16n16k16) --------------------
#pragma unroll
      for (int n_in_half = 0; n_in_half < 2; ++n_in_half) {
        const int n_col = my_n_col_base + n_in_half * 16;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char,
                       wmma::row_major>
            q_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char,
                       wmma::col_major>
            k_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> s_frag;
        wmma::fill_fragment(s_frag, 0);
#pragma unroll
        for (int d = 0; d < kSageD; d += 16) {
          wmma::load_matrix_sync(q_frag,
                                 sm_Q_int8 + my_m_row_base * kSageD + d,
                                 kSageD);
          wmma::load_matrix_sync(k_frag, sm_K_int8 + n_col * kSageD + d,
                                 kSageD);
          wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
        }
        wmma::store_matrix_sync(
            reinterpret_cast<int32_t *>(sm_S) + my_m_row_base * kSageN + n_col,
            s_frag, kSageN, wmma::mem_row_major);
      }
      __syncthreads();

      // -------------------- dequant + causal mask + qk_scale --------------------
      for (size_t i = threadIdx.x; i < kSageM * kSageN; i += blockDim.x) {
        const size_t r = i / kSageN;
        const size_t c = i % kSageN;
        const size_t row_pos = start_pos + token_base + r;
        const size_t col_pos = k_base + c;
        if (r >= rows_in_tile || col_pos > row_pos || col_pos >= kv_total) {
          sm_S[i] = -INFINITY;
        } else {
          const int32_t s_i32 = reinterpret_cast<int32_t *>(sm_S)[i];
          sm_S[i] = static_cast<float>(s_i32) * sm_qscale[r] * sm_kscale[c] *
                    qk_scale;
        }
      }
      __syncthreads();

      // -------------------- online softmax row update --------------------
      if (threadIdx.x < rows_in_tile) {
        const int r = threadIdx.x;
        float row_max = -INFINITY;
#pragma unroll
        for (int c = 0; c < kSageN; ++c) {
          row_max = fmaxf(row_max, sm_S[r * kSageN + c]);
        }
        const float m_old = sm_m[r];
        const float m_new = fmaxf(m_old, row_max);
        const float alpha =
            (isinf(m_old) && m_old < 0.0f) ? 0.0f : expf(m_old - m_new);
        const bool row_all_masked = isinf(m_new) && m_new < 0.0f;
        float row_sum_p = 0.0f;
#pragma unroll
        for (int c = 0; c < kSageN; ++c) {
          float p = 0.0f;
          if (!row_all_masked) {
            p = expf(sm_S[r * kSageN + c] - m_new);
          }
          sm_P[r * kSageN + c] = __float2bfloat16(p);
          row_sum_p += p;
        }
        sm_alpha[r] = alpha;
        sm_l[r] = alpha * sm_l[r] + row_sum_p;
        sm_m[r] = m_new;
      } else if (threadIdx.x < kSageM) {
        sm_alpha[threadIdx.x] = 0.0f;
#pragma unroll
        for (int c = 0; c < kSageN; ++c) {
          sm_P[threadIdx.x * kSageN + c] = __float2bfloat16(0.0f);
        }
      }
      __syncthreads();

      // -------------------- rescale o_frags by alpha[row] --------------------
      const int row_lo = my_m_row_base + (lane_id >> 2);
      const int row_hi = row_lo + 8;
      const float alpha_lo = (row_lo < kSageM) ? sm_alpha[row_lo] : 0.0f;
      const float alpha_hi = (row_hi < kSageM) ? sm_alpha[row_hi] : 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        o_frags[i].x[0] *= alpha_lo;
        o_frags[i].x[1] *= alpha_lo;
        o_frags[i].x[2] *= alpha_hi;
        o_frags[i].x[3] *= alpha_hi;
        o_frags[i].x[4] *= alpha_lo;
        o_frags[i].x[5] *= alpha_lo;
        o_frags[i].x[6] *= alpha_hi;
        o_frags[i].x[7] *= alpha_hi;
      }

      // -------------------- O += P @ V (BF16, unchanged from Phase A) --------------------
#pragma unroll
      for (int d_local = 0; d_local < 8; ++d_local) {
        const int d_tile = my_d_tile_base + d_local;
        const int d_col = d_tile * 16;
#pragma unroll
        for (int n_frag = 0; n_frag < kSageN / 16; ++n_frag) {
          const int n_offset = n_frag * 16;
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                         wmma::row_major>
              p_frag;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                         wmma::row_major>
              v_frag;
          wmma::load_matrix_sync(
              p_frag, sm_P + my_m_row_base * kSageN + n_offset, kSageN);
          wmma::load_matrix_sync(v_frag, sm_V + n_offset * kSageD + d_col,
                                 kSageD);
          wmma::mma_sync(o_frags[d_local], p_frag, v_frag, o_frags[d_local]);
        }
      }
      __syncthreads();
    }

    // Reuse sm_V as f32 O scratch (32 KB → 8K floats = 32×256 f32).
    float *sm_O_scratch = reinterpret_cast<float *>(sm_V);
    __syncthreads();
#pragma unroll
    for (int d_local = 0; d_local < 8; ++d_local) {
      const int d_tile = my_d_tile_base + d_local;
      const int d_col = d_tile * 16;
      float *tile_base = sm_O_scratch + my_m_tile * 16 * kSageD;
      wmma::store_matrix_sync(tile_base + d_col, o_frags[d_local], kSageD,
                              wmma::mem_row_major);
    }
    __syncthreads();
    for (size_t i = threadIdx.x; i < kSageM * kSageD; i += blockDim.x) {
      const size_t r = i / kSageD;
      const size_t d = i % kSageD;
      if (r >= rows_in_tile) {
        continue;
      }
      const float l = sm_l[r];
      const float val = (l > 0.0f) ? (sm_O_scratch[i] / l) : 0.0f;
      output[((token_base + r) * shape.q_heads + qh) * head_dim + d] =
          __float2bfloat16(val);
    }
    __syncthreads();
  }
}

} // namespace

extern "C" int
qwen36_attention_sage_prefill(const qwen36_attention_prefill_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->tokens == 0 || spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0 ||
      spec->v_bf16.ptr == 0 || spec->kv_cache_k.ptr == 0 ||
      spec->kv_cache_v.ptr == 0 || spec->output_bf16.ptr == 0 ||
      spec->shape.head_dim != kSageD || spec->shape.q_heads == 0 ||
      spec->shape.kv_heads == 0 ||
      spec->shape.q_heads % spec->shape.kv_heads != 0 ||
      (spec->kv_cache_dtype != kKvCacheBf16 &&
       spec->kv_cache_dtype != kKvCacheFp8)) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

  // sm_V (32 KB) + sm_Q_int8 (8 KB) + sm_K_int8 (16 KB) + sm_S (8 KB)
  // + sm_P (4 KB) + sm_qscale (128 B) + sm_kscale (256 B) + 3·M f32 row state
  // ≈ 69 KB.  Comfortably below the 100 KB sm_120a per-block cap.
  const size_t smem_bytes =
      kSageN * kSageD * sizeof(__nv_bfloat16) +
      kSageM * kSageD * sizeof(int8_t) + kSageN * kSageD * sizeof(int8_t) +
      kSageM * kSageN * sizeof(float) +
      kSageM * kSageN * sizeof(__nv_bfloat16) +
      kSageM * sizeof(float) + kSageN * sizeof(float) +
      3 * kSageM * sizeof(float);
  cudaFuncSetAttribute(attention_sage_prefill_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem_bytes));

  const dim3 grid(
      static_cast<unsigned int>(spec->shape.kv_heads),
      static_cast<unsigned int>((spec->tokens + kSageM - 1) / kSageM));
  attention_sage_prefill_kernel<<<grid, kSageThreads, smem_bytes,
                                  qwen36_internal_active_stream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<uintptr_t>(spec->q_bf16.ptr)),
      reinterpret_cast<const void *>(
          static_cast<uintptr_t>(spec->kv_cache_k.ptr)),
      reinterpret_cast<const void *>(
          static_cast<uintptr_t>(spec->kv_cache_v.ptr)),
      spec->kv_cache_dtype,
      reinterpret_cast<__nv_bfloat16 *>(
          static_cast<uintptr_t>(spec->output_bf16.ptr)),
      spec->start_position,
      spec->start_position_device_i32.ptr != 0
          ? reinterpret_cast<const int32_t *>(
                static_cast<uintptr_t>(spec->start_position_device_i32.ptr))
          : nullptr,
      spec->tokens, spec->shape);

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
