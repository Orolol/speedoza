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

// Inline-PTX wrapper for the m16n8k32 INT8 mma atom.
//   D[16×8] (s32) = A[16×32] (s8, row-major) * B[32×8] (s8, col-major) + C[16×8] (s32)
// Per-thread fragment layout (laneid l ∈ 0..31):
//   A: 4 b32 regs (16 packed s8 each, 4 s8 per reg)
//     a0 = A[l/4 + 0, (l%4)*4 + 0..3]
//     a1 = A[l/4 + 8, (l%4)*4 + 0..3]
//     a2 = A[l/4 + 0, 16 + (l%4)*4 + 0..3]
//     a3 = A[l/4 + 8, 16 + (l%4)*4 + 0..3]
//   B: 2 b32 regs (8 packed s8 each)
//     b0 = B[(l%4)*4 + 0..3,      l/4]
//     b1 = B[16 + (l%4)*4 + 0..3, l/4]
//   D: 4 s32 regs
//     d0 = D[l/4 + 0, (l%4)*2 + 0]
//     d1 = D[l/4 + 0, (l%4)*2 + 1]
//     d2 = D[l/4 + 8, (l%4)*2 + 0]
//     d3 = D[l/4 + 8, (l%4)*2 + 1]
__device__ __forceinline__ void
mma_m16n8k32_s8(int32_t (&d)[4], const uint32_t (&a)[4],
                const uint32_t (&b)[2], const int32_t (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
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
  // sm_V doubles as K-staging buffer (we load K bf16 here, compute the
  // per-channel mean, quantise to sm_K_int8, then overwrite with V).
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
  float *sm_k_mean = sm_alpha + kSageM; // [kSageD] per-channel K mean (smooth-K)
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

      // Stage K bf16 into sm_V (temp: we overwrite with V below once smooth-K
      // quant has run).
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
              sage_load_kv_f32(cache_k, kv_cache_dtype, cache_index));
        }
      }
      __syncthreads();

      // Smooth-K: compute per-channel mean of K and subtract before INT8
      // quantisation.  The resulting per-row "Q · K_mean" bias is a constant
      // across all N positions of S, so softmax shift-invariance cancels it
      // and we never need to add it back.
      for (size_t d = threadIdx.x; d < kSageD; d += blockDim.x) {
        float sum = 0.0f;
#pragma unroll
        for (int n = 0; n < kSageN; ++n) {
          sum += __bfloat162float(sm_V[n * kSageD + d]);
        }
        sm_k_mean[d] = sum / static_cast<float>(kSageN);
      }
      __syncthreads();

      // Quantize K (smooth) from the staged bf16 tile → sm_K_int8.
      sage_per_row_quantize<kSageKRowsPerWarp>(
          sm_K_int8, sm_kscale, [&](int row, int d) -> float {
            return __bfloat162float(sm_V[row * kSageD + d]) - sm_k_mean[d];
          });
      __syncthreads();

      // Now load V bf16 over sm_V (BF16 PV path; switches to FP8 in B.2).
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
      __syncthreads();

      // -------------------- S = Q @ K^T (inline PTX m16n8k32 INT8) --------------------
      // Each warp owns 4 N-atoms of 8 cols each, covering its 32-col N-half.
      // Per K-iter (k=32), one mma_m16n8k32_s8 per atom; 8 K-iters total for D=256.
      // s_acc[n_atom][4] holds the accumulator regs across K-iters.
      int32_t s_acc[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          s_acc[nt][r] = 0;
        }
      }
      const int a_row_lo = my_m_row_base + (lane_id >> 2);
      const int a_row_hi = a_row_lo + 8;
      const int ab_col_off = (lane_id & 3) * 4;
#pragma unroll
      for (int k_iter_q = 0; k_iter_q < kSageD / 32; ++k_iter_q) {
        const int k_base_q = k_iter_q * 32;
        uint32_t a[4];
        a[0] = *reinterpret_cast<const uint32_t *>(
            &sm_Q_int8[a_row_lo * kSageD + k_base_q + ab_col_off]);
        a[1] = *reinterpret_cast<const uint32_t *>(
            &sm_Q_int8[a_row_hi * kSageD + k_base_q + ab_col_off]);
        a[2] = *reinterpret_cast<const uint32_t *>(
            &sm_Q_int8[a_row_lo * kSageD + k_base_q + ab_col_off + 16]);
        a[3] = *reinterpret_cast<const uint32_t *>(
            &sm_Q_int8[a_row_hi * kSageD + k_base_q + ab_col_off + 16]);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int n_atom_base = my_n_col_base + nt * 8;
          const int b_n = n_atom_base + (lane_id >> 2);
          uint32_t b[2];
          b[0] = *reinterpret_cast<const uint32_t *>(
              &sm_K_int8[b_n * kSageD + k_base_q + ab_col_off]);
          b[1] = *reinterpret_cast<const uint32_t *>(
              &sm_K_int8[b_n * kSageD + k_base_q + ab_col_off + 16]);
          int32_t c[4];
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            c[r] = s_acc[nt][r];
          }
          int32_t d_out[4];
          mma_m16n8k32_s8(d_out, a, b, c);
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            s_acc[nt][r] = d_out[r];
          }
        }
      }
      // Store the 4 atoms' D-fragments to sm_S as int32 (will be dequant'd to f32 below).
      {
        int32_t *sm_S_i32 = reinterpret_cast<int32_t *>(sm_S);
        const int s_row_lo = my_m_row_base + (lane_id >> 2);
        const int s_row_hi = s_row_lo + 8;
        const int s_col_off = (lane_id & 3) * 2;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int n_atom_base = my_n_col_base + nt * 8;
          const int col_lo = n_atom_base + s_col_off;
          const int col_hi = col_lo + 1;
          sm_S_i32[s_row_lo * kSageN + col_lo] = s_acc[nt][0];
          sm_S_i32[s_row_lo * kSageN + col_hi] = s_acc[nt][1];
          sm_S_i32[s_row_hi * kSageN + col_lo] = s_acc[nt][2];
          sm_S_i32[s_row_hi * kSageN + col_hi] = s_acc[nt][3];
        }
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
  // + sm_k_mean (D f32 = 1 KB) ≈ 70 KB.  Below the 100 KB sm_120a cap.
  const size_t smem_bytes =
      kSageN * kSageD * sizeof(__nv_bfloat16) +
      kSageM * kSageD * sizeof(int8_t) + kSageN * kSageD * sizeof(int8_t) +
      kSageM * kSageN * sizeof(float) +
      kSageM * kSageN * sizeof(__nv_bfloat16) +
      kSageM * sizeof(float) + kSageN * sizeof(float) +
      3 * kSageM * sizeof(float) + kSageD * sizeof(float);
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
