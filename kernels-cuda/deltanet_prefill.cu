#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

template <typename T> T *ptr(qwen36_device_ptr_t value) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(value.ptr));
}

// Gated DeltaNet chunked-parallel prefill kernel — WY-representation form
// from Yang et al. (arXiv:2412.06464), validated in /tmp/gdn_ref/ against
// flash-linear-attention's `naive_chunk_gated_delta_rule`.
//
// Per chunk of length C, per v_head:
//   D[c]   = cumsum_c(g[c])
//   L[i,j] = exp(D[i]-D[j]) for j<=i
//   M[i,j] = -(k_β[i]·k[j]) * L[i,j] for j<i (strict lower)
//   A      = (I - M)^{-1} = I + M + M² + … (forward sub)
//   u_new  = A @ (v*β)
//   w_dec  = A @ (k*β*exp(D))
//   v_new  = u_new - w_dec @ S_prev
//   attn   = (q@k^T) * L, strict-upper masked
//   o      = (q*exp(D)) @ S_prev + attn @ v_new
//   S_next = exp(D[-1])*S_prev + (k * exp(D[-1]-D))^T @ v_new
//
// v1 design (no tensor cores, plain CUDA FMAs):
//   - Grid:  (v_heads, 1, 1)  — each block owns one v_head
//   - Block: 128 threads — sized to cover [K=128, V=128] cells one-per-thread
//   - Chunk size fixed at C=32 (validated at entry)
//   - State S kept resident in shared memory in BF16 across all chunks
//   - One sync barrier between phases inside a chunk
//
// Memory budget per block (~84 KB shared, requires opt-in dyn SMEM on SM_120):
//   S [K*V] bf16            32768 B
//   q [C*K] bf16             8192 B
//   k [C*K] bf16             8192 B
//   v [C*V] bf16             8192 B  (reused as o output buffer after step 10)
//   A [C*C] f32              4096 B
//   attn_tri [C*C] f32       4096 B
//   v_new [C*V] bf16         8192 B
//   w_dec [C*K] bf16         8192 B
//   scratch [C] x 4 f32       512 B  (g, D, β, exp(D))
//   ----                   ------
//                          ~83 KB

constexpr int kPrefillChunk = 32;
constexpr int kPrefillKeyDim = 128;
constexpr int kPrefillValDim = 128;
constexpr int kPrefillThreads = 128;
constexpr int kPrefillQkHeads = 16;
constexpr int kPrefillVHeads = 48;

__device__ __forceinline__ float bf2f(const __nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f2bf(float v) {
  return __float2bfloat16(v);
}

__global__ void __launch_bounds__(kPrefillThreads, 1)
deltanet_prefill_kernel(
    const __nv_bfloat16 *__restrict__ q,         // [T, qk_heads, K]
    const __nv_bfloat16 *__restrict__ k,         // [T, qk_heads, K]
    const __nv_bfloat16 *__restrict__ v,         // [T, v_heads,  V]
    const float *__restrict__ gate,              // [T, v_heads]   log-decay g
    const float *__restrict__ beta,              // [T, v_heads]
    __nv_bfloat16 *__restrict__ state,           // [v_heads, V, K] in/out
                                                 // (canonical layout shared with
                                                 // qwen36_deltanet_decode; SMEM
                                                 // copy is [K, V])
    __nv_bfloat16 *__restrict__ output,          // [T, v_heads, V]
    qwen36_deltanet_shape_t shape,
    size_t tokens,
    size_t q_token_stride,
    size_t k_token_stride,
    size_t v_token_stride,
    float state_decay,
    float update_scale,
    bool qk_l2norm) {
  constexpr int C = kPrefillChunk;
  constexpr int K = kPrefillKeyDim;
  constexpr int V = kPrefillValDim;
  const int tid = threadIdx.x;
  const int vh = blockIdx.x;
  const int v_heads = static_cast<int>(shape.v_heads);
  const int qk_heads = static_cast<int>(shape.qk_heads);
  const int q_repeat = v_heads / qk_heads;
  const int qh = vh / q_repeat;
  const float inv_sqrt_k = rsqrtf(static_cast<float>(K));

  // Dynamic shared memory layout (opt-in via cudaFuncSetAttribute).
  extern __shared__ unsigned char smem[];
  unsigned char *cursor = smem;

  __nv_bfloat16 *sm_S = reinterpret_cast<__nv_bfloat16 *>(cursor);
  cursor += K * V * sizeof(__nv_bfloat16);
  __nv_bfloat16 *sm_q = reinterpret_cast<__nv_bfloat16 *>(cursor);
  cursor += C * K * sizeof(__nv_bfloat16);
  __nv_bfloat16 *sm_k = reinterpret_cast<__nv_bfloat16 *>(cursor);
  cursor += C * K * sizeof(__nv_bfloat16);
  __nv_bfloat16 *sm_v = reinterpret_cast<__nv_bfloat16 *>(cursor);
  cursor += C * V * sizeof(__nv_bfloat16);
  float *sm_A = reinterpret_cast<float *>(cursor);
  cursor += C * C * sizeof(float);
  float *sm_attn = reinterpret_cast<float *>(cursor);
  cursor += C * C * sizeof(float);
  __nv_bfloat16 *sm_vnew = reinterpret_cast<__nv_bfloat16 *>(cursor);
  cursor += C * V * sizeof(__nv_bfloat16);
  __nv_bfloat16 *sm_wdec = reinterpret_cast<__nv_bfloat16 *>(cursor);
  cursor += C * K * sizeof(__nv_bfloat16);
  float *sm_g = reinterpret_cast<float *>(cursor);
  cursor += C * sizeof(float);
  float *sm_D = reinterpret_cast<float *>(cursor);
  cursor += C * sizeof(float);
  float *sm_beta = reinterpret_cast<float *>(cursor);
  cursor += C * sizeof(float);
  float *sm_expD = reinterpret_cast<float *>(cursor);
  cursor += C * sizeof(float);
  float *sm_coef = reinterpret_cast<float *>(cursor);
  cursor += C * sizeof(float);
  // Per-warp wmma scratch for store-and-combine.  4 warps × 16×16 f32 = 4 KB.
  float *sm_wmma_scratch = reinterpret_cast<float *>(cursor);
  cursor += 4 * 16 * 16 * sizeof(float);
  // BF16 view of sm_A after forward-sub completes (needed as matrix_a for wmma
  // in phases 7 and 8).  2 KB.
  __nv_bfloat16 *sm_A_bf16 = reinterpret_cast<__nv_bfloat16 *>(cursor);
  cursor += C * C * sizeof(__nv_bfloat16);

  // --- Load initial state for this v_head from global to SMEM (BF16) ---
  // Global layout is [V, K] (the canonical layout written and read by the
  // sequential qwen36_deltanet_decode kernel); the SMEM working copy is
  // [K, V] (what the wmma phases consume). Transpose on the way in. Global
  // reads stay coalesced; the strided SMEM writes are a one-off per call.
  {
    const __nv_bfloat16 *state_in = state + (size_t)vh * K * V;
    const int total = K * V;
    for (int idx = tid; idx < total; idx += blockDim.x) {
      const int vd = idx / K;
      const int kd = idx % K;
      sm_S[kd * V + vd] = state_in[idx];
    }
  }
  __syncthreads();

  const int num_chunks = static_cast<int>((tokens + C - 1) / C);
  for (int n = 0; n < num_chunks; ++n) {
    const int chunk_start = n * C;
    const int valid = max(0, min(C, static_cast<int>(tokens) - chunk_start));

    // === Phase 1: Load q, k, v, g, β for this chunk into SMEM ===
    // q/k row stride = q_token_stride (in BF16 elements). qh selects which of the
    // qk_heads is broadcast across this v_head's q_repeat=3 group.
    for (int row = 0; row < C; ++row) {
      const int tok = chunk_start + row;
      const bool live = (tok < (int)tokens);
      const __nv_bfloat16 *q_row =
          live ? q + tok * q_token_stride + qh * K : nullptr;
      const __nv_bfloat16 *k_row =
          live ? k + tok * k_token_stride + qh * K : nullptr;
      for (int d = tid; d < K; d += blockDim.x) {
        sm_q[row * K + d] = live ? q_row[d] : __float2bfloat16(0.0f);
        sm_k[row * K + d] = live ? k_row[d] : __float2bfloat16(0.0f);
      }
      const __nv_bfloat16 *v_row =
          live ? v + tok * v_token_stride + vh * V : nullptr;
      for (int d = tid; d < V; d += blockDim.x) {
        sm_v[row * V + d] = live ? v_row[d] : __float2bfloat16(0.0f);
      }
    }
    if (tid < C) {
      const int tok = chunk_start + tid;
      const bool live = (tok < (int)tokens);
      sm_g[tid] = live ? gate[tok * v_heads + vh] * state_decay : 0.0f;
      sm_beta[tid] = live ? beta[tok * v_heads + vh] * update_scale : 0.0f;
    }
    __syncthreads();

    // === Phase 2: Cumulative log-decay D and exp(D) ===
    if (tid == 0) {
      float acc = 0.0f;
      for (int c = 0; c < C; ++c) {
        acc += sm_g[c];
        sm_D[c] = acc;
        sm_expD[c] = __expf(acc);
      }
    }
    __syncthreads();
    const float expD_last = sm_expD[C - 1];
    // Precompute coef[i] = exp(D_last - D[i]) — used in the S_next update.
    // Saves ~32 expf per output cell (16K cells × 32 = 524K expf per chunk).
    if (tid < C) {
      sm_coef[tid] = __expf(sm_D[C - 1] - sm_D[tid]);
    }
    __syncthreads();

    // === Phase 3: L2-normalize q, k rows (if requested), then q *= 1/sqrt(K) ===
    // 4 threads per row × 32 rows = 128 lanes (one per row pair (q-row, k-row)
    // or one per row each).  Each warp owns 8 rows; lanes do tree-reduce within
    // the 128-wide row.
    {
      // Row layout: 128 threads handle 32 rows of q + 32 rows of k = 64 streams.
      // Two threads per stream (K=128 / 64? no…).  Simpler: each thread owns one
      // (row, stream_kind) cell, reduces the entire K=128 row sequentially.
      // 128 threads = 64 streams ×  2 — but we only have 64 streams (32 q + 32 k).
      // Reduce serialization: 2 threads share the same row reduction, doing half each.
      const int stream = tid >> 1;       // 0..63
      const int half   = tid & 1;        // 0 or 1
      const bool is_q  = (stream < C);
      const int row    = is_q ? stream : (stream - C);
      const __nv_bfloat16 *row_src = is_q ? &sm_q[row * K] : &sm_k[row * K];

      float partial = 0.0f;
      if (qk_l2norm) {
        const int d_start = half * (K / 2);
        const int d_end = d_start + (K / 2);
        for (int d = d_start; d < d_end; ++d) {
          const float x = bf2f(row_src[d]);
          partial += x * x;
        }
      }
      __shared__ float row_sumsq[2 * C];
      if (qk_l2norm) {
        // Combine halves via simple atomicAdd-free pair using shared.
        if (half == 0) row_sumsq[stream] = partial;
        // Sync needed before half 1 writes; use a tiny atomic-free combine via
        // shared-array pair indices.
      }
      __syncthreads();
      if (qk_l2norm && half == 1) {
        row_sumsq[stream] += partial;
      }
      __syncthreads();

      // Apply normalization and scaling.
      __nv_bfloat16 *row_dst = is_q ? &sm_q[row * K] : &sm_k[row * K];
      const float norm = qk_l2norm
          ? rsqrtf(row_sumsq[stream] + 1e-6f)
          : 1.0f;
      const float scale = is_q ? (norm * inv_sqrt_k) : norm;
      const int d_start = half * (K / 2);
      const int d_end = d_start + (K / 2);
      for (int d = d_start; d < d_end; ++d) {
        row_dst[d] = f2bf(bf2f(row_dst[d]) * scale);
      }
    }
    __syncthreads();

    // === Phase 4: attn[i,j] = (q[i]·k[j]) * exp(D[i]-D[j]) for j<=i, 0 else ===
    // wmma: q @ k^T with K=128 reduction.  Output [C=32, C=32] = 4 tiles; 1 per warp.
    // k stored [C, K] row-major; reading as col_major gives k^T natively (ldm=K).
    {
      using namespace nvcuda::wmma;
      const int warp_id = tid / 32;
      const int lane    = tid % 32;
      float *warp_scratch = sm_wmma_scratch + warp_id * 16 * 16;
      const int m_tile = warp_id / 2;   // 0..1
      const int n_tile = warp_id % 2;   // 0..1

      fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
      fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
      fragment<accumulator, 16, 16, 16, float> c_frag;
      fill_fragment(c_frag, 0.0f);

      #pragma unroll
      for (int k_tile = 0; k_tile < K / 16; ++k_tile) {
        load_matrix_sync(a_frag, sm_q + m_tile * 16 * K + k_tile * 16, K);
        load_matrix_sync(b_frag, sm_k + n_tile * 16 * K + k_tile * 16, K);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      store_matrix_sync(warp_scratch, c_frag, 16, mem_row_major);
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int idx_in_tile = lane * 8 + e;
        const int row_in_tile = idx_in_tile / 16;
        const int col_in_tile = idx_in_tile % 16;
        const int i = m_tile * 16 + row_in_tile;
        const int j = n_tile * 16 + col_in_tile;
        if (j > i) {
          sm_attn[i * C + j] = 0.0f;
        } else {
          const float qk = warp_scratch[row_in_tile * 16 + col_in_tile];
          sm_attn[i * C + j] = qk * __expf(sm_D[i] - sm_D[j]);
        }
      }
    }
    __syncthreads();

    // === Phase 5a: materialize k_beta = k * β into sm_wdec (temporary) ===
    // sm_wdec is free until Phase 8 (which overwrites it with w_dec).
    {
      for (int idx = tid; idx < C * K; idx += blockDim.x) {
        const int i = idx / K;
        sm_wdec[idx] = f2bf(bf2f(sm_k[idx]) * sm_beta[i]);
      }
    }
    __syncthreads();

    // === Phase 5b: M[i,j] = -(k_β[i]·k[j]) * exp(D[i]-D[j]) for j<i, 0 else ===
    // wmma: k_beta @ k^T. Same tile layout as Phase 4. Stored into sm_A for in-place forward-sub.
    {
      using namespace nvcuda::wmma;
      const int warp_id = tid / 32;
      const int lane    = tid % 32;
      float *warp_scratch = sm_wmma_scratch + warp_id * 16 * 16;
      const int m_tile = warp_id / 2;
      const int n_tile = warp_id % 2;

      fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
      fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
      fragment<accumulator, 16, 16, 16, float> c_frag;
      fill_fragment(c_frag, 0.0f);

      #pragma unroll
      for (int k_tile = 0; k_tile < K / 16; ++k_tile) {
        load_matrix_sync(a_frag, sm_wdec + m_tile * 16 * K + k_tile * 16, K);
        load_matrix_sync(b_frag, sm_k    + n_tile * 16 * K + k_tile * 16, K);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      store_matrix_sync(warp_scratch, c_frag, 16, mem_row_major);
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int idx_in_tile = lane * 8 + e;
        const int row_in_tile = idx_in_tile / 16;
        const int col_in_tile = idx_in_tile % 16;
        const int i = m_tile * 16 + row_in_tile;
        const int j = n_tile * 16 + col_in_tile;
        if (j >= i) {
          sm_A[i * C + j] = 0.0f;
        } else {
          const float dot = warp_scratch[row_in_tile * 16 + col_in_tile];
          sm_A[i * C + j] = -dot * __expf(sm_D[i] - sm_D[j]);
        }
      }
    }
    __syncthreads();

    // === Phase 6: Forward-sub:  A[i, :i] += sum_{l<i} M[i,l] * A[l, :i] ===
    // Sequential along i.  At iteration i, threads with j<i update A[i,j].
    // No thread writes a value that another thread reads inside one i (each
    // thread writes only A[i,j]; reads are A[i,l] for l!=j and A[l,j] for l<i).
    for (int i = 1; i < C; ++i) {
      float acc = 0.0f;
      if (tid < i) {
        const int j = tid;
        acc = sm_A[i * C + j]; // M[i,j]
        for (int l = 0; l < i; ++l) {
          acc += sm_A[i * C + l] * sm_A[l * C + j];
        }
      }
      __syncthreads();
      if (tid < i) {
        sm_A[i * C + tid] = acc;
      }
      __syncthreads();
    }
    // Add identity to make A = I + (M+M²+…).
    if (tid < C) {
      sm_A[tid * C + tid] += 1.0f;
    }
    __syncthreads();

    // Convert sm_A (f32) → sm_A_bf16 for wmma matrix_a use in Phases 7 and 8.
    {
      for (int idx = tid; idx < C * C; idx += blockDim.x) {
        sm_A_bf16[idx] = f2bf(sm_A[idx]);
      }
    }
    __syncthreads();

    // === Phase 7a: v_pre = v * β in-place in sm_v ===
    // After Phase 7b, sm_v no longer holds v; we'll repurpose it for Phase 8a.
    {
      for (int idx = tid; idx < C * V; idx += blockDim.x) {
        const int i = idx / V;
        sm_v[idx] = f2bf(bf2f(sm_v[idx]) * sm_beta[i]);
      }
    }
    __syncthreads();

    // === Phase 7b: u_new = A @ v_pre via wmma ===
    // [C×C] @ [C×V] → [C×V]. M_tiles=2, N_tiles=8, K_tiles=2. 16 tiles, 4/warp.
    {
      using namespace nvcuda::wmma;
      const int warp_id = tid / 32;
      const int lane    = tid % 32;
      float *warp_scratch = sm_wmma_scratch + warp_id * 16 * 16;

      for (int tile_id = warp_id * 4; tile_id < (warp_id + 1) * 4; ++tile_id) {
        const int m_tile = tile_id / 8;
        const int n_tile = tile_id % 8;

        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        #pragma unroll
        for (int k_tile = 0; k_tile < C / 16; ++k_tile) {  // C=32 / 16 = 2
          load_matrix_sync(a_frag, sm_A_bf16 + m_tile * 16 * C + k_tile * 16, C);
          load_matrix_sync(b_frag, sm_v      + k_tile * 16 * V + n_tile * 16, V);
          mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        store_matrix_sync(warp_scratch, c_frag, 16, mem_row_major);
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int idx_in_tile = lane * 8 + e;
          const int row_in_tile = idx_in_tile / 16;
          const int col_in_tile = idx_in_tile % 16;
          const int i  = m_tile * 16 + row_in_tile;
          const int vd = n_tile * 16 + col_in_tile;
          sm_vnew[i * V + vd] = f2bf(warp_scratch[row_in_tile * 16 + col_in_tile]);
        }
      }
    }
    __syncthreads();

    // === Phase 8a: scaled_k = k * β * exp(D) into sm_v (overwriting v_pre) ===
    {
      for (int idx = tid; idx < C * K; idx += blockDim.x) {
        const int i = idx / K;
        const int kd = idx % K;
        sm_v[idx] = f2bf(bf2f(sm_k[i * K + kd]) * sm_beta[i] * sm_expD[i]);
      }
    }
    __syncthreads();

    // === Phase 8b: w_dec = A @ scaled_k via wmma → sm_wdec ===
    // [C×C] @ [C×K] → [C×K]. Same tile structure as Phase 7b but N dim = K.
    {
      using namespace nvcuda::wmma;
      const int warp_id = tid / 32;
      const int lane    = tid % 32;
      float *warp_scratch = sm_wmma_scratch + warp_id * 16 * 16;

      for (int tile_id = warp_id * 4; tile_id < (warp_id + 1) * 4; ++tile_id) {
        const int m_tile = tile_id / 8;
        const int n_tile = tile_id % 8;

        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        #pragma unroll
        for (int k_tile = 0; k_tile < C / 16; ++k_tile) {
          load_matrix_sync(a_frag, sm_A_bf16 + m_tile * 16 * C + k_tile * 16, C);
          load_matrix_sync(b_frag, sm_v      + k_tile * 16 * K + n_tile * 16, K);
          mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        store_matrix_sync(warp_scratch, c_frag, 16, mem_row_major);
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int idx_in_tile = lane * 8 + e;
          const int row_in_tile = idx_in_tile / 16;
          const int col_in_tile = idx_in_tile % 16;
          const int i  = m_tile * 16 + row_in_tile;
          const int kd = n_tile * 16 + col_in_tile;
          sm_wdec[i * K + kd] = f2bf(warp_scratch[row_in_tile * 16 + col_in_tile]);
        }
      }
    }
    __syncthreads();

    // === Phase 9: v_new = u_new - (w_dec @ S_prev) via wmma ===
    // Output [C=32, V=128] = 2 M_tiles × 8 N_tiles = 16 tiles; 4 warps × 4 tiles.
    {
      using namespace nvcuda::wmma;
      const int warp_id = tid / 32;
      const int lane    = tid % 32;
      float *warp_scratch = sm_wmma_scratch + warp_id * 16 * 16;

      for (int tile_id = warp_id * 4; tile_id < (warp_id + 1) * 4; ++tile_id) {
        const int m_tile = tile_id / 8;   // 0..1 (C/16)
        const int n_tile = tile_id % 8;   // 0..7 (V/16)

        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        #pragma unroll
        for (int k_tile = 0; k_tile < K / 16; ++k_tile) {  // K=128 / 16 = 8
          load_matrix_sync(a_frag, sm_wdec + m_tile * 16 * K + k_tile * 16, K);
          load_matrix_sync(b_frag, sm_S    + k_tile * 16 * V + n_tile * 16, V);
          mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        store_matrix_sync(warp_scratch, c_frag, 16, mem_row_major);
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int idx_in_tile = lane * 8 + e;
          const int row_in_tile = idx_in_tile / 16;
          const int col_in_tile = idx_in_tile % 16;
          const int i  = m_tile * 16 + row_in_tile;
          const int vd = n_tile * 16 + col_in_tile;
          const float vprime = warp_scratch[row_in_tile * 16 + col_in_tile];
          const float u = bf2f(sm_vnew[i * V + vd]);
          sm_vnew[i * V + vd] = f2bf(u - vprime);
        }
      }
    }
    __syncthreads();

    // === Phase 10a: q_scaled[i, d] = q[i, d] * exp(D[i]) in-place ===
    // q is no longer needed after this chunk's o_inter matmul, so we overwrite.
    {
      for (int idx = tid; idx < C * K; idx += blockDim.x) {
        const int i = idx / K;
        sm_q[idx] = f2bf(bf2f(sm_q[idx]) * sm_expD[i]);
      }
    }
    __syncthreads();

    // === Phase 10b: o_inter = q_scaled @ S_prev via wmma; combine with o_intra ===
    // sm_v slot is free (consumed in Phase 7); we write the final o = o_inter + o_intra.
    {
      using namespace nvcuda::wmma;
      __nv_bfloat16 *sm_out = sm_v;
      const int warp_id = tid / 32;
      const int lane    = tid % 32;
      float *warp_scratch = sm_wmma_scratch + warp_id * 16 * 16;

      for (int tile_id = warp_id * 4; tile_id < (warp_id + 1) * 4; ++tile_id) {
        const int m_tile = tile_id / 8;
        const int n_tile = tile_id % 8;

        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        #pragma unroll
        for (int k_tile = 0; k_tile < K / 16; ++k_tile) {
          load_matrix_sync(a_frag, sm_q + m_tile * 16 * K + k_tile * 16, K);
          load_matrix_sync(b_frag, sm_S + k_tile * 16 * V + n_tile * 16, V);
          mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        store_matrix_sync(warp_scratch, c_frag, 16, mem_row_major);
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int idx_in_tile = lane * 8 + e;
          const int row_in_tile = idx_in_tile / 16;
          const int col_in_tile = idx_in_tile % 16;
          const int i  = m_tile * 16 + row_in_tile;
          const int vd = n_tile * 16 + col_in_tile;
          const float o_inter = warp_scratch[row_in_tile * 16 + col_in_tile];
          // o_intra plain: sum_j attn[i, j] * v_new[j, vd] (j > i contribution is 0).
          float o_intra = 0.0f;
          #pragma unroll
          for (int j = 0; j < C; ++j) {
            o_intra += sm_attn[i * C + j] * bf2f(sm_vnew[j * V + vd]);
          }
          sm_out[i * V + vd] = f2bf(o_inter + o_intra);
        }
      }
    }
    __syncthreads();

    // === Phase 11: Write output for this chunk ===
    // output[(chunk_start+i) * v_heads * V + vh * V + vd]
    {
      const __nv_bfloat16 *sm_out = sm_v;
      for (int i = 0; i < valid; ++i) {
        const int tok = chunk_start + i;
        __nv_bfloat16 *out_row = output + (size_t)tok * v_heads * V + vh * V;
        for (int d = tid; d < V; d += blockDim.x) {
          out_row[d] = sm_out[i * V + d];
        }
      }
    }

    // === Phase 12a: Materialize k_scaled[i, kd] = k[i, kd] * coef[i] in sm_wdec ===
    // sm_wdec is free after Phase 9 (v_prime computation consumed it).
    // Pre-multiplying once saves an FMA per S_next cell and a redundant SMEM read.
    {
      __nv_bfloat16 *sm_k_scaled = sm_wdec;
      for (int idx = tid; idx < C * K; idx += blockDim.x) {
        const int i = idx / K;
        const int kd = idx % K;
        sm_k_scaled[idx] = f2bf(bf2f(sm_k[i * K + kd]) * sm_coef[i]);
      }
    }
    __syncthreads();

    // === Phase 12b: State update via wmma tensor cores ===
    // S[kd, vd] = expD_last * S[kd, vd] + (k_scaled^T @ v_new)[kd, vd]
    //
    // k_scaled is stored as [C=32, K=128] row-major; transposing to [K=128, C=32]
    // (which is what k_scaled^T is for the matmul) reads as col-major with ldm=128.
    // v_new is [C=32, V=128] row-major.  Output is [K=128, V=128] = 64 tiles of 16×16.
    // 4 warps × 16 tiles/warp.  Each tile accumulates over K_wmma = 32/16 = 2 mma ops.
    {
      using namespace nvcuda::wmma;
      const __nv_bfloat16 *sm_k_scaled = sm_wdec;
      const int warp_id = tid / 32;
      const int lane    = tid % 32;
      float *warp_scratch = sm_wmma_scratch + warp_id * 16 * 16;

      // 64 output tiles total, indexed linearly tile_id = m_tile * 8 + n_tile.
      // Each warp owns tile_ids [warp_id*16, (warp_id+1)*16).
      for (int tile_id = warp_id * 16; tile_id < (warp_id + 1) * 16; ++tile_id) {
        const int m_tile = tile_id / 8;   // 0..7 (over K=128/16)
        const int n_tile = tile_id % 8;   // 0..7 (over V=128/16)

        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major> a_frag;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        // K_wmma loop: 32/16 = 2 iterations.
        #pragma unroll
        for (int k_tile = 0; k_tile < 2; ++k_tile) {
          // A (k_scaled^T) as col_major: storage offset for tile starting at
          // (logical_row=m_tile*16, logical_col=k_tile*16) is k_tile*16*K + m_tile*16.
          load_matrix_sync(a_frag,
              sm_k_scaled + k_tile * 16 * K + m_tile * 16, K);
          load_matrix_sync(b_frag,
              sm_vnew + k_tile * 16 * V + n_tile * 16, V);
          mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store to per-warp scratch as fp32, then combine with existing S as bf16.
        store_matrix_sync(warp_scratch, c_frag, 16, mem_row_major);
        // Lane combine: 16x16 = 256 elements, 32 lanes → 8 elements per lane.
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int idx_in_tile = lane * 8 + e;
          const int row_in_tile = idx_in_tile / 16;
          const int col_in_tile = idx_in_tile % 16;
          const int kd = m_tile * 16 + row_in_tile;
          const int vd = n_tile * 16 + col_in_tile;
          const float delta = warp_scratch[row_in_tile * 16 + col_in_tile];
          const float old_s = bf2f(sm_S[kd * V + vd]);
          sm_S[kd * V + vd] = f2bf(expD_last * old_s + delta);
        }
      }
    }
    __syncthreads();
  }

  // --- Write final state for this v_head back to global ---
  // Transpose SMEM [K, V] back to the canonical global [V, K] layout so the
  // decode-time qwen36_deltanet_decode kernel (and the interpreter
  // DELTANET_RECUR opcode) read the state they expect. Writing the SMEM
  // copy out untransposed silently corrupts every decode step after a
  // chunked prefill — see the deltanet prefill/decode state parity case in
  // smoke.cu, which gates exactly this.
  {
    __nv_bfloat16 *state_out = state + (size_t)vh * K * V;
    const int total = K * V;
    for (int idx = tid; idx < total; idx += blockDim.x) {
      const int vd = idx / K;
      const int kd = idx % K;
      state_out[idx] = sm_S[kd * V + vd];
    }
  }
}

} // namespace

extern "C" int
qwen36_deltanet_prefill(const qwen36_deltanet_prefill_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->q_bf16.ptr == 0 || spec->k_bf16.ptr == 0 ||
      spec->v_bf16.ptr == 0 || spec->state_bf16.ptr == 0 ||
      spec->output_bf16.ptr == 0 ||
      spec->gate_f32.ptr == 0 || spec->beta_f32.ptr == 0) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->shape.qk_heads == 0 || spec->shape.v_heads == 0 ||
      spec->shape.key_dim == 0 || spec->shape.value_dim == 0 ||
      spec->tokens == 0 ||
      spec->shape.v_heads % spec->shape.qk_heads != 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  // v1 only supports the Qwen3.6 shape and chunk_size=32.
  if (spec->shape.qk_heads != kPrefillQkHeads ||
      spec->shape.v_heads != kPrefillVHeads ||
      spec->shape.key_dim != kPrefillKeyDim ||
      spec->shape.value_dim != kPrefillValDim) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }
  if (spec->chunk_size != 0 && spec->chunk_size != kPrefillChunk) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  const float state_decay = spec->state_decay == 0.0f ? 1.0f : spec->state_decay;
  const float update_scale =
      spec->update_scale == 0.0f ? 1.0f : spec->update_scale;

  // Shared memory layout — must match the kernel's `extern __shared__` cursor.
  constexpr int C = kPrefillChunk;
  constexpr int K = kPrefillKeyDim;
  constexpr int V = kPrefillValDim;
  const size_t bytes_S    = K * V * sizeof(__nv_bfloat16);
  const size_t bytes_q    = C * K * sizeof(__nv_bfloat16);
  const size_t bytes_k    = C * K * sizeof(__nv_bfloat16);
  const size_t bytes_v    = C * V * sizeof(__nv_bfloat16);
  const size_t bytes_A    = C * C * sizeof(float);
  const size_t bytes_attn = C * C * sizeof(float);
  const size_t bytes_vnew = C * V * sizeof(__nv_bfloat16);
  const size_t bytes_wdec = C * K * sizeof(__nv_bfloat16);
  const size_t bytes_scratch = 5 * C * sizeof(float)         // g, D, beta, expD, coef
                              + 4 * 16 * 16 * sizeof(float)  // wmma per-warp scratch
                              + C * C * sizeof(__nv_bfloat16); // sm_A_bf16
  const size_t smem_bytes =
      bytes_S + bytes_q + bytes_k + bytes_v + bytes_A + bytes_attn +
      bytes_vnew + bytes_wdec + bytes_scratch;

  // Opt in to dynamic shared memory on SM_120 (cap ~100 KB).
  cudaError_t attr_err = cudaFuncSetAttribute(
      deltanet_prefill_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  if (attr_err != cudaSuccess) {
    return QWEN36_STATUS_CUDA_ERROR;
  }

  const dim3 grid(static_cast<unsigned int>(spec->shape.v_heads), 1, 1);
  const dim3 block(kPrefillThreads, 1, 1);

  deltanet_prefill_kernel<<<grid, block, smem_bytes,
                            qwen36_internal_active_stream()>>>(
      ptr<const __nv_bfloat16>(spec->q_bf16),
      ptr<const __nv_bfloat16>(spec->k_bf16),
      ptr<const __nv_bfloat16>(spec->v_bf16),
      ptr<const float>(spec->gate_f32),
      ptr<const float>(spec->beta_f32),
      ptr<__nv_bfloat16>(spec->state_bf16),
      ptr<__nv_bfloat16>(spec->output_bf16),
      spec->shape, spec->tokens,
      spec->q_token_stride, spec->k_token_stride, spec->v_token_stride,
      state_decay, update_scale, spec->qk_l2norm != 0);

  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
