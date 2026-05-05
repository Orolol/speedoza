// Direction B NVFP4 gemv kernel for Blackwell SM_120a — Phase B3.
//
// Hand-rolled tensor-core gemv that drives the SM120 NVFP4 block-scaled MMA
// atom (mma.sync.aligned.kind::mxf4nvf4.scale_vec::4X.m16n8k64.f32.e2m1.e2m1
// .f32.ue4m3) directly via inline PTX from cute. CUTLASS's high-level
// CollectiveBuilder rejects narrow-N tiles at N=1 (see Phase B2 notes), so
// we drop down to the atom and stage register tiles by hand.
//
// B3.7: kernel is now templated on kWarpsPerBlockTpl with two compiled
// specializations:
//   - 16 warps / CTA, 512 threads, K%1024==0 — preferred path. Higher
//     SM-resident warp count, best latency hiding for "fat" decode shapes
//     (e.g. K=5120 for q_proj, K=17408 for o_proj).
//   - 8 warps / CTA, 256 threads, K%512==0 — fallback for shapes whose K
//     is a multiple of 512 but not 1024 (e.g. K=3584 for linear-attention
//     out_proj where value_dim=3584). Same architecture as B3.5; bench
//     showed +5-15% over cuBLASLt at MTP=0/4.
// The entry point picks the largest specialization the K dimension will
// admit; shapes that don't satisfy K%512==0 fall through to cuBLASLt.
//
// CTA layout (B3.7: intra-CTA split-K, kWarpsPerBlockTpl warps)
//   - Each CTA owns ONE m16 MMA tile (16 rows); all warps cooperate on
//     the SAME 16 output rows but DIFFERENT K shards. This multiplies
//     parallelism for low-M shapes by Nwarps× compared to the original
//     "1 warp = 1 m16 tile" layout.
//   - blockDim.x = kWarpsPerBlockTpl * 32, gridDim.x = ceil(M / 16).
//   - For k_chunk in [warp_id*K_per_warp/64, (warp_id+1)*K_per_warp/64),
//     every warp issues one MMA accumulating into its private 4-register
//     float accumulator, then a final cross-warp reduction in smem sums
//     the partial dot products and writes the n=0 column to gmem.
//
// Operand staging (lane L ∈ [0,32))
//   t0 = L & 3, t1 = L >> 2.
//   A[r] for r = v1 + 2*v2, v1,v2 ∈ {0,1}: 8 packed fp4 from row (t1 + 8*v2)
//   at fp4 offset (8*t0 + 32*v1) — load as one uint32 from the row's
//   packed-fp4 buffer.
//   B[r] for r = v1 ∈ {0,1}: 8 packed fp4 from the single activation column
//   at fp4 offset (8*t0 + 32*v1).
//   SFA: lane decomposition is (t0_sf=L&1, t2_sf=L>>2) → m_row_sf =
//   8*(L&1) + (L>>2). 4 packed e4m3 bytes (k_group 0..3 in low..high
//   nibbles).
//   SFB: n_col_sf = L>>2 (broadcast). At N=1 outer is always 0, so the
//   scale-byte address only varies with k_group.
//   D[r] for r = v0 + 2*v1: row (t1 + 8*v1), col (2*t0 + v0). For the
//   n=0 output we keep lanes with t0==0 → D[0] writes row t1, D[2] writes
//   row t1+8.
//
// Scale layout: identical vec16_scale_offset swizzle as the cuBLASLt and
// CUTLASS paths so we can read SFA/SFB straight from the same buffers.
//
// Soft regime: n==1 && m%16==0 && k%512==0 (with k%1024==0 routing to the
// 16-warp specialization). The k-alignment constraint comes from split-K:
// K is sharded into Nwarps equal pieces (one per warp), each of which must
// be a multiple of the kKPerMma=64 inner-loop chunk. For shapes the MMA
// cannot service we return QWEN36_STATUS_NOT_IMPLEMENTED so the dispatcher
// routes to cuBLASLt. Active env var: QWEN36_DECODE_GEMV=1.

#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// We deliberately AVOID including <cutlass/...> or <cute/...> here even
// though that header defines the MMA atom we want. Pulling cute into a
// translation unit that contains a `__global__` kernel triggers a host
// stub-generator bug where it tries to register cuda::std::__cpo entries
// (begin/end/cbegin/cend etc.) as device variables, but `::cuda::std`
// isn't declared in the host compilation unit, so g++ chokes with
// "'::cuda' has not been declared". Mirror the exact PTX from the cute
// atom (cute/arch/mma_sm120.hpp:3215, kind::mxf4nvf4.scale_vec::4X
// .m16n8k64 with e2m1 × e2m1 and ue4m3 scales) inline below. This keeps
// the TU self-contained — no cute, no cuda::std.
//
// The `kind::mxf4nvf4` opcode is sm_120a-only. Building with
// `-arch=sm_120a` emits BOTH a compute_120 (PTX-forward) image AND a
// compute_120a image; the compute_120 image must softly fall through, so
// we gate the asm on __CUDA_ARCH_FEAT_SM120_ALL.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200) &&                       \
    defined(__CUDA_ARCH_FEAT_SM120_ALL)
#define QWEN36_DECODE_GEMV_MMA_DEVICE 1
#else
#define QWEN36_DECODE_GEMV_MMA_DEVICE 0
#endif

// Host-side guard: enabled whenever the CUDA toolchain can target sm_120a
// (driver / nvcc 12.8+). Nothing on the host depends on the device-side
// macro. We always compile the kernel; the device-side body is
// conditionally a no-op for the compute_120 fallback PTX image.
#define QWEN36_DECODE_GEMV_MMA 1

namespace {

__host__ __device__ size_t gemv_div_ceil(size_t a, size_t b) {
  return (a + b - 1) / b;
}

__host__ __device__ size_t gemv_round_up(size_t v, size_t m) {
  return gemv_div_ceil(v, m) * m;
}

// cuBLASLt vec16 scale-swizzle layout. See ops.cu:vec16_scale_offset.
__host__ __device__ size_t gemv_vec16_scale_offset(size_t inner, size_t outer,
                                                   size_t sf_inner_dim) {
  const size_t block_inner = (inner / 4) * 4;
  const size_t block_outer = outer / 128;
  const size_t block_offset = (block_inner + block_outer * sf_inner_dim) * 128;
  const size_t tile_outer = outer % 128;
  const size_t tile_inner = inner % 4;
  return block_offset + (tile_outer % 32) * 16 + (tile_outer / 32) * 4 +
         tile_inner;
}

// File-scope constants that are independent of the warp count — these stay
// here so the entry-point can compute the launch geometry uniformly.
constexpr int kRowsPerWarp = 16;
constexpr int kRowsPerBlock = kRowsPerWarp;  // 16 — one m16 MMA tile / CTA
constexpr int kKPerMma = 64;
constexpr unsigned kATilePerWarpBytes =
    static_cast<unsigned>(kRowsPerBlock) * 32u;  // 512

// ---- cp.async helpers (sm_80+; available on sm_120). ----
// `cg` (cache global, bypass L1) is the right choice for streaming the A
// weight tile — we don't reuse it after consumption. Predicated form uses
// the trailing src-size operand: when src_bytes==0, the destination smem
// is filled with 16 zeros (PTX cp.async semantics). This lets us drop the
// per-thread tail-row branch.
__device__ __forceinline__ void cp_async_16_pred(unsigned smem_addr,
                                                 const void *gmem_ptr,
                                                 bool valid) {
#if QWEN36_DECODE_GEMV_MMA_DEVICE
  const unsigned src_bytes = valid ? 16u : 0u;
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
      :
      : "r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes));
#else
  (void)smem_addr;
  (void)gmem_ptr;
  (void)valid;
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if QWEN36_DECODE_GEMV_MMA_DEVICE
  asm volatile("cp.async.commit_group;\n");
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if QWEN36_DECODE_GEMV_MMA_DEVICE
  asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
#endif
}

// Inline-PTX wrapper for the SM120 mxf4nvf4 scale_vec::4X m16n8k64 atom.
// Direct mirror of cute/arch/mma_sm120.hpp:3215. Lives at file scope so
// the kernel body stays readable.
__device__ __forceinline__ void mma_mxf4nvf4_4x_m16n8k64(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint32_t sfa0, uint32_t sfb0) {
#if QWEN36_DECODE_GEMV_MMA_DEVICE
  constexpr uint16_t bid = 0;
  constexpr uint16_t tid = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(sfa0), "h"(bid), "h"(tid),
        "r"(sfb0), "h"(bid), "h"(tid));
#else
  d0 = c0; d1 = c1; d2 = c2; d3 = c3;
  (void)a0; (void)a1; (void)a2; (void)a3;
  (void)b0; (void)b1; (void)sfa0; (void)sfb0;
#endif
}

}  // namespace

// Templated kernel. Two specializations are emitted (kWarpsPerBlockTpl =
// 16 and 8). The kernel is declared at namespace scope so explicit
// instantiation can force the symbols into the .so.
// Tensor scales (a_scale_2, b_scale_2) intentionally NOT in the kernel
// signature: the runtime caller (engine.rs) already folds them into
// `spec->alpha` before invoking the kernel, mirroring the cuBLASLt
// contract (nvfp4_gemm.cu only passes `alpha`, never dereferences
// a_scale_2/b_scale_2). Multiplying by them inside the kernel would
// square the tensor-scale factor and produce silent gibberish on real
// model weights.
template <int kWarpsPerBlockTpl>
__global__ void __launch_bounds__(kWarpsPerBlockTpl * 32)
nvfp4_gemv_mma_kernel_tpl(const uint8_t *__restrict__ a_fp4,
                          const uint8_t *__restrict__ a_scale,
                          const uint8_t *__restrict__ b_fp4,
                          const uint8_t *__restrict__ b_scale,
                          float alpha,
                          __nv_bfloat16 *__restrict__ output,
                          size_t M, size_t K) {
  constexpr int kWarpsPerBlock = kWarpsPerBlockTpl;
  constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
  // Per-warp K-shard alignment: K must be a multiple of (Nwarps * 64) so
  // every warp gets an integral number of 64-element MMA chunks.
  constexpr int kKShardChunkAlign = kWarpsPerBlock * kKPerMma;
  (void)kKShardChunkAlign;  // documentation; the entry point enforces it.

  const unsigned warp_id = threadIdx.x >> 5;
  const unsigned lane = threadIdx.x & 31;
  // All kWarpsPerBlock warps in the CTA share the same m16 tile (split-K).
  const size_t m_base = static_cast<size_t>(blockIdx.x) * kRowsPerBlock;
  // NOTE: do NOT early-return for `m_base >= M` — every thread must reach
  // every __syncthreads() in this kernel (cooperative B load + final
  // cross-warp reduction). Tail rows are zero-filled in the cooperative
  // A load and discarded by the epilogue bound checks.

  const size_t packed_cols = K / 2;       // bytes per fp4 row
  const size_t scale_cols = K / 16;       // scale groups per row
  const size_t sf_inner_dim = gemv_round_up(scale_cols, 4);

  // ---- Cooperative load of the activation column into smem (CTA-shared).
  // The B operand is a single column of K packed fp4 (K/2 bytes). All warps
  // share the same buffer; each warp will read only the bytes for its
  // K-shard during the MMA loop.
  //
  // Dynamic shared memory layout (parameterized on kWarpsPerBlock):
  //   - K/2 bytes for the activation column (B operand, CTA-shared).
  //   - kWarpsPerBlock * 2 * 512 bytes for the per-warp DOUBLE-BUFFERED
  //     A-operand staging tile (each warp owns its own buffer pair since
  //     all warps process the SAME 16 rows but different K chunks).
  //   - 2 * kRowsPerBlock * kWarpsPerBlock * 4 bytes for the cross-warp
  //     reduction scratch (2 halves × 16 rows × Nwarps × 4 B float).
  //
  // The MMA gate guarantees K % (kWarpsPerBlock*64) == 0, so K/2 is a
  // multiple of 16 — the uint4 stride below covers the whole buffer with
  // no tail.
  constexpr unsigned kATileBufBytes =
      static_cast<unsigned>(kWarpsPerBlock) * 2u * kATilePerWarpBytes;
  extern __shared__ uint8_t smem[];
  uint8_t *smem_b_fp4 = smem;                                      // K/2 B
  uint8_t *smem_a_base = smem + K / 2;                             // tile region
  // Per-warp double buffer: smem_a_buf[w][b] points to a 512-byte tile.
  // Layout: [warp0_buf0 | warp0_buf1 | warp1_buf0 | warp1_buf1 | ...].
  uint8_t *const smem_a_buf_w0 =
      smem_a_base + warp_id * (2u * kATilePerWarpBytes);
  uint8_t *smem_a_buf[2] = {smem_a_buf_w0,
                            smem_a_buf_w0 + kATilePerWarpBytes};
  // Reduction scratch follows the A tile region.
  // reduction[hi/lo][row_in_tile][warp] → laid out as
  // [2][16][kWarpsPerBlock] floats.
  float *smem_reduction =
      reinterpret_cast<float *>(smem_a_base + kATileBufBytes);
  {
    const size_t b_bytes = packed_cols;
    const size_t b_vecs = b_bytes / 16;  // K/2 % 16 == 0
    const uint4 *b_fp4_vec = reinterpret_cast<const uint4 *>(b_fp4);
    uint4 *smem_b_vec = reinterpret_cast<uint4 *>(smem_b_fp4);
    for (size_t i = threadIdx.x; i < b_vecs;
         i += static_cast<size_t>(kThreadsPerBlock)) {
      smem_b_vec[i] = b_fp4_vec[i];
    }
  }
  __syncthreads();

  // Lane decomposition for the operand layouts (canonical m16n8k* form).
  const unsigned t0 = lane & 3u;
  const unsigned t1 = lane >> 2;

  // SFA decomposition: m_row_sf = 8*(L&1) + (L>>2).
  const unsigned t0_sf_a = lane & 1u;
  const unsigned t2_sf_a = lane >> 2;
  const unsigned m_row_sf = 8u * t0_sf_a + t2_sf_a;
  // Clamp to the last valid row for tail warps (m_base >= M). Such warps
  // still participate in the cooperative A-tile sync below, so they must
  // execute the K loop; we route their SFA read to a safe row to avoid
  // OOB gmem fetches. Their MMA output is discarded by the epilogue
  // bound check (`row_lo < M` / `row_hi < M`), and their A-tile rows are
  // zero-filled by the cooperative load below.
  const size_t a_row_for_sf_raw = m_base + m_row_sf;
  const size_t a_row_for_sf =
      a_row_for_sf_raw < M ? a_row_for_sf_raw : (M - 1);

  // SFB at N=1: outer is always 0, no n-decomposition needed.

  // ---- Per-lane smem row offsets into the per-warp A-tile (16 rows × 32 B).
  // Tile is row-major: row r occupies bytes [r*32, r*32+32). All warps
  // share the same 16 rows (split-K), so the row offsets do NOT depend on
  // warp_id (in contrast to the pre-B3.4 layout).
  const unsigned a_row0_byte_off = t1 * 32u;
  const unsigned a_row1_byte_off = (t1 + 8u) * 32u;
  const unsigned a_off_v0 = 4u * t0;        // bytes within row, v1=0
  const unsigned a_off_v1 = 4u * t0 + 16u;  // bytes within row, v1=1

  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

  // ---- Split-K: each warp owns K_per_warp = K/kWarpsPerBlock. ----
  // K_per_warp is a multiple of kKPerMma=64 (entry-point gate enforces
  // K % (kWarpsPerBlock*kKPerMma) == 0), so each warp processes
  // k_chunks_per_warp = K / (kWarpsPerBlock*64) chunks of 64 elements,
  // starting at warp_id * k_chunks_per_warp.
  const size_t k_chunks_total = K / kKPerMma;                 // K/64
  const size_t k_chunks_per_warp =
      k_chunks_total / static_cast<size_t>(kWarpsPerBlock);
  const size_t kc_warp_start =
      static_cast<size_t>(warp_id) * k_chunks_per_warp;

  // Cooperative A-tile load mapping (per-WARP, not per-CTA, since each
  // warp loads its own 16-row × 32-byte buffer):
  //   32 lanes × uint4 (16 B) = 512 B = exactly one tile.
  //   row_in_tile = lane / 2,  byte_off_in_row = (lane % 2) * 16
  const unsigned a_load_row_in_tile = lane >> 1;
  const unsigned a_load_byte_off = (lane & 1u) << 4;  // 0 or 16
  const size_t a_load_global_row =
      static_cast<size_t>(blockIdx.x) * kRowsPerBlock + a_load_row_in_tile;
  const bool a_load_row_valid = a_load_global_row < M;
  const uint8_t *a_load_row_ptr =
      a_load_row_valid ? (a_fp4 + a_load_global_row * packed_cols) : nullptr;

  // ---- cp.async double-buffered pipeline (per-warp). Prologue issues
  // chunk 0 (warp's first chunk) into buf[0]. The loop body issues the
  // NEXT chunk into the other buffer, then waits for the CURRENT one.
  // Synchronization is per-WARP via __syncwarp() — we deliberately do
  // NOT use __syncthreads() inside the inner loop, because that would
  // force the warps to march in lockstep and defeat the split-K
  // parallelism (each warp's buffer is private to that warp's lanes).
  auto issue_chunk_async = [&](size_t kc_global, uint8_t *dst_buf) {
    const size_t k_byte_base = kc_global * (kKPerMma / 2);  // 32 B per chunk
    const void *src = a_load_row_valid
                          ? static_cast<const void *>(
                                a_load_row_ptr + k_byte_base + a_load_byte_off)
                          : static_cast<const void *>(a_fp4);  // dummy ptr
    const unsigned smem_addr =
        __cvta_generic_to_shared(dst_buf + lane * 16u);
    cp_async_16_pred(smem_addr, src, a_load_row_valid);
  };

  // Prologue: stage this warp's first chunk into buf[0].
  if (k_chunks_per_warp > 0) {
    issue_chunk_async(kc_warp_start, smem_a_buf[0]);
    cp_async_commit();
  }

  for (size_t kc_local = 0; kc_local < k_chunks_per_warp; ++kc_local) {
    const size_t kc_global = kc_warp_start + kc_local;
    const size_t k_byte_base = kc_global * (kKPerMma / 2);  // 32 B per chunk
    const size_t k_group_base = kc_global * 4;              // 4 SF groups
    const unsigned cur_buf = static_cast<unsigned>(kc_local) & 1u;

    // Issue the NEXT chunk into the OTHER buffer (no race — distinct
    // smem region within this warp's private slot).
    const bool has_next = (kc_local + 1) < k_chunks_per_warp;
    if (has_next) {
      issue_chunk_async(kc_global + 1, smem_a_buf[cur_buf ^ 1u]);
      cp_async_commit();
    }

    // Wait for the current chunk's load to retire. With "load-ahead by
    // 1", at most one group is in flight after this wait → wait_group(1).
    // On the final iteration no next load was issued, so drain fully.
    if (has_next) {
      cp_async_wait_group<1>();
    } else {
      cp_async_wait_group<0>();
    }
    // Per-warp visibility for the freshly staged tile. cp.async per-thread
    // commit semantics + __syncwarp() suffice because each lane only reads
    // its own warp's smem slot. NO __syncthreads() here — split-K demands
    // that warps run independently to overlap their K shards.
    __syncwarp();

    uint8_t *smem_a_tile_cur = smem_a_buf[cur_buf];

    // ---- A operand: 4 uint32, each = 4 bytes = 8 fp4 elements. ----
    // Sourced from the active buffer of this warp's per-warp tile.
    uint32_t a0 = *reinterpret_cast<const uint32_t *>(
        smem_a_tile_cur + a_row0_byte_off + a_off_v0);
    uint32_t a1 = *reinterpret_cast<const uint32_t *>(
        smem_a_tile_cur + a_row1_byte_off + a_off_v0);
    uint32_t a2 = *reinterpret_cast<const uint32_t *>(
        smem_a_tile_cur + a_row0_byte_off + a_off_v1);
    uint32_t a3 = *reinterpret_cast<const uint32_t *>(
        smem_a_tile_cur + a_row1_byte_off + a_off_v1);

    // ---- B operand: 2 uint32. Single activation column at N=1. ----
    // Reads from CTA-shared smem; each warp reads only its K-shard's bytes.
    const size_t b_byte_off_v0 = k_byte_base + 4u * t0;
    const size_t b_byte_off_v1 = k_byte_base + 4u * t0 + 16u;
    uint32_t b0 =
        *reinterpret_cast<const uint32_t *>(smem_b_fp4 + b_byte_off_v0);
    uint32_t b1 =
        *reinterpret_cast<const uint32_t *>(smem_b_fp4 + b_byte_off_v1);

    // ---- SFA/SFB coalesced as one uint32 each (B3.3.0). ----
    const size_t sfa_off =
        gemv_vec16_scale_offset(k_group_base, a_row_for_sf, sf_inner_dim);
    const uint32_t sfa =
        *reinterpret_cast<const uint32_t *>(a_scale + sfa_off);

    // SFB: outer = 0 at N=1.
    const size_t sfb_off =
        gemv_vec16_scale_offset(k_group_base, 0, sf_inner_dim);
    const uint32_t sfb =
        *reinterpret_cast<const uint32_t *>(b_scale + sfb_off);

    mma_mxf4nvf4_4x_m16n8k64(acc0, acc1, acc2, acc3,
                             a0, a1, a2, a3,
                             b0, b1,
                             acc0, acc1, acc2, acc3,
                             sfa, sfb);
  }

  // ---- Cross-warp reduction (split-K finalization). Each warp now holds
  // its private partial sum for the same 16 output rows. We sum the
  // kWarpsPerBlock warp partials in smem and let warp 0 emit the result.
  //
  // Lane decomposition: lanes with t0==0 hold the n=0 column.
  //   acc0 holds row t1 (m=0..7), acc2 holds row t1+8 (m=8..15).
  // Indexing: smem_reduction[hi*16*kWarpsPerBlock + row*kWarpsPerBlock
  //                          + warp].
  constexpr unsigned kRedRowStride = static_cast<unsigned>(kWarpsPerBlock);
  constexpr unsigned kRedHalfStride = 16u * kRedRowStride;
  if (t0 == 0u) {
    smem_reduction[0u * kRedHalfStride + t1 * kRedRowStride + warp_id] = acc0;
    smem_reduction[1u * kRedHalfStride + t1 * kRedRowStride + warp_id] = acc2;
  }
  __syncthreads();  // the ONLY __syncthreads() after activation cache load.

  if (warp_id == 0u && t0 == 0u) {
    float sum_lo = 0.f;
    float sum_hi = 0.f;
    #pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      sum_lo += smem_reduction[0u * kRedHalfStride + t1 * kRedRowStride +
                               static_cast<unsigned>(w)];
      sum_hi += smem_reduction[1u * kRedHalfStride + t1 * kRedRowStride +
                               static_cast<unsigned>(w)];
    }
    const size_t row_lo = m_base + t1;
    const size_t row_hi = m_base + t1 + 8u;
    if (row_lo < M) {
      output[row_lo] = __float2bfloat16(sum_lo * alpha);
    }
    if (row_hi < M) {
      output[row_hi] = __float2bfloat16(sum_hi * alpha);
    }
  }
}

// Explicit instantiations — force both specialization symbols into the .so.
template __global__ void
nvfp4_gemv_mma_kernel_tpl<16>(const uint8_t *, const uint8_t *, const uint8_t *,
                              const uint8_t *, float, __nv_bfloat16 *, size_t,
                              size_t);
template __global__ void
nvfp4_gemv_mma_kernel_tpl<8>(const uint8_t *, const uint8_t *, const uint8_t *,
                             const uint8_t *, float, __nv_bfloat16 *, size_t,
                             size_t);

namespace {

template <typename T> T *as_device_ptr(qwen36_device_ptr_t p) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(p.ptr));
}

}  // namespace

extern "C" int qwen36_decode_nvfp4_gemv(
    const qwen36_nvfp4_gemm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->m == 0 || spec->n == 0 || spec->k == 0 ||
      spec->a_fp4.ptr == 0 || spec->a_scale.ptr == 0 ||
      spec->b_fp4.ptr == 0 || spec->b_scale.ptr == 0 ||
      spec->c_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }

#if QWEN36_DECODE_GEMV_MMA
  // MMA regime: N=1, M aligned to the m16 MMA tile, K aligned to the
  // split-K chunk. We pick the largest-warp specialization the K dimension
  // admits:
  //   - kAlign16 = 16 * 64 = 1024  → 16 warps / CTA (preferred).
  //   - kAlign8  =  8 * 64 =  512  → 8 warps / CTA  (fallback).
  // K not divisible by 512 returns NOT_IMPLEMENTED so cuBLASLt picks it up.
  constexpr size_t kAlign16 = 16u * static_cast<size_t>(kKPerMma);  // 1024
  constexpr size_t kAlign8 = 8u * static_cast<size_t>(kKPerMma);    // 512

  if (spec->n != 1 || (spec->m % 16) != 0) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  int chosen_warps = 0;
  if ((spec->k % kAlign16) == 0) {
    chosen_warps = 16;
  } else if ((spec->k % kAlign8) == 0) {
    chosen_warps = 8;
  } else {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  const size_t M = spec->m;
  const size_t K = spec->k;
  const dim3 grid(static_cast<unsigned>(gemv_div_ceil(M, kRowsPerBlock)), 1, 1);
  cudaStream_t stream = qwen36_internal_active_stream();

  // Smem footprint (parameterized on chosen_warps):
  //   - K/2 bytes activation column (B operand, CTA-shared).
  //   - chosen_warps * 2 * 512 bytes per-warp double-buffered A tile.
  //   - 2 * 16 * chosen_warps * 4 bytes cross-warp reduction scratch.
  // E.g. K=5120, chosen_warps=16 → 2560 + 16384 + 2048 = 20.5 KiB.
  //      K=5120, chosen_warps=8  → 2560 +  8192 + 1024 = 11.5 KiB.
  // Both well under the 48 KiB default per-block smem cap on SM_120.
  const size_t a_tile_bytes = static_cast<size_t>(chosen_warps) * 2u *
                              static_cast<size_t>(kATilePerWarpBytes);
  const size_t reduction_bytes =
      2u * static_cast<size_t>(kRowsPerBlock) *
      static_cast<size_t>(chosen_warps) * sizeof(float);
  const size_t smem_bytes = K / 2 + a_tile_bytes + reduction_bytes;

  if (chosen_warps == 16) {
    const dim3 block(16u * 32u, 1, 1);
    nvfp4_gemv_mma_kernel_tpl<16><<<grid, block, smem_bytes, stream>>>(
        as_device_ptr<const uint8_t>(spec->a_fp4),
        as_device_ptr<const uint8_t>(spec->a_scale),
        as_device_ptr<const uint8_t>(spec->b_fp4),
        as_device_ptr<const uint8_t>(spec->b_scale), spec->alpha,
        as_device_ptr<__nv_bfloat16>(spec->c_bf16), M, K);
  } else {
    const dim3 block(8u * 32u, 1, 1);
    nvfp4_gemv_mma_kernel_tpl<8><<<grid, block, smem_bytes, stream>>>(
        as_device_ptr<const uint8_t>(spec->a_fp4),
        as_device_ptr<const uint8_t>(spec->a_scale),
        as_device_ptr<const uint8_t>(spec->b_fp4),
        as_device_ptr<const uint8_t>(spec->b_scale), spec->alpha,
        as_device_ptr<__nv_bfloat16>(spec->c_bf16), M, K);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return QWEN36_STATUS_CUDA_ERROR;
  }
  return QWEN36_STATUS_SUCCESS;
#else
  (void)spec;
  return QWEN36_STATUS_NOT_IMPLEMENTED;
#endif
}
