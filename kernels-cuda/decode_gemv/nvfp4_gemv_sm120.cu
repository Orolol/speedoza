// Direction B NVFP4 gemv kernel for Blackwell SM_120 — Phase B2.
//
// Naive CUTLASS-based implementation specialised for the gemv shape
// (M%128==0, K%128==0, N==1). Mirrors the layout/typing of the existing
// Mirage megakernel (`kernels-cuda/megakernel/nvfp4_matvec_sm120.cu`)
// but uses the smallest valid N tile (8) instead of 128 to avoid wasting
// SM occupancy on padding columns at decode-time N=1. Cluster shape
// stays <1,1,1>; persistent grid + TMA multicast land in Phase B3/B4.
//
// On any unsupported shape we return QWEN36_STATUS_NOT_IMPLEMENTED (5)
// and the Rust dispatcher routes back to the cuBLASLt path. Active env
// var: `QWEN36_DECODE_GEMV=1`.
//
// See `docs/superpowers/specs/2026-05-04-direction-b-nvfp4-gemv-design.md`.
#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_runtime.h>
#include <cstdint>

#include <cutlass/cutlass.h>
#include <cutlass/version.h>
// `<cutlass/cutlass.h>` does NOT transitively pull in arch/config.h, so the
// CUTLASS_ARCH_MMA_SM12x_SUPPORTED macros stay undefined and the #if below
// silently falls through to the NOT_IMPLEMENTED branch. Include it
// explicitly so the SM120 path is actually compiled.
#include <cutlass/arch/config.h>

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) ||                                \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#endif

namespace {

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) ||                                \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;    // weight matrix [M, K]
using LayoutBTag = cutlass::layout::ColumnMajor; // activation [K, N]
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;

using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Phase B2 tile shape: 128×128×128. The Direction B spec calls for a
// narrower N tile (e.g. 128×8×128) to match the FP4 MMA atom's natural
// N=8 — but the SM120 BlockScaledMma builder + cooperative scheduler
// statically rejects N<128 ("Invalid tile shape N." +
// "EPI_TILE_N must divide CTA_N"). Phase B3 will switch to a different
// schedule (warp-specialised non-cooperative or a hand-rolled persistent
// kernel) that admits a smaller N tile; until then we use the same
// baseline tile as the Mirage megakernel and rely on CUTLASS's epilogue
// masking for N=1.
using ThreadBlockShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
        LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
        LayoutBTag, AlignmentB, ElementAccumulator, ThreadBlockShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

template <typename T> T *as_device_ptr(qwen36_device_ptr_t p) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(p.ptr));
}

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED || CUTLASS_ARCH_MMA_SM121_SUPPORTED

} // namespace

extern "C" int qwen36_decode_nvfp4_gemv(
    const qwen36_nvfp4_gemm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  static_assert(CUTLASS_MAJOR >= 4,
                "decode_gemv requires CUTLASS 4.x (Blackwell FP4 path)");

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) ||                                \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (spec->m == 0 || spec->n == 0 || spec->k == 0 ||
      spec->a_fp4.ptr == 0 || spec->a_scale.ptr == 0 ||
      spec->b_fp4.ptr == 0 || spec->b_scale.ptr == 0 ||
      spec->c_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  // Supported regime: gemv-shaped, both M and K aligned to the tile.
  if (spec->n != 1 || (spec->m % 128) != 0 || (spec->k % 128) != 0) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }
  // BLOCKER: see docs/superpowers/notes/2026-05-04-direction-b-cutlass-blockers.md
  // The CUTLASS SM120 BlockScaled cooperative scheduler cannot serve N=1:
  //   - TMA requires output stride alignment that N=1 violates (run-time
  //     assertion `(gmem_prob_stride[1] & 0b1111) == 0` in
  //     cute/atom/copy_traits_sm90_tma.hpp).
  //   - Cluster builder rejects N<128 ("Invalid tile shape N." +
  //     "EPI_TILE_N must divide CTA_N") so we cannot use a smaller N tile.
  // Until B3 swaps in a non-TMA / smaller-N schedule (or a hand-rolled
  // kernel), this path stays soft-disabled so the dispatcher routes to
  // cuBLASLt unchanged.
  return QWEN36_STATUS_NOT_IMPLEMENTED;

  using namespace cute;

  const int M = static_cast<int>(spec->m);
  const int N = static_cast<int>(spec->n);
  const int K = static_cast<int>(spec->k);

  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          as_device_ptr<const typename ElementA::DataType>(spec->a_fp4),
          stride_A,
          as_device_ptr<const typename ElementB::DataType>(spec->b_fp4),
          stride_B,
          as_device_ptr<const typename ElementA::ScaleFactorType>(
              spec->a_scale),
          layout_SFA,
          as_device_ptr<const typename ElementB::ScaleFactorType>(
              spec->b_scale),
          layout_SFB,
      },
      {
          {spec->alpha, 0.0f},
          as_device_ptr<ElementD>(spec->c_bf16),
          stride_C,
          as_device_ptr<ElementD>(spec->c_bf16),
          stride_D,
      },
  };

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  size_t workspace_bytes = Gemm::get_workspace_size(arguments);
  void *workspace_ptr = nullptr;
  bool owns_workspace = false;
  if (workspace_bytes > 0) {
    if (spec->workspace.ptr != 0 && spec->workspace_bytes >= workspace_bytes) {
      workspace_ptr =
          reinterpret_cast<void *>(static_cast<uintptr_t>(spec->workspace.ptr));
    } else {
      cudaError_t alloc_err = cudaMalloc(&workspace_ptr, workspace_bytes);
      if (alloc_err != cudaSuccess) {
        return QWEN36_STATUS_CUDA_ERROR;
      }
      owns_workspace = true;
    }
  }

  cudaStream_t stream = qwen36_internal_active_stream();
  auto status = gemm.initialize(arguments, workspace_ptr, stream);
  if (status == cutlass::Status::kSuccess) {
    status = gemm.run(stream);
  }

  if (owns_workspace) {
    cudaFree(workspace_ptr);
  }

  if (status != cutlass::Status::kSuccess) {
    return QWEN36_STATUS_CUDA_ERROR;
  }
  return QWEN36_STATUS_SUCCESS;
#else
  (void)spec;
  return QWEN36_STATUS_NOT_IMPLEMENTED;
#endif
}
