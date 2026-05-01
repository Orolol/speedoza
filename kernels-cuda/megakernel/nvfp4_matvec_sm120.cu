// Mirage-style megakernel for Qwen3.6 NVFP4 decode.
//
// Phase 1 step 3: actually invoke the CUTLASS NVFP4 → BF16 GemmUniversalAdapter
// for Blackwell SM120 (RTX 5090) on our existing per-block scale layout.
//
// Layout compatibility note: our cuBLASLt-format `block_scale` tensors use
// the `vec16_scale_offset` swizzle (128-row outer-block, 4 k-groups inner-
// block, with the within-block stride pattern `m_lo*16 + m_hi*4 + k_idx`).
// CUTLASS's `Sm1xxBlockScaledConfig::SfKMajorAtom` for SFVecSize=16 has the
// IDENTICAL within-tile address formula (Layout<((32,4),(16,4)),
// ((16,4),(0,1))>), and the total `M*K/16`-byte allocation matches the
// safetensor `weight_scale` size exactly. So we can pass our existing
// SFA/SFB pointers straight into CUTLASS — no re-tile pass is required.
//
// On any unsupported shape or runtime error we return
// `QWEN36_STATUS_NOT_IMPLEMENTED` (5) so the Rust dispatcher routes back
// through the cuBLASLt path. The active env var is
// `QWEN36_USE_MEGAKERNEL_GEMM=1`.
#include "qwen36_fp4.h"
#include "active_stream.h"

#include <cuda_runtime.h>
#include <cstdint>

#include <cutlass/cutlass.h>
#include <cutlass/version.h>

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

// Element types — same as Example 79a (NVFP4 → BF16 on SM120).
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

// One device pointer untyped → typed cast helper.
template <typename T> T *as_device_ptr(qwen36_device_ptr_t p) {
  return reinterpret_cast<T *>(static_cast<uintptr_t>(p.ptr));
}

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED || CUTLASS_ARCH_MMA_SM121_SUPPORTED

} // namespace

extern "C" int qwen36_megakernel_nvfp4_gemm(
    const qwen36_nvfp4_gemm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  static_assert(CUTLASS_MAJOR >= 4,
                "Mirage megakernel requires CUTLASS 4.x (Blackwell FP4 path)");

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) ||                                \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (spec->m == 0 || spec->n == 0 || spec->k == 0 ||
      spec->a_fp4.ptr == 0 || spec->a_scale.ptr == 0 ||
      spec->b_fp4.ptr == 0 || spec->b_scale.ptr == 0 ||
      spec->c_bf16.ptr == 0) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  // Tile shape is 128×128×128; the adapter requires M and K to be
  // multiples of the tile in the relevant dim. K=hidden is 5120 (40·128)
  // for every Qwen3.6 hot decode shape, M is similarly aligned for the
  // fused weights (gate+up=34816, down=5120, in_proj fused=16640, etc.).
  // N=1 is fine: the epilogue handles partial N-tiles. Bail out softly on
  // anything else and let the cuBLASLt fallback take over.
  if ((spec->m % 128) != 0 || (spec->k % 128) != 0) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

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
          // Mainloop arguments.
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
          // Epilogue arguments.
          {spec->alpha, 0.0f},
          /* C ptr — unused since beta=0; pass D so types still match.   */
          as_device_ptr<ElementD>(spec->c_bf16),
          stride_C,
          as_device_ptr<ElementD>(spec->c_bf16),
          stride_D,
      },
  };

  Gemm gemm;

  // Reject shapes / layouts the kernel cannot service so the Rust
  // dispatcher routes back to cuBLASLt instead of producing wrong output.
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    return QWEN36_STATUS_NOT_IMPLEMENTED;
  }

  // Workspace: prefer the engine-supplied buffer (already sized for the
  // largest cuBLASLt plan); allocate inline only as a last resort.
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
    // Surface CUTLASS-specific errors as CUDA_ERROR; the dispatcher will
    // propagate this rather than fall back, so the caller actually sees
    // the failure (we don't want to silently mask kernel bugs).
    return QWEN36_STATUS_CUDA_ERROR;
  }
  return QWEN36_STATUS_SUCCESS;
#else
  // SM120 / SM121 not enabled in this build — caller falls back.
  (void)spec;
  return QWEN36_STATUS_NOT_IMPLEMENTED;
#endif
}
