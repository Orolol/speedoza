// Mirage-style megakernel for Qwen3.6 NVFP4 decode.
//
// Phase 1 step 2: instantiate the CUTLASS NVFP4 → BF16 GEMM template for
// Blackwell SM120 (RTX 5090). Modelled on
// `cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu`.
//
// At this commit the kernel types are defined and the build path links the
// CUTLASS device adapter, but the entry point still returns
// `QWEN36_STATUS_NOT_IMPLEMENTED` — the cuBLASLt path stays the active
// GEMM. The next commit hooks up `gemm.run()` once the SFA / SFB scale
// layout translation between our existing block_scale buffers (vLLM-style
// 128-row outer-block tiling) and CUTLASS's expected layout is settled.
#include "qwen36_fp4.h"

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/version.h>

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) ||                                \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#endif

namespace {

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) ||                                \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

// NVFP4 inputs (E2M1 with E4M3 per-block scales).
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;    // weight matrix [M, K]
using LayoutBTag = cutlass::layout::ColumnMajor; // activation [K, N=1]
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;

// BF16 output. Our cuBLASLt path also writes BF16, so we keep the same
// element type for parity. CUTLASS expects C (bias) of the same type as D
// when using the auto epilogue schedule; we bind C=D=BF16.
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Tile shape: 128×128×128. SM120 has 99 KB SMEM/SM, which fits this tile
// for NVFP4 inputs + BF16 output. Larger tiles (e.g. 256×128×128) overflow
// per the documented SM120 NVFP4 build constraints; smaller tiles
// (64×128) leave compute on the table for our M » N=1 shapes.
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

// Sanity: instantiating `Gemm` is enough to trigger CUTLASS's static_asserts
// at compile time, so a successful nvcc build is itself the first proof
// that the SM120 NVFP4 path is reachable in this checkout.
static_assert(sizeof(Gemm) > 0,
              "CUTLASS NVFP4 SM120 GemmUniversalAdapter failed to instantiate");

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED || CUTLASS_ARCH_MMA_SM121_SUPPORTED

} // namespace

extern "C" int qwen36_megakernel_nvfp4_gemm(
    const qwen36_nvfp4_gemm_spec_t *spec) {
  if (spec == nullptr) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  static_assert(CUTLASS_MAJOR >= 4,
                "Mirage megakernel requires CUTLASS 4.x (Blackwell FP4 path)");

  // Phase 1 step 2 stops here: the kernel types compile but we do not yet
  // launch. The next step bridges our existing per-tensor scale layout to
  // CUTLASS's `LayoutSFA` / `LayoutSFB` and wires up `gemm.run()`. Until
  // then the Rust dispatcher falls back to the cuBLASLt path on this
  // sentinel.
  return QWEN36_STATUS_NOT_IMPLEMENTED;
}
