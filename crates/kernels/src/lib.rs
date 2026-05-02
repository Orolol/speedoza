pub mod attention;
pub mod backend;
pub mod deltanet;
#[cfg(feature = "cuda")]
pub mod graph;
pub mod memory;
pub mod nvfp4_gemm;
pub mod ops;
pub mod rmsnorm;
pub mod rope;
pub mod sampling;
pub mod swiglu;
pub mod turboquant;

pub use attention::{AttentionDecodeSpec, AttentionPrefillSpec, AttentionShape};
#[cfg(feature = "cuda")]
pub use backend::CudaBackend;
#[cfg(feature = "cuda")]
pub use backend::nvfp4_retile_scales;
pub use backend::{DevicePtr, KernelBackend, NoCudaBackend};
pub use deltanet::{DeltaNetDecodeSpec, DeltaNetPrefillSpec, DeltaNetShape};
#[cfg(feature = "cuda")]
pub use memory::{
    CudaDeviceBuffer, cuda_clear_l2_access_window, cuda_set_l2_access_window, cuda_synchronize,
};
pub use nvfp4_gemm::{CublasLtFp4ScaleMode, Nvfp4GemmPlan, Nvfp4GemmSpec};
pub use ops::{
    Bf16GemmSpec, Bf16MatVecSpec, Conv1dGdnGateFusedSpec, Conv1dPrefillSpec, Conv1dUpdateSpec,
    CopyStridedRowsSpec, EmbeddingLookupSpec, GdnGateSpec, Nvfp4MatVecSpec, Nvfp4QuantizeRowsSpec,
    Nvfp4QuantizeSpec, Nvfp4RetileScalesSpec, QProjDeinterleaveSpec, QProjSigmoidGateSpec,
    RmsNormNvfp4QuantizeSpec, SigmoidGateSpec, SigmoidGateStridedSpec,
};
pub use rmsnorm::RmsNormSpec;
pub use rope::PartialRopeSpec;
pub use sampling::SamplingSpec;
pub use swiglu::{SwiGluNvfp4QuantizeSpec, SwiGluSpec};
