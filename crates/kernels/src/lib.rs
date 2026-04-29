pub mod attention;
pub mod backend;
pub mod deltanet;
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
pub use memory::{CudaDeviceBuffer, cuda_synchronize};
pub use nvfp4_gemm::{CublasLtFp4ScaleMode, Nvfp4GemmPlan, Nvfp4GemmSpec};
pub use ops::{
    Bf16GemmSpec, Bf16MatVecSpec, Conv1dUpdateSpec, EmbeddingLookupSpec, GdnGateSpec,
    Nvfp4MatVecSpec, Nvfp4QuantizeSpec, Nvfp4RetileScalesSpec, RmsNormNvfp4QuantizeSpec,
    SigmoidGateSpec,
};
pub use rmsnorm::RmsNormSpec;
pub use rope::PartialRopeSpec;
pub use sampling::SamplingSpec;
pub use swiglu::SwiGluSpec;
