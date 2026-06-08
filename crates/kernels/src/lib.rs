pub mod attention;
pub mod backend;
pub mod deltanet;
pub mod drafter_attention;
#[cfg(feature = "cuda")]
pub mod graph;
pub mod interpreter;
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
pub use backend::nvfp4_retile_scales;
#[cfg(feature = "cuda")]
pub use backend::CudaBackend;
pub use backend::{DevicePtr, KernelBackend, NoCudaBackend};
pub use deltanet::{DeltaNetDecodeSpec, DeltaNetPrefillSpec, DeltaNetShape};
pub use drafter_attention::DrafterAttentionBlockSpec;
pub use interpreter::{
    interpreter_opcodes_enabled_from_env, InterpreterDep, InterpreterInstruction,
    InterpreterOpcode, InterpreterOpcodeSet, InterpreterProgram, InterpreterProgramSpec,
};
#[cfg(feature = "cuda")]
pub use memory::{
    cuda_clear_l2_access_window, cuda_counters_read, cuda_counters_reset, cuda_diagnostics,
    cuda_set_l2_access_window, cuda_synchronize, CudaCounters, CudaDeviceBuffer, CudaDiagnostics,
};
pub use nvfp4_gemm::{CublasLtFp4ScaleMode, Nvfp4GemmPlan, Nvfp4GemmSpec};
pub use ops::{
    Bf16GemmSpec, Bf16MatVecSpec, Conv1dGdnGateFusedSpec, Conv1dPrefillSpec, Conv1dUpdateSpec,
    CopyStridedRowsSpec, EmbeddingLookupSpec, GdnGateSpec, MegakernelFullAttnStageBQProjSpec,
    MegakernelFullAttnStageESpec, MegakernelFullAttnStageF4Spec, Nvfp4MatVecSpec,
    Nvfp4QuantizeRowsSpec, Nvfp4QuantizeSpec, Nvfp4RetileScalesSpec, QProjDeinterleaveSpec,
    QProjSigmoidGateSpec, RmsNormNvfp4QuantizeSpec, SigmoidGateSpec, SigmoidGateStridedSpec,
};
pub use rmsnorm::RmsNormSpec;
pub use rope::PartialRopeSpec;
pub use sampling::{SamplingRowsSpec, SamplingSpec};
pub use swiglu::{SwiGluNvfp4QuantizeSpec, SwiGluSpec};
