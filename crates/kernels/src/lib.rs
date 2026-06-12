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
pub use backend::CudaBackend;
pub use backend::{DevicePtr, KernelBackend, NoCudaBackend};
#[cfg(feature = "cuda")]
pub use backend::{
    attention_decode_spec_abi_bytes, attention_decode_spec_abi_size,
    deltanet_decode_spec_abi_bytes, deltanet_decode_spec_abi_size, nvfp4_retile_scales,
};
pub use deltanet::{DeltaNetDecodeSpec, DeltaNetPrefillSpec, DeltaNetShape};
pub use drafter_attention::DrafterAttentionBlockSpec;
pub use interpreter::{
    InterpreterDep, InterpreterInstruction, InterpreterOpcode, InterpreterOpcodeSet,
    InterpreterProgram, InterpreterProgramSpec, interpreter_opcodes_enabled_from_env,
};
#[cfg(feature = "cuda")]
pub use memory::{
    CudaCounters, CudaDeviceBuffer, CudaDiagnostics, cuda_clear_l2_access_window,
    cuda_counters_read, cuda_counters_reset, cuda_diagnostics, cuda_set_l2_access_window,
    cuda_synchronize,
};
pub use nvfp4_gemm::{CublasLtFp4ScaleMode, Nvfp4GemmPlan, Nvfp4GemmSpec};
pub use ops::{
    Bf16GemmSpec, Bf16MatVecArgmaxRowsSpec, Bf16MatVecSpec, Conv1dGdnGateFusedSpec,
    Conv1dPrefillSpec, Conv1dUpdateSpec, CopyStridedRowsSpec, EmbeddingLookupSpec, GdnGateSpec,
    lm_head_top2_workspace_bytes, LmHeadFp8GemvSpec, LmHeadFp8QuantizeSpec, LmHeadTop2MarginSpec,
    Nvfp4MatVecSpec, Nvfp4QuantizeRowsSpec,
    Nvfp4QuantizeSpec, Nvfp4RetileScalesSpec, QProjDeinterleaveSpec, QProjSigmoidGateSpec,
    RmsNormNvfp4QuantizeSpec, SigmoidGateSpec, SigmoidGateStridedSpec,
};
pub use rmsnorm::RmsNormSpec;
pub use rope::PartialRopeSpec;
pub use sampling::{SamplingRowsSpec, SamplingSpec};
pub use swiglu::{SwiGluNvfp4QuantizeSpec, SwiGluSpec};
