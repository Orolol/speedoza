use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda")]
use std::mem::size_of;
#[cfg(feature = "cuda")]
use std::sync::OnceLock;

#[cfg(feature = "cuda")]
use qwen36_fp4_core::TensorInfo;
use qwen36_fp4_core::{CoreError, KvCacheDtype, ModelLayout, ModelTopology, Result};
#[cfg(feature = "cuda")]
use qwen36_fp4_kernels::{
    AttentionDecodeSpec, AttentionPrefillSpec, AttentionShape, Bf16GemmSpec, Bf16MatVecSpec,
    Conv1dGdnGateFusedSpec, Conv1dPrefillSpec, CopyStridedRowsSpec, CublasLtFp4ScaleMode,
    CudaBackend, CudaDeviceBuffer, DeltaNetDecodeSpec, DeltaNetPrefillSpec, DeltaNetShape,
    DevicePtr, EmbeddingLookupSpec, Fp8MatVecSpec, Fp8QuantizeRowsSpec, GdnGateSpec,
    InterpreterOpcode, InterpreterOpcodeSet, InterpreterProgramSpec, Nvfp4GemmSpec,
    Nvfp4QuantizeRowsSpec, Nvfp4QuantizeSpec, PartialRopeSpec, QProjDeinterleaveSpec,
    QProjSigmoidGateSpec, RmsNormNvfp4QuantizeSpec, RmsNormSpec, SamplingRowsSpec, SamplingSpec,
    SwiGluNvfp4QuantizeSpec, SwiGluSpec, attention_decode_spec_abi_bytes,
    attention_decode_spec_abi_size, deltanet_decode_spec_abi_bytes, deltanet_decode_spec_abi_size,
    interpreter_opcodes_enabled_from_env,
};
use qwen36_fp4_kernels::{KernelBackend, NoCudaBackend};
#[cfg(feature = "cuda")]
use qwen36_fp4_loader::MappedModel;

use crate::cuda_graph::CudaGraphPlan;
#[cfg(feature = "cuda")]
use crate::gpu::{
    ATTN_MIN_SPLIT_TIMESTEPS_PER_BLOCK, GpuForwardBuffers, GpuPrefillBuffers, GpuRuntimeBuffers,
    GpuWeightStore, LinearAttnInProjFused, LinearAttnInProjFusedStore, MlpFusedLayer,
    MlpFusedStore, MtpKvSnapshotLayout,
};
#[cfg(feature = "cuda")]
use crate::interpreter_compile::{
    DecodeInterpreterAttentionParams, DecodeInterpreterConv1dGdnGateFusedParams,
    DecodeInterpreterDeltaNetParams, DecodeInterpreterFullAttentionInputLayerParams,
    DecodeInterpreterFullAttentionLayerParams, DecodeInterpreterFullTransformerLayerParams,
    DecodeInterpreterLinearAttentionInputLayerParams, DecodeInterpreterLinearAttentionLayerParams,
    DecodeInterpreterLinearAttentionPostInProjParams, DecodeInterpreterLinearAttentionTailParams,
    DecodeInterpreterLinearTransformerLayerParams, DecodeInterpreterLogitsParams,
    DecodeInterpreterMlpParams, DecodeInterpreterNormMlpParams, DecodeInterpreterNvfp4GemvParams,
    DecodeInterpreterNvfp4QuantizeParams, DecodeInterpreterProgram,
    DecodeInterpreterQProjDeinterleaveParams, DecodeInterpreterQProjSigmoidGateParams,
    DecodeInterpreterRmsNormBf16Params, DecodeInterpreterRmsNormNvfp4QuantParams,
    DecodeInterpreterRopeAttentionParams, DecodeInterpreterRopeParams,
    DecodeInterpreterSwiGluBf16Params, instructions_as_bytes,
};
use crate::kv_cache::KvCachePlan;
use crate::state::{DeltaNetStatePlan, RuntimeState};
use crate::weights::ModelWeightsManifest;
#[cfg(feature = "cuda")]
use crate::weights::{
    CommonLayerWeights, FullAttentionLayerWeights, LayerWeights, LinearAttentionLayerWeights,
    LinearWeightBinding,
};

#[cfg(feature = "cuda")]
const MTP_MAX_DRAFT_TOKENS: usize = 8;
#[cfg(feature = "cuda")]
const MTP_GRAPH_BUNDLE_U32S: usize = 24;
#[cfg(feature = "cuda")]
const MTP_GRAPH_VERIFIED_BASE: usize = 5;
#[cfg(feature = "cuda")]
const MTP_GRAPH_NEXT_DRAFT_BASE: usize = 9;

#[cfg(feature = "cuda")]
fn cuda_env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

#[cfg(feature = "cuda")]
fn cuda_env_bool_value(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .and_then(|value| match value.as_str() {
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON" => Some(true),
            "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF" => Some(false),
            _ => None,
        })
}

#[cfg(feature = "cuda")]
fn cuda_env_bool(name: &str) -> bool {
    cuda_env_bool_value(name).unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn cuda_env_bool_default_true(name: &str) -> bool {
    cuda_env_bool_value(name).unwrap_or(true)
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeInterpreterDecodeMode {
    Off,
    On,
    Auto,
}

#[cfg(feature = "cuda")]
fn decode_interpreter_decode_mode() -> DecodeInterpreterDecodeMode {
    use std::sync::OnceLock;
    static MODE: OnceLock<DecodeInterpreterDecodeMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let Ok(value) = std::env::var("QWEN36_INTERPRETER_DECODE") else {
            return DecodeInterpreterDecodeMode::Auto;
        };
        match value.as_str() {
            "auto" | "AUTO" | "Auto" => DecodeInterpreterDecodeMode::Auto,
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON" => DecodeInterpreterDecodeMode::On,
            "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF" => {
                DecodeInterpreterDecodeMode::Off
            }
            other => panic!("QWEN36_INTERPRETER_DECODE must be 0, 1, or auto, got {other:?}"),
        }
    })
}

#[cfg(feature = "cuda")]
fn decode_interpreter_decode_enabled_for_mtp(mtp_speculative_tokens: usize) -> bool {
    match decode_interpreter_decode_mode() {
        DecodeInterpreterDecodeMode::Off => false,
        DecodeInterpreterDecodeMode::On => true,
        DecodeInterpreterDecodeMode::Auto => mtp_speculative_tokens > 0,
    }
}

#[cfg(feature = "cuda")]
fn decode_interpreter_opcodes_enabled() -> InterpreterOpcodeSet {
    use std::sync::OnceLock;
    static OPCODES: OnceLock<InterpreterOpcodeSet> = OnceLock::new();
    *OPCODES.get_or_init(interpreter_opcodes_enabled_from_env)
}

#[cfg(feature = "cuda")]
fn decode_interpreter_gate_enabled(
    master_enabled: bool,
    explicit_env: &str,
    required_opcodes: &[InterpreterOpcode],
) -> bool {
    if !master_enabled && !cuda_env_bool(explicit_env) {
        return false;
    }
    let opcodes = decode_interpreter_opcodes_enabled();
    required_opcodes
        .iter()
        .copied()
        .all(|opcode| opcodes.contains(opcode))
}

/// Bit 0 of `InterpreterProgramSpec::flags` controls L2 weight prefetch
/// lookahead in the dispatch loop (see
/// `kernels-cuda/interpreter/prefetch.cuh`). Default off until the
/// implementation reliably beats the no-prefetch baseline on MTP=0.
#[cfg(feature = "cuda")]
fn interpreter_launch_flags() -> u32 {
    use std::sync::OnceLock;
    static FLAGS: OnceLock<u32> = OnceLock::new();
    *FLAGS.get_or_init(|| {
        let mut bits = 0u32;
        if cuda_env_bool("QWEN36_INTERPRETER_PREFETCH") {
            bits |= 1;
        }
        bits
    })
}

#[cfg(feature = "cuda")]
fn productive_spin_enabled() -> bool {
    cuda_env_bool("QWEN36_PRODUCTIVE_SPIN")
}

/// Gate for the first runtime-backed decode-interpreter slice. This replaces
/// only the final RMSNorm + lm_head logits pair, stays outside CUDA Graph
/// capture for now, and also respects the opcode allow-list used by op-level
/// bring-up.
#[cfg(feature = "cuda")]
fn decode_interpreter_logits_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if !cuda_env_bool("QWEN36_INTERPRETER_LOGITS") {
            return false;
        }
        let opcodes = decode_interpreter_opcodes_enabled();
        opcodes.contains(InterpreterOpcode::RmsNormNvfp4Quant)
            && opcodes.contains(InterpreterOpcode::LmHeadTiled)
    })
}

#[cfg(feature = "cuda")]
fn decode_interpreter_mlp_enabled(master_enabled: bool) -> bool {
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_MLP",
        &[
            InterpreterOpcode::Nvfp4GemvPair,
            InterpreterOpcode::Nvfp4Gemv,
            InterpreterOpcode::SwiGluNvfp4Quant,
        ],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_mlp_chunked_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let opcodes = decode_interpreter_opcodes_enabled();
        cuda_env_bool("QWEN36_INTERPRETER_MLP_CHUNKED")
            && opcodes.contains(InterpreterOpcode::SwiGluNvfp4QuantChunk)
            && opcodes.contains(InterpreterOpcode::Nvfp4GemvChunkAccum)
    })
}

#[cfg(feature = "cuda")]
fn decode_interpreter_norm_mlp_enabled(master_enabled: bool) -> bool {
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_NORM_MLP",
        &[
            InterpreterOpcode::RmsNormNvfp4Quant,
            InterpreterOpcode::Nvfp4GemvPair,
            InterpreterOpcode::Nvfp4Gemv,
            InterpreterOpcode::SwiGluNvfp4Quant,
        ],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_rmsnorm_enabled(master_enabled: bool) -> bool {
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_RMSNORM",
        &[InterpreterOpcode::RmsNormNvfp4Quant],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_deltanet_enabled(master_enabled: bool) -> bool {
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_DELTANET",
        &[InterpreterOpcode::DeltaNetRecur],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_attention_enabled(master_enabled: bool) -> bool {
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_ATTN",
        &[InterpreterOpcode::AttnDecodeFull],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_rope_enabled(master_enabled: bool) -> bool {
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_ROPE",
        &[InterpreterOpcode::RopePartial],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_full_attention_enabled(master_enabled: bool) -> bool {
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_FULL_ATTN",
        &[
            InterpreterOpcode::RopePartial,
            InterpreterOpcode::AttnDecodeFull,
        ],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_full_attention_layer_enabled(master_enabled: bool) -> bool {
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_FULL_ATTN_LAYER",
        &[
            InterpreterOpcode::Nvfp4Gemv,
            InterpreterOpcode::QProjDeinterleave,
            InterpreterOpcode::RmsNormBf16,
            InterpreterOpcode::RopePartial,
            InterpreterOpcode::AttnDecodeFull,
            InterpreterOpcode::QProjSigmoidGate,
            InterpreterOpcode::Nvfp4Quantize,
        ],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_full_attention_input_layer_enabled(master_enabled: bool) -> bool {
    if cuda_env_bool("QWEN36_INTERPRETER_FULL_ATTN_INPUT_LAYER_DISABLE") {
        return false;
    }
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_FULL_ATTN_INPUT_LAYER",
        &[
            InterpreterOpcode::RmsNormNvfp4Quant,
            InterpreterOpcode::Nvfp4Gemv,
            InterpreterOpcode::QProjDeinterleave,
            InterpreterOpcode::RmsNormBf16,
            InterpreterOpcode::RopePartial,
            InterpreterOpcode::AttnDecodeFull,
            InterpreterOpcode::QProjSigmoidGate,
            InterpreterOpcode::Nvfp4Quantize,
        ],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_full_transformer_layer_enabled(master_enabled: bool) -> bool {
    if cuda_env_bool("QWEN36_INTERPRETER_FULL_TRANSFORMER_LAYER_DISABLE") {
        return false;
    }
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_FULL_TRANSFORMER_LAYER",
        &[
            InterpreterOpcode::RmsNormNvfp4Quant,
            InterpreterOpcode::Nvfp4Gemv,
            InterpreterOpcode::QProjDeinterleave,
            InterpreterOpcode::RmsNormBf16,
            InterpreterOpcode::RopePartial,
            InterpreterOpcode::AttnDecodeFull,
            InterpreterOpcode::QProjSigmoidGate,
            InterpreterOpcode::Nvfp4Quantize,
            InterpreterOpcode::Nvfp4GemvPair,
            InterpreterOpcode::SwiGluNvfp4Quant,
        ],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_linear_attention_tail_enabled(master_enabled: bool) -> bool {
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_LINEAR_ATTN_TAIL",
        &[
            InterpreterOpcode::RmsNormBf16,
            InterpreterOpcode::SwiGluBf16,
            InterpreterOpcode::Nvfp4Quantize,
            InterpreterOpcode::Nvfp4Gemv,
        ],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_linear_attention_post_inproj_enabled(master_enabled: bool) -> bool {
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_LINEAR_ATTN_POST_INPROJ",
        &[
            InterpreterOpcode::Conv1dGdnGateFused,
            InterpreterOpcode::DeltaNetRecur,
            InterpreterOpcode::RmsNormBf16,
            InterpreterOpcode::SwiGluBf16,
            InterpreterOpcode::Nvfp4Quantize,
            InterpreterOpcode::Nvfp4Gemv,
        ],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_linear_attention_layer_enabled(master_enabled: bool) -> bool {
    if cuda_env_bool("QWEN36_INTERPRETER_LINEAR_ATTN_LAYER_DISABLE") {
        return false;
    }
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_LINEAR_ATTN_LAYER",
        &[
            InterpreterOpcode::Nvfp4Gemv,
            InterpreterOpcode::Conv1dGdnGateFused,
            InterpreterOpcode::DeltaNetRecur,
            InterpreterOpcode::RmsNormBf16,
            InterpreterOpcode::SwiGluBf16,
            InterpreterOpcode::Nvfp4Quantize,
        ],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_linear_attention_input_layer_enabled(master_enabled: bool) -> bool {
    if cuda_env_bool("QWEN36_INTERPRETER_LINEAR_ATTN_INPUT_LAYER_DISABLE") {
        return false;
    }
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_LINEAR_ATTN_INPUT_LAYER",
        &[
            InterpreterOpcode::RmsNormNvfp4Quant,
            InterpreterOpcode::Nvfp4Gemv,
            InterpreterOpcode::Conv1dGdnGateFused,
            InterpreterOpcode::DeltaNetRecur,
            InterpreterOpcode::RmsNormBf16,
            InterpreterOpcode::SwiGluBf16,
            InterpreterOpcode::Nvfp4Quantize,
        ],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_linear_transformer_layer_enabled(master_enabled: bool) -> bool {
    if cuda_env_bool("QWEN36_INTERPRETER_LINEAR_TRANSFORMER_LAYER_DISABLE") {
        return false;
    }
    decode_interpreter_gate_enabled(
        master_enabled,
        "QWEN36_INTERPRETER_LINEAR_TRANSFORMER_LAYER",
        &[
            InterpreterOpcode::RmsNormNvfp4Quant,
            InterpreterOpcode::Nvfp4Gemv,
            InterpreterOpcode::Conv1dGdnGateFused,
            InterpreterOpcode::DeltaNetRecur,
            InterpreterOpcode::RmsNormBf16,
            InterpreterOpcode::SwiGluBf16,
            InterpreterOpcode::Nvfp4Quantize,
            InterpreterOpcode::Nvfp4GemvPair,
            InterpreterOpcode::SwiGluNvfp4Quant,
        ],
    )
}

#[cfg(feature = "cuda")]
fn decode_interpreter_nvfp4_gemv_supports(m: usize, k: usize) -> bool {
    const GEMV_K_ALIGNMENT: usize = 1024;
    m > 0 && k > 0 && m % 16 == 0 && k % GEMV_K_ALIGNMENT == 0
}

#[cfg(feature = "cuda")]
fn decode_interpreter_mlp_supports(hidden: usize, intermediate: usize) -> bool {
    hidden > 0
        && intermediate > 0
        && hidden % 16 == 0
        && intermediate % 16 == 0
        && decode_interpreter_nvfp4_gemv_supports(intermediate, hidden)
        && decode_interpreter_nvfp4_gemv_supports(hidden, intermediate)
}

#[cfg(feature = "cuda")]
fn productive_spin_ctas() -> i32 {
    cuda_env_usize("QWEN36_PRODUCTIVE_SPIN_CTAS")
        .map(|v| v as i32)
        .filter(|&v| (1..=1024).contains(&v))
        .unwrap_or(128)
}

#[cfg(feature = "cuda")]
fn mtp_recurrent_snapshot_enabled() -> bool {
    std::env::var("QWEN36_MTP_SNAPSHOT_RECURRENT")
        .ok()
        .is_none_or(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
}

#[cfg(feature = "cuda")]
fn mtp_assume_accept_enabled() -> bool {
    cuda_env_bool("QWEN36_MTP_ASSUME_ACCEPT")
}

#[cfg(feature = "cuda")]
fn mtp_device_chain_enabled() -> bool {
    cuda_env_bool("QWEN36_MTP_DEVICE_CHAIN")
}

#[cfg(feature = "cuda")]
fn mtp_device_chain_batch() -> usize {
    cuda_env_usize("QWEN36_MTP_DEVICE_CHAIN_BATCH")
        .unwrap_or(2)
        .clamp(1, 64)
}

#[cfg(feature = "cuda")]
fn mtp_batched_lm_head_enabled() -> bool {
    !cuda_env_bool("QWEN36_MTP_BATCH_LM_HEAD_DISABLE")
}

#[cfg(feature = "cuda")]
fn mtp_tree_disable_enabled() -> bool {
    cuda_env_bool("QWEN36_MTP_TREE_DISABLE")
}

/// Reject-recovery graph (re-prefill + next-draft chain in one capture).
/// Default ON; `QWEN36_MTP_RECOVER_GRAPH=0` falls back to the host-launched
/// recovery path (bisect/kill switch).
#[cfg(feature = "cuda")]
fn mtp_recover_graph_enabled() -> bool {
    cuda_env_bool_default_true("QWEN36_MTP_RECOVER_GRAPH")
}

#[cfg(feature = "cuda")]
fn cuda_env_workspace_bytes() -> usize {
    cuda_env_usize("QWEN36_CUDA_WORKSPACE_BYTES")
        .or_else(|| {
            cuda_env_usize("QWEN36_CUDA_WORKSPACE_MIB").and_then(|mib| mib.checked_mul(1024 * 1024))
        })
        .unwrap_or(256 * 1024 * 1024)
}

#[cfg(feature = "cuda")]
const CUDA_PREFILL_CAPACITY_SHORT_CONTEXT: usize = 8192;
#[cfg(feature = "cuda")]
const CUDA_PREFILL_CAPACITY_LONG_CONTEXT: usize = 2048;
#[cfg(feature = "cuda")]
const CUDA_PREFILL_CAPACITY_SHORT_CONTEXT_MAX: usize = 32768;

#[cfg(feature = "cuda")]
fn cuda_prefill_capacity(max_context: usize) -> usize {
    let auto_capacity = if max_context <= CUDA_PREFILL_CAPACITY_SHORT_CONTEXT_MAX {
        CUDA_PREFILL_CAPACITY_SHORT_CONTEXT
    } else {
        CUDA_PREFILL_CAPACITY_LONG_CONTEXT
    };
    cuda_env_usize("QWEN36_PREFILL_CAPACITY")
        .unwrap_or(auto_capacity)
        .clamp(1, max_context.max(1))
}

#[cfg(feature = "cuda")]
const CUDA_LONG_CONTEXT_AUTO_MIN_CONTEXT: usize = 8192;

#[cfg(feature = "cuda")]
fn cuda_long_context_auto_min_context() -> usize {
    cuda_env_usize("QWEN36_LONG_CONTEXT_AUTO_MIN_CONTEXT")
        .unwrap_or(CUDA_LONG_CONTEXT_AUTO_MIN_CONTEXT)
}

#[cfg(feature = "cuda")]
const CUDA_DECODE_ATTENTION_BUCKET_MIN_CONTEXT: usize = 8192;

#[cfg(feature = "cuda")]
fn cuda_decode_attention_bucket_min_context() -> usize {
    cuda_env_usize("QWEN36_DECODE_ATTENTION_BUCKET_MIN_CONTEXT")
        .unwrap_or(CUDA_DECODE_ATTENTION_BUCKET_MIN_CONTEXT)
        .max(1)
}

#[cfg(feature = "cuda")]
fn cuda_long_context_mode_enabled(max_context: usize) -> bool {
    cuda_env_bool_value("QWEN36_LONG_CONTEXT_MODE")
        .unwrap_or_else(|| max_context >= cuda_long_context_auto_min_context())
}

#[cfg(feature = "cuda")]
fn cuda_mlp_fused_enabled(max_context: usize) -> bool {
    !cuda_long_context_mode_enabled(max_context) && !cuda_env_bool("QWEN36_DISABLE_MLP_FUSED")
}

#[cfg(feature = "cuda")]
fn cuda_linear_attn_fused_enabled(max_context: usize) -> bool {
    !cuda_long_context_mode_enabled(max_context)
        && !cuda_env_bool("QWEN36_DISABLE_LINEAR_ATTN_FUSED")
}

#[cfg(feature = "cuda")]
fn cuda_prefill_fused_mlp_enabled(max_context: usize) -> bool {
    cuda_mlp_fused_enabled(max_context) && cuda_env_bool("QWEN36_PREFILL_FUSED_MLP")
}

#[cfg(feature = "cuda")]
fn cuda_prefill_fused_linear_attn_enabled(max_context: usize) -> bool {
    cuda_linear_attn_fused_enabled(max_context)
        && !cuda_env_bool("QWEN36_PREFILL_FUSED_LINEAR_ATTN_DISABLE")
}

#[cfg(feature = "cuda")]
fn cuda_deltanet_chunked_prefill_enabled() -> bool {
    cuda_env_bool_value("QWEN36_DELTANET_CHUNKED_PREFILL").unwrap_or(true)
}

/// Default `max_context` for engines built from `EngineConfig::default()`.
///
/// Deliberately far below the checkpoint's 262_144 `max_position_embeddings`:
/// a full-length KV cache is a multi-GB allocation (~17 GB at BF16) that
/// silently eats the 32 GB 5090 before weights and fused stores land. Long
/// contexts are an explicit opt-in — set `EngineConfig.max_context` directly
/// or export `QWEN36_MAX_CONTEXT` (validated against the model ceiling at
/// engine construction).
pub const DEFAULT_MAX_CONTEXT: usize = 16_384;

/// The shipped checkpoint's `max_position_embeddings`. Engine construction
/// rejects any `max_context` above this; the per-model ceiling from the
/// loaded topology is the authoritative bound.
pub const MODEL_MAX_CONTEXT: usize = 262_144;

fn validate_max_context(config: &EngineConfig, topology: &ModelTopology) -> Result<()> {
    if config.max_context == 0 {
        return Err(CoreError::Runtime(
            "max_context must be at least 1".to_owned(),
        ));
    }
    let ceiling = topology.max_position_embeddings;
    if config.max_context > ceiling {
        return Err(CoreError::Runtime(format!(
            "max_context {} exceeds the model's max_position_embeddings {ceiling}; long \
             contexts are an explicit opt-in up to that ceiling (set \
             EngineConfig.max_context or QWEN36_MAX_CONTEXT)",
            config.max_context
        )));
    }
    Ok(())
}

fn default_max_context() -> usize {
    let Ok(raw) = std::env::var("QWEN36_MAX_CONTEXT") else {
        return DEFAULT_MAX_CONTEXT;
    };
    let parsed: usize = raw
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("QWEN36_MAX_CONTEXT must be a positive integer, got {raw:?}"));
    assert!(
        (1..=MODEL_MAX_CONTEXT).contains(&parsed),
        "QWEN36_MAX_CONTEXT must be in 1..={MODEL_MAX_CONTEXT}, got {parsed}"
    );
    parsed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// KV-cache / position capacity for this engine. Defaults to
    /// [`DEFAULT_MAX_CONTEXT`] (overridable via `QWEN36_MAX_CONTEXT`);
    /// the model ceiling of [`MODEL_MAX_CONTEXT`] is explicit opt-in.
    pub max_context: usize,
    pub kv_cache_dtype: KvCacheDtype,
    pub turboquant: bool,
    pub mtp_speculative_tokens: usize,
    pub cuda_graphs: CudaGraphPlan,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_context: default_max_context(),
            kv_cache_dtype: KvCacheDtype::Fp8,
            turboquant: true,
            mtp_speculative_tokens: 0,
            cuda_graphs: CudaGraphPlan::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardOutput {
    pub logits_device_ptr: u64,
    pub produced_tokens: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryItem {
    pub name: String,
    pub bytes: u64,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryGroup {
    pub total_bytes: u64,
    pub items: Vec<GpuMemoryItem>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryReport {
    pub total_reported_bytes: u64,
    pub weights: GpuMemoryGroup,
    pub runtime: GpuMemoryGroup,
    pub forward: GpuMemoryGroup,
    pub prefill: GpuMemoryGroup,
    pub fused: GpuMemoryGroup,
    pub max_context: usize,
    pub prefill_capacity: usize,
    pub kv_cache_dtype: KvCacheDtype,
    pub mtp_speculative_tokens: usize,
    pub long_context_mode: bool,
    pub long_context_auto_min_context: usize,
    pub mlp_fused_enabled: bool,
    pub linear_attn_fused_enabled: bool,
    pub prefill_fused_linear_attn_enabled: bool,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MtpVerifyResult {
    pub accepted: bool,
    pub verified_token: u32,
    pub next_token: Option<u32>,
    pub next_draft_token: Option<u32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpMultiVerifyResult {
    pub accepted_drafts: usize,
    pub rejected: bool,
    pub next_token: Option<u32>,
    pub next_draft_tokens: Vec<u32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpDeviceChainResult {
    pub cycles: usize,
    pub generated_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub next_token: u32,
    pub next_draft_tokens: Vec<u32>,
}

/// Hook the prefill loop fires after each layer's `input_layernorm`,
/// passing the engine's prefill `layer_idx` (= the transformers
/// `hidden_states[layer_idx]` index), a device pointer to the freshly-
/// updated residual buffer (`[tokens, hidden]` BF16), and `tokens`.
/// Used by `crates/drafter` to scatter target hidden states into the
/// DFlash drafter's conditioning buffer. The hook is invoked
/// synchronously on the active CUDA stream; implementations must not
/// block on host signals.
#[cfg(feature = "cuda")]
pub type DrafterHiddenCaptureHook =
    std::sync::Arc<dyn Fn(usize, DevicePtr, usize) -> Result<()> + Send + Sync>;

pub struct Engine<B: KernelBackend = NoCudaBackend> {
    pub topology: ModelTopology,
    pub config: EngineConfig,
    pub state: RuntimeState,
    pub weights: Option<ModelWeightsManifest>,
    #[cfg(feature = "cuda")]
    pub gpu_weights: Option<GpuWeightStore>,
    #[cfg(feature = "cuda")]
    pub gpu_buffers: Option<GpuRuntimeBuffers>,
    #[cfg(feature = "cuda")]
    pub gpu_forward: Option<GpuForwardBuffers>,
    #[cfg(feature = "cuda")]
    pub gpu_prefill: Option<GpuPrefillBuffers>,
    /// Pre-concatenated gate_proj + up_proj NVFP4 weights for the decode MLP
    /// fast path (one cuBLASLt FP4 GEMM instead of two). Only valid when
    /// every layer's gate/up share `tensor_scale` and `input_scale`, which
    /// holds for the shipped Qwen3.6 NVFP4 checkpoint.
    #[cfg(feature = "cuda")]
    pub mlp_fused: Option<MlpFusedStore>,
    /// Pre-concatenated DeltaNet in_proj_qkv/_b/_a/_z NVFP4 weights for the
    /// decode linear-attention fast path (one combined cuBLASLt FP4 GEMM
    /// instead of four). Indexed by global layer index; `None` for full-attn
    /// layers.
    #[cfg(feature = "cuda")]
    pub linear_attn_in_proj_fused: Option<LinearAttnInProjFusedStore>,
    /// Capture-mode CUDA stream + instantiated decode-and-sample graph. When
    /// `Some`, decode kernels read the current position from `forward.position_i32`
    /// instead of a host scalar so the same graph can replay across iterations.
    #[cfg(feature = "cuda")]
    decode_graph: Option<DecodeGraphState>,
    /// Captured MTP reject-recovery graphs, keyed by their `MtpRecover`
    /// kind. Kept in a separate cache from `decode_graph` so a rejection
    /// (which alternates verify-graph / recover-graph launches within one
    /// cycle) never forces a re-capture of either side.
    #[cfg(feature = "cuda")]
    mtp_recover_graphs: Vec<DecodeGraphState>,
    /// Per-layer interpreter programs uploaded before CUDA graph capture.
    /// Graph replay cannot depend on host-side instruction/spec copies into a
    /// single scratch buffer because every graph node would then observe the
    /// same final contents. These entries give each fused layer stable device
    /// storage for instructions, counters, and its ABI spec.
    #[cfg(feature = "cuda")]
    decode_interpreter_graph_programs: Option<DecodeInterpreterGraphPrograms>,
    /// Secondary prefetch stream + reusable event pool. Lazily constructed on
    /// first use by the productive-spin path; `None` until then.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    decode_aux: Option<DecodeAuxStreams>,
    /// Optional DFlash drafter hidden-state capture hook (Phase E).
    /// Fired once per prefill layer after the input_layernorm has
    /// produced the post-fused-residual tensor in `prefill.residual`.
    /// `None` is the default; the speculative controller arms this
    /// before each prefill call when DFlash is active.
    #[cfg(feature = "cuda")]
    drafter_hidden_capture: Option<DrafterHiddenCaptureHook>,
    /// lm_head quantized to FP8 e4m3 with per-row f32 scales at load time
    /// (default ON; kill: `QWEN36_LM_HEAD_FP8=0`). When `Some`, every
    /// lm_head matvec/gemm-rows consumer routes through `fp8_matvec` so all
    /// MTP modes see the same logits (parity-floor requirement). Halves the
    /// 1.65 ms/token BF16 lm_head read. Offline probe: 0/27 argmax flips.
    #[cfg(feature = "cuda")]
    lm_head_fp8: Option<LmHeadFp8>,
    backend: B,
}

#[cfg(feature = "cuda")]
struct LmHeadFp8 {
    tensor_name: String,
    weight_e4m3: CudaDeviceBuffer,
    row_scale_f32: CudaDeviceBuffer,
}

#[cfg(feature = "cuda")]
struct DecodeGraphState {
    kind: DecodeGraphKind,
    attention_context_limit: usize,
    stream: qwen36_fp4_kernels::graph::OwnedCudaStream,
    exec: qwen36_fp4_kernels::graph::CudaGraphExec,
    raw_graph: qwen36_fp4_kernels::graph::CudaGraph,
}

#[cfg(feature = "cuda")]
struct DecodeInterpreterGraphLayerProgram {
    instructions: CudaDeviceBuffer,
    counters_i32: CudaDeviceBuffer,
    _spec: Option<CudaDeviceBuffer>,
    instruction_count: usize,
    counter_count: usize,
}

#[cfg(feature = "cuda")]
struct DecodeInterpreterGraphPrograms {
    attention_context_limit: usize,
    position_device_i32: DevicePtr,
    layers: Vec<Option<DecodeInterpreterGraphLayerProgram>>,
}

#[cfg(feature = "cuda")]
impl DecodeInterpreterGraphLayerProgram {
    fn upload(program: DecodeInterpreterProgram, spec: Option<CudaDeviceBuffer>) -> Result<Self> {
        let instruction_bytes = instructions_as_bytes(&program.program.instructions);
        let instructions = CudaDeviceBuffer::alloc(instruction_bytes.len())?;
        instructions.copy_from_host(instruction_bytes)?;
        let counters_i32 =
            CudaDeviceBuffer::zeroed(program.program.counter_count * size_of::<i32>())?;
        Ok(Self {
            instructions,
            counters_i32,
            _spec: spec,
            instruction_count: program.program.instructions.len(),
            counter_count: program.program.counter_count,
        })
    }

    fn run<B: KernelBackend>(&self, backend: &B) -> Result<()> {
        self.counters_i32.memset_async(0)?;
        backend.interpreter_decode_sm120(&InterpreterProgramSpec {
            instructions: self.instructions.ptr(),
            instruction_count: self.instruction_count,
            counters_i32: self.counters_i32.ptr(),
            counter_count: self.counter_count,
            cta_count: 0,
            flags: interpreter_launch_flags(),
        })
    }
}

/// Auxiliary CUDA stream + event pool used by the decode hot path to overlap
/// productive-spin L2 prefetch with the main stream — reusable cross-stream
/// fork/join infrastructure. Lifetime spans the engine's CUDA-active
/// session: the prefetch stream is registered with the kernel library on
/// construction and unregistered on Drop so dispatches can't chase a dangling
/// `cudaStream_t`.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
struct DecodeAuxStreams {
    prefetch_stream: qwen36_fp4_kernels::graph::OwnedCudaStream,
    events: Vec<qwen36_fp4_kernels::graph::OwnedCudaEvent>,
    next_event: std::cell::Cell<usize>,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
impl DecodeAuxStreams {
    fn new(event_pool_size: usize) -> Result<Self> {
        let stream = qwen36_fp4_kernels::graph::CudaStream::create()?;
        // Register with the kernel library so kernel-side dispatchers can
        // pick it up via `qwen36_internal_prefetch_stream()`.
        qwen36_fp4_kernels::graph::set_prefetch_stream(stream.handle());
        let mut events = Vec::with_capacity(event_pool_size);
        for _ in 0..event_pool_size {
            events.push(qwen36_fp4_kernels::graph::CudaEvent::create()?);
        }
        Ok(Self {
            prefetch_stream: stream,
            events,
            next_event: std::cell::Cell::new(0),
        })
    }

    fn prefetch_stream(&self) -> qwen36_fp4_kernels::graph::CudaStream {
        self.prefetch_stream.handle()
    }

    /// Round-robin a sync token from the event pool. Callers must record and
    /// wait on the returned event before the pool wraps around (pool size is
    /// chosen at construction to make that easy).
    fn next_event(&self) -> qwen36_fp4_kernels::graph::CudaEvent {
        let idx = self.next_event.get();
        self.next_event.set((idx + 1) % self.events.len());
        self.events[idx].handle()
    }
}

#[cfg(feature = "cuda")]
impl Drop for DecodeAuxStreams {
    fn drop(&mut self) {
        // Clear the kernel-lib registration before the OwnedCudaStream Drop
        // frees the underlying cudaStream_t.
        qwen36_fp4_kernels::graph::set_prefetch_stream(qwen36_fp4_kernels::graph::CudaStream::NULL);
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeGraphKind {
    Decode,
    MtpDecodeOne,
    MtpVerifyOne,
    MtpVerifyMulti {
        drafts: usize,
        assume_accept: bool,
        batched_lm_head: bool,
        device_chain: bool,
        device_chain_batch: usize,
    },
    /// Reject-recovery: re-prefill of the committed tokens + the full
    /// next-draft MTP chain, all in one graph (one shape per
    /// (committed, drafts) pair, cached in `mtp_recover_graphs`).
    MtpRecover {
        committed: usize,
        drafts: usize,
    },
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct Nvfp4ActivationQuant<'a> {
    in_features: usize,
    input_scale: &'a TensorInfo,
}

#[cfg(feature = "cuda")]
impl Drop for DecodeGraphState {
    fn drop(&mut self) {
        // The graph_exec and raw graph must be torn down before the owning
        // stream goes away. We swallow errors here because Drop has no return
        // path; surfacing them would mask the original error that triggered
        // engine shutdown.
        let _ = qwen36_fp4_kernels::graph::destroy_graph_exec(self.exec);
        let _ = qwen36_fp4_kernels::graph::destroy_graph(self.raw_graph);
        // Reset the global active stream so any kernels run after engine
        // teardown go back to the legacy default stream.
        qwen36_fp4_kernels::graph::set_active_stream(qwen36_fp4_kernels::graph::CudaStream::NULL);
    }
}

impl Engine<NoCudaBackend> {
    pub fn no_cuda(layout: &ModelLayout, config: EngineConfig) -> Self {
        Self::new(layout.topology.clone(), config, NoCudaBackend)
    }

    pub fn no_cuda_with_weights(layout: &ModelLayout, config: EngineConfig) -> Result<Self> {
        Self::from_layout(layout, config, NoCudaBackend)
    }
}

impl<B: KernelBackend> Engine<B> {
    pub fn new(topology: ModelTopology, config: EngineConfig, backend: B) -> Self {
        let kv_cache = KvCachePlan::new(&topology, config.max_context, config.kv_cache_dtype);
        let deltanet = DeltaNetStatePlan::new(&topology);
        let state = RuntimeState::new(kv_cache, deltanet);
        Self {
            topology,
            config,
            state,
            weights: None,
            #[cfg(feature = "cuda")]
            gpu_weights: None,
            #[cfg(feature = "cuda")]
            gpu_buffers: None,
            #[cfg(feature = "cuda")]
            gpu_forward: None,
            #[cfg(feature = "cuda")]
            gpu_prefill: None,
            #[cfg(feature = "cuda")]
            mlp_fused: None,
            #[cfg(feature = "cuda")]
            linear_attn_in_proj_fused: None,
            #[cfg(feature = "cuda")]
            decode_graph: None,
            #[cfg(feature = "cuda")]
            mtp_recover_graphs: Vec::new(),
            #[cfg(feature = "cuda")]
            decode_interpreter_graph_programs: None,
            #[cfg(feature = "cuda")]
            decode_aux: None,
            #[cfg(feature = "cuda")]
            drafter_hidden_capture: None,
            #[cfg(feature = "cuda")]
            lm_head_fp8: None,
            backend,
        }
    }

    #[cfg(feature = "cuda")]
    fn decode_interpreter_decode_enabled(&self) -> bool {
        decode_interpreter_decode_enabled_for_mtp(self.config.mtp_speculative_tokens)
    }

    /// Arms (or disarms) the DFlash drafter hidden-state capture hook
    /// for subsequent prefill calls. The hook fires per layer after
    /// each input_layernorm; see [`DrafterHiddenCaptureHook`].
    #[cfg(feature = "cuda")]
    pub fn set_drafter_hidden_capture(&mut self, hook: Option<DrafterHiddenCaptureHook>) {
        self.drafter_hidden_capture = hook;
    }

    pub fn from_layout(layout: &ModelLayout, config: EngineConfig, backend: B) -> Result<Self> {
        validate_max_context(&config, &layout.topology)?;
        let weights = ModelWeightsManifest::from_layout(layout)?;
        let mut engine = Self::new(layout.topology.clone(), config, backend);
        engine.weights = Some(weights);
        Ok(engine)
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<ForwardOutput> {
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda("engine_prefill"));
        }
        #[cfg(feature = "cuda")]
        {
            self.prefill_cuda(prompt_tokens)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = prompt_tokens;
            Err(CoreError::UnsupportedNoCuda("engine_prefill"))
        }
    }

    /// Batched verify forward for a block of `tokens` at the current
    /// `state.position`. Advances state by `tokens.len()`, runs one
    /// prefill chunk through the full target stack, then per-position
    /// final RMSNorm + lm_head + greedy argmax, returning the `k`
    /// argmax token ids. Caller checks them against the drafted block
    /// to find the accepted prefix. ~10× cheaper than `k` sequential
    /// `prefill(&[t])` calls because the per-token forward share one
    /// batched layer pass.
    #[cfg(feature = "cuda")]
    pub fn verify_block_batched(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let k = tokens.len();
        let prefill_cap = self.cuda_prefill()?.capacity;
        if k > prefill_cap {
            return Err(CoreError::Runtime(format!(
                "verify_block_batched: block size {k} exceeds prefill capacity {prefill_cap}"
            )));
        }

        // Run prefill on the block. State advances by k; prefill.hidden
        // and prefill.residual now hold the pre-final-norm hidden/
        // residual for each of the k positions at offsets
        // `[0, k * hidden * 2)`.
        self.prefill(tokens)?;

        let final_norm_weight = {
            let manifest = self.weights.as_ref().ok_or_else(|| {
                CoreError::Runtime("verify_block_batched: missing weights manifest".to_owned())
            })?;
            let weights = self.cuda_weights()?;
            self.tensor_ptr(weights, &manifest.final_norm)?
        };
        let lm_head_weight = {
            let manifest = self.weights.as_ref().ok_or_else(|| {
                CoreError::Runtime("verify_block_batched: missing weights manifest".to_owned())
            })?;
            let weights = self.cuda_weights()?;
            self.tensor_ptr(weights, &manifest.lm_head)?
        };

        let hidden = self.topology.hidden_size;
        let vocab = self.topology.vocab_size;
        let (workspace, workspace_bytes) = {
            let runtime = self.cuda_runtime()?;
            let ws = runtime
                .workspace
                .as_ref()
                .map(|b| b.ptr())
                .unwrap_or(DevicePtr::NULL);
            let wsb = runtime.workspace.as_ref().map(|b| b.bytes()).unwrap_or(0);
            (ws, wsb)
        };
        let (prefill_hidden_ptr, prefill_residual_ptr, prefill_normed_ptr) = {
            let prefill = self.cuda_prefill()?;
            (
                prefill.hidden.ptr(),
                prefill.residual.ptr(),
                prefill.normed.ptr(),
            )
        };

        let logits_buf = qwen36_fp4_kernels::CudaDeviceBuffer::alloc(k * vocab * 2)?;
        let tokens_buf = qwen36_fp4_kernels::CudaDeviceBuffer::alloc(k * 4)?;

        self.backend.rmsnorm(&RmsNormSpec {
            rows: k,
            hidden,
            eps: 1.0e-6,
            input_bf16: prefill_hidden_ptr,
            weight_bf16: final_norm_weight,
            residual_bf16: prefill_residual_ptr,
            residual_out_bf16: DevicePtr::NULL,
            output_bf16: prefill_normed_ptr,
            direct_weight: false,
        })?;

        self.backend.bf16_gemm(&Bf16GemmSpec {
            m: vocab,
            n: k,
            k: hidden,
            a_bf16: lm_head_weight,
            b_bf16: prefill_normed_ptr,
            c_bf16: logits_buf.ptr(),
            workspace,
            workspace_bytes,
        })?;

        self.backend
            .sample_rows(&qwen36_fp4_kernels::SamplingRowsSpec {
                rows: k,
                vocab_size: vocab,
                logits_bf16: logits_buf.ptr(),
                output_token_u32: tokens_buf.ptr(),
                mirror_last_output_token_u32: DevicePtr::NULL,
                temperature: 0.0,
            })?;

        qwen36_fp4_kernels::cuda_synchronize()?;

        let mut bytes = vec![0u8; k * 4];
        tokens_buf.copy_to_host(&mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Truncate `state.position` to `new_position` (must be ≤ current).
    /// The target KV cache data past `new_position` is left in place
    /// but will be overwritten by the next forward write. Companion to
    /// `verify_block_batched` for speculative decoding: after the
    /// batched verify advances state by `k`, the controller calls
    /// `crop_state_position(prompt + accepted + 1)` to discard the
    /// rejected speculative tail.
    #[cfg(feature = "cuda")]
    pub fn crop_state_position(&mut self, new_position: usize) -> Result<()> {
        if new_position > self.state.position {
            return Err(CoreError::Runtime(format!(
                "crop_state_position: new {new_position} > current {}",
                self.state.position,
            )));
        }
        self.state.position = new_position;
        Ok(())
    }

    pub fn decode_one(&mut self, token: u32) -> Result<ForwardOutput> {
        self.decode_one_with_sync(token, true)
    }

    #[cfg(feature = "cuda")]
    pub fn decode_one_queued(&mut self, token: u32) -> Result<ForwardOutput> {
        self.decode_one_with_sync(token, false)
    }

    #[cfg(feature = "cuda")]
    pub fn decode_sampled_queued(&mut self) -> Result<ForwardOutput> {
        self.decode_sampled_with_sync(false)
    }

    fn decode_one_with_sync(&mut self, token: u32, sync_after: bool) -> Result<ForwardOutput> {
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda("engine_decode_one"));
        }
        #[cfg(feature = "cuda")]
        {
            self.forward_token_cuda(token, self.state.position, true, sync_after)?;
            self.state.advance(1);
            Ok(ForwardOutput {
                logits_device_ptr: self.cuda_forward()?.logits.ptr().0,
                produced_tokens: 1,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = token;
            let _ = sync_after;
            Err(CoreError::UnsupportedNoCuda("engine_decode_one"))
        }
    }

    #[cfg(feature = "cuda")]
    fn decode_sampled_with_sync(&mut self, sync_after: bool) -> Result<ForwardOutput> {
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda("engine_decode_one"));
        }
        self.forward_sampled_token_cuda(self.state.position, true, sync_after)?;
        self.state.advance(1);
        Ok(ForwardOutput {
            logits_device_ptr: self.cuda_forward()?.logits.ptr().0,
            produced_tokens: 1,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn sample_greedy(&self) -> Result<u32> {
        self.queue_sample_greedy()?;
        self.read_sampled_token()
    }

    #[cfg(feature = "cuda")]
    pub fn read_sampled_token(&self) -> Result<u32> {
        self.synchronize_active_stream_for_host_read()?;
        let mut token = [0_u8; 4];
        self.cuda_forward()?
            .sampled_token_u32
            .copy_to_host(&mut token)?;
        Ok(u32::from_ne_bytes(token))
    }

    #[cfg(feature = "cuda")]
    pub fn read_current_token(&self) -> Result<u32> {
        self.synchronize_active_stream_for_host_read()?;
        let mut token = [0_u8; 4];
        self.cuda_forward()?.token_u32.copy_to_host(&mut token)?;
        Ok(u32::from_ne_bytes(token))
    }

    /// Queue the greedy-sample kernel without copying the result back to the
    /// host. Useful when the next forward pass consumes `sampled_token_u32`
    /// directly via `decode_sampled_queued`, so the host stays off the
    /// critical path until a final `cuda_synchronize`.
    #[cfg(feature = "cuda")]
    pub fn queue_sample_greedy(&self) -> Result<()> {
        self.queue_sample_greedy_into(self.cuda_forward()?.sampled_token_u32.ptr())
    }

    #[cfg(feature = "cuda")]
    pub fn queue_sample_greedy_to_current_token(&self) -> Result<()> {
        self.queue_sample_greedy_into(self.cuda_forward()?.token_u32.ptr())
    }

    #[cfg(feature = "cuda")]
    fn queue_sample_greedy_into(&self, output_token_u32: DevicePtr) -> Result<()> {
        self.queue_sample_greedy_into_with_mirror(output_token_u32, DevicePtr::NULL)
    }

    /// Queue a top-K argmax sample from the engine's current decode logits
    /// buffer into `output_token_u32_kvec`. The output buffer must be at least
    /// `k * 4` bytes. Used by the tree-MTP draft generation path.
    #[cfg(feature = "cuda")]
    fn queue_sample_topk_into(&self, output_token_u32_kvec: DevicePtr, k: usize) -> Result<()> {
        use qwen36_fp4_kernels::sampling::topk_argmax_device;
        let logits = self.cuda_forward()?.logits.ptr();
        let vocab_size = self.topology.vocab_size;
        topk_argmax_device(vocab_size, k, logits, output_token_u32_kvec)
    }

    #[cfg(feature = "cuda")]
    fn queue_sample_greedy_into_with_mirror(
        &self,
        output_token_u32: DevicePtr,
        mirror_output_token_u32: DevicePtr,
    ) -> Result<()> {
        self.backend.sample(&SamplingSpec {
            vocab_size: self.topology.vocab_size,
            logits_bf16: self.cuda_forward()?.logits.ptr(),
            output_token_u32,
            mirror_output_token_u32,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        })
    }

    #[cfg(feature = "cuda")]
    fn queue_sample_greedy_rows_into(
        &self,
        logits_bf16: DevicePtr,
        rows: usize,
        output_token_u32: DevicePtr,
        mirror_last_output_token_u32: DevicePtr,
    ) -> Result<()> {
        self.backend.sample_rows(&SamplingRowsSpec {
            rows,
            vocab_size: self.topology.vocab_size,
            logits_bf16,
            output_token_u32,
            mirror_last_output_token_u32,
            temperature: 1.0,
        })
    }

    #[cfg(feature = "cuda")]
    fn synchronize_active_stream_for_host_read(&self) -> Result<()> {
        let stream = qwen36_fp4_kernels::graph::get_active_stream();
        if stream.is_null() {
            Ok(())
        } else {
            stream.synchronize()
        }
    }

    /// Capture a single decode-and-sample iteration into a CUDA graph for
    /// replay. After this returns, [`decode_graph_step`] can be called
    /// repeatedly without going through the regular host launch path,
    /// dropping ~600 host kernel launches per token down to one
    /// `cudaGraphLaunch`.
    ///
    /// Preconditions:
    /// - `prefill` has run, so `state.position` is the prompt length and
    ///   `forward.logits` holds the logits used by the first sample.
    /// - `queue_sample_greedy` has been called once already, populating
    ///   `forward.sampled_token_u32` from the prefill logits. The graph
    ///   captures `decode_sampled_queued` + `queue_sample_greedy` + a
    ///   device-side position increment, so each replay produces exactly
    ///   one new token.
    #[cfg(feature = "cuda")]
    pub fn enable_decode_graph(&mut self) -> Result<()> {
        use qwen36_fp4_kernels::graph::{self, CudaStream};

        let attention_context_limit = self.graph_attention_context_limit(DecodeGraphKind::Decode);
        if self.decode_graph.as_ref().is_some_and(|graph| {
            graph.kind == DecodeGraphKind::Decode
                && graph.attention_context_limit == attention_context_limit
        }) {
            return Ok(());
        }
        if self.decode_graph.is_some() {
            self.disable_decode_graph()?;
        }
        Self::ensure_graph_capture_allowed()?;
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda("enable_decode_graph"));
        }

        // Seed the device-side position counter with the current position.
        let position_i32 = i32::try_from(self.state.position).map_err(|_| {
            CoreError::Runtime(format!(
                "position {} does not fit i32 for graph capture",
                self.state.position
            ))
        })?;
        let position_buffer_ptr = self.cuda_forward()?.position_i32.ptr();
        self.cuda_forward()?
            .position_i32
            .copy_from_host(&position_i32.to_ne_bytes())?;

        // Allocate a non-default stream and route every kernel through it
        // so the entire forward + sample + increment can be captured.
        let stream = CudaStream::create()?;
        let stream_handle = stream.handle();
        graph::set_active_stream(stream_handle);
        // Productive-spin prefetch stream + event pool must exist before
        // capture begins so kernels can fork to it inside the graph.
        self.ensure_decode_aux_if_enabled()?;
        self.prepare_decode_interpreter_graph_programs(
            position_buffer_ptr,
            attention_context_limit,
        )?;

        // Capture: decode forward (using device position) + sample +
        // increment_i32. Wrapped in a closure so any error reverts the
        // active stream and avoids leaking the capture session.
        let capture_result = (|| -> Result<(graph::CudaGraph, graph::CudaGraphExec)> {
            graph::begin_capture(stream_handle)?;
            self.forward_device_token_cuda_inner(
                self.cuda_forward()?.sampled_token_u32.ptr(),
                self.state.position,
                position_buffer_ptr,
                true,
                false,
            )?;
            self.queue_sample_greedy()?;
            graph::increment_i32(position_buffer_ptr)?;
            let raw_graph = graph::end_capture(stream_handle)?;
            let exec = graph::instantiate(raw_graph)?;
            Ok((raw_graph, exec))
        })();

        let (raw_graph, exec) = match capture_result {
            Ok(value) => value,
            Err(err) => {
                graph::set_active_stream(CudaStream::NULL);
                return Err(err);
            }
        };

        // Stream capture records the decode/sample/increment sequence but does
        // not execute it. Launch once here so callers can enable the graph and
        // immediately read the next sampled token, matching the previous
        // host-launched decode semantics.
        graph::launch(exec, stream_handle)?;
        self.state.advance(1);

        self.decode_graph = Some(DecodeGraphState {
            kind: DecodeGraphKind::Decode,
            attention_context_limit,
            stream,
            exec,
            raw_graph,
        });
        Ok(())
    }

    /// Replay the captured decode-and-sample graph once. Callers must ensure
    /// `enable_decode_graph` was called first.
    #[cfg(feature = "cuda")]
    pub fn decode_graph_step(&mut self) -> Result<()> {
        let attention_context_limit = self.graph_attention_context_limit(DecodeGraphKind::Decode);
        if self.decode_graph.as_ref().is_some_and(|graph| {
            graph.kind == DecodeGraphKind::Decode
                && graph.attention_context_limit != attention_context_limit
        }) {
            self.enable_decode_graph()?;
            return Ok(());
        }
        let graph_state = self.decode_graph.as_ref().ok_or_else(|| {
            CoreError::Runtime("decode_graph_step called without an active capture".to_owned())
        })?;
        if graph_state.kind != DecodeGraphKind::Decode {
            return Err(CoreError::Runtime(
                "decode_graph_step found a non-decode graph".to_owned(),
            ));
        }
        qwen36_fp4_kernels::graph::launch(graph_state.exec, graph_state.stream.handle())?;
        // Mirror the device-side position bump on the host so callers that
        // read `state.position` see the truth.
        self.state.advance(1);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn ensure_mtp_verify_graph_two_tokens(&mut self, start_position: usize) -> Result<()> {
        use qwen36_fp4_kernels::graph::{self, CudaStream};

        let attention_context_limit =
            self.decode_attention_context_limit_for_active_context(start_position + 2);
        if self.decode_graph.as_ref().is_some_and(|graph| {
            graph.kind == DecodeGraphKind::MtpVerifyOne
                && graph.attention_context_limit == attention_context_limit
        }) {
            return Ok(());
        }
        if self.decode_graph.is_some() {
            self.disable_decode_graph()?;
        }
        Self::ensure_graph_capture_allowed()?;
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP verify graph requested with MTP disabled".to_owned(),
            ));
        }
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda(
                "ensure_mtp_verify_graph_two_tokens",
            ));
        }

        let stream = CudaStream::create()?;
        let stream_handle = stream.handle();
        let start_position_device_i32 = self.cuda_prefill()?.position_i32.ptr();
        graph::set_active_stream(stream_handle);
        self.ensure_decode_aux_if_enabled()?;

        let capture_result = (|| -> Result<(graph::CudaGraph, graph::CudaGraphExec)> {
            graph::begin_capture(stream_handle)?;
            self.prefill_cuda_chunk(2, start_position, start_position_device_i32, false)?;
            self.final_norm_prefill_rows(2)?;
            self.prefill_row_logits(0)?;
            let verified_token_ptr = self.cuda_forward()?.mtp_verify_token_u32.ptr_at(8)?;
            self.queue_sample_greedy_into_with_mirror(
                self.cuda_forward()?.token_u32.ptr(),
                verified_token_ptr,
            )?;
            self.prefill_row_logits(1)?;
            let next_token_ptr = self.cuda_forward()?.mtp_verify_token_u32.ptr_at(4)?;
            self.queue_sample_greedy_into(next_token_ptr)?;
            self.run_mtp_prefill_chunk_with_tokens(
                2,
                start_position,
                start_position_device_i32,
                self.cuda_prefill()?.normed.ptr(),
                self.cuda_forward()?.mtp_verify_token_u32.ptr(),
                true,
            )?;
            let next_draft_ptr = self.cuda_forward()?.mtp_verify_token_u32.ptr_at(12)?;
            self.queue_sample_greedy_into_with_mirror(
                self.cuda_forward()?.sampled_token_u32.ptr(),
                next_draft_ptr,
            )?;
            let raw_graph = graph::end_capture(stream_handle)?;
            let exec = graph::instantiate(raw_graph)?;
            Ok((raw_graph, exec))
        })();

        let (raw_graph, exec) = match capture_result {
            Ok(value) => value,
            Err(err) => {
                graph::set_active_stream(CudaStream::NULL);
                return Err(err);
            }
        };

        self.decode_graph = Some(DecodeGraphState {
            kind: DecodeGraphKind::MtpVerifyOne,
            attention_context_limit,
            stream,
            exec,
            raw_graph,
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn launch_mtp_verify_graph_two_tokens(&self) -> Result<()> {
        let graph_state = self.decode_graph.as_ref().ok_or_else(|| {
            CoreError::Runtime("MTP verify graph launch requested before capture".to_owned())
        })?;
        if graph_state.kind != DecodeGraphKind::MtpVerifyOne {
            return Err(CoreError::Runtime(
                "MTP=1 verify graph launch found a different active graph".to_owned(),
            ));
        }
        qwen36_fp4_kernels::graph::launch(graph_state.exec, graph_state.stream.handle())?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn ensure_mtp_verify_graph_multi_tokens(
        &mut self,
        draft_count: usize,
        start_position: usize,
        device_chain_batch: usize,
    ) -> Result<()> {
        use qwen36_fp4_kernels::graph::{self, CudaStream};

        if !(2..=MTP_MAX_DRAFT_TOKENS).contains(&draft_count) {
            return Err(CoreError::Runtime(format!(
                "MTP multi verify graph expects 2..={MTP_MAX_DRAFT_TOKENS} drafts, got {draft_count}"
            )));
        }
        let assume_accept = mtp_assume_accept_enabled();
        let batched_lm_head = mtp_batched_lm_head_enabled();
        let device_chain = assume_accept && mtp_device_chain_enabled();
        let device_chain_batch = if device_chain {
            device_chain_batch.max(1)
        } else {
            1
        };
        let verify_tokens = draft_count + 1;
        let active_context = start_position + verify_tokens * device_chain_batch;
        let attention_context_limit =
            self.decode_attention_context_limit_for_active_context(active_context);
        if self.decode_graph.as_ref().is_some_and(|graph| {
            graph.kind
                == (DecodeGraphKind::MtpVerifyMulti {
                    drafts: draft_count,
                    assume_accept,
                    batched_lm_head,
                    device_chain,
                    device_chain_batch,
                })
                && graph.attention_context_limit == attention_context_limit
        }) {
            return Ok(());
        }
        if self.decode_graph.is_some() {
            self.disable_decode_graph()?;
        }
        Self::ensure_graph_capture_allowed()?;
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP multi verify graph requested with MTP disabled".to_owned(),
            ));
        }
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda(
                "ensure_mtp_verify_graph_multi_tokens",
            ));
        }

        let stream = CudaStream::create()?;
        let stream_handle = stream.handle();
        let start_position_device_i32 = self.cuda_prefill()?.position_i32.ptr();
        graph::set_active_stream(stream_handle);
        self.ensure_decode_aux_if_enabled()?;

        let capture_result = (|| -> Result<(graph::CudaGraph, graph::CudaGraphExec)> {
            graph::begin_capture(stream_handle)?;

            for _ in 0..device_chain_batch {
                if device_chain {
                    self.prefill_cuda_chunk(
                        verify_tokens,
                        start_position,
                        start_position_device_i32,
                        false,
                    )?;
                    self.final_norm_prefill_rows(verify_tokens)?;

                    self.prefill_row_logits(draft_count)?;
                    let shifted_next_token_ptr = self
                        .cuda_prefill()?
                        .token_u32
                        .ptr_at((1 + draft_count) * 4)?;
                    self.queue_sample_greedy_into_with_mirror(
                        shifted_next_token_ptr,
                        self.cuda_prefill()?.token_u32.ptr(),
                    )?;

                    self.run_mtp_prefill_chunk_with_tokens(
                        verify_tokens,
                        start_position,
                        start_position_device_i32,
                        self.cuda_prefill()?.normed.ptr(),
                        self.cuda_prefill()?.token_u32.ptr_at(4)?,
                        true,
                    )?;

                    let first_next_draft_ptr = self.cuda_prefill()?.token_u32.ptr_at(4)?;
                    self.queue_sample_greedy_into(first_next_draft_ptr)?;

                    if draft_count > 1 {
                        let hidden = self.topology.hidden_size;
                        let last_hidden = Self::ptr_offset(
                            self.cuda_prefill()?.normed.ptr(),
                            (verify_tokens - 1) * hidden * 2,
                        )?;
                        for draft_idx in 1..draft_count {
                            let position = start_position
                                .checked_add(verify_tokens)
                                .and_then(|value| value.checked_add(draft_idx - 1))
                                .ok_or_else(|| {
                                    CoreError::Runtime(
                                        "MTP graph draft position overflow".to_owned(),
                                    )
                                })?;
                            let position_ptr = Self::ptr_offset(
                                start_position_device_i32,
                                (verify_tokens + draft_idx - 1) * 4,
                            )?;
                            let input_token =
                                self.cuda_prefill()?.token_u32.ptr_at(draft_idx * 4)?;
                            let target_hidden = if draft_idx == 1 {
                                last_hidden
                            } else {
                                self.cuda_prefill()?.normed.ptr()
                            };
                            self.run_mtp_prefill_chunk_with_tokens(
                                1,
                                position,
                                position_ptr,
                                target_hidden,
                                input_token,
                                true,
                            )?;
                            let output_token =
                                self.cuda_prefill()?.token_u32.ptr_at((1 + draft_idx) * 4)?;
                            self.queue_sample_greedy_into(output_token)?;
                        }
                    }

                    let position_delta = i32::try_from(verify_tokens).map_err(|_| {
                        CoreError::Runtime(format!(
                            "MTP graph verify_tokens {verify_tokens} does not fit i32"
                        ))
                    })?;
                    graph::mtp_assume_accept_chain_advance(
                        start_position_device_i32,
                        draft_count,
                        verify_tokens + draft_count,
                        position_delta,
                    )?;
                    continue;
                }

                let draft_input_src = self.cuda_prefill()?.token_u32.ptr_at(4)?;
                self.cuda_forward()?
                    .mtp_verify_token_u32
                    .copy_from_device_ptr_at(0, draft_input_src, draft_count * 4)?;
                self.prefill_cuda_chunk(
                    verify_tokens,
                    start_position,
                    start_position_device_i32,
                    false,
                )?;
                self.final_norm_prefill_rows(verify_tokens)?;
                if !assume_accept && mtp_batched_lm_head_enabled() {
                    self.prefill_rows_logits_for_mtp_verify(verify_tokens)?;
                    let verified_base_ptr = self
                        .cuda_forward()?
                        .mtp_verify_token_u32
                        .ptr_at(MTP_GRAPH_VERIFIED_BASE * 4)?;
                    let next_token_ptr = self
                        .cuda_forward()?
                        .mtp_verify_token_u32
                        .ptr_at(draft_count * 4)?;
                    self.queue_sample_greedy_rows_into(
                        self.cuda_forward()?.mtp_logits.ptr(),
                        verify_tokens,
                        verified_base_ptr,
                        next_token_ptr,
                    )?;
                } else {
                    if !assume_accept {
                        for draft_idx in 0..draft_count {
                            self.prefill_row_logits(draft_idx)?;
                            let verified_ptr = self
                                .cuda_forward()?
                                .mtp_verify_token_u32
                                .ptr_at((MTP_GRAPH_VERIFIED_BASE + draft_idx) * 4)?;
                            self.queue_sample_greedy_into(verified_ptr)?;
                        }
                    }

                    self.prefill_row_logits(draft_count)?;
                    let next_token_ptr = self
                        .cuda_forward()?
                        .mtp_verify_token_u32
                        .ptr_at(draft_count * 4)?;
                    self.queue_sample_greedy_into(next_token_ptr)?;
                }

                self.run_mtp_prefill_chunk_with_tokens(
                    verify_tokens,
                    start_position,
                    start_position_device_i32,
                    self.cuda_prefill()?.normed.ptr(),
                    self.cuda_forward()?.mtp_verify_token_u32.ptr(),
                    true,
                )?;
                let first_next_draft_ptr = self
                    .cuda_forward()?
                    .mtp_verify_token_u32
                    .ptr_at(MTP_GRAPH_NEXT_DRAFT_BASE * 4)?;
                self.queue_sample_greedy_into(first_next_draft_ptr)?;

                if draft_count > 1 {
                    let hidden = self.topology.hidden_size;
                    let last_hidden = Self::ptr_offset(
                        self.cuda_prefill()?.normed.ptr(),
                        (verify_tokens - 1) * hidden * 2,
                    )?;
                    for draft_idx in 1..draft_count {
                        let position = start_position
                            .checked_add(verify_tokens)
                            .and_then(|value| value.checked_add(draft_idx - 1))
                            .ok_or_else(|| {
                                CoreError::Runtime("MTP graph draft position overflow".to_owned())
                            })?;
                        let position_ptr = Self::ptr_offset(
                            start_position_device_i32,
                            (verify_tokens + draft_idx - 1) * 4,
                        )?;
                        let input_slot = MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx - 1;
                        let output_slot = MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx;
                        let input_token = self
                            .cuda_forward()?
                            .mtp_verify_token_u32
                            .ptr_at(input_slot * 4)?;
                        let target_hidden = if draft_idx == 1 {
                            last_hidden
                        } else {
                            self.cuda_prefill()?.normed.ptr()
                        };
                        self.run_mtp_prefill_chunk_with_tokens(
                            1,
                            position,
                            position_ptr,
                            target_hidden,
                            input_token,
                            true,
                        )?;
                        let output_token = self
                            .cuda_forward()?
                            .mtp_verify_token_u32
                            .ptr_at(output_slot * 4)?;
                        self.queue_sample_greedy_into(output_token)?;
                    }
                }
            }

            let raw_graph = graph::end_capture(stream_handle)?;
            let exec = graph::instantiate(raw_graph)?;
            Ok((raw_graph, exec))
        })();

        let (raw_graph, exec) = match capture_result {
            Ok(value) => value,
            Err(err) => {
                graph::set_active_stream(CudaStream::NULL);
                return Err(err);
            }
        };

        self.decode_graph = Some(DecodeGraphState {
            kind: DecodeGraphKind::MtpVerifyMulti {
                drafts: draft_count,
                assume_accept,
                batched_lm_head,
                device_chain,
                device_chain_batch,
            },
            attention_context_limit,
            stream,
            exec,
            raw_graph,
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn launch_mtp_verify_graph_multi_tokens(
        &self,
        draft_count: usize,
        device_chain_batch: usize,
    ) -> Result<()> {
        let graph_state = self.decode_graph.as_ref().ok_or_else(|| {
            CoreError::Runtime("MTP multi verify graph launch requested before capture".to_owned())
        })?;
        let device_chain = mtp_assume_accept_enabled() && mtp_device_chain_enabled();
        let device_chain_batch = if device_chain {
            device_chain_batch.max(1)
        } else {
            1
        };
        if graph_state.kind
            != (DecodeGraphKind::MtpVerifyMulti {
                drafts: draft_count,
                assume_accept: mtp_assume_accept_enabled(),
                batched_lm_head: mtp_batched_lm_head_enabled(),
                device_chain,
                device_chain_batch,
            })
        {
            return Err(CoreError::Runtime(
                "MTP multi verify graph launch found a different active graph".to_owned(),
            ));
        }
        qwen36_fp4_kernels::graph::launch(graph_state.exec, graph_state.stream.handle())?;
        Ok(())
    }

    /// Capture (or re-capture on context-bucket growth) the reject-recovery
    /// graph for one `(committed, drafts)` shape: re-prefill of the committed
    /// tokens + the full next-draft MTP chain. Per-launch inputs are staged
    /// by the caller: committed tokens + consecutive positions in the prefill
    /// token/position buffers, shifted MTP input tokens in
    /// `mtp_verify_token_u32[0..committed]`. Outputs land in the
    /// `MTP_GRAPH_NEXT_DRAFT_BASE` bundle slots.
    #[cfg(feature = "cuda")]
    fn ensure_mtp_recover_graph(
        &mut self,
        committed: usize,
        drafts: usize,
        start_position: usize,
    ) -> Result<()> {
        use qwen36_fp4_kernels::graph::{self, CudaStream};

        if committed == 0 || committed > MTP_GRAPH_NEXT_DRAFT_BASE {
            return Err(CoreError::Runtime(format!(
                "MTP recover graph expects 1..={MTP_GRAPH_NEXT_DRAFT_BASE} committed tokens, got {committed}"
            )));
        }
        if drafts == 0 || MTP_GRAPH_NEXT_DRAFT_BASE + drafts > MTP_GRAPH_BUNDLE_U32S {
            return Err(CoreError::Runtime(format!(
                "MTP recover graph draft count {drafts} exceeds bundle capacity"
            )));
        }
        let kind = DecodeGraphKind::MtpRecover { committed, drafts };
        let active_context = start_position + committed + drafts;
        let attention_context_limit =
            self.decode_attention_context_limit_for_active_context(active_context);
        if let Some(idx) = self
            .mtp_recover_graphs
            .iter()
            .position(|graph| graph.kind == kind)
        {
            if self.mtp_recover_graphs[idx].attention_context_limit == attention_context_limit {
                return Ok(());
            }
            // Context bucket grew: drain and drop the stale capture.
            let stale = self.mtp_recover_graphs.swap_remove(idx);
            let synchronize_result = stale.stream.handle().synchronize();
            graph::set_active_stream(CudaStream::NULL);
            synchronize_result?;
            drop(stale);
        }
        Self::ensure_graph_capture_allowed()?;
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP recover graph requested with MTP disabled".to_owned(),
            ));
        }
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda("ensure_mtp_recover_graph"));
        }

        let stream = CudaStream::create()?;
        let stream_handle = stream.handle();
        let position_device_i32 = self.cuda_prefill()?.position_i32.ptr();
        graph::set_active_stream(stream_handle);
        self.ensure_decode_aux_if_enabled()?;

        let capture_result = (|| -> Result<(graph::CudaGraph, graph::CudaGraphExec)> {
            graph::begin_capture(stream_handle)?;

            self.prefill_cuda_chunk(committed, start_position, position_device_i32, false)?;
            self.final_norm_prefill_rows(committed)?;

            // Next-draft chain — mirrors generate_mtp_drafts_from_committed_prefill
            // with device-side positions and pre-staged shifted tokens.
            self.run_mtp_prefill_chunk_with_tokens(
                committed,
                start_position,
                position_device_i32,
                self.cuda_prefill()?.normed.ptr(),
                self.cuda_forward()?.mtp_verify_token_u32.ptr(),
                true,
            )?;
            let first_out = self
                .cuda_forward()?
                .mtp_verify_token_u32
                .ptr_at(MTP_GRAPH_NEXT_DRAFT_BASE * 4)?;
            self.queue_sample_greedy_into(first_out)?;

            if drafts > 1 {
                let hidden = self.topology.hidden_size;
                let last_hidden = Self::ptr_offset(
                    self.cuda_prefill()?.normed.ptr(),
                    (committed - 1) * hidden * 2,
                )?;
                for draft_idx in 1..drafts {
                    let position = start_position
                        .checked_add(committed)
                        .and_then(|value| value.checked_add(draft_idx - 1))
                        .ok_or_else(|| {
                            CoreError::Runtime("MTP recover draft position overflow".to_owned())
                        })?;
                    let position_ptr = Self::ptr_offset(
                        position_device_i32,
                        (committed + draft_idx - 1) * 4,
                    )?;
                    let input_token = self
                        .cuda_forward()?
                        .mtp_verify_token_u32
                        .ptr_at((MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx - 1) * 4)?;
                    let target_hidden = if draft_idx == 1 {
                        last_hidden
                    } else {
                        self.cuda_prefill()?.normed.ptr()
                    };
                    self.run_mtp_prefill_chunk_with_tokens(
                        1,
                        position,
                        position_ptr,
                        target_hidden,
                        input_token,
                        true,
                    )?;
                    let output_token = self
                        .cuda_forward()?
                        .mtp_verify_token_u32
                        .ptr_at((MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx) * 4)?;
                    self.queue_sample_greedy_into(output_token)?;
                }
            }

            let raw_graph = graph::end_capture(stream_handle)?;
            let exec = graph::instantiate(raw_graph)?;
            Ok((raw_graph, exec))
        })();

        let (raw_graph, exec) = match capture_result {
            Ok(value) => value,
            Err(err) => {
                graph::set_active_stream(CudaStream::NULL);
                return Err(err);
            }
        };

        if std::env::var("QWEN36_MTP_TRACE").is_ok() {
            eprintln!(
                "mtp.trace recover_graph captured committed={committed} drafts={drafts} limit={attention_context_limit}"
            );
        }
        self.mtp_recover_graphs.push(DecodeGraphState {
            kind,
            attention_context_limit,
            stream,
            exec,
            raw_graph,
        });
        Ok(())
    }

    /// Route subsequent buffer writes onto the recover graph's stream so they
    /// are ordered before its launch (same pattern as
    /// `activate_existing_graph_stream` for the single-slot graphs).
    #[cfg(feature = "cuda")]
    fn activate_mtp_recover_graph_stream(&self, committed: usize, drafts: usize) {
        let kind = DecodeGraphKind::MtpRecover { committed, drafts };
        if let Some(graph_state) = self
            .mtp_recover_graphs
            .iter()
            .find(|graph| graph.kind == kind)
        {
            qwen36_fp4_kernels::graph::set_active_stream(graph_state.stream.handle());
        }
    }

    #[cfg(feature = "cuda")]
    fn launch_mtp_recover_graph(&self, committed: usize, drafts: usize) -> Result<()> {
        let kind = DecodeGraphKind::MtpRecover { committed, drafts };
        let graph_state = self
            .mtp_recover_graphs
            .iter()
            .find(|graph| graph.kind == kind)
            .ok_or_else(|| {
                CoreError::Runtime("MTP recover graph launch requested before capture".to_owned())
            })?;
        qwen36_fp4_kernels::graph::launch(graph_state.exec, graph_state.stream.handle())?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn run_mtp_assume_accept_device_chain(
        &mut self,
        current_token: u32,
        draft_tokens: &[u32],
        max_generated_tokens: usize,
        next_draft_count: usize,
    ) -> Result<MtpDeviceChainResult> {
        if !mtp_assume_accept_enabled() {
            return Err(CoreError::Runtime(
                "MTP device chain requires QWEN36_MTP_ASSUME_ACCEPT=1".to_owned(),
            ));
        }
        if !mtp_device_chain_enabled() {
            return Err(CoreError::Runtime(
                "MTP device chain requires QWEN36_MTP_DEVICE_CHAIN=1".to_owned(),
            ));
        }
        if !(2..=MTP_MAX_DRAFT_TOKENS).contains(&draft_tokens.len()) {
            return Err(CoreError::Runtime(format!(
                "MTP device chain expects 2..={MTP_MAX_DRAFT_TOKENS} drafts, got {}",
                draft_tokens.len()
            )));
        }
        if next_draft_count != draft_tokens.len() {
            return Err(CoreError::Runtime(format!(
                "MTP device chain requires next_draft_count to match the draft window (got {next_draft_count}, expected {})",
                draft_tokens.len()
            )));
        }

        let start_position = self.state.position;
        let verify_tokens = draft_tokens.len() + 1;
        let available_tokens = self.config.max_context.saturating_sub(start_position);
        let mut cycles = max_generated_tokens
            .min(available_tokens)
            .checked_div(verify_tokens)
            .unwrap_or(0);
        let attention_context_limit =
            self.decode_attention_context_limit_for_active_context(start_position + verify_tokens);
        while cycles > 0
            && self.decode_attention_context_limit_for_active_context(
                start_position + cycles * verify_tokens,
            ) != attention_context_limit
        {
            cycles -= 1;
        }

        if cycles == 0 {
            return Ok(MtpDeviceChainResult {
                cycles: 0,
                generated_tokens: 0,
                accepted_draft_tokens: 0,
                next_token: current_token,
                next_draft_tokens: draft_tokens.to_vec(),
            });
        }

        let mut verify_input = Vec::with_capacity(verify_tokens);
        verify_input.push(current_token);
        verify_input.extend_from_slice(draft_tokens);
        self.write_prefill_tokens_and_position_count(
            &verify_input,
            start_position,
            verify_tokens + next_draft_count,
        )?;

        let max_batch = mtp_device_chain_batch().min(cycles).max(1);
        let mut remaining_cycles = cycles;
        while remaining_cycles > 0 {
            let batch = remaining_cycles.min(max_batch);
            self.activate_existing_graph_stream(DecodeGraphKind::MtpVerifyMulti {
                drafts: draft_tokens.len(),
                assume_accept: true,
                batched_lm_head: mtp_batched_lm_head_enabled(),
                device_chain: true,
                device_chain_batch: batch,
            });
            self.ensure_mtp_verify_graph_multi_tokens(draft_tokens.len(), start_position, batch)?;
            self.launch_mtp_verify_graph_multi_tokens(draft_tokens.len(), batch)?;
            remaining_cycles -= batch;
        }

        let mut final_tokens = self.read_prefill_token_bundle(0, 1 + next_draft_count)?;
        let next_token = final_tokens[0];
        let next_draft_tokens = final_tokens.split_off(1);
        let generated_tokens = cycles * verify_tokens;
        self.state.advance(generated_tokens);
        Ok(MtpDeviceChainResult {
            cycles,
            generated_tokens,
            accepted_draft_tokens: cycles * draft_tokens.len(),
            next_token,
            next_draft_tokens,
        })
    }

    /// Capture one MTP verification iteration:
    /// main decode from `forward.token_u32`, greedy-sample verified token back
    /// into `forward.token_u32`, run the MTP layer from that verified token and
    /// the target hidden state, then greedy-sample the next draft into
    /// `forward.sampled_token_u32`.
    #[cfg(feature = "cuda")]
    pub fn enable_mtp_decode_graph(&mut self) -> Result<()> {
        use qwen36_fp4_kernels::graph::{self, CudaStream};

        let attention_context_limit =
            self.graph_attention_context_limit(DecodeGraphKind::MtpDecodeOne);
        if self.decode_graph.as_ref().is_some_and(|graph| {
            graph.kind == DecodeGraphKind::MtpDecodeOne
                && graph.attention_context_limit == attention_context_limit
        }) {
            return Ok(());
        }
        if self.decode_graph.is_some() {
            self.disable_decode_graph()?;
        }
        Self::ensure_graph_capture_allowed()?;
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "enable_mtp_decode_graph called with MTP disabled".to_owned(),
            ));
        }
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda("enable_mtp_decode_graph"));
        }

        let position_i32 = i32::try_from(self.state.position).map_err(|_| {
            CoreError::Runtime(format!(
                "position {} does not fit i32 for graph capture",
                self.state.position
            ))
        })?;
        let forward = self.cuda_forward()?;
        let position_buffer_ptr = forward.position_i32.ptr();
        forward
            .position_i32
            .copy_from_host(&position_i32.to_ne_bytes())?;

        let stream = CudaStream::create()?;
        let stream_handle = stream.handle();
        graph::set_active_stream(stream_handle);
        self.ensure_decode_aux_if_enabled()?;

        let capture_result = (|| -> Result<(graph::CudaGraph, graph::CudaGraphExec)> {
            graph::begin_capture(stream_handle)?;
            self.forward_device_token_cuda_inner(
                self.cuda_forward()?.token_u32.ptr(),
                self.state.position,
                position_buffer_ptr,
                true,
                false,
            )?;
            self.queue_sample_greedy_into(self.cuda_forward()?.token_u32.ptr())?;
            self.run_mtp_decode_from_target_hidden(
                self.cuda_forward()?.token_u32.ptr(),
                self.state.position,
                position_buffer_ptr,
                self.cuda_forward()?.normed.ptr(),
            )?;
            self.queue_sample_greedy_into(self.cuda_forward()?.sampled_token_u32.ptr())?;
            graph::increment_i32(position_buffer_ptr)?;
            let raw_graph = graph::end_capture(stream_handle)?;
            let exec = graph::instantiate(raw_graph)?;
            Ok((raw_graph, exec))
        })();

        let (raw_graph, exec) = match capture_result {
            Ok(value) => value,
            Err(err) => {
                graph::set_active_stream(CudaStream::NULL);
                return Err(err);
            }
        };

        graph::launch(exec, stream_handle)?;
        self.state.advance(1);

        self.decode_graph = Some(DecodeGraphState {
            kind: DecodeGraphKind::MtpDecodeOne,
            attention_context_limit,
            stream,
            exec,
            raw_graph,
        });
        Ok(())
    }

    /// Tear down the captured decode graph and restore the default stream.
    /// Idempotent.
    #[cfg(feature = "cuda")]
    pub fn disable_decode_graph(&mut self) -> Result<()> {
        if let Some(graph_state) = self.decode_graph.take() {
            // Drain any in-flight launches before freeing the graph.
            let synchronize_result = graph_state.stream.handle().synchronize();
            qwen36_fp4_kernels::graph::set_active_stream(
                qwen36_fp4_kernels::graph::CudaStream::NULL,
            );
            synchronize_result?;
            // Drop runs the destructors below.
            drop(graph_state);
        }
        for graph_state in self.mtp_recover_graphs.drain(..) {
            let synchronize_result = graph_state.stream.handle().synchronize();
            qwen36_fp4_kernels::graph::set_active_stream(
                qwen36_fp4_kernels::graph::CudaStream::NULL,
            );
            synchronize_result?;
            drop(graph_state);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn activate_existing_graph_stream(&self, kind: DecodeGraphKind) {
        if let Some(graph_state) = self.decode_graph.as_ref() {
            if graph_state.kind == kind {
                qwen36_fp4_kernels::graph::set_active_stream(graph_state.stream.handle());
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn ensure_graph_capture_allowed() -> Result<()> {
        const DEBUG_DUMP_ENVS: [&str; 3] = [
            "QWEN36_DEBUG_DUMP_DIR",
            "QWEN36_DEBUG_DUMP_DECODE",
            "QWEN36_DEBUG_DUMP_ALL_LAYERS",
        ];
        if DEBUG_DUMP_ENVS
            .iter()
            .any(|name| std::env::var_os(name).is_some())
        {
            return Err(CoreError::Runtime(
                "graph capture is incompatible with debug dumps; unset QWEN36_DEBUG_DUMP_*"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn reset_cuda_state(&mut self) -> Result<()> {
        self.disable_decode_graph()?;
        let runtime = self.cuda_runtime()?;
        if let Some(kv_cache) = &runtime.kv_cache {
            kv_cache.memset(0)?;
        }
        if let Some(mtp_kv_cache) = &runtime.mtp_kv_cache {
            mtp_kv_cache.memset(0)?;
        }
        runtime.deltanet_state.memset(0)?;
        if let Some(deltanet_checkpoint) = &runtime.deltanet_checkpoint {
            deltanet_checkpoint.memset(0)?;
        }
        runtime.conv_history.memset(0)?;
        if let Some(conv_history_checkpoint) = &runtime.conv_history_checkpoint {
            conv_history_checkpoint.memset(0)?;
        }
        self.state.position = 0;
        self.state.accepted_tokens = 0;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn prefill_cuda(&mut self, prompt_tokens: &[u32]) -> Result<ForwardOutput> {
        if prompt_tokens.is_empty() {
            return Err(CoreError::Runtime(
                "prefill requires at least one prompt token".to_owned(),
            ));
        }
        if self.state.position + prompt_tokens.len() > self.config.max_context {
            return Err(CoreError::Runtime(format!(
                "prefill of {} tokens at position {} exceeds configured max_context {}",
                prompt_tokens.len(),
                self.state.position,
                self.config.max_context
            )));
        }
        let mut consumed = 0;
        while consumed < prompt_tokens.len() {
            let start_position = self.state.position;
            let capacity = self.cuda_prefill()?.capacity;
            let chunk = (prompt_tokens.len() - consumed).min(capacity);
            let chunk_tokens = &prompt_tokens[consumed..consumed + chunk];
            let mut token_bytes = Vec::with_capacity(chunk * 4);
            for token in chunk_tokens {
                token_bytes.extend_from_slice(&token.to_ne_bytes());
            }
            let mut position_bytes = Vec::with_capacity(chunk * 4);
            for idx in 0..chunk {
                let position = start_position + idx;
                let position = i32::try_from(position).map_err(|_| {
                    CoreError::Runtime(format!("position {position} does not fit i32 for RoPE"))
                })?;
                position_bytes.extend_from_slice(&position.to_ne_bytes());
            }
            {
                let prefill = self.cuda_prefill()?;
                prefill.token_u32.copy_from_host(&token_bytes)?;
                prefill.position_i32.copy_from_host(&position_bytes)?;
            }

            let emit_logits = consumed + chunk == prompt_tokens.len();
            self.prefill_cuda_chunk(chunk, start_position, DevicePtr::NULL, emit_logits)?;
            if self.config.mtp_speculative_tokens > 0 && !emit_logits {
                let shifted_tokens = &prompt_tokens[consumed + 1..consumed + chunk + 1];
                self.run_mtp_prefill_chunk_from_current_prefill(
                    chunk,
                    start_position,
                    DevicePtr::NULL,
                    shifted_tokens,
                    false,
                )?;
            }
            self.state.advance(chunk);
            consumed += chunk;
        }
        qwen36_fp4_kernels::cuda_synchronize()?;
        Ok(ForwardOutput {
            logits_device_ptr: self.cuda_forward()?.logits.ptr().0,
            produced_tokens: prompt_tokens.len(),
        })
    }

    #[cfg(feature = "cuda")]
    pub fn prepare_mtp_prefill_from_sampled(
        &self,
        prompt_tokens: &[u32],
        sampled_token: u32,
    ) -> Result<()> {
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP prefill requested while mtp_speculative_tokens is 0".to_owned(),
            ));
        }
        if prompt_tokens.is_empty() {
            return Err(CoreError::Runtime(
                "MTP prefill requires at least one prompt token".to_owned(),
            ));
        }

        let capacity = self.cuda_prefill()?.capacity;
        let final_chunk = (prompt_tokens.len() - 1) % capacity + 1;
        let start = prompt_tokens.len() - final_chunk;
        let mut shifted_tokens = Vec::with_capacity(final_chunk);
        for idx in start..prompt_tokens.len() {
            let token = prompt_tokens.get(idx + 1).copied().unwrap_or(sampled_token);
            shifted_tokens.push(token);
        }
        self.run_mtp_prefill_chunk_from_current_prefill(
            final_chunk,
            start,
            DevicePtr::NULL,
            &shifted_tokens,
            true,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn prepare_mtp_decode_from_sampled(&self, target_position: usize) -> Result<()> {
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP decode requested while mtp_speculative_tokens is 0".to_owned(),
            ));
        }
        let forward = self.cuda_forward()?;
        self.run_mtp_decode_from_target_hidden(
            forward.sampled_token_u32.ptr(),
            target_position,
            DevicePtr::NULL,
            forward.normed.ptr(),
        )
    }

    #[cfg(feature = "cuda")]
    pub fn prepare_mtp_drafts_from_sampled(
        &self,
        prompt_tokens: &[u32],
        sampled_token: u32,
        draft_count: usize,
    ) -> Result<Vec<u32>> {
        if draft_count == 0 {
            return Ok(Vec::new());
        }
        if draft_count > MTP_MAX_DRAFT_TOKENS {
            return Err(CoreError::Runtime(format!(
                "MTP draft_count {draft_count} exceeds the supported maximum of {MTP_MAX_DRAFT_TOKENS}"
            )));
        }
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP prefill requested while mtp_speculative_tokens is 0".to_owned(),
            ));
        }
        if prompt_tokens.is_empty() {
            return Err(CoreError::Runtime(
                "MTP prefill requires at least one prompt token".to_owned(),
            ));
        }

        let capacity = self.cuda_prefill()?.capacity;
        let final_chunk = (prompt_tokens.len() - 1) % capacity + 1;
        let start = prompt_tokens.len() - final_chunk;
        let mut shifted_tokens = Vec::with_capacity(final_chunk);
        for idx in start..prompt_tokens.len() {
            let token = prompt_tokens.get(idx + 1).copied().unwrap_or(sampled_token);
            shifted_tokens.push(token);
        }
        self.run_mtp_prefill_chunk_from_current_prefill(
            final_chunk,
            start,
            DevicePtr::NULL,
            &shifted_tokens,
            true,
        )?;
        self.queue_sample_greedy()?;
        let first_draft = self.read_sampled_token()?;
        let mut drafts = vec![first_draft];

        if draft_count > 1 {
            let hidden = self.topology.hidden_size;
            let target_hidden = Self::ptr_offset(
                self.cuda_prefill()?.normed.ptr(),
                (final_chunk - 1) * hidden * 2,
            )?;
            drafts.extend(self.generate_mtp_drafts_from_target(
                first_draft,
                target_hidden,
                prompt_tokens.len(),
                draft_count - 1,
            )?);
        }
        Ok(drafts)
    }

    /// Like `prepare_mtp_drafts_from_sampled` but additionally returns K
    /// top-K leaf candidates from the MTP head's LAST recursive step.
    /// Used for tree-MTP: the chain (top-1 each step) drives the speculative
    /// chain, while the leaves are alternative candidates for the last
    /// position so `verify_mtp_tree_draft` can accept any of K matches.
    #[cfg(feature = "cuda")]
    pub fn prepare_mtp_drafts_with_leaves(
        &self,
        prompt_tokens: &[u32],
        sampled_token: u32,
        chain_depth: usize,
        leaf_count: usize,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        // Generate chain via the existing path.
        let chain =
            self.prepare_mtp_drafts_from_sampled(prompt_tokens, sampled_token, chain_depth)?;

        if leaf_count <= 1 {
            // K=1: the single leaf is the top-1 = chain.last().
            let leaf = chain.last().copied().ok_or_else(|| {
                CoreError::Runtime("prepare_mtp_drafts_with_leaves: chain empty".into())
            })?;
            return Ok((chain, vec![leaf]));
        }

        // After prepare_mtp_drafts_from_sampled returns, the forward logits
        // buffer holds the LAST MTP head step's logits (the one that produced
        // chain.last()). Sample top-K from those same logits to get the K leaf
        // candidates. The top-1 leaf equals chain.last() — intentional.
        let leaf_buf_ptr = self.cuda_forward()?.leaf_tokens_u32.ptr();
        self.queue_sample_topk_into(leaf_buf_ptr, leaf_count)?;

        // D2H read the K leaf token IDs.
        let mut leaf_bytes = vec![0u8; leaf_count * 4];
        self.cuda_forward()?
            .leaf_tokens_u32
            .copy_to_host(&mut leaf_bytes)?;
        let leaves: Vec<u32> = leaf_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok((chain, leaves))
    }

    /// Snapshot the runtime state needed to roll back an MTP verify chunk if
    /// a draft is rejected. Captures DeltaNet recurrent state and the conv1d
    /// history buffer by default. K/V slices are intentionally skipped on the
    /// hot path; `QWEN36_MTP_SNAPSHOT_KV=1` restores the conservative copy.
    #[cfg(feature = "cuda")]
    fn mtp_snapshot_state(&self, start_position: usize, token_count: usize) -> Result<()> {
        let runtime = self.cuda_runtime()?;
        if mtp_recurrent_snapshot_enabled() {
            let deltanet_bytes = runtime.deltanet_state.bytes();
            let deltanet_checkpoint = runtime.deltanet_checkpoint.as_ref().ok_or_else(|| {
                CoreError::Runtime(
                    "MTP recurrent snapshot requested but DeltaNet checkpoint was not allocated"
                        .to_owned(),
                )
            })?;
            deltanet_checkpoint.copy_from_device(&runtime.deltanet_state, deltanet_bytes)?;
            let conv_bytes = runtime.conv_history.bytes();
            if conv_bytes > 0 {
                let conv_history_checkpoint =
                    runtime.conv_history_checkpoint.as_ref().ok_or_else(|| {
                        CoreError::Runtime(
                            "MTP recurrent snapshot requested but conv-history checkpoint was not allocated"
                                .to_owned(),
                        )
                    })?;
                conv_history_checkpoint.copy_from_device(&runtime.conv_history, conv_bytes)?;
            }
        }
        if cuda_env_bool("QWEN36_MTP_SNAPSHOT_KV") {
            self.mtp_kv_slice_copy(start_position, token_count, /* save = */ true)?;
        }
        Ok(())
    }

    /// Restore the runtime state captured by [`mtp_snapshot_state`]. Must only
    /// be called once per snapshot — calling it twice will overwrite live
    /// state with stale data.
    #[cfg(feature = "cuda")]
    fn mtp_restore_state(&self, start_position: usize, token_count: usize) -> Result<()> {
        if !mtp_recurrent_snapshot_enabled() {
            return Err(CoreError::Runtime(
                "MTP rejection requires QWEN36_MTP_SNAPSHOT_RECURRENT=1; fast mode cannot recover recurrent state".to_owned(),
            ));
        }
        let runtime = self.cuda_runtime()?;
        let deltanet_bytes = runtime.deltanet_state.bytes();
        let deltanet_checkpoint = runtime.deltanet_checkpoint.as_ref().ok_or_else(|| {
            CoreError::Runtime(
                "MTP recurrent restore requested but DeltaNet checkpoint was not allocated"
                    .to_owned(),
            )
        })?;
        runtime
            .deltanet_state
            .copy_from_device(deltanet_checkpoint, deltanet_bytes)?;
        let conv_bytes = runtime.conv_history.bytes();
        if conv_bytes > 0 {
            let conv_history_checkpoint =
                runtime.conv_history_checkpoint.as_ref().ok_or_else(|| {
                    CoreError::Runtime(
                    "MTP recurrent restore requested but conv-history checkpoint was not allocated"
                        .to_owned(),
                )
                })?;
            runtime
                .conv_history
                .copy_from_device(conv_history_checkpoint, conv_bytes)?;
        }
        if cuda_env_bool("QWEN36_MTP_SNAPSHOT_KV") {
            self.mtp_kv_slice_copy(start_position, token_count, /* save = */ false)?;
        }
        Ok(())
    }

    /// Snapshot the current DeltaNet recurrent state + conv history into the
    /// shared "chain end" checkpoint buffers (the existing `deltanet_checkpoint`
    /// / `conv_history_checkpoint`). Used in tree-MTP to capture the state
    /// after the chain rows have been processed, before per-leaf restore loop.
    #[cfg(feature = "cuda")]
    fn mtp_snapshot_to_chain_end(&self) -> Result<()> {
        let runtime = self.cuda_runtime()?;
        let deltanet_bytes = runtime.deltanet_state.bytes();
        let deltanet_checkpoint = runtime.deltanet_checkpoint.as_ref().ok_or_else(|| {
            CoreError::Runtime(
                "tree-MTP chain-end snapshot requires DeltaNet checkpoint allocation".into(),
            )
        })?;
        deltanet_checkpoint.copy_from_device(&runtime.deltanet_state, deltanet_bytes)?;
        let conv_bytes = runtime.conv_history.bytes();
        if conv_bytes > 0 {
            let conv_history_checkpoint =
                runtime.conv_history_checkpoint.as_ref().ok_or_else(|| {
                    CoreError::Runtime(
                        "tree-MTP chain-end snapshot requires conv-history checkpoint allocation"
                            .into(),
                    )
                })?;
            conv_history_checkpoint.copy_from_device(&runtime.conv_history, conv_bytes)?;
        }
        Ok(())
    }

    /// Restore DeltaNet state + conv history from the chain-end snapshot.
    #[cfg(feature = "cuda")]
    fn mtp_restore_from_chain_end(&self) -> Result<()> {
        let runtime = self.cuda_runtime()?;
        let deltanet_bytes = runtime.deltanet_state.bytes();
        let deltanet_checkpoint = runtime.deltanet_checkpoint.as_ref().ok_or_else(|| {
            CoreError::Runtime(
                "tree-MTP chain-end restore requires DeltaNet checkpoint allocation".into(),
            )
        })?;
        runtime
            .deltanet_state
            .copy_from_device(deltanet_checkpoint, deltanet_bytes)?;
        let conv_bytes = runtime.conv_history.bytes();
        if conv_bytes > 0 {
            let conv_history_checkpoint =
                runtime.conv_history_checkpoint.as_ref().ok_or_else(|| {
                    CoreError::Runtime(
                        "tree-MTP chain-end restore requires conv-history checkpoint allocation"
                            .into(),
                    )
                })?;
            runtime
                .conv_history
                .copy_from_device(conv_history_checkpoint, conv_bytes)?;
        }
        Ok(())
    }

    /// Snapshot the current DeltaNet state + conv history into per-leaf
    /// checkpoint slot j. Reserved for Phase 2 batched-leaf tree-MTP path
    /// where each leaf's resulting state is captured for a later restore on
    /// the accepted leaf. Currently UNUSED by the Phase 1 α
    /// `verify_mtp_tree_draft` (which restores from chain_end before each
    /// leaf and re-runs the accepted leaf at the end). Allow `dead_code`
    /// while we keep the infra for the upcoming batched path.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn mtp_snapshot_to_leaf(&self, leaf_idx: usize) -> Result<()> {
        use qwen36_fp4_mtp::MTP_TREE_MAX_LEAVES;
        if leaf_idx >= MTP_TREE_MAX_LEAVES {
            return Err(CoreError::Runtime(format!(
                "mtp_snapshot_to_leaf: leaf_idx {leaf_idx} >= {MTP_TREE_MAX_LEAVES}"
            )));
        }
        let runtime = self.cuda_runtime()?;
        let leaf_deltanet = &runtime.deltanet_leaf_checkpoints[leaf_idx];
        let leaf_conv = &runtime.conv_history_leaf_checkpoints[leaf_idx];
        let deltanet_bytes = runtime.deltanet_state.bytes();
        leaf_deltanet.copy_from_device(&runtime.deltanet_state, deltanet_bytes)?;
        let conv_bytes = runtime.conv_history.bytes();
        if conv_bytes > 0 {
            leaf_conv.copy_from_device(&runtime.conv_history, conv_bytes)?;
        }
        Ok(())
    }

    /// Restore DeltaNet state + conv history from per-leaf checkpoint slot j.
    /// Reserved for Phase 2 batched-leaf tree-MTP. Currently UNUSED by α
    /// (see `mtp_snapshot_to_leaf` doc-comment).
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn mtp_restore_from_leaf(&self, leaf_idx: usize) -> Result<()> {
        use qwen36_fp4_mtp::MTP_TREE_MAX_LEAVES;
        if leaf_idx >= MTP_TREE_MAX_LEAVES {
            return Err(CoreError::Runtime(format!(
                "mtp_restore_from_leaf: leaf_idx {leaf_idx} >= {MTP_TREE_MAX_LEAVES}"
            )));
        }
        let runtime = self.cuda_runtime()?;
        let leaf_deltanet = &runtime.deltanet_leaf_checkpoints[leaf_idx];
        let leaf_conv = &runtime.conv_history_leaf_checkpoints[leaf_idx];
        let deltanet_bytes = runtime.deltanet_state.bytes();
        runtime
            .deltanet_state
            .copy_from_device(leaf_deltanet, deltanet_bytes)?;
        let conv_bytes = runtime.conv_history.bytes();
        if conv_bytes > 0 {
            runtime
                .conv_history
                .copy_from_device(leaf_conv, conv_bytes)?;
        }
        Ok(())
    }

    /// Copy the verify K/V slice at `start_position` between live caches and
    /// `mtp_kv_snapshot`. This is only needed when `QWEN36_MTP_SNAPSHOT_KV=1`;
    /// the default rollback path restores recurrent state, then reruns the
    /// committed prefix and overwrites the only K/V slots future attention can
    /// observe.
    #[cfg(feature = "cuda")]
    fn mtp_kv_slice_copy(
        &self,
        start_position: usize,
        token_count: usize,
        save: bool,
    ) -> Result<()> {
        if token_count == 0 || token_count > MtpKvSnapshotLayout::VERIFY_TOKENS {
            return Err(CoreError::Runtime(format!(
                "MTP KV snapshot token_count {token_count} is outside 1..={}",
                MtpKvSnapshotLayout::VERIFY_TOKENS
            )));
        }
        let runtime = self.cuda_runtime()?;
        let mtp_kv_snapshot = runtime.mtp_kv_snapshot.as_ref().ok_or_else(|| {
            CoreError::Runtime(
                "MTP KV snapshot requested but snapshot buffer was not allocated".to_owned(),
            )
        })?;
        let layout = &runtime.mtp_kv_snapshot_layout;
        let main_row_bytes = layout.main_slice_bytes / MtpKvSnapshotLayout::VERIFY_TOKENS;
        let main_slice_bytes = main_row_bytes
            .checked_mul(token_count)
            .ok_or_else(|| CoreError::Runtime("MTP KV slice size overflow".to_owned()))?;
        let main_position_offset = start_position
            .checked_mul(main_row_bytes)
            .ok_or_else(|| CoreError::Runtime("MTP KV slice offset overflow".to_owned()))?;

        if let Some(kv_cache) = &runtime.kv_cache {
            for (idx, layer) in self.state.kv_cache.layers.iter().enumerate() {
                let snapshot_k = layout.main_k_offsets[idx];
                let snapshot_v = layout.main_v_offsets[idx];
                let live_k = (layer.k_offset_bytes as usize)
                    .checked_add(main_position_offset)
                    .ok_or_else(|| CoreError::Runtime("KV K offset overflow".to_owned()))?;
                let live_v = (layer.v_offset_bytes as usize)
                    .checked_add(main_position_offset)
                    .ok_or_else(|| CoreError::Runtime("KV V offset overflow".to_owned()))?;
                if save {
                    let src_k = kv_cache.ptr_at(live_k)?;
                    let src_v = kv_cache.ptr_at(live_v)?;
                    mtp_kv_snapshot.copy_from_device_ptr_at(snapshot_k, src_k, main_slice_bytes)?;
                    mtp_kv_snapshot.copy_from_device_ptr_at(snapshot_v, src_v, main_slice_bytes)?;
                } else {
                    let src_k = mtp_kv_snapshot.ptr_at(snapshot_k)?;
                    let src_v = mtp_kv_snapshot.ptr_at(snapshot_v)?;
                    kv_cache.copy_from_device_ptr_at(live_k, src_k, main_slice_bytes)?;
                    kv_cache.copy_from_device_ptr_at(live_v, src_v, main_slice_bytes)?;
                }
            }
        }

        if let (Some(mtp_kv_cache), Some(mtp_k_off), Some(mtp_v_off)) = (
            runtime.mtp_kv_cache.as_ref(),
            layout.mtp_k_offset,
            layout.mtp_v_offset,
        ) {
            let mtp_row_bytes = layout.mtp_slice_bytes / MtpKvSnapshotLayout::VERIFY_TOKENS;
            let mtp_slice_bytes = mtp_row_bytes
                .checked_mul(token_count)
                .ok_or_else(|| CoreError::Runtime("MTP KV slice size overflow".to_owned()))?;
            let mtp_position_offset = start_position
                .checked_mul(mtp_row_bytes)
                .ok_or_else(|| CoreError::Runtime("MTP KV slice offset overflow".to_owned()))?;
            let plane_bytes = self.mtp_kv_cache_plane_bytes()?;
            let live_k = mtp_position_offset;
            let live_v = plane_bytes
                .checked_add(mtp_position_offset)
                .ok_or_else(|| CoreError::Runtime("MTP V plane offset overflow".to_owned()))?;
            if save {
                let src_k = mtp_kv_cache.ptr_at(live_k)?;
                let src_v = mtp_kv_cache.ptr_at(live_v)?;
                mtp_kv_snapshot.copy_from_device_ptr_at(mtp_k_off, src_k, mtp_slice_bytes)?;
                mtp_kv_snapshot.copy_from_device_ptr_at(mtp_v_off, src_v, mtp_slice_bytes)?;
            } else {
                let src_k = mtp_kv_snapshot.ptr_at(mtp_k_off)?;
                let src_v = mtp_kv_snapshot.ptr_at(mtp_v_off)?;
                mtp_kv_cache.copy_from_device_ptr_at(live_k, src_k, mtp_slice_bytes)?;
                mtp_kv_cache.copy_from_device_ptr_at(live_v, src_v, mtp_slice_bytes)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn write_prefill_tokens_and_positions(
        &self,
        tokens: &[u32],
        start_position: usize,
    ) -> Result<()> {
        self.write_prefill_tokens_and_position_count(tokens, start_position, tokens.len())
    }

    #[cfg(feature = "cuda")]
    fn write_prefill_tokens_and_position_count(
        &self,
        tokens: &[u32],
        start_position: usize,
        position_count: usize,
    ) -> Result<()> {
        if position_count < tokens.len() {
            return Err(CoreError::Runtime(format!(
                "position_count {position_count} is smaller than token count {}",
                tokens.len()
            )));
        }
        let mut token_bytes = Vec::with_capacity(tokens.len() * 4);
        for token in tokens.iter().copied() {
            token_bytes.extend_from_slice(&token.to_ne_bytes());
        }
        let mut position_bytes = Vec::with_capacity(position_count * 4);
        for idx in 0..position_count {
            let position = start_position
                .checked_add(idx)
                .ok_or_else(|| CoreError::Runtime("prefill position overflow".to_owned()))?;
            let position = i32::try_from(position).map_err(|_| {
                CoreError::Runtime(format!("position {position} does not fit i32 for RoPE"))
            })?;
            position_bytes.extend_from_slice(&position.to_ne_bytes());
        }
        let prefill = self.cuda_prefill()?;
        prefill.token_u32.copy_from_host(&token_bytes)?;
        prefill.position_i32.copy_from_host(&position_bytes)
    }

    #[cfg(feature = "cuda")]
    fn sample_prefill_row_to_host(&self, row: usize) -> Result<u32> {
        self.prefill_row_logits(row)?;
        self.queue_sample_greedy_to_current_token()?;
        self.read_current_token()
    }

    #[cfg(feature = "cuda")]
    fn read_mtp_token_bundle(&self, base_slot: usize, count: usize) -> Result<Vec<u32>> {
        if base_slot + count > MTP_GRAPH_BUNDLE_U32S {
            return Err(CoreError::Runtime(format!(
                "MTP token bundle read {base_slot}..{} exceeds bundle slots {MTP_GRAPH_BUNDLE_U32S}",
                base_slot + count
            )));
        }
        let mut bytes = vec![0_u8; count * 4];
        self.cuda_forward()?
            .mtp_verify_token_u32
            .copy_to_host_at(base_slot * 4, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    #[cfg(feature = "cuda")]
    fn read_prefill_token_bundle(&self, base_slot: usize, count: usize) -> Result<Vec<u32>> {
        let mut bytes = vec![0_u8; count * 4];
        self.cuda_prefill()?
            .token_u32
            .copy_to_host_at(base_slot * 4, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    #[cfg(feature = "cuda")]
    fn generate_mtp_drafts_from_target(
        &self,
        first_shifted_token: u32,
        first_target_hidden_bf16: DevicePtr,
        first_mtp_position: usize,
        draft_count: usize,
    ) -> Result<Vec<u32>> {
        if draft_count == 0 {
            return Ok(Vec::new());
        }
        if MTP_GRAPH_NEXT_DRAFT_BASE + draft_count > MTP_GRAPH_BUNDLE_U32S {
            return Err(CoreError::Runtime(format!(
                "MTP draft_count {draft_count} exceeds bundle capacity"
            )));
        }
        self.cuda_prefill()?
            .token_u32
            .copy_from_host(&first_shifted_token.to_ne_bytes())?;
        self.run_mtp_prefill_chunk(
            1,
            first_mtp_position,
            DevicePtr::NULL,
            first_target_hidden_bf16,
            true,
        )?;
        let first_out = self
            .cuda_forward()?
            .mtp_verify_token_u32
            .ptr_at(MTP_GRAPH_NEXT_DRAFT_BASE * 4)?;
        self.queue_sample_greedy_into(first_out)?;

        for draft_idx in 0..draft_count {
            if draft_idx == 0 {
                continue;
            }
            let position = first_mtp_position.checked_add(draft_idx).ok_or_else(|| {
                CoreError::Runtime("MTP recursive draft position overflow".to_owned())
            })?;
            let input_slot = MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx - 1;
            let output_slot = MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx;
            let input_token = self
                .cuda_forward()?
                .mtp_verify_token_u32
                .ptr_at(input_slot * 4)?;
            self.run_mtp_prefill_chunk_with_tokens(
                1,
                position,
                DevicePtr::NULL,
                self.cuda_prefill()?.normed.ptr(),
                input_token,
                true,
            )?;
            let output_token = self
                .cuda_forward()?
                .mtp_verify_token_u32
                .ptr_at(output_slot * 4)?;
            self.queue_sample_greedy_into(output_token)?;
        }
        self.read_mtp_token_bundle(MTP_GRAPH_NEXT_DRAFT_BASE, draft_count)
    }

    #[cfg(feature = "cuda")]
    fn generate_mtp_drafts_from_committed_prefill(
        &self,
        shifted_tokens: &[u32],
        start_position: usize,
        target_hidden_bf16: DevicePtr,
        draft_count: usize,
    ) -> Result<Vec<u32>> {
        if draft_count == 0 {
            return Ok(Vec::new());
        }
        if shifted_tokens.is_empty() {
            return Err(CoreError::Runtime(
                "MTP committed prefill requires at least one shifted token".to_owned(),
            ));
        }

        let mut token_bytes = Vec::with_capacity(shifted_tokens.len() * 4);
        for token in shifted_tokens {
            token_bytes.extend_from_slice(&token.to_ne_bytes());
        }
        self.cuda_prefill()?
            .token_u32
            .copy_from_host(&token_bytes)?;
        self.run_mtp_prefill_chunk(
            shifted_tokens.len(),
            start_position,
            DevicePtr::NULL,
            target_hidden_bf16,
            true,
        )?;
        let first_out = self
            .cuda_forward()?
            .mtp_verify_token_u32
            .ptr_at(MTP_GRAPH_NEXT_DRAFT_BASE * 4)?;
        self.queue_sample_greedy_into(first_out)?;

        if draft_count > 1 {
            let hidden = self.topology.hidden_size;
            let last_hidden = Self::ptr_offset(
                self.cuda_prefill()?.normed.ptr(),
                (shifted_tokens.len() - 1) * hidden * 2,
            )?;
            for draft_idx in 1..draft_count {
                let position = start_position
                    .checked_add(shifted_tokens.len())
                    .and_then(|value| value.checked_add(draft_idx - 1))
                    .ok_or_else(|| {
                        CoreError::Runtime("MTP recursive draft position overflow".to_owned())
                    })?;
                let input_slot = MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx - 1;
                let output_slot = MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx;
                let input_token = self
                    .cuda_forward()?
                    .mtp_verify_token_u32
                    .ptr_at(input_slot * 4)?;
                let target_hidden = if draft_idx == 1 {
                    last_hidden
                } else {
                    self.cuda_prefill()?.normed.ptr()
                };
                self.run_mtp_prefill_chunk_with_tokens(
                    1,
                    position,
                    DevicePtr::NULL,
                    target_hidden,
                    input_token,
                    true,
                )?;
                let output_token = self
                    .cuda_forward()?
                    .mtp_verify_token_u32
                    .ptr_at(output_slot * 4)?;
                self.queue_sample_greedy_into(output_token)?;
            }
        }
        self.read_mtp_token_bundle(MTP_GRAPH_NEXT_DRAFT_BASE, draft_count)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn recover_after_mtp_multi_reject(
        &mut self,
        current_token: u32,
        draft_tokens: &[u32],
        rejected_draft_idx: usize,
        verified_token: u32,
        start_position: usize,
        verify_tokens: usize,
        next_draft_count: usize,
    ) -> Result<MtpMultiVerifyResult> {
        let committed_tokens = rejected_draft_idx + 1;

        let mut committed_input = Vec::with_capacity(committed_tokens);
        committed_input.push(current_token);
        committed_input.extend_from_slice(&draft_tokens[..rejected_draft_idx]);

        // Graph path: the recovery re-prefill + the entire next-draft chain
        // replay as one capture (the host-launched chain was the dominant
        // per-cycle cost at acceptance < 1). State restore and input staging
        // stay host-side; they are ordered before the launch by activating
        // the recover graph's stream first.
        let use_recover_graph = next_draft_count > 0
            && committed_tokens <= MTP_GRAPH_NEXT_DRAFT_BASE
            && MTP_GRAPH_NEXT_DRAFT_BASE + next_draft_count <= MTP_GRAPH_BUNDLE_U32S
            && mtp_recover_graph_enabled()
            && std::env::var("QWEN36_MTP_MULTI_GRAPH_DISABLE").is_err()
            && Self::ensure_graph_capture_allowed().is_ok()
            && self.backend.name() != "no-cuda";
        if use_recover_graph {
            self.ensure_mtp_recover_graph(committed_tokens, next_draft_count, start_position)?;
            self.activate_mtp_recover_graph_stream(committed_tokens, next_draft_count);
            self.mtp_restore_state(start_position, verify_tokens)?;
            self.write_prefill_tokens_and_position_count(
                &committed_input,
                start_position,
                committed_tokens + next_draft_count,
            )?;
            let mut shifted_bytes = Vec::with_capacity(committed_tokens * 4);
            for token in draft_tokens[..rejected_draft_idx].iter().copied() {
                shifted_bytes.extend_from_slice(&token.to_ne_bytes());
            }
            shifted_bytes.extend_from_slice(&verified_token.to_ne_bytes());
            self.cuda_forward()?
                .mtp_verify_token_u32
                .copy_from_host(&shifted_bytes)?;
            self.launch_mtp_recover_graph(committed_tokens, next_draft_count)?;
            let next_draft_tokens =
                self.read_mtp_token_bundle(MTP_GRAPH_NEXT_DRAFT_BASE, next_draft_count)?;

            self.state.advance(committed_tokens);
            return Ok(MtpMultiVerifyResult {
                accepted_drafts: rejected_draft_idx,
                rejected: true,
                next_token: Some(verified_token),
                next_draft_tokens,
            });
        }

        self.mtp_restore_state(start_position, verify_tokens)?;
        self.write_prefill_tokens_and_positions(&committed_input, start_position)?;
        self.prefill_cuda_chunk(committed_tokens, start_position, DevicePtr::NULL, false)?;
        self.final_norm_prefill_rows(committed_tokens)?;

        let next_draft_tokens = if next_draft_count > 0 {
            let target_hidden = self.cuda_prefill()?.normed.ptr();
            let mut shifted_tokens = Vec::with_capacity(committed_tokens);
            shifted_tokens.extend_from_slice(&draft_tokens[..rejected_draft_idx]);
            shifted_tokens.push(verified_token);
            self.generate_mtp_drafts_from_committed_prefill(
                &shifted_tokens,
                start_position,
                target_hidden,
                next_draft_count,
            )?
        } else {
            Vec::new()
        };

        self.state.advance(committed_tokens);
        Ok(MtpMultiVerifyResult {
            accepted_drafts: rejected_draft_idx,
            rejected: true,
            next_token: Some(verified_token),
            next_draft_tokens,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn verify_mtp_draft_two_tokens(
        &mut self,
        current_token: u32,
        draft_token: u32,
        need_next_token: bool,
        need_next_draft: bool,
    ) -> Result<MtpVerifyResult> {
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP two-token verification requested while mtp_speculative_tokens is 0".to_owned(),
            ));
        }
        let start_position = self.state.position;
        if start_position + 2 > self.config.max_context {
            return Err(CoreError::Runtime(format!(
                "MTP two-token verification at position {start_position} exceeds max_context {}",
                self.config.max_context
            )));
        }

        let mut token_bytes = [0_u8; 8];
        token_bytes[..4].copy_from_slice(&current_token.to_ne_bytes());
        token_bytes[4..].copy_from_slice(&draft_token.to_ne_bytes());
        let position_0 = i32::try_from(start_position).map_err(|_| {
            CoreError::Runtime(format!(
                "position {start_position} does not fit i32 for RoPE"
            ))
        })?;
        let position_1_usize = start_position + 1;
        let position_1 = i32::try_from(position_1_usize).map_err(|_| {
            CoreError::Runtime(format!(
                "position {position_1_usize} does not fit i32 for RoPE"
            ))
        })?;
        if need_next_draft {
            self.activate_existing_graph_stream(DecodeGraphKind::MtpVerifyOne);
        }
        let mut position_bytes = [0_u8; 8];
        position_bytes[..4].copy_from_slice(&position_0.to_ne_bytes());
        position_bytes[4..].copy_from_slice(&position_1.to_ne_bytes());
        {
            let prefill = self.cuda_prefill()?;
            prefill.token_u32.copy_from_host(&token_bytes)?;
            prefill.position_i32.copy_from_host(&position_bytes)?;
        }

        // Snapshot DeltaNet recurrent state, conv1d history, and the K/V
        // slices for the two verify positions before the verify chunk mutates
        // them. On rejection we roll back these in-place rather than doing a
        // catastrophic reset+full-reprefill from the CLI side.
        self.mtp_snapshot_state(start_position, 2)?;

        if need_next_draft {
            let mut mtp_verify_tokens = [0_u8; 16];
            mtp_verify_tokens[..4].copy_from_slice(&draft_token.to_ne_bytes());
            self.cuda_forward()?
                .mtp_verify_token_u32
                .copy_from_host(&mtp_verify_tokens)?;
            self.ensure_mtp_verify_graph_two_tokens(start_position)?;
            self.launch_mtp_verify_graph_two_tokens()?;

            let mut verify_bytes = [0_u8; 16];
            self.cuda_forward()?
                .mtp_verify_token_u32
                .copy_to_host(&mut verify_bytes)?;
            let verified_token = u32::from_ne_bytes([
                verify_bytes[8],
                verify_bytes[9],
                verify_bytes[10],
                verify_bytes[11],
            ]);
            if verified_token != draft_token {
                return self.recover_after_mtp_reject(
                    current_token,
                    verified_token,
                    start_position,
                    /* need_new_draft = */ need_next_token,
                );
            }
            let next_token = u32::from_ne_bytes([
                verify_bytes[4],
                verify_bytes[5],
                verify_bytes[6],
                verify_bytes[7],
            ]);
            let next_draft_token = u32::from_ne_bytes([
                verify_bytes[12],
                verify_bytes[13],
                verify_bytes[14],
                verify_bytes[15],
            ]);
            self.state.advance(2);

            return Ok(MtpVerifyResult {
                accepted: true,
                verified_token,
                next_token: Some(next_token),
                next_draft_token: Some(next_draft_token),
            });
        }

        self.prefill_cuda_chunk(2, start_position, DevicePtr::NULL, false)?;
        self.final_norm_prefill_rows(2)?;

        self.prefill_row_logits(0)?;
        self.queue_sample_greedy_to_current_token()?;
        let verified_token = self.read_current_token()?;
        if verified_token != draft_token {
            return self.recover_after_mtp_reject(
                current_token,
                verified_token,
                start_position,
                need_next_token,
            );
        }

        if !need_next_token {
            self.state.advance(2);
            return Ok(MtpVerifyResult {
                accepted: true,
                verified_token,
                next_token: None,
                next_draft_token: None,
            });
        }

        self.prefill_row_logits(1)?;
        self.queue_sample_greedy_to_current_token()?;
        let next_token = self.read_current_token()?;

        if !need_next_draft {
            self.state.advance(2);
            return Ok(MtpVerifyResult {
                accepted: true,
                verified_token,
                next_token: Some(next_token),
                next_draft_token: None,
            });
        }

        let mut mtp_tokens = [0_u8; 8];
        mtp_tokens[..4].copy_from_slice(&draft_token.to_ne_bytes());
        mtp_tokens[4..].copy_from_slice(&next_token.to_ne_bytes());
        self.cuda_prefill()?.token_u32.copy_from_host(&mtp_tokens)?;
        self.run_mtp_prefill_chunk(
            2,
            start_position,
            DevicePtr::NULL,
            self.cuda_prefill()?.normed.ptr(),
            true,
        )?;
        self.queue_sample_greedy()?;
        let next_draft_token = self.read_sampled_token()?;
        self.state.advance(2);

        Ok(MtpVerifyResult {
            accepted: true,
            verified_token,
            next_token: Some(next_token),
            next_draft_token: Some(next_draft_token),
        })
    }

    #[cfg(feature = "cuda")]
    pub fn verify_mtp_draft_tokens(
        &mut self,
        current_token: u32,
        draft_tokens: &[u32],
        need_next_token_on_full_accept: bool,
        next_draft_count: usize,
    ) -> Result<MtpMultiVerifyResult> {
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP multi-token verification requested while mtp_speculative_tokens is 0"
                    .to_owned(),
            ));
        }
        if draft_tokens.is_empty() || draft_tokens.len() > MTP_MAX_DRAFT_TOKENS {
            return Err(CoreError::Runtime(format!(
                "MTP multi-token verification expects 1..={MTP_MAX_DRAFT_TOKENS} drafts, got {}",
                draft_tokens.len()
            )));
        }
        if next_draft_count > MTP_MAX_DRAFT_TOKENS {
            return Err(CoreError::Runtime(format!(
                "MTP next_draft_count {next_draft_count} exceeds the supported maximum of {MTP_MAX_DRAFT_TOKENS}"
            )));
        }

        let start_position = self.state.position;
        let verify_tokens = draft_tokens.len() + 1;
        if start_position + verify_tokens > self.config.max_context {
            return Err(CoreError::Runtime(format!(
                "MTP verification of {verify_tokens} tokens at position {start_position} exceeds max_context {}",
                self.config.max_context
            )));
        }
        let available_rows_after_verify =
            self.config.max_context - (start_position + verify_tokens);
        let effective_next_draft_count = next_draft_count.min(available_rows_after_verify + 1);
        let use_multi_graph = draft_tokens.len() >= 2
            && need_next_token_on_full_accept
            && effective_next_draft_count == draft_tokens.len()
            && std::env::var("QWEN36_MTP_MULTI_GRAPH_DISABLE").is_err();
        let trace_mtp = std::env::var("QWEN36_MTP_TRACE").is_ok();
        if use_multi_graph {
            self.activate_existing_graph_stream(DecodeGraphKind::MtpVerifyMulti {
                drafts: draft_tokens.len(),
                assume_accept: mtp_assume_accept_enabled(),
                batched_lm_head: mtp_batched_lm_head_enabled(),
                device_chain: mtp_assume_accept_enabled() && mtp_device_chain_enabled(),
                device_chain_batch: 1,
            });
        }

        let mut verify_input = Vec::with_capacity(verify_tokens);
        verify_input.push(current_token);
        verify_input.extend_from_slice(draft_tokens);
        let position_count = if use_multi_graph {
            verify_tokens + effective_next_draft_count
        } else {
            verify_tokens
        };
        self.write_prefill_tokens_and_position_count(
            &verify_input,
            start_position,
            position_count,
        )?;

        let assume_accept = mtp_assume_accept_enabled();
        if !(use_multi_graph && assume_accept) {
            self.mtp_snapshot_state(start_position, verify_tokens)?;
        }
        if use_multi_graph {
            self.ensure_mtp_verify_graph_multi_tokens(draft_tokens.len(), start_position, 1)?;
            self.launch_mtp_verify_graph_multi_tokens(draft_tokens.len(), 1)?;

            let mut verify_bytes = [0_u8; MTP_GRAPH_BUNDLE_U32S * 4];
            self.cuda_forward()?
                .mtp_verify_token_u32
                .copy_to_host(&mut verify_bytes)?;
            let mut verified_tokens = Vec::with_capacity(draft_tokens.len());
            if assume_accept {
                verified_tokens.extend_from_slice(draft_tokens);
            } else {
                for (draft_idx, draft_token) in draft_tokens.iter().copied().enumerate() {
                    let offset = (MTP_GRAPH_VERIFIED_BASE + draft_idx) * 4;
                    let verified_token = u32::from_ne_bytes([
                        verify_bytes[offset],
                        verify_bytes[offset + 1],
                        verify_bytes[offset + 2],
                        verify_bytes[offset + 3],
                    ]);
                    verified_tokens.push(verified_token);
                    if verified_token != draft_token {
                        if trace_mtp {
                            eprintln!(
                                "mtp.trace graph start={start_position} drafts={draft_tokens:?} verified={verified_tokens:?} reject_idx={draft_idx}"
                            );
                        }
                        return self.recover_after_mtp_multi_reject(
                            current_token,
                            draft_tokens,
                            draft_idx,
                            verified_token,
                            start_position,
                            verify_tokens,
                            effective_next_draft_count,
                        );
                    }
                }
            }

            let next_token_offset = draft_tokens.len() * 4;
            let next_token = u32::from_ne_bytes([
                verify_bytes[next_token_offset],
                verify_bytes[next_token_offset + 1],
                verify_bytes[next_token_offset + 2],
                verify_bytes[next_token_offset + 3],
            ]);
            let next_draft_tokens = (0..effective_next_draft_count)
                .map(|idx| {
                    let offset = (MTP_GRAPH_NEXT_DRAFT_BASE + idx) * 4;
                    u32::from_ne_bytes([
                        verify_bytes[offset],
                        verify_bytes[offset + 1],
                        verify_bytes[offset + 2],
                        verify_bytes[offset + 3],
                    ])
                })
                .collect::<Vec<_>>();
            if trace_mtp {
                eprintln!(
                    "mtp.trace graph start={start_position} drafts={draft_tokens:?} verified={verified_tokens:?} next={next_token} next_drafts={next_draft_tokens:?}"
                );
            }

            self.state.advance(verify_tokens);
            return Ok(MtpMultiVerifyResult {
                accepted_drafts: draft_tokens.len(),
                rejected: false,
                next_token: Some(next_token),
                next_draft_tokens,
            });
        }

        self.prefill_cuda_chunk(verify_tokens, start_position, DevicePtr::NULL, false)?;
        self.final_norm_prefill_rows(verify_tokens)?;

        for (draft_idx, draft_token) in draft_tokens.iter().copied().enumerate() {
            let verified_token = self.sample_prefill_row_to_host(draft_idx)?;
            if verified_token != draft_token {
                if trace_mtp {
                    eprintln!(
                        "mtp.trace host start={start_position} drafts={draft_tokens:?} reject_idx={draft_idx} verified={verified_token}"
                    );
                }
                return self.recover_after_mtp_multi_reject(
                    current_token,
                    draft_tokens,
                    draft_idx,
                    verified_token,
                    start_position,
                    verify_tokens,
                    effective_next_draft_count,
                );
            }
        }

        let accepted_drafts = draft_tokens.len();
        let next_token = if need_next_token_on_full_accept {
            Some(self.sample_prefill_row_to_host(accepted_drafts)?)
        } else {
            None
        };
        let next_draft_tokens =
            if let (Some(next_token), true) = (next_token, effective_next_draft_count > 0) {
                let target_hidden = self.cuda_prefill()?.normed.ptr();
                let mut shifted_tokens = Vec::with_capacity(verify_tokens);
                shifted_tokens.extend_from_slice(draft_tokens);
                shifted_tokens.push(next_token);
                self.generate_mtp_drafts_from_committed_prefill(
                    &shifted_tokens,
                    start_position,
                    target_hidden,
                    effective_next_draft_count,
                )?
            } else {
                Vec::new()
            };
        if trace_mtp {
            eprintln!(
                "mtp.trace host start={start_position} drafts={draft_tokens:?} next={next_token:?} next_drafts={next_draft_tokens:?}"
            );
        }

        self.state.advance(verify_tokens);
        Ok(MtpMultiVerifyResult {
            accepted_drafts,
            rejected: false,
            next_token,
            next_draft_tokens,
        })
    }

    /// Tree-MTP cycle entry point. Combines the existing chain MTP verify
    /// chunk with K leaf candidates branching from the chain end. Each leaf
    /// is processed via a single-token decode forward (`forward_token_cuda`)
    /// after restoring DeltaNet/conv state to the chain-end snapshot, so all
    /// K leaves are siblings (not chained). On acceptance of a leaf, the
    /// verified-after-leaf token is also committed (gives +1 over today's
    /// chain MTP).
    ///
    /// Falls back to chain MTP semantics when K = 1 (single leaf is just an
    /// extended chain step).
    #[cfg(feature = "cuda")]
    pub fn verify_mtp_tree_draft(
        &mut self,
        current_token: u32,
        chain_tokens: &[u32],
        leaf_tokens: &[u32],
        next_draft_count: usize,
    ) -> Result<qwen36_fp4_mtp::TreeVerifyResult> {
        use qwen36_fp4_mtp::{MTP_TREE_MAX_LEAVES, TreeDraft, walk_tree_acceptance};

        if leaf_tokens.is_empty() {
            return Err(CoreError::Runtime(
                "tree leaf_tokens cannot be empty (use verify_mtp_draft_tokens for chain-only)"
                    .into(),
            ));
        }
        if leaf_tokens.len() > MTP_TREE_MAX_LEAVES {
            return Err(CoreError::Runtime(format!(
                "tree leaf_tokens.len() {} exceeds {MTP_TREE_MAX_LEAVES}",
                leaf_tokens.len()
            )));
        }
        if mtp_tree_disable_enabled() {
            // Kill switch: degrade to chain MTP via verify_mtp_draft_tokens.
            return self.verify_mtp_tree_draft_chain_fallback(
                current_token,
                chain_tokens,
                leaf_tokens,
                next_draft_count,
            );
        }

        let chain_depth = chain_tokens.len();
        let leaf_count = leaf_tokens.len();
        let start_position = self.state.position;
        let chain_verify_tokens = chain_depth + 1;
        let total_positions_written = chain_verify_tokens + 1; // chain + leaf canonical slot

        if start_position + total_positions_written > self.config.max_context {
            return Err(CoreError::Runtime(format!(
                "tree MTP at position {start_position} would exceed max_context {} (needs {total_positions_written})",
                self.config.max_context
            )));
        }

        // 1. Snapshot KV slice + DeltaNet/conv state for the entire region we'll touch.
        //    The chain pass writes positions start_position..start_position+chain_depth.
        //    Leaves all write to start_position+chain_depth+1 (canonical leaf slot).
        self.mtp_snapshot_state(start_position, total_positions_written)?;

        // 2. Run chain pass via the existing prefill_cuda_chunk.
        let mut verify_input: Vec<u32> = Vec::with_capacity(chain_verify_tokens);
        verify_input.push(current_token);
        verify_input.extend_from_slice(chain_tokens);
        self.write_prefill_tokens_and_position_count(
            &verify_input,
            start_position,
            chain_verify_tokens,
        )?;
        self.prefill_cuda_chunk(chain_verify_tokens, start_position, DevicePtr::NULL, false)?;

        // 3. Sample chain row outputs via final norm + batched lm_head + sample_rows.
        //    This populates verified[0..=chain_depth].
        self.final_norm_prefill_rows(chain_verify_tokens)?;
        let mut verified: Vec<u32> = Vec::with_capacity(chain_verify_tokens + leaf_count);
        for row in 0..chain_verify_tokens {
            verified.push(self.sample_prefill_row_to_host(row)?);
        }

        // 4. Snapshot DeltaNet/conv state at chain end.
        self.mtp_snapshot_to_chain_end()?;

        // 5. Per-leaf forward via decode primitive.
        let leaf_position = start_position + chain_depth + 1;
        for &leaf_token in leaf_tokens {
            self.mtp_restore_from_chain_end()?;
            self.forward_token_cuda(leaf_token, leaf_position, true, true)?;
            verified.push(self.sample_greedy()?);
        }

        // 6. Walk acceptance.
        let draft = TreeDraft {
            chain_tokens: chain_tokens.to_vec(),
            leaf_tokens: leaf_tokens.to_vec(),
        };
        let mut result = walk_tree_acceptance(&verified, &draft);

        // Number of chain top-1 drafts to pre-compute for the next cycle. We
        // mirror the chain MTP path which uses `chain_depth` recursive top-1
        // steps. Caller's `next_draft_count` is currently ignored — kept in
        // the signature for API stability but tree always pre-computes
        // `chain_depth` drafts.
        let _ = next_draft_count;
        let next_chain_count = chain_depth;

        // 7. Restore final state for the accepted path AND pre-compute next
        //    cycle's chain drafts via generate_mtp_drafts_from_committed_prefill.
        //    This mirrors what verify_mtp_draft_tokens does for chain MTP and
        //    advances MTP head KV by the committed token count, so the next
        //    tree cycle starts from a coherent MTP head state.
        if let Some(accepted_leaf_idx) = result.accepted_leaf {
            // Case A: full chain + leaf accept. Re-run the accepted leaf so its
            // K/V lands at the canonical slot and DeltaNet/conv state matches
            // "after leaf accepted_leaf_idx". forward.normed[0] then holds the
            // leaf's hidden state.
            self.mtp_restore_from_chain_end()?;
            self.forward_token_cuda(leaf_tokens[accepted_leaf_idx], leaf_position, false, true)?;
            // Copy forward.normed[0] into prefill.normed[chain_depth + 1] so
            // target_hidden has hidden states for [chain rows 0..chain_depth,
            // chain end, accepted_leaf]. The next-token row is appended by
            // run_mtp_prefill_chunk itself when shifted_tokens.len() = chain+2.
            let hidden_bytes = self.topology.hidden_size * 2; // BF16
            let leaf_normed_src = self.cuda_forward()?.normed.ptr();
            let dst_offset = (chain_depth + 1) * hidden_bytes;
            self.cuda_prefill()?.normed.copy_from_device_ptr_at(
                dst_offset,
                leaf_normed_src,
                hidden_bytes,
            )?;

            let mut shifted_tokens: Vec<u32> = chain_tokens.to_vec();
            shifted_tokens.push(leaf_tokens[accepted_leaf_idx]);
            shifted_tokens.push(result.next_token);
            let target_hidden = self.cuda_prefill()?.normed.ptr();
            result.next_chain_drafts = self.generate_mtp_drafts_from_committed_prefill(
                &shifted_tokens,
                start_position,
                target_hidden,
                next_chain_count,
            )?;
            self.state.advance(chain_depth + 2);
        } else if result.accepted_chain == chain_depth {
            // Case B: full chain accept, no leaf. The chain-end hidden state is
            // already at prefill.normed[chain_depth], and result.next_token is
            // verified[chain_depth]. shifted_tokens = chain + [next_token].
            self.mtp_restore_from_chain_end()?;

            let mut shifted_tokens: Vec<u32> = chain_tokens.to_vec();
            shifted_tokens.push(result.next_token);
            let target_hidden = self.cuda_prefill()?.normed.ptr();
            result.next_chain_drafts = self.generate_mtp_drafts_from_committed_prefill(
                &shifted_tokens,
                start_position,
                target_hidden,
                next_chain_count,
            )?;
            self.state.advance(chain_verify_tokens);
        } else {
            // Case C: chain rejects at j < chain_depth. recover_after_mtp_multi_reject
            // already calls generate_mtp_drafts_from_committed_prefill internally;
            // capture its next_draft_tokens.
            let recover = self.recover_after_mtp_multi_reject(
                current_token,
                chain_tokens,
                result.accepted_chain,
                result.next_token,
                start_position,
                total_positions_written,
                next_chain_count,
            )?;
            result.next_chain_drafts = recover.next_draft_tokens;
            // recover_after_mtp_multi_reject already advances state.position.
        }

        // Sample top-K leaves from forward.logits. The guard
        // `!result.next_chain_drafts.is_empty()` is what keeps this safe:
        // when generate_mtp_drafts_from_committed_prefill (or the recovery
        // path that delegates to it) actually ran, it left forward.logits
        // holding the LAST MTP head step's logits — exactly what we want
        // for top-K leaf sampling. If next_chain_drafts is empty, that
        // primitive was a no-op and forward.logits is stale (whichever
        // unrelated kernel last wrote it), so skip sampling.
        if leaf_count > 0 && !result.next_chain_drafts.is_empty() {
            let leaf_buffer = self.cuda_forward()?.leaf_tokens_u32.ptr();
            self.queue_sample_topk_into(leaf_buffer, leaf_count)?;
            let mut leaf_bytes = vec![0u8; leaf_count * 4];
            self.cuda_forward()?
                .leaf_tokens_u32
                .copy_to_host(&mut leaf_bytes)?;
            result.next_leaf_drafts = leaf_bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
        }

        Ok(result)
    }

    /// Fallback when QWEN36_MTP_TREE_DISABLE=1: degrade to chain MTP using only
    /// the top-1 leaf as next chain draft (effectively MTP=chain_depth+1 chain).
    #[cfg(feature = "cuda")]
    fn verify_mtp_tree_draft_chain_fallback(
        &mut self,
        current_token: u32,
        chain_tokens: &[u32],
        leaf_tokens: &[u32],
        next_draft_count: usize,
    ) -> Result<qwen36_fp4_mtp::TreeVerifyResult> {
        use qwen36_fp4_mtp::TreeVerifyResult;

        // When tree is disabled, just run the chain (drop leaves entirely).
        // Construct a TreeVerifyResult that matches the chain-only behaviour.
        let chain_result =
            self.verify_mtp_draft_tokens(current_token, chain_tokens, true, next_draft_count)?;

        let accepted_chain = chain_result.accepted_drafts;
        // Fail loudly rather than silently substituting token 0 (a real
        // BOS-like token in Qwen vocab); a missing next_token here would mean
        // the underlying chain MTP returned no result on a path where it
        // promised one.
        let next_token = chain_result.next_token.ok_or_else(|| {
            CoreError::Runtime(
                "tree-MTP kill-switch fallback: chain MTP returned no next_token".into(),
            )
        })?;
        let mut committed: Vec<u32> = Vec::with_capacity(accepted_chain + 1);
        for chain_token in chain_tokens.iter().take(accepted_chain).copied() {
            committed.push(chain_token);
        }
        committed.push(next_token);
        let _ = leaf_tokens;
        Ok(TreeVerifyResult {
            committed,
            accepted_chain,
            accepted_leaf: None,
            next_token,
            next_chain_drafts: chain_result.next_draft_tokens,
            next_leaf_drafts: Vec::new(),
        })
    }

    /// Roll back the MTP verify chunk and commit just `current_token` at
    /// `start_position`, leaving the engine in the same state as if the CLI
    /// had run a single non-speculative decode step. When the caller needs a
    /// new draft token (i.e. it will iterate again), this also runs the MTP
    /// layer using `verified_token` as the shifted input so the returned
    /// draft is the same one a fresh `prepare_mtp_prefill_from_sampled` call
    /// would produce — at the cost of one extra layer-pass instead of a full
    /// prompt reprefill.
    #[cfg(feature = "cuda")]
    fn recover_after_mtp_reject(
        &mut self,
        current_token: u32,
        verified_token: u32,
        start_position: usize,
        need_new_draft: bool,
    ) -> Result<MtpVerifyResult> {
        self.mtp_restore_state(start_position, 2)?;

        let token_bytes = current_token.to_ne_bytes();
        let position = i32::try_from(start_position).map_err(|_| {
            CoreError::Runtime(format!(
                "position {start_position} does not fit i32 for RoPE"
            ))
        })?;
        let position_bytes = position.to_ne_bytes();
        {
            let prefill = self.cuda_prefill()?;
            prefill.token_u32.copy_from_host(&token_bytes)?;
            prefill.position_i32.copy_from_host(&position_bytes)?;
        }

        // Single-token main forward to commit `current_token` at
        // `start_position` (writes its K/V into the cache and advances the
        // DeltaNet recurrent state).
        self.prefill_cuda_chunk(1, start_position, DevicePtr::NULL, false)?;
        self.final_norm_prefill_rows(1)?;

        let next_draft_token = if need_new_draft {
            // Run MTP using `verified_token` as the shifted input on top of
            // the hidden state we just produced for `current_token`. This is
            // the same computation a fresh `prepare_mtp_prefill_from_sampled`
            // would do after the prompt prefill, but reusing the recovery
            // step we already paid for.
            let shifted_bytes = verified_token.to_ne_bytes();
            self.cuda_prefill()?
                .token_u32
                .copy_from_host(&shifted_bytes)?;
            let normed_ptr = self.cuda_prefill()?.normed.ptr();
            self.run_mtp_prefill_chunk(1, start_position, DevicePtr::NULL, normed_ptr, true)?;
            self.queue_sample_greedy()?;
            Some(self.read_sampled_token()?)
        } else {
            None
        };

        self.state.advance(1);
        Ok(MtpVerifyResult {
            accepted: false,
            verified_token,
            next_token: None,
            next_draft_token,
        })
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_prefill_chunk_from_current_prefill(
        &self,
        tokens: usize,
        start_position: usize,
        start_position_device_i32: DevicePtr,
        shifted_tokens: &[u32],
        emit_logits: bool,
    ) -> Result<()> {
        if shifted_tokens.len() != tokens {
            return Err(CoreError::Runtime(format!(
                "MTP shifted token count {} does not match chunk size {tokens}",
                shifted_tokens.len()
            )));
        }
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let prefill = self.cuda_prefill()?;
        let mut token_bytes = Vec::with_capacity(tokens * 4);
        for token in shifted_tokens {
            token_bytes.extend_from_slice(&token.to_ne_bytes());
        }
        prefill.token_u32.copy_from_host(&token_bytes)?;
        self.rmsnorm(
            tokens,
            self.topology.hidden_size,
            prefill.hidden.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &manifest.final_norm)?,
            prefill.residual.ptr(),
            DevicePtr::NULL,
            prefill.normed.ptr(),
        )?;
        self.run_mtp_prefill_chunk(
            tokens,
            start_position,
            start_position_device_i32,
            prefill.normed.ptr(),
            emit_logits,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_prefill_chunk(
        &self,
        tokens: usize,
        start_position: usize,
        start_position_device_i32: DevicePtr,
        target_hidden_bf16: DevicePtr,
        emit_logits: bool,
    ) -> Result<()> {
        self.run_mtp_prefill_chunk_with_tokens(
            tokens,
            start_position,
            start_position_device_i32,
            target_hidden_bf16,
            self.cuda_prefill()?.token_u32.ptr(),
            emit_logits,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_prefill_chunk_with_tokens(
        &self,
        tokens: usize,
        start_position: usize,
        start_position_device_i32: DevicePtr,
        target_hidden_bf16: DevicePtr,
        token_ids_u32: DevicePtr,
        emit_logits: bool,
    ) -> Result<()> {
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let mtp = manifest
            .mtp
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("MTP weights are not available".to_owned()))?;
        let layer = mtp
            .layer(0)
            .ok_or_else(|| CoreError::Runtime("MTP has no layers".to_owned()))?;
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let prefill = self.cuda_prefill()?;
        let hidden = self.topology.hidden_size;

        self.backend.embedding_lookup(&EmbeddingLookupSpec {
            tokens,
            hidden,
            vocab_size: self.topology.vocab_size,
            token_ids_u32,
            embedding_bf16: self.tensor_ptr(weights, &manifest.embed_tokens)?,
            output_bf16: prefill.hidden.ptr(),
        })?;
        self.rmsnorm(
            tokens,
            hidden,
            prefill.hidden.ptr(),
            self.tensor_ptr(weights, &mtp.pre_fc_norm_embedding)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.aux.ptr(),
        )?;
        self.rmsnorm(
            tokens,
            hidden,
            target_hidden_bf16,
            self.tensor_ptr(weights, &mtp.pre_fc_norm_hidden)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.block_out.ptr(),
        )?;
        self.concat_mtp_fc_input_rows(
            tokens,
            prefill.aux.ptr(),
            prefill.block_out.ptr(),
            prefill.qkv.ptr(),
        )?;
        self.linear_rows(
            &mtp.fc,
            prefill.qkv.ptr(),
            prefill.hidden.ptr(),
            tokens,
            prefill,
        )?;

        self.rmsnorm(
            tokens,
            hidden,
            prefill.hidden.ptr(),
            self.tensor_ptr(weights, &layer.common.input_layernorm)?,
            DevicePtr::NULL,
            prefill.residual.ptr(),
            prefill.normed.ptr(),
        )?;
        self.run_mtp_full_attention_layer_prefill(
            layer,
            runtime,
            prefill,
            tokens,
            start_position,
            start_position_device_i32,
        )?;
        self.rmsnorm(
            tokens,
            hidden,
            prefill.block_out.ptr(),
            self.tensor_ptr(weights, &layer.common.post_attention_layernorm)?,
            prefill.residual.ptr(),
            prefill.residual.ptr(),
            prefill.normed.ptr(),
        )?;
        self.run_mtp_mlp_prefill(&layer.common, prefill, tokens)?;
        self.rmsnorm(
            tokens,
            hidden,
            prefill.hidden.ptr(),
            self.tensor_ptr(weights, &mtp.norm)?,
            prefill.residual.ptr(),
            DevicePtr::NULL,
            prefill.normed.ptr(),
        )?;
        if emit_logits {
            let last_hidden = Self::ptr_offset(prefill.normed.ptr(), (tokens - 1) * hidden * 2)?;
            self.bf16_matvec(
                &manifest.lm_head,
                last_hidden,
                self.cuda_forward()?.logits.ptr(),
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_decode_from_target_hidden(
        &self,
        token_ids_u32: DevicePtr,
        position: usize,
        position_device_i32: DevicePtr,
        target_hidden_bf16: DevicePtr,
    ) -> Result<()> {
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let mtp = manifest
            .mtp
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("MTP weights are not available".to_owned()))?;
        let layer = mtp
            .layer(0)
            .ok_or_else(|| CoreError::Runtime("MTP has no layers".to_owned()))?;
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let forward = self.cuda_forward()?;
        let hidden = self.topology.hidden_size;

        self.backend.embedding_lookup(&EmbeddingLookupSpec {
            tokens: 1,
            hidden,
            vocab_size: self.topology.vocab_size,
            token_ids_u32,
            embedding_bf16: self.tensor_ptr(weights, &manifest.embed_tokens)?,
            output_bf16: forward.hidden.ptr(),
        })?;
        self.rmsnorm(
            1,
            hidden,
            forward.hidden.ptr(),
            self.tensor_ptr(weights, &mtp.pre_fc_norm_embedding)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.aux.ptr(),
        )?;
        self.rmsnorm(
            1,
            hidden,
            target_hidden_bf16,
            self.tensor_ptr(weights, &mtp.pre_fc_norm_hidden)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.block_out.ptr(),
        )?;
        self.concat_mtp_fc_input_rows(
            1,
            forward.aux.ptr(),
            forward.block_out.ptr(),
            forward.qkv.ptr(),
        )?;
        self.linear(&mtp.fc, forward.qkv.ptr(), forward.hidden.ptr())?;

        self.rmsnorm(
            1,
            hidden,
            forward.hidden.ptr(),
            self.tensor_ptr(weights, &layer.common.input_layernorm)?,
            DevicePtr::NULL,
            forward.residual.ptr(),
            forward.normed.ptr(),
        )?;
        self.run_mtp_full_attention_layer_decode(
            layer,
            runtime,
            forward,
            position,
            position_device_i32,
        )?;
        self.rmsnorm(
            1,
            hidden,
            forward.block_out.ptr(),
            self.tensor_ptr(weights, &layer.common.post_attention_layernorm)?,
            forward.residual.ptr(),
            forward.residual.ptr(),
            forward.normed.ptr(),
        )?;
        self.run_mtp_mlp_decode(&layer.common, forward)?;
        self.rmsnorm(
            1,
            hidden,
            forward.hidden.ptr(),
            self.tensor_ptr(weights, &mtp.norm)?,
            forward.residual.ptr(),
            DevicePtr::NULL,
            forward.normed.ptr(),
        )?;
        self.bf16_matvec(
            &manifest.lm_head,
            forward.normed.ptr(),
            forward.logits.ptr(),
        )
    }

    #[cfg(feature = "cuda")]
    fn prefill_cuda_chunk(
        &self,
        tokens: usize,
        start_position: usize,
        start_position_device_i32: DevicePtr,
        emit_logits: bool,
    ) -> Result<()> {
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let prefill = self.cuda_prefill()?;

        let profile_prefill = std::env::var("QWEN36_PROFILE_PREFILL_CHUNKS").is_ok()
            && start_position_device_i32 == DevicePtr::NULL;
        let chunk_profile_start = profile_prefill.then(std::time::Instant::now);
        let mut prof_embed_ms = 0.0_f64;
        let mut prof_input_norm_quant_ms = 0.0_f64;
        let mut prof_linear_attn_ms = 0.0_f64;
        let mut prof_full_attn_ms = 0.0_f64;
        let mut prof_post_norm_quant_ms = 0.0_f64;
        let mut prof_mlp_ms = 0.0_f64;
        let mut prof_logits_ms = 0.0_f64;

        let embed_start = profile_prefill.then(std::time::Instant::now);
        self.backend.embedding_lookup(&EmbeddingLookupSpec {
            tokens,
            hidden: self.topology.hidden_size,
            vocab_size: self.topology.vocab_size,
            token_ids_u32: prefill.token_u32.ptr(),
            embedding_bf16: self.tensor_ptr(weights, &manifest.embed_tokens)?,
            output_bf16: prefill.hidden.ptr(),
        })?;
        if let Some(embed_start) = embed_start {
            qwen36_fp4_kernels::cuda_synchronize()?;
            prof_embed_ms += embed_start.elapsed().as_secs_f64() * 1000.0;
        }

        let trace_layers = std::env::var("QWEN36_DEBUG_LAYER_TRACE").is_ok();
        let dump_dir = std::env::var("QWEN36_DEBUG_DUMP_DIR").ok();
        let dump_all_layers = std::env::var("QWEN36_DEBUG_DUMP_ALL_LAYERS").is_ok();
        if trace_layers {
            self.trace_buffer_stats(
                "post-embed",
                prefill.hidden.ptr(),
                tokens * self.topology.hidden_size,
            )?;
        }
        if let Some(dir) = &dump_dir {
            self.dump_buffer_to_disk(
                dir,
                "post_embed.bf16",
                prefill.hidden.ptr(),
                tokens * self.topology.hidden_size,
            )?;
        }

        let mut residual_initialized = false;
        for (layer_idx, layer) in manifest.layers.iter().enumerate() {
            let input_residual = if residual_initialized {
                prefill.residual.ptr()
            } else {
                DevicePtr::NULL
            };
            let input_norm_start = profile_prefill.then(std::time::Instant::now);
            self.rmsnorm(
                tokens,
                self.topology.hidden_size,
                prefill.hidden.ptr(),
                self.tensor_ptr(weights, layer_common_input_norm(layer))?,
                input_residual,
                prefill.residual.ptr(),
                prefill.normed.ptr(),
            )?;
            residual_initialized = true;
            // Phase E: DFlash drafter hidden-state capture. `prefill.residual`
            // now equals `hidden_states[layer_idx]` (the post-fused-residual
            // input to layer `layer_idx` in transformers' convention). The
            // hook is a no-op when `layer_idx` is not in the configured
            // capture set.
            if let Some(hook) = &self.drafter_hidden_capture {
                hook(layer_idx, prefill.residual.ptr(), tokens)?;
            }
            if dump_all_layers {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_input_normed.bf16"),
                        prefill.normed.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_input_residual.bf16"),
                        prefill.residual.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                }
            }

            let quantized_normed = if let Some(quantized) = Self::layer_input_nvfp4_quant(layer)? {
                self.quantize_nvfp4_activation_rows(
                    prefill.normed.ptr(),
                    tokens,
                    quantized,
                    prefill,
                )?;
                Some(quantized)
            } else {
                None
            };
            if let Some(input_norm_start) = input_norm_start {
                qwen36_fp4_kernels::cuda_synchronize()?;
                prof_input_norm_quant_ms += input_norm_start.elapsed().as_secs_f64() * 1000.0;
            }

            let attn_start = profile_prefill.then(std::time::Instant::now);
            let is_linear_attn = matches!(layer, LayerWeights::LinearAttention(_));
            match layer {
                LayerWeights::LinearAttention(layer) => self.run_linear_attention_layer_prefill(
                    layer,
                    runtime,
                    prefill,
                    tokens,
                    quantized_normed,
                )?,
                LayerWeights::FullAttention(layer) => self.run_full_attention_layer_prefill(
                    layer,
                    runtime,
                    prefill,
                    tokens,
                    start_position,
                    start_position_device_i32,
                    quantized_normed,
                )?,
            }
            if let Some(attn_start) = attn_start {
                qwen36_fp4_kernels::cuda_synchronize()?;
                let elapsed_ms = attn_start.elapsed().as_secs_f64() * 1000.0;
                if is_linear_attn {
                    prof_linear_attn_ms += elapsed_ms;
                } else {
                    prof_full_attn_ms += elapsed_ms;
                }
            }

            if trace_layers {
                let kind = match layer {
                    LayerWeights::LinearAttention(_) => "deltanet",
                    LayerWeights::FullAttention(_) => "fullattn",
                };
                self.trace_buffer_stats(
                    &format!("layer{layer_idx:02}.{kind}.attn_out"),
                    prefill.block_out.ptr(),
                    tokens * self.topology.hidden_size,
                )?;
            }
            if dump_all_layers {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_attn_out.bf16"),
                        prefill.block_out.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                }
            }
            if let Some(dir) = &dump_dir {
                if layer_idx == 0 {
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_attn_out.bf16",
                        prefill.block_out.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_normed.bf16",
                        prefill.normed.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                } else if layer_idx == 3 {
                    self.dump_buffer_to_disk(
                        dir,
                        "layer3_normed.bf16",
                        prefill.normed.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                }
            }

            let common = layer_common(layer);
            let mlp_input_linears = [
                (&common.mlp_gate_proj, DevicePtr::NULL),
                (&common.mlp_up_proj, DevicePtr::NULL),
            ];
            let mlp_quantized = Self::common_nvfp4_quant(&mlp_input_linears)?;
            let post_norm_start = profile_prefill.then(std::time::Instant::now);
            self.rmsnorm(
                tokens,
                self.topology.hidden_size,
                prefill.block_out.ptr(),
                self.tensor_ptr(weights, &common.post_attention_layernorm)?,
                prefill.residual.ptr(),
                prefill.residual.ptr(),
                prefill.normed.ptr(),
            )?;
            if dump_all_layers {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_post_attn_normed.bf16"),
                        prefill.normed.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_residual_after_attn.bf16"),
                        prefill.residual.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                }
            }
            if let Some(quantized) = mlp_quantized {
                self.quantize_nvfp4_activation_rows(
                    prefill.normed.ptr(),
                    tokens,
                    quantized,
                    prefill,
                )?;
            }
            if let Some(post_norm_start) = post_norm_start {
                qwen36_fp4_kernels::cuda_synchronize()?;
                prof_post_norm_quant_ms += post_norm_start.elapsed().as_secs_f64() * 1000.0;
            }

            let mlp_start = profile_prefill.then(std::time::Instant::now);
            if let Some(quantized) = mlp_quantized {
                self.run_mlp_with_quantized_input_prefill(layer, prefill, tokens, quantized)?;
            } else {
                self.run_mlp_prefill(layer, prefill, tokens)?;
            }
            if let Some(mlp_start) = mlp_start {
                qwen36_fp4_kernels::cuda_synchronize()?;
                prof_mlp_ms += mlp_start.elapsed().as_secs_f64() * 1000.0;
            }
            if dump_all_layers {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_mlp_gate.bf16"),
                        prefill.aux.ptr(),
                        tokens * self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_mlp_up.bf16"),
                        prefill.aux2.ptr(),
                        tokens * self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_mlp_swiglu.bf16"),
                        prefill.aux3.ptr(),
                        tokens * self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_mlp_out.bf16"),
                        prefill.hidden.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                }
            }

            if layer_idx == 0 {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_mlp_out.bf16",
                        prefill.hidden.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_mlp_gate.bf16",
                        prefill.aux.ptr(),
                        tokens * self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_mlp_up.bf16",
                        prefill.aux2.ptr(),
                        tokens * self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_mlp_swiglu.bf16",
                        prefill.aux3.ptr(),
                        tokens * self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_post_attn_normed.bf16",
                        prefill.normed.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_residual.bf16",
                        prefill.residual.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                }
            }

            if trace_layers {
                self.trace_buffer_stats(
                    &format!("layer{layer_idx:02}.mlp_out"),
                    prefill.hidden.ptr(),
                    tokens * self.topology.hidden_size,
                )?;
                self.trace_buffer_stats(
                    &format!("layer{layer_idx:02}.residual"),
                    prefill.residual.ptr(),
                    tokens * self.topology.hidden_size,
                )?;
            }
        }

        if emit_logits {
            let logits_start = profile_prefill.then(std::time::Instant::now);
            let last_hidden = Self::ptr_offset(
                prefill.hidden.ptr(),
                (tokens - 1) * self.topology.hidden_size * 2,
            )?;
            let last_residual = Self::ptr_offset(
                prefill.residual.ptr(),
                (tokens - 1) * self.topology.hidden_size * 2,
            )?;
            let forward = self.cuda_forward()?;
            // Dump pre-final-norm residual + hidden for the parity harness.
            if let Some(dir) = &dump_dir {
                self.dump_buffer_to_disk(
                    dir,
                    "final_last_hidden.bf16",
                    last_hidden,
                    self.topology.hidden_size,
                )?;
                self.dump_buffer_to_disk(
                    dir,
                    "final_last_residual.bf16",
                    last_residual,
                    self.topology.hidden_size,
                )?;
            }
            self.rmsnorm(
                1,
                self.topology.hidden_size,
                last_hidden,
                self.tensor_ptr(weights, &manifest.final_norm)?,
                last_residual,
                DevicePtr::NULL,
                forward.normed.ptr(),
            )?;
            if let Some(dir) = &dump_dir {
                self.dump_buffer_to_disk(
                    dir,
                    "final_normed.bf16",
                    forward.normed.ptr(),
                    self.topology.hidden_size,
                )?;
            }
            self.bf16_matvec(
                &manifest.lm_head,
                forward.normed.ptr(),
                forward.logits.ptr(),
            )?;
            if let Some(logits_start) = logits_start {
                qwen36_fp4_kernels::cuda_synchronize()?;
                prof_logits_ms += logits_start.elapsed().as_secs_f64() * 1000.0;
            }
        }

        if let Some(chunk_profile_start) = chunk_profile_start {
            let measured_ms = prof_embed_ms
                + prof_input_norm_quant_ms
                + prof_linear_attn_ms
                + prof_full_attn_ms
                + prof_post_norm_quant_ms
                + prof_mlp_ms
                + prof_logits_ms;
            eprintln!(
                "prefill.profile.chunk start={start_position} tokens={tokens} embed={:.3} input_norm_quant={:.3} linear_attn={:.3} full_attn={:.3} post_norm_quant={:.3} mlp={:.3} logits={:.3} total_measured={:.3} wall={:.3}",
                prof_embed_ms,
                prof_input_norm_quant_ms,
                prof_linear_attn_ms,
                prof_full_attn_ms,
                prof_post_norm_quant_ms,
                prof_mlp_ms,
                prof_logits_ms,
                measured_ms,
                chunk_profile_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn final_norm_prefill_rows(&self, tokens: usize) -> Result<()> {
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let prefill = self.cuda_prefill()?;
        self.rmsnorm(
            tokens,
            self.topology.hidden_size,
            prefill.hidden.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &manifest.final_norm)?,
            prefill.residual.ptr(),
            DevicePtr::NULL,
            prefill.normed.ptr(),
        )
    }

    #[cfg(feature = "cuda")]
    fn prefill_row_logits(&self, row: usize) -> Result<()> {
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let prefill = self.cuda_prefill()?;
        let hidden = Self::ptr_offset(prefill.normed.ptr(), row * self.topology.hidden_size * 2)?;
        self.bf16_matvec(&manifest.lm_head, hidden, self.cuda_forward()?.logits.ptr())
    }

    #[cfg(feature = "cuda")]
    fn prefill_rows_logits_for_mtp_verify(&self, rows: usize) -> Result<()> {
        if rows == 0 || rows > MTP_MAX_DRAFT_TOKENS + 1 {
            return Err(CoreError::Runtime(format!(
                "MTP batched lm_head expects 1..={} rows, got {rows}",
                MTP_MAX_DRAFT_TOKENS + 1
            )));
        }
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        self.bf16_gemm_rows(
            &manifest.lm_head,
            self.cuda_prefill()?.normed.ptr(),
            self.cuda_forward()?.mtp_logits.ptr(),
            rows,
        )
    }

    #[cfg(feature = "cuda")]
    fn forward_token_cuda(
        &self,
        token: u32,
        position: usize,
        emit_logits: bool,
        sync_after: bool,
    ) -> Result<()> {
        let forward = self.cuda_forward()?;
        forward.token_u32.copy_from_host(&token.to_ne_bytes())?;
        self.forward_device_token_cuda(forward.token_u32.ptr(), position, emit_logits, sync_after)
    }

    #[cfg(feature = "cuda")]
    fn forward_sampled_token_cuda(
        &self,
        position: usize,
        emit_logits: bool,
        sync_after: bool,
    ) -> Result<()> {
        self.forward_device_token_cuda(
            self.cuda_forward()?.sampled_token_u32.ptr(),
            position,
            emit_logits,
            sync_after,
        )
    }

    #[cfg(feature = "cuda")]
    fn forward_device_token_cuda(
        &self,
        token_ids_u32: DevicePtr,
        position: usize,
        emit_logits: bool,
        sync_after: bool,
    ) -> Result<()> {
        self.forward_device_token_cuda_inner(
            token_ids_u32,
            position,
            DevicePtr::NULL,
            emit_logits,
            sync_after,
        )
    }

    #[cfg(feature = "cuda")]
    fn forward_device_token_cuda_inner(
        &self,
        token_ids_u32: DevicePtr,
        position: usize,
        position_device_i32: DevicePtr,
        emit_logits: bool,
        sync_after: bool,
    ) -> Result<()> {
        // The bound check applies to the host-side scalar position only. In
        // graph-capture mode the device-side counter advances per replay, so
        // the caller is responsible for not stepping the graph past
        // max_context.
        if position_device_i32 == DevicePtr::NULL && position >= self.config.max_context {
            return Err(CoreError::Runtime(format!(
                "position {position} exceeds configured max_context {}",
                self.config.max_context
            )));
        }
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let forward = self.cuda_forward()?;
        let dump_decode = std::env::var("QWEN36_DEBUG_DUMP_DECODE").is_ok();
        let dump_dir = std::env::var("QWEN36_DEBUG_DUMP_DIR").ok();
        let dump_prefix = format!("decode_pos{position:05}");
        let profile_decode = std::env::var("QWEN36_PROFILE_DECODE_LAYERS").is_ok()
            && position_device_i32 == DevicePtr::NULL;
        let mut prof_embed_ms = 0.0_f64;
        let mut prof_linear_attn_ms = 0.0_f64;
        let mut prof_full_attn_ms = 0.0_f64;
        let mut prof_mlp_ms = 0.0_f64;
        let mut prof_lm_head_ms = 0.0_f64;

        let embed_start = profile_decode.then(std::time::Instant::now);
        self.backend.embedding_lookup(&EmbeddingLookupSpec {
            tokens: 1,
            hidden: self.topology.hidden_size,
            vocab_size: self.topology.vocab_size,
            token_ids_u32,
            embedding_bf16: self.tensor_ptr(weights, &manifest.embed_tokens)?,
            output_bf16: forward.hidden.ptr(),
        })?;
        if let Some(embed_start) = embed_start {
            qwen36_fp4_kernels::cuda_synchronize()?;
            prof_embed_ms += embed_start.elapsed().as_secs_f64() * 1000.0;
        }
        if dump_decode {
            if let Some(dir) = &dump_dir {
                self.dump_buffer_to_disk(
                    dir,
                    &format!("{dump_prefix}_post_embed.bf16"),
                    forward.hidden.ptr(),
                    self.topology.hidden_size,
                )?;
            }
        }

        let mut residual_initialized = false;
        for (layer_idx, layer) in manifest.layers.iter().enumerate() {
            let input_residual = if residual_initialized {
                forward.residual.ptr()
            } else {
                DevicePtr::NULL
            };
            let quantized_normed = Self::layer_input_nvfp4_quant(layer)?;
            let mut ran_attention = false;
            let mut ran_transformer_layer = false;

            if !dump_decode && position_device_i32 != DevicePtr::NULL {
                if let Some(program) =
                    self.decode_interpreter_graph_layer_program(layer_idx, position_device_i32)
                {
                    program.run(&self.backend)?;
                    residual_initialized = true;
                    ran_attention = true;
                    ran_transformer_layer = true;
                }
            }

            if !ran_transformer_layer && !dump_decode && position_device_i32 == DevicePtr::NULL {
                if let LayerWeights::LinearAttention(linear) = layer {
                    if let Some(input_quantized) = quantized_normed {
                        let common = &linear.common;
                        let mlp_input_linears = [
                            (&common.mlp_gate_proj, DevicePtr::NULL),
                            (&common.mlp_up_proj, DevicePtr::NULL),
                        ];
                        if let Some(mlp_quantized) = Self::common_nvfp4_quant(&mlp_input_linears)? {
                            let layer_start = profile_decode.then(std::time::Instant::now);
                            ran_transformer_layer = self
                                .run_interpreter_linear_transformer_layer_decode(
                                    linear,
                                    runtime,
                                    forward,
                                    input_quantized,
                                    mlp_quantized,
                                    input_residual,
                                    self.tensor_ptr(weights, &common.input_layernorm)?,
                                    self.tensor_ptr(weights, &common.post_attention_layernorm)?,
                                )?;
                            if ran_transformer_layer {
                                residual_initialized = true;
                                ran_attention = true;
                                if let Some(layer_start) = layer_start {
                                    qwen36_fp4_kernels::cuda_synchronize()?;
                                    prof_linear_attn_ms +=
                                        layer_start.elapsed().as_secs_f64() * 1000.0;
                                }
                            }
                        }
                    }
                }
                if !ran_transformer_layer {
                    if let LayerWeights::FullAttention(full) = layer {
                        if let Some(input_quantized) = quantized_normed {
                            let common = &full.common;
                            let mlp_input_linears = [
                                (&common.mlp_gate_proj, DevicePtr::NULL),
                                (&common.mlp_up_proj, DevicePtr::NULL),
                            ];
                            if let Some(mlp_quantized) =
                                Self::common_nvfp4_quant(&mlp_input_linears)?
                            {
                                let layer_start = profile_decode.then(std::time::Instant::now);
                                ran_transformer_layer = self
                                    .run_interpreter_full_transformer_layer_decode(
                                        full,
                                        runtime,
                                        forward,
                                        position,
                                        input_quantized,
                                        mlp_quantized,
                                        input_residual,
                                        self.tensor_ptr(weights, &common.input_layernorm)?,
                                        self.tensor_ptr(weights, &common.post_attention_layernorm)?,
                                    )?;
                                if ran_transformer_layer {
                                    residual_initialized = true;
                                    ran_attention = true;
                                    if let Some(layer_start) = layer_start {
                                        qwen36_fp4_kernels::cuda_synchronize()?;
                                        prof_full_attn_ms +=
                                            layer_start.elapsed().as_secs_f64() * 1000.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if !ran_attention && !dump_decode && position_device_i32 == DevicePtr::NULL {
                if let Some(quantized) = quantized_normed {
                    let attn_start = profile_decode.then(std::time::Instant::now);
                    ran_attention = match layer {
                        LayerWeights::LinearAttention(linear) => self
                            .run_interpreter_linear_attention_input_layer_decode(
                                linear,
                                runtime,
                                forward,
                                quantized,
                                input_residual,
                                self.tensor_ptr(weights, layer_common_input_norm(layer))?,
                            )?,
                        LayerWeights::FullAttention(full) => self
                            .run_interpreter_full_attention_input_layer_decode(
                                full,
                                runtime,
                                forward,
                                position,
                                quantized,
                                input_residual,
                                self.tensor_ptr(weights, layer_common_input_norm(layer))?,
                            )?,
                    };
                    if ran_attention {
                        residual_initialized = true;
                        if let Some(hook) = &self.drafter_hidden_capture {
                            hook(layer_idx, forward.residual.ptr(), 1)?;
                        }
                        if let Some(attn_start) = attn_start {
                            qwen36_fp4_kernels::cuda_synchronize()?;
                            let elapsed = attn_start.elapsed().as_secs_f64() * 1000.0;
                            match layer {
                                LayerWeights::LinearAttention(_) => prof_linear_attn_ms += elapsed,
                                LayerWeights::FullAttention(_) => prof_full_attn_ms += elapsed,
                            }
                        }
                    }
                }
            }

            if !ran_attention {
                if let Some(quantized) = quantized_normed {
                    let output_bf16 = if dump_decode {
                        forward.normed.ptr()
                    } else {
                        DevicePtr::NULL
                    };
                    self.rmsnorm_nvfp4_quantize(
                        self.topology.hidden_size,
                        forward.hidden.ptr(),
                        self.tensor_ptr(weights, layer_common_input_norm(layer))?,
                        input_residual,
                        forward.residual.ptr(),
                        output_bf16,
                        quantized.input_scale,
                        position_device_i32 == DevicePtr::NULL,
                    )?;
                } else {
                    self.rmsnorm(
                        1,
                        self.topology.hidden_size,
                        forward.hidden.ptr(),
                        self.tensor_ptr(weights, layer_common_input_norm(layer))?,
                        input_residual,
                        forward.residual.ptr(),
                        forward.normed.ptr(),
                    )?;
                };
                residual_initialized = true;
                // Phase F.2: DFlash drafter hidden-state capture on the
                // decode path. Mirrors the prefill hook (Phase E); tokens=1
                // since decode is per-token. `forward.residual` now equals
                // `hidden_states[layer_idx]` for the single decoded token.
                if let Some(hook) = &self.drafter_hidden_capture {
                    hook(layer_idx, forward.residual.ptr(), 1)?;
                }
                if dump_decode {
                    if let Some(dir) = &dump_dir {
                        self.dump_buffer_to_disk(
                            dir,
                            &format!("{dump_prefix}_layer{layer_idx:02}_input_normed.bf16"),
                            forward.normed.ptr(),
                            self.topology.hidden_size,
                        )?;
                    }
                }

                let attn_start = profile_decode.then(std::time::Instant::now);
                let attention_was_linear = matches!(layer, LayerWeights::LinearAttention(_));
                match layer {
                    LayerWeights::LinearAttention(layer) => {
                        self.run_linear_attention_layer(
                            layer,
                            runtime,
                            forward,
                            quantized_normed,
                            position_device_i32 == DevicePtr::NULL,
                        )?;
                    }
                    LayerWeights::FullAttention(layer) => {
                        self.run_full_attention_layer(
                            layer,
                            runtime,
                            forward,
                            position,
                            position_device_i32,
                            quantized_normed,
                        )?;
                    }
                }
                if let Some(attn_start) = attn_start {
                    qwen36_fp4_kernels::cuda_synchronize()?;
                    let elapsed = attn_start.elapsed().as_secs_f64() * 1000.0;
                    if attention_was_linear {
                        prof_linear_attn_ms += elapsed;
                    } else {
                        prof_full_attn_ms += elapsed;
                    }
                }
            }
            if ran_transformer_layer {
                if let Some(hook) = &self.drafter_hidden_capture {
                    hook(layer_idx, forward.normed.ptr(), 1)?;
                }
                continue;
            }
            if dump_decode {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_attn_out.bf16"),
                        forward.block_out.ptr(),
                        self.topology.hidden_size,
                    )?;
                }
            }

            let common = layer_common(layer);
            let mlp_input_linears = [
                (&common.mlp_gate_proj, DevicePtr::NULL),
                (&common.mlp_up_proj, DevicePtr::NULL),
            ];
            if let Some(quantized) = Self::common_nvfp4_quant(&mlp_input_linears)? {
                let output_bf16 = if dump_decode {
                    forward.normed.ptr()
                } else {
                    DevicePtr::NULL
                };
                let post_attention_norm_weight =
                    self.tensor_ptr(weights, &common.post_attention_layernorm)?;
                let mut ran_norm_mlp = false;
                if !dump_decode
                    && position_device_i32 == DevicePtr::NULL
                    && decode_interpreter_norm_mlp_enabled(self.decode_interpreter_decode_enabled())
                {
                    let mlp_start = profile_decode.then(std::time::Instant::now);
                    ran_norm_mlp = self.run_interpreter_norm_mlp(
                        layer,
                        forward,
                        quantized,
                        self.topology.hidden_size,
                        forward.block_out.ptr(),
                        post_attention_norm_weight,
                        forward.residual.ptr(),
                        forward.residual.ptr(),
                        output_bf16,
                        quantized.input_scale,
                    )?;
                    if ran_norm_mlp {
                        if let Some(mlp_start) = mlp_start {
                            qwen36_fp4_kernels::cuda_synchronize()?;
                            prof_mlp_ms += mlp_start.elapsed().as_secs_f64() * 1000.0;
                        }
                    }
                }
                if !ran_norm_mlp {
                    self.rmsnorm_nvfp4_quantize(
                        self.topology.hidden_size,
                        forward.block_out.ptr(),
                        post_attention_norm_weight,
                        forward.residual.ptr(),
                        forward.residual.ptr(),
                        output_bf16,
                        quantized.input_scale,
                        position_device_i32 == DevicePtr::NULL,
                    )?;
                    if dump_decode {
                        if let Some(dir) = &dump_dir {
                            self.dump_buffer_to_disk(
                                dir,
                                &format!("{dump_prefix}_layer{layer_idx:02}_post_attn_normed.bf16"),
                                forward.normed.ptr(),
                                self.topology.hidden_size,
                            )?;
                            self.dump_bytes_to_disk(
                                dir,
                                &format!(
                                    "{dump_prefix}_layer{layer_idx:02}_post_attn_activation.fp4"
                                ),
                                forward.activation_fp4.ptr(),
                                self.topology.hidden_size.div_ceil(2),
                            )?;
                            self.dump_bytes_to_disk(
                                dir,
                                &format!(
                                    "{dump_prefix}_layer{layer_idx:02}_post_attn_activation_scale.e4m3"
                                ),
                                forward.activation_scale.ptr(),
                                self.topology.hidden_size.div_ceil(16).div_ceil(4) * 512,
                            )?;
                        }
                    }
                    let mlp_start = profile_decode.then(std::time::Instant::now);
                    let interpreter_mlp_eligible =
                        decode_interpreter_mlp_enabled(self.decode_interpreter_decode_enabled())
                            && position_device_i32 == DevicePtr::NULL
                            && decode_interpreter_mlp_supports(
                                self.topology.hidden_size,
                                self.topology.intermediate_size,
                            );
                    let ran_interpreter_mlp = interpreter_mlp_eligible
                        && self
                            .run_interpreter_mlp_with_quantized_input(layer, forward, quantized)?;
                    if !ran_interpreter_mlp {
                        self.run_mlp_with_quantized_input(layer, forward, quantized)?;
                    }
                    if let Some(mlp_start) = mlp_start {
                        qwen36_fp4_kernels::cuda_synchronize()?;
                        prof_mlp_ms += mlp_start.elapsed().as_secs_f64() * 1000.0;
                    }
                }
            } else {
                self.rmsnorm(
                    1,
                    self.topology.hidden_size,
                    forward.block_out.ptr(),
                    self.tensor_ptr(weights, &common.post_attention_layernorm)?,
                    forward.residual.ptr(),
                    forward.residual.ptr(),
                    forward.normed.ptr(),
                )?;
                if dump_decode {
                    if let Some(dir) = &dump_dir {
                        self.dump_buffer_to_disk(
                            dir,
                            &format!("{dump_prefix}_layer{layer_idx:02}_post_attn_normed.bf16"),
                            forward.normed.ptr(),
                            self.topology.hidden_size,
                        )?;
                    }
                }
                let mlp_start = profile_decode.then(std::time::Instant::now);
                self.run_mlp(layer, forward)?;
                if let Some(mlp_start) = mlp_start {
                    qwen36_fp4_kernels::cuda_synchronize()?;
                    prof_mlp_ms += mlp_start.elapsed().as_secs_f64() * 1000.0;
                }
            }
            if dump_decode {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_mlp_gate.bf16"),
                        forward.aux.ptr(),
                        self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_mlp_up.bf16"),
                        forward.aux2.ptr(),
                        self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_mlp_swiglu.bf16"),
                        forward.aux3.ptr(),
                        self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_residual_after_attn.bf16"),
                        forward.residual.ptr(),
                        self.topology.hidden_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_mlp_out.bf16"),
                        forward.hidden.ptr(),
                        self.topology.hidden_size,
                    )?;
                }
            }
        }

        if emit_logits {
            let logits_start = profile_decode.then(std::time::Instant::now);
            if decode_interpreter_logits_enabled() && position_device_i32 == DevicePtr::NULL {
                self.run_interpreter_final_logits(manifest, weights, forward)?;
            } else {
                self.rmsnorm(
                    1,
                    self.topology.hidden_size,
                    forward.hidden.ptr(),
                    self.tensor_ptr(weights, &manifest.final_norm)?,
                    forward.residual.ptr(),
                    DevicePtr::NULL,
                    forward.normed.ptr(),
                )?;
                self.bf16_matvec(
                    &manifest.lm_head,
                    forward.normed.ptr(),
                    forward.logits.ptr(),
                )?;
            }
            if dump_decode {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_final_normed.bf16"),
                        forward.normed.ptr(),
                        self.topology.hidden_size,
                    )?;
                }
            }
            if let Some(logits_start) = logits_start {
                qwen36_fp4_kernels::cuda_synchronize()?;
                prof_lm_head_ms += logits_start.elapsed().as_secs_f64() * 1000.0;
            }
            if dump_decode {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_logits.bf16"),
                        forward.logits.ptr(),
                        self.topology.vocab_size,
                    )?;
                }
            }
        }
        if sync_after {
            qwen36_fp4_kernels::cuda_synchronize()?;
        }
        if profile_decode {
            let total_ms = prof_embed_ms
                + prof_linear_attn_ms
                + prof_full_attn_ms
                + prof_mlp_ms
                + prof_lm_head_ms;
            eprintln!(
                "decode.profile.summary pos={position} embed={:.3} linear_attn={:.3} full_attn={:.3} mlp={:.3} lm_head={:.3} total_measured={:.3}",
                prof_embed_ms,
                prof_linear_attn_ms,
                prof_full_attn_ms,
                prof_mlp_ms,
                prof_lm_head_ms,
                total_ms
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn run_linear_attention_layer(
        &self,
        layer: &LinearAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        prequantized_normed: Option<Nvfp4ActivationQuant<'_>>,
        interpreter_allowed: bool,
    ) -> Result<()> {
        let qkv_dim = self.topology.linear_attention_qkv_dim();
        let key_dim = self.topology.linear_num_key_heads * self.topology.linear_key_head_dim;
        let value_dim = self.topology.linear_attention_value_dim();
        let layer_ordinal = self.linear_layer_ordinal(layer.layer_index)?;
        let conv_history = runtime.conv_history.ptr_at(
            layer_ordinal * qkv_dim * self.topology.linear_conv_kernel_dim.saturating_sub(1) * 2,
        )?;
        let state = runtime
            .deltanet_state
            .ptr_at(layer_ordinal * self.state.deltanet.state_bytes_per_layer as usize)?;

        if interpreter_allowed {
            if let Some(fused) = self.linear_attn_in_proj_fused_layer_opt(layer.layer_index) {
                let quantized = match prequantized_normed {
                    Some(q) => q,
                    None => {
                        let q = self.linear_attn_in_proj_quant(layer)?;
                        self.quantize_nvfp4_activation(forward.normed.ptr(), q)?;
                        q
                    }
                };
                if self.run_interpreter_linear_attention_layer_decode(
                    layer,
                    fused,
                    forward,
                    quantized,
                    conv_history,
                    state,
                )? {
                    return Ok(());
                }
            }
        }

        let (conv_input_ptr, b_ptr, a_ptr, z_ptr) = if let Some(fused) =
            self.linear_attn_in_proj_fused_layer_opt(layer.layer_index)
        {
            // Combined qkv+b+a+z GEMM. Output layout in `forward.qkv` (BF16):
            //   [qkv: qkv_dim] [b padded: 128] [a padded: 128] [z: value_dim].
            // The 80-row pad after b and a sits at FP4 weight 0 so the GEMM
            // emits zeros there — the engine simply does not read those slots.
            let quantized = match prequantized_normed {
                Some(q) => q,
                None => {
                    let q = self.linear_attn_in_proj_quant(layer)?;
                    self.quantize_nvfp4_activation(forward.normed.ptr(), q)?;
                    q
                }
            };
            self.run_linear_attn_in_proj_fused_gemm(layer, fused, quantized, forward.qkv.ptr())?;

            let b_ptr = forward
                .qkv
                .ptr()
                .offset_bytes(fused.b_offset * 2)
                .ok_or_else(|| CoreError::Runtime("fused DeltaNet b offset overflow".to_owned()))?;
            let a_ptr = forward
                .qkv
                .ptr()
                .offset_bytes(fused.a_offset * 2)
                .ok_or_else(|| CoreError::Runtime("fused DeltaNet a offset overflow".to_owned()))?;
            let z_ptr = forward
                .qkv
                .ptr()
                .offset_bytes(fused.z_offset * 2)
                .ok_or_else(|| CoreError::Runtime("fused DeltaNet z offset overflow".to_owned()))?;
            (forward.qkv.ptr(), b_ptr, a_ptr, z_ptr)
        } else {
            let in_proj_linears = [
                (&layer.in_proj_qkv, forward.qkv.ptr()),
                (&layer.in_proj_b, forward.aux2.ptr()),
                (&layer.in_proj_a, forward.aux3.ptr()),
            ];
            if let Some(quantized) = prequantized_normed {
                for &(binding, output) in &in_proj_linears {
                    self.linear_with_quantized_nvfp4(binding, output, quantized)?;
                }
            } else {
                self.linears_same_input(forward.normed.ptr(), &in_proj_linears)?;
            }
            (
                forward.qkv.ptr(),
                forward.aux2.ptr(),
                forward.aux3.ptr(),
                forward.qkv.ptr(),
            )
        };

        let deltanet_spec = DeltaNetDecodeSpec {
            layer_index: layer.layer_index,
            tokens_in_persistent_loop: 1,
            q_token_stride: 0,
            k_token_stride: 0,
            v_token_stride: 0,
            q_bf16: forward.aux.ptr(),
            k_bf16: forward.aux.ptr_at(key_dim * 2)?,
            v_bf16: forward.aux.ptr_at(key_dim * 4)?,
            state_bf16: state,
            conv_history_bf16: conv_history,
            output_bf16: forward.aux3.ptr(),
            gate_f32: forward.gate_f32.ptr(),
            beta_f32: forward.beta_f32.ptr(),
            shape: DeltaNetShape {
                qk_heads: self.topology.linear_num_key_heads,
                v_heads: self.topology.linear_num_value_heads,
                key_dim: self.topology.linear_key_head_dim,
                value_dim: self.topology.linear_value_head_dim,
                conv_kernel: self.topology.linear_conv_kernel_dim,
            },
            state_decay: 1.0,
            update_scale: 1.0,
            qk_l2norm: true,
        };
        if interpreter_allowed
            && z_ptr != conv_input_ptr
            && self.run_interpreter_linear_attention_post_inproj_decode(
                layer,
                forward,
                conv_input_ptr,
                conv_history,
                a_ptr,
                b_ptr,
                z_ptr,
                &deltanet_spec,
            )?
        {
            return Ok(());
        }
        self.backend
            .conv1d_gdn_gate_fused(&Conv1dGdnGateFusedSpec {
                channels: qkv_dim,
                kernel_size: self.topology.linear_conv_kernel_dim,
                conv_input_bf16: conv_input_ptr,
                conv_history_bf16: conv_history,
                conv_weight_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.conv1d_weight)?,
                conv_output_bf16: forward.aux.ptr(),
                heads: self.topology.linear_num_value_heads,
                gdn_a_bf16: a_ptr,
                gdn_b_bf16: b_ptr,
                gdn_a_log_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.a_log)?,
                gdn_dt_bias_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.dt_bias)?,
                gate_f32: forward.gate_f32.ptr(),
                beta_f32: forward.beta_f32.ptr(),
            })?;
        if interpreter_allowed
            && decode_interpreter_deltanet_enabled(self.decode_interpreter_decode_enabled())
        {
            self.run_interpreter_deltanet_decode(&deltanet_spec, forward)?;
        } else {
            self.backend.deltanet_decode(&deltanet_spec)?;
        }
        if self
            .linear_attn_in_proj_fused_layer_opt(layer.layer_index)
            .is_none()
        {
            if let Some(quantized) = prequantized_normed {
                self.linear_with_quantized_nvfp4(&layer.in_proj_z, forward.qkv.ptr(), quantized)?;
            } else {
                self.linear(&layer.in_proj_z, forward.normed.ptr(), forward.qkv.ptr())?;
            }
        }
        // `z_ptr` is either the fused-GEMM slice or the fallback z projection.
        if interpreter_allowed
            && self.run_interpreter_linear_attention_tail_decode(
                layer,
                forward,
                z_ptr,
                forward.aux3.ptr(),
            )?
        {
            return Ok(());
        }
        self.rmsnorm_direct_weight(
            self.topology.linear_num_value_heads,
            self.topology.linear_value_head_dim,
            forward.aux3.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.norm_weight)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.aux2.ptr(),
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: 1,
            intermediate: value_dim,
            gate_bf16: z_ptr,
            up_bf16: forward.aux2.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.linear(&layer.out_proj, forward.aux3.ptr(), forward.block_out.ptr())
    }

    #[cfg(feature = "cuda")]
    fn run_full_attention_layer(
        &self,
        layer: &FullAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        position: usize,
        position_device_i32: DevicePtr,
        prequantized_normed: Option<Nvfp4ActivationQuant<'_>>,
    ) -> Result<()> {
        let cache = runtime
            .kv_cache
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("KV cache was not allocated".to_owned()))?;
        let layout = self
            .state
            .kv_cache
            .layers
            .iter()
            .find(|layout| layout.global_layer_index == layer.layer_index)
            .ok_or_else(|| {
                CoreError::Runtime(format!(
                    "missing KV-cache layout for layer {}",
                    layer.layer_index
                ))
            })?;

        if let Some(quantized) = prequantized_normed {
            if self.run_interpreter_full_attention_layer_decode(
                layer,
                runtime,
                forward,
                position,
                position_device_i32,
                quantized,
            )? {
                return Ok(());
            }
        }

        let qkv_linears = [
            (&layer.q_proj, forward.qkv.ptr()),
            (&layer.k_proj, forward.aux.ptr()),
            (&layer.v_proj, forward.aux2.ptr()),
        ];
        if let Some(quantized) = prequantized_normed {
            for &(binding, output) in &qkv_linears {
                self.linear_with_quantized_nvfp4(binding, output, quantized)?;
            }
        } else {
            self.linears_same_input(forward.normed.ptr(), &qkv_linears)?;
        }
        self.backend.q_proj_deinterleave(&QProjDeinterleaveSpec {
            rows: 1,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            input_bf16: forward.qkv.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.rmsnorm(
            self.topology.attention_num_heads,
            self.topology.attention_head_dim,
            forward.aux3.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.q_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.aux3.ptr(),
        )?;
        self.rmsnorm(
            self.topology.attention_num_kv_heads,
            self.topology.attention_head_dim,
            forward.aux.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.k_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.aux.ptr(),
        )?;
        let rope_spec = PartialRopeSpec {
            tokens: 1,
            q_heads: self.topology.attention_num_heads,
            kv_heads: self.topology.attention_num_kv_heads,
            head_dim: self.topology.attention_head_dim,
            rope_dims: self.topology.attention_rope_dims(),
            base_theta: self.topology.rope_theta,
            position_i32: position as i32,
            use_scalar_position: true,
            positions_i32: DevicePtr::NULL,
            q_bf16: forward.aux3.ptr(),
            k_bf16: forward.aux.ptr(),
            scalar_position_device_i32: position_device_i32,
        };
        let attention_context_limit = self.decode_attention_context_limit_for_position(position);
        let attention_spec = AttentionDecodeSpec {
            layer_index: layer.layer_index,
            position,
            q_bf16: forward.aux3.ptr(),
            k_bf16: forward.aux.ptr(),
            v_bf16: forward.aux2.ptr(),
            kv_cache_k: cache.ptr_at(layout.k_offset_bytes as usize)?,
            kv_cache_v: cache.ptr_at(layout.v_offset_bytes as usize)?,
            kv_cache_metadata: cache.ptr_at(layout.metadata_offset_bytes as usize)?,
            output_bf16: forward.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            kv_cache_dtype: Self::attention_kv_cache_dtype_code(self.config.kv_cache_dtype)?,
            position_device_i32,
            partial_acc_f32: forward.attn_partial_acc.ptr(),
            partial_max_f32: forward.attn_partial_max.ptr(),
            partial_denom_f32: forward.attn_partial_denom.ptr(),
            decode_n_splits: self
                .decode_attention_n_splits_for_context_limit(attention_context_limit),
            split_timesteps_per_block: self
                .attention_split_timesteps_per_block_for(attention_context_limit),
        };

        // Productive spin: fork L2 prefetch of this layer's MLP combined
        // weight onto the secondary stream so it overlaps the small-CTA
        // attention work. `None` when productive spin is disabled or no MLP
        // fused store exists for this layer.
        let prefetch_join = if decode_interpreter_full_attention_enabled(
            self.decode_interpreter_decode_enabled(),
        ) {
            let prefetch_join = self.fork_productive_spin(layer.layer_index)?;
            self.run_interpreter_rope_attention_decode(&rope_spec, &attention_spec, forward)?;
            prefetch_join
        } else {
            if decode_interpreter_rope_enabled(self.decode_interpreter_decode_enabled()) {
                self.run_interpreter_partial_rope(&rope_spec, forward)?;
            } else {
                self.backend.partial_rope(&rope_spec)?;
            }
            let prefetch_join = self.fork_productive_spin(layer.layer_index)?;
            if decode_interpreter_attention_enabled(self.decode_interpreter_decode_enabled()) {
                self.run_interpreter_attention_decode(&attention_spec, forward)?;
            } else {
                self.backend.attention_decode(&attention_spec)?;
            }
            prefetch_join
        };
        self.backend.q_proj_sigmoid_gate(&QProjSigmoidGateSpec {
            rows: 1,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            gate_bf16: forward.qkv.ptr(),
            input_bf16: forward.aux3.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.linear(&layer.o_proj, forward.aux3.ptr(), forward.block_out.ptr())?;
        // Productive spin: join the prefetch so the MLP that follows (in the
        // caller) finds the L2 warmed up.
        if let Some(event) = prefetch_join {
            self.join_productive_spin(event)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn concat_mtp_fc_input_rows(
        &self,
        rows: usize,
        embedding_normed: DevicePtr,
        target_hidden_normed: DevicePtr,
        output_bf16: DevicePtr,
    ) -> Result<()> {
        let hidden = self.topology.hidden_size;
        self.backend.copy_strided_rows(&CopyStridedRowsSpec {
            rows,
            values: hidden,
            input_stride: hidden,
            output_stride: hidden * 2,
            input_bf16: embedding_normed,
            output_bf16,
        })?;
        self.backend.copy_strided_rows(&CopyStridedRowsSpec {
            rows,
            values: hidden,
            input_stride: hidden,
            output_stride: hidden * 2,
            input_bf16: target_hidden_normed,
            output_bf16: Self::ptr_offset(output_bf16, hidden * 2)?,
        })
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_full_attention_layer_decode(
        &self,
        layer: &FullAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        position: usize,
        position_device_i32: DevicePtr,
    ) -> Result<()> {
        let (kv_cache_k, kv_cache_v) = self.mtp_cache_ptrs(runtime)?;
        let qkv_linears = [
            (&layer.q_proj, forward.qkv.ptr()),
            (&layer.k_proj, forward.aux.ptr()),
            (&layer.v_proj, forward.aux2.ptr()),
        ];
        self.linears_same_input(forward.normed.ptr(), &qkv_linears)?;
        self.backend.q_proj_deinterleave(&QProjDeinterleaveSpec {
            rows: 1,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            input_bf16: forward.qkv.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.rmsnorm(
            self.topology.attention_num_heads,
            self.topology.attention_head_dim,
            forward.aux3.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.q_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.aux3.ptr(),
        )?;
        self.rmsnorm(
            self.topology.attention_num_kv_heads,
            self.topology.attention_head_dim,
            forward.aux.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.k_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.aux.ptr(),
        )?;
        self.backend.partial_rope(&PartialRopeSpec {
            tokens: 1,
            q_heads: self.topology.attention_num_heads,
            kv_heads: self.topology.attention_num_kv_heads,
            head_dim: self.topology.attention_head_dim,
            rope_dims: self.topology.attention_rope_dims(),
            base_theta: self.topology.rope_theta,
            position_i32: position as i32,
            use_scalar_position: true,
            positions_i32: DevicePtr::NULL,
            q_bf16: forward.aux3.ptr(),
            k_bf16: forward.aux.ptr(),
            scalar_position_device_i32: position_device_i32,
        })?;
        let attention_context_limit = self.decode_attention_context_limit_for_position(position);
        self.backend.attention_decode(&AttentionDecodeSpec {
            layer_index: layer.layer_index,
            position,
            q_bf16: forward.aux3.ptr(),
            k_bf16: forward.aux.ptr(),
            v_bf16: forward.aux2.ptr(),
            kv_cache_k,
            kv_cache_v,
            kv_cache_metadata: DevicePtr::NULL,
            output_bf16: forward.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            kv_cache_dtype: Self::attention_kv_cache_dtype_code(KvCacheDtype::Bf16)?,
            position_device_i32,
            partial_acc_f32: forward.attn_partial_acc.ptr(),
            partial_max_f32: forward.attn_partial_max.ptr(),
            partial_denom_f32: forward.attn_partial_denom.ptr(),
            decode_n_splits: self
                .decode_attention_n_splits_for_context_limit(attention_context_limit),
            split_timesteps_per_block: self
                .attention_split_timesteps_per_block_for(attention_context_limit),
        })?;
        self.backend.q_proj_sigmoid_gate(&QProjSigmoidGateSpec {
            rows: 1,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            gate_bf16: forward.qkv.ptr(),
            input_bf16: forward.aux3.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.linear(&layer.o_proj, forward.aux3.ptr(), forward.block_out.ptr())
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_full_attention_layer_prefill(
        &self,
        layer: &FullAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        prefill: &GpuPrefillBuffers,
        tokens: usize,
        start_position: usize,
        start_position_device_i32: DevicePtr,
    ) -> Result<()> {
        let (kv_cache_k, kv_cache_v) = self.mtp_cache_ptrs(runtime)?;
        let qkv_linears = [
            (&layer.q_proj, prefill.qkv.ptr()),
            (&layer.k_proj, prefill.aux.ptr()),
            (&layer.v_proj, prefill.aux2.ptr()),
        ];
        self.linears_same_input_rows(prefill.normed.ptr(), &qkv_linears, tokens, prefill)?;
        self.backend.q_proj_deinterleave(&QProjDeinterleaveSpec {
            rows: tokens,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            input_bf16: prefill.qkv.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.rmsnorm(
            tokens * self.topology.attention_num_heads,
            self.topology.attention_head_dim,
            prefill.aux3.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.q_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.aux3.ptr(),
        )?;
        self.rmsnorm(
            tokens * self.topology.attention_num_kv_heads,
            self.topology.attention_head_dim,
            prefill.aux.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.k_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.aux.ptr(),
        )?;
        self.backend.partial_rope(&PartialRopeSpec {
            tokens,
            q_heads: self.topology.attention_num_heads,
            kv_heads: self.topology.attention_num_kv_heads,
            head_dim: self.topology.attention_head_dim,
            rope_dims: self.topology.attention_rope_dims(),
            base_theta: self.topology.rope_theta,
            position_i32: 0,
            use_scalar_position: false,
            positions_i32: prefill.position_i32.ptr(),
            q_bf16: prefill.aux3.ptr(),
            k_bf16: prefill.aux.ptr(),
            scalar_position_device_i32: DevicePtr::NULL,
        })?;
        self.backend.attention_prefill(&AttentionPrefillSpec {
            layer_index: layer.layer_index,
            start_position,
            tokens,
            q_bf16: prefill.aux3.ptr(),
            k_bf16: prefill.aux.ptr(),
            v_bf16: prefill.aux2.ptr(),
            kv_cache_k,
            kv_cache_v,
            kv_cache_metadata: DevicePtr::NULL,
            output_bf16: prefill.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            kv_cache_dtype: Self::attention_kv_cache_dtype_code(KvCacheDtype::Bf16)?,
            start_position_device_i32,
            partial_acc_f32: self.cuda_forward()?.attn_partial_acc.ptr(),
            partial_max_f32: self.cuda_forward()?.attn_partial_max.ptr(),
            partial_denom_f32: self.cuda_forward()?.attn_partial_denom.ptr(),
            prefill_n_splits: self
                .prefill_attention_n_splits(start_position + tokens, start_position_device_i32),
            split_timesteps_per_block: if start_position_device_i32 == DevicePtr::NULL {
                self.attention_split_timesteps_per_block_for(start_position + tokens)
            } else {
                let context_limit =
                    self.decode_attention_context_limit_for_active_context(start_position + tokens);
                self.attention_split_timesteps_per_block_for(context_limit)
            },
            tree_ancestor_bitmap_u64: DevicePtr::NULL,
            verify_chunk_rows: 0,
        })?;
        self.backend.q_proj_sigmoid_gate(&QProjSigmoidGateSpec {
            rows: tokens,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            gate_bf16: prefill.qkv.ptr(),
            input_bf16: prefill.aux3.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.linear_rows(
            &layer.o_proj,
            prefill.aux3.ptr(),
            prefill.block_out.ptr(),
            tokens,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_mlp_decode(
        &self,
        common: &CommonLayerWeights,
        forward: &GpuForwardBuffers,
    ) -> Result<()> {
        // MTP head weights are BF16, so the fused FP4 GEMM path does not
        // apply. Stay on the per-projection path here.
        self.linears_same_input(
            forward.normed.ptr(),
            &[
                (&common.mlp_gate_proj, forward.aux.ptr()),
                (&common.mlp_up_proj, forward.aux2.ptr()),
            ],
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: 1,
            intermediate: self.topology.intermediate_size,
            gate_bf16: forward.aux.ptr(),
            up_bf16: forward.aux2.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.linear(
            &common.mlp_down_proj,
            forward.aux3.ptr(),
            forward.hidden.ptr(),
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_mlp_prefill(
        &self,
        common: &CommonLayerWeights,
        prefill: &GpuPrefillBuffers,
        tokens: usize,
    ) -> Result<()> {
        self.linears_same_input_rows(
            prefill.normed.ptr(),
            &[
                (&common.mlp_gate_proj, prefill.aux.ptr()),
                (&common.mlp_up_proj, prefill.aux2.ptr()),
            ],
            tokens,
            prefill,
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: tokens,
            intermediate: self.topology.intermediate_size,
            gate_bf16: prefill.aux.ptr(),
            up_bf16: prefill.aux2.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.linear_rows(
            &common.mlp_down_proj,
            prefill.aux3.ptr(),
            prefill.hidden.ptr(),
            tokens,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    fn mtp_cache_ptrs(&self, runtime: &GpuRuntimeBuffers) -> Result<(DevicePtr, DevicePtr)> {
        let cache = runtime
            .mtp_kv_cache
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("MTP KV cache was not allocated".to_owned()))?;
        let plane_bytes = self.mtp_kv_cache_plane_bytes()?;
        Ok((cache.ptr(), cache.ptr_at(plane_bytes)?))
    }

    #[cfg(feature = "cuda")]
    fn run_linear_attention_layer_prefill(
        &self,
        layer: &LinearAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        prefill: &GpuPrefillBuffers,
        tokens: usize,
        prequantized_normed: Option<Nvfp4ActivationQuant<'_>>,
    ) -> Result<()> {
        let qkv_dim = self.topology.linear_attention_qkv_dim();
        let key_dim = self.topology.linear_num_key_heads * self.topology.linear_key_head_dim;
        let value_dim = self.topology.linear_attention_value_dim();
        let layer_ordinal = self.linear_layer_ordinal(layer.layer_index)?;
        let conv_history = runtime.conv_history.ptr_at(
            layer_ordinal * qkv_dim * self.topology.linear_conv_kernel_dim.saturating_sub(1) * 2,
        )?;
        let state = runtime
            .deltanet_state
            .ptr_at(layer_ordinal * self.state.deltanet.state_bytes_per_layer as usize)?;

        let in_proj_linears = [
            (&layer.in_proj_qkv, prefill.qkv.ptr()),
            (&layer.in_proj_b, prefill.aux2.ptr()),
            (&layer.in_proj_a, prefill.aux3.ptr()),
            (&layer.in_proj_z, prefill.qkv.ptr()),
        ];
        let shared_in_proj = match prequantized_normed {
            Some(quantized) => Some(quantized),
            None => Self::common_nvfp4_quant(&in_proj_linears)?,
        };
        let fused_prefill = if cuda_prefill_fused_linear_attn_enabled(self.config.max_context) {
            self.linear_attn_in_proj_fused_layer_opt(layer.layer_index)
        } else {
            None
        };
        let mut used_fused_in_proj = false;
        if let (Some(fused), Some(quantized)) = (fused_prefill, shared_in_proj) {
            if prequantized_normed.is_none() {
                self.quantize_nvfp4_activation_rows(
                    prefill.normed.ptr(),
                    tokens,
                    quantized,
                    prefill,
                )?;
            }
            self.run_linear_attn_in_proj_fused_gemm_rows(
                layer,
                fused,
                quantized,
                prefill.qkv.ptr(),
                tokens,
                prefill,
            )?;
            self.backend.copy_strided_rows(&CopyStridedRowsSpec {
                rows: tokens,
                values: qkv_dim,
                input_stride: fused.combined_out_features,
                output_stride: qkv_dim,
                input_bf16: prefill.qkv.ptr(),
                output_bf16: prefill.block_out.ptr(),
            })?;
            self.backend.copy_strided_rows(&CopyStridedRowsSpec {
                rows: tokens,
                values: self.topology.linear_num_value_heads,
                input_stride: fused.combined_out_features,
                output_stride: self.topology.linear_num_value_heads,
                input_bf16: Self::ptr_offset(prefill.qkv.ptr(), fused.b_offset * 2)?,
                output_bf16: prefill.aux2.ptr(),
            })?;
            self.backend.copy_strided_rows(&CopyStridedRowsSpec {
                rows: tokens,
                values: self.topology.linear_num_value_heads,
                input_stride: fused.combined_out_features,
                output_stride: self.topology.linear_num_value_heads,
                input_bf16: Self::ptr_offset(prefill.qkv.ptr(), fused.a_offset * 2)?,
                output_bf16: prefill.aux3.ptr(),
            })?;
            used_fused_in_proj = true;
        } else if let Some(quantized) = shared_in_proj {
            if prequantized_normed.is_none() {
                self.quantize_nvfp4_activation_rows(
                    prefill.normed.ptr(),
                    tokens,
                    quantized,
                    prefill,
                )?;
            }
            for &(binding, output) in &in_proj_linears[..3] {
                self.linear_with_quantized_nvfp4_rows(binding, output, tokens, quantized, prefill)?;
            }
        } else {
            self.linears_same_input_rows(
                prefill.normed.ptr(),
                &in_proj_linears[..3],
                tokens,
                prefill,
            )?;
        }

        // Dump per-layer intermediates for the parity harness when requested.
        if layer.layer_index == 0 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer0_qkv_raw.bf16",
                    if used_fused_in_proj {
                        prefill.block_out.ptr()
                    } else {
                        prefill.qkv.ptr()
                    },
                    tokens * qkv_dim,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer0_b_raw.bf16",
                    prefill.aux2.ptr(),
                    tokens * self.topology.linear_num_value_heads,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer0_a_raw.bf16",
                    prefill.aux3.ptr(),
                    tokens * self.topology.linear_num_value_heads,
                )?;
            }
        }

        self.backend.conv1d_prefill(&Conv1dPrefillSpec {
            tokens,
            channels: qkv_dim,
            kernel_size: self.topology.linear_conv_kernel_dim,
            input_bf16: if used_fused_in_proj {
                prefill.block_out.ptr()
            } else {
                prefill.qkv.ptr()
            },
            conv_history_bf16: conv_history,
            weight_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.conv1d_weight)?,
            output_bf16: prefill.aux.ptr(),
        })?;
        if layer.layer_index == 0 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer0_conv_out.bf16",
                    prefill.aux.ptr(),
                    tokens * qkv_dim,
                )?;
            }
        }
        self.backend.gdn_gate(&GdnGateSpec {
            rows: tokens,
            heads: self.topology.linear_num_value_heads,
            a_bf16: prefill.aux3.ptr(),
            b_bf16: prefill.aux2.ptr(),
            a_log_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.a_log)?,
            dt_bias_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.dt_bias)?,
            gate_f32: prefill.gate_f32.ptr(),
            beta_f32: prefill.beta_f32.ptr(),
        })?;
        if layer.layer_index == 0 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                let v_heads = self.topology.linear_num_value_heads;
                self.dump_f32_to_disk(
                    &dir,
                    "layer0_gate.f32",
                    prefill.gate_f32.ptr(),
                    tokens * v_heads,
                )?;
                self.dump_f32_to_disk(
                    &dir,
                    "layer0_beta.f32",
                    prefill.beta_f32.ptr(),
                    tokens * v_heads,
                )?;
            }
        }
        let delta_shape = DeltaNetShape {
            qk_heads: self.topology.linear_num_key_heads,
            v_heads: self.topology.linear_num_value_heads,
            key_dim: self.topology.linear_key_head_dim,
            value_dim: self.topology.linear_value_head_dim,
            conv_kernel: self.topology.linear_conv_kernel_dim,
        };
        // Gate on env var + Qwen3.6 shape match (the chunked kernel currently
        // only supports {qk=16, v=48, K=V=128, C=32}).  Falls back to the
        // sequential per-token kernel otherwise.
        let use_chunked = cuda_deltanet_chunked_prefill_enabled()
            && delta_shape.qk_heads == 16
            && delta_shape.v_heads == 48
            && delta_shape.key_dim == 128
            && delta_shape.value_dim == 128;
        if use_chunked {
            self.backend.deltanet_prefill(&DeltaNetPrefillSpec {
                layer_index: layer.layer_index,
                tokens,
                chunk_size: 32,
                q_token_stride: qkv_dim,
                k_token_stride: qkv_dim,
                v_token_stride: qkv_dim,
                q_bf16: prefill.aux.ptr(),
                k_bf16: Self::ptr_offset(prefill.aux.ptr(), key_dim * 2)?,
                v_bf16: Self::ptr_offset(prefill.aux.ptr(), key_dim * 4)?,
                state_bf16: state,
                output_bf16: prefill.aux3.ptr(),
                gate_f32: prefill.gate_f32.ptr(),
                beta_f32: prefill.beta_f32.ptr(),
                workspace: DevicePtr::NULL,
                workspace_bytes: 0,
                shape: delta_shape,
                state_decay: 1.0,
                update_scale: 1.0,
                qk_l2norm: true,
            })?;
        } else {
            self.backend.deltanet_decode(&DeltaNetDecodeSpec {
                layer_index: layer.layer_index,
                tokens_in_persistent_loop: tokens,
                q_token_stride: qkv_dim,
                k_token_stride: qkv_dim,
                v_token_stride: qkv_dim,
                q_bf16: prefill.aux.ptr(),
                k_bf16: Self::ptr_offset(prefill.aux.ptr(), key_dim * 2)?,
                v_bf16: Self::ptr_offset(prefill.aux.ptr(), key_dim * 4)?,
                state_bf16: state,
                conv_history_bf16: conv_history,
                output_bf16: prefill.aux3.ptr(),
                gate_f32: prefill.gate_f32.ptr(),
                beta_f32: prefill.beta_f32.ptr(),
                shape: delta_shape,
                state_decay: 1.0,
                update_scale: 1.0,
                qk_l2norm: true,
            })?;
        }
        if layer.layer_index == 0 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer0_deltanet_out.bf16",
                    prefill.aux3.ptr(),
                    tokens * value_dim,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer0_deltanet_state.bf16",
                    state,
                    self.topology.linear_num_value_heads
                        * self.topology.linear_key_head_dim
                        * self.topology.linear_value_head_dim,
                )?;
            }
        }

        let z_bf16 = if used_fused_in_proj {
            let fused = fused_prefill.ok_or_else(|| {
                CoreError::Runtime("fused DeltaNet prefill state was lost".to_owned())
            })?;
            self.backend.copy_strided_rows(&CopyStridedRowsSpec {
                rows: tokens,
                values: value_dim,
                input_stride: fused.combined_out_features,
                output_stride: value_dim,
                input_bf16: Self::ptr_offset(prefill.qkv.ptr(), fused.z_offset * 2)?,
                output_bf16: prefill.block_out.ptr(),
            })?;
            prefill.block_out.ptr()
        } else if let Some(quantized) = shared_in_proj {
            self.linear_with_quantized_nvfp4_rows(
                &layer.in_proj_z,
                prefill.qkv.ptr(),
                tokens,
                quantized,
                prefill,
            )?;
            prefill.qkv.ptr()
        } else {
            self.linear_rows(
                &layer.in_proj_z,
                prefill.normed.ptr(),
                prefill.qkv.ptr(),
                tokens,
                prefill,
            )?;
            prefill.qkv.ptr()
        };
        if layer.layer_index == 0 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(&dir, "layer0_z.bf16", z_bf16, tokens * value_dim)?;
            }
        }
        self.rmsnorm_direct_weight(
            tokens * self.topology.linear_num_value_heads,
            self.topology.linear_value_head_dim,
            prefill.aux3.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.norm_weight)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.aux2.ptr(),
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: tokens,
            intermediate: value_dim,
            gate_bf16: z_bf16,
            up_bf16: prefill.aux2.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.linear_rows(
            &layer.out_proj,
            prefill.aux3.ptr(),
            prefill.block_out.ptr(),
            tokens,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn run_full_attention_layer_prefill(
        &self,
        layer: &FullAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        prefill: &GpuPrefillBuffers,
        tokens: usize,
        start_position: usize,
        start_position_device_i32: DevicePtr,
        prequantized_normed: Option<Nvfp4ActivationQuant<'_>>,
    ) -> Result<()> {
        let q_dim = self.topology.full_attention_q_dim();
        let q_dim_with_gate = self.topology.full_attention_q_dim_with_gate();
        let cache = runtime
            .kv_cache
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("KV cache was not allocated".to_owned()))?;
        let layout = self
            .state
            .kv_cache
            .layers
            .iter()
            .find(|layout| layout.global_layer_index == layer.layer_index)
            .ok_or_else(|| {
                CoreError::Runtime(format!(
                    "missing KV-cache layout for layer {}",
                    layer.layer_index
                ))
            })?;

        let qkv_linears = [
            (&layer.q_proj, prefill.qkv.ptr()),
            (&layer.k_proj, prefill.aux.ptr()),
            (&layer.v_proj, prefill.aux2.ptr()),
        ];
        if let Some(quantized) = prequantized_normed {
            for &(binding, output) in &qkv_linears {
                self.linear_with_quantized_nvfp4_rows(binding, output, tokens, quantized, prefill)?;
            }
        } else {
            self.linears_same_input_rows(prefill.normed.ptr(), &qkv_linears, tokens, prefill)?;
        }
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                let kv_dim = self.topology.full_attention_kv_dim();
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_q_proj_raw.bf16",
                    prefill.qkv.ptr(),
                    tokens * q_dim_with_gate,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_k_raw.bf16",
                    prefill.aux.ptr(),
                    tokens * kv_dim,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_v_raw.bf16",
                    prefill.aux2.ptr(),
                    tokens * kv_dim,
                )?;
            }
        }

        self.backend.q_proj_deinterleave(&QProjDeinterleaveSpec {
            rows: tokens,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            input_bf16: prefill.qkv.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_q_extracted.bf16",
                    prefill.aux3.ptr(),
                    tokens * q_dim,
                )?;
            }
        }
        self.rmsnorm(
            tokens * self.topology.attention_num_heads,
            self.topology.attention_head_dim,
            prefill.aux3.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.q_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.aux3.ptr(),
        )?;
        self.rmsnorm(
            tokens * self.topology.attention_num_kv_heads,
            self.topology.attention_head_dim,
            prefill.aux.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.k_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.aux.ptr(),
        )?;
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                let kv_dim = self.topology.full_attention_kv_dim();
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_q_normed.bf16",
                    prefill.aux3.ptr(),
                    tokens * q_dim,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_k_normed.bf16",
                    prefill.aux.ptr(),
                    tokens * kv_dim,
                )?;
            }
        }
        self.backend.partial_rope(&PartialRopeSpec {
            tokens,
            q_heads: self.topology.attention_num_heads,
            kv_heads: self.topology.attention_num_kv_heads,
            head_dim: self.topology.attention_head_dim,
            rope_dims: self.topology.attention_rope_dims(),
            base_theta: self.topology.rope_theta,
            position_i32: 0,
            use_scalar_position: false,
            positions_i32: prefill.position_i32.ptr(),
            q_bf16: prefill.aux3.ptr(),
            k_bf16: prefill.aux.ptr(),
            scalar_position_device_i32: DevicePtr::NULL,
        })?;
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                let kv_dim = self.topology.full_attention_kv_dim();
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_q_rope.bf16",
                    prefill.aux3.ptr(),
                    tokens * q_dim,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_k_rope.bf16",
                    prefill.aux.ptr(),
                    tokens * kv_dim,
                )?;
            }
        }
        self.backend.attention_prefill(&AttentionPrefillSpec {
            layer_index: layer.layer_index,
            start_position,
            tokens,
            q_bf16: prefill.aux3.ptr(),
            k_bf16: prefill.aux.ptr(),
            v_bf16: prefill.aux2.ptr(),
            kv_cache_k: cache.ptr_at(layout.k_offset_bytes as usize)?,
            kv_cache_v: cache.ptr_at(layout.v_offset_bytes as usize)?,
            kv_cache_metadata: cache.ptr_at(layout.metadata_offset_bytes as usize)?,
            output_bf16: prefill.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            kv_cache_dtype: Self::attention_kv_cache_dtype_code(self.config.kv_cache_dtype)?,
            start_position_device_i32,
            partial_acc_f32: self.cuda_forward()?.attn_partial_acc.ptr(),
            partial_max_f32: self.cuda_forward()?.attn_partial_max.ptr(),
            partial_denom_f32: self.cuda_forward()?.attn_partial_denom.ptr(),
            prefill_n_splits: self
                .prefill_attention_n_splits(start_position + tokens, start_position_device_i32),
            split_timesteps_per_block: if start_position_device_i32 == DevicePtr::NULL {
                self.attention_split_timesteps_per_block_for(start_position + tokens)
            } else {
                let context_limit =
                    self.decode_attention_context_limit_for_active_context(start_position + tokens);
                self.attention_split_timesteps_per_block_for(context_limit)
            },
            tree_ancestor_bitmap_u64: DevicePtr::NULL,
            verify_chunk_rows: 0,
        })?;
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_attn_pre_gate.bf16",
                    prefill.aux3.ptr(),
                    tokens * q_dim,
                )?;
            }
        }
        self.backend.q_proj_sigmoid_gate(&QProjSigmoidGateSpec {
            rows: tokens,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            gate_bf16: prefill.qkv.ptr(),
            input_bf16: prefill.aux3.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_attn_gated.bf16",
                    prefill.aux3.ptr(),
                    tokens * q_dim,
                )?;
            }
        }
        self.linear_rows(
            &layer.o_proj,
            prefill.aux3.ptr(),
            prefill.block_out.ptr(),
            tokens,
            prefill,
        )?;
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_attn_out.bf16",
                    prefill.block_out.ptr(),
                    tokens * self.topology.hidden_size,
                )?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn run_mlp(&self, layer: &LayerWeights, forward: &GpuForwardBuffers) -> Result<()> {
        let common = layer_common(layer);
        let layer_idx = match layer {
            LayerWeights::LinearAttention(layer) => layer.layer_index,
            LayerWeights::FullAttention(layer) => layer.layer_index,
        };
        if let Some(fused) = self.mlp_fused_main_opt(layer_idx) {
            return self.run_mlp_fused_combined_gemm(common, fused, forward, None);
        }
        self.linears_same_input(
            forward.normed.ptr(),
            &[
                (&common.mlp_gate_proj, forward.aux.ptr()),
                (&common.mlp_up_proj, forward.aux2.ptr()),
            ],
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: 1,
            intermediate: self.topology.intermediate_size,
            gate_bf16: forward.aux.ptr(),
            up_bf16: forward.aux2.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.linear(
            &common.mlp_down_proj,
            forward.aux3.ptr(),
            forward.hidden.ptr(),
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mlp_with_quantized_input(
        &self,
        layer: &LayerWeights,
        forward: &GpuForwardBuffers,
        quantized: Nvfp4ActivationQuant<'_>,
    ) -> Result<()> {
        let common = layer_common(layer);
        let layer_idx = match layer {
            LayerWeights::LinearAttention(layer) => layer.layer_index,
            LayerWeights::FullAttention(layer) => layer.layer_index,
        };
        if let Some(fused) = self.mlp_fused_main_opt(layer_idx) {
            return self.run_mlp_fused_combined_gemm(common, fused, forward, Some(quantized));
        }
        self.linear_with_quantized_nvfp4(&common.mlp_gate_proj, forward.aux.ptr(), quantized)?;
        self.linear_with_quantized_nvfp4(&common.mlp_up_proj, forward.aux2.ptr(), quantized)?;
        self.backend.swiglu(&SwiGluSpec {
            rows: 1,
            intermediate: self.topology.intermediate_size,
            gate_bf16: forward.aux.ptr(),
            up_bf16: forward.aux2.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.linear(
            &common.mlp_down_proj,
            forward.aux3.ptr(),
            forward.hidden.ptr(),
        )
    }

    #[cfg(feature = "cuda")]
    fn mlp_fused_main_opt(&self, layer_idx: usize) -> Option<&MlpFusedLayer> {
        self.mlp_fused.as_ref()?.layers.get(layer_idx)
    }

    #[cfg(feature = "cuda")]
    fn linear_attn_in_proj_fused_layer_opt(
        &self,
        layer_idx: usize,
    ) -> Option<&LinearAttnInProjFused> {
        self.linear_attn_in_proj_fused
            .as_ref()?
            .layers
            .get(layer_idx)
            .and_then(|entry| entry.as_ref())
    }

    #[cfg(feature = "cuda")]
    fn linear_attn_in_proj_quant<'a>(
        &self,
        layer: &'a LinearAttentionLayerWeights,
    ) -> Result<Nvfp4ActivationQuant<'a>> {
        let LinearWeightBinding::Nvfp4 {
            weight,
            input_scale,
            ..
        } = &layer.in_proj_qkv
        else {
            return Err(CoreError::Runtime(
                "fused DeltaNet in_proj requires NVFP4 in_proj_qkv".to_owned(),
            ));
        };
        Ok(Nvfp4ActivationQuant {
            in_features: Self::nvfp4_in_features(weight)?,
            input_scale,
        })
    }

    #[cfg(feature = "cuda")]
    fn run_linear_attn_in_proj_fused_gemm(
        &self,
        layer: &LinearAttentionLayerWeights,
        fused: &LinearAttnInProjFused,
        quantized: Nvfp4ActivationQuant<'_>,
        output: DevicePtr,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let forward = self.cuda_forward()?;
        let LinearWeightBinding::Nvfp4 {
            tensor_scale: qkv_tensor_scale,
            ..
        } = &layer.in_proj_qkv
        else {
            return Err(CoreError::Runtime(
                "fused DeltaNet in_proj requires NVFP4 in_proj_qkv".to_owned(),
            ));
        };
        let workspace = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr())
            .unwrap_or(DevicePtr::NULL);
        let workspace_bytes = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.bytes())
            .unwrap_or(0);
        let alpha = self.tensor_scalar_f32(weights, qkv_tensor_scale)?
            * self.tensor_scalar_f32(weights, quantized.input_scale)?;
        let gemm_spec = Nvfp4GemmSpec {
            m: fused.combined_out_features,
            n: 1,
            k: quantized.in_features,
            a_fp4: fused.combined_weight.ptr(),
            a_scale: fused.combined_block_scale.ptr(),
            a_scale_2: self.tensor_ptr(weights, qkv_tensor_scale)?,
            b_fp4: forward.activation_fp4.ptr(),
            b_scale: forward.activation_scale.ptr(),
            b_scale_2: forward.activation_scale_2.ptr(),
            c_bf16: output,
            workspace,
            workspace_bytes,
            alpha,
            scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
        };
        self.backend.nvfp4_gemm(&gemm_spec).map_err(|err| {
            CoreError::Runtime(format!(
                "fused DeltaNet in_proj GEMM failed (m={}, n=1, k={}): {err}",
                fused.combined_out_features, quantized.in_features
            ))
        })
    }

    #[cfg(feature = "cuda")]
    fn run_linear_attn_in_proj_fused_gemm_rows(
        &self,
        layer: &LinearAttentionLayerWeights,
        fused: &LinearAttnInProjFused,
        quantized: Nvfp4ActivationQuant<'_>,
        output: DevicePtr,
        rows: usize,
        prefill: &GpuPrefillBuffers,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let LinearWeightBinding::Nvfp4 {
            tensor_scale: qkv_tensor_scale,
            ..
        } = &layer.in_proj_qkv
        else {
            return Err(CoreError::Runtime(
                "fused DeltaNet in_proj requires NVFP4 in_proj_qkv".to_owned(),
            ));
        };
        let workspace = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr())
            .unwrap_or(DevicePtr::NULL);
        let workspace_bytes = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.bytes())
            .unwrap_or(0);
        let alpha = self.tensor_scalar_f32(weights, qkv_tensor_scale)?
            * self.tensor_scalar_f32(weights, quantized.input_scale)?;
        let gemm_spec = Nvfp4GemmSpec {
            m: fused.combined_out_features,
            n: rows,
            k: quantized.in_features,
            a_fp4: fused.combined_weight.ptr(),
            a_scale: fused.combined_block_scale.ptr(),
            a_scale_2: self.tensor_ptr(weights, qkv_tensor_scale)?,
            b_fp4: prefill.activation_fp4.ptr(),
            b_scale: prefill.activation_scale.ptr(),
            b_scale_2: prefill.activation_scale_2.ptr(),
            c_bf16: output,
            workspace,
            workspace_bytes,
            alpha,
            scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
        };
        self.backend.nvfp4_gemm(&gemm_spec).map_err(|err| {
            CoreError::Runtime(format!(
                "fused DeltaNet in_proj prefill GEMM failed (m={}, n={}, k={}): {err}",
                fused.combined_out_features, rows, quantized.in_features
            ))
        })
    }

    /// Single combined-GEMM MLP: writes [gate || up] into `forward.aux` in
    /// one cuBLASLt FP4 GEMM, then runs SwiGLU on the two halves and the
    /// down_proj GEMM. Saves one FP4 GEMM launch per layer (× 64 layers per
    /// decode token).
    #[cfg(feature = "cuda")]
    fn run_mlp_fused_combined_gemm(
        &self,
        common: &CommonLayerWeights,
        fused: &MlpFusedLayer,
        forward: &GpuForwardBuffers,
        pre_quantized: Option<Nvfp4ActivationQuant<'_>>,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let intermediate = self.topology.intermediate_size;
        let LinearWeightBinding::Nvfp4 {
            weight: gate_weight,
            tensor_scale: gate_tensor_scale,
            input_scale: gate_input_scale,
            ..
        } = &common.mlp_gate_proj
        else {
            return Err(CoreError::Runtime(
                "fused MLP path requires NVFP4 gate_proj".to_owned(),
            ));
        };
        let in_features = Self::nvfp4_in_features(gate_weight)?;

        let quantized = match pre_quantized {
            Some(q) => q,
            None => {
                let q = Nvfp4ActivationQuant {
                    in_features,
                    input_scale: gate_input_scale,
                };
                self.quantize_nvfp4_activation(forward.normed.ptr(), q)?;
                q
            }
        };

        let workspace = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr())
            .unwrap_or(DevicePtr::NULL);
        let workspace_bytes = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.bytes())
            .unwrap_or(0);
        let alpha = self.tensor_scalar_f32(weights, gate_tensor_scale)?
            * self.tensor_scalar_f32(weights, quantized.input_scale)?;
        let gemm_spec = Nvfp4GemmSpec {
            m: fused.out_features,
            n: 1,
            k: quantized.in_features,
            a_fp4: fused.combined_weight.ptr(),
            a_scale: fused.combined_block_scale.ptr(),
            a_scale_2: self.tensor_ptr(weights, gate_tensor_scale)?,
            b_fp4: forward.activation_fp4.ptr(),
            b_scale: forward.activation_scale.ptr(),
            b_scale_2: forward.activation_scale_2.ptr(),
            c_bf16: forward.aux.ptr(),
            workspace,
            workspace_bytes,
            alpha,
            scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
        };
        self.backend.nvfp4_gemm(&gemm_spec).map_err(|err| {
            CoreError::Runtime(format!(
                "fused MLP NVFP4 GEMM failed (m={}, n=1, k={}): {err}",
                fused.out_features, quantized.in_features
            ))
        })?;

        let up_offset_bytes = intermediate * 2; // BF16 size
        let up_ptr = forward
            .aux
            .ptr()
            .offset_bytes(up_offset_bytes)
            .ok_or_else(|| {
                CoreError::Runtime("fused MLP up_proj output offset overflow".to_owned())
            })?;

        // Fused SwiGLU + NVFP4 activation quantization: writes the down_proj
        // input directly into `forward.activation_fp4` / `activation_scale`,
        // skipping the BF16 round-trip through `forward.aux3` and the separate
        // quantize launch.
        let LinearWeightBinding::Nvfp4 {
            input_scale: down_input_scale,
            ..
        } = &common.mlp_down_proj
        else {
            return Err(CoreError::Runtime(
                "fused MLP path requires NVFP4 down_proj".to_owned(),
            ));
        };
        let down_input_scale_f32 = self.tensor_scalar_f32(weights, down_input_scale)?;
        self.backend
            .swiglu_nvfp4_quantize(&SwiGluNvfp4QuantizeSpec {
                intermediate,
                gate_bf16: forward.aux.ptr(),
                up_bf16: up_ptr,
                output_fp4: forward.activation_fp4.ptr(),
                output_scale_e4m3: forward.activation_scale.ptr(),
                output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                input_tensor_scale_f32: down_input_scale_f32,
            })?;
        let down_quantized = Nvfp4ActivationQuant {
            in_features: intermediate,
            input_scale: down_input_scale,
        };
        self.linear_with_quantized_nvfp4(
            &common.mlp_down_proj,
            forward.hidden.ptr(),
            down_quantized,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mlp_fused_combined_gemm_rows(
        &self,
        common: &CommonLayerWeights,
        fused: &MlpFusedLayer,
        prefill: &GpuPrefillBuffers,
        rows: usize,
        pre_quantized: Option<Nvfp4ActivationQuant<'_>>,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let intermediate = self.topology.intermediate_size;
        let LinearWeightBinding::Nvfp4 {
            weight: gate_weight,
            tensor_scale: gate_tensor_scale,
            input_scale: gate_input_scale,
            ..
        } = &common.mlp_gate_proj
        else {
            return Err(CoreError::Runtime(
                "fused MLP prefill path requires NVFP4 gate_proj".to_owned(),
            ));
        };
        let in_features = Self::nvfp4_in_features(gate_weight)?;
        let quantized = match pre_quantized {
            Some(q) => q,
            None => {
                let q = Nvfp4ActivationQuant {
                    in_features,
                    input_scale: gate_input_scale,
                };
                self.quantize_nvfp4_activation_rows(prefill.normed.ptr(), rows, q, prefill)?;
                q
            }
        };

        let workspace = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr())
            .unwrap_or(DevicePtr::NULL);
        let workspace_bytes = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.bytes())
            .unwrap_or(0);
        let alpha = self.tensor_scalar_f32(weights, gate_tensor_scale)?
            * self.tensor_scalar_f32(weights, quantized.input_scale)?;
        self.backend.nvfp4_gemm(&Nvfp4GemmSpec {
            m: fused.out_features,
            n: rows,
            k: quantized.in_features,
            a_fp4: fused.combined_weight.ptr(),
            a_scale: fused.combined_block_scale.ptr(),
            a_scale_2: self.tensor_ptr(weights, gate_tensor_scale)?,
            b_fp4: prefill.activation_fp4.ptr(),
            b_scale: prefill.activation_scale.ptr(),
            b_scale_2: prefill.activation_scale_2.ptr(),
            c_bf16: prefill.block_out.ptr(),
            workspace,
            workspace_bytes,
            alpha,
            scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
        })?;

        self.backend.copy_strided_rows(&CopyStridedRowsSpec {
            rows,
            values: intermediate,
            input_stride: 2 * intermediate,
            output_stride: intermediate,
            input_bf16: prefill.block_out.ptr(),
            output_bf16: prefill.aux.ptr(),
        })?;
        self.backend.copy_strided_rows(&CopyStridedRowsSpec {
            rows,
            values: intermediate,
            input_stride: 2 * intermediate,
            output_stride: intermediate,
            input_bf16: Self::ptr_offset(prefill.block_out.ptr(), intermediate * 2)?,
            output_bf16: prefill.aux2.ptr(),
        })?;
        self.backend.swiglu(&SwiGluSpec {
            rows,
            intermediate,
            gate_bf16: prefill.aux.ptr(),
            up_bf16: prefill.aux2.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.linear_rows(
            &common.mlp_down_proj,
            prefill.aux3.ptr(),
            prefill.hidden.ptr(),
            rows,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mlp_prefill(
        &self,
        layer: &LayerWeights,
        prefill: &GpuPrefillBuffers,
        tokens: usize,
    ) -> Result<()> {
        let common = layer_common(layer);
        let layer_idx = match layer {
            LayerWeights::LinearAttention(layer) => layer.layer_index,
            LayerWeights::FullAttention(layer) => layer.layer_index,
        };
        if cuda_prefill_fused_mlp_enabled(self.config.max_context)
            && prefill.block_out.bytes() >= tokens * 2 * self.topology.intermediate_size * 2
        {
            if let Some(fused) = self.mlp_fused_main_opt(layer_idx) {
                return self.run_mlp_fused_combined_gemm_rows(common, fused, prefill, tokens, None);
            }
        }
        self.linears_same_input_rows(
            prefill.normed.ptr(),
            &[
                (&common.mlp_gate_proj, prefill.aux.ptr()),
                (&common.mlp_up_proj, prefill.aux2.ptr()),
            ],
            tokens,
            prefill,
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: tokens,
            intermediate: self.topology.intermediate_size,
            gate_bf16: prefill.aux.ptr(),
            up_bf16: prefill.aux2.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.linear_rows(
            &common.mlp_down_proj,
            prefill.aux3.ptr(),
            prefill.hidden.ptr(),
            tokens,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mlp_with_quantized_input_prefill(
        &self,
        layer: &LayerWeights,
        prefill: &GpuPrefillBuffers,
        tokens: usize,
        quantized: Nvfp4ActivationQuant<'_>,
    ) -> Result<()> {
        let common = layer_common(layer);
        let layer_idx = match layer {
            LayerWeights::LinearAttention(layer) => layer.layer_index,
            LayerWeights::FullAttention(layer) => layer.layer_index,
        };
        if cuda_prefill_fused_mlp_enabled(self.config.max_context)
            && prefill.block_out.bytes() >= tokens * 2 * self.topology.intermediate_size * 2
        {
            if let Some(fused) = self.mlp_fused_main_opt(layer_idx) {
                return self.run_mlp_fused_combined_gemm_rows(
                    common,
                    fused,
                    prefill,
                    tokens,
                    Some(quantized),
                );
            }
        }
        self.linear_with_quantized_nvfp4_rows(
            &common.mlp_gate_proj,
            prefill.aux.ptr(),
            tokens,
            quantized,
            prefill,
        )?;
        self.linear_with_quantized_nvfp4_rows(
            &common.mlp_up_proj,
            prefill.aux2.ptr(),
            tokens,
            quantized,
            prefill,
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: tokens,
            intermediate: self.topology.intermediate_size,
            gate_bf16: prefill.aux.ptr(),
            up_bf16: prefill.aux2.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.linear_rows(
            &common.mlp_down_proj,
            prefill.aux3.ptr(),
            prefill.hidden.ptr(),
            tokens,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    fn linears_same_input(
        &self,
        input: DevicePtr,
        linears: &[(&LinearWeightBinding, DevicePtr)],
    ) -> Result<()> {
        let Some(quantized) = Self::common_nvfp4_quant(linears)? else {
            for &(binding, output) in linears {
                self.linear(binding, input, output)?;
            }
            return Ok(());
        };

        self.quantize_nvfp4_activation(input, quantized)?;
        for &(binding, output) in linears {
            self.linear_with_quantized_nvfp4(binding, output, quantized)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn linears_same_input_rows(
        &self,
        input: DevicePtr,
        linears: &[(&LinearWeightBinding, DevicePtr)],
        rows: usize,
        prefill: &GpuPrefillBuffers,
    ) -> Result<()> {
        let Some(quantized) = Self::common_nvfp4_quant(linears)? else {
            for &(binding, output) in linears {
                self.linear_rows(binding, input, output, rows, prefill)?;
            }
            return Ok(());
        };

        self.quantize_nvfp4_activation_rows(input, rows, quantized, prefill)?;
        for &(binding, output) in linears {
            self.linear_with_quantized_nvfp4_rows(binding, output, rows, quantized, prefill)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn common_nvfp4_quant<'a>(
        linears: &'a [(&'a LinearWeightBinding, DevicePtr)],
    ) -> Result<Option<Nvfp4ActivationQuant<'a>>> {
        Self::common_nvfp4_quant_bindings(linears.iter().map(|(binding, _)| *binding))
    }

    #[cfg(feature = "cuda")]
    fn common_nvfp4_quant_bindings<'a>(
        linears: impl IntoIterator<Item = &'a LinearWeightBinding>,
    ) -> Result<Option<Nvfp4ActivationQuant<'a>>> {
        let mut common: Option<Nvfp4ActivationQuant<'a>> = None;
        for binding in linears {
            let LinearWeightBinding::Nvfp4 {
                weight,
                input_scale,
                ..
            } = binding
            else {
                return Ok(None);
            };
            let in_features = Self::nvfp4_in_features(weight)?;
            match common {
                Some(previous) if previous.in_features != in_features => return Ok(None),
                Some(_) => {}
                None => {
                    common = Some(Nvfp4ActivationQuant {
                        in_features,
                        input_scale,
                    });
                }
            }
        }
        Ok(common)
    }

    #[cfg(feature = "cuda")]
    fn layer_input_nvfp4_quant(layer: &LayerWeights) -> Result<Option<Nvfp4ActivationQuant<'_>>> {
        match layer {
            LayerWeights::LinearAttention(layer) => Self::common_nvfp4_quant_bindings([
                &layer.in_proj_qkv,
                &layer.in_proj_b,
                &layer.in_proj_a,
                &layer.in_proj_z,
            ]),
            LayerWeights::FullAttention(layer) => {
                Self::common_nvfp4_quant_bindings([&layer.q_proj, &layer.k_proj, &layer.v_proj])
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn linear(
        &self,
        binding: &LinearWeightBinding,
        input: DevicePtr,
        output: DevicePtr,
    ) -> Result<()> {
        match binding {
            LinearWeightBinding::Nvfp4 {
                weight,
                block_scale,
                tensor_scale,
                input_scale,
            } => {
                let in_features = Self::nvfp4_in_features(weight)?;
                let quantized = Nvfp4ActivationQuant {
                    in_features,
                    input_scale,
                };
                self.quantize_nvfp4_activation(input, quantized)?;
                self.nvfp4_gemm_with_quantized_activation(
                    weight,
                    block_scale,
                    tensor_scale,
                    input_scale,
                    output,
                    in_features,
                )
            }
            LinearWeightBinding::Bf16 { weight } => self.bf16_matvec(weight, input, output),
        }
    }

    #[cfg(feature = "cuda")]
    fn linear_rows(
        &self,
        binding: &LinearWeightBinding,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        prefill: &GpuPrefillBuffers,
    ) -> Result<()> {
        match binding {
            LinearWeightBinding::Nvfp4 {
                weight,
                block_scale,
                tensor_scale,
                input_scale,
            } => {
                let in_features = Self::nvfp4_in_features(weight)?;
                let quantized = Nvfp4ActivationQuant {
                    in_features,
                    input_scale,
                };
                self.quantize_nvfp4_activation_rows(input, rows, quantized, prefill)?;
                self.nvfp4_gemm_with_quantized_activation_rows(
                    weight,
                    block_scale,
                    tensor_scale,
                    input_scale,
                    output,
                    rows,
                    in_features,
                    prefill,
                )
            }
            LinearWeightBinding::Bf16 { weight } => {
                self.bf16_gemm_rows(weight, input, output, rows)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn linear_with_quantized_nvfp4(
        &self,
        binding: &LinearWeightBinding,
        output: DevicePtr,
        quantized: Nvfp4ActivationQuant<'_>,
    ) -> Result<()> {
        let LinearWeightBinding::Nvfp4 {
            weight,
            block_scale,
            tensor_scale,
            input_scale,
        } = binding
        else {
            return Err(CoreError::Runtime(
                "quantized NVFP4 path received a BF16 linear".to_owned(),
            ));
        };
        let in_features = Self::nvfp4_in_features(weight)?;
        if in_features != quantized.in_features {
            return Err(CoreError::Runtime(format!(
                "quantized activation has {} values but {} expects {in_features}",
                quantized.in_features, weight.name
            )));
        }
        self.validate_nvfp4_input_scale(weight, quantized.input_scale, input_scale)?;
        self.nvfp4_gemm_with_quantized_activation(
            weight,
            block_scale,
            tensor_scale,
            quantized.input_scale,
            output,
            in_features,
        )
    }

    #[cfg(feature = "cuda")]
    fn linear_with_quantized_nvfp4_rows(
        &self,
        binding: &LinearWeightBinding,
        output: DevicePtr,
        rows: usize,
        quantized: Nvfp4ActivationQuant<'_>,
        prefill: &GpuPrefillBuffers,
    ) -> Result<()> {
        let LinearWeightBinding::Nvfp4 {
            weight,
            block_scale,
            tensor_scale,
            input_scale,
        } = binding
        else {
            return Err(CoreError::Runtime(
                "quantized NVFP4 path received a BF16 linear".to_owned(),
            ));
        };
        let in_features = Self::nvfp4_in_features(weight)?;
        if in_features != quantized.in_features {
            return Err(CoreError::Runtime(format!(
                "quantized activation has {} values but {} expects {in_features}",
                quantized.in_features, weight.name
            )));
        }
        self.validate_nvfp4_input_scale(weight, quantized.input_scale, input_scale)?;
        self.nvfp4_gemm_with_quantized_activation_rows(
            weight,
            block_scale,
            tensor_scale,
            quantized.input_scale,
            output,
            rows,
            in_features,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    fn quantize_nvfp4_activation(
        &self,
        input: DevicePtr,
        quantized: Nvfp4ActivationQuant<'_>,
    ) -> Result<()> {
        let forward = self.cuda_forward()?;
        let input_tensor_scale_f32 =
            self.tensor_scalar_f32(self.cuda_weights()?, quantized.input_scale)?;
        self.backend.nvfp4_quantize_bf16(&Nvfp4QuantizeSpec {
            values: quantized.in_features,
            input_bf16: input,
            output_fp4: forward.activation_fp4.ptr(),
            output_scale_e4m3: forward.activation_scale.ptr(),
            output_tensor_scale_f32: forward.activation_scale_2.ptr(),
            input_tensor_scale_f32,
        })
    }

    #[cfg(feature = "cuda")]
    fn quantize_nvfp4_activation_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        quantized: Nvfp4ActivationQuant<'_>,
        prefill: &GpuPrefillBuffers,
    ) -> Result<()> {
        let input_tensor_scale_f32 =
            self.tensor_scalar_f32(self.cuda_weights()?, quantized.input_scale)?;
        self.backend.nvfp4_quantize_rows(&Nvfp4QuantizeRowsSpec {
            rows,
            values: quantized.in_features,
            input_bf16: input,
            output_fp4: prefill.activation_fp4.ptr(),
            output_scale_e4m3: prefill.activation_scale.ptr(),
            output_tensor_scale_f32: prefill.activation_scale_2.ptr(),
            input_tensor_scale_f32,
        })
    }

    #[cfg(feature = "cuda")]
    fn nvfp4_gemm_with_quantized_activation(
        &self,
        weight: &TensorInfo,
        block_scale: &TensorInfo,
        tensor_scale: &TensorInfo,
        input_scale: &TensorInfo,
        output: DevicePtr,
        in_features: usize,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let forward = self.cuda_forward()?;
        let runtime = self.cuda_runtime()?;
        let out_features = *weight
            .shape
            .first()
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} has empty shape", weight.name)))?;
        let workspace = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr())
            .unwrap_or(DevicePtr::NULL);
        let workspace_bytes = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.bytes())
            .unwrap_or(0);
        let gemm_spec = Nvfp4GemmSpec {
            m: out_features,
            n: 1,
            k: in_features,
            a_fp4: self.tensor_ptr(weights, weight)?,
            a_scale: self.tensor_ptr(weights, block_scale)?,
            a_scale_2: self.tensor_ptr(weights, tensor_scale)?,
            b_fp4: forward.activation_fp4.ptr(),
            b_scale: forward.activation_scale.ptr(),
            b_scale_2: forward.activation_scale_2.ptr(),
            c_bf16: output,
            workspace,
            workspace_bytes,
            alpha: self.tensor_scalar_f32(weights, tensor_scale)?
                * self.tensor_scalar_f32(weights, input_scale)?,
            scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
        };
        self.backend.nvfp4_gemm(&gemm_spec).map_err(|err| {
            CoreError::Runtime(format!(
                "NVFP4 cuBLASLt GEMM failed for {} (m={}, n={}, k={}): {err}",
                weight.name, gemm_spec.m, gemm_spec.n, gemm_spec.k
            ))
        })
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn nvfp4_gemm_with_quantized_activation_rows(
        &self,
        weight: &TensorInfo,
        block_scale: &TensorInfo,
        tensor_scale: &TensorInfo,
        input_scale: &TensorInfo,
        output: DevicePtr,
        rows: usize,
        in_features: usize,
        prefill: &GpuPrefillBuffers,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let out_features = *weight
            .shape
            .first()
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} has empty shape", weight.name)))?;
        let workspace = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr())
            .unwrap_or(DevicePtr::NULL);
        let workspace_bytes = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.bytes())
            .unwrap_or(0);
        let gemm_spec = Nvfp4GemmSpec {
            m: out_features,
            n: rows,
            k: in_features,
            a_fp4: self.tensor_ptr(weights, weight)?,
            a_scale: self.tensor_ptr(weights, block_scale)?,
            a_scale_2: self.tensor_ptr(weights, tensor_scale)?,
            b_fp4: prefill.activation_fp4.ptr(),
            b_scale: prefill.activation_scale.ptr(),
            b_scale_2: prefill.activation_scale_2.ptr(),
            c_bf16: output,
            workspace,
            workspace_bytes,
            alpha: self.tensor_scalar_f32(weights, tensor_scale)?
                * self.tensor_scalar_f32(weights, input_scale)?,
            scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
        };
        self.backend.nvfp4_gemm(&gemm_spec).map_err(|err| {
            CoreError::Runtime(format!(
                "NVFP4 cuBLASLt GEMM failed for {} (m={}, n={}, k={}): {err}",
                weight.name, gemm_spec.m, gemm_spec.n, gemm_spec.k
            ))
        })
    }

    #[cfg(feature = "cuda")]
    fn nvfp4_in_features(weight: &TensorInfo) -> Result<usize> {
        let packed_in = *weight
            .shape
            .get(1)
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} is not a matrix", weight.name)))?;
        Ok(packed_in * 2)
    }

    #[cfg(feature = "cuda")]
    fn validate_nvfp4_input_scale(
        &self,
        weight: &TensorInfo,
        quantized_input_scale: &TensorInfo,
        binding_input_scale: &TensorInfo,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let quantized_value = self.tensor_scalar_f32(weights, quantized_input_scale)?;
        let binding_value = self.tensor_scalar_f32(weights, binding_input_scale)?;
        if quantized_value.to_bits() != binding_value.to_bits() {
            return Err(CoreError::Runtime(format!(
                "shared NVFP4 activation for {} used input scale {} ({quantized_value}) but binding expects {} ({binding_value})",
                weight.name, quantized_input_scale.name, binding_input_scale.name
            )));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn run_interpreter_rmsnorm_nvfp4_quantize(
        &self,
        params: DecodeInterpreterRmsNormNvfp4QuantParams,
        forward: &GpuForwardBuffers,
    ) -> Result<()> {
        let compiled = DecodeInterpreterProgram::compile_rmsnorm_nvfp4_quant(params);
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_norm_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter RMSNorm program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_norm_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_norm_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter RMSNorm program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_norm_counters.bytes()
            )));
        }

        forward
            .interpreter_norm_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_norm_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_norm_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_norm_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })
    }

    #[cfg(feature = "cuda")]
    fn run_interpreter_deltanet_decode(
        &self,
        spec: &DeltaNetDecodeSpec,
        forward: &GpuForwardBuffers,
    ) -> Result<()> {
        if spec.tokens_in_persistent_loop != 1
            || spec.q_token_stride != 0
            || spec.k_token_stride != 0
            || spec.v_token_stride != 0
        {
            self.backend.deltanet_decode(spec)?;
            return Ok(());
        }

        let spec_bytes = deltanet_decode_spec_abi_bytes(spec);
        if spec_bytes.len() > forward.interpreter_deltanet_spec.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter DeltaNet spec needs {} bytes but buffer has {}",
                spec_bytes.len(),
                forward.interpreter_deltanet_spec.bytes()
            )));
        }
        forward
            .interpreter_deltanet_spec
            .copy_from_host(&spec_bytes)?;

        let compiled =
            DecodeInterpreterProgram::compile_deltanet_recur(DecodeInterpreterDeltaNetParams {
                spec: forward.interpreter_deltanet_spec.ptr(),
            });
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_deltanet_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter DeltaNet program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_deltanet_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_deltanet_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter DeltaNet program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_deltanet_counters.bytes()
            )));
        }

        forward
            .interpreter_deltanet_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_deltanet_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_deltanet_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_deltanet_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })
    }

    #[cfg(feature = "cuda")]
    fn run_interpreter_partial_rope(
        &self,
        spec: &PartialRopeSpec,
        forward: &GpuForwardBuffers,
    ) -> Result<()> {
        if spec.tokens != 1
            || !spec.use_scalar_position
            || spec.positions_i32 != DevicePtr::NULL
            || spec.scalar_position_device_i32 != DevicePtr::NULL
        {
            self.backend.partial_rope(spec)?;
            return Ok(());
        }

        let compiled =
            DecodeInterpreterProgram::compile_rope_partial(DecodeInterpreterRopeParams {
                tokens: spec.tokens,
                q_heads: spec.q_heads,
                kv_heads: spec.kv_heads,
                head_dim: spec.head_dim,
                rope_dims: spec.rope_dims,
                base_theta: spec.base_theta,
                position_i32: spec.position_i32,
                use_scalar_position: spec.use_scalar_position,
                positions_i32: spec.positions_i32,
                q_bf16: spec.q_bf16,
                k_bf16: spec.k_bf16,
                scalar_position_device_i32: spec.scalar_position_device_i32,
            });
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_rope_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter RoPE program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_rope_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_rope_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter RoPE program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_rope_counters.bytes()
            )));
        }

        forward
            .interpreter_rope_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_rope_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_rope_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_rope_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })
    }

    #[cfg(feature = "cuda")]
    fn run_interpreter_attention_decode(
        &self,
        spec: &AttentionDecodeSpec,
        forward: &GpuForwardBuffers,
    ) -> Result<()> {
        let bf16_kv_cache_dtype = Self::attention_kv_cache_dtype_code(KvCacheDtype::Bf16)?;
        if spec.kv_cache_dtype != bf16_kv_cache_dtype
            || spec.decode_n_splits > 1
            || spec.position_device_i32 != DevicePtr::NULL
        {
            self.backend.attention_decode(spec)?;
            return Ok(());
        }

        let mut interpreter_spec = spec.clone();
        interpreter_spec.kv_cache_metadata = DevicePtr::NULL;
        interpreter_spec.partial_acc_f32 = DevicePtr::NULL;
        interpreter_spec.partial_max_f32 = DevicePtr::NULL;
        interpreter_spec.partial_denom_f32 = DevicePtr::NULL;
        interpreter_spec.decode_n_splits = interpreter_spec.decode_n_splits.min(1);
        interpreter_spec.split_timesteps_per_block = 0;

        let spec_bytes = attention_decode_spec_abi_bytes(&interpreter_spec);
        if spec_bytes.len() > forward.interpreter_attention_spec.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter Attention spec needs {} bytes but buffer has {}",
                spec_bytes.len(),
                forward.interpreter_attention_spec.bytes()
            )));
        }
        forward
            .interpreter_attention_spec
            .copy_from_host(&spec_bytes)?;

        let compiled = DecodeInterpreterProgram::compile_attention_decode_full(
            DecodeInterpreterAttentionParams {
                spec: forward.interpreter_attention_spec.ptr(),
            },
        );
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_attention_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter Attention program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_attention_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_attention_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter Attention program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_attention_counters.bytes()
            )));
        }

        forward
            .interpreter_attention_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_attention_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_attention_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_attention_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })
    }

    #[cfg(feature = "cuda")]
    fn run_interpreter_rope_attention_decode(
        &self,
        rope_spec: &PartialRopeSpec,
        attention_spec: &AttentionDecodeSpec,
        forward: &GpuForwardBuffers,
    ) -> Result<()> {
        let bf16_kv_cache_dtype = Self::attention_kv_cache_dtype_code(KvCacheDtype::Bf16)?;
        if rope_spec.tokens != 1
            || !rope_spec.use_scalar_position
            || rope_spec.positions_i32 != DevicePtr::NULL
            || rope_spec.scalar_position_device_i32 != DevicePtr::NULL
            || attention_spec.kv_cache_dtype != bf16_kv_cache_dtype
            || attention_spec.decode_n_splits > 1
            || attention_spec.position_device_i32 != DevicePtr::NULL
        {
            self.backend.partial_rope(rope_spec)?;
            self.backend.attention_decode(attention_spec)?;
            return Ok(());
        }

        let mut interpreter_attention_spec = attention_spec.clone();
        interpreter_attention_spec.kv_cache_metadata = DevicePtr::NULL;
        interpreter_attention_spec.partial_acc_f32 = DevicePtr::NULL;
        interpreter_attention_spec.partial_max_f32 = DevicePtr::NULL;
        interpreter_attention_spec.partial_denom_f32 = DevicePtr::NULL;
        interpreter_attention_spec.decode_n_splits =
            interpreter_attention_spec.decode_n_splits.min(1);
        interpreter_attention_spec.split_timesteps_per_block = 0;

        let spec_bytes = attention_decode_spec_abi_bytes(&interpreter_attention_spec);
        if spec_bytes.len() > forward.interpreter_attention_spec.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter RoPE+Attention spec needs {} bytes but buffer has {}",
                spec_bytes.len(),
                forward.interpreter_attention_spec.bytes()
            )));
        }
        forward
            .interpreter_attention_spec
            .copy_from_host(&spec_bytes)?;

        let compiled = DecodeInterpreterProgram::compile_rope_attention_decode(
            DecodeInterpreterRopeAttentionParams {
                rope: DecodeInterpreterRopeParams {
                    tokens: rope_spec.tokens,
                    q_heads: rope_spec.q_heads,
                    kv_heads: rope_spec.kv_heads,
                    head_dim: rope_spec.head_dim,
                    rope_dims: rope_spec.rope_dims,
                    base_theta: rope_spec.base_theta,
                    position_i32: rope_spec.position_i32,
                    use_scalar_position: rope_spec.use_scalar_position,
                    positions_i32: rope_spec.positions_i32,
                    q_bf16: rope_spec.q_bf16,
                    k_bf16: rope_spec.k_bf16,
                    scalar_position_device_i32: rope_spec.scalar_position_device_i32,
                },
                attention: DecodeInterpreterAttentionParams {
                    spec: forward.interpreter_attention_spec.ptr(),
                },
            },
        );
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_attention_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter RoPE+Attention program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_attention_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_attention_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter RoPE+Attention program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_attention_counters.bytes()
            )));
        }

        forward
            .interpreter_attention_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_attention_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_attention_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_attention_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })
    }

    #[cfg(feature = "cuda")]
    fn decode_interpreter_nvfp4_gemv_params(
        &self,
        binding: &LinearWeightBinding,
        quantized: Nvfp4ActivationQuant<'_>,
        input_fp4: DevicePtr,
        input_scale_e4m3: DevicePtr,
        output_bf16: DevicePtr,
    ) -> Result<Option<DecodeInterpreterNvfp4GemvParams>> {
        let LinearWeightBinding::Nvfp4 {
            weight,
            block_scale,
            tensor_scale,
            input_scale,
        } = binding
        else {
            return Ok(None);
        };

        let in_features = Self::nvfp4_in_features(weight)?;
        if in_features != quantized.in_features {
            return Ok(None);
        }
        let out_features = *weight
            .shape
            .first()
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} has empty shape", weight.name)))?;
        if !decode_interpreter_nvfp4_gemv_supports(out_features, in_features) {
            return Ok(None);
        }

        self.validate_nvfp4_input_scale(weight, quantized.input_scale, input_scale)?;
        let weights = self.cuda_weights()?;
        Ok(Some(DecodeInterpreterNvfp4GemvParams {
            m: out_features,
            k: in_features,
            alpha: self.tensor_scalar_f32(weights, tensor_scale)?
                * self.tensor_scalar_f32(weights, quantized.input_scale)?,
            a_fp4: self.tensor_ptr(weights, weight)?,
            a_scale_e4m3: self.tensor_ptr(weights, block_scale)?,
            b_fp4: input_fp4,
            b_scale_e4m3: input_scale_e4m3,
            c_bf16: output_bf16,
        }))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn run_interpreter_full_attention_input_layer_decode(
        &self,
        layer: &FullAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        position: usize,
        quantized: Nvfp4ActivationQuant<'_>,
        input_residual_bf16: DevicePtr,
        input_norm_weight_bf16: DevicePtr,
    ) -> Result<bool> {
        if !decode_interpreter_full_attention_input_layer_enabled(
            self.decode_interpreter_decode_enabled(),
        ) {
            return Ok(false);
        }

        let bf16_kv_cache_dtype = Self::attention_kv_cache_dtype_code(KvCacheDtype::Bf16)?;
        let configured_kv_cache_dtype =
            Self::attention_kv_cache_dtype_code(self.config.kv_cache_dtype)?;
        if configured_kv_cache_dtype != bf16_kv_cache_dtype {
            return Ok(false);
        }

        let cache = runtime
            .kv_cache
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("KV cache was not allocated".to_owned()))?;
        let layout = self
            .state
            .kv_cache
            .layers
            .iter()
            .find(|layout| layout.global_layer_index == layer.layer_index)
            .ok_or_else(|| {
                CoreError::Runtime(format!(
                    "missing KV-cache layout for layer {}",
                    layer.layer_index
                ))
            })?;

        let q_values = self.topology.attention_num_heads * self.topology.attention_head_dim;
        let kv_values = self.topology.attention_num_kv_heads * self.topology.attention_head_dim;
        let hidden = self.topology.hidden_size;
        let input_fp4 = forward.activation_fp4.ptr();
        let input_scale_e4m3 = forward.activation_scale.ptr();

        let Some(q_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.q_proj,
            quantized,
            input_fp4,
            input_scale_e4m3,
            forward.qkv.ptr(),
        )?
        else {
            return Ok(false);
        };
        let Some(k_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.k_proj,
            quantized,
            input_fp4,
            input_scale_e4m3,
            forward.aux.ptr(),
        )?
        else {
            return Ok(false);
        };
        let Some(v_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.v_proj,
            quantized,
            input_fp4,
            input_scale_e4m3,
            forward.aux2.ptr(),
        )?
        else {
            return Ok(false);
        };
        if q_proj.m != q_values * 2
            || q_proj.k != hidden
            || k_proj.m != kv_values
            || k_proj.k != hidden
            || v_proj.m != kv_values
            || v_proj.k != hidden
        {
            return Ok(false);
        }

        let LinearWeightBinding::Nvfp4 {
            input_scale: o_input_scale,
            ..
        } = &layer.o_proj
        else {
            return Ok(false);
        };
        let o_quantized = Nvfp4ActivationQuant {
            in_features: q_values,
            input_scale: o_input_scale,
        };
        let Some(o_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.o_proj,
            o_quantized,
            forward.activation_fp4.ptr(),
            forward.activation_scale.ptr(),
            forward.block_out.ptr(),
        )?
        else {
            return Ok(false);
        };
        if o_proj.m != hidden || o_proj.k != q_values {
            return Ok(false);
        }

        let attention_context_limit = self.decode_attention_context_limit_for_position(position);
        let decode_n_splits =
            self.decode_attention_n_splits_for_context_limit(attention_context_limit);

        let mut attention_spec = AttentionDecodeSpec {
            layer_index: layer.layer_index,
            position,
            q_bf16: forward.aux3.ptr(),
            k_bf16: forward.aux.ptr(),
            v_bf16: forward.aux2.ptr(),
            kv_cache_k: cache.ptr_at(layout.k_offset_bytes as usize)?,
            kv_cache_v: cache.ptr_at(layout.v_offset_bytes as usize)?,
            kv_cache_metadata: DevicePtr::NULL,
            output_bf16: forward.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            kv_cache_dtype: bf16_kv_cache_dtype,
            position_device_i32: DevicePtr::NULL,
            partial_acc_f32: DevicePtr::NULL,
            partial_max_f32: DevicePtr::NULL,
            partial_denom_f32: DevicePtr::NULL,
            decode_n_splits: decode_n_splits.min(1),
            split_timesteps_per_block: 0,
        };
        attention_spec.decode_n_splits = attention_spec.decode_n_splits.min(1);

        let spec_bytes = attention_decode_spec_abi_bytes(&attention_spec);
        if spec_bytes.len() > forward.interpreter_attention_spec.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter full-attn input-layer spec needs {} bytes but buffer has {}",
                spec_bytes.len(),
                forward.interpreter_attention_spec.bytes()
            )));
        }
        forward
            .interpreter_attention_spec
            .copy_from_host(&spec_bytes)?;

        let weights = self.cuda_weights()?;
        let o_input_tensor_scale_f32 = self.tensor_scalar_f32(weights, o_input_scale)?;
        let compiled = DecodeInterpreterProgram::compile_full_attention_input_layer_decode(
            DecodeInterpreterFullAttentionInputLayerParams {
                input_norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                    hidden,
                    eps: 1.0e-6,
                    input_tensor_scale_f32: self
                        .tensor_scalar_f32(weights, quantized.input_scale)?,
                    input_bf16: forward.hidden.ptr(),
                    weight_bf16: input_norm_weight_bf16,
                    residual_bf16: input_residual_bf16,
                    residual_out_bf16: forward.residual.ptr(),
                    output_bf16: DevicePtr::NULL,
                    output_fp4: forward.activation_fp4.ptr(),
                    output_scale_e4m3: forward.activation_scale.ptr(),
                    output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                },
                layer: DecodeInterpreterFullAttentionLayerParams {
                    q_proj,
                    k_proj,
                    v_proj,
                    q_proj_deinterleave: DecodeInterpreterQProjDeinterleaveParams {
                        rows: 1,
                        heads: self.topology.attention_num_heads,
                        head_dim: self.topology.attention_head_dim,
                        input_bf16: forward.qkv.ptr(),
                        output_bf16: forward.aux3.ptr(),
                    },
                    q_norm: DecodeInterpreterRmsNormBf16Params {
                        rows: self.topology.attention_num_heads,
                        hidden: self.topology.attention_head_dim,
                        eps: 1.0e-6,
                        direct_weight: false,
                        input_bf16: forward.aux3.ptr(),
                        weight_bf16: self.tensor_ptr(weights, &layer.q_norm)?,
                        residual_bf16: DevicePtr::NULL,
                        residual_out_bf16: DevicePtr::NULL,
                        output_bf16: forward.aux3.ptr(),
                    },
                    k_norm: DecodeInterpreterRmsNormBf16Params {
                        rows: self.topology.attention_num_kv_heads,
                        hidden: self.topology.attention_head_dim,
                        eps: 1.0e-6,
                        direct_weight: false,
                        input_bf16: forward.aux.ptr(),
                        weight_bf16: self.tensor_ptr(weights, &layer.k_norm)?,
                        residual_bf16: DevicePtr::NULL,
                        residual_out_bf16: DevicePtr::NULL,
                        output_bf16: forward.aux.ptr(),
                    },
                    rope: DecodeInterpreterRopeParams {
                        tokens: 1,
                        q_heads: self.topology.attention_num_heads,
                        kv_heads: self.topology.attention_num_kv_heads,
                        head_dim: self.topology.attention_head_dim,
                        rope_dims: self.topology.attention_rope_dims(),
                        base_theta: self.topology.rope_theta,
                        position_i32: position as i32,
                        use_scalar_position: true,
                        positions_i32: DevicePtr::NULL,
                        q_bf16: forward.aux3.ptr(),
                        k_bf16: forward.aux.ptr(),
                        scalar_position_device_i32: DevicePtr::NULL,
                    },
                    attention: DecodeInterpreterAttentionParams {
                        spec: forward.interpreter_attention_spec.ptr(),
                    },
                    q_proj_gate: DecodeInterpreterQProjSigmoidGateParams {
                        rows: 1,
                        heads: self.topology.attention_num_heads,
                        head_dim: self.topology.attention_head_dim,
                        gate_bf16: forward.qkv.ptr(),
                        input_bf16: forward.aux3.ptr(),
                        output_bf16: forward.aux3.ptr(),
                    },
                    o_input_quant: DecodeInterpreterNvfp4QuantizeParams {
                        values: q_values,
                        input_tensor_scale_f32: o_input_tensor_scale_f32,
                        input_bf16: forward.aux3.ptr(),
                        output_fp4: forward.activation_fp4.ptr(),
                        output_scale_e4m3: forward.activation_scale.ptr(),
                        output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                    },
                    o_proj,
                },
            },
        );
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_attention_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter full-attn input-layer program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_attention_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_attention_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter full-attn input-layer program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_attention_counters.bytes()
            )));
        }

        forward
            .interpreter_attention_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_attention_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_attention_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_attention_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })?;
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn build_interpreter_full_transformer_layer_decode(
        &self,
        layer: &FullAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        position: usize,
        position_device_i32: DevicePtr,
        attention_context_limit: usize,
        input_quantized: Nvfp4ActivationQuant<'_>,
        mlp_quantized: Nvfp4ActivationQuant<'_>,
        input_residual_bf16: DevicePtr,
        input_norm_weight_bf16: DevicePtr,
        post_attention_norm_weight_bf16: DevicePtr,
        attention_spec_ptr: DevicePtr,
    ) -> Result<Option<(DecodeInterpreterProgram, Vec<u8>)>> {
        let bf16_kv_cache_dtype = Self::attention_kv_cache_dtype_code(KvCacheDtype::Bf16)?;
        let configured_kv_cache_dtype =
            Self::attention_kv_cache_dtype_code(self.config.kv_cache_dtype)?;
        if configured_kv_cache_dtype != bf16_kv_cache_dtype {
            return Ok(None);
        }

        let cache = runtime
            .kv_cache
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("KV cache was not allocated".to_owned()))?;
        let layout = self
            .state
            .kv_cache
            .layers
            .iter()
            .find(|layout| layout.global_layer_index == layer.layer_index)
            .ok_or_else(|| {
                CoreError::Runtime(format!(
                    "missing KV-cache layout for layer {}",
                    layer.layer_index
                ))
            })?;

        let q_values = self.topology.attention_num_heads * self.topology.attention_head_dim;
        let kv_values = self.topology.attention_num_kv_heads * self.topology.attention_head_dim;
        let hidden = self.topology.hidden_size;
        let input_fp4 = forward.activation_fp4.ptr();
        let input_scale_e4m3 = forward.activation_scale.ptr();

        let Some(q_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.q_proj,
            input_quantized,
            input_fp4,
            input_scale_e4m3,
            forward.qkv.ptr(),
        )?
        else {
            return Ok(None);
        };
        let Some(k_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.k_proj,
            input_quantized,
            input_fp4,
            input_scale_e4m3,
            forward.aux.ptr(),
        )?
        else {
            return Ok(None);
        };
        let Some(v_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.v_proj,
            input_quantized,
            input_fp4,
            input_scale_e4m3,
            forward.aux2.ptr(),
        )?
        else {
            return Ok(None);
        };
        if q_proj.m != q_values * 2
            || q_proj.k != hidden
            || k_proj.m != kv_values
            || k_proj.k != hidden
            || v_proj.m != kv_values
            || v_proj.k != hidden
        {
            return Ok(None);
        }

        let LinearWeightBinding::Nvfp4 {
            input_scale: o_input_scale,
            ..
        } = &layer.o_proj
        else {
            return Ok(None);
        };
        let o_quantized = Nvfp4ActivationQuant {
            in_features: q_values,
            input_scale: o_input_scale,
        };
        let Some(o_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.o_proj,
            o_quantized,
            forward.activation_fp4.ptr(),
            forward.activation_scale.ptr(),
            forward.block_out.ptr(),
        )?
        else {
            return Ok(None);
        };
        if o_proj.m != hidden || o_proj.k != q_values {
            return Ok(None);
        }

        let Some(mlp) =
            self.decode_interpreter_mlp_params_from_common(&layer.common, forward, mlp_quantized)?
        else {
            return Ok(None);
        };

        let decode_n_splits =
            self.decode_attention_n_splits_for_context_limit(attention_context_limit);

        let mut attention_spec = AttentionDecodeSpec {
            layer_index: layer.layer_index,
            position,
            q_bf16: forward.aux3.ptr(),
            k_bf16: forward.aux.ptr(),
            v_bf16: forward.aux2.ptr(),
            kv_cache_k: cache.ptr_at(layout.k_offset_bytes as usize)?,
            kv_cache_v: cache.ptr_at(layout.v_offset_bytes as usize)?,
            kv_cache_metadata: DevicePtr::NULL,
            output_bf16: forward.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            kv_cache_dtype: bf16_kv_cache_dtype,
            position_device_i32,
            partial_acc_f32: DevicePtr::NULL,
            partial_max_f32: DevicePtr::NULL,
            partial_denom_f32: DevicePtr::NULL,
            decode_n_splits: decode_n_splits.min(1),
            split_timesteps_per_block: 0,
        };
        attention_spec.decode_n_splits = attention_spec.decode_n_splits.min(1);
        let spec_bytes = attention_decode_spec_abi_bytes(&attention_spec);

        let weights = self.cuda_weights()?;
        let program = DecodeInterpreterProgram::compile_full_transformer_layer_decode(
            DecodeInterpreterFullTransformerLayerParams {
                attention: DecodeInterpreterFullAttentionInputLayerParams {
                    input_norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                        hidden,
                        eps: 1.0e-6,
                        input_tensor_scale_f32: self
                            .tensor_scalar_f32(weights, input_quantized.input_scale)?,
                        input_bf16: forward.hidden.ptr(),
                        weight_bf16: input_norm_weight_bf16,
                        residual_bf16: input_residual_bf16,
                        residual_out_bf16: forward.normed.ptr(),
                        output_bf16: DevicePtr::NULL,
                        output_fp4: forward.activation_fp4.ptr(),
                        output_scale_e4m3: forward.activation_scale.ptr(),
                        output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                    },
                    layer: DecodeInterpreterFullAttentionLayerParams {
                        q_proj,
                        k_proj,
                        v_proj,
                        q_proj_deinterleave: DecodeInterpreterQProjDeinterleaveParams {
                            rows: 1,
                            heads: self.topology.attention_num_heads,
                            head_dim: self.topology.attention_head_dim,
                            input_bf16: forward.qkv.ptr(),
                            output_bf16: forward.aux3.ptr(),
                        },
                        q_norm: DecodeInterpreterRmsNormBf16Params {
                            rows: self.topology.attention_num_heads,
                            hidden: self.topology.attention_head_dim,
                            eps: 1.0e-6,
                            direct_weight: false,
                            input_bf16: forward.aux3.ptr(),
                            weight_bf16: self.tensor_ptr(weights, &layer.q_norm)?,
                            residual_bf16: DevicePtr::NULL,
                            residual_out_bf16: DevicePtr::NULL,
                            output_bf16: forward.aux3.ptr(),
                        },
                        k_norm: DecodeInterpreterRmsNormBf16Params {
                            rows: self.topology.attention_num_kv_heads,
                            hidden: self.topology.attention_head_dim,
                            eps: 1.0e-6,
                            direct_weight: false,
                            input_bf16: forward.aux.ptr(),
                            weight_bf16: self.tensor_ptr(weights, &layer.k_norm)?,
                            residual_bf16: DevicePtr::NULL,
                            residual_out_bf16: DevicePtr::NULL,
                            output_bf16: forward.aux.ptr(),
                        },
                        rope: DecodeInterpreterRopeParams {
                            tokens: 1,
                            q_heads: self.topology.attention_num_heads,
                            kv_heads: self.topology.attention_num_kv_heads,
                            head_dim: self.topology.attention_head_dim,
                            rope_dims: self.topology.attention_rope_dims(),
                            base_theta: self.topology.rope_theta,
                            position_i32: position as i32,
                            use_scalar_position: true,
                            positions_i32: DevicePtr::NULL,
                            q_bf16: forward.aux3.ptr(),
                            k_bf16: forward.aux.ptr(),
                            scalar_position_device_i32: position_device_i32,
                        },
                        attention: DecodeInterpreterAttentionParams {
                            spec: attention_spec_ptr,
                        },
                        q_proj_gate: DecodeInterpreterQProjSigmoidGateParams {
                            rows: 1,
                            heads: self.topology.attention_num_heads,
                            head_dim: self.topology.attention_head_dim,
                            gate_bf16: forward.qkv.ptr(),
                            input_bf16: forward.aux3.ptr(),
                            output_bf16: forward.aux3.ptr(),
                        },
                        o_input_quant: DecodeInterpreterNvfp4QuantizeParams {
                            values: q_values,
                            input_tensor_scale_f32: self
                                .tensor_scalar_f32(weights, o_input_scale)?,
                            input_bf16: forward.aux3.ptr(),
                            output_fp4: forward.activation_fp4.ptr(),
                            output_scale_e4m3: forward.activation_scale.ptr(),
                            output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                        },
                        o_proj,
                    },
                },
                post: DecodeInterpreterNormMlpParams {
                    norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                        hidden,
                        eps: 1.0e-6,
                        input_tensor_scale_f32: self
                            .tensor_scalar_f32(weights, mlp_quantized.input_scale)?,
                        input_bf16: forward.block_out.ptr(),
                        weight_bf16: post_attention_norm_weight_bf16,
                        residual_bf16: forward.normed.ptr(),
                        residual_out_bf16: forward.residual.ptr(),
                        output_bf16: DevicePtr::NULL,
                        output_fp4: forward.activation_fp4.ptr(),
                        output_scale_e4m3: forward.activation_scale.ptr(),
                        output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                    },
                    mlp,
                },
            },
        );
        Ok(Some((program, spec_bytes)))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn run_interpreter_full_transformer_layer_decode(
        &self,
        layer: &FullAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        position: usize,
        input_quantized: Nvfp4ActivationQuant<'_>,
        mlp_quantized: Nvfp4ActivationQuant<'_>,
        input_residual_bf16: DevicePtr,
        input_norm_weight_bf16: DevicePtr,
        post_attention_norm_weight_bf16: DevicePtr,
    ) -> Result<bool> {
        if !decode_interpreter_full_transformer_layer_enabled(
            self.decode_interpreter_decode_enabled(),
        ) {
            return Ok(false);
        }

        let bf16_kv_cache_dtype = Self::attention_kv_cache_dtype_code(KvCacheDtype::Bf16)?;
        let configured_kv_cache_dtype =
            Self::attention_kv_cache_dtype_code(self.config.kv_cache_dtype)?;
        if configured_kv_cache_dtype != bf16_kv_cache_dtype {
            return Ok(false);
        }

        let cache = runtime
            .kv_cache
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("KV cache was not allocated".to_owned()))?;
        let layout = self
            .state
            .kv_cache
            .layers
            .iter()
            .find(|layout| layout.global_layer_index == layer.layer_index)
            .ok_or_else(|| {
                CoreError::Runtime(format!(
                    "missing KV-cache layout for layer {}",
                    layer.layer_index
                ))
            })?;

        let q_values = self.topology.attention_num_heads * self.topology.attention_head_dim;
        let kv_values = self.topology.attention_num_kv_heads * self.topology.attention_head_dim;
        let hidden = self.topology.hidden_size;
        let input_fp4 = forward.activation_fp4.ptr();
        let input_scale_e4m3 = forward.activation_scale.ptr();

        let Some(q_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.q_proj,
            input_quantized,
            input_fp4,
            input_scale_e4m3,
            forward.qkv.ptr(),
        )?
        else {
            return Ok(false);
        };
        let Some(k_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.k_proj,
            input_quantized,
            input_fp4,
            input_scale_e4m3,
            forward.aux.ptr(),
        )?
        else {
            return Ok(false);
        };
        let Some(v_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.v_proj,
            input_quantized,
            input_fp4,
            input_scale_e4m3,
            forward.aux2.ptr(),
        )?
        else {
            return Ok(false);
        };
        if q_proj.m != q_values * 2
            || q_proj.k != hidden
            || k_proj.m != kv_values
            || k_proj.k != hidden
            || v_proj.m != kv_values
            || v_proj.k != hidden
        {
            return Ok(false);
        }

        let LinearWeightBinding::Nvfp4 {
            input_scale: o_input_scale,
            ..
        } = &layer.o_proj
        else {
            return Ok(false);
        };
        let o_quantized = Nvfp4ActivationQuant {
            in_features: q_values,
            input_scale: o_input_scale,
        };
        let Some(o_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.o_proj,
            o_quantized,
            forward.activation_fp4.ptr(),
            forward.activation_scale.ptr(),
            forward.block_out.ptr(),
        )?
        else {
            return Ok(false);
        };
        if o_proj.m != hidden || o_proj.k != q_values {
            return Ok(false);
        }

        let Some(mlp) =
            self.decode_interpreter_mlp_params_from_common(&layer.common, forward, mlp_quantized)?
        else {
            return Ok(false);
        };

        let attention_context_limit = self.decode_attention_context_limit_for_position(position);
        let decode_n_splits =
            self.decode_attention_n_splits_for_context_limit(attention_context_limit);
        if decode_n_splits > 1 {
            return Ok(false);
        }

        let mut attention_spec = AttentionDecodeSpec {
            layer_index: layer.layer_index,
            position,
            q_bf16: forward.aux3.ptr(),
            k_bf16: forward.aux.ptr(),
            v_bf16: forward.aux2.ptr(),
            kv_cache_k: cache.ptr_at(layout.k_offset_bytes as usize)?,
            kv_cache_v: cache.ptr_at(layout.v_offset_bytes as usize)?,
            kv_cache_metadata: DevicePtr::NULL,
            output_bf16: forward.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            kv_cache_dtype: bf16_kv_cache_dtype,
            position_device_i32: DevicePtr::NULL,
            partial_acc_f32: DevicePtr::NULL,
            partial_max_f32: DevicePtr::NULL,
            partial_denom_f32: DevicePtr::NULL,
            decode_n_splits,
            split_timesteps_per_block: 0,
        };
        attention_spec.decode_n_splits = attention_spec.decode_n_splits.min(1);

        let spec_bytes = attention_decode_spec_abi_bytes(&attention_spec);
        if spec_bytes.len() > forward.interpreter_attention_spec.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter full transformer layer spec needs {} bytes but buffer has {}",
                spec_bytes.len(),
                forward.interpreter_attention_spec.bytes()
            )));
        }
        forward
            .interpreter_attention_spec
            .copy_from_host(&spec_bytes)?;

        let weights = self.cuda_weights()?;
        let compiled = DecodeInterpreterProgram::compile_full_transformer_layer_decode(
            DecodeInterpreterFullTransformerLayerParams {
                attention: DecodeInterpreterFullAttentionInputLayerParams {
                    input_norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                        hidden,
                        eps: 1.0e-6,
                        input_tensor_scale_f32: self
                            .tensor_scalar_f32(weights, input_quantized.input_scale)?,
                        input_bf16: forward.hidden.ptr(),
                        weight_bf16: input_norm_weight_bf16,
                        residual_bf16: input_residual_bf16,
                        residual_out_bf16: forward.normed.ptr(),
                        output_bf16: DevicePtr::NULL,
                        output_fp4: forward.activation_fp4.ptr(),
                        output_scale_e4m3: forward.activation_scale.ptr(),
                        output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                    },
                    layer: DecodeInterpreterFullAttentionLayerParams {
                        q_proj,
                        k_proj,
                        v_proj,
                        q_proj_deinterleave: DecodeInterpreterQProjDeinterleaveParams {
                            rows: 1,
                            heads: self.topology.attention_num_heads,
                            head_dim: self.topology.attention_head_dim,
                            input_bf16: forward.qkv.ptr(),
                            output_bf16: forward.aux3.ptr(),
                        },
                        q_norm: DecodeInterpreterRmsNormBf16Params {
                            rows: self.topology.attention_num_heads,
                            hidden: self.topology.attention_head_dim,
                            eps: 1.0e-6,
                            direct_weight: false,
                            input_bf16: forward.aux3.ptr(),
                            weight_bf16: self.tensor_ptr(weights, &layer.q_norm)?,
                            residual_bf16: DevicePtr::NULL,
                            residual_out_bf16: DevicePtr::NULL,
                            output_bf16: forward.aux3.ptr(),
                        },
                        k_norm: DecodeInterpreterRmsNormBf16Params {
                            rows: self.topology.attention_num_kv_heads,
                            hidden: self.topology.attention_head_dim,
                            eps: 1.0e-6,
                            direct_weight: false,
                            input_bf16: forward.aux.ptr(),
                            weight_bf16: self.tensor_ptr(weights, &layer.k_norm)?,
                            residual_bf16: DevicePtr::NULL,
                            residual_out_bf16: DevicePtr::NULL,
                            output_bf16: forward.aux.ptr(),
                        },
                        rope: DecodeInterpreterRopeParams {
                            tokens: 1,
                            q_heads: self.topology.attention_num_heads,
                            kv_heads: self.topology.attention_num_kv_heads,
                            head_dim: self.topology.attention_head_dim,
                            rope_dims: self.topology.attention_rope_dims(),
                            base_theta: self.topology.rope_theta,
                            position_i32: position as i32,
                            use_scalar_position: true,
                            positions_i32: DevicePtr::NULL,
                            q_bf16: forward.aux3.ptr(),
                            k_bf16: forward.aux.ptr(),
                            scalar_position_device_i32: DevicePtr::NULL,
                        },
                        attention: DecodeInterpreterAttentionParams {
                            spec: forward.interpreter_attention_spec.ptr(),
                        },
                        q_proj_gate: DecodeInterpreterQProjSigmoidGateParams {
                            rows: 1,
                            heads: self.topology.attention_num_heads,
                            head_dim: self.topology.attention_head_dim,
                            gate_bf16: forward.qkv.ptr(),
                            input_bf16: forward.aux3.ptr(),
                            output_bf16: forward.aux3.ptr(),
                        },
                        o_input_quant: DecodeInterpreterNvfp4QuantizeParams {
                            values: q_values,
                            input_tensor_scale_f32: self
                                .tensor_scalar_f32(weights, o_input_scale)?,
                            input_bf16: forward.aux3.ptr(),
                            output_fp4: forward.activation_fp4.ptr(),
                            output_scale_e4m3: forward.activation_scale.ptr(),
                            output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                        },
                        o_proj,
                    },
                },
                post: DecodeInterpreterNormMlpParams {
                    norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                        hidden,
                        eps: 1.0e-6,
                        input_tensor_scale_f32: self
                            .tensor_scalar_f32(weights, mlp_quantized.input_scale)?,
                        input_bf16: forward.block_out.ptr(),
                        weight_bf16: post_attention_norm_weight_bf16,
                        residual_bf16: forward.normed.ptr(),
                        residual_out_bf16: forward.residual.ptr(),
                        output_bf16: DevicePtr::NULL,
                        output_fp4: forward.activation_fp4.ptr(),
                        output_scale_e4m3: forward.activation_scale.ptr(),
                        output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                    },
                    mlp,
                },
            },
        );
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_attention_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter full transformer layer program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_attention_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_attention_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter full transformer layer program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_attention_counters.bytes()
            )));
        }

        forward
            .interpreter_attention_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_attention_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_attention_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_attention_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })?;
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn run_interpreter_full_attention_layer_decode(
        &self,
        layer: &FullAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        position: usize,
        position_device_i32: DevicePtr,
        quantized: Nvfp4ActivationQuant<'_>,
    ) -> Result<bool> {
        if !decode_interpreter_full_attention_layer_enabled(
            self.decode_interpreter_decode_enabled(),
        ) || position_device_i32 != DevicePtr::NULL
        {
            return Ok(false);
        }

        let bf16_kv_cache_dtype = Self::attention_kv_cache_dtype_code(KvCacheDtype::Bf16)?;
        let configured_kv_cache_dtype =
            Self::attention_kv_cache_dtype_code(self.config.kv_cache_dtype)?;
        if configured_kv_cache_dtype != bf16_kv_cache_dtype {
            return Ok(false);
        }

        let cache = runtime
            .kv_cache
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("KV cache was not allocated".to_owned()))?;
        let layout = self
            .state
            .kv_cache
            .layers
            .iter()
            .find(|layout| layout.global_layer_index == layer.layer_index)
            .ok_or_else(|| {
                CoreError::Runtime(format!(
                    "missing KV-cache layout for layer {}",
                    layer.layer_index
                ))
            })?;

        let q_values = self.topology.attention_num_heads * self.topology.attention_head_dim;
        let kv_values = self.topology.attention_num_kv_heads * self.topology.attention_head_dim;
        let hidden = self.topology.hidden_size;
        let input_fp4 = forward.activation_fp4.ptr();
        let input_scale_e4m3 = forward.activation_scale.ptr();

        let Some(q_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.q_proj,
            quantized,
            input_fp4,
            input_scale_e4m3,
            forward.qkv.ptr(),
        )?
        else {
            return Ok(false);
        };
        let Some(k_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.k_proj,
            quantized,
            input_fp4,
            input_scale_e4m3,
            forward.aux.ptr(),
        )?
        else {
            return Ok(false);
        };
        let Some(v_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.v_proj,
            quantized,
            input_fp4,
            input_scale_e4m3,
            forward.aux2.ptr(),
        )?
        else {
            return Ok(false);
        };
        if q_proj.m != q_values * 2
            || q_proj.k != hidden
            || k_proj.m != kv_values
            || k_proj.k != hidden
            || v_proj.m != kv_values
            || v_proj.k != hidden
        {
            return Ok(false);
        }

        let LinearWeightBinding::Nvfp4 {
            input_scale: o_input_scale,
            ..
        } = &layer.o_proj
        else {
            return Ok(false);
        };
        let o_quantized = Nvfp4ActivationQuant {
            in_features: q_values,
            input_scale: o_input_scale,
        };
        let Some(o_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.o_proj,
            o_quantized,
            forward.activation_fp4.ptr(),
            forward.activation_scale.ptr(),
            forward.block_out.ptr(),
        )?
        else {
            return Ok(false);
        };
        if o_proj.m != hidden || o_proj.k != q_values {
            return Ok(false);
        }

        let attention_context_limit = self.decode_attention_context_limit_for_position(position);
        let decode_n_splits =
            self.decode_attention_n_splits_for_context_limit(attention_context_limit);
        if decode_n_splits > 1 {
            return Ok(false);
        }

        let mut attention_spec = AttentionDecodeSpec {
            layer_index: layer.layer_index,
            position,
            q_bf16: forward.aux3.ptr(),
            k_bf16: forward.aux.ptr(),
            v_bf16: forward.aux2.ptr(),
            kv_cache_k: cache.ptr_at(layout.k_offset_bytes as usize)?,
            kv_cache_v: cache.ptr_at(layout.v_offset_bytes as usize)?,
            kv_cache_metadata: DevicePtr::NULL,
            output_bf16: forward.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            kv_cache_dtype: bf16_kv_cache_dtype,
            position_device_i32: DevicePtr::NULL,
            partial_acc_f32: DevicePtr::NULL,
            partial_max_f32: DevicePtr::NULL,
            partial_denom_f32: DevicePtr::NULL,
            decode_n_splits,
            split_timesteps_per_block: 0,
        };
        attention_spec.decode_n_splits = attention_spec.decode_n_splits.min(1);

        let spec_bytes = attention_decode_spec_abi_bytes(&attention_spec);
        if spec_bytes.len() > forward.interpreter_attention_spec.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter full-attn layer spec needs {} bytes but buffer has {}",
                spec_bytes.len(),
                forward.interpreter_attention_spec.bytes()
            )));
        }
        forward
            .interpreter_attention_spec
            .copy_from_host(&spec_bytes)?;

        let weights = self.cuda_weights()?;
        let o_input_tensor_scale_f32 = self.tensor_scalar_f32(weights, o_input_scale)?;
        let compiled = DecodeInterpreterProgram::compile_full_attention_layer_decode(
            DecodeInterpreterFullAttentionLayerParams {
                q_proj,
                k_proj,
                v_proj,
                q_proj_deinterleave: DecodeInterpreterQProjDeinterleaveParams {
                    rows: 1,
                    heads: self.topology.attention_num_heads,
                    head_dim: self.topology.attention_head_dim,
                    input_bf16: forward.qkv.ptr(),
                    output_bf16: forward.aux3.ptr(),
                },
                q_norm: DecodeInterpreterRmsNormBf16Params {
                    rows: self.topology.attention_num_heads,
                    hidden: self.topology.attention_head_dim,
                    eps: 1.0e-6,
                    direct_weight: false,
                    input_bf16: forward.aux3.ptr(),
                    weight_bf16: self.tensor_ptr(weights, &layer.q_norm)?,
                    residual_bf16: DevicePtr::NULL,
                    residual_out_bf16: DevicePtr::NULL,
                    output_bf16: forward.aux3.ptr(),
                },
                k_norm: DecodeInterpreterRmsNormBf16Params {
                    rows: self.topology.attention_num_kv_heads,
                    hidden: self.topology.attention_head_dim,
                    eps: 1.0e-6,
                    direct_weight: false,
                    input_bf16: forward.aux.ptr(),
                    weight_bf16: self.tensor_ptr(weights, &layer.k_norm)?,
                    residual_bf16: DevicePtr::NULL,
                    residual_out_bf16: DevicePtr::NULL,
                    output_bf16: forward.aux.ptr(),
                },
                rope: DecodeInterpreterRopeParams {
                    tokens: 1,
                    q_heads: self.topology.attention_num_heads,
                    kv_heads: self.topology.attention_num_kv_heads,
                    head_dim: self.topology.attention_head_dim,
                    rope_dims: self.topology.attention_rope_dims(),
                    base_theta: self.topology.rope_theta,
                    position_i32: position as i32,
                    use_scalar_position: true,
                    positions_i32: DevicePtr::NULL,
                    q_bf16: forward.aux3.ptr(),
                    k_bf16: forward.aux.ptr(),
                    scalar_position_device_i32: DevicePtr::NULL,
                },
                attention: DecodeInterpreterAttentionParams {
                    spec: forward.interpreter_attention_spec.ptr(),
                },
                q_proj_gate: DecodeInterpreterQProjSigmoidGateParams {
                    rows: 1,
                    heads: self.topology.attention_num_heads,
                    head_dim: self.topology.attention_head_dim,
                    gate_bf16: forward.qkv.ptr(),
                    input_bf16: forward.aux3.ptr(),
                    output_bf16: forward.aux3.ptr(),
                },
                o_input_quant: DecodeInterpreterNvfp4QuantizeParams {
                    values: q_values,
                    input_tensor_scale_f32: o_input_tensor_scale_f32,
                    input_bf16: forward.aux3.ptr(),
                    output_fp4: forward.activation_fp4.ptr(),
                    output_scale_e4m3: forward.activation_scale.ptr(),
                    output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                },
                o_proj,
            },
        );
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_attention_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter full-attn layer program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_attention_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_attention_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter full-attn layer program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_attention_counters.bytes()
            )));
        }

        forward
            .interpreter_attention_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_attention_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_attention_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_attention_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })?;
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn run_interpreter_linear_attention_tail_decode(
        &self,
        layer: &LinearAttentionLayerWeights,
        forward: &GpuForwardBuffers,
        z_bf16: DevicePtr,
        deltanet_output_bf16: DevicePtr,
    ) -> Result<bool> {
        if !decode_interpreter_linear_attention_tail_enabled(
            self.decode_interpreter_decode_enabled(),
        ) {
            return Ok(false);
        }
        let value_dim = self.topology.linear_attention_value_dim();
        let hidden = self.topology.hidden_size;
        let LinearWeightBinding::Nvfp4 {
            input_scale: out_input_scale,
            ..
        } = &layer.out_proj
        else {
            return Ok(false);
        };
        let out_quantized = Nvfp4ActivationQuant {
            in_features: value_dim,
            input_scale: out_input_scale,
        };
        let Some(out_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.out_proj,
            out_quantized,
            forward.activation_fp4.ptr(),
            forward.activation_scale.ptr(),
            forward.block_out.ptr(),
        )?
        else {
            return Ok(false);
        };
        if out_proj.m != hidden || out_proj.k != value_dim {
            return Ok(false);
        }

        let weights = self.cuda_weights()?;
        let compiled = DecodeInterpreterProgram::compile_linear_attention_tail_decode(
            DecodeInterpreterLinearAttentionTailParams {
                norm: DecodeInterpreterRmsNormBf16Params {
                    rows: self.topology.linear_num_value_heads,
                    hidden: self.topology.linear_value_head_dim,
                    eps: 1.0e-6,
                    direct_weight: true,
                    input_bf16: deltanet_output_bf16,
                    weight_bf16: self.tensor_ptr(weights, &layer.norm_weight)?,
                    residual_bf16: DevicePtr::NULL,
                    residual_out_bf16: DevicePtr::NULL,
                    output_bf16: forward.aux2.ptr(),
                },
                swiglu: DecodeInterpreterSwiGluBf16Params {
                    rows: 1,
                    intermediate: value_dim,
                    gate_bf16: z_bf16,
                    up_bf16: forward.aux2.ptr(),
                    output_bf16: forward.aux3.ptr(),
                },
                quant: DecodeInterpreterNvfp4QuantizeParams {
                    values: value_dim,
                    input_tensor_scale_f32: self.tensor_scalar_f32(weights, out_input_scale)?,
                    input_bf16: forward.aux3.ptr(),
                    output_fp4: forward.activation_fp4.ptr(),
                    output_scale_e4m3: forward.activation_scale.ptr(),
                    output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                },
                out_proj,
            },
        );
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_mlp_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear-attn tail program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_mlp_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_mlp_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear-attn tail program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_mlp_counters.bytes()
            )));
        }

        forward
            .interpreter_mlp_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_mlp_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_mlp_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_mlp_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })?;
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn run_interpreter_linear_attention_layer_decode(
        &self,
        layer: &LinearAttentionLayerWeights,
        fused: &LinearAttnInProjFused,
        forward: &GpuForwardBuffers,
        quantized: Nvfp4ActivationQuant<'_>,
        conv_history_bf16: DevicePtr,
        state_bf16: DevicePtr,
    ) -> Result<bool> {
        if !decode_interpreter_linear_attention_layer_enabled(
            self.decode_interpreter_decode_enabled(),
        ) {
            return Ok(false);
        }

        let qkv_dim = self.topology.linear_attention_qkv_dim();
        let key_dim = self.topology.linear_num_key_heads * self.topology.linear_key_head_dim;
        let value_dim = self.topology.linear_attention_value_dim();
        let hidden = self.topology.hidden_size;

        let b_bf16 = forward
            .qkv
            .ptr()
            .offset_bytes(fused.b_offset * 2)
            .ok_or_else(|| CoreError::Runtime("fused DeltaNet b offset overflow".to_owned()))?;
        let a_bf16 = forward
            .qkv
            .ptr()
            .offset_bytes(fused.a_offset * 2)
            .ok_or_else(|| CoreError::Runtime("fused DeltaNet a offset overflow".to_owned()))?;
        let z_bf16 = forward
            .qkv
            .ptr()
            .offset_bytes(fused.z_offset * 2)
            .ok_or_else(|| CoreError::Runtime("fused DeltaNet z offset overflow".to_owned()))?;

        let LinearWeightBinding::Nvfp4 {
            weight: qkv_weight,
            tensor_scale: qkv_tensor_scale,
            input_scale: qkv_input_scale,
            ..
        } = &layer.in_proj_qkv
        else {
            return Ok(false);
        };
        let in_features = Self::nvfp4_in_features(qkv_weight)?;
        if in_features != quantized.in_features
            || !decode_interpreter_nvfp4_gemv_supports(fused.combined_out_features, in_features)
        {
            return Ok(false);
        }
        self.validate_nvfp4_input_scale(qkv_weight, quantized.input_scale, qkv_input_scale)?;

        let LinearWeightBinding::Nvfp4 {
            input_scale: out_input_scale,
            ..
        } = &layer.out_proj
        else {
            return Ok(false);
        };
        let out_quantized = Nvfp4ActivationQuant {
            in_features: value_dim,
            input_scale: out_input_scale,
        };
        let Some(out_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.out_proj,
            out_quantized,
            forward.activation_fp4.ptr(),
            forward.activation_scale.ptr(),
            forward.block_out.ptr(),
        )?
        else {
            return Ok(false);
        };
        if out_proj.m != hidden || out_proj.k != value_dim {
            return Ok(false);
        }

        let weights = self.cuda_weights()?;
        let deltanet_spec = DeltaNetDecodeSpec {
            layer_index: layer.layer_index,
            tokens_in_persistent_loop: 1,
            q_token_stride: 0,
            k_token_stride: 0,
            v_token_stride: 0,
            q_bf16: forward.aux.ptr(),
            k_bf16: forward.aux.ptr_at(key_dim * 2)?,
            v_bf16: forward.aux.ptr_at(key_dim * 4)?,
            state_bf16,
            conv_history_bf16,
            output_bf16: forward.aux3.ptr(),
            gate_f32: forward.gate_f32.ptr(),
            beta_f32: forward.beta_f32.ptr(),
            shape: DeltaNetShape {
                qk_heads: self.topology.linear_num_key_heads,
                v_heads: self.topology.linear_num_value_heads,
                key_dim: self.topology.linear_key_head_dim,
                value_dim: self.topology.linear_value_head_dim,
                conv_kernel: self.topology.linear_conv_kernel_dim,
            },
            state_decay: 1.0,
            update_scale: 1.0,
            qk_l2norm: true,
        };
        let spec_bytes = deltanet_decode_spec_abi_bytes(&deltanet_spec);
        if spec_bytes.len() > forward.interpreter_deltanet_spec.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear-attn layer DeltaNet spec needs {} bytes but buffer has {}",
                spec_bytes.len(),
                forward.interpreter_deltanet_spec.bytes()
            )));
        }
        forward
            .interpreter_deltanet_spec
            .copy_from_host(&spec_bytes)?;

        let compiled = DecodeInterpreterProgram::compile_linear_attention_layer_decode(
            DecodeInterpreterLinearAttentionLayerParams {
                in_proj: DecodeInterpreterNvfp4GemvParams {
                    m: fused.combined_out_features,
                    k: in_features,
                    alpha: self.tensor_scalar_f32(weights, qkv_tensor_scale)?
                        * self.tensor_scalar_f32(weights, quantized.input_scale)?,
                    a_fp4: fused.combined_weight.ptr(),
                    a_scale_e4m3: fused.combined_block_scale.ptr(),
                    b_fp4: forward.activation_fp4.ptr(),
                    b_scale_e4m3: forward.activation_scale.ptr(),
                    c_bf16: forward.qkv.ptr(),
                },
                post_inproj: DecodeInterpreterLinearAttentionPostInProjParams {
                    conv_gdn: DecodeInterpreterConv1dGdnGateFusedParams {
                        channels: qkv_dim,
                        kernel_size: self.topology.linear_conv_kernel_dim,
                        conv_input_bf16: forward.qkv.ptr(),
                        conv_history_bf16,
                        conv_weight_bf16: self.tensor_ptr(weights, &layer.conv1d_weight)?,
                        conv_output_bf16: forward.aux.ptr(),
                        heads: self.topology.linear_num_value_heads,
                        gdn_a_bf16: a_bf16,
                        gdn_b_bf16: b_bf16,
                        gdn_a_log_bf16: self.tensor_ptr(weights, &layer.a_log)?,
                        gdn_dt_bias_bf16: self.tensor_ptr(weights, &layer.dt_bias)?,
                        gate_f32: forward.gate_f32.ptr(),
                        beta_f32: forward.beta_f32.ptr(),
                    },
                    deltanet: DecodeInterpreterDeltaNetParams {
                        spec: forward.interpreter_deltanet_spec.ptr(),
                    },
                    tail: DecodeInterpreterLinearAttentionTailParams {
                        norm: DecodeInterpreterRmsNormBf16Params {
                            rows: self.topology.linear_num_value_heads,
                            hidden: self.topology.linear_value_head_dim,
                            eps: 1.0e-6,
                            direct_weight: true,
                            input_bf16: forward.aux3.ptr(),
                            weight_bf16: self.tensor_ptr(weights, &layer.norm_weight)?,
                            residual_bf16: DevicePtr::NULL,
                            residual_out_bf16: DevicePtr::NULL,
                            output_bf16: forward.aux2.ptr(),
                        },
                        swiglu: DecodeInterpreterSwiGluBf16Params {
                            rows: 1,
                            intermediate: value_dim,
                            gate_bf16: z_bf16,
                            up_bf16: forward.aux2.ptr(),
                            output_bf16: forward.aux3.ptr(),
                        },
                        quant: DecodeInterpreterNvfp4QuantizeParams {
                            values: value_dim,
                            input_tensor_scale_f32: self
                                .tensor_scalar_f32(weights, out_input_scale)?,
                            input_bf16: forward.aux3.ptr(),
                            output_fp4: forward.activation_fp4.ptr(),
                            output_scale_e4m3: forward.activation_scale.ptr(),
                            output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                        },
                        out_proj,
                    },
                },
            },
        );
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_deltanet_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear-attn layer program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_deltanet_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_deltanet_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear-attn layer program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_deltanet_counters.bytes()
            )));
        }

        forward
            .interpreter_deltanet_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_deltanet_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_deltanet_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_deltanet_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })?;
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn run_interpreter_linear_attention_input_layer_decode(
        &self,
        layer: &LinearAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        quantized: Nvfp4ActivationQuant<'_>,
        input_residual_bf16: DevicePtr,
        input_norm_weight_bf16: DevicePtr,
    ) -> Result<bool> {
        if !decode_interpreter_linear_attention_input_layer_enabled(
            self.decode_interpreter_decode_enabled(),
        ) {
            return Ok(false);
        }
        let Some(fused) = self.linear_attn_in_proj_fused_layer_opt(layer.layer_index) else {
            return Ok(false);
        };

        let qkv_dim = self.topology.linear_attention_qkv_dim();
        let key_dim = self.topology.linear_num_key_heads * self.topology.linear_key_head_dim;
        let value_dim = self.topology.linear_attention_value_dim();
        let hidden = self.topology.hidden_size;
        let layer_ordinal = self.linear_layer_ordinal(layer.layer_index)?;
        let conv_history_bf16 = runtime.conv_history.ptr_at(
            layer_ordinal * qkv_dim * self.topology.linear_conv_kernel_dim.saturating_sub(1) * 2,
        )?;
        let state_bf16 = runtime
            .deltanet_state
            .ptr_at(layer_ordinal * self.state.deltanet.state_bytes_per_layer as usize)?;

        let b_bf16 = forward
            .qkv
            .ptr()
            .offset_bytes(fused.b_offset * 2)
            .ok_or_else(|| CoreError::Runtime("fused DeltaNet b offset overflow".to_owned()))?;
        let a_bf16 = forward
            .qkv
            .ptr()
            .offset_bytes(fused.a_offset * 2)
            .ok_or_else(|| CoreError::Runtime("fused DeltaNet a offset overflow".to_owned()))?;
        let z_bf16 = forward
            .qkv
            .ptr()
            .offset_bytes(fused.z_offset * 2)
            .ok_or_else(|| CoreError::Runtime("fused DeltaNet z offset overflow".to_owned()))?;

        let LinearWeightBinding::Nvfp4 {
            weight: qkv_weight,
            tensor_scale: qkv_tensor_scale,
            input_scale: qkv_input_scale,
            ..
        } = &layer.in_proj_qkv
        else {
            return Ok(false);
        };
        let in_features = Self::nvfp4_in_features(qkv_weight)?;
        if in_features != quantized.in_features
            || !decode_interpreter_nvfp4_gemv_supports(fused.combined_out_features, in_features)
        {
            return Ok(false);
        }
        self.validate_nvfp4_input_scale(qkv_weight, quantized.input_scale, qkv_input_scale)?;

        let LinearWeightBinding::Nvfp4 {
            input_scale: out_input_scale,
            ..
        } = &layer.out_proj
        else {
            return Ok(false);
        };
        let out_quantized = Nvfp4ActivationQuant {
            in_features: value_dim,
            input_scale: out_input_scale,
        };
        let Some(out_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.out_proj,
            out_quantized,
            forward.activation_fp4.ptr(),
            forward.activation_scale.ptr(),
            forward.block_out.ptr(),
        )?
        else {
            return Ok(false);
        };
        if out_proj.m != hidden || out_proj.k != value_dim {
            return Ok(false);
        }

        let weights = self.cuda_weights()?;
        let deltanet_spec = DeltaNetDecodeSpec {
            layer_index: layer.layer_index,
            tokens_in_persistent_loop: 1,
            q_token_stride: 0,
            k_token_stride: 0,
            v_token_stride: 0,
            q_bf16: forward.aux.ptr(),
            k_bf16: forward.aux.ptr_at(key_dim * 2)?,
            v_bf16: forward.aux.ptr_at(key_dim * 4)?,
            state_bf16,
            conv_history_bf16,
            output_bf16: forward.aux3.ptr(),
            gate_f32: forward.gate_f32.ptr(),
            beta_f32: forward.beta_f32.ptr(),
            shape: DeltaNetShape {
                qk_heads: self.topology.linear_num_key_heads,
                v_heads: self.topology.linear_num_value_heads,
                key_dim: self.topology.linear_key_head_dim,
                value_dim: self.topology.linear_value_head_dim,
                conv_kernel: self.topology.linear_conv_kernel_dim,
            },
            state_decay: 1.0,
            update_scale: 1.0,
            qk_l2norm: true,
        };
        let spec_bytes = deltanet_decode_spec_abi_bytes(&deltanet_spec);
        if spec_bytes.len() > forward.interpreter_deltanet_spec.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear-attn input-layer DeltaNet spec needs {} bytes but buffer has {}",
                spec_bytes.len(),
                forward.interpreter_deltanet_spec.bytes()
            )));
        }
        forward
            .interpreter_deltanet_spec
            .copy_from_host(&spec_bytes)?;

        let compiled = DecodeInterpreterProgram::compile_linear_attention_input_layer_decode(
            DecodeInterpreterLinearAttentionInputLayerParams {
                input_norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                    hidden,
                    eps: 1.0e-6,
                    input_tensor_scale_f32: self
                        .tensor_scalar_f32(weights, quantized.input_scale)?,
                    input_bf16: forward.hidden.ptr(),
                    weight_bf16: input_norm_weight_bf16,
                    residual_bf16: input_residual_bf16,
                    residual_out_bf16: forward.residual.ptr(),
                    output_bf16: DevicePtr::NULL,
                    output_fp4: forward.activation_fp4.ptr(),
                    output_scale_e4m3: forward.activation_scale.ptr(),
                    output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                },
                layer: DecodeInterpreterLinearAttentionLayerParams {
                    in_proj: DecodeInterpreterNvfp4GemvParams {
                        m: fused.combined_out_features,
                        k: in_features,
                        alpha: self.tensor_scalar_f32(weights, qkv_tensor_scale)?
                            * self.tensor_scalar_f32(weights, quantized.input_scale)?,
                        a_fp4: fused.combined_weight.ptr(),
                        a_scale_e4m3: fused.combined_block_scale.ptr(),
                        b_fp4: forward.activation_fp4.ptr(),
                        b_scale_e4m3: forward.activation_scale.ptr(),
                        c_bf16: forward.qkv.ptr(),
                    },
                    post_inproj: DecodeInterpreterLinearAttentionPostInProjParams {
                        conv_gdn: DecodeInterpreterConv1dGdnGateFusedParams {
                            channels: qkv_dim,
                            kernel_size: self.topology.linear_conv_kernel_dim,
                            conv_input_bf16: forward.qkv.ptr(),
                            conv_history_bf16,
                            conv_weight_bf16: self.tensor_ptr(weights, &layer.conv1d_weight)?,
                            conv_output_bf16: forward.aux.ptr(),
                            heads: self.topology.linear_num_value_heads,
                            gdn_a_bf16: a_bf16,
                            gdn_b_bf16: b_bf16,
                            gdn_a_log_bf16: self.tensor_ptr(weights, &layer.a_log)?,
                            gdn_dt_bias_bf16: self.tensor_ptr(weights, &layer.dt_bias)?,
                            gate_f32: forward.gate_f32.ptr(),
                            beta_f32: forward.beta_f32.ptr(),
                        },
                        deltanet: DecodeInterpreterDeltaNetParams {
                            spec: forward.interpreter_deltanet_spec.ptr(),
                        },
                        tail: DecodeInterpreterLinearAttentionTailParams {
                            norm: DecodeInterpreterRmsNormBf16Params {
                                rows: self.topology.linear_num_value_heads,
                                hidden: self.topology.linear_value_head_dim,
                                eps: 1.0e-6,
                                direct_weight: true,
                                input_bf16: forward.aux3.ptr(),
                                weight_bf16: self.tensor_ptr(weights, &layer.norm_weight)?,
                                residual_bf16: DevicePtr::NULL,
                                residual_out_bf16: DevicePtr::NULL,
                                output_bf16: forward.aux2.ptr(),
                            },
                            swiglu: DecodeInterpreterSwiGluBf16Params {
                                rows: 1,
                                intermediate: value_dim,
                                gate_bf16: z_bf16,
                                up_bf16: forward.aux2.ptr(),
                                output_bf16: forward.aux3.ptr(),
                            },
                            quant: DecodeInterpreterNvfp4QuantizeParams {
                                values: value_dim,
                                input_tensor_scale_f32: self
                                    .tensor_scalar_f32(weights, out_input_scale)?,
                                input_bf16: forward.aux3.ptr(),
                                output_fp4: forward.activation_fp4.ptr(),
                                output_scale_e4m3: forward.activation_scale.ptr(),
                                output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                            },
                            out_proj,
                        },
                    },
                },
            },
        );
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_deltanet_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear-attn input-layer program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_deltanet_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_deltanet_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear-attn input-layer program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_deltanet_counters.bytes()
            )));
        }

        forward
            .interpreter_deltanet_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_deltanet_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_deltanet_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_deltanet_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })?;
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn run_interpreter_linear_transformer_layer_decode(
        &self,
        layer: &LinearAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        input_quantized: Nvfp4ActivationQuant<'_>,
        mlp_quantized: Nvfp4ActivationQuant<'_>,
        input_residual_bf16: DevicePtr,
        input_norm_weight_bf16: DevicePtr,
        post_attention_norm_weight_bf16: DevicePtr,
    ) -> Result<bool> {
        if !decode_interpreter_linear_transformer_layer_enabled(
            self.decode_interpreter_decode_enabled(),
        ) {
            return Ok(false);
        }
        let Some(fused) = self.linear_attn_in_proj_fused_layer_opt(layer.layer_index) else {
            return Ok(false);
        };

        let qkv_dim = self.topology.linear_attention_qkv_dim();
        let key_dim = self.topology.linear_num_key_heads * self.topology.linear_key_head_dim;
        let value_dim = self.topology.linear_attention_value_dim();
        let hidden = self.topology.hidden_size;
        let layer_ordinal = self.linear_layer_ordinal(layer.layer_index)?;
        let conv_history_bf16 = runtime.conv_history.ptr_at(
            layer_ordinal * qkv_dim * self.topology.linear_conv_kernel_dim.saturating_sub(1) * 2,
        )?;
        let state_bf16 = runtime
            .deltanet_state
            .ptr_at(layer_ordinal * self.state.deltanet.state_bytes_per_layer as usize)?;

        let b_bf16 = forward
            .qkv
            .ptr()
            .offset_bytes(fused.b_offset * 2)
            .ok_or_else(|| CoreError::Runtime("fused DeltaNet b offset overflow".to_owned()))?;
        let a_bf16 = forward
            .qkv
            .ptr()
            .offset_bytes(fused.a_offset * 2)
            .ok_or_else(|| CoreError::Runtime("fused DeltaNet a offset overflow".to_owned()))?;
        let z_bf16 = forward
            .qkv
            .ptr()
            .offset_bytes(fused.z_offset * 2)
            .ok_or_else(|| CoreError::Runtime("fused DeltaNet z offset overflow".to_owned()))?;

        let LinearWeightBinding::Nvfp4 {
            weight: qkv_weight,
            tensor_scale: qkv_tensor_scale,
            input_scale: qkv_input_scale,
            ..
        } = &layer.in_proj_qkv
        else {
            return Ok(false);
        };
        let in_features = Self::nvfp4_in_features(qkv_weight)?;
        if in_features != input_quantized.in_features
            || !decode_interpreter_nvfp4_gemv_supports(fused.combined_out_features, in_features)
        {
            return Ok(false);
        }
        self.validate_nvfp4_input_scale(qkv_weight, input_quantized.input_scale, qkv_input_scale)?;

        let LinearWeightBinding::Nvfp4 {
            input_scale: out_input_scale,
            ..
        } = &layer.out_proj
        else {
            return Ok(false);
        };
        let out_quantized = Nvfp4ActivationQuant {
            in_features: value_dim,
            input_scale: out_input_scale,
        };
        let Some(out_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.out_proj,
            out_quantized,
            forward.activation_fp4.ptr(),
            forward.activation_scale.ptr(),
            forward.block_out.ptr(),
        )?
        else {
            return Ok(false);
        };
        if out_proj.m != hidden || out_proj.k != value_dim {
            return Ok(false);
        }

        let Some(mlp) =
            self.decode_interpreter_mlp_params_from_common(&layer.common, forward, mlp_quantized)?
        else {
            return Ok(false);
        };

        let weights = self.cuda_weights()?;
        let deltanet_spec = DeltaNetDecodeSpec {
            layer_index: layer.layer_index,
            tokens_in_persistent_loop: 1,
            q_token_stride: 0,
            k_token_stride: 0,
            v_token_stride: 0,
            q_bf16: forward.aux.ptr(),
            k_bf16: forward.aux.ptr_at(key_dim * 2)?,
            v_bf16: forward.aux.ptr_at(key_dim * 4)?,
            state_bf16,
            conv_history_bf16,
            output_bf16: forward.aux3.ptr(),
            gate_f32: forward.gate_f32.ptr(),
            beta_f32: forward.beta_f32.ptr(),
            shape: DeltaNetShape {
                qk_heads: self.topology.linear_num_key_heads,
                v_heads: self.topology.linear_num_value_heads,
                key_dim: self.topology.linear_key_head_dim,
                value_dim: self.topology.linear_value_head_dim,
                conv_kernel: self.topology.linear_conv_kernel_dim,
            },
            state_decay: 1.0,
            update_scale: 1.0,
            qk_l2norm: true,
        };
        let spec_bytes = deltanet_decode_spec_abi_bytes(&deltanet_spec);
        if spec_bytes.len() > forward.interpreter_deltanet_spec.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear transformer layer DeltaNet spec needs {} bytes but buffer has {}",
                spec_bytes.len(),
                forward.interpreter_deltanet_spec.bytes()
            )));
        }
        forward
            .interpreter_deltanet_spec
            .copy_from_host(&spec_bytes)?;

        let compiled = DecodeInterpreterProgram::compile_linear_transformer_layer_decode(
            DecodeInterpreterLinearTransformerLayerParams {
                attention: DecodeInterpreterLinearAttentionInputLayerParams {
                    input_norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                        hidden,
                        eps: 1.0e-6,
                        input_tensor_scale_f32: self
                            .tensor_scalar_f32(weights, input_quantized.input_scale)?,
                        input_bf16: forward.hidden.ptr(),
                        weight_bf16: input_norm_weight_bf16,
                        residual_bf16: input_residual_bf16,
                        residual_out_bf16: forward.normed.ptr(),
                        output_bf16: DevicePtr::NULL,
                        output_fp4: forward.activation_fp4.ptr(),
                        output_scale_e4m3: forward.activation_scale.ptr(),
                        output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                    },
                    layer: DecodeInterpreterLinearAttentionLayerParams {
                        in_proj: DecodeInterpreterNvfp4GemvParams {
                            m: fused.combined_out_features,
                            k: in_features,
                            alpha: self.tensor_scalar_f32(weights, qkv_tensor_scale)?
                                * self.tensor_scalar_f32(weights, input_quantized.input_scale)?,
                            a_fp4: fused.combined_weight.ptr(),
                            a_scale_e4m3: fused.combined_block_scale.ptr(),
                            b_fp4: forward.activation_fp4.ptr(),
                            b_scale_e4m3: forward.activation_scale.ptr(),
                            c_bf16: forward.qkv.ptr(),
                        },
                        post_inproj: DecodeInterpreterLinearAttentionPostInProjParams {
                            conv_gdn: DecodeInterpreterConv1dGdnGateFusedParams {
                                channels: qkv_dim,
                                kernel_size: self.topology.linear_conv_kernel_dim,
                                conv_input_bf16: forward.qkv.ptr(),
                                conv_history_bf16,
                                conv_weight_bf16: self.tensor_ptr(weights, &layer.conv1d_weight)?,
                                conv_output_bf16: forward.aux.ptr(),
                                heads: self.topology.linear_num_value_heads,
                                gdn_a_bf16: a_bf16,
                                gdn_b_bf16: b_bf16,
                                gdn_a_log_bf16: self.tensor_ptr(weights, &layer.a_log)?,
                                gdn_dt_bias_bf16: self.tensor_ptr(weights, &layer.dt_bias)?,
                                gate_f32: forward.gate_f32.ptr(),
                                beta_f32: forward.beta_f32.ptr(),
                            },
                            deltanet: DecodeInterpreterDeltaNetParams {
                                spec: forward.interpreter_deltanet_spec.ptr(),
                            },
                            tail: DecodeInterpreterLinearAttentionTailParams {
                                norm: DecodeInterpreterRmsNormBf16Params {
                                    rows: self.topology.linear_num_value_heads,
                                    hidden: self.topology.linear_value_head_dim,
                                    eps: 1.0e-6,
                                    direct_weight: true,
                                    input_bf16: forward.aux3.ptr(),
                                    weight_bf16: self.tensor_ptr(weights, &layer.norm_weight)?,
                                    residual_bf16: DevicePtr::NULL,
                                    residual_out_bf16: DevicePtr::NULL,
                                    output_bf16: forward.aux2.ptr(),
                                },
                                swiglu: DecodeInterpreterSwiGluBf16Params {
                                    rows: 1,
                                    intermediate: value_dim,
                                    gate_bf16: z_bf16,
                                    up_bf16: forward.aux2.ptr(),
                                    output_bf16: forward.aux3.ptr(),
                                },
                                quant: DecodeInterpreterNvfp4QuantizeParams {
                                    values: value_dim,
                                    input_tensor_scale_f32: self
                                        .tensor_scalar_f32(weights, out_input_scale)?,
                                    input_bf16: forward.aux3.ptr(),
                                    output_fp4: forward.activation_fp4.ptr(),
                                    output_scale_e4m3: forward.activation_scale.ptr(),
                                    output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                                },
                                out_proj,
                            },
                        },
                    },
                },
                post: DecodeInterpreterNormMlpParams {
                    norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                        hidden,
                        eps: 1.0e-6,
                        input_tensor_scale_f32: self
                            .tensor_scalar_f32(weights, mlp_quantized.input_scale)?,
                        input_bf16: forward.block_out.ptr(),
                        weight_bf16: post_attention_norm_weight_bf16,
                        residual_bf16: forward.normed.ptr(),
                        residual_out_bf16: forward.residual.ptr(),
                        output_bf16: DevicePtr::NULL,
                        output_fp4: forward.activation_fp4.ptr(),
                        output_scale_e4m3: forward.activation_scale.ptr(),
                        output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                    },
                    mlp,
                },
            },
        );
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_deltanet_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear transformer layer program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_deltanet_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_deltanet_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear transformer layer program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_deltanet_counters.bytes()
            )));
        }

        forward
            .interpreter_deltanet_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_deltanet_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_deltanet_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_deltanet_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })?;
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn build_interpreter_linear_transformer_layer_decode(
        &self,
        layer: &LinearAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        input_quantized: Nvfp4ActivationQuant<'_>,
        mlp_quantized: Nvfp4ActivationQuant<'_>,
        input_residual_bf16: DevicePtr,
        input_norm_weight_bf16: DevicePtr,
        post_attention_norm_weight_bf16: DevicePtr,
        deltanet_spec_ptr: DevicePtr,
    ) -> Result<Option<(DecodeInterpreterProgram, Vec<u8>)>> {
        let Some(fused) = self.linear_attn_in_proj_fused_layer_opt(layer.layer_index) else {
            return Ok(None);
        };

        let qkv_dim = self.topology.linear_attention_qkv_dim();
        let key_dim = self.topology.linear_num_key_heads * self.topology.linear_key_head_dim;
        let value_dim = self.topology.linear_attention_value_dim();
        let hidden = self.topology.hidden_size;
        let layer_ordinal = self.linear_layer_ordinal(layer.layer_index)?;
        let conv_history_bf16 = runtime.conv_history.ptr_at(
            layer_ordinal * qkv_dim * self.topology.linear_conv_kernel_dim.saturating_sub(1) * 2,
        )?;
        let state_bf16 = runtime
            .deltanet_state
            .ptr_at(layer_ordinal * self.state.deltanet.state_bytes_per_layer as usize)?;

        let b_bf16 = forward
            .qkv
            .ptr()
            .offset_bytes(fused.b_offset * 2)
            .ok_or_else(|| CoreError::Runtime("fused DeltaNet b offset overflow".to_owned()))?;
        let a_bf16 = forward
            .qkv
            .ptr()
            .offset_bytes(fused.a_offset * 2)
            .ok_or_else(|| CoreError::Runtime("fused DeltaNet a offset overflow".to_owned()))?;
        let z_bf16 = forward
            .qkv
            .ptr()
            .offset_bytes(fused.z_offset * 2)
            .ok_or_else(|| CoreError::Runtime("fused DeltaNet z offset overflow".to_owned()))?;

        let LinearWeightBinding::Nvfp4 {
            weight: qkv_weight,
            tensor_scale: qkv_tensor_scale,
            input_scale: qkv_input_scale,
            ..
        } = &layer.in_proj_qkv
        else {
            return Ok(None);
        };
        let in_features = Self::nvfp4_in_features(qkv_weight)?;
        if in_features != input_quantized.in_features
            || !decode_interpreter_nvfp4_gemv_supports(fused.combined_out_features, in_features)
        {
            return Ok(None);
        }
        self.validate_nvfp4_input_scale(qkv_weight, input_quantized.input_scale, qkv_input_scale)?;

        let LinearWeightBinding::Nvfp4 {
            input_scale: out_input_scale,
            ..
        } = &layer.out_proj
        else {
            return Ok(None);
        };
        let out_quantized = Nvfp4ActivationQuant {
            in_features: value_dim,
            input_scale: out_input_scale,
        };
        let Some(out_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.out_proj,
            out_quantized,
            forward.activation_fp4.ptr(),
            forward.activation_scale.ptr(),
            forward.block_out.ptr(),
        )?
        else {
            return Ok(None);
        };
        if out_proj.m != hidden || out_proj.k != value_dim {
            return Ok(None);
        }

        let Some(mlp) =
            self.decode_interpreter_mlp_params_from_common(&layer.common, forward, mlp_quantized)?
        else {
            return Ok(None);
        };

        let weights = self.cuda_weights()?;
        let deltanet_spec = DeltaNetDecodeSpec {
            layer_index: layer.layer_index,
            tokens_in_persistent_loop: 1,
            q_token_stride: 0,
            k_token_stride: 0,
            v_token_stride: 0,
            q_bf16: forward.aux.ptr(),
            k_bf16: forward.aux.ptr_at(key_dim * 2)?,
            v_bf16: forward.aux.ptr_at(key_dim * 4)?,
            state_bf16,
            conv_history_bf16,
            output_bf16: forward.aux3.ptr(),
            gate_f32: forward.gate_f32.ptr(),
            beta_f32: forward.beta_f32.ptr(),
            shape: DeltaNetShape {
                qk_heads: self.topology.linear_num_key_heads,
                v_heads: self.topology.linear_num_value_heads,
                key_dim: self.topology.linear_key_head_dim,
                value_dim: self.topology.linear_value_head_dim,
                conv_kernel: self.topology.linear_conv_kernel_dim,
            },
            state_decay: 1.0,
            update_scale: 1.0,
            qk_l2norm: true,
        };
        let spec_bytes = deltanet_decode_spec_abi_bytes(&deltanet_spec);

        let program = DecodeInterpreterProgram::compile_linear_transformer_layer_decode(
            DecodeInterpreterLinearTransformerLayerParams {
                attention: DecodeInterpreterLinearAttentionInputLayerParams {
                    input_norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                        hidden,
                        eps: 1.0e-6,
                        input_tensor_scale_f32: self
                            .tensor_scalar_f32(weights, input_quantized.input_scale)?,
                        input_bf16: forward.hidden.ptr(),
                        weight_bf16: input_norm_weight_bf16,
                        residual_bf16: input_residual_bf16,
                        residual_out_bf16: forward.normed.ptr(),
                        output_bf16: DevicePtr::NULL,
                        output_fp4: forward.activation_fp4.ptr(),
                        output_scale_e4m3: forward.activation_scale.ptr(),
                        output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                    },
                    layer: DecodeInterpreterLinearAttentionLayerParams {
                        in_proj: DecodeInterpreterNvfp4GemvParams {
                            m: fused.combined_out_features,
                            k: in_features,
                            alpha: self.tensor_scalar_f32(weights, qkv_tensor_scale)?
                                * self.tensor_scalar_f32(weights, input_quantized.input_scale)?,
                            a_fp4: fused.combined_weight.ptr(),
                            a_scale_e4m3: fused.combined_block_scale.ptr(),
                            b_fp4: forward.activation_fp4.ptr(),
                            b_scale_e4m3: forward.activation_scale.ptr(),
                            c_bf16: forward.qkv.ptr(),
                        },
                        post_inproj: DecodeInterpreterLinearAttentionPostInProjParams {
                            conv_gdn: DecodeInterpreterConv1dGdnGateFusedParams {
                                channels: qkv_dim,
                                kernel_size: self.topology.linear_conv_kernel_dim,
                                conv_input_bf16: forward.qkv.ptr(),
                                conv_history_bf16,
                                conv_weight_bf16: self.tensor_ptr(weights, &layer.conv1d_weight)?,
                                conv_output_bf16: forward.aux.ptr(),
                                heads: self.topology.linear_num_value_heads,
                                gdn_a_bf16: a_bf16,
                                gdn_b_bf16: b_bf16,
                                gdn_a_log_bf16: self.tensor_ptr(weights, &layer.a_log)?,
                                gdn_dt_bias_bf16: self.tensor_ptr(weights, &layer.dt_bias)?,
                                gate_f32: forward.gate_f32.ptr(),
                                beta_f32: forward.beta_f32.ptr(),
                            },
                            deltanet: DecodeInterpreterDeltaNetParams {
                                spec: deltanet_spec_ptr,
                            },
                            tail: DecodeInterpreterLinearAttentionTailParams {
                                norm: DecodeInterpreterRmsNormBf16Params {
                                    rows: self.topology.linear_num_value_heads,
                                    hidden: self.topology.linear_value_head_dim,
                                    eps: 1.0e-6,
                                    direct_weight: true,
                                    input_bf16: forward.aux3.ptr(),
                                    weight_bf16: self.tensor_ptr(weights, &layer.norm_weight)?,
                                    residual_bf16: DevicePtr::NULL,
                                    residual_out_bf16: DevicePtr::NULL,
                                    output_bf16: forward.aux2.ptr(),
                                },
                                swiglu: DecodeInterpreterSwiGluBf16Params {
                                    rows: 1,
                                    intermediate: value_dim,
                                    gate_bf16: z_bf16,
                                    up_bf16: forward.aux2.ptr(),
                                    output_bf16: forward.aux3.ptr(),
                                },
                                quant: DecodeInterpreterNvfp4QuantizeParams {
                                    values: value_dim,
                                    input_tensor_scale_f32: self
                                        .tensor_scalar_f32(weights, out_input_scale)?,
                                    input_bf16: forward.aux3.ptr(),
                                    output_fp4: forward.activation_fp4.ptr(),
                                    output_scale_e4m3: forward.activation_scale.ptr(),
                                    output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                                },
                                out_proj,
                            },
                        },
                    },
                },
                post: DecodeInterpreterNormMlpParams {
                    norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                        hidden,
                        eps: 1.0e-6,
                        input_tensor_scale_f32: self
                            .tensor_scalar_f32(weights, mlp_quantized.input_scale)?,
                        input_bf16: forward.block_out.ptr(),
                        weight_bf16: post_attention_norm_weight_bf16,
                        residual_bf16: forward.normed.ptr(),
                        residual_out_bf16: forward.residual.ptr(),
                        output_bf16: DevicePtr::NULL,
                        output_fp4: forward.activation_fp4.ptr(),
                        output_scale_e4m3: forward.activation_scale.ptr(),
                        output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                    },
                    mlp,
                },
            },
        );
        Ok(Some((program, spec_bytes)))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn run_interpreter_linear_attention_post_inproj_decode(
        &self,
        layer: &LinearAttentionLayerWeights,
        forward: &GpuForwardBuffers,
        conv_input_bf16: DevicePtr,
        conv_history_bf16: DevicePtr,
        gdn_a_bf16: DevicePtr,
        gdn_b_bf16: DevicePtr,
        z_bf16: DevicePtr,
        deltanet_spec: &DeltaNetDecodeSpec,
    ) -> Result<bool> {
        if z_bf16 == conv_input_bf16 {
            return Ok(false);
        }
        if !decode_interpreter_linear_attention_post_inproj_enabled(
            self.decode_interpreter_decode_enabled(),
        ) || deltanet_spec.tokens_in_persistent_loop != 1
            || deltanet_spec.q_token_stride != 0
            || deltanet_spec.k_token_stride != 0
            || deltanet_spec.v_token_stride != 0
        {
            return Ok(false);
        }

        let value_dim = self.topology.linear_attention_value_dim();
        let hidden = self.topology.hidden_size;
        let LinearWeightBinding::Nvfp4 {
            input_scale: out_input_scale,
            ..
        } = &layer.out_proj
        else {
            return Ok(false);
        };
        let out_quantized = Nvfp4ActivationQuant {
            in_features: value_dim,
            input_scale: out_input_scale,
        };
        let Some(out_proj) = self.decode_interpreter_nvfp4_gemv_params(
            &layer.out_proj,
            out_quantized,
            forward.activation_fp4.ptr(),
            forward.activation_scale.ptr(),
            forward.block_out.ptr(),
        )?
        else {
            return Ok(false);
        };
        if out_proj.m != hidden || out_proj.k != value_dim {
            return Ok(false);
        }

        let spec_bytes = deltanet_decode_spec_abi_bytes(deltanet_spec);
        if spec_bytes.len() > forward.interpreter_deltanet_spec.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear-attn post-inproj DeltaNet spec needs {} bytes but buffer has {}",
                spec_bytes.len(),
                forward.interpreter_deltanet_spec.bytes()
            )));
        }
        forward
            .interpreter_deltanet_spec
            .copy_from_host(&spec_bytes)?;

        let weights = self.cuda_weights()?;
        let compiled = DecodeInterpreterProgram::compile_linear_attention_post_inproj_decode(
            DecodeInterpreterLinearAttentionPostInProjParams {
                conv_gdn: DecodeInterpreterConv1dGdnGateFusedParams {
                    channels: self.topology.linear_attention_qkv_dim(),
                    kernel_size: self.topology.linear_conv_kernel_dim,
                    conv_input_bf16,
                    conv_history_bf16,
                    conv_weight_bf16: self.tensor_ptr(weights, &layer.conv1d_weight)?,
                    conv_output_bf16: forward.aux.ptr(),
                    heads: self.topology.linear_num_value_heads,
                    gdn_a_bf16,
                    gdn_b_bf16,
                    gdn_a_log_bf16: self.tensor_ptr(weights, &layer.a_log)?,
                    gdn_dt_bias_bf16: self.tensor_ptr(weights, &layer.dt_bias)?,
                    gate_f32: forward.gate_f32.ptr(),
                    beta_f32: forward.beta_f32.ptr(),
                },
                deltanet: DecodeInterpreterDeltaNetParams {
                    spec: forward.interpreter_deltanet_spec.ptr(),
                },
                tail: DecodeInterpreterLinearAttentionTailParams {
                    norm: DecodeInterpreterRmsNormBf16Params {
                        rows: self.topology.linear_num_value_heads,
                        hidden: self.topology.linear_value_head_dim,
                        eps: 1.0e-6,
                        direct_weight: true,
                        input_bf16: forward.aux3.ptr(),
                        weight_bf16: self.tensor_ptr(weights, &layer.norm_weight)?,
                        residual_bf16: DevicePtr::NULL,
                        residual_out_bf16: DevicePtr::NULL,
                        output_bf16: forward.aux2.ptr(),
                    },
                    swiglu: DecodeInterpreterSwiGluBf16Params {
                        rows: 1,
                        intermediate: value_dim,
                        gate_bf16: z_bf16,
                        up_bf16: forward.aux2.ptr(),
                        output_bf16: forward.aux3.ptr(),
                    },
                    quant: DecodeInterpreterNvfp4QuantizeParams {
                        values: value_dim,
                        input_tensor_scale_f32: self.tensor_scalar_f32(weights, out_input_scale)?,
                        input_bf16: forward.aux3.ptr(),
                        output_fp4: forward.activation_fp4.ptr(),
                        output_scale_e4m3: forward.activation_scale.ptr(),
                        output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                    },
                    out_proj,
                },
            },
        );
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_deltanet_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear-attn post-inproj program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_deltanet_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_deltanet_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter linear-attn post-inproj program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_deltanet_counters.bytes()
            )));
        }

        forward
            .interpreter_deltanet_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_deltanet_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_deltanet_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_deltanet_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })?;
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn decode_interpreter_mlp_params(
        &self,
        layer: &LayerWeights,
        forward: &GpuForwardBuffers,
        quantized: Nvfp4ActivationQuant<'_>,
    ) -> Result<Option<DecodeInterpreterMlpParams>> {
        self.decode_interpreter_mlp_params_from_common(layer_common(layer), forward, quantized)
    }

    #[cfg(feature = "cuda")]
    fn decode_interpreter_mlp_params_from_common(
        &self,
        common: &CommonLayerWeights,
        forward: &GpuForwardBuffers,
        quantized: Nvfp4ActivationQuant<'_>,
    ) -> Result<Option<DecodeInterpreterMlpParams>> {
        let weights = self.cuda_weights()?;
        let intermediate = self.topology.intermediate_size;
        let hidden = self.topology.hidden_size;
        let LinearWeightBinding::Nvfp4 {
            weight: gate_weight,
            block_scale: gate_block_scale,
            tensor_scale: gate_tensor_scale,
            input_scale: gate_input_scale,
        } = &common.mlp_gate_proj
        else {
            return Ok(None);
        };
        let LinearWeightBinding::Nvfp4 {
            weight: up_weight,
            block_scale: up_block_scale,
            tensor_scale: up_tensor_scale,
            input_scale: up_input_scale,
        } = &common.mlp_up_proj
        else {
            return Ok(None);
        };
        let LinearWeightBinding::Nvfp4 {
            weight: down_weight,
            block_scale: down_block_scale,
            tensor_scale: down_tensor_scale,
            input_scale: down_input_scale,
        } = &common.mlp_down_proj
        else {
            return Ok(None);
        };

        if !decode_interpreter_mlp_supports(hidden, intermediate) {
            return Ok(None);
        }
        if Self::nvfp4_in_features(gate_weight)? != quantized.in_features
            || Self::nvfp4_in_features(up_weight)? != quantized.in_features
            || Self::nvfp4_in_features(down_weight)? != intermediate
        {
            return Ok(None);
        }
        let gate_out = *gate_weight.shape.first().ok_or_else(|| {
            CoreError::Runtime(format!("tensor {} has empty shape", gate_weight.name))
        })?;
        let up_out = *up_weight.shape.first().ok_or_else(|| {
            CoreError::Runtime(format!("tensor {} has empty shape", up_weight.name))
        })?;
        let down_out = *down_weight.shape.first().ok_or_else(|| {
            CoreError::Runtime(format!("tensor {} has empty shape", down_weight.name))
        })?;
        if gate_out != intermediate || up_out != intermediate || down_out != hidden {
            return Ok(None);
        }

        self.validate_nvfp4_input_scale(gate_weight, quantized.input_scale, gate_input_scale)?;
        self.validate_nvfp4_input_scale(up_weight, quantized.input_scale, up_input_scale)?;

        let gate_alpha = self.tensor_scalar_f32(weights, gate_tensor_scale)?
            * self.tensor_scalar_f32(weights, quantized.input_scale)?;
        let up_alpha = self.tensor_scalar_f32(weights, up_tensor_scale)?
            * self.tensor_scalar_f32(weights, quantized.input_scale)?;
        let down_input_tensor_scale = self.tensor_scalar_f32(weights, down_input_scale)?;
        let down_alpha =
            self.tensor_scalar_f32(weights, down_tensor_scale)? * down_input_tensor_scale;
        let (chunk_accum_f32, chunk_intermediate) = if decode_interpreter_mlp_chunked_enabled()
            && forward.attn_partial_acc.bytes() >= hidden * size_of::<f32>()
            && intermediate % 32 == 0
        {
            (forward.attn_partial_acc.ptr(), intermediate / 2)
        } else {
            (DevicePtr::NULL, 0)
        };

        Ok(Some(DecodeInterpreterMlpParams {
            hidden,
            intermediate,
            input_fp4: forward.activation_fp4.ptr(),
            input_scale_e4m3: forward.activation_scale.ptr(),
            gate_weight_fp4: self.tensor_ptr(weights, gate_weight)?,
            gate_weight_scale_e4m3: self.tensor_ptr(weights, gate_block_scale)?,
            gate_alpha,
            gate_out_bf16: forward.aux.ptr(),
            up_weight_fp4: self.tensor_ptr(weights, up_weight)?,
            up_weight_scale_e4m3: self.tensor_ptr(weights, up_block_scale)?,
            up_alpha,
            up_out_bf16: forward.aux2.ptr(),
            swiglu_fp4: forward.activation_fp4.ptr(),
            swiglu_scale_e4m3: forward.activation_scale.ptr(),
            swiglu_tensor_scale_f32: forward.activation_scale_2.ptr(),
            down_input_tensor_scale,
            down_weight_fp4: self.tensor_ptr(weights, down_weight)?,
            down_weight_scale_e4m3: self.tensor_ptr(weights, down_block_scale)?,
            down_alpha,
            output_bf16: forward.hidden.ptr(),
            chunk_accum_f32,
            chunk_intermediate,
        }))
    }

    #[cfg(feature = "cuda")]
    fn run_interpreter_mlp_with_quantized_input(
        &self,
        layer: &LayerWeights,
        forward: &GpuForwardBuffers,
        quantized: Nvfp4ActivationQuant<'_>,
    ) -> Result<bool> {
        let Some(params) = self.decode_interpreter_mlp_params(layer, forward, quantized)? else {
            return Ok(false);
        };
        let compiled = DecodeInterpreterProgram::compile_mlp(params);
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_mlp_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter MLP program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_mlp_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_mlp_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter MLP program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_mlp_counters.bytes()
            )));
        }

        forward
            .interpreter_mlp_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_mlp_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_mlp_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_mlp_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })?;
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn run_interpreter_norm_mlp(
        &self,
        layer: &LayerWeights,
        forward: &GpuForwardBuffers,
        quantized: Nvfp4ActivationQuant<'_>,
        hidden: usize,
        input_bf16: DevicePtr,
        weight_bf16: DevicePtr,
        residual_bf16: DevicePtr,
        residual_out_bf16: DevicePtr,
        output_bf16: DevicePtr,
        input_scale: &TensorInfo,
    ) -> Result<bool> {
        let Some(mlp) = self.decode_interpreter_mlp_params(layer, forward, quantized)? else {
            return Ok(false);
        };
        let input_tensor_scale_f32 = self.tensor_scalar_f32(self.cuda_weights()?, input_scale)?;
        let compiled =
            DecodeInterpreterProgram::compile_rmsnorm_mlp(DecodeInterpreterNormMlpParams {
                norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                    hidden,
                    eps: 1.0e-6,
                    input_tensor_scale_f32,
                    input_bf16,
                    weight_bf16,
                    residual_bf16,
                    residual_out_bf16,
                    output_bf16,
                    output_fp4: forward.activation_fp4.ptr(),
                    output_scale_e4m3: forward.activation_scale.ptr(),
                    output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                },
                mlp,
            });
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_mlp_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter RMSNorm+MLP program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_mlp_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_mlp_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter RMSNorm+MLP program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_mlp_counters.bytes()
            )));
        }

        forward
            .interpreter_mlp_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_mlp_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_mlp_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_mlp_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })?;
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    fn run_interpreter_final_logits(
        &self,
        manifest: &ModelWeightsManifest,
        weights: &GpuWeightStore,
        forward: &GpuForwardBuffers,
    ) -> Result<()> {
        let vocab_size = *manifest.lm_head.shape.first().ok_or_else(|| {
            CoreError::Runtime(format!("tensor {} has empty shape", manifest.lm_head.name))
        })?;
        let hidden = *manifest.lm_head.shape.get(1).ok_or_else(|| {
            CoreError::Runtime(format!("tensor {} is not a matrix", manifest.lm_head.name))
        })?;
        if hidden != self.topology.hidden_size {
            return Err(CoreError::Runtime(format!(
                "interpreter logits expected lm_head in_features {} but {} has {hidden}",
                self.topology.hidden_size, manifest.lm_head.name
            )));
        }

        let compiled =
            DecodeInterpreterProgram::compile_final_logits(DecodeInterpreterLogitsParams {
                hidden,
                vocab_size,
                hidden_bf16: forward.hidden.ptr(),
                residual_bf16: forward.residual.ptr(),
                final_norm_weight_bf16: self.tensor_ptr(weights, &manifest.final_norm)?,
                normed_bf16: forward.normed.ptr(),
                activation_fp4: forward.activation_fp4.ptr(),
                activation_scale_e4m3: forward.activation_scale.ptr(),
                activation_tensor_scale_f32: forward.activation_scale_2.ptr(),
                lm_head_weight_bf16: self.tensor_ptr(weights, &manifest.lm_head)?,
                logits_bf16: forward.logits.ptr(),
            });
        let instruction_bytes = instructions_as_bytes(&compiled.program.instructions);
        if instruction_bytes.len() > forward.interpreter_logits_instructions.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter logits program needs {} instruction bytes but buffer has {}",
                instruction_bytes.len(),
                forward.interpreter_logits_instructions.bytes()
            )));
        }
        let counter_bytes = compiled.program.counter_count * size_of::<i32>();
        if counter_bytes > forward.interpreter_logits_counters.bytes() {
            return Err(CoreError::Runtime(format!(
                "interpreter logits program needs {counter_bytes} counter bytes but buffer has {}",
                forward.interpreter_logits_counters.bytes()
            )));
        }

        forward
            .interpreter_logits_instructions
            .copy_from_host(instruction_bytes)?;
        forward.interpreter_logits_counters.memset_async(0)?;
        self.backend
            .interpreter_decode_sm120(&InterpreterProgramSpec {
                instructions: forward.interpreter_logits_instructions.ptr(),
                instruction_count: compiled.program.instructions.len(),
                counters_i32: forward.interpreter_logits_counters.ptr(),
                counter_count: compiled.program.counter_count,
                cta_count: 0,
                flags: interpreter_launch_flags(),
            })
    }

    #[cfg(feature = "cuda")]
    fn bf16_matvec(&self, weight: &TensorInfo, input: DevicePtr, output: DevicePtr) -> Result<()> {
        let out_features = *weight
            .shape
            .first()
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} has empty shape", weight.name)))?;
        let in_features = *weight
            .shape
            .get(1)
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} is not a matrix", weight.name)))?;
        if let Some(fp8) = &self.lm_head_fp8 {
            if weight.name == fp8.tensor_name {
                return self.backend.fp8_matvec(&Fp8MatVecSpec {
                    out_features,
                    in_features,
                    rows: 1,
                    input_stride: in_features,
                    weight_e4m3: fp8.weight_e4m3.ptr(),
                    row_scale_f32: fp8.row_scale_f32.ptr(),
                    input_bf16: input,
                    output_bf16: output,
                });
            }
        }
        let runtime = self.cuda_runtime()?;
        let workspace = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr())
            .unwrap_or(DevicePtr::NULL);
        let workspace_bytes = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.bytes())
            .unwrap_or(0);
        let input_bf16 = input;
        let weight_bf16 = self.tensor_ptr(self.cuda_weights()?, weight)?;
        let gemm_result = self.backend.bf16_gemm(&Bf16GemmSpec {
            m: out_features,
            n: 1,
            k: in_features,
            a_bf16: weight_bf16,
            b_bf16: input_bf16,
            c_bf16: output,
            workspace,
            workspace_bytes,
        });
        if gemm_result.is_ok() {
            return gemm_result;
        }

        self.backend.bf16_matvec(&Bf16MatVecSpec {
            out_features,
            in_features,
            input_bf16,
            weight_bf16,
            output_bf16: output,
        })
    }

    #[cfg(feature = "cuda")]
    fn bf16_gemm_rows(
        &self,
        weight: &TensorInfo,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        let out_features = *weight
            .shape
            .first()
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} has empty shape", weight.name)))?;
        let in_features = *weight
            .shape
            .get(1)
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} is not a matrix", weight.name)))?;
        if let Some(fp8) = &self.lm_head_fp8 {
            if weight.name == fp8.tensor_name {
                return self.backend.fp8_matvec(&Fp8MatVecSpec {
                    out_features,
                    in_features,
                    rows,
                    input_stride: in_features,
                    weight_e4m3: fp8.weight_e4m3.ptr(),
                    row_scale_f32: fp8.row_scale_f32.ptr(),
                    input_bf16: input,
                    output_bf16: output,
                });
            }
        }
        let runtime = self.cuda_runtime()?;
        let workspace = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.ptr())
            .unwrap_or(DevicePtr::NULL);
        let workspace_bytes = runtime
            .workspace
            .as_ref()
            .map(|buffer| buffer.bytes())
            .unwrap_or(0);
        let weight_bf16 = self.tensor_ptr(self.cuda_weights()?, weight)?;
        let gemm_result = self.backend.bf16_gemm(&Bf16GemmSpec {
            m: out_features,
            n: rows,
            k: in_features,
            a_bf16: weight_bf16,
            b_bf16: input,
            c_bf16: output,
            workspace,
            workspace_bytes,
        });
        if gemm_result.is_ok() || rows > 1 {
            return gemm_result;
        }
        self.backend.bf16_matvec(&Bf16MatVecSpec {
            out_features,
            in_features,
            input_bf16: input,
            weight_bf16,
            output_bf16: output,
        })
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm(
        &self,
        rows: usize,
        hidden: usize,
        input: DevicePtr,
        weight: DevicePtr,
        residual: DevicePtr,
        residual_out: DevicePtr,
        output: DevicePtr,
    ) -> Result<()> {
        self.backend.rmsnorm(&RmsNormSpec {
            rows,
            hidden,
            eps: 1.0e-6,
            input_bf16: input,
            weight_bf16: weight,
            residual_bf16: residual,
            residual_out_bf16: residual_out,
            output_bf16: output,
            direct_weight: false,
        })
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_direct_weight(
        &self,
        rows: usize,
        hidden: usize,
        input: DevicePtr,
        weight: DevicePtr,
        residual: DevicePtr,
        residual_out: DevicePtr,
        output: DevicePtr,
    ) -> Result<()> {
        self.backend.rmsnorm(&RmsNormSpec {
            rows,
            hidden,
            eps: 1.0e-6,
            input_bf16: input,
            weight_bf16: weight,
            residual_bf16: residual,
            residual_out_bf16: residual_out,
            output_bf16: output,
            direct_weight: true,
        })
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_nvfp4_quantize(
        &self,
        hidden: usize,
        input: DevicePtr,
        weight: DevicePtr,
        residual: DevicePtr,
        residual_out: DevicePtr,
        output_bf16: DevicePtr,
        input_scale: &TensorInfo,
        interpreter_allowed: bool,
    ) -> Result<()> {
        let forward = self.cuda_forward()?;
        let input_tensor_scale_f32 = self.tensor_scalar_f32(self.cuda_weights()?, input_scale)?;
        if interpreter_allowed
            && decode_interpreter_rmsnorm_enabled(self.decode_interpreter_decode_enabled())
        {
            return self.run_interpreter_rmsnorm_nvfp4_quantize(
                DecodeInterpreterRmsNormNvfp4QuantParams {
                    hidden,
                    eps: 1.0e-6,
                    input_tensor_scale_f32,
                    input_bf16: input,
                    weight_bf16: weight,
                    residual_bf16: residual,
                    residual_out_bf16: residual_out,
                    output_bf16,
                    output_fp4: forward.activation_fp4.ptr(),
                    output_scale_e4m3: forward.activation_scale.ptr(),
                    output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                },
                forward,
            );
        }
        self.backend
            .rmsnorm_nvfp4_quantize(&RmsNormNvfp4QuantizeSpec {
                hidden,
                eps: 1.0e-6,
                input_bf16: input,
                weight_bf16: weight,
                residual_bf16: residual,
                residual_out_bf16: residual_out,
                output_bf16,
                output_fp4: forward.activation_fp4.ptr(),
                output_scale_e4m3: forward.activation_scale.ptr(),
                output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                input_tensor_scale_f32,
            })
    }

    #[cfg(feature = "cuda")]
    fn tensor_ptr(&self, weights: &GpuWeightStore, info: &TensorInfo) -> Result<DevicePtr> {
        weights
            .tensor(&info.name)
            .map(|tensor| tensor.ptr())
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} was not uploaded", info.name)))
    }

    #[cfg(feature = "cuda")]
    fn tensor_scalar_f32(&self, weights: &GpuWeightStore, info: &TensorInfo) -> Result<f32> {
        weights.scalar_f32(&info.name).ok_or_else(|| {
            CoreError::Runtime(format!(
                "tensor {} was not uploaded as a scalar f32",
                info.name
            ))
        })
    }

    #[cfg(feature = "cuda")]
    fn ptr_offset(ptr: DevicePtr, offset_bytes: usize) -> Result<DevicePtr> {
        ptr.offset_bytes(offset_bytes)
            .ok_or_else(|| CoreError::Runtime("CUDA pointer offset overflow".to_owned()))
    }

    #[cfg(feature = "cuda")]
    fn attention_kv_cache_dtype_code(dtype: KvCacheDtype) -> Result<i32> {
        match dtype {
            KvCacheDtype::Bf16 => Ok(0),
            KvCacheDtype::Fp8 => Ok(1),
            KvCacheDtype::TurboQuant3 => Ok(2),
            KvCacheDtype::TurboQuant35 => Ok(3),
        }
    }

    /// Pick the split tile size for full-attention decode/prefill. 512 keeps
    /// launch/reduce overhead down at short contexts; 128 starts paying off
    /// around 1K context for MTP short-chunk attention; 64 exposes more
    /// T-axis parallelism once the graph shape reaches ~2K context on the 5090.
    #[cfg(feature = "cuda")]
    fn attention_split_timesteps_per_block_for(&self, context: usize) -> usize {
        if let Some(value) = cuda_env_usize("QWEN36_ATTENTION_SPLIT_TIMESTEPS") {
            return value.max(ATTN_MIN_SPLIT_TIMESTEPS_PER_BLOCK);
        }
        if context >= 2048 {
            ATTN_MIN_SPLIT_TIMESTEPS_PER_BLOCK
        } else if context >= 1024 {
            128
        } else {
            512
        }
    }

    #[cfg(feature = "cuda")]
    fn decode_attention_context_limit_for_position(&self, position: usize) -> usize {
        self.decode_attention_context_limit_for_active_context(position.saturating_add(1))
    }

    #[cfg(feature = "cuda")]
    fn decode_attention_context_limit_for_active_context(&self, active_context: usize) -> usize {
        if cuda_env_bool("QWEN36_DECODE_ATTENTION_BUCKET_DISABLE") {
            return self.config.max_context.max(1);
        }

        let max_context = self.config.max_context.max(1);
        let active_context = active_context.clamp(1, max_context);
        let bucket_floor = cuda_decode_attention_bucket_min_context().min(max_context);
        let wanted = active_context.max(bucket_floor);
        wanted
            .checked_next_power_of_two()
            .unwrap_or(max_context)
            .min(max_context)
            .max(active_context)
    }

    #[cfg(feature = "cuda")]
    fn graph_attention_context_limit(&self, kind: DecodeGraphKind) -> usize {
        match kind {
            DecodeGraphKind::Decode | DecodeGraphKind::MtpDecodeOne => {
                self.decode_attention_context_limit_for_position(self.state.position)
            }
            DecodeGraphKind::MtpVerifyOne => {
                self.decode_attention_context_limit_for_active_context(self.state.position + 2)
            }
            DecodeGraphKind::MtpVerifyMulti { drafts, .. } => self
                .decode_attention_context_limit_for_active_context(
                    self.state.position + drafts + 1,
                ),
            DecodeGraphKind::MtpRecover { committed, drafts } => self
                .decode_attention_context_limit_for_active_context(
                    self.state.position + committed + drafts,
                ),
        }
    }

    #[cfg(feature = "cuda")]
    fn decode_interpreter_graph_layer_program(
        &self,
        layer_idx: usize,
        position_device_i32: DevicePtr,
    ) -> Option<&DecodeInterpreterGraphLayerProgram> {
        let programs = self.decode_interpreter_graph_programs.as_ref()?;
        if programs.position_device_i32 != position_device_i32 {
            return None;
        }
        programs.layers.get(layer_idx)?.as_ref()
    }

    #[cfg(feature = "cuda")]
    fn prepare_decode_interpreter_graph_programs(
        &mut self,
        position_device_i32: DevicePtr,
        attention_context_limit: usize,
    ) -> Result<()> {
        let interpreter_master_enabled = self.decode_interpreter_decode_enabled();
        if position_device_i32 == DevicePtr::NULL || !interpreter_master_enabled {
            self.decode_interpreter_graph_programs = None;
            return Ok(());
        }
        if self
            .decode_interpreter_graph_programs
            .as_ref()
            .is_some_and(|programs| {
                programs.position_device_i32 == position_device_i32
                    && programs.attention_context_limit == attention_context_limit
            })
        {
            return Ok(());
        }

        let layers = {
            let manifest = self
                .weights
                .as_ref()
                .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
            let weights = self.cuda_weights()?;
            let runtime = self.cuda_runtime()?;
            let forward = self.cuda_forward()?;
            let mut layers = Vec::with_capacity(manifest.layers.len());
            for (layer_idx, layer) in manifest.layers.iter().enumerate() {
                let input_quantized = match Self::layer_input_nvfp4_quant(layer)? {
                    Some(quantized) => quantized,
                    None => {
                        layers.push(None);
                        continue;
                    }
                };
                let common = layer_common(layer);
                let mlp_input_linears = [
                    (&common.mlp_gate_proj, DevicePtr::NULL),
                    (&common.mlp_up_proj, DevicePtr::NULL),
                ];
                let mlp_quantized = match Self::common_nvfp4_quant(&mlp_input_linears)? {
                    Some(quantized) => quantized,
                    None => {
                        layers.push(None);
                        continue;
                    }
                };
                let input_residual = if layer_idx == 0 {
                    DevicePtr::NULL
                } else {
                    forward.residual.ptr()
                };

                let program = match layer {
                    LayerWeights::LinearAttention(linear)
                        if decode_interpreter_linear_transformer_layer_enabled(
                            interpreter_master_enabled,
                        ) =>
                    {
                        let spec = CudaDeviceBuffer::alloc(deltanet_decode_spec_abi_size())?;
                        match self.build_interpreter_linear_transformer_layer_decode(
                            linear,
                            runtime,
                            forward,
                            input_quantized,
                            mlp_quantized,
                            input_residual,
                            self.tensor_ptr(weights, &common.input_layernorm)?,
                            self.tensor_ptr(weights, &common.post_attention_layernorm)?,
                            spec.ptr(),
                        )? {
                            Some((program, spec_bytes)) => {
                                spec.copy_from_host(&spec_bytes)?;
                                Some(DecodeInterpreterGraphLayerProgram::upload(
                                    program,
                                    Some(spec),
                                )?)
                            }
                            None => None,
                        }
                    }
                    LayerWeights::FullAttention(full)
                        if decode_interpreter_full_transformer_layer_enabled(
                            interpreter_master_enabled,
                        ) =>
                    {
                        let spec = CudaDeviceBuffer::alloc(attention_decode_spec_abi_size())?;
                        match self.build_interpreter_full_transformer_layer_decode(
                            full,
                            runtime,
                            forward,
                            self.state.position,
                            position_device_i32,
                            attention_context_limit,
                            input_quantized,
                            mlp_quantized,
                            input_residual,
                            self.tensor_ptr(weights, &common.input_layernorm)?,
                            self.tensor_ptr(weights, &common.post_attention_layernorm)?,
                            spec.ptr(),
                        )? {
                            Some((program, spec_bytes)) => {
                                spec.copy_from_host(&spec_bytes)?;
                                Some(DecodeInterpreterGraphLayerProgram::upload(
                                    program,
                                    Some(spec),
                                )?)
                            }
                            None => None,
                        }
                    }
                    _ => None,
                };
                layers.push(program);
            }
            layers
        };

        self.decode_interpreter_graph_programs = Some(DecodeInterpreterGraphPrograms {
            attention_context_limit,
            position_device_i32,
            layers,
        });
        Ok(())
    }

    /// Pick the number of split-KV blocks per q-head for decode attention.
    /// Sized from a context bucket instead of the configured `max_context`.
    /// This keeps CUDA graph launch shapes stable inside the bucket while
    /// avoiding thousands of empty split-KV blocks when a run reserves a very
    /// large context window but is currently decoding around 8K-32K tokens.
    #[cfg(feature = "cuda")]
    fn decode_attention_n_splits_for_context_limit(&self, context_limit: usize) -> usize {
        if cuda_env_bool("QWEN36_ATTENTION_SPLIT_DISABLE") {
            return 0;
        }
        if let Some(value) = cuda_env_usize("QWEN36_DECODE_ATTENTION_N_SPLITS") {
            return value;
        }
        let n_splits = context_limit
            .max(1)
            .div_ceil(self.attention_split_timesteps_per_block_for(context_limit));
        if n_splits >= 2 { n_splits } else { 0 }
    }

    #[cfg(feature = "cuda")]
    fn prefill_attention_n_splits(
        &self,
        context: usize,
        start_position_device_i32: DevicePtr,
    ) -> usize {
        if start_position_device_i32 != DevicePtr::NULL {
            let context_limit = self.decode_attention_context_limit_for_active_context(context);
            return self.decode_attention_n_splits_for_context_limit(context_limit);
        }
        if cuda_env_bool("QWEN36_ATTENTION_SPLIT_DISABLE") {
            return 0;
        }
        if let Some(value) = cuda_env_usize("QWEN36_PREFILL_ATTENTION_N_SPLITS") {
            return value;
        }
        let n_splits = context.div_ceil(self.attention_split_timesteps_per_block_for(context));
        if n_splits >= 2 { n_splits } else { 0 }
    }

    #[cfg(feature = "cuda")]
    fn mtp_kv_cache_plane_bytes(&self) -> Result<usize> {
        let values = self
            .config
            .max_context
            .checked_mul(self.topology.attention_num_kv_heads)
            .and_then(|value| value.checked_mul(self.topology.attention_head_dim))
            .ok_or_else(|| CoreError::Runtime("MTP KV cache size overflow".to_owned()))?;
        values
            .checked_mul(2)
            .ok_or_else(|| CoreError::Runtime("MTP KV cache byte size overflow".to_owned()))
    }

    #[cfg(feature = "cuda")]
    fn linear_layer_ordinal(&self, layer_index: usize) -> Result<usize> {
        self.topology
            .linear_attention_layers()
            .into_iter()
            .position(|idx| idx == layer_index)
            .ok_or_else(|| {
                CoreError::Runtime(format!("layer {layer_index} is not linear attention"))
            })
    }

    /// Dump `count` FP32 values to disk for the parity harness.
    #[cfg(feature = "cuda")]
    fn dump_f32_to_disk(&self, dir: &str, name: &str, src: DevicePtr, count: usize) -> Result<()> {
        unsafe extern "C" {
            fn qwen36_cuda_memcpy_d2h(dst: *mut core::ffi::c_void, src: u64, bytes: usize) -> i32;
        }
        qwen36_fp4_kernels::cuda_synchronize()?;
        let bytes = count * 4;
        let mut buf = vec![0u8; bytes];
        let status = unsafe { qwen36_cuda_memcpy_d2h(buf.as_mut_ptr() as *mut _, src.0, bytes) };
        if status != 0 {
            return Err(CoreError::Runtime(format!(
                "dump_f32_to_disk memcpy failed: status {status}"
            )));
        }
        let path = std::path::Path::new(dir).join(name);
        std::fs::write(&path, &buf).map_err(|e| {
            CoreError::Runtime(format!("dump_f32_to_disk write {}: {e}", path.display()))
        })?;
        Ok(())
    }

    /// Dump raw bytes from a device buffer. Used for packed FP4 activations
    /// and UE4M3 scale bytes while debugging fused quantization.
    #[cfg(feature = "cuda")]
    fn dump_bytes_to_disk(
        &self,
        dir: &str,
        name: &str,
        src: DevicePtr,
        bytes: usize,
    ) -> Result<()> {
        unsafe extern "C" {
            fn qwen36_cuda_memcpy_d2h(dst: *mut core::ffi::c_void, src: u64, bytes: usize) -> i32;
        }
        qwen36_fp4_kernels::cuda_synchronize()?;
        let mut buf = vec![0u8; bytes];
        let status = unsafe { qwen36_cuda_memcpy_d2h(buf.as_mut_ptr() as *mut _, src.0, bytes) };
        if status != 0 {
            return Err(CoreError::Runtime(format!(
                "dump_bytes_to_disk memcpy failed: status {status}"
            )));
        }
        let path = std::path::Path::new(dir).join(name);
        std::fs::write(&path, &buf).map_err(|e| {
            CoreError::Runtime(format!("dump_bytes_to_disk write {}: {e}", path.display()))
        })?;
        Ok(())
    }

    /// Dump `count` BF16 values from a device buffer to disk as raw little-
    /// endian bytes. Used by the parity harness to compare intermediate
    /// activations against a Python ground-truth.
    #[cfg(feature = "cuda")]
    fn dump_buffer_to_disk(
        &self,
        dir: &str,
        name: &str,
        src: DevicePtr,
        count: usize,
    ) -> Result<()> {
        unsafe extern "C" {
            fn qwen36_cuda_memcpy_d2h(dst: *mut core::ffi::c_void, src: u64, bytes: usize) -> i32;
        }
        qwen36_fp4_kernels::cuda_synchronize()?;
        let bytes = count * 2;
        let mut buf = vec![0u8; bytes];
        let status = unsafe { qwen36_cuda_memcpy_d2h(buf.as_mut_ptr() as *mut _, src.0, bytes) };
        if status != 0 {
            return Err(CoreError::Runtime(format!(
                "dump_buffer_to_disk memcpy failed: status {status}"
            )));
        }
        let path = std::path::Path::new(dir).join(name);
        std::fs::write(&path, &buf).map_err(|e| {
            CoreError::Runtime(format!("dump_buffer_to_disk write {}: {e}", path.display()))
        })?;
        Ok(())
    }

    /// Numerical-parity helper: copy `count` BF16 values from `src` to host
    /// and print min / max / mean-abs / NaN+Inf counts. Guarded by the
    /// QWEN36_DEBUG_LAYER_TRACE env var so the cost only appears when the
    /// caller asks for it.
    #[cfg(feature = "cuda")]
    fn trace_buffer_stats(&self, label: &str, src: DevicePtr, count: usize) -> Result<()> {
        unsafe extern "C" {
            fn qwen36_cuda_memcpy_d2h(dst: *mut core::ffi::c_void, src: u64, bytes: usize) -> i32;
        }
        qwen36_fp4_kernels::cuda_synchronize()?;
        let bytes = count * 2;
        let mut buf = vec![0u8; bytes];
        let status = unsafe { qwen36_cuda_memcpy_d2h(buf.as_mut_ptr() as *mut _, src.0, bytes) };
        if status != 0 {
            return Err(CoreError::Runtime(format!(
                "trace_buffer_stats memcpy failed: status {status}"
            )));
        }
        let mut max = f32::NEG_INFINITY;
        let mut min = f32::INFINITY;
        let mut sum_abs: f64 = 0.0;
        let mut nans = 0usize;
        let mut infs = 0usize;
        for chunk in buf.chunks_exact(2) {
            let bits: u32 = (u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16;
            let v = f32::from_bits(bits);
            if v.is_nan() {
                nans += 1;
            } else if v.is_infinite() {
                infs += 1;
            } else {
                if v > max {
                    max = v;
                }
                if v < min {
                    min = v;
                }
                sum_abs += v.abs() as f64;
            }
        }
        eprintln!(
            "trace[{label}] n={count} mean_abs={:.6} min={min:.4} max={max:.4} nans={nans} infs={infs}",
            sum_abs / count.max(1) as f64,
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_weights(&self) -> Result<&GpuWeightStore> {
        self.gpu_weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("CUDA weights are not uploaded".to_owned()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_runtime(&self) -> Result<&GpuRuntimeBuffers> {
        self.gpu_buffers
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("CUDA runtime buffers are not allocated".to_owned()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_forward(&self) -> Result<&GpuForwardBuffers> {
        self.gpu_forward
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("CUDA forward buffers are not allocated".to_owned()))
    }

    /// Ensure the secondary prefetch stream + event pool exist when productive
    /// spin is enabled via `QWEN36_PRODUCTIVE_SPIN`. Called from every graph
    /// capture entry point so the captured graph can reference the prefetch
    /// stream. No-op (and no allocation) when the env var is off.
    #[cfg(feature = "cuda")]
    fn ensure_decode_aux_if_enabled(&mut self) -> Result<()> {
        if !productive_spin_enabled() {
            return Ok(());
        }
        if self.decode_aux.is_none() {
            self.decode_aux = Some(DecodeAuxStreams::new(4)?);
        }
        Ok(())
    }

    /// Productive-spin fork: launches the L2 prefetch kernel on the secondary
    /// stream targeting the given full-attn layer's MLP combined weight. The
    /// prefetch runs concurrently with the attention kernel that follows on
    /// the main stream. Returns the event to wait on before MLP starts.
    ///
    /// No-op when `decode_aux` is unset (productive spin disabled or no MLP
    /// fused store for this layer). Caller must pair every `Some` return with
    /// a matching [`Self::join_productive_spin`].
    #[cfg(feature = "cuda")]
    fn fork_productive_spin(
        &self,
        layer_idx: usize,
    ) -> Result<Option<qwen36_fp4_kernels::graph::CudaEvent>> {
        let aux = match self.decode_aux.as_ref() {
            Some(aux) => aux,
            None => return Ok(None),
        };
        let mlp = match self.mlp_fused_main_opt(layer_idx) {
            Some(layer) => layer,
            None => return Ok(None),
        };
        let active = qwen36_fp4_kernels::graph::get_active_stream();
        let prefetch = aux.prefetch_stream();
        // Fork: main → prefetch.
        let fork_event = aux.next_event();
        fork_event.record(active)?;
        qwen36_fp4_kernels::graph::stream_wait_event(prefetch, fork_event)?;
        // Launch prefetch on the prefetch stream (reads internal prefetch
        // stream registered by DecodeAuxStreams::new).
        qwen36_fp4_kernels::memory::cuda_l2_prefetch(
            mlp.combined_weight.ptr(),
            mlp.combined_weight.bytes(),
            productive_spin_ctas(),
        )?;
        // Record join event on prefetch — caller will have main wait on it.
        let join_event = aux.next_event();
        join_event.record(prefetch)?;
        Ok(Some(join_event))
    }

    /// Productive-spin join: blocks the main stream until the prefetch kernel
    /// recorded for `event` has completed. Must be called once per `Some`
    /// returned by [`Self::fork_productive_spin`].
    #[cfg(feature = "cuda")]
    fn join_productive_spin(&self, event: qwen36_fp4_kernels::graph::CudaEvent) -> Result<()> {
        let active = qwen36_fp4_kernels::graph::get_active_stream();
        qwen36_fp4_kernels::graph::stream_wait_event(active, event)
    }

    #[cfg(feature = "cuda")]
    fn cuda_prefill(&self) -> Result<&GpuPrefillBuffers> {
        self.gpu_prefill
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("CUDA prefill buffers are not allocated".to_owned()))
    }
}

#[cfg(feature = "cuda")]
fn layer_common(layer: &LayerWeights) -> &CommonLayerWeights {
    match layer {
        LayerWeights::LinearAttention(layer) => &layer.common,
        LayerWeights::FullAttention(layer) => &layer.common,
    }
}

#[cfg(feature = "cuda")]
fn layer_common_input_norm(layer: &LayerWeights) -> &TensorInfo {
    match layer {
        LayerWeights::LinearAttention(layer) => &layer.common.input_layernorm,
        LayerWeights::FullAttention(layer) => &layer.common.input_layernorm,
    }
}

#[cfg(feature = "cuda")]
impl Engine<CudaBackend> {
    pub fn cuda_with_mapped_weights(model: &MappedModel, config: EngineConfig) -> Result<Self> {
        validate_max_context(&config, &model.layout.topology)?;
        if config.mtp_speculative_tokens > MTP_MAX_DRAFT_TOKENS {
            return Err(CoreError::Runtime(format!(
                "MTP runtime currently supports up to {MTP_MAX_DRAFT_TOKENS} speculative tokens"
            )));
        }
        let manifest = ModelWeightsManifest::from_layout(&model.layout)?;
        let include_mtp = config.mtp_speculative_tokens > 0;
        if include_mtp && manifest.mtp.is_none() {
            return Err(CoreError::Runtime(
                "MTP speculative decoding requested but no structured MTP weights were found"
                    .to_owned(),
            ));
        }
        let gpu_weights = GpuWeightStore::upload_required(model, &manifest, include_mtp)?;
        let mut engine = Self::new(model.layout.topology.clone(), config, CudaBackend);
        let mtp_kv_cache_bytes = if include_mtp {
            engine
                .mtp_kv_cache_plane_bytes()?
                .checked_mul(2)
                .ok_or_else(|| CoreError::Runtime("MTP KV cache size overflow".to_owned()))?
                as u64
        } else {
            0
        };
        if include_mtp
            && cuda_env_bool("QWEN36_MTP_SNAPSHOT_KV")
            && engine
                .state
                .kv_cache
                .layers
                .iter()
                .any(|layer| layer.metadata_bytes != 0 || layer.k_bytes != layer.v_bytes)
        {
            return Err(CoreError::Runtime(
                "MTP KV snapshot is not implemented for TurboQuant KV layout".to_owned(),
            ));
        }
        let gpu_buffers = GpuRuntimeBuffers::allocate(
            &engine.state,
            cuda_env_workspace_bytes(),
            mtp_kv_cache_bytes,
            &engine.topology,
            include_mtp && mtp_recurrent_snapshot_enabled(),
            include_mtp && cuda_env_bool("QWEN36_MTP_SNAPSHOT_KV"),
        )?;
        let gpu_forward = GpuForwardBuffers::allocate(&engine.topology, engine.config.max_context)?;
        let prefill_capacity = cuda_prefill_capacity(engine.config.max_context);
        let fused_mlp_prefill = cuda_prefill_fused_mlp_enabled(engine.config.max_context)
            && cuda_mlp_fused_enabled(engine.config.max_context);
        let gpu_prefill =
            GpuPrefillBuffers::allocate(&engine.topology, prefill_capacity, fused_mlp_prefill)?;
        let mlp_fused = if cuda_mlp_fused_enabled(engine.config.max_context)
            || cuda_prefill_fused_mlp_enabled(engine.config.max_context)
        {
            Some(MlpFusedStore::build(
                &gpu_weights,
                &manifest,
                engine.topology.intermediate_size,
            )?)
        } else {
            None
        };
        let linear_attn_in_proj_fused = if cuda_linear_attn_fused_enabled(engine.config.max_context)
            || cuda_prefill_fused_linear_attn_enabled(engine.config.max_context)
        {
            Some(LinearAttnInProjFusedStore::build(&gpu_weights, &manifest)?)
        } else {
            None
        };
        engine.weights = Some(manifest);
        engine.gpu_weights = Some(gpu_weights);
        engine.gpu_buffers = Some(gpu_buffers);
        engine.gpu_forward = Some(gpu_forward);
        engine.gpu_prefill = Some(gpu_prefill);
        engine.mlp_fused = mlp_fused;
        engine.linear_attn_in_proj_fused = linear_attn_in_proj_fused;
        engine.build_lm_head_fp8()?;
        Ok(engine)
    }

    /// One-time GPU quantization of the BF16 lm_head to FP8 e4m3 + per-row
    /// f32 scales (default ON; kill: `QWEN36_LM_HEAD_FP8=0`). All lm_head
    /// consumers (decode logits, prefill logits, MTP verify rows) then read
    /// 1 byte/weight instead of 2. Must be all-or-nothing across consumers:
    /// mixing FP8 and BF16 logits across MTP modes would break the MTP
    /// parity floor on borderline-argmax tokens.
    #[cfg(feature = "cuda")]
    fn build_lm_head_fp8(&mut self) -> Result<()> {
        if !cuda_env_bool_default_true("QWEN36_LM_HEAD_FP8") {
            return Ok(());
        }
        // The opt-in interpreter logits slice (QWEN36_INTERPRETER_LOGITS)
        // reads the BF16 lm_head via its LM_HEAD_TILED opcode; mixing it
        // with FP8 logits elsewhere would diverge across paths. The
        // diagnostic mode wins.
        if decode_interpreter_logits_enabled() {
            return Ok(());
        }
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let lm_head = manifest.lm_head.clone();
        let out_features = *lm_head.shape.first().ok_or_else(|| {
            CoreError::Runtime(format!("tensor {} has empty shape", lm_head.name))
        })?;
        let in_features = *lm_head.shape.get(1).ok_or_else(|| {
            CoreError::Runtime(format!("tensor {} is not a matrix", lm_head.name))
        })?;
        if in_features % 16 != 0 {
            return Ok(()); // shape outside the fp8_matvec contract: keep BF16
        }
        let weights = self.cuda_weights()?;
        let weight_bf16 = self.tensor_ptr(weights, &lm_head)?;
        let weight_e4m3 = CudaDeviceBuffer::alloc(out_features * in_features)
            .map_err(|err| CoreError::Runtime(format!("lm_head fp8 alloc: {err}")))?;
        let row_scale_f32 = CudaDeviceBuffer::alloc(out_features * size_of::<f32>())
            .map_err(|err| CoreError::Runtime(format!("lm_head fp8 scale alloc: {err}")))?;
        self.backend.fp8_quantize_rows(&Fp8QuantizeRowsSpec {
            out_features,
            in_features,
            weight_bf16,
            weight_e4m3: weight_e4m3.ptr(),
            row_scale_f32: row_scale_f32.ptr(),
        })?;
        qwen36_fp4_kernels::cuda_synchronize()?;
        self.lm_head_fp8 = Some(LmHeadFp8 {
            tensor_name: lm_head.name.clone(),
            weight_e4m3,
            row_scale_f32,
        });
        Ok(())
    }

    pub fn gpu_weight_summary(&self) -> Option<(usize, u64)> {
        self.gpu_weights
            .as_ref()
            .map(|weights| (weights.tensor_count(), weights.total_bytes()))
    }

    pub fn gpu_buffer_bytes(&self) -> Option<u64> {
        Some(
            self.gpu_buffers.as_ref()?.total_bytes()
                + self.gpu_forward.as_ref()?.total_bytes()
                + self.gpu_prefill.as_ref()?.total_bytes(),
        )
    }

    pub fn gpu_memory_report(&self) -> Option<GpuMemoryReport> {
        fn item(name: &str, bytes: u64) -> GpuMemoryItem {
            GpuMemoryItem {
                name: name.to_owned(),
                bytes,
            }
        }
        fn group(items: Vec<GpuMemoryItem>) -> GpuMemoryGroup {
            let total_bytes = items.iter().map(|item| item.bytes).sum();
            GpuMemoryGroup { total_bytes, items }
        }

        let weights = self.gpu_weights.as_ref()?;
        let runtime = self.gpu_buffers.as_ref()?;
        let forward = self.gpu_forward.as_ref()?;
        let prefill = self.gpu_prefill.as_ref()?;

        let weights_group = group(vec![item("uploaded_model_tensors", weights.total_bytes())]);

        let runtime_group = group(vec![
            item(
                "kv_cache",
                runtime
                    .kv_cache
                    .as_ref()
                    .map(|buffer| buffer.bytes() as u64)
                    .unwrap_or(0),
            ),
            item(
                "mtp_kv_cache",
                runtime
                    .mtp_kv_cache
                    .as_ref()
                    .map(|buffer| buffer.bytes() as u64)
                    .unwrap_or(0),
            ),
            item("deltanet_state", runtime.deltanet_state.bytes() as u64),
            item(
                "deltanet_checkpoint",
                runtime
                    .deltanet_checkpoint
                    .as_ref()
                    .map(|buffer| buffer.bytes() as u64)
                    .unwrap_or(0),
            ),
            item("conv_history", runtime.conv_history.bytes() as u64),
            item(
                "conv_history_checkpoint",
                runtime
                    .conv_history_checkpoint
                    .as_ref()
                    .map(|buffer| buffer.bytes() as u64)
                    .unwrap_or(0),
            ),
            item(
                "mtp_kv_snapshot",
                runtime
                    .mtp_kv_snapshot
                    .as_ref()
                    .map(|buffer| buffer.bytes() as u64)
                    .unwrap_or(0),
            ),
            item(
                "workspace",
                runtime
                    .workspace
                    .as_ref()
                    .map(|buffer| buffer.bytes() as u64)
                    .unwrap_or(0),
            ),
            item(
                "deltanet_leaf_checkpoints",
                runtime
                    .deltanet_leaf_checkpoints
                    .iter()
                    .map(|buffer| buffer.bytes() as u64)
                    .sum(),
            ),
            item(
                "conv_history_leaf_checkpoints",
                runtime
                    .conv_history_leaf_checkpoints
                    .iter()
                    .map(|buffer| buffer.bytes() as u64)
                    .sum(),
            ),
        ]);

        let forward_group = group(vec![
            item("hidden", forward.hidden.bytes() as u64),
            item("residual", forward.residual.bytes() as u64),
            item("normed", forward.normed.bytes() as u64),
            item("block_out", forward.block_out.bytes() as u64),
            item("qkv", forward.qkv.bytes() as u64),
            item("aux", forward.aux.bytes() as u64),
            item("aux2", forward.aux2.bytes() as u64),
            item("aux3", forward.aux3.bytes() as u64),
            item("gate_f32", forward.gate_f32.bytes() as u64),
            item("beta_f32", forward.beta_f32.bytes() as u64),
            item("activation_fp4", forward.activation_fp4.bytes() as u64),
            item("activation_scale", forward.activation_scale.bytes() as u64),
            item(
                "activation_scale_2",
                forward.activation_scale_2.bytes() as u64,
            ),
            item("token_u32", forward.token_u32.bytes() as u64),
            item("position_i32", forward.position_i32.bytes() as u64),
            item("logits", forward.logits.bytes() as u64),
            item("mtp_logits", forward.mtp_logits.bytes() as u64),
            item(
                "sampled_token_u32",
                forward.sampled_token_u32.bytes() as u64,
            ),
            item(
                "mtp_verify_token_u32",
                forward.mtp_verify_token_u32.bytes() as u64,
            ),
            item("attn_partial_acc", forward.attn_partial_acc.bytes() as u64),
            item("attn_partial_max", forward.attn_partial_max.bytes() as u64),
            item(
                "attn_partial_denom",
                forward.attn_partial_denom.bytes() as u64,
            ),
            item(
                "interpreter_logits_instructions",
                forward.interpreter_logits_instructions.bytes() as u64,
            ),
            item(
                "interpreter_logits_counters",
                forward.interpreter_logits_counters.bytes() as u64,
            ),
            item(
                "interpreter_norm_instructions",
                forward.interpreter_norm_instructions.bytes() as u64,
            ),
            item(
                "interpreter_norm_counters",
                forward.interpreter_norm_counters.bytes() as u64,
            ),
            item(
                "interpreter_mlp_instructions",
                forward.interpreter_mlp_instructions.bytes() as u64,
            ),
            item(
                "interpreter_mlp_counters",
                forward.interpreter_mlp_counters.bytes() as u64,
            ),
            item(
                "interpreter_rope_instructions",
                forward.interpreter_rope_instructions.bytes() as u64,
            ),
            item(
                "interpreter_rope_counters",
                forward.interpreter_rope_counters.bytes() as u64,
            ),
            item(
                "interpreter_deltanet_spec",
                forward.interpreter_deltanet_spec.bytes() as u64,
            ),
            item(
                "interpreter_deltanet_instructions",
                forward.interpreter_deltanet_instructions.bytes() as u64,
            ),
            item(
                "interpreter_deltanet_counters",
                forward.interpreter_deltanet_counters.bytes() as u64,
            ),
            item(
                "interpreter_attention_spec",
                forward.interpreter_attention_spec.bytes() as u64,
            ),
            item(
                "interpreter_attention_instructions",
                forward.interpreter_attention_instructions.bytes() as u64,
            ),
            item(
                "interpreter_attention_counters",
                forward.interpreter_attention_counters.bytes() as u64,
            ),
        ]);

        let prefill_group = group(vec![
            item("hidden", prefill.hidden.bytes() as u64),
            item("residual", prefill.residual.bytes() as u64),
            item("normed", prefill.normed.bytes() as u64),
            item("block_out", prefill.block_out.bytes() as u64),
            item("qkv", prefill.qkv.bytes() as u64),
            item("aux", prefill.aux.bytes() as u64),
            item("aux2", prefill.aux2.bytes() as u64),
            item("aux3", prefill.aux3.bytes() as u64),
            item("gate_f32", prefill.gate_f32.bytes() as u64),
            item("beta_f32", prefill.beta_f32.bytes() as u64),
            item("activation_fp4", prefill.activation_fp4.bytes() as u64),
            item("activation_scale", prefill.activation_scale.bytes() as u64),
            item(
                "activation_scale_2",
                prefill.activation_scale_2.bytes() as u64,
            ),
            item("token_u32", prefill.token_u32.bytes() as u64),
            item("position_i32", prefill.position_i32.bytes() as u64),
        ]);

        let fused_group = group(vec![
            item(
                "mlp_fused_store",
                self.mlp_fused
                    .as_ref()
                    .map(|store| store.total_bytes)
                    .unwrap_or(0),
            ),
            item(
                "linear_attn_in_proj_fused_store",
                self.linear_attn_in_proj_fused
                    .as_ref()
                    .map(|store| store.total_bytes)
                    .unwrap_or(0),
            ),
        ]);

        let total_reported_bytes = weights_group.total_bytes
            + runtime_group.total_bytes
            + forward_group.total_bytes
            + prefill_group.total_bytes
            + fused_group.total_bytes;
        let max_context = self.config.max_context;

        Some(GpuMemoryReport {
            total_reported_bytes,
            weights: weights_group,
            runtime: runtime_group,
            forward: forward_group,
            prefill: prefill_group,
            fused: fused_group,
            max_context,
            prefill_capacity: prefill.capacity,
            kv_cache_dtype: self.config.kv_cache_dtype,
            mtp_speculative_tokens: self.config.mtp_speculative_tokens,
            long_context_mode: cuda_long_context_mode_enabled(max_context),
            long_context_auto_min_context: cuda_long_context_auto_min_context(),
            mlp_fused_enabled: cuda_mlp_fused_enabled(max_context),
            linear_attn_fused_enabled: cuda_linear_attn_fused_enabled(max_context),
            prefill_fused_linear_attn_enabled: cuda_prefill_fused_linear_attn_enabled(max_context),
        })
    }
}

#[cfg(test)]
mod max_context_tests {
    use super::*;

    #[test]
    fn default_is_sane_and_below_model_ceiling() {
        let default = DEFAULT_MAX_CONTEXT;
        assert!((8_192..=32_768).contains(&default));
        assert!(default <= MODEL_MAX_CONTEXT);
    }

    #[test]
    fn validate_rejects_zero_and_above_model_ceiling() {
        let topology = ModelTopology::expected_qwen36_text_mtp();
        assert_eq!(topology.max_position_embeddings, MODEL_MAX_CONTEXT);
        let mut config = EngineConfig {
            max_context: 0,
            ..EngineConfig::default()
        };
        assert!(validate_max_context(&config, &topology).is_err());
        config.max_context = 1;
        assert!(validate_max_context(&config, &topology).is_ok());
        // The model ceiling itself is reachable — but only by explicit opt-in.
        config.max_context = topology.max_position_embeddings;
        assert!(validate_max_context(&config, &topology).is_ok());
        config.max_context = topology.max_position_embeddings + 1;
        assert!(validate_max_context(&config, &topology).is_err());
    }
}
