use serde::{Deserialize, Serialize};

use crate::backend::DevicePtr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingLookupSpec {
    pub tokens: usize,
    pub hidden: usize,
    pub vocab_size: usize,
    pub token_ids_u32: DevicePtr,
    pub embedding_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bf16MatVecSpec {
    pub out_features: usize,
    pub in_features: usize,
    pub input_bf16: DevicePtr,
    pub weight_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bf16GemmSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_bf16: DevicePtr,
    pub b_bf16: DevicePtr,
    pub c_bf16: DevicePtr,
    pub workspace: DevicePtr,
    pub workspace_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nvfp4MatVecSpec {
    pub out_features: usize,
    pub in_features: usize,
    pub input_bf16: DevicePtr,
    pub weight_u8: DevicePtr,
    pub block_scale_e4m3: DevicePtr,
    pub tensor_scale_f32: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nvfp4QuantizeSpec {
    pub values: usize,
    pub input_bf16: DevicePtr,
    pub output_fp4: DevicePtr,
    pub output_scale_e4m3: DevicePtr,
    pub output_tensor_scale_f32: DevicePtr,
    pub input_tensor_scale_f32: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nvfp4QuantizeRowsSpec {
    pub rows: usize,
    pub values: usize,
    pub input_bf16: DevicePtr,
    pub output_fp4: DevicePtr,
    pub output_scale_e4m3: DevicePtr,
    pub output_tensor_scale_f32: DevicePtr,
    pub input_tensor_scale_f32: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmsNormNvfp4QuantizeSpec {
    pub hidden: usize,
    pub eps: f32,
    pub input_bf16: DevicePtr,
    pub weight_bf16: DevicePtr,
    pub residual_bf16: DevicePtr,
    pub residual_out_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
    pub output_fp4: DevicePtr,
    pub output_scale_e4m3: DevicePtr,
    pub output_tensor_scale_f32: DevicePtr,
    pub input_tensor_scale_f32: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nvfp4RetileScalesSpec {
    pub rows: usize,
    pub inner_groups: usize,
    pub input_row_major_u8: DevicePtr,
    pub output_tiled_u8: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv1dUpdateSpec {
    pub channels: usize,
    pub kernel_size: usize,
    pub input_bf16: DevicePtr,
    pub conv_history_bf16: DevicePtr,
    pub weight_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv1dPrefillSpec {
    pub tokens: usize,
    pub channels: usize,
    pub kernel_size: usize,
    pub input_bf16: DevicePtr,
    pub conv_history_bf16: DevicePtr,
    pub weight_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdnGateSpec {
    pub rows: usize,
    pub heads: usize,
    pub a_bf16: DevicePtr,
    pub b_bf16: DevicePtr,
    pub a_log_bf16: DevicePtr,
    pub dt_bias_bf16: DevicePtr,
    pub gate_f32: DevicePtr,
    pub beta_f32: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv1dGdnGateFusedSpec {
    pub channels: usize,
    pub kernel_size: usize,
    pub conv_input_bf16: DevicePtr,
    pub conv_history_bf16: DevicePtr,
    pub conv_weight_bf16: DevicePtr,
    pub conv_output_bf16: DevicePtr,
    pub heads: usize,
    pub gdn_a_bf16: DevicePtr,
    pub gdn_b_bf16: DevicePtr,
    pub gdn_a_log_bf16: DevicePtr,
    pub gdn_dt_bias_bf16: DevicePtr,
    pub gate_f32: DevicePtr,
    pub beta_f32: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigmoidGateSpec {
    pub elements: usize,
    pub gate_bf16: DevicePtr,
    pub input_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigmoidGateStridedSpec {
    pub rows: usize,
    pub elements_per_row: usize,
    pub gate_stride: usize,
    pub input_stride: usize,
    pub output_stride: usize,
    pub gate_bf16: DevicePtr,
    pub input_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QProjDeinterleaveSpec {
    pub rows: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub input_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QProjSigmoidGateSpec {
    pub rows: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub gate_bf16: DevicePtr,
    pub input_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyStridedRowsSpec {
    pub rows: usize,
    pub values: usize,
    pub input_stride: usize,
    pub output_stride: usize,
    pub input_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

/// Per-block megakernel Stage B.3: fused RMSNorm + NVFP4 quantize + Q
/// projection NVFP4 GEMV in one launch. See
/// `kernels-cuda/include/qwen36_fp4.h` for the full ABI documentation.
///
/// Caller pre-zeroes `barrier_state` (≥ 8 bytes) on the active stream
/// (use `MegakernelBarrierState::reset_async`). `hidden_size` must be a
/// multiple of 512 and `q_features` must be a multiple of 16.
/// `q_alpha = q_weight_tensor_scale * input_tensor_scale` is folded host-
/// side; the kernel ignores `output_tensor_scale_f32` because downstream
/// GEMMs also fold their alphas host-side via `tensor_scalar_f32`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MegakernelFullAttnStageBQProjSpec {
    pub hidden_size: usize,
    pub q_features: usize,
    pub eps: f32,
    pub input_tensor_scale: f32,
    pub q_alpha: f32,
    pub hidden_in: DevicePtr,
    pub input_norm_weight: DevicePtr,
    pub q_weight_fp4: DevicePtr,
    pub q_weight_scale: DevicePtr,
    pub hidden_normed_out_bf16: DevicePtr,
    pub quantized_fp4: DevicePtr,
    pub quantized_scale_e4m3: DevicePtr,
    pub q_out: DevicePtr,
    pub barrier_state: DevicePtr,
}

/// Per-block megakernel Stage E: o_proj GEMV + residual add + post-attn
/// RMSNorm + NVFP4 quantize, fused into one launch. See
/// `kernels-cuda/include/qwen36_fp4.h::qwen36_full_attn_block_stage_e
/// _o_proj_residual_norm` for the full ABI. Caller pre-zeroes
/// `barrier_state` (≥ 16 bytes — 4 phase barrier slots). Alignment:
/// `q_features % 512 == 0`, `hidden_size % 16 == 0`.
/// `o_alpha = o_proj_weight_tensor_scale * attention_output_tensor_scale`
/// is folded host-side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MegakernelFullAttnStageESpec {
    pub q_features: usize,
    pub hidden_size: usize,
    pub eps: f32,
    pub post_input_tensor_scale: f32,
    pub attention_output_tensor_scale: f32,
    pub o_alpha: f32,
    pub attention_out: DevicePtr,
    pub residual_in: DevicePtr,
    pub o_proj_fp4: DevicePtr,
    pub o_proj_scale: DevicePtr,
    pub post_norm_weight: DevicePtr,
    pub attention_quantized_fp4: DevicePtr,
    pub attention_quantized_scale: DevicePtr,
    pub o_proj_out: DevicePtr,
    pub residual_out: DevicePtr,
    pub post_normed_out: DevicePtr,
    pub post_quantized_fp4: DevicePtr,
    pub post_quantized_scale: DevicePtr,
    pub barrier_state: DevicePtr,
}

/// Per-block megakernel Stage F.4: complete MLP block (gate+up GEMV +
/// SwiGLU + NVFP4 quantize + down GEMV + optional residual add). Caller
/// pre-zeroes `barrier_state` (≥ 32 bytes — 4 work counters interleaved
/// with 4 phase barriers). Alignment: `hidden_size % 512 == 0`,
/// `intermediate % 512 == 0`. Set `residual` to NULL to skip phase 3 —
/// the engine integration passes NULL so the next layer's input-norm
/// fuse handles the residual add, matching the standalone pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MegakernelFullAttnStageF4Spec {
    pub intermediate: usize,
    pub hidden_size: usize,
    pub gate_up_alpha: f32,
    pub down_alpha: f32,
    pub down_input_tensor_scale: f32,
    pub hidden_quantized_fp4: DevicePtr,
    pub hidden_quantized_scale: DevicePtr,
    pub mlp_gate_up_fp4: DevicePtr,
    pub mlp_gate_up_scale: DevicePtr,
    pub mlp_down_fp4: DevicePtr,
    pub mlp_down_scale: DevicePtr,
    pub gate_up_out: DevicePtr,
    pub swiglu_fp4: DevicePtr,
    pub swiglu_scale: DevicePtr,
    pub down_out: DevicePtr,
    pub residual: DevicePtr,
    pub barrier_state: DevicePtr,
}
