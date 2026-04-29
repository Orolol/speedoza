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
pub struct GdnGateSpec {
    pub heads: usize,
    pub a_bf16: DevicePtr,
    pub b_bf16: DevicePtr,
    pub a_log_bf16: DevicePtr,
    pub dt_bias_bf16: DevicePtr,
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
