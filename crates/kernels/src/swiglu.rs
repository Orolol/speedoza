use serde::{Deserialize, Serialize};

use crate::backend::DevicePtr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwiGluSpec {
    pub rows: usize,
    pub intermediate: usize,
    pub gate_bf16: DevicePtr,
    pub up_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwiGluNvfp4QuantizeSpec {
    pub intermediate: usize,
    pub gate_bf16: DevicePtr,
    pub up_bf16: DevicePtr,
    pub output_fp4: DevicePtr,
    pub output_scale_e4m3: DevicePtr,
    pub output_tensor_scale_f32: DevicePtr,
    pub input_tensor_scale_f32: f32,
}
