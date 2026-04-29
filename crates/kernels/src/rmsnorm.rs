use serde::{Deserialize, Serialize};

use crate::backend::DevicePtr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmsNormSpec {
    pub rows: usize,
    pub hidden: usize,
    pub eps: f32,
    pub input_bf16: DevicePtr,
    pub weight_bf16: DevicePtr,
    pub residual_bf16: DevicePtr,
    pub residual_out_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}
