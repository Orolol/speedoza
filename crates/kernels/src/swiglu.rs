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
