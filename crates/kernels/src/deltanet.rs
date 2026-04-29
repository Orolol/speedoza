use serde::{Deserialize, Serialize};

use crate::backend::DevicePtr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeltaNetShape {
    pub qk_heads: usize,
    pub v_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
}

impl DeltaNetShape {
    pub fn qwen36() -> Self {
        Self {
            qk_heads: 16,
            v_heads: 48,
            key_dim: 128,
            value_dim: 128,
            conv_kernel: 4,
        }
    }

    pub fn recurrent_state_bytes(self) -> usize {
        self.v_heads * self.value_dim * self.key_dim * 2
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaNetPrefillSpec {
    pub layer_index: usize,
    pub tokens: usize,
    pub chunk_size: usize,
    pub hidden_bf16: DevicePtr,
    pub state_bf16: DevicePtr,
    pub conv_history_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
    pub workspace: DevicePtr,
    pub workspace_bytes: usize,
    pub shape: DeltaNetShape,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaNetDecodeSpec {
    pub layer_index: usize,
    pub tokens_in_persistent_loop: usize,
    pub q_bf16: DevicePtr,
    pub k_bf16: DevicePtr,
    pub v_bf16: DevicePtr,
    pub state_bf16: DevicePtr,
    pub conv_history_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
    pub gate_f32: DevicePtr,
    pub beta_f32: DevicePtr,
    pub shape: DeltaNetShape,
    pub state_decay: f32,
    pub update_scale: f32,
    pub qk_l2norm: bool,
}
