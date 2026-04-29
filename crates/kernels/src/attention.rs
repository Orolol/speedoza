use serde::{Deserialize, Serialize};

use crate::backend::DevicePtr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionShape {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
}

impl AttentionShape {
    pub fn qwen36() -> Self {
        Self {
            q_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            rope_dims: 64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPrefillSpec {
    pub layer_index: usize,
    pub tokens: usize,
    pub q_bf16: DevicePtr,
    pub k_bf16: DevicePtr,
    pub v_bf16: DevicePtr,
    pub kv_cache_k: DevicePtr,
    pub kv_cache_v: DevicePtr,
    pub output_bf16: DevicePtr,
    pub shape: AttentionShape,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionDecodeSpec {
    pub layer_index: usize,
    pub position: usize,
    pub q_bf16: DevicePtr,
    pub k_bf16: DevicePtr,
    pub v_bf16: DevicePtr,
    pub kv_cache_k: DevicePtr,
    pub kv_cache_v: DevicePtr,
    pub output_bf16: DevicePtr,
    pub shape: AttentionShape,
}

