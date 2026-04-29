use serde::{Deserialize, Serialize};

use qwen36_fp4_core::ModelTopology;

use crate::kv_cache::KvCachePlan;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaNetStatePlan {
    pub layers: usize,
    pub state_bytes_per_layer: u64,
    pub total_state_bytes: u64,
    pub conv_history_bytes: u64,
    pub checkpoint_bytes: u64,
}

impl DeltaNetStatePlan {
    pub fn new(topology: &ModelTopology) -> Self {
        let layers = topology.linear_attention_layers().len();
        let state_bytes_per_layer = (topology.linear_num_value_heads
            * topology.linear_value_head_dim
            * topology.linear_key_head_dim
            * 2) as u64;
        let total_state_bytes = state_bytes_per_layer * layers as u64;
        let conv_history_bytes = (layers
            * topology.linear_attention_qkv_dim()
            * topology.linear_conv_kernel_dim.saturating_sub(1)
            * 2) as u64;
        Self {
            layers,
            state_bytes_per_layer,
            total_state_bytes,
            conv_history_bytes,
            checkpoint_bytes: total_state_bytes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeState {
    pub position: usize,
    pub accepted_tokens: usize,
    pub kv_cache: KvCachePlan,
    pub deltanet: DeltaNetStatePlan,
}

impl RuntimeState {
    pub fn new(kv_cache: KvCachePlan, deltanet: DeltaNetStatePlan) -> Self {
        Self {
            position: 0,
            accepted_tokens: 0,
            kv_cache,
            deltanet,
        }
    }

    pub fn advance(&mut self, accepted: usize) {
        self.position += accepted;
        self.accepted_tokens += accepted;
    }
}
