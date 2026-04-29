use serde::{Deserialize, Serialize};

use crate::config::ModelTopology;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheDtype {
    Bf16,
    Fp8,
    TurboQuant3,
    TurboQuant35,
}

impl KvCacheDtype {
    pub fn bits_per_value(self) -> f64 {
        match self {
            Self::Bf16 => 16.0,
            Self::Fp8 => 8.0,
            Self::TurboQuant3 => 3.0,
            Self::TurboQuant35 => 3.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBudget {
    pub context_tokens: usize,
    pub kv_cache_dtype: KvCacheDtype,
    pub kv_cache_bytes: u64,
    pub deltanet_state_bytes: u64,
    pub deltanet_conv_history_bytes: u64,
    pub estimated_activation_workspace_bytes: u64,
}

impl MemoryBudget {
    pub fn estimate(
        topology: &ModelTopology,
        context_tokens: usize,
        kv_cache_dtype: KvCacheDtype,
    ) -> Self {
        let attention_layers = topology.attention_layers().len() as u64;
        let kv_values = attention_layers
            * 2
            * context_tokens as u64
            * topology.attention_num_kv_heads as u64
            * topology.attention_head_dim as u64;
        let kv_cache_bytes =
            ((kv_values as f64 * kv_cache_dtype.bits_per_value()) / 8.0).ceil() as u64;
        let deltanet_state_bytes = topology.deltanet_state_bytes() as u64;
        let deltanet_conv_history_values = topology.linear_attention_layers().len() as u64
            * topology.hidden_size as u64
            * topology.linear_conv_kernel_dim.saturating_sub(1) as u64;
        Self {
            context_tokens,
            kv_cache_dtype,
            kv_cache_bytes,
            deltanet_state_bytes,
            deltanet_conv_history_bytes: deltanet_conv_history_values * 2,
            estimated_activation_workspace_bytes: 2_u64 * 1024 * 1024 * 1024,
        }
    }
}
