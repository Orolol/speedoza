use serde::{Deserialize, Serialize};

use qwen36_fp4_core::{KvCacheDtype, ModelTopology};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCachePlan {
    pub max_context: usize,
    pub dtype: KvCacheDtype,
    pub layers: Vec<KvCacheLayout>,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheLayout {
    pub global_layer_index: usize,
    pub attention_layer_index: usize,
    pub k_offset_bytes: u64,
    pub v_offset_bytes: u64,
    pub layer_bytes: u64,
}

impl KvCachePlan {
    pub fn new(topology: &ModelTopology, max_context: usize, dtype: KvCacheDtype) -> Self {
        let bits = dtype.bits_per_value();
        let values_per_layer = max_context as u64
            * topology.attention_num_kv_heads as u64
            * topology.attention_head_dim as u64;
        let plane_bytes = ((values_per_layer as f64 * bits) / 8.0).ceil() as u64;
        let mut cursor = 0;
        let mut layers = Vec::new();
        for (attention_layer_index, global_layer_index) in
            topology.attention_layers().into_iter().enumerate()
        {
            let k_offset_bytes = cursor;
            let v_offset_bytes = cursor + plane_bytes;
            let layer_bytes = plane_bytes * 2;
            layers.push(KvCacheLayout {
                global_layer_index,
                attention_layer_index,
                k_offset_bytes,
                v_offset_bytes,
                layer_bytes,
            });
            cursor += layer_bytes;
        }
        Self {
            max_context,
            dtype,
            layers,
            total_bytes: cursor,
        }
    }
}
