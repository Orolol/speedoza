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
    pub metadata_offset_bytes: u64,
    pub k_bytes: u64,
    pub v_bytes: u64,
    pub metadata_bytes: u64,
    pub layer_bytes: u64,
}

impl KvCachePlan {
    pub fn new(topology: &ModelTopology, max_context: usize, dtype: KvCacheDtype) -> Self {
        let values_per_layer = max_context as u64
            * topology.attention_num_kv_heads as u64
            * topology.attention_head_dim as u64;
        let bytes_for_bits = |bits: u64| -> u64 { (values_per_layer * bits).div_ceil(8) };
        let (k_bytes, v_bytes, metadata_bytes) = match dtype {
            KvCacheDtype::Bf16 => (values_per_layer * 2, values_per_layer * 2, 0),
            KvCacheDtype::Fp8 => (values_per_layer, values_per_layer, 0),
            KvCacheDtype::TurboQuant3 => {
                let key_bytes = bytes_for_bits(2) + bytes_for_bits(1);
                let value_bytes = bytes_for_bits(3);
                let metadata_vectors = max_context as u64 * topology.attention_num_kv_heads as u64;
                (key_bytes, value_bytes, metadata_vectors * 4 * 4)
            }
            KvCacheDtype::TurboQuant35 => {
                let key_bytes = bytes_for_bits(2) + bytes_for_bits(1);
                let value_bytes = bytes_for_bits(4);
                let metadata_vectors = max_context as u64 * topology.attention_num_kv_heads as u64;
                (key_bytes, value_bytes, metadata_vectors * 4 * 4)
            }
        };
        let mut cursor = 0;
        let mut layers = Vec::new();
        for (attention_layer_index, global_layer_index) in
            topology.attention_layers().into_iter().enumerate()
        {
            let k_offset_bytes = cursor;
            let v_offset_bytes = cursor + k_bytes;
            let metadata_offset_bytes = v_offset_bytes + v_bytes;
            let layer_bytes = k_bytes + v_bytes + metadata_bytes;
            layers.push(KvCacheLayout {
                global_layer_index,
                attention_layer_index,
                k_offset_bytes,
                v_offset_bytes,
                metadata_offset_bytes,
                k_bytes,
                v_bytes,
                metadata_bytes,
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
