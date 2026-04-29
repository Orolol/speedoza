use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};

pub const QWEN36_TEXT_NVFP4_MTP_MODEL_ID: &str = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub language_model_only: Option<bool>,
    #[serde(default)]
    pub model_type: Option<String>,
    pub text_config: QwenTextConfig,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenTextConfig {
    #[serde(default)]
    pub full_attention_interval: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    pub linear_conv_kernel_dim: Option<usize>,
    #[serde(default)]
    pub linear_key_head_dim: Option<usize>,
    #[serde(default)]
    pub linear_num_key_heads: Option<usize>,
    #[serde(default)]
    pub linear_num_value_heads: Option<usize>,
    #[serde(default)]
    pub linear_value_head_dim: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub mtp_num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTopology {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<LayerType>,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub mtp_num_hidden_layers: usize,
    pub full_attention_interval: usize,
    pub attention_num_heads: usize,
    pub attention_num_kv_heads: usize,
    pub attention_head_dim: usize,
    pub partial_rotary_factor: f32,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rope_theta: f64,
}

impl TryFrom<&HuggingFaceConfig> for ModelTopology {
    type Error = CoreError;

    fn try_from(value: &HuggingFaceConfig) -> Result<Self> {
        let cfg = &value.text_config;
        let rope = cfg.rope_parameters.as_ref();
        let topology = Self {
            hidden_size: required("hidden_size", cfg.hidden_size)?,
            vocab_size: required("vocab_size", cfg.vocab_size)?,
            num_hidden_layers: required("num_hidden_layers", cfg.num_hidden_layers)?,
            layer_types: cfg.layer_types.clone(),
            intermediate_size: required("intermediate_size", cfg.intermediate_size)?,
            max_position_embeddings: required(
                "max_position_embeddings",
                cfg.max_position_embeddings,
            )?,
            mtp_num_hidden_layers: cfg.mtp_num_hidden_layers.unwrap_or(0),
            full_attention_interval: cfg.full_attention_interval.unwrap_or(4),
            attention_num_heads: required("num_attention_heads", cfg.num_attention_heads)?,
            attention_num_kv_heads: required("num_key_value_heads", cfg.num_key_value_heads)?,
            attention_head_dim: required("head_dim", cfg.head_dim)?,
            partial_rotary_factor: cfg
                .partial_rotary_factor
                .unwrap_or_else(|| rope.and_then(|r| r.partial_rotary_factor).unwrap_or(0.25)),
            linear_num_key_heads: required("linear_num_key_heads", cfg.linear_num_key_heads)?,
            linear_num_value_heads: required("linear_num_value_heads", cfg.linear_num_value_heads)?,
            linear_key_head_dim: required("linear_key_head_dim", cfg.linear_key_head_dim)?,
            linear_value_head_dim: required("linear_value_head_dim", cfg.linear_value_head_dim)?,
            linear_conv_kernel_dim: required("linear_conv_kernel_dim", cfg.linear_conv_kernel_dim)?,
            rope_theta: rope.and_then(|r| r.rope_theta).unwrap_or(10_000_000.0),
        };
        topology.validate_qwen36()?;
        Ok(topology)
    }
}

impl ModelTopology {
    pub fn expected_qwen36_text_mtp() -> Self {
        let mut layer_types = Vec::with_capacity(64);
        for _ in 0..16 {
            layer_types.extend([
                LayerType::LinearAttention,
                LayerType::LinearAttention,
                LayerType::LinearAttention,
                LayerType::FullAttention,
            ]);
        }
        Self {
            hidden_size: 5120,
            vocab_size: 248320,
            num_hidden_layers: 64,
            layer_types,
            intermediate_size: 17408,
            max_position_embeddings: 262144,
            mtp_num_hidden_layers: 1,
            full_attention_interval: 4,
            attention_num_heads: 24,
            attention_num_kv_heads: 4,
            attention_head_dim: 256,
            partial_rotary_factor: 0.25,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000_000.0,
        }
    }

    pub fn validate_qwen36(&self) -> Result<()> {
        if self.num_hidden_layers != self.layer_types.len() {
            return Err(CoreError::InvalidTopology(format!(
                "num_hidden_layers={} but layer_types has {} entries",
                self.num_hidden_layers,
                self.layer_types.len()
            )));
        }
        if self.num_hidden_layers != 64 {
            return Err(CoreError::InvalidTopology(format!(
                "expected 64 layers, got {}",
                self.num_hidden_layers
            )));
        }
        for (idx, layer_type) in self.layer_types.iter().enumerate() {
            let expected = if idx % self.full_attention_interval == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            };
            if *layer_type != expected {
                return Err(CoreError::InvalidTopology(format!(
                    "layer {idx} is {layer_type:?}, expected {expected:?}"
                )));
            }
        }
        if self.attention_rope_dims() != 64 {
            return Err(CoreError::InvalidTopology(format!(
                "partial RoPE dims must be 64, got {}",
                self.attention_rope_dims()
            )));
        }
        if self.linear_conv_kernel_dim != 4 {
            return Err(CoreError::InvalidTopology(format!(
                "linear_conv_kernel_dim must be 4, got {}",
                self.linear_conv_kernel_dim
            )));
        }
        Ok(())
    }

    pub fn attention_layers(&self) -> Vec<usize> {
        self.layer_types
            .iter()
            .enumerate()
            .filter_map(|(idx, kind)| (*kind == LayerType::FullAttention).then_some(idx))
            .collect()
    }

    pub fn linear_attention_layers(&self) -> Vec<usize> {
        self.layer_types
            .iter()
            .enumerate()
            .filter_map(|(idx, kind)| (*kind == LayerType::LinearAttention).then_some(idx))
            .collect()
    }

    pub fn turboquant_skip_layers(&self) -> Vec<usize> {
        let attention_layers = self.attention_layers();
        match (attention_layers.first(), attention_layers.last()) {
            (Some(first), Some(last)) if first != last => vec![*first, *last],
            (Some(only), _) => vec![*only],
            _ => Vec::new(),
        }
    }

    pub fn attention_rope_dims(&self) -> usize {
        (self.attention_head_dim as f32 * self.partial_rotary_factor).round() as usize
    }

    pub fn linear_attention_qkv_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim * 2
            + self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn linear_attention_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn full_attention_q_dim_with_gate(&self) -> usize {
        self.attention_num_heads * self.attention_head_dim * 2
    }

    pub fn full_attention_q_dim(&self) -> usize {
        self.attention_num_heads * self.attention_head_dim
    }

    pub fn full_attention_kv_dim(&self) -> usize {
        self.attention_num_kv_heads * self.attention_head_dim
    }

    pub fn deltanet_state_bytes(&self) -> usize {
        self.linear_attention_layers().len()
            * self.linear_num_value_heads
            * self.linear_value_head_dim
            * self.linear_key_head_dim
            * 2
    }
}

fn required(name: &'static str, value: Option<usize>) -> Result<usize> {
    value.ok_or_else(|| CoreError::InvalidTopology(format!("missing {name}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen36_attention_layers_are_every_fourth_layer() {
        let topology = ModelTopology::expected_qwen36_text_mtp();
        assert_eq!(topology.attention_layers()[0], 3);
        assert_eq!(topology.attention_layers()[15], 63);
        assert_eq!(topology.turboquant_skip_layers(), vec![3, 63]);
        assert_eq!(topology.attention_rope_dims(), 64);
    }

    #[test]
    fn deltanet_state_matches_doc_budget() {
        let topology = ModelTopology::expected_qwen36_text_mtp();
        assert_eq!(topology.linear_attention_layers().len(), 48);
        assert_eq!(topology.deltanet_state_bytes(), 75_497_472);
    }
}
