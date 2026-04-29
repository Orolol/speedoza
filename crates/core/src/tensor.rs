use serde::{Deserialize, Serialize};

use crate::dtype::TensorDtype;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub file: String,
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    pub size_bytes: u64,
    pub role: TensorRole,
    pub layer_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    Nvfp4PackedWeight,
    Nvfp4BlockScale,
    Nvfp4TensorScale,
    Bf16Weight,
    Fp8Scale,
    LmHeadBf16,
    MtpBf16,
    Conv1dBf16,
    Embedding,
    Other,
}

impl TensorInfo {
    pub fn new(
        name: String,
        file: String,
        dtype: TensorDtype,
        shape: Vec<usize>,
        size_bytes: u64,
    ) -> Self {
        let layer_index = parse_layer_index(&name);
        let role = classify_tensor(&name, &dtype);
        Self {
            name,
            file,
            dtype,
            shape,
            size_bytes,
            role,
            layer_index,
        }
    }
}

pub fn parse_layer_index(name: &str) -> Option<usize> {
    let markers = [
        "model.language_model.layers.",
        "model.layers.",
        "language_model.layers.",
    ];
    let marker = markers.iter().find(|marker| name.contains(*marker))?;
    let start = name.find(marker)? + marker.len();
    let rest = &name[start..];
    let digits = rest
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

pub fn classify_tensor(name: &str, dtype: &TensorDtype) -> TensorRole {
    if name.starts_with("mtp.") {
        return TensorRole::MtpBf16;
    }
    if name == "lm_head.weight" || name.starts_with("lm_head.") {
        return TensorRole::LmHeadBf16;
    }
    if name.contains("linear_attn.conv1d") {
        return TensorRole::Conv1dBf16;
    }
    if name.contains("embed_tokens") {
        return TensorRole::Embedding;
    }
    if name.ends_with("weight_scale_2") {
        return TensorRole::Nvfp4TensorScale;
    }
    if name.ends_with("weight_scale") {
        return match dtype {
            TensorDtype::F8E4M3 | TensorDtype::U8 => TensorRole::Nvfp4BlockScale,
            _ => TensorRole::Fp8Scale,
        };
    }
    if matches!(dtype, TensorDtype::U8) && name.ends_with(".weight") {
        return TensorRole::Nvfp4PackedWeight;
    }
    if matches!(dtype, TensorDtype::Bf16) && name.ends_with(".weight") {
        return TensorRole::Bf16Weight;
    }
    TensorRole::Other
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_layer_indices() {
        assert_eq!(
            parse_layer_index("model.language_model.layers.63.self_attn.q_proj.weight"),
            Some(63)
        );
        assert_eq!(parse_layer_index("mtp.layers.0.self_attn.q_proj.weight"), None);
        assert_eq!(parse_layer_index("lm_head.weight"), None);
    }
}
