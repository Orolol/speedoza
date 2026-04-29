use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDtype {
    Bf16,
    F16,
    F32,
    F8E4M3,
    F8E5M2,
    U8,
    I8,
    I32,
    I64,
    Bool,
    Unknown(String),
}

impl TensorDtype {
    pub fn from_safetensors_debug(value: &str) -> Self {
        match value {
            "BF16" => Self::Bf16,
            "F16" => Self::F16,
            "F32" => Self::F32,
            "F8_E4M3" | "F8E4M3" => Self::F8E4M3,
            "F8_E5M2" | "F8E5M2" => Self::F8E5M2,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "BOOL" | "Bool" => Self::Bool,
            other => Self::Unknown(other.to_owned()),
        }
    }

    pub fn bytes_per_element(&self) -> Option<usize> {
        match self {
            Self::Bool | Self::I8 | Self::U8 | Self::F8E4M3 | Self::F8E5M2 => Some(1),
            Self::Bf16 | Self::F16 => Some(2),
            Self::F32 | Self::I32 => Some(4),
            Self::I64 => Some(8),
            Self::Unknown(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantStorage {
    Bf16,
    Fp8,
    Nvfp4Packed,
    Fp32Scale,
    Other,
}
