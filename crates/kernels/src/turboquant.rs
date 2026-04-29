use serde::{Deserialize, Serialize};

use crate::attention::AttentionShape;
use crate::backend::DevicePtr;
use qwen36_fp4_core::ModelTopology;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurboQuantMode {
    Off = 0,
    Bits3 = 3,
    Bits35 = 35,
    Bits4 = 4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantPolicy {
    pub mode: TurboQuantMode,
    pub skip_global_layers: Vec<usize>,
    pub hadamard_rotation: bool,
    pub qjl_residual: bool,
}

impl TurboQuantPolicy {
    pub fn for_topology(topology: &ModelTopology, mode: TurboQuantMode) -> Self {
        Self {
            mode,
            skip_global_layers: topology.turboquant_skip_layers(),
            hadamard_rotation: true,
            qjl_residual: matches!(
                mode,
                TurboQuantMode::Bits3 | TurboQuantMode::Bits35 | TurboQuantMode::Bits4
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantEncodeSpec {
    pub layer_index: usize,
    pub position: usize,
    pub k_bf16: DevicePtr,
    pub v_bf16: DevicePtr,
    pub k_quantized: DevicePtr,
    pub v_quantized: DevicePtr,
    pub metadata: DevicePtr,
    pub shape: AttentionShape,
    pub mode: TurboQuantMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantAttentionSpec {
    pub layer_index: usize,
    pub position: usize,
    pub q_bf16: DevicePtr,
    pub k_quantized: DevicePtr,
    pub v_quantized: DevicePtr,
    pub metadata: DevicePtr,
    pub output_bf16: DevicePtr,
    pub workspace: DevicePtr,
    pub workspace_bytes: usize,
    pub shape: AttentionShape,
    pub mode: TurboQuantMode,
}
