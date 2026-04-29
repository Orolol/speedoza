use serde::{Deserialize, Serialize};

use crate::backend::DevicePtr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CublasLtFp4ScaleMode {
    Vec16Ue4m3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nvfp4GemmSpec {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_fp4: DevicePtr,
    pub a_scale: DevicePtr,
    pub a_scale_2: DevicePtr,
    pub b_fp4: DevicePtr,
    pub b_scale: DevicePtr,
    pub b_scale_2: DevicePtr,
    pub c_bf16: DevicePtr,
    pub workspace: DevicePtr,
    pub workspace_bytes: usize,
    pub alpha: f32,
    pub scale_mode: CublasLtFp4ScaleMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nvfp4GemmPlan {
    pub spec: Nvfp4GemmSpec,
    pub requires_sm: u32,
    pub requires_cuda_major: u32,
    pub requires_cuda_minor: u32,
    pub requires_tn_layout: bool,
}

impl Nvfp4GemmPlan {
    pub fn blackwell_native(spec: Nvfp4GemmSpec) -> Self {
        Self {
            spec,
            requires_sm: 120,
            requires_cuda_major: 13,
            requires_cuda_minor: 0,
            requires_tn_layout: true,
        }
    }
}
