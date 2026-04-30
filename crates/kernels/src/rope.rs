use serde::{Deserialize, Serialize};

use crate::backend::DevicePtr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialRopeSpec {
    pub tokens: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub base_theta: f64,
    pub position_i32: i32,
    pub use_scalar_position: bool,
    pub positions_i32: DevicePtr,
    pub q_bf16: DevicePtr,
    pub k_bf16: DevicePtr,
    /// When non-NULL and `use_scalar_position` is set, the kernel reads the
    /// scalar token position from this device `int32_t` instead of the
    /// `position_i32` host field. Lets a captured graph step through positions
    /// without re-recording.
    #[serde(default)]
    pub scalar_position_device_i32: DevicePtr,
}
