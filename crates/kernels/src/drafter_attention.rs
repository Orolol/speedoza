use serde::{Deserialize, Serialize};

use crate::backend::DevicePtr;

/// First-light DFlash drafter attention. Caller pre-concatenates
/// `K = [k_ctx; k_noise]` and `V = [v_ctx; v_noise]` into a single
/// contiguous buffer of length `kv_seq_len`. No KV-cache append/crop
/// happens inside the kernel; the speculative controller manages cache
/// lifecycle at the Rust layer.
///
/// Layout convention:
/// - Q: `[q_len, q_heads, head_dim]` BF16
/// - K, V: `[kv_seq_len, kv_heads, head_dim]` BF16
/// - Output: `[q_len, q_heads, head_dim]` BF16
///
/// `sliding_window == 0` selects full (non-causal) attention. A non-zero
/// value applies a symmetric SWA mask with the query's absolute position
/// taken as `kv_seq_len - q_len + q_pos`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrafterAttentionBlockSpec {
    pub q_bf16: DevicePtr,
    pub k_bf16: DevicePtr,
    pub v_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
    pub q_len: usize,
    pub kv_seq_len: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
}
