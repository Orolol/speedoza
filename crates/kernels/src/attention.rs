use serde::{Deserialize, Serialize};

use crate::backend::DevicePtr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionShape {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
}

impl AttentionShape {
    pub fn qwen36() -> Self {
        Self {
            q_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            rope_dims: 64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPrefillSpec {
    pub layer_index: usize,
    pub start_position: usize,
    pub tokens: usize,
    pub q_bf16: DevicePtr,
    pub k_bf16: DevicePtr,
    pub v_bf16: DevicePtr,
    pub kv_cache_k: DevicePtr,
    pub kv_cache_v: DevicePtr,
    pub output_bf16: DevicePtr,
    pub shape: AttentionShape,
    /// When non-NULL, prefill reads the base cache position from this device
    /// `int32_t` instead of `start_position`. This lets a captured graph replay
    /// across advancing positions.
    #[serde(default)]
    pub start_position_device_i32: DevicePtr,
    /// Optional scratch buffers used by the short-prefill split-KV path for
    /// MTP verification/recovery chunks. Layout matches `AttentionDecodeSpec`
    /// and is reused once per token in the chunk.
    #[serde(default)]
    pub partial_acc_f32: DevicePtr,
    #[serde(default)]
    pub partial_max_f32: DevicePtr,
    #[serde(default)]
    pub partial_denom_f32: DevicePtr,
    /// Number of split-KV blocks per q-head for short prefill. Values 0/1 keep
    /// the normal prefill kernels.
    #[serde(default)]
    pub prefill_n_splits: usize,
    /// Timesteps covered by each split block. When 0, the CUDA default is used.
    #[serde(default)]
    pub split_timesteps_per_block: usize,
    /// Optional tree ancestor bitmap (one u64 per verify-chunk row). Use
    /// `DevicePtr::NULL` to select the existing causal mask.
    #[serde(default)]
    pub tree_ancestor_bitmap_u64: DevicePtr,
    /// Verify-chunk row count (number of valid bitmap entries). 0 = causal.
    #[serde(default)]
    pub verify_chunk_rows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionDecodeSpec {
    pub layer_index: usize,
    pub position: usize,
    pub q_bf16: DevicePtr,
    pub k_bf16: DevicePtr,
    pub v_bf16: DevicePtr,
    pub kv_cache_k: DevicePtr,
    pub kv_cache_v: DevicePtr,
    pub output_bf16: DevicePtr,
    pub shape: AttentionShape,
    /// When non-NULL, the kernel reads the current position from this device
    /// `int32_t` instead of `position`. Used by graph-captured decode so a
    /// single recording can be replayed across iterations.
    #[serde(default)]
    pub position_device_i32: DevicePtr,
    /// Optional scratch buffers used by the split-KV (FlashDecoding-style)
    /// path. When all three are non-NULL the kernel splits the attention
    /// over the time axis into chunks of `split_timesteps_per_block`,
    /// trading per-block work for SM occupancy on long contexts. Must be
    /// large enough for the model's `max_context`. Layout:
    ///   - `partial_acc_f32`: `[q_heads, n_splits, head_dim]` FP32
    ///   - `partial_max_f32`:  `[q_heads, n_splits]` FP32
    ///   - `partial_denom_f32`: `[q_heads, n_splits]` FP32
    ///     where `n_splits = ceil((max_context) / split_timesteps_per_block)`.
    #[serde(default)]
    pub partial_acc_f32: DevicePtr,
    #[serde(default)]
    pub partial_max_f32: DevicePtr,
    #[serde(default)]
    pub partial_denom_f32: DevicePtr,
    /// Number of split-KV blocks per q-head. Set by the engine from
    /// `max_context`, *not* the current position, so the same value works
    /// for graph capture *and* graph replay even as the position grows. A
    /// value of 0 (or 1) tells the kernel to skip the split path and run
    /// the per-q-head kernel inline.
    #[serde(default)]
    pub decode_n_splits: usize,
    /// Timesteps covered by each split block. When 0, the CUDA default is used.
    #[serde(default)]
    pub split_timesteps_per_block: usize,
}
