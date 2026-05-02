use serde::{Deserialize, Serialize};

use crate::backend::DevicePtr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingSpec {
    pub vocab_size: usize,
    pub logits_bf16: DevicePtr,
    pub output_token_u32: DevicePtr,
    pub mirror_output_token_u32: DevicePtr,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingRowsSpec {
    pub rows: usize,
    pub vocab_size: usize,
    /// Column-major logits as produced by `Bf16GemmSpec { m: vocab, n: rows }`.
    pub logits_bf16: DevicePtr,
    /// Contiguous `rows` u32 outputs.
    pub output_token_u32: DevicePtr,
    /// Optional mirror for the final row's token.
    pub mirror_last_output_token_u32: DevicePtr,
    pub temperature: f32,
}

pub fn greedy_argmax(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
}
