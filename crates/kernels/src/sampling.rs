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

/// CPU reference for top-K argmax used in tests and as a fallback.
/// Returns the K vocab indices with the highest logits, sorted by descending
/// logit (ties broken via `f32::total_cmp`).
pub fn topk_argmax(logits: &[f32], k: usize) -> Vec<u32> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    indexed
        .into_iter()
        .take(k)
        .map(|(idx, _)| idx as u32)
        .collect()
}

pub fn greedy_argmax(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_argmax_returns_sorted_top_k_indices() {
        let logits = vec![0.1, 5.0, 2.0, 5.5, 1.0];
        assert_eq!(topk_argmax(&logits, 3), vec![3, 1, 2]);
    }

    #[test]
    fn topk_argmax_caps_at_input_length() {
        let logits = vec![0.0, 1.0];
        assert_eq!(topk_argmax(&logits, 8), vec![1, 0]);
    }

    #[test]
    fn topk_argmax_zero_k_is_empty() {
        assert_eq!(topk_argmax(&[1.0, 2.0], 0), Vec::<u32>::new());
    }

    #[test]
    fn topk_argmax_k_one_matches_greedy_argmax() {
        let logits = vec![0.1, 0.4, 0.3, 0.4001];
        assert_eq!(topk_argmax(&logits, 1), vec![3]);
    }
}
