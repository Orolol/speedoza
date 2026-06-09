//! Target → drafter hidden-state handoff (Phase E).
//!
//! The DFlash drafter conditions its attention on hidden states pulled
//! from a sparse set of target layers (`target_layer_ids` in the
//! drafter config; for `z-lab/Qwen3.6-27B-DFlash` this is
//! `[1, 16, 31, 46, 61]`). The transformers reference reads
//! `output.hidden_states[layer_id + 1]`, i.e. the post-input-layernorm
//! residual at the *next* target layer iteration.
//!
//! `TargetHiddenCapture` owns a single contiguous
//! `[max_tokens, hidden * n_target_layers]` BF16 buffer whose layout
//! matches what `DrafterForward::forward(target_hidden_raw, ...)`
//! expects. The engine calls [`TargetHiddenCapture::capture_layer`]
//! after each layer's input_layernorm; when the layer index matches
//! one of the configured slots, the per-token hidden state is scattered
//! into the right column block via `copy_strided_rows`.

use anyhow::{Result, anyhow, bail};
use qwen36_fp4_kernels::{CopyStridedRowsSpec, CudaDeviceBuffer, DevicePtr, KernelBackend};

use crate::dflash::DFlashConfig;

/// One scattered hidden-state capture.
#[derive(Debug, Clone, Copy)]
pub struct TargetHiddenCaptureSlot {
    /// Engine prefill layer index *after* whose `input_layernorm` we
    /// capture. Equal to `target_layer_id + 1` per the transformers
    /// `hidden_states[layer_id + 1]` convention.
    pub engine_layer_idx: usize,
    /// Position of this layer's hidden state inside the concatenated
    /// `target_hidden_raw` row (i.e. column offset = `target_slot *
    /// hidden`).
    pub target_slot: usize,
}

pub struct TargetHiddenCapture {
    pub max_tokens: usize,
    pub hidden_size: usize,
    pub n_target_layers: usize,
    /// `[max_tokens, hidden * n_target_layers]` BF16 row-major.
    pub buffer: CudaDeviceBuffer,
    pub slots: Vec<TargetHiddenCaptureSlot>,
}

const BF16_BYTES: usize = 2;

impl TargetHiddenCapture {
    /// Allocate a capture buffer sized for `max_tokens` and configured
    /// for the drafter's `target_layer_ids`. `engine_layer_idx` is
    /// derived as `target_layer_id + 1` to match the transformers
    /// `hidden_states[layer_id + 1]` index.
    pub fn alloc(config: &DFlashConfig, max_tokens: usize) -> Result<Self> {
        if max_tokens == 0 {
            bail!("max_tokens must be > 0");
        }
        let hidden_size = config.hidden_size;
        let n_target_layers = config.dflash_config.target_layer_ids.len();
        if n_target_layers == 0 {
            bail!("dflash_config.target_layer_ids is empty");
        }
        let row_bytes = hidden_size * n_target_layers * BF16_BYTES;
        let buffer = CudaDeviceBuffer::alloc(max_tokens * row_bytes)
            .map_err(|e| anyhow!("alloc target_hidden_raw: {e}"))?;
        let slots = config
            .dflash_config
            .target_layer_ids
            .iter()
            .enumerate()
            .map(|(target_slot, &layer_id)| TargetHiddenCaptureSlot {
                engine_layer_idx: layer_id + 1,
                target_slot,
            })
            .collect();
        Ok(Self {
            max_tokens,
            hidden_size,
            n_target_layers,
            buffer,
            slots,
        })
    }

    pub fn output_ptr(&self) -> DevicePtr {
        self.buffer.ptr()
    }

    pub fn buffer(&self) -> &CudaDeviceBuffer {
        &self.buffer
    }

    pub fn output_stride_bytes(&self) -> usize {
        self.hidden_size * self.n_target_layers * BF16_BYTES
    }

    /// Returns the slot that fires at `engine_layer_idx`, if any.
    pub fn slot_for(&self, engine_layer_idx: usize) -> Option<TargetHiddenCaptureSlot> {
        self.slots
            .iter()
            .find(|slot| slot.engine_layer_idx == engine_layer_idx)
            .copied()
    }

    /// Called by the engine for each prefill layer iteration. NO-OP if
    /// no capture slot matches `engine_layer_idx`. The engine guarantees
    /// `residual_ptr` points at a contiguous `[tokens, hidden]` BF16
    /// tensor.
    pub fn capture_layer<B: KernelBackend>(
        &self,
        backend: &B,
        engine_layer_idx: usize,
        residual_ptr: DevicePtr,
        tokens: usize,
    ) -> Result<()> {
        let Some(slot) = self.slot_for(engine_layer_idx) else {
            return Ok(());
        };
        if tokens > self.max_tokens {
            bail!(
                "capture_layer tokens {tokens} > max_tokens {}",
                self.max_tokens,
            );
        }
        let output_stride = self.hidden_size * self.n_target_layers;
        let column_offset_bytes = slot.target_slot * self.hidden_size * BF16_BYTES;
        let dest_ptr = self
            .buffer
            .ptr_at(column_offset_bytes)
            .map_err(|e| anyhow!("capture column offset: {e}"))?;
        backend
            .copy_strided_rows(&CopyStridedRowsSpec {
                rows: tokens,
                values: self.hidden_size,
                input_stride: self.hidden_size,
                output_stride,
                input_bf16: residual_ptr,
                output_bf16: dest_ptr,
            })
            .map_err(|e| anyhow!("copy_strided_rows layer {engine_layer_idx}: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dflash::DFlashSubConfig;

    fn config_with_target_ids(ids: Vec<usize>) -> DFlashConfig {
        DFlashConfig {
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 5,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 248320,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            sliding_window: 2048,
            use_sliding_window: true,
            layer_types: vec![
                "sliding_attention".into(),
                "sliding_attention".into(),
                "sliding_attention".into(),
                "sliding_attention".into(),
                "full_attention".into(),
            ],
            num_target_layers: 64,
            dflash_config: DFlashSubConfig {
                mask_token_id: 248070,
                target_layer_ids: ids,
            },
            block_size: 16,
            attention_bias: false,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn slots_apply_plus_one_offset() {
        // Reproduce the layout `for_drafter` would derive for the
        // z-lab/Qwen3.6-27B-DFlash drafter (without actually allocating
        // GPU memory). engine_layer_idx is `target_layer_id + 1`.
        let config = config_with_target_ids(vec![1, 16, 31, 46, 61]);
        let derived: Vec<_> = config
            .dflash_config
            .target_layer_ids
            .iter()
            .enumerate()
            .map(|(slot, &id)| (id + 1, slot))
            .collect();
        let expected = vec![(2, 0), (17, 1), (32, 2), (47, 3), (62, 4)];
        assert_eq!(derived, expected);
    }
}
