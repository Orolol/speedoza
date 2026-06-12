# DFlash drafter — kernel reuse audit

**Date:** 2026-06-08
**Status:** Bring-up notes; resolves task #9 of the DFlash Phase B slice.

Inventory of which existing kernels the DFlash drafter forward can reuse
as-is and which need new CUDA code. Reference target: the drafter
checkpoint at `z-lab/Qwen3.6-27B-DFlash` (5 layers, hidden 5120, head_dim
128, 32 Q heads / 8 KV heads, block_size 16, 4× sliding_attention + 1×
full_attention, sliding_window 2048).

## Reusable as-is

| Existing spec | DFlash use | Notes |
|---|---|---|
| `EmbeddingLookupSpec` | `noise_embedding = target.embed(MASK)` for the block | Reuses the target's `embed_tokens` device buffer. Already BF16. |
| `Bf16GemmSpec` | Drafter Q/K/V/O projections, `fc`, MLP gate/up/down | All shapes are dense BF16 matmuls; the 5-layer drafter has < 50 GEMM calls per block forward. |
| `Bf16MatVecSpec` | Same as above when `q_len == 1` | Useful if we ever run the drafter in a per-token mode (not in the standard DFlash loop). |
| `RmsNormSpec` | All 4 layernorms per layer + `norm` + `hidden_norm` | `direct_weight=true` for `q_norm`/`k_norm` (head-dim sized) to skip the `(1 + weight)` parameterisation; `direct_weight=false` for `input_layernorm`/`post_attention_layernorm`/`norm`/`hidden_norm` if Qwen3 uses the `(1 + weight)` form **— TO VERIFY against the reference DFlash forward**. The standalone Qwen3RMSNorm in the dflash repo uses `weight * hidden / rms` directly (no `+1`), so we likely want `direct_weight=true` everywhere on the drafter side. |
| `SwiGluSpec` | MLP activation: `silu(gate) * up` BF16 → BF16 | No quantize needed; the drafter stays in BF16 end-to-end. |

## Needs new CUDA code

### DFlash attention kernel (the only non-trivial new kernel)

Existing `AttentionDecodeSpec` is single-Q causal full-attention with a
KV cache; `AttentionPrefillSpec` is multi-Q causal full-attention. The
drafter attention is none of those.

Per `Qwen3DFlashAttention.forward` (dflash repo, `dflash/model.py:211`):

- `q` is `[bsz, q_len=block_size, num_heads, head_dim]`, derived from
  `noise_embedding` only.
- `k` is the concatenation of `k_ctx` (from `target_hidden`,
  shape `[bsz, ctx_len=block_size, kv_heads, head_dim]`) and `k_noise`
  (from `noise_embedding`, same shape). Same for `v`.
- Therefore K and V have length `2 × q_len` at the current cache
  step (no past KV from prior decoder steps after `crop(start)`).
- `is_causal = False`. The forward applies whatever
  `attention_mask` the caller passes plus optional SWA
  (`sliding_window=2048` on the 4 SWA layers, `None` on the full
  layer).
- After attention: `o_proj` (standard BF16 GEMM, reusable).

**Spec sketch** for the new kernel (call it
`drafter_attention_block_bf16`):

```rust
pub struct DrafterAttentionBlockSpec {
    pub q_len: usize,          // == block_size
    pub ctx_len: usize,        // == block_size for non-prefill steps
    pub q_heads: usize,        // 32
    pub kv_heads: usize,       // 8
    pub head_dim: usize,       // 128
    pub sliding_window: usize, // 0 = full attention
    pub q_bf16: DevicePtr,     // [q_len, q_heads, head_dim], post-RoPE, post-q_norm
    pub k_bf16: DevicePtr,     // [ctx_len + q_len, kv_heads, head_dim], post-RoPE, post-k_norm
    pub v_bf16: DevicePtr,     // [ctx_len + q_len, kv_heads, head_dim]
    pub output_bf16: DevicePtr, // [q_len, q_heads, head_dim]
}
```

Algorithm: standard FlashAttention-style tiled GQA over key length
`ctx_len + q_len`. Non-causal: every query attends to every key. SWA
masks key positions outside the window relative to the query position
(only relevant when `ctx_len + q_len > sliding_window` = 2048, so
unaffected for the standard `block_size=16` decode).

**Estimated effort:** medium (2–3 days). Most of the work is the
GQA broadcast and the boundary handling for SWA; the math itself is
simpler than the existing kernels because no causal mask, no KV cache
update, no FP4.

### Full RoPE (head_dim 128, rope_theta 1e7)

The drafter uses **full** RoPE on `head_dim = 128` with
`rope_theta = 10_000_000`. The existing `PartialRopeSpec` takes
`rope_dims` and `base_theta` as fields, so this should be a pure
parameterisation change — `rope_dims = head_dim = 128`,
`base_theta = 1e7`. Confirmed against `kernels-cuda/ops.cu` —
**no new CUDA code needed**, just an additional call site with the
drafter's parameters.

Cache write/read: the drafter does `k = apply_rotary_pos_emb(k, ...)`
over the concatenated `[k_ctx; k_noise]` with the same cos/sin table
that covers the whole position range `[0, max_position_embeddings)`. We
need the RoPE kernel to support starting at a configurable absolute
position (so `past_key_values.get_seq_length()` controls the offset).
Existing `PartialRopeSpec` already supports `positions_i32` for
per-token positions — covers this.

## Out of scope for this PR / future work

- **Hidden-state handoff target → drafter.** The engine already dumps
  per-layer hidden states for parity (`QWEN36_DEBUG_DUMP_DIR`), but
  zero-copy GPU residency is a separate change. For first-light, can
  copy-back through host (slow, parity-correct, ~3 ms per block) and
  optimise later.
- **`fc` collapse.** `fc.weight` is `[hidden, 5 * hidden] = [5120,
  25600]` — a standard BF16 GEMM. Reusable with `Bf16GemmSpec`. The
  input is the channel-concatenation of 5 target hidden-state tensors
  along the last dim; we just need to allocate a single
  `[block_size, 5 * hidden]` BF16 buffer and write the 5 hidden slices
  into it contiguously before the `fc` call.
- **Layer-norm `(1+w)` vs direct.** Needs a single one-token parity
  smoke (drafter layer 0 only) against the Python reference to
  confirm which `direct_weight` value matches. Plan to land that
  smoke alongside the first end-to-end drafter forward.

## Net summary

| Component | Status |
|---|---|
| BF16 GEMMs (≈ 11 sites per layer + fc) | reuse `Bf16GemmSpec` |
| RMSNorm (4 per layer + `hidden_norm` + `norm`) | reuse `RmsNormSpec` |
| SwiGLU | reuse `SwiGluSpec` |
| RoPE | reuse `PartialRopeSpec` with `rope_dims=head_dim=128`, `base_theta=1e7` |
| Embedding | reuse `EmbeddingLookupSpec` |
| **Attention** | **new kernel** `drafter_attention_block_bf16` |
| Hidden-state handoff (target → drafter) | new buffer, copy-strided concat (Rust-side, no new CUDA) |
| `fc` collapse | reuse `Bf16GemmSpec` after the concat |

**One new CUDA kernel** is the only blocker for an end-to-end drafter
forward. Everything else is glue and bring-up.
