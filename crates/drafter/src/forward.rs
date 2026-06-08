//! DFlash drafter forward pass (Phase D v1).
//!
//! Runs the 5-layer drafter model end-to-end on GPU using existing
//! BF16 kernels (RMSNorm, BF16 GEMM, SwiGLU, partial RoPE) plus the
//! new `drafter_attention_block_bf16` kernel landed in Phase C.
//!
//! v1 scope:
//! - Caller pre-collapses `target_hidden` through `fc` + `hidden_norm`
//!   (deferred to Phase D.2 — see TODO at `forward`).
//! - No past KV cache: each drafter call attends only over the current
//!   `[k_ctx; k_noise]` window of length `ctx_len + q_len`. Sufficient
//!   for the first drafter forward after a target prefill; multi-step
//!   decode requires the cache and is Phase D.3.
//! - RMSNorm residual fusion is used to chain the per-layer residual
//!   carries without standalone vector-add kernels.

use anyhow::{Context, Result, anyhow, bail};
use qwen36_fp4_kernels::{
    Bf16GemmSpec, CudaDeviceBuffer, DevicePtr, DrafterAttentionBlockSpec, KernelBackend,
    PartialRopeSpec, RmsNormSpec, SwiGluSpec,
};

use crate::dflash::{DFlashConfig, LayerAttentionKind};
use crate::gpu::{DFlashDrafterDevice, DFlashLayerDevice, TensorOnDevice};

/// Pre-allocated intermediate buffers for one drafter forward. Sized
/// for the worst-case `(q_len, ctx_len)` the caller commits to at
/// construction time.
pub struct DrafterForwardWorkspace {
    pub q_len_max: usize,
    pub ctx_len_max: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,

    // Residual ping-pong carries. After each RMSNorm with residual
    // fusion, the post-add tensor lands in one of these and the next
    // RMSNorm reads it. `carry_a` always starts as the initial input
    // (noise_embedding) at the layer-0 entry.
    carry_a: CudaDeviceBuffer,
    carry_b: CudaDeviceBuffer,

    // Per-layer intermediates.
    normed: CudaDeviceBuffer,   // [q_len, hidden] BF16
    q: CudaDeviceBuffer,        // [q_len, q_heads * head_dim] BF16
    k_full: CudaDeviceBuffer,   // [(ctx_len + q_len), kv_heads * head_dim] BF16
    v_full: CudaDeviceBuffer,   // [(ctx_len + q_len), kv_heads * head_dim] BF16
    attn_out: CudaDeviceBuffer, // [q_len, q_heads * head_dim] BF16
    attn_proj: CudaDeviceBuffer, // [q_len, hidden] BF16
    gate: CudaDeviceBuffer,     // [q_len, intermediate] BF16
    up: CudaDeviceBuffer,       // [q_len, intermediate] BF16
    silu: CudaDeviceBuffer,     // [q_len, intermediate] BF16
    mlp_out: CudaDeviceBuffer,  // [q_len, hidden] BF16

    // RoPE scratch buffers. The existing `partial_rope` kernel expects
    // matched Q and K tokens; the drafter needs different positions for
    // Q (length q_len) and K (length ctx_len + q_len). We call the
    // kernel twice with the unused side pointed at a throwaway buffer.
    rope_k_scratch: CudaDeviceBuffer, // [q_len, kv_heads * head_dim] BF16
    rope_q_scratch: CudaDeviceBuffer, // [(ctx_len + q_len), q_heads * head_dim] BF16

    // Forward output. After `forward()` runs, this holds the post-norm
    // final hidden state shape [q_len, hidden] BF16.
    output: CudaDeviceBuffer,

    // cuBLASLt workspace + i32 position buffer (one entry per absolute
    // position in [0, ctx_len + q_len)).
    gemm_workspace: CudaDeviceBuffer,
    position_ids: CudaDeviceBuffer,

    bytes_total: usize,
}

#[derive(Debug, Clone)]
pub struct DrafterWorkspaceReport {
    pub q_len_max: usize,
    pub ctx_len_max: usize,
    pub gemm_workspace_bytes: usize,
    pub intermediates_bytes: usize,
    pub total_bytes: usize,
}

const GEMM_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const BF16_BYTES: usize = 2;
const I32_BYTES: usize = 4;

impl DrafterForwardWorkspace {
    pub fn alloc(
        config: &DFlashConfig,
        q_len_max: usize,
        ctx_len_max: usize,
    ) -> Result<Self> {
        if q_len_max == 0 {
            bail!("q_len_max must be > 0");
        }
        let kv_len_max = ctx_len_max + q_len_max;
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let q_heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_features = q_heads * head_dim;
        let kv_features = kv_heads * head_dim;

        let alloc = |bytes: usize, label: &str| -> Result<CudaDeviceBuffer> {
            CudaDeviceBuffer::alloc(bytes).map_err(|e| anyhow!("alloc {label}: {e}"))
        };

        let carry_a = alloc(q_len_max * hidden * BF16_BYTES, "carry_a")?;
        let carry_b = alloc(q_len_max * hidden * BF16_BYTES, "carry_b")?;
        let normed = alloc(q_len_max * hidden * BF16_BYTES, "normed")?;
        let q = alloc(q_len_max * q_features * BF16_BYTES, "q")?;
        let k_full = alloc(kv_len_max * kv_features * BF16_BYTES, "k_full")?;
        let v_full = alloc(kv_len_max * kv_features * BF16_BYTES, "v_full")?;
        let attn_out = alloc(q_len_max * q_features * BF16_BYTES, "attn_out")?;
        let attn_proj = alloc(q_len_max * hidden * BF16_BYTES, "attn_proj")?;
        let gate = alloc(q_len_max * intermediate * BF16_BYTES, "gate")?;
        let up = alloc(q_len_max * intermediate * BF16_BYTES, "up")?;
        let silu = alloc(q_len_max * intermediate * BF16_BYTES, "silu")?;
        let mlp_out = alloc(q_len_max * hidden * BF16_BYTES, "mlp_out")?;
        let rope_k_scratch = alloc(q_len_max * kv_features * BF16_BYTES, "rope_k_scratch")?;
        let rope_q_scratch = alloc(kv_len_max * q_features * BF16_BYTES, "rope_q_scratch")?;
        let output = alloc(q_len_max * hidden * BF16_BYTES, "output")?;
        let gemm_workspace = alloc(GEMM_WORKSPACE_BYTES, "gemm_workspace")?;
        let position_ids = alloc(kv_len_max * I32_BYTES, "position_ids")?;

        let bytes_total = carry_a.bytes()
            + carry_b.bytes()
            + normed.bytes()
            + q.bytes()
            + k_full.bytes()
            + v_full.bytes()
            + attn_out.bytes()
            + attn_proj.bytes()
            + gate.bytes()
            + up.bytes()
            + silu.bytes()
            + mlp_out.bytes()
            + rope_k_scratch.bytes()
            + rope_q_scratch.bytes()
            + output.bytes()
            + gemm_workspace.bytes()
            + position_ids.bytes();

        Ok(Self {
            q_len_max,
            ctx_len_max,
            hidden,
            intermediate,
            q_heads,
            kv_heads,
            head_dim,
            carry_a,
            carry_b,
            normed,
            q,
            k_full,
            v_full,
            attn_out,
            attn_proj,
            gate,
            up,
            silu,
            mlp_out,
            rope_k_scratch,
            rope_q_scratch,
            output,
            gemm_workspace,
            position_ids,
            bytes_total,
        })
    }

    pub fn report(&self) -> DrafterWorkspaceReport {
        DrafterWorkspaceReport {
            q_len_max: self.q_len_max,
            ctx_len_max: self.ctx_len_max,
            gemm_workspace_bytes: self.gemm_workspace.bytes(),
            intermediates_bytes: self.bytes_total - self.gemm_workspace.bytes(),
            total_bytes: self.bytes_total,
        }
    }

    pub fn position_ids_ptr(&self) -> DevicePtr {
        self.position_ids.ptr()
    }

    pub fn position_ids_buffer(&self) -> &CudaDeviceBuffer {
        &self.position_ids
    }

    pub fn output_ptr(&self) -> DevicePtr {
        self.output.ptr()
    }

    pub fn output_buffer(&self) -> &CudaDeviceBuffer {
        &self.output
    }

    pub fn position_ids_capacity(&self) -> usize {
        self.position_ids.bytes() / I32_BYTES
    }
}

/// Drafter forward engine. Holds borrowed references to the GPU
/// weights and an owned `DrafterForwardWorkspace`.
pub struct DrafterForward<'w> {
    device: &'w DFlashDrafterDevice,
    config: &'w DFlashConfig,
    workspace: DrafterForwardWorkspace,
}

impl<'w> DrafterForward<'w> {
    pub fn new(
        device: &'w DFlashDrafterDevice,
        config: &'w DFlashConfig,
        workspace: DrafterForwardWorkspace,
    ) -> Result<Self> {
        if workspace.hidden != config.hidden_size
            || workspace.q_heads != config.num_attention_heads
            || workspace.kv_heads != config.num_key_value_heads
            || workspace.head_dim != config.head_dim
            || workspace.intermediate != config.intermediate_size
        {
            bail!("workspace dimensions do not match drafter config");
        }
        if device.layers.len() != config.num_hidden_layers {
            bail!(
                "device exposes {} layers but config has {}",
                device.layers.len(),
                config.num_hidden_layers,
            );
        }
        Ok(Self {
            device,
            config,
            workspace,
        })
    }

    pub fn workspace(&self) -> &DrafterForwardWorkspace {
        &self.workspace
    }

    /// One drafter forward. Caller-owned inputs:
    ///   - `noise_embedding`: `[q_len, hidden]` BF16
    ///   - `target_hidden_collapsed`: `[ctx_len, hidden]` BF16, already
    ///     passed through `fc` + `hidden_norm` (Phase D.2 will wrap
    ///     this).
    ///
    /// `self.workspace.position_ids` must already hold the
    /// `ctx_len + q_len` absolute positions `[0, 1, …, ctx_len + q_len
    /// - 1]` as i32. The caller fills it once at session start; it's
    /// stable across drafter forwards as long as the prefix layout
    /// doesn't shift.
    ///
    /// Output lands in `self.workspace.output` (`[q_len, hidden]`
    /// BF16, post final `norm`).
    pub fn forward<B: KernelBackend>(
        &mut self,
        backend: &B,
        noise_embedding: DevicePtr,
        target_hidden_collapsed: DevicePtr,
        q_len: usize,
        ctx_len: usize,
    ) -> Result<DevicePtr> {
        if q_len == 0 || q_len > self.workspace.q_len_max {
            bail!("q_len {q_len} out of range (max {})", self.workspace.q_len_max);
        }
        if ctx_len > self.workspace.ctx_len_max {
            bail!(
                "ctx_len {ctx_len} out of range (max {})",
                self.workspace.ctx_len_max,
            );
        }

        // Residual chain: `carry` is the live residual pointer; we
        // alternate between `carry_a` and `carry_b` on each fused
        // RMSNorm+residual_add call.
        let mut carry: DevicePtr = noise_embedding;
        let mut carry_buf_idx: u8 = 0; // 0 = use carry_a next, 1 = carry_b

        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &self.device.layers[layer_idx];

            // --- Attention sub-block ------------------------------
            // Layer 0 needs a standalone input_layernorm: `carry`
            // still points at the caller-owned `noise_embedding`
            // (read-only) and there is no prior residual to fuse.
            // Layer N>0 inherits `normed` from the previous
            // iteration's fused `mlp_out → next_input_layernorm`
            // call, so the input rmsnorm is already done.
            if layer_idx == 0 {
                backend
                    .rmsnorm(&RmsNormSpec {
                        rows: q_len,
                        hidden: self.workspace.hidden,
                        eps: self.config.rms_norm_eps,
                        input_bf16: carry,
                        weight_bf16: layer.input_layernorm.ptr,
                        residual_bf16: DevicePtr::NULL,
                        residual_out_bf16: DevicePtr::NULL,
                        output_bf16: self.workspace.normed.ptr(),
                        direct_weight: true,
                    })
                    .map_err(|e| anyhow!("layer 0 input_layernorm: {e}"))?;
            }

            self.attention_block(
                backend,
                layer,
                q_len,
                ctx_len,
                target_hidden_collapsed,
            )
            .with_context(|| format!("layer {layer_idx} attention"))?;

            // Post-attention RMSNorm fused with residual update.
            // residual_out := carry + attn_proj; normed := rmsnorm
            // of that tensor weighted by post_attention_layernorm.
            let next_carry_ptr = self.next_carry_ptr(carry_buf_idx);
            backend
                .rmsnorm(&RmsNormSpec {
                    rows: q_len,
                    hidden: self.workspace.hidden,
                    eps: self.config.rms_norm_eps,
                    input_bf16: self.workspace.attn_proj.ptr(),
                    weight_bf16: layer.post_attention_layernorm.ptr,
                    residual_bf16: carry,
                    residual_out_bf16: next_carry_ptr,
                    output_bf16: self.workspace.normed.ptr(),
                    direct_weight: true,
                })
                .map_err(|e| anyhow!("layer {layer_idx} post_attn rmsnorm: {e}"))?;
            carry = next_carry_ptr;
            carry_buf_idx ^= 1;

            self.mlp_block(backend, layer, q_len)
                .with_context(|| format!("layer {layer_idx} mlp"))?;

            if layer_idx + 1 < self.config.num_hidden_layers {
                // Fuse residual add into next layer's input_layernorm:
                // residual_out := carry + mlp_out; normed := rmsnorm
                // weighted by the next layer's input_layernorm. The
                // next iteration's attention block reads `normed` as-is.
                let next_carry_ptr = self.next_carry_ptr(carry_buf_idx);
                backend
                    .rmsnorm(&RmsNormSpec {
                        rows: q_len,
                        hidden: self.workspace.hidden,
                        eps: self.config.rms_norm_eps,
                        input_bf16: self.workspace.mlp_out.ptr(),
                        weight_bf16: self.device.layers[layer_idx + 1].input_layernorm.ptr,
                        residual_bf16: carry,
                        residual_out_bf16: next_carry_ptr,
                        output_bf16: self.workspace.normed.ptr(),
                        direct_weight: true,
                    })
                    .map_err(|e| {
                        anyhow!("layer {layer_idx} mlp→next input fused rmsnorm: {e}")
                    })?;
                carry = next_carry_ptr;
                carry_buf_idx ^= 1;
            } else {
                // Final layer: fuse the last residual into the model's
                // top-level `norm`. The output is the drafter's final
                // hidden state, ready for the target's `lm_head`.
                backend
                    .rmsnorm(&RmsNormSpec {
                        rows: q_len,
                        hidden: self.workspace.hidden,
                        eps: self.config.rms_norm_eps,
                        input_bf16: self.workspace.mlp_out.ptr(),
                        weight_bf16: self.device.norm.ptr,
                        residual_bf16: carry,
                        residual_out_bf16: DevicePtr::NULL,
                        output_bf16: self.workspace.output.ptr(),
                        direct_weight: true,
                    })
                    .map_err(|e| anyhow!("final norm: {e}"))?;
            }
        }

        Ok(self.workspace.output.ptr())
    }

    fn next_carry_ptr(&self, idx: u8) -> DevicePtr {
        if idx == 0 {
            self.workspace.carry_a.ptr()
        } else {
            self.workspace.carry_b.ptr()
        }
    }

    fn attention_block<B: KernelBackend>(
        &mut self,
        backend: &B,
        layer: &DFlashLayerDevice,
        q_len: usize,
        ctx_len: usize,
        target_hidden_collapsed: DevicePtr,
    ) -> Result<()> {
        let hidden = self.workspace.hidden;
        let q_features = self.workspace.q_heads * self.workspace.head_dim;
        let kv_features = self.workspace.kv_heads * self.workspace.head_dim;
        let kv_seq_len = ctx_len + q_len;

        // Q projection from `normed` (the input_layernorm output).
        gemm(backend, &self.workspace, self.workspace.normed.ptr(), layer.q_proj, q_len, hidden, q_features, self.workspace.q.ptr())?;

        // K and V projections: stack k_ctx (target_hidden) + k_noise
        // (normed) directly into `k_full` so no concat copy is needed.
        // Same for V.
        let k_full_ctx_ptr = self.workspace.k_full.ptr();
        let k_full_noise_ptr = self
            .workspace
            .k_full
            .ptr_at(ctx_len * kv_features * BF16_BYTES)
            .map_err(|e| anyhow!("k_full noise offset: {e}"))?;
        let v_full_ctx_ptr = self.workspace.v_full.ptr();
        let v_full_noise_ptr = self
            .workspace
            .v_full
            .ptr_at(ctx_len * kv_features * BF16_BYTES)
            .map_err(|e| anyhow!("v_full noise offset: {e}"))?;

        if ctx_len > 0 {
            gemm(backend, &self.workspace, target_hidden_collapsed, layer.k_proj, ctx_len, hidden, kv_features, k_full_ctx_ptr)?;
            gemm(backend, &self.workspace, target_hidden_collapsed, layer.v_proj, ctx_len, hidden, kv_features, v_full_ctx_ptr)?;
        }
        gemm(backend, &self.workspace, self.workspace.normed.ptr(), layer.k_proj, q_len, hidden, kv_features, k_full_noise_ptr)?;
        gemm(backend, &self.workspace, self.workspace.normed.ptr(), layer.v_proj, q_len, hidden, kv_features, v_full_noise_ptr)?;

        // Q norm: rmsnorm over head_dim, viewed as [q_len * q_heads,
        // head_dim] rows. Same for K norm with [(kv_seq_len * kv_heads),
        // head_dim].
        backend
            .rmsnorm(&RmsNormSpec {
                rows: q_len * self.workspace.q_heads,
                hidden: self.workspace.head_dim,
                eps: self.config.rms_norm_eps,
                input_bf16: self.workspace.q.ptr(),
                weight_bf16: layer.q_norm.ptr,
                residual_bf16: DevicePtr::NULL,
                residual_out_bf16: DevicePtr::NULL,
                output_bf16: self.workspace.q.ptr(),
                direct_weight: true,
            })
            .map_err(|e| anyhow!("q_norm: {e}"))?;
        backend
            .rmsnorm(&RmsNormSpec {
                rows: kv_seq_len * self.workspace.kv_heads,
                hidden: self.workspace.head_dim,
                eps: self.config.rms_norm_eps,
                input_bf16: self.workspace.k_full.ptr(),
                weight_bf16: layer.k_norm.ptr,
                residual_bf16: DevicePtr::NULL,
                residual_out_bf16: DevicePtr::NULL,
                output_bf16: self.workspace.k_full.ptr(),
                direct_weight: true,
            })
            .map_err(|e| anyhow!("k_norm: {e}"))?;

        // RoPE — two calls because q and k have different lengths and
        // different position ranges. Position buffer is filled by the
        // caller with absolute positions [0, kv_seq_len).
        //
        // Call 1: rotate Q at positions [ctx_len, ctx_len + q_len).
        let q_positions_ptr = self
            .workspace
            .position_ids
            .ptr_at(ctx_len * I32_BYTES)
            .map_err(|e| anyhow!("q_positions offset: {e}"))?;
        backend
            .partial_rope(&PartialRopeSpec {
                tokens: q_len,
                q_heads: self.workspace.q_heads,
                kv_heads: self.workspace.kv_heads,
                head_dim: self.workspace.head_dim,
                rope_dims: self.workspace.head_dim, // full RoPE
                base_theta: self.config.rope_theta,
                position_i32: 0,
                use_scalar_position: false,
                positions_i32: q_positions_ptr,
                q_bf16: self.workspace.q.ptr(),
                k_bf16: self.workspace.rope_k_scratch.ptr(),
                scalar_position_device_i32: DevicePtr::NULL,
            })
            .map_err(|e| anyhow!("rope q: {e}"))?;
        // Call 2: rotate K at positions [0, kv_seq_len).
        backend
            .partial_rope(&PartialRopeSpec {
                tokens: kv_seq_len,
                q_heads: self.workspace.q_heads,
                kv_heads: self.workspace.kv_heads,
                head_dim: self.workspace.head_dim,
                rope_dims: self.workspace.head_dim,
                base_theta: self.config.rope_theta,
                position_i32: 0,
                use_scalar_position: false,
                positions_i32: self.workspace.position_ids.ptr(),
                q_bf16: self.workspace.rope_q_scratch.ptr(),
                k_bf16: self.workspace.k_full.ptr(),
                scalar_position_device_i32: DevicePtr::NULL,
            })
            .map_err(|e| anyhow!("rope k: {e}"))?;

        // Attention.
        let sliding_window = match layer.kind {
            LayerAttentionKind::SlidingAttention => self.config.sliding_window,
            LayerAttentionKind::FullAttention => 0,
        };
        backend
            .drafter_attention_block_bf16(&DrafterAttentionBlockSpec {
                q_bf16: self.workspace.q.ptr(),
                k_bf16: self.workspace.k_full.ptr(),
                v_bf16: self.workspace.v_full.ptr(),
                output_bf16: self.workspace.attn_out.ptr(),
                q_len,
                kv_seq_len,
                q_heads: self.workspace.q_heads,
                kv_heads: self.workspace.kv_heads,
                head_dim: self.workspace.head_dim,
                sliding_window,
            })
            .map_err(|e| anyhow!("attention: {e}"))?;

        // O projection.
        gemm(backend, &self.workspace, self.workspace.attn_out.ptr(), layer.o_proj, q_len, q_features, hidden, self.workspace.attn_proj.ptr())?;
        Ok(())
    }

    fn mlp_block<B: KernelBackend>(
        &mut self,
        backend: &B,
        layer: &DFlashLayerDevice,
        q_len: usize,
    ) -> Result<()> {
        let hidden = self.workspace.hidden;
        let intermediate = self.workspace.intermediate;

        // Gate + Up projections from the post-attn-normed buffer.
        gemm(backend, &self.workspace, self.workspace.normed.ptr(), layer.mlp_gate_proj, q_len, hidden, intermediate, self.workspace.gate.ptr())?;
        gemm(backend, &self.workspace, self.workspace.normed.ptr(), layer.mlp_up_proj, q_len, hidden, intermediate, self.workspace.up.ptr())?;

        // SwiGLU.
        backend
            .swiglu(&SwiGluSpec {
                rows: q_len,
                intermediate,
                gate_bf16: self.workspace.gate.ptr(),
                up_bf16: self.workspace.up.ptr(),
                output_bf16: self.workspace.silu.ptr(),
            })
            .map_err(|e| anyhow!("swiglu: {e}"))?;

        // Down projection.
        gemm(backend, &self.workspace, self.workspace.silu.ptr(), layer.mlp_down_proj, q_len, intermediate, hidden, self.workspace.mlp_out.ptr())?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm<B: KernelBackend>(
    backend: &B,
    workspace: &DrafterForwardWorkspace,
    input: DevicePtr,
    weight: TensorOnDevice,
    rows: usize,
    in_features: usize,
    out_features: usize,
    output: DevicePtr,
) -> Result<()> {
    backend
        .bf16_gemm(&Bf16GemmSpec {
            m: out_features,
            n: rows,
            k: in_features,
            a_bf16: weight.ptr,
            b_bf16: input,
            c_bf16: output,
            workspace: workspace.gemm_workspace.ptr(),
            workspace_bytes: workspace.gemm_workspace.bytes(),
        })
        .map_err(|e| anyhow!("bf16_gemm({rows}x{in_features}→{out_features}): {e}"))
}
