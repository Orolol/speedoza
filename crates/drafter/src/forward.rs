//! DFlash drafter forward pass (Phase D v2 — D.1 + D.2 + D.3).
//!
//! Runs the 5-layer drafter model end-to-end on GPU using existing
//! BF16 kernels (RMSNorm, BF16 GEMM, SwiGLU, partial RoPE) plus the
//! `drafter_attention_block_bf16` kernel from Phase C. Includes:
//!
//! - **D.2** — internal `fc` + `hidden_norm` collapse: `forward` accepts
//!   `target_hidden_raw` of shape `[ctx_len, hidden * n_target_layers]`
//!   and produces the post-`hidden_norm` collapsed tensor on the fly.
//! - **D.3** — per-layer KV cache. Each call appends `ctx_len + q_len`
//!   new K/V entries at offset `current_kv_len` and attention reads
//!   the full live cache up to `current_kv_len + ctx_len + q_len`.
//!   Controller managed via [`DrafterForward::reset_kv_cache`] /
//!   [`DrafterForward::crop_kv_cache`].
//!
//! Residual chain still uses RMSNorm + residual fusion to avoid a
//! standalone vector-add kernel.

use anyhow::{Context, Result, anyhow, bail};
use qwen36_fp4_kernels::{
    Bf16GemmSpec, CudaDeviceBuffer, DevicePtr, DrafterAttentionBlockSpec, KernelBackend,
    PartialRopeSpec, RmsNormSpec, SwiGluSpec,
};

use crate::dflash::{DFlashConfig, LayerAttentionKind};
use crate::gpu::{DFlashDrafterDevice, DFlashLayerDevice, TensorOnDevice};

/// Pre-allocated intermediate buffers for one drafter forward.
pub struct DrafterForwardWorkspace {
    pub q_len_max: usize,
    pub ctx_len_max: usize,
    pub kv_cache_max_len: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub fc_in_features: usize,

    // Residual ping-pong carries.
    carry_a: CudaDeviceBuffer,
    carry_b: CudaDeviceBuffer,

    // Target-hidden collapse: `target_raw` (uncollapsed,
    // `[ctx_len, hidden * n_target]`) → `fc` → `target_collapsed_pre_norm`
    // → `hidden_norm` → `target_collapsed`. Two buffers because we keep
    // the layout simple (rmsnorm reads from pre, writes to collapsed).
    target_collapsed_pre_norm: CudaDeviceBuffer, // [ctx_len, hidden]
    target_collapsed: CudaDeviceBuffer,          // [ctx_len, hidden]

    // Per-layer intermediates.
    normed: CudaDeviceBuffer,
    q: CudaDeviceBuffer,
    attn_out: CudaDeviceBuffer,
    attn_proj: CudaDeviceBuffer,
    gate: CudaDeviceBuffer,
    up: CudaDeviceBuffer,
    silu: CudaDeviceBuffer,
    mlp_out: CudaDeviceBuffer,

    // RoPE scratch buffers. Sized for the worst-case "new K" range
    // (ctx_len + q_len) since that's the only slice RoPE writes per
    // forward.
    rope_k_scratch: CudaDeviceBuffer,
    rope_q_scratch: CudaDeviceBuffer,

    // Per-layer (K, V) caches. Each buffer holds
    // `[kv_cache_max_len, kv_heads * head_dim]` BF16 (logically
    // `[kv_cache_max_len, kv_heads, head_dim]` row-major).
    kv_caches: Vec<(CudaDeviceBuffer, CudaDeviceBuffer)>,

    // Final post-norm hidden state, [q_len, hidden] BF16.
    output: CudaDeviceBuffer,

    // cuBLASLt workspace.
    gemm_workspace: CudaDeviceBuffer,
    // Absolute position ids (i32) for [0, kv_cache_max_len). Caller
    // pre-fills the prefix it needs.
    position_ids: CudaDeviceBuffer,

    bytes_total: usize,
}

#[derive(Debug, Clone)]
pub struct DrafterWorkspaceReport {
    pub q_len_max: usize,
    pub ctx_len_max: usize,
    pub kv_cache_max_len: usize,
    pub gemm_workspace_bytes: usize,
    pub kv_caches_bytes: usize,
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
        kv_cache_max_len: usize,
    ) -> Result<Self> {
        if q_len_max == 0 {
            bail!("q_len_max must be > 0");
        }
        let new_kv_max = ctx_len_max + q_len_max;
        if kv_cache_max_len < new_kv_max {
            bail!(
                "kv_cache_max_len ({kv_cache_max_len}) must be ≥ ctx_len_max + q_len_max ({new_kv_max})"
            );
        }
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let q_heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_features = q_heads * head_dim;
        let kv_features = kv_heads * head_dim;
        let fc_in_features = config.fc_in_features();

        let alloc = |bytes: usize, label: &str| -> Result<CudaDeviceBuffer> {
            CudaDeviceBuffer::alloc(bytes).map_err(|e| anyhow!("alloc {label}: {e}"))
        };

        let carry_a = alloc(q_len_max * hidden * BF16_BYTES, "carry_a")?;
        let carry_b = alloc(q_len_max * hidden * BF16_BYTES, "carry_b")?;
        let target_collapsed_pre_norm = alloc(
            ctx_len_max * hidden * BF16_BYTES,
            "target_collapsed_pre_norm",
        )?;
        let target_collapsed = alloc(ctx_len_max * hidden * BF16_BYTES, "target_collapsed")?;
        let normed = alloc(q_len_max * hidden * BF16_BYTES, "normed")?;
        let q = alloc(q_len_max * q_features * BF16_BYTES, "q")?;
        let attn_out = alloc(q_len_max * q_features * BF16_BYTES, "attn_out")?;
        let attn_proj = alloc(q_len_max * hidden * BF16_BYTES, "attn_proj")?;
        let gate = alloc(q_len_max * intermediate * BF16_BYTES, "gate")?;
        let up = alloc(q_len_max * intermediate * BF16_BYTES, "up")?;
        let silu = alloc(q_len_max * intermediate * BF16_BYTES, "silu")?;
        let mlp_out = alloc(q_len_max * hidden * BF16_BYTES, "mlp_out")?;
        let rope_k_scratch = alloc(new_kv_max * kv_features * BF16_BYTES, "rope_k_scratch")?;
        let rope_q_scratch = alloc(new_kv_max * q_features * BF16_BYTES, "rope_q_scratch")?;
        let output = alloc(q_len_max * hidden * BF16_BYTES, "output")?;
        let gemm_workspace = alloc(GEMM_WORKSPACE_BYTES, "gemm_workspace")?;
        let position_ids = alloc(kv_cache_max_len * I32_BYTES, "position_ids")?;

        let kv_caches_bytes_per_layer = kv_cache_max_len * kv_features * BF16_BYTES;
        let mut kv_caches: Vec<(CudaDeviceBuffer, CudaDeviceBuffer)> =
            Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let k = alloc(
                kv_caches_bytes_per_layer,
                &format!("kv_cache_K[{layer_idx}]"),
            )?;
            let v = alloc(
                kv_caches_bytes_per_layer,
                &format!("kv_cache_V[{layer_idx}]"),
            )?;
            kv_caches.push((k, v));
        }
        let kv_caches_total = kv_caches_bytes_per_layer * 2 * config.num_hidden_layers;

        let bytes_total = carry_a.bytes()
            + carry_b.bytes()
            + target_collapsed_pre_norm.bytes()
            + target_collapsed.bytes()
            + normed.bytes()
            + q.bytes()
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
            + position_ids.bytes()
            + kv_caches_total;

        Ok(Self {
            q_len_max,
            ctx_len_max,
            kv_cache_max_len,
            hidden,
            intermediate,
            q_heads,
            kv_heads,
            head_dim,
            fc_in_features,
            carry_a,
            carry_b,
            target_collapsed_pre_norm,
            target_collapsed,
            normed,
            q,
            attn_out,
            attn_proj,
            gate,
            up,
            silu,
            mlp_out,
            rope_k_scratch,
            rope_q_scratch,
            kv_caches,
            output,
            gemm_workspace,
            position_ids,
            bytes_total,
        })
    }

    pub fn report(&self) -> DrafterWorkspaceReport {
        let kv_caches_bytes: usize = self
            .kv_caches
            .iter()
            .map(|(k, v)| k.bytes() + v.bytes())
            .sum();
        DrafterWorkspaceReport {
            q_len_max: self.q_len_max,
            ctx_len_max: self.ctx_len_max,
            kv_cache_max_len: self.kv_cache_max_len,
            gemm_workspace_bytes: self.gemm_workspace.bytes(),
            kv_caches_bytes,
            intermediates_bytes: self.bytes_total - self.gemm_workspace.bytes() - kv_caches_bytes,
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

/// Drafter forward engine. Holds borrowed weights and an owned
/// workspace plus the live KV-cache length.
pub struct DrafterForward<'w> {
    device: &'w DFlashDrafterDevice,
    config: &'w DFlashConfig,
    workspace: DrafterForwardWorkspace,
    current_kv_len: usize,
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
            || workspace.fc_in_features != config.fc_in_features()
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
        if workspace.kv_caches.len() != config.num_hidden_layers {
            bail!(
                "workspace has {} kv caches but config has {} layers",
                workspace.kv_caches.len(),
                config.num_hidden_layers,
            );
        }
        Ok(Self {
            device,
            config,
            workspace,
            current_kv_len: 0,
        })
    }

    pub fn workspace(&self) -> &DrafterForwardWorkspace {
        &self.workspace
    }

    pub fn current_kv_len(&self) -> usize {
        self.current_kv_len
    }

    /// Mirror of `DynamicCache.crop(new_len)`. Logically truncates the
    /// per-layer KV caches; data past `new_len` is left in place and
    /// will be overwritten by the next `forward` append.
    pub fn crop_kv_cache(&mut self, new_len: usize) -> Result<()> {
        if new_len > self.current_kv_len {
            bail!(
                "crop_kv_cache({new_len}) > current_kv_len {}",
                self.current_kv_len,
            );
        }
        self.current_kv_len = new_len;
        Ok(())
    }

    pub fn reset_kv_cache(&mut self) {
        self.current_kv_len = 0;
    }

    /// Run one drafter forward.
    ///
    /// Inputs (caller-owned device pointers, all BF16 row-major):
    ///   - `noise_embedding`: `[q_len, hidden]`.
    ///   - `target_hidden_raw`: `[ctx_len, hidden * n_target_layers]`
    ///     (the uncollapsed concatenation of the target's selected layer
    ///     hidden states).
    ///
    /// The workspace's `position_ids` buffer must hold absolute
    /// positions `[0, current_kv_len + ctx_len + q_len)` as i32 before
    /// the call. The forward writes to positions
    /// `[current_kv_len, current_kv_len + ctx_len + q_len)` in the KV
    /// cache and reads RoPE values from the corresponding position_ids
    /// slice. After return, `current_kv_len` is advanced by
    /// `ctx_len + q_len`; the controller cancels speculative entries
    /// via `crop_kv_cache(committed_len)`.
    ///
    /// Output lands in `workspace.output` (`[q_len, hidden]` BF16, post
    /// final `norm`).
    pub fn forward<B: KernelBackend>(
        &mut self,
        backend: &B,
        noise_embedding: DevicePtr,
        target_hidden_raw: DevicePtr,
        q_len: usize,
        ctx_len: usize,
    ) -> Result<DevicePtr> {
        if q_len == 0 || q_len > self.workspace.q_len_max {
            bail!(
                "q_len {q_len} out of range (max {})",
                self.workspace.q_len_max
            );
        }
        if ctx_len > self.workspace.ctx_len_max {
            bail!(
                "ctx_len {ctx_len} out of range (max {})",
                self.workspace.ctx_len_max,
            );
        }
        let kv_seq_len = self.current_kv_len + ctx_len + q_len;
        if kv_seq_len > self.workspace.kv_cache_max_len {
            bail!(
                "kv_seq_len {kv_seq_len} exceeds kv_cache_max_len {}",
                self.workspace.kv_cache_max_len,
            );
        }
        if ctx_len == 0 && self.current_kv_len == 0 {
            bail!(
                "drafter forward needs either ctx_len > 0 or a non-empty KV cache; both are zero"
            );
        }

        // --- Target hidden collapse: fc + hidden_norm (Phase D.2) -----
        if ctx_len > 0 {
            gemm(
                backend,
                &self.workspace,
                target_hidden_raw,
                self.device.fc,
                ctx_len,
                self.workspace.fc_in_features,
                self.workspace.hidden,
                self.workspace.target_collapsed_pre_norm.ptr(),
            )?;
            backend
                .rmsnorm(&RmsNormSpec {
                    rows: ctx_len,
                    hidden: self.workspace.hidden,
                    eps: self.config.rms_norm_eps,
                    input_bf16: self.workspace.target_collapsed_pre_norm.ptr(),
                    weight_bf16: self.device.hidden_norm.ptr,
                    residual_bf16: DevicePtr::NULL,
                    residual_out_bf16: DevicePtr::NULL,
                    output_bf16: self.workspace.target_collapsed.ptr(),
                    direct_weight: true,
                })
                .map_err(|e| anyhow!("hidden_norm: {e}"))?;
        }
        let target_collapsed_ptr = self.workspace.target_collapsed.ptr();

        let mut carry: DevicePtr = noise_embedding;
        let mut carry_buf_idx: u8 = 0;

        for layer_idx in 0..self.config.num_hidden_layers {
            let layer_kind = self.device.layers[layer_idx].kind;
            let input_layernorm_ptr = self.device.layers[layer_idx].input_layernorm.ptr;
            let post_attn_layernorm_ptr =
                self.device.layers[layer_idx].post_attention_layernorm.ptr;
            let next_input_layernorm_ptr = if layer_idx + 1 < self.config.num_hidden_layers {
                Some(self.device.layers[layer_idx + 1].input_layernorm.ptr)
            } else {
                None
            };

            if layer_idx == 0 {
                backend
                    .rmsnorm(&RmsNormSpec {
                        rows: q_len,
                        hidden: self.workspace.hidden,
                        eps: self.config.rms_norm_eps,
                        input_bf16: carry,
                        weight_bf16: input_layernorm_ptr,
                        residual_bf16: DevicePtr::NULL,
                        residual_out_bf16: DevicePtr::NULL,
                        output_bf16: self.workspace.normed.ptr(),
                        direct_weight: true,
                    })
                    .map_err(|e| anyhow!("layer 0 input_layernorm: {e}"))?;
            }

            self.attention_block(backend, layer_idx, q_len, ctx_len, target_collapsed_ptr)
                .with_context(|| format!("layer {layer_idx} attention"))?;
            let _ = layer_kind;

            // Post-attention RMSNorm fused with residual update.
            let next_carry_ptr = self.next_carry_ptr(carry_buf_idx);
            backend
                .rmsnorm(&RmsNormSpec {
                    rows: q_len,
                    hidden: self.workspace.hidden,
                    eps: self.config.rms_norm_eps,
                    input_bf16: self.workspace.attn_proj.ptr(),
                    weight_bf16: post_attn_layernorm_ptr,
                    residual_bf16: carry,
                    residual_out_bf16: next_carry_ptr,
                    output_bf16: self.workspace.normed.ptr(),
                    direct_weight: true,
                })
                .map_err(|e| anyhow!("layer {layer_idx} post_attn rmsnorm: {e}"))?;
            carry = next_carry_ptr;
            carry_buf_idx ^= 1;

            self.mlp_block(backend, layer_idx, q_len)
                .with_context(|| format!("layer {layer_idx} mlp"))?;

            if let Some(next_input_ptr) = next_input_layernorm_ptr {
                let next_carry_ptr = self.next_carry_ptr(carry_buf_idx);
                backend
                    .rmsnorm(&RmsNormSpec {
                        rows: q_len,
                        hidden: self.workspace.hidden,
                        eps: self.config.rms_norm_eps,
                        input_bf16: self.workspace.mlp_out.ptr(),
                        weight_bf16: next_input_ptr,
                        residual_bf16: carry,
                        residual_out_bf16: next_carry_ptr,
                        output_bf16: self.workspace.normed.ptr(),
                        direct_weight: true,
                    })
                    .map_err(|e| anyhow!("layer {layer_idx} mlp→next input fused rmsnorm: {e}"))?;
                carry = next_carry_ptr;
                carry_buf_idx ^= 1;
            } else {
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

        // Advance the KV-cache cursor; the controller can `crop_kv_cache`
        // to discard rejected speculative entries.
        self.current_kv_len += ctx_len + q_len;
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
        layer_idx: usize,
        q_len: usize,
        ctx_len: usize,
        target_collapsed_ptr: DevicePtr,
    ) -> Result<()> {
        let layer: &DFlashLayerDevice = &self.device.layers[layer_idx];
        let hidden = self.workspace.hidden;
        let q_features = self.workspace.q_heads * self.workspace.head_dim;
        let kv_features = self.workspace.kv_heads * self.workspace.head_dim;
        let head_dim = self.workspace.head_dim;
        let q_heads = self.workspace.q_heads;
        let kv_heads = self.workspace.kv_heads;
        let past_len = self.current_kv_len;
        let new_kv_len = ctx_len + q_len;
        let kv_seq_len = past_len + new_kv_len;

        // Q projection from `normed`.
        gemm(
            backend,
            &self.workspace,
            self.workspace.normed.ptr(),
            layer.q_proj,
            q_len,
            hidden,
            q_features,
            self.workspace.q.ptr(),
        )?;

        // Per-layer KV cache pointers.
        let (k_cache_buf, v_cache_buf) = &self.workspace.kv_caches[layer_idx];
        let kv_row_bytes = kv_features * BF16_BYTES;
        let k_new_ctx_ptr = k_cache_buf
            .ptr_at(past_len * kv_row_bytes)
            .map_err(|e| anyhow!("k_cache offset (ctx slot): {e}"))?;
        let k_new_noise_ptr = k_cache_buf
            .ptr_at((past_len + ctx_len) * kv_row_bytes)
            .map_err(|e| anyhow!("k_cache offset (noise slot): {e}"))?;
        let v_new_ctx_ptr = v_cache_buf
            .ptr_at(past_len * kv_row_bytes)
            .map_err(|e| anyhow!("v_cache offset (ctx slot): {e}"))?;
        let v_new_noise_ptr = v_cache_buf
            .ptr_at((past_len + ctx_len) * kv_row_bytes)
            .map_err(|e| anyhow!("v_cache offset (noise slot): {e}"))?;

        if ctx_len > 0 {
            gemm(
                backend,
                &self.workspace,
                target_collapsed_ptr,
                layer.k_proj,
                ctx_len,
                hidden,
                kv_features,
                k_new_ctx_ptr,
            )?;
            gemm(
                backend,
                &self.workspace,
                target_collapsed_ptr,
                layer.v_proj,
                ctx_len,
                hidden,
                kv_features,
                v_new_ctx_ptr,
            )?;
        }
        gemm(
            backend,
            &self.workspace,
            self.workspace.normed.ptr(),
            layer.k_proj,
            q_len,
            hidden,
            kv_features,
            k_new_noise_ptr,
        )?;
        gemm(
            backend,
            &self.workspace,
            self.workspace.normed.ptr(),
            layer.v_proj,
            q_len,
            hidden,
            kv_features,
            v_new_noise_ptr,
        )?;

        // q_norm over head_dim (per token, per head).
        backend
            .rmsnorm(&RmsNormSpec {
                rows: q_len * q_heads,
                hidden: head_dim,
                eps: self.config.rms_norm_eps,
                input_bf16: self.workspace.q.ptr(),
                weight_bf16: layer.q_norm.ptr,
                residual_bf16: DevicePtr::NULL,
                residual_out_bf16: DevicePtr::NULL,
                output_bf16: self.workspace.q.ptr(),
                direct_weight: true,
            })
            .map_err(|e| anyhow!("q_norm: {e}"))?;
        // k_norm only on the NEW entries (rows past_len..kv_seq_len).
        let k_new_ptr = k_cache_buf
            .ptr_at(past_len * kv_row_bytes)
            .map_err(|e| anyhow!("k_cache slice for k_norm: {e}"))?;
        backend
            .rmsnorm(&RmsNormSpec {
                rows: new_kv_len * kv_heads,
                hidden: head_dim,
                eps: self.config.rms_norm_eps,
                input_bf16: k_new_ptr,
                weight_bf16: layer.k_norm.ptr,
                residual_bf16: DevicePtr::NULL,
                residual_out_bf16: DevicePtr::NULL,
                output_bf16: k_new_ptr,
                direct_weight: true,
            })
            .map_err(|e| anyhow!("k_norm: {e}"))?;

        // RoPE — two calls because Q and K have different positions.
        // Position buffer carries absolute positions [0, kv_cache_max_len).
        // Q positions: [past_len + ctx_len, kv_seq_len).
        let q_positions_ptr = self
            .workspace
            .position_ids
            .ptr_at((past_len + ctx_len) * I32_BYTES)
            .map_err(|e| anyhow!("q_positions offset: {e}"))?;
        backend
            .partial_rope(&PartialRopeSpec {
                tokens: q_len,
                q_heads,
                kv_heads,
                head_dim,
                rope_dims: head_dim,
                base_theta: self.config.rope_theta,
                position_i32: 0,
                use_scalar_position: false,
                positions_i32: q_positions_ptr,
                q_bf16: self.workspace.q.ptr(),
                k_bf16: self.workspace.rope_k_scratch.ptr(),
                scalar_position_device_i32: DevicePtr::NULL,
            })
            .map_err(|e| anyhow!("rope q: {e}"))?;
        // K positions: [past_len, kv_seq_len). Applies to the NEW
        // entries only — the cached prefix was already RoPE'd by prior
        // forwards.
        let k_positions_ptr = self
            .workspace
            .position_ids
            .ptr_at(past_len * I32_BYTES)
            .map_err(|e| anyhow!("k_positions offset: {e}"))?;
        backend
            .partial_rope(&PartialRopeSpec {
                tokens: new_kv_len,
                q_heads,
                kv_heads,
                head_dim,
                rope_dims: head_dim,
                base_theta: self.config.rope_theta,
                position_i32: 0,
                use_scalar_position: false,
                positions_i32: k_positions_ptr,
                q_bf16: self.workspace.rope_q_scratch.ptr(),
                k_bf16: k_new_ptr,
                scalar_position_device_i32: DevicePtr::NULL,
            })
            .map_err(|e| anyhow!("rope k: {e}"))?;

        // Attention reads the full live KV slice [0, kv_seq_len).
        //
        // Long-context AL probes (see 2026-06-09 strategic assessment §3):
        // QWEN36_DRAFTER_SWA_WINDOW=N overrides the checkpoint's sliding
        // window size on the sliding layers; QWEN36_DRAFTER_SWA_ALL=1 also
        // applies the window to the full-attention layer (testing whether
        // its softmax dilution over long context is what degrades the
        // drafter's conditioning, AL 9→2.8 between 3K and 7.8K ctx).
        let sliding_window = {
            use std::sync::OnceLock;
            static WINDOW_OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
            static FORCE_ALL: OnceLock<bool> = OnceLock::new();
            let window_override = *WINDOW_OVERRIDE.get_or_init(|| {
                std::env::var("QWEN36_DRAFTER_SWA_WINDOW")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
            });
            let force_all = *FORCE_ALL
                .get_or_init(|| std::env::var("QWEN36_DRAFTER_SWA_ALL").is_ok_and(|v| v == "1"));
            let effective = window_override.unwrap_or(self.config.sliding_window);
            match layer.kind {
                LayerAttentionKind::SlidingAttention => effective,
                LayerAttentionKind::FullAttention if force_all => effective,
                LayerAttentionKind::FullAttention => 0,
            }
        };
        backend
            .drafter_attention_block_bf16(&DrafterAttentionBlockSpec {
                q_bf16: self.workspace.q.ptr(),
                k_bf16: k_cache_buf.ptr(),
                v_bf16: v_cache_buf.ptr(),
                output_bf16: self.workspace.attn_out.ptr(),
                q_len,
                kv_seq_len,
                q_heads,
                kv_heads,
                head_dim,
                sliding_window,
            })
            .map_err(|e| anyhow!("attention: {e}"))?;

        gemm(
            backend,
            &self.workspace,
            self.workspace.attn_out.ptr(),
            layer.o_proj,
            q_len,
            q_features,
            hidden,
            self.workspace.attn_proj.ptr(),
        )?;
        Ok(())
    }

    fn mlp_block<B: KernelBackend>(
        &mut self,
        backend: &B,
        layer_idx: usize,
        q_len: usize,
    ) -> Result<()> {
        let layer: &DFlashLayerDevice = &self.device.layers[layer_idx];
        let hidden = self.workspace.hidden;
        let intermediate = self.workspace.intermediate;

        gemm(
            backend,
            &self.workspace,
            self.workspace.normed.ptr(),
            layer.mlp_gate_proj,
            q_len,
            hidden,
            intermediate,
            self.workspace.gate.ptr(),
        )?;
        gemm(
            backend,
            &self.workspace,
            self.workspace.normed.ptr(),
            layer.mlp_up_proj,
            q_len,
            hidden,
            intermediate,
            self.workspace.up.ptr(),
        )?;
        backend
            .swiglu(&SwiGluSpec {
                rows: q_len,
                intermediate,
                gate_bf16: self.workspace.gate.ptr(),
                up_bf16: self.workspace.up.ptr(),
                output_bf16: self.workspace.silu.ptr(),
            })
            .map_err(|e| anyhow!("swiglu: {e}"))?;
        gemm(
            backend,
            &self.workspace,
            self.workspace.silu.ptr(),
            layer.mlp_down_proj,
            q_len,
            intermediate,
            hidden,
            self.workspace.mlp_out.ptr(),
        )?;
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
