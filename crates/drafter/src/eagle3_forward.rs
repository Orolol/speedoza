use anyhow::{Context, Result, anyhow, bail};
use qwen36_fp4_kernels::{
    Bf16GemmSpec, CopyStridedRowsSpec, CudaDeviceBuffer, DevicePtr, DrafterAttentionBlockSpec,
    EmbeddingLookupSpec, KernelBackend, PartialRopeSpec, RmsNormSpec, SamplingRowsSpec, SwiGluSpec,
};

use crate::eagle3::Eagle3Config;
use crate::eagle3_gpu::{Eagle3DrafterDevice, Eagle3LayerDevice};
use crate::gpu::TensorOnDevice;

const BF16_BYTES: usize = 2;
const U32_BYTES: usize = 4;
const I32_BYTES: usize = 4;
const GEMM_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

pub struct Eagle3ForwardWorkspace {
    pub kv_cache_max_len: usize,
    hidden: usize,
    intermediate: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    fc_in_features: usize,
    attention_in_features: usize,
    draft_vocab_size: usize,

    token_id: CudaDeviceBuffer,
    input_embedding: CudaDeviceBuffer,
    input_normed: CudaDeviceBuffer,
    hidden_pre: CudaDeviceBuffer,
    hidden_normed: CudaDeviceBuffer,
    attn_input: CudaDeviceBuffer,
    q: CudaDeviceBuffer,
    attn_out: CudaDeviceBuffer,
    attn_proj: CudaDeviceBuffer,
    attn_residual: CudaDeviceBuffer,
    post_attn_normed: CudaDeviceBuffer,
    gate: CudaDeviceBuffer,
    up: CudaDeviceBuffer,
    silu: CudaDeviceBuffer,
    mlp_out: CudaDeviceBuffer,
    hidden_a: CudaDeviceBuffer,
    hidden_b: CudaDeviceBuffer,
    final_normed: CudaDeviceBuffer,
    logits: CudaDeviceBuffer,
    sampled_token: CudaDeviceBuffer,
    k_cache: CudaDeviceBuffer,
    v_cache: CudaDeviceBuffer,
    gemm_workspace: CudaDeviceBuffer,
    position_ids: CudaDeviceBuffer,
    bytes_total: usize,
}

#[derive(Debug, Clone)]
pub struct Eagle3WorkspaceReport {
    pub kv_cache_max_len: usize,
    pub gemm_workspace_bytes: usize,
    pub kv_caches_bytes: usize,
    pub logits_bytes: usize,
    pub intermediates_bytes: usize,
    pub total_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Eagle3DraftToken {
    pub draft_id: u32,
    pub target_id: u32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct Eagle3DraftChain {
    pub tokens: Vec<Eagle3DraftToken>,
    pub stopped_by_confidence: bool,
}

#[derive(Debug, Clone, Copy)]
struct Eagle3Sample {
    draft_id: u32,
    confidence: f32,
}

impl Eagle3ForwardWorkspace {
    pub fn alloc(config: &Eagle3Config, kv_cache_max_len: usize) -> Result<Self> {
        if kv_cache_max_len == 0 {
            bail!("kv_cache_max_len must be > 0");
        }
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let q_heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_features = config.q_proj_out();
        let kv_features = config.kv_proj_out();
        let fc_in_features = config.fc_in_features();
        let attention_in_features = config.attention_in_features();
        let draft_vocab_size = config.draft_vocab_size;

        let alloc = |bytes: usize, label: &str| -> Result<CudaDeviceBuffer> {
            CudaDeviceBuffer::alloc(bytes).map_err(|e| anyhow!("alloc {label}: {e}"))
        };

        let token_id = alloc(U32_BYTES, "eagle3_token_id")?;
        let input_embedding = alloc(hidden * BF16_BYTES, "eagle3_input_embedding")?;
        let input_normed = alloc(hidden * BF16_BYTES, "eagle3_input_normed")?;
        let hidden_pre = alloc(hidden * BF16_BYTES, "eagle3_hidden_pre")?;
        let hidden_normed = alloc(hidden * BF16_BYTES, "eagle3_hidden_normed")?;
        let attn_input = alloc(attention_in_features * BF16_BYTES, "eagle3_attn_input")?;
        let q = alloc(q_features * BF16_BYTES, "eagle3_q")?;
        let attn_out = alloc(q_features * BF16_BYTES, "eagle3_attn_out")?;
        let attn_proj = alloc(hidden * BF16_BYTES, "eagle3_attn_proj")?;
        let attn_residual = alloc(hidden * BF16_BYTES, "eagle3_attn_residual")?;
        let post_attn_normed = alloc(hidden * BF16_BYTES, "eagle3_post_attn_normed")?;
        let gate = alloc(intermediate * BF16_BYTES, "eagle3_gate")?;
        let up = alloc(intermediate * BF16_BYTES, "eagle3_up")?;
        let silu = alloc(intermediate * BF16_BYTES, "eagle3_silu")?;
        let mlp_out = alloc(hidden * BF16_BYTES, "eagle3_mlp_out")?;
        let hidden_a = alloc(hidden * BF16_BYTES, "eagle3_hidden_a")?;
        let hidden_b = alloc(hidden * BF16_BYTES, "eagle3_hidden_b")?;
        let final_normed = alloc(hidden * BF16_BYTES, "eagle3_final_normed")?;
        let logits = alloc(draft_vocab_size * BF16_BYTES, "eagle3_logits")?;
        let sampled_token = alloc(U32_BYTES, "eagle3_sampled_token")?;
        let k_cache = alloc(
            kv_cache_max_len * kv_features * BF16_BYTES,
            "eagle3_k_cache",
        )?;
        let v_cache = alloc(
            kv_cache_max_len * kv_features * BF16_BYTES,
            "eagle3_v_cache",
        )?;
        let gemm_workspace = alloc(GEMM_WORKSPACE_BYTES, "eagle3_gemm_workspace")?;
        let position_ids = alloc(kv_cache_max_len * I32_BYTES, "eagle3_position_ids")?;

        let bytes_total = token_id.bytes()
            + input_embedding.bytes()
            + input_normed.bytes()
            + hidden_pre.bytes()
            + hidden_normed.bytes()
            + attn_input.bytes()
            + q.bytes()
            + attn_out.bytes()
            + attn_proj.bytes()
            + attn_residual.bytes()
            + post_attn_normed.bytes()
            + gate.bytes()
            + up.bytes()
            + silu.bytes()
            + mlp_out.bytes()
            + hidden_a.bytes()
            + hidden_b.bytes()
            + final_normed.bytes()
            + logits.bytes()
            + sampled_token.bytes()
            + k_cache.bytes()
            + v_cache.bytes()
            + gemm_workspace.bytes()
            + position_ids.bytes();

        Ok(Self {
            kv_cache_max_len,
            hidden,
            intermediate,
            q_heads,
            kv_heads,
            head_dim,
            fc_in_features,
            attention_in_features,
            draft_vocab_size,
            token_id,
            input_embedding,
            input_normed,
            hidden_pre,
            hidden_normed,
            attn_input,
            q,
            attn_out,
            attn_proj,
            attn_residual,
            post_attn_normed,
            gate,
            up,
            silu,
            mlp_out,
            hidden_a,
            hidden_b,
            final_normed,
            logits,
            sampled_token,
            k_cache,
            v_cache,
            gemm_workspace,
            position_ids,
            bytes_total,
        })
    }

    pub fn report(&self) -> Eagle3WorkspaceReport {
        let kv_caches_bytes = self.k_cache.bytes() + self.v_cache.bytes();
        let logits_bytes = self.logits.bytes();
        Eagle3WorkspaceReport {
            kv_cache_max_len: self.kv_cache_max_len,
            gemm_workspace_bytes: self.gemm_workspace.bytes(),
            kv_caches_bytes,
            logits_bytes,
            intermediates_bytes: self.bytes_total
                - self.gemm_workspace.bytes()
                - kv_caches_bytes
                - logits_bytes,
            total_bytes: self.bytes_total,
        }
    }

    pub fn position_ids_buffer(&self) -> &CudaDeviceBuffer {
        &self.position_ids
    }

    fn hidden_slot(&self, slot: usize) -> DevicePtr {
        if slot & 1 == 0 {
            self.hidden_a.ptr()
        } else {
            self.hidden_b.ptr()
        }
    }
}

pub struct Eagle3Forward<'w> {
    device: &'w Eagle3DrafterDevice,
    config: &'w Eagle3Config,
    workspace: Eagle3ForwardWorkspace,
    stable_kv_len: usize,
    next_output_slot: usize,
    logits_host_bytes: Vec<u8>,
}

impl<'w> Eagle3Forward<'w> {
    pub fn new(
        device: &'w Eagle3DrafterDevice,
        config: &'w Eagle3Config,
        workspace: Eagle3ForwardWorkspace,
    ) -> Result<Self> {
        if workspace.hidden != config.hidden_size
            || workspace.intermediate != config.intermediate_size
            || workspace.q_heads != config.num_attention_heads
            || workspace.kv_heads != config.num_key_value_heads
            || workspace.head_dim != config.head_dim
            || workspace.fc_in_features != config.fc_in_features()
            || workspace.attention_in_features != config.attention_in_features()
            || workspace.draft_vocab_size != config.draft_vocab_size
        {
            bail!("workspace dimensions do not match EAGLE3 config");
        }
        Ok(Self {
            device,
            config,
            logits_host_bytes: vec![0u8; config.draft_vocab_size * BF16_BYTES],
            workspace,
            stable_kv_len: 0,
            next_output_slot: 0,
        })
    }

    pub fn workspace(&self) -> &Eagle3ForwardWorkspace {
        &self.workspace
    }

    pub fn current_kv_len(&self) -> usize {
        self.stable_kv_len
    }

    pub fn reset_kv_cache(&mut self) {
        self.stable_kv_len = 0;
        self.next_output_slot = 0;
    }

    pub fn crop_kv_cache(&mut self, new_len: usize) -> Result<()> {
        if new_len > self.stable_kv_len {
            bail!(
                "crop_kv_cache({new_len}) > current_kv_len {}",
                self.stable_kv_len,
            );
        }
        self.stable_kv_len = new_len;
        Ok(())
    }

    pub fn append_aux_rows<B: KernelBackend>(
        &mut self,
        backend: &B,
        target_embed_ptr: DevicePtr,
        target_vocab_size: usize,
        token_ids: &[u32],
        target_hidden_raw: DevicePtr,
    ) -> Result<DevicePtr> {
        if token_ids.is_empty() {
            bail!("append_aux_rows requires at least one token");
        }
        let mut last_hidden = DevicePtr::NULL;
        for (row, &token_id) in token_ids.iter().enumerate() {
            let aux_row = target_hidden_raw
                .offset_bytes(row * self.config.fc_in_features() * BF16_BYTES)
                .ok_or_else(|| anyhow!("target hidden row offset overflow"))?;
            let out_slot = self.next_output_slot;
            last_hidden = self.forward_aux_one(
                backend,
                target_embed_ptr,
                target_vocab_size,
                token_id,
                aux_row,
                self.stable_kv_len,
                out_slot,
            )?;
            self.stable_kv_len += 1;
            self.next_output_slot ^= 1;
        }
        Ok(last_hidden)
    }

    pub fn draft_chain<B: KernelBackend>(
        &mut self,
        backend: &B,
        target_embed_ptr: DevicePtr,
        target_vocab_size: usize,
        start_hidden: DevicePtr,
        draft_tokens: usize,
        min_confidence: f32,
        mut map_draft_token: impl FnMut(u32) -> Result<u32>,
    ) -> Result<Eagle3DraftChain> {
        if !min_confidence.is_finite() || !(0.0..=1.0).contains(&min_confidence) {
            bail!("min_confidence must be finite and in 0..=1, got {min_confidence}");
        }
        if draft_tokens == 0 {
            return Ok(Eagle3DraftChain {
                tokens: Vec::new(),
                stopped_by_confidence: false,
            });
        }
        let mut out = Vec::with_capacity(draft_tokens);
        let mut current_hidden = start_hidden;
        let mut local_kv_len = self.stable_kv_len;
        let mut output_slot = self.next_output_slot;
        let mut stopped_by_confidence = false;
        let compute_confidence = min_confidence > 0.0;

        for idx in 0..draft_tokens {
            let sample = self.sample_from_hidden(backend, current_hidden, compute_confidence)?;
            if compute_confidence && sample.confidence < min_confidence {
                stopped_by_confidence = true;
                break;
            }
            let target_token = map_draft_token(sample.draft_id)?;
            out.push(Eagle3DraftToken {
                draft_id: sample.draft_id,
                target_id: target_token,
                confidence: sample.confidence,
            });
            if idx + 1 == draft_tokens {
                break;
            }
            current_hidden = self.forward_hidden_one(
                backend,
                target_embed_ptr,
                target_vocab_size,
                target_token,
                current_hidden,
                local_kv_len,
                output_slot,
            )?;
            local_kv_len += 1;
            output_slot ^= 1;
        }
        Ok(Eagle3DraftChain {
            tokens: out,
            stopped_by_confidence,
        })
    }

    fn forward_aux_one<B: KernelBackend>(
        &mut self,
        backend: &B,
        target_embed_ptr: DevicePtr,
        target_vocab_size: usize,
        token_id: u32,
        aux_row: DevicePtr,
        past_len: usize,
        output_slot: usize,
    ) -> Result<DevicePtr> {
        if past_len >= self.workspace.kv_cache_max_len {
            bail!(
                "EAGLE3 KV position {past_len} exceeds max {}",
                self.workspace.kv_cache_max_len,
            );
        }
        self.embed_token(backend, target_embed_ptr, target_vocab_size, token_id)?;
        gemm(
            backend,
            &self.workspace,
            aux_row,
            self.device.fc,
            1,
            self.workspace.fc_in_features,
            self.workspace.hidden,
            self.workspace.hidden_pre.ptr(),
        )?;
        self.forward_layer_one(
            backend,
            self.workspace.hidden_pre.ptr(),
            past_len,
            output_slot,
        )
    }

    fn forward_hidden_one<B: KernelBackend>(
        &mut self,
        backend: &B,
        target_embed_ptr: DevicePtr,
        target_vocab_size: usize,
        token_id: u32,
        hidden: DevicePtr,
        past_len: usize,
        output_slot: usize,
    ) -> Result<DevicePtr> {
        if past_len >= self.workspace.kv_cache_max_len {
            bail!(
                "EAGLE3 KV position {past_len} exceeds max {}",
                self.workspace.kv_cache_max_len,
            );
        }
        self.embed_token(backend, target_embed_ptr, target_vocab_size, token_id)?;
        self.forward_layer_one(backend, hidden, past_len, output_slot)
    }

    fn embed_token<B: KernelBackend>(
        &mut self,
        backend: &B,
        target_embed_ptr: DevicePtr,
        target_vocab_size: usize,
        token_id: u32,
    ) -> Result<()> {
        self.workspace
            .token_id
            .copy_from_host(&token_id.to_le_bytes())
            .with_context(|| "upload EAGLE3 token id")?;
        backend
            .embedding_lookup(&EmbeddingLookupSpec {
                tokens: 1,
                hidden: self.workspace.hidden,
                vocab_size: target_vocab_size,
                token_ids_u32: self.workspace.token_id.ptr(),
                embedding_bf16: target_embed_ptr,
                output_bf16: self.workspace.input_embedding.ptr(),
            })
            .map_err(|e| anyhow!("EAGLE3 embedding_lookup: {e}"))
    }

    fn forward_layer_one<B: KernelBackend>(
        &mut self,
        backend: &B,
        hidden_residual: DevicePtr,
        past_len: usize,
        output_slot: usize,
    ) -> Result<DevicePtr> {
        let layer = &self.device.layer;
        let output_hidden = self.workspace.hidden_slot(output_slot);

        backend
            .rmsnorm(&RmsNormSpec {
                rows: 1,
                hidden: self.workspace.hidden,
                eps: self.config.rms_norm_eps,
                input_bf16: hidden_residual,
                weight_bf16: layer.hidden_norm.ptr,
                residual_bf16: DevicePtr::NULL,
                residual_out_bf16: DevicePtr::NULL,
                output_bf16: self.workspace.hidden_normed.ptr(),
                direct_weight: true,
            })
            .map_err(|e| anyhow!("EAGLE3 hidden_norm: {e}"))?;

        backend
            .rmsnorm(&RmsNormSpec {
                rows: 1,
                hidden: self.workspace.hidden,
                eps: self.config.rms_norm_eps,
                input_bf16: self.workspace.input_embedding.ptr(),
                weight_bf16: layer.input_layernorm.ptr,
                residual_bf16: DevicePtr::NULL,
                residual_out_bf16: DevicePtr::NULL,
                output_bf16: self.workspace.input_normed.ptr(),
                direct_weight: true,
            })
            .map_err(|e| anyhow!("EAGLE3 input_layernorm: {e}"))?;

        self.workspace
            .attn_input
            .copy_from_device_ptr_at(
                0,
                self.workspace.input_normed.ptr(),
                self.workspace.hidden * BF16_BYTES,
            )
            .map_err(|e| anyhow!("EAGLE3 concat input half: {e}"))?;
        self.workspace
            .attn_input
            .copy_from_device_ptr_at(
                self.workspace.hidden * BF16_BYTES,
                self.workspace.hidden_normed.ptr(),
                self.workspace.hidden * BF16_BYTES,
            )
            .map_err(|e| anyhow!("EAGLE3 concat hidden half: {e}"))?;

        self.attention_one(backend, layer, past_len)
            .with_context(|| "EAGLE3 attention")?;

        backend
            .rmsnorm(&RmsNormSpec {
                rows: 1,
                hidden: self.workspace.hidden,
                eps: self.config.rms_norm_eps,
                input_bf16: self.workspace.attn_proj.ptr(),
                weight_bf16: layer.post_attention_layernorm.ptr,
                residual_bf16: hidden_residual,
                residual_out_bf16: self.workspace.attn_residual.ptr(),
                output_bf16: self.workspace.post_attn_normed.ptr(),
                direct_weight: true,
            })
            .map_err(|e| anyhow!("EAGLE3 post_attention_layernorm: {e}"))?;

        self.mlp_one(backend, layer).with_context(|| "EAGLE3 MLP")?;

        // Use the RMSNorm kernel only as a fused vector add. Its normalized
        // output is discarded; residual_out is the decoder-layer output.
        backend
            .rmsnorm(&RmsNormSpec {
                rows: 1,
                hidden: self.workspace.hidden,
                eps: self.config.rms_norm_eps,
                input_bf16: self.workspace.mlp_out.ptr(),
                weight_bf16: self.device.norm.ptr,
                residual_bf16: self.workspace.attn_residual.ptr(),
                residual_out_bf16: output_hidden,
                output_bf16: self.workspace.final_normed.ptr(),
                direct_weight: true,
            })
            .map_err(|e| anyhow!("EAGLE3 residual add via rmsnorm: {e}"))?;

        Ok(output_hidden)
    }

    fn attention_one<B: KernelBackend>(
        &mut self,
        backend: &B,
        layer: &Eagle3LayerDevice,
        past_len: usize,
    ) -> Result<()> {
        let hidden = self.workspace.hidden;
        let q_features = self.config.q_proj_out();
        let kv_features = self.config.kv_proj_out();
        let kv_seq_len = past_len + 1;

        gemm(
            backend,
            &self.workspace,
            self.workspace.attn_input.ptr(),
            layer.q_proj,
            1,
            self.workspace.attention_in_features,
            q_features,
            self.workspace.q.ptr(),
        )?;

        let kv_row_bytes = kv_features * BF16_BYTES;
        let k_new = self
            .workspace
            .k_cache
            .ptr_at(past_len * kv_row_bytes)
            .map_err(|e| anyhow!("EAGLE3 k_cache offset: {e}"))?;
        let v_new = self
            .workspace
            .v_cache
            .ptr_at(past_len * kv_row_bytes)
            .map_err(|e| anyhow!("EAGLE3 v_cache offset: {e}"))?;

        gemm(
            backend,
            &self.workspace,
            self.workspace.attn_input.ptr(),
            layer.k_proj,
            1,
            self.workspace.attention_in_features,
            kv_features,
            k_new,
        )?;
        gemm(
            backend,
            &self.workspace,
            self.workspace.attn_input.ptr(),
            layer.v_proj,
            1,
            self.workspace.attention_in_features,
            kv_features,
            v_new,
        )?;

        let pos_ptr = self
            .workspace
            .position_ids
            .ptr_at(past_len * I32_BYTES)
            .map_err(|e| anyhow!("EAGLE3 position offset: {e}"))?;
        backend
            .partial_rope(&PartialRopeSpec {
                tokens: 1,
                q_heads: self.workspace.q_heads,
                kv_heads: self.workspace.kv_heads,
                head_dim: self.workspace.head_dim,
                rope_dims: self.workspace.head_dim,
                base_theta: self.config.rope_theta,
                position_i32: 0,
                use_scalar_position: false,
                positions_i32: pos_ptr,
                q_bf16: self.workspace.q.ptr(),
                k_bf16: k_new,
                scalar_position_device_i32: DevicePtr::NULL,
            })
            .map_err(|e| anyhow!("EAGLE3 rope: {e}"))?;

        backend
            .drafter_attention_block_bf16(&DrafterAttentionBlockSpec {
                q_bf16: self.workspace.q.ptr(),
                k_bf16: self.workspace.k_cache.ptr(),
                v_bf16: self.workspace.v_cache.ptr(),
                output_bf16: self.workspace.attn_out.ptr(),
                q_len: 1,
                kv_seq_len,
                q_heads: self.workspace.q_heads,
                kv_heads: self.workspace.kv_heads,
                head_dim: self.workspace.head_dim,
                sliding_window: 0,
            })
            .map_err(|e| anyhow!("EAGLE3 attention kernel: {e}"))?;

        gemm(
            backend,
            &self.workspace,
            self.workspace.attn_out.ptr(),
            layer.o_proj,
            1,
            q_features,
            hidden,
            self.workspace.attn_proj.ptr(),
        )?;
        Ok(())
    }

    fn mlp_one<B: KernelBackend>(&mut self, backend: &B, layer: &Eagle3LayerDevice) -> Result<()> {
        let hidden = self.workspace.hidden;
        let intermediate = self.workspace.intermediate;
        gemm(
            backend,
            &self.workspace,
            self.workspace.post_attn_normed.ptr(),
            layer.mlp_gate_proj,
            1,
            hidden,
            intermediate,
            self.workspace.gate.ptr(),
        )?;
        gemm(
            backend,
            &self.workspace,
            self.workspace.post_attn_normed.ptr(),
            layer.mlp_up_proj,
            1,
            hidden,
            intermediate,
            self.workspace.up.ptr(),
        )?;
        backend
            .swiglu(&SwiGluSpec {
                rows: 1,
                intermediate,
                gate_bf16: self.workspace.gate.ptr(),
                up_bf16: self.workspace.up.ptr(),
                output_bf16: self.workspace.silu.ptr(),
            })
            .map_err(|e| anyhow!("EAGLE3 swiglu: {e}"))?;
        gemm(
            backend,
            &self.workspace,
            self.workspace.silu.ptr(),
            layer.mlp_down_proj,
            1,
            intermediate,
            hidden,
            self.workspace.mlp_out.ptr(),
        )
    }

    fn sample_from_hidden<B: KernelBackend>(
        &mut self,
        backend: &B,
        hidden: DevicePtr,
        compute_confidence: bool,
    ) -> Result<Eagle3Sample> {
        backend
            .rmsnorm(&RmsNormSpec {
                rows: 1,
                hidden: self.workspace.hidden,
                eps: self.config.rms_norm_eps,
                input_bf16: hidden,
                weight_bf16: self.device.norm.ptr,
                residual_bf16: DevicePtr::NULL,
                residual_out_bf16: DevicePtr::NULL,
                output_bf16: self.workspace.final_normed.ptr(),
                direct_weight: true,
            })
            .map_err(|e| anyhow!("EAGLE3 final norm: {e}"))?;
        gemm(
            backend,
            &self.workspace,
            self.workspace.final_normed.ptr(),
            self.device.lm_head,
            1,
            self.workspace.hidden,
            self.config.draft_vocab_size,
            self.workspace.logits.ptr(),
        )?;

        if compute_confidence {
            return self.sample_logits_on_host();
        }

        backend
            .sample_rows(&SamplingRowsSpec {
                rows: 1,
                vocab_size: self.config.draft_vocab_size,
                logits_bf16: self.workspace.logits.ptr(),
                output_token_u32: self.workspace.sampled_token.ptr(),
                mirror_last_output_token_u32: DevicePtr::NULL,
                temperature: 0.0,
            })
            .map_err(|e| anyhow!("EAGLE3 sample_rows: {e}"))?;

        let mut bytes = [0u8; U32_BYTES];
        self.workspace.sampled_token.copy_to_host(&mut bytes)?;
        let draft_token = u32::from_le_bytes(bytes);
        Ok(Eagle3Sample {
            draft_id: draft_token,
            confidence: f32::NAN,
        })
    }

    fn sample_logits_on_host(&mut self) -> Result<Eagle3Sample> {
        self.workspace
            .logits
            .copy_to_host(&mut self.logits_host_bytes)?;
        let (draft_id, confidence) = top1_confidence_from_bf16_logits(&self.logits_host_bytes)?;
        Ok(Eagle3Sample {
            draft_id,
            confidence,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm<B: KernelBackend>(
    backend: &B,
    workspace: &Eagle3ForwardWorkspace,
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
        .map_err(|e| anyhow!("EAGLE3 bf16_gemm({rows}x{in_features}->{out_features}): {e}"))
}

#[allow(dead_code)]
fn copy_rows_spec(input: DevicePtr, output: DevicePtr, values: usize) -> CopyStridedRowsSpec {
    CopyStridedRowsSpec {
        rows: 1,
        values,
        input_stride: values,
        output_stride: values,
        input_bf16: input,
        output_bf16: output,
    }
}

fn top1_confidence_from_bf16_logits(bytes: &[u8]) -> Result<(u32, f32)> {
    if bytes.is_empty() || bytes.len() % BF16_BYTES != 0 {
        bail!("BF16 logits byte length {} is invalid", bytes.len());
    }

    let mut best_idx = None;
    let mut best_logit = f32::NEG_INFINITY;
    for (idx, chunk) in bytes.chunks_exact(BF16_BYTES).enumerate() {
        let value = bf16_le_to_f32(chunk);
        if !value.is_finite() {
            continue;
        }
        if value > best_logit {
            best_idx = Some(idx);
            best_logit = value;
        }
    }

    let best_idx = best_idx.ok_or_else(|| anyhow!("EAGLE3 logits contain no finite values"))?;
    let mut denom = 0.0_f64;
    for chunk in bytes.chunks_exact(BF16_BYTES) {
        let value = bf16_le_to_f32(chunk);
        if value.is_finite() {
            denom += f64::from(value - best_logit).exp();
        }
    }
    if denom <= 0.0 || !denom.is_finite() {
        bail!("EAGLE3 logits produced invalid softmax denominator {denom}");
    }
    Ok((best_idx as u32, (1.0 / denom) as f32))
}

fn bf16_le_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_bf16(out: &mut Vec<u8>, value: f32) {
        let bits = (value.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }

    #[test]
    fn top1_confidence_uses_softmax_probability() {
        let mut logits = Vec::new();
        push_bf16(&mut logits, 1.0);
        push_bf16(&mut logits, 2.0);
        push_bf16(&mut logits, 0.0);

        let (idx, confidence) = top1_confidence_from_bf16_logits(&logits).unwrap();
        let expected = 1.0 / (1.0 + f32::exp(-1.0) + f32::exp(-2.0));

        assert_eq!(idx, 1);
        assert!((confidence - expected).abs() < 1.0e-4);
    }
}
