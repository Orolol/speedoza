use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
use qwen36_fp4_core::TensorInfo;
use qwen36_fp4_core::{CoreError, KvCacheDtype, ModelLayout, ModelTopology, Result};
#[cfg(feature = "cuda")]
use qwen36_fp4_kernels::{
    AttentionDecodeSpec, AttentionShape, Bf16MatVecSpec, Conv1dUpdateSpec, CublasLtFp4ScaleMode,
    CudaBackend, DeltaNetDecodeSpec, DeltaNetShape, DevicePtr, EmbeddingLookupSpec, GdnGateSpec,
    Nvfp4GemmSpec, Nvfp4QuantizeSpec, PartialRopeSpec, RmsNormSpec, SamplingSpec, SigmoidGateSpec,
    SwiGluSpec,
};
use qwen36_fp4_kernels::{KernelBackend, NoCudaBackend};
#[cfg(feature = "cuda")]
use qwen36_fp4_loader::MappedModel;

use crate::cuda_graph::CudaGraphPlan;
#[cfg(feature = "cuda")]
use crate::gpu::{GpuForwardBuffers, GpuRuntimeBuffers, GpuWeightStore};
use crate::kv_cache::KvCachePlan;
use crate::state::{DeltaNetStatePlan, RuntimeState};
use crate::weights::ModelWeightsManifest;
#[cfg(feature = "cuda")]
use crate::weights::{
    FullAttentionLayerWeights, LayerWeights, LinearAttentionLayerWeights, LinearWeightBinding,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub max_context: usize,
    pub kv_cache_dtype: KvCacheDtype,
    pub turboquant: bool,
    pub mtp_speculative_tokens: usize,
    pub cuda_graphs: CudaGraphPlan,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_context: 262144,
            kv_cache_dtype: KvCacheDtype::Fp8,
            turboquant: true,
            mtp_speculative_tokens: 3,
            cuda_graphs: CudaGraphPlan::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardOutput {
    pub logits_device_ptr: u64,
    pub produced_tokens: usize,
}

pub struct Engine<B: KernelBackend = NoCudaBackend> {
    pub topology: ModelTopology,
    pub config: EngineConfig,
    pub state: RuntimeState,
    pub weights: Option<ModelWeightsManifest>,
    #[cfg(feature = "cuda")]
    pub gpu_weights: Option<GpuWeightStore>,
    #[cfg(feature = "cuda")]
    pub gpu_buffers: Option<GpuRuntimeBuffers>,
    #[cfg(feature = "cuda")]
    pub gpu_forward: Option<GpuForwardBuffers>,
    backend: B,
}

impl Engine<NoCudaBackend> {
    pub fn no_cuda(layout: &ModelLayout, config: EngineConfig) -> Self {
        Self::new(layout.topology.clone(), config, NoCudaBackend)
    }

    pub fn no_cuda_with_weights(layout: &ModelLayout, config: EngineConfig) -> Result<Self> {
        Self::from_layout(layout, config, NoCudaBackend)
    }
}

impl<B: KernelBackend> Engine<B> {
    pub fn new(topology: ModelTopology, config: EngineConfig, backend: B) -> Self {
        let kv_cache = KvCachePlan::new(&topology, config.max_context, config.kv_cache_dtype);
        let deltanet = DeltaNetStatePlan::new(&topology);
        let state = RuntimeState::new(kv_cache, deltanet);
        Self {
            topology,
            config,
            state,
            weights: None,
            #[cfg(feature = "cuda")]
            gpu_weights: None,
            #[cfg(feature = "cuda")]
            gpu_buffers: None,
            #[cfg(feature = "cuda")]
            gpu_forward: None,
            backend,
        }
    }

    pub fn from_layout(layout: &ModelLayout, config: EngineConfig, backend: B) -> Result<Self> {
        let weights = ModelWeightsManifest::from_layout(layout)?;
        let mut engine = Self::new(layout.topology.clone(), config, backend);
        engine.weights = Some(weights);
        Ok(engine)
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<ForwardOutput> {
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda("engine_prefill"));
        }
        #[cfg(feature = "cuda")]
        {
            self.prefill_cuda(prompt_tokens)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = prompt_tokens;
            Err(CoreError::UnsupportedNoCuda("engine_prefill"))
        }
    }

    pub fn decode_one(&mut self, token: u32) -> Result<ForwardOutput> {
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda("engine_decode_one"));
        }
        #[cfg(feature = "cuda")]
        {
            self.forward_token_cuda(token, self.state.position)?;
            self.state.advance(1);
            Ok(ForwardOutput {
                logits_device_ptr: self.cuda_forward()?.logits.ptr().0,
                produced_tokens: 1,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = token;
            Err(CoreError::UnsupportedNoCuda("engine_decode_one"))
        }
    }

    #[cfg(feature = "cuda")]
    pub fn sample_greedy(&self) -> Result<u32> {
        let forward = self.cuda_forward()?;
        self.backend.sample(&SamplingSpec {
            vocab_size: self.topology.vocab_size,
            logits_bf16: forward.logits.ptr(),
            output_token_u32: forward.sampled_token_u32.ptr(),
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        })?;
        qwen36_fp4_kernels::cuda_synchronize()?;
        let mut token = [0_u8; 4];
        forward.sampled_token_u32.copy_to_host(&mut token)?;
        Ok(u32::from_ne_bytes(token))
    }

    #[cfg(feature = "cuda")]
    fn prefill_cuda(&mut self, prompt_tokens: &[u32]) -> Result<ForwardOutput> {
        if prompt_tokens.is_empty() {
            return Err(CoreError::Runtime(
                "prefill requires at least one prompt token".to_owned(),
            ));
        }
        for &token in prompt_tokens {
            self.forward_token_cuda(token, self.state.position)?;
            self.state.advance(1);
        }
        Ok(ForwardOutput {
            logits_device_ptr: self.cuda_forward()?.logits.ptr().0,
            produced_tokens: prompt_tokens.len(),
        })
    }

    #[cfg(feature = "cuda")]
    fn forward_token_cuda(&self, token: u32, position: usize) -> Result<()> {
        if position >= self.config.max_context {
            return Err(CoreError::Runtime(format!(
                "position {position} exceeds configured max_context {}",
                self.config.max_context
            )));
        }
        if self.config.kv_cache_dtype != KvCacheDtype::Bf16 {
            return Err(CoreError::Runtime(
                "reference CUDA scheduler currently requires BF16 KV cache".to_owned(),
            ));
        }

        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let forward = self.cuda_forward()?;

        forward.token_u32.copy_from_host(&token.to_ne_bytes())?;
        forward
            .position_i32
            .copy_from_host(&(position as i32).to_ne_bytes())?;
        self.backend.embedding_lookup(&EmbeddingLookupSpec {
            tokens: 1,
            hidden: self.topology.hidden_size,
            vocab_size: self.topology.vocab_size,
            token_ids_u32: forward.token_u32.ptr(),
            embedding_bf16: self.tensor_ptr(weights, &manifest.embed_tokens)?,
            output_bf16: forward.hidden.ptr(),
        })?;

        let mut residual_initialized = false;
        for layer in &manifest.layers {
            self.run_input_norm(
                layer_common_input_norm(layer),
                residual_initialized,
                forward.hidden.ptr(),
                forward.normed.ptr(),
            )?;
            residual_initialized = true;

            match layer {
                LayerWeights::LinearAttention(layer) => {
                    self.run_linear_attention_layer(layer, runtime, forward)?;
                }
                LayerWeights::FullAttention(layer) => {
                    self.run_full_attention_layer(layer, runtime, forward, position)?;
                }
            }

            self.rmsnorm(
                1,
                self.topology.hidden_size,
                forward.block_out.ptr(),
                self.tensor_ptr(weights, layer_common_post_norm(layer))?,
                forward.residual.ptr(),
                forward.residual.ptr(),
                forward.normed.ptr(),
            )?;
            self.run_mlp(layer, forward)?;
        }

        self.rmsnorm(
            1,
            self.topology.hidden_size,
            forward.hidden.ptr(),
            self.tensor_ptr(weights, &manifest.final_norm)?,
            forward.residual.ptr(),
            DevicePtr::NULL,
            forward.normed.ptr(),
        )?;
        self.bf16_matvec(
            &manifest.lm_head,
            forward.normed.ptr(),
            forward.logits.ptr(),
        )?;
        qwen36_fp4_kernels::cuda_synchronize()?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn run_input_norm(
        &self,
        norm: &TensorInfo,
        residual_initialized: bool,
        input: DevicePtr,
        output: DevicePtr,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let residual_in = if residual_initialized {
            self.cuda_forward()?.residual.ptr()
        } else {
            DevicePtr::NULL
        };
        self.rmsnorm(
            1,
            self.topology.hidden_size,
            input,
            self.tensor_ptr(weights, norm)?,
            residual_in,
            self.cuda_forward()?.residual.ptr(),
            output,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_linear_attention_layer(
        &self,
        layer: &LinearAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
    ) -> Result<()> {
        let qkv_dim = self.topology.linear_attention_qkv_dim();
        let key_dim = self.topology.linear_num_key_heads * self.topology.linear_key_head_dim;
        let value_dim = self.topology.linear_attention_value_dim();
        let layer_ordinal = self.linear_layer_ordinal(layer.layer_index)?;
        let conv_history = runtime.conv_history.ptr_at(
            layer_ordinal * qkv_dim * self.topology.linear_conv_kernel_dim.saturating_sub(1) * 2,
        )?;
        let state = runtime
            .deltanet_state
            .ptr_at(layer_ordinal * self.state.deltanet.state_bytes_per_layer as usize)?;

        self.linear(&layer.in_proj_qkv, forward.normed.ptr(), forward.qkv.ptr())?;
        self.backend.conv1d_update(&Conv1dUpdateSpec {
            channels: qkv_dim,
            kernel_size: self.topology.linear_conv_kernel_dim,
            input_bf16: forward.qkv.ptr(),
            conv_history_bf16: conv_history,
            weight_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.conv1d_weight)?,
            output_bf16: forward.aux.ptr(),
        })?;
        self.linear(&layer.in_proj_b, forward.normed.ptr(), forward.aux2.ptr())?;
        self.linear(&layer.in_proj_a, forward.normed.ptr(), forward.aux3.ptr())?;
        self.backend.gdn_gate(&GdnGateSpec {
            heads: self.topology.linear_num_value_heads,
            a_bf16: forward.aux3.ptr(),
            b_bf16: forward.aux2.ptr(),
            a_log_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.a_log)?,
            dt_bias_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.dt_bias)?,
            gate_f32: forward.gate_f32.ptr(),
            beta_f32: forward.beta_f32.ptr(),
        })?;
        self.backend.deltanet_decode(&DeltaNetDecodeSpec {
            layer_index: layer.layer_index,
            tokens_in_persistent_loop: 1,
            q_bf16: forward.aux.ptr(),
            k_bf16: forward.aux.ptr_at(key_dim * 2)?,
            v_bf16: forward.aux.ptr_at(key_dim * 4)?,
            state_bf16: state,
            conv_history_bf16: conv_history,
            output_bf16: forward.aux3.ptr(),
            gate_f32: forward.gate_f32.ptr(),
            beta_f32: forward.beta_f32.ptr(),
            shape: DeltaNetShape {
                qk_heads: self.topology.linear_num_key_heads,
                v_heads: self.topology.linear_num_value_heads,
                key_dim: self.topology.linear_key_head_dim,
                value_dim: self.topology.linear_value_head_dim,
                conv_kernel: self.topology.linear_conv_kernel_dim,
            },
            state_decay: 1.0,
            update_scale: 1.0,
            qk_l2norm: true,
        })?;
        self.linear(&layer.in_proj_z, forward.normed.ptr(), forward.qkv.ptr())?;
        self.rmsnorm(
            self.topology.linear_num_value_heads,
            self.topology.linear_value_head_dim,
            forward.aux3.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.norm_weight)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.aux2.ptr(),
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: 1,
            intermediate: value_dim,
            gate_bf16: forward.qkv.ptr(),
            up_bf16: forward.aux2.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.linear(&layer.out_proj, forward.aux3.ptr(), forward.block_out.ptr())
    }

    #[cfg(feature = "cuda")]
    fn run_full_attention_layer(
        &self,
        layer: &FullAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        position: usize,
    ) -> Result<()> {
        let q_dim = self.topology.full_attention_q_dim();
        let cache = runtime
            .kv_cache
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("KV cache was not allocated".to_owned()))?;
        let layout = self
            .state
            .kv_cache
            .layers
            .iter()
            .find(|layout| layout.global_layer_index == layer.layer_index)
            .ok_or_else(|| {
                CoreError::Runtime(format!(
                    "missing KV-cache layout for layer {}",
                    layer.layer_index
                ))
            })?;

        self.linear(&layer.q_proj, forward.normed.ptr(), forward.qkv.ptr())?;
        self.linear(&layer.k_proj, forward.normed.ptr(), forward.aux.ptr())?;
        self.linear(&layer.v_proj, forward.normed.ptr(), forward.aux2.ptr())?;
        self.rmsnorm(
            self.topology.attention_num_heads,
            self.topology.attention_head_dim,
            forward.qkv.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.q_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.qkv.ptr(),
        )?;
        self.rmsnorm(
            self.topology.attention_num_kv_heads,
            self.topology.attention_head_dim,
            forward.aux.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.k_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.aux.ptr(),
        )?;
        self.backend.partial_rope(&PartialRopeSpec {
            tokens: 1,
            q_heads: self.topology.attention_num_heads,
            kv_heads: self.topology.attention_num_kv_heads,
            head_dim: self.topology.attention_head_dim,
            rope_dims: self.topology.attention_rope_dims(),
            base_theta: self.topology.rope_theta,
            positions_i32: forward.position_i32.ptr(),
            q_bf16: forward.qkv.ptr(),
            k_bf16: forward.aux.ptr(),
        })?;
        self.backend.attention_decode(&AttentionDecodeSpec {
            layer_index: layer.layer_index,
            position,
            q_bf16: forward.qkv.ptr(),
            k_bf16: forward.aux.ptr(),
            v_bf16: forward.aux2.ptr(),
            kv_cache_k: cache.ptr_at(layout.k_offset_bytes as usize)?,
            kv_cache_v: cache.ptr_at(layout.v_offset_bytes as usize)?,
            output_bf16: forward.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
        })?;
        self.backend.sigmoid_gate(&SigmoidGateSpec {
            elements: q_dim,
            gate_bf16: forward.qkv.ptr_at(q_dim * 2)?,
            input_bf16: forward.aux3.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.linear(&layer.o_proj, forward.aux3.ptr(), forward.block_out.ptr())
    }

    #[cfg(feature = "cuda")]
    fn run_mlp(&self, layer: &LayerWeights, forward: &GpuForwardBuffers) -> Result<()> {
        let common = match layer {
            LayerWeights::LinearAttention(layer) => &layer.common,
            LayerWeights::FullAttention(layer) => &layer.common,
        };
        self.linear(
            &common.mlp_gate_proj,
            forward.normed.ptr(),
            forward.aux.ptr(),
        )?;
        self.linear(
            &common.mlp_up_proj,
            forward.normed.ptr(),
            forward.aux2.ptr(),
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: 1,
            intermediate: self.topology.intermediate_size,
            gate_bf16: forward.aux.ptr(),
            up_bf16: forward.aux2.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.linear(
            &common.mlp_down_proj,
            forward.aux3.ptr(),
            forward.hidden.ptr(),
        )
    }

    #[cfg(feature = "cuda")]
    fn linear(
        &self,
        binding: &LinearWeightBinding,
        input: DevicePtr,
        output: DevicePtr,
    ) -> Result<()> {
        match binding {
            LinearWeightBinding::Nvfp4 {
                weight,
                block_scale,
                tensor_scale,
            } => {
                let weights = self.cuda_weights()?;
                let forward = self.cuda_forward()?;
                let runtime = self.cuda_runtime()?;
                let out_features = *weight.shape.first().ok_or_else(|| {
                    CoreError::Runtime(format!("tensor {} has empty shape", weight.name))
                })?;
                let packed_in = *weight.shape.get(1).ok_or_else(|| {
                    CoreError::Runtime(format!("tensor {} is not a matrix", weight.name))
                })?;
                let in_features = packed_in * 2;
                self.backend.nvfp4_quantize_bf16(&Nvfp4QuantizeSpec {
                    values: in_features,
                    input_bf16: input,
                    output_fp4: forward.activation_fp4.ptr(),
                    output_scale_e4m3: forward.activation_scale.ptr(),
                    output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                })?;
                let workspace = runtime
                    .workspace
                    .as_ref()
                    .map(|buffer| buffer.ptr())
                    .unwrap_or(DevicePtr::NULL);
                let workspace_bytes = runtime
                    .workspace
                    .as_ref()
                    .map(|buffer| buffer.bytes())
                    .unwrap_or(0);
                let gemm_spec = Nvfp4GemmSpec {
                    m: out_features,
                    n: 1,
                    k: in_features,
                    a_fp4: self.tensor_ptr(weights, weight)?,
                    a_scale: self.tensor_ptr(weights, block_scale)?,
                    a_scale_2: self.tensor_ptr(weights, tensor_scale)?,
                    b_fp4: forward.activation_fp4.ptr(),
                    b_scale: forward.activation_scale.ptr(),
                    b_scale_2: forward.activation_scale_2.ptr(),
                    c_bf16: output,
                    workspace,
                    workspace_bytes,
                    alpha: self.tensor_scalar_f32(weights, tensor_scale)?,
                    scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
                };
                self.backend.nvfp4_gemm(&gemm_spec).map_err(|err| {
                    CoreError::Runtime(format!(
                        "NVFP4 cuBLASLt GEMM failed for {} (m={}, n={}, k={}): {err}",
                        weight.name, gemm_spec.m, gemm_spec.n, gemm_spec.k
                    ))
                })
            }
            LinearWeightBinding::Bf16 { weight } => self.bf16_matvec(weight, input, output),
        }
    }

    #[cfg(feature = "cuda")]
    fn bf16_matvec(&self, weight: &TensorInfo, input: DevicePtr, output: DevicePtr) -> Result<()> {
        let out_features = *weight
            .shape
            .first()
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} has empty shape", weight.name)))?;
        let in_features = *weight
            .shape
            .get(1)
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} is not a matrix", weight.name)))?;
        self.backend.bf16_matvec(&Bf16MatVecSpec {
            out_features,
            in_features,
            input_bf16: input,
            weight_bf16: self.tensor_ptr(self.cuda_weights()?, weight)?,
            output_bf16: output,
        })
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm(
        &self,
        rows: usize,
        hidden: usize,
        input: DevicePtr,
        weight: DevicePtr,
        residual: DevicePtr,
        residual_out: DevicePtr,
        output: DevicePtr,
    ) -> Result<()> {
        self.backend.rmsnorm(&RmsNormSpec {
            rows,
            hidden,
            eps: 1.0e-6,
            input_bf16: input,
            weight_bf16: weight,
            residual_bf16: residual,
            residual_out_bf16: residual_out,
            output_bf16: output,
        })
    }

    #[cfg(feature = "cuda")]
    fn tensor_ptr(&self, weights: &GpuWeightStore, info: &TensorInfo) -> Result<DevicePtr> {
        weights
            .tensor(&info.name)
            .map(|tensor| tensor.ptr())
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} was not uploaded", info.name)))
    }

    #[cfg(feature = "cuda")]
    fn tensor_scalar_f32(&self, weights: &GpuWeightStore, info: &TensorInfo) -> Result<f32> {
        weights.scalar_f32(&info.name).ok_or_else(|| {
            CoreError::Runtime(format!(
                "tensor {} was not uploaded as a scalar f32",
                info.name
            ))
        })
    }

    #[cfg(feature = "cuda")]
    fn linear_layer_ordinal(&self, layer_index: usize) -> Result<usize> {
        self.topology
            .linear_attention_layers()
            .into_iter()
            .position(|idx| idx == layer_index)
            .ok_or_else(|| {
                CoreError::Runtime(format!("layer {layer_index} is not linear attention"))
            })
    }

    #[cfg(feature = "cuda")]
    fn cuda_weights(&self) -> Result<&GpuWeightStore> {
        self.gpu_weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("CUDA weights are not uploaded".to_owned()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_runtime(&self) -> Result<&GpuRuntimeBuffers> {
        self.gpu_buffers
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("CUDA runtime buffers are not allocated".to_owned()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_forward(&self) -> Result<&GpuForwardBuffers> {
        self.gpu_forward
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("CUDA forward buffers are not allocated".to_owned()))
    }
}

#[cfg(feature = "cuda")]
fn layer_common_input_norm(layer: &LayerWeights) -> &TensorInfo {
    match layer {
        LayerWeights::LinearAttention(layer) => &layer.common.input_layernorm,
        LayerWeights::FullAttention(layer) => &layer.common.input_layernorm,
    }
}

#[cfg(feature = "cuda")]
fn layer_common_post_norm(layer: &LayerWeights) -> &TensorInfo {
    match layer {
        LayerWeights::LinearAttention(layer) => &layer.common.post_attention_layernorm,
        LayerWeights::FullAttention(layer) => &layer.common.post_attention_layernorm,
    }
}

#[cfg(feature = "cuda")]
impl Engine<CudaBackend> {
    pub fn cuda_with_mapped_weights(model: &MappedModel, config: EngineConfig) -> Result<Self> {
        let manifest = ModelWeightsManifest::from_layout(&model.layout)?;
        let gpu_weights = GpuWeightStore::upload_required(model, &manifest)?;
        let mut engine = Self::new(model.layout.topology.clone(), config, CudaBackend);
        let gpu_buffers = GpuRuntimeBuffers::allocate(&engine.state, 256 * 1024 * 1024)?;
        let gpu_forward = GpuForwardBuffers::allocate(&engine.topology)?;
        engine.weights = Some(manifest);
        engine.gpu_weights = Some(gpu_weights);
        engine.gpu_buffers = Some(gpu_buffers);
        engine.gpu_forward = Some(gpu_forward);
        Ok(engine)
    }

    pub fn gpu_weight_summary(&self) -> Option<(usize, u64)> {
        self.gpu_weights
            .as_ref()
            .map(|weights| (weights.tensor_count(), weights.total_bytes()))
    }

    pub fn gpu_buffer_bytes(&self) -> Option<u64> {
        Some(self.gpu_buffers.as_ref()?.total_bytes() + self.gpu_forward.as_ref()?.total_bytes())
    }
}
