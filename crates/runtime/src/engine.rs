use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
use qwen36_fp4_core::TensorInfo;
use qwen36_fp4_core::{CoreError, KvCacheDtype, ModelLayout, ModelTopology, Result};
#[cfg(feature = "cuda")]
use qwen36_fp4_kernels::{
    AttentionDecodeSpec, AttentionPrefillSpec, AttentionShape, Bf16GemmSpec, Bf16MatVecSpec,
    Conv1dPrefillSpec, Conv1dUpdateSpec, CopyStridedRowsSpec, CublasLtFp4ScaleMode, CudaBackend,
    DeltaNetDecodeSpec, DeltaNetShape, DevicePtr, EmbeddingLookupSpec, GdnGateSpec, Nvfp4GemmSpec,
    Nvfp4QuantizeRowsSpec, Nvfp4QuantizeSpec, PartialRopeSpec, QProjDeinterleaveSpec,
    QProjSigmoidGateSpec, RmsNormNvfp4QuantizeSpec, RmsNormSpec, SamplingSpec, SwiGluSpec,
};
use qwen36_fp4_kernels::{KernelBackend, NoCudaBackend};
#[cfg(feature = "cuda")]
use qwen36_fp4_loader::MappedModel;

use crate::cuda_graph::CudaGraphPlan;
#[cfg(feature = "cuda")]
use crate::gpu::{GpuForwardBuffers, GpuPrefillBuffers, GpuRuntimeBuffers, GpuWeightStore};
use crate::kv_cache::KvCachePlan;
use crate::state::{DeltaNetStatePlan, RuntimeState};
use crate::weights::ModelWeightsManifest;
#[cfg(feature = "cuda")]
use crate::weights::{
    CommonLayerWeights, FullAttentionLayerWeights, LayerWeights, LinearAttentionLayerWeights,
    LinearWeightBinding,
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
            mtp_speculative_tokens: 0,
            cuda_graphs: CudaGraphPlan::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardOutput {
    pub logits_device_ptr: u64,
    pub produced_tokens: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MtpVerifyResult {
    pub accepted: bool,
    pub verified_token: u32,
    pub next_token: Option<u32>,
    pub next_draft_token: Option<u32>,
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
    #[cfg(feature = "cuda")]
    pub gpu_prefill: Option<GpuPrefillBuffers>,
    /// Capture-mode CUDA stream + instantiated decode-and-sample graph. When
    /// `Some`, decode kernels read the current position from `forward.position_i32`
    /// instead of a host scalar so the same graph can replay across iterations.
    #[cfg(feature = "cuda")]
    decode_graph: Option<DecodeGraphState>,
    backend: B,
}

#[cfg(feature = "cuda")]
struct DecodeGraphState {
    stream: qwen36_fp4_kernels::graph::OwnedCudaStream,
    exec: qwen36_fp4_kernels::graph::CudaGraphExec,
    raw_graph: qwen36_fp4_kernels::graph::CudaGraph,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct Nvfp4ActivationQuant<'a> {
    in_features: usize,
    input_scale: &'a TensorInfo,
}

#[cfg(feature = "cuda")]
impl Drop for DecodeGraphState {
    fn drop(&mut self) {
        // The graph_exec and raw graph must be torn down before the owning
        // stream goes away. We swallow errors here because Drop has no return
        // path; surfacing them would mask the original error that triggered
        // engine shutdown.
        let _ = qwen36_fp4_kernels::graph::destroy_graph_exec(self.exec);
        let _ = qwen36_fp4_kernels::graph::destroy_graph(self.raw_graph);
        // Reset the global active stream so any kernels run after engine
        // teardown go back to the legacy default stream.
        qwen36_fp4_kernels::graph::set_active_stream(qwen36_fp4_kernels::graph::CudaStream::NULL);
    }
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
            #[cfg(feature = "cuda")]
            gpu_prefill: None,
            #[cfg(feature = "cuda")]
            decode_graph: None,
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
        self.decode_one_with_sync(token, true)
    }

    #[cfg(feature = "cuda")]
    pub fn decode_one_queued(&mut self, token: u32) -> Result<ForwardOutput> {
        self.decode_one_with_sync(token, false)
    }

    #[cfg(feature = "cuda")]
    pub fn decode_sampled_queued(&mut self) -> Result<ForwardOutput> {
        self.decode_sampled_with_sync(false)
    }

    fn decode_one_with_sync(&mut self, token: u32, sync_after: bool) -> Result<ForwardOutput> {
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda("engine_decode_one"));
        }
        #[cfg(feature = "cuda")]
        {
            self.forward_token_cuda(token, self.state.position, true, sync_after)?;
            self.state.advance(1);
            Ok(ForwardOutput {
                logits_device_ptr: self.cuda_forward()?.logits.ptr().0,
                produced_tokens: 1,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = token;
            let _ = sync_after;
            Err(CoreError::UnsupportedNoCuda("engine_decode_one"))
        }
    }

    #[cfg(feature = "cuda")]
    fn decode_sampled_with_sync(&mut self, sync_after: bool) -> Result<ForwardOutput> {
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda("engine_decode_one"));
        }
        self.forward_sampled_token_cuda(self.state.position, true, sync_after)?;
        self.state.advance(1);
        Ok(ForwardOutput {
            logits_device_ptr: self.cuda_forward()?.logits.ptr().0,
            produced_tokens: 1,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn sample_greedy(&self) -> Result<u32> {
        self.queue_sample_greedy()?;
        self.read_sampled_token()
    }

    #[cfg(feature = "cuda")]
    pub fn read_sampled_token(&self) -> Result<u32> {
        let mut token = [0_u8; 4];
        self.cuda_forward()?
            .sampled_token_u32
            .copy_to_host(&mut token)?;
        Ok(u32::from_ne_bytes(token))
    }

    #[cfg(feature = "cuda")]
    pub fn read_current_token(&self) -> Result<u32> {
        let mut token = [0_u8; 4];
        self.cuda_forward()?.token_u32.copy_to_host(&mut token)?;
        Ok(u32::from_ne_bytes(token))
    }

    /// Queue the greedy-sample kernel without copying the result back to the
    /// host. Useful when the next forward pass consumes `sampled_token_u32`
    /// directly via `decode_sampled_queued`, so the host stays off the
    /// critical path until a final `cuda_synchronize`.
    #[cfg(feature = "cuda")]
    pub fn queue_sample_greedy(&self) -> Result<()> {
        self.queue_sample_greedy_into(self.cuda_forward()?.sampled_token_u32.ptr())
    }

    #[cfg(feature = "cuda")]
    pub fn queue_sample_greedy_to_current_token(&self) -> Result<()> {
        self.queue_sample_greedy_into(self.cuda_forward()?.token_u32.ptr())
    }

    #[cfg(feature = "cuda")]
    fn queue_sample_greedy_into(&self, output_token_u32: DevicePtr) -> Result<()> {
        self.backend.sample(&SamplingSpec {
            vocab_size: self.topology.vocab_size,
            logits_bf16: self.cuda_forward()?.logits.ptr(),
            output_token_u32,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        })
    }

    /// Capture a single decode-and-sample iteration into a CUDA graph for
    /// replay. After this returns, [`decode_graph_step`] can be called
    /// repeatedly without going through the regular host launch path,
    /// dropping ~600 host kernel launches per token down to one
    /// `cudaGraphLaunch`.
    ///
    /// Preconditions:
    /// - `prefill` has run, so `state.position` is the prompt length and
    ///   `forward.logits` holds the logits used by the first sample.
    /// - `queue_sample_greedy` has been called once already, populating
    ///   `forward.sampled_token_u32` from the prefill logits. The graph
    ///   captures `decode_sampled_queued` + `queue_sample_greedy` + a
    ///   device-side position increment, so each replay produces exactly
    ///   one new token.
    #[cfg(feature = "cuda")]
    pub fn enable_decode_graph(&mut self) -> Result<()> {
        use qwen36_fp4_kernels::graph::{self, CudaStream};

        if self.decode_graph.is_some() {
            return Ok(());
        }
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda("enable_decode_graph"));
        }

        // Seed the device-side position counter with the current position.
        let position_i32 = i32::try_from(self.state.position).map_err(|_| {
            CoreError::Runtime(format!(
                "position {} does not fit i32 for graph capture",
                self.state.position
            ))
        })?;
        let position_buffer_ptr = self.cuda_forward()?.position_i32.ptr();
        self.cuda_forward()?
            .position_i32
            .copy_from_host(&position_i32.to_ne_bytes())?;

        // Allocate a non-default stream and route every kernel through it
        // so the entire forward + sample + increment can be captured.
        let stream = CudaStream::create()?;
        let stream_handle = stream.handle();
        graph::set_active_stream(stream_handle);

        // Capture: decode forward (using device position) + sample +
        // increment_i32. Wrapped in a closure so any error reverts the
        // active stream and avoids leaking the capture session.
        let capture_result = (|| -> Result<(graph::CudaGraph, graph::CudaGraphExec)> {
            graph::begin_capture(stream_handle)?;
            self.forward_device_token_cuda_inner(
                self.cuda_forward()?.sampled_token_u32.ptr(),
                self.state.position,
                position_buffer_ptr,
                true,
                false,
            )?;
            self.queue_sample_greedy()?;
            graph::increment_i32(position_buffer_ptr)?;
            let raw_graph = graph::end_capture(stream_handle)?;
            let exec = graph::instantiate(raw_graph)?;
            Ok((raw_graph, exec))
        })();

        let (raw_graph, exec) = match capture_result {
            Ok(value) => value,
            Err(err) => {
                graph::set_active_stream(CudaStream::NULL);
                return Err(err);
            }
        };

        // Stream capture records the decode/sample/increment sequence but does
        // not execute it. Launch once here so callers can enable the graph and
        // immediately read the next sampled token, matching the previous
        // host-launched decode semantics.
        graph::launch(exec, stream_handle)?;
        self.state.advance(1);

        self.decode_graph = Some(DecodeGraphState {
            stream,
            exec,
            raw_graph,
        });
        Ok(())
    }

    /// Replay the captured decode-and-sample graph once. Callers must ensure
    /// `enable_decode_graph` was called first.
    #[cfg(feature = "cuda")]
    pub fn decode_graph_step(&mut self) -> Result<()> {
        let graph_state = self.decode_graph.as_ref().ok_or_else(|| {
            CoreError::Runtime("decode_graph_step called without an active capture".to_owned())
        })?;
        qwen36_fp4_kernels::graph::launch(graph_state.exec, graph_state.stream.handle())?;
        // Mirror the device-side position bump on the host so callers that
        // read `state.position` see the truth.
        self.state.advance(1);
        Ok(())
    }

    /// Capture one MTP verification iteration:
    /// main decode from `forward.token_u32`, greedy-sample verified token back
    /// into `forward.token_u32`, run the MTP layer from that verified token and
    /// the target hidden state, then greedy-sample the next draft into
    /// `forward.sampled_token_u32`.
    #[cfg(feature = "cuda")]
    pub fn enable_mtp_decode_graph(&mut self) -> Result<()> {
        use qwen36_fp4_kernels::graph::{self, CudaStream};

        if self.decode_graph.is_some() {
            return Ok(());
        }
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "enable_mtp_decode_graph called with MTP disabled".to_owned(),
            ));
        }
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda("enable_mtp_decode_graph"));
        }

        let position_i32 = i32::try_from(self.state.position).map_err(|_| {
            CoreError::Runtime(format!(
                "position {} does not fit i32 for graph capture",
                self.state.position
            ))
        })?;
        let forward = self.cuda_forward()?;
        let position_buffer_ptr = forward.position_i32.ptr();
        forward
            .position_i32
            .copy_from_host(&position_i32.to_ne_bytes())?;

        let stream = CudaStream::create()?;
        let stream_handle = stream.handle();
        graph::set_active_stream(stream_handle);

        let capture_result = (|| -> Result<(graph::CudaGraph, graph::CudaGraphExec)> {
            graph::begin_capture(stream_handle)?;
            self.forward_device_token_cuda_inner(
                self.cuda_forward()?.token_u32.ptr(),
                self.state.position,
                position_buffer_ptr,
                true,
                false,
            )?;
            self.queue_sample_greedy_into(self.cuda_forward()?.token_u32.ptr())?;
            self.run_mtp_decode_from_target_hidden(
                self.cuda_forward()?.token_u32.ptr(),
                self.state.position,
                position_buffer_ptr,
                self.cuda_forward()?.normed.ptr(),
            )?;
            self.queue_sample_greedy_into(self.cuda_forward()?.sampled_token_u32.ptr())?;
            graph::increment_i32(position_buffer_ptr)?;
            let raw_graph = graph::end_capture(stream_handle)?;
            let exec = graph::instantiate(raw_graph)?;
            Ok((raw_graph, exec))
        })();

        let (raw_graph, exec) = match capture_result {
            Ok(value) => value,
            Err(err) => {
                graph::set_active_stream(CudaStream::NULL);
                return Err(err);
            }
        };

        graph::launch(exec, stream_handle)?;
        self.state.advance(1);

        self.decode_graph = Some(DecodeGraphState {
            stream,
            exec,
            raw_graph,
        });
        Ok(())
    }

    /// Tear down the captured decode graph and restore the default stream.
    /// Idempotent.
    #[cfg(feature = "cuda")]
    pub fn disable_decode_graph(&mut self) -> Result<()> {
        if let Some(graph_state) = self.decode_graph.take() {
            // Drain any in-flight launches before freeing the graph.
            graph_state.stream.handle().synchronize()?;
            // Drop runs the destructors below.
            drop(graph_state);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn reset_cuda_state(&mut self) -> Result<()> {
        self.disable_decode_graph()?;
        let runtime = self.cuda_runtime()?;
        if let Some(kv_cache) = &runtime.kv_cache {
            kv_cache.memset(0)?;
        }
        if let Some(mtp_kv_cache) = &runtime.mtp_kv_cache {
            mtp_kv_cache.memset(0)?;
        }
        runtime.deltanet_state.memset(0)?;
        runtime.deltanet_checkpoint.memset(0)?;
        runtime.conv_history.memset(0)?;
        self.state.position = 0;
        self.state.accepted_tokens = 0;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn prefill_cuda(&mut self, prompt_tokens: &[u32]) -> Result<ForwardOutput> {
        if prompt_tokens.is_empty() {
            return Err(CoreError::Runtime(
                "prefill requires at least one prompt token".to_owned(),
            ));
        }
        if self.state.position + prompt_tokens.len() > self.config.max_context {
            return Err(CoreError::Runtime(format!(
                "prefill of {} tokens at position {} exceeds configured max_context {}",
                prompt_tokens.len(),
                self.state.position,
                self.config.max_context
            )));
        }
        if self.config.kv_cache_dtype != KvCacheDtype::Bf16 {
            return Err(CoreError::Runtime(
                "reference CUDA scheduler currently requires BF16 KV cache".to_owned(),
            ));
        }

        let mut consumed = 0;
        while consumed < prompt_tokens.len() {
            let start_position = self.state.position;
            let capacity = self.cuda_prefill()?.capacity;
            let chunk = (prompt_tokens.len() - consumed).min(capacity);
            let chunk_tokens = &prompt_tokens[consumed..consumed + chunk];
            let mut token_bytes = Vec::with_capacity(chunk * 4);
            for token in chunk_tokens {
                token_bytes.extend_from_slice(&token.to_ne_bytes());
            }
            let mut position_bytes = Vec::with_capacity(chunk * 4);
            for idx in 0..chunk {
                let position = start_position + idx;
                let position = i32::try_from(position).map_err(|_| {
                    CoreError::Runtime(format!("position {position} does not fit i32 for RoPE"))
                })?;
                position_bytes.extend_from_slice(&position.to_ne_bytes());
            }
            {
                let prefill = self.cuda_prefill()?;
                prefill.token_u32.copy_from_host(&token_bytes)?;
                prefill.position_i32.copy_from_host(&position_bytes)?;
            }

            let emit_logits = consumed + chunk == prompt_tokens.len();
            self.prefill_cuda_chunk(chunk, start_position, emit_logits)?;
            if self.config.mtp_speculative_tokens > 0 && !emit_logits {
                let shifted_tokens = &prompt_tokens[consumed + 1..consumed + chunk + 1];
                self.run_mtp_prefill_chunk_from_current_prefill(
                    chunk,
                    start_position,
                    shifted_tokens,
                    false,
                )?;
            }
            self.state.advance(chunk);
            consumed += chunk;
        }
        qwen36_fp4_kernels::cuda_synchronize()?;
        Ok(ForwardOutput {
            logits_device_ptr: self.cuda_forward()?.logits.ptr().0,
            produced_tokens: prompt_tokens.len(),
        })
    }

    #[cfg(feature = "cuda")]
    pub fn prepare_mtp_prefill_from_sampled(
        &self,
        prompt_tokens: &[u32],
        sampled_token: u32,
    ) -> Result<()> {
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP prefill requested while mtp_speculative_tokens is 0".to_owned(),
            ));
        }
        if prompt_tokens.is_empty() {
            return Err(CoreError::Runtime(
                "MTP prefill requires at least one prompt token".to_owned(),
            ));
        }

        let capacity = self.cuda_prefill()?.capacity;
        let final_chunk = (prompt_tokens.len() - 1) % capacity + 1;
        let start = prompt_tokens.len() - final_chunk;
        let mut shifted_tokens = Vec::with_capacity(final_chunk);
        for idx in start..prompt_tokens.len() {
            let token = prompt_tokens.get(idx + 1).copied().unwrap_or(sampled_token);
            shifted_tokens.push(token);
        }
        self.run_mtp_prefill_chunk_from_current_prefill(final_chunk, start, &shifted_tokens, true)
    }

    #[cfg(feature = "cuda")]
    pub fn prepare_mtp_decode_from_sampled(&self, target_position: usize) -> Result<()> {
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP decode requested while mtp_speculative_tokens is 0".to_owned(),
            ));
        }
        let forward = self.cuda_forward()?;
        self.run_mtp_decode_from_target_hidden(
            forward.sampled_token_u32.ptr(),
            target_position,
            DevicePtr::NULL,
            forward.normed.ptr(),
        )
    }

    #[cfg(feature = "cuda")]
    pub fn verify_mtp_draft_two_tokens(
        &mut self,
        current_token: u32,
        draft_token: u32,
    ) -> Result<MtpVerifyResult> {
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP two-token verification requested while mtp_speculative_tokens is 0".to_owned(),
            ));
        }
        let start_position = self.state.position;
        if start_position + 2 > self.config.max_context {
            return Err(CoreError::Runtime(format!(
                "MTP two-token verification at position {start_position} exceeds max_context {}",
                self.config.max_context
            )));
        }

        let mut token_bytes = Vec::with_capacity(8);
        token_bytes.extend_from_slice(&current_token.to_ne_bytes());
        token_bytes.extend_from_slice(&draft_token.to_ne_bytes());
        let mut position_bytes = Vec::with_capacity(8);
        for position in [start_position, start_position + 1] {
            let position = i32::try_from(position).map_err(|_| {
                CoreError::Runtime(format!("position {position} does not fit i32 for RoPE"))
            })?;
            position_bytes.extend_from_slice(&position.to_ne_bytes());
        }
        {
            let prefill = self.cuda_prefill()?;
            prefill.token_u32.copy_from_host(&token_bytes)?;
            prefill.position_i32.copy_from_host(&position_bytes)?;
        }

        // This mutates the target recurrent state through both tokens. If the
        // draft is rejected, the caller must rebuild from committed tokens via
        // `reset_cuda_state`; we avoid a large per-step full-state snapshot in
        // the overwhelmingly common accepted path.
        self.prefill_cuda_chunk(2, start_position, false)?;
        self.final_norm_prefill_rows(2)?;

        self.prefill_row_logits(0)?;
        self.queue_sample_greedy_to_current_token()?;
        let verified_token = self.read_current_token()?;
        if verified_token != draft_token {
            return Ok(MtpVerifyResult {
                accepted: false,
                verified_token,
                next_token: None,
                next_draft_token: None,
            });
        }

        self.prefill_row_logits(1)?;
        self.queue_sample_greedy_to_current_token()?;
        let next_token = self.read_current_token()?;

        let mut mtp_tokens = Vec::with_capacity(8);
        mtp_tokens.extend_from_slice(&draft_token.to_ne_bytes());
        mtp_tokens.extend_from_slice(&next_token.to_ne_bytes());
        self.cuda_prefill()?.token_u32.copy_from_host(&mtp_tokens)?;
        self.run_mtp_prefill_chunk(2, start_position, self.cuda_prefill()?.normed.ptr(), true)?;
        self.queue_sample_greedy()?;
        let next_draft_token = self.read_sampled_token()?;
        self.state.advance(2);

        Ok(MtpVerifyResult {
            accepted: true,
            verified_token,
            next_token: Some(next_token),
            next_draft_token: Some(next_draft_token),
        })
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_prefill_chunk_from_current_prefill(
        &self,
        tokens: usize,
        start_position: usize,
        shifted_tokens: &[u32],
        emit_logits: bool,
    ) -> Result<()> {
        if shifted_tokens.len() != tokens {
            return Err(CoreError::Runtime(format!(
                "MTP shifted token count {} does not match chunk size {tokens}",
                shifted_tokens.len()
            )));
        }
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let prefill = self.cuda_prefill()?;
        let mut token_bytes = Vec::with_capacity(tokens * 4);
        for token in shifted_tokens {
            token_bytes.extend_from_slice(&token.to_ne_bytes());
        }
        prefill.token_u32.copy_from_host(&token_bytes)?;
        self.rmsnorm(
            tokens,
            self.topology.hidden_size,
            prefill.hidden.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &manifest.final_norm)?,
            prefill.residual.ptr(),
            DevicePtr::NULL,
            prefill.normed.ptr(),
        )?;
        self.run_mtp_prefill_chunk(tokens, start_position, prefill.normed.ptr(), emit_logits)
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_prefill_chunk(
        &self,
        tokens: usize,
        start_position: usize,
        target_hidden_bf16: DevicePtr,
        emit_logits: bool,
    ) -> Result<()> {
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let mtp = manifest
            .mtp
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("MTP weights are not available".to_owned()))?;
        let layer = mtp
            .layer(0)
            .ok_or_else(|| CoreError::Runtime("MTP has no layers".to_owned()))?;
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let prefill = self.cuda_prefill()?;
        let hidden = self.topology.hidden_size;

        self.backend.embedding_lookup(&EmbeddingLookupSpec {
            tokens,
            hidden,
            vocab_size: self.topology.vocab_size,
            token_ids_u32: prefill.token_u32.ptr(),
            embedding_bf16: self.tensor_ptr(weights, &manifest.embed_tokens)?,
            output_bf16: prefill.hidden.ptr(),
        })?;
        self.rmsnorm(
            tokens,
            hidden,
            prefill.hidden.ptr(),
            self.tensor_ptr(weights, &mtp.pre_fc_norm_embedding)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.aux.ptr(),
        )?;
        self.rmsnorm(
            tokens,
            hidden,
            target_hidden_bf16,
            self.tensor_ptr(weights, &mtp.pre_fc_norm_hidden)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.block_out.ptr(),
        )?;
        self.concat_mtp_fc_input_rows(
            tokens,
            prefill.aux.ptr(),
            prefill.block_out.ptr(),
            prefill.qkv.ptr(),
        )?;
        self.linear_rows(
            &mtp.fc,
            prefill.qkv.ptr(),
            prefill.hidden.ptr(),
            tokens,
            prefill,
        )?;

        self.rmsnorm(
            tokens,
            hidden,
            prefill.hidden.ptr(),
            self.tensor_ptr(weights, &layer.common.input_layernorm)?,
            DevicePtr::NULL,
            prefill.residual.ptr(),
            prefill.normed.ptr(),
        )?;
        self.run_mtp_full_attention_layer_prefill(layer, runtime, prefill, tokens, start_position)?;
        self.rmsnorm(
            tokens,
            hidden,
            prefill.block_out.ptr(),
            self.tensor_ptr(weights, &layer.common.post_attention_layernorm)?,
            prefill.residual.ptr(),
            prefill.residual.ptr(),
            prefill.normed.ptr(),
        )?;
        self.run_mtp_mlp_prefill(&layer.common, prefill, tokens)?;
        self.rmsnorm(
            tokens,
            hidden,
            prefill.hidden.ptr(),
            self.tensor_ptr(weights, &mtp.norm)?,
            prefill.residual.ptr(),
            DevicePtr::NULL,
            prefill.normed.ptr(),
        )?;
        if emit_logits {
            let last_hidden = Self::ptr_offset(prefill.normed.ptr(), (tokens - 1) * hidden * 2)?;
            self.bf16_matvec(
                &manifest.lm_head,
                last_hidden,
                self.cuda_forward()?.logits.ptr(),
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_decode_from_target_hidden(
        &self,
        token_ids_u32: DevicePtr,
        position: usize,
        position_device_i32: DevicePtr,
        target_hidden_bf16: DevicePtr,
    ) -> Result<()> {
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let mtp = manifest
            .mtp
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("MTP weights are not available".to_owned()))?;
        let layer = mtp
            .layer(0)
            .ok_or_else(|| CoreError::Runtime("MTP has no layers".to_owned()))?;
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let forward = self.cuda_forward()?;
        let hidden = self.topology.hidden_size;

        self.backend.embedding_lookup(&EmbeddingLookupSpec {
            tokens: 1,
            hidden,
            vocab_size: self.topology.vocab_size,
            token_ids_u32,
            embedding_bf16: self.tensor_ptr(weights, &manifest.embed_tokens)?,
            output_bf16: forward.hidden.ptr(),
        })?;
        self.rmsnorm(
            1,
            hidden,
            forward.hidden.ptr(),
            self.tensor_ptr(weights, &mtp.pre_fc_norm_embedding)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.aux.ptr(),
        )?;
        self.rmsnorm(
            1,
            hidden,
            target_hidden_bf16,
            self.tensor_ptr(weights, &mtp.pre_fc_norm_hidden)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.block_out.ptr(),
        )?;
        self.concat_mtp_fc_input_rows(
            1,
            forward.aux.ptr(),
            forward.block_out.ptr(),
            forward.qkv.ptr(),
        )?;
        self.linear(&mtp.fc, forward.qkv.ptr(), forward.hidden.ptr())?;

        self.rmsnorm(
            1,
            hidden,
            forward.hidden.ptr(),
            self.tensor_ptr(weights, &layer.common.input_layernorm)?,
            DevicePtr::NULL,
            forward.residual.ptr(),
            forward.normed.ptr(),
        )?;
        self.run_mtp_full_attention_layer_decode(
            layer,
            runtime,
            forward,
            position,
            position_device_i32,
        )?;
        self.rmsnorm(
            1,
            hidden,
            forward.block_out.ptr(),
            self.tensor_ptr(weights, &layer.common.post_attention_layernorm)?,
            forward.residual.ptr(),
            forward.residual.ptr(),
            forward.normed.ptr(),
        )?;
        self.run_mtp_mlp_decode(&layer.common, forward)?;
        self.rmsnorm(
            1,
            hidden,
            forward.hidden.ptr(),
            self.tensor_ptr(weights, &mtp.norm)?,
            forward.residual.ptr(),
            DevicePtr::NULL,
            forward.normed.ptr(),
        )?;
        self.bf16_matvec(
            &manifest.lm_head,
            forward.normed.ptr(),
            forward.logits.ptr(),
        )
    }

    #[cfg(feature = "cuda")]
    fn prefill_cuda_chunk(
        &self,
        tokens: usize,
        start_position: usize,
        emit_logits: bool,
    ) -> Result<()> {
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let prefill = self.cuda_prefill()?;

        self.backend.embedding_lookup(&EmbeddingLookupSpec {
            tokens,
            hidden: self.topology.hidden_size,
            vocab_size: self.topology.vocab_size,
            token_ids_u32: prefill.token_u32.ptr(),
            embedding_bf16: self.tensor_ptr(weights, &manifest.embed_tokens)?,
            output_bf16: prefill.hidden.ptr(),
        })?;

        let trace_layers = std::env::var("QWEN36_DEBUG_LAYER_TRACE").is_ok();
        let dump_dir = std::env::var("QWEN36_DEBUG_DUMP_DIR").ok();
        let dump_all_layers = std::env::var("QWEN36_DEBUG_DUMP_ALL_LAYERS").is_ok();
        if trace_layers {
            self.trace_buffer_stats(
                "post-embed",
                prefill.hidden.ptr(),
                tokens * self.topology.hidden_size,
            )?;
        }
        if let Some(dir) = &dump_dir {
            self.dump_buffer_to_disk(
                dir,
                "post_embed.bf16",
                prefill.hidden.ptr(),
                tokens * self.topology.hidden_size,
            )?;
        }

        let mut residual_initialized = false;
        for (layer_idx, layer) in manifest.layers.iter().enumerate() {
            let input_residual = if residual_initialized {
                prefill.residual.ptr()
            } else {
                DevicePtr::NULL
            };
            self.rmsnorm(
                tokens,
                self.topology.hidden_size,
                prefill.hidden.ptr(),
                self.tensor_ptr(weights, layer_common_input_norm(layer))?,
                input_residual,
                prefill.residual.ptr(),
                prefill.normed.ptr(),
            )?;
            residual_initialized = true;
            if dump_all_layers {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_input_normed.bf16"),
                        prefill.normed.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_input_residual.bf16"),
                        prefill.residual.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                }
            }

            let quantized_normed = if let Some(quantized) = Self::layer_input_nvfp4_quant(layer)? {
                self.quantize_nvfp4_activation_rows(
                    prefill.normed.ptr(),
                    tokens,
                    quantized,
                    prefill,
                )?;
                Some(quantized)
            } else {
                None
            };

            match layer {
                LayerWeights::LinearAttention(layer) => self.run_linear_attention_layer_prefill(
                    layer,
                    runtime,
                    prefill,
                    tokens,
                    quantized_normed,
                )?,
                LayerWeights::FullAttention(layer) => self.run_full_attention_layer_prefill(
                    layer,
                    runtime,
                    prefill,
                    tokens,
                    start_position,
                    quantized_normed,
                )?,
            }

            if trace_layers {
                let kind = match layer {
                    LayerWeights::LinearAttention(_) => "deltanet",
                    LayerWeights::FullAttention(_) => "fullattn",
                };
                self.trace_buffer_stats(
                    &format!("layer{layer_idx:02}.{kind}.attn_out"),
                    prefill.block_out.ptr(),
                    tokens * self.topology.hidden_size,
                )?;
            }
            if dump_all_layers {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_attn_out.bf16"),
                        prefill.block_out.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                }
            }
            if let Some(dir) = &dump_dir {
                if layer_idx == 0 {
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_attn_out.bf16",
                        prefill.block_out.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_normed.bf16",
                        prefill.normed.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                } else if layer_idx == 3 {
                    self.dump_buffer_to_disk(
                        dir,
                        "layer3_normed.bf16",
                        prefill.normed.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                }
            }

            let common = layer_common(layer);
            let mlp_input_linears = [
                (&common.mlp_gate_proj, DevicePtr::NULL),
                (&common.mlp_up_proj, DevicePtr::NULL),
            ];
            self.rmsnorm(
                tokens,
                self.topology.hidden_size,
                prefill.block_out.ptr(),
                self.tensor_ptr(weights, &common.post_attention_layernorm)?,
                prefill.residual.ptr(),
                prefill.residual.ptr(),
                prefill.normed.ptr(),
            )?;
            if dump_all_layers {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_post_attn_normed.bf16"),
                        prefill.normed.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_residual_after_attn.bf16"),
                        prefill.residual.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                }
            }
            if let Some(quantized) = Self::common_nvfp4_quant(&mlp_input_linears)? {
                self.quantize_nvfp4_activation_rows(
                    prefill.normed.ptr(),
                    tokens,
                    quantized,
                    prefill,
                )?;
                self.run_mlp_with_quantized_input_prefill(layer, prefill, tokens, quantized)?;
            } else {
                self.run_mlp_prefill(layer, prefill, tokens)?;
            }
            if dump_all_layers {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_mlp_gate.bf16"),
                        prefill.aux.ptr(),
                        tokens * self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_mlp_up.bf16"),
                        prefill.aux2.ptr(),
                        tokens * self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_mlp_swiglu.bf16"),
                        prefill.aux3.ptr(),
                        tokens * self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("layer{layer_idx:02}_mlp_out.bf16"),
                        prefill.hidden.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                }
            }

            if layer_idx == 0 {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_mlp_out.bf16",
                        prefill.hidden.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_mlp_gate.bf16",
                        prefill.aux.ptr(),
                        tokens * self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_mlp_up.bf16",
                        prefill.aux2.ptr(),
                        tokens * self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_mlp_swiglu.bf16",
                        prefill.aux3.ptr(),
                        tokens * self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_post_attn_normed.bf16",
                        prefill.normed.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        "layer0_residual.bf16",
                        prefill.residual.ptr(),
                        tokens * self.topology.hidden_size,
                    )?;
                }
            }

            if trace_layers {
                self.trace_buffer_stats(
                    &format!("layer{layer_idx:02}.mlp_out"),
                    prefill.hidden.ptr(),
                    tokens * self.topology.hidden_size,
                )?;
                self.trace_buffer_stats(
                    &format!("layer{layer_idx:02}.residual"),
                    prefill.residual.ptr(),
                    tokens * self.topology.hidden_size,
                )?;
            }
        }

        if emit_logits {
            let last_hidden = Self::ptr_offset(
                prefill.hidden.ptr(),
                (tokens - 1) * self.topology.hidden_size * 2,
            )?;
            let last_residual = Self::ptr_offset(
                prefill.residual.ptr(),
                (tokens - 1) * self.topology.hidden_size * 2,
            )?;
            let forward = self.cuda_forward()?;
            // Dump pre-final-norm residual + hidden for the parity harness.
            if let Some(dir) = &dump_dir {
                self.dump_buffer_to_disk(
                    dir,
                    "final_last_hidden.bf16",
                    last_hidden,
                    self.topology.hidden_size,
                )?;
                self.dump_buffer_to_disk(
                    dir,
                    "final_last_residual.bf16",
                    last_residual,
                    self.topology.hidden_size,
                )?;
            }
            self.rmsnorm(
                1,
                self.topology.hidden_size,
                last_hidden,
                self.tensor_ptr(weights, &manifest.final_norm)?,
                last_residual,
                DevicePtr::NULL,
                forward.normed.ptr(),
            )?;
            if let Some(dir) = &dump_dir {
                self.dump_buffer_to_disk(
                    dir,
                    "final_normed.bf16",
                    forward.normed.ptr(),
                    self.topology.hidden_size,
                )?;
            }
            self.bf16_matvec(
                &manifest.lm_head,
                forward.normed.ptr(),
                forward.logits.ptr(),
            )?;
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn final_norm_prefill_rows(&self, tokens: usize) -> Result<()> {
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let prefill = self.cuda_prefill()?;
        self.rmsnorm(
            tokens,
            self.topology.hidden_size,
            prefill.hidden.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &manifest.final_norm)?,
            prefill.residual.ptr(),
            DevicePtr::NULL,
            prefill.normed.ptr(),
        )
    }

    #[cfg(feature = "cuda")]
    fn prefill_row_logits(&self, row: usize) -> Result<()> {
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        let prefill = self.cuda_prefill()?;
        let hidden = Self::ptr_offset(prefill.normed.ptr(), row * self.topology.hidden_size * 2)?;
        self.bf16_matvec(&manifest.lm_head, hidden, self.cuda_forward()?.logits.ptr())
    }

    #[cfg(feature = "cuda")]
    fn forward_token_cuda(
        &self,
        token: u32,
        position: usize,
        emit_logits: bool,
        sync_after: bool,
    ) -> Result<()> {
        let forward = self.cuda_forward()?;
        forward.token_u32.copy_from_host(&token.to_ne_bytes())?;
        self.forward_device_token_cuda(forward.token_u32.ptr(), position, emit_logits, sync_after)
    }

    #[cfg(feature = "cuda")]
    fn forward_sampled_token_cuda(
        &self,
        position: usize,
        emit_logits: bool,
        sync_after: bool,
    ) -> Result<()> {
        self.forward_device_token_cuda(
            self.cuda_forward()?.sampled_token_u32.ptr(),
            position,
            emit_logits,
            sync_after,
        )
    }

    #[cfg(feature = "cuda")]
    fn forward_device_token_cuda(
        &self,
        token_ids_u32: DevicePtr,
        position: usize,
        emit_logits: bool,
        sync_after: bool,
    ) -> Result<()> {
        self.forward_device_token_cuda_inner(
            token_ids_u32,
            position,
            DevicePtr::NULL,
            emit_logits,
            sync_after,
        )
    }

    #[cfg(feature = "cuda")]
    fn forward_device_token_cuda_inner(
        &self,
        token_ids_u32: DevicePtr,
        position: usize,
        position_device_i32: DevicePtr,
        emit_logits: bool,
        sync_after: bool,
    ) -> Result<()> {
        // The bound check applies to the host-side scalar position only. In
        // graph-capture mode the device-side counter advances per replay, so
        // the caller is responsible for not stepping the graph past
        // max_context.
        if position_device_i32 == DevicePtr::NULL && position >= self.config.max_context {
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
        let dump_decode = std::env::var("QWEN36_DEBUG_DUMP_DECODE").is_ok();
        let dump_dir = std::env::var("QWEN36_DEBUG_DUMP_DIR").ok();
        let dump_prefix = format!("decode_pos{position:05}");
        let profile_decode = std::env::var("QWEN36_PROFILE_DECODE_LAYERS").is_ok()
            && position_device_i32 == DevicePtr::NULL;

        self.backend.embedding_lookup(&EmbeddingLookupSpec {
            tokens: 1,
            hidden: self.topology.hidden_size,
            vocab_size: self.topology.vocab_size,
            token_ids_u32,
            embedding_bf16: self.tensor_ptr(weights, &manifest.embed_tokens)?,
            output_bf16: forward.hidden.ptr(),
        })?;
        if dump_decode {
            if let Some(dir) = &dump_dir {
                self.dump_buffer_to_disk(
                    dir,
                    &format!("{dump_prefix}_post_embed.bf16"),
                    forward.hidden.ptr(),
                    self.topology.hidden_size,
                )?;
            }
        }

        let mut residual_initialized = false;
        for (layer_idx, layer) in manifest.layers.iter().enumerate() {
            let input_residual = if residual_initialized {
                forward.residual.ptr()
            } else {
                DevicePtr::NULL
            };
            let quantized_normed = Self::layer_input_nvfp4_quant(layer)?;
            if let Some(quantized) = quantized_normed {
                let output_bf16 = if dump_decode {
                    forward.normed.ptr()
                } else {
                    DevicePtr::NULL
                };
                self.rmsnorm_nvfp4_quantize(
                    self.topology.hidden_size,
                    forward.hidden.ptr(),
                    self.tensor_ptr(weights, layer_common_input_norm(layer))?,
                    input_residual,
                    forward.residual.ptr(),
                    output_bf16,
                    quantized.input_scale,
                )?;
            } else {
                self.rmsnorm(
                    1,
                    self.topology.hidden_size,
                    forward.hidden.ptr(),
                    self.tensor_ptr(weights, layer_common_input_norm(layer))?,
                    input_residual,
                    forward.residual.ptr(),
                    forward.normed.ptr(),
                )?;
            };
            residual_initialized = true;
            if dump_decode {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_input_normed.bf16"),
                        forward.normed.ptr(),
                        self.topology.hidden_size,
                    )?;
                }
            }

            let attn_start = profile_decode.then(std::time::Instant::now);
            match layer {
                LayerWeights::LinearAttention(layer) => {
                    self.run_linear_attention_layer(layer, runtime, forward, quantized_normed)?;
                }
                LayerWeights::FullAttention(layer) => {
                    self.run_full_attention_layer(
                        layer,
                        runtime,
                        forward,
                        position,
                        position_device_i32,
                        quantized_normed,
                    )?;
                }
            }
            if let Some(attn_start) = attn_start {
                qwen36_fp4_kernels::cuda_synchronize()?;
                eprintln!(
                    "decode.profile layer{layer_idx:02}.attn_ms={:.3}",
                    attn_start.elapsed().as_secs_f64() * 1000.0
                );
            }
            if dump_decode {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_attn_out.bf16"),
                        forward.block_out.ptr(),
                        self.topology.hidden_size,
                    )?;
                }
            }

            let common = layer_common(layer);
            let mlp_input_linears = [
                (&common.mlp_gate_proj, DevicePtr::NULL),
                (&common.mlp_up_proj, DevicePtr::NULL),
            ];
            if let Some(quantized) = Self::common_nvfp4_quant(&mlp_input_linears)? {
                let output_bf16 = if dump_decode {
                    forward.normed.ptr()
                } else {
                    DevicePtr::NULL
                };
                self.rmsnorm_nvfp4_quantize(
                    self.topology.hidden_size,
                    forward.block_out.ptr(),
                    self.tensor_ptr(weights, &common.post_attention_layernorm)?,
                    forward.residual.ptr(),
                    forward.residual.ptr(),
                    output_bf16,
                    quantized.input_scale,
                )?;
                if dump_decode {
                    if let Some(dir) = &dump_dir {
                        self.dump_buffer_to_disk(
                            dir,
                            &format!("{dump_prefix}_layer{layer_idx:02}_post_attn_normed.bf16"),
                            forward.normed.ptr(),
                            self.topology.hidden_size,
                        )?;
                        self.dump_bytes_to_disk(
                            dir,
                            &format!("{dump_prefix}_layer{layer_idx:02}_post_attn_activation.fp4"),
                            forward.activation_fp4.ptr(),
                            self.topology.hidden_size.div_ceil(2),
                        )?;
                        self.dump_bytes_to_disk(
                            dir,
                            &format!(
                                "{dump_prefix}_layer{layer_idx:02}_post_attn_activation_scale.e4m3"
                            ),
                            forward.activation_scale.ptr(),
                            self.topology.hidden_size.div_ceil(16).div_ceil(4) * 512,
                        )?;
                    }
                }
                let mlp_start = profile_decode.then(std::time::Instant::now);
                self.run_mlp_with_quantized_input(layer, forward, quantized)?;
                if let Some(mlp_start) = mlp_start {
                    qwen36_fp4_kernels::cuda_synchronize()?;
                    eprintln!(
                        "decode.profile layer{layer_idx:02}.mlp_ms={:.3}",
                        mlp_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
            } else {
                self.rmsnorm(
                    1,
                    self.topology.hidden_size,
                    forward.block_out.ptr(),
                    self.tensor_ptr(weights, &common.post_attention_layernorm)?,
                    forward.residual.ptr(),
                    forward.residual.ptr(),
                    forward.normed.ptr(),
                )?;
                if dump_decode {
                    if let Some(dir) = &dump_dir {
                        self.dump_buffer_to_disk(
                            dir,
                            &format!("{dump_prefix}_layer{layer_idx:02}_post_attn_normed.bf16"),
                            forward.normed.ptr(),
                            self.topology.hidden_size,
                        )?;
                    }
                }
                let mlp_start = profile_decode.then(std::time::Instant::now);
                self.run_mlp(layer, forward)?;
                if let Some(mlp_start) = mlp_start {
                    qwen36_fp4_kernels::cuda_synchronize()?;
                    eprintln!(
                        "decode.profile layer{layer_idx:02}.mlp_ms={:.3}",
                        mlp_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
            if dump_decode {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_mlp_gate.bf16"),
                        forward.aux.ptr(),
                        self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_mlp_up.bf16"),
                        forward.aux2.ptr(),
                        self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_mlp_swiglu.bf16"),
                        forward.aux3.ptr(),
                        self.topology.intermediate_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_residual_after_attn.bf16"),
                        forward.residual.ptr(),
                        self.topology.hidden_size,
                    )?;
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_layer{layer_idx:02}_mlp_out.bf16"),
                        forward.hidden.ptr(),
                        self.topology.hidden_size,
                    )?;
                }
            }
        }

        if emit_logits {
            let logits_start = profile_decode.then(std::time::Instant::now);
            self.rmsnorm(
                1,
                self.topology.hidden_size,
                forward.hidden.ptr(),
                self.tensor_ptr(weights, &manifest.final_norm)?,
                forward.residual.ptr(),
                DevicePtr::NULL,
                forward.normed.ptr(),
            )?;
            if dump_decode {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_final_normed.bf16"),
                        forward.normed.ptr(),
                        self.topology.hidden_size,
                    )?;
                }
            }
            self.bf16_matvec(
                &manifest.lm_head,
                forward.normed.ptr(),
                forward.logits.ptr(),
            )?;
            if let Some(logits_start) = logits_start {
                qwen36_fp4_kernels::cuda_synchronize()?;
                eprintln!(
                    "decode.profile final_logits_ms={:.3}",
                    logits_start.elapsed().as_secs_f64() * 1000.0
                );
            }
            if dump_decode {
                if let Some(dir) = &dump_dir {
                    self.dump_buffer_to_disk(
                        dir,
                        &format!("{dump_prefix}_logits.bf16"),
                        forward.logits.ptr(),
                        self.topology.vocab_size,
                    )?;
                }
            }
        }
        if sync_after {
            qwen36_fp4_kernels::cuda_synchronize()?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn run_linear_attention_layer(
        &self,
        layer: &LinearAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        prequantized_normed: Option<Nvfp4ActivationQuant<'_>>,
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

        let in_proj_linears = [
            (&layer.in_proj_qkv, forward.qkv.ptr()),
            (&layer.in_proj_b, forward.aux2.ptr()),
            (&layer.in_proj_a, forward.aux3.ptr()),
            (&layer.in_proj_z, forward.qkv.ptr()),
        ];
        let shared_in_proj = match prequantized_normed {
            Some(quantized) => Some(quantized),
            None => Self::common_nvfp4_quant(&in_proj_linears)?,
        };
        if let Some(quantized) = shared_in_proj {
            if prequantized_normed.is_none() {
                self.quantize_nvfp4_activation(forward.normed.ptr(), quantized)?;
            }
            for &(binding, output) in &in_proj_linears[..3] {
                self.linear_with_quantized_nvfp4(binding, output, quantized)?;
            }
        } else {
            self.linears_same_input(forward.normed.ptr(), &in_proj_linears[..3])?;
        }
        self.backend.conv1d_update(&Conv1dUpdateSpec {
            channels: qkv_dim,
            kernel_size: self.topology.linear_conv_kernel_dim,
            input_bf16: forward.qkv.ptr(),
            conv_history_bf16: conv_history,
            weight_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.conv1d_weight)?,
            output_bf16: forward.aux.ptr(),
        })?;
        self.backend.gdn_gate(&GdnGateSpec {
            rows: 1,
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
            q_token_stride: 0,
            k_token_stride: 0,
            v_token_stride: 0,
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
        if let Some(quantized) = shared_in_proj {
            self.linear_with_quantized_nvfp4(&layer.in_proj_z, forward.qkv.ptr(), quantized)?;
        } else {
            self.linear(&layer.in_proj_z, forward.normed.ptr(), forward.qkv.ptr())?;
        }
        self.rmsnorm_direct_weight(
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
        position_device_i32: DevicePtr,
        prequantized_normed: Option<Nvfp4ActivationQuant<'_>>,
    ) -> Result<()> {
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

        let qkv_linears = [
            (&layer.q_proj, forward.qkv.ptr()),
            (&layer.k_proj, forward.aux.ptr()),
            (&layer.v_proj, forward.aux2.ptr()),
        ];
        if let Some(quantized) = prequantized_normed {
            for &(binding, output) in &qkv_linears {
                self.linear_with_quantized_nvfp4(binding, output, quantized)?;
            }
        } else {
            self.linears_same_input(forward.normed.ptr(), &qkv_linears)?;
        }
        self.backend.q_proj_deinterleave(&QProjDeinterleaveSpec {
            rows: 1,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            input_bf16: forward.qkv.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.rmsnorm(
            self.topology.attention_num_heads,
            self.topology.attention_head_dim,
            forward.aux3.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.q_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.aux3.ptr(),
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
            position_i32: position as i32,
            use_scalar_position: true,
            positions_i32: DevicePtr::NULL,
            q_bf16: forward.aux3.ptr(),
            k_bf16: forward.aux.ptr(),
            scalar_position_device_i32: position_device_i32,
        })?;
        self.backend.attention_decode(&AttentionDecodeSpec {
            layer_index: layer.layer_index,
            position,
            q_bf16: forward.aux3.ptr(),
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
            position_device_i32,
        })?;
        self.backend.q_proj_sigmoid_gate(&QProjSigmoidGateSpec {
            rows: 1,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            gate_bf16: forward.qkv.ptr(),
            input_bf16: forward.aux3.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.linear(&layer.o_proj, forward.aux3.ptr(), forward.block_out.ptr())
    }

    #[cfg(feature = "cuda")]
    fn concat_mtp_fc_input_rows(
        &self,
        rows: usize,
        embedding_normed: DevicePtr,
        target_hidden_normed: DevicePtr,
        output_bf16: DevicePtr,
    ) -> Result<()> {
        let hidden = self.topology.hidden_size;
        self.backend.copy_strided_rows(&CopyStridedRowsSpec {
            rows,
            values: hidden,
            input_stride: hidden,
            output_stride: hidden * 2,
            input_bf16: embedding_normed,
            output_bf16,
        })?;
        self.backend.copy_strided_rows(&CopyStridedRowsSpec {
            rows,
            values: hidden,
            input_stride: hidden,
            output_stride: hidden * 2,
            input_bf16: target_hidden_normed,
            output_bf16: Self::ptr_offset(output_bf16, hidden * 2)?,
        })
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_full_attention_layer_decode(
        &self,
        layer: &FullAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        forward: &GpuForwardBuffers,
        position: usize,
        position_device_i32: DevicePtr,
    ) -> Result<()> {
        let (kv_cache_k, kv_cache_v) = self.mtp_cache_ptrs(runtime)?;
        let qkv_linears = [
            (&layer.q_proj, forward.qkv.ptr()),
            (&layer.k_proj, forward.aux.ptr()),
            (&layer.v_proj, forward.aux2.ptr()),
        ];
        self.linears_same_input(forward.normed.ptr(), &qkv_linears)?;
        self.backend.q_proj_deinterleave(&QProjDeinterleaveSpec {
            rows: 1,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            input_bf16: forward.qkv.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.rmsnorm(
            self.topology.attention_num_heads,
            self.topology.attention_head_dim,
            forward.aux3.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.q_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            forward.aux3.ptr(),
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
            position_i32: position as i32,
            use_scalar_position: true,
            positions_i32: DevicePtr::NULL,
            q_bf16: forward.aux3.ptr(),
            k_bf16: forward.aux.ptr(),
            scalar_position_device_i32: position_device_i32,
        })?;
        self.backend.attention_decode(&AttentionDecodeSpec {
            layer_index: layer.layer_index,
            position,
            q_bf16: forward.aux3.ptr(),
            k_bf16: forward.aux.ptr(),
            v_bf16: forward.aux2.ptr(),
            kv_cache_k,
            kv_cache_v,
            output_bf16: forward.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            position_device_i32,
        })?;
        self.backend.q_proj_sigmoid_gate(&QProjSigmoidGateSpec {
            rows: 1,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            gate_bf16: forward.qkv.ptr(),
            input_bf16: forward.aux3.ptr(),
            output_bf16: forward.aux3.ptr(),
        })?;
        self.linear(&layer.o_proj, forward.aux3.ptr(), forward.block_out.ptr())
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_full_attention_layer_prefill(
        &self,
        layer: &FullAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        prefill: &GpuPrefillBuffers,
        tokens: usize,
        start_position: usize,
    ) -> Result<()> {
        let (kv_cache_k, kv_cache_v) = self.mtp_cache_ptrs(runtime)?;
        let qkv_linears = [
            (&layer.q_proj, prefill.qkv.ptr()),
            (&layer.k_proj, prefill.aux.ptr()),
            (&layer.v_proj, prefill.aux2.ptr()),
        ];
        self.linears_same_input_rows(prefill.normed.ptr(), &qkv_linears, tokens, prefill)?;
        self.backend.q_proj_deinterleave(&QProjDeinterleaveSpec {
            rows: tokens,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            input_bf16: prefill.qkv.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.rmsnorm(
            tokens * self.topology.attention_num_heads,
            self.topology.attention_head_dim,
            prefill.aux3.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.q_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.aux3.ptr(),
        )?;
        self.rmsnorm(
            tokens * self.topology.attention_num_kv_heads,
            self.topology.attention_head_dim,
            prefill.aux.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.k_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.aux.ptr(),
        )?;
        self.backend.partial_rope(&PartialRopeSpec {
            tokens,
            q_heads: self.topology.attention_num_heads,
            kv_heads: self.topology.attention_num_kv_heads,
            head_dim: self.topology.attention_head_dim,
            rope_dims: self.topology.attention_rope_dims(),
            base_theta: self.topology.rope_theta,
            position_i32: 0,
            use_scalar_position: false,
            positions_i32: prefill.position_i32.ptr(),
            q_bf16: prefill.aux3.ptr(),
            k_bf16: prefill.aux.ptr(),
            scalar_position_device_i32: DevicePtr::NULL,
        })?;
        self.backend.attention_prefill(&AttentionPrefillSpec {
            layer_index: layer.layer_index,
            start_position,
            tokens,
            q_bf16: prefill.aux3.ptr(),
            k_bf16: prefill.aux.ptr(),
            v_bf16: prefill.aux2.ptr(),
            kv_cache_k,
            kv_cache_v,
            output_bf16: prefill.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
        })?;
        self.backend.q_proj_sigmoid_gate(&QProjSigmoidGateSpec {
            rows: tokens,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            gate_bf16: prefill.qkv.ptr(),
            input_bf16: prefill.aux3.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.linear_rows(
            &layer.o_proj,
            prefill.aux3.ptr(),
            prefill.block_out.ptr(),
            tokens,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_mlp_decode(
        &self,
        common: &CommonLayerWeights,
        forward: &GpuForwardBuffers,
    ) -> Result<()> {
        self.linears_same_input(
            forward.normed.ptr(),
            &[
                (&common.mlp_gate_proj, forward.aux.ptr()),
                (&common.mlp_up_proj, forward.aux2.ptr()),
            ],
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
    fn run_mtp_mlp_prefill(
        &self,
        common: &CommonLayerWeights,
        prefill: &GpuPrefillBuffers,
        tokens: usize,
    ) -> Result<()> {
        self.linears_same_input_rows(
            prefill.normed.ptr(),
            &[
                (&common.mlp_gate_proj, prefill.aux.ptr()),
                (&common.mlp_up_proj, prefill.aux2.ptr()),
            ],
            tokens,
            prefill,
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: tokens,
            intermediate: self.topology.intermediate_size,
            gate_bf16: prefill.aux.ptr(),
            up_bf16: prefill.aux2.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.linear_rows(
            &common.mlp_down_proj,
            prefill.aux3.ptr(),
            prefill.hidden.ptr(),
            tokens,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    fn mtp_cache_ptrs(&self, runtime: &GpuRuntimeBuffers) -> Result<(DevicePtr, DevicePtr)> {
        let cache = runtime
            .mtp_kv_cache
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("MTP KV cache was not allocated".to_owned()))?;
        let plane_bytes = self.mtp_kv_cache_plane_bytes()?;
        Ok((cache.ptr(), cache.ptr_at(plane_bytes)?))
    }

    #[cfg(feature = "cuda")]
    fn run_linear_attention_layer_prefill(
        &self,
        layer: &LinearAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        prefill: &GpuPrefillBuffers,
        tokens: usize,
        prequantized_normed: Option<Nvfp4ActivationQuant<'_>>,
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

        let in_proj_linears = [
            (&layer.in_proj_qkv, prefill.qkv.ptr()),
            (&layer.in_proj_b, prefill.aux2.ptr()),
            (&layer.in_proj_a, prefill.aux3.ptr()),
            (&layer.in_proj_z, prefill.qkv.ptr()),
        ];
        let shared_in_proj = match prequantized_normed {
            Some(quantized) => Some(quantized),
            None => Self::common_nvfp4_quant(&in_proj_linears)?,
        };
        if let Some(quantized) = shared_in_proj {
            if prequantized_normed.is_none() {
                self.quantize_nvfp4_activation_rows(
                    prefill.normed.ptr(),
                    tokens,
                    quantized,
                    prefill,
                )?;
            }
            for &(binding, output) in &in_proj_linears[..3] {
                self.linear_with_quantized_nvfp4_rows(binding, output, tokens, quantized, prefill)?;
            }
        } else {
            self.linears_same_input_rows(
                prefill.normed.ptr(),
                &in_proj_linears[..3],
                tokens,
                prefill,
            )?;
        }

        // Dump per-layer intermediates for the parity harness when requested.
        if layer.layer_index == 0 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer0_qkv_raw.bf16",
                    prefill.qkv.ptr(),
                    tokens * qkv_dim,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer0_b_raw.bf16",
                    prefill.aux2.ptr(),
                    tokens * self.topology.linear_num_value_heads,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer0_a_raw.bf16",
                    prefill.aux3.ptr(),
                    tokens * self.topology.linear_num_value_heads,
                )?;
            }
        }

        self.backend.conv1d_prefill(&Conv1dPrefillSpec {
            tokens,
            channels: qkv_dim,
            kernel_size: self.topology.linear_conv_kernel_dim,
            input_bf16: prefill.qkv.ptr(),
            conv_history_bf16: conv_history,
            weight_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.conv1d_weight)?,
            output_bf16: prefill.aux.ptr(),
        })?;
        if layer.layer_index == 0 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer0_conv_out.bf16",
                    prefill.aux.ptr(),
                    tokens * qkv_dim,
                )?;
            }
        }
        self.backend.gdn_gate(&GdnGateSpec {
            rows: tokens,
            heads: self.topology.linear_num_value_heads,
            a_bf16: prefill.aux3.ptr(),
            b_bf16: prefill.aux2.ptr(),
            a_log_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.a_log)?,
            dt_bias_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.dt_bias)?,
            gate_f32: prefill.gate_f32.ptr(),
            beta_f32: prefill.beta_f32.ptr(),
        })?;
        if layer.layer_index == 0 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                let v_heads = self.topology.linear_num_value_heads;
                self.dump_f32_to_disk(
                    &dir,
                    "layer0_gate.f32",
                    prefill.gate_f32.ptr(),
                    tokens * v_heads,
                )?;
                self.dump_f32_to_disk(
                    &dir,
                    "layer0_beta.f32",
                    prefill.beta_f32.ptr(),
                    tokens * v_heads,
                )?;
            }
        }
        self.backend.deltanet_decode(&DeltaNetDecodeSpec {
            layer_index: layer.layer_index,
            tokens_in_persistent_loop: tokens,
            q_token_stride: qkv_dim,
            k_token_stride: qkv_dim,
            v_token_stride: qkv_dim,
            q_bf16: prefill.aux.ptr(),
            k_bf16: Self::ptr_offset(prefill.aux.ptr(), key_dim * 2)?,
            v_bf16: Self::ptr_offset(prefill.aux.ptr(), key_dim * 4)?,
            state_bf16: state,
            conv_history_bf16: conv_history,
            output_bf16: prefill.aux3.ptr(),
            gate_f32: prefill.gate_f32.ptr(),
            beta_f32: prefill.beta_f32.ptr(),
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
        if layer.layer_index == 0 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer0_deltanet_out.bf16",
                    prefill.aux3.ptr(),
                    tokens * value_dim,
                )?;
            }
        }

        if let Some(quantized) = shared_in_proj {
            self.linear_with_quantized_nvfp4_rows(
                &layer.in_proj_z,
                prefill.qkv.ptr(),
                tokens,
                quantized,
                prefill,
            )?;
        } else {
            self.linear_rows(
                &layer.in_proj_z,
                prefill.normed.ptr(),
                prefill.qkv.ptr(),
                tokens,
                prefill,
            )?;
        }
        if layer.layer_index == 0 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer0_z.bf16",
                    prefill.qkv.ptr(),
                    tokens * value_dim,
                )?;
            }
        }
        self.rmsnorm_direct_weight(
            tokens * self.topology.linear_num_value_heads,
            self.topology.linear_value_head_dim,
            prefill.aux3.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.norm_weight)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.aux2.ptr(),
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: tokens,
            intermediate: value_dim,
            gate_bf16: prefill.qkv.ptr(),
            up_bf16: prefill.aux2.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.linear_rows(
            &layer.out_proj,
            prefill.aux3.ptr(),
            prefill.block_out.ptr(),
            tokens,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_full_attention_layer_prefill(
        &self,
        layer: &FullAttentionLayerWeights,
        runtime: &GpuRuntimeBuffers,
        prefill: &GpuPrefillBuffers,
        tokens: usize,
        start_position: usize,
        prequantized_normed: Option<Nvfp4ActivationQuant<'_>>,
    ) -> Result<()> {
        let q_dim = self.topology.full_attention_q_dim();
        let q_dim_with_gate = self.topology.full_attention_q_dim_with_gate();
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

        let qkv_linears = [
            (&layer.q_proj, prefill.qkv.ptr()),
            (&layer.k_proj, prefill.aux.ptr()),
            (&layer.v_proj, prefill.aux2.ptr()),
        ];
        if let Some(quantized) = prequantized_normed {
            for &(binding, output) in &qkv_linears {
                self.linear_with_quantized_nvfp4_rows(binding, output, tokens, quantized, prefill)?;
            }
        } else {
            self.linears_same_input_rows(prefill.normed.ptr(), &qkv_linears, tokens, prefill)?;
        }
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                let kv_dim = self.topology.full_attention_kv_dim();
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_q_proj_raw.bf16",
                    prefill.qkv.ptr(),
                    tokens * q_dim_with_gate,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_k_raw.bf16",
                    prefill.aux.ptr(),
                    tokens * kv_dim,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_v_raw.bf16",
                    prefill.aux2.ptr(),
                    tokens * kv_dim,
                )?;
            }
        }

        self.backend.q_proj_deinterleave(&QProjDeinterleaveSpec {
            rows: tokens,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            input_bf16: prefill.qkv.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_q_extracted.bf16",
                    prefill.aux3.ptr(),
                    tokens * q_dim,
                )?;
            }
        }
        self.rmsnorm(
            tokens * self.topology.attention_num_heads,
            self.topology.attention_head_dim,
            prefill.aux3.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.q_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.aux3.ptr(),
        )?;
        self.rmsnorm(
            tokens * self.topology.attention_num_kv_heads,
            self.topology.attention_head_dim,
            prefill.aux.ptr(),
            self.tensor_ptr(self.cuda_weights()?, &layer.k_norm)?,
            DevicePtr::NULL,
            DevicePtr::NULL,
            prefill.aux.ptr(),
        )?;
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                let kv_dim = self.topology.full_attention_kv_dim();
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_q_normed.bf16",
                    prefill.aux3.ptr(),
                    tokens * q_dim,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_k_normed.bf16",
                    prefill.aux.ptr(),
                    tokens * kv_dim,
                )?;
            }
        }
        self.backend.partial_rope(&PartialRopeSpec {
            tokens,
            q_heads: self.topology.attention_num_heads,
            kv_heads: self.topology.attention_num_kv_heads,
            head_dim: self.topology.attention_head_dim,
            rope_dims: self.topology.attention_rope_dims(),
            base_theta: self.topology.rope_theta,
            position_i32: 0,
            use_scalar_position: false,
            positions_i32: prefill.position_i32.ptr(),
            q_bf16: prefill.aux3.ptr(),
            k_bf16: prefill.aux.ptr(),
            scalar_position_device_i32: DevicePtr::NULL,
        })?;
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                let kv_dim = self.topology.full_attention_kv_dim();
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_q_rope.bf16",
                    prefill.aux3.ptr(),
                    tokens * q_dim,
                )?;
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_k_rope.bf16",
                    prefill.aux.ptr(),
                    tokens * kv_dim,
                )?;
            }
        }
        self.backend.attention_prefill(&AttentionPrefillSpec {
            layer_index: layer.layer_index,
            start_position,
            tokens,
            q_bf16: prefill.aux3.ptr(),
            k_bf16: prefill.aux.ptr(),
            v_bf16: prefill.aux2.ptr(),
            kv_cache_k: cache.ptr_at(layout.k_offset_bytes as usize)?,
            kv_cache_v: cache.ptr_at(layout.v_offset_bytes as usize)?,
            output_bf16: prefill.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
        })?;
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_attn_pre_gate.bf16",
                    prefill.aux3.ptr(),
                    tokens * q_dim,
                )?;
            }
        }
        self.backend.q_proj_sigmoid_gate(&QProjSigmoidGateSpec {
            rows: tokens,
            heads: self.topology.attention_num_heads,
            head_dim: self.topology.attention_head_dim,
            gate_bf16: prefill.qkv.ptr(),
            input_bf16: prefill.aux3.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_attn_gated.bf16",
                    prefill.aux3.ptr(),
                    tokens * q_dim,
                )?;
            }
        }
        self.linear_rows(
            &layer.o_proj,
            prefill.aux3.ptr(),
            prefill.block_out.ptr(),
            tokens,
            prefill,
        )?;
        if layer.layer_index == 3 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(
                    &dir,
                    "layer3_attn_out.bf16",
                    prefill.block_out.ptr(),
                    tokens * self.topology.hidden_size,
                )?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn run_mlp(&self, layer: &LayerWeights, forward: &GpuForwardBuffers) -> Result<()> {
        let common = match layer {
            LayerWeights::LinearAttention(layer) => &layer.common,
            LayerWeights::FullAttention(layer) => &layer.common,
        };
        self.linears_same_input(
            forward.normed.ptr(),
            &[
                (&common.mlp_gate_proj, forward.aux.ptr()),
                (&common.mlp_up_proj, forward.aux2.ptr()),
            ],
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
    fn run_mlp_with_quantized_input(
        &self,
        layer: &LayerWeights,
        forward: &GpuForwardBuffers,
        quantized: Nvfp4ActivationQuant<'_>,
    ) -> Result<()> {
        let common = layer_common(layer);
        self.linear_with_quantized_nvfp4(&common.mlp_gate_proj, forward.aux.ptr(), quantized)?;
        self.linear_with_quantized_nvfp4(&common.mlp_up_proj, forward.aux2.ptr(), quantized)?;
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
    fn run_mlp_prefill(
        &self,
        layer: &LayerWeights,
        prefill: &GpuPrefillBuffers,
        tokens: usize,
    ) -> Result<()> {
        let common = layer_common(layer);
        self.linears_same_input_rows(
            prefill.normed.ptr(),
            &[
                (&common.mlp_gate_proj, prefill.aux.ptr()),
                (&common.mlp_up_proj, prefill.aux2.ptr()),
            ],
            tokens,
            prefill,
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: tokens,
            intermediate: self.topology.intermediate_size,
            gate_bf16: prefill.aux.ptr(),
            up_bf16: prefill.aux2.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.linear_rows(
            &common.mlp_down_proj,
            prefill.aux3.ptr(),
            prefill.hidden.ptr(),
            tokens,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mlp_with_quantized_input_prefill(
        &self,
        layer: &LayerWeights,
        prefill: &GpuPrefillBuffers,
        tokens: usize,
        quantized: Nvfp4ActivationQuant<'_>,
    ) -> Result<()> {
        let common = layer_common(layer);
        self.linear_with_quantized_nvfp4_rows(
            &common.mlp_gate_proj,
            prefill.aux.ptr(),
            tokens,
            quantized,
            prefill,
        )?;
        self.linear_with_quantized_nvfp4_rows(
            &common.mlp_up_proj,
            prefill.aux2.ptr(),
            tokens,
            quantized,
            prefill,
        )?;
        self.backend.swiglu(&SwiGluSpec {
            rows: tokens,
            intermediate: self.topology.intermediate_size,
            gate_bf16: prefill.aux.ptr(),
            up_bf16: prefill.aux2.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.linear_rows(
            &common.mlp_down_proj,
            prefill.aux3.ptr(),
            prefill.hidden.ptr(),
            tokens,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    fn linears_same_input(
        &self,
        input: DevicePtr,
        linears: &[(&LinearWeightBinding, DevicePtr)],
    ) -> Result<()> {
        let Some(quantized) = Self::common_nvfp4_quant(linears)? else {
            for &(binding, output) in linears {
                self.linear(binding, input, output)?;
            }
            return Ok(());
        };

        self.quantize_nvfp4_activation(input, quantized)?;
        for &(binding, output) in linears {
            self.linear_with_quantized_nvfp4(binding, output, quantized)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn linears_same_input_rows(
        &self,
        input: DevicePtr,
        linears: &[(&LinearWeightBinding, DevicePtr)],
        rows: usize,
        prefill: &GpuPrefillBuffers,
    ) -> Result<()> {
        let Some(quantized) = Self::common_nvfp4_quant(linears)? else {
            for &(binding, output) in linears {
                self.linear_rows(binding, input, output, rows, prefill)?;
            }
            return Ok(());
        };

        self.quantize_nvfp4_activation_rows(input, rows, quantized, prefill)?;
        for &(binding, output) in linears {
            self.linear_with_quantized_nvfp4_rows(binding, output, rows, quantized, prefill)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn common_nvfp4_quant<'a>(
        linears: &'a [(&'a LinearWeightBinding, DevicePtr)],
    ) -> Result<Option<Nvfp4ActivationQuant<'a>>> {
        Self::common_nvfp4_quant_bindings(linears.iter().map(|(binding, _)| *binding))
    }

    #[cfg(feature = "cuda")]
    fn common_nvfp4_quant_bindings<'a>(
        linears: impl IntoIterator<Item = &'a LinearWeightBinding>,
    ) -> Result<Option<Nvfp4ActivationQuant<'a>>> {
        let mut common: Option<Nvfp4ActivationQuant<'a>> = None;
        for binding in linears {
            let LinearWeightBinding::Nvfp4 {
                weight,
                input_scale,
                ..
            } = binding
            else {
                return Ok(None);
            };
            let in_features = Self::nvfp4_in_features(weight)?;
            match common {
                Some(previous) if previous.in_features != in_features => return Ok(None),
                Some(_) => {}
                None => {
                    common = Some(Nvfp4ActivationQuant {
                        in_features,
                        input_scale,
                    });
                }
            }
        }
        Ok(common)
    }

    #[cfg(feature = "cuda")]
    fn layer_input_nvfp4_quant(layer: &LayerWeights) -> Result<Option<Nvfp4ActivationQuant<'_>>> {
        match layer {
            LayerWeights::LinearAttention(layer) => Self::common_nvfp4_quant_bindings([
                &layer.in_proj_qkv,
                &layer.in_proj_b,
                &layer.in_proj_a,
                &layer.in_proj_z,
            ]),
            LayerWeights::FullAttention(layer) => {
                Self::common_nvfp4_quant_bindings([&layer.q_proj, &layer.k_proj, &layer.v_proj])
            }
        }
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
                input_scale,
            } => {
                let in_features = Self::nvfp4_in_features(weight)?;
                let quantized = Nvfp4ActivationQuant {
                    in_features,
                    input_scale,
                };
                self.quantize_nvfp4_activation(input, quantized)?;
                self.nvfp4_gemm_with_quantized_activation(
                    weight,
                    block_scale,
                    tensor_scale,
                    input_scale,
                    output,
                    in_features,
                )
            }
            LinearWeightBinding::Bf16 { weight } => self.bf16_matvec(weight, input, output),
        }
    }

    #[cfg(feature = "cuda")]
    fn linear_rows(
        &self,
        binding: &LinearWeightBinding,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        prefill: &GpuPrefillBuffers,
    ) -> Result<()> {
        match binding {
            LinearWeightBinding::Nvfp4 {
                weight,
                block_scale,
                tensor_scale,
                input_scale,
            } => {
                let in_features = Self::nvfp4_in_features(weight)?;
                let quantized = Nvfp4ActivationQuant {
                    in_features,
                    input_scale,
                };
                self.quantize_nvfp4_activation_rows(input, rows, quantized, prefill)?;
                self.nvfp4_gemm_with_quantized_activation_rows(
                    weight,
                    block_scale,
                    tensor_scale,
                    input_scale,
                    output,
                    rows,
                    in_features,
                    prefill,
                )
            }
            LinearWeightBinding::Bf16 { weight } => {
                self.bf16_gemm_rows(weight, input, output, rows)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn linear_with_quantized_nvfp4(
        &self,
        binding: &LinearWeightBinding,
        output: DevicePtr,
        quantized: Nvfp4ActivationQuant<'_>,
    ) -> Result<()> {
        let LinearWeightBinding::Nvfp4 {
            weight,
            block_scale,
            tensor_scale,
            input_scale,
        } = binding
        else {
            return Err(CoreError::Runtime(
                "quantized NVFP4 path received a BF16 linear".to_owned(),
            ));
        };
        let in_features = Self::nvfp4_in_features(weight)?;
        if in_features != quantized.in_features {
            return Err(CoreError::Runtime(format!(
                "quantized activation has {} values but {} expects {in_features}",
                quantized.in_features, weight.name
            )));
        }
        self.validate_nvfp4_input_scale(weight, quantized.input_scale, input_scale)?;
        self.nvfp4_gemm_with_quantized_activation(
            weight,
            block_scale,
            tensor_scale,
            quantized.input_scale,
            output,
            in_features,
        )
    }

    #[cfg(feature = "cuda")]
    fn linear_with_quantized_nvfp4_rows(
        &self,
        binding: &LinearWeightBinding,
        output: DevicePtr,
        rows: usize,
        quantized: Nvfp4ActivationQuant<'_>,
        prefill: &GpuPrefillBuffers,
    ) -> Result<()> {
        let LinearWeightBinding::Nvfp4 {
            weight,
            block_scale,
            tensor_scale,
            input_scale,
        } = binding
        else {
            return Err(CoreError::Runtime(
                "quantized NVFP4 path received a BF16 linear".to_owned(),
            ));
        };
        let in_features = Self::nvfp4_in_features(weight)?;
        if in_features != quantized.in_features {
            return Err(CoreError::Runtime(format!(
                "quantized activation has {} values but {} expects {in_features}",
                quantized.in_features, weight.name
            )));
        }
        self.validate_nvfp4_input_scale(weight, quantized.input_scale, input_scale)?;
        self.nvfp4_gemm_with_quantized_activation_rows(
            weight,
            block_scale,
            tensor_scale,
            quantized.input_scale,
            output,
            rows,
            in_features,
            prefill,
        )
    }

    #[cfg(feature = "cuda")]
    fn quantize_nvfp4_activation(
        &self,
        input: DevicePtr,
        quantized: Nvfp4ActivationQuant<'_>,
    ) -> Result<()> {
        let forward = self.cuda_forward()?;
        let input_tensor_scale_f32 =
            self.tensor_scalar_f32(self.cuda_weights()?, quantized.input_scale)?;
        self.backend.nvfp4_quantize_bf16(&Nvfp4QuantizeSpec {
            values: quantized.in_features,
            input_bf16: input,
            output_fp4: forward.activation_fp4.ptr(),
            output_scale_e4m3: forward.activation_scale.ptr(),
            output_tensor_scale_f32: forward.activation_scale_2.ptr(),
            input_tensor_scale_f32,
        })
    }

    #[cfg(feature = "cuda")]
    fn quantize_nvfp4_activation_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        quantized: Nvfp4ActivationQuant<'_>,
        prefill: &GpuPrefillBuffers,
    ) -> Result<()> {
        let input_tensor_scale_f32 =
            self.tensor_scalar_f32(self.cuda_weights()?, quantized.input_scale)?;
        self.backend.nvfp4_quantize_rows(&Nvfp4QuantizeRowsSpec {
            rows,
            values: quantized.in_features,
            input_bf16: input,
            output_fp4: prefill.activation_fp4.ptr(),
            output_scale_e4m3: prefill.activation_scale.ptr(),
            output_tensor_scale_f32: prefill.activation_scale_2.ptr(),
            input_tensor_scale_f32,
        })
    }

    #[cfg(feature = "cuda")]
    fn nvfp4_gemm_with_quantized_activation(
        &self,
        weight: &TensorInfo,
        block_scale: &TensorInfo,
        tensor_scale: &TensorInfo,
        input_scale: &TensorInfo,
        output: DevicePtr,
        in_features: usize,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let forward = self.cuda_forward()?;
        let runtime = self.cuda_runtime()?;
        let out_features = *weight
            .shape
            .first()
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} has empty shape", weight.name)))?;
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
            alpha: self.tensor_scalar_f32(weights, tensor_scale)?
                * self.tensor_scalar_f32(weights, input_scale)?,
            scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
        };
        self.backend.nvfp4_gemm(&gemm_spec).map_err(|err| {
            CoreError::Runtime(format!(
                "NVFP4 cuBLASLt GEMM failed for {} (m={}, n={}, k={}): {err}",
                weight.name, gemm_spec.m, gemm_spec.n, gemm_spec.k
            ))
        })
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn nvfp4_gemm_with_quantized_activation_rows(
        &self,
        weight: &TensorInfo,
        block_scale: &TensorInfo,
        tensor_scale: &TensorInfo,
        input_scale: &TensorInfo,
        output: DevicePtr,
        rows: usize,
        in_features: usize,
        prefill: &GpuPrefillBuffers,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let out_features = *weight
            .shape
            .first()
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} has empty shape", weight.name)))?;
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
            n: rows,
            k: in_features,
            a_fp4: self.tensor_ptr(weights, weight)?,
            a_scale: self.tensor_ptr(weights, block_scale)?,
            a_scale_2: self.tensor_ptr(weights, tensor_scale)?,
            b_fp4: prefill.activation_fp4.ptr(),
            b_scale: prefill.activation_scale.ptr(),
            b_scale_2: prefill.activation_scale_2.ptr(),
            c_bf16: output,
            workspace,
            workspace_bytes,
            alpha: self.tensor_scalar_f32(weights, tensor_scale)?
                * self.tensor_scalar_f32(weights, input_scale)?,
            scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
        };
        self.backend.nvfp4_gemm(&gemm_spec).map_err(|err| {
            CoreError::Runtime(format!(
                "NVFP4 cuBLASLt GEMM failed for {} (m={}, n={}, k={}): {err}",
                weight.name, gemm_spec.m, gemm_spec.n, gemm_spec.k
            ))
        })
    }

    #[cfg(feature = "cuda")]
    fn nvfp4_in_features(weight: &TensorInfo) -> Result<usize> {
        let packed_in = *weight
            .shape
            .get(1)
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} is not a matrix", weight.name)))?;
        Ok(packed_in * 2)
    }

    #[cfg(feature = "cuda")]
    fn validate_nvfp4_input_scale(
        &self,
        weight: &TensorInfo,
        quantized_input_scale: &TensorInfo,
        binding_input_scale: &TensorInfo,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let quantized_value = self.tensor_scalar_f32(weights, quantized_input_scale)?;
        let binding_value = self.tensor_scalar_f32(weights, binding_input_scale)?;
        if quantized_value.to_bits() != binding_value.to_bits() {
            return Err(CoreError::Runtime(format!(
                "shared NVFP4 activation for {} used input scale {} ({quantized_value}) but binding expects {} ({binding_value})",
                weight.name, quantized_input_scale.name, binding_input_scale.name
            )));
        }
        Ok(())
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
        let runtime = self.cuda_runtime()?;
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
        let input_bf16 = input;
        let weight_bf16 = self.tensor_ptr(self.cuda_weights()?, weight)?;
        let gemm_result = self.backend.bf16_gemm(&Bf16GemmSpec {
            m: out_features,
            n: 1,
            k: in_features,
            a_bf16: weight_bf16,
            b_bf16: input_bf16,
            c_bf16: output,
            workspace,
            workspace_bytes,
        });
        if gemm_result.is_ok() {
            return gemm_result;
        }

        self.backend.bf16_matvec(&Bf16MatVecSpec {
            out_features,
            in_features,
            input_bf16,
            weight_bf16,
            output_bf16: output,
        })
    }

    #[cfg(feature = "cuda")]
    fn bf16_gemm_rows(
        &self,
        weight: &TensorInfo,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        let out_features = *weight
            .shape
            .first()
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} has empty shape", weight.name)))?;
        let in_features = *weight
            .shape
            .get(1)
            .ok_or_else(|| CoreError::Runtime(format!("tensor {} is not a matrix", weight.name)))?;
        let runtime = self.cuda_runtime()?;
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
        let weight_bf16 = self.tensor_ptr(self.cuda_weights()?, weight)?;
        let gemm_result = self.backend.bf16_gemm(&Bf16GemmSpec {
            m: out_features,
            n: rows,
            k: in_features,
            a_bf16: weight_bf16,
            b_bf16: input,
            c_bf16: output,
            workspace,
            workspace_bytes,
        });
        if gemm_result.is_ok() || rows > 1 {
            return gemm_result;
        }
        self.backend.bf16_matvec(&Bf16MatVecSpec {
            out_features,
            in_features,
            input_bf16: input,
            weight_bf16,
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
            direct_weight: false,
        })
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_direct_weight(
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
            direct_weight: true,
        })
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_nvfp4_quantize(
        &self,
        hidden: usize,
        input: DevicePtr,
        weight: DevicePtr,
        residual: DevicePtr,
        residual_out: DevicePtr,
        output_bf16: DevicePtr,
        input_scale: &TensorInfo,
    ) -> Result<()> {
        let forward = self.cuda_forward()?;
        let input_tensor_scale_f32 = self.tensor_scalar_f32(self.cuda_weights()?, input_scale)?;
        self.backend
            .rmsnorm_nvfp4_quantize(&RmsNormNvfp4QuantizeSpec {
                hidden,
                eps: 1.0e-6,
                input_bf16: input,
                weight_bf16: weight,
                residual_bf16: residual,
                residual_out_bf16: residual_out,
                output_bf16,
                output_fp4: forward.activation_fp4.ptr(),
                output_scale_e4m3: forward.activation_scale.ptr(),
                output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                input_tensor_scale_f32,
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
    fn ptr_offset(ptr: DevicePtr, offset_bytes: usize) -> Result<DevicePtr> {
        ptr.offset_bytes(offset_bytes)
            .ok_or_else(|| CoreError::Runtime("CUDA pointer offset overflow".to_owned()))
    }

    #[cfg(feature = "cuda")]
    fn mtp_kv_cache_plane_bytes(&self) -> Result<usize> {
        let values = self
            .config
            .max_context
            .checked_mul(self.topology.attention_num_kv_heads)
            .and_then(|value| value.checked_mul(self.topology.attention_head_dim))
            .ok_or_else(|| CoreError::Runtime("MTP KV cache size overflow".to_owned()))?;
        values
            .checked_mul(2)
            .ok_or_else(|| CoreError::Runtime("MTP KV cache byte size overflow".to_owned()))
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

    /// Dump `count` FP32 values to disk for the parity harness.
    #[cfg(feature = "cuda")]
    fn dump_f32_to_disk(&self, dir: &str, name: &str, src: DevicePtr, count: usize) -> Result<()> {
        unsafe extern "C" {
            fn qwen36_cuda_memcpy_d2h(dst: *mut core::ffi::c_void, src: u64, bytes: usize) -> i32;
        }
        qwen36_fp4_kernels::cuda_synchronize()?;
        let bytes = count * 4;
        let mut buf = vec![0u8; bytes];
        let status = unsafe { qwen36_cuda_memcpy_d2h(buf.as_mut_ptr() as *mut _, src.0, bytes) };
        if status != 0 {
            return Err(CoreError::Runtime(format!(
                "dump_f32_to_disk memcpy failed: status {status}"
            )));
        }
        let path = std::path::Path::new(dir).join(name);
        std::fs::write(&path, &buf).map_err(|e| {
            CoreError::Runtime(format!("dump_f32_to_disk write {}: {e}", path.display()))
        })?;
        Ok(())
    }

    /// Dump raw bytes from a device buffer. Used for packed FP4 activations
    /// and UE4M3 scale bytes while debugging fused quantization.
    #[cfg(feature = "cuda")]
    fn dump_bytes_to_disk(
        &self,
        dir: &str,
        name: &str,
        src: DevicePtr,
        bytes: usize,
    ) -> Result<()> {
        unsafe extern "C" {
            fn qwen36_cuda_memcpy_d2h(dst: *mut core::ffi::c_void, src: u64, bytes: usize) -> i32;
        }
        qwen36_fp4_kernels::cuda_synchronize()?;
        let mut buf = vec![0u8; bytes];
        let status = unsafe { qwen36_cuda_memcpy_d2h(buf.as_mut_ptr() as *mut _, src.0, bytes) };
        if status != 0 {
            return Err(CoreError::Runtime(format!(
                "dump_bytes_to_disk memcpy failed: status {status}"
            )));
        }
        let path = std::path::Path::new(dir).join(name);
        std::fs::write(&path, &buf).map_err(|e| {
            CoreError::Runtime(format!("dump_bytes_to_disk write {}: {e}", path.display()))
        })?;
        Ok(())
    }

    /// Dump `count` BF16 values from a device buffer to disk as raw little-
    /// endian bytes. Used by the parity harness to compare intermediate
    /// activations against a Python ground-truth.
    #[cfg(feature = "cuda")]
    fn dump_buffer_to_disk(
        &self,
        dir: &str,
        name: &str,
        src: DevicePtr,
        count: usize,
    ) -> Result<()> {
        unsafe extern "C" {
            fn qwen36_cuda_memcpy_d2h(dst: *mut core::ffi::c_void, src: u64, bytes: usize) -> i32;
        }
        qwen36_fp4_kernels::cuda_synchronize()?;
        let bytes = count * 2;
        let mut buf = vec![0u8; bytes];
        let status = unsafe { qwen36_cuda_memcpy_d2h(buf.as_mut_ptr() as *mut _, src.0, bytes) };
        if status != 0 {
            return Err(CoreError::Runtime(format!(
                "dump_buffer_to_disk memcpy failed: status {status}"
            )));
        }
        let path = std::path::Path::new(dir).join(name);
        std::fs::write(&path, &buf).map_err(|e| {
            CoreError::Runtime(format!("dump_buffer_to_disk write {}: {e}", path.display()))
        })?;
        Ok(())
    }

    /// Numerical-parity helper: copy `count` BF16 values from `src` to host
    /// and print min / max / mean-abs / NaN+Inf counts. Guarded by the
    /// QWEN36_DEBUG_LAYER_TRACE env var so the cost only appears when the
    /// caller asks for it.
    #[cfg(feature = "cuda")]
    fn trace_buffer_stats(&self, label: &str, src: DevicePtr, count: usize) -> Result<()> {
        unsafe extern "C" {
            fn qwen36_cuda_memcpy_d2h(dst: *mut core::ffi::c_void, src: u64, bytes: usize) -> i32;
        }
        qwen36_fp4_kernels::cuda_synchronize()?;
        let bytes = count * 2;
        let mut buf = vec![0u8; bytes];
        let status = unsafe { qwen36_cuda_memcpy_d2h(buf.as_mut_ptr() as *mut _, src.0, bytes) };
        if status != 0 {
            return Err(CoreError::Runtime(format!(
                "trace_buffer_stats memcpy failed: status {status}"
            )));
        }
        let mut max = f32::NEG_INFINITY;
        let mut min = f32::INFINITY;
        let mut sum_abs: f64 = 0.0;
        let mut nans = 0usize;
        let mut infs = 0usize;
        for chunk in buf.chunks_exact(2) {
            let bits: u32 = (u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16;
            let v = f32::from_bits(bits);
            if v.is_nan() {
                nans += 1;
            } else if v.is_infinite() {
                infs += 1;
            } else {
                if v > max {
                    max = v;
                }
                if v < min {
                    min = v;
                }
                sum_abs += v.abs() as f64;
            }
        }
        eprintln!(
            "trace[{label}] n={count} mean_abs={:.6} min={min:.4} max={max:.4} nans={nans} infs={infs}",
            sum_abs / count.max(1) as f64,
        );
        Ok(())
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

    #[cfg(feature = "cuda")]
    fn cuda_prefill(&self) -> Result<&GpuPrefillBuffers> {
        self.gpu_prefill
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("CUDA prefill buffers are not allocated".to_owned()))
    }
}

#[cfg(feature = "cuda")]
fn layer_common(layer: &LayerWeights) -> &CommonLayerWeights {
    match layer {
        LayerWeights::LinearAttention(layer) => &layer.common,
        LayerWeights::FullAttention(layer) => &layer.common,
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
impl Engine<CudaBackend> {
    pub fn cuda_with_mapped_weights(model: &MappedModel, config: EngineConfig) -> Result<Self> {
        if config.mtp_speculative_tokens > 1 {
            return Err(CoreError::Runtime(
                "MTP runtime currently supports a 1-token validation path; pass --mtp-speculative-tokens 0 or 1"
                    .to_owned(),
            ));
        }
        let manifest = ModelWeightsManifest::from_layout(&model.layout)?;
        let include_mtp = config.mtp_speculative_tokens > 0;
        if include_mtp && manifest.mtp.is_none() {
            return Err(CoreError::Runtime(
                "MTP speculative decoding requested but no structured MTP weights were found"
                    .to_owned(),
            ));
        }
        let gpu_weights = GpuWeightStore::upload_required(model, &manifest, include_mtp)?;
        let mut engine = Self::new(model.layout.topology.clone(), config, CudaBackend);
        let mtp_kv_cache_bytes = if include_mtp {
            engine
                .mtp_kv_cache_plane_bytes()?
                .checked_mul(2)
                .ok_or_else(|| CoreError::Runtime("MTP KV cache size overflow".to_owned()))?
                as u64
        } else {
            0
        };
        let gpu_buffers =
            GpuRuntimeBuffers::allocate(&engine.state, 256 * 1024 * 1024, mtp_kv_cache_bytes)?;
        let gpu_forward = GpuForwardBuffers::allocate(&engine.topology)?;
        let prefill_capacity = engine.config.max_context.min(512).max(1);
        let gpu_prefill = GpuPrefillBuffers::allocate(&engine.topology, prefill_capacity)?;
        engine.weights = Some(manifest);
        engine.gpu_weights = Some(gpu_weights);
        engine.gpu_buffers = Some(gpu_buffers);
        engine.gpu_forward = Some(gpu_forward);
        engine.gpu_prefill = Some(gpu_prefill);
        Ok(engine)
    }

    pub fn gpu_weight_summary(&self) -> Option<(usize, u64)> {
        self.gpu_weights
            .as_ref()
            .map(|weights| (weights.tensor_count(), weights.total_bytes()))
    }

    pub fn gpu_buffer_bytes(&self) -> Option<u64> {
        Some(
            self.gpu_buffers.as_ref()?.total_bytes()
                + self.gpu_forward.as_ref()?.total_bytes()
                + self.gpu_prefill.as_ref()?.total_bytes(),
        )
    }
}
