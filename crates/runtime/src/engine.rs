use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
use qwen36_fp4_core::TensorInfo;
use qwen36_fp4_core::{CoreError, KvCacheDtype, ModelLayout, ModelTopology, Result};
#[cfg(feature = "cuda")]
use qwen36_fp4_kernels::{
    AttentionDecodeSpec, AttentionPrefillSpec, AttentionShape, Bf16GemmSpec, Bf16MatVecSpec,
    Conv1dGdnGateFusedSpec, Conv1dPrefillSpec, CopyStridedRowsSpec, CublasLtFp4ScaleMode,
    CudaBackend, DeltaNetDecodeSpec, DeltaNetShape, DevicePtr, EmbeddingLookupSpec, GdnGateSpec,
    Nvfp4GemmSpec, Nvfp4QuantizeRowsSpec, Nvfp4QuantizeSpec, PartialRopeSpec,
    QProjDeinterleaveSpec, QProjSigmoidGateSpec, RmsNormNvfp4QuantizeSpec, RmsNormSpec,
    SamplingRowsSpec, SamplingSpec, SwiGluNvfp4QuantizeSpec, SwiGluSpec,
};
use qwen36_fp4_kernels::{KernelBackend, NoCudaBackend};
#[cfg(feature = "cuda")]
use qwen36_fp4_loader::MappedModel;

use crate::cuda_graph::CudaGraphPlan;
#[cfg(feature = "cuda")]
use crate::gpu::{
    ATTN_MIN_SPLIT_TIMESTEPS_PER_BLOCK, GpuForwardBuffers, GpuPrefillBuffers, GpuRuntimeBuffers,
    GpuWeightStore, LinearAttnInProjFused, LinearAttnInProjFusedStore, MlpFusedLayer,
    MlpFusedStore, MtpKvSnapshotLayout,
};
use crate::kv_cache::KvCachePlan;
use crate::state::{DeltaNetStatePlan, RuntimeState};
use crate::weights::ModelWeightsManifest;
#[cfg(feature = "cuda")]
use crate::weights::{
    CommonLayerWeights, FullAttentionLayerWeights, LayerWeights, LinearAttentionLayerWeights,
    LinearWeightBinding,
};

#[cfg(feature = "cuda")]
const MTP_MAX_DRAFT_TOKENS: usize = 4;
#[cfg(feature = "cuda")]
const MTP_GRAPH_BUNDLE_U32S: usize = 16;
#[cfg(feature = "cuda")]
const MTP_GRAPH_VERIFIED_BASE: usize = 5;
#[cfg(feature = "cuda")]
const MTP_GRAPH_NEXT_DRAFT_BASE: usize = 9;

#[cfg(feature = "cuda")]
fn cuda_env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

#[cfg(feature = "cuda")]
fn cuda_env_bool(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

#[cfg(feature = "cuda")]
fn mtp_recurrent_snapshot_enabled() -> bool {
    std::env::var("QWEN36_MTP_SNAPSHOT_RECURRENT")
        .ok()
        .is_none_or(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
}

#[cfg(feature = "cuda")]
fn mtp_assume_accept_enabled() -> bool {
    cuda_env_bool("QWEN36_MTP_ASSUME_ACCEPT")
}

#[cfg(feature = "cuda")]
fn mtp_batched_lm_head_enabled() -> bool {
    !cuda_env_bool("QWEN36_MTP_BATCH_LM_HEAD_DISABLE")
}

#[cfg(feature = "cuda")]
fn cuda_env_workspace_bytes() -> usize {
    cuda_env_usize("QWEN36_CUDA_WORKSPACE_BYTES")
        .or_else(|| {
            cuda_env_usize("QWEN36_CUDA_WORKSPACE_MIB").and_then(|mib| mib.checked_mul(1024 * 1024))
        })
        .unwrap_or(256 * 1024 * 1024)
}

#[cfg(feature = "cuda")]
fn cuda_prefill_capacity(max_context: usize) -> usize {
    cuda_env_usize("QWEN36_PREFILL_CAPACITY")
        .unwrap_or_else(|| max_context.min(512))
        .clamp(1, max_context.max(1))
}

#[cfg(feature = "cuda")]
fn cuda_long_context_mode_enabled() -> bool {
    cuda_env_bool("QWEN36_LONG_CONTEXT_MODE")
}

#[cfg(feature = "cuda")]
fn cuda_mlp_fused_enabled() -> bool {
    !cuda_long_context_mode_enabled() && !cuda_env_bool("QWEN36_DISABLE_MLP_FUSED")
}

#[cfg(feature = "cuda")]
fn cuda_linear_attn_fused_enabled() -> bool {
    !cuda_long_context_mode_enabled() && !cuda_env_bool("QWEN36_DISABLE_LINEAR_ATTN_FUSED")
}

#[cfg(feature = "cuda")]
fn cuda_prefill_fused_mlp_enabled() -> bool {
    cuda_mlp_fused_enabled() && cuda_env_bool("QWEN36_PREFILL_FUSED_MLP")
}

#[cfg(feature = "cuda")]
fn cuda_prefill_fused_linear_attn_enabled() -> bool {
    cuda_linear_attn_fused_enabled()
        && cuda_env_bool("QWEN36_PREFILL_FUSED_LINEAR_ATTN")
        && !cuda_env_bool("QWEN36_PREFILL_FUSED_LINEAR_ATTN_DISABLE")
}

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryItem {
    pub name: String,
    pub bytes: u64,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryGroup {
    pub total_bytes: u64,
    pub items: Vec<GpuMemoryItem>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryReport {
    pub total_reported_bytes: u64,
    pub weights: GpuMemoryGroup,
    pub runtime: GpuMemoryGroup,
    pub forward: GpuMemoryGroup,
    pub prefill: GpuMemoryGroup,
    pub fused: GpuMemoryGroup,
    pub max_context: usize,
    pub prefill_capacity: usize,
    pub kv_cache_dtype: KvCacheDtype,
    pub mtp_speculative_tokens: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MtpVerifyResult {
    pub accepted: bool,
    pub verified_token: u32,
    pub next_token: Option<u32>,
    pub next_draft_token: Option<u32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpMultiVerifyResult {
    pub accepted_drafts: usize,
    pub rejected: bool,
    pub next_token: Option<u32>,
    pub next_draft_tokens: Vec<u32>,
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
    /// Pre-concatenated gate_proj + up_proj NVFP4 weights for the decode MLP
    /// fast path (one cuBLASLt FP4 GEMM instead of two). Only valid when
    /// every layer's gate/up share `tensor_scale` and `input_scale`, which
    /// holds for the shipped Qwen3.6 NVFP4 checkpoint.
    #[cfg(feature = "cuda")]
    pub mlp_fused: Option<MlpFusedStore>,
    /// Pre-concatenated DeltaNet in_proj_qkv/_b/_a/_z NVFP4 weights for the
    /// decode linear-attention fast path (one combined cuBLASLt FP4 GEMM
    /// instead of four). Indexed by global layer index; `None` for full-attn
    /// layers.
    #[cfg(feature = "cuda")]
    pub linear_attn_in_proj_fused: Option<LinearAttnInProjFusedStore>,
    /// Capture-mode CUDA stream + instantiated decode-and-sample graph. When
    /// `Some`, decode kernels read the current position from `forward.position_i32`
    /// instead of a host scalar so the same graph can replay across iterations.
    #[cfg(feature = "cuda")]
    decode_graph: Option<DecodeGraphState>,
    backend: B,
}

#[cfg(feature = "cuda")]
struct DecodeGraphState {
    kind: DecodeGraphKind,
    stream: qwen36_fp4_kernels::graph::OwnedCudaStream,
    exec: qwen36_fp4_kernels::graph::CudaGraphExec,
    raw_graph: qwen36_fp4_kernels::graph::CudaGraph,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeGraphKind {
    Decode,
    MtpDecodeOne,
    MtpVerifyOne,
    MtpVerifyMulti {
        drafts: usize,
        assume_accept: bool,
        batched_lm_head: bool,
    },
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
            mlp_fused: None,
            #[cfg(feature = "cuda")]
            linear_attn_in_proj_fused: None,
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
        self.synchronize_active_stream_for_host_read()?;
        let mut token = [0_u8; 4];
        self.cuda_forward()?
            .sampled_token_u32
            .copy_to_host(&mut token)?;
        Ok(u32::from_ne_bytes(token))
    }

    #[cfg(feature = "cuda")]
    pub fn read_current_token(&self) -> Result<u32> {
        self.synchronize_active_stream_for_host_read()?;
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
        self.queue_sample_greedy_into_with_mirror(output_token_u32, DevicePtr::NULL)
    }

    #[cfg(feature = "cuda")]
    fn queue_sample_greedy_into_with_mirror(
        &self,
        output_token_u32: DevicePtr,
        mirror_output_token_u32: DevicePtr,
    ) -> Result<()> {
        self.backend.sample(&SamplingSpec {
            vocab_size: self.topology.vocab_size,
            logits_bf16: self.cuda_forward()?.logits.ptr(),
            output_token_u32,
            mirror_output_token_u32,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        })
    }

    #[cfg(feature = "cuda")]
    fn queue_sample_greedy_rows_into(
        &self,
        logits_bf16: DevicePtr,
        rows: usize,
        output_token_u32: DevicePtr,
        mirror_last_output_token_u32: DevicePtr,
    ) -> Result<()> {
        self.backend.sample_rows(&SamplingRowsSpec {
            rows,
            vocab_size: self.topology.vocab_size,
            logits_bf16,
            output_token_u32,
            mirror_last_output_token_u32,
            temperature: 1.0,
        })
    }

    #[cfg(feature = "cuda")]
    fn synchronize_active_stream_for_host_read(&self) -> Result<()> {
        let stream = qwen36_fp4_kernels::graph::get_active_stream();
        if stream.is_null() {
            Ok(())
        } else {
            stream.synchronize()
        }
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

        if self
            .decode_graph
            .as_ref()
            .is_some_and(|graph| graph.kind == DecodeGraphKind::Decode)
        {
            return Ok(());
        }
        if self.decode_graph.is_some() {
            self.disable_decode_graph()?;
        }
        Self::ensure_graph_capture_allowed()?;
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
            kind: DecodeGraphKind::Decode,
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

    #[cfg(feature = "cuda")]
    fn ensure_mtp_verify_graph_two_tokens(&mut self, start_position: usize) -> Result<()> {
        use qwen36_fp4_kernels::graph::{self, CudaStream};

        if self
            .decode_graph
            .as_ref()
            .is_some_and(|graph| graph.kind == DecodeGraphKind::MtpVerifyOne)
        {
            return Ok(());
        }
        if self.decode_graph.is_some() {
            self.disable_decode_graph()?;
        }
        Self::ensure_graph_capture_allowed()?;
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP verify graph requested with MTP disabled".to_owned(),
            ));
        }
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda(
                "ensure_mtp_verify_graph_two_tokens",
            ));
        }

        let stream = CudaStream::create()?;
        let stream_handle = stream.handle();
        let start_position_device_i32 = self.cuda_prefill()?.position_i32.ptr();
        graph::set_active_stream(stream_handle);

        let capture_result = (|| -> Result<(graph::CudaGraph, graph::CudaGraphExec)> {
            graph::begin_capture(stream_handle)?;
            self.prefill_cuda_chunk(2, start_position, start_position_device_i32, false)?;
            self.final_norm_prefill_rows(2)?;
            self.prefill_row_logits(0)?;
            let verified_token_ptr = self.cuda_forward()?.mtp_verify_token_u32.ptr_at(8)?;
            self.queue_sample_greedy_into_with_mirror(
                self.cuda_forward()?.token_u32.ptr(),
                verified_token_ptr,
            )?;
            self.prefill_row_logits(1)?;
            let next_token_ptr = self.cuda_forward()?.mtp_verify_token_u32.ptr_at(4)?;
            self.queue_sample_greedy_into(next_token_ptr)?;
            self.run_mtp_prefill_chunk_with_tokens(
                2,
                start_position,
                start_position_device_i32,
                self.cuda_prefill()?.normed.ptr(),
                self.cuda_forward()?.mtp_verify_token_u32.ptr(),
                true,
            )?;
            let next_draft_ptr = self.cuda_forward()?.mtp_verify_token_u32.ptr_at(12)?;
            self.queue_sample_greedy_into_with_mirror(
                self.cuda_forward()?.sampled_token_u32.ptr(),
                next_draft_ptr,
            )?;
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

        self.decode_graph = Some(DecodeGraphState {
            kind: DecodeGraphKind::MtpVerifyOne,
            stream,
            exec,
            raw_graph,
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn launch_mtp_verify_graph_two_tokens(&self) -> Result<()> {
        let graph_state = self.decode_graph.as_ref().ok_or_else(|| {
            CoreError::Runtime("MTP verify graph launch requested before capture".to_owned())
        })?;
        if graph_state.kind != DecodeGraphKind::MtpVerifyOne {
            return Err(CoreError::Runtime(
                "MTP=1 verify graph launch found a different active graph".to_owned(),
            ));
        }
        qwen36_fp4_kernels::graph::launch(graph_state.exec, graph_state.stream.handle())?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn ensure_mtp_verify_graph_multi_tokens(
        &mut self,
        draft_count: usize,
        start_position: usize,
    ) -> Result<()> {
        use qwen36_fp4_kernels::graph::{self, CudaStream};

        if !(2..=MTP_MAX_DRAFT_TOKENS).contains(&draft_count) {
            return Err(CoreError::Runtime(format!(
                "MTP multi verify graph expects 2..={MTP_MAX_DRAFT_TOKENS} drafts, got {draft_count}"
            )));
        }
        let assume_accept = mtp_assume_accept_enabled();
        let batched_lm_head = mtp_batched_lm_head_enabled();
        if self.decode_graph.as_ref().is_some_and(|graph| {
            graph.kind
                == (DecodeGraphKind::MtpVerifyMulti {
                    drafts: draft_count,
                    assume_accept,
                    batched_lm_head,
                })
        }) {
            return Ok(());
        }
        if self.decode_graph.is_some() {
            self.disable_decode_graph()?;
        }
        Self::ensure_graph_capture_allowed()?;
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP multi verify graph requested with MTP disabled".to_owned(),
            ));
        }
        if self.backend.name() == "no-cuda" {
            return Err(CoreError::UnsupportedNoCuda(
                "ensure_mtp_verify_graph_multi_tokens",
            ));
        }

        let stream = CudaStream::create()?;
        let stream_handle = stream.handle();
        let verify_tokens = draft_count + 1;
        let start_position_device_i32 = self.cuda_prefill()?.position_i32.ptr();
        graph::set_active_stream(stream_handle);

        let capture_result = (|| -> Result<(graph::CudaGraph, graph::CudaGraphExec)> {
            graph::begin_capture(stream_handle)?;

            let draft_input_src = self.cuda_prefill()?.token_u32.ptr_at(4)?;
            self.cuda_forward()?
                .mtp_verify_token_u32
                .copy_from_device_ptr_at(0, draft_input_src, draft_count * 4)?;
            self.prefill_cuda_chunk(
                verify_tokens,
                start_position,
                start_position_device_i32,
                false,
            )?;
            self.final_norm_prefill_rows(verify_tokens)?;
            if !assume_accept && mtp_batched_lm_head_enabled() {
                self.prefill_rows_logits_for_mtp_verify(verify_tokens)?;
                let verified_base_ptr = self
                    .cuda_forward()?
                    .mtp_verify_token_u32
                    .ptr_at(MTP_GRAPH_VERIFIED_BASE * 4)?;
                let next_token_ptr = self
                    .cuda_forward()?
                    .mtp_verify_token_u32
                    .ptr_at(draft_count * 4)?;
                self.queue_sample_greedy_rows_into(
                    self.cuda_forward()?.mtp_logits.ptr(),
                    verify_tokens,
                    verified_base_ptr,
                    next_token_ptr,
                )?;
            } else {
                if !assume_accept {
                    for draft_idx in 0..draft_count {
                        self.prefill_row_logits(draft_idx)?;
                        let verified_ptr = self
                            .cuda_forward()?
                            .mtp_verify_token_u32
                            .ptr_at((MTP_GRAPH_VERIFIED_BASE + draft_idx) * 4)?;
                        self.queue_sample_greedy_into(verified_ptr)?;
                    }
                }

                self.prefill_row_logits(draft_count)?;
                let next_token_ptr = self
                    .cuda_forward()?
                    .mtp_verify_token_u32
                    .ptr_at(draft_count * 4)?;
                self.queue_sample_greedy_into(next_token_ptr)?;
            }

            self.run_mtp_prefill_chunk_with_tokens(
                verify_tokens,
                start_position,
                start_position_device_i32,
                self.cuda_prefill()?.normed.ptr(),
                self.cuda_forward()?.mtp_verify_token_u32.ptr(),
                true,
            )?;
            let first_next_draft_ptr = self
                .cuda_forward()?
                .mtp_verify_token_u32
                .ptr_at(MTP_GRAPH_NEXT_DRAFT_BASE * 4)?;
            self.queue_sample_greedy_into(first_next_draft_ptr)?;

            if draft_count > 1 {
                let hidden = self.topology.hidden_size;
                let last_hidden = Self::ptr_offset(
                    self.cuda_prefill()?.normed.ptr(),
                    (verify_tokens - 1) * hidden * 2,
                )?;
                for draft_idx in 1..draft_count {
                    let position = start_position
                        .checked_add(verify_tokens)
                        .and_then(|value| value.checked_add(draft_idx - 1))
                        .ok_or_else(|| {
                            CoreError::Runtime("MTP graph draft position overflow".to_owned())
                        })?;
                    let position_ptr = Self::ptr_offset(
                        start_position_device_i32,
                        (verify_tokens + draft_idx - 1) * 4,
                    )?;
                    let input_slot = MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx - 1;
                    let output_slot = MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx;
                    let input_token = self
                        .cuda_forward()?
                        .mtp_verify_token_u32
                        .ptr_at(input_slot * 4)?;
                    let target_hidden = if draft_idx == 1 {
                        last_hidden
                    } else {
                        self.cuda_prefill()?.normed.ptr()
                    };
                    self.run_mtp_prefill_chunk_with_tokens(
                        1,
                        position,
                        position_ptr,
                        target_hidden,
                        input_token,
                        true,
                    )?;
                    let output_token = self
                        .cuda_forward()?
                        .mtp_verify_token_u32
                        .ptr_at(output_slot * 4)?;
                    self.queue_sample_greedy_into(output_token)?;
                }
            }

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

        self.decode_graph = Some(DecodeGraphState {
            kind: DecodeGraphKind::MtpVerifyMulti {
                drafts: draft_count,
                assume_accept,
                batched_lm_head,
            },
            stream,
            exec,
            raw_graph,
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn launch_mtp_verify_graph_multi_tokens(&self, draft_count: usize) -> Result<()> {
        let graph_state = self.decode_graph.as_ref().ok_or_else(|| {
            CoreError::Runtime("MTP multi verify graph launch requested before capture".to_owned())
        })?;
        if graph_state.kind
            != (DecodeGraphKind::MtpVerifyMulti {
                drafts: draft_count,
                assume_accept: mtp_assume_accept_enabled(),
                batched_lm_head: mtp_batched_lm_head_enabled(),
            })
        {
            return Err(CoreError::Runtime(
                "MTP multi verify graph launch found a different active graph".to_owned(),
            ));
        }
        qwen36_fp4_kernels::graph::launch(graph_state.exec, graph_state.stream.handle())?;
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

        if self
            .decode_graph
            .as_ref()
            .is_some_and(|graph| graph.kind == DecodeGraphKind::MtpDecodeOne)
        {
            return Ok(());
        }
        if self.decode_graph.is_some() {
            self.disable_decode_graph()?;
        }
        Self::ensure_graph_capture_allowed()?;
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
            kind: DecodeGraphKind::MtpDecodeOne,
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
            let synchronize_result = graph_state.stream.handle().synchronize();
            qwen36_fp4_kernels::graph::set_active_stream(
                qwen36_fp4_kernels::graph::CudaStream::NULL,
            );
            synchronize_result?;
            // Drop runs the destructors below.
            drop(graph_state);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn activate_existing_graph_stream(&self, kind: DecodeGraphKind) {
        if let Some(graph_state) = self.decode_graph.as_ref() {
            if graph_state.kind == kind {
                qwen36_fp4_kernels::graph::set_active_stream(graph_state.stream.handle());
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn ensure_graph_capture_allowed() -> Result<()> {
        const DEBUG_DUMP_ENVS: [&str; 3] = [
            "QWEN36_DEBUG_DUMP_DIR",
            "QWEN36_DEBUG_DUMP_DECODE",
            "QWEN36_DEBUG_DUMP_ALL_LAYERS",
        ];
        if DEBUG_DUMP_ENVS
            .iter()
            .any(|name| std::env::var_os(name).is_some())
        {
            return Err(CoreError::Runtime(
                "graph capture is incompatible with debug dumps; unset QWEN36_DEBUG_DUMP_*"
                    .to_owned(),
            ));
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
        if let Some(deltanet_checkpoint) = &runtime.deltanet_checkpoint {
            deltanet_checkpoint.memset(0)?;
        }
        runtime.conv_history.memset(0)?;
        if let Some(conv_history_checkpoint) = &runtime.conv_history_checkpoint {
            conv_history_checkpoint.memset(0)?;
        }
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
            self.prefill_cuda_chunk(chunk, start_position, DevicePtr::NULL, emit_logits)?;
            if self.config.mtp_speculative_tokens > 0 && !emit_logits {
                let shifted_tokens = &prompt_tokens[consumed + 1..consumed + chunk + 1];
                self.run_mtp_prefill_chunk_from_current_prefill(
                    chunk,
                    start_position,
                    DevicePtr::NULL,
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
        self.run_mtp_prefill_chunk_from_current_prefill(
            final_chunk,
            start,
            DevicePtr::NULL,
            &shifted_tokens,
            true,
        )
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
    pub fn prepare_mtp_drafts_from_sampled(
        &self,
        prompt_tokens: &[u32],
        sampled_token: u32,
        draft_count: usize,
    ) -> Result<Vec<u32>> {
        if draft_count == 0 {
            return Ok(Vec::new());
        }
        if draft_count > MTP_MAX_DRAFT_TOKENS {
            return Err(CoreError::Runtime(format!(
                "MTP draft_count {draft_count} exceeds the supported maximum of {MTP_MAX_DRAFT_TOKENS}"
            )));
        }
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
        self.run_mtp_prefill_chunk_from_current_prefill(
            final_chunk,
            start,
            DevicePtr::NULL,
            &shifted_tokens,
            true,
        )?;
        self.queue_sample_greedy()?;
        let first_draft = self.read_sampled_token()?;
        let mut drafts = vec![first_draft];

        if draft_count > 1 {
            let hidden = self.topology.hidden_size;
            let target_hidden = Self::ptr_offset(
                self.cuda_prefill()?.normed.ptr(),
                (final_chunk - 1) * hidden * 2,
            )?;
            drafts.extend(self.generate_mtp_drafts_from_target(
                first_draft,
                target_hidden,
                prompt_tokens.len(),
                draft_count - 1,
            )?);
        }
        Ok(drafts)
    }

    /// Snapshot the runtime state needed to roll back an MTP verify chunk if
    /// a draft is rejected. Captures DeltaNet recurrent state and the conv1d
    /// history buffer by default. K/V slices are intentionally skipped on the
    /// hot path; `QWEN36_MTP_SNAPSHOT_KV=1` restores the conservative copy.
    #[cfg(feature = "cuda")]
    fn mtp_snapshot_state(&self, start_position: usize, token_count: usize) -> Result<()> {
        let runtime = self.cuda_runtime()?;
        if mtp_recurrent_snapshot_enabled() {
            let deltanet_bytes = runtime.deltanet_state.bytes();
            let deltanet_checkpoint = runtime.deltanet_checkpoint.as_ref().ok_or_else(|| {
                CoreError::Runtime(
                    "MTP recurrent snapshot requested but DeltaNet checkpoint was not allocated"
                        .to_owned(),
                )
            })?;
            deltanet_checkpoint.copy_from_device(&runtime.deltanet_state, deltanet_bytes)?;
            let conv_bytes = runtime.conv_history.bytes();
            if conv_bytes > 0 {
                let conv_history_checkpoint =
                    runtime.conv_history_checkpoint.as_ref().ok_or_else(|| {
                        CoreError::Runtime(
                            "MTP recurrent snapshot requested but conv-history checkpoint was not allocated"
                                .to_owned(),
                        )
                    })?;
                conv_history_checkpoint.copy_from_device(&runtime.conv_history, conv_bytes)?;
            }
        }
        if cuda_env_bool("QWEN36_MTP_SNAPSHOT_KV") {
            self.mtp_kv_slice_copy(start_position, token_count, /* save = */ true)?;
        }
        Ok(())
    }

    /// Restore the runtime state captured by [`mtp_snapshot_state`]. Must only
    /// be called once per snapshot — calling it twice will overwrite live
    /// state with stale data.
    #[cfg(feature = "cuda")]
    fn mtp_restore_state(&self, start_position: usize, token_count: usize) -> Result<()> {
        if !mtp_recurrent_snapshot_enabled() {
            return Err(CoreError::Runtime(
                "MTP rejection requires QWEN36_MTP_SNAPSHOT_RECURRENT=1; fast mode cannot recover recurrent state".to_owned(),
            ));
        }
        let runtime = self.cuda_runtime()?;
        let deltanet_bytes = runtime.deltanet_state.bytes();
        let deltanet_checkpoint = runtime.deltanet_checkpoint.as_ref().ok_or_else(|| {
            CoreError::Runtime(
                "MTP recurrent restore requested but DeltaNet checkpoint was not allocated"
                    .to_owned(),
            )
        })?;
        runtime
            .deltanet_state
            .copy_from_device(deltanet_checkpoint, deltanet_bytes)?;
        let conv_bytes = runtime.conv_history.bytes();
        if conv_bytes > 0 {
            let conv_history_checkpoint =
                runtime.conv_history_checkpoint.as_ref().ok_or_else(|| {
                    CoreError::Runtime(
                    "MTP recurrent restore requested but conv-history checkpoint was not allocated"
                        .to_owned(),
                )
                })?;
            runtime
                .conv_history
                .copy_from_device(conv_history_checkpoint, conv_bytes)?;
        }
        if cuda_env_bool("QWEN36_MTP_SNAPSHOT_KV") {
            self.mtp_kv_slice_copy(start_position, token_count, /* save = */ false)?;
        }
        Ok(())
    }

    /// Copy the verify K/V slice at `start_position` between live caches and
    /// `mtp_kv_snapshot`. This is only needed when `QWEN36_MTP_SNAPSHOT_KV=1`;
    /// the default rollback path restores recurrent state, then reruns the
    /// committed prefix and overwrites the only K/V slots future attention can
    /// observe.
    #[cfg(feature = "cuda")]
    fn mtp_kv_slice_copy(
        &self,
        start_position: usize,
        token_count: usize,
        save: bool,
    ) -> Result<()> {
        if token_count == 0 || token_count > MtpKvSnapshotLayout::VERIFY_TOKENS {
            return Err(CoreError::Runtime(format!(
                "MTP KV snapshot token_count {token_count} is outside 1..={}",
                MtpKvSnapshotLayout::VERIFY_TOKENS
            )));
        }
        let runtime = self.cuda_runtime()?;
        let mtp_kv_snapshot = runtime.mtp_kv_snapshot.as_ref().ok_or_else(|| {
            CoreError::Runtime(
                "MTP KV snapshot requested but snapshot buffer was not allocated".to_owned(),
            )
        })?;
        let layout = &runtime.mtp_kv_snapshot_layout;
        let main_row_bytes = layout.main_slice_bytes / MtpKvSnapshotLayout::VERIFY_TOKENS;
        let main_slice_bytes = main_row_bytes
            .checked_mul(token_count)
            .ok_or_else(|| CoreError::Runtime("MTP KV slice size overflow".to_owned()))?;
        let main_position_offset = start_position
            .checked_mul(main_row_bytes)
            .ok_or_else(|| CoreError::Runtime("MTP KV slice offset overflow".to_owned()))?;

        if let Some(kv_cache) = &runtime.kv_cache {
            for (idx, layer) in self.state.kv_cache.layers.iter().enumerate() {
                let snapshot_k = layout.main_k_offsets[idx];
                let snapshot_v = layout.main_v_offsets[idx];
                let live_k = (layer.k_offset_bytes as usize)
                    .checked_add(main_position_offset)
                    .ok_or_else(|| CoreError::Runtime("KV K offset overflow".to_owned()))?;
                let live_v = (layer.v_offset_bytes as usize)
                    .checked_add(main_position_offset)
                    .ok_or_else(|| CoreError::Runtime("KV V offset overflow".to_owned()))?;
                if save {
                    let src_k = kv_cache.ptr_at(live_k)?;
                    let src_v = kv_cache.ptr_at(live_v)?;
                    mtp_kv_snapshot.copy_from_device_ptr_at(snapshot_k, src_k, main_slice_bytes)?;
                    mtp_kv_snapshot.copy_from_device_ptr_at(snapshot_v, src_v, main_slice_bytes)?;
                } else {
                    let src_k = mtp_kv_snapshot.ptr_at(snapshot_k)?;
                    let src_v = mtp_kv_snapshot.ptr_at(snapshot_v)?;
                    kv_cache.copy_from_device_ptr_at(live_k, src_k, main_slice_bytes)?;
                    kv_cache.copy_from_device_ptr_at(live_v, src_v, main_slice_bytes)?;
                }
            }
        }

        if let (Some(mtp_kv_cache), Some(mtp_k_off), Some(mtp_v_off)) = (
            runtime.mtp_kv_cache.as_ref(),
            layout.mtp_k_offset,
            layout.mtp_v_offset,
        ) {
            let mtp_row_bytes = layout.mtp_slice_bytes / MtpKvSnapshotLayout::VERIFY_TOKENS;
            let mtp_slice_bytes = mtp_row_bytes
                .checked_mul(token_count)
                .ok_or_else(|| CoreError::Runtime("MTP KV slice size overflow".to_owned()))?;
            let mtp_position_offset = start_position
                .checked_mul(mtp_row_bytes)
                .ok_or_else(|| CoreError::Runtime("MTP KV slice offset overflow".to_owned()))?;
            let plane_bytes = self.mtp_kv_cache_plane_bytes()?;
            let live_k = mtp_position_offset;
            let live_v = plane_bytes
                .checked_add(mtp_position_offset)
                .ok_or_else(|| CoreError::Runtime("MTP V plane offset overflow".to_owned()))?;
            if save {
                let src_k = mtp_kv_cache.ptr_at(live_k)?;
                let src_v = mtp_kv_cache.ptr_at(live_v)?;
                mtp_kv_snapshot.copy_from_device_ptr_at(mtp_k_off, src_k, mtp_slice_bytes)?;
                mtp_kv_snapshot.copy_from_device_ptr_at(mtp_v_off, src_v, mtp_slice_bytes)?;
            } else {
                let src_k = mtp_kv_snapshot.ptr_at(mtp_k_off)?;
                let src_v = mtp_kv_snapshot.ptr_at(mtp_v_off)?;
                mtp_kv_cache.copy_from_device_ptr_at(live_k, src_k, mtp_slice_bytes)?;
                mtp_kv_cache.copy_from_device_ptr_at(live_v, src_v, mtp_slice_bytes)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn write_prefill_tokens_and_positions(
        &self,
        tokens: &[u32],
        start_position: usize,
    ) -> Result<()> {
        self.write_prefill_tokens_and_position_count(tokens, start_position, tokens.len())
    }

    #[cfg(feature = "cuda")]
    fn write_prefill_tokens_and_position_count(
        &self,
        tokens: &[u32],
        start_position: usize,
        position_count: usize,
    ) -> Result<()> {
        if position_count < tokens.len() {
            return Err(CoreError::Runtime(format!(
                "position_count {position_count} is smaller than token count {}",
                tokens.len()
            )));
        }
        let mut token_bytes = Vec::with_capacity(tokens.len() * 4);
        for token in tokens.iter().copied() {
            token_bytes.extend_from_slice(&token.to_ne_bytes());
        }
        let mut position_bytes = Vec::with_capacity(position_count * 4);
        for idx in 0..position_count {
            let position = start_position
                .checked_add(idx)
                .ok_or_else(|| CoreError::Runtime("prefill position overflow".to_owned()))?;
            let position = i32::try_from(position).map_err(|_| {
                CoreError::Runtime(format!("position {position} does not fit i32 for RoPE"))
            })?;
            position_bytes.extend_from_slice(&position.to_ne_bytes());
        }
        let prefill = self.cuda_prefill()?;
        prefill.token_u32.copy_from_host(&token_bytes)?;
        prefill.position_i32.copy_from_host(&position_bytes)
    }

    #[cfg(feature = "cuda")]
    fn sample_prefill_row_to_host(&self, row: usize) -> Result<u32> {
        self.prefill_row_logits(row)?;
        self.queue_sample_greedy_to_current_token()?;
        self.read_current_token()
    }

    #[cfg(feature = "cuda")]
    fn read_mtp_token_bundle(&self, base_slot: usize, count: usize) -> Result<Vec<u32>> {
        if base_slot + count > MTP_GRAPH_BUNDLE_U32S {
            return Err(CoreError::Runtime(format!(
                "MTP token bundle read {base_slot}..{} exceeds bundle slots {MTP_GRAPH_BUNDLE_U32S}",
                base_slot + count
            )));
        }
        let mut bytes = vec![0_u8; count * 4];
        self.cuda_forward()?
            .mtp_verify_token_u32
            .copy_to_host_at(base_slot * 4, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    #[cfg(feature = "cuda")]
    fn generate_mtp_drafts_from_target(
        &self,
        first_shifted_token: u32,
        first_target_hidden_bf16: DevicePtr,
        first_mtp_position: usize,
        draft_count: usize,
    ) -> Result<Vec<u32>> {
        if draft_count == 0 {
            return Ok(Vec::new());
        }
        if MTP_GRAPH_NEXT_DRAFT_BASE + draft_count > MTP_GRAPH_BUNDLE_U32S {
            return Err(CoreError::Runtime(format!(
                "MTP draft_count {draft_count} exceeds bundle capacity"
            )));
        }
        self.cuda_prefill()?
            .token_u32
            .copy_from_host(&first_shifted_token.to_ne_bytes())?;
        self.run_mtp_prefill_chunk(
            1,
            first_mtp_position,
            DevicePtr::NULL,
            first_target_hidden_bf16,
            true,
        )?;
        let first_out = self
            .cuda_forward()?
            .mtp_verify_token_u32
            .ptr_at(MTP_GRAPH_NEXT_DRAFT_BASE * 4)?;
        self.queue_sample_greedy_into(first_out)?;

        for draft_idx in 0..draft_count {
            if draft_idx == 0 {
                continue;
            }
            let position = first_mtp_position.checked_add(draft_idx).ok_or_else(|| {
                CoreError::Runtime("MTP recursive draft position overflow".to_owned())
            })?;
            let input_slot = MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx - 1;
            let output_slot = MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx;
            let input_token = self
                .cuda_forward()?
                .mtp_verify_token_u32
                .ptr_at(input_slot * 4)?;
            self.run_mtp_prefill_chunk_with_tokens(
                1,
                position,
                DevicePtr::NULL,
                self.cuda_prefill()?.normed.ptr(),
                input_token,
                true,
            )?;
            let output_token = self
                .cuda_forward()?
                .mtp_verify_token_u32
                .ptr_at(output_slot * 4)?;
            self.queue_sample_greedy_into(output_token)?;
        }
        self.read_mtp_token_bundle(MTP_GRAPH_NEXT_DRAFT_BASE, draft_count)
    }

    #[cfg(feature = "cuda")]
    fn generate_mtp_drafts_from_committed_prefill(
        &self,
        shifted_tokens: &[u32],
        start_position: usize,
        target_hidden_bf16: DevicePtr,
        draft_count: usize,
    ) -> Result<Vec<u32>> {
        if draft_count == 0 {
            return Ok(Vec::new());
        }
        if shifted_tokens.is_empty() {
            return Err(CoreError::Runtime(
                "MTP committed prefill requires at least one shifted token".to_owned(),
            ));
        }

        let mut token_bytes = Vec::with_capacity(shifted_tokens.len() * 4);
        for token in shifted_tokens {
            token_bytes.extend_from_slice(&token.to_ne_bytes());
        }
        self.cuda_prefill()?
            .token_u32
            .copy_from_host(&token_bytes)?;
        self.run_mtp_prefill_chunk(
            shifted_tokens.len(),
            start_position,
            DevicePtr::NULL,
            target_hidden_bf16,
            true,
        )?;
        let first_out = self
            .cuda_forward()?
            .mtp_verify_token_u32
            .ptr_at(MTP_GRAPH_NEXT_DRAFT_BASE * 4)?;
        self.queue_sample_greedy_into(first_out)?;

        if draft_count > 1 {
            let hidden = self.topology.hidden_size;
            let last_hidden = Self::ptr_offset(
                self.cuda_prefill()?.normed.ptr(),
                (shifted_tokens.len() - 1) * hidden * 2,
            )?;
            for draft_idx in 1..draft_count {
                let position = start_position
                    .checked_add(shifted_tokens.len())
                    .and_then(|value| value.checked_add(draft_idx - 1))
                    .ok_or_else(|| {
                        CoreError::Runtime("MTP recursive draft position overflow".to_owned())
                    })?;
                let input_slot = MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx - 1;
                let output_slot = MTP_GRAPH_NEXT_DRAFT_BASE + draft_idx;
                let input_token = self
                    .cuda_forward()?
                    .mtp_verify_token_u32
                    .ptr_at(input_slot * 4)?;
                let target_hidden = if draft_idx == 1 {
                    last_hidden
                } else {
                    self.cuda_prefill()?.normed.ptr()
                };
                self.run_mtp_prefill_chunk_with_tokens(
                    1,
                    position,
                    DevicePtr::NULL,
                    target_hidden,
                    input_token,
                    true,
                )?;
                let output_token = self
                    .cuda_forward()?
                    .mtp_verify_token_u32
                    .ptr_at(output_slot * 4)?;
                self.queue_sample_greedy_into(output_token)?;
            }
        }
        self.read_mtp_token_bundle(MTP_GRAPH_NEXT_DRAFT_BASE, draft_count)
    }

    #[cfg(feature = "cuda")]
    fn recover_after_mtp_multi_reject(
        &mut self,
        current_token: u32,
        draft_tokens: &[u32],
        rejected_draft_idx: usize,
        verified_token: u32,
        start_position: usize,
        verify_tokens: usize,
        next_draft_count: usize,
    ) -> Result<MtpMultiVerifyResult> {
        let committed_tokens = rejected_draft_idx + 1;
        self.mtp_restore_state(start_position, verify_tokens)?;

        let mut committed_input = Vec::with_capacity(committed_tokens);
        committed_input.push(current_token);
        committed_input.extend_from_slice(&draft_tokens[..rejected_draft_idx]);
        self.write_prefill_tokens_and_positions(&committed_input, start_position)?;
        self.prefill_cuda_chunk(committed_tokens, start_position, DevicePtr::NULL, false)?;
        self.final_norm_prefill_rows(committed_tokens)?;

        let next_draft_tokens = if next_draft_count > 0 {
            let target_hidden = self.cuda_prefill()?.normed.ptr();
            let mut shifted_tokens = Vec::with_capacity(committed_tokens);
            shifted_tokens.extend_from_slice(&draft_tokens[..rejected_draft_idx]);
            shifted_tokens.push(verified_token);
            self.generate_mtp_drafts_from_committed_prefill(
                &shifted_tokens,
                start_position,
                target_hidden,
                next_draft_count,
            )?
        } else {
            Vec::new()
        };

        self.state.advance(committed_tokens);
        Ok(MtpMultiVerifyResult {
            accepted_drafts: rejected_draft_idx,
            rejected: true,
            next_token: Some(verified_token),
            next_draft_tokens,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn verify_mtp_draft_two_tokens(
        &mut self,
        current_token: u32,
        draft_token: u32,
        need_next_token: bool,
        need_next_draft: bool,
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

        let mut token_bytes = [0_u8; 8];
        token_bytes[..4].copy_from_slice(&current_token.to_ne_bytes());
        token_bytes[4..].copy_from_slice(&draft_token.to_ne_bytes());
        let position_0 = i32::try_from(start_position).map_err(|_| {
            CoreError::Runtime(format!(
                "position {start_position} does not fit i32 for RoPE"
            ))
        })?;
        let position_1_usize = start_position + 1;
        let position_1 = i32::try_from(position_1_usize).map_err(|_| {
            CoreError::Runtime(format!(
                "position {position_1_usize} does not fit i32 for RoPE"
            ))
        })?;
        if need_next_draft {
            self.activate_existing_graph_stream(DecodeGraphKind::MtpVerifyOne);
        }
        let mut position_bytes = [0_u8; 8];
        position_bytes[..4].copy_from_slice(&position_0.to_ne_bytes());
        position_bytes[4..].copy_from_slice(&position_1.to_ne_bytes());
        {
            let prefill = self.cuda_prefill()?;
            prefill.token_u32.copy_from_host(&token_bytes)?;
            prefill.position_i32.copy_from_host(&position_bytes)?;
        }

        // Snapshot DeltaNet recurrent state, conv1d history, and the K/V
        // slices for the two verify positions before the verify chunk mutates
        // them. On rejection we roll back these in-place rather than doing a
        // catastrophic reset+full-reprefill from the CLI side.
        self.mtp_snapshot_state(start_position, 2)?;

        if need_next_draft {
            let mut mtp_verify_tokens = [0_u8; 16];
            mtp_verify_tokens[..4].copy_from_slice(&draft_token.to_ne_bytes());
            self.cuda_forward()?
                .mtp_verify_token_u32
                .copy_from_host(&mtp_verify_tokens)?;
            self.ensure_mtp_verify_graph_two_tokens(start_position)?;
            self.launch_mtp_verify_graph_two_tokens()?;

            let mut verify_bytes = [0_u8; 16];
            self.cuda_forward()?
                .mtp_verify_token_u32
                .copy_to_host(&mut verify_bytes)?;
            let verified_token = u32::from_ne_bytes([
                verify_bytes[8],
                verify_bytes[9],
                verify_bytes[10],
                verify_bytes[11],
            ]);
            if verified_token != draft_token {
                return self.recover_after_mtp_reject(
                    current_token,
                    verified_token,
                    start_position,
                    /* need_new_draft = */ need_next_token,
                );
            }
            let next_token = u32::from_ne_bytes([
                verify_bytes[4],
                verify_bytes[5],
                verify_bytes[6],
                verify_bytes[7],
            ]);
            let next_draft_token = u32::from_ne_bytes([
                verify_bytes[12],
                verify_bytes[13],
                verify_bytes[14],
                verify_bytes[15],
            ]);
            self.state.advance(2);

            return Ok(MtpVerifyResult {
                accepted: true,
                verified_token,
                next_token: Some(next_token),
                next_draft_token: Some(next_draft_token),
            });
        }

        self.prefill_cuda_chunk(2, start_position, DevicePtr::NULL, false)?;
        self.final_norm_prefill_rows(2)?;

        self.prefill_row_logits(0)?;
        self.queue_sample_greedy_to_current_token()?;
        let verified_token = self.read_current_token()?;
        if verified_token != draft_token {
            return self.recover_after_mtp_reject(
                current_token,
                verified_token,
                start_position,
                need_next_token,
            );
        }

        if !need_next_token {
            self.state.advance(2);
            return Ok(MtpVerifyResult {
                accepted: true,
                verified_token,
                next_token: None,
                next_draft_token: None,
            });
        }

        self.prefill_row_logits(1)?;
        self.queue_sample_greedy_to_current_token()?;
        let next_token = self.read_current_token()?;

        if !need_next_draft {
            self.state.advance(2);
            return Ok(MtpVerifyResult {
                accepted: true,
                verified_token,
                next_token: Some(next_token),
                next_draft_token: None,
            });
        }

        let mut mtp_tokens = [0_u8; 8];
        mtp_tokens[..4].copy_from_slice(&draft_token.to_ne_bytes());
        mtp_tokens[4..].copy_from_slice(&next_token.to_ne_bytes());
        self.cuda_prefill()?.token_u32.copy_from_host(&mtp_tokens)?;
        self.run_mtp_prefill_chunk(
            2,
            start_position,
            DevicePtr::NULL,
            self.cuda_prefill()?.normed.ptr(),
            true,
        )?;
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
    pub fn verify_mtp_draft_tokens(
        &mut self,
        current_token: u32,
        draft_tokens: &[u32],
        need_next_token_on_full_accept: bool,
        next_draft_count: usize,
    ) -> Result<MtpMultiVerifyResult> {
        if self.config.mtp_speculative_tokens == 0 {
            return Err(CoreError::Runtime(
                "MTP multi-token verification requested while mtp_speculative_tokens is 0"
                    .to_owned(),
            ));
        }
        if draft_tokens.is_empty() || draft_tokens.len() > MTP_MAX_DRAFT_TOKENS {
            return Err(CoreError::Runtime(format!(
                "MTP multi-token verification expects 1..={MTP_MAX_DRAFT_TOKENS} drafts, got {}",
                draft_tokens.len()
            )));
        }
        if next_draft_count > MTP_MAX_DRAFT_TOKENS {
            return Err(CoreError::Runtime(format!(
                "MTP next_draft_count {next_draft_count} exceeds the supported maximum of {MTP_MAX_DRAFT_TOKENS}"
            )));
        }

        let start_position = self.state.position;
        let verify_tokens = draft_tokens.len() + 1;
        if start_position + verify_tokens > self.config.max_context {
            return Err(CoreError::Runtime(format!(
                "MTP verification of {verify_tokens} tokens at position {start_position} exceeds max_context {}",
                self.config.max_context
            )));
        }
        let available_rows_after_verify =
            self.config.max_context - (start_position + verify_tokens);
        let effective_next_draft_count = next_draft_count.min(available_rows_after_verify + 1);
        let use_multi_graph = draft_tokens.len() >= 2
            && need_next_token_on_full_accept
            && effective_next_draft_count == draft_tokens.len()
            && std::env::var("QWEN36_MTP_MULTI_GRAPH_DISABLE").is_err();
        let trace_mtp = std::env::var("QWEN36_MTP_TRACE").is_ok();
        if use_multi_graph {
            self.activate_existing_graph_stream(DecodeGraphKind::MtpVerifyMulti {
                drafts: draft_tokens.len(),
                assume_accept: mtp_assume_accept_enabled(),
                batched_lm_head: mtp_batched_lm_head_enabled(),
            });
        }

        let mut verify_input = Vec::with_capacity(verify_tokens);
        verify_input.push(current_token);
        verify_input.extend_from_slice(draft_tokens);
        let position_count = if use_multi_graph {
            verify_tokens + effective_next_draft_count
        } else {
            verify_tokens
        };
        self.write_prefill_tokens_and_position_count(
            &verify_input,
            start_position,
            position_count,
        )?;

        self.mtp_snapshot_state(start_position, verify_tokens)?;
        if use_multi_graph {
            self.ensure_mtp_verify_graph_multi_tokens(draft_tokens.len(), start_position)?;
            self.launch_mtp_verify_graph_multi_tokens(draft_tokens.len())?;

            let mut verify_bytes = [0_u8; MTP_GRAPH_BUNDLE_U32S * 4];
            self.cuda_forward()?
                .mtp_verify_token_u32
                .copy_to_host(&mut verify_bytes)?;
            let mut verified_tokens = Vec::with_capacity(draft_tokens.len());
            if mtp_assume_accept_enabled() {
                verified_tokens.extend_from_slice(draft_tokens);
            } else {
                for (draft_idx, draft_token) in draft_tokens.iter().copied().enumerate() {
                    let offset = (MTP_GRAPH_VERIFIED_BASE + draft_idx) * 4;
                    let verified_token = u32::from_ne_bytes([
                        verify_bytes[offset],
                        verify_bytes[offset + 1],
                        verify_bytes[offset + 2],
                        verify_bytes[offset + 3],
                    ]);
                    verified_tokens.push(verified_token);
                    if verified_token != draft_token {
                        if trace_mtp {
                            eprintln!(
                                "mtp.trace graph start={start_position} drafts={draft_tokens:?} verified={verified_tokens:?} reject_idx={draft_idx}"
                            );
                        }
                        return self.recover_after_mtp_multi_reject(
                            current_token,
                            draft_tokens,
                            draft_idx,
                            verified_token,
                            start_position,
                            verify_tokens,
                            effective_next_draft_count,
                        );
                    }
                }
            }

            let next_token_offset = draft_tokens.len() * 4;
            let next_token = u32::from_ne_bytes([
                verify_bytes[next_token_offset],
                verify_bytes[next_token_offset + 1],
                verify_bytes[next_token_offset + 2],
                verify_bytes[next_token_offset + 3],
            ]);
            let next_draft_tokens = (0..effective_next_draft_count)
                .map(|idx| {
                    let offset = (MTP_GRAPH_NEXT_DRAFT_BASE + idx) * 4;
                    u32::from_ne_bytes([
                        verify_bytes[offset],
                        verify_bytes[offset + 1],
                        verify_bytes[offset + 2],
                        verify_bytes[offset + 3],
                    ])
                })
                .collect::<Vec<_>>();
            if trace_mtp {
                eprintln!(
                    "mtp.trace graph start={start_position} drafts={draft_tokens:?} verified={verified_tokens:?} next={next_token} next_drafts={next_draft_tokens:?}"
                );
            }

            self.state.advance(verify_tokens);
            return Ok(MtpMultiVerifyResult {
                accepted_drafts: draft_tokens.len(),
                rejected: false,
                next_token: Some(next_token),
                next_draft_tokens,
            });
        }

        self.prefill_cuda_chunk(verify_tokens, start_position, DevicePtr::NULL, false)?;
        self.final_norm_prefill_rows(verify_tokens)?;

        for (draft_idx, draft_token) in draft_tokens.iter().copied().enumerate() {
            let verified_token = self.sample_prefill_row_to_host(draft_idx)?;
            if verified_token != draft_token {
                if trace_mtp {
                    eprintln!(
                        "mtp.trace host start={start_position} drafts={draft_tokens:?} reject_idx={draft_idx} verified={verified_token}"
                    );
                }
                return self.recover_after_mtp_multi_reject(
                    current_token,
                    draft_tokens,
                    draft_idx,
                    verified_token,
                    start_position,
                    verify_tokens,
                    effective_next_draft_count,
                );
            }
        }

        let accepted_drafts = draft_tokens.len();
        let next_token = if need_next_token_on_full_accept {
            Some(self.sample_prefill_row_to_host(accepted_drafts)?)
        } else {
            None
        };
        let next_draft_tokens =
            if let (Some(next_token), true) = (next_token, effective_next_draft_count > 0) {
                let target_hidden = self.cuda_prefill()?.normed.ptr();
                let mut shifted_tokens = Vec::with_capacity(verify_tokens);
                shifted_tokens.extend_from_slice(draft_tokens);
                shifted_tokens.push(next_token);
                self.generate_mtp_drafts_from_committed_prefill(
                    &shifted_tokens,
                    start_position,
                    target_hidden,
                    effective_next_draft_count,
                )?
            } else {
                Vec::new()
            };
        if trace_mtp {
            eprintln!(
                "mtp.trace host start={start_position} drafts={draft_tokens:?} next={next_token:?} next_drafts={next_draft_tokens:?}"
            );
        }

        self.state.advance(verify_tokens);
        Ok(MtpMultiVerifyResult {
            accepted_drafts,
            rejected: false,
            next_token,
            next_draft_tokens,
        })
    }

    /// Roll back the MTP verify chunk and commit just `current_token` at
    /// `start_position`, leaving the engine in the same state as if the CLI
    /// had run a single non-speculative decode step. When the caller needs a
    /// new draft token (i.e. it will iterate again), this also runs the MTP
    /// layer using `verified_token` as the shifted input so the returned
    /// draft is the same one a fresh `prepare_mtp_prefill_from_sampled` call
    /// would produce — at the cost of one extra layer-pass instead of a full
    /// prompt reprefill.
    #[cfg(feature = "cuda")]
    fn recover_after_mtp_reject(
        &mut self,
        current_token: u32,
        verified_token: u32,
        start_position: usize,
        need_new_draft: bool,
    ) -> Result<MtpVerifyResult> {
        self.mtp_restore_state(start_position, 2)?;

        let token_bytes = current_token.to_ne_bytes();
        let position = i32::try_from(start_position).map_err(|_| {
            CoreError::Runtime(format!(
                "position {start_position} does not fit i32 for RoPE"
            ))
        })?;
        let position_bytes = position.to_ne_bytes();
        {
            let prefill = self.cuda_prefill()?;
            prefill.token_u32.copy_from_host(&token_bytes)?;
            prefill.position_i32.copy_from_host(&position_bytes)?;
        }

        // Single-token main forward to commit `current_token` at
        // `start_position` (writes its K/V into the cache and advances the
        // DeltaNet recurrent state).
        self.prefill_cuda_chunk(1, start_position, DevicePtr::NULL, false)?;
        self.final_norm_prefill_rows(1)?;

        let next_draft_token = if need_new_draft {
            // Run MTP using `verified_token` as the shifted input on top of
            // the hidden state we just produced for `current_token`. This is
            // the same computation a fresh `prepare_mtp_prefill_from_sampled`
            // would do after the prompt prefill, but reusing the recovery
            // step we already paid for.
            let shifted_bytes = verified_token.to_ne_bytes();
            self.cuda_prefill()?
                .token_u32
                .copy_from_host(&shifted_bytes)?;
            let normed_ptr = self.cuda_prefill()?.normed.ptr();
            self.run_mtp_prefill_chunk(1, start_position, DevicePtr::NULL, normed_ptr, true)?;
            self.queue_sample_greedy()?;
            Some(self.read_sampled_token()?)
        } else {
            None
        };

        self.state.advance(1);
        Ok(MtpVerifyResult {
            accepted: false,
            verified_token,
            next_token: None,
            next_draft_token,
        })
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_prefill_chunk_from_current_prefill(
        &self,
        tokens: usize,
        start_position: usize,
        start_position_device_i32: DevicePtr,
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
        self.run_mtp_prefill_chunk(
            tokens,
            start_position,
            start_position_device_i32,
            prefill.normed.ptr(),
            emit_logits,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_prefill_chunk(
        &self,
        tokens: usize,
        start_position: usize,
        start_position_device_i32: DevicePtr,
        target_hidden_bf16: DevicePtr,
        emit_logits: bool,
    ) -> Result<()> {
        self.run_mtp_prefill_chunk_with_tokens(
            tokens,
            start_position,
            start_position_device_i32,
            target_hidden_bf16,
            self.cuda_prefill()?.token_u32.ptr(),
            emit_logits,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mtp_prefill_chunk_with_tokens(
        &self,
        tokens: usize,
        start_position: usize,
        start_position_device_i32: DevicePtr,
        target_hidden_bf16: DevicePtr,
        token_ids_u32: DevicePtr,
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
            token_ids_u32,
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
        self.run_mtp_full_attention_layer_prefill(
            layer,
            runtime,
            prefill,
            tokens,
            start_position,
            start_position_device_i32,
        )?;
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
        start_position_device_i32: DevicePtr,
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
                    start_position_device_i32,
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
    fn prefill_rows_logits_for_mtp_verify(&self, rows: usize) -> Result<()> {
        if rows == 0 || rows > MTP_MAX_DRAFT_TOKENS + 1 {
            return Err(CoreError::Runtime(format!(
                "MTP batched lm_head expects 1..={} rows, got {rows}",
                MTP_MAX_DRAFT_TOKENS + 1
            )));
        }
        let manifest = self
            .weights
            .as_ref()
            .ok_or_else(|| CoreError::Runtime("missing weight manifest".to_owned()))?;
        self.bf16_gemm_rows(
            &manifest.lm_head,
            self.cuda_prefill()?.normed.ptr(),
            self.cuda_forward()?.mtp_logits.ptr(),
            rows,
        )
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
        let mut prof_embed_ms = 0.0_f64;
        let mut prof_linear_attn_ms = 0.0_f64;
        let mut prof_full_attn_ms = 0.0_f64;
        let mut prof_mlp_ms = 0.0_f64;
        let mut prof_lm_head_ms = 0.0_f64;

        let embed_start = profile_decode.then(std::time::Instant::now);
        self.backend.embedding_lookup(&EmbeddingLookupSpec {
            tokens: 1,
            hidden: self.topology.hidden_size,
            vocab_size: self.topology.vocab_size,
            token_ids_u32,
            embedding_bf16: self.tensor_ptr(weights, &manifest.embed_tokens)?,
            output_bf16: forward.hidden.ptr(),
        })?;
        if let Some(embed_start) = embed_start {
            qwen36_fp4_kernels::cuda_synchronize()?;
            prof_embed_ms += embed_start.elapsed().as_secs_f64() * 1000.0;
        }
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
            let is_linear = matches!(layer, LayerWeights::LinearAttention(_));
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
                let elapsed = attn_start.elapsed().as_secs_f64() * 1000.0;
                if is_linear {
                    prof_linear_attn_ms += elapsed;
                } else {
                    prof_full_attn_ms += elapsed;
                }
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
                    prof_mlp_ms += mlp_start.elapsed().as_secs_f64() * 1000.0;
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
                    prof_mlp_ms += mlp_start.elapsed().as_secs_f64() * 1000.0;
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
                prof_lm_head_ms += logits_start.elapsed().as_secs_f64() * 1000.0;
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
        if profile_decode {
            let total_ms = prof_embed_ms
                + prof_linear_attn_ms
                + prof_full_attn_ms
                + prof_mlp_ms
                + prof_lm_head_ms;
            eprintln!(
                "decode.profile.summary pos={position} embed={:.3} linear_attn={:.3} full_attn={:.3} mlp={:.3} lm_head={:.3} total_measured={:.3}",
                prof_embed_ms,
                prof_linear_attn_ms,
                prof_full_attn_ms,
                prof_mlp_ms,
                prof_lm_head_ms,
                total_ms
            );
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

        let (conv_input_ptr, b_ptr, a_ptr, z_ptr) = if let Some(fused) =
            self.linear_attn_in_proj_fused_layer_opt(layer.layer_index)
        {
            // Combined qkv+b+a+z GEMM. Output layout in `forward.qkv` (BF16):
            //   [qkv: qkv_dim] [b padded: 128] [a padded: 128] [z: value_dim].
            // The 80-row pad after b and a sits at FP4 weight 0 so the GEMM
            // emits zeros there — the engine simply does not read those slots.
            let quantized = match prequantized_normed {
                Some(q) => q,
                None => {
                    let q = self.linear_attn_in_proj_quant(layer)?;
                    self.quantize_nvfp4_activation(forward.normed.ptr(), q)?;
                    q
                }
            };
            self.run_linear_attn_in_proj_fused_gemm(layer, fused, quantized, forward.qkv.ptr())?;

            let b_ptr = forward
                .qkv
                .ptr()
                .offset_bytes(fused.b_offset * 2)
                .ok_or_else(|| CoreError::Runtime("fused DeltaNet b offset overflow".to_owned()))?;
            let a_ptr = forward
                .qkv
                .ptr()
                .offset_bytes(fused.a_offset * 2)
                .ok_or_else(|| CoreError::Runtime("fused DeltaNet a offset overflow".to_owned()))?;
            let z_ptr = forward
                .qkv
                .ptr()
                .offset_bytes(fused.z_offset * 2)
                .ok_or_else(|| CoreError::Runtime("fused DeltaNet z offset overflow".to_owned()))?;
            (forward.qkv.ptr(), b_ptr, a_ptr, z_ptr)
        } else {
            let in_proj_linears = [
                (&layer.in_proj_qkv, forward.qkv.ptr()),
                (&layer.in_proj_b, forward.aux2.ptr()),
                (&layer.in_proj_a, forward.aux3.ptr()),
            ];
            if let Some(quantized) = prequantized_normed {
                for &(binding, output) in &in_proj_linears {
                    self.linear_with_quantized_nvfp4(binding, output, quantized)?;
                }
            } else {
                self.linears_same_input(forward.normed.ptr(), &in_proj_linears)?;
            }
            (
                forward.qkv.ptr(),
                forward.aux2.ptr(),
                forward.aux3.ptr(),
                forward.qkv.ptr(),
            )
        };

        self.backend
            .conv1d_gdn_gate_fused(&Conv1dGdnGateFusedSpec {
                channels: qkv_dim,
                kernel_size: self.topology.linear_conv_kernel_dim,
                conv_input_bf16: conv_input_ptr,
                conv_history_bf16: conv_history,
                conv_weight_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.conv1d_weight)?,
                conv_output_bf16: forward.aux.ptr(),
                heads: self.topology.linear_num_value_heads,
                gdn_a_bf16: a_ptr,
                gdn_b_bf16: b_ptr,
                gdn_a_log_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.a_log)?,
                gdn_dt_bias_bf16: self.tensor_ptr(self.cuda_weights()?, &layer.dt_bias)?,
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
        if self
            .linear_attn_in_proj_fused_layer_opt(layer.layer_index)
            .is_none()
        {
            if let Some(quantized) = prequantized_normed {
                self.linear_with_quantized_nvfp4(&layer.in_proj_z, forward.qkv.ptr(), quantized)?;
            } else {
                self.linear(&layer.in_proj_z, forward.normed.ptr(), forward.qkv.ptr())?;
            }
        }
        // `z_ptr` is either the fused-GEMM slice or the fallback z projection.
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
            gate_bf16: z_ptr,
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
            kv_cache_metadata: cache.ptr_at(layout.metadata_offset_bytes as usize)?,
            output_bf16: forward.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            kv_cache_dtype: Self::attention_kv_cache_dtype_code(self.config.kv_cache_dtype)?,
            position_device_i32,
            partial_acc_f32: forward.attn_partial_acc.ptr(),
            partial_max_f32: forward.attn_partial_max.ptr(),
            partial_denom_f32: forward.attn_partial_denom.ptr(),
            decode_n_splits: self.decode_attention_n_splits(),
            split_timesteps_per_block: self.attention_split_timesteps_per_block(),
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
            kv_cache_metadata: DevicePtr::NULL,
            output_bf16: forward.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            kv_cache_dtype: Self::attention_kv_cache_dtype_code(KvCacheDtype::Bf16)?,
            position_device_i32,
            partial_acc_f32: forward.attn_partial_acc.ptr(),
            partial_max_f32: forward.attn_partial_max.ptr(),
            partial_denom_f32: forward.attn_partial_denom.ptr(),
            decode_n_splits: self.decode_attention_n_splits(),
            split_timesteps_per_block: self.attention_split_timesteps_per_block(),
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
        start_position_device_i32: DevicePtr,
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
            kv_cache_metadata: DevicePtr::NULL,
            output_bf16: prefill.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            kv_cache_dtype: Self::attention_kv_cache_dtype_code(KvCacheDtype::Bf16)?,
            start_position_device_i32,
            partial_acc_f32: self.cuda_forward()?.attn_partial_acc.ptr(),
            partial_max_f32: self.cuda_forward()?.attn_partial_max.ptr(),
            partial_denom_f32: self.cuda_forward()?.attn_partial_denom.ptr(),
            prefill_n_splits: self
                .prefill_attention_n_splits(start_position + tokens, start_position_device_i32),
            split_timesteps_per_block: if start_position_device_i32 == DevicePtr::NULL {
                self.attention_split_timesteps_per_block_for(start_position + tokens)
            } else {
                self.attention_split_timesteps_per_block()
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
        // MTP head weights are BF16, so the fused FP4 GEMM path does not
        // apply. Stay on the per-projection path here.
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
        let fused_prefill = if cuda_prefill_fused_linear_attn_enabled() {
            self.linear_attn_in_proj_fused_layer_opt(layer.layer_index)
        } else {
            None
        };
        let mut used_fused_in_proj = false;
        if let (Some(fused), Some(quantized)) = (fused_prefill, shared_in_proj) {
            if prequantized_normed.is_none() {
                self.quantize_nvfp4_activation_rows(
                    prefill.normed.ptr(),
                    tokens,
                    quantized,
                    prefill,
                )?;
            }
            self.run_linear_attn_in_proj_fused_gemm_rows(
                layer,
                fused,
                quantized,
                prefill.qkv.ptr(),
                tokens,
                prefill,
            )?;
            self.backend.copy_strided_rows(&CopyStridedRowsSpec {
                rows: tokens,
                values: qkv_dim,
                input_stride: fused.combined_out_features,
                output_stride: qkv_dim,
                input_bf16: prefill.qkv.ptr(),
                output_bf16: prefill.block_out.ptr(),
            })?;
            self.backend.copy_strided_rows(&CopyStridedRowsSpec {
                rows: tokens,
                values: self.topology.linear_num_value_heads,
                input_stride: fused.combined_out_features,
                output_stride: self.topology.linear_num_value_heads,
                input_bf16: Self::ptr_offset(prefill.qkv.ptr(), fused.b_offset * 2)?,
                output_bf16: prefill.aux2.ptr(),
            })?;
            self.backend.copy_strided_rows(&CopyStridedRowsSpec {
                rows: tokens,
                values: self.topology.linear_num_value_heads,
                input_stride: fused.combined_out_features,
                output_stride: self.topology.linear_num_value_heads,
                input_bf16: Self::ptr_offset(prefill.qkv.ptr(), fused.a_offset * 2)?,
                output_bf16: prefill.aux3.ptr(),
            })?;
            used_fused_in_proj = true;
        } else if let Some(quantized) = shared_in_proj {
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
                    if used_fused_in_proj {
                        prefill.block_out.ptr()
                    } else {
                        prefill.qkv.ptr()
                    },
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
            input_bf16: if used_fused_in_proj {
                prefill.block_out.ptr()
            } else {
                prefill.qkv.ptr()
            },
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

        let z_bf16 = if used_fused_in_proj {
            let fused = fused_prefill.ok_or_else(|| {
                CoreError::Runtime("fused DeltaNet prefill state was lost".to_owned())
            })?;
            self.backend.copy_strided_rows(&CopyStridedRowsSpec {
                rows: tokens,
                values: value_dim,
                input_stride: fused.combined_out_features,
                output_stride: value_dim,
                input_bf16: Self::ptr_offset(prefill.qkv.ptr(), fused.z_offset * 2)?,
                output_bf16: prefill.block_out.ptr(),
            })?;
            prefill.block_out.ptr()
        } else if let Some(quantized) = shared_in_proj {
            self.linear_with_quantized_nvfp4_rows(
                &layer.in_proj_z,
                prefill.qkv.ptr(),
                tokens,
                quantized,
                prefill,
            )?;
            prefill.qkv.ptr()
        } else {
            self.linear_rows(
                &layer.in_proj_z,
                prefill.normed.ptr(),
                prefill.qkv.ptr(),
                tokens,
                prefill,
            )?;
            prefill.qkv.ptr()
        };
        if layer.layer_index == 0 {
            if let Ok(dir) = std::env::var("QWEN36_DEBUG_DUMP_DIR") {
                self.dump_buffer_to_disk(&dir, "layer0_z.bf16", z_bf16, tokens * value_dim)?;
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
            gate_bf16: z_bf16,
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
        start_position_device_i32: DevicePtr,
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
            kv_cache_metadata: cache.ptr_at(layout.metadata_offset_bytes as usize)?,
            output_bf16: prefill.aux3.ptr(),
            shape: AttentionShape {
                q_heads: self.topology.attention_num_heads,
                kv_heads: self.topology.attention_num_kv_heads,
                head_dim: self.topology.attention_head_dim,
                rope_dims: self.topology.attention_rope_dims(),
            },
            kv_cache_dtype: Self::attention_kv_cache_dtype_code(self.config.kv_cache_dtype)?,
            start_position_device_i32,
            partial_acc_f32: self.cuda_forward()?.attn_partial_acc.ptr(),
            partial_max_f32: self.cuda_forward()?.attn_partial_max.ptr(),
            partial_denom_f32: self.cuda_forward()?.attn_partial_denom.ptr(),
            prefill_n_splits: self
                .prefill_attention_n_splits(start_position + tokens, start_position_device_i32),
            split_timesteps_per_block: if start_position_device_i32 == DevicePtr::NULL {
                self.attention_split_timesteps_per_block_for(start_position + tokens)
            } else {
                self.attention_split_timesteps_per_block()
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
        let common = layer_common(layer);
        let layer_idx = match layer {
            LayerWeights::LinearAttention(layer) => layer.layer_index,
            LayerWeights::FullAttention(layer) => layer.layer_index,
        };
        if let Some(fused) = self.mlp_fused_main_opt(layer_idx) {
            return self.run_mlp_fused_combined_gemm(common, fused, forward, None);
        }
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
        let layer_idx = match layer {
            LayerWeights::LinearAttention(layer) => layer.layer_index,
            LayerWeights::FullAttention(layer) => layer.layer_index,
        };
        if let Some(fused) = self.mlp_fused_main_opt(layer_idx) {
            return self.run_mlp_fused_combined_gemm(common, fused, forward, Some(quantized));
        }
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
    fn mlp_fused_main_opt(&self, layer_idx: usize) -> Option<&MlpFusedLayer> {
        self.mlp_fused.as_ref()?.layers.get(layer_idx)
    }

    #[cfg(feature = "cuda")]
    fn linear_attn_in_proj_fused_layer_opt(
        &self,
        layer_idx: usize,
    ) -> Option<&LinearAttnInProjFused> {
        self.linear_attn_in_proj_fused
            .as_ref()?
            .layers
            .get(layer_idx)
            .and_then(|entry| entry.as_ref())
    }

    #[cfg(feature = "cuda")]
    fn linear_attn_in_proj_quant<'a>(
        &self,
        layer: &'a LinearAttentionLayerWeights,
    ) -> Result<Nvfp4ActivationQuant<'a>> {
        let LinearWeightBinding::Nvfp4 {
            weight,
            input_scale,
            ..
        } = &layer.in_proj_qkv
        else {
            return Err(CoreError::Runtime(
                "fused DeltaNet in_proj requires NVFP4 in_proj_qkv".to_owned(),
            ));
        };
        Ok(Nvfp4ActivationQuant {
            in_features: Self::nvfp4_in_features(weight)?,
            input_scale,
        })
    }

    #[cfg(feature = "cuda")]
    fn run_linear_attn_in_proj_fused_gemm(
        &self,
        layer: &LinearAttentionLayerWeights,
        fused: &LinearAttnInProjFused,
        quantized: Nvfp4ActivationQuant<'_>,
        output: DevicePtr,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let forward = self.cuda_forward()?;
        let LinearWeightBinding::Nvfp4 {
            tensor_scale: qkv_tensor_scale,
            ..
        } = &layer.in_proj_qkv
        else {
            return Err(CoreError::Runtime(
                "fused DeltaNet in_proj requires NVFP4 in_proj_qkv".to_owned(),
            ));
        };
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
        let alpha = self.tensor_scalar_f32(weights, qkv_tensor_scale)?
            * self.tensor_scalar_f32(weights, quantized.input_scale)?;
        let gemm_spec = Nvfp4GemmSpec {
            m: fused.combined_out_features,
            n: 1,
            k: quantized.in_features,
            a_fp4: fused.combined_weight.ptr(),
            a_scale: fused.combined_block_scale.ptr(),
            a_scale_2: self.tensor_ptr(weights, qkv_tensor_scale)?,
            b_fp4: forward.activation_fp4.ptr(),
            b_scale: forward.activation_scale.ptr(),
            b_scale_2: forward.activation_scale_2.ptr(),
            c_bf16: output,
            workspace,
            workspace_bytes,
            alpha,
            scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
        };
        self.backend.nvfp4_gemm(&gemm_spec).map_err(|err| {
            CoreError::Runtime(format!(
                "fused DeltaNet in_proj GEMM failed (m={}, n=1, k={}): {err}",
                fused.combined_out_features, quantized.in_features
            ))
        })
    }

    #[cfg(feature = "cuda")]
    fn run_linear_attn_in_proj_fused_gemm_rows(
        &self,
        layer: &LinearAttentionLayerWeights,
        fused: &LinearAttnInProjFused,
        quantized: Nvfp4ActivationQuant<'_>,
        output: DevicePtr,
        rows: usize,
        prefill: &GpuPrefillBuffers,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let LinearWeightBinding::Nvfp4 {
            tensor_scale: qkv_tensor_scale,
            ..
        } = &layer.in_proj_qkv
        else {
            return Err(CoreError::Runtime(
                "fused DeltaNet in_proj requires NVFP4 in_proj_qkv".to_owned(),
            ));
        };
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
        let alpha = self.tensor_scalar_f32(weights, qkv_tensor_scale)?
            * self.tensor_scalar_f32(weights, quantized.input_scale)?;
        let gemm_spec = Nvfp4GemmSpec {
            m: fused.combined_out_features,
            n: rows,
            k: quantized.in_features,
            a_fp4: fused.combined_weight.ptr(),
            a_scale: fused.combined_block_scale.ptr(),
            a_scale_2: self.tensor_ptr(weights, qkv_tensor_scale)?,
            b_fp4: prefill.activation_fp4.ptr(),
            b_scale: prefill.activation_scale.ptr(),
            b_scale_2: prefill.activation_scale_2.ptr(),
            c_bf16: output,
            workspace,
            workspace_bytes,
            alpha,
            scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
        };
        self.backend.nvfp4_gemm(&gemm_spec).map_err(|err| {
            CoreError::Runtime(format!(
                "fused DeltaNet in_proj prefill GEMM failed (m={}, n={}, k={}): {err}",
                fused.combined_out_features, rows, quantized.in_features
            ))
        })
    }

    /// Single combined-GEMM MLP: writes [gate || up] into `forward.aux` in
    /// one cuBLASLt FP4 GEMM, then runs SwiGLU on the two halves and the
    /// down_proj GEMM. Saves one FP4 GEMM launch per layer (× 64 layers per
    /// decode token).
    #[cfg(feature = "cuda")]
    fn run_mlp_fused_combined_gemm(
        &self,
        common: &CommonLayerWeights,
        fused: &MlpFusedLayer,
        forward: &GpuForwardBuffers,
        pre_quantized: Option<Nvfp4ActivationQuant<'_>>,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let intermediate = self.topology.intermediate_size;
        let LinearWeightBinding::Nvfp4 {
            weight: gate_weight,
            tensor_scale: gate_tensor_scale,
            input_scale: gate_input_scale,
            ..
        } = &common.mlp_gate_proj
        else {
            return Err(CoreError::Runtime(
                "fused MLP path requires NVFP4 gate_proj".to_owned(),
            ));
        };
        let in_features = Self::nvfp4_in_features(gate_weight)?;

        let quantized = match pre_quantized {
            Some(q) => q,
            None => {
                let q = Nvfp4ActivationQuant {
                    in_features,
                    input_scale: gate_input_scale,
                };
                self.quantize_nvfp4_activation(forward.normed.ptr(), q)?;
                q
            }
        };

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
        let alpha = self.tensor_scalar_f32(weights, gate_tensor_scale)?
            * self.tensor_scalar_f32(weights, quantized.input_scale)?;
        let gemm_spec = Nvfp4GemmSpec {
            m: fused.out_features,
            n: 1,
            k: quantized.in_features,
            a_fp4: fused.combined_weight.ptr(),
            a_scale: fused.combined_block_scale.ptr(),
            a_scale_2: self.tensor_ptr(weights, gate_tensor_scale)?,
            b_fp4: forward.activation_fp4.ptr(),
            b_scale: forward.activation_scale.ptr(),
            b_scale_2: forward.activation_scale_2.ptr(),
            c_bf16: forward.aux.ptr(),
            workspace,
            workspace_bytes,
            alpha,
            scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
        };
        self.backend.nvfp4_gemm(&gemm_spec).map_err(|err| {
            CoreError::Runtime(format!(
                "fused MLP NVFP4 GEMM failed (m={}, n=1, k={}): {err}",
                fused.out_features, quantized.in_features
            ))
        })?;

        let up_offset_bytes = intermediate * 2; // BF16 size
        let up_ptr = forward
            .aux
            .ptr()
            .offset_bytes(up_offset_bytes)
            .ok_or_else(|| {
                CoreError::Runtime("fused MLP up_proj output offset overflow".to_owned())
            })?;

        // Fused SwiGLU + NVFP4 activation quantization: writes the down_proj
        // input directly into `forward.activation_fp4` / `activation_scale`,
        // skipping the BF16 round-trip through `forward.aux3` and the separate
        // quantize launch.
        let LinearWeightBinding::Nvfp4 {
            input_scale: down_input_scale,
            ..
        } = &common.mlp_down_proj
        else {
            return Err(CoreError::Runtime(
                "fused MLP path requires NVFP4 down_proj".to_owned(),
            ));
        };
        let down_input_scale_f32 = self.tensor_scalar_f32(weights, down_input_scale)?;
        self.backend
            .swiglu_nvfp4_quantize(&SwiGluNvfp4QuantizeSpec {
                intermediate,
                gate_bf16: forward.aux.ptr(),
                up_bf16: up_ptr,
                output_fp4: forward.activation_fp4.ptr(),
                output_scale_e4m3: forward.activation_scale.ptr(),
                output_tensor_scale_f32: forward.activation_scale_2.ptr(),
                input_tensor_scale_f32: down_input_scale_f32,
            })?;
        let down_quantized = Nvfp4ActivationQuant {
            in_features: intermediate,
            input_scale: down_input_scale,
        };
        self.linear_with_quantized_nvfp4(
            &common.mlp_down_proj,
            forward.hidden.ptr(),
            down_quantized,
        )
    }

    #[cfg(feature = "cuda")]
    fn run_mlp_fused_combined_gemm_rows(
        &self,
        common: &CommonLayerWeights,
        fused: &MlpFusedLayer,
        prefill: &GpuPrefillBuffers,
        rows: usize,
        pre_quantized: Option<Nvfp4ActivationQuant<'_>>,
    ) -> Result<()> {
        let weights = self.cuda_weights()?;
        let runtime = self.cuda_runtime()?;
        let intermediate = self.topology.intermediate_size;
        let LinearWeightBinding::Nvfp4 {
            weight: gate_weight,
            tensor_scale: gate_tensor_scale,
            input_scale: gate_input_scale,
            ..
        } = &common.mlp_gate_proj
        else {
            return Err(CoreError::Runtime(
                "fused MLP prefill path requires NVFP4 gate_proj".to_owned(),
            ));
        };
        let in_features = Self::nvfp4_in_features(gate_weight)?;
        let quantized = match pre_quantized {
            Some(q) => q,
            None => {
                let q = Nvfp4ActivationQuant {
                    in_features,
                    input_scale: gate_input_scale,
                };
                self.quantize_nvfp4_activation_rows(prefill.normed.ptr(), rows, q, prefill)?;
                q
            }
        };

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
        let alpha = self.tensor_scalar_f32(weights, gate_tensor_scale)?
            * self.tensor_scalar_f32(weights, quantized.input_scale)?;
        self.backend.nvfp4_gemm(&Nvfp4GemmSpec {
            m: fused.out_features,
            n: rows,
            k: quantized.in_features,
            a_fp4: fused.combined_weight.ptr(),
            a_scale: fused.combined_block_scale.ptr(),
            a_scale_2: self.tensor_ptr(weights, gate_tensor_scale)?,
            b_fp4: prefill.activation_fp4.ptr(),
            b_scale: prefill.activation_scale.ptr(),
            b_scale_2: prefill.activation_scale_2.ptr(),
            c_bf16: prefill.block_out.ptr(),
            workspace,
            workspace_bytes,
            alpha,
            scale_mode: CublasLtFp4ScaleMode::Vec16Ue4m3,
        })?;

        self.backend.copy_strided_rows(&CopyStridedRowsSpec {
            rows,
            values: intermediate,
            input_stride: 2 * intermediate,
            output_stride: intermediate,
            input_bf16: prefill.block_out.ptr(),
            output_bf16: prefill.aux.ptr(),
        })?;
        self.backend.copy_strided_rows(&CopyStridedRowsSpec {
            rows,
            values: intermediate,
            input_stride: 2 * intermediate,
            output_stride: intermediate,
            input_bf16: Self::ptr_offset(prefill.block_out.ptr(), intermediate * 2)?,
            output_bf16: prefill.aux2.ptr(),
        })?;
        self.backend.swiglu(&SwiGluSpec {
            rows,
            intermediate,
            gate_bf16: prefill.aux.ptr(),
            up_bf16: prefill.aux2.ptr(),
            output_bf16: prefill.aux3.ptr(),
        })?;
        self.linear_rows(
            &common.mlp_down_proj,
            prefill.aux3.ptr(),
            prefill.hidden.ptr(),
            rows,
            prefill,
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
        let layer_idx = match layer {
            LayerWeights::LinearAttention(layer) => layer.layer_index,
            LayerWeights::FullAttention(layer) => layer.layer_index,
        };
        if cuda_prefill_fused_mlp_enabled()
            && prefill.block_out.bytes() >= tokens * 2 * self.topology.intermediate_size * 2
        {
            if let Some(fused) = self.mlp_fused_main_opt(layer_idx) {
                return self.run_mlp_fused_combined_gemm_rows(common, fused, prefill, tokens, None);
            }
        }
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
        let layer_idx = match layer {
            LayerWeights::LinearAttention(layer) => layer.layer_index,
            LayerWeights::FullAttention(layer) => layer.layer_index,
        };
        if cuda_prefill_fused_mlp_enabled()
            && prefill.block_out.bytes() >= tokens * 2 * self.topology.intermediate_size * 2
        {
            if let Some(fused) = self.mlp_fused_main_opt(layer_idx) {
                return self.run_mlp_fused_combined_gemm_rows(
                    common,
                    fused,
                    prefill,
                    tokens,
                    Some(quantized),
                );
            }
        }
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
    fn attention_kv_cache_dtype_code(dtype: KvCacheDtype) -> Result<i32> {
        match dtype {
            KvCacheDtype::Bf16 => Ok(0),
            KvCacheDtype::Fp8 => Ok(1),
            KvCacheDtype::TurboQuant3 => Ok(2),
            KvCacheDtype::TurboQuant35 => Ok(3),
        }
    }

    /// Pick the split tile size for full-attention decode/prefill. 512 keeps
    /// launch/reduce overhead down at short contexts; 128 starts paying off
    /// around 1K context for MTP short-chunk attention; 64 exposes more
    /// T-axis parallelism once the graph shape reaches ~2K context on the 5090.
    #[cfg(feature = "cuda")]
    fn attention_split_timesteps_per_block(&self) -> usize {
        self.attention_split_timesteps_per_block_for(self.config.max_context)
    }

    #[cfg(feature = "cuda")]
    fn attention_split_timesteps_per_block_for(&self, context: usize) -> usize {
        if let Some(value) = cuda_env_usize("QWEN36_ATTENTION_SPLIT_TIMESTEPS") {
            return value.max(ATTN_MIN_SPLIT_TIMESTEPS_PER_BLOCK);
        }
        if context >= 2048 {
            ATTN_MIN_SPLIT_TIMESTEPS_PER_BLOCK
        } else if context >= 1024 {
            128
        } else {
            512
        }
    }

    /// Pick the number of split-KV blocks per q-head for decode attention.
    /// Sized from `max_context` (not the current position) so the same value
    /// is valid for both fresh kernel calls and CUDA-graph replays where the
    /// position grows after capture. Returns 0 when the configured context
    /// is short enough that the per-q-head kernel is faster (split + reduce
    /// launch overhead would dominate).
    #[cfg(feature = "cuda")]
    fn decode_attention_n_splits(&self) -> usize {
        if cuda_env_bool("QWEN36_ATTENTION_SPLIT_DISABLE") {
            return 0;
        }
        if let Some(value) = cuda_env_usize("QWEN36_DECODE_ATTENTION_N_SPLITS") {
            return value;
        }
        let n_splits = self
            .config
            .max_context
            .div_ceil(self.attention_split_timesteps_per_block());
        if n_splits >= 2 { n_splits } else { 0 }
    }

    #[cfg(feature = "cuda")]
    fn prefill_attention_n_splits(
        &self,
        context: usize,
        start_position_device_i32: DevicePtr,
    ) -> usize {
        if start_position_device_i32 != DevicePtr::NULL {
            return self.decode_attention_n_splits();
        }
        if cuda_env_bool("QWEN36_ATTENTION_SPLIT_DISABLE") {
            return 0;
        }
        if let Some(value) = cuda_env_usize("QWEN36_PREFILL_ATTENTION_N_SPLITS") {
            return value;
        }
        let n_splits = context.div_ceil(self.attention_split_timesteps_per_block_for(context));
        if n_splits >= 2 { n_splits } else { 0 }
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
        if config.mtp_speculative_tokens > MTP_MAX_DRAFT_TOKENS {
            return Err(CoreError::Runtime(format!(
                "MTP runtime currently supports up to {MTP_MAX_DRAFT_TOKENS} speculative tokens"
            )));
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
        if include_mtp
            && cuda_env_bool("QWEN36_MTP_SNAPSHOT_KV")
            && engine
                .state
                .kv_cache
                .layers
                .iter()
                .any(|layer| layer.metadata_bytes != 0 || layer.k_bytes != layer.v_bytes)
        {
            return Err(CoreError::Runtime(
                "MTP KV snapshot is not implemented for TurboQuant KV layout".to_owned(),
            ));
        }
        let gpu_buffers = GpuRuntimeBuffers::allocate(
            &engine.state,
            cuda_env_workspace_bytes(),
            mtp_kv_cache_bytes,
            &engine.topology,
            include_mtp && mtp_recurrent_snapshot_enabled(),
            include_mtp && cuda_env_bool("QWEN36_MTP_SNAPSHOT_KV"),
        )?;
        let gpu_forward = GpuForwardBuffers::allocate(&engine.topology, engine.config.max_context)?;
        let prefill_capacity = cuda_prefill_capacity(engine.config.max_context);
        let fused_mlp_prefill = cuda_prefill_fused_mlp_enabled() && cuda_mlp_fused_enabled();
        let gpu_prefill =
            GpuPrefillBuffers::allocate(&engine.topology, prefill_capacity, fused_mlp_prefill)?;
        let mlp_fused = if cuda_mlp_fused_enabled() || cuda_prefill_fused_mlp_enabled() {
            Some(MlpFusedStore::build(
                &gpu_weights,
                &manifest,
                engine.topology.intermediate_size,
            )?)
        } else {
            None
        };
        let linear_attn_in_proj_fused =
            if cuda_linear_attn_fused_enabled() || cuda_prefill_fused_linear_attn_enabled() {
                Some(LinearAttnInProjFusedStore::build(&gpu_weights, &manifest)?)
            } else {
                None
            };
        engine.weights = Some(manifest);
        engine.gpu_weights = Some(gpu_weights);
        engine.gpu_buffers = Some(gpu_buffers);
        engine.gpu_forward = Some(gpu_forward);
        engine.gpu_prefill = Some(gpu_prefill);
        engine.mlp_fused = mlp_fused;
        engine.linear_attn_in_proj_fused = linear_attn_in_proj_fused;
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

    pub fn gpu_memory_report(&self) -> Option<GpuMemoryReport> {
        fn item(name: &str, bytes: u64) -> GpuMemoryItem {
            GpuMemoryItem {
                name: name.to_owned(),
                bytes,
            }
        }
        fn group(items: Vec<GpuMemoryItem>) -> GpuMemoryGroup {
            let total_bytes = items.iter().map(|item| item.bytes).sum();
            GpuMemoryGroup { total_bytes, items }
        }

        let weights = self.gpu_weights.as_ref()?;
        let runtime = self.gpu_buffers.as_ref()?;
        let forward = self.gpu_forward.as_ref()?;
        let prefill = self.gpu_prefill.as_ref()?;

        let weights_group = group(vec![item("uploaded_model_tensors", weights.total_bytes())]);

        let runtime_group = group(vec![
            item(
                "kv_cache",
                runtime
                    .kv_cache
                    .as_ref()
                    .map(|buffer| buffer.bytes() as u64)
                    .unwrap_or(0),
            ),
            item(
                "mtp_kv_cache",
                runtime
                    .mtp_kv_cache
                    .as_ref()
                    .map(|buffer| buffer.bytes() as u64)
                    .unwrap_or(0),
            ),
            item("deltanet_state", runtime.deltanet_state.bytes() as u64),
            item(
                "deltanet_checkpoint",
                runtime
                    .deltanet_checkpoint
                    .as_ref()
                    .map(|buffer| buffer.bytes() as u64)
                    .unwrap_or(0),
            ),
            item("conv_history", runtime.conv_history.bytes() as u64),
            item(
                "conv_history_checkpoint",
                runtime
                    .conv_history_checkpoint
                    .as_ref()
                    .map(|buffer| buffer.bytes() as u64)
                    .unwrap_or(0),
            ),
            item(
                "mtp_kv_snapshot",
                runtime
                    .mtp_kv_snapshot
                    .as_ref()
                    .map(|buffer| buffer.bytes() as u64)
                    .unwrap_or(0),
            ),
            item(
                "workspace",
                runtime
                    .workspace
                    .as_ref()
                    .map(|buffer| buffer.bytes() as u64)
                    .unwrap_or(0),
            ),
        ]);

        let forward_group = group(vec![
            item("hidden", forward.hidden.bytes() as u64),
            item("residual", forward.residual.bytes() as u64),
            item("normed", forward.normed.bytes() as u64),
            item("block_out", forward.block_out.bytes() as u64),
            item("qkv", forward.qkv.bytes() as u64),
            item("aux", forward.aux.bytes() as u64),
            item("aux2", forward.aux2.bytes() as u64),
            item("aux3", forward.aux3.bytes() as u64),
            item("gate_f32", forward.gate_f32.bytes() as u64),
            item("beta_f32", forward.beta_f32.bytes() as u64),
            item("activation_fp4", forward.activation_fp4.bytes() as u64),
            item("activation_scale", forward.activation_scale.bytes() as u64),
            item(
                "activation_scale_2",
                forward.activation_scale_2.bytes() as u64,
            ),
            item("token_u32", forward.token_u32.bytes() as u64),
            item("position_i32", forward.position_i32.bytes() as u64),
            item("logits", forward.logits.bytes() as u64),
            item("mtp_logits", forward.mtp_logits.bytes() as u64),
            item(
                "sampled_token_u32",
                forward.sampled_token_u32.bytes() as u64,
            ),
            item(
                "mtp_verify_token_u32",
                forward.mtp_verify_token_u32.bytes() as u64,
            ),
            item("attn_partial_acc", forward.attn_partial_acc.bytes() as u64),
            item("attn_partial_max", forward.attn_partial_max.bytes() as u64),
            item(
                "attn_partial_denom",
                forward.attn_partial_denom.bytes() as u64,
            ),
        ]);

        let prefill_group = group(vec![
            item("hidden", prefill.hidden.bytes() as u64),
            item("residual", prefill.residual.bytes() as u64),
            item("normed", prefill.normed.bytes() as u64),
            item("block_out", prefill.block_out.bytes() as u64),
            item("qkv", prefill.qkv.bytes() as u64),
            item("aux", prefill.aux.bytes() as u64),
            item("aux2", prefill.aux2.bytes() as u64),
            item("aux3", prefill.aux3.bytes() as u64),
            item("gate_f32", prefill.gate_f32.bytes() as u64),
            item("beta_f32", prefill.beta_f32.bytes() as u64),
            item("activation_fp4", prefill.activation_fp4.bytes() as u64),
            item("activation_scale", prefill.activation_scale.bytes() as u64),
            item(
                "activation_scale_2",
                prefill.activation_scale_2.bytes() as u64,
            ),
            item("token_u32", prefill.token_u32.bytes() as u64),
            item("position_i32", prefill.position_i32.bytes() as u64),
        ]);

        let fused_group = group(vec![
            item(
                "mlp_fused_store",
                self.mlp_fused
                    .as_ref()
                    .map(|store| store.total_bytes)
                    .unwrap_or(0),
            ),
            item(
                "linear_attn_in_proj_fused_store",
                self.linear_attn_in_proj_fused
                    .as_ref()
                    .map(|store| store.total_bytes)
                    .unwrap_or(0),
            ),
        ]);

        let total_reported_bytes = weights_group.total_bytes
            + runtime_group.total_bytes
            + forward_group.total_bytes
            + prefill_group.total_bytes
            + fused_group.total_bytes;

        Some(GpuMemoryReport {
            total_reported_bytes,
            weights: weights_group,
            runtime: runtime_group,
            forward: forward_group,
            prefill: prefill_group,
            fused: fused_group,
            max_context: self.config.max_context,
            prefill_capacity: prefill.capacity,
            kv_cache_dtype: self.config.kv_cache_dtype,
            mtp_speculative_tokens: self.config.mtp_speculative_tokens,
        })
    }
}
