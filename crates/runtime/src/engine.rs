use serde::{Deserialize, Serialize};

use qwen36_fp4_core::{KvCacheDtype, ModelLayout, ModelTopology, Result};
use qwen36_fp4_kernels::{KernelBackend, NoCudaBackend};

use crate::cuda_graph::CudaGraphPlan;
use crate::kv_cache::KvCachePlan;
use crate::state::{DeltaNetStatePlan, RuntimeState};

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
    backend: B,
}

impl Engine<NoCudaBackend> {
    pub fn no_cuda(layout: &ModelLayout, config: EngineConfig) -> Self {
        Self::new(layout.topology.clone(), config, NoCudaBackend)
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
            backend,
        }
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<ForwardOutput> {
        let _ = prompt_tokens;
        Err(qwen36_fp4_core::CoreError::UnsupportedNoCuda(
            "engine_prefill",
        ))
    }

    pub fn decode_one(&mut self, token: u32) -> Result<ForwardOutput> {
        let _ = token;
        Err(qwen36_fp4_core::CoreError::UnsupportedNoCuda(
            "engine_decode_one",
        ))
    }
}
