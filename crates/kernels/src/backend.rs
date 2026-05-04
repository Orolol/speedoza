use qwen36_fp4_core::{CoreError, Result};
use serde::{Deserialize, Serialize};

use crate::attention::{AttentionDecodeSpec, AttentionPrefillSpec};
use crate::deltanet::{DeltaNetDecodeSpec, DeltaNetPrefillSpec};
use crate::nvfp4_gemm::Nvfp4GemmSpec;
use crate::ops::{
    Bf16GemmSpec, Bf16MatVecSpec, Conv1dGdnGateFusedSpec, Conv1dPrefillSpec, Conv1dUpdateSpec,
    CopyStridedRowsSpec, EmbeddingLookupSpec, GdnGateSpec, Nvfp4MatVecSpec, Nvfp4QuantizeRowsSpec,
    Nvfp4QuantizeSpec, Nvfp4RetileScalesSpec, QProjDeinterleaveSpec, QProjSigmoidGateSpec,
    RmsNormNvfp4QuantizeSpec, SigmoidGateSpec, SigmoidGateStridedSpec,
};
use crate::rmsnorm::RmsNormSpec;
use crate::rope::PartialRopeSpec;
use crate::sampling::{SamplingRowsSpec, SamplingSpec};
use crate::swiglu::{SwiGluNvfp4QuantizeSpec, SwiGluSpec};
use crate::turboquant::{TurboQuantAttentionSpec, TurboQuantEncodeSpec};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct DevicePtr(pub u64);

impl DevicePtr {
    pub const NULL: Self = Self(0);

    pub fn offset_bytes(self, bytes: usize) -> Option<Self> {
        self.0.checked_add(bytes as u64).map(Self)
    }
}

pub trait KernelBackend: Send + Sync {
    fn name(&self) -> &'static str;

    fn nvfp4_gemm(&self, _spec: &Nvfp4GemmSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("nvfp4_gemm"))
    }

    fn deltanet_prefill(&self, _spec: &DeltaNetPrefillSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("deltanet_prefill"))
    }

    fn deltanet_decode(&self, _spec: &DeltaNetDecodeSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("deltanet_decode"))
    }

    fn attention_prefill(&self, _spec: &AttentionPrefillSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("attention_prefill"))
    }

    fn attention_decode(&self, _spec: &AttentionDecodeSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("attention_decode"))
    }

    fn turboquant_encode_kv(&self, _spec: &TurboQuantEncodeSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("turboquant_encode_kv"))
    }

    fn turboquant_attention(&self, _spec: &TurboQuantAttentionSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("turboquant_attention"))
    }

    fn rmsnorm(&self, _spec: &RmsNormSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("rmsnorm"))
    }

    fn rmsnorm_nvfp4_quantize(&self, _spec: &RmsNormNvfp4QuantizeSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("rmsnorm_nvfp4_quantize"))
    }

    fn partial_rope(&self, _spec: &PartialRopeSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("partial_rope"))
    }

    fn swiglu(&self, _spec: &SwiGluSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("swiglu"))
    }

    fn swiglu_nvfp4_quantize(&self, _spec: &SwiGluNvfp4QuantizeSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("swiglu_nvfp4_quantize"))
    }

    fn sample(&self, _spec: &SamplingSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("sample"))
    }

    fn sample_rows(&self, _spec: &SamplingRowsSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("sample_rows"))
    }

    fn embedding_lookup(&self, _spec: &EmbeddingLookupSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("embedding_lookup"))
    }

    fn bf16_matvec(&self, _spec: &Bf16MatVecSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("bf16_matvec"))
    }

    fn bf16_gemm(&self, _spec: &Bf16GemmSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("bf16_gemm"))
    }

    fn nvfp4_matvec(&self, _spec: &Nvfp4MatVecSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("nvfp4_matvec"))
    }

    fn nvfp4_quantize_bf16(&self, _spec: &Nvfp4QuantizeSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("nvfp4_quantize_bf16"))
    }

    fn nvfp4_quantize_rows(&self, _spec: &Nvfp4QuantizeRowsSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("nvfp4_quantize_rows"))
    }

    fn nvfp4_retile_scales(&self, _spec: &Nvfp4RetileScalesSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("nvfp4_retile_scales"))
    }

    fn conv1d_update(&self, _spec: &Conv1dUpdateSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("conv1d_update"))
    }

    fn conv1d_prefill(&self, _spec: &Conv1dPrefillSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("conv1d_prefill"))
    }

    fn gdn_gate(&self, _spec: &GdnGateSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("gdn_gate"))
    }

    fn conv1d_gdn_gate_fused(&self, _spec: &Conv1dGdnGateFusedSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("conv1d_gdn_gate_fused"))
    }

    fn sigmoid_gate(&self, _spec: &SigmoidGateSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("sigmoid_gate"))
    }

    fn sigmoid_gate_strided(&self, _spec: &SigmoidGateStridedSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("sigmoid_gate_strided"))
    }

    fn q_proj_deinterleave(&self, _spec: &QProjDeinterleaveSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("q_proj_deinterleave"))
    }

    fn q_proj_sigmoid_gate(&self, _spec: &QProjSigmoidGateSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("q_proj_sigmoid_gate"))
    }

    fn copy_strided_rows(&self, _spec: &CopyStridedRowsSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("copy_strided_rows"))
    }
}

#[derive(Debug, Default)]
pub struct NoCudaBackend;

impl KernelBackend for NoCudaBackend {
    fn name(&self) -> &'static str {
        "no-cuda"
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Default)]
pub struct CudaBackend;

#[cfg(feature = "cuda")]
impl KernelBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn nvfp4_gemm(&self, spec: &Nvfp4GemmSpec) -> Result<()> {
        let ffi_spec = ffi::Nvfp4GemmSpec::from(spec);
        // When the Mirage megakernel path is enabled (env var) we try the
        // CUTLASS-templated NVFP4 GEMM first. The kernel is being built up
        // shape-by-shape (`docs/mirage-megakernel.md`); on any unsupported
        // shape it returns QWEN36_STATUS_NOT_IMPLEMENTED, in which case we
        // transparently fall back to the cuBLASLt path. This keeps the
        // engine perf-neutral while individual shapes get migrated.
        if megakernel_enabled() {
            let code = unsafe { ffi::qwen36_megakernel_nvfp4_gemm(&ffi_spec) };
            // 5 == QWEN36_STATUS_NOT_IMPLEMENTED. Any other non-zero is a
            // real failure and surfaces through `check()` like the rest of
            // the FFI surface.
            if code != 5 {
                return check("qwen36_megakernel_nvfp4_gemm", code);
            }
        }
        check("qwen36_nvfp4_gemm", unsafe {
            ffi::qwen36_nvfp4_gemm(&ffi_spec)
        })
    }

    fn deltanet_decode(&self, spec: &DeltaNetDecodeSpec) -> Result<()> {
        let ffi_spec = ffi::DeltaNetDecodeSpec::from(spec);
        check("qwen36_deltanet_decode", unsafe {
            ffi::qwen36_deltanet_decode(&ffi_spec)
        })
    }

    fn attention_prefill(&self, spec: &AttentionPrefillSpec) -> Result<()> {
        let ffi_spec = ffi::AttentionPrefillSpec::from(spec);
        check("qwen36_attention_prefill", unsafe {
            ffi::qwen36_attention_prefill(&ffi_spec)
        })
    }

    fn attention_decode(&self, spec: &AttentionDecodeSpec) -> Result<()> {
        let ffi_spec = ffi::AttentionDecodeSpec::from(spec);
        check("qwen36_attention_decode", unsafe {
            ffi::qwen36_attention_decode(&ffi_spec)
        })
    }

    fn turboquant_encode_kv(&self, spec: &TurboQuantEncodeSpec) -> Result<()> {
        let ffi_spec = ffi::TurboQuantEncodeSpec::from(spec);
        check("qwen36_turboquant_encode_kv", unsafe {
            ffi::qwen36_turboquant_encode_kv(&ffi_spec)
        })
    }

    fn turboquant_attention(&self, spec: &TurboQuantAttentionSpec) -> Result<()> {
        let ffi_spec = ffi::TurboQuantAttentionSpec::from(spec);
        check("qwen36_turboquant_attention", unsafe {
            ffi::qwen36_turboquant_attention(&ffi_spec)
        })
    }

    fn rmsnorm(&self, spec: &RmsNormSpec) -> Result<()> {
        let ffi_spec = ffi::RmsNormSpec::from(spec);
        check("qwen36_rmsnorm", unsafe { ffi::qwen36_rmsnorm(&ffi_spec) })
    }

    fn rmsnorm_nvfp4_quantize(&self, spec: &RmsNormNvfp4QuantizeSpec) -> Result<()> {
        let ffi_spec = ffi::RmsNormNvfp4QuantizeSpec::from(spec);
        check("qwen36_rmsnorm_nvfp4_quantize", unsafe {
            ffi::qwen36_rmsnorm_nvfp4_quantize(&ffi_spec)
        })
    }

    fn partial_rope(&self, spec: &PartialRopeSpec) -> Result<()> {
        let ffi_spec = ffi::PartialRopeSpec::from(spec);
        check("qwen36_partial_rope", unsafe {
            ffi::qwen36_partial_rope(&ffi_spec)
        })
    }

    fn swiglu(&self, spec: &SwiGluSpec) -> Result<()> {
        let ffi_spec = ffi::SwiGluSpec::from(spec);
        check("qwen36_swiglu", unsafe { ffi::qwen36_swiglu(&ffi_spec) })
    }

    fn swiglu_nvfp4_quantize(&self, spec: &SwiGluNvfp4QuantizeSpec) -> Result<()> {
        let ffi_spec = ffi::SwiGluNvfp4QuantizeSpec::from(spec);
        check("qwen36_swiglu_nvfp4_quantize", unsafe {
            ffi::qwen36_swiglu_nvfp4_quantize(&ffi_spec)
        })
    }

    fn sample(&self, spec: &SamplingSpec) -> Result<()> {
        let ffi_spec = ffi::SamplingSpec::from(spec);
        check("qwen36_sample", unsafe { ffi::qwen36_sample(&ffi_spec) })
    }

    fn sample_rows(&self, spec: &SamplingRowsSpec) -> Result<()> {
        let ffi_spec = ffi::SamplingRowsSpec::from(spec);
        check("qwen36_sample_rows", unsafe {
            ffi::qwen36_sample_rows(&ffi_spec)
        })
    }

    fn embedding_lookup(&self, spec: &EmbeddingLookupSpec) -> Result<()> {
        let ffi_spec = ffi::EmbeddingLookupSpec::from(spec);
        check("qwen36_embedding_lookup", unsafe {
            ffi::qwen36_embedding_lookup(&ffi_spec)
        })
    }

    fn bf16_matvec(&self, spec: &Bf16MatVecSpec) -> Result<()> {
        let ffi_spec = ffi::Bf16MatVecSpec::from(spec);
        check("qwen36_bf16_matvec", unsafe {
            ffi::qwen36_bf16_matvec(&ffi_spec)
        })
    }

    fn bf16_gemm(&self, spec: &Bf16GemmSpec) -> Result<()> {
        let ffi_spec = ffi::Bf16GemmSpec::from(spec);
        check("qwen36_bf16_gemm", unsafe {
            ffi::qwen36_bf16_gemm(&ffi_spec)
        })
    }

    fn nvfp4_matvec(&self, spec: &Nvfp4MatVecSpec) -> Result<()> {
        let ffi_spec = ffi::Nvfp4MatVecSpec::from(spec);
        check("qwen36_nvfp4_matvec", unsafe {
            ffi::qwen36_nvfp4_matvec(&ffi_spec)
        })
    }

    fn nvfp4_quantize_bf16(&self, spec: &Nvfp4QuantizeSpec) -> Result<()> {
        let ffi_spec = ffi::Nvfp4QuantizeSpec::from(spec);
        check("qwen36_nvfp4_quantize_bf16", unsafe {
            ffi::qwen36_nvfp4_quantize_bf16(&ffi_spec)
        })
    }

    fn nvfp4_quantize_rows(&self, spec: &Nvfp4QuantizeRowsSpec) -> Result<()> {
        let ffi_spec = ffi::Nvfp4QuantizeRowsSpec::from(spec);
        check("qwen36_nvfp4_quantize_rows", unsafe {
            ffi::qwen36_nvfp4_quantize_rows(&ffi_spec)
        })
    }

    fn nvfp4_retile_scales(&self, spec: &Nvfp4RetileScalesSpec) -> Result<()> {
        nvfp4_retile_scales(spec)
    }

    fn conv1d_update(&self, spec: &Conv1dUpdateSpec) -> Result<()> {
        let ffi_spec = ffi::Conv1dUpdateSpec::from(spec);
        check("qwen36_conv1d_update", unsafe {
            ffi::qwen36_conv1d_update(&ffi_spec)
        })
    }

    fn conv1d_prefill(&self, spec: &Conv1dPrefillSpec) -> Result<()> {
        let ffi_spec = ffi::Conv1dPrefillSpec::from(spec);
        check("qwen36_conv1d_prefill", unsafe {
            ffi::qwen36_conv1d_prefill(&ffi_spec)
        })
    }

    fn gdn_gate(&self, spec: &GdnGateSpec) -> Result<()> {
        let ffi_spec = ffi::GdnGateSpec::from(spec);
        check("qwen36_gdn_gate", unsafe {
            ffi::qwen36_gdn_gate(&ffi_spec)
        })
    }

    fn conv1d_gdn_gate_fused(&self, spec: &Conv1dGdnGateFusedSpec) -> Result<()> {
        let ffi_spec = ffi::Conv1dGdnGateFusedSpec::from(spec);
        check("qwen36_conv1d_gdn_gate_fused", unsafe {
            ffi::qwen36_conv1d_gdn_gate_fused(&ffi_spec)
        })
    }

    fn sigmoid_gate(&self, spec: &SigmoidGateSpec) -> Result<()> {
        let ffi_spec = ffi::SigmoidGateSpec::from(spec);
        check("qwen36_sigmoid_gate", unsafe {
            ffi::qwen36_sigmoid_gate(&ffi_spec)
        })
    }

    fn sigmoid_gate_strided(&self, spec: &SigmoidGateStridedSpec) -> Result<()> {
        let ffi_spec = ffi::SigmoidGateStridedSpec::from(spec);
        check("qwen36_sigmoid_gate_strided", unsafe {
            ffi::qwen36_sigmoid_gate_strided(&ffi_spec)
        })
    }

    fn q_proj_deinterleave(&self, spec: &QProjDeinterleaveSpec) -> Result<()> {
        let ffi_spec = ffi::QProjDeinterleaveSpec::from(spec);
        check("qwen36_q_proj_deinterleave", unsafe {
            ffi::qwen36_q_proj_deinterleave(&ffi_spec)
        })
    }

    fn q_proj_sigmoid_gate(&self, spec: &QProjSigmoidGateSpec) -> Result<()> {
        let ffi_spec = ffi::QProjSigmoidGateSpec::from(spec);
        check("qwen36_q_proj_sigmoid_gate", unsafe {
            ffi::qwen36_q_proj_sigmoid_gate(&ffi_spec)
        })
    }

    fn copy_strided_rows(&self, spec: &CopyStridedRowsSpec) -> Result<()> {
        let ffi_spec = ffi::CopyStridedRowsSpec::from(spec);
        check("qwen36_copy_strided_rows", unsafe {
            ffi::qwen36_copy_strided_rows(&ffi_spec)
        })
    }
}

#[cfg(feature = "cuda")]
pub fn nvfp4_retile_scales(spec: &Nvfp4RetileScalesSpec) -> Result<()> {
    let ffi_spec = ffi::Nvfp4RetileScalesSpec::from(spec);
    check("qwen36_nvfp4_retile_scales", unsafe {
        ffi::qwen36_nvfp4_retile_scales(&ffi_spec)
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn check(kernel: &'static str, code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(CoreError::KernelLaunch { kernel, code })
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn topk_argmax_raw(
    vocab_size: usize,
    k: usize,
    logits_bf16: DevicePtr,
    output_token_u32: DevicePtr,
) -> i32 {
    let spec = ffi::TopkArgmaxSpec {
        vocab_size,
        k,
        logits_bf16,
        output_token_u32,
    };
    unsafe { ffi::qwen36_topk_argmax(&spec) }
}

/// Cached env-var lookup gating the CUTLASS-templated NVFP4 GEMM path.
/// Set `QWEN36_USE_MEGAKERNEL_GEMM=1` to opt in; the default (unset / 0)
/// keeps the cuBLASLt path active. Cached so the dispatch hot path does
/// not parse the environment on every GEMM call.
#[cfg(feature = "cuda")]
fn megakernel_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN36_USE_MEGAKERNEL_GEMM").ok().as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    })
}

#[cfg(feature = "cuda")]
mod ffi {
    use crate::attention::AttentionShape as RustAttentionShape;
    use crate::deltanet::DeltaNetShape as RustDeltaNetShape;
    use crate::nvfp4_gemm::Nvfp4GemmSpec as RustNvfp4GemmSpec;

    use super::DevicePtr;

    #[repr(C)]
    pub struct Nvfp4GemmSpec {
        pub m: usize,
        pub n: usize,
        pub k: usize,
        pub a_fp4: DevicePtr,
        pub a_scale: DevicePtr,
        pub a_scale_2: DevicePtr,
        pub b_fp4: DevicePtr,
        pub b_scale: DevicePtr,
        pub b_scale_2: DevicePtr,
        pub c_bf16: DevicePtr,
        pub workspace: DevicePtr,
        pub workspace_bytes: usize,
        pub alpha: f32,
    }

    impl From<&RustNvfp4GemmSpec> for Nvfp4GemmSpec {
        fn from(value: &RustNvfp4GemmSpec) -> Self {
            Self {
                m: value.m,
                n: value.n,
                k: value.k,
                a_fp4: value.a_fp4,
                a_scale: value.a_scale,
                a_scale_2: value.a_scale_2,
                b_fp4: value.b_fp4,
                b_scale: value.b_scale,
                b_scale_2: value.b_scale_2,
                c_bf16: value.c_bf16,
                workspace: value.workspace,
                workspace_bytes: value.workspace_bytes,
                alpha: value.alpha,
            }
        }
    }

    #[repr(C)]
    pub struct AttentionPrefillSpec {
        pub layer_index: usize,
        pub start_position: usize,
        pub tokens: usize,
        pub q_bf16: DevicePtr,
        pub k_bf16: DevicePtr,
        pub v_bf16: DevicePtr,
        pub kv_cache_k: DevicePtr,
        pub kv_cache_v: DevicePtr,
        pub output_bf16: DevicePtr,
        pub shape: AttentionShape,
        pub kv_cache_dtype: i32,
        pub start_position_device_i32: DevicePtr,
        pub partial_acc_f32: DevicePtr,
        pub partial_max_f32: DevicePtr,
        pub partial_denom_f32: DevicePtr,
        pub prefill_n_splits: usize,
        pub split_timesteps_per_block: usize,
        pub tree_ancestor_bitmap_u64: DevicePtr,
        pub verify_chunk_rows: usize,
    }

    impl From<&crate::attention::AttentionPrefillSpec> for AttentionPrefillSpec {
        fn from(value: &crate::attention::AttentionPrefillSpec) -> Self {
            Self {
                layer_index: value.layer_index,
                start_position: value.start_position,
                tokens: value.tokens,
                q_bf16: value.q_bf16,
                k_bf16: value.k_bf16,
                v_bf16: value.v_bf16,
                kv_cache_k: value.kv_cache_k,
                kv_cache_v: value.kv_cache_v,
                output_bf16: value.output_bf16,
                shape: AttentionShape::from(value.shape),
                kv_cache_dtype: value.kv_cache_dtype,
                start_position_device_i32: value.start_position_device_i32,
                partial_acc_f32: value.partial_acc_f32,
                partial_max_f32: value.partial_max_f32,
                partial_denom_f32: value.partial_denom_f32,
                prefill_n_splits: value.prefill_n_splits,
                split_timesteps_per_block: value.split_timesteps_per_block,
                tree_ancestor_bitmap_u64: value.tree_ancestor_bitmap_u64,
                verify_chunk_rows: value.verify_chunk_rows,
            }
        }
    }

    #[repr(C)]
    pub struct AttentionDecodeSpec {
        pub layer_index: usize,
        pub position: usize,
        pub q_bf16: DevicePtr,
        pub k_bf16: DevicePtr,
        pub v_bf16: DevicePtr,
        pub kv_cache_k: DevicePtr,
        pub kv_cache_v: DevicePtr,
        pub output_bf16: DevicePtr,
        pub shape: AttentionShape,
        pub kv_cache_dtype: i32,
        pub position_device_i32: DevicePtr,
        pub partial_acc_f32: DevicePtr,
        pub partial_max_f32: DevicePtr,
        pub partial_denom_f32: DevicePtr,
        pub decode_n_splits: usize,
        pub split_timesteps_per_block: usize,
    }

    impl From<&crate::attention::AttentionDecodeSpec> for AttentionDecodeSpec {
        fn from(value: &crate::attention::AttentionDecodeSpec) -> Self {
            Self {
                layer_index: value.layer_index,
                position: value.position,
                q_bf16: value.q_bf16,
                k_bf16: value.k_bf16,
                v_bf16: value.v_bf16,
                kv_cache_k: value.kv_cache_k,
                kv_cache_v: value.kv_cache_v,
                output_bf16: value.output_bf16,
                shape: AttentionShape::from(value.shape),
                kv_cache_dtype: value.kv_cache_dtype,
                position_device_i32: value.position_device_i32,
                partial_acc_f32: value.partial_acc_f32,
                partial_max_f32: value.partial_max_f32,
                partial_denom_f32: value.partial_denom_f32,
                decode_n_splits: value.decode_n_splits,
                split_timesteps_per_block: value.split_timesteps_per_block,
            }
        }
    }

    #[repr(C)]
    pub struct DeltaNetDecodeSpec {
        pub layer_index: usize,
        pub tokens_in_persistent_loop: usize,
        pub q_token_stride: usize,
        pub k_token_stride: usize,
        pub v_token_stride: usize,
        pub q_bf16: DevicePtr,
        pub k_bf16: DevicePtr,
        pub v_bf16: DevicePtr,
        pub state_bf16: DevicePtr,
        pub conv_history_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
        pub gate_f32: DevicePtr,
        pub beta_f32: DevicePtr,
        pub shape: DeltaNetShape,
        pub state_decay: f32,
        pub update_scale: f32,
        pub qk_l2norm: i32,
    }

    impl From<&crate::deltanet::DeltaNetDecodeSpec> for DeltaNetDecodeSpec {
        fn from(value: &crate::deltanet::DeltaNetDecodeSpec) -> Self {
            Self {
                layer_index: value.layer_index,
                tokens_in_persistent_loop: value.tokens_in_persistent_loop,
                q_token_stride: value.q_token_stride,
                k_token_stride: value.k_token_stride,
                v_token_stride: value.v_token_stride,
                q_bf16: value.q_bf16,
                k_bf16: value.k_bf16,
                v_bf16: value.v_bf16,
                state_bf16: value.state_bf16,
                conv_history_bf16: value.conv_history_bf16,
                output_bf16: value.output_bf16,
                gate_f32: value.gate_f32,
                beta_f32: value.beta_f32,
                shape: DeltaNetShape::from(value.shape),
                state_decay: value.state_decay,
                update_scale: value.update_scale,
                qk_l2norm: i32::from(value.qk_l2norm),
            }
        }
    }

    #[repr(C)]
    pub struct TurboQuantEncodeSpec {
        pub layer_index: usize,
        pub position: usize,
        pub k_bf16: DevicePtr,
        pub v_bf16: DevicePtr,
        pub k_quantized_i8: DevicePtr,
        pub v_quantized_i8: DevicePtr,
        pub metadata_f32: DevicePtr,
        pub shape: AttentionShape,
    }

    impl From<&crate::turboquant::TurboQuantEncodeSpec> for TurboQuantEncodeSpec {
        fn from(value: &crate::turboquant::TurboQuantEncodeSpec) -> Self {
            Self {
                layer_index: value.layer_index,
                position: value.position,
                k_bf16: value.k_bf16,
                v_bf16: value.v_bf16,
                k_quantized_i8: value.k_quantized,
                v_quantized_i8: value.v_quantized,
                metadata_f32: value.metadata,
                shape: AttentionShape::from(value.shape),
            }
        }
    }

    #[repr(C)]
    pub struct TurboQuantAttentionSpec {
        pub layer_index: usize,
        pub position: usize,
        pub q_bf16: DevicePtr,
        pub k_quantized_i8: DevicePtr,
        pub v_quantized_i8: DevicePtr,
        pub metadata_f32: DevicePtr,
        pub output_bf16: DevicePtr,
        pub workspace: DevicePtr,
        pub workspace_bytes: usize,
        pub shape: AttentionShape,
        pub mode: i32,
    }

    impl From<&crate::turboquant::TurboQuantAttentionSpec> for TurboQuantAttentionSpec {
        fn from(value: &crate::turboquant::TurboQuantAttentionSpec) -> Self {
            Self {
                layer_index: value.layer_index,
                position: value.position,
                q_bf16: value.q_bf16,
                k_quantized_i8: value.k_quantized,
                v_quantized_i8: value.v_quantized,
                metadata_f32: value.metadata,
                output_bf16: value.output_bf16,
                workspace: value.workspace,
                workspace_bytes: value.workspace_bytes,
                shape: AttentionShape::from(value.shape),
                mode: value.mode as i32,
            }
        }
    }

    #[repr(C)]
    pub struct RmsNormSpec {
        pub rows: usize,
        pub hidden: usize,
        pub eps: f32,
        pub input_bf16: DevicePtr,
        pub weight_bf16: DevicePtr,
        pub residual_bf16: DevicePtr,
        pub residual_out_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
        pub direct_weight: i32,
    }

    impl From<&crate::rmsnorm::RmsNormSpec> for RmsNormSpec {
        fn from(value: &crate::rmsnorm::RmsNormSpec) -> Self {
            Self {
                rows: value.rows,
                hidden: value.hidden,
                eps: value.eps,
                input_bf16: value.input_bf16,
                weight_bf16: value.weight_bf16,
                residual_bf16: value.residual_bf16,
                residual_out_bf16: value.residual_out_bf16,
                output_bf16: value.output_bf16,
                direct_weight: i32::from(value.direct_weight),
            }
        }
    }

    #[repr(C)]
    pub struct RmsNormNvfp4QuantizeSpec {
        pub hidden: usize,
        pub eps: f32,
        pub input_bf16: DevicePtr,
        pub weight_bf16: DevicePtr,
        pub residual_bf16: DevicePtr,
        pub residual_out_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
        pub output_fp4: DevicePtr,
        pub output_scale_e4m3: DevicePtr,
        pub output_tensor_scale_f32: DevicePtr,
        pub input_tensor_scale_f32: f32,
    }

    impl From<&crate::ops::RmsNormNvfp4QuantizeSpec> for RmsNormNvfp4QuantizeSpec {
        fn from(value: &crate::ops::RmsNormNvfp4QuantizeSpec) -> Self {
            Self {
                hidden: value.hidden,
                eps: value.eps,
                input_bf16: value.input_bf16,
                weight_bf16: value.weight_bf16,
                residual_bf16: value.residual_bf16,
                residual_out_bf16: value.residual_out_bf16,
                output_bf16: value.output_bf16,
                output_fp4: value.output_fp4,
                output_scale_e4m3: value.output_scale_e4m3,
                output_tensor_scale_f32: value.output_tensor_scale_f32,
                input_tensor_scale_f32: value.input_tensor_scale_f32,
            }
        }
    }

    #[repr(C)]
    pub struct PartialRopeSpec {
        pub tokens: usize,
        pub q_heads: usize,
        pub kv_heads: usize,
        pub head_dim: usize,
        pub rope_dims: usize,
        pub base_theta: f64,
        pub position_i32: i32,
        pub use_scalar_position: i32,
        pub positions_i32: DevicePtr,
        pub q_bf16: DevicePtr,
        pub k_bf16: DevicePtr,
        pub scalar_position_device_i32: DevicePtr,
    }

    impl From<&crate::rope::PartialRopeSpec> for PartialRopeSpec {
        fn from(value: &crate::rope::PartialRopeSpec) -> Self {
            Self {
                tokens: value.tokens,
                q_heads: value.q_heads,
                kv_heads: value.kv_heads,
                head_dim: value.head_dim,
                rope_dims: value.rope_dims,
                base_theta: value.base_theta,
                position_i32: value.position_i32,
                use_scalar_position: i32::from(value.use_scalar_position),
                positions_i32: value.positions_i32,
                q_bf16: value.q_bf16,
                k_bf16: value.k_bf16,
                scalar_position_device_i32: value.scalar_position_device_i32,
            }
        }
    }

    #[repr(C)]
    pub struct SwiGluSpec {
        pub rows: usize,
        pub intermediate: usize,
        pub gate_bf16: DevicePtr,
        pub up_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
    }

    impl From<&crate::swiglu::SwiGluSpec> for SwiGluSpec {
        fn from(value: &crate::swiglu::SwiGluSpec) -> Self {
            Self {
                rows: value.rows,
                intermediate: value.intermediate,
                gate_bf16: value.gate_bf16,
                up_bf16: value.up_bf16,
                output_bf16: value.output_bf16,
            }
        }
    }

    #[repr(C)]
    pub struct SwiGluNvfp4QuantizeSpec {
        pub intermediate: usize,
        pub gate_bf16: DevicePtr,
        pub up_bf16: DevicePtr,
        pub output_fp4: DevicePtr,
        pub output_scale_e4m3: DevicePtr,
        pub output_tensor_scale_f32: DevicePtr,
        pub input_tensor_scale_f32: f32,
    }

    impl From<&crate::swiglu::SwiGluNvfp4QuantizeSpec> for SwiGluNvfp4QuantizeSpec {
        fn from(value: &crate::swiglu::SwiGluNvfp4QuantizeSpec) -> Self {
            Self {
                intermediate: value.intermediate,
                gate_bf16: value.gate_bf16,
                up_bf16: value.up_bf16,
                output_fp4: value.output_fp4,
                output_scale_e4m3: value.output_scale_e4m3,
                output_tensor_scale_f32: value.output_tensor_scale_f32,
                input_tensor_scale_f32: value.input_tensor_scale_f32,
            }
        }
    }

    #[repr(C)]
    pub struct SamplingSpec {
        pub vocab_size: usize,
        pub logits_bf16: DevicePtr,
        pub output_token_u32: DevicePtr,
        pub temperature: f32,
        pub top_k: usize,
        pub top_p: f32,
        pub repetition_penalty: f32,
        pub mirror_output_token_u32: DevicePtr,
    }

    impl From<&crate::sampling::SamplingSpec> for SamplingSpec {
        fn from(value: &crate::sampling::SamplingSpec) -> Self {
            Self {
                vocab_size: value.vocab_size,
                logits_bf16: value.logits_bf16,
                output_token_u32: value.output_token_u32,
                temperature: value.temperature,
                top_k: value.top_k,
                top_p: value.top_p,
                repetition_penalty: value.repetition_penalty,
                mirror_output_token_u32: value.mirror_output_token_u32,
            }
        }
    }

    #[repr(C)]
    pub struct SamplingRowsSpec {
        pub rows: usize,
        pub vocab_size: usize,
        pub logits_bf16: DevicePtr,
        pub output_token_u32: DevicePtr,
        pub mirror_last_output_token_u32: DevicePtr,
        pub temperature: f32,
    }

    impl From<&crate::sampling::SamplingRowsSpec> for SamplingRowsSpec {
        fn from(value: &crate::sampling::SamplingRowsSpec) -> Self {
            Self {
                rows: value.rows,
                vocab_size: value.vocab_size,
                logits_bf16: value.logits_bf16,
                output_token_u32: value.output_token_u32,
                mirror_last_output_token_u32: value.mirror_last_output_token_u32,
                temperature: value.temperature,
            }
        }
    }

    #[repr(C)]
    pub struct TopkArgmaxSpec {
        pub vocab_size: usize,
        pub k: usize,
        pub logits_bf16: DevicePtr,
        pub output_token_u32: DevicePtr,
    }

    #[repr(C)]
    pub struct EmbeddingLookupSpec {
        pub tokens: usize,
        pub hidden: usize,
        pub vocab_size: usize,
        pub token_ids_u32: DevicePtr,
        pub embedding_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
    }

    impl From<&crate::ops::EmbeddingLookupSpec> for EmbeddingLookupSpec {
        fn from(value: &crate::ops::EmbeddingLookupSpec) -> Self {
            Self {
                tokens: value.tokens,
                hidden: value.hidden,
                vocab_size: value.vocab_size,
                token_ids_u32: value.token_ids_u32,
                embedding_bf16: value.embedding_bf16,
                output_bf16: value.output_bf16,
            }
        }
    }

    #[repr(C)]
    pub struct Bf16MatVecSpec {
        pub out_features: usize,
        pub in_features: usize,
        pub input_bf16: DevicePtr,
        pub weight_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
    }

    impl From<&crate::ops::Bf16MatVecSpec> for Bf16MatVecSpec {
        fn from(value: &crate::ops::Bf16MatVecSpec) -> Self {
            Self {
                out_features: value.out_features,
                in_features: value.in_features,
                input_bf16: value.input_bf16,
                weight_bf16: value.weight_bf16,
                output_bf16: value.output_bf16,
            }
        }
    }

    #[repr(C)]
    pub struct Bf16GemmSpec {
        pub m: usize,
        pub n: usize,
        pub k: usize,
        pub a_bf16: DevicePtr,
        pub b_bf16: DevicePtr,
        pub c_bf16: DevicePtr,
        pub workspace: DevicePtr,
        pub workspace_bytes: usize,
    }

    impl From<&crate::ops::Bf16GemmSpec> for Bf16GemmSpec {
        fn from(value: &crate::ops::Bf16GemmSpec) -> Self {
            Self {
                m: value.m,
                n: value.n,
                k: value.k,
                a_bf16: value.a_bf16,
                b_bf16: value.b_bf16,
                c_bf16: value.c_bf16,
                workspace: value.workspace,
                workspace_bytes: value.workspace_bytes,
            }
        }
    }

    #[repr(C)]
    pub struct Nvfp4MatVecSpec {
        pub out_features: usize,
        pub in_features: usize,
        pub input_bf16: DevicePtr,
        pub weight_u8: DevicePtr,
        pub block_scale_e4m3: DevicePtr,
        pub tensor_scale_f32: DevicePtr,
        pub output_bf16: DevicePtr,
    }

    impl From<&crate::ops::Nvfp4MatVecSpec> for Nvfp4MatVecSpec {
        fn from(value: &crate::ops::Nvfp4MatVecSpec) -> Self {
            Self {
                out_features: value.out_features,
                in_features: value.in_features,
                input_bf16: value.input_bf16,
                weight_u8: value.weight_u8,
                block_scale_e4m3: value.block_scale_e4m3,
                tensor_scale_f32: value.tensor_scale_f32,
                output_bf16: value.output_bf16,
            }
        }
    }

    #[repr(C)]
    pub struct Nvfp4QuantizeSpec {
        pub values: usize,
        pub input_bf16: DevicePtr,
        pub output_fp4: DevicePtr,
        pub output_scale_e4m3: DevicePtr,
        pub output_tensor_scale_f32: DevicePtr,
        pub input_tensor_scale_f32: f32,
    }

    impl From<&crate::ops::Nvfp4QuantizeSpec> for Nvfp4QuantizeSpec {
        fn from(value: &crate::ops::Nvfp4QuantizeSpec) -> Self {
            Self {
                values: value.values,
                input_bf16: value.input_bf16,
                output_fp4: value.output_fp4,
                output_scale_e4m3: value.output_scale_e4m3,
                output_tensor_scale_f32: value.output_tensor_scale_f32,
                input_tensor_scale_f32: value.input_tensor_scale_f32,
            }
        }
    }

    #[repr(C)]
    pub struct Nvfp4QuantizeRowsSpec {
        pub rows: usize,
        pub values: usize,
        pub input_bf16: DevicePtr,
        pub output_fp4: DevicePtr,
        pub output_scale_e4m3: DevicePtr,
        pub output_tensor_scale_f32: DevicePtr,
        pub input_tensor_scale_f32: f32,
    }

    impl From<&crate::ops::Nvfp4QuantizeRowsSpec> for Nvfp4QuantizeRowsSpec {
        fn from(value: &crate::ops::Nvfp4QuantizeRowsSpec) -> Self {
            Self {
                rows: value.rows,
                values: value.values,
                input_bf16: value.input_bf16,
                output_fp4: value.output_fp4,
                output_scale_e4m3: value.output_scale_e4m3,
                output_tensor_scale_f32: value.output_tensor_scale_f32,
                input_tensor_scale_f32: value.input_tensor_scale_f32,
            }
        }
    }

    #[repr(C)]
    pub struct Nvfp4RetileScalesSpec {
        pub rows: usize,
        pub inner_groups: usize,
        pub input_row_major_u8: DevicePtr,
        pub output_tiled_u8: DevicePtr,
    }

    impl From<&crate::ops::Nvfp4RetileScalesSpec> for Nvfp4RetileScalesSpec {
        fn from(value: &crate::ops::Nvfp4RetileScalesSpec) -> Self {
            Self {
                rows: value.rows,
                inner_groups: value.inner_groups,
                input_row_major_u8: value.input_row_major_u8,
                output_tiled_u8: value.output_tiled_u8,
            }
        }
    }

    #[repr(C)]
    pub struct Conv1dUpdateSpec {
        pub channels: usize,
        pub kernel_size: usize,
        pub input_bf16: DevicePtr,
        pub conv_history_bf16: DevicePtr,
        pub weight_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
    }

    impl From<&crate::ops::Conv1dUpdateSpec> for Conv1dUpdateSpec {
        fn from(value: &crate::ops::Conv1dUpdateSpec) -> Self {
            Self {
                channels: value.channels,
                kernel_size: value.kernel_size,
                input_bf16: value.input_bf16,
                conv_history_bf16: value.conv_history_bf16,
                weight_bf16: value.weight_bf16,
                output_bf16: value.output_bf16,
            }
        }
    }

    #[repr(C)]
    pub struct Conv1dPrefillSpec {
        pub tokens: usize,
        pub channels: usize,
        pub kernel_size: usize,
        pub input_bf16: DevicePtr,
        pub conv_history_bf16: DevicePtr,
        pub weight_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
    }

    impl From<&crate::ops::Conv1dPrefillSpec> for Conv1dPrefillSpec {
        fn from(value: &crate::ops::Conv1dPrefillSpec) -> Self {
            Self {
                tokens: value.tokens,
                channels: value.channels,
                kernel_size: value.kernel_size,
                input_bf16: value.input_bf16,
                conv_history_bf16: value.conv_history_bf16,
                weight_bf16: value.weight_bf16,
                output_bf16: value.output_bf16,
            }
        }
    }

    #[repr(C)]
    pub struct GdnGateSpec {
        pub rows: usize,
        pub heads: usize,
        pub a_bf16: DevicePtr,
        pub b_bf16: DevicePtr,
        pub a_log_bf16: DevicePtr,
        pub dt_bias_bf16: DevicePtr,
        pub gate_f32: DevicePtr,
        pub beta_f32: DevicePtr,
    }

    impl From<&crate::ops::GdnGateSpec> for GdnGateSpec {
        fn from(value: &crate::ops::GdnGateSpec) -> Self {
            Self {
                rows: value.rows,
                heads: value.heads,
                a_bf16: value.a_bf16,
                b_bf16: value.b_bf16,
                a_log_bf16: value.a_log_bf16,
                dt_bias_bf16: value.dt_bias_bf16,
                gate_f32: value.gate_f32,
                beta_f32: value.beta_f32,
            }
        }
    }

    #[repr(C)]
    pub struct Conv1dGdnGateFusedSpec {
        pub channels: usize,
        pub kernel_size: usize,
        pub conv_input_bf16: DevicePtr,
        pub conv_history_bf16: DevicePtr,
        pub conv_weight_bf16: DevicePtr,
        pub conv_output_bf16: DevicePtr,
        pub heads: usize,
        pub gdn_a_bf16: DevicePtr,
        pub gdn_b_bf16: DevicePtr,
        pub gdn_a_log_bf16: DevicePtr,
        pub gdn_dt_bias_bf16: DevicePtr,
        pub gate_f32: DevicePtr,
        pub beta_f32: DevicePtr,
    }

    impl From<&crate::ops::Conv1dGdnGateFusedSpec> for Conv1dGdnGateFusedSpec {
        fn from(value: &crate::ops::Conv1dGdnGateFusedSpec) -> Self {
            Self {
                channels: value.channels,
                kernel_size: value.kernel_size,
                conv_input_bf16: value.conv_input_bf16,
                conv_history_bf16: value.conv_history_bf16,
                conv_weight_bf16: value.conv_weight_bf16,
                conv_output_bf16: value.conv_output_bf16,
                heads: value.heads,
                gdn_a_bf16: value.gdn_a_bf16,
                gdn_b_bf16: value.gdn_b_bf16,
                gdn_a_log_bf16: value.gdn_a_log_bf16,
                gdn_dt_bias_bf16: value.gdn_dt_bias_bf16,
                gate_f32: value.gate_f32,
                beta_f32: value.beta_f32,
            }
        }
    }

    #[repr(C)]
    pub struct SigmoidGateSpec {
        pub elements: usize,
        pub gate_bf16: DevicePtr,
        pub input_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
    }

    impl From<&crate::ops::SigmoidGateSpec> for SigmoidGateSpec {
        fn from(value: &crate::ops::SigmoidGateSpec) -> Self {
            Self {
                elements: value.elements,
                gate_bf16: value.gate_bf16,
                input_bf16: value.input_bf16,
                output_bf16: value.output_bf16,
            }
        }
    }

    #[repr(C)]
    pub struct SigmoidGateStridedSpec {
        pub rows: usize,
        pub elements_per_row: usize,
        pub gate_stride: usize,
        pub input_stride: usize,
        pub output_stride: usize,
        pub gate_bf16: DevicePtr,
        pub input_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
    }

    impl From<&crate::ops::SigmoidGateStridedSpec> for SigmoidGateStridedSpec {
        fn from(value: &crate::ops::SigmoidGateStridedSpec) -> Self {
            Self {
                rows: value.rows,
                elements_per_row: value.elements_per_row,
                gate_stride: value.gate_stride,
                input_stride: value.input_stride,
                output_stride: value.output_stride,
                gate_bf16: value.gate_bf16,
                input_bf16: value.input_bf16,
                output_bf16: value.output_bf16,
            }
        }
    }

    #[repr(C)]
    pub struct QProjDeinterleaveSpec {
        pub rows: usize,
        pub heads: usize,
        pub head_dim: usize,
        pub input_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
    }

    impl From<&crate::ops::QProjDeinterleaveSpec> for QProjDeinterleaveSpec {
        fn from(value: &crate::ops::QProjDeinterleaveSpec) -> Self {
            Self {
                rows: value.rows,
                heads: value.heads,
                head_dim: value.head_dim,
                input_bf16: value.input_bf16,
                output_bf16: value.output_bf16,
            }
        }
    }

    #[repr(C)]
    pub struct QProjSigmoidGateSpec {
        pub rows: usize,
        pub heads: usize,
        pub head_dim: usize,
        pub gate_bf16: DevicePtr,
        pub input_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
    }

    impl From<&crate::ops::QProjSigmoidGateSpec> for QProjSigmoidGateSpec {
        fn from(value: &crate::ops::QProjSigmoidGateSpec) -> Self {
            Self {
                rows: value.rows,
                heads: value.heads,
                head_dim: value.head_dim,
                gate_bf16: value.gate_bf16,
                input_bf16: value.input_bf16,
                output_bf16: value.output_bf16,
            }
        }
    }

    #[repr(C)]
    pub struct CopyStridedRowsSpec {
        pub rows: usize,
        pub values: usize,
        pub input_stride: usize,
        pub output_stride: usize,
        pub input_bf16: DevicePtr,
        pub output_bf16: DevicePtr,
    }

    impl From<&crate::ops::CopyStridedRowsSpec> for CopyStridedRowsSpec {
        fn from(value: &crate::ops::CopyStridedRowsSpec) -> Self {
            Self {
                rows: value.rows,
                values: value.values,
                input_stride: value.input_stride,
                output_stride: value.output_stride,
                input_bf16: value.input_bf16,
                output_bf16: value.output_bf16,
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct AttentionShape {
        pub q_heads: usize,
        pub kv_heads: usize,
        pub head_dim: usize,
        pub rope_dims: usize,
    }

    impl From<RustAttentionShape> for AttentionShape {
        fn from(value: RustAttentionShape) -> Self {
            Self {
                q_heads: value.q_heads,
                kv_heads: value.kv_heads,
                head_dim: value.head_dim,
                rope_dims: value.rope_dims,
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct DeltaNetShape {
        pub qk_heads: usize,
        pub v_heads: usize,
        pub key_dim: usize,
        pub value_dim: usize,
        pub conv_kernel: usize,
    }

    impl From<RustDeltaNetShape> for DeltaNetShape {
        fn from(value: RustDeltaNetShape) -> Self {
            Self {
                qk_heads: value.qk_heads,
                v_heads: value.v_heads,
                key_dim: value.key_dim,
                value_dim: value.value_dim,
                conv_kernel: value.conv_kernel,
            }
        }
    }

    #[link(name = "qwen36_fp4_kernels")]
    unsafe extern "C" {
        pub fn qwen36_nvfp4_gemm(spec: *const Nvfp4GemmSpec) -> i32;
        pub fn qwen36_megakernel_nvfp4_gemm(spec: *const Nvfp4GemmSpec) -> i32;
        pub fn qwen36_attention_prefill(spec: *const AttentionPrefillSpec) -> i32;
        pub fn qwen36_deltanet_decode(spec: *const DeltaNetDecodeSpec) -> i32;
        pub fn qwen36_attention_decode(spec: *const AttentionDecodeSpec) -> i32;
        pub fn qwen36_turboquant_encode_kv(spec: *const TurboQuantEncodeSpec) -> i32;
        pub fn qwen36_turboquant_attention(spec: *const TurboQuantAttentionSpec) -> i32;
        pub fn qwen36_rmsnorm(spec: *const RmsNormSpec) -> i32;
        pub fn qwen36_rmsnorm_nvfp4_quantize(spec: *const RmsNormNvfp4QuantizeSpec) -> i32;
        pub fn qwen36_partial_rope(spec: *const PartialRopeSpec) -> i32;
        pub fn qwen36_swiglu(spec: *const SwiGluSpec) -> i32;
        pub fn qwen36_swiglu_nvfp4_quantize(spec: *const SwiGluNvfp4QuantizeSpec) -> i32;
        pub fn qwen36_sample(spec: *const SamplingSpec) -> i32;
        pub fn qwen36_sample_rows(spec: *const SamplingRowsSpec) -> i32;
        pub fn qwen36_topk_argmax(spec: *const TopkArgmaxSpec) -> i32;
        pub fn qwen36_embedding_lookup(spec: *const EmbeddingLookupSpec) -> i32;
        pub fn qwen36_bf16_gemm(spec: *const Bf16GemmSpec) -> i32;
        pub fn qwen36_bf16_matvec(spec: *const Bf16MatVecSpec) -> i32;
        pub fn qwen36_nvfp4_matvec(spec: *const Nvfp4MatVecSpec) -> i32;
        pub fn qwen36_nvfp4_quantize_bf16(spec: *const Nvfp4QuantizeSpec) -> i32;
        pub fn qwen36_nvfp4_quantize_rows(spec: *const Nvfp4QuantizeRowsSpec) -> i32;
        pub fn qwen36_nvfp4_retile_scales(spec: *const Nvfp4RetileScalesSpec) -> i32;
        pub fn qwen36_conv1d_update(spec: *const Conv1dUpdateSpec) -> i32;
        pub fn qwen36_conv1d_prefill(spec: *const Conv1dPrefillSpec) -> i32;
        pub fn qwen36_gdn_gate(spec: *const GdnGateSpec) -> i32;
        pub fn qwen36_conv1d_gdn_gate_fused(spec: *const Conv1dGdnGateFusedSpec) -> i32;
        pub fn qwen36_sigmoid_gate(spec: *const SigmoidGateSpec) -> i32;
        pub fn qwen36_sigmoid_gate_strided(spec: *const SigmoidGateStridedSpec) -> i32;
        pub fn qwen36_q_proj_deinterleave(spec: *const QProjDeinterleaveSpec) -> i32;
        pub fn qwen36_q_proj_sigmoid_gate(spec: *const QProjSigmoidGateSpec) -> i32;
        pub fn qwen36_copy_strided_rows(spec: *const CopyStridedRowsSpec) -> i32;
    }
}
