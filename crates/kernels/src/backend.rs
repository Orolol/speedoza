use qwen36_fp4_core::{CoreError, Result};
use serde::{Deserialize, Serialize};

use crate::attention::{AttentionDecodeSpec, AttentionPrefillSpec};
use crate::deltanet::{DeltaNetDecodeSpec, DeltaNetPrefillSpec};
use crate::nvfp4_gemm::Nvfp4GemmSpec;
use crate::ops::{
    Bf16GemmSpec, Bf16MatVecSpec, Conv1dUpdateSpec, EmbeddingLookupSpec, GdnGateSpec,
    Nvfp4MatVecSpec, Nvfp4QuantizeSpec, Nvfp4RetileScalesSpec, SigmoidGateSpec,
};
use crate::rmsnorm::RmsNormSpec;
use crate::rope::PartialRopeSpec;
use crate::sampling::SamplingSpec;
use crate::swiglu::SwiGluSpec;
use crate::turboquant::{TurboQuantAttentionSpec, TurboQuantEncodeSpec};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

    fn partial_rope(&self, _spec: &PartialRopeSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("partial_rope"))
    }

    fn swiglu(&self, _spec: &SwiGluSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("swiglu"))
    }

    fn sample(&self, _spec: &SamplingSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("sample"))
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

    fn nvfp4_retile_scales(&self, _spec: &Nvfp4RetileScalesSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("nvfp4_retile_scales"))
    }

    fn conv1d_update(&self, _spec: &Conv1dUpdateSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("conv1d_update"))
    }

    fn gdn_gate(&self, _spec: &GdnGateSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("gdn_gate"))
    }

    fn sigmoid_gate(&self, _spec: &SigmoidGateSpec) -> Result<()> {
        Err(CoreError::UnsupportedNoCuda("sigmoid_gate"))
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

    fn sample(&self, spec: &SamplingSpec) -> Result<()> {
        let ffi_spec = ffi::SamplingSpec::from(spec);
        check("qwen36_sample", unsafe { ffi::qwen36_sample(&ffi_spec) })
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

    fn nvfp4_retile_scales(&self, spec: &Nvfp4RetileScalesSpec) -> Result<()> {
        nvfp4_retile_scales(spec)
    }

    fn conv1d_update(&self, spec: &Conv1dUpdateSpec) -> Result<()> {
        let ffi_spec = ffi::Conv1dUpdateSpec::from(spec);
        check("qwen36_conv1d_update", unsafe {
            ffi::qwen36_conv1d_update(&ffi_spec)
        })
    }

    fn gdn_gate(&self, spec: &GdnGateSpec) -> Result<()> {
        let ffi_spec = ffi::GdnGateSpec::from(spec);
        check("qwen36_gdn_gate", unsafe {
            ffi::qwen36_gdn_gate(&ffi_spec)
        })
    }

    fn sigmoid_gate(&self, spec: &SigmoidGateSpec) -> Result<()> {
        let ffi_spec = ffi::SigmoidGateSpec::from(spec);
        check("qwen36_sigmoid_gate", unsafe {
            ffi::qwen36_sigmoid_gate(&ffi_spec)
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
fn check(kernel: &'static str, code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(CoreError::KernelLaunch { kernel, code })
    }
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
            }
        }
    }

    #[repr(C)]
    pub struct DeltaNetDecodeSpec {
        pub layer_index: usize,
        pub tokens_in_persistent_loop: usize,
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
        pub positions_i32: DevicePtr,
        pub q_bf16: DevicePtr,
        pub k_bf16: DevicePtr,
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
                positions_i32: value.positions_i32,
                q_bf16: value.q_bf16,
                k_bf16: value.k_bf16,
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
    pub struct SamplingSpec {
        pub vocab_size: usize,
        pub logits_bf16: DevicePtr,
        pub output_token_u32: DevicePtr,
        pub temperature: f32,
        pub top_k: usize,
        pub top_p: f32,
        pub repetition_penalty: f32,
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
            }
        }
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
    }

    impl From<&crate::ops::Nvfp4QuantizeSpec> for Nvfp4QuantizeSpec {
        fn from(value: &crate::ops::Nvfp4QuantizeSpec) -> Self {
            Self {
                values: value.values,
                input_bf16: value.input_bf16,
                output_fp4: value.output_fp4,
                output_scale_e4m3: value.output_scale_e4m3,
                output_tensor_scale_f32: value.output_tensor_scale_f32,
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
    pub struct GdnGateSpec {
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
        pub fn qwen36_deltanet_decode(spec: *const DeltaNetDecodeSpec) -> i32;
        pub fn qwen36_attention_decode(spec: *const AttentionDecodeSpec) -> i32;
        pub fn qwen36_turboquant_encode_kv(spec: *const TurboQuantEncodeSpec) -> i32;
        pub fn qwen36_turboquant_attention(spec: *const TurboQuantAttentionSpec) -> i32;
        pub fn qwen36_rmsnorm(spec: *const RmsNormSpec) -> i32;
        pub fn qwen36_partial_rope(spec: *const PartialRopeSpec) -> i32;
        pub fn qwen36_swiglu(spec: *const SwiGluSpec) -> i32;
        pub fn qwen36_sample(spec: *const SamplingSpec) -> i32;
        pub fn qwen36_embedding_lookup(spec: *const EmbeddingLookupSpec) -> i32;
        pub fn qwen36_bf16_gemm(spec: *const Bf16GemmSpec) -> i32;
        pub fn qwen36_bf16_matvec(spec: *const Bf16MatVecSpec) -> i32;
        pub fn qwen36_nvfp4_matvec(spec: *const Nvfp4MatVecSpec) -> i32;
        pub fn qwen36_nvfp4_quantize_bf16(spec: *const Nvfp4QuantizeSpec) -> i32;
        pub fn qwen36_nvfp4_retile_scales(spec: *const Nvfp4RetileScalesSpec) -> i32;
        pub fn qwen36_conv1d_update(spec: *const Conv1dUpdateSpec) -> i32;
        pub fn qwen36_gdn_gate(spec: *const GdnGateSpec) -> i32;
        pub fn qwen36_sigmoid_gate(spec: *const SigmoidGateSpec) -> i32;
    }
}
