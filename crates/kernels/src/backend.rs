use qwen36_fp4_core::{CoreError, Result};
use serde::{Deserialize, Serialize};

use crate::attention::{AttentionDecodeSpec, AttentionPrefillSpec};
use crate::deltanet::{DeltaNetDecodeSpec, DeltaNetPrefillSpec};
use crate::nvfp4_gemm::Nvfp4GemmSpec;
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
        pub shape: DeltaNetShape,
        pub state_decay: f32,
        pub update_scale: f32,
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
                shape: DeltaNetShape::from(value.shape),
                state_decay: value.state_decay,
                update_scale: value.update_scale,
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
    }
}
