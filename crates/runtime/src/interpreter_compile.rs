#[cfg(feature = "cuda")]
use qwen36_fp4_core::Result;
use qwen36_fp4_core::{LayerType, ModelTopology};
#[cfg(feature = "cuda")]
use qwen36_fp4_kernels::{CudaDeviceBuffer, KernelBackend};
use qwen36_fp4_kernels::{
    DevicePtr, InterpreterInstruction, InterpreterOpcode, InterpreterProgram,
    InterpreterProgramSpec,
};

/// Host-side compiler for the decode interpreter instruction stream.
///
/// Stage 0 emits the same static shape as the planned megakernel and records
/// the requested opcode in each payload. These topology-only instructions do
/// not own real tensor/weight pointers, so they are always routed to
/// `FALLBACK_TRAMPOLINE`; concrete per-op instructions are built with the
/// typed constructors in `qwen36_fp4_kernels::interpreter`.
#[derive(Debug, Clone)]
pub struct DecodeInterpreterProgram {
    pub program: InterpreterProgram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeInterpreterLogitsParams {
    pub hidden: usize,
    pub vocab_size: usize,
    pub hidden_bf16: DevicePtr,
    pub residual_bf16: DevicePtr,
    pub final_norm_weight_bf16: DevicePtr,
    pub normed_bf16: DevicePtr,
    pub activation_fp4: DevicePtr,
    pub activation_scale_e4m3: DevicePtr,
    pub activation_tensor_scale_f32: DevicePtr,
    pub lm_head_weight_bf16: DevicePtr,
    pub logits_bf16: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterMlpParams {
    pub hidden: usize,
    pub intermediate: usize,
    pub input_fp4: DevicePtr,
    pub input_scale_e4m3: DevicePtr,
    pub gate_weight_fp4: DevicePtr,
    pub gate_weight_scale_e4m3: DevicePtr,
    pub gate_alpha: f32,
    pub gate_out_bf16: DevicePtr,
    pub up_weight_fp4: DevicePtr,
    pub up_weight_scale_e4m3: DevicePtr,
    pub up_alpha: f32,
    pub up_out_bf16: DevicePtr,
    pub swiglu_fp4: DevicePtr,
    pub swiglu_scale_e4m3: DevicePtr,
    pub swiglu_tensor_scale_f32: DevicePtr,
    pub down_input_tensor_scale: f32,
    pub down_weight_fp4: DevicePtr,
    pub down_weight_scale_e4m3: DevicePtr,
    pub down_alpha: f32,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterRmsNormNvfp4QuantParams {
    pub hidden: usize,
    pub eps: f32,
    pub input_tensor_scale_f32: f32,
    pub input_bf16: DevicePtr,
    pub weight_bf16: DevicePtr,
    pub residual_bf16: DevicePtr,
    pub residual_out_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
    pub output_fp4: DevicePtr,
    pub output_scale_e4m3: DevicePtr,
    pub output_tensor_scale_f32: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterNormMlpParams {
    pub norm: DecodeInterpreterRmsNormNvfp4QuantParams,
    pub mlp: DecodeInterpreterMlpParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeInterpreterResidualAddParams {
    pub values: usize,
    pub input_bf16: DevicePtr,
    pub residual_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterRopeParams {
    pub tokens: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub base_theta: f64,
    pub position_i32: i32,
    pub use_scalar_position: bool,
    pub positions_i32: DevicePtr,
    pub q_bf16: DevicePtr,
    pub k_bf16: DevicePtr,
    pub scalar_position_device_i32: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeInterpreterDeltaNetParams {
    pub spec: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeInterpreterAttentionParams {
    pub spec: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterRopeAttentionParams {
    pub rope: DecodeInterpreterRopeParams,
    pub attention: DecodeInterpreterAttentionParams,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterNvfp4GemvParams {
    pub m: usize,
    pub k: usize,
    pub alpha: f32,
    pub a_fp4: DevicePtr,
    pub a_scale_e4m3: DevicePtr,
    pub b_fp4: DevicePtr,
    pub b_scale_e4m3: DevicePtr,
    pub c_bf16: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterRmsNormBf16Params {
    pub rows: usize,
    pub hidden: usize,
    pub eps: f32,
    pub direct_weight: bool,
    pub input_bf16: DevicePtr,
    pub weight_bf16: DevicePtr,
    pub residual_bf16: DevicePtr,
    pub residual_out_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterNvfp4QuantizeParams {
    pub values: usize,
    pub input_tensor_scale_f32: f32,
    pub input_bf16: DevicePtr,
    pub output_fp4: DevicePtr,
    pub output_scale_e4m3: DevicePtr,
    pub output_tensor_scale_f32: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterSwiGluNvfp4QuantParams {
    pub values: usize,
    pub input_tensor_scale_f32: f32,
    pub gate_bf16: DevicePtr,
    pub up_bf16: DevicePtr,
    pub output_fp4: DevicePtr,
    pub output_scale_e4m3: DevicePtr,
    pub output_tensor_scale_f32: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeInterpreterSwiGluBf16Params {
    pub rows: usize,
    pub intermediate: usize,
    pub gate_bf16: DevicePtr,
    pub up_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeInterpreterConv1dGdnGateFusedParams {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeInterpreterQProjDeinterleaveParams {
    pub rows: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub input_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeInterpreterQProjSigmoidGateParams {
    pub rows: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub gate_bf16: DevicePtr,
    pub input_bf16: DevicePtr,
    pub output_bf16: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterFullAttentionLayerParams {
    pub q_proj: DecodeInterpreterNvfp4GemvParams,
    pub k_proj: DecodeInterpreterNvfp4GemvParams,
    pub v_proj: DecodeInterpreterNvfp4GemvParams,
    pub q_proj_deinterleave: DecodeInterpreterQProjDeinterleaveParams,
    pub q_norm: DecodeInterpreterRmsNormBf16Params,
    pub k_norm: DecodeInterpreterRmsNormBf16Params,
    pub rope: DecodeInterpreterRopeParams,
    pub attention: DecodeInterpreterAttentionParams,
    pub q_proj_gate: DecodeInterpreterQProjSigmoidGateParams,
    pub o_input_quant: DecodeInterpreterNvfp4QuantizeParams,
    pub o_proj: DecodeInterpreterNvfp4GemvParams,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterFullAttentionInputLayerParams {
    pub input_norm: DecodeInterpreterRmsNormNvfp4QuantParams,
    pub layer: DecodeInterpreterFullAttentionLayerParams,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterFullTransformerLayerParams {
    pub attention: DecodeInterpreterFullAttentionInputLayerParams,
    pub post: DecodeInterpreterNormMlpParams,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterLinearAttentionTailParams {
    pub norm: DecodeInterpreterRmsNormBf16Params,
    pub swiglu: DecodeInterpreterSwiGluBf16Params,
    pub quant: DecodeInterpreterNvfp4QuantizeParams,
    pub out_proj: DecodeInterpreterNvfp4GemvParams,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterLinearAttentionPostInProjParams {
    pub conv_gdn: DecodeInterpreterConv1dGdnGateFusedParams,
    pub deltanet: DecodeInterpreterDeltaNetParams,
    pub tail: DecodeInterpreterLinearAttentionTailParams,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterLinearAttentionLayerParams {
    pub in_proj: DecodeInterpreterNvfp4GemvParams,
    pub post_inproj: DecodeInterpreterLinearAttentionPostInProjParams,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterLinearAttentionInputLayerParams {
    pub input_norm: DecodeInterpreterRmsNormNvfp4QuantParams,
    pub layer: DecodeInterpreterLinearAttentionLayerParams,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeInterpreterLinearTransformerLayerParams {
    pub attention: DecodeInterpreterLinearAttentionInputLayerParams,
    pub post: DecodeInterpreterNormMlpParams,
}

impl DecodeInterpreterProgram {
    pub fn compile(topology: &ModelTopology) -> Self {
        let mut compiler = DecodeInterpreterCompiler {
            program: InterpreterProgram::new(),
            next_counter: 0,
            last_counter: None,
        };

        for (layer_idx, layer_type) in topology.layer_types.iter().copied().enumerate() {
            compiler.push(layer_idx, 0, InterpreterOpcode::RmsNormNvfp4Quant);
            match layer_type {
                LayerType::LinearAttention => {
                    compiler.push(layer_idx, 1, InterpreterOpcode::Nvfp4Gemv);
                    compiler.push(layer_idx, 2, InterpreterOpcode::DeltaNetRecur);
                    compiler.push(layer_idx, 3, InterpreterOpcode::Nvfp4Gemv);
                }
                LayerType::FullAttention => {
                    compiler.push(layer_idx, 1, InterpreterOpcode::Nvfp4Gemv);
                    compiler.push(layer_idx, 2, InterpreterOpcode::RopePartial);
                    compiler.push(layer_idx, 3, InterpreterOpcode::AttnDecodeFull);
                    compiler.push(layer_idx, 4, InterpreterOpcode::Nvfp4Gemv);
                }
            }
            compiler.push(layer_idx, 5, InterpreterOpcode::ResidualAdd);
            compiler.push(layer_idx, 6, InterpreterOpcode::RmsNormNvfp4Quant);
            compiler.push(layer_idx, 7, InterpreterOpcode::Nvfp4Gemv);
            compiler.push(layer_idx, 8, InterpreterOpcode::SwiGluNvfp4Quant);
            compiler.push(layer_idx, 9, InterpreterOpcode::Nvfp4Gemv);
        }

        compiler.push(
            topology.num_hidden_layers,
            0,
            InterpreterOpcode::RmsNormNvfp4Quant,
        );
        compiler.push(
            topology.num_hidden_layers,
            1,
            InterpreterOpcode::LmHeadTiled,
        );

        Self {
            program: compiler.program.finish(),
        }
    }

    /// Compile the first runtime-backed slice of the decode interpreter:
    /// final RMSNorm followed by tiled BF16 lm_head. The RMSNorm opcode also
    /// emits FP4 scratch, but only its BF16 output feeds lm_head.
    pub fn compile_final_logits(params: DecodeInterpreterLogitsParams) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::rmsnorm_nvfp4_quant(
                params.hidden,
                1.0e-6,
                1.0,
                params.hidden_bf16,
                params.final_norm_weight_bf16,
                params.residual_bf16,
                DevicePtr::NULL,
                params.normed_bf16,
                params.activation_fp4,
                params.activation_scale_e4m3,
                params.activation_tensor_scale_f32,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        program.push(
            InterpreterInstruction::lm_head_tiled(
                params.vocab_size,
                params.hidden,
                params.normed_bf16,
                params.lm_head_weight_bf16,
                params.logits_bf16,
            )
            .with_dep(0, 1)
            .with_publish(2, 1)
            .with_arrival_counter(3),
        );
        Self {
            program: program.finish(),
        }
    }

    /// Compile the runtime-backed decode MLP slice. The input activation is
    /// already NVFP4-quantized by the preceding post-attention RMSNorm.
    pub fn compile_mlp(params: DecodeInterpreterMlpParams) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.intermediate,
                params.hidden,
                params.gate_alpha,
                params.gate_weight_fp4,
                params.gate_weight_scale_e4m3,
                params.input_fp4,
                params.input_scale_e4m3,
                params.gate_out_bf16,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.intermediate,
                params.hidden,
                params.up_alpha,
                params.up_weight_fp4,
                params.up_weight_scale_e4m3,
                params.input_fp4,
                params.input_scale_e4m3,
                params.up_out_bf16,
            )
            .with_dep(0, 1)
            .with_publish(2, 1)
            .with_arrival_counter(3),
        );
        program.push(
            InterpreterInstruction::swiglu_nvfp4_quant(
                params.intermediate,
                params.down_input_tensor_scale,
                params.gate_out_bf16,
                params.up_out_bf16,
                params.swiglu_fp4,
                params.swiglu_scale_e4m3,
                params.swiglu_tensor_scale_f32,
            )
            .with_dep(2, 1)
            .with_publish(4, 1)
            .with_arrival_counter(5),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.hidden,
                params.intermediate,
                params.down_alpha,
                params.down_weight_fp4,
                params.down_weight_scale_e4m3,
                params.swiglu_fp4,
                params.swiglu_scale_e4m3,
                params.output_bf16,
            )
            .with_dep(4, 1)
            .with_publish(6, 1)
            .with_arrival_counter(7),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_rmsnorm_nvfp4_quant(params: DecodeInterpreterRmsNormNvfp4QuantParams) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::rmsnorm_nvfp4_quant(
                params.hidden,
                params.eps,
                params.input_tensor_scale_f32,
                params.input_bf16,
                params.weight_bf16,
                params.residual_bf16,
                params.residual_out_bf16,
                params.output_bf16,
                params.output_fp4,
                params.output_scale_e4m3,
                params.output_tensor_scale_f32,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_rmsnorm_mlp(params: DecodeInterpreterNormMlpParams) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::rmsnorm_nvfp4_quant(
                params.norm.hidden,
                params.norm.eps,
                params.norm.input_tensor_scale_f32,
                params.norm.input_bf16,
                params.norm.weight_bf16,
                params.norm.residual_bf16,
                params.norm.residual_out_bf16,
                params.norm.output_bf16,
                params.norm.output_fp4,
                params.norm.output_scale_e4m3,
                params.norm.output_tensor_scale_f32,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.mlp.intermediate,
                params.mlp.hidden,
                params.mlp.gate_alpha,
                params.mlp.gate_weight_fp4,
                params.mlp.gate_weight_scale_e4m3,
                params.mlp.input_fp4,
                params.mlp.input_scale_e4m3,
                params.mlp.gate_out_bf16,
            )
            .with_dep(0, 1)
            .with_publish(2, 1)
            .with_arrival_counter(3),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.mlp.intermediate,
                params.mlp.hidden,
                params.mlp.up_alpha,
                params.mlp.up_weight_fp4,
                params.mlp.up_weight_scale_e4m3,
                params.mlp.input_fp4,
                params.mlp.input_scale_e4m3,
                params.mlp.up_out_bf16,
            )
            .with_dep(2, 1)
            .with_publish(4, 1)
            .with_arrival_counter(5),
        );
        program.push(
            InterpreterInstruction::swiglu_nvfp4_quant(
                params.mlp.intermediate,
                params.mlp.down_input_tensor_scale,
                params.mlp.gate_out_bf16,
                params.mlp.up_out_bf16,
                params.mlp.swiglu_fp4,
                params.mlp.swiglu_scale_e4m3,
                params.mlp.swiglu_tensor_scale_f32,
            )
            .with_dep(4, 1)
            .with_publish(6, 1)
            .with_arrival_counter(7),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.mlp.hidden,
                params.mlp.intermediate,
                params.mlp.down_alpha,
                params.mlp.down_weight_fp4,
                params.mlp.down_weight_scale_e4m3,
                params.mlp.swiglu_fp4,
                params.mlp.swiglu_scale_e4m3,
                params.mlp.output_bf16,
            )
            .with_dep(6, 1)
            .with_publish(8, 1)
            .with_arrival_counter(9),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_residual_add(params: DecodeInterpreterResidualAddParams) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::residual_add(
                params.values,
                params.input_bf16,
                params.residual_bf16,
                params.output_bf16,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_rope_partial(params: DecodeInterpreterRopeParams) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::rope_partial(
                params.tokens,
                params.q_heads,
                params.kv_heads,
                params.head_dim,
                params.rope_dims,
                params.base_theta,
                params.position_i32,
                params.use_scalar_position,
                params.positions_i32,
                params.q_bf16,
                params.k_bf16,
                params.scalar_position_device_i32,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_deltanet_recur(params: DecodeInterpreterDeltaNetParams) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::deltanet_recur_spec(params.spec)
                .with_publish(0, 1)
                .with_arrival_counter(1),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_attention_decode_full(params: DecodeInterpreterAttentionParams) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::attn_decode_full_spec(params.spec)
                .with_publish(0, 1)
                .with_arrival_counter(1),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_rope_attention_decode(params: DecodeInterpreterRopeAttentionParams) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::rope_partial(
                params.rope.tokens,
                params.rope.q_heads,
                params.rope.kv_heads,
                params.rope.head_dim,
                params.rope.rope_dims,
                params.rope.base_theta,
                params.rope.position_i32,
                params.rope.use_scalar_position,
                params.rope.positions_i32,
                params.rope.q_bf16,
                params.rope.k_bf16,
                params.rope.scalar_position_device_i32,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        program.push(
            InterpreterInstruction::attn_decode_full_spec(params.attention.spec)
                .with_dep(0, 1)
                .with_publish(2, 1)
                .with_arrival_counter(3),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_full_attention_layer_decode(
        params: DecodeInterpreterFullAttentionLayerParams,
    ) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.q_proj.m,
                params.q_proj.k,
                params.q_proj.alpha,
                params.q_proj.a_fp4,
                params.q_proj.a_scale_e4m3,
                params.q_proj.b_fp4,
                params.q_proj.b_scale_e4m3,
                params.q_proj.c_bf16,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.k_proj.m,
                params.k_proj.k,
                params.k_proj.alpha,
                params.k_proj.a_fp4,
                params.k_proj.a_scale_e4m3,
                params.k_proj.b_fp4,
                params.k_proj.b_scale_e4m3,
                params.k_proj.c_bf16,
            )
            .with_dep(0, 1)
            .with_publish(2, 1)
            .with_arrival_counter(3),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.v_proj.m,
                params.v_proj.k,
                params.v_proj.alpha,
                params.v_proj.a_fp4,
                params.v_proj.a_scale_e4m3,
                params.v_proj.b_fp4,
                params.v_proj.b_scale_e4m3,
                params.v_proj.c_bf16,
            )
            .with_dep(2, 1)
            .with_publish(4, 1)
            .with_arrival_counter(5),
        );
        program.push(
            InterpreterInstruction::q_proj_deinterleave(
                params.q_proj_deinterleave.rows,
                params.q_proj_deinterleave.heads,
                params.q_proj_deinterleave.head_dim,
                params.q_proj_deinterleave.input_bf16,
                params.q_proj_deinterleave.output_bf16,
            )
            .with_dep(4, 1)
            .with_publish(6, 1)
            .with_arrival_counter(7),
        );
        program.push(
            InterpreterInstruction::rmsnorm_bf16(
                params.q_norm.rows,
                params.q_norm.hidden,
                params.q_norm.eps,
                params.q_norm.direct_weight,
                params.q_norm.input_bf16,
                params.q_norm.weight_bf16,
                params.q_norm.residual_bf16,
                params.q_norm.residual_out_bf16,
                params.q_norm.output_bf16,
            )
            .with_dep(6, 1)
            .with_publish(8, 1)
            .with_arrival_counter(9),
        );
        program.push(
            InterpreterInstruction::rmsnorm_bf16(
                params.k_norm.rows,
                params.k_norm.hidden,
                params.k_norm.eps,
                params.k_norm.direct_weight,
                params.k_norm.input_bf16,
                params.k_norm.weight_bf16,
                params.k_norm.residual_bf16,
                params.k_norm.residual_out_bf16,
                params.k_norm.output_bf16,
            )
            .with_dep(8, 1)
            .with_publish(10, 1)
            .with_arrival_counter(11),
        );
        program.push(
            InterpreterInstruction::rope_partial(
                params.rope.tokens,
                params.rope.q_heads,
                params.rope.kv_heads,
                params.rope.head_dim,
                params.rope.rope_dims,
                params.rope.base_theta,
                params.rope.position_i32,
                params.rope.use_scalar_position,
                params.rope.positions_i32,
                params.rope.q_bf16,
                params.rope.k_bf16,
                params.rope.scalar_position_device_i32,
            )
            .with_dep(10, 1)
            .with_publish(12, 1)
            .with_arrival_counter(13),
        );
        program.push(
            InterpreterInstruction::attn_decode_full_spec(params.attention.spec)
                .with_dep(12, 1)
                .with_publish(14, 1)
                .with_arrival_counter(15),
        );
        program.push(
            InterpreterInstruction::q_proj_sigmoid_gate(
                params.q_proj_gate.rows,
                params.q_proj_gate.heads,
                params.q_proj_gate.head_dim,
                params.q_proj_gate.gate_bf16,
                params.q_proj_gate.input_bf16,
                params.q_proj_gate.output_bf16,
            )
            .with_dep(14, 1)
            .with_publish(16, 1)
            .with_arrival_counter(17),
        );
        program.push(
            InterpreterInstruction::nvfp4_quantize(
                params.o_input_quant.values,
                params.o_input_quant.input_tensor_scale_f32,
                params.o_input_quant.input_bf16,
                params.o_input_quant.output_fp4,
                params.o_input_quant.output_scale_e4m3,
                params.o_input_quant.output_tensor_scale_f32,
            )
            .with_dep(16, 1)
            .with_publish(18, 1)
            .with_arrival_counter(19),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.o_proj.m,
                params.o_proj.k,
                params.o_proj.alpha,
                params.o_proj.a_fp4,
                params.o_proj.a_scale_e4m3,
                params.o_proj.b_fp4,
                params.o_proj.b_scale_e4m3,
                params.o_proj.c_bf16,
            )
            .with_dep(18, 1)
            .with_publish(20, 1)
            .with_arrival_counter(21),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_full_attention_input_layer_decode(
        params: DecodeInterpreterFullAttentionInputLayerParams,
    ) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::rmsnorm_nvfp4_quant(
                params.input_norm.hidden,
                params.input_norm.eps,
                params.input_norm.input_tensor_scale_f32,
                params.input_norm.input_bf16,
                params.input_norm.weight_bf16,
                params.input_norm.residual_bf16,
                params.input_norm.residual_out_bf16,
                params.input_norm.output_bf16,
                params.input_norm.output_fp4,
                params.input_norm.output_scale_e4m3,
                params.input_norm.output_tensor_scale_f32,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.layer.q_proj.m,
                params.layer.q_proj.k,
                params.layer.q_proj.alpha,
                params.layer.q_proj.a_fp4,
                params.layer.q_proj.a_scale_e4m3,
                params.layer.q_proj.b_fp4,
                params.layer.q_proj.b_scale_e4m3,
                params.layer.q_proj.c_bf16,
            )
            .with_dep(0, 1)
            .with_publish(2, 1)
            .with_arrival_counter(3),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.layer.k_proj.m,
                params.layer.k_proj.k,
                params.layer.k_proj.alpha,
                params.layer.k_proj.a_fp4,
                params.layer.k_proj.a_scale_e4m3,
                params.layer.k_proj.b_fp4,
                params.layer.k_proj.b_scale_e4m3,
                params.layer.k_proj.c_bf16,
            )
            .with_dep(2, 1)
            .with_publish(4, 1)
            .with_arrival_counter(5),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.layer.v_proj.m,
                params.layer.v_proj.k,
                params.layer.v_proj.alpha,
                params.layer.v_proj.a_fp4,
                params.layer.v_proj.a_scale_e4m3,
                params.layer.v_proj.b_fp4,
                params.layer.v_proj.b_scale_e4m3,
                params.layer.v_proj.c_bf16,
            )
            .with_dep(4, 1)
            .with_publish(6, 1)
            .with_arrival_counter(7),
        );
        program.push(
            InterpreterInstruction::q_proj_deinterleave(
                params.layer.q_proj_deinterleave.rows,
                params.layer.q_proj_deinterleave.heads,
                params.layer.q_proj_deinterleave.head_dim,
                params.layer.q_proj_deinterleave.input_bf16,
                params.layer.q_proj_deinterleave.output_bf16,
            )
            .with_dep(6, 1)
            .with_publish(8, 1)
            .with_arrival_counter(9),
        );
        program.push(
            InterpreterInstruction::rmsnorm_bf16(
                params.layer.q_norm.rows,
                params.layer.q_norm.hidden,
                params.layer.q_norm.eps,
                params.layer.q_norm.direct_weight,
                params.layer.q_norm.input_bf16,
                params.layer.q_norm.weight_bf16,
                params.layer.q_norm.residual_bf16,
                params.layer.q_norm.residual_out_bf16,
                params.layer.q_norm.output_bf16,
            )
            .with_dep(8, 1)
            .with_publish(10, 1)
            .with_arrival_counter(11),
        );
        program.push(
            InterpreterInstruction::rmsnorm_bf16(
                params.layer.k_norm.rows,
                params.layer.k_norm.hidden,
                params.layer.k_norm.eps,
                params.layer.k_norm.direct_weight,
                params.layer.k_norm.input_bf16,
                params.layer.k_norm.weight_bf16,
                params.layer.k_norm.residual_bf16,
                params.layer.k_norm.residual_out_bf16,
                params.layer.k_norm.output_bf16,
            )
            .with_dep(10, 1)
            .with_publish(12, 1)
            .with_arrival_counter(13),
        );
        program.push(
            InterpreterInstruction::rope_partial(
                params.layer.rope.tokens,
                params.layer.rope.q_heads,
                params.layer.rope.kv_heads,
                params.layer.rope.head_dim,
                params.layer.rope.rope_dims,
                params.layer.rope.base_theta,
                params.layer.rope.position_i32,
                params.layer.rope.use_scalar_position,
                params.layer.rope.positions_i32,
                params.layer.rope.q_bf16,
                params.layer.rope.k_bf16,
                params.layer.rope.scalar_position_device_i32,
            )
            .with_dep(12, 1)
            .with_publish(14, 1)
            .with_arrival_counter(15),
        );
        program.push(
            InterpreterInstruction::attn_decode_full_spec(params.layer.attention.spec)
                .with_dep(14, 1)
                .with_publish(16, 1)
                .with_arrival_counter(17),
        );
        program.push(
            InterpreterInstruction::q_proj_sigmoid_gate(
                params.layer.q_proj_gate.rows,
                params.layer.q_proj_gate.heads,
                params.layer.q_proj_gate.head_dim,
                params.layer.q_proj_gate.gate_bf16,
                params.layer.q_proj_gate.input_bf16,
                params.layer.q_proj_gate.output_bf16,
            )
            .with_dep(16, 1)
            .with_publish(18, 1)
            .with_arrival_counter(19),
        );
        program.push(
            InterpreterInstruction::nvfp4_quantize(
                params.layer.o_input_quant.values,
                params.layer.o_input_quant.input_tensor_scale_f32,
                params.layer.o_input_quant.input_bf16,
                params.layer.o_input_quant.output_fp4,
                params.layer.o_input_quant.output_scale_e4m3,
                params.layer.o_input_quant.output_tensor_scale_f32,
            )
            .with_dep(18, 1)
            .with_publish(20, 1)
            .with_arrival_counter(21),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.layer.o_proj.m,
                params.layer.o_proj.k,
                params.layer.o_proj.alpha,
                params.layer.o_proj.a_fp4,
                params.layer.o_proj.a_scale_e4m3,
                params.layer.o_proj.b_fp4,
                params.layer.o_proj.b_scale_e4m3,
                params.layer.o_proj.c_bf16,
            )
            .with_dep(20, 1)
            .with_publish(22, 1)
            .with_arrival_counter(23),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_full_transformer_layer_decode(
        params: DecodeInterpreterFullTransformerLayerParams,
    ) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::rmsnorm_nvfp4_quant(
                params.attention.input_norm.hidden,
                params.attention.input_norm.eps,
                params.attention.input_norm.input_tensor_scale_f32,
                params.attention.input_norm.input_bf16,
                params.attention.input_norm.weight_bf16,
                params.attention.input_norm.residual_bf16,
                params.attention.input_norm.residual_out_bf16,
                params.attention.input_norm.output_bf16,
                params.attention.input_norm.output_fp4,
                params.attention.input_norm.output_scale_e4m3,
                params.attention.input_norm.output_tensor_scale_f32,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.attention.layer.q_proj.m,
                params.attention.layer.q_proj.k,
                params.attention.layer.q_proj.alpha,
                params.attention.layer.q_proj.a_fp4,
                params.attention.layer.q_proj.a_scale_e4m3,
                params.attention.layer.q_proj.b_fp4,
                params.attention.layer.q_proj.b_scale_e4m3,
                params.attention.layer.q_proj.c_bf16,
            )
            .with_dep(0, 1)
            .with_publish(2, 1)
            .with_arrival_counter(3),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.attention.layer.k_proj.m,
                params.attention.layer.k_proj.k,
                params.attention.layer.k_proj.alpha,
                params.attention.layer.k_proj.a_fp4,
                params.attention.layer.k_proj.a_scale_e4m3,
                params.attention.layer.k_proj.b_fp4,
                params.attention.layer.k_proj.b_scale_e4m3,
                params.attention.layer.k_proj.c_bf16,
            )
            .with_dep(2, 1)
            .with_publish(4, 1)
            .with_arrival_counter(5),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.attention.layer.v_proj.m,
                params.attention.layer.v_proj.k,
                params.attention.layer.v_proj.alpha,
                params.attention.layer.v_proj.a_fp4,
                params.attention.layer.v_proj.a_scale_e4m3,
                params.attention.layer.v_proj.b_fp4,
                params.attention.layer.v_proj.b_scale_e4m3,
                params.attention.layer.v_proj.c_bf16,
            )
            .with_dep(4, 1)
            .with_publish(6, 1)
            .with_arrival_counter(7),
        );
        program.push(
            InterpreterInstruction::q_proj_deinterleave(
                params.attention.layer.q_proj_deinterleave.rows,
                params.attention.layer.q_proj_deinterleave.heads,
                params.attention.layer.q_proj_deinterleave.head_dim,
                params.attention.layer.q_proj_deinterleave.input_bf16,
                params.attention.layer.q_proj_deinterleave.output_bf16,
            )
            .with_dep(6, 1)
            .with_publish(8, 1)
            .with_arrival_counter(9),
        );
        program.push(
            InterpreterInstruction::rmsnorm_bf16(
                params.attention.layer.q_norm.rows,
                params.attention.layer.q_norm.hidden,
                params.attention.layer.q_norm.eps,
                params.attention.layer.q_norm.direct_weight,
                params.attention.layer.q_norm.input_bf16,
                params.attention.layer.q_norm.weight_bf16,
                params.attention.layer.q_norm.residual_bf16,
                params.attention.layer.q_norm.residual_out_bf16,
                params.attention.layer.q_norm.output_bf16,
            )
            .with_dep(8, 1)
            .with_publish(10, 1)
            .with_arrival_counter(11),
        );
        program.push(
            InterpreterInstruction::rmsnorm_bf16(
                params.attention.layer.k_norm.rows,
                params.attention.layer.k_norm.hidden,
                params.attention.layer.k_norm.eps,
                params.attention.layer.k_norm.direct_weight,
                params.attention.layer.k_norm.input_bf16,
                params.attention.layer.k_norm.weight_bf16,
                params.attention.layer.k_norm.residual_bf16,
                params.attention.layer.k_norm.residual_out_bf16,
                params.attention.layer.k_norm.output_bf16,
            )
            .with_dep(10, 1)
            .with_publish(12, 1)
            .with_arrival_counter(13),
        );
        program.push(
            InterpreterInstruction::rope_partial(
                params.attention.layer.rope.tokens,
                params.attention.layer.rope.q_heads,
                params.attention.layer.rope.kv_heads,
                params.attention.layer.rope.head_dim,
                params.attention.layer.rope.rope_dims,
                params.attention.layer.rope.base_theta,
                params.attention.layer.rope.position_i32,
                params.attention.layer.rope.use_scalar_position,
                params.attention.layer.rope.positions_i32,
                params.attention.layer.rope.q_bf16,
                params.attention.layer.rope.k_bf16,
                params.attention.layer.rope.scalar_position_device_i32,
            )
            .with_dep(12, 1)
            .with_publish(14, 1)
            .with_arrival_counter(15),
        );
        program.push(
            InterpreterInstruction::attn_decode_full_spec(params.attention.layer.attention.spec)
                .with_dep(14, 1)
                .with_publish(16, 1)
                .with_arrival_counter(17),
        );
        program.push(
            InterpreterInstruction::q_proj_sigmoid_gate(
                params.attention.layer.q_proj_gate.rows,
                params.attention.layer.q_proj_gate.heads,
                params.attention.layer.q_proj_gate.head_dim,
                params.attention.layer.q_proj_gate.gate_bf16,
                params.attention.layer.q_proj_gate.input_bf16,
                params.attention.layer.q_proj_gate.output_bf16,
            )
            .with_dep(16, 1)
            .with_publish(18, 1)
            .with_arrival_counter(19),
        );
        program.push(
            InterpreterInstruction::nvfp4_quantize(
                params.attention.layer.o_input_quant.values,
                params.attention.layer.o_input_quant.input_tensor_scale_f32,
                params.attention.layer.o_input_quant.input_bf16,
                params.attention.layer.o_input_quant.output_fp4,
                params.attention.layer.o_input_quant.output_scale_e4m3,
                params.attention.layer.o_input_quant.output_tensor_scale_f32,
            )
            .with_dep(18, 1)
            .with_publish(20, 1)
            .with_arrival_counter(21),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.attention.layer.o_proj.m,
                params.attention.layer.o_proj.k,
                params.attention.layer.o_proj.alpha,
                params.attention.layer.o_proj.a_fp4,
                params.attention.layer.o_proj.a_scale_e4m3,
                params.attention.layer.o_proj.b_fp4,
                params.attention.layer.o_proj.b_scale_e4m3,
                params.attention.layer.o_proj.c_bf16,
            )
            .with_dep(20, 1)
            .with_publish(22, 1)
            .with_arrival_counter(23),
        );
        program.push(
            InterpreterInstruction::rmsnorm_nvfp4_quant(
                params.post.norm.hidden,
                params.post.norm.eps,
                params.post.norm.input_tensor_scale_f32,
                params.post.norm.input_bf16,
                params.post.norm.weight_bf16,
                params.post.norm.residual_bf16,
                params.post.norm.residual_out_bf16,
                params.post.norm.output_bf16,
                params.post.norm.output_fp4,
                params.post.norm.output_scale_e4m3,
                params.post.norm.output_tensor_scale_f32,
            )
            .with_dep(22, 1)
            .with_publish(24, 1)
            .with_arrival_counter(25),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.post.mlp.intermediate,
                params.post.mlp.hidden,
                params.post.mlp.gate_alpha,
                params.post.mlp.gate_weight_fp4,
                params.post.mlp.gate_weight_scale_e4m3,
                params.post.mlp.input_fp4,
                params.post.mlp.input_scale_e4m3,
                params.post.mlp.gate_out_bf16,
            )
            .with_dep(24, 1)
            .with_publish(26, 1)
            .with_arrival_counter(27),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.post.mlp.intermediate,
                params.post.mlp.hidden,
                params.post.mlp.up_alpha,
                params.post.mlp.up_weight_fp4,
                params.post.mlp.up_weight_scale_e4m3,
                params.post.mlp.input_fp4,
                params.post.mlp.input_scale_e4m3,
                params.post.mlp.up_out_bf16,
            )
            .with_dep(26, 1)
            .with_publish(28, 1)
            .with_arrival_counter(29),
        );
        program.push(
            InterpreterInstruction::swiglu_nvfp4_quant(
                params.post.mlp.intermediate,
                params.post.mlp.down_input_tensor_scale,
                params.post.mlp.gate_out_bf16,
                params.post.mlp.up_out_bf16,
                params.post.mlp.swiglu_fp4,
                params.post.mlp.swiglu_scale_e4m3,
                params.post.mlp.swiglu_tensor_scale_f32,
            )
            .with_dep(28, 1)
            .with_publish(30, 1)
            .with_arrival_counter(31),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.post.mlp.hidden,
                params.post.mlp.intermediate,
                params.post.mlp.down_alpha,
                params.post.mlp.down_weight_fp4,
                params.post.mlp.down_weight_scale_e4m3,
                params.post.mlp.swiglu_fp4,
                params.post.mlp.swiglu_scale_e4m3,
                params.post.mlp.output_bf16,
            )
            .with_dep(30, 1)
            .with_publish(32, 1)
            .with_arrival_counter(33),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_linear_attention_tail_decode(
        params: DecodeInterpreterLinearAttentionTailParams,
    ) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::rmsnorm_bf16(
                params.norm.rows,
                params.norm.hidden,
                params.norm.eps,
                params.norm.direct_weight,
                params.norm.input_bf16,
                params.norm.weight_bf16,
                params.norm.residual_bf16,
                params.norm.residual_out_bf16,
                params.norm.output_bf16,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        program.push(
            InterpreterInstruction::swiglu_bf16(
                params.swiglu.rows,
                params.swiglu.intermediate,
                params.swiglu.gate_bf16,
                params.swiglu.up_bf16,
                params.swiglu.output_bf16,
            )
            .with_dep(0, 1)
            .with_publish(2, 1)
            .with_arrival_counter(3),
        );
        program.push(
            InterpreterInstruction::nvfp4_quantize(
                params.quant.values,
                params.quant.input_tensor_scale_f32,
                params.quant.input_bf16,
                params.quant.output_fp4,
                params.quant.output_scale_e4m3,
                params.quant.output_tensor_scale_f32,
            )
            .with_dep(2, 1)
            .with_publish(4, 1)
            .with_arrival_counter(5),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.out_proj.m,
                params.out_proj.k,
                params.out_proj.alpha,
                params.out_proj.a_fp4,
                params.out_proj.a_scale_e4m3,
                params.out_proj.b_fp4,
                params.out_proj.b_scale_e4m3,
                params.out_proj.c_bf16,
            )
            .with_dep(4, 1)
            .with_publish(6, 1)
            .with_arrival_counter(7),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_linear_attention_post_inproj_decode(
        params: DecodeInterpreterLinearAttentionPostInProjParams,
    ) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::conv1d_gdn_gate_fused(
                params.conv_gdn.channels,
                params.conv_gdn.kernel_size,
                params.conv_gdn.conv_input_bf16,
                params.conv_gdn.conv_history_bf16,
                params.conv_gdn.conv_weight_bf16,
                params.conv_gdn.conv_output_bf16,
                params.conv_gdn.heads,
                params.conv_gdn.gdn_a_bf16,
                params.conv_gdn.gdn_b_bf16,
                params.conv_gdn.gdn_a_log_bf16,
                params.conv_gdn.gdn_dt_bias_bf16,
                params.conv_gdn.gate_f32,
                params.conv_gdn.beta_f32,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        program.push(
            InterpreterInstruction::deltanet_recur_spec(params.deltanet.spec)
                .with_dep(0, 1)
                .with_publish(2, 1)
                .with_arrival_counter(3),
        );
        program.push(
            InterpreterInstruction::rmsnorm_bf16(
                params.tail.norm.rows,
                params.tail.norm.hidden,
                params.tail.norm.eps,
                params.tail.norm.direct_weight,
                params.tail.norm.input_bf16,
                params.tail.norm.weight_bf16,
                params.tail.norm.residual_bf16,
                params.tail.norm.residual_out_bf16,
                params.tail.norm.output_bf16,
            )
            .with_dep(2, 1)
            .with_publish(4, 1)
            .with_arrival_counter(5),
        );
        program.push(
            InterpreterInstruction::swiglu_bf16(
                params.tail.swiglu.rows,
                params.tail.swiglu.intermediate,
                params.tail.swiglu.gate_bf16,
                params.tail.swiglu.up_bf16,
                params.tail.swiglu.output_bf16,
            )
            .with_dep(4, 1)
            .with_publish(6, 1)
            .with_arrival_counter(7),
        );
        program.push(
            InterpreterInstruction::nvfp4_quantize(
                params.tail.quant.values,
                params.tail.quant.input_tensor_scale_f32,
                params.tail.quant.input_bf16,
                params.tail.quant.output_fp4,
                params.tail.quant.output_scale_e4m3,
                params.tail.quant.output_tensor_scale_f32,
            )
            .with_dep(6, 1)
            .with_publish(8, 1)
            .with_arrival_counter(9),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.tail.out_proj.m,
                params.tail.out_proj.k,
                params.tail.out_proj.alpha,
                params.tail.out_proj.a_fp4,
                params.tail.out_proj.a_scale_e4m3,
                params.tail.out_proj.b_fp4,
                params.tail.out_proj.b_scale_e4m3,
                params.tail.out_proj.c_bf16,
            )
            .with_dep(8, 1)
            .with_publish(10, 1)
            .with_arrival_counter(11),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_linear_attention_layer_decode(
        params: DecodeInterpreterLinearAttentionLayerParams,
    ) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.in_proj.m,
                params.in_proj.k,
                params.in_proj.alpha,
                params.in_proj.a_fp4,
                params.in_proj.a_scale_e4m3,
                params.in_proj.b_fp4,
                params.in_proj.b_scale_e4m3,
                params.in_proj.c_bf16,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        program.push(
            InterpreterInstruction::conv1d_gdn_gate_fused(
                params.post_inproj.conv_gdn.channels,
                params.post_inproj.conv_gdn.kernel_size,
                params.post_inproj.conv_gdn.conv_input_bf16,
                params.post_inproj.conv_gdn.conv_history_bf16,
                params.post_inproj.conv_gdn.conv_weight_bf16,
                params.post_inproj.conv_gdn.conv_output_bf16,
                params.post_inproj.conv_gdn.heads,
                params.post_inproj.conv_gdn.gdn_a_bf16,
                params.post_inproj.conv_gdn.gdn_b_bf16,
                params.post_inproj.conv_gdn.gdn_a_log_bf16,
                params.post_inproj.conv_gdn.gdn_dt_bias_bf16,
                params.post_inproj.conv_gdn.gate_f32,
                params.post_inproj.conv_gdn.beta_f32,
            )
            .with_dep(0, 1)
            .with_publish(2, 1)
            .with_arrival_counter(3),
        );
        program.push(
            InterpreterInstruction::deltanet_recur_spec(params.post_inproj.deltanet.spec)
                .with_dep(2, 1)
                .with_publish(4, 1)
                .with_arrival_counter(5),
        );
        program.push(
            InterpreterInstruction::rmsnorm_bf16(
                params.post_inproj.tail.norm.rows,
                params.post_inproj.tail.norm.hidden,
                params.post_inproj.tail.norm.eps,
                params.post_inproj.tail.norm.direct_weight,
                params.post_inproj.tail.norm.input_bf16,
                params.post_inproj.tail.norm.weight_bf16,
                params.post_inproj.tail.norm.residual_bf16,
                params.post_inproj.tail.norm.residual_out_bf16,
                params.post_inproj.tail.norm.output_bf16,
            )
            .with_dep(4, 1)
            .with_publish(6, 1)
            .with_arrival_counter(7),
        );
        program.push(
            InterpreterInstruction::swiglu_bf16(
                params.post_inproj.tail.swiglu.rows,
                params.post_inproj.tail.swiglu.intermediate,
                params.post_inproj.tail.swiglu.gate_bf16,
                params.post_inproj.tail.swiglu.up_bf16,
                params.post_inproj.tail.swiglu.output_bf16,
            )
            .with_dep(6, 1)
            .with_publish(8, 1)
            .with_arrival_counter(9),
        );
        program.push(
            InterpreterInstruction::nvfp4_quantize(
                params.post_inproj.tail.quant.values,
                params.post_inproj.tail.quant.input_tensor_scale_f32,
                params.post_inproj.tail.quant.input_bf16,
                params.post_inproj.tail.quant.output_fp4,
                params.post_inproj.tail.quant.output_scale_e4m3,
                params.post_inproj.tail.quant.output_tensor_scale_f32,
            )
            .with_dep(8, 1)
            .with_publish(10, 1)
            .with_arrival_counter(11),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.post_inproj.tail.out_proj.m,
                params.post_inproj.tail.out_proj.k,
                params.post_inproj.tail.out_proj.alpha,
                params.post_inproj.tail.out_proj.a_fp4,
                params.post_inproj.tail.out_proj.a_scale_e4m3,
                params.post_inproj.tail.out_proj.b_fp4,
                params.post_inproj.tail.out_proj.b_scale_e4m3,
                params.post_inproj.tail.out_proj.c_bf16,
            )
            .with_dep(10, 1)
            .with_publish(12, 1)
            .with_arrival_counter(13),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_linear_attention_input_layer_decode(
        params: DecodeInterpreterLinearAttentionInputLayerParams,
    ) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::rmsnorm_nvfp4_quant(
                params.input_norm.hidden,
                params.input_norm.eps,
                params.input_norm.input_tensor_scale_f32,
                params.input_norm.input_bf16,
                params.input_norm.weight_bf16,
                params.input_norm.residual_bf16,
                params.input_norm.residual_out_bf16,
                params.input_norm.output_bf16,
                params.input_norm.output_fp4,
                params.input_norm.output_scale_e4m3,
                params.input_norm.output_tensor_scale_f32,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.layer.in_proj.m,
                params.layer.in_proj.k,
                params.layer.in_proj.alpha,
                params.layer.in_proj.a_fp4,
                params.layer.in_proj.a_scale_e4m3,
                params.layer.in_proj.b_fp4,
                params.layer.in_proj.b_scale_e4m3,
                params.layer.in_proj.c_bf16,
            )
            .with_dep(0, 1)
            .with_publish(2, 1)
            .with_arrival_counter(3),
        );
        program.push(
            InterpreterInstruction::conv1d_gdn_gate_fused(
                params.layer.post_inproj.conv_gdn.channels,
                params.layer.post_inproj.conv_gdn.kernel_size,
                params.layer.post_inproj.conv_gdn.conv_input_bf16,
                params.layer.post_inproj.conv_gdn.conv_history_bf16,
                params.layer.post_inproj.conv_gdn.conv_weight_bf16,
                params.layer.post_inproj.conv_gdn.conv_output_bf16,
                params.layer.post_inproj.conv_gdn.heads,
                params.layer.post_inproj.conv_gdn.gdn_a_bf16,
                params.layer.post_inproj.conv_gdn.gdn_b_bf16,
                params.layer.post_inproj.conv_gdn.gdn_a_log_bf16,
                params.layer.post_inproj.conv_gdn.gdn_dt_bias_bf16,
                params.layer.post_inproj.conv_gdn.gate_f32,
                params.layer.post_inproj.conv_gdn.beta_f32,
            )
            .with_dep(2, 1)
            .with_publish(4, 1)
            .with_arrival_counter(5),
        );
        program.push(
            InterpreterInstruction::deltanet_recur_spec(params.layer.post_inproj.deltanet.spec)
                .with_dep(4, 1)
                .with_publish(6, 1)
                .with_arrival_counter(7),
        );
        program.push(
            InterpreterInstruction::rmsnorm_bf16(
                params.layer.post_inproj.tail.norm.rows,
                params.layer.post_inproj.tail.norm.hidden,
                params.layer.post_inproj.tail.norm.eps,
                params.layer.post_inproj.tail.norm.direct_weight,
                params.layer.post_inproj.tail.norm.input_bf16,
                params.layer.post_inproj.tail.norm.weight_bf16,
                params.layer.post_inproj.tail.norm.residual_bf16,
                params.layer.post_inproj.tail.norm.residual_out_bf16,
                params.layer.post_inproj.tail.norm.output_bf16,
            )
            .with_dep(6, 1)
            .with_publish(8, 1)
            .with_arrival_counter(9),
        );
        program.push(
            InterpreterInstruction::swiglu_bf16(
                params.layer.post_inproj.tail.swiglu.rows,
                params.layer.post_inproj.tail.swiglu.intermediate,
                params.layer.post_inproj.tail.swiglu.gate_bf16,
                params.layer.post_inproj.tail.swiglu.up_bf16,
                params.layer.post_inproj.tail.swiglu.output_bf16,
            )
            .with_dep(8, 1)
            .with_publish(10, 1)
            .with_arrival_counter(11),
        );
        program.push(
            InterpreterInstruction::nvfp4_quantize(
                params.layer.post_inproj.tail.quant.values,
                params.layer.post_inproj.tail.quant.input_tensor_scale_f32,
                params.layer.post_inproj.tail.quant.input_bf16,
                params.layer.post_inproj.tail.quant.output_fp4,
                params.layer.post_inproj.tail.quant.output_scale_e4m3,
                params.layer.post_inproj.tail.quant.output_tensor_scale_f32,
            )
            .with_dep(10, 1)
            .with_publish(12, 1)
            .with_arrival_counter(13),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.layer.post_inproj.tail.out_proj.m,
                params.layer.post_inproj.tail.out_proj.k,
                params.layer.post_inproj.tail.out_proj.alpha,
                params.layer.post_inproj.tail.out_proj.a_fp4,
                params.layer.post_inproj.tail.out_proj.a_scale_e4m3,
                params.layer.post_inproj.tail.out_proj.b_fp4,
                params.layer.post_inproj.tail.out_proj.b_scale_e4m3,
                params.layer.post_inproj.tail.out_proj.c_bf16,
            )
            .with_dep(12, 1)
            .with_publish(14, 1)
            .with_arrival_counter(15),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn compile_linear_transformer_layer_decode(
        params: DecodeInterpreterLinearTransformerLayerParams,
    ) -> Self {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::rmsnorm_nvfp4_quant(
                params.attention.input_norm.hidden,
                params.attention.input_norm.eps,
                params.attention.input_norm.input_tensor_scale_f32,
                params.attention.input_norm.input_bf16,
                params.attention.input_norm.weight_bf16,
                params.attention.input_norm.residual_bf16,
                params.attention.input_norm.residual_out_bf16,
                params.attention.input_norm.output_bf16,
                params.attention.input_norm.output_fp4,
                params.attention.input_norm.output_scale_e4m3,
                params.attention.input_norm.output_tensor_scale_f32,
            )
            .with_publish(0, 1)
            .with_arrival_counter(1),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.attention.layer.in_proj.m,
                params.attention.layer.in_proj.k,
                params.attention.layer.in_proj.alpha,
                params.attention.layer.in_proj.a_fp4,
                params.attention.layer.in_proj.a_scale_e4m3,
                params.attention.layer.in_proj.b_fp4,
                params.attention.layer.in_proj.b_scale_e4m3,
                params.attention.layer.in_proj.c_bf16,
            )
            .with_dep(0, 1)
            .with_publish(2, 1)
            .with_arrival_counter(3),
        );
        program.push(
            InterpreterInstruction::conv1d_gdn_gate_fused(
                params.attention.layer.post_inproj.conv_gdn.channels,
                params.attention.layer.post_inproj.conv_gdn.kernel_size,
                params.attention.layer.post_inproj.conv_gdn.conv_input_bf16,
                params
                    .attention
                    .layer
                    .post_inproj
                    .conv_gdn
                    .conv_history_bf16,
                params.attention.layer.post_inproj.conv_gdn.conv_weight_bf16,
                params.attention.layer.post_inproj.conv_gdn.conv_output_bf16,
                params.attention.layer.post_inproj.conv_gdn.heads,
                params.attention.layer.post_inproj.conv_gdn.gdn_a_bf16,
                params.attention.layer.post_inproj.conv_gdn.gdn_b_bf16,
                params.attention.layer.post_inproj.conv_gdn.gdn_a_log_bf16,
                params.attention.layer.post_inproj.conv_gdn.gdn_dt_bias_bf16,
                params.attention.layer.post_inproj.conv_gdn.gate_f32,
                params.attention.layer.post_inproj.conv_gdn.beta_f32,
            )
            .with_dep(2, 1)
            .with_publish(4, 1)
            .with_arrival_counter(5),
        );
        program.push(
            InterpreterInstruction::deltanet_recur_spec(
                params.attention.layer.post_inproj.deltanet.spec,
            )
            .with_dep(4, 1)
            .with_publish(6, 1)
            .with_arrival_counter(7),
        );
        program.push(
            InterpreterInstruction::rmsnorm_bf16(
                params.attention.layer.post_inproj.tail.norm.rows,
                params.attention.layer.post_inproj.tail.norm.hidden,
                params.attention.layer.post_inproj.tail.norm.eps,
                params.attention.layer.post_inproj.tail.norm.direct_weight,
                params.attention.layer.post_inproj.tail.norm.input_bf16,
                params.attention.layer.post_inproj.tail.norm.weight_bf16,
                params.attention.layer.post_inproj.tail.norm.residual_bf16,
                params
                    .attention
                    .layer
                    .post_inproj
                    .tail
                    .norm
                    .residual_out_bf16,
                params.attention.layer.post_inproj.tail.norm.output_bf16,
            )
            .with_dep(6, 1)
            .with_publish(8, 1)
            .with_arrival_counter(9),
        );
        program.push(
            InterpreterInstruction::swiglu_bf16(
                params.attention.layer.post_inproj.tail.swiglu.rows,
                params.attention.layer.post_inproj.tail.swiglu.intermediate,
                params.attention.layer.post_inproj.tail.swiglu.gate_bf16,
                params.attention.layer.post_inproj.tail.swiglu.up_bf16,
                params.attention.layer.post_inproj.tail.swiglu.output_bf16,
            )
            .with_dep(8, 1)
            .with_publish(10, 1)
            .with_arrival_counter(11),
        );
        program.push(
            InterpreterInstruction::nvfp4_quantize(
                params.attention.layer.post_inproj.tail.quant.values,
                params
                    .attention
                    .layer
                    .post_inproj
                    .tail
                    .quant
                    .input_tensor_scale_f32,
                params.attention.layer.post_inproj.tail.quant.input_bf16,
                params.attention.layer.post_inproj.tail.quant.output_fp4,
                params
                    .attention
                    .layer
                    .post_inproj
                    .tail
                    .quant
                    .output_scale_e4m3,
                params
                    .attention
                    .layer
                    .post_inproj
                    .tail
                    .quant
                    .output_tensor_scale_f32,
            )
            .with_dep(10, 1)
            .with_publish(12, 1)
            .with_arrival_counter(13),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.attention.layer.post_inproj.tail.out_proj.m,
                params.attention.layer.post_inproj.tail.out_proj.k,
                params.attention.layer.post_inproj.tail.out_proj.alpha,
                params.attention.layer.post_inproj.tail.out_proj.a_fp4,
                params
                    .attention
                    .layer
                    .post_inproj
                    .tail
                    .out_proj
                    .a_scale_e4m3,
                params.attention.layer.post_inproj.tail.out_proj.b_fp4,
                params
                    .attention
                    .layer
                    .post_inproj
                    .tail
                    .out_proj
                    .b_scale_e4m3,
                params.attention.layer.post_inproj.tail.out_proj.c_bf16,
            )
            .with_dep(12, 1)
            .with_publish(14, 1)
            .with_arrival_counter(15),
        );
        program.push(
            InterpreterInstruction::rmsnorm_nvfp4_quant(
                params.post.norm.hidden,
                params.post.norm.eps,
                params.post.norm.input_tensor_scale_f32,
                params.post.norm.input_bf16,
                params.post.norm.weight_bf16,
                params.post.norm.residual_bf16,
                params.post.norm.residual_out_bf16,
                params.post.norm.output_bf16,
                params.post.norm.output_fp4,
                params.post.norm.output_scale_e4m3,
                params.post.norm.output_tensor_scale_f32,
            )
            .with_dep(14, 1)
            .with_publish(16, 1)
            .with_arrival_counter(17),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.post.mlp.intermediate,
                params.post.mlp.hidden,
                params.post.mlp.gate_alpha,
                params.post.mlp.gate_weight_fp4,
                params.post.mlp.gate_weight_scale_e4m3,
                params.post.mlp.input_fp4,
                params.post.mlp.input_scale_e4m3,
                params.post.mlp.gate_out_bf16,
            )
            .with_dep(16, 1)
            .with_publish(18, 1)
            .with_arrival_counter(19),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.post.mlp.intermediate,
                params.post.mlp.hidden,
                params.post.mlp.up_alpha,
                params.post.mlp.up_weight_fp4,
                params.post.mlp.up_weight_scale_e4m3,
                params.post.mlp.input_fp4,
                params.post.mlp.input_scale_e4m3,
                params.post.mlp.up_out_bf16,
            )
            .with_dep(18, 1)
            .with_publish(20, 1)
            .with_arrival_counter(21),
        );
        program.push(
            InterpreterInstruction::swiglu_nvfp4_quant(
                params.post.mlp.intermediate,
                params.post.mlp.down_input_tensor_scale,
                params.post.mlp.gate_out_bf16,
                params.post.mlp.up_out_bf16,
                params.post.mlp.swiglu_fp4,
                params.post.mlp.swiglu_scale_e4m3,
                params.post.mlp.swiglu_tensor_scale_f32,
            )
            .with_dep(20, 1)
            .with_publish(22, 1)
            .with_arrival_counter(23),
        );
        program.push(
            InterpreterInstruction::nvfp4_gemv(
                params.post.mlp.hidden,
                params.post.mlp.intermediate,
                params.post.mlp.down_alpha,
                params.post.mlp.down_weight_fp4,
                params.post.mlp.down_weight_scale_e4m3,
                params.post.mlp.swiglu_fp4,
                params.post.mlp.swiglu_scale_e4m3,
                params.post.mlp.output_bf16,
            )
            .with_dep(22, 1)
            .with_publish(24, 1)
            .with_arrival_counter(25),
        );
        Self {
            program: program.finish(),
        }
    }

    pub fn spec(
        &self,
        instructions: qwen36_fp4_kernels::DevicePtr,
        counters_i32: qwen36_fp4_kernels::DevicePtr,
        cta_count: u32,
    ) -> InterpreterProgramSpec {
        InterpreterProgramSpec {
            instructions,
            instruction_count: self.program.instructions.len(),
            counters_i32,
            counter_count: self.program.counter_count,
            cta_count,
            flags: 0,
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaDecodeInterpreterProgram {
    pub host: DecodeInterpreterProgram,
    instructions: CudaDeviceBuffer,
    counters_i32: CudaDeviceBuffer,
}

#[cfg(feature = "cuda")]
impl CudaDecodeInterpreterProgram {
    pub fn upload(topology: &ModelTopology) -> Result<Self> {
        let host = DecodeInterpreterProgram::compile(topology);
        let instruction_bytes = instructions_as_bytes(&host.program.instructions);
        let instructions = CudaDeviceBuffer::alloc(instruction_bytes.len())?;
        instructions.copy_from_host(instruction_bytes)?;
        let counters_i32 = CudaDeviceBuffer::zeroed(host.program.counter_count * 4)?;
        Ok(Self {
            host,
            instructions,
            counters_i32,
        })
    }

    pub fn run<B: KernelBackend>(&self, backend: &B, cta_count: u32) -> Result<()> {
        self.counters_i32.memset_async(0)?;
        backend.interpreter_decode_sm120(&self.host.spec(
            self.instructions.ptr(),
            self.counters_i32.ptr(),
            cta_count,
        ))
    }

    pub fn instruction_ptr(&self) -> qwen36_fp4_kernels::DevicePtr {
        self.instructions.ptr()
    }

    pub fn counters_ptr(&self) -> qwen36_fp4_kernels::DevicePtr {
        self.counters_i32.ptr()
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn instructions_as_bytes(instructions: &[InterpreterInstruction]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            instructions.as_ptr().cast::<u8>(),
            std::mem::size_of_val(instructions),
        )
    }
}

struct DecodeInterpreterCompiler {
    program: InterpreterProgram,
    next_counter: u32,
    last_counter: Option<u32>,
}

impl DecodeInterpreterCompiler {
    fn push(&mut self, layer_idx: usize, op_ordinal: u64, opcode: InterpreterOpcode) {
        let routed_opcode = InterpreterOpcode::FallbackTrampoline;
        let publish_counter = self.next_counter;
        let arrival_counter = self.next_counter + 1;
        self.next_counter += 2;

        let mut instruction = InterpreterInstruction::new(routed_opcode)
            .with_publish(publish_counter, 1)
            .with_arrival_counter(arrival_counter);
        if let Some(last_counter) = self.last_counter {
            instruction = instruction.with_dep(last_counter, 1);
        }
        instruction.payload[0] = layer_idx as u64;
        instruction.payload[1] = op_ordinal;
        instruction.payload[2] = opcode.code() as u64;
        self.program.push(instruction);
        self.last_counter = Some(publish_counter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen36_decode_program_has_static_shape() {
        let topology = ModelTopology::expected_qwen36_text_mtp();
        let compiled = DecodeInterpreterProgram::compile(&topology);
        let non_exit = compiled.program.instructions.len() - 1;
        // Linear layers emit 9 instructions, full-attn layers emit 10, plus
        // final RMSNorm + lm_head.
        assert_eq!(non_exit, 48 * 9 + 16 * 10 + 2);
        assert_eq!(compiled.program.counter_count, non_exit * 2);
    }

    #[test]
    fn final_logits_program_uses_real_opcodes_and_counters() {
        let compiled =
            DecodeInterpreterProgram::compile_final_logits(DecodeInterpreterLogitsParams {
                hidden: 8,
                vocab_size: 4,
                hidden_bf16: DevicePtr(10),
                residual_bf16: DevicePtr(11),
                final_norm_weight_bf16: DevicePtr(12),
                normed_bf16: DevicePtr(13),
                activation_fp4: DevicePtr(14),
                activation_scale_e4m3: DevicePtr(15),
                activation_tensor_scale_f32: DevicePtr(16),
                lm_head_weight_bf16: DevicePtr(17),
                logits_bf16: DevicePtr(18),
            });
        assert_eq!(compiled.program.instructions.len(), 3);
        assert_eq!(compiled.program.counter_count, 4);
        assert_eq!(
            compiled.program.instructions[0].opcode(),
            Some(InterpreterOpcode::RmsNormNvfp4Quant)
        );
        assert_eq!(
            compiled.program.instructions[1].opcode(),
            Some(InterpreterOpcode::LmHeadTiled)
        );
        assert_eq!(compiled.program.instructions[1].deps[0].counter_id, 0);
        assert_eq!(compiled.program.instructions[1].deps[0].target, 1);
        assert_eq!(
            compiled.program.instructions[2].opcode(),
            Some(InterpreterOpcode::Exit)
        );
    }

    #[test]
    fn mlp_program_uses_real_opcodes_and_counters() {
        let compiled = DecodeInterpreterProgram::compile_mlp(DecodeInterpreterMlpParams {
            hidden: 8,
            intermediate: 16,
            input_fp4: DevicePtr(20),
            input_scale_e4m3: DevicePtr(21),
            gate_weight_fp4: DevicePtr(22),
            gate_weight_scale_e4m3: DevicePtr(23),
            gate_alpha: 0.5,
            gate_out_bf16: DevicePtr(24),
            up_weight_fp4: DevicePtr(25),
            up_weight_scale_e4m3: DevicePtr(26),
            up_alpha: 0.75,
            up_out_bf16: DevicePtr(27),
            swiglu_fp4: DevicePtr(28),
            swiglu_scale_e4m3: DevicePtr(29),
            swiglu_tensor_scale_f32: DevicePtr(30),
            down_input_tensor_scale: 1.25,
            down_weight_fp4: DevicePtr(31),
            down_weight_scale_e4m3: DevicePtr(32),
            down_alpha: 1.5,
            output_bf16: DevicePtr(33),
        });
        assert_eq!(compiled.program.instructions.len(), 5);
        assert_eq!(compiled.program.counter_count, 8);
        assert_eq!(
            compiled.program.instructions[0].opcode(),
            Some(InterpreterOpcode::Nvfp4Gemv)
        );
        assert_eq!(
            compiled.program.instructions[1].opcode(),
            Some(InterpreterOpcode::Nvfp4Gemv)
        );
        assert_eq!(
            compiled.program.instructions[2].opcode(),
            Some(InterpreterOpcode::SwiGluNvfp4Quant)
        );
        assert_eq!(
            compiled.program.instructions[3].opcode(),
            Some(InterpreterOpcode::Nvfp4Gemv)
        );
        assert_eq!(compiled.program.instructions[1].deps[0].counter_id, 0);
        assert_eq!(compiled.program.instructions[2].deps[0].counter_id, 2);
        assert_eq!(compiled.program.instructions[3].deps[0].counter_id, 4);
        assert_eq!(
            compiled.program.instructions[4].opcode(),
            Some(InterpreterOpcode::Exit)
        );
    }

    #[test]
    fn rmsnorm_mlp_program_uses_real_opcodes_and_counters() {
        let compiled =
            DecodeInterpreterProgram::compile_rmsnorm_mlp(DecodeInterpreterNormMlpParams {
                norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                    hidden: 8,
                    eps: 1.0e-6,
                    input_tensor_scale_f32: 0.5,
                    input_bf16: DevicePtr(10),
                    weight_bf16: DevicePtr(11),
                    residual_bf16: DevicePtr(12),
                    residual_out_bf16: DevicePtr(13),
                    output_bf16: DevicePtr(14),
                    output_fp4: DevicePtr(20),
                    output_scale_e4m3: DevicePtr(21),
                    output_tensor_scale_f32: DevicePtr(22),
                },
                mlp: DecodeInterpreterMlpParams {
                    hidden: 8,
                    intermediate: 16,
                    input_fp4: DevicePtr(20),
                    input_scale_e4m3: DevicePtr(21),
                    gate_weight_fp4: DevicePtr(23),
                    gate_weight_scale_e4m3: DevicePtr(24),
                    gate_alpha: 0.5,
                    gate_out_bf16: DevicePtr(25),
                    up_weight_fp4: DevicePtr(26),
                    up_weight_scale_e4m3: DevicePtr(27),
                    up_alpha: 0.75,
                    up_out_bf16: DevicePtr(28),
                    swiglu_fp4: DevicePtr(20),
                    swiglu_scale_e4m3: DevicePtr(21),
                    swiglu_tensor_scale_f32: DevicePtr(22),
                    down_input_tensor_scale: 1.25,
                    down_weight_fp4: DevicePtr(29),
                    down_weight_scale_e4m3: DevicePtr(30),
                    down_alpha: 1.5,
                    output_bf16: DevicePtr(31),
                },
            });
        assert_eq!(compiled.program.instructions.len(), 6);
        assert_eq!(compiled.program.counter_count, 10);
        assert_eq!(
            compiled.program.instructions[0].opcode(),
            Some(InterpreterOpcode::RmsNormNvfp4Quant)
        );
        assert_eq!(
            compiled.program.instructions[1].opcode(),
            Some(InterpreterOpcode::Nvfp4Gemv)
        );
        assert_eq!(
            compiled.program.instructions[2].opcode(),
            Some(InterpreterOpcode::Nvfp4Gemv)
        );
        assert_eq!(
            compiled.program.instructions[3].opcode(),
            Some(InterpreterOpcode::SwiGluNvfp4Quant)
        );
        assert_eq!(
            compiled.program.instructions[4].opcode(),
            Some(InterpreterOpcode::Nvfp4Gemv)
        );
        assert_eq!(compiled.program.instructions[1].deps[0].counter_id, 0);
        assert_eq!(compiled.program.instructions[2].deps[0].counter_id, 2);
        assert_eq!(compiled.program.instructions[3].deps[0].counter_id, 4);
        assert_eq!(compiled.program.instructions[4].deps[0].counter_id, 6);
        assert_eq!(
            compiled.program.instructions[5].opcode(),
            Some(InterpreterOpcode::Exit)
        );
    }

    #[test]
    fn rmsnorm_quant_program_uses_real_opcode_and_counter() {
        let compiled = DecodeInterpreterProgram::compile_rmsnorm_nvfp4_quant(
            DecodeInterpreterRmsNormNvfp4QuantParams {
                hidden: 8,
                eps: 1.0e-6,
                input_tensor_scale_f32: 0.5,
                input_bf16: DevicePtr(10),
                weight_bf16: DevicePtr(11),
                residual_bf16: DevicePtr(12),
                residual_out_bf16: DevicePtr(13),
                output_bf16: DevicePtr(14),
                output_fp4: DevicePtr(15),
                output_scale_e4m3: DevicePtr(16),
                output_tensor_scale_f32: DevicePtr(17),
            },
        );
        assert_eq!(compiled.program.instructions.len(), 2);
        assert_eq!(compiled.program.counter_count, 2);
        assert_eq!(
            compiled.program.instructions[0].opcode(),
            Some(InterpreterOpcode::RmsNormNvfp4Quant)
        );
        assert_eq!(compiled.program.instructions[0].payload[0], 8);
        assert_eq!(compiled.program.instructions[0].payload[1], 10);
        assert_eq!(compiled.program.instructions[0].payload[2], 11);
        assert_eq!(compiled.program.instructions[0].payload[3], 12);
        assert_eq!(compiled.program.instructions[0].payload[4], 13);
        assert_eq!(compiled.program.instructions[0].payload[5], 14);
        assert_eq!(compiled.program.instructions[0].payload[6], 15);
        assert_eq!(compiled.program.instructions[0].payload[7], 16);
        assert_eq!(compiled.program.instructions[0].payload[8], 17);
        assert_eq!(
            compiled.program.instructions[1].opcode(),
            Some(InterpreterOpcode::Exit)
        );
    }

    #[test]
    fn residual_add_program_uses_real_opcode_and_counter() {
        let compiled =
            DecodeInterpreterProgram::compile_residual_add(DecodeInterpreterResidualAddParams {
                values: 8,
                input_bf16: DevicePtr(20),
                residual_bf16: DevicePtr(21),
                output_bf16: DevicePtr(22),
            });
        assert_eq!(compiled.program.instructions.len(), 2);
        assert_eq!(compiled.program.counter_count, 2);
        assert_eq!(
            compiled.program.instructions[0].opcode(),
            Some(InterpreterOpcode::ResidualAdd)
        );
        assert_eq!(compiled.program.instructions[0].payload[0], 8);
        assert_eq!(compiled.program.instructions[0].payload[1], 20);
        assert_eq!(compiled.program.instructions[0].payload[2], 21);
        assert_eq!(compiled.program.instructions[0].payload[3], 22);
        assert_eq!(
            compiled.program.instructions[1].opcode(),
            Some(InterpreterOpcode::Exit)
        );
    }

    #[test]
    fn rope_program_uses_real_opcode_and_counter() {
        let compiled =
            DecodeInterpreterProgram::compile_rope_partial(DecodeInterpreterRopeParams {
                tokens: 1,
                q_heads: 24,
                kv_heads: 4,
                head_dim: 256,
                rope_dims: 64,
                base_theta: 10000.0,
                position_i32: 7,
                use_scalar_position: true,
                positions_i32: DevicePtr::NULL,
                q_bf16: DevicePtr(100),
                k_bf16: DevicePtr(200),
                scalar_position_device_i32: DevicePtr::NULL,
            });
        assert_eq!(compiled.program.instructions.len(), 2);
        assert_eq!(compiled.program.counter_count, 2);
        assert_eq!(
            compiled.program.instructions[0].opcode(),
            Some(InterpreterOpcode::RopePartial)
        );
        assert_eq!(compiled.program.instructions[0].payload[0], 1);
        assert_eq!(compiled.program.instructions[0].payload[1], 24);
        assert_eq!(compiled.program.instructions[0].payload[2], 4);
        assert_eq!(compiled.program.instructions[0].payload[3], 256);
        assert_eq!(compiled.program.instructions[0].payload[4], 64);
        assert_eq!(
            compiled.program.instructions[0].payload[6],
            7 | (1_u64 << 32)
        );
        assert_eq!(compiled.program.instructions[0].payload[8], 100);
        assert_eq!(compiled.program.instructions[0].payload[9], 200);
        assert_eq!(
            compiled.program.instructions[1].opcode(),
            Some(InterpreterOpcode::Exit)
        );
    }

    #[test]
    fn deltanet_program_uses_real_opcode_and_counter() {
        let compiled =
            DecodeInterpreterProgram::compile_deltanet_recur(DecodeInterpreterDeltaNetParams {
                spec: DevicePtr(40),
            });
        assert_eq!(compiled.program.instructions.len(), 2);
        assert_eq!(compiled.program.counter_count, 2);
        assert_eq!(
            compiled.program.instructions[0].opcode(),
            Some(InterpreterOpcode::DeltaNetRecur)
        );
        assert_eq!(compiled.program.instructions[0].payload[0], 40);
        assert_eq!(
            compiled.program.instructions[1].opcode(),
            Some(InterpreterOpcode::Exit)
        );
    }

    #[test]
    fn attention_program_uses_real_opcode_and_counter() {
        let compiled = DecodeInterpreterProgram::compile_attention_decode_full(
            DecodeInterpreterAttentionParams {
                spec: DevicePtr(41),
            },
        );
        assert_eq!(compiled.program.instructions.len(), 2);
        assert_eq!(compiled.program.counter_count, 2);
        assert_eq!(
            compiled.program.instructions[0].opcode(),
            Some(InterpreterOpcode::AttnDecodeFull)
        );
        assert_eq!(compiled.program.instructions[0].payload[0], 41);
        assert_eq!(
            compiled.program.instructions[1].opcode(),
            Some(InterpreterOpcode::Exit)
        );
    }

    #[test]
    fn rope_attention_program_uses_real_opcodes_and_counters() {
        let compiled = DecodeInterpreterProgram::compile_rope_attention_decode(
            DecodeInterpreterRopeAttentionParams {
                rope: DecodeInterpreterRopeParams {
                    tokens: 1,
                    q_heads: 24,
                    kv_heads: 4,
                    head_dim: 256,
                    rope_dims: 64,
                    base_theta: 10000.0,
                    position_i32: 7,
                    use_scalar_position: true,
                    positions_i32: DevicePtr::NULL,
                    q_bf16: DevicePtr(100),
                    k_bf16: DevicePtr(200),
                    scalar_position_device_i32: DevicePtr::NULL,
                },
                attention: DecodeInterpreterAttentionParams {
                    spec: DevicePtr(41),
                },
            },
        );
        assert_eq!(compiled.program.instructions.len(), 3);
        assert_eq!(compiled.program.counter_count, 4);
        assert_eq!(
            compiled.program.instructions[0].opcode(),
            Some(InterpreterOpcode::RopePartial)
        );
        assert_eq!(
            compiled.program.instructions[1].opcode(),
            Some(InterpreterOpcode::AttnDecodeFull)
        );
        assert_eq!(compiled.program.instructions[1].deps[0].counter_id, 0);
        assert_eq!(compiled.program.instructions[1].deps[0].target, 1);
        assert_eq!(compiled.program.instructions[1].payload[0], 41);
        assert_eq!(
            compiled.program.instructions[2].opcode(),
            Some(InterpreterOpcode::Exit)
        );
    }

    #[test]
    fn full_attention_layer_program_uses_real_opcodes_and_counters() {
        let gemv = |m, k, a, b, c| DecodeInterpreterNvfp4GemvParams {
            m,
            k,
            alpha: 0.5,
            a_fp4: DevicePtr(a),
            a_scale_e4m3: DevicePtr(a + 1),
            b_fp4: DevicePtr(b),
            b_scale_e4m3: DevicePtr(b + 1),
            c_bf16: DevicePtr(c),
        };
        let norm = |rows, input, weight, output| DecodeInterpreterRmsNormBf16Params {
            rows,
            hidden: 256,
            eps: 1.0e-6,
            direct_weight: true,
            input_bf16: DevicePtr(input),
            weight_bf16: DevicePtr(weight),
            residual_bf16: DevicePtr::NULL,
            residual_out_bf16: DevicePtr::NULL,
            output_bf16: DevicePtr(output),
        };
        let compiled = DecodeInterpreterProgram::compile_full_attention_layer_decode(
            DecodeInterpreterFullAttentionLayerParams {
                q_proj: gemv(12288, 5120, 10, 100, 200),
                k_proj: gemv(1024, 5120, 20, 100, 201),
                v_proj: gemv(1024, 5120, 30, 100, 202),
                q_proj_deinterleave: DecodeInterpreterQProjDeinterleaveParams {
                    rows: 1,
                    heads: 24,
                    head_dim: 256,
                    input_bf16: DevicePtr(200),
                    output_bf16: DevicePtr(203),
                },
                q_norm: norm(24, 203, 40, 203),
                k_norm: norm(4, 201, 41, 201),
                rope: DecodeInterpreterRopeParams {
                    tokens: 1,
                    q_heads: 24,
                    kv_heads: 4,
                    head_dim: 256,
                    rope_dims: 64,
                    base_theta: 10000.0,
                    position_i32: 7,
                    use_scalar_position: true,
                    positions_i32: DevicePtr::NULL,
                    q_bf16: DevicePtr(203),
                    k_bf16: DevicePtr(201),
                    scalar_position_device_i32: DevicePtr::NULL,
                },
                attention: DecodeInterpreterAttentionParams {
                    spec: DevicePtr(300),
                },
                q_proj_gate: DecodeInterpreterQProjSigmoidGateParams {
                    rows: 1,
                    heads: 24,
                    head_dim: 256,
                    gate_bf16: DevicePtr(200),
                    input_bf16: DevicePtr(203),
                    output_bf16: DevicePtr(203),
                },
                o_input_quant: DecodeInterpreterNvfp4QuantizeParams {
                    values: 6144,
                    input_tensor_scale_f32: 1.25,
                    input_bf16: DevicePtr(203),
                    output_fp4: DevicePtr(100),
                    output_scale_e4m3: DevicePtr(101),
                    output_tensor_scale_f32: DevicePtr(102),
                },
                o_proj: gemv(5120, 6144, 50, 100, 204),
            },
        );
        assert_eq!(compiled.program.instructions.len(), 12);
        assert_eq!(compiled.program.counter_count, 22);
        let opcodes: Vec<_> = compiled
            .program
            .instructions
            .iter()
            .map(InterpreterInstruction::opcode)
            .collect();
        assert_eq!(
            opcodes,
            vec![
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::QProjDeinterleave),
                Some(InterpreterOpcode::RmsNormBf16),
                Some(InterpreterOpcode::RmsNormBf16),
                Some(InterpreterOpcode::RopePartial),
                Some(InterpreterOpcode::AttnDecodeFull),
                Some(InterpreterOpcode::QProjSigmoidGate),
                Some(InterpreterOpcode::Nvfp4Quantize),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Exit),
            ]
        );
        for (idx, instruction) in compiled.program.instructions.iter().take(11).enumerate() {
            assert_eq!(instruction.publishes_counter, (idx * 2) as u32);
            assert_eq!(instruction.arrival_counter, (idx * 2 + 1) as u32);
            if idx > 0 {
                assert_eq!(instruction.deps[0].counter_id, ((idx - 1) * 2) as u32);
                assert_eq!(instruction.deps[0].target, 1);
            }
        }
    }

    #[test]
    fn full_attention_input_layer_program_uses_real_opcodes_and_counters() {
        let gemv = |m, k, a, b, c| DecodeInterpreterNvfp4GemvParams {
            m,
            k,
            alpha: 0.5,
            a_fp4: DevicePtr(a),
            a_scale_e4m3: DevicePtr(a + 1),
            b_fp4: DevicePtr(b),
            b_scale_e4m3: DevicePtr(b + 1),
            c_bf16: DevicePtr(c),
        };
        let norm = |rows, input, weight, output| DecodeInterpreterRmsNormBf16Params {
            rows,
            hidden: 256,
            eps: 1.0e-6,
            direct_weight: true,
            input_bf16: DevicePtr(input),
            weight_bf16: DevicePtr(weight),
            residual_bf16: DevicePtr::NULL,
            residual_out_bf16: DevicePtr::NULL,
            output_bf16: DevicePtr(output),
        };
        let compiled = DecodeInterpreterProgram::compile_full_attention_input_layer_decode(
            DecodeInterpreterFullAttentionInputLayerParams {
                input_norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                    hidden: 5120,
                    eps: 1.0e-6,
                    input_tensor_scale_f32: 0.75,
                    input_bf16: DevicePtr(1),
                    weight_bf16: DevicePtr(2),
                    residual_bf16: DevicePtr(3),
                    residual_out_bf16: DevicePtr(4),
                    output_bf16: DevicePtr::NULL,
                    output_fp4: DevicePtr(100),
                    output_scale_e4m3: DevicePtr(101),
                    output_tensor_scale_f32: DevicePtr(102),
                },
                layer: DecodeInterpreterFullAttentionLayerParams {
                    q_proj: gemv(12288, 5120, 10, 100, 200),
                    k_proj: gemv(1024, 5120, 20, 100, 201),
                    v_proj: gemv(1024, 5120, 30, 100, 202),
                    q_proj_deinterleave: DecodeInterpreterQProjDeinterleaveParams {
                        rows: 1,
                        heads: 24,
                        head_dim: 256,
                        input_bf16: DevicePtr(200),
                        output_bf16: DevicePtr(203),
                    },
                    q_norm: norm(24, 203, 40, 203),
                    k_norm: norm(4, 201, 41, 201),
                    rope: DecodeInterpreterRopeParams {
                        tokens: 1,
                        q_heads: 24,
                        kv_heads: 4,
                        head_dim: 256,
                        rope_dims: 64,
                        base_theta: 10000.0,
                        position_i32: 7,
                        use_scalar_position: true,
                        positions_i32: DevicePtr::NULL,
                        q_bf16: DevicePtr(203),
                        k_bf16: DevicePtr(201),
                        scalar_position_device_i32: DevicePtr::NULL,
                    },
                    attention: DecodeInterpreterAttentionParams {
                        spec: DevicePtr(300),
                    },
                    q_proj_gate: DecodeInterpreterQProjSigmoidGateParams {
                        rows: 1,
                        heads: 24,
                        head_dim: 256,
                        gate_bf16: DevicePtr(200),
                        input_bf16: DevicePtr(203),
                        output_bf16: DevicePtr(203),
                    },
                    o_input_quant: DecodeInterpreterNvfp4QuantizeParams {
                        values: 6144,
                        input_tensor_scale_f32: 1.25,
                        input_bf16: DevicePtr(203),
                        output_fp4: DevicePtr(100),
                        output_scale_e4m3: DevicePtr(101),
                        output_tensor_scale_f32: DevicePtr(102),
                    },
                    o_proj: gemv(5120, 6144, 50, 100, 204),
                },
            },
        );
        assert_eq!(compiled.program.instructions.len(), 13);
        assert_eq!(compiled.program.counter_count, 24);
        let opcodes: Vec<_> = compiled
            .program
            .instructions
            .iter()
            .map(InterpreterInstruction::opcode)
            .collect();
        assert_eq!(
            opcodes,
            vec![
                Some(InterpreterOpcode::RmsNormNvfp4Quant),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::QProjDeinterleave),
                Some(InterpreterOpcode::RmsNormBf16),
                Some(InterpreterOpcode::RmsNormBf16),
                Some(InterpreterOpcode::RopePartial),
                Some(InterpreterOpcode::AttnDecodeFull),
                Some(InterpreterOpcode::QProjSigmoidGate),
                Some(InterpreterOpcode::Nvfp4Quantize),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Exit),
            ]
        );
        for (idx, instruction) in compiled.program.instructions.iter().take(12).enumerate() {
            assert_eq!(instruction.publishes_counter, (idx * 2) as u32);
            assert_eq!(instruction.arrival_counter, (idx * 2 + 1) as u32);
            if idx > 0 {
                assert_eq!(instruction.deps[0].counter_id, ((idx - 1) * 2) as u32);
                assert_eq!(instruction.deps[0].target, 1);
            }
        }
    }

    #[test]
    fn full_transformer_layer_program_uses_real_opcodes_and_counters() {
        let gemv = |m, k, a, b, c| DecodeInterpreterNvfp4GemvParams {
            m,
            k,
            alpha: 0.5,
            a_fp4: DevicePtr(a),
            a_scale_e4m3: DevicePtr(a + 1),
            b_fp4: DevicePtr(b),
            b_scale_e4m3: DevicePtr(b + 1),
            c_bf16: DevicePtr(c),
        };
        let norm = |rows, input, weight, output| DecodeInterpreterRmsNormBf16Params {
            rows,
            hidden: 256,
            eps: 1.0e-6,
            direct_weight: true,
            input_bf16: DevicePtr(input),
            weight_bf16: DevicePtr(weight),
            residual_bf16: DevicePtr::NULL,
            residual_out_bf16: DevicePtr::NULL,
            output_bf16: DevicePtr(output),
        };
        let compiled = DecodeInterpreterProgram::compile_full_transformer_layer_decode(
            DecodeInterpreterFullTransformerLayerParams {
                attention: DecodeInterpreterFullAttentionInputLayerParams {
                    input_norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                        hidden: 5120,
                        eps: 1.0e-6,
                        input_tensor_scale_f32: 0.75,
                        input_bf16: DevicePtr(1),
                        weight_bf16: DevicePtr(2),
                        residual_bf16: DevicePtr(3),
                        residual_out_bf16: DevicePtr(4),
                        output_bf16: DevicePtr::NULL,
                        output_fp4: DevicePtr(100),
                        output_scale_e4m3: DevicePtr(101),
                        output_tensor_scale_f32: DevicePtr(102),
                    },
                    layer: DecodeInterpreterFullAttentionLayerParams {
                        q_proj: gemv(12288, 5120, 10, 100, 200),
                        k_proj: gemv(1024, 5120, 20, 100, 201),
                        v_proj: gemv(1024, 5120, 30, 100, 202),
                        q_proj_deinterleave: DecodeInterpreterQProjDeinterleaveParams {
                            rows: 1,
                            heads: 24,
                            head_dim: 256,
                            input_bf16: DevicePtr(200),
                            output_bf16: DevicePtr(203),
                        },
                        q_norm: norm(24, 203, 40, 203),
                        k_norm: norm(4, 201, 41, 201),
                        rope: DecodeInterpreterRopeParams {
                            tokens: 1,
                            q_heads: 24,
                            kv_heads: 4,
                            head_dim: 256,
                            rope_dims: 64,
                            base_theta: 10000.0,
                            position_i32: 7,
                            use_scalar_position: true,
                            positions_i32: DevicePtr::NULL,
                            q_bf16: DevicePtr(203),
                            k_bf16: DevicePtr(201),
                            scalar_position_device_i32: DevicePtr::NULL,
                        },
                        attention: DecodeInterpreterAttentionParams {
                            spec: DevicePtr(300),
                        },
                        q_proj_gate: DecodeInterpreterQProjSigmoidGateParams {
                            rows: 1,
                            heads: 24,
                            head_dim: 256,
                            gate_bf16: DevicePtr(200),
                            input_bf16: DevicePtr(203),
                            output_bf16: DevicePtr(203),
                        },
                        o_input_quant: DecodeInterpreterNvfp4QuantizeParams {
                            values: 6144,
                            input_tensor_scale_f32: 1.25,
                            input_bf16: DevicePtr(203),
                            output_fp4: DevicePtr(100),
                            output_scale_e4m3: DevicePtr(101),
                            output_tensor_scale_f32: DevicePtr(102),
                        },
                        o_proj: gemv(5120, 6144, 50, 100, 204),
                    },
                },
                post: DecodeInterpreterNormMlpParams {
                    norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                        hidden: 5120,
                        eps: 1.0e-6,
                        input_tensor_scale_f32: 0.875,
                        input_bf16: DevicePtr(204),
                        weight_bf16: DevicePtr(205),
                        residual_bf16: DevicePtr(4),
                        residual_out_bf16: DevicePtr(4),
                        output_bf16: DevicePtr::NULL,
                        output_fp4: DevicePtr(100),
                        output_scale_e4m3: DevicePtr(101),
                        output_tensor_scale_f32: DevicePtr(102),
                    },
                    mlp: DecodeInterpreterMlpParams {
                        hidden: 5120,
                        intermediate: 27648,
                        input_fp4: DevicePtr(100),
                        input_scale_e4m3: DevicePtr(101),
                        gate_weight_fp4: DevicePtr(400),
                        gate_weight_scale_e4m3: DevicePtr(401),
                        gate_alpha: 0.5,
                        gate_out_bf16: DevicePtr(500),
                        up_weight_fp4: DevicePtr(402),
                        up_weight_scale_e4m3: DevicePtr(403),
                        up_alpha: 0.625,
                        up_out_bf16: DevicePtr(501),
                        swiglu_fp4: DevicePtr(100),
                        swiglu_scale_e4m3: DevicePtr(101),
                        swiglu_tensor_scale_f32: DevicePtr(102),
                        down_input_tensor_scale: 0.25,
                        down_weight_fp4: DevicePtr(404),
                        down_weight_scale_e4m3: DevicePtr(405),
                        down_alpha: 0.75,
                        output_bf16: DevicePtr(600),
                    },
                },
            },
        );
        assert_eq!(compiled.program.instructions.len(), 18);
        assert_eq!(compiled.program.counter_count, 34);
        let opcodes: Vec<_> = compiled
            .program
            .instructions
            .iter()
            .map(InterpreterInstruction::opcode)
            .collect();
        assert_eq!(
            opcodes,
            vec![
                Some(InterpreterOpcode::RmsNormNvfp4Quant),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::QProjDeinterleave),
                Some(InterpreterOpcode::RmsNormBf16),
                Some(InterpreterOpcode::RmsNormBf16),
                Some(InterpreterOpcode::RopePartial),
                Some(InterpreterOpcode::AttnDecodeFull),
                Some(InterpreterOpcode::QProjSigmoidGate),
                Some(InterpreterOpcode::Nvfp4Quantize),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::RmsNormNvfp4Quant),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::SwiGluNvfp4Quant),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Exit),
            ]
        );
        for (idx, instruction) in compiled.program.instructions.iter().take(17).enumerate() {
            assert_eq!(instruction.publishes_counter, (idx * 2) as u32);
            assert_eq!(instruction.arrival_counter, (idx * 2 + 1) as u32);
            if idx > 0 {
                assert_eq!(instruction.deps[0].counter_id, ((idx - 1) * 2) as u32);
                assert_eq!(instruction.deps[0].target, 1);
            }
        }
    }

    #[test]
    fn linear_attention_tail_program_uses_real_opcodes_and_counters() {
        let compiled = DecodeInterpreterProgram::compile_linear_attention_tail_decode(
            DecodeInterpreterLinearAttentionTailParams {
                norm: DecodeInterpreterRmsNormBf16Params {
                    rows: 48,
                    hidden: 128,
                    eps: 1.0e-6,
                    direct_weight: true,
                    input_bf16: DevicePtr(10),
                    weight_bf16: DevicePtr(11),
                    residual_bf16: DevicePtr::NULL,
                    residual_out_bf16: DevicePtr::NULL,
                    output_bf16: DevicePtr(12),
                },
                swiglu: DecodeInterpreterSwiGluBf16Params {
                    rows: 1,
                    intermediate: 6144,
                    gate_bf16: DevicePtr(13),
                    up_bf16: DevicePtr(12),
                    output_bf16: DevicePtr(14),
                },
                quant: DecodeInterpreterNvfp4QuantizeParams {
                    values: 6144,
                    input_tensor_scale_f32: 1.25,
                    input_bf16: DevicePtr(14),
                    output_fp4: DevicePtr(20),
                    output_scale_e4m3: DevicePtr(21),
                    output_tensor_scale_f32: DevicePtr(22),
                },
                out_proj: DecodeInterpreterNvfp4GemvParams {
                    m: 5120,
                    k: 6144,
                    alpha: 0.5,
                    a_fp4: DevicePtr(30),
                    a_scale_e4m3: DevicePtr(31),
                    b_fp4: DevicePtr(20),
                    b_scale_e4m3: DevicePtr(21),
                    c_bf16: DevicePtr(40),
                },
            },
        );
        assert_eq!(compiled.program.instructions.len(), 5);
        assert_eq!(compiled.program.counter_count, 8);
        assert_eq!(
            compiled.program.instructions[0].opcode(),
            Some(InterpreterOpcode::RmsNormBf16)
        );
        assert_eq!(
            compiled.program.instructions[1].opcode(),
            Some(InterpreterOpcode::SwiGluBf16)
        );
        assert_eq!(
            compiled.program.instructions[2].opcode(),
            Some(InterpreterOpcode::Nvfp4Quantize)
        );
        assert_eq!(
            compiled.program.instructions[3].opcode(),
            Some(InterpreterOpcode::Nvfp4Gemv)
        );
        assert_eq!(compiled.program.instructions[1].deps[0].counter_id, 0);
        assert_eq!(compiled.program.instructions[2].deps[0].counter_id, 2);
        assert_eq!(compiled.program.instructions[3].deps[0].counter_id, 4);
        assert_eq!(
            compiled.program.instructions[4].opcode(),
            Some(InterpreterOpcode::Exit)
        );
    }

    #[test]
    fn linear_attention_post_inproj_program_uses_real_opcodes_and_counters() {
        let tail = DecodeInterpreterLinearAttentionTailParams {
            norm: DecodeInterpreterRmsNormBf16Params {
                rows: 48,
                hidden: 128,
                eps: 1.0e-6,
                direct_weight: true,
                input_bf16: DevicePtr(10),
                weight_bf16: DevicePtr(11),
                residual_bf16: DevicePtr::NULL,
                residual_out_bf16: DevicePtr::NULL,
                output_bf16: DevicePtr(12),
            },
            swiglu: DecodeInterpreterSwiGluBf16Params {
                rows: 1,
                intermediate: 6144,
                gate_bf16: DevicePtr(13),
                up_bf16: DevicePtr(12),
                output_bf16: DevicePtr(14),
            },
            quant: DecodeInterpreterNvfp4QuantizeParams {
                values: 6144,
                input_tensor_scale_f32: 1.25,
                input_bf16: DevicePtr(14),
                output_fp4: DevicePtr(20),
                output_scale_e4m3: DevicePtr(21),
                output_tensor_scale_f32: DevicePtr(22),
            },
            out_proj: DecodeInterpreterNvfp4GemvParams {
                m: 5120,
                k: 6144,
                alpha: 0.5,
                a_fp4: DevicePtr(30),
                a_scale_e4m3: DevicePtr(31),
                b_fp4: DevicePtr(20),
                b_scale_e4m3: DevicePtr(21),
                c_bf16: DevicePtr(40),
            },
        };
        let compiled = DecodeInterpreterProgram::compile_linear_attention_post_inproj_decode(
            DecodeInterpreterLinearAttentionPostInProjParams {
                conv_gdn: DecodeInterpreterConv1dGdnGateFusedParams {
                    channels: 10240,
                    kernel_size: 4,
                    conv_input_bf16: DevicePtr(100),
                    conv_history_bf16: DevicePtr(101),
                    conv_weight_bf16: DevicePtr(102),
                    conv_output_bf16: DevicePtr(103),
                    heads: 48,
                    gdn_a_bf16: DevicePtr(104),
                    gdn_b_bf16: DevicePtr(105),
                    gdn_a_log_bf16: DevicePtr(106),
                    gdn_dt_bias_bf16: DevicePtr(107),
                    gate_f32: DevicePtr(108),
                    beta_f32: DevicePtr(109),
                },
                deltanet: DecodeInterpreterDeltaNetParams {
                    spec: DevicePtr(200),
                },
                tail,
            },
        );
        assert_eq!(compiled.program.instructions.len(), 7);
        assert_eq!(compiled.program.counter_count, 12);
        let opcodes: Vec<_> = compiled
            .program
            .instructions
            .iter()
            .map(InterpreterInstruction::opcode)
            .collect();
        assert_eq!(
            opcodes,
            vec![
                Some(InterpreterOpcode::Conv1dGdnGateFused),
                Some(InterpreterOpcode::DeltaNetRecur),
                Some(InterpreterOpcode::RmsNormBf16),
                Some(InterpreterOpcode::SwiGluBf16),
                Some(InterpreterOpcode::Nvfp4Quantize),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Exit),
            ]
        );
        for (idx, instruction) in compiled.program.instructions.iter().take(6).enumerate() {
            assert_eq!(instruction.publishes_counter, (idx * 2) as u32);
            assert_eq!(instruction.arrival_counter, (idx * 2 + 1) as u32);
            if idx > 0 {
                assert_eq!(instruction.deps[0].counter_id, ((idx - 1) * 2) as u32);
                assert_eq!(instruction.deps[0].target, 1);
            }
        }
    }

    #[test]
    fn linear_attention_layer_program_uses_real_opcodes_and_counters() {
        let post_inproj = DecodeInterpreterLinearAttentionPostInProjParams {
            conv_gdn: DecodeInterpreterConv1dGdnGateFusedParams {
                channels: 10240,
                kernel_size: 4,
                conv_input_bf16: DevicePtr(100),
                conv_history_bf16: DevicePtr(101),
                conv_weight_bf16: DevicePtr(102),
                conv_output_bf16: DevicePtr(103),
                heads: 48,
                gdn_a_bf16: DevicePtr(104),
                gdn_b_bf16: DevicePtr(105),
                gdn_a_log_bf16: DevicePtr(106),
                gdn_dt_bias_bf16: DevicePtr(107),
                gate_f32: DevicePtr(108),
                beta_f32: DevicePtr(109),
            },
            deltanet: DecodeInterpreterDeltaNetParams {
                spec: DevicePtr(200),
            },
            tail: DecodeInterpreterLinearAttentionTailParams {
                norm: DecodeInterpreterRmsNormBf16Params {
                    rows: 48,
                    hidden: 128,
                    eps: 1.0e-6,
                    direct_weight: true,
                    input_bf16: DevicePtr(10),
                    weight_bf16: DevicePtr(11),
                    residual_bf16: DevicePtr::NULL,
                    residual_out_bf16: DevicePtr::NULL,
                    output_bf16: DevicePtr(12),
                },
                swiglu: DecodeInterpreterSwiGluBf16Params {
                    rows: 1,
                    intermediate: 6144,
                    gate_bf16: DevicePtr(13),
                    up_bf16: DevicePtr(12),
                    output_bf16: DevicePtr(14),
                },
                quant: DecodeInterpreterNvfp4QuantizeParams {
                    values: 6144,
                    input_tensor_scale_f32: 1.25,
                    input_bf16: DevicePtr(14),
                    output_fp4: DevicePtr(20),
                    output_scale_e4m3: DevicePtr(21),
                    output_tensor_scale_f32: DevicePtr(22),
                },
                out_proj: DecodeInterpreterNvfp4GemvParams {
                    m: 5120,
                    k: 6144,
                    alpha: 0.5,
                    a_fp4: DevicePtr(30),
                    a_scale_e4m3: DevicePtr(31),
                    b_fp4: DevicePtr(20),
                    b_scale_e4m3: DevicePtr(21),
                    c_bf16: DevicePtr(40),
                },
            },
        };
        let compiled = DecodeInterpreterProgram::compile_linear_attention_layer_decode(
            DecodeInterpreterLinearAttentionLayerParams {
                in_proj: DecodeInterpreterNvfp4GemvParams {
                    m: 16640,
                    k: 5120,
                    alpha: 0.25,
                    a_fp4: DevicePtr(300),
                    a_scale_e4m3: DevicePtr(301),
                    b_fp4: DevicePtr(302),
                    b_scale_e4m3: DevicePtr(303),
                    c_bf16: DevicePtr(100),
                },
                post_inproj,
            },
        );
        assert_eq!(compiled.program.instructions.len(), 8);
        assert_eq!(compiled.program.counter_count, 14);
        let opcodes: Vec<_> = compiled
            .program
            .instructions
            .iter()
            .map(InterpreterInstruction::opcode)
            .collect();
        assert_eq!(
            opcodes,
            vec![
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Conv1dGdnGateFused),
                Some(InterpreterOpcode::DeltaNetRecur),
                Some(InterpreterOpcode::RmsNormBf16),
                Some(InterpreterOpcode::SwiGluBf16),
                Some(InterpreterOpcode::Nvfp4Quantize),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Exit),
            ]
        );
        for (idx, instruction) in compiled.program.instructions.iter().take(7).enumerate() {
            assert_eq!(instruction.publishes_counter, (idx * 2) as u32);
            assert_eq!(instruction.arrival_counter, (idx * 2 + 1) as u32);
            if idx > 0 {
                assert_eq!(instruction.deps[0].counter_id, ((idx - 1) * 2) as u32);
                assert_eq!(instruction.deps[0].target, 1);
            }
        }
    }

    #[test]
    fn linear_attention_input_layer_program_uses_real_opcodes_and_counters() {
        let compiled = DecodeInterpreterProgram::compile_linear_attention_input_layer_decode(
            DecodeInterpreterLinearAttentionInputLayerParams {
                input_norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                    hidden: 5120,
                    eps: 1.0e-6,
                    input_tensor_scale_f32: 0.75,
                    input_bf16: DevicePtr(1),
                    weight_bf16: DevicePtr(2),
                    residual_bf16: DevicePtr(3),
                    residual_out_bf16: DevicePtr(4),
                    output_bf16: DevicePtr::NULL,
                    output_fp4: DevicePtr(5),
                    output_scale_e4m3: DevicePtr(6),
                    output_tensor_scale_f32: DevicePtr(7),
                },
                layer: DecodeInterpreterLinearAttentionLayerParams {
                    in_proj: DecodeInterpreterNvfp4GemvParams {
                        m: 16640,
                        k: 5120,
                        alpha: 0.25,
                        a_fp4: DevicePtr(300),
                        a_scale_e4m3: DevicePtr(301),
                        b_fp4: DevicePtr(5),
                        b_scale_e4m3: DevicePtr(6),
                        c_bf16: DevicePtr(100),
                    },
                    post_inproj: DecodeInterpreterLinearAttentionPostInProjParams {
                        conv_gdn: DecodeInterpreterConv1dGdnGateFusedParams {
                            channels: 10240,
                            kernel_size: 4,
                            conv_input_bf16: DevicePtr(100),
                            conv_history_bf16: DevicePtr(101),
                            conv_weight_bf16: DevicePtr(102),
                            conv_output_bf16: DevicePtr(103),
                            heads: 48,
                            gdn_a_bf16: DevicePtr(104),
                            gdn_b_bf16: DevicePtr(105),
                            gdn_a_log_bf16: DevicePtr(106),
                            gdn_dt_bias_bf16: DevicePtr(107),
                            gate_f32: DevicePtr(108),
                            beta_f32: DevicePtr(109),
                        },
                        deltanet: DecodeInterpreterDeltaNetParams {
                            spec: DevicePtr(200),
                        },
                        tail: DecodeInterpreterLinearAttentionTailParams {
                            norm: DecodeInterpreterRmsNormBf16Params {
                                rows: 48,
                                hidden: 128,
                                eps: 1.0e-6,
                                direct_weight: true,
                                input_bf16: DevicePtr(10),
                                weight_bf16: DevicePtr(11),
                                residual_bf16: DevicePtr::NULL,
                                residual_out_bf16: DevicePtr::NULL,
                                output_bf16: DevicePtr(12),
                            },
                            swiglu: DecodeInterpreterSwiGluBf16Params {
                                rows: 1,
                                intermediate: 6144,
                                gate_bf16: DevicePtr(13),
                                up_bf16: DevicePtr(12),
                                output_bf16: DevicePtr(14),
                            },
                            quant: DecodeInterpreterNvfp4QuantizeParams {
                                values: 6144,
                                input_tensor_scale_f32: 1.25,
                                input_bf16: DevicePtr(14),
                                output_fp4: DevicePtr(20),
                                output_scale_e4m3: DevicePtr(21),
                                output_tensor_scale_f32: DevicePtr(22),
                            },
                            out_proj: DecodeInterpreterNvfp4GemvParams {
                                m: 5120,
                                k: 6144,
                                alpha: 0.5,
                                a_fp4: DevicePtr(30),
                                a_scale_e4m3: DevicePtr(31),
                                b_fp4: DevicePtr(20),
                                b_scale_e4m3: DevicePtr(21),
                                c_bf16: DevicePtr(40),
                            },
                        },
                    },
                },
            },
        );
        assert_eq!(compiled.program.instructions.len(), 9);
        assert_eq!(compiled.program.counter_count, 16);
        let opcodes: Vec<_> = compiled
            .program
            .instructions
            .iter()
            .map(InterpreterInstruction::opcode)
            .collect();
        assert_eq!(
            opcodes,
            vec![
                Some(InterpreterOpcode::RmsNormNvfp4Quant),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Conv1dGdnGateFused),
                Some(InterpreterOpcode::DeltaNetRecur),
                Some(InterpreterOpcode::RmsNormBf16),
                Some(InterpreterOpcode::SwiGluBf16),
                Some(InterpreterOpcode::Nvfp4Quantize),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Exit),
            ]
        );
        for (idx, instruction) in compiled.program.instructions.iter().take(8).enumerate() {
            assert_eq!(instruction.publishes_counter, (idx * 2) as u32);
            assert_eq!(instruction.arrival_counter, (idx * 2 + 1) as u32);
            if idx > 0 {
                assert_eq!(instruction.deps[0].counter_id, ((idx - 1) * 2) as u32);
                assert_eq!(instruction.deps[0].target, 1);
            }
        }
    }

    #[test]
    fn linear_transformer_layer_program_uses_real_opcodes_and_counters() {
        let compiled = DecodeInterpreterProgram::compile_linear_transformer_layer_decode(
            DecodeInterpreterLinearTransformerLayerParams {
                attention: DecodeInterpreterLinearAttentionInputLayerParams {
                    input_norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                        hidden: 5120,
                        eps: 1.0e-6,
                        input_tensor_scale_f32: 0.75,
                        input_bf16: DevicePtr(1),
                        weight_bf16: DevicePtr(2),
                        residual_bf16: DevicePtr(3),
                        residual_out_bf16: DevicePtr(4),
                        output_bf16: DevicePtr::NULL,
                        output_fp4: DevicePtr(5),
                        output_scale_e4m3: DevicePtr(6),
                        output_tensor_scale_f32: DevicePtr(7),
                    },
                    layer: DecodeInterpreterLinearAttentionLayerParams {
                        in_proj: DecodeInterpreterNvfp4GemvParams {
                            m: 16640,
                            k: 5120,
                            alpha: 0.25,
                            a_fp4: DevicePtr(300),
                            a_scale_e4m3: DevicePtr(301),
                            b_fp4: DevicePtr(5),
                            b_scale_e4m3: DevicePtr(6),
                            c_bf16: DevicePtr(100),
                        },
                        post_inproj: DecodeInterpreterLinearAttentionPostInProjParams {
                            conv_gdn: DecodeInterpreterConv1dGdnGateFusedParams {
                                channels: 10240,
                                kernel_size: 4,
                                conv_input_bf16: DevicePtr(100),
                                conv_history_bf16: DevicePtr(101),
                                conv_weight_bf16: DevicePtr(102),
                                conv_output_bf16: DevicePtr(103),
                                heads: 48,
                                gdn_a_bf16: DevicePtr(104),
                                gdn_b_bf16: DevicePtr(105),
                                gdn_a_log_bf16: DevicePtr(106),
                                gdn_dt_bias_bf16: DevicePtr(107),
                                gate_f32: DevicePtr(108),
                                beta_f32: DevicePtr(109),
                            },
                            deltanet: DecodeInterpreterDeltaNetParams {
                                spec: DevicePtr(200),
                            },
                            tail: DecodeInterpreterLinearAttentionTailParams {
                                norm: DecodeInterpreterRmsNormBf16Params {
                                    rows: 48,
                                    hidden: 128,
                                    eps: 1.0e-6,
                                    direct_weight: true,
                                    input_bf16: DevicePtr(10),
                                    weight_bf16: DevicePtr(11),
                                    residual_bf16: DevicePtr::NULL,
                                    residual_out_bf16: DevicePtr::NULL,
                                    output_bf16: DevicePtr(12),
                                },
                                swiglu: DecodeInterpreterSwiGluBf16Params {
                                    rows: 1,
                                    intermediate: 6144,
                                    gate_bf16: DevicePtr(13),
                                    up_bf16: DevicePtr(12),
                                    output_bf16: DevicePtr(14),
                                },
                                quant: DecodeInterpreterNvfp4QuantizeParams {
                                    values: 6144,
                                    input_tensor_scale_f32: 1.25,
                                    input_bf16: DevicePtr(14),
                                    output_fp4: DevicePtr(20),
                                    output_scale_e4m3: DevicePtr(21),
                                    output_tensor_scale_f32: DevicePtr(22),
                                },
                                out_proj: DecodeInterpreterNvfp4GemvParams {
                                    m: 5120,
                                    k: 6144,
                                    alpha: 0.5,
                                    a_fp4: DevicePtr(30),
                                    a_scale_e4m3: DevicePtr(31),
                                    b_fp4: DevicePtr(20),
                                    b_scale_e4m3: DevicePtr(21),
                                    c_bf16: DevicePtr(40),
                                },
                            },
                        },
                    },
                },
                post: DecodeInterpreterNormMlpParams {
                    norm: DecodeInterpreterRmsNormNvfp4QuantParams {
                        hidden: 5120,
                        eps: 1.0e-6,
                        input_tensor_scale_f32: 0.875,
                        input_bf16: DevicePtr(40),
                        weight_bf16: DevicePtr(41),
                        residual_bf16: DevicePtr(4),
                        residual_out_bf16: DevicePtr(4),
                        output_bf16: DevicePtr::NULL,
                        output_fp4: DevicePtr(42),
                        output_scale_e4m3: DevicePtr(43),
                        output_tensor_scale_f32: DevicePtr(44),
                    },
                    mlp: DecodeInterpreterMlpParams {
                        hidden: 5120,
                        intermediate: 27648,
                        input_fp4: DevicePtr(42),
                        input_scale_e4m3: DevicePtr(43),
                        gate_weight_fp4: DevicePtr(50),
                        gate_weight_scale_e4m3: DevicePtr(51),
                        gate_alpha: 0.5,
                        gate_out_bf16: DevicePtr(60),
                        up_weight_fp4: DevicePtr(52),
                        up_weight_scale_e4m3: DevicePtr(53),
                        up_alpha: 0.625,
                        up_out_bf16: DevicePtr(61),
                        swiglu_fp4: DevicePtr(42),
                        swiglu_scale_e4m3: DevicePtr(43),
                        swiglu_tensor_scale_f32: DevicePtr(44),
                        down_input_tensor_scale: 0.25,
                        down_weight_fp4: DevicePtr(54),
                        down_weight_scale_e4m3: DevicePtr(55),
                        down_alpha: 0.75,
                        output_bf16: DevicePtr(70),
                    },
                },
            },
        );
        assert_eq!(compiled.program.instructions.len(), 14);
        assert_eq!(compiled.program.counter_count, 26);
        let opcodes: Vec<_> = compiled
            .program
            .instructions
            .iter()
            .map(InterpreterInstruction::opcode)
            .collect();
        assert_eq!(
            opcodes,
            vec![
                Some(InterpreterOpcode::RmsNormNvfp4Quant),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Conv1dGdnGateFused),
                Some(InterpreterOpcode::DeltaNetRecur),
                Some(InterpreterOpcode::RmsNormBf16),
                Some(InterpreterOpcode::SwiGluBf16),
                Some(InterpreterOpcode::Nvfp4Quantize),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::RmsNormNvfp4Quant),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::SwiGluNvfp4Quant),
                Some(InterpreterOpcode::Nvfp4Gemv),
                Some(InterpreterOpcode::Exit),
            ]
        );
        for (idx, instruction) in compiled.program.instructions.iter().take(13).enumerate() {
            assert_eq!(instruction.publishes_counter, (idx * 2) as u32);
            assert_eq!(instruction.arrival_counter, (idx * 2 + 1) as u32);
            if idx > 0 {
                assert_eq!(instruction.deps[0].counter_id, ((idx - 1) * 2) as u32);
                assert_eq!(instruction.deps[0].target, 1);
            }
        }
    }

    #[test]
    fn topology_placeholder_program_routes_everything_to_fallback() {
        let topology = ModelTopology::expected_qwen36_text_mtp();
        let compiled = DecodeInterpreterProgram::compile(&topology);
        for instruction in compiled
            .program
            .instructions
            .iter()
            .take(compiled.program.instructions.len().saturating_sub(1))
        {
            assert_eq!(
                instruction.opcode(),
                Some(InterpreterOpcode::FallbackTrampoline)
            );
            assert!(InterpreterOpcode::from_code(instruction.payload[2] as u16).is_some());
        }
    }
}
