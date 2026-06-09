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
