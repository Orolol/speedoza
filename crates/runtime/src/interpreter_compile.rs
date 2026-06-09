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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeInterpreterDeltaNetParams {
    pub spec: DevicePtr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeInterpreterAttentionParams {
    pub spec: DevicePtr,
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
