#[cfg(feature = "cuda")]
use qwen36_fp4_core::Result;
use qwen36_fp4_core::{LayerType, ModelTopology};
use qwen36_fp4_kernels::{
    interpreter_opcodes_enabled_from_env, InterpreterInstruction, InterpreterOpcode,
    InterpreterProgram, InterpreterProgramSpec,
};
#[cfg(feature = "cuda")]
use qwen36_fp4_kernels::{CudaDeviceBuffer, KernelBackend};

/// Host-side compiler for the decode interpreter instruction stream.
///
/// Stage 0 emits the same static shape as the planned megakernel, but opcodes
/// not enabled by `QWEN36_INTERPRETER_OPCODES_ENABLED` are rewritten to
/// `FALLBACK_TRAMPOLINE`. The CUDA interpreter currently routes all non-EXIT
/// opcodes through that no-op fallback; later stages can port one opcode at a
/// time without changing this program ABI.
#[derive(Debug, Clone)]
pub struct DecodeInterpreterProgram {
    pub program: InterpreterProgram,
}

impl DecodeInterpreterProgram {
    pub fn compile(topology: &ModelTopology) -> Self {
        let enabled = interpreter_opcodes_enabled_from_env();
        let mut compiler = DecodeInterpreterCompiler {
            program: InterpreterProgram::new(),
            next_counter: 0,
            last_counter: None,
        };

        for (layer_idx, layer_type) in topology.layer_types.iter().copied().enumerate() {
            compiler.push(layer_idx, 0, InterpreterOpcode::RmsNormNvfp4Quant, &enabled);
            match layer_type {
                LayerType::LinearAttention => {
                    compiler.push(layer_idx, 1, InterpreterOpcode::Nvfp4Gemv, &enabled);
                    compiler.push(layer_idx, 2, InterpreterOpcode::DeltaNetRecur, &enabled);
                    compiler.push(layer_idx, 3, InterpreterOpcode::Nvfp4Gemv, &enabled);
                }
                LayerType::FullAttention => {
                    compiler.push(layer_idx, 1, InterpreterOpcode::Nvfp4Gemv, &enabled);
                    compiler.push(layer_idx, 2, InterpreterOpcode::RopePartial, &enabled);
                    compiler.push(layer_idx, 3, InterpreterOpcode::AttnDecodeFull, &enabled);
                    compiler.push(layer_idx, 4, InterpreterOpcode::Nvfp4Gemv, &enabled);
                }
            }
            compiler.push(layer_idx, 5, InterpreterOpcode::ResidualAdd, &enabled);
            compiler.push(layer_idx, 6, InterpreterOpcode::RmsNormNvfp4Quant, &enabled);
            compiler.push(layer_idx, 7, InterpreterOpcode::Nvfp4Gemv, &enabled);
            compiler.push(layer_idx, 8, InterpreterOpcode::SwiGluNvfp4Quant, &enabled);
            compiler.push(layer_idx, 9, InterpreterOpcode::Nvfp4Gemv, &enabled);
        }

        compiler.push(
            topology.num_hidden_layers,
            0,
            InterpreterOpcode::RmsNormNvfp4Quant,
            &enabled,
        );
        compiler.push(
            topology.num_hidden_layers,
            1,
            InterpreterOpcode::LmHeadTiled,
            &enabled,
        );

        Self {
            program: compiler.program.finish(),
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
fn instructions_as_bytes(instructions: &[InterpreterInstruction]) -> &[u8] {
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
    fn push(
        &mut self,
        layer_idx: usize,
        op_ordinal: u64,
        opcode: InterpreterOpcode,
        enabled: &qwen36_fp4_kernels::InterpreterOpcodeSet,
    ) {
        let routed_opcode = if enabled.contains(opcode) {
            opcode
        } else {
            InterpreterOpcode::FallbackTrampoline
        };
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
}
