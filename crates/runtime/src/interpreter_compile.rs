#[cfg(feature = "cuda")]
use qwen36_fp4_core::Result;
use qwen36_fp4_core::{LayerType, ModelTopology};
#[cfg(feature = "cuda")]
use qwen36_fp4_kernels::{CudaDeviceBuffer, KernelBackend};
use qwen36_fp4_kernels::{
    InterpreterInstruction, InterpreterOpcode, InterpreterProgram, InterpreterProgramSpec,
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
