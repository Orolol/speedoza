use serde::{Deserialize, Serialize};

use crate::backend::DevicePtr;

pub const INTERPRETER_MAX_DEPS: usize = 4;
pub const INTERPRETER_PAYLOAD_U64S: usize = 12;
pub const INTERPRETER_OPCODE_COUNT: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InterpreterOpcode {
    Exit,
    FallbackTrampoline,
    RmsNormNvfp4Quant,
    Nvfp4Gemv,
    SwiGluNvfp4Quant,
    RopePartial,
    AttnDecodeFull,
    DeltaNetRecur,
    ResidualAdd,
    LmHeadTiled,
}

impl InterpreterOpcode {
    pub const ALL: [Self; 10] = [
        Self::Exit,
        Self::FallbackTrampoline,
        Self::RmsNormNvfp4Quant,
        Self::Nvfp4Gemv,
        Self::SwiGluNvfp4Quant,
        Self::RopePartial,
        Self::AttnDecodeFull,
        Self::DeltaNetRecur,
        Self::ResidualAdd,
        Self::LmHeadTiled,
    ];

    pub fn code(self) -> u16 {
        match self {
            Self::Exit => 0,
            Self::FallbackTrampoline => 1,
            Self::RmsNormNvfp4Quant => 2,
            Self::Nvfp4Gemv => 3,
            Self::SwiGluNvfp4Quant => 4,
            Self::RopePartial => 5,
            Self::AttnDecodeFull => 6,
            Self::DeltaNetRecur => 7,
            Self::ResidualAdd => 8,
            Self::LmHeadTiled => 9,
        }
    }

    pub fn from_code(code: u16) -> Option<Self> {
        Some(match code {
            0 => Self::Exit,
            1 => Self::FallbackTrampoline,
            2 => Self::RmsNormNvfp4Quant,
            3 => Self::Nvfp4Gemv,
            4 => Self::SwiGluNvfp4Quant,
            5 => Self::RopePartial,
            6 => Self::AttnDecodeFull,
            7 => Self::DeltaNetRecur,
            8 => Self::ResidualAdd,
            9 => Self::LmHeadTiled,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Exit => "EXIT",
            Self::FallbackTrampoline => "FALLBACK_TRAMPOLINE",
            Self::RmsNormNvfp4Quant => "RMSNORM_NVFP4_QUANT",
            Self::Nvfp4Gemv => "NVFP4_GEMV",
            Self::SwiGluNvfp4Quant => "SWIGLU_NVFP4_QUANT",
            Self::RopePartial => "ROPE_PARTIAL",
            Self::AttnDecodeFull => "ATTN_DECODE_FULL",
            Self::DeltaNetRecur => "DELTANET_RECUR",
            Self::ResidualAdd => "RESIDUAL_ADD",
            Self::LmHeadTiled => "LM_HEAD_TILED",
        }
    }

    pub fn parse_name(name: &str) -> Option<Self> {
        let normalized = name.trim().to_ascii_uppercase();
        Self::ALL
            .into_iter()
            .find(|opcode| opcode.name() == normalized)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct InterpreterDep {
    pub counter_id: u32,
    pub target: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterpreterInstruction {
    pub opcode: u16,
    pub flags: u16,
    pub dep_count: u16,
    pub reserved: u16,
    pub publishes_counter: u32,
    pub publish_value: u32,
    pub arrival_counter: u32,
    pub deps: [InterpreterDep; INTERPRETER_MAX_DEPS],
    pub payload: [u64; INTERPRETER_PAYLOAD_U64S],
}

impl Default for InterpreterInstruction {
    fn default() -> Self {
        Self {
            opcode: InterpreterOpcode::Exit.code(),
            flags: 0,
            dep_count: 0,
            reserved: 0,
            publishes_counter: 0,
            publish_value: 0,
            arrival_counter: u32::MAX,
            deps: [InterpreterDep::default(); INTERPRETER_MAX_DEPS],
            payload: [0; INTERPRETER_PAYLOAD_U64S],
        }
    }
}

impl InterpreterInstruction {
    pub fn exit() -> Self {
        Self::default()
    }

    pub fn new(opcode: InterpreterOpcode) -> Self {
        Self {
            opcode: opcode.code(),
            ..Self::default()
        }
    }

    pub fn fallback_trampoline() -> Self {
        Self::new(InterpreterOpcode::FallbackTrampoline)
    }

    pub fn with_publish(mut self, counter_id: u32, value: u32) -> Self {
        self.publishes_counter = counter_id;
        self.publish_value = value;
        self
    }

    pub fn with_arrival_counter(mut self, counter_id: u32) -> Self {
        self.arrival_counter = counter_id;
        self
    }

    pub fn with_dep(mut self, counter_id: u32, target: u32) -> Self {
        let idx = usize::from(self.dep_count);
        assert!(
            idx < INTERPRETER_MAX_DEPS,
            "interpreter instruction dependency overflow"
        );
        self.deps[idx] = InterpreterDep { counter_id, target };
        self.dep_count += 1;
        self
    }

    pub fn opcode(&self) -> Option<InterpreterOpcode> {
        InterpreterOpcode::from_code(self.opcode)
    }

    pub fn residual_add(
        values: usize,
        input_bf16: DevicePtr,
        residual_bf16: DevicePtr,
        output_bf16: DevicePtr,
    ) -> Self {
        let mut instruction = Self::new(InterpreterOpcode::ResidualAdd);
        instruction.payload[0] = values as u64;
        instruction.payload[1] = input_bf16.0;
        instruction.payload[2] = residual_bf16.0;
        instruction.payload[3] = output_bf16.0;
        instruction
    }

    pub fn rmsnorm_nvfp4_quant(
        hidden: usize,
        eps: f32,
        input_tensor_scale_f32: f32,
        input_bf16: DevicePtr,
        weight_bf16: DevicePtr,
        residual_bf16: DevicePtr,
        residual_out_bf16: DevicePtr,
        output_bf16: DevicePtr,
        output_fp4: DevicePtr,
        output_scale_e4m3: DevicePtr,
        output_tensor_scale_f32: DevicePtr,
    ) -> Self {
        let mut instruction = Self::new(InterpreterOpcode::RmsNormNvfp4Quant);
        instruction.payload[0] = hidden as u64;
        instruction.payload[1] = input_bf16.0;
        instruction.payload[2] = weight_bf16.0;
        instruction.payload[3] = residual_bf16.0;
        instruction.payload[4] = residual_out_bf16.0;
        instruction.payload[5] = output_bf16.0;
        instruction.payload[6] = output_fp4.0;
        instruction.payload[7] = output_scale_e4m3.0;
        instruction.payload[8] = output_tensor_scale_f32.0;
        instruction.payload[9] =
            u64::from(eps.to_bits()) | (u64::from(input_tensor_scale_f32.to_bits()) << 32);
        instruction
    }

    pub fn swiglu_nvfp4_quant(
        intermediate: usize,
        input_tensor_scale_f32: f32,
        gate_bf16: DevicePtr,
        up_bf16: DevicePtr,
        output_fp4: DevicePtr,
        output_scale_e4m3: DevicePtr,
        output_tensor_scale_f32: DevicePtr,
    ) -> Self {
        let mut instruction = Self::new(InterpreterOpcode::SwiGluNvfp4Quant);
        instruction.payload[0] = intermediate as u64;
        instruction.payload[1] = gate_bf16.0;
        instruction.payload[2] = up_bf16.0;
        instruction.payload[3] = output_fp4.0;
        instruction.payload[4] = output_scale_e4m3.0;
        instruction.payload[5] = output_tensor_scale_f32.0;
        instruction.payload[6] = u64::from(input_tensor_scale_f32.to_bits());
        instruction
    }

    pub fn nvfp4_gemv(
        m: usize,
        k: usize,
        alpha: f32,
        a_fp4: DevicePtr,
        a_scale: DevicePtr,
        b_fp4: DevicePtr,
        b_scale: DevicePtr,
        c_bf16: DevicePtr,
    ) -> Self {
        let mut instruction = Self::new(InterpreterOpcode::Nvfp4Gemv);
        instruction.payload[0] = m as u64;
        instruction.payload[1] = k as u64;
        instruction.payload[2] = a_fp4.0;
        instruction.payload[3] = a_scale.0;
        instruction.payload[4] = b_fp4.0;
        instruction.payload[5] = b_scale.0;
        instruction.payload[6] = c_bf16.0;
        instruction.payload[7] = u64::from(alpha.to_bits());
        instruction
    }

    pub fn deltanet_recur_spec(spec: DevicePtr) -> Self {
        let mut instruction = Self::new(InterpreterOpcode::DeltaNetRecur);
        instruction.payload[0] = spec.0;
        instruction
    }

    pub fn attn_decode_full_spec(spec: DevicePtr) -> Self {
        let mut instruction = Self::new(InterpreterOpcode::AttnDecodeFull);
        instruction.payload[0] = spec.0;
        instruction
    }

    pub fn rope_partial(
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rope_dims: usize,
        base_theta: f64,
        position_i32: i32,
        use_scalar_position: bool,
        positions_i32: DevicePtr,
        q_bf16: DevicePtr,
        k_bf16: DevicePtr,
        scalar_position_device_i32: DevicePtr,
    ) -> Self {
        let mut instruction = Self::new(InterpreterOpcode::RopePartial);
        instruction.payload[0] = tokens as u64;
        instruction.payload[1] = q_heads as u64;
        instruction.payload[2] = kv_heads as u64;
        instruction.payload[3] = head_dim as u64;
        instruction.payload[4] = rope_dims as u64;
        instruction.payload[5] = u64::from((base_theta as f32).to_bits());
        instruction.payload[6] =
            u64::from(position_i32 as u32) | (u64::from(u32::from(use_scalar_position)) << 32);
        instruction.payload[7] = positions_i32.0;
        instruction.payload[8] = q_bf16.0;
        instruction.payload[9] = k_bf16.0;
        instruction.payload[10] = scalar_position_device_i32.0;
        instruction
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterpreterProgramSpec {
    pub instructions: DevicePtr,
    pub instruction_count: usize,
    pub counters_i32: DevicePtr,
    pub counter_count: usize,
    pub cta_count: u32,
    pub flags: u32,
}

impl InterpreterProgramSpec {
    pub fn validate(self) -> bool {
        self.instructions != DevicePtr::NULL
            && self.instruction_count > 0
            && self.counters_i32 != DevicePtr::NULL
            && self.counter_count > 0
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InterpreterProgram {
    pub instructions: Vec<InterpreterInstruction>,
    pub counter_count: usize,
}

impl InterpreterProgram {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, instruction: InterpreterInstruction) {
        self.counter_count = self
            .counter_count
            .max(instruction.publishes_counter as usize + 1);
        if instruction.arrival_counter != u32::MAX {
            self.counter_count = self
                .counter_count
                .max(instruction.arrival_counter as usize + 1);
        }
        for dep in instruction
            .deps
            .iter()
            .take(usize::from(instruction.dep_count))
        {
            self.counter_count = self.counter_count.max(dep.counter_id as usize + 1);
        }
        self.instructions.push(instruction);
    }

    pub fn finish(mut self) -> Self {
        if !matches!(
            self.instructions
                .last()
                .and_then(InterpreterInstruction::opcode),
            Some(InterpreterOpcode::Exit)
        ) {
            self.instructions.push(InterpreterInstruction::exit());
        }
        self.counter_count = self.counter_count.max(1);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterpreterOpcodeSet {
    enabled: [bool; INTERPRETER_OPCODE_COUNT],
}

impl InterpreterOpcodeSet {
    pub fn all() -> Self {
        Self {
            enabled: [true; INTERPRETER_OPCODE_COUNT],
        }
    }

    pub fn none() -> Self {
        Self {
            enabled: [false; INTERPRETER_OPCODE_COUNT],
        }
    }

    pub fn contains(self, opcode: InterpreterOpcode) -> bool {
        self.enabled[usize::from(opcode.code())]
    }

    pub fn insert(&mut self, opcode: InterpreterOpcode) {
        self.enabled[usize::from(opcode.code())] = true;
    }

    pub fn parse_csv(value: &str) -> Self {
        let mut set = Self::none();
        for item in value.split(',') {
            if let Some(opcode) = InterpreterOpcode::parse_name(item) {
                set.insert(opcode);
            }
        }
        set
    }
}

pub fn interpreter_opcodes_enabled_from_env() -> InterpreterOpcodeSet {
    std::env::var("QWEN36_INTERPRETER_OPCODES_ENABLED")
        .ok()
        .map(|value| InterpreterOpcodeSet::parse_csv(&value))
        .unwrap_or_else(InterpreterOpcodeSet::all)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_sizes_match_cuda_header() {
        assert_eq!(std::mem::size_of::<InterpreterDep>(), 8);
        assert_eq!(std::mem::size_of::<InterpreterInstruction>(), 152);
        assert_eq!(std::mem::size_of::<InterpreterProgramSpec>(), 40);
    }

    #[test]
    fn program_finish_appends_exit_and_counts_counters() {
        let mut program = InterpreterProgram::new();
        program.push(
            InterpreterInstruction::fallback_trampoline()
                .with_arrival_counter(0)
                .with_dep(2, 7)
                .with_publish(3, 1),
        );
        let program = program.finish();
        assert_eq!(program.instructions.len(), 2);
        assert_eq!(program.counter_count, 4);
        assert_eq!(
            program
                .instructions
                .last()
                .and_then(InterpreterInstruction::opcode),
            Some(InterpreterOpcode::Exit)
        );
    }

    #[test]
    fn parses_enabled_opcode_csv() {
        let set = InterpreterOpcodeSet::parse_csv("RMSNORM_NVFP4_QUANT,residual_add,missing");
        assert!(set.contains(InterpreterOpcode::RmsNormNvfp4Quant));
        assert!(set.contains(InterpreterOpcode::ResidualAdd));
        assert!(!set.contains(InterpreterOpcode::Nvfp4Gemv));
    }

    #[test]
    fn gemv_swiglu_and_rope_constructors_pack_payloads() {
        let gemv = InterpreterInstruction::nvfp4_gemv(
            16,
            1024,
            0.5,
            DevicePtr(20),
            DevicePtr(21),
            DevicePtr(22),
            DevicePtr(23),
            DevicePtr(24),
        );
        assert_eq!(gemv.opcode(), Some(InterpreterOpcode::Nvfp4Gemv));
        assert_eq!(gemv.payload[0], 16);
        assert_eq!(gemv.payload[1], 1024);
        assert_eq!(gemv.payload[7] as u32, 0.5f32.to_bits());

        let deltanet = InterpreterInstruction::deltanet_recur_spec(DevicePtr(30));
        assert_eq!(deltanet.opcode(), Some(InterpreterOpcode::DeltaNetRecur));
        assert_eq!(deltanet.payload[0], 30);

        let attn = InterpreterInstruction::attn_decode_full_spec(DevicePtr(31));
        assert_eq!(attn.opcode(), Some(InterpreterOpcode::AttnDecodeFull));
        assert_eq!(attn.payload[0], 31);

        let swiglu = InterpreterInstruction::swiglu_nvfp4_quant(
            17,
            2.0,
            DevicePtr(1),
            DevicePtr(2),
            DevicePtr(3),
            DevicePtr(4),
            DevicePtr(5),
        );
        assert_eq!(swiglu.opcode(), Some(InterpreterOpcode::SwiGluNvfp4Quant));
        assert_eq!(swiglu.payload[0], 17);
        assert_eq!(swiglu.payload[6] as u32, 2.0f32.to_bits());

        let rope = InterpreterInstruction::rope_partial(
            1,
            2,
            3,
            4,
            2,
            10000.0,
            -7,
            true,
            DevicePtr(10),
            DevicePtr(11),
            DevicePtr(12),
            DevicePtr(13),
        );
        assert_eq!(rope.opcode(), Some(InterpreterOpcode::RopePartial));
        assert_eq!(rope.payload[5] as u32, 10000.0f32.to_bits());
        assert_eq!(rope.payload[6] as u32, (-7i32) as u32);
        assert_eq!((rope.payload[6] >> 32) as u32, 1);
    }
}
