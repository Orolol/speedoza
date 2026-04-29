use std::collections::{BTreeMap, BTreeSet};

use qwen36_fp4_core::{CoreError, ModelTopology, Result, TensorDtype, TensorInfo, TensorRole};
use qwen36_fp4_kernels::{
    CudaDeviceBuffer, DevicePtr, Nvfp4RetileScalesSpec, cuda_synchronize, nvfp4_retile_scales,
};
use qwen36_fp4_loader::MappedModel;

use crate::state::RuntimeState;
use crate::weights::ModelWeightsManifest;

#[derive(Debug)]
pub struct GpuTensor {
    pub info: TensorInfo,
    pub buffer: CudaDeviceBuffer,
    pub scalar_f32: Option<f32>,
}

impl GpuTensor {
    pub fn ptr(&self) -> DevicePtr {
        self.buffer.ptr()
    }

    pub fn scalar_f32(&self) -> Option<f32> {
        self.scalar_f32
    }
}

#[derive(Debug, Default)]
pub struct GpuWeightStore {
    tensors: BTreeMap<String, GpuTensor>,
    total_bytes: u64,
}

impl GpuWeightStore {
    pub fn upload_required(model: &MappedModel, manifest: &ModelWeightsManifest) -> Result<Self> {
        let names = manifest
            .tensor_infos()
            .into_iter()
            .map(|tensor| tensor.name.clone())
            .collect::<BTreeSet<_>>();
        let mut tensors = BTreeMap::new();
        let mut staging_buffers = Vec::new();
        let mut total_bytes = 0_u64;

        for name in names {
            let info = model
                .tensor_info(&name)
                .cloned()
                .ok_or_else(|| CoreError::Runtime(format!("tensor {name} is missing")))?;
            let upload = upload_tensor(model, info)?;
            total_bytes += upload.tensor.buffer.bytes() as u64;
            if let Some(staging) = upload.staging {
                staging_buffers.push(staging);
            }
            tensors.insert(name, upload.tensor);
        }
        cuda_synchronize()?;
        drop(staging_buffers);

        Ok(Self {
            tensors,
            total_bytes,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&GpuTensor> {
        self.tensors.get(name)
    }

    pub fn scalar_f32(&self, name: &str) -> Option<f32> {
        self.tensor(name).and_then(GpuTensor::scalar_f32)
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

#[derive(Debug)]
pub struct GpuRuntimeBuffers {
    pub kv_cache: Option<CudaDeviceBuffer>,
    pub deltanet_state: CudaDeviceBuffer,
    pub deltanet_checkpoint: CudaDeviceBuffer,
    pub conv_history: CudaDeviceBuffer,
    pub workspace: Option<CudaDeviceBuffer>,
}

#[derive(Debug)]
pub struct GpuForwardBuffers {
    pub hidden: CudaDeviceBuffer,
    pub residual: CudaDeviceBuffer,
    pub normed: CudaDeviceBuffer,
    pub block_out: CudaDeviceBuffer,
    pub qkv: CudaDeviceBuffer,
    pub aux: CudaDeviceBuffer,
    pub aux2: CudaDeviceBuffer,
    pub aux3: CudaDeviceBuffer,
    pub gate_f32: CudaDeviceBuffer,
    pub beta_f32: CudaDeviceBuffer,
    pub activation_fp4: CudaDeviceBuffer,
    pub activation_scale: CudaDeviceBuffer,
    pub activation_scale_2: CudaDeviceBuffer,
    pub token_u32: CudaDeviceBuffer,
    pub position_i32: CudaDeviceBuffer,
    pub logits: CudaDeviceBuffer,
    pub sampled_token_u32: CudaDeviceBuffer,
}

#[derive(Debug)]
pub struct GpuPrefillBuffers {
    pub capacity: usize,
    pub hidden: CudaDeviceBuffer,
    pub residual: CudaDeviceBuffer,
    pub normed: CudaDeviceBuffer,
    pub block_out: CudaDeviceBuffer,
    pub qkv: CudaDeviceBuffer,
    pub aux: CudaDeviceBuffer,
    pub aux2: CudaDeviceBuffer,
    pub aux3: CudaDeviceBuffer,
    pub gate_f32: CudaDeviceBuffer,
    pub beta_f32: CudaDeviceBuffer,
    pub activation_fp4: CudaDeviceBuffer,
    pub activation_scale: CudaDeviceBuffer,
    pub activation_scale_2: CudaDeviceBuffer,
    pub token_u32: CudaDeviceBuffer,
    pub position_i32: CudaDeviceBuffer,
}

impl GpuRuntimeBuffers {
    pub fn allocate(state: &RuntimeState, workspace_bytes: usize) -> Result<Self> {
        Ok(Self {
            kv_cache: alloc_optional(state.kv_cache.total_bytes)?,
            deltanet_state: CudaDeviceBuffer::zeroed(usize_from_u64(
                state.deltanet.total_state_bytes,
                "DeltaNet state",
            )?)?,
            deltanet_checkpoint: CudaDeviceBuffer::zeroed(usize_from_u64(
                state.deltanet.checkpoint_bytes,
                "DeltaNet checkpoint",
            )?)?,
            conv_history: CudaDeviceBuffer::zeroed(usize_from_u64(
                state.deltanet.conv_history_bytes,
                "DeltaNet conv history",
            )?)?,
            workspace: if workspace_bytes == 0 {
                None
            } else {
                Some(CudaDeviceBuffer::alloc(workspace_bytes)?)
            },
        })
    }

    pub fn total_bytes(&self) -> u64 {
        let mut total = self.deltanet_state.bytes() as u64
            + self.deltanet_checkpoint.bytes() as u64
            + self.conv_history.bytes() as u64;
        if let Some(kv_cache) = &self.kv_cache {
            total += kv_cache.bytes() as u64;
        }
        if let Some(workspace) = &self.workspace {
            total += workspace.bytes() as u64;
        }
        total
    }
}

impl GpuForwardBuffers {
    pub fn allocate(topology: &ModelTopology) -> Result<Self> {
        let hidden_bytes = topology.hidden_size * 2;
        let wide_bf16_values = topology
            .intermediate_size
            .max(topology.hidden_size)
            .max(topology.linear_attention_qkv_dim())
            .max(topology.linear_attention_value_dim())
            .max(topology.full_attention_q_dim_with_gate())
            .max(topology.full_attention_q_dim());
        let wide_bytes = wide_bf16_values * 2;
        let activation_fp4_bytes = wide_bf16_values.div_ceil(2);
        let activation_scale_bytes = vec16_scale_bytes(wide_bf16_values, 1);
        let linear_heads = topology.linear_num_value_heads;
        Ok(Self {
            hidden: CudaDeviceBuffer::alloc(hidden_bytes)?,
            residual: CudaDeviceBuffer::alloc(hidden_bytes)?,
            normed: CudaDeviceBuffer::alloc(hidden_bytes)?,
            block_out: CudaDeviceBuffer::alloc(hidden_bytes)?,
            qkv: CudaDeviceBuffer::alloc(wide_bytes)?,
            aux: CudaDeviceBuffer::alloc(wide_bytes)?,
            aux2: CudaDeviceBuffer::alloc(wide_bytes)?,
            aux3: CudaDeviceBuffer::alloc(wide_bytes)?,
            gate_f32: CudaDeviceBuffer::alloc(linear_heads * 4)?,
            beta_f32: CudaDeviceBuffer::alloc(linear_heads * 4)?,
            activation_fp4: CudaDeviceBuffer::alloc(activation_fp4_bytes)?,
            activation_scale: CudaDeviceBuffer::alloc(activation_scale_bytes)?,
            activation_scale_2: CudaDeviceBuffer::alloc(4)?,
            token_u32: CudaDeviceBuffer::alloc(4)?,
            position_i32: CudaDeviceBuffer::alloc(4)?,
            logits: CudaDeviceBuffer::alloc(topology.vocab_size * 2)?,
            sampled_token_u32: CudaDeviceBuffer::alloc(4)?,
        })
    }

    pub fn total_bytes(&self) -> u64 {
        [
            self.hidden.bytes(),
            self.residual.bytes(),
            self.normed.bytes(),
            self.block_out.bytes(),
            self.qkv.bytes(),
            self.aux.bytes(),
            self.aux2.bytes(),
            self.aux3.bytes(),
            self.gate_f32.bytes(),
            self.beta_f32.bytes(),
            self.activation_fp4.bytes(),
            self.activation_scale.bytes(),
            self.activation_scale_2.bytes(),
            self.token_u32.bytes(),
            self.position_i32.bytes(),
            self.logits.bytes(),
            self.sampled_token_u32.bytes(),
        ]
        .into_iter()
        .map(|bytes| bytes as u64)
        .sum()
    }
}

impl GpuPrefillBuffers {
    pub fn allocate(topology: &ModelTopology, capacity: usize) -> Result<Self> {
        let capacity = capacity.max(1);
        let hidden_bytes = capacity * topology.hidden_size * 2;
        let wide_bf16_values = topology
            .intermediate_size
            .max(topology.hidden_size)
            .max(topology.linear_attention_qkv_dim())
            .max(topology.linear_attention_value_dim())
            .max(topology.full_attention_q_dim_with_gate())
            .max(topology.full_attention_q_dim());
        let wide_bytes = capacity * wide_bf16_values * 2;
        let activation_fp4_bytes = capacity * wide_bf16_values.div_ceil(2);
        let activation_scale_bytes = vec16_scale_bytes(wide_bf16_values, capacity);
        let linear_heads = topology.linear_num_value_heads;
        Ok(Self {
            capacity,
            hidden: CudaDeviceBuffer::alloc(hidden_bytes)?,
            residual: CudaDeviceBuffer::alloc(hidden_bytes)?,
            normed: CudaDeviceBuffer::alloc(hidden_bytes)?,
            block_out: CudaDeviceBuffer::alloc(wide_bytes)?,
            qkv: CudaDeviceBuffer::alloc(wide_bytes)?,
            aux: CudaDeviceBuffer::alloc(wide_bytes)?,
            aux2: CudaDeviceBuffer::alloc(wide_bytes)?,
            aux3: CudaDeviceBuffer::alloc(wide_bytes)?,
            gate_f32: CudaDeviceBuffer::alloc(capacity * linear_heads * 4)?,
            beta_f32: CudaDeviceBuffer::alloc(capacity * linear_heads * 4)?,
            activation_fp4: CudaDeviceBuffer::alloc(activation_fp4_bytes)?,
            activation_scale: CudaDeviceBuffer::alloc(activation_scale_bytes)?,
            activation_scale_2: CudaDeviceBuffer::alloc(4)?,
            token_u32: CudaDeviceBuffer::alloc(capacity * 4)?,
            position_i32: CudaDeviceBuffer::alloc(capacity * 4)?,
        })
    }

    pub fn total_bytes(&self) -> u64 {
        [
            self.hidden.bytes(),
            self.residual.bytes(),
            self.normed.bytes(),
            self.block_out.bytes(),
            self.qkv.bytes(),
            self.aux.bytes(),
            self.aux2.bytes(),
            self.aux3.bytes(),
            self.gate_f32.bytes(),
            self.beta_f32.bytes(),
            self.activation_fp4.bytes(),
            self.activation_scale.bytes(),
            self.activation_scale_2.bytes(),
            self.token_u32.bytes(),
            self.position_i32.bytes(),
        ]
        .into_iter()
        .map(|bytes| bytes as u64)
        .sum()
    }
}

#[derive(Debug)]
struct UploadedTensor {
    tensor: GpuTensor,
    staging: Option<CudaDeviceBuffer>,
}

fn upload_tensor(model: &MappedModel, info: TensorInfo) -> Result<UploadedTensor> {
    let name = info.name.clone();
    model
        .with_tensor(&name, move |view| {
            let data = view.data();
            if data.len() != info.size_bytes as usize {
                anyhow::bail!(
                    "tensor {} metadata says {} bytes but safetensors view has {} bytes",
                    info.name,
                    info.size_bytes,
                    data.len()
                );
            }
            let scalar_f32 = (info.dtype == TensorDtype::F32 && data.len() == 4)
                .then(|| f32::from_le_bytes(data.try_into().expect("four bytes were checked")));
            let mut staging = None;
            let buffer = if info.role == TensorRole::Nvfp4BlockScale {
                let upload = upload_retiled_scales(&info, data)?;
                staging = Some(upload.staging);
                upload.tiled
            } else {
                let buffer = CudaDeviceBuffer::alloc(data.len())
                    .map_err(|err| anyhow::anyhow!(err.to_string()))?;
                buffer
                    .copy_from_host(data)
                    .map_err(|err| anyhow::anyhow!(err.to_string()))?;
                buffer
            };
            Ok(UploadedTensor {
                tensor: GpuTensor {
                    info,
                    buffer,
                    scalar_f32,
                },
                staging,
            })
        })
        .map_err(|err| CoreError::Runtime(err.to_string()))
}

fn alloc_optional(bytes: u64) -> Result<Option<CudaDeviceBuffer>> {
    if bytes == 0 {
        Ok(None)
    } else {
        Ok(Some(CudaDeviceBuffer::zeroed(usize_from_u64(
            bytes,
            "optional runtime buffer",
        )?)?))
    }
}

fn usize_from_u64(value: u64, label: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| CoreError::Runtime(format!("{label} size {value} does not fit usize")))
}

fn vec16_scale_bytes(inner_values: usize, outer_values: usize) -> usize {
    let inner_groups = inner_values.div_ceil(16);
    outer_values.div_ceil(128) * inner_groups.div_ceil(4) * 512
}

#[derive(Debug)]
struct RetiledScaleUpload {
    tiled: CudaDeviceBuffer,
    staging: CudaDeviceBuffer,
}

fn upload_retiled_scales(
    info: &TensorInfo,
    row_major: &[u8],
) -> anyhow::Result<RetiledScaleUpload> {
    let outer = *info
        .shape
        .first()
        .ok_or_else(|| anyhow::anyhow!("tensor {} has empty scale shape", info.name))?;
    let inner_groups = *info
        .shape
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("tensor {} scale tensor is not rank-2", info.name))?;
    if row_major.len() != outer * inner_groups {
        anyhow::bail!(
            "tensor {} scale bytes {} do not match shape [{}, {}]",
            info.name,
            row_major.len(),
            outer,
            inner_groups
        );
    }

    let raw =
        CudaDeviceBuffer::alloc(row_major.len()).map_err(|err| anyhow::anyhow!(err.to_string()))?;
    raw.copy_from_host(row_major)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let tiled_bytes = outer.div_ceil(128) * inner_groups.div_ceil(4) * 512;
    let tiled =
        CudaDeviceBuffer::alloc(tiled_bytes).map_err(|err| anyhow::anyhow!(err.to_string()))?;
    nvfp4_retile_scales(&Nvfp4RetileScalesSpec {
        rows: outer,
        inner_groups,
        input_row_major_u8: raw.ptr(),
        output_tiled_u8: tiled.ptr(),
    })
    .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    Ok(RetiledScaleUpload {
        tiled,
        staging: raw,
    })
}
