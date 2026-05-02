use std::collections::{BTreeSet, HashMap};

use qwen36_fp4_core::{CoreError, ModelTopology, Result, TensorDtype, TensorInfo, TensorRole};
use qwen36_fp4_kernels::{
    CudaDeviceBuffer, DevicePtr, Nvfp4RetileScalesSpec, cuda_synchronize, nvfp4_retile_scales,
};
use qwen36_fp4_loader::MappedModel;

use crate::state::RuntimeState;
use crate::weights::{LayerWeights, LinearWeightBinding, ModelWeightsManifest};

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
    tensors: HashMap<String, GpuTensor>,
    total_bytes: u64,
}

impl GpuWeightStore {
    pub fn upload_required(
        model: &MappedModel,
        manifest: &ModelWeightsManifest,
        include_mtp: bool,
    ) -> Result<Self> {
        let names = manifest
            .tensor_infos_for_upload(include_mtp)
            .into_iter()
            .map(|tensor| tensor.name.clone())
            .collect::<BTreeSet<_>>();
        let mut tensors = HashMap::with_capacity(names.len());
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

/// Pre-fused MLP weights (gate_proj concatenated with up_proj along the output
/// dim) so a single FP4 GEMM produces both halves in one call. The fusion is
/// only valid when gate_proj and up_proj share `weight_scale_2` and
/// `input_scale` exactly — `MlpFusedStore::build` validates that and refuses
/// to construct otherwise. The shared scale path matches every layer of the
/// shipped Qwen3.6 NVFP4 checkpoint.
#[derive(Debug)]
pub struct MlpFusedLayer {
    pub combined_weight: CudaDeviceBuffer,
    pub combined_block_scale: CudaDeviceBuffer,
    pub out_features: usize,
}

#[derive(Debug, Default)]
pub struct MlpFusedStore {
    pub layers: Vec<MlpFusedLayer>,
    pub mtp_layers: Vec<MlpFusedLayer>,
    pub total_bytes: u64,
}

/// Pre-fused DeltaNet input projections: `in_proj_qkv`, `_b`, `_a`, `_z`
/// concatenated along the output dim into a single FP4 weight + block_scale.
/// All four share `weight_scale_2` and `input_scale` in the shipped Qwen3.6
/// checkpoint, so a single combined GEMM with one alpha covers all four.
///
/// `_b` and `_a` are 48 rows each (not a multiple of 128, the FP4 block_scale
/// outer-block boundary). They are padded with zeros to 128 rows in the
/// combined weight; the GEMM computes zero outputs for the padding rows,
/// which downstream consumers ignore. Output offsets land at:
///   qkv  : 0
///   b    : qkv_dim
///   a    : qkv_dim + 128
///   z    : qkv_dim + 256
#[derive(Debug)]
pub struct LinearAttnInProjFused {
    pub combined_weight: CudaDeviceBuffer,
    pub combined_block_scale: CudaDeviceBuffer,
    pub qkv_offset: usize,
    pub b_offset: usize,
    pub a_offset: usize,
    pub z_offset: usize,
    pub combined_out_features: usize,
}

#[derive(Debug, Default)]
pub struct LinearAttnInProjFusedStore {
    /// Indexed by global layer_index. `None` for full-attention layers.
    pub layers: Vec<Option<LinearAttnInProjFused>>,
    pub total_bytes: u64,
}

impl LinearAttnInProjFusedStore {
    pub fn build(weights: &GpuWeightStore, manifest: &ModelWeightsManifest) -> Result<Self> {
        let mut layers: Vec<Option<LinearAttnInProjFused>> =
            Vec::with_capacity(manifest.layers.len());
        let mut total_bytes = 0_u64;
        for layer in &manifest.layers {
            match layer {
                LayerWeights::LinearAttention(linear) => {
                    let entry = build_linear_attn_layer_fused(weights, linear)?;
                    total_bytes += entry.combined_weight.bytes() as u64
                        + entry.combined_block_scale.bytes() as u64;
                    layers.push(Some(entry));
                }
                LayerWeights::FullAttention(_) => layers.push(None),
            }
        }
        cuda_synchronize()?;
        Ok(Self {
            layers,
            total_bytes,
        })
    }
}

fn unwrap_nvfp4<'a>(
    binding: &'a LinearWeightBinding,
    name: &str,
) -> Result<(
    &'a TensorInfo,
    &'a TensorInfo,
    &'a TensorInfo,
    &'a TensorInfo,
)> {
    match binding {
        LinearWeightBinding::Nvfp4 {
            weight,
            block_scale,
            tensor_scale,
            input_scale,
        } => Ok((weight, block_scale, tensor_scale, input_scale)),
        LinearWeightBinding::Bf16 { .. } => Err(CoreError::Runtime(format!(
            "fused DeltaNet in_proj requires NVFP4 {name}"
        ))),
    }
}

fn build_linear_attn_layer_fused(
    weights: &GpuWeightStore,
    layer: &crate::weights::LinearAttentionLayerWeights,
) -> Result<LinearAttnInProjFused> {
    let (qkv_w, qkv_bs, qkv_ts, qkv_is) = unwrap_nvfp4(&layer.in_proj_qkv, "in_proj_qkv")?;
    let (b_w, b_bs, b_ts, b_is) = unwrap_nvfp4(&layer.in_proj_b, "in_proj_b")?;
    let (a_w, a_bs, a_ts, a_is) = unwrap_nvfp4(&layer.in_proj_a, "in_proj_a")?;
    let (z_w, z_bs, z_ts, z_is) = unwrap_nvfp4(&layer.in_proj_z, "in_proj_z")?;

    // Validate scales match across all four projections.
    let scalars = |info: &TensorInfo| weights.scalar_f32(&info.name);
    let ts = [qkv_ts, b_ts, a_ts, z_ts]
        .iter()
        .map(|t| scalars(t).ok_or_else(|| CoreError::Runtime(format!("missing scalar {}", t.name))))
        .collect::<Result<Vec<_>>>()?;
    let is_ = [qkv_is, b_is, a_is, z_is]
        .iter()
        .map(|t| scalars(t).ok_or_else(|| CoreError::Runtime(format!("missing scalar {}", t.name))))
        .collect::<Result<Vec<_>>>()?;
    let ts_ref = ts[0];
    let is_ref = is_[0];
    if ts.iter().any(|v| (v - ts_ref).abs() > 1e-9) || is_.iter().any(|v| (v - is_ref).abs() > 1e-9)
    {
        return Err(CoreError::Runtime(format!(
            "fused DeltaNet in_proj requires matching tensor_scale and input_scale across qkv/b/a/z; got tensor_scales={:?}, input_scales={:?}",
            ts, is_,
        )));
    }

    // Look up GPU buffers.
    let lookup = |info: &TensorInfo| -> Result<&GpuTensor> {
        weights
            .tensor(&info.name)
            .ok_or_else(|| CoreError::Runtime(format!("missing GPU buffer for {}", info.name)))
    };
    let qkv_w_buf = lookup(qkv_w)?;
    let b_w_buf = lookup(b_w)?;
    let a_w_buf = lookup(a_w)?;
    let z_w_buf = lookup(z_w)?;
    let qkv_bs_buf = lookup(qkv_bs)?;
    let b_bs_buf = lookup(b_bs)?;
    let a_bs_buf = lookup(a_bs)?;
    let z_bs_buf = lookup(z_bs)?;

    // Out-features (rows) and FP4-packed K bytes per row are read straight
    // from the safetensor shape so we do not depend on topology helpers.
    let qkv_dim = qkv_w
        .shape
        .first()
        .copied()
        .ok_or_else(|| CoreError::Runtime("in_proj_qkv has empty shape".to_owned()))?;
    let b_dim = b_w
        .shape
        .first()
        .copied()
        .ok_or_else(|| CoreError::Runtime("in_proj_b has empty shape".to_owned()))?;
    let a_dim = a_w
        .shape
        .first()
        .copied()
        .ok_or_else(|| CoreError::Runtime("in_proj_a has empty shape".to_owned()))?;
    let z_dim = z_w
        .shape
        .first()
        .copied()
        .ok_or_else(|| CoreError::Runtime("in_proj_z has empty shape".to_owned()))?;
    let row_bytes = qkv_w
        .shape
        .get(1)
        .copied()
        .ok_or_else(|| CoreError::Runtime("in_proj_qkv shape missing inner dim".to_owned()))?;
    if [b_w, a_w, z_w]
        .iter()
        .any(|t| t.shape.get(1).copied() != Some(row_bytes))
    {
        return Err(CoreError::Runtime(
            "fused DeltaNet in_proj requires identical packed K bytes across qkv/b/a/z".to_owned(),
        ));
    }

    // Round each output dim up to the next 128 — this matches the FP4
    // block_scale outer-block boundary, so byte-concatenation of the existing
    // weight + block_scale buffers stays well-formed. Padding rows have FP4
    // weight 0 (memset below); the GEMM emits 0 for those rows, which the
    // engine never reads back.
    let round_up = |n: usize| (n + 127) & !127usize;
    let qkv_padded = round_up(qkv_dim);
    let b_padded = round_up(b_dim);
    let a_padded = round_up(a_dim);
    let z_padded = round_up(z_dim);
    let combined_out = qkv_padded + b_padded + a_padded + z_padded;

    let qkv_offset = 0;
    let b_offset = qkv_padded;
    let a_offset = qkv_padded + b_padded;
    let z_offset = qkv_padded + b_padded + a_padded;

    // Allocate + zero, then copy each tensor's bytes into its slot.
    let combined_w_bytes = combined_out * row_bytes;
    let combined_weight = CudaDeviceBuffer::alloc(combined_w_bytes)?;
    combined_weight.memset(0)?;
    combined_weight.copy_from_device_ptr_at(
        qkv_offset * row_bytes,
        qkv_w_buf.buffer.ptr(),
        qkv_w_buf.buffer.bytes(),
    )?;
    combined_weight.copy_from_device_ptr_at(
        b_offset * row_bytes,
        b_w_buf.buffer.ptr(),
        b_w_buf.buffer.bytes(),
    )?;
    combined_weight.copy_from_device_ptr_at(
        a_offset * row_bytes,
        a_w_buf.buffer.ptr(),
        a_w_buf.buffer.bytes(),
    )?;
    combined_weight.copy_from_device_ptr_at(
        z_offset * row_bytes,
        z_w_buf.buffer.ptr(),
        z_w_buf.buffer.bytes(),
    )?;

    // Block-scale concatenation: each outer-block of 128 rows is exactly
    // `qkv_bs_buf.bytes() / (qkv_padded/128)` bytes; we recover it from the
    // qkv buffer instead of recomputing the swizzled formula here.
    let outer_block_bs_bytes = qkv_bs_buf.buffer.bytes() / (qkv_padded / 128);
    let combined_bs_bytes = (combined_out / 128) * outer_block_bs_bytes;
    let combined_block_scale = CudaDeviceBuffer::alloc(combined_bs_bytes)?;
    combined_block_scale.memset(0)?;
    combined_block_scale.copy_from_device_ptr_at(
        (qkv_offset / 128) * outer_block_bs_bytes,
        qkv_bs_buf.buffer.ptr(),
        qkv_bs_buf.buffer.bytes(),
    )?;
    combined_block_scale.copy_from_device_ptr_at(
        (b_offset / 128) * outer_block_bs_bytes,
        b_bs_buf.buffer.ptr(),
        b_bs_buf.buffer.bytes(),
    )?;
    combined_block_scale.copy_from_device_ptr_at(
        (a_offset / 128) * outer_block_bs_bytes,
        a_bs_buf.buffer.ptr(),
        a_bs_buf.buffer.bytes(),
    )?;
    combined_block_scale.copy_from_device_ptr_at(
        (z_offset / 128) * outer_block_bs_bytes,
        z_bs_buf.buffer.ptr(),
        z_bs_buf.buffer.bytes(),
    )?;

    Ok(LinearAttnInProjFused {
        combined_weight,
        combined_block_scale,
        qkv_offset,
        b_offset,
        a_offset,
        z_offset,
        combined_out_features: combined_out,
    })
}

impl MlpFusedStore {
    pub fn build(
        weights: &GpuWeightStore,
        manifest: &ModelWeightsManifest,
        intermediate_size: usize,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(manifest.layers.len());
        let mut total_bytes = 0_u64;
        for layer in &manifest.layers {
            let common = match layer {
                LayerWeights::LinearAttention(l) => &l.common,
                LayerWeights::FullAttention(l) => &l.common,
            };
            let entry = build_layer_pair(
                weights,
                &common.mlp_gate_proj,
                &common.mlp_up_proj,
                intermediate_size,
            )?;
            total_bytes +=
                entry.combined_weight.bytes() as u64 + entry.combined_block_scale.bytes() as u64;
            layers.push(entry);
        }
        // MTP head MLP weights ship as BF16 in the Qwen3.6 checkpoint (not
        // NVFP4), so they cannot use the fused FP4 GEMM path. Skip them here;
        // `run_mtp_mlp_decode` keeps the original BF16 path. If a future
        // checkpoint quantizes the MTP head, this can be relaxed.
        let mtp_layers = Vec::new();
        let _ = manifest.mtp.as_ref();
        cuda_synchronize()?;
        Ok(Self {
            layers,
            mtp_layers,
            total_bytes,
        })
    }
}

fn build_layer_pair(
    weights: &GpuWeightStore,
    gate: &LinearWeightBinding,
    up: &LinearWeightBinding,
    intermediate_size: usize,
) -> Result<MlpFusedLayer> {
    let (g_weight, g_block_scale, g_tensor_scale, g_input_scale) = match gate {
        LinearWeightBinding::Nvfp4 {
            weight,
            block_scale,
            tensor_scale,
            input_scale,
        } => (weight, block_scale, tensor_scale, input_scale),
        LinearWeightBinding::Bf16 { .. } => {
            return Err(CoreError::Runtime(
                "fused MLP requires NVFP4 gate_proj".to_owned(),
            ));
        }
    };
    let (u_weight, u_block_scale, u_tensor_scale, u_input_scale) = match up {
        LinearWeightBinding::Nvfp4 {
            weight,
            block_scale,
            tensor_scale,
            input_scale,
        } => (weight, block_scale, tensor_scale, input_scale),
        LinearWeightBinding::Bf16 { .. } => {
            return Err(CoreError::Runtime(
                "fused MLP requires NVFP4 up_proj".to_owned(),
            ));
        }
    };

    let g_ts = weights.scalar_f32(&g_tensor_scale.name).ok_or_else(|| {
        CoreError::Runtime(format!(
            "missing tensor_scale scalar for {}",
            g_tensor_scale.name
        ))
    })?;
    let u_ts = weights.scalar_f32(&u_tensor_scale.name).ok_or_else(|| {
        CoreError::Runtime(format!(
            "missing tensor_scale scalar for {}",
            u_tensor_scale.name
        ))
    })?;
    let g_is = weights.scalar_f32(&g_input_scale.name).ok_or_else(|| {
        CoreError::Runtime(format!(
            "missing input_scale scalar for {}",
            g_input_scale.name
        ))
    })?;
    let u_is = weights.scalar_f32(&u_input_scale.name).ok_or_else(|| {
        CoreError::Runtime(format!(
            "missing input_scale scalar for {}",
            u_input_scale.name
        ))
    })?;
    if (g_ts - u_ts).abs() > 1e-9 || (g_is - u_is).abs() > 1e-9 {
        return Err(CoreError::Runtime(format!(
            "fused MLP requires matching gate/up scales; gate_proj={}: tensor_scale={}, input_scale={} | up_proj={}: tensor_scale={}, input_scale={}",
            g_tensor_scale.name, g_ts, g_is, u_tensor_scale.name, u_ts, u_is
        )));
    }

    let g_w = weights
        .tensor(&g_weight.name)
        .ok_or_else(|| CoreError::Runtime(format!("missing GPU buffer for {}", g_weight.name)))?;
    let u_w = weights
        .tensor(&u_weight.name)
        .ok_or_else(|| CoreError::Runtime(format!("missing GPU buffer for {}", u_weight.name)))?;
    let g_bs = weights.tensor(&g_block_scale.name).ok_or_else(|| {
        CoreError::Runtime(format!("missing GPU buffer for {}", g_block_scale.name))
    })?;
    let u_bs = weights.tensor(&u_block_scale.name).ok_or_else(|| {
        CoreError::Runtime(format!("missing GPU buffer for {}", u_block_scale.name))
    })?;

    if g_w.buffer.bytes() != u_w.buffer.bytes() || g_bs.buffer.bytes() != u_bs.buffer.bytes() {
        return Err(CoreError::Runtime(
            "fused MLP requires gate/up weights and block_scales of identical byte size".to_owned(),
        ));
    }
    let weight_bytes = g_w.buffer.bytes();
    let scale_bytes = g_bs.buffer.bytes();
    let combined_weight = CudaDeviceBuffer::alloc(weight_bytes * 2)?;
    combined_weight.copy_from_device_ptr_at(0, g_w.buffer.ptr(), weight_bytes)?;
    combined_weight.copy_from_device_ptr_at(weight_bytes, u_w.buffer.ptr(), weight_bytes)?;
    let combined_block_scale = CudaDeviceBuffer::alloc(scale_bytes * 2)?;
    combined_block_scale.copy_from_device_ptr_at(0, g_bs.buffer.ptr(), scale_bytes)?;
    combined_block_scale.copy_from_device_ptr_at(scale_bytes, u_bs.buffer.ptr(), scale_bytes)?;

    Ok(MlpFusedLayer {
        combined_weight,
        combined_block_scale,
        out_features: intermediate_size * 2,
    })
}

#[derive(Debug)]
pub struct GpuRuntimeBuffers {
    pub kv_cache: Option<CudaDeviceBuffer>,
    pub mtp_kv_cache: Option<CudaDeviceBuffer>,
    pub deltanet_state: CudaDeviceBuffer,
    pub deltanet_checkpoint: CudaDeviceBuffer,
    pub conv_history: CudaDeviceBuffer,
    /// Snapshot of `conv_history` taken before an MTP verify chunk so we can
    /// roll back the linear-attention conv1d state on draft rejection.
    pub conv_history_checkpoint: CudaDeviceBuffer,
    /// Scratch buffer for the K/V slice of the speculative verify positions, packed as
    /// `[layer 0 K | layer 0 V | layer 1 K | layer 1 V | ... | MTP K | MTP V]`.
    /// Sized for the maximum (5 tokens) chunk written by the MTP=4 verify path.
    pub mtp_kv_snapshot: CudaDeviceBuffer,
    /// Layout that lets the engine address each layer's slice inside
    /// `mtp_kv_snapshot` without recomputing offsets every call.
    pub mtp_kv_snapshot_layout: MtpKvSnapshotLayout,
    pub workspace: Option<CudaDeviceBuffer>,
}

/// Byte offsets into [`GpuRuntimeBuffers::mtp_kv_snapshot`] for each layer's
/// K and V slice for the MTP verify positions. All slices are
/// `slice_bytes` long (= VERIFY_TOKENS × kv_heads × head_dim × bytes_per_value).
#[derive(Debug, Clone)]
pub struct MtpKvSnapshotLayout {
    pub slice_bytes: usize,
    /// Offsets for the 16 main attention layers (in attention_layer_index order).
    pub main_k_offsets: Vec<usize>,
    pub main_v_offsets: Vec<usize>,
    /// Offsets for the single MTP attention layer (None if no MTP).
    pub mtp_k_offset: Option<usize>,
    pub mtp_v_offset: Option<usize>,
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
    pub mtp_logits: CudaDeviceBuffer,
    pub sampled_token_u32: CudaDeviceBuffer,
    /// MTP verify graph token bundle. The first four u32s keep the MTP=1
    /// layout `[draft_input, next_token, verified_token, next_draft_token]`;
    /// the remaining slots are used by the MTP=2..4 graph fast path.
    pub mtp_verify_token_u32: CudaDeviceBuffer,
    /// Per-q-head per-split scratch for split-KV decode attention. Sized
    /// for `n_splits = ceil(max_context / kSplitTimestepsPerBlock)`.
    /// Layout `[q_heads, n_splits, head_dim]` FP32. Keeping the partial
    /// accumulator in FP32 avoids adding a BF16 rounding point before the final
    /// softmax reduction.
    pub attn_partial_acc: CudaDeviceBuffer,
    /// `[q_heads, n_splits]` FP32 — per-split running max from online softmax.
    pub attn_partial_max: CudaDeviceBuffer,
    /// `[q_heads, n_splits]` FP32 — per-split softmax denominator.
    pub attn_partial_denom: CudaDeviceBuffer,
}

/// Must match the smallest supported split size in `kernels-cuda/attention.cu`.
/// Runtime scratch is sized for this worst case so decode can choose larger
/// per-call split sizes without reallocating or invalidating CUDA graphs.
pub const ATTN_MIN_SPLIT_TIMESTEPS_PER_BLOCK: usize = 64;

fn attention_partial_n_splits(max_context: usize) -> usize {
    max_context
        .div_ceil(ATTN_MIN_SPLIT_TIMESTEPS_PER_BLOCK)
        .max(1)
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
    pub fn allocate(
        state: &RuntimeState,
        workspace_bytes: usize,
        mtp_kv_cache_bytes: u64,
        topology: &ModelTopology,
    ) -> Result<Self> {
        let conv_history_bytes =
            usize_from_u64(state.deltanet.conv_history_bytes, "DeltaNet conv history")?;
        let snapshot_layout = MtpKvSnapshotLayout::new(topology, &state.kv_cache)?;
        let snapshot_bytes = snapshot_layout.total_bytes().max(1);
        Ok(Self {
            kv_cache: alloc_optional(state.kv_cache.total_bytes)?,
            mtp_kv_cache: alloc_optional(mtp_kv_cache_bytes)?,
            deltanet_state: CudaDeviceBuffer::zeroed(usize_from_u64(
                state.deltanet.total_state_bytes,
                "DeltaNet state",
            )?)?,
            deltanet_checkpoint: CudaDeviceBuffer::zeroed(usize_from_u64(
                state.deltanet.checkpoint_bytes,
                "DeltaNet checkpoint",
            )?)?,
            conv_history: CudaDeviceBuffer::zeroed(conv_history_bytes)?,
            conv_history_checkpoint: CudaDeviceBuffer::zeroed(conv_history_bytes.max(1))?,
            mtp_kv_snapshot: CudaDeviceBuffer::zeroed(snapshot_bytes)?,
            mtp_kv_snapshot_layout: snapshot_layout,
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
            + self.conv_history.bytes() as u64
            + self.conv_history_checkpoint.bytes() as u64
            + self.mtp_kv_snapshot.bytes() as u64;
        if let Some(kv_cache) = &self.kv_cache {
            total += kv_cache.bytes() as u64;
        }
        if let Some(mtp_kv_cache) = &self.mtp_kv_cache {
            total += mtp_kv_cache.bytes() as u64;
        }
        if let Some(workspace) = &self.workspace {
            total += workspace.bytes() as u64;
        }
        total
    }
}

impl MtpKvSnapshotLayout {
    pub const VERIFY_TOKENS: usize = 5;

    fn new(topology: &ModelTopology, kv_cache: &crate::kv_cache::KvCachePlan) -> Result<Self> {
        if kv_cache.max_context == 0 {
            return Err(CoreError::Runtime(
                "MTP KV snapshot requires a non-zero max_context".to_owned(),
            ));
        }
        let attention_layers = topology.attention_layers().len();
        if kv_cache.layers.len() != attention_layers {
            return Err(CoreError::Runtime(format!(
                "KV cache plan has {} attention layers, expected {attention_layers}",
                kv_cache.layers.len()
            )));
        }

        let plane_bytes = kv_cache
            .layers
            .first()
            .map(|layer| layer.layer_bytes / 2)
            .unwrap_or(0);
        if plane_bytes == 0 || plane_bytes % kv_cache.max_context as u64 != 0 {
            return Err(CoreError::Runtime(
                "KV cache plane size is not divisible by max_context; cannot snapshot token slices safely"
                    .to_owned(),
            ));
        }
        for layer in &kv_cache.layers {
            if layer.layer_bytes / 2 != plane_bytes || layer.layer_bytes % 2 != 0 {
                return Err(CoreError::Runtime(
                    "KV cache plan uses non-uniform layer strides; cannot snapshot token slices safely"
                        .to_owned(),
                ));
            }
        }
        let row_bytes = usize::try_from(plane_bytes / kv_cache.max_context as u64)
            .map_err(|_| CoreError::Runtime("KV cache row size overflows usize".to_owned()))?;
        let slice_bytes = row_bytes * Self::VERIFY_TOKENS;

        let mut main_k_offsets = Vec::with_capacity(attention_layers);
        let mut main_v_offsets = Vec::with_capacity(attention_layers);
        let mut cursor = 0_usize;
        for _ in 0..attention_layers {
            main_k_offsets.push(cursor);
            cursor += slice_bytes;
            main_v_offsets.push(cursor);
            cursor += slice_bytes;
        }
        let (mtp_k_offset, mtp_v_offset) = if topology.mtp_num_hidden_layers > 0 {
            let k = cursor;
            cursor += slice_bytes;
            let v = cursor;
            cursor += slice_bytes;
            (Some(k), Some(v))
        } else {
            (None, None)
        };
        let _ = cursor;
        Ok(Self {
            slice_bytes,
            main_k_offsets,
            main_v_offsets,
            mtp_k_offset,
            mtp_v_offset,
        })
    }

    pub fn total_bytes(&self) -> usize {
        let main = (self.main_k_offsets.len() + self.main_v_offsets.len()) * self.slice_bytes;
        let mtp = (self.mtp_k_offset.is_some() as usize + self.mtp_v_offset.is_some() as usize)
            * self.slice_bytes;
        main + mtp
    }
}

impl GpuForwardBuffers {
    pub fn allocate(topology: &ModelTopology, max_context: usize) -> Result<Self> {
        let hidden_bytes = topology.hidden_size * 2;
        let wide_bf16_values = topology
            .intermediate_size
            .max(topology.hidden_size)
            .max(topology.linear_attention_qkv_dim())
            .max(topology.linear_attention_value_dim())
            .max(topology.full_attention_q_dim_with_gate())
            .max(topology.full_attention_q_dim())
            // The fused gate+up MLP path writes both halves into `aux` in a
            // single GEMM, so the buffer must hold 2*intermediate BF16 values.
            .max(2 * topology.intermediate_size);
        let wide_bytes = wide_bf16_values * 2;
        let activation_fp4_bytes = wide_bf16_values.div_ceil(2);
        let activation_scale_bytes = vec16_scale_bytes(wide_bf16_values, 1);
        let linear_heads = topology.linear_num_value_heads;
        let n_splits = attention_partial_n_splits(max_context);
        let attn_q_heads = topology.attention_num_heads;
        let attn_head_dim = topology.attention_head_dim;
        let attn_partial_acc_bytes = attn_q_heads * n_splits * attn_head_dim * 4;
        let attn_partial_scalar_bytes = attn_q_heads * n_splits * 4;
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
            mtp_logits: CudaDeviceBuffer::alloc(topology.vocab_size * 5 * 2)?,
            sampled_token_u32: CudaDeviceBuffer::alloc(4)?,
            mtp_verify_token_u32: CudaDeviceBuffer::alloc(64)?,
            attn_partial_acc: CudaDeviceBuffer::alloc(attn_partial_acc_bytes.max(1))?,
            attn_partial_max: CudaDeviceBuffer::alloc(attn_partial_scalar_bytes.max(1))?,
            attn_partial_denom: CudaDeviceBuffer::alloc(attn_partial_scalar_bytes.max(1))?,
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
            self.mtp_logits.bytes(),
            self.sampled_token_u32.bytes(),
            self.mtp_verify_token_u32.bytes(),
            self.attn_partial_acc.bytes(),
            self.attn_partial_max.bytes(),
            self.attn_partial_denom.bytes(),
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
