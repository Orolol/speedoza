use anyhow::{Context, Result, bail};
use qwen36_fp4_core::CoreError;
use qwen36_fp4_kernels::{CudaDeviceBuffer, DevicePtr};

use crate::dflash::{
    DFlashDrafter, DFlashLayerWeights, DFlashManifest, DFlashWeightRef, LayerAttentionKind,
};

/// Device-resident view of one drafter layer. Each `DevicePtr` points into
/// the layer's `CudaDeviceBuffer`; the buffers themselves live in the
/// owning `DFlashDrafterDevice` so dropping the device handle frees the
/// VRAM. Pointers are only valid for the lifetime of the parent device
/// struct.
pub struct DFlashLayerDevice {
    pub kind: LayerAttentionKind,
    pub input_layernorm: TensorOnDevice,
    pub post_attention_layernorm: TensorOnDevice,
    pub q_proj: TensorOnDevice,
    pub k_proj: TensorOnDevice,
    pub v_proj: TensorOnDevice,
    pub o_proj: TensorOnDevice,
    pub q_norm: TensorOnDevice,
    pub k_norm: TensorOnDevice,
    pub mlp_gate_proj: TensorOnDevice,
    pub mlp_up_proj: TensorOnDevice,
    pub mlp_down_proj: TensorOnDevice,
}

#[derive(Debug, Clone, Copy)]
pub struct TensorOnDevice {
    pub ptr: DevicePtr,
    pub bytes: usize,
}

/// Owns every device buffer the drafter needs. Allocations happen once at
/// `upload`; weights are copied straight from the underlying mmap. The
/// `DFlashDrafter` host handle (and its mmap) must outlive this struct
/// only for the duration of the upload — after `upload` returns, the
/// device buffers are self-sufficient.
pub struct DFlashDrafterDevice {
    pub layers: Vec<DFlashLayerDevice>,
    pub fc: TensorOnDevice,
    pub hidden_norm: TensorOnDevice,
    pub norm: TensorOnDevice,
    // Buffers are kept alongside the per-tensor `TensorOnDevice` views.
    // Dropping the device struct frees the VRAM.
    _buffers: Vec<CudaDeviceBuffer>,
    total_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct DrafterVramReport {
    pub layer_bytes: usize,
    pub fc_bytes: usize,
    pub hidden_norm_bytes: usize,
    pub norm_bytes: usize,
    pub total_bytes: usize,
    pub tensor_count: usize,
}

impl DFlashDrafterDevice {
    pub fn upload(host: &DFlashDrafter) -> Result<Self> {
        let manifest = &host.manifest;
        let mut buffers: Vec<CudaDeviceBuffer> = Vec::with_capacity(manifest.tensor_count());
        let mut total_bytes = 0_usize;

        let mut upload_one =
            |entry: &DFlashWeightRef,
             buffers: &mut Vec<CudaDeviceBuffer>,
             total: &mut usize|
             -> Result<TensorOnDevice> {
                let buffer = CudaDeviceBuffer::alloc(entry.size_bytes)
                    .map_err(|e: CoreError| anyhow::anyhow!("alloc {}: {e}", entry.name))?;
                host.with_tensor(&entry.name, |tensor| {
                    buffer
                        .copy_from_host(tensor.data())
                        .map_err(|e: CoreError| anyhow::anyhow!("upload {}: {e}", entry.name))?;
                    Ok(())
                })
                .with_context(|| format!("drafter upload {}", entry.name))?;
                let view = TensorOnDevice {
                    ptr: buffer.ptr(),
                    bytes: buffer.bytes(),
                };
                *total += buffer.bytes();
                buffers.push(buffer);
                Ok(view)
            };

        let mut layers = Vec::with_capacity(manifest.layers.len());
        for layer in &manifest.layers {
            let device_layer = upload_layer(layer, &mut buffers, &mut total_bytes, &mut upload_one)?;
            layers.push(device_layer);
        }

        let fc = upload_one(&manifest.fc, &mut buffers, &mut total_bytes)?;
        let hidden_norm = upload_one(&manifest.hidden_norm, &mut buffers, &mut total_bytes)?;
        let norm = upload_one(&manifest.norm, &mut buffers, &mut total_bytes)?;

        if buffers.len() != manifest.tensor_count() {
            bail!(
                "drafter upload allocated {} buffers but manifest claimed {}",
                buffers.len(),
                manifest.tensor_count(),
            );
        }

        Ok(Self {
            layers,
            fc,
            hidden_norm,
            norm,
            _buffers: buffers,
            total_bytes,
        })
    }

    pub fn report(&self, manifest: &DFlashManifest) -> DrafterVramReport {
        let layer_bytes = self
            .layers
            .iter()
            .map(layer_bytes)
            .sum::<usize>();
        DrafterVramReport {
            layer_bytes,
            fc_bytes: self.fc.bytes,
            hidden_norm_bytes: self.hidden_norm.bytes,
            norm_bytes: self.norm.bytes,
            total_bytes: self.total_bytes,
            tensor_count: manifest.tensor_count(),
        }
    }

    pub fn layer(&self, idx: usize) -> Option<&DFlashLayerDevice> {
        self.layers.get(idx)
    }
}

fn upload_layer(
    layer: &DFlashLayerWeights,
    buffers: &mut Vec<CudaDeviceBuffer>,
    total: &mut usize,
    upload_one: &mut impl FnMut(
        &DFlashWeightRef,
        &mut Vec<CudaDeviceBuffer>,
        &mut usize,
    ) -> Result<TensorOnDevice>,
) -> Result<DFlashLayerDevice> {
    Ok(DFlashLayerDevice {
        kind: layer.kind,
        input_layernorm: upload_one(&layer.input_layernorm, buffers, total)?,
        post_attention_layernorm: upload_one(&layer.post_attention_layernorm, buffers, total)?,
        q_proj: upload_one(&layer.q_proj, buffers, total)?,
        k_proj: upload_one(&layer.k_proj, buffers, total)?,
        v_proj: upload_one(&layer.v_proj, buffers, total)?,
        o_proj: upload_one(&layer.o_proj, buffers, total)?,
        q_norm: upload_one(&layer.q_norm, buffers, total)?,
        k_norm: upload_one(&layer.k_norm, buffers, total)?,
        mlp_gate_proj: upload_one(&layer.mlp_gate_proj, buffers, total)?,
        mlp_up_proj: upload_one(&layer.mlp_up_proj, buffers, total)?,
        mlp_down_proj: upload_one(&layer.mlp_down_proj, buffers, total)?,
    })
}

fn layer_bytes(layer: &DFlashLayerDevice) -> usize {
    layer.input_layernorm.bytes
        + layer.post_attention_layernorm.bytes
        + layer.q_proj.bytes
        + layer.k_proj.bytes
        + layer.v_proj.bytes
        + layer.o_proj.bytes
        + layer.q_norm.bytes
        + layer.k_norm.bytes
        + layer.mlp_gate_proj.bytes
        + layer.mlp_up_proj.bytes
        + layer.mlp_down_proj.bytes
}
