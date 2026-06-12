use anyhow::{Context, Result, bail};
use qwen36_fp4_core::CoreError;
use qwen36_fp4_kernels::{CudaDeviceBuffer, DevicePtr};

use crate::eagle3::{Eagle3Drafter, Eagle3LayerWeights, Eagle3Manifest, Eagle3WeightRef};
use crate::gpu::TensorOnDevice;

pub struct Eagle3LayerDevice {
    pub hidden_norm: TensorOnDevice,
    pub input_layernorm: TensorOnDevice,
    pub post_attention_layernorm: TensorOnDevice,
    pub q_proj: TensorOnDevice,
    pub k_proj: TensorOnDevice,
    pub v_proj: TensorOnDevice,
    pub o_proj: TensorOnDevice,
    pub mlp_gate_proj: TensorOnDevice,
    pub mlp_up_proj: TensorOnDevice,
    pub mlp_down_proj: TensorOnDevice,
}

pub struct Eagle3DrafterDevice {
    pub layer: Eagle3LayerDevice,
    pub fc: TensorOnDevice,
    pub norm: TensorOnDevice,
    pub lm_head: TensorOnDevice,
    _buffers: Vec<CudaDeviceBuffer>,
    total_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct Eagle3VramReport {
    pub layer_bytes: usize,
    pub fc_bytes: usize,
    pub norm_bytes: usize,
    pub lm_head_bytes: usize,
    pub total_bytes: usize,
    pub tensor_count: usize,
}

impl Eagle3DrafterDevice {
    pub fn upload(host: &Eagle3Drafter) -> Result<Self> {
        let manifest = &host.manifest;
        let mut buffers: Vec<CudaDeviceBuffer> = Vec::with_capacity(manifest.tensor_count());
        let mut total_bytes = 0_usize;

        let mut upload_one = |entry: &Eagle3WeightRef,
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
            .with_context(|| format!("EAGLE3 upload {}", entry.name))?;
            let view = TensorOnDevice {
                ptr: buffer.ptr(),
                bytes: buffer.bytes(),
            };
            *total += buffer.bytes();
            buffers.push(buffer);
            Ok(view)
        };

        let layer = upload_layer(
            &manifest.layer,
            &mut buffers,
            &mut total_bytes,
            &mut upload_one,
        )?;
        let fc = upload_one(&manifest.fc, &mut buffers, &mut total_bytes)?;
        let norm = upload_one(&manifest.norm, &mut buffers, &mut total_bytes)?;
        let lm_head = upload_one(&manifest.lm_head, &mut buffers, &mut total_bytes)?;

        if buffers.len() != manifest.tensor_count() - usize::from(manifest.d2t.is_some()) {
            bail!(
                "EAGLE3 upload allocated {} buffers but manifest claimed {} GPU tensors",
                buffers.len(),
                manifest.tensor_count() - usize::from(manifest.d2t.is_some()),
            );
        }

        Ok(Self {
            layer,
            fc,
            norm,
            lm_head,
            _buffers: buffers,
            total_bytes,
        })
    }

    pub fn report(&self, manifest: &Eagle3Manifest) -> Eagle3VramReport {
        let layer_bytes = layer_bytes(&self.layer);
        Eagle3VramReport {
            layer_bytes,
            fc_bytes: self.fc.bytes,
            norm_bytes: self.norm.bytes,
            lm_head_bytes: self.lm_head.bytes,
            total_bytes: self.total_bytes,
            tensor_count: manifest.tensor_count(),
        }
    }
}

fn upload_layer(
    layer: &Eagle3LayerWeights,
    buffers: &mut Vec<CudaDeviceBuffer>,
    total: &mut usize,
    upload_one: &mut impl FnMut(
        &Eagle3WeightRef,
        &mut Vec<CudaDeviceBuffer>,
        &mut usize,
    ) -> Result<TensorOnDevice>,
) -> Result<Eagle3LayerDevice> {
    Ok(Eagle3LayerDevice {
        hidden_norm: upload_one(&layer.hidden_norm, buffers, total)?,
        input_layernorm: upload_one(&layer.input_layernorm, buffers, total)?,
        post_attention_layernorm: upload_one(&layer.post_attention_layernorm, buffers, total)?,
        q_proj: upload_one(&layer.q_proj, buffers, total)?,
        k_proj: upload_one(&layer.k_proj, buffers, total)?,
        v_proj: upload_one(&layer.v_proj, buffers, total)?,
        o_proj: upload_one(&layer.o_proj, buffers, total)?,
        mlp_gate_proj: upload_one(&layer.mlp_gate_proj, buffers, total)?,
        mlp_up_proj: upload_one(&layer.mlp_up_proj, buffers, total)?,
        mlp_down_proj: upload_one(&layer.mlp_down_proj, buffers, total)?,
    })
}

fn layer_bytes(layer: &Eagle3LayerDevice) -> usize {
    layer.hidden_norm.bytes
        + layer.input_layernorm.bytes
        + layer.post_attention_layernorm.bytes
        + layer.q_proj.bytes
        + layer.k_proj.bytes
        + layer.v_proj.bytes
        + layer.o_proj.bytes
        + layer.mlp_gate_proj.bytes
        + layer.mlp_up_proj.bytes
        + layer.mlp_down_proj.bytes
}

#[allow(dead_code)]
fn _ptr(view: TensorOnDevice) -> DevicePtr {
    view.ptr
}
