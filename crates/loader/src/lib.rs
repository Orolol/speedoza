use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use memmap2::{Mmap, MmapOptions};
use qwen36_fp4_core::{
    HuggingFaceConfig, LayoutFile, ModelLayout, ModelTopology, QWEN36_TEXT_NVFP4_MTP_MODEL_ID,
    QuantizationSummary, TensorDtype, TensorInfo,
};
use safetensors::{SafeTensors, tensor::TensorView};

pub fn read_hf_config(model_dir: impl AsRef<Path>) -> Result<HuggingFaceConfig> {
    let path = model_dir.as_ref().join("config.json");
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn read_topology(model_dir: impl AsRef<Path>) -> Result<ModelTopology> {
    let config = read_hf_config(model_dir)?;
    ModelTopology::try_from(&config).map_err(anyhow::Error::from)
}

pub fn discover_model_layout(model_dir: impl AsRef<Path>) -> Result<ModelLayout> {
    discover_model_layout_with_id(model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)
}

pub fn discover_model_layout_with_id(
    model_dir: impl AsRef<Path>,
    model_id: impl Into<String>,
) -> Result<ModelLayout> {
    let model_dir = model_dir.as_ref();
    let hf_config = read_hf_config(model_dir)?;
    let topology = ModelTopology::try_from(&hf_config).map_err(anyhow::Error::from)?;
    let quantization = QuantizationSummary::from_hf_config(&hf_config);
    let safetensor_files = find_safetensors(model_dir)?;
    if safetensor_files.is_empty() {
        return Err(anyhow!(
            "no .safetensors files found in {}",
            model_dir.display()
        ));
    }

    let mut files = Vec::new();
    let mut tensors = Vec::new();
    for path in safetensor_files {
        let relative = path
            .strip_prefix(model_dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        let metadata = fs::metadata(&path).with_context(|| format!("stat {}", path.display()))?;
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("mmap {}", path.display()))?;
        let safetensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("parse safetensors {}", path.display()))?;
        let mut tensor_count = 0;
        for (name, view) in safetensors.tensors() {
            tensor_count += 1;
            let dtype = TensorDtype::from_safetensors_debug(&format!("{:?}", view.dtype()));
            let shape = view.shape().to_vec();
            let size_bytes = view.data().len() as u64;
            tensors.push(TensorInfo::new(
                name,
                relative.clone(),
                dtype,
                shape,
                size_bytes,
            ));
        }
        files.push(LayoutFile {
            path: relative,
            size_bytes: metadata.len(),
            tensor_count,
        });
    }

    Ok(ModelLayout::from_parts(
        model_id.into(),
        topology,
        quantization,
        files,
        tensors,
    ))
}

pub fn write_model_layout_json(layout: &ModelLayout, output: impl AsRef<Path>) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(layout)?;
    fs::write(output.as_ref(), bytes)
        .with_context(|| format!("write {}", output.as_ref().display()))
}

pub struct MappedModel {
    pub model_dir: PathBuf,
    pub layout: ModelLayout,
    files: BTreeMap<String, MappedSafetensorFile>,
    tensor_files: BTreeMap<String, String>,
}

struct MappedSafetensorFile {
    mmap: Mmap,
}

impl MappedModel {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let layout = discover_model_layout(&model_dir)?;
        Self::open_with_layout(model_dir, layout)
    }

    pub fn open_with_layout(model_dir: impl AsRef<Path>, layout: ModelLayout) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let mut files = BTreeMap::new();
        for file in &layout.files {
            let path = model_dir.join(&file.path);
            let handle = File::open(&path).with_context(|| format!("open {}", path.display()))?;
            let mmap = unsafe { MmapOptions::new().map(&handle) }
                .with_context(|| format!("mmap {}", path.display()))?;
            files.insert(file.path.clone(), MappedSafetensorFile { mmap });
        }
        let tensor_files = layout
            .tensors
            .iter()
            .map(|tensor| (tensor.name.clone(), tensor.file.clone()))
            .collect();
        Ok(Self {
            model_dir,
            layout,
            files,
            tensor_files,
        })
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.layout
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
    }

    pub fn with_tensor<R>(
        &self,
        name: &str,
        f: impl for<'data> FnOnce(TensorView<'data>) -> Result<R>,
    ) -> Result<R> {
        let file = self
            .tensor_files
            .get(name)
            .ok_or_else(|| anyhow!("tensor {name} is not present in model layout"))?;
        let shard = self
            .files
            .get(file)
            .ok_or_else(|| anyhow!("safetensors shard {file} is not mapped"))?;
        let safetensors = SafeTensors::deserialize(&shard.mmap).with_context(|| {
            format!("parse safetensors {}", self.model_dir.join(file).display())
        })?;
        let tensor = safetensors
            .tensor(name)
            .with_context(|| format!("read tensor {name} from {file}"))?;
        f(tensor)
    }
}

fn find_safetensors(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in
        fs::read_dir(model_dir).with_context(|| format!("read_dir {}", model_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext == "safetensors")
        {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen36_fp4_core::LayerType;
    use safetensors::tensor::{Dtype, TensorView, serialize};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn expected_topology_is_valid_without_files() {
        ModelTopology::expected_qwen36_text_mtp()
            .validate_qwen36()
            .unwrap();
    }

    #[test]
    fn mapped_model_reads_tensor_bytes_without_owning_the_tensor() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir);

        let data = [1_u8, 2, 3, 4];
        let view = TensorView::new(Dtype::BF16, vec![2], &data).unwrap();
        let bytes = serialize([("model.language_model.embed_tokens.weight", view)], None).unwrap();
        fs::write(dir.join("model.safetensors"), bytes).unwrap();

        let model = MappedModel::open(&dir).unwrap();
        let observed = model
            .with_tensor("model.language_model.embed_tokens.weight", |tensor| {
                Ok((
                    tensor.dtype(),
                    tensor.shape().to_vec(),
                    tensor.data().to_vec(),
                ))
            })
            .unwrap();

        assert_eq!(observed.0, Dtype::BF16);
        assert_eq!(observed.1, vec![2]);
        assert_eq!(observed.2, data);

        fs::remove_dir_all(dir).unwrap();
    }

    fn unique_temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("qwen36-loader-test-{nanos}"))
    }

    fn write_config(dir: &Path) {
        let topology = ModelTopology::expected_qwen36_text_mtp();
        let layer_types = topology
            .layer_types
            .iter()
            .map(|kind| match kind {
                LayerType::LinearAttention => "linear_attention",
                LayerType::FullAttention => "full_attention",
            })
            .collect::<Vec<_>>();
        let config = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": topology.hidden_size,
                "vocab_size": topology.vocab_size,
                "num_hidden_layers": topology.num_hidden_layers,
                "layer_types": layer_types,
                "intermediate_size": topology.intermediate_size,
                "max_position_embeddings": topology.max_position_embeddings,
                "mtp_num_hidden_layers": topology.mtp_num_hidden_layers,
                "full_attention_interval": topology.full_attention_interval,
                "num_attention_heads": topology.attention_num_heads,
                "num_key_value_heads": topology.attention_num_kv_heads,
                "head_dim": topology.attention_head_dim,
                "partial_rotary_factor": topology.partial_rotary_factor,
                "linear_num_key_heads": topology.linear_num_key_heads,
                "linear_num_value_heads": topology.linear_num_value_heads,
                "linear_key_head_dim": topology.linear_key_head_dim,
                "linear_value_head_dim": topology.linear_value_head_dim,
                "linear_conv_kernel_dim": topology.linear_conv_kernel_dim,
                "rope_parameters": {
                    "partial_rotary_factor": topology.partial_rotary_factor,
                    "rope_theta": topology.rope_theta,
                    "rope_type": "default"
                }
            }
        });
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }
}
