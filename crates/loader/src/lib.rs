use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use memmap2::MmapOptions;
use qwen36_fp4_core::{
    HuggingFaceConfig, LayoutFile, ModelLayout, ModelTopology, QuantizationSummary, TensorDtype,
    TensorInfo, QWEN36_TEXT_NVFP4_MTP_MODEL_ID,
};
use safetensors::SafeTensors;

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
        return Err(anyhow!("no .safetensors files found in {}", model_dir.display()));
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
    fs::write(output.as_ref(), bytes).with_context(|| format!("write {}", output.as_ref().display()))
}

fn find_safetensors(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(model_dir).with_context(|| format!("read_dir {}", model_dir.display()))? {
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

    #[test]
    fn expected_topology_is_valid_without_files() {
        ModelTopology::expected_qwen36_text_mtp()
            .validate_qwen36()
            .unwrap();
    }
}
