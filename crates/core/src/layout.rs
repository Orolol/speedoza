use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::budget::{KvCacheDtype, MemoryBudget};
use crate::config::{HuggingFaceConfig, ModelTopology};
use crate::dtype::TensorDtype;
use crate::tensor::{TensorInfo, TensorRole};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLayout {
    pub model_id: String,
    pub generated_by: String,
    pub topology: ModelTopology,
    pub quantization: QuantizationSummary,
    pub files: Vec<LayoutFile>,
    pub tensors: Vec<TensorInfo>,
    pub layers: Vec<LayerSummary>,
    pub derived: DerivedLayout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutFile {
    pub path: String,
    pub size_bytes: u64,
    pub tensor_count: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantizationSummary {
    pub quant_method: Option<String>,
    pub quant_algo: Option<String>,
    pub producer_name: Option<String>,
    pub producer_version: Option<String>,
    pub ignored_modules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSummary {
    pub layer_index: usize,
    pub tensor_count: usize,
    pub dtype_counts: BTreeMap<String, usize>,
    pub role_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedLayout {
    pub attention_layers: Vec<usize>,
    pub linear_attention_layers: Vec<usize>,
    pub turboquant_skip_layers: Vec<usize>,
    pub tensor_count: usize,
    pub total_safetensor_bytes: u64,
    pub nvfp4_weight_bytes: u64,
    pub bf16_weight_bytes: u64,
    pub fp8_scale_bytes: u64,
    pub ctx_32k_fp8_budget: MemoryBudget,
    pub warnings: Vec<String>,
}

impl ModelLayout {
    pub fn from_parts(
        model_id: String,
        topology: ModelTopology,
        quantization: QuantizationSummary,
        files: Vec<LayoutFile>,
        mut tensors: Vec<TensorInfo>,
    ) -> Self {
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        let layers = summarize_layers(&tensors);
        let total_safetensor_bytes = files.iter().map(|f| f.size_bytes).sum();
        let nvfp4_weight_bytes = tensors
            .iter()
            .filter(|t| t.role == TensorRole::Nvfp4PackedWeight)
            .map(|t| t.size_bytes)
            .sum();
        let bf16_weight_bytes = tensors
            .iter()
            .filter(|t| {
                matches!(
                    t.role,
                    TensorRole::Bf16Weight
                        | TensorRole::LmHeadBf16
                        | TensorRole::MtpBf16
                        | TensorRole::Conv1dBf16
                        | TensorRole::Embedding
                )
            })
            .map(|t| t.size_bytes)
            .sum();
        let fp8_scale_bytes = tensors
            .iter()
            .filter(|t| {
                matches!(
                    t.role,
                    TensorRole::Nvfp4BlockScale
                        | TensorRole::Nvfp4TensorScale
                        | TensorRole::Fp8Scale
                )
            })
            .map(|t| t.size_bytes)
            .sum();
        let warnings = derive_warnings(&topology, &tensors);
        let derived = DerivedLayout {
            attention_layers: topology.attention_layers(),
            linear_attention_layers: topology.linear_attention_layers(),
            turboquant_skip_layers: topology.turboquant_skip_layers(),
            tensor_count: tensors.len(),
            total_safetensor_bytes,
            nvfp4_weight_bytes,
            bf16_weight_bytes,
            fp8_scale_bytes,
            ctx_32k_fp8_budget: MemoryBudget::estimate(&topology, 32768, KvCacheDtype::Fp8),
            warnings,
        };
        Self {
            model_id,
            generated_by: "qwen36-fp4 loader".to_owned(),
            topology,
            quantization,
            files,
            tensors,
            layers,
            derived,
        }
    }
}

impl QuantizationSummary {
    pub fn from_hf_config(config: &HuggingFaceConfig) -> Self {
        let Some(value) = config.quantization_config.as_ref() else {
            return Self::default();
        };
        let quant_method = string_at(value, &["quant_method"]);
        let quant_algo = string_at(value, &["quant_algo"]);
        let producer_name = string_at(value, &["producer", "name"]);
        let producer_version = string_at(value, &["producer", "version"]);
        let ignored_modules = value
            .get("ignore")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.as_str().map(ToOwned::to_owned))
            .collect();
        Self {
            quant_method,
            quant_algo,
            producer_name,
            producer_version,
            ignored_modules,
        }
    }
}

fn string_at(value: &serde_json::Value, path: &[&str]) -> Option<String> {
    let mut cursor = value;
    for segment in path {
        cursor = cursor.get(*segment)?;
    }
    cursor.as_str().map(ToOwned::to_owned)
}

fn summarize_layers(tensors: &[TensorInfo]) -> Vec<LayerSummary> {
    let mut map: BTreeMap<usize, LayerSummary> = BTreeMap::new();
    for tensor in tensors {
        let Some(layer_index) = tensor.layer_index else {
            continue;
        };
        let entry = map.entry(layer_index).or_insert_with(|| LayerSummary {
            layer_index,
            tensor_count: 0,
            dtype_counts: BTreeMap::new(),
            role_counts: BTreeMap::new(),
        });
        entry.tensor_count += 1;
        *entry
            .dtype_counts
            .entry(dtype_key(&tensor.dtype))
            .or_insert(0) += 1;
        *entry
            .role_counts
            .entry(format!("{:?}", tensor.role))
            .or_insert(0) += 1;
    }
    map.into_values().collect()
}

fn dtype_key(dtype: &TensorDtype) -> String {
    match dtype {
        TensorDtype::Unknown(value) => format!("unknown:{value}"),
        other => format!("{other:?}"),
    }
}

fn derive_warnings(topology: &ModelTopology, tensors: &[TensorInfo]) -> Vec<String> {
    let mut warnings = Vec::new();
    let mtp_count = tensors
        .iter()
        .filter(|t| t.name.starts_with("mtp."))
        .count();
    if topology.mtp_num_hidden_layers > 0 && mtp_count == 0 {
        warnings.push("config declares MTP but no mtp.* tensors were found".to_owned());
    }
    let conv_count = tensors
        .iter()
        .filter(|t| t.name.contains("linear_attn.conv1d"))
        .count();
    if conv_count == 0 {
        warnings.push(
            "no linear_attn.conv1d tensors found; DeltaNet conv assumptions need verification"
                .to_owned(),
        );
    }
    warnings
}
