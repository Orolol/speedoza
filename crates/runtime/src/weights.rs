use std::collections::BTreeMap;

use qwen36_fp4_core::{CoreError, LayerType, ModelLayout, Result, TensorInfo, TensorRole};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeightsManifest {
    pub embed_tokens: TensorInfo,
    pub final_norm: TensorInfo,
    pub lm_head: TensorInfo,
    pub layers: Vec<LayerWeights>,
    pub mtp: Option<MtpWeights>,
    pub mtp_tensors: Vec<TensorInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LayerWeights {
    LinearAttention(Box<LinearAttentionLayerWeights>),
    FullAttention(Box<FullAttentionLayerWeights>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonLayerWeights {
    pub input_layernorm: TensorInfo,
    pub post_attention_layernorm: TensorInfo,
    pub mlp_gate_proj: LinearWeightBinding,
    pub mlp_up_proj: LinearWeightBinding,
    pub mlp_down_proj: LinearWeightBinding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearAttentionLayerWeights {
    pub layer_index: usize,
    pub common: CommonLayerWeights,
    pub in_proj_qkv: LinearWeightBinding,
    pub in_proj_z: LinearWeightBinding,
    pub in_proj_b: LinearWeightBinding,
    pub in_proj_a: LinearWeightBinding,
    pub out_proj: LinearWeightBinding,
    pub conv1d_weight: TensorInfo,
    pub dt_bias: TensorInfo,
    pub a_log: TensorInfo,
    pub norm_weight: TensorInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullAttentionLayerWeights {
    pub layer_index: usize,
    pub common: CommonLayerWeights,
    pub q_proj: LinearWeightBinding,
    pub k_proj: LinearWeightBinding,
    pub v_proj: LinearWeightBinding,
    pub o_proj: LinearWeightBinding,
    pub q_norm: TensorInfo,
    pub k_norm: TensorInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpWeights {
    pub fc: LinearWeightBinding,
    pub pre_fc_norm_embedding: TensorInfo,
    pub pre_fc_norm_hidden: TensorInfo,
    pub layers: Vec<FullAttentionLayerWeights>,
    pub norm: TensorInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "dtype", rename_all = "snake_case")]
pub enum LinearWeightBinding {
    Nvfp4 {
        weight: Box<TensorInfo>,
        block_scale: Box<TensorInfo>,
        tensor_scale: Box<TensorInfo>,
        input_scale: Box<TensorInfo>,
    },
    Bf16 {
        weight: Box<TensorInfo>,
    },
}

impl ModelWeightsManifest {
    pub fn from_layout(layout: &ModelLayout) -> Result<Self> {
        let lookup = TensorLookup::new(&layout.tensors);
        let embed_tokens = lookup.required_any(&[
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
        ])?;
        let final_norm = lookup.required_any(&[
            "model.language_model.norm.weight",
            "model.norm.weight",
            "language_model.norm.weight",
        ])?;
        let lm_head = lookup.required_any(&["lm_head.weight", "model.lm_head.weight"])?;
        let mtp_tensors = layout
            .tensors
            .iter()
            .filter(|tensor| tensor.name.starts_with("mtp."))
            .cloned()
            .collect::<Vec<_>>();

        if layout.topology.mtp_num_hidden_layers > 0 && mtp_tensors.is_empty() {
            return Err(missing_tensor(
                "mtp.*",
                "config declares MTP layers but no mtp.* tensors were found",
            ));
        }
        let mtp = if layout.topology.mtp_num_hidden_layers > 0 {
            let fc = lookup.required_linear(&["mtp".to_owned()], "fc")?;
            let pre_fc_norm_embedding =
                lookup.required_any(&["mtp.pre_fc_norm_embedding.weight"])?;
            let pre_fc_norm_hidden = lookup.required_any(&["mtp.pre_fc_norm_hidden.weight"])?;
            let norm = lookup.required_any(&["mtp.norm.weight"])?;
            let mut layers = Vec::with_capacity(layout.topology.mtp_num_hidden_layers);
            for mtp_layer_index in 0..layout.topology.mtp_num_hidden_layers {
                let prefixes = vec![format!("mtp.layers.{mtp_layer_index}")];
                let common = CommonLayerWeights {
                    input_layernorm: lookup.required_suffix(&prefixes, "input_layernorm.weight")?,
                    post_attention_layernorm: lookup
                        .required_suffix(&prefixes, "post_attention_layernorm.weight")?,
                    mlp_gate_proj: lookup.required_linear(&prefixes, "mlp.gate_proj")?,
                    mlp_up_proj: lookup.required_linear(&prefixes, "mlp.up_proj")?,
                    mlp_down_proj: lookup.required_linear(&prefixes, "mlp.down_proj")?,
                };
                layers.push(FullAttentionLayerWeights {
                    layer_index: layout.topology.num_hidden_layers + mtp_layer_index,
                    common,
                    q_proj: lookup.required_linear(&prefixes, "self_attn.q_proj")?,
                    k_proj: lookup.required_linear(&prefixes, "self_attn.k_proj")?,
                    v_proj: lookup.required_linear(&prefixes, "self_attn.v_proj")?,
                    o_proj: lookup.required_linear(&prefixes, "self_attn.o_proj")?,
                    q_norm: lookup.required_suffix(&prefixes, "self_attn.q_norm.weight")?,
                    k_norm: lookup.required_suffix(&prefixes, "self_attn.k_norm.weight")?,
                });
            }
            Some(MtpWeights {
                fc,
                pre_fc_norm_embedding,
                pre_fc_norm_hidden,
                layers,
                norm,
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(layout.topology.num_hidden_layers);
        for (layer_index, layer_type) in layout.topology.layer_types.iter().copied().enumerate() {
            let prefixes = layer_prefixes(layer_index);
            let common = CommonLayerWeights {
                input_layernorm: lookup.required_suffix(&prefixes, "input_layernorm.weight")?,
                post_attention_layernorm: lookup
                    .required_suffix(&prefixes, "post_attention_layernorm.weight")?,
                mlp_gate_proj: lookup.required_linear(&prefixes, "mlp.gate_proj")?,
                mlp_up_proj: lookup.required_linear(&prefixes, "mlp.up_proj")?,
                mlp_down_proj: lookup.required_linear(&prefixes, "mlp.down_proj")?,
            };

            let layer = match layer_type {
                LayerType::LinearAttention => {
                    LayerWeights::LinearAttention(Box::new(LinearAttentionLayerWeights {
                        layer_index,
                        common,
                        in_proj_qkv: lookup
                            .required_linear(&prefixes, "linear_attn.in_proj_qkv")?,
                        in_proj_z: lookup.required_linear(&prefixes, "linear_attn.in_proj_z")?,
                        in_proj_b: lookup.required_linear(&prefixes, "linear_attn.in_proj_b")?,
                        in_proj_a: lookup.required_linear(&prefixes, "linear_attn.in_proj_a")?,
                        out_proj: lookup.required_linear(&prefixes, "linear_attn.out_proj")?,
                        conv1d_weight: lookup
                            .required_suffix(&prefixes, "linear_attn.conv1d.weight")?,
                        dt_bias: lookup.required_suffix(&prefixes, "linear_attn.dt_bias")?,
                        a_log: lookup.required_suffix(&prefixes, "linear_attn.A_log")?,
                        norm_weight: lookup
                            .required_suffix(&prefixes, "linear_attn.norm.weight")?,
                    }))
                }
                LayerType::FullAttention => {
                    LayerWeights::FullAttention(Box::new(FullAttentionLayerWeights {
                        layer_index,
                        common,
                        q_proj: lookup.required_linear(&prefixes, "self_attn.q_proj")?,
                        k_proj: lookup.required_linear(&prefixes, "self_attn.k_proj")?,
                        v_proj: lookup.required_linear(&prefixes, "self_attn.v_proj")?,
                        o_proj: lookup.required_linear(&prefixes, "self_attn.o_proj")?,
                        q_norm: lookup.required_suffix(&prefixes, "self_attn.q_norm.weight")?,
                        k_norm: lookup.required_suffix(&prefixes, "self_attn.k_norm.weight")?,
                    }))
                }
            };
            layers.push(layer);
        }

        Ok(Self {
            embed_tokens,
            final_norm,
            lm_head,
            layers,
            mtp,
            mtp_tensors,
        })
    }

    pub fn tensor_infos(&self) -> Vec<&TensorInfo> {
        self.tensor_infos_for_upload(true)
    }

    pub fn tensor_infos_for_upload(&self, include_mtp: bool) -> Vec<&TensorInfo> {
        let mut tensors = vec![&self.embed_tokens, &self.final_norm, &self.lm_head];
        for layer in &self.layers {
            match layer {
                LayerWeights::LinearAttention(layer) => {
                    layer.common.append_tensor_infos(&mut tensors);
                    layer.in_proj_qkv.append_tensor_infos(&mut tensors);
                    layer.in_proj_z.append_tensor_infos(&mut tensors);
                    layer.in_proj_b.append_tensor_infos(&mut tensors);
                    layer.in_proj_a.append_tensor_infos(&mut tensors);
                    layer.out_proj.append_tensor_infos(&mut tensors);
                    tensors.extend([
                        &layer.conv1d_weight,
                        &layer.dt_bias,
                        &layer.a_log,
                        &layer.norm_weight,
                    ]);
                }
                LayerWeights::FullAttention(layer) => {
                    layer.common.append_tensor_infos(&mut tensors);
                    layer.q_proj.append_tensor_infos(&mut tensors);
                    layer.k_proj.append_tensor_infos(&mut tensors);
                    layer.v_proj.append_tensor_infos(&mut tensors);
                    layer.o_proj.append_tensor_infos(&mut tensors);
                    tensors.extend([&layer.q_norm, &layer.k_norm]);
                }
            }
        }
        if include_mtp {
            tensors.extend(self.mtp_tensors.iter());
        }
        tensors
    }
}

impl MtpWeights {
    pub fn layer(&self, mtp_layer_index: usize) -> Option<&FullAttentionLayerWeights> {
        let len = self.layers.len();
        if len == 0 {
            None
        } else {
            self.layers.get(mtp_layer_index % len)
        }
    }
}

impl LayerWeights {
    pub fn layer_index(&self) -> usize {
        match self {
            Self::LinearAttention(layer) => layer.layer_index,
            Self::FullAttention(layer) => layer.layer_index,
        }
    }
}

impl LinearWeightBinding {
    pub fn weight(&self) -> &TensorInfo {
        match self {
            Self::Nvfp4 { weight, .. } | Self::Bf16 { weight } => weight,
        }
    }

    pub fn is_nvfp4(&self) -> bool {
        matches!(self, Self::Nvfp4 { .. })
    }

    pub fn tensor_infos(&self) -> Vec<&TensorInfo> {
        let mut tensors = Vec::new();
        self.append_tensor_infos(&mut tensors);
        tensors
    }

    fn append_tensor_infos<'a>(&'a self, tensors: &mut Vec<&'a TensorInfo>) {
        match self {
            Self::Nvfp4 {
                weight,
                block_scale,
                tensor_scale,
                input_scale,
            } => tensors.extend([
                weight.as_ref(),
                block_scale.as_ref(),
                tensor_scale.as_ref(),
                input_scale.as_ref(),
            ]),
            Self::Bf16 { weight } => tensors.push(weight.as_ref()),
        }
    }
}

impl CommonLayerWeights {
    fn append_tensor_infos<'a>(&'a self, tensors: &mut Vec<&'a TensorInfo>) {
        tensors.extend([&self.input_layernorm, &self.post_attention_layernorm]);
        self.mlp_gate_proj.append_tensor_infos(tensors);
        self.mlp_up_proj.append_tensor_infos(tensors);
        self.mlp_down_proj.append_tensor_infos(tensors);
    }
}

struct TensorLookup<'a> {
    tensors: BTreeMap<&'a str, &'a TensorInfo>,
}

impl<'a> TensorLookup<'a> {
    fn new(tensors: &'a [TensorInfo]) -> Self {
        Self {
            tensors: tensors
                .iter()
                .map(|tensor| (tensor.name.as_str(), tensor))
                .collect(),
        }
    }

    fn required_any(&self, names: &[&str]) -> Result<TensorInfo> {
        names
            .iter()
            .find_map(|name| self.tensors.get(name).copied())
            .cloned()
            .ok_or_else(|| missing_tensor(names.join("|"), "required tensor is missing"))
    }

    fn required_suffix(&self, prefixes: &[String], suffix: &str) -> Result<TensorInfo> {
        let candidates = prefixes
            .iter()
            .map(|prefix| format!("{prefix}.{suffix}"))
            .collect::<Vec<_>>();
        self.required_owned_any(&candidates)
    }

    fn required_linear(&self, prefixes: &[String], suffix: &str) -> Result<LinearWeightBinding> {
        let weight = self.required_suffix(prefixes, &format!("{suffix}.weight"))?;
        match weight.role {
            TensorRole::Nvfp4PackedWeight => {
                let block_scale =
                    self.required_suffix(prefixes, &format!("{suffix}.weight_scale"))?;
                let tensor_scale =
                    self.required_suffix(prefixes, &format!("{suffix}.weight_scale_2"))?;
                let input_scale =
                    self.required_suffix(prefixes, &format!("{suffix}.input_scale"))?;
                Ok(LinearWeightBinding::Nvfp4 {
                    weight: Box::new(weight),
                    block_scale: Box::new(block_scale),
                    tensor_scale: Box::new(tensor_scale),
                    input_scale: Box::new(input_scale),
                })
            }
            TensorRole::Bf16Weight
            | TensorRole::LmHeadBf16
            | TensorRole::MtpBf16
            | TensorRole::Conv1dBf16
            | TensorRole::Embedding => Ok(LinearWeightBinding::Bf16 {
                weight: Box::new(weight),
            }),
            role => Err(CoreError::InvalidTensor {
                name: weight.name,
                reason: format!("expected NVFP4 or BF16 linear weight, got role {role:?}"),
            }),
        }
    }

    fn required_owned_any(&self, names: &[String]) -> Result<TensorInfo> {
        names
            .iter()
            .find_map(|name| self.tensors.get(name.as_str()).copied())
            .cloned()
            .ok_or_else(|| missing_tensor(names.join("|"), "required tensor is missing"))
    }
}

fn layer_prefixes(layer_index: usize) -> Vec<String> {
    [
        format!("model.language_model.layers.{layer_index}"),
        format!("model.layers.{layer_index}"),
        format!("language_model.layers.{layer_index}"),
    ]
    .into()
}

fn missing_tensor(name: impl Into<String>, reason: impl Into<String>) -> CoreError {
    CoreError::InvalidTensor {
        name: name.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen36_fp4_core::{ModelLayout, ModelTopology, QuantizationSummary, TensorDtype};

    #[test]
    fn builds_qwen36_manifest_from_expected_tensor_names() {
        let topology = ModelTopology::expected_qwen36_text_mtp();
        let mut tensors = vec![
            tensor(
                "model.language_model.embed_tokens.weight",
                TensorDtype::Bf16,
            ),
            tensor("model.language_model.norm.weight", TensorDtype::Bf16),
            tensor("lm_head.weight", TensorDtype::Bf16),
        ];
        add_mtp_tensors(&mut tensors);

        for (layer_index, layer_type) in topology.layer_types.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{layer_index}");
            tensors.push(tensor(
                &format!("{prefix}.input_layernorm.weight"),
                TensorDtype::Bf16,
            ));
            tensors.push(tensor(
                &format!("{prefix}.post_attention_layernorm.weight"),
                TensorDtype::Bf16,
            ));
            add_linear(&mut tensors, &prefix, "mlp.gate_proj", TensorDtype::U8);
            add_linear(&mut tensors, &prefix, "mlp.up_proj", TensorDtype::U8);
            add_linear(&mut tensors, &prefix, "mlp.down_proj", TensorDtype::U8);

            match layer_type {
                LayerType::LinearAttention => {
                    add_linear(
                        &mut tensors,
                        &prefix,
                        "linear_attn.in_proj_qkv",
                        TensorDtype::U8,
                    );
                    add_linear(
                        &mut tensors,
                        &prefix,
                        "linear_attn.in_proj_z",
                        TensorDtype::U8,
                    );
                    add_linear(
                        &mut tensors,
                        &prefix,
                        "linear_attn.in_proj_b",
                        TensorDtype::U8,
                    );
                    add_linear(
                        &mut tensors,
                        &prefix,
                        "linear_attn.in_proj_a",
                        TensorDtype::U8,
                    );
                    add_linear(
                        &mut tensors,
                        &prefix,
                        "linear_attn.out_proj",
                        TensorDtype::U8,
                    );
                    tensors.push(tensor(
                        &format!("{prefix}.linear_attn.conv1d.weight"),
                        TensorDtype::Bf16,
                    ));
                    tensors.push(tensor(
                        &format!("{prefix}.linear_attn.dt_bias"),
                        TensorDtype::Bf16,
                    ));
                    tensors.push(tensor(
                        &format!("{prefix}.linear_attn.A_log"),
                        TensorDtype::Bf16,
                    ));
                    tensors.push(tensor(
                        &format!("{prefix}.linear_attn.norm.weight"),
                        TensorDtype::Bf16,
                    ));
                }
                LayerType::FullAttention => {
                    add_linear(&mut tensors, &prefix, "self_attn.q_proj", TensorDtype::U8);
                    add_linear(&mut tensors, &prefix, "self_attn.k_proj", TensorDtype::U8);
                    add_linear(&mut tensors, &prefix, "self_attn.v_proj", TensorDtype::U8);
                    add_linear(&mut tensors, &prefix, "self_attn.o_proj", TensorDtype::U8);
                    tensors.push(tensor(
                        &format!("{prefix}.self_attn.q_norm.weight"),
                        TensorDtype::Bf16,
                    ));
                    tensors.push(tensor(
                        &format!("{prefix}.self_attn.k_norm.weight"),
                        TensorDtype::Bf16,
                    ));
                }
            }
        }

        let layout = ModelLayout::from_parts(
            "test".to_owned(),
            topology,
            QuantizationSummary::default(),
            Vec::new(),
            tensors,
        );
        let manifest = ModelWeightsManifest::from_layout(&layout).unwrap();

        assert_eq!(manifest.layers.len(), 64);
        assert_eq!(manifest.mtp_tensors.len(), 15);
        assert!(matches!(
            manifest.mtp.as_ref().and_then(|mtp| mtp.layer(0)),
            Some(layer) if matches!(layer.q_proj, LinearWeightBinding::Bf16 { .. })
        ));
        assert!(
            manifest
                .tensor_infos()
                .iter()
                .any(|tensor| tensor.name.starts_with("mtp."))
        );
        assert!(
            !manifest
                .tensor_infos_for_upload(false)
                .iter()
                .any(|tensor| tensor.name.starts_with("mtp."))
        );
        assert!(matches!(
            &manifest.layers[0],
            LayerWeights::LinearAttention(layer) if layer.in_proj_qkv.is_nvfp4()
        ));
        assert!(matches!(
            &manifest.layers[3],
            LayerWeights::FullAttention(layer) if layer.q_proj.is_nvfp4()
        ));
    }

    #[test]
    fn rejects_missing_nvfp4_scale() {
        let topology = ModelTopology::expected_qwen36_text_mtp();
        let layout = ModelLayout::from_parts(
            "test".to_owned(),
            topology,
            QuantizationSummary::default(),
            Vec::new(),
            vec![
                tensor(
                    "model.language_model.embed_tokens.weight",
                    TensorDtype::Bf16,
                ),
                tensor("model.language_model.norm.weight", TensorDtype::Bf16),
                tensor("lm_head.weight", TensorDtype::Bf16),
                tensor("mtp.fc.weight", TensorDtype::Bf16),
                tensor("mtp.pre_fc_norm_embedding.weight", TensorDtype::Bf16),
                tensor("mtp.pre_fc_norm_hidden.weight", TensorDtype::Bf16),
                tensor("mtp.norm.weight", TensorDtype::Bf16),
                tensor("mtp.layers.0.input_layernorm.weight", TensorDtype::Bf16),
                tensor(
                    "mtp.layers.0.post_attention_layernorm.weight",
                    TensorDtype::Bf16,
                ),
                tensor("mtp.layers.0.mlp.gate_proj.weight", TensorDtype::Bf16),
                tensor("mtp.layers.0.mlp.up_proj.weight", TensorDtype::Bf16),
                tensor("mtp.layers.0.mlp.down_proj.weight", TensorDtype::Bf16),
                tensor("mtp.layers.0.self_attn.q_proj.weight", TensorDtype::Bf16),
                tensor("mtp.layers.0.self_attn.k_proj.weight", TensorDtype::Bf16),
                tensor("mtp.layers.0.self_attn.v_proj.weight", TensorDtype::Bf16),
                tensor("mtp.layers.0.self_attn.o_proj.weight", TensorDtype::Bf16),
                tensor("mtp.layers.0.self_attn.q_norm.weight", TensorDtype::Bf16),
                tensor("mtp.layers.0.self_attn.k_norm.weight", TensorDtype::Bf16),
                tensor(
                    "model.language_model.layers.0.input_layernorm.weight",
                    TensorDtype::Bf16,
                ),
                tensor(
                    "model.language_model.layers.0.post_attention_layernorm.weight",
                    TensorDtype::Bf16,
                ),
                tensor(
                    "model.language_model.layers.0.mlp.gate_proj.weight",
                    TensorDtype::U8,
                ),
            ],
        );

        let err = ModelWeightsManifest::from_layout(&layout).unwrap_err();
        assert!(err.to_string().contains("weight_scale"));
    }

    fn add_linear(tensors: &mut Vec<TensorInfo>, prefix: &str, suffix: &str, dtype: TensorDtype) {
        tensors.push(tensor(&format!("{prefix}.{suffix}.weight"), dtype.clone()));
        if dtype == TensorDtype::U8 {
            tensors.push(tensor(
                &format!("{prefix}.{suffix}.weight_scale"),
                TensorDtype::F8E4M3,
            ));
            tensors.push(tensor(
                &format!("{prefix}.{suffix}.weight_scale_2"),
                TensorDtype::F32,
            ));
            tensors.push(tensor(
                &format!("{prefix}.{suffix}.input_scale"),
                TensorDtype::F32,
            ));
        }
    }

    fn add_mtp_tensors(tensors: &mut Vec<TensorInfo>) {
        add_linear(tensors, "mtp", "fc", TensorDtype::Bf16);
        tensors.push(tensor(
            "mtp.pre_fc_norm_embedding.weight",
            TensorDtype::Bf16,
        ));
        tensors.push(tensor("mtp.pre_fc_norm_hidden.weight", TensorDtype::Bf16));
        tensors.push(tensor("mtp.norm.weight", TensorDtype::Bf16));
        let prefix = "mtp.layers.0";
        tensors.push(tensor(
            &format!("{prefix}.input_layernorm.weight"),
            TensorDtype::Bf16,
        ));
        tensors.push(tensor(
            &format!("{prefix}.post_attention_layernorm.weight"),
            TensorDtype::Bf16,
        ));
        add_linear(tensors, prefix, "mlp.gate_proj", TensorDtype::Bf16);
        add_linear(tensors, prefix, "mlp.up_proj", TensorDtype::Bf16);
        add_linear(tensors, prefix, "mlp.down_proj", TensorDtype::Bf16);
        add_linear(tensors, prefix, "self_attn.q_proj", TensorDtype::Bf16);
        add_linear(tensors, prefix, "self_attn.k_proj", TensorDtype::Bf16);
        add_linear(tensors, prefix, "self_attn.v_proj", TensorDtype::Bf16);
        add_linear(tensors, prefix, "self_attn.o_proj", TensorDtype::Bf16);
        tensors.push(tensor(
            &format!("{prefix}.self_attn.q_norm.weight"),
            TensorDtype::Bf16,
        ));
        tensors.push(tensor(
            &format!("{prefix}.self_attn.k_norm.weight"),
            TensorDtype::Bf16,
        ));
    }

    fn tensor(name: &str, dtype: TensorDtype) -> TensorInfo {
        TensorInfo::new(
            name.to_owned(),
            "model.safetensors".to_owned(),
            dtype,
            vec![1],
            1,
        )
    }
}
