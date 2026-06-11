use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use memmap2::{Mmap, MmapOptions};
use safetensors::{SafeTensors, tensor::TensorView};
use serde::Deserialize;

/// Parsed `config.json` of a z-lab DFlash drafter checkpoint (e.g.
/// `z-lab/Qwen3.6-27B-DFlash`). The drafter reuses the target model's
/// `embed_tokens` and `lm_head`, so those tensors are not part of the
/// drafter's own safetensors — `embed_tokens` / `lm_head` references stay
/// on the target side of the runtime.
#[derive(Debug, Clone, Deserialize)]
pub struct DFlashConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub sliding_window: usize,
    pub use_sliding_window: bool,
    pub layer_types: Vec<String>,
    pub num_target_layers: usize,
    pub dflash_config: DFlashSubConfig,
    pub block_size: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DFlashSubConfig {
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerAttentionKind {
    SlidingAttention,
    FullAttention,
}

impl LayerAttentionKind {
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "sliding_attention" => Ok(Self::SlidingAttention),
            "full_attention" => Ok(Self::FullAttention),
            other => bail!("unsupported drafter layer_type {other:?}"),
        }
    }
}

impl DFlashConfig {
    pub fn load(drafter_dir: impl AsRef<Path>) -> Result<Self> {
        let path = drafter_dir.as_ref().join("config.json");
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let config: Self =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.layer_types.len() != self.num_hidden_layers {
            bail!(
                "layer_types has {} entries but num_hidden_layers = {}",
                self.layer_types.len(),
                self.num_hidden_layers,
            );
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            bail!(
                "num_attention_heads ({}) not divisible by num_key_value_heads ({})",
                self.num_attention_heads,
                self.num_key_value_heads,
            );
        }
        if self.dflash_config.target_layer_ids.is_empty() {
            bail!("dflash_config.target_layer_ids is empty");
        }
        if self
            .dflash_config
            .target_layer_ids
            .iter()
            .any(|&id| id >= self.num_target_layers)
        {
            bail!(
                "dflash_config.target_layer_ids contains an index >= num_target_layers ({})",
                self.num_target_layers,
            );
        }
        for kind in &self.layer_types {
            LayerAttentionKind::from_str(kind)?;
        }
        Ok(())
    }

    pub fn layer_kind(&self, layer: usize) -> Result<LayerAttentionKind> {
        let kind = self
            .layer_types
            .get(layer)
            .ok_or_else(|| anyhow!("layer index {layer} out of range"))?;
        LayerAttentionKind::from_str(kind)
    }

    pub fn q_proj_out(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_proj_out(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// `target_hidden` is the concatenation of `len(target_layer_ids)` hidden
    /// states along the channel dim, then collapsed by `fc` back to
    /// `hidden_size`.
    pub fn fc_in_features(&self) -> usize {
        self.dflash_config.target_layer_ids.len() * self.hidden_size
    }

    /// `output_hidden_states[layer_id + 1]` is the output of target layer
    /// `layer_id` (index 0 of `output_hidden_states` is the embed output).
    /// Returns the indices into `output_hidden_states` that the drafter
    /// reads for conditioning.
    pub fn target_hidden_indices(&self) -> Vec<usize> {
        self.dflash_config
            .target_layer_ids
            .iter()
            .map(|&id| id + 1)
            .collect()
    }
}

/// Reference to one tensor inside the drafter's safetensors shard. The
/// drafter is a single-file checkpoint today (~3.5 GB), so `file` is
/// effectively constant.
#[derive(Debug, Clone)]
pub struct DFlashWeightRef {
    pub name: String,
    pub file: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub size_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct DFlashLayerWeights {
    pub kind: LayerAttentionKind,
    pub input_layernorm: DFlashWeightRef,
    pub post_attention_layernorm: DFlashWeightRef,
    pub q_proj: DFlashWeightRef,
    pub k_proj: DFlashWeightRef,
    pub v_proj: DFlashWeightRef,
    pub o_proj: DFlashWeightRef,
    pub q_norm: DFlashWeightRef,
    pub k_norm: DFlashWeightRef,
    pub mlp_gate_proj: DFlashWeightRef,
    pub mlp_up_proj: DFlashWeightRef,
    pub mlp_down_proj: DFlashWeightRef,
}

#[derive(Debug, Clone)]
pub struct DFlashManifest {
    pub layers: Vec<DFlashLayerWeights>,
    pub fc: DFlashWeightRef,
    pub hidden_norm: DFlashWeightRef,
    pub norm: DFlashWeightRef,
}

impl DFlashManifest {
    pub fn tensor_count(&self) -> usize {
        // 11 tensors per layer + fc + hidden_norm + norm.
        self.layers.len() * 11 + 3
    }
}

/// Owns the safetensors mmap and exposes byte slices for the drafter's
/// 58 tensors. Mirrors `qwen36_fp4_loader::MappedModel` but specialised
/// to the drafter's single-file layout.
pub struct DFlashDrafter {
    pub drafter_dir: PathBuf,
    pub config: DFlashConfig,
    pub manifest: DFlashManifest,
    files: BTreeMap<String, Mmap>,
}

impl DFlashDrafter {
    pub fn open(drafter_dir: impl AsRef<Path>) -> Result<Self> {
        let drafter_dir = drafter_dir.as_ref().to_path_buf();
        let config = DFlashConfig::load(&drafter_dir)?;

        let shards = find_safetensors(&drafter_dir)?;
        if shards.is_empty() {
            bail!("no .safetensors files found in {}", drafter_dir.display(),);
        }

        let mut files = BTreeMap::new();
        let mut tensors: BTreeMap<String, DFlashWeightRef> = BTreeMap::new();
        for path in &shards {
            let relative = path
                .strip_prefix(&drafter_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            let handle = File::open(path).with_context(|| format!("open {}", path.display()))?;
            let mmap = unsafe { MmapOptions::new().map(&handle) }
                .with_context(|| format!("mmap {}", path.display()))?;
            let safetensors = SafeTensors::deserialize(&mmap)
                .with_context(|| format!("parse safetensors {}", path.display()))?;
            for (name, view) in safetensors.tensors() {
                tensors.insert(
                    name.to_string(),
                    DFlashWeightRef {
                        name: name.to_string(),
                        file: relative.clone(),
                        dtype: format!("{:?}", view.dtype()),
                        shape: view.shape().to_vec(),
                        size_bytes: view.data().len(),
                    },
                );
            }
            files.insert(relative, mmap);
        }

        let manifest = build_manifest(&config, &tensors)?;

        Ok(Self {
            drafter_dir,
            config,
            manifest,
            files,
        })
    }

    /// Read a tensor's bytes via the underlying mmap. Cheap; safetensors
    /// is re-parsed per call but the mmap is reused.
    pub fn with_tensor<R>(
        &self,
        name: &str,
        f: impl for<'data> FnOnce(TensorView<'data>) -> Result<R>,
    ) -> Result<R> {
        let weight = self
            .manifest_entry(name)
            .ok_or_else(|| anyhow!("drafter tensor {name} not present"))?;
        let mmap = self
            .files
            .get(&weight.file)
            .ok_or_else(|| anyhow!("safetensors shard {} not mapped", weight.file))?;
        let safetensors = SafeTensors::deserialize(mmap).with_context(|| {
            format!(
                "parse drafter shard {}",
                self.drafter_dir.join(&weight.file).display(),
            )
        })?;
        let tensor = safetensors
            .tensor(name)
            .with_context(|| format!("read drafter tensor {name}"))?;
        f(tensor)
    }

    fn manifest_entry(&self, name: &str) -> Option<&DFlashWeightRef> {
        // Scan in declared order — 58 entries, cost is negligible. Avoids
        // duplicating a name → ref index that would have to be kept in
        // sync with `DFlashManifest` field changes.
        for layer in &self.manifest.layers {
            for entry in layer.iter() {
                if entry.name == name {
                    return Some(entry);
                }
            }
        }
        [
            &self.manifest.fc,
            &self.manifest.hidden_norm,
            &self.manifest.norm,
        ]
        .into_iter()
        .find(|entry| entry.name == name)
    }
}

impl DFlashLayerWeights {
    fn iter(&self) -> impl Iterator<Item = &DFlashWeightRef> {
        [
            &self.input_layernorm,
            &self.post_attention_layernorm,
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            &self.o_proj,
            &self.q_norm,
            &self.k_norm,
            &self.mlp_gate_proj,
            &self.mlp_up_proj,
            &self.mlp_down_proj,
        ]
        .into_iter()
    }
}

fn build_manifest(
    config: &DFlashConfig,
    tensors: &BTreeMap<String, DFlashWeightRef>,
) -> Result<DFlashManifest> {
    let take = |name: &str, expected_shape: &[usize]| -> Result<DFlashWeightRef> {
        let entry = tensors
            .get(name)
            .ok_or_else(|| anyhow!("drafter tensor {name} missing from safetensors"))?
            .clone();
        if entry.shape != expected_shape {
            bail!(
                "drafter tensor {name} shape {:?} != expected {:?}",
                entry.shape,
                expected_shape,
            );
        }
        if entry.dtype != "BF16" {
            bail!("drafter tensor {name} dtype {} != BF16", entry.dtype,);
        }
        Ok(entry)
    };

    let hidden = config.hidden_size;
    let q_out = config.q_proj_out();
    let kv_out = config.kv_proj_out();
    let intermediate = config.intermediate_size;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        let prefix = format!("layers.{layer_idx}");
        let layer = DFlashLayerWeights {
            kind: config.layer_kind(layer_idx)?,
            input_layernorm: take(&format!("{prefix}.input_layernorm.weight"), &[hidden])?,
            post_attention_layernorm: take(
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[hidden],
            )?,
            q_proj: take(
                &format!("{prefix}.self_attn.q_proj.weight"),
                &[q_out, hidden],
            )?,
            k_proj: take(
                &format!("{prefix}.self_attn.k_proj.weight"),
                &[kv_out, hidden],
            )?,
            v_proj: take(
                &format!("{prefix}.self_attn.v_proj.weight"),
                &[kv_out, hidden],
            )?,
            o_proj: take(
                &format!("{prefix}.self_attn.o_proj.weight"),
                &[hidden, q_out],
            )?,
            q_norm: take(
                &format!("{prefix}.self_attn.q_norm.weight"),
                &[config.head_dim],
            )?,
            k_norm: take(
                &format!("{prefix}.self_attn.k_norm.weight"),
                &[config.head_dim],
            )?,
            mlp_gate_proj: take(
                &format!("{prefix}.mlp.gate_proj.weight"),
                &[intermediate, hidden],
            )?,
            mlp_up_proj: take(
                &format!("{prefix}.mlp.up_proj.weight"),
                &[intermediate, hidden],
            )?,
            mlp_down_proj: take(
                &format!("{prefix}.mlp.down_proj.weight"),
                &[hidden, intermediate],
            )?,
        };
        layers.push(layer);
    }

    let fc = take("fc.weight", &[hidden, config.fc_in_features()])?;
    let hidden_norm = take("hidden_norm.weight", &[hidden])?;
    let norm = take("norm.weight", &[hidden])?;

    // Sanity: every safetensors entry should be claimed exactly once.
    let claimed: usize = layers.len() * 11 + 3;
    if claimed != tensors.len() {
        bail!(
            "drafter manifest claimed {claimed} tensors but safetensors holds {}",
            tensors.len(),
        );
    }

    Ok(DFlashManifest {
        layers,
        fc,
        hidden_norm,
        norm,
    })
}

fn find_safetensors(drafter_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in
        fs::read_dir(drafter_dir).with_context(|| format!("read_dir {}", drafter_dir.display()))?
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

    #[test]
    fn target_hidden_indices_apply_plus_one_offset() {
        let config = DFlashConfig {
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 5,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 248320,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            sliding_window: 2048,
            use_sliding_window: true,
            layer_types: vec![
                "sliding_attention".into(),
                "sliding_attention".into(),
                "sliding_attention".into(),
                "sliding_attention".into(),
                "full_attention".into(),
            ],
            num_target_layers: 64,
            dflash_config: DFlashSubConfig {
                mask_token_id: 248070,
                target_layer_ids: vec![1, 16, 31, 46, 61],
            },
            block_size: 16,
            attention_bias: false,
            tie_word_embeddings: false,
        };
        assert_eq!(config.target_hidden_indices(), vec![2, 17, 32, 47, 62]);
        assert_eq!(config.fc_in_features(), 5 * 5120);
        assert_eq!(config.q_proj_out(), 32 * 128);
        assert_eq!(config.kv_proj_out(), 8 * 128);
        config.validate().unwrap();
    }

    #[test]
    fn rejects_target_layer_id_out_of_range() {
        let mut config = base_config();
        config.dflash_config.target_layer_ids = vec![1, 64];
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("target_layer_ids"));
    }

    #[test]
    fn rejects_mismatched_layer_types_length() {
        let mut config = base_config();
        config.layer_types.pop();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("layer_types"));
    }

    fn base_config() -> DFlashConfig {
        DFlashConfig {
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 5,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 248320,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            sliding_window: 2048,
            use_sliding_window: true,
            layer_types: vec![
                "sliding_attention".into(),
                "sliding_attention".into(),
                "sliding_attention".into(),
                "sliding_attention".into(),
                "full_attention".into(),
            ],
            num_target_layers: 64,
            dflash_config: DFlashSubConfig {
                mask_token_id: 248070,
                target_layer_ids: vec![1, 16, 31, 46, 61],
            },
            block_size: 16,
            attention_bias: false,
            tie_word_embeddings: false,
        }
    }
}
