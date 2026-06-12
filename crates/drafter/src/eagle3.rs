use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct Eagle3Config {
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub draft_vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<Eagle3RopeScaling>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub eagle_config: Eagle3SubConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Eagle3RopeScaling {
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Eagle3SubConfig {
    pub eagle_aux_hidden_state_layer_ids: Vec<usize>,
    #[serde(default)]
    pub use_aux_hidden_state: bool,
    #[serde(default)]
    pub use_input_layernorm_in_first_layer: bool,
    #[serde(default)]
    pub use_last_layernorm: bool,
    #[serde(default)]
    pub use_mtp_layernorm: bool,
    #[serde(default)]
    pub next_layer_regular: bool,
    #[serde(default)]
    pub parallel_draft_step: usize,
    #[serde(default)]
    pub parallel_draft_heads_num_layers: usize,
}

impl Eagle3Config {
    pub fn load(drafter_dir: impl AsRef<Path>) -> Result<Self> {
        let path = drafter_dir.as_ref().join("config.json");
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let config: Self =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if !self
            .architectures
            .iter()
            .any(|arch| arch == "LlamaForCausalLMEagle3")
        {
            bail!(
                "unsupported EAGLE3 architectures {:?}; expected LlamaForCausalLMEagle3",
                self.architectures
            );
        }
        if self.num_hidden_layers != 1 {
            bail!(
                "EAGLE3 v1 integration supports exactly 1 drafter layer, got {}",
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
        if self.num_attention_heads * self.head_dim == 0
            || self.num_key_value_heads * self.head_dim == 0
        {
            bail!("invalid EAGLE3 attention shape");
        }
        if self
            .eagle_config
            .eagle_aux_hidden_state_layer_ids
            .is_empty()
        {
            bail!("eagle_config.eagle_aux_hidden_state_layer_ids is empty");
        }
        if self.draft_vocab_size == 0 || self.draft_vocab_size > self.vocab_size {
            bail!(
                "draft_vocab_size {} must be in 1..={}",
                self.draft_vocab_size,
                self.vocab_size,
            );
        }
        Ok(())
    }

    pub fn aux_layer_ids(&self) -> &[usize] {
        &self.eagle_config.eagle_aux_hidden_state_layer_ids
    }

    pub fn fc_in_features(&self) -> usize {
        self.aux_layer_ids().len() * self.hidden_size
    }

    pub fn attention_in_features(&self) -> usize {
        self.hidden_size * 2
    }

    pub fn q_proj_out(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_proj_out(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn uses_compressed_vocab(&self) -> bool {
        self.draft_vocab_size != self.vocab_size
    }
}

#[derive(Debug, Clone)]
pub struct Eagle3WeightRef {
    pub name: String,
    pub file: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub size_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct Eagle3LayerWeights {
    pub hidden_norm: Eagle3WeightRef,
    pub input_layernorm: Eagle3WeightRef,
    pub post_attention_layernorm: Eagle3WeightRef,
    pub q_proj: Eagle3WeightRef,
    pub k_proj: Eagle3WeightRef,
    pub v_proj: Eagle3WeightRef,
    pub o_proj: Eagle3WeightRef,
    pub mlp_gate_proj: Eagle3WeightRef,
    pub mlp_up_proj: Eagle3WeightRef,
    pub mlp_down_proj: Eagle3WeightRef,
}

impl Eagle3LayerWeights {
    pub fn iter(&self) -> [&Eagle3WeightRef; 10] {
        [
            &self.hidden_norm,
            &self.input_layernorm,
            &self.post_attention_layernorm,
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            &self.o_proj,
            &self.mlp_gate_proj,
            &self.mlp_up_proj,
            &self.mlp_down_proj,
        ]
    }
}

#[derive(Debug, Clone)]
pub struct Eagle3Manifest {
    pub layer: Eagle3LayerWeights,
    pub fc: Eagle3WeightRef,
    pub norm: Eagle3WeightRef,
    pub lm_head: Eagle3WeightRef,
    pub d2t: Option<Eagle3WeightRef>,
}

impl Eagle3Manifest {
    pub fn tensor_count(&self) -> usize {
        10 + 3 + usize::from(self.d2t.is_some())
    }
}

pub struct Eagle3Drafter {
    pub drafter_dir: PathBuf,
    pub config: Eagle3Config,
    pub manifest: Eagle3Manifest,
    pub d2t: Option<Vec<i64>>,
    pub target_token_ids: Option<Vec<u32>>,
    files: BTreeMap<String, Mmap>,
}

impl Eagle3Drafter {
    pub fn open(drafter_dir: impl AsRef<Path>) -> Result<Self> {
        let drafter_dir = drafter_dir.as_ref().to_path_buf();
        let config = Eagle3Config::load(&drafter_dir)?;

        let shards = find_safetensors(&drafter_dir)?;
        if shards.is_empty() {
            bail!("no .safetensors files found in {}", drafter_dir.display());
        }

        let mut files = BTreeMap::new();
        let mut tensors: BTreeMap<String, Eagle3WeightRef> = BTreeMap::new();
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
                    Eagle3WeightRef {
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
        let mut drafter = Self {
            drafter_dir,
            config,
            manifest,
            d2t: None,
            target_token_ids: None,
            files,
        };
        drafter.d2t = drafter.load_d2t()?;
        drafter.target_token_ids = drafter.build_target_token_ids()?;
        Ok(drafter)
    }

    pub fn with_tensor<R>(
        &self,
        name: &str,
        f: impl for<'data> FnOnce(TensorView<'data>) -> Result<R>,
    ) -> Result<R> {
        let weight = self
            .manifest_entry(name)
            .ok_or_else(|| anyhow!("EAGLE3 tensor {name} not present"))?;
        let mmap = self
            .files
            .get(&weight.file)
            .ok_or_else(|| anyhow!("safetensors shard {} not mapped", weight.file))?;
        let safetensors = SafeTensors::deserialize(mmap).with_context(|| {
            format!(
                "parse EAGLE3 shard {}",
                self.drafter_dir.join(&weight.file).display(),
            )
        })?;
        let tensor = safetensors
            .tensor(name)
            .with_context(|| format!("read EAGLE3 tensor {name}"))?;
        f(tensor)
    }

    pub fn map_draft_token(&self, draft_token: u32) -> Result<u32> {
        if !self.config.uses_compressed_vocab() {
            return Ok(draft_token);
        }
        let target_token_ids = self
            .target_token_ids
            .as_ref()
            .ok_or_else(|| anyhow!("compressed EAGLE3 drafter missing target token map"))?;
        let idx = draft_token as usize;
        let Some(&target) = target_token_ids.get(idx) else {
            bail!(
                "draft token {draft_token} outside target token map of len {}",
                target_token_ids.len()
            );
        };
        Ok(target)
    }

    fn manifest_entry(&self, name: &str) -> Option<&Eagle3WeightRef> {
        for entry in self.manifest.layer.iter() {
            if entry.name == name {
                return Some(entry);
            }
        }
        [
            &self.manifest.fc,
            &self.manifest.norm,
            &self.manifest.lm_head,
        ]
        .into_iter()
        .find(|entry| entry.name == name)
        .or_else(|| {
            self.manifest
                .d2t
                .as_ref()
                .filter(|entry| entry.name == name)
        })
    }

    fn load_d2t(&self) -> Result<Option<Vec<i64>>> {
        let Some(entry) = &self.manifest.d2t else {
            return Ok(None);
        };
        self.with_tensor(&entry.name, |tensor| {
            if tensor.dtype() != Dtype::I64 {
                bail!("d2t must be I64, got {:?}", tensor.dtype());
            }
            let data = tensor.data();
            if data.len() % 8 != 0 {
                bail!("d2t byte length {} is not divisible by 8", data.len());
            }
            Ok(Some(
                data.chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect(),
            ))
        })
    }

    fn build_target_token_ids(&self) -> Result<Option<Vec<u32>>> {
        if !self.config.uses_compressed_vocab() {
            return Ok(None);
        }
        let d2t = self
            .d2t
            .as_ref()
            .ok_or_else(|| anyhow!("compressed EAGLE3 drafter missing d2t map"))?;
        if d2t.len() != self.config.draft_vocab_size {
            bail!(
                "d2t has {} entries, expected {}",
                d2t.len(),
                self.config.draft_vocab_size,
            );
        }

        let mut target_token_ids = Vec::with_capacity(d2t.len());
        for (draft_token, &delta) in d2t.iter().enumerate() {
            let target = draft_token as i64 + delta;
            if target < 0 || target >= self.config.vocab_size as i64 {
                bail!(
                    "d2t mapped draft token {draft_token} to invalid target id {target} for vocab_size {}",
                    self.config.vocab_size,
                );
            }
            target_token_ids.push(target as u32);
        }
        Ok(Some(target_token_ids))
    }
}

fn build_manifest(
    config: &Eagle3Config,
    tensors: &BTreeMap<String, Eagle3WeightRef>,
) -> Result<Eagle3Manifest> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let attn_in = config.attention_in_features();
    let q_out = config.q_proj_out();
    let kv_out = config.kv_proj_out();
    let fc_in = config.fc_in_features();

    let take = |name: &str, shape: &[usize]| -> Result<Eagle3WeightRef> {
        let entry = tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing EAGLE3 tensor {name}"))?;
        if entry.shape != shape {
            bail!(
                "EAGLE3 tensor {name} has shape {:?}, expected {:?}",
                entry.shape,
                shape,
            );
        }
        if entry.dtype != "BF16" {
            bail!("EAGLE3 tensor {name} must be BF16, got {}", entry.dtype);
        }
        Ok(entry.clone())
    };

    let layer = Eagle3LayerWeights {
        hidden_norm: take("layers.0.hidden_norm.weight", &[hidden])?,
        input_layernorm: take("layers.0.input_layernorm.weight", &[hidden])?,
        post_attention_layernorm: take("layers.0.post_attention_layernorm.weight", &[hidden])?,
        q_proj: take("layers.0.self_attn.q_proj.weight", &[q_out, attn_in])?,
        k_proj: take("layers.0.self_attn.k_proj.weight", &[kv_out, attn_in])?,
        v_proj: take("layers.0.self_attn.v_proj.weight", &[kv_out, attn_in])?,
        o_proj: take("layers.0.self_attn.o_proj.weight", &[hidden, q_out])?,
        mlp_gate_proj: take("layers.0.mlp.gate_proj.weight", &[intermediate, hidden])?,
        mlp_up_proj: take("layers.0.mlp.up_proj.weight", &[intermediate, hidden])?,
        mlp_down_proj: take("layers.0.mlp.down_proj.weight", &[hidden, intermediate])?,
    };
    let fc = take("fc.weight", &[hidden, fc_in])?;
    let norm = take("norm.weight", &[hidden])?;
    let lm_head = take("lm_head.weight", &[config.draft_vocab_size, hidden])?;
    let d2t = if config.uses_compressed_vocab() {
        let entry = tensors
            .get("d2t")
            .ok_or_else(|| anyhow!("compressed EAGLE3 drafter missing d2t tensor"))?;
        if entry.shape != [config.draft_vocab_size] {
            bail!(
                "d2t has shape {:?}, expected [{}]",
                entry.shape,
                config.draft_vocab_size,
            );
        }
        if entry.dtype != "I64" {
            bail!("d2t must be I64, got {}", entry.dtype);
        }
        Some(entry.clone())
    } else {
        None
    };

    Ok(Eagle3Manifest {
        layer,
        fc,
        norm,
        lm_head,
        d2t,
    })
}

fn find_safetensors(root: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(root).with_context(|| format!("read_dir {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}
