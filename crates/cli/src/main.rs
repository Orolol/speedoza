#[cfg(feature = "cuda")]
use std::io::{self, Write};
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

use anyhow::Result;
#[cfg(not(feature = "cuda"))]
use anyhow::bail;
use clap::{Parser, Subcommand, ValueEnum};
use qwen36_fp4_core::{KvCacheDtype, MemoryBudget, ModelTopology, QWEN36_TEXT_NVFP4_MTP_MODEL_ID};
use qwen36_fp4_loader::{
    MappedModel, discover_model_layout_with_id, read_topology, write_model_layout_json,
};
use qwen36_fp4_runtime::{Engine, EngineConfig, LayerWeights, ModelWeightsManifest};
use qwen36_fp4_tokenizer::{ChatMessage, QwenTokenizer};

#[derive(Debug, Parser)]
#[command(name = "qwen36")]
#[command(about = "Qwen3.6-27B Text NVFP4 MTP single-stream runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Discover {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "model_layout.json")]
        output: PathBuf,
        #[arg(long, default_value = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP")]
        model_id: String,
    },
    InspectConfig {
        #[arg(long)]
        model_dir: PathBuf,
    },
    Budget {
        #[arg(long, default_value_t = 32768)]
        ctx: usize,
        #[arg(long, value_enum, default_value = "fp8")]
        kv: KvArg,
    },
    Tokenize {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        text: String,
        #[arg(long, default_value_t = false)]
        add_special_tokens: bool,
    },
    ValidateWeights {
        #[arg(long)]
        model_dir: PathBuf,
    },
    GpuLoad {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 2256)]
        max_context: usize,
    },
    Chat {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 3)]
        mtp_speculative_tokens: usize,
    },
    Bench {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 2000)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 256)]
        max_new_tokens: usize,
        #[arg(long, default_value = "x")]
        token_text: String,
        #[arg(long, default_value_t = 0)]
        mtp_speculative_tokens: usize,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum KvArg {
    Bf16,
    Fp8,
    Turboquant3,
    Turboquant35,
}

impl From<KvArg> for KvCacheDtype {
    fn from(value: KvArg) -> Self {
        match value {
            KvArg::Bf16 => Self::Bf16,
            KvArg::Fp8 => Self::Fp8,
            KvArg::Turboquant3 => Self::TurboQuant3,
            KvArg::Turboquant35 => Self::TurboQuant35,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Discover {
            model_dir,
            output,
            model_id,
        } => {
            let layout = discover_model_layout_with_id(&model_dir, model_id)?;
            write_model_layout_json(&layout, &output)?;
            println!(
                "wrote {} tensors across {} files to {}",
                layout.derived.tensor_count,
                layout.files.len(),
                output.display()
            );
        }
        Command::InspectConfig { model_dir } => {
            let topology = read_topology(&model_dir)?;
            println!("{}", serde_json::to_string_pretty(&topology)?);
        }
        Command::Budget { ctx, kv } => {
            let topology = ModelTopology::expected_qwen36_text_mtp();
            let budget = MemoryBudget::estimate(&topology, ctx, kv.into());
            println!("{}", serde_json::to_string_pretty(&budget)?);
        }
        Command::Tokenize {
            model_dir,
            text,
            add_special_tokens,
        } => {
            let tokenizer = QwenTokenizer::from_model_dir(model_dir)?;
            let tokens = tokenizer.encode(&text, add_special_tokens)?;
            println!("{}", serde_json::to_string(&tokens)?);
        }
        Command::ValidateWeights { model_dir } => {
            let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
            let manifest = ModelWeightsManifest::from_layout(&layout)?;
            let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
            let sampled_tensors = sample_weight_tensors(&mapped_model, &manifest)?;
            let full_attention_layers = manifest
                .layers
                .iter()
                .filter(|layer| matches!(layer, LayerWeights::FullAttention(_)))
                .count();
            let linear_attention_layers = manifest.layers.len() - full_attention_layers;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "layers": manifest.layers.len(),
                    "full_attention_layers": full_attention_layers,
                    "linear_attention_layers": linear_attention_layers,
                    "mtp_tensors": manifest.mtp_tensors.len(),
                    "embed_tokens": manifest.embed_tokens.name,
                    "final_norm": manifest.final_norm.name,
                    "lm_head": manifest.lm_head.name,
                    "sampled_tensors": sampled_tensors,
                }))?
            );
        }
        Command::GpuLoad {
            model_dir,
            max_context,
        } => {
            gpu_load(model_dir, max_context)?;
        }
        Command::Chat {
            model_dir,
            prompt,
            max_new_tokens,
            mtp_speculative_tokens,
        } => {
            run_chat(model_dir, prompt, max_new_tokens, mtp_speculative_tokens)?;
        }
        Command::Bench {
            model_dir,
            prompt_tokens,
            max_new_tokens,
            token_text,
            mtp_speculative_tokens,
        } => {
            run_bench(
                model_dir,
                prompt_tokens,
                max_new_tokens,
                token_text,
                mtp_speculative_tokens,
            )?;
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_chat(
    model_dir: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    mtp_speculative_tokens: usize,
) -> Result<()> {
    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let tokenizer = QwenTokenizer::from_model_dir(&model_dir)?;
    let messages = vec![ChatMessage {
        role: "user".to_owned(),
        content: prompt,
    }];
    let prompt_tokens = tokenizer.encode_chat(&messages, true)?;
    let config = EngineConfig {
        max_context: prompt_tokens.len().saturating_add(max_new_tokens).max(1),
        kv_cache_dtype: KvCacheDtype::Bf16,
        mtp_speculative_tokens,
        ..EngineConfig::default()
    };
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, config)?;
    engine.prefill(&prompt_tokens)?;

    let mut generated = Vec::new();
    for idx in 0..max_new_tokens {
        let token = engine.sample_greedy()?;
        generated.push(token);
        let text = tokenizer.decode(&[token], true)?;
        print!("{text}");
        io::stdout().flush()?;
        if token == 248044 {
            break;
        }
        if idx + 1 < max_new_tokens {
            engine.decode_sampled_queued()?;
        }
    }
    println!();
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_bench(
    model_dir: PathBuf,
    prompt_token_count: usize,
    max_new_tokens: usize,
    token_text: String,
    mtp_speculative_tokens: usize,
) -> Result<()> {
    let total_start = Instant::now();
    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let tokenizer = QwenTokenizer::from_model_dir(&model_dir)?;
    let prompt_tokens = synthetic_prompt_tokens(&tokenizer, &token_text, prompt_token_count)?;
    let config = EngineConfig {
        max_context: prompt_tokens.len().saturating_add(max_new_tokens).max(1),
        kv_cache_dtype: KvCacheDtype::Bf16,
        mtp_speculative_tokens,
        ..EngineConfig::default()
    };

    let load_start = Instant::now();
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, config)?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let prefill_start = Instant::now();
    engine.prefill(&prompt_tokens)?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    let decode_start = Instant::now();
    let mut generated = 0_usize;
    for idx in 0..max_new_tokens {
        let _token = engine.sample_greedy()?;
        generated += 1;
        if idx + 1 < max_new_tokens {
            engine.decode_sampled_queued()?;
        }
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let total_seconds = total_start.elapsed().as_secs_f64();

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "backend": "cuda",
            "prompt_tokens": prompt_tokens.len(),
            "generated_tokens": generated,
            "load_seconds": load_seconds,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "prefill_tokens_per_second": rate(prompt_tokens.len(), prefill_seconds),
            "decode_tokens_per_second": rate(generated, decode_seconds),
            "mtp_speculative_tokens": mtp_speculative_tokens,
        }))?
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn synthetic_prompt_tokens(
    tokenizer: &QwenTokenizer,
    token_text: &str,
    target_tokens: usize,
) -> Result<Vec<u32>> {
    if target_tokens == 0 {
        return Ok(Vec::new());
    }
    let seed = tokenizer.encode(token_text, false)?;
    let seed = if seed.is_empty() {
        tokenizer.encode("x", false)?
    } else {
        seed
    };
    if seed.is_empty() {
        anyhow::bail!("tokenizer produced no tokens for benchmark seed text");
    }
    let mut tokens = Vec::with_capacity(target_tokens);
    while tokens.len() < target_tokens {
        let remaining = target_tokens - tokens.len();
        tokens.extend(seed.iter().copied().take(remaining));
    }
    Ok(tokens)
}

#[cfg(feature = "cuda")]
fn rate(tokens: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        tokens as f64 / seconds
    } else {
        0.0
    }
}

#[cfg(not(feature = "cuda"))]
fn run_chat(
    model_dir: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    mtp_speculative_tokens: usize,
) -> Result<()> {
    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let tokenizer = QwenTokenizer::from_model_dir(&model_dir)?;
    let messages = vec![ChatMessage {
        role: "user".to_owned(),
        content: prompt,
    }];
    let prompt_tokens = tokenizer.encode_chat(&messages, true)?;
    let config = EngineConfig {
        mtp_speculative_tokens,
        ..EngineConfig::default()
    };
    let mut engine = Engine::no_cuda_with_weights(&layout, config)?;
    let _ = max_new_tokens;
    if let Err(err) = engine.prefill(&prompt_tokens) {
        bail!(
            "CUDA backend is not linked; prefill/decode cannot run with backend {}: {err}",
            engine.backend_name()
        );
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run_bench(
    _model_dir: PathBuf,
    _prompt_tokens: usize,
    _max_new_tokens: usize,
    _token_text: String,
    _mtp_speculative_tokens: usize,
) -> Result<()> {
    bail!("bench requires rebuilding qwen36 with --features cuda and the CUDA shared library")
}

#[cfg(feature = "cuda")]
fn gpu_load(model_dir: PathBuf, max_context: usize) -> Result<()> {
    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let config = EngineConfig {
        max_context,
        ..EngineConfig::default()
    };
    let engine = Engine::cuda_with_mapped_weights(&mapped_model, config)?;
    let (gpu_tensors, gpu_weight_bytes) = engine
        .gpu_weight_summary()
        .ok_or_else(|| anyhow::anyhow!("CUDA engine did not expose uploaded weights"))?;
    let gpu_buffer_bytes = engine
        .gpu_buffer_bytes()
        .ok_or_else(|| anyhow::anyhow!("CUDA engine did not expose runtime buffers"))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "backend": engine.backend_name(),
            "max_context": engine.config.max_context,
            "gpu_tensors": gpu_tensors,
            "gpu_weight_bytes": gpu_weight_bytes,
            "gpu_weight_gib": bytes_to_gib(gpu_weight_bytes),
            "gpu_runtime_buffer_bytes": gpu_buffer_bytes,
            "gpu_runtime_buffer_gib": bytes_to_gib(gpu_buffer_bytes),
        }))?
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn gpu_load(_model_dir: PathBuf, _max_context: usize) -> Result<()> {
    bail!("gpu-load requires rebuilding qwen36 with --features cuda and the CUDA shared library")
}

#[cfg(feature = "cuda")]
fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

fn sample_weight_tensors(
    model: &MappedModel,
    manifest: &ModelWeightsManifest,
) -> Result<Vec<serde_json::Value>> {
    representative_weight_names(manifest)
        .into_iter()
        .map(|name| {
            let tensor_name = name.clone();
            model.with_tensor(&name, |tensor| {
                Ok(serde_json::json!({
                    "name": tensor_name,
                    "dtype": format!("{:?}", tensor.dtype()),
                    "shape": tensor.shape().to_vec(),
                    "size_bytes": tensor.data().len(),
                }))
            })
        })
        .collect()
}

fn representative_weight_names(manifest: &ModelWeightsManifest) -> Vec<String> {
    let mut names = vec![
        manifest.embed_tokens.name.clone(),
        manifest.final_norm.name.clone(),
        manifest.lm_head.name.clone(),
    ];
    if let Some(layer) = manifest.layers.first() {
        match layer {
            LayerWeights::LinearAttention(layer) => {
                names.push(layer.in_proj_qkv.weight().name.clone());
                names.push(layer.conv1d_weight.name.clone());
            }
            LayerWeights::FullAttention(layer) => {
                names.push(layer.q_proj.weight().name.clone());
                names.push(layer.k_norm.name.clone());
            }
        }
    }
    if let Some(tensor) = manifest.mtp_tensors.first() {
        names.push(tensor.name.clone());
    }
    names
}
