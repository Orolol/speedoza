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
    /// Dump the post-prefill logits (top-K, plus full vector to a binary file)
    /// so we can diff them against a reference forward pass for parity work.
    DumpLogits {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "What is 2+2?")]
        prompt: String,
        #[arg(long, default_value_t = false)]
        chat_template: bool,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run prefill, decode one explicit token, then dump post-decode logits.
    DumpDecode {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "hello")]
        prompt: String,
        #[arg(long, default_value_t = false)]
        chat_template: bool,
        #[arg(long)]
        decode_token_id: u32,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long)]
        out: Option<PathBuf>,
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
        Command::DumpLogits {
            model_dir,
            prompt,
            chat_template,
            top_k,
            out,
        } => {
            #[cfg(feature = "cuda")]
            {
                run_dump_logits(model_dir, prompt, chat_template, top_k, out)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (model_dir, prompt, chat_template, top_k, out);
                anyhow::bail!(
                    "dump-logits requires the cuda feature; rebuild with --features cuda"
                );
            }
        }
        Command::DumpDecode {
            model_dir,
            prompt,
            chat_template,
            decode_token_id,
            top_k,
            out,
        } => {
            #[cfg(feature = "cuda")]
            {
                run_dump_decode(
                    model_dir,
                    prompt,
                    chat_template,
                    decode_token_id,
                    top_k,
                    out,
                )?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (
                    model_dir,
                    prompt,
                    chat_template,
                    decode_token_id,
                    top_k,
                    out,
                );
                anyhow::bail!(
                    "dump-decode requires the cuda feature; rebuild with --features cuda"
                );
            }
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
fn run_dump_logits(
    model_dir: PathBuf,
    prompt: String,
    chat_template: bool,
    top_k: usize,
    out: Option<PathBuf>,
) -> Result<()> {
    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let tokenizer = QwenTokenizer::from_model_dir(&model_dir)?;

    let prompt_tokens = if chat_template {
        let messages = vec![ChatMessage {
            role: "user".to_owned(),
            content: prompt.clone(),
        }];
        tokenizer.encode_chat(&messages, true)?
    } else {
        tokenizer.encode(&prompt, true)?
    };

    eprintln!(
        "prompt {:?} -> {} tokens: {:?}",
        prompt,
        prompt_tokens.len(),
        prompt_tokens
    );

    let topology = qwen36_fp4_core::ModelTopology::expected_qwen36_text_mtp();
    let vocab = topology.vocab_size;

    let config = EngineConfig {
        max_context: prompt_tokens.len().max(1),
        kv_cache_dtype: KvCacheDtype::Bf16,
        ..EngineConfig::default()
    };
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, config)?;
    let forward = engine.prefill(&prompt_tokens)?;
    qwen36_fp4_runtime::cuda_synchronize()?;

    // Logits live on the device as `vocab` BF16 values for the last token.
    let logits_dev = forward.logits_device_ptr;
    if logits_dev == 0 {
        anyhow::bail!("prefill returned a null logits pointer");
    }
    let mut bf16_bytes = vec![0u8; vocab * 2];
    unsafe {
        let status =
            qwen36_cuda_memcpy_d2h(bf16_bytes.as_mut_ptr() as *mut _, logits_dev, vocab * 2);
        if status != 0 {
            anyhow::bail!("cudaMemcpy d2h returned status {status}");
        }
    }

    // BF16 -> f32 by zero-extending into the upper 16 bits of an f32.
    let mut logits = Vec::with_capacity(vocab);
    for chunk in bf16_bytes.chunks_exact(2) {
        let bits: u32 = (u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16;
        logits.push(f32::from_bits(bits));
    }

    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (i, v))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("top-{top_k} logits after prefill:");
    for (rank, (id, logit)) in indexed.iter().take(top_k).enumerate() {
        let text = tokenizer.decode(&[*id as u32], true).unwrap_or_default();
        println!("  #{rank:<2} id={id:<7} logit={logit:>10.4}  text={text:?}");
    }

    let finite = logits.iter().filter(|x| x.is_finite()).count();
    let nans = logits.iter().filter(|x| x.is_nan()).count();
    let infs = logits.iter().filter(|x| x.is_infinite()).count();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let sum: f64 = logits.iter().map(|x| *x as f64).sum();
    println!(
        "summary: vocab={vocab} finite={finite} nan={nans} inf={infs} max={max:.4} min={min:.4} mean={:.4}",
        sum / vocab as f64
    );

    if let Some(path) = out {
        std::fs::write(&path, &bf16_bytes)?;
        eprintln!(
            "wrote {} bytes (BF16 little-endian) to {}",
            bf16_bytes.len(),
            path.display()
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_dump_decode(
    model_dir: PathBuf,
    prompt: String,
    chat_template: bool,
    decode_token_id: u32,
    top_k: usize,
    out: Option<PathBuf>,
) -> Result<()> {
    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let tokenizer = QwenTokenizer::from_model_dir(&model_dir)?;

    let prompt_tokens = if chat_template {
        let messages = vec![ChatMessage {
            role: "user".to_owned(),
            content: prompt.clone(),
        }];
        tokenizer.encode_chat(&messages, true)?
    } else {
        tokenizer.encode(&prompt, true)?
    };
    let decode_text = tokenizer
        .decode(&[decode_token_id], true)
        .unwrap_or_else(|_| String::new());

    eprintln!(
        "prompt {:?} -> {} tokens: {:?}; decoding id={} text={:?}",
        prompt,
        prompt_tokens.len(),
        prompt_tokens,
        decode_token_id,
        decode_text
    );

    let topology = qwen36_fp4_core::ModelTopology::expected_qwen36_text_mtp();
    let vocab = topology.vocab_size;

    let config = EngineConfig {
        max_context: prompt_tokens.len().saturating_add(1).max(1),
        kv_cache_dtype: KvCacheDtype::Bf16,
        ..EngineConfig::default()
    };
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, config)?;
    engine.prefill(&prompt_tokens)?;
    let forward = engine.decode_one(decode_token_id)?;
    qwen36_fp4_runtime::cuda_synchronize()?;

    let logits_dev = forward.logits_device_ptr;
    if logits_dev == 0 {
        anyhow::bail!("decode returned a null logits pointer");
    }
    let mut bf16_bytes = vec![0u8; vocab * 2];
    unsafe {
        let status =
            qwen36_cuda_memcpy_d2h(bf16_bytes.as_mut_ptr() as *mut _, logits_dev, vocab * 2);
        if status != 0 {
            anyhow::bail!("cudaMemcpy d2h returned status {status}");
        }
    }

    let mut logits = Vec::with_capacity(vocab);
    for chunk in bf16_bytes.chunks_exact(2) {
        let bits: u32 = (u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16;
        logits.push(f32::from_bits(bits));
    }

    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (i, v))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("top-{top_k} logits after one decode step:");
    for (rank, (id, logit)) in indexed.iter().take(top_k).enumerate() {
        let text = tokenizer.decode(&[*id as u32], true).unwrap_or_default();
        println!("  #{rank:<2} id={id:<7} logit={logit:>10.4}  text={text:?}");
    }

    let finite = logits.iter().filter(|x| x.is_finite()).count();
    let nans = logits.iter().filter(|x| x.is_nan()).count();
    let infs = logits.iter().filter(|x| x.is_infinite()).count();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let sum: f64 = logits.iter().map(|x| *x as f64).sum();
    println!(
        "summary: vocab={vocab} finite={finite} nan={nans} inf={infs} max={max:.4} min={min:.4} mean={:.4}",
        sum / vocab as f64
    );

    if let Some(path) = out {
        std::fs::write(&path, &bf16_bytes)?;
        eprintln!(
            "wrote {} bytes (BF16 little-endian) to {}",
            bf16_bytes.len(),
            path.display()
        );
    }
    Ok(())
}

// FFI shim so we can read the device-side logits buffer for the parity
// dump. The runtime/kernels crates already link the CUDA kernel library,
// so the symbol is available transitively at link time.
#[cfg(feature = "cuda")]
#[link(name = "qwen36_fp4_kernels")]
unsafe extern "C" {
    fn qwen36_cuda_memcpy_d2h(dst: *mut core::ffi::c_void, src: u64, bytes: usize) -> i32;
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

    // Pipeline: queue sample-after-prefill, capture one decode+sample
    // iteration into a CUDA graph, then replay it for every remaining
    // token. The graph collapses ~600 host kernel launches per token into
    // a single cudaGraphLaunch and reads the position from a device-side
    // counter so the same recording works for every iteration.
    let decode_start = Instant::now();
    engine.queue_sample_greedy()?;
    let mut generated = 1_usize;
    if max_new_tokens > 1 {
        engine.enable_decode_graph()?;
        generated += 1;
        for _ in 2..max_new_tokens {
            engine.decode_graph_step()?;
            generated += 1;
        }
        engine.disable_decode_graph()?;
    }
    qwen36_fp4_runtime::cuda_synchronize()?;
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
