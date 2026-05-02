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

#[cfg(feature = "cuda")]
const DEFAULT_MTP_MAX_PROMPT_TOKENS: usize = 1_000_000;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct MtpSchedule {
    requested_tokens: usize,
    effective_tokens: usize,
    max_prompt_tokens: usize,
    auto_disabled: bool,
}

#[cfg(feature = "cuda")]
fn mtp_schedule(requested_tokens: usize, prompt_tokens: usize) -> MtpSchedule {
    let max_prompt_tokens = std::env::var("QWEN36_MTP_MAX_PROMPT_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MTP_MAX_PROMPT_TOKENS);
    let auto_disabled = requested_tokens > 0 && prompt_tokens > max_prompt_tokens;
    MtpSchedule {
        requested_tokens,
        effective_tokens: if auto_disabled { 0 } else { requested_tokens },
        max_prompt_tokens,
        auto_disabled,
    }
}

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
    CudaDiag,
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
        #[arg(long, default_value_t = 0)]
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
        #[arg(long)]
        max_context: Option<usize>,
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
        #[arg(long)]
        max_context: Option<usize>,
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
        Command::CudaDiag => {
            #[cfg(feature = "cuda")]
            {
                let diagnostics = qwen36_fp4_runtime::cuda_diagnostics()?;
                println!("{}", serde_json::to_string_pretty(&diagnostics)?);
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("cuda-diag requires rebuilding qwen36 with --features cuda")
            }
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
            max_context,
        } => {
            #[cfg(feature = "cuda")]
            {
                run_dump_logits(model_dir, prompt, chat_template, top_k, out, max_context)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (model_dir, prompt, chat_template, top_k, out, max_context);
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
            max_context,
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
                    max_context,
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
                    max_context,
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
    let mtp_schedule = mtp_schedule(mtp_speculative_tokens, prompt_tokens.len());
    let config = EngineConfig {
        max_context: prompt_tokens.len().saturating_add(max_new_tokens).max(1),
        kv_cache_dtype: KvCacheDtype::Bf16,
        mtp_speculative_tokens: mtp_schedule.effective_tokens,
        ..EngineConfig::default()
    };
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, config)?;
    engine.prefill(&prompt_tokens)?;

    if mtp_schedule.effective_tokens > 0 {
        if mtp_schedule.effective_tokens > 1 {
            return run_chat_mtp_multi(
                &mut engine,
                &tokenizer,
                &prompt_tokens,
                max_new_tokens,
                mtp_schedule.effective_tokens,
            );
        }
        engine.queue_sample_greedy_to_current_token()?;
        qwen36_fp4_runtime::cuda_synchronize()?;
        let mut current_token = engine.read_current_token()?;
        engine.prepare_mtp_prefill_from_sampled(&prompt_tokens, current_token)?;
        engine.queue_sample_greedy()?;
        qwen36_fp4_runtime::cuda_synchronize()?;
        let mut draft_token = engine.read_sampled_token()?;
        let mut generated = 0_usize;
        let mut emitted_tokens = Vec::new();
        let mut accepted_draft_tokens = 0_usize;
        let mut rejected_draft_tokens = 0_usize;

        while generated < max_new_tokens {
            let text = tokenizer.decode(&[current_token], true)?;
            print!("{text}");
            io::stdout().flush()?;
            emitted_tokens.push(current_token);
            generated += 1;
            if current_token == 248044 || generated >= max_new_tokens {
                break;
            }

            let remaining_after_current = max_new_tokens - generated;
            let need_next_token = remaining_after_current > 1;
            let need_next_draft = remaining_after_current > 2;
            let verify = engine.verify_mtp_draft_two_tokens(
                current_token,
                draft_token,
                need_next_token,
                need_next_draft,
            )?;
            if verify.accepted {
                accepted_draft_tokens += 1;
                let text = tokenizer.decode(&[draft_token], true)?;
                print!("{text}");
                io::stdout().flush()?;
                emitted_tokens.push(draft_token);
                generated += 1;
                if draft_token == 248044 || generated >= max_new_tokens {
                    break;
                }
                current_token = verify.next_token.ok_or_else(|| {
                    anyhow::anyhow!("accepted MTP verification did not return next_token")
                })?;
                if let Some(next_draft_token) = verify.next_draft_token {
                    draft_token = next_draft_token;
                }
            } else {
                rejected_draft_tokens += 1;
                // The engine has already rolled back the verify chunk and
                // committed `current_token` at `start_position` — including a
                // fresh MTP draft sampled from the newly-verified hidden
                // state, when the loop will iterate at least once more.
                current_token = verify.verified_token;
                if let Some(next_draft) = verify.next_draft_token {
                    draft_token = next_draft;
                }
            }
        }
        if std::env::var("QWEN36_MTP_STATS").is_ok() {
            let drafts = accepted_draft_tokens + rejected_draft_tokens;
            eprintln!(
                "mtp.stats accepted={} rejected={} acceptance_rate={:.4}",
                accepted_draft_tokens,
                rejected_draft_tokens,
                if drafts > 0 {
                    accepted_draft_tokens as f64 / drafts as f64
                } else {
                    0.0
                }
            );
        }
        println!();
        return Ok(());
    }

    let mut generated = Vec::new();
    let mut graph_enabled = false;
    engine.queue_sample_greedy()?;
    for idx in 0..max_new_tokens {
        qwen36_fp4_runtime::cuda_synchronize()?;
        let token = engine.read_sampled_token()?;
        generated.push(token);
        let text = tokenizer.decode(&[token], true)?;
        print!("{text}");
        io::stdout().flush()?;
        if token == 248044 {
            break;
        }
        if idx + 1 < max_new_tokens {
            if graph_enabled {
                engine.decode_graph_step()?;
            } else {
                engine.enable_decode_graph()?;
                graph_enabled = true;
            }
        }
    }
    if graph_enabled {
        engine.disable_decode_graph()?;
    }
    println!();
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_chat_mtp_multi(
    engine: &mut Engine<qwen36_fp4_runtime::CudaBackend>,
    tokenizer: &QwenTokenizer,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    draft_window: usize,
) -> Result<()> {
    engine.queue_sample_greedy_to_current_token()?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    let mut current_token = engine.read_current_token()?;
    let mut draft_tokens =
        engine.prepare_mtp_drafts_from_sampled(prompt_tokens, current_token, draft_window)?;

    let mut generated = 0_usize;
    let mut accepted_draft_tokens = 0_usize;
    let mut rejected_draft_tokens = 0_usize;

    while generated < max_new_tokens {
        let text = tokenizer.decode(&[current_token], true)?;
        print!("{text}");
        io::stdout().flush()?;
        generated += 1;
        if current_token == 248044 || generated >= max_new_tokens {
            break;
        }
        if draft_tokens.is_empty() {
            anyhow::bail!("MTP multi loop lost its draft window before generation completed");
        }

        let remaining_after_current = max_new_tokens - generated;
        let verify_count = draft_tokens.len().min(remaining_after_current);
        let need_next_token_on_full_accept = remaining_after_current > verify_count;
        let verify = engine.verify_mtp_draft_tokens(
            current_token,
            &draft_tokens[..verify_count],
            need_next_token_on_full_accept,
            draft_window,
        )?;
        accepted_draft_tokens += verify.accepted_drafts;
        if verify.rejected {
            rejected_draft_tokens += 1;
        }

        let mut stopped = false;
        for token in draft_tokens.iter().copied().take(verify.accepted_drafts) {
            let text = tokenizer.decode(&[token], true)?;
            print!("{text}");
            io::stdout().flush()?;
            generated += 1;
            if token == 248044 || generated >= max_new_tokens {
                stopped = true;
                break;
            }
        }
        if stopped {
            break;
        }

        current_token = verify.next_token.ok_or_else(|| {
            anyhow::anyhow!("MTP multi verification did not return the next current token")
        })?;
        draft_tokens = verify.next_draft_tokens;
    }

    if std::env::var("QWEN36_MTP_STATS").is_ok() {
        let drafts = accepted_draft_tokens + rejected_draft_tokens;
        eprintln!(
            "mtp.stats accepted={} rejected={} acceptance_rate={:.4}",
            accepted_draft_tokens,
            rejected_draft_tokens,
            if drafts > 0 {
                accepted_draft_tokens as f64 / drafts as f64
            } else {
                0.0
            }
        );
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
    max_context: Option<usize>,
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
        max_context: max_context
            .unwrap_or_else(|| prompt_tokens.len().max(1))
            .max(prompt_tokens.len())
            .max(1),
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
    max_context: Option<usize>,
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
        max_context: max_context
            .unwrap_or_else(|| prompt_tokens.len().saturating_add(1))
            .max(prompt_tokens.len().saturating_add(1))
            .max(1),
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
    if mtp_speculative_tokens > 3 {
        anyhow::bail!("bench currently supports --mtp-speculative-tokens 0..=3");
    }
    let total_start = Instant::now();
    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let tokenizer = QwenTokenizer::from_model_dir(&model_dir)?;
    let prompt_tokens = synthetic_prompt_tokens(&tokenizer, &token_text, prompt_token_count)?;
    let mtp_schedule = mtp_schedule(mtp_speculative_tokens, prompt_tokens.len());
    let config = EngineConfig {
        max_context: prompt_tokens.len().saturating_add(max_new_tokens).max(1),
        kv_cache_dtype: KvCacheDtype::Bf16,
        mtp_speculative_tokens: mtp_schedule.effective_tokens,
        ..EngineConfig::default()
    };

    let load_start = Instant::now();
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, config)?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let prefill_start = Instant::now();
    engine.prefill(&prompt_tokens)?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    if mtp_schedule.effective_tokens == 1 {
        return run_bench_mtp_one(
            engine,
            prompt_tokens,
            max_new_tokens,
            total_start,
            load_seconds,
            prefill_seconds,
            mtp_schedule,
        );
    }
    if mtp_schedule.effective_tokens > 1 {
        return run_bench_mtp_multi(
            engine,
            prompt_tokens,
            max_new_tokens,
            total_start,
            load_seconds,
            prefill_seconds,
            mtp_schedule,
        );
    }

    // Pipeline: queue sample-after-prefill, capture one decode+sample
    // iteration into a CUDA graph, then replay it for every remaining
    // token. The graph collapses ~600 host kernel launches per token into
    // a single cudaGraphLaunch and reads the position from a device-side
    // counter so the same recording works for every iteration.
    //
    // When QWEN36_PROFILE_DECODE_LAYERS=1 is set, fall back to the host
    // launch path (decode_sampled_queued + queue_sample_greedy) so that the
    // per-layer instrumentation in forward_device_token_cuda_inner fires —
    // graph capture would otherwise hide the per-section sync points the
    // profiler relies on.
    let profile_decode = std::env::var("QWEN36_PROFILE_DECODE_LAYERS").is_ok();
    qwen36_fp4_runtime::cuda_counters_reset()?;
    let decode_start = Instant::now();
    engine.queue_sample_greedy()?;
    let mut generated = 1_usize;
    if max_new_tokens > 1 {
        if profile_decode {
            for _ in 1..max_new_tokens {
                engine.decode_sampled_queued()?;
                engine.queue_sample_greedy()?;
                generated += 1;
            }
        } else {
            engine.enable_decode_graph()?;
            generated += 1;
            for _ in 2..max_new_tokens {
                engine.decode_graph_step()?;
                generated += 1;
            }
            engine.disable_decode_graph()?;
        }
    }
    qwen36_fp4_runtime::cuda_synchronize()?;
    let cuda_counters = qwen36_fp4_runtime::cuda_counters_read()?;
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
            "mtp_speculative_tokens": mtp_schedule.effective_tokens,
            "mtp_requested_speculative_tokens": mtp_schedule.requested_tokens,
            "mtp_auto_disabled": mtp_schedule.auto_disabled,
            "mtp_max_prompt_tokens": mtp_schedule.max_prompt_tokens,
            "cuda_counters_decode": cuda_counters,
        }))?
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_bench_mtp_one(
    mut engine: Engine<qwen36_fp4_runtime::CudaBackend>,
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    total_start: Instant,
    load_seconds: f64,
    prefill_seconds: f64,
    mtp_schedule: MtpSchedule,
) -> Result<()> {
    qwen36_fp4_runtime::cuda_counters_reset()?;
    let decode_start = Instant::now();
    if max_new_tokens == 0 {
        qwen36_fp4_runtime::cuda_synchronize()?;
        let cuda_counters = qwen36_fp4_runtime::cuda_counters_read()?;
        let decode_seconds = decode_start.elapsed().as_secs_f64();
        let total_seconds = total_start.elapsed().as_secs_f64();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "backend": "cuda",
                "prompt_tokens": prompt_tokens.len(),
                "generated_tokens": 0,
                "load_seconds": load_seconds,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "total_seconds": total_seconds,
                "prefill_tokens_per_second": rate(prompt_tokens.len(), prefill_seconds),
                "decode_tokens_per_second": 0.0,
                "mtp_speculative_tokens": mtp_schedule.effective_tokens,
                "mtp_requested_speculative_tokens": mtp_schedule.requested_tokens,
                "mtp_auto_disabled": mtp_schedule.auto_disabled,
                "mtp_max_prompt_tokens": mtp_schedule.max_prompt_tokens,
                "mtp_accepted_draft_tokens": 0,
                "mtp_rejected_draft_tokens": 0,
                "mtp_acceptance_rate": 0.0,
                "main_decode_steps": 0,
                "mtp_decode_steps": 0,
                "cuda_counters_decode": cuda_counters,
            }))?
        );
        return Ok(());
    }

    let setup_start = Instant::now();
    engine.queue_sample_greedy_to_current_token()?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    let mut current_token = engine.read_current_token()?;
    engine.prepare_mtp_prefill_from_sampled(&prompt_tokens, current_token)?;
    engine.queue_sample_greedy()?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    let mut draft_token = engine.read_sampled_token()?;
    let mtp_setup_seconds = setup_start.elapsed().as_secs_f64();

    let mut generated = 0_usize;
    let mut emitted_tokens = Vec::new();
    let mut accepted_draft_tokens = 0_usize;
    let mut rejected_draft_tokens = 0_usize;
    let mut main_decode_steps = 0_usize;
    let mut mtp_decode_steps = 1_usize;
    // Full reset+reprefill rebuilds should stay at 0 now that rejection recovery
    // happens in-engine. Track rollback recoveries separately so rejection-heavy
    // runs remain visible in the bench JSON.
    let rebuilds = 0_usize;
    let mut rollback_recoveries = 0_usize;
    let mut mtp_verify_seconds = 0.0_f64;

    while generated < max_new_tokens {
        emitted_tokens.push(current_token);
        generated += 1;
        if generated >= max_new_tokens {
            break;
        }

        let remaining_after_current = max_new_tokens - generated;
        let need_next_token = remaining_after_current > 1;
        let need_next_draft = remaining_after_current > 2;
        let verify_start = Instant::now();
        let verify = engine.verify_mtp_draft_two_tokens(
            current_token,
            draft_token,
            need_next_token,
            need_next_draft,
        )?;
        mtp_verify_seconds += verify_start.elapsed().as_secs_f64();
        main_decode_steps += 1;
        if verify.accepted {
            accepted_draft_tokens += 1;
            emitted_tokens.push(draft_token);
            generated += 1;
            if verify.next_draft_token.is_some() {
                mtp_decode_steps += 1;
            }
            if generated >= max_new_tokens {
                break;
            }
            current_token = verify.next_token.ok_or_else(|| {
                anyhow::anyhow!("accepted MTP verification did not return next_token")
            })?;
            if let Some(next_draft_token) = verify.next_draft_token {
                draft_token = next_draft_token;
            }
        } else {
            rejected_draft_tokens += 1;
            rollback_recoveries += 1;
            // S3: in-engine rollback. No reset + reprefill needed; the
            // verify path restored the recurrent state, committed
            // `current_token`, and produced the next draft directly.
            current_token = verify.verified_token;
            if let Some(next_draft) = verify.next_draft_token {
                draft_token = next_draft;
            }
            mtp_decode_steps += 1;
        }
    }
    qwen36_fp4_runtime::cuda_synchronize()?;
    let cuda_counters = qwen36_fp4_runtime::cuda_counters_read()?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let total_seconds = total_start.elapsed().as_secs_f64();
    let evaluated_drafts = accepted_draft_tokens + rejected_draft_tokens;
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
            "mtp_speculative_tokens": mtp_schedule.effective_tokens,
            "mtp_requested_speculative_tokens": mtp_schedule.requested_tokens,
            "mtp_auto_disabled": mtp_schedule.auto_disabled,
            "mtp_max_prompt_tokens": mtp_schedule.max_prompt_tokens,
            "mtp_accepted_draft_tokens": accepted_draft_tokens,
            "mtp_rejected_draft_tokens": rejected_draft_tokens,
            "mtp_acceptance_rate": if evaluated_drafts > 0 {
                accepted_draft_tokens as f64 / evaluated_drafts as f64
            } else {
                0.0
            },
            "main_decode_steps": main_decode_steps,
            "mtp_decode_steps": mtp_decode_steps,
            "mtp_rollback_recoveries": rollback_recoveries,
            "mtp_rebuilds": rebuilds,
            "mtp_setup_seconds": mtp_setup_seconds,
            "mtp_verify_seconds": mtp_verify_seconds,
            "cuda_counters_decode": cuda_counters,
        }))?
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_bench_mtp_multi(
    mut engine: Engine<qwen36_fp4_runtime::CudaBackend>,
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    total_start: Instant,
    load_seconds: f64,
    prefill_seconds: f64,
    mtp_schedule: MtpSchedule,
) -> Result<()> {
    qwen36_fp4_runtime::cuda_counters_reset()?;
    let decode_start = Instant::now();
    if max_new_tokens == 0 {
        qwen36_fp4_runtime::cuda_synchronize()?;
        let cuda_counters = qwen36_fp4_runtime::cuda_counters_read()?;
        let decode_seconds = decode_start.elapsed().as_secs_f64();
        let total_seconds = total_start.elapsed().as_secs_f64();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "backend": "cuda",
                "prompt_tokens": prompt_tokens.len(),
                "generated_tokens": 0,
                "load_seconds": load_seconds,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "total_seconds": total_seconds,
                "prefill_tokens_per_second": rate(prompt_tokens.len(), prefill_seconds),
                "decode_tokens_per_second": 0.0,
                "mtp_speculative_tokens": mtp_schedule.effective_tokens,
                "mtp_requested_speculative_tokens": mtp_schedule.requested_tokens,
                "mtp_auto_disabled": mtp_schedule.auto_disabled,
                "mtp_max_prompt_tokens": mtp_schedule.max_prompt_tokens,
                "mtp_accepted_draft_tokens": 0,
                "mtp_rejected_draft_tokens": 0,
                "mtp_acceptance_rate": 0.0,
                "main_decode_steps": 0,
                "mtp_decode_steps": 0,
                "mtp_rollback_recoveries": 0,
                "mtp_rebuilds": 0,
                "cuda_counters_decode": cuda_counters,
            }))?
        );
        return Ok(());
    }

    let draft_window = mtp_schedule.effective_tokens;
    let setup_start = Instant::now();
    engine.queue_sample_greedy_to_current_token()?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    let mut current_token = engine.read_current_token()?;
    let mut draft_tokens =
        engine.prepare_mtp_drafts_from_sampled(&prompt_tokens, current_token, draft_window)?;
    let mtp_setup_seconds = setup_start.elapsed().as_secs_f64();

    let mut generated = 0_usize;
    let mut accepted_draft_tokens = 0_usize;
    let mut rejected_draft_tokens = 0_usize;
    let mut main_decode_steps = 0_usize;
    let mut mtp_decode_steps = draft_tokens.len();
    let mut rollback_recoveries = 0_usize;
    let rebuilds = 0_usize;
    let mut mtp_verify_seconds = 0.0_f64;

    while generated < max_new_tokens {
        generated += 1;
        if generated >= max_new_tokens {
            break;
        }
        if draft_tokens.is_empty() {
            anyhow::bail!("MTP multi loop lost its draft window before generation completed");
        }

        let remaining_after_current = max_new_tokens - generated;
        let verify_count = draft_tokens.len().min(remaining_after_current);
        let need_next_token_on_full_accept = remaining_after_current > verify_count;
        let verify_start = Instant::now();
        let verify = engine.verify_mtp_draft_tokens(
            current_token,
            &draft_tokens[..verify_count],
            need_next_token_on_full_accept,
            draft_window,
        )?;
        mtp_verify_seconds += verify_start.elapsed().as_secs_f64();
        main_decode_steps += 1;
        accepted_draft_tokens += verify.accepted_drafts;
        if verify.rejected {
            rejected_draft_tokens += 1;
            rollback_recoveries += 1;
        }

        generated += verify.accepted_drafts;
        if generated >= max_new_tokens {
            break;
        }
        current_token = verify.next_token.ok_or_else(|| {
            anyhow::anyhow!("MTP multi verification did not return the next current token")
        })?;
        mtp_decode_steps += verify.next_draft_tokens.len();
        draft_tokens = verify.next_draft_tokens;
    }

    qwen36_fp4_runtime::cuda_synchronize()?;
    let cuda_counters = qwen36_fp4_runtime::cuda_counters_read()?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let total_seconds = total_start.elapsed().as_secs_f64();
    let evaluated_drafts = accepted_draft_tokens + rejected_draft_tokens;
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
            "mtp_speculative_tokens": mtp_schedule.effective_tokens,
            "mtp_requested_speculative_tokens": mtp_schedule.requested_tokens,
            "mtp_auto_disabled": mtp_schedule.auto_disabled,
            "mtp_max_prompt_tokens": mtp_schedule.max_prompt_tokens,
            "mtp_accepted_draft_tokens": accepted_draft_tokens,
            "mtp_rejected_draft_tokens": rejected_draft_tokens,
            "mtp_acceptance_rate": if evaluated_drafts > 0 {
                accepted_draft_tokens as f64 / evaluated_drafts as f64
            } else {
                0.0
            },
            "main_decode_steps": main_decode_steps,
            "mtp_decode_steps": mtp_decode_steps,
            "mtp_rollback_recoveries": rollback_recoveries,
            "mtp_rebuilds": rebuilds,
            "mtp_setup_seconds": mtp_setup_seconds,
            "mtp_verify_seconds": mtp_verify_seconds,
            "cuda_counters_decode": cuda_counters,
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
