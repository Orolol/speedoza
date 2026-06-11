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
use qwen36_fp4_drafter::DFlashDrafter;
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

/// Extra `max_context` headroom needed when tree-MTP is active. Each tree
/// cycle writes one more K/V slot than chain MTP (the canonical leaf slot
/// past the chain), so a chunk that fits chain MTP exactly hits a
/// "position N exceeds max_context M" error mid-generation under tree.
/// 25% of `max_new_tokens` plus an 8-slot floor covers the worst case
/// (chain_depth=3, all cycles full-accept + leaf-accept).
#[cfg(feature = "cuda")]
fn tree_max_context_overhead(mtp_tree_leaves: usize, max_new_tokens: usize) -> usize {
    if mtp_tree_leaves > 1 {
        max_new_tokens / 4 + 8
    } else {
        0
    }
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

#[cfg(feature = "cuda")]
fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        matches!(
            value.as_str(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        )
    })
}

/// MTP online auto-fallback (recovery-plan step 4): after
/// `QWEN36_MTP_FALLBACK_WINDOW` verify cycles, if the observed per-draft
/// acceptance is below `QWEN36_MTP_FALLBACK_MIN_ACCEPTANCE`, stop
/// speculating and finish the generation on the plain decode graph —
/// `--mtp-speculative-tokens N` then never loses much to MTP=0 on
/// low-acceptance content. `QWEN36_MTP_AUTO_FALLBACK=0` disables (use for
/// pure-MTP perf measurements, e.g. dashboard before/after work).
#[cfg(feature = "cuda")]
fn mtp_auto_fallback_enabled() -> bool {
    !matches!(
        std::env::var("QWEN36_MTP_AUTO_FALLBACK").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

#[cfg(feature = "cuda")]
fn mtp_fallback_window() -> usize {
    std::env::var("QWEN36_MTP_FALLBACK_WINDOW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
}

/// Default 0.55: with today's verify-cycle cost (3.2-4.6x an MTP=0 token),
/// chain MTP-4 only breaks even above ~0.55 draft acceptance even at short
/// context. Lower this as the cycle gets cheaper.
#[cfg(feature = "cuda")]
fn mtp_fallback_min_acceptance() -> f64 {
    std::env::var("QWEN36_MTP_FALLBACK_MIN_ACCEPTANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.55)
}

#[cfg(feature = "cuda")]
fn cuda_kv_cache_dtype(default: KvCacheDtype) -> KvCacheDtype {
    match std::env::var("QWEN36_KV_CACHE_DTYPE")
        .ok()
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("bf16") => KvCacheDtype::Bf16,
        Some("fp8") => KvCacheDtype::Fp8,
        Some("turboquant3" | "tq3") => KvCacheDtype::TurboQuant3,
        Some("turboquant35" | "turboquant3.5" | "tq35" | "tq3.5") => KvCacheDtype::TurboQuant35,
        _ => default,
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
    /// Measure the interpreter substrate cost with chained no-op
    /// instructions. This is a design gate for full-stack decode programs:
    /// it times FALLBACK_TRAMPOLINE instructions plus interpreter barriers,
    /// without loading a model.
    InterpreterOverheadBench {
        /// No-op instruction counts to benchmark. Values are comma-separated.
        #[arg(long, value_delimiter = ',', default_value = "1,64,128,256,512")]
        instruction_counts: Vec<usize>,
        /// Explicit interpreter CTA counts to benchmark. Values are comma-separated.
        #[arg(long, value_delimiter = ',', default_value = "24,48,128")]
        cta_counts: Vec<u32>,
        #[arg(long, default_value_t = 20)]
        iterations: usize,
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        #[arg(long, default_value_t = false)]
        json: bool,
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
    /// Validate a z-lab DFlash drafter checkpoint (e.g.
    /// `z-lab/Qwen3.6-27B-DFlash`) against the expected tensor manifest
    /// derived from its `config.json`.
    ValidateDrafter {
        #[arg(long)]
        drafter_dir: PathBuf,
    },
    /// Upload a DFlash drafter checkpoint to the GPU and report per-tensor
    /// VRAM usage. Smoke for the drafter device path; no forward pass yet.
    DrafterLoad {
        #[arg(long)]
        drafter_dir: PathBuf,
    },
    /// End-to-end Phase E smoke: load target + drafter, prefill the
    /// target on a short synthetic prompt with the DFlash hidden-state
    /// capture hook armed, then run one drafter forward consuming the
    /// captured `target_hidden_raw`. Verifies finite output and reports
    /// per-component VRAM.
    DrafterHandoffSmoke {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        drafter_dir: PathBuf,
        #[arg(long, default_value_t = 32)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 16)]
        q_len: usize,
    },
    /// Phase F.0: run ONE DFlash speculative iteration end-to-end.
    /// Prefills target on a real chat prompt with hidden-state capture
    /// armed, samples the first target token, then asks the drafter to
    /// propose `block_size - 1` follow-up tokens via the propose helper
    /// (embed → drafter forward → lm_head → greedy argmax). Prints the
    /// proposed token ids + decoded text. Verify-back and multi-iter
    /// loop are Phase F.1.
    DrafterStepSmoke {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        drafter_dir: PathBuf,
        #[arg(long, default_value = "Hello, how are you")]
        prompt: String,
        #[arg(long, default_value_t = true)]
        chat_template: bool,
    },
    /// Phase F.1: full propose + verify cycle on ONE block. Same setup
    /// as drafter-step-smoke, then sequentially decodes [seed,
    /// drafted_0, … drafted_{k-1}] through the target with greedy
    /// argmax compares to find the accepted prefix. Prints accepted
    /// count + accepted tokens + bonus.
    DrafterIterSmoke {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        drafter_dir: PathBuf,
        #[arg(long, default_value = "The quick brown fox jumps over the")]
        prompt: String,
        #[arg(long, default_value_t = false)]
        chat_template: bool,
    },
    /// Phase F.2: multi-iter DFlash speculative chat. Stitches the
    /// propose+verify cycle into a loop driven by acceptance,
    /// capturing hidden states from each verify decode for the next
    /// iter's drafter call. Stops at `--max-new-tokens` or EOS.
    DrafterChatSmoke {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        drafter_dir: PathBuf,
        #[arg(long, default_value = "The quick brown fox jumps over the")]
        prompt: String,
        #[arg(long, default_value_t = false)]
        chat_template: bool,
        #[arg(long, default_value_t = 64)]
        max_new_tokens: usize,
    },
    /// Diagnostic: compare the engine's decode vs prefill numerics at
    /// the same input. Session A = prefill(prompt) then decode_one(seed);
    /// session B = prefill(prompt+[seed]). Both sessions reload the
    /// engine. Reports per-session argmax and cosine similarity of
    /// full BF16 logits. Used to test whether the decode kernel path
    /// drifts numerically from the prefill kernel path on NVFP4
    /// (low-AL diagnostic for Phase F.2 follow-up).
    DecodeVsPrefillCheck {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "The quick brown fox jumps over the")]
        prompt: String,
    },
    /// Run one DFlash drafter forward on synthetic random inputs and
    /// report basic sanity (finite values, deterministic). With
    /// --fixture-dir, load inputs + expected output from a fixture
    /// produced by `scripts/dflash_parity.py` and compare via cos sim.
    DrafterForwardSmoke {
        #[arg(long)]
        drafter_dir: PathBuf,
        #[arg(long, default_value_t = 16)]
        q_len: usize,
        #[arg(long, default_value_t = 16)]
        ctx_len: usize,
        #[arg(long, default_value_t = 2)]
        iterations: usize,
        /// Fixture directory containing noise.bf16, target_raw.bf16,
        /// positions.i32, expected_output.bf16, config.json. When set,
        /// inputs are loaded from disk and the output is parity-checked
        /// against expected_output.bf16 (cos sim ≥ 0.998).
        #[arg(long)]
        fixture_dir: Option<PathBuf>,
    },
    GpuLoad {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 2256)]
        max_context: usize,
        #[arg(long, default_value_t = 0)]
        mtp_speculative_tokens: usize,
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
        #[arg(long, default_value_t = 1)]
        mtp_tree_leaves: usize,
        /// Drafter for speculative decoding. `none` uses the
        /// chain/tree MTP path (with `--mtp-speculative-tokens`).
        /// `dflash` routes through the DFlash drafter end-to-end via
        /// batched verify, requires `--drafter-dir`.
        #[arg(long, value_enum, default_value = "none")]
        drafter: DrafterArg,
        #[arg(long)]
        drafter_dir: Option<PathBuf>,
    },
    Bench {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 2000)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 256)]
        max_new_tokens: usize,
        #[arg(long, default_value = "x", conflicts_with = "prompt_file")]
        token_text: String,
        /// Read prompt text from a UTF-8 file and use the first
        /// `--prompt-tokens` tokens after tokenization. Mutually exclusive
        /// with `--token-text` (which synthesizes a repeating prompt).
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        mtp_speculative_tokens: usize,
        #[arg(long, default_value_t = 1)]
        mtp_tree_leaves: usize,
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

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum DrafterArg {
    /// No DFlash drafter; the chat path uses chain/tree MTP via
    /// `--mtp-speculative-tokens`.
    None,
    /// DFlash drafter (`z-lab/Qwen3.6-27B-DFlash` style) end-to-end:
    /// drafter propose + batched verify per iteration.
    Dflash,
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
        Command::InterpreterOverheadBench {
            instruction_counts,
            cta_counts,
            iterations,
            warmup,
            json,
        } => {
            #[cfg(feature = "cuda")]
            {
                interpreter_overhead_bench(
                    &instruction_counts,
                    &cta_counts,
                    iterations,
                    warmup,
                    json,
                )?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (instruction_counts, cta_counts, iterations, warmup, json);
                anyhow::bail!(
                    "interpreter-overhead-bench requires rebuilding qwen36 with --features cuda"
                );
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
        Command::ValidateDrafter { drafter_dir } => {
            let drafter = DFlashDrafter::open(&drafter_dir)?;
            let config = &drafter.config;
            let sliding_layers = config
                .layer_types
                .iter()
                .filter(|kind| kind.as_str() == "sliding_attention")
                .count();
            let full_layers = config.layer_types.len() - sliding_layers;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "drafter_dir": drafter.drafter_dir.display().to_string(),
                    "hidden_size": config.hidden_size,
                    "intermediate_size": config.intermediate_size,
                    "num_hidden_layers": config.num_hidden_layers,
                    "sliding_attention_layers": sliding_layers,
                    "full_attention_layers": full_layers,
                    "sliding_window": config.sliding_window,
                    "num_attention_heads": config.num_attention_heads,
                    "num_key_value_heads": config.num_key_value_heads,
                    "head_dim": config.head_dim,
                    "vocab_size": config.vocab_size,
                    "block_size": config.block_size,
                    "mask_token_id": config.dflash_config.mask_token_id,
                    "target_layer_ids": config.dflash_config.target_layer_ids,
                    "target_hidden_indices": config.target_hidden_indices(),
                    "num_target_layers": config.num_target_layers,
                    "tensors_validated": drafter.manifest.tensor_count(),
                    "fc_shape": drafter.manifest.fc.shape,
                    "fc_in_features": config.fc_in_features(),
                }))?,
            );
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
            mtp_speculative_tokens,
        } => {
            gpu_load(model_dir, max_context, mtp_speculative_tokens)?;
        }
        Command::DrafterLoad { drafter_dir } => {
            #[cfg(feature = "cuda")]
            {
                drafter_load(drafter_dir)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = drafter_dir;
                anyhow::bail!(
                    "drafter-load requires the cuda feature; rebuild with --features cuda"
                );
            }
        }
        Command::DrafterStepSmoke {
            model_dir,
            drafter_dir,
            prompt,
            chat_template,
        } => {
            #[cfg(feature = "cuda")]
            {
                drafter_step_smoke(model_dir, drafter_dir, prompt, chat_template)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (model_dir, drafter_dir, prompt, chat_template);
                anyhow::bail!(
                    "drafter-step-smoke requires the cuda feature; rebuild with --features cuda"
                );
            }
        }
        Command::DrafterIterSmoke {
            model_dir,
            drafter_dir,
            prompt,
            chat_template,
        } => {
            #[cfg(feature = "cuda")]
            {
                drafter_iter_smoke(model_dir, drafter_dir, prompt, chat_template)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (model_dir, drafter_dir, prompt, chat_template);
                anyhow::bail!(
                    "drafter-iter-smoke requires the cuda feature; rebuild with --features cuda"
                );
            }
        }
        Command::DecodeVsPrefillCheck { model_dir, prompt } => {
            #[cfg(feature = "cuda")]
            {
                decode_vs_prefill_check(model_dir, prompt)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (model_dir, prompt);
                anyhow::bail!(
                    "decode-vs-prefill-check requires the cuda feature; rebuild with --features cuda"
                );
            }
        }
        Command::DrafterChatSmoke {
            model_dir,
            drafter_dir,
            prompt,
            chat_template,
            max_new_tokens,
        } => {
            #[cfg(feature = "cuda")]
            {
                drafter_chat_smoke(
                    model_dir,
                    drafter_dir,
                    prompt,
                    chat_template,
                    max_new_tokens,
                )?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (
                    model_dir,
                    drafter_dir,
                    prompt,
                    chat_template,
                    max_new_tokens,
                );
                anyhow::bail!(
                    "drafter-chat-smoke requires the cuda feature; rebuild with --features cuda"
                );
            }
        }
        Command::DrafterHandoffSmoke {
            model_dir,
            drafter_dir,
            prompt_tokens,
            q_len,
        } => {
            #[cfg(feature = "cuda")]
            {
                drafter_handoff_smoke(model_dir, drafter_dir, prompt_tokens, q_len)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (model_dir, drafter_dir, prompt_tokens, q_len);
                anyhow::bail!(
                    "drafter-handoff-smoke requires the cuda feature; rebuild with --features cuda"
                );
            }
        }
        Command::DrafterForwardSmoke {
            drafter_dir,
            q_len,
            ctx_len,
            iterations,
            fixture_dir,
        } => {
            #[cfg(feature = "cuda")]
            {
                drafter_forward_smoke(drafter_dir, q_len, ctx_len, iterations, fixture_dir)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (drafter_dir, q_len, ctx_len, iterations, fixture_dir);
                anyhow::bail!(
                    "drafter-forward-smoke requires the cuda feature; rebuild with --features cuda"
                );
            }
        }
        Command::Chat {
            model_dir,
            prompt,
            max_new_tokens,
            mtp_speculative_tokens,
            mtp_tree_leaves,
            drafter,
            drafter_dir,
        } => {
            if mtp_tree_leaves == 0 || mtp_tree_leaves > 8 {
                anyhow::bail!("--mtp-tree-leaves must be in 1..=8, got {mtp_tree_leaves}");
            }
            if drafter == DrafterArg::Dflash {
                let drafter_dir = drafter_dir.ok_or_else(|| {
                    anyhow::anyhow!(
                        "--drafter dflash requires --drafter-dir <path to DFlash drafter>"
                    )
                })?;
                #[cfg(feature = "cuda")]
                {
                    return run_chat_dflash(model_dir, drafter_dir, prompt, max_new_tokens);
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (model_dir, drafter_dir, prompt, max_new_tokens);
                    anyhow::bail!(
                        "chat --drafter dflash requires the cuda feature; rebuild with --features cuda"
                    );
                }
            }
            if drafter_dir.is_some() {
                anyhow::bail!("--drafter-dir is only valid with --drafter dflash");
            }
            run_chat(
                model_dir,
                prompt,
                max_new_tokens,
                mtp_speculative_tokens,
                mtp_tree_leaves,
            )?;
        }
        Command::Bench {
            model_dir,
            prompt_tokens,
            max_new_tokens,
            token_text,
            prompt_file,
            mtp_speculative_tokens,
            mtp_tree_leaves,
        } => {
            if mtp_tree_leaves == 0 || mtp_tree_leaves > 8 {
                anyhow::bail!("--mtp-tree-leaves must be in 1..=8, got {mtp_tree_leaves}");
            }
            run_bench(
                model_dir,
                prompt_tokens,
                max_new_tokens,
                token_text,
                prompt_file,
                mtp_speculative_tokens,
                mtp_tree_leaves,
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
fn interpreter_overhead_bench(
    instruction_counts: &[usize],
    cta_counts: &[u32],
    iterations: usize,
    warmup: usize,
    json: bool,
) -> Result<()> {
    use qwen36_fp4_kernels::{
        CudaBackend, CudaDeviceBuffer, InterpreterProgramSpec, KernelBackend, cuda_synchronize,
    };

    if iterations == 0 {
        anyhow::bail!("--iterations must be > 0");
    }

    let mut instruction_counts = instruction_counts.to_vec();
    if !instruction_counts.contains(&1) {
        instruction_counts.push(1);
    }
    instruction_counts.sort_unstable();
    instruction_counts.dedup();
    if instruction_counts.iter().any(|&count| count == 0) {
        anyhow::bail!("--instruction-counts must all be > 0");
    }

    let mut cta_counts = cta_counts.to_vec();
    cta_counts.sort_unstable();
    cta_counts.dedup();
    if cta_counts.is_empty() || cta_counts.iter().any(|&count| count == 0) {
        anyhow::bail!("--cta-counts must all be > 0");
    }

    let max_instruction_count = *instruction_counts
        .last()
        .ok_or_else(|| anyhow::anyhow!("missing instruction counts"))?;
    let max_counter_count = max_instruction_count.saturating_mul(2).max(1);
    let counters = CudaDeviceBuffer::zeroed(max_counter_count * std::mem::size_of::<i32>())?;
    let backend = CudaBackend;

    let mut rows = Vec::new();
    if !json {
        println!(
            "{:>5} {:>6} {:>10} {:>12} {:>14} {:>14}",
            "CTAs", "noops", "launch_us", "delta_us", "extra_us/op", "iters"
        );
    }
    for &cta_count in &cta_counts {
        let mut baseline_us = None;
        for &instruction_count in &instruction_counts {
            let program = chained_trampoline_program(instruction_count).finish();
            let instruction_bytes = interpreter_instruction_bytes(&program.instructions);
            let instructions = CudaDeviceBuffer::alloc(instruction_bytes.len())?;
            instructions.copy_from_host(instruction_bytes)?;
            let spec = InterpreterProgramSpec {
                instructions: instructions.ptr(),
                instruction_count: program.instructions.len(),
                counters_i32: counters.ptr(),
                counter_count: max_counter_count,
                cta_count,
                flags: 0,
            };

            for _ in 0..warmup {
                counters.memset_async(0)?;
                backend.interpreter_decode_sm120(&spec)?;
                cuda_synchronize()?;
            }

            let started = Instant::now();
            for _ in 0..iterations {
                counters.memset_async(0)?;
                backend.interpreter_decode_sm120(&spec)?;
                cuda_synchronize()?;
            }
            let elapsed = started.elapsed();
            let us_per_launch = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
            if instruction_count == 1 {
                baseline_us = Some(us_per_launch);
            }
            let baseline = baseline_us.unwrap_or(us_per_launch);
            let delta_vs_one_us = us_per_launch - baseline;
            let per_extra_instruction_us = if instruction_count > 1 {
                delta_vs_one_us / (instruction_count - 1) as f64
            } else {
                0.0
            };
            let row = serde_json::json!({
                "cta_count": cta_count,
                "noop_instructions": instruction_count,
                "program_instructions_including_exit": program.instructions.len(),
                "counter_count": max_counter_count,
                "iterations": iterations,
                "warmup": warmup,
                "us_per_launch": us_per_launch,
                "delta_vs_one_instruction_us": delta_vs_one_us,
                "per_extra_instruction_us": per_extra_instruction_us,
            });
            if !json {
                println!(
                    "{:>5} {:>6} {:>10.3} {:>12.3} {:>14.4} {:>14}",
                    row["cta_count"].as_u64().unwrap_or_default(),
                    row["noop_instructions"].as_u64().unwrap_or_default(),
                    row["us_per_launch"].as_f64().unwrap_or_default(),
                    row["delta_vs_one_instruction_us"]
                        .as_f64()
                        .unwrap_or_default(),
                    row["per_extra_instruction_us"].as_f64().unwrap_or_default(),
                    row["iterations"].as_u64().unwrap_or_default(),
                );
                io::stdout().flush()?;
            }
            rows.push(row);
        }
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn chained_trampoline_program(instruction_count: usize) -> qwen36_fp4_kernels::InterpreterProgram {
    use qwen36_fp4_kernels::InterpreterInstruction;

    let mut program = qwen36_fp4_kernels::InterpreterProgram::new();
    for idx in 0..instruction_count {
        let publish_counter = (idx * 2) as u32;
        let arrival_counter = publish_counter + 1;
        let mut instruction = InterpreterInstruction::fallback_trampoline()
            .with_publish(publish_counter, 1)
            .with_arrival_counter(arrival_counter);
        if idx > 0 {
            instruction = instruction.with_dep(((idx - 1) * 2) as u32, 1);
        }
        program.push(instruction);
    }
    program
}

#[cfg(feature = "cuda")]
fn interpreter_instruction_bytes(
    instructions: &[qwen36_fp4_kernels::InterpreterInstruction],
) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            instructions.as_ptr().cast::<u8>(),
            std::mem::size_of_val(instructions),
        )
    }
}

#[cfg(feature = "cuda")]
fn run_chat(
    model_dir: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    mtp_speculative_tokens: usize,
    mtp_tree_leaves: usize,
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
        max_context: prompt_tokens
            .len()
            .saturating_add(max_new_tokens)
            .saturating_add(tree_max_context_overhead(mtp_tree_leaves, max_new_tokens))
            .max(1),
        kv_cache_dtype: cuda_kv_cache_dtype(KvCacheDtype::Bf16),
        mtp_speculative_tokens: mtp_schedule.effective_tokens,
        ..EngineConfig::default()
    };
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, config)?;
    engine.prefill(&prompt_tokens)?;

    // Tree branch: takes precedence over chain MTP when K > 1 and MTP depth > 0.
    if mtp_tree_leaves > 1 && mtp_schedule.effective_tokens > 0 {
        return run_chat_mtp_tree(
            &mut engine,
            &tokenizer,
            &prompt_tokens,
            max_new_tokens,
            mtp_schedule.effective_tokens,
            mtp_tree_leaves,
        );
    }

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
    let mut proposed_draft_tokens = 0_usize;
    let mut verify_cycles = 0_usize;
    let auto_fallback = mtp_auto_fallback_enabled();
    let fallback_window = mtp_fallback_window();
    let fallback_min_acceptance = mtp_fallback_min_acceptance();
    let mut fallback_active = false;
    let mut took_fallback = false;

    while generated < max_new_tokens {
        let text = tokenizer.decode(&[current_token], true)?;
        print!("{text}");
        io::stdout().flush()?;
        generated += 1;
        if current_token == 248044 || generated >= max_new_tokens {
            break;
        }
        if fallback_active {
            took_fallback = true;
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
        proposed_draft_tokens += verify_count;
        verify_cycles += 1;
        if verify.rejected {
            rejected_draft_tokens += 1;
        }
        if auto_fallback
            && !fallback_active
            && verify_cycles >= fallback_window
            && proposed_draft_tokens > 0
            && (accepted_draft_tokens as f64 / proposed_draft_tokens as f64)
                < fallback_min_acceptance
        {
            fallback_active = true;
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

    if took_fallback && generated < max_new_tokens {
        // Acceptance cannot pay for the verify cycle on this content —
        // finish on the plain decode graph (the MTP=0 chat loop shape).
        engine.seed_sampled_token(current_token)?;
        let mut graph_on = false;
        while generated < max_new_tokens {
            if graph_on {
                engine.decode_graph_step()?;
            } else {
                engine.enable_decode_graph()?;
                graph_on = true;
            }
            qwen36_fp4_runtime::cuda_synchronize()?;
            let token = engine.read_sampled_token()?;
            let text = tokenizer.decode(&[token], true)?;
            print!("{text}");
            io::stdout().flush()?;
            generated += 1;
            if token == 248044 {
                break;
            }
        }
        if graph_on {
            engine.disable_decode_graph()?;
        }
    }

    if std::env::var("QWEN36_MTP_STATS").is_ok() {
        let drafts = accepted_draft_tokens + rejected_draft_tokens;
        eprintln!(
            "mtp.stats accepted={} rejected={} acceptance_rate={:.4} draft_acceptance={:.4} fallback={}",
            accepted_draft_tokens,
            rejected_draft_tokens,
            if drafts > 0 {
                accepted_draft_tokens as f64 / drafts as f64
            } else {
                0.0
            },
            if proposed_draft_tokens > 0 {
                accepted_draft_tokens as f64 / proposed_draft_tokens as f64
            } else {
                0.0
            },
            if took_fallback {
                format!("cycle{verify_cycles}")
            } else {
                "none".to_owned()
            }
        );
    }
    println!();
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_chat_mtp_tree(
    engine: &mut Engine<qwen36_fp4_runtime::CudaBackend>,
    tokenizer: &QwenTokenizer,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    chain_depth: usize,
    leaf_count: usize,
) -> Result<()> {
    engine.queue_sample_greedy_to_current_token()?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    let mut current_token = engine.read_current_token()?;
    let (mut chain_tokens, mut leaf_tokens) = engine.prepare_mtp_drafts_with_leaves(
        prompt_tokens,
        current_token,
        chain_depth,
        leaf_count,
    )?;

    let mut generated = 0_usize;
    let mut accepted_chain_total = 0_usize;
    let mut accepted_leaf_cycles = 0_usize;
    let mut full_chain_cycles = 0_usize;
    let mut total_cycles = 0_usize;

    while generated < max_new_tokens {
        let text = tokenizer.decode(&[current_token], true)?;
        print!("{text}");
        io::stdout().flush()?;
        generated += 1;
        if current_token == 248044 || generated >= max_new_tokens {
            break;
        }
        if chain_tokens.is_empty() {
            anyhow::bail!("tree MTP: chain_tokens empty before generation completed");
        }

        let result = engine.verify_mtp_tree_draft(
            current_token,
            &chain_tokens,
            &leaf_tokens,
            chain_depth,
        )?;
        total_cycles += 1;
        accepted_chain_total += result.accepted_chain;
        if result.accepted_chain == chain_tokens.len() {
            full_chain_cycles += 1;
        }
        if result.accepted_leaf.is_some() {
            accepted_leaf_cycles += 1;
        }

        // Print committed tokens (all but the last, which becomes next current_token).
        let mut stopped = false;
        let n_committed = result.committed.len();
        for token in result
            .committed
            .iter()
            .take(n_committed.saturating_sub(1))
            .copied()
        {
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

        current_token = result.next_token;
        // Use pre-computed drafts from the result (mid-loop re-prefill via
        // prepare_mtp_drafts_with_leaves would corrupt MTP head KV state).
        chain_tokens = result.next_chain_drafts;
        leaf_tokens = result.next_leaf_drafts;
        if chain_tokens.is_empty() || leaf_tokens.is_empty() {
            // Pre-computation skipped (e.g., near max_context). Stop and let
            // caller decide; remaining tokens will be emitted on next launch.
            break;
        }
    }

    if std::env::var("QWEN36_MTP_STATS").is_ok() {
        eprintln!(
            "mtp_tree.stats cycles={total_cycles} accepted_chain_avg={:.2} full_chain_rate={:.4} leaf_accept_rate={:.4}",
            if total_cycles > 0 {
                accepted_chain_total as f64 / total_cycles as f64
            } else {
                0.0
            },
            if total_cycles > 0 {
                full_chain_cycles as f64 / total_cycles as f64
            } else {
                0.0
            },
            if full_chain_cycles > 0 {
                accepted_leaf_cycles as f64 / full_chain_cycles as f64
            } else {
                0.0
            },
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
        kv_cache_dtype: cuda_kv_cache_dtype(KvCacheDtype::Bf16),
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

    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
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
        kv_cache_dtype: cuda_kv_cache_dtype(KvCacheDtype::Bf16),
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

    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
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
    prompt_file: Option<PathBuf>,
    mtp_speculative_tokens: usize,
    mtp_tree_leaves: usize,
) -> Result<()> {
    if mtp_speculative_tokens > 4 {
        anyhow::bail!("bench currently supports --mtp-speculative-tokens 0..=4");
    }
    let total_start = Instant::now();
    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let tokenizer = QwenTokenizer::from_model_dir(&model_dir)?;
    let prompt_tokens = match prompt_file.as_deref() {
        Some(path) => prompt_tokens_from_file(&tokenizer, path, prompt_token_count)?,
        None => synthetic_prompt_tokens(&tokenizer, &token_text, prompt_token_count)?,
    };
    let mtp_schedule = mtp_schedule(mtp_speculative_tokens, prompt_tokens.len());
    let config = EngineConfig {
        max_context: prompt_tokens
            .len()
            .saturating_add(max_new_tokens)
            .saturating_add(tree_max_context_overhead(mtp_tree_leaves, max_new_tokens))
            .max(1),
        kv_cache_dtype: cuda_kv_cache_dtype(KvCacheDtype::Bf16),
        mtp_speculative_tokens: mtp_schedule.effective_tokens,
        ..EngineConfig::default()
    };

    let load_start = Instant::now();
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, config)?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let prefill_start = Instant::now();
    engine.prefill(&prompt_tokens)?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    // Tree branch: takes precedence over chain MTP when K > 1 and MTP depth > 0.
    if mtp_tree_leaves > 1 && mtp_schedule.effective_tokens > 0 {
        return run_bench_mtp_tree(
            engine,
            prompt_tokens,
            max_new_tokens,
            total_start,
            load_seconds,
            prefill_seconds,
            mtp_schedule,
            mtp_tree_leaves,
        );
    }

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
    let mut proposed_draft_tokens = 0_usize;
    let mut proposed_per_position = vec![0_usize; draft_window];
    let mut accepted_per_position = vec![0_usize; draft_window];
    let mut full_accept_cycles = 0_usize;
    let auto_fallback = mtp_auto_fallback_enabled();
    let fallback_window = mtp_fallback_window();
    let fallback_min_acceptance = mtp_fallback_min_acceptance();
    let mut fallback_cycle: Option<usize> = None;
    let mut main_decode_steps = 0_usize;
    let mut mtp_decode_steps = draft_tokens.len();
    let mut rollback_recoveries = 0_usize;
    let rebuilds = 0_usize;
    let mut mtp_verify_seconds = 0.0_f64;

    if env_flag_enabled("QWEN36_MTP_DEVICE_CHAIN")
        && env_flag_enabled("QWEN36_MTP_ASSUME_ACCEPT")
        && draft_tokens.len() >= 2
    {
        let chain_start = Instant::now();
        let chain = engine.run_mtp_assume_accept_device_chain(
            current_token,
            &draft_tokens,
            max_new_tokens,
            draft_window,
        )?;
        mtp_verify_seconds += chain_start.elapsed().as_secs_f64();
        if chain.cycles > 0 {
            generated += chain.generated_tokens;
            main_decode_steps += chain.cycles;
            accepted_draft_tokens += chain.accepted_draft_tokens;
            // Assume-accept chains propose the full window every cycle; no
            // per-cycle detail comes back, so per-position stays untracked here.
            proposed_draft_tokens += chain.cycles * draft_window;
            mtp_decode_steps += chain.cycles * draft_window;
            current_token = chain.next_token;
            draft_tokens = chain.next_draft_tokens;
        }
    }

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
        proposed_draft_tokens += verify_count;
        for slot in proposed_per_position.iter_mut().take(verify_count) {
            *slot += 1;
        }
        for slot in accepted_per_position
            .iter_mut()
            .take(verify.accepted_drafts)
        {
            *slot += 1;
        }
        if verify.accepted_drafts == verify_count {
            full_accept_cycles += 1;
        }
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

        if auto_fallback
            && fallback_cycle.is_none()
            && main_decode_steps >= fallback_window
            && proposed_draft_tokens > 0
            && (accepted_draft_tokens as f64 / proposed_draft_tokens as f64)
                < fallback_min_acceptance
        {
            fallback_cycle = Some(main_decode_steps);
            break;
        }
    }

    // Auto-fallback: observed acceptance cannot pay for the verify cycle —
    // finish the run on the plain decode graph from the committed state.
    if fallback_cycle.is_some() && generated < max_new_tokens {
        engine.seed_sampled_token(current_token)?;
        let mut graph_on = false;
        while generated < max_new_tokens {
            generated += 1;
            if generated >= max_new_tokens {
                break;
            }
            if graph_on {
                engine.decode_graph_step()?;
            } else {
                engine.enable_decode_graph()?;
                graph_on = true;
            }
        }
        if graph_on {
            engine.disable_decode_graph()?;
        }
    }

    qwen36_fp4_runtime::cuda_synchronize()?;
    let cuda_counters = qwen36_fp4_runtime::cuda_counters_read()?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let total_seconds = total_start.elapsed().as_secs_f64();
    let evaluated_drafts = accepted_draft_tokens + rejected_draft_tokens;
    let acceptance_rate_per_position: Vec<f64> = proposed_per_position
        .iter()
        .zip(accepted_per_position.iter())
        .map(|(&proposed, &accepted)| {
            if proposed > 0 {
                accepted as f64 / proposed as f64
            } else {
                0.0
            }
        })
        .collect();
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
            "mtp_proposed_draft_tokens": proposed_draft_tokens,
            "mtp_draft_acceptance_rate": if proposed_draft_tokens > 0 {
                accepted_draft_tokens as f64 / proposed_draft_tokens as f64
            } else {
                0.0
            },
            "mtp_full_accept_cycles": full_accept_cycles,
            "mtp_fallback_cycle": fallback_cycle,
            "mtp_fallback_min_acceptance": fallback_min_acceptance,
            "mtp_proposed_per_position": proposed_per_position,
            "mtp_accepted_per_position": accepted_per_position,
            "mtp_acceptance_rate_per_position": acceptance_rate_per_position,
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
#[allow(clippy::too_many_arguments)]
fn run_bench_mtp_tree(
    mut engine: Engine<qwen36_fp4_runtime::CudaBackend>,
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    total_start: Instant,
    load_seconds: f64,
    prefill_seconds: f64,
    mtp_schedule: MtpSchedule,
    leaf_count: usize,
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
                "mtp_tree_leaf_count": leaf_count,
                "mtp_accepted_chain_total": 0,
                "mtp_full_chain_cycles": 0,
                "mtp_leaf_accept_rate": 0.0,
                "cuda_counters_decode": cuda_counters,
            }))?
        );
        return Ok(());
    }

    let chain_depth = mtp_schedule.effective_tokens;
    let setup_start = Instant::now();
    engine.queue_sample_greedy_to_current_token()?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    let mut current_token = engine.read_current_token()?;
    let (mut chain_tokens, mut leaf_tokens) = engine.prepare_mtp_drafts_with_leaves(
        &prompt_tokens,
        current_token,
        chain_depth,
        leaf_count,
    )?;
    let mtp_setup_seconds = setup_start.elapsed().as_secs_f64();

    let mut generated = 0_usize;
    let mut accepted_chain_total = 0_usize;
    let mut accepted_leaf_cycles = 0_usize;
    let mut full_chain_cycles = 0_usize;
    let mut total_cycles = 0_usize;
    let mut mtp_verify_seconds = 0.0_f64;

    while generated < max_new_tokens {
        generated += 1;
        if generated >= max_new_tokens {
            break;
        }
        if chain_tokens.is_empty() {
            anyhow::bail!("tree MTP bench: chain_tokens empty before generation completed");
        }

        let verify_start = Instant::now();
        let result = engine.verify_mtp_tree_draft(
            current_token,
            &chain_tokens,
            &leaf_tokens,
            chain_depth,
        )?;
        mtp_verify_seconds += verify_start.elapsed().as_secs_f64();

        total_cycles += 1;
        accepted_chain_total += result.accepted_chain;
        if result.accepted_chain == chain_tokens.len() {
            full_chain_cycles += 1;
        }
        if result.accepted_leaf.is_some() {
            accepted_leaf_cycles += 1;
        }

        // Count committed tokens (excluding the last which is the seed for next cycle).
        let committed_extra = result.committed.len().saturating_sub(1);
        generated += committed_extra;
        if generated >= max_new_tokens {
            break;
        }

        current_token = result.next_token;
        // Use pre-computed drafts from the result (mid-loop re-prefill via
        // prepare_mtp_drafts_with_leaves would corrupt MTP head KV state).
        chain_tokens = result.next_chain_drafts;
        leaf_tokens = result.next_leaf_drafts;
        if chain_tokens.is_empty() || leaf_tokens.is_empty() {
            break;
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
            "mtp_tree_leaf_count": leaf_count,
            "mtp_tree_cycles": total_cycles,
            "mtp_accepted_chain_total": accepted_chain_total,
            "mtp_full_chain_cycles": full_chain_cycles,
            "mtp_leaf_accept_rate": if full_chain_cycles > 0 {
                accepted_leaf_cycles as f64 / full_chain_cycles as f64
            } else {
                0.0
            },
            "mtp_setup_seconds": mtp_setup_seconds,
            "mtp_verify_seconds": mtp_verify_seconds,
            "cuda_counters_decode": cuda_counters,
        }))?
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn prompt_tokens_from_file(
    tokenizer: &QwenTokenizer,
    path: &std::path::Path,
    target_tokens: usize,
) -> Result<Vec<u32>> {
    if target_tokens == 0 {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path)
        .map_err(|err| anyhow::anyhow!("failed to read --prompt-file {}: {err}", path.display()))?;
    let tokens = tokenizer.encode(&text, false)?;
    if tokens.len() < target_tokens {
        anyhow::bail!(
            "prompt-file {} has only {} tokens, requested {}",
            path.display(),
            tokens.len(),
            target_tokens,
        );
    }
    eprintln!(
        "bench: using {} tokens from {}",
        target_tokens,
        path.display(),
    );
    Ok(tokens.into_iter().take(target_tokens).collect())
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
    mtp_tree_leaves: usize,
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
    let _ = (max_new_tokens, mtp_tree_leaves);
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
    _prompt_file: Option<PathBuf>,
    _mtp_speculative_tokens: usize,
    _mtp_tree_leaves: usize,
) -> Result<()> {
    bail!("bench requires rebuilding qwen36 with --features cuda and the CUDA shared library")
}

#[cfg(feature = "cuda")]
fn gpu_load(model_dir: PathBuf, max_context: usize, mtp_speculative_tokens: usize) -> Result<()> {
    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let config = EngineConfig {
        max_context,
        kv_cache_dtype: cuda_kv_cache_dtype(EngineConfig::default().kv_cache_dtype),
        mtp_speculative_tokens,
        ..EngineConfig::default()
    };
    let engine = Engine::cuda_with_mapped_weights(&mapped_model, config)?;
    let (gpu_tensors, gpu_weight_bytes) = engine
        .gpu_weight_summary()
        .ok_or_else(|| anyhow::anyhow!("CUDA engine did not expose uploaded weights"))?;
    let gpu_buffer_bytes = engine
        .gpu_buffer_bytes()
        .ok_or_else(|| anyhow::anyhow!("CUDA engine did not expose runtime buffers"))?;
    let gpu_memory_report = engine
        .gpu_memory_report()
        .ok_or_else(|| anyhow::anyhow!("CUDA engine did not expose detailed memory report"))?;

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
            "gpu_memory_report": gpu_memory_report,
        }))?
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn gpu_load(
    _model_dir: PathBuf,
    _max_context: usize,
    _mtp_speculative_tokens: usize,
) -> Result<()> {
    bail!("gpu-load requires rebuilding qwen36 with --features cuda and the CUDA shared library")
}

#[cfg(feature = "cuda")]
fn drafter_load(drafter_dir: PathBuf) -> Result<()> {
    use qwen36_fp4_drafter::{DFlashDrafter, DFlashDrafterDevice};

    let drafter = DFlashDrafter::open(&drafter_dir)?;
    let manifest = drafter.manifest.clone();
    let device = DFlashDrafterDevice::upload(&drafter)?;
    let report = device.report(&manifest);
    qwen36_fp4_runtime::cuda_synchronize()?;

    let sliding_layers = drafter
        .config
        .layer_types
        .iter()
        .filter(|kind| kind.as_str() == "sliding_attention")
        .count();
    let full_layers = drafter.config.layer_types.len() - sliding_layers;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "drafter_dir": drafter.drafter_dir.display().to_string(),
            "num_hidden_layers": drafter.config.num_hidden_layers,
            "sliding_attention_layers": sliding_layers,
            "full_attention_layers": full_layers,
            "tensors_uploaded": report.tensor_count,
            "layer_bytes": report.layer_bytes,
            "fc_bytes": report.fc_bytes,
            "hidden_norm_bytes": report.hidden_norm_bytes,
            "norm_bytes": report.norm_bytes,
            "total_bytes": report.total_bytes,
            "total_gib": bytes_to_gib(report.total_bytes as u64),
            "block_size": drafter.config.block_size,
            "target_layer_ids": drafter.config.dflash_config.target_layer_ids,
        }))?,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn drafter_step_smoke(
    model_dir: PathBuf,
    drafter_dir: PathBuf,
    prompt: String,
    chat_template: bool,
) -> Result<()> {
    use std::sync::Arc;

    use qwen36_fp4_drafter::{
        DFlashDrafter, DFlashDrafterDevice, DFlashProposeWorkspace, DrafterForward,
        DrafterForwardWorkspace, TargetHiddenCapture, propose_block,
    };
    use qwen36_fp4_kernels::CudaBackend;

    // --- Tokenize the prompt ----------------------------------------
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
    if prompt_tokens.is_empty() {
        anyhow::bail!("prompt produced 0 tokens");
    }
    let ctx_len = prompt_tokens.len();

    // --- Open drafter (host only) -----------------------------------
    let drafter = DFlashDrafter::open(&drafter_dir)?;
    if drafter.config.head_dim != 128 {
        anyhow::bail!(
            "drafter-step-smoke v1 only supports head_dim=128, got {}",
            drafter.config.head_dim,
        );
    }
    let mask_token_id = drafter.config.dflash_config.mask_token_id;
    let block_size = drafter.config.block_size;
    let q_len = block_size;
    let vocab_size = drafter.config.vocab_size;

    // --- Load target first (heavy 17 GB) ----------------------------
    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let target_config = EngineConfig {
        max_context: ctx_len.saturating_add(block_size).max(256),
        kv_cache_dtype: cuda_kv_cache_dtype(KvCacheDtype::Fp8),
        ..EngineConfig::default()
    };
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, target_config)?;

    // --- Drafter weights on GPU + capture buffer --------------------
    let drafter_device = DFlashDrafterDevice::upload(&drafter)?;
    let capture = Arc::new(TargetHiddenCapture::alloc(&drafter.config, ctx_len)?);
    let capture_for_hook = capture.clone();
    let hook: qwen36_fp4_runtime::DrafterHiddenCaptureHook =
        Arc::new(move |layer_idx, residual_ptr, tokens| {
            capture_for_hook
                .capture_layer(&CudaBackend, layer_idx, residual_ptr, tokens)
                .map_err(|e| qwen36_fp4_core::CoreError::Runtime(format!("drafter handoff: {e}")))
        });
    engine.set_drafter_hidden_capture(Some(hook));

    // --- Target prefill --------------------------------------------
    engine.prefill(&prompt_tokens)?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    engine.set_drafter_hidden_capture(None);

    // --- Sample the seed token from the prefill ---------------------
    engine.queue_sample_greedy_to_current_token()?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    let seed_token = engine.read_current_token()?;

    // --- Resolve target weight pointers needed by propose ----------
    let manifest = engine
        .weights
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("engine has no weights manifest after prefill"))?;
    let gpu_weights = engine
        .gpu_weights
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("engine has no GPU weights after prefill"))?;
    let target_embed_ptr = gpu_weights
        .tensor(&manifest.embed_tokens.name)
        .ok_or_else(|| anyhow::anyhow!("embed_tokens tensor missing from GPU store"))?
        .ptr();
    let target_lm_head_ptr = gpu_weights
        .tensor(&manifest.lm_head.name)
        .ok_or_else(|| anyhow::anyhow!("lm_head tensor missing from GPU store"))?
        .ptr();

    // --- Drafter workspace + KV cache --------------------------------
    let kv_cache_max_len = ctx_len + q_len;
    let workspace =
        DrafterForwardWorkspace::alloc(&drafter.config, q_len, ctx_len, kv_cache_max_len)?;
    // Pre-fill absolute positions [0, kv_cache_max_len) in the
    // drafter's position_ids buffer.
    let mut pos_bytes = Vec::with_capacity(kv_cache_max_len * 4);
    for p in 0..kv_cache_max_len {
        pos_bytes.extend_from_slice(&(p as i32).to_le_bytes());
    }
    workspace.position_ids_buffer().copy_from_host(&pos_bytes)?;

    let backend = CudaBackend;
    let mut forward = DrafterForward::new(&drafter_device, &drafter.config, workspace)?;
    forward.reset_kv_cache();

    // --- Propose block ---------------------------------------------
    let propose_ws = DFlashProposeWorkspace::alloc(&drafter.config, q_len)?;
    let mut noise_token_ids = Vec::with_capacity(q_len);
    noise_token_ids.push(seed_token);
    for _ in 1..q_len {
        noise_token_ids.push(mask_token_id);
    }
    let proposed_tokens = propose_block(
        &backend,
        &mut forward,
        &propose_ws,
        &noise_token_ids,
        capture.output_ptr(),
        ctx_len,
        target_embed_ptr,
        target_lm_head_ptr,
        vocab_size,
    )?;
    qwen36_fp4_runtime::cuda_synchronize()?;

    // Per the dflash_generate reference, the drafter denoises positions
    // 1..block_size and the seed at position 0 is held as the "bonus"
    // from the prior target sample. Report both.
    let seed_text = tokenizer.decode(&[seed_token], true).unwrap_or_default();
    let proposed_after_seed: Vec<u32> = proposed_tokens[1..].to_vec();
    let proposed_text = tokenizer
        .decode(&proposed_after_seed, true)
        .unwrap_or_default();
    let full_block_text = tokenizer.decode(&proposed_tokens, true).unwrap_or_default();

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model_dir": model_dir.display().to_string(),
            "drafter_dir": drafter_dir.display().to_string(),
            "prompt": prompt,
            "chat_template": chat_template,
            "prompt_tokens": ctx_len,
            "block_size": block_size,
            "mask_token_id": mask_token_id,
            "seed_token": seed_token,
            "seed_decoded": seed_text,
            "proposed_tokens": proposed_tokens,
            "proposed_tokens_after_seed": proposed_after_seed,
            "proposed_decoded_after_seed": proposed_text,
            "full_block_decoded": full_block_text,
        }))?,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn decode_vs_prefill_check(model_dir: PathBuf, prompt: String) -> Result<()> {
    let tokenizer = QwenTokenizer::from_model_dir(&model_dir)?;
    let prompt_tokens = tokenizer.encode(&prompt, true)?;
    if prompt_tokens.is_empty() {
        anyhow::bail!("prompt produced 0 tokens");
    }

    let make_engine = || -> Result<Engine<qwen36_fp4_kernels::CudaBackend>> {
        let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
        let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
        let config = EngineConfig {
            max_context: prompt_tokens.len().saturating_add(16).max(256),
            kv_cache_dtype: cuda_kv_cache_dtype(KvCacheDtype::Fp8),
            ..EngineConfig::default()
        };
        Ok(Engine::cuda_with_mapped_weights(&mapped_model, config)?)
    };

    let read_logits =
        |engine: &Engine<qwen36_fp4_kernels::CudaBackend>, vocab: usize| -> Result<Vec<u8>> {
            let forward = engine
                .gpu_forward
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("engine has no gpu_forward buffers"))?;
            let mut buf = vec![0u8; vocab * 2];
            forward.logits.copy_to_host(&mut buf)?;
            Ok(buf)
        };

    let vocab_size = {
        let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
        layout.topology.vocab_size
    };

    // --- Session A: prefill + decode_one(seed) ---------------------
    eprintln!("[A] loading engine for decode path...");
    let (seed, argmax_decode, logits_decode) = {
        let mut engine = make_engine()?;
        engine.prefill(&prompt_tokens)?;
        qwen36_fp4_runtime::cuda_synchronize()?;
        engine.queue_sample_greedy_to_current_token()?;
        qwen36_fp4_runtime::cuda_synchronize()?;
        let seed = engine.read_current_token()?;

        engine.decode_one_queued(seed)?;
        engine.queue_sample_greedy()?;
        qwen36_fp4_runtime::cuda_synchronize()?;
        let argmax = engine.read_sampled_token()?;
        let logits = read_logits(&engine, vocab_size)?;
        (seed, argmax, logits)
    };
    eprintln!("[A] seed={seed} argmax_decode={argmax_decode}");

    // --- Session B: prefill(prompt + [seed]) -----------------------
    eprintln!("[B] loading engine for prefill path...");
    let (argmax_prefill, logits_prefill) = {
        let mut engine = make_engine()?;
        let mut tokens = prompt_tokens.clone();
        tokens.push(seed);
        engine.prefill(&tokens)?;
        qwen36_fp4_runtime::cuda_synchronize()?;
        engine.queue_sample_greedy_to_current_token()?;
        qwen36_fp4_runtime::cuda_synchronize()?;
        let argmax = engine.read_current_token()?;
        let logits = read_logits(&engine, vocab_size)?;
        (argmax, logits)
    };
    eprintln!("[B] argmax_prefill={argmax_prefill}");

    let cos_sim = bf16_cosine_similarity(&logits_decode, &logits_prefill);
    let decode_text = tokenizer.decode(&[argmax_decode], true).unwrap_or_default();
    let prefill_text = tokenizer
        .decode(&[argmax_prefill], true)
        .unwrap_or_default();
    let seed_text = tokenizer.decode(&[seed], true).unwrap_or_default();

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model_dir": model_dir.display().to_string(),
            "prompt": prompt,
            "prompt_tokens": prompt_tokens.len(),
            "vocab_size": vocab_size,
            "seed_token": seed,
            "seed_decoded": seed_text,
            "argmax_via_decode": argmax_decode,
            "argmax_via_decode_decoded": decode_text,
            "argmax_via_prefill": argmax_prefill,
            "argmax_via_prefill_decoded": prefill_text,
            "argmax_match": argmax_decode == argmax_prefill,
            "logits_cos_sim": cos_sim,
        }))?,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_chat_dflash(
    model_dir: PathBuf,
    drafter_dir: PathBuf,
    prompt: String,
    max_new_tokens: usize,
) -> Result<()> {
    use std::sync::Arc;
    use std::time::Instant;

    use qwen36_fp4_drafter::{
        DFlashDrafter, DFlashDrafterDevice, DFlashProposeWorkspace, DrafterForward,
        DrafterForwardWorkspace, TargetHiddenCapture, propose_block,
    };
    use qwen36_fp4_kernels::CudaBackend;

    let chat_start = Instant::now();

    let tokenizer = QwenTokenizer::from_model_dir(&model_dir)?;
    let messages = vec![ChatMessage {
        role: "user".to_owned(),
        content: prompt.clone(),
    }];
    let prompt_tokens = tokenizer.encode_chat(&messages, true)?;
    if prompt_tokens.is_empty() {
        anyhow::bail!("prompt produced 0 tokens");
    }
    let prompt_len = prompt_tokens.len();

    let drafter = DFlashDrafter::open(&drafter_dir)?;
    if drafter.config.head_dim != 128 {
        anyhow::bail!(
            "chat --drafter dflash only supports head_dim=128, got {}",
            drafter.config.head_dim,
        );
    }
    let mask_token_id = drafter.config.dflash_config.mask_token_id;
    let block_size = drafter.config.block_size;
    let q_len = block_size;
    let vocab_size = drafter.config.vocab_size;
    let eos_token_id: u32 = 248044;

    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let target_config = EngineConfig {
        max_context: prompt_len
            .saturating_add(max_new_tokens)
            .saturating_add(block_size)
            .max(256),
        kv_cache_dtype: cuda_kv_cache_dtype(KvCacheDtype::Fp8),
        ..EngineConfig::default()
    };
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, target_config)?;

    let drafter_device = DFlashDrafterDevice::upload(&drafter)?;
    let capture_max_tokens = prompt_len.max(block_size + 1);
    let capture = Arc::new(TargetHiddenCapture::alloc(
        &drafter.config,
        capture_max_tokens,
    )?);
    let capture_for_hook = capture.clone();
    let hook: qwen36_fp4_runtime::DrafterHiddenCaptureHook =
        Arc::new(move |layer_idx, residual_ptr, tokens| {
            capture_for_hook
                .capture_layer(&CudaBackend, layer_idx, residual_ptr, tokens)
                .map_err(|e| qwen36_fp4_core::CoreError::Runtime(format!("drafter handoff: {e}")))
        });

    capture.set_write_row(0);
    engine.set_drafter_hidden_capture(Some(hook.clone()));
    engine.prefill(&prompt_tokens)?;
    qwen36_fp4_runtime::cuda_synchronize()?;

    engine.queue_sample_greedy_to_current_token()?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    let mut seed_token = engine.read_current_token()?;

    let (target_embed_ptr, target_lm_head_ptr) = {
        let manifest = engine
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("engine has no weights manifest after prefill"))?;
        let gpu_weights = engine
            .gpu_weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("engine has no GPU weights after prefill"))?;
        let embed = gpu_weights
            .tensor(&manifest.embed_tokens.name)
            .ok_or_else(|| anyhow::anyhow!("embed_tokens tensor missing"))?
            .ptr();
        let lm = gpu_weights
            .tensor(&manifest.lm_head.name)
            .ok_or_else(|| anyhow::anyhow!("lm_head tensor missing"))?
            .ptr();
        (embed, lm)
    };

    let ctx_len_max = prompt_len.max(block_size);
    let kv_cache_max_len = prompt_len + max_new_tokens + ctx_len_max + block_size + 16;
    let workspace =
        DrafterForwardWorkspace::alloc(&drafter.config, q_len, ctx_len_max, kv_cache_max_len)?;
    let mut pos_bytes = Vec::with_capacity(kv_cache_max_len * 4);
    for p in 0..kv_cache_max_len {
        pos_bytes.extend_from_slice(&(p as i32).to_le_bytes());
    }
    workspace.position_ids_buffer().copy_from_host(&pos_bytes)?;
    let backend = CudaBackend;
    let mut forward = DrafterForward::new(&drafter_device, &drafter.config, workspace)?;
    let propose_ws = DFlashProposeWorkspace::alloc(&drafter.config, q_len)?;

    // Same off-by-one bookkeeping as `drafter-chat-smoke`: include the
    // seed in iter 0's `iter_committed`. This gives the drafter one
    // extra cached attention slot per iter and empirically raises AL by
    // ~20% on every prompt vs the strict dflash crop semantics. See
    // docs/superpowers/notes/2026-06-09-dflash-benchmarks.md.
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut stdout = io::stdout();

    let mut ctx_len = prompt_len;
    let mut total_committed_after_prompt = 0_usize;
    let mut iter_accepts: Vec<usize> = Vec::new();

    let decode_start = Instant::now();
    while total_committed_after_prompt < max_new_tokens {
        let prefix_len_for_drafter = prompt_len + total_committed_after_prompt;
        if total_committed_after_prompt == 0 {
            forward.reset_kv_cache();
        } else {
            // Clamp: when the off-by-one bookkeeping would crop past
            // the cache's actual extent (happens on near-full-accept
            // iters after a long chat-template prompt), keep the cache
            // as-is.
            let safe = prefix_len_for_drafter.min(forward.current_kv_len());
            forward.crop_kv_cache(safe)?;
        }

        let mut noise_token_ids = Vec::with_capacity(q_len);
        noise_token_ids.push(seed_token);
        for _ in 1..q_len {
            noise_token_ids.push(mask_token_id);
        }
        let proposed_tokens = propose_block(
            &backend,
            &mut forward,
            &propose_ws,
            &noise_token_ids,
            capture.output_ptr(),
            ctx_len,
            target_embed_ptr,
            target_lm_head_ptr,
            vocab_size,
        )?;
        qwen36_fp4_runtime::cuda_synchronize()?;
        let drafts: Vec<u32> = proposed_tokens[1..].to_vec();

        capture.set_write_row(0);
        let mut verify_input = Vec::with_capacity(drafts.len() + 1);
        verify_input.push(seed_token);
        verify_input.extend_from_slice(&drafts);
        let argmaxes = engine.verify_block_batched(&verify_input)?;

        let mut accepted = 0_usize;
        let mut bonus_token: u32 = 0;
        for (i, &drafted) in drafts.iter().enumerate() {
            if argmaxes[i] == drafted {
                accepted += 1;
            } else {
                bonus_token = argmaxes[i];
                break;
            }
        }
        if accepted == drafts.len() {
            bonus_token = argmaxes[drafts.len()];
        }
        iter_accepts.push(accepted);

        let committed_target_position = prompt_len + total_committed_after_prompt + accepted + 1;
        let clamped_target = committed_target_position.min(engine.state.position);
        engine.crop_state_position(clamped_target)?;

        let mut iter_committed: Vec<u32> = Vec::with_capacity(accepted + 2);
        if total_committed_after_prompt == 0 {
            iter_committed.push(seed_token);
        }
        iter_committed.extend(drafts.iter().copied().take(accepted));
        iter_committed.push(bonus_token);

        // Stream iter_committed to stdout.
        let text = tokenizer.decode(&iter_committed, true).unwrap_or_default();
        write!(stdout, "{text}")?;
        stdout.flush().ok();

        let mut hit_eos = false;
        for &t in &iter_committed {
            generated.push(t);
            if t == eos_token_id {
                hit_eos = true;
                break;
            }
        }
        if hit_eos {
            break;
        }

        seed_token = bonus_token;
        ctx_len = accepted + 1;
        total_committed_after_prompt = generated.len();
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let total_seconds = chat_start.elapsed().as_secs_f64();
    engine.set_drafter_hidden_capture(None);

    let tokens_per_second = if decode_seconds > 0.0 {
        generated.len() as f64 / decode_seconds
    } else {
        0.0
    };
    let iters = iter_accepts.len();
    let avg_accept = if iters > 0 {
        iter_accepts.iter().sum::<usize>() as f64 / iters as f64
    } else {
        0.0
    };
    let acceptance_length = if iters > 0 {
        generated.len() as f64 / iters as f64
    } else {
        0.0
    };

    eprintln!(
        "\n[dflash] generated {} tokens in {} iters | AL={:.2} avg_accept={:.2} | decode {:.2}s ({:.1} tok/s) | total {:.2}s",
        generated.len(),
        iters,
        acceptance_length,
        avg_accept,
        decode_seconds,
        tokens_per_second,
        total_seconds,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn drafter_chat_smoke(
    model_dir: PathBuf,
    drafter_dir: PathBuf,
    prompt: String,
    chat_template: bool,
    max_new_tokens: usize,
) -> Result<()> {
    use std::sync::Arc;
    use std::time::Instant;

    use qwen36_fp4_drafter::{
        DFlashDrafter, DFlashDrafterDevice, DFlashProposeWorkspace, DrafterForward,
        DrafterForwardWorkspace, TargetHiddenCapture, propose_block,
    };
    use qwen36_fp4_kernels::CudaBackend;

    let bench_start = Instant::now();

    if max_new_tokens == 0 {
        anyhow::bail!("max_new_tokens must be > 0");
    }

    // --- Tokenize ----------------------------------------------------
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
    if prompt_tokens.is_empty() {
        anyhow::bail!("prompt produced 0 tokens");
    }
    let prompt_len = prompt_tokens.len();

    // --- Open drafter -----------------------------------------------
    let drafter = DFlashDrafter::open(&drafter_dir)?;
    if drafter.config.head_dim != 128 {
        anyhow::bail!(
            "drafter-chat-smoke v1 only supports head_dim=128, got {}",
            drafter.config.head_dim,
        );
    }
    let mask_token_id = drafter.config.dflash_config.mask_token_id;
    let block_size = drafter.config.block_size;
    let q_len = block_size;
    let vocab_size = drafter.config.vocab_size;
    let eos_token_id: u32 = 248044; // Qwen3.6 standard eos (see CLI chat path)

    // --- Target engine (long-context mode to fit drafter alongside) -
    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let target_config = EngineConfig {
        max_context: prompt_len
            .saturating_add(max_new_tokens)
            .saturating_add(block_size)
            .max(256),
        kv_cache_dtype: cuda_kv_cache_dtype(KvCacheDtype::Fp8),
        ..EngineConfig::default()
    };
    let target_load_start = Instant::now();
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, target_config)?;
    let target_load_seconds = target_load_start.elapsed().as_secs_f64();

    // --- Drafter weights + capture (sized for the largest single
    //     capture event: prompt prefill writes `prompt_len` rows; each
    //     verify writes up to `block_size + 1` rows). -----------------
    let drafter_load_start = Instant::now();
    let drafter_device = DFlashDrafterDevice::upload(&drafter)?;
    let capture_max_tokens = prompt_len.max(block_size + 1);
    let capture = Arc::new(TargetHiddenCapture::alloc(
        &drafter.config,
        capture_max_tokens,
    )?);
    let capture_for_hook = capture.clone();
    let hook: qwen36_fp4_runtime::DrafterHiddenCaptureHook =
        Arc::new(move |layer_idx, residual_ptr, tokens| {
            capture_for_hook
                .capture_layer(&CudaBackend, layer_idx, residual_ptr, tokens)
                .map_err(|e| qwen36_fp4_core::CoreError::Runtime(format!("drafter handoff: {e}")))
        });
    let drafter_load_seconds = drafter_load_start.elapsed().as_secs_f64();

    // --- Prefill: captures rows [0, prompt_len) in capture buffer ----
    let prefill_start = Instant::now();
    capture.set_write_row(0);
    engine.set_drafter_hidden_capture(Some(hook.clone()));
    engine.prefill(&prompt_tokens)?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    engine.queue_sample_greedy_to_current_token()?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    let mut seed_token = engine.read_current_token()?;

    // --- Drafter workspace + propose scratch ------------------------
    let manifest = engine
        .weights
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("engine has no weights manifest after prefill"))?;
    let gpu_weights = engine
        .gpu_weights
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("engine has no GPU weights after prefill"))?;
    let target_embed_ptr = gpu_weights
        .tensor(&manifest.embed_tokens.name)
        .ok_or_else(|| anyhow::anyhow!("embed_tokens tensor missing"))?
        .ptr();
    let target_lm_head_ptr = gpu_weights
        .tensor(&manifest.lm_head.name)
        .ok_or_else(|| anyhow::anyhow!("lm_head tensor missing"))?
        .ptr();

    // Peak drafter KV per iter N: `prefix_len + ctx_len + q_len`
    // where `prefix_len ≤ prompt_len + max_new_tokens`, `ctx_len ≤
    // max(prompt_len, block_size)`, and `q_len = block_size`. Bound the
    // budget by the sum + small headroom.
    let ctx_len_max = prompt_len.max(block_size);
    let kv_cache_max_len = prompt_len + max_new_tokens + ctx_len_max + block_size + 16;
    let workspace =
        DrafterForwardWorkspace::alloc(&drafter.config, q_len, ctx_len_max, kv_cache_max_len)?;
    let mut pos_bytes = Vec::with_capacity(kv_cache_max_len * 4);
    for p in 0..kv_cache_max_len {
        pos_bytes.extend_from_slice(&(p as i32).to_le_bytes());
    }
    workspace.position_ids_buffer().copy_from_host(&pos_bytes)?;
    let backend = CudaBackend;
    let mut forward = DrafterForward::new(&drafter_device, &drafter.config, workspace)?;
    let propose_ws = DFlashProposeWorkspace::alloc(&drafter.config, q_len)?;

    // --- Outer iter loop --------------------------------------------
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut per_iter: Vec<serde_json::Value> = Vec::new();
    let mut ctx_len = prompt_len; // capture rows from previous step
    let mut total_committed_after_prompt = 0_usize; // tokens generated so far

    let decode_start = Instant::now();
    while total_committed_after_prompt < max_new_tokens {
        // Drafter KV must reflect the prefix length the drafter has
        // already "seen". For iter 0 we start fresh; otherwise crop to
        // the just-committed end-of-prefix position. Note: we keep
        // total_committed_after_prompt counting seed-in-iter-0; in
        // practice this off-by-one improved AL on every prompt vs the
        // strict dflash crop semantics (drafter has slightly more
        // attention context). The bench numbers in
        // docs/superpowers/notes/2026-06-09-dflash-benchmarks.md use
        // this formula.
        let prefix_len_for_drafter = prompt_len + total_committed_after_prompt;
        if total_committed_after_prompt == 0 {
            forward.reset_kv_cache();
        } else {
            forward.crop_kv_cache(prefix_len_for_drafter)?;
        }

        // Build noise tokens: [seed, MASK, ..., MASK].
        let mut noise_token_ids = Vec::with_capacity(q_len);
        noise_token_ids.push(seed_token);
        for _ in 1..q_len {
            noise_token_ids.push(mask_token_id);
        }

        let proposed_tokens = propose_block(
            &backend,
            &mut forward,
            &propose_ws,
            &noise_token_ids,
            capture.output_ptr(),
            ctx_len,
            target_embed_ptr,
            target_lm_head_ptr,
            vocab_size,
        )?;
        qwen36_fp4_runtime::cuda_synchronize()?;
        let drafts: Vec<u32> = proposed_tokens[1..].to_vec();

        // Verify via sequential decode_one with per-decode capture
        // write_row updates so each decode lands at a fresh row.
        // Batched verify: one prefill of [seed, drafts[0]..drafts[k-1]]
        // produces argmaxes at every input position in a single forward
        // pass through the target. ~10× fewer target forwards per
        // iteration than the sequential `engine.prefill(&[t])` chain.
        // The Phase E hook fires once per layer with tokens=k+1, writing
        // capture rows [0, k+1).
        capture.set_write_row(0);
        let mut verify_input = Vec::with_capacity(drafts.len() + 1);
        verify_input.push(seed_token);
        verify_input.extend_from_slice(&drafts);
        let argmaxes = engine.verify_block_batched(&verify_input)?;

        let mut accepted = 0_usize;
        let mut bonus_token: u32 = 0;
        for (i, &drafted) in drafts.iter().enumerate() {
            // argmaxes[i] is the target's prediction for the token at
            // position i+1 given inputs [verify_input[0..=i]] — i.e.
            // "what should follow verify_input[i]". We compare to
            // drafts[i].
            let target_argmax = argmaxes[i];
            if target_argmax == drafted {
                accepted += 1;
            } else {
                bonus_token = target_argmax;
                break;
            }
        }
        if accepted == drafts.len() {
            bonus_token = argmaxes[drafts.len()];
        }

        // Rollback target state for rejected speculative tail. Batched
        // verify advanced state by `verify_input.len() = drafts.len() +
        // 1`; only `accepted + 1` of those entries (seed + accepted
        // drafts) belong in the committed prefix. Crop state back so
        // the next iter's verify writes at the correct positions.
        let committed_target_position = prompt_len + total_committed_after_prompt + accepted + 1;
        engine.crop_state_position(committed_target_position)?;

        // Commit accepted drafts + bonus. The seed itself was committed
        // by the PREVIOUS iter as its bonus (or sampled from prefill
        // for iter 0); we don't re-emit it here.
        let mut iter_committed: Vec<u32> = Vec::with_capacity(accepted + 1);
        if total_committed_after_prompt == 0 {
            // Iter 0: seed is brand new, never emitted. Include it.
            iter_committed.push(seed_token);
        }
        iter_committed.extend(drafts.iter().copied().take(accepted));
        iter_committed.push(bonus_token);

        let iter_text = tokenizer.decode(&iter_committed, true).unwrap_or_default();
        per_iter.push(serde_json::json!({
            "iter": per_iter.len(),
            "ctx_len": ctx_len,
            "accepted": accepted,
            "bonus_token": bonus_token,
            "committed_tokens": iter_committed,
            "committed_decoded": iter_text,
        }));

        // EOS check before mutating outer state.
        let mut hit_eos = false;
        for &t in &iter_committed {
            generated.push(t);
            if t == eos_token_id {
                hit_eos = true;
                break;
            }
        }
        if hit_eos {
            break;
        }

        seed_token = bonus_token;
        ctx_len = accepted + 1;
        total_committed_after_prompt = generated.len();
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let total_seconds = bench_start.elapsed().as_secs_f64();

    engine.set_drafter_hidden_capture(None);
    let generated_text = tokenizer.decode(&generated, true).unwrap_or_default();
    let tokens_per_second = if decode_seconds > 0.0 {
        generated.len() as f64 / decode_seconds
    } else {
        0.0
    };
    let acceptance_length = if per_iter.is_empty() {
        0.0
    } else {
        generated.len() as f64 / per_iter.len() as f64
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model_dir": model_dir.display().to_string(),
            "drafter_dir": drafter_dir.display().to_string(),
            "prompt": prompt,
            "chat_template": chat_template,
            "prompt_tokens": prompt_len,
            "max_new_tokens": max_new_tokens,
            "block_size": block_size,
            "iterations": per_iter.len(),
            "generated_token_count": generated.len(),
            "generated_text": generated_text,
            "timings_seconds": {
                "target_load": target_load_seconds,
                "drafter_load": drafter_load_seconds,
                "prefill": prefill_seconds,
                "decode": decode_seconds,
                "total": total_seconds,
            },
            "acceptance_length": acceptance_length,
            "tokens_per_second": tokens_per_second,
            "per_iter": per_iter,
        }))?,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn drafter_iter_smoke(
    model_dir: PathBuf,
    drafter_dir: PathBuf,
    prompt: String,
    chat_template: bool,
) -> Result<()> {
    use std::sync::Arc;

    use qwen36_fp4_drafter::{
        DFlashDrafter, DFlashDrafterDevice, DFlashProposeWorkspace, DrafterForward,
        DrafterForwardWorkspace, TargetHiddenCapture, propose_block,
    };
    use qwen36_fp4_kernels::CudaBackend;

    // === Setup (mirrors drafter_step_smoke) ========================
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
    if prompt_tokens.is_empty() {
        anyhow::bail!("prompt produced 0 tokens");
    }
    let ctx_len = prompt_tokens.len();

    let drafter = DFlashDrafter::open(&drafter_dir)?;
    if drafter.config.head_dim != 128 {
        anyhow::bail!(
            "drafter-iter-smoke v1 only supports head_dim=128, got {}",
            drafter.config.head_dim,
        );
    }
    let mask_token_id = drafter.config.dflash_config.mask_token_id;
    let block_size = drafter.config.block_size;
    let q_len = block_size;
    let vocab_size = drafter.config.vocab_size;

    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let target_config = EngineConfig {
        // Verify advances target state by up to block_size positions.
        max_context: ctx_len
            .saturating_add(block_size)
            .saturating_add(8)
            .max(256),
        kv_cache_dtype: cuda_kv_cache_dtype(KvCacheDtype::Fp8),
        ..EngineConfig::default()
    };
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, target_config)?;

    let drafter_device = DFlashDrafterDevice::upload(&drafter)?;
    let capture = Arc::new(TargetHiddenCapture::alloc(&drafter.config, ctx_len)?);
    let capture_for_hook = capture.clone();
    let hook: qwen36_fp4_runtime::DrafterHiddenCaptureHook =
        Arc::new(move |layer_idx, residual_ptr, tokens| {
            capture_for_hook
                .capture_layer(&CudaBackend, layer_idx, residual_ptr, tokens)
                .map_err(|e| qwen36_fp4_core::CoreError::Runtime(format!("drafter handoff: {e}")))
        });
    engine.set_drafter_hidden_capture(Some(hook));

    engine.prefill(&prompt_tokens)?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    engine.set_drafter_hidden_capture(None);

    engine.queue_sample_greedy_to_current_token()?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    let seed_token = engine.read_current_token()?;

    let manifest = engine
        .weights
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("engine has no weights manifest after prefill"))?;
    let gpu_weights = engine
        .gpu_weights
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("engine has no GPU weights after prefill"))?;
    let target_embed_ptr = gpu_weights
        .tensor(&manifest.embed_tokens.name)
        .ok_or_else(|| anyhow::anyhow!("embed_tokens tensor missing from GPU store"))?
        .ptr();
    let target_lm_head_ptr = gpu_weights
        .tensor(&manifest.lm_head.name)
        .ok_or_else(|| anyhow::anyhow!("lm_head tensor missing from GPU store"))?
        .ptr();

    let kv_cache_max_len = ctx_len + q_len;
    let workspace =
        DrafterForwardWorkspace::alloc(&drafter.config, q_len, ctx_len, kv_cache_max_len)?;
    let mut pos_bytes = Vec::with_capacity(kv_cache_max_len * 4);
    for p in 0..kv_cache_max_len {
        pos_bytes.extend_from_slice(&(p as i32).to_le_bytes());
    }
    workspace.position_ids_buffer().copy_from_host(&pos_bytes)?;

    let backend = CudaBackend;
    let mut forward = DrafterForward::new(&drafter_device, &drafter.config, workspace)?;
    forward.reset_kv_cache();

    // === Propose (Phase F.0) =======================================
    let propose_ws = DFlashProposeWorkspace::alloc(&drafter.config, q_len)?;
    let mut noise_token_ids = Vec::with_capacity(q_len);
    noise_token_ids.push(seed_token);
    for _ in 1..q_len {
        noise_token_ids.push(mask_token_id);
    }
    let proposed_tokens = propose_block(
        &backend,
        &mut forward,
        &propose_ws,
        &noise_token_ids,
        capture.output_ptr(),
        ctx_len,
        target_embed_ptr,
        target_lm_head_ptr,
        vocab_size,
    )?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    // Proposed[0] is the drafter's denoised seed (typically ~= seed
    // itself); the dflash protocol uses proposed[1..] as the drafts.
    let drafts: Vec<u32> = proposed_tokens[1..].to_vec();

    // === Verify via sequential prefill chunks ========================
    // Walk the target through [seed, drafts[0], drafts[1], …,
    // drafts[k-2]] one token at a time. We use engine.prefill(&[t])
    // rather than decode_one because the decode kernel path diverges
    // numerically from prefill on NVFP4 (cos sim 0.76–0.81 on the
    // logits — surfaced by `decode-vs-prefill-check`). Same state
    // advance, correct kernel.
    let mut accepted = 0_usize;
    let mut prev = seed_token;
    let mut bonus_token: u32 = 0;
    let mut comparisons: Vec<serde_json::Value> = Vec::with_capacity(drafts.len());

    for (idx, &drafted) in drafts.iter().enumerate() {
        engine.prefill(&[prev])?;
        engine.queue_sample_greedy_to_current_token()?;
        qwen36_fp4_runtime::cuda_synchronize()?;
        let target_argmax = engine.read_current_token()?;
        comparisons.push(serde_json::json!({
            "block_pos": idx + 1,
            "drafted": drafted,
            "target_argmax": target_argmax,
            "match": target_argmax == drafted,
        }));
        if target_argmax == drafted {
            accepted += 1;
            prev = drafted;
        } else {
            bonus_token = target_argmax;
            break;
        }
    }
    if accepted == drafts.len() {
        // Full accept: one more prefill for the bonus prediction
        // beyond the last accepted draft.
        engine.prefill(&[prev])?;
        engine.queue_sample_greedy_to_current_token()?;
        qwen36_fp4_runtime::cuda_synchronize()?;
        bonus_token = engine.read_current_token()?;
    }

    let committed: Vec<u32> = std::iter::once(seed_token)
        .chain(drafts.iter().copied().take(accepted))
        .chain(std::iter::once(bonus_token))
        .collect();
    let committed_text = tokenizer.decode(&committed, true).unwrap_or_default();
    let drafts_text = tokenizer.decode(&drafts, true).unwrap_or_default();

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model_dir": model_dir.display().to_string(),
            "drafter_dir": drafter_dir.display().to_string(),
            "prompt": prompt,
            "chat_template": chat_template,
            "prompt_tokens": ctx_len,
            "block_size": block_size,
            "seed_token": seed_token,
            "drafts": drafts,
            "drafts_decoded": drafts_text,
            "accepted_count": accepted,
            "bonus_token": bonus_token,
            "committed_tokens": committed,
            "committed_decoded": committed_text,
            "per_position_comparisons": comparisons,
        }))?,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn drafter_handoff_smoke(
    model_dir: PathBuf,
    drafter_dir: PathBuf,
    prompt_tokens_count: usize,
    q_len: usize,
) -> Result<()> {
    use std::sync::Arc;

    use qwen36_fp4_drafter::{
        DFlashDrafter, DFlashDrafterDevice, DrafterForward, DrafterForwardWorkspace,
        TargetHiddenCapture,
    };
    use qwen36_fp4_kernels::{CudaBackend, CudaDeviceBuffer};

    if prompt_tokens_count == 0 || q_len == 0 {
        anyhow::bail!("prompt_tokens and q_len must both be > 0");
    }

    // Open drafter (host mmap only — no GPU alloc yet).
    let drafter = DFlashDrafter::open(&drafter_dir)?;
    if drafter.config.head_dim != 128 {
        anyhow::bail!(
            "drafter-handoff-smoke v1 only supports head_dim=128, got {}",
            drafter.config.head_dim,
        );
    }

    // --- Target side first: load the heavy 17 GB before any drafter
    // allocations so the target gets contiguous VRAM.
    let layout = discover_model_layout_with_id(&model_dir, QWEN36_TEXT_NVFP4_MTP_MODEL_ID)?;
    let mapped_model = MappedModel::open_with_layout(&model_dir, layout)?;
    let target_config = EngineConfig {
        max_context: prompt_tokens_count.max(256),
        kv_cache_dtype: cuda_kv_cache_dtype(KvCacheDtype::Fp8),
        ..EngineConfig::default()
    };
    let mut engine = Engine::cuda_with_mapped_weights(&mapped_model, target_config)?;

    // Now stage the drafter pieces on GPU.
    let drafter_device = DFlashDrafterDevice::upload(&drafter)?;
    let capture = Arc::new(TargetHiddenCapture::alloc(
        &drafter.config,
        prompt_tokens_count,
    )?);
    let capture_for_hook = capture.clone();
    let hook: qwen36_fp4_runtime::DrafterHiddenCaptureHook =
        Arc::new(move |layer_idx, residual_ptr, tokens| {
            capture_for_hook
                .capture_layer(&CudaBackend, layer_idx, residual_ptr, tokens)
                .map_err(|e| qwen36_fp4_core::CoreError::Runtime(format!("drafter handoff: {e}")))
        });
    engine.set_drafter_hidden_capture(Some(hook));

    // Synthetic prompt tokens. The model's vocab is large; use ids
    // small enough to land on common bytes. We don't decode them.
    let synthetic_tokens: Vec<u32> = (0..prompt_tokens_count)
        .map(|i| ((i % 1000) + 1) as u32)
        .collect();
    engine.prefill(&synthetic_tokens)?;
    qwen36_fp4_runtime::cuda_synchronize()?;
    engine.set_drafter_hidden_capture(None);

    // --- Drafter forward on captured target_hidden_raw -----------------
    let ctx_len = prompt_tokens_count;
    let kv_cache_max_len = ctx_len + q_len;
    let workspace =
        DrafterForwardWorkspace::alloc(&drafter.config, q_len, ctx_len, kv_cache_max_len)?;

    let hidden = drafter.config.hidden_size;
    let n_target_layers = drafter.config.dflash_config.target_layer_ids.len();
    let _ = n_target_layers; // documented; pulled from capture's buffer sizing

    // Synthetic noise embedding for the drafter (deterministic bytes).
    let mut noise_bytes = vec![0u8; q_len * hidden * 2];
    let mut rng: u32 = 0xCAFE_F00D;
    for chunk in noise_bytes.chunks_exact_mut(2) {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let raw = (rng >> 16) as u16;
        // Force BF16 exponent to 125 (≈ ±0.25) so we get tame values.
        let bits = (raw & 0x807F) | (125u16 << 7);
        chunk[0] = bits as u8;
        chunk[1] = (bits >> 8) as u8;
    }
    let noise_buf = CudaDeviceBuffer::alloc(q_len * hidden * 2)?;
    noise_buf.copy_from_host(&noise_bytes)?;

    // Position ids: [0, ctx_len + q_len).
    let mut pos_bytes = Vec::with_capacity(kv_cache_max_len * 4);
    for p in 0..kv_cache_max_len {
        pos_bytes.extend_from_slice(&(p as i32).to_le_bytes());
    }
    workspace.position_ids_buffer().copy_from_host(&pos_bytes)?;

    let backend = CudaBackend;
    let mut forward = DrafterForward::new(&drafter_device, &drafter.config, workspace)?;
    forward.reset_kv_cache();
    forward.forward(
        &backend,
        noise_buf.ptr(),
        capture.output_ptr(),
        q_len,
        ctx_len,
    )?;
    qwen36_fp4_runtime::cuda_synchronize()?;

    let mut out_bytes = vec![0u8; q_len * hidden * 2];
    forward
        .workspace()
        .output_buffer()
        .copy_to_host(&mut out_bytes)?;
    let mut all_finite = true;
    let mut sample_min = f32::INFINITY;
    let mut sample_max = f32::NEG_INFINITY;
    for chunk in out_bytes.chunks_exact(2).step_by(16) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let f = f32::from_bits((bits as u32) << 16);
        if !f.is_finite() {
            all_finite = false;
            break;
        }
        if f < sample_min {
            sample_min = f;
        }
        if f > sample_max {
            sample_max = f;
        }
    }

    // Also sanity-check a couple of capture buffer entries.
    let mut capture_finite = true;
    let mut cap_sample = vec![0u8; (hidden * 2).min(capture.buffer().bytes())];
    capture.buffer().copy_to_host(&mut cap_sample)?;
    for chunk in cap_sample.chunks_exact(2).step_by(4) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let f = f32::from_bits((bits as u32) << 16);
        if !f.is_finite() {
            capture_finite = false;
            break;
        }
    }

    let drafter_report = drafter_device.report(&drafter.manifest);
    let workspace_report = forward.workspace().report();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model_dir": model_dir.display().to_string(),
            "drafter_dir": drafter_dir.display().to_string(),
            "prompt_tokens": prompt_tokens_count,
            "q_len": q_len,
            "ctx_len": ctx_len,
            "target_layer_ids": drafter.config.dflash_config.target_layer_ids,
            "capture_buffer_bytes": capture.buffer().bytes(),
            "capture_finite": capture_finite,
            "drafter_vram_bytes": drafter_report.total_bytes,
            "workspace_vram_bytes": workspace_report.total_bytes,
            "kv_cache_vram_bytes": workspace_report.kv_caches_bytes,
            "drafter_output_finite": all_finite,
            "drafter_output_sample_min": sample_min,
            "drafter_output_sample_max": sample_max,
        }))?,
    );

    if !all_finite || !capture_finite {
        anyhow::bail!("non-finite values in capture or drafter output");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn drafter_forward_smoke(
    drafter_dir: PathBuf,
    q_len_arg: usize,
    ctx_len_arg: usize,
    iterations: usize,
    fixture_dir: Option<PathBuf>,
) -> Result<()> {
    use qwen36_fp4_drafter::{
        DFlashDrafter, DFlashDrafterDevice, DrafterForward, DrafterForwardWorkspace,
    };
    use qwen36_fp4_kernels::{CudaBackend, CudaDeviceBuffer, DevicePtr};

    if iterations == 0 {
        anyhow::bail!("iterations must be > 0");
    }

    let drafter = DFlashDrafter::open(&drafter_dir)?;
    if drafter.config.head_dim != 128 {
        anyhow::bail!(
            "drafter-forward-smoke v1 only supports head_dim=128, got {}",
            drafter.config.head_dim,
        );
    }
    let hidden = drafter.config.hidden_size;
    let n_target_layers = drafter.config.dflash_config.target_layer_ids.len();
    let target_raw_row_bytes = hidden * n_target_layers * 2;

    // Decide inputs: fixture vs synthetic deterministic LCG. The
    // fixture path lets us parity-check against the Python reference
    // produced by `scripts/dflash_parity.py`.
    let (q_len, ctx_len, noise_bytes, target_bytes, position_bytes, expected_output) =
        if let Some(dir) = fixture_dir.as_ref() {
            load_fixture(dir)?
        } else {
            if q_len_arg == 0 {
                anyhow::bail!("q_len must be > 0 in synthetic mode");
            }
            let (n, t) = synthesize_inputs(q_len_arg, ctx_len_arg, hidden, n_target_layers);
            let kv = ctx_len_arg + q_len_arg;
            let mut pos_bytes = Vec::with_capacity(kv * 4);
            for p in 0..kv {
                pos_bytes.extend_from_slice(&(p as i32).to_le_bytes());
            }
            (q_len_arg, ctx_len_arg, n, t, pos_bytes, None)
        };

    if q_len * hidden * 2 != noise_bytes.len() {
        anyhow::bail!(
            "noise input size {} bytes does not match q_len*hidden*2 = {}",
            noise_bytes.len(),
            q_len * hidden * 2,
        );
    }
    if ctx_len > 0 && ctx_len * target_raw_row_bytes != target_bytes.len() {
        anyhow::bail!(
            "target_raw input size {} bytes does not match ctx_len*hidden*n_target_layers*2 = {}",
            target_bytes.len(),
            ctx_len * target_raw_row_bytes,
        );
    }

    let device = DFlashDrafterDevice::upload(&drafter)?;
    let kv_cache_max_len = (ctx_len + q_len).max(1);
    let workspace =
        DrafterForwardWorkspace::alloc(&drafter.config, q_len, ctx_len, kv_cache_max_len)?;
    let report = workspace.report();
    let kv_seq_len = ctx_len + q_len;

    let noise_buf = CudaDeviceBuffer::alloc(q_len * hidden * 2)?;
    noise_buf.copy_from_host(&noise_bytes)?;
    let target_buf = if ctx_len > 0 {
        let buf = CudaDeviceBuffer::alloc(ctx_len * target_raw_row_bytes)?;
        buf.copy_from_host(&target_bytes)?;
        Some(buf)
    } else {
        None
    };
    let target_ptr = target_buf
        .as_ref()
        .map(|b| b.ptr())
        .unwrap_or(DevicePtr::NULL);
    workspace
        .position_ids_buffer()
        .copy_from_host(&position_bytes)?;

    let backend = CudaBackend;
    let mut forward = DrafterForward::new(&device, &drafter.config, workspace)?;

    let mut first_output: Option<Vec<u8>> = None;
    let mut all_finite = true;
    for iter in 0..iterations {
        // Reset cache between iterations so each forward sees identical
        // state (iter-1 semantics). A future controller drives this via
        // `crop_kv_cache(accepted_prefix)` instead.
        forward.reset_kv_cache();
        forward.forward(&backend, noise_buf.ptr(), target_ptr, q_len, ctx_len)?;
        qwen36_fp4_runtime::cuda_synchronize()?;
        let mut out_bytes = vec![0u8; q_len * hidden * 2];
        forward
            .workspace()
            .output_buffer()
            .copy_to_host(&mut out_bytes)?;

        let mut local_finite = true;
        for chunk in out_bytes.chunks_exact(2).step_by(16) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let f32_bits = (bits as u32) << 16;
            let value = f32::from_bits(f32_bits);
            if !value.is_finite() {
                local_finite = false;
                break;
            }
        }
        all_finite &= local_finite;

        if iter == 0 {
            first_output = Some(out_bytes);
        } else if let Some(prev) = &first_output {
            if prev != &out_bytes {
                anyhow::bail!(
                    "drafter forward is non-deterministic across iterations \
                     (iteration {iter} bytes differ from iteration 0)"
                );
            }
        }
    }

    let cos_sim = expected_output.as_ref().map(|expected| {
        let got = first_output.as_ref().expect("ran at least one iteration");
        bf16_cosine_similarity(got, expected)
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "drafter_dir": drafter.drafter_dir.display().to_string(),
            "fixture_dir": fixture_dir.as_ref().map(|p| p.display().to_string()),
            "q_len": q_len,
            "ctx_len": ctx_len,
            "kv_seq_len": kv_seq_len,
            "iterations": iterations,
            "workspace_total_bytes": report.total_bytes,
            "workspace_total_gib": bytes_to_gib(report.total_bytes as u64),
            "drafter_vram_bytes": device.report(&drafter.manifest).total_bytes,
            "output_bytes": q_len * hidden * 2,
            "finite": all_finite,
            "deterministic": true,
            "cos_sim_vs_reference": cos_sim,
        }))?,
    );

    if let Some(cos) = cos_sim {
        const FLOOR: f64 = 0.998;
        if cos < FLOOR {
            anyhow::bail!("drafter forward cos sim {cos:.6} < floor {FLOOR}; parity regression");
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn synthesize_inputs(
    q_len: usize,
    ctx_len: usize,
    hidden: usize,
    n_target_layers: usize,
) -> (Vec<u8>, Vec<u8>) {
    // Small deterministic LCG. Range ~[-0.5, 0.5) via BF16 with
    // exponent forced to 125 (≈ 2^-2).
    let mut rng_state: u32 = 0x9E37_79B9;
    let next_u16 = |state: &mut u32| -> u16 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*state >> 16) as u16
    };
    let bf16_random = |state: &mut u32| -> [u8; 2] {
        let raw = next_u16(state);
        let mantissa = (raw & 0x007F) as u32;
        let exp_bias_127 = 125u32;
        let sign = ((raw >> 15) & 1) as u32;
        let bits = (sign << 15) | (exp_bias_127 << 7) | mantissa;
        (bits as u16).to_le_bytes()
    };
    let mut noise = vec![0u8; q_len * hidden * 2];
    for i in 0..(q_len * hidden) {
        let b = bf16_random(&mut rng_state);
        noise[i * 2] = b[0];
        noise[i * 2 + 1] = b[1];
    }
    let target_elems = ctx_len * hidden * n_target_layers;
    let mut target = vec![0u8; ctx_len.max(1) * hidden * n_target_layers * 2];
    for i in 0..target_elems {
        let b = bf16_random(&mut rng_state);
        target[i * 2] = b[0];
        target[i * 2 + 1] = b[1];
    }
    (noise, target)
}

#[cfg(feature = "cuda")]
type DFlashFixture = (usize, usize, Vec<u8>, Vec<u8>, Vec<u8>, Option<Vec<u8>>);

#[cfg(feature = "cuda")]
fn load_fixture(dir: &std::path::Path) -> Result<DFlashFixture> {
    let config_path = dir.join("config.json");
    let config_bytes = std::fs::read(&config_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", config_path.display()))?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes)?;
    let q_len = config["q_len"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("fixture config.json missing q_len"))?
        as usize;
    let ctx_len = config["ctx_len"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("fixture config.json missing ctx_len"))?
        as usize;
    let noise = std::fs::read(dir.join("noise.bf16"))?;
    let target = std::fs::read(dir.join("target_raw.bf16"))?;
    let positions = std::fs::read(dir.join("positions.i32"))?;
    let expected = std::fs::read(dir.join("expected_output.bf16"))?;
    Ok((q_len, ctx_len, noise, target, positions, Some(expected)))
}

#[cfg(feature = "cuda")]
fn bf16_cosine_similarity(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (ca, cb) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        let av = bf16_bytes_to_f32(ca);
        let bv = bf16_bytes_to_f32(cb);
        dot += av as f64 * bv as f64;
        na += av as f64 * av as f64;
        nb += bv as f64 * bv as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(feature = "cuda")]
fn bf16_bytes_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let f32_bits = (bits as u32) << 16;
    f32::from_bits(f32_bits)
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
