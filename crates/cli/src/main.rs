use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use qwen36_fp4_core::{
    KvCacheDtype, MemoryBudget, ModelTopology, QWEN36_TEXT_NVFP4_MTP_MODEL_ID,
};
use qwen36_fp4_loader::{
    discover_model_layout_with_id, read_topology, write_model_layout_json,
};
use qwen36_fp4_runtime::{Engine, EngineConfig};
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
        Command::Chat {
            model_dir,
            prompt,
            max_new_tokens,
            mtp_speculative_tokens,
        } => {
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
            let mut engine = Engine::no_cuda(&layout, config);
            let _ = max_new_tokens;
            if let Err(err) = engine.prefill(&prompt_tokens) {
                bail!(
                    "CUDA backend is not linked; prefill/decode cannot run with backend {}: {err}",
                    engine.backend_name()
                );
            }
        }
    }
    Ok(())
}
