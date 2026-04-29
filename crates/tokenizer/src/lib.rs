use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct QwenTokenizer {
    inner: Tokenizer,
    chat_template: Option<String>,
}

impl QwenTokenizer {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow!("{err}"))
            .with_context(|| format!("load {}", tokenizer_path.display()))?;
        let chat_template = read_chat_template(model_dir)?;
        Ok(Self {
            inner,
            chat_template,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|err| anyhow!("{err}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(tokens, skip_special_tokens)
            .map_err(|err| anyhow!("{err}"))
    }

    pub fn render_chat(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let _template_available = self.chat_template.is_some();
        render_minimal_qwen_chat(messages, add_generation_prompt)
    }

    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>> {
        let prompt = self.render_chat(messages, add_generation_prompt);
        self.encode(&prompt, false)
    }
}

fn read_chat_template(model_dir: &Path) -> Result<Option<String>> {
    let jinja = model_dir.join("chat_template.jinja");
    if jinja.exists() {
        return fs::read_to_string(&jinja)
            .map(Some)
            .with_context(|| format!("read {}", jinja.display()));
    }

    let tokenizer_config = model_dir.join("tokenizer_config.json");
    if tokenizer_config.exists() {
        let value: serde_json::Value = serde_json::from_slice(
            &fs::read(&tokenizer_config)
                .with_context(|| format!("read {}", tokenizer_config.display()))?,
        )
        .with_context(|| format!("parse {}", tokenizer_config.display()))?;
        if let Some(template) = value.get("chat_template").and_then(|v| v.as_str()) {
            return Ok(Some(template.to_owned()));
        }
    }
    Ok(None)
}

fn render_minimal_qwen_chat(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut rendered = String::new();
    for message in messages {
        rendered.push_str("<|im_start|>");
        rendered.push_str(&message.role);
        rendered.push('\n');
        rendered.push_str(&message.content);
        rendered.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        rendered.push_str("<|im_start|>assistant\n");
    }
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_basic_chat_prompt() {
        let prompt = render_minimal_qwen_chat(
            &[ChatMessage {
                role: "user".to_owned(),
                content: "Bonjour".to_owned(),
            }],
            true,
        );
        assert!(prompt.contains("<|im_start|>user\nBonjour<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
