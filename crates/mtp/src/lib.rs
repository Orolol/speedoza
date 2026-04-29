use serde::{Deserialize, Serialize};

use qwen36_fp4_core::Result;
use qwen36_fp4_kernels::sampling::greedy_argmax;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpConfig {
    pub num_speculative_tokens: usize,
    pub greedy: bool,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 3,
            greedy: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logits {
    pub values: Vec<f32>,
}

impl Logits {
    pub fn argmax(&self) -> Option<u32> {
        greedy_argmax(&self.values)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpDraft {
    pub main_logits: Logits,
    pub draft_logits: Vec<Logits>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpStepResult {
    pub committed_tokens: Vec<u32>,
    pub drafted_tokens: Vec<u32>,
    pub accepted_draft_tokens: usize,
}

pub trait MtpRuntime {
    fn snapshot_recurrent_state(&mut self) -> Result<()>;
    fn restore_recurrent_state(&mut self) -> Result<()>;
    fn replay_committed(&mut self, tokens: &[u32]) -> Result<()>;
    fn forward_main_and_mtp(&mut self, last_token: u32, draft_tokens: usize) -> Result<MtpDraft>;
    fn forward_main_only(&mut self, token: u32) -> Result<Logits>;
}

#[derive(Debug, Clone)]
pub struct SpeculativeDecoder {
    config: MtpConfig,
}

impl SpeculativeDecoder {
    pub fn new(config: MtpConfig) -> Self {
        assert!(config.num_speculative_tokens > 0);
        Self { config }
    }

    pub fn step<R: MtpRuntime>(
        &self,
        runtime: &mut R,
        last_token: u32,
    ) -> Result<MtpStepResult> {
        runtime.snapshot_recurrent_state()?;
        let draft = runtime.forward_main_and_mtp(last_token, self.config.num_speculative_tokens)?;
        let Some(main_token) = draft.main_logits.argmax() else {
            return Ok(MtpStepResult {
                committed_tokens: Vec::new(),
                drafted_tokens: Vec::new(),
                accepted_draft_tokens: 0,
            });
        };
        let drafted_tokens = draft
            .draft_logits
            .iter()
            .filter_map(Logits::argmax)
            .collect::<Vec<_>>();

        let mut committed = vec![main_token];
        let mut accepted = 0;
        for &candidate in &drafted_tokens {
            let previous_token = committed[committed.len() - 1];
            let verify_logits = runtime.forward_main_only(previous_token)?;
            let verified = verify_logits.argmax();
            if verified == Some(candidate) {
                committed.push(candidate);
                accepted += 1;
            } else {
                runtime.restore_recurrent_state()?;
                runtime.replay_committed(&committed)?;
                break;
            }
        }

        Ok(MtpStepResult {
            committed_tokens: committed,
            drafted_tokens,
            accepted_draft_tokens: accepted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logits_argmax_is_stable() {
        let logits = Logits {
            values: vec![0.1, 2.0, 1.0],
        };
        assert_eq!(logits.argmax(), Some(1));
    }
}
