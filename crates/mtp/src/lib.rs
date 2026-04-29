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

    pub fn step<R: MtpRuntime>(&self, runtime: &mut R, last_token: u32) -> Result<MtpStepResult> {
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
        runtime.snapshot_recurrent_state()?;
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

    #[test]
    fn rejection_restores_to_state_after_last_token_then_replays_committed() {
        let decoder = SpeculativeDecoder::new(MtpConfig::default());
        let mut runtime = MockRuntime::new(vec![99], vec![12]);

        let result = decoder.step(&mut runtime, 10).unwrap();

        assert_eq!(result.committed_tokens, vec![11]);
        assert_eq!(result.drafted_tokens, vec![99]);
        assert_eq!(result.accepted_draft_tokens, 0);
        assert_eq!(runtime.consumed_tokens, vec![10, 11]);
        assert!(runtime.restored);
    }

    #[test]
    fn accepted_draft_leaves_state_ready_for_next_last_token() {
        let decoder = SpeculativeDecoder::new(MtpConfig::default());
        let mut runtime = MockRuntime::new(vec![12], vec![12]);

        let result = decoder.step(&mut runtime, 10).unwrap();

        assert_eq!(result.committed_tokens, vec![11, 12]);
        assert_eq!(result.drafted_tokens, vec![12]);
        assert_eq!(result.accepted_draft_tokens, 1);
        assert_eq!(runtime.consumed_tokens, vec![10, 11]);
        assert!(!runtime.restored);
    }

    struct MockRuntime {
        consumed_tokens: Vec<u32>,
        snapshot: Vec<u32>,
        draft_tokens: Vec<u32>,
        verify_tokens: Vec<u32>,
        restored: bool,
    }

    impl MockRuntime {
        fn new(draft_tokens: Vec<u32>, verify_tokens: Vec<u32>) -> Self {
            Self {
                consumed_tokens: Vec::new(),
                snapshot: Vec::new(),
                draft_tokens,
                verify_tokens,
                restored: false,
            }
        }
    }

    impl MtpRuntime for MockRuntime {
        fn snapshot_recurrent_state(&mut self) -> Result<()> {
            self.snapshot = self.consumed_tokens.clone();
            Ok(())
        }

        fn restore_recurrent_state(&mut self) -> Result<()> {
            self.consumed_tokens = self.snapshot.clone();
            self.restored = true;
            Ok(())
        }

        fn replay_committed(&mut self, tokens: &[u32]) -> Result<()> {
            self.consumed_tokens.extend_from_slice(tokens);
            Ok(())
        }

        fn forward_main_and_mtp(
            &mut self,
            last_token: u32,
            draft_tokens: usize,
        ) -> Result<MtpDraft> {
            self.consumed_tokens.push(last_token);
            Ok(MtpDraft {
                main_logits: logits_for(11),
                draft_logits: self
                    .draft_tokens
                    .iter()
                    .take(draft_tokens)
                    .copied()
                    .map(logits_for)
                    .collect(),
            })
        }

        fn forward_main_only(&mut self, token: u32) -> Result<Logits> {
            self.consumed_tokens.push(token);
            let next = self.verify_tokens.remove(0);
            Ok(logits_for(next))
        }
    }

    fn logits_for(token: u32) -> Logits {
        let mut values = vec![0.0; 128];
        values[token as usize] = 1.0;
        Logits { values }
    }
}
