use serde::{Deserialize, Serialize};

use qwen36_fp4_core::Result;
use qwen36_fp4_kernels::sampling::greedy_argmax;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpConfig {
    pub num_speculative_tokens: usize,
    pub greedy: bool,
    /// Top-K branching at the last MTP head position. 1 = chain MTP behaviour
    /// (Phase 1 default).
    pub tree_leaves: usize,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 3,
            greedy: true,
            tree_leaves: 1,
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

pub const MTP_TREE_MAX_LEAVES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeDraft {
    /// Length = chain_depth (= today's MTP=N draft count). First chain draft
    /// follows last_token.
    pub chain_tokens: Vec<u32>,
    /// Length = K. Top-K candidates from the MTP head's last forward,
    /// sorted by descending logit. K = 1 reproduces chain MTP exactly.
    pub leaf_tokens: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeVerifyResult {
    /// Full ordered list of tokens committed this cycle. Always satisfies
    /// `committed.len() == accepted_chain + 1 + (accepted_leaf.is_some() ? 1 : 0)`
    /// and `committed.last() == Some(next_token)`. When `accepted_leaf` is
    /// `Some(idx)`, `committed[committed.len() - 2] == leaf_tokens[idx]`.
    pub committed: Vec<u32>,
    pub accepted_chain: usize,        // 0..=chain_depth
    pub accepted_leaf: Option<usize>, // 0..K
    /// Verified token at the last accepted position; seed for next cycle's
    /// `last_token`.
    pub next_token: u32,
    /// Pre-computed chain drafts for the next cycle. Length =
    /// `chain_depth` (matches the chain_tokens length passed in).
    /// Empty when verify_mtp_tree_draft was called with chain_depth=0
    /// or when MTP next-draft generation was skipped.
    pub next_chain_drafts: Vec<u32>,
    /// Pre-computed leaf drafts for the next cycle (top-K from MTP head's
    /// last step). Length = `leaf_count` (matches the leaf_tokens length
    /// passed in).
    pub next_leaf_drafts: Vec<u32>,
}

/// Walk a branched-tail tree given the model's argmax at each chunk row.
///
/// `verified` length must be `1 + chain_depth + leaf_count`:
/// - `verified[0..=chain_depth]` are chain row outputs.
/// - `verified[chain_depth + 1 + j]` is the output of leaf row j (= the
///   model's prediction one position past leaf j).
///
/// Acceptance:
/// - Chain row i accepts `chain_tokens[i]` iff `verified[i] == chain_tokens[i]`.
/// - On the first chain reject, `next_token = verified[i]`.
/// - If chain fully accepts, leaves are scanned in input order against
///   `verified[chain_depth]`; first match wins. The accepted leaf's row output
///   `verified[chain_depth + 1 + j]` becomes `next_token`.
///
/// Invariants:
/// - `committed.len() == accepted_chain + 1 + (accepted_leaf.is_some() ? 1 : 0)`.
/// - `committed.last() == Some(next_token)` always.
/// - When `accepted_leaf == Some(idx)`: `committed[committed.len() - 2] ==
///   leaf_tokens[idx]`.
pub fn walk_tree_acceptance(verified: &[u32], draft: &TreeDraft) -> TreeVerifyResult {
    let chain_depth = draft.chain_tokens.len();
    let leaf_count = draft.leaf_tokens.len();
    debug_assert!(verified.len() >= 1 + chain_depth + leaf_count);
    let mut committed = Vec::with_capacity(chain_depth + 2);
    let mut accepted_chain = 0;
    for (i, &candidate) in draft.chain_tokens.iter().enumerate() {
        if verified[i] == candidate {
            committed.push(candidate);
            accepted_chain = i + 1;
        } else {
            committed.push(verified[i]);
            return TreeVerifyResult {
                committed,
                accepted_chain,
                accepted_leaf: None,
                next_token: verified[i],
                next_chain_drafts: Vec::new(),
                next_leaf_drafts: Vec::new(),
            };
        }
    }
    let chain_verified = verified[chain_depth];
    let accepted_leaf = draft
        .leaf_tokens
        .iter()
        .position(|&leaf| leaf == chain_verified);
    if let Some(idx) = accepted_leaf {
        let after_leaf = verified[chain_depth + 1 + idx];
        committed.push(draft.leaf_tokens[idx]);
        committed.push(after_leaf);
        TreeVerifyResult {
            committed,
            accepted_chain,
            accepted_leaf: Some(idx),
            next_token: after_leaf,
            next_chain_drafts: Vec::new(),
            next_leaf_drafts: Vec::new(),
        }
    } else {
        committed.push(chain_verified);
        TreeVerifyResult {
            committed,
            accepted_chain,
            accepted_leaf: None,
            next_token: chain_verified,
            next_chain_drafts: Vec::new(),
            next_leaf_drafts: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tree_tests {
    use super::*;

    fn assert_committed_invariants(result: &TreeVerifyResult, draft: &TreeDraft) {
        let expected_len =
            result.accepted_chain + 1 + if result.accepted_leaf.is_some() { 1 } else { 0 };
        assert_eq!(
            result.committed.len(),
            expected_len,
            "committed.len() must equal accepted_chain + 1 + (leaf? 1 : 0)"
        );
        assert_eq!(
            result.committed.last().copied(),
            Some(result.next_token),
            "committed.last() must equal next_token"
        );
        if let Some(idx) = result.accepted_leaf {
            let n = result.committed.len();
            assert_eq!(
                result.committed[n - 2],
                draft.leaf_tokens[idx],
                "committed[len-2] must equal leaf_tokens[accepted_leaf]"
            );
        }
    }

    #[test]
    fn k1_chain_full_accept_leaf_match() {
        let draft = TreeDraft {
            chain_tokens: vec![10, 20, 30],
            leaf_tokens: vec![50],
        };
        let verified = vec![10, 20, 30, 50, 99];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![10, 20, 30, 50, 99]);
        assert_eq!(r.accepted_chain, 3);
        assert_eq!(r.accepted_leaf, Some(0));
        assert_eq!(r.next_token, 99);
        assert_committed_invariants(&r, &draft);
    }

    #[test]
    fn k1_chain_full_accept_no_leaf_match() {
        let draft = TreeDraft {
            chain_tokens: vec![10, 20, 30],
            leaf_tokens: vec![999],
        };
        let verified = vec![10, 20, 30, 50, 0];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![10, 20, 30, 50]);
        assert_eq!(r.accepted_chain, 3);
        assert_eq!(r.accepted_leaf, None);
        assert_eq!(r.next_token, 50);
        assert_committed_invariants(&r, &draft);
    }

    #[test]
    fn full_chain_first_leaf_match() {
        let draft = TreeDraft {
            chain_tokens: vec![10, 20, 30],
            leaf_tokens: vec![100, 200, 300],
        };
        let verified = vec![10, 20, 30, 200, 70, 71, 72];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![10, 20, 30, 200, 71]);
        assert_eq!(r.accepted_chain, 3);
        assert_eq!(r.accepted_leaf, Some(1));
        assert_eq!(r.next_token, 71);
        assert_committed_invariants(&r, &draft);
    }

    #[test]
    fn full_chain_top_leaf_wins_over_lower_match() {
        let draft = TreeDraft {
            chain_tokens: vec![10],
            leaf_tokens: vec![42, 100, 42],
        };
        let verified = vec![10, 42, 555, 666, 777];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.accepted_leaf, Some(0));
        assert_eq!(r.committed, vec![10, 42, 555]);
        assert_eq!(r.next_token, 555);
        assert_committed_invariants(&r, &draft);
    }

    #[test]
    fn chain_rejects_at_first_mismatch() {
        let draft = TreeDraft {
            chain_tokens: vec![10, 20, 30],
            leaf_tokens: vec![100, 200],
        };
        let verified = vec![10, 99, 0, 0, 0, 0];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![10, 99]);
        assert_eq!(r.accepted_chain, 1);
        assert_eq!(r.accepted_leaf, None);
        assert_eq!(r.next_token, 99);
        assert_committed_invariants(&r, &draft);
    }

    #[test]
    fn chain_rejects_at_root_skips_leaves() {
        let draft = TreeDraft {
            chain_tokens: vec![10, 20],
            leaf_tokens: vec![999],
        };
        let verified = vec![888, 0, 0, 0];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![888]);
        assert_eq!(r.accepted_chain, 0);
        assert_eq!(r.accepted_leaf, None);
        assert_eq!(r.next_token, 888);
        assert_committed_invariants(&r, &draft);
    }

    #[test]
    fn empty_chain_with_leaf_match() {
        let draft = TreeDraft {
            chain_tokens: vec![],
            leaf_tokens: vec![55, 66],
        };
        let verified = vec![66, 77, 88];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![66, 88]);
        assert_eq!(r.accepted_chain, 0);
        assert_eq!(r.accepted_leaf, Some(1));
        assert_eq!(r.next_token, 88);
        assert_committed_invariants(&r, &draft);
    }

    #[test]
    fn empty_chain_no_leaf_match() {
        let draft = TreeDraft {
            chain_tokens: vec![],
            leaf_tokens: vec![55, 66],
        };
        let verified = vec![999, 0, 0];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![999]);
        assert_eq!(r.accepted_chain, 0);
        assert_eq!(r.accepted_leaf, None);
        assert_eq!(r.next_token, 999);
        assert_committed_invariants(&r, &draft);
    }
}
