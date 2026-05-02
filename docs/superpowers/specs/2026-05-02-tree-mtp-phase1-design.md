# Tree-MTP Phase 1: Last-Position Top-K Acceptance

**Date:** 2026-05-02 (revised same-day after design review)
**Branch:** `feat/perf-tree-mtp-stack`
**Status:** Design — pending user approval
**Phase:** 1 of 3 (Phase 2: tree-mask attention + per-branch DeltaNet for true multi-level branching, Phase 3: adaptive shape)

> **Revision note.** An earlier draft of this design put the K leaves into
> the verify chunk and required a tree-mask attention kernel + per-branch
> DeltaNet replay + KV scratch slots. Re-reading §4 of that draft showed
> that the leaves are only ever compared against `verified[D-1]` (the
> model's argmax at the *last chain* row), and never against logits at the
> leaves' own positions. The leaves therefore do not need to be in the
> verify chunk at all. This revision drops the tree-mask kernel and the
> per-branch state machinery; they are deferred to Phase 2 where true
> multi-level branching needs them. Phase 1 keeps the same observable
> behaviour and the same expected gain.

## 1. Goal and non-goals

**Goal.** Increase decode throughput on `Qwen3.6-27B-Text-NVFP4-MTP` / RTX 5090 by raising the speculative-decoding acceptance length on prompts where the model is uncertain at the *last* speculative position. Today's chain MTP=3 commits `1 + accepted_chain` ≤ 4 tokens per cycle. By proposing **K candidates at the last position** (top-K from the MTP head's last output) instead of 1, and accepting the highest-prob leaf that matches the base model's argmax at that position, expected accepted length grows by up to +1 per cycle.

**Target gain.** +20-40 % over the current MTP=3 throughput (~86 → ~105-120 tok/s) on real prompts (not the full-accept gated bench). On the gated bench (full-accept on `hello` / `hello world`) the change should be neutral, since chain acceptance is already 1.0 and the model's argmax at the last chain row is already deterministic.

**Non-goals (out of Phase 1).**
- Multi-level tree branching (Phase 2).
- Tree-mask attention kernel, per-branch DeltaNet recurrent state, KV scratch per branch (Phase 2 primitives).
- Adaptive tree shape based on MTP head confidence (Phase 3).
- Stochastic / non-greedy sampling. Phase 1 stays greedy.
- Touching the prefill path. Only decode-side MTP draft generation + acceptance changes.
- Any new CUDA Graph kind. The existing `MtpVerifyOne` / `MtpVerifyMulti` graphs are reused unchanged.

## 2. Quality contract

**Hard parity (regression gate, blocks merge):**
- `chat --prompt "hello" --max-new-tokens 12` produces identical token streams for `--mtp-speculative-tokens` ∈ {0, 1, 2, 3} with **`--mtp-tree-leaves` ∈ {1, 2, 4}**.
- `chat --prompt "hello world" --max-new-tokens 12` same.

**Soft parity (acceptable, must not regress):**
- The same envelope of 1-2 token drift on borderline prompts (`Write a short poem about cats.`, `Count from 1 to 5.`, `Write Python hello world`) that today's chunked-verify produces. Branching only changes which token is committed at the boundary, not the verify chunk's numerical path.

**Op-level parity (kernel changes):**
- The new `qwen36_topk_argmax` kernel must agree exactly with the CPU `topk_argmax` reference for K ∈ {1..8} on a fixed-seed random vocab.
- No other kernels change.

## 3. The pivot: last-position top-K acceptance

A general tree spec decoding scheme branches at every level. For our hybrid model that requires per-branch DeltaNet recurrent state for 48 of 64 layers (Phase 2 territory).

**Phase 1 trades multi-level branching for zero kernel-graph surgery.** The chain verify chunk is unchanged. The MTP head's last forward (which today samples one greedy draft for next cycle's `last_token` seed) instead samples top-K candidates. After verify, we already have `verified[D-1]` — the model's greedy argmax at the last chain position. Acceptance walk:

```
verified = [v_0, v_1, ..., v_{D-1}]   (one per chain row, sampled by today's verify graph)
chain    = [c_1, c_2, ..., c_{D-1}]   (today's chain drafts)
leaves   = [L_1, L_2, ..., L_K]       (NEW: top-K from MTP head's last output, K=1 = today's behaviour)

walk:
  commit v_0
  for i in 1..D-1:
    if v_{i-1} == c_i:  commit c_i
    else:               STOP, next_token = v_{i-1}, accepted_leaf = None
  # Full chain accepted; leaf-level check uses the verify chunk's last sample.
  for j in 0..K:
    if leaves[j] == v_{D-1}:
      commit leaves[j]
      next_token = leaves[j]
      accepted_leaf = j
      return
  # No leaf matched; commit v_{D-1} as today.
  next_token = v_{D-1}
  accepted_leaf = None
```

Tokens per cycle:
- Today (K=1): `1 + accepted_chain` ≤ `1 + D-1` = D
- Phase 1 (K>1): `1 + accepted_chain + (leaf_accepted ? 1 : 0)` ≤ `D + 1`

The +1 only fires on cycles where the chain fully accepts AND `v_{D-1}` lands in the K-best of the MTP head. Empirically this is the regime where today's MTP=3 stalls — the model is confident through the chain but the MTP head's top-1 doesn't match.

**KV cache and DeltaNet state.** When a leaf is accepted, the leaf token becomes the next cycle's `last_token`. The next cycle's verify chunk includes that leaf at row 0, so the leaf's K/V and DeltaNet state are computed naturally on the next forward — no scratch buffers, no replay.

**Why this is strictly simpler than the earlier draft.**
- The verify chunk shape, kernels, and graph are unchanged.
- No tree-mask attention. No `tree_ancestor_bitmap_u64`. No per-branch DeltaNet. No KV scratch slots.
- The only new CUDA primitive is `qwen36_topk_argmax`. Everything else is host-side Rust.

## 4. MTP head: top-K at the last position

Today the MTP head is recursive: each draft step samples greedy argmax from the head's logits and feeds it into the next step. Phase 1 keeps draft steps 1..D-1 as today (greedy chain) and changes only the last step (`draft_idx == D-1`):

- Run the MTP head from the last accepted hidden state to produce logits (today's behaviour).
- Sample **top-K argmax** from those logits → K leaf candidates, sorted by descending logit.
- The top-1 leaf equals today's `next_draft_token`. The acceptance walk (§3) tries leaves in input order, so the top-1 is checked first — when K=1 (default), Phase 1 reduces to today's chain MTP exactly.

Implementation: a new kernel `qwen36_topk_argmax` that takes BF16 logits `[V]` and produces K `u32` token IDs `[K]`. K is small (≤8 in Phase 1), single-block kernel sufficient.

## 5. CUDA changes (kernels-cuda/)

### 5.1 Top-K argmax kernel

New entry in `ops.cu`:

```c
typedef struct {
  size_t vocab_size;
  size_t k;                                // 1..8
  qwen36_device_ptr_t logits_bf16;
  qwen36_device_ptr_t output_token_u32;    // [k] u32, sorted desc by logit
} qwen36_topk_argmax_spec_t;

#define QWEN36_TOPK_MAX 8

int qwen36_topk_argmax(const qwen36_topk_argmax_spec_t *spec);
```

Single block, 512 threads. Each thread maintains a thread-local sorted top-K array in registers, scans `vocab_size` elements with stride 512, then a single-thread block-level merge of the per-thread arrays (BLOCK × K ≤ 4096 entries, cheap). Vocab is ~152k.

**No other kernels change.** The existing `attention_prefill_kernel` family, `qwen36_deltanet_*`, `qwen36_swiglu_nvfp4_quantize`, etc., are untouched.

## 6. Runtime changes (crates/runtime/src/engine.rs)

### 6.1 No new constants

`MtpKvSnapshotLayout::VERIFY_TOKENS` stays at 4. `MTP_MAX_DRAFT_TOKENS` stays at 3. The verify chunk is unchanged.

### 6.2 New entry point

```rust
pub struct TreeDraft {
    /// Length = chain_depth (= D-1, today's MTP=N draft count). The first
    /// chain draft follows last_token.
    pub chain_tokens: Vec<u32>,
    /// Length = K. Top-K candidates from the MTP head's last forward,
    /// sorted by descending logit. K=1 reproduces chain MTP exactly.
    pub leaf_tokens: Vec<u32>,
}

pub struct TreeVerifyResult {
    /// Full ordered list of tokens committed this cycle. Always satisfies
    /// `committed.len() == accepted_chain + 1` and
    /// `committed.last() == Some(next_token)`. When `accepted_leaf == Some(idx)`,
    /// `committed.last() == leaf_tokens[idx]`. When the chain rejects at row j,
    /// `committed = chain_tokens[0..j] + [verified[j]]` (length j+1).
    pub committed: Vec<u32>,
    pub accepted_chain: usize,        // 0..=chain_depth
    pub accepted_leaf: Option<usize>, // 0..K
    /// Verified token at the last accepted position; seed for next cycle's
    /// last_token. Equal to `committed.last()`.
    pub next_token: u32,
}

/// Drives one decode cycle using the existing chunked verify graph plus
/// last-position top-K acceptance.
pub fn verify_mtp_tree_draft(
    &mut self,
    draft: &TreeDraft,
    start_position: usize,
) -> Result<TreeVerifyResult>;
```

The body is small (~50 lines):

1. Run the existing `MtpVerifyMulti` (or `MtpVerifyOne` if chain_depth==1) verify graph for `draft.chain_tokens.len()` drafts. Today's call.
2. Read `verified[0..=chain_depth]` from the existing `mtp_verify_token_u32` slots. Today's call.
3. Walk the chain in pure Rust (existing logic).
4. If chain fully accepted: scan `draft.leaf_tokens` for the first match against `verified[chain_depth]`. If found, commit it.
5. Return `TreeVerifyResult`.

No graph changes. No new CUDA kernels invoked beyond what today's path already runs (the top-K kernel is invoked separately during draft generation, not during verify).

### 6.3 Draft generation extension

Today's `generate_chain_drafts` returns `chain_tokens`. Add a sibling helper:

```rust
fn generate_top_k_leaves(
    &mut self,
    last_chain_hidden: DevicePtr,
    k: usize,
) -> Result<Vec<u32>>;
```

Body: 1 MTP head forward (already done as part of the chain's last step today, hidden state is reusable) + 1 `qwen36_topk_argmax` launch + 1 D2H copy of K u32. When k == 1, return `[chain_tokens.last_top_1]` and skip the kernel.

### 6.4 Pure-Rust acceptance walk

Implemented in `crates/mtp/src/lib.rs` so it can be unit-tested without a GPU:

```rust
pub fn walk_tree_acceptance(
    verified: &[u32],
    draft: &TreeDraft,
) -> TreeVerifyResult;
```

See §3 for the algorithm. Tests cover: chain rejects at root / mid / end-without-leaf-match / end-with-leaf-match / K=1 reproduces chain MTP.

## 7. CLI / config

Extend `--mtp-speculative-tokens` semantics: keep the integer for chain depth, add `--mtp-tree-leaves <K>` (default 1 = current behaviour, no branching). Bench reports both `accepted_chain / chain_depth` and `leaf_accepted_rate` to make the tree gain measurable.

```bash
qwen36 chat --mtp-speculative-tokens 3 --mtp-tree-leaves 4 ...
qwen36 bench --mtp-speculative-tokens 3 --mtp-tree-leaves 4 ...
```

`MtpConfig` gains a `tree_leaves: usize` field, defaulting to 1.

## 8. Testing strategy

1. **Pure-Rust unit tests** for `topk_argmax` (CPU reference) and `walk_tree_acceptance` in `crates/mtp/src/lib.rs`. No GPU needed.
2. **CUDA smoke** for `qwen36_topk_argmax`: K=4 on a 1024-vocab BF16 logits array with planted top-4. Compare device output against the planted indices.
3. **Engine-level parity** (the gate): `chat --prompt "hello" --max-new-tokens 12` for `--mtp-tree-leaves` ∈ {1, 2, 4} against MTP=0 baseline and against today's `--mtp-tree-leaves 1` for each chain depth. Hard gate.
4. **Bench**: `qwen36 bench --prompt-tokens 128 --max-new-tokens 32` for `--mtp-tree-leaves` ∈ {1, 2, 4, 8} on the existing reference prompt set + 5 borderline / open prompts. Median of 5 runs. Report decode tok/s + leaf-accept rate per K.

## 9. Parity harness

No changes to `scripts/decode_parity.py`. The verify chunk path is unchanged, so the existing op-level parity gate covers Phase 1.

## 10. Rollback / kill-switch

`QWEN36_MTP_TREE_DISABLE=1` → forces `tree_leaves = 1` regardless of CLI, falling back to today's chain MTP. Mirror the existing `QWEN36_MTP_MULTI_GRAPH_DISABLE` env-var pattern. Bisecting tree numerical issues uses this env var.

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
|--|--|--|
| `qwen36_topk_argmax` kernel disagrees with CPU reference for some K | Low | Smoke + Rust-side unit comparison on multiple seeds |
| MTP head's top-K matches `verified[D-1]` less often than estimated, gain < +20 % | Medium | Bench reports leaf-accept rate; if < 0.3 we revisit K and / or move to Phase 2 |
| K=8 head latency dominates the gain on cycles where chain rejects early | Low | When chain rejects, the leaves were generated speculatively but never used — sunk cost ≈ 50 µs (top-K kernel + 1 MTP head forward, latter is reusable from chain). Re-bench with K=2 and K=4 if K=8 regresses |
| Real-world acceptance gain is smaller than the +20-40 % estimate | Medium | Measure on a varied prompt set before claiming the win; report acceptance distribution, not just tok/s |

## 12. Phase 2 entry points (preserved by Phase 1)

Phase 1 leaves the path open to Phase 2 (true multi-level tree branching). When we get there, these are the artefacts that need to land:

- A `qwen36_attention_prefill` variant accepting a per-row ancestor bitmap (the tree-mask kernel originally drafted in this spec).
- `MtpKvSnapshotLayout::VERIFY_TOKENS` bumped to handle multi-level tree row counts.
- A batched / fan-out DeltaNet decode kernel (`qwen36_deltanet_update_tree`) consuming a parent-index array.
- A new `MtpVerifyTree` graph kind, with KV scratch slots per branch.
- An adaptive tree-shape generator (Phase 3) that calls into the Phase 2 kernels.

Phase 1's `TreeDraft` / `TreeVerifyResult` types are forward-compatible with multi-level trees (the `chain_tokens` / `leaf_tokens` split generalises naturally to a flat `tokens` + `parents` representation when Phase 2 lands).

## 13. Implementation phases (within Phase 1)

Each phase is a separate PR / merge candidate, gated on its own parity check:

1. **P1.1** — Top-K argmax kernel (CUDA + ABI + Rust wrapper + smoke + CPU reference + unit tests). No engine changes yet.
2. **P1.2** — Engine-side `verify_mtp_tree_draft` + draft-generation extension + acceptance walk unit tests + CLI flag. Hard parity gate.
3. **P1.3** — Bench matrix on RTX 5090 across K ∈ {1, 2, 4, 8} and a varied prompt set; pick default K; document results in `AGENT.md`.

## 14. Success criteria

Phase 1 is done when:

- All hard parity gates pass (P1.2 onward).
- `qwen36 bench --mtp-tree-leaves 4` shows ≥ +15 % decode tok/s over `--mtp-tree-leaves 1` on the varied prompt set (median of 5 runs).
- `QWEN36_MTP_TREE_DISABLE=1` recovers the current behaviour bit-for-bit.
- The single-prompt full-accept gated bench (`hello` / `hello world`) is unchanged within ±2 %.
- `cargo clippy --workspace --features qwen36-fp4-kernels/cuda -- -D warnings` and the existing CUDA test suite stay green.
