# Tree-MTP Phase 1: Branched-Tail Speculative Decoding

**Date:** 2026-05-02
**Branch:** `feat/perf-tree-mtp-stack`
**Status:** Design — pending user approval
**Phase:** 1 of 3 (Phase 2: full-tree DeltaNet, Phase 3: adaptive shape)

## 1. Goal and non-goals

**Goal.** Increase decode throughput on `Qwen3.6-27B-Text-NVFP4-MTP` / RTX 5090 by raising the speculative-decoding acceptance length on prompts where the model is uncertain at the *last* speculative position. Today's chain MTP=3 commits 1 + accepted_drafts ≤ 4 tokens per cycle and acceptance falls below 1.0 on borderline argmax positions. By proposing **K candidates at the last position** instead of 1, expected accepted length grows without changing the chain-recurrent path that DeltaNet requires.

**Target gain.** +20-40 % over the current MTP=3 throughput (~86 → ~105-120 tok/s) on real prompts (not the full-accept gated bench). On the gated bench (full-accept on `hello` / `hello world`) the change should be neutral, since chain acceptance is already 1.0.

**Non-goals (out of Phase 1).**
- Full tree branching at every depth (Phase 2; requires per-branch DeltaNet recurrent state).
- Adaptive tree shape based on MTP head confidence (Phase 3).
- Stochastic / non-greedy sampling. Phase 1 stays greedy.
- Touching the prefill path. Only decode-side MTP verify changes.
- Tree-MTP for full attention layers in isolation. The change is end-to-end (head + verify + acceptance + snapshot/restore).

## 2. Quality contract

**Hard parity (regression gate, blocks merge):**
- `chat --prompt "hello" --max-new-tokens 12` produces identical token streams for `--mtp-speculative-tokens` ∈ {0, 1, 2, 3, 4} with **branching ∈ {1, 2, 4}** at the last position.
- `chat --prompt "hello world" --max-new-tokens 12` same.

**Soft parity (acceptable, must not regress):**
- The same envelope of 1-2 token drift on borderline prompts (`Write a short poem about cats.`, `Count from 1 to 5.`, `Write Python hello world`) that today's chunked-verify produces. Branching must not widen this envelope.

**Op-level parity (kernel changes):**
- Any new CUDA kernel introduced (tree-mask attention prefill, top-K sampling) must pass cos sim ≥ 0.998 against a CPU / PyTorch reference at the layer boundary. The existing parity harness (`/tmp/parity_check.py`, `scripts/decode_parity.py`) extends naturally — see §10.

## 3. The pivot: "branched-tail" tree

A general tree spec decoding scheme branches at every level. For our hybrid model that requires per-branch DeltaNet recurrent state for 48 of 64 layers, which is Phase 2 territory.

**Phase 1 trades generality for tractability**: the tree is **a single chain of length D-1 followed by K branches at depth D**. Concretely with D=4, K=4:

```
                  committed_token
                         │
                  draft_chain_1
                         │
                  draft_chain_2
                         │
                  draft_chain_3
                       /│ │\
                      / │ │ \
                 leaf_a leaf_b leaf_c leaf_d   <- top-K from MTP head at depth D
```

**Why this works without per-branch DeltaNet state:**
- Tokens 1..D-1 form a linear chunk → DeltaNet sees a normal sequential chunk (current code).
- Leaves a..K all share the same DeltaNet input state (the state after `draft_chain_3`). Each leaf is a *single-step* DeltaNet update from that state. We snapshot the state at depth D-1, compute each leaf's resulting state independently, and commit only the accepted leaf's state.
- Full attention layers naturally handle this via tree-mask: leaves attend to chain prefix + themselves, never to siblings.

**Verify chunk layout (host launch order):**
```
row 0 : committed_token        (causal mask: standard prefix)
row 1 : draft_chain_1          (causal: row 0)
row 2 : draft_chain_2          (causal: rows 0..1)
row 3 : draft_chain_3          (causal: rows 0..2)
row 4 : leaf_a                 (tree-mask: rows 0..3)
row 5 : leaf_b                 (tree-mask: rows 0..3, NOT row 4)
row 6 : leaf_c                 (tree-mask: rows 0..3, NOT rows 4, 5)
row 7 : leaf_d                 (tree-mask: rows 0..3, NOT rows 4, 5, 6)
```
Total verify chunk size = `(D-1) + 1 + K = D + K = 8 rows` for D=4, K=4 (vs `D = 4` rows today on chain MTP=3).

## 4. Acceptance walk

Greedy acceptance, walked once per verify cycle:

1. Sample `verified[i]` = argmax of logits at row `i` for `i = 0..D-1`. (Reuses today's queue_sample_greedy_into.)
2. Walk the chain: for `i = 1..D-1`, accept `draft_chain_i` iff `verified[i-1] == draft_chain_i`. Stop on first reject.
3. If chain accepted fully (depth D-1 reached): walk the K leaves. Accept the *first* leaf `leaf_x` whose `draft_token == verified[D-1]`. If none match, no leaf accepted; chain commits up to `verified[D-1]`.
4. Commit count = `1 + accepted_chain + (leaf_accepted ? 1 : 0)`.

This delivers **+1 expected token per cycle when the chain accepts fully and any of K leaves matches** — which is exactly the case where today's chain MTP=3 stalls at acceptance < 1.0.

## 5. MTP head: top-K at depth D

Today the MTP head is recursive: each draft step samples greedy argmax from the head's logits and feeds it into the next step. Phase 1 keeps draft steps 1..D-1 as today (greedy chain) and changes only the last step (`draft_idx == D-1`):

- Run the MTP head from the last accepted hidden state to produce logits.
- Sample **top-K argmax** from those logits → K leaf candidates.

Implementation: a new kernel `qwen36_topk_argmax_kernel` that takes BF16 logits `[V]` and produces K `u32` token IDs `[K]`. K is small (≤8 in Phase 1), so a single-block reduction with K-element insertion sort in shared memory is sufficient. No need for a full top-K sort.

## 6. CUDA changes (kernels-cuda/)

### 6.1 Tree-mask in `attention_prefill_kernel`

`attention_prefill_kernel` currently masks via position index: each query row `q` attends to KV rows `0..=q`. Need a variant that takes a per-row **ancestor bitmap** instead.

**API change** (`include/qwen36_fp4.h`):
```c
typedef struct qwen36_attention_prefill_spec_t {
    // ... existing fields ...
    /// Tree-mask bitmap: row i attends to KV row j iff bit j of word i is set.
    /// Length = 2 * verify_tokens words (supports up to 128 rows).
    /// NULL → use causal mask (current behaviour).
    const uint64_t *ancestor_bitmap;
    uint32_t verify_chunk_rows;    /// rows in the tree chunk (D + K)
} qwen36_attention_prefill_spec_t;
```

Implementation: replace `if (kv_idx > q_idx) skip` with `if (!(bitmap[q_idx] & (1ULL << kv_idx))) skip` for the `kv_idx < verify_chunk_rows` band. KV positions before the verify chunk (the committed prefix in the KV cache) remain fully visible.

**Affected kernels**: `attention_prefill_kernel`, `attention_prefill_split_kernel`, `attention_prefill_gqa_kernel`. Same pattern in all three.

**Smoke / parity**: extend `kernels-cuda/smoke.cu` with a 4-leaf tree-mask case. Compare against the reference Python that builds the same mask explicitly.

### 6.2 Top-K argmax kernel

New entry in `ops.cu`:
```c
qwen36_status_t qwen36_topk_argmax(
    const __nv_bfloat16 *logits,    // [V]
    uint32_t vocab_size,
    uint32_t k,                      // 1..8
    uint32_t *out_token_ids          // [k]
);
```
Single block, V threads (V=151936 padded to 152064 = 1188 × 128). Each thread loads a vocab slice, finds local top-K via insertion into a thread-local K-array, then warp + block reduction merges to global top-K. K is small; the merge cost is negligible.

### 6.3 DeltaNet "fan-out" single-step

For the K leaves: each is a single-token DeltaNet update from the same input state `S`. Today the runtime computes one update at a time. For K=4 leaves we need to compute 4 (state, output) pairs from `S`.

Two options:
- **(a) Loop K times**, calling `qwen36_deltanet_update` once per leaf. Simple, K small. Costs K kernel launches × 48 layers = 192 launches per cycle.
- **(b) Add a batched variant** `qwen36_deltanet_update_batched` that takes K input tokens and produces K (state, output) pairs from the same input state. One launch per layer.

Phase 1 picks **(a)**. K=4 is small, launch overhead per leaf is ~3 µs × 48 = 600 µs total — already amortised by CUDA Graph capture (see §7). Move to (b) only if profiling shows leaf launches dominate.

### 6.4 No new conv1d primitives needed

The DeltaNet conv1d update reads the previous 4 tokens; we replay it K times for K leaves the same way as 6.3. The conv state snapshot/restore code already exists in `MtpKvSnapshotLayout`.

## 7. Runtime changes (crates/runtime/src/engine.rs)

### 7.1 Constants

```rust
const MTP_TREE_MAX_CHAIN_DEPTH: usize = 3;   // D-1, matches today's MTP=3 cap
const MTP_TREE_MAX_LEAVES: usize = 8;        // K cap
const MTP_MAX_DRAFT_TOKENS: usize =
    MTP_TREE_MAX_CHAIN_DEPTH + MTP_TREE_MAX_LEAVES;       // 11
const MTP_TREE_MAX_VERIFY_ROWS: usize =
    1 + MTP_MAX_DRAFT_TOKENS;                              // 12 (committed + drafts)
```

Bitmap uses `u64`, so up to 64 rows fits with the current encoding. The runtime hard-asserts `verify_rows ≤ 64`.

### 7.2 New entry point

```rust
pub struct TreeDraft {
    pub chain_tokens: Vec<u32>,    // length D-1
    pub leaf_tokens: Vec<u32>,     // length K (top-K from MTP head)
}

pub struct TreeVerifyResult {
    pub committed: Vec<u32>,         // committed_token + accepted chain + (accepted leaf if any)
    pub accepted_chain: usize,       // 0..=chain_depth
    pub accepted_leaf: Option<usize>,// index 0..K of accepted leaf, if any
    pub next_token: u32,             // sampled at the last accepted position; seed for next cycle's draft generation
}

pub fn verify_mtp_tree_draft(
    &mut self,
    draft: &TreeDraft,
    start_position: usize,
    chain_depth: usize,    // D-1 (≤ MTP_MAX_DRAFT_TOKENS - 1)
    leaf_count: usize,     // K (≤ MTP_TREE_MAX_LEAVES)
) -> Result<TreeVerifyResult>;
```

### 7.3 Verify graph capture

Add a third graph variant alongside the current `MtpVerifyOne` and `MtpVerifyMulti`:

```rust
enum DecodeGraphKind {
    Decode,
    MtpVerifyOne,
    MtpVerifyMulti { drafts: usize, assume_accept: bool },
    MtpVerifyTree { chain_depth: usize, leaf_count: usize },  // NEW
}
```

Verify rows = `1 + chain_depth + leaf_count` (committed + chain drafts + leaves). With D=4, K=4: 8 rows.

Capture sequence (mirrors `ensure_mtp_verify_graph_multi_tokens`):

1. Set tree-mask bitmap on the device-side `cuda_forward()?.tree_ancestor_bitmap` (host-built once, H2D in the graph).
2. `prefill_cuda_chunk(verify_rows, start_position, …, tree_mask=true)` — runs all 64 layers on the verify chunk with the ancestor bitmap.
3. `final_norm_prefill_rows(verify_rows)`.
4. For `i in 0..(1 + chain_depth)`: `prefill_row_logits(i)` + `queue_sample_greedy_into(verified_slot(i))` — produces `verified[0..D]` used by the chain acceptance walk.
5. For each leaf `j in 0..leaf_count`: `prefill_row_logits(1 + chain_depth + j)` + `queue_sample_greedy_into(leaf_next_slot(j))`. `leaf_next_slot(j)` is the *next-token-after-leaf-j* sampled greedily from the base model — needed only for the next cycle's `last_token` if leaf `j` is accepted.

**What is NOT captured in this graph (Phase 1):** the next cycle's MTP draft generation (chain + top-K leaves). After acceptance, the runtime exits the verify graph, switches to the existing MTP draft graph, and generates next drafts from the accepted hidden state via a separate (already-captured) MTP forward. Trade-off: one extra graph launch per cycle (~5 µs amortised) vs. capturing K parallel MTP futures inside the verify graph (~K × chain_depth extra MTP forwards, dwarfing the saving).

Re-evaluate this choice in P1.5 if profiling shows the inter-graph hop dominates.

### 7.4 Snapshot / restore extension

`MtpKvSnapshotLayout::VERIFY_TOKENS` is currently 4. Bump to 8 (D + K cap). The snapshot now covers the chain rows + the K leaves. Restore-then-replay logic walks the accepted chain length + (0 or 1) leaf to commit only the accepted path.

### 7.5 Tree ancestor bitmap (host-built)

Host-side helper builds the bitmap once per cycle:

```rust
fn build_tree_bitmap(chain_depth: usize, leaf_count: usize) -> [u64; MTP_TREE_MAX_VERIFY_ROWS] {
    let mut rows = [0u64; MTP_TREE_MAX_VERIFY_ROWS];
    // Chain rows: causal triangle.
    for i in 0..=chain_depth {
        rows[i] = (1u64 << (i + 1)) - 1;
    }
    // Leaf rows: each sees the full chain, never any sibling leaf.
    let chain_mask = (1u64 << (chain_depth + 1)) - 1;
    for j in 0..leaf_count {
        rows[chain_depth + 1 + j] = chain_mask | (1u64 << (chain_depth + 1 + j));
    }
    rows
}
```

H2D copy of 64 bytes once per verify cycle is free.

## 8. CLI / config

Extend `--mtp-speculative-tokens` semantics: keep the integer for chain depth, add `--mtp-tree-leaves <K>` (default 1 = current behaviour, no branching). Bench reports both `accepted_chain / chain_depth` and `leaf_accepted (yes|no)` to make the tree gain measurable.

```bash
qwen36 chat --mtp-speculative-tokens 3 --mtp-tree-leaves 4 ...
qwen36 bench --mtp-speculative-tokens 3 --mtp-tree-leaves 4 ...
```

## 9. Testing strategy

1. **Pure-Rust unit tests** for `build_tree_bitmap`, `walk_tree_acceptance`, snapshot restore arithmetic. Mock runtime extension in `crates/mtp/src/lib.rs` to cover tree shapes without GPU.
2. **CUDA smoke** for `qwen36_attention_prefill` with tree-mask: synthetic Q/K/V, manual ancestor bitmap, compare against a Python causal+manual-mask reference.
3. **CUDA smoke** for `qwen36_topk_argmax`: compare top-K against `np.argpartition` on random BF16 logits.
4. **Engine-level parity**: `chat --prompt "hello" --max-new-tokens 12` for `--mtp-tree-leaves` ∈ {1, 2, 4} against MTP=0. Hard gate.
5. **Bench**: `qwen36 bench --prompt-tokens 128 --max-new-tokens 32` for `--mtp-tree-leaves` ∈ {1, 2, 4, 8} on the existing reference prompt set + 3 borderline prompts. Median of 5 runs.

## 10. Parity harness extension

Add a tree-aware variant of `scripts/decode_parity.py`:

- New env var `QWEN36_PARITY_TREE_DEPTH` and `QWEN36_PARITY_TREE_LEAVES`.
- The Python reference builds the same ancestor bitmap and applies it to its causal-attention reference.
- Asserts cos sim ≥ 0.998 on each verify-row hidden state (D + K rows total).

## 11. Rollback / kill-switch

`QWEN36_MTP_TREE_DISABLE=1` → forces `leaf_count = 1` regardless of CLI, falling back to current chain MTP. Same pattern as `QWEN36_MTP_MULTI_GRAPH_DISABLE`. Bisecting tree numerical issues uses this env var.

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|--|--|--|
| Tree-mask attention introduces a numerical drift > current chunked-verify envelope | Medium | Op-level parity gate (cos sim ≥ 0.998) + hard gate on `hello` / `hello world` |
| K=4 leaf MTP head launches dominate cycle time, killing the gain | Low | Profile early; option (b) batched DeltaNet update available as fallback |
| Snapshot buffer growth (4 → 8 rows) pressures VRAM | Very low | ~2× of a small allocation; <100 MB total |
| Verify graph capture cost grows (more nodes) and amortisation breaks for short generations | Low | Re-use the same captured graph for all cycles in a chat/bench run; capture cost paid once |
| Real-world acceptance gain is smaller than the +20-40 % estimate | Medium | Measure on a varied prompt set before claiming the win; report acceptance distribution, not just tok/s |

## 13. Out of scope (Phase 2 / 3 hooks)

- Phase 2 will introduce a tree-aware DeltaNet recurrent kernel (`qwen36_deltanet_update_tree`) that consumes a parent-index array + a flat tree of token rows, computing branching states in parallel. Phase 1's bitmap representation already extends to arbitrary trees, so the runtime API is forward-compatible.
- Phase 3 will replace the fixed `(chain_depth, leaf_count)` with a confidence-driven tree-shape generator (à la EAGLE-2) that calls into the same Phase 1/2 kernels.

## 14. Implementation phases (within Phase 1)

Each phase is a separate PR / merge candidate, gated on its own parity check:

1. **P1.1** — Top-K sampling kernel + Rust wrapper + unit tests. No engine changes yet.
2. **P1.2** — Tree-mask in `attention_prefill_kernel` (and the two siblings) + smoke parity test. Behind env-gated `QWEN36_TREE_MASK_TEST`.
3. **P1.3** — `verify_mtp_tree_draft` host-launch path (no graph). Wire through CLI flag, bench, parity harness. Hard gate.
4. **P1.4** — Tree verify graph capture + soft-fallback to host launch if capture fails.
5. **P1.5** — Profiling pass on RTX 5090; tune K (likely 2 or 4) based on real-prompt acceptance gain. Decide whether to schedule a batched DeltaNet kernel.

## 15. Success criteria

Phase 1 is done when:

- All four hard parity gates pass (P1.3 onward).
- `qwen36 bench --mtp-tree-leaves 4` shows ≥ +15 % decode tok/s over `--mtp-tree-leaves 1` on a varied 8-prompt set (median).
- `QWEN36_MTP_TREE_DISABLE=1` recovers the current behaviour bit-for-bit.
- The single-prompt full-accept gated bench is unchanged (within ±2 %).
- `cargo clippy --workspace --features qwen36-fp4-kernels/cuda -- -D warnings` and the existing CUDA test suite stay green.
