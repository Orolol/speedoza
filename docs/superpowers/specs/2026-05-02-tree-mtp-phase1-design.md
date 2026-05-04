# Tree-MTP Phase 1 (revised v3): Branched-Tail with Leaves IN Chunk

**Date:** 2026-05-02 (third revision after iterating with the user)
**Branch:** `feat/perf-tree-mtp-stack`
**Status:** Design — pending user approval
**Depends on:** main HEAD `097e692` (PR #2: MTP4 + batched lm_head + sample_rows kernel)
**Phase:** 1 of 3 (Phase 2: per-branch DeltaNet + multi-level branching, Phase 3: adaptive shape)

> **Revision history.**
> - **v1** (`56ed8dc`): full design with leaves in chunk + tree-mask attention + per-leaf DeltaNet replay.
> - **v2** (`e27ad9b`): "simplified" design with leaves OUTSIDE chunk. Discovered mid-implementation
>   that this design has zero token-per-cycle gain because the leaves predict the *same* position
>   as `verified[chain_depth]`. Leaves outside the chunk give no extra commit.
> - **v3** (this doc): back to v1's leaves-in-chunk approach, on top of PR #2's MTP4 baseline.
>   Explicit measurement gate because the per-row DeltaNet cost may eat the per-cycle gain
>   given the high baseline.

## 1. Goal and economic context

**Goal.** Increase decode tokens per cycle by extending the verify chunk with K leaf candidates AT POSITIONS BEYOND THE CHAIN. When chain fully accepts AND a leaf matches `verified[chain_depth]`, we commit the leaf AND the next-position verified token, giving +2 commit count over chain MTP today.

**Baseline (post PR #2, main `097e692`):**

| MTP | tokens/cycle (full accept) | tok/s | speedup vs MTP=3 |
|-----|----|----|----|
| 0 | 1 | ~45 | 0.4x |
| 3 | 4 | 110 | 1.0x |
| 4 | 5 | 117-120 | 1.07x |

**Target (with tree-MTP=3, K=4) per acceptance regime:**

| Regime | tokens/cycle | extra vs MTP=4 |
|-----|----|----|
| chain rejects at depth j (j < 3) | j + 1 | 0 |
| chain fully accepts, no leaf match | 4 | -1 |
| chain fully accepts, leaf match + verified-after-leaf | 6 | +1 |

The +1 token in the best case has to OVERCOME the verify-chunk extension cost (1+chain+K = 8 rows for chain=3,K=4 vs MTP=4's 5 rows = 1.6x chunk size).

**Risk-honest target:** ≥ +5% decode tok/s vs MTP=4 on a varied prompt set; otherwise the design ships as Phase 2 infrastructure with K=1 default (no perf gain delivered, infra ready for true multi-level branching).

## 2. Quality contract

**Hard parity (regression gate, blocks merge):**
- `chat --prompt "hello" --max-new-tokens 12` produces identical token streams for `--mtp-speculative-tokens` ∈ {0, 1, 2, 3, 4} with `--mtp-tree-leaves` ∈ {1, 2, 4}.
- `chat --prompt "hello world" --max-new-tokens 12` same.

**Soft parity:** same envelope of 1-2 token drift on borderline prompts that today's chunked verify produces. Tree-mask attention must not widen the envelope.

**Op-level parity:**
- New `qwen36_attention_prefill` tree-mask path: cos sim ≥ 0.998 against a Python reference that builds the same per-row ancestor mask explicitly.
- Per-leaf DeltaNet snapshot/restore: cos sim ≥ 0.998 at each leaf row's hidden state vs. a Python reference that snapshots and replays.

## 3. The branched-tail tree

```
                  current_token   (row 0, position P)
                         │
                  draft_chain_0   (row 1, position P+1)   ← chain_tokens[0]
                         │
                  draft_chain_1   (row 2, position P+2)   ← chain_tokens[1]
                         │
                  draft_chain_2   (row 3, position P+3)   ← chain_tokens[2]
                       /│ │\
                      / │ │ \
                 leaf_0  ...  leaf_K-1                    ← K candidates, all at position P+chain_depth+1
              (rows 4, 5, 6, 7)
```

Verify chunk row count: `1 + chain_depth + K`.

**Position semantics:**
- Rows 0..chain_depth occupy positions P..P+chain_depth (sequential chain).
- All K leaf rows logically sit at position P + chain_depth + 1 (siblings competing for the same slot).
- `verified[i]` (model argmax at row i) is the prediction for the position AFTER row i's input:
  - `verified[0..chain_depth]` predicts positions P+1..P+chain_depth+1 (chain successors).
  - `verified[chain_depth + 1 + j]` predicts position P+chain_depth+2 (after leaf j).

**Why this gives +2 tokens per cycle when both accept:**
1. Chain fully accepts → commit chain[0..chain_depth] at positions P+1..P+chain_depth.
2. `verified[chain_depth]` predicts position P+chain_depth+1. If leaf_j == `verified[chain_depth]`, leaf_j is the correct token at P+chain_depth+1 → commit it.
3. `verified[chain_depth + 1 + j]` (the accepted leaf's row output) predicts position P+chain_depth+2, conditioned on leaf_j being correct. Since leaf_j was just accepted, this prediction is valid → commit it as the cycle's `next_token`.
4. Total: chain (chain_depth) + leaf (1) + verified-after-leaf (1) = chain_depth + 2 tokens.

For chain_depth = 3, K = 4: up to 5 + 1 = 6 tokens per cycle on full accept + leaf accept.

## 4. Acceptance walk

```rust
verified = [v_0, v_1, ..., v_{chain_depth}, v_{chain_depth+1}, ..., v_{chain_depth+K}]

walk:
  // Chain (today's logic)
  accepted_chain = 0
  for i in 0..chain_depth:
    if v_i == chain[i]:  accepted_chain = i + 1
    else:
      committed = chain[0..accepted_chain] + [v_i]
      next_token = v_i
      return TreeVerifyResult { committed, accepted_chain, accepted_leaf: None, next_token }

  // Full chain accepted; check leaves
  let chain_verified = v_{chain_depth}     // predicts position P+chain_depth+1
  for j in 0..K:
    if leaves[j] == chain_verified:
      let next_token = v_{chain_depth + 1 + j}
      committed = chain[0..chain_depth] + [leaves[j]] + [next_token]
      return TreeVerifyResult { committed, accepted_chain, accepted_leaf: Some(j), next_token }

  // No leaf matched
  committed = chain[0..chain_depth] + [chain_verified]
  next_token = chain_verified
  return TreeVerifyResult { committed, accepted_chain, accepted_leaf: None, next_token }
```

**Invariants** (enforced by `assert_committed_invariants` in tests):
- `committed.last() == Some(next_token)` always.
- `committed.len() == accepted_chain + 1 + (accepted_leaf.is_some() ? 1 : 0)`.
- When `accepted_leaf.is_some()`: `committed[committed.len() - 2] == leaves[accepted_leaf.unwrap()]`.

**This differs from v2's invariant** — `committed.len()` can now be `accepted_chain + 2` when a leaf accepts, because the verified-after-leaf is also committed. The walk_tree_acceptance code already shipped (`8fb5931`) needs an update for this.

## 5. CUDA changes

### 5.1 Tree-mask attention prefill (kernels-cuda/attention.cu)

Three prefill kernels (`attention_prefill_kernel`, `attention_prefill_split_kernel`, `attention_prefill_gqa_kernel`) gain a per-row ancestor bitmap path. NULL bitmap → existing causal behaviour (zero regression on chain-only paths).

**API change** (`include/qwen36_fp4.h`):

```c
typedef struct qwen36_attention_prefill_spec_t {
    // ... existing fields ...

    /// Tree-mask bitmap. When non-NULL, row i of the verify chunk attends to
    /// KV row j (within the same chunk) iff bit j of word i is set. KV positions
    /// before `start_position` (cache prefix) remain fully visible regardless.
    /// NULL → causal mask (existing behaviour).
    qwen36_device_ptr_t tree_ancestor_bitmap_u64;
    /// Verify-chunk row count (number of valid bitmap entries). 0 = causal.
    /// Capped at 64 by the bitmap encoding.
    size_t verify_chunk_rows;
} qwen36_attention_prefill_spec_t;
```

Kernel modification: replace the existing causal gate
```cpp
if (kv_idx > q_idx) skip;
```
with
```cpp
const bool in_chunk = (kv_idx >= chunk_base) && (kv_idx < chunk_base + verify_chunk_rows);
if (verify_chunk_rows > 0 && tree_ancestor_bitmap_u64 != nullptr && in_chunk) {
  uint32_t row = q_idx - chunk_base;
  uint32_t col = kv_idx - chunk_base;
  if (row >= verify_chunk_rows) continue;
  uint64_t mask = tree_ancestor_bitmap_u64[row];
  if (!(mask & (1ULL << col))) continue;
} else {
  if (kv_idx > q_idx) continue;     // causal default
}
```

`chunk_base` = `start_position` (the absolute position the chunk starts at).

### 5.2 Per-leaf DeltaNet handling (host-orchestrated, no new kernel in Phase 1)

DeltaNet is recurrent; the chunked DeltaNet kernel processes a chunk sequentially. For tree leaves all branching from the same parent state, processing them as a chain would corrupt the recurrent state.

**Phase 1 approach:**

1. Run `prefill_cuda_chunk(chain_depth + 1, ...)` for chain rows ONLY (rows 0..chain_depth). DeltaNet processes them as a normal chunk; conv_history and DeltaNet state advance correctly.
2. Snapshot DeltaNet state + conv history per layer (extend existing `mtp_snapshot_state`).
3. For each leaf j in 0..K:
   - Restore DeltaNet/conv state to the chain-end snapshot.
   - Run `prefill_cuda_chunk(1, ..., tree_mask=…)` for the leaf row.
   - Snapshot the resulting DeltaNet/conv state into per-leaf scratch.
4. Walk acceptance.
5. Restore DeltaNet/conv to either the chain-end snapshot (if no leaf accepted) OR the accepted leaf's snapshot.

**Cost:** K × per-layer DeltaNet single-token launch + K snapshot/restore round trips. With K=4, 48 DeltaNet layers, ~5 µs per layer launch in WSL2 ≈ 1 ms launches. Plus state copies. Estimated 2-4 ms per cycle of overhead — needs measurement (see §11).

**Phase 2 alternative:** A new `qwen36_deltanet_decode_batched` kernel that takes K input tokens + 1 input state and produces K (state, output) pairs in parallel. Saves the launches. Out of scope for Phase 1.

### 5.3 KV cache scratch slots per leaf

Full attention layers write K/V to the cache at the leaf positions. Since all K leaves logically sit at position P+chain_depth+1, they'd overwrite each other if written to the same slot.

**Approach:** write each leaf's K/V to a **separate scratch buffer per leaf per layer** (extend `MtpKvSnapshotLayout`). On acceptance, copy the accepted leaf's scratch K/V into the main cache at position P+chain_depth+1.

`MtpKvSnapshotLayout::VERIFY_TOKENS` already at 5 (post PR #2). Bump to 13 to support chain=4 + K=8 = 13 verify rows.

Memory cost per layer: K × 2 (K, V) × kv_heads × head_dim × bytes. For Qwen3.6: ~2 KB per leaf per layer × 16 full-attn layers × 8 leaves = ~256 KB scratch per cycle. Trivial.

### 5.4 Top-K kernel: already shipped

Already landed in commits `2b6535d` (CPU reference) through `eafd742` (Rust wrapper). Reused as-is.

## 6. Runtime changes

### 6.1 Constants

```rust
// crates/mtp/src/lib.rs (already landed: MTP_TREE_MAX_LEAVES = 8)

// crates/runtime/src/gpu.rs
impl MtpKvSnapshotLayout {
    pub const VERIFY_TOKENS: usize = 13;        // bump from 5; chain=4 + K=8 = 13
}
```

### 6.2 New entry point

```rust
pub struct TreeDraft {
    pub chain_tokens: Vec<u32>,
    pub leaf_tokens: Vec<u32>,        // length K, sorted desc by logit
}

pub struct TreeVerifyResult {
    pub committed: Vec<u32>,
    pub accepted_chain: usize,
    pub accepted_leaf: Option<usize>,
    pub next_token: u32,               // == committed.last()
}

pub fn verify_mtp_tree_draft(
    &mut self,
    current_token: u32,
    chain_tokens: &[u32],
    leaf_tokens: &[u32],
    next_draft_count: usize,
) -> Result<TreeVerifyResult>;
```

### 6.3 New CUDA Graph variant

```rust
enum DecodeGraphKind {
    Decode,
    MtpDecodeOne,
    MtpVerifyOne,
    MtpVerifyMulti { drafts, assume_accept, batched_lm_head },
    MtpVerifyTree { chain_depth: usize, leaf_count: usize, batched_lm_head: bool },  // NEW
}
```

Capture sequence (mirrors `ensure_mtp_verify_graph_multi_tokens` but threads the tree bitmap and adds the per-leaf DeltaNet replays):
1. Set tree-mask bitmap on `cuda_forward()?.tree_ancestor_bitmap_u64`.
2. Chain chunk: `prefill_cuda_chunk(chain_depth + 1, ..., tree_mask=NULL)` — chain rows only.
3. Snapshot DeltaNet/conv state.
4. For each leaf j: restore + `prefill_cuda_chunk(1, ..., tree_mask=…)` for the leaf row. Snapshot per-leaf state.
5. Run batched lm_head + sample_rows for verify[0..=chain_depth] AND verify[chain_depth+1..chain_depth+K+1].
6. Run MTP head chunk for next-cycle drafts.

### 6.4 Tree ancestor bitmap (host-built once per cycle)

```rust
fn build_tree_bitmap(chain_depth: usize, leaf_count: usize) -> [u64; 13] {
    let mut rows = [0u64; 13];
    for i in 0..=chain_depth {
        rows[i] = (1u64 << (i + 1)) - 1;          // causal triangle for chain
    }
    let chain_mask = (1u64 << (chain_depth + 1)) - 1;
    for j in 0..leaf_count {
        let row = chain_depth + 1 + j;
        rows[row] = chain_mask | (1u64 << row);   // leaf attends to chain prefix + itself
    }
    rows
}
```

H2D copy of 104 bytes once per cycle.

## 7. CLI / config

`MtpConfig` already has `tree_leaves: usize` (default 1). Add the CLI flag and dispatch.

```bash
qwen36 chat --mtp-speculative-tokens 3 --mtp-tree-leaves 4 ...
qwen36 bench --mtp-speculative-tokens 3 --mtp-tree-leaves 4 ...
```

Bench reports both `accepted_chain / chain_depth` and `leaf_accepted_rate` to make the tree gain measurable.

## 8. Testing strategy

1. **Pure-Rust unit tests** for v3 `walk_tree_acceptance` (extended for the +2 case) and `build_tree_bitmap`.
2. **CUDA smoke** for tree-mask attention: hand-built 4-row bitmap, compare against CPU reference.
3. **CUDA smoke** for top-K kernel (already shipped).
4. **Engine-level parity** (the gate): `chat` parity matrix for `--mtp-tree-leaves` ∈ {1, 2, 4} × `--mtp-speculative-tokens` ∈ {0..4}.
5. **Op-level parity** via `scripts/decode_parity.py` extended with `apply_tree_mask(scores, bitmap, chunk_base)`.
6. **Bench** for K ∈ {1, 2, 4, 8} on varied prompts; pick K (or default to 1 if no K beats MTP=4 by ≥5%).

## 9. Parity harness

`scripts/decode_parity.py` extended:
- New env vars `QWEN36_PARITY_TREE_DEPTH`, `QWEN36_PARITY_TREE_LEAVES`.
- Python reference builds the same ancestor bitmap and applies it to its causal-attention scores.
- Asserts cos sim ≥ 0.998 on each verify-row hidden state.

## 10. Rollback / kill-switch

`QWEN36_MTP_TREE_DISABLE=1` → forces `tree_leaves = 1`, falls back to today's chain MTP.

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
|--|--|--|
| Tree-mask attention introduces drift > current chunked-verify envelope | Medium | Op-level parity gate (cos sim ≥ 0.998), hard gate on `hello` / `hello world` |
| **Per-leaf DeltaNet replay overhead eats the leaf-accept gain** | **High** | **Measurement gate at P1.L: if K=4 doesn't beat K=1 by ≥5%, default to K=1 and ship infra only.** |
| K leaves × 48 DeltaNet layer launches × WSL2 latency dominates cycle time | Medium-High | If P1.L shows this, schedule batched DeltaNet kernel as Phase 2 |
| KV scratch + tree bitmap allocation grows VRAM | Low | <1 MB per cycle, trivial on 32 GB |
| CUDA Graph capture cost grows (more nodes in tree-verify graph) | Low | Re-use captured graph across cycles; capture cost paid once |
| Real-world acceptance gain < +5% target | Medium | Document negative result honestly; ship as Phase 2 prep |

## 12. Phase 2 entry points (preserved)

- `qwen36_deltanet_decode_batched` kernel that fans K (state, output) pairs from one input state in a single launch.
- Multi-level tree (branching at intermediate positions). Bitmap representation already supports arbitrary trees up to 64 rows.
- `MtpVerifyTree` graph kind extends to multi-level via the same bitmap.
- Adaptive tree-shape generator (Phase 3) calls into the same kernels.

## 13. Implementation phases (within Phase 1)

**Already landed (commits `2b6535d` → `07bb12a`):**
- P1.A — Top-K argmax kernel + Rust wrapper + smoke + tests
- P1.B — `walk_tree_acceptance` + `TreeDraft` / `TreeVerifyResult` types + unit tests (v2 invariant — needs P1.F update)
- P1.C — `leaf_tokens_u32` GPU buffer + `queue_sample_topk_into` engine helper

**Remaining:**
- P1.D — Tree-mask attention prefill kernel (3 variants) + ABI + smoke + parity check
- P1.E — Bump `MtpKvSnapshotLayout::VERIFY_TOKENS` to 13 + tree ancestor bitmap GPU buffer + KV scratch slots per leaf
- P1.F — Update `walk_tree_acceptance` for v3 invariants (verified-after-leaf in committed)
- P1.G — `verify_mtp_tree_draft` host-launched (no graph) + per-leaf DeltaNet/conv replay + acceptance walk wiring
- P1.H — `MtpVerifyTree` graph capture + soft fallback to host launch
- P1.I — `--mtp-tree-leaves` CLI flag + dispatch in chat/bench
- P1.J — Hard parity gate matrix (chat output equality)
- P1.K — Op-level parity via `decode_parity.py` extension
- P1.L — Bench matrix on RTX 5090 + AGENT.md update + K decision

## 14. Success criteria

- All hard parity gates pass (P1.J).
- Op-level parity ≥ 0.998 cos sim (P1.K).
- Bench shows ≥ +5% decode tok/s for `--mtp-tree-leaves 4` over `--mtp-tree-leaves 1` on varied prompts (or honest negative documented in AGENT.md).
- `QWEN36_MTP_TREE_DISABLE=1` recovers exact pre-Phase-1 behaviour bit-for-bit.
- `cargo clippy --workspace --features qwen36-fp4-kernels/cuda -- -D warnings` and the existing CUDA test suite stay green.

## 15. Phase 1 outcome (2026-05-04)

**Infrastructure: complete and parity-validated ✅**

All Phase 1 sub-tasks landed (P1.A → P1.L on `feat/perf-tree-mtp-stack`,
~22 commits ahead of `main`):

- P1.A — Top-K argmax CUDA kernel + Rust wrapper + smoke + tests.
- P1.B — `walk_tree_acceptance` v3 + `TreeDraft`/`TreeVerifyResult`
  types + 8 unit tests with shared `assert_committed_invariants`.
- P1.C — Leaf token GPU buffer + `queue_sample_topk_into` engine helper.
- P1.D — Tree-mask path in `attention_prefill_kernel` (ABI ext + kernel
  + smoke for hand-built 4-row bitmap).
- P1.E — `MtpKvSnapshotLayout::VERIFY_TOKENS` 5 → 13, tree ancestor
  bitmap GPU buffer.
- P1.F — `walk_tree_acceptance` v3 invariant fix (verified-after-leaf
  in committed; `committed.last() == next_token` always).
- P1.G — Per-leaf DeltaNet/conv snapshot buffers + 4 helpers,
  `verify_mtp_tree_draft` host-launched orchestrator.
- P1.I (revised α) — MTP head KV state advance + next-cycle pre-compute
  (pre-cycle drafts via `generate_mtp_drafts_from_committed_prefill`,
  leaf hidden-state D2D copy from `forward.normed` to
  `prefill.normed[chain_depth+1]` before MTP advance).
- P1.J — Hard parity gate ✅ (chat hello / hello world × MTP {0..4}
  identical token streams).
- P1.K — Skipped (parity validated end-to-end via P1.J).

**Parity gate: ✅ PASS** for K ∈ {1, 2, 4} on `hello` / `hello world`,
chain MTP=3 baseline.

**Performance: NEGATIVE result.** Tree-MTP K>1 dispatch is dramatically
slower than chain MTP on this hardware:

| MTP | K | tok/s |
|--|--|--|
| 3 | 1 | 110 (chain fallback) |
| 3 | 2 | 41 |
| 3 | 4 | 27 |
| 4 | 1 | 123 (chain fallback) |
| 4 | 2 | 49 |

**Root cause** (matches the analysis dismissed as "option β" in the
P1.G design discussion): each leaf is processed via a single-token
`forward_token_cuda` (~25 ms full 64-layer forward in WSL2). K=2 adds
~50 ms / cycle, K=4 adds ~100 ms / cycle, swamping the chain MTP cycle
time of ~10 ms.

**leaf_accept_rate = 0** on the synthetic bench prompt ("x" repeated
128 tokens) because MTP head's top-K disagrees with the base model's
argmax for the next position. Real prompts would show non-zero leaf
accept, but the per-cycle overhead would still swamp any gain at this
architecture.

**Path to make tree-MTP profitable** — Phase 2 work:

1. **Batched leaf forward** in a single chunk pass. Use the tree-mask
   `attention_prefill_kernel` (already implemented in P1.D) inside a
   custom `prefill_cuda_chunk_tree` variant that processes chain + all
   K leaves as one (1 + chain + K)-row chunk. For full-attention layers
   the tree mask handles per-leaf isolation. For DeltaNet layers, add a
   batched `qwen36_deltanet_decode_tree` kernel that fans K (state,
   output) pairs from one input state in a single launch.
2. With (1), leaf cost collapses from K × 25 ms to ~1 × 30 ms (one
   chunk pass for all K leaves, ~3 ms / leaf). At K=4 that's ~10 ms
   extra per cycle vs the +1 token gain, finally net positive.
3. The current `verify_mtp_tree_draft` α implementation can be dropped
   in favour of the proper batched path. The MTP head KV advance logic
   (next_chain_drafts / next_leaf_drafts pre-compute) carries over.

**Recommendation:** ship Phase 1 to `main` as infrastructure, kill the
tree dispatch by leaving `--mtp-tree-leaves` defaulted to 1 (chain
fallback). Schedule Phase 2 as a separate work item once batched
DeltaNet kernel design is ready.

If user opts to pivot away from tree-MTP entirely: the top-K kernel +
walk_tree_acceptance v3 + leaf buffers / snapshots remain useful
infrastructure for any speculative-decoding extension, including
EAGLE-2-style multi-level trees. The wasted work is the
`verify_mtp_tree_draft` orchestrator (~200 lines in engine.rs) — easy
to rip out if needed.
