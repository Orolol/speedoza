# DFlash speculative decoding for Qwen3.6-27B

**Date:** 2026-06-08
**Status:** Design — pending user approval
**Depends on:** main HEAD `9cc92fc`; existing `crates/mtp` controller
**Estimated effort:** 5–7 weeks; the longest single risk is drafter training
**Related:** `2026-06-08-interpreter-megakernel-design.md` (orthogonal,
multiplicative)

## 1. Goal and motivation

Replace (or stack on top of) the current MTP-head chain speculative
decoder with **DFlash**: a block diffusion drafter that emits *k*
candidate tokens in a single forward pass by denoising a masked
sequence conditioned on the target model's hidden states.

The DFlash paper ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036))
reports up to **6× lossless acceleration** on Qwen3-class targets and
**2.5× over EAGLE-3**. Xiaomi's MiMo team uses this technique to claim
1000+ TPS in their FP4-DFlash variant
([HF model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash)).
vLLM has a reference implementation
([docs](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)).

### Why DFlash specifically, not EAGLE-3 or tree-MTP

The current `crates/mtp` controller runs **chain MTP** with `num_speculative_
tokens = 3`, taking us from 55.27 → 110.78 tok/s (×2.0). Tree-MTP Phase 1
already shipped and bench-regressed vs chain MTP on this hardware (per
`AGENT.md` 2026-05-04: per-leaf forward overhead dominates K>1 at
batch=1). So the question is what else can push past chain MTP×2.

| Drafter family | Drafts/pass | Verification cost | Reported lossless speedup | Drafter training cost |
|---|---|---|---|---|
| Chain MTP (today) | 1 (autoregressive across k passes) | k forwards | ~2× (measured here) | none — built into model |
| Tree-MTP K>1 | K branches × k depth | K×k forwards | regressed on RTX 5090 | none |
| EAGLE-3 | 1 per pass × k passes | k forwards | 3–6× vs no-spec | small head, ~$300 / Qwen3-class |
| **DFlash** | **k in one pass (block diffusion)** | **1 verify pass** | **6× vs no-spec, 2.5× over EAGLE-3** | diffusion drafter, ~$1–3K / 6B-scale drafter |

DFlash is the only family that **eliminates serial drafter passes** for
the cost of one extra verify-shape forward — a tradeoff the existing MTP
verify path already pays once. It is multiplicative with the megakernel
work because it changes *how many tokens move per host iteration*, not
how each token is computed.

### Bench projection

Reference numbers (DFlash paper, normalised to "no-spec baseline"):

| Workload | DFlash multiplier on no-spec | Realistic on Qwen3.6-27B / RTX 5090 |
|---|---:|---:|
| Coding (high acceptance, ~6.3 tokens/block) | 6.0× | ~4.5× |
| Math/reasoning (~5.6) | 5.5× | ~4.0× |
| Chat (~4.3) | 4.5× | ~3.0× |

Applied to the 55.27 tok/s MTP=0 baseline:

| Workload | Today (chain MTP=4) | DFlash projection |
|---|---:|---:|
| Coding | 130–150 | **220–250 tok/s** |
| Reasoning | 110–130 | **180–220 tok/s** |
| Chat (`hello`, `hello world` benches) | 100–115 | **140–170 tok/s** |

The chat number is the closest to the existing bench harness, so the
Phase 4 gate below uses it.

## 2. Quality contract

DFlash is **lossless** by construction (the target model verifies and
rejects mismatching block proposals), but only if the verify pass is
implemented faithfully — block masking + per-position rejection.

**Hard parity (regression gate, blocks merge):**
- `chat --prompt "hello" --max-new-tokens 32` with DFlash enabled
  produces **identical token sequences** to `chat ... --mtp-speculative-
  tokens 0` (the no-spec ground truth), for the same seed and decoding
  config (`greedy=true`).
- Same for `chat --prompt "hello world" --max-new-tokens 32`.
- Same parity contract used since `AGENT.md` 2026-05-04; DFlash must
  not weaken it.

**Soft parity:** acceptance rate per workload measured against the
DFlash paper. Coding ≥ 5.0 tokens/block, chat ≥ 3.5 tokens/block. If
below, the drafter quality is the regression to chase, not the
runtime.

**Op-level parity:** the **block-masked verify forward** is a new
attention shape (`N = k` queries against `seq_len + k` keys with a
specific causal mask). Add it to the parity harness via
`scripts/decode_parity.py` extension — a synthetic block of k known
tokens must dump the same per-layer activations as k sequential decode
passes through the existing path.

**Soft fallback:** `QWEN36_DRAFTER=mtp_chain|mtp_tree|dflash|none`
selects the drafter at runtime. Engine falls back to `mtp_chain` if the
DFlash drafter file is missing or fails to load.

## 3. Hardware and model constraints

- RTX 5090, 32 GB HBM3. Target weights are 27 B × ~0.6 B / param NVFP4 +
  scales ≈ 17 GB. Drafter must fit in the remaining ~12 GB after KV
  cache reservation.
- Drafter size budget: **≤ 1 B params NVFP4**. Even smaller is better —
  the diffusion drafter runs once per accepted block, but its size adds
  to the verify-pass tail.
- Target hidden size: 5120. Drafter must consume hidden states from
  multiple target layers (DFlash recipe: typically last 3–6 layers).
- Block size *k*: paper uses 8 for plain DFlash, optimal in our regime
  is TBD. Start at 8, sweep {4, 6, 8, 12, 16} in Stage 3.
- The existing tree-MTP verify path already supports multi-position
  forwards — DFlash's verify shape is a strict subset of tree verify
  (linear block, not tree).

### What Xiaomi's MiMo-V2.5-Pro-FP4-DFlash does NOT give us

It is a drafter trained against the **MiMo target hidden states**, not
Qwen3.6 hidden states. We cannot just download and plug it in. We need
a **Qwen3.6-conditioned DFlash drafter**. This is the long-pole risk.

Options for getting that drafter:

1. **Train one ourselves** following the paper recipe. ~$1–3K of A100
   time + 1–2 weeks of person-time per the paper's published recipe.
   Highest control.
2. **Wait for an open release.** No known Qwen3.6-conditioned DFlash
   drafter exists as of 2026-06-08. Banks on community publication.
3. **Adapt EAGLE-3 as a stepping stone.** EAGLE-3 drafters for Qwen3-class
   models exist on HF; speedup is ~3× rather than ~5×, but bring-up is
   immediate and the verify path is identical. Useful for de-risking
   everything *except* the diffusion drafter itself.

Recommended sequence: **(3) first** to land the verify-block
infrastructure with a real (smaller) acceptance lift, **(1) after** if
the EAGLE-3 result justifies the drafter training spend.

## 4. Existing baseline and integration points

| Code site | Today | DFlash impact |
|---|---|---|
| `crates/mtp/src/lib.rs::SpeculativeDecoder::step` | Chain: 1 main+MTP forward, then k verify forwards | Replaced: 1 drafter forward (emits k tokens) + 1 verify forward |
| `crates/mtp/src/lib.rs::MtpRuntime` trait | `forward_main_and_mtp`, `forward_main_only` | Add `forward_drafter_block(k, target_hidden_states)` and `forward_verify_block(tokens)` |
| `crates/runtime/src/engine.rs` MTP head call | Reads MTP-head logits from main forward | DFlash drafter has its own weights and forward; main forward exposes only hidden states |
| `kernels-cuda/attention.cu` decode kernel | One query per call | Verify kernel needs to support k-query attention with block causal mask |
| Existing tree-MTP verify path (`AGENT.md` 2026-05-04) | Multi-position verify shape exists | Reuse, restrict mask to block causal |
| Parity harness `scripts/decode_parity.py` | Per-token decode dump | Add per-block decode dump |

The diffusion drafter itself is a **new model** with its own weights,
config, and forward path. It needs its own crate or module:
`crates/drafter/dflash/`. Loading uses the existing `loader` crate's
safetensors machinery.

## 5. Architecture

### 5.1 Drafter forward shape

Per the DFlash paper, the drafter is a small transformer (typically 6–12
layers) trained as a **discrete diffusion model**:

```
INPUTS:
  prefix_tokens   : [T]                  # already-committed sequence
  target_hiddens  : [T, L_subset, D]     # hidden states from a few target layers
  mask_block      : [k]  initialised to <MASK> token id
OUTPUTS:
  drafted_tokens  : [k]                  # one denoising pass produces all k
```

Hidden-state conditioning is injected into the drafter's KV projections
(paper §3.2). We already dump intermediate hidden states for parity
checks (`QWEN36_DEBUG_DUMP_DIR`), so the plumbing exists — but it needs
to become zero-copy + GPU-resident, not a debug-disk write.

### 5.2 Verify forward shape

The target model runs **one forward** on `[prefix_tokens ++ drafted_block]`
with:
- Causal mask: standard causal up to the prefix; block-causal across
  the k drafted positions (token *i* sees tokens 0..prefix_len+i-1, but
  not the other drafted positions — same shape as standard causal).
- KV writes: provisional. The committed prefix length advances by the
  number of accepted tokens; uncommitted KV positions are discarded by
  the existing rollback mechanism (`MtpRuntime::snapshot_recurrent_state`
  / `restore_recurrent_state`).

Rejection rule (greedy lossless):
- For each *i* in 0..k, compare `argmax(target_logits[i])` against
  `drafted_tokens[i]`. Accept the longest prefix that matches; commit
  it; discard the rest.

This is **identical** to chain MTP rejection, just done across one
batch of k positions instead of k sequential calls.

### 5.3 KV cache and DeltaNet recurrence handling

The hard part of any drafter on this model is the **hybrid topology**:
48 linear-attention (DeltaNet) layers + 16 full-attention layers. The
DeltaNet recurrence is stateful — verifying a block requires that the
recurrence either:

(a) runs k steps forward at verify time, then is rolled back on rejection
    via the existing snapshot/restore controller, **or**
(b) runs in parallel via a chunked-recurrence kernel that can compute
    k steps' outputs from one initial state in a single pass.

Option (a) is what the existing MTP path already does. **DFlash inherits
this for free**; the verify pass calls `forward_main_only` k times,
which steps the DeltaNet recurrence k times, and rolls back on
rejection. The block-causal full-attention is implemented in the
attention kernel, not the recurrence.

This means **the verify path can be implemented exactly like the existing
chain-MTP verify path**, just iterated over k drafted tokens in one
host iteration. The "one verify forward" claim in the DFlash paper is
specific to pure-attention models; on a hybrid model we still pay k
recurrent steps, but they're cheap. Bench projection in §1 already
accounts for this.

### 5.4 File / module layout

```
crates/
  drafter/                          <- new workspace member
    dflash/
      src/
        lib.rs                      <- DFlash drafter struct + forward
        diffusion.rs                <- masked denoising loop
        condition.rs                <- hidden-state injection
        config.rs                   <- drafter topology
    Cargo.toml
  mtp/src/
    lib.rs                          <- extend MtpRuntime trait
    dflash_controller.rs            <- new: DFlash-specific step() loop
  runtime/src/
    drafter_handoff.rs              <- expose target hidden states to drafter
    engine.rs                       <- wire QWEN36_DRAFTER env var

kernels-cuda/
  attention.cu                      <- extend with block-verify kernel variant
  drafter_diffusion/                <- new directory
    dflash_forward_sm120.cu         <- drafter-specific NVFP4 GEMV variants

docs/superpowers/specs/
  2026-06-08-dflash-speculative-decoding-design.md   <- this file

scripts/
  train_dflash_drafter.py           <- recipe wrapper (one-shot training)
  dflash_acceptance_bench.py        <- per-workload acceptance rate measurement
```

ABI sync rule (`AGENT.md`) extends to drafter weights — manifest
validation in `crates/loader` adds a `dflash_drafter.safetensors`
expected file when `QWEN36_DRAFTER=dflash`.

## 6. Phased delivery

### Stage 0 — Verify-block infrastructure with EAGLE-3 drafter (1.5 weeks)

De-risk the **block verify pass** before committing to DFlash drafter
training. EAGLE-3 drafters for Qwen3-class models exist on HF; bring one
up against Qwen3.6 (may not be a perfect fit — accept lower acceptance
rate) and:

- Land `MtpRuntime::forward_verify_block(tokens, k)` calling the
  existing attention path k times with appropriate residual + KV writes
  and the rollback controller.
- Land a verify-block parity smoke: k sequential decode steps and one
  block verify of the same k known tokens produce bit-identical
  per-layer activations.
- Wire `QWEN36_DRAFTER=eagle3` end-to-end.
- Gate: parity green; **EAGLE-3 chat speedup ≥ 1.3× over chain MTP=4**.
  If even this fails, the verify-block infrastructure itself is broken
  and DFlash cannot land. Stop here; debug the verify path.

### Stage 1 — DFlash drafter training (2 weeks, runs in parallel with Stage 0)

- Train a Qwen3.6-conditioned DFlash drafter on the recipe from arXiv
  2602.06036, using rented A100 or H100 time.
- Drafter size: 0.5–1 B params, layers conditioned on Qwen3.6 hidden
  states from layers {16, 32, 48, 63}.
- Output: `dflash_drafter.safetensors` + a config describing layer
  conditioning, k=8.
- Gate: training loss converges, drafter standalone accepts ≥ 4.0
  tokens/block on a held-out chat set against Qwen3.6 (measured in
  PyTorch, before kernel integration).

### Stage 2 — DFlash forward in CUDA (1 week)

- Implement diffusion drafter forward: `dflash_forward_sm120.cu`. The
  drafter is small; standalone NVFP4 GEMV bodies (same as decode-gemv)
  + one masked self-attention pass. No new attention algorithm needed.
- Hidden-state handoff: zero-copy expose `target.hidden[layer_idx]` to
  the drafter via existing buffer-sharing primitives.
- Op-level parity: drafter forward in CUDA matches PyTorch reference
  cos sim ≥ 0.998.
- Gate: parity green.

### Stage 3 — End-to-end DFlash + block sweep (1 week)

- Replace EAGLE-3 with DFlash drafter under `QWEN36_DRAFTER=dflash`.
- Sweep block size *k* ∈ {4, 6, 8, 12, 16} on the chat parity prompts.
- Measure acceptance rate per workload.
- Gate: parity green; **chat speedup ≥ 2.5× over chain MTP=4** (110.78
  → ~280 tok/s) at the optimal k.
  - If gain is **< 1.5×**, treat as a drafter quality regression
    (Stage 1 recipe likely diverged from paper) — debug the drafter,
    not the runtime.
  - If gain is **1.5–2.5×**, ship as opt-in but keep chain MTP as
    default until the gap is understood.

### Stage 4 — Default-on flip (after a soak week)

- Soak: 24 hours of continuous decode across coding, reasoning, chat
  prompts. No parity drift; acceptance rate stable per workload.
- Flip `QWEN36_DRAFTER=dflash` as default.
- Update `AGENT.md` with bench tables per workload.

## 7. Risks and exit ramps

| Risk | Severity | Exit ramp |
|---|---|---|
| Qwen3.6-conditioned DFlash drafter training diverges or underperforms paper | **High** | Ship the EAGLE-3 path from Stage 0 as the new default drafter. Smaller speedup (1.5×) but real and avoids drafter training. |
| Hidden-state handoff from target to drafter adds enough latency to eat the win | Medium | Cache hidden states in a ring buffer; the drafter reads layer outputs after they're written but before they're freed. Same trick as parity-dump zero-copy. |
| Block verify changes the existing borderline 1–2 token drift envelope (1.5 % of prompts in `AGENT.md` checks) | Medium | The drift is from accumulating BF16/FP4 noise across a forward, not from the verify itself. Block verify should not widen it; if it does, narrow `k`. |
| Drafter size leaves no room for max-context KV cache (262144 tokens) | Medium | DFlash needs only the chat-relevant context; long context can opt out via `QWEN36_DRAFTER=mtp_chain` automatic fallback when reserved context > threshold. |
| MTP head weight is left allocated alongside DFlash drafter, wasting VRAM | Low | `QWEN36_DRAFTER=dflash` skips MTP head load. Manifest validation in `loader` makes this explicit. |
| Tree-DDTree variant (256–512 token tree budget) lures us into another tree-MTP-like regression | Medium | Keep DDTree explicitly out of scope (§8). The chain-DFlash result must land before any tree variant is even prototyped. |
| Sampling is non-greedy (future work) and DFlash rejection rule becomes stochastic | Low | Today the engine is greedy-only (`MtpConfig::greedy = true`). When non-greedy lands, port the speculative sampling rule from the DFlash paper §3.4. Out of scope here. |

## 8. Out of scope

- **DDTree.** Tree-based DFlash variant. The chain DFlash result has to
  land first.
- **Non-greedy sampling rule.** Engine is greedy-only today. The
  speculative sampling rule for non-greedy DFlash is a follow-up.
- **Multi-batch decode.** Same as the interpreter megakernel spec — N=1
  only.
- **Cross-drafter ensembles** (EAGLE-3 + DFlash). Pick one drafter per
  decode call; no mixing.
- **Long-context drafter** (sliding-window draft attention from the
  paper §4.2). Adds complexity for the < 8K-token chat regime where
  most usage sits.

## 9. Cost summary

| Line item | Cost |
|---|---|
| Drafter training (Stage 1) | $1–3K of A100/H100 time |
| EAGLE-3 drafter (Stage 0 bring-up) | $0 — open weights on HF |
| Engineering | 5–7 weeks (single person), Stages 0/1 partially parallel |
| Inference time impact | net negative (faster decoding); +1 GPU model load (~700 MB drafter weights) |

## 10. References

- DFlash paper — *DFlash: Block Diffusion for Flash Speculative
  Decoding*, arXiv 2602.06036.
  <https://arxiv.org/abs/2602.06036>
- DFlash + DDTree deep-dive (third-party explanation).
  <https://maloyan.xyz/blog/dflash-ddtree-block-diffusion-speculative-decoding>
- vLLM Speculators DFlash docs (reference implementation).
  <https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/>
- Xiaomi MiMo-V2.5-Pro-FP4-DFlash (an existing FP4 DFlash drafter, but
  conditioned on MiMo not Qwen3.6 hidden states).
  <https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash>
- EAGLE-3 — *Scaling up Inference Acceleration via Training-Time Test*,
  arXiv 2503.01840 (Stage 0 bring-up drafter).
  <https://arxiv.org/abs/2503.01840>
- `crates/mtp/src/lib.rs` — existing speculative controller to extend.
- `AGENT.md` §"2026-05-04 — Tree-MTP Phase 1" — empirical reason DFlash
  is preferred over tree-MTP K>1 on this hardware.
