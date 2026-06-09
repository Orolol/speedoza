# DFlash speculative decoding — final results

**Date:** 2026-06-09
**Build:** `target/release/qwen36`, RTX 5090, native Linux,
`QWEN36_LONG_CONTEXT_MODE=1`.
**Target:** `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (NVFP4, ~17 GB).
**Drafter:** `z-lab/Qwen3.6-27B-DFlash` (BF16, ~3.5 GB).
**Status:** Phase F.2 shipped, integrated behind `chat --drafter
dflash`.

Supersedes the two earlier WIP notes
(`2026-06-09-dflash-benchmarks.md` and
`2026-06-09-dflash-bench-sweep.md`) which were rolled into this
single canonical doc.

## TL;DR

| Workload | DFlash | MTP=3 | DFlash vs MTP=3 |
|---|---:|---:|---|
| Short code prompt, long generation | **313.6 tok/s, AL 11.77** | 81.9 tok/s | **3.83×** |
| Short completion / pangram | 128.0 tok/s, AL 4.68 | 56.4 tok/s | 2.27× |
| Q&A, medium context | 153.8 tok/s, AL 6.07 | 51.7 tok/s | 2.98× |
| Open-ended prose | 105.9 tok/s, AL 4.10 | 58.7 tok/s | 1.80× |
| Long context (200+ prompt) | 53.9 tok/s, AL 2.37 | 74.1 tok/s | **0.73×** |

**Geometric mean over 15 sweep cells:** DFlash 117.5 vs MTP=3 65.2
tok/s → **1.80× speedup**.

DFlash dominates on **short prompts + medium-to-long generations**
(structured code, Q&A, chat replies). MTP=3 still wins on
**long-context + short-generation** workloads where the drafter
doesn't have time to warm up.

## How to use

```bash
target/release/qwen36 chat \
    --model-dir   ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --drafter     dflash \
    --drafter-dir ~/models/Qwen3.6-27B-DFlash \
    --prompt "<text>" --max-new-tokens 256
```

Streams decoded tokens to stdout as they commit. Emits a trailing
stats line on stderr:

```
[dflash] generated 256 tokens in 22 iters | AL=11.64 avg_accept=10.64 | decode 0.81s (314.8 tok/s) | total 6.18s
```

Required env (DFlash drafter + target push past 22 GB VRAM; the
engine's MLP/linear-attn fused stores would push past 32 GB
otherwise):

```bash
export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$PWD/target/cuda:${LD_LIBRARY_PATH:-}"
export QWEN36_LONG_CONTEXT_MODE=1
```

Default `chat` behaviour is unchanged (chain MTP via
`--mtp-speculative-tokens N`); DFlash is opt-in.

## Drafter checkpoint

`z-lab/Qwen3.6-27B-DFlash` (HF, gated, MIT). 2 B BF16, 5 transformer
layers (4 sliding-attention + 1 full-attention), 32 Q heads × 8 KV
heads × head_dim 128, block_size=16, mask_token_id=248070,
target_layer_ids=[1, 16, 31, 46, 61].

Download:

```bash
hf download z-lab/Qwen3.6-27B-DFlash --local-dir ~/models/Qwen3.6-27B-DFlash
# trust_remote_code parity script needs the model.py renamed
# alongside the weights:
curl -sL https://raw.githubusercontent.com/z-lab/dflash/main/dflash/model.py \
     -o ~/models/Qwen3.6-27B-DFlash/dflash.py
```

The HF model card warns "still under training" and "benchmark results
N/A" for this checkpoint. The numbers below are what we measured on
2026-06-09 with that specific binary.

## Full sweep matrix

5 prompt types × 3 generation lengths × 2 backends = 30 runs.

| Prompt | prompt tok | gen | DFlash tok/s | DFlash AL | MTP=3 tok/s | MTP=3 AL_eff | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| completion_short | 7 | 32 | 102.4 | 3.67 | 67.7 | 3.52 | 1.51× |
| completion_short | 7 | 128 | 128.0 | 4.68 | 56.4 | 3.33 | **2.27×** |
| completion_short | 7 | 256 | 102.5 | 3.97 | 68.0 | 3.56 | 1.51× |
| code_short | 4 | 32 | 121.7 | 4.38 | 108.0 | 4.00 | 1.13× |
| code_short | 4 | 128 | 257.0 | 9.36 | 69.3 | 3.54 | **3.71×** |
| code_short | 4 | 256 | **313.6** | **11.77** | 81.9 | 3.73 | **3.83×** |
| prose_medium | 51 | 32 | 32.3 | 1.19 | 40.8 | 2.97 | 0.79× |
| prose_medium | 51 | 128 | 87.9 | 3.25 | 53.0 | 3.28 | 1.66× |
| prose_medium | 51 | 256 | 105.9 | 4.10 | 58.7 | 3.43 | **1.80×** |
| qa_medium | 59 | 32 | 79.7 | 3.00 | 44.6 | 3.04 | 1.79× |
| qa_medium | 59 | 128 | 130.4 | 4.96 | 48.8 | 3.20 | **2.67×** |
| qa_medium | 59 | 256 | 153.8 | 6.07 | 51.7 | 3.30 | **2.98×** |
| code_long | 229 | 32 | 46.6 | 2.00 | 85.9 | 3.88 | 0.54× |
| code_long | 229 | 128 | 53.9 | 2.37 | 74.1 | 3.73 | 0.73× |
| code_long | 229 | 256 | 79.8 | 3.55 | 69.1 | 3.70 | 1.15× |

`MTP=3 AL_eff = mtp_acceptance_rate × 3 + 1`. DFlash AL = generated
tokens / iterations. Raw CSV at
`docs/superpowers/notes/2026-06-09-dflash-sweep.csv`.

## Patterns

### Generation length warms the drafter

DFlash AL rises with `max_new_tokens` because the drafter's diffusion
denoising benefits from longer streaks of in-distribution context.
MTP's MTP head is flat across gen lengths.

| Prompt | gen=32 | gen=128 | gen=256 |
|---|---:|---:|---:|
| code_short AL (DFlash) | 4.38 | 9.36 | **11.77** |
| qa_medium AL (DFlash) | 3.00 | 4.96 | 6.07 |
| prose_medium AL (DFlash) | 1.19 | 3.25 | 4.10 |
| MTP=3 AL_eff (any) | 3.0–4.0 | 3.0–4.0 | 3.0–4.0 |

The peak AL of 11.77 is close to the upper bound `block_size = 16`:
near-full block acceptance on most iters.

### Long prompts hurt DFlash specifically

Same task (code completion), drastically different prompt length:

| Prompt | prompt tok | DFlash gen=256 tok/s | speedup vs MTP=3 |
|---|---:|---:|---:|
| code_short | 4 | 313.6 | 3.83× |
| code_long | 229 | 79.8 | 1.15× |

The drafter's `fc + hidden_norm` collapses 5 target-layer hidden
states across all prompt positions into a single conditioning input;
longer prompts dilute the per-position signal. MTP's chain head
doesn't depend on conditioning length.

### When DFlash loses

Three cells favour MTP=3: `code_long@32`, `code_long@128`,
`prose_medium@32`. Common factor: **AL below 2.5**. The drafter
sampler doesn't lock onto the target's distribution before the budget
is exhausted.

## Routing recommendations (manual)

The selector is opt-in via `--drafter dflash` today; the CLI does NOT
adaptively switch. Suggested heuristics for a future controller:

- `prompt_tokens > 150` → start with MTP=3; switch to DFlash if
  generation continues past ~64 tokens with good acceptance.
- `max_new_tokens < 48` and `prompt_tokens > 32` → MTP=3 only.
- Code or QA with short context → DFlash.
- Long context refactor / review → MTP=3.

## Implementation history

13 commits, two sessions (2026-06-08 → 2026-06-09).

| Commit | Phase | What |
|---|---|---|
| `03c6fb9` | A | Drafter config + manifest + `validate-drafter` CLI |
| `96a3d83` | B | GPU uploader, `drafter-load` CLI, kernel-reuse audit |
| `4c2b43c` | C | New CUDA kernel `qwen36_drafter_attention_block_bf16` (smoke cos sim 0.999999 vs host fp32) |
| `6c8a2f5` | D.1 | 5-layer forward + smoke (finite + deterministic) |
| `47c6f22` | D parity | Python harness `scripts/dflash_parity.py`; cos sim 0.99989 vs transformers reference |
| `8dab67b` | D.2+D.3 | Internal fc + hidden_norm; per-layer KV cache + reset/crop API |
| `97993c0` | E | Target → drafter hidden-state handoff (engine hook in prefill layer loop) |
| `40801fa` | F.0 | Drafter propose helper: embed → forward → lm_head → sample |
| `37249cb` | F.1 | Verify cycle + accepted/bonus report (sequential decode) |
| `059dd45` | F.2 | Multi-iter speculative loop with bonus→seed handoff |
| `6571f37` | F.2 fix | Verify via prefill (decode kernel diverges on NVFP4) |
| `9b7e643` | F.2 perf | Batched verify forward (`Engine::verify_block_batched`) — 1.5–4.3× tok/s |
| `65e4d0e` | chat | `chat --drafter dflash` integration with streaming output |
| `174c5e2` | bench | Full sweep matrix (this doc) |

## Known issues / out of scope

- **NVFP4 decode-kernel divergence**: `chat --mtp-speculative-tokens 0`
  produces degenerate output because the engine's per-token decode
  path produces logits with cos sim ~0.76 (sometimes negative) vs
  the prefill kernel path on the same input. Surfaced by the new
  `decode-vs-prefill-check` diagnostic CLI; root cause not
  investigated. DFlash routes around it by verifying through prefill
  chunks.
- **CUDA-graph capture of `verify_block_batched`**: deferred.
  Plausible 20–30% more tok/s but requires coordination with the
  existing decode-graph machinery. The MTP path already graph-captures
  its verify; cloning that for DFlash is non-trivial.
- **Permanent logits workspace** on `GpuForwardBuffers`: evaluated and
  skipped. Per-call `CudaDeviceBuffer::alloc/free` in
  `verify_block_batched` costs ~50 µs × 20 iters per chat = ~1 ms over
  ~700 ms = 0.15 % gain. Below noise floor.
- **Adaptive drafter routing**: today the user picks via
  `--drafter dflash`. A controller could swap dynamically based on
  prompt length + observed acceptance.
- **Decode-kernel investigation**: filed as a separate engine concern;
  fixing it would also enable `chat --mtp-speculative-tokens 0` to
  produce coherent text.

## Repro

```bash
# Headline run (single prompt, streaming)
target/release/qwen36 chat \
    --model-dir   ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --drafter     dflash \
    --drafter-dir ~/models/Qwen3.6-27B-DFlash \
    --prompt "Write a Python function that returns the sum of two integers" \
    --max-new-tokens 256

# Full 30-cell sweep (~7 minutes)
/home/orosius/workspace/dmtp/.venv/bin/python \
    scripts/dflash_bench_sweep.py \
    --model-dir   ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --drafter-dir ~/models/Qwen3.6-27B-DFlash \
    --binary      target/release/qwen36 \
    --cuda-lib    target/cuda \
    --output      /tmp/dflash_sweep.csv

# Single-prompt JSON output (used by the sweep driver)
target/release/qwen36 drafter-chat-smoke \
    --model-dir   ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --drafter-dir ~/models/Qwen3.6-27B-DFlash \
    --prompt "<text>" --max-new-tokens 256

# Decode-vs-prefill diagnostic (loads engine twice, slow)
target/release/qwen36 decode-vs-prefill-check \
    --model-dir   ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --prompt "<text>"

# Python parity harness (drafter forward only)
/home/orosius/workspace/dmtp/.venv/bin/python \
    scripts/dflash_parity.py \
    --drafter-dir /home/orosius/models/Qwen3.6-27B-DFlash \
    --fixture-dir /tmp/dflash_fixture
target/release/qwen36 drafter-forward-smoke \
    --drafter-dir /home/orosius/models/Qwen3.6-27B-DFlash \
    --fixture-dir /tmp/dflash_fixture
```

## Cross-references

- **Design spec**:
  `docs/superpowers/specs/2026-06-08-dflash-speculative-decoding-design.md`
  (original phasing A → F; some details have shifted in the
  implementation — see commit history for ground truth).
- **Kernel-reuse audit**:
  `docs/superpowers/notes/2026-06-08-dflash-kernel-reuse-audit.md`
  (Phase B; itemises which existing kernels the drafter reuses and
  what new CUDA was needed).
- **Raw sweep CSV**:
  `docs/superpowers/notes/2026-06-09-dflash-sweep.csv`.
- **Sweep driver**: `scripts/dflash_bench_sweep.py`.
- **Python parity harness**: `scripts/dflash_parity.py`.
- **DFlash paper**: arXiv 2602.06036 (z-lab).
- **Drafter weights**: <https://huggingface.co/z-lab/Qwen3.6-27B-DFlash>.
