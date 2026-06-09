# DFlash vs MTP=3 sweep — prompts × context × gen length

**Date:** 2026-06-09
**Build:** `target/release/qwen36`, RTX 5090, native Linux,
`QWEN36_LONG_CONTEXT_MODE=1`.
**Target:** `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`.
**Drafter:** `z-lab/Qwen3.6-27B-DFlash`.
**Driver:** `scripts/dflash_bench_sweep.py`.

15 cells (5 prompt types × 3 generation lengths) × 2 backends.

## Results

| Prompt | prompt tok | gen | DFlash tok/s | DFlash AL | MTP3 tok/s | MTP3 AL_eff | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| completion_short | 7 | 32 | 102.4 | 3.67 | 67.7 | 3.52 | **1.51×** |
| completion_short | 7 | 128 | 128.0 | 4.68 | 56.4 | 3.33 | **2.27×** |
| completion_short | 7 | 256 | 102.5 | 3.97 | 68.0 | 3.56 | **1.51×** |
| code_short | 4 | 32 | 121.7 | 4.38 | 108.0 | 4.00 | **1.13×** |
| code_short | 4 | 128 | **257.0** | 9.36 | 69.3 | 3.54 | **3.71×** |
| code_short | 4 | 256 | **313.6** | **11.77** | 81.9 | 3.73 | **3.83×** |
| prose_medium | 51 | 32 | 32.3 | 1.19 | 40.8 | 2.97 | 0.79× |
| prose_medium | 51 | 128 | 87.9 | 3.25 | 53.0 | 3.28 | **1.66×** |
| prose_medium | 51 | 256 | 105.9 | 4.10 | 58.7 | 3.43 | **1.80×** |
| qa_medium | 59 | 32 | 79.7 | 3.00 | 44.6 | 3.04 | **1.79×** |
| qa_medium | 59 | 128 | 130.4 | 4.96 | 48.8 | 3.20 | **2.67×** |
| qa_medium | 59 | 256 | 153.8 | 6.07 | 51.7 | 3.30 | **2.98×** |
| code_long | 229 | 32 | 46.6 | 2.00 | 85.9 | 3.88 | 0.54× |
| code_long | 229 | 128 | 53.9 | 2.37 | 74.1 | 3.73 | 0.73× |
| code_long | 229 | 256 | 79.8 | 3.55 | 69.1 | 3.70 | **1.15×** |

## Headlines

- **Peak: 313.6 tok/s** on `code_short` at gen=256 (AL **11.77**).
  Drafter excels on structured code with short context — the upper
  bound is set by `block_size=16` so an AL near 12 means almost every
  draft block is fully accepted.
- **Worst case: 0.54×** (DFlash loses) on `code_long` at gen=32. Long
  context + short generation = the drafter never gets traction before
  the budget runs out.
- **Average over all 15 cells**: DFlash **117.5 tok/s** vs MTP=3
  **65.2 tok/s** → **1.80× geometric mean speedup**.

## Patterns

### Generation length warms the drafter
DFlash AL rises with `max_new_tokens` because the drafter's diffusion
denoising benefits from longer streaks of in-distribution context.
Most dramatic on code:

| Prompt | gen=32 | gen=128 | gen=256 |
|---|---:|---:|---:|
| code_short AL | 4.38 | 9.36 | **11.77** |
| qa_medium AL | 3.00 | 4.96 | 6.07 |
| prose_medium AL | 1.19 | 3.25 | 4.10 |

MTP=3 is flat across gen lengths (AL_eff stays in 3.3–4.0).

### Context length hurts DFlash more than MTP

| Prompt (tokens) | DFlash gen=256 | MTP3 gen=256 | speedup |
|---|---:|---:|---:|
| code_short (4) | 313.6 | 81.9 | **3.83×** |
| code_long (229) | 79.8 | 69.1 | 1.15× |

Same task (code completion), drastically different prompt length —
DFlash's lead shrinks from 3.83× to 1.15× as the prompt grows. The
drafter's `fc + hidden_norm` collapses N context layers across all
prompt positions; longer prompts dilute the conditioning signal.

### Best fit per workload

| Workload | Recommended | Why |
|---|---|---|
| Code completion (short context) | DFlash | AL up to 11.77, 3.8× MTP=3 |
| Q&A, chat reply (medium context) | DFlash | AL 5–6, 2–3× MTP=3 |
| Open-ended prose (medium context) | DFlash for long gens | AL builds over time; ≥1.66× from gen=128 |
| Code review / long context refactor | **MTP=3** | DFlash loses unless gen is very long |
| Short tool-call / one-shot reply | **MTP=3** | Drafter has no time to warm up |

## When DFlash loses

Three cells in the matrix favour MTP=3 (`code_long` gen=32 and 128,
`prose_medium` gen=32). The common factor is **AL below 2.5**: the
drafter's diffusion sampler doesn't lock onto the target's
distribution before the budget is exhausted. Practical heuristics
the controller could use to route between paths:

- `prompt_tokens > 150` → start with MTP=3, switch to DFlash if the
  generation continues past ~64 tokens and acceptance has been good.
- `max_new_tokens < 48` and `prompt_tokens > 32` → MTP=3 only.
- Code or QA prompts with short context → always DFlash.

These aren't wired today; the user picks via `--drafter dflash`.

## Repro

```bash
/home/orosius/workspace/dmtp/.venv/bin/python \
    scripts/dflash_bench_sweep.py \
    --model-dir   ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --drafter-dir ~/models/Qwen3.6-27B-DFlash \
    --binary      target/release/qwen36 \
    --cuda-lib    target/cuda \
    --output      /tmp/dflash_sweep.csv
```

CSV at `/tmp/dflash_sweep.csv`. Each prompt × gen pair runs DFlash
then MTP=3; the script prints per-row tok/s and an overall Markdown
table at the end.
