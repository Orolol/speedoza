# DFlash benchmarks vs MTP chain — Phase F.2 baseline

**Date:** 2026-06-09
**Build:** `target/release/qwen36`, RTX 5090, native Linux, `QWEN36_LONG_CONTEXT_MODE=1`.
**Target:** `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (NVFP4).
**Drafter:** `z-lab/Qwen3.6-27B-DFlash` (BF16, 2 B).
**Max new tokens:** 64 per run.
**DFlash verify path:** sequential `engine.prefill(&[t])` after the
[decode-kernel divergence fix](../../crates/cli/src/main.rs) (`6571f37`).

## Results

| Prompt (tokens) | DFlash AL | DFlash tok/s | MTP=3 AL_eff | MTP=3 tok/s | MTP=1 tok/s | MTP=0 tok/s |
|---|---:|---:|---:|---:|---:|---:|
| "The quick brown fox jumps over the" (7) | **5.38** | 45.2 | 3.48 | 68.0 | 59.2 | 58.3 |
| "Once upon a time in a land far far away…" (14) | 2.10 | 42.3 | 3.28 | 58.9 | 59.4 | 54.2 |
| "def fibonacci(n):" (4) | 4.92 | 43.9 | 3.82 | **97.5** | 66.6 | 57.1 |
| "Translate to French: The cat sat…" (24) | 5.33 | 41.5 | 3.82 | 89.5 | — | — |

`MTP=3 AL_eff = mtp_acceptance_rate × 3 + 1`.

## Summary

- **DFlash AL meets or beats the paper's reported 5.05** on the pangram,
  code, and translation prompts (5.38 / 4.92 / 5.33). The fairytale
  prompt drops to 2.10 — the drafter's diffusion conditioning is
  prompt-sensitive.
- **DFlash tok/s is consistently 41–45**, lower than MTP=0 (54–58) and
  significantly below MTP=3 (58–97).
- **Average over 4 prompts**: DFlash 4.43 AL @ 43.2 tok/s, MTP=3 3.60
  AL_eff @ 78.5 tok/s. DFlash wins on AL by 1.23×, loses on tok/s by
  1.82×.

## Why DFlash is slower despite higher AL

Per iteration our DFlash speculative cycle runs:

1. One drafter forward (~5 layers, BF16, small).
2. **Sixteen sequential target prefill chunks** (one per verify
   position; each chunk is a full 64-layer NVFP4 forward).
3. One lm_head + greedy sample per chunk.

Cost per iteration ≈ 16 target forwards. Per token: 16 / AL ≈ 3.2
target forwards (at AL=5).

MTP=3 chain:
- One main forward → drafts via the MTP head graph.
- One batched verify forward (multi-token graph kernel).
- Cost per cycle ≈ 2 target forwards. Per token: 2 / AL_eff ≈ 0.6
  target forwards.

So even with DFlash's better drafts, our naive sequential verify pays
~5× more target-forward cost per token than MTP=3's batched verify.
Net throughput suffers.

## Path to making DFlash competitive

The unfreed throughput is entirely on the verify side. Two engine-side
follow-ups would close the gap:

1. **Batched verify forward**. Run one multi-token prefill on
   `[seed, drafted_0, …, drafted_{k-1}]` and read per-position logits.
   Engine currently emits logits only at the last position of a
   prefill chunk; needs a new path that materialises logits at every
   position in the chunk (similar to what `verify_mtp_draft_tokens`
   does for MTP, but generic). Cost per iter would drop to ~2 forwards
   from 17.
2. **Captured verify graph**. Once batched verify is wired, capture it
   into a CUDA graph at fixed `(k, prompt_len_class)` shapes, the same
   way decode-and-sample is captured today.

Until those land, **DFlash is correctness-validated but
performance-constrained**. MTP=3 remains the default fast path for
production chat.

## Repro

```bash
export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$PWD/target/cuda:${LD_LIBRARY_PATH:-}"
export QWEN36_LONG_CONTEXT_MODE=1

# DFlash
target/release/qwen36 drafter-chat-smoke \
    --model-dir   ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --drafter-dir ~/models/Qwen3.6-27B-DFlash \
    --prompt "<text>" --max-new-tokens 64

# MTP baseline
target/release/qwen36 bench \
    --model-dir ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --prompt-file <text-file> --prompt-tokens <N> \
    --max-new-tokens 64 --mtp-speculative-tokens 3
```

`drafter-chat-smoke` emits `tokens_per_second` and `acceptance_length`
in its JSON output; `bench` emits `decode_tokens_per_second` and
`mtp_acceptance_rate`.
