# DFlash benchmarks vs MTP chain on NVFP4

**Date:** 2026-06-09 (updated after batched verify landed)
**Build:** `target/release/qwen36`, RTX 5090, native Linux,
`QWEN36_LONG_CONTEXT_MODE=1`.
**Target:** `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (NVFP4, ~17 GB).
**Drafter:** `z-lab/Qwen3.6-27B-DFlash` (BF16, ~3.5 GB).
**Max new tokens:** 64 per run.

## Headline numbers (decode tok/s, after batched verify)

| Prompt (tokens) | DFlash AL | **DFlash tok/s** | MTP=3 AL_eff | MTP=3 tok/s | MTP=1 tok/s | MTP=0 tok/s |
|---|---:|---:|---:|---:|---:|---:|
| "The quick brown fox jumps over the" (7) | 4.53 | **128.4** | 3.48 | 68.0 | 59.2 | 58.3 |
| "Once upon a time in a land far far away…" (14) | 2.08 | 59.3 | 3.28 | 58.9 | 59.4 | 54.2 |
| "def fibonacci(n):" (4) | **6.70** | **188.9** | 3.82 | 97.5 | 66.6 | 57.1 |
| "Translate to French: The cat sat…" (24) | 3.23 | 88.9 | 3.82 | 89.5 | — | — |

`MTP=3 AL_eff = mtp_acceptance_rate × 3 + 1`. DFlash AL = generated
tokens / iterations.

**Average over 4 prompts:** DFlash 4.14 AL @ **116.4 tok/s**, MTP=3
3.60 AL_eff @ 78.5 tok/s. **DFlash is 1.48× faster than MTP=3 on
average and tops out at 1.94× on code.**

## Progression: sequential prefill → batched verify

The Phase F.2 initial verify path did `k=16` sequential
`engine.prefill(&[t])` calls per iteration (≈ 17 target forwards per
iter). Each call paid the full target-stack cost for one input token.
Phase F.3 collapsed that into a single batched verify forward:
`engine.verify_block_batched(verify_input)` runs one multi-token
prefill chunk through the target, then a batched final RMSNorm +
`bf16_gemm(m=vocab, n=k, k=hidden)` + `sample_rows` greedy argmax,
returning all `k+1` argmaxes in one call. Cost per iter dropped from
~17 to ~2 target forwards.

| Prompt | sequential verify tok/s | batched verify tok/s | speedup |
|---|---:|---:|---:|
| fox | 45.2 | 128.4 | **2.84×** |
| tale | 42.3 | 59.3 | 1.40× |
| fib | 43.9 | 188.9 | **4.30×** |
| translate | 41.5 | 88.9 | 2.14× |

Bonus: AL on the code prompt actually **rose** with batched verify
(4.92 → 6.70) — small numerical differences between bf16_matvec
(rows=1) and bf16_gemm (rows=k) for the lm_head sometimes flip
argmaxes to better align with the drafter's predictions.

## How the loop runs after the fix

```
prefill(prompt)                       # → seed + capture target_hidden
for iter while < max_new_tokens:
    drafter.propose(noise=[seed, MASK*15]) → 15 candidate drafts
    engine.verify_block_batched([seed, *drafts]) → 16 argmaxes
    accepted = longest prefix of (argmax[i] == drafts[i])
    bonus = argmax[accepted]
    engine.crop_state_position(committed_prefix_len)  # roll back rejected tail
    drafter.crop_kv_cache(committed_prefix_len)
    commit drafts[:accepted] + bonus; bonus → next seed
```

`engine.crop_state_position` is a new public method that truncates
`state.position` (KV data past the cut is left in place but
overwritten by the next forward write). It pairs with
`verify_block_batched` to drop the rejected speculative tail.

## When DFlash wins, when MTP=3 wins

| Workload | Winner | Reason |
|---|---|---|
| Pangram/quote completion (fox) | DFlash 1.89× | Drafter has strong context signal; AL ~4.5 on a block of 16. |
| Open-ended prose (tale) | tie | Drafter struggles to align with target distribution; AL ~2.1. |
| Code (fib) | DFlash 1.94× | Highest AL of the bunch (6.7); drafter excels on structured token sequences. |
| Translation | tie | AL 3.2 — middling. |

The drafter's strength on code/structured prompts and weakness on
open-ended prose matches the DFlash paper's findings (Coding AL = 6.30
vs Agent AL = 4.29 in their published numbers).

## Repro

```bash
export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$PWD/target/cuda:${LD_LIBRARY_PATH:-}"
export QWEN36_LONG_CONTEXT_MODE=1

# DFlash with batched verify
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

## Follow-ups not done yet

- CUDA-graph capture of `verify_block_batched` at fixed
  `(k, prompt_class)` shapes. The MTP path already does this for its
  verify graph; ~20–30% more is plausibly on the table for DFlash.
- Per-allocation overhead: the batched logits buffer (~8 MB) is
  allocated/freed per call. Promoting it to a pre-allocated workspace
  on `GpuForwardBuffers` would save a few hundred microseconds per
  iter — small but nonzero.
- Integration behind a `chat --drafter dflash` flag on the existing
  chat CLI.
- Investigation of the underlying decode-kernel divergence on NVFP4
  (this is the `chat --mtp-speculative-tokens 0` bug; orthogonal to
  DFlash but blocks `decode_one` for any direct use).
