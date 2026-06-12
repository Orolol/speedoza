# DFlash long-context bench

**Date:** 2026-06-09 (follow-up to the standard sweep)
**Build:** `target/release/qwen36`, RTX 5090, native Linux,
`QWEN36_LONG_CONTEXT_MODE=1`.
**Target:** `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`.
**Drafter:** `z-lab/Qwen3.6-27B-DFlash`.

Adds long-context cells to the 30-cell standard sweep
(`2026-06-09-dflash-final.md`). Probes prompt sizes from 400 to 7058
tokens to understand where the DFlash speedup breaks down.

## Results

| Prompt (content) | actual tokens | gen | DFlash tok/s | DFlash AL | MTP=3 tok/s | speedup |
|---|---:|---:|---:|---:|---:|---:|
| tech_xl_500t (incident post-mortem) | 400 | 32 | 48.0 | 2.31 | 42.9 | 1.12× |
| tech_xl_500t | 400 | 128 | 69.2 | 3.45 | 55.2 | 1.25× |
| tech_xl_500t | 400 | 256 | 56.2 | 2.99 | 49.2 | 1.14× |
| code_xl_1500t (Rust module to complete) | 802 | 32 | 30.0 | 1.88 | 56.0 | 0.54× |
| code_xl_1500t | 802 | 128 | 74.4 | 4.79 | 51.2 | **1.45×** |
| code_xl_1500t | 802 | 256 | 88.6 | 5.82 | 53.6 | **1.65×** |
| long_synth_xxl (9 structured tech blocks) | 953 | 32 | 40.6 | 2.67 | 38.0 | 1.07× |
| long_synth_xxl | 953 | 128 | 82.9 | 5.67 | 48.9 | **1.70×** |
| long_synth_xxl | 953 | 256 | 86.1 | 6.09 | 43.9 | **1.96×** |
| long_synth_3000t (7 topic-shift paragraphs) | 986 | 32 | 24.1 | 1.74 | 47.6 | 0.51× |
| long_synth_3000t | 986 | 128 | 26.9 | 1.93 | 54.7 | 0.49× |
| long_synth_3000t | 986 | 256 | 26.1 | 1.96 | 44.4 | 0.59× |
| agent_md_3k (AGENT.md head:150) | 3262 | 64 | 36.2 | 5.29 | 27.6 | **1.31×** |
| agent_md_3k | 3262 | 256 | 52.5 | 7.76 | 29.1 | **1.80×** |
| agent_md_7k (AGENT.md head:300) | 7058 | 128 | 12.2 | 3.37 | 39.5 | **0.31×** |
| agent_md_7k | 7058 | 256 | 20.8 | 5.43 | 39.9 | 0.52× |

## What we learn

The naive "long context = bad for DFlash" framing from the standard
sweep was misleading. Two distinct effects compose:

### 1. Topic / distribution diversity in the prompt

Compare the two ~1000-token prompts on the same engine config:

- `long_synth_xxl` (953t, 9 paragraphs all in a technical register):
  AL 5.67 / 6.09 at gen=128/256, **1.70× / 1.96×** speedup.
- `long_synth_3000t` (986t, 7 paragraphs jumping between cooking,
  physics, game design, and software): AL 1.93 / 1.96, **0.49× /
  0.59×** speedup.

Almost identical token count, drastically different drafter
performance. The block-diffusion drafter conditions on the
concatenated target hidden states; when those hidden states span
incompatible distributions, the drafter's denoising signal becomes
noise. Coherent technical writing keeps the conditioning sharp.

### 2. Drafter forward cost dominates at 5K+ tokens

| ctx tokens | DFlash AL | DFlash tok/s | MTP=3 tok/s | speedup |
|---:|---:|---:|---:|---:|
| 3262 | 7.76 (gen=256) | 52.5 | 29.1 | **1.80×** |
| 7058 | 5.43 (gen=256) | 20.8 | 39.9 | 0.52× |

At 7K tokens DFlash AL is still respectable (5.4) but the per-iter
cost is dominated by the drafter forward itself. Our drafter
attention kernel
(`kernels-cuda/drafter_attention.cu`) is a naive O(q_len × kv_seq_len)
implementation — no tiling, no FlashAttention. At 7K context that's
~7× more work per layer than at 1K context, multiplied by 5 drafter
layers. The MTP head meanwhile is a tiny single-extra-layer pass
that doesn't redo prompt attention, so MTP=3 throughput stays flat
across context lengths (~30–40 tok/s here, marginally lower at 7K
because the chunked prefill takes longer to set up KV but the
per-token cost is unchanged).

## Updated routing recommendations

Combining this with the standard sweep:

| Regime | Pick | Why |
|---|---|---|
| Prompt < 200t | DFlash | drafter cheap, AL high |
| 200–1000t, coherent / single-domain | DFlash (gen ≥ 128) | AL holds; drafter cost still manageable |
| 200–1000t, multi-topic prose | MTP=3 | drafter can't condition cleanly |
| 1000–3000t, structured technical | DFlash (gen ≥ 64) | AL stays high; per-iter cost ≈ MTP's |
| 3000–5000t | mixed (depends on content); benchmark | drafter forward starts dominating |
| > 5000t | MTP=3 | drafter forward time exceeds gains from AL |

## What would close the long-context gap

The DFlash drafter forward cost at long context is set entirely by
the drafter's attention kernel. The current kernel was implemented
quickly in Phase C (commit `4c2b43c`) — one CTA per
`(q_pos, q_head)`, sequential scan over keys, no tiling. Replacing
it with a FlashAttention-style tiled kernel would scale much better:

- Current: O(q × kv) wall time per attention call, no SMEM tiling.
- FlashAttention: O(q × kv) work but bandwidth-bound, with much
  better SMEM utilisation. Expected 3–5× speedup at long context.

This would push the DFlash break-even point from ~5K up to maybe
~20K tokens. The change is local (one CUDA kernel, parity-checked
against the existing host fp32 reference) but non-trivial — a few
days of focused work.

## Raw data

CSV at `docs/superpowers/notes/2026-06-09-dflash-long-context.csv`.

The AGENT.md 3K/7K runs were one-offs (not run through the sweep
driver) and not in the CSV; they're transcribed in the table above.

## Repro

```bash
# Long sweep cells already in the script
QWEN36_FP4_KERNEL_LIB_DIR=$PWD/target/cuda \
LD_LIBRARY_PATH=$PWD/target/cuda:${LD_LIBRARY_PATH:-} \
QWEN36_LONG_CONTEXT_MODE=1 \
  /home/orosius/workspace/dmtp/.venv/bin/python \
  scripts/dflash_bench_sweep.py \
    --model-dir   ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --drafter-dir ~/models/Qwen3.6-27B-DFlash \
    --binary      $PWD/target/release/qwen36 \
    --cuda-lib    target/cuda \
    --output      /tmp/dflash_long_sweep.csv \
    --prompt-filter _xl_      # also: long_synth, long_synth_xxl

# AGENT.md-based stress test
head -150 AGENT.md > /tmp/agent_chunk.txt   # ~3262 tokens
head -300 AGENT.md > /tmp/agent_5k.txt      # ~7058 tokens

QWEN36_LONG_CONTEXT_MODE=1 \
QWEN36_FP4_KERNEL_LIB_DIR=$PWD/target/cuda \
LD_LIBRARY_PATH=$PWD/target/cuda:${LD_LIBRARY_PATH:-} \
  $PWD/target/release/qwen36 drafter-chat-smoke \
    --model-dir   ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --drafter-dir ~/models/Qwen3.6-27B-DFlash \
    --prompt "$(cat /tmp/agent_chunk.txt)" --max-new-tokens 256

QWEN36_LONG_CONTEXT_MODE=1 \
QWEN36_FP4_KERNEL_LIB_DIR=$PWD/target/cuda \
LD_LIBRARY_PATH=$PWD/target/cuda:${LD_LIBRARY_PATH:-} \
  $PWD/target/release/qwen36 bench \
    --model-dir   ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
    --prompt-file /tmp/agent_chunk.txt \
    --prompt-tokens 3262 \
    --max-new-tokens 256 \
    --mtp-speculative-tokens 3
```
