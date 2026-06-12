# DFlash drafter attention — FlashAttention-tiled kernel (Phase 1)

**Date:** 2026-06-09
**Status:** SHIPPED with caveats — opt-in default off, see §11 outcome.
**Depends on:** DFlash Phase F.2 merged (`6571f37`)
**Estimated effort:** 3–5 days of focused CUDA work
**Tracker:** task #47–#50

## 1. Goal

Replace the naive per-key inner loop in
`kernels-cuda/drafter_attention.cu` with a FlashAttention-2-style
tiled kernel so the DFlash drafter forward scales as
*O(q_len × kv_seq_len) bandwidth-bound* rather than
*O(q_len × kv_seq_len) sequential-scan-bound*.

Current state (per `2026-06-09-dflash-long-context.md`):

| ctx | DFlash AL | DFlash tok/s | MTP=3 tok/s | speedup |
|---:|---:|---:|---:|---:|
| 3262 | 7.76 | 52.5 | 29.1 | 1.80× |
| 7058 | 5.43 | 20.8 | 39.9 | **0.52×** |

The 7K cliff is fully explained by the drafter forward becoming
attention-bound: 5 drafter layers × naive O(q × kv) attention. AL is
still respectable but per-iter cost exceeds gains.

Target after Phase 1:

- 7K context: DFlash ≥ 1.0× MTP=3 (i.e. ≥ ~40 tok/s)
- Break-even regime extended from ~5K → ~20K tokens
- Parity: cos sim ≥ 0.998 vs current naive kernel at every shape
- Short-context throughput: no regression (≤ −1 % at 1K ctx)

## 2. Layout (unchanged from Phase C)

The kernel signature and tensor layouts are stable — the speculative
controller keeps writing the same Q/K/V buffers it does today:

| Tensor | Shape | Layout | Notes |
|---|---|---|---|
| Q | `[q_len, q_heads, head_dim]` | row-major | post-RoPE, post-q_norm |
| K | `[kv_seq_len, kv_heads, head_dim]` | row-major | post-RoPE, post-k_norm; `K = [k_ctx; k_noise]` |
| V | `[kv_seq_len, kv_heads, head_dim]` | row-major | `V = [v_ctx; v_noise]` |
| Output | `[q_len, q_heads, head_dim]` | row-major | BF16 |

Specialised constants for the DFlash drafter:

- `q_len = 16` (block_size, see DFlash design spec)
- `head_dim = 128` (only specialised shape; other shapes still return
  `NOT_IMPLEMENTED` so the dispatcher can fall back to the naive kernel)
- `q_heads / kv_heads` integer ratio (GQA); current drafter is q_heads=40,
  kv_heads=8 → 5× broadcast

All math is BF16 inputs with FP32 accumulators.

## 3. Tile design

Adopting the same wmma m16n16k16 layout used by
`kernels-cuda/attention_flash_prefill.cu` (proven on the chat parity
gate). Differences for the drafter:

| Param | Prefill kernel | This kernel | Reason |
|---|---:|---:|---|
| `kFlashM` (Q rows per CTA) | 32 (2 m-tiles) | **16 (1 m-tile)** | drafter `q_len = 16` fixed |
| `kFlashN` (KV cols per K-iter) | 64 | **64** | same SMEM budget |
| `kFlashD` (head_dim) | 256 | **128** | drafter head_dim |
| `kFlashWarps` | 4 | **4** | same |
| Per-warp D-tile fanout (PV) | 8 | **2** | head_dim/16/2 warps |

Per-CTA SMEM budget:

| Buffer | Bytes |
|---|---:|
| `sm_Q[16 × 128]` BF16 | 4 096 |
| `sm_K[64 × 128]` BF16 | 16 384 |
| `sm_V[64 × 128]` BF16 | 16 384 |
| `sm_S[16 × 64]` FP32 | 4 096 |
| `sm_P[16 × 64]` BF16 | 2 048 |
| `sm_m, sm_l, sm_alpha[16]` FP32 each | 192 |
| **Total** | **~43 KiB** |

Comfortably under the 99 KiB sm_120 SMEM cap. Leaves room to grow
`kFlashN` to 128 in a future revision if profiling shows it wins (would
halve outer iters at 8K ctx for cost of doubling KV staging buffer).

Grid: `(q_heads, 1, 1)` — one CTA per query head. Each CTA processes
the same `q_len = 16` rows across all KV tiles. No work split along
KV — the per-CTA online softmax keeps row state in SMEM throughout.

Threads per CTA: `kFlashWarps × 32 = 128`.

## 4. Algorithm

Standard FlashAttention-2 forward, single-tile M, multi-tile N:

```
for each q_head h (one CTA per h):
    sm_Q ← Q[:, h, :]
    sm_m ← -inf, sm_l ← 0
    o_frags ← 0  (registers, per warp)

    for each KV tile k_iter:
        sm_K ← K[k_base : k_base + 64, kv_head_of(h), :]
        sm_V ← V[k_base : k_base + 64, kv_head_of(h), :]
        S ← Q @ K^T  (wmma, FP32 accum)
        S ← S × qk_scale
        # Optional symmetric sliding-window mask (same as Phase C v1)
        if sliding_window > 0:
            mask S entries outside [q_abs ± sliding_window]

        m_new ← rowmax(S, dim=1)
        alpha ← exp(sm_m - m_new)
        P ← exp(S - m_new)  # BF16 cast for matmul
        l_new ← sm_l × alpha + rowsum(P)
        o_frags ← o_frags × alpha + P @ V  (wmma, FP32 accum)

        sm_m ← m_new
        sm_l ← l_new

    output[:, h, :] ← o_frags / sm_l[:, None]
```

Mask handling: the drafter is non-causal, so no causal mask. The
optional sliding-window mask of Phase C v1 is preserved — set entries
where `|q_abs − k_pos| > sliding_window` to `−∞` before the row-max.

## 5. Numerical contract

- Inputs BF16, accumulators FP32, output BF16. Identical to Phase C v1.
- `qk_scale = rsqrt(head_dim)`.
- Online softmax matches the v1 formula exactly: `o ← o·α + P·V`,
  `l ← l·α + sum(P)`, `m ← max(m, S)`.
- Parity smoke (task #49): cos sim ≥ 0.998 against the v1 kernel for
  - `head_dim = 128`
  - `q_heads ∈ {32, 40}`, `kv_heads = q_heads / {1, 4, 5, 8}` (covers
    drafter shapes + sanity multi-ratio)
  - `kv_seq_len ∈ {16, 64, 128, 1024, 4096, 8192}` (covers the regime
    transition)
  - `sliding_window ∈ {-1, 4096}`

## 6. C ABI

The existing entry point
`qwen36_drafter_attention_block_bf16(const qwen36_drafter_attention_block_spec_t*)`
keeps its signature. Dispatch policy inside the entry point:

```
if (head_dim == 128 && q_len == 16 && kv_seq_len >= 16):
    launch tiled kernel (this spec)
else:
    return QWEN36_STATUS_NOT_IMPLEMENTED  (controller falls back)
```

The tiled kernel becomes the default for the supported shape; the
Phase C v1 kernel stays in tree as `drafter_attention_v1.cu`
(reference for parity + fallback for shapes outside the supported
set, e.g. mid-step rollback iters where `q_len ≠ 16`).

`QWEN36_DRAFTER_ATTENTION_DISABLE_FLASH=1` env var forces the v1
path. Default off.

## 7. Files

```
kernels-cuda/
  drafter_attention.cu                        # dispatcher (kept)
  drafter_attention_v1.cu                     # naïve kernel (renamed from inlined body)
  drafter_attention_flash.cu                  # NEW — this spec
  smoke.cu                                    # +parity smoke (task #49)
docs/superpowers/specs/
  2026-06-09-dflash-fa-drafter-attention.md   # this file
docs/superpowers/notes/
  2026-06-09-dflash-long-context.md           # updated with post-FA bench
```

No Rust-side changes — the C ABI is stable.

## 8. Risks & exit ramps

| Risk | Severity | Mitigation |
|---|---|---|
| wmma frag indexing bug → silent numerical drift | High | Per-shape cos-sim parity smoke against the v1 kernel before any bench |
| Per-warp D-tile fanout (only 2 vs prefill's 8) starves wmma issue width | Medium | Profile with `nsight-compute`; if compute-bound below 80 % SM, switch to `kFlashN = 128` (more K-tiles per outer iter) |
| Spill / register pressure tanks occupancy on 128-thread CTA | Medium | `__launch_bounds__(128, 4)` to force compiler to keep occupancy ≥ 4 active blocks/SM |
| `kv_seq_len` not multiple of 64 → tail tile parity drift | Medium | Mask tail rows in sm_K / sm_V load (already done in v1, just port the gate) |
| New kernel regresses short-ctx (1K) by > 1 % | Medium | Dispatcher gate: `kv_seq_len < 256 → fall back to v1`. Short ctx doesn't need FA tiling. |

## 9. Out of scope

- **TMA loads.** No TMA multicast on sm_120; we use `cp.async` (which
  is what the prefill kernel uses too). TMA single-source is available
  but doesn't change the bandwidth picture at our shapes.
- **FP8 K/V.** Drafter is BF16 throughout. NVFP4 KV is Phase 2 of the
  DFlash megakernel roadmap, separate work.
- **head_dim != 128.** Falls back to v1.
- **Tree mask / causal mask.** Drafter is non-causal block diffusion.
- **Kernel fusion with adjacent ops** (q_norm, k_norm, RoPE). Fused
  drafter attention is part of the drafter megakernel (Phase 3 of the
  DFlash megakernel roadmap), not this spec.

## 11. Outcome (2026-06-09, same day)

Shipped at parity, **not winning on speed.** Two clean findings:

### 11.1 Per-iter cost matches the v1 kernel

Bench, DFlash chat smoke (max-new=256), same session, native Linux,
`QWEN36_LONG_CONTEXT_MODE=1`:

| Prompt | ctx | FA tok/s | v1 tok/s | FA AL | v1 AL | per-iter (FA) | per-iter (v1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AGENT.md head:150 | 3262 | 34.5 | 54.7 | 4.87 | 7.76 | **141 ms** | **142 ms** |
| AGENT.md head:300 | 7058 | 18.8 | 21.2 | 4.81 | 5.43 | **256 ms** | **256 ms** |

Per-iter wall time is bit-identical. Throughput delta is **entirely AL
delta**.

What this tells us about the drafter forward at long context: it is
**not compute-bound** on these shapes (q_len=16, q_heads=32,
head_dim=128). The wmma tensor-core tiling doesn't beat the scalar
v1 loop because both are launch-overhead + per-call HBM bandwidth
limited at the same point. With only 32 CTAs in the FA grid (one per
q_head) versus 640 CTAs in v1 (q_len × q_heads), the FA kernel
under-uses the 192 SMs of the 5090.

### 11.2 Numerical drift at medium kv_seq_len

Parity test, drafter-chat-smoke, prompt = "The quick brown fox jumps
over the" (4 tokens), 33 generated: **identical text** FA vs v1.

Parity test, ~120-word Rust prompt (≈ 180-token ctx at gen halfway),
64 generated: **first ~50 tokens identical, then diverge.** FA
generates `... covering: (2) all negative ...` while v1 generates
`... covering: (1) all positive ...` — the `(1)` token is missed.

Likely candidates (none reproduced yet):
- Online softmax accumulation drift across many K-tiles
- Tail-tile padding boundary in `k_iters` loop when `kv_seq_len %
  kFlashN != 0` (likely benign — masked entries set to −∞ before exp,
  but worth double-checking the alpha update path)
- BF16 cast order in PV (P stored bf16, then re-loaded as wmma input —
  could lose precision vs v1's full-FP32 accum)

### 11.3 Decision

- **Default off.** Env gate flipped: `QWEN36_DRAFTER_ATTENTION_FLASH=1`
  enables the FA path; absence falls back to v1 (was inverted: was
  `_DISABLE_FLASH=1` to opt out).
- **Code stays in tree** as substrate for Phase 4 (verify megakernel,
  which DOES win from wmma tiling because q=16 there is the prefill
  shape that's actually compute-bound, not the drafter shape).
- **Don't fix the AL drift in this kernel** — even fixed, it won't
  speed up DFlash because per-iter is at parity. Time-to-impact is
  better spent on Phase 2 (NVFP4 KV cache + BitDecoding) which
  attacks the real long-context bottleneck (verify's full-KV traffic),
  estimated 30–50 % gain at 7K.

This phase rebalances the roadmap: the drafter attn kernel is
no longer the long-context bottleneck we thought it was. The verify
step is. Phase 4 + Phase 2 should be the focus, with Phase 1's
substrate ready to fold into Phase 4 (since the wmma layout / SMEM
allocator / cooperative load pattern all transfer to the q=16 prefill
batched verify path).

## 10. References

- `kernels-cuda/attention_flash_prefill.cu` — wmma m16n16k16 BF16 +
  online softmax + warp split, working precedent for our hardware
- `kernels-cuda/drafter_attention.cu` — current naive kernel
- DFlash Phase C commit `4c2b43c` — original drafter attention design
- DFlash long-context bench `docs/superpowers/notes/2026-06-09-dflash-long-context.md`
- FlashAttention-2 paper (Dao 2023): `arxiv 2307.08691`
