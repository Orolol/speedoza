#!/usr/bin/env python3
"""Offline falsification probe: lm_head FP8 e4m3 (W8A16) argmax parity.

Same method as the 2026-06-10 NVFP4 probe that KILLED the NVFP4 lm_head
(1 argmax flip / 27 vectors): simulate the quantized weight matrix in
numpy, compare logits against the BF16 reference on real `final_normed`
vectors dumped by the engine, and gate on top-1 flips (the lm_head feeds
greedy sampling directly — any flip breaks MTP/DFlash parity).

Inputs: a directory tree of engine dumps, one subdir per prompt, each
containing `final_normed.bf16` (QWEN36_DEBUG_DUMP_DIR dumps, hidden_size
BF16 values). Collect them with chat runs:

    for i, prompt in prompts:
        QWEN36_DEBUG_DUMP_DIR=/tmp/lmhead_probe/$i \
        qwen36 chat --model-dir $M --prompt "$prompt" --max-new-tokens 1 ...

Run:  uv run --with numpy --with ml_dtypes \
          python scripts/lmhead_fp8_probe.py /tmp/lmhead_probe

Variants probed (all W8A16: weights e4m3, activations BF16, FP32 accum):
  per-tensor  scale = amax(W) / 448
  per-row     scale = amax(W_row) / 448         (per output channel)
  per-block   scale = amax over 128-col blocks  (plausible kernel layout)

Gate (docs/perf-roadmap.md): top-1 flips == 0, else the lane dies like
NVFP4 did. Report top-5 overlap and |dlogit| stats for the record.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import ml_dtypes

MODEL = Path.home() / "models/Qwen3.6-27B-Text-NVFP4-MTP/model.safetensors"
HIDDEN = 5120
ROW_CHUNK = 16384  # rows of W processed at a time (keeps RAM < ~1 GB)


def load_lm_head_bf16():
    """Memory-map lm_head.weight [vocab, HIDDEN] BF16 from the safetensors."""
    with open(MODEL, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    meta = hdr["lm_head.weight"]
    assert meta["dtype"] == "BF16", meta
    vocab, hidden = meta["shape"]
    assert hidden == HIDDEN
    off0, off1 = meta["data_offsets"]
    mm = np.memmap(MODEL, dtype=np.uint16, mode="r", offset=base + off0,
                   shape=(vocab, hidden))
    return mm, vocab


def bf16_rows_to_f32(mm_u16, lo, hi):
    return (mm_u16[lo:hi].astype(np.uint32) << 16).view(np.float32)


def e4m3(x_f32):
    """Round-trip through float8_e4m3fn (saturating, max 448)."""
    return x_f32.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)


def load_vec(path):
    raw = np.fromfile(path, dtype=np.uint16)
    assert raw.size == HIDDEN, f"{path}: {raw.size} values, expected {HIDDEN}"
    return (raw.astype(np.uint32) << 16).view(np.float32)


def main():
    dump_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lmhead_probe")
    vec_paths = sorted(dump_root.glob("**/final_normed.bf16"))
    if not vec_paths:
        sys.exit(f"no final_normed.bf16 under {dump_root}")
    X = np.stack([load_vec(p) for p in vec_paths])  # [n, HIDDEN] f32 (from bf16)
    n = X.shape[0]
    print(f"{n} final_normed vectors from {dump_root}")

    mm, vocab = load_lm_head_bf16()
    variants = ["per-tensor", "per-row", "per-block128"]
    logits = {v: np.zeros((n, vocab), np.float32) for v in ["ref"] + variants}

    # Pass 1: global amax for the per-tensor scale.
    amax_global = 0.0
    for lo in range(0, vocab, ROW_CHUNK):
        hi = min(lo + ROW_CHUNK, vocab)
        amax_global = max(amax_global, np.abs(bf16_rows_to_f32(mm, lo, hi)).max())
    s_tensor = amax_global / 448.0
    print(f"weight amax={amax_global:.4f}  per-tensor scale={s_tensor:.6f}")

    # Pass 2: chunked logits for reference + each variant.
    for lo in range(0, vocab, ROW_CHUNK):
        hi = min(lo + ROW_CHUNK, vocab)
        W = bf16_rows_to_f32(mm, lo, hi)  # [chunk, HIDDEN] f32
        logits["ref"][:, lo:hi] = X @ W.T

        Wq = e4m3(W / s_tensor) * s_tensor
        logits["per-tensor"][:, lo:hi] = X @ Wq.T

        s_row = np.abs(W).max(axis=1, keepdims=True) / 448.0
        s_row[s_row == 0] = 1.0
        Wq = e4m3(W / s_row) * s_row
        logits["per-row"][:, lo:hi] = X @ Wq.T

        Wb = W.reshape(hi - lo, HIDDEN // 128, 128)
        s_blk = np.abs(Wb).max(axis=2, keepdims=True) / 448.0
        s_blk[s_blk == 0] = 1.0
        Wq = (e4m3(Wb / s_blk) * s_blk).reshape(hi - lo, HIDDEN)
        logits["per-block128"][:, lo:hi] = X @ Wq.T
        print(f"  rows {lo}..{hi} done", flush=True)

    ref = logits["ref"]
    ref_top1 = ref.argmax(axis=1)
    ref_top5 = np.argsort(-ref, axis=1)[:, :5]
    part = np.partition(ref, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    print(f"\nref top-1 margin: median={np.median(margin):.3f} "
          f"p10={np.percentile(margin, 10):.3f} min={margin.min():.3f}")

    for v in variants:
        L = logits[v]
        top1 = L.argmax(axis=1)
        flips = int((top1 != ref_top1).sum())
        top5 = np.argsort(-L, axis=1)[:, :5]
        overlap = np.mean([len(set(a) & set(b)) for a, b in zip(top5, ref_top5)])
        d = np.abs(L - ref)
        print(f"{v:14s} top1_flips={flips}/{n}  top5_overlap={overlap:.2f}/5  "
              f"|dlogit| mean={d.mean():.4f} max={d.max():.4f}")
        if flips:
            for i in np.nonzero(top1 != ref_top1)[0]:
                print(f"    flip at {vec_paths[i]}: ref={ref_top1[i]} "
                      f"(margin {margin[i]:.3f}) -> {top1[i]}")

    print("\ngate: top1_flips must be 0 for the FP8 lm_head lane to open.")

    # --- Margin-gate analysis (2026-06-12, two-stage exact argmax) --------
    # Design: FP8 scan + top1-top2 margin guard; margin < eps falls back to
    # a full BF16 rescore. Exactness condition: if every |dlogit| <= e_max,
    # then fp8_margin >= 2*e_max implies fp8 top-1 == ref top-1 (each logit
    # moved by at most e_max, so no contender can overtake). Report the
    # fallback rate at candidate eps and verify zero flips survive outside
    # the fallback zone for the per-row variant (the shipped kernel layout).
    L = logits["per-row"]
    e_max = float(np.abs(L - ref).max())
    p2 = np.partition(L, -2, axis=1)
    m_fp8 = p2[:, -1] - p2[:, -2]
    top1_fp8 = L.argmax(axis=1)
    print(f"\nmargin-gate (per-row): e_max={e_max:.4f}  "
          f"fp8 margin: median={np.median(m_fp8):.3f} "
          f"p10={np.percentile(m_fp8, 10):.3f} min={m_fp8.min():.3f}")
    for mult in (1.0, 1.5, 2.0, 3.0, 4.0):
        eps = mult * e_max
        fb = m_fp8 < eps
        flips_outside = int(((top1_fp8 != ref_top1) & ~fb).sum())
        print(f"  eps={eps:.4f} ({mult:.1f}x e_max): fallback "
              f"{int(fb.sum())}/{n} ({100*fb.mean():.1f}%)  "
              f"flips outside fallback={flips_outside} (must be 0)")


if __name__ == "__main__":
    main()
