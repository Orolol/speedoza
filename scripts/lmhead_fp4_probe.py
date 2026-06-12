#!/usr/bin/env python3
"""Offline probe: NVFP4 (e2m1 + e4m3 block-16 scales) lm_head SCAN for the
two-stage top-8 exact argmax.

The 2026-06-10 NVFP4 probe killed the DIRECT FP4 argmax (1 top-1 flip / 27
vectors). The two-stage architecture (PR #28) changes the question: the FP4
pass only needs to (a) place the true argmax inside its top-8 candidates and
(b) admit a usable guard `bf16_best >= fp4_9th + eps`. A bigger e_max costs
eps and fallbacks, not correctness. This probe measures, per dumped hidden
vector:
  - e_max = max |fp4_logit - ref_logit| (the eps floor)
  - the RANK of the true (ref) argmax under the FP4 scores (must be <= 8
    for the candidate set to contain it)
  - the guard fallback rate at an eps grid (offline proxy: bound B = 9th
    largest FP4 score; the kernel's B = max(9th winner, max block-v2) is
    >= this, so the real fallback rate is an OVERestimate of these numbers)
  - the safety property: any sample whose true argmax ranks > 8 MUST also
    fail the guard (else the two-stage would silently return a wrong token).

Inputs: same dump tree as lmhead_fp8_probe.py (final_normed.bf16 per prompt).
Run:  uv run --with numpy --with ml_dtypes \
          python scripts/lmhead_fp4_probe.py /tmp/lmhead_probe
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import ml_dtypes

MODEL = Path.home() / "models/Qwen3.6-27B-Text-NVFP4-MTP/model.safetensors"
HIDDEN = 5120
BLOCK = 16
ROW_CHUNK = 16384


def load_lm_head_bf16():
    with open(MODEL, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    meta = hdr["lm_head.weight"]
    assert meta["dtype"] == "BF16", meta
    vocab, hidden = meta["shape"]
    assert hidden == HIDDEN
    off0, _ = meta["data_offsets"]
    mm = np.memmap(MODEL, dtype=np.uint16, mode="r", offset=base + off0,
                   shape=(vocab, hidden))
    return mm, vocab


def bf16_rows_to_f32(mm_u16, lo, hi):
    return (mm_u16[lo:hi].astype(np.uint32) << 16).view(np.float32)


def nvfp4_roundtrip(w_f32):
    """Quantize [rows, HIDDEN] f32 to NVFP4 (e2m1 values, e4m3 block-16
    scales, tensor scale 1.0) and dequantize back to f32."""
    rows = w_f32.shape[0]
    blocks = w_f32.reshape(rows, HIDDEN // BLOCK, BLOCK)
    amax = np.abs(blocks).max(axis=2, keepdims=True)
    scale = (amax / 6.0).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    scale[scale == 0] = 1.0
    q = (blocks / scale).astype(ml_dtypes.float4_e2m1fn).astype(np.float32)
    return (q * scale).reshape(rows, HIDDEN)


def load_vec(path):
    raw = np.fromfile(path, dtype=np.uint16)
    assert raw.size == HIDDEN, f"{path}: {raw.size} values, expected {HIDDEN}"
    return (raw.astype(np.uint32) << 16).view(np.float32)


def main():
    dump_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lmhead_probe")
    vec_paths = sorted(dump_root.glob("**/final_normed.bf16"))
    if not vec_paths:
        sys.exit(f"no final_normed.bf16 under {dump_root}")
    X = np.stack([load_vec(p) for p in vec_paths])
    n = X.shape[0]
    print(f"{n} final_normed vectors from {dump_root}")

    mm, vocab = load_lm_head_bf16()
    # The real scan is W4A4 (the Direction B GEMV is e2m1 x e2m1): the
    # activations go through the same NVFP4 block-16 quantization.
    Xq = nvfp4_roundtrip(X)
    ref = np.zeros((n, vocab), np.float32)
    l4 = np.zeros((n, vocab), np.float32)
    for lo in range(0, vocab, ROW_CHUNK):
        hi = min(lo + ROW_CHUNK, vocab)
        W = bf16_rows_to_f32(mm, lo, hi)
        ref[:, lo:hi] = X @ W.T
        l4[:, lo:hi] = Xq @ nvfp4_roundtrip(W).T
        print(f"  rows {lo}..{hi} done", flush=True)

    e_max = float(np.abs(l4 - ref).max())
    e_mean = float(np.abs(l4 - ref).mean())
    ref_top1 = ref.argmax(axis=1)
    ref_best = ref.max(axis=1)

    # Rank of the true argmax under the FP4 scores.
    order4 = np.argsort(-l4, axis=1)
    ranks = np.array([
        int(np.nonzero(order4[i] == ref_top1[i])[0][0]) for i in range(n)
    ])
    top9_4 = -np.partition(-l4, 8, axis=1)[:, :9]
    ninth = np.sort(top9_4, axis=1)[:, 0]  # 9th largest FP4 score

    print(f"\nnvfp4 scan: |dlogit| mean={e_mean:.4f} max={e_max:.4f}")
    print(f"true-argmax rank under FP4: max={ranks.max()} "
          f"dist={np.bincount(np.minimum(ranks, 9), minlength=10).tolist()} "
          f"(index 9 = rank>=9 = MISS)")

    direct_flips = int((order4[:, 0] != ref_top1).sum())
    print(f"direct FP4 argmax flips (the 2026-06-10 framing): {direct_flips}/{n}")

    for mult in (1.0, 1.5, 2.0):
        eps = mult * e_max
        # Guard proxy: certified iff ref_best >= ninth + eps. (Real kernel
        # bound is >= ninth, so real fallbacks >= these.)
        fb = ref_best < ninth + eps
        missed = ranks >= 8
        unsafe = int((missed & ~fb).sum())  # missed candidate AND certified
        print(f"  eps={eps:.3f} ({mult:.1f}x e_max): fallback proxy "
              f"{int(fb.sum())}/{n} ({100 * fb.mean():.1f}%)  "
              f"UNSAFE (miss certified)={unsafe} (must be 0)")

    print("\ngate: rank<=7 partout (ou miss toujours en fallback) ET "
          "fallback ~<15% pour ouvrir la lane FP4-scan.")


if __name__ == "__main__":
    main()
