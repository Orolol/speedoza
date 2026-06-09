#!/usr/bin/env python3
"""DFlash vs MTP=3 chain bench sweep.

Runs `qwen36 drafter-chat-smoke` (DFlash batched verify) and `qwen36
bench --mtp-speculative-tokens 3` (chain MTP) across a matrix of
prompt types, prompt lengths, and generation lengths. Outputs a CSV
plus a Markdown summary table.

Usage:
  python3 scripts/dflash_bench_sweep.py \
      --model-dir   ~/models/Qwen3.6-27B-Text-NVFP4-MTP \
      --drafter-dir ~/models/Qwen3.6-27B-DFlash \
      --binary      target/release/qwen36 \
      --cuda-lib    target/cuda \
      --output      /tmp/dflash_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROMPTS = [
    (
        "completion_short",
        "The quick brown fox jumps over the",
    ),
    (
        "code_short",
        "def fibonacci(n):",
    ),
    (
        "prose_medium",
        "Once upon a time in a distant kingdom there lived a young prince. "
        "He was wise and brave, beloved by his subjects, and famous throughout "
        "the realm for his kindness toward children and animals. One spring day "
        "he decided to journey beyond the castle walls. ",
    ),
    (
        "qa_medium",
        "You are an experienced systems engineer. A user asks you the following "
        "question and you need to answer it concisely and technically.\n\n"
        "Question: How does the Linux kernel schedule processes on a "
        "multi-core machine? Please cover the main scheduler, fair scheduling, "
        "and load balancing.\n\nAnswer: ",
    ),
    (
        "code_long",
        "# Below is a Python module that implements a simple LRU cache.\n"
        "# The cache supports `get(key)`, `put(key, value)`, eviction on size\n"
        "# overflow, and a `stats()` method that returns a dict with hits,\n"
        "# misses, and current size.\n\n"
        "from collections import OrderedDict\n\n"
        "class LRUCache:\n"
        "    def __init__(self, capacity: int):\n"
        "        if capacity <= 0:\n"
        "            raise ValueError('capacity must be positive')\n"
        "        self.capacity = capacity\n"
        "        self._store: OrderedDict = OrderedDict()\n"
        "        self._hits = 0\n"
        "        self._misses = 0\n\n"
        "    def get(self, key):\n"
        "        if key in self._store:\n"
        "            self._hits += 1\n"
        "            self._store.move_to_end(key)\n"
        "            return self._store[key]\n"
        "        self._misses += 1\n"
        "        return None\n\n"
        "    def put(self, key, value):\n"
        "        # Add to the cache, evicting the least-recently-used entry\n"
        "        # if the cache is at capacity.\n"
        "        ",
    ),
]

GEN_LENGTHS = [32, 128, 256]


def run_dflash(
    binary: str,
    env: dict,
    model_dir: str,
    drafter_dir: str,
    prompt: str,
    max_new: int,
) -> dict:
    """Returns {prompt_tokens, generated, tok_s, al, decode_s}."""
    cmd = [
        binary,
        "drafter-chat-smoke",
        "--model-dir",
        model_dir,
        "--drafter-dir",
        drafter_dir,
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new),
    ]
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"drafter-chat-smoke failed (rc={result.returncode}): {result.stderr[-500:]}"
        )
    # drafter-chat-smoke emits one big JSON. Find it.
    data = json.loads(result.stdout)
    timings = data["timings_seconds"]
    return {
        "prompt_tokens": data["prompt_tokens"],
        "generated": data["generated_token_count"],
        "iters": data["iterations"],
        "tok_s": data["tokens_per_second"],
        "al": data["acceptance_length"],
        "decode_s": timings["decode"],
    }


def run_mtp(
    binary: str,
    env: dict,
    model_dir: str,
    prompt: str,
    max_new: int,
    prompt_tokens_count: int,
) -> dict:
    """Runs bench with --mtp-speculative-tokens 3 and the given prompt as a
    prompt-file with explicit token count."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        f.write(prompt)
        prompt_path = f.name
    try:
        cmd = [
            binary,
            "bench",
            "--model-dir",
            model_dir,
            "--prompt-file",
            prompt_path,
            "--prompt-tokens",
            str(prompt_tokens_count),
            "--max-new-tokens",
            str(max_new),
            "--mtp-speculative-tokens",
            "3",
        ]
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
    finally:
        os.unlink(prompt_path)
    if result.returncode != 0:
        raise RuntimeError(
            f"bench failed (rc={result.returncode}): {result.stderr[-500:]}"
        )
    data = json.loads(result.stdout)
    accept = data.get("mtp_acceptance_rate", 0.0)
    return {
        "prompt_tokens": data["prompt_tokens"],
        "generated": data["generated_tokens"],
        "tok_s": data["decode_tokens_per_second"],
        "al_eff": accept * 3 + 1,
        "accept_rate": accept,
        "decode_s": data["decode_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--drafter-dir", required=True)
    parser.add_argument("--binary", default="target/release/qwen36")
    parser.add_argument(
        "--cuda-lib",
        default="target/cuda",
        help="Path containing libqwen36_fp4_kernels.so",
    )
    parser.add_argument("--output", default="/tmp/dflash_sweep.csv")
    parser.add_argument(
        "--prompt-filter",
        default=None,
        help="If set, only run prompts whose name matches this substring.",
    )
    parser.add_argument(
        "--gen-filter",
        type=int,
        default=None,
        help="If set, only run this max_new_tokens value.",
    )
    args = parser.parse_args()

    cuda_lib = Path(args.cuda_lib).resolve()
    env = os.environ.copy()
    env["QWEN36_FP4_KERNEL_LIB_DIR"] = str(cuda_lib)
    env["LD_LIBRARY_PATH"] = f"{cuda_lib}:" + env.get("LD_LIBRARY_PATH", "")
    env["QWEN36_LONG_CONTEXT_MODE"] = "1"

    rows = []
    total = 0
    expected = sum(
        1
        for name, _ in PROMPTS
        if args.prompt_filter is None or args.prompt_filter in name
    ) * len(
        [g for g in GEN_LENGTHS if args.gen_filter is None or g == args.gen_filter]
    ) * 2
    print(f"Starting sweep — expected {expected} runs", flush=True)

    for prompt_name, prompt_text in PROMPTS:
        if args.prompt_filter and args.prompt_filter not in prompt_name:
            continue
        for max_new in GEN_LENGTHS:
            if args.gen_filter is not None and max_new != args.gen_filter:
                continue

            total += 1
            print(
                f"[{total:>2}/{expected}] dflash {prompt_name:>16}  gen={max_new:>3}",
                flush=True,
            )
            try:
                df = run_dflash(
                    args.binary,
                    env,
                    args.model_dir,
                    args.drafter_dir,
                    prompt_text,
                    max_new,
                )
            except Exception as e:
                print(f"   FAILED: {e}", flush=True)
                continue

            total += 1
            print(
                f"[{total:>2}/{expected}] mtp3   {prompt_name:>16}  gen={max_new:>3}",
                flush=True,
            )
            try:
                mtp = run_mtp(
                    args.binary,
                    env,
                    args.model_dir,
                    prompt_text,
                    max_new,
                    df["prompt_tokens"],
                )
            except Exception as e:
                print(f"   FAILED: {e}", flush=True)
                mtp = None

            speedup = (df["tok_s"] / mtp["tok_s"]) if mtp else None
            row = {
                "prompt": prompt_name,
                "prompt_tokens": df["prompt_tokens"],
                "max_new_tokens": max_new,
                "dflash_generated": df["generated"],
                "dflash_iters": df["iters"],
                "dflash_tok_s": round(df["tok_s"], 1),
                "dflash_al": round(df["al"], 2),
                "dflash_decode_s": round(df["decode_s"], 2),
                "mtp_tok_s": round(mtp["tok_s"], 1) if mtp else None,
                "mtp_al_eff": round(mtp["al_eff"], 2) if mtp else None,
                "mtp_decode_s": round(mtp["decode_s"], 2) if mtp else None,
                "speedup_dflash_vs_mtp": round(speedup, 2) if speedup else None,
            }
            rows.append(row)
            print(
                f"   dflash {df['tok_s']:>7.1f} tok/s AL={df['al']:>5.2f}"
                f"  mtp3 {mtp['tok_s']:>7.1f} tok/s AL_eff={mtp['al_eff']:>5.2f}"
                f"  speedup {speedup:>5.2f}×"
                if mtp
                else f"   dflash {df['tok_s']:>7.1f} tok/s",
                flush=True,
            )

    if not rows:
        print("No rows captured.", file=sys.stderr)
        return 1

    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to {csv_path}")

    # Markdown table grouped by prompt.
    print("\n## Sweep results (markdown)\n")
    headers = [
        "prompt",
        "prompt_tok",
        "gen",
        "DF tok/s",
        "DF AL",
        "MTP3 tok/s",
        "MTP3 AL_eff",
        "speedup",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print(
            "| "
            + " | ".join(
                [
                    r["prompt"],
                    str(r["prompt_tokens"]),
                    str(r["max_new_tokens"]),
                    str(r["dflash_tok_s"]),
                    str(r["dflash_al"]),
                    str(r["mtp_tok_s"]) if r["mtp_tok_s"] else "—",
                    str(r["mtp_al_eff"]) if r["mtp_al_eff"] else "—",
                    str(r["speedup_dflash_vs_mtp"])
                    if r["speedup_dflash_vs_mtp"]
                    else "—",
                ]
            )
            + " |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
