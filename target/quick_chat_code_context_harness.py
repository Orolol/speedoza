#!/usr/bin/env python3
"""Quick chat/code context bench harness for Qwen3.6 MTP0/MTP4/DFlash.

This is the harness used to reproduce the quick table:
  workloads: chat, code
  contexts: 1024, 8192, 32768
  modes: MTP0, MTP4, DFlash

It writes prompts, raw JSONL, and CSV under target/quick_chat_code_context_runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path


CHAT_UNIT = """System: You are a concise senior software engineer helping during an incident review.
User: We run a Rust CUDA inference service on a single RTX 5090. Latency regressed after a speculative decoding patch. Please analyze the symptoms and give a prioritized debugging plan.
Assistant: I would separate prefill, target decode, drafter decode, and verification cost. I would pin the commit, CUDA library timestamp, context length, max new tokens, KV dtype, and CUDA graph setting. Then I would compare MTP=0, MTP=4, and DFlash on the same chat prompt, recording prefill seconds, decode seconds, generated tokens, acceptance length, and memory headroom.
User: The regression appears beyond 8k context and the answer still looks plausible. What next?
Assistant: Run the parity floor first, then compare chat and code prompts at fixed context lengths. If MTP acceptance collapses, inspect per-position acceptance. If DFlash has stable acceptance length but poor tok/s, profile batched verify and hidden-state capture.
"""


CODE_UNIT = """// Code review prompt: Rust CUDA inference cache and scheduler.
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub key: String,
    pub tokens: Vec<u32>,
    pub created_at: Instant,
    pub ttl: Duration,
}

pub struct PrefixCache {
    entries: HashMap<String, CacheEntry>,
    lru: VecDeque<String>,
    max_entries: usize,
}

impl PrefixCache {
    pub fn new(max_entries: usize) -> Self {
        Self { entries: HashMap::new(), lru: VecDeque::new(), max_entries }
    }

    pub fn get(&mut self, key: &str, now: Instant) -> Option<Vec<u32>> {
        let entry = self.entries.get(key)?;
        if now.duration_since(entry.created_at) > entry.ttl {
            self.entries.remove(key);
            return None;
        }
        self.lru.retain(|k| k != key);
        self.lru.push_back(key.to_string());
        Some(entry.tokens.clone())
    }

    pub fn insert(&mut self, key: String, tokens: Vec<u32>, ttl: Duration, now: Instant) {
        let entry = CacheEntry { key: key.clone(), tokens, created_at: now, ttl };
        self.entries.insert(key.clone(), entry);
        self.lru.retain(|k| k != &key);
        self.lru.push_back(key);
        while self.entries.len() > self.max_entries {
            if let Some(old) = self.lru.pop_front() { self.entries.remove(&old); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_hit_then_expiry() {
        let mut cache = PrefixCache::new(8);
        let now = Instant::now();
        cache.insert("chat:alpha".into(), vec![1, 2, 3], Duration::from_secs(30), now);
        assert_eq!(cache.get("chat:alpha", now + Duration::from_secs(1)), Some(vec![1, 2, 3]));
        assert_eq!(cache.get("chat:alpha", now + Duration::from_secs(60)), None);
    }
}
"""


# Compact prompts were used only where `drafter-chat-smoke --prompt` hit
# Linux argv length limits or the original repeated code prompt hit a DFlash
# crop-state bug.
COMPACT_CHAT_UNIT = "User: a a a a a a a a a a\nAssistant: a a a a a a a a a a\n"
COMPACT_CODE_UNIT = "# a a a a a a a a a a\nfn main() { a a a a a a a a a a }\n"


FIELDS = [
    "family",
    "target_ctx",
    "mode",
    "status",
    "prompt_tokens",
    "generated_tokens",
    "prefill_seconds",
    "prefill_tps",
    "decode_seconds",
    "tokgen_tps",
    "mtp_draft_acceptance_rate",
    "acceptance_length",
    "total_seconds",
    "notes",
    "error",
]


def run_json(cmd: list[str], env: dict[str, str], cwd: Path) -> tuple[dict | None, str, int]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return None, (proc.stderr + "\n" + proc.stdout).strip(), proc.returncode
    try:
        return json.loads(proc.stdout), proc.stderr.strip(), 0
    except Exception as exc:
        msg = f"json parse failed: {exc}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout[:4000]}"
        return None, msg, proc.returncode


def token_count(bin_path: Path, model_dir: Path, cuda_dir: Path, text: str) -> int:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{cuda_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    out = subprocess.check_output(
        [
            str(bin_path),
            "tokenize",
            "--model-dir",
            str(model_dir),
            "--text",
            text,
            "--add-special-tokens",
        ],
        env=env,
        text=True,
    )
    return len(json.loads(out))


def write_prompts(prompt_dir: Path, bin_path: Path, model_dir: Path, cuda_dir: Path) -> Path:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for family, unit in [("chat", CHAT_UNIT), ("code", CODE_UNIT)]:
        unit_tokens = token_count(bin_path, model_dir, cuda_dir, unit)
        for target in [1024, 8192, 32768]:
            reps = max(1, math.ceil(target / unit_tokens))
            text = unit * reps
            path = prompt_dir / f"{family}_{target}.txt"
            path.write_text(text, encoding="utf-8")
            manifest.append(
                {
                    "family": family,
                    "target_tokens": target,
                    "path": str(path),
                    "unit_tokens": unit_tokens,
                    "repetitions": reps,
                    "estimated_tokens": reps * unit_tokens,
                    "bytes": len(text.encode("utf-8")),
                }
            )

    compact_chat32 = prompt_dir / "chat_32768_dflash_compact.txt"
    compact_chat32.write_text(COMPACT_CHAT_UNIT * 1261, encoding="utf-8")
    manifest.append(
        {
            "family": "chat",
            "target_tokens": 32768,
            "path": str(compact_chat32),
            "compact_for": "DFlash argv-safe 32k chat",
            "bytes": compact_chat32.stat().st_size,
        }
    )

    compact_code8 = prompt_dir / "code_8192_dflash_compact.txt"
    compact_code8.write_text(COMPACT_CODE_UNIT * 293, encoding="utf-8")
    manifest.append(
        {
            "family": "code",
            "target_tokens": 8192,
            "path": str(compact_code8),
            "compact_for": "DFlash crop-state retry 8k code",
            "bytes": compact_code8.stat().st_size,
        }
    )

    manifest_path = prompt_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/orosius/speedoza"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/orosius/models/Qwen3.6-27B-Text-NVFP4-MTP"))
    parser.add_argument("--drafter-dir", type=Path, default=Path("/home/orosius/models/Qwen3.6-27B-DFlash"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--long-context-1k", action="store_true", default=True)
    args = parser.parse_args()

    root = args.root
    bin_path = root / "target/release/qwen36"
    cuda_dir = root / "target/cuda"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = root / "target/quick_chat_code_context_runs" / stamp
    prompt_dir = run_dir / "prompts"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = write_prompts(prompt_dir, bin_path, args.model_dir, cuda_dir)
    jsonl_path = run_dir / "raw.jsonl"
    csv_path = run_dir / "results.csv"

    env_base = os.environ.copy()
    env_base["QWEN36_FP4_KERNEL_LIB_DIR"] = str(cuda_dir)
    env_base["LD_LIBRARY_PATH"] = f"{cuda_dir}:{env_base.get('LD_LIBRARY_PATH', '')}"

    rows: list[dict] = []

    def save(row: dict, raw: dict | None = None) -> None:
        rows.append(row)
        with jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"row": row, "raw": raw}, ensure_ascii=False) + "\n")
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for saved in rows:
                writer.writerow({key: saved.get(key, "") for key in FIELDS})
        if row["status"] == "ok":
            print(
                f"OK {row['family']} ctx={row['target_ctx']} {row['mode']}: "
                f"prompt={row['prompt_tokens']} prefill={row['prefill_tps']:.1f} "
                f"tokgen={row['tokgen_tps']:.1f} {row.get('notes', '')}",
                flush=True,
            )
        else:
            print(
                f"ERR {row['family']} ctx={row['target_ctx']} {row['mode']}: "
                f"{row['error'][:260]}",
                flush=True,
            )

    print(f"manifest={manifest_path}")
    print(f"jsonl={jsonl_path}")
    print(f"csv={csv_path}")

    for family in ["chat", "code"]:
        for ctx in [1024, 8192, 32768]:
            for mtp in [0, 4]:
                mode = f"MTP{mtp}"
                env = env_base.copy()
                notes = ""
                if ctx == 1024:
                    env["QWEN36_LONG_CONTEXT_MODE"] = "1"
                    notes = "long-context retry-compatible"
                cmd = [
                    str(bin_path),
                    "bench",
                    "--model-dir",
                    str(args.model_dir),
                    "--prompt-file",
                    str(prompt_dir / f"{family}_{ctx}.txt"),
                    "--prompt-tokens",
                    str(ctx),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--mtp-speculative-tokens",
                    str(mtp),
                ]
                data, err, code = run_json(cmd, env, root)
                if data is None:
                    save(
                        {
                            "family": family,
                            "target_ctx": ctx,
                            "mode": mode,
                            "status": "error",
                            "notes": notes,
                            "error": f"exit={code}: {err}",
                        },
                        {"stderr_stdout": err},
                    )
                    continue
                save(
                    {
                        "family": family,
                        "target_ctx": ctx,
                        "mode": mode,
                        "status": "ok",
                        "prompt_tokens": data.get("prompt_tokens"),
                        "generated_tokens": data.get("generated_tokens"),
                        "prefill_seconds": data.get("prefill_seconds"),
                        "prefill_tps": data.get("prefill_tokens_per_second"),
                        "decode_seconds": data.get("decode_seconds"),
                        "tokgen_tps": data.get("decode_tokens_per_second"),
                        "mtp_draft_acceptance_rate": data.get("mtp_draft_acceptance_rate", ""),
                        "acceptance_length": "",
                        "total_seconds": data.get("total_seconds"),
                        "notes": notes,
                        "error": "",
                    },
                    data,
                )

    dflash_jobs = [
        ("chat", 1024, prompt_dir / "chat_1024.txt", ""),
        ("chat", 8192, prompt_dir / "chat_8192.txt", ""),
        ("chat", 32768, prompt_dir / "chat_32768_dflash_compact.txt", "compact argv-safe chat prompt"),
        ("code", 1024, prompt_dir / "code_1024.txt", ""),
        ("code", 8192, prompt_dir / "code_8192_dflash_compact.txt", "compact retry"),
        ("code", 32768, prompt_dir / "code_32768.txt", ""),
    ]
    for family, ctx, prompt_path, notes in dflash_jobs:
        env = env_base.copy()
        env["QWEN36_LONG_CONTEXT_MODE"] = "1"
        cmd = [
            str(bin_path),
            "drafter-chat-smoke",
            "--model-dir",
            str(args.model_dir),
            "--drafter-dir",
            str(args.drafter_dir),
            "--prompt",
            prompt_path.read_text(encoding="utf-8"),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        data, err, code = run_json(cmd, env, root)
        if data is None:
            save(
                {
                    "family": family,
                    "target_ctx": ctx,
                    "mode": "DFlash",
                    "status": "error",
                    "notes": notes,
                    "error": f"exit={code}: {err}",
                },
                {"stderr_stdout": err},
            )
            continue
        timings = data.get("timings_seconds", {})
        prompt_tokens = data.get("prompt_tokens") or 0
        prefill_seconds = timings.get("prefill") or 0.0
        save(
            {
                "family": family,
                "target_ctx": ctx,
                "mode": "DFlash",
                "status": "ok",
                "prompt_tokens": prompt_tokens,
                "generated_tokens": data.get("generated_token_count"),
                "prefill_seconds": prefill_seconds,
                "prefill_tps": (prompt_tokens / prefill_seconds) if prefill_seconds else 0.0,
                "decode_seconds": timings.get("decode"),
                "tokgen_tps": data.get("tokens_per_second"),
                "mtp_draft_acceptance_rate": "",
                "acceptance_length": data.get("acceptance_length"),
                "total_seconds": timings.get("total"),
                "notes": notes,
                "error": "",
            },
            data,
        )

    print(f"DONE csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
