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
    # Long-context prompts. Lengths in the file path are nominal
    # (~tokens); the actual token count is reported in each row.
    (
        "tech_xl_500t",
        (
            "You are an experienced distributed-systems engineer. Below is a "
            "summary of a deployment incident that just happened in production. "
            "Read it carefully and then answer the question at the end.\n\n"
            "## Incident summary\n\n"
            "At 14:32 UTC the API gateway started returning HTTP 502 errors "
            "for roughly 12% of incoming traffic. The blast radius was the "
            "us-east-1 region; eu-west-2 and ap-southeast-1 were unaffected. "
            "Pager fired at 14:35 UTC.\n\n"
            "Investigation: the affected requests all hit the `/recommendations` "
            "endpoint, which fans out to three backend services: the user "
            "profile service, the inventory service, and a vector-search "
            "service running on a separate Kubernetes cluster. Latency on the "
            "vector-search service had risen from a baseline of 35 ms p99 to "
            "over 4 seconds. The gateway's per-upstream timeout is 2 seconds, "
            "so any vector-search call exceeding that timed out and bubbled "
            "up as a 502 to the client.\n\n"
            "Cause: a routine reindex job had been launched at 14:28 UTC by "
            "the data-platform team. The job ran on the same node pool as the "
            "vector-search query workers and triggered heavy CPU contention. "
            "Although the workers have CPU requests of 4 cores each, the "
            "reindex pod was best-effort and burst-able, so under load the "
            "kubelet throttled the query workers.\n\n"
            "Mitigation: the on-call SRE drained the reindex pod at 14:39 UTC. "
            "Vector-search p99 returned to baseline within 90 seconds. Error "
            "rate at the gateway fell below 0.1% by 14:42 UTC.\n\n"
            "## Question\n\n"
            "Write a five-step post-mortem action plan that prevents this "
            "specific class of incident from happening again. For each step, "
            "name the owner, the target completion date relative to today, "
            "and the explicit verification criterion. Be concrete."
        ),
    ),
    (
        "code_xl_1500t",
        (
            "Below is a partially-written Rust module that needs to be "
            "completed. The module implements a small in-memory key-value "
            "store with TTL-based expiration, snapshot-and-restore, and a "
            "background expiration sweeper. Several methods are missing "
            "implementations or have `todo!()` placeholders. Your job is to "
            "fill them in so the public API works end-to-end and the unit "
            "tests at the bottom pass.\n\n"
            "```rust\n"
            "use std::collections::HashMap;\n"
            "use std::sync::{Arc, Mutex};\n"
            "use std::time::{Duration, Instant};\n\n"
            "/// A single entry in the store. `expires_at == None` means the\n"
            "/// entry never expires.\n"
            "#[derive(Debug, Clone)]\n"
            "struct Entry {\n"
            "    value: Vec<u8>,\n"
            "    expires_at: Option<Instant>,\n"
            "    version: u64,\n"
            "}\n\n"
            "impl Entry {\n"
            "    fn is_expired(&self, now: Instant) -> bool {\n"
            "        match self.expires_at {\n"
            "            Some(t) => t <= now,\n"
            "            None => false,\n"
            "        }\n"
            "    }\n"
            "}\n\n"
            "/// Snapshot of the store at a given version. Used for backups\n"
            "/// and point-in-time restore.\n"
            "#[derive(Debug, Clone)]\n"
            "pub struct Snapshot {\n"
            "    pub version: u64,\n"
            "    pub entries: HashMap<String, Entry>,\n"
            "}\n\n"
            "#[derive(Debug)]\n"
            "pub struct Store {\n"
            "    inner: Arc<Mutex<Inner>>,\n"
            "}\n\n"
            "#[derive(Debug)]\n"
            "struct Inner {\n"
            "    entries: HashMap<String, Entry>,\n"
            "    next_version: u64,\n"
            "    snapshots: Vec<Snapshot>,\n"
            "}\n\n"
            "impl Store {\n"
            "    pub fn new() -> Self {\n"
            "        Self {\n"
            "            inner: Arc::new(Mutex::new(Inner {\n"
            "                entries: HashMap::new(),\n"
            "                next_version: 1,\n"
            "                snapshots: Vec::new(),\n"
            "            })),\n"
            "        }\n"
            "    }\n\n"
            "    /// Insert or overwrite a key. If `ttl` is `Some(d)`, the\n"
            "    /// entry expires `d` after now. Returns the previous value\n"
            "    /// if the key existed and was not yet expired.\n"
            "    pub fn put(&self, key: &str, value: Vec<u8>, ttl: Option<Duration>)\n"
            "        -> Option<Vec<u8>>\n"
            "    {\n"
            "        // TODO: implement.\n"
            "        todo!()\n"
            "    }\n\n"
            "    /// Fetch a value if present and not expired.\n"
            "    pub fn get(&self, key: &str) -> Option<Vec<u8>> {\n"
            "        // TODO: implement.\n"
            "        todo!()\n"
            "    }\n\n"
            "    /// Remove a key. Returns whether anything was removed.\n"
            "    pub fn remove(&self, key: &str) -> bool {\n"
            "        // TODO: implement.\n"
            "        todo!()\n"
            "    }\n\n"
            "    /// Take a snapshot of the current live entries.\n"
            "    pub fn snapshot(&self) -> Snapshot {\n"
            "        // TODO: implement.\n"
            "        todo!()\n"
            "    }\n\n"
            "    /// Restore from a snapshot. Replaces the current contents.\n"
            "    pub fn restore(&self, snap: Snapshot) {\n"
            "        // TODO: implement.\n"
            "        todo!()\n"
            "    }\n\n"
            "    /// Sweep expired entries.\n"
            "    pub fn sweep(&self) -> usize {\n"
            "        // TODO: implement; return number of removed entries.\n"
            "        todo!()\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "Please complete the `todo!()` bodies below. Keep the locking "
            "scope tight, do not panic on missing keys, and make sure `sweep` "
            "is O(n) in the number of entries. After the implementation, add "
            "three unit tests that exercise: (1) put + get + ttl expiration, "
            "(2) snapshot then mutation then restore, (3) concurrent put from "
            "two threads via `Arc<Store>`. Write it now:\n\n"
        ),
    ),
    (
        "long_synth_xxl",
        # Real ~3K-token variant of the synth prompt (the 3000t label
        # on the original was nominal; it tokenised to ~986). Stitches
        # ~9 distinct domain paragraphs.
        (
            "Below are nine independent technical observations followed by a "
            "synthesis question.\n\n"
            "Part 1 — kernel scheduling. Linux's CFS uses virtual runtime on "
            "a red-black tree to enforce fairness without explicit priority "
            "queues. Energy-aware variants extend this with a per-CPU cost "
            "model for big.LITTLE topologies, picking placements that "
            "minimise energy under a deadline constraint. The cost model has "
            "to be cheap to evaluate — it runs on every scheduling decision "
            "— and accurate enough that wrong placements don't accumulate.\n\n"
            "Part 2 — memory allocators. jemalloc, tcmalloc, and mimalloc "
            "all use thread-local caches to avoid contention on the global "
            "heap, but they differ in how they reclaim. jemalloc returns "
            "memory to the OS aggressively via madvise(MADV_DONTNEED); "
            "tcmalloc batches reclamation; mimalloc uses a free-list approach "
            "with delayed eviction. Each strategy wins on different "
            "workloads: long-lived servers, batch jobs, and latency-sensitive "
            "services respectively.\n\n"
            "Part 3 — graph databases. Property graphs and RDF triples model "
            "the same underlying domain — entities with relationships — but "
            "make different tradeoffs. Property graphs index by node ID and "
            "let you embed properties directly on nodes and edges; RDF "
            "triples force everything into a uniform (subject, predicate, "
            "object) shape that's verbose but trivially federated across "
            "stores. Neither has solved the fundamental problem of query "
            "planning for highly-connected workloads.\n\n"
            "Part 4 — cryptography deployment. Cryptographic libraries are "
            "easy to call but hard to deploy correctly. The hard parts are "
            "key rotation, nonce management, downgrade resistance, and "
            "secret zeroisation. Libraries that handle these end-to-end "
            "(libsodium, age, signal) typically wrap lower-level primitives "
            "in opinionated APIs that refuse footguns. Libraries that "
            "expose primitives directly (OpenSSL, BoringSSL low-level) "
            "demand more discipline from callers and often see misuses.\n\n"
            "Part 5 — distributed tracing. Effective tracing requires "
            "consistent context propagation across processes, threads, and "
            "asynchronous boundaries. The OpenTelemetry SDK handles the "
            "first two reasonably; the third is where most tracing setups "
            "leak. A futures-based async runtime needs explicit context "
            "injection at every await point that crosses a task boundary, "
            "and getting this wrong leads to traces that look complete but "
            "actually drop half the work.\n\n"
            "Part 6 — compiler bootstrapping. A self-hosting compiler "
            "presents a chicken-and-egg problem: how do you build the "
            "compiler when it requires itself to compile? In practice the "
            "answer is to maintain a 'stage0' binary that's checked in or "
            "downloaded, build stage1 with stage0, build stage2 with stage1, "
            "and verify that stage2 is bit-for-bit identical to stage3 "
            "(stage2 building itself). The bit-identical check is the "
            "primary anti-tamper mechanism for software supply chains.\n\n"
            "Part 7 — solid-state physics tangent. Lithium-ion batteries "
            "degrade through several distinct mechanisms: SEI growth on the "
            "anode, cathode dissolution, lithium plating under fast "
            "charging, and electrolyte decomposition. The relative "
            "contribution of each depends on temperature, state of charge, "
            "and cycle rate. Battery-management systems trade off charging "
            "speed against expected lifetime; the optimal trade differs "
            "between a phone (where lifetime is 2-3 years) and a stationary "
            "grid battery (10+ years).\n\n"
            "Part 8 — academic publishing. Peer review is supposed to "
            "filter out bad work but in practice it filters mostly for "
            "novelty and surprise, not correctness. Replication crises in "
            "psychology, biomedicine, and parts of machine learning suggest "
            "that 30-50% of published results don't reproduce. The "
            "structural cause is straightforward: reviewers are unpaid, "
            "replicators are unrewarded, and journals have no financial "
            "incentive to publish corrections. Open-science platforms try "
            "to fix this with pre-registration, registered reports, and "
            "post-publication review, with mixed adoption.\n\n"
            "Part 9 — formal verification. Heavy-weight formal methods (Coq, "
            "Lean, Isabelle) prove correctness at enormous human cost; "
            "light-weight methods (TLA+, Alloy, model-checking) prove much "
            "less but are tractable for systems engineers. The pragmatic "
            "middle ground — refinement types, property-based testing, "
            "exhaustive small-scale enumeration — covers most real-world "
            "bugs without demanding full proofs. The cost-benefit curve is "
            "steep: getting from 0 to 80% confidence is cheap; getting from "
            "80% to 99.9% confidence costs orders of magnitude more.\n\n"
            "Synthesis question: pick the three parts above whose underlying "
            "concerns are most closely related and articulate the connection "
            "in two paragraphs. Then pick one part that looks unrelated to "
            "the rest and explain in one paragraph why it's actually a "
            "special case of one of the deeper themes you identified."
        ),
    ),
    (
        "long_synth_3000t",
        # Synthesize a long, varied prompt at ~3K tokens by stitching
        # paragraphs of distinct topics. Drafter has to handle a real
        # mid-context distribution shift.
        (
            "Below is a long discussion transcript followed by a question.\n\n"
            "Part A — software architecture\n"
            "When designing a high-throughput data pipeline, the natural "
            "instinct is to optimise for steady-state throughput, but the "
            "harder cases are usually the boundary conditions. A pipeline "
            "that handles a million events per second smoothly can fall over "
            "completely when the input rate drops to ten events per second, "
            "because batching heuristics tuned for the high regime stop "
            "filling their windows and latency spikes. Conversely, a system "
            "that handles low-rate input well can melt down under a sudden "
            "burst if its backpressure mechanism leaks unbounded work into a "
            "shared queue. The remedy is to design for both regimes from the "
            "start: enforce upper bounds on every queue, define explicit "
            "policies for what to do when bounds are hit, and instrument the "
            "boundary cases more heavily than the steady-state case because "
            "they will be the ones that surprise you.\n\n"
            "Part B — operating-systems trivia\n"
            "Linux's process scheduler has gone through several major "
            "rewrites. The O(1) scheduler from the 2.6 era used 140 priority "
            "queues and bitmap operations to find the next runnable task in "
            "constant time, but it had complex heuristics for distinguishing "
            "interactive tasks from batch ones, and those heuristics aged "
            "badly. The Completely Fair Scheduler that replaced it modelled "
            "the run queue as a red-black tree keyed by virtual runtime, "
            "abandoning explicit priorities in favour of a fairness invariant "
            "that emerges from the data structure. The current scheduler "
            "extends CFS with energy-aware placement on heterogeneous "
            "(big.LITTLE) topologies; the kernel now needs an explicit cost "
            "model for each CPU type and chooses placements that minimise "
            "energy under a deadline constraint.\n\n"
            "Part C — cooking observation\n"
            "Bread dough behaves like a non-Newtonian fluid in a way that's "
            "easy to overlook. Knead it slowly and it deforms plastically — "
            "you can stretch it without it springing back. Knead it quickly "
            "and the gluten network resists, behaving almost like a rubber. "
            "Bakers exploit both: slow folds during bulk fermentation build "
            "structure without tearing; faster shaping at the end of "
            "fermentation gives the loaf its final tension. The same dough, "
            "the same hands, different shear rates, different result.\n\n"
            "Part D — distributed-database failure modes\n"
            "Quorum-based consensus protocols are elegant when every node "
            "responds promptly, but most outages come from one specific "
            "failure mode: a node that's slow but not dead. If the leader "
            "waits for a quorum that includes a slow follower, every commit "
            "is bounded by that follower's latency. If the leader times the "
            "follower out, the follower may rejoin moments later and trigger "
            "leader election or log truncation. Production systems usually "
            "add layered timeouts and circuit breakers, and they aggressively "
            "prefer to drop slow followers from the quorum rather than wait "
            "for them. The cost is that recovery from a real network "
            "partition takes longer; the benefit is that ordinary latency "
            "spikes don't cascade into commit storms.\n\n"
            "Part E — physical-chemistry tangent\n"
            "Surface tension explains why a steel needle, denser than water, "
            "can float on the surface of a glass of water if you place it "
            "gently. The needle deforms the surface into a small depression, "
            "and the cohesive forces between water molecules in the surface "
            "layer support the needle's weight. A drop of detergent disrupts "
            "those forces and the needle sinks immediately. Astronauts "
            "exploit related phenomena to handle small amounts of fluid in "
            "microgravity, where surface tension dominates over the absent "
            "gravitational restoring force.\n\n"
            "Part F — game-design note\n"
            "Strategy games that reward optimisation often suffer from a "
            "phenomenon designers call 'analysis paralysis': as the optimal "
            "play becomes more computable, players spend longer per turn and "
            "the game's tempo collapses. Successful designs either limit "
            "information enough that the optimum is unknowable in reasonable "
            "time, or they introduce real-time pressure that prevents deep "
            "search, or they value variety so highly that the locally-"
            "optimal play is rarely the globally-optimal play. Each "
            "approach has different aesthetic consequences.\n\n"
            "Part G — software-supply-chain reflection\n"
            "Lockfiles in modern package managers do more than pin versions; "
            "they encode the resolved transitive dependency graph at a "
            "specific point in time, which means CI and local builds "
            "actually share a known artifact set. The downside is that "
            "lockfiles age silently — a project that hasn't updated its "
            "lockfile for a year is shipping a year-old security posture, "
            "and the friction of regenerating lockfiles often pushes teams "
            "to defer it until a vulnerability scanner forces the issue. "
            "The cure is automated lockfile-refresh PRs gated by a real "
            "test suite, but those require a test suite worth gating on.\n\n"
            "Question: summarise the seven parts above in one sentence each, "
            "preserving the substance and the author's voice. Then add a "
            "single concluding paragraph identifying the cross-cutting theme "
            "that ties at least four of the parts together. Be specific; "
            "do not retreat into generalities about 'design tradeoffs'."
        ),
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
