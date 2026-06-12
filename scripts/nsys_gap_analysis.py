#!/usr/bin/env python3
"""Quantify GPU idle gaps in the MTP verify cycle from an nsys sqlite export.

The verify-cycle re-diagnosis (2026-06-12, DAILY) says the chunk-GEMM bucket
is latency/launch-structure-bound, not byte-bound. This script measures the
actual idle-gap budget: over the decode tail of a bench trace (last
--window-ms of kernel activity), it computes wall span vs busy time (union
of kernel intervals), the gap histogram, and per-kernel totals — the ceiling
for a verify-graph capture or call-fusion lever.

Usage: nsys_gap_analysis.py <trace.sqlite> [--window-ms 500]
"""
import argparse
import sqlite3
import sys
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("sqlite")
p.add_argument("--window-ms", type=float, default=500.0)
args = p.parse_args()

db = sqlite3.connect(args.sqlite)
db.row_factory = sqlite3.Row

rows = db.execute(
    """
    SELECT k.start AS s, k.end AS e, ids.value AS name
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds ids ON k.demangledName = ids.id
    ORDER BY k.start
    """
).fetchall()
if not rows:
    sys.exit("no kernel rows")

t_end = max(r["e"] for r in rows)
t_lo = t_end - args.window_ms * 1e6  # ns
win = [r for r in rows if r["s"] >= t_lo]
print(f"kernels in window: {len(win)} (of {len(rows)} total)")

# Union of busy intervals (all streams — single-stream engine).
busy = 0
gaps = []
cur_s, cur_e = win[0]["s"], win[0]["e"]
for r in win[1:]:
    if r["s"] <= cur_e:
        cur_e = max(cur_e, r["e"])
    else:
        busy += cur_e - cur_s
        gaps.append(r["s"] - cur_e)
        cur_s, cur_e = r["s"], r["e"]
busy += cur_e - cur_s
span = win[-1]["e"] - win[0]["s"]

print(f"span {span/1e6:.1f} ms | busy {busy/1e6:.1f} ms | "
      f"idle {(span-busy)/1e6:.1f} ms ({100*(span-busy)/span:.1f}%)")
print(f"gaps: {len(gaps)} | "
      f">2us: {sum(1 for g in gaps if g > 2000)} "
      f"({sum(g for g in gaps if g > 2000)/1e6:.2f} ms) | "
      f">5us: {sum(1 for g in gaps if g > 5000)} "
      f"({sum(g for g in gaps if g > 5000)/1e6:.2f} ms) | "
      f">20us: {sum(1 for g in gaps if g > 20000)} "
      f"({sum(g for g in gaps if g > 20000)/1e6:.2f} ms)")

agg = defaultdict(lambda: [0, 0])
for r in win:
    a = agg[r["name"]]
    a[0] += 1
    a[1] += r["e"] - r["s"]
print("\ntop kernels by total time in window:")
for name, (cnt, tot) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:14]:
    print(f"  {tot/1e6:8.2f} ms  x{cnt:6d}  avg {tot/cnt/1e3:7.1f} us  "
          f"{name[:90]}")
