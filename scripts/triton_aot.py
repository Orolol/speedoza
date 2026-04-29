#!/usr/bin/env python3
"""Entry point reserved for Triton AOT compilation.

The production path should compile prototype kernels to PTX and load them from
Rust through the CUDA driver API. This file is intentionally non-functional
until the concrete Triton kernels are added.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("target/triton"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raise SystemExit("no Triton kernels have been added yet")


if __name__ == "__main__":
    main()
