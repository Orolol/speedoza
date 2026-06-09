#!/usr/bin/env python3
"""DFlash drafter parity harness.

Loads `z-lab/Qwen3.6-27B-DFlash` via transformers, runs one drafter
forward on deterministic synthetic inputs, and dumps the inputs + the
expected output as fixture files that the Rust drafter-forward-smoke
can replay.

Inputs the Rust side cannot currently compute on its own (fc +
hidden_norm) are pre-applied here so the fixture's
`target_collapsed.bf16` matches what the Rust v1 forward expects.

Usage:
  /home/orosius/workspace/dmtp/.venv/bin/python \
      scripts/dflash_parity.py \
      --drafter-dir /home/orosius/models/Qwen3.6-27B-DFlash \
      --fixture-dir /tmp/dflash_fixture \
      --q-len 16 --ctx-len 16 --seed 424242
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafter-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--q-len", type=int, default=16)
    parser.add_argument("--ctx-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def save_bf16(path: Path, tensor: torch.Tensor) -> None:
    """Save a contiguous BF16 tensor as raw little-endian u16 bytes."""
    assert tensor.dtype == torch.bfloat16, f"{path}: expected bf16, got {tensor.dtype}"
    contig = tensor.detach().contiguous().cpu()
    raw = contig.view(torch.uint16).numpy().tobytes()
    path.write_bytes(raw)


def save_i32(path: Path, tensor: torch.Tensor) -> None:
    assert tensor.dtype == torch.int32
    contig = tensor.detach().contiguous().cpu()
    path.write_bytes(contig.numpy().tobytes())


def main() -> int:
    args = parse_args()
    args.fixture_dir.mkdir(parents=True, exist_ok=True)

    # Load drafter. trust_remote_code=True is required because the
    # DFlashDraftModel class lives in the model repo, not transformers.
    from transformers import AutoModel

    device = torch.device(args.device)
    print(f"loading drafter from {args.drafter_dir} on {device}", flush=True)
    model = AutoModel.from_pretrained(
        str(args.drafter_dir),
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    cfg = model.config
    hidden = cfg.hidden_size
    n_target = len(cfg.dflash_config["target_layer_ids"])
    print(
        f"drafter: {cfg.num_hidden_layers} layers, hidden={hidden}, "
        f"target_layer_ids={cfg.dflash_config['target_layer_ids']}, "
        f"block_size={cfg.block_size}",
        flush=True,
    )

    # Deterministic inputs. Seed both random generators we use.
    torch.manual_seed(args.seed)
    noise = torch.randn(1, args.q_len, hidden, dtype=torch.bfloat16, device=device) * 0.05
    target_raw = (
        torch.randn(1, args.ctx_len, hidden * n_target, dtype=torch.bfloat16, device=device)
        * 0.05
    )

    # Collapse target_hidden via the drafter's own fc + hidden_norm so
    # the fixture matches what the Rust v1 forward consumes.
    with torch.no_grad():
        target_collapsed = model.hidden_norm(model.fc(target_raw))

    # Position ids: absolute positions [0, ctx_len + q_len).
    kv_seq_len = args.ctx_len + args.q_len
    position_ids = torch.arange(kv_seq_len, dtype=torch.int64, device=device).unsqueeze(0)

    # Monkey-patch the drafter forward to skip the internal fc +
    # hidden_norm (we feed it the already-collapsed tensor). Simplest
    # patch: replace fc and hidden_norm with identity.
    original_fc = model.fc
    original_hidden_norm = model.hidden_norm
    model.fc = torch.nn.Identity()
    model.hidden_norm = torch.nn.Identity()

    # Forward. The drafter expects `target_hidden` to be already
    # collapsed-then-normalized (= what fc+hidden_norm would have
    # produced), so we pass target_collapsed and rely on the
    # identity-patched fc/hidden_norm to no-op.
    with torch.no_grad():
        output = model(
            position_ids=position_ids,
            noise_embedding=noise,
            target_hidden=target_collapsed,
            past_key_values=None,
            use_cache=False,
            attention_mask=None,
        )

    # Restore in case the user re-uses the model in the same process.
    model.fc = original_fc
    model.hidden_norm = original_hidden_norm

    print(f"output shape: {tuple(output.shape)} dtype: {output.dtype}", flush=True)
    assert output.shape == (1, args.q_len, hidden), output.shape

    # Sanity check on output finiteness.
    finite = torch.isfinite(output.float()).all().item()
    print(f"output finite: {finite}", flush=True)
    if not finite:
        print("ERROR: drafter output contains non-finite values", file=sys.stderr)
        return 1

    # Dump fixture. Layouts match the Rust workspace:
    #   noise.bf16:           [q_len, hidden]              row-major BF16
    #   target_collapsed.bf16:[ctx_len, hidden]            row-major BF16
    #   positions.i32:        [ctx_len + q_len]            i32
    #   expected_output.bf16: [q_len, hidden]              row-major BF16
    save_bf16(args.fixture_dir / "noise.bf16", noise.squeeze(0))
    save_bf16(args.fixture_dir / "target_collapsed.bf16", target_collapsed.squeeze(0))
    save_i32(
        args.fixture_dir / "positions.i32",
        position_ids.squeeze(0).to(torch.int32),
    )
    save_bf16(args.fixture_dir / "expected_output.bf16", output.squeeze(0))

    meta = {
        "drafter_dir": str(args.drafter_dir),
        "q_len": args.q_len,
        "ctx_len": args.ctx_len,
        "kv_seq_len": kv_seq_len,
        "hidden": hidden,
        "num_target_layers": n_target,
        "seed": args.seed,
        "noise_bytes": args.q_len * hidden * 2,
        "target_collapsed_bytes": args.ctx_len * hidden * 2,
        "positions_bytes": kv_seq_len * 4,
        "expected_output_bytes": args.q_len * hidden * 2,
    }
    (args.fixture_dir / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"fixture written to {args.fixture_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
