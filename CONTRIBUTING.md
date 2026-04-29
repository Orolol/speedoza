# Contributing

This repo is performance-sensitive and hardware-specific. Changes should prefer explicit failures over silent fallback behavior.

## Baseline Workflow

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
./scripts/build_cuda.sh
./scripts/smoke_cuda.sh
```

If CUDA is available:

```bash
export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$QWEN36_FP4_KERNEL_LIB_DIR:${LD_LIBRARY_PATH:-}"
cargo test --workspace --features qwen36-fp4-kernels/cuda
```

## Rules

- Do not add fake inference paths that return plausible tokens without running the real model path.
- Keep Rust/CUDA ABI changes synchronized between `kernels-cuda/include/qwen36_fp4.h` and `crates/kernels/src/backend.rs`.
- Keep model assumptions discoverable through `qwen36 discover`; avoid burying checkpoint facts in comments only.
- Add a validation note when changing kernel math, quantization layout, or recurrent state layout.
- Prefer correctness tests first, then performance work.

