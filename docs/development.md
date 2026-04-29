# Development

## Local Workflow

Without CUDA:

```bash
cargo fmt --all
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

With CUDA:

```bash
./scripts/build_cuda.sh
export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$QWEN36_FP4_KERNEL_LIB_DIR:${LD_LIBRARY_PATH:-}"
./scripts/smoke_cuda.sh
cargo test --workspace --features qwen36-fp4-kernels/cuda
cargo clippy --workspace --features qwen36-fp4-kernels/cuda -- -D warnings
```

## Kernel Development Loop

1. Add or change the spec in `kernels-cuda/include/qwen36_fp4.h`.
2. Mirror it in `crates/kernels/src/backend.rs`.
3. Update the typed Rust spec in `crates/kernels/src/`.
4. Implement the CUDA kernel.
5. Add smoke coverage in `kernels-cuda/smoke.cu`.
6. Compare against a CPU/PyTorch reference before optimizing.
7. Profile with Nsight Compute on the target 5090.

## Correctness Policy

- Prefer explicit `UnsupportedNoCuda` errors over mock behavior.
- Do not return generated tokens from incomplete inference paths.
- Keep checkpoint-derived assumptions in `qwen36 discover` output.
- Treat `model_layout.json` as the first debugging artifact for any model mismatch.

## Formatting

Rust:

```bash
cargo fmt --all
```

CUDA formatting is manual for now. Keep C ABI structs simple and `repr(C)` friendly.

## Performance Work

Correctness comes first. For each hot-path optimization, capture:

- benchmark shape
- context length
- kernel launch count
- achieved bandwidth/Tensor Core utilization
- numerical tolerance versus reference
- regression risk

## Pull Request Checklist

- `cargo fmt --all`
- `cargo test --workspace`
- `./scripts/build_cuda.sh`
- `./scripts/smoke_cuda.sh`
- `cargo test --workspace --features qwen36-fp4-kernels/cuda`
- `cargo clippy --workspace --features qwen36-fp4-kernels/cuda -- -D warnings`
- `model_layout.json` regenerated if model metadata logic changed
- docs updated if commands, environment variables, or ABI changed
