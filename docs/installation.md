# Installation

This guide targets a Linux x86_64 workstation with an RTX 5090-class Blackwell GPU.

## Requirements

- NVIDIA Blackwell GPU with compute capability compatible with SM120.
- NVIDIA driver new enough for CUDA 13.
- CUDA Toolkit 13.0 or newer, with `nvcc` and cuBLASLt installed.
- Rust 1.85+ with edition 2024 support.
- Python 3.10+ for helper scripts and optional Hugging Face tooling.
- Enough disk for the checkpoint and generated metadata.

The code currently assumes local CLI/library usage. It does not install a service, API server, or system daemon.

## System Packages

Ubuntu/Debian baseline:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  git \
  git-lfs \
  pkg-config \
  python3 \
  python3-pip
```

Enable Git LFS once:

```bash
git lfs install
```

## Rust

Install Rust through `rustup` if needed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup toolchain install 1.85
rustup component add rustfmt clippy
```

The repo includes `rust-toolchain.toml`, so `cargo` will use the pinned toolchain automatically.

## CUDA

Install CUDA Toolkit 13.x from NVIDIA packages for your distribution. Then set:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

Check:

```bash
nvidia-smi
nvcc --version
```

`nvcc --version` must report CUDA 13.x for the FP4/cuBLASLt path expected by this repo.

## Clone And Build

```bash
git clone <repo-url> qwen36-fp4
cd qwen36-fp4

cargo fetch
cargo check --workspace
```

Build CUDA kernels:

```bash
./scripts/build_cuda.sh
```

Expose the shared object to Rust and the dynamic linker:

```bash
export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$QWEN36_FP4_KERNEL_LIB_DIR:${LD_LIBRARY_PATH:-}"
```

Check the CUDA-enabled Rust bindings:

```bash
cargo check -p qwen36-fp4-kernels --features cuda
```

Run the CUDA smoke test:

```bash
./scripts/smoke_cuda.sh
```

## Model Download

Install Hugging Face CLI:

```bash
python3 -m pip install --user -U "huggingface_hub[cli]"
```

If the model requires authentication:

```bash
hf auth login
```

Download:

```bash
MODEL_DIR=/models/Qwen3.6-27B-Text-NVFP4-MTP
mkdir -p "$MODEL_DIR"

hf download sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP \
  --local-dir "$MODEL_DIR"
```

## First Commands

Inspect the HF config:

```bash
cargo run -p qwen36-fp4 -- inspect-config --model-dir "$MODEL_DIR"
```

Generate model metadata:

```bash
cargo run -p qwen36-fp4 -- discover \
  --model-dir "$MODEL_DIR" \
  --output model_layout.json
```

Validate weight bindings against the checkpoint:

```bash
cargo run -p qwen36-fp4 -- validate-weights --model-dir "$MODEL_DIR"
```

Validate CUDA runtime allocation and upload with the real checkpoint:

```bash
cargo run -p qwen36-fp4 --features cuda -- gpu-load \
  --model-dir "$MODEL_DIR" \
  --max-context 2256
```

Estimate KV/state budget:

```bash
cargo run -p qwen36-fp4 -- budget --ctx 32768 --kv fp8
```

Tokenize:

```bash
cargo run -p qwen36-fp4 -- tokenize \
  --model-dir "$MODEL_DIR" \
  --text "Bonjour"
```

## Environment Summary

Useful exports for `.envrc` or shell profile:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$QWEN36_FP4_KERNEL_LIB_DIR:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export QWEN36_FP4_SM=120
```
