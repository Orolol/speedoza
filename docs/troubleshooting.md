# Troubleshooting

## `cargo: command not found`

Install Rust through `rustup`, then reload the shell:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

## `nvcc: command not found`

Set `CUDA_HOME` and update `PATH`:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

If `nvcc` still fails, install the CUDA Toolkit package, not only the NVIDIA driver.

## `libqwen36_fp4_kernels.so: cannot open shared object file`

Build and export the library path:

```bash
./scripts/build_cuda.sh
export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$QWEN36_FP4_KERNEL_LIB_DIR:${LD_LIBRARY_PATH:-}"
```

## `CUDA_R_4F_E2M1` or scale-mode symbols are missing

The installed CUDA Toolkit is too old for the FP4 path. Use CUDA 13.x.

## cuBLASLt returns status 4 from `qwen36_nvfp4_gemm`

Likely causes:

- wrong FP4 packed layout
- wrong UE4M3 scale layout
- dimensions not aligned for the block-scaled FP4 kernel
- CUDA/cuBLASLt version mismatch
- running on a GPU without Blackwell FP4 support

First verify with a small known-good FP4 input, then test checkpoint tensors.

## `qwen36 discover` reports missing MTP or conv1d tensors

Do not continue to inference. Check:

- model directory points to `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
- all safetensors shards downloaded
- tensor naming has not changed upstream
- `config.json` matches the checkpoint

## Tokenizer loads but chat output looks wrong

The current tokenizer wrapper preserves the HF chat template if present, but prompt rendering is still a minimal Qwen-format renderer. Integrate exact Jinja rendering before using chat prompts as a correctness baseline.

## Smoke test passes but full inference fails

The smoke test validates ABI and basic kernel launches only. Full inference still needs:

- layer weight binding
- exact Gated DeltaNet math
- exact MTP head path
- real TurboQuant implementation
- final logits/sampling integration

