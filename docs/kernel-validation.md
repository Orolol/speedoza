# Kernel Validation Plan

Run on the RTX 5090 host with CUDA 13.0+.

## Build

```bash
./scripts/build_cuda.sh
export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$QWEN36_FP4_KERNEL_LIB_DIR:${LD_LIBRARY_PATH:-}"
cargo check --workspace --features qwen36-fp4-kernels/cuda
./scripts/smoke_cuda.sh
```

## First Smoke Tests

- Call `qwen36_attention_decode` with small synthetic BF16 tensors and compare against a CPU/PyTorch reference for one token.
- Call `qwen36_turboquant_encode_kv` then `qwen36_turboquant_attention`; compare against BF16 attention with expected quantization tolerance.
- Call `qwen36_deltanet_decode` on synthetic Q/K/V/state and compare against the recurrence `S = decay*S + update_scale*v*k^T`, `y = S*q`.
- Call `qwen36_nvfp4_gemm` only after confirming packed FP4 and UE4M3 scale layout from the checkpoint loader; mismatched layout will produce wrong answers even if cuBLASLt succeeds.

## Known Gaps

- DeltaNet is a bring-up recurrence, not the exact Gated DeltaNet kernel.
- TurboQuant path is int8 per-vector KV quantization, not full TurboQuant random rotation/QJL.
- Attention decode is correct-oriented and simple; it recomputes dot products and is not the final optimized kernel.
