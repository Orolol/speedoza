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
- Call `qwen36_deltanet_decode` with gate/beta tensors enabled and compare against the Gated DeltaNet single-token recurrence.
- Call `qwen36_rmsnorm`, `qwen36_partial_rope`, `qwen36_swiglu`, and `qwen36_sample` with synthetic tensors and compare against CPU references.
- Call `qwen36_nvfp4_gemm` only after confirming packed FP4 and UE4M3 scale layout from the checkpoint loader; mismatched layout will produce wrong answers even if cuBLASLt succeeds.

## Real Checkpoint Validation

With the checkpoint downloaded, validate metadata and required tensor bindings:

```bash
MODEL_DIR=/models/Qwen3.6-27B-Text-NVFP4-MTP

cargo run -p qwen36-fp4 -- discover \
  --model-dir "$MODEL_DIR" \
  --output /tmp/qwen36_model_layout.json

cargo run -p qwen36-fp4 -- validate-weights \
  --model-dir "$MODEL_DIR"

cargo run -p qwen36-fp4 --features cuda -- gpu-load \
  --model-dir "$MODEL_DIR" \
  --max-context 2256

jq '.derived.warnings' /tmp/qwen36_model_layout.json
```

The expected manifest has 64 layers, 48 linear-attention layers, 16 full-attention layers, and 15 MTP tensors. `.derived.warnings` should be empty for the target checkpoint. `gpu-load` should report the `cuda` backend and the number of uploaded tensors before inference scheduling work begins.

## Known Gaps

- DeltaNet decode has an exact single-token recurrence path, but prefill, conv1d, and projection fusion are not final.
- TurboQuant path is int8 per-vector KV quantization, not full TurboQuant random rotation/QJL.
- Attention decode is correct-oriented and simple; it recomputes dot products and is not the final optimized kernel.
- End-to-end reference decode is wired through the runtime engine, but numerical parity and Tensor Core hot-path replacement are still pending.
