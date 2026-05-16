# qwen36-fp4

Single-stream inference engine for `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`, targeting RTX 5090 / Blackwell SM120.

This repo implements the project structure from [doc.md](doc.md): Rust host orchestration, safetensors discovery, tokenizer wrapper, runtime state planning, MTP rollback contracts, TurboQuant policy, CUDA ABI, and baseline CUDA kernels for bring-up.

## Status

Implemented:

- Rust workspace with `core`, `loader`, `tokenizer`, `kernels`, `runtime`, `mtp`, and `cli` crates.
- `qwen36 discover` path that mmaps `.safetensors` and writes `model_layout.json`.
- Zero-copy `MappedModel` tensor access for safetensors shards.
- HF `config.json` parser and Qwen3.6 topology validation.
- Runtime weight manifest validation for all layer, MTP, lm-head, embedding, and NVFP4 scale tensors.
- CUDA runtime memory ABI plus Rust RAII buffers for device allocation, copy, memset, and synchronization.
- Real-checkpoint GPU upload path for required manifest tensors and runtime buffers.
- Hybrid-aware TurboQuant attention skip policy: first and last full-attention layer.
- KV-cache and DeltaNet-state memory planning.
- CUDA shared-library ABI and baseline kernels for FP4 GEMM, attention decode, int8 KV quantization, quantized attention, RMSNorm, partial RoPE, SwiGLU, greedy sampling, and DeltaNet decode.
- MTP speculative controller with rollback/replay tests.

Not final yet:

- DeltaNet decode has an exact single-token recurrence path when gate/beta tensors are supplied, but conv/projection fusion and prefill are not final.
- TurboQuant is currently int8 per-vector KV quantization, not the full rotation/QJL implementation.
- End-to-end reference decode is wired through all layers, but it uses slow scalar CUDA matvecs and still needs numerical parity work against vLLM/Transformers.
- MTP speculative execution is not yet wired into the runtime scheduler.
- CUDA Graph capture and final hot-path tuning are pending.

## Quick Start

```bash
git clone <repo-url> qwen36-fp4
cd qwen36-fp4

rustup show
./scripts/build_cuda.sh
./scripts/smoke_cuda.sh

export QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda"
export LD_LIBRARY_PATH="$QWEN36_FP4_KERNEL_LIB_DIR:${LD_LIBRARY_PATH:-}"
cargo check --workspace --features qwen36-fp4-kernels/cuda
```

Download the model, then inspect it:

```bash
hf download sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP \
  --local-dir /models/Qwen3.6-27B-Text-NVFP4-MTP

cargo run -p qwen36-fp4 -- discover \
  --model-dir /models/Qwen3.6-27B-Text-NVFP4-MTP \
  --output model_layout.json

cargo run -p qwen36-fp4 -- validate-weights \
  --model-dir /models/Qwen3.6-27B-Text-NVFP4-MTP

cargo run -p qwen36-fp4 --features cuda -- gpu-load \
  --model-dir /models/Qwen3.6-27B-Text-NVFP4-MTP \
  --max-context 2256
```

## Documentation

- [Installation](docs/installation.md): full host setup for Rust, CUDA, model download, and first commands.
- [Model Setup](docs/model-setup.md): checkpoint layout, `model_layout.json`, and validation checks.
- [Kernel Validation](docs/kernel-validation.md): CUDA build, smoke tests, and numerical validation plan.
- [Repository Layout](docs/repo-layout.md): crate responsibilities and source tree.
- [Development](docs/development.md): local workflow, ABI rules, and PR checklist.
- [Troubleshooting](docs/troubleshooting.md): common build/runtime failures.
- [Roadmap](docs/roadmap.md): path from baseline kernels to final optimized inference.
- [Research Notes](docs/research.md): verified external assumptions and links.

## Core Commands

```bash
cargo run -p qwen36-fp4 -- inspect-config --model-dir /path/to/model
cargo run -p qwen36-fp4 -- budget --ctx 32768 --kv fp8
cargo run -p qwen36-fp4 -- tokenize --model-dir /path/to/model --text "Bonjour"
cargo run -p qwen36-fp4 -- validate-weights --model-dir /path/to/model
cargo run -p qwen36-fp4 --features cuda -- gpu-load --model-dir /path/to/model --max-context 2256
```

## CUDA Runtime Knobs

- `QWEN36_KV_CACHE_DTYPE=tq35` selects the TurboQuant 3.5-bit KV cache.
- `QWEN36_LONG_CONTEXT_MODE` overrides the automatic long-context policy. When unset, contexts at or above 8192 tokens disable the large fused weight stores to save VRAM; set `QWEN36_LONG_CONTEXT_MODE=0` to force the fused stores back on, or `=1` to force long-context mode on smaller runs.
- `QWEN36_LONG_CONTEXT_AUTO_MIN_CONTEXT=<tokens>` changes the automatic threshold.
- `QWEN36_PREFILL_SPLIT_MAX_TOKENS=<tokens>` raises the opt-in split-KV prefill chunk limit for long-context experiments. The default remains tuned for short MTP verify chunks.
- `QWEN36_PREFILL_SPLIT_MIN_SPLITS=<count>` changes the split-KV prefill activation threshold when the max-token override is used.
- `QWEN36_DECODE_ATTENTION_BUCKET_MIN_CONTEXT=<tokens>` changes the minimum active-context bucket used to size decode split-KV launches. The default is 8192, so a run reserving 262K context no longer launches 262K-sized decode attention splits while decoding near 8K.
- `QWEN36_DECODE_ATTENTION_BUCKET_DISABLE=1` restores the older behavior where decode split-KV launch shapes are sized from the configured `max_context`.
- `QWEN36_MTP_ASSUME_ACCEPT=1` skips per-draft verification checks in the multi-token MTP graph path. It also skips recurrent snapshots for that path, so use it only when rejection is known to be impossible or acceptable for the run.
- `QWEN36_PROFILE_PREFILL_CHUNKS=1` prints per-prefill-chunk timing buckets for profiling. It synchronizes between buckets and should not be used for throughput numbers.

## License

Apache-2.0. See [LICENSE](LICENSE).
