# Repository Layout

```text
qwen36-fp4/
├── crates/
│   ├── core/       # topology, dtype, tensor classification, budget models
│   ├── loader/     # config parsing, safetensors mmap, model_layout.json
│   ├── tokenizer/  # HF tokenizers wrapper and Qwen chat rendering
│   ├── kernels/    # Rust kernel traits, CUDA FFI, no-cuda backend
│   ├── runtime/    # KV cache, DeltaNet state, CUDA graph plans, engine shell
│   ├── mtp/        # speculative decoding controller and rollback contract
│   └── cli/        # qwen36 binary
├── kernels-cuda/   # CUDA sources and public C ABI
├── scripts/        # CUDA build and smoke-test helpers
├── benches/        # benchmark placeholders
├── docs/           # operational documentation
└── doc.md          # original implementation specification
```

## Crate Responsibilities

`qwen36-fp4-core`

- Owns stable model facts and derived layout logic.
- Validates the Qwen3.6 hybrid layer pattern.
- Classifies tensors and estimates memory budgets.

`qwen36-fp4-loader`

- Reads `config.json`.
- Discovers `.safetensors` metadata and provides zero-copy `MappedModel` tensor views.
- Emits `model_layout.json`.

`qwen36-fp4-tokenizer`

- Loads `tokenizer.json`.
- Provides encode/decode and minimal Qwen chat prompt rendering.

`qwen36-fp4-kernels`

- Defines Rust-side kernel specs.
- Exposes `NoCudaBackend` and optional `CudaBackend`.
- Mirrors the C ABI in `kernels-cuda/include/qwen36_fp4.h`.

`qwen36-fp4-runtime`

- Plans KV cache allocation.
- Plans DeltaNet recurrent state and checkpoint buffers.
- Builds the per-layer model weight manifest from discovered checkpoint tensors.
- Uploads required checkpoint tensors and runtime buffers to CUDA when built with the `cuda` feature.
- Provides the engine shell plus the current CUDA reference prefill/decode scheduler.

`qwen36-fp4-mtp`

- Implements the speculative decode controller.
- Requires runtime hooks for snapshot, restore, replay, main forward, and MTP forward.

`qwen36-fp4`

- CLI entrypoint: discovery, config inspection, budget estimation, tokenization, weight validation, CUDA GPU-load validation, and chat path.

## ABI Rule

Any field added to `kernels-cuda/include/qwen36_fp4.h` must be mirrored in:

- `crates/kernels/src/backend.rs`
- the relevant Rust spec module under `crates/kernels/src/`
- the smoke test if the new field is required

Do not change ABI layout casually. Prefer adding fields at the end of structs while the ABI is still evolving.
