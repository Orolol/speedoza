# Research Notes

Verified against public sources on 2026-04-29.

Sources:

- https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP
- https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3.6-27B
- https://docs.nvidia.com/cuda/cublas/
- https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- https://docs.rs/safetensors/
- https://docs.rs/tokenizers/

## Model

- HF repo: `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`.
- Base model: `Qwen/Qwen3.6-27B`.
- Topology: 64 layers, repeated `[linear_attention, linear_attention, linear_attention, full_attention]`.
- Hidden size: 5120.
- Vocab size: 248320.
- FFN intermediate size: 17408.
- DeltaNet conv kernel: 4.
- Full-attention heads: 24 Q, 4 KV, head dim 256, partial RoPE factor 0.25.
- Linear attention heads: 16 QK, 48 V, head dim 128.
- MTP hidden layers: 1.
- Context length: 262144.

## Quantization

- Model card states `modelopt` NVFP4, `nvidia-modelopt` 0.43.0.
- Kept in bf16: `lm_head`, all `*linear_attn.conv1d*`, and the 15 `mtp.*` tensors.
- cuBLASLt FP4 block-scaled kernels require FP4 E2M1 values, `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`, TN layout, `CUBLAS_COMPUTE_32F`, and `CUDA_R_32F` scale type.

## Speculative Decoding

The model card recommends `num_speculative_tokens=3`; vLLM applies the single MTP layer recursively. `doc.md` specified one draft token per cycle, so the code defaults to 3 but supports any positive value.

## TurboQuant

Google Research describes TurboQuant as random-rotation/vector-quantization plus QJL residual correction for KV-cache compression. This codebase implements the model-specific skip policy and runtime contracts, but not the fused GPU attention kernel.
