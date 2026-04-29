# Model Setup

Target checkpoint:

```text
sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP
```

The loader expects a local Hugging Face directory containing:

- `config.json`
- `tokenizer.json`
- optional `tokenizer_config.json` or `chat_template.jinja`
- one or more `.safetensors` files

## Download

```bash
MODEL_DIR=/models/Qwen3.6-27B-Text-NVFP4-MTP
huggingface-cli download sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False
```

## Discovery

Run:

```bash
cargo run -p qwen36-fp4 -- discover \
  --model-dir "$MODEL_DIR" \
  --output model_layout.json
```

The output confirms:

- `hidden_size=5120`
- 64 layers with `[linear_attention, linear_attention, linear_attention, full_attention]`
- full-attention global layers `{3, 7, 11, ..., 63}`
- TurboQuant skip layers `{3, 63}`
- DeltaNet conv kernel dimension `4`
- MTP hidden layer count `1`
- dtype and role distribution across tensors

## Useful Checks

```bash
jq '.topology' model_layout.json
jq '.derived.attention_layers' model_layout.json
jq '.derived.turboquant_skip_layers' model_layout.json
jq '.quantization' model_layout.json
jq '.derived.warnings' model_layout.json
```

Expected attention layers:

```json
[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]
```

Expected TurboQuant skip layers:

```json
[3,63]
```

## Tensor Roles

The discovery pass classifies tensors into:

- `nvfp4_packed_weight`
- `nvfp4_block_scale`
- `nvfp4_tensor_scale`
- `bf16_weight`
- `lm_head_bf16`
- `mtp_bf16`
- `conv1d_bf16`
- `embedding`
- `other`

If `conv1d_bf16` or `mtp_bf16` are missing, do not continue to full inference until the checkpoint naming/classification has been corrected.

## Local Directory Hygiene

Do not commit:

- checkpoint files
- `model_layout.json`
- generated `.so`, `.ptx`, `.cubin`, or benchmark artifacts

These are ignored by `.gitignore`.

