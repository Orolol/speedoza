# `qwen36-fp4` — Inference Engine Spec

> Single-stream inference engine for **Qwen3.6-27B-Text-NVFP4-MTP** on **NVIDIA RTX 5090** (Blackwell SM120). From-scratch implementation, no compromise on the hot path.

---

## 0. Objectives

### Primary
Maximize tokens/sec in **single-stream (batch=1) decode** for `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` on RTX 5090, with:
- **MTP speculative decoding** functional (with state rollback for DeltaNet layers)
- **TurboQuant** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) for the KV cache of the 16 attention layers
- **NVFP4** native compute path via cuBLASLt on SM120

**Target**: ≥ 1.5× the throughput of vLLM (same model, same hardware, same context length).

### Secondary
Fast prefill for long contexts up to 262K tokens (native, no YaRN initially).

### Non-objectives — explicitly rejected
- ❌ Multi-batch / continuous batching / serving (vLLM exists)
- ❌ Multi-model / multi-architecture / dynamic dispatch
- ❌ Multi-GPU
- ❌ Vision tower (we use the `-Text-` variant which has it stripped)
- ❌ Training, fine-tuning, gradient computation
- ❌ Quantization formats other than NVFP4 (E2M1, micro-block 16, scale E4M3 + scale FP32 per-tensor)
- ❌ Custom tokenizer (use HF `tokenizers` Rust crate)
- ❌ HTTP / OpenAI-compatible API (CLI + library only)

---

## 1. Model topology (verified)

```
Qwen3.6-27B (27.78B params, hybrid linear-attn + full-attn)
├── 64 layers organized into 16 macro-blocks of 4 layers each
│   └── Pattern per macro-block: [DeltaNet, DeltaNet, DeltaNet, GatedAttention]
│       each sublayer followed by a SwiGLU FFN (intermediate=17408)
├── Vocab: 248K tokens
├── 1 MTP head (mtp_num_hidden_layers=1) — kept in bf16
└── Hidden dim: TBD at load (likely 5120, confirm against Qwen3-Next-80B-A3B)
```

### 48 DeltaNet layers (linear attention)
- **Gated DeltaNet** with delta-rule + scalar gate ([Yang et al. 2024](https://arxiv.org/abs/2412.06464))
- **Includes conv1d** (Mamba-style short causal conv, kernel size ~4) — confirmed by model card via `*linear_attn.conv1d*` in the quantization ignore list
- 48 V heads, 16 QK heads, head_dim 128
- **State**: matrix `S ∈ R^{48 × 128 × 128}` per layer (bf16) + small conv1d history buffer (~3 tokens)
- **No KV cache**: the recurrent state replaces it
- conv1d weights stay in **bf16** (not NVFP4)

### 16 Gated Attention layers (full softmax attention)
- GQA: 24 Q heads, 4 KV heads, head_dim 256
- **Partial RoPE on 64 dims** (not on the full 256)
- **Output gate** (sigmoid on a projection)
- KV cache: standard `(K, V) ∈ R^{2 × max_ctx × 4 × 256}` per layer

### MTP head
- 1 layer kept in bf16
- 15 tensors total, ~850 MB

### Critical implication for TurboQuant
The vLLM heuristic "skip first/last 2 layers" for numerical stability is **broken on hybrid models** (`vllm/engine/arg_utils.py:1652` rejects them outright).

In `Qwen3.6-27B`, attention layers are at global indices `{3, 7, 11, ..., 63}`. We **redefine the skip policy as "skip first and last attention layer"**, i.e. skip global layers `{3, 63}`. See §6.

---

## 2. Storage and loading

### NVFP4 modelopt format on disk
The checkpoint exposes:
- **NVFP4 weights**: `uint8` tensors (2 E2M1 values packed per byte) + `weight_scale` (FP8 E4M3, one per 16-element micro-block) + `weight_scale_2` (FP32 per-tensor)
- **bf16 weights** (from quantization ignore list):
  - `lm_head`
  - `mtp.*` (15 tensors, ~850 MB)
  - `*linear_attn.conv1d*` (48 conv1d, one per DeltaNet layer)
  - Embeddings (probable, confirm at load)
- **Activations**: dynamically NVFP4-quantized at runtime (not pre-quantized)

References:
- NVFP4 spec: [NVIDIA blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- modelopt format: [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)

### Layout pipeline disk → RAM → VRAM
1. `mmap` of safetensors files (zero-copy CPU)
2. Pre-allocate one contiguous GPU buffer per layer (reduces fragmentation)
3. **Never decompress NVFP4 ahead of time** — keep it packed in VRAM, dequant is fused inside the matmul kernels via cuBLASLt

### VRAM budget (target: ctx=32K)
| Component | Size |
|---|---|
| NVFP4 weights | ~14 GB |
| bf16 weights (MTP + 48× conv1d + lm_head) | ~850 MB |
| KV cache 16 attn layers, ctx=32K, FP8 (TurboQuant) | ~1 GB |
| DeltaNet state (48 layers) | ~75 MB |
| Activations + workspace | ~2 GB |
| **Total** | **~18 GB** of 32 GB |

→ comfortably fits 128K context, 262K achievable with more aggressive KV quant.

---

## 3. Code architecture

```
qwen36-fp4/
├── crates/
│   ├── core/              # types, layout descriptors, traits
│   ├── loader/            # safetensors mmap + GPU upload
│   ├── tokenizer/         # wrapper around HF `tokenizers` crate
│   ├── kernels/           # CUDA kernels + Rust bindings
│   │   ├── nvfp4_gemm/    # NVFP4 × bf16 → bf16 GEMM via cuBLASLt
│   │   ├── deltanet/      # chunkwise prefill + recurrent decode
│   │   ├── attention/     # FA3 binding + custom decode kernel
│   │   ├── turboquant/    # KV cache quant + Q×Kquant matmul
│   │   ├── rmsnorm/       # fused RMSNorm + residual
│   │   ├── rope/          # partial RoPE (64 dims of 256)
│   │   ├── swiglu/        # fused SwiGLU activation
│   │   └── sampling/      # fused top-k / top-p / temp / penalty
│   ├── runtime/           # CUDA graph capture, KV cache mgmt, DeltaNet state
│   ├── mtp/               # speculative decoding with rollback
│   └── cli/               # binary entrypoint
├── kernels-cuda/          # raw .cu sources compiled to .so
└── benches/               # criterion benches vs vLLM
```

---

## 4. Phased roadmap

### Phase 0 — Discovery prototype (day 1)
Before writing anything structural: a 200-line Rust binary that:
- mmaps the safetensors of `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
- lists every tensor with its name, shape, dtype
- confirms: hidden_dim, conv1d kernel_size, RoPE base, exact ignore list, position of `mtp.*` tensors
- prints the layer-by-layer dtype distribution (which layers are NVFP4 vs bf16)

**Deliverable**: a `model_layout.json` confirming all assumptions in §1 before phase 1.

### Phase 1 — Reference forward pass, unoptimized (week 1–2)
Goal: generate 10 coherent tokens, verify correctness against `transformers` HF.

- [ ] safetensors NVFP4 modelopt loader
- [ ] Tokenizer wrapper (Qwen3.6 chat template included)
- [ ] Naive forward path via cuBLASLt:
  - NVFP4×bf16 GEMM via `cublasLtMatmul` with `CUBLASLT_MATMUL_DESC_FP4_SCALE_MODE` (Blackwell native path)
  - **Reference DeltaNet**: temporarily reuse [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) via Python FFI — lets us debug correctness before writing custom kernels
  - **Reference attention**: Flash Attention 3 binding (used as-is, already optimal for prefill)
  - RMSNorm / RoPE / SwiGLU: trivial Triton kernels
- [ ] Minimal sampling (greedy + top-p)
- [ ] **Numerical equality test** with `transformers` to ε precision over 100 prompts

**Deliverable**: `qwen36 chat "..."` works, ~5–10 tok/s. Not impressive, but correct.

### Phase 2 — Custom DeltaNet kernels (week 3–4)
The hottest path least well served by existing libraries. Reference: [Gated DeltaNet paper (Yang et al. 2024)](https://arxiv.org/abs/2412.06464) + [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) source. Specific innovations for our case:

1. **Fused chunkwise prefill kernel**: a single Triton kernel per layer doing `RMSNorm → conv1d → QKV proj NVFP4 → chunkwise delta-rule → output gate → output proj NVFP4`. Chunk size = 64 (sweet spot for SM120 shared-mem). The state matrix is held in SRAM between chunks.

2. **Sequential decode kernel**: for batch=1, DeltaNet is **fully memory-bound** (a 128×128 mat-vec per head + a rank-1 update). Persistent kernel that loops over tokens within a single launch when in spec decode (1 MTP draft + verify).

3. **State checkpoint for MTP rollback**: before each MTP draft batch, snapshot the 48 DeltaNet state matrices into a dedicated buffer (~75 MB). On rejection, restore from snapshot and replay the K accepted tokens. Cost: a 75 MB GPU→GPU memcpy = ~40 µs at 5090 bandwidth. Negligible.

**Deliverable**: DeltaNet kernel 1.3–1.5× faster than `flash-linear-attention` on this specific shape.

### Phase 3 — TurboQuant for attention KV cache (week 5)
**Required reading first**: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — TurboQuant.

Implementation for the 16 attention layers only:
- K quantization in FP8 E4M3 with Hadamard rotation (TurboQuant's key trick)
- V quantization in FP8 or INT4 — pick based on FamilyBench validation
- **Hybrid-aware skip policy**: skip *attention layers* at positions `{first, last}` in the **attention layer sequence**, not in the global 0–63 index. For Qwen3.6-27B with pattern `[D D D A]×16`, attention layers are at global indices `{3, 7, 11, ..., 63}` → skip = `{3, 63}`.
- **Fused Q×Kquant matmul**: a single Triton kernel doing dequant → matmul → softmax → V dequant → matmul, keeping intermediates in SRAM. Inspired by FA3 but with custom FP8 code path.

**Deliverable**: KV cache halved vs FP8 baseline, quartered vs BF16. Correctness validated against FamilyBench (used as regression test).

### Phase 4 — MTP speculative decoding (week 6)
Per the model card: `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'`. Single MTP layer → drafts **1 token per cycle**.

Algorithm per step:
1. Forward pass `[main + MTP]` on the last accepted token → produces `logit_main` and `logit_mtp_draft`
2. Sample `t_main` from `logit_main`, sample `t_draft` from `logit_mtp_draft`
3. Forward main-only on `t_main` with **DeltaNet state snapshot taken** → produces `logit_verify`
4. If `argmax(logit_verify) == t_draft` (greedy) or via probabilistic rejection sampling, **accept and advance 2 tokens**. Otherwise advance 1 token.

With ~70% acceptance rate on code (realistic per existing MTP benchmarks), this gains ~1.5× on pure decode. Combined with other optimizations, this is where we build our lead over vLLM.

### Phase 5 — CUDA Graph + final tuning (week 7)
- Capture the full decode loop into CUDA graphs (one per typical KV cache size: 1K, 4K, 16K, 64K, 256K). Eliminates ~30 µs of launch overhead per token.
- Profiling with Nsight Compute, fix kernels not saturating bandwidth.
- Tune DeltaNet chunk sizes and attention block sizes via grid search on actual hardware.
- Fused sampling kernel (single launch for temp + top-k + top-p + penalty + multinomial).

**Final deliverable**: binary running Qwen3.6-27B NVFP4 + MTP + TurboQuant at **>2× vLLM stock** on RTX 5090.

---

## 5. Language and toolchain

**Choice**: **Rust** for host orchestration + **CUDA C++** for custom kernels + **Triton (Python)** for prototype/exotic kernels.

### Rationale
1. **Not pure Python**: Python+PyTorch launch overhead on SM120 is 20–50 µs per op. With ~200 ops per decode token, that's 4–10 ms of pure host overhead per token. Killer for >150 tok/s targets.

2. **Not pure C++**: modern tooling (cargo, reproducible builds, safe FFI) and the type system for GPU memory management are clearly superior in Rust. ~5000 lines of host orchestration to write — no reason to suffer.

3. **Why Rust specifically over Zig or others**:
   - [`cudarc`](https://crates.io/crates/cudarc) is mature, zero-cost CUDA driver bindings, used in production (Candle, llm-rs)
   - `safetensors` has an excellent native crate
   - HF `tokenizers` is **natively written in Rust** — direct API, no FFI

4. **Why Triton for exotic kernels**: writing the DeltaNet chunkwise directly in CUDA C++ costs an extra 3 weeks. Triton lets us iterate in hours. Always possible to rewrite hot paths in CUDA C++ post-validation where Triton leaves 10–15% on the table (typically the persistent decode kernel). For standard NVFP4 GEMMs, cuBLASLt is unbeatable anyway.

5. **Not Mojo**: not mature enough for Blackwell, ecosystem still tiny, compiler still has SM120 bugs. Maybe in 2 years.

### Concrete stack
- **Rust 1.85+** with edition 2024
- **`cudarc` 0.13+** for driver / cuBLAS / cuDNN bindings
- **`safetensors`** crate
- **`tokenizers`** crate (HF, native Rust)
- **`pyo3`** to invoke Triton kernels compiled ahead-of-time (`triton.compile()` produces PTX, loaded in Rust via `cudarc`)
- **CUDA 13.0** ⚠️ **NOT 13.2** (produces gibberish on Qwen3.6 per Unsloth docs)
- Build pipeline: `cargo` + `nvcc` + Triton AOT compile script

---

## 6. Identified technical risks

1. **conv1d in DeltaNet** — not described in most Gated DeltaNet writeups but present in the checkpoint. Likely a short conv (kernel size 4) Mamba-style. **Verify in `config.json` of the HF repo before Phase 1.**

2. **TurboQuant + DeltaNet rollback** — if K/V quantization noise diverges between draft and verify, MTP acceptance rate drops. **Mitigation**: Phase 4 includes a "TurboQuant off for verify-side attention layers" fallback variant for ablation.

3. **CUDA Graph + state mutation** — CUDA graphs don't tolerate pointer mutation well. For the growing KV cache, use a pre-allocated `max_ctx` buffer with masking. For the DeltaNet state (fixed size), no problem.

4. **NVFP4 GEMM correctness on SM120** — early cuBLASLt implementations on SM120 had bugs through CUDA 12.9. **Must be on CUDA 13.0+**, with systematic numerical regression tests.

5. **conv1d quantization** — model card keeps it in bf16. So the 48 DeltaNet conv1d remain a separate bf16 kernel. Not a problem, but **don't accidentally quantize them**.

---

## 7. Validation plan

### Numerical correctness
- Reference: `transformers` HF on the bf16 unquantized base model
- Tolerance: ε = 1e-2 on logits (NVFP4 introduces visible noise; we measure against quantized baseline too)
- Suite: 100 prompts, ranging over short/long, code/text, single/multi-turn

### Behavioral regression
- **FamilyBench / TreeEval** as primary regression test (we already trust it)
- 9 question types, up to 10 generations / 1000 people
- Run after each phase, compare against vLLM baseline on the same model

### Performance benchmarks
- Single-stream decode tok/s at contexts: 256, 1K, 4K, 16K, 64K, 128K
- Prefill throughput at the same contexts
- **Baseline**: vLLM 0.19+ with the same NVFP4-MTP model and `--speculative-config qwen3_5_mtp`
- Reporting: criterion benches with regression tracking across commits

---

## 8. Reference resources

- **Base model**: [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
- **Quantized model used**: [sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP](https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP)
- **Gated DeltaNet paper**: [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- **TurboQuant paper**: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **Reference DeltaNet impl**: [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- **NVFP4 specification**: [NVIDIA Technical Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- **modelopt quantizer**: [NVIDIA/TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- **vLLM hybrid TurboQuant patch**: [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches) (reference for the skip-policy bug, not used directly)

---

## 9. Suggested execution order

1. **Phase 0 first** — do not skip. Confirms every assumption listed in §1 against the actual checkpoint. ~1 day.
2. **Phase 1** — reference forward, focus exclusively on correctness, not performance. ~2 weeks.
3. **Phases 2–4** — can be parallelized across multiple Claude Code sessions if convenient (DeltaNet kernels and TurboQuant are mostly independent).
4. **Phase 5** — final integration, no new features, only profiling and tuning.

**Total wall-clock estimate**: 7 weeks of focused work for a single experienced developer. Significantly faster if Phases 2 and 3 are parallelized.