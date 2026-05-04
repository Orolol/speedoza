# Direction B: Custom NVFP4 gemv kernel (Marlin-style for Blackwell SM_120)

**Date:** 2026-05-04
**Status:** Design — pending user approval after context reset
**Depends on:** main HEAD `4ba622e` (post tree-MTP Phase 1 merge)
**Estimated effort:** 3–4 weeks of focused CUDA work, ~5 distinct sub-deliverables

## 1. Goal and motivation

Replace the cuBLASLt FP4 GEMM call for the **decode-time projection layers** (gate/up/down, q/k/v/o, in_proj_*) with a hand-written CUDA kernel optimised for `(M, N=1, K)` shapes.

Today every decode token funnels through ~14 NVFP4 GEMMs per layer × 64 layers = ~900 GEMM launches, each with `N = 1` (single-stream, single-token decode). cuBLASLt's NVFP4 GEMM is tuned for `N ≥ 8` (per `CUBLASLT_MATMUL_TILE_*` minimum tile sizes); at `N = 1` it pads `N` and wastes ~87 % of the tensor-core throughput on the padding columns. A purpose-built gemv kernel keeps the entire MMA pipeline busy on the single output column.

**Bench projection (best case):**

| Shape regime | cuBLASLt today | Marlin-style gemv (target) | Speedup |
|---|---|---|---|
| Linear-projection layers @ N=1 | ~baseline | ~3-4× theoretical, ~2× realistic | +50-100% on those layers |
| Decode end-to-end | MTP=4 = 123 tok/s | MTP=4 = 145-180 tok/s | +18-46% |

The kernel runs alongside everything else (chain MTP, tree-MTP infra, future PDL chains) — gains are **multiplicative** with PR #2 / future spec-decoding work, not in conflict.

**Why this is more attractive than further tree-MTP investment:** the bench result on Phase 1 (`AGENT.md` 2026-05-04 section) confirmed tree-MTP K>1 is ~3× slower than chain MTP on this hardware because per-leaf forward overhead dominates. Phase 2 batched-leaf work would claw back maybe 10-30 % over chain MTP at best. NVFP4 gemv attacks the **GEMM bottleneck itself** which is the largest single decode cost (`linear_attn` ~9 ms + `mlp` ~14 ms per token in the WSL2 profile, mostly NVFP4 GEMM).

## 2. Quality contract

**Hard parity (regression gate, blocks merge):**
- `chat --prompt "hello" --max-new-tokens 12` produces identical token streams for `--mtp-speculative-tokens` ∈ {0, 1, 2, 3, 4} with the gemv path enabled vs. cuBLASLt path.
- `chat --prompt "hello world" --max-new-tokens 12` same.

**Soft parity:** the existing borderline-prompt 1–2 token drift envelope must not widen. Op-level cos sim ≥ 0.998 against the cuBLASLt reference at every layer.

**Op-level parity gate:**
- Cos sim ≥ 0.998 between cuBLASLt and gemv outputs for every shape exercised in the decode path. Must hold for **all 64 layers' gate / up / down / qkv / out_proj** and the **MTP head's** gate / up / down.
- The existing parity harness (`scripts/decode_parity.py` + `QWEN36_DEBUG_DUMP_DIR`) extends naturally — add an env var to force the gemv path for one layer at a time and diff against cuBLASLt.

**Soft fallback:** if the gemv kernel returns `QWEN36_STATUS_NOT_IMPLEMENTED` for an unsupported shape, the runtime transparently falls back to cuBLASLt (mirror the Mirage megakernel pattern in `crates/kernels/src/backend.rs`). This lets the kernel ship shape-by-shape.

## 3. Hardware target

- RTX 5090, Blackwell SM_120.
- Compute capability 12.0 → MMA atom `m16n8k64.kind::mxf4` (PTX ISA 8.7 §9.7.14).
- 5th-gen Tensor Cores with FP4 microscaling (block_size = 16).
- 32 GB HBM3, 128 MB L2, 192 SMs.
- TMA (Tensor Memory Accelerator) with multicast support (`cp.async.bulk.tensor.*.multicast`).
- Thread block clusters with DSMEM (256 KB shared across cluster CTAs).
- WSL2 environment: ~3 µs per kernel launch overhead.

## 4. Existing GEMM call patterns (decode path)

Profile output (`QWEN36_PROFILE_DECODE_LAYERS=1`) lists the NVFP4 GEMMs invoked per decode token:

| Layer block | Op | Shape (M, K) | Notes |
|---|---|---|---|
| Linear attention (×48) | `in_proj_qkv` | (10240, 5120) | Pre-fused with `_b`/`_a`/`_z` → combined M=16640 in `LinearAttnInProjFusedStore` |
| Linear attention (×48) | `out_proj` | (5120, value_dim≈3584) | Independent |
| Full attention (×16) | `q_proj` | (8192, 5120) | Includes gate dim |
| Full attention (×16) | `k_proj` | (1024, 5120) | GQA, kv_heads=4 |
| Full attention (×16) | `v_proj` | (1024, 5120) | Same |
| Full attention (×16) | `o_proj` | (5120, 6144) | head_dim × num_q_heads |
| MLP (×64) | `gate+up_fused` | (M=2·intermediate=34816, K=5120) | Already fused via `MlpFusedStore` |
| MLP (×64) | `down_proj` | (5120, 17408) | |
| LM head | `lm_head` | (V=152064, K=5120) | Once per token, BF16 weight today |
| MTP head | gate / up / down | similar shapes | Smaller subset |

`N = 1` for all of these in the decode hot path. `K` is always a power of 2 between 1024 and 17408. `M` ranges from 1024 to ~35 K.

**Key constraint:** the kernel must support **arbitrary M** (from 1024 to 35 K) with reasonable autotuning. K is fixed per call but the M values are scattered.

## 5. Proposed kernel design

### 5.1 Tile shape

Marlin's design (Frantar et al., 2024): `BLOCK_M × BLOCK_N × BLOCK_K` with `BLOCK_N = 1` (gemv-shaped). For NVFP4 + Blackwell's FP4 MMA atom (m16n8k64), the natural macro tile is:

- `BLOCK_M = 128` (8 MMA rows = 8 × m16 = 128).
- `BLOCK_N = 8` (1 MMA column = n8; pad output to 8, mask to 1).
- `BLOCK_K = 128` (2 MMA k-segments = 2 × k64 = 128).

`BLOCK_N = 8` is the smallest supported by the MMA atom. We compute 8 output columns per CTA but only the first column is meaningful — the kernel masks the others. Padding cost is amortised by the per-CTA work being dominated by weight loads.

### 5.2 Persistent grid + warp specialization

Spawn one CTA per `BLOCK_M` slice of output. With `BLOCK_M = 128` and `M` up to 35 K, that's at most 273 CTAs — well under the 192 SMs × persistent factor of ~2-4. Use a persistent kernel that processes multiple `BLOCK_M` slices per CTA (for the largest M cases) to amortise launch.

Inside each CTA, dedicate warps:
- **Producer warps (2 warps, 64 threads):** issue `cp.async.bulk.tensor` (TMA) loads for the next `BLOCK_M × BLOCK_K` weight tile + its NVFP4 scales tile. Pipelined with the consumer's MMA work.
- **Consumer warps (4 warps, 128 threads):** drive `mma.m16n8k64.kind::mxf4` on the previous tile, accumulate in FP32 registers.

Pipeline depth = 2 (double-buffered). One warp does scale dequant on-the-fly (e4m3 → f32) and stages into shared memory.

### 5.3 TMA multicast for activation broadcast

The `N = 1` activation vector is shared by all CTAs. Use `cp.async.bulk.tensor.*.multicast` to fan a single TMA load to all CTAs in a thread block cluster (Blackwell supports up to 16-CTA clusters). This eliminates redundant HBM loads of the activation vector.

For our shapes:
- Activation vector: `K * 0.5` bytes (NVFP4 = 4 bits/element, packed 2/byte). For K=5120: 2.5 KB.
- 192 SMs × 2.5 KB = 480 KB redundant load if not multicast. Negligible bandwidth-wise, but the multicast pattern simplifies the cluster design.

Cluster size: 4 CTAs (4 × 128 = 512 output rows per cluster). Activation tile (K=128) loaded once per cluster.

### 5.4 NVFP4 scale handling

Each `BLOCK_M × BLOCK_K` weight tile has scales:
- One e4m3 scale per 16-element block along K (so `BLOCK_M × (BLOCK_K / 16) = 128 × 8 = 1024 scales per tile`).
- One f32 tensor scale per layer (constant, broadcast).

Empirical from Mirage megakernel branch (`feat/mirage-megakernel`): cuBLASLt's `vec16_scale_offset` layout matches CUTLASS's `Sm1xxBlkScaledConfig::SfKMajorAtom`. **No re-tile pass needed** — read the scales directly from the weight buffer's existing layout.

Dequantize on-the-fly inside the consumer warps: `f32_acc = mma_acc * scale_e4m3 * tensor_scale_f32`. Fold the tensor_scale into the alpha at the epilogue.

### 5.5 Epilogue (BF16 output)

Standard FMA epilogue: `output_bf16[m, 0] = __float2bfloat16(acc_f32 * alpha)`. Single column write per CTA (since N=1 effective). For the K=8 padding columns, write but mask the read at the call site (or have the kernel skip writes 1..7).

### 5.6 Soft-fallback contract

The kernel returns `QWEN36_STATUS_NOT_IMPLEMENTED` (= 5) when:
- `M` is not a multiple of 128 (the BLOCK_M).
- `K` is not a multiple of 128 (the BLOCK_K).
- `N != 1` (we only target the gemv shape).

The Rust-side wrapper in `crates/kernels/src/backend.rs` mirrors the Mirage megakernel pattern: try the gemv kernel first, fall back to cuBLASLt on `5`. Env var `QWEN36_DECODE_GEMV=1` gates the path. This lets us ship shape-by-shape and validate parity per-layer.

## 6. CUDA changes

### 6.1 New file: `kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu`

Mirror the layout of `kernels-cuda/megakernel/nvfp4_matvec_sm120.cu`. ~600-800 lines of CUDA template + entry point. Conditionally compiled (gated on a `QWEN36_FP4_DECODE_GEMV=1` cargo build feature).

### 6.2 ABI extension (`kernels-cuda/include/qwen36_fp4.h`)

Reuse `qwen36_nvfp4_gemm_spec_t` (already exists). Add a new entry-point declaration:

```c
/// Single-stream gemv path for NVFP4 weights at N=1. Returns
/// QWEN36_STATUS_NOT_IMPLEMENTED for shapes outside the supported set
/// (M%128, K%128, N==1). Caller falls back to qwen36_nvfp4_gemm.
int qwen36_decode_nvfp4_gemv(const qwen36_nvfp4_gemm_spec_t *spec);
```

ABI rule (AGENT.md): mirror in `crates/kernels/src/backend.rs` FFI block.

### 6.3 Build script changes

`scripts/build_cuda.sh` detects `kernels-cuda/decode_gemv/` and compiles it when present. Standard `nvcc --extended-lambda --expt-relaxed-constexpr` flags. Targets SM_120 only (no fallback to SM_90 etc.).

### 6.4 Smoke test (`kernels-cuda/smoke.cu`)

Minimum viable: `M = 1024, K = 1024, N = 1` with planted weights and activation. Compare gemv output against a CPU reference (BF16 dequant) and assert max-abs error < 0.01. Mirror the existing top-K argmax + tree-mask attention smoke patterns.

### 6.5 Op-level parity (`scripts/decode_parity.py` extension)

Add an env var path `QWEN36_PARITY_GEMV_LAYER=<global_layer_idx>` that runs ONE forward with the gemv kernel forced on for the specified layer (cuBLASLt for everything else), dumps that layer's output, and the script diffs against a cuBLASLt-only reference run. Cos sim ≥ 0.998 gate.

## 7. Runtime changes

### 7.1 Backend dispatch (`crates/kernels/src/backend.rs`)

Mirror the Mirage megakernel dispatch:

```rust
fn nvfp4_gemm(&self, spec: &Nvfp4GemmSpec) -> Result<()> {
    let ffi_spec = ffi::Nvfp4GemmSpec::from(spec);

    if decode_gemv_enabled() && spec.n == 1 {
        let code = unsafe { ffi::qwen36_decode_nvfp4_gemv(&ffi_spec) };
        if code != 5 {  // != NOT_IMPLEMENTED
            return check("qwen36_decode_nvfp4_gemv", code);
        }
        // Fall through to cuBLASLt on shape mismatch
    }

    // ... existing megakernel + cuBLASLt dispatch unchanged ...
}
```

`decode_gemv_enabled()` reads `QWEN36_DECODE_GEMV`.

### 7.2 No engine-level changes needed

The dispatch is fully transparent at the kernel layer. Engine code (which calls `backend.nvfp4_gemm(...)`) doesn't change.

## 8. Testing strategy

1. **CUDA smoke** — minimum-shape gemv vs CPU reference (`kernels-cuda/smoke.cu`).
2. **Op-level parity** — every shape exercised in the decode path, cos sim ≥ 0.998 vs cuBLASLt. Layer-by-layer via the harness extension.
3. **End-to-end parity gate** — `chat hello / hello world × MTP {0..4}` with `QWEN36_DECODE_GEMV=1` matches the cuBLASLt-only baseline bit-for-bit.
4. **Bench** — `qwen36 bench --prompt-tokens 128 --max-new-tokens 128` with and without `QWEN36_DECODE_GEMV=1`. Median of 5 runs. Per-MTP-level breakdown.
5. **Microbenchmarks** — standalone `nvbench` or hand-rolled timing of the kernel at the actual decode shapes (using `cudaEventElapsedTime`). Verify the kernel-level speedup translates to the runtime.

## 9. Rollback / kill-switch

`QWEN36_DECODE_GEMV=1` opt-in (default OFF). Default behaviour is cuBLASLt — zero risk to production users until the kernel is validated on every shape.

`QWEN36_DECODE_GEMV_LAYER=<idx>` enables the gemv path for ONE specific layer only (for parity bisection).

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| MMA atom alignment constraints reject some M/K shapes | High | Soft-fallback to cuBLASLt; ship shape-by-shape |
| TMA multicast doesn't help at our cluster sizes | Medium | Profile with Nsight; can disable multicast and use plain TMA |
| FP4 numerical precision drift across all layers compounds | Medium | Op-level parity gate ≥ 0.998 catches per-op drift before integration |
| Persistent grid causes occupancy issues at small M | Low | Fall back to non-persistent launch for M < 4 × num_SMs |
| WSL2 launch overhead masks the gemv win | Medium | Native Linux re-bench (mentioned in AGENT.md as "highest-ROI experiment") would amplify gains |
| Marlin's published numbers don't reproduce on Blackwell | Medium | Marlin upstream has a Blackwell branch in progress; the design here is informed by the IST-DASLab paper but tuned to SM_120 |

## 11. Reference implementations

- **Marlin** (Frantar et al., 2024) — github.com/IST-DASLab/marlin. INT4 gemv for batch=1 LLM inference. The persistent + warp-specialized + TMA-multicast pattern originated here. Blackwell branch in flight (issue #110 last we checked).
- **CUTLASS 3.5+** — `sm90_tma_warpspecialized_cooperative` pattern for Hopper. Blackwell extensions land in CUTLASS 4.x.
- **Existing Mirage megakernel branch** (`feat/mirage-megakernel`) — already integrates CUTLASS NVFP4 GEMM at SM_120 and has the soft-fallback dispatch wired. The decode_gemv kernel reuses the dispatch pattern; the kernel itself is independent.

## 12. Implementation phases (within Direction B)

Each phase is a separate PR / merge candidate, gated on its own parity check:

- **B1** — ABI extension (`qwen36_decode_nvfp4_gemv` declaration) + Rust FFI mirror + soft-fallback dispatch in `backend.rs`. No kernel implementation yet; kernel returns `NOT_IMPLEMENTED` for all shapes. Validates the dispatch pipeline.
- **B2** — Minimal kernel for ONE shape (e.g., M=5120, K=5120 — the most common projection). Naive implementation: no TMA multicast, no persistent grid, just MMA + scale dequant. Op-level parity gate vs cuBLASLt.
- **B3** — Full Marlin-style kernel: persistent grid + warp specialization + TMA double-buffering. Handles all M%128 == 0, K%128 == 0 shapes.
- **B4** — TMA multicast for cluster-shared activation. Profile-driven cluster size tuning.
- **B5** — Bench matrix on RTX 5090 across all decode shapes. Update AGENT.md with the per-shape speedup table + end-to-end gain.
- **B6** — Default-on for shipped Qwen3.6 NVFP4 checkpoint (after every shape passes parity). Keep `QWEN36_DECODE_GEMV_DISABLE=1` as kill-switch.

## 13. Success criteria

- All hard parity gates pass (B5 onward).
- Op-level parity ≥ 0.998 cos sim on every decode-path NVFP4 GEMM shape.
- Bench shows ≥ +20 % decode tok/s vs cuBLASLt baseline at MTP=4 on the gated bench prompt (`qwen36 bench --prompt-tokens 128 --max-new-tokens 128`).
- `QWEN36_DECODE_GEMV_DISABLE=1` recovers exact cuBLASLt behaviour bit-for-bit.
- `cargo clippy --workspace --features qwen36-fp4-kernels/cuda -- -D warnings` and the existing CUDA test suite stay green.

## 14. Out of scope / Phase 2 hooks

- **Prefill-path NVFP4 GEMM optimisation.** Prefill uses N > 1 (chunked tokens), where cuBLASLt is already efficient. Skip for Direction B.
- **MTP head GEMM via gemv.** The MTP head's GEMMs run with similar shapes; the gemv kernel can extend to them naturally once Phase B5 lands. Add as a follow-up.
- **lm_head BF16 → NVFP4.** The LM head is the largest per-token GEMM (V × K = 152 K × 5 K). Today it's BF16. Quantising the lm_head to NVFP4 + using the gemv kernel could give another ~10 % decode gain. Out of scope here; depends on a separate quality/parity analysis.

## 15. Estimated payoff

Best case (every shape supported, Marlin-class speedup, no WSL2 overhead masking): 145–180 tok/s at MTP=4 (vs 123 today) = +18 to +46 %.

Realistic case (some shapes fall back to cuBLASLt, WSL2 overhead caps gains, scales work conservatively): 130–150 tok/s at MTP=4 = +6 to +22 %.

Worst case (kernel correctness work eats the whole month, no shapes ready): zero deployed gain, ship infrastructure for B6+ to flesh out later.

The cuBLASLt FP4 N=1 inefficiency is a real, structural issue (not an optimisation for the marginal case). Even a 50 % gemv-vs-cuBLASLt kernel speedup is a measurable end-to-end win because NVFP4 GEMMs are the largest single decode cost on this stack.
