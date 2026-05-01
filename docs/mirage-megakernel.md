# Mirage-style Megakernel for Qwen3.6-27B Decode

Branch: `feat/mirage-megakernel`. Long-running, can be reset/rebased without
pressure on `codex/numerical-parity-guardrails`.

## Goal

Reduce the per-decode-token cost on RTX 5090 below the current 22 ms
(MTP=0 ≈ 45 tok/s) toward the published single-stream NVFP4 reference of
~100 tok/s, by replacing the launch-overhead-dominated decode path with a
chain of CUTLASS-templated kernels and (eventually) a persistent
mega-kernel that holds inter-layer state on-chip.

Not a goal: change behaviour. Parity gate
(`chat --prompt "hello" --max-new-tokens 12` matches MTP 0/2/3) must stay
green at every commit.

## Why current cuBLASLt FP4 path leaves perf on the table

Per-token decode currently dispatches roughly **150 NVFP4 GEMMs** through
cuBLASLt. cuBLASLt at batch=1 on Blackwell SM120:

- picks suboptimal kernels for `M » N=1, K=hidden` shapes (vLLM HF blog
  benchmarks: cuBLASLt picks `192x144_2cta` while CUTLASS PingPong 64x128
  is 1.78× faster for the same shape on B200 / RTX 5090).
- adds ~10–20 µs of launch / dispatch overhead per call. Even inside a
  CUDA Graph the per-node setup is non-zero and accumulates over a 600+
  node graph.
- can't be trivially fused with surrounding ops (RMSNorm, SwiGLU,
  quantize) because the GEMM kernel is opaque.

A custom CUTLASS kernel templated on our exact `(M, N=1, K)` shapes,
schedule (`KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100` adapted to
SM120's 99 KB SMEM with PingPong 64×128), and FP4 layout gives us:

- a tighter kernel for our actual shape mix.
- the ability to fuse epilogues (RMSNorm, SwiGLU, NVFP4 quantize) into
  the GEMM directly.
- the foundation for a persistent kernel that loops across layers.

## Architectural choice — what mega-kernel actually means here

True Mirage compiles the entire model into one persistent kernel that
holds layer boundaries via mbarriers. That's a multi-week rewrite of the
compute graph and is over-scope for one branch.

We aim for an *incremental* path that captures most of the benefit:

1. **Foundation:** vendor CUTLASS, build a single-kernel NVFP4 matvec
   that beats cuBLASLt for our hot shapes, swap it in behind an env var,
   verify parity + measurable bench gain.
2. **Epilogue fusion:** extend the CUTLASS kernel with epilogues that
   apply RMSNorm / SwiGLU / NVFP4 quantize directly on the GEMM output,
   eliminating intermediate BF16 round-trips.
3. **Per-layer mega-kernels:** one CUTLASS persistent kernel per layer
   that sequences `(input rmsnorm + quantize) → GEMM₁ → (epilogue) →
   GEMM₂ → ...` without inter-kernel host launches.
4. **Cross-layer persistent kernel:** a single persistent kernel that
   walks the 64 layers via mbarrier synchronisation, holds DeltaNet
   recurrent state in SMEM/registers across layers, and only exits at
   the final RMSNorm + lm_head + sample. This is the true Mirage style.

We commit to delivering at least Phase 1 + 2, judging Phase 3 on the
bench numbers Phase 1+2 produce, and keeping Phase 4 as a stretch.

## Constraints — RTX 5090 SM120 specifics

- 99 KB SMEM per SM (vs 228 KB on B200). 128×128 FP4 tile schedules
  overflow; **64×128 PingPong** is the documented working tile config
  for SM120 NVFP4 GEMMs.
- No Tensor Memory (TMEM): all accumulator state lives in registers and
  shared memory.
- L1: 128 KB / SM.
- Native FP4 MMA: `mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32`,
  11-cycle latency, throughput ~3.85 PFLOP/s peak FP4.
- CUTLASS FP4 path requires CUDA 12.8+ (we have CUDA 13.0+), nvcc with
  `--expt-relaxed-constexpr --extended-lambda`.

## File / module layout

```
kernels-cuda/
  cutlass/                     <- vendored CUTLASS (submodule or copy)
  megakernel/
    nvfp4_matvec_sm120.cu      <- Phase 1: CUTLASS NVFP4 (M,1,K) kernel(s)
    layer_mlp_sm120.cu         <- Phase 3: per-layer MLP mega-kernel
    layer_deltanet_sm120.cu    <- Phase 3: per-layer DeltaNet mega-kernel
    decode_persistent_sm120.cu <- Phase 4: full forward persistent kernel
crates/kernels/src/megakernel.rs  <- Rust spec + FFI bindings
```

ABI sync rule (see `AGENT.md`) still applies: any change to the C spec
in `kernels-cuda/include/qwen36_fp4.h` mirrors into
`crates/kernels/src/backend.rs` and the relevant typed Rust spec.

## Phase 1 — CUTLASS NVFP4 GEMM at small N

**Targets** (all M × 1 × 5120 except where noted):

| Shape (M, K) | Site | Per-token count |
|--|--|--|
| `(34816, 5120)` | MLP fused gate+up | 64 |
| `(5120, 17408)` | MLP down_proj | 64 |
| `(16640, 5120)` | DeltaNet fused in_proj | 48 |
| `(5120, 6144)`  | DeltaNet out_proj | 48 |
| `(12288, 5120)` | Full-attn q_proj | 16 |
| `(1024, 5120)`  | Full-attn k_proj / v_proj | 32 |
| `(5120, 6144)`  | Full-attn o_proj | 16 |
| `(248K, 5120)`  | lm_head (BF16 today, candidate for FP4) | 1 |

**Implementation:**

1. Vendor CUTLASS at `kernels-cuda/cutlass/` (header-only, ~300 MB).
2. Build a templated kernel based on
   [Example 72b](https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu)
   with schedule:
   ```
   ThreadBlockShape = Shape<_64, _128, _128>;
   ClusterShape     = Shape<_1, _1, _1>;     // SM120 single-SM
   KernelSchedule   = KernelTmaWarpSpecializedPingpongFP8Fast;  // adapt for SM120 NVFP4
   ```
3. Expose via FFI:
   `qwen36_nvfp4_gemm_megakernel(spec)` with the same Nvfp4GemmSpec
   shape but a different code path.
4. Add an env var gate `QWEN36_USE_MEGAKERNEL_GEMM=1` so we can A/B test
   without code edits.
5. Per-shape parity: compare combined gate+up output BF16-element-wise
   against the existing cuBLASLt path within FP4 quantisation noise.

**Gate to land Phase 1:** parity green on hello+12 MTP 0/2/3 with
`QWEN36_USE_MEGAKERNEL_GEMM=1`, **and** measurable bench delta on at
least MTP=0 (target ≥ +5 tok/s, i.e. 45.5 → 50+).

## Phase 1 status — measured outcome

After landing the live CUTLASS NVFP4→BF16 GemmUniversalAdapter for
SM120 (`kernels-cuda/megakernel/nvfp4_matvec_sm120.cu`) and exercising
it through the env-gated dispatcher on every decode-path NVFP4 GEMM:

| ThreadBlockShape | Bench MTP=0 (vs cuBLASLt baseline) | Notes |
|--|--|--|
| `<128, 128, 128>` | -5 % | wide N tile wastes SM occupancy at N=1 |
| `<128, 8, 128>` | parity | matches SM120 FP4 MMA atom N=8 |
| `<64, 8, 128>` | parity | more M-axis tiles, no measurable gain |

**Finding:** raw GEMM swap from cuBLASLt → CUTLASS at our N=1 single-
token decode shapes is **at parity, not faster**. cuBLASLt's FP4 path on
Blackwell appears well-tuned for batch=1; the published "1.78× over
cuBLASLt at BS=1" reference (vLLM HF blog) does not reproduce on our
exact shape mix on RTX 5090 / SM120 / CUTLASS 4.4.2 with the schedules
the `CollectiveBuilder` auto-selects for these template parameters.

The remaining leverage that justifies the megakernel branch sits in the
**epilogue fusion** (Phase 2) and **persistent multi-layer kernel**
(Phase 3/4): the cuBLASLt path emits 1 GEMM kernel per call, so anything
fused into the same launch (RMSNorm + SwiGLU + activation quantize on
the GEMM output) is pure savings. CUTLASS's CollectiveEpilogue is the
mechanism — that is the next concrete step.

## Phase 2 — Epilogue fusion (deep-dive)

Originally proposed to fuse SwiGLU + NVFP4 quantize into the gate+up
GEMM and RMSNorm + NVFP4 quantize into the down_proj GEMM. After
investigating CUTLASS 4.4.2's EVT (Epilogue Visitor Tree) and the actual
data flow, the realistic gains are **smaller than initially estimated
and the implementation is genuinely multi-day**. Documented here so the
next iteration starts from the right baseline.

### What CUTLASS provides out of the box

* `cutlass::epilogue::fusion::LinCombBlockScaleFactor` (Example 79b):
  `D = alpha * A*B + beta * C` with NVFP4 + e4m3 scale factor generation
  on the output. Pre-built fusion, drop-in replacement for the
  `LinearCombination` we use today.
* `LinearCombinationBiasElementwise`: linear combination + per-row /
  per-column bias add. Pre-built. Our model has no bias, so unused.
* Custom EVT visitors via `cutlass::epilogue::fusion::Sm90EVT` (also
  applies to SM100/SM120): chainable element-wise visitors for
  arbitrary post-GEMM transformations.

### Why fusing SwiGLU into the gate+up GEMM is mathematically blocked

In our current `MlpFusedStore` layout the combined weight stacks
gate_proj and up_proj **along the M dim** (M = 2·intermediate). The
GEMM tile partition is `(Tile_M, Tile_N)`; for our `<128, 8, 128>`
schedule each tile sees `M_start..M_start+128` rows of the output.

Because `intermediate = 17408 = 136 × 128` is a multiple of the M tile
size, every tile is **either entirely "gate output" (m < intermediate)
or entirely "up output" (m ≥ intermediate)** — never both. SwiGLU
needs to pair `gate[i]` with `up[i + intermediate]`, which live in
different tiles. CUTLASS epilogues cannot share state across tiles, so
the fusion cannot land inside a single GEMM call without restructuring.

Restructuring to put gate and up side-by-side in the N dim (batched
GEMM with L=2) would let a single epilogue tile see both halves, but
batched-GEMM epilogues also do not share state across batches, so the
pairing problem just moves. There is no clean CUTLASS-native fusion.

### Why fusing RMSNorm into down_proj is hard

RMSNorm requires a `sum(x²)` reduction across the **M dim** (the output
rows for a single N=1 column) before the per-element scaling. Each
GEMM epilogue tile only sees a 128-row slice of the M=5120 down_proj
output, so the reduction must accumulate across 40 tiles. CUTLASS does
support split-K reductions via a follow-up "ReduceK" kernel, but
split-M reductions across the *output* M for a single N column are not
a standard pattern and require either an explicit second pass or a
custom persistent kernel.

### What still has tractable upside

* **`LinCombBlockScaleFactor` for down_proj** — emit FP4 + scales
  directly so the next layer's `rmsnorm_nvfp4_quantize` can read FP4
  input instead of BF16. Requires a new RMSNorm-from-NVFP4 kernel
  variant. Estimated saving: ~10–20 µs per layer × 64 = 0.6–1.3 ms /
  token = **3–6 % MTP=0**. Effort: ~1–2 focused days for the kernel
  pair + parity harness.

* **In-place residual add via `beta=1, C=residual`** for down_proj
  GEMM. Currently the residual add happens inline inside the next
  layer's rmsnorm; pulling it into the GEMM epilogue saves one BF16
  buffer write + read per layer, ~0.5 µs / layer × 64 = **0.03 ms /
  token = ~0.15 %**. Negligible. Skip.

* **CUTLASS persistent / streamK schedules** for very tall + skinny
  GEMMs (M=large, N=1, K=hidden). Could shift the perf parity vs
  cuBLASLt, but the CollectiveBuilder defaults are already in this
  family on SM120, so this is iterative tile / schedule tuning. Effort
  unclear, gain unclear.

### Bottom line

The Mirage-style branch lands a **complete, parity-verified CUTLASS
NVFP4 substrate** that is at perf parity with cuBLASLt at our shapes.
The next CUTLASS-side perf gain (3–6 % via `LinCombBlockScaleFactor`
on down_proj + RMSNorm-from-NVFP4) is multi-day work for modest
return.

Higher-leverage next experiments to try BEFORE returning to CUTLASS
EVT work:

1. Boot a native Linux kernel and re-bench: WSL2 is documented to add
   1–3 µs per kernel launch; with ~30 cudaGraphLaunches per second of
   decode + sync points, the expected delta is **5–10 % free**.
2. MTP head BF16 → NVFP4 quantization with a parity harness on draft
   acceptance rate. Higher gain on MTP=2/3 but high parity risk.
3. True Mirage cross-layer persistent kernel (Phase 4). Multi-week
   effort, gain unclear without prototype.

## Phase 2 — Epilogue fusion (deferred / planning only)

## Phase 3 — Per-layer mega-kernels

Single CUTLASS persistent kernel per layer that sequences:

- DeltaNet layer: input RMSNorm+quant → fused in_proj GEMM → conv1d
  update → gdn_gate → recurrent decode (existing kernel called inline)
  → output RMSNorm+gate → out_proj GEMM. One kernel launch per layer
  instead of ~7.
- Full-attn layer: input RMSNorm+quant → fused Q/K/V GEMM → q
  deinterleave → q/k norms → partial RoPE → attention decode (existing
  kernel) → q sigmoid gate → o_proj GEMM. One launch instead of ~10.
- MLP block: input RMSNorm+quant → combined gate+up GEMM → SwiGLU →
  NVFP4 quantize → down_proj GEMM. One launch instead of ~4.

Total target: from ~757 launches/token down to ~64 (one per layer) + a
few global ops (embed, final norm, lm_head, sample). Within the CUDA
Graph this should still produce a measurable wall-clock saving because
each launch has graph-replay setup overhead.

**Gate to land Phase 3:** parity green, MTP=0 ≥ 65 tok/s.

## Phase 4 — Cross-layer persistent kernel (stretch)

One persistent kernel for the entire decode forward. SMs split into
roles (loader / mma / softmax / store) and walk through layers via
mbarriers. State (DeltaNet recurrent, KV writes) handed across layers
in shared memory or registers without HBM round-trip.

**Gate to land Phase 4:** parity green, MTP=0 ≥ 80 tok/s. Otherwise
revert to Phase 3 and document why Phase 4 didn't pay off.

## Verification harness

- Existing parity gate stays the gate of record.
- New per-shape parity script under `scripts/megakernel_parity.py`:
  enumerate the 6 hot GEMM shapes, compare CUTLASS-megakernel output
  vs cuBLASLt output for the same FP4 inputs, fail above
  `cos_sim ≥ 0.999` and `max abs diff > 0.05` (FP4 quant noise band).
- Bench harness: existing `bench` command, with `QWEN36_USE_MEGAKERNEL_*`
  env vars to A/B between cuBLASLt and CUTLASS paths.
- Profile harness: existing `QWEN36_PROFILE_DECODE_LAYERS=1` shows
  per-bucket time deltas across phases.

## Risks and exit ramps

- **CUTLASS NVFP4 SM120 build complexity** (Phase 1 may take longer
  than expected if SMEM constraints force tile re-tuning). Exit ramp:
  ship a single shape (the biggest, MLP combined gate+up at M=34816)
  and fall back to cuBLASLt for the others.
- **Parity drift in epilogue fusion** (Phase 2). Exit ramp: keep the
  fused epilogue env-gated and ship the GEMM-only path.
- **Persistent kernel scheduling bugs** (Phases 3/4). Exit ramp: per-
  layer mega-kernels are still ahead of Phase 0; the cross-layer
  variant stays optional.

Worst case the branch lands Phase 1 only, which is itself worth
~+5–10 % bench and is the foundation for any future kernel work.
