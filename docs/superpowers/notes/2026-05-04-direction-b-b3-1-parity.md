# Direction B Phase B3.1 — parity blocker (RESOLVED)

**Date:** 2026-05-04
**Predecessor commit:** `a1f1dd9` (B3.1 MMA kernel landed, smoke green)
**Status:** **Root cause identified by codex consult: tensor scales were applied twice.** Fix lands in commit after `3cf72e6`. Awaiting end-to-end chat-parity verification on a GPU-free window before declaring B3.1 closed.

## Resolution (added 2026-05-04 after codex review)

The runtime caller in `crates/runtime/src/engine.rs` already pre-folds both per-tensor scales into `spec->alpha` before invoking the kernel:

```rust
// engine.rs:4677-4678 (and parallel sites at 4734, 4810, 4925, 5397, 5451)
let alpha = self.tensor_scalar_f32(weights, qkv_tensor_scale)?
          * self.tensor_scalar_f32(weights, quantized.input_scale)?;
```

cuBLASLt at `kernels-cuda/nvfp4_gemm.cu:380` consumes only that pre-baked `alpha` and never dereferences `a_scale_2` / `b_scale_2` for the multiply (those device pointers are unused on the cuBLASLt path).

Our gemv kernel was loading `a_scale_2` / `b_scale_2`, multiplying them in *again* on top of `alpha`, effectively squaring the per-tensor scales. With realistic per-channel scales like `a_ts ≈ 1/8` and `b_ts ≈ 1/4`, the output is ~`1/1024`× too small — pure gibberish.

The uniform smoke missed it because both tensor scales resolve to 1.0 in that setup (`a_scale_2` is null → 1.0; `b_scale_2 == 1.0` is asserted by the test). `1.0² == 1.0`, no observable error. None of the original ranked hypotheses (SF byte order, A register swap, fp4 shift, scale-offset edge case, B broadcast) was the actual bug.

## The fix

`kernels-cuda/decode_gemv/nvfp4_gemv_sm120.cu` epilogue:

```cpp
// Before:
const float scale = alpha * a_ts * b_ts;
output[row_lo] = __float2bfloat16(acc0 * scale);

// After:
output[row_lo] = __float2bfloat16(acc0 * alpha);
```

Also dropped the `__ldg(a_tensor_scale)` / `__ldg(b_tensor_scale)` loads at the top of the kernel; the kernel parameters stay (ABI-stable, ignored). Soft-disable removed; the entry point now launches the MMA kernel for `n==1 && m%16==0 && k%64==0`.

## Lesson

Layout bugs aren't the only thing uniform smoke misses — **value-scaling bugs are equally invisible at unit-magnitude inputs**. Any future kernel-validation harness should plant tensor scales ≠ 1.0 on at least one of the two operands, in addition to heterogeneous per-element values.

## Symptom

`./scripts/smoke_cuda.sh` passes (`decode_gemv b2[0] = decode_gemv b2[last] = 132.0` matching cuBLASLt) but end-to-end chat with `QWEN36_DECODE_GEMV=1` produces gibberish:

```
# Baseline (cuBLASLt, env var unset):
$ ./target/release/qwen36 chat --prompt hello --max-new-tokens 12 --mtp-speculative-tokens 0
Here's a thinking process:
1.  **An...

# Gemv path (QWEN36_DECODE_GEMV=1):
$ QWEN36_DECODE_GEMV=1 ./target/release/qwen36 chat --prompt hello --max-new-tokens 12 --mtp-speculative-tokens 0
HerepiteASNullenlehr proven�事小人oughtascar济
```

## Why the smoke didn't catch this

The planted-data probe at `M=K=128, N=1` uses:
- weight bytes = `0x22` (every nibble = e2m1 +1.0)
- weight scales = `0x38` (every byte = e4m3 1.0)
- activation = quantize(`std::vector<float>(K, 1.0f)`)

Every `(m, k)` weight slot holds the same value, every per-block scale holds the same value, every activation slot holds the same value. The MMA result `Σ (a_mk · w_mk · s_a · s_b)` is invariant under permutation of `m`, `k`, scale-group index, register order. Any layout bug — register pack order, SF byte order, m-row mapping, k-chunk indexing — produces the same numeric output as the correct layout. **The smoke is fundamentally non-discriminating for layout bugs.**

Real model weights vary across `(m, k)` slots; layout bugs surface immediately as garbage logits.

## Verified facts

The B3.1 implementation matches the research-agent reports verbatim:
- A operand register-to-(m, k) mapping (per `MMA_Traits<...>::ALayout`).
- B operand register-to-(n, k) mapping (per `BLayout`).
- C/D operand register-to-(m, n) mapping; column n=0 in `D[0]` (rows 0..7) and `D[2]` (rows 8..15) for lanes with `t0==0`.
- SFA / SFB lane decomposition (`m_row_sf = 8*(L&1) + (L>>2)`, `n_col_sf = L>>2`).
- SFA / SFB byte-packing: byte `g` of the uint32 = e4m3 for `k_group g` ∈ {0,1,2,3}.
- PTX call signature mirrors `cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS::fma(...)` byte-for-byte (`cute/arch/mma_sm120.hpp:3215`).
- `-arch=sm_120a` correctly assembles the `kind::mxf4nvf4` PTX; the device asm is gated on `__CUDA_ARCH__ == 1200 && __CUDA_ARCH_FEAT_SM120_ALL`.
- Build, link, smoke at the planted-uniform shape all green.

## Hypotheses ranked

1. **SF byte order reversed** (most likely). Agent D's report header said "k_group 0..3 in low..high nibbles" — likely a misstatement (SF entries are bytes, not nibbles), but the underlying cute layout encoding deserves a re-verification. Reversing the byte order in the SFA/SFB pack loop is a 1-line experiment.
2. **A operand register r → (m, k) swap.** Agent A's table says `A[r]` for `r = v1 + 2*v2` with v1 affecting m, v2 affecting k. Our code matches that; but the in-file comment header has it reversed — could indicate the agent's report was internally inconsistent and the code accidentally followed the comment's mistaken framing.
3. **fp4 nibble shift required for `kind::mxf4nvf4`.** Agent C suggested the shift may NOT be needed for the k64 atom (only for k32 f8f6f4) but flagged it as unverified. The PTX itself doesn't document the bit position requirement explicitly. Worth a 1-line `<< 2` experiment on A and B uint32 inputs.
4. **Scale offset formula wrong for non-power-of-2 `sf_inner_dim`** at production K values (e.g. K=5120 → sf_inner_dim=320). Smoke uses K=128 → sf_inner_dim=8 which is a degenerate case.
5. **B operand n_col mismatch at N=1.** B layout per Agent A: `B[r] n_col = t1`. With N=1, lanes with `t1>0` are computing for non-existent N columns; the MMA accumulator may interpret this differently than the docs imply.

## Recommended bisect plan

1. **Add a heterogeneous regression smoke.** Plant weights with values that vary across `(m, k)` (e.g. `weight[m][k] = (m + k) % 6`, encoded as e2m1 indices). Plant scales that vary across `(m, k_group)`. Plant activation that varies across `k`. Compute the expected output via a CPU reference reading the same buffers through the existing `decode_e2m1` / `decode_e4m3` helpers in `kernels-cuda/ops.cu`. Compare cuBLASLt vs MMA outputs row-by-row.
2. **Re-enable the MMA path locally** (revert this note's soft-disable temporarily) and run the new smoke. The diff between cuBLASLt and MMA output identifies which `(m, k)` slots are getting the wrong contribution.
3. **Bisect by zeroing out parts of the inputs**: if zeroing all but row 0 of A still produces wrong output for row 0, the A or SFA layout is wrong. Etc. for B / SFB.
4. **Cross-check against the cute traits source directly** (`cute/atom/mma_traits_sm120.hpp` → `MMA_Traits<SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<...>>::ALayout` etc.) instead of relying on the agent's decoded report.

## What ships in this commit

- Entry-point `qwen36_decode_nvfp4_gemv` returns `NOT_IMPLEMENTED` unconditionally; the dispatcher routes to cuBLASLt as before.
- Smoke probe at `M=K=128, N=1` removed (the test was non-discriminating; keeping it would create false confidence).
- Kernel implementation retained in-file as the starting point for the bisect.
- Build script + ABI + Rust dispatch stay unchanged so the env-var contract is preserved.

## What's safe to assume going forward

- Phase B2 (commit `0e81174`) was *also* validated only against the same uniform smoke. We do **not** know whether the B2 scalar kernel produces correct output on real model weights either. The pragmatic stance: treat both B2 and B3.1 as unvalidated against real workloads until the heterogeneous smoke + cuBLASLt cross-check lands.
- The cuBLASLt path remains the production-correct reference; nothing in this investigation changes that.
- The `QWEN36_DECODE_GEMV` env var is still wired and remains the kill switch; today its only effect is to add one extra C-FFI call per NVFP4 GEMM (which immediately returns 5/NOT_IMPLEMENTED) and route to cuBLASLt — measurable overhead but no correctness risk.
