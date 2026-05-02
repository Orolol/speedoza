# Tree-MTP Phase 1 Implementation Plan (simplified architecture)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add last-position top-K acceptance to the MTP decode loop. The MTP head's last forward samples K candidates instead of one; after the existing chain verify chunk completes, accept the highest-prob leaf whose token matches the model's argmax at the last chain position. Expected +20-40 % decode throughput on borderline prompts with no kernel-graph surgery.

**Architecture:** Two layers of change. (1) One new CUDA kernel — `qwen36_topk_argmax` (single block, K ≤ 8). (2) Runtime-side `verify_mtp_tree_draft` that wraps today's verify graph and adds a host-side leaf-acceptance check, plus `generate_top_k_leaves` for draft generation. CLI flag `--mtp-tree-leaves` exposes K. Env var `QWEN36_MTP_TREE_DISABLE=1` is the kill-switch.

**Tech Stack:** Rust 1.85 (edition 2024), Cargo workspace, CUDA 13.0+ targeting SM_120 (Blackwell, RTX 5090), cuBLASLt, custom CUDA kernels in `kernels-cuda/`.

**Spec:** `docs/superpowers/specs/2026-05-02-tree-mtp-phase1-design.md` (revised same-day after design review — see "Revision note" at top of spec).

---

## Conventions

- **Worktree:** all work happens in `/home/orolol/workspace/speedoza-tree-mtp` on branch `feat/perf-tree-mtp-stack`.
- **CPU-only loop:** `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`. Always green before any commit.
- **CUDA loop:** `./scripts/build_cuda.sh && ./scripts/smoke_cuda.sh && QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda" LD_LIBRARY_PATH="$PWD/target/cuda:$LD_LIBRARY_PATH" cargo test --workspace --features qwen36-fp4-kernels/cuda`.
- **ABI rule (from AGENT.md):** changes to `kernels-cuda/include/qwen36_fp4.h` MUST be mirrored in `crates/kernels/src/backend.rs` AND the relevant typed spec module under `crates/kernels/src/`. New struct fields go at the end while the ABI is still evolving.
- **Parity rule:** no CUDA-kernel optimization lands without a parity check. The new kernel ships with both a smoke (planted top-K) and a Rust-side CPU-reference comparison.
- **Commit style:** Conventional commits, signed with the project's `Co-Authored-By` line.

## File structure

| Path | Role | New / Modified |
|---|---|---|
| `kernels-cuda/include/qwen36_fp4.h` | C ABI for `qwen36_topk_argmax` | Modified |
| `kernels-cuda/ops.cu` | New `qwen36_topk_argmax` kernel + entry point | Modified |
| `kernels-cuda/smoke.cu` | New smoke case for top-K | Modified |
| `crates/kernels/src/sampling.rs` | CPU `topk_argmax` reference + Rust wrapper for the kernel | Modified |
| `crates/kernels/src/backend.rs` | FFI mirror of the new C ABI | Modified |
| `crates/mtp/src/lib.rs` | New `TreeDraft` / `TreeVerifyResult` types, `walk_tree_acceptance`, `MtpConfig.tree_leaves` | Modified |
| `crates/runtime/src/engine.rs` | `verify_mtp_tree_draft` + `generate_top_k_leaves` + tree-mode dispatch | Modified |
| `crates/cli/src/main.rs` | New `--mtp-tree-leaves` flag for `chat` and `bench`, plumbing into config | Modified |
| `AGENT.md` | New table with Phase 1 bench results | Modified (P1.3 only) |

**Files we deliberately do NOT touch in Phase 1:** `kernels-cuda/attention.cu`, `kernels-cuda/deltanet.cu`, `crates/runtime/src/gpu.rs` (no `MtpKvSnapshotLayout` bump), `crates/runtime/src/cuda_graph.rs`, `scripts/decode_parity.py`. The verify chunk path is unchanged.

---

# P1.1 — Top-K argmax CUDA kernel

Smallest surface, no engine impact. Land first.

## Task 1.1.1 — CPU `topk_argmax` reference + unit tests

**Files:**
- Modify: `crates/kernels/src/sampling.rs`

- [ ] **Step 1: Add the CPU reference at the top of `sampling.rs`**

```rust
/// CPU reference for top-K argmax used in tests and as a fallback.
/// Returns the K vocab indices with the highest logits, sorted by descending
/// logit (ties broken via `f32::total_cmp`).
pub fn topk_argmax(logits: &[f32], k: usize) -> Vec<u32> {
    let mut indexed: Vec<(usize, f32)> =
        logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    indexed
        .into_iter()
        .take(k)
        .map(|(idx, _)| idx as u32)
        .collect()
}
```

- [ ] **Step 2: Add unit tests in the existing test module (or create `#[cfg(test)] mod tests` if absent)**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_argmax_returns_sorted_top_k_indices() {
        let logits = vec![0.1, 5.0, 2.0, 5.5, 1.0];
        assert_eq!(topk_argmax(&logits, 3), vec![3, 1, 2]);
    }

    #[test]
    fn topk_argmax_caps_at_input_length() {
        let logits = vec![0.0, 1.0];
        assert_eq!(topk_argmax(&logits, 8), vec![1, 0]);
    }

    #[test]
    fn topk_argmax_zero_k_is_empty() {
        assert_eq!(topk_argmax(&[1.0, 2.0], 0), Vec::<u32>::new());
    }

    #[test]
    fn topk_argmax_k_one_matches_greedy_argmax() {
        let logits = vec![0.1, 0.4, 0.3, 0.4001];
        assert_eq!(topk_argmax(&logits, 1), vec![3]);
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p qwen36-fp4-kernels --lib sampling`
Expected: 4 new tests pass alongside the existing `greedy_argmax` test.

- [ ] **Step 4: Commit**

```bash
git add crates/kernels/src/sampling.rs
git commit -m "feat(kernels): add CPU topk_argmax reference and tests

$(cat <<'EOF'

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 1.1.2 — C ABI for `qwen36_topk_argmax`

**Files:**
- Modify: `kernels-cuda/include/qwen36_fp4.h`
- Modify: `crates/kernels/src/backend.rs`

- [ ] **Step 1: Add the spec struct and entry-point declaration to `qwen36_fp4.h`**

After the `qwen36_sampling_spec_t` block (line 279-288), add:

```c
typedef struct {
  size_t vocab_size;
  size_t k;                                // 1..QWEN36_TOPK_MAX
  qwen36_device_ptr_t logits_bf16;
  qwen36_device_ptr_t output_token_u32;    // [k] u32, sorted desc by logit
} qwen36_topk_argmax_spec_t;

#define QWEN36_TOPK_MAX 8
```

After the `int qwen36_sample(...)` declaration (around line 475), add:

```c
int qwen36_topk_argmax(const qwen36_topk_argmax_spec_t *spec);
```

- [ ] **Step 2: Mirror in `crates/kernels/src/backend.rs`**

Find the existing `extern "C" { pub fn qwen36_sample(...) }` block (around line 1245 area, inside `mod ffi { ... }`). Add an identically-styled block:

```rust
#[repr(C)]
pub struct TopkArgmaxSpec {
    pub vocab_size: usize,
    pub k: usize,
    pub logits_bf16: DevicePtr,
    pub output_token_u32: DevicePtr,
}

extern "C" {
    pub fn qwen36_topk_argmax(spec: *const TopkArgmaxSpec) -> i32;
}
```

Match the surrounding visibility / module placement — most likely inside the same `mod ffi`.

- [ ] **Step 3: Verify the workspace still builds**

Run: `cargo build --workspace`
Expected: clean (declarations only, no kernel impl yet).

- [ ] **Step 4: Commit**

```bash
git add kernels-cuda/include/qwen36_fp4.h crates/kernels/src/backend.rs
git commit -m "feat(abi): declare qwen36_topk_argmax C entry point

$(cat <<'EOF'

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 1.1.3 — Implement `qwen36_topk_argmax` CUDA kernel

**Files:**
- Modify: `kernels-cuda/ops.cu`

- [ ] **Step 1: Add the templated kernel and the C entry point at the end of `ops.cu`**

```cpp
// === Top-K argmax ============================================================
// Single-block kernel. Each thread maintains a thread-local sorted top-K array
// in registers (descending), scans `vocab_size` elements with stride
// blockDim.x, then a single-thread block-level merge of the per-thread arrays
// (BLOCK*K <= 4096, cheap) produces the final sorted top-K. K capped at 8.

template <int K>
__global__ void topk_argmax_kernel(
    const __nv_bfloat16 *logits, size_t vocab_size, uint32_t *out) {
  constexpr int BLOCK = 512;
  __shared__ float s_vals[BLOCK * K];
  __shared__ uint32_t s_idx[BLOCK * K];

  float local_v[K];
  uint32_t local_i[K];
#pragma unroll
  for (int j = 0; j < K; ++j) {
    local_v[j] = -INFINITY;
    local_i[j] = 0xFFFFFFFFu;
  }

  for (size_t i = threadIdx.x; i < vocab_size; i += BLOCK) {
    float v = __bfloat162float(logits[i]);
    if (v > local_v[K - 1]) {
      int p = K - 1;
      local_v[p] = v;
      local_i[p] = static_cast<uint32_t>(i);
      while (p > 0 && local_v[p] > local_v[p - 1]) {
        float tv = local_v[p - 1];
        uint32_t ti = local_i[p - 1];
        local_v[p - 1] = local_v[p];
        local_i[p - 1] = local_i[p];
        local_v[p] = tv;
        local_i[p] = ti;
        --p;
      }
    }
  }

#pragma unroll
  for (int j = 0; j < K; ++j) {
    s_vals[threadIdx.x * K + j] = local_v[j];
    s_idx[threadIdx.x * K + j] = local_i[j];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float merged_v[K];
    uint32_t merged_i[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
      merged_v[j] = -INFINITY;
      merged_i[j] = 0xFFFFFFFFu;
    }
    for (int t = 0; t < BLOCK; ++t) {
      for (int j = 0; j < K; ++j) {
        float v = s_vals[t * K + j];
        if (v > merged_v[K - 1]) {
          int p = K - 1;
          merged_v[p] = v;
          merged_i[p] = s_idx[t * K + j];
          while (p > 0 && merged_v[p] > merged_v[p - 1]) {
            float tv = merged_v[p - 1];
            uint32_t ti = merged_i[p - 1];
            merged_v[p - 1] = merged_v[p];
            merged_i[p - 1] = merged_i[p];
            merged_v[p] = tv;
            merged_i[p] = ti;
            --p;
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < K; ++j) {
      out[j] = merged_i[j];
    }
  }
}

extern "C" int qwen36_topk_argmax(const qwen36_topk_argmax_spec_t *spec) {
  if (!spec || spec->logits_bf16.ptr == 0 || spec->output_token_u32.ptr == 0) {
    return QWEN36_STATUS_NULL_POINTER;
  }
  if (spec->vocab_size == 0 || spec->k == 0 || spec->k > QWEN36_TOPK_MAX) {
    return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  const auto *logits =
      reinterpret_cast<const __nv_bfloat16 *>(spec->logits_bf16.ptr);
  auto *out = reinterpret_cast<uint32_t *>(spec->output_token_u32.ptr);
  cudaStream_t stream = qwen36_internal_active_stream();
  switch (spec->k) {
    case 1: topk_argmax_kernel<1><<<1, 512, 0, stream>>>(logits, spec->vocab_size, out); break;
    case 2: topk_argmax_kernel<2><<<1, 512, 0, stream>>>(logits, spec->vocab_size, out); break;
    case 3: topk_argmax_kernel<3><<<1, 512, 0, stream>>>(logits, spec->vocab_size, out); break;
    case 4: topk_argmax_kernel<4><<<1, 512, 0, stream>>>(logits, spec->vocab_size, out); break;
    case 5: topk_argmax_kernel<5><<<1, 512, 0, stream>>>(logits, spec->vocab_size, out); break;
    case 6: topk_argmax_kernel<6><<<1, 512, 0, stream>>>(logits, spec->vocab_size, out); break;
    case 7: topk_argmax_kernel<7><<<1, 512, 0, stream>>>(logits, spec->vocab_size, out); break;
    case 8: topk_argmax_kernel<8><<<1, 512, 0, stream>>>(logits, spec->vocab_size, out); break;
    default: return QWEN36_STATUS_INVALID_ARGUMENT;
  }
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? QWEN36_STATUS_SUCCESS : QWEN36_STATUS_CUDA_ERROR;
}
```

- [ ] **Step 2: Build CUDA**

Run: `./scripts/build_cuda.sh`
Expected: `target/cuda/libqwen36_fp4_kernels.so` rebuilt without errors.

- [ ] **Step 3: Commit**

```bash
git add kernels-cuda/ops.cu
git commit -m "feat(cuda): implement qwen36_topk_argmax kernel

Single-block, K up to 8, ~152k vocab. Per-thread top-K array in
registers, single-thread block-level merge sufficient at this K range.

$(cat <<'EOF'

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 1.1.4 — Smoke parity for `qwen36_topk_argmax`

**Files:**
- Modify: `kernels-cuda/smoke.cu`

- [ ] **Step 1: Add a smoke case in `smoke.cu`**

Find the entry point (search for `int main` or the existing smoke driver function). Add a block alongside the other smoke cases:

```cpp
{
  // Top-K argmax smoke: K=4 over a 1024-vocab BF16 logits array with
  // planted top-4 at known indices.
  constexpr size_t V = 1024;
  std::vector<__nv_bfloat16> h_logits(V);
  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  for (size_t i = 0; i < V; ++i) {
    h_logits[i] = __float2bfloat16(dist(rng));
  }
  uint32_t expect[4] = {17, 200, 999, 42};
  float vals[4] = {10.0f, 9.5f, 9.0f, 8.5f};
  for (int i = 0; i < 4; ++i) {
    h_logits[expect[i]] = __float2bfloat16(vals[i]);
  }

  __nv_bfloat16 *d_logits = nullptr;
  uint32_t *d_out = nullptr;
  cudaMalloc(&d_logits, V * sizeof(__nv_bfloat16));
  cudaMalloc(&d_out, 4 * sizeof(uint32_t));
  cudaMemcpy(d_logits, h_logits.data(), V * sizeof(__nv_bfloat16),
             cudaMemcpyHostToDevice);

  qwen36_topk_argmax_spec_t spec{};
  spec.vocab_size = V;
  spec.k = 4;
  spec.logits_bf16.ptr = reinterpret_cast<uint64_t>(d_logits);
  spec.output_token_u32.ptr = reinterpret_cast<uint64_t>(d_out);
  int rc = qwen36_topk_argmax(&spec);
  if (rc != QWEN36_STATUS_SUCCESS) {
    std::fprintf(stderr, "topk smoke: rc=%d\n", rc);
    return 1;
  }
  cudaDeviceSynchronize();
  uint32_t got[4]{};
  cudaMemcpy(got, d_out, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 4; ++i) {
    if (got[i] != expect[i]) {
      std::fprintf(stderr, "topk smoke mismatch at %d: got %u want %u\n",
                   i, got[i], expect[i]);
      cudaFree(d_logits);
      cudaFree(d_out);
      return 1;
    }
  }
  cudaFree(d_logits);
  cudaFree(d_out);
  std::printf("topk smoke OK\n");
}
```

If the existing smoke driver uses different memory helpers, mirror them; only the kernel call + assertion logic above is mandatory.

- [ ] **Step 2: Run the CUDA smoke loop**

Run: `./scripts/build_cuda.sh && ./scripts/smoke_cuda.sh`
Expected: prints `topk smoke OK` and exits 0.

- [ ] **Step 3: Commit**

```bash
git add kernels-cuda/smoke.cu
git commit -m "test(cuda): smoke for qwen36_topk_argmax K=4

$(cat <<'EOF'

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 1.1.5 — Rust wrapper

**Files:**
- Modify: `crates/kernels/src/sampling.rs`

- [ ] **Step 1: Add the wrapper next to the other CUDA-feature-gated helpers in `sampling.rs`**

```rust
use crate::backend::{self as backend, DevicePtr};
use qwen36_fp4_core::{CoreError, Result};

/// Launch `qwen36_topk_argmax` on the active stream. K must be 1..=8.
#[cfg(feature = "cuda")]
pub fn topk_argmax_device(
    vocab_size: usize,
    k: usize,
    logits_bf16: DevicePtr,
    output_token_u32: DevicePtr,
) -> Result<()> {
    if k == 0 || k > 8 {
        return Err(CoreError::Runtime(format!(
            "topk_argmax_device: k must be 1..=8, got {k}"
        )));
    }
    let spec = backend::ffi::TopkArgmaxSpec {
        vocab_size,
        k,
        logits_bf16,
        output_token_u32,
    };
    let rc = unsafe { backend::ffi::qwen36_topk_argmax(&spec) };
    if rc != 0 {
        return Err(CoreError::Runtime(format!(
            "qwen36_topk_argmax failed rc={rc}"
        )));
    }
    Ok(())
}
```

If `sampling.rs` already imports `backend` at a different path, mirror it. The exact module path of the FFI block (`backend::ffi::...`) depends on how `backend.rs` exposes it — confirm by reading the `qwen36_sample` wrapper at the top of `sampling.rs`.

- [ ] **Step 2: Verify build + tests**

Run: `cargo test -p qwen36-fp4-kernels`
Expected: pure-Rust tests pass; CUDA-gated wrapper compiles when feature enabled.

Run: `cargo build --workspace --features qwen36-fp4-kernels/cuda`
Expected: clean.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/kernels/src/sampling.rs
git commit -m "feat(kernels): Rust wrapper for qwen36_topk_argmax

$(cat <<'EOF'

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# P1.2 — Tree acceptance walk + engine integration + CLI

End-to-end Tree-MTP via host-side acceptance. The verify chunk path is unchanged; this section only adds host-side logic and one MTP-head top-K call per cycle.

## Task 1.2.1 — Pure-Rust types and acceptance walk

**Files:**
- Modify: `crates/mtp/src/lib.rs`

- [ ] **Step 1: Add `tree_leaves` to `MtpConfig`**

Find `pub struct MtpConfig` (line 7-10). Update:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpConfig {
    pub num_speculative_tokens: usize,
    pub greedy: bool,
    /// Top-K branching at the last MTP head position. 1 = chain MTP behaviour
    /// (Phase 1 default).
    pub tree_leaves: usize,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 3,
            greedy: true,
            tree_leaves: 1,
        }
    }
}
```

- [ ] **Step 2: Add `TreeDraft`, `TreeVerifyResult`, and `walk_tree_acceptance`**

Append to `crates/mtp/src/lib.rs`, after the `SpeculativeDecoder` impl:

```rust
pub const MTP_TREE_MAX_LEAVES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeDraft {
    /// Length = chain_depth (= today's MTP=N draft count). First chain draft
    /// follows last_token.
    pub chain_tokens: Vec<u32>,
    /// Length = K. Top-K candidates from the MTP head's last forward,
    /// sorted by descending logit. K = 1 reproduces chain MTP exactly.
    pub leaf_tokens: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeVerifyResult {
    /// Always starts with the verified-from-`last_token` (committed) token,
    /// then accepted chain tokens, then the accepted leaf if any.
    pub committed: Vec<u32>,
    pub accepted_chain: usize,         // 0..=chain_depth
    pub accepted_leaf: Option<usize>,  // 0..K
    /// Verified token at the last accepted position; seed for next cycle's
    /// `last_token`.
    pub next_token: u32,
}

/// Walk a branched-tail tree given the model's argmax at each chain row.
///
/// `verified[i]` is the model's argmax at chain row `i` (i = 0..=chain_depth).
/// The first chain draft is `draft.chain_tokens[0]`, accepted iff
/// `verified[0] == chain_tokens[0]`. After full chain acceptance, the leaves
/// are scanned in input order; top-K is sorted desc by logit, so the first
/// match is the highest-prob leaf.
pub fn walk_tree_acceptance(
    verified: &[u32],
    draft: &TreeDraft,
) -> TreeVerifyResult {
    let chain_depth = draft.chain_tokens.len();
    debug_assert!(verified.len() >= 1 + chain_depth);
    let mut committed = vec![verified[0]];
    let mut accepted_chain = 0;
    for (i, &candidate) in draft.chain_tokens.iter().enumerate() {
        if verified[i] == candidate {
            committed.push(candidate);
            accepted_chain = i + 1;
        } else {
            return TreeVerifyResult {
                committed,
                accepted_chain,
                accepted_leaf: None,
                next_token: verified[i],
            };
        }
    }
    let chain_verified = verified[chain_depth];
    let accepted_leaf = draft
        .leaf_tokens
        .iter()
        .position(|&leaf| leaf == chain_verified);
    if let Some(idx) = accepted_leaf {
        committed.push(draft.leaf_tokens[idx]);
        TreeVerifyResult {
            committed,
            accepted_chain,
            accepted_leaf: Some(idx),
            next_token: draft.leaf_tokens[idx],
        }
    } else {
        TreeVerifyResult {
            committed,
            accepted_chain,
            accepted_leaf: None,
            next_token: chain_verified,
        }
    }
}
```

- [ ] **Step 3: Add unit tests at the bottom of `lib.rs` in a new `mod tree_tests`**

```rust
#[cfg(test)]
mod tree_tests {
    use super::*;

    #[test]
    fn k1_reduces_to_chain_mtp() {
        // K=1, top-1 leaf should never override chain when chain rejects.
        let draft = TreeDraft {
            chain_tokens: vec![10, 20, 30],
            leaf_tokens: vec![999],
        };
        let verified = vec![10, 20, 30, 50];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![10, 20, 30]);
        assert_eq!(r.accepted_chain, 3);
        assert_eq!(r.accepted_leaf, None);
        assert_eq!(r.next_token, 50);
    }

    #[test]
    fn full_chain_no_leaf_match() {
        let draft = TreeDraft {
            chain_tokens: vec![10, 20, 30],
            leaf_tokens: vec![100, 200],
        };
        let verified = vec![10, 20, 30, 999];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![10, 20, 30]);
        assert_eq!(r.accepted_chain, 3);
        assert_eq!(r.accepted_leaf, None);
        assert_eq!(r.next_token, 999);
    }

    #[test]
    fn full_chain_first_leaf_match() {
        let draft = TreeDraft {
            chain_tokens: vec![10, 20, 30],
            leaf_tokens: vec![100, 200, 300],
        };
        let verified = vec![10, 20, 30, 200];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![10, 20, 30, 200]);
        assert_eq!(r.accepted_chain, 3);
        assert_eq!(r.accepted_leaf, Some(1));
        assert_eq!(r.next_token, 200);
    }

    #[test]
    fn full_chain_top_leaf_wins_over_lower_match() {
        // Both leaf 0 and leaf 2 happen to equal verified — leaf 0 (top-1)
        // must win.
        let draft = TreeDraft {
            chain_tokens: vec![10],
            leaf_tokens: vec![42, 100, 42],
        };
        let verified = vec![10, 42];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.accepted_leaf, Some(0));
    }

    #[test]
    fn chain_rejects_at_first_mismatch() {
        let draft = TreeDraft {
            chain_tokens: vec![10, 20, 30],
            leaf_tokens: vec![100],
        };
        let verified = vec![10, 99, 0, 0];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![10]);
        assert_eq!(r.accepted_chain, 1);
        assert_eq!(r.accepted_leaf, None);
        assert_eq!(r.next_token, 99);
    }

    #[test]
    fn chain_rejects_at_root_skips_leaves() {
        let draft = TreeDraft {
            chain_tokens: vec![10, 20],
            leaf_tokens: vec![999],
        };
        let verified = vec![888, 0, 0];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![888]);
        assert_eq!(r.accepted_chain, 0);
        assert_eq!(r.accepted_leaf, None);
        assert_eq!(r.next_token, 888);
    }

    #[test]
    fn empty_chain_with_leaves() {
        // chain_depth = 0 → leaf check uses verified[0] directly.
        let draft = TreeDraft {
            chain_tokens: vec![],
            leaf_tokens: vec![55, 66],
        };
        let verified = vec![66];
        let r = walk_tree_acceptance(&verified, &draft);
        assert_eq!(r.committed, vec![66, 66]);
        assert_eq!(r.accepted_chain, 0);
        assert_eq!(r.accepted_leaf, Some(1));
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p qwen36-fp4-mtp`
Expected: all tree_tests pass alongside the existing tests.

- [ ] **Step 5: Commit**

```bash
git add crates/mtp/src/lib.rs
git commit -m "feat(mtp): TreeDraft/TreeVerifyResult + walk_tree_acceptance

Pure-Rust types and helpers for last-position top-K acceptance, with
unit tests covering K=1 reduction, full chain accept (with and without
leaf match), top-leaf-wins tie-break, and chain reject at root / mid.

$(cat <<'EOF'

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 1.2.2 — Engine: `generate_top_k_leaves`

**Files:**
- Modify: `crates/runtime/src/engine.rs`

- [ ] **Step 1: Read the existing MTP draft generation path**

The existing chain MTP draft generation lives in the verify graph capture (`ensure_mtp_verify_graph_multi_tokens`, line 573 onwards). The MTP head is invoked via `run_mtp_prefill_chunk_with_tokens`, then `queue_sample_greedy_into` samples greedy argmax from the resulting logits. The hidden state is in `cuda_prefill()?.normed.ptr()` after `run_mtp_prefill_chunk_with_tokens`.

For Phase 1 we need a top-K sample from the MTP head's *last* forward output. The cleanest insertion point is right after the existing `queue_sample_greedy_into(first_next_draft_ptr)` call (around line 656) — at that point the MTP head logits for the next draft are sitting in the forward logits buffer.

- [ ] **Step 2: Add a new helper near the existing sampling helpers**

Locate the helper cluster around line 329-430 (`queue_sample_greedy*`, `decode_sampled_queued`, etc.). Add:

```rust
/// Queue a top-K argmax sample from the engine's current logits buffer into
/// the device-side `output_token_u32_kvec` slot. The output buffer must be at
/// least `k * 4` bytes. Used by the tree-MTP draft generation path.
#[cfg(feature = "cuda")]
fn queue_sample_topk_into(
    &self,
    output_token_u32_kvec: DevicePtr,
    k: usize,
) -> Result<()> {
    use qwen36_fp4_kernels::sampling::topk_argmax_device;
    let logits = self.cuda_forward()?.logits_bf16.ptr();
    let vocab_size = self.topology.vocab_size;
    topk_argmax_device(vocab_size, k, logits, output_token_u32_kvec)
}
```

`topology.vocab_size` may be at a slightly different name in this codebase; if the existing `prefill_row_logits` reads vocab from elsewhere (e.g., `self.config.vocab_size` or a constant), mirror that. The point is one source of truth.

- [ ] **Step 3: Add a device buffer to hold the K leaf token IDs**

Find `GpuForwardBuffers` in `crates/runtime/src/gpu.rs` (search `pub struct GpuForwardBuffers`). Add a field (use the file's existing allocation idiom):

```rust
    /// Up to MTP_TREE_MAX_LEAVES u32 leaf token IDs sampled by top-K.
    pub leaf_tokens_u32: DeviceAllocation,
```

In the constructor, allocate `MTP_TREE_MAX_LEAVES * size_of::<u32>() = 32` bytes. Import `MTP_TREE_MAX_LEAVES` from `qwen36_fp4_mtp`.

- [ ] **Step 4: Verify CUDA build**

Run: `./scripts/build_cuda.sh && cargo build --workspace --features qwen36-fp4-kernels/cuda`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/runtime/src/gpu.rs crates/runtime/src/engine.rs
git commit -m "feat(runtime): leaf token buffer + queue_sample_topk_into helper

$(cat <<'EOF'

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 1.2.3 — Engine: `verify_mtp_tree_draft`

**Files:**
- Modify: `crates/runtime/src/engine.rs`

- [ ] **Step 1: Add the entry point next to `verify_mtp_draft_tokens` (line 1633)**

```rust
/// Tree-MTP one-cycle entry. Wraps the existing chunked verify path with a
/// host-side last-position top-K acceptance check. When `tree_leaves <= 1`
/// this is a thin shim around `verify_mtp_draft_tokens`.
pub fn verify_mtp_tree_draft(
    &mut self,
    chain_tokens: &[u32],
    leaf_tokens: &[u32],
    start_position: usize,
) -> Result<qwen36_fp4_mtp::TreeVerifyResult> {
    use qwen36_fp4_mtp::{walk_tree_acceptance, TreeDraft, TreeVerifyResult, MTP_TREE_MAX_LEAVES};
    if leaf_tokens.is_empty() {
        return Err(CoreError::Runtime("tree leaf_tokens cannot be empty".into()));
    }
    if leaf_tokens.len() > MTP_TREE_MAX_LEAVES {
        return Err(CoreError::Runtime(format!(
            "tree leaf_tokens.len() = {} exceeds {MTP_TREE_MAX_LEAVES}",
            leaf_tokens.len()
        )));
    }
    if mtp_tree_disable_enabled() {
        // Force K=1 fallback: drop all leaves except top-1.
        let trimmed_leaves = &leaf_tokens[..1];
        return self.verify_mtp_tree_draft_inner(chain_tokens, trimmed_leaves, start_position);
    }
    self.verify_mtp_tree_draft_inner(chain_tokens, leaf_tokens, start_position)
}

#[cfg(feature = "cuda")]
fn verify_mtp_tree_draft_inner(
    &mut self,
    chain_tokens: &[u32],
    leaf_tokens: &[u32],
    start_position: usize,
) -> Result<qwen36_fp4_mtp::TreeVerifyResult> {
    use qwen36_fp4_mtp::{walk_tree_acceptance, TreeDraft};
    // 1. Run today's chunked verify on the chain. This re-uses
    //    verify_mtp_draft_tokens which already handles MtpVerifyOne /
    //    MtpVerifyMulti graph variants and produces verified[0..=chain_depth]
    //    in `mtp_verify_token_u32` slots.
    let chain_result = self.verify_mtp_draft_tokens(chain_tokens, start_position)?;
    let chain_depth = chain_tokens.len();
    let mut verified: Vec<u32> = Vec::with_capacity(chain_depth + 1);
    for i in 0..=chain_depth {
        verified.push(self.read_mtp_verified_slot(i)?);
    }
    // 2. Host-side acceptance walk.
    let draft = TreeDraft {
        chain_tokens: chain_tokens.to_vec(),
        leaf_tokens: leaf_tokens.to_vec(),
    };
    let result = walk_tree_acceptance(&verified, &draft);
    // 3. If the leaf-acceptance changed the next-cycle's last_token relative
    //    to the chain-only result, persist that as the engine's current token.
    //    The KV cache and DeltaNet state at the leaf's position will be
    //    written naturally on next cycle's verify chunk row 0 — no extra work.
    if let Some(_) = result.accepted_leaf {
        self.set_current_token(result.next_token)?;
    }
    let _ = chain_result;  // logged elsewhere
    Ok(result)
}
```

The two helpers `read_mtp_verified_slot` and `set_current_token` need to exist or be added:

- `read_mtp_verified_slot(i)`: D2H copy of `cuda_forward()?.mtp_verify_token_u32.ptr_at((MTP_GRAPH_VERIFIED_BASE + i) * 4)?` into a `u32`. Mirror the existing `read_sampled_token` (around line 1532-1554 patterns). Stream sync before D2H per the AGENT.md invariant.
- `set_current_token(token)`: H2D copy of the u32 into `cuda_forward()?.token_u32.ptr()`. There's likely already a primitive for this (search for `forward.token_u32` writes from host).

`mtp_tree_disable_enabled()` is a small helper:

```rust
fn mtp_tree_disable_enabled() -> bool {
    std::env::var("QWEN36_MTP_TREE_DISABLE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}
```

Place it next to `mtp_assume_accept_enabled` (search the file).

- [ ] **Step 2: Build + run the existing runtime tests**

Run: `cargo build --workspace --features qwen36-fp4-kernels/cuda`
Run: `cargo test -p qwen36-fp4-runtime`
Expected: all green; no behavioural change yet because nothing calls `verify_mtp_tree_draft` until P1.2.4.

- [ ] **Step 3: Commit**

```bash
git add crates/runtime/src/engine.rs
git commit -m "feat(runtime): verify_mtp_tree_draft + tree-disable env switch

Wraps verify_mtp_draft_tokens with a host-side last-position top-K
acceptance walk. When QWEN36_MTP_TREE_DISABLE=1 or leaf_tokens.len()==1,
behaviour reduces to today's chain MTP exactly.

$(cat <<'EOF'

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 1.2.4 — CLI: `--mtp-tree-leaves` flag and dispatch

**Files:**
- Modify: `crates/cli/src/main.rs`

- [ ] **Step 1: Add the flag to the `chat` and `bench` clap subcommands**

Locate the existing `mtp_speculative_tokens` clap field (around line 100, 112, 247, 256, 263). For each, add an adjacent field:

```rust
#[arg(long, default_value_t = 1, help = "Top-K leaves at the last MTP position (1 = chain MTP)")]
mtp_tree_leaves: usize,
```

Validate at parse time that `1 <= mtp_tree_leaves <= 8`. Use clap's `value_parser = clap::value_parser!(u8).range(1..=8)` then cast, or a manual check at the top of `run_chat` / `run_bench`.

- [ ] **Step 2: Wire the flag through `run_chat` and `run_bench`**

Each function signature gains `mtp_tree_leaves: usize` next to `mtp_speculative_tokens`. Pass into `MtpConfig`:

```rust
let mtp_config = MtpConfig {
    num_speculative_tokens: mtp_schedule.effective_tokens,
    greedy: true,
    tree_leaves: mtp_tree_leaves,
};
```

- [ ] **Step 3: In the decode loop, branch to the tree path when `tree_leaves > 1`**

Find the call to `decoder.step(&mut engine, last_token)?` in the chat / bench loops. Add a branch:

```rust
let result = if mtp_config.tree_leaves > 1 {
    // Generate chain drafts as today, plus leaves via top-K from the MTP
    // head's last forward.
    let draft = engine.generate_tree_draft(
        last_token,
        mtp_config.num_speculative_tokens,
        mtp_config.tree_leaves,
    )?;
    let tree_result = engine.verify_mtp_tree_draft(
        &draft.chain_tokens,
        &draft.leaf_tokens,
        start_position,
    )?;
    // Map back to the existing MtpStepResult shape so downstream reporting
    // (acceptance counters, traces) stays unchanged.
    MtpStepResult {
        committed_tokens: tree_result.committed.clone(),
        drafted_tokens: draft
            .chain_tokens
            .iter()
            .chain(draft.leaf_tokens.iter())
            .copied()
            .collect(),
        accepted_draft_tokens: tree_result.accepted_chain
            + tree_result.accepted_leaf.map_or(0, |_| 1),
    }
} else {
    decoder.step(&mut engine, last_token)?
};
```

`engine.generate_tree_draft(...)` is a new engine method:

```rust
pub fn generate_tree_draft(
    &mut self,
    last_token: u32,
    chain_depth: usize,
    leaf_count: usize,
) -> Result<TreeDraft>;
```

Implementation: extend the existing chain draft generation to additionally invoke `queue_sample_topk_into` on the MTP head's last logits before the chain's final greedy sample, then D2H read K leaf tokens. Add it next to the existing draft generation in `engine.rs`.

The minimum change: factor today's chain draft generation into a new method that returns just `Vec<u32>` (the chain), then wrap it in `generate_tree_draft` which calls the chain method, then runs one extra MTP head forward (or reuses the last hidden state if it's still in `cuda_prefill()?.normed`), then `queue_sample_topk_into` on the result, sync, D2H read.

- [ ] **Step 4: Adjust bench reporting to expose the leaf-accept rate**

In the bench loop, accumulate counters:
- `cycles_total`
- `chain_full_accept_cycles`
- `leaf_accepted_cycles`

At end of bench, log:
```
leaf_accept_rate = leaf_accepted_cycles / chain_full_accept_cycles
```

(If `chain_full_accept_cycles == 0`, log `n/a`.)

- [ ] **Step 5: Smoke test the CLI parsing without GPU**

Run: `cargo run -p qwen36-fp4 -- chat --help | grep -q mtp-tree-leaves`
Expected: prints the flag.

Run: `cargo run -p qwen36-fp4 -- bench --help | grep -q mtp-tree-leaves`
Expected: prints the flag.

Run: `cargo build --workspace`
Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/cli/src/main.rs crates/runtime/src/engine.rs
git commit -m "feat(cli): --mtp-tree-leaves flag + tree draft dispatch

Default tree_leaves=1 keeps today's chain MTP behaviour bit-for-bit.
Bench loop accumulates leaf-accept rate alongside the existing
acceptance counters.

$(cat <<'EOF'

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 1.2.5 — Hard parity gate

This is a process gate, not a code change. Run it to confirm Phase 1 didn't break anything before moving on.

- [ ] **Step 1: Build release binary**

Run:
```bash
QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda" \
LD_LIBRARY_PATH="$PWD/target/cuda:$LD_LIBRARY_PATH" \
cargo build --release -p qwen36-fp4 --features cuda
```

- [ ] **Step 2: Run the parity matrix**

```bash
MODEL_DIR=~/models/Qwen3.6-27B-Text-NVFP4-MTP
OUT=/tmp/tree_parity
mkdir -p "$OUT"
for prompt in "hello" "hello world"; do
  for mtp in 0 1 2 3; do
    for leaves in 1 2 4; do
      [ "$mtp" = "0" ] && [ "$leaves" != "1" ] && continue
      tag=$(echo "$prompt" | tr ' ' _)_${mtp}_${leaves}
      echo "=== prompt='$prompt' mtp=$mtp leaves=$leaves ==="
      QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda" \
      LD_LIBRARY_PATH="$PWD/target/cuda:$LD_LIBRARY_PATH" \
        ./target/release/qwen36 chat \
          --model-dir "$MODEL_DIR" \
          --prompt "$prompt" \
          --max-new-tokens 12 \
          --mtp-speculative-tokens "$mtp" \
          --mtp-tree-leaves "$leaves" \
          > "$OUT/$tag.txt"
    done
  done
done
```

- [ ] **Step 3: Diff against MTP=0 baseline**

```bash
for prompt in "hello" "hello world"; do
  base="$OUT/$(echo "$prompt" | tr ' ' _)_0_1.txt"
  for f in "$OUT"/$(echo "$prompt" | tr ' ' _)_*.txt; do
    [ "$f" = "$base" ] && continue
    if ! diff -q "$base" "$f" > /dev/null; then
      echo "FAIL parity: $f differs from $base"
      diff "$base" "$f" | head -20
    fi
  done
done
```

Expected: zero "FAIL parity" lines.

- [ ] **Step 4: Bisection plan if it fails**

If any combo with `--mtp-tree-leaves 1` fails, the regression is in the dispatch / wiring (P1.2.3 or P1.2.4) — leaves=1 should never enter the `if mtp_config.tree_leaves > 1` branch. Verify the branch condition.

If only `--mtp-tree-leaves >= 2` combos fail:
1. First check `QWEN36_MTP_TREE_DISABLE=1 ./target/release/qwen36 chat --mtp-tree-leaves 4 ...` matches the leaves=1 case. If it does, the kill-switch works and the bug is in the leaf-acceptance path.
2. Compare per-cycle traces by setting `QWEN36_MTP_TRACE=1`. The chain acceptance numbers should match between leaves=1 and leaves>1 modulo the leaf-accept events.
3. If the chain acceptance differs, `verify_mtp_tree_draft_inner` is wrongly mutating engine state — most likely culprit is `set_current_token` running at the wrong moment.

- [ ] **Step 5: Commit (typically empty if the gate passes cleanly — gate is a process gate)**

```bash
git commit --allow-empty -m "test(runtime): hard parity gate passed for tree MTP leaves in {1,2,4}

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If a code fix was required to pass the gate, commit it normally instead and skip the empty commit.

---

# P1.3 — Bench, decision, document

No new code by default. Measure, decide, write up.

## Task 1.3.1 — Bench matrix on RTX 5090

- [ ] **Step 1: Re-build release**

```bash
cargo build --release -p qwen36-fp4 --features cuda
```

- [ ] **Step 2: Run the matrix on the gated bench prompt**

```bash
MODEL_DIR=~/models/Qwen3.6-27B-Text-NVFP4-MTP
for mtp in 3; do
  for leaves in 1 2 4 8; do
    echo "=== mtp=$mtp leaves=$leaves ==="
    QWEN36_FP4_KERNEL_LIB_DIR="$PWD/target/cuda" \
    LD_LIBRARY_PATH="$PWD/target/cuda:$LD_LIBRARY_PATH" \
      ./target/release/qwen36 bench \
        --model-dir "$MODEL_DIR" \
        --prompt-tokens 128 --max-new-tokens 32 \
        --mtp-speculative-tokens "$mtp" \
        --mtp-tree-leaves "$leaves"
  done
done
```

Run each combo 5 times; report median tok/s.

- [ ] **Step 3: Run the same matrix on a varied prompt set**

For each prompt below, time decode of 64 new tokens externally with `time --format=%e`. (If the bench command supports a prompt file, use it; otherwise use `chat` and time the wall clock minus a fixed prompt-processing baseline.)

Suggested prompts:
- `"hello"` (gated)
- `"hello world"` (gated)
- `"Write Python hello world"` (borderline)
- `"Count from 1 to 5."` (borderline)
- `"Write a short poem about cats."` (borderline)
- `"What is 2 + 2?"` (deterministic)
- `"Translate good morning to French."` (deterministic)
- `"Explain quantum entanglement in one sentence."` (open)

- [ ] **Step 4: Tabulate**

Build the table:

```
| MTP | leaves | decode tok/s (gated) | decode tok/s (varied median) | speedup vs leaves=1 | leaf_accept_rate |
|-----|--------|----------------------|------------------------------|---------------------|------------------|
| 3   | 1      | xx.x                 | xx.x                         | 1.00x               | n/a              |
| 3   | 2      | xx.x                 | xx.x                         | x.xxx               | xx.x%            |
| 3   | 4      | xx.x                 | xx.x                         | x.xxx               | xx.x%            |
| 3   | 8      | xx.x                 | xx.x                         | x.xxx               | xx.x%            |
```

- [ ] **Step 5: Decide K**

Spec success criterion: `≥ +15 % decode tok/s` for `--mtp-tree-leaves` ∈ {2, 4, 8} (best of) over `--mtp-tree-leaves 1` on the varied prompt set median. Pick the K that maximises tok/s × (1 / variance).

If even the best K shows `< +15 %`, that's a Phase 1 negative result. Document the negative result honestly and discuss whether Phase 2's true multi-level branching is worth pursuing immediately (the answer is probably yes — single-level branching is the cheapest win available).

## Task 1.3.2 — Update `AGENT.md`

**Files:**
- Modify: `AGENT.md`

- [ ] **Step 1: Append a new dated section under "Current optimization status"**

Do not delete the existing tables. Append:

```markdown
### 2026-05-XX — Tree-MTP Phase 1 (last-position top-K)

Bench reference (RTX 5090, --prompt-tokens 128 --max-new-tokens 32, full-accept regime, median of 5 runs):

| MTP | leaves | decode tok/s | speedup | leaf_accept_rate |
|--|--|--|--|--|
| 3 | 1 | xx.x | 1.00x | n/a |
| 3 | 2 | xx.x | x.xxx | xx.x% |
| 3 | 4 | xx.x | x.xxx | xx.x% |
| 3 | 8 | xx.x | x.xxx | xx.x% |

Default K chosen: **<chosen value>**.

Kill switch: `QWEN36_MTP_TREE_DISABLE=1` forces leaf_count=1.

Phase 2 entry points (NOT YET IMPLEMENTED):
- Tree-mask `attention_prefill_kernel` variant (per-row ancestor bitmap)
- Per-branch DeltaNet state machinery (`qwen36_deltanet_update_tree`)
- KV scratch slots per branch + `MtpVerifyTree` graph kind
- Adaptive tree-shape generator (Phase 3)
```

- [ ] **Step 2: Commit**

```bash
git add AGENT.md
git commit -m "docs: tree-MTP Phase 1 bench results + Phase 2 entry points

$(cat <<'EOF'

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 1.3.3 — Close out the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-tree-mtp-phase1-design.md`

- [ ] **Step 1: Append a closing section to the spec**

```markdown
## 15. Phase 1 outcome

- Bench: see `AGENT.md` "2026-05-XX — Tree-MTP Phase 1" section.
- Default K: <chosen>
- Known follow-ups: <list any deferred items, profiling notes, or anomalies>
- Phase 2 path: track in §12.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-02-tree-mtp-phase1-design.md
git commit -m "docs(spec): close out tree-MTP Phase 1 with bench outcome

$(cat <<'EOF'

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# Self-review checklist (run before merging Phase 1)

Run these before declaring Phase 1 done:

- [ ] `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] `cargo clippy --workspace --features qwen36-fp4-kernels/cuda -- -D warnings` clean.
- [ ] `cargo test --workspace` and `cargo test --workspace --features qwen36-fp4-kernels/cuda` both green.
- [ ] `./scripts/smoke_cuda.sh` green (includes top-K smoke).
- [ ] Hard parity gate (P1.2.5) passes.
- [ ] Soft parity envelope (borderline prompts) is no wider than today's MTP=2/3.
- [ ] Bench (P1.3.1) shows the spec's success criterion (≥ +15 % decode tok/s on varied prompt set) — OR documents a negative result honestly.
- [ ] `QWEN36_MTP_TREE_DISABLE=1` recovers exact pre-Phase-1 behaviour.
- [ ] `AGENT.md` updated with bench results, kill-switch, and Phase 2 entry points.
