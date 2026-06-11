//! DFlash drafter "propose" half (Phase F.0).
//!
//! Wraps the drafter forward with the surrounding glue needed to turn
//! `(noise_token_ids, target_hidden_raw)` into a list of greedy-sampled
//! candidate tokens:
//!
//! 1. Embed `noise_token_ids` via the target's `embed_tokens` weight →
//!    `noise_embedding` `[q_len, hidden]` BF16.
//! 2. Drafter forward → `[q_len, hidden]` BF16 hidden state.
//! 3. `lm_head` GEMM `[m=vocab, n=q_len, k=hidden]` → column-major
//!    logits `[vocab, q_len]` BF16.
//! 4. `sample_rows` greedy argmax → `[q_len]` u32 token ids.
//!
//! The verify half (run drafted block through target, compare argmax,
//! drive cache crops) is Phase F.1.

use anyhow::{Context, Result, anyhow, bail};
use qwen36_fp4_kernels::{
    Bf16GemmSpec, CudaDeviceBuffer, DevicePtr, EmbeddingLookupSpec, KernelBackend, SamplingRowsSpec,
};

use crate::dflash::DFlashConfig;
use crate::forward::DrafterForward;

const BF16_BYTES: usize = 2;
const U32_BYTES: usize = 4;
const LM_HEAD_GEMM_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

/// Per-call scratch for the propose helper. Sized for a worst-case
/// `q_len`; the lm_head logits buffer dominates (q_len_max × vocab_size
/// × 2 bytes, e.g. 16 × 248320 × 2 = 7.6 MiB on Qwen3.6's vocab).
pub struct DFlashProposeWorkspace {
    pub q_len_max: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,

    noise_token_ids: CudaDeviceBuffer, // [q_len_max] u32
    noise_embedding: CudaDeviceBuffer, // [q_len_max, hidden] BF16
    logits: CudaDeviceBuffer,          // [vocab, q_len_max] col-major BF16
    sampled_tokens: CudaDeviceBuffer,  // [q_len_max] u32
    gemm_workspace: CudaDeviceBuffer,

    bytes_total: usize,
}

impl DFlashProposeWorkspace {
    pub fn alloc(config: &DFlashConfig, q_len_max: usize) -> Result<Self> {
        if q_len_max == 0 {
            bail!("q_len_max must be > 0");
        }
        let alloc = |bytes: usize, label: &str| -> Result<CudaDeviceBuffer> {
            CudaDeviceBuffer::alloc(bytes).map_err(|e| anyhow!("alloc {label}: {e}"))
        };
        let noise_token_ids = alloc(q_len_max * U32_BYTES, "noise_token_ids")?;
        let noise_embedding = alloc(
            q_len_max * config.hidden_size * BF16_BYTES,
            "noise_embedding",
        )?;
        let logits = alloc(q_len_max * config.vocab_size * BF16_BYTES, "propose_logits")?;
        let sampled_tokens = alloc(q_len_max * U32_BYTES, "sampled_tokens")?;
        let gemm_workspace = alloc(LM_HEAD_GEMM_WORKSPACE_BYTES, "lm_head_gemm_workspace")?;

        let bytes_total = noise_token_ids.bytes()
            + noise_embedding.bytes()
            + logits.bytes()
            + sampled_tokens.bytes()
            + gemm_workspace.bytes();
        Ok(Self {
            q_len_max,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            noise_token_ids,
            noise_embedding,
            logits,
            sampled_tokens,
            gemm_workspace,
            bytes_total,
        })
    }

    pub fn total_bytes(&self) -> usize {
        self.bytes_total
    }
}

/// Run one drafter "propose" step. Caller must have:
///   - reset the drafter's KV cache (or set it to the desired
///     `current_kv_len`),
///   - filled the drafter workspace's `position_ids` buffer with
///     `[0, current_kv_len + ctx_len + q_len)` as i32,
///   - confirmed `noise_token_ids.len() == q_len`.
///
/// Returns the `q_len` greedy-sampled token ids on host.
#[allow(clippy::too_many_arguments)]
pub fn propose_block<B: KernelBackend>(
    backend: &B,
    drafter: &mut DrafterForward<'_>,
    workspace: &DFlashProposeWorkspace,
    noise_token_ids_host: &[u32],
    target_hidden_raw_ptr: DevicePtr,
    ctx_len: usize,
    target_embed_ptr: DevicePtr,
    target_lm_head_ptr: DevicePtr,
    target_vocab_size: usize,
) -> Result<Vec<u32>> {
    let q_len = noise_token_ids_host.len();
    if q_len == 0 || q_len > workspace.q_len_max {
        bail!(
            "noise_token_ids.len() {q_len} out of range (max {})",
            workspace.q_len_max,
        );
    }
    if target_vocab_size != workspace.vocab_size {
        bail!(
            "target vocab_size {} != propose workspace vocab_size {}",
            target_vocab_size,
            workspace.vocab_size,
        );
    }
    let hidden = workspace.hidden_size;

    // 1. Upload noise token ids.
    let token_bytes: Vec<u8> = noise_token_ids_host
        .iter()
        .flat_map(|t| t.to_le_bytes())
        .collect();
    workspace
        .noise_token_ids
        .copy_from_host(&token_bytes)
        .with_context(|| "upload noise_token_ids")?;

    // 2. Embedding lookup using target's embed_tokens weight.
    backend
        .embedding_lookup(&EmbeddingLookupSpec {
            tokens: q_len,
            hidden,
            vocab_size: target_vocab_size,
            token_ids_u32: workspace.noise_token_ids.ptr(),
            embedding_bf16: target_embed_ptr,
            output_bf16: workspace.noise_embedding.ptr(),
        })
        .map_err(|e| anyhow!("embedding_lookup: {e}"))?;

    // 3. Drafter forward consumes the embedded noise + captured target
    //    hidden state. Caller pre-set `current_kv_len` + positions.
    drafter
        .forward(
            backend,
            workspace.noise_embedding.ptr(),
            target_hidden_raw_ptr,
            q_len,
            ctx_len,
        )
        .with_context(|| "drafter forward in propose_block")?;

    // 4. lm_head GEMM: m=vocab, n=q_len, k=hidden. Output is column-
    //    major [vocab, q_len] which matches what SamplingRowsSpec wants.
    backend
        .bf16_gemm(&Bf16GemmSpec {
            m: target_vocab_size,
            n: q_len,
            k: hidden,
            a_bf16: target_lm_head_ptr,
            b_bf16: drafter.workspace().output_ptr(),
            c_bf16: workspace.logits.ptr(),
            workspace: workspace.gemm_workspace.ptr(),
            workspace_bytes: workspace.gemm_workspace.bytes(),
        })
        .map_err(|e| anyhow!("lm_head bf16_gemm: {e}"))?;

    // 5. Greedy argmax across rows. temperature == 0 selects the
    //    deterministic argmax path inside the kernel.
    backend
        .sample_rows(&SamplingRowsSpec {
            rows: q_len,
            vocab_size: target_vocab_size,
            logits_bf16: workspace.logits.ptr(),
            output_token_u32: workspace.sampled_tokens.ptr(),
            mirror_last_output_token_u32: DevicePtr::NULL,
            temperature: 0.0,
        })
        .map_err(|e| anyhow!("sample_rows: {e}"))?;

    let mut tok_bytes = vec![0u8; q_len * U32_BYTES];
    workspace.sampled_tokens.copy_to_host(&mut tok_bytes)?;
    let tokens: Vec<u32> = tok_bytes
        .chunks_exact(U32_BYTES)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(tokens)
}
