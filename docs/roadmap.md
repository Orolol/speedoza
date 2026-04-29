# Roadmap

This repo is currently in bring-up mode. The path to the final target is:

## Phase 0: Discovery

- Generate `model_layout.json`.
- Confirm tensor names, dtypes, shapes, and ignore-list.
- Confirm topology from `config.json`.

Status: implemented.

## Phase 1: CUDA ABI And Baseline Kernels

- Build shared CUDA library.
- Call CUDA kernels from Rust through typed specs.
- Validate attention, quantized KV, RMSNorm, partial RoPE, SwiGLU, sampling, DeltaNet state layout, and FP4 GEMM independently.

Status: implemented as baseline kernels.

## Phase 2: Real Checkpoint Binding

- Bind actual model weights to runtime layer descriptors.
- Validate required NVFP4 triplets and bf16 exceptions against the downloaded checkpoint.
- Add zero-copy safetensors tensor access for future GPU upload.

Status: implemented for metadata/manifest validation; GPU upload is pending.

## Phase 3: Reference Forward Pass

- Implement RMSNorm, projections, attention, DeltaNet, FFN, residuals, logits.
- Compare against Transformers on short prompts.

Status: pending.

## Phase 4: Exact DeltaNet

- Add conv1d history path.
- Implement exact beta/gate/delta-rule recurrence.
- Add snapshot/restore correctness tests for MTP rollback.
- Replace bring-up recurrence.

Status: exact decode recurrence and rollback controller tests implemented; conv1d/prefill/projection fusion pending.

## Phase 5: TurboQuant

- Replace int8 per-vector KV path with TurboQuant rotation/QJL.
- Preserve hybrid-aware skip policy `{3, 63}`.
- Validate against BF16/FP8 attention references.

Status: pending.

## Phase 6: MTP Integration

- Wire main logits and recursive MTP draft path.
- Validate accept/reject and rollback.
- Track acceptance rate by prompt category.

Status: controller implemented, model integration pending.

## Phase 7: CUDA Graphs And Tuning

- Capture decode graphs for context buckets.
- Profile launch overhead, memory bandwidth, and Tensor Core utilization.
- Tune block sizes and persistent decode paths on RTX 5090.

Status: planning structs implemented, graph capture pending.
