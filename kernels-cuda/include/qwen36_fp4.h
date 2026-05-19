#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t ptr;
} qwen36_device_ptr_t;

typedef struct {
  qwen36_device_ptr_t ptr;
  size_t bytes;
} qwen36_device_allocation_t;

typedef struct {
  int driver_version;
  int runtime_version;
  int device_count;
  int active_device;
  int sm_major;
  int sm_minor;
  int multiprocessor_count;
  size_t total_global_mem;
  char device_name[128];
  char libcuda_path[512];
  char cudart_path[512];
  int last_cuda_error;
  char last_cuda_error_name[64];
  char last_cuda_error_string[256];
} qwen36_cuda_diagnostics_t;

typedef struct {
  uint64_t malloc_calls;
  uint64_t free_calls;
  uint64_t h2d_calls;
  uint64_t h2d_bytes;
  uint64_t d2h_calls;
  uint64_t d2h_bytes;
  uint64_t d2d_calls;
  uint64_t d2d_bytes;
  uint64_t d2d_async_calls;
  uint64_t d2d_async_bytes;
  uint64_t memset_calls;
  uint64_t memset_bytes;
  uint64_t synchronize_calls;
  uint64_t stream_synchronize_calls;
  uint64_t graph_launch_calls;
} qwen36_cuda_counters_t;

enum {
  QWEN36_STATUS_SUCCESS = 0,
  QWEN36_STATUS_NULL_POINTER = 1,
  QWEN36_STATUS_INVALID_ARGUMENT = 2,
  QWEN36_STATUS_CUDA_ERROR = 3,
  QWEN36_STATUS_CUBLAS_ERROR = 4,
  // Returned by entry points whose kernel implementation has not yet
  // landed (e.g. the Mirage megakernel path while it is being built up).
  // The Rust runtime treats this as a soft fallback signal so it can route
  // back to the existing cuBLASLt path without breaking parity.
  QWEN36_STATUS_NOT_IMPLEMENTED = 5
};

typedef struct {
  size_t m;
  size_t n;
  size_t k;
  qwen36_device_ptr_t a_fp4;
  qwen36_device_ptr_t a_scale;
  qwen36_device_ptr_t a_scale_2;
  qwen36_device_ptr_t b_fp4;
  qwen36_device_ptr_t b_scale;
  qwen36_device_ptr_t b_scale_2;
  qwen36_device_ptr_t c_bf16;
  qwen36_device_ptr_t workspace;
  size_t workspace_bytes;
  float alpha;
} qwen36_nvfp4_gemm_spec_t;

typedef struct {
  size_t m;
  size_t n;
  size_t k;
  qwen36_device_ptr_t a_bf16;
  qwen36_device_ptr_t b_bf16;
  qwen36_device_ptr_t c_bf16;
  qwen36_device_ptr_t workspace;
  size_t workspace_bytes;
} qwen36_bf16_gemm_spec_t;

typedef struct {
  size_t q_heads;
  size_t kv_heads;
  size_t head_dim;
  size_t rope_dims;
} qwen36_attention_shape_t;

typedef struct {
  size_t qk_heads;
  size_t v_heads;
  size_t key_dim;
  size_t value_dim;
  size_t conv_kernel;
} qwen36_deltanet_shape_t;

typedef struct {
  size_t layer_index;
  size_t start_position;
  size_t tokens;
  qwen36_device_ptr_t q_bf16;
  qwen36_device_ptr_t k_bf16;
  qwen36_device_ptr_t v_bf16;
  qwen36_device_ptr_t kv_cache_k;
  qwen36_device_ptr_t kv_cache_v;
  qwen36_device_ptr_t kv_cache_metadata;
  qwen36_device_ptr_t output_bf16;
  qwen36_attention_shape_t shape;
  // 0 = BF16, 1 = FP8 E4M3, 2 = TurboQuant3, 3 = TurboQuant3.5.
  int kv_cache_dtype;
  // When non-zero, prefill reads the base cache position from this device
  // pointer (int32) instead of `start_position`. Used by graph-captured MTP
  // verification where the host can no longer pass advancing scalar args.
  qwen36_device_ptr_t start_position_device_i32;
  // Optional scratch buffers backing the short-prefill split-KV path used by
  // MTP verification/recovery chunks. Reused once per token in the chunk.
  // Layout (n_splits = ceil(max_context / SPLIT_SIZE)):
  //   partial_acc_f32   : [q_heads, n_splits, head_dim] FP32
  //   partial_max_f32   : [q_heads, n_splits] FP32
  //   partial_denom_f32 : [q_heads, n_splits] FP32
  qwen36_device_ptr_t partial_acc_f32;
  qwen36_device_ptr_t partial_max_f32;
  qwen36_device_ptr_t partial_denom_f32;
  // Number of split-KV blocks per q-head. A value of 0 or 1 disables this path.
  size_t prefill_n_splits;
  // Timesteps covered by each split block. A value of 0 uses the CUDA default.
  size_t split_timesteps_per_block;
  /// Tree-mask bitmap. When non-NULL, row i of the verify chunk attends to
  /// KV row j (within the same chunk) iff bit j of word i is set. KV positions
  /// before `start_position` (cache prefix) remain fully visible regardless.
  /// NULL → causal mask (existing behaviour). Capped at 64 rows.
  qwen36_device_ptr_t tree_ancestor_bitmap_u64;
  /// Verify-chunk row count (number of valid bitmap entries). 0 = causal.
  size_t verify_chunk_rows;
} qwen36_attention_prefill_spec_t;

typedef struct {
  size_t layer_index;
  size_t position;
  qwen36_device_ptr_t q_bf16;
  qwen36_device_ptr_t k_bf16;
  qwen36_device_ptr_t v_bf16;
  qwen36_device_ptr_t kv_cache_k;
  qwen36_device_ptr_t kv_cache_v;
  qwen36_device_ptr_t kv_cache_metadata;
  qwen36_device_ptr_t output_bf16;
  qwen36_attention_shape_t shape;
  // 0 = BF16, 1 = FP8 E4M3, 2 = TurboQuant3, 3 = TurboQuant3.5.
  int kv_cache_dtype;
  // When non-zero, the kernel reads the current position from this device
  // pointer (int32, big enough to express the full max_context). This lets
  // CUDA-Graph captures stay valid across decode steps without re-recording.
  qwen36_device_ptr_t position_device_i32;
  // Optional scratch buffers backing the split-KV (FlashDecoding-style)
  // path. When all three are non-zero AND the current sequence is long
  // enough to benefit, the kernel partitions the timestep loop across
  // multiple blocks per q-head and reduces partials in a follow-up pass.
  // Layout (n_splits = ceil(max_context / SPLIT_SIZE)):
  //   partial_acc_f32   : [q_heads, n_splits, head_dim] FP32
  //   partial_max_f32   : [q_heads, n_splits] FP32
  //   partial_denom_f32 : [q_heads, n_splits] FP32
  qwen36_device_ptr_t partial_acc_f32;
  qwen36_device_ptr_t partial_max_f32;
  qwen36_device_ptr_t partial_denom_f32;
  // Number of split-KV blocks per q-head. Engine-side value derived from
  // max_context so a graph captured at one position stays valid as it grows.
  // A value of 0 or 1 disables the split path.
  size_t decode_n_splits;
  // Timesteps covered by each split block. A value of 0 uses the CUDA default.
  size_t split_timesteps_per_block;
} qwen36_attention_decode_spec_t;

typedef struct {
  size_t layer_index;
  size_t position;
  qwen36_device_ptr_t k_bf16;
  qwen36_device_ptr_t v_bf16;
  qwen36_device_ptr_t k_quantized_i8;
  qwen36_device_ptr_t v_quantized_i8;
  qwen36_device_ptr_t metadata_f32;
  qwen36_attention_shape_t shape;
} qwen36_turboquant_encode_spec_t;

typedef struct {
  size_t layer_index;
  size_t position;
  qwen36_device_ptr_t q_bf16;
  qwen36_device_ptr_t k_quantized_i8;
  qwen36_device_ptr_t v_quantized_i8;
  qwen36_device_ptr_t metadata_f32;
  qwen36_device_ptr_t output_bf16;
  qwen36_device_ptr_t workspace;
  size_t workspace_bytes;
  qwen36_attention_shape_t shape;
  int mode;
} qwen36_turboquant_attention_spec_t;

typedef struct {
  size_t layer_index;
  size_t tokens_in_persistent_loop;
  size_t q_token_stride;
  size_t k_token_stride;
  size_t v_token_stride;
  qwen36_device_ptr_t q_bf16;
  qwen36_device_ptr_t k_bf16;
  qwen36_device_ptr_t v_bf16;
  qwen36_device_ptr_t state_bf16;
  qwen36_device_ptr_t conv_history_bf16;
  qwen36_device_ptr_t output_bf16;
  qwen36_device_ptr_t gate_f32;
  qwen36_device_ptr_t beta_f32;
  qwen36_deltanet_shape_t shape;
  float state_decay;
  float update_scale;
  int qk_l2norm;
} qwen36_deltanet_decode_spec_t;

typedef struct {
  size_t layer_index;
  size_t tokens;
  size_t chunk_size;
  size_t q_token_stride;
  size_t k_token_stride;
  size_t v_token_stride;
  qwen36_device_ptr_t q_bf16;
  qwen36_device_ptr_t k_bf16;
  qwen36_device_ptr_t v_bf16;
  qwen36_device_ptr_t state_bf16;
  qwen36_device_ptr_t output_bf16;
  qwen36_device_ptr_t gate_f32;
  qwen36_device_ptr_t beta_f32;
  qwen36_device_ptr_t workspace;
  size_t workspace_bytes;
  qwen36_deltanet_shape_t shape;
  float state_decay;
  float update_scale;
  int qk_l2norm;
} qwen36_deltanet_prefill_spec_t;

typedef struct {
  size_t rows;
  size_t hidden;
  float eps;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t weight_bf16;
  qwen36_device_ptr_t residual_bf16;
  qwen36_device_ptr_t residual_out_bf16;
  qwen36_device_ptr_t output_bf16;
  int direct_weight;
} qwen36_rmsnorm_spec_t;

typedef struct {
  size_t hidden;
  float eps;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t weight_bf16;
  qwen36_device_ptr_t residual_bf16;
  qwen36_device_ptr_t residual_out_bf16;
  qwen36_device_ptr_t output_bf16;
  qwen36_device_ptr_t output_fp4;
  qwen36_device_ptr_t output_scale_e4m3;
  qwen36_device_ptr_t output_tensor_scale_f32;
  float input_tensor_scale_f32;
} qwen36_rmsnorm_nvfp4_quantize_spec_t;

typedef struct {
  size_t tokens;
  size_t q_heads;
  size_t kv_heads;
  size_t head_dim;
  size_t rope_dims;
  double base_theta;
  int32_t position_i32;
  int use_scalar_position;
  qwen36_device_ptr_t positions_i32;
  qwen36_device_ptr_t q_bf16;
  qwen36_device_ptr_t k_bf16;
  // When non-zero (and `use_scalar_position` is set), the kernel reads the
  // scalar position from this device pointer instead of `position_i32`. Used
  // by graph-captured decode where the host can no longer pass scalar args.
  qwen36_device_ptr_t scalar_position_device_i32;
} qwen36_partial_rope_spec_t;

typedef struct {
  size_t rows;
  size_t intermediate;
  qwen36_device_ptr_t gate_bf16;
  qwen36_device_ptr_t up_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_swiglu_spec_t;

typedef struct {
  size_t intermediate;
  qwen36_device_ptr_t gate_bf16;
  qwen36_device_ptr_t up_bf16;
  qwen36_device_ptr_t output_fp4;
  qwen36_device_ptr_t output_scale_e4m3;
  qwen36_device_ptr_t output_tensor_scale_f32;
  float input_tensor_scale_f32;
} qwen36_swiglu_nvfp4_quantize_spec_t;

typedef struct {
  size_t vocab_size;
  qwen36_device_ptr_t logits_bf16;
  qwen36_device_ptr_t output_token_u32;
  float temperature;
  size_t top_k;
  float top_p;
  float repetition_penalty;
  qwen36_device_ptr_t mirror_output_token_u32;
} qwen36_sampling_spec_t;

#define QWEN36_TOPK_MAX 8

typedef struct {
  size_t vocab_size;
  size_t k;                                // 1..QWEN36_TOPK_MAX
  qwen36_device_ptr_t logits_bf16;
  qwen36_device_ptr_t output_token_u32;    // [k] u32, sorted desc by logit
} qwen36_topk_argmax_spec_t;

typedef struct {
  size_t rows;
  size_t vocab_size;
  qwen36_device_ptr_t logits_bf16;
  qwen36_device_ptr_t output_token_u32;
  qwen36_device_ptr_t mirror_last_output_token_u32;
  float temperature;
} qwen36_sampling_rows_spec_t;

typedef struct {
  size_t tokens;
  size_t hidden;
  size_t vocab_size;
  qwen36_device_ptr_t token_ids_u32;
  qwen36_device_ptr_t embedding_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_embedding_lookup_spec_t;

typedef struct {
  size_t out_features;
  size_t in_features;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t weight_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_bf16_matvec_spec_t;

typedef struct {
  size_t out_features;
  size_t in_features;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t weight_u8;
  qwen36_device_ptr_t block_scale_e4m3;
  qwen36_device_ptr_t tensor_scale_f32;
  qwen36_device_ptr_t output_bf16;
} qwen36_nvfp4_matvec_spec_t;

typedef struct {
  size_t values;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t output_fp4;
  qwen36_device_ptr_t output_scale_e4m3;
  qwen36_device_ptr_t output_tensor_scale_f32;
  float input_tensor_scale_f32;
} qwen36_nvfp4_quantize_spec_t;

typedef struct {
  size_t rows;
  size_t values;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t output_fp4;
  qwen36_device_ptr_t output_scale_e4m3;
  qwen36_device_ptr_t output_tensor_scale_f32;
  float input_tensor_scale_f32;
} qwen36_nvfp4_quantize_rows_spec_t;

typedef struct {
  size_t rows;
  size_t inner_groups;
  qwen36_device_ptr_t input_row_major_u8;
  qwen36_device_ptr_t output_tiled_u8;
} qwen36_nvfp4_retile_scales_spec_t;

typedef struct {
  size_t channels;
  size_t kernel_size;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t conv_history_bf16;
  qwen36_device_ptr_t weight_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_conv1d_update_spec_t;

typedef struct {
  size_t tokens;
  size_t channels;
  size_t kernel_size;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t conv_history_bf16;
  qwen36_device_ptr_t weight_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_conv1d_prefill_spec_t;

typedef struct {
  size_t rows;
  size_t heads;
  qwen36_device_ptr_t a_bf16;
  qwen36_device_ptr_t b_bf16;
  qwen36_device_ptr_t a_log_bf16;
  qwen36_device_ptr_t dt_bias_bf16;
  qwen36_device_ptr_t gate_f32;
  qwen36_device_ptr_t beta_f32;
} qwen36_gdn_gate_spec_t;

typedef struct {
  // conv1d_update params (rows = 1).
  size_t channels;
  size_t kernel_size;
  qwen36_device_ptr_t conv_input_bf16;
  qwen36_device_ptr_t conv_history_bf16;
  qwen36_device_ptr_t conv_weight_bf16;
  qwen36_device_ptr_t conv_output_bf16;
  // gdn_gate params (rows = 1, single-token decode).
  size_t heads;
  qwen36_device_ptr_t gdn_a_bf16;
  qwen36_device_ptr_t gdn_b_bf16;
  qwen36_device_ptr_t gdn_a_log_bf16;
  qwen36_device_ptr_t gdn_dt_bias_bf16;
  qwen36_device_ptr_t gate_f32;
  qwen36_device_ptr_t beta_f32;
} qwen36_conv1d_gdn_gate_fused_spec_t;

typedef struct {
  size_t elements;
  qwen36_device_ptr_t gate_bf16;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_sigmoid_gate_spec_t;

typedef struct {
  size_t rows;
  size_t elements_per_row;
  size_t gate_stride;
  size_t input_stride;
  size_t output_stride;
  qwen36_device_ptr_t gate_bf16;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_sigmoid_gate_strided_spec_t;

typedef struct {
  size_t rows;
  size_t heads;
  size_t head_dim;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_q_proj_deinterleave_spec_t;

typedef struct {
  size_t rows;
  size_t heads;
  size_t head_dim;
  qwen36_device_ptr_t gate_bf16;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_q_proj_sigmoid_gate_spec_t;

typedef struct {
  size_t rows;
  size_t values;
  size_t input_stride;
  size_t output_stride;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_copy_strided_rows_spec_t;

int qwen36_cuda_malloc(qwen36_device_allocation_t *out, size_t bytes);
int qwen36_cuda_free(qwen36_device_ptr_t ptr);
int qwen36_cuda_memcpy_h2d(qwen36_device_ptr_t dst, const void *src,
                           size_t bytes);
int qwen36_cuda_memcpy_d2h(void *dst, qwen36_device_ptr_t src, size_t bytes);
int qwen36_cuda_memcpy_d2d(qwen36_device_ptr_t dst, qwen36_device_ptr_t src,
                           size_t bytes);
int qwen36_cuda_memcpy_d2d_async(qwen36_device_ptr_t dst,
                                 qwen36_device_ptr_t src, size_t bytes);
int qwen36_cuda_memset(qwen36_device_ptr_t dst, int value, size_t bytes);
int qwen36_cuda_synchronize(void);
int qwen36_cuda_get_diagnostics(qwen36_cuda_diagnostics_t *out);
int qwen36_cuda_counters_reset(void);
int qwen36_cuda_counters_read(qwen36_cuda_counters_t *out);
int qwen36_cuda_set_l2_access_window(qwen36_device_ptr_t base, size_t bytes,
                                      float hit_ratio);
int qwen36_cuda_clear_l2_access_window(void);

// Productive-spin L2 warmup. Read-only walk over `[base, base+bytes)` that
// pulls every byte into L2 without computing. Launches `target_cta_count`
// CTAs (default 128 when 0) onto the registered prefetch stream so it can
// overlap the small-CTA full-attn decode kernel running on the main stream.
// Returns QWEN36_STATUS_INVALID_ARGUMENT if no prefetch stream is registered.
int qwen36_l2_prefetch(qwen36_device_ptr_t base, size_t bytes,
                       int target_cta_count);

// Per-block megakernel for full-attn decode — Stage A (skeleton).
// Identity copy of `hidden_in` to `hidden_out` through a single phase
// barrier. Exercises the persistent-grid + atomic-barrier infrastructure
// that Stages B-F will plug computation into. Caller must zero
// `barrier_state` (≥ `phases * 4` bytes; Stage A uses 4 bytes) before
// every launch. Runs on the active stream.
int qwen36_full_attn_block_stage_a(qwen36_device_ptr_t hidden_in,
                                   qwen36_device_ptr_t hidden_out,
                                   qwen36_device_ptr_t barrier_state,
                                   size_t hidden_size);

// Per-block megakernel — Stage B.1: RMSNorm phase only. CTA 0 cooperates
// on one row of length `hidden_size` (decode is N=1) using the (1+weight)
// parameterization (matches Qwen base layer norms). Other CTAs idle at
// the barrier; later stages fill them. Designed to be byte-exact with
// `qwen36_rmsnorm` invoked with the same input, no residual, and
// `direct_weight = 0`. `barrier_state` ≥ 4 bytes, zeroed by caller.
int qwen36_full_attn_block_stage_b_rmsnorm(
    qwen36_device_ptr_t hidden_in, qwen36_device_ptr_t input_norm_weight,
    qwen36_device_ptr_t hidden_normed_out, qwen36_device_ptr_t barrier_state,
    size_t hidden_size, float eps);

// Per-block megakernel — Stage B.2: fused RMSNorm + NVFP4 quantize phase.
// Produces FP4-packed bytes + e4m3 per-block scales (in the vec16_scale
// tile layout the Q proj GEMV expects) so downstream stages can chain
// directly. Optional bf16 normed copy and f32 tensor scale propagation
// match `qwen36_rmsnorm_nvfp4_quantize`. Caller pre-zeroes `barrier_state`.
// Pass `input_tensor_scale = 0.0` to use 1.0 as the global scale.
int qwen36_full_attn_block_stage_b_rmsnorm_quantize(
    qwen36_device_ptr_t hidden_in, qwen36_device_ptr_t input_norm_weight,
    qwen36_device_ptr_t hidden_normed_out_bf16,
    qwen36_device_ptr_t output_fp4, qwen36_device_ptr_t output_scale_e4m3,
    qwen36_device_ptr_t output_tensor_scale_f32,
    qwen36_device_ptr_t barrier_state, size_t hidden_size, float eps,
    float input_tensor_scale);

// Per-block megakernel — Stage B.3: Stage B.2 + Q projection NVFP4 GEMV
// fused into one launch. Phase layout: CTA 0 RMSNorm+quantize → barrier →
// all CTAs run the 8-warp GEMV body for Q (each CTA owns one m16 row tile)
// → barrier. Grid = ceil(q_features / 16); block = 256 threads = 8 warps;
// caller pre-zeroes `barrier_state` (≥ 2 × 4 bytes). `hidden_size` must be
// a multiple of 512 (the 8-warp GEMV K-shard alignment) and `q_features`
// must be a multiple of 16. `q_alpha` is the pre-folded product of the
// per-tensor scales (`q_weight_tensor_scale * input_tensor_scale`).
int qwen36_full_attn_block_stage_b_q_proj(
    qwen36_device_ptr_t hidden_in, qwen36_device_ptr_t input_norm_weight,
    qwen36_device_ptr_t q_weight_fp4, qwen36_device_ptr_t q_weight_scale,
    float q_alpha, qwen36_device_ptr_t hidden_normed_out_bf16,
    qwen36_device_ptr_t quantized_fp4,
    qwen36_device_ptr_t quantized_scale_e4m3, qwen36_device_ptr_t q_out,
    qwen36_device_ptr_t barrier_state, size_t hidden_size, size_t q_features,
    float eps, float input_tensor_scale);

// Per-block megakernel — Stage E: attention output → NVFP4 quantize →
// o_proj GEMV → residual add → post-attn RMSNorm + NVFP4 quantize, all
// fused into one launch. Picks up where Stage D (attention) leaves off
// in the full-attn layer pipeline. Caller pre-zeroes `barrier_state`
// (≥ 4 × 4 bytes). `q_features` % 512 == 0 (GEMV K-shard alignment);
// `hidden_size` % 16 == 0 (GEMV m-tile alignment). `o_alpha` is the
// pre-folded product of the o_proj per-tensor scales. Pass tensor
// scales (≤ 0 selects 1.0 internally).
int qwen36_full_attn_block_stage_e_o_proj_residual_norm(
    qwen36_device_ptr_t attention_out, qwen36_device_ptr_t residual_in,
    qwen36_device_ptr_t o_proj_fp4, qwen36_device_ptr_t o_proj_scale,
    float o_alpha, qwen36_device_ptr_t post_norm_weight,
    qwen36_device_ptr_t attention_quantized_fp4,
    qwen36_device_ptr_t attention_quantized_scale,
    qwen36_device_ptr_t o_proj_out, qwen36_device_ptr_t residual_out,
    qwen36_device_ptr_t post_normed_out,
    qwen36_device_ptr_t post_quantized_fp4,
    qwen36_device_ptr_t post_quantized_scale,
    qwen36_device_ptr_t barrier_state, size_t q_features, size_t hidden_size,
    float eps, float post_input_tensor_scale,
    float attention_output_tensor_scale);

// Per-block megakernel — Stage C: Stage B.3 + K projection + V projection
// + partial RoPE on Q/K, fused into one launch. Grid sized to the widest
// phase (Q proj: ceil(q_features/16) CTAs); smaller K/V phases share the
// same grid and skip tail CTAs via the GEMV body's bounds checks.
// Caller pre-zeroes `barrier_state` (≥ 5 × 4 bytes for 5 phase barriers).
// `hidden_size` % 512 == 0, `q_features` % 16 == 0, `kv_features` % 16 == 0.
// `position` is the scalar token position (decode is N=1); `base_theta`
// matches the model's RoPE theta. Partial RoPE is split-half (Qwen3.6
// convention), applied in place to `q_out` and `k_out`.
int qwen36_full_attn_block_stage_c_qkv_rope(
    qwen36_device_ptr_t hidden_in, qwen36_device_ptr_t input_norm_weight,
    qwen36_device_ptr_t q_weight_fp4, qwen36_device_ptr_t q_weight_scale,
    float q_alpha, qwen36_device_ptr_t k_weight_fp4,
    qwen36_device_ptr_t k_weight_scale, float k_alpha,
    qwen36_device_ptr_t v_weight_fp4, qwen36_device_ptr_t v_weight_scale,
    float v_alpha, qwen36_device_ptr_t hidden_normed_out_bf16,
    qwen36_device_ptr_t quantized_fp4,
    qwen36_device_ptr_t quantized_scale_e4m3, qwen36_device_ptr_t q_out,
    qwen36_device_ptr_t k_out, qwen36_device_ptr_t v_out,
    qwen36_device_ptr_t barrier_state, size_t hidden_size, size_t q_features,
    size_t kv_features, size_t q_heads, size_t kv_heads, size_t head_dim,
    size_t rope_dims, int32_t position, float base_theta, float eps,
    float input_tensor_scale);

// Per-block megakernel — Stage F.1: MLP gate+up NVFP4 GEMV. First
// sub-phase of the MLP fusion. Output is BF16 [2 * intermediate]
// arranged as gate||up (matching the engine's combined gate+up store).
//
// Uses a persistent grid + atomic work counter: m-tiles (2*intermediate
// / 16 of them, e.g. 2176 for Qwen3.6) are distributed across a fixed
// CTA pool sized to the SM's true concurrent capacity. The caller
// pre-zeroes `barrier_state` (the first 4 bytes are the work counter;
// no inter-CTA spinlock is used in F.1 because it is the only phase
// today). Alignment: 2*intermediate % 16 == 0, hidden_size % 512 == 0.
// `gate_up_alpha = gate_up_weight_tensor_scale * input_tensor_scale`
// is folded host-side. Activation is the post-attn-quantized FP4
// produced by Stage E (so this kernel is the natural successor in
// the full-attn pipeline). F.2/F.3/F.4 will append SwiGLU + down GEMV
// + residual_add phases on the same launch and use the remaining
// barrier slots (still safe because the persistent grid stays ≤
// concurrent CTA capacity).
int qwen36_full_attn_block_stage_f1_gate_up(
    qwen36_device_ptr_t hidden_quantized_fp4,
    qwen36_device_ptr_t hidden_quantized_scale,
    qwen36_device_ptr_t mlp_gate_up_fp4,
    qwen36_device_ptr_t mlp_gate_up_scale, float gate_up_alpha,
    qwen36_device_ptr_t gate_up_out, qwen36_device_ptr_t barrier_state,
    size_t intermediate, size_t hidden_size);

// Per-block megakernel — Stage F.2: Stage F.1 + fused SwiGLU + NVFP4
// quantize of the down-projection input. Outputs in addition to
// `gate_up_out` (BF16, [2 * intermediate], gate||up): `swiglu_fp4`
// (packed, [intermediate / 2]) and `swiglu_scale` (e4m3 per 16-element
// group, vec16 tile layout matching `qwen36_swiglu_nvfp4_quantize`).
// `barrier_state` must hold ≥ 4 zeroed u32 slots (two work counters +
// two phase spinlocks). `down_input_tensor_scale` is the pre-folded
// per-tensor scale for the down GEMV's NVFP4 input — equivalent to
// the standalone path's `input_tensor_scale_f32`.
int qwen36_full_attn_block_stage_f2_gate_up_swiglu(
    qwen36_device_ptr_t hidden_quantized_fp4,
    qwen36_device_ptr_t hidden_quantized_scale,
    qwen36_device_ptr_t mlp_gate_up_fp4,
    qwen36_device_ptr_t mlp_gate_up_scale, float gate_up_alpha,
    qwen36_device_ptr_t gate_up_out, qwen36_device_ptr_t swiglu_fp4,
    qwen36_device_ptr_t swiglu_scale, qwen36_device_ptr_t barrier_state,
    size_t intermediate, size_t hidden_size,
    float down_input_tensor_scale);

int qwen36_nvfp4_gemm(const qwen36_nvfp4_gemm_spec_t *spec);

// Mirage megakernel NVFP4 GEMM: hand-tuned CUTLASS kernel for the hot
// decode shapes (M » N=1, K=hidden) on Blackwell SM120. Uses the same
// Nvfp4GemmSpec contract as `qwen36_nvfp4_gemm` so callers can A/B route
// via env var. Returns QWEN36_STATUS_NOT_IMPLEMENTED for shapes the
// kernel does not yet specialise; the Rust dispatcher then falls back
// to the cuBLASLt path. See `docs/mirage-megakernel.md`.
int qwen36_megakernel_nvfp4_gemm(const qwen36_nvfp4_gemm_spec_t *spec);
// Direction B decode-time NVFP4 gemv: hand-written kernel optimised for the
// (M, N=1, K) shapes that dominate decode. Reuses `qwen36_nvfp4_gemm_spec_t`.
// Returns QWEN36_STATUS_NOT_IMPLEMENTED (5) for shapes outside the supported
// set (M%128==0, K%128==0, N==1); the Rust dispatcher falls back to the
// existing megakernel/cuBLASLt path on that code, mirroring the Mirage
// pattern. Gated by `QWEN36_DECODE_GEMV=1`. See
// `docs/superpowers/specs/2026-05-04-direction-b-nvfp4-gemv-design.md`.
int qwen36_decode_nvfp4_gemv(const qwen36_nvfp4_gemm_spec_t *spec);
int qwen36_bf16_gemm(const qwen36_bf16_gemm_spec_t *spec);
int qwen36_attention_prefill(const qwen36_attention_prefill_spec_t *spec);
int qwen36_attention_flash_prefill(const qwen36_attention_prefill_spec_t *spec);
int qwen36_attention_sage_prefill(const qwen36_attention_prefill_spec_t *spec);
int qwen36_deltanet_decode(const qwen36_deltanet_decode_spec_t *spec);
int qwen36_deltanet_prefill(const qwen36_deltanet_prefill_spec_t *spec);
int qwen36_attention_decode(const qwen36_attention_decode_spec_t *spec);
int qwen36_turboquant_encode_kv(const qwen36_turboquant_encode_spec_t *spec);
int qwen36_turboquant_attention(const qwen36_turboquant_attention_spec_t *spec);
int qwen36_rmsnorm(const qwen36_rmsnorm_spec_t *spec);
int qwen36_rmsnorm_nvfp4_quantize(
    const qwen36_rmsnorm_nvfp4_quantize_spec_t *spec);
int qwen36_partial_rope(const qwen36_partial_rope_spec_t *spec);
int qwen36_swiglu(const qwen36_swiglu_spec_t *spec);
int qwen36_swiglu_nvfp4_quantize(
    const qwen36_swiglu_nvfp4_quantize_spec_t *spec);
int qwen36_sample(const qwen36_sampling_spec_t *spec);
int qwen36_sample_rows(const qwen36_sampling_rows_spec_t *spec);
int qwen36_topk_argmax(const qwen36_topk_argmax_spec_t *spec);
int qwen36_embedding_lookup(const qwen36_embedding_lookup_spec_t *spec);
int qwen36_bf16_matvec(const qwen36_bf16_matvec_spec_t *spec);
int qwen36_nvfp4_matvec(const qwen36_nvfp4_matvec_spec_t *spec);
int qwen36_nvfp4_quantize_bf16(const qwen36_nvfp4_quantize_spec_t *spec);
int qwen36_nvfp4_quantize_rows(const qwen36_nvfp4_quantize_rows_spec_t *spec);
int qwen36_nvfp4_retile_scales(const qwen36_nvfp4_retile_scales_spec_t *spec);
int qwen36_conv1d_update(const qwen36_conv1d_update_spec_t *spec);
int qwen36_conv1d_prefill(const qwen36_conv1d_prefill_spec_t *spec);
int qwen36_gdn_gate(const qwen36_gdn_gate_spec_t *spec);
int qwen36_conv1d_gdn_gate_fused(
    const qwen36_conv1d_gdn_gate_fused_spec_t *spec);
int qwen36_sigmoid_gate(const qwen36_sigmoid_gate_spec_t *spec);
int qwen36_sigmoid_gate_strided(
    const qwen36_sigmoid_gate_strided_spec_t *spec);
int qwen36_q_proj_deinterleave(
    const qwen36_q_proj_deinterleave_spec_t *spec);
int qwen36_q_proj_sigmoid_gate(
    const qwen36_q_proj_sigmoid_gate_spec_t *spec);
int qwen36_copy_strided_rows(const qwen36_copy_strided_rows_spec_t *spec);

// Atomically advance a device-side int32 by 1. Used to step the decode
// position inside a captured CUDA graph so the same graph can be replayed
// across decode iterations without host parameter updates.
int qwen36_increment_i32(qwen36_device_ptr_t target_i32);

// Advance graph-captured assume-accept MTP positions in-place:
//   position_i32[0..count] += position_delta
// Token samplers write the next window directly into token_u32, so repeated
// CUDA graph launches consume prior MTP outputs without host readback/re-upload.
int qwen36_mtp_assume_accept_chain_advance(qwen36_device_ptr_t position_i32,
                                           size_t draft_count,
                                           size_t position_count,
                                           int32_t position_delta);

// All kernel launches funnel through a single ambient CUDA stream so callers
// can switch streams (e.g. to enable graph capture) without touching every
// kernel call site. The default value is the legacy default stream (0),
// matching the historical behavior.
typedef struct CUstream_st *qwen36_cuda_stream_t;
qwen36_cuda_stream_t qwen36_get_active_stream(void);
void qwen36_set_active_stream(qwen36_cuda_stream_t stream);
int qwen36_cuda_stream_create(qwen36_cuda_stream_t *out);
int qwen36_cuda_stream_destroy(qwen36_cuda_stream_t stream);
int qwen36_cuda_stream_synchronize(qwen36_cuda_stream_t stream);

// Secondary "prefetch" stream used by the decode path to overlap idle-SM L2
// prefetch (productive spin during full-attn) and any future megakernel-side
// concurrent work with the main stream. Lifetime is owned by the engine: the
// engine creates it at boot via `qwen36_cuda_stream_create` (cudaStreamNon-
// Blocking) and registers it here with `qwen36_set_prefetch_stream`. Kernels
// that want to dispatch onto it use `qwen36_internal_prefetch_stream` from
// active_stream.h. Defaults to nullptr (= unused).
qwen36_cuda_stream_t qwen36_get_prefetch_stream(void);
void qwen36_set_prefetch_stream(qwen36_cuda_stream_t stream);

// Generic CUDA event handle for cross-stream synchronization. Used by the
// productive-spin and megakernel paths to record an event on one stream and
// have another stream wait on it; the pattern is graph-captureable so the
// decode CUDA graph can include cross-stream waits without re-recording.
// Events are created with `cudaEventDisableTiming` to avoid the timer cost.
typedef struct CUevent_st *qwen36_cuda_event_t;
int qwen36_cuda_event_create(qwen36_cuda_event_t *out);
int qwen36_cuda_event_destroy(qwen36_cuda_event_t event);
int qwen36_cuda_event_record(qwen36_cuda_event_t event,
                             qwen36_cuda_stream_t stream);
int qwen36_cuda_stream_wait_event(qwen36_cuda_stream_t stream,
                                  qwen36_cuda_event_t event);

// Opaque handles for CUDA graph plumbing.
typedef struct CUgraph_st *qwen36_cuda_graph_t;
typedef struct CUgraphExec_st *qwen36_cuda_graph_exec_t;

int qwen36_cuda_stream_begin_capture(qwen36_cuda_stream_t stream);
int qwen36_cuda_stream_end_capture(qwen36_cuda_stream_t stream,
                                   qwen36_cuda_graph_t *out);
int qwen36_cuda_graph_instantiate(qwen36_cuda_graph_t graph,
                                  qwen36_cuda_graph_exec_t *out);
int qwen36_cuda_graph_destroy(qwen36_cuda_graph_t graph);
int qwen36_cuda_graph_exec_destroy(qwen36_cuda_graph_exec_t exec);
int qwen36_cuda_graph_launch(qwen36_cuda_graph_exec_t exec,
                             qwen36_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
