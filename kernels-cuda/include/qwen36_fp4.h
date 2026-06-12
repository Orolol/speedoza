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
  // Returned by entry points for shapes their kernel does not support
  // (e.g. the decode NVFP4 gemv outside its specialised shape set).
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

/* lm_head FP8 e4m3 (W8A16). Quantize is a one-shot init pass; the gemv
 * serves both the single-token logits (n=1) and the batched MTP verify
 * logits (n <= QWEN36_LM_HEAD_FP8_MAX_N). Layouts: input [n, cols] and
 * output [n, rows], both row-major BF16. */
#define QWEN36_LM_HEAD_FP8_MAX_N 16

typedef struct {
  size_t rows; /* vocab */
  size_t cols; /* hidden; must be a multiple of 4 */
  qwen36_device_ptr_t weight_bf16;
  qwen36_device_ptr_t weight_e4m3;
  qwen36_device_ptr_t row_scales_f32;
} qwen36_lm_head_fp8_quantize_spec_t;

typedef struct {
  size_t rows;
  size_t cols;
  size_t n;
  qwen36_device_ptr_t weight_e4m3;
  qwen36_device_ptr_t row_scales_f32;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_lm_head_fp8_gemv_spec_t;

typedef struct {
  size_t rows;
  size_t out_features;
  size_t in_features;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t weight_bf16;
  qwen36_device_ptr_t output_token_u32;
  qwen36_device_ptr_t mirror_last_output_token_u32;
  qwen36_device_ptr_t workspace;
  size_t workspace_bytes;
  /* Optional [rows] u32 device flags: a non-zero flag SKIPS that row (its
   * matvec grid exits immediately and finalize leaves output/mirror
   * untouched). NULL (ptr 0) processes every row — the historical
   * behavior. Used as the predicated BF16 rescore of the two-stage exact
   * lm_head argmax (flags come from qwen36_lm_head_top2_margin). */
  qwen36_device_ptr_t skip_flags_u32;
} qwen36_bf16_matvec_argmax_rows_spec_t;

/* Two-stage exact lm_head argmax, stage 1 verdict: per-row top-2 over the
 * FP8-path logits [rows, vocab] (row-major BF16). Writes per row:
 *   tokens_u32[row]  = FP8 argmax token (unconditionally; stage 2
 *                      overwrites it only when the guard fails)
 *   flags_u32[row]   = 1 when margin top1-top2 >= eps (FP8 argmax provably
 *                      equals the BF16 argmax when eps >= 2*max|dlogit|),
 *                      0 when stage 2 must rescore the row in BF16.
 * Optional mirror gets the LAST row's token (same contract as the argmax
 * rows finalize). fallback_count_u32 (optional) is atomically incremented
 * by the number of rows whose guard failed — a cheap device counter the
 * bench reads to report the measured fallback rate.
 * Workspace: rows * QWEN36_LM_HEAD_TOP2_BLOCKS * 16 bytes. */
#define QWEN36_LM_HEAD_TOP2_BLOCKS 240

typedef struct {
  size_t rows;
  size_t vocab;
  float eps;
  qwen36_device_ptr_t logits_bf16;
  qwen36_device_ptr_t tokens_u32;
  qwen36_device_ptr_t flags_u32;
  qwen36_device_ptr_t mirror_last_token_u32;
  qwen36_device_ptr_t fallback_count_u32;
  qwen36_device_ptr_t workspace;
  size_t workspace_bytes;
} qwen36_lm_head_top2_margin_spec_t;

/* Two-stage exact lm_head argmax, v2 ("top-8 rescore"): stage 1 verdict
 * that rescores the top-8 FP8 candidates against the BF16 weight instead
 * of guarding on the top1-top2 margin (v1 fell back ~42% on the
 * tight-margin MTP-head rows). Per row of the FP8-path logits:
 *   - block top-2 pass (shared with the v1 entry) -> workspace
 *   - select the top-8 BLOCK WINNERS + guard bound
 *         B = max(9th winner, max block-v2)
 *     (a block hiding two leaders raises its v2 -> B -> guard fails ->
 *     conservative full fallback, never a silent miss)
 *   - rescore the 8 candidates against weight_bf16/input_bf16 in FP64
 *     accumulation (8 x cols FMAs, order-stable)
 *   - guard: best_rescored >= B + eps certifies every non-candidate
 *     (bf16 logit <= fp8 logit + e_max <= B + e_max); recommended
 *     eps ~ 1.5x the probe e_max (0.344) = 0.5.
 * Outputs match the v1 contract: tokens written unconditionally,
 * flags_u32[row]=1 when certified (else the predicated full BF16 rescore
 * runs), optional mirror of the last row, fallback counter. Workspace:
 * rows * QWEN36_LM_HEAD_TOP2_BLOCKS * 16 bytes. */
typedef struct {
  size_t rows;
  size_t vocab;
  size_t cols;
  float eps;
  qwen36_device_ptr_t logits_bf16;  /* [rows, vocab] FP8-path logits */
  qwen36_device_ptr_t weight_bf16;  /* [vocab, cols] lm_head */
  qwen36_device_ptr_t input_bf16;   /* [rows, cols] hidden vectors */
  qwen36_device_ptr_t tokens_u32;
  qwen36_device_ptr_t flags_u32;
  qwen36_device_ptr_t mirror_last_token_u32;
  qwen36_device_ptr_t fallback_count_u32;
  qwen36_device_ptr_t workspace;
  size_t workspace_bytes;
} qwen36_lm_head_top8_rescore_spec_t;

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

#define QWEN36_INTERPRETER_MAX_DEPS 4
#define QWEN36_INTERPRETER_PAYLOAD_U64S 12

typedef struct {
  uint32_t counter_id;
  uint32_t target;
} qwen36_interpreter_dep_t;

typedef struct {
  uint16_t opcode;
  uint16_t flags;
  uint16_t dep_count;
  uint16_t reserved;
  uint32_t publishes_counter;
  uint32_t publish_value;
  // Per-instruction arrival slot. Every CTA increments this after executing
  // the opcode body; only the last CTA publishes `publishes_counter`.
  uint32_t arrival_counter;
  qwen36_interpreter_dep_t deps[QWEN36_INTERPRETER_MAX_DEPS];
  // Opaque per-opcode payload. Stage 0 only dispatches fallback/no-op
  // instructions; future inline opcode bodies reinterpret these slots.
  uint64_t payload[QWEN36_INTERPRETER_PAYLOAD_U64S];
} qwen36_interpreter_instruction_t;

typedef struct {
  qwen36_device_ptr_t instructions;
  size_t instruction_count;
  qwen36_device_ptr_t counters_i32;
  size_t counter_count;
  // 0 means derive from cudaOccupancyMaxActiveBlocksPerMultiprocessor.
  uint32_t cta_count;
  uint32_t flags;
} qwen36_interpreter_program_t;

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

int qwen36_nvfp4_gemm(const qwen36_nvfp4_gemm_spec_t *spec);

// Stage-0 interpreter shell for decode. It executes a static instruction
// stream, waits/publishes GMEM counters, initialises the SMEM page allocator,
// and routes known non-EXIT opcodes through a device-side fallback no-op.
// Real opcode bodies are added behind the same ABI in later stages.
int qwen36_interpreter_decode_sm120(const qwen36_interpreter_program_t *program);

// Direction B decode-time NVFP4 gemv: hand-written kernel optimised for the
// (M, N=1, K) shapes that dominate decode. Reuses `qwen36_nvfp4_gemm_spec_t`.
// Returns QWEN36_STATUS_NOT_IMPLEMENTED (5) for shapes outside the supported
// set (M%128==0, K%128==0, N==1); the Rust dispatcher falls back to the
// existing cuBLASLt path on that code. Gated by `QWEN36_DECODE_GEMV=1`. See
// `docs/superpowers/specs/2026-05-04-direction-b-nvfp4-gemv-design.md`.
int qwen36_decode_nvfp4_gemv(const qwen36_nvfp4_gemm_spec_t *spec);
// Test/bench hook for the multi-N chunk path: force the M-tiles-per-CTA
// factor (>=1), or pass 0 to restore the heuristic/env default
// (QWEN36_CHUNK_GEMV_MTILE). Smoke uses this to sweep T on one shape.
void qwen36_nvfp4_chunk_gemv_set_mtile(int mtile);
int qwen36_bf16_gemm(const qwen36_bf16_gemm_spec_t *spec);
int qwen36_attention_prefill(const qwen36_attention_prefill_spec_t *spec);
int qwen36_attention_flash_prefill(const qwen36_attention_prefill_spec_t *spec);
int qwen36_attention_sage_prefill(const qwen36_attention_prefill_spec_t *spec);
int qwen36_deltanet_decode(const qwen36_deltanet_decode_spec_t *spec);
/* Test/bench hook for the FP32-resident multi-token DeltaNet path:
 * 1 forces it ON, 0 forces it OFF (generic per-token body), any other
 * value restores the default (resident when eligible: gated, tokens > 1,
 * key_dim = value_dim = 128; env kill: QWEN36_DELTANET_RESIDENT=0). */
void qwen36_deltanet_set_resident(int mode);
/* Test/bench hook for the chunked-prefill tiny-chunk (T <= 16) fast path:
 * 1 forces ON, 0 forces OFF (full C=32 phases), other values restore the
 * default (on unless QWEN36_DELTANET_TINY_CHUNK=0). Bit-exact by
 * construction; gated by memcmp in smoke. */
void qwen36_deltanet_prefill_set_tiny(int mode);
int qwen36_deltanet_prefill(const qwen36_deltanet_prefill_spec_t *spec);
int qwen36_attention_decode(const qwen36_attention_decode_spec_t *spec);

// DFlash drafter attention (Phase C v1): non-causal BF16 attention with
// GQA. Caller pre-concatenates K = [k_ctx; k_noise] and V = [v_ctx;
// v_noise] as a single contiguous buffer of length `kv_seq_len`. No KV
// cache management inside the kernel; the controller manages append/crop
// at the Rust layer. `sliding_window = 0` selects full attention; non-
// zero applies a symmetric SWA mask centred on the query's absolute
// position (key absolute position is `j`, query absolute position is
// `kv_seq_len - q_len + q_pos`).
typedef struct {
  qwen36_device_ptr_t q_bf16;        // [q_len, q_heads, head_dim]
  qwen36_device_ptr_t k_bf16;        // [kv_seq_len, kv_heads, head_dim]
  qwen36_device_ptr_t v_bf16;        // [kv_seq_len, kv_heads, head_dim]
  qwen36_device_ptr_t output_bf16;   // [q_len, q_heads, head_dim]
  size_t q_len;
  size_t kv_seq_len;
  size_t q_heads;
  size_t kv_heads;
  size_t head_dim;
  size_t sliding_window;             // 0 = full attention
} qwen36_drafter_attention_block_spec_t;

int qwen36_drafter_attention_block_bf16(
    const qwen36_drafter_attention_block_spec_t *spec);

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
int qwen36_bf16_matvec_argmax_rows(
    const qwen36_bf16_matvec_argmax_rows_spec_t *spec);
int qwen36_lm_head_fp8_quantize(const qwen36_lm_head_fp8_quantize_spec_t *spec);
int qwen36_lm_head_fp8_gemv(const qwen36_lm_head_fp8_gemv_spec_t *spec);
int qwen36_lm_head_top2_margin(const qwen36_lm_head_top2_margin_spec_t *spec);
int qwen36_lm_head_top8_rescore(const qwen36_lm_head_top8_rescore_spec_t *spec);
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
// prefetch (productive spin during full-attn) and any future concurrent
// work with the main stream. Lifetime is owned by the engine: the
// engine creates it at boot via `qwen36_cuda_stream_create` (cudaStreamNon-
// Blocking) and registers it here with `qwen36_set_prefetch_stream`. Kernels
// that want to dispatch onto it use `qwen36_internal_prefetch_stream` from
// active_stream.h. Defaults to nullptr (= unused).
qwen36_cuda_stream_t qwen36_get_prefetch_stream(void);
void qwen36_set_prefetch_stream(qwen36_cuda_stream_t stream);

// Generic CUDA event handle for cross-stream synchronization. Used by the
// productive-spin path to record an event on one stream and
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
