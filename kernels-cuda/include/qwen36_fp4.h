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
  qwen36_device_ptr_t output_bf16;
  qwen36_attention_shape_t shape;
  // 0 = BF16, 1 = FP8 E4M3.
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
} qwen36_attention_prefill_spec_t;

typedef struct {
  size_t layer_index;
  size_t position;
  qwen36_device_ptr_t q_bf16;
  qwen36_device_ptr_t k_bf16;
  qwen36_device_ptr_t v_bf16;
  qwen36_device_ptr_t kv_cache_k;
  qwen36_device_ptr_t kv_cache_v;
  qwen36_device_ptr_t output_bf16;
  qwen36_attention_shape_t shape;
  // 0 = BF16, 1 = FP8 E4M3.
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

int qwen36_nvfp4_gemm(const qwen36_nvfp4_gemm_spec_t *spec);

// Mirage megakernel NVFP4 GEMM: hand-tuned CUTLASS kernel for the hot
// decode shapes (M » N=1, K=hidden) on Blackwell SM120. Uses the same
// Nvfp4GemmSpec contract as `qwen36_nvfp4_gemm` so callers can A/B route
// via env var. Returns QWEN36_STATUS_NOT_IMPLEMENTED for shapes the
// kernel does not yet specialise; the Rust dispatcher then falls back
// to the cuBLASLt path. See `docs/mirage-megakernel.md`.
int qwen36_megakernel_nvfp4_gemm(const qwen36_nvfp4_gemm_spec_t *spec);
int qwen36_bf16_gemm(const qwen36_bf16_gemm_spec_t *spec);
int qwen36_attention_prefill(const qwen36_attention_prefill_spec_t *spec);
int qwen36_deltanet_decode(const qwen36_deltanet_decode_spec_t *spec);
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
