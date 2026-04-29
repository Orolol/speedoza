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

enum {
  QWEN36_STATUS_SUCCESS = 0,
  QWEN36_STATUS_NULL_POINTER = 1,
  QWEN36_STATUS_INVALID_ARGUMENT = 2,
  QWEN36_STATUS_CUDA_ERROR = 3,
  QWEN36_STATUS_CUBLAS_ERROR = 4
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
  size_t position;
  qwen36_device_ptr_t q_bf16;
  qwen36_device_ptr_t k_bf16;
  qwen36_device_ptr_t v_bf16;
  qwen36_device_ptr_t kv_cache_k;
  qwen36_device_ptr_t kv_cache_v;
  qwen36_device_ptr_t output_bf16;
  qwen36_attention_shape_t shape;
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
} qwen36_partial_rope_spec_t;

typedef struct {
  size_t rows;
  size_t intermediate;
  qwen36_device_ptr_t gate_bf16;
  qwen36_device_ptr_t up_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_swiglu_spec_t;

typedef struct {
  size_t vocab_size;
  qwen36_device_ptr_t logits_bf16;
  qwen36_device_ptr_t output_token_u32;
  float temperature;
  size_t top_k;
  float top_p;
  float repetition_penalty;
} qwen36_sampling_spec_t;

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
} qwen36_nvfp4_quantize_spec_t;

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
  size_t heads;
  qwen36_device_ptr_t a_bf16;
  qwen36_device_ptr_t b_bf16;
  qwen36_device_ptr_t a_log_bf16;
  qwen36_device_ptr_t dt_bias_bf16;
  qwen36_device_ptr_t gate_f32;
  qwen36_device_ptr_t beta_f32;
} qwen36_gdn_gate_spec_t;

typedef struct {
  size_t elements;
  qwen36_device_ptr_t gate_bf16;
  qwen36_device_ptr_t input_bf16;
  qwen36_device_ptr_t output_bf16;
} qwen36_sigmoid_gate_spec_t;

int qwen36_cuda_malloc(qwen36_device_allocation_t *out, size_t bytes);
int qwen36_cuda_free(qwen36_device_ptr_t ptr);
int qwen36_cuda_memcpy_h2d(qwen36_device_ptr_t dst, const void *src,
                           size_t bytes);
int qwen36_cuda_memcpy_d2h(void *dst, qwen36_device_ptr_t src, size_t bytes);
int qwen36_cuda_memcpy_d2d(qwen36_device_ptr_t dst, qwen36_device_ptr_t src,
                           size_t bytes);
int qwen36_cuda_memset(qwen36_device_ptr_t dst, int value, size_t bytes);
int qwen36_cuda_synchronize(void);

int qwen36_nvfp4_gemm(const qwen36_nvfp4_gemm_spec_t *spec);
int qwen36_bf16_gemm(const qwen36_bf16_gemm_spec_t *spec);
int qwen36_deltanet_decode(const qwen36_deltanet_decode_spec_t *spec);
int qwen36_attention_decode(const qwen36_attention_decode_spec_t *spec);
int qwen36_turboquant_encode_kv(const qwen36_turboquant_encode_spec_t *spec);
int qwen36_turboquant_attention(const qwen36_turboquant_attention_spec_t *spec);
int qwen36_rmsnorm(const qwen36_rmsnorm_spec_t *spec);
int qwen36_rmsnorm_nvfp4_quantize(
    const qwen36_rmsnorm_nvfp4_quantize_spec_t *spec);
int qwen36_partial_rope(const qwen36_partial_rope_spec_t *spec);
int qwen36_swiglu(const qwen36_swiglu_spec_t *spec);
int qwen36_sample(const qwen36_sampling_spec_t *spec);
int qwen36_embedding_lookup(const qwen36_embedding_lookup_spec_t *spec);
int qwen36_bf16_matvec(const qwen36_bf16_matvec_spec_t *spec);
int qwen36_nvfp4_matvec(const qwen36_nvfp4_matvec_spec_t *spec);
int qwen36_nvfp4_quantize_bf16(const qwen36_nvfp4_quantize_spec_t *spec);
int qwen36_nvfp4_retile_scales(const qwen36_nvfp4_retile_scales_spec_t *spec);
int qwen36_conv1d_update(const qwen36_conv1d_update_spec_t *spec);
int qwen36_gdn_gate(const qwen36_gdn_gate_spec_t *spec);
int qwen36_sigmoid_gate(const qwen36_sigmoid_gate_spec_t *spec);

#ifdef __cplusplus
}
#endif
