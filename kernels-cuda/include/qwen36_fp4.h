#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t ptr;
} qwen36_device_ptr_t;

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
  qwen36_deltanet_shape_t shape;
  float state_decay;
  float update_scale;
} qwen36_deltanet_decode_spec_t;

int qwen36_nvfp4_gemm(const qwen36_nvfp4_gemm_spec_t *spec);
int qwen36_deltanet_decode(const qwen36_deltanet_decode_spec_t *spec);
int qwen36_attention_decode(const qwen36_attention_decode_spec_t *spec);
int qwen36_turboquant_encode_kv(const qwen36_turboquant_encode_spec_t *spec);
int qwen36_turboquant_attention(const qwen36_turboquant_attention_spec_t *spec);

#ifdef __cplusplus
}
#endif
