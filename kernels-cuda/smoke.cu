#include "qwen36_fp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace {

template <typename T> void must_cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    exit(1);
  }
}

void must_status(int status, const char *what) {
  if (status != QWEN36_STATUS_SUCCESS) {
    fprintf(stderr, "%s returned status %d\n", what, status);
    exit(1);
  }
  must_cuda<int>(cudaDeviceSynchronize(), what);
}

template <typename T> qwen36_device_ptr_t dev_alloc(size_t count) {
  T *ptr = nullptr;
  must_cuda<T>(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)),
               "cudaMalloc");
  return qwen36_device_ptr_t{reinterpret_cast<uint64_t>(ptr)};
}

template <typename T> void dev_free(qwen36_device_ptr_t ptr) {
  if (ptr.ptr != 0) {
    must_cuda<T>(cudaFree(reinterpret_cast<void *>(ptr.ptr)), "cudaFree");
  }
}

void copy_bf16(qwen36_device_ptr_t dst, const std::vector<float> &src) {
  std::vector<__nv_bfloat16> tmp(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    tmp[i] = __float2bfloat16(src[i]);
  }
  must_cuda<__nv_bfloat16>(
      cudaMemcpy(reinterpret_cast<void *>(dst.ptr), tmp.data(),
                 tmp.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
      "cudaMemcpy H2D");
}

} // namespace

int main() {
  qwen36_attention_shape_t attn{2, 1, 8, 0};
  const size_t q_values = attn.q_heads * attn.head_dim;
  const size_t kv_values = attn.kv_heads * attn.head_dim;

  qwen36_device_ptr_t q = dev_alloc<__nv_bfloat16>(q_values);
  qwen36_device_ptr_t k = dev_alloc<__nv_bfloat16>(kv_values);
  qwen36_device_ptr_t v = dev_alloc<__nv_bfloat16>(kv_values);
  qwen36_device_ptr_t cache_k = dev_alloc<__nv_bfloat16>(kv_values);
  qwen36_device_ptr_t cache_v = dev_alloc<__nv_bfloat16>(kv_values);
  qwen36_device_ptr_t out = dev_alloc<__nv_bfloat16>(q_values);

  copy_bf16(q, std::vector<float>(q_values, 0.25f));
  copy_bf16(k, std::vector<float>(kv_values, 0.5f));
  copy_bf16(v, std::vector<float>(kv_values, 1.0f));

  qwen36_attention_decode_spec_t attention_spec{};
  attention_spec.position = 0;
  attention_spec.q_bf16 = q;
  attention_spec.k_bf16 = k;
  attention_spec.v_bf16 = v;
  attention_spec.kv_cache_k = cache_k;
  attention_spec.kv_cache_v = cache_v;
  attention_spec.output_bf16 = out;
  attention_spec.shape = attn;
  must_status(qwen36_attention_decode(&attention_spec), "attention");

  qwen36_device_ptr_t kq = dev_alloc<int8_t>(kv_values);
  qwen36_device_ptr_t vq = dev_alloc<int8_t>(kv_values);
  qwen36_device_ptr_t meta = dev_alloc<float>(attn.kv_heads * 2);
  qwen36_turboquant_encode_spec_t encode_spec{};
  encode_spec.position = 0;
  encode_spec.k_bf16 = k;
  encode_spec.v_bf16 = v;
  encode_spec.k_quantized_i8 = kq;
  encode_spec.v_quantized_i8 = vq;
  encode_spec.metadata_f32 = meta;
  encode_spec.shape = attn;
  must_status(qwen36_turboquant_encode_kv(&encode_spec), "turboquant encode");

  qwen36_turboquant_attention_spec_t tq_spec{};
  tq_spec.position = 0;
  tq_spec.q_bf16 = q;
  tq_spec.k_quantized_i8 = kq;
  tq_spec.v_quantized_i8 = vq;
  tq_spec.metadata_f32 = meta;
  tq_spec.output_bf16 = out;
  tq_spec.shape = attn;
  tq_spec.mode = 0;
  must_status(qwen36_turboquant_attention(&tq_spec), "turboquant attention");

  qwen36_deltanet_shape_t delta_shape{1, 2, 8, 8, 4};
  qwen36_device_ptr_t dq = dev_alloc<__nv_bfloat16>(delta_shape.qk_heads *
                                                    delta_shape.key_dim);
  qwen36_device_ptr_t dk = dev_alloc<__nv_bfloat16>(delta_shape.qk_heads *
                                                    delta_shape.key_dim);
  qwen36_device_ptr_t dv = dev_alloc<__nv_bfloat16>(delta_shape.v_heads *
                                                    delta_shape.value_dim);
  qwen36_device_ptr_t state = dev_alloc<__nv_bfloat16>(
      delta_shape.v_heads * delta_shape.value_dim * delta_shape.key_dim);
  qwen36_device_ptr_t dout =
      dev_alloc<__nv_bfloat16>(delta_shape.v_heads * delta_shape.value_dim);
  copy_bf16(dq, std::vector<float>(delta_shape.qk_heads * delta_shape.key_dim,
                                   0.125f));
  copy_bf16(dk, std::vector<float>(delta_shape.qk_heads * delta_shape.key_dim,
                                   0.25f));
  copy_bf16(dv, std::vector<float>(delta_shape.v_heads * delta_shape.value_dim,
                                   0.5f));
  must_cuda<__nv_bfloat16>(
      cudaMemset(reinterpret_cast<void *>(state.ptr), 0,
                 delta_shape.v_heads * delta_shape.value_dim *
                     delta_shape.key_dim * sizeof(__nv_bfloat16)),
      "cudaMemset state");

  qwen36_deltanet_decode_spec_t delta_spec{};
  delta_spec.tokens_in_persistent_loop = 1;
  delta_spec.q_bf16 = dq;
  delta_spec.k_bf16 = dk;
  delta_spec.v_bf16 = dv;
  delta_spec.state_bf16 = state;
  delta_spec.output_bf16 = dout;
  delta_spec.shape = delta_shape;
  delta_spec.state_decay = 1.0f;
  delta_spec.update_scale = 1.0f;
  must_status(qwen36_deltanet_decode(&delta_spec), "deltanet");

  dev_free<__nv_bfloat16>(q);
  dev_free<__nv_bfloat16>(k);
  dev_free<__nv_bfloat16>(v);
  dev_free<__nv_bfloat16>(cache_k);
  dev_free<__nv_bfloat16>(cache_v);
  dev_free<__nv_bfloat16>(out);
  dev_free<int8_t>(kq);
  dev_free<int8_t>(vq);
  dev_free<float>(meta);
  dev_free<__nv_bfloat16>(dq);
  dev_free<__nv_bfloat16>(dk);
  dev_free<__nv_bfloat16>(dv);
  dev_free<__nv_bfloat16>(state);
  dev_free<__nv_bfloat16>(dout);

  printf("qwen36 CUDA smoke test passed\n");
  return 0;
}
