#include "qwen36_fp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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

void expect_status(int status, int expected, const char *what) {
  if (status != expected) {
    fprintf(stderr, "%s expected status %d got %d\n", what, expected, status);
    exit(1);
  }
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

template <typename T>
void copy_raw(qwen36_device_ptr_t dst, const std::vector<T> &src) {
  must_cuda<T>(
      cudaMemcpy(reinterpret_cast<void *>(dst.ptr), src.data(),
                 src.size() * sizeof(T), cudaMemcpyHostToDevice),
      "cudaMemcpy H2D");
}

std::vector<float> read_bf16(qwen36_device_ptr_t src, size_t count) {
  std::vector<__nv_bfloat16> tmp(count);
  must_cuda<__nv_bfloat16>(
      cudaMemcpy(tmp.data(), reinterpret_cast<void *>(src.ptr),
                 tmp.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
      "cudaMemcpy D2H");
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = __bfloat162float(tmp[i]);
  }
  return out;
}

template <typename T> T read_one(qwen36_device_ptr_t src) {
  T value{};
  must_cuda<T>(cudaMemcpy(&value, reinterpret_cast<void *>(src.ptr), sizeof(T),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H");
  return value;
}

template <typename T> std::vector<T> read_raw(qwen36_device_ptr_t src, size_t count) {
  std::vector<T> values(count);
  must_cuda<T>(cudaMemcpy(values.data(), reinterpret_cast<void *>(src.ptr),
                          count * sizeof(T), cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H");
  return values;
}

void expect_close(float actual, float expected, float tolerance,
                  const char *what) {
  if (fabsf(actual - expected) > tolerance) {
    fprintf(stderr, "%s expected %.6f got %.6f\n", what, expected, actual);
    exit(1);
  }
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

  constexpr size_t split_position = 513;
  constexpr size_t split_n_splits = 2;
  qwen36_device_ptr_t split_cache_k =
      dev_alloc<__nv_bfloat16>((split_position + 1) * kv_values);
  qwen36_device_ptr_t split_cache_v =
      dev_alloc<__nv_bfloat16>((split_position + 1) * kv_values);
  qwen36_device_ptr_t split_out = dev_alloc<__nv_bfloat16>(q_values);
  qwen36_device_ptr_t split_partial_acc =
      dev_alloc<float>(attn.q_heads * split_n_splits * attn.head_dim);
  qwen36_device_ptr_t split_partial_max =
      dev_alloc<float>(attn.q_heads * split_n_splits);
  qwen36_device_ptr_t split_partial_denom =
      dev_alloc<float>(attn.q_heads * split_n_splits);
  copy_bf16(split_cache_k,
            std::vector<float>((split_position + 1) * kv_values, 0.5f));
  copy_bf16(split_cache_v,
            std::vector<float>((split_position + 1) * kv_values, 1.0f));
  qwen36_attention_decode_spec_t split_attention_spec = attention_spec;
  split_attention_spec.position = split_position;
  split_attention_spec.kv_cache_k = split_cache_k;
  split_attention_spec.kv_cache_v = split_cache_v;
  split_attention_spec.output_bf16 = split_out;
  split_attention_spec.partial_acc_f32 = split_partial_acc;
  split_attention_spec.partial_max_f32 = split_partial_max;
  split_attention_spec.partial_denom_f32 = split_partial_denom;
  split_attention_spec.decode_n_splits = split_n_splits;
  must_status(qwen36_attention_decode(&split_attention_spec),
              "attention split decode");
  std::vector<float> split_values = read_bf16(split_out, q_values);
  for (size_t i = 0; i < split_values.size(); ++i) {
    expect_close(split_values[i], 1.0f, 0.02f, "attention split decode");
  }

  qwen36_device_ptr_t prefill_start = dev_alloc<int32_t>(1);
  qwen36_device_ptr_t prefill_cache_k = dev_alloc<__nv_bfloat16>(kv_values * 2);
  qwen36_device_ptr_t prefill_cache_v = dev_alloc<__nv_bfloat16>(kv_values * 2);
  qwen36_device_ptr_t prefill_out = dev_alloc<__nv_bfloat16>(q_values);
  copy_raw<int32_t>(prefill_start, {1});
  copy_bf16(prefill_cache_k, std::vector<float>(kv_values * 2, 0.0f));
  copy_bf16(prefill_cache_v, std::vector<float>(kv_values * 2, 0.0f));
  qwen36_attention_prefill_spec_t prefill_spec{};
  prefill_spec.start_position = 0;
  prefill_spec.tokens = 1;
  prefill_spec.q_bf16 = q;
  prefill_spec.k_bf16 = k;
  prefill_spec.v_bf16 = v;
  prefill_spec.kv_cache_k = prefill_cache_k;
  prefill_spec.kv_cache_v = prefill_cache_v;
  prefill_spec.output_bf16 = prefill_out;
  prefill_spec.shape = attn;
  prefill_spec.start_position_device_i32 = prefill_start;
  must_status(qwen36_attention_prefill(&prefill_spec),
              "attention prefill device start");
  std::vector<float> prefill_cache_k_values =
      read_bf16(prefill_cache_k, kv_values * 2);
  std::vector<float> prefill_cache_v_values =
      read_bf16(prefill_cache_v, kv_values * 2);
  expect_close(prefill_cache_k_values[0], 0.0f, 0.02f,
               "attention prefill scalar start ignored");
  expect_close(prefill_cache_k_values[kv_values], 0.5f, 0.02f,
               "attention prefill device start cache k");
  expect_close(prefill_cache_v_values[kv_values], 1.0f, 0.02f,
               "attention prefill device start cache v");

  qwen36_device_ptr_t prefill_split_q =
      dev_alloc<__nv_bfloat16>(q_values * 2);
  qwen36_device_ptr_t prefill_split_k =
      dev_alloc<__nv_bfloat16>(kv_values * 2);
  qwen36_device_ptr_t prefill_split_v =
      dev_alloc<__nv_bfloat16>(kv_values * 2);
  qwen36_device_ptr_t prefill_split_cache_k =
      dev_alloc<__nv_bfloat16>((split_position + 2) * kv_values);
  qwen36_device_ptr_t prefill_split_cache_v =
      dev_alloc<__nv_bfloat16>((split_position + 2) * kv_values);
  qwen36_device_ptr_t prefill_split_out =
      dev_alloc<__nv_bfloat16>(q_values * 2);
  copy_bf16(prefill_split_q, std::vector<float>(q_values * 2, 0.25f));
  copy_bf16(prefill_split_k, std::vector<float>(kv_values * 2, 0.5f));
  copy_bf16(prefill_split_v, std::vector<float>(kv_values * 2, 1.0f));
  copy_bf16(prefill_split_cache_k,
            std::vector<float>((split_position + 2) * kv_values, 0.5f));
  copy_bf16(prefill_split_cache_v,
            std::vector<float>((split_position + 2) * kv_values, 1.0f));
  qwen36_attention_prefill_spec_t prefill_split_spec{};
  prefill_split_spec.start_position = split_position;
  prefill_split_spec.tokens = 2;
  prefill_split_spec.q_bf16 = prefill_split_q;
  prefill_split_spec.k_bf16 = prefill_split_k;
  prefill_split_spec.v_bf16 = prefill_split_v;
  prefill_split_spec.kv_cache_k = prefill_split_cache_k;
  prefill_split_spec.kv_cache_v = prefill_split_cache_v;
  prefill_split_spec.output_bf16 = prefill_split_out;
  prefill_split_spec.shape = attn;
  prefill_split_spec.partial_acc_f32 = split_partial_acc;
  prefill_split_spec.partial_max_f32 = split_partial_max;
  prefill_split_spec.partial_denom_f32 = split_partial_denom;
  prefill_split_spec.prefill_n_splits = split_n_splits;
  must_status(qwen36_attention_prefill(&prefill_split_spec),
              "attention prefill split");
  std::vector<float> prefill_split_values =
      read_bf16(prefill_split_out, q_values * 2);
  for (size_t i = 0; i < prefill_split_values.size(); ++i) {
    expect_close(prefill_split_values[i], 1.0f, 0.02f,
                 "attention prefill split");
  }

  constexpr size_t tile2_tokens = 16;
  qwen36_device_ptr_t prefill_tile2_q =
      dev_alloc<__nv_bfloat16>(q_values * tile2_tokens);
  qwen36_device_ptr_t prefill_tile2_k =
      dev_alloc<__nv_bfloat16>(kv_values * tile2_tokens);
  qwen36_device_ptr_t prefill_tile2_v =
      dev_alloc<__nv_bfloat16>(kv_values * tile2_tokens);
  qwen36_device_ptr_t prefill_tile2_cache_k =
      dev_alloc<__nv_bfloat16>(kv_values * tile2_tokens);
  qwen36_device_ptr_t prefill_tile2_cache_v =
      dev_alloc<__nv_bfloat16>(kv_values * tile2_tokens);
  qwen36_device_ptr_t prefill_tile2_out =
      dev_alloc<__nv_bfloat16>(q_values * tile2_tokens);
  copy_bf16(prefill_tile2_q,
            std::vector<float>(q_values * tile2_tokens, 0.25f));
  copy_bf16(prefill_tile2_k,
            std::vector<float>(kv_values * tile2_tokens, 0.5f));
  copy_bf16(prefill_tile2_v,
            std::vector<float>(kv_values * tile2_tokens, 1.0f));
  copy_bf16(prefill_tile2_cache_k,
            std::vector<float>(kv_values * tile2_tokens, 0.0f));
  copy_bf16(prefill_tile2_cache_v,
            std::vector<float>(kv_values * tile2_tokens, 0.0f));
  qwen36_attention_prefill_spec_t prefill_tile2_spec{};
  prefill_tile2_spec.start_position = 0;
  prefill_tile2_spec.tokens = tile2_tokens;
  prefill_tile2_spec.q_bf16 = prefill_tile2_q;
  prefill_tile2_spec.k_bf16 = prefill_tile2_k;
  prefill_tile2_spec.v_bf16 = prefill_tile2_v;
  prefill_tile2_spec.kv_cache_k = prefill_tile2_cache_k;
  prefill_tile2_spec.kv_cache_v = prefill_tile2_cache_v;
  prefill_tile2_spec.output_bf16 = prefill_tile2_out;
  prefill_tile2_spec.shape = attn;
  setenv("QWEN36_PREFILL_GQA_TILE2", "1", 1);
  must_status(qwen36_attention_prefill(&prefill_tile2_spec),
              "attention prefill gqa tile2");
  unsetenv("QWEN36_PREFILL_GQA_TILE2");
  std::vector<float> prefill_tile2_values =
      read_bf16(prefill_tile2_out, q_values * tile2_tokens);
  for (size_t i = 0; i < prefill_tile2_values.size(); ++i) {
    expect_close(prefill_tile2_values[i], 1.0f, 0.02f,
                 "attention prefill gqa tile2");
  }

  qwen36_device_ptr_t prefill_tile4_out =
      dev_alloc<__nv_bfloat16>(q_values * tile2_tokens);
  prefill_tile2_spec.output_bf16 = prefill_tile4_out;
  setenv("QWEN36_PREFILL_GQA_TILE_TOKENS", "4", 1);
  must_status(qwen36_attention_prefill(&prefill_tile2_spec),
              "attention prefill gqa tile4");
  unsetenv("QWEN36_PREFILL_GQA_TILE_TOKENS");
  std::vector<float> prefill_tile4_values =
      read_bf16(prefill_tile4_out, q_values * tile2_tokens);
  for (size_t i = 0; i < prefill_tile4_values.size(); ++i) {
    expect_close(prefill_tile4_values[i], 1.0f, 0.02f,
                 "attention prefill gqa tile4");
  }

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

  qwen36_deltanet_shape_t delta_shape{2, 4, 2, 1, 4};
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
  copy_bf16(dq, {1.0f, 0.0f, 2.0f, 0.0f});
  copy_bf16(dk, {1.0f, 0.0f, 1.0f, 0.0f});
  copy_bf16(dv, std::vector<float>(delta_shape.v_heads * delta_shape.value_dim,
                                   1.0f));
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
  std::vector<float> delta_values = read_bf16(dout, 4);
  expect_close(delta_values[0], 1.0f, 0.02f, "deltanet head repeat[0]");
  expect_close(delta_values[1], 1.0f, 0.02f, "deltanet head repeat[1]");
  expect_close(delta_values[2], 2.0f, 0.02f, "deltanet head repeat[2]");
  expect_close(delta_values[3], 2.0f, 0.02f, "deltanet head repeat[3]");

  qwen36_device_ptr_t exact_state = dev_alloc<__nv_bfloat16>(
      delta_shape.v_heads * delta_shape.value_dim * delta_shape.key_dim);
  qwen36_device_ptr_t exact_out =
      dev_alloc<__nv_bfloat16>(delta_shape.v_heads * delta_shape.value_dim);
  qwen36_device_ptr_t exact_gate = dev_alloc<float>(delta_shape.v_heads);
  qwen36_device_ptr_t exact_beta = dev_alloc<float>(delta_shape.v_heads);
  must_cuda<__nv_bfloat16>(
      cudaMemset(reinterpret_cast<void *>(exact_state.ptr), 0,
                 delta_shape.v_heads * delta_shape.value_dim *
                     delta_shape.key_dim * sizeof(__nv_bfloat16)),
      "cudaMemset exact state");
  copy_raw<float>(exact_gate, std::vector<float>(delta_shape.v_heads, 0.0f));
  copy_raw<float>(exact_beta, std::vector<float>(delta_shape.v_heads, 0.5f));
  qwen36_deltanet_decode_spec_t exact_delta_spec = delta_spec;
  exact_delta_spec.state_bf16 = exact_state;
  exact_delta_spec.output_bf16 = exact_out;
  exact_delta_spec.gate_f32 = exact_gate;
  exact_delta_spec.beta_f32 = exact_beta;
  exact_delta_spec.qk_l2norm = 0;
  must_status(qwen36_deltanet_decode(&exact_delta_spec), "deltanet exact");
  std::vector<float> exact_delta_values = read_bf16(exact_out, 4);
  const float exact_scale = 1.0f / sqrtf(static_cast<float>(delta_shape.key_dim));
  expect_close(exact_delta_values[0], 0.5f * exact_scale, 0.02f,
               "deltanet exact[0]");
  expect_close(exact_delta_values[1], 0.5f * exact_scale, 0.02f,
               "deltanet exact[1]");
  expect_close(exact_delta_values[2], 1.0f * exact_scale, 0.02f,
               "deltanet exact[2]");
  expect_close(exact_delta_values[3], 1.0f * exact_scale, 0.02f,
               "deltanet exact[3]");
  qwen36_deltanet_decode_spec_t invalid_delta_spec = delta_spec;
  invalid_delta_spec.shape.key_dim = 257;
  expect_status(qwen36_deltanet_decode(&invalid_delta_spec),
                QWEN36_STATUS_INVALID_ARGUMENT, "deltanet key_dim guard");

  qwen36_device_ptr_t norm_in = dev_alloc<__nv_bfloat16>(4);
  qwen36_device_ptr_t norm_weight = dev_alloc<__nv_bfloat16>(4);
  qwen36_device_ptr_t norm_residual = dev_alloc<__nv_bfloat16>(4);
  qwen36_device_ptr_t norm_out = dev_alloc<__nv_bfloat16>(4);
  copy_bf16(norm_in, {1.0f, 2.0f, 3.0f, 4.0f});
  copy_bf16(norm_weight, {0.0f, 0.0f, 0.0f, 0.0f});
  copy_bf16(norm_residual, {1.0f, 0.0f, -1.0f, 0.0f});
  qwen36_rmsnorm_spec_t norm_spec{};
  norm_spec.rows = 1;
  norm_spec.hidden = 4;
  norm_spec.eps = 1.0e-6f;
  norm_spec.input_bf16 = norm_in;
  norm_spec.weight_bf16 = norm_weight;
  norm_spec.residual_bf16 = norm_residual;
  norm_spec.output_bf16 = norm_out;
  must_status(qwen36_rmsnorm(&norm_spec), "rmsnorm");
  std::vector<float> norm_values = read_bf16(norm_out, 4);
  const float norm_scale = 1.0f / sqrtf(7.0f + 1.0e-6f);
  expect_close(norm_values[0], 2.0f * norm_scale, 0.02f, "rmsnorm[0]");
  expect_close(norm_values[3], 4.0f * norm_scale, 0.02f, "rmsnorm[3]");

  qwen36_device_ptr_t fused_norm_in = dev_alloc<__nv_bfloat16>(16);
  qwen36_device_ptr_t fused_norm_weight = dev_alloc<__nv_bfloat16>(16);
  qwen36_device_ptr_t fused_norm_out = dev_alloc<__nv_bfloat16>(16);
  qwen36_device_ptr_t fused_norm_fp4 = dev_alloc<uint8_t>(8);
  qwen36_device_ptr_t fused_norm_scale = dev_alloc<uint8_t>(512);
  qwen36_device_ptr_t fused_norm_global = dev_alloc<float>(1);
  copy_bf16(fused_norm_in, std::vector<float>(16, 1.0f));
  copy_bf16(fused_norm_weight, std::vector<float>(16, 0.0f));
  qwen36_rmsnorm_nvfp4_quantize_spec_t fused_norm_spec{};
  fused_norm_spec.hidden = 16;
  fused_norm_spec.eps = 1.0e-6f;
  fused_norm_spec.input_bf16 = fused_norm_in;
  fused_norm_spec.weight_bf16 = fused_norm_weight;
  fused_norm_spec.output_bf16 = fused_norm_out;
  fused_norm_spec.output_fp4 = fused_norm_fp4;
  fused_norm_spec.output_scale_e4m3 = fused_norm_scale;
  fused_norm_spec.output_tensor_scale_f32 = fused_norm_global;
  must_status(qwen36_rmsnorm_nvfp4_quantize(&fused_norm_spec),
              "rmsnorm nvfp4 quantize");
  std::vector<float> fused_norm_values = read_bf16(fused_norm_out, 16);
  std::vector<uint8_t> fused_norm_packed = read_raw<uint8_t>(fused_norm_fp4, 8);
  expect_close(fused_norm_values[0], 1.0f, 0.02f,
               "rmsnorm nvfp4 quantize bf16");
  if (fused_norm_packed[0] != 0x77 || fused_norm_packed[7] != 0x77) {
    fprintf(stderr, "rmsnorm nvfp4 quantize expected packed 0x77 got 0x%02x 0x%02x\n",
            fused_norm_packed[0], fused_norm_packed[7]);
    exit(1);
  }

  qwen36_device_ptr_t rope_pos = dev_alloc<int32_t>(1);
  qwen36_device_ptr_t rope_q = dev_alloc<__nv_bfloat16>(6);
  qwen36_device_ptr_t rope_k = dev_alloc<__nv_bfloat16>(6);
  copy_raw<int32_t>(rope_pos, {1});
  copy_bf16(rope_q, {1.0f, 2.0f, 0.0f, 0.0f, 9.0f, 9.0f});
  copy_bf16(rope_k, {0.0f, 0.0f, 1.0f, 2.0f, 8.0f, 8.0f});
  qwen36_partial_rope_spec_t rope_spec{};
  rope_spec.tokens = 1;
  rope_spec.q_heads = 1;
  rope_spec.kv_heads = 1;
  rope_spec.head_dim = 6;
  rope_spec.rope_dims = 4;
  rope_spec.base_theta = 10000.0;
  rope_spec.positions_i32 = rope_pos;
  rope_spec.q_bf16 = rope_q;
  rope_spec.k_bf16 = rope_k;
  must_status(qwen36_partial_rope(&rope_spec), "partial rope");
  std::vector<float> rope_q_values = read_bf16(rope_q, 6);
  std::vector<float> rope_k_values = read_bf16(rope_k, 6);
  expect_close(rope_q_values[0], cosf(1.0f), 0.02f, "rope q[0]");
  expect_close(rope_q_values[2], sinf(1.0f), 0.02f, "rope q[2]");
  expect_close(rope_q_values[1], 2.0f * cosf(0.01f), 0.02f, "rope q[1]");
  expect_close(rope_q_values[3], 2.0f * sinf(0.01f), 0.02f, "rope q[3]");
  expect_close(rope_q_values[4], 9.0f, 0.02f, "rope q tail");
  expect_close(rope_k_values[0], -sinf(1.0f), 0.02f, "rope k[0]");
  expect_close(rope_k_values[2], cosf(1.0f), 0.02f, "rope k[2]");

  qwen36_device_ptr_t swiglu_gate = dev_alloc<__nv_bfloat16>(4);
  qwen36_device_ptr_t swiglu_up = dev_alloc<__nv_bfloat16>(4);
  qwen36_device_ptr_t swiglu_out = dev_alloc<__nv_bfloat16>(4);
  copy_bf16(swiglu_gate, {0.0f, 1.0f, -1.0f, 2.0f});
  copy_bf16(swiglu_up, {1.0f, 2.0f, 3.0f, 4.0f});
  qwen36_swiglu_spec_t swiglu_spec{};
  swiglu_spec.rows = 1;
  swiglu_spec.intermediate = 4;
  swiglu_spec.gate_bf16 = swiglu_gate;
  swiglu_spec.up_bf16 = swiglu_up;
  swiglu_spec.output_bf16 = swiglu_out;
  must_status(qwen36_swiglu(&swiglu_spec), "swiglu");
  std::vector<float> swiglu_values = read_bf16(swiglu_out, 4);
  expect_close(swiglu_values[0], 0.0f, 0.02f, "swiglu[0]");
  expect_close(swiglu_values[1], 2.0f / (1.0f + expf(-1.0f)), 0.02f,
               "swiglu[1]");

  qwen36_device_ptr_t logits = dev_alloc<__nv_bfloat16>(4);
  qwen36_device_ptr_t sampled = dev_alloc<uint32_t>(1);
  qwen36_device_ptr_t sampled_mirror = dev_alloc<uint32_t>(1);
  qwen36_device_ptr_t logits_rows = dev_alloc<__nv_bfloat16>(8);
  qwen36_device_ptr_t sampled_rows = dev_alloc<uint32_t>(2);
  qwen36_device_ptr_t sampled_rows_mirror = dev_alloc<uint32_t>(1);
  copy_bf16(logits, {0.5f, 4.0f, 3.0f, 4.0f});
  qwen36_sampling_spec_t sampling_spec{};
  sampling_spec.vocab_size = 4;
  sampling_spec.logits_bf16 = logits;
  sampling_spec.output_token_u32 = sampled;
  sampling_spec.mirror_output_token_u32 = sampled_mirror;
  sampling_spec.temperature = 1.0f;
  must_status(qwen36_sample(&sampling_spec), "sample");
  const uint32_t token = read_one<uint32_t>(sampled);
  const uint32_t mirror_token = read_one<uint32_t>(sampled_mirror);
  if (token != 1 || mirror_token != 1) {
    fprintf(stderr, "sample expected token 1 got %u mirror %u\n", token,
            mirror_token);
    exit(1);
  }
  copy_bf16(logits_rows, {0.5f, 4.0f, 3.0f, 4.0f, 1.0f, 2.0f, 5.0f, 4.0f});
  qwen36_sampling_rows_spec_t sampling_rows_spec{};
  sampling_rows_spec.rows = 2;
  sampling_rows_spec.vocab_size = 4;
  sampling_rows_spec.logits_bf16 = logits_rows;
  sampling_rows_spec.output_token_u32 = sampled_rows;
  sampling_rows_spec.mirror_last_output_token_u32 = sampled_rows_mirror;
  sampling_rows_spec.temperature = 1.0f;
  must_status(qwen36_sample_rows(&sampling_rows_spec), "sample rows");
  const std::vector<uint32_t> row_tokens = read_raw<uint32_t>(sampled_rows, 2);
  const uint32_t row_mirror_token = read_one<uint32_t>(sampled_rows_mirror);
  if (row_tokens[0] != 1 || row_tokens[1] != 2 || row_mirror_token != 2) {
    fprintf(stderr, "sample rows expected [1, 2] mirror 2 got [%u, %u] mirror %u\n",
            row_tokens[0], row_tokens[1], row_mirror_token);
    exit(1);
  }

  qwen36_device_ptr_t token_ids = dev_alloc<uint32_t>(1);
  qwen36_device_ptr_t embedding = dev_alloc<__nv_bfloat16>(8);
  qwen36_device_ptr_t embedding_out = dev_alloc<__nv_bfloat16>(4);
  copy_raw<uint32_t>(token_ids, {1});
  copy_bf16(embedding, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  qwen36_embedding_lookup_spec_t embedding_spec{};
  embedding_spec.tokens = 1;
  embedding_spec.hidden = 4;
  embedding_spec.vocab_size = 2;
  embedding_spec.token_ids_u32 = token_ids;
  embedding_spec.embedding_bf16 = embedding;
  embedding_spec.output_bf16 = embedding_out;
  must_status(qwen36_embedding_lookup(&embedding_spec), "embedding lookup");
  std::vector<float> embedding_values = read_bf16(embedding_out, 4);
  expect_close(embedding_values[0], 5.0f, 0.02f, "embedding[0]");
  expect_close(embedding_values[3], 8.0f, 0.02f, "embedding[3]");

  qwen36_device_ptr_t matvec_input = dev_alloc<__nv_bfloat16>(4);
  qwen36_device_ptr_t matvec_weight = dev_alloc<__nv_bfloat16>(8);
  qwen36_device_ptr_t matvec_out = dev_alloc<__nv_bfloat16>(2);
  copy_bf16(matvec_input, {1.0f, 2.0f, 3.0f, 4.0f});
  copy_bf16(matvec_weight, {1.0f, 0.0f, 0.0f, 1.0f,
                            0.5f, 0.5f, 0.5f, 0.5f});
  qwen36_bf16_matvec_spec_t matvec_spec{};
  matvec_spec.out_features = 2;
  matvec_spec.in_features = 4;
  matvec_spec.input_bf16 = matvec_input;
  matvec_spec.weight_bf16 = matvec_weight;
  matvec_spec.output_bf16 = matvec_out;
  must_status(qwen36_bf16_matvec(&matvec_spec), "bf16 matvec");
  std::vector<float> matvec_values = read_bf16(matvec_out, 2);
  expect_close(matvec_values[0], 5.0f, 0.02f, "bf16 matvec[0]");
  expect_close(matvec_values[1], 5.0f, 0.02f, "bf16 matvec[1]");

  qwen36_device_ptr_t bf16_gemm_input = dev_alloc<__nv_bfloat16>(128);
  qwen36_device_ptr_t bf16_gemm_weight = dev_alloc<__nv_bfloat16>(128 * 128);
  qwen36_device_ptr_t bf16_gemm_out = dev_alloc<__nv_bfloat16>(128);
  qwen36_device_ptr_t bf16_gemm_workspace = dev_alloc<uint8_t>(4 * 1024 * 1024);
  std::vector<float> bf16_gemm_input_values(128);
  for (size_t idx = 0; idx < bf16_gemm_input_values.size(); ++idx) {
    bf16_gemm_input_values[idx] = static_cast<float>(idx + 1);
  }
  std::vector<float> bf16_gemm_weight_values(128 * 128, 0.0f);
  bf16_gemm_weight_values[0] = 1.0f;
  bf16_gemm_weight_values[128 + 1] = 2.0f;
  bf16_gemm_weight_values[(127 * 128) + 127] = 3.0f;
  copy_bf16(bf16_gemm_input, bf16_gemm_input_values);
  copy_bf16(bf16_gemm_weight, bf16_gemm_weight_values);
  qwen36_bf16_gemm_spec_t bf16_gemm_spec{};
  bf16_gemm_spec.m = 128;
  bf16_gemm_spec.n = 1;
  bf16_gemm_spec.k = 128;
  bf16_gemm_spec.a_bf16 = bf16_gemm_weight;
  bf16_gemm_spec.b_bf16 = bf16_gemm_input;
  bf16_gemm_spec.c_bf16 = bf16_gemm_out;
  bf16_gemm_spec.workspace = bf16_gemm_workspace;
  bf16_gemm_spec.workspace_bytes = 4 * 1024 * 1024;
  must_status(qwen36_bf16_gemm(&bf16_gemm_spec), "bf16 gemm");
  std::vector<float> bf16_gemm_values = read_bf16(bf16_gemm_out, 128);
  expect_close(bf16_gemm_values[0], 1.0f, 0.5f, "bf16 gemm[0]");
  expect_close(bf16_gemm_values[1], 4.0f, 0.5f, "bf16 gemm[1]");
  expect_close(bf16_gemm_values[127], 384.0f, 0.5f, "bf16 gemm[last]");

  qwen36_device_ptr_t fp4_weight = dev_alloc<uint8_t>(2);
  qwen36_device_ptr_t fp4_scale_raw = dev_alloc<uint8_t>(1);
  qwen36_device_ptr_t fp4_scale = dev_alloc<uint8_t>(512);
  qwen36_device_ptr_t fp4_global = dev_alloc<float>(1);
  qwen36_device_ptr_t fp4_out = dev_alloc<__nv_bfloat16>(1);
  copy_raw<uint8_t>(fp4_weight, {0x42, 0xA1});
  copy_raw<uint8_t>(fp4_scale_raw, {0x38});
  copy_raw<float>(fp4_global, {1.0f});
  qwen36_nvfp4_retile_scales_spec_t retile_spec{};
  retile_spec.rows = 1;
  retile_spec.inner_groups = 1;
  retile_spec.input_row_major_u8 = fp4_scale_raw;
  retile_spec.output_tiled_u8 = fp4_scale;
  must_status(qwen36_nvfp4_retile_scales(&retile_spec),
              "nvfp4 retile scales");
  qwen36_nvfp4_matvec_spec_t fp4_spec{};
  fp4_spec.out_features = 1;
  fp4_spec.in_features = 4;
  fp4_spec.input_bf16 = matvec_input;
  fp4_spec.weight_u8 = fp4_weight;
  fp4_spec.block_scale_e4m3 = fp4_scale;
  fp4_spec.tensor_scale_f32 = fp4_global;
  fp4_spec.output_bf16 = fp4_out;
  must_status(qwen36_nvfp4_matvec(&fp4_spec), "nvfp4 matvec");
  expect_close(read_bf16(fp4_out, 1)[0], 2.5f, 0.02f, "nvfp4 matvec");

  const size_t gemm_m = 128;
  const size_t gemm_k = 128;
  qwen36_device_ptr_t quant_input = dev_alloc<__nv_bfloat16>(gemm_k);
  qwen36_device_ptr_t quant_fp4 = dev_alloc<uint8_t>(gemm_k / 2);
  qwen36_device_ptr_t quant_scale = dev_alloc<uint8_t>((gemm_k / 64) * 512);
  qwen36_device_ptr_t quant_global = dev_alloc<float>(1);
  copy_bf16(quant_input, std::vector<float>(gemm_k, 1.0f));
  qwen36_nvfp4_quantize_spec_t quant_spec{};
  quant_spec.values = gemm_k;
  quant_spec.input_bf16 = quant_input;
  quant_spec.output_fp4 = quant_fp4;
  quant_spec.output_scale_e4m3 = quant_scale;
  quant_spec.output_tensor_scale_f32 = quant_global;
  must_status(qwen36_nvfp4_quantize_bf16(&quant_spec),
              "nvfp4 quantize");
  expect_close(read_one<float>(quant_global), 1.0f, 0.001f,
               "nvfp4 quantize global");

  qwen36_device_ptr_t gemm_weight = dev_alloc<uint8_t>(gemm_m * gemm_k / 2);
  qwen36_device_ptr_t gemm_scale = dev_alloc<uint8_t>(gemm_m * gemm_k / 16);
  qwen36_device_ptr_t gemm_out = dev_alloc<__nv_bfloat16>(gemm_m);
  qwen36_device_ptr_t gemm_workspace = dev_alloc<uint8_t>(4 * 1024 * 1024);
  copy_raw<uint8_t>(gemm_weight, std::vector<uint8_t>(gemm_m * gemm_k / 2, 0x22));
  copy_raw<uint8_t>(gemm_scale, std::vector<uint8_t>(gemm_m * gemm_k / 16, 0x38));
  qwen36_nvfp4_gemm_spec_t gemm_spec{};
  gemm_spec.m = gemm_m;
  gemm_spec.n = 1;
  gemm_spec.k = gemm_k;
  gemm_spec.a_fp4 = gemm_weight;
  gemm_spec.a_scale = gemm_scale;
  gemm_spec.b_fp4 = quant_fp4;
  gemm_spec.b_scale = quant_scale;
  gemm_spec.b_scale_2 = quant_global;
  gemm_spec.c_bf16 = gemm_out;
  gemm_spec.workspace = gemm_workspace;
  gemm_spec.workspace_bytes = 4 * 1024 * 1024;
  gemm_spec.alpha = 1.0f;
  must_status(qwen36_nvfp4_gemm(&gemm_spec), "nvfp4 gemm");
  std::vector<float> gemm_values = read_bf16(gemm_out, gemm_m);
  expect_close(gemm_values[0], 132.0f, 4.0f, "nvfp4 gemm[0]");
  expect_close(gemm_values[gemm_m - 1], 132.0f, 4.0f, "nvfp4 gemm[last]");

  // Direction B decode_gemv is currently soft-disabled (B3.1 MMA
  // kernel passes uniform-data smoke but fails real-model parity —
  // see docs/superpowers/notes/2026-05-04-direction-b-b3-1-parity.md).
  // The dispatcher must therefore see NOT_IMPLEMENTED for every
  // shape so it falls back to cuBLASLt cleanly.
  qwen36_nvfp4_gemm_spec_t gemv_b1_spec = gemm_spec;
  gemv_b1_spec.m = gemm_m + 1;  // deliberately not multiple of 16
  gemv_b1_spec.n = 1;
  int gemv_b1_code = qwen36_decode_nvfp4_gemv(&gemv_b1_spec);
  if (gemv_b1_code != QWEN36_STATUS_NOT_IMPLEMENTED) {
    fprintf(stderr,
            "decode_gemv B1 expected NOT_IMPLEMENTED (5) for unsupported "
            "shape, got %d\n",
            gemv_b1_code);
    return 1;
  }

  qwen36_device_ptr_t conv_input = dev_alloc<__nv_bfloat16>(1);
  qwen36_device_ptr_t conv_history = dev_alloc<__nv_bfloat16>(3);
  qwen36_device_ptr_t conv_weight = dev_alloc<__nv_bfloat16>(4);
  qwen36_device_ptr_t conv_out = dev_alloc<__nv_bfloat16>(1);
  copy_bf16(conv_input, {2.0f});
  copy_bf16(conv_history, {0.0f, 0.0f, 0.0f});
  copy_bf16(conv_weight, {1.0f, 1.0f, 1.0f, 1.0f});
  qwen36_conv1d_update_spec_t conv_spec{};
  conv_spec.channels = 1;
  conv_spec.kernel_size = 4;
  conv_spec.input_bf16 = conv_input;
  conv_spec.conv_history_bf16 = conv_history;
  conv_spec.weight_bf16 = conv_weight;
  conv_spec.output_bf16 = conv_out;
  must_status(qwen36_conv1d_update(&conv_spec), "conv1d update");
  expect_close(read_bf16(conv_out, 1)[0], 2.0f / (1.0f + expf(-2.0f)), 0.02f,
               "conv1d update");

  qwen36_device_ptr_t gate_a = dev_alloc<__nv_bfloat16>(1);
  qwen36_device_ptr_t gate_b = dev_alloc<__nv_bfloat16>(1);
  qwen36_device_ptr_t gate_a_log = dev_alloc<__nv_bfloat16>(1);
  qwen36_device_ptr_t gate_dt = dev_alloc<__nv_bfloat16>(1);
  qwen36_device_ptr_t gate_out = dev_alloc<float>(1);
  qwen36_device_ptr_t beta_out = dev_alloc<float>(1);
  copy_bf16(gate_a, {0.0f});
  copy_bf16(gate_b, {0.0f});
  copy_bf16(gate_a_log, {0.0f});
  copy_bf16(gate_dt, {0.0f});
  qwen36_gdn_gate_spec_t gate_spec{};
  gate_spec.rows = 1;
  gate_spec.heads = 1;
  gate_spec.a_bf16 = gate_a;
  gate_spec.b_bf16 = gate_b;
  gate_spec.a_log_bf16 = gate_a_log;
  gate_spec.dt_bias_bf16 = gate_dt;
  gate_spec.gate_f32 = gate_out;
  gate_spec.beta_f32 = beta_out;
  must_status(qwen36_gdn_gate(&gate_spec), "gdn gate");
  expect_close(read_raw<float>(gate_out, 1)[0], -logf(2.0f), 0.02f,
               "gdn gate");
  expect_close(read_raw<float>(beta_out, 1)[0], 0.5f, 0.02f, "gdn beta");

  qwen36_device_ptr_t sigmoid_gate = dev_alloc<__nv_bfloat16>(1);
  qwen36_device_ptr_t sigmoid_input = dev_alloc<__nv_bfloat16>(1);
  qwen36_device_ptr_t sigmoid_out = dev_alloc<__nv_bfloat16>(1);
  copy_bf16(sigmoid_gate, {0.0f});
  copy_bf16(sigmoid_input, {2.0f});
  qwen36_sigmoid_gate_spec_t sigmoid_spec{};
  sigmoid_spec.elements = 1;
  sigmoid_spec.gate_bf16 = sigmoid_gate;
  sigmoid_spec.input_bf16 = sigmoid_input;
  sigmoid_spec.output_bf16 = sigmoid_out;
  must_status(qwen36_sigmoid_gate(&sigmoid_spec), "sigmoid gate");
  expect_close(read_bf16(sigmoid_out, 1)[0], 1.0f, 0.02f, "sigmoid gate");

  // Top-K argmax smoke: K=4 over a 1024-vocab BF16 logits array with
  // planted top-4 at known indices.
  {
    constexpr size_t V = 1024;
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> h_logits_f(V);
    for (size_t i = 0; i < V; ++i) {
      h_logits_f[i] = dist(rng);
    }
    const uint32_t expect[4] = {17, 200, 999, 42};
    const float vals[4] = {10.0f, 9.5f, 9.0f, 8.5f};
    for (int i = 0; i < 4; ++i) {
      h_logits_f[expect[i]] = vals[i];
    }

    qwen36_device_ptr_t topk_logits = dev_alloc<__nv_bfloat16>(V);
    qwen36_device_ptr_t topk_out = dev_alloc<uint32_t>(4);
    copy_bf16(topk_logits, h_logits_f);

    qwen36_topk_argmax_spec_t topk_spec{};
    topk_spec.vocab_size = V;
    topk_spec.k = 4;
    topk_spec.logits_bf16 = topk_logits;
    topk_spec.output_token_u32 = topk_out;
    must_status(qwen36_topk_argmax(&topk_spec), "topk argmax");

    std::vector<uint32_t> got = read_raw<uint32_t>(topk_out, 4);
    for (int i = 0; i < 4; ++i) {
      if (got[i] != expect[i]) {
        fprintf(stderr, "topk smoke mismatch at %d: got %u want %u\n",
                i, got[i], expect[i]);
        exit(1);
      }
    }
    dev_free<__nv_bfloat16>(topk_logits);
    dev_free<uint32_t>(topk_out);
    printf("topk smoke OK\n");
  }

  // Tree-mask attention prefill smoke: chunk=4 rows, q_heads=2, kv_heads=1,
  // head_dim=8. All Q and K are zero (so Q·K = 0 → softmax is uniform over
  // visible positions). V values are 1, 10, 100, 1000 at chunk rows 0..3.
  // The bitmap restricts row 3 to attend to rows {0, 3} only (skipping 1, 2),
  // so its output must be (V[0] + V[3]) / 2 = 500.5 instead of the causal
  // average 277.75.
  {
    constexpr size_t CHUNK = 4;
    constexpr size_t Q_HEADS = 2;
    constexpr size_t KV_HEADS = 1;
    constexpr size_t HEAD_DIM = 8;
    qwen36_attention_shape_t tree_attn{};
    tree_attn.q_heads = Q_HEADS;
    tree_attn.kv_heads = KV_HEADS;
    tree_attn.head_dim = HEAD_DIM;
    tree_attn.rope_dims = 0;

    const size_t q_total = CHUNK * Q_HEADS * HEAD_DIM;
    const size_t kv_per_row = KV_HEADS * HEAD_DIM;
    const size_t kv_total = CHUNK * kv_per_row;

    qwen36_device_ptr_t tree_q = dev_alloc<__nv_bfloat16>(q_total);
    qwen36_device_ptr_t tree_k = dev_alloc<__nv_bfloat16>(kv_total);
    qwen36_device_ptr_t tree_v = dev_alloc<__nv_bfloat16>(kv_total);
    qwen36_device_ptr_t tree_cache_k = dev_alloc<__nv_bfloat16>(kv_total);
    qwen36_device_ptr_t tree_cache_v = dev_alloc<__nv_bfloat16>(kv_total);
    qwen36_device_ptr_t tree_out = dev_alloc<__nv_bfloat16>(q_total);

    // Q and K are zero (Q·K = 0 for all positions; softmax becomes uniform
    // over visible positions).
    copy_bf16(tree_q, std::vector<float>(q_total, 0.0f));
    copy_bf16(tree_k, std::vector<float>(kv_total, 0.0f));
    copy_bf16(tree_cache_k, std::vector<float>(kv_total, 0.0f));

    // V[row r] = (1, 10, 100, 1000)[r] across all KV_HEADS * HEAD_DIM dims.
    std::vector<float> v_host(kv_total, 0.0f);
    const float v_per_row[CHUNK] = {1.0f, 10.0f, 100.0f, 1000.0f};
    for (size_t r = 0; r < CHUNK; ++r) {
      for (size_t d = 0; d < kv_per_row; ++d) {
        v_host[r * kv_per_row + d] = v_per_row[r];
      }
    }
    copy_bf16(tree_v, v_host);
    copy_bf16(tree_cache_v, v_host);  // cache also pre-populated

    // Bitmap: causal for rows 0..2, row 3 sees only {0, 3}.
    std::vector<uint64_t> bitmap_host = {
        0b0001ULL, 0b0011ULL, 0b0111ULL, 0b1001ULL,
    };
    qwen36_device_ptr_t tree_bitmap = dev_alloc<uint64_t>(CHUNK);
    copy_raw<uint64_t>(tree_bitmap, bitmap_host);

    qwen36_attention_prefill_spec_t tree_spec{};
    tree_spec.start_position = 0;
    tree_spec.tokens = CHUNK;
    tree_spec.q_bf16 = tree_q;
    tree_spec.k_bf16 = tree_k;
    tree_spec.v_bf16 = tree_v;
    tree_spec.kv_cache_k = tree_cache_k;
    tree_spec.kv_cache_v = tree_cache_v;
    tree_spec.output_bf16 = tree_out;
    tree_spec.shape = tree_attn;
    tree_spec.tree_ancestor_bitmap_u64 = tree_bitmap;
    tree_spec.verify_chunk_rows = CHUNK;

    must_status(qwen36_attention_prefill(&tree_spec),
                "attention prefill tree-mask");

    std::vector<float> tree_out_values = read_bf16(tree_out, q_total);
    // Expected per row (uniform softmax over visible positions since Q·K = 0):
    //   Row 0: V[0] = 1.0
    //   Row 1: (V[0] + V[1]) / 2 = 5.5
    //   Row 2: (V[0] + V[1] + V[2]) / 3 = 37.0
    //   Row 3 (tree, sees {0,3}): (V[0] + V[3]) / 2 = 500.5
    //     (causal would give (1+10+100+1000)/4 = 277.75 — distinguishable)
    const float expected[CHUNK] = {1.0f, 5.5f, 37.0f, 500.5f};
    for (size_t r = 0; r < CHUNK; ++r) {
      for (size_t qh = 0; qh < Q_HEADS; ++qh) {
        for (size_t d = 0; d < HEAD_DIM; ++d) {
          const float got =
              tree_out_values[(r * Q_HEADS + qh) * HEAD_DIM + d];
          // BF16 has ~7 mantissa bits (~0.5% relative error). Use a wider
          // tolerance for the large row-3 value (500.5 → tol ~5.0).
          const float tol = expected[r] > 100.0f ? 5.0f : 0.5f;
          expect_close(got, expected[r], tol, "attention prefill tree-mask");
        }
      }
    }

    dev_free<__nv_bfloat16>(tree_q);
    dev_free<__nv_bfloat16>(tree_k);
    dev_free<__nv_bfloat16>(tree_v);
    dev_free<__nv_bfloat16>(tree_cache_k);
    dev_free<__nv_bfloat16>(tree_cache_v);
    dev_free<__nv_bfloat16>(tree_out);
    dev_free<uint64_t>(tree_bitmap);
    std::printf("attention prefill tree-mask smoke OK\n");
  }

  dev_free<__nv_bfloat16>(q);
  dev_free<__nv_bfloat16>(k);
  dev_free<__nv_bfloat16>(v);
  dev_free<__nv_bfloat16>(cache_k);
  dev_free<__nv_bfloat16>(cache_v);
  dev_free<__nv_bfloat16>(out);
  dev_free<__nv_bfloat16>(split_cache_k);
  dev_free<__nv_bfloat16>(split_cache_v);
  dev_free<__nv_bfloat16>(split_out);
  dev_free<float>(split_partial_acc);
  dev_free<float>(split_partial_max);
  dev_free<float>(split_partial_denom);
  dev_free<int32_t>(prefill_start);
  dev_free<__nv_bfloat16>(prefill_cache_k);
  dev_free<__nv_bfloat16>(prefill_cache_v);
  dev_free<__nv_bfloat16>(prefill_out);
  dev_free<__nv_bfloat16>(prefill_split_q);
  dev_free<__nv_bfloat16>(prefill_split_k);
  dev_free<__nv_bfloat16>(prefill_split_v);
  dev_free<__nv_bfloat16>(prefill_split_cache_k);
  dev_free<__nv_bfloat16>(prefill_split_cache_v);
  dev_free<__nv_bfloat16>(prefill_split_out);
  dev_free<int8_t>(kq);
  dev_free<int8_t>(vq);
  dev_free<float>(meta);
  dev_free<__nv_bfloat16>(dq);
  dev_free<__nv_bfloat16>(dk);
  dev_free<__nv_bfloat16>(dv);
  dev_free<__nv_bfloat16>(state);
  dev_free<__nv_bfloat16>(dout);
  dev_free<__nv_bfloat16>(exact_state);
  dev_free<__nv_bfloat16>(exact_out);
  dev_free<float>(exact_gate);
  dev_free<float>(exact_beta);
  dev_free<__nv_bfloat16>(norm_in);
  dev_free<__nv_bfloat16>(norm_weight);
  dev_free<__nv_bfloat16>(norm_residual);
  dev_free<__nv_bfloat16>(norm_out);
  dev_free<__nv_bfloat16>(fused_norm_in);
  dev_free<__nv_bfloat16>(fused_norm_weight);
  dev_free<__nv_bfloat16>(fused_norm_out);
  dev_free<uint8_t>(fused_norm_fp4);
  dev_free<uint8_t>(fused_norm_scale);
  dev_free<float>(fused_norm_global);
  dev_free<int32_t>(rope_pos);
  dev_free<__nv_bfloat16>(rope_q);
  dev_free<__nv_bfloat16>(rope_k);
  dev_free<__nv_bfloat16>(swiglu_gate);
  dev_free<__nv_bfloat16>(swiglu_up);
  dev_free<__nv_bfloat16>(swiglu_out);
  dev_free<__nv_bfloat16>(logits);
  dev_free<uint32_t>(sampled);
  dev_free<uint32_t>(sampled_mirror);
  dev_free<__nv_bfloat16>(logits_rows);
  dev_free<uint32_t>(sampled_rows);
  dev_free<uint32_t>(sampled_rows_mirror);
  dev_free<uint32_t>(token_ids);
  dev_free<__nv_bfloat16>(embedding);
  dev_free<__nv_bfloat16>(embedding_out);
  dev_free<__nv_bfloat16>(matvec_input);
  dev_free<__nv_bfloat16>(matvec_weight);
  dev_free<__nv_bfloat16>(matvec_out);
  dev_free<__nv_bfloat16>(bf16_gemm_input);
  dev_free<__nv_bfloat16>(bf16_gemm_weight);
  dev_free<__nv_bfloat16>(bf16_gemm_out);
  dev_free<uint8_t>(bf16_gemm_workspace);
  dev_free<uint8_t>(fp4_weight);
  dev_free<uint8_t>(fp4_scale_raw);
  dev_free<uint8_t>(fp4_scale);
  dev_free<float>(fp4_global);
  dev_free<__nv_bfloat16>(fp4_out);
  dev_free<__nv_bfloat16>(quant_input);
  dev_free<uint8_t>(quant_fp4);
  dev_free<uint8_t>(quant_scale);
  dev_free<float>(quant_global);
  dev_free<uint8_t>(gemm_weight);
  dev_free<uint8_t>(gemm_scale);
  dev_free<__nv_bfloat16>(gemm_out);
  dev_free<uint8_t>(gemm_workspace);
  dev_free<__nv_bfloat16>(conv_input);
  dev_free<__nv_bfloat16>(conv_history);
  dev_free<__nv_bfloat16>(conv_weight);
  dev_free<__nv_bfloat16>(conv_out);
  dev_free<__nv_bfloat16>(gate_a);
  dev_free<__nv_bfloat16>(gate_b);
  dev_free<__nv_bfloat16>(gate_a_log);
  dev_free<__nv_bfloat16>(gate_dt);
  dev_free<float>(gate_out);
  dev_free<float>(beta_out);
  dev_free<__nv_bfloat16>(sigmoid_gate);
  dev_free<__nv_bfloat16>(sigmoid_input);
  dev_free<__nv_bfloat16>(sigmoid_out);

  // ---------------------------------------------------------------------
  // Phase 1.1 — L2 prefetch kernel does not crash on a realistic-sized
  // buffer (32 MB). Just dispatches the kernel on the registered prefetch
  // stream and synchronizes; correctness is validated indirectly via the
  // end-to-end parity gate.
  // ---------------------------------------------------------------------
  {
    constexpr size_t kBytes = 32 * 1024 * 1024;
    qwen36_device_ptr_t buf = dev_alloc<uint8_t>(kBytes);
    must_status(qwen36_cuda_memset(buf, 0x5A, kBytes), "memset prefetch buf");

    qwen36_cuda_stream_t pre_stream = nullptr;
    must_status(qwen36_cuda_stream_create(&pre_stream),
                "stream_create for prefetch smoke");
    qwen36_set_prefetch_stream(pre_stream);

    must_status(qwen36_l2_prefetch(buf, kBytes, 128),
                "l2_prefetch dispatch");
    must_status(qwen36_cuda_stream_synchronize(pre_stream),
                "stream_synchronize after prefetch");

    // Missing prefetch stream is a programming error → expect INVALID_ARGUMENT.
    qwen36_set_prefetch_stream(nullptr);
    expect_status(qwen36_l2_prefetch(buf, kBytes, 128),
                  QWEN36_STATUS_INVALID_ARGUMENT,
                  "l2_prefetch without registered prefetch stream");

    must_status(qwen36_cuda_stream_destroy(pre_stream),
                "stream_destroy prefetch smoke");
    dev_free<uint8_t>(buf);
  }

  // ---------------------------------------------------------------------
  // Phase 2.1 — Megakernel Stage A skeleton: identity copy through the
  // atomic-barrier infrastructure. Validates that the 272-CTA persistent
  // grid + monotonic-counter barrier doesn't deadlock and produces an
  // exact byte copy. Real computation lands in later stages.
  // ---------------------------------------------------------------------
  {
    constexpr size_t kHidden = 5120; // Qwen3.6 hidden_size
    qwen36_device_ptr_t hidden_in = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t hidden_out = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t barrier = dev_alloc<uint32_t>(4);

    // Fill input with a deterministic ramp.
    std::vector<float> ramp(kHidden);
    for (size_t i = 0; i < kHidden; ++i) {
      ramp[i] = static_cast<float>(i % 257) * 0.0125f - 1.0f;
    }
    copy_bf16(hidden_in, ramp);
    must_status(qwen36_cuda_memset(hidden_out, 0xCC, kHidden * sizeof(__nv_bfloat16)),
                "memset hidden_out poison");
    must_status(qwen36_cuda_memset(barrier, 0, 4 * sizeof(uint32_t)),
                "memset barrier zero");

    must_status(qwen36_full_attn_block_stage_a(hidden_in, hidden_out, barrier,
                                                kHidden),
                "full_attn_block_stage_a");

    auto got = read_bf16(hidden_out, kHidden);
    for (size_t i = 0; i < kHidden; ++i) {
      const float expected = __bfloat162float(__float2bfloat16(ramp[i]));
      if (got[i] != expected) {
        fprintf(stderr,
                "megakernel stage A identity copy mismatch at %zu: got %.6f "
                "expected %.6f\n",
                i, got[i], expected);
        exit(1);
      }
    }
    printf("megakernel stage A skeleton OK\n");

    dev_free<uint32_t>(barrier);
    dev_free<__nv_bfloat16>(hidden_out);
    dev_free<__nv_bfloat16>(hidden_in);
  }

  // ---------------------------------------------------------------------
  // Phase 2.2.1 — Megakernel Stage B.1: RMSNorm phase parity.
  // Targets byte-exact match with qwen36_rmsnorm (the reference path used
  // by the engine for non-quantized layer norms).
  // ---------------------------------------------------------------------
  {
    constexpr size_t kHidden = 5120; // Qwen3.6 hidden_size
    constexpr float kEps = 1e-6f;

    qwen36_device_ptr_t b1_input = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t b1_weight = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t b1_ref_out = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t b1_meg_out = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t b1_barrier = dev_alloc<uint32_t>(4);

    // Deterministic input + weight, mixing positive/negative magnitudes
    // typical for hidden activations and small layer-norm weights.
    std::vector<float> input(kHidden);
    std::vector<float> weight(kHidden);
    std::mt19937 rng(0xB1B1B1B1u);
    std::normal_distribution<float> in_dist(0.0f, 0.5f);
    std::normal_distribution<float> w_dist(0.0f, 0.02f);
    for (size_t i = 0; i < kHidden; ++i) {
      input[i] = in_dist(rng);
      weight[i] = w_dist(rng);
    }
    copy_bf16(b1_input, input);
    copy_bf16(b1_weight, weight);
    must_status(qwen36_cuda_memset(b1_ref_out, 0, kHidden * sizeof(__nv_bfloat16)),
                "memset b1_ref_out");
    must_status(qwen36_cuda_memset(b1_meg_out, 0, kHidden * sizeof(__nv_bfloat16)),
                "memset b1_meg_out");
    must_status(qwen36_cuda_memset(b1_barrier, 0, 4 * sizeof(uint32_t)),
                "memset b1_barrier");

    // Reference path: qwen36_rmsnorm with (1+weight) parameterization
    // (direct_weight = 0), no residual.
    qwen36_rmsnorm_spec_t ref_spec{};
    ref_spec.rows = 1;
    ref_spec.hidden = kHidden;
    ref_spec.eps = kEps;
    ref_spec.input_bf16 = b1_input;
    ref_spec.weight_bf16 = b1_weight;
    ref_spec.residual_bf16 = qwen36_device_ptr_t{0};
    ref_spec.residual_out_bf16 = qwen36_device_ptr_t{0};
    ref_spec.output_bf16 = b1_ref_out;
    ref_spec.direct_weight = 0;
    must_status(qwen36_rmsnorm(&ref_spec), "qwen36_rmsnorm reference");

    // Megakernel path.
    must_status(qwen36_full_attn_block_stage_b_rmsnorm(
                    b1_input, b1_weight, b1_meg_out, b1_barrier, kHidden, kEps),
                "qwen36_full_attn_block_stage_b_rmsnorm");

    auto ref = read_bf16(b1_ref_out, kHidden);
    auto meg = read_bf16(b1_meg_out, kHidden);

    // Byte-exact target. If a divergence shows up, downgrade to cosine
    // similarity ≥ 0.998 per the project parity convention.
    double dot = 0.0;
    double nr = 0.0;
    double nm = 0.0;
    size_t mismatches = 0;
    for (size_t i = 0; i < kHidden; ++i) {
      if (ref[i] != meg[i] && mismatches < 4) {
        fprintf(stderr,
                "  b.1 rmsnorm mismatch @ %zu: ref=%.6f meg=%.6f diff=%.3e\n",
                i, ref[i], meg[i], ref[i] - meg[i]);
        ++mismatches;
      } else if (ref[i] != meg[i]) {
        ++mismatches;
      }
      dot += static_cast<double>(ref[i]) * static_cast<double>(meg[i]);
      nr += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
      nm += static_cast<double>(meg[i]) * static_cast<double>(meg[i]);
    }
    const double cos_sim = dot / sqrt(nr * nm);
    if (mismatches == 0) {
      printf("megakernel stage B.1 rmsnorm OK (byte-exact)\n");
    } else {
      printf("megakernel stage B.1 rmsnorm mismatches=%zu cos_sim=%.6f\n",
             mismatches, cos_sim);
      if (cos_sim < 0.998) {
        fprintf(stderr,
                "  cos_sim %.6f below 0.998 floor — failing smoke\n", cos_sim);
        exit(1);
      }
    }

    dev_free<uint32_t>(b1_barrier);
    dev_free<__nv_bfloat16>(b1_meg_out);
    dev_free<__nv_bfloat16>(b1_ref_out);
    dev_free<__nv_bfloat16>(b1_weight);
    dev_free<__nv_bfloat16>(b1_input);
  }

  // ---------------------------------------------------------------------
  // Phase 2.2.2 — Megakernel Stage B.2: fused RMSNorm + NVFP4 quantize
  // parity. Compares the FP4-packed bytes, the per-block E4M3 scales (at
  // the vec16_scale tile positions), the optional bf16 normed copy, and
  // the propagated f32 tensor scale against the reference
  // `qwen36_rmsnorm_nvfp4_quantize` for the no-residual / direct_weight=0
  // path.
  // ---------------------------------------------------------------------
  {
    constexpr size_t kHidden = 5120;
    constexpr float kEps = 1e-6f;
    constexpr float kInputTensorScale = 1.0f;
    constexpr size_t kGroups = (kHidden + 15) / 16; // 320
    constexpr size_t kSfInnerDim = ((kGroups + 3) / 4) * 4; // 320
    // vec16_scale_bytes for one row: div_ceil(1, 128) * (sf_inner_dim/4) * 512
    constexpr size_t kScaleBytes = 1 * (kSfInnerDim / 4) * 512; // 40,960

    qwen36_device_ptr_t b2_input = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t b2_weight = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t b2_ref_bf16 = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t b2_ref_fp4 = dev_alloc<uint8_t>(kHidden / 2);
    qwen36_device_ptr_t b2_ref_scale = dev_alloc<uint8_t>(kScaleBytes);
    qwen36_device_ptr_t b2_ref_tscale = dev_alloc<float>(1);
    qwen36_device_ptr_t b2_meg_bf16 = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t b2_meg_fp4 = dev_alloc<uint8_t>(kHidden / 2);
    qwen36_device_ptr_t b2_meg_scale = dev_alloc<uint8_t>(kScaleBytes);
    qwen36_device_ptr_t b2_meg_tscale = dev_alloc<float>(1);
    qwen36_device_ptr_t b2_barrier = dev_alloc<uint32_t>(4);

    std::vector<float> input(kHidden);
    std::vector<float> weight(kHidden);
    std::mt19937 rng(0xB2B2B2B2u);
    std::normal_distribution<float> in_dist(0.0f, 0.5f);
    std::normal_distribution<float> w_dist(0.0f, 0.02f);
    for (size_t i = 0; i < kHidden; ++i) {
      input[i] = in_dist(rng);
      weight[i] = w_dist(rng);
    }
    copy_bf16(b2_input, input);
    copy_bf16(b2_weight, weight);
    must_status(qwen36_cuda_memset(b2_ref_bf16, 0, kHidden * sizeof(__nv_bfloat16)),
                "memset b2_ref_bf16");
    must_status(qwen36_cuda_memset(b2_ref_fp4, 0, kHidden / 2),
                "memset b2_ref_fp4");
    must_status(qwen36_cuda_memset(b2_ref_scale, 0, kScaleBytes),
                "memset b2_ref_scale");
    must_status(qwen36_cuda_memset(b2_ref_tscale, 0, sizeof(float)),
                "memset b2_ref_tscale");
    must_status(qwen36_cuda_memset(b2_meg_bf16, 0, kHidden * sizeof(__nv_bfloat16)),
                "memset b2_meg_bf16");
    must_status(qwen36_cuda_memset(b2_meg_fp4, 0, kHidden / 2),
                "memset b2_meg_fp4");
    must_status(qwen36_cuda_memset(b2_meg_scale, 0, kScaleBytes),
                "memset b2_meg_scale");
    must_status(qwen36_cuda_memset(b2_meg_tscale, 0, sizeof(float)),
                "memset b2_meg_tscale");
    must_status(qwen36_cuda_memset(b2_barrier, 0, 4 * sizeof(uint32_t)),
                "memset b2_barrier");

    // Reference path.
    qwen36_rmsnorm_nvfp4_quantize_spec_t ref{};
    ref.hidden = kHidden;
    ref.eps = kEps;
    ref.input_bf16 = b2_input;
    ref.weight_bf16 = b2_weight;
    ref.residual_bf16 = qwen36_device_ptr_t{0};
    ref.residual_out_bf16 = qwen36_device_ptr_t{0};
    ref.output_bf16 = b2_ref_bf16;
    ref.output_fp4 = b2_ref_fp4;
    ref.output_scale_e4m3 = b2_ref_scale;
    ref.output_tensor_scale_f32 = b2_ref_tscale;
    ref.input_tensor_scale_f32 = kInputTensorScale;
    must_status(qwen36_rmsnorm_nvfp4_quantize(&ref),
                "qwen36_rmsnorm_nvfp4_quantize reference");

    // Megakernel path.
    must_status(qwen36_full_attn_block_stage_b_rmsnorm_quantize(
                    b2_input, b2_weight, b2_meg_bf16, b2_meg_fp4, b2_meg_scale,
                    b2_meg_tscale, b2_barrier, kHidden, kEps,
                    kInputTensorScale),
                "qwen36_full_attn_block_stage_b_rmsnorm_quantize");

    // Compare bf16, fp4, scales (only the 320 logical positions), tensor scale.
    auto ref_bf16 = read_bf16(b2_ref_bf16, kHidden);
    auto meg_bf16 = read_bf16(b2_meg_bf16, kHidden);
    auto ref_fp4 = read_raw<uint8_t>(b2_ref_fp4, kHidden / 2);
    auto meg_fp4 = read_raw<uint8_t>(b2_meg_fp4, kHidden / 2);
    auto ref_scale_buf = read_raw<uint8_t>(b2_ref_scale, kScaleBytes);
    auto meg_scale_buf = read_raw<uint8_t>(b2_meg_scale, kScaleBytes);
    const float ref_tscale = read_one<float>(b2_ref_tscale);
    const float meg_tscale = read_one<float>(b2_meg_tscale);

    size_t bf16_mismatches = 0;
    for (size_t i = 0; i < kHidden; ++i) {
      if (ref_bf16[i] != meg_bf16[i]) ++bf16_mismatches;
    }
    size_t fp4_mismatches = 0;
    for (size_t i = 0; i < kHidden / 2; ++i) {
      if (ref_fp4[i] != meg_fp4[i]) {
        if (fp4_mismatches < 4) {
          fprintf(stderr, "  b.2 fp4 mismatch @ %zu: ref=0x%02x meg=0x%02x\n",
                  i, ref_fp4[i], meg_fp4[i]);
        }
        ++fp4_mismatches;
      }
    }
    size_t scale_mismatches = 0;
    auto sf_offset = [&](size_t group) {
      const size_t block_inner = (group / 4) * 4;
      const size_t block_offset = block_inner * 128;
      return block_offset + (group % 4);
    };
    for (size_t g = 0; g < kGroups; ++g) {
      const size_t off = sf_offset(g);
      if (ref_scale_buf[off] != meg_scale_buf[off]) {
        if (scale_mismatches < 4) {
          fprintf(stderr,
                  "  b.2 scale mismatch group=%zu @ %zu: ref=0x%02x meg=0x%02x\n",
                  g, off, ref_scale_buf[off], meg_scale_buf[off]);
        }
        ++scale_mismatches;
      }
    }
    const bool tscale_ok = ref_tscale == meg_tscale;

    if (bf16_mismatches == 0 && fp4_mismatches == 0 && scale_mismatches == 0 &&
        tscale_ok) {
      printf("megakernel stage B.2 rmsnorm+quantize OK (byte-exact)\n");
    } else {
      fprintf(stderr,
              "megakernel stage B.2 byte-mismatches: bf16=%zu fp4=%zu scale=%zu "
              "tscale_ok=%d (ref=%.6f meg=%.6f)\n",
              bf16_mismatches, fp4_mismatches, scale_mismatches, tscale_ok,
              ref_tscale, meg_tscale);
      exit(1);
    }

    dev_free<uint32_t>(b2_barrier);
    dev_free<float>(b2_meg_tscale);
    dev_free<uint8_t>(b2_meg_scale);
    dev_free<uint8_t>(b2_meg_fp4);
    dev_free<__nv_bfloat16>(b2_meg_bf16);
    dev_free<float>(b2_ref_tscale);
    dev_free<uint8_t>(b2_ref_scale);
    dev_free<uint8_t>(b2_ref_fp4);
    dev_free<__nv_bfloat16>(b2_ref_bf16);
    dev_free<__nv_bfloat16>(b2_weight);
    dev_free<__nv_bfloat16>(b2_input);
  }

  // ---------------------------------------------------------------------
  // Phase 2.2.3 — Megakernel Stage B.3: Q projection NVFP4 GEMV fused.
  // Compares the megakernel's Q output against the standalone reference
  // path (qwen36_rmsnorm_nvfp4_quantize → qwen36_decode_nvfp4_gemv). Both
  // share the SAME __device__ GEMV body now that the kernel-body extract
  // is in place, so byte-exact equality is the target.
  // ---------------------------------------------------------------------
  {
    constexpr size_t kHidden = 5120;     // K for Q proj — Qwen3.6 hidden
    constexpr size_t kQFeatures = 6144;  // M for Q proj — q_heads * head_dim
    constexpr float kEps = 1e-6f;
    constexpr float kInputTensorScale = 1.0f;
    constexpr float kQAlpha = 1.0f; // pre-folded tensor scales product

    constexpr size_t kActScaleBytes = 40960; // vec16 tile for 320×1 inner×outer
    constexpr size_t kWeightScaleInnerDim = 320; // round_up(5120/16, 4)
    constexpr size_t kWeightScaleBytes =
        ((kQFeatures + 127) / 128) * (kWeightScaleInnerDim / 4) * 512;

    qwen36_device_ptr_t b3_hidden = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t b3_norm_w = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t b3_qweight_fp4 =
        dev_alloc<uint8_t>(kQFeatures * kHidden / 2);
    qwen36_device_ptr_t b3_qweight_scale = dev_alloc<uint8_t>(kWeightScaleBytes);
    qwen36_device_ptr_t b3_normed_bf16 = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t b3_normed_fp4 = dev_alloc<uint8_t>(kHidden / 2);
    qwen36_device_ptr_t b3_normed_scale = dev_alloc<uint8_t>(kActScaleBytes);
    qwen36_device_ptr_t b3_normed_tscale = dev_alloc<float>(1);
    qwen36_device_ptr_t b3_q_ref = dev_alloc<__nv_bfloat16>(kQFeatures);
    qwen36_device_ptr_t b3_q_meg = dev_alloc<__nv_bfloat16>(kQFeatures);
    qwen36_device_ptr_t b3_barrier = dev_alloc<uint32_t>(8);

    // Inputs: small-magnitude random bf16. Weight FP4 bytes: random nibbles
    // restricted to indices 0..3 (FP4 values 0.0, 0.5, 1.0, 1.5) to keep
    // accumulated dot products in bf16-safe range. Scales: all 0x38 (e4m3
    // value 1.0) so the layout assumption is exercised end-to-end without
    // numerical surprises.
    std::vector<float> hidden(kHidden);
    std::vector<float> norm_w(kHidden);
    std::mt19937 rng(0xB3B3B3B3u);
    std::normal_distribution<float> nrm(0.0f, 0.5f);
    std::normal_distribution<float> w_dist(0.0f, 0.02f);
    for (size_t i = 0; i < kHidden; ++i) {
      hidden[i] = nrm(rng);
      norm_w[i] = w_dist(rng);
    }
    copy_bf16(b3_hidden, hidden);
    copy_bf16(b3_norm_w, norm_w);

    std::vector<uint8_t> qweight_bytes(kQFeatures * kHidden / 2);
    std::uniform_int_distribution<uint32_t> nibble_dist(0, 3);
    for (auto &b : qweight_bytes) {
      b = static_cast<uint8_t>(nibble_dist(rng) | (nibble_dist(rng) << 4));
    }
    copy_raw<uint8_t>(b3_qweight_fp4, qweight_bytes);

    std::vector<uint8_t> qweight_scale_bytes(kWeightScaleBytes, 0x38);
    copy_raw<uint8_t>(b3_qweight_scale, qweight_scale_bytes);

    must_status(qwen36_cuda_memset(b3_normed_bf16, 0,
                                   kHidden * sizeof(__nv_bfloat16)),
                "memset b3_normed_bf16");
    must_status(qwen36_cuda_memset(b3_normed_fp4, 0, kHidden / 2),
                "memset b3_normed_fp4");
    must_status(qwen36_cuda_memset(b3_normed_scale, 0, kActScaleBytes),
                "memset b3_normed_scale");
    must_status(qwen36_cuda_memset(b3_normed_tscale, 0, sizeof(float)),
                "memset b3_normed_tscale");
    must_status(qwen36_cuda_memset(b3_q_ref, 0,
                                   kQFeatures * sizeof(__nv_bfloat16)),
                "memset b3_q_ref");
    must_status(qwen36_cuda_memset(b3_q_meg, 0,
                                   kQFeatures * sizeof(__nv_bfloat16)),
                "memset b3_q_meg");
    must_status(qwen36_cuda_memset(b3_barrier, 0, 8 * sizeof(uint32_t)),
                "memset b3_barrier");

    // Reference path: rmsnorm+quantize → standalone GEMV.
    qwen36_rmsnorm_nvfp4_quantize_spec_t rspec{};
    rspec.hidden = kHidden;
    rspec.eps = kEps;
    rspec.input_bf16 = b3_hidden;
    rspec.weight_bf16 = b3_norm_w;
    rspec.residual_bf16 = qwen36_device_ptr_t{0};
    rspec.residual_out_bf16 = qwen36_device_ptr_t{0};
    rspec.output_bf16 = b3_normed_bf16;
    rspec.output_fp4 = b3_normed_fp4;
    rspec.output_scale_e4m3 = b3_normed_scale;
    rspec.output_tensor_scale_f32 = b3_normed_tscale;
    rspec.input_tensor_scale_f32 = kInputTensorScale;
    must_status(qwen36_rmsnorm_nvfp4_quantize(&rspec),
                "b.3 reference rmsnorm_nvfp4_quantize");

    qwen36_nvfp4_gemm_spec_t gspec{};
    gspec.m = kQFeatures;
    gspec.n = 1;
    gspec.k = kHidden;
    gspec.a_fp4 = b3_qweight_fp4;
    gspec.a_scale = b3_qweight_scale;
    gspec.b_fp4 = b3_normed_fp4;
    gspec.b_scale = b3_normed_scale;
    gspec.c_bf16 = b3_q_ref;
    gspec.alpha = kQAlpha;
    must_status(qwen36_decode_nvfp4_gemv(&gspec),
                "b.3 reference decode_nvfp4_gemv");

    // Megakernel Stage B.3 path. Reuses the same intermediate buffers
    // (they get overwritten with the same content — both paths run the
    // identical RMSNorm+quantize math). Validate Q output.
    must_status(qwen36_full_attn_block_stage_b_q_proj(
                    b3_hidden, b3_norm_w, b3_qweight_fp4, b3_qweight_scale,
                    kQAlpha, b3_normed_bf16, b3_normed_fp4, b3_normed_scale,
                    b3_q_meg, b3_barrier, kHidden, kQFeatures, kEps,
                    kInputTensorScale),
                "qwen36_full_attn_block_stage_b_q_proj");

    auto q_ref = read_bf16(b3_q_ref, kQFeatures);
    auto q_meg = read_bf16(b3_q_meg, kQFeatures);
    size_t mismatches = 0;
    double dot = 0.0, nr = 0.0, nm = 0.0;
    for (size_t i = 0; i < kQFeatures; ++i) {
      if (q_ref[i] != q_meg[i] && mismatches < 4) {
        fprintf(stderr,
                "  b.3 Q mismatch @ %zu: ref=%.6f meg=%.6f diff=%.3e\n", i,
                q_ref[i], q_meg[i], q_ref[i] - q_meg[i]);
        ++mismatches;
      } else if (q_ref[i] != q_meg[i]) {
        ++mismatches;
      }
      dot += static_cast<double>(q_ref[i]) * static_cast<double>(q_meg[i]);
      nr += static_cast<double>(q_ref[i]) * static_cast<double>(q_ref[i]);
      nm += static_cast<double>(q_meg[i]) * static_cast<double>(q_meg[i]);
    }
    const double cos_sim = dot / sqrt(nr * nm);
    if (mismatches == 0) {
      printf("megakernel stage B.3 Q proj OK (byte-exact)\n");
    } else {
      printf("megakernel stage B.3 Q proj mismatches=%zu cos_sim=%.6f\n",
             mismatches, cos_sim);
      if (cos_sim < 0.998) {
        fprintf(stderr,
                "  cos_sim %.6f below 0.998 floor — failing smoke\n",
                cos_sim);
        exit(1);
      }
    }

    dev_free<uint32_t>(b3_barrier);
    dev_free<__nv_bfloat16>(b3_q_meg);
    dev_free<__nv_bfloat16>(b3_q_ref);
    dev_free<float>(b3_normed_tscale);
    dev_free<uint8_t>(b3_normed_scale);
    dev_free<uint8_t>(b3_normed_fp4);
    dev_free<__nv_bfloat16>(b3_normed_bf16);
    dev_free<uint8_t>(b3_qweight_scale);
    dev_free<uint8_t>(b3_qweight_fp4);
    dev_free<__nv_bfloat16>(b3_norm_w);
    dev_free<__nv_bfloat16>(b3_hidden);
  }

  // ---------------------------------------------------------------------
  // Phase 2.3 — Megakernel Stage C: Q + K + V projections + partial RoPE
  // fused. Compares (Q_post_rope, K_post_rope, V) bytes against the
  // standalone reference: rmsnorm_nvfp4_quantize → decode_nvfp4_gemv × 3
  // → partial_rope. Same __device__ GEMV body + same RoPE math → byte-
  // exact target. Skips q_proj_deinterleave and q/k norms (those are
  // engine-layer concerns handled in Stage E/engine integration).
  // ---------------------------------------------------------------------
  {
    constexpr size_t kHidden = 5120;
    constexpr size_t kQFeatures = 6144;   // 24 heads × 256 dim
    constexpr size_t kKvFeatures = 1024;  // 4 heads × 256 dim
    constexpr size_t kQHeads = 24;
    constexpr size_t kKvHeads = 4;
    constexpr size_t kHeadDim = 256;
    constexpr size_t kRopeDims = 64;
    constexpr int32_t kPosition = 7;
    constexpr float kBaseTheta = 1000000.0f;
    constexpr float kEps = 1e-6f;
    constexpr float kInputTensorScale = 1.0f;
    constexpr float kQAlpha = 1.0f;
    constexpr float kKAlpha = 1.0f;
    constexpr float kVAlpha = 1.0f;

    constexpr size_t kActScaleBytes = 40960;
    constexpr size_t kQWeightScaleBytes =
        ((kQFeatures + 127) / 128) * (320 / 4) * 512;
    constexpr size_t kKvWeightScaleBytes =
        ((kKvFeatures + 127) / 128) * (320 / 4) * 512;

    qwen36_device_ptr_t c_hidden = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t c_norm_w = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t c_qw_fp4 =
        dev_alloc<uint8_t>(kQFeatures * kHidden / 2);
    qwen36_device_ptr_t c_qw_scale = dev_alloc<uint8_t>(kQWeightScaleBytes);
    qwen36_device_ptr_t c_kw_fp4 =
        dev_alloc<uint8_t>(kKvFeatures * kHidden / 2);
    qwen36_device_ptr_t c_kw_scale = dev_alloc<uint8_t>(kKvWeightScaleBytes);
    qwen36_device_ptr_t c_vw_fp4 =
        dev_alloc<uint8_t>(kKvFeatures * kHidden / 2);
    qwen36_device_ptr_t c_vw_scale = dev_alloc<uint8_t>(kKvWeightScaleBytes);

    qwen36_device_ptr_t c_normed_bf16 = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t c_normed_fp4 = dev_alloc<uint8_t>(kHidden / 2);
    qwen36_device_ptr_t c_normed_scale = dev_alloc<uint8_t>(kActScaleBytes);
    qwen36_device_ptr_t c_normed_tscale = dev_alloc<float>(1);
    qwen36_device_ptr_t c_q_ref = dev_alloc<__nv_bfloat16>(kQFeatures);
    qwen36_device_ptr_t c_k_ref = dev_alloc<__nv_bfloat16>(kKvFeatures);
    qwen36_device_ptr_t c_v_ref = dev_alloc<__nv_bfloat16>(kKvFeatures);
    qwen36_device_ptr_t c_q_meg = dev_alloc<__nv_bfloat16>(kQFeatures);
    qwen36_device_ptr_t c_k_meg = dev_alloc<__nv_bfloat16>(kKvFeatures);
    qwen36_device_ptr_t c_v_meg = dev_alloc<__nv_bfloat16>(kKvFeatures);
    qwen36_device_ptr_t c_barrier = dev_alloc<uint32_t>(8);

    std::vector<float> hidden(kHidden);
    std::vector<float> norm_w(kHidden);
    std::mt19937 rng(0xC0C0C0C0u);
    std::normal_distribution<float> nrm(0.0f, 0.5f);
    std::normal_distribution<float> w_dist(0.0f, 0.02f);
    for (size_t i = 0; i < kHidden; ++i) {
      hidden[i] = nrm(rng);
      norm_w[i] = w_dist(rng);
    }
    copy_bf16(c_hidden, hidden);
    copy_bf16(c_norm_w, norm_w);

    std::uniform_int_distribution<uint32_t> nibble_dist(0, 3);
    auto fill_random_fp4 = [&](qwen36_device_ptr_t dst, size_t bytes) {
      std::vector<uint8_t> buf(bytes);
      for (auto &b : buf) {
        b = static_cast<uint8_t>(nibble_dist(rng) | (nibble_dist(rng) << 4));
      }
      copy_raw<uint8_t>(dst, buf);
    };
    fill_random_fp4(c_qw_fp4, kQFeatures * kHidden / 2);
    fill_random_fp4(c_kw_fp4, kKvFeatures * kHidden / 2);
    fill_random_fp4(c_vw_fp4, kKvFeatures * kHidden / 2);
    std::vector<uint8_t> uniform_scale_q(kQWeightScaleBytes, 0x38);
    std::vector<uint8_t> uniform_scale_kv(kKvWeightScaleBytes, 0x38);
    copy_raw<uint8_t>(c_qw_scale, uniform_scale_q);
    copy_raw<uint8_t>(c_kw_scale, uniform_scale_kv);
    copy_raw<uint8_t>(c_vw_scale, uniform_scale_kv);

    auto zero_outputs = [&]() {
      must_status(qwen36_cuda_memset(c_normed_bf16, 0,
                                     kHidden * sizeof(__nv_bfloat16)),
                  "memset c_normed_bf16");
      must_status(qwen36_cuda_memset(c_normed_fp4, 0, kHidden / 2),
                  "memset c_normed_fp4");
      must_status(qwen36_cuda_memset(c_normed_scale, 0, kActScaleBytes),
                  "memset c_normed_scale");
      must_status(qwen36_cuda_memset(c_normed_tscale, 0, sizeof(float)),
                  "memset c_normed_tscale");
      must_status(qwen36_cuda_memset(c_barrier, 0, 8 * sizeof(uint32_t)),
                  "memset c_barrier");
    };
    zero_outputs();
    must_status(qwen36_cuda_memset(c_q_ref, 0, kQFeatures * sizeof(__nv_bfloat16)),
                "memset c_q_ref");
    must_status(qwen36_cuda_memset(c_k_ref, 0, kKvFeatures * sizeof(__nv_bfloat16)),
                "memset c_k_ref");
    must_status(qwen36_cuda_memset(c_v_ref, 0, kKvFeatures * sizeof(__nv_bfloat16)),
                "memset c_v_ref");
    must_status(qwen36_cuda_memset(c_q_meg, 0, kQFeatures * sizeof(__nv_bfloat16)),
                "memset c_q_meg");
    must_status(qwen36_cuda_memset(c_k_meg, 0, kKvFeatures * sizeof(__nv_bfloat16)),
                "memset c_k_meg");
    must_status(qwen36_cuda_memset(c_v_meg, 0, kKvFeatures * sizeof(__nv_bfloat16)),
                "memset c_v_meg");

    // Reference: rmsnorm+quantize → Q → K → V → partial_rope.
    qwen36_rmsnorm_nvfp4_quantize_spec_t rspec{};
    rspec.hidden = kHidden;
    rspec.eps = kEps;
    rspec.input_bf16 = c_hidden;
    rspec.weight_bf16 = c_norm_w;
    rspec.residual_bf16 = qwen36_device_ptr_t{0};
    rspec.residual_out_bf16 = qwen36_device_ptr_t{0};
    rspec.output_bf16 = c_normed_bf16;
    rspec.output_fp4 = c_normed_fp4;
    rspec.output_scale_e4m3 = c_normed_scale;
    rspec.output_tensor_scale_f32 = c_normed_tscale;
    rspec.input_tensor_scale_f32 = kInputTensorScale;
    must_status(qwen36_rmsnorm_nvfp4_quantize(&rspec),
                "c reference rmsnorm_nvfp4_quantize");

    auto run_gemv = [&](qwen36_device_ptr_t w_fp4, qwen36_device_ptr_t w_scale,
                        qwen36_device_ptr_t out, size_t m, float alpha) {
      qwen36_nvfp4_gemm_spec_t gspec{};
      gspec.m = m;
      gspec.n = 1;
      gspec.k = kHidden;
      gspec.a_fp4 = w_fp4;
      gspec.a_scale = w_scale;
      gspec.b_fp4 = c_normed_fp4;
      gspec.b_scale = c_normed_scale;
      gspec.c_bf16 = out;
      gspec.alpha = alpha;
      must_status(qwen36_decode_nvfp4_gemv(&gspec),
                  "c reference decode_nvfp4_gemv");
    };
    run_gemv(c_qw_fp4, c_qw_scale, c_q_ref, kQFeatures, kQAlpha);
    run_gemv(c_kw_fp4, c_kw_scale, c_k_ref, kKvFeatures, kKAlpha);
    run_gemv(c_vw_fp4, c_vw_scale, c_v_ref, kKvFeatures, kVAlpha);

    qwen36_partial_rope_spec_t pspec{};
    pspec.tokens = 1;
    pspec.q_heads = kQHeads;
    pspec.kv_heads = kKvHeads;
    pspec.head_dim = kHeadDim;
    pspec.rope_dims = kRopeDims;
    pspec.base_theta = kBaseTheta;
    pspec.position_i32 = kPosition;
    pspec.use_scalar_position = 1;
    pspec.positions_i32 = qwen36_device_ptr_t{0};
    pspec.q_bf16 = c_q_ref;
    pspec.k_bf16 = c_k_ref;
    pspec.scalar_position_device_i32 = qwen36_device_ptr_t{0};
    must_status(qwen36_partial_rope(&pspec), "c reference partial_rope");

    // Megakernel Stage C: replays the intermediates, writes _meg outputs.
    zero_outputs();
    must_status(qwen36_full_attn_block_stage_c_qkv_rope(
                    c_hidden, c_norm_w, c_qw_fp4, c_qw_scale, kQAlpha,
                    c_kw_fp4, c_kw_scale, kKAlpha, c_vw_fp4, c_vw_scale,
                    kVAlpha, c_normed_bf16, c_normed_fp4, c_normed_scale,
                    c_q_meg, c_k_meg, c_v_meg, c_barrier, kHidden,
                    kQFeatures, kKvFeatures, kQHeads, kKvHeads, kHeadDim,
                    kRopeDims, kPosition, kBaseTheta, kEps,
                    kInputTensorScale),
                "qwen36_full_attn_block_stage_c_qkv_rope");

    auto compare = [&](qwen36_device_ptr_t a, qwen36_device_ptr_t b,
                       size_t n, const char *label) {
      auto va = read_bf16(a, n);
      auto vb = read_bf16(b, n);
      size_t mm = 0;
      double dot = 0, nr = 0, nm = 0;
      for (size_t i = 0; i < n; ++i) {
        if (va[i] != vb[i]) ++mm;
        dot += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
        nr += static_cast<double>(va[i]) * static_cast<double>(va[i]);
        nm += static_cast<double>(vb[i]) * static_cast<double>(vb[i]);
      }
      const double cs = dot / sqrt(nr * nm);
      if (mm == 0) {
        printf("  stage C %s OK (byte-exact, n=%zu)\n", label, n);
      } else {
        printf("  stage C %s mismatches=%zu cos_sim=%.6f\n", label, mm, cs);
        if (cs < 0.998) {
          fprintf(stderr, "  cos_sim below 0.998 floor — failing smoke\n");
          exit(1);
        }
      }
    };
    compare(c_q_ref, c_q_meg, kQFeatures, "Q (post-RoPE)");
    compare(c_k_ref, c_k_meg, kKvFeatures, "K (post-RoPE)");
    compare(c_v_ref, c_v_meg, kKvFeatures, "V");
    printf("megakernel stage C OK\n");

    dev_free<uint32_t>(c_barrier);
    dev_free<__nv_bfloat16>(c_v_meg);
    dev_free<__nv_bfloat16>(c_k_meg);
    dev_free<__nv_bfloat16>(c_q_meg);
    dev_free<__nv_bfloat16>(c_v_ref);
    dev_free<__nv_bfloat16>(c_k_ref);
    dev_free<__nv_bfloat16>(c_q_ref);
    dev_free<float>(c_normed_tscale);
    dev_free<uint8_t>(c_normed_scale);
    dev_free<uint8_t>(c_normed_fp4);
    dev_free<__nv_bfloat16>(c_normed_bf16);
    dev_free<uint8_t>(c_vw_scale);
    dev_free<uint8_t>(c_vw_fp4);
    dev_free<uint8_t>(c_kw_scale);
    dev_free<uint8_t>(c_kw_fp4);
    dev_free<uint8_t>(c_qw_scale);
    dev_free<uint8_t>(c_qw_fp4);
    dev_free<__nv_bfloat16>(c_norm_w);
    dev_free<__nv_bfloat16>(c_hidden);
  }

  // ---------------------------------------------------------------------
  // Phase 2.5 — Megakernel Stage E: attention output → NVFP4 quantize →
  // o_proj GEMV → residual add → post-attn RMSNorm + NVFP4 quantize, all
  // fused. Compares (residual_out, post_normed, post_fp4, post_scale)
  // against the reference: nvfp4_quantize_bf16 → decode_nvfp4_gemv →
  // rmsnorm_nvfp4_quantize (with residual input).
  // ---------------------------------------------------------------------
  {
    constexpr size_t kHidden = 5120;
    constexpr size_t kQFeatures = 6144;
    constexpr float kEps = 1e-6f;
    constexpr float kAttnTensorScale = 1.0f;
    constexpr float kPostTensorScale = 1.0f;
    constexpr float kOAlpha = 1.0f;

    // Activation scale buffers (vec16 layout). Two sizes:
    //   attn   : 1 row × ceil(q_features/16) inner groups (K side).
    //   normed : 1 row × ceil(hidden/16) inner groups (output of post-norm).
    constexpr size_t kAttnScaleBytes =
        ((1 + 127) / 128) * (((kQFeatures / 16 + 3) / 4) * 4 / 4) * 512;
    constexpr size_t kPostScaleBytes = 40960; // same as kHidden=5120 / 16
    constexpr size_t kOWeightScaleBytes =
        ((kHidden + 127) / 128) *
        (((kQFeatures / 16 + 3) / 4) * 4 / 4) * 512;

    qwen36_device_ptr_t e_attn = dev_alloc<__nv_bfloat16>(kQFeatures);
    qwen36_device_ptr_t e_resid_in = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t e_post_norm_w = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t e_ow_fp4 =
        dev_alloc<uint8_t>(kHidden * kQFeatures / 2);
    qwen36_device_ptr_t e_ow_scale = dev_alloc<uint8_t>(kOWeightScaleBytes);

    // Intermediate (shared between reference and megakernel since both
    // overwrite). attn_quantized: (FP4 + scales) of attention output.
    qwen36_device_ptr_t e_attn_q_fp4 = dev_alloc<uint8_t>(kQFeatures / 2);
    qwen36_device_ptr_t e_attn_q_scale = dev_alloc<uint8_t>(kAttnScaleBytes);
    qwen36_device_ptr_t e_oproj_out = dev_alloc<__nv_bfloat16>(kHidden);

    // Reference outputs.
    qwen36_device_ptr_t e_resid_ref = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t e_normed_ref = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t e_post_fp4_ref = dev_alloc<uint8_t>(kHidden / 2);
    qwen36_device_ptr_t e_post_scale_ref = dev_alloc<uint8_t>(kPostScaleBytes);

    // Megakernel outputs.
    qwen36_device_ptr_t e_resid_meg = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t e_normed_meg = dev_alloc<__nv_bfloat16>(kHidden);
    qwen36_device_ptr_t e_post_fp4_meg = dev_alloc<uint8_t>(kHidden / 2);
    qwen36_device_ptr_t e_post_scale_meg = dev_alloc<uint8_t>(kPostScaleBytes);

    qwen36_device_ptr_t e_barrier = dev_alloc<uint32_t>(8);

    std::mt19937 rng(0xE0E0E0E0u);
    std::normal_distribution<float> in_dist(0.0f, 0.3f);
    std::normal_distribution<float> w_dist(0.0f, 0.02f);
    std::vector<float> attn(kQFeatures), resid(kHidden), norm_w(kHidden);
    for (auto &v : attn) v = in_dist(rng);
    for (auto &v : resid) v = in_dist(rng);
    for (auto &v : norm_w) v = w_dist(rng);
    copy_bf16(e_attn, attn);
    copy_bf16(e_resid_in, resid);
    copy_bf16(e_post_norm_w, norm_w);

    std::vector<uint8_t> ow_bytes(kHidden * kQFeatures / 2);
    std::uniform_int_distribution<uint32_t> nibble_dist(0, 3);
    for (auto &b : ow_bytes) {
      b = static_cast<uint8_t>(nibble_dist(rng) | (nibble_dist(rng) << 4));
    }
    copy_raw<uint8_t>(e_ow_fp4, ow_bytes);
    std::vector<uint8_t> ow_scale_bytes(kOWeightScaleBytes, 0x38);
    copy_raw<uint8_t>(e_ow_scale, ow_scale_bytes);

    auto zero_inter = [&]() {
      must_status(qwen36_cuda_memset(e_attn_q_fp4, 0, kQFeatures / 2),
                  "memset e_attn_q_fp4");
      must_status(qwen36_cuda_memset(e_attn_q_scale, 0, kAttnScaleBytes),
                  "memset e_attn_q_scale");
      must_status(qwen36_cuda_memset(
                      e_oproj_out, 0, kHidden * sizeof(__nv_bfloat16)),
                  "memset e_oproj_out");
      must_status(qwen36_cuda_memset(e_barrier, 0, 8 * sizeof(uint32_t)),
                  "memset e_barrier");
    };
    zero_inter();
    must_status(qwen36_cuda_memset(e_resid_ref, 0,
                                   kHidden * sizeof(__nv_bfloat16)),
                "memset e_resid_ref");
    must_status(qwen36_cuda_memset(e_normed_ref, 0,
                                   kHidden * sizeof(__nv_bfloat16)),
                "memset e_normed_ref");
    must_status(qwen36_cuda_memset(e_post_fp4_ref, 0, kHidden / 2),
                "memset e_post_fp4_ref");
    must_status(qwen36_cuda_memset(e_post_scale_ref, 0, kPostScaleBytes),
                "memset e_post_scale_ref");
    must_status(qwen36_cuda_memset(e_resid_meg, 0,
                                   kHidden * sizeof(__nv_bfloat16)),
                "memset e_resid_meg");
    must_status(qwen36_cuda_memset(e_normed_meg, 0,
                                   kHidden * sizeof(__nv_bfloat16)),
                "memset e_normed_meg");
    must_status(qwen36_cuda_memset(e_post_fp4_meg, 0, kHidden / 2),
                "memset e_post_fp4_meg");
    must_status(qwen36_cuda_memset(e_post_scale_meg, 0, kPostScaleBytes),
                "memset e_post_scale_meg");

    // Reference path.
    // Step 1: quantize attention output.
    qwen36_nvfp4_quantize_spec_t qspec{};
    qspec.values = kQFeatures;
    qspec.input_bf16 = e_attn;
    qspec.output_fp4 = e_attn_q_fp4;
    qspec.output_scale_e4m3 = e_attn_q_scale;
    qspec.output_tensor_scale_f32 = qwen36_device_ptr_t{0};
    qspec.input_tensor_scale_f32 = kAttnTensorScale;
    must_status(qwen36_nvfp4_quantize_bf16(&qspec),
                "e reference nvfp4_quantize_bf16");

    // Step 2: o_proj GEMV.
    qwen36_nvfp4_gemm_spec_t gspec{};
    gspec.m = kHidden;
    gspec.n = 1;
    gspec.k = kQFeatures;
    gspec.a_fp4 = e_ow_fp4;
    gspec.a_scale = e_ow_scale;
    gspec.b_fp4 = e_attn_q_fp4;
    gspec.b_scale = e_attn_q_scale;
    gspec.c_bf16 = e_oproj_out;
    gspec.alpha = kOAlpha;
    must_status(qwen36_decode_nvfp4_gemv(&gspec),
                "e reference decode_nvfp4_gemv");

    // Step 3: post-norm + quantize with residual_in folded in.
    qwen36_rmsnorm_nvfp4_quantize_spec_t rspec{};
    rspec.hidden = kHidden;
    rspec.eps = kEps;
    rspec.input_bf16 = e_oproj_out;
    rspec.weight_bf16 = e_post_norm_w;
    rspec.residual_bf16 = e_resid_in;
    rspec.residual_out_bf16 = e_resid_ref;
    rspec.output_bf16 = e_normed_ref;
    rspec.output_fp4 = e_post_fp4_ref;
    rspec.output_scale_e4m3 = e_post_scale_ref;
    rspec.output_tensor_scale_f32 = qwen36_device_ptr_t{0};
    rspec.input_tensor_scale_f32 = kPostTensorScale;
    must_status(qwen36_rmsnorm_nvfp4_quantize(&rspec),
                "e reference rmsnorm_nvfp4_quantize");

    // Megakernel path (overwrites intermediates with identical math).
    zero_inter();
    must_status(qwen36_full_attn_block_stage_e_o_proj_residual_norm(
                    e_attn, e_resid_in, e_ow_fp4, e_ow_scale, kOAlpha,
                    e_post_norm_w, e_attn_q_fp4, e_attn_q_scale, e_oproj_out,
                    e_resid_meg, e_normed_meg, e_post_fp4_meg,
                    e_post_scale_meg, e_barrier, kQFeatures, kHidden, kEps,
                    kPostTensorScale, kAttnTensorScale),
                "qwen36_full_attn_block_stage_e_o_proj_residual_norm");

    // Compare.
    auto cmp_bf16 = [&](qwen36_device_ptr_t a, qwen36_device_ptr_t b,
                        size_t n, const char *label) {
      auto va = read_bf16(a, n);
      auto vb = read_bf16(b, n);
      size_t mm = 0;
      double dot = 0, nr = 0, nm = 0;
      for (size_t i = 0; i < n; ++i) {
        if (va[i] != vb[i]) ++mm;
        dot += static_cast<double>(va[i]) * static_cast<double>(vb[i]);
        nr += static_cast<double>(va[i]) * static_cast<double>(va[i]);
        nm += static_cast<double>(vb[i]) * static_cast<double>(vb[i]);
      }
      const double cs = dot / sqrt(nr * nm);
      if (mm == 0) {
        printf("  stage E %s OK (byte-exact, n=%zu)\n", label, n);
      } else {
        printf("  stage E %s mismatches=%zu cos_sim=%.6f\n", label, mm, cs);
        if (cs < 0.998) {
          fprintf(stderr, "  cos_sim below 0.998 floor — failing smoke\n");
          exit(1);
        }
      }
    };
    auto cmp_bytes = [&](qwen36_device_ptr_t a, qwen36_device_ptr_t b,
                         size_t n, const char *label) {
      auto va = read_raw<uint8_t>(a, n);
      auto vb = read_raw<uint8_t>(b, n);
      size_t mm = 0;
      for (size_t i = 0; i < n; ++i)
        if (va[i] != vb[i]) ++mm;
      if (mm == 0) {
        printf("  stage E %s OK (byte-exact, n=%zu)\n", label, n);
      } else {
        fprintf(stderr, "  stage E %s mismatches=%zu\n", label, mm);
        exit(1);
      }
    };
    cmp_bf16(e_resid_ref, e_resid_meg, kHidden, "residual_out");
    cmp_bf16(e_normed_ref, e_normed_meg, kHidden, "post_normed");
    cmp_bytes(e_post_fp4_ref, e_post_fp4_meg, kHidden / 2, "post_fp4");

    // Scales: compare only the 320 logical positions (the rest is tile
    // padding written-but-meaningless).
    auto ref_s = read_raw<uint8_t>(e_post_scale_ref, kPostScaleBytes);
    auto meg_s = read_raw<uint8_t>(e_post_scale_meg, kPostScaleBytes);
    auto sf_off = [&](size_t g) {
      const size_t block_inner = (g / 4) * 4;
      return block_inner * 128 + (g % 4);
    };
    size_t scale_mm = 0;
    for (size_t g = 0; g < kHidden / 16; ++g) {
      if (ref_s[sf_off(g)] != meg_s[sf_off(g)]) ++scale_mm;
    }
    if (scale_mm == 0) {
      printf("  stage E post_scale OK (byte-exact, n=%zu groups)\n",
             kHidden / 16);
    } else {
      fprintf(stderr, "  stage E post_scale mismatches=%zu\n", scale_mm);
      exit(1);
    }
    printf("megakernel stage E OK\n");

    dev_free<uint32_t>(e_barrier);
    dev_free<uint8_t>(e_post_scale_meg);
    dev_free<uint8_t>(e_post_fp4_meg);
    dev_free<__nv_bfloat16>(e_normed_meg);
    dev_free<__nv_bfloat16>(e_resid_meg);
    dev_free<uint8_t>(e_post_scale_ref);
    dev_free<uint8_t>(e_post_fp4_ref);
    dev_free<__nv_bfloat16>(e_normed_ref);
    dev_free<__nv_bfloat16>(e_resid_ref);
    dev_free<__nv_bfloat16>(e_oproj_out);
    dev_free<uint8_t>(e_attn_q_scale);
    dev_free<uint8_t>(e_attn_q_fp4);
    dev_free<uint8_t>(e_ow_scale);
    dev_free<uint8_t>(e_ow_fp4);
    dev_free<__nv_bfloat16>(e_post_norm_w);
    dev_free<__nv_bfloat16>(e_resid_in);
    dev_free<__nv_bfloat16>(e_attn);
  }

  // ---------------------------------------------------------------------
  // Phase 2.6 — Megakernel Stage F.1: MLP gate+up NVFP4 GEMV fused.
  // Compares the megakernel's BF16 output against the standalone
  // qwen36_decode_nvfp4_gemv path. Both call the same __device__ GEMV
  // body (post-refactor), so byte-exact equality is the target. Stage
  // F.1 is the first MLP sub-phase; F.2/F.3/F.4 will append SwiGLU +
  // down GEMV + residual_add to the same launch.
  //
  // Shape uses K=5120 (production hidden_size) but a reduced
  // intermediate (=4096 → M=8192) so the smoke's weight allocation
  // stays ~20 MB instead of the production 89 MB. The persistent
  // work-stealing loop is still exercised because M-tile count (512)
  // exceeds the persistent grid size (256 CTAs) — each CTA grabs ~2
  // tiles. Engine-side parity (chat / dump-logits) covers the
  // production M=34816 path.
  // ---------------------------------------------------------------------
  {
    constexpr size_t kHidden = 5120;        // K for gate+up GEMV
    constexpr size_t kIntermediate = 4096;  // smoke-only reduction
    constexpr size_t kTwoIntermediate = 2 * kIntermediate;
    constexpr float kGateUpAlpha = 1.0f; // pre-folded tensor scales product

    // Activation: NVFP4 quantized [hidden_size] in the vec16 tile layout.
    constexpr size_t kActFp4Bytes = kHidden / 2;
    constexpr size_t kActScaleBytes = 40960; // vec16 tile for 320×1 inner×outer

    // Weight: NVFP4 [2*intermediate, hidden_size/2] packed.
    constexpr size_t kWeightFp4Bytes = kTwoIntermediate * kHidden / 2;
    constexpr size_t kWeightScaleInnerDim = 320; // round_up(5120/16, 4)
    constexpr size_t kWeightScaleBytes =
        ((kTwoIntermediate + 127) / 128) * (kWeightScaleInnerDim / 4) * 512;

    qwen36_device_ptr_t f_act_fp4 = dev_alloc<uint8_t>(kActFp4Bytes);
    qwen36_device_ptr_t f_act_scale = dev_alloc<uint8_t>(kActScaleBytes);
    qwen36_device_ptr_t f_w_fp4 = dev_alloc<uint8_t>(kWeightFp4Bytes);
    qwen36_device_ptr_t f_w_scale = dev_alloc<uint8_t>(kWeightScaleBytes);
    qwen36_device_ptr_t f_gu_ref = dev_alloc<__nv_bfloat16>(kTwoIntermediate);
    qwen36_device_ptr_t f_gu_meg = dev_alloc<__nv_bfloat16>(kTwoIntermediate);
    qwen36_device_ptr_t f_barrier = dev_alloc<uint32_t>(4);

    // Random FP4 nibbles restricted to 0..3 (values 0.0, 0.5, 1.0, 1.5) so
    // K=5120 accumulated dot products stay bf16-safe. Scales: 0x38 = e4m3
    // value 1.0 throughout. Matches the Stage B.3 smoke's recipe.
    std::mt19937 rng(0xF1F1F1F1u);
    std::uniform_int_distribution<uint32_t> nibble_dist(0, 3);
    std::vector<uint8_t> act_bytes(kActFp4Bytes);
    for (auto &b : act_bytes) {
      b = static_cast<uint8_t>(nibble_dist(rng) | (nibble_dist(rng) << 4));
    }
    copy_raw<uint8_t>(f_act_fp4, act_bytes);
    std::vector<uint8_t> act_scale_bytes(kActScaleBytes, 0x38);
    copy_raw<uint8_t>(f_act_scale, act_scale_bytes);

    std::vector<uint8_t> w_bytes(kWeightFp4Bytes);
    for (auto &b : w_bytes) {
      b = static_cast<uint8_t>(nibble_dist(rng) | (nibble_dist(rng) << 4));
    }
    copy_raw<uint8_t>(f_w_fp4, w_bytes);
    std::vector<uint8_t> w_scale_bytes(kWeightScaleBytes, 0x38);
    copy_raw<uint8_t>(f_w_scale, w_scale_bytes);

    must_status(qwen36_cuda_memset(f_gu_ref, 0,
                                   kTwoIntermediate * sizeof(__nv_bfloat16)),
                "memset f_gu_ref");
    must_status(qwen36_cuda_memset(f_gu_meg, 0,
                                   kTwoIntermediate * sizeof(__nv_bfloat16)),
                "memset f_gu_meg");
    must_status(qwen36_cuda_memset(f_barrier, 0, 4 * sizeof(uint32_t)),
                "memset f_barrier");

    // Reference path: standalone gate+up NVFP4 GEMV.
    qwen36_nvfp4_gemm_spec_t gspec{};
    gspec.m = kTwoIntermediate;
    gspec.n = 1;
    gspec.k = kHidden;
    gspec.a_fp4 = f_w_fp4;
    gspec.a_scale = f_w_scale;
    gspec.b_fp4 = f_act_fp4;
    gspec.b_scale = f_act_scale;
    gspec.c_bf16 = f_gu_ref;
    gspec.alpha = kGateUpAlpha;
    must_status(qwen36_decode_nvfp4_gemv(&gspec),
                "f.1 reference decode_nvfp4_gemv");

    // Megakernel Stage F.1 path.
    must_status(qwen36_full_attn_block_stage_f1_gate_up(
                    f_act_fp4, f_act_scale, f_w_fp4, f_w_scale, kGateUpAlpha,
                    f_gu_meg, f_barrier, kIntermediate, kHidden),
                "qwen36_full_attn_block_stage_f1_gate_up");

    auto gu_ref = read_bf16(f_gu_ref, kTwoIntermediate);
    auto gu_meg = read_bf16(f_gu_meg, kTwoIntermediate);
    size_t mismatches = 0;
    double dot = 0.0, nr = 0.0, nm = 0.0;
    for (size_t i = 0; i < kTwoIntermediate; ++i) {
      if (gu_ref[i] != gu_meg[i] && mismatches < 4) {
        fprintf(stderr,
                "  f.1 gate_up mismatch @ %zu: ref=%.6f meg=%.6f diff=%.3e\n",
                i, gu_ref[i], gu_meg[i], gu_ref[i] - gu_meg[i]);
        ++mismatches;
      } else if (gu_ref[i] != gu_meg[i]) {
        ++mismatches;
      }
      dot += static_cast<double>(gu_ref[i]) * static_cast<double>(gu_meg[i]);
      nr += static_cast<double>(gu_ref[i]) * static_cast<double>(gu_ref[i]);
      nm += static_cast<double>(gu_meg[i]) * static_cast<double>(gu_meg[i]);
    }
    const double cos_sim = dot / sqrt(nr * nm);
    if (mismatches == 0) {
      printf("megakernel stage F.1 gate+up OK (byte-exact, n=%zu)\n",
             kTwoIntermediate);
    } else {
      printf("megakernel stage F.1 gate+up mismatches=%zu cos_sim=%.6f\n",
             mismatches, cos_sim);
      if (cos_sim < 0.998) {
        fprintf(stderr, "  cos_sim %.6f below 0.998 floor — failing smoke\n",
                cos_sim);
        exit(1);
      }
    }

    dev_free<uint32_t>(f_barrier);
    dev_free<__nv_bfloat16>(f_gu_meg);
    dev_free<__nv_bfloat16>(f_gu_ref);
    dev_free<uint8_t>(f_w_scale);
    dev_free<uint8_t>(f_w_fp4);
    dev_free<uint8_t>(f_act_scale);
    dev_free<uint8_t>(f_act_fp4);
  }

  // ---------------------------------------------------------------------
  // Phase 2.6 — Megakernel Stage F.2: gate+up + SwiGLU + NVFP4 quantize.
  // Compares the megakernel's swiglu FP4 + scale output against the
  // standalone reference path (qwen36_decode_nvfp4_gemv →
  // qwen36_swiglu_nvfp4_quantize). The megakernel's fused phase 1
  // mirrors the standalone kernel exactly (same amax reduction, same
  // e4m3 scale rounding, same e2m1 nibble packing, same vec16 layout)
  // so byte-exact equality is the target.
  // ---------------------------------------------------------------------
  {
    constexpr size_t kHidden = 5120;
    constexpr size_t kIntermediate = 4096;
    constexpr size_t kTwoIntermediate = 2 * kIntermediate;
    constexpr float kGateUpAlpha = 1.0f;
    constexpr float kDownInputTensorScale = 1.0f;

    constexpr size_t kActFp4Bytes = kHidden / 2;
    constexpr size_t kActScaleBytes = 40960;
    constexpr size_t kWeightFp4Bytes = kTwoIntermediate * kHidden / 2;
    constexpr size_t kWeightScaleInnerDim = 320;
    constexpr size_t kWeightScaleBytes =
        ((kTwoIntermediate + 127) / 128) * (kWeightScaleInnerDim / 4) * 512;
    constexpr size_t kSwigluScaleInnerDim =
        ((kIntermediate / 16) + 3) & ~size_t(3);
    constexpr size_t kSwigluFp4Bytes = kIntermediate / 2;
    constexpr size_t kSwigluScaleBytes =
        ((1 + 127) / 128 + 1) * (kSwigluScaleInnerDim / 4) * 512;

    qwen36_device_ptr_t f2_act_fp4 = dev_alloc<uint8_t>(kActFp4Bytes);
    qwen36_device_ptr_t f2_act_scale = dev_alloc<uint8_t>(kActScaleBytes);
    qwen36_device_ptr_t f2_w_fp4 = dev_alloc<uint8_t>(kWeightFp4Bytes);
    qwen36_device_ptr_t f2_w_scale = dev_alloc<uint8_t>(kWeightScaleBytes);
    qwen36_device_ptr_t f2_gu_ref = dev_alloc<__nv_bfloat16>(kTwoIntermediate);
    qwen36_device_ptr_t f2_gu_meg = dev_alloc<__nv_bfloat16>(kTwoIntermediate);
    qwen36_device_ptr_t f2_sw_fp4_ref = dev_alloc<uint8_t>(kSwigluFp4Bytes);
    qwen36_device_ptr_t f2_sw_scale_ref = dev_alloc<uint8_t>(kSwigluScaleBytes);
    qwen36_device_ptr_t f2_sw_tscale_ref = dev_alloc<float>(1);
    qwen36_device_ptr_t f2_sw_fp4_meg = dev_alloc<uint8_t>(kSwigluFp4Bytes);
    qwen36_device_ptr_t f2_sw_scale_meg = dev_alloc<uint8_t>(kSwigluScaleBytes);
    qwen36_device_ptr_t f2_barrier = dev_alloc<uint32_t>(8);

    std::mt19937 rng(0xF2F2F2F2u);
    std::uniform_int_distribution<uint32_t> nibble_dist(0, 3);
    std::vector<uint8_t> act_bytes(kActFp4Bytes);
    for (auto &b : act_bytes) {
      b = static_cast<uint8_t>(nibble_dist(rng) | (nibble_dist(rng) << 4));
    }
    copy_raw<uint8_t>(f2_act_fp4, act_bytes);
    std::vector<uint8_t> act_scale_bytes(kActScaleBytes, 0x38);
    copy_raw<uint8_t>(f2_act_scale, act_scale_bytes);
    std::vector<uint8_t> w_bytes(kWeightFp4Bytes);
    for (auto &b : w_bytes) {
      b = static_cast<uint8_t>(nibble_dist(rng) | (nibble_dist(rng) << 4));
    }
    copy_raw<uint8_t>(f2_w_fp4, w_bytes);
    std::vector<uint8_t> w_scale_bytes(kWeightScaleBytes, 0x38);
    copy_raw<uint8_t>(f2_w_scale, w_scale_bytes);

    must_status(qwen36_cuda_memset(f2_gu_ref, 0,
                                   kTwoIntermediate * sizeof(__nv_bfloat16)),
                "memset f2_gu_ref");
    must_status(qwen36_cuda_memset(f2_gu_meg, 0,
                                   kTwoIntermediate * sizeof(__nv_bfloat16)),
                "memset f2_gu_meg");
    must_status(qwen36_cuda_memset(f2_sw_fp4_ref, 0, kSwigluFp4Bytes),
                "memset f2_sw_fp4_ref");
    must_status(qwen36_cuda_memset(f2_sw_scale_ref, 0, kSwigluScaleBytes),
                "memset f2_sw_scale_ref");
    must_status(qwen36_cuda_memset(f2_sw_tscale_ref, 0, sizeof(float)),
                "memset f2_sw_tscale_ref");
    must_status(qwen36_cuda_memset(f2_sw_fp4_meg, 0, kSwigluFp4Bytes),
                "memset f2_sw_fp4_meg");
    must_status(qwen36_cuda_memset(f2_sw_scale_meg, 0, kSwigluScaleBytes),
                "memset f2_sw_scale_meg");
    must_status(qwen36_cuda_memset(f2_barrier, 0, 8 * sizeof(uint32_t)),
                "memset f2_barrier");

    // Reference path: gate+up GEMV → swiglu_nvfp4_quantize.
    qwen36_nvfp4_gemm_spec_t gspec{};
    gspec.m = kTwoIntermediate;
    gspec.n = 1;
    gspec.k = kHidden;
    gspec.a_fp4 = f2_w_fp4;
    gspec.a_scale = f2_w_scale;
    gspec.b_fp4 = f2_act_fp4;
    gspec.b_scale = f2_act_scale;
    gspec.c_bf16 = f2_gu_ref;
    gspec.alpha = kGateUpAlpha;
    must_status(qwen36_decode_nvfp4_gemv(&gspec),
                "f.2 reference decode_nvfp4_gemv");

    qwen36_swiglu_nvfp4_quantize_spec_t swspec{};
    swspec.intermediate = kIntermediate;
    swspec.gate_bf16 = f2_gu_ref;
    swspec.up_bf16 =
        qwen36_device_ptr_t{f2_gu_ref.ptr +
                            kIntermediate * sizeof(__nv_bfloat16)};
    swspec.output_fp4 = f2_sw_fp4_ref;
    swspec.output_scale_e4m3 = f2_sw_scale_ref;
    swspec.output_tensor_scale_f32 = f2_sw_tscale_ref;
    swspec.input_tensor_scale_f32 = kDownInputTensorScale;
    must_status(qwen36_swiglu_nvfp4_quantize(&swspec),
                "f.2 reference swiglu_nvfp4_quantize");

    // Megakernel path: F.2 (gate+up + swiglu fused).
    must_status(qwen36_full_attn_block_stage_f2_gate_up_swiglu(
                    f2_act_fp4, f2_act_scale, f2_w_fp4, f2_w_scale,
                    kGateUpAlpha, f2_gu_meg, f2_sw_fp4_meg, f2_sw_scale_meg,
                    f2_barrier, kIntermediate, kHidden,
                    kDownInputTensorScale),
                "qwen36_full_attn_block_stage_f2_gate_up_swiglu");

    auto sw_fp4_ref = read_raw<uint8_t>(f2_sw_fp4_ref, kSwigluFp4Bytes);
    auto sw_fp4_meg = read_raw<uint8_t>(f2_sw_fp4_meg, kSwigluFp4Bytes);
    size_t fp4_mm = 0;
    for (size_t i = 0; i < kSwigluFp4Bytes; ++i) {
      if (sw_fp4_ref[i] != sw_fp4_meg[i]) {
        if (fp4_mm < 4) {
          fprintf(stderr,
                  "  f.2 swiglu fp4 mismatch @ %zu: ref=0x%02x meg=0x%02x\n",
                  i, sw_fp4_ref[i], sw_fp4_meg[i]);
        }
        ++fp4_mm;
      }
    }
    if (fp4_mm == 0) {
      printf("  stage F.2 swiglu_fp4 OK (byte-exact, n=%zu)\n",
             kSwigluFp4Bytes);
    } else {
      fprintf(stderr, "  stage F.2 swiglu_fp4 mismatches=%zu\n", fp4_mm);
      exit(1);
    }

    auto sw_scale_ref = read_raw<uint8_t>(f2_sw_scale_ref, kSwigluScaleBytes);
    auto sw_scale_meg = read_raw<uint8_t>(f2_sw_scale_meg, kSwigluScaleBytes);
    size_t scale_mm = 0;
    for (size_t i = 0; i < kSwigluScaleBytes; ++i) {
      if (sw_scale_ref[i] != sw_scale_meg[i]) {
        if (scale_mm < 4) {
          fprintf(stderr,
                  "  f.2 swiglu scale mismatch @ %zu: ref=0x%02x meg=0x%02x\n",
                  i, sw_scale_ref[i], sw_scale_meg[i]);
        }
        ++scale_mm;
      }
    }
    if (scale_mm == 0) {
      printf("  stage F.2 swiglu_scale OK (byte-exact, n=%zu groups)\n",
             kIntermediate / 16);
    } else {
      fprintf(stderr, "  stage F.2 swiglu_scale mismatches=%zu\n", scale_mm);
      exit(1);
    }
    printf("megakernel stage F.2 OK\n");

    dev_free<uint32_t>(f2_barrier);
    dev_free<uint8_t>(f2_sw_scale_meg);
    dev_free<uint8_t>(f2_sw_fp4_meg);
    dev_free<float>(f2_sw_tscale_ref);
    dev_free<uint8_t>(f2_sw_scale_ref);
    dev_free<uint8_t>(f2_sw_fp4_ref);
    dev_free<__nv_bfloat16>(f2_gu_meg);
    dev_free<__nv_bfloat16>(f2_gu_ref);
    dev_free<uint8_t>(f2_w_scale);
    dev_free<uint8_t>(f2_w_fp4);
    dev_free<uint8_t>(f2_act_scale);
    dev_free<uint8_t>(f2_act_fp4);
  }

  // ---------------------------------------------------------------------
  // Phase 0.4 — Cross-stream sync inside a captured CUDA graph.
  // Validates that the prefetch stream + event ABI added in Phase 0.1 is
  // graph-captureable: main writes A, prefetch waits, prefetch writes B,
  // main waits, main writes C. If event capture is wired wrong the graph
  // will deadlock at instantiate or produce stale bytes.
  // ---------------------------------------------------------------------
  {
    qwen36_device_ptr_t buf_a = dev_alloc<uint8_t>(1);
    qwen36_device_ptr_t buf_b = dev_alloc<uint8_t>(1);
    qwen36_device_ptr_t buf_c = dev_alloc<uint8_t>(1);
    must_status(qwen36_cuda_memset(buf_a, 0x00, 1), "memset buf_a init");
    must_status(qwen36_cuda_memset(buf_b, 0x00, 1), "memset buf_b init");
    must_status(qwen36_cuda_memset(buf_c, 0x00, 1), "memset buf_c init");

    // must_status() calls cudaDeviceSynchronize() which is illegal during
    // capture, so we use a no-sync variant for everything between begin and
    // end_capture (and for begin_capture itself, which leaves the stream in
    // capture mode immediately).
    auto must_status_async = [](int status, const char *what) {
      if (status != QWEN36_STATUS_SUCCESS) {
        fprintf(stderr, "%s returned status %d\n", what, status);
        exit(1);
      }
    };

    qwen36_cuda_stream_t xstream_main = nullptr;
    qwen36_cuda_stream_t xstream_pref = nullptr;
    must_status(qwen36_cuda_stream_create(&xstream_main),
                "stream_create main");
    must_status(qwen36_cuda_stream_create(&xstream_pref),
                "stream_create prefetch");
    qwen36_set_active_stream(xstream_main);
    qwen36_set_prefetch_stream(xstream_pref);

    qwen36_cuda_event_t event_a = nullptr;
    qwen36_cuda_event_t event_b = nullptr;
    must_status(qwen36_cuda_event_create(&event_a), "event_create a");
    must_status(qwen36_cuda_event_create(&event_b), "event_create b");

    must_status_async(qwen36_cuda_stream_begin_capture(xstream_main),
                      "begin_capture main");

    // 1) main writes 0xAA into buf_a.
    must_cuda<uint8_t>(
        cudaMemsetAsync(reinterpret_cast<void *>(buf_a.ptr), 0xAA, 1,
                        reinterpret_cast<cudaStream_t>(xstream_main)),
        "memsetAsync buf_a on main");
    // 2) record event on main → prefetch waits.
    must_status_async(qwen36_cuda_event_record(event_a, xstream_main),
                      "event_record a on main");
    must_status_async(qwen36_cuda_stream_wait_event(xstream_pref, event_a),
                      "stream_wait_event prefetch waits for a");
    // 3) prefetch writes 0xBB into buf_b (must happen after main's write).
    must_cuda<uint8_t>(
        cudaMemsetAsync(reinterpret_cast<void *>(buf_b.ptr), 0xBB, 1,
                        reinterpret_cast<cudaStream_t>(xstream_pref)),
        "memsetAsync buf_b on prefetch");
    // 4) record event on prefetch → main waits.
    must_status_async(qwen36_cuda_event_record(event_b, xstream_pref),
                      "event_record b on prefetch");
    must_status_async(qwen36_cuda_stream_wait_event(xstream_main, event_b),
                      "stream_wait_event main waits for b");
    // 5) main writes 0xCC into buf_c (must happen after prefetch's write).
    must_cuda<uint8_t>(
        cudaMemsetAsync(reinterpret_cast<void *>(buf_c.ptr), 0xCC, 1,
                        reinterpret_cast<cudaStream_t>(xstream_main)),
        "memsetAsync buf_c on main");

    qwen36_cuda_graph_t graph = nullptr;
    must_status_async(qwen36_cuda_stream_end_capture(xstream_main, &graph),
                      "end_capture main");
    qwen36_cuda_graph_exec_t exec = nullptr;
    must_status(qwen36_cuda_graph_instantiate(graph, &exec),
                "graph_instantiate");
    must_status(qwen36_cuda_graph_launch(exec, xstream_main),
                "graph_launch");
    must_status(qwen36_cuda_stream_synchronize(xstream_main),
                "stream_synchronize after replay");

    const uint8_t got_a = read_one<uint8_t>(buf_a);
    const uint8_t got_b = read_one<uint8_t>(buf_b);
    const uint8_t got_c = read_one<uint8_t>(buf_c);
    if (got_a != 0xAA || got_b != 0xBB || got_c != 0xCC) {
      fprintf(stderr,
              "cross-stream graph capture failed: a=0x%02x b=0x%02x c=0x%02x "
              "(expected 0xAA, 0xBB, 0xCC)\n",
              got_a, got_b, got_c);
      exit(1);
    }

    must_status(qwen36_cuda_graph_exec_destroy(exec), "graph_exec_destroy");
    must_status(qwen36_cuda_graph_destroy(graph), "graph_destroy");
    must_status(qwen36_cuda_event_destroy(event_a), "event_destroy a");
    must_status(qwen36_cuda_event_destroy(event_b), "event_destroy b");
    qwen36_set_prefetch_stream(nullptr);
    qwen36_set_active_stream(nullptr);
    must_status(qwen36_cuda_stream_destroy(xstream_pref),
                "stream_destroy prefetch");
    must_status(qwen36_cuda_stream_destroy(xstream_main),
                "stream_destroy main");
    dev_free<uint8_t>(buf_a);
    dev_free<uint8_t>(buf_b);
    dev_free<uint8_t>(buf_c);
  }

  printf("qwen36 CUDA smoke test passed\n");
  return 0;
}
