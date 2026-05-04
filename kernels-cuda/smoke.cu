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

  // Direction B Phase B2 (Option C): hand-rolled NVFP4 gemv at N=1.
  // Reuses the same planted weights/activation as the qwen36_nvfp4_gemm
  // probe above; expected per-row output is the same value (~132).
  qwen36_nvfp4_gemm_spec_t gemv_b2_spec = gemm_spec;
  gemv_b2_spec.n = 1;
  must_status(qwen36_decode_nvfp4_gemv(&gemv_b2_spec), "decode_gemv b2");
  std::vector<float> gemv_b2_values = read_bf16(gemm_spec.c_bf16, gemm_m);
  expect_close(gemv_b2_values[0], 132.0f, 4.0f, "decode_gemv b2[0]");
  expect_close(gemv_b2_values[gemm_m - 1], 132.0f, 4.0f,
               "decode_gemv b2[last]");

  // B1 unsupported-shape probe stays as the soft-fallback regression
  // gate: must still return NOT_IMPLEMENTED for shapes outside the
  // supported regime so the dispatcher routes to cuBLASLt.
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

  printf("qwen36 CUDA smoke test passed\n");
  return 0;
}
