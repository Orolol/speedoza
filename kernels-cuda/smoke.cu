#include "qwen36_fp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

// FA-tiled drafter attention entry — not in the public header (it is the
// internal fast path the dispatcher tries first). Declared here so the
// parity gate can call it directly. extern "C" must be at namespace scope.
extern "C" int qwen36_drafter_attention_block_flash_bf16(
    const qwen36_drafter_attention_block_spec_t *spec);

// Flash-Decoding split-K prefill entry (P2). Not in the public header;
// declared here so the parity gate can compare it against the scalar GQA
// reference (qwen36_attention_prefill) at the q=16 verify shape.
extern "C" int qwen36_attention_flash_splitk_prefill_bf16(
    const qwen36_attention_prefill_spec_t *spec, int n_splits);

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
  qwen36_device_ptr_t interpreter_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(3);
  qwen36_device_ptr_t interpreter_counters = dev_alloc<int32_t>(4);
  qwen36_interpreter_instruction_t interpreter_program_host[3]{};
  interpreter_program_host[0].opcode = 1; // FALLBACK_TRAMPOLINE
  interpreter_program_host[0].publishes_counter = 0;
  interpreter_program_host[0].publish_value = 1;
  interpreter_program_host[0].arrival_counter = 2;
  interpreter_program_host[1].opcode = 1; // FALLBACK_TRAMPOLINE
  interpreter_program_host[1].dep_count = 1;
  interpreter_program_host[1].deps[0] = qwen36_interpreter_dep_t{0, 1};
  interpreter_program_host[1].publishes_counter = 1;
  interpreter_program_host[1].publish_value = 1;
  interpreter_program_host[1].arrival_counter = 3;
  interpreter_program_host[2].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interpreter_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interpreter_program_host, interpreter_program_host + 3));
  must_cuda<int32_t>(cudaMemset(reinterpret_cast<void *>(interpreter_counters.ptr),
                                0, 4 * sizeof(int32_t)),
                     "cudaMemset interpreter counters");
  qwen36_interpreter_program_t interpreter_spec{};
  interpreter_spec.instructions = interpreter_instructions;
  interpreter_spec.instruction_count = 3;
  interpreter_spec.counters_i32 = interpreter_counters;
  interpreter_spec.counter_count = 4;
  interpreter_spec.cta_count = 1;
  must_status(qwen36_interpreter_decode_sm120(&interpreter_spec),
              "interpreter decode stage0");
  std::vector<int32_t> interpreter_counter_values =
      read_raw<int32_t>(interpreter_counters, 2);
  if (interpreter_counter_values[0] != 1 ||
      interpreter_counter_values[1] != 1) {
    fprintf(stderr, "interpreter counters expected [1, 1] got [%d, %d]\n",
            interpreter_counter_values[0], interpreter_counter_values[1]);
    exit(1);
  }

  qwen36_device_ptr_t residual_add_input = dev_alloc<__nv_bfloat16>(8);
  qwen36_device_ptr_t residual_add_residual = dev_alloc<__nv_bfloat16>(8);
  qwen36_device_ptr_t residual_add_output = dev_alloc<__nv_bfloat16>(8);
  qwen36_device_ptr_t residual_add_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(2);
  qwen36_device_ptr_t residual_add_counters = dev_alloc<int32_t>(2);
  copy_bf16(residual_add_input,
            {1.0f, 2.0f, -3.0f, 4.0f, 0.5f, -0.5f, 8.0f, -8.0f});
  copy_bf16(residual_add_residual,
            {0.5f, -2.0f, 3.5f, 1.0f, 0.25f, 0.75f, -4.0f, 4.0f});
  qwen36_interpreter_instruction_t residual_add_program_host[2]{};
  residual_add_program_host[0].opcode = 8; // RESIDUAL_ADD
  residual_add_program_host[0].publishes_counter = 0;
  residual_add_program_host[0].publish_value = 1;
  residual_add_program_host[0].arrival_counter = 1;
  residual_add_program_host[0].payload[0] = 8;
  residual_add_program_host[0].payload[1] = residual_add_input.ptr;
  residual_add_program_host[0].payload[2] = residual_add_residual.ptr;
  residual_add_program_host[0].payload[3] = residual_add_output.ptr;
  residual_add_program_host[1].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      residual_add_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          residual_add_program_host, residual_add_program_host + 2));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(residual_add_counters.ptr), 0,
                 2 * sizeof(int32_t)),
      "cudaMemset residual_add counters");
  qwen36_interpreter_program_t residual_add_spec{};
  residual_add_spec.instructions = residual_add_instructions;
  residual_add_spec.instruction_count = 2;
  residual_add_spec.counters_i32 = residual_add_counters;
  residual_add_spec.counter_count = 2;
  residual_add_spec.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&residual_add_spec),
              "interpreter residual_add");
  std::vector<float> residual_add_values =
      read_bf16(residual_add_output, 8);
  const float residual_add_expected[8] = {1.5f, 0.0f, 0.5f, 5.0f,
                                          0.75f, 0.25f, 4.0f, -4.0f};
  for (size_t i = 0; i < 8; ++i) {
    expect_close(residual_add_values[i], residual_add_expected[i], 0.02f,
                 "interpreter residual_add");
  }

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

  qwen36_device_ptr_t interp_attn_cache_k_ref =
      dev_alloc<__nv_bfloat16>(kv_values * 2);
  qwen36_device_ptr_t interp_attn_cache_v_ref =
      dev_alloc<__nv_bfloat16>(kv_values * 2);
  qwen36_device_ptr_t interp_attn_out_ref =
      dev_alloc<__nv_bfloat16>(q_values);
  qwen36_device_ptr_t interp_attn_cache_k =
      dev_alloc<__nv_bfloat16>(kv_values * 2);
  qwen36_device_ptr_t interp_attn_cache_v =
      dev_alloc<__nv_bfloat16>(kv_values * 2);
  qwen36_device_ptr_t interp_attn_out = dev_alloc<__nv_bfloat16>(q_values);
  qwen36_device_ptr_t interp_attn_spec_device =
      dev_alloc<qwen36_attention_decode_spec_t>(1);
  qwen36_device_ptr_t interp_attn_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(2);
  qwen36_device_ptr_t interp_attn_counters = dev_alloc<int32_t>(2);
  copy_bf16(interp_attn_cache_k_ref,
            std::vector<float>(kv_values * 2, 0.25f));
  copy_bf16(interp_attn_cache_v_ref,
            std::vector<float>(kv_values * 2, 0.75f));
  copy_bf16(interp_attn_cache_k, std::vector<float>(kv_values * 2, 0.25f));
  copy_bf16(interp_attn_cache_v, std::vector<float>(kv_values * 2, 0.75f));
  qwen36_attention_decode_spec_t interp_attn_spec_host{};
  interp_attn_spec_host.position = 1;
  interp_attn_spec_host.q_bf16 = q;
  interp_attn_spec_host.k_bf16 = k;
  interp_attn_spec_host.v_bf16 = v;
  interp_attn_spec_host.kv_cache_k = interp_attn_cache_k_ref;
  interp_attn_spec_host.kv_cache_v = interp_attn_cache_v_ref;
  interp_attn_spec_host.output_bf16 = interp_attn_out_ref;
  interp_attn_spec_host.shape = attn;
  must_status(qwen36_attention_decode(&interp_attn_spec_host),
              "interpreter attention reference");
  interp_attn_spec_host.kv_cache_k = interp_attn_cache_k;
  interp_attn_spec_host.kv_cache_v = interp_attn_cache_v;
  interp_attn_spec_host.output_bf16 = interp_attn_out;
  copy_raw<qwen36_attention_decode_spec_t>(interp_attn_spec_device,
                                           {interp_attn_spec_host});
  qwen36_interpreter_instruction_t interp_attn_program_host[2]{};
  interp_attn_program_host[0].opcode = 6; // ATTN_DECODE_FULL
  interp_attn_program_host[0].publishes_counter = 0;
  interp_attn_program_host[0].publish_value = 1;
  interp_attn_program_host[0].arrival_counter = 1;
  interp_attn_program_host[0].payload[0] = interp_attn_spec_device.ptr;
  interp_attn_program_host[1].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_attn_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_attn_program_host, interp_attn_program_host + 2));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_attn_counters.ptr), 0,
                 2 * sizeof(int32_t)),
      "cudaMemset interp attn counters");
  qwen36_interpreter_program_t interp_attn_program{};
  interp_attn_program.instructions = interp_attn_instructions;
  interp_attn_program.instruction_count = 2;
  interp_attn_program.counters_i32 = interp_attn_counters;
  interp_attn_program.counter_count = 2;
  interp_attn_program.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_attn_program),
              "interpreter attention decode");
  std::vector<float> interp_attn_ref_values =
      read_bf16(interp_attn_out_ref, q_values);
  std::vector<float> interp_attn_values = read_bf16(interp_attn_out, q_values);
  std::vector<float> interp_attn_cache_k_ref_values =
      read_bf16(interp_attn_cache_k_ref, kv_values * 2);
  std::vector<float> interp_attn_cache_v_ref_values =
      read_bf16(interp_attn_cache_v_ref, kv_values * 2);
  std::vector<float> interp_attn_cache_k_values =
      read_bf16(interp_attn_cache_k, kv_values * 2);
  std::vector<float> interp_attn_cache_v_values =
      read_bf16(interp_attn_cache_v, kv_values * 2);
  for (size_t i = 0; i < q_values; ++i) {
    expect_close(interp_attn_values[i], interp_attn_ref_values[i], 0.0f,
                 "interpreter attention output");
  }
  for (size_t i = 0; i < kv_values * 2; ++i) {
    expect_close(interp_attn_cache_k_values[i],
                 interp_attn_cache_k_ref_values[i], 0.0f,
                 "interpreter attention cache k");
    expect_close(interp_attn_cache_v_values[i],
                 interp_attn_cache_v_ref_values[i], 0.0f,
                 "interpreter attention cache v");
  }

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

  qwen36_device_ptr_t interp_delta_state = dev_alloc<__nv_bfloat16>(
      delta_shape.v_heads * delta_shape.value_dim * delta_shape.key_dim);
  qwen36_device_ptr_t interp_delta_out =
      dev_alloc<__nv_bfloat16>(delta_shape.v_heads * delta_shape.value_dim);
  qwen36_device_ptr_t interp_delta_spec_device =
      dev_alloc<qwen36_deltanet_decode_spec_t>(1);
  qwen36_device_ptr_t interp_delta_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(2);
  qwen36_device_ptr_t interp_delta_counters = dev_alloc<int32_t>(2);
  must_cuda<__nv_bfloat16>(
      cudaMemset(reinterpret_cast<void *>(interp_delta_state.ptr), 0,
                 delta_shape.v_heads * delta_shape.value_dim *
                     delta_shape.key_dim * sizeof(__nv_bfloat16)),
      "cudaMemset interpreter delta state");
  qwen36_deltanet_decode_spec_t interp_delta_spec_host = exact_delta_spec;
  interp_delta_spec_host.state_bf16 = interp_delta_state;
  interp_delta_spec_host.output_bf16 = interp_delta_out;
  copy_raw<qwen36_deltanet_decode_spec_t>(interp_delta_spec_device,
                                          {interp_delta_spec_host});
  qwen36_interpreter_instruction_t interp_delta_program_host[2]{};
  interp_delta_program_host[0].opcode = 7; // DELTANET_RECUR
  interp_delta_program_host[0].publishes_counter = 0;
  interp_delta_program_host[0].publish_value = 1;
  interp_delta_program_host[0].arrival_counter = 1;
  interp_delta_program_host[0].payload[0] = interp_delta_spec_device.ptr;
  interp_delta_program_host[1].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_delta_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_delta_program_host, interp_delta_program_host + 2));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_delta_counters.ptr), 0,
                 2 * sizeof(int32_t)),
      "cudaMemset interp deltanet counters");
  qwen36_interpreter_program_t interp_delta_program{};
  interp_delta_program.instructions = interp_delta_instructions;
  interp_delta_program.instruction_count = 2;
  interp_delta_program.counters_i32 = interp_delta_counters;
  interp_delta_program.counter_count = 2;
  interp_delta_program.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_delta_program),
              "interpreter deltanet recur");
  std::vector<float> interp_delta_values = read_bf16(interp_delta_out, 4);
  std::vector<float> interp_delta_state_values =
      read_bf16(interp_delta_state,
                delta_shape.v_heads * delta_shape.value_dim *
                    delta_shape.key_dim);
  std::vector<float> exact_delta_state_values =
      read_bf16(exact_state, delta_shape.v_heads * delta_shape.value_dim *
                                 delta_shape.key_dim);
  for (size_t i = 0; i < interp_delta_values.size(); ++i) {
    expect_close(interp_delta_values[i], exact_delta_values[i], 0.0f,
                 "interpreter deltanet output");
  }
  for (size_t i = 0; i < interp_delta_state_values.size(); ++i) {
    expect_close(interp_delta_state_values[i], exact_delta_state_values[i], 0.0f,
                 "interpreter deltanet state");
  }

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

  qwen36_device_ptr_t interp_norm_out = dev_alloc<__nv_bfloat16>(16);
  qwen36_device_ptr_t interp_norm_fp4 = dev_alloc<uint8_t>(8);
  qwen36_device_ptr_t interp_norm_scale = dev_alloc<uint8_t>(512);
  qwen36_device_ptr_t interp_norm_global = dev_alloc<float>(1);
  qwen36_device_ptr_t interp_norm_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(2);
  qwen36_device_ptr_t interp_norm_counters = dev_alloc<int32_t>(2);
  union F32Bits {
    float f;
    uint32_t u;
  };
  F32Bits eps_bits{1.0e-6f};
  F32Bits scale_bits{1.0f};
  qwen36_interpreter_instruction_t interp_norm_program_host[2]{};
  interp_norm_program_host[0].opcode = 2; // RMSNORM_NVFP4_QUANT
  interp_norm_program_host[0].publishes_counter = 0;
  interp_norm_program_host[0].publish_value = 1;
  interp_norm_program_host[0].arrival_counter = 1;
  interp_norm_program_host[0].payload[0] = 16;
  interp_norm_program_host[0].payload[1] = fused_norm_in.ptr;
  interp_norm_program_host[0].payload[2] = fused_norm_weight.ptr;
  interp_norm_program_host[0].payload[3] = 0;
  interp_norm_program_host[0].payload[4] = 0;
  interp_norm_program_host[0].payload[5] = interp_norm_out.ptr;
  interp_norm_program_host[0].payload[6] = interp_norm_fp4.ptr;
  interp_norm_program_host[0].payload[7] = interp_norm_scale.ptr;
  interp_norm_program_host[0].payload[8] = interp_norm_global.ptr;
  interp_norm_program_host[0].payload[9] =
      static_cast<uint64_t>(eps_bits.u) |
      (static_cast<uint64_t>(scale_bits.u) << 32);
  interp_norm_program_host[1].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_norm_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_norm_program_host, interp_norm_program_host + 2));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_norm_counters.ptr), 0,
                 2 * sizeof(int32_t)),
      "cudaMemset interp rmsnorm counters");
  qwen36_interpreter_program_t interp_norm_spec{};
  interp_norm_spec.instructions = interp_norm_instructions;
  interp_norm_spec.instruction_count = 2;
  interp_norm_spec.counters_i32 = interp_norm_counters;
  interp_norm_spec.counter_count = 2;
  interp_norm_spec.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_norm_spec),
              "interpreter rmsnorm nvfp4 quantize");
  std::vector<float> interp_norm_values = read_bf16(interp_norm_out, 16);
  std::vector<uint8_t> interp_norm_packed =
      read_raw<uint8_t>(interp_norm_fp4, 8);
  const uint8_t ref_scale0 = read_one<uint8_t>(fused_norm_scale);
  const uint8_t interp_scale0 = read_one<uint8_t>(interp_norm_scale);
  for (size_t i = 0; i < 16; ++i) {
    expect_close(interp_norm_values[i], fused_norm_values[i], 0.0f,
                 "interpreter rmsnorm bf16");
  }
  for (size_t i = 0; i < 8; ++i) {
    if (interp_norm_packed[i] != fused_norm_packed[i]) {
      fprintf(stderr, "interpreter rmsnorm fp4 byte %zu expected 0x%02x got 0x%02x\n",
              i, fused_norm_packed[i], interp_norm_packed[i]);
      exit(1);
    }
  }
  if (interp_scale0 != ref_scale0) {
    fprintf(stderr, "interpreter rmsnorm scale expected 0x%02x got 0x%02x\n",
            ref_scale0, interp_scale0);
    exit(1);
  }
  expect_close(read_one<float>(interp_norm_global), 1.0f, 0.0f,
               "interpreter rmsnorm tensor scale");

  const size_t interp_rms_bf16_rows = 2;
  const size_t interp_rms_bf16_hidden = 8;
  const size_t interp_rms_bf16_value_count =
      interp_rms_bf16_rows * interp_rms_bf16_hidden;
  qwen36_device_ptr_t interp_rms_bf16_input =
      dev_alloc<__nv_bfloat16>(interp_rms_bf16_value_count);
  qwen36_device_ptr_t interp_rms_bf16_weight =
      dev_alloc<__nv_bfloat16>(interp_rms_bf16_hidden);
  qwen36_device_ptr_t interp_rms_bf16_residual =
      dev_alloc<__nv_bfloat16>(interp_rms_bf16_value_count);
  qwen36_device_ptr_t interp_rms_bf16_residual_ref =
      dev_alloc<__nv_bfloat16>(interp_rms_bf16_value_count);
  qwen36_device_ptr_t interp_rms_bf16_residual_out =
      dev_alloc<__nv_bfloat16>(interp_rms_bf16_value_count);
  qwen36_device_ptr_t interp_rms_bf16_out_ref =
      dev_alloc<__nv_bfloat16>(interp_rms_bf16_value_count);
  qwen36_device_ptr_t interp_rms_bf16_out =
      dev_alloc<__nv_bfloat16>(interp_rms_bf16_value_count);
  qwen36_device_ptr_t interp_rms_bf16_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(2);
  qwen36_device_ptr_t interp_rms_bf16_counters = dev_alloc<int32_t>(2);
  std::vector<float> interp_rms_bf16_input_values(interp_rms_bf16_value_count);
  std::vector<float> interp_rms_bf16_residual_values(
      interp_rms_bf16_value_count);
  for (size_t i = 0; i < interp_rms_bf16_value_count; ++i) {
    interp_rms_bf16_input_values[i] = (static_cast<int>(i % 11) - 5) * 0.125f;
    interp_rms_bf16_residual_values[i] =
        (static_cast<int>(i % 7) - 3) * 0.0625f;
  }
  copy_bf16(interp_rms_bf16_input, interp_rms_bf16_input_values);
  copy_bf16(interp_rms_bf16_residual, interp_rms_bf16_residual_values);
  copy_bf16(interp_rms_bf16_weight,
            {-0.125f, 0.0f, 0.125f, 0.25f, -0.25f, 0.5f, -0.5f, 0.75f});
  qwen36_rmsnorm_spec_t interp_rms_bf16_ref_spec{};
  interp_rms_bf16_ref_spec.rows = interp_rms_bf16_rows;
  interp_rms_bf16_ref_spec.hidden = interp_rms_bf16_hidden;
  interp_rms_bf16_ref_spec.eps = 1.0e-6f;
  interp_rms_bf16_ref_spec.input_bf16 = interp_rms_bf16_input;
  interp_rms_bf16_ref_spec.weight_bf16 = interp_rms_bf16_weight;
  interp_rms_bf16_ref_spec.residual_bf16 = interp_rms_bf16_residual;
  interp_rms_bf16_ref_spec.residual_out_bf16 = interp_rms_bf16_residual_ref;
  interp_rms_bf16_ref_spec.output_bf16 = interp_rms_bf16_out_ref;
  interp_rms_bf16_ref_spec.direct_weight = 0;
  must_status(qwen36_rmsnorm(&interp_rms_bf16_ref_spec),
              "interpreter rmsnorm bf16 reference");
  qwen36_interpreter_instruction_t interp_rms_bf16_program_host[2]{};
  interp_rms_bf16_program_host[0].opcode = 10; // RMSNORM_BF16
  interp_rms_bf16_program_host[0].publishes_counter = 0;
  interp_rms_bf16_program_host[0].publish_value = 1;
  interp_rms_bf16_program_host[0].arrival_counter = 1;
  interp_rms_bf16_program_host[0].payload[0] = interp_rms_bf16_rows;
  interp_rms_bf16_program_host[0].payload[1] = interp_rms_bf16_hidden;
  interp_rms_bf16_program_host[0].payload[2] = interp_rms_bf16_input.ptr;
  interp_rms_bf16_program_host[0].payload[3] = interp_rms_bf16_weight.ptr;
  interp_rms_bf16_program_host[0].payload[4] = interp_rms_bf16_residual.ptr;
  interp_rms_bf16_program_host[0].payload[5] = interp_rms_bf16_residual_out.ptr;
  interp_rms_bf16_program_host[0].payload[6] = interp_rms_bf16_out.ptr;
  interp_rms_bf16_program_host[0].payload[7] = eps_bits.u;
  interp_rms_bf16_program_host[1].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_rms_bf16_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_rms_bf16_program_host, interp_rms_bf16_program_host + 2));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_rms_bf16_counters.ptr), 0,
                 2 * sizeof(int32_t)),
      "cudaMemset interp rmsnorm bf16 counters");
  qwen36_interpreter_program_t interp_rms_bf16_spec{};
  interp_rms_bf16_spec.instructions = interp_rms_bf16_instructions;
  interp_rms_bf16_spec.instruction_count = 2;
  interp_rms_bf16_spec.counters_i32 = interp_rms_bf16_counters;
  interp_rms_bf16_spec.counter_count = 2;
  interp_rms_bf16_spec.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_rms_bf16_spec),
              "interpreter rmsnorm bf16");
  std::vector<float> interp_rms_bf16_ref_values =
      read_bf16(interp_rms_bf16_out_ref, interp_rms_bf16_value_count);
  std::vector<float> interp_rms_bf16_values =
      read_bf16(interp_rms_bf16_out, interp_rms_bf16_value_count);
  std::vector<float> interp_rms_bf16_residual_ref_values =
      read_bf16(interp_rms_bf16_residual_ref, interp_rms_bf16_value_count);
  std::vector<float> interp_rms_bf16_residual_out_values =
      read_bf16(interp_rms_bf16_residual_out, interp_rms_bf16_value_count);
  for (size_t i = 0; i < interp_rms_bf16_values.size(); ++i) {
    expect_close(interp_rms_bf16_values[i], interp_rms_bf16_ref_values[i],
                 0.0f, "interpreter rmsnorm bf16 out");
    expect_close(interp_rms_bf16_residual_out_values[i],
                 interp_rms_bf16_residual_ref_values[i], 0.0f,
                 "interpreter rmsnorm bf16 residual");
  }

  const size_t interp_quant_values = 33;
  const size_t interp_quant_fp4_bytes = (interp_quant_values + 1) / 2;
  qwen36_device_ptr_t interp_quant_input =
      dev_alloc<__nv_bfloat16>(interp_quant_values);
  qwen36_device_ptr_t interp_quant_fp4_ref =
      dev_alloc<uint8_t>(interp_quant_fp4_bytes);
  qwen36_device_ptr_t interp_quant_scale_ref = dev_alloc<uint8_t>(512);
  qwen36_device_ptr_t interp_quant_global_ref = dev_alloc<float>(1);
  qwen36_device_ptr_t interp_quant_fp4 =
      dev_alloc<uint8_t>(interp_quant_fp4_bytes);
  qwen36_device_ptr_t interp_quant_scale = dev_alloc<uint8_t>(512);
  qwen36_device_ptr_t interp_quant_global = dev_alloc<float>(1);
  qwen36_device_ptr_t interp_quant_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(2);
  qwen36_device_ptr_t interp_quant_counters = dev_alloc<int32_t>(2);
  std::vector<float> interp_quant_input_values(interp_quant_values);
  for (size_t i = 0; i < interp_quant_values; ++i) {
    interp_quant_input_values[i] = (static_cast<int>(i % 13) - 6) * 0.1875f;
  }
  copy_bf16(interp_quant_input, interp_quant_input_values);
  must_cuda<uint8_t>(
      cudaMemset(reinterpret_cast<void *>(interp_quant_fp4_ref.ptr), 0,
                 interp_quant_fp4_bytes),
      "cudaMemset nvfp4 quant fp4 ref");
  must_cuda<uint8_t>(
      cudaMemset(reinterpret_cast<void *>(interp_quant_scale_ref.ptr), 0, 512),
      "cudaMemset nvfp4 quant scale ref");
  must_cuda<uint8_t>(
      cudaMemset(reinterpret_cast<void *>(interp_quant_fp4.ptr), 0,
                 interp_quant_fp4_bytes),
      "cudaMemset nvfp4 quant fp4");
  must_cuda<uint8_t>(
      cudaMemset(reinterpret_cast<void *>(interp_quant_scale.ptr), 0, 512),
      "cudaMemset nvfp4 quant scale");
  qwen36_nvfp4_quantize_spec_t interp_quant_ref_spec{};
  interp_quant_ref_spec.values = interp_quant_values;
  interp_quant_ref_spec.input_bf16 = interp_quant_input;
  interp_quant_ref_spec.output_fp4 = interp_quant_fp4_ref;
  interp_quant_ref_spec.output_scale_e4m3 = interp_quant_scale_ref;
  interp_quant_ref_spec.output_tensor_scale_f32 = interp_quant_global_ref;
  interp_quant_ref_spec.input_tensor_scale_f32 = 1.25f;
  must_status(qwen36_nvfp4_quantize_bf16(&interp_quant_ref_spec),
              "interpreter nvfp4 quantize reference");
  F32Bits interp_quant_scale_bits{1.25f};
  qwen36_interpreter_instruction_t interp_quant_program_host[2]{};
  interp_quant_program_host[0].opcode = 13; // NVFP4_QUANTIZE
  interp_quant_program_host[0].publishes_counter = 0;
  interp_quant_program_host[0].publish_value = 1;
  interp_quant_program_host[0].arrival_counter = 1;
  interp_quant_program_host[0].payload[0] = interp_quant_values;
  interp_quant_program_host[0].payload[1] = interp_quant_input.ptr;
  interp_quant_program_host[0].payload[2] = interp_quant_fp4.ptr;
  interp_quant_program_host[0].payload[3] = interp_quant_scale.ptr;
  interp_quant_program_host[0].payload[4] = interp_quant_global.ptr;
  interp_quant_program_host[0].payload[5] = interp_quant_scale_bits.u;
  interp_quant_program_host[1].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_quant_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_quant_program_host, interp_quant_program_host + 2));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_quant_counters.ptr), 0,
                 2 * sizeof(int32_t)),
      "cudaMemset interp nvfp4 quant counters");
  qwen36_interpreter_program_t interp_quant_spec{};
  interp_quant_spec.instructions = interp_quant_instructions;
  interp_quant_spec.instruction_count = 2;
  interp_quant_spec.counters_i32 = interp_quant_counters;
  interp_quant_spec.counter_count = 2;
  interp_quant_spec.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_quant_spec),
              "interpreter nvfp4 quantize");
  std::vector<uint8_t> interp_quant_fp4_ref_values =
      read_raw<uint8_t>(interp_quant_fp4_ref, interp_quant_fp4_bytes);
  std::vector<uint8_t> interp_quant_fp4_values =
      read_raw<uint8_t>(interp_quant_fp4, interp_quant_fp4_bytes);
  std::vector<uint8_t> interp_quant_scale_ref_values =
      read_raw<uint8_t>(interp_quant_scale_ref, 512);
  std::vector<uint8_t> interp_quant_scale_values =
      read_raw<uint8_t>(interp_quant_scale, 512);
  for (size_t i = 0; i < interp_quant_fp4_bytes; ++i) {
    if (interp_quant_fp4_values[i] != interp_quant_fp4_ref_values[i]) {
      fprintf(stderr, "interpreter nvfp4 quant fp4 byte %zu expected 0x%02x got 0x%02x\n",
              i, interp_quant_fp4_ref_values[i], interp_quant_fp4_values[i]);
      exit(1);
    }
  }
  for (size_t i = 0; i < 512; ++i) {
    if (interp_quant_scale_values[i] != interp_quant_scale_ref_values[i]) {
      fprintf(stderr, "interpreter nvfp4 quant scale byte %zu expected 0x%02x got 0x%02x\n",
              i, interp_quant_scale_ref_values[i],
              interp_quant_scale_values[i]);
      exit(1);
    }
  }
  expect_close(read_one<float>(interp_quant_global),
               read_one<float>(interp_quant_global_ref), 0.0f,
               "interpreter nvfp4 quant tensor scale");

  const size_t interp_qproj_rows = 2;
  const size_t interp_qproj_heads = 2;
  const size_t interp_qproj_head_dim = 4;
  const size_t interp_qproj_q_values =
      interp_qproj_heads * interp_qproj_head_dim;
  const size_t interp_qproj_raw_values =
      interp_qproj_rows * interp_qproj_q_values * 2;
  qwen36_device_ptr_t interp_qproj_raw =
      dev_alloc<__nv_bfloat16>(interp_qproj_raw_values);
  qwen36_device_ptr_t interp_qproj_deint_ref =
      dev_alloc<__nv_bfloat16>(interp_qproj_rows * interp_qproj_q_values);
  qwen36_device_ptr_t interp_qproj_deint =
      dev_alloc<__nv_bfloat16>(interp_qproj_rows * interp_qproj_q_values);
  qwen36_device_ptr_t interp_qproj_gate_ref =
      dev_alloc<__nv_bfloat16>(interp_qproj_rows * interp_qproj_q_values);
  qwen36_device_ptr_t interp_qproj_gate =
      dev_alloc<__nv_bfloat16>(interp_qproj_rows * interp_qproj_q_values);
  qwen36_device_ptr_t interp_qproj_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(3);
  qwen36_device_ptr_t interp_qproj_counters = dev_alloc<int32_t>(4);
  std::vector<float> interp_qproj_raw_values_host(interp_qproj_raw_values);
  for (size_t i = 0; i < interp_qproj_raw_values; ++i) {
    interp_qproj_raw_values_host[i] =
        (static_cast<int>(i % 17) - 8) * 0.125f;
  }
  copy_bf16(interp_qproj_raw, interp_qproj_raw_values_host);
  qwen36_q_proj_deinterleave_spec_t interp_qproj_deint_ref_spec{};
  interp_qproj_deint_ref_spec.rows = interp_qproj_rows;
  interp_qproj_deint_ref_spec.heads = interp_qproj_heads;
  interp_qproj_deint_ref_spec.head_dim = interp_qproj_head_dim;
  interp_qproj_deint_ref_spec.input_bf16 = interp_qproj_raw;
  interp_qproj_deint_ref_spec.output_bf16 = interp_qproj_deint_ref;
  must_status(qwen36_q_proj_deinterleave(&interp_qproj_deint_ref_spec),
              "interpreter q_proj deinterleave reference");
  qwen36_q_proj_sigmoid_gate_spec_t interp_qproj_gate_ref_spec{};
  interp_qproj_gate_ref_spec.rows = interp_qproj_rows;
  interp_qproj_gate_ref_spec.heads = interp_qproj_heads;
  interp_qproj_gate_ref_spec.head_dim = interp_qproj_head_dim;
  interp_qproj_gate_ref_spec.gate_bf16 = interp_qproj_raw;
  interp_qproj_gate_ref_spec.input_bf16 = interp_qproj_deint_ref;
  interp_qproj_gate_ref_spec.output_bf16 = interp_qproj_gate_ref;
  must_status(qwen36_q_proj_sigmoid_gate(&interp_qproj_gate_ref_spec),
              "interpreter q_proj sigmoid gate reference");
  qwen36_interpreter_instruction_t interp_qproj_program_host[3]{};
  interp_qproj_program_host[0].opcode = 11; // Q_PROJ_DEINTERLEAVE
  interp_qproj_program_host[0].publishes_counter = 0;
  interp_qproj_program_host[0].publish_value = 1;
  interp_qproj_program_host[0].arrival_counter = 1;
  interp_qproj_program_host[0].payload[0] = interp_qproj_rows;
  interp_qproj_program_host[0].payload[1] = interp_qproj_heads;
  interp_qproj_program_host[0].payload[2] = interp_qproj_head_dim;
  interp_qproj_program_host[0].payload[3] = interp_qproj_raw.ptr;
  interp_qproj_program_host[0].payload[4] = interp_qproj_deint.ptr;
  interp_qproj_program_host[1].opcode = 12; // Q_PROJ_SIGMOID_GATE
  interp_qproj_program_host[1].dep_count = 1;
  interp_qproj_program_host[1].deps[0] = qwen36_interpreter_dep_t{0, 1};
  interp_qproj_program_host[1].publishes_counter = 2;
  interp_qproj_program_host[1].publish_value = 1;
  interp_qproj_program_host[1].arrival_counter = 3;
  interp_qproj_program_host[1].payload[0] = interp_qproj_rows;
  interp_qproj_program_host[1].payload[1] = interp_qproj_heads;
  interp_qproj_program_host[1].payload[2] = interp_qproj_head_dim;
  interp_qproj_program_host[1].payload[3] = interp_qproj_raw.ptr;
  interp_qproj_program_host[1].payload[4] = interp_qproj_deint.ptr;
  interp_qproj_program_host[1].payload[5] = interp_qproj_gate.ptr;
  interp_qproj_program_host[2].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_qproj_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_qproj_program_host, interp_qproj_program_host + 3));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_qproj_counters.ptr), 0,
                 4 * sizeof(int32_t)),
      "cudaMemset interp q_proj counters");
  qwen36_interpreter_program_t interp_qproj_spec{};
  interp_qproj_spec.instructions = interp_qproj_instructions;
  interp_qproj_spec.instruction_count = 3;
  interp_qproj_spec.counters_i32 = interp_qproj_counters;
  interp_qproj_spec.counter_count = 4;
  interp_qproj_spec.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_qproj_spec),
              "interpreter q_proj deinterleave+gate");
  std::vector<float> interp_qproj_deint_ref_values =
      read_bf16(interp_qproj_deint_ref,
                interp_qproj_rows * interp_qproj_q_values);
  std::vector<float> interp_qproj_deint_values =
      read_bf16(interp_qproj_deint,
                interp_qproj_rows * interp_qproj_q_values);
  std::vector<float> interp_qproj_gate_ref_values =
      read_bf16(interp_qproj_gate_ref,
                interp_qproj_rows * interp_qproj_q_values);
  std::vector<float> interp_qproj_gate_values =
      read_bf16(interp_qproj_gate,
                interp_qproj_rows * interp_qproj_q_values);
  for (size_t i = 0; i < interp_qproj_rows * interp_qproj_q_values; ++i) {
    expect_close(interp_qproj_deint_values[i], interp_qproj_deint_ref_values[i],
                 0.0f, "interpreter q_proj deinterleave");
    expect_close(interp_qproj_gate_values[i], interp_qproj_gate_ref_values[i],
                 0.0f, "interpreter q_proj sigmoid gate");
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

  qwen36_device_ptr_t interp_rope_q_ref = dev_alloc<__nv_bfloat16>(12);
  qwen36_device_ptr_t interp_rope_k_ref = dev_alloc<__nv_bfloat16>(12);
  qwen36_device_ptr_t interp_rope_q = dev_alloc<__nv_bfloat16>(12);
  qwen36_device_ptr_t interp_rope_k = dev_alloc<__nv_bfloat16>(12);
  qwen36_device_ptr_t interp_rope_pos = dev_alloc<int32_t>(2);
  qwen36_device_ptr_t interp_rope_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(2);
  qwen36_device_ptr_t interp_rope_counters = dev_alloc<int32_t>(2);
  const std::vector<float> rope_q_init = {1.0f, 2.0f, 0.0f, 0.0f,
                                          9.0f, 9.0f, 0.5f, -1.0f,
                                          2.0f, 3.0f, 7.0f, 8.0f};
  const std::vector<float> rope_k_init = {0.0f, 0.0f, 1.0f, 2.0f,
                                          8.0f, 8.0f, -2.0f, 4.0f,
                                          0.25f, 0.75f, 6.0f, 5.0f};
  copy_raw<int32_t>(interp_rope_pos, {1, 3});
  copy_bf16(interp_rope_q_ref, rope_q_init);
  copy_bf16(interp_rope_k_ref, rope_k_init);
  copy_bf16(interp_rope_q, rope_q_init);
  copy_bf16(interp_rope_k, rope_k_init);
  qwen36_partial_rope_spec_t interp_rope_ref_spec{};
  interp_rope_ref_spec.tokens = 2;
  interp_rope_ref_spec.q_heads = 1;
  interp_rope_ref_spec.kv_heads = 1;
  interp_rope_ref_spec.head_dim = 6;
  interp_rope_ref_spec.rope_dims = 4;
  interp_rope_ref_spec.base_theta = 10000.0;
  interp_rope_ref_spec.positions_i32 = interp_rope_pos;
  interp_rope_ref_spec.q_bf16 = interp_rope_q_ref;
  interp_rope_ref_spec.k_bf16 = interp_rope_k_ref;
  must_status(qwen36_partial_rope(&interp_rope_ref_spec),
              "interpreter rope reference");
  F32Bits rope_base_bits{10000.0f};
  qwen36_interpreter_instruction_t interp_rope_program_host[2]{};
  interp_rope_program_host[0].opcode = 5; // ROPE_PARTIAL
  interp_rope_program_host[0].publishes_counter = 0;
  interp_rope_program_host[0].publish_value = 1;
  interp_rope_program_host[0].arrival_counter = 1;
  interp_rope_program_host[0].payload[0] = 2;
  interp_rope_program_host[0].payload[1] = 1;
  interp_rope_program_host[0].payload[2] = 1;
  interp_rope_program_host[0].payload[3] = 6;
  interp_rope_program_host[0].payload[4] = 4;
  interp_rope_program_host[0].payload[5] = rope_base_bits.u;
  interp_rope_program_host[0].payload[6] = 0;
  interp_rope_program_host[0].payload[7] = interp_rope_pos.ptr;
  interp_rope_program_host[0].payload[8] = interp_rope_q.ptr;
  interp_rope_program_host[0].payload[9] = interp_rope_k.ptr;
  interp_rope_program_host[0].payload[10] = 0;
  interp_rope_program_host[1].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_rope_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_rope_program_host, interp_rope_program_host + 2));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_rope_counters.ptr), 0,
                 2 * sizeof(int32_t)),
      "cudaMemset interp rope counters");
  qwen36_interpreter_program_t interp_rope_spec{};
  interp_rope_spec.instructions = interp_rope_instructions;
  interp_rope_spec.instruction_count = 2;
  interp_rope_spec.counters_i32 = interp_rope_counters;
  interp_rope_spec.counter_count = 2;
  interp_rope_spec.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_rope_spec),
              "interpreter rope partial");
  std::vector<float> interp_rope_q_ref_values =
      read_bf16(interp_rope_q_ref, 12);
  std::vector<float> interp_rope_k_ref_values =
      read_bf16(interp_rope_k_ref, 12);
  std::vector<float> interp_rope_q_values = read_bf16(interp_rope_q, 12);
  std::vector<float> interp_rope_k_values = read_bf16(interp_rope_k, 12);
  for (size_t i = 0; i < 12; ++i) {
    expect_close(interp_rope_q_values[i], interp_rope_q_ref_values[i], 0.0f,
                 "interpreter rope q");
    expect_close(interp_rope_k_values[i], interp_rope_k_ref_values[i], 0.0f,
                 "interpreter rope k");
  }

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

  qwen36_device_ptr_t interp_swiglu_bf16_gate =
      dev_alloc<__nv_bfloat16>(17);
  qwen36_device_ptr_t interp_swiglu_bf16_up =
      dev_alloc<__nv_bfloat16>(17);
  qwen36_device_ptr_t interp_swiglu_bf16_ref =
      dev_alloc<__nv_bfloat16>(17);
  qwen36_device_ptr_t interp_swiglu_bf16_out =
      dev_alloc<__nv_bfloat16>(17);
  qwen36_device_ptr_t interp_swiglu_bf16_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(2);
  qwen36_device_ptr_t interp_swiglu_bf16_counters = dev_alloc<int32_t>(2);
  std::vector<float> interp_swiglu_bf16_gate_values(17);
  std::vector<float> interp_swiglu_bf16_up_values(17);
  for (size_t i = 0; i < 17; ++i) {
    interp_swiglu_bf16_gate_values[i] =
        (static_cast<int>(i % 9) - 4) * 0.25f;
    interp_swiglu_bf16_up_values[i] = 0.5f + static_cast<float>(i) * 0.03125f;
  }
  copy_bf16(interp_swiglu_bf16_gate, interp_swiglu_bf16_gate_values);
  copy_bf16(interp_swiglu_bf16_up, interp_swiglu_bf16_up_values);
  qwen36_swiglu_spec_t interp_swiglu_bf16_ref_spec{};
  interp_swiglu_bf16_ref_spec.rows = 1;
  interp_swiglu_bf16_ref_spec.intermediate = 17;
  interp_swiglu_bf16_ref_spec.gate_bf16 = interp_swiglu_bf16_gate;
  interp_swiglu_bf16_ref_spec.up_bf16 = interp_swiglu_bf16_up;
  interp_swiglu_bf16_ref_spec.output_bf16 = interp_swiglu_bf16_ref;
  must_status(qwen36_swiglu(&interp_swiglu_bf16_ref_spec),
              "interpreter swiglu bf16 reference");
  qwen36_interpreter_instruction_t interp_swiglu_bf16_program_host[2]{};
  interp_swiglu_bf16_program_host[0].opcode = 14; // SWIGLU_BF16
  interp_swiglu_bf16_program_host[0].publishes_counter = 0;
  interp_swiglu_bf16_program_host[0].publish_value = 1;
  interp_swiglu_bf16_program_host[0].arrival_counter = 1;
  interp_swiglu_bf16_program_host[0].payload[0] = 1;
  interp_swiglu_bf16_program_host[0].payload[1] = 17;
  interp_swiglu_bf16_program_host[0].payload[2] =
      interp_swiglu_bf16_gate.ptr;
  interp_swiglu_bf16_program_host[0].payload[3] = interp_swiglu_bf16_up.ptr;
  interp_swiglu_bf16_program_host[0].payload[4] = interp_swiglu_bf16_out.ptr;
  interp_swiglu_bf16_program_host[1].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_swiglu_bf16_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_swiglu_bf16_program_host,
          interp_swiglu_bf16_program_host + 2));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_swiglu_bf16_counters.ptr),
                 0, 2 * sizeof(int32_t)),
      "cudaMemset interp swiglu bf16 counters");
  qwen36_interpreter_program_t interp_swiglu_bf16_spec{};
  interp_swiglu_bf16_spec.instructions = interp_swiglu_bf16_instructions;
  interp_swiglu_bf16_spec.instruction_count = 2;
  interp_swiglu_bf16_spec.counters_i32 = interp_swiglu_bf16_counters;
  interp_swiglu_bf16_spec.counter_count = 2;
  interp_swiglu_bf16_spec.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_swiglu_bf16_spec),
              "interpreter swiglu bf16");
  std::vector<float> interp_swiglu_bf16_ref_values =
      read_bf16(interp_swiglu_bf16_ref, 17);
  std::vector<float> interp_swiglu_bf16_values =
      read_bf16(interp_swiglu_bf16_out, 17);
  for (size_t i = 0; i < 17; ++i) {
    expect_close(interp_swiglu_bf16_values[i],
                 interp_swiglu_bf16_ref_values[i], 0.0f,
                 "interpreter swiglu bf16");
  }

  const size_t interp_swiglu_values_count = 33;
  const size_t interp_swiglu_fp4_bytes = (interp_swiglu_values_count + 1) / 2;
  qwen36_device_ptr_t interp_swiglu_gate =
      dev_alloc<__nv_bfloat16>(interp_swiglu_values_count);
  qwen36_device_ptr_t interp_swiglu_up =
      dev_alloc<__nv_bfloat16>(interp_swiglu_values_count);
  qwen36_device_ptr_t interp_swiglu_fp4_ref =
      dev_alloc<uint8_t>(interp_swiglu_fp4_bytes);
  qwen36_device_ptr_t interp_swiglu_scale_ref = dev_alloc<uint8_t>(512);
  qwen36_device_ptr_t interp_swiglu_global_ref = dev_alloc<float>(1);
  qwen36_device_ptr_t interp_swiglu_fp4 =
      dev_alloc<uint8_t>(interp_swiglu_fp4_bytes);
  qwen36_device_ptr_t interp_swiglu_scale = dev_alloc<uint8_t>(512);
  qwen36_device_ptr_t interp_swiglu_global = dev_alloc<float>(1);
  qwen36_device_ptr_t interp_swiglu_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(2);
  qwen36_device_ptr_t interp_swiglu_counters = dev_alloc<int32_t>(2);
  std::vector<float> interp_swiglu_gate_values(interp_swiglu_values_count);
  std::vector<float> interp_swiglu_up_values(interp_swiglu_values_count);
  for (size_t i = 0; i < interp_swiglu_values_count; ++i) {
    interp_swiglu_gate_values[i] = (static_cast<int>(i % 9) - 4) * 0.25f;
    interp_swiglu_up_values[i] = 0.5f + static_cast<float>(i % 7) * 0.125f;
  }
  copy_bf16(interp_swiglu_gate, interp_swiglu_gate_values);
  copy_bf16(interp_swiglu_up, interp_swiglu_up_values);
  must_cuda<uint8_t>(
      cudaMemset(reinterpret_cast<void *>(interp_swiglu_fp4_ref.ptr), 0,
                 interp_swiglu_fp4_bytes),
      "cudaMemset swiglu fp4 ref");
  must_cuda<uint8_t>(
      cudaMemset(reinterpret_cast<void *>(interp_swiglu_scale_ref.ptr), 0, 512),
      "cudaMemset swiglu scale ref");
  must_cuda<uint8_t>(
      cudaMemset(reinterpret_cast<void *>(interp_swiglu_fp4.ptr), 0,
                 interp_swiglu_fp4_bytes),
      "cudaMemset swiglu fp4");
  must_cuda<uint8_t>(
      cudaMemset(reinterpret_cast<void *>(interp_swiglu_scale.ptr), 0, 512),
      "cudaMemset swiglu scale");
  qwen36_swiglu_nvfp4_quantize_spec_t interp_swiglu_ref_spec{};
  interp_swiglu_ref_spec.intermediate = interp_swiglu_values_count;
  interp_swiglu_ref_spec.gate_bf16 = interp_swiglu_gate;
  interp_swiglu_ref_spec.up_bf16 = interp_swiglu_up;
  interp_swiglu_ref_spec.output_fp4 = interp_swiglu_fp4_ref;
  interp_swiglu_ref_spec.output_scale_e4m3 = interp_swiglu_scale_ref;
  interp_swiglu_ref_spec.output_tensor_scale_f32 = interp_swiglu_global_ref;
  interp_swiglu_ref_spec.input_tensor_scale_f32 = 1.0f;
  must_status(qwen36_swiglu_nvfp4_quantize(&interp_swiglu_ref_spec),
              "interpreter swiglu reference");
  F32Bits swiglu_scale_bits{1.0f};
  qwen36_interpreter_instruction_t interp_swiglu_program_host[2]{};
  interp_swiglu_program_host[0].opcode = 4; // SWIGLU_NVFP4_QUANT
  interp_swiglu_program_host[0].publishes_counter = 0;
  interp_swiglu_program_host[0].publish_value = 1;
  interp_swiglu_program_host[0].arrival_counter = 1;
  interp_swiglu_program_host[0].payload[0] = interp_swiglu_values_count;
  interp_swiglu_program_host[0].payload[1] = interp_swiglu_gate.ptr;
  interp_swiglu_program_host[0].payload[2] = interp_swiglu_up.ptr;
  interp_swiglu_program_host[0].payload[3] = interp_swiglu_fp4.ptr;
  interp_swiglu_program_host[0].payload[4] = interp_swiglu_scale.ptr;
  interp_swiglu_program_host[0].payload[5] = interp_swiglu_global.ptr;
  interp_swiglu_program_host[0].payload[6] = swiglu_scale_bits.u;
  interp_swiglu_program_host[1].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_swiglu_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_swiglu_program_host, interp_swiglu_program_host + 2));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_swiglu_counters.ptr), 0,
                 2 * sizeof(int32_t)),
      "cudaMemset interp swiglu counters");
  qwen36_interpreter_program_t interp_swiglu_spec{};
  interp_swiglu_spec.instructions = interp_swiglu_instructions;
  interp_swiglu_spec.instruction_count = 2;
  interp_swiglu_spec.counters_i32 = interp_swiglu_counters;
  interp_swiglu_spec.counter_count = 2;
  interp_swiglu_spec.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_swiglu_spec),
              "interpreter swiglu nvfp4 quantize");
  std::vector<uint8_t> interp_swiglu_fp4_ref_values =
      read_raw<uint8_t>(interp_swiglu_fp4_ref, interp_swiglu_fp4_bytes);
  std::vector<uint8_t> interp_swiglu_fp4_values =
      read_raw<uint8_t>(interp_swiglu_fp4, interp_swiglu_fp4_bytes);
  std::vector<uint8_t> interp_swiglu_scale_ref_values =
      read_raw<uint8_t>(interp_swiglu_scale_ref, 512);
  std::vector<uint8_t> interp_swiglu_scale_values =
      read_raw<uint8_t>(interp_swiglu_scale, 512);
  for (size_t i = 0; i < interp_swiglu_fp4_bytes; ++i) {
    if (interp_swiglu_fp4_values[i] != interp_swiglu_fp4_ref_values[i]) {
      fprintf(stderr, "interpreter swiglu fp4 byte %zu expected 0x%02x got 0x%02x\n",
              i, interp_swiglu_fp4_ref_values[i], interp_swiglu_fp4_values[i]);
      exit(1);
    }
  }
  for (size_t i = 0; i < 512; ++i) {
    if (interp_swiglu_scale_values[i] != interp_swiglu_scale_ref_values[i]) {
      fprintf(stderr, "interpreter swiglu scale byte %zu expected 0x%02x got 0x%02x\n",
              i, interp_swiglu_scale_ref_values[i],
              interp_swiglu_scale_values[i]);
      exit(1);
    }
  }
  expect_close(read_one<float>(interp_swiglu_global),
               read_one<float>(interp_swiglu_global_ref), 0.0f,
               "interpreter swiglu tensor scale");

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

  qwen36_device_ptr_t interp_lm_out = dev_alloc<__nv_bfloat16>(2);
  qwen36_device_ptr_t interp_lm_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(2);
  qwen36_device_ptr_t interp_lm_counters = dev_alloc<int32_t>(2);
  qwen36_interpreter_instruction_t interp_lm_program_host[2]{};
  interp_lm_program_host[0].opcode = 9; // LM_HEAD_TILED
  interp_lm_program_host[0].publishes_counter = 0;
  interp_lm_program_host[0].publish_value = 1;
  interp_lm_program_host[0].arrival_counter = 1;
  interp_lm_program_host[0].payload[0] = 2;
  interp_lm_program_host[0].payload[1] = 4;
  interp_lm_program_host[0].payload[2] = matvec_input.ptr;
  interp_lm_program_host[0].payload[3] = matvec_weight.ptr;
  interp_lm_program_host[0].payload[4] = interp_lm_out.ptr;
  interp_lm_program_host[1].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_lm_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_lm_program_host, interp_lm_program_host + 2));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_lm_counters.ptr), 0,
                 2 * sizeof(int32_t)),
      "cudaMemset interp lm_head counters");
  qwen36_interpreter_program_t interp_lm_program{};
  interp_lm_program.instructions = interp_lm_instructions;
  interp_lm_program.instruction_count = 2;
  interp_lm_program.counters_i32 = interp_lm_counters;
  interp_lm_program.counter_count = 2;
  interp_lm_program.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_lm_program),
              "interpreter lm_head tiled");
  std::vector<float> interp_lm_values = read_bf16(interp_lm_out, 2);
  for (size_t i = 0; i < 2; ++i) {
    expect_close(interp_lm_values[i], matvec_values[i], 0.0f,
                 "interpreter lm_head tiled");
  }

  qwen36_device_ptr_t interp_logits_hidden = dev_alloc<__nv_bfloat16>(16);
  qwen36_device_ptr_t interp_logits_residual = dev_alloc<__nv_bfloat16>(16);
  qwen36_device_ptr_t interp_logits_norm_weight = dev_alloc<__nv_bfloat16>(16);
  qwen36_device_ptr_t interp_logits_lm_weight = dev_alloc<__nv_bfloat16>(32);
  qwen36_device_ptr_t interp_logits_norm_ref = dev_alloc<__nv_bfloat16>(16);
  qwen36_device_ptr_t interp_logits_ref = dev_alloc<__nv_bfloat16>(2);
  qwen36_device_ptr_t interp_logits_norm = dev_alloc<__nv_bfloat16>(16);
  qwen36_device_ptr_t interp_logits_out = dev_alloc<__nv_bfloat16>(2);
  qwen36_device_ptr_t interp_logits_fp4 = dev_alloc<uint8_t>(8);
  qwen36_device_ptr_t interp_logits_scale = dev_alloc<uint8_t>(512);
  qwen36_device_ptr_t interp_logits_global = dev_alloc<float>(1);
  qwen36_device_ptr_t interp_logits_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(3);
  qwen36_device_ptr_t interp_logits_counters = dev_alloc<int32_t>(4);
  copy_bf16(interp_logits_hidden,
            {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f,
             0.5f, -1.5f, 2.5f, -3.5f, 4.5f, -5.5f, 6.5f, -7.5f});
  copy_bf16(interp_logits_residual,
            {0.25f, 0.5f, -0.25f, -0.5f, 0.75f, -0.75f, 1.0f, -1.0f,
             0.125f, -0.125f, 0.375f, -0.375f, 0.625f, -0.625f, 0.875f,
             -0.875f});
  copy_bf16(interp_logits_norm_weight,
            {0.0f, 0.05f, -0.05f, 0.1f, -0.1f, 0.15f, -0.15f, 0.2f,
             -0.2f, 0.25f, -0.25f, 0.3f, -0.3f, 0.35f, -0.35f, 0.4f});
  copy_bf16(interp_logits_lm_weight,
            {0.25f, -0.25f, 0.5f, -0.5f, 0.75f, -0.75f, 1.0f, -1.0f,
             0.125f, -0.125f, 0.375f, -0.375f, 0.625f, -0.625f, 0.875f,
             -0.875f,
             -0.5f, 0.5f, -0.25f, 0.25f, -0.125f, 0.125f, -0.75f,
             0.75f, -1.0f, 1.0f, -0.625f, 0.625f, -0.375f, 0.375f,
             -0.875f, 0.875f});
  qwen36_rmsnorm_spec_t interp_logits_norm_ref_spec{};
  interp_logits_norm_ref_spec.rows = 1;
  interp_logits_norm_ref_spec.hidden = 16;
  interp_logits_norm_ref_spec.eps = 1.0e-6f;
  interp_logits_norm_ref_spec.input_bf16 = interp_logits_hidden;
  interp_logits_norm_ref_spec.weight_bf16 = interp_logits_norm_weight;
  interp_logits_norm_ref_spec.residual_bf16 = interp_logits_residual;
  interp_logits_norm_ref_spec.output_bf16 = interp_logits_norm_ref;
  must_status(qwen36_rmsnorm(&interp_logits_norm_ref_spec),
              "interpreter logits reference rmsnorm");
  qwen36_bf16_matvec_spec_t interp_logits_ref_spec{};
  interp_logits_ref_spec.out_features = 2;
  interp_logits_ref_spec.in_features = 16;
  interp_logits_ref_spec.input_bf16 = interp_logits_norm_ref;
  interp_logits_ref_spec.weight_bf16 = interp_logits_lm_weight;
  interp_logits_ref_spec.output_bf16 = interp_logits_ref;
  must_status(qwen36_bf16_matvec(&interp_logits_ref_spec),
              "interpreter logits reference lm_head");

  qwen36_interpreter_instruction_t interp_logits_program_host[3]{};
  interp_logits_program_host[0].opcode = 2; // RMSNORM_NVFP4_QUANT
  interp_logits_program_host[0].publishes_counter = 0;
  interp_logits_program_host[0].publish_value = 1;
  interp_logits_program_host[0].arrival_counter = 1;
  interp_logits_program_host[0].payload[0] = 16;
  interp_logits_program_host[0].payload[1] = interp_logits_hidden.ptr;
  interp_logits_program_host[0].payload[2] = interp_logits_norm_weight.ptr;
  interp_logits_program_host[0].payload[3] = interp_logits_residual.ptr;
  interp_logits_program_host[0].payload[4] = 0;
  interp_logits_program_host[0].payload[5] = interp_logits_norm.ptr;
  interp_logits_program_host[0].payload[6] = interp_logits_fp4.ptr;
  interp_logits_program_host[0].payload[7] = interp_logits_scale.ptr;
  interp_logits_program_host[0].payload[8] = interp_logits_global.ptr;
  interp_logits_program_host[0].payload[9] =
      static_cast<uint64_t>(eps_bits.u) |
      (static_cast<uint64_t>(scale_bits.u) << 32);
  interp_logits_program_host[1].opcode = 9; // LM_HEAD_TILED
  interp_logits_program_host[1].dep_count = 1;
  interp_logits_program_host[1].deps[0] = qwen36_interpreter_dep_t{0, 1};
  interp_logits_program_host[1].publishes_counter = 2;
  interp_logits_program_host[1].publish_value = 1;
  interp_logits_program_host[1].arrival_counter = 3;
  interp_logits_program_host[1].payload[0] = 2;
  interp_logits_program_host[1].payload[1] = 16;
  interp_logits_program_host[1].payload[2] = interp_logits_norm.ptr;
  interp_logits_program_host[1].payload[3] = interp_logits_lm_weight.ptr;
  interp_logits_program_host[1].payload[4] = interp_logits_out.ptr;
  interp_logits_program_host[2].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_logits_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_logits_program_host, interp_logits_program_host + 3));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_logits_counters.ptr), 0,
                 4 * sizeof(int32_t)),
      "cudaMemset interp final logits counters");
  qwen36_interpreter_program_t interp_logits_program{};
  interp_logits_program.instructions = interp_logits_instructions;
  interp_logits_program.instruction_count = 3;
  interp_logits_program.counters_i32 = interp_logits_counters;
  interp_logits_program.counter_count = 4;
  interp_logits_program.cta_count = 3;
  must_status(qwen36_interpreter_decode_sm120(&interp_logits_program),
              "interpreter final logits");
  std::vector<float> interp_logits_values = read_bf16(interp_logits_out, 2);
  std::vector<float> interp_logits_ref_values =
      read_bf16(interp_logits_ref, 2);
  for (size_t i = 0; i < 2; ++i) {
    expect_close(interp_logits_values[i], interp_logits_ref_values[i], 0.0f,
                 "interpreter final logits");
  }

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

  const size_t interp_gemv_m = 16;
  const size_t interp_gemv_k = 1024;
  const size_t interp_gemv_scale_bytes = 8192;
  qwen36_device_ptr_t interp_gemv_a_fp4 =
      dev_alloc<uint8_t>(interp_gemv_m * interp_gemv_k / 2);
  qwen36_device_ptr_t interp_gemv_a_scale =
      dev_alloc<uint8_t>(interp_gemv_scale_bytes);
  qwen36_device_ptr_t interp_gemv_b_fp4 =
      dev_alloc<uint8_t>(interp_gemv_k / 2);
  qwen36_device_ptr_t interp_gemv_b_scale =
      dev_alloc<uint8_t>(interp_gemv_scale_bytes);
  qwen36_device_ptr_t interp_gemv_out_ref =
      dev_alloc<__nv_bfloat16>(interp_gemv_m);
  qwen36_device_ptr_t interp_gemv_out =
      dev_alloc<__nv_bfloat16>(interp_gemv_m);
  qwen36_device_ptr_t interp_gemv_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(2);
  qwen36_device_ptr_t interp_gemv_counters = dev_alloc<int32_t>(2);
  copy_raw<uint8_t>(
      interp_gemv_a_fp4,
      std::vector<uint8_t>(interp_gemv_m * interp_gemv_k / 2, 0x22));
  copy_raw<uint8_t>(interp_gemv_a_scale,
                    std::vector<uint8_t>(interp_gemv_scale_bytes, 0x38));
  copy_raw<uint8_t>(interp_gemv_b_fp4,
                    std::vector<uint8_t>(interp_gemv_k / 2, 0x22));
  copy_raw<uint8_t>(interp_gemv_b_scale,
                    std::vector<uint8_t>(interp_gemv_scale_bytes, 0x38));
  qwen36_nvfp4_gemm_spec_t interp_gemv_ref_spec{};
  interp_gemv_ref_spec.m = interp_gemv_m;
  interp_gemv_ref_spec.n = 1;
  interp_gemv_ref_spec.k = interp_gemv_k;
  interp_gemv_ref_spec.a_fp4 = interp_gemv_a_fp4;
  interp_gemv_ref_spec.a_scale = interp_gemv_a_scale;
  interp_gemv_ref_spec.b_fp4 = interp_gemv_b_fp4;
  interp_gemv_ref_spec.b_scale = interp_gemv_b_scale;
  interp_gemv_ref_spec.c_bf16 = interp_gemv_out_ref;
  interp_gemv_ref_spec.alpha = 1.0f;
  must_status(qwen36_decode_nvfp4_gemv(&interp_gemv_ref_spec),
              "interpreter gemv reference");
  F32Bits interp_gemv_alpha_bits{1.0f};
  qwen36_interpreter_instruction_t interp_gemv_program_host[2]{};
  interp_gemv_program_host[0].opcode = 3; // NVFP4_GEMV
  interp_gemv_program_host[0].publishes_counter = 0;
  interp_gemv_program_host[0].publish_value = 1;
  interp_gemv_program_host[0].arrival_counter = 1;
  interp_gemv_program_host[0].payload[0] = interp_gemv_m;
  interp_gemv_program_host[0].payload[1] = interp_gemv_k;
  interp_gemv_program_host[0].payload[2] = interp_gemv_a_fp4.ptr;
  interp_gemv_program_host[0].payload[3] = interp_gemv_a_scale.ptr;
  interp_gemv_program_host[0].payload[4] = interp_gemv_b_fp4.ptr;
  interp_gemv_program_host[0].payload[5] = interp_gemv_b_scale.ptr;
  interp_gemv_program_host[0].payload[6] = interp_gemv_out.ptr;
  interp_gemv_program_host[0].payload[7] = interp_gemv_alpha_bits.u;
  interp_gemv_program_host[1].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_gemv_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_gemv_program_host, interp_gemv_program_host + 2));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_gemv_counters.ptr), 0,
                 2 * sizeof(int32_t)),
      "cudaMemset interp gemv counters");
  qwen36_interpreter_program_t interp_gemv_spec{};
  interp_gemv_spec.instructions = interp_gemv_instructions;
  interp_gemv_spec.instruction_count = 2;
  interp_gemv_spec.counters_i32 = interp_gemv_counters;
  interp_gemv_spec.counter_count = 2;
  interp_gemv_spec.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_gemv_spec),
              "interpreter nvfp4 gemv");
  std::vector<float> interp_gemv_ref_values =
      read_bf16(interp_gemv_out_ref, interp_gemv_m);
  std::vector<float> interp_gemv_values =
      read_bf16(interp_gemv_out, interp_gemv_m);
  for (size_t i = 0; i < interp_gemv_m; ++i) {
    expect_close(interp_gemv_values[i], interp_gemv_ref_values[i], 0.0f,
                 "interpreter nvfp4 gemv");
  }

  const size_t interp_mlp_hidden = 1024;
  const size_t interp_mlp_intermediate = 1024;
  const size_t interp_mlp_act_scale_bytes = 8192;
  const size_t interp_mlp_weight_scale_bytes = 65536;
  const size_t interp_mlp_weight_bytes =
      interp_mlp_hidden * interp_mlp_intermediate / 2;
  qwen36_device_ptr_t interp_mlp_input_fp4 =
      dev_alloc<uint8_t>(interp_mlp_hidden / 2);
  qwen36_device_ptr_t interp_mlp_input_scale =
      dev_alloc<uint8_t>(interp_mlp_act_scale_bytes);
  qwen36_device_ptr_t interp_mlp_gate_w =
      dev_alloc<uint8_t>(interp_mlp_weight_bytes);
  qwen36_device_ptr_t interp_mlp_gate_scale =
      dev_alloc<uint8_t>(interp_mlp_weight_scale_bytes);
  qwen36_device_ptr_t interp_mlp_up_w =
      dev_alloc<uint8_t>(interp_mlp_weight_bytes);
  qwen36_device_ptr_t interp_mlp_up_scale =
      dev_alloc<uint8_t>(interp_mlp_weight_scale_bytes);
  qwen36_device_ptr_t interp_mlp_down_w =
      dev_alloc<uint8_t>(interp_mlp_weight_bytes);
  qwen36_device_ptr_t interp_mlp_down_scale =
      dev_alloc<uint8_t>(interp_mlp_weight_scale_bytes);
  qwen36_device_ptr_t interp_mlp_gate_ref =
      dev_alloc<__nv_bfloat16>(interp_mlp_intermediate);
  qwen36_device_ptr_t interp_mlp_up_ref =
      dev_alloc<__nv_bfloat16>(interp_mlp_intermediate);
  qwen36_device_ptr_t interp_mlp_swiglu_fp4_ref =
      dev_alloc<uint8_t>(interp_mlp_intermediate / 2);
  qwen36_device_ptr_t interp_mlp_swiglu_scale_ref =
      dev_alloc<uint8_t>(interp_mlp_act_scale_bytes);
  qwen36_device_ptr_t interp_mlp_swiglu_global_ref = dev_alloc<float>(1);
  qwen36_device_ptr_t interp_mlp_out_ref =
      dev_alloc<__nv_bfloat16>(interp_mlp_hidden);
  qwen36_device_ptr_t interp_mlp_gate =
      dev_alloc<__nv_bfloat16>(interp_mlp_intermediate);
  qwen36_device_ptr_t interp_mlp_up =
      dev_alloc<__nv_bfloat16>(interp_mlp_intermediate);
  qwen36_device_ptr_t interp_mlp_swiglu_fp4 =
      dev_alloc<uint8_t>(interp_mlp_intermediate / 2);
  qwen36_device_ptr_t interp_mlp_swiglu_scale =
      dev_alloc<uint8_t>(interp_mlp_act_scale_bytes);
  qwen36_device_ptr_t interp_mlp_swiglu_global = dev_alloc<float>(1);
  qwen36_device_ptr_t interp_mlp_out =
      dev_alloc<__nv_bfloat16>(interp_mlp_hidden);
  qwen36_device_ptr_t interp_mlp_chunk_swiglu_fp4 =
      dev_alloc<uint8_t>(interp_mlp_intermediate / 2);
  qwen36_device_ptr_t interp_mlp_chunk_swiglu_scale =
      dev_alloc<uint8_t>(interp_mlp_act_scale_bytes);
  qwen36_device_ptr_t interp_mlp_chunk_swiglu_global = dev_alloc<float>(1);
  qwen36_device_ptr_t interp_mlp_chunk_out =
      dev_alloc<__nv_bfloat16>(interp_mlp_hidden);
  qwen36_device_ptr_t interp_mlp_chunk_accum =
      dev_alloc<float>(interp_mlp_hidden);
  qwen36_device_ptr_t interp_mlp_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(4);
  qwen36_device_ptr_t interp_mlp_counters = dev_alloc<int32_t>(6);
  qwen36_device_ptr_t interp_mlp_chunk_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(6);
  qwen36_device_ptr_t interp_mlp_chunk_counters = dev_alloc<int32_t>(10);
  copy_raw<uint8_t>(interp_mlp_input_fp4,
                    std::vector<uint8_t>(interp_mlp_hidden / 2, 0x22));
  copy_raw<uint8_t>(interp_mlp_input_scale,
                    std::vector<uint8_t>(interp_mlp_act_scale_bytes, 0x38));
  copy_raw<uint8_t>(interp_mlp_gate_w,
                    std::vector<uint8_t>(interp_mlp_weight_bytes, 0x22));
  copy_raw<uint8_t>(interp_mlp_gate_scale,
                    std::vector<uint8_t>(interp_mlp_weight_scale_bytes, 0x38));
  copy_raw<uint8_t>(interp_mlp_up_w,
                    std::vector<uint8_t>(interp_mlp_weight_bytes, 0x22));
  copy_raw<uint8_t>(interp_mlp_up_scale,
                    std::vector<uint8_t>(interp_mlp_weight_scale_bytes, 0x38));
  copy_raw<uint8_t>(interp_mlp_down_w,
                    std::vector<uint8_t>(interp_mlp_weight_bytes, 0x22));
  copy_raw<uint8_t>(interp_mlp_down_scale,
                    std::vector<uint8_t>(interp_mlp_weight_scale_bytes, 0x38));

  qwen36_nvfp4_gemm_spec_t interp_mlp_gate_ref_spec{};
  interp_mlp_gate_ref_spec.m = interp_mlp_intermediate;
  interp_mlp_gate_ref_spec.n = 1;
  interp_mlp_gate_ref_spec.k = interp_mlp_hidden;
  interp_mlp_gate_ref_spec.a_fp4 = interp_mlp_gate_w;
  interp_mlp_gate_ref_spec.a_scale = interp_mlp_gate_scale;
  interp_mlp_gate_ref_spec.b_fp4 = interp_mlp_input_fp4;
  interp_mlp_gate_ref_spec.b_scale = interp_mlp_input_scale;
  interp_mlp_gate_ref_spec.c_bf16 = interp_mlp_gate_ref;
  interp_mlp_gate_ref_spec.alpha = 1.0f;
  must_status(qwen36_decode_nvfp4_gemv(&interp_mlp_gate_ref_spec),
              "interpreter MLP gate reference");
  qwen36_nvfp4_gemm_spec_t interp_mlp_up_ref_spec = interp_mlp_gate_ref_spec;
  interp_mlp_up_ref_spec.a_fp4 = interp_mlp_up_w;
  interp_mlp_up_ref_spec.a_scale = interp_mlp_up_scale;
  interp_mlp_up_ref_spec.c_bf16 = interp_mlp_up_ref;
  must_status(qwen36_decode_nvfp4_gemv(&interp_mlp_up_ref_spec),
              "interpreter MLP up reference");
  qwen36_swiglu_nvfp4_quantize_spec_t interp_mlp_swiglu_ref_spec{};
  interp_mlp_swiglu_ref_spec.intermediate = interp_mlp_intermediate;
  interp_mlp_swiglu_ref_spec.gate_bf16 = interp_mlp_gate_ref;
  interp_mlp_swiglu_ref_spec.up_bf16 = interp_mlp_up_ref;
  interp_mlp_swiglu_ref_spec.output_fp4 = interp_mlp_swiglu_fp4_ref;
  interp_mlp_swiglu_ref_spec.output_scale_e4m3 =
      interp_mlp_swiglu_scale_ref;
  interp_mlp_swiglu_ref_spec.output_tensor_scale_f32 =
      interp_mlp_swiglu_global_ref;
  interp_mlp_swiglu_ref_spec.input_tensor_scale_f32 = 1.0f;
  must_status(qwen36_swiglu_nvfp4_quantize(&interp_mlp_swiglu_ref_spec),
              "interpreter MLP swiglu reference");
  qwen36_nvfp4_gemm_spec_t interp_mlp_down_ref_spec{};
  interp_mlp_down_ref_spec.m = interp_mlp_hidden;
  interp_mlp_down_ref_spec.n = 1;
  interp_mlp_down_ref_spec.k = interp_mlp_intermediate;
  interp_mlp_down_ref_spec.a_fp4 = interp_mlp_down_w;
  interp_mlp_down_ref_spec.a_scale = interp_mlp_down_scale;
  interp_mlp_down_ref_spec.b_fp4 = interp_mlp_swiglu_fp4_ref;
  interp_mlp_down_ref_spec.b_scale = interp_mlp_swiglu_scale_ref;
  interp_mlp_down_ref_spec.c_bf16 = interp_mlp_out_ref;
  interp_mlp_down_ref_spec.alpha = 1.0f;
  must_status(qwen36_decode_nvfp4_gemv(&interp_mlp_down_ref_spec),
              "interpreter MLP down reference");

  qwen36_interpreter_instruction_t interp_mlp_program_host[4]{};
  interp_mlp_program_host[0].opcode = 16; // NVFP4_GEMV_PAIR gate+up
  interp_mlp_program_host[0].publishes_counter = 0;
  interp_mlp_program_host[0].publish_value = 1;
  interp_mlp_program_host[0].arrival_counter = 1;
  interp_mlp_program_host[0].payload[0] = interp_mlp_intermediate;
  interp_mlp_program_host[0].payload[1] = interp_mlp_hidden;
  interp_mlp_program_host[0].payload[2] = interp_mlp_gate_w.ptr;
  interp_mlp_program_host[0].payload[3] = interp_mlp_gate_scale.ptr;
  interp_mlp_program_host[0].payload[4] = interp_mlp_up_w.ptr;
  interp_mlp_program_host[0].payload[5] = interp_mlp_up_scale.ptr;
  interp_mlp_program_host[0].payload[6] = interp_mlp_input_fp4.ptr;
  interp_mlp_program_host[0].payload[7] = interp_mlp_input_scale.ptr;
  interp_mlp_program_host[0].payload[8] = interp_mlp_gate.ptr;
  interp_mlp_program_host[0].payload[9] = interp_mlp_up.ptr;
  interp_mlp_program_host[0].payload[10] = interp_gemv_alpha_bits.u;
  interp_mlp_program_host[0].payload[11] = interp_gemv_alpha_bits.u;
  interp_mlp_program_host[1].opcode = 4; // SWIGLU_NVFP4_QUANT
  interp_mlp_program_host[1].dep_count = 1;
  interp_mlp_program_host[1].deps[0] = qwen36_interpreter_dep_t{0, 1};
  interp_mlp_program_host[1].publishes_counter = 2;
  interp_mlp_program_host[1].publish_value = 1;
  interp_mlp_program_host[1].arrival_counter = 3;
  interp_mlp_program_host[1].payload[0] = interp_mlp_intermediate;
  interp_mlp_program_host[1].payload[1] = interp_mlp_gate.ptr;
  interp_mlp_program_host[1].payload[2] = interp_mlp_up.ptr;
  interp_mlp_program_host[1].payload[3] = interp_mlp_swiglu_fp4.ptr;
  interp_mlp_program_host[1].payload[4] = interp_mlp_swiglu_scale.ptr;
  interp_mlp_program_host[1].payload[5] = interp_mlp_swiglu_global.ptr;
  interp_mlp_program_host[1].payload[6] = swiglu_scale_bits.u;
  interp_mlp_program_host[2].opcode = 3; // NVFP4_GEMV down
  interp_mlp_program_host[2].dep_count = 1;
  interp_mlp_program_host[2].deps[0] = qwen36_interpreter_dep_t{2, 1};
  interp_mlp_program_host[2].publishes_counter = 4;
  interp_mlp_program_host[2].publish_value = 1;
  interp_mlp_program_host[2].arrival_counter = 5;
  interp_mlp_program_host[2].payload[0] = interp_mlp_hidden;
  interp_mlp_program_host[2].payload[1] = interp_mlp_intermediate;
  interp_mlp_program_host[2].payload[2] = interp_mlp_down_w.ptr;
  interp_mlp_program_host[2].payload[3] = interp_mlp_down_scale.ptr;
  interp_mlp_program_host[2].payload[4] = interp_mlp_swiglu_fp4.ptr;
  interp_mlp_program_host[2].payload[5] = interp_mlp_swiglu_scale.ptr;
  interp_mlp_program_host[2].payload[6] = interp_mlp_out.ptr;
  interp_mlp_program_host[2].payload[7] = interp_gemv_alpha_bits.u;
  interp_mlp_program_host[3].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_mlp_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_mlp_program_host, interp_mlp_program_host + 4));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_mlp_counters.ptr), 0,
                 6 * sizeof(int32_t)),
      "cudaMemset interp MLP counters");
  qwen36_interpreter_program_t interp_mlp_spec{};
  interp_mlp_spec.instructions = interp_mlp_instructions;
  interp_mlp_spec.instruction_count = 4;
  interp_mlp_spec.counters_i32 = interp_mlp_counters;
  interp_mlp_spec.counter_count = 6;
  interp_mlp_spec.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_mlp_spec),
              "interpreter MLP");
  std::vector<float> interp_mlp_ref_values =
      read_bf16(interp_mlp_out_ref, interp_mlp_hidden);
  std::vector<float> interp_mlp_values =
      read_bf16(interp_mlp_out, interp_mlp_hidden);
  for (size_t i = 0; i < interp_mlp_hidden; ++i) {
    expect_close(interp_mlp_values[i], interp_mlp_ref_values[i], 0.0f,
                 "interpreter MLP");
  }

  constexpr size_t interp_mlp_chunk_k = 512;
  qwen36_interpreter_instruction_t interp_mlp_chunk_program_host[6]{};
  interp_mlp_chunk_program_host[0] = interp_mlp_program_host[0];
  interp_mlp_chunk_program_host[1].opcode =
      17; // SWIGLU_NVFP4_QUANT_CHUNK
  interp_mlp_chunk_program_host[1].dep_count = 1;
  interp_mlp_chunk_program_host[1].deps[0] = qwen36_interpreter_dep_t{0, 1};
  interp_mlp_chunk_program_host[1].publishes_counter = 2;
  interp_mlp_chunk_program_host[1].publish_value = 1;
  interp_mlp_chunk_program_host[1].arrival_counter = 3;
  interp_mlp_chunk_program_host[1].payload[0] = 0;
  interp_mlp_chunk_program_host[1].payload[1] = interp_mlp_chunk_k;
  interp_mlp_chunk_program_host[1].payload[2] = interp_mlp_intermediate;
  interp_mlp_chunk_program_host[1].payload[3] = interp_mlp_gate.ptr;
  interp_mlp_chunk_program_host[1].payload[4] = interp_mlp_up.ptr;
  interp_mlp_chunk_program_host[1].payload[5] =
      interp_mlp_chunk_swiglu_fp4.ptr;
  interp_mlp_chunk_program_host[1].payload[6] =
      interp_mlp_chunk_swiglu_scale.ptr;
  interp_mlp_chunk_program_host[1].payload[7] =
      interp_mlp_chunk_swiglu_global.ptr;
  interp_mlp_chunk_program_host[1].payload[8] = swiglu_scale_bits.u;
  interp_mlp_chunk_program_host[2].opcode =
      18; // NVFP4_GEMV_CHUNK_ACCUM
  interp_mlp_chunk_program_host[2].dep_count = 1;
  interp_mlp_chunk_program_host[2].deps[0] = qwen36_interpreter_dep_t{2, 1};
  interp_mlp_chunk_program_host[2].publishes_counter = 4;
  interp_mlp_chunk_program_host[2].publish_value = 1;
  interp_mlp_chunk_program_host[2].arrival_counter = 5;
  interp_mlp_chunk_program_host[2].payload[0] = interp_mlp_hidden;
  interp_mlp_chunk_program_host[2].payload[1] = interp_mlp_intermediate;
  interp_mlp_chunk_program_host[2].payload[2] = 0;
  interp_mlp_chunk_program_host[2].payload[3] = interp_mlp_chunk_k;
  interp_mlp_chunk_program_host[2].payload[4] = interp_mlp_down_w.ptr;
  interp_mlp_chunk_program_host[2].payload[5] = interp_mlp_down_scale.ptr;
  interp_mlp_chunk_program_host[2].payload[6] =
      interp_mlp_chunk_swiglu_fp4.ptr;
  interp_mlp_chunk_program_host[2].payload[7] =
      interp_mlp_chunk_swiglu_scale.ptr;
  interp_mlp_chunk_program_host[2].payload[8] = interp_mlp_chunk_accum.ptr;
  interp_mlp_chunk_program_host[2].payload[9] = interp_mlp_chunk_out.ptr;
  interp_mlp_chunk_program_host[2].payload[10] = interp_gemv_alpha_bits.u;
  interp_mlp_chunk_program_host[2].payload[11] = 1; // reset accumulation
  interp_mlp_chunk_program_host[3].opcode =
      17; // SWIGLU_NVFP4_QUANT_CHUNK
  interp_mlp_chunk_program_host[3].dep_count = 1;
  interp_mlp_chunk_program_host[3].deps[0] = qwen36_interpreter_dep_t{4, 1};
  interp_mlp_chunk_program_host[3].publishes_counter = 6;
  interp_mlp_chunk_program_host[3].publish_value = 1;
  interp_mlp_chunk_program_host[3].arrival_counter = 7;
  interp_mlp_chunk_program_host[3].payload[0] = interp_mlp_chunk_k;
  interp_mlp_chunk_program_host[3].payload[1] = interp_mlp_chunk_k;
  interp_mlp_chunk_program_host[3].payload[2] = interp_mlp_intermediate;
  interp_mlp_chunk_program_host[3].payload[3] = interp_mlp_gate.ptr;
  interp_mlp_chunk_program_host[3].payload[4] = interp_mlp_up.ptr;
  interp_mlp_chunk_program_host[3].payload[5] =
      interp_mlp_chunk_swiglu_fp4.ptr;
  interp_mlp_chunk_program_host[3].payload[6] =
      interp_mlp_chunk_swiglu_scale.ptr;
  interp_mlp_chunk_program_host[3].payload[7] =
      interp_mlp_chunk_swiglu_global.ptr;
  interp_mlp_chunk_program_host[3].payload[8] = swiglu_scale_bits.u;
  interp_mlp_chunk_program_host[4].opcode =
      18; // NVFP4_GEMV_CHUNK_ACCUM
  interp_mlp_chunk_program_host[4].dep_count = 1;
  interp_mlp_chunk_program_host[4].deps[0] = qwen36_interpreter_dep_t{6, 1};
  interp_mlp_chunk_program_host[4].publishes_counter = 8;
  interp_mlp_chunk_program_host[4].publish_value = 1;
  interp_mlp_chunk_program_host[4].arrival_counter = 9;
  interp_mlp_chunk_program_host[4].payload[0] = interp_mlp_hidden;
  interp_mlp_chunk_program_host[4].payload[1] = interp_mlp_intermediate;
  interp_mlp_chunk_program_host[4].payload[2] = interp_mlp_chunk_k;
  interp_mlp_chunk_program_host[4].payload[3] = interp_mlp_chunk_k;
  interp_mlp_chunk_program_host[4].payload[4] = interp_mlp_down_w.ptr;
  interp_mlp_chunk_program_host[4].payload[5] = interp_mlp_down_scale.ptr;
  interp_mlp_chunk_program_host[4].payload[6] =
      interp_mlp_chunk_swiglu_fp4.ptr;
  interp_mlp_chunk_program_host[4].payload[7] =
      interp_mlp_chunk_swiglu_scale.ptr;
  interp_mlp_chunk_program_host[4].payload[8] = interp_mlp_chunk_accum.ptr;
  interp_mlp_chunk_program_host[4].payload[9] = interp_mlp_chunk_out.ptr;
  interp_mlp_chunk_program_host[4].payload[10] = interp_gemv_alpha_bits.u;
  interp_mlp_chunk_program_host[4].payload[11] = 2; // finalize output
  interp_mlp_chunk_program_host[5].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_mlp_chunk_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_mlp_chunk_program_host, interp_mlp_chunk_program_host + 6));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_mlp_chunk_counters.ptr), 0,
                 10 * sizeof(int32_t)),
      "cudaMemset interp MLP chunk counters");
  qwen36_interpreter_program_t interp_mlp_chunk_spec{};
  interp_mlp_chunk_spec.instructions = interp_mlp_chunk_instructions;
  interp_mlp_chunk_spec.instruction_count = 6;
  interp_mlp_chunk_spec.counters_i32 = interp_mlp_chunk_counters;
  interp_mlp_chunk_spec.counter_count = 10;
  interp_mlp_chunk_spec.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_mlp_chunk_spec),
              "interpreter MLP chunked");
  std::vector<uint8_t> interp_mlp_chunk_fp4 =
      read_raw<uint8_t>(interp_mlp_chunk_swiglu_fp4,
                        interp_mlp_intermediate / 2);
  std::vector<uint8_t> interp_mlp_ref_fp4 =
      read_raw<uint8_t>(interp_mlp_swiglu_fp4_ref,
                        interp_mlp_intermediate / 2);
  for (size_t i = 0; i < interp_mlp_ref_fp4.size(); ++i) {
    if (interp_mlp_chunk_fp4[i] != interp_mlp_ref_fp4[i]) {
      fprintf(stderr,
              "interpreter MLP chunked SwiGLU fp4 byte %zu expected %u got %u\n",
              i, static_cast<unsigned>(interp_mlp_ref_fp4[i]),
              static_cast<unsigned>(interp_mlp_chunk_fp4[i]));
      exit(1);
    }
  }
  std::vector<float> interp_mlp_chunk_values =
      read_bf16(interp_mlp_chunk_out, interp_mlp_hidden);
  for (size_t i = 0; i < interp_mlp_hidden; ++i) {
    expect_close(interp_mlp_chunk_values[i], interp_mlp_ref_values[i],
                 32768.0f, "interpreter MLP chunked");
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

  const size_t interp_conv_channels = 4;
  const size_t interp_conv_kernel = 4;
  const size_t interp_conv_heads = 2;
  qwen36_device_ptr_t interp_conv_input =
      dev_alloc<__nv_bfloat16>(interp_conv_channels);
  qwen36_device_ptr_t interp_conv_history_ref =
      dev_alloc<__nv_bfloat16>(interp_conv_channels * (interp_conv_kernel - 1));
  qwen36_device_ptr_t interp_conv_history =
      dev_alloc<__nv_bfloat16>(interp_conv_channels * (interp_conv_kernel - 1));
  qwen36_device_ptr_t interp_conv_weight =
      dev_alloc<__nv_bfloat16>(interp_conv_channels * interp_conv_kernel);
  qwen36_device_ptr_t interp_conv_out_ref =
      dev_alloc<__nv_bfloat16>(interp_conv_channels);
  qwen36_device_ptr_t interp_conv_out =
      dev_alloc<__nv_bfloat16>(interp_conv_channels);
  qwen36_device_ptr_t interp_gdn_a =
      dev_alloc<__nv_bfloat16>(interp_conv_heads);
  qwen36_device_ptr_t interp_gdn_b =
      dev_alloc<__nv_bfloat16>(interp_conv_heads);
  qwen36_device_ptr_t interp_gdn_a_log =
      dev_alloc<__nv_bfloat16>(interp_conv_heads);
  qwen36_device_ptr_t interp_gdn_dt =
      dev_alloc<__nv_bfloat16>(interp_conv_heads);
  qwen36_device_ptr_t interp_gdn_gate_ref = dev_alloc<float>(interp_conv_heads);
  qwen36_device_ptr_t interp_gdn_beta_ref = dev_alloc<float>(interp_conv_heads);
  qwen36_device_ptr_t interp_gdn_gate = dev_alloc<float>(interp_conv_heads);
  qwen36_device_ptr_t interp_gdn_beta = dev_alloc<float>(interp_conv_heads);
  qwen36_device_ptr_t interp_conv_gdn_instructions =
      dev_alloc<qwen36_interpreter_instruction_t>(2);
  qwen36_device_ptr_t interp_conv_gdn_counters = dev_alloc<int32_t>(2);
  copy_bf16(interp_conv_input, {0.25f, -0.5f, 1.0f, 2.0f});
  const std::vector<float> interp_conv_history_values = {
      0.0f, 0.25f, -0.25f, 1.0f, 0.5f, -0.5f,
      0.75f, 0.0f, 0.125f, -0.125f, 0.375f, -0.375f};
  copy_bf16(interp_conv_history_ref, interp_conv_history_values);
  copy_bf16(interp_conv_history, interp_conv_history_values);
  copy_bf16(interp_conv_weight,
            {1.0f, 0.5f, -0.25f, 0.125f, -0.5f, 1.0f, 0.25f,
             -0.125f, 0.25f, 0.25f, 0.25f, 0.25f, 1.0f, -1.0f,
             0.5f, -0.5f});
  copy_bf16(interp_gdn_a, {0.0f, 1.0f});
  copy_bf16(interp_gdn_b, {0.0f, -1.0f});
  copy_bf16(interp_gdn_a_log, {0.0f, 0.5f});
  copy_bf16(interp_gdn_dt, {0.0f, -0.25f});
  qwen36_conv1d_gdn_gate_fused_spec_t interp_conv_gdn_ref_spec{};
  interp_conv_gdn_ref_spec.channels = interp_conv_channels;
  interp_conv_gdn_ref_spec.kernel_size = interp_conv_kernel;
  interp_conv_gdn_ref_spec.conv_input_bf16 = interp_conv_input;
  interp_conv_gdn_ref_spec.conv_history_bf16 = interp_conv_history_ref;
  interp_conv_gdn_ref_spec.conv_weight_bf16 = interp_conv_weight;
  interp_conv_gdn_ref_spec.conv_output_bf16 = interp_conv_out_ref;
  interp_conv_gdn_ref_spec.heads = interp_conv_heads;
  interp_conv_gdn_ref_spec.gdn_a_bf16 = interp_gdn_a;
  interp_conv_gdn_ref_spec.gdn_b_bf16 = interp_gdn_b;
  interp_conv_gdn_ref_spec.gdn_a_log_bf16 = interp_gdn_a_log;
  interp_conv_gdn_ref_spec.gdn_dt_bias_bf16 = interp_gdn_dt;
  interp_conv_gdn_ref_spec.gate_f32 = interp_gdn_gate_ref;
  interp_conv_gdn_ref_spec.beta_f32 = interp_gdn_beta_ref;
  must_status(qwen36_conv1d_gdn_gate_fused(&interp_conv_gdn_ref_spec),
              "interpreter conv1d+gdn reference");
  qwen36_interpreter_instruction_t interp_conv_gdn_program_host[2]{};
  interp_conv_gdn_program_host[0].opcode = 15; // CONV1D_GDN_GATE_FUSED
  interp_conv_gdn_program_host[0].publishes_counter = 0;
  interp_conv_gdn_program_host[0].publish_value = 1;
  interp_conv_gdn_program_host[0].arrival_counter = 1;
  interp_conv_gdn_program_host[0].payload[0] = interp_conv_channels;
  interp_conv_gdn_program_host[0].payload[1] =
      interp_conv_kernel | (static_cast<uint64_t>(interp_conv_heads) << 32);
  interp_conv_gdn_program_host[0].payload[2] = interp_conv_input.ptr;
  interp_conv_gdn_program_host[0].payload[3] = interp_conv_history.ptr;
  interp_conv_gdn_program_host[0].payload[4] = interp_conv_weight.ptr;
  interp_conv_gdn_program_host[0].payload[5] = interp_conv_out.ptr;
  interp_conv_gdn_program_host[0].payload[6] = interp_gdn_a.ptr;
  interp_conv_gdn_program_host[0].payload[7] = interp_gdn_b.ptr;
  interp_conv_gdn_program_host[0].payload[8] = interp_gdn_a_log.ptr;
  interp_conv_gdn_program_host[0].payload[9] = interp_gdn_dt.ptr;
  interp_conv_gdn_program_host[0].payload[10] = interp_gdn_gate.ptr;
  interp_conv_gdn_program_host[0].payload[11] = interp_gdn_beta.ptr;
  interp_conv_gdn_program_host[1].opcode = 0; // EXIT
  copy_raw<qwen36_interpreter_instruction_t>(
      interp_conv_gdn_instructions,
      std::vector<qwen36_interpreter_instruction_t>(
          interp_conv_gdn_program_host, interp_conv_gdn_program_host + 2));
  must_cuda<int32_t>(
      cudaMemset(reinterpret_cast<void *>(interp_conv_gdn_counters.ptr), 0,
                 2 * sizeof(int32_t)),
      "cudaMemset interp conv1d+gdn counters");
  qwen36_interpreter_program_t interp_conv_gdn_spec{};
  interp_conv_gdn_spec.instructions = interp_conv_gdn_instructions;
  interp_conv_gdn_spec.instruction_count = 2;
  interp_conv_gdn_spec.counters_i32 = interp_conv_gdn_counters;
  interp_conv_gdn_spec.counter_count = 2;
  interp_conv_gdn_spec.cta_count = 2;
  must_status(qwen36_interpreter_decode_sm120(&interp_conv_gdn_spec),
              "interpreter conv1d+gdn");
  std::vector<float> interp_conv_out_ref_values =
      read_bf16(interp_conv_out_ref, interp_conv_channels);
  std::vector<float> interp_conv_out_values =
      read_bf16(interp_conv_out, interp_conv_channels);
  std::vector<float> interp_conv_history_ref_values =
      read_bf16(interp_conv_history_ref,
                interp_conv_channels * (interp_conv_kernel - 1));
  std::vector<float> interp_conv_history_out_values =
      read_bf16(interp_conv_history,
                interp_conv_channels * (interp_conv_kernel - 1));
  std::vector<float> interp_gdn_gate_ref_values =
      read_raw<float>(interp_gdn_gate_ref, interp_conv_heads);
  std::vector<float> interp_gdn_gate_values =
      read_raw<float>(interp_gdn_gate, interp_conv_heads);
  std::vector<float> interp_gdn_beta_ref_values =
      read_raw<float>(interp_gdn_beta_ref, interp_conv_heads);
  std::vector<float> interp_gdn_beta_values =
      read_raw<float>(interp_gdn_beta, interp_conv_heads);
  for (size_t i = 0; i < interp_conv_channels; ++i) {
    expect_close(interp_conv_out_values[i], interp_conv_out_ref_values[i],
                 0.0f, "interpreter conv1d+gdn conv output");
  }
  for (size_t i = 0; i < interp_conv_history_ref_values.size(); ++i) {
    expect_close(interp_conv_history_out_values[i],
                 interp_conv_history_ref_values[i], 0.0f,
                 "interpreter conv1d+gdn history");
  }
  for (size_t i = 0; i < interp_conv_heads; ++i) {
    expect_close(interp_gdn_gate_values[i], interp_gdn_gate_ref_values[i],
                 0.0f, "interpreter conv1d+gdn gate");
    expect_close(interp_gdn_beta_values[i], interp_gdn_beta_ref_values[i],
                 0.0f, "interpreter conv1d+gdn beta");
  }

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
  dev_free<__nv_bfloat16>(interp_attn_cache_k_ref);
  dev_free<__nv_bfloat16>(interp_attn_cache_v_ref);
  dev_free<__nv_bfloat16>(interp_attn_out_ref);
  dev_free<__nv_bfloat16>(interp_attn_cache_k);
  dev_free<__nv_bfloat16>(interp_attn_cache_v);
  dev_free<__nv_bfloat16>(interp_attn_out);
  dev_free<qwen36_attention_decode_spec_t>(interp_attn_spec_device);
  dev_free<qwen36_interpreter_instruction_t>(interp_attn_instructions);
  dev_free<int32_t>(interp_attn_counters);
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
  dev_free<__nv_bfloat16>(interp_delta_state);
  dev_free<__nv_bfloat16>(interp_delta_out);
  dev_free<qwen36_deltanet_decode_spec_t>(interp_delta_spec_device);
  dev_free<qwen36_interpreter_instruction_t>(interp_delta_instructions);
  dev_free<int32_t>(interp_delta_counters);
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
  dev_free<__nv_bfloat16>(interp_norm_out);
  dev_free<uint8_t>(interp_norm_fp4);
  dev_free<uint8_t>(interp_norm_scale);
  dev_free<float>(interp_norm_global);
  dev_free<qwen36_interpreter_instruction_t>(interp_norm_instructions);
  dev_free<int32_t>(interp_norm_counters);
  dev_free<int32_t>(rope_pos);
  dev_free<__nv_bfloat16>(rope_q);
  dev_free<__nv_bfloat16>(rope_k);
  dev_free<__nv_bfloat16>(interp_rope_q_ref);
  dev_free<__nv_bfloat16>(interp_rope_k_ref);
  dev_free<__nv_bfloat16>(interp_rope_q);
  dev_free<__nv_bfloat16>(interp_rope_k);
  dev_free<int32_t>(interp_rope_pos);
  dev_free<qwen36_interpreter_instruction_t>(interp_rope_instructions);
  dev_free<int32_t>(interp_rope_counters);
  dev_free<__nv_bfloat16>(swiglu_gate);
  dev_free<__nv_bfloat16>(swiglu_up);
  dev_free<__nv_bfloat16>(swiglu_out);
  dev_free<__nv_bfloat16>(interp_swiglu_gate);
  dev_free<__nv_bfloat16>(interp_swiglu_up);
  dev_free<uint8_t>(interp_swiglu_fp4_ref);
  dev_free<uint8_t>(interp_swiglu_scale_ref);
  dev_free<float>(interp_swiglu_global_ref);
  dev_free<uint8_t>(interp_swiglu_fp4);
  dev_free<uint8_t>(interp_swiglu_scale);
  dev_free<float>(interp_swiglu_global);
  dev_free<qwen36_interpreter_instruction_t>(interp_swiglu_instructions);
  dev_free<int32_t>(interp_swiglu_counters);
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
  dev_free<__nv_bfloat16>(interp_lm_out);
  dev_free<qwen36_interpreter_instruction_t>(interp_lm_instructions);
  dev_free<int32_t>(interp_lm_counters);
  dev_free<__nv_bfloat16>(interp_logits_hidden);
  dev_free<__nv_bfloat16>(interp_logits_residual);
  dev_free<__nv_bfloat16>(interp_logits_norm_weight);
  dev_free<__nv_bfloat16>(interp_logits_lm_weight);
  dev_free<__nv_bfloat16>(interp_logits_norm_ref);
  dev_free<__nv_bfloat16>(interp_logits_ref);
  dev_free<__nv_bfloat16>(interp_logits_norm);
  dev_free<__nv_bfloat16>(interp_logits_out);
  dev_free<uint8_t>(interp_logits_fp4);
  dev_free<uint8_t>(interp_logits_scale);
  dev_free<float>(interp_logits_global);
  dev_free<qwen36_interpreter_instruction_t>(interp_logits_instructions);
  dev_free<int32_t>(interp_logits_counters);
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
  dev_free<uint8_t>(interp_gemv_a_fp4);
  dev_free<uint8_t>(interp_gemv_a_scale);
  dev_free<uint8_t>(interp_gemv_b_fp4);
  dev_free<uint8_t>(interp_gemv_b_scale);
  dev_free<__nv_bfloat16>(interp_gemv_out_ref);
  dev_free<__nv_bfloat16>(interp_gemv_out);
  dev_free<qwen36_interpreter_instruction_t>(interp_gemv_instructions);
  dev_free<int32_t>(interp_gemv_counters);
  dev_free<uint8_t>(interp_mlp_input_fp4);
  dev_free<uint8_t>(interp_mlp_input_scale);
  dev_free<uint8_t>(interp_mlp_gate_w);
  dev_free<uint8_t>(interp_mlp_gate_scale);
  dev_free<uint8_t>(interp_mlp_up_w);
  dev_free<uint8_t>(interp_mlp_up_scale);
  dev_free<uint8_t>(interp_mlp_down_w);
  dev_free<uint8_t>(interp_mlp_down_scale);
  dev_free<__nv_bfloat16>(interp_mlp_gate_ref);
  dev_free<__nv_bfloat16>(interp_mlp_up_ref);
  dev_free<uint8_t>(interp_mlp_swiglu_fp4_ref);
  dev_free<uint8_t>(interp_mlp_swiglu_scale_ref);
  dev_free<float>(interp_mlp_swiglu_global_ref);
  dev_free<__nv_bfloat16>(interp_mlp_out_ref);
  dev_free<__nv_bfloat16>(interp_mlp_gate);
  dev_free<__nv_bfloat16>(interp_mlp_up);
  dev_free<uint8_t>(interp_mlp_swiglu_fp4);
  dev_free<uint8_t>(interp_mlp_swiglu_scale);
  dev_free<float>(interp_mlp_swiglu_global);
  dev_free<__nv_bfloat16>(interp_mlp_out);
  dev_free<qwen36_interpreter_instruction_t>(interp_mlp_instructions);
  dev_free<int32_t>(interp_mlp_counters);
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

  // ------------------------------------------------------------------
  // DFlash drafter attention (Phase C v1) smoke
  // ------------------------------------------------------------------
  // Tiny shapes vs a CPU reference. Goal: catch obvious bugs in the
  // online-softmax accumulation, the GQA broadcast, and the SWA mask.
  // The kernel currently only specialises head_dim=128 so we have to
  // exercise it at that head_dim — q_len/kv_seq_len/heads stay small to
  // keep the host reference cheap (one or two ms).
  {
    const int q_len = 2;
    const int kv_seq_len = 5;
    const int q_heads = 4;
    const int kv_heads = 2;
    const int head_dim = 128;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    std::mt19937 rng(424242);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> q_host(q_len * q_heads * head_dim);
    std::vector<float> k_host(kv_seq_len * kv_heads * head_dim);
    std::vector<float> v_host(kv_seq_len * kv_heads * head_dim);
    for (auto &x : q_host) x = dist(rng);
    for (auto &x : k_host) x = dist(rng);
    for (auto &x : v_host) x = dist(rng);

    // Host reference: full attention, online softmax in fp32, BF16
    // round-trip on Q/K/V to match what the kernel sees.
    auto bf16_round = [](float x) {
      return __bfloat162float(__float2bfloat16(x));
    };
    std::vector<float> ref(q_len * q_heads * head_dim, 0.0f);
    for (int qp = 0; qp < q_len; ++qp) {
      for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = (qh * kv_heads) / q_heads;
        std::vector<float> scores(kv_seq_len, 0.0f);
        for (int j = 0; j < kv_seq_len; ++j) {
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            float q = bf16_round(q_host[(qp * q_heads + qh) * head_dim + d]);
            float k = bf16_round(k_host[(j * kv_heads + kvh) * head_dim + d]);
            dot += q * k;
          }
          scores[j] = dot * scale;
        }
        float m = -INFINITY;
        for (float s : scores) m = fmaxf(m, s);
        float sum = 0.0f;
        for (float &s : scores) {
          s = expf(s - m);
          sum += s;
        }
        for (int d = 0; d < head_dim; ++d) {
          float acc = 0.0f;
          for (int j = 0; j < kv_seq_len; ++j) {
            float v = bf16_round(v_host[(j * kv_heads + kvh) * head_dim + d]);
            acc += scores[j] * v;
          }
          ref[(qp * q_heads + qh) * head_dim + d] = acc / sum;
        }
      }
    }

    qwen36_device_ptr_t q_dev = dev_alloc<__nv_bfloat16>(q_host.size());
    qwen36_device_ptr_t k_dev = dev_alloc<__nv_bfloat16>(k_host.size());
    qwen36_device_ptr_t v_dev = dev_alloc<__nv_bfloat16>(v_host.size());
    qwen36_device_ptr_t out_dev = dev_alloc<__nv_bfloat16>(ref.size());
    copy_bf16(q_dev, q_host);
    copy_bf16(k_dev, k_host);
    copy_bf16(v_dev, v_host);

    qwen36_drafter_attention_block_spec_t spec{};
    spec.q_bf16 = q_dev;
    spec.k_bf16 = k_dev;
    spec.v_bf16 = v_dev;
    spec.output_bf16 = out_dev;
    spec.q_len = q_len;
    spec.kv_seq_len = kv_seq_len;
    spec.q_heads = q_heads;
    spec.kv_heads = kv_heads;
    spec.head_dim = head_dim;
    spec.sliding_window = 0;

    must_status(qwen36_drafter_attention_block_bf16(&spec),
                "drafter_attention_block_bf16 full");

    std::vector<float> got = read_bf16(out_dev, ref.size());
    double num = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      num += static_cast<double>(got[i]) * static_cast<double>(ref[i]);
      denom_a += static_cast<double>(got[i]) * static_cast<double>(got[i]);
      denom_b += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    double cos = num / sqrt(denom_a * denom_b);
    if (cos < 0.998) {
      fprintf(stderr, "drafter attention full cos sim %.6f < 0.998\n", cos);
      exit(1);
    }
    printf("drafter attention full cos sim %.6f\n", cos);

    // NOT_IMPLEMENTED smoke: head_dim != 128 should soft-fall back.
    qwen36_drafter_attention_block_spec_t bad = spec;
    bad.head_dim = 64;
    expect_status(qwen36_drafter_attention_block_bf16(&bad),
                  QWEN36_STATUS_NOT_IMPLEMENTED,
                  "drafter_attention_block_bf16 head_dim=64");

    dev_free<__nv_bfloat16>(q_dev);
    dev_free<__nv_bfloat16>(k_dev);
    dev_free<__nv_bfloat16>(v_dev);
    dev_free<__nv_bfloat16>(out_dev);
  }

  // ------------------------------------------------------------------
  // DFlash drafter attention — Phase 1 FA-tiled parity gate (P0).
  // ------------------------------------------------------------------
  // Closes Phase 1 task #49, which was never exercised because the
  // block above uses q_len=2 — below the FA gate (q_len == kFlashM ==
  // 16) — so the FA path always returned NOT_IMPLEMENTED and only v1
  // was ever tested. This block runs the SPECIALISED FA shape
  // (q_len=16, head_dim=128) and compares FA vs v1 vs an FP32 CPU
  // reference across the kv_seq_len regime transition and both SWA
  // configs, at the GQA ratios the DFlash drafter actually uses.
  //
  // The investigation harness (workflow wf_2128592f-3a2) measured cos
  // 0.999998 here; this gate locks that in so any future drift in the
  // FA kernel turns the smoke RED instead of silently degrading AL.
  {
    // The FA entry qwen36_drafter_attention_block_flash_bf16 is
    // forward-declared at file scope (extern "C") — calling it directly
    // avoids dispatcher ambiguity; it self-gates on
    // QWEN36_DRAFTER_ATTENTION_FLASH so we set the env first.
    const int q_len = 16; // == kFlashM, the FA-specialised q_len
    const int head_dim = 128;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const int gqa[][2] = {{32, 8}, {40, 8}};
    const int kv_lens[] = {16, 64, 128, 1024, 4096};
    const int swas[] = {0, 2048};

    auto bf16_round = [](float x) {
      return __bfloat162float(__float2bfloat16(x));
    };
    auto cos_sim = [](const std::vector<float> &a,
                      const std::vector<float> &b) -> double {
      double num = 0.0, da = 0.0, db = 0.0;
      for (size_t i = 0; i < a.size(); ++i) {
        num += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        da += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        db += static_cast<double>(b[i]) * static_cast<double>(b[i]);
      }
      return num / sqrt(da * db);
    };

    std::mt19937 rng(909090);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    int cases = 0;

    for (const auto &g : gqa) {
      const int q_heads = g[0];
      const int kv_heads = g[1];
      for (int kv_seq_len : kv_lens) {
        for (int sliding_window : swas) {
          std::vector<float> q_host(q_len * q_heads * head_dim);
          std::vector<float> k_host(kv_seq_len * kv_heads * head_dim);
          std::vector<float> v_host(kv_seq_len * kv_heads * head_dim);
          for (auto &x : q_host) x = dist(rng);
          for (auto &x : k_host) x = dist(rng);
          for (auto &x : v_host) x = dist(rng);

          // CPU FP32 reference: non-causal attention with the symmetric
          // sliding-window mask matching drafter_attention.cu:91-104.
          std::vector<float> ref(q_len * q_heads * head_dim, 0.0f);
          for (int qp = 0; qp < q_len; ++qp) {
            const int q_abs = kv_seq_len - q_len + qp;
            for (int qh = 0; qh < q_heads; ++qh) {
              const int kvh = (qh * kv_heads) / q_heads;
              std::vector<float> scores(kv_seq_len, -INFINITY);
              for (int j = 0; j < kv_seq_len; ++j) {
                if (sliding_window > 0) {
                  const int delta = q_abs - j;
                  const int ad = delta < 0 ? -delta : delta;
                  if (ad > sliding_window) continue;
                }
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                  float q =
                      bf16_round(q_host[(qp * q_heads + qh) * head_dim + d]);
                  float kk =
                      bf16_round(k_host[(j * kv_heads + kvh) * head_dim + d]);
                  dot += q * kk;
                }
                scores[j] = dot * scale;
              }
              float m = -INFINITY;
              for (float s : scores) m = fmaxf(m, s);
              float sum = 0.0f;
              for (float &s : scores) {
                s = (m == -INFINITY) ? 0.0f : expf(s - m);
                sum += s;
              }
              for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < kv_seq_len; ++j) {
                  float vv =
                      bf16_round(v_host[(j * kv_heads + kvh) * head_dim + d]);
                  acc += scores[j] * vv;
                }
                ref[(qp * q_heads + qh) * head_dim + d] =
                    (sum > 0.0f) ? acc / sum : 0.0f;
              }
            }
          }

          qwen36_device_ptr_t q_dev = dev_alloc<__nv_bfloat16>(q_host.size());
          qwen36_device_ptr_t k_dev = dev_alloc<__nv_bfloat16>(k_host.size());
          qwen36_device_ptr_t v_dev = dev_alloc<__nv_bfloat16>(v_host.size());
          qwen36_device_ptr_t out_v1 = dev_alloc<__nv_bfloat16>(ref.size());
          qwen36_device_ptr_t out_fa = dev_alloc<__nv_bfloat16>(ref.size());
          copy_bf16(q_dev, q_host);
          copy_bf16(k_dev, k_host);
          copy_bf16(v_dev, v_host);

          qwen36_drafter_attention_block_spec_t spec{};
          spec.q_bf16 = q_dev;
          spec.k_bf16 = k_dev;
          spec.v_bf16 = v_dev;
          spec.q_len = q_len;
          spec.kv_seq_len = kv_seq_len;
          spec.q_heads = q_heads;
          spec.kv_heads = kv_heads;
          spec.head_dim = head_dim;
          spec.sliding_window = sliding_window;

          // v1 path: env unset -> FA self-gate returns NOT_IMPLEMENTED ->
          // dispatcher falls through to the v1 scalar kernel.
          unsetenv("QWEN36_DRAFTER_ATTENTION_FLASH");
          spec.output_bf16 = out_v1;
          must_status(qwen36_drafter_attention_block_bf16(&spec),
                      "drafter FA parity: v1 path");
          std::vector<float> got_v1 = read_bf16(out_v1, ref.size());

          // FA path: env set, call the FA entry directly.
          setenv("QWEN36_DRAFTER_ATTENTION_FLASH", "1", 1);
          spec.output_bf16 = out_fa;
          must_status(qwen36_drafter_attention_block_flash_bf16(&spec),
                      "drafter FA parity: flash path");
          std::vector<float> got_fa = read_bf16(out_fa, ref.size());
          unsetenv("QWEN36_DRAFTER_ATTENTION_FLASH");

#ifdef QWEN36_SMOKE_PROVE_FAIL
          // Fail-ability proof (P0 requirement): inject a perturbation
          // and confirm the gate turns RED. Compiled only under
          // -DQWEN36_SMOKE_PROVE_FAIL; never in the default build.
          for (size_t i = 0; i < got_fa.size(); ++i) {
            got_fa[i] += (i % 3 == 0) ? 0.25f : 0.0f;
          }
#endif

          const double cos_fa_ref = cos_sim(got_fa, ref);
          const double cos_v1_ref = cos_sim(got_v1, ref);
          const double cos_fa_v1 = cos_sim(got_fa, got_v1);

          if (cos_fa_ref < 0.998 || cos_v1_ref < 0.998 ||
              cos_fa_v1 < 0.998) {
            fprintf(stderr,
                    "drafter FA parity FAIL [q_heads=%d kv_heads=%d "
                    "kv=%d swa=%d]: cos(fa,ref)=%.6f cos(v1,ref)=%.6f "
                    "cos(fa,v1)=%.6f < 0.998\n",
                    q_heads, kv_heads, kv_seq_len, sliding_window,
                    cos_fa_ref, cos_v1_ref, cos_fa_v1);
            exit(1);
          }

          dev_free<__nv_bfloat16>(q_dev);
          dev_free<__nv_bfloat16>(k_dev);
          dev_free<__nv_bfloat16>(v_dev);
          dev_free<__nv_bfloat16>(out_v1);
          dev_free<__nv_bfloat16>(out_fa);
          ++cases;
        }
      }
    }
    printf("drafter FA parity gate passed (%d cases, q_len=16)\n", cases);
  }

  // ------------------------------------------------------------------
  // Flash-Decoding split-K prefill parity gate (P2).
  // ------------------------------------------------------------------
  // Compares the split-K flash kernel against the scalar GQA prefill
  // (qwen36_attention_prefill, the production reference at tokens<1024)
  // at the real verify shape (q_heads=24, kv_heads=4, head_dim=256), over
  // BOTH KV dtypes (BF16 + FP8 — FP8 is the production default), token
  // counts in the default-on redirect band [9,32], start_positions past
  // 2048, and intermediate split counts. The split-K must be a numerically
  // faithful (cos>=0.998) drop-in for the scalar attention. Coverage
  // mandated by the adversarial review (wf_a36ff789-8b8) before default-on.
  // For a kernel-vs-kernel parity test the exact FP8 codes don't matter —
  // both kernels decode the SAME bytes with byte-identical e4m3 logic — so
  // the FP8 cache is filled with valid small-magnitude random codes.
  {
    const int q_heads = 24;
    const int kv_heads = 4;
    const int head_dim = 256;
    const int token_counts[] = {9, 16, 32};
    const int starts[] = {0, 64, 2048, 4096};
    const int split_counts[] = {1, 8, 48};
    const int dtypes[] = {0, 1}; // 0=bf16, 1=fp8 e4m3

    qwen36_attention_shape_t shape{};
    shape.q_heads = q_heads;
    shape.kv_heads = kv_heads;
    shape.head_dim = head_dim;
    shape.rope_dims = 0;

    auto cos_sim = [](const std::vector<float> &a,
                      const std::vector<float> &b) -> double {
      double num = 0.0, da = 0.0, db = 0.0;
      for (size_t i = 0; i < a.size(); ++i) {
        num += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        da += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        db += static_cast<double>(b[i]) * static_cast<double>(b[i]);
      }
      return num / sqrt(da * db);
    };

    std::mt19937 rng(515151);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    int cases = 0;

    for (int kv_dtype : dtypes) {
      for (int tokens : token_counts) {
        for (int start_position : starts) {
          const int ctx = start_position + tokens;
          const size_t q_total = (size_t)tokens * q_heads * head_dim;
          const size_t kv_elems = (size_t)ctx * kv_heads * head_dim;

          std::vector<float> q_host(q_total);
          for (auto &x : q_host) x = dist(rng);
          qwen36_device_ptr_t q_dev = dev_alloc<__nv_bfloat16>(q_total);
          copy_bf16(q_dev, q_host);

          // KV cache: BF16 (2 bytes) or FP8 e4m3 (1 byte) per element.
          qwen36_device_ptr_t ck_dev, cv_dev;
          if (kv_dtype == 0) {
            std::vector<float> ck(kv_elems), cv(kv_elems);
            for (auto &x : ck) x = dist(rng);
            for (auto &x : cv) x = dist(rng);
            ck_dev = dev_alloc<__nv_bfloat16>(kv_elems);
            cv_dev = dev_alloc<__nv_bfloat16>(kv_elems);
            copy_bf16(ck_dev, ck);
            copy_bf16(cv_dev, cv);
          } else {
            // Valid e4m3 codes with exponent in [0,9] (|value| <~ 7.5) and
            // random sign+mantissa. Avoids the 0x7f/0xff -> 448 specials.
            std::uniform_int_distribution<int> ed(0, 9), md(0, 7), sd(0, 1);
            std::vector<uint8_t> ck(kv_elems), cv(kv_elems);
            for (auto &b : ck)
              b = (uint8_t)((sd(rng) << 7) | (ed(rng) << 3) | md(rng));
            for (auto &b : cv)
              b = (uint8_t)((sd(rng) << 7) | (ed(rng) << 3) | md(rng));
            ck_dev = dev_alloc<uint8_t>(kv_elems);
            cv_dev = dev_alloc<uint8_t>(kv_elems);
            copy_raw<uint8_t>(ck_dev, ck);
            copy_raw<uint8_t>(cv_dev, cv);
          }

          qwen36_device_ptr_t out_ref = dev_alloc<__nv_bfloat16>(q_total);
          qwen36_device_ptr_t out_sk = dev_alloc<__nv_bfloat16>(q_total);

          qwen36_attention_prefill_spec_t spec{};
          spec.start_position = start_position;
          spec.tokens = tokens;
          spec.q_bf16 = q_dev;
          spec.k_bf16 = q_dev; // unused by these kernels (read from cache)
          spec.v_bf16 = q_dev;
          spec.kv_cache_k = ck_dev;
          spec.kv_cache_v = cv_dev;
          spec.shape = shape;
          spec.kv_cache_dtype = kv_dtype;

          spec.output_bf16 = out_ref;
          must_status(qwen36_attention_prefill(&spec),
                      "splitk parity: scalar GQA reference");
          std::vector<float> ref = read_bf16(out_ref, q_total);

          for (int n_splits : split_counts) {
            const size_t pcount =
                (size_t)tokens * q_heads * (size_t)n_splits;
            // Two paths, both must match the scalar reference:
            //  A = fallback process-global scratch (partials NULL)
            //  B = engine-owned partials passed via the spec (production path)
            for (int path = 0; path < 2; ++path) {
              qwen36_device_ptr_t pacc{}, pmax{}, pden{};
              if (path == 1) {
                pacc = dev_alloc<float>(pcount * head_dim);
                pmax = dev_alloc<float>(pcount);
                pden = dev_alloc<float>(pcount);
              }
              spec.partial_acc_f32 = pacc;
              spec.partial_max_f32 = pmax;
              spec.partial_denom_f32 = pden;
              spec.output_bf16 = out_sk;
              must_status(
                  qwen36_attention_flash_splitk_prefill_bf16(&spec, n_splits),
                  "splitk parity: flash split-K");
              std::vector<float> got = read_bf16(out_sk, q_total);
              const double cos = cos_sim(got, ref);
              if (cos < 0.998) {
                fprintf(stderr,
                        "flash split-K parity FAIL [dtype=%d tokens=%d "
                        "start=%d n_splits=%d path=%s]: cos=%.6f < 0.998\n",
                        kv_dtype, tokens, start_position, n_splits,
                        path == 0 ? "scratch" : "engine", cos);
                exit(1);
              }
              if (path == 1) {
                dev_free<float>(pacc);
                dev_free<float>(pmax);
                dev_free<float>(pden);
              }
              ++cases;
            }
          }
          spec.partial_acc_f32 = qwen36_device_ptr_t{};
          spec.partial_max_f32 = qwen36_device_ptr_t{};
          spec.partial_denom_f32 = qwen36_device_ptr_t{};

          dev_free<__nv_bfloat16>(q_dev);
          if (kv_dtype == 0) {
            dev_free<__nv_bfloat16>(ck_dev);
            dev_free<__nv_bfloat16>(cv_dev);
          } else {
            dev_free<uint8_t>(ck_dev);
            dev_free<uint8_t>(cv_dev);
          }
          dev_free<__nv_bfloat16>(out_ref);
          dev_free<__nv_bfloat16>(out_sk);
        }
      }
    }
    printf("flash split-K parity gate passed (%d cases, BF16+FP8, "
           "tokens{9,16,32} starts{0,64,2048,4096} splits{1,8,48}, "
           "scratch+engine paths)\n",
           cases);
  }

  // ------------------------------------------------------------------
  // Tiled decode attention parity gate (decode long-context fix).
  // ------------------------------------------------------------------
  // Compares the register-tiled decode split kernel
  // (attention_decode_tiled.cu, QWEN36_DECODE_TILED_ATTENTION=1) against
  // the v1 scalar split-GQA kernel (=0) through the public
  // qwen36_attention_decode entry, at the production shape (q_heads=24,
  // kv_heads=4, head_dim=256) across positions, dtypes and engine-like
  // n_splits (incl. empty splits). Also asserts the cache-append side
  // effect (the kernel stores the current token's K/V) is byte-identical.
  {
    const int q_heads = 24;
    const int kv_heads = 4;
    const int head_dim = 256;
    const size_t positions[] = {255, 2047, 8191, 24575};
    const int dtypes[] = {0, 1}; // bf16, fp8

    qwen36_attention_shape_t shape{};
    shape.q_heads = q_heads;
    shape.kv_heads = kv_heads;
    shape.head_dim = head_dim;
    shape.rope_dims = 0;

    auto cos_sim = [](const std::vector<float> &a,
                      const std::vector<float> &b) -> double {
      double num = 0.0, da = 0.0, db = 0.0;
      for (size_t i = 0; i < a.size(); ++i) {
        num += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        da += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        db += static_cast<double>(b[i]) * static_cast<double>(b[i]);
      }
      return num / sqrt(da * db);
    };
    auto next_pow2 = [](size_t v) {
      size_t p = 1;
      while (p < v) p <<= 1;
      return p;
    };

    std::mt19937 rng(626262);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    int cases = 0;

    for (int kv_dtype : dtypes) {
      for (size_t position : positions) {
        const size_t ctx = position + 1;
        // Engine-like split sizing: power-of-two context bucket with a
        // 2048 floor, 64 timesteps per block (>= 32 splits, the GQA gate).
        const size_t bucket = next_pow2(ctx) < 2048 ? 2048 : next_pow2(ctx);
        const int n_splits = static_cast<int>(bucket / 64);
        const size_t kv_elems = ctx * kv_heads * head_dim;
        const size_t row_elems = (size_t)kv_heads * head_dim;

        std::vector<float> q_host((size_t)q_heads * head_dim);
        std::vector<float> knew_host(row_elems), vnew_host(row_elems);
        for (auto &x : q_host) x = dist(rng);
        for (auto &x : knew_host) x = dist(rng);
        for (auto &x : vnew_host) x = dist(rng);

        qwen36_device_ptr_t q_dev = dev_alloc<__nv_bfloat16>(q_host.size());
        qwen36_device_ptr_t knew_dev = dev_alloc<__nv_bfloat16>(row_elems);
        qwen36_device_ptr_t vnew_dev = dev_alloc<__nv_bfloat16>(row_elems);
        copy_bf16(q_dev, q_host);
        copy_bf16(knew_dev, knew_host);
        copy_bf16(vnew_dev, vnew_host);

        qwen36_device_ptr_t ck_dev, cv_dev;
        size_t row_bytes;
        if (kv_dtype == 0) {
          std::vector<float> ck(kv_elems), cv(kv_elems);
          for (auto &x : ck) x = dist(rng);
          for (auto &x : cv) x = dist(rng);
          ck_dev = dev_alloc<__nv_bfloat16>(kv_elems);
          cv_dev = dev_alloc<__nv_bfloat16>(kv_elems);
          copy_bf16(ck_dev, ck);
          copy_bf16(cv_dev, cv);
          row_bytes = row_elems * 2;
        } else {
          std::uniform_int_distribution<int> ed(0, 9), md(0, 7), sd(0, 1);
          std::vector<uint8_t> ck(kv_elems), cv(kv_elems);
          for (auto &b : ck)
            b = (uint8_t)((sd(rng) << 7) | (ed(rng) << 3) | md(rng));
          for (auto &b : cv)
            b = (uint8_t)((sd(rng) << 7) | (ed(rng) << 3) | md(rng));
          ck_dev = dev_alloc<uint8_t>(kv_elems);
          cv_dev = dev_alloc<uint8_t>(kv_elems);
          copy_raw<uint8_t>(ck_dev, ck);
          copy_raw<uint8_t>(cv_dev, cv);
          row_bytes = row_elems;
        }

        const size_t pcount = (size_t)q_heads * n_splits;
        qwen36_device_ptr_t pacc = dev_alloc<float>(pcount * head_dim);
        qwen36_device_ptr_t pmax = dev_alloc<float>(pcount);
        qwen36_device_ptr_t pden = dev_alloc<float>(pcount);
        qwen36_device_ptr_t out_dev =
            dev_alloc<__nv_bfloat16>((size_t)q_heads * head_dim);

        qwen36_attention_decode_spec_t spec{};
        spec.layer_index = 0;
        spec.position = position;
        spec.q_bf16 = q_dev;
        spec.k_bf16 = knew_dev;
        spec.v_bf16 = vnew_dev;
        spec.kv_cache_k = ck_dev;
        spec.kv_cache_v = cv_dev;
        spec.output_bf16 = out_dev;
        spec.shape = shape;
        spec.kv_cache_dtype = kv_dtype;
        spec.partial_acc_f32 = pacc;
        spec.partial_max_f32 = pmax;
        spec.partial_denom_f32 = pden;
        spec.decode_n_splits = (size_t)n_splits;
        spec.split_timesteps_per_block = 64;

        const size_t append_off = position * row_bytes; // row of all kv heads
        std::vector<uint8_t> app_k_ref(row_bytes), app_v_ref(row_bytes);
        std::vector<uint8_t> app_k_new(row_bytes), app_v_new(row_bytes);

        // v1 scalar reference.
        setenv("QWEN36_DECODE_TILED_ATTENTION", "0", 1);
        must_status(qwen36_attention_decode(&spec), "decode tiled parity: v1");
        std::vector<float> ref = read_bf16(out_dev, (size_t)q_heads * head_dim);
        must_cuda<int>(
            cudaMemcpy(app_k_ref.data(),
                       reinterpret_cast<const uint8_t *>(
                           static_cast<uintptr_t>(ck_dev.ptr)) +
                           append_off,
                       row_bytes, cudaMemcpyDeviceToHost),
            "append k ref");
        must_cuda<int>(
            cudaMemcpy(app_v_ref.data(),
                       reinterpret_cast<const uint8_t *>(
                           static_cast<uintptr_t>(cv_dev.ptr)) +
                           append_off,
                       row_bytes, cudaMemcpyDeviceToHost),
            "append v ref");

        // v2 tiled.
        setenv("QWEN36_DECODE_TILED_ATTENTION", "1", 1);
        must_status(qwen36_attention_decode(&spec), "decode tiled parity: v2");
        unsetenv("QWEN36_DECODE_TILED_ATTENTION");
        std::vector<float> got = read_bf16(out_dev, (size_t)q_heads * head_dim);
        must_cuda<int>(
            cudaMemcpy(app_k_new.data(),
                       reinterpret_cast<const uint8_t *>(
                           static_cast<uintptr_t>(ck_dev.ptr)) +
                           append_off,
                       row_bytes, cudaMemcpyDeviceToHost),
            "append k new");
        must_cuda<int>(
            cudaMemcpy(app_v_new.data(),
                       reinterpret_cast<const uint8_t *>(
                           static_cast<uintptr_t>(cv_dev.ptr)) +
                           append_off,
                       row_bytes, cudaMemcpyDeviceToHost),
            "append v new");

        const double cos = cos_sim(got, ref);
        if (cos < 0.998) {
          fprintf(stderr,
                  "decode tiled parity FAIL [dtype=%d pos=%zu n_splits=%d]: "
                  "cos=%.6f < 0.998\n",
                  kv_dtype, position, n_splits, cos);
          exit(1);
        }
        if (app_k_ref != app_k_new || app_v_ref != app_v_new) {
          fprintf(stderr,
                  "decode tiled parity FAIL [dtype=%d pos=%zu]: cache append "
                  "differs from v1\n",
                  kv_dtype, position);
          exit(1);
        }

        dev_free<__nv_bfloat16>(q_dev);
        dev_free<__nv_bfloat16>(knew_dev);
        dev_free<__nv_bfloat16>(vnew_dev);
        if (kv_dtype == 0) {
          dev_free<__nv_bfloat16>(ck_dev);
          dev_free<__nv_bfloat16>(cv_dev);
        } else {
          dev_free<uint8_t>(ck_dev);
          dev_free<uint8_t>(cv_dev);
        }
        dev_free<float>(pacc);
        dev_free<float>(pmax);
        dev_free<float>(pden);
        dev_free<__nv_bfloat16>(out_dev);
        ++cases;
      }
    }
    printf("decode tiled attention parity gate passed (%d cases, BF16+FP8, "
           "pos{255,2047,8191,24575}, append byte-identical)\n",
           cases);
  }

  printf("qwen36 CUDA smoke test passed\n");
  return 0;
}
