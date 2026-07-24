#include <cuda_fp16.h>
#include <cuda_runtime.h>

static __device__ __forceinline__ float ferrum_load_value(const float* ptr,
                                                          int idx) {
  return ptr[idx];
}

static __device__ __forceinline__ float ferrum_load_value(const __half* ptr,
                                                          int idx) {
  return __half2float(ptr[idx]);
}

static __device__ __forceinline__ void ferrum_store_value(float* ptr,
                                                          int idx,
                                                          float value) {
  ptr[idx] = value;
}

static __device__ __forceinline__ void ferrum_store_value(__half* ptr,
                                                          int idx,
                                                          float value) {
  ptr[idx] = __float2half(value);
}

static __device__ __forceinline__ float ferrum_sigmoid(float x) {
  if (x >= 0.0f) {
    const float z = expf(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(x);
  return z / (1.0f + z);
}

static __device__ __forceinline__ float ferrum_silu(float x) {
  return x * ferrum_sigmoid(x);
}

static __device__ __forceinline__ float ferrum_softplus(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return expf(x);
  return log1pf(expf(x));
}

template <typename InputT, typename ParamT>
static __device__ void linear_attention_prepare_impl(
    const InputT* __restrict__ mixed_qkv_raw,
    const InputT* __restrict__ conv_weight,
    const InputT* __restrict__ a_raw,
    const InputT* __restrict__ b_raw,
    const ParamT* __restrict__ a_log,
    const ParamT* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  const int qk_total = key_heads * key_dim;
  const int value_total = value_heads * value_dim;
  const int conv_channels = 2 * qk_total + value_total;
  const int conv_total = tokens * conv_channels;
  const int gate_total = tokens * value_heads;
  const int total = max(conv_total, gate_total);
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  if (idx < conv_total) {
    const int token = idx / conv_channels;
    const int channel = idx - token * conv_channels;
    const int pad = conv_kernel - 1;
    float acc = 0.0f;
    for (int kernel_idx = 0; kernel_idx < conv_kernel; ++kernel_idx) {
      const int padded = token + kernel_idx;
      if (padded >= pad) {
        const int src_token = padded - pad;
        if (src_token < tokens) {
          acc += ferrum_load_value(mixed_qkv_raw,
                                   src_token * conv_channels + channel) *
                 ferrum_load_value(conv_weight,
                                   channel * conv_kernel + kernel_idx);
        }
      }
    }
    const float conv = ferrum_silu(acc);
    if (channel < qk_total) {
      query[token * qk_total + channel] = conv;
    } else if (channel < 2 * qk_total) {
      key[token * qk_total + (channel - qk_total)] = conv;
    } else {
      value[token * value_total + (channel - 2 * qk_total)] = conv;
    }
  }

  if (idx < gate_total) {
    const int token = idx / value_heads;
    const int head = idx - token * value_heads;
    const float a = ferrum_load_value(a_raw, idx) +
                    ferrum_load_value(dt_bias, head);
    g[idx] = -expf(ferrum_load_value(a_log, head)) * ferrum_softplus(a);
    beta[idx] = ferrum_sigmoid(ferrum_load_value(b_raw, idx));
  }
}

extern "C" __global__ void linear_attention_prepare_f32(
    const float* __restrict__ mixed_qkv_raw,
    const float* __restrict__ conv_weight,
    const float* __restrict__ a_raw,
    const float* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_prepare_impl<float, float>(
      mixed_qkv_raw, conv_weight, a_raw, b_raw, a_log, dt_bias, query, key,
      value, g, beta, tokens, key_heads, value_heads, key_dim, value_dim,
      conv_kernel);
}

extern "C" __global__ void linear_attention_prepare_f16_to_f32(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const __half* __restrict__ a_log,
    const __half* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_prepare_impl<__half, __half>(
      mixed_qkv_raw, conv_weight, a_raw, b_raw, a_log, dt_bias, query, key,
      value, g, beta, tokens, key_heads, value_heads, key_dim, value_dim,
      conv_kernel);
}

extern "C" __global__ void linear_attention_prepare_f16_params_f32(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_prepare_impl<__half, float>(
      mixed_qkv_raw, conv_weight, a_raw, b_raw, a_log, dt_bias, query, key,
      value, g, beta, tokens, key_heads, value_heads, key_dim, value_dim,
      conv_kernel);
}

template <typename InputT, typename ParamT, typename StateT>
static __device__ void linear_attention_prepare_varlen_impl(
    const InputT* __restrict__ mixed_qkv_raw,
    const InputT* __restrict__ conv_weight,
    const StateT* __restrict__ initial_conv_states,
    const InputT* __restrict__ a_raw,
    const InputT* __restrict__ b_raw,
    const ParamT* __restrict__ a_log,
    const ParamT* __restrict__ dt_bias,
    const unsigned int* __restrict__ cu_seqlens,
    const unsigned int* __restrict__ token_seq_indices,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    StateT* __restrict__ final_conv_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  const int qk_total = key_heads * key_dim;
  const int value_total = value_heads * value_dim;
  const int conv_channels = 2 * qk_total + value_total;
  const int state_len = conv_kernel - 1;
  const int conv_state_len = conv_channels * state_len;
  const int conv_total = total_tokens * conv_channels;
  const int gate_total = total_tokens * value_heads;
  const int state_total = batch * conv_state_len;
  const int total = max(max(conv_total, gate_total), state_total);
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  if (idx < conv_total) {
    const int token = idx / conv_channels;
    const int channel = idx - token * conv_channels;
    const int seq = static_cast<int>(token_seq_indices[token]);
    if (seq < 0 || seq >= batch) return;
    const int token_start = static_cast<int>(cu_seqlens[seq]);
    const int local_token = token - token_start;
    const int state_base = seq * conv_state_len + channel * state_len;

    float acc = 0.0f;
    for (int kernel_idx = 0; kernel_idx < conv_kernel; ++kernel_idx) {
      const int source = local_token + kernel_idx - state_len;
      const float x = source >= 0
                          ? ferrum_load_value(
                                mixed_qkv_raw,
                                (token_start + source) * conv_channels + channel)
                          : ferrum_load_value(
                                initial_conv_states,
                                state_base + state_len + source);
      acc += x *
             ferrum_load_value(conv_weight,
                               channel * conv_kernel + kernel_idx);
    }
    const float conv = ferrum_silu(acc);
    if (channel < qk_total) {
      query[token * qk_total + channel] = conv;
    } else if (channel < 2 * qk_total) {
      key[token * qk_total + (channel - qk_total)] = conv;
    } else {
      value[token * value_total + (channel - 2 * qk_total)] = conv;
    }
  }

  if (idx < gate_total) {
    const int token = idx / value_heads;
    const int head = idx - token * value_heads;
    const float a = ferrum_load_value(a_raw, idx) +
                    ferrum_load_value(dt_bias, head);
    g[idx] = -expf(ferrum_load_value(a_log, head)) * ferrum_softplus(a);
    beta[idx] = ferrum_sigmoid(ferrum_load_value(b_raw, idx));
  }

  if (idx < state_total) {
    const int seq = idx / conv_state_len;
    const int state_offset = idx - seq * conv_state_len;
    const int channel = state_offset / state_len;
    const int pos = state_offset - channel * state_len;
    const int token_start = static_cast<int>(cu_seqlens[seq]);
    const int token_end = static_cast<int>(cu_seqlens[seq + 1]);
    const int seq_tokens = token_end - token_start;
    const int source = seq_tokens + pos - state_len;
    const int state_base = seq * conv_state_len + channel * state_len;
    const float final_value =
        source >= 0
            ? ferrum_load_value(mixed_qkv_raw,
                                (token_start + source) * conv_channels + channel)
            : ferrum_load_value(initial_conv_states,
                                state_base + state_len + source);
    ferrum_store_value(final_conv_states, idx, final_value);
  }
}

extern "C" __global__ void linear_attention_prepare_varlen_f32(
    const float* __restrict__ mixed_qkv_raw,
    const float* __restrict__ conv_weight,
    const float* __restrict__ initial_conv_states,
    const float* __restrict__ a_raw,
    const float* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const unsigned int* __restrict__ cu_seqlens,
    const unsigned int* __restrict__ token_seq_indices,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ final_conv_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_prepare_varlen_impl<float, float, float>(
      mixed_qkv_raw, conv_weight, initial_conv_states, a_raw, b_raw, a_log,
      dt_bias, cu_seqlens, token_seq_indices, query, key, value, g, beta,
      final_conv_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_prepare_varlen_f16_to_f32(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    const float* __restrict__ initial_conv_states,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const __half* __restrict__ a_log,
    const __half* __restrict__ dt_bias,
    const unsigned int* __restrict__ cu_seqlens,
    const unsigned int* __restrict__ token_seq_indices,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ final_conv_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_prepare_varlen_impl<__half, __half, float>(
      mixed_qkv_raw, conv_weight, initial_conv_states, a_raw, b_raw, a_log,
      dt_bias, cu_seqlens, token_seq_indices, query, key, value, g, beta,
      final_conv_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_prepare_varlen_f16_params_f32(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    const float* __restrict__ initial_conv_states,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const unsigned int* __restrict__ cu_seqlens,
    const unsigned int* __restrict__ token_seq_indices,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ final_conv_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_prepare_varlen_impl<__half, float, float>(
      mixed_qkv_raw, conv_weight, initial_conv_states, a_raw, b_raw, a_log,
      dt_bias, cu_seqlens, token_seq_indices, query, key, value, g, beta,
      final_conv_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_prepare_varlen_f16_to_f32_state_f16(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    const __half* __restrict__ initial_conv_states,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const __half* __restrict__ a_log,
    const __half* __restrict__ dt_bias,
    const unsigned int* __restrict__ cu_seqlens,
    const unsigned int* __restrict__ token_seq_indices,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    __half* __restrict__ final_conv_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_prepare_varlen_impl<__half, __half, __half>(
      mixed_qkv_raw, conv_weight, initial_conv_states, a_raw, b_raw, a_log,
      dt_bias, cu_seqlens, token_seq_indices, query, key, value, g, beta,
      final_conv_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_prepare_varlen_f16_params_f32_state_f16(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    const __half* __restrict__ initial_conv_states,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const unsigned int* __restrict__ cu_seqlens,
    const unsigned int* __restrict__ token_seq_indices,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    __half* __restrict__ final_conv_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_prepare_varlen_impl<__half, float, __half>(
      mixed_qkv_raw, conv_weight, initial_conv_states, a_raw, b_raw, a_log,
      dt_bias, cu_seqlens, token_seq_indices, query, key, value, g, beta,
      final_conv_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_prepare_varlen_f16_params_f32_state_f16_indirect(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    const unsigned long long* __restrict__ state_bindings,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const unsigned int* __restrict__ cu_seqlens,
    const unsigned int* __restrict__ token_seq_indices,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    __half* __restrict__ final_conv_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  const __half* initial_conv_states =
      reinterpret_cast<const __half*>(state_bindings[0]);
  linear_attention_prepare_varlen_impl<__half, float, __half>(
      mixed_qkv_raw, conv_weight, initial_conv_states, a_raw, b_raw, a_log,
      dt_bias, cu_seqlens, token_seq_indices, query, key, value, g, beta,
      final_conv_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, conv_kernel);
}

extern "C" __global__ void recurrent_conv_state_commit_f16_indirect(
    const __half* __restrict__ source,
    const unsigned long long* __restrict__ state_bindings,
    const int elements) {
  __half* destination = reinterpret_cast<__half*>(state_bindings[0]);
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    destination[index] = source[index];
  }
}

template <typename InputT, typename ParamT>
static __device__ void linear_attention_prepare_varlen_packed_qkvz_ba_impl(
    const InputT* __restrict__ mixed_qkvz_raw,
    const InputT* __restrict__ ba_raw,
    const InputT* __restrict__ conv_weight,
    const float* __restrict__ initial_conv_states,
    const ParamT* __restrict__ a_log,
    const ParamT* __restrict__ dt_bias,
    const unsigned int* __restrict__ cu_seqlens,
    const unsigned int* __restrict__ token_seq_indices,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ z,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ final_conv_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  const int qk_total = key_heads * key_dim;
  const int value_total = value_heads * value_dim;
  const int conv_channels = 2 * qk_total + value_total;
  const int qkvz_width = conv_channels + value_total;
  const int ba_width = 2 * value_heads;
  const int state_len = conv_kernel - 1;
  const int conv_state_len = conv_channels * state_len;
  const int conv_total = total_tokens * conv_channels;
  const int z_total = total_tokens * value_total;
  const int gate_total = total_tokens * value_heads;
  const int state_total = batch * conv_state_len;
  const int total = max(max(max(conv_total, z_total), gate_total), state_total);
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  if (idx < conv_total) {
    const int token = idx / conv_channels;
    const int channel = idx - token * conv_channels;
    const int seq = static_cast<int>(token_seq_indices[token]);
    if (seq < 0 || seq >= batch) return;
    const int token_start = static_cast<int>(cu_seqlens[seq]);
    const int local_token = token - token_start;
    const int state_base = seq * conv_state_len + channel * state_len;

    float acc = 0.0f;
    for (int kernel_idx = 0; kernel_idx < conv_kernel; ++kernel_idx) {
      const int source = local_token + kernel_idx - state_len;
      const float x = source >= 0
                          ? ferrum_load_value(
                                mixed_qkvz_raw,
                                (token_start + source) * qkvz_width + channel)
                          : initial_conv_states[state_base + state_len + source];
      acc += x *
             ferrum_load_value(conv_weight,
                               channel * conv_kernel + kernel_idx);
    }
    const float conv = ferrum_silu(acc);
    if (channel < qk_total) {
      query[token * qk_total + channel] = conv;
    } else if (channel < 2 * qk_total) {
      key[token * qk_total + (channel - qk_total)] = conv;
    } else {
      value[token * value_total + (channel - 2 * qk_total)] = conv;
    }
  }

  if (idx < z_total) {
    const int token = idx / value_total;
    const int offset = idx - token * value_total;
    z[idx] = ferrum_load_value(
        mixed_qkvz_raw, token * qkvz_width + conv_channels + offset);
  }

  if (idx < gate_total) {
    const int token = idx / value_heads;
    const int head = idx - token * value_heads;
    const int ba_base = token * ba_width;
    const float b_raw = ferrum_load_value(ba_raw, ba_base + head);
    const float a_raw = ferrum_load_value(ba_raw, ba_base + value_heads + head);
    const float a = a_raw + ferrum_load_value(dt_bias, head);
    g[idx] = -expf(ferrum_load_value(a_log, head)) * ferrum_softplus(a);
    beta[idx] = ferrum_sigmoid(b_raw);
  }

  if (idx < state_total) {
    const int seq = idx / conv_state_len;
    const int state_offset = idx - seq * conv_state_len;
    const int channel = state_offset / state_len;
    const int pos = state_offset - channel * state_len;
    const int token_start = static_cast<int>(cu_seqlens[seq]);
    const int token_end = static_cast<int>(cu_seqlens[seq + 1]);
    const int seq_tokens = token_end - token_start;
    const int source = seq_tokens + pos - state_len;
    const int state_base = seq * conv_state_len + channel * state_len;
    final_conv_states[idx] =
        source >= 0
            ? ferrum_load_value(mixed_qkvz_raw,
                                (token_start + source) * qkvz_width + channel)
            : initial_conv_states[state_base + state_len + source];
  }
}

extern "C" __global__ void linear_attention_prepare_varlen_packed_qkvz_ba_f32(
    const float* __restrict__ mixed_qkvz_raw,
    const float* __restrict__ ba_raw,
    const float* __restrict__ conv_weight,
    const float* __restrict__ initial_conv_states,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const unsigned int* __restrict__ cu_seqlens,
    const unsigned int* __restrict__ token_seq_indices,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ z,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ final_conv_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_prepare_varlen_packed_qkvz_ba_impl<float, float>(
      mixed_qkvz_raw, ba_raw, conv_weight, initial_conv_states, a_log, dt_bias,
      cu_seqlens, token_seq_indices, query, key, value, z, g, beta,
      final_conv_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_prepare_varlen_packed_qkvz_ba_f16_to_f32(
    const __half* __restrict__ mixed_qkvz_raw,
    const __half* __restrict__ ba_raw,
    const __half* __restrict__ conv_weight,
    const float* __restrict__ initial_conv_states,
    const __half* __restrict__ a_log,
    const __half* __restrict__ dt_bias,
    const unsigned int* __restrict__ cu_seqlens,
    const unsigned int* __restrict__ token_seq_indices,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ z,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ final_conv_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_prepare_varlen_packed_qkvz_ba_impl<__half, __half>(
      mixed_qkvz_raw, ba_raw, conv_weight, initial_conv_states, a_log, dt_bias,
      cu_seqlens, token_seq_indices, query, key, value, z, g, beta,
      final_conv_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_prepare_varlen_packed_qkvz_ba_f16_params_f32(
    const __half* __restrict__ mixed_qkvz_raw,
    const __half* __restrict__ ba_raw,
    const __half* __restrict__ conv_weight,
    const float* __restrict__ initial_conv_states,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const unsigned int* __restrict__ cu_seqlens,
    const unsigned int* __restrict__ token_seq_indices,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ z,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ final_conv_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_prepare_varlen_packed_qkvz_ba_impl<__half, float>(
      mixed_qkvz_raw, ba_raw, conv_weight, initial_conv_states, a_log, dt_bias,
      cu_seqlens, token_seq_indices, query, key, value, z, g, beta,
      final_conv_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, conv_kernel);
}

template <typename InputT, typename ParamT>
static __device__ void linear_attention_decode_prepare_impl(
    const InputT* __restrict__ mixed_qkv_raw,
    const InputT* __restrict__ conv_weight,
    const float* __restrict__ conv_state,
    const InputT* __restrict__ a_raw,
    const InputT* __restrict__ b_raw,
    const ParamT* __restrict__ a_log,
    const ParamT* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ next_conv_state,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  const int qk_total = key_heads * key_dim;
  const int value_total = value_heads * value_dim;
  const int conv_channels = 2 * qk_total + value_total;
  const int state_len = conv_kernel - 1;
  const int total = max(conv_channels, value_heads);
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  if (idx < conv_channels) {
    const int channel = idx;
    const int state_base = channel * state_len;
    float acc = 0.0f;
    for (int kernel_idx = 0; kernel_idx < conv_kernel; ++kernel_idx) {
      const float x = kernel_idx < state_len
                          ? conv_state[state_base + kernel_idx]
                          : ferrum_load_value(mixed_qkv_raw, channel);
      acc += x *
             ferrum_load_value(conv_weight,
                               channel * conv_kernel + kernel_idx);
    }

    if (state_len > 0) {
      for (int pos = 0; pos < state_len; ++pos) {
        next_conv_state[state_base + pos] =
            (pos + 1 < state_len) ? conv_state[state_base + pos + 1]
                                  : ferrum_load_value(mixed_qkv_raw, channel);
      }
    }

    const float conv = ferrum_silu(acc);
    if (channel < qk_total) {
      query[channel] = conv;
    } else if (channel < 2 * qk_total) {
      key[channel - qk_total] = conv;
    } else {
      value[channel - 2 * qk_total] = conv;
    }
  }

  if (idx < value_heads) {
    const float a = ferrum_load_value(a_raw, idx) +
                    ferrum_load_value(dt_bias, idx);
    g[idx] = -expf(ferrum_load_value(a_log, idx)) * ferrum_softplus(a);
    beta[idx] = ferrum_sigmoid(ferrum_load_value(b_raw, idx));
  }
}

extern "C" __global__ void linear_attention_decode_prepare_f32(
    const float* __restrict__ mixed_qkv_raw,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_state,
    const float* __restrict__ a_raw,
    const float* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ next_conv_state,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_impl<float, float>(
      mixed_qkv_raw, conv_weight, conv_state, a_raw, b_raw, a_log, dt_bias,
      query, key, value, g, beta, next_conv_state, key_heads, value_heads,
      key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_f16_to_f32(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    const float* __restrict__ conv_state,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const __half* __restrict__ a_log,
    const __half* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ next_conv_state,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_impl<__half, __half>(
      mixed_qkv_raw, conv_weight, conv_state, a_raw, b_raw, a_log, dt_bias,
      query, key, value, g, beta, next_conv_state, key_heads, value_heads,
      key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_f16_params_f32(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    const float* __restrict__ conv_state,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ next_conv_state,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_impl<__half, float>(
      mixed_qkv_raw, conv_weight, conv_state, a_raw, b_raw, a_log, dt_bias,
      query, key, value, g, beta, next_conv_state, key_heads, value_heads,
      key_dim, value_dim, conv_kernel);
}

template <typename InputT, typename ParamT>
static __device__ void linear_attention_decode_prepare_batch_impl(
    const InputT* __restrict__ mixed_qkv_raw,
    const InputT* __restrict__ conv_weight,
    const float* __restrict__ conv_states,
    const InputT* __restrict__ a_raw,
    const InputT* __restrict__ b_raw,
    const ParamT* __restrict__ a_log,
    const ParamT* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ next_conv_states,
    const int batch,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  const int qk_total = key_heads * key_dim;
  const int value_total = value_heads * value_dim;
  const int conv_channels = 2 * qk_total + value_total;
  const int state_len = conv_kernel - 1;
  const int conv_state_len = conv_channels * state_len;
  const int row_total = max(conv_channels, value_heads);
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * row_total;
  if (global >= total) return;

  const int row = global / row_total;
  const int idx = global - row * row_total;

  if (idx < conv_channels) {
    const int channel = idx;
    const int state_base = row * conv_state_len + channel * state_len;
    const int input_base = row * conv_channels;
    float acc = 0.0f;
    for (int kernel_idx = 0; kernel_idx < conv_kernel; ++kernel_idx) {
      const float x = kernel_idx < state_len
                          ? conv_states[state_base + kernel_idx]
                          : ferrum_load_value(mixed_qkv_raw,
                                              input_base + channel);
      acc += x *
             ferrum_load_value(conv_weight,
                               channel * conv_kernel + kernel_idx);
    }

    if (state_len > 0) {
      for (int pos = 0; pos < state_len; ++pos) {
        next_conv_states[state_base + pos] =
            (pos + 1 < state_len)
                ? conv_states[state_base + pos + 1]
                : ferrum_load_value(mixed_qkv_raw, input_base + channel);
      }
    }

    const float conv = ferrum_silu(acc);
    if (channel < qk_total) {
      query[row * qk_total + channel] = conv;
    } else if (channel < 2 * qk_total) {
      key[row * qk_total + (channel - qk_total)] = conv;
    } else {
      value[row * value_total + (channel - 2 * qk_total)] = conv;
    }
  }

  if (idx < value_heads) {
    const int gate_idx = row * value_heads + idx;
    const float a = ferrum_load_value(a_raw, gate_idx) +
                    ferrum_load_value(dt_bias, idx);
    g[gate_idx] = -expf(ferrum_load_value(a_log, idx)) *
                  ferrum_softplus(a);
    beta[gate_idx] = ferrum_sigmoid(ferrum_load_value(b_raw, gate_idx));
  }
}

extern "C" __global__ void linear_attention_decode_prepare_batch_f32(
    const float* __restrict__ mixed_qkv_raw,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_states,
    const float* __restrict__ a_raw,
    const float* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ next_conv_states,
    const int batch,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_impl<float, float>(
      mixed_qkv_raw, conv_weight, conv_states, a_raw, b_raw, a_log, dt_bias,
      query, key, value, g, beta, next_conv_states, batch, key_heads,
      value_heads, key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_f16_to_f32(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    const float* __restrict__ conv_states,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const __half* __restrict__ a_log,
    const __half* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ next_conv_states,
    const int batch,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_impl<__half, __half>(
      mixed_qkv_raw, conv_weight, conv_states, a_raw, b_raw, a_log, dt_bias,
      query, key, value, g, beta, next_conv_states, batch, key_heads,
      value_heads, key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_f16_params_f32(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    const float* __restrict__ conv_states,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    float* __restrict__ next_conv_states,
    const int batch,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_impl<__half, float>(
      mixed_qkv_raw, conv_weight, conv_states, a_raw, b_raw, a_log, dt_bias,
      query, key, value, g, beta, next_conv_states, batch, key_heads,
      value_heads, key_dim, value_dim, conv_kernel);
}

template <typename InputT, typename ParamT, typename StateT>
static __device__ void linear_attention_decode_prepare_batch_indexed_impl(
    const InputT* __restrict__ mixed_qkv_raw,
    const InputT* __restrict__ conv_weight,
    StateT* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const InputT* __restrict__ a_raw,
    const InputT* __restrict__ b_raw,
    const ParamT* __restrict__ a_log,
    const ParamT* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  const int qk_total = key_heads * key_dim;
  const int value_total = value_heads * value_dim;
  const int conv_channels = 2 * qk_total + value_total;
  const int state_len = conv_kernel - 1;
  const int conv_state_len = conv_channels * state_len;
  const int row_total = max(conv_channels, value_heads);
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * row_total;
  if (global >= total) return;

  const int row = global / row_total;
  const int idx = global - row * row_total;
  const unsigned int slot_u = slot_indices[row];
  if (slot_u >= static_cast<unsigned int>(max_slots)) return;
  const int slot = static_cast<int>(slot_u);

  if (idx < conv_channels) {
    const int channel = idx;
    const int state_base = slot * conv_state_len + channel * state_len;
    const int input_base = row * conv_channels;
    float acc = 0.0f;
    for (int kernel_idx = 0; kernel_idx < conv_kernel; ++kernel_idx) {
      const float x = kernel_idx < state_len
                          ? ferrum_load_value(conv_state_slots,
                                              state_base + kernel_idx)
                          : ferrum_load_value(mixed_qkv_raw,
                                              input_base + channel);
      acc += x *
             ferrum_load_value(conv_weight,
                               channel * conv_kernel + kernel_idx);
    }

    if (state_len > 0) {
      for (int pos = 0; pos < state_len; ++pos) {
        ferrum_store_value(
            conv_state_slots, state_base + pos,
            (pos + 1 < state_len)
                ? ferrum_load_value(conv_state_slots, state_base + pos + 1)
                : ferrum_load_value(mixed_qkv_raw, input_base + channel));
      }
    }

    const float conv = ferrum_silu(acc);
    if (channel < qk_total) {
      query[row * qk_total + channel] = conv;
    } else if (channel < 2 * qk_total) {
      key[row * qk_total + (channel - qk_total)] = conv;
    } else {
      value[row * value_total + (channel - 2 * qk_total)] = conv;
    }
  }

  if (idx < value_heads) {
    const int gate_idx = row * value_heads + idx;
    const float a = ferrum_load_value(a_raw, gate_idx) +
                    ferrum_load_value(dt_bias, idx);
    g[gate_idx] = -expf(ferrum_load_value(a_log, idx)) *
                  ferrum_softplus(a);
    beta[gate_idx] = ferrum_sigmoid(ferrum_load_value(b_raw, gate_idx));
  }
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_f32(
    const float* __restrict__ mixed_qkv_raw,
    const float* __restrict__ conv_weight,
    float* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const float* __restrict__ a_raw,
    const float* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_impl<float, float, float>(
      mixed_qkv_raw, conv_weight, conv_state_slots, slot_indices, a_raw, b_raw,
      a_log, dt_bias, query, key, value, g, beta, batch, max_slots, key_heads,
      value_heads, key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_f32_state_f16(
    const float* __restrict__ mixed_qkv_raw,
    const float* __restrict__ conv_weight,
    __half* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const float* __restrict__ a_raw,
    const float* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_impl<float, float, __half>(
      mixed_qkv_raw, conv_weight, conv_state_slots, slot_indices, a_raw, b_raw,
      a_log, dt_bias, query, key, value, g, beta, batch, max_slots, key_heads,
      value_heads, key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_f16_to_f32(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    float* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const __half* __restrict__ a_log,
    const __half* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_impl<__half, __half, float>(
      mixed_qkv_raw, conv_weight, conv_state_slots, slot_indices, a_raw, b_raw,
      a_log, dt_bias, query, key, value, g, beta, batch, max_slots, key_heads,
      value_heads, key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_f16_to_f32_state_f16(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    __half* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const __half* __restrict__ a_log,
    const __half* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_impl<__half, __half, __half>(
      mixed_qkv_raw, conv_weight, conv_state_slots, slot_indices, a_raw, b_raw,
      a_log, dt_bias, query, key, value, g, beta, batch, max_slots, key_heads,
      value_heads, key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_f16_params_f32(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    float* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_impl<__half, float, float>(
      mixed_qkv_raw, conv_weight, conv_state_slots, slot_indices, a_raw, b_raw,
      a_log, dt_bias, query, key, value, g, beta, batch, max_slots, key_heads,
      value_heads, key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_f16_params_f32_state_f16(
    const __half* __restrict__ mixed_qkv_raw,
    const __half* __restrict__ conv_weight,
    __half* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const __half* __restrict__ a_raw,
    const __half* __restrict__ b_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_impl<__half, float, __half>(
      mixed_qkv_raw, conv_weight, conv_state_slots, slot_indices, a_raw, b_raw,
      a_log, dt_bias, query, key, value, g, beta, batch, max_slots, key_heads,
      value_heads, key_dim, value_dim, conv_kernel);
}

template <typename InputT, typename ParamT, typename StateT>
static __device__ void linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_impl(
    const InputT* __restrict__ mixed_qkvz_raw,
    const InputT* __restrict__ ba_raw,
    const InputT* __restrict__ conv_weight,
    StateT* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const ParamT* __restrict__ a_log,
    const ParamT* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ z,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  const int qk_total = key_heads * key_dim;
  const int value_total = value_heads * value_dim;
  const int conv_channels = 2 * qk_total + value_total;
  const int qkvz_width = conv_channels + value_total;
  const int ba_width = 2 * value_heads;
  const int state_len = conv_kernel - 1;
  const int conv_state_len = conv_channels * state_len;
  const int row_total = max(max(conv_channels, value_total), value_heads);
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * row_total;
  if (global >= total) return;

  const int row = global / row_total;
  const int idx = global - row * row_total;
  const unsigned int slot_u = slot_indices[row];
  if (slot_u >= static_cast<unsigned int>(max_slots)) return;
  const int slot = static_cast<int>(slot_u);

  if (idx < conv_channels) {
    const int channel = idx;
    const int state_base = slot * conv_state_len + channel * state_len;
    const int input_base = row * qkvz_width;
    float acc = 0.0f;
    for (int kernel_idx = 0; kernel_idx < conv_kernel; ++kernel_idx) {
      const float x = kernel_idx < state_len
                          ? ferrum_load_value(conv_state_slots,
                                              state_base + kernel_idx)
                          : ferrum_load_value(mixed_qkvz_raw,
                                              input_base + channel);
      acc += x *
             ferrum_load_value(conv_weight,
                               channel * conv_kernel + kernel_idx);
    }

    if (state_len > 0) {
      for (int pos = 0; pos < state_len; ++pos) {
        ferrum_store_value(
            conv_state_slots, state_base + pos,
            (pos + 1 < state_len)
                ? ferrum_load_value(conv_state_slots, state_base + pos + 1)
                : ferrum_load_value(mixed_qkvz_raw, input_base + channel));
      }
    }

    const float conv = ferrum_silu(acc);
    if (channel < qk_total) {
      query[row * qk_total + channel] = conv;
    } else if (channel < 2 * qk_total) {
      key[row * qk_total + (channel - qk_total)] = conv;
    } else {
      value[row * value_total + (channel - 2 * qk_total)] = conv;
    }
  }

  if (idx < value_total) {
    z[row * value_total + idx] =
        ferrum_load_value(mixed_qkvz_raw, row * qkvz_width + conv_channels + idx);
  }

  if (idx < value_heads) {
    const int gate_idx = row * value_heads + idx;
    const int ba_base = row * ba_width;
    const float b_raw = ferrum_load_value(ba_raw, ba_base + idx);
    const float a_raw = ferrum_load_value(ba_raw, ba_base + value_heads + idx);
    const float a = a_raw + ferrum_load_value(dt_bias, idx);
    g[gate_idx] = -expf(ferrum_load_value(a_log, idx)) *
                  ferrum_softplus(a);
    beta[gate_idx] = ferrum_sigmoid(b_raw);
  }
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f32(
    const float* __restrict__ mixed_qkvz_raw,
    const float* __restrict__ ba_raw,
    const float* __restrict__ conv_weight,
    float* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ z,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_impl<float, float, float>(
      mixed_qkvz_raw, ba_raw, conv_weight, conv_state_slots, slot_indices,
      a_log, dt_bias, query, key, value, z, g, beta, batch, max_slots,
      key_heads, value_heads, key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f32_state_f16(
    const float* __restrict__ mixed_qkvz_raw,
    const float* __restrict__ ba_raw,
    const float* __restrict__ conv_weight,
    __half* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ z,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_impl<float, float, __half>(
      mixed_qkvz_raw, ba_raw, conv_weight, conv_state_slots, slot_indices,
      a_log, dt_bias, query, key, value, z, g, beta, batch, max_slots,
      key_heads, value_heads, key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f16_to_f32(
    const __half* __restrict__ mixed_qkvz_raw,
    const __half* __restrict__ ba_raw,
    const __half* __restrict__ conv_weight,
    float* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const __half* __restrict__ a_log,
    const __half* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ z,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_impl<__half, __half, float>(
      mixed_qkvz_raw, ba_raw, conv_weight, conv_state_slots, slot_indices,
      a_log, dt_bias, query, key, value, z, g, beta, batch, max_slots,
      key_heads, value_heads, key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f16_to_f32_state_f16(
    const __half* __restrict__ mixed_qkvz_raw,
    const __half* __restrict__ ba_raw,
    const __half* __restrict__ conv_weight,
    __half* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const __half* __restrict__ a_log,
    const __half* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ z,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_impl<__half, __half, __half>(
      mixed_qkvz_raw, ba_raw, conv_weight, conv_state_slots, slot_indices,
      a_log, dt_bias, query, key, value, z, g, beta, batch, max_slots,
      key_heads, value_heads, key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f16_params_f32(
    const __half* __restrict__ mixed_qkvz_raw,
    const __half* __restrict__ ba_raw,
    const __half* __restrict__ conv_weight,
    float* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ z,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_impl<__half, float, float>(
      mixed_qkvz_raw, ba_raw, conv_weight, conv_state_slots, slot_indices,
      a_log, dt_bias, query, key, value, z, g, beta, batch, max_slots,
      key_heads, value_heads, key_dim, value_dim, conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f16_params_f32_state_f16(
    const __half* __restrict__ mixed_qkvz_raw,
    const __half* __restrict__ ba_raw,
    const __half* __restrict__ conv_weight,
    __half* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ z,
    float* __restrict__ g,
    float* __restrict__ beta,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_impl<__half, float, __half>(
      mixed_qkvz_raw, ba_raw, conv_weight, conv_state_slots, slot_indices,
      a_log, dt_bias, query, key, value, z, g, beta, batch, max_slots,
      key_heads, value_heads, key_dim, value_dim, conv_kernel);
}

template <typename InputT, typename StateT>
static __device__ void linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_impl(
    const InputT* __restrict__ mixed_qkvz_raw,
    const InputT* __restrict__ conv_weight,
    StateT* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ mixed_qkv,
    float* __restrict__ z,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  const int qk_total = key_heads * key_dim;
  const int value_total = value_heads * value_dim;
  const int conv_channels = 2 * qk_total + value_total;
  const int qkvz_width = conv_channels + value_total;
  const int state_len = conv_kernel - 1;
  const int conv_state_len = conv_channels * state_len;
  const int row_total = max(conv_channels, value_total);
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * row_total;
  if (global >= total) return;

  const int row = global / row_total;
  const int idx = global - row * row_total;
  const unsigned int slot_u = slot_indices[row];
  if (slot_u >= static_cast<unsigned int>(max_slots)) return;
  const int slot = static_cast<int>(slot_u);

  if (idx < conv_channels) {
    const int channel = idx;
    const int state_base = slot * conv_state_len + channel * state_len;
    const int input_base = row * qkvz_width;
    float acc = 0.0f;
    for (int kernel_idx = 0; kernel_idx < conv_kernel; ++kernel_idx) {
      const float x = kernel_idx < state_len
                          ? ferrum_load_value(conv_state_slots,
                                              state_base + kernel_idx)
                          : ferrum_load_value(mixed_qkvz_raw,
                                              input_base + channel);
      acc += x *
             ferrum_load_value(conv_weight,
                               channel * conv_kernel + kernel_idx);
    }

    if (state_len > 0) {
      for (int pos = 0; pos < state_len; ++pos) {
        ferrum_store_value(
            conv_state_slots, state_base + pos,
            (pos + 1 < state_len)
                ? ferrum_load_value(conv_state_slots, state_base + pos + 1)
                : ferrum_load_value(mixed_qkvz_raw, input_base + channel));
      }
    }

    mixed_qkv[row * conv_channels + channel] = ferrum_silu(acc);
  }

  if (idx < value_total) {
    z[row * value_total + idx] =
        ferrum_load_value(mixed_qkvz_raw, row * qkvz_width + conv_channels + idx);
  }
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f32(
    const float* __restrict__ mixed_qkvz_raw,
    const float* __restrict__ conv_weight,
    float* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ mixed_qkv,
    float* __restrict__ z,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_impl<float, float>(
      mixed_qkvz_raw, conv_weight, conv_state_slots, slot_indices, mixed_qkv,
      z, batch, max_slots, key_heads, value_heads, key_dim, value_dim,
      conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f32_state_f16(
    const float* __restrict__ mixed_qkvz_raw,
    const float* __restrict__ conv_weight,
    __half* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ mixed_qkv,
    float* __restrict__ z,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_impl<float, __half>(
      mixed_qkvz_raw, conv_weight, conv_state_slots, slot_indices, mixed_qkv,
      z, batch, max_slots, key_heads, value_heads, key_dim, value_dim,
      conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f16_to_f32(
    const __half* __restrict__ mixed_qkvz_raw,
    const __half* __restrict__ conv_weight,
    float* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ mixed_qkv,
    float* __restrict__ z,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_impl<__half, float>(
      mixed_qkvz_raw, conv_weight, conv_state_slots, slot_indices, mixed_qkv,
      z, batch, max_slots, key_heads, value_heads, key_dim, value_dim,
      conv_kernel);
}

extern "C" __global__ void linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f16_to_f32_state_f16(
    const __half* __restrict__ mixed_qkvz_raw,
    const __half* __restrict__ conv_weight,
    __half* __restrict__ conv_state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ mixed_qkv,
    float* __restrict__ z,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int conv_kernel) {
  linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_impl<__half, __half>(
      mixed_qkvz_raw, conv_weight, conv_state_slots, slot_indices, mixed_qkv,
      z, batch, max_slots, key_heads, value_heads, key_dim, value_dim,
      conv_kernel);
}

extern "C" __global__ void linear_attention_qk_l2norm_f32(
    float* __restrict__ query,
    float* __restrict__ key,
    const int tokens,
    const int key_heads,
    const int key_dim,
    const float eps) {
  const int row = blockIdx.x;
  if (row >= tokens * key_heads) return;
  const int base = row * key_dim;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int d = threadIdx.x; d < key_dim; d += blockDim.x) {
    const float q = query[base + d];
    const float k = key[base + d];
    q_sum += q * q;
    k_sum += k * k;
  }

  __shared__ float q_shared[256];
  __shared__ float k_shared[256];
  q_shared[threadIdx.x] = q_sum;
  k_shared[threadIdx.x] = k_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      q_shared[threadIdx.x] += q_shared[threadIdx.x + stride];
      k_shared[threadIdx.x] += k_shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float q_inv = rsqrtf(q_shared[0] + eps);
  const float k_inv = rsqrtf(k_shared[0] + eps);
  for (int d = threadIdx.x; d < key_dim; d += blockDim.x) {
    query[base + d] *= q_inv;
    key[base + d] *= k_inv;
  }
}

template <typename InputT, typename WeightT>
static __device__ void gated_rms_norm_impl(
    const float* __restrict__ core,
    const InputT* __restrict__ z,
    const WeightT* __restrict__ weight,
    float* __restrict__ out,
    const int rows,
    const int dim,
    const float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int base = row * dim;

  float sum_sq = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float x = core[base + d];
    sum_sq += x * x;
  }

  __shared__ float shared[256];
  shared[threadIdx.x] = sum_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(shared[0] / static_cast<float>(dim) + eps);
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    out[base + d] = core[base + d] * inv * ferrum_load_value(weight, d) *
                    ferrum_silu(ferrum_load_value(z, base + d));
  }
}

extern "C" __global__ void gated_rms_norm_f32(
    const float* __restrict__ core,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    float* __restrict__ out,
    const int rows,
    const int dim,
    const float eps) {
  gated_rms_norm_impl<float, float>(core, z, weight, out, rows, dim, eps);
}

extern "C" __global__ void gated_rms_norm_f16_to_f32(
    const float* __restrict__ core,
    const __half* __restrict__ z,
    const __half* __restrict__ weight,
    float* __restrict__ out,
    const int rows,
    const int dim,
    const float eps) {
  gated_rms_norm_impl<__half, __half>(core, z, weight, out, rows, dim, eps);
}

extern "C" __global__ void gated_rms_norm_f16_z_f32_weight(
    const float* __restrict__ core,
    const __half* __restrict__ z,
    const float* __restrict__ weight,
    float* __restrict__ out,
    const int rows,
    const int dim,
    const float eps) {
  gated_rms_norm_impl<__half, float>(core, z, weight, out, rows, dim, eps);
}
