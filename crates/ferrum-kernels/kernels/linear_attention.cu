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

template <typename InputT>
static __device__ void linear_attention_prepare_impl(
    const InputT* __restrict__ mixed_qkv_raw,
    const InputT* __restrict__ conv_weight,
    const InputT* __restrict__ a_raw,
    const InputT* __restrict__ b_raw,
    const InputT* __restrict__ a_log,
    const InputT* __restrict__ dt_bias,
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
  linear_attention_prepare_impl<float>(
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
  linear_attention_prepare_impl<__half>(
      mixed_qkv_raw, conv_weight, a_raw, b_raw, a_log, dt_bias, query, key,
      value, g, beta, tokens, key_heads, value_heads, key_dim, value_dim,
      conv_kernel);
}

template <typename InputT>
static __device__ void linear_attention_decode_prepare_impl(
    const InputT* __restrict__ mixed_qkv_raw,
    const InputT* __restrict__ conv_weight,
    const float* __restrict__ conv_state,
    const InputT* __restrict__ a_raw,
    const InputT* __restrict__ b_raw,
    const InputT* __restrict__ a_log,
    const InputT* __restrict__ dt_bias,
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
  linear_attention_decode_prepare_impl<float>(
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
  linear_attention_decode_prepare_impl<__half>(
      mixed_qkv_raw, conv_weight, conv_state, a_raw, b_raw, a_log, dt_bias,
      query, key, value, g, beta, next_conv_state, key_heads, value_heads,
      key_dim, value_dim, conv_kernel);
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

extern "C" __global__ void gated_rms_norm_f32(
    const float* __restrict__ core,
    const float* __restrict__ z,
    const float* __restrict__ weight,
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
    out[base + d] = core[base + d] * inv * weight[d] * ferrum_silu(z[base + d]);
  }
}
