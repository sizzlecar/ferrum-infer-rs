#include <cuda_runtime.h>

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
          acc += mixed_qkv_raw[src_token * conv_channels + channel] *
                 conv_weight[channel * conv_kernel + kernel_idx];
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
    const float a = a_raw[idx] + dt_bias[head];
    g[idx] = -expf(a_log[head]) * ferrum_softplus(a);
    beta[idx] = ferrum_sigmoid(b_raw[idx]);
  }
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
