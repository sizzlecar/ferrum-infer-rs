#include <cuda_fp16.h>
#include <cuda_runtime.h>

static __device__ __forceinline__ float ferrum_gdr_load_value(const float* ptr,
                                                              int idx) {
  return ptr[idx];
}

static __device__ __forceinline__ float ferrum_gdr_load_value(const __half* ptr,
                                                              int idx) {
  return __half2float(ptr[idx]);
}

static __device__ __forceinline__ void ferrum_gdr_store_value(float* ptr,
                                                              int idx,
                                                              float value) {
  ptr[idx] = value;
}

static __device__ __forceinline__ void ferrum_gdr_store_value(__half* ptr,
                                                              int idx,
                                                              float value) {
  ptr[idx] = __float2half(value);
}

static __device__ __forceinline__ float ferrum_gdr_sigmoid(float x) {
  if (x >= 0.0f) {
    const float z = expf(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(x);
  return z / (1.0f + z);
}

static __device__ __forceinline__ float ferrum_gdr_softplus(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return expf(x);
  return log1pf(expf(x));
}

extern "C" __global__ void recurrent_gated_delta_rule_f32(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const float* __restrict__ initial_state,
    float* __restrict__ out,
    float* __restrict__ final_state,
    const int tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int use_qk_l2norm,
    const float scale) {
  const int value_head = blockIdx.x;
  if (value_head >= value_heads) return;

  const int repeat_factor = value_heads / key_heads;
  const int key_head = value_head / repeat_factor;

  for (int value_offset = threadIdx.x; value_offset < value_dim; value_offset += blockDim.x) {
    const int state_base = (value_head * value_dim + value_offset) * key_dim;

    for (int kd = 0; kd < key_dim; ++kd) {
      final_state[state_base + kd] = initial_state[state_base + kd];
    }

    for (int token = 0; token < tokens; ++token) {
      float q_inv = 1.0f;
      float k_inv = 1.0f;
      if (use_qk_l2norm != 0) {
        float q_norm = 0.0f;
        float k_norm = 0.0f;
        for (int kd = 0; kd < key_dim; ++kd) {
          const int qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
          const float qv = query[qk_idx];
          const float kv = key[qk_idx];
          q_norm += qv * qv;
          k_norm += kv * kv;
        }
        q_inv = rsqrtf(q_norm + 1.0e-6f);
        k_inv = rsqrtf(k_norm + 1.0e-6f);
      }

      const int gate_idx = token * value_heads + value_head;
      const float decay = expf(g[gate_idx]);
      const float beta_t = beta[gate_idx];

      float kv_mem = 0.0f;
      for (int kd = 0; kd < key_dim; ++kd) {
        const int state_idx = state_base + kd;
        const int qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
        final_state[state_idx] *= decay;
        kv_mem += final_state[state_idx] * (key[qk_idx] * k_inv);
      }

      const int value_idx = ((token * value_heads + value_head) * value_dim) + value_offset;
      const float delta = (value[value_idx] - kv_mem) * beta_t;
      for (int kd = 0; kd < key_dim; ++kd) {
        const int state_idx = state_base + kd;
        const int qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
        final_state[state_idx] += delta * (key[qk_idx] * k_inv);
      }

      float acc = 0.0f;
      for (int kd = 0; kd < key_dim; ++kd) {
        const int state_idx = state_base + kd;
        const int qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
        acc += final_state[state_idx] * (query[qk_idx] * q_inv * scale);
      }
      out[value_idx] = acc;
    }
  }
}

extern "C" __global__ void recurrent_gated_delta_rule_batch_f32(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const float* __restrict__ initial_states,
    float* __restrict__ out,
    float* __restrict__ final_states,
    const int batch,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int use_qk_l2norm,
    const float scale) {
  const int value_head = blockIdx.x;
  const int row = blockIdx.y;
  if (value_head >= value_heads || row >= batch) return;

  const int repeat_factor = value_heads / key_heads;
  const int key_head = value_head / repeat_factor;
  const int state_len = value_heads * value_dim * key_dim;
  const int row_state_base = row * state_len;

  for (int value_offset = threadIdx.x; value_offset < value_dim;
       value_offset += blockDim.x) {
    const int state_base =
        row_state_base + (value_head * value_dim + value_offset) * key_dim;

    for (int kd = 0; kd < key_dim; ++kd) {
      final_states[state_base + kd] = initial_states[state_base + kd];
    }

    float q_inv = 1.0f;
    float k_inv = 1.0f;
    if (use_qk_l2norm != 0) {
      float q_norm = 0.0f;
      float k_norm = 0.0f;
      for (int kd = 0; kd < key_dim; ++kd) {
        const int qk_idx = ((row * key_heads + key_head) * key_dim) + kd;
        const float qv = query[qk_idx];
        const float kv = key[qk_idx];
        q_norm += qv * qv;
        k_norm += kv * kv;
      }
      q_inv = rsqrtf(q_norm + 1.0e-6f);
      k_inv = rsqrtf(k_norm + 1.0e-6f);
    }

    const int gate_idx = row * value_heads + value_head;
    const float decay = expf(g[gate_idx]);
    const float beta_t = beta[gate_idx];

    float kv_mem = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
      const int state_idx = state_base + kd;
      const int qk_idx = ((row * key_heads + key_head) * key_dim) + kd;
      final_states[state_idx] *= decay;
      kv_mem += final_states[state_idx] * (key[qk_idx] * k_inv);
    }

    const int value_idx =
        ((row * value_heads + value_head) * value_dim) + value_offset;
    const float delta = (value[value_idx] - kv_mem) * beta_t;
    for (int kd = 0; kd < key_dim; ++kd) {
      const int state_idx = state_base + kd;
      const int qk_idx = ((row * key_heads + key_head) * key_dim) + kd;
      final_states[state_idx] += delta * (key[qk_idx] * k_inv);
    }

    float acc = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
      const int state_idx = state_base + kd;
      const int qk_idx = ((row * key_heads + key_head) * key_dim) + kd;
      acc += final_states[state_idx] * (query[qk_idx] * q_inv * scale);
    }
    out[value_idx] = acc;
  }
}

extern "C" __global__ void recurrent_gated_delta_rule_batch_indexed_f32(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ out,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int use_qk_l2norm,
    const float scale) {
  const int value_head = blockIdx.x;
  const int row = blockIdx.y;
  if (value_head >= value_heads || row >= batch) return;

  const unsigned int slot_u = slot_indices[row];
  if (slot_u >= static_cast<unsigned int>(max_slots)) return;
  const int slot = static_cast<int>(slot_u);

  const int repeat_factor = value_heads / key_heads;
  const int key_head = value_head / repeat_factor;
  const int state_len = value_heads * value_dim * key_dim;
  const int row_state_base = slot * state_len;

  for (int value_offset = threadIdx.x; value_offset < value_dim;
       value_offset += blockDim.x) {
    const int state_base =
        row_state_base + (value_head * value_dim + value_offset) * key_dim;

    float q_inv = 1.0f;
    float k_inv = 1.0f;
    if (use_qk_l2norm != 0) {
      float q_norm = 0.0f;
      float k_norm = 0.0f;
      for (int kd = 0; kd < key_dim; ++kd) {
        const int qk_idx = ((row * key_heads + key_head) * key_dim) + kd;
        const float qv = query[qk_idx];
        const float kv = key[qk_idx];
        q_norm += qv * qv;
        k_norm += kv * kv;
      }
      q_inv = rsqrtf(q_norm + 1.0e-6f);
      k_inv = rsqrtf(k_norm + 1.0e-6f);
    }

    const int gate_idx = row * value_heads + value_head;
    const float decay = expf(g[gate_idx]);
    const float beta_t = beta[gate_idx];

    float kv_mem = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
      const int state_idx = state_base + kd;
      const int qk_idx = ((row * key_heads + key_head) * key_dim) + kd;
      state_slots[state_idx] *= decay;
      kv_mem += state_slots[state_idx] * (key[qk_idx] * k_inv);
    }

    const int value_idx =
        ((row * value_heads + value_head) * value_dim) + value_offset;
    const float delta = (value[value_idx] - kv_mem) * beta_t;
    for (int kd = 0; kd < key_dim; ++kd) {
      const int state_idx = state_base + kd;
      const int qk_idx = ((row * key_heads + key_head) * key_dim) + kd;
      state_slots[state_idx] += delta * (key[qk_idx] * k_inv);
    }

    float acc = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
      const int state_idx = state_base + kd;
      const int qk_idx = ((row * key_heads + key_head) * key_dim) + kd;
      acc += state_slots[state_idx] * (query[qk_idx] * q_inv * scale);
    }
    out[value_idx] = acc;
  }
}

template <typename StateT, int BV_TILE>
static __device__ void recurrent_gated_delta_rule_batch_indexed_tiled_f32_impl(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    StateT* __restrict__ state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ out,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  const int value_tile = blockIdx.x;
  const int value_head = blockIdx.y;
  const int row = blockIdx.z;
  if (value_head >= value_heads || row >= batch) return;

  const unsigned int slot_u = slot_indices[row];
  if (slot_u >= static_cast<unsigned int>(max_slots)) return;
  const int slot = static_cast<int>(slot_u);

  const int value_start = value_tile * BV_TILE;
  const int repeat_factor = value_heads / key_heads;
  const int key_head = value_head / repeat_factor;
  const int state_len = value_heads * value_dim * key_dim;
  const int row_state_base = slot * state_len;
  const int gate_idx = row * value_heads + value_head;
  const float decay = expf(g[gate_idx]);
  const float beta_t = beta[gate_idx];

  __shared__ float partial[BV_TILE][256];
  __shared__ float delta[BV_TILE];
  float local[BV_TILE];

#pragma unroll
  for (int i = 0; i < BV_TILE; ++i) {
    local[i] = 0.0f;
  }

  for (int i = threadIdx.x; i < BV_TILE * key_dim; i += blockDim.x) {
    const int local_v = i / key_dim;
    const int kd = i - local_v * key_dim;
    const int value_offset = value_start + local_v;
    if (value_offset >= value_dim) continue;

    const int state_idx =
        row_state_base + (value_head * value_dim + value_offset) * key_dim + kd;
    const int qk_idx = ((row * key_heads + key_head) * key_dim) + kd;
    const float state = ferrum_gdr_load_value(state_slots, state_idx) * decay;
    ferrum_gdr_store_value(state_slots, state_idx, state);
    local[local_v] += state * key[qk_idx];
  }

#pragma unroll
  for (int local_v = 0; local_v < BV_TILE; ++local_v) {
    partial[local_v][threadIdx.x] = local[local_v];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int local_v = 0; local_v < BV_TILE; ++local_v) {
        partial[local_v][threadIdx.x] += partial[local_v][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int local_v = 0; local_v < BV_TILE; ++local_v) {
      const int value_offset = value_start + local_v;
      if (value_offset < value_dim) {
        const int value_idx =
            ((row * value_heads + value_head) * value_dim) + value_offset;
        delta[local_v] = (value[value_idx] - partial[local_v][0]) * beta_t;
      } else {
        delta[local_v] = 0.0f;
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < BV_TILE; ++i) {
    local[i] = 0.0f;
  }

  for (int i = threadIdx.x; i < BV_TILE * key_dim; i += blockDim.x) {
    const int local_v = i / key_dim;
    const int kd = i - local_v * key_dim;
    const int value_offset = value_start + local_v;
    if (value_offset >= value_dim) continue;

    const int state_idx =
        row_state_base + (value_head * value_dim + value_offset) * key_dim + kd;
    const int qk_idx = ((row * key_heads + key_head) * key_dim) + kd;
    const float updated =
        ferrum_gdr_load_value(state_slots, state_idx) + delta[local_v] * key[qk_idx];
    ferrum_gdr_store_value(state_slots, state_idx, updated);
    local[local_v] += updated * (query[qk_idx] * scale);
  }

#pragma unroll
  for (int local_v = 0; local_v < BV_TILE; ++local_v) {
    partial[local_v][threadIdx.x] = local[local_v];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int local_v = 0; local_v < BV_TILE; ++local_v) {
        partial[local_v][threadIdx.x] += partial[local_v][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int local_v = 0; local_v < BV_TILE; ++local_v) {
      const int value_offset = value_start + local_v;
      if (value_offset < value_dim) {
        const int value_idx =
            ((row * value_heads + value_head) * value_dim) + value_offset;
        out[value_idx] = partial[local_v][0];
      }
    }
  }
}

extern "C" __global__ void recurrent_gated_delta_rule_batch_indexed_tiled16_f32(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ out,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  recurrent_gated_delta_rule_batch_indexed_tiled_f32_impl<float, 16>(
      query, key, value, g, beta, state_slots, slot_indices, out, batch,
      max_slots, key_heads, value_heads, key_dim, value_dim, scale);
}

extern "C" __global__ void recurrent_gated_delta_rule_batch_indexed_tiled16_state_f16(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    __half* __restrict__ state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ out,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  recurrent_gated_delta_rule_batch_indexed_tiled_f32_impl<__half, 16>(
      query, key, value, g, beta, state_slots, slot_indices, out, batch,
      max_slots, key_heads, value_heads, key_dim, value_dim, scale);
}

template <typename GateT, typename ParamT, typename StateT, int BV_TILE,
          bool INDIRECT_STATE, bool QK_PRENORMALIZED>
static __device__ void recurrent_gated_delta_rule_batch_indexed_packed_tiled_f32_impl(
    const float* __restrict__ mixed_qkv,
    const GateT* __restrict__ ba_raw,
    const ParamT* __restrict__ a_log,
    const ParamT* __restrict__ dt_bias,
    StateT* __restrict__ direct_state_slots,
    const unsigned long long* __restrict__ state_bindings,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ out,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  const int value_tile = blockIdx.x;
  const int value_head = blockIdx.y;
  const int row = blockIdx.z;
  if (value_head >= value_heads || row >= batch) return;

  StateT* state_slots = direct_state_slots;
  int slot = 0;
  if (INDIRECT_STATE) {
    if (row != 0) return;
    state_slots = reinterpret_cast<StateT*>(state_bindings[1]);
  } else {
    const unsigned int slot_u = slot_indices[row];
    if (slot_u >= static_cast<unsigned int>(max_slots)) return;
    slot = static_cast<int>(slot_u);
  }

  const int value_start = value_tile * BV_TILE;
  const int repeat_factor = value_heads / key_heads;
  const int key_head = value_head / repeat_factor;
  const int qk_total = key_heads * key_dim;
  const int value_total = value_heads * value_dim;
  const int conv_channels = 2 * qk_total + value_total;
  const int state_len = value_heads * value_dim * key_dim;
  const int row_state_base = slot * state_len;
  const int row_mixed_base = row * conv_channels;
  const int ba_base = row * 2 * value_heads;

  __shared__ float partial[BV_TILE][256];
  __shared__ float delta[BV_TILE];
  float q_inv = 1.0f;
  float k_inv = 1.0f;
  if constexpr (!QK_PRENORMALIZED) {
    float q_norm_local = 0.0f;
    float k_norm_local = 0.0f;
    for (int kd = threadIdx.x; kd < key_dim; kd += blockDim.x) {
      const int q_idx = row_mixed_base + key_head * key_dim + kd;
      const int k_idx =
          row_mixed_base + qk_total + key_head * key_dim + kd;
      const float q = mixed_qkv[q_idx];
      const float k = mixed_qkv[k_idx];
      q_norm_local += q * q;
      k_norm_local += k * k;
    }

    __shared__ float reduce_q[256];
    __shared__ float reduce_k[256];
    __shared__ float q_inv_shared;
    __shared__ float k_inv_shared;
    reduce_q[threadIdx.x] = q_norm_local;
    reduce_k[threadIdx.x] = k_norm_local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduce_q[threadIdx.x] += reduce_q[threadIdx.x + stride];
        reduce_k[threadIdx.x] += reduce_k[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      q_inv_shared = rsqrtf(reduce_q[0] + 1.0e-6f);
      k_inv_shared = rsqrtf(reduce_k[0] + 1.0e-6f);
    }
    __syncthreads();
    q_inv = q_inv_shared;
    k_inv = k_inv_shared;
  }

  const float b_raw = ferrum_gdr_load_value(ba_raw, ba_base + value_head);
  const float a_raw =
      ferrum_gdr_load_value(ba_raw, ba_base + value_heads + value_head);
  const float a = a_raw + ferrum_gdr_load_value(dt_bias, value_head);
  const float g_val = -expf(ferrum_gdr_load_value(a_log, value_head)) *
                      ferrum_gdr_softplus(a);
  const float decay = expf(g_val);
  const float beta_t = ferrum_gdr_sigmoid(b_raw);

  float local[BV_TILE];

#pragma unroll
  for (int i = 0; i < BV_TILE; ++i) {
    local[i] = 0.0f;
  }

  for (int i = threadIdx.x; i < BV_TILE * key_dim; i += blockDim.x) {
    const int local_v = i / key_dim;
    const int kd = i - local_v * key_dim;
    const int value_offset = value_start + local_v;
    if (value_offset >= value_dim) continue;

    const int state_idx =
        row_state_base + (value_head * value_dim + value_offset) * key_dim + kd;
    const int k_idx = row_mixed_base + qk_total + key_head * key_dim + kd;
    const float k = mixed_qkv[k_idx] * k_inv;
    const float state = ferrum_gdr_load_value(state_slots, state_idx) * decay;
    ferrum_gdr_store_value(state_slots, state_idx, state);
    local[local_v] += state * k;
  }

#pragma unroll
  for (int local_v = 0; local_v < BV_TILE; ++local_v) {
    partial[local_v][threadIdx.x] = local[local_v];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int local_v = 0; local_v < BV_TILE; ++local_v) {
        partial[local_v][threadIdx.x] += partial[local_v][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int local_v = 0; local_v < BV_TILE; ++local_v) {
      const int value_offset = value_start + local_v;
      if (value_offset < value_dim) {
        const int value_idx =
            row_mixed_base + 2 * qk_total + value_head * value_dim + value_offset;
        delta[local_v] = (mixed_qkv[value_idx] - partial[local_v][0]) * beta_t;
      } else {
        delta[local_v] = 0.0f;
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < BV_TILE; ++i) {
    local[i] = 0.0f;
  }

  for (int i = threadIdx.x; i < BV_TILE * key_dim; i += blockDim.x) {
    const int local_v = i / key_dim;
    const int kd = i - local_v * key_dim;
    const int value_offset = value_start + local_v;
    if (value_offset >= value_dim) continue;

    const int state_idx =
        row_state_base + (value_head * value_dim + value_offset) * key_dim + kd;
    const int q_idx = row_mixed_base + key_head * key_dim + kd;
    const int k_idx = row_mixed_base + qk_total + key_head * key_dim + kd;
    const float k = mixed_qkv[k_idx] * k_inv;
    const float q = mixed_qkv[q_idx] * q_inv * scale;
    const float updated = ferrum_gdr_load_value(state_slots, state_idx) + delta[local_v] * k;
    ferrum_gdr_store_value(state_slots, state_idx, updated);
    local[local_v] += updated * q;
  }

#pragma unroll
  for (int local_v = 0; local_v < BV_TILE; ++local_v) {
    partial[local_v][threadIdx.x] = local[local_v];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int local_v = 0; local_v < BV_TILE; ++local_v) {
        partial[local_v][threadIdx.x] += partial[local_v][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int local_v = 0; local_v < BV_TILE; ++local_v) {
      const int value_offset = value_start + local_v;
      if (value_offset < value_dim) {
        const int value_idx =
            ((row * value_heads + value_head) * value_dim) + value_offset;
        out[value_idx] = partial[local_v][0];
      }
    }
  }
}

extern "C" __global__ void recurrent_gated_delta_rule_batch_indexed_packed_f32_ba_f32_params_f32(
    const float* __restrict__ mixed_qkv,
    const float* __restrict__ ba_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ out,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  recurrent_gated_delta_rule_batch_indexed_packed_tiled_f32_impl<
      float, float, float, 16, false, false>(
      mixed_qkv, ba_raw, a_log, dt_bias, state_slots, nullptr, slot_indices,
      out, batch, max_slots, key_heads, value_heads, key_dim, value_dim, scale);
}

extern "C" __global__ void recurrent_gated_delta_rule_batch_indexed_packed_f32_ba_f16_params_f16(
    const float* __restrict__ mixed_qkv,
    const __half* __restrict__ ba_raw,
    const __half* __restrict__ a_log,
    const __half* __restrict__ dt_bias,
    float* __restrict__ state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ out,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  recurrent_gated_delta_rule_batch_indexed_packed_tiled_f32_impl<
      __half, __half, float, 16, false, false>(
      mixed_qkv, ba_raw, a_log, dt_bias, state_slots, nullptr, slot_indices,
      out, batch, max_slots, key_heads, value_heads, key_dim, value_dim, scale);
}

extern "C" __global__ void recurrent_gated_delta_rule_batch_indexed_packed_f32_ba_f16_params_f32(
    const float* __restrict__ mixed_qkv,
    const __half* __restrict__ ba_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ out,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  recurrent_gated_delta_rule_batch_indexed_packed_tiled_f32_impl<
      __half, float, float, 16, false, false>(
      mixed_qkv, ba_raw, a_log, dt_bias, state_slots, nullptr, slot_indices,
      out, batch, max_slots, key_heads, value_heads, key_dim, value_dim, scale);
}

extern "C" __global__ void recurrent_gated_delta_rule_batch_indexed_packed_f32_ba_f32_params_f32_state_f16(
    const float* __restrict__ mixed_qkv,
    const float* __restrict__ ba_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    __half* __restrict__ state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ out,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  recurrent_gated_delta_rule_batch_indexed_packed_tiled_f32_impl<
      float, float, __half, 16, false, false>(
      mixed_qkv, ba_raw, a_log, dt_bias, state_slots, nullptr, slot_indices,
      out, batch, max_slots, key_heads, value_heads, key_dim, value_dim, scale);
}

extern "C" __global__ void recurrent_gated_delta_rule_batch_indexed_packed_f32_ba_f16_params_f16_state_f16(
    const float* __restrict__ mixed_qkv,
    const __half* __restrict__ ba_raw,
    const __half* __restrict__ a_log,
    const __half* __restrict__ dt_bias,
    __half* __restrict__ state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ out,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  recurrent_gated_delta_rule_batch_indexed_packed_tiled_f32_impl<
      __half, __half, __half, 16, false, false>(
      mixed_qkv, ba_raw, a_log, dt_bias, state_slots, nullptr, slot_indices,
      out, batch, max_slots, key_heads, value_heads, key_dim, value_dim, scale);
}

extern "C" __global__ void recurrent_gated_delta_rule_batch_indexed_packed_f32_ba_f16_params_f32_state_f16(
    const float* __restrict__ mixed_qkv,
    const __half* __restrict__ ba_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    __half* __restrict__ state_slots,
    const unsigned int* __restrict__ slot_indices,
    float* __restrict__ out,
    const int batch,
    const int max_slots,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  recurrent_gated_delta_rule_batch_indexed_packed_tiled_f32_impl<
      __half, float, __half, 16, false, false>(
      mixed_qkv, ba_raw, a_log, dt_bias, state_slots, nullptr, slot_indices,
      out, batch, max_slots, key_heads, value_heads, key_dim, value_dim, scale);
}

extern "C" __global__ void
recurrent_gated_delta_rule_decode_prenormalized_packed_f32_ba_f16_params_f32_indirect(
    const float* __restrict__ mixed_qkv,
    const __half* __restrict__ ba_raw,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const unsigned long long* __restrict__ state_bindings,
    float* __restrict__ out,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  recurrent_gated_delta_rule_batch_indexed_packed_tiled_f32_impl<
      __half, float, float, 16, true, true>(
      mixed_qkv, ba_raw, a_log, dt_bias, nullptr, state_bindings, nullptr, out,
      1, 1, key_heads, value_heads, key_dim, value_dim, scale);
}

template <typename StateT>
static __device__ void recurrent_gated_delta_rule_varlen_f32_impl(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const StateT* __restrict__ initial_states,
    const unsigned int* __restrict__ cu_seqlens,
    float* __restrict__ out,
    StateT* __restrict__ final_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int use_qk_l2norm,
    const float scale) {
  const int value_head = blockIdx.x;
  const int seq = blockIdx.y;
  if (value_head >= value_heads || seq >= batch) return;

  const int token_start = static_cast<int>(cu_seqlens[seq]);
  const int token_end = static_cast<int>(cu_seqlens[seq + 1]);
  if (token_start < 0 || token_end <= token_start || token_end > total_tokens) {
    return;
  }

  const int repeat_factor = value_heads / key_heads;
  const int key_head = value_head / repeat_factor;
  const int state_len = value_heads * value_dim * key_dim;
  const int row_state_base = seq * state_len;

  for (int value_offset = threadIdx.x; value_offset < value_dim;
       value_offset += blockDim.x) {
    const int state_base =
        row_state_base + (value_head * value_dim + value_offset) * key_dim;

    for (int kd = 0; kd < key_dim; ++kd) {
      ferrum_gdr_store_value(
          final_states, state_base + kd,
          ferrum_gdr_load_value(initial_states, state_base + kd));
    }

    for (int token = token_start; token < token_end; ++token) {
      float q_inv = 1.0f;
      float k_inv = 1.0f;
      if (use_qk_l2norm != 0) {
        float q_norm = 0.0f;
        float k_norm = 0.0f;
        for (int kd = 0; kd < key_dim; ++kd) {
          const int qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
          const float qv = query[qk_idx];
          const float kv = key[qk_idx];
          q_norm += qv * qv;
          k_norm += kv * kv;
        }
        q_inv = rsqrtf(q_norm + 1.0e-6f);
        k_inv = rsqrtf(k_norm + 1.0e-6f);
      }

      const int gate_idx = token * value_heads + value_head;
      const float decay = expf(g[gate_idx]);
      const float beta_t = beta[gate_idx];

      float kv_mem = 0.0f;
      for (int kd = 0; kd < key_dim; ++kd) {
        const int state_idx = state_base + kd;
        const int qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
        const float state =
            ferrum_gdr_load_value(final_states, state_idx) * decay;
        ferrum_gdr_store_value(final_states, state_idx, state);
        kv_mem += state * (key[qk_idx] * k_inv);
      }

      const int value_idx =
          ((token * value_heads + value_head) * value_dim) + value_offset;
      const float delta = (value[value_idx] - kv_mem) * beta_t;
      for (int kd = 0; kd < key_dim; ++kd) {
        const int state_idx = state_base + kd;
        const int qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
        const float updated = ferrum_gdr_load_value(final_states, state_idx) +
                              delta * (key[qk_idx] * k_inv);
        ferrum_gdr_store_value(final_states, state_idx, updated);
      }

      float acc = 0.0f;
      for (int kd = 0; kd < key_dim; ++kd) {
        const int state_idx = state_base + kd;
        const int qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
        acc += ferrum_gdr_load_value(final_states, state_idx) *
               (query[qk_idx] * q_inv * scale);
      }
      out[value_idx] = acc;
    }
  }
}

extern "C" __global__ void recurrent_gated_delta_rule_varlen_f32(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const float* __restrict__ initial_states,
    const unsigned int* __restrict__ cu_seqlens,
    float* __restrict__ out,
    float* __restrict__ final_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int use_qk_l2norm,
    const float scale) {
  recurrent_gated_delta_rule_varlen_f32_impl<float>(
      query, key, value, g, beta, initial_states, cu_seqlens, out,
      final_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, use_qk_l2norm, scale);
}

extern "C" __global__ void recurrent_gated_delta_rule_varlen_f32_indirect(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const unsigned long long* __restrict__ state_bindings,
    const unsigned int* __restrict__ cu_seqlens,
    float* __restrict__ out,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int use_qk_l2norm,
    const float scale) {
  float* state = reinterpret_cast<float*>(state_bindings[1]);
  recurrent_gated_delta_rule_varlen_f32_impl<float>(
      query, key, value, g, beta, state, cu_seqlens, out, state, batch,
      total_tokens, key_heads, value_heads, key_dim, value_dim,
      use_qk_l2norm, scale);
}

extern "C" __global__ void recurrent_gated_delta_rule_varlen_f32_state_f16(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const __half* __restrict__ initial_states,
    const unsigned int* __restrict__ cu_seqlens,
    float* __restrict__ out,
    __half* __restrict__ final_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const int use_qk_l2norm,
    const float scale) {
  recurrent_gated_delta_rule_varlen_f32_impl<__half>(
      query, key, value, g, beta, initial_states, cu_seqlens, out,
      final_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, use_qk_l2norm, scale);
}

template <typename StateT, int BV_TILE>
static __device__ void recurrent_gated_delta_rule_varlen_tiled_f32_impl(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const StateT* __restrict__ initial_states,
    const unsigned int* __restrict__ cu_seqlens,
    float* __restrict__ out,
    StateT* __restrict__ final_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  const int value_tile = blockIdx.x;
  const int value_head = blockIdx.y;
  const int seq = blockIdx.z;
  if (value_head >= value_heads || seq >= batch) return;

  const int token_start = static_cast<int>(cu_seqlens[seq]);
  const int token_end = static_cast<int>(cu_seqlens[seq + 1]);
  if (token_start < 0 || token_end <= token_start || token_end > total_tokens) {
    return;
  }

  const int value_start = value_tile * BV_TILE;
  const int repeat_factor = value_heads / key_heads;
  const int key_head = value_head / repeat_factor;
  const int state_len = value_heads * value_dim * key_dim;
  const int row_state_base = seq * state_len;

  __shared__ float partial[BV_TILE][256];
  __shared__ float delta[BV_TILE];
  float local[BV_TILE];

  for (int i = threadIdx.x; i < BV_TILE * key_dim; i += blockDim.x) {
    const int local_v = i / key_dim;
    const int kd = i - local_v * key_dim;
    const int value_offset = value_start + local_v;
    if (value_offset >= value_dim) continue;

    const int state_idx =
        row_state_base + (value_head * value_dim + value_offset) * key_dim + kd;
    ferrum_gdr_store_value(
        final_states, state_idx,
        ferrum_gdr_load_value(initial_states, state_idx));
  }
  __syncthreads();

  for (int token = token_start; token < token_end; ++token) {
    const int gate_idx = token * value_heads + value_head;
    const float decay = expf(g[gate_idx]);
    const float beta_t = beta[gate_idx];

#pragma unroll
    for (int i = 0; i < BV_TILE; ++i) {
      local[i] = 0.0f;
    }

    for (int i = threadIdx.x; i < BV_TILE * key_dim; i += blockDim.x) {
      const int local_v = i / key_dim;
      const int kd = i - local_v * key_dim;
      const int value_offset = value_start + local_v;
      if (value_offset >= value_dim) continue;

      const int state_idx =
          row_state_base + (value_head * value_dim + value_offset) * key_dim + kd;
      const int qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
      const float state =
          ferrum_gdr_load_value(final_states, state_idx) * decay;
      ferrum_gdr_store_value(final_states, state_idx, state);
      local[local_v] += state * key[qk_idx];
    }

#pragma unroll
    for (int local_v = 0; local_v < BV_TILE; ++local_v) {
      partial[local_v][threadIdx.x] = local[local_v];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
#pragma unroll
        for (int local_v = 0; local_v < BV_TILE; ++local_v) {
          partial[local_v][threadIdx.x] += partial[local_v][threadIdx.x + stride];
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
#pragma unroll
      for (int local_v = 0; local_v < BV_TILE; ++local_v) {
        const int value_offset = value_start + local_v;
        if (value_offset < value_dim) {
          const int value_idx =
              ((token * value_heads + value_head) * value_dim) + value_offset;
          delta[local_v] = (value[value_idx] - partial[local_v][0]) * beta_t;
        } else {
          delta[local_v] = 0.0f;
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < BV_TILE; ++i) {
      local[i] = 0.0f;
    }

    for (int i = threadIdx.x; i < BV_TILE * key_dim; i += blockDim.x) {
      const int local_v = i / key_dim;
      const int kd = i - local_v * key_dim;
      const int value_offset = value_start + local_v;
      if (value_offset >= value_dim) continue;

      const int state_idx =
          row_state_base + (value_head * value_dim + value_offset) * key_dim + kd;
      const int qk_idx = ((token * key_heads + key_head) * key_dim) + kd;
      const float updated = ferrum_gdr_load_value(final_states, state_idx) +
                            delta[local_v] * key[qk_idx];
      ferrum_gdr_store_value(final_states, state_idx, updated);
      local[local_v] += updated * (query[qk_idx] * scale);
    }

#pragma unroll
    for (int local_v = 0; local_v < BV_TILE; ++local_v) {
      partial[local_v][threadIdx.x] = local[local_v];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
#pragma unroll
        for (int local_v = 0; local_v < BV_TILE; ++local_v) {
          partial[local_v][threadIdx.x] += partial[local_v][threadIdx.x + stride];
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
#pragma unroll
      for (int local_v = 0; local_v < BV_TILE; ++local_v) {
        const int value_offset = value_start + local_v;
        if (value_offset < value_dim) {
          const int value_idx =
              ((token * value_heads + value_head) * value_dim) + value_offset;
          out[value_idx] = partial[local_v][0];
        }
      }
    }
    __syncthreads();
  }
}

extern "C" __global__ void recurrent_gated_delta_rule_varlen_tiled16_f32(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const float* __restrict__ initial_states,
    const unsigned int* __restrict__ cu_seqlens,
    float* __restrict__ out,
    float* __restrict__ final_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  recurrent_gated_delta_rule_varlen_tiled_f32_impl<float, 16>(
      query, key, value, g, beta, initial_states, cu_seqlens, out,
      final_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, scale);
}

extern "C" __global__ void recurrent_gated_delta_rule_varlen_tiled16_f32_indirect(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const unsigned long long* __restrict__ state_bindings,
    const unsigned int* __restrict__ cu_seqlens,
    float* __restrict__ out,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  float* state = reinterpret_cast<float*>(state_bindings[1]);
  recurrent_gated_delta_rule_varlen_tiled_f32_impl<float, 16>(
      query, key, value, g, beta, state, cu_seqlens, out, state, batch,
      total_tokens, key_heads, value_heads, key_dim, value_dim, scale);
}

extern "C" __global__ void recurrent_gated_delta_rule_varlen_tiled16_f32_state_f16(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const __half* __restrict__ initial_states,
    const unsigned int* __restrict__ cu_seqlens,
    float* __restrict__ out,
    __half* __restrict__ final_states,
    const int batch,
    const int total_tokens,
    const int key_heads,
    const int value_heads,
    const int key_dim,
    const int value_dim,
    const float scale) {
  recurrent_gated_delta_rule_varlen_tiled_f32_impl<__half, 16>(
      query, key, value, g, beta, initial_states, cu_seqlens, out,
      final_states, batch, total_tokens, key_heads, value_heads, key_dim,
      value_dim, scale);
}
