// GPT-OSS causal attention over Ferrum's fixed-size paged KV storage.
//
// This deliberately remains a distinct ABI from the standard causal-attention
// kernels. GPT-OSS adds projection biases, half-split YaRN rotary embedding,
// learned per-query-head attention sinks, and alternating full/local causal
// windows. Treating it as the standard operation would silently change model
// semantics.

#include "common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

#define GPT_OSS_WARP_SIZE 32
#define GPT_OSS_HEAD_DIM 64
#define GPT_OSS_HEAD_CHUNKS 2

__device__ __forceinline__ __half* gpt_oss_paged_element(
    const unsigned long long* __restrict__ page_pointers,
    const int page_count,
    const int page_elements,
    const long long logical_element) {
  const long long page = logical_element / page_elements;
  if (page < 0 || page >= page_count) return nullptr;
  const long long offset = logical_element - page * page_elements;
  return reinterpret_cast<__half*>(
             static_cast<uintptr_t>(page_pointers[page])) +
         offset;
}

__device__ __forceinline__ long long gpt_oss_kv_element_index(
    const int token,
    const int kind,
    const int head,
    const int dim,
    const int kv_heads,
    const int head_dim) {
  return (((long long)token * 2 + kind) * kv_heads + head) * head_dim + dim;
}

__device__ __forceinline__ void gpt_oss_store_kv(
    const unsigned long long* __restrict__ page_pointers,
    const int page_count,
    const int page_elements,
    const int token,
    const int kind,
    const int head,
    const int dim,
    const int kv_heads,
    const int head_dim,
    const float value) {
  __half* destination = gpt_oss_paged_element(
      page_pointers, page_count, page_elements,
      gpt_oss_kv_element_index(token, kind, head, dim, kv_heads, head_dim));
  if (destination != nullptr) *destination = __float2half(value);
}

__device__ __forceinline__ float gpt_oss_load_kv(
    const unsigned long long* __restrict__ page_pointers,
    const int page_count,
    const int page_elements,
    const int token,
    const int kind,
    const int head,
    const int dim,
    const int kv_heads,
    const int head_dim) {
  const __half* source = gpt_oss_paged_element(
      page_pointers, page_count, page_elements,
      gpt_oss_kv_element_index(token, kind, head, dim, kv_heads, head_dim));
  return source == nullptr ? 0.0f : __half2float(*source);
}

// OpenAI's reference computes YaRN with an NTK-by-parts ramp:
//   low  = d/2 * log(original / (beta_fast * 2pi)) / log(theta)
//   high = d/2 * log(original / (beta_slow * 2pi)) / log(theta)
// and blends extrapolated and factor-scaled frequencies across that ramp.
__device__ __forceinline__ float gpt_oss_yarn_inv_frequency(
    const int pair,
    const int rope_dim,
    const float rope_theta,
    const float factor,
    const float original_context,
    const float beta_fast,
    const float beta_slow) {
  const float half_dim = 0.5f * (float)rope_dim;
  const float base_frequency =
      powf(rope_theta, (2.0f * (float)pair) / (float)rope_dim);
  const float low =
      half_dim * logf(original_context / (beta_fast * 2.0f * CUDART_PI_F)) /
      logf(rope_theta);
  const float high =
      half_dim * logf(original_context / (beta_slow * 2.0f * CUDART_PI_F)) /
      logf(rope_theta);
  const float ramp = fminf(1.0f, fmaxf(0.0f, ((float)pair - low) / (high - low)));
  const float extrapolation_weight = 1.0f - ramp;
  const float interpolation = 1.0f / (factor * base_frequency);
  const float extrapolation = 1.0f / base_frequency;
  return interpolation * (1.0f - extrapolation_weight) +
         extrapolation * extrapolation_weight;
}

extern "C" __global__ void gpt_oss_prepare_qkv_yarn_f16(
    const __half* __restrict__ query_raw,
    const __half* __restrict__ key_raw,
    const __half* __restrict__ value_raw,
    const __half* __restrict__ query_bias,
    const __half* __restrict__ key_bias,
    const __half* __restrict__ value_bias,
    __half* __restrict__ query,
    const int* __restrict__ control,
    const unsigned long long* __restrict__ page_pointers,
    const int page_elements,
    const int query_heads,
    const int kv_heads,
    const int head_dim,
    const int rope_dim,
    const int query_features,
    const int kv_features,
    const float rope_theta,
    const float yarn_factor,
    const float yarn_original_context,
    const float yarn_beta_fast,
    const float yarn_beta_slow) {
  const int page_count = control[0];
  const int position_start = control[1];
  const int tokens = control[2];
  const int token = blockIdx.x;
  const int combined_head = blockIdx.y;
  const int lane = threadIdx.x;
  const int total_heads = query_heads + 2 * kv_heads;
  if (token >= tokens || combined_head >= total_heads ||
      lane >= GPT_OSS_WARP_SIZE)
    return;

  const bool is_query = combined_head < query_heads;
  const bool is_key =
      !is_query && combined_head < query_heads + kv_heads;
  const int head = is_query
                       ? combined_head
                       : (is_key ? combined_head - query_heads
                                 : combined_head - query_heads - kv_heads);
  const int absolute_position = position_start + token;

  if (!is_query && !is_key) {
    const long long row = (long long)token * kv_features + head * head_dim;
    const int bias_row = head * head_dim;
    for (int dim = lane; dim < head_dim; dim += GPT_OSS_WARP_SIZE) {
      const float value = __half2float(value_raw[row + dim]) +
                          __half2float(value_bias[bias_row + dim]);
      gpt_oss_store_kv(page_pointers, page_count, page_elements,
                        absolute_position, 1, head, dim, kv_heads, head_dim,
                        value);
    }
    return;
  }

  const __half* source = is_query
                             ? query_raw +
                                   (long long)token * query_features +
                                   head * head_dim
                             : key_raw + (long long)token * kv_features +
                                   head * head_dim;
  const __half* bias =
      is_query ? query_bias + head * head_dim : key_bias + head * head_dim;
  const int half_rope = rope_dim / 2;
  const float concentration = 0.1f * logf(yarn_factor) + 1.0f;

  // GPT-OSS rotates the two contiguous halves, not adjacent pairs.
  for (int pair = lane; pair < half_rope; pair += GPT_OSS_WARP_SIZE) {
    const int low = pair;
    const int high = pair + half_rope;
    const float x0 = __half2float(source[low]) + __half2float(bias[low]);
    const float x1 = __half2float(source[high]) + __half2float(bias[high]);
    const float angle =
        (float)absolute_position *
        gpt_oss_yarn_inv_frequency(pair, rope_dim, rope_theta, yarn_factor,
                                   yarn_original_context, yarn_beta_fast,
                                   yarn_beta_slow);
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(angle, &sine, &cosine);
    sine *= concentration;
    cosine *= concentration;
    const float rotated_low = x0 * cosine - x1 * sine;
    const float rotated_high = x1 * cosine + x0 * sine;
    if (is_query) {
      const long long destination =
          ((long long)token * query_heads + head) * head_dim;
      query[destination + low] = __float2half(rotated_low);
      query[destination + high] = __float2half(rotated_high);
    } else {
      gpt_oss_store_kv(page_pointers, page_count, page_elements,
                       absolute_position, 0, head, low, kv_heads, head_dim,
                       rotated_low);
      gpt_oss_store_kv(page_pointers, page_count, page_elements,
                       absolute_position, 0, head, high, kv_heads, head_dim,
                       rotated_high);
    }
  }

  for (int dim = rope_dim + lane; dim < head_dim; dim += GPT_OSS_WARP_SIZE) {
    const float value = __half2float(source[dim]) + __half2float(bias[dim]);
    if (is_query) {
      query[((long long)token * query_heads + head) * head_dim + dim] =
          __float2half(value);
    } else {
      gpt_oss_store_kv(page_pointers, page_count, page_elements,
                       absolute_position, 0, head, dim, kv_heads, head_dim,
                       value);
    }
  }
}

extern "C" __global__ void gpt_oss_paged_attention_sink_f16(
    const __half* __restrict__ query,
    const __half* __restrict__ sinks,
    const int* __restrict__ control,
    const unsigned long long* __restrict__ page_pointers,
    __half* __restrict__ output,
    const int page_elements,
    const int query_heads,
    const int kv_heads,
    const int head_dim,
    const int sliding_window) {
  const int page_count = control[0];
  const int position_start = control[1];
  const int tokens = control[2];
  const int token = blockIdx.x;
  const int query_head = blockIdx.y;
  const int lane = threadIdx.x;
  if (token >= tokens || query_head >= query_heads ||
      lane >= GPT_OSS_WARP_SIZE)
    return;

  const int kv_head = query_head / (query_heads / kv_heads);
  const int absolute_position = position_start + token;
  const int first_key =
      sliding_window == 0
          ? 0
          : max(0, absolute_position - sliding_window + 1);
  float query_values[GPT_OSS_HEAD_CHUNKS];
  float accumulated[GPT_OSS_HEAD_CHUNKS];

#pragma unroll
  for (int chunk = 0; chunk < GPT_OSS_HEAD_CHUNKS; ++chunk) {
    const int dim = lane + chunk * GPT_OSS_WARP_SIZE;
    query_values[chunk] =
        dim < head_dim
            ? __half2float(query[((long long)token * query_heads + query_head) *
                                 head_dim + dim])
            : 0.0f;
    accumulated[chunk] = 0.0f;
  }

  // The learned sink is a zero-valued pseudo-token included in the softmax
  // denominator. Initializing the online softmax with exp(sink-sink)=1 is
  // exactly equivalent to concatenating the sink logit and dropping its value.
  float running_max = __half2float(sinks[query_head]);
  float running_sum = 1.0f;
  const float attention_scale = rsqrtf((float)head_dim);
  for (int key_position = first_key; key_position <= absolute_position;
       ++key_position) {
    float partial_dot = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < GPT_OSS_HEAD_CHUNKS; ++chunk) {
      const int dim = lane + chunk * GPT_OSS_WARP_SIZE;
      if (dim < head_dim) {
        partial_dot +=
            query_values[chunk] *
            gpt_oss_load_kv(page_pointers, page_count, page_elements,
                            key_position, 0, kv_head, dim, kv_heads, head_dim);
      }
    }
    const float score = warp_reduce_sum(partial_dot) * attention_scale;
    const float next_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - next_max);
    const float value_scale = expf(score - next_max);
    running_sum = running_sum * previous_scale + value_scale;
#pragma unroll
    for (int chunk = 0; chunk < GPT_OSS_HEAD_CHUNKS; ++chunk) {
      const int dim = lane + chunk * GPT_OSS_WARP_SIZE;
      if (dim < head_dim) {
        const float value =
            gpt_oss_load_kv(page_pointers, page_count, page_elements,
                            key_position, 1, kv_head, dim, kv_heads, head_dim);
        accumulated[chunk] =
            accumulated[chunk] * previous_scale + value * value_scale;
      }
    }
    running_max = next_max;
  }

  const float inverse_sum = 1.0f / running_sum;
#pragma unroll
  for (int chunk = 0; chunk < GPT_OSS_HEAD_CHUNKS; ++chunk) {
    const int dim = lane + chunk * GPT_OSS_WARP_SIZE;
    if (dim < head_dim) {
      output[((long long)token * query_heads + query_head) * head_dim + dim] =
          __float2half(accumulated[chunk] * inverse_sum);
    }
  }
}

extern "C" __global__ void gpt_oss_residual_output_bias_f16(
    const __half* __restrict__ residual,
    const __half* __restrict__ branch,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int hidden_size,
    const int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const int feature = index % hidden_size;
  output[index] = __float2half(__half2float(residual[index]) +
                              __half2float(branch[index]) +
                              __half2float(bias[feature]));
}
