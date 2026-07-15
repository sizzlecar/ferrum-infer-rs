// Generic vNext causal attention over independently allocated fixed-size pages.
// Logical KV layout is [token, K_or_V, kv_head, head_dim]. The provider owns
// the page-pointer table and the core owns allocation, admission, and lifetime.

#include "common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

#define VNEXT_WARP_SIZE 32
#define VNEXT_MAX_HEAD_CHUNKS 8

__device__ __forceinline__ __half* vnext_paged_element(
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

__device__ __forceinline__ long long vnext_kv_element_index(
    const int token,
    const int kind,
    const int head,
    const int dim,
    const int kv_heads,
    const int head_dim) {
  return (((long long)token * 2 + kind) * kv_heads + head) * head_dim + dim;
}

__device__ __forceinline__ void vnext_store_kv(
    const unsigned long long* __restrict__ page_pointers,
    const int page_count,
    const int page_elements,
    const int token,
    const int kind,
    const int head,
    const int dim,
    const int kv_heads,
    const int head_dim,
    const __half value) {
  __half* destination = vnext_paged_element(
      page_pointers, page_count, page_elements,
      vnext_kv_element_index(token, kind, head, dim, kv_heads, head_dim));
  if (destination != nullptr) *destination = value;
}

__device__ __forceinline__ float vnext_load_kv(
    const unsigned long long* __restrict__ page_pointers,
    const int page_count,
    const int page_elements,
    const int token,
    const int kind,
    const int head,
    const int dim,
    const int kv_heads,
    const int head_dim) {
  const __half* source = vnext_paged_element(
      page_pointers, page_count, page_elements,
      vnext_kv_element_index(token, kind, head, dim, kv_heads, head_dim));
  return source == nullptr ? 0.0f : __half2float(*source);
}

__device__ __forceinline__ void vnext_store_prepared_value(
    __half* __restrict__ query,
    const unsigned long long* __restrict__ page_pointers,
    const int page_count,
    const int page_elements,
    const bool is_query,
    const int token,
    const int absolute_position,
    const int head,
    const int dim,
    const int query_heads,
    const int kv_heads,
    const int head_dim,
    const float value) {
  const __half converted = __float2half(value);
  if (is_query) {
    query[((long long)token * query_heads + head) * head_dim + dim] = converted;
  } else {
    vnext_store_kv(page_pointers, page_count, page_elements, absolute_position,
                   0, head, dim, kv_heads, head_dim, converted);
  }
}

extern "C" __global__ void vnext_causal_prepare_f16(
    const __half* __restrict__ query_raw,
    const __half* __restrict__ key_raw,
    const __half* __restrict__ value_raw,
    const __half* __restrict__ query_norm_weight,
    const __half* __restrict__ key_norm_weight,
    __half* __restrict__ query,
    const unsigned long long* __restrict__ page_pointers,
    const int page_count,
    const int page_elements,
    const int position_start,
    const int tokens,
    const int query_heads,
    const int kv_heads,
    const int head_dim,
    const int rope_dim,
    const int query_projection_stride,
    const int query_head_stride,
    const int kv_projection_stride,
    const float epsilon,
    const float rope_theta,
    const int rope_interleaved) {
  const int token = blockIdx.x;
  const int combined_head = blockIdx.y;
  const int lane = threadIdx.x;
  const int total_heads = query_heads + 2 * kv_heads;
  if (token >= tokens || combined_head >= total_heads || lane >= VNEXT_WARP_SIZE)
    return;

  const bool is_query = combined_head < query_heads;
  const bool is_key = !is_query && combined_head < query_heads + kv_heads;
  const int head = is_query
                       ? combined_head
                       : (is_key ? combined_head - query_heads
                                 : combined_head - query_heads - kv_heads);
  const int absolute_position = position_start + token;

  if (!is_query && !is_key) {
    const __half* source =
        value_raw + (long long)token * kv_projection_stride + head * head_dim;
    for (int dim = lane; dim < head_dim; dim += VNEXT_WARP_SIZE) {
      vnext_store_kv(page_pointers, page_count, page_elements, absolute_position,
                     1, head, dim, kv_heads, head_dim, source[dim]);
    }
    return;
  }

  const __half* source = is_query
                             ? query_raw +
                                   (long long)token * query_projection_stride +
                                   head * query_head_stride
                             : key_raw +
                                   (long long)token * kv_projection_stride +
                                   head * head_dim;
  const __half* weight = is_query ? query_norm_weight : key_norm_weight;
  float sum_squares = 0.0f;
  for (int dim = lane; dim < head_dim; dim += VNEXT_WARP_SIZE) {
    const float value = __half2float(source[dim]);
    sum_squares += value * value;
  }
  sum_squares = warp_reduce_sum(sum_squares);
  const float norm_scale = rsqrtf(sum_squares / (float)head_dim + epsilon);
  const int half_rope = rope_dim / 2;

  if (rope_interleaved != 0) {
    for (int pair = lane; pair < half_rope; pair += VNEXT_WARP_SIZE) {
      const int low = 2 * pair;
      const int high = low + 1;
      const float x0 = __half2float(source[low]) * norm_scale *
                       __half2float(weight[low]);
      const float x1 = __half2float(source[high]) * norm_scale *
                       __half2float(weight[high]);
      const float exponent = -(2.0f * pair) / (float)rope_dim;
      const float angle =
          absolute_position * powf(rope_theta, exponent);
      float sine = 0.0f;
      float cosine = 0.0f;
      sincosf(angle, &sine, &cosine);
      vnext_store_prepared_value(
          query, page_pointers, page_count, page_elements, is_query, token,
          absolute_position, head, low, query_heads, kv_heads, head_dim,
          x0 * cosine - x1 * sine);
      vnext_store_prepared_value(
          query, page_pointers, page_count, page_elements, is_query, token,
          absolute_position, head, high, query_heads, kv_heads, head_dim,
          x1 * cosine + x0 * sine);
    }
  } else {
    for (int pair = lane; pair < half_rope; pair += VNEXT_WARP_SIZE) {
      const int low = pair;
      const int high = pair + half_rope;
      const float x0 = __half2float(source[low]) * norm_scale *
                       __half2float(weight[low]);
      const float x1 = __half2float(source[high]) * norm_scale *
                       __half2float(weight[high]);
      const float exponent = -(2.0f * pair) / (float)rope_dim;
      const float angle =
          absolute_position * powf(rope_theta, exponent);
      float sine = 0.0f;
      float cosine = 0.0f;
      sincosf(angle, &sine, &cosine);
      vnext_store_prepared_value(
          query, page_pointers, page_count, page_elements, is_query, token,
          absolute_position, head, low, query_heads, kv_heads, head_dim,
          x0 * cosine - x1 * sine);
      vnext_store_prepared_value(
          query, page_pointers, page_count, page_elements, is_query, token,
          absolute_position, head, high, query_heads, kv_heads, head_dim,
          x1 * cosine + x0 * sine);
    }
  }

  for (int dim = rope_dim + lane; dim < head_dim; dim += VNEXT_WARP_SIZE) {
    const float value = __half2float(source[dim]) * norm_scale *
                        __half2float(weight[dim]);
    vnext_store_prepared_value(
        query, page_pointers, page_count, page_elements, is_query, token,
        absolute_position, head, dim, query_heads, kv_heads, head_dim, value);
  }
}

extern "C" __global__ void vnext_causal_attention_f16(
    const __half* __restrict__ query,
    const __half* __restrict__ query_raw,
    const unsigned long long* __restrict__ page_pointers,
    __half* __restrict__ output,
    const int page_count,
    const int page_elements,
    const int position_start,
    const int tokens,
    const int query_heads,
    const int kv_heads,
    const int head_dim,
    const int query_projection_stride,
    const int output_gate) {
  const int token = blockIdx.x;
  const int query_head = blockIdx.y;
  const int lane = threadIdx.x;
  if (token >= tokens || query_head >= query_heads || lane >= VNEXT_WARP_SIZE)
    return;

  const int kv_head = query_head / (query_heads / kv_heads);
  const int absolute_position = position_start + token;
  float query_values[VNEXT_MAX_HEAD_CHUNKS];
  float accumulated[VNEXT_MAX_HEAD_CHUNKS];

#pragma unroll
  for (int chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
    const int dim = lane + chunk * VNEXT_WARP_SIZE;
    query_values[chunk] =
        dim < head_dim
            ? __half2float(query[((long long)token * query_heads + query_head) *
                                 head_dim + dim])
            : 0.0f;
    accumulated[chunk] = 0.0f;
  }

  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;
  const float attention_scale = rsqrtf((float)head_dim);
  for (int key_position = 0; key_position <= absolute_position;
       ++key_position) {
    float partial_dot = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
      const int dim = lane + chunk * VNEXT_WARP_SIZE;
      if (dim < head_dim) {
        partial_dot += query_values[chunk] *
                       vnext_load_kv(page_pointers, page_count, page_elements,
                                     key_position, 0, kv_head, dim, kv_heads,
                                     head_dim);
      }
    }
    const float score = warp_reduce_sum(partial_dot) * attention_scale;
    const float next_max = fmaxf(running_max, score);
    const float previous_scale =
        isinf(running_max) ? 0.0f : expf(running_max - next_max);
    const float value_scale = expf(score - next_max);
    running_sum = running_sum * previous_scale + value_scale;
#pragma unroll
    for (int chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
      const int dim = lane + chunk * VNEXT_WARP_SIZE;
      if (dim < head_dim) {
        const float value =
            vnext_load_kv(page_pointers, page_count, page_elements,
                          key_position, 1, kv_head, dim, kv_heads, head_dim);
        accumulated[chunk] = accumulated[chunk] * previous_scale +
                             value * value_scale;
      }
    }
    running_max = next_max;
  }

  const float inverse_sum = 1.0f / running_sum;
#pragma unroll
  for (int chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
    const int dim = lane + chunk * VNEXT_WARP_SIZE;
    if (dim < head_dim) {
      float value = accumulated[chunk] * inverse_sum;
      if (output_gate != 0) {
        const long long gate_index =
            (long long)token * query_projection_stride +
            query_head * (2 * head_dim) + head_dim + dim;
        const float gate = __half2float(query_raw[gate_index]);
        value *= 1.0f / (1.0f + expf(-gate));
      }
      output[((long long)token * query_heads + query_head) * head_dim + dim] =
          __float2half(value);
    }
  }
}
