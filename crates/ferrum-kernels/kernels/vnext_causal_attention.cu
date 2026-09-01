// vNext causal attention over independently allocated fixed-size pages. The
// provider owns the address table and the core owns allocation, admission, and
// lifetime. The token-major kernels remain the generic fallback. The addressed
// kernels use vLLM's 16-token K/V block layout so decode can dispatch the
// existing tiled vLLM attention without changing core resource ownership.

#include "common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

#define VNEXT_WARP_SIZE 32
#define VNEXT_MAX_HEAD_CHUNKS 16
#define VNEXT_VLLM_BLOCK_TOKENS 16
#define VNEXT_VLLM_K_PACK 8

// A non-zero binding stride selects packed multi-participant execution. Each
// participant owns one binding slot containing six control words followed by
// its aligned addressed page table. Control word four is its token-major
// packed offset, so every block resolves its participant in O(1).
__device__ __forceinline__ const int* vnext_participant_control(
    const int* __restrict__ binding_base,
    const unsigned long long binding_slot_bytes,
    const int participant) {
  const char* binding_bytes = reinterpret_cast<const char*>(binding_base);
  return reinterpret_cast<const int*>(
      binding_bytes + (unsigned long long)participant * binding_slot_bytes);
}

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

__device__ __forceinline__ __half* vnext_vllm_block_pointer(
    const unsigned long long* __restrict__ block_pointers,
    const int block_count,
    const int token) {
  const int block = token / VNEXT_VLLM_BLOCK_TOKENS;
  if (block < 0 || block >= block_count) return nullptr;
  return reinterpret_cast<__half*>(
      static_cast<uintptr_t>(block_pointers[block]));
}

__device__ __forceinline__ long long vnext_vllm_k_offset(
    const int token_offset,
    const int head,
    const int dim,
    const int head_dim) {
  return (long long)head * head_dim * VNEXT_VLLM_BLOCK_TOKENS +
         (dim / VNEXT_VLLM_K_PACK) * VNEXT_VLLM_BLOCK_TOKENS *
             VNEXT_VLLM_K_PACK +
         token_offset * VNEXT_VLLM_K_PACK + dim % VNEXT_VLLM_K_PACK;
}

__device__ __forceinline__ long long vnext_vllm_v_offset(
    const int token_offset,
    const int head,
    const int dim,
    const int kv_heads,
    const int head_dim) {
  const long long key_block_elements =
      (long long)kv_heads * head_dim * VNEXT_VLLM_BLOCK_TOKENS;
  return key_block_elements +
         (long long)head * head_dim * VNEXT_VLLM_BLOCK_TOKENS +
         (long long)dim * VNEXT_VLLM_BLOCK_TOKENS + token_offset;
}

__device__ __forceinline__ void vnext_store_vllm_kv(
    const unsigned long long* __restrict__ block_pointers,
    const int block_count,
    const int token,
    const int kind,
    const int head,
    const int dim,
    const int kv_heads,
    const int head_dim,
    const __half value) {
  __half* block = vnext_vllm_block_pointer(block_pointers, block_count, token);
  if (block == nullptr) return;
  const int token_offset = token % VNEXT_VLLM_BLOCK_TOKENS;
  const long long offset =
      kind == 0
          ? vnext_vllm_k_offset(token_offset, head, dim, head_dim)
          : vnext_vllm_v_offset(token_offset, head, dim, kv_heads, head_dim);
  block[offset] = value;
}

__device__ __forceinline__ float vnext_load_vllm_kv(
    const unsigned long long* __restrict__ block_pointers,
    const int block_count,
    const int token,
    const int kind,
    const int head,
    const int dim,
    const int kv_heads,
    const int head_dim) {
  const __half* block =
      vnext_vllm_block_pointer(block_pointers, block_count, token);
  if (block == nullptr) return 0.0f;
  const int token_offset = token % VNEXT_VLLM_BLOCK_TOKENS;
  const long long offset =
      kind == 0
          ? vnext_vllm_k_offset(token_offset, head, dim, head_dim)
          : vnext_vllm_v_offset(token_offset, head, dim, kv_heads, head_dim);
  return __half2float(block[offset]);
}

__device__ __forceinline__ void vnext_store_prepared_value(
    __half* __restrict__ query,
    const unsigned long long* __restrict__ page_pointers,
    const int page_count,
    const int page_elements,
    const int kv_layout,
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
  } else if (kv_layout != 0) {
    vnext_store_vllm_kv(page_pointers, page_count, absolute_position, 0, head,
                        dim, kv_heads, head_dim, converted);
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
    const int* __restrict__ control,
    const unsigned long long* __restrict__ page_pointers,
    const int page_elements,
    const int kv_layout,
    const int query_heads,
    const int kv_heads,
    const int head_dim,
    const int rope_dim,
    const int rope_frequency_denominator,
    const int query_projection_stride,
    const int query_head_stride,
    const int kv_projection_stride,
    const float epsilon,
    const float rope_theta,
    const int rope_interleaved,
    const int value_rms_norm,
    const unsigned long long binding_slot_bytes) {
  const int participant = blockIdx.z;
  if (binding_slot_bytes != 0) {
    control = vnext_participant_control(
        control, binding_slot_bytes, participant);
    page_pointers = reinterpret_cast<const unsigned long long*>(
        reinterpret_cast<const char*>(page_pointers) +
        (unsigned long long)participant * binding_slot_bytes);
  }
  const int packed_token_start = binding_slot_bytes == 0 ? 0 : control[4];
  const int page_count = control[0];
  const int position_start = control[1];
  const int tokens = control[2];
  const int token = blockIdx.x;
  const int packed_token = packed_token_start + token;
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
        value_raw + (long long)packed_token * kv_projection_stride +
        head * head_dim;
    float norm_scale = 1.0f;
    if (value_rms_norm != 0) {
      float sum_squares = 0.0f;
      for (int dim = lane; dim < head_dim; dim += VNEXT_WARP_SIZE) {
        const float value = __half2float(source[dim]);
        sum_squares += value * value;
      }
      sum_squares = warp_reduce_sum(sum_squares);
      norm_scale = rsqrtf(sum_squares / (float)head_dim + epsilon);
    }
    for (int dim = lane; dim < head_dim; dim += VNEXT_WARP_SIZE) {
      const __half value =
          value_rms_norm != 0
              ? __float2half(__half2float(source[dim]) * norm_scale)
              : source[dim];
      if (kv_layout != 0) {
        vnext_store_vllm_kv(page_pointers, page_count, absolute_position, 1,
                            head, dim, kv_heads, head_dim, value);
      } else {
        vnext_store_kv(page_pointers, page_count, page_elements,
                       absolute_position, 1, head, dim, kv_heads, head_dim,
                       value);
      }
    }
    return;
  }

  const __half* source = is_query
                             ? query_raw +
                                   (long long)packed_token *
                                       query_projection_stride +
                                   head * query_head_stride
                             : key_raw +
                                   (long long)packed_token *
                                       kv_projection_stride +
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
  const int neox_half = head_dim / 2;

  if (rope_interleaved != 0) {
    for (int pair = lane; pair < half_rope; pair += VNEXT_WARP_SIZE) {
      const int low = 2 * pair;
      const int high = low + 1;
      const float x0 = __half2float(source[low]) * norm_scale *
                       __half2float(weight[low]);
      const float x1 = __half2float(source[high]) * norm_scale *
                       __half2float(weight[high]);
      const float exponent =
          -(2.0f * pair) / (float)rope_frequency_denominator;
      const float angle =
          absolute_position * powf(rope_theta, exponent);
      float sine = 0.0f;
      float cosine = 0.0f;
      sincosf(angle, &sine, &cosine);
      vnext_store_prepared_value(
          query, page_pointers, page_count, page_elements, kv_layout, is_query,
          packed_token, absolute_position, head, low, query_heads, kv_heads,
          head_dim, x0 * cosine - x1 * sine);
      vnext_store_prepared_value(
          query, page_pointers, page_count, page_elements, kv_layout, is_query,
          packed_token, absolute_position, head, high, query_heads, kv_heads,
          head_dim, x1 * cosine + x0 * sine);
    }
  } else {
    for (int pair = lane; pair < half_rope; pair += VNEXT_WARP_SIZE) {
      const int low = pair;
      // Gemma 4 proportional partial RoPE pads the inactive frequencies to
      // head_dim/2 and then applies the standard NeoX half split. Therefore
      // an active pair mixes [pair, pair + head_dim/2], not
      // [pair, pair + rope_dim/2]. Full-width RoPE is unchanged because the
      // two offsets are equal in that case.
      const int high = pair + neox_half;
      const float x0 = __half2float(source[low]) * norm_scale *
                       __half2float(weight[low]);
      const float x1 = __half2float(source[high]) * norm_scale *
                       __half2float(weight[high]);
      const float exponent =
          -(2.0f * pair) / (float)rope_frequency_denominator;
      const float angle =
          absolute_position * powf(rope_theta, exponent);
      float sine = 0.0f;
      float cosine = 0.0f;
      sincosf(angle, &sine, &cosine);
      vnext_store_prepared_value(
          query, page_pointers, page_count, page_elements, kv_layout, is_query,
          packed_token, absolute_position, head, low, query_heads, kv_heads,
          head_dim, x0 * cosine - x1 * sine);
      vnext_store_prepared_value(
          query, page_pointers, page_count, page_elements, kv_layout, is_query,
          packed_token, absolute_position, head, high, query_heads, kv_heads,
          head_dim, x1 * cosine + x0 * sine);
    }
  }

  for (int dim = lane; dim < head_dim; dim += VNEXT_WARP_SIZE) {
    const bool rotated = rope_interleaved != 0
                             ? dim < rope_dim
                             : dim < half_rope ||
                                   (dim >= neox_half &&
                                    dim < neox_half + half_rope);
    if (rotated) {
      continue;
    }
    const float value = __half2float(source[dim]) * norm_scale *
                        __half2float(weight[dim]);
    vnext_store_prepared_value(
        query, page_pointers, page_count, page_elements, kv_layout, is_query,
        packed_token, absolute_position, head, dim, query_heads, kv_heads,
        head_dim, value);
  }
}

extern "C" __global__ void vnext_causal_attention_f16(
    const __half* __restrict__ query,
    const __half* __restrict__ query_raw,
    const int* __restrict__ control,
    const unsigned long long* __restrict__ page_pointers,
    __half* __restrict__ output,
    const int page_elements,
    const int kv_layout,
    const int query_heads,
    const int kv_heads,
    const int head_dim,
    const int query_projection_stride,
    const int output_gate,
    const float attention_scale,
    const int sliding_window,
    const unsigned long long binding_slot_bytes) {
  const int participant = blockIdx.z;
  if (binding_slot_bytes != 0) {
    control = vnext_participant_control(
        control, binding_slot_bytes, participant);
    page_pointers = reinterpret_cast<const unsigned long long*>(
        reinterpret_cast<const char*>(page_pointers) +
        (unsigned long long)participant * binding_slot_bytes);
  }
  const int packed_token_start = binding_slot_bytes == 0 ? 0 : control[4];
  const int page_count = control[0];
  const int position_start = control[1];
  const int tokens = control[2];
  const int token = blockIdx.x;
  const int packed_token = packed_token_start + token;
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
            ? __half2float(query[((long long)packed_token * query_heads +
                                  query_head) *
                                     head_dim +
                                 dim])
            : 0.0f;
    accumulated[chunk] = 0.0f;
  }

  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;
  const int key_start =
      sliding_window > 0
          ? max(0, absolute_position - sliding_window + 1)
          : 0;
  for (int key_position = key_start; key_position <= absolute_position;
       ++key_position) {
    float partial_dot = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < VNEXT_MAX_HEAD_CHUNKS; ++chunk) {
      const int dim = lane + chunk * VNEXT_WARP_SIZE;
      if (dim < head_dim) {
        const float key =
            kv_layout != 0
                ? vnext_load_vllm_kv(page_pointers, page_count, key_position,
                                     0, kv_head, dim, kv_heads, head_dim)
                : vnext_load_kv(page_pointers, page_count, page_elements,
                                key_position, 0, kv_head, dim, kv_heads,
                                head_dim);
        partial_dot += query_values[chunk] * key;
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
            kv_layout != 0
                ? vnext_load_vllm_kv(page_pointers, page_count, key_position,
                                     1, kv_head, dim, kv_heads, head_dim)
                : vnext_load_kv(page_pointers, page_count, page_elements,
                                key_position, 1, kv_head, dim, kv_heads,
                                head_dim);
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
            (long long)packed_token * query_projection_stride +
            query_head * (2 * head_dim) + head_dim + dim;
        const float gate = __half2float(query_raw[gate_index]);
        value *= 1.0f / (1.0f + expf(-gate));
      }
      output[((long long)packed_token * query_heads + query_head) * head_dim +
             dim] = __float2half(value);
    }
  }
}
