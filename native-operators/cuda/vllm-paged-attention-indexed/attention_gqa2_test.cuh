/*
 * Test-only specialization of the adjacent vLLM attention kernel.
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "attention_kernels.cuh"

#ifdef FERRUM_TEST_CAUSAL_GQA2
namespace vllm {
// Test-only two-query-head specialization. The original four-warps-per-head
// arithmetic is executed independently for each head, on the same 128 lanes.
// Only the K/V register loads are shared; partitioning and the public reduce
// kernel remain unchanged. No dense KV staging or inter-head reduction.
__global__ void paged_attention_v2_gqa2_f16_h256_b16_indexed_test_kernel(
    float* __restrict__ exp_sums, float* __restrict__ max_logits,
    uint16_t* __restrict__ out, const uint16_t* __restrict__ q,
    int num_kv_heads, float scale,
    const uint64_t* __restrict__ block_tables,
    const int* __restrict__ seq_lens, const int* __restrict__ row_map,
    int original_num_seqs, int max_num_blocks_per_seq, int q_stride,
    int kv_block_stride, int kv_head_stride) {
  constexpr int HEAD_SIZE = 256;
  constexpr int BLOCK_SIZE = 16;
  constexpr int NUM_THREADS = 128;
  constexpr int NUM_WARPS = 4;
  constexpr int PARTITION_SIZE = 512;
  constexpr int THREAD_GROUP_SIZE = 2;
  constexpr int VEC_SIZE = 4;
  constexpr int NUM_VECS_PER_THREAD = 32;
  constexpr int NUM_THREAD_GROUPS = 64;
  constexpr int X = 8;
  using K_vec = typename Vec<uint16_t, VEC_SIZE>::Type;
  using Q_vec = K_vec;
  const int group_idx = blockIdx.y;
  const int seq_idx = row_map[group_idx];
  if (seq_idx < 0 || seq_idx >= original_num_seqs) return;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  const int seq_len = seq_lens[seq_idx];
  if (partition_idx * PARTITION_SIZE >= seq_len) return;
  const int num_seq_blocks = ((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
  const int start_block_idx = partition_idx * (PARTITION_SIZE / BLOCK_SIZE);
  const int partition_end_block = start_block_idx + PARTITION_SIZE / BLOCK_SIZE;
  const int end_block_idx = partition_end_block < num_seq_blocks
                                ? partition_end_block : num_seq_blocks;
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int block_end_token = end_block_idx * BLOCK_SIZE;
  const int num_tokens = (block_end_token < seq_len ? block_end_token : seq_len) - start_token_idx;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;
  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;
  const int head_base = blockIdx.x * 2;
  const int num_heads = gridDim.x * 2;
  const int kv_head_idx = head_base / (num_heads / num_kv_heads);
  __shared__ Q_vec q_vecs[2][THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
  __shared__ float red_smem[2 * NUM_WARPS];
  extern __shared__ char shared_mem[];
  float* logits[2] = {reinterpret_cast<float*>(shared_mem),
                      reinterpret_cast<float*>(shared_mem) + PARTITION_SIZE};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint16_t* q_ptr = q + seq_idx * q_stride + (head_base + h) * HEAD_SIZE;
#pragma unroll
    for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
         i += NUM_THREAD_GROUPS) {
      const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
      q_vecs[h][thread_group_offset][i] =
          *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
    }
  }
  __syncthreads();
  float qk_max[2] = {-FLT_MAX, -FLT_MAX};
  const uint64_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    const uint64_t block_ref = block_table[block_idx];
    const uint16_t* k_block = key_block_base<uint16_t, uint64_t>(
        nullptr, block_ref, kv_block_stride);
    const int physical_block_offset = thread_group_idx % BLOCK_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    K_vec k_vecs[NUM_VECS_PER_THREAD];
#pragma unroll
    for (int j = 0; j < NUM_VECS_PER_THREAD; ++j) {
      const uint16_t* k_ptr = k_block + kv_head_idx * kv_head_stride +
                             physical_block_offset * X;
      const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
      const int offset1 = (vec_idx * VEC_SIZE) / X;
      const int offset2 = (vec_idx * VEC_SIZE) % X;
      k_vecs[j] = *reinterpret_cast<const K_vec*>(
          k_ptr + offset1 * BLOCK_SIZE * X + offset2);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float qk = scale * Qk_dot<uint16_t, THREAD_GROUP_SIZE>::dot(
          q_vecs[h][thread_group_offset], k_vecs);
      qk += 0.f;  // Preserve the original no-ALiBi addition, including signed zero.
      if (thread_group_offset == 0) {
        const bool mask = token_idx >= seq_len;
        logits[h][token_idx - start_token_idx] = mask ? 0.f : qk;
        qk_max[h] = mask ? qk_max[h] : fmaxf(qk_max[h], qk);
      }
    }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float maximum = qk_max[h];
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2)
      maximum = fmaxf(maximum, VLLM_SHFL_XOR_SYNC(maximum, mask));
    if (lane == 0) red_smem[warp_idx] = maximum;
    __syncthreads();
    maximum = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2)
      maximum = fmaxf(maximum, VLLM_SHFL_XOR_SYNC(maximum, mask));
    maximum = VLLM_SHFL_SYNC(maximum, 0);
    float exp_sum = 0.f;
    for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
      float val = __expf(logits[h][i] - maximum);
      logits[h][i] = val;
      exp_sum += val;
    }
    exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);
    const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
    for (int i = thread_idx; i < num_tokens; i += NUM_THREADS)
      logits[h][i] *= inv_sum;
    __syncthreads();
    if (thread_idx == 0) {
      const int index = group_idx * num_heads * max_num_partitions +
                        (head_base + h) * max_num_partitions + partition_idx;
      max_logits[index] = maximum;
      exp_sums[index] = exp_sum;
    }
  }
  constexpr int V_VEC_SIZE = 8;
  constexpr int NUM_V_VECS_PER_ROW = 2;
  constexpr int NUM_ROWS_PER_ITER = 16;
  constexpr int NUM_ROWS_PER_THREAD = 16;
  using V_vec = typename Vec<uint16_t, V_VEC_SIZE>::Type;
  using L_vec = V_vec;
  using Float_L_vec = typename FloatVec<L_vec>::Type;
  float accs[2][NUM_ROWS_PER_THREAD] = {};
  uint16_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    const uint64_t block_ref = block_table[block_idx];
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec[2];
#pragma unroll
    for (int h = 0; h < 2; ++h)
      from_float(logits_vec[h], *reinterpret_cast<Float_L_vec*>(
          logits[h] + token_idx - start_token_idx));
    const uint16_t* v_ptr = value_block_base<uint16_t, uint64_t>(
        nullptr, block_ref, kv_block_stride) + kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; ++i) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
      V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
      if (block_idx == num_seq_blocks - 1) {
        uint16_t* v_vec_ptr = reinterpret_cast<uint16_t*>(&v_vec);
#pragma unroll
        for (int j = 0; j < V_VEC_SIZE; ++j)
          v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) accs[h][i] += dot(logits_vec[h], v_vec);
    }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; ++i) {
      float acc = accs[h][i];
#pragma unroll
      for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2)
        acc += VLLM_SHFL_XOR_SYNC(acc, mask);
      accs[h][i] = acc;
    }
  }
  __syncthreads();
  // Both logits are dead. Reuse the original output-reduction workspace,
  // finishing one head before reusing it for the other.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int count = NUM_WARPS; count > 1; count /= 2) {
      const int mid = count / 2;
      if (warp_idx >= mid && warp_idx < count) {
        float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
        for (int i = 0; i < NUM_ROWS_PER_THREAD; ++i) {
          const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
          if (lane % NUM_V_VECS_PER_ROW == 0) dst[row_idx] = accs[h][i];
        }
      }
      __syncthreads();
      if (warp_idx < mid) {
        const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
        for (int i = 0; i < NUM_ROWS_PER_THREAD; ++i) {
          const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
          if (lane % NUM_V_VECS_PER_ROW == 0) accs[h][i] += src[row_idx];
        }
      }
      __syncthreads();
    }
    if (warp_idx == 0) {
      uint16_t* out_ptr = out + group_idx * num_heads * max_num_partitions * HEAD_SIZE +
                         (head_base + h) * max_num_partitions * HEAD_SIZE +
                         partition_idx * HEAD_SIZE;
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; ++i) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (lane % NUM_V_VECS_PER_ROW == 0) from_float(out_ptr[row_idx], accs[h][i]);
      }
    }
  }
}
}  // namespace vllm
#endif  // FERRUM_TEST_CAUSAL_GQA2
