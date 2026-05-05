// Paged variable-length attention with GQA + causal mask.
//
// Supports a unified mixed batch of decode (q_len=1) and prefill chunk
// (q_len>1) sequences in one kernel launch — the workhorse for ferrum's
// chunked-prefill mixed-forward path.
//
// Shapes:
//   q             : [M_total, num_q_heads, head_dim]
//   k_block_pool  : [pool_blocks * block_size, num_kv_heads, head_dim]
//   v_block_pool  : same layout as K
//   cu_seqlens_q  : [num_seqs + 1] — prefix sum, cu_seqlens_q[0]=0,
//                                    cu_seqlens_q[num_seqs]=M_total
//   pos_offsets   : [num_seqs]     — first-q-token absolute kv position
//                                    per seq (= prior kv_len)
//   block_tables  : [num_seqs, max_blocks_per_seq]
//   output        : [M_total, num_q_heads, head_dim]
//
// Grid:  (num_q_heads, M_total)
// Block: (256,)
// Shared: max(valid_kv_len) * sizeof(float) for the score buffer.
//
// Each block handles one (head, global_query_token):
//   1. Locate the token's owning seq via linear scan of cu_seqlens_q.
//   2. Compute its absolute kv position = pos_offsets[seq] + local_idx
//      and its causal range = [0, abs_pos] inclusive.
//   3. Run the same warp-cooperative Q·K^T → softmax → weighted-V sum
//      as `paged_decode_attention`, but using its own block_table and
//      its own valid_kv_len.
//
// Linear scan over cu_seqlens_q is fine for the typical c≤32 batch
// sizes. Switch to binary search if num_seqs grows past ~64.

#include "common.cuh"

#define WARP_SIZE 32

extern "C" __global__ void paged_varlen_attn_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_block_pool,
    const __half* __restrict__ v_block_pool,
    const int*    __restrict__ cu_seqlens_q,
    const int*    __restrict__ pos_offsets,
    const int*    __restrict__ block_tables,
    __half* __restrict__ output,
    const int num_seqs,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const int block_size,
    const float scale
) {
    const int q_head = blockIdx.x;
    const int token_global = blockIdx.y;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    // Locate owning sequence by linear scan (fine for small num_seqs).
    int seq_idx = 0;
    while (seq_idx + 1 < num_seqs &&
           cu_seqlens_q[seq_idx + 1] <= token_global) {
        seq_idx++;
    }
    const int local_idx = token_global - cu_seqlens_q[seq_idx];
    const int abs_kv_pos = pos_offsets[seq_idx] + local_idx;
    const int valid_kv_len = abs_kv_pos + 1; // causal: attend to [0, abs_kv_pos]

    if (valid_kv_len <= 0) {
        // Defensive — shouldn't happen for well-formed input.
        return;
    }

    // Per-seq paged address parameters.
    const int kv_stride = num_kv_heads * head_dim;
    const int block_stride_val = block_size * kv_stride;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Q / output pointers for this (token, head).
    // Layout: [M_total, num_q_heads, head_dim].
    const __half* q_ptr =
        q + ((size_t)token_global * num_q_heads + q_head) * head_dim;
    __half* out_ptr =
        output + ((size_t)token_global * num_q_heads + q_head) * head_dim;

    extern __shared__ float s_scores[];

    // ====== Step 1: Q·K^T (paged) ======
    // Load Q into registers (warp-cooperative).
    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float q_reg[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = (threadIdx.x % WARP_SIZE) + i * WARP_SIZE;
        q_reg[i] = (i < elems_per_thread && d < head_dim)
            ? __half2float(q_ptr[d]) : 0.0f;
    }

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    for (int kv_pos = warp_id; kv_pos < valid_kv_len; kv_pos += num_warps) {
        int logical_block = kv_pos / block_size;
        int slot = kv_pos % block_size;
        int physical_block = my_block_table[logical_block];
        const __half* k_row =
            k_block_pool + physical_block * block_stride_val
                         + slot * kv_stride
                         + kv_head * head_dim;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (i < elems_per_thread && d < head_dim)
                dot += q_reg[i] * __half2float(k_row[d]);
        }
        float score = warp_reduce_sum(dot) * scale;
        if (lane_id == 0)
            s_scores[kv_pos] = score;
    }
    __syncthreads();

    // ====== Step 2: Softmax ======
    float thread_max = -1e20f;
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x)
        thread_max = fmaxf(thread_max, s_scores[i]);
    __shared__ float s_global_max;
    float bmax = block_reduce_max(thread_max);
    if (threadIdx.x == 0) s_global_max = bmax;
    __syncthreads();
    float global_max = s_global_max;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x) {
        float val = expf(s_scores[i] - global_max);
        s_scores[i] = val;
        thread_sum += val;
    }
    __syncthreads();

    __shared__ float s_global_sum;
    float bsum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0) s_global_sum = bsum;
    __syncthreads();
    float inv_sum = 1.0f / s_global_sum;

    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x)
        s_scores[i] *= inv_sum;
    __syncthreads();

    // ====== Step 3: Weighted V sum (paged) ======
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < valid_kv_len; kv_pos++) {
            int logical_block = kv_pos / block_size;
            int slot = kv_pos % block_size;
            int physical_block = my_block_table[logical_block];
            const __half* v_row =
                v_block_pool + physical_block * block_stride_val
                             + slot * kv_stride
                             + kv_head * head_dim;
            acc += s_scores[kv_pos] * __half2float(v_row[d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}
