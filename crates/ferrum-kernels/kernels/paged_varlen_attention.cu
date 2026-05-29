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

// FA-compatible paged-layout varlen attention with a small Q tile and chunked
// online softmax. This keeps K/V in the legacy/FA-friendly
// [block, slot, kv_head, head_dim] layout, but avoids materializing the full
// score vector per query and reuses each K/V row across up to 8 adjacent
// causal query tokens.
extern "C" __global__ void paged_varlen_attn_fa_flash_q8_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_block_pool,
    const __half* __restrict__ v_block_pool,
    const int*    __restrict__ cu_seqlens_q,
    const int*    __restrict__ pos_offsets,
    const int*    __restrict__ block_tables,
    const int*    __restrict__ tile_seqs,
    const int*    __restrict__ tile_starts,
    __half* __restrict__ output,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const int block_size,
    const float scale
) {
    constexpr int TILE_Q = 8;
    constexpr int BLOCK_K = 64;

    const int q_head = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int seq_idx = tile_seqs[tile_id];
    const int tile_start = tile_starts[tile_id];
    const int seq_q_start = cu_seqlens_q[seq_idx];
    const int q_len = cu_seqlens_q[seq_idx + 1] - seq_q_start;
    const int actual_tile = min(TILE_Q, q_len - tile_start);
    if (actual_tile <= 0) return;

    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int pos0 = pos_offsets[seq_idx] + tile_start;
    const int max_valid = pos0 + actual_tile;
    const int kv_stride = num_kv_heads * head_dim;
    const int block_stride_val = block_size * kv_stride;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    extern __shared__ float s_scores[];
    __shared__ float s_chunk_max[TILE_Q];
    __shared__ float s_chunk_sum[TILE_Q];

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;

    float q_reg[TILE_Q][8];
#pragma unroll
    for (int qi = 0; qi < TILE_Q; ++qi) {
        const int token_global = seq_q_start + tile_start + qi;
        const __half* q_ptr =
            q + ((size_t)token_global * num_q_heads + q_head) * head_dim;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = lane_id + i * WARP_SIZE;
            q_reg[qi][i] = (qi < actual_tile && i < elems_per_thread && d < head_dim)
                               ? __half2float(q_ptr[d])
                               : 0.0f;
        }
    }

    float m[TILE_Q];
    float l[TILE_Q];
    float acc[TILE_Q];
#pragma unroll
    for (int qi = 0; qi < TILE_Q; ++qi) {
        m[qi] = -1e20f;
        l[qi] = 0.0f;
        acc[qi] = 0.0f;
    }

    const bool has_dim = threadIdx.x < head_dim;
    const int out_dim = threadIdx.x;

    for (int k_start = 0; k_start < max_valid; k_start += BLOCK_K) {
        const int chunk_len = min(BLOCK_K, max_valid - k_start);

        for (int pos = warp_id; pos < chunk_len; pos += num_warps) {
            const int kv_pos = k_start + pos;
            const int logical_block = kv_pos / block_size;
            const int slot = kv_pos % block_size;
            const int physical_block = my_block_table[logical_block];
            const __half* k_row =
                k_block_pool + physical_block * block_stride_val
                             + slot * kv_stride
                             + kv_head * head_dim;

            float k_reg[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d = lane_id + i * WARP_SIZE;
                k_reg[i] = (i < elems_per_thread && d < head_dim)
                               ? __half2float(k_row[d])
                               : 0.0f;
            }

#pragma unroll
            for (int qi = 0; qi < TILE_Q; ++qi) {
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dot += q_reg[qi][i] * k_reg[i];
                }
                const int valid = pos0 + qi + 1;
                const float score = (qi < actual_tile && kv_pos < valid)
                                        ? warp_reduce_sum(dot) * scale
                                        : -1e20f;
                if (lane_id == 0) {
                    s_scores[qi * BLOCK_K + pos] = score;
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int qi = 0; qi < TILE_Q; ++qi) {
            float thread_max = -1e20f;
            for (int i = threadIdx.x; i < chunk_len; i += blockDim.x) {
                thread_max = fmaxf(thread_max, s_scores[qi * BLOCK_K + i]);
            }
            const float bmax = block_reduce_max(thread_max);
            if (threadIdx.x == 0) s_chunk_max[qi] = bmax;
            __syncthreads();

            const float chunk_max = s_chunk_max[qi];
            if (chunk_max <= -1e19f) {
                __syncthreads();
                continue;
            }
            const float new_m = fmaxf(m[qi], chunk_max);

            float thread_sum = 0.0f;
            for (int i = threadIdx.x; i < chunk_len; i += blockDim.x) {
                thread_sum += expf(s_scores[qi * BLOCK_K + i] - new_m);
            }
            const float bsum = block_reduce_sum(thread_sum);
            if (threadIdx.x == 0) s_chunk_sum[qi] = bsum;
            __syncthreads();

            const float alpha = (m[qi] <= -1e19f) ? 0.0f : expf(m[qi] - new_m);
            if (has_dim) acc[qi] *= alpha;

            if (has_dim) {
                for (int pos = 0; pos < chunk_len; ++pos) {
                    const int kv_pos = k_start + pos;
                    const int logical_block = kv_pos / block_size;
                    const int slot = kv_pos % block_size;
                    const int physical_block = my_block_table[logical_block];
                    const __half* v_row =
                        v_block_pool + physical_block * block_stride_val
                                     + slot * kv_stride
                                     + kv_head * head_dim;
                    const float w = expf(s_scores[qi * BLOCK_K + pos] - new_m);
                    acc[qi] += w * __half2float(v_row[out_dim]);
                }
            }

            m[qi] = new_m;
            l[qi] = l[qi] * alpha + s_chunk_sum[qi];
            __syncthreads();
        }
    }

    if (has_dim) {
#pragma unroll
        for (int qi = 0; qi < TILE_Q; ++qi) {
            if (qi < actual_tile) {
                const int token_global = seq_q_start + tile_start + qi;
                __half* out_ptr =
                    output + ((size_t)token_global * num_q_heads + q_head) * head_dim;
                const float inv_l = l[qi] > 0.0f ? 1.0f / l[qi] : 0.0f;
                out_ptr[out_dim] = __float2half(acc[qi] * inv_l);
            }
        }
    }
}

// ─── Split-K variant ─────────────────────────────────────────────────
// Phase 1: each block scans 1/N of the kv_len for one (head, q_token, split).
// Stores partial output (unnormalized) + local m + local l for online merge.
//
// Microbench (RTX 4090, scripts/microbench_split_k.cu) shows this wins
// big at low concurrency (c=4 +103% kv_len=384, c=1 +801% kv_len=4096)
// and modest at c=16+ long context (kv_len>=768 → +6-16%). Marginal/neg
// at c=16 short kv_len.
//
// Output layout (caller-allocated):
//   partial_out : [M_total, num_q_heads, num_splits, head_dim]  (float32)
//   partial_m   : [M_total, num_q_heads, num_splits]            (float32)
//   partial_l   : [M_total, num_q_heads, num_splits]            (float32)
//
// Grid:  (num_q_heads, M_total, num_splits)
// Block: (256,)
// Shared: ceil(kv_len / num_splits) * sizeof(float)
extern "C" __global__ void paged_varlen_attn_split_k_phase1_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_block_pool,
    const __half* __restrict__ v_block_pool,
    const int*    __restrict__ cu_seqlens_q,
    const int*    __restrict__ pos_offsets,
    const int*    __restrict__ block_tables,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    const int num_seqs,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const int block_size,
    const float scale,
    const int num_splits
) {
    const int q_head = blockIdx.x;
    const int token_global = blockIdx.y;
    const int split_id = blockIdx.z;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    int seq_idx = 0;
    while (seq_idx + 1 < num_seqs &&
           cu_seqlens_q[seq_idx + 1] <= token_global) {
        seq_idx++;
    }
    const int local_idx = token_global - cu_seqlens_q[seq_idx];
    const int abs_kv_pos = pos_offsets[seq_idx] + local_idx;
    const int valid_kv_len = abs_kv_pos + 1;

    const int chunk = (valid_kv_len + num_splits - 1) / num_splits;
    const int split_start = split_id * chunk;
    const int split_end = min(split_start + chunk, valid_kv_len);
    const int my_len = split_end - split_start;

    const int out_idx =
        (token_global * num_q_heads + q_head) * num_splits + split_id;

    if (my_len <= 0) {
        if (threadIdx.x == 0) {
            partial_m[out_idx] = -1e20f;
            partial_l[out_idx] = 0.0f;
        }
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            partial_out[out_idx * head_dim + d] = 0.0f;
        }
        return;
    }

    const int kv_stride = num_kv_heads * head_dim;
    const int block_stride_val = block_size * kv_stride;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const __half* q_ptr =
        q + ((size_t)token_global * num_q_heads + q_head) * head_dim;

    extern __shared__ float smem[];

    // Q into registers (warp-cooperative).
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

    // Step 1: Q·K^T over our chunk.
    for (int pos = warp_id; pos < my_len; pos += num_warps) {
        int kv_pos = split_start + pos;
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
            smem[pos] = score;
    }
    __syncthreads();

    // Step 2: local max, exp, local sum.
    float thread_max = -1e20f;
    for (int i = threadIdx.x; i < my_len; i += blockDim.x)
        thread_max = fmaxf(thread_max, smem[i]);
    float local_max = block_reduce_max(thread_max);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    local_max = s_max;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < my_len; i += blockDim.x) {
        float v = expf(smem[i] - local_max);
        smem[i] = v;
        thread_sum += v;
    }
    __syncthreads();
    float local_sum = block_reduce_sum(thread_sum);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        partial_m[out_idx] = local_max;
        partial_l[out_idx] = s_sum;
    }

    // Step 3: weighted V sum (UNNORMALIZED — store sum_i exp(s_i - m_local) * v_i).
    // The reduce kernel will normalize using the global max + global denom.
    float* out_ptr_p = partial_out + out_idx * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < my_len; i++) {
            int kv_pos = split_start + i;
            int logical_block = kv_pos / block_size;
            int slot = kv_pos % block_size;
            int physical_block = my_block_table[logical_block];
            const __half* v_row =
                v_block_pool + physical_block * block_stride_val
                             + slot * kv_stride
                             + kv_head * head_dim;
            acc += smem[i] * __half2float(v_row[d]);
        }
        out_ptr_p[d] = acc;
    }
}

// Phase 2: merge num_splits partial outputs per (head, token) using online softmax.
//
// Grid:  (num_q_heads, M_total)
// Block: (128,) — head_dim parallelism is enough.
extern "C" __global__ void paged_varlen_split_k_reduce_f16(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __half* __restrict__ output,
    const int num_q_heads,
    const int head_dim,
    const int num_splits
) {
    const int q_head = blockIdx.x;
    const int token = blockIdx.y;

    // Global max across splits for stable softmax.
    float gmax = -1e20f;
    for (int s = 0; s < num_splits; s++) {
        int idx = (token * num_q_heads + q_head) * num_splits + s;
        gmax = fmaxf(gmax, partial_m[idx]);
    }

    // Global denominator: sum_s exp(m_s - gmax) * l_s.
    float gden = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        int idx = (token * num_q_heads + q_head) * num_splits + s;
        gden += expf(partial_m[idx] - gmax) * partial_l[idx];
    }
    float inv_gden = 1.0f / gden;

    // Merge weighted outputs.
    // Each partial holds sum_i exp(s_i - m_local) * v_i — to merge into
    // global softmax-weighted output, scale by exp(m_local - gmax) /
    // gden, NOT by l_local separately (it's already absorbed in the
    // unnormalized partial).
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            int idx = (token * num_q_heads + q_head) * num_splits + s;
            float w = expf(partial_m[idx] - gmax) * inv_gden;
            acc += w * partial_out[idx * head_dim + d];
        }
        output[((size_t)token * num_q_heads + q_head) * head_dim + d]
            = __float2half(acc);
    }
}
