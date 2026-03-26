// Paged decode attention with GQA support.
//
// Same algorithm as decode_attention.cu, but K/V are stored in a paged
// block pool with block-table indirection instead of contiguous memory.
//
// Block pool layout (per layer, per K/V):
//   [max_blocks, block_size, num_kv_heads, head_dim]
//   Physical block i starts at: i * block_size * num_kv_heads * head_dim
//
// Block table: block_table[logical_block] = physical_block_id
// Address translation: kv_pos → (logical_block, slot) → physical offset
//   logical_block = kv_pos / block_size    (shift when block_size is power of 2)
//   slot          = kv_pos % block_size    (mask when block_size is power of 2)
//   physical      = block_table[logical_block]
//   offset        = physical * block_stride + slot * kv_stride + kv_head * head_dim
//
// Grid:  (num_q_heads,)
// Block: (256,)
// Shared: valid_kv_len * sizeof(float) for attention scores

#include <cuda_fp16.h>

#define WARP_SIZE 32

__inline__ __device__ float warp_reduce_sum_pa(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__inline__ __device__ float block_reduce_max_pa(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e20f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__inline__ __device__ float block_reduce_sum_pa(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum_pa(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_pa(val);
    return val;
}

// Translate a logical KV position to a global memory pointer in the block pool.
__inline__ __device__ const __half* paged_kv_ptr(
    const __half* block_pool,
    const int* block_table,
    int kv_pos,
    int block_size,
    int block_stride,  // = block_size * num_kv_heads * head_dim
    int kv_stride,     // = num_kv_heads * head_dim
    int kv_head,
    int head_dim
) {
    int logical_block = kv_pos / block_size;
    int slot = kv_pos % block_size;
    int physical_block = block_table[logical_block];
    return block_pool
        + physical_block * block_stride
        + slot * kv_stride
        + kv_head * head_dim;
}

extern "C" __global__ void paged_decode_attention_f16(
    const __half* __restrict__ q,              // [num_q_heads, head_dim]
    const __half* __restrict__ k_block_pool,   // [max_blocks * block_size, num_kv_heads, head_dim]
    const __half* __restrict__ v_block_pool,   // same layout
    const int*    __restrict__ block_table,    // [max_logical_blocks]
    __half* __restrict__ output,               // [num_q_heads, head_dim]
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int valid_kv_len,
    const int block_size,
    const float scale
) {
    const int q_head = blockIdx.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_stride = num_kv_heads * head_dim;
    const int block_stride = block_size * kv_stride;

    const __half* q_ptr = q + q_head * head_dim;
    __half* out_ptr = output + q_head * head_dim;

    extern __shared__ float s_scores[];

    // Load Q into registers (warp-cooperative)
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

    // ====== Step 1: Q·K^T via paged KV ======
    for (int kv_pos = warp_id; kv_pos < valid_kv_len; kv_pos += num_warps) {
        const __half* k_row = paged_kv_ptr(
            k_block_pool, block_table, kv_pos,
            block_size, block_stride, kv_stride, kv_head, head_dim);

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (i < elems_per_thread && d < head_dim)
                dot += q_reg[i] * __half2float(k_row[d]);
        }
        float score = warp_reduce_sum_pa(dot) * scale;
        if (lane_id == 0)
            s_scores[kv_pos] = score;
    }
    __syncthreads();

    // ====== Step 2: Softmax ======
    float thread_max = -1e20f;
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x)
        thread_max = fmaxf(thread_max, s_scores[i]);

    __shared__ float s_global_max;
    float bmax = block_reduce_max_pa(thread_max);
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
    float bsum = block_reduce_sum_pa(thread_sum);
    if (threadIdx.x == 0) s_global_sum = bsum;
    __syncthreads();
    float inv_sum = 1.0f / s_global_sum;

    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x)
        s_scores[i] *= inv_sum;
    __syncthreads();

    // ====== Step 3: Weighted V sum via paged KV ======
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < valid_kv_len; kv_pos++) {
            const __half* v_row = paged_kv_ptr(
                v_block_pool, block_table, kv_pos,
                block_size, block_stride, kv_stride, kv_head, head_dim);
            acc += s_scores[kv_pos] * __half2float(v_row[d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}

// ===================== Flash Decode variant (split-K + paged) =====================

#define MAX_SPLITS 32

// Phase 1: Split-K paged attention
extern "C" __global__ void paged_flash_decode_attn_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_block_pool,
    const __half* __restrict__ v_block_pool,
    const int*    __restrict__ block_table,
    float* __restrict__ partial_out,       // [num_q_heads, num_splits, head_dim]
    float* __restrict__ partial_m,         // [num_q_heads, num_splits]
    float* __restrict__ partial_l,         // [num_q_heads, num_splits]
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int valid_kv_len,
    const int block_size,
    const float scale,
    const int num_splits
) {
    const int q_head = blockIdx.x;
    const int split_id = blockIdx.y;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_stride = num_kv_heads * head_dim;
    const int block_stride_val = block_size * kv_stride;

    const int chunk_size = (valid_kv_len + num_splits - 1) / num_splits;
    const int split_start = split_id * chunk_size;
    const int split_end = min(split_start + chunk_size, valid_kv_len);
    const int my_len = split_end - split_start;

    const int out_idx = q_head * num_splits + split_id;

    if (my_len <= 0) {
        if (threadIdx.x == 0) {
            partial_m[out_idx] = -1e20f;
            partial_l[out_idx] = 0.0f;
        }
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            partial_out[out_idx * head_dim + d] = 0.0f;
        return;
    }

    const __half* q_ptr = q + q_head * head_dim;
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

    extern __shared__ float smem[];

    // Step 1: Q·K^T for this chunk (paged)
    for (int pos = warp_id; pos < my_len; pos += num_warps) {
        int kv_pos = split_start + pos;
        const __half* k_row = paged_kv_ptr(
            k_block_pool, block_table, kv_pos,
            block_size, block_stride_val, kv_stride, kv_head, head_dim);
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (i < elems_per_thread && d < head_dim)
                dot += q_reg[i] * __half2float(k_row[d]);
        }
        float score = warp_reduce_sum_pa(dot) * scale;
        if (lane_id == 0)
            smem[pos] = score;
    }
    __syncthreads();

    // Step 2: Local softmax
    float thread_max = -1e20f;
    for (int i = threadIdx.x; i < my_len; i += blockDim.x)
        thread_max = fmaxf(thread_max, smem[i]);
    float local_max = block_reduce_max_pa(thread_max);
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
    float local_sum = block_reduce_sum_pa(thread_sum);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        partial_m[out_idx] = local_max;
        partial_l[out_idx] = s_sum;
    }

    // Step 3: Weighted V sum (paged, unnormalized)
    float* out_ptr = partial_out + out_idx * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < my_len; i++) {
            int kv_pos = split_start + i;
            const __half* v_row = paged_kv_ptr(
                v_block_pool, block_table, kv_pos,
                block_size, block_stride_val, kv_stride, kv_head, head_dim);
            acc += smem[i] * __half2float(v_row[d]);
        }
        out_ptr[d] = acc;
    }
}

// Phase 2 reduce is identical to flash_decode_attention.cu — reuse that kernel.
// The reduce kernel only reads from partial_out/m/l which are layout-independent.
