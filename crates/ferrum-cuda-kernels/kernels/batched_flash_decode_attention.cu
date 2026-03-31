// Batched Flash Decoding for multi-query attention with GQA support.
//
// Same split-K algorithm as flash_decode_attention.cu but processes
// all batch items in a single launch. Each batch item can have a
// different valid_kv_len (and thus different effective num_splits).
//
// Phase 1: Grid (num_q_heads, max_num_splits, batch_size)
// Phase 2: Grid (num_q_heads, batch_size)
//
// K/V caches passed as device pointer arrays (one per batch item).

#include <cuda_fp16.h>

#define WARP_SIZE 32
#define MAX_SPLITS 32

__inline__ __device__ float warp_reduce_sum_bfd(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__inline__ __device__ float block_reduce_max_bfd(float val) {
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

__inline__ __device__ float block_reduce_sum_bfd(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum_bfd(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_bfd(val);
    return val;
}

// Phase 1: Split-K attention for all batch items.
// Grid: (num_q_heads, max_num_splits, batch_size)
extern "C" __global__ void batched_flash_decode_attn_f16(
    const __half* __restrict__ q_all,            // [B, num_q_heads, head_dim]
    const __half* const* __restrict__ k_cache_ptrs, // [B] pointers
    const __half* const* __restrict__ v_cache_ptrs, // [B] pointers
    float* __restrict__ partial_out,             // [B, num_q_heads, max_splits, head_dim]
    float* __restrict__ partial_m,               // [B, num_q_heads, max_splits]
    float* __restrict__ partial_l,               // [B, num_q_heads, max_splits]
    const int* __restrict__ valid_kv_lens,       // [B]
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    const int max_num_splits
) {
    const int q_head = blockIdx.x;
    const int split_id = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int valid_kv_len = valid_kv_lens[batch_idx];
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_stride = num_kv_heads * head_dim;

    // Per-batch output offsets
    const int q_dim = num_q_heads * head_dim;
    const int out_base = batch_idx * num_q_heads * max_num_splits;
    const int out_idx = out_base + q_head * max_num_splits + split_id;

    // KV range for this split
    const int chunk_size = (valid_kv_len + max_num_splits - 1) / max_num_splits;
    const int split_start = split_id * chunk_size;
    const int split_end = min(split_start + chunk_size, valid_kv_len);
    const int my_len = split_end - split_start;

    if (my_len <= 0) {
        if (threadIdx.x == 0) {
            partial_m[out_idx] = -1e20f;
            partial_l[out_idx] = 0.0f;
        }
        int po_base = (out_base + q_head * max_num_splits + split_id) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            partial_out[po_base + d] = 0.0f;
        return;
    }

    const __half* q_ptr = q_all + batch_idx * q_dim + q_head * head_dim;
    const __half* k_cache = k_cache_ptrs[batch_idx];
    const __half* v_cache = v_cache_ptrs[batch_idx];

    // Load Q into registers
    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float q_reg[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = (threadIdx.x % WARP_SIZE) + i * WARP_SIZE;
        q_reg[i] = (i < elems_per_thread && d < head_dim)
            ? __half2float(q_ptr[d]) : 0.0f;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    extern __shared__ float smem[];

    // Q·K^T for this chunk
    for (int pos = warp_id; pos < my_len; pos += num_warps) {
        const int kv_pos = split_start + pos;
        const __half* k_row = k_cache + kv_pos * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (i < elems_per_thread && d < head_dim)
                dot += q_reg[i] * __half2float(k_row[d]);
        }
        float score = warp_reduce_sum_bfd(dot) * scale;
        if (lane_id == 0)
            smem[pos] = score;
    }
    __syncthreads();

    // Local softmax
    float thread_max = -1e20f;
    for (int i = threadIdx.x; i < my_len; i += blockDim.x)
        thread_max = fmaxf(thread_max, smem[i]);
    float local_max = block_reduce_max_bfd(thread_max);

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

    float local_sum = block_reduce_sum_bfd(thread_sum);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        partial_m[out_idx] = local_max;
        partial_l[out_idx] = s_sum;
    }

    // Weighted V accumulation
    int po_base = (out_base + q_head * max_num_splits + split_id) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < my_len; i++) {
            acc += smem[i] * __half2float(
                v_cache[(split_start + i) * kv_stride + kv_head * head_dim + d]);
        }
        partial_out[po_base + d] = acc;
    }
}

// Phase 2: Reduce across splits for all batch items.
// Grid: (num_q_heads, batch_size)
extern "C" __global__ void batched_flash_decode_reduce_f16(
    const float* __restrict__ partial_out, // [B, num_q_heads, max_splits, head_dim]
    const float* __restrict__ partial_m,   // [B, num_q_heads, max_splits]
    const float* __restrict__ partial_l,   // [B, num_q_heads, max_splits]
    __half* __restrict__ output,           // [B, num_q_heads, head_dim]
    const int head_dim,
    const int max_num_splits
) {
    const int q_head = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int num_q_heads_grid = gridDim.x;

    const int base = batch_idx * num_q_heads_grid * max_num_splits
                   + q_head * max_num_splits;

    float global_max = -1e20f;
    for (int s = 0; s < max_num_splits; s++)
        global_max = fmaxf(global_max, partial_m[base + s]);

    float rescale[MAX_SPLITS];
    float global_sum = 0.0f;
    for (int s = 0; s < max_num_splits; s++) {
        rescale[s] = expf(partial_m[base + s] - global_max);
        global_sum += partial_l[base + s] * rescale[s];
    }
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    const int q_dim = num_q_heads_grid * head_dim;
    __half* out_ptr = output + batch_idx * q_dim + q_head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < max_num_splits; s++)
            acc += partial_out[(base + s) * head_dim + d] * rescale[s];
        out_ptr[d] = __float2half(acc * inv_sum);
    }
}
