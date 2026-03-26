// Single-query decode attention with GQA.
// Warp-cooperative Q·K dot product + standard 3-pass softmax.
//
// Grid:  (num_q_heads,)
// Block: (256,)  — threads split work over KV positions and head_dim
//
// K/V cache layout: [seq_len, num_kv_heads, head_dim] (candle's layout)

#include <cuda_fp16.h>

#define WARP_SIZE 32

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

// Block-level reduction via shared memory
__inline__ __device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__inline__ __device__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e20f;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

extern "C" __global__ void decode_attention_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half* __restrict__ output,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_kv_len,
    const int valid_kv_len,
    const float scale
) {
    const int q_head = blockIdx.x;
    const int num_kv_groups = num_q_heads / num_kv_heads;
    const int kv_head = q_head / num_kv_groups;

    const __half* q_ptr = q + q_head * head_dim;
    __half* out_ptr = output + q_head * head_dim;

    // K/V cache layout: [seq_len, num_kv_heads, head_dim]
    const int kv_stride = num_kv_heads * head_dim;

    // Shared memory for attention scores
    extern __shared__ float s_scores[];

    // ====== Step 1: Q·K^T with warp-cooperative dot product ======
    // Each thread loads its portion of Q into registers
    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float q_reg[4]; // max 128/32 = 4
    for (int i = 0; i < elems_per_thread; i++) {
        int d = threadIdx.x % WARP_SIZE + i * WARP_SIZE;
        q_reg[i] = (d < head_dim) ? __half2float(q_ptr[d]) : 0.0f;
    }

    // Each thread participates in computing scores for multiple KV positions
    // Threads within a warp cooperate on the dot product
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    float local_max = -1e20f;
    for (int kv_pos = warp_id; kv_pos < valid_kv_len; kv_pos += num_warps) {
        const __half* k_row = k_cache + kv_pos * kv_stride + kv_head * head_dim;
        float partial = 0.0f;
        for (int i = 0; i < elems_per_thread; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (d < head_dim)
                partial += q_reg[i] * __half2float(k_row[d]);
        }
        float score = warp_reduce_sum(partial) * scale;
        if (lane_id == 0) {
            s_scores[kv_pos] = score;
            local_max = fmaxf(local_max, score);
        }
    }
    __syncthreads();

    // ====== Step 2: Softmax ======
    // Find global max
    // Each thread reads its assigned scores
    float thread_max = -1e20f;
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x)
        thread_max = fmaxf(thread_max, s_scores[i]);

    __shared__ float s_global_max;
    float bmax = block_reduce_max(thread_max);
    if (threadIdx.x == 0) s_global_max = bmax;
    __syncthreads();
    float global_max = s_global_max;

    // Exp and sum
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

    // Normalize
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x)
        s_scores[i] *= inv_sum;
    __syncthreads();

    // ====== Step 3: Weighted sum of V ======
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < valid_kv_len; kv_pos++) {
            acc += s_scores[kv_pos] * __half2float(
                v_cache[kv_pos * kv_stride + kv_head * head_dim + d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}
