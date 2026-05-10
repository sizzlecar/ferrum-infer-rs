// INT8 paged decode attention — Dim 5 INT8 KV.
//
// Mirrors paged_decode_attention.cu but K/V are stored as int8 + per-token
// per-kv-head FP16 scales (vLLM-style symmetric quantization).
//
// Block pool layouts:
//   k_block_pool / v_block_pool: int8
//     [max_blocks * block_size * num_kv_heads * head_dim]
//   k_scales_pool / v_scales_pool: __half
//     [max_blocks * block_size * num_kv_heads]
//
// Address translation (kv_pos → ptrs):
//   logical_block = kv_pos / block_size
//   slot          = kv_pos % block_size
//   physical      = block_table[logical_block]
//   data_offset   = physical * block_size * num_kv_heads * head_dim
//                 + slot * num_kv_heads * head_dim
//                 + kv_head * head_dim
//   scale_offset  = physical * block_size * num_kv_heads
//                 + slot * num_kv_heads
//                 + kv_head
//
// Each int8 element is dequantized as `scale * (float)int8_val`.
//
// Grid:  (num_q_heads,)
// Block: (256,)
// Shared: valid_kv_len * sizeof(float) for attention scores

#include "common.cuh"

#define WARP_SIZE 32

extern "C" __global__ void paged_decode_attention_int8(
    const __half* __restrict__ q,                 // [num_q_heads, head_dim] FP16
    const int8_t* __restrict__ k_block_pool,      // INT8 paged
    const int8_t* __restrict__ v_block_pool,      // INT8 paged
    const __half* __restrict__ k_scales_pool,     // FP16 per-token scales
    const __half* __restrict__ v_scales_pool,     // FP16 per-token scales
    const int*    __restrict__ block_table,       // [max_logical_blocks]
    __half* __restrict__ output,                  // [num_q_heads, head_dim]
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int valid_kv_len,
    const int block_size,
    const float scale                             // attention scale = 1/sqrt(hd)
) {
    const int q_head = blockIdx.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_stride = num_kv_heads * head_dim;
    const int block_stride = block_size * kv_stride;
    const int scale_block_stride = block_size * num_kv_heads;
    const int scale_kv_stride = num_kv_heads;

    const __half* q_ptr = q + q_head * head_dim;
    __half* out_ptr = output + q_head * head_dim;

    // Dynamic shmem layout (caller allocates 3 × valid_kv_len floats):
    //   [0 .. valid_kv_len)             — s_scores  (Q·K^T → softmax → weights)
    //   [valid_kv_len .. 2*valid_kv_len) — s_k_scales (per-(token, kv_head) FP16 scale, broadcast across head_dim threads in Step 1)
    //   [2*valid_kv_len .. 3*valid_kv_len) — s_v_scales (same broadcast pattern in Step 3)
    //
    // Why preload scales: Step 3 has 256 threads × valid_kv_len global
    // loads of `v_scales_pool[scale_offset]` — same value across the
    // 256 threads of a (q_head, kv_head) block. With shmem cache the
    // 256-way broadcast collapses into 1 shmem read per kv_pos. Step 1
    // sees a similar (smaller) win.
    extern __shared__ float s_dyn[];
    float* s_scores   = s_dyn;
    float* s_k_scales = s_dyn + valid_kv_len;
    float* s_v_scales = s_dyn + 2 * valid_kv_len;

    // Block-cooperative prologue: one-shot load of all per-(token, kv_head)
    // K/V scales into shmem. The (kv_head) component is the same for the
    // whole block — only the (token) varies — so we need one load per
    // kv_pos. Coalesces well: consecutive threads load consecutive kv_pos.
    for (int kv_pos = threadIdx.x; kv_pos < valid_kv_len; kv_pos += blockDim.x) {
        int logical_block  = kv_pos / block_size;
        int slot           = kv_pos % block_size;
        int physical_block = block_table[logical_block];
        int scale_offset   = physical_block * scale_block_stride
                           + slot * scale_kv_stride
                           + kv_head;
        s_k_scales[kv_pos] = __half2float(k_scales_pool[scale_offset]);
        s_v_scales[kv_pos] = __half2float(v_scales_pool[scale_offset]);
    }

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
    __syncthreads();  // wait for s_k_scales / s_v_scales to be visible

    // ====== Step 1: Q·K^T (dequantizing K on the fly) ======
    for (int kv_pos = warp_id; kv_pos < valid_kv_len; kv_pos += num_warps) {
        int logical_block = kv_pos / block_size;
        int slot = kv_pos % block_size;
        int physical_block = block_table[logical_block];

        const int8_t* k_row = k_block_pool
            + physical_block * block_stride
            + slot * kv_stride
            + kv_head * head_dim;
        const float k_scale = s_k_scales[kv_pos];  // shmem read (was per-warp global)

        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (i < elems_per_thread && d < head_dim) {
                float k_dq = k_scale * (float)k_row[d];
                dot += q_reg[i] * k_dq;
            }
        }
        float score = warp_reduce_sum(dot) * scale;
        if (lane_id == 0)
            s_scores[kv_pos] = score;
    }
    __syncthreads();

    // ====== Step 2: Softmax (identical to FP16) ======
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

    // ====== Step 3: Weighted V sum (dequantizing V on the fly) ======
    // Pre-fold v_scale into the score: weight[kv_pos] = s_scores[kv_pos] * s_v_scales[kv_pos].
    // This collapses Step 3's inner-loop ops to (1 shmem + 1 INT8 load + 1 cast + 1 fma)
    // vs the original (1 shmem + 1 global v_scale + 1 cast + 2 fma + scalar mul).
    // The fold uses s_scores in-place — safe because no thread re-reads
    // the un-folded values after Step 2's softmax.
    for (int kv_pos = threadIdx.x; kv_pos < valid_kv_len; kv_pos += blockDim.x) {
        s_scores[kv_pos] *= s_v_scales[kv_pos];
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < valid_kv_len; kv_pos++) {
            int logical_block = kv_pos / block_size;
            int slot = kv_pos % block_size;
            int physical_block = block_table[logical_block];

            const int8_t* v_row = v_block_pool
                + physical_block * block_stride
                + slot * kv_stride
                + kv_head * head_dim;

            acc += s_scores[kv_pos] * (float)v_row[d];
        }
        out_ptr[d] = __float2half(acc);
    }
}

// ===================== INT8 KV append =====================
//
// Take FP16 K/V tokens, compute per-token per-kv-head symmetric scale
// `s = max(|x|)/127`, write INT8 quantized values to the paged cache.
//
// Layouts:
//   k_in / v_in  : [num_tokens, num_kv_heads, head_dim] FP16
//   k_out_pool / v_out_pool : same paged INT8 layout as the kernel above
//   k_scales_pool / v_scales_pool : same FP16 scales
//   slot_mapping : [num_tokens] i32 — flat KV position for each input token,
//                  i.e. slot_mapping[t] = (physical_block * block_size + slot).
//                  Caller fills this from block_table + per-token (block_idx, slot).
//
// Grid:  (num_tokens, num_kv_heads)
// Block: (head_dim,) — one thread per head element. Assumes head_dim ≤ 256.
//
// Algorithm per (token, kv_head):
//   1. Each thread holds one head element of K and V.
//   2. block-wide reductions to find max(|K|) and max(|V|).
//   3. Compute scale_k = max_k / 127, scale_v = max_v / 127 (in thread 0).
//   4. Each thread quantizes its element and writes int8 + the shared scale.

extern "C" __global__ void int8_kv_cache_append(
    const __half* __restrict__ k_in,             // FP16 [num_tokens, num_kv_heads, head_dim]
    const __half* __restrict__ v_in,             // FP16 same shape
    int8_t* __restrict__ k_out_pool,             // INT8 paged
    int8_t* __restrict__ v_out_pool,             // INT8 paged
    __half* __restrict__ k_scales_pool,          // FP16 [pool_tokens, num_kv_heads]
    __half* __restrict__ v_scales_pool,          // FP16 same shape
    const int*    __restrict__ slot_mapping,     // i32 [num_tokens]
    const int num_kv_heads,
    const int head_dim,
    const int num_tokens
) {
    const int token = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;
    if (token >= num_tokens || kv_head >= num_kv_heads || tid >= head_dim) return;

    const int slot = slot_mapping[token];

    // Input pointers (token-major, packed by head_dim).
    const int in_offset = (token * num_kv_heads + kv_head) * head_dim;
    float kv_k = __half2float(k_in[in_offset + tid]);
    float kv_v = __half2float(v_in[in_offset + tid]);

    // Block-wide max(|x|) for K and V.
    __shared__ float s_max_k, s_max_v;

    float my_abs_k = fabsf(kv_k);
    float my_abs_v = fabsf(kv_v);

    // block_reduce_max from common.cuh works on warp+block scope but here
    // the block has at most head_dim threads (≤256). Do it manually.
    __shared__ float s_warp_max_k[8];
    __shared__ float s_warp_max_v[8];

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        my_abs_k = fmaxf(my_abs_k, __shfl_xor_sync(0xffffffff, my_abs_k, offset));
        my_abs_v = fmaxf(my_abs_v, __shfl_xor_sync(0xffffffff, my_abs_v, offset));
    }
    if (lane_id == 0) {
        s_warp_max_k[warp_id] = my_abs_k;
        s_warp_max_v[warp_id] = my_abs_v;
    }
    __syncthreads();

    // First warp reduces the per-warp maxes.
    if (warp_id == 0) {
        int num_warps = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
        float wm_k = (lane_id < num_warps) ? s_warp_max_k[lane_id] : 0.0f;
        float wm_v = (lane_id < num_warps) ? s_warp_max_v[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            wm_k = fmaxf(wm_k, __shfl_xor_sync(0xffffffff, wm_k, offset));
            wm_v = fmaxf(wm_v, __shfl_xor_sync(0xffffffff, wm_v, offset));
        }
        if (lane_id == 0) {
            // Avoid scale=0 (would make dequant always 0); clamp to 1e-8.
            s_max_k = fmaxf(wm_k, 1e-8f);
            s_max_v = fmaxf(wm_v, 1e-8f);
        }
    }
    __syncthreads();

    const float scale_k = s_max_k / 127.0f;
    const float scale_v = s_max_v / 127.0f;
    const float inv_scale_k = 1.0f / scale_k;
    const float inv_scale_v = 1.0f / scale_v;

    // Quantize-and-clamp, write int8.
    int8_t qk = (int8_t)__float2int_rn(fmaxf(-127.0f, fminf(127.0f, kv_k * inv_scale_k)));
    int8_t qv = (int8_t)__float2int_rn(fmaxf(-127.0f, fminf(127.0f, kv_v * inv_scale_v)));

    const int out_offset = slot * num_kv_heads * head_dim + kv_head * head_dim + tid;
    k_out_pool[out_offset] = qk;
    v_out_pool[out_offset] = qv;

    // Thread 0 writes the per-(token, head) scales.
    if (tid == 0) {
        const int scale_offset = slot * num_kv_heads + kv_head;
        k_scales_pool[scale_offset] = __float2half(scale_k);
        v_scales_pool[scale_offset] = __float2half(scale_v);
    }
}
