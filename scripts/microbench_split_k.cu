// Standalone microbench: paged_varlen_attention (current) vs naive split-K
// variant on the same inputs. Measures wall time over many iterations to
// decide if split-K rewrite is worth the integration cost.
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 microbench_split_k.cu -o /tmp/mb_split_k
// Run:   /tmp/mb_split_k
//
// Workload mirrors c=16 INT4 decode steady state on Llama-3.1-8B:
//   num_seqs=16, M_total=16, num_q_heads=32, num_kv_heads=8, head_dim=128
//   block_size=16, max_blocks_per_seq=24, kv_len≈384

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

#define WARP_SIZE 32
#define CHECK(call) do { auto e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while(0)

// ─── Reduction helpers ─────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_xor_sync(0xffffffff, v, off);
        v = fmaxf(v, o);
    }
    return v;
}
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float s[32];
    int wid = threadIdx.x / WARP_SIZE;
    int lid = threadIdx.x % WARP_SIZE;
    v = warp_reduce_sum(v);
    if (lid == 0) s[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = lid < (blockDim.x / WARP_SIZE) ? s[lid] : 0.0f;
        v = warp_reduce_sum(v);
    }
    return v;
}
__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float s[32];
    int wid = threadIdx.x / WARP_SIZE;
    int lid = threadIdx.x % WARP_SIZE;
    v = warp_reduce_max(v);
    if (lid == 0) s[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = lid < (blockDim.x / WARP_SIZE) ? s[lid] : -1e20f;
        v = warp_reduce_max(v);
    }
    return v;
}

// ─── Variant A: current paged_varlen_attention ─────────────────────
__global__ void paged_varlen_attn_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_pool,
    const __half* __restrict__ v_pool,
    const int*    __restrict__ cu_seqlens_q,
    const int*    __restrict__ pos_offsets,
    const int*    __restrict__ block_tables,
    __half* __restrict__ output,
    int num_seqs, int num_q_heads, int num_kv_heads, int head_dim,
    int max_blocks_per_seq, int block_size, float scale)
{
    const int q_head = blockIdx.x;
    const int token_global = blockIdx.y;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    int seq_idx = 0;
    while (seq_idx + 1 < num_seqs && cu_seqlens_q[seq_idx + 1] <= token_global) seq_idx++;
    const int local_idx = token_global - cu_seqlens_q[seq_idx];
    const int abs_kv_pos = pos_offsets[seq_idx] + local_idx;
    const int valid_kv_len = abs_kv_pos + 1;
    if (valid_kv_len <= 0) return;

    const int kv_stride = num_kv_heads * head_dim;
    const int block_stride_val = block_size * kv_stride;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    const __half* q_ptr = q + ((size_t)token_global * num_q_heads + q_head) * head_dim;
    __half* out_ptr = output + ((size_t)token_global * num_q_heads + q_head) * head_dim;

    extern __shared__ float s_scores[];

    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float q_reg[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = (threadIdx.x % WARP_SIZE) + i * WARP_SIZE;
        q_reg[i] = (i < elems_per_thread && d < head_dim) ? __half2float(q_ptr[d]) : 0.0f;
    }
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    for (int kv_pos = warp_id; kv_pos < valid_kv_len; kv_pos += num_warps) {
        int logical_block = kv_pos / block_size;
        int slot = kv_pos % block_size;
        int physical_block = my_block_table[logical_block];
        const __half* k_row = k_pool + physical_block * block_stride_val + slot * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (i < elems_per_thread && d < head_dim) dot += q_reg[i] * __half2float(k_row[d]);
        }
        float score = warp_reduce_sum(dot) * scale;
        if (lane_id == 0) s_scores[kv_pos] = score;
    }
    __syncthreads();

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

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < valid_kv_len; kv_pos++) {
            int logical_block = kv_pos / block_size;
            int slot = kv_pos % block_size;
            int physical_block = my_block_table[logical_block];
            const __half* v_row = v_pool + physical_block * block_stride_val + slot * kv_stride + kv_head * head_dim;
            acc += s_scores[kv_pos] * __half2float(v_row[d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}

// ─── Variant B: split-K paged_varlen_attn ───────────────────────────
// Phase 1: per-(head, token, split) block. Each block scans kv_len/N_SPLITS
// elements. Writes partial (m, l, output) for online-softmax merge.
__global__ void paged_varlen_attn_split_k_phase1_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_pool,
    const __half* __restrict__ v_pool,
    const int*    __restrict__ cu_seqlens_q,
    const int*    __restrict__ pos_offsets,
    const int*    __restrict__ block_tables,
    float* __restrict__ partial_out,    // [M_total, num_q_heads, num_splits, head_dim]
    float* __restrict__ partial_m,      // [M_total, num_q_heads, num_splits]
    float* __restrict__ partial_l,      // [M_total, num_q_heads, num_splits]
    int num_seqs, int num_q_heads, int num_kv_heads, int head_dim,
    int max_blocks_per_seq, int block_size, float scale, int num_splits)
{
    const int q_head = blockIdx.x;
    const int token_global = blockIdx.y;
    const int split_id = blockIdx.z;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    int seq_idx = 0;
    while (seq_idx + 1 < num_seqs && cu_seqlens_q[seq_idx + 1] <= token_global) seq_idx++;
    const int local_idx = token_global - cu_seqlens_q[seq_idx];
    const int abs_kv_pos = pos_offsets[seq_idx] + local_idx;
    const int valid_kv_len = abs_kv_pos + 1;

    const int chunk = (valid_kv_len + num_splits - 1) / num_splits;
    const int split_start = split_id * chunk;
    const int split_end = min(split_start + chunk, valid_kv_len);
    const int my_len = split_end - split_start;

    const int out_idx = (token_global * num_q_heads + q_head) * num_splits + split_id;

    if (my_len <= 0) {
        if (threadIdx.x == 0) {
            partial_m[out_idx] = -1e20f;
            partial_l[out_idx] = 0.0f;
        }
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            partial_out[out_idx * head_dim + d] = 0.0f;
        return;
    }

    const int kv_stride = num_kv_heads * head_dim;
    const int block_stride_val = block_size * kv_stride;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const __half* q_ptr = q + ((size_t)token_global * num_q_heads + q_head) * head_dim;

    extern __shared__ float smem[];

    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float q_reg[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = (threadIdx.x % WARP_SIZE) + i * WARP_SIZE;
        q_reg[i] = (i < elems_per_thread && d < head_dim) ? __half2float(q_ptr[d]) : 0.0f;
    }

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    for (int pos = warp_id; pos < my_len; pos += num_warps) {
        int kv_pos = split_start + pos;
        int logical_block = kv_pos / block_size;
        int slot = kv_pos % block_size;
        int physical_block = my_block_table[logical_block];
        const __half* k_row = k_pool + physical_block * block_stride_val + slot * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (i < elems_per_thread && d < head_dim) dot += q_reg[i] * __half2float(k_row[d]);
        }
        float score = warp_reduce_sum(dot) * scale;
        if (lane_id == 0) smem[pos] = score;
    }
    __syncthreads();

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

    float* out_ptr = partial_out + out_idx * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < my_len; i++) {
            int kv_pos = split_start + i;
            int logical_block = kv_pos / block_size;
            int slot = kv_pos % block_size;
            int physical_block = my_block_table[logical_block];
            const __half* v_row = v_pool + physical_block * block_stride_val + slot * kv_stride + kv_head * head_dim;
            acc += smem[i] * __half2float(v_row[d]);
        }
        out_ptr[d] = acc;
    }
}

// Phase 2: reduce N_SPLITS partial results per (head, token).
__global__ void paged_varlen_split_k_reduce_f16(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __half* __restrict__ output,
    int num_q_heads, int head_dim, int num_splits)
{
    const int q_head = blockIdx.x;
    const int token = blockIdx.y;

    // Find global max across splits.
    float gmax = -1e20f;
    for (int s = 0; s < num_splits; s++) {
        int idx = (token * num_q_heads + q_head) * num_splits + s;
        gmax = fmaxf(gmax, partial_m[idx]);
    }

    // Compute global denominator.
    float gden = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        int idx = (token * num_q_heads + q_head) * num_splits + s;
        gden += expf(partial_m[idx] - gmax) * partial_l[idx];
    }
    float inv_gden = 1.0f / gden;

    // Merge weighted outputs.
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            int idx = (token * num_q_heads + q_head) * num_splits + s;
            float w = expf(partial_m[idx] - gmax) * partial_l[idx] * inv_gden;
            // partial_out is unnormalized (sum of softmax(local) * v); to merge,
            // scale by partial_l (its local denominator) was already absorbed
            // since smem stored softmax(local) NOT raw exp. So partial_out
            // is already weighted by softmax LOCAL — to combine with global,
            // multiply by ratio (local_l / global_l) * exp(local_m - global_m).
            // But our partial_out used softmax already-normalized in phase1
            // (smem *= 1/local_sum is missing — see phase1 code: smem stores
            // exp(x - local_max), output is sum(smem * v), NOT normalized).
            // So scale = exp(local_m - global_m) / global_l.
            float scale = expf(partial_m[idx] - gmax) * inv_gden;
            acc += scale * partial_out[idx * head_dim + d];
        }
        output[((size_t)token * num_q_heads + q_head) * head_dim + d] = __float2half(acc);
    }
}

// ─── Test driver ────────────────────────────────────────────────────
struct Workload {
    int num_seqs;        // = M_total since q_len=1 per seq for decode
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int block_size;
    int max_blocks_per_seq;
    int avg_kv_len;       // controls pos_offsets
};

double bench_ms(cudaEvent_t s, cudaEvent_t e, int iters) {
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    return ms / iters;
}

struct Cfg { int c; int kv_len; int max_blocks; int splits; };

int main() {
    std::vector<Cfg> sweep = {
        {16,  256,  20,  2}, {16,  256,  20,  4},
        {16,  384,  28,  2}, {16,  384,  28,  4}, {16,  384,  28,  8},
        {16,  768,  52,  4}, {16,  768,  52,  8},
        {16, 1536, 100,  4}, {16, 1536, 100,  8}, {16, 1536, 100, 16},
        {16, 4096, 260,  8}, {16, 4096, 260, 16},
        {32,  256,  20,  2}, {32,  256,  20,  4},
        {32,  384,  28,  2}, {32,  384,  28,  4},
        {32,  768,  52,  4}, {32, 1536, 100,  8},
        { 8,  384,  28,  2}, { 8,  384,  28,  4},
        { 4,  384,  28,  4}, { 4, 1536, 100,  8},
        { 1, 4096, 260, 16}, { 1, 4096, 260, 32},
    };

    printf("=== paged attention split-K sweep (RTX 4090) ===\n");
    printf("c   kv_len  N | A (ms)   B (ms)   B/A   delta%%\n");
    printf("-------------+--------------------------------\n");

    for (const auto& cfg : sweep) {
    Workload w {
        .num_seqs = cfg.c, .num_q_heads = 32, .num_kv_heads = 8,
        .head_dim = 128, .block_size = 16, .max_blocks_per_seq = cfg.max_blocks,
        .avg_kv_len = cfg.kv_len,
    };
    int M_total = w.num_seqs;
    int kv_len = w.avg_kv_len;
    int pool_blocks = w.num_seqs * w.max_blocks_per_seq;
    float scale = 1.0f / sqrtf((float)w.head_dim);
    int NUM_SPLITS = cfg.splits;

    // Allocate dummy buffers
    size_t q_bytes = (size_t)M_total * w.num_q_heads * w.head_dim * sizeof(__half);
    size_t kv_pool_bytes = (size_t)pool_blocks * w.block_size * w.num_kv_heads * w.head_dim * sizeof(__half);
    size_t out_bytes = q_bytes;

    __half *d_q, *d_k, *d_v, *d_out_a, *d_out_b;
    int *d_cu, *d_pos, *d_blk;
    CHECK(cudaMalloc(&d_q, q_bytes));
    CHECK(cudaMalloc(&d_k, kv_pool_bytes));
    CHECK(cudaMalloc(&d_v, kv_pool_bytes));
    CHECK(cudaMalloc(&d_out_a, out_bytes));
    CHECK(cudaMalloc(&d_out_b, out_bytes));
    CHECK(cudaMalloc(&d_cu, (w.num_seqs + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_pos, w.num_seqs * sizeof(int)));
    CHECK(cudaMalloc(&d_blk, w.num_seqs * w.max_blocks_per_seq * sizeof(int)));

    // Init host data
    std::vector<__half> h_q(M_total * w.num_q_heads * w.head_dim);
    std::vector<__half> h_kv(pool_blocks * w.block_size * w.num_kv_heads * w.head_dim);
    for (auto& v : h_q) v = __float2half((float)rand() / RAND_MAX * 0.1f - 0.05f);
    for (auto& v : h_kv) v = __float2half((float)rand() / RAND_MAX * 0.1f - 0.05f);
    std::vector<int> h_cu(w.num_seqs + 1);
    for (int i = 0; i <= w.num_seqs; i++) h_cu[i] = i; // q_len=1 per seq
    std::vector<int> h_pos(w.num_seqs, kv_len - 1);    // each seq at kv_len-1
    std::vector<int> h_blk(w.num_seqs * w.max_blocks_per_seq);
    for (int s = 0; s < w.num_seqs; s++)
        for (int b = 0; b < w.max_blocks_per_seq; b++)
            h_blk[s * w.max_blocks_per_seq + b] = s * w.max_blocks_per_seq + b;

    CHECK(cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_k, h_kv.data(), kv_pool_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v, h_kv.data(), kv_pool_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cu, h_cu.data(), h_cu.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_pos, h_pos.data(), h_pos.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_blk, h_blk.data(), h_blk.size() * sizeof(int), cudaMemcpyHostToDevice));

    float *d_pmax, *d_psum, *d_pout;
    CHECK(cudaMalloc(&d_pout, (size_t)M_total * w.num_q_heads * NUM_SPLITS * w.head_dim * sizeof(float)));
    CHECK(cudaMalloc(&d_pmax, (size_t)M_total * w.num_q_heads * NUM_SPLITS * sizeof(float)));
    CHECK(cudaMalloc(&d_psum, (size_t)M_total * w.num_q_heads * NUM_SPLITS * sizeof(float)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS = 1000;
    const int WARMUP = 100;

    auto run_a = [&]() {
        size_t shmem_a = (size_t)kv_len * sizeof(float);
        dim3 grid(w.num_q_heads, M_total);
        dim3 block(256);
        paged_varlen_attn_f16<<<grid, block, shmem_a>>>(
            d_q, d_k, d_v, d_cu, d_pos, d_blk, d_out_a,
            w.num_seqs, w.num_q_heads, w.num_kv_heads, w.head_dim,
            w.max_blocks_per_seq, w.block_size, scale);
    };
    auto run_b = [&]() {
        int chunk = (kv_len + NUM_SPLITS - 1) / NUM_SPLITS;
        size_t shmem_b = (size_t)chunk * sizeof(float);
        dim3 grid_p1(w.num_q_heads, M_total, NUM_SPLITS);
        dim3 block_p1(256);
        paged_varlen_attn_split_k_phase1_f16<<<grid_p1, block_p1, shmem_b>>>(
            d_q, d_k, d_v, d_cu, d_pos, d_blk,
            d_pout, d_pmax, d_psum,
            w.num_seqs, w.num_q_heads, w.num_kv_heads, w.head_dim,
            w.max_blocks_per_seq, w.block_size, scale, NUM_SPLITS);
        dim3 grid_p2(w.num_q_heads, M_total);
        dim3 block_p2(128);
        paged_varlen_split_k_reduce_f16<<<grid_p2, block_p2>>>(
            d_pout, d_pmax, d_psum, d_out_b,
            w.num_q_heads, w.head_dim, NUM_SPLITS);
    };

    // Warmup A
    for (int i = 0; i < WARMUP; i++) run_a();
    cudaDeviceSynchronize();
    // Time A
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) run_a();
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    double a_ms = bench_ms(start, stop, ITERS);

    // Warmup B
    for (int i = 0; i < WARMUP; i++) run_b();
    cudaDeviceSynchronize();
    // Time B
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) run_b();
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    double b_ms = bench_ms(start, stop, ITERS);

    printf("%-3d %-6d  %-2d| %.4f  %.4f  %.2fx  %+.1f%%\n",
           w.num_seqs, kv_len, NUM_SPLITS, a_ms, b_ms,
           a_ms / b_ms, (a_ms / b_ms - 1.0) * 100.0);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_out_a); cudaFree(d_out_b);
    cudaFree(d_cu); cudaFree(d_pos); cudaFree(d_blk);
    cudaFree(d_pout); cudaFree(d_pmax); cudaFree(d_psum);
    }
    return 0;
}
