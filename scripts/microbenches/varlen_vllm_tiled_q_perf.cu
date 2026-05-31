// Standalone microbench: vLLM-layout paged varlen attention, current
// one-block-per-query kernel vs a Q-tiled kernel that reuses K/V loads across
// several causal prefill query tokens from the same sequence.
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 scripts/microbenches/varlen_vllm_tiled_q_perf.cu \
//     -o /tmp/varlen_vllm_tiled_q_perf
//
// This is intentionally independent from ferrum's Rust build. It is a
// CUDA-level go/no-go for a full FlashAttention-style vLLM-layout prefill
// kernel, not a production implementation.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define WARP_SIZE 32
#define VLLM_X 8
#define CHECK(call)                                                             \
    do {                                                                        \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err__));                                 \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    v = warp_reduce_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    v = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    v = warp_reduce_max(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    v = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -1e20f;
    if (wid == 0) v = warp_reduce_max(v);
    return v;
}

__device__ __forceinline__ const __half* k_vllm_ptr(
    const __half* pool,
    int physical_block,
    int kv_head,
    int dim,
    int slot,
    int head_dim,
    int num_kv_heads,
    int block_size)
{
    int x_chunk = dim / VLLM_X;
    int x_off = dim - x_chunk * VLLM_X;
    long long per_block = (long long)num_kv_heads * head_dim * block_size;
    long long per_head = (long long)head_dim * block_size;
    long long off = (long long)physical_block * per_block +
                    (long long)kv_head * per_head +
                    (long long)x_chunk * (block_size * VLLM_X) +
                    (long long)slot * VLLM_X + x_off;
    return pool + off;
}

__device__ __forceinline__ const __half* v_vllm_ptr(
    const __half* pool,
    int physical_block,
    int kv_head,
    int dim,
    int slot,
    int head_dim,
    int num_kv_heads,
    int block_size)
{
    long long per_block = (long long)num_kv_heads * head_dim * block_size;
    long long per_head = (long long)head_dim * block_size;
    long long off = (long long)physical_block * per_block +
                    (long long)kv_head * per_head +
                    (long long)dim * block_size + slot;
    return pool + off;
}

__global__ void paged_varlen_attn_vllm_ref_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_pool,
    const __half* __restrict__ v_pool,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ pos_offsets,
    const int* __restrict__ block_tables,
    __half* __restrict__ output,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    int block_size,
    float scale)
{
    int q_head = blockIdx.x;
    int token_global = blockIdx.y;
    int kv_head = q_head / (num_q_heads / num_kv_heads);

    int seq_idx = 0;
    while (seq_idx + 1 < num_seqs && cu_seqlens_q[seq_idx + 1] <= token_global) {
        seq_idx++;
    }
    int local_idx = token_global - cu_seqlens_q[seq_idx];
    int valid_kv_len = pos_offsets[seq_idx] + local_idx + 1;
    if (valid_kv_len <= 0) return;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const __half* q_ptr =
        q + ((size_t)token_global * num_q_heads + q_head) * head_dim;
    __half* out_ptr =
        output + ((size_t)token_global * num_q_heads + q_head) * head_dim;

    extern __shared__ float s_scores[];
    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float q_reg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int d = lane + i * WARP_SIZE;
        q_reg[i] = (i < elems_per_thread && d < head_dim) ? __half2float(q_ptr[d]) : 0.0f;
    }

    for (int kv_pos = warp; kv_pos < valid_kv_len; kv_pos += num_warps) {
        int logical_block = kv_pos / block_size;
        int slot = kv_pos % block_size;
        int physical_block = my_block_table[logical_block];
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int d = lane + i * WARP_SIZE;
            if (i < elems_per_thread && d < head_dim) {
                dot += q_reg[i] *
                       __half2float(*k_vllm_ptr(k_pool, physical_block, kv_head,
                                                d, slot, head_dim, num_kv_heads,
                                                block_size));
            }
        }
        float score = warp_reduce_sum(dot) * scale;
        if (lane == 0) s_scores[kv_pos] = score;
    }
    __syncthreads();

    float thread_max = -1e20f;
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x) {
        thread_max = fmaxf(thread_max, s_scores[i]);
    }
    __shared__ float s_global_max;
    float bmax = block_reduce_max(thread_max);
    if (threadIdx.x == 0) s_global_max = bmax;
    __syncthreads();

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x) {
        float val = expf(s_scores[i] - s_global_max);
        s_scores[i] = val;
        thread_sum += val;
    }
    __syncthreads();
    __shared__ float s_global_sum;
    float bsum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0) s_global_sum = bsum;
    __syncthreads();
    float inv_sum = 1.0f / s_global_sum;
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x) {
        s_scores[i] *= inv_sum;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < valid_kv_len; ++kv_pos) {
            int logical_block = kv_pos / block_size;
            int slot = kv_pos % block_size;
            int physical_block = my_block_table[logical_block];
            acc += s_scores[kv_pos] *
                   __half2float(*v_vllm_ptr(v_pool, physical_block, kv_head,
                                            d, slot, head_dim, num_kv_heads,
                                            block_size));
        }
        out_ptr[d] = __float2half(acc);
    }
}

template <int TILE_Q>
__global__ void paged_varlen_attn_vllm_tiled_q_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_pool,
    const __half* __restrict__ v_pool,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ pos_offsets,
    const int* __restrict__ block_tables,
    const int* __restrict__ tile_seqs,
    const int* __restrict__ tile_starts,
    __half* __restrict__ output,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    int block_size,
    int score_stride,
    float scale)
{
    int q_head = blockIdx.x;
    int tile_id = blockIdx.y;
    int seq_idx = tile_seqs[tile_id];
    int tile_start = tile_starts[tile_id];
    int seq_q_start = cu_seqlens_q[seq_idx];
    int q_len = cu_seqlens_q[seq_idx + 1] - seq_q_start;

    int actual_tile = min(TILE_Q, q_len - tile_start);
    int kv_head = q_head / (num_q_heads / num_kv_heads);
    int pos0 = pos_offsets[seq_idx] + tile_start;
    int max_valid = pos0 + actual_tile;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    extern __shared__ float s_scores[];
    __shared__ float s_global_max[TILE_Q];
    __shared__ float s_global_sum[TILE_Q];

    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float q_reg[TILE_Q][8];

#pragma unroll
    for (int qi = 0; qi < TILE_Q; ++qi) {
        int token_global = seq_q_start + tile_start + qi;
        const __half* q_ptr =
            q + ((size_t)token_global * num_q_heads + q_head) * head_dim;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int d = lane + i * WARP_SIZE;
            q_reg[qi][i] = (qi < actual_tile && i < elems_per_thread && d < head_dim)
                               ? __half2float(q_ptr[d])
                               : 0.0f;
        }
    }

    for (int kv_pos = warp; kv_pos < max_valid; kv_pos += num_warps) {
        int logical_block = kv_pos / block_size;
        int slot = kv_pos % block_size;
        int physical_block = my_block_table[logical_block];
        float k_reg[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int d = lane + i * WARP_SIZE;
            k_reg[i] = (i < elems_per_thread && d < head_dim)
                           ? __half2float(*k_vllm_ptr(k_pool, physical_block, kv_head,
                                                      d, slot, head_dim, num_kv_heads,
                                                      block_size))
                           : 0.0f;
        }

#pragma unroll
        for (int qi = 0; qi < TILE_Q; ++qi) {
            int valid = pos0 + qi + 1;
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                dot += q_reg[qi][i] * k_reg[i];
            }
            float score = warp_reduce_sum(dot) * scale;
            if (lane == 0 && qi < actual_tile && kv_pos < valid) {
                s_scores[qi * score_stride + kv_pos] = score;
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int qi = 0; qi < TILE_Q; ++qi) {
        if (qi >= actual_tile) continue;
        int valid = pos0 + qi + 1;
        float thread_max = -1e20f;
        for (int i = threadIdx.x; i < valid; i += blockDim.x) {
            thread_max = fmaxf(thread_max, s_scores[qi * score_stride + i]);
        }
        float bmax = block_reduce_max(thread_max);
        if (threadIdx.x == 0) s_global_max[qi] = bmax;
        __syncthreads();

        float thread_sum = 0.0f;
        for (int i = threadIdx.x; i < valid; i += blockDim.x) {
            float val = expf(s_scores[qi * score_stride + i] - s_global_max[qi]);
            s_scores[qi * score_stride + i] = val;
            thread_sum += val;
        }
        __syncthreads();
        float bsum = block_reduce_sum(thread_sum);
        if (threadIdx.x == 0) s_global_sum[qi] = bsum;
        __syncthreads();
        float inv_sum = 1.0f / s_global_sum[qi];
        for (int i = threadIdx.x; i < valid; i += blockDim.x) {
            s_scores[qi * score_stride + i] *= inv_sum;
        }
        __syncthreads();
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc[TILE_Q];
#pragma unroll
        for (int qi = 0; qi < TILE_Q; ++qi) acc[qi] = 0.0f;

        for (int kv_pos = 0; kv_pos < max_valid; ++kv_pos) {
            int logical_block = kv_pos / block_size;
            int slot = kv_pos % block_size;
            int physical_block = my_block_table[logical_block];
            float v = __half2float(*v_vllm_ptr(v_pool, physical_block, kv_head,
                                               d, slot, head_dim, num_kv_heads,
                                               block_size));
#pragma unroll
            for (int qi = 0; qi < TILE_Q; ++qi) {
                int valid = pos0 + qi + 1;
                if (qi < actual_tile && kv_pos < valid) {
                    acc[qi] += s_scores[qi * score_stride + kv_pos] * v;
                }
            }
        }

#pragma unroll
        for (int qi = 0; qi < TILE_Q; ++qi) {
            if (qi < actual_tile) {
                int token_global = seq_q_start + tile_start + qi;
                __half* out_ptr =
                    output + ((size_t)token_global * num_q_heads + q_head) * head_dim;
                out_ptr[d] = __float2half(acc[qi]);
            }
        }
    }
}

struct Scenario {
    std::string name;
    std::vector<int> q_lens;
    std::vector<int> pos_offsets;
};

template <typename F>
float time_kernel(F&& launch, int iters) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 20; ++i) {
        launch();
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch();
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms * 1000.0f / iters;
}

void run_scenario(const Scenario& sc) {
    constexpr int num_q_heads = 16;
    constexpr int num_kv_heads = 4;
    constexpr int head_dim = 128;
    constexpr int block_size = 16;
    constexpr int block_threads = 256;
    constexpr int iters = 200;

    int num_seqs = (int)sc.q_lens.size();
    std::vector<int> cu(num_seqs + 1, 0);
    int max_q_len = 0;
    int max_kv_len = 0;
    for (int i = 0; i < num_seqs; ++i) {
        cu[i + 1] = cu[i] + sc.q_lens[i];
        max_q_len = std::max(max_q_len, sc.q_lens[i]);
        max_kv_len = std::max(max_kv_len, sc.pos_offsets[i] + sc.q_lens[i]);
    }
    int total_q = cu.back();
    int max_blocks_per_seq = (max_kv_len + block_size - 1) / block_size;
    int total_blocks = num_seqs * max_blocks_per_seq;
    size_t q_elems = (size_t)total_q * num_q_heads * head_dim;
    size_t pool_elems = (size_t)total_blocks * num_kv_heads * head_dim * block_size;
    size_t out_elems = q_elems;

    std::vector<int> block_tables(num_seqs * max_blocks_per_seq);
    for (int s = 0; s < num_seqs; ++s) {
        for (int b = 0; b < max_blocks_per_seq; ++b) {
            block_tables[s * max_blocks_per_seq + b] = s * max_blocks_per_seq + b;
        }
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<__half> h_q(q_elems), h_k(pool_elems), h_v(pool_elems);
    for (auto& x : h_q) x = __float2half(dist(rng));
    for (auto& x : h_k) x = __float2half(dist(rng));
    for (auto& x : h_v) x = __float2half(dist(rng));

    __half *d_q, *d_k, *d_v, *d_ref, *d_t4, *d_t8;
    int *d_cu, *d_pos, *d_bt, *d_t4_seq, *d_t4_start, *d_t8_seq, *d_t8_start;
    CHECK(cudaMalloc(&d_q, q_elems * sizeof(__half)));
    CHECK(cudaMalloc(&d_k, pool_elems * sizeof(__half)));
    CHECK(cudaMalloc(&d_v, pool_elems * sizeof(__half)));
    CHECK(cudaMalloc(&d_ref, out_elems * sizeof(__half)));
    CHECK(cudaMalloc(&d_t4, out_elems * sizeof(__half)));
    CHECK(cudaMalloc(&d_t8, out_elems * sizeof(__half)));
    CHECK(cudaMalloc(&d_cu, cu.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_pos, sc.pos_offsets.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_bt, block_tables.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_k, h_k.data(), pool_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v, h_v.data(), pool_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cu, cu.data(), cu.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_pos, sc.pos_offsets.data(), sc.pos_offsets.size() * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bt, block_tables.data(), block_tables.size() * sizeof(int),
                     cudaMemcpyHostToDevice));

    auto make_tiles = [&](int tile_q, std::vector<int>& seqs, std::vector<int>& starts) {
        for (int s = 0; s < num_seqs; ++s) {
            for (int start = 0; start < sc.q_lens[s]; start += tile_q) {
                seqs.push_back(s);
                starts.push_back(start);
            }
        }
    };
    std::vector<int> t4_seq, t4_start, t8_seq, t8_start;
    make_tiles(4, t4_seq, t4_start);
    make_tiles(8, t8_seq, t8_start);
    CHECK(cudaMalloc(&d_t4_seq, t4_seq.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_t4_start, t4_start.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_t8_seq, t8_seq.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_t8_start, t8_start.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_t4_seq, t4_seq.data(), t4_seq.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t4_start, t4_start.data(), t4_start.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t8_seq, t8_seq.data(), t8_seq.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t8_start, t8_start.data(), t8_start.size() * sizeof(int), cudaMemcpyHostToDevice));

    float scale = 1.0f / std::sqrt((float)head_dim);
    dim3 ref_grid(num_q_heads, total_q, 1);
    dim3 tile4_grid(num_q_heads, (unsigned)t4_seq.size(), 1);
    dim3 tile8_grid(num_q_heads, (unsigned)t8_seq.size(), 1);
    size_t ref_smem = (size_t)max_kv_len * sizeof(float);
    size_t t4_smem = (size_t)4 * max_kv_len * sizeof(float);
    size_t t8_smem = (size_t)8 * max_kv_len * sizeof(float);

    auto launch_ref = [&]() {
        paged_varlen_attn_vllm_ref_f16<<<ref_grid, block_threads, ref_smem>>>(
            d_q, d_k, d_v, d_cu, d_pos, d_bt, d_ref, num_seqs, num_q_heads,
            num_kv_heads, head_dim, max_blocks_per_seq, block_size, scale);
    };
    auto launch_t4 = [&]() {
        paged_varlen_attn_vllm_tiled_q_f16<4><<<tile4_grid, block_threads, t4_smem>>>(
            d_q, d_k, d_v, d_cu, d_pos, d_bt, d_t4_seq, d_t4_start, d_t4,
            num_q_heads, num_kv_heads, head_dim, max_blocks_per_seq, block_size,
            max_kv_len, scale);
    };
    auto launch_t8 = [&]() {
        paged_varlen_attn_vllm_tiled_q_f16<8><<<tile8_grid, block_threads, t8_smem>>>(
            d_q, d_k, d_v, d_cu, d_pos, d_bt, d_t8_seq, d_t8_start, d_t8,
            num_q_heads, num_kv_heads, head_dim, max_blocks_per_seq, block_size,
            max_kv_len, scale);
    };

    launch_ref();
    launch_t4();
    launch_t8();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<__half> out_ref(out_elems), out_t4(out_elems), out_t8(out_elems);
    CHECK(cudaMemcpy(out_ref.data(), d_ref, out_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(out_t4.data(), d_t4, out_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(out_t8.data(), d_t8, out_elems * sizeof(__half), cudaMemcpyDeviceToHost));

    auto max_abs_err = [&](const std::vector<__half>& got) {
        float err = 0.0f;
        for (size_t i = 0; i < got.size(); ++i) {
            err = std::max(err, std::fabs(__half2float(got[i]) - __half2float(out_ref[i])));
        }
        return err;
    };
    float err4 = max_abs_err(out_t4);
    float err8 = max_abs_err(out_t8);

    float ref_us = time_kernel(launch_ref, iters);
    float t4_us = time_kernel(launch_t4, iters);
    float t8_us = time_kernel(launch_t8, iters);
    CHECK(cudaGetLastError());

    printf("SCENARIO %-18s seqs=%d total_q=%d max_q=%d max_kv=%d blocks/seq=%d\n",
           sc.name.c_str(), num_seqs, total_q, max_q_len, max_kv_len, max_blocks_per_seq);
    printf("  ref      %8.2f us\n", ref_us);
    printf("  tiled_q4 %8.2f us  speedup=%6.2f%%  max_abs_err=%g\n",
           t4_us, (ref_us / t4_us - 1.0f) * 100.0f, err4);
    printf("  tiled_q8 %8.2f us  speedup=%6.2f%%  max_abs_err=%g\n",
           t8_us, (ref_us / t8_us - 1.0f) * 100.0f, err8);

    CHECK(cudaFree(d_q));
    CHECK(cudaFree(d_k));
    CHECK(cudaFree(d_v));
    CHECK(cudaFree(d_ref));
    CHECK(cudaFree(d_t4));
    CHECK(cudaFree(d_t8));
    CHECK(cudaFree(d_cu));
    CHECK(cudaFree(d_pos));
    CHECK(cudaFree(d_bt));
    CHECK(cudaFree(d_t4_seq));
    CHECK(cudaFree(d_t4_start));
    CHECK(cudaFree(d_t8_seq));
    CHECK(cudaFree(d_t8_start));
}

int main() {
    CHECK(cudaSetDevice(0));
    std::vector<Scenario> scenarios = {
        {"prefill_4x256", {256, 256, 256, 256}, {0, 0, 0, 0}},
        {"mixed_3x256_4x1", {256, 256, 256, 1, 1, 1, 1}, {0, 0, 0, 256, 256, 256, 256}},
        {"prefill_4x512", {512, 512, 512, 512}, {0, 0, 0, 0}},
    };

    bool any_positive = false;
    for (const auto& sc : scenarios) {
        run_scenario(sc);
        printf("\n");
    }
    // Human-readable verdict is computed by inspecting the printed speedups.
    // Keep process exit 0 so scripts can archive negative controls cleanly.
    printf("VERDICT: integrate only if a tiled_q row is materially positive on the mixed/prefill shapes above.\n");
    return any_positive ? 0 : 0;
}
