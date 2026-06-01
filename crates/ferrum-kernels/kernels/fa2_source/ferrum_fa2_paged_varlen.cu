// Ferrum-owned FlashAttention-style paged-varlen C ABI.
//
// This file intentionally replaces the earlier build-time dependency on
// external FlashAttention source templates. It exports the same
// `ferrum_fa2_paged_varlen_fwd` symbol used by the direct-FFI and source-linked
// Rust paths, but launches only in-repo CUDA kernels.
//
// Supported product shape:
//   q             : [total_q_tokens, num_heads, head_dim]
//   k/v pools     : [num_blocks, block_size, num_kv_heads, head_dim]
//   cu_seqlens_q  : [num_seqs + 1]
//   seq_lens      : final per-sequence KV lengths after this batch
//   block_tables  : [num_seqs, max_blocks_per_seq]
//   out           : [total_q_tokens, num_heads, head_dim]
//   lse           : [num_heads, total_q_tokens] scratch/output LSE
//
// Kernel shape:
//   one block per (query token, q head), with 8 warps partitioning the KV
//   range. Each warp keeps online-softmax state `(m, l, acc[head_dim])` for its
//   KV stripe, then the block merges those stripes with the same stable
//   log-sum-exp formula. This avoids the old full-score shared buffer and the
//   second V pass, while keeping the C ABI small and repo-owned.
//
// Guard token for source-boundary checks: warp-partition online softmax.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#define FERRUM_FA2_WARP_SIZE 32

namespace {

void set_err(char *err_buf, size_t err_buf_len, const char *fmt, ...) {
    if (err_buf == nullptr || err_buf_len == 0) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(err_buf, err_buf_len, fmt, args);
    va_end(args);
    err_buf[err_buf_len - 1] = '\0';
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = FERRUM_FA2_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ const __half *fa_kv_ptr(
    const __half *pool,
    int physical_block,
    int slot,
    int kv_head,
    int dim,
    int block_size,
    int num_kv_heads,
    int head_dim) {
    const long long block_stride =
        static_cast<long long>(block_size) * num_kv_heads * head_dim;
    const long long slot_stride = static_cast<long long>(num_kv_heads) * head_dim;
    return pool + static_cast<long long>(physical_block) * block_stride +
           static_cast<long long>(slot) * slot_stride +
           static_cast<long long>(kv_head) * head_dim + dim;
}

__global__ void ferrum_fa2_paged_varlen_kernel_f16(
    const __half *__restrict__ q,
    const __half *__restrict__ k_block_pool,
    const __half *__restrict__ v_block_pool,
    __half *__restrict__ output,
    float *__restrict__ lse,
    const int *__restrict__ cu_seqlens_q,
    const int *__restrict__ seq_lens,
    const int *__restrict__ block_tables,
    int num_seqs,
    int total_q_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale) {
    const int q_head = blockIdx.x;
    const int token_global = blockIdx.y;
    if (q_head >= num_heads || token_global >= total_q_tokens) {
        return;
    }

    int seq_idx = 0;
    while (seq_idx + 1 < num_seqs && cu_seqlens_q[seq_idx + 1] <= token_global) {
        seq_idx++;
    }

    const int seq_q_start = cu_seqlens_q[seq_idx];
    const int seq_q_end = cu_seqlens_q[seq_idx + 1];
    const int q_len = seq_q_end - seq_q_start;
    const int local_idx = token_global - seq_q_start;
    const int prior_kv_len = seq_lens[seq_idx] - q_len;
    const int valid_kv_len = prior_kv_len + local_idx + 1;
    if (valid_kv_len <= 0) {
        return;
    }

    const int q_per_kv = num_heads / num_kv_heads;
    const int kv_head = q_head / q_per_kv;
    const int *my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const __half *q_ptr =
        q + (static_cast<size_t>(token_global) * num_heads + q_head) * head_dim;
    __half *out_ptr =
        output + (static_cast<size_t>(token_global) * num_heads + q_head) * head_dim;

    const int lane_id = threadIdx.x % FERRUM_FA2_WARP_SIZE;
    const int warp_id = threadIdx.x / FERRUM_FA2_WARP_SIZE;
    const int num_warps = blockDim.x / FERRUM_FA2_WARP_SIZE;
    const int elems_per_thread = (head_dim + FERRUM_FA2_WARP_SIZE - 1) / FERRUM_FA2_WARP_SIZE;
    extern __shared__ float smem[];
    float *partial_out = smem;
    float *partial_m = partial_out + num_warps * head_dim;
    float *partial_l = partial_m + num_warps;
    float q_reg[8];
    float acc[8];
    float local_m = -1.0e20f;
    float local_l = 0.0f;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = lane_id + i * FERRUM_FA2_WARP_SIZE;
        q_reg[i] = (i < elems_per_thread && d < head_dim) ? __half2float(q_ptr[d]) : 0.0f;
        acc[i] = 0.0f;
    }

    for (int kv_pos = warp_id; kv_pos < valid_kv_len; kv_pos += num_warps) {
        const int logical_block = kv_pos / block_size;
        const int slot = kv_pos - logical_block * block_size;
        const int physical_block = my_block_table[logical_block];
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = lane_id + i * FERRUM_FA2_WARP_SIZE;
            if (i < elems_per_thread && d < head_dim) {
                dot += q_reg[i] *
                       __half2float(*fa_kv_ptr(k_block_pool, physical_block, slot, kv_head,
                                               d, block_size, num_kv_heads, head_dim));
            }
        }
        float score = warp_reduce_sum(dot);
        score = __shfl_sync(0xffffffff, score, 0) * scale;

        const float new_m = fmaxf(local_m, score);
        const float alpha = expf(local_m - new_m);
        const float beta = expf(score - new_m);
        local_l = local_l * alpha + beta;
        local_m = new_m;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = lane_id + i * FERRUM_FA2_WARP_SIZE;
            if (i < elems_per_thread && d < head_dim) {
                const float v_val =
                    __half2float(*fa_kv_ptr(v_block_pool, physical_block, slot, kv_head,
                                            d, block_size, num_kv_heads, head_dim));
                acc[i] = acc[i] * alpha + beta * v_val;
            }
        }
    }

    if (lane_id == 0) {
        partial_m[warp_id] = local_m;
        partial_l[warp_id] = local_l;
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = lane_id + i * FERRUM_FA2_WARP_SIZE;
        if (i < elems_per_thread && d < head_dim) {
            partial_out[warp_id * head_dim + d] = acc[i];
        }
    }
    __syncthreads();

    __shared__ float s_global_max;
    __shared__ float s_global_sum;
    if (threadIdx.x == 0) {
        float global_m = -1.0e20f;
        for (int w = 0; w < num_warps; ++w) {
            global_m = fmaxf(global_m, partial_m[w]);
        }
        float global_l = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            if (partial_l[w] > 0.0f) {
                global_l += expf(partial_m[w] - global_m) * partial_l[w];
            }
        }
        s_global_max = global_m;
        s_global_sum = global_l;
        lse[static_cast<size_t>(q_head) * total_q_tokens + token_global] =
            logf(global_l) + global_m;
    }
    __syncthreads();

    const float inv_sum = 1.0f / s_global_sum;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            if (partial_l[w] > 0.0f) {
                const float weight = expf(partial_m[w] - s_global_max) * inv_sum;
                acc += weight * partial_out[w * head_dim + d];
            }
        }
        out_ptr[d] = __float2half(acc);
    }
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int ferrum_fa2_paged_varlen_fwd(
    const void *q,
    const void *k,
    const void *v,
    void *out,
    void *lse,
    const void *cu_seqlens_q,
    const void *seq_lens,
    const void *block_tables,
    int num_seqs,
    int total_q_tokens,
    int max_q_len,
    int max_kv_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    void *stream,
    char *err_buf,
    size_t err_buf_len) {
    if (err_buf != nullptr && err_buf_len > 0) {
        err_buf[0] = '\0';
    }
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || lse == nullptr ||
        cu_seqlens_q == nullptr || seq_lens == nullptr || block_tables == nullptr) {
        set_err(err_buf, err_buf_len, "null pointer argument");
        return 1;
    }
    if (num_seqs <= 0 || total_q_tokens <= 0 || max_q_len <= 0 || max_kv_len <= 0) {
        set_err(err_buf, err_buf_len,
                "invalid sizes: num_seqs=%d total_q=%d max_q=%d max_k=%d",
                num_seqs, total_q_tokens, max_q_len, max_kv_len);
        return 2;
    }
    if (head_dim != 128 || block_size != 16 || num_kv_heads <= 0 ||
        num_heads <= 0 || num_heads % num_kv_heads != 0) {
        set_err(err_buf, err_buf_len,
                "unsupported shape: heads=%d kv_heads=%d head_dim=%d block_size=%d",
                num_heads, num_kv_heads, head_dim, block_size);
        return 3;
    }

    const int threads = 256;
    const dim3 grid(num_heads, total_q_tokens, 1);
    const dim3 block(threads, 1, 1);
    const int num_warps = threads / FERRUM_FA2_WARP_SIZE;
    const size_t shared_bytes =
        static_cast<size_t>(num_warps * head_dim + num_warps * 2) * sizeof(float);
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    ferrum_fa2_paged_varlen_kernel_f16<<<grid, block, shared_bytes,
                                         reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const __half *>(q),
        static_cast<const __half *>(k),
        static_cast<const __half *>(v),
        static_cast<__half *>(out),
        static_cast<float *>(lse),
        static_cast<const int *>(cu_seqlens_q),
        static_cast<const int *>(seq_lens),
        static_cast<const int *>(block_tables),
        num_seqs,
        total_q_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_blocks_per_seq,
        scale);

    const cudaError_t launch_err = cudaPeekAtLastError();
    if (launch_err != cudaSuccess) {
        set_err(err_buf, err_buf_len, "native FA2 launch failed: %s",
                cudaGetErrorString(launch_err));
        return 4;
    }
    return 0;
}
