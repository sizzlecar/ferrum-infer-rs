// Standalone native-CUDA correctness probe for paged_varlen_attention
// sliding-window semantics.
//
// This intentionally bypasses Cargo and model loading. It links directly
// against crates/ferrum-kernels/kernels/paged_varlen_attention.cu and compares
// both the one-pass and split-K kernels against a CPU reference for
// sliding_window=0 (full causal) and sliding_window=3.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" __global__ void paged_varlen_attn_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_block_pool,
    const __half* __restrict__ v_block_pool,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ pos_offsets,
    const int* __restrict__ block_tables,
    __half* __restrict__ output,
    const int num_seqs,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const int block_size,
    const float scale,
    const int sliding_window);

extern "C" __global__ void paged_varlen_attn_split_k_phase1_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_block_pool,
    const __half* __restrict__ v_block_pool,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ pos_offsets,
    const int* __restrict__ block_tables,
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
    const int sliding_window,
    const int num_splits);

extern "C" __global__ void paged_varlen_split_k_reduce_f16(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __half* __restrict__ output,
    const int num_q_heads,
    const int head_dim,
    const int num_splits);

#define CUDA_CHECK(expr)                                                            \
    do {                                                                           \
        cudaError_t err__ = (expr);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(err__));                               \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

namespace {

constexpr int kNumSeqs = 2;
constexpr int kTotalQ = 5;
constexpr int kNumQHeads = 4;
constexpr int kNumKvHeads = 2;
constexpr int kHeadDim = 16;
constexpr int kBlockSize = 4;
constexpr int kMaxBlocksPerSeq = 3;
constexpr int kPoolBlocks = 3;
constexpr int kThreads = 256;
constexpr int kNumSplits = 3;
constexpr float kScale = 0.125f;
constexpr float kTolerance = 2.5e-3f;

size_t q_index(int token, int q_head, int d) {
    return (static_cast<size_t>(token) * kNumQHeads + q_head) * kHeadDim + d;
}

size_t kv_index(int physical_block, int slot, int kv_head, int d) {
    const int kv_stride = kNumKvHeads * kHeadDim;
    return static_cast<size_t>(physical_block) * kBlockSize * kv_stride +
           static_cast<size_t>(slot) * kv_stride +
           static_cast<size_t>(kv_head) * kHeadDim + d;
}

float q_value(int token, int q_head, int d) {
    return 0.02f * static_cast<float>((token + 1) * (q_head + 2)) +
           0.003f * static_cast<float>(d - 7);
}

float k_value(int physical_block, int slot, int kv_head, int d) {
    return 0.015f * static_cast<float>((physical_block + 1) * (slot + 2)) +
           0.004f * static_cast<float>(kv_head + 1) -
           0.002f * static_cast<float>(d);
}

float v_value(int physical_block, int slot, int kv_head, int d) {
    return 0.01f * static_cast<float>((physical_block + 3) * (kv_head + 1)) +
           0.006f * static_cast<float>(slot + 1) +
           0.0015f * static_cast<float>(d - 5);
}

std::vector<__half> make_q() {
    std::vector<__half> q(kTotalQ * kNumQHeads * kHeadDim);
    for (int token = 0; token < kTotalQ; token++) {
        for (int q_head = 0; q_head < kNumQHeads; q_head++) {
            for (int d = 0; d < kHeadDim; d++) {
                q[q_index(token, q_head, d)] = __float2half(q_value(token, q_head, d));
            }
        }
    }
    return q;
}

std::vector<__half> make_k_or_v(bool is_v) {
    std::vector<__half> data(kPoolBlocks * kBlockSize * kNumKvHeads * kHeadDim);
    for (int block = 0; block < kPoolBlocks; block++) {
        for (int slot = 0; slot < kBlockSize; slot++) {
            for (int kv_head = 0; kv_head < kNumKvHeads; kv_head++) {
                for (int d = 0; d < kHeadDim; d++) {
                    const float val = is_v ? v_value(block, slot, kv_head, d)
                                           : k_value(block, slot, kv_head, d);
                    data[kv_index(block, slot, kv_head, d)] = __float2half(val);
                }
            }
        }
    }
    return data;
}

std::vector<float> cpu_reference(
    const std::vector<__half>& q,
    const std::vector<__half>& k,
    const std::vector<__half>& v,
    const std::vector<int>& cu_seqlens_q,
    const std::vector<int>& pos_offsets,
    const std::vector<int>& block_tables,
    int sliding_window) {
    std::vector<float> out(kTotalQ * kNumQHeads * kHeadDim, 0.0f);

    for (int token = 0; token < kTotalQ; token++) {
        int seq_idx = 0;
        while (seq_idx + 1 < kNumSeqs && cu_seqlens_q[seq_idx + 1] <= token) {
            seq_idx++;
        }
        const int local_idx = token - cu_seqlens_q[seq_idx];
        const int abs_kv_pos = pos_offsets[seq_idx] + local_idx;
        const int valid_kv_len = abs_kv_pos + 1;
        const int attend_start =
            (sliding_window > 0 && valid_kv_len > sliding_window)
                ? valid_kv_len - sliding_window
                : 0;
        const int active_kv_len = valid_kv_len - attend_start;

        for (int q_head = 0; q_head < kNumQHeads; q_head++) {
            const int kv_head = q_head / (kNumQHeads / kNumKvHeads);
            std::vector<float> scores(active_kv_len);
            float max_score = -1.0e20f;
            for (int i = 0; i < active_kv_len; i++) {
                const int kv_pos = attend_start + i;
                const int logical_block = kv_pos / kBlockSize;
                const int slot = kv_pos % kBlockSize;
                const int physical_block =
                    block_tables[seq_idx * kMaxBlocksPerSeq + logical_block];
                float dot = 0.0f;
                for (int d = 0; d < kHeadDim; d++) {
                    dot += __half2float(q[q_index(token, q_head, d)]) *
                           __half2float(k[kv_index(physical_block, slot, kv_head, d)]);
                }
                scores[i] = dot * kScale;
                max_score = std::max(max_score, scores[i]);
            }

            float denom = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - max_score);
                denom += score;
            }

            for (int d = 0; d < kHeadDim; d++) {
                float acc = 0.0f;
                for (int i = 0; i < active_kv_len; i++) {
                    const int kv_pos = attend_start + i;
                    const int logical_block = kv_pos / kBlockSize;
                    const int slot = kv_pos % kBlockSize;
                    const int physical_block =
                        block_tables[seq_idx * kMaxBlocksPerSeq + logical_block];
                    acc += (scores[i] / denom) *
                           __half2float(v[kv_index(physical_block, slot, kv_head, d)]);
                }
                out[q_index(token, q_head, d)] = acc;
            }
        }
    }
    return out;
}

float max_abs_error(const std::vector<__half>& got, const std::vector<float>& want) {
    float max_err = 0.0f;
    for (size_t i = 0; i < got.size(); i++) {
        max_err = std::max(max_err, std::fabs(__half2float(got[i]) - want[i]));
    }
    return max_err;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

template <typename T>
T* copy_to_device(const std::vector<T>& host) {
    T* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return dev;
}

std::vector<__half> run_one_pass(
    const __half* d_q,
    const __half* d_k,
    const __half* d_v,
    const int* d_cu,
    const int* d_pos,
    const int* d_blocks,
    int sliding_window) {
    __half* d_out = nullptr;
    const size_t out_count = kTotalQ * kNumQHeads * kHeadDim;
    CUDA_CHECK(cudaMalloc(&d_out, out_count * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_out, 0, out_count * sizeof(__half)));

    const int max_active = sliding_window > 0 ? sliding_window : 7;
    paged_varlen_attn_f16<<<dim3(kNumQHeads, kTotalQ, 1), kThreads,
                            max_active * sizeof(float)>>>(
        d_q, d_k, d_v, d_cu, d_pos, d_blocks, d_out, kNumSeqs, kNumQHeads, kNumKvHeads,
        kHeadDim, kMaxBlocksPerSeq, kBlockSize, kScale, sliding_window);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> out(out_count);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_count * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    return out;
}

std::vector<__half> run_split_k(
    const __half* d_q,
    const __half* d_k,
    const __half* d_v,
    const int* d_cu,
    const int* d_pos,
    const int* d_blocks,
    int sliding_window) {
    __half* d_out = nullptr;
    float* d_partial_out = nullptr;
    float* d_partial_m = nullptr;
    float* d_partial_l = nullptr;
    const size_t out_count = kTotalQ * kNumQHeads * kHeadDim;
    const size_t partial_count = kTotalQ * kNumQHeads * kNumSplits;
    CUDA_CHECK(cudaMalloc(&d_out, out_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_partial_out, partial_count * kHeadDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_m, partial_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_l, partial_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, out_count * sizeof(__half)));

    const int max_active = sliding_window > 0 ? sliding_window : 7;
    const int max_chunk = (max_active + kNumSplits - 1) / kNumSplits;
    paged_varlen_attn_split_k_phase1_f16<<<dim3(kNumQHeads, kTotalQ, kNumSplits), kThreads,
                                           max_chunk * sizeof(float)>>>(
        d_q, d_k, d_v, d_cu, d_pos, d_blocks, d_partial_out, d_partial_m, d_partial_l,
        kNumSeqs, kNumQHeads, kNumKvHeads, kHeadDim, kMaxBlocksPerSeq, kBlockSize, kScale,
        sliding_window, kNumSplits);
    CUDA_CHECK(cudaGetLastError());
    paged_varlen_split_k_reduce_f16<<<dim3(kNumQHeads, kTotalQ, 1), 128>>>(
        d_partial_out, d_partial_m, d_partial_l, d_out, kNumQHeads, kHeadDim, kNumSplits);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> out(out_count);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_count * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial_out));
    CUDA_CHECK(cudaFree(d_partial_m));
    CUDA_CHECK(cudaFree(d_partial_l));
    return out;
}

} // namespace

int main() {
    const std::vector<int> cu_seqlens_q = {0, 3, 5};
    const std::vector<int> pos_offsets = {2, 5};
    const std::vector<int> block_tables = {
        1, 0, 2,
        2, 0, 1,
    };
    const std::vector<__half> q = make_q();
    const std::vector<__half> k = make_k_or_v(false);
    const std::vector<__half> v = make_k_or_v(true);

    __half* d_q = copy_to_device(q);
    __half* d_k = copy_to_device(k);
    __half* d_v = copy_to_device(v);
    int* d_cu = copy_to_device(cu_seqlens_q);
    int* d_pos = copy_to_device(pos_offsets);
    int* d_blocks = copy_to_device(block_tables);

    const std::vector<float> ref_full =
        cpu_reference(q, k, v, cu_seqlens_q, pos_offsets, block_tables, 0);
    const std::vector<float> ref_window =
        cpu_reference(q, k, v, cu_seqlens_q, pos_offsets, block_tables, 3);
    const float semantic_delta = max_abs_diff(ref_full, ref_window);
    if (semantic_delta < 1.0e-4f) {
        std::fprintf(stderr,
                     "sliding_window=3 did not differ from full causal enough: %.8f\n",
                     semantic_delta);
        return 1;
    }

    bool ok = true;
    for (int sliding_window : {0, 3}) {
        const std::vector<float>& ref = sliding_window == 0 ? ref_full : ref_window;
        const std::vector<__half> one_pass =
            run_one_pass(d_q, d_k, d_v, d_cu, d_pos, d_blocks, sliding_window);
        const std::vector<__half> split_k =
            run_split_k(d_q, d_k, d_v, d_cu, d_pos, d_blocks, sliding_window);
        const float one_pass_err = max_abs_error(one_pass, ref);
        const float split_k_err = max_abs_error(split_k, ref);
        std::printf("sliding_window=%d one_pass_max_abs_err=%.8f split_k_max_abs_err=%.8f\n",
                    sliding_window, one_pass_err, split_k_err);
        ok = ok && one_pass_err <= kTolerance && split_k_err <= kTolerance;
    }

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_cu));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_blocks));

    if (!ok) {
        std::fprintf(stderr, "VERDICT: paged varlen window correctness FAIL\n");
        return 1;
    }
    std::printf("semantic_delta_full_vs_window=%.8f\n", semantic_delta);
    std::printf("VERDICT: paged varlen window correctness PASS\n");
    return 0;
}
