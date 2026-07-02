// Standalone native-CUDA correctness probe for the unified Gemma/Llama
// paged-varlen write + attention chain.
//
// This intentionally bypasses Cargo and model loading. It links directly
// against:
//   - split_qkv_norm_rope_into_paged_cache.cu
//   - paged_varlen_attention.cu
//
// The probe compares:
//   1. varlen split_qkv_norm_rope_into_paged_cache output Q/K/V buffers;
//   2. paged_varlen_attention output consuming those buffers;
// against a CPU reference for qk_mode=1 (QK-norm + half-split RoPE), qk_mode=2,
// qk_mode=3, and both full-causal / sliding-window attention.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

extern "C" __global__ void split_qkv_norm_rope_into_paged_cache_varlen_f16(
    const __half* __restrict__ qkv_base,
    const __half* __restrict__ q_norm_w,
    const __half* __restrict__ k_norm_w,
    const __half* __restrict__ cos_tab,
    const __half* __restrict__ sin_tab,
    __half* __restrict__ q_out,
    __half* __restrict__ cache_k,
    __half* __restrict__ cache_v,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ pos_offsets,
    const int* __restrict__ block_tables,
    const int num_seqs,
    const int m_total,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const float eps,
    const int qk_mode,
    const int block_size,
    const int max_blocks_per_seq);

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

#define CUDA_CHECK(expr)                                                            \
    do {                                                                            \
        cudaError_t err__ = (expr);                                                  \
        if (err__ != cudaSuccess) {                                                  \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                         cudaGetErrorString(err__));                                \
            std::exit(1);                                                           \
        }                                                                           \
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
constexpr int kTotalHeads = kNumQHeads + 2 * kNumKvHeads;
constexpr int kHalfDim = kHeadDim / 2;
constexpr int kMaxRopeRows = 8;
constexpr float kEps = 1.0e-5f;
constexpr float kScale = 0.125f;
constexpr float kSplitTolerance = 1.5e-3f;
constexpr float kAttnTolerance = 3.0e-3f;

size_t qkv_index(int token, int head_idx, int d) {
    return (static_cast<size_t>(token) * kTotalHeads + head_idx) * kHeadDim + d;
}

size_t q_index(int token, int q_head, int d) {
    return (static_cast<size_t>(token) * kNumQHeads + q_head) * kHeadDim + d;
}

size_t kv_index(int physical_block, int slot, int kv_head, int d) {
    const int kv_stride = kNumKvHeads * kHeadDim;
    return static_cast<size_t>(physical_block) * kBlockSize * kv_stride +
           static_cast<size_t>(slot) * kv_stride +
           static_cast<size_t>(kv_head) * kHeadDim + d;
}

size_t rope_index(int pos, int pair) {
    return static_cast<size_t>(pos) * kHalfDim + pair;
}

float qkv_value(int token, int head_idx, int d) {
    return 0.013f * static_cast<float>((token + 1) * (head_idx + 2)) +
           0.0025f * static_cast<float>(d - 5) -
           0.0007f * static_cast<float>((token + d) % 3);
}

float norm_value(bool is_k, int d) {
    const float base = is_k ? 0.91f : 1.07f;
    return base + 0.003f * static_cast<float>((d % 5) - 2);
}

float rope_angle(int pos, int pair) {
    return 0.031f * static_cast<float>(pos) * static_cast<float>(pair + 1);
}

float initial_kv_value(bool is_v, int physical_block, int slot, int kv_head, int d) {
    const float base = is_v ? 0.019f : 0.023f;
    return base * static_cast<float>((physical_block + 1) * (kv_head + 2)) +
           0.004f * static_cast<float>(slot + 1) -
           0.0013f * static_cast<float>(d - 3);
}

std::vector<__half> make_qkv() {
    std::vector<__half> qkv(kTotalQ * kTotalHeads * kHeadDim);
    for (int token = 0; token < kTotalQ; token++) {
        for (int head = 0; head < kTotalHeads; head++) {
            for (int d = 0; d < kHeadDim; d++) {
                qkv[qkv_index(token, head, d)] = __float2half(qkv_value(token, head, d));
            }
        }
    }
    return qkv;
}

std::vector<__half> make_norm(bool is_k) {
    std::vector<__half> norm(kHeadDim);
    for (int d = 0; d < kHeadDim; d++) {
        norm[d] = __float2half(norm_value(is_k, d));
    }
    return norm;
}

std::vector<__half> make_rope(bool make_sin) {
    std::vector<__half> table(kMaxRopeRows * kHalfDim);
    for (int pos = 0; pos < kMaxRopeRows; pos++) {
        for (int pair = 0; pair < kHalfDim; pair++) {
            const float angle = rope_angle(pos, pair);
            table[rope_index(pos, pair)] = __float2half(make_sin ? std::sin(angle) : std::cos(angle));
        }
    }
    return table;
}

std::vector<__half> make_initial_kv(bool is_v) {
    std::vector<__half> data(kPoolBlocks * kBlockSize * kNumKvHeads * kHeadDim);
    for (int block = 0; block < kPoolBlocks; block++) {
        for (int slot = 0; slot < kBlockSize; slot++) {
            for (int kv_head = 0; kv_head < kNumKvHeads; kv_head++) {
                for (int d = 0; d < kHeadDim; d++) {
                    data[kv_index(block, slot, kv_head, d)] =
                        __float2half(initial_kv_value(is_v, block, slot, kv_head, d));
                }
            }
        }
    }
    return data;
}

struct SplitReference {
    std::vector<__half> q;
    std::vector<__half> k;
    std::vector<__half> v;
};

void write_half_split_rope(
    const std::vector<float>& src,
    const std::vector<__half>& norm,
    const std::vector<__half>& cos_tab,
    const std::vector<__half>& sin_tab,
    int pos,
    int qk_mode,
    std::vector<__half>& dst,
    size_t base) {
    float scale = 1.0f;
    if (qk_mode == 1) {
        float sum_sq = 0.0f;
        for (float value : src) {
            sum_sq += value * value;
        }
        scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(kHeadDim) + kEps);
    }

    for (int i = 0; i < kHalfDim; i++) {
        float x0 = src[i];
        float x1 = src[i + kHalfDim];
        if (qk_mode == 1) {
            x0 *= scale * __half2float(norm[i]);
            x1 *= scale * __half2float(norm[i + kHalfDim]);
        }
        const float c = __half2float(cos_tab[rope_index(pos, i)]);
        const float s = __half2float(sin_tab[rope_index(pos, i)]);
        dst[base + i] = __float2half(x0 * c - x1 * s);
        dst[base + i + kHalfDim] = __float2half(x1 * c + x0 * s);
    }
}

void write_interleaved_rope(
    const std::vector<float>& src,
    const std::vector<__half>& cos_tab,
    const std::vector<__half>& sin_tab,
    int pos,
    std::vector<__half>& dst,
    size_t base) {
    for (int i = 0; i < kHalfDim; i++) {
        const int j = 2 * i;
        const float x0 = src[j];
        const float x1 = src[j + 1];
        const float c = __half2float(cos_tab[rope_index(pos, i)]);
        const float s = __half2float(sin_tab[rope_index(pos, i)]);
        dst[base + j] = __float2half(x0 * c - x1 * s);
        dst[base + j + 1] = __float2half(x1 * c + x0 * s);
    }
}

SplitReference cpu_split_reference(
    const std::vector<__half>& qkv,
    const std::vector<__half>& q_norm,
    const std::vector<__half>& k_norm,
    const std::vector<__half>& cos_tab,
    const std::vector<__half>& sin_tab,
    const std::vector<int>& cu_seqlens_q,
    const std::vector<int>& pos_offsets,
    const std::vector<int>& block_tables,
    int qk_mode) {
    SplitReference ref{
        std::vector<__half>(kTotalQ * kNumQHeads * kHeadDim, __float2half(0.0f)),
        make_initial_kv(false),
        make_initial_kv(true),
    };

    for (int global_tok = 0; global_tok < kTotalQ; global_tok++) {
        int seq_idx = 0;
        while (seq_idx + 1 < kNumSeqs && cu_seqlens_q[seq_idx + 1] <= global_tok) {
            seq_idx++;
        }
        const int local_tok = global_tok - cu_seqlens_q[seq_idx];
        const int pos = pos_offsets[seq_idx] + local_tok;

        for (int head_idx = 0; head_idx < kTotalHeads; head_idx++) {
            const bool is_q = head_idx < kNumQHeads;
            const bool is_k = !is_q && head_idx < kNumQHeads + kNumKvHeads;
            const int local_head =
                is_q ? head_idx : (is_k ? head_idx - kNumQHeads : head_idx - kNumQHeads - kNumKvHeads);

            std::vector<float> src(kHeadDim);
            for (int d = 0; d < kHeadDim; d++) {
                src[d] = __half2float(qkv[qkv_index(global_tok, head_idx, d)]);
            }

            std::vector<__half>* dst = nullptr;
            size_t base = 0;
            if (is_q) {
                dst = &ref.q;
                base = q_index(global_tok, local_head, 0);
            } else {
                const int logical_block = pos / kBlockSize;
                const int slot = pos % kBlockSize;
                const int physical_block = block_tables[seq_idx * kMaxBlocksPerSeq + logical_block];
                dst = is_k ? &ref.k : &ref.v;
                base = kv_index(physical_block, slot, local_head, 0);
            }

            if (!is_q && !is_k) {
                for (int d = 0; d < kHeadDim; d++) {
                    (*dst)[base + d] = __float2half(src[d]);
                }
            } else if (qk_mode == 3) {
                write_interleaved_rope(src, cos_tab, sin_tab, pos, *dst, base);
            } else {
                write_half_split_rope(src, is_k ? k_norm : q_norm, cos_tab, sin_tab, pos, qk_mode, *dst, base);
            }
        }
    }

    return ref;
}

std::vector<float> cpu_attention_reference(
    const SplitReference& split,
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
            (sliding_window > 0 && valid_kv_len > sliding_window) ? valid_kv_len - sliding_window : 0;
        const int active_kv_len = valid_kv_len - attend_start;

        for (int q_head = 0; q_head < kNumQHeads; q_head++) {
            const int kv_head = q_head / (kNumQHeads / kNumKvHeads);
            std::vector<float> scores(active_kv_len);
            float max_score = -1.0e20f;
            for (int i = 0; i < active_kv_len; i++) {
                const int kv_pos = attend_start + i;
                const int logical_block = kv_pos / kBlockSize;
                const int slot = kv_pos % kBlockSize;
                const int physical_block = block_tables[seq_idx * kMaxBlocksPerSeq + logical_block];
                float dot = 0.0f;
                for (int d = 0; d < kHeadDim; d++) {
                    dot += __half2float(split.q[q_index(token, q_head, d)]) *
                           __half2float(split.k[kv_index(physical_block, slot, kv_head, d)]);
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
                    const int physical_block = block_tables[seq_idx * kMaxBlocksPerSeq + logical_block];
                    acc += (scores[i] / denom) *
                           __half2float(split.v[kv_index(physical_block, slot, kv_head, d)]);
                }
                out[q_index(token, q_head, d)] = acc;
            }
        }
    }
    return out;
}

template <typename T>
T* copy_to_device(const std::vector<T>& host) {
    T* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return dev;
}

float max_abs_error(const std::vector<__half>& got, const std::vector<__half>& want) {
    float max_err = 0.0f;
    for (size_t i = 0; i < got.size(); i++) {
        max_err = std::max(max_err, std::fabs(__half2float(got[i]) - __half2float(want[i])));
    }
    return max_err;
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

bool run_case(int qk_mode, int sliding_window) {
    const std::vector<int> cu_seqlens_q = {0, 3, 5};
    const std::vector<int> pos_offsets = {2, 5};
    const std::vector<int> block_tables = {
        1, 0, 2,
        2, 0, 1,
    };
    const std::vector<__half> qkv = make_qkv();
    const std::vector<__half> q_norm = make_norm(false);
    const std::vector<__half> k_norm = make_norm(true);
    const std::vector<__half> cos_tab = make_rope(false);
    const std::vector<__half> sin_tab = make_rope(true);
    const std::vector<__half> initial_k = make_initial_kv(false);
    const std::vector<__half> initial_v = make_initial_kv(true);
    const SplitReference split_ref =
        cpu_split_reference(qkv, q_norm, k_norm, cos_tab, sin_tab, cu_seqlens_q, pos_offsets,
                            block_tables, qk_mode);
    const std::vector<float> attn_ref =
        cpu_attention_reference(split_ref, cu_seqlens_q, pos_offsets, block_tables, sliding_window);

    __half* d_qkv = copy_to_device(qkv);
    __half* d_q_norm = copy_to_device(q_norm);
    __half* d_k_norm = copy_to_device(k_norm);
    __half* d_cos = copy_to_device(cos_tab);
    __half* d_sin = copy_to_device(sin_tab);
    int* d_cu = copy_to_device(cu_seqlens_q);
    int* d_pos = copy_to_device(pos_offsets);
    int* d_blocks = copy_to_device(block_tables);
    __half* d_k = copy_to_device(initial_k);
    __half* d_v = copy_to_device(initial_v);

    __half* d_q = nullptr;
    __half* d_attn_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, split_ref.q.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_attn_out, attn_ref.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_q, 0, split_ref.q.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_attn_out, 0, attn_ref.size() * sizeof(__half)));

    split_qkv_norm_rope_into_paged_cache_varlen_f16<<<dim3(kTotalQ, kTotalHeads, 1), 32>>>(
        d_qkv, d_q_norm, d_k_norm, d_cos, d_sin, d_q, d_k, d_v, d_cu, d_pos, d_blocks,
        kNumSeqs, kTotalQ, kNumQHeads, kNumKvHeads, kHeadDim, kEps, qk_mode, kBlockSize,
        kMaxBlocksPerSeq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> got_q(split_ref.q.size());
    std::vector<__half> got_k(split_ref.k.size());
    std::vector<__half> got_v(split_ref.v.size());
    CUDA_CHECK(cudaMemcpy(got_q.data(), d_q, got_q.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_k.data(), d_k, got_k.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_v.data(), d_v, got_v.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    const int max_active = sliding_window > 0 ? sliding_window : 7;
    paged_varlen_attn_f16<<<dim3(kNumQHeads, kTotalQ, 1), kThreads,
                            max_active * sizeof(float)>>>(
        d_q, d_k, d_v, d_cu, d_pos, d_blocks, d_attn_out, kNumSeqs, kNumQHeads, kNumKvHeads,
        kHeadDim, kMaxBlocksPerSeq, kBlockSize, kScale, sliding_window);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> got_attn(attn_ref.size());
    CUDA_CHECK(cudaMemcpy(got_attn.data(), d_attn_out, got_attn.size() * sizeof(__half),
                          cudaMemcpyDeviceToHost));

    const float q_err = max_abs_error(got_q, split_ref.q);
    const float k_err = max_abs_error(got_k, split_ref.k);
    const float v_err = max_abs_error(got_v, split_ref.v);
    const float attn_err = max_abs_error(got_attn, attn_ref);
    const bool ok = q_err <= kSplitTolerance && k_err <= kSplitTolerance &&
                    v_err <= kSplitTolerance && attn_err <= kAttnTolerance;

    std::printf("qk_mode=%d sliding_window=%d q_err=%.8f k_err=%.8f v_err=%.8f attn_err=%.8f %s\n",
                qk_mode, sliding_window, q_err, k_err, v_err, attn_err, ok ? "ok" : "FAIL");

    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_q_norm));
    CUDA_CHECK(cudaFree(d_k_norm));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_cu));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_blocks));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_attn_out));

    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok = run_case(1, 0) && ok;
    ok = run_case(1, 3) && ok;
    ok = run_case(2, 3) && ok;
    ok = run_case(3, 3) && ok;

    const std::vector<int> cu_seqlens_q = {0, 3, 5};
    const std::vector<int> pos_offsets = {2, 5};
    const std::vector<int> block_tables = {
        1, 0, 2,
        2, 0, 1,
    };
    const std::vector<__half> qkv = make_qkv();
    const std::vector<__half> q_norm = make_norm(false);
    const std::vector<__half> k_norm = make_norm(true);
    const std::vector<__half> cos_tab = make_rope(false);
    const std::vector<__half> sin_tab = make_rope(true);
    const SplitReference split_ref =
        cpu_split_reference(qkv, q_norm, k_norm, cos_tab, sin_tab, cu_seqlens_q, pos_offsets,
                            block_tables, 1);
    const std::vector<float> ref_full =
        cpu_attention_reference(split_ref, cu_seqlens_q, pos_offsets, block_tables, 0);
    const std::vector<float> ref_window =
        cpu_attention_reference(split_ref, cu_seqlens_q, pos_offsets, block_tables, 3);
    const float semantic_delta = max_abs_diff(ref_full, ref_window);
    std::printf("qk_mode=1 semantic_delta_full_vs_window=%.8f\n", semantic_delta);
    ok = ok && semantic_delta > 1.0e-4f;

    if (!ok) {
        std::fprintf(stderr, "VERDICT: paged varlen split-qkv correctness FAIL\n");
        return 1;
    }
    std::printf("VERDICT: paged varlen split-qkv correctness PASS\n");
    return 0;
}
