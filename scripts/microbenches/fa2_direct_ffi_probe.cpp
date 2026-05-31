// Direct vLLM FlashAttention-2 FFI probe.
//
// This intentionally bypasses Python and torch.ops: it fills Flash_fwd_params
// directly and calls the flash::run_mha_fwd symbol exported by
// vllm.vllm_flash_attn._vllm_fa2_C.abi3.so. It is a feasibility probe for a
// future C ABI shim; it is not linked into Ferrum.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "flash.h"

namespace flash {
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel);
}

namespace {

constexpr int kNumHeads = 32;
constexpr int kNumKvHeads = 4;
constexpr int kHeadDim = 128;
constexpr int kPageBlockSize = 16;

void cuda_check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

int round_multiple(int x, int m) { return ((x + m - 1) / m) * m; }

int ceil_div(int x, int y) { return (x + y - 1) / y; }

int get_env_int(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::max(1, std::atoi(value));
}

template <typename T>
T *cuda_alloc(size_t count, const char *name) {
    T *ptr = nullptr;
    cuda_check(cudaMalloc(&ptr, count * sizeof(T)), name);
    return ptr;
}

void fill_half(std::vector<half> &dst, float scale) {
    for (size_t i = 0; i < dst.size(); ++i) {
        const float centered = static_cast<float>((i * 17 + 13) % 257) - 128.0f;
        dst[i] = __float2half(centered * scale);
    }
}

struct ProbeCase {
    std::string name;
    std::vector<int> q_lens;
    std::vector<int> kv_lens;
};

void run_case(const ProbeCase &c, int warmup, int iters) {
    const int batch = static_cast<int>(c.q_lens.size());
    if (batch == 0 || c.kv_lens.size() != c.q_lens.size()) {
        throw std::runtime_error("invalid case lengths");
    }

    const int total_q = std::accumulate(c.q_lens.begin(), c.q_lens.end(), 0);
    const int max_q = *std::max_element(c.q_lens.begin(), c.q_lens.end());
    const int max_k = *std::max_element(c.kv_lens.begin(), c.kv_lens.end());
    const int max_blocks_per_seq = ceil_div(max_k, kPageBlockSize);

    std::vector<int> h_cu_q(batch + 1, 0);
    std::vector<int> h_cu_k(batch + 1, 0);
    std::vector<int> h_block_table(batch * max_blocks_per_seq, 0);

    int next_block = 0;
    for (int b = 0; b < batch; ++b) {
        h_cu_q[b + 1] = h_cu_q[b] + c.q_lens[b];
        h_cu_k[b + 1] = h_cu_k[b] + c.kv_lens[b];
        const int blocks = ceil_div(c.kv_lens[b], kPageBlockSize);
        for (int blk = 0; blk < blocks; ++blk) {
            h_block_table[b * max_blocks_per_seq + blk] = next_block++;
        }
    }

    const int num_blocks = next_block;
    const size_t q_elems = static_cast<size_t>(total_q) * kNumHeads * kHeadDim;
    const size_t kv_elems =
        static_cast<size_t>(num_blocks) * kPageBlockSize * kNumKvHeads * kHeadDim;
    const size_t lse_elems = static_cast<size_t>(kNumHeads) * total_q;

    std::vector<half> h_q(q_elems);
    std::vector<half> h_k(kv_elems);
    std::vector<half> h_v(kv_elems);
    fill_half(h_q, 1.0f / 128.0f);
    fill_half(h_k, 1.0f / 96.0f);
    fill_half(h_v, 1.0f / 64.0f);

    half *d_q = cuda_alloc<half>(q_elems, "cudaMalloc q");
    half *d_k = cuda_alloc<half>(kv_elems, "cudaMalloc k");
    half *d_v = cuda_alloc<half>(kv_elems, "cudaMalloc v");
    half *d_out = cuda_alloc<half>(q_elems, "cudaMalloc out");
    float *d_lse = cuda_alloc<float>(lse_elems, "cudaMalloc lse");
    int *d_cu_q = cuda_alloc<int>(h_cu_q.size(), "cudaMalloc cu_q");
    int *d_cu_k = cuda_alloc<int>(h_cu_k.size(), "cudaMalloc cu_k");
    int *d_seq_k = cuda_alloc<int>(c.kv_lens.size(), "cudaMalloc seq_k");
    int *d_block_table = cuda_alloc<int>(h_block_table.size(), "cudaMalloc block_table");

    cuda_check(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(half), cudaMemcpyHostToDevice),
               "copy q");
    cuda_check(cudaMemcpy(d_k, h_k.data(), kv_elems * sizeof(half), cudaMemcpyHostToDevice),
               "copy k");
    cuda_check(cudaMemcpy(d_v, h_v.data(), kv_elems * sizeof(half), cudaMemcpyHostToDevice),
               "copy v");
    cuda_check(cudaMemcpy(d_cu_q, h_cu_q.data(), h_cu_q.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "copy cu_q");
    cuda_check(cudaMemcpy(d_cu_k, h_cu_k.data(), h_cu_k.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "copy cu_k");
    cuda_check(cudaMemcpy(d_seq_k, c.kv_lens.data(), c.kv_lens.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "copy seq_k");
    cuda_check(cudaMemcpy(d_block_table, h_block_table.data(),
                          h_block_table.size() * sizeof(int), cudaMemcpyHostToDevice),
               "copy block_table");
    cuda_check(cudaMemset(d_out, 0, q_elems * sizeof(half)), "zero out");
    cuda_check(cudaMemset(d_lse, 0, lse_elems * sizeof(float)), "zero lse");

    flash::Flash_fwd_params params{};
    params.q_ptr = d_q;
    params.k_ptr = d_k;
    params.v_ptr = d_v;
    params.o_ptr = d_out;
    params.softmax_lse_ptr = d_lse;

    params.q_row_stride = kNumHeads * kHeadDim;
    params.k_row_stride = kNumKvHeads * kHeadDim;
    params.v_row_stride = kNumKvHeads * kHeadDim;
    params.o_row_stride = kNumHeads * kHeadDim;

    params.q_head_stride = kHeadDim;
    params.k_head_stride = kHeadDim;
    params.v_head_stride = kHeadDim;
    params.o_head_stride = kHeadDim;

    params.k_batch_stride = static_cast<int64_t>(kPageBlockSize) * kNumKvHeads * kHeadDim;
    params.v_batch_stride = static_cast<int64_t>(kPageBlockSize) * kNumKvHeads * kHeadDim;

    params.h = kNumHeads;
    params.h_k = kNumKvHeads;
    params.h_h_k_ratio = kNumHeads / kNumKvHeads;
    params.b = batch;
    params.seqlen_q = max_q;
    params.seqlen_k = max_k;
    params.seqlen_q_rounded = round_multiple(max_q, 128);
    params.seqlen_k_rounded = round_multiple(max_k, 128);
    params.d = kHeadDim;
    params.d_rounded = kHeadDim;
    params.total_q = total_q;

    params.scale_softmax = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    params.scale_softmax_log2 = params.scale_softmax * static_cast<float>(M_LOG2E);
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.window_size_left = -1;
    params.window_size_right = 0;
    params.softcap = 0.0f;

    params.cu_seqlens_q = d_cu_q;
    params.cu_seqlens_k = d_cu_k;
    params.seqused_k = d_seq_k;
    params.block_table = d_block_table;
    params.block_table_batch_stride = max_blocks_per_seq;
    params.page_block_size = kPageBlockSize;

    params.is_bf16 = false;
    params.is_causal = true;
    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = true;
    params.seqlenq_ngroups_swapped = false;
    params.num_splits = 0;

    cudaStream_t stream = nullptr;
    for (int i = 0; i < warmup; ++i) {
        flash::run_mha_fwd(params, stream, true);
    }
    cuda_check(cudaDeviceSynchronize(), "warmup synchronize");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "event create start");
    cuda_check(cudaEventCreate(&stop), "event create stop");
    cuda_check(cudaEventRecord(start, stream), "event record start");
    for (int i = 0; i < iters; ++i) {
        flash::run_mha_fwd(params, stream, true);
    }
    cuda_check(cudaEventRecord(stop, stream), "event record stop");
    cuda_check(cudaEventSynchronize(stop), "event sync");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "event elapsed");

    const size_t sample = std::min<size_t>(q_elems, 4096);
    std::vector<half> h_out(sample);
    cuda_check(cudaMemcpy(h_out.data(), d_out, sample * sizeof(half), cudaMemcpyDeviceToHost),
               "copy out sample");
    double checksum = 0.0;
    for (half value : h_out) {
        const float f = __half2float(value);
        if (!std::isfinite(f)) {
            throw std::runtime_error("non-finite output in " + c.name);
        }
        checksum += f;
    }

    std::cout << c.name << " total_q=" << total_q << " max_q=" << max_q
              << " max_k=" << max_k << " blocks=" << num_blocks
              << " avg_us=" << (elapsed_ms * 1000.0f / static_cast<float>(iters))
              << " checksum4096=" << checksum << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_lse);
    cudaFree(d_cu_q);
    cudaFree(d_cu_k);
    cudaFree(d_seq_k);
    cudaFree(d_block_table);
}

}  // namespace

int main() {
    try {
        int device = 0;
        cuda_check(cudaSetDevice(device), "set device");
        cudaDeviceProp prop{};
        cuda_check(cudaGetDeviceProperties(&prop, device), "get device props");
        const int warmup = get_env_int("FA2_WARMUP", 20);
        const int iters = get_env_int("FA2_ITERS", 200);
        std::cout << "device=" << prop.name << " sm=" << prop.major << prop.minor
                  << " warmup=" << warmup << " iters=" << iters << "\n";

        run_case({"prefill_4x256", {256, 256, 256, 256}, {256, 256, 256, 256}},
                 warmup, iters);
        run_case({"mixed_3x256_4x1",
                  {256, 256, 256, 1, 1, 1, 1},
                  {256, 256, 256, 257, 257, 257, 257}},
                 warmup, iters);
        run_case({"prefill_4x512", {512, 512, 512, 512}, {512, 512, 512, 512}},
                 warmup, iters);

        std::cout << "VERDICT direct_fa2_ffi_ok\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR " << e.what() << "\n";
        return 1;
    }
}
