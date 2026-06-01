#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" int ferrum_fa2_paged_varlen_fwd(
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
    size_t err_buf_len);

static void ck(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void fill_half_kernel(__half *p, size_t n, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int x = static_cast<int>((i * 1103515245ull + 12345ull) & 1023ull);
        p[i] = __float2half(((x % 257) - 128) * scale);
    }
}

__global__ void finite_check_kernel(const __half *p, size_t n, int *bad) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(p[i]);
        if (!isfinite(v)) {
            atomicAdd(bad, 1);
        }
    }
}

int main(int argc, char **argv) {
    int num_seqs = argc > 1 ? std::atoi(argv[1]) : 32;
    int q_len = argc > 2 ? std::atoi(argv[2]) : 1;
    int kv_len = argc > 3 ? std::atoi(argv[3]) : 257;
    int iters = argc > 4 ? std::atoi(argv[4]) : 200;
    const int num_heads = 32;
    const int num_kv_heads = 4;
    const int head_dim = 128;
    const int block_size = 16;
    const int max_blocks = (kv_len + block_size - 1) / block_size;
    const int total_q = num_seqs * q_len;
    const int num_blocks = num_seqs * max_blocks;

    std::vector<int> h_cu(num_seqs + 1), h_lens(num_seqs), h_table(num_seqs * max_blocks);
    for (int s = 0; s <= num_seqs; ++s) h_cu[s] = s * q_len;
    for (int s = 0; s < num_seqs; ++s) {
        h_lens[s] = kv_len;
        for (int b = 0; b < max_blocks; ++b) h_table[s * max_blocks + b] = s * max_blocks + b;
    }

    __half *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr;
    float *lse = nullptr;
    int *cu = nullptr, *lens = nullptr, *table = nullptr, *bad = nullptr;
    size_t q_elems = static_cast<size_t>(total_q) * num_heads * head_dim;
    size_t kv_elems = static_cast<size_t>(num_blocks) * block_size * num_kv_heads * head_dim;
    size_t out_elems = q_elems;
    ck(cudaMalloc(&q, q_elems * sizeof(__half)), "malloc q");
    ck(cudaMalloc(&k, kv_elems * sizeof(__half)), "malloc k");
    ck(cudaMalloc(&v, kv_elems * sizeof(__half)), "malloc v");
    ck(cudaMalloc(&out, out_elems * sizeof(__half)), "malloc out");
    ck(cudaMalloc(&lse, static_cast<size_t>(num_heads) * total_q * sizeof(float)), "malloc lse");
    ck(cudaMalloc(&cu, h_cu.size() * sizeof(int)), "malloc cu");
    ck(cudaMalloc(&lens, h_lens.size() * sizeof(int)), "malloc lens");
    ck(cudaMalloc(&table, h_table.size() * sizeof(int)), "malloc table");
    ck(cudaMalloc(&bad, sizeof(int)), "malloc bad");
    ck(cudaMemcpy(cu, h_cu.data(), h_cu.size() * sizeof(int), cudaMemcpyHostToDevice), "copy cu");
    ck(cudaMemcpy(lens, h_lens.data(), h_lens.size() * sizeof(int), cudaMemcpyHostToDevice), "copy lens");
    ck(cudaMemcpy(table, h_table.data(), h_table.size() * sizeof(int), cudaMemcpyHostToDevice), "copy table");
    fill_half_kernel<<<(q_elems + 255) / 256, 256>>>(q, q_elems, 0.001f);
    fill_half_kernel<<<(kv_elems + 255) / 256, 256>>>(k, kv_elems, 0.001f);
    fill_half_kernel<<<(kv_elems + 255) / 256, 256>>>(v, kv_elems, 0.001f);
    ck(cudaDeviceSynchronize(), "fill sync");

    char err[512];
    cudaStream_t stream;
    ck(cudaStreamCreate(&stream), "stream");
    for (int i = 0; i < 20; ++i) {
        int rc = ferrum_fa2_paged_varlen_fwd(q, k, v, out, lse, cu, lens, table, num_seqs,
                                             total_q, q_len, kv_len, num_heads, num_kv_heads,
                                             head_dim, block_size, max_blocks, stream, err, sizeof(err));
        if (rc != 0) { std::fprintf(stderr, "warmup rc=%d err=%s\n", rc, err); return 2; }
    }
    ck(cudaStreamSynchronize(stream), "warmup sync");
    cudaEvent_t start, stop;
    ck(cudaEventCreate(&start), "event start");
    ck(cudaEventCreate(&stop), "event stop");
    ck(cudaEventRecord(start, stream), "record start");
    for (int i = 0; i < iters; ++i) {
        int rc = ferrum_fa2_paged_varlen_fwd(q, k, v, out, lse, cu, lens, table, num_seqs,
                                             total_q, q_len, kv_len, num_heads, num_kv_heads,
                                             head_dim, block_size, max_blocks, stream, err, sizeof(err));
        if (rc != 0) { std::fprintf(stderr, "iter rc=%d err=%s\n", rc, err); return 3; }
    }
    ck(cudaEventRecord(stop, stream), "record stop");
    ck(cudaEventSynchronize(stop), "event sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    ck(cudaMemset(bad, 0, sizeof(int)), "bad zero");
    finite_check_kernel<<<(out_elems + 255) / 256, 256, 0, stream>>>(out, out_elems, bad);
    int h_bad = 0;
    ck(cudaMemcpyAsync(&h_bad, bad, sizeof(int), cudaMemcpyDeviceToHost, stream), "copy bad");
    ck(cudaStreamSynchronize(stream), "check sync");
    if (h_bad != 0) {
        std::fprintf(stderr, "non-finite output count=%d\n", h_bad);
        return 4;
    }
    std::printf("native_fa2_bench num_seqs=%d q_len=%d kv_len=%d total_q=%d heads=%d kv_heads=%d iters=%d avg_us=%.3f finite_ok=1\n",
                num_seqs, q_len, kv_len, total_q, num_heads, num_kv_heads, iters, (ms * 1000.0f) / iters);
    return 0;
}
