// Native-CUDA microbench: does per-replay cuGraphUpload help or hurt?
//
// Captures a simple graph with N kernel launches that mimic our decode
// pattern (small kernels, sequential, with stream-host memcpys), then
// replays many times under three configurations:
//   1. baseline: launch + sync (no upload per replay)
//   2. with upload: upload + launch + sync per replay
//   3. multi-slot: alternate two graphs of different shapes
//
// Compile:
//   nvcc -O3 -arch=sm_89 graph_upload_bench.cu -o graph_upload_bench -lcuda
// Run:
//   ./graph_upload_bench

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e)); exit(1); } } while (0)

#define CHECK_DRV(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char* msg; cuGetErrorString(r, &msg); \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, msg); exit(1); } } while (0)

// Tiny kernel: y[i] = x[i] * 2.0 + b. Mimics a simple per-token op.
__global__ void scale_add(float* y, const float* x, int n, float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * 2.0f + b;
}

// Capture a graph that runs `num_kernels` scale_add launches sequentially,
// chained through `bufs[]`. Returns instantiated CUgraphExec.
CUgraphExec capture_graph(cudaStream_t stream, float** bufs, int n,
                          int num_kernels) {
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    int block = 256;
    int grid = (n + block - 1) / block;
    for (int k = 0; k < num_kernels; ++k) {
        scale_add<<<grid, block, 0, stream>>>(bufs[k+1], bufs[k], n, 1.0f);
    }
    cudaGraph_t g;
    CHECK(cudaStreamEndCapture(stream, &g));
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    CHECK(cudaGraphDestroy(g));
    // Initial upload (matching ferrum's end_capture flow).
    CHECK(cudaGraphUpload(exec, stream));
    return exec;
}

double bench_replays(cudaStream_t stream, cudaGraphExec_t exec,
                     int iters, bool upload_each) {
    // Warmup
    for (int i = 0; i < 50; ++i) {
        if (upload_each) CHECK(cudaGraphUpload(exec, stream));
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaStreamSynchronize(stream));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (upload_each) CHECK(cudaGraphUpload(exec, stream));
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaStreamSynchronize(stream));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

double bench_multi_slot(cudaStream_t stream, cudaGraphExec_t a, cudaGraphExec_t b,
                        int iters, bool upload_each) {
    // Warmup with both
    for (int i = 0; i < 25; ++i) {
        cudaGraphExec_t exec_pick = (i & 1) ? a : b;
        if (upload_each) CHECK(cudaGraphUpload(exec_pick, stream));
        CHECK(cudaGraphLaunch(exec_pick, stream));
        CHECK(cudaStreamSynchronize(stream));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cudaGraphExec_t exec_pick = (i & 1) ? a : b;
        if (upload_each) CHECK(cudaGraphUpload(exec_pick, stream));
        CHECK(cudaGraphLaunch(exec_pick, stream));
        CHECK(cudaStreamSynchronize(stream));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main() {
    // Mirror ferrum's decode hot path roughly:
    //   - Per layer: ~10 kernels (rms_norm, qkv_proj fused, qk_norm_rope,
    //     kv_append, flash_attn, o_proj, fused_add_rms, gate_up_proj,
    //     silu_mul, down_proj, residual_add)
    //   - 32 layers + final norm + lm_head ≈ ~320 kernel launches per token
    // Use 320 for the "decode-shaped" graph.
    const int n = 4096;
    const int num_kernels = 320;
    const int iters = 500;

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Allocate num_kernels+1 buffers, chain through them.
    std::vector<float*> bufs(num_kernels + 1, nullptr);
    for (int i = 0; i < (int)bufs.size(); ++i) {
        CHECK(cudaMalloc(&bufs[i], n * sizeof(float)));
        CHECK(cudaMemsetAsync(bufs[i], 0, n * sizeof(float), stream));
    }
    CHECK(cudaStreamSynchronize(stream));

    cudaGraphExec_t exec_a = capture_graph(stream, bufs.data(), n, num_kernels);

    printf("=== Single-slot graph, %d launches/graph, %d iters ===\n", num_kernels, iters);
    double ms_no_upload = bench_replays(stream, exec_a, iters, /*upload_each=*/false);
    double ms_with_upload = bench_replays(stream, exec_a, iters, /*upload_each=*/true);
    printf("no upload per replay  : %.3f ms/iter\n", ms_no_upload);
    printf("upload per replay     : %.3f ms/iter (delta %+.3f ms)\n",
           ms_with_upload, ms_with_upload - ms_no_upload);

    // Multi-slot: capture a SECOND graph of a different shape (n=2048).
    const int n2 = 2048;
    std::vector<float*> bufs2(num_kernels + 1, nullptr);
    for (int i = 0; i < (int)bufs2.size(); ++i) {
        CHECK(cudaMalloc(&bufs2[i], n2 * sizeof(float)));
        CHECK(cudaMemsetAsync(bufs2[i], 0, n2 * sizeof(float), stream));
    }
    CHECK(cudaStreamSynchronize(stream));
    cudaGraphExec_t exec_b = capture_graph(stream, bufs2.data(), n2, num_kernels);

    printf("\n=== Multi-slot (alternating two graphs), %d iters ===\n", iters);
    double ms_alt_no_upload = bench_multi_slot(stream, exec_a, exec_b, iters, false);
    double ms_alt_with_upload = bench_multi_slot(stream, exec_a, exec_b, iters, true);
    printf("alternating no upload : %.3f ms/iter\n", ms_alt_no_upload);
    printf("alternating w/ upload : %.3f ms/iter (delta %+.3f ms)\n",
           ms_alt_with_upload, ms_alt_with_upload - ms_alt_no_upload);

    // Cleanup
    CHECK(cudaGraphExecDestroy(exec_a));
    CHECK(cudaGraphExecDestroy(exec_b));
    for (auto p : bufs) CHECK(cudaFree(p));
    for (auto p : bufs2) CHECK(cudaFree(p));
    CHECK(cudaStreamDestroy(stream));
    return 0;
}
