// sync_barrier_bench.cu — direct measurement of the
// "kernel → DtoH → cuStreamSynchronize" barrier pattern that
// ferrum's host-route MoE path uses per layer per iter.
//
// Hypothesis: each barrier forces the GPU pipeline to drain. If the
// stream has ~600 µs of queued work, each sync wastes ~600 µs of
// host-side wait time. Compounded over 48 layers/iter → ~28 ms/iter
// loss = the bulk of ferrum's c=32 TPOT gap vs vLLM.
//
// Compare:
//   A. async pipeline:  N back-to-back kernels with NO sync — fully pipelined
//   B. barrier pattern: each kernel followed by tiny DtoH + cuStreamSync
//   C. graph capture:   N kernels captured + replayed (no host barriers)
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 sync_barrier_bench.cu -o sync_barrier_bench

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>
#include <cstdlib>

#define CK(x) do { auto e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %d at %d: %s\n", e, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while(0)

// Simulates a single MoE layer's compute (~600 µs at RTX 4090 INT4 m=1 for 128 experts).
// We use a busy-loop kernel for repeatable shape.
__global__ void layer_compute_kernel(int* x, int loop) {
    int sum = threadIdx.x;
    for (int i = 0; i < loop; i++) {
        sum = (sum * 17 + i) ^ blockIdx.x;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        x[0] = sum;
    }
}

const int N_LAYERS = 48;  // M3 has 48 layers
const int N_ITERS = 100;  // simulate 100 decode tokens
const int WARMUP = 5;
const int LAYER_WORK = 4000;  // tune to get ~30-50 µs per kernel

void bench_async(cudaStream_t s, int* d_x) {
    // Mode A: pure async pipeline, no sync per layer
    for (int w = 0; w < WARMUP; w++) {
        for (int l = 0; l < N_LAYERS; l++) {
            layer_compute_kernel<<<1, 32, 0, s>>>(d_x, LAYER_WORK);
        }
    }
    cudaStreamSynchronize(s);

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < N_ITERS; it++) {
        for (int l = 0; l < N_LAYERS; l++) {
            layer_compute_kernel<<<1, 32, 0, s>>>(d_x, LAYER_WORK);
        }
    }
    cudaStreamSynchronize(s);  // single sync at end-of-iter only
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)N_ITERS;
    printf("  async pipeline      : %7.1f µs/iter  (%5.2f µs/layer effective)\n",
           us, us / N_LAYERS);
}

void bench_barrier(cudaStream_t s, int* d_x) {
    // Mode B: per-layer DtoH + cuStreamSynchronize (mimics ferrum host route)
    int host_buf;

    for (int w = 0; w < WARMUP; w++) {
        for (int l = 0; l < N_LAYERS; l++) {
            layer_compute_kernel<<<1, 32, 0, s>>>(d_x, LAYER_WORK);
            cudaMemcpyAsync(&host_buf, d_x, sizeof(int), cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);  // barrier
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < N_ITERS; it++) {
        for (int l = 0; l < N_LAYERS; l++) {
            layer_compute_kernel<<<1, 32, 0, s>>>(d_x, LAYER_WORK);
            cudaMemcpyAsync(&host_buf, d_x, sizeof(int), cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);  // 48 barriers per iter
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)N_ITERS;
    printf("  per-layer barrier   : %7.1f µs/iter  (%5.2f µs/layer including barrier)\n",
           us, us / N_LAYERS);
}

void bench_graph(cudaStream_t s, int* d_x) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int l = 0; l < N_LAYERS; l++) {
        layer_compute_kernel<<<1, 32, 0, s>>>(d_x, LAYER_WORK);
    }
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));

    for (int w = 0; w < WARMUP; w++) CK(cudaGraphLaunch(ge, s));
    cudaStreamSynchronize(s);

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < N_ITERS; it++) {
        CK(cudaGraphLaunch(ge, s));
    }
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)N_ITERS;
    printf("  graph replay        : %7.1f µs/iter  (%5.2f µs/layer effective)\n",
           us, us / N_LAYERS);

    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Pattern: %d layers/iter, %d iters bench, LAYER_WORK=%d (~30-50 µs each)\n\n",
           N_LAYERS, N_ITERS, LAYER_WORK);

    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    int* d_x;
    CK(cudaMalloc(&d_x, sizeof(int)));

    printf("=== M3-like 48-layer decode iter (kernel ~30 µs each) ===\n");
    bench_async(s, d_x);
    bench_barrier(s, d_x);
    bench_graph(s, d_x);
    printf("\n");

    // Now repeat with HEAVIER kernels (~150 µs each) — closer to small MoE expert matmul shape
    printf("=== Heavier kernels (each ~150 µs) ===\n");
    {
        int saved = N_ITERS;
        // Build new kernels with bigger LOOP — inline into a one-off bench
        cudaStreamSynchronize(s);
        for (int w = 0; w < WARMUP; w++) {
            for (int l = 0; l < N_LAYERS; l++) layer_compute_kernel<<<1, 32, 0, s>>>(d_x, 20000);
        }
        cudaStreamSynchronize(s);

        // async
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < N_ITERS; it++) {
            for (int l = 0; l < N_LAYERS; l++) layer_compute_kernel<<<1, 32, 0, s>>>(d_x, 20000);
        }
        cudaStreamSynchronize(s);
        auto t1 = std::chrono::steady_clock::now();
        double async_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)N_ITERS;
        printf("  async pipeline      : %7.1f µs/iter\n", async_us);

        // barrier
        int host_buf;
        t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < N_ITERS; it++) {
            for (int l = 0; l < N_LAYERS; l++) {
                layer_compute_kernel<<<1, 32, 0, s>>>(d_x, 20000);
                cudaMemcpyAsync(&host_buf, d_x, sizeof(int), cudaMemcpyDeviceToHost, s);
                cudaStreamSynchronize(s);
            }
        }
        t1 = std::chrono::steady_clock::now();
        double bar_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)N_ITERS;
        printf("  per-layer barrier   : %7.1f µs/iter   (overhead = %+.1f µs/iter)\n",
               bar_us, bar_us - async_us);

        // graph
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int l = 0; l < N_LAYERS; l++) layer_compute_kernel<<<1, 32, 0, s>>>(d_x, 20000);
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int w = 0; w < WARMUP; w++) CK(cudaGraphLaunch(ge, s));
        cudaStreamSynchronize(s);
        t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < N_ITERS; it++) CK(cudaGraphLaunch(ge, s));
        cudaStreamSynchronize(s);
        t1 = std::chrono::steady_clock::now();
        double gr_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)N_ITERS;
        printf("  graph replay        : %7.1f µs/iter   (savings vs barrier = %+.1f µs/iter, %.1f×)\n",
               gr_us, gr_us - bar_us, bar_us/gr_us);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }

    printf("\n=== Translate to ferrum c=32 (TPOT_gap = 19800 µs) ===\n");
    printf("If ferrum's 48-layer iter today is barrier-pattern, then\n");
    printf("the iter time is dominated by the barrier overhead measured above.\n");
    printf("Switching to async (or graph) recovers that overhead directly.\n");

    cudaFree(d_x);
    cudaStreamDestroy(s);
    return 0;
}
