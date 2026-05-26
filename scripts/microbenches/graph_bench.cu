// graph_bench.cu — directly measure CUDA Graph speedup for the M3
// decode launch pattern.
//
// Simulates ferrum's per-iter decode shape: 48 layers × ~10 small
// kernels/layer = 480 kernel launches. Each kernel is a noop (just
// reads/writes 1 byte) so the measurement IS purely launch overhead
// + minimal GPU pipeline depth.
//
// Compare:
//   A. naive_loop:   loop N=480 launches per iter, REPEAT 200 iters
//   B. graph_replay: capture 480 launches into a graph ONCE, replay 200×
//
// If naive_loop is much slower than graph_replay, the gap = launch
// overhead × N_kernels × N_iters. That's exactly what ferrum pays
// today vs vLLM.
//
// Build: nvcc -O3 -arch=sm_89 -std=c++17 graph_bench.cu -o graph_bench
// Run:   ./graph_bench

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>
#include <cstdlib>

#define CK(x) do { auto e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA err %d at %d: %s\n", e, __LINE__, cudaGetErrorString(e)); \
  std::exit(1); } } while(0)

// Trivial kernel that does minimal GPU work — represents the
// "launch dominates" regime that small per-token decode kernels live in.
__global__ void noop_kernel(int* x) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *x += 1;
    }
}

// A slightly more realistic kernel — does ~10 µs of work to simulate
// kernels like rms_norm / silu / small Marlin tiles. Gives us a 2nd
// measurement at higher utilization.
__global__ void small_work_kernel(int* x, int iters) {
    int sum = 0;
    for (int i = 0; i < iters; i++) {
        sum += (i * 17) ^ threadIdx.x;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *x = sum;
    }
}

const int N_KERNELS_PER_ITER = 480;  // ferrum decode iter ≈ 48 layer × 10 kern
const int N_ITERS = 200;             // simulate 200 decode tokens
const int WARMUP = 10;

double bench_naive(cudaStream_t s, int* d_x, bool small_work) {
    // warmup
    for (int w = 0; w < WARMUP; w++) {
        for (int k = 0; k < N_KERNELS_PER_ITER; k++) {
            if (small_work)
                small_work_kernel<<<1, 32, 0, s>>>(d_x, 100);
            else
                noop_kernel<<<1, 32, 0, s>>>(d_x);
        }
    }
    cudaStreamSynchronize(s);

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < N_ITERS; it++) {
        for (int k = 0; k < N_KERNELS_PER_ITER; k++) {
            if (small_work)
                small_work_kernel<<<1, 32, 0, s>>>(d_x, 100);
            else
                noop_kernel<<<1, 32, 0, s>>>(d_x);
        }
    }
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    return us / N_ITERS;  // µs per iter
}

double bench_graph(cudaStream_t s, int* d_x, bool small_work) {
    // 1. capture
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < N_KERNELS_PER_ITER; k++) {
        if (small_work)
            small_work_kernel<<<1, 32, 0, s>>>(d_x, 100);
        else
            noop_kernel<<<1, 32, 0, s>>>(d_x);
    }
    CK(cudaStreamEndCapture(s, &graph));
    CK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    // warmup replays
    for (int w = 0; w < WARMUP; w++) {
        CK(cudaGraphLaunch(graph_exec, s));
    }
    cudaStreamSynchronize(s);

    // bench
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < N_ITERS; it++) {
        CK(cudaGraphLaunch(graph_exec, s));
    }
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    return us / N_ITERS;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Pattern: %d kernel launches per iter, %d iters bench\n\n",
           N_KERNELS_PER_ITER, N_ITERS);

    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    int* d_x;
    CK(cudaMalloc(&d_x, sizeof(int)));

    // === Test 1: noop kernels (pure launch overhead) ===
    printf("=== Test 1: NOOP kernels (pure launch overhead) ===\n");
    double naive_us = bench_naive(s, d_x, false);
    double graph_us = bench_graph(s, d_x, false);
    printf("  naive loop  : %8.1f µs/iter  (%.2f µs per launch)\n",
           naive_us, naive_us / N_KERNELS_PER_ITER);
    printf("  graph replay: %8.1f µs/iter  (%.2f µs per launch equiv)\n",
           graph_us, graph_us / N_KERNELS_PER_ITER);
    printf("  graph WIN   : %8.1f µs/iter  (%.1f×)\n",
           naive_us - graph_us, naive_us / graph_us);

    printf("\n");

    // === Test 2: small-work kernels (10 µs GPU work each) ===
    printf("=== Test 2: SMALL-WORK kernels (~10 µs GPU each) ===\n");
    double naive_us2 = bench_naive(s, d_x, true);
    double graph_us2 = bench_graph(s, d_x, true);
    printf("  naive loop  : %8.1f µs/iter  (%.2f µs per launch+kern)\n",
           naive_us2, naive_us2 / N_KERNELS_PER_ITER);
    printf("  graph replay: %8.1f µs/iter  (%.2f µs per launch+kern)\n",
           graph_us2, graph_us2 / N_KERNELS_PER_ITER);
    printf("  graph WIN   : %8.1f µs/iter  (%.1f×)\n",
           naive_us2 - graph_us2, naive_us2 / graph_us2);

    printf("\n");

    // === Test 3: scaling — what if N_KERNELS is bigger? ===
    printf("=== Test 3: how does win scale with launch count? ===\n");
    printf("(noop, varying N_KERNELS, same N_ITERS=%d)\n", N_ITERS);
    printf("  N_kernels |  naive µs/iter |  graph µs/iter |  win\n");
    for (int nk : {100, 240, 480, 960, 1920}) {
        // re-bench with this count
        for (int w = 0; w < WARMUP; w++) {
            for (int k = 0; k < nk; k++) noop_kernel<<<1, 32, 0, s>>>(d_x);
        }
        cudaStreamSynchronize(s);

        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < N_ITERS; it++) {
            for (int k = 0; k < nk; k++) noop_kernel<<<1, 32, 0, s>>>(d_x);
        }
        cudaStreamSynchronize(s);
        auto t1 = std::chrono::steady_clock::now();
        double n_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)N_ITERS;

        cudaGraph_t graph2; cudaGraphExec_t graph_exec2;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int k = 0; k < nk; k++) noop_kernel<<<1, 32, 0, s>>>(d_x);
        CK(cudaStreamEndCapture(s, &graph2));
        CK(cudaGraphInstantiate(&graph_exec2, graph2, nullptr, nullptr, 0));
        for (int w = 0; w < WARMUP; w++) CK(cudaGraphLaunch(graph_exec2, s));
        cudaStreamSynchronize(s);

        auto t2 = std::chrono::steady_clock::now();
        for (int it = 0; it < N_ITERS; it++) {
            CK(cudaGraphLaunch(graph_exec2, s));
        }
        cudaStreamSynchronize(s);
        auto t3 = std::chrono::steady_clock::now();
        double g_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / (double)N_ITERS;

        printf("  %9d | %14.1f | %14.1f | %.1f×\n", nk, n_us, g_us, n_us/g_us);
        cudaGraphExecDestroy(graph_exec2);
        cudaGraphDestroy(graph2);
    }

    cudaFree(d_x);
    cudaStreamDestroy(s);
    return 0;
}
