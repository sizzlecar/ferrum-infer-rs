// h2d_microbench.cu — measure cost of N small u32 H2Ds vs 1 batched.
// Compile: nvcc -O3 -arch=sm_89 -o h2d_microbench h2d_microbench.cu
// Run: ./h2d_microbench
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CHK(x) do { cudaError_t e=(x); if (e!=cudaSuccess) { fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)

int main(int argc, char** argv) {
    int n_iters  = (argc>1) ? atoi(argv[1]) : 1000;
    int n_layers = (argc>2) ? atoi(argv[2]) : 48;  // matches Qwen3-30B-A3B

    cudaStream_t stream;
    CHK(cudaStreamCreate(&stream));

    // Allocate device buffers: 48 separate 4-byte u32 buffers (PER-LAYER)
    // vs 1 shared 4-byte u32 buffer
    uint32_t* d_per_layer[48];
    for (int i = 0; i < n_layers; i++) CHK(cudaMalloc(&d_per_layer[i], 4));
    uint32_t* d_shared;
    CHK(cudaMalloc(&d_shared, 4));

    uint32_t host_val = 256;  // matches Qwen prefill kv_len

    // Warmup
    for (int i = 0; i < 5; i++) {
        for (int l = 0; l < n_layers; l++) {
            CHK(cudaMemcpyAsync(d_per_layer[l], &host_val, 4, cudaMemcpyHostToDevice, stream));
        }
    }
    CHK(cudaStreamSynchronize(stream));

    // Variant A: N iters × 48 per-layer H2Ds (current ferrum)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iters; i++) {
        for (int l = 0; l < n_layers; l++) {
            CHK(cudaMemcpyAsync(d_per_layer[l], &host_val, 4, cudaMemcpyHostToDevice, stream));
        }
    }
    CHK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_A = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double us_per_layer = us_A / (n_iters * n_layers);
    printf("[A] %d iter × %d layers (per-layer H2D): total=%.1f us, per-H2D=%.2f us\n",
           n_iters, n_layers, us_A, us_per_layer);

    // Variant B: N iters × 1 shared H2D (proposed fix)
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iters; i++) {
        CHK(cudaMemcpyAsync(d_shared, &host_val, 4, cudaMemcpyHostToDevice, stream));
    }
    CHK(cudaStreamSynchronize(stream));
    t1 = std::chrono::high_resolution_clock::now();
    double us_B = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double us_per_iter_B = us_B / n_iters;
    printf("[B] %d iter × 1 shared H2D     : total=%.1f us, per-iter=%.2f us\n",
           n_iters, us_B, us_per_iter_B);

    double saved_per_iter = (us_A - us_B) / n_iters;
    double saved_pct = 100.0 * (us_A - us_B) / us_A;
    printf("[Δ] per-iter saving: %.2f us (%.1f%% of A)\n", saved_per_iter, saved_pct);
    printf("[Δ] total saving over %d iters: %.1f us = %.2f ms\n",
           n_iters, us_A - us_B, (us_A - us_B) / 1000.0);

    cudaFree(d_shared);
    for (int i = 0; i < n_layers; i++) cudaFree(d_per_layer[i]);
    cudaStreamDestroy(stream);
    return 0;
}
