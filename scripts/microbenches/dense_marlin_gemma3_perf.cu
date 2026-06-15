// Standalone native-CUDA probe for Ferrum's dense IST-DASLab Marlin GEMM at
// Gemma3-27B GPTQ shapes.
//
// Goal: validate dense Marlin kernel/tile hypotheses without loading a model
// or rebuilding the Rust workspace. This directly calls `marlin_cuda` from
// crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu with synthetic buffers.
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 \
//     scripts/microbenches/dense_marlin_gemma3_perf.cu \
//     crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu \
//     -lcuda -lcudart -o /tmp/dense_marlin_gemma3_perf

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int marlin_cuda(
    const void* A, const void* B, void* C, void* s, int prob_m, int prob_n,
    int prob_k, void* workspace, int groupsize, int dev, cudaStream_t stream,
    int thread_k, int thread_n, int sms, int max_par, int prob_n_full);

#define CHK(stmt)                                                                \
    do {                                                                         \
        cudaError_t err__ = (stmt);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(err__));                            \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

struct Shape {
    const char* name;
    int k;
    int n;
};

struct Tile {
    const char* name;
    int thread_k;
    int thread_n;
};

constexpr int GROUP_SIZE = 128;
constexpr int MAX_PAR = 16;
constexpr int WARMUP_ITERS = 5;
constexpr int TIMED_ITERS = 80;
constexpr double RTX4090_FP16_TFLOPS_PEAK = 165.0;

static size_t round_up(size_t value, size_t align) {
    return ((value + align - 1) / align) * align;
}

static int padded_m(int m) {
    return ((m + 15) / 16) * 16;
}

static void* cuda_alloc_bytes(size_t bytes, int pattern) {
    void* ptr = nullptr;
    CHK(cudaMalloc(&ptr, bytes));
    CHK(cudaMemset(ptr, pattern, bytes));
    return ptr;
}

static float time_one(
    cudaStream_t stream,
    void* a,
    void* b,
    void* c,
    void* scales,
    void* workspace,
    size_t workspace_bytes,
    const Shape& shape,
    int m,
    const Tile& tile,
    bool include_workspace_zero,
    int sms) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    const int total_iters = WARMUP_ITERS + TIMED_ITERS;
    for (int iter = 0; iter < total_iters; ++iter) {
        if (include_workspace_zero) {
            CHK(cudaEventRecord(start, stream));
            CHK(cudaMemsetAsync(workspace, 0, workspace_bytes, stream));
        } else {
            CHK(cudaMemsetAsync(workspace, 0, workspace_bytes, stream));
            CHK(cudaEventRecord(start, stream));
        }
        int ret = marlin_cuda(
            a,
            b,
            c,
            scales,
            m,
            shape.n,
            shape.k,
            workspace,
            GROUP_SIZE,
            0,
            stream,
            tile.thread_k,
            tile.thread_n,
            sms,
            MAX_PAR,
            -1);
        if (ret != 0) {
            CHK(cudaEventDestroy(start));
            CHK(cudaEventDestroy(stop));
            return -static_cast<float>(ret);
        }
        CHK(cudaEventRecord(stop, stream));
        CHK(cudaEventSynchronize(stop));
        CHK(cudaGetLastError());
        if (iter >= WARMUP_ITERS) {
            float ms = 0.0f;
            CHK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }
    }

    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
    return (total_ms * 1000.0f) / static_cast<float>(TIMED_ITERS);
}

static void run_shape(const Shape& shape, const std::vector<int>& m_values) {
    int sms = 0;
    CHK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    cudaStream_t stream;
    CHK(cudaStreamCreate(&stream));

    int max_m = 0;
    for (int m : m_values) max_m = max_m > m ? max_m : m;
    size_t a_bytes = round_up(static_cast<size_t>(max_m) * shape.k * 2, 16);
    size_t b_bytes = round_up(static_cast<size_t>(shape.k) * shape.n / 2, 16);
    size_t c_bytes = round_up(static_cast<size_t>(max_m) * shape.n * 2, 16);
    size_t scale_bytes =
        round_up(static_cast<size_t>(shape.k / GROUP_SIZE) * shape.n * 2, 16);
    size_t workspace_bytes =
        round_up(static_cast<size_t>(shape.n / 128) * MAX_PAR * sizeof(int), 16);

    void* a = cuda_alloc_bytes(a_bytes, 0x3c);
    void* b = cuda_alloc_bytes(b_bytes, 0x11);
    void* c = cuda_alloc_bytes(c_bytes, 0x00);
    void* scales = cuda_alloc_bytes(scale_bytes, 0x3c);
    void* workspace = cuda_alloc_bytes(workspace_bytes, 0x00);

    const Tile tiles[] = {
        {"auto", -1, -1},
        {"128x128", 128, 128},
        {"64x256", 64, 256},
    };

    std::printf("\nshape=%s k=%d n=%d qweight_mb=%.1f scale_mb=%.1f\n",
                shape.name,
                shape.k,
                shape.n,
                static_cast<double>(b_bytes) / (1024.0 * 1024.0),
                static_cast<double>(scale_bytes) / (1024.0 * 1024.0));
    std::printf("mode,m,tile,us,useful_tflops,padded_tflops,ret\n");
    for (int m : m_values) {
        for (const Tile& tile : tiles) {
            for (int mode_idx = 0; mode_idx < 2; ++mode_idx) {
                bool include_ws = mode_idx != 0;
                float us = time_one(
                    stream,
                    a,
                    b,
                    c,
                    scales,
                    workspace,
                    workspace_bytes,
                    shape,
                    m,
                    tile,
                    include_ws,
                    sms);
                const char* mode = include_ws ? "ws_plus_kernel" : "kernel_only";
                if (us < 0.0f) {
                    std::printf("%s,%d,%s,NA,NA,NA,%d\n", mode, m, tile.name,
                                static_cast<int>(-us));
                    continue;
                }
                double useful_ops = 2.0 * static_cast<double>(m) * shape.k * shape.n;
                double padded_ops =
                    2.0 * static_cast<double>(padded_m(m)) * shape.k * shape.n;
                double useful_tflops = useful_ops / (static_cast<double>(us) * 1.0e6);
                double padded_tflops = padded_ops / (static_cast<double>(us) * 1.0e6);
                std::printf("%s,%d,%s,%.3f,%.2f,%.2f,0\n", mode, m, tile.name, us,
                            useful_tflops, padded_tflops);
            }
        }
    }

    CHK(cudaFree(a));
    CHK(cudaFree(b));
    CHK(cudaFree(c));
    CHK(cudaFree(scales));
    CHK(cudaFree(workspace));
    CHK(cudaStreamDestroy(stream));
}

int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    std::printf("dense_marlin_gemma3_perf device=%s sm=%d.%d peak_ref_tflops=%.1f\n",
                prop.name,
                prop.major,
                prop.minor,
                RTX4090_FP16_TFLOPS_PEAK);

    std::vector<int> m_values = {1, 3, 6, 9, 12, 16, 23, 32};
    const Shape shapes[] = {
        {"qkv", 5376, 8192},
        {"o_proj", 4096, 5376},
        {"gate_up", 5376, 43008},
        {"down", 21504, 5376},
    };
    for (const Shape& shape : shapes) {
        run_shape(shape, m_values);
    }

    std::printf("\nVERDICT: dense Marlin native CUDA probe complete\n");
    return 0;
}
