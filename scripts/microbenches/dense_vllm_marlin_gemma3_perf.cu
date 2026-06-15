// Standalone native-CUDA probe for vendored vLLM dense GPTQ-Marlin at
// Gemma3-27B GPTQ projection shapes.
//
// This is diagnostic only. It bypasses Rust/product loading and calls the
// extern "C" ferrum_marlin_mm_f16_u4b8 wrapper from vllm_marlin/marlin.cu.
// The companion build script patches a minimal selector into a temporary copy
// of vllm_marlin so we can test whether the vLLM dense kernel is a viable
// same-shape speed target for Ferrum's default IST-DASLab Marlin path.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void ferrum_marlin_mm_f16_u4b8(
    const void* A, const void* B, void* C, void* C_tmp, void* a_s, void* b_s,
    void* g_idx, void* perm, void* a_tmp, int prob_m, int prob_n, int prob_k,
    int lda, void* workspace, bool has_act_order, bool is_k_full,
    int num_groups, int group_size, int dev, cudaStream_t stream,
    int thread_k_init, int thread_n_init, int sms, bool use_atomic_add,
    bool use_fp32_reduce);

#define CHK(stmt)                                                               \
    do {                                                                        \
        cudaError_t err__ = (stmt);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err__));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

struct Shape {
    const char* name;
    int k;
    int n;
};

constexpr int GROUP_SIZE = 128;
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

static float time_vllm_marlin(
    cudaStream_t stream,
    void* a,
    void* b,
    void* c,
    void* c_tmp,
    void* scales,
    void* workspace,
    size_t workspace_bytes,
    const Shape& shape,
    int m,
    int sms) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    const int total_iters = WARMUP_ITERS + TIMED_ITERS;
    for (int iter = 0; iter < total_iters; ++iter) {
        CHK(cudaMemsetAsync(workspace, 0, workspace_bytes, stream));
        CHK(cudaEventRecord(start, stream));
        ferrum_marlin_mm_f16_u4b8(
            a,
            b,
            c,
            c_tmp,
            nullptr,
            scales,
            nullptr,
            nullptr,
            nullptr,
            m,
            shape.n,
            shape.k,
            shape.k,
            workspace,
            false,
            true,
            shape.k / GROUP_SIZE,
            GROUP_SIZE,
            0,
            stream,
            -1,
            -1,
            sms,
            false,
            false);
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
    size_t workspace_ints =
        std::max(static_cast<size_t>(sms * 4),
                 static_cast<size_t>(shape.n / 64) * 16);
    size_t workspace_bytes = round_up(workspace_ints * sizeof(int), 16);

    void* a = cuda_alloc_bytes(a_bytes, 0x3c);
    void* b = cuda_alloc_bytes(b_bytes, 0x11);
    void* c = cuda_alloc_bytes(c_bytes, 0x00);
    void* c_tmp = cuda_alloc_bytes(c_bytes, 0x00);
    void* scales = cuda_alloc_bytes(scale_bytes, 0x3c);
    void* workspace = cuda_alloc_bytes(workspace_bytes, 0x00);

    std::printf("\nshape=%s k=%d n=%d qweight_mb=%.1f scale_mb=%.1f\n",
                shape.name,
                shape.k,
                shape.n,
                static_cast<double>(b_bytes) / (1024.0 * 1024.0),
                static_cast<double>(scale_bytes) / (1024.0 * 1024.0));
    std::printf("mode,m,us,useful_tflops,padded_tflops,ret\n");
    for (int m : m_values) {
        float us = time_vllm_marlin(
            stream,
            a,
            b,
            c,
            c_tmp,
            scales,
            workspace,
            workspace_bytes,
            shape,
            m,
            sms);
        double useful_ops = 2.0 * static_cast<double>(m) * shape.k * shape.n;
        double padded_ops =
            2.0 * static_cast<double>(padded_m(m)) * shape.k * shape.n;
        double useful_tflops = useful_ops / (static_cast<double>(us) * 1.0e6);
        double padded_tflops = padded_ops / (static_cast<double>(us) * 1.0e6);
        std::printf("auto,%d,%.3f,%.2f,%.2f,0\n", m, us, useful_tflops,
                    padded_tflops);
    }

    CHK(cudaFree(a));
    CHK(cudaFree(b));
    CHK(cudaFree(c));
    CHK(cudaFree(c_tmp));
    CHK(cudaFree(scales));
    CHK(cudaFree(workspace));
    CHK(cudaStreamDestroy(stream));
}

int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    std::printf(
        "dense_vllm_marlin_gemma3_perf device=%s sm=%d.%d peak_ref_tflops=%.1f\n",
        prop.name,
        prop.major,
        prop.minor,
        RTX4090_FP16_TFLOPS_PEAK);

    std::vector<int> m_values = {16, 23, 32};
    const Shape shapes[] = {
        {"qkv", 5376, 8192},
        {"gate_up", 5376, 43008},
        {"down", 21504, 5376},
    };
    for (const Shape& shape : shapes) {
        run_shape(shape, m_values);
    }

    std::printf("\nVERDICT: dense vLLM Marlin native CUDA probe complete\n");
    return 0;
}
