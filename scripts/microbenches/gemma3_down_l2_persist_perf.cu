// Standalone native-CUDA probe for a Gemma3-27B GPTQ down_proj L2 residency lever.
//
// The input-source probe showed down_proj is fast only when repeated on warm
// input/weight state; running product-shaped gate_up+GeGLU before down makes
// down cold even if down reads an unrelated constant input. This diagnostic
// applies CUDA's persisting L2 access-policy window to down qweight and checks
// whether a simple stream-level residency hint can preserve down performance
// across the gate_up -> GeGLU producer sequence.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" int marlin_cuda(
    const void* A, const void* B, void* C, void* s, int prob_m, int prob_n,
    int prob_k, void* workspace, int groupsize, int dev, cudaStream_t stream,
    int thread_k, int thread_n, int sms, int max_par, int prob_n_full);

extern "C" __global__ void fused_gelu_tanh_mul_interleaved_f16(
    const __half* gate_up,
    __half* output,
    const int inter,
    const int total);

#define CHK(stmt)                                                                \
    do {                                                                         \
        cudaError_t err__ = (stmt);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(err__));                            \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

constexpr int HIDDEN = 5376;
constexpr int INTERMEDIATE = 21504;
constexpr int GATE_UP_N = INTERMEDIATE * 2;
constexpr int GROUP_SIZE = 128;
constexpr int MAX_PAR = 16;
constexpr int WARMUP_ITERS = 5;
constexpr int TIMED_ITERS = 80;

static int ceil_div_int(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

static size_t round_up(size_t value, size_t align) {
    return ((value + align - 1) / align) * align;
}

static void* cuda_alloc_bytes(size_t bytes, int pattern) {
    void* ptr = nullptr;
    CHK(cudaMalloc(&ptr, bytes));
    CHK(cudaMemset(ptr, pattern, bytes));
    return ptr;
}

__global__ void fill_half_kernel(__half* ptr, float value, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = __float2half(value);
    }
}

static void fill_half(cudaStream_t stream, __half* ptr, float value, int n) {
    const int block = 256;
    fill_half_kernel<<<ceil_div_int(n, block), block, 0, stream>>>(ptr, value, n);
    CHK(cudaGetLastError());
}

struct Shape {
    const char* name;
    int k;
    int n;
};

struct MarlinBuffers {
    void* qweight = nullptr;
    size_t qweight_bytes = 0;
    __half* scales = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
};

struct Buffers {
    __half* norm_input = nullptr;
    __half* gate_up_out = nullptr;
    __half* geglu_out = nullptr;
    __half* const_input = nullptr;
    __half* down_out = nullptr;
    MarlinBuffers gate_up;
    MarlinBuffers down;
};

struct PolicyMode {
    const char* name;
    bool enable;
    double window_fraction;
    double hit_ratio;
};

static MarlinBuffers alloc_marlin(cudaStream_t stream, const Shape& shape) {
    const size_t qweight_bytes =
        round_up(static_cast<size_t>(shape.k) * static_cast<size_t>(shape.n) / 2, 16);
    const size_t scale_count =
        static_cast<size_t>(shape.k / GROUP_SIZE) * static_cast<size_t>(shape.n);
    const size_t scale_bytes = round_up(scale_count * sizeof(__half), 16);
    const size_t workspace_bytes =
        round_up(static_cast<size_t>(shape.n / 128) * MAX_PAR * sizeof(int), 16);
    MarlinBuffers buffers;
    buffers.qweight = cuda_alloc_bytes(qweight_bytes, 0x11);
    buffers.qweight_bytes = qweight_bytes;
    buffers.scales = static_cast<__half*>(cuda_alloc_bytes(scale_bytes, 0x00));
    buffers.workspace = cuda_alloc_bytes(workspace_bytes, 0x00);
    buffers.workspace_bytes = workspace_bytes;
    fill_half(stream, buffers.scales, 0.001f, static_cast<int>(scale_count));
    return buffers;
}

static Buffers alloc_buffers(cudaStream_t stream, int max_m) {
    const Shape gate_up_shape{"gate_up", HIDDEN, GATE_UP_N};
    const Shape down_shape{"down", INTERMEDIATE, HIDDEN};
    Buffers buffers;
    buffers.norm_input = static_cast<__half*>(
        cuda_alloc_bytes(static_cast<size_t>(max_m) * HIDDEN * sizeof(__half), 0x00));
    buffers.gate_up_out = static_cast<__half*>(
        cuda_alloc_bytes(static_cast<size_t>(max_m) * GATE_UP_N * sizeof(__half), 0x00));
    buffers.geglu_out = static_cast<__half*>(
        cuda_alloc_bytes(static_cast<size_t>(max_m) * INTERMEDIATE * sizeof(__half), 0x00));
    buffers.const_input = static_cast<__half*>(
        cuda_alloc_bytes(static_cast<size_t>(max_m) * INTERMEDIATE * sizeof(__half), 0x00));
    buffers.down_out = static_cast<__half*>(
        cuda_alloc_bytes(static_cast<size_t>(max_m) * HIDDEN * sizeof(__half), 0x00));
    buffers.gate_up = alloc_marlin(stream, gate_up_shape);
    buffers.down = alloc_marlin(stream, down_shape);
    fill_half(stream, buffers.norm_input, 0.125f, max_m * HIDDEN);
    fill_half(stream, buffers.const_input, 0.125f, max_m * INTERMEDIATE);
    CHK(cudaStreamSynchronize(stream));
    return buffers;
}

static void free_marlin(MarlinBuffers& buffers) {
    CHK(cudaFree(buffers.qweight));
    CHK(cudaFree(buffers.scales));
    CHK(cudaFree(buffers.workspace));
}

static void free_buffers(Buffers& buffers) {
    CHK(cudaFree(buffers.norm_input));
    CHK(cudaFree(buffers.gate_up_out));
    CHK(cudaFree(buffers.geglu_out));
    CHK(cudaFree(buffers.const_input));
    CHK(cudaFree(buffers.down_out));
    free_marlin(buffers.gate_up);
    free_marlin(buffers.down);
}

static void launch_marlin(
    cudaStream_t stream,
    const char* label,
    const void* input,
    MarlinBuffers& weight,
    void* output,
    int m,
    const Shape& shape) {
    CHK(cudaMemsetAsync(weight.workspace, 0, weight.workspace_bytes, stream));
    const int ret = marlin_cuda(
        input,
        weight.qweight,
        output,
        weight.scales,
        m,
        shape.n,
        shape.k,
        weight.workspace,
        GROUP_SIZE,
        0,
        stream,
        -1,
        -1,
        -1,
        MAX_PAR,
        -1);
    if (ret != 0) {
        std::fprintf(
            stderr,
            "marlin_cuda failed label=%s ret=%d m=%d n=%d k=%d\n",
            label,
            ret,
            m,
            shape.n,
            shape.k);
        std::exit(2);
    }
}

static void launch_geglu(cudaStream_t stream, Buffers& buffers, int m) {
    const int total = m * INTERMEDIATE;
    const int block = 256;
    fused_gelu_tanh_mul_interleaved_f16<<<ceil_div_int(total, block), block, 0, stream>>>(
        buffers.gate_up_out, buffers.geglu_out, INTERMEDIATE, total);
    CHK(cudaGetLastError());
}

static void prepare_geglu(cudaStream_t stream, Buffers& buffers, int m) {
    const Shape gate_up_shape{"gate_up", HIDDEN, GATE_UP_N};
    launch_marlin(
        stream,
        "gate_up",
        buffers.norm_input,
        buffers.gate_up,
        buffers.gate_up_out,
        m,
        gate_up_shape);
    launch_geglu(stream, buffers, m);
}

static void reset_stream_policy(cudaStream_t stream) {
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = nullptr;
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitRatio = 0.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    CHK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
    CHK(cudaCtxResetPersistingL2Cache());
}

static size_t apply_down_policy(
    cudaStream_t stream,
    const cudaDeviceProp& prop,
    const MarlinBuffers& down,
    const PolicyMode& mode) {
    reset_stream_policy(stream);
    if (!mode.enable || prop.persistingL2CacheMaxSize <= 0 || prop.accessPolicyMaxWindowSize <= 0) {
        return 0;
    }

    const size_t max_persist = static_cast<size_t>(prop.persistingL2CacheMaxSize);
    const size_t max_window = static_cast<size_t>(prop.accessPolicyMaxWindowSize);
    const size_t requested =
        std::max<size_t>(4096, static_cast<size_t>(down.qweight_bytes * mode.window_fraction));
    const size_t window = std::min({down.qweight_bytes, max_window, requested});
    const size_t limit = std::min(max_persist, std::max<size_t>(window, 4096));
    CHK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, limit));

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = down.qweight;
    attr.accessPolicyWindow.num_bytes = window;
    attr.accessPolicyWindow.hitRatio = static_cast<float>(mode.hit_ratio);
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    CHK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
    return window;
}

static float time_down_after_gate_up_us(
    cudaStream_t stream,
    Buffers& buffers,
    const __half* input,
    int m,
    bool warm_down_once) {
    const Shape down_shape{"down", INTERMEDIATE, HIDDEN};
    cudaEvent_t start;
    cudaEvent_t stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));
    float total_ms = 0.0f;
    for (int iter = 0; iter < WARMUP_ITERS + TIMED_ITERS; ++iter) {
        if (warm_down_once) {
            launch_marlin(stream, "down_warm", input, buffers.down, buffers.down_out, m, down_shape);
        }
        prepare_geglu(stream, buffers, m);
        CHK(cudaEventRecord(start, stream));
        launch_marlin(stream, "down", input, buffers.down, buffers.down_out, m, down_shape);
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

static float time_down_repeated_us(cudaStream_t stream, Buffers& buffers, const __half* input, int m) {
    const Shape down_shape{"down", INTERMEDIATE, HIDDEN};
    cudaEvent_t start;
    cudaEvent_t stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));
    float total_ms = 0.0f;
    for (int iter = 0; iter < WARMUP_ITERS + TIMED_ITERS; ++iter) {
        CHK(cudaEventRecord(start, stream));
        launch_marlin(stream, "down", input, buffers.down, buffers.down_out, m, down_shape);
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

static void print_row(const char* mode, int m, size_t policy_window, float us) {
    const double ops = 2.0 * static_cast<double>(m) * INTERMEDIATE * HIDDEN;
    const double tflops = ops / (static_cast<double>(us) * 1.0e6);
    std::printf("%s,%d,%zu,%.3f,%.2f,0\n", mode, m, policy_window, us, tflops);
}

int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    std::printf(
        "gemma3_down_l2_persist_perf device=%s sm=%d.%d h=%d im=%d l2=%d "
        "persist_max=%d access_window_max=%d\n",
        prop.name,
        prop.major,
        prop.minor,
        HIDDEN,
        INTERMEDIATE,
        prop.l2CacheSize,
        prop.persistingL2CacheMaxSize,
        prop.accessPolicyMaxWindowSize);
    std::printf("mode,m,policy_window_bytes,down_us,useful_tflops,ret\n");

    const std::vector<int> m_values = {1, 10, 16, 23, 32};
    int max_m = 0;
    for (int m : m_values) {
        max_m = max_m > m ? max_m : m;
    }

    cudaStream_t stream;
    CHK(cudaStreamCreate(&stream));
    Buffers buffers = alloc_buffers(stream, max_m);
    std::fprintf(
        stderr,
        "qweight_bytes gate_up=%zu down=%zu\n",
        buffers.gate_up.qweight_bytes,
        buffers.down.qweight_bytes);

    const PolicyMode modes[] = {
        {"no_policy_after_gate_up", false, 0.0, 0.0},
        {"persist_down_full_hit_100", true, 1.0, 1.0},
        {"persist_down_half_hit_100", true, 0.5, 1.0},
        {"persist_down_full_hit_60", true, 1.0, 0.6},
    };

    for (int m : m_values) {
        reset_stream_policy(stream);
        print_row(
            "down_repeated_warm_baseline",
            m,
            0,
            time_down_repeated_us(stream, buffers, buffers.const_input, m));
        for (const PolicyMode& mode : modes) {
            const size_t window = apply_down_policy(stream, prop, buffers.down, mode);
            print_row(
                mode.name,
                m,
                window,
                time_down_after_gate_up_us(stream, buffers, buffers.const_input, m, false));
        }
        const size_t warm_window = apply_down_policy(
            stream,
            prop,
            buffers.down,
            PolicyMode{"persist_down_full_hit_100_warm", true, 1.0, 1.0});
        print_row(
            "persist_down_full_hit_100_after_down_warm",
            m,
            warm_window,
            time_down_after_gate_up_us(stream, buffers, buffers.const_input, m, true));
    }

    reset_stream_policy(stream);
    free_buffers(buffers);
    CHK(cudaStreamDestroy(stream));
    std::printf("\nVERDICT: gemma3 down L2 persistence native CUDA probe complete\n");
    return 0;
}
