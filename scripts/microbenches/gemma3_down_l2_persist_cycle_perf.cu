// Standalone native-CUDA probe for Gemma3-27B GPTQ down_proj L2 persistence
// under product-like layer weight rotation.
//
// The single-layer L2 persistence probe showed CUDA access-policy can keep a
// layer's down_proj qweight hot across that layer's gate_up+GeGLU producer.
// Product decode, however, revisits the same layer only after many other layer
// weights have run. This probe allocates multiple synthetic layer weight sets
// and measures whether simple per-layer access-policy still helps when layers
// rotate. If it only helps in single-layer loops, it is not directly
// productizable without an additional prefetch/reuse strategy.

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
constexpr int MAX_LAYERS = 8;
constexpr int WARMUP_ITERS = 3;
constexpr int TIMED_ITERS = 24;

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

struct LayerWeights {
    MarlinBuffers gate_up;
    MarlinBuffers down;
};

struct Buffers {
    __half* norm_input = nullptr;
    __half* gate_up_out = nullptr;
    __half* geglu_out = nullptr;
    __half* const_input = nullptr;
    __half* down_out = nullptr;
    std::vector<LayerWeights> layers;
};

struct Mode {
    const char* name;
    int active_layers;
    bool persist_down;
    double hit_ratio;
    bool warm_down_before_gate_up;
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

static LayerWeights alloc_layer(cudaStream_t stream) {
    const Shape gate_up_shape{"gate_up", HIDDEN, GATE_UP_N};
    const Shape down_shape{"down", INTERMEDIATE, HIDDEN};
    return LayerWeights{
        alloc_marlin(stream, gate_up_shape),
        alloc_marlin(stream, down_shape),
    };
}

static Buffers alloc_buffers(cudaStream_t stream, int max_m) {
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
    buffers.layers.reserve(MAX_LAYERS);
    for (int i = 0; i < MAX_LAYERS; ++i) {
        buffers.layers.push_back(alloc_layer(stream));
    }
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
    for (LayerWeights& layer : buffers.layers) {
        free_marlin(layer.gate_up);
        free_marlin(layer.down);
    }
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

static void prepare_geglu(cudaStream_t stream, Buffers& buffers, LayerWeights& layer, int m) {
    const Shape gate_up_shape{"gate_up", HIDDEN, GATE_UP_N};
    launch_marlin(
        stream,
        "gate_up",
        buffers.norm_input,
        layer.gate_up,
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
    bool enable,
    double hit_ratio) {
    if (!enable || prop.persistingL2CacheMaxSize <= 0 || prop.accessPolicyMaxWindowSize <= 0) {
        reset_stream_policy(stream);
        return 0;
    }
    const size_t max_persist = static_cast<size_t>(prop.persistingL2CacheMaxSize);
    const size_t max_window = static_cast<size_t>(prop.accessPolicyMaxWindowSize);
    const size_t window = std::min({down.qweight_bytes, max_window});
    CHK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, std::min(max_persist, window)));

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = down.qweight;
    attr.accessPolicyWindow.num_bytes = window;
    attr.accessPolicyWindow.hitRatio = static_cast<float>(hit_ratio);
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    CHK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
    return window;
}

static double time_mode_us(
    cudaStream_t stream,
    const cudaDeviceProp& prop,
    Buffers& buffers,
    const Mode& mode,
    int m) {
    const Shape down_shape{"down", INTERMEDIATE, HIDDEN};
    cudaEvent_t start;
    cudaEvent_t stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));
    double total_us = 0.0;
    int timed_calls = 0;

    for (int iter = 0; iter < WARMUP_ITERS + TIMED_ITERS; ++iter) {
        for (int layer_idx = 0; layer_idx < mode.active_layers; ++layer_idx) {
            LayerWeights& layer = buffers.layers[layer_idx];
            apply_down_policy(stream, prop, layer.down, mode.persist_down, mode.hit_ratio);
            if (mode.warm_down_before_gate_up) {
                launch_marlin(
                    stream,
                    "down_warm",
                    buffers.const_input,
                    layer.down,
                    buffers.down_out,
                    m,
                    down_shape);
            }
            prepare_geglu(stream, buffers, layer, m);

            CHK(cudaEventRecord(start, stream));
            launch_marlin(
                stream,
                "down",
                buffers.const_input,
                layer.down,
                buffers.down_out,
                m,
                down_shape);
            CHK(cudaEventRecord(stop, stream));
            CHK(cudaEventSynchronize(stop));
            CHK(cudaGetLastError());
            if (iter >= WARMUP_ITERS) {
                float ms = 0.0f;
                CHK(cudaEventElapsedTime(&ms, start, stop));
                total_us += static_cast<double>(ms) * 1000.0;
                timed_calls += 1;
            }
        }
    }

    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
    return total_us / static_cast<double>(timed_calls);
}

static void print_row(const char* mode, int layers, int m, double hit_ratio, size_t window, double us) {
    const double ops = 2.0 * static_cast<double>(m) * INTERMEDIATE * HIDDEN;
    const double tflops = ops / (us * 1.0e6);
    std::printf("%s,%d,%d,%.2f,%zu,%.3f,%.2f,0\n", mode, layers, m, hit_ratio, window, us, tflops);
}

int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    std::printf(
        "gemma3_down_l2_persist_cycle_perf device=%s sm=%d.%d h=%d im=%d layers=%d "
        "l2=%d persist_max=%d access_window_max=%d\n",
        prop.name,
        prop.major,
        prop.minor,
        HIDDEN,
        INTERMEDIATE,
        MAX_LAYERS,
        prop.l2CacheSize,
        prop.persistingL2CacheMaxSize,
        prop.accessPolicyMaxWindowSize);
    std::printf("mode,layers,m,hit_ratio,policy_window_bytes,down_us,useful_tflops,ret\n");

    const std::vector<int> m_values = {16, 32};
    int max_m = 0;
    for (int m : m_values) {
        max_m = max_m > m ? max_m : m;
    }

    cudaStream_t stream;
    CHK(cudaStreamCreate(&stream));
    Buffers buffers = alloc_buffers(stream, max_m);
    std::fprintf(
        stderr,
        "qweight_bytes gate_up=%zu down=%zu total_layers=%d\n",
        buffers.layers[0].gate_up.qweight_bytes,
        buffers.layers[0].down.qweight_bytes,
        MAX_LAYERS);

    const Mode modes[] = {
        {"single_layer_no_policy", 1, false, 0.0, false},
        {"single_layer_persist_hit60", 1, true, 0.6, false},
        {"cycle8_no_policy", MAX_LAYERS, false, 0.0, false},
        {"cycle8_persist_hit60", MAX_LAYERS, true, 0.6, false},
        {"cycle8_persist_hit60_down_warm", MAX_LAYERS, true, 0.6, true},
    };

    for (int m : m_values) {
        for (const Mode& mode : modes) {
            reset_stream_policy(stream);
            const size_t window = mode.persist_down
                ? std::min(
                      buffers.layers[0].down.qweight_bytes,
                      static_cast<size_t>(std::max(0, prop.accessPolicyMaxWindowSize)))
                : 0;
            const double us = time_mode_us(stream, prop, buffers, mode, m);
            print_row(mode.name, mode.active_layers, m, mode.hit_ratio, window, us);
        }
    }

    reset_stream_policy(stream);
    free_buffers(buffers);
    CHK(cudaStreamDestroy(stream));
    std::printf("\nVERDICT: gemma3 down L2 persistence cycle native CUDA probe complete\n");
    return 0;
}
