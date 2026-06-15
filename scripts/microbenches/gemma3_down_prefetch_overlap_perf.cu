// Standalone native-CUDA probe for overlapping Gemma3-27B GPTQ down_proj
// qweight warmup with the gate_up+GeGLU producer.
//
// Prior probes showed:
//   - simple stream L2 policy works in a single-layer loop;
//   - it does not survive product-like multi-layer weight rotation;
//   - explicit down warmup is an upper bound but adds an extra down read.
//
// This probe tests the next concrete lever: a lightweight qweight/scales read
// kernel on a second stream, overlapped with gate_up+GeGLU, followed by down.
// It reports both down kernel time and host-synchronized segment time so a
// warmup that only improves down by adding equal or greater wall time can be
// rejected before touching product code.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
constexpr int PREFETCH_BLOCKS = 512;
constexpr int PREFETCH_THREADS = 256;

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

__global__ void warm_uint4_kernel(const uint4* ptr, int n_vec4, unsigned int* sink) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    unsigned int acc = static_cast<unsigned int>(idx);
    for (int i = idx; i < n_vec4; i += stride) {
        const uint4 v = ptr[i];
        acc ^= v.x;
        acc = acc * 1664525u + v.y;
        acc ^= v.z;
        acc = acc * 1013904223u + v.w;
    }
    if (threadIdx.x == 0) {
        sink[blockIdx.x] = acc;
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
    size_t scales_bytes = 0;
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
    unsigned int* warm_sink = nullptr;
    std::vector<LayerWeights> layers;
};

struct Mode {
    const char* name;
    bool access_policy;
    bool warm_qweight;
    bool warm_scales;
    bool overlap_warm;
    bool down_warm_upper_bound;
};

struct Timing {
    double down_us = 0.0;
    double segment_host_us = 0.0;
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
    buffers.scales_bytes = scale_bytes;
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
    buffers.warm_sink = static_cast<unsigned int*>(
        cuda_alloc_bytes(static_cast<size_t>(PREFETCH_BLOCKS) * sizeof(unsigned int), 0x00));
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
    CHK(cudaFree(buffers.warm_sink));
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

static void warm_bytes(cudaStream_t stream, const void* ptr, size_t bytes, unsigned int* sink) {
    const int n_vec4 = static_cast<int>(bytes / sizeof(uint4));
    if (n_vec4 <= 0) {
        return;
    }
    warm_uint4_kernel<<<PREFETCH_BLOCKS, PREFETCH_THREADS, 0, stream>>>(
        static_cast<const uint4*>(ptr),
        n_vec4,
        sink);
    CHK(cudaGetLastError());
}

static void warm_down(cudaStream_t stream, Buffers& buffers, LayerWeights& layer, bool scales) {
    warm_bytes(stream, layer.down.qweight, layer.down.qweight_bytes, buffers.warm_sink);
    if (scales) {
        warm_bytes(stream, layer.down.scales, layer.down.scales_bytes, buffers.warm_sink);
    }
}

static void reset_stream_policy(cudaStream_t stream) {
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = nullptr;
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitRatio = 0.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    CHK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
}

static void set_down_policy(cudaStream_t stream, const cudaDeviceProp& prop, const MarlinBuffers& down) {
    if (prop.persistingL2CacheMaxSize <= 0 || prop.accessPolicyMaxWindowSize <= 0) {
        reset_stream_policy(stream);
        return;
    }
    const size_t max_persist = static_cast<size_t>(prop.persistingL2CacheMaxSize);
    const size_t max_window = static_cast<size_t>(prop.accessPolicyMaxWindowSize);
    const size_t window = std::min(down.qweight_bytes, max_window);
    CHK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, std::min(max_persist, window)));

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = down.qweight;
    attr.accessPolicyWindow.num_bytes = window;
    attr.accessPolicyWindow.hitRatio = 0.6f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    CHK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
}

static Timing time_mode(
    cudaStream_t main_stream,
    cudaStream_t warm_stream,
    cudaEvent_t warm_done,
    const cudaDeviceProp& prop,
    Buffers& buffers,
    const Mode& mode,
    int m) {
    const Shape down_shape{"down", INTERMEDIATE, HIDDEN};
    cudaEvent_t down_start;
    cudaEvent_t down_stop;
    CHK(cudaEventCreate(&down_start));
    CHK(cudaEventCreate(&down_stop));

    Timing timing;
    int timed_calls = 0;
    for (int iter = 0; iter < WARMUP_ITERS + TIMED_ITERS; ++iter) {
        for (int layer_idx = 0; layer_idx < MAX_LAYERS; ++layer_idx) {
            LayerWeights& layer = buffers.layers[layer_idx];
            CHK(cudaStreamSynchronize(main_stream));
            CHK(cudaStreamSynchronize(warm_stream));
            CHK(cudaCtxResetPersistingL2Cache());
            reset_stream_policy(main_stream);
            reset_stream_policy(warm_stream);
            if (mode.access_policy) {
                set_down_policy(main_stream, prop, layer.down);
                set_down_policy(warm_stream, prop, layer.down);
            }

            const auto segment_start = std::chrono::steady_clock::now();
            if (mode.down_warm_upper_bound) {
                launch_marlin(
                    main_stream,
                    "down_warm",
                    buffers.const_input,
                    layer.down,
                    buffers.down_out,
                    m,
                    down_shape);
            } else if (mode.warm_qweight && mode.overlap_warm) {
                warm_down(warm_stream, buffers, layer, mode.warm_scales);
                CHK(cudaEventRecord(warm_done, warm_stream));
            } else if (mode.warm_qweight) {
                warm_down(main_stream, buffers, layer, mode.warm_scales);
            }

            prepare_geglu(main_stream, buffers, layer, m);
            if (mode.warm_qweight && mode.overlap_warm) {
                CHK(cudaStreamWaitEvent(main_stream, warm_done, 0));
            }

            CHK(cudaEventRecord(down_start, main_stream));
            launch_marlin(
                main_stream,
                "down",
                buffers.const_input,
                layer.down,
                buffers.down_out,
                m,
                down_shape);
            CHK(cudaEventRecord(down_stop, main_stream));
            CHK(cudaStreamSynchronize(main_stream));
            CHK(cudaStreamSynchronize(warm_stream));
            const auto segment_stop = std::chrono::steady_clock::now();
            CHK(cudaGetLastError());
            if (iter >= WARMUP_ITERS) {
                float ms = 0.0f;
                CHK(cudaEventElapsedTime(&ms, down_start, down_stop));
                timing.down_us += static_cast<double>(ms) * 1000.0;
                timing.segment_host_us +=
                    std::chrono::duration<double, std::micro>(segment_stop - segment_start).count();
                timed_calls += 1;
            }
        }
    }

    CHK(cudaEventDestroy(down_start));
    CHK(cudaEventDestroy(down_stop));
    timing.down_us /= static_cast<double>(timed_calls);
    timing.segment_host_us /= static_cast<double>(timed_calls);
    return timing;
}

static void print_row(const char* mode, int m, const Timing& timing) {
    const double ops = 2.0 * static_cast<double>(m) * INTERMEDIATE * HIDDEN;
    const double tflops = ops / (timing.down_us * 1.0e6);
    std::printf(
        "%s,%d,%.3f,%.3f,%.2f,0\n",
        mode,
        m,
        timing.down_us,
        timing.segment_host_us,
        tflops);
}

int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    std::printf(
        "gemma3_down_prefetch_overlap_perf device=%s sm=%d.%d h=%d im=%d layers=%d "
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
    std::printf("mode,m,down_us,segment_host_us,useful_tflops,ret\n");

    const std::vector<int> m_values = {16, 32};
    int max_m = 0;
    for (int m : m_values) {
        max_m = max_m > m ? max_m : m;
    }

    cudaStream_t main_stream;
    cudaStream_t warm_stream;
    CHK(cudaStreamCreate(&main_stream));
    CHK(cudaStreamCreateWithFlags(&warm_stream, cudaStreamNonBlocking));
    cudaEvent_t warm_done;
    CHK(cudaEventCreateWithFlags(&warm_done, cudaEventDisableTiming));
    Buffers buffers = alloc_buffers(main_stream, max_m);
    std::fprintf(
        stderr,
        "qweight_bytes gate_up=%zu down=%zu down_scales=%zu total_layers=%d\n",
        buffers.layers[0].gate_up.qweight_bytes,
        buffers.layers[0].down.qweight_bytes,
        buffers.layers[0].down.scales_bytes,
        MAX_LAYERS);

    const Mode modes[] = {
        {"cycle8_no_prefetch", false, false, false, false, false},
        {"cycle8_policy_only", true, false, false, false, false},
        {"cycle8_overlap_qweight", true, true, false, true, false},
        {"cycle8_overlap_qweight_scales", true, true, true, true, false},
        {"cycle8_serial_qweight_scales", true, true, true, false, false},
        {"cycle8_down_warm_upper", true, false, false, false, true},
    };

    for (int m : m_values) {
        for (const Mode& mode : modes) {
            const Timing timing =
                time_mode(main_stream, warm_stream, warm_done, prop, buffers, mode, m);
            print_row(mode.name, m, timing);
        }
    }

    CHK(cudaCtxResetPersistingL2Cache());
    reset_stream_policy(main_stream);
    reset_stream_policy(warm_stream);
    free_buffers(buffers);
    CHK(cudaEventDestroy(warm_done));
    CHK(cudaStreamDestroy(warm_stream));
    CHK(cudaStreamDestroy(main_stream));
    std::printf("\nVERDICT: gemma3 down prefetch-overlap native CUDA probe complete\n");
    return 0;
}
