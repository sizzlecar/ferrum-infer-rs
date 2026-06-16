// Standalone native-CUDA probe for Gemma3-27B GPTQ gate_up projection shape.
//
// Product tail MLP currently uses one fused Marlin projection:
//   [m, hidden] x [hidden, 2 * intermediate] -> [m, 2 * intermediate]
// followed by GeGLU over the two halves.
//
// This probe compares that product-shaped fused projection against a possible
// split design:
//   gate: [m, hidden] x [hidden, intermediate]
//   up:   [m, hidden] x [hidden, intermediate]
// followed by a GeGLU kernel over separate gate/up buffers.
//
// It intentionally bypasses Cargo and model loading. The goal is to decide
// whether the largest tail-MLP projection is worth product work as a split
// projection before touching loader/runtime code.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
constexpr int WEIGHT_CYCLE_LAYERS = 8;
constexpr int WARMUP_ITERS = 3;
constexpr int TIMED_ITERS = 32;

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

__global__ void fused_gelu_tanh_mul_separate_f16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ output,
    const int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const float g = __half2float(gate[idx]);
        const float u = __half2float(up[idx]);
        float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
        inner = fminf(fmaxf(inner, -9.5f), 9.5f);
        const float gelu_g = 0.5f * g * (1.0f + tanhf(inner));
        output[idx] = __float2half(gelu_g * u);
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
    __half* scales = nullptr;
    void* workspace = nullptr;
    size_t qweight_bytes = 0;
    size_t scale_count = 0;
    size_t workspace_bytes = 0;
};

struct LayerWeights {
    MarlinBuffers fused_gate_up;
    MarlinBuffers gate;
    MarlinBuffers up;
};

struct Buffers {
    __half* input = nullptr;
    __half* fused_gate_up_out = nullptr;
    __half* gate_out = nullptr;
    __half* up_out = nullptr;
    __half* act_out = nullptr;
    std::vector<LayerWeights> layers;
};

static MarlinBuffers alloc_marlin(cudaStream_t stream, const Shape& shape, int pattern) {
    const size_t qweight_bytes =
        round_up(static_cast<size_t>(shape.k) * static_cast<size_t>(shape.n) / 2, 16);
    const size_t scale_count =
        static_cast<size_t>(shape.k / GROUP_SIZE) * static_cast<size_t>(shape.n);
    const size_t scale_bytes = round_up(scale_count * sizeof(__half), 16);
    const size_t workspace_bytes =
        round_up(static_cast<size_t>(shape.n / 128) * MAX_PAR * sizeof(int), 16);

    MarlinBuffers buffers;
    buffers.qweight = cuda_alloc_bytes(qweight_bytes, pattern);
    buffers.scales = static_cast<__half*>(cuda_alloc_bytes(scale_bytes, 0x00));
    buffers.workspace = cuda_alloc_bytes(workspace_bytes, 0x00);
    buffers.qweight_bytes = qweight_bytes;
    buffers.scale_count = scale_count;
    buffers.workspace_bytes = workspace_bytes;
    fill_half(stream, buffers.scales, 0.001f, static_cast<int>(scale_count));
    return buffers;
}

static LayerWeights alloc_layer(cudaStream_t stream, int layer_idx) {
    const Shape fused{"fused_gate_up", HIDDEN, GATE_UP_N};
    const Shape split{"split_gate_or_up", HIDDEN, INTERMEDIATE};
    return LayerWeights{
        alloc_marlin(stream, fused, 0x11 + layer_idx),
        alloc_marlin(stream, split, 0x31 + layer_idx),
        alloc_marlin(stream, split, 0x51 + layer_idx),
    };
}

static Buffers alloc_buffers(cudaStream_t stream, int max_m) {
    Buffers buffers;
    buffers.input = static_cast<__half*>(
        cuda_alloc_bytes(static_cast<size_t>(max_m) * HIDDEN * sizeof(__half), 0x00));
    buffers.fused_gate_up_out = static_cast<__half*>(cuda_alloc_bytes(
        static_cast<size_t>(max_m) * GATE_UP_N * sizeof(__half), 0x00));
    buffers.gate_out = static_cast<__half*>(cuda_alloc_bytes(
        static_cast<size_t>(max_m) * INTERMEDIATE * sizeof(__half), 0x00));
    buffers.up_out = static_cast<__half*>(cuda_alloc_bytes(
        static_cast<size_t>(max_m) * INTERMEDIATE * sizeof(__half), 0x00));
    buffers.act_out = static_cast<__half*>(cuda_alloc_bytes(
        static_cast<size_t>(max_m) * INTERMEDIATE * sizeof(__half), 0x00));

    buffers.layers.reserve(WEIGHT_CYCLE_LAYERS);
    for (int i = 0; i < WEIGHT_CYCLE_LAYERS; ++i) {
        buffers.layers.push_back(alloc_layer(stream, i));
    }
    fill_half(stream, buffers.input, 0.125f, max_m * HIDDEN);
    CHK(cudaStreamSynchronize(stream));
    return buffers;
}

static void free_marlin(MarlinBuffers& buffers) {
    CHK(cudaFree(buffers.qweight));
    CHK(cudaFree(buffers.scales));
    CHK(cudaFree(buffers.workspace));
}

static void free_buffers(Buffers& buffers) {
    CHK(cudaFree(buffers.input));
    CHK(cudaFree(buffers.fused_gate_up_out));
    CHK(cudaFree(buffers.gate_out));
    CHK(cudaFree(buffers.up_out));
    CHK(cudaFree(buffers.act_out));
    for (LayerWeights& layer : buffers.layers) {
        free_marlin(layer.fused_gate_up);
        free_marlin(layer.gate);
        free_marlin(layer.up);
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

static void launch_geglu_interleaved(cudaStream_t stream, Buffers& buffers, int m) {
    const int total = m * INTERMEDIATE;
    const int block = 256;
    fused_gelu_tanh_mul_interleaved_f16<<<ceil_div_int(total, block), block, 0, stream>>>(
        buffers.fused_gate_up_out,
        buffers.act_out,
        INTERMEDIATE,
        total);
    CHK(cudaGetLastError());
}

static void launch_geglu_separate(cudaStream_t stream, Buffers& buffers, int m) {
    const int total = m * INTERMEDIATE;
    const int block = 256;
    fused_gelu_tanh_mul_separate_f16<<<ceil_div_int(total, block), block, 0, stream>>>(
        buffers.gate_out,
        buffers.up_out,
        buffers.act_out,
        total);
    CHK(cudaGetLastError());
}

static void launch_product_fused(cudaStream_t stream, Buffers& buffers, LayerWeights& layer, int m) {
    const Shape fused{"fused_gate_up", HIDDEN, GATE_UP_N};
    launch_marlin(
        stream,
        "fused_gate_up",
        buffers.input,
        layer.fused_gate_up,
        buffers.fused_gate_up_out,
        m,
        fused);
    launch_geglu_interleaved(stream, buffers, m);
}

static void launch_split_serial(cudaStream_t stream, Buffers& buffers, LayerWeights& layer, int m) {
    const Shape split{"split_gate_or_up", HIDDEN, INTERMEDIATE};
    launch_marlin(stream, "split_gate", buffers.input, layer.gate, buffers.gate_out, m, split);
    launch_marlin(stream, "split_up", buffers.input, layer.up, buffers.up_out, m, split);
    launch_geglu_separate(stream, buffers, m);
}

static void launch_split_overlap(
    cudaStream_t main_stream,
    cudaStream_t aux_stream,
    cudaEvent_t aux_done,
    Buffers& buffers,
    LayerWeights& layer,
    int m) {
    const Shape split{"split_gate_or_up", HIDDEN, INTERMEDIATE};
    launch_marlin(
        main_stream,
        "split_gate_overlap",
        buffers.input,
        layer.gate,
        buffers.gate_out,
        m,
        split);
    launch_marlin(
        aux_stream,
        "split_up_overlap",
        buffers.input,
        layer.up,
        buffers.up_out,
        m,
        split);
    CHK(cudaEventRecord(aux_done, aux_stream));
    CHK(cudaStreamWaitEvent(main_stream, aux_done, 0));
    launch_geglu_separate(main_stream, buffers, m);
}

enum class Mode {
    ProductFused,
    SplitSerial,
    SplitOverlap,
};

static const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::ProductFused:
            return "cycle8_product_fused_gate_up_act";
        case Mode::SplitSerial:
            return "cycle8_split_serial_gate_up_act";
        case Mode::SplitOverlap:
            return "cycle8_split_overlap_gate_up_act";
    }
    return "unknown";
}

static double time_mode_us(
    cudaStream_t main_stream,
    cudaStream_t aux_stream,
    cudaEvent_t aux_done,
    Buffers& buffers,
    Mode mode,
    int m) {
    double total_us = 0.0;
    int timed_calls = 0;
    for (int iter = 0; iter < WARMUP_ITERS + TIMED_ITERS; ++iter) {
        for (int layer_idx = 0; layer_idx < WEIGHT_CYCLE_LAYERS; ++layer_idx) {
            LayerWeights& layer = buffers.layers[layer_idx];
            CHK(cudaStreamSynchronize(main_stream));
            CHK(cudaStreamSynchronize(aux_stream));
            const auto start = std::chrono::steady_clock::now();
            switch (mode) {
                case Mode::ProductFused:
                    launch_product_fused(main_stream, buffers, layer, m);
                    break;
                case Mode::SplitSerial:
                    launch_split_serial(main_stream, buffers, layer, m);
                    break;
                case Mode::SplitOverlap:
                    launch_split_overlap(main_stream, aux_stream, aux_done, buffers, layer, m);
                    break;
            }
            CHK(cudaStreamSynchronize(main_stream));
            CHK(cudaStreamSynchronize(aux_stream));
            CHK(cudaGetLastError());
            const auto stop = std::chrono::steady_clock::now();
            if (iter >= WARMUP_ITERS) {
                total_us += std::chrono::duration<double, std::micro>(stop - start).count();
                timed_calls += 1;
            }
        }
    }
    return total_us / static_cast<double>(timed_calls);
}

static void print_row(Mode mode, int m, double us, double product_us) {
    const double projection_ops =
        2.0 * static_cast<double>(m) * HIDDEN * static_cast<double>(GATE_UP_N);
    const double tflops = projection_ops / (us * 1.0e6);
    const double speedup = product_us / us;
    std::printf("%s,%d,%.3f,%.4f,%.2f,0\n", mode_name(mode), m, us, speedup, tflops);
}

int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    std::printf(
        "gemma3_gate_up_split_perf device=%s sm=%d.%d h=%d im=%d layers=%d\n",
        prop.name,
        prop.major,
        prop.minor,
        HIDDEN,
        INTERMEDIATE,
        WEIGHT_CYCLE_LAYERS);
    std::printf("mode,m,segment_host_us,speedup_vs_product,effective_projection_tflops,ret\n");

    const std::vector<int> m_values = {1, 10, 16, 23, 32};
    int max_m = 0;
    for (int m : m_values) {
        max_m = max_m > m ? max_m : m;
    }

    cudaStream_t main_stream;
    cudaStream_t aux_stream;
    CHK(cudaStreamCreate(&main_stream));
    CHK(cudaStreamCreateWithFlags(&aux_stream, cudaStreamNonBlocking));
    cudaEvent_t aux_done;
    CHK(cudaEventCreateWithFlags(&aux_done, cudaEventDisableTiming));

    Buffers buffers = alloc_buffers(main_stream, max_m);
    std::fprintf(
        stderr,
        "qweight_bytes fused=%.1fMB split_total=%.1fMB layers=%d\n",
        static_cast<double>(buffers.layers[0].fused_gate_up.qweight_bytes) / (1024.0 * 1024.0),
        static_cast<double>(
            buffers.layers[0].gate.qweight_bytes + buffers.layers[0].up.qweight_bytes)
            / (1024.0 * 1024.0),
        WEIGHT_CYCLE_LAYERS);

    for (int m : m_values) {
        const double product_us =
            time_mode_us(main_stream, aux_stream, aux_done, buffers, Mode::ProductFused, m);
        print_row(Mode::ProductFused, m, product_us, product_us);
        const double split_serial_us =
            time_mode_us(main_stream, aux_stream, aux_done, buffers, Mode::SplitSerial, m);
        print_row(Mode::SplitSerial, m, split_serial_us, product_us);
        const double split_overlap_us =
            time_mode_us(main_stream, aux_stream, aux_done, buffers, Mode::SplitOverlap, m);
        print_row(Mode::SplitOverlap, m, split_overlap_us, product_us);
    }

    free_buffers(buffers);
    CHK(cudaEventDestroy(aux_done));
    CHK(cudaStreamDestroy(aux_stream));
    CHK(cudaStreamDestroy(main_stream));
    std::printf("\nVERDICT: gemma3 gate_up split native CUDA probe complete\n");
    return 0;
}
