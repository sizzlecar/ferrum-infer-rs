// Native CUDA probe for monolithic vs segmented CUDA graph capture.
//
// This intentionally avoids Cargo, model loading, Torch, and vLLM runtime.
// It answers one narrow question before changing Ferrum's product graph path:
// for a Gemma3-like decode launch count, does a segmented graph strategy
// instantiate and replay cleanly, and what replay overhead does it add versus
// one monolithic graph?
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 \
//     scripts/microbenches/cuda_graph_segment_probe.cu \
//     -lcuda -lcudart -o /tmp/cuda_graph_segment_probe

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                            \
    do {                                                                           \
        cudaError_t err__ = (expr);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(err__));                                \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

namespace {

struct Options {
    int layers = 62;
    int kernels_per_layer = 8;
    int final_kernels = 2;
    int batch = 16;
    int hidden = 5376;
    int work_iters = 8;
    int segment_layers = 1;
    int warmup_iters = 8;
    int timed_iters = 80;
    bool eager_gap = true;
};

struct GraphExec {
    cudaGraphExec_t exec = nullptr;
    int start_layer = 0;
    int layers = 0;
    bool has_final = false;
};

__device__ __forceinline__ float clamp8(float x) {
    return fminf(fmaxf(x, -8.0f), 8.0f);
}

__global__ void init_kernel(float* a, float* b, int* state, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = static_cast<float>((i % 251) - 125) * 0.0007f;
        a[i] = v;
        b[i] = 0.0f;
    }
    if (i == 0) {
        state[0] = 1;
    }
}

__global__ void work_kernel(float* dst, const float* src, const int* state, int n, int salt, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = src[i] + static_cast<float>((state[0] + salt) & 15) * 0.00011f;
        for (int j = 0; j < iters; ++j) {
            x = fmaf(x, 1.00013f, 0.00017f * static_cast<float>((salt + j) & 31));
            x = x / (1.0f + fabsf(x) * 0.0003f);
        }
        dst[i] = clamp8(x);
    }
}

__global__ void eager_gap_kernel(float* a, int n, int salt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = clamp8(a[i] + static_cast<float>(salt & 7) * 0.00003f);
    }
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int parse_int_arg(const char* arg, const char* name, int fallback) {
    size_t len = std::strlen(name);
    if (std::strncmp(arg, name, len) == 0 && arg[len] == '=') {
        return std::atoi(arg + len + 1);
    }
    return fallback;
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        opt.layers = parse_int_arg(arg, "--layers", opt.layers);
        opt.kernels_per_layer = parse_int_arg(arg, "--kernels-per-layer", opt.kernels_per_layer);
        opt.final_kernels = parse_int_arg(arg, "--final-kernels", opt.final_kernels);
        opt.batch = parse_int_arg(arg, "--batch", opt.batch);
        opt.hidden = parse_int_arg(arg, "--hidden", opt.hidden);
        opt.work_iters = parse_int_arg(arg, "--work-iters", opt.work_iters);
        opt.segment_layers = parse_int_arg(arg, "--segment-layers", opt.segment_layers);
        opt.warmup_iters = parse_int_arg(arg, "--warmup-iters", opt.warmup_iters);
        opt.timed_iters = parse_int_arg(arg, "--timed-iters", opt.timed_iters);
        if (std::strcmp(arg, "--no-eager-gap") == 0) {
            opt.eager_gap = false;
        }
    }
    if (opt.layers <= 0 || opt.kernels_per_layer <= 0 || opt.batch <= 0 ||
        opt.hidden <= 0 || opt.segment_layers <= 0 || opt.timed_iters <= 0) {
        std::fprintf(stderr, "invalid non-positive option\n");
        std::exit(2);
    }
    return opt;
}

void upload_state(cudaStream_t stream, int* state, int value) {
    CUDA_CHECK(cudaMemcpyAsync(state, &value, sizeof(int), cudaMemcpyHostToDevice, stream));
}

void launch_layer_range(cudaStream_t stream,
                        float* a,
                        float* b,
                        int* state,
                        int n,
                        int blocks,
                        int threads,
                        const Options& opt,
                        int start_layer,
                        int layer_count,
                        bool include_final) {
    for (int layer = start_layer; layer < start_layer + layer_count; ++layer) {
        for (int k = 0; k < opt.kernels_per_layer; ++k) {
            int salt = layer * 131 + k * 17 + 3;
            if ((k & 1) == 0) {
                work_kernel<<<blocks, threads, 0, stream>>>(b, a, state, n, salt, opt.work_iters);
            } else {
                work_kernel<<<blocks, threads, 0, stream>>>(a, b, state, n, salt, opt.work_iters);
            }
        }
    }
    if (include_final) {
        for (int k = 0; k < opt.final_kernels; ++k) {
            int salt = 900000 + k * 19;
            if ((k & 1) == 0) {
                work_kernel<<<blocks, threads, 0, stream>>>(b, a, state, n, salt, opt.work_iters);
            } else {
                work_kernel<<<blocks, threads, 0, stream>>>(a, b, state, n, salt, opt.work_iters);
            }
        }
    }
}

bool capture_graph(cudaStream_t stream,
                   float* a,
                   float* b,
                   int* state,
                   int n,
                   int blocks,
                   int threads,
                   const Options& opt,
                   int start_layer,
                   int layer_count,
                   bool include_final,
                   GraphExec* out,
                   double* instantiate_ms,
                   std::string* error) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t0 = std::chrono::steady_clock::now();
    cudaGraph_t graph = nullptr;
    cudaError_t st = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (st != cudaSuccess) {
        *error = std::string("begin_capture: ") + cudaGetErrorString(st);
        return false;
    }
    launch_layer_range(stream, a, b, state, n, blocks, threads, opt, start_layer, layer_count, include_final);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
        cudaStreamEndCapture(stream, &graph);
        if (graph) {
            cudaGraphDestroy(graph);
        }
        *error = std::string("captured launch: ") + cudaGetErrorString(st);
        return false;
    }
    st = cudaStreamEndCapture(stream, &graph);
    if (st != cudaSuccess) {
        *error = std::string("end_capture: ") + cudaGetErrorString(st);
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    st = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (st != cudaSuccess) {
        cudaGraphDestroy(graph);
        *error = std::string("instantiate: ") + cudaGetErrorString(st);
        return false;
    }
    st = cudaGraphUpload(exec, stream);
    if (st != cudaSuccess) {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        *error = std::string("upload: ") + cudaGetErrorString(st);
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::steady_clock::now();
    cudaGraphDestroy(graph);

    out->exec = exec;
    out->start_layer = start_layer;
    out->layers = layer_count;
    out->has_final = include_final;
    *instantiate_ms = elapsed_ms(t0, t1);
    return true;
}

double bench_eager(cudaStream_t stream,
                   float* a,
                   float* b,
                   int* state,
                   int n,
                   int blocks,
                   int threads,
                   const Options& opt) {
    for (int i = 0; i < opt.warmup_iters; ++i) {
        upload_state(stream, state, 1000 + i);
        launch_layer_range(stream, a, b, state, n, blocks, threads, opt, 0, opt.layers, true);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.timed_iters; ++i) {
        upload_state(stream, state, 2000 + i);
        launch_layer_range(stream, a, b, state, n, blocks, threads, opt, 0, opt.layers, true);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto t1 = std::chrono::steady_clock::now();
    return elapsed_ms(t0, t1) / static_cast<double>(opt.timed_iters);
}

double bench_mono(cudaStream_t stream, cudaGraphExec_t exec, int* state, const Options& opt) {
    for (int i = 0; i < opt.warmup_iters; ++i) {
        upload_state(stream, state, 3000 + i);
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.timed_iters; ++i) {
        upload_state(stream, state, 4000 + i);
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto t1 = std::chrono::steady_clock::now();
    return elapsed_ms(t0, t1) / static_cast<double>(opt.timed_iters);
}

double bench_segments(cudaStream_t stream,
                      const std::vector<GraphExec>& graphs,
                      float* a,
                      int* state,
                      int n,
                      int blocks,
                      int threads,
                      const Options& opt) {
    auto replay_once = [&](int iter) {
        upload_state(stream, state, 5000 + iter);
        for (size_t i = 0; i < graphs.size(); ++i) {
            CUDA_CHECK(cudaGraphLaunch(graphs[i].exec, stream));
            if (opt.eager_gap && i + 1 < graphs.size()) {
                eager_gap_kernel<<<blocks, threads, 0, stream>>>(a, n, static_cast<int>(i));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    for (int i = 0; i < opt.warmup_iters; ++i) {
        replay_once(i);
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.timed_iters; ++i) {
        replay_once(1000 + i);
    }
    auto t1 = std::chrono::steady_clock::now();
    return elapsed_ms(t0, t1) / static_cast<double>(opt.timed_iters);
}

float checksum(float* a) {
    float host[32] = {};
    CUDA_CHECK(cudaMemcpy(host, a, sizeof(host), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (float v : host) {
        sum += v;
    }
    return sum;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int n = opt.batch * opt.hidden;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int total_kernels = opt.layers * opt.kernels_per_layer + opt.final_kernels;

    std::printf("GPU: %s\n", prop.name);
    std::printf("Config: layers=%d kernels_per_layer=%d final_kernels=%d total_kernels=%d\n",
                opt.layers, opt.kernels_per_layer, opt.final_kernels, total_kernels);
    std::printf("Config: batch=%d hidden=%d elements=%d work_iters=%d segment_layers=%d eager_gap=%d\n",
                opt.batch, opt.hidden, n, opt.work_iters, opt.segment_layers, opt.eager_gap ? 1 : 0);
    std::printf("Config: warmup_iters=%d timed_iters=%d\n\n", opt.warmup_iters, opt.timed_iters);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float* a = nullptr;
    float* b = nullptr;
    int* state = nullptr;
    CUDA_CHECK(cudaMalloc(&a, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&b, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&state, sizeof(int)));
    init_kernel<<<blocks, threads, 0, stream>>>(a, b, state, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double eager_ms = bench_eager(stream, a, b, state, n, blocks, threads, opt);
    std::printf("METRIC eager_ms_per_step %.6f\n", eager_ms);

    GraphExec mono {};
    double mono_instantiate_ms = 0.0;
    std::string mono_error;
    bool mono_ok = capture_graph(stream, a, b, state, n, blocks, threads, opt,
                                 0, opt.layers, true, &mono, &mono_instantiate_ms, &mono_error);
    if (mono_ok) {
        double mono_ms = bench_mono(stream, mono.exec, state, opt);
        std::printf("METRIC monolithic_instantiate_ms %.6f\n", mono_instantiate_ms);
        std::printf("METRIC monolithic_replay_ms_per_step %.6f\n", mono_ms);
        std::printf("METRIC monolithic_speedup_vs_eager %.6f\n", eager_ms / mono_ms);
    } else {
        std::printf("METRIC monolithic_status FAIL %s\n", mono_error.c_str());
    }

    std::vector<GraphExec> segments;
    double segment_instantiate_total_ms = 0.0;
    bool segment_ok = true;
    std::string segment_error;
    for (int start = 0; start < opt.layers; start += opt.segment_layers) {
        int count = opt.segment_layers;
        if (start + count > opt.layers) {
            count = opt.layers - start;
        }
        bool include_final = (start + count == opt.layers);
        GraphExec g {};
        double inst_ms = 0.0;
        std::string err;
        bool ok = capture_graph(stream, a, b, state, n, blocks, threads, opt,
                                start, count, include_final, &g, &inst_ms, &err);
        if (!ok) {
            segment_ok = false;
            segment_error = err;
            break;
        }
        segment_instantiate_total_ms += inst_ms;
        segments.push_back(g);
    }
    if (segment_ok) {
        double segment_ms = bench_segments(stream, segments, a, state, n, blocks, threads, opt);
        std::printf("METRIC segmented_graphs %zu\n", segments.size());
        std::printf("METRIC segmented_instantiate_total_ms %.6f\n", segment_instantiate_total_ms);
        std::printf("METRIC segmented_replay_ms_per_step %.6f\n", segment_ms);
        std::printf("METRIC segmented_speedup_vs_eager %.6f\n", eager_ms / segment_ms);
        if (mono_ok) {
            std::printf("METRIC segmented_over_monolithic_replay %.6f\n", segment_ms / bench_mono(stream, mono.exec, state, opt));
        }
    } else {
        std::printf("METRIC segmented_status FAIL %s\n", segment_error.c_str());
    }

    float sum = checksum(a);
    std::printf("METRIC checksum32 %.8f\n", sum);
    if (!std::isfinite(sum)) {
        std::fprintf(stderr, "non-finite checksum\n");
        return 3;
    }

    if (mono.exec) {
        CUDA_CHECK(cudaGraphExecDestroy(mono.exec));
    }
    for (auto& g : segments) {
        if (g.exec) {
            CUDA_CHECK(cudaGraphExecDestroy(g.exec));
        }
    }
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(state));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (!segment_ok) {
        std::fprintf(stderr, "segmented graph capture failed\n");
        return 4;
    }

    std::printf("VERDICT: CUDA graph segment probe complete\n");
    return 0;
}
