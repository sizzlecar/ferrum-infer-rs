// Standalone native-CUDA probe for Gemma3-style device F32 residual shadow
// graph replay.
//
// Goal: validate the CUDA-level part of a possible Ferrum Gemma3 batched
// decode graph path without loading a model or rebuilding the Rust workspace.
// The product path currently keeps batched CUDA graph disabled when the
// sandwich-norm device F32 residual shadow is active. This probe answers a
// narrower question: can a 62-layer, persistent-buffer, shadow-update decode
// step be captured and replayed stably, and what launch-overhead headroom does
// graph replay have versus eager launches?
//
// Build:
//   nvcc -O3 -arch=sm_89 -std=c++17 \
//     scripts/microbenches/gemma3_shadow_graph_bench.cu \
//     -lcuda -lcudart -o /tmp/gemma3_shadow_graph_bench

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

constexpr int kLayers = 62;
constexpr int kBatch = 16;
constexpr int kHidden = 5376;
constexpr int kElements = kBatch * kHidden;
constexpr int kThreads = 256;
constexpr int kBlocks = (kElements + kThreads - 1) / kThreads;
constexpr int kWarmupIters = 8;
constexpr int kTimedIters = 120;
constexpr int kInnerWork = 4;

__device__ __forceinline__ float mix_value(float x, float scale, int state, int salt) {
    float y = x * scale + static_cast<float>((state + salt) & 7) * 0.001f;
#pragma unroll
    for (int i = 0; i < kInnerWork; ++i) {
        y = y * 1.00017f + 0.00031f * static_cast<float>((salt + i) & 15);
    }
    return y;
}

__global__ void init_buffers(__half* residual, float* shadow, float* branch, int* state, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = static_cast<float>((i % 257) - 128) * 0.0005f;
        residual[i] = __float2half(v);
        shadow[i] = v;
        branch[i] = 0.0f;
    }
    if (i == 0) {
        state[0] = 1;
    }
}

__global__ void activation_to_shadow(const __half* residual, float* shadow, const int* state, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        shadow[i] = __half2float(residual[i]) + static_cast<float>(state[0] & 3) * 0.0001f;
    }
}

__global__ void fake_norm(float* shadow, const int* state, int n, int salt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = shadow[i];
        shadow[i] = mix_value(x, 0.997f, state[0], salt);
    }
}

__global__ void fake_projection(const float* shadow, float* branch, const int* state, int n, int salt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        branch[i] = mix_value(shadow[i], 0.03125f * static_cast<float>((salt % 5) + 1), state[0], salt);
    }
}

__global__ void fake_activation(float* branch, const int* state, int n, int salt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = branch[i];
        branch[i] = x / (1.0f + fabsf(x)) + static_cast<float>((state[0] + salt) & 1) * 0.0002f;
    }
}

__global__ void add_branch_to_shadow(float* shadow, const float* branch, const int* state, int n, int salt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        shadow[i] += branch[i] * (0.125f + static_cast<float>((state[0] + salt) & 3) * 0.005f);
    }
}

__global__ void shadow_to_activation(const float* shadow, __half* residual, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        residual[i] = __float2half(shadow[i]);
    }
}

void launch_shadow_step(cudaStream_t stream, __half* residual, float* shadow, float* branch, int* state) {
    activation_to_shadow<<<kBlocks, kThreads, 0, stream>>>(residual, shadow, state, kElements);
    for (int layer = 0; layer < kLayers; ++layer) {
        fake_norm<<<kBlocks, kThreads, 0, stream>>>(shadow, state, kElements, layer * 11 + 1);
        fake_projection<<<kBlocks, kThreads, 0, stream>>>(shadow, branch, state, kElements, layer * 11 + 2);
        add_branch_to_shadow<<<kBlocks, kThreads, 0, stream>>>(shadow, branch, state, kElements, layer * 11 + 3);
        fake_norm<<<kBlocks, kThreads, 0, stream>>>(shadow, state, kElements, layer * 11 + 4);
        fake_projection<<<kBlocks, kThreads, 0, stream>>>(shadow, branch, state, kElements, layer * 11 + 5);
        fake_activation<<<kBlocks, kThreads, 0, stream>>>(branch, state, kElements, layer * 11 + 6);
        fake_projection<<<kBlocks, kThreads, 0, stream>>>(branch, branch, state, kElements, layer * 11 + 7);
        add_branch_to_shadow<<<kBlocks, kThreads, 0, stream>>>(shadow, branch, state, kElements, layer * 11 + 8);
    }
    shadow_to_activation<<<kBlocks, kThreads, 0, stream>>>(shadow, residual, kElements);
}

void upload_state(cudaStream_t stream, int* state, int value, bool pre_sync) {
    CUDA_CHECK(cudaMemcpyAsync(state, &value, sizeof(int), cudaMemcpyHostToDevice, stream));
    if (pre_sync) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double bench_eager(
    cudaStream_t stream,
    __half* residual,
    float* shadow,
    float* branch,
    int* state,
    bool pre_sync) {
    for (int i = 0; i < kWarmupIters; ++i) {
        upload_state(stream, state, i + 10, pre_sync);
        launch_shadow_step(stream, residual, shadow, branch, state);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kTimedIters; ++i) {
        upload_state(stream, state, i + 100, pre_sync);
        launch_shadow_step(stream, residual, shadow, branch, state);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto t1 = std::chrono::steady_clock::now();
    return elapsed_ms(t0, t1) / static_cast<double>(kTimedIters);
}

cudaGraphExec_t capture_shadow_graph(
    cudaStream_t stream,
    __half* residual,
    float* shadow,
    float* branch,
    int* state) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_shadow_step(stream, residual, shadow, branch, state);
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphUpload(exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return exec;
}

double bench_graph(
    cudaStream_t stream,
    cudaGraphExec_t exec,
    int* state,
    bool pre_sync) {
    for (int i = 0; i < kWarmupIters; ++i) {
        upload_state(stream, state, i + 1000, pre_sync);
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kTimedIters; ++i) {
        upload_state(stream, state, i + 2000, pre_sync);
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto t1 = std::chrono::steady_clock::now();
    return elapsed_ms(t0, t1) / static_cast<double>(kTimedIters);
}

float sample_checksum(float* shadow) {
    float host[16] = {};
    CUDA_CHECK(cudaMemcpy(host, shadow, sizeof(host), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (float v : host) {
        sum += v;
    }
    return sum;
}

} // namespace

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s\n", prop.name);
    std::printf("Pattern: Gemma3-style shadow decode, layers=%d, batch=%d, hidden=%d, kernels/step=%d\n",
                kLayers, kBatch, kHidden, 2 + kLayers * 8);
    std::printf("Timed iterations: %d, warmup iterations: %d\n\n", kTimedIters, kWarmupIters);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    __half* residual = nullptr;
    float* shadow = nullptr;
    float* branch = nullptr;
    int* state = nullptr;
    CUDA_CHECK(cudaMalloc(&residual, sizeof(__half) * kElements));
    CUDA_CHECK(cudaMalloc(&shadow, sizeof(float) * kElements));
    CUDA_CHECK(cudaMalloc(&branch, sizeof(float) * kElements));
    CUDA_CHECK(cudaMalloc(&state, sizeof(int)));

    init_buffers<<<kBlocks, kThreads, 0, stream>>>(residual, shadow, branch, state, kElements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double eager_ordered_ms = bench_eager(stream, residual, shadow, branch, state, false);
    double eager_presync_ms = bench_eager(stream, residual, shadow, branch, state, true);

    upload_state(stream, state, 777, true);
    cudaGraphExec_t exec = capture_shadow_graph(stream, residual, shadow, branch, state);

    double graph_ordered_ms = bench_graph(stream, exec, state, false);
    double graph_presync_ms = bench_graph(stream, exec, state, true);
    float checksum = sample_checksum(shadow);

    std::printf("Results (host wall time, includes state upload and final stream sync):\n");
    std::printf("  eager ordered state upload : %.3f ms/step\n", eager_ordered_ms);
    std::printf("  eager pre-sync state upload: %.3f ms/step\n", eager_presync_ms);
    std::printf("  graph ordered state upload : %.3f ms/step (speedup %.2fx vs eager ordered)\n",
                graph_ordered_ms, eager_ordered_ms / graph_ordered_ms);
    std::printf("  graph pre-sync state upload: %.3f ms/step (speedup %.2fx vs eager pre-sync)\n",
                graph_presync_ms, eager_presync_ms / graph_presync_ms);
    std::printf("  checksum16: %.8f\n\n", checksum);

    if (!std::isfinite(checksum)) {
        std::fprintf(stderr, "non-finite checksum\n");
        return 2;
    }
    if (graph_ordered_ms <= 0.0 || graph_presync_ms <= 0.0) {
        std::fprintf(stderr, "invalid timing\n");
        return 3;
    }

    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaFree(residual));
    CUDA_CHECK(cudaFree(shadow));
    CUDA_CHECK(cudaFree(branch));
    CUDA_CHECK(cudaFree(state));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::printf("VERDICT: Gemma3 shadow graph native CUDA probe complete\n");
    return 0;
}
