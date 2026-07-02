// Standalone native-CUDA probe for the Gemma3-27B GPTQ tail MLP chain.
//
// This models the product path after attention in unified decode:
//   rms_norm_f32_to_f16 -> Marlin gate_up -> GeGLU -> Marlin down
//   -> rms_norm_f16_add_to_f32
//
// It intentionally does not load a model or run the Rust workspace. The goal is
// to measure the launch chain around the already-profiled gate_up/down Marlin
// kernels with synthetic buffers and product-shaped dimensions.

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

extern "C" __global__ void rms_norm_f32_to_f16(
    const float* input,
    const __half* weight,
    __half* output,
    const int row_size,
    const float eps);

extern "C" __global__ void rms_norm_f16_add_to_f32(
    const __half* input,
    const __half* weight,
    float* residual,
    const int row_size,
    const float eps);

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
constexpr float RMS_EPS = 1.0e-6f;

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

__global__ void fill_float_kernel(float* ptr, float value, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = value;
    }
}

static void fill_half(cudaStream_t stream, __half* ptr, float value, int n) {
    const int block = 256;
    fill_half_kernel<<<ceil_div_int(n, block), block, 0, stream>>>(ptr, value, n);
    CHK(cudaGetLastError());
}

static void fill_float(cudaStream_t stream, float* ptr, float value, int n) {
    const int block = 256;
    fill_float_kernel<<<ceil_div_int(n, block), block, 0, stream>>>(ptr, value, n);
    CHK(cudaGetLastError());
}

struct MarlinShape {
    const char* name;
    int k;
    int n;
};

struct MarlinBuffers {
    void* qweight;
    __half* scales;
    void* workspace;
    size_t workspace_bytes;
};

struct Buffers {
    float* residual_f32;
    __half* post_ln_w;
    __half* post_ffn_w;
    __half* norm_out;
    __half* gate_up_out;
    __half* geglu_out;
    __half* mlp_out;
    MarlinBuffers gate_up;
    MarlinBuffers down;
};

struct PhaseTiming {
    double pre_norm_us = 0.0;
    double gate_up_us = 0.0;
    double geglu_us = 0.0;
    double down_us = 0.0;
    double final_norm_us = 0.0;

    double sum_us() const {
        return pre_norm_us + gate_up_us + geglu_us + down_us + final_norm_us;
    }
};

static MarlinBuffers alloc_marlin(cudaStream_t stream, const MarlinShape& shape) {
    const size_t qweight_bytes =
        round_up(static_cast<size_t>(shape.k) * static_cast<size_t>(shape.n) / 2, 16);
    const size_t scale_count =
        static_cast<size_t>(shape.k / GROUP_SIZE) * static_cast<size_t>(shape.n);
    const size_t scale_bytes = round_up(scale_count * sizeof(__half), 16);
    const size_t workspace_bytes =
        round_up(static_cast<size_t>(shape.n / 128) * MAX_PAR * sizeof(int), 16);

    MarlinBuffers buffers{
        cuda_alloc_bytes(qweight_bytes, 0x11),
        static_cast<__half*>(cuda_alloc_bytes(scale_bytes, 0x00)),
        cuda_alloc_bytes(workspace_bytes, 0x00),
        workspace_bytes,
    };
    fill_half(stream, buffers.scales, 0.001f, static_cast<int>(scale_count));
    return buffers;
}

static Buffers alloc_buffers(cudaStream_t stream, int max_m) {
    const MarlinShape gate_up_shape{"gate_up", HIDDEN, GATE_UP_N};
    const MarlinShape down_shape{"down", INTERMEDIATE, HIDDEN};
    Buffers buffers{
        static_cast<float*>(
            cuda_alloc_bytes(static_cast<size_t>(max_m) * HIDDEN * sizeof(float), 0x00)),
        static_cast<__half*>(cuda_alloc_bytes(HIDDEN * sizeof(__half), 0x00)),
        static_cast<__half*>(cuda_alloc_bytes(HIDDEN * sizeof(__half), 0x00)),
        static_cast<__half*>(
            cuda_alloc_bytes(static_cast<size_t>(max_m) * HIDDEN * sizeof(__half), 0x00)),
        static_cast<__half*>(
            cuda_alloc_bytes(static_cast<size_t>(max_m) * GATE_UP_N * sizeof(__half), 0x00)),
        static_cast<__half*>(
            cuda_alloc_bytes(static_cast<size_t>(max_m) * INTERMEDIATE * sizeof(__half), 0x00)),
        static_cast<__half*>(
            cuda_alloc_bytes(static_cast<size_t>(max_m) * HIDDEN * sizeof(__half), 0x00)),
        alloc_marlin(stream, gate_up_shape),
        alloc_marlin(stream, down_shape),
    };
    fill_float(stream, buffers.residual_f32, 0.125f, max_m * HIDDEN);
    fill_half(stream, buffers.post_ln_w, 1.0f, HIDDEN);
    fill_half(stream, buffers.post_ffn_w, 1.0f, HIDDEN);
    CHK(cudaStreamSynchronize(stream));
    return buffers;
}

static void free_marlin(MarlinBuffers& buffers) {
    CHK(cudaFree(buffers.qweight));
    CHK(cudaFree(buffers.scales));
    CHK(cudaFree(buffers.workspace));
}

static void free_buffers(Buffers& buffers) {
    CHK(cudaFree(buffers.residual_f32));
    CHK(cudaFree(buffers.post_ln_w));
    CHK(cudaFree(buffers.post_ffn_w));
    CHK(cudaFree(buffers.norm_out));
    CHK(cudaFree(buffers.gate_up_out));
    CHK(cudaFree(buffers.geglu_out));
    CHK(cudaFree(buffers.mlp_out));
    free_marlin(buffers.gate_up);
    free_marlin(buffers.down);
}

static void launch_pre_norm(cudaStream_t stream, Buffers& buffers, int m) {
    rms_norm_f32_to_f16<<<m, HIDDEN < 1024 ? HIDDEN : 1024, 0, stream>>>(
        buffers.residual_f32, buffers.post_ln_w, buffers.norm_out, HIDDEN, RMS_EPS);
    CHK(cudaGetLastError());
}

static void launch_final_norm(cudaStream_t stream, Buffers& buffers, int m) {
    rms_norm_f16_add_to_f32<<<m, HIDDEN < 1024 ? HIDDEN : 1024, 0, stream>>>(
        buffers.mlp_out, buffers.post_ffn_w, buffers.residual_f32, HIDDEN, RMS_EPS);
    CHK(cudaGetLastError());
}

static void launch_geglu(cudaStream_t stream, Buffers& buffers, int m) {
    const int total = m * INTERMEDIATE;
    const int block = 256;
    fused_gelu_tanh_mul_interleaved_f16<<<ceil_div_int(total, block), block, 0, stream>>>(
        buffers.gate_up_out, buffers.geglu_out, INTERMEDIATE, total);
    CHK(cudaGetLastError());
}

static void launch_marlin(
    cudaStream_t stream,
    const char* label,
    const void* input,
    MarlinBuffers& weight,
    void* output,
    int m,
    const MarlinShape& shape,
    bool include_workspace_zero) {
    if (include_workspace_zero) {
        CHK(cudaMemsetAsync(weight.workspace, 0, weight.workspace_bytes, stream));
    }
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

static void launch_gate_up(
    cudaStream_t stream,
    Buffers& buffers,
    int m,
    bool include_workspace_zero) {
    const MarlinShape shape{"gate_up", HIDDEN, GATE_UP_N};
    launch_marlin(
        stream,
        "gate_up",
        buffers.norm_out,
        buffers.gate_up,
        buffers.gate_up_out,
        m,
        shape,
        include_workspace_zero);
}

static void launch_down(
    cudaStream_t stream,
    Buffers& buffers,
    int m,
    bool include_workspace_zero) {
    const MarlinShape shape{"down", INTERMEDIATE, HIDDEN};
    launch_marlin(
        stream,
        "down",
        buffers.geglu_out,
        buffers.down,
        buffers.mlp_out,
        m,
        shape,
        include_workspace_zero);
}

static void zero_marlin_workspaces(cudaStream_t stream, Buffers& buffers) {
    CHK(cudaMemsetAsync(
        buffers.gate_up.workspace,
        0,
        buffers.gate_up.workspace_bytes,
        stream));
    CHK(cudaMemsetAsync(buffers.down.workspace, 0, buffers.down.workspace_bytes, stream));
}

static void launch_chain(
    cudaStream_t stream,
    Buffers& buffers,
    int m,
    bool include_workspace_zero) {
    launch_pre_norm(stream, buffers, m);
    launch_gate_up(stream, buffers, m, include_workspace_zero);
    launch_geglu(stream, buffers, m);
    launch_down(stream, buffers, m, include_workspace_zero);
    launch_final_norm(stream, buffers, m);
}

template <typename F>
static float event_time_one_us(cudaStream_t stream, F fn) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));
    CHK(cudaEventRecord(start, stream));
    fn();
    CHK(cudaEventRecord(stop, stream));
    CHK(cudaEventSynchronize(stop));
    CHK(cudaGetLastError());
    float ms = 0.0f;
    CHK(cudaEventElapsedTime(&ms, start, stop));
    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
    return ms * 1000.0f;
}

static PhaseTiming time_phase_breakdown(
    cudaStream_t stream,
    Buffers& buffers,
    int m,
    bool include_workspace_zero) {
    PhaseTiming total;
    const int total_iters = WARMUP_ITERS + TIMED_ITERS;
    for (int iter = 0; iter < total_iters; ++iter) {
        if (iter < WARMUP_ITERS) {
            if (!include_workspace_zero) {
                zero_marlin_workspaces(stream, buffers);
            }
            launch_chain(stream, buffers, m, include_workspace_zero);
            CHK(cudaStreamSynchronize(stream));
            continue;
        }
        total.pre_norm_us += event_time_one_us(stream, [&] {
            launch_pre_norm(stream, buffers, m);
        });
        if (!include_workspace_zero) {
            CHK(cudaMemsetAsync(
                buffers.gate_up.workspace,
                0,
                buffers.gate_up.workspace_bytes,
                stream));
            CHK(cudaStreamSynchronize(stream));
        }
        total.gate_up_us += event_time_one_us(stream, [&] {
            launch_gate_up(stream, buffers, m, include_workspace_zero);
        });
        total.geglu_us += event_time_one_us(stream, [&] {
            launch_geglu(stream, buffers, m);
        });
        if (!include_workspace_zero) {
            CHK(cudaMemsetAsync(
                buffers.down.workspace,
                0,
                buffers.down.workspace_bytes,
                stream));
            CHK(cudaStreamSynchronize(stream));
        }
        total.down_us += event_time_one_us(stream, [&] {
            launch_down(stream, buffers, m, include_workspace_zero);
        });
        total.final_norm_us += event_time_one_us(stream, [&] {
            launch_final_norm(stream, buffers, m);
        });
    }
    total.pre_norm_us /= TIMED_ITERS;
    total.gate_up_us /= TIMED_ITERS;
    total.geglu_us /= TIMED_ITERS;
    total.down_us /= TIMED_ITERS;
    total.final_norm_us /= TIMED_ITERS;
    return total;
}

static double time_chain_event_us(
    cudaStream_t stream,
    Buffers& buffers,
    int m,
    bool include_workspace_zero) {
    double total = 0.0;
    const int total_iters = WARMUP_ITERS + TIMED_ITERS;
    for (int iter = 0; iter < total_iters; ++iter) {
        if (!include_workspace_zero) {
            zero_marlin_workspaces(stream, buffers);
            CHK(cudaStreamSynchronize(stream));
        }
        const float us = event_time_one_us(stream, [&] {
            launch_chain(stream, buffers, m, include_workspace_zero);
        });
        if (iter >= WARMUP_ITERS) {
            total += us;
        }
    }
    return total / TIMED_ITERS;
}

static double time_chain_host_sync_us(
    cudaStream_t stream,
    Buffers& buffers,
    int m,
    bool include_workspace_zero) {
    double total = 0.0;
    const int total_iters = WARMUP_ITERS + TIMED_ITERS;
    for (int iter = 0; iter < total_iters; ++iter) {
        CHK(cudaStreamSynchronize(stream));
        if (!include_workspace_zero) {
            zero_marlin_workspaces(stream, buffers);
            CHK(cudaStreamSynchronize(stream));
        }
        const auto start = std::chrono::steady_clock::now();
        launch_chain(stream, buffers, m, include_workspace_zero);
        CHK(cudaStreamSynchronize(stream));
        const auto stop = std::chrono::steady_clock::now();
        if (iter >= WARMUP_ITERS) {
            total += std::chrono::duration<double, std::micro>(stop - start).count();
        }
    }
    return total / TIMED_ITERS;
}

static void print_row(
    const char* mode,
    int m,
    const PhaseTiming& phases,
    double chain_event_us,
    double chain_host_sync_us) {
    std::printf(
        "%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,0\n",
        mode,
        m,
        phases.pre_norm_us,
        phases.gate_up_us,
        phases.geglu_us,
        phases.down_us,
        phases.final_norm_us,
        phases.sum_us(),
        chain_event_us,
        chain_host_sync_us);
}

static void run_one_mode(
    cudaStream_t stream,
    Buffers& buffers,
    int m,
    const char* mode,
    bool include_workspace_zero) {
    fill_float(stream, buffers.residual_f32, 0.125f, m * HIDDEN);
    CHK(cudaStreamSynchronize(stream));
    const PhaseTiming phases = time_phase_breakdown(stream, buffers, m, include_workspace_zero);
    const double chain_event_us = time_chain_event_us(stream, buffers, m, include_workspace_zero);
    const double chain_host_sync_us =
        time_chain_host_sync_us(stream, buffers, m, include_workspace_zero);
    print_row(mode, m, phases, chain_event_us, chain_host_sync_us);
}

int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    std::printf(
        "gemma3_tail_mlp_chain_perf device=%s sm=%d.%d h=%d im=%d\n",
        prop.name,
        prop.major,
        prop.minor,
        HIDDEN,
        INTERMEDIATE);
    std::printf(
        "mode,m,pre_norm_us,gate_up_us,geglu_us,down_us,final_norm_us,phase_sum_us,"
        "chain_event_us,chain_host_sync_us,ret\n");

    const std::vector<int> m_values = {1, 10, 16, 23, 32};
    int max_m = 0;
    for (int m : m_values) {
        max_m = max_m > m ? max_m : m;
    }

    cudaStream_t stream;
    CHK(cudaStreamCreate(&stream));
    Buffers buffers = alloc_buffers(stream, max_m);

    for (int m : m_values) {
        run_one_mode(stream, buffers, m, "product_ws_zero", true);
        run_one_mode(stream, buffers, m, "kernel_only_ws_prezero_diagnostic", false);
    }

    free_buffers(buffers);
    CHK(cudaStreamDestroy(stream));
    std::printf("\nVERDICT: gemma3 tail MLP chain native CUDA probe complete\n");
    return 0;
}
