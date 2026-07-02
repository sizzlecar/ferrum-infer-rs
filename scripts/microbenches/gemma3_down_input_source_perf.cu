// Standalone native-CUDA probe for Gemma3-27B GPTQ down_proj input effects.
//
// The tail-MLP chain probe measured down_proj around 68-75us after GeGLU,
// while the isolated dense probe measured the same Marlin shape around 30-55us.
// This diagnostic keeps the Marlin down shape fixed and varies only the input
// source / preceding kernels to distinguish data-shape effects from timing
// artifacts or cache/producer-state effects.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
constexpr int FLUSH_WORDS = 16 * 1024 * 1024;

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

__global__ void flush_l2_kernel(unsigned int* ptr, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        ptr[i] = ptr[i] + 1;
    }
}

static void fill_half(cudaStream_t stream, __half* ptr, float value, int n) {
    const int block = 256;
    fill_half_kernel<<<ceil_div_int(n, block), block, 0, stream>>>(ptr, value, n);
    CHK(cudaGetLastError());
}

static void flush_l2(cudaStream_t stream, unsigned int* ptr) {
    const int block = 256;
    const int grid = 1024;
    flush_l2_kernel<<<grid, block, 0, stream>>>(ptr, FLUSH_WORDS);
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
    size_t workspace_bytes = 0;
};

struct Buffers {
    __half* norm_input = nullptr;
    __half* gate_up_out = nullptr;
    __half* geglu_out = nullptr;
    __half* copied_geglu = nullptr;
    __half* const_input = nullptr;
    __half* small_input = nullptr;
    __half* down_out = nullptr;
    unsigned int* flush_buf = nullptr;
    MarlinBuffers gate_up;
    MarlinBuffers down;
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
    buffers.copied_geglu = static_cast<__half*>(
        cuda_alloc_bytes(static_cast<size_t>(max_m) * INTERMEDIATE * sizeof(__half), 0x00));
    buffers.const_input = static_cast<__half*>(
        cuda_alloc_bytes(static_cast<size_t>(max_m) * INTERMEDIATE * sizeof(__half), 0x00));
    buffers.small_input = static_cast<__half*>(
        cuda_alloc_bytes(static_cast<size_t>(max_m) * INTERMEDIATE * sizeof(__half), 0x00));
    buffers.down_out = static_cast<__half*>(
        cuda_alloc_bytes(static_cast<size_t>(max_m) * HIDDEN * sizeof(__half), 0x00));
    buffers.flush_buf = static_cast<unsigned int*>(
        cuda_alloc_bytes(static_cast<size_t>(FLUSH_WORDS) * sizeof(unsigned int), 0x01));
    buffers.gate_up = alloc_marlin(stream, gate_up_shape);
    buffers.down = alloc_marlin(stream, down_shape);
    fill_half(stream, buffers.norm_input, 0.125f, max_m * HIDDEN);
    fill_half(stream, buffers.const_input, 0.125f, max_m * INTERMEDIATE);
    fill_half(stream, buffers.small_input, 1.0e-4f, max_m * INTERMEDIATE);
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
    CHK(cudaFree(buffers.copied_geglu));
    CHK(cudaFree(buffers.const_input));
    CHK(cudaFree(buffers.small_input));
    CHK(cudaFree(buffers.down_out));
    CHK(cudaFree(buffers.flush_buf));
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

template <typename Prep>
static float time_down_us(
    cudaStream_t stream,
    Buffers& buffers,
    const __half* input,
    int m,
    Prep prep,
    bool sync_after_prep) {
    const Shape down_shape{"down", INTERMEDIATE, HIDDEN};
    cudaEvent_t start;
    cudaEvent_t stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));
    float total_ms = 0.0f;
    for (int iter = 0; iter < WARMUP_ITERS + TIMED_ITERS; ++iter) {
        prep();
        if (sync_after_prep) {
            CHK(cudaStreamSynchronize(stream));
        }
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

static void print_row(const char* mode, int m, float us) {
    const double ops = 2.0 * static_cast<double>(m) * INTERMEDIATE * HIDDEN;
    const double tflops = ops / (static_cast<double>(us) * 1.0e6);
    std::printf("%s,%d,%.3f,%.2f,0\n", mode, m, us, tflops);
}

static void run_one_m(cudaStream_t stream, Buffers& buffers, int m) {
    print_row(
        "const_baseline",
        m,
        time_down_us(
            stream,
            buffers,
            buffers.const_input,
            m,
            [] {},
            false));
    print_row(
        "small_const_baseline",
        m,
        time_down_us(
            stream,
            buffers,
            buffers.small_input,
            m,
            [] {},
            false));
    print_row(
        "const_after_geglu_immediate",
        m,
        time_down_us(
            stream,
            buffers,
            buffers.const_input,
            m,
            [&] {
                prepare_geglu(stream, buffers, m);
            },
            false));
    print_row(
        "const_after_l2_flush",
        m,
        time_down_us(
            stream,
            buffers,
            buffers.const_input,
            m,
            [&] {
                flush_l2(stream, buffers.flush_buf);
            },
            true));
    print_row(
        "geglu_immediate",
        m,
        time_down_us(
            stream,
            buffers,
            buffers.geglu_out,
            m,
            [&] {
                prepare_geglu(stream, buffers, m);
            },
            false));
    print_row(
        "geglu_after_sync",
        m,
        time_down_us(
            stream,
            buffers,
            buffers.geglu_out,
            m,
            [&] {
                prepare_geglu(stream, buffers, m);
            },
            true));
    print_row(
        "copied_geglu_after_sync",
        m,
        time_down_us(
            stream,
            buffers,
            buffers.copied_geglu,
            m,
            [&] {
                prepare_geglu(stream, buffers, m);
                CHK(cudaMemcpyAsync(
                    buffers.copied_geglu,
                    buffers.geglu_out,
                    static_cast<size_t>(m) * INTERMEDIATE * sizeof(__half),
                    cudaMemcpyDeviceToDevice,
                    stream));
            },
            true));
}

int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    std::printf(
        "gemma3_down_input_source_perf device=%s sm=%d.%d h=%d im=%d\n",
        prop.name,
        prop.major,
        prop.minor,
        HIDDEN,
        INTERMEDIATE);
    std::printf("mode,m,down_us,useful_tflops,ret\n");

    const std::vector<int> m_values = {1, 10, 16, 23, 32};
    int max_m = 0;
    for (int m : m_values) {
        max_m = max_m > m ? max_m : m;
    }

    cudaStream_t stream;
    CHK(cudaStreamCreate(&stream));
    Buffers buffers = alloc_buffers(stream, max_m);

    for (int m : m_values) {
        run_one_m(stream, buffers, m);
    }

    free_buffers(buffers);
    CHK(cudaStreamDestroy(stream));
    std::printf("\nVERDICT: gemma3 down input-source native CUDA probe complete\n");
    return 0;
}
