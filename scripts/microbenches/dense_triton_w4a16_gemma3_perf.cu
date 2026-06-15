// Standalone native-CUDA probe comparing dense Gemma3 GPTQ Marlin vs the
// existing Triton-rs W4A16 GPTQ PTX at tail-MLP projection shapes.
//
// This is diagnostic only. It does not change the product path and does not
// require loading a model. The intent is to decide whether the existing
// Triton W4A16 backend is a credible dense MLP compute lever before wiring any
// typed product option.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
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

#define CHK_CU(stmt)                                                             \
    do {                                                                         \
        CUresult err__ = (stmt);                                                 \
        if (err__ != CUDA_SUCCESS) {                                             \
            const char* msg__ = nullptr;                                         \
            cuGetErrorString(err__, &msg__);                                     \
            std::fprintf(stderr, "CUDA driver %s:%d: %s\n", __FILE__, __LINE__, \
                         msg__ ? msg__ : "unknown");                           \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

struct Shape {
    const char* name;
    int k;
    int n;
};

constexpr int GROUP_SIZE = 128;
constexpr int MAX_PAR = 16;
constexpr int WARMUP_ITERS = 5;
constexpr int TIMED_ITERS = 80;
constexpr int TRITON_BM = 64;
constexpr int TRITON_BN = 64;
constexpr int TRITON_BLOCK_THREADS = 128;
constexpr unsigned TRITON_SHARED_MEM = 8192;
constexpr const char* TRITON_FN = "w4a16_gptq_gemm_typed";

static size_t round_up(size_t value, size_t align) {
    return ((value + align - 1) / align) * align;
}

static int ceil_div_int(int value, int divisor) {
    return (value + divisor - 1) / divisor;
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

static std::string read_file(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "failed to open PTX path: %s\n", path);
        std::exit(1);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

struct TritonKernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;

    explicit TritonKernel(const char* ptx_path) {
        std::string ptx = read_file(ptx_path);
        CHK_CU(cuInit(0));
        CHK_CU(cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr));
        CHK_CU(cuModuleGetFunction(&function, module, TRITON_FN));
    }

    ~TritonKernel() {
        if (module != nullptr) {
            cuModuleUnload(module);
        }
    }
};

struct MarlinBuffers {
    void* qweight = nullptr;
    __half* scales = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
};

struct TritonBuffers {
    void* qweight = nullptr;
    __half* scales = nullptr;
    void* qzeros = nullptr;
    void* scratch = nullptr;
    void* profile = nullptr;
};

struct ShapeBuffers {
    __half* input = nullptr;
    __half* output = nullptr;
    MarlinBuffers marlin;
    TritonBuffers triton;
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

static TritonBuffers alloc_triton(cudaStream_t stream, const Shape& shape) {
    const size_t qweight_bytes =
        round_up(static_cast<size_t>(shape.k / 8) * static_cast<size_t>(shape.n) * sizeof(int), 16);
    const size_t scale_count =
        static_cast<size_t>(shape.k / GROUP_SIZE) * static_cast<size_t>(shape.n);
    const size_t scale_bytes = round_up(scale_count * sizeof(__half), 16);
    const size_t qzeros_bytes = round_up(
        static_cast<size_t>(shape.k / GROUP_SIZE) * static_cast<size_t>(shape.n / 8)
            * sizeof(int),
        16);
    TritonBuffers buffers;
    buffers.qweight = cuda_alloc_bytes(qweight_bytes, 0x33);
    buffers.scales = static_cast<__half*>(cuda_alloc_bytes(scale_bytes, 0x00));
    buffers.qzeros = cuda_alloc_bytes(qzeros_bytes, 0x77);
    buffers.scratch = cuda_alloc_bytes(1, 0x00);
    buffers.profile = cuda_alloc_bytes(1, 0x00);
    fill_half(stream, buffers.scales, 0.001f, static_cast<int>(scale_count));
    return buffers;
}

static ShapeBuffers alloc_shape(cudaStream_t stream, const Shape& shape, int max_m) {
    ShapeBuffers buffers;
    buffers.input = static_cast<__half*>(
        cuda_alloc_bytes(static_cast<size_t>(max_m) * shape.k * sizeof(__half), 0x3c));
    buffers.output = static_cast<__half*>(
        cuda_alloc_bytes(static_cast<size_t>(max_m) * shape.n * sizeof(__half), 0x00));
    buffers.marlin = alloc_marlin(stream, shape);
    buffers.triton = alloc_triton(stream, shape);
    CHK(cudaStreamSynchronize(stream));
    return buffers;
}

static void free_marlin(MarlinBuffers& buffers) {
    CHK(cudaFree(buffers.qweight));
    CHK(cudaFree(buffers.scales));
    CHK(cudaFree(buffers.workspace));
}

static void free_triton(TritonBuffers& buffers) {
    CHK(cudaFree(buffers.qweight));
    CHK(cudaFree(buffers.scales));
    CHK(cudaFree(buffers.qzeros));
    CHK(cudaFree(buffers.scratch));
    CHK(cudaFree(buffers.profile));
}

static void free_shape(ShapeBuffers& buffers) {
    CHK(cudaFree(buffers.input));
    CHK(cudaFree(buffers.output));
    free_marlin(buffers.marlin);
    free_triton(buffers.triton);
}

static void launch_marlin(
    cudaStream_t stream,
    ShapeBuffers& buffers,
    const Shape& shape,
    int m,
    bool include_workspace_zero) {
    if (include_workspace_zero) {
        CHK(cudaMemsetAsync(
            buffers.marlin.workspace,
            0,
            buffers.marlin.workspace_bytes,
            stream));
    }
    int ret = marlin_cuda(
        buffers.input,
        buffers.marlin.qweight,
        buffers.output,
        buffers.marlin.scales,
        m,
        shape.n,
        shape.k,
        buffers.marlin.workspace,
        GROUP_SIZE,
        0,
        stream,
        -1,
        -1,
        -1,
        MAX_PAR,
        -1);
    if (ret != 0) {
        std::fprintf(stderr, "marlin_cuda failed shape=%s m=%d ret=%d\n", shape.name, m, ret);
        std::exit(2);
    }
}

static void launch_triton(
    cudaStream_t stream,
    TritonKernel& kernel,
    ShapeBuffers& buffers,
    const Shape& shape,
    int m) {
    const int n = shape.n;
    const int k = shape.k;
    const int group_size = GROUP_SIZE;
    const int stride_am = k;
    const int stride_ak = 1;
    const int stride_qwk = n;
    const int stride_qwn = 1;
    const int stride_sk = n;
    const int stride_sn = 1;
    const int stride_qzk = n / 8;
    const int stride_qzn = 1;
    const int stride_cm = n;
    const int stride_cn = 1;
    void* args[] = {
        &buffers.input,
        &buffers.triton.qweight,
        &buffers.triton.scales,
        &buffers.triton.qzeros,
        &buffers.output,
        const_cast<int*>(&m),
        const_cast<int*>(&n),
        const_cast<int*>(&k),
        const_cast<int*>(&group_size),
        const_cast<int*>(&stride_am),
        const_cast<int*>(&stride_ak),
        const_cast<int*>(&stride_qwk),
        const_cast<int*>(&stride_qwn),
        const_cast<int*>(&stride_sk),
        const_cast<int*>(&stride_sn),
        const_cast<int*>(&stride_qzk),
        const_cast<int*>(&stride_qzn),
        const_cast<int*>(&stride_cm),
        const_cast<int*>(&stride_cn),
        &buffers.triton.scratch,
        &buffers.triton.profile,
    };
    CHK_CU(cuLaunchKernel(
        kernel.function,
        ceil_div_int(m, TRITON_BM),
        ceil_div_int(n, TRITON_BN),
        1,
        TRITON_BLOCK_THREADS,
        1,
        1,
        TRITON_SHARED_MEM,
        reinterpret_cast<CUstream>(stream),
        args,
        nullptr));
}

template <typename F>
static float time_launch(cudaStream_t stream, F fn) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));
    float total_ms = 0.0f;
    for (int iter = 0; iter < WARMUP_ITERS + TIMED_ITERS; ++iter) {
        CHK(cudaEventRecord(start, stream));
        fn();
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

template <typename Prep, typename F>
static float time_launch_with_prep(cudaStream_t stream, Prep prep, F fn) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));
    float total_ms = 0.0f;
    for (int iter = 0; iter < WARMUP_ITERS + TIMED_ITERS; ++iter) {
        prep();
        CHK(cudaStreamSynchronize(stream));
        CHK(cudaEventRecord(start, stream));
        fn();
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

static void print_row(const char* backend, const Shape& shape, int m, float us) {
    const double ops = 2.0 * static_cast<double>(m) * shape.k * shape.n;
    const double tflops = ops / (static_cast<double>(us) * 1.0e6);
    std::printf("%s,%s,%d,%.3f,%.2f,0\n", backend, shape.name, m, us, tflops);
}

static void run_shape(cudaStream_t stream, TritonKernel& kernel, const Shape& shape) {
    const std::vector<int> m_values = {1, 10, 16, 23, 32};
    int max_m = 0;
    for (int m : m_values) {
        max_m = max_m > m ? max_m : m;
    }
    ShapeBuffers buffers = alloc_shape(stream, shape, max_m);
    for (int m : m_values) {
        float marlin_us = time_launch(stream, [&] {
            launch_marlin(stream, buffers, shape, m, true);
        });
        print_row("marlin_product_ws_zero", shape, m, marlin_us);
        float marlin_kernel_us = time_launch_with_prep(
            stream,
            [&] {
                CHK(cudaMemsetAsync(
                    buffers.marlin.workspace,
                    0,
                    buffers.marlin.workspace_bytes,
                    stream));
            },
            [&] {
                launch_marlin(stream, buffers, shape, m, false);
            });
        print_row("marlin_kernel_only_ws_prezero_diagnostic", shape, m, marlin_kernel_us);
        float triton_us = time_launch(stream, [&] {
            launch_triton(stream, kernel, buffers, shape, m);
        });
        print_row("triton_w4a16", shape, m, triton_us);
    }
    free_shape(buffers);
}

int main(int argc, char** argv) {
    const char* ptx_path = argc > 1 ? argv[1] : "crates/ferrum-kernels/triton_ptx/w4a16_gptq_f16.ptx";
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    std::printf(
        "dense_triton_w4a16_gemma3_perf device=%s sm=%d.%d ptx=%s\n",
        prop.name,
        prop.major,
        prop.minor,
        ptx_path);
    std::printf("backend,shape,m,us,useful_tflops,ret\n");
    cudaStream_t stream;
    CHK(cudaStreamCreate(&stream));
    TritonKernel kernel(ptx_path);
    const Shape shapes[] = {
        {"gate_up", 5376, 43008},
        {"down", 21504, 5376},
    };
    for (const Shape& shape : shapes) {
        run_shape(stream, kernel, shape);
    }
    CHK(cudaStreamDestroy(stream));
    std::printf("\nVERDICT: dense Triton W4A16 Gemma3 native CUDA probe complete\n");
    return 0;
}
