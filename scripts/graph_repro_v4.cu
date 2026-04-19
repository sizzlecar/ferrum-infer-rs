// Graph reproducer v4 — add cuBLAS GEMM inside capture. Our Rust uses
// cuBLAS heavily for per-layer projections, and cuBLAS has its own
// internal stream-ordered state (workspace pointer set via
// cublasSetWorkspace) that might be what graph capture chokes on.
//
// v1/v2/v3: all bare CUDA ops — graph works fine.
// v4 adds cuBLAS → if this segfaults, cuBLAS + graph is the trigger.

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call)                                                            \
    do {                                                                       \
        CUresult err = (call);                                                 \
        if (err != CUDA_SUCCESS) {                                             \
            const char *name = nullptr;                                        \
            cuGetErrorName(err, &name);                                        \
            fprintf(stderr, "driver call failed at %s:%d: %s (%d)\n",          \
                    __FILE__, __LINE__, name ? name : "?", (int)err);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t err = (call);                                           \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS call failed at %s:%d: %d\n",               \
                    __FILE__, __LINE__, (int)err);                             \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

int main() {
    fprintf(stderr, "[v4] init\n");
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUctxCreateParams params = {};
    CUcontext ctx;
    CHECK(cuCtxCreate_v4(&ctx, &params, 0, dev));
    CHECK(cuCtxSetCurrent(ctx));

    CUstream stream;
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    cublasHandle_t blas;
    CHECK_CUBLAS(cublasCreate(&blas));
    CHECK_CUBLAS(cublasSetStream(blas, (cudaStream_t)stream));
    // Match our Rust: DEVICE pointer mode + explicit workspace.
    CHECK_CUBLAS(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE));

    // Allocate workspace like our code does: 32 MB is the typical cuBLAS L2
    // workspace recommendation on Blackwell.
    void *workspace;
    const size_t ws_size = 32 * 1024 * 1024;
    CHECK(cuMemAlloc((CUdeviceptr *)&workspace, ws_size));
    CHECK_CUBLAS(cublasSetWorkspace(blas, workspace, ws_size));

    // Allocate A, B, C for a GEMM: C[M,N] = A[M,K] @ B[K,N].
    const int M = 1, N = 1024, K = 1024;
    float *d_a, *d_b, *d_c;
    CHECK(cuMemAlloc((CUdeviceptr *)&d_a, M * K * sizeof(float)));
    CHECK(cuMemAlloc((CUdeviceptr *)&d_b, K * N * sizeof(float)));
    CHECK(cuMemAlloc((CUdeviceptr *)&d_c, M * N * sizeof(float)));

    // Seed scratch with 1.0s.
    std::vector<float> ones_a(M * K, 1.0f), ones_b(K * N, 0.5f);
    CHECK(cuMemcpyHtoD((CUdeviceptr)d_a, ones_a.data(), ones_a.size() * sizeof(float)));
    CHECK(cuMemcpyHtoD((CUdeviceptr)d_b, ones_b.data(), ones_b.size() * sizeof(float)));

    // DEVICE pointer mode: alpha/beta must live on device.
    float *d_alpha, *d_beta;
    float h_alpha = 1.0f, h_beta = 0.0f;
    CHECK(cuMemAlloc((CUdeviceptr *)&d_alpha, sizeof(float)));
    CHECK(cuMemAlloc((CUdeviceptr *)&d_beta, sizeof(float)));
    CHECK(cuMemcpyHtoD((CUdeviceptr)d_alpha, &h_alpha, sizeof(float)));
    CHECK(cuMemcpyHtoD((CUdeviceptr)d_beta, &h_beta, sizeof(float)));

    // Warm up outside capture.
    fprintf(stderr, "[v4] warm-up GEMM\n");
    CHECK_CUBLAS(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             d_alpha, d_b, N, d_a, K, d_beta, d_c, N));
    CHECK(cuStreamSynchronize(stream));

    fprintf(stderr, "[v4] begin capture\n");
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));
    for (int i = 0; i < 3; ++i) {
        CHECK_CUBLAS(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 d_alpha, d_b, N, d_a, K, d_beta, d_c, N));
    }
    CUgraph graph;
    CHECK(cuStreamEndCapture(stream, &graph));
    CUgraphExec graph_exec;
    fprintf(stderr, "[v4] instantiate\n");
    CHECK(cuGraphInstantiateWithFlags(&graph_exec, graph, 0));
    CHECK(cuGraphUpload(graph_exec, stream));
    CHECK(cuStreamSynchronize(stream));

    for (int r = 0; r < 6; ++r) {
        fprintf(stderr, "[v4] cuGraphLaunch #%d (with cuBLAS work)\n", r + 1);
        CHECK(cuGraphLaunch(graph_exec, stream));
        CHECK(cuStreamSynchronize(stream));
    }
    fprintf(stderr, "[v4] done — cuBLAS + graph WORKS\n");

    CHECK(cuGraphExecDestroy(graph_exec));
    CHECK(cuGraphDestroy(graph));
    CHECK(cuMemFree((CUdeviceptr)d_a));
    CHECK(cuMemFree((CUdeviceptr)d_b));
    CHECK(cuMemFree((CUdeviceptr)d_c));
    CHECK(cuMemFree((CUdeviceptr)d_alpha));
    CHECK(cuMemFree((CUdeviceptr)d_beta));
    CHECK(cuMemFree((CUdeviceptr)workspace));
    CHECK_CUBLAS(cublasDestroy(blas));
    CHECK(cuStreamDestroy(stream));
    CHECK(cuCtxDestroy(ctx));
    return 0;
}
