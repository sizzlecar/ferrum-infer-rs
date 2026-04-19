// Graph reproducer v3 — use cuCtxCreate_v4 instead of runtime primary ctx.
// This matches cudarc's `CudaContext::new` which creates a fresh ctx.
//
// If v3 SEGFAULTs and v1/v2 pass, the bug is specific to the
// non-primary-context + graph combination on Blackwell + CUDA 13.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

__global__ void touch_kernel(float *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        buf[i] = buf[i] + 1.0f;
    }
}

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

int main() {
    fprintf(stderr, "[v3] init\n");
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));

    // Match cudarc: explicit CUcontext via cuCtxCreate_v4.
    CUctxCreateParams params = {};
    CUcontext ctx;
    CHECK(cuCtxCreate_v4(&ctx, &params, 0, dev));
    CHECK(cuCtxSetCurrent(ctx));

    CUstream stream;
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    // Stream-ordered alloc via default pool.
    CUmemoryPool default_pool;
    CHECK(cuDeviceGetDefaultMemPool(&default_pool, dev));

    const int N = 4096;
    CUdeviceptr buf;
    CHECK(cuMemAllocFromPoolAsync(&buf, N * sizeof(float), default_pool, stream));
    CHECK(cuStreamSynchronize(stream));

    // Warm-up outside capture.
    touch_kernel<<<(N + 127) / 128, 128, 0, stream>>>((float *)buf, N);
    CHECK(cuStreamSynchronize(stream));

    fprintf(stderr, "[v3] begin capture\n");
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));
    for (int i = 0; i < 4; ++i) {
        touch_kernel<<<(N + 127) / 128, 128, 0, stream>>>((float *)buf, N);
    }
    CUgraph graph;
    CHECK(cuStreamEndCapture(stream, &graph));
    CUgraphExec graph_exec;
    CHECK(cuGraphInstantiateWithFlags(&graph_exec, graph, 0));
    CHECK(cuGraphUpload(graph_exec, stream));
    CHECK(cuStreamSynchronize(stream));

    for (int r = 0; r < 6; ++r) {
        fprintf(stderr, "[v3] cuGraphLaunch #%d\n", r + 1);
        CHECK(cuGraphLaunch(graph_exec, stream));
        CHECK(cuStreamSynchronize(stream));
    }
    fprintf(stderr, "[v3] done — graph + non-primary ctx WORKS\n");

    CHECK(cuGraphExecDestroy(graph_exec));
    CHECK(cuGraphDestroy(graph));
    CHECK(cuMemFreeAsync(buf, stream));
    CHECK(cuStreamSynchronize(stream));
    CHECK(cuStreamDestroy(stream));
    CHECK(cuCtxDestroy(ctx));
    return 0;
}
