// Graph reproducer v5 — capture on thread A, launch on thread B.
// Matches tokio worker switching between decode steps in our Rust path.

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>

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

struct SharedState {
    CUcontext ctx;
    CUstream stream;
    CUdeviceptr buf;
    int N;
    CUgraphExec graph_exec;
};

static void capture_on_thread_a(SharedState *s) {
    fprintf(stderr, "[v5:A] bind ctx\n");
    CHECK(cuCtxSetCurrent(s->ctx));

    fprintf(stderr, "[v5:A] warm-up kernel\n");
    touch_kernel<<<(s->N + 127) / 128, 128, 0, s->stream>>>((float *)s->buf, s->N);
    CHECK(cuStreamSynchronize(s->stream));

    fprintf(stderr, "[v5:A] begin capture\n");
    CHECK(cuStreamBeginCapture(s->stream, CU_STREAM_CAPTURE_MODE_RELAXED));
    for (int i = 0; i < 4; ++i) {
        touch_kernel<<<(s->N + 127) / 128, 128, 0, s->stream>>>((float *)s->buf, s->N);
    }
    CUgraph graph;
    CHECK(cuStreamEndCapture(s->stream, &graph));
    CHECK(cuGraphInstantiateWithFlags(&s->graph_exec, graph, 0));
    CHECK(cuGraphUpload(s->graph_exec, s->stream));
    CHECK(cuStreamSynchronize(s->stream));
    CHECK(cuGraphDestroy(graph));
    fprintf(stderr, "[v5:A] capture done — ready to launch on thread B\n");
}

static void replay_on_thread_b(SharedState *s) {
    fprintf(stderr, "[v5:B] bind ctx to thread B\n");
    CHECK(cuCtxSetCurrent(s->ctx));

    for (int r = 0; r < 6; ++r) {
        fprintf(stderr, "[v5:B] cuGraphLaunch #%d on thread B\n", r + 1);
        CHECK(cuGraphLaunch(s->graph_exec, s->stream));
        CHECK(cuStreamSynchronize(s->stream));
    }
    fprintf(stderr, "[v5:B] all replays done — cross-thread capture/replay WORKS\n");
}

int main() {
    fprintf(stderr, "[v5] init\n");
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUctxCreateParams params = {};
    CUcontext ctx;
    CHECK(cuCtxCreate_v4(&ctx, &params, 0, dev));

    CUstream stream;
    CHECK(cuCtxSetCurrent(ctx));
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    SharedState s = {};
    s.ctx = ctx;
    s.stream = stream;
    s.N = 4096;
    CHECK(cuMemAlloc(&s.buf, s.N * sizeof(float)));
    CHECK(cuMemsetD8(s.buf, 0, s.N * sizeof(float)));

    // Release the ctx from main thread — worker A will acquire it.
    CHECK(cuCtxPopCurrent(nullptr));

    std::thread ta(capture_on_thread_a, &s);
    ta.join();

    std::thread tb(replay_on_thread_b, &s);
    tb.join();

    CHECK(cuCtxSetCurrent(ctx));
    CHECK(cuGraphExecDestroy(s.graph_exec));
    CHECK(cuMemFree(s.buf));
    CHECK(cuStreamDestroy(stream));
    CHECK(cuCtxDestroy(ctx));
    return 0;
}
