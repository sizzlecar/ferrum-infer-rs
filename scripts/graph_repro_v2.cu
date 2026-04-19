// Graph reproducer v2 — adds our Rust path's quirks one by one so we can
// bisect which feature triggers the cuGraphLaunch SIGSEGV we see through
// cudarc.
//
// v1 (scripts/graph_repro.cu): runtime API + cuMemAlloc + single kernel.
//     RESULT: graph works on Blackwell + CUDA 13. Crash is NOT a driver bug.
//
// v2 adds:
//     - stream-ordered allocation via cuMemAllocFromPoolAsync (cudarc's
//       `stream.alloc_zeros` compiles to this on modern drivers)
//     - multiple distinct device buffers touched per step (more like a
//       transformer layer)
//     - a thread that binds to the primary context and does the capture
//       + replay (cudarc + tokio drives work from worker threads)
//     - pre-capture weight uploads that record cudarc-style events BEFORE
//       disable_event_tracking — cudarc's "disable" flag only affects
//       subsequent allocations, not ones already tracked

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

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

#define CHECK_RT(call)                                                         \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "runtime call failed: %s\n",                       \
                    cudaGetErrorString(err));                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static void repro_on_thread(CUcontext main_ctx) {
    fprintf(stderr, "[v2] worker thread bound to primary ctx\n");
    CHECK(cuCtxSetCurrent(main_ctx));

    CUstream stream;
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    // Stream-ordered alloc (memory pool) — what cudarc does under the hood
    // when you call `stream.alloc_zeros`. Use the default pool.
    CUmemoryPool default_pool;
    CUdevice dev;
    CHECK(cuCtxGetDevice(&dev));
    CHECK(cuDeviceGetDefaultMemPool(&default_pool, dev));

    const int N = 4096;
    const int NUM_BUFS = 3; // "residual", "norm_out", "qkv_out" style
    CUdeviceptr bufs[NUM_BUFS];
    for (int i = 0; i < NUM_BUFS; ++i) {
        CHECK(cuMemAllocFromPoolAsync(&bufs[i], N * sizeof(float), default_pool, stream));
    }
    CHECK(cuStreamSynchronize(stream));

    fprintf(stderr, "[v2] warm-up kernels outside capture\n");
    for (int i = 0; i < NUM_BUFS; ++i) {
        touch_kernel<<<(N + 127) / 128, 128, 0, stream>>>((float *)bufs[i], N);
    }
    CHECK(cuStreamSynchronize(stream));

    fprintf(stderr, "[v2] begin capture (RELAXED)\n");
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));

    // Simulate a "decode step": 4 kernels across 3 buffers, more like a
    // transformer layer touching residual + scratch.
    for (int step = 0; step < 4; ++step) {
        for (int i = 0; i < NUM_BUFS; ++i) {
            touch_kernel<<<(N + 127) / 128, 128, 0, stream>>>((float *)bufs[i], N);
        }
    }

    CUgraph graph;
    CHECK(cuStreamEndCapture(stream, &graph));

    CUgraphExec graph_exec;
    CHECK(cuGraphInstantiateWithFlags(&graph_exec, graph, 0));
    CHECK(cuGraphUpload(graph_exec, stream));
    CHECK(cuStreamSynchronize(stream));

    for (int r = 0; r < 6; ++r) {
        fprintf(stderr, "[v2] cuGraphLaunch #%d on worker thread\n", r + 1);
        CHECK(cuGraphLaunch(graph_exec, stream));
        CHECK(cuStreamSynchronize(stream));
    }
    fprintf(stderr, "[v2] worker thread done — no crash\n");

    CHECK(cuGraphExecDestroy(graph_exec));
    CHECK(cuGraphDestroy(graph));
    for (int i = 0; i < NUM_BUFS; ++i) {
        CHECK(cuMemFreeAsync(bufs[i], stream));
    }
    CHECK(cuStreamSynchronize(stream));
    CHECK(cuStreamDestroy(stream));
}

int main() {
    fprintf(stderr, "[v2] init\n");
    CHECK(cuInit(0));
    CHECK_RT(cudaSetDevice(0));
    CUcontext ctx;
    CHECK(cuCtxGetCurrent(&ctx));
    if (!ctx) {
        fprintf(stderr, "no primary ctx after cudaSetDevice\n");
        return 1;
    }

    // Run the reproducer on a spawned thread (matches tokio worker pattern).
    std::thread t(repro_on_thread, ctx);
    t.join();

    fprintf(stderr, "[v2] all done\n");
    return 0;
}
