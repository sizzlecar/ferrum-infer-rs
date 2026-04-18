// Minimal cuGraphLaunch reproducer for the Blackwell + CUDA 13 libcuda.so
// SIGSEGV we observed in ferrum-infer-rs.
//
// Mirrors the exact pattern our Rust code uses via cudarc:
//   cuCtxCreate / cuStreamCreate
//   cuStreamBeginCapture(RELAXED)
//   a few kernel launches on that stream
//   cuStreamEndCapture → cu_graph
//   cuGraphInstantiateWithFlags(flags=0) → cu_graph_exec
//   cuGraphUpload(exec, stream)       [optional]
//   cuGraphLaunch(exec, stream)       ← the crash site
//   cuStreamSynchronize(stream)
//
// If this binary segfaults inside libcuda.so with the same stack
// (frame #0 ?? in libcuda.so, frame #3 cuGraphLaunch), the bug is
// purely driver-side — nothing in our Rust layer triggered it.
//
// If it succeeds, something in our code path (event tracking, memory
// pool state, disabled event tracking interaction, cudarc wrapper,
// ...) is the actual trigger.
//
// Build:
//   nvcc -o graph_repro graph_repro.cu -lcuda
// Run:
//   gdb --batch -ex "handle SIGSEGV stop" -ex run -ex "bt 10" ./graph_repro

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// A trivial kernel — same "shape" as our decode work: one thread per element
// doing a tiny op. Enough to put something into the captured graph.
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
            fprintf(stderr, "CUDA driver call failed at %s:%d: %s (%d)\n",     \
                    __FILE__, __LINE__, name ? name : "?", (int)err);          \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_RT(call)                                                         \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA runtime call failed: %s\n",                  \
                    cudaGetErrorString(err));                                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

int main() {
    fprintf(stderr, "[repro] init driver\n");
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    // cuCtxCreate in CUDA 13 resolves to cuCtxCreate_v4 which takes extra
    // args; use _v2 explicitly to keep the signature stable across SDK
    // versions (our Rust uses v2 via cudarc 0.19 too).
    CHECK(cuCtxCreate_v2(&ctx, 0, dev));

    // Equivalent of cudarc's default stream + our context.
    CUstream stream;
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    // Allocate a working buffer, mirror our "scratch" allocation pattern.
    const int N = 1024;
    float *dev_buf = nullptr;
    CHECK(cuMemAlloc((CUdeviceptr *)&dev_buf, N * sizeof(float)));
    CHECK_RT(cudaMemsetAsync(dev_buf, 0, N * sizeof(float), stream));
    CHECK(cuStreamSynchronize(stream));

    fprintf(stderr, "[repro] warm-up launch outside capture\n");
    // Warm up — some drivers misbehave if the first launch happens inside
    // capture. This matches our Rust path where model construction runs
    // plenty of kernel work before the first decode step.
    touch_kernel<<<(N + 127) / 128, 128, 0, stream>>>(dev_buf, N);
    CHECK(cuStreamSynchronize(stream));

    fprintf(stderr, "[repro] begin capture (RELAXED)\n");
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));

    // A couple of kernel launches inside capture — mirrors a decode step.
    for (int i = 0; i < 4; ++i) {
        touch_kernel<<<(N + 127) / 128, 128, 0, stream>>>(dev_buf, N);
    }

    CUgraph graph;
    fprintf(stderr, "[repro] end capture\n");
    CHECK(cuStreamEndCapture(stream, &graph));

    CUgraphExec graph_exec;
    fprintf(stderr, "[repro] instantiate (flags=0)\n");
    CHECK(cuGraphInstantiateWithFlags(&graph_exec, graph, 0));

    fprintf(stderr, "[repro] upload\n");
    CHECK(cuGraphUpload(graph_exec, stream));
    CHECK(cuStreamSynchronize(stream));

    // The moneyshot — this is where we see the SIGSEGV in ferrum.
    fprintf(stderr, "[repro] cuGraphLaunch #1\n");
    CHECK(cuGraphLaunch(graph_exec, stream));
    fprintf(stderr, "[repro] post-launch sync\n");
    CHECK(cuStreamSynchronize(stream));

    // A few more replays to match our decode loop usage.
    for (int i = 0; i < 5; ++i) {
        fprintf(stderr, "[repro] cuGraphLaunch #%d\n", i + 2);
        CHECK(cuGraphLaunch(graph_exec, stream));
        CHECK(cuStreamSynchronize(stream));
    }

    fprintf(stderr, "[repro] all replays succeeded — graph works on this box\n");

    // Cleanup.
    CHECK(cuGraphExecDestroy(graph_exec));
    CHECK(cuGraphDestroy(graph));
    CHECK(cuMemFree((CUdeviceptr)dev_buf));
    CHECK(cuStreamDestroy(stream));
    CHECK(cuCtxDestroy(ctx));
    return 0;
}
