// Graph reproducer v8 — isolate per-launch cuGraphUpload as the
// remaining suspect for ferrum's "SIGSEGV at launch #15" pattern.
//
// Cumulative findings v1-v7 RULE OUT:
//   * AUTO_FREE_ON_LAUNCH (v6-A)     — 30 OK
//   * 128 captured cuBLAS (v7-D)     — 30 OK
//   * 128 captured memcpy (v7-E)     — 30 OK
//   * host rewrite between replays (v7-F) — 30 OK + correct values
//   * pre-allocated buffers (v6-C)   — 30 OK
//
// The ONLY remaining ferrum-specific wart that v1-v7 don't reproduce is:
//
//   replay_last_graph() in cuda.rs:417-445 calls
//     unsafe { cuGraphUpload(g.cu_graph_exec, cu_stream) };  // each replay
//     unsafe { cuGraphLaunch(g.cu_graph_exec, cu_stream) };
//     unsafe { cuStreamSynchronize(cu_stream) };
//
//   Standard CUDA Graph usage (vLLM, NVIDIA samples) calls cuGraphUpload
//   ONCE after instantiate, then cuGraphLaunch in a loop. Per-launch
//   upload is documented to "re-upload to refresh device-side state"
//   but is non-standard.
//
// v8 tests:
//   G: alloc-in-capture + AUTO_FREE + per-launch upload + 50 replays
//      → if dies at #15-ish, per-launch upload is the bug
//   H: same setup but NO per-launch upload (one upload after instantiate)
//      → control: should be 50 OK
//   I: same as G but with many ops (64 captured ops, mimics ferrum scale)
//      → if dies sooner with more ops, per-launch upload + scale interact
//
// Build:
//   nvcc -O2 -arch=sm_89 -o /tmp/graph_repro_v8 graph_repro_v8.cu -lcuda
//   /tmp/graph_repro_v8
//   compute-sanitizer --tool=memcheck /tmp/graph_repro_v8

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

__global__ void touch_u64(unsigned long long *scratch, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) scratch[i] = scratch[i] + 1;
}

#define CHECK(call)                                                            \
    do {                                                                       \
        CUresult err = (call);                                                 \
        if (err != CUDA_SUCCESS) {                                             \
            const char *name = nullptr;                                        \
            cuGetErrorName(err, &name);                                        \
            fprintf(stderr, "[v8] driver call failed at %s:%d: %s (%d)\n",     \
                    __FILE__, __LINE__, name ? name : "?", (int)err);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_RT(call)                                                         \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "[v8] runtime call failed: %s\n",                  \
                    cudaGetErrorString(err));                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Capture a graph with `num_allocs` cuMemAllocFromPoolAsync calls and
// `num_kernels` touch_u64 launches inside the captured region. Replay
// `num_replays` times. If `per_launch_upload`, call cuGraphUpload before
// each cuGraphLaunch (mirrors ferrum's replay_last_graph path).
static int run_variant(const char *tag,
                       CUstream stream,
                       CUmemoryPool pool,
                       int num_allocs,
                       int num_kernels,
                       int num_replays,
                       bool per_launch_upload) {
    fprintf(stderr, "\n[v8:%s] === %d allocs, %d kernels, %d replays, per-launch-upload=%d ===\n",
            tag, num_allocs, num_kernels, num_replays, per_launch_upload ? 1 : 0);

    const int N = 4096;

    fprintf(stderr, "[v8:%s] begin capture\n", tag);
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));

    std::vector<CUdeviceptr> scratch(num_allocs);
    for (int i = 0; i < num_allocs; ++i) {
        CHECK(cuMemAllocFromPoolAsync(&scratch[i], N * sizeof(unsigned long long), pool, stream));
    }
    for (int i = 0; i < num_kernels; ++i) {
        // Cycle through allocated scratch buffers (or use first if none)
        CUdeviceptr buf = num_allocs > 0 ? scratch[i % num_allocs] : 0;
        if (buf) {
            touch_u64<<<(N + 127) / 128, 128, 0, stream>>>(
                (unsigned long long *)buf, N);
        }
    }
    CUgraph graph;
    CHECK(cuStreamEndCapture(stream, &graph));

    CUgraphExec graph_exec;
    fprintf(stderr, "[v8:%s] instantiate AUTO_FREE\n", tag);
    CHECK(cuGraphInstantiateWithFlags(
        &graph_exec, graph,
        CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH));
    if (!per_launch_upload) {
        // One-shot upload after instantiate (standard pattern).
        CHECK(cuGraphUpload(graph_exec, stream));
        CHECK(cuStreamSynchronize(stream));
    }

    int crashed_at = -1;
    for (int r = 0; r < num_replays; ++r) {
        if (per_launch_upload) {
            CUresult st_u = cuGraphUpload(graph_exec, stream);
            if (st_u != CUDA_SUCCESS) {
                const char *name = nullptr;
                cuGetErrorName(st_u, &name);
                fprintf(stderr, "[v8:%s] cuGraphUpload #%d FAIL: %s\n",
                        tag, r + 1, name ? name : "?");
                crashed_at = r + 1;
                break;
            }
        }
        CUresult st_l = cuGraphLaunch(graph_exec, stream);
        if (st_l != CUDA_SUCCESS) {
            const char *name = nullptr;
            cuGetErrorName(st_l, &name);
            fprintf(stderr, "[v8:%s] cuGraphLaunch #%d FAIL: %s\n",
                    tag, r + 1, name ? name : "?");
            crashed_at = r + 1;
            break;
        }
        CUresult st_s = cuStreamSynchronize(stream);
        if (st_s != CUDA_SUCCESS) {
            const char *name = nullptr;
            cuGetErrorName(st_s, &name);
            fprintf(stderr, "[v8:%s] cuStreamSynchronize #%d FAIL: %s\n",
                    tag, r + 1, name ? name : "?");
            crashed_at = r + 1;
            break;
        }
        if ((r + 1) % 5 == 0 || r < 5) {
            size_t free_b, total_b;
            CHECK(cuMemGetInfo(&free_b, &total_b));
            fprintf(stderr, "[v8:%s] replay #%d OK  free=%.1f MB\n",
                    tag, r + 1, (double)free_b / (1024.0 * 1024.0));
        }
    }

    if (crashed_at < 0) {
        fprintf(stderr, "[v8:%s] all %d replays survived\n", tag, num_replays);
    } else {
        fprintf(stderr, "[v8:%s] CRASHED at replay #%d\n", tag, crashed_at);
    }

    CHECK(cuGraphExecDestroy(graph_exec));
    CHECK(cuGraphDestroy(graph));
    return crashed_at;
}

int main() {
    fprintf(stderr, "[v8] init\n");
    CHECK(cuInit(0));
    CHECK_RT(cudaSetDevice(0));
    CUcontext ctx;
    CHECK(cuCtxGetCurrent(&ctx));

    CUstream stream;
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    CUdevice dev;
    CHECK(cuCtxGetDevice(&dev));
    CUmemoryPool pool;
    CHECK(cuDeviceGetDefaultMemPool(&pool, dev));

    // G: 3 allocs + 8 kernels + per-launch upload (ferrum's pattern)
    int g = run_variant("G", stream, pool, 3, 8, 50, true);

    // H: same but ONE-SHOT upload (standard pattern)
    int h = run_variant("H", stream, pool, 3, 8, 50, false);

    // I: scale up — 16 allocs + 64 kernels + per-launch upload
    //    (closer to ferrum's 32-layer scale)
    int i = run_variant("I", stream, pool, 16, 64, 50, true);

    // J: same scale but ONE-SHOT upload
    int j = run_variant("J", stream, pool, 16, 64, 50, false);

    fprintf(stderr, "\n[v8] === SUMMARY ===\n");
    auto fmt = [](int x) -> std::string {
        if (x < 0) return "OK";
        return "CRASHED at #" + std::to_string(x);
    };
    fprintf(stderr, "[v8] G (3 allocs + 8 kernels + per-launch upload):    %s\n", fmt(g).c_str());
    fprintf(stderr, "[v8] H (3 allocs + 8 kernels + ONE-SHOT upload):      %s\n", fmt(h).c_str());
    fprintf(stderr, "[v8] I (16 allocs + 64 kernels + per-launch upload):  %s\n", fmt(i).c_str());
    fprintf(stderr, "[v8] J (16 allocs + 64 kernels + ONE-SHOT upload):    %s\n", fmt(j).c_str());

    if (g >= 0 && h < 0) {
        fprintf(stderr, "\n[v8] HYPOTHESIS CONFIRMED: per-launch cuGraphUpload is the bug.\n");
        fprintf(stderr, "[v8] Fix: instantiate once, upload once, then loop cuGraphLaunch only.\n");
    } else if (g < 0 && h < 0) {
        fprintf(stderr, "\n[v8] per-launch upload is BENIGN at small scale; check I vs J for scale interaction.\n");
    }

    CHECK(cuStreamDestroy(stream));
    return 0;
}
