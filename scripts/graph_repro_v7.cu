// Graph reproducer v7 — stress test the variables ferrum's path has but
// v1-v6 didn't:
//   D: MANY captured cuBLAS sgemm calls (128, mimicking 32 layers × 4 GEMMs)
//   E: MANY captured cuMemcpyHtoDAsync from STABLE heap arrays (128 calls,
//      mimicking 32 layers × 4 batched ptr arrays in ferrum)
//   F: same as E but each capture's host array address is DIFFERENT
//      (mimics ferrum's per-call CudaState that drops + reallocates host
//      arrays between captures)
//
// v6 results recap:
//   A: alloc-in-capture + AUTO_FREE → 30 launches OK, 0 leak
//   B: alloc-in-capture + flags=0   → CUDA_ERROR_INVALID_VALUE at launch #2
//   C: pre-alloc + AUTO_FREE        → 30 launches OK
//
// So AUTO_FREE handles allocs cleanly; flags=0 is hard-broken with allocs.
// But ferrum's "pre-populate (no in-capture allocs) + AUTO_FREE" still
// SIGSEGVs at launch #15. So there's a SECOND bug. v7 isolates it.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
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
            fprintf(stderr, "[v7] driver call failed at %s:%d: %s (%d)\n",     \
                    __FILE__, __LINE__, name ? name : "?", (int)err);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_RT(call)                                                         \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "[v7] runtime call failed at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t err = (call);                                           \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[v7] cuBLAS call failed at %s:%d: %d\n",          \
                    __FILE__, __LINE__, (int)err);                             \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static void print_vram(const char *tag) {
    size_t free_b = 0, total_b = 0;
    CHECK(cuMemGetInfo(&free_b, &total_b));
    fprintf(stderr, "[v7:%s] VRAM free=%.1f MB / %.1f MB\n",
            tag,
            (double)free_b / (1024.0 * 1024.0),
            (double)total_b / (1024.0 * 1024.0));
}

// ── PART D: 128 cuBLAS sgemm calls inside capture ───────────────────
//
// v4 did 3 sgemms inside capture and was OK. ferrum's full path has
// 32 layers × 4 GEMMs = 128. If cuBLAS internal state (workspace
// management, tensor cache, pointer mode handles) doesn't scale
// linearly inside graph capture, this will reveal it.
static int part_d_many_cublas(CUcontext ctx, CUstream stream) {
    fprintf(stderr, "\n[v7:D] === 128 captured cuBLAS sgemm ===\n");
    print_vram("D:before");

    cublasHandle_t blas;
    CHECK_CUBLAS(cublasCreate(&blas));
    CHECK_CUBLAS(cublasSetStream(blas, (cudaStream_t)stream));
    CHECK_CUBLAS(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE));

    void *workspace;
    const size_t ws_size = 32 * 1024 * 1024;
    CHECK(cuMemAlloc((CUdeviceptr *)&workspace, ws_size));
    CHECK_CUBLAS(cublasSetWorkspace(blas, workspace, ws_size));

    const int M = 16, N = 4096, K = 4096;
    float *d_a, *d_b, *d_c;
    CHECK(cuMemAlloc((CUdeviceptr *)&d_a, M * K * sizeof(float)));
    CHECK(cuMemAlloc((CUdeviceptr *)&d_b, K * N * sizeof(float)));
    CHECK(cuMemAlloc((CUdeviceptr *)&d_c, M * N * sizeof(float)));

    float *d_alpha, *d_beta;
    float h_alpha = 1.0f, h_beta = 0.0f;
    CHECK(cuMemAlloc((CUdeviceptr *)&d_alpha, sizeof(float)));
    CHECK(cuMemAlloc((CUdeviceptr *)&d_beta, sizeof(float)));
    CHECK(cuMemcpyHtoD((CUdeviceptr)d_alpha, &h_alpha, sizeof(float)));
    CHECK(cuMemcpyHtoD((CUdeviceptr)d_beta, &h_beta, sizeof(float)));

    // Warm up.
    for (int i = 0; i < 4; ++i) {
        CHECK_CUBLAS(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 d_alpha, d_b, N, d_a, K, d_beta, d_c, N));
    }
    CHECK(cuStreamSynchronize(stream));
    print_vram("D:after-warmup");

    fprintf(stderr, "[v7:D] begin capture — 128 sgemm calls\n");
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));
    for (int i = 0; i < 128; ++i) {
        CHECK_CUBLAS(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 d_alpha, d_b, N, d_a, K, d_beta, d_c, N));
    }
    CUgraph graph;
    CHECK(cuStreamEndCapture(stream, &graph));

    CUgraphExec graph_exec;
    fprintf(stderr, "[v7:D] instantiate (AUTO_FREE)\n");
    CHECK(cuGraphInstantiateWithFlags(
        &graph_exec, graph,
        CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH));
    CHECK(cuGraphUpload(graph_exec, stream));
    CHECK(cuStreamSynchronize(stream));
    print_vram("D:after-instantiate");

    int crashed_at = -1;
    for (int r = 0; r < 30; ++r) {
        fprintf(stderr, "[v7:D] cuGraphLaunch #%d ", r + 1);
        CUresult st_l = cuGraphLaunch(graph_exec, stream);
        if (st_l != CUDA_SUCCESS) {
            const char *name = nullptr;
            cuGetErrorName(st_l, &name);
            fprintf(stderr, "→ launch FAIL: %s\n", name ? name : "?");
            crashed_at = r + 1;
            break;
        }
        CUresult st_s = cuStreamSynchronize(stream);
        if (st_s != CUDA_SUCCESS) {
            const char *name = nullptr;
            cuGetErrorName(st_s, &name);
            fprintf(stderr, "→ sync FAIL: %s\n", name ? name : "?");
            crashed_at = r + 1;
            break;
        }
        fprintf(stderr, "OK\n");
        if ((r + 1) % 5 == 0) print_vram("D:loop");
    }

    if (crashed_at < 0) {
        fprintf(stderr, "[v7:D] all 30 launches survived — 128 cuBLAS WORKS\n");
    } else {
        fprintf(stderr, "[v7:D] CRASHED at launch #%d — cuBLAS-count is the trigger\n", crashed_at);
    }

    CHECK(cuGraphExecDestroy(graph_exec));
    CHECK(cuGraphDestroy(graph));
    CHECK(cuMemFree((CUdeviceptr)d_a));
    CHECK(cuMemFree((CUdeviceptr)d_b));
    CHECK(cuMemFree((CUdeviceptr)d_c));
    CHECK(cuMemFree((CUdeviceptr)d_alpha));
    CHECK(cuMemFree((CUdeviceptr)d_beta));
    CHECK(cuMemFree((CUdeviceptr)workspace));
    CHECK_CUBLAS(cublasDestroy(blas));
    return crashed_at;
}

// ── PART E: 128 captured cuMemcpyHtoDAsync from STABLE heap arrays ──
//
// ferrum's batched_decode captures stream.memcpy_htod inside per-layer
// kv_cache_append + flash_attention. Each call uses a unique slot of a
// process-stable heap-allocated host array (post slot fix). Total: 32
// layers × ~4 captured memcpys = 128 captured copies.
//
// The host array lives on the heap (Box<[u64; HOST_STAGING_TOTAL]>) and
// in this part we keep it alive across all 30 replays. So this is the
// CORRECT-LIFETIME case.
static int part_e_many_memcpy_stable(CUcontext ctx, CUstream stream) {
    fprintf(stderr, "\n[v7:E] === 128 captured memcpy from stable heap host ===\n");
    print_vram("E:before");

    const int N = 4096;
    const int NUM_OPS = 128;

    // 128 device buffers (one per captured memcpy target).
    std::vector<CUdeviceptr> dev_bufs(NUM_OPS);
    for (int i = 0; i < NUM_OPS; ++i) {
        CHECK(cuMemAlloc(&dev_bufs[i], 8 * sizeof(unsigned long long)));
    }

    // ONE stable heap array, large enough for 128 × 8 u64 = 8KB. Lives
    // until end of part_e — captured memcpys reference it across all
    // 30 replays.
    std::vector<unsigned long long> host_array(NUM_OPS * 8);
    for (int i = 0; i < NUM_OPS * 8; ++i) host_array[i] = (unsigned long long)i;

    // Workspace: one buffer the kernel will touch per layer (mirrors
    // ferrum's per-layer activation tensor).
    CUdeviceptr work;
    CHECK(cuMemAlloc(&work, N * sizeof(unsigned long long)));
    CHECK(cuMemsetD8(work, 0, N * sizeof(unsigned long long)));

    // Warm up.
    touch_u64<<<(N + 127) / 128, 128, 0, stream>>>((unsigned long long *)work, N);
    CHECK(cuStreamSynchronize(stream));

    fprintf(stderr, "[v7:E] begin capture — 128 memcpy + 128 kernel\n");
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));
    for (int i = 0; i < NUM_OPS; ++i) {
        // Captured async memcpy from per-slot region of stable host array.
        CHECK(cuMemcpyHtoDAsync(dev_bufs[i],
                                &host_array[i * 8],
                                8 * sizeof(unsigned long long),
                                stream));
        // A kernel that touches activation work — proxy for transformer ops.
        touch_u64<<<(N + 127) / 128, 128, 0, stream>>>(
            (unsigned long long *)work, N);
    }
    CUgraph graph;
    CHECK(cuStreamEndCapture(stream, &graph));

    CUgraphExec graph_exec;
    fprintf(stderr, "[v7:E] instantiate (AUTO_FREE)\n");
    CHECK(cuGraphInstantiateWithFlags(
        &graph_exec, graph,
        CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH));
    CHECK(cuGraphUpload(graph_exec, stream));
    CHECK(cuStreamSynchronize(stream));
    print_vram("E:after-instantiate");

    int crashed_at = -1;
    for (int r = 0; r < 30; ++r) {
        fprintf(stderr, "[v7:E] cuGraphLaunch #%d ", r + 1);
        CUresult st_l = cuGraphLaunch(graph_exec, stream);
        if (st_l != CUDA_SUCCESS) {
            const char *name = nullptr;
            cuGetErrorName(st_l, &name);
            fprintf(stderr, "→ launch FAIL: %s\n", name ? name : "?");
            crashed_at = r + 1;
            break;
        }
        CUresult st_s = cuStreamSynchronize(stream);
        if (st_s != CUDA_SUCCESS) {
            const char *name = nullptr;
            cuGetErrorName(st_s, &name);
            fprintf(stderr, "→ sync FAIL: %s\n", name ? name : "?");
            crashed_at = r + 1;
            break;
        }
        fprintf(stderr, "OK\n");
        if ((r + 1) % 5 == 0) print_vram("E:loop");
    }

    if (crashed_at < 0) {
        fprintf(stderr, "[v7:E] all 30 launches survived — 128 stable-heap memcpy WORKS\n");
    } else {
        fprintf(stderr, "[v7:E] CRASHED at launch #%d — captured memcpy count is the trigger\n", crashed_at);
    }

    // Verify dev_bufs[0] got [0..8] (or last replay's value).
    std::vector<unsigned long long> check(8);
    CHECK(cuMemcpyDtoH(check.data(), dev_bufs[0], 8 * sizeof(unsigned long long)));
    fprintf(stderr, "[v7:E] dev_bufs[0] = [");
    for (int i = 0; i < 8; ++i) fprintf(stderr, "%llu%s", check[i], i == 7 ? "" : ",");
    fprintf(stderr, "] (expected [0,1,...,7])\n");

    for (int i = 0; i < NUM_OPS; ++i) CHECK(cuMemFree(dev_bufs[i]));
    CHECK(cuMemFree(work));
    CHECK(cuGraphExecDestroy(graph_exec));
    CHECK(cuGraphDestroy(graph));
    return crashed_at;
}

// ── PART F: same as E but per-replay re-write the host array between
// launches (mimics ferrum's "host array is per-call CudaState; new
// decode call rewrites the same memory location with NEW pointers").
//
// Per the cudarc test `cudarc_graph_shared_host_array_multi_memcpy`
// already known: captured memcpy reads host pointer at LAUNCH time.
// So if we change host_array between replays, replay reads the new
// values. Question: does it still RUN cleanly, or does writing host
// memory between launches break some graph state? ferrum's exact
// failure was hang/SIGSEGV, not just wrong values — this part proves
// whether the rewrite-between-replay pattern itself is unsafe.
static int part_f_rewrite_host_between_replays(CUcontext ctx, CUstream stream) {
    fprintf(stderr, "\n[v7:F] === captured memcpy + rewrite host between replays ===\n");

    const int NUM_OPS = 128;
    std::vector<CUdeviceptr> dev_bufs(NUM_OPS);
    for (int i = 0; i < NUM_OPS; ++i) {
        CHECK(cuMemAlloc(&dev_bufs[i], 8 * sizeof(unsigned long long)));
    }
    std::vector<unsigned long long> host_array(NUM_OPS * 8, 0);

    CUdeviceptr work;
    const int N = 4096;
    CHECK(cuMemAlloc(&work, N * sizeof(unsigned long long)));
    CHECK(cuMemsetD8(work, 0, N * sizeof(unsigned long long)));

    fprintf(stderr, "[v7:F] begin capture\n");
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));
    for (int i = 0; i < NUM_OPS; ++i) {
        CHECK(cuMemcpyHtoDAsync(dev_bufs[i],
                                &host_array[i * 8],
                                8 * sizeof(unsigned long long),
                                stream));
        touch_u64<<<(N + 127) / 128, 128, 0, stream>>>(
            (unsigned long long *)work, N);
    }
    CUgraph graph;
    CHECK(cuStreamEndCapture(stream, &graph));

    CUgraphExec graph_exec;
    CHECK(cuGraphInstantiateWithFlags(
        &graph_exec, graph,
        CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH));
    CHECK(cuGraphUpload(graph_exec, stream));
    CHECK(cuStreamSynchronize(stream));

    int crashed_at = -1;
    for (int r = 0; r < 30; ++r) {
        // Rewrite host_array to (r+1) before each replay.
        for (int i = 0; i < NUM_OPS * 8; ++i) {
            host_array[i] = (unsigned long long)((r + 1) * 1000 + i);
        }
        fprintf(stderr, "[v7:F] cuGraphLaunch #%d ", r + 1);
        CUresult st_l = cuGraphLaunch(graph_exec, stream);
        if (st_l != CUDA_SUCCESS) {
            const char *name = nullptr;
            cuGetErrorName(st_l, &name);
            fprintf(stderr, "→ launch FAIL: %s\n", name ? name : "?");
            crashed_at = r + 1;
            break;
        }
        CUresult st_s = cuStreamSynchronize(stream);
        if (st_s != CUDA_SUCCESS) {
            const char *name = nullptr;
            cuGetErrorName(st_s, &name);
            fprintf(stderr, "→ sync FAIL: %s\n", name ? name : "?");
            crashed_at = r + 1;
            break;
        }
        fprintf(stderr, "OK\n");
    }

    if (crashed_at < 0) {
        fprintf(stderr, "[v7:F] all 30 launches survived — rewrite-between-replay safe\n");
    } else {
        fprintf(stderr, "[v7:F] CRASHED at launch #%d\n", crashed_at);
    }

    // Verify dev_bufs[0] reflects last (replay 30) value: 30000 + 0..7.
    std::vector<unsigned long long> check(8);
    CHECK(cuMemcpyDtoH(check.data(), dev_bufs[0], 8 * sizeof(unsigned long long)));
    fprintf(stderr, "[v7:F] dev_bufs[0] = [");
    for (int i = 0; i < 8; ++i) fprintf(stderr, "%llu%s", check[i], i == 7 ? "" : ",");
    fprintf(stderr, "] (expected ~[30000..30007])\n");

    for (int i = 0; i < NUM_OPS; ++i) CHECK(cuMemFree(dev_bufs[i]));
    CHECK(cuMemFree(work));
    CHECK(cuGraphExecDestroy(graph_exec));
    CHECK(cuGraphDestroy(graph));
    return crashed_at;
}

int main() {
    fprintf(stderr, "[v7] init\n");
    CHECK(cuInit(0));
    CHECK_RT(cudaSetDevice(0));
    CUcontext ctx;
    CHECK(cuCtxGetCurrent(&ctx));

    CUstream stream;
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    int d = part_d_many_cublas(ctx, stream);
    int e = part_e_many_memcpy_stable(ctx, stream);
    int f = part_f_rewrite_host_between_replays(ctx, stream);

    fprintf(stderr, "\n[v7] === SUMMARY ===\n");
    fprintf(stderr, "[v7] D (128 captured cuBLAS sgemm):                  %s%s\n",
            d < 0 ? "OK" : "CRASHED at #",
            d < 0 ? "" : std::to_string(d).c_str());
    fprintf(stderr, "[v7] E (128 captured memcpy stable heap):            %s%s\n",
            e < 0 ? "OK" : "CRASHED at #",
            e < 0 ? "" : std::to_string(e).c_str());
    fprintf(stderr, "[v7] F (128 captured memcpy + rewrite per replay):   %s%s\n",
            f < 0 ? "OK" : "CRASHED at #",
            f < 0 ? "" : std::to_string(f).c_str());

    CHECK(cuStreamDestroy(stream));
    return 0;
}
