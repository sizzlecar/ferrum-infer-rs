// Graph reproducer v6 — the actual hypothesis for ferrum's
// "SIGSEGV after N launches" Phase 4d failure.
//
// Background. v1-v5 all PASS on Blackwell + CUDA 13 with these patterns:
//   v1: runtime API + cuMemAlloc + 1 kernel  → OK
//   v2: stream-pool alloc + 3 bufs + worker thread → OK
//   v3: cuCtxCreate_v4 + pool alloc → OK
//   v4: cuBLAS sgemm × 3 inside capture → OK
//   v5: capture on thread A, replay on thread B → OK
//
// What ferrum's actual capture path has that v1-v5 LACK:
//
// `kv_cache_append_batched_per_cache` (called per layer × 32):
//     if (ctx.batched_scratch_u64_cache.is_none()) {
//         ctx.batched_scratch_u64_cache = Some(
//             stream.alloc_zeros::<u64>(HOST_STAGING_TOTAL)  // ← cuMemAllocFromPoolAsync
//                 .map_err(...)?,
//         );
//     }
//     stream.memcpy_htod(host_slice, &mut view)?;            // ← cuMemcpyHtoDAsync
//     stream.launch_builder(...).launch(...)?;               // ← kernel launch
//
// `flash_attention_batched_per_cache`: same pattern × 2 (k_scratch + v_scratch).
//
// First call inside `decode_batch_internal` triggers:
//   - 3 × `stream.alloc_zeros` queued onto the stream → recorded as
//     CU_GRAPH_NODE_TYPE_MEM_ALLOC nodes in the captured graph
//   - With `AUTO_FREE_ON_LAUNCH`: each launch frees those allocations.
//     But subsequent kernels' parameters still point at those (now-freed)
//     addresses → SIGSEGV after some number of replays.
//   - Without `AUTO_FREE_ON_LAUNCH` (flags=0): allocations leak → faster
//     OOM → also SIGSEGV.
//
// This file tests THAT EXACT pattern: cuMemAllocFromPoolAsync inside
// the captured region, then a kernel that reads from those allocations,
// then 30 replays under both flag settings.
//
// Build (CUDA 12.4 toolkit, sm_89 = Ada/RTX 4090):
//   nvcc -O2 -arch=sm_89 -o /tmp/graph_repro_v6 graph_repro_v6.cu -lcuda
// Run:
//   /tmp/graph_repro_v6                      # natural exit / crash
//   compute-sanitizer --tool=memcheck /tmp/graph_repro_v6
//   compute-sanitizer --tool=initcheck /tmp/graph_repro_v6
//   cuda-gdb --batch -ex run -ex "bt 20" /tmp/graph_repro_v6

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Tiny kernel that reads + writes its scratch — proxies for ferrum's
// kv_cache_append / flash_attn / etc. Uses scratch[i] += 1 so we can
// verify replays via a final dtoh.
__global__ void touch_scratch(unsigned long long *scratch, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        scratch[i] = scratch[i] + 1;
    }
}

#define CHECK(call)                                                            \
    do {                                                                       \
        CUresult err = (call);                                                 \
        if (err != CUDA_SUCCESS) {                                             \
            const char *name = nullptr;                                        \
            cuGetErrorName(err, &name);                                        \
            fprintf(stderr, "[v6] driver call failed at %s:%d: %s (%d)\n",     \
                    __FILE__, __LINE__, name ? name : "?", (int)err);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_RT(call)                                                         \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "[v6] runtime call failed at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static void print_vram(const char *tag) {
    size_t free_b = 0, total_b = 0;
    CHECK(cuMemGetInfo(&free_b, &total_b));
    fprintf(stderr, "[v6:%s] VRAM free=%.1f MB / %.1f MB\n",
            tag,
            (double)free_b / (1024.0 * 1024.0),
            (double)total_b / (1024.0 * 1024.0));
}

// ── PART A: alloc INSIDE capture + AUTO_FREE_ON_LAUNCH ──────────────
//
// Three cuMemAllocFromPoolAsync calls inside capture (mirrors ferrum's
// 3 scratch allocs in kv_cache_append + flash_attn). One kernel per
// scratch reads it. Captured graph contains:
//   - 3 × MEM_ALLOC nodes
//   - 3 × KERNEL nodes referencing those allocations
//   - 3 × MEM_FREE nodes (auto-inserted when AUTO_FREE_ON_LAUNCH is set?)
//
// AUTO_FREE_ON_LAUNCH semantics: per CUDA docs, this flag causes the
// graph to free its stream-ordered allocations at LAUNCH end. This is
// supposed to enable the same pool-backed memory to be reused across
// replays. But if kernel params capture the alloc'd address at capture
// time and AUTO_FREE makes the next replay's alloc return a DIFFERENT
// address, the kernel reads stale memory.
static int part_a_alloc_inside_capture_autofree(CUcontext ctx,
                                                CUstream stream,
                                                CUmemoryPool pool) {
    fprintf(stderr, "\n[v6:A] === alloc-inside-capture + AUTO_FREE_ON_LAUNCH ===\n");
    print_vram("A:before");

    const int N = 4096;
    const int NUM_SCRATCH = 3;

    // Warm-up: alloc + free a single buffer outside capture so the pool
    // is initialised and won't pick up "first-time" cost on replay 1.
    CUdeviceptr warm;
    CHECK(cuMemAllocFromPoolAsync(&warm, N * sizeof(unsigned long long), pool, stream));
    CHECK(cuMemFreeAsync(warm, stream));
    CHECK(cuStreamSynchronize(stream));
    print_vram("A:after-warmup");

    fprintf(stderr, "[v6:A] begin capture\n");
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));

    CUdeviceptr scratch[NUM_SCRATCH];
    for (int i = 0; i < NUM_SCRATCH; ++i) {
        // ← THIS is the suspect call. Stream-ordered alloc inside
        //   captured region.
        CHECK(cuMemAllocFromPoolAsync(&scratch[i], N * sizeof(unsigned long long), pool, stream));
    }
    for (int i = 0; i < NUM_SCRATCH; ++i) {
        // Kernel reads from the alloc'd scratch.
        touch_scratch<<<(N + 127) / 128, 128, 0, stream>>>(
            (unsigned long long *)scratch[i], N);
    }
    // Don't explicitly cuMemFreeAsync — let AUTO_FREE handle it.

    CUgraph graph;
    CHECK(cuStreamEndCapture(stream, &graph));
    fprintf(stderr, "[v6:A] capture ended, instantiating with AUTO_FREE_ON_LAUNCH\n");

    CUgraphExec graph_exec;
    CHECK(cuGraphInstantiateWithFlags(
        &graph_exec, graph,
        CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH));
    CHECK(cuGraphUpload(graph_exec, stream));
    CHECK(cuStreamSynchronize(stream));
    print_vram("A:after-instantiate");

    int crashed_at = -1;
    for (int r = 0; r < 30; ++r) {
        fprintf(stderr, "[v6:A] cuGraphLaunch #%d ", r + 1);
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
        if ((r + 1) % 5 == 0) print_vram("A:loop");
    }

    if (crashed_at < 0) {
        fprintf(stderr, "[v6:A] all 30 launches survived — alloc-inside-capture + AUTO_FREE WORKS\n");
    } else {
        fprintf(stderr, "[v6:A] CRASHED at launch #%d — hypothesis CONFIRMED for AUTO_FREE\n", crashed_at);
    }

    CHECK(cuGraphExecDestroy(graph_exec));
    CHECK(cuGraphDestroy(graph));
    return crashed_at;
}

// ── PART B: same pattern but flags=0 (no AUTO_FREE) ─────────────────
//
// Without AUTO_FREE, allocations should LEAK each launch. This either:
//  - Fails fast with OOM after a few launches
//  - Reads correct (un-freed) memory and succeeds (slower SIGSEGV path)
static int part_b_alloc_inside_capture_no_autofree(CUcontext ctx,
                                                   CUstream stream,
                                                   CUmemoryPool pool) {
    fprintf(stderr, "\n[v6:B] === alloc-inside-capture + flags=0 (NO AUTO_FREE) ===\n");
    print_vram("B:before");

    const int N = 4096;
    const int NUM_SCRATCH = 3;

    fprintf(stderr, "[v6:B] begin capture\n");
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));

    CUdeviceptr scratch[NUM_SCRATCH];
    for (int i = 0; i < NUM_SCRATCH; ++i) {
        CHECK(cuMemAllocFromPoolAsync(&scratch[i], N * sizeof(unsigned long long), pool, stream));
    }
    for (int i = 0; i < NUM_SCRATCH; ++i) {
        touch_scratch<<<(N + 127) / 128, 128, 0, stream>>>(
            (unsigned long long *)scratch[i], N);
    }
    // No explicit free, no AUTO_FREE → expect leak

    CUgraph graph;
    CHECK(cuStreamEndCapture(stream, &graph));
    fprintf(stderr, "[v6:B] capture ended, instantiating with flags=0\n");

    CUgraphExec graph_exec;
    CHECK(cuGraphInstantiateWithFlags(&graph_exec, graph, 0));
    CHECK(cuGraphUpload(graph_exec, stream));
    CHECK(cuStreamSynchronize(stream));
    print_vram("B:after-instantiate");

    int crashed_at = -1;
    for (int r = 0; r < 30; ++r) {
        fprintf(stderr, "[v6:B] cuGraphLaunch #%d ", r + 1);
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
        if ((r + 1) % 5 == 0) print_vram("B:loop");
    }

    if (crashed_at < 0) {
        fprintf(stderr, "[v6:B] all 30 launches survived\n");
    } else {
        fprintf(stderr, "[v6:B] CRASHED at launch #%d\n", crashed_at);
    }

    CHECK(cuGraphExecDestroy(graph_exec));
    CHECK(cuGraphDestroy(graph));
    return crashed_at;
}

// ── PART C: pre-allocated buffers + AUTO_FREE_ON_LAUNCH ─────────────
//
// Move all allocations OUTSIDE capture (mirrors ferrum's
// `populate_batched_pointers` pre-fill approach). If this passes 30
// launches cleanly, the fix is "pre-allocate everything before
// begin_graph_capture; never alloc inside captured region".
static int part_c_prealloc_autofree(CUcontext ctx,
                                    CUstream stream,
                                    CUmemoryPool pool) {
    fprintf(stderr, "\n[v6:C] === pre-allocated + AUTO_FREE_ON_LAUNCH ===\n");
    print_vram("C:before");

    const int N = 4096;
    const int NUM_SCRATCH = 3;

    CUdeviceptr scratch[NUM_SCRATCH];
    for (int i = 0; i < NUM_SCRATCH; ++i) {
        CHECK(cuMemAlloc(&scratch[i], N * sizeof(unsigned long long)));
        CHECK(cuMemsetD8(scratch[i], 0, N * sizeof(unsigned long long)));
    }
    print_vram("C:after-prealloc");

    fprintf(stderr, "[v6:C] begin capture\n");
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));

    for (int i = 0; i < NUM_SCRATCH; ++i) {
        touch_scratch<<<(N + 127) / 128, 128, 0, stream>>>(
            (unsigned long long *)scratch[i], N);
    }

    CUgraph graph;
    CHECK(cuStreamEndCapture(stream, &graph));

    CUgraphExec graph_exec;
    CHECK(cuGraphInstantiateWithFlags(
        &graph_exec, graph,
        CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH));
    CHECK(cuGraphUpload(graph_exec, stream));
    CHECK(cuStreamSynchronize(stream));
    print_vram("C:after-instantiate");

    int crashed_at = -1;
    for (int r = 0; r < 30; ++r) {
        fprintf(stderr, "[v6:C] cuGraphLaunch #%d ", r + 1);
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
        if ((r + 1) % 5 == 0) print_vram("C:loop");
    }

    if (crashed_at < 0) {
        fprintf(stderr, "[v6:C] all 30 launches survived — PRE-ALLOC + AUTO_FREE WORKS\n");
    } else {
        fprintf(stderr, "[v6:C] CRASHED at launch #%d — even pre-alloc fails\n", crashed_at);
    }

    // Verify scratch was actually written (each replay adds 1 to each elem).
    std::vector<unsigned long long> host_check(N, 0);
    CHECK(cuMemcpyDtoH(host_check.data(), scratch[0], N * sizeof(unsigned long long)));
    int expected = (crashed_at < 0) ? 30 : (crashed_at - 1);
    fprintf(stderr, "[v6:C] scratch[0][0]=%llu (expected ~%d if all replays ran)\n",
            host_check[0], expected);

    for (int i = 0; i < NUM_SCRATCH; ++i) {
        CHECK(cuMemFree(scratch[i]));
    }
    CHECK(cuGraphExecDestroy(graph_exec));
    CHECK(cuGraphDestroy(graph));
    return crashed_at;
}

int main() {
    fprintf(stderr, "[v6] init driver\n");
    CHECK(cuInit(0));

    // Use runtime API to materialise primary ctx (avoids
    // cuCtxCreate_v4 which needs CUDA 13 headers).
    CHECK_RT(cudaSetDevice(0));
    CUcontext ctx;
    CHECK(cuCtxGetCurrent(&ctx));
    if (!ctx) {
        fprintf(stderr, "no primary ctx\n");
        return 1;
    }

    CUstream stream;
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    CUdevice dev;
    CHECK(cuCtxGetDevice(&dev));
    CUmemoryPool pool;
    CHECK(cuDeviceGetDefaultMemPool(&pool, dev));

    int a = part_a_alloc_inside_capture_autofree(ctx, stream, pool);
    int b = part_b_alloc_inside_capture_no_autofree(ctx, stream, pool);
    int c = part_c_prealloc_autofree(ctx, stream, pool);

    fprintf(stderr, "\n[v6] === SUMMARY ===\n");
    fprintf(stderr, "[v6] A (alloc-in-capture + AUTO_FREE):     %s%s\n",
            a < 0 ? "OK" : "CRASHED at launch #",
            a < 0 ? "" : std::to_string(a).c_str());
    fprintf(stderr, "[v6] B (alloc-in-capture + flags=0):       %s%s\n",
            b < 0 ? "OK" : "CRASHED at launch #",
            b < 0 ? "" : std::to_string(b).c_str());
    fprintf(stderr, "[v6] C (pre-alloc + AUTO_FREE):            %s%s\n",
            c < 0 ? "OK" : "CRASHED at launch #",
            c < 0 ? "" : std::to_string(c).c_str());
    fprintf(stderr, "[v6] If A/B crash but C is OK → fix is to move allocs outside capture.\n");

    CHECK(cuStreamDestroy(stream));
    return 0;
}
