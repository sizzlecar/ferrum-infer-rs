# C=32 perf tracker — live doc

**Goal**: ferrum M3 c=32 → **80% of vLLM** (1506 tok/s; vLLM 0.20.2 baseline = 1883 tok/s).
**Method**: append a row to the **Run log** below after each optimization. Update only if you have a fresh `nsys` profile (not just bench tok/s).

Reproduce baseline / new run:
```
ssh -p 35702 root@ssh4.vast.ai 'cd /workspace/ferrum-infer-rs && bash bench/v0.2-cuda/nsys_profile.sh ferrum'
# Generates /tmp/ferrum_bench.nsys-rep + ferrum_kernels.csv + ferrum_apis.csv
# Decode-portion API counts:
nsys stats --report cuda_api_trace --format csv /tmp/ferrum_bench.nsys-rep | \
  awk -F, 'NR>1 && $1>10000000000{print $3}' | sort | uniq -c | sort -rn
```

---

## vLLM 0.20.2 reference (M3 c=32, RTX 4090, 25-sec nsys window)

Captured 2026-05-13 @ commit `0f2dadb` (main, post PR #175/#183 merge), via `nohup nsys profile --delay 35 --duration 25 vllm serve …`.

- **Throughput**: 1867.7 tok/s (TPOT_p50 16.1 ms, TPOT_p99 17.2 ms; 128/128 ok)
- **Iter count in window**: ~1006 (from `cuGraphLaunch` count)
- **API per iter** (selected):

| API | total in 25 s | per iter |
|---|---:|---:|
| **cudaGraphLaunch** | 1006 | **1.0** ← entire forward replayed once |
| cudaEventSynchronize | 2072 | 2.0 |
| cudaLaunchKernel | 33596 | 33 |
| cuLaunchKernel | 7063 | 7.0 |
| cudaMemcpyAsync | 12834 | 13 |
| cudaMemsetAsync | 2353 | 2.3 |
| cudaEventRecord | 2072 | 2.0 |
| **cuMemAllocAsync / cuMemFreeAsync** | 0 | **0** ← scratch fully preallocated |

- **Marlin tile**: `Marlin<256, 4, 16, 4, 4, 8>` — 4 m-tiles per launch, 540 µs avg
- **Top GPU kernels**: marlin_moe 35.7%, cutlass lm_head 28.6%, dense marlin 13.5%, flash_fwd 7.2%

---

## ferrum baseline (M3 c=32, same hardware/workload)

| run | commit | tok/s | ratio vs vLLM | notes |
|---|---|---:|---:|---|
| **R0 baseline** | `0f2dadb` (2026-05-13) | **1044** | **55%** | apples; SKIP_VLLM rerun = 1044, matches |

### R0 per-iter breakdown

Bench: `ferrum bench M3 --concurrency 32 --max-tokens 64 --rounds 1`, ~42 decode iter captured (kernel-call division).

**GPU side**: kernel time ~7 ms / iter (131 µs/layer × 48 layers + lm_head 652 µs + argmax 85 µs).
**Wall side**: 30 ms / iter (= 32 / 1044 tok/s) → **23 ms / iter idle waiting for CPU**.

**API per iter (decode portion, t>10e9 ns in nsys)**:

| API | total in window | per iter | vs vLLM | comment |
|---|---:|---:|---:|---|
| cuLaunchKernel | 29398 | **700** | 33 → **21× worse** | no full-forward graph |
| cuKernelGetFunction | 30740 | 731 | — | matches launch count |
| cuMemFreeAsync | 10650 | **253** | 0 → **∞** | scratch not reused |
| cudaLaunchKernel | 10114 | 240 | — | candle/cudarc dispatch path |
| **cuMemcpyHtoDAsync** | 9687 | **231** | 13 → **18×** | host→device per-iter data |
| **cuMemAllocAsync** | 9396 | **224** | 0 → **∞** | new alloc each iter |
| cuMemsetD32Async | 8064 | 192 | 2.3 → 83× | zero-init scratch repeatedly |
| cuMemsetD8Async | 6096 | 145 | — | same |
| **cuStreamSynchronize** | 4946 | **117** | 2 → **58× worse** | main wall-time source |
| cuMemcpyDtoHAsync | 1635 | 39 | — | D2H reads (routing → host) |
| cuMemcpyDtoDAsync | 416 | 10 | — | intra-device copies |
| cuGraphLaunch | 59 | 1.4 | 1.0 → similar | piecewise graph (not full fwd) |

**Marlin tile (ferrum R0)**: `Marlin<256, 1, 8, 8, 4, 8>` — 1 m-tile per launch, 37 µs avg. **5376 calls** vs vLLM's 1920 (2.8× more launches).

---

## Bottleneck taxonomy (by ROI)

| # | item | cost (R0) | leverage | risk |
|---|---|---:|---|---|
| **A** | cuStreamSynchronize 117/iter | ~23 ms idle | **highest** — main wall time | high (root cause = per-layer D2H+sync for routing/dispatch; may need full-forward graph) |
| **B** | cuMemAlloc + Free 477/iter | ~0.7 ms driver-lock | **easy win** | low — pre-allocate scratch, reuse |
| **C** | Marlin tile <256,1,8,…> too narrow | 2.8× extra launches | medium | medium — change tile heuristic; PR #177 already added dynamic block_size for MoE |
| D | HtoD copies 231/iter | bookkeeping | medium | medium — same data uploaded each iter; cache on device |
| E | Memset 192+145/iter | scratch zero-init | low | low — skip when not needed (FERRUM_MARLIN_SKIP_WS_ZERO=1 already on) |

---

## Run log

Append after every optimization. Even if bench is unchanged, capture the API delta to know whether the change had the intended micro-effect.

| date | commit | change | bench tok/s | ratio | sync/iter | alloc+free/iter | launches/iter | HtoD/iter | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026-05-13 | `0f2dadb` | **baseline R0** | 1044 | 55% | 117 | 477 | 700 | 231 | — |
| | | | | | | | | | |

### R0 raw artifacts on pod
- `/tmp/ferrum_bench.nsys-rep` (6.5 MB) — `nsys-ui` to open
- `/tmp/ferrum_kernels.csv`, `/tmp/ferrum_apis.csv`
- `/tmp/vllm_bench.nsys-rep` (4.6 MB)
- `/tmp/vllm_kernels.csv`, `/tmp/vllm_apis.csv`
