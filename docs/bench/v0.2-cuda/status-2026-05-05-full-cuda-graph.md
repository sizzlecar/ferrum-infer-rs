# Phase 13: FULL CUDA Graph A/B — null result

**Branch:** `bench/v0.2-cuda` HEAD `1ece52d`
**Date:** 2026-05-05
**Bench:** `bench/v0.2-cuda/graph_ab.sh` — 4 graph configs vs baseline,
c=16 32 prompts × 128 tokens

## TL;DR

The existing FERRUM_BATCHED_GRAPH=1 path (full-iter CUDA graph capture
of `decode_batch_internal`'s 32-layer forward) **runs but provides zero
throughput improvement** on Llama-3.1-8B INT4 + RTX 4090 + c=16. Tested
4 combinations of `cuGraphUpload` + `cuStreamSynchronize` toggles; all
land at 510-513 tok/s = baseline (520). The Phase 9 conclusion
"per-kernel GPU time 53µs >> launch overhead 5µs, graph not the
bottleneck" extrapolates to the full graph too — even with all sync
overhead stripped (replay total = 0.95ms), the savings don't translate
to wall-clock gain because downstream `to_vec` sync still waits for
the same GPU work.

## Numbers

| Config (graph) | tok/s | TPOT_p50 | replay total | upload | launch | sync |
|---|---|---|---|---|---|---|
| **baseline (no graph)** | **520** | 22.7 ms | — | — | — | — |
| default (upload + sync) | 511 | 22.95 | 9.3 ms | 380µs | 750µs | 8200µs |
| skip post-launch sync | 510 | 22.94 | 1.1 ms | 380µs | 780µs | 0 |
| skip pre-launch upload | 513 | 22.80 | 8.9 ms | 0 | 920µs | 8000µs |
| skip both | **512** | **22.83** | **0.95 ms** | 0 | 950µs | 0 |

All within run-to-run noise (±5%).

## Anatomy: why no win

`replay_graph` per call:
- `cuGraphUpload` — re-uploads graph state to GPU before launch. ~380µs.
- `cuGraphLaunch` — async, dispatches all captured kernels. ~750-950µs.
- `cuStreamSynchronize` — host blocks until graph fully done. ~8000µs
  (this IS the GPU compute time for the 32-layer forward).

The 8ms sync **is the actual GPU work**. Removing the sync doesn't speed
anything up because the very next call (`B::sync` + `B::to_vec` for
batch_logits in `decode_batch_internal`) has to wait for the same GPU
work to finish anyway.

What graph CAN save: host-side dispatch overhead per kernel launch.
For ferrum's batched decode at m=16:
- 320 launches per iter × 5µs host dispatch = 1.6ms total host work
- 320 launches × 53µs avg GPU time = 17ms total GPU work
- → Host dispatch is 9% of iter; even fully eliminated → +10% throughput max

But measured graph-replay savings is **0%**, not +10%. The host dispatch
isn't actually visible at the iter level because:
1. cudarc's `launch_builder` queues kernels onto the stream and returns
   immediately. Host is free to do the next launch while GPU runs prior.
2. So in eager mode, the 1.6ms of host-side launch dispatch is OVERLAPPED
   with the 17ms of GPU work, not added on top.
3. Graph mode does the same launches in one batched dispatch, saves
   maybe 0.5-1ms of cumulative cudarc overhead, but that's already
   overlapped with GPU work too.

Net: graph buys us nothing on this workload.

## Why vLLM's "FULL CUDA Graph = 3×" claim doesn't apply

Per project_vllm_decode_design memory, vLLM gets 3× decode speedup from
FULL CUDA Graph at small m. But:
- vLLM's claim is for SMALLER MODELS where per-kernel GPU time is tiny
  (~5-10µs each) and launch overhead dominates. For those models, graph
  could plausibly remove half the iter time.
- For Llama-3.1-8B at INT4 m=16, kernels are big enough (Marlin GEMM
  ~64µs, attention ~54µs) that launch overhead is fundamentally not
  the bottleneck.

If we benchmarked a tiny model (Qwen3-0.6B or smaller) with FULL graph,
we might see the 3× then. But that's not our M2 use case.

## Phase 13 closes

Existing FERRUM_BATCHED_GRAPH=1 code stays (correct, doesn't crash, no
gain — opt-in only). Multiple new env switches added for future debug:
- FERRUM_GRAPH_PROF=1 — log per-replay upload/launch/sync µs
- FERRUM_GRAPH_SKIP_UPLOAD=1 — skip cuGraphUpload pre-launch
- FERRUM_GRAPH_SKIP_SYNC=1 — skip post-launch cuStreamSynchronize

## Remaining levers (post-Phase-13)

ferrum c=16 stuck at 520 tok/s = 33% of vLLM's 1597. Marlin and graph
both proven null. Levers ranked:

1. **Chunked prefill** (engine architecture, ~1-2 weeks): split prefill
   into ≤512-token chunks, mix with decode in same iter. Eliminates the
   2.9s of pure prefill time (per c=16 op-profile finding). Estimated
   +10-20% throughput.

2. **Per-request streaming over single SSE multiplex** (engine,
   ~3-5 days): per the engine-prof finding, ~3.8s of wall-clock is in
   the SSE/HTTP path. Coalescing per-token chunks into batched updates
   could save ~1-2s.

3. **Sampling on GPU** (kernel, ~3-5 days): post-loop sample 1.3ms/iter
   = 8% of iter. Move top-k/p + argmax to GPU.

4. **Batched paged attn split-K** (kernel, ~3-5 days): attn 13.5% iter
   ceiling; even fully eliminated → +13% max.

None get us to vLLM parity alone. The 3× gap may be irreducible without
matching vLLM's full architecture (chunked prefill + paged attention V2
+ scheduler tuning + tokenizer optimizations).
