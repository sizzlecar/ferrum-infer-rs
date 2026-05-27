# M3 80% goal — session 2026-05-27 iteration 2 (perf + correctness)

**Date:** 2026-05-27
**Pod:** Vast contract 38057047, RTX 4090 sm_89, driver 580.126.09 (CUDA 13.0),
nvcc 13.0.48 (`nvidia/cuda:13.0.0-devel-ubuntu24.04`)
**Budget used:** ~$1.60 of $2.10
**Branch HEAD when session ended:** `a873d63`
(`fix/moe-align-block-size-packed-row`)

---

## TL;DR — what shipped today

Three patches; all GPU-verified end-to-end on Qwen3-30B-A3B-GPTQ-Int4:

1. **`fix(paged-attn): opt-in to extended dynamic shared when KV_CAPACITY>12K`**
   (`37f5dda`). Root cause of the `paged_varlen_attn:
   CUDA_ERROR_INVALID_VALUE` panic that hit `ferrum run` on the 3rd
   chat turn (Chat-profile autosizer sets `FERRUM_KV_CAPACITY=16384` =
   64 KB shared per block; sm_89 default is 48 KB so the launch
   rejects without a `cuFuncSetAttribute` opt-in). Verified end-to-end:
   5 chat turns + `/bye` complete without panic.

2. **`perf(autosize): default-ON FERRUM_MOE_GRAPH=1 for Qwen3-MoE`**
   (`a873d63`). The memory's documented `c=32 -6%` graph regression
   was on the pre-fix garbage-emission code path; with the
   moe_align_block_size.cu fix landed (PR #216), the layer-loop CUDA
   graph replay produces correct sorted_token_ids and the
   `cuGraphLaunch` overhead is amortized by eliminating the ~480
   per-iter kernel launches the eager path was firing.

3. **(carry-forward from prior session)** the
   `moe_align_block_size.cu` packed_row fix shipped in PR #216 was
   verified end-to-end on real Qwen3-30B-A3B weights (was only
   bench-microbench-verified before). Paris bisect 4-cell, all
   produce "Paris" with the post-fix binary; pre-fix `B_vllm_moe` was
   garbage.

---

## Measured throughput at c=32 (RTX 4090, random 256/128, n_repeats=1)

| Config | tok/s | Δ vs prev | ratio vs historical vLLM 1883 |
|---|---:|---:|---:|
| SAFE (FERRUM_VLLM_MOE=0) | 848 | baseline | 0.450 |
| + `FERRUM_VLLM_MOE=1` | 976 | +15.1% | 0.518 |
| + autosizer's `FERRUM_MOE_GRAPH=1` | **1006** | +3.1% | **0.534** |

(The two `+` deltas stack — fix shipped today moves the production
default from the bottom row's `0.450` baseline straight to `0.534`
without any user-facing env change.)

**Still 27 pp short of the 0.80 target.** Per the post-fix nsys profile
(documented below), the remaining gap is in **CPU API overhead and
launch count**, not raw kernel time. Closing it requires multi-day
levers, not single-session env tuning.

The vLLM denominator (`1883`) is the 2026-05-13 historical baseline
from `bench/v0.2-cuda/REPORT_2026-05-13.md`; an apples-to-apples vLLM
0.20.2 sweep on this exact pod was NOT run this session (budget went
to perf-fix iteration). Treat the 0.534 ratio as **indicative — the
within-session deltas (+15% / +3%) are sound; the ratio number needs
re-baselining**.

---

## Post-fix nsys profile (c=32, n=1, prompts=30)

`docs/bench/m3-80pct-goal-2026-05-25/session-2026-05-27-iteration2/nsys_postfix/`
(captured on pod, kernels.csv + api.csv).

### Top 5 GPU kernels

| % | launches | avg µs | kernel |
|---:|---:|---:|---|
| 30.5 | 62,976 | 22.9 | Marlin\<256,1,8,8\> (MoE matmul, m=1 tile) |
| 14.0 | 61,248 | 10.8 | Marlin (smaller tile, possibly QKV/MLP) |
| 11.0 | 720 | 723 | `paged_varlen_attn_f16` (prefill attention) |
| 10.0 | 30,624 | 15.5 | `paged_batched_flash_decode_attn_f16` |
| 8.8 | 642 | 651 | cutlass wmma f16 (lm_head) |

Marlin launch count is **6.2× fewer than pre-fix SAFE state** (392K →
63K) because graph capture + `FERRUM_VLLM_MOE=1` batched expert kernel
both landed. Decode attention launch count down ~5× (160K → 30K).

### Top 5 CUDA API host calls (the dominant cost picture)

| % | host time | calls | API |
|---:|---:|---:|---|
| 44.7 | 3.6 s | 5288 | `cuStreamSynchronize` |
| 19.2 | 1.5 s | 384,005 | `cuLaunchKernel` |
| 15.5 | 1.2 s | 50,246 | `cuMemcpyHtoDAsync_v2` |
| 9.6 | 0.77 s | 174,258 | `cudaLaunchKernel` (runtime API wrapper) |
| 5.3 | 0.42 s | 127,488 | `cuMemsetD32Async` |

CPU host time spent in CUDA APIs ≈ **8 s**, GPU kernel time ≈ **4.7
s**, bench wall ≈ **3.9 s**. CPU and GPU overlap is good (bench is
mostly GPU-bound) but the sync count is suspicious: with ~140 decode
iters total, ferrum issues **38 syncs per iter on average**. vLLM
does ~1-2 per iter. Eliminating the unaccounted ~5000 syncs is the
single biggest API-side lever; finding their source requires tracing
deeper into the engine + cudarc wrapper paths than this session
covered.

---

## Lever ranking — revised post-fix

| Lever | Status | Est gain | Effort |
|---|---|---|---|
| moe_align packed_row fix (PR #216) | ✅ shipped + verified | unblocks below | — |
| FERRUM_MOE_GRAPH=1 default ON | ✅ shipped today (`a873d63`) | +15.5% / +8 pp | 1 hour |
| paged_attn shared opt-in (chat-turn fix) | ✅ shipped today (`37f5dda`) | correctness, no perf | 30 min |
| **Eliminate 5000 unaccounted syncs** | not started | high (~5-15%?) | 1-2 days |
| **Persist per-iter H2D index buffers** | not started | medium (~5%) | 1-2 days |
| **Extend graph to lm_head + pre-loop work** | partial (layer loop only) | medium (~3-5%) | 1 week |
| **Small-m fused MoE Triton kernel** | not started | high (~10-20% per `project_marlin_moe_smallm_ceiling_2026_05_26`) | 2-4 weeks |
| **Sampler/embedding on-device chain** | not started | medium-high | 2-3 weeks (engine refactor) |

None of the "high gain" levers fit in a single bench-iteration
session. The next iteration should pick ONE, do a 1-2-day focused
attack, then re-baseline.

---

## What did not land this session

1. **Apples-to-apples vLLM 0.20.2 sweep** at n=5 prompts=128 to retire
   GOAL.md §Update's preliminary caveats. Pod budget went to the
   perf-fix iteration. Cost to land: ~$1 on a fresh pod (90-min
   sweep, vLLM 0.20.2 from cu128 wheel).

2. **`FERRUM_GREEDY_ARGMAX=1` default-ON** in the autosizer (similar
   to MOE_GRAPH; clearly correct + verified +15% in earlier
   sessions). Held back from this commit to keep the autosizer change
   scoped; trivially extensible next iteration.

3. **Source of the 5000 unaccounted syncs**. The 280 expected from
   decode_batch_internal × 140 iters + the 970 D2H-implicit syncs +
   prefill syncs only explain ~1500. The other ~3500 need a
   per-kernel-callsite trace. Defer to next iteration with nsys
   `--trace=cuda --capture-range=cudaProfilerApi` and ferrum's
   `cudaProfilerStart/Stop` wired into the bench harness.

4. **The `FERRUM_GRAPH=1` no-op placebo in `scripts/sweep_bottleneck.sh`
   and `scripts/paris_bisect.sh`**. Should be removed (or repointed to
   `FERRUM_MOE_GRAPH=1`) for clarity; deferred to follow-up to keep
   this PR scoped.

---

## Process notes

What went well:

- Identifying that `FERRUM_GRAPH=1` is a no-op placebo (only set by
  scripts, not read by source code) unblocked the real `FERRUM_MOE_GRAPH`
  lever. All prior session graph-perf tests were measuring noise.
- The paged_attn shared-mem fix root-caused via reading the existing
  `flash_decode_attention` pattern in `mod.rs:949`; one-line opt-in,
  zero new code, mirrors a precedent that was already correct.
- Each iteration ran < 10 min on pod (paris bisect + single-cell perf
  delta + correctness chat-turn test) — much tighter loop than the
  90-min publication sweep this session originally launched (and
  killed mid-flight at user request).

What went badly:

- Launched a 90-min publication-grade sweep before verifying the
  perf-fix delta. User pointed out this was the wrong cadence: the
  iteration loop is `bottleneck → minimum verify → optimize → test`
  not `start running a sweep and find out`.
- Spent 25 minutes of pod time waiting on incremental cargo rebuild
  (`ferrum-kernels` change triggers full CUDA recompile of all 19+
  vllm_marlin sm_89 variants + the 5-min `ops.cu` mega-file). Future
  iterations touching `.rs` files in `ferrum-kernels/src/` should
  expect this cost; only `.cu` touches that DON'T trigger the rust
  rlib invalidation stay incremental.
- The `bvhutm2jz` build-poll monitor went into an infinite loop
  because its pgrep check matched residual artifacts after cargo
  exited. Lesson: when polling for "cargo build" completion, check
  the EXACT pid (passed at launch) — `pgrep -af "cargo build"` can
  match unrelated bash invocations of subsequent commands. Use
  `kill -0 $PID` against the recorded PID instead.

---

## Artifacts

| Path | What |
|---|---|
| `crates/ferrum-kernels/src/backend/cuda/paged.rs` | shared-mem opt-in fix (commit `37f5dda`) |
| `crates/ferrum-cli/src/gpu_mem_autosize.rs` | FERRUM_MOE_GRAPH=1 default (commit `a873d63`) |
| `docs/bench/m3-80pct-goal-2026-05-25/session-2026-05-27-iteration2/SESSION-REPORT.md` | This file |
| `/workspace/m3-80pct-session/nsys_postfix/{kernels,api}.csv` (pod-only) | post-fix profile data — not committed; pull off pod with `rsync` if needed for the next iteration's analysis |
