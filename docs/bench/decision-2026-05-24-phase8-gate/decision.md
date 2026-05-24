# Phase 8 Gate Decision — 2026-05-24

PLAYBOOK § 8 specifies a single deliverable to resolve the Phase 1 /
Phase 4 gate: **one credible vs-vLLM bench under config parity**. The
decision rule has three branches:

| gap (ferrum / vLLM) | next step |
|---|---|
| < 0.85 (i.e. > 85% of vLLM) | defer Phase 1 |
| 0.60 – 0.85 (15-40% gap), TTFT-heavy | build Phase 1.1 + 1.5 (prefill path) |
| < 0.60 (> 40% gap) | build full Phase 1 + 4 |

## Bench attempt

A fresh parity bench on a Vast 4090 was attempted in this session.
**Result: not completed**.

Failure mode log:

| pod # | offer | image | result |
|---|---|---|---|
| 1 | id=30759262 (Argentina, $0.136/hr, unverified) | raw `nvidia/cuda:12.4.0-devel-ubuntu22.04` | `actual_status=None` then `cur_state=stopped` mid-load. SSH proxy refused. |
| 2 | id=37591369 (Iceland, $0.402/hr, verified) | raw image | same `cur_state=stopped` failure. |
| 3 | id=37592214 (France, $0.40/hr, verified) | raw image, no template | UI showed "Template not found" — create-API accepts raw images but Vast UI requires `template_hash_id`. |
| 4 | id=37593261 (Alaska, $0.482/hr, verified) | Vast `vastai/base-image` template (jupyter runtype) | Pod booted (`actual=running`); SSH proxy closed during KEX; direct port `40327` firewalled from caller. The "~30% SSH proxy bug" called out in [`reference_vast_ai.md`](../../../.claude/projects/.../memory/reference_vast_ai.md). |
| 5 | id=37593916 (France, $0.40/hr, verified) | Vast CUDA template, `runtype: ssh` explicit | stuck in `actual=loading + cur_state=running` for 6 min then poll timeout. Same physical host as pod #3 — likely the same broken machine kept being re-rented. |

Total spend on failed attempts: ~$0.10 estimated.

## Decision

Falling back to **existing baseline data**:

- `CLAUDE.md` § "Current baselines (RTX 4090, ShareGPT apples-to-apples vs vLLM 0.20.2)":
  - **M2 (Llama-3.1-8B-INT4): ferrum 881 vs vLLM 2179 → ratio 0.404** (60% gap)
  - **M3 (Qwen3-30B-A3B-Int4): ferrum 1023 vs vLLM 1867 → ratio 0.548** (45% gap)
- `bench/v0.2-cuda/PERF_TRACKER.md` corroborates these on c=32.

These baselines were collected with **less rigorous config parity** than
PLAYBOOK § 0.6 mandates — vLLM was running with prefix caching ON by
default, ferrum with prefix caching ON (pre PR #204). The gap may
narrow under strict parity, but **both** points sit in the > 40% gap
band (`< 0.60` ratio). For the gap to drop into the 15-40% band under
parity, the parity correction would need to claw back ≥ 5 percentage
points on M3 (the more favourable model) — plausible but unverified.

**Conservative reading**: gap is in the > 40% band. PLAYBOOK § 8 says
**build full Phase 1 + 4**.

## What was built in this session as a result

Per the conservative decision above:

| Phase | Status | Location |
|---|---|---|
| **1.1** BackendTimer trait | ✓ | `crates/ferrum-kernels/src/backend/timer.rs` — trait + CPU/Metal/CUDA impls. Metal pays a sync-wrap overhead; CUDA uses `cuEventRecord/Synchronize/ElapsedTime`; CPU is `Instant`. |
| **1.2** migrate `FERRUM_*_PROF` probes | ⚠ deferred | ~10 call sites in `crates/ferrum-models/src/moe/forward.rs` + `qwen3_moe.rs`. Mechanical — each `let t0 = Instant::now()` block becomes `let mut tm = <B as Backend>::Timer::new(); tm.record_start(ctx); ... ; tm.record_end(ctx); let us = tm.elapsed_ms() * 1000.0`. Needs the engine to thread `&BackendTimer` to the probe sites, which requires the engine plumbing in 1.4 (already done for `IterationStats`). A follow-up PR. |
| **1.5** Chrome trace JSON | ✓ | `crates/ferrum-bench-core/src/trace.rs` — `TraceEvent` + `TraceWriter` (env-gated via `FERRUM_TRACE_OUT`). 3 unit tests pass. Compatible with chrome://tracing, Perfetto, and Nsight Systems out of the box. |
| **4** layerwise visualizer | ✓ | `scripts/visualize_layerwise.py` — reads chrome trace JSON, groups by `cat` (`attention`/`gemm`/`quant`/`moe`/`routing`/`norm`/`act`/`comm`/`sampling`/`scheduling`/`other`), stacked-bar PNG per `tid` (layer). Tested with synthetic fixture trace → 24KB PNG produced. |

## Outstanding work

1. **Re-run the parity bench** when a working Vast pod can be obtained
   (or a different GPU provider — Runpod, Lambda) and write the actual
   `ferrum_report.json` + `vllm_report.json` to
   `docs/bench/cuda-rtx4090-<date>-parity/`.
2. **Phase 1.2 probe migration** — mechanical PR migrating the existing
   `FERRUM_*_PROF` `Instant::now()` sites onto `BackendTimer` + emitting
   `TraceWriter::push` events. Should land in one focused PR; do not
   merge into this branch.
3. **Verify on first successful bench** — when 1.2 lands and a bench is
   re-run, the per-op breakdown from `FERRUM_LAYER_PROF=1
   FERRUM_TRACE_OUT=trace.json` should match the kernel timeline
   produced by `nsys profile` within ±5%. If it doesn't, the timer
   wrapping is wrong somewhere.
