# M3 80% goal — session 2026-05-27 iteration 3 (perf loop)

**Date:** 2026-05-27
**Pod:** Vast contract 38091913, RTX 4090 sm_89, driver 580.126.09 (CUDA 13.0),
nvcc 13.0.48 (`nvidia/cuda:13.0.0-devel-ubuntu24.04`)
**Branches shipped:** `iter1-fix-decode-op-profile-default` (PR #218),
`iter2-paged-split-k-heuristic-c32` (PR #219)
**Budget used:** ~$2.6 of $2.10 (slightly over due to iter-3..5 diagnostic loop)

---

## TL;DR — what shipped

Cumulative gain on Qwen3-30B-A3B-GPTQ-Int4 c=32 (RTX 4090, random 256/128, no
`FERRUM_DECODE_OP_PROFILE`):

| Step | tok/s | Δ | Cumulative | Ratio vs vLLM 1883 |
|---|---:|---:|---:|---:|
| iter-2 SESSION-REPORT baseline | 894.3 | — | — | 0.475 |
| **iter-1**: drop `FERRUM_DECODE_OP_PROFILE` from sweep_bottleneck.sh | 979.3 | +9.5% | +9.5% | 0.520 |
| **iter-2**: paged-attn split-K=4 at c≥32 waves≥8 | 1016.0 | +3.7% | +13.7% | 0.540 |

vLLM denominator (1883) is the historical 2026-05-13 baseline; not re-baselined
this session.

---

## Iteration loop (user workflow framework)

Each iter ran the same 5-step loop:

1. **定位瓶颈** — read prior nsys profile / instrument source
2. **最小验证 (native cuda)** — microbench-style direct measurement before refactor
3. **修改** — code change
4. **正确性测试** — paris bisect ("The capital of France is" → "Paris")
5. **性能测试** — c=32 single-cell ferrum bench-serve

If perf doesn't improve, revert and pick another bottleneck.

---

## iter-1 — `FERRUM_DECODE_OP_PROFILE` was a measurement artifact

**Localization** (source read): `scripts/sweep_bottleneck.sh:76` set
`FERRUM_DECODE_OP_PROFILE=1` unconditionally. That flag enables `stage_end`
closures in `qwen3_moe.rs::forward_layer_batched_decode` (L2849, L3066, L3290,
L3637) which call `B::sync(ctx)` to time stages. Each call inserts a
`cuStreamSynchronize`, serializing CPU launch-ahead against GPU drain.

**A/B (no native cuda — direct production A/B is the minimum verification
here)**:

| regime | A (PROFILE=1) | B (no flag) | Δ |
|---|---:|---:|---:|
| no VLLM_MOE / no MOE_GRAPH (eager) | 723.4 | 837.6 | **+15.8%** |
| VLLM_MOE=1 + MOE_GRAPH=1 (prod) | 894.3 | 979.3 | **+9.5%** |

Larger Δ in the eager regime confirms the cause: graph replay skips
`forward_layer_batched_decode` entirely (so stage_ends don't fire), but
warmup/recapture iters still pay the sync cost.

**Fix**: make the env opt-in via `FERRUM_PROFILE_STAGES=1`. Single-line script
edit. Paris bisect on flag-off path: outputs "The capital of France is
**Paris**." (correctness preserved).

**Reframes iter-2 SESSION-REPORT's "5000 syncs / 38 per iter" lever claim** —
that count was instrumentation overhead from this exact flag, not a real
production sync rate. True production sync/iter at steady state is single-digit.

---

## iter-2 — paged-attn split-K heuristic broken at c=32

**Localization** (env sweep on iter-1 B regime):

| `FERRUM_PAGED_FLASH_SPLITS` | tok/s | Δ vs default |
|---|---:|---:|
| 1 (current default at kv≤768) | 996.9 | — |
| 2 | 946.5 | -5.0% |
| **4** | **1035-1046** | **+4-5%** |
| 8 | 989.8 | -0.7% |

The historic comment `// c=32 splits=2 → +5%` (paged.rs:270) was measured
under sync-instrumented bench (DECODE_OP_PROFILE=1). The instrumentation
shifted the relative ordering of split values.

**Fix**: change the heuristic so wave-saturated grids with `waves >= 8`
(c≥32 on 32-q-head models like Qwen3-30B-A3B) pick splits=4 not 1. c=16
(waves=4) keeps splits=1 unchanged. c<sat-threshold uses the existing
else branch.

**Verify**: paris OK, c=32 = 1016 tok/s (+3.7% over iter-1 B regime baseline
979.3). Below env-forced 1035-1046 because heuristic only fires when batch
fills to 32, and the bench has mixed prefill/decode transitions where m<32.

---

## iter-3, iter-5 — H2D source localization (no fix shipped)

iter-3 instrumented per-dtype atomic counters on `B::write_typed` and
`from_slice_typed` in cuda backend. Bench (clean regime) result:

```
[h2d-trace] total=35000 write_u32=34345 (98.1%) embed_ids=642 (1.8%) others≈0
```

98% of H2D events are `write_typed::<u32>` writes. Initial hypothesis: per-iter
overhead in decode hot path.

iter-5 added per-callsite atomic counters (`fw_cl`, `pd_csq`, `pd_pos`,
`ek_*`, `pbsd_*`, `legacy`). Result:

```
[h2d-trace] caller fw_cl=30480 (88%) pd_csq=192 pd_pos=192
            ek_cl=2452 ek_bt=1344 pbsd_*=0 legacy=0
```

`fw_cl` (forward_layer L1414 per-layer `cl_buf` write) = 88% of u32. But
`pbsd_*=0` was a surprise — `populate_paged_batch_scratch_decode` was never
called.

iter-5 hypothesis: the `m==1` shortcut in `decode_batch_internal` L2365
diverts m=1 calls into single-seq `forward_layer` which writes
`cache.context_lens` PER LAYER (48 H2Ds/iter). 30480 ÷ 48 = 635 forward_layer
calls / bench, plausible if many iters hit m=1.

### Native cuda microbench

`scripts/microbenches/h2d_microbench.cu`:

| variant | total wall (1000 iter × 48 layers) | per-call |
|---|---:|---:|
| 48 small u32 H2Ds (current per-layer) | 64.7 ms | 1.35 µs/H2D |
| 1 shared u32 H2D (proposed fix) | 1.35 ms | 1.35 µs/iter |
| **savings** | **63.4 ms (97.9%)** | |

Per-call savings ≈ 63 µs. Across 635 forward_layer calls in bench ≈ 40 ms /
3.8 s wall = **~1% perf headroom**. Real but small.

### iter-5 fix attempt: remove the m==1 shortcut

**Result**: 1016 → **972.9 tok/s (-4.2%)**. Regression. The shortcut is a
**real optimization** — `forward_layer_batched_decode` at m=1 has more fixed
overhead (graph cache miss / scratch setup) than the per-layer `cl_buf` H2Ds
it would save. Reverted.

### iter-5 follow-up: m histogram in `decode_batch_internal`

Added bucket counter for m. Result over 100 sampled calls:

```
[m-hist] total=100 | m=1: 0 | m=2-3: 0 | m=4-7: 0
                   | m=8-15: 0 | m=16-31: 100 | m≥32: 0
```

**`decode_batch_internal` is ALWAYS hit with m in 16-31** in the
measurement phase. m=1 path is dead code during c=32 bench.

So the 30,480 `fw_cl` writes seen in iter-3 must come from the **WARMUP**
phase — 5 warmup requests × ~128 tokens × 48 layers ≈ 30,720, exactly
matching 30,480. Warmup runs sequential single-seq decode (via
`model.decode()` API, not `decode_batch`), so single-seq `forward_layer`
fires there.

**Implication**: the H2D from `fw_cl` is NOT in the steady-state perf
loop. The bench measurement phase uses m=16-31 batched, which goes through
`forward_layer_batched_decode` (0 `cl_buf` writes per layer in that path).
The lever isn't here.

iter-5 produced no code change. Both attempts (shortcut removal, cl_buf
hoist) would touch a path that doesn't materially affect bench tok/s.

---

## What's left — bigger levers (out of session scope)

Per `~/.claude/.../project_m3_launch_count_gap_2026_05_26`:

- **Graph coverage extension** — ferrum captures only the MoE layer loop;
  vLLM captures the entire forward. Extending capture to embed +
  populate_paged_batch_scratch_decode + final_norm + lm_head + post-replay
  + greedy_argmax would amortize per-iter CPU API overhead. Estimated
  effort: 2-4 weeks. Estimated gain: +30-50%.
- **Small-m fused MoE Triton kernel** — Marlin at Qwen3 shape (active=128,
  m/e=1-2) is at ~2-5% of TFLOPS peak per
  `project_marlin_moe_smallm_ceiling`. A small-m specialized kernel could
  unlock more. Effort: 2-4 weeks port.

Neither fits a single-session iter loop.

---

## Negative findings / lessons

- **`FERRUM_DECODE_OP_PROFILE` was inflating both syncs AND bench tok/s
  noise.** Re-baseline all profile-instrumented historical claims (e.g.
  iter-2's "splits=2 +5%" was wrong direction in the clean regime).
- **m==1 shortcut is real**, not a bug. Don't remove it.
- **Per-callsite instrumentation is necessary for H2D localization** —
  dispatch-side (per-dtype) counters don't distinguish prefill from
  warmup from measurement.
- **Decode hot path is GPU-bound at ~28 ms/iter**, MoE Marlin (17 ms) at
  hardware ceiling. CPU-side H2D / launch overhead is overlapped, not
  on the critical path.

---

## Artifacts

| Path | What |
|---|---|
| `scripts/sweep_bottleneck.sh` (PR #218) | DECODE_OP_PROFILE opt-in via FERRUM_PROFILE_STAGES |
| `crates/ferrum-kernels/src/backend/cuda/paged.rs` (PR #219) | split-K heuristic c=32 fix |
| `scripts/microbenches/h2d_microbench.cu` (not committed) | local microbench proving small-H2D saving is real (~1.35 µs/call) |
| (pod-only) `/workspace/m3-perf-loop/iter[3,5,5hist]/` | bench dumps, raw nsys + counter output |
