# route_topk_softmax widen 32 → 256 threads — null result

**Date**: 2026-05-01 · **Branch**: perf/route-topk-widen (abandoned)

## Hypothesis

PR #58 widened `weighted_sum_residual_norm_stacked` from 32 → 256 threads (1 SIMD → 8 SIMDs in one TG) and got tg128 28 → 38 t/s on Qwen3-30B-A3B (+36%). Naively the same fix on `moe_router_topk_softmax_f32` (also a single-TG × 32-thread dispatch) might give a similar bump.

## Test

Wrote the widened kernel: 256 threads, cross-simdgroup max/sum reduces through threadgroup memory for phase 1 (load + max), phase 2 (exp + sum), phase 4 (per-iteration argmax for top-K). Compiled, correctness-tested (Paris / Rome / Madrid), benched.

Qwen3-30B-A3B Q4_K_M / M1 Max:
- Before (32 threads): tg128 = 38.0-38.3 t/s (steady state)
- After (256 threads): tg128 = 38.0-38.4 t/s (steady state)

Within noise. **No measurable improvement.**

## Why

The wsum kernel did real work — for each of `hidden=2048` elements: weighted-sum across 8 slots + sum-of-squares accumulate + the rms scale write. ~2048 × ~10 FLOPs + 6 mem accesses per element = enough actual GPU compute that ALU occupancy mattered.

The router kernel does ~128 elements (one per expert): 4 passes of (load/max/exp/argmax). Total work per kernel call is < 1 µs of actual GPU compute. Apple's per-dispatch fixed cost (~10-30 µs encoding + scheduling on M1 Max) dominates regardless of how many threads we use to do the µs of work.

Same structural pattern, different work density → very different sensitivity to thread count.

## Lesson

**Single-TG kernel widening only helps when there's enough per-element work to saturate ALUs.** Quick way to predict the win: estimate (work_bytes_or_FLOPs / 150 GB/s) — if that's small compared to ~30 µs, the kernel is dispatch-bound and widening won't help.

Future router optimisation should be:
- Fuse with `compute_ids_tpe` (1 dispatch instead of 2; both touch `selected_ids`)
- Or fuse with the `weighted_sum_residual_norm_stacked` tail of the *previous* layer's MoE (cross-layer fusion: this layer's residual update + next layer's QKV gate routing)

Neither is a small change. Logging this null result so the next person doesn't repeat the experiment.
