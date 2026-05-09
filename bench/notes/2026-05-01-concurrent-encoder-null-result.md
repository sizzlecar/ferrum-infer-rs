# Metal concurrent encoder — null result

**Date**: 2026-05-01 · **Branch**: `perf/metal-concurrent-encoder`
**Hypothesis from**: `bench/notes/2026-05-01-decode-bottleneck-analysis.md` (PR #64)
**Hardware**: M1 Max 32 GB · **Model**: Qwen3-30B-A3B-Q4_K_M

## Hypothesis

Decode bottleneck analysis pointed at the serial Metal compute encoder as the likely 7 ms/token gap vs llama.cpp:
- llama.cpp uses `MTLDispatchTypeConcurrent` + per-buffer barriers
- ferrum uses default serial encoder (every dispatch waits for prior)
- gate↔up are independent (same input, disjoint outputs); under concurrent dispatch they should overlap
- Estimated win: 2-5 ms/token

## What was built

- Switched encoder creation to `compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent)` when `FERRUM_METAL_CONCURRENT=1` is set
- Added `MetalContext::barrier_for_inputs(&[input_buffers])` helper — emits `memoryBarrierWithResources` in concurrent mode, no-op in serial
- Wired barriers into all dispatch sites in the decode hot path: route_topk_softmax, gemv_quant_moe_id, silu_mul_stacked, weighted_sum_residual_norm_stacked, weighted_sum_residual_stacked, gemm, gemm_quant, fused_add_rms_norm, rms_norm, split_qkv*, qk_norm_rope, residual_add, kv_cache_append, layer_norm, gelu, add_bias, transpose, scaled_add_inplace (~25 sites)
- Verified correctness: same generated text in serial and concurrent modes ("The capital of France is Paris…")

## Bench results (3 trials each)

Qwen3-30B-A3B-Q4_K_M / FERRUM_KV_CAPACITY=512 / M1 Max:

| Mode | tg128 | pp512 |
|------|-------|-------|
| Serial (default) | 37.6 / 38.0 / 37.8 (avg 37.8 t/s) | 600.8 / 603.1 (avg 602) |
| `FERRUM_METAL_CONCURRENT=1` | 37.7 / 37.6 / 38.2 (avg 37.8 t/s) | 609.2 / 607.6 (avg 608) |

Decode: identical (within ±1% noise band).
Prefill: +1% — within noise, but trends positive; not the 5-15% the hypothesis predicted.

## Why it didn't work

The MoE GEMVs (gate / up / down) each dispatch ~1536 threadgroups × 64 threads = ~98K threads. M1 Max has 32 GPU cores × ~8 hardware-resident TGs each = ~256 TGs concurrent. So a single MoE GEMV already runs for ~6 scheduling rounds at full GPU occupancy.

Running gate concurrent with up doesn't speed anything up — the GPU is already saturated by either one alone. Concurrent dispatch only helps when one kernel under-uses the GPU.

The wsum widen (PR #58) helped because that kernel was a single threadgroup with 32 threads → 1 simdgroup → severely under-occupied. Widening to 256 threads filled all 8 simdgroups in one TG. Concurrent encoder is conceptually similar (better occupancy) but the GEMV kernels are already at full occupancy so the lever doesn't apply.

## Where the 7 ms gap actually lives

Still unknown. Candidates ruled out by this experiment:
- ❌ Serial encoder forcing dispatch ordering — concurrent doesn't help because GPU is saturated.

Candidates remaining (in rough order of plausibility):
- Per-kernel ALU efficiency: do our hand-ported Q4_K kernels run at the same instruction throughput as llama.cpp's? Both source files are byte-identical, but Apple compiler has cache-versioning quirks that can produce different machine code from the same MSL.
- Buffer binding overhead per dispatch: metal-rs `set_buffer` goes through Rust→Objective-C dynamic dispatch; might have measurable cost vs a native ObjC build.
- Per-token host-side bookkeeping outside the Metal command queue (sampling, KV bookkeeping, scratch ensure).
- Some kernel ferrum hasn't ported the latest llama.cpp optimisation for (e.g., a recent `kernel_mul_mv_q4_K_f32` tweak).

Productive next investigation: Metal Frame Capture in Xcode Instruments to measure per-kernel GPU time directly. The host-side `B::sync(...)` profile we used is not granular enough — a single kernel's GPU time gets conflated with the encoder switch / barrier latency.

## Decision

The code change in `perf/metal-concurrent-encoder` is correct and behaves identically in serial mode (default), but adds maintenance burden (every new dispatch site needs `barrier_for_inputs` to keep concurrent mode safe). Without a measurable performance benefit, the right call is to **not ship**.

This note is the trail for the next person, so they don't repeat the same experiment.
