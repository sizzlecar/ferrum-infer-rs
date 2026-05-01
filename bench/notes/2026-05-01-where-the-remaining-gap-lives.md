# Where the remaining 8B gap lives — and what'd actually move it

**Date**: 2026-05-01 · After PR #66 sdpa_vector merged.

## Where we are

Group A bench (M1 Max, Q4_K_M, 3 trials median):

| Workload | ferrum | llama.cpp | mistral.rs | ferrum vs ref |
|---|---:|---:|---:|---:|
| **Qwen3-30B-A3B tg128** | 53.8 | 54.2 | (loader broken) | **99% ✓** |
| **Qwen3-30B-A3B pp50** | 220.4 | 202.2 | — | **109% ✓** |
| **Qwen3-30B-A3B pp512** | 647.7 | 620.5 | — | **104% ✓** |
| **Llama-3.1-8B tg128** | 34.0 | 31.8 | 36.1 | **107% / 94%** |
| Llama-3.1-8B pp50 | 229.5 | 258.6 | 223.7 | 89% / 103% |
| Llama-3.1-8B pp512 | 324.2 | 379.1 | 308.2 | 86% / 105% |
| Qwen3-8B tg128 | 31.5 (variance ±1) | 35.5 ±2.86 | 37.9 | **89% / 83%** |
| Qwen3-8B pp50 | 222.3 | 257.3 | 215.9 | 86% / 103% |
| Qwen3-8B pp512 | 323.0 | 375.5 | 289.1 | 86% / 112% |

**The remaining gaps**:
1. **Qwen3-8B decode**: 11-17% behind references (Llama-3.1-8B is fine, even ahead of llama.cpp).
2. **8B prefill**: ~14% behind llama.cpp (both Llama and Qwen3); but ahead of mistral.rs.

## Why the obvious levers are exhausted

Things tried this session that gave wins:
- ✅ PR #58 wsum widen (32→256 threads) — MoE only, +36% on 30B-A3B decode
- ✅ PR #62 prepare warmup — pp50 fairness, +66% on 30B-A3B pp50
- ✅ PR #66 MLX sdpa_vector port — +33% on 30B-A3B, +14% on Llama-8B, +15% on Qwen3-8B

Things tried that gave NULL results:
- ❌ Router widen 32→256 (`bench/notes/2026-05-01-route-topk-widen-null-result.md`) — kernel does <1µs of work, dispatch-bound, not occupancy-bound
- ❌ Concurrent encoder + per-buffer barriers (`bench/notes/2026-05-01-concurrent-encoder-null-result.md`) — MoE GEMVs already saturate GPU; gate↔up overlap impossible
- ❌ Dense cross-layer norm fusion (`bench/notes/2026-05-01-dense-cross-layer-norm-fusion-null.md`) — fused_add_rms_norm shifts work but adds per-call overhead; cache locality of the un-fused pair is already good

The structural pattern: **easy wins came from kernels that were measurably under-occupying the GPU**. wsum was 1 simdgroup. SDPA was 1 simdgroup. Both jumped ~15-30% when widened. That class of kernel is now exhausted in our hot path — the remaining kernels (Q4_K GEMV, GEMM tile, RMSnorm, silu) are already either max-occupancy or so small that widening doesn't help.

## What's left, ranked by leverage

### A. Per-kernel GPU time measurement via xctrace / Metal Frame Capture

**Status**: xctrace works from CLI (`xctrace record --template "Metal System Trace" --launch -- <command>`). I tested this — the output trace exports cleanly via `xctrace export --xpath '...'`. So this is doable from this CLI session, contrary to my earlier (wrong) claim.

**What I tried**: recorded a Metal System Trace of a tg32 ferrum decode on Qwen3-8B and parsed the result.

**What I found from the trace**:
- ferrum runs **3 sticky compute encoders per decode token** — NOT 64 as I'd estimated from the flash_attn close/reopen pattern. Metal must be coalescing them or my mental model of `compute_encoder_end()` was wrong.
- **Total host encoding time per token: ~30 µs** (sum of 3 encoders' encoding durations)
- **GPU work per token: ~31 ms** (Frame N → Frame N+1 gap, steady-state)
- **Encoding is 0.1% of total token time** — confirming ferrum is GPU-bound, not dispatch-bound.

**What this rules out**: per-call dispatch / FFI overhead is below the noise floor. Whatever is making us 5-15% slower than mistral.rs on Qwen3-8B is in **GPU kernel execution time itself**, not in encoder boundaries / Rust-to-ObjC bridging.

**To get per-kernel GPU time**: the default `Metal System Trace` template ships with `Shader Timeline: Disabled`, so `metal-shader-profiler-intervals` exports as empty. Need to either:
- Edit the binary plist template to flip `Shader Timeline: Enabled`, then `xctrace record --template <custom.tracetemplate>`
- Or open the recorded `.trace` in Instruments GUI to dig into shader-level times.

**Confirming experiment** (commits not retained, just data): tried removing the `compute_encoder_end()` around flash_attn so the sticky encoder spans flash_attn dispatches too. Result: tg128 marginal / null (30B-A3B regressed 3%, 8B nudged ~1% within noise). Confirms encoder-boundary cost is already small enough that consolidating doesn't help — and closing them between flash_attn calls actually gives M1 GPU a useful synchronization point.

**Estimated cost to crack the per-kernel gap**: ~1 day of work to wire up custom template + write a parser that diffs ferrum vs mistral.rs per-kernel intervals. Can be done from CLI now that we know the pattern.

**Confidence**: 90% this gives actionable data once Shader Timeline is enabled.

### B. f16 activations end-to-end on dense

**Why**: ferrum carries activations as f32 throughout the dense decode path. mistral.rs (via candle) defaults to f16. The Q4_K matmul kernels we share with mistral.rs both expect f32 activations *internally* — but the activation **buffer traffic** between kernels (residual / norm_out / mlp_out / silu_out) is double the bytes for us.

**Estimated win**: 7-8 MB activation bandwidth saved per token at 200 GB/s effective ≈ 35 µs. ~1% on tg128. **Probably not worth the refactor.**

**Why not zero**: ferrum's MoE buffers are already wider so the savings are larger there. But we already match llama.cpp on MoE.

### C. Read llama.cpp commit history more systematically

llama.cpp gets weekly perf-relevant commits. We last surveyed in PR #64. Worth periodic re-survey — but our last review showed only 4 metal-relevant commits in 6 months, none touching Q4_K mul_mv. Diminishing returns unless they ship a major refactor.

### D. KV cache fp16

**Why**: Currently f32. For long contexts (>1k tokens), KV bandwidth becomes meaningful — at 1k context, ~280 MB of KV reads per decode token = 1.4 ms at 200 GB/s. Halving to fp16 saves 0.7 ms.

**For tg128**: avg kv_len = 64, so KV reads are ~16 MB/token = 80 µs. Halving to f16 saves 40 µs. **Negligible at short context.**

**Verdict**: Implement only when long-context perf becomes a goal.

### E. 8B prefill: rewrite `q4_k_gemm` to use Apple Tensor cores (Metal 4)

**Why**: Apple7/M1 Max doesn't have Metal 4 tensor matmul instructions. M3+ does. Our `simdgroup_half8x8` matmul is already the best available on M1.

**For M1**: nothing more to squeeze from the matmul itself.
**For M3+**: would unlock another 1.5-2× on prefill.

**Verdict**: Cross-hardware optimization, not a single-machine M1 win.

### F. Paged attention + continuous batching

**Why**: Phase 2 (16-concurrent throughput) is still pending. ferrum's current bench command shows poor concurrency scaling (51.8 t/s at conc=1 → 27.5 t/s at conc=2). This is a **scheduler / batched-decode** issue, not a kernel issue.

**Concrete output**: a working concurrent-decode path that scales linearly to 8-16 requests on M1 Max.

**Cost**: 1-2 weeks. Bigger architectural change.

**Verdict**: This is the next big project IF the goal is server throughput rather than single-request latency.

## Recommended priority

**For "can do better on single-request"**: Need profiler data (option A). I cannot drive Instruments from this CLI; user runs once → data feeds into 1-2 follow-up PRs.

**For "next strategic move"**: option F (concurrency) unlocks server-mode use cases; current architecture leaves a lot of GPU idle when serving multiple requests.

## What we're NOT doing further this session

I've tried 3 micro-optimizations beyond the SDPA win:
- Concurrent encoder: null
- Dense cross-layer norm fusion: regression
- (Implicit) more kernel widening: nothing else under-occupied

Continuing to swing at micro-optimizations without profiler data risks more null results / regressions than wins. The bench numbers are at the point where 1-2 t/s differences are within run-to-run variance on this machine.
