# Qwen3-30B-A3B decode — bottleneck analysis (post PR #62)

**Date**: 2026-05-01 · **Hardware**: M1 Max 32 GB (32 GPU cores)
**Model**: Qwen3-30B-A3B-Q4_K_M.gguf (17 GB)

## Where we are

| Engine | tg128 t/s | per-token |
|--------|-----------|-----------|
| ferrum (post #58 wsum widen, #62 prepare) | 38 | 26.3 ms |
| llama.cpp (brew, b6940-ish) | 52 | 19.2 ms |

**Gap**: 7 ms/token (27% behind). Decode is at 73% of llama.cpp.

## Per-stage breakdown (FERRUM_DECODE_OP_PROFILE=1, sync-inflated 6×)

```
[decode-prof] total=155 ms | attn=24 (15%) | moe=102 (66%)
              [route=14 gate=20 up=18 silu=11 down=22 wsum=14]
              | embed=0 fnorm=0 lmhead=4 other=26 (16%)
```

De-inflated by 6× (sync overhead between stages):

| Stage | per-token | what it is |
|-------|-----------|------------|
| attn  | 4 ms (15%) | flash_attn + qkv post-ops (incl. KV append) |
| route | 2.4 ms (9%) | router GEMV + topk softmax |
| gate  | 3.4 ms (13%) | gate Q4_K MoE-id GEMV (1 dispatch, 8 slots) |
| up    | 3.0 ms (12%) | up Q4_K MoE-id GEMV (1 dispatch, 8 slots) |
| silu  | 1.8 ms (7%) | silu_mul_stacked (already 256-thread TG) |
| down  | 3.7 ms (14%) | down Q4_K MoE-id GEMV (1 dispatch, 8 slots) |
| wsum  | 2.5 ms (10%) | wsum_residual_norm_stacked (already 256-thread TG) |
| other | 4 ms (15%) | QKV proj + O proj + lm_head + residuals |
| **total** | **~25 ms** | matches 38 t/s ≈ 26.3 ms/token |

## What's already done

- PR #58 wsum widen (32→256 threads): +36% (28→38 t/s)
- PR #55-#57 GPU-side router topk (no host topk roundtrip)
- PR #50 zero-copy GGUF mmap (decode path no longer paging-bound)
- PR #62 `DecoderOnlyLLM::prepare` warms scratch + KV + Metal pipelines (clean prefill timing)

## Null results — saving so the next person doesn't repeat

- **Widen `moe_router_topk_softmax_f32` 32 → 256 threads** ([2026-05-01-route-topk-widen-null-result.md](2026-05-01-route-topk-widen-null-result.md)): null result. Router does <1 µs of compute per call; dispatch overhead dominates regardless of thread count. Wsum widen worked because that kernel had real per-element work (~2048 elem × 10 FLOPs).
- **Fuse gate+up+silu into one Q4_K MoE-id kernel** ([2026-05-01-gate-up-silu-fuse-attempt.md](2026-05-01-gate-up-silu-fuse-attempt.md)): null result, but bench environment was paging-affected at the time (3.8 GB swap, mid-bench growth). The win, if any, is small — see "Why fusion alone won't close the gap" below.

## Where the 7 ms gap likely lives

### Hypothesis 1 — Concurrent dispatch (HIGH confidence, untested in ferrum)

llama.cpp creates compute encoders with `MTLDispatchTypeConcurrent` and tracks per-buffer dependencies, inserting `memoryBarrierWithResources` only when the next op actually reads/writes a prior op's output. Ferrum uses default `MTLDispatchTypeSerial` — every dispatch waits for the prior to fully drain on the GPU even when there's no data dependency.

In Qwen3-MoE per-layer body, `gate` and `up` are independent (same input `norm_out`, disjoint output buffers). Under serial mode they run back-to-back; under concurrent mode the GPU can overlap them.

**Estimate**: gate+up are memory-bound (~6.3 MB weights each, ~70 µs/layer in serial). Running concurrently cuts the pair from 140 µs → ~84 µs (limited by combined 12.6 MB / 150 GB/s ≈ 84 µs). Per layer: 56 µs saved. × 48 = **2.7 ms**.

Plus there's a hidden inter-dispatch cost in serial mode that's not present under concurrent: pipeline stall between every kernel that doesn't actually conflict. With ~500 dispatches/token, even 10 µs of avoidable stall per pair adds up to ~5 ms. So **realistic save: 2-5 ms** of the 7 ms gap.

**Implementation** (sketch):
1. Switch encoder creation to `compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent)`.
2. Track per-dispatch read/write buffer ranges in `MetalContext`.
3. Each new dispatch helper checks for read/write conflicts with pending writes; emit `memory_barrier_with_resources(&conflicting)` only when needed.
4. Reset tracking on `compute_encoder_end` / `flush`.

This is a 200-400 LOC change touching `metal.rs` + every dispatch helper. Risk: medium (correctness bugs surface as silent numerical drift). Test plan: full Group A bench with prompts that already produce known-good outputs.

llama.cpp reference: `ggml-metal-device.m:462-475` (encoder init), `ggml-metal-ops.cpp:147-173` (concurrency_check + barrier).

### Hypothesis 2 — gate+up tensor pre-merge (LOW confidence, marginal)

llama.cpp's `build_moe_ffn` has a "merged gate_up" path (`gate_up_exps`) that runs ONE `mul_mat_id` producing `[2*n_ff, n_expert_used, n_tokens]`, then splits via view (no copy) into gate and up. Saves 1 dispatch per layer.

But Qwen3-30B-A3B's GGUF stores `ffn_gate_exps` and `ffn_up_exps` SEPARATELY — llama.cpp falls back to the same 2-dispatch path we use. So this isn't where their decode wins come from on this model. (Could still be a win for ferrum if we pre-merge at GGUF load time, but only ~1 ms upper bound on dispatch overhead at our tiny per-call cost.)

### Hypothesis 3 — Widen GEMV TG to NSG=8 (MEDIUM confidence)

Current Q4_K MoE-id GEMV: TG = (32, 2, 1) = 64 threads, 2 simdgroups × N_R0=2 rows = 4 outputs per TG. For N=768 this is 192 TGs × 8 slots = 1536 TGs.

Widening to NSG=8 (256 threads/TG, 16 outputs per TG): 48 TGs × 8 slots = 384 TGs.

Does NOT reduce per-thread work (each thread still does 2 ib-blocks × 2 rows = ~80 ALU ops × 2). Does reduce scheduling rounds: 1536 TGs / 256 hardware-concurrent = 6 rounds vs 384 / 256 = ~1.5 rounds. Less per-round overhead.

llama.cpp uses NSG_Q4_K=2 (same as us). They didn't bother widening — implies the win, if any, is small. **Estimate: 0-1 ms save.**

### Hypothesis 4 — Attention path (LOW remaining)

attn = 4 ms includes flash_attn + qkv post-ops + O proj. PR #47-#48 already fused split_qkv + 3× qk_norm + rope + KV append. flash_attn itself has been touched (PR #53). Likely close to bandwidth-bound at this point.

## Recommendation

**Concurrent dispatch encoder is the single largest leveraged change remaining.** It's structural — would also benefit attention path, prefill, and any other dispatch-heavy path — and has a clear reference in llama.cpp.

Order of work:
1. Implement concurrent encoder + per-buffer dependency tracking (this PR or its own).
2. Re-run Group A bench. If 30B-A3B tg128 lands at 45-48 t/s, gap is essentially closed.
3. Only then revisit GEMV widen / gate-up fusion as polish.

If the concurrent encoder change doesn't move the needle, that itself is a strong negative signal that points investigation toward the per-kernel ALU/bandwidth efficiency (Metal frame capture + Instruments profiling).

## Why fusion alone won't close the gap

Per-call dispatch overhead in ferrum is small (sticky compute encoder, no per-call commit). Fusing 3 dispatches into 1 saves ~2 dispatches × ~10-30 µs each × 48 layers = 1-3 ms upper bound. Doesn't account for the 7 ms gap.

The gap is structural — sequential vs parallel scheduling — not per-kernel.
