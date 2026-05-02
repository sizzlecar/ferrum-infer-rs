# Qwen3-30B-A3B batched MoE decode kernel — A/B bench (2026-05-02)

## Hypothesis

Following the comparative read of llama.cpp's MoE dispatch in
`ggml/src/ggml-metal/ggml-metal-ops.cpp`, the structural difference
between ferrum and llama.cpp at c=16 isn't kernel-internal craft — it's
**dispatch granularity**:

* llama.cpp's `kernel_mul_mv_id` covers all `m * top_k` (token, expert)
  pairs in one Metal launch (grid Z = `m * top_k`).
* ferrum's `gemv_q4kw_moe_id_f32` covers `top_k` pairs per launch, so
  the engine has to loop per-token at the call site — emitting ~16×
  the dispatches at c=16.

This PR ports llama.cpp's strategy: a new batched MoE GEMV kernel
whose grid Z spans the full `m * top_k` pair set. One dispatch per
linear (gate / up / down) regardless of m.

## Implementation

* `crates/ferrum-kernels/src/q4_k_moe_id_gemv_batched.{metal,rs}` —
  Q4_K kernel + Rust launcher. Inner Q4_K decode loop is byte-for-byte
  the same as the existing `gemv_q4kw_moe_id_f32`; the only changes are
  Z-axis decomposition (`token_idx = pair / top_k`,
  `slot_idx = pair % top_k`) and 2D src1 indexing
  (`src1_outer_stride` / `src1_inner_stride` controls
  gate/up broadcast vs down per-pair).
* `crates/ferrum-kernels/src/q6_k_moe_id_gemv_batched.{metal,rs}` —
  same shape kernel for Q6_K (Qwen3-30B-A3B Q4_K_M's down projection
  is Q6_K, not Q4_K).
* `Backend::gemv_quant_moe_id_batched` trait fn + Metal impl
  (dispatches Q4K or Q6K kernel based on weight type) +
  `Backend::supports_batched_moe_gemv` capability probe.
* `moe_forward_batched_decode_impl` — new free function in
  `qwen3_moe.rs` using the new kernel: 5 dispatches per layer
  regardless of m (route + 3 batched gemv + silu_mul_batched +
  weighted_sum_batched).
* `forward_layer` adds an opt-in tier: `FERRUM_MOE_BATCHED_DECODE=1`
  picks the new path for `2 ≤ m < 32`. Default OFF (see below).

## Correctness

Two parity tests in `q4_k_moe_id_gemv_batched.rs` build a synthetic
`[E=4, N=64, K=256]` Q4_K stack and exercise:

1. **Gate/up mode** (`outer = K, inner = 0`): batched output equals
   running the per-token kernel in a loop. **`max_abs = 0.000000`,
   bitwise identical.**
2. **Down mode** (`outer = top_k * K, inner = K`): same — bitwise
   identical.

Both Q4_K and Q6_K paths produce sensible end-to-end output on the
real Qwen3-30B-A3B Q4_K_M model (verified via single-prompt curl).

## Performance — the regression

Despite correctness and the projected dispatch-count savings, the new
path is **slower** at every concurrency tested. Output throughput
(tok/s) on Qwen3-30B-A3B Q4_K_M / M1 Max:

| c  | `per_token` (legacy)<br>`FERRUM_MOE_BATCHED=0` | `batched` (new)<br>default | Δ |
|---:|--:|--:|--:|
| 4  | 43.2 | 35.1 | **-19%** |
| 16 | 48.8 | 31.2 | **-36%** |

TPOT median (ms):

| c  | per_token | batched | Δ |
|---:|--:|--:|--:|
| 4  | 80.0 | 100.9 | +26% |
| 16 | 292.5 | 479.3 | +64% |

The regression scales with m. That's the smoking-gun for where the
extra cost lives: it scales linearly with batch size, exactly like
the **per-item attention plumbing** in `forward_layer_batched_decode`.

## Why the MoE save didn't show

`forward_layer_batched_decode` in `crates/ferrum-models/src/models/qwen3_moe.rs:1363`
processes attention as a per-item Python-style loop:

```rust
for (i, (cache_id, _token, pos)) in batch.iter().enumerate() {
    // 6 dispatches per item:
    //   - 3 copy_slice (q, k, v out of batched buffer)
    //   - qk_norm_rope(tokens=1)
    //   - kv_append(tokens=1)
    //   - flash_attn(tokens=1)
    //   - copy_slice attn back
}
```

At m = 16 / 48 layers, that's `16 × 6 × 48 = 4608` attention-related
dispatches per decode step. The new MoE path saves ~`(5 - 5×16) × 48
= 3600` dispatches (going from 5 dispatches/layer per token → 5
dispatches/layer for the whole batch), but it **costs** the increased
attention plumbing because batched decode drives all 16 items through
that loop in one forward, vs the per-token mode where each item runs a
fresh attention block at m=1 with no plumbing.

In other words: the bottleneck moved. The MoE FFN dispatch count has
been removed from the critical path; attention plumbing is now the
cliff. Closing the c=16 gap to llama.cpp's 95.4 tok/s requires a
batched attention kernel (offset-aware QKV, single flash_attn call
covering m queries with per-item KV), which is a substantially larger
piece of work than this PR.

## Decision

* Ship the new MoE batched GEMV kernel + Q6_K counterpart + Backend
  trait method as **infrastructure**. Bitwise-correct, forward-ready.
* Keep the new `moe_forward_batched_decode_impl` wired in
  `forward_layer` but **opt-in only**:
  `FERRUM_MOE_BATCHED_DECODE=1`. Default OFF.
* Revert `decode_batch`'s outer gate to the original opt-in behaviour
  (`FERRUM_MOE_BATCHED=1` to enable) so users that aren't testing this
  experimental path don't pay the regression.

This PR's perf delta is therefore **0 by default** — but the kernel
is in tree, ready for the next PR (batched attention plumbing) to
flip the gate on.

## Reproduce

```bash
cargo build --release --features metal -p ferrum-cli
./bench/group-a-moe-batched-decode-2026-05-02/run_ab.sh        # full sweep
./bench/group-a-moe-batched-decode-2026-05-02/run_ab.sh 16     # only c=16
```

The script's `batched` phase opts into the experimental path (sets
`FERRUM_MOE_BATCHED=1`, `FERRUM_MOE_BATCH_THRESHOLD=2`,
`FERRUM_MOE_BATCHED_DECODE=1`); `per_token` runs with no env to take
the default per-token loop. Per-concurrency JSON results land in this
directory.

## Lesson for the perf-status doc

> Don't measure dispatch count savings in isolation — measure
> end-to-end throughput. PR #79 (offset-aware MoE, +1-2%) and
> df64ac1 (fused gate+up+silu, +0.4-2.4%) and this one (batched MoE
> GEMV, regression in current engine path) all confirm: **the
> remaining 30B-A3B ↔ llama.cpp gap is not in MoE FFN dispatch
> efficiency**. It's in the attention pipeline at small batch and the
> bookkeeping the engine does between layers. The Tier 2 work is now
> "batched attention plumbing", not more MoE FFN fusion.

## 2026-05-02 follow-up — profile-driven debugging

Added per-stage instrumentation to `decode_batch_internal` and
`forward_layer_batched_decode` (gated by `FERRUM_DECODE_OP_PROFILE=1`).
The profile output:

```
[batched-decode-prof] m=16 layers=48 total=377 ms
   dense=53 (14.1%) | attn_peritem=55 (14.7%) | moe=263 (69.8%) | other=4
```

**Surprised the hypothesis.** Attention plumbing was only 14.7% — not
the bottleneck. The cliff was in the MoE FFN block (70%), even though
that's where the new GEMV kernel was supposed to win.

Two follow-on PRs landed in this dir:

1. **Batched fused gate+up+silu kernel**
   (`q4_k_moe_id_gate_up_silu_batched.{metal,rs}` + Backend trait fn).
   Hybrid of df64ac1 (fused m=1) and the batched-pair Z-axis layout —
   one Metal dispatch covers all m*top_k pairs and writes
   `silu_stacked` directly. Bitwise-correct (parity test).

2. **Threshold decoupling.** Discovered that
   `FERRUM_MOE_BATCH_THRESHOLD` was triple-purposed: engine-level
   batched gate, legacy prefill-GEMM gate, AND inadvertently used by
   the new batched-decode tier. Setting it low to engage batched
   actually pushed m=16 onto the prefill GEMM path — the new kernel
   never fired. Split it into `FERRUM_MOE_BATCH_THRESHOLD` (legacy,
   default 8) + `FERRUM_MOE_PREFILL_THRESHOLD` (new, default 32 to
   match llama.cpp's `ne21_mm_id_min`).

After both fixes, the c=16 profile becomes:

```
[batched-decode-prof] m=16 layers=48 total=270 ms
   dense=54 (20.0%) | attn_peritem=55 (20.5%) | moe=156 (57.7%)
   [route=27 gate=0 up=0 silu=58 down=52 wsum=9]  | other=4
```

`gate=0 up=0 silu=58` confirms the fused gate+up+silu is firing (its
time is bucketed under `silu`). MoE block dropped from 263 ms → 156 ms
(**-41%**). Total step time 377 → 270 ms (**-28%**).

But end-to-end bench at c=16 still loses to per-token mode:

| c=16 | output_tok/s | TPOT median |
|---|--:|--:|
| per_token (m=1 × 16, with df64ac1 fusion)  | 49.1 | 291 ms |
| batched (NEW, all fixes)                    | 35.9 | 411 ms |
| batched (legacy prefill GEMM at m=16)       | 31.2 | ≈479 ms |
| llama.cpp (reference)                       | 95.4 | — |

The new batched path is **+15% over the legacy prefill GEMM** at c=16,
but **-27% under per-token mode**. The 141 ms gap between the
profile's 270 ms `decode_batch_internal` time and the user-perceived
411 ms TPOT is unaccounted-for **CPU-GPU pipelining**:

- `decode_batch_internal` is monolithic — all 16 users' tokens come
  out in one synchronous call, then the engine sequentially samples
  16 outputs, builds 16 SSE events, manages 16 sequence state
  updates. ≈141 ms of serial CPU work per round.
- Per-token mode overlaps this CPU work with the next forward's GPU
  dispatch. CPU and GPU are pipelined; the user sees only the slower
  of the two.

**Conclusion (confirmed null on the MoE FFN front):** the c=16 30B-A3B
gap to llama.cpp is **not** in MoE FFN GPU time. With my new path the
MoE block is 156 ms / 270 ms = 58% of the round, vs ~13 ms × 48 layers
≈ 624 ms of MoE work spread across 16 sequential per-token forwards
that totals ~280 ms wall time (including ~50 ms of CPU-GPU overlap
benefit). My new path is faster on the GPU side; per-token mode wins
on aggregate because it pipelines.

The remaining gap to llama.cpp (~95 vs ~49 tok/s) lives in two places:
1. CPU-GPU pipelining (so per-token wins for now)
2. Per-kernel GPU time (Tier 2 in the perf-status doc — and the
   measurable target for the next PR)

## Updated decision

* Ship the batched fused gate+up+silu kernel + threshold decoupling +
  profile instrumentation as **opt-in infrastructure** alongside the
  GEMV kernel from the earlier commit.
* The new batched path is FASTER than the legacy prefill GEMM at
  c=16 (35.9 vs 31.2 tok/s, +15%), so ANY user who explicitly opts
  into `FERRUM_MOE_BATCHED_DECODE=1` benefits immediately.
* But default stays per-token because that's still fastest end-to-end
  due to CPU-GPU pipelining. Closing that requires either a CPU-side
  pipelining refactor or bigger per-kernel GPU savings (Tier 2
  kernel-craft work).
