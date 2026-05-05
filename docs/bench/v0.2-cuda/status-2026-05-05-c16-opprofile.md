# c=16 op-profile — measurement that resolves the Phase-12 question

**Branch:** `bench/v0.2-cuda` HEAD `a6bb718`
**Pod:** Vast.ai RTX 4090, /workspace/models/M2 (Llama-3.1-8B GPTQ-INT4)
**Run:** 2026-05-05 ~17:19, 32 prompts × 128 output tokens at c=16, random dataset
**Throughput:** 433.7 tok/s output, mean TPOT 27.5 ms, P99 35.2 ms

## TL;DR

The Phase 11 c=1 conclusion ("matmul = 73% of iter time") **extrapolates
to c=16** essentially unchanged. The decode-iter breakdown at steady-state
m=16 (195 iterations averaged) is:

| op | time/iter (us) | % of iter | calls/iter | per-call (us) |
|---|---|---|---|---|
| matmul (Marlin INT4) | 8,323 | **65.3%** | 129 | 64 |
| attn (batched flash) | 1,718 | **13.5%** | 32 | 54 |
| qkr | 429 | 3.4% | 32 | 13 |
| norm | 530 | 4.2% | 65 | 8 |
| other | 1,332 | 10.5% | 128 | 10 |
| unwrapped | 410 | 3.2% | — | — |
| **total** | **12,745** | 100% | — | — |

vs Phase 11 c=1 single-token decode:

| op | c=1 % | c=16 % | shift |
|---|---|---|---|
| matmul | 73% | 65% | -8pp |
| attn | 5% | 13.5% | +8.5pp |
| qkr | 7.4% | 3.4% | -4pp |
| norm | 5.6% | 4.2% | -1.4pp |
| other | 4% | 10.5% | +6.5pp |

**Marlin GEMM stays dominant. The vLLM Marlin port (Phase 12 direction)
is the correct kernel-level priority.**

Earlier hypothesis (mine) that "attention takes a much larger share at
c=16 because of per-seq KV scan" was wrong. The actual c=16 attn share
is 13.5% — bigger than c=1's 5%, but still nowhere near matmul.

## Why my prediction failed

The hypothesis was: per-seq KV scan can't be amortized across the batch,
so attention scales linearly with c, while GEMM amortizes the K dimension
across m=16. That math:
- per-iter GEMM: scales 3-4× from c=1 (sub-linear)
- per-iter attn: scales 16× from c=1 (linear)
- → GEMM share drops, attn share rises

What I missed: ferrum's `batched_decode_attention.cu` launches a grid of
`(num_q_heads, m)` blocks with **single-block-per-(head,seq) attention**,
no split-K. For Llama-8B kv_len ≈ 384 + GQA (32 q_heads → 8 kv_heads),
each block walks ~48 KV positions × 8 warps cooperatively. RTX 4090 has
1 TB/s memory bandwidth; the per-token KV read budget (3.2 GB) takes
~3.2 ms of pure memory time. Ferrum measures 1.7 ms total attention →
implementation is actually USING 53% of theoretical bandwidth. Not bad.

The "per-seq KV scan" cost is real but small in absolute terms; my
bandwidth math was mis-scaled (forgot GQA reduces kv_head reads 4x and
that block-cooperative warps walk efficiently).

## What this means for the v0.2-cuda roadmap

**Stays:**
- Phase 12 (vLLM Marlin port). At m=16, ferrum's Marlin call is 64 us;
  vLLM's estimated 32 us. A clean 2× would cut matmul from 65% → 32.5%
  → iter time 12.7 ms → 8.5 ms → ~50% throughput improvement.

**Drops (off the recommendation list):**
- Batched paged-attn split-K (the direction I was about to recommend
  before the data landed). Even completely eliminating attention only
  saves 1.7 ms / 12.7 ms = 13% iter speedup.

## NEW finding: huge engine-level overhead outside decode

Throughput math doesn't reconcile with decode-iter numbers:

- 199 iters × 12.7 ms = **2.53 seconds of pure decode time**
- Tokens produced = 195×16 + 4×15 = **3,180 tokens**
- Pure-decode rate: 3,180 / 2.53 = **1,257 tok/s**
- Measured throughput: **433 tok/s**
- → **65% of wall-clock time is outside the timed decode iter**

What lives outside the timer (top of `decode_batch_internal` to bottom):
- scheduler.next_batch / iteration_lock
- per-request sampling (RNG state, top-k/p, repetition penalty)
- to_vec on batch_logits (sync device→host transfer)
- HTTP / SSE chunking per request per token
- prefills interleaved with decode (this run had 32 prompts → ~2 prefills)

Even an ideal 2× Marlin port won't move the needle past ~600 tok/s if
the wall-clock is 65% engine. **The next investigation should profile
the engine path, not the kernels.**

A start: time the section between `batched_iter_t0`'s end (after lm_head
sync) and the next iter's `batched_iter_t0` start. Anything > 1 ms there
is engine-overhead per token.

## Methodology

Instrumented `decode_batch_internal` + `forward_layer_batched_decode`
+ `_post_attn` (commits `f34a005`, `a6bb718`) with B::sync()-bracketed
timers around 5 op categories, drained per-iter so each `[batched-op-profile]`
line is independent.

Caveat: B::sync() barriers serialize the GPU pipeline. Real eager
decode with no instrumentation is faster (probably 8-10 ms/iter at
m=16 on this pod). The percentages are still valid; the absolute
12.7 ms is inflated by ~2-4 ms of synthetic syncs.

To verify: turn off `FERRUM_DECODE_OP_PROFILE` and re-bench — the
measured throughput should land closer to the previous Phase 8+9
number (484 tok/s on the prior pod, similar HW).
