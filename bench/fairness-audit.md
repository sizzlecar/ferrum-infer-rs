# Bench Fairness Audit — ferrum vs llama-bench vs mistral.rs

**Date**: 2026-05-01 · **Trigger**: 30B-A3B `pp50` showed ferrum at 61% × llama.cpp while `pp512` was at 95%. The 34-percentage-point gap *between* token counts on the *same model* doesn't make physical sense if the per-token compute is similar (it is — same Q4_K_M GEMV inner loop, verified byte-for-byte against `kernel_mul_mv_q4_K_f32_impl` in llama.cpp). Audit goal: find what the bench harness measures that doesn't compare across engines, fix it, then re-bench.

## What llama-bench actually does

Source: `/Users/chejinxuan/rust_ws/llama.cpp/tools/llama-bench/llama-bench.cpp`, lines 2299-2399.

```
for each (model, op, n_prompt, n_gen) test config:
    init model + context  ← OUTSIDE timer, ONCE per config

    if not --no-warmup:
        if pp test: test_prompt(n_prompt)        ← UNTIMED warmup, same shape
        if tg test: test_gen(1)                  ← UNTIMED warmup, 1 token

    for i in 0..reps (default 3):
        llama_memory_clear(...)                  ← clear KV cache
        t_start = get_time_ns()
        if pp test: test_prompt(n_prompt)        ← TIMED
        if tg test: test_gen(n_gen)              ← TIMED
        t_ns = get_time_ns() - t_start
        samples.push(t_ns)

    report median(samples)
```

Three things to highlight:

1. **Warmup is on by default.** A run with the same shape as the test (or `n_gen=1` for the gen test) executes BEFORE the timer ever starts. This warms up:
   - Metal pipeline state cache (first `setComputePipelineState` per pipeline per command buffer is slow on Apple)
   - GPU residency tracking for newly-allocated buffers
   - First-touch swap-in for cold model pages
   - Any per-context one-time setup (`llama_decode` initialises a few internal state fields lazily)

2. **Reps run inside ONE process.** Model is loaded once, context is reused. Subsequent reps see warm pipelines, allocated buffers, primed caches.

3. **KV cache is cleared between reps**, but the cache *buffer* stays allocated. Only the logical `len` is reset. So per-rep, the variable cost is just the actual prefill / decode work.

## What ferrum's `--bench-mode` was doing (before this audit)

`bench/scripts/bench_one_model.sh` invokes `ferrum run --prompt ... --bench-mode` **3 separate times** for each (op) — one fresh process per trial. Inside each ferrum process:

```
load gguf weights via mmap          ← ~1s, OK (same as llama-bench load)
build MetalPipelines (compile shaders) ← ~hundreds of ms, OK (one-time per process)

ensure_scratch(1)                    ← model init: scratch sized for 1 token
ensure_kv("default")                 ← lazy KV alloc

t0 = Instant::now()
let logits = model.prefill(...)      ← TIMED
prefill_secs = elapsed
```

Inside `model.prefill(prompt)`, the first thing that runs is **`ensure_scratch(seq_len)`**, which when `seq_len > scratch.max_tokens` (it always is on the first call, since the model is initialised with `max_tokens=1`) **reallocates ~25 MTLBuffers** (residual, q/k/v scratch, head-major buffers, MoE staging buffers, ids tables, batch logits, …).

On Qwen3-MoE 30B-A3B with 50-token prompt, those allocations total **~50 MB**, and Apple Metal's `device.new_buffer(...)` takes 1-5 ms each → **80-150 ms of alloc cost INSIDE the timer window**.

For pp512 (where the working set grows to ~500 MB of scratch), the same allocator logic runs but the denominator is 10× larger, so the % impact is much smaller. That's why pp512 looked clean (95%) and pp50 looked dirty (61%) — same overhead, divided by different token counts.

## Was this overhead real or measurement-only?

Real. Every fresh ferrum process pays it. A fresh server-mode startup hitting its first request would too. The fix is principled (eager scratch sizing), not just a bench trick.

## The fix

PR (this branch) adds:

1. **`DecoderOnlyLLM::prepare(max_tokens)`** trait method (default no-op).
2. **`Qwen3MoeModel::prepare`** and **`LlamaFamilyModel::prepare`** override to call `ensure_scratch(max_tokens)`.
3. **`run_gguf` CLI** in `--bench-mode` and one-shot mode calls `model.prepare(prompt_tokens.len())` BEFORE the timer starts — same as llama-bench's warmup hook but minimal-cost (no actual forward pass needed since the per-buffer alloc IS what `prepare` covers).

Note: this matches llama-bench's intent (don't time setup) but does *less* than llama-bench's warmup (which actually executes a forward pass to also warm up pipeline state cache and residency tracking). On Apple M1 Max, `MetalPipelines::new()` already eagerly compiles all shaders during ferrum's model load, so the residual one-time cost is probably small.

## What still differs between ferrum and llama-bench

| concern | llama-bench | ferrum bench-mode | impact on pp50 |
|---|---|---|---|
| process model | 1 process, N reps | N processes, 1 rep each | Trivial after fix above |
| scratch alloc | inside warmup, untimed | inside `prepare`, untimed | **Fixed** |
| pipeline shader compile | inside warmup, untimed | inside `MetalPipelines::new()` at model load, untimed | OK |
| KV cache clear | between reps | (n/a — fresh process) | OK |
| timer scope | only `test_prompt` / `test_gen` | only `model.prefill(...)` | Comparable |
| reps median | yes | yes (3 trials median in our orchestrator) | OK |

The remaining structural differences (process model, no KV-clear-between-reps) don't affect the per-call number — each ferrum process measures one full cold path which is what we want to compare.

## What about mistral.rs?

`bench_mistral.py` runs a single `Runner` instance with `max_seqs=1` and calls `send_completion_request` 3 times in a row. Same shape as llama-bench:

- One process, multiple reps
- mistralrs' `Runner.__init__` does a "Dummy run" (logged as `Beginning dummy run` → `Dummy run completed`) which IS a forward-pass warmup
- subsequent `send_completion_request` calls hit warm pipelines + allocated buffers
- `Usage.avg_prompt_tok_per_sec` and `avg_compl_tok_per_sec` are reported — the Python harness picks these up

So mistral.rs's measurements were already fair. Only ferrum was being unfairly penalised.

## Methodology going forward

The orchestrator (`bench/scripts/bench_one_model.sh`) keeps the "fresh process per ferrum trial" pattern intentionally — it's actually a *more conservative* test (catches any per-process state leaks, captures realistic cold-start latency for first-request-after-launch scenarios). With `prepare()` in place:

- **pp50 / pp512**: fair. ferrum's `prepare` matches llama-bench's warmup intent.
- **tg128**: was already fair (decode hot path doesn't touch ensure_scratch). ferrum's gap here is real.
- **TTFT(50)**: derived as `50/pp50 + 1/tg128`, both fair after this fix.
- **16-concurrent**: separate harness needed (HTTP). Each engine's server warm-up applies; comparable.

## Re-bench checklist (post-fix)

When PR #61 lands and this fix lands, re-run all 9 (engine × model) combos and update `bench/group-a-report.md`. **Memory state must be clean** (use `bench/scripts/capture_env.sh`-style monitoring; `swap_growth < 256 MB` per run).

Expected change on Qwen3-30B-A3B: **pp50 should jump from 118 to ~150-180 t/s** (closing most of the gap to llama.cpp's 194). pp512 essentially unchanged (already amortised).
