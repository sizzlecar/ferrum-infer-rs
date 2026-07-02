# W3 Qwen3.5 Deep Review Addendum — 2026-06-26

This addendum corrects the depth of
`W3_QWEN35_RETROSPECTIVE_20260626.md`.

The first document was a first-pass evidence index. It was useful, but it was
not a complete review of the recent 500 commits, core code, ledger, archived
artifacts, and vLLM source behavior. The user challenged that correctly.

No GPU work was run for this addendum. No live vLLM rerun was used.

## What Was Actually Reviewed

- Current branch after `git pull --rebase --autostash`: local and origin were
  already up to date at `1ec261ef`.
- Recent 500 commits, grouped by day, scope, and large diff size.
- Core Ferrum state paths:
  - `crates/ferrum-scheduler/src/implementations/continuous.rs`
  - `crates/ferrum-engine/src/continuous_engine/inner/batch.rs`
  - `crates/ferrum-kv/src/managers/paged.rs`
  - `crates/ferrum-models/src/models/qwen35.rs`
  - `crates/ferrum-types/src/auto_config.rs`
  - `crates/ferrum-cli/src/gpu_mem_autosize.rs`
- Archived Qwen35 c32 diagnostic summaries:
  - 40 `perf/bench_summary.json` files under the W3 artifact tree.
  - 3 `perf/failure_summary.json` files.
  - Historical vLLM baseline summary
    `artifacts/w3_vllm_sharegpt_baseline_20260619/vllm_baseline_summary.json`.
- vLLM source comparison from local source:
  - `../_external/vllm-v0.20.2`
  - HEAD `bc150f50299199599673614f80d12a196f377655`
  - files:
    - `vllm/v1/core/sched/scheduler.py`
    - `vllm/v1/core/kv_cache_manager.py`
    - `vllm/v1/core/single_type_kv_cache_manager.py`
- Local time ledger tail and the already recorded ledger aggregate from the
  first-pass retrospective.

What was not reviewed exhaustively:

- I did not line-read every line of the 18k-line `qwen35.rs`.
- I did not line-read every JSONL scheduler trace in all artifacts.
- I did not run a new CUDA build, GPU diagnostic, or live vLLM process.

## Commit-Stage Finding

The last 500 commits do not describe one linear fix. They describe a sequence
of adjacent state-machine repairs:

| Date | Dominant shape | Meaning |
| --- | --- | --- |
| 2026-06-17 | dynamic KV, recurrent-state lifecycle, W2 gates | The resource model started broadening beyond plain KV. |
| 2026-06-18 | 41 `feat(w3)` commits | Qwen3.5 reference/runtime pieces were assembled quickly. |
| 2026-06-20 to 2026-06-21 | many Qwen35 perf attempts plus reverts | Kernel/scratch/metadata levers were unstable and often reversed. |
| 2026-06-22 | runner, manifest, product evidence hardening | Evidence plumbing expanded faster than final performance progress. |
| 2026-06-23 | autosize, c32 concurrency, recurrent compacting | c32 admission and recurrent-state pressure became central. |
| 2026-06-24 | recurrent-state cleanup, OOM-adjacent fixes | Many paths were fixed to release KV/recurrent/model resources. |
| 2026-06-25 | scheduler/KV/recompute/backpressure churn | The loop became many small admission-state fixes plus reject artifacts. |

The largest diffs in the 500-commit window were mostly artifact and status
records. Important source-changing commits were smaller but clustered around
the same state boundary:

- `e8bea515 fix(scheduler): defer capacity-blocked prefills`
- `b8116dc4 fix(engine): release existing kv on capacity defer`
- `d62b18cf fix(kv): roll back partial paged allocations`
- `8150ca06 fix(engine): defer prefills on unified forward exhaustion`
- `2bc1e6bb fix(scheduler): let decode progress under capacity backpressure`
- `f304fd8d fix(scheduler): keep capacity backpressure through decode`
- `2f5a375e perf(engine): reopen recompute after mixed kv pressure`

This supports a narrower conclusion than the first document: the work did make
real local fixes, but paid c32 artifacts were being used as the first complete
integration test for a resource state machine that did not yet have a local
transaction invariant.

## Artifact Finding

The corrected c32 artifact summary is:

- 40 `perf/bench_summary.json` files found.
- 24 had numeric output throughput.
- All 24 numeric rows were request-clean:
  - `completed_per_run=[32]`
  - zero request errors
  - zero HTTP 500
  - zero panic
  - zero OOM mentions
- Numeric verdicts:
  - 23 `reject`
  - 1 diagnostic `keep`
- Numeric throughput:
  - min `274.637 tok/s`
  - max `488.596 tok/s`
  - average `392.886 tok/s`
- Complete p95 ITL rows:
  - n=23
  - min `48.888 ms`
  - max `70.847 ms`
  - average `59.300 ms`
- KV admission failures in numeric rows:
  - min `8`
  - max `50`
  - average `20.33`

The diagnostic `keep` was
`w3_qwen35_mixed_prefill_immediate_kv_c32_9fda1101_20260625T034421Z`:

- output throughput `488.596 tok/s`
- `completed_per_run=[32]`
- zero request errors, zero HTTP 500, zero panic, zero OOM mentions
- no p95 ITL and no `trace_stats.mixed_iterations` in the summary
- threshold floor was only `458.066 tok/s`
- `release_evidence=false`

So it is not W3 performance evidence.

Among complete reject rows, the best was
`w3_qwen35_decode_defer_waits_for_independent_kv_release_c32_f1bc7b7e_20260625T112605Z`:

- `476.311 tok/s`
- p95 ITL `48.888 ms`
- `mixed_iterations=0`
- zero request errors, zero HTTP 500, zero panic, zero OOM mentions

The mixed-recompute metric was also misleading as a primary target:

- rows with `mixed_iterations >= 64`: n=9, average `325.169 tok/s`,
  max `378.392 tok/s`
- rows with `mixed_iterations < 64`: n=15, average `433.517 tok/s`,
  max `488.596 tok/s`

That means the loop spent too much time optimizing a proxy metric that did not
correlate with the required c32 throughput.

Historical vLLM c32 evidence remains much higher:

- vLLM c32 output TPS mean `1708.527850`
- vLLM c32 lower confidence bound `1687.396531`
- 80% target from LCB `1349.917225`

Earlier status text also referenced a `1366.822 tok/s` target. Both targets are
far above the 24-row Ferrum c32 diagnostic range above.

## Core Code Finding

### Scheduler

`ContinuousBatchRequest` now carries three separate capacity-defer state fields:

- `capacity_deferred_until_release_epoch`
- `capacity_deferred_mixed_attempt_epoch`
- `capacity_deferred_from_decode`

The scheduler also keeps global atomics for:

- capacity backpressure
- decode capacity backpressure
- capacity release epoch
- mixed recompute epoch
- mixed recompute blocked-until epoch
- required blocks per mixed slot
- observed free blocks

Relevant reviewed lines:

- `continuous.rs:99-104`
- `continuous.rs:200-208`
- `continuous.rs:531-559`
- `continuous.rs:583-691`
- `continuous.rs:700-725`
- `continuous.rs:755-789`
- `continuous.rs:1178-1371`
- `continuous.rs:1490-1525`

This means scheduler state is not simply "waiting vs active". A request can be
waiting, release-blocked, decode-derived, already attempted in the current
mixed epoch, or globally blocked by structured KV pressure.

### Engine

`process_batch_unified()` can touch all resource owners in one iteration:

- pre-allocate recurrent state;
- allocate or reuse engine KV handles;
- ask the model executor to reserve model-owned KV slots;
- dispatch unified decode;
- on resource failure, release model cache, KV cache, draft KV, recurrent
  state, then move scheduler state.

Relevant reviewed lines:

- `batch.rs:268-304`
- `batch.rs:307-333`
- `batch.rs:463-505`
- `batch.rs:520-535`
- `batch.rs:910-950`
- `batch.rs:954-1020`

The latest `2f5a375e` class of bug fits this structure: the order of "mark
prefills as capacity-deferred" vs "advance mixed recompute feedback epoch" can
make requests appear already attempted in the reopened epoch. That is a
transaction-order bug, not a kernel bug.

### KV And Recurrent State

Paged KV now rolls back partial block allocation:

- `crates/ferrum-kv/src/managers/paged.rs:434-452`
- tests at `paged.rs:1087-1145`

Qwen35 recurrent state now has capacity-aware allocation:

- `qwen35.rs:11175-11227`
- slot overflow returns `ResourceExhausted` at `qwen35.rs:1018-1021`
- indexed linear-state slots can be capped independently from paged sequences
  at `qwen35.rs:998-1005`
- tests at `qwen35.rs:14299-14345` and `qwen35.rs:14665-14710`

Autosize and typed config now propagate recurrent-state budget:

- `auto_config.rs:396-440`
- `auto_config.rs:482-486`
- `auto_config.rs:706-717`
- `auto_config.rs:839-846`
- `auto_config.rs:1074-1098`
- `auto_config.rs:1438-1469`
- `auto_config.rs:2123-2143`
- tests at `auto_config.rs:2508-2698`

That answers the OOM/dynamic-KV question more precisely:

- The code does have dynamic/deferring pieces.
- It also has partial allocation rollback and recurrent-state capacity checks.
- But W3 c32 still failed because resource ownership is split across scheduler,
  engine KV, model-owned KV, model cache, draft KV, and recurrent state.
- The missing thing was not "add dynamic KV"; it was a single invariant that
  says which owner is authoritative after every defer, preempt, recompute,
  fallback, and failed reservation.

## vLLM Comparison

vLLM V1 scheduler behavior checked in local source:

- `scheduler.py:385-389`: every step starts KV manager state, then schedules
  RUNNING requests first.
- `scheduler.py:465-475`: it calls `kv_cache_manager.allocate_slots()` for one
  running request and proceeds only if blocks are returned.
- `scheduler.py:477-514`: if allocation returns `None`, it preempts a running
  request, frees KV, and retries. If the same request was preempted, it stops
  scheduling that request.
- `scheduler.py:567-571`: WAITING requests are scheduled only when there were
  no preempted requests in that step.
- `scheduler.py:965-985`: preemption has a single owner transition:
  free KV/encoder cache, reset computed tokens, mark preempted, prepend to
  waiting.

vLLM KV manager behavior checked in local source:

- `kv_cache_manager.py:228-264`: `can_fit_full_sequence()` is a pre-admission
  gate for chunked prefill.
- `kv_cache_manager.py:375-397`: before allocation, vLLM frees skipped blocks,
  computes the number of blocks needed, and returns `None` if there are not
  enough free blocks.
- `kv_cache_manager.py:412-435`: blocks are allocated and cached only after the
  fit check passes.
- `single_type_kv_cache_manager.py:893-958`: Mamba/linear-attention state uses
  a block accounting path that can force "not enough blocks" for same-step
  dependencies and limits old requests to a small incremental allocation.
- `single_type_kv_cache_manager.py:960-1036`: Mamba align mode keeps/reuses the
  last state block and allocates at most the expected incremental blocks.

The actionable contrast:

- vLLM's scheduler and KV manager define a clear "allocate or return None"
  boundary for each request.
- Ferrum's W3 path built larger mixed batches first, then fixed failures after
  model-owned KV reservation or unified forward failed.
- vLLM skips WAITING admission for a step after preemption; Ferrum repeatedly
  had to add local backpressure and epoch markers to avoid immediate re-admit
  churn.

The next Ferrum fix should copy the invariant, not the exact Python structure:

1. Decide the authoritative resource owner before scheduling a mixed batch.
2. If capacity cannot fit, return a scheduler-level "not schedulable now" state
   before model execution.
3. If resources were allocated, rollback must be atomic across engine KV,
   model-owned KV, model cache, draft KV, recurrent state, and scheduler phase.
4. After any capacity preemption/defer, do not admit new WAITING work in the
   same step unless there is explicit free-capacity evidence.

## Time-Ledger Review

The first-pass retrospective recorded:

- 1095 completed entries.
- About `40.28 h` recorded from 2026-06-23 through 2026-06-25.
- Largest buckets included `git_sync`, `code_edit`, `gpu_diagnostic_run`,
  `validation`, `artifact_handling`, `artifact_analysis`, and `code_reading`.

The deeper read changes the interpretation:

- It was not just "waiting on compile" or "Vast unavailable".
- Too much time went into GPU diagnostics whose expected outcome was only
  "does this new local patch move a proxy metric".
- The repeated status/doc commits were useful for evidence, but they also made
  it too easy to count activity as progress.
- The point where paid GPU should have stopped was earlier: once clean c32 runs
  proved zero OOM but only 300-488 tok/s, the next step should have been local
  invariant/model tracing rather than more c32 scheduler knob attempts.

## Revised Plan

Do not start another GPU just to test another small scheduler tweak.

Concrete next step before paid GPU:

1. Add a local scheduler/engine resource-state invariant harness.
2. Encode at least these states:
   - active decode owns model KV and engine KV;
   - capacity-deferred waiting owns no KV/recurrent/model cache;
   - decode-derived waiting requires recompute and has no live model cache;
   - mixed reserve failure does not advance the epoch before affected requests
     are marked consistently;
   - after preempt/defer, same-step WAITING admission is blocked unless a
     capacity-release signal exists.
3. Only after that harness passes should `2f5a375e` be validated on the
   retained 1x RTX 4090 if it becomes startable.

The current state remains:

- W3 is not complete.
- There is no `MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`.
- `2f5a375e` remains a local source candidate, not GPU evidence.
- The retained Vast instance was last recorded as stopped/exited and
  unavailable.
