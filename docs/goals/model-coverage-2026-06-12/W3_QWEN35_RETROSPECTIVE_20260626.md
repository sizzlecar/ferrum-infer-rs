# W3 Qwen3.5 Retrospective — 2026-06-26

Correction: this document is a first-pass evidence index, not the complete deep
review. The deeper follow-up is
`docs/goals/model-coverage-2026-06-12/W3_QWEN35_DEEP_REVIEW_ADDENDUM_20260626.md`.

Scope: current branch `goal/w2-w3-release-grade` at
`7e55d935e1836979908ef63aa97008fd45bda88c`, recent 500 commits, core
scheduler/KV/engine code, local time ledger `ACTION_TIME_LEDGER_20260623.jsonl`,
and archived W3/Qwen35 artifacts.

No GPU work was run for this review. No live vLLM rerun was used.

## Bottom Line

W3 is not complete. There is still no final
`MODEL_RELEASE_GRADE_W3 PASS: <out_dir>`.

The recent work did produce real stability and correctness improvements:

- c32 no longer dies as a simple immediate OOM in the latest successful
  diagnostics.
- Several artifacts completed `32/32` requests with zero HTTP 500, zero panic,
  zero OOM mentions, and `output_token_count_source=usage`.
- The latest source candidate, `2f5a375e`, fixes a real engine/scheduler order
  bug in the mixed KV pressure path and has targeted local tests.

But that work did not convert into final-goal progress because the performance
loop stayed far below target and repeatedly discovered adjacent resource-state
bugs one GPU run at a time.

## Evidence Reviewed

Recent 500 commits:

- Commit type counts:
  - `docs`: 117
  - `test`: 115
  - `perf`: 108
  - `fix`: 85
  - `feat`: 57
  - `Revert`: 14 total
- 168 commit subjects contain failure/evidence words such as `reject`,
  `blocker`, `diagnostic`, `failure`, `timeout`, `unavailable`, `OOM`, or
  `record`.
- 2026-06-25 alone has 129 commits, mostly status/evidence updates, diagnostic
  target flips, and small scheduler/engine changes.

Most touched paths in those 500 commits:

- `docs/goals/model-coverage-2026-06-12/STATUS.md`: 335 commits
- `crates/ferrum-models/src/models/qwen35.rs`: 114 commits
- `crates/ferrum-scheduler/src/implementations/continuous.rs`: 45 commits
- `docs/goals/model-coverage-2026-06-12/HANDOFF_W3_QWEN35_20260622.md`: 39 commits
- `crates/ferrum-engine/src/continuous_engine/inner/batch.rs`: 38 commits
- `crates/ferrum-engine/src/continuous_engine/tests.rs`: 37 commits
- `scripts/release/w3_qwen35_vast_c32_diagnostic.py`: 28 commits
- `scripts/release/model_release_grade_goal_gate.py`: 28 commits

Local time ledger:

- 1095 completed ledger entries.
- Total recorded time: 145012 seconds, about 40.28 hours.
- Largest categories:
  - `git_sync`: 4.03 h
  - `code_edit`: 3.16 h
  - `gpu_diagnostic_run`: 2.99 h
  - `validation`: 2.04 h
  - `artifact_handling`: 1.52 h
  - `artifact_analysis`: 1.40 h
  - `code_reading`: 1.24 h
  - `local_validation`: 1.21 h
  - `commit`: 1.20 h
  - `source_trace`: 1.19 h
- Interpretation: the loop was not dominated by one compile. It was dominated
  by repeated sync, small edits, local tests, remote diagnostics, artifact
  handling, and status bookkeeping.

Archived artifacts:

- `docs/goals/model-coverage-2026-06-12/artifacts/` contains 261 top-level
  artifact directories.
- 128 are Qwen35-related.
- The c32 diagnostic family repeatedly completed correctness while rejecting
  performance or trace shape.

Representative performance facts:

| Artifact / SHA | Status | c32 output tok/s | p95 ITL ms | mixed iterations | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `w3_qwen35_default_full_l5_fixed_output_20260623_39ffe5db` | L5 local pass only | 627.235 | 20.924 | n/a | c=1/4/16/32, n=3, zero errors, but far below vLLM 80% target |
| `9fda1101` c32 | keep diagnostic | 488.596 | n/a | n/a | not release evidence |
| `f1bc7b7e` c32 | reject | 476.311 | 48.888 | 0 | correctness clean, no OOM |
| `d54f634b` c32 | reject | 450.984 | 51.363 | 1 | decode width improved but recompute stuck |
| `e50ff975` c32 | reject | 452.406 | 50.601 | 1 | scheduler feedback fix did not take effect |
| `2f5a375e` c32 | missing | n/a | n/a | n/a | source fix exists, Vast retained instance unavailable |

The target context from the cancelled handoff remains important: accepted local
vLLM c32 baseline was `1708.52785 output tok/s`, and the 80% target was
`1366.82228 output tok/s`. The later c32 diagnostics were mostly in the
`300-488 tok/s` range, below the older `627 tok/s` L5 artifact and far below
the release target.

## What Actually Improved

Correctness/stability did improve:

- Recurrent-state and KV admission paths now defer instead of hard-failing in
  more cases.
- Existing KV/recurrent state leaks on capacity defer were found and fixed.
- Product smoke coverage for `ferrum run` and `ferrum serve` appeared in the
  scoped diagnostics.
- The latest clean diagnostics have zero request errors, zero panic, zero OOM,
  and usage-based token counting.

These are real engineering improvements, but they are not release-grade W3
progress by themselves because the final target is throughput/latency plus
L2-L5 release evidence.

## Root Causes

### 1. The Resource State Machine Is Split Across Too Many Owners

The scheduler tracks request phase and pressure state:

- `capacity_deferred_until_release_epoch`
- `capacity_deferred_mixed_attempt_epoch`
- `capacity_deferred_from_decode`
- `capacity_backpressure_limit`
- `decode_capacity_backpressure_limit`
- `capacity_mixed_recompute_epoch`
- `capacity_mixed_recompute_blocked_until_epoch`
- required/observed KV free-block feedback

The engine separately owns or releases physical resources:

- model executor cache ids,
- scheduler phase,
- engine sequence state,
- KV manager handles,
- draft KV handles,
- recurrent-state handles.

This split is visible in:

- `crates/ferrum-scheduler/src/implementations/continuous.rs`
- `crates/ferrum-engine/src/continuous_engine/inner/batch.rs`
- `crates/ferrum-engine/src/continuous_engine/tests.rs`

The repeated failures came from edges between those owners:

- prefill moved back to waiting but physical KV was not released;
- recurrent state allocated, then fallback/defer path did not release it;
- structured KV feedback advanced a scheduler epoch before the failed request
  had been moved back to waiting;
- decode-origin pressure and ordinary prefill pressure needed different
  backpressure semantics but shared fields for too long.

This architecture made every small fix locally plausible but globally fragile.

### 2. Tests Were Reactive, Not State-Model Driven

The repository now has many useful targeted tests, for example:

- capacity defer releases existing KV;
- reserve ResourceExhausted defers without fallback;
- structured pressure reopens recompute in the next epoch;
- prefill-first resumes decode when active target is reached.

The problem is that these tests were added after each GPU failure. They prove
individual paths, but they do not define the full resource-state invariant.

Missing test layer:

- a scheduler/engine resource transaction model that says, for every defer,
  fallback, reserve failure, decode pressure, prefill pressure, and cancellation:
  - which logical queue owns the request;
  - which physical resources must still be live;
  - which resources must be released;
  - whether the next scheduler iteration may re-admit the request;
  - which epoch/attempt markers are legal.
- replay tests from compressed scheduler traces or minimized trace summaries.
- property-style tests that fail if a request is in waiting while holding
  physical KV/recurrent resources, or active while all resource handles are gone.

Without that layer, GPU was used as the first complete system test.

### 3. Diagnostic Thresholds Were Not Strongly Tied To The Final Goal

The scoped c32 runner used useful diagnostic gates such as:

- `output_throughput_tps > 600`
- `mixed_iterations >= 64`
- `p95_itl_ms <= 25`
- bounded KV admission failures and capacity-deferred counts.

Those thresholds caught regressions, but they were still far below the W3 final
performance bar. A candidate could improve one diagnostic counter while not
meaningfully moving toward the `1366.82228 tok/s` c32 target.

The result was a local optimization loop around trace shape rather than a
release-goal loop around the final ratio.

### 4. The Loop Accumulated Adjacent Scheduler Changes Too Quickly

The 2026-06-25 commit sequence shows many narrow scheduler changes:

- defer capacity-blocked prefills;
- back off capacity-deferred admission;
- wait to re-admit capacity-deferred decodes;
- limit blocked recompute admission;
- share blocked recompute mixed slot;
- suppress blocked recompute retry churn;
- keep scanning blocked recompute candidates;
- gate mixed recompute on KV capacity;
- pace by KV snapshot;
- reserve KV headroom;
- cap decode width after KV pressure;
- keep decode survivors wide;
- reopen recompute from KV feedback;
- fix mixed KV feedback order.

Each was defensible from the immediately preceding artifact, but the series was
too long without a higher-level invariant gate. That made it hard to tell
whether a later artifact was measuring the intended fix or a new interaction.

### 5. GPU/Infrastructure Friction Became Part Of The Critical Path

The user was right to object to repeated machine churn. The ledger and STATUS
show time lost to:

- multiple Vast start failures;
- `resources_unavailable`;
- SSL timeout/EOF during Vast API reads;
- remote PATH/build/script mistakes;
- artifact copy and gzip cleanup;
- full CUDA build waits.

The retained instance policy and later `resources_unavailable` fast-fail runner
reduced this, but they were added after time had already been spent.

### 6. Status Documentation Was Accurate But Too Expensive

`STATUS.md` was updated 335 times in the recent 500 commits. The discipline
helped avoid false completion claims, but the process also consumed a large
amount of time.

The issue is not that evidence was recorded. The issue is that each narrow
candidate required:

1. source edit,
2. local validation,
3. runner target update,
4. remote diagnostic,
5. artifact copy,
6. STATUS update,
7. commit/push,
8. next hypothesis.

That loop is too expensive unless each candidate is backed by a stronger local
model or a much sharper expected signal.

## Answer To The OOM / Dynamic KV Concern

The user remembered the right design direction: if capacity is insufficient,
the system should wait/defer until resources are released instead of submitting
work into OOM.

What happened here is that dynamic KV covered only part of the problem:

- Physical paged KV can be deferred/released.
- Qwen3.5 also has large recurrent-state pools and model-side cache handles.
- Unified decode+prefill reserve failures happen after some logical admission
  and sometimes after resource handles are partially allocated.
- Engine fallback paths could previously clear sequence fields without freeing
  the corresponding physical resource.

So the failure class was not simply "KV is not dynamic". It was "resource
admission is not transactional across KV, model cache, recurrent state,
scheduler phase, and mixed recompute attempt state."

## Decision On Latest Candidate

Keep `2f5a375e` as an unverified source candidate. It fixes a real ordering
bug and has a targeted regression test. Do not stack another scheduler change
on top of it until either:

- a real 1x RTX 4090 c32 diagnostic for `2f5a375e` is collected, or
- a stronger local resource-state invariant test shows a different bug that
  invalidates the candidate before GPU.

Do not start a different paid GPU unless the user explicitly approves it. The
retained Vast instance `42216671` was last recorded as stopped/exited and
unavailable with `resources_unavailable`.

## Required Process Change

Before the next paid W3 c32 diagnostic after any scheduler/engine change:

1. Write the exact expected artifact signal:
   - e.g. `mixed_iterations` should increase from `1`;
   - `unified_prof.mixed.count` should become non-zero;
   - average decode items should not collapse below the previous good value;
   - throughput should beat the latest comparable clean artifact, not only the
     diagnostic floor.
2. Add or update a local test that encodes the relevant resource-state
   invariant, not only the latest branch of code.
3. State whether the change is expected to move:
   - stability,
   - correctness,
   - diagnostic trace shape,
   - or final W3 performance.
4. If it only moves stability or trace shape, do not call it progress toward W3
   final performance until the artifact proves it.
5. If two consecutive artifacts miss the same failure class, stop GPU work and
   build the local invariant/replay test first.

## Recommended Next Engineering Step

Build a small scheduler/engine resource-state audit harness before more W3
performance work:

- Input: a synthetic sequence of events covering prefill/decode admission,
  reserve failures, capacity pressure, recurrent allocation failure, fallback,
  cancellation, completion, and decode recompute.
- Output invariant:
  - a request is in exactly one logical queue/state;
  - waiting requests do not hold physical KV/model cache/recurrent handles
    unless explicitly documented for a valid prefix-cache case;
  - active requests have the resources their phase requires;
  - capacity-deferred requests cannot be re-admitted in the same blocked epoch
    unless a recorded capacity-progress event makes that legal;
  - mixed recompute attempt markers are written before/after epoch changes in a
    tested order.

Only after that should the branch spend another paid GPU cycle on a new
scheduler lever.
