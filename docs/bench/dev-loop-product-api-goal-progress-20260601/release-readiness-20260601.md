# Release Readiness - 2026-06-01

Scope: converge the dev-loop/product-API goal into a formal release candidate.

Release performance threshold:

- User-adjusted M3 release gate: Ferrum throughput at or above `0.75x` of
  same-pod vLLM counts as performance-pass for this release.
- The previous `0.80x` target remains a stretch goal, not a release blocker.

Version:

- Workspace package version: `0.7.3`

Evidence summary:

| Area | Status | Evidence |
|---|---|---|
| CUDA release iteration loop | pass | `/workspace/m3-release-touch-probe-thinlto-20260601-20260601_064127`, p50 `33.164s`, p95 `34.454s`, all `39` CUDA rows cache-hit |
| Native source FA2 build boundary | pass | `scripts/check_fa2_source_native.py`, no external FlashAttention/CUTLASS product dependency |
| Runtime snapshot boundary | pass | `scripts/check_runtime_snapshot_boundary.py` |
| Source FA2 all-cell correctness | pass | `/workspace/m3-fa2-source-current-allcells-n3-20260601`, Paris, multi-turn, and three-turn `Paris/basalt` recall passed for c=1/4/16/32 |
| Source FA2 all-cell performance | pass at `0.75x` release threshold | c1 `0.855x`, c4 `0.875x`, c16 `0.838x`, c32 `0.754x` versus same-pod vLLM |
| q2 native FA2 candidate | rejected | microbench-positive but full-model c32 regressed to `1462.15 tok/s`; reverted by `2197077` |
| Real-model OpenAI API smoke | in progress / required before release sign-off | first attempt hit HF pull failure; `Qwen/Qwen3-0.6B` cache was populated, then smoke restarted with `PULL_MODEL=0` |

M3 release-threshold table:

| c | source FA2 tok/s | vLLM tok/s | ratio | release gate |
|---:|---:|---:|---:|---|
| 1 | `157.18` | `183.9` | `0.855x` | pass |
| 4 | `448.36` | `512.5` | `0.875x` | pass |
| 16 | `1115.58` | `1331.9` | `0.838x` | pass |
| 32 | `1488.08` | `1972.9` | `0.754x` | pass |

Release blockers remaining as of this checkpoint:

- Real-model OpenAI API smoke must finish and validate, or be explicitly
  waived with the failed/pending reason documented.
- Final release checkpoint/tag has not yet been created.
- If the release requires source FA2 to be enabled by default rather than
  release-supported opt-in, the default/selector decision must be made before
  tagging.
