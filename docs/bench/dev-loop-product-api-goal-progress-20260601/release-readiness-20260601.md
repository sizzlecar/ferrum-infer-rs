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
| Real-model OpenAI API smoke | pass for direct release-binary smoke | `/workspace/m3-real-model-api-direct-smoke-20260601`, `all_passed=true` |
| Alias serve ergonomics | pass | `serve qwen3:0.6b` release-binary smoke passed without `FERRUM_MODEL_PATH` at `/workspace/release-alias-serve-qwen3-06b-8ec0858` |
| Qwen3-8B GGUF CUDA serve | pass smoke | `qwen3:8b-q4_k_m` release-binary OpenAI smoke passed at `/workspace/release-qwen3-8b-gguf-cuda-smoke-42ffbe2` |
| LLaMA-3.1-8B GGUF CUDA serve | pass smoke | `llama3.1:8b-q4_k_m` release-binary OpenAI smoke passed at `/workspace/release-llama31-8b-gguf-cuda-smoke-42ffbe2` |
| 8B GGUF Ferrum/vLLM benchmark | pending | Ferrum smoke is unblocked, but vLLM comparison is pending a runnable GPU instance; Vast creation failed with `insufficient_credit` after stopped instance `38872161` could not restart |

M3 release-threshold table:

| c | source FA2 tok/s | vLLM tok/s | ratio | release gate |
|---:|---:|---:|---:|---|
| 1 | `157.18` | `183.9` | `0.855x` | pass |
| 4 | `448.36` | `512.5` | `0.875x` | pass |
| 16 | `1115.58` | `1331.9` | `0.838x` | pass |
| 32 | `1488.08` | `1972.9` | `0.754x` | pass |

Release blockers remaining as of this checkpoint:

- The ignored `cargo test` SDK smoke remains blocked by a debug CUDA build
  script hang; direct release-binary smoke passed and is sufficient for release
  evidence if accepted.
- Final release checkpoint/tag has not yet been created.
- If the release requires source FA2 to be enabled by default rather than
  release-supported opt-in, the default/selector decision must be made before
  tagging.
- Saved 8B GGUF Ferrum/vLLM comparison tables are still pending because no
  runnable GPU instance is available. Vast instance `38872161` could not be
  restarted (`resources_unavailable`), was destroyed, and replacement creation
  failed with `insufficient_credit`.

8B GGUF CUDA serve notes:

- Commit `42ffbe2` enables a CUDA GGUF eager-dequant fallback path. It is a
  product compatibility path, not a native CUDA k-quant performance path.
- `qwen3:8b-q4_k_m` and `llama3.1:8b-q4_k_m` both passed OpenAI-compatible
  health/chat smoke on RTX 4090 before the Vast instance stopped.
- Release benchmark tables must label this clearly if used: Ferrum currently
  loads GGUF weights through eager-dequant/fp16 dense CUDA fallback, while vLLM
  is expected to run its experimental GGUF path.
