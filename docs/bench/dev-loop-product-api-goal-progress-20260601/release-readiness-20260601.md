# Release Readiness - 2026-06-01

Scope: converge the dev-loop/product-API goal into a formal release candidate.

Release candidate packet:

- `docs/bench/dev-loop-product-api-goal-progress-20260601/release-candidate-0.7.3-20260601.md`

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
| 8B GGUF Ferrum/vLLM benchmark | pass with caveats | Saved GGUF-vs-GGUF tables are mirrored in `gguf-8b-remote-artifacts/` and summarized in `gguf-8b-release-benchmarks-20260601.md` |
| Metal Qwen3-MoE prefill readback | pass smoke | `1e3ce42` adds a pre-readback sync; local Metal smoke returns Paris and no longer triggers the encoder assertion |
| Metal Qwen3-8B GGUF smoke | pass | `metal-qwen3-8b-q4km-smoke-20260601`, Paris correctness passed; 64-token decode median `23.5 tok/s` with local swap active |
| Post-fix GPU quick regression | pass | `/workspace/m3-quick-regress-1e3ce42-c32-20260601`, c32 source FA2 `1403.98 tok/s`, correctness/multi-turn passed, perf gate `ok=true` |

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
- The 8B GGUF Ferrum/vLLM tables are measured, but release notes must label
  them as GGUF-vs-GGUF and explain that Ferrum uses eager-dequant/fp16 dense
  CUDA fallback while vLLM GGUF is experimental.
- The local Metal performance run is not clean enough for a release
  performance claim because the Mac had active swap and `run --bench-mode`
  stopped early on EOS; only the Metal correctness/syncfix evidence should be
  used for this release.
- A smaller Qwen3-8B Metal smoke did produce a stable 64-token decode run:
  median `23.5 tok/s` over 3 runs, with swap still active. Treat it as a
  regression smoke, not a clean headline benchmark.

8B GGUF CUDA serve notes:

- Commit `42ffbe2` enables a CUDA GGUF eager-dequant fallback path. It is a
  product compatibility path, not a native CUDA k-quant performance path.
- `qwen3:8b-q4_k_m` and `llama3.1:8b-q4_k_m` both passed OpenAI-compatible
  health/chat smoke on RTX 4090 before the Vast instance stopped.
- Release benchmark tables must label this clearly if used: Ferrum currently
  loads GGUF weights through eager-dequant/fp16 dense CUDA fallback, while vLLM
  is expected to run its experimental GGUF path.
- Qwen3-8B GGUF measured Ferrum/vLLM ratios: c1 `0.477x`, c4 `0.735x`,
  c16 `1.40x`, c32 `1.71x`.
- LLaMA-3.1-8B GGUF measured Ferrum/vLLM ratios: c1 `0.471x`, c4 `0.786x`,
  c16 `1.55x`, c32 `2.09x`.
