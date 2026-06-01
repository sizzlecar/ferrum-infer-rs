# Goal: Shorten Performance Iteration Loop and Harden Product API

**Status:** draft @ 2026-05-30
**Owner:** ferrum core
**Source review baseline:** local HEAD `36e3dc2`
**Primary scope:** build iteration speed, profiling/benchmark discipline, runtime config clarity, auto configuration, OpenAI-compatible product API

**Evidence refresh:** 2026-06-01 10:19:37 +0800 (local self-test health passes)

## Evidence freshness checks (latest run at 10:19:37 +0800)

- `python3 scripts/m3_cuda_build_boundary_probe.py --self-test` → `ok`
- `python3 scripts/validate_cuda_build_summary.py --self-test` → `ok`
- `python3 scripts/validate_cuda_build_boundary_manifest.py --self-test` → `ok`
- `python3 scripts/m3_validate_runner_artifact.py --self-test` → `ok`
- `python3 scripts/m3_collect_allcell_runner_artifacts.py --self-test` → `ok`
- `python3 scripts/validate_real_model_api_smoke.py --self-test` → `ok`
- `python3 scripts/check_ferrum_env_registry.py --self-test` → `ok`
- `python3 scripts/m3_cuda_build_boundary_probe.py --iterations 5 --out /tmp/m3-release-touch-probe-20260601-01 --fail-on-limit --no-cargo-verbose` → failed (`nvcc --version` not found / `nvidia-smi` not found)
- `python3 scripts/check_ferrum_env_registry.py --json --fail-on-registry-gap > docs/bench/dev-loop-product-api-goal-progress-20260601/registry-json-snapshot-20260601.json` → `ok`
- `python3 scripts/check_ferrum_env_registry.py --json --fail-on-registry-gap --max-direct-env-reads 75 --max-process-env-writes 24 --max-non-test-process-env-writes 1 --max-hot-direct-env-reads 4 > /tmp/registry-threshold-check-20260601.json` → `pass` (values: direct=75, hot=4, non_test_writes=1)
- `python3 scripts/check_fa2_source_native.py --self-test` → `ok`
- `python3 scripts/check_fa2_source_native.py` → `ok`
- `python3 scripts/check_runtime_snapshot_boundary.py --self-test` → `ok`
- `python3 scripts/check_runtime_snapshot_boundary.py` → `ok`
- Vast RTX 4090 5-run Milestone A cache-hit release touch probe at `/workspace/m3-release-touch-probe-cachehit-20260601-20260601_043825` → executed successfully but timing gate failed (`p50=231.517s`, `p95=234.608s`, limits `75s/90s`; every run had `cache_hit=39` CUDA summary rows)
- Vast RTX 4090 5-run Milestone A thin-LTO release touch probe at `/workspace/m3-release-touch-probe-thinlto-20260601-20260601_064127` → passed timing gate (`p50=33.164s`, `p95=34.454s`, limits `75s/90s`; every run had `cache_hit=39` CUDA summary rows)
- Native source FA2 all-cell N=3 at `/workspace/m3-fa2-source-current-allcells-n3-20260601` → artifact validator `ok=true`, all c=1/4/16/32 correctness gates passed, c32 source FA2 `1488.08 tok/s` (`0.754×` vLLM)
- User-adjusted formal release performance threshold is `0.75× vLLM`; the native source FA2 all-cell N=3 packet passes this release threshold for all cells.
- Native source FA2 q2 grouping experiment → microbench positive but full-model c32 negative (`1462.15 tok/s`), reverted by `2197077`
- Real-model API smoke attempt at `/workspace/m3-real-model-api-smoke-20260601` → failed before SDK tests because `ferrum pull qwen3:0.6b` returned HuggingFace `401 Unauthorized`
- Real-model direct release-binary API smoke at `/workspace/m3-real-model-api-direct-smoke-20260601` → passed health, chat, usage, streaming usage, json_object, and three-turn recall on `Qwen/Qwen3-0.6B`
- Alias release-binary smoke at `/workspace/release-alias-serve-qwen3-06b-8ec0858` → `serve qwen3:0.6b` passed without `FERRUM_MODEL_PATH`
- 8B GGUF CUDA serve smoke at `/workspace/release-qwen3-8b-gguf-cuda-smoke-42ffbe2` → `qwen3:8b-q4_k_m` passed health + OpenAI chat through the CUDA eager-dequant fallback path
- 8B GGUF CUDA serve smoke at `/workspace/release-llama31-8b-gguf-cuda-smoke-42ffbe2` → `llama3.1:8b-q4_k_m` passed health + OpenAI chat through the CUDA eager-dequant fallback path
- 8B GGUF Ferrum/vLLM benchmark packet remains pending because Vast instance `38872161` could not restart (`resources_unavailable`), was destroyed, and replacement 4090 creation failed with `insufficient_credit`

All commands above have explicit status noted above; all tooling self-tests passed while the Milestone A 5-run probe failed in this environment due missing CUDA binaries.

### Evidence index for this run

- `docs/bench/dev-loop-product-api-goal-progress-20260601/local-selftest-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/registry-json-snapshot-20260601.json`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/registry-threshold-check-20260601.json`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-release-touch-probe-20260601-01.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-release-touch-probe-20260601-01-run1-build.log`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-default-path-allcells-local-validate-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-real-model-api-smoke-script-local-validate-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/fa2-source-native-restore-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/local-static-boundary-guards-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-release-touch-probe-cachehit-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-release-touch-probe-thinlto-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-fa2-source-current-allcells-n3-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-native-fa2-q2-negative-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-real-model-api-smoke-hf401-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/release-readiness-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-real-model-api-direct-smoke-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/release-benchmark-plan-20260601.md`
- `docs/bench/dev-loop-product-api-goal-progress-20260601/release-candidate-0.7.3-20260601.md`

### Next-turn execution path (from this evidence state)

1. Do not rerun Milestone A as an evidence-only loop: the thin-LTO restored-pod
   5-run release touch artifact now passes the timing target at
   `/workspace/m3-release-touch-probe-thinlto-20260601-20260601_064127`.
2. On the same host, complete the all-cell default-path packet, and after the wrapper returns run
   `python3 scripts/m3_validate_runner_artifact.py --require-bench "$OUT_I"` where `OUT_I` is the aggregate root.
3. Only after the all-cell packet, real-model API smoke evidence, and remaining
   runtime-default ownership work are complete, update this objective file’s
   blocker list from `hard-blocked` to `closing`.
## Current Progress Snapshot (2026-06-01)

This objective is being tracked through the following module status packets:

- `docs/status/cuda-build-cache-2026-05-30.md` (Milestone A)
- `docs/status/structured-profile-2026-05-30.md` (Milestone B)
- `docs/status/m3-ab-runner-2026-05-30.md` (Milestone C)
- `docs/status/runtime-env-registry-2026-05-30.md` (Milestone D)
- `docs/status/auto-config-2026-05-30.md` (Milestone E)
- `docs/status/openai-api-compat-2026-05-30.md` (Milestone F/G)
- `docs/status/codebase-shape-2026-05-30.md` (Milestone H)
- `docs/status/non-regression-gates-2026-05-31.md` (Milestone I)
- `docs/bench/m3-80pct-goal-2026-05-25/GOAL.md` as M3 80% performance source of truth
- `docs/bench/dev-loop-product-api-goal-progress-20260601/next-runbook-20260601.md` (next execution runbook)

### Milestone status vs this objective

- A: largely implemented for content-hash cache stamps and structured CUDA build summaries; the thin-LTO 5-iteration restored-pod release probe now passes (`p50=33.164s`, `p95=34.454s` versus `75s/90s`) with all `39` CUDA summary rows at `cache_hit`.
- B: structured profile schema/events are in use and validated; remaining gap is complete migration of all primary producer coverage and routine use of `structured_jsonl` with required profile groups.
- C: reusable runner is in place and wrappers migrated for FA2, profile, scheduler, and route/profile flows; remaining gap is broader script migration and stable all-cell publishable packets.
- D: registry coverage is at `146/146` (scan scope), with parser gates and CI enforcement; remaining work is reducing hot-path direct reads in the remaining small surface and tightening non-product ownership across older compatibility bridges.
- E: startup `FerrumConfigBuilder`, preset, decision trace, and typed runtime artifacts are in place; remaining gap is making selector-driven defaults the sole owner across all runtime/model/admin paths and further hardware/GPU memory validation.
- F: OpenAI compatibility coverage is broad for the stub path and contract tests; remaining gap is real-model GPU smoke evidence and a few execution-order tradeoffs around strict schema streaming.
- G: strict schema handling is implemented for supported subset and rejected for unsupported `response_format.json_schema.strict=true`; remaining gap is a complete production-ready streaming latency story for strict schema while keeping hard validation.
- H: file ownership and line-count targets are mostly achieved; remaining gap is lowering typed-parameter-arity baseline and final cleanup for long signatures.
- I: correctness/performance checklist gates are tightened and benchmark impact metadata is required; source-FA2 all-cell N=3 evidence now exists, but remaining gap is a published full c1/4/16/32 all-cell, same-pod non-regression packet for final default-path claims.

The objective remains incomplete because Milestones I and E have explicit
remaining acceptance gaps and Milestone F/G still depend on full end-to-end real-model
artifact proof before final completion. Milestone A is no longer a binding
completion blocker for the current checkpoint.

## Turn log

- `2026-06-01 10:19:37 +0800`: ran and confirmed all configured local self-tests still pass in this workspace:
  - `m3_cuda_build_boundary_probe`, `validate_cuda_build_summary`,
    `validate_cuda_build_boundary_manifest`,
    `m3_validate_runner_artifact`,
    `m3_collect_allcell_runner_artifacts`,
    `check_ferrum_env_registry`.
  - Evidence artifact: `docs/bench/dev-loop-product-api-goal-progress-20260601/local-selftest-20260601.md`
- `2026-06-01 10:21:00 +0800`: captured registry machine-readable snapshot for target proof replay:
  - `docs/bench/dev-loop-product-api-goal-progress-20260601/registry-json-snapshot-20260601.json`
- `2026-06-01 10:22:00 +0800`: attempted Milestone A 5-run boundary probe in local environment. Probe failed on run 1 because CUDA tooling was unavailable (`nvcc` / `nvidia-smi` not found). Evidence saved in:
  - `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-release-touch-probe-20260601-01.md`
  - `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-release-touch-probe-20260601-01-run1-build.log`
- `2026-06-01 10:40:00 +0800`: added dedicated default-path all-cell
  executor wrapper and runbook shortcut:
  - `scripts/m3_default_path_allcells_ab.sh`
  - `docs/bench/dev-loop-product-api-goal-progress-20260601/next-runbook-20260601.md`
  - status sync entry in `docs/status/m3-ab-runner-2026-05-30.md`
- `2026-06-01 10:33:00 +0800`: fixed local validation gating for
  `scripts/m3_default_path_allcells_ab.sh`:
  - removed deprecated `validation.touched_areas` token to match validator vocabulary.
  - validated local `VALIDATE_ONLY=1` execution for all cells via
    `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-default-path-allcells-local-validate-20260601.md`
- `2026-06-01 11:05:00 +0800`: hardened the F/G real-model smoke executor:
  - `scripts/m3_real_model_api_smoke.sh` now writes `run_summary.json` before
    returning non-zero when a command fails.
  - fixed `all_passed` calculation so `rc=0` commands count as passed.
  - added `FERRUM_BIN` for explicit `ferrum pull` binary selection.
  - added `scripts/validate_real_model_api_smoke.py` so real-model F/G
    artifacts have a reusable shape/full-suite/pass validator.
  - local script-only validation is recorded in
    `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-real-model-api-smoke-script-local-validate-20260601.md`.
- `2026-06-01 11:39:19 +0800`: restored the `fa2-source` product build path away
  from the accidental external FlashAttention-source dependency:
  - `crates/ferrum-kernels/build.rs` now builds only the in-repo native C ABI
    kernel for `fa2-source`.
  - Added `crates/ferrum-kernels/kernels/fa2_source/ferrum_fa2_paged_varlen.cu`
    exporting `ferrum_fa2_paged_varlen_fwd`.
  - Removed the product build dependency on `FERRUM_FA2_SRC_DIR`,
    `FERRUM_CUTLASS_INCLUDE_DIR`, external FlashAttention source, and CUTLASS
    headers. GPU build/smoke validation is still pending.
  - Added `scripts/check_fa2_source_native.py` as a static source-boundary
    guard; it still needs to be run in the next evidence refresh.
  - Split runtime semantics so `FERRUM_FA2_SOURCE=1` no longer reports itself
    through the `fa2_direct_ffi` flag while still using the required
    FA-compatible K/V pool and C ABI attention dispatch.
- `2026-06-01 12:00:00 +0800`: checkpointed the restored native FA2 source path
  in git and synced it to the GPU host using a git bare remote + branch checkout:
  - local checkpoint commit: `ac3dfab983c25c696c8865f9fd18e3a2a5cd2914`
  - GPU bare repo: `/workspace/ferrum-git-remotes/ferrum-infer-rs-ac3dfab.git`
  - GPU branch: `codex/fa2-native-restore-ac3dfab`
  - GPU clean checkout: `/workspace/ferrum-fa2-native-restore-git-ac3dfab`
  - No GPU build, native guard, smoke, or benchmark evidence has been run from
    this checkpoint yet.
- `2026-06-01 12:20:00 +0800`: advanced Milestone E default ownership by
  carrying the startup `RuntimeConfigSnapshot` through `EngineConfig.backend`
  options into model/backend factories. Qwen3-MoE safetensors construction now
  has a typed snapshot path for `Qwen3MoeRuntimeEnv` and falls back to process
  env only when no `EngineConfig` snapshot is supplied. This reduces the M3
  model startup surface that must reconstruct selector defaults from shell env;
  compile/runtime validation is still pending.
- `2026-06-01 12:32:00 +0800`: added
  `scripts/check_runtime_snapshot_boundary.py` as a static Milestone E guard
  for the typed runtime-snapshot path into Qwen3-MoE safetensors startup, and
  added it to the next-runbook preconditions/finalization checklist. The guard
  has not been executed yet in this checkpoint.
- `2026-06-01 13:20:00 +0800`: restored GPU execution on Vast instance
  `38872161` after confirming `38237968` is no longer recoverable
  (`no_such_instance`). The Qwen3 GPTQ model cache was repopulated under
  `/workspace/hf-cache/.../9b534e4318b7ebc3c961a839f13eb18b1833f441`.
  The first default-path all-cell attempts exposed a runner startup bug where
  `ferrum serve <MODEL_DIR>` did not also populate `FERRUM_MODEL_PATH` for the
  tokenizer/model component factories. Commit `c9ba8fd` fixes the runner by
  carrying `FERRUM_MODEL_PATH` in case env and metadata. In-flight GPU run
  `/workspace/m3-default-path-allcells-20260601-20260601_051417` passed Paris
  and multi-turn gates for c=1 baseline and is consuming GPU; final all-cell
  publishability is still pending.
- `2026-06-01 13:25:00 +0800`: captured Milestone A restored-pod 5-run
  cache-hit release touch evidence:
  `/workspace/m3-release-touch-probe-cachehit-20260601-20260601_043825`.
  All 5 runs exited `0` and each had `39` CUDA summary rows at
  `status=cache_hit`, but timing failed (`p50=231.517s`,
  `p95=234.608s`, limits `75s/90s`). Milestone A is therefore no longer
  blocked by missing evidence; it is blocked by measured Rust/Cargo release
  dirtying and link time.
- `2026-06-01 15:55:11 +0800`: superseded the Milestone A restored-pod release
  timing failure with thin-LTO release-boundary evidence:
  `/workspace/m3-release-touch-probe-thinlto-20260601-20260601_064127`.
  All 5 runs exited `0`, each had `39` CUDA summary rows at
  `status=cache_hit`, and timing passed (`p50=33.164s`, `p95=34.454s`,
  limits `75s/90s`). Milestone A is no longer a binding blocker for the
  current checkpoint.
- `2026-06-01 15:55:11 +0800`: closed two local static guard evidence gaps:
  `scripts/check_fa2_source_native.py --self-test`,
  `scripts/check_fa2_source_native.py`,
  `scripts/check_runtime_snapshot_boundary.py --self-test`, and
  `scripts/check_runtime_snapshot_boundary.py` all passed. Evidence saved in
  `docs/bench/dev-loop-product-api-goal-progress-20260601/local-static-boundary-guards-20260601.md`.
- `2026-06-01 16:10:00 +0800`: completed native source FA2 all-cell N=3
  validation at `/workspace/m3-fa2-source-current-allcells-n3-20260601`.
  Artifact validation passed with 8 bench rows. Source FA2 measured c1
  `157.18`, c4 `448.36`, c16 `1115.58`, c32 `1488.08` tok/s; c32 remains
  about `0.754×` of same-pod vLLM. Under the user-adjusted formal release
  threshold of `0.75× vLLM`, all four cells pass; the previous `0.80×` target
  remains a stretch target.
- `2026-06-01 16:10:00 +0800`: tested native FA2 q2 grouping candidate
  `3a5ab00`. Standalone nvcc microbench improved large prefill-like shapes by
  about `+34%/+36%`, but full-model c32 N=3 regressed to `1462.15 tok/s`, so
  the candidate was reverted in `2197077`.
- `2026-06-01 16:10:00 +0800`: attempted F/G real-model API smoke at
  `/workspace/m3-real-model-api-smoke-20260601`; it failed at model pull with
  HuggingFace `401 Unauthorized` for `qwen3:0.6b`, before SDK tests ran.
  Vast instance `38872161` was stopped through the Vast API after GPU work
  completed.
- `2026-06-01 16:31:00 +0800`: completed direct release-binary real-model API
  smoke at `/workspace/m3-real-model-api-direct-smoke-20260601`.
  `Qwen/Qwen3-0.6B` passed health, non-streaming chat, usage fields,
  streaming `[DONE]`, streaming usage, `json_object`, and three-turn
  `basalt`/`Paris` recall. The ignored SDK cargo wrapper is still not used for
  release because it blocked in a debug CUDA build-script path.
- `2026-06-01 16:31:00 +0800`: added release benchmark plan for saved
  Ferrum/vLLM artifacts: M3 Qwen3-30B-A3B GPTQ Int4 plus GGUF-vs-GGUF
  Qwen3-8B and LLaMA-3.1-8B comparison tables.
- `2026-06-01 16:55:00 +0800`: closed the `serve qwen3:0.6b` alias
  release blocker. Commit `8ec0858` makes the tokenizer factory use typed
  `model_path` from component config instead of requiring process
  `FERRUM_MODEL_PATH`; GPU artifact
  `/workspace/release-alias-serve-qwen3-06b-8ec0858` passed health and
  OpenAI chat with the alias.
- `2026-06-01 17:05:00 +0800`: fixed GGUF tokenizer sidecar handling for
  LLaMA-3.1-8B. Commits `27d12b8` and `f346a87` map the LLaMA GGUF alias to
  a public tokenizer source and download only tokenizer/config sidecars
  instead of a full safetensors sibling. Artifact
  `/workspace/release-llama-gguf-tokenizer-f346a87` passed with GGUF,
  `tokenizer.json`, and `tokenizer_config.json` present.
- `2026-06-01 17:20:00 +0800`: added CUDA GGUF eager-dequant fallback support
  in commit `42ffbe2`. Qwen3-8B and LLaMA-3.1-8B GGUF both passed
  release-binary OpenAI-compatible smoke on RTX 4090:
  `/workspace/release-qwen3-8b-gguf-cuda-smoke-42ffbe2` and
  `/workspace/release-llama31-8b-gguf-cuda-smoke-42ffbe2`. This is a
  compatibility path, not a native CUDA k-quant performance path.
- `2026-06-01 17:35:00 +0800`: Vast instance `38872161` was found stopped and
  could not be restarted after three `state=running` attempts
  (`resources_unavailable`). It was destroyed per the GPU recovery rule.
  Replacement RTX 4090 creation on offer `32736582` failed with
  `insufficient_credit`, so saved 8B Ferrum/vLLM benchmark tables are pending
  more Vast credit or another available GPU source.
- `2026-06-01 17:50:00 +0800`: prepared the saved 8B GGUF Ferrum/vLLM release
  benchmark wrapper `scripts/release_gguf_8b_vs_vllm.sh`. It runs Qwen3-8B
  and LLaMA-3.1-8B GGUF-vs-GGUF through the shared `bench_vs_vllm.sh` harness
  with explicit vLLM GGUF model IDs and tokenizer models. A fresh RTX 4090
  creation attempt on offer `38712898` still failed with `insufficient_credit`,
  so execution remains blocked on Vast credit.
- `2026-06-01 18:00:00 +0800`: added the formal `0.7.3` release-candidate
  packet at
  `docs/bench/dev-loop-product-api-goal-progress-20260601/release-candidate-0.7.3-20260601.md`.
  It separates satisfied release gates from the pending 8B Ferrum/vLLM
  publicity benchmark and records release-note wording constraints.
  comparisons for Qwen3-8B and LLaMA-3.1-8B. vLLM GGUF support is treated as
  experimental and must be labeled separately.
- Next hard-stop decision points are now I/E/F/G blockers; Milestone A must
  stay green but is no longer a binding blocker for the current checkpoint.

### As-of-now blocker state

- `Milestone A` release-boundary timing is unblocked by the thin-LTO 5-run
  restored-pod proof (`p50=33.164s`, `p95=34.454s`, required
  `<=75s/<=90s`) with all CUDA artifacts at cache-hit.
- `Milestone E` is hard-blocked by unresolved auto-config ownership in benchmark/model/admin startup default branches.
- `Milestone I` has source-FA2 all-cell N=3 evidence and passes the adjusted
  `0.75× vLLM` formal release threshold. It remains open only if the release
  requires source FA2 to become the default path rather than a release-supported
  opt-in path.
- `Milestone F` and `Milestone G` have direct release-binary real-model smoke
  evidence. The ignored SDK cargo wrapper remains blocked by a debug CUDA
  build-script hang and should be treated as a post-release harness issue
  unless strict SDK-wrapper evidence is required.

---

## Objective

Make Ferrum easier to optimize and safer to expose as a product server by fixing structural issues that slow the loop:

1. bottleneck localization is too dependent on ad hoc env vars, hidden switches, and log grep;
2. small kernel edits can still trigger broad CUDA rebuild/link cycles;
3. M3 A/B validation scripts duplicate fragile server/process/artifact logic;
4. API compatibility is bench-oriented rather than full OpenAI-compatible;
5. strict JSON/schema/tool-calling behavior is not represented in the internal request/response model;
6. best-known runtime paths are encoded as shell env bundles instead of a validated auto-selection policy.

This goal is achieved only when a developer can make a narrow kernel/API/scheduler change, validate it with a scoped harness, and get comparable artifacts without manually reconstructing environment state from shell scripts and logs.

### Objective item coverage (6-item scope)

| Objective item | Status | Responsible milestone(s) | Evidence anchors |
|---|---|---|---|
| 1) Reduce ad-hoc bottleneck localization and log-grep dependency | Partial | B, C | `docs/status/structured-profile-2026-05-30.md`, `docs/status/m3-ab-runner-2026-05-30.md` |
| 2) Reduce narrow CUDA edit rebuild blast radius | Partial | A, D | `docs/status/cuda-build-cache-2026-05-30.md` |
| 3) Unify M3 A/B scripts and lifecycle | Partial | C | `docs/status/m3-ab-runner-2026-05-30.md` |
| 4) Complete OpenAI-compatible product API (tools, errors, usage, contracts) | Partial | F | `docs/status/openai-api-compat-2026-05-30.md` |
| 5) Structured strict-JSON/tooling in internal request/response model | Partial | F, G | `docs/status/openai-api-compat-2026-05-30.md` |
| 6) Move defaults from shell/env bundle to validated selector ownership | Partial | E, D | `docs/status/auto-config-2026-05-30.md`, `docs/status/runtime-env-registry-2026-05-30.md` |

## Completion Audit vs Source-of-Truth (2026-06-01)

| Milestone | Evidence status | Coverage in status/docs artifacts | Remaining acceptance gap |
|---|---|---|---|
| A (Build cache boundary) | Mostly complete | `docs/status/cuda-build-cache-2026-05-30.md` | Thin-LTO 5-iteration restored-pod release rebuild p50/p95 probe passes (`33.164s/34.454s` versus `75s/90s`) and every run reports all `39` CUDA artifacts at cache-hit. |
| B (Structured profiling) | Partially complete | `docs/status/structured-profile-2026-05-30.md` | Core schema/events are in place, but some producer/profile-path migration remains and overhead proof must be kept current for new rows. |
| C (Unified runner) | Mostly complete | `docs/status/m3-ab-runner-2026-05-30.md` | Need broader wrapper migration stability and publishable all-cell summaries for default-path sweeps. |
| D (Registry + snapshots) | Complete at current scope, with residual reduction potential | `docs/status/runtime-env-registry-2026-05-30.md` | Remaining work is reducing residual hot-path direct reads to keep all new code from regressing and shrinking product surface where possible. |
| E (Auto-config + selector) | Partially complete | `docs/status/auto-config-2026-05-30.md` | Selector logic is implemented and persisted to artifacts, but not yet the universal source-of-truth owner for all runtime defaults and model/admin paths. |
| F (Product API) | Partially complete | `docs/status/openai-api-compat-2026-05-30.md` | Deterministic and stub-contracts are strong; full real-model SDK smoke evidence is still required for completion posture. |
| G (Strict JSON/schema) | Partially complete | `docs/status/openai-api-compat-2026-05-30.md` | Strict schema is enforced/rejected correctly in many paths; strict streaming still needs final production-ready latency/correctness framing while preserving hard validation guarantees. |
| H (Codebase shape) | Mostly complete | `docs/status/codebase-shape-2026-05-30.md` | File-size limits are achieved; remaining long-signature baseline cleanup still required to reduce future drift. |
| I (Gates + non-regression) | Partial hardening complete | `docs/status/non-regression-gates-2026-05-31.md` | Full publishable default-path packet still missing: same-pod c1/4/16/32 (n_repeats >= 3), full correctness/performance regression table, and explicit baseline comparisons. |

### Current objective-impacting gaps

- M3 performance source-of-truth remains `docs/bench/m3-80pct-goal-2026-05-25/GOAL.md`. As of 2026-06-01, formal release performance threshold is `0.75× vLLM`; `0.80×` remains a stretch goal.
- The active goal is currently blocked by evidence completeness, not design intent: E, I, and F/G are the binding gaps for completion with current repo state.
- Milestone A now has CUDA-hosted release-boundary proof for this checkpoint; future kernel/build changes must keep the same probe green.

### Current completion debt (authoritative, as of 2026-06-01)

- Binding blockers for completion:
  - `Milestone E`: not all runtime default branches are fully sourced from startup builder/selector defaults with validated precedence metadata.
  - `Milestone I`: source-FA2 full-cell same-pod packet exists and passes the
    `0.75× vLLM` release threshold. The remaining decision is whether this
    release ships source FA2 as opt-in or makes it selector/default-owned.
- Partial blockers that still need closure work:
  - `Milestone B`: producer migrations and required-event coverage are in place for migrated paths, but remaining paths still need periodic validation as they move.
  - `Milestone C`: wrapper migration is broad but still depends on final stable all-cell publishable outputs for all active default-path scripts.
  - `Milestone F/G`: real-model ignored-smoke evidence remains a remaining
    production proof requirement; the latest restored-pod attempt is blocked by
    a `Qwen/Qwen3-0.6B` pull/cache authentication failure.
- Any completion claim that omits one of the binding blockers is invalid regardless of local green checks.

## Milestone Evidence Ledger (current)

The rows below are tied to explicit acceptance requirements in this objective. “Proved” means evidence is present in the status packet(s) and the cited command/output.

### A (Incremental CUDA Build Boundary)

- `cuda build.rs` now uses per-kernel stamped artifacts and emits unified summary lines for each artifact.
  Evidence: `docs/status/cuda-build-cache-2026-05-30.md` (core-ptx + static library cache-hit summary).
- Touching `crates/ferrum-kernels/kernels/paged_varlen_attention_vllm.cu` gives attention cache hits for unrelated kernels/libraries when signature unchanged; vLLM-MoE change rebuild scope is isolated to its target artifact.
  Evidence: `docs/status/cuda-build-cache-2026-05-30.md` (remote no-content/content-change validations).
- 5 consecutive release rebuild timing gate was run on restored RTX 4090 and
  initially failed:
  `/workspace/m3-release-touch-probe-cachehit-20260601-20260601_043825`
  measured `p50=231.517s`, `p95=234.608s` versus `75s/90s`. All runs had
  `39` CUDA summary rows at `status=cache_hit`, proving the miss was outside
  nvcc artifact rebuild.
- Thin-LTO follow-up 5-run release rebuild timing gate passed:
  `/workspace/m3-release-touch-probe-thinlto-20260601-20260601_064127`
  measured `p50=33.164s`, `p95=34.454s` versus `75s/90s`. All runs exited
  `0` and had `39` CUDA summary rows at `status=cache_hit`.
- **Required for completion:** keep this probe green for the final checkpoint;
  no additional Milestone A timing fix is currently blocking completion.

### B (Structured Profiling and Artifact Schema)

- `ProfileEvent` schema and required field validators are in place.
  Evidence: `docs/status/structured-profile-2026-05-30.md` (profile parser coverage + self-test).
- Route/profile scripts now consume structured JSONL events and do not rely on required grep patterns for required events.
  Evidence: `docs/status/structured-profile-2026-05-30.md`, `docs/status/m3-ab-runner-2026-05-30.md`.
- Producer coverage and overhead proof are partially in place and must remain current with any newly migrated scripts.
  Evidence: same.

### C (Unified A/B Runner)

- Runner owns process lifecycle, gate orchestration, manifest writing, and summary metrics.
  Evidence: `docs/status/m3-ab-runner-2026-05-30.md` and script self-tests.
- ≥4 scripts migrated; current migrated set is beyond 4 (FA2, FA-layout, profile, route, scheduler/admission variants).
  Evidence: same.
- Publishable all-cell summary examples are progressing but still need the final default-path full packet.

### D (Runtime Env Registry and Snapshot)

- Registry coverage currently complete in scan scope (`146/146`) with CI gate.
  Evidence: `docs/status/runtime-env-registry-2026-05-30.md`.
- Hot-path direct reads reduced to 4 (vs prior baseline target `<=26`) and are classified.
  Evidence: same.
- Remaining improvement: remove remaining compatibility bridge/direct read surfaces as broader conversion lands; keep the current 4-path baseline from regressing.

### E (Auto Configuration and Backend Selection)

- Typed builder, presets, decision trace, and startup artifacts (`effective_config.json`, `decision_trace.jsonl`) are implemented and schema-validated.
  Evidence: `docs/status/auto-config-2026-05-30.md`.
- Builder-aware default selection runs through runner-backed M3 preset flow and is used by FA2/FA-layout / profile wrapper paths.
  Evidence: same.
- **Not proved yet:** builder as universal owner for all runtime default branches across all benchmark/model/admin hot paths, plus GPU probe-backed memory validation for sequence/token sizing.

### F/G (API + Strict Schema)

- Deterministic contract coverage is broad for tools/functionality and strict schema handling (supported-enforced / unsupported-rejected).
  Evidence: `docs/status/openai-api-compat-2026-05-30.md`.
- Ignored real-model SDK smokes exist and validate intended paths but are not yet executed for production evidence.
  Evidence: same.
- Open item: document final strict-schema streaming tradeoff and keep proof that non-streaming/streaming gating meets the 100× / 20× deterministic expectations.

### H (Codebase Shape and Ownership)

- Line-count and file split targets for core owned surfaces are currently met.
  Evidence: `docs/status/codebase-shape-2026-05-30.md`.
- Remaining work is baseline-signature cleanup to prevent future 16+ param drift.

### I (Correctness and Non-Regression)

- Mandatory checklist fields and benchmark-impact traceability are now enforced in validator schema.
  Evidence: `docs/status/non-regression-gates-2026-05-31.md`.
- Default-path all-cell non-regression packet (c1/4/16/32, n_repeats≥3, same-pod baseline comparison) is still missing.
  Evidence: gap noted in `docs/status/non-regression-gates-2026-05-31.md`.

### Completion packet requirement mapping

To satisfy the final packet requirement, the following artifacts must be present under `docs/bench/` or `docs/status/` before marking complete:

1. `build timing table before/after` — from Milestone A probe manifest + log.
2. `structured profile sample + schema validation` — from Milestone B fixture/runner smoke.
3. `migrated runner example artifact` — from a publishable M3 wrapper run + validator.
4. `env registry + preset + runtime config snapshot example` — from runner `effective_config.json` and registry report.
5. `auto-config decision trace + selector validation report` — from runner/server artifacts and auto-config tests.
6. `OpenAI API compatibility matrix + strict JSON/schema report` — from `docs/openai-api-compatibility.md` plus status packet.
7. `code-size / ownership summary` — from `docs/status/codebase-shape-2026-05-30.md`.
8. `correctness + performance non-regression report` — from a signed full-cell default-path benchmark packet with `baseline_`/`candidate_` comparison.
9. `exact local + GPU commands` — in the corresponding status/docs evidence blocks.

Current missing completion packets to generate in this objective run:

- `docs/bench/m3-release-touch-probe-20260601-*.json/.log` — Milestone A timing evidence with limits pass/fail.
- `docs/bench/m3-default-path-all-cell-80pct-*.*/` — Milestone I publishable packet with `n_repeats>=3` and concurrency cells `1/4/16/32`.
- `docs/status/auto-config-default-owner-closure-20260601.md` or equivalent section update — Milestone E ownership closure with explicit branch coverage.
- `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-real-model-api-smoke-20260601-*/` — Milestone F/G real-model evidence and streaming strict-schema latency summary (artifact should include `commands.md`, `run_summary.json`, `cargo-test-*.log`, and a passing `scripts/validate_real_model_api_smoke.py "$OUT_F"` result).

### Completion evidence matrix (hard check, no substitutions)

- A completion claim is invalid unless each requirement below has direct evidence in this document or linked status packets:
  - `Milestone A`: timing packet with manifest + per-run validation; no pass/fail inferred from local notes.
  - `Milestone B`: profile sample + validator result from structured JSONL path for each claimed migrated profile run.
  - `Milestone C/I`: one publishable all-cell default-path artifact (`c1/4/16/32`, `n_repeats>=3`) with checklist fields and explicit `not_publishable=false` justification.
  - `Milestone D/E`: same manifest must carry sorted config snapshots, source/effect metadata, and decision trace IDs.
  - `Milestone F/G`: strict-schema/compatibility claims require corresponding test/packet citation plus `scripts/validate_real_model_api_smoke.py` pass for the real-model artifact.
  - `Milestone H`: file-shape/ownership claims point to the latest codebase-shape status snapshot.
  - `Exact commands`: every bullet above must be traceable to a concrete command line in either a status doc or this file.

## Milestone completion readiness (as-of-now)

| Milestone | Required evidence | Current local/state evidence | Closure status |
|---|---|---|---|
| A | 5-run release build boundary manifest (`p50`, `p95`, `limits_pass`), per-run `cuda-build-summary` validation, and `scripts/validate_cuda_build_boundary_manifest.py --require-limits-pass` pass | `scripts/m3_cuda_build_boundary_probe.py` and `scripts/validate_cuda_build_boundary_manifest.py` self-tests pass locally; status packet not yet produced for 5-run release boundary | **blocked** (requires GPU-bound release loop execution) |
| B | Publishable profile run with structured JSONL event groups and parser/fixture validation | schema + parser + migrated runner validations are in place; status includes structured profile artifacts | **partial** (no new full publishable packet written in this branch) |
| C | Publishable all-cell runner output with manifest + summaries + validator pass | runner/validator tooling exists and self-tests pass; no new default-path full-cell publish artifact from this change set | **partial** |
| D | Registry + env snapshot + diff artifact fields in runner manifests | static scan and schema gates pass; status shows `146/146` entries and bounded reads | **close to done** |
| E | Builder-owned default path decisions for all benchmark/model/admin runtime defaults; same manifest includes source/effect metadata and decision trace parity | runtime selector path exists and artifacts are schema-valid; universal ownership still incomplete | **blocked** |
| F | Real-model API compatibility packet + strict-schema behavior report | stub contracts are covered in status; `scripts/m3_real_model_api_smoke.sh` defines required command set and artifact shape; real-model packet with `commands.md` + `run_summary.json` still missing | **blocked** |
| G | Strict-schema production/streaming tradeoff doc + evidence of hard-gated behavior under strict schema | partial in status; final production story still pending | **partial** |
| H | Shape split, line limits, and signature controls verified in codebase-shape status | verified and tracked | **close to done** |
| I | Same-pod publishable default-path full-cell non-regression packet (c1/4/16/32, n_repeats>=3) with baseline/candidate and correctness gates | status requires this packet; currently absent | **blocked** |

### Evidence quality rule for this objective

- If a claim in Milestone D/I/F/G is supported only by a status file status line and no corresponding artifact path, the claim is treated as `partial` and does not satisfy completion.
- For any final completion decision, each row above must be `done` and point to at least one concrete artifact path plus one validation command.

### Priority next validation tasks (next turn)

1. Produce the Milestone A timing packet for 5 consecutive attention-only `ferrum-cli --release` rebuilds and record pass/fail against the `<=75s/<=90s` gate in a committed `docs/bench/*` artifact.
2. Run/publish a default-path, same-pod, all-cell (c1/4/16/32), n_repeats>=3 comparison packet with full `validation_checklist`, latency/throughput deltas, and local+artifact checks to close Milestone I. Use the explicit per-concurrency+aggregation sequence in `docs/bench/dev-loop-product-api-goal-progress-20260601/next-runbook-20260601.md`. The candidate must be the restored native in-repo `FERRUM_FA2_SOURCE=1` path, not the earlier external FlashAttention-source build or vLLM/Torch direct FFI shim.
3. Close the remaining auto-config ownership gap by making startup selectors the default owner for all runtime default branches covered by current benchmarks, then document the behavioral deltas (if any) with artifact-backed proof for Milestone E.
4. Produce a real-model API evidence packet (including strict schema behavior and strict/non-strict streaming behavior) for Milestones F/G via `scripts/m3_real_model_api_smoke.sh`, validate it with `scripts/validate_real_model_api_smoke.py "$OUT_F"`, then include it under the final completion packet. Required artifacts: `commands.md`, `run_summary.json`, and `cargo-test-*.log` in a timestamped `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-real-model-api-smoke-*` path.
5. After #1/#2/#3/#4 complete, run:
   - `python3 scripts/m3_validate_runner_artifact.py <artifact-root>`
   - `python3 scripts/validate_cuda_build_boundary_manifest.py --require-limits-pass <manifest>`
   - `python3 scripts/check_fa2_source_native.py`

### Next runbook template (reproducible commands)

Canonical executor-facing sequence for this next turn is tracked here:

- `docs/bench/dev-loop-product-api-goal-progress-20260601/next-runbook-20260601.md`

Use these templates when you resume execution (edit paths/tags only):

```bash
cd /Users/chejinxuan/rust_ws/ferrum-infer-rs

# Milestone A: 5-run release rebuild boundary
OUT_A=/workspace/m3-release-touch-probe-20260601-$(date +%Y%m%d_%H%M%S)
python3 scripts/m3_cuda_build_boundary_probe.py \
  --iterations 5 \
  --out "$OUT_A" \
  --fail-on-limit
python3 scripts/validate_cuda_build_boundary_manifest.py \
  --require-limits-pass \
  "$OUT_A/build_boundary_manifest.json"

# Milestone I: same-pod all-cell default-path packet (example orchestrator)
OUT_I=/workspace/m3-default-path-allcell-20260601-$(date +%Y%m%d_%H%M%S)
REPEATS=3 \
NUM_PROMPTS=128 \
WARMUP_REQUESTS=10 \
VALIDATION_CHANGE_TYPE=default_path \
CONCURRENCIES="1 4 16 32" \
OUT_ROOT="$OUT_I" \
BASELINE_ENV_JSON='{"FERRUM_FA_LAYOUT_VARLEN":"1"}' \
CANDIDATE_ENV_JSON='{"FERRUM_FA_LAYOUT_VARLEN":"1","FERRUM_FA2_SOURCE":"1"}' \
bash scripts/m3_default_path_allcells_ab.sh
python3 scripts/m3_validate_runner_artifact.py \
  --require-bench \
  "$OUT_I"
python3 scripts/check_fa2_source_native.py

# Milestone I practical note:
# - Milestone I now uses the dedicated default-path wrapper above.
#   The wrapper aggregates per-cell outputs and runs artifact validation by
#   default; set `VALIDATE_ARTIFACT=0` only when collecting validation manually.
#   Do not claim default-path closure until c1/4/16/32 rows are present and
#   both baseline/candidate rows are `not_publishable=false` in the same
#   aggregate artifact.

# Milestone E: auto-config ownership and decision-trace parity checks
cargo test -q -p ferrum-types auto_config -- --nocapture
cargo test -q -p ferrum-types runtime_config -- --nocapture
cargo test -q -p ferrum-cli config -- --nocapture
cargo test -q -p ferrum-cli runtime-env -- --nocapture
cargo test -q -p ferrum-cli source-resolver -- --nocapture
cargo test -q -p ferrum-cli commands::serve -- --nocapture
cargo test -q -p ferrum-server route_health_includes_runtime_config_snapshot -- --nocapture
python3 scripts/m3_ab_runner.py --self-test
python3 scripts/m3_validate_runner_artifact.py --self-test
python3 scripts/validate_real_model_api_smoke.py --self-test
python3 scripts/check_fa2_source_native.py --self-test
python3 scripts/check_fa2_source_native.py
python3 scripts/check_ferrum_env_registry.py --fail-on-registry-gap

# Milestone F/G: real-model API evidence packet
# Run the ignored SDK smoke tests documented in docs/status/openai-api-compat-2026-05-30.md
# (async-openai + Python OpenAI client) against a real model on a controlled GPU path,
# then store command output, logs, and pass/fail trace in a dedicated completion artifact.
OUT_F=${OUT_F:-/workspace/m3-real-model-api-smoke-20260601-$(date +%Y%m%d_%H%M%S)}
MODEL=${MODEL:-qwen3:0.6b} \
FERRUM_BIN=${FERRUM_BIN:-ferrum} \
OUT_ROOT="$OUT_F" \
CARGO_FEATURES=${CARGO_FEATURES:-metal} \
bash scripts/m3_real_model_api_smoke.sh
python3 scripts/validate_real_model_api_smoke.py "$OUT_F"
```

## Acceptance Summary

| Area | Current pain | Target |
|---|---|---|
| CUDA edit loop | attention-only edits can take minutes to tens of minutes depending on touched files | attention-only rebuild p95 `<= 90s`; unrelated Marlin/MoE-Marlin objects are not rebuilt |
| Profiler output | scripts grep human log lines | all required profile data emitted as JSONL or chrome trace with schema tests |
| Bench harness | many copied `m3_*_ab.sh` scripts | one reusable runner covers A/B, gates, cleanup, metadata, and summary |
| Runtime config | `FERRUM_*` switches are scattered across scripts, Rust hot paths, and CUDA/C++ code | central env registry plus typed config; hot-path direct env reads are reduced by `>= 85%` and every knob is visible in artifacts |
| Auto configuration | best M3 path is assembled from many manual env switches | vLLM-style typed config builder selects backend, graph, scheduler, and KV settings from model + hardware + workload and emits a decision trace |
| OpenAI API | chat works for smoke/bench, missing tools and several fields | SDK contract tests cover tools, stream options, JSON schema, usage, errors |
| Strict structured output | `json_object` is soft-biased; schema support is partial | unsupported strict schema is rejected; supported strict schema is validated before response |
| Correctness and regression gates | perf experiments can pass throughput while silently breaking behavior or another cell | every default-path change has mandatory correctness gates and no material perf regression |

## Milestone A: Incremental CUDA Build Boundary

### Required Changes

- Split the PTX build path in `crates/ferrum-kernels/build.rs` into per-kernel content-hash artifacts instead of one `bindgen_cuda` batch for every core `.cu`.
- Keep Marlin, vLLM-Marlin, vLLM-MoE-Marlin, vLLM paged-attn, and FA2 source builds behind independently cached artifacts.
- Add a build summary line for every CUDA artifact: `built`, `cache_hit`, `reason`, `elapsed_ms`, `inputs_hash`.
- Add a documented fast path for attention-only iteration that does not compile Marlin or MoE-Marlin unless their inputs changed.

### Quantitative Acceptance

- After a clean release CUDA build, touching only `crates/ferrum-kernels/kernels/paged_varlen_attention_vllm.cu` and rebuilding `ferrum-cli` with the M3 feature set completes in:
  - p50 `<= 75s`
  - p95 `<= 90s`
  - measured across 5 consecutive rebuilds on the restored RTX 4090 pod.
- The same attention-only rebuild logs `cache_hit` for:
  - `marlin`
  - `vllm_marlin`
  - `vllm_moe_marlin`
  - `vllm_paged_attn` unless that exact library is the edited target.
- Touching one vLLM-MoE-Marlin source file rebuilds only the vLLM-MoE-Marlin static library plus final Rust link; it does not rebuild PTX core kernels.
- `cargo check -q -p ferrum-kernels --features cuda,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source` still succeeds on the GPU pod.

## Milestone B: Structured Profiling and Artifact Schema

### Required Changes

- Replace primary grep-only profile lines (`[unified-prof]`, `[bucket-prof]`, `[iter-prof]`, `[graph-prof]`, `[MOE_DUMP:*]`, `[vllm-moe-config]`) with structured JSONL events.
- Keep human log lines only as a convenience layer; scripts must consume JSONL or chrome trace.
- Define a stable schema for profile events in `ferrum-bench-core`, including:
  - `event`
  - `commit_sha`
  - `env_hash`
  - `model`
  - `concurrency`
  - `shape`
  - `stage_us`
  - `graph_enabled`
  - `runtime_flags`
- Emit profile output path in the bench artifact manifest.

### Quantitative Acceptance

- `scripts/m3_route_unified_profile.sh` no longer greps server logs for required fields; it fails only by validating structured events.
- A c32 profile run produces at least these structured event groups:
  - `moe_dump`
  - `vllm_moe_config`
  - `unified_prof`
  - `unified_layer_prof` or `batched_decode_prof`
  - `iter_prof`
  - `bucket_prof` when MoE profiling is enabled
- JSON schema/unit tests reject missing required profile fields.
- Profile parser tests cover at least 3 fixture artifacts: default graph-on, graph-off route dump, and FA2/FA-layout attention A/B.
- Profile overhead remains bounded:
  - graph-on c32 throughput with structured low-intrusion profiling disabled changes by `< 1%` versus current default;
  - graph-off sync-timer profile is explicitly labeled as non-throughput data.

## Milestone C: Unified A/B Bench Runner

### Required Changes

- Replace copied `scripts/m3_*_ab.sh` logic with a single reusable runner that supports:
  - case matrix definitions;
  - env per case;
  - server launch and health wait;
  - process cleanup;
  - Paris gate;
  - multi-turn gate;
  - optional structured profile validation;
  - `bench-serve`;
  - artifact manifest;
  - same-binary A/B summary.
- Keep existing scripts as thin wrappers or delete them after migration.
- Add a preflight check that records GPU process state, git status, binary hash, feature set, and effective runtime config.

### Quantitative Acceptance

- At least 4 existing M3 scripts are migrated to the runner:
  - FA2 direct/source A/B;
  - FA-layout A/B;
  - route/unified profile;
  - one scheduler/admission A/B.
- Every run writes `manifest.json` with:
  - `git_head`
  - `git_status_short`
  - `binary_sha256`
  - `features`
  - `env_hash`
  - `case_env`
  - `model_dir`
  - `server_log`
  - `bench_json`
  - `profile_jsonl`
  - `correctness_gates`
  - `cleanup_status`
- Failed correctness gates stop before throughput measurement.
- Failed server health or interrupted runs leave no live `ferrum`, `cargo`, `nvcc`, `vllm`, or benchmark client process owned by the script.
- A same-binary N=3 c32 A/B can be started with one command and yields a summary table with mean, stddev, CI95 half-width, TTFT p50, TPOT p50, ITL p95, completed, and errored.

## Milestone D: Runtime Config Registry and Snapshot

### Required Changes

- Add a single environment-variable registry for all `FERRUM_*` knobs. Each entry must declare:
  - name;
  - type;
  - default;
  - owner crate/module;
  - scope: `runtime`, `benchmark`, `debug`, `build`, or `test`;
  - stability: `default`, `experimental`, `diagnostic`, or `deprecated`;
  - read phase: `build`, `startup`, `request`, or `test-only`;
  - replacement config key or CLI flag when applicable;
  - sunset condition for experimental and diagnostic knobs.
- Parse performance-affecting runtime env vars once into typed config structs and pass those structs through engine/model/backend boundaries instead of reading `std::env::var` throughout hot code.
- Replace CUDA/C++ `std::getenv` dispatch switches with typed launcher parameters or one-time init state. Remaining `getenv` reads must be diagnostic-only and registry allowlisted.
- Move common M3 benchmark defaults into named presets so scripts set a preset plus only case-specific overrides.
- Preserve explicit debug overrides, but make defaults, sources, and effective values visible.
- Include config snapshot in:
  - `/health` or a new diagnostics endpoint;
  - `bench-serve` reports;
  - structured profile events;
  - artifact manifests.

### Quantitative Acceptance

- Baseline measured on 2026-05-30:
  - Current status snapshot reports `152` unique `FERRUM_*` token names in scan scope and `146/146` registry entries;
  - `75` direct env reads across the same scope (`std::env::var`, `env::var`, `var_os`, `std::getenv`, `getenv`);
  - hot core paths (`ferrum-engine/src`, `ferrum-models/src`, `ferrum-kernels/src`, `ferrum-kernels/kernels`) contain `120` unique `FERRUM_*` names and `4` hot-path direct env reads.
- Registry coverage is `100%` for all `FERRUM_*` names found by static scan.
- CI/static check fails if a new `FERRUM_*` name appears without a registry entry and parser test.
- Direct env reads in hot core paths are now at `4` and remain fully classified.
- No direct env read is allowed inside per-token, per-layer, scheduler-decision, kernel-launch, or request-handling hot loops except allowlisted diagnostic probes that are cached before the loop.
- CUDA/C++ source has `<= 5` direct `getenv` reads, all diagnostic or build/probe related and registry allowlisted.
- All default-on behavior for M3 is represented in typed config:
  - `FERRUM_MOE_GRAPH`
  - `FERRUM_VLLM_MOE`
  - `FERRUM_VLLM_MOE_PAIR_IDS`
  - `FERRUM_USE_VLLM_PAGED_ATTN`
  - `FERRUM_VLLM_PAGED_ATTN_V1_SHORT`
  - `FERRUM_GREEDY_ARGMAX`
  - `FERRUM_FA_LAYOUT_VARLEN`
  - `FERRUM_FA2_SOURCE`
  - `FERRUM_FA2_DIRECT_FFI`
- No known no-op or placebo runtime switch remains undocumented. Deprecated aliases must warn once, map to the typed config, and have a removal target.
- M3 A/B scripts use a named runtime preset for common defaults and set `<= 3` case-specific env overrides per case, excluding port/path/log variables.
- Config snapshot is stable sorted JSON and includes for every non-default value:
  - key;
  - effective value;
  - source: default, config file, CLI, env, or script case;
  - whether it affects correctness, performance, memory, or diagnostics.
- Config snapshot diff between two A/B cases is machine-readable and included in the bench summary.
- Unit tests cover env parsing for boolean, integer, path, and tri-state default/forced-off/forced-on cases.

Notes:

- The numeric baseline in this section is the historical anchor from the initial status snapshot. The latest registry scan (from `docs/status/runtime-env-registry-2026-05-30.md`) currently reports `152` candidate env tokens, `146` registered entries, `75` direct read sites, and `4` hot-path direct reads in the current scan scope.
- Milestone D acceptance against this update is currently met if coverage remains complete and hot-path direct reads remain within the historical upper bound (`<= 26`), with direct pressure to reduce that further as conversion continues.

## Milestone E: vLLM-Style Auto Configuration and Backend Selection

### Reference Pattern

vLLM does not make the user discover the fast path by manually composing dozens of env vars. Its normal path is:

- parse CLI/config/env into typed engine config;
- infer model properties from model metadata;
- inspect platform and GPU capabilities;
- choose compatible backends from priority-ordered selectors;
- profile available GPU memory at startup and size the KV cache;
- allow manual overrides, but validate them and log why they changed the automatic choice.

Ferrum should adopt the same control-plane shape while keeping Ferrum-specific kernels, M3 constraints, and correctness gates.

### Required Changes

- Add a `FerrumConfigBuilder` or equivalent startup resolver that combines:
  - CLI args;
  - config file values;
  - registry-backed env overrides;
  - model metadata;
  - hardware capabilities;
  - workload preset.
- Add typed capability structs:
  - `ModelCapabilities`: architecture, quantization, MoE shape, context length, head dim, KV heads, supported dtypes, graph-safety constraints;
  - `HardwareCapabilities`: backend, CUDA runtime, compute capability, VRAM, SM count, supported dtypes, graph support, available compiled kernel features;
  - `WorkloadProfile`: serving mode, target concurrency, prompt/output length class, latency/throughput priority.
- Add selector modules with explicit candidate lists, validation, and priority order for:
  - attention prefill/mixed path;
  - attention decode path;
  - MoE route/GEMM path;
  - CUDA graph policy;
  - scheduler/admission policy;
  - KV cache sizing and max batched tokens.
- Emit an auto-config decision trace at startup and into artifacts. Each selected field must include:
  - selected value;
  - source: default, CLI, config file, env, model metadata, hardware capability, memory profile, or workload preset;
  - candidates considered;
  - rejected candidates with reasons;
  - whether the selection affects correctness, performance, memory, or diagnostics.
- Keep manual overrides, but validate them. Invalid combinations must fail fast or explicitly degrade with a warning and decision-trace reason.
- Treat diagnostic paths separately from product defaults:
  - runtime-loaded vLLM/Torch FA2 direct FFI can remain diagnostic-only;
  - source-built/Ferrum-owned FA2 or native kernels are eligible for auto selection only after correctness and non-regression gates pass.

### Ferrum M3 Target Behavior

- A single named preset, for example `m3_qwen3_30b_a3b_int4`, should replace the common M3 env bundle currently repeated across scripts.
- With the M3 preset and no case-specific overrides, the builder should select the current safe default equivalents for:
  - CUDA backend;
  - device-route MoE;
  - vLLM-Marlin MoE when available;
  - pair-id MoE combine path;
  - greedy GPU argmax;
  - paged attention;
  - graph-clean decode policy;
  - prefix cache off for the current benchmark profile unless explicitly enabled.
- If FA-layout or FA2-source paths are compiled and validated, the selector may choose them for M3 only when:
  - the dependency is Ferrum-owned or source-built, not a runtime vLLM/Torch extension;
  - the selected path has a current same-pod correctness and non-regression artifact;
  - the decision trace names the artifact used to justify default selection.

### Quantitative Acceptance

- Running the M3 server with the named preset requires `<= 2` performance-affecting env vars in scripts, excluding paths, ports, logging, and artifact destinations.
- At least `90%` of the current M3 default/fast-path switches are expressed as typed selector outputs rather than direct env-controlled branches.
- Startup writes `effective_config.json` and `decision_trace.jsonl`; schema tests validate both files.
- Decision trace covers at least these selections:
  - attention prefill/mixed backend;
  - attention decode backend;
  - MoE implementation;
  - MoE graph policy;
  - KV block count or KV memory budget;
  - max sequences;
  - max batched tokens;
  - scheduler chunk/admission policy;
  - sampling/readback path.
- Invalid override tests cover at least 10 combinations, including:
  - graph enabled with a graph-unsafe MoE path;
  - FA2 selected without compiled/source-built support;
  - BF16 selected on unsupported hardware;
  - max batched tokens smaller than max sequences;
  - KV cache budget too small for the requested max model length.
- The M3 named preset without the old env bundle is performance equivalent to the old explicit-env default:
  - c=1,4,16,32 full sweep with `n_repeats >= 3`;
  - no cell regresses by more than `3%`;
  - c32 does not regress by more than `2%`;
  - Paris single-turn and multi-turn gates pass.
- Auto-selection must not hide performance experiments:
  - every experimental candidate remains opt-in until it has a publishable artifact;
  - selector priority changes require the correctness and non-regression gates from Milestone I.

## Milestone F: Product API Compatibility

### Required Changes

- Extend OpenAI chat request/response support to include:
  - `tools`
  - `tool_choice`
  - assistant `tool_calls`
  - tool messages
  - legacy `functions` / `function_call` compatibility if needed by SDKs
  - `stream_options.include_usage`
  - `logprobs` / `top_logprobs` rejection or implementation with correct error semantics
  - tokenizer-based usage accounting
  - `n` either implemented or rejected when `n != 1`
- Replace internal `InferenceRequest { prompt: String }` as the only product boundary with a structured request that can carry messages, tool definitions, tool results, and response constraints.
- Keep prompt rendering in a model-family chat-template layer, not in the HTTP handler.
- Implement `/v1/completions` or remove the claim of compatibility for that endpoint from product docs.

### Quantitative Acceptance

- Add non-ignored API contract tests using a stub or deterministic model for:
  - basic chat;
  - streaming chat;
  - `stream_options.include_usage`;
  - `n=2` rejection or two-choice response;
  - tool call request parsing;
  - assistant tool call response serialization;
  - tool role message parsing;
  - unsupported multimodal content returning 400 instead of silent drop;
  - `logit_bias` rejection or implementation;
  - completions endpoint behavior.
- Add ignored real-model SDK tests for `async-openai` and one Python OpenAI SDK compatibility smoke.
- Usage accounting uses tokenizer counts, not whitespace counts:
  - prompt token count differs from tokenizer result by `0` on test fixtures;
  - completion token count equals generated token count.
- Error responses map correctly:
  - invalid request -> HTTP 400, `type=invalid_request_error`;
  - unsupported feature -> HTTP 400 or 422 with clear `param`;
  - engine unavailable -> HTTP 503;
  - internal generation failure -> HTTP 500.
- OpenAI API compatibility report documents each supported/unsupported field with tests linked.

## Milestone G: Strict JSON and Schema Output

### Required Changes

- Treat `response_format.json_schema.strict=true` as a contract:
  - if the schema subset is supported, enforce it with hard masking and validate returned JSON;
  - if unsupported, reject at request boundary.
- Treat `json_object` as best-effort unless hard JSON grammar masking is implemented; document behavior accurately.
- Add response validation before non-streaming return and final streaming completion when feasible.

### Quantitative Acceptance

- Supported strict schemas pass 100 consecutive deterministic stub-model tests without invalid JSON.
- Unsupported strict schemas reject with HTTP 400 and `param=response_format.json_schema`.
- Real-model smoke for a simple object schema passes 20/20 runs at temperature 0.
- No markdown fence stripping is needed for strict schema success in the deterministic test path.
- Existing `response_format=json_object` tests remain green, but product docs explicitly mark it as best-effort until a full JSON grammar mask is shipped.

## Milestone H: Codebase Shape and Ownership

### Required Changes

- Split oversized modules along operational boundaries:
  - `qwen3_moe.rs`: model state, scratch allocation, MoE config, and profiling counters separated;
  - `qwen3_moe_forward_unified.rs`: attention plan, MoE plan, final sampling/readback, and graph/profile wrappers separated;
  - `continuous_engine.rs`: request state, scheduler loop, batch materialization, streaming, and completion handling separated;
  - `backend/traits.rs`: capability traits moved to smaller files.
- Convert long kernel launch signatures into typed parameter structs where the same group of fields recurs.

### Quantitative Acceptance

- No single Rust source file in these core paths exceeds `1500` lines:
  - `ferrum-engine/src/continuous_engine.rs`
  - `ferrum-models/src/models/qwen3_moe.rs`
  - `ferrum-kernels/src/backend/traits.rs`
- `qwen3_moe_forward_unified.rs` is `<= 700` lines or split by stage.
- No new function in model/backend hot paths has more than `15` parameters unless it is a low-level FFI boundary.
- Existing local gates pass:
  - `cargo fmt --all -- --check`
  - `cargo check -q -p ferrum-cli`
  - `cargo test -q -p ferrum-engine --test continuous_batch_test`
  - `cargo test -q -p ferrum-scheduler`

## Milestone I: Correctness and Performance Non-Regression Gates

### Required Changes

- Add a mandatory validation checklist for every change that touches:
  - CUDA kernels or CUDA build logic;
  - model forward paths;
  - scheduler/admission policy;
  - sampling / structured output;
  - OpenAI server request or response types;
  - runtime default flags.
- Make benchmark artifacts record both correctness gates and performance regression gates, not just throughput.
- Define a stable default-path baseline artifact before changing defaults.
- Keep opt-in experiments clearly separated from default-runtime validation.

### Correctness Acceptance

- Any default-path performance PR must pass all relevant local gates:
  - `cargo fmt --all -- --check`
  - `cargo check -q -p ferrum-cli`
  - `cargo test -q -p ferrum-engine --test continuous_batch_test`
  - `cargo test -q -p ferrum-scheduler`
  - kernel-specific parity tests for changed kernel families.
- Any CUDA kernel change must include a parity gate with documented tolerance:
  - attention kernels: max absolute error `<= 5e-3` and max relative error `<= 1e-2` against the current reference on representative decode, prefill, and mixed shapes;
  - MoE/Marlin kernels: existing quantized parity thresholds must be preserved or tightened, never loosened without a written numerical reason;
  - graph-enabled and graph-disabled paths must both pass at least one smoke when the changed code can run under graph capture.
- Any server/API change must pass non-ignored deterministic contract tests for request parsing, response serialization, streaming SSE shape, error mapping, and usage accounting.
- Any full-model GPU validation used for performance claims must first pass:
  - Paris single-turn gate;
  - Paris multi-turn gate;
  - `bench-serve` completion rate `100%`;
  - error count `0`;
  - no server panic or CUDA error in logs.
- If a correctness gate is skipped, the artifact manifest must mark the run `not_publishable=true` and include the reason.

### Performance Non-Regression Acceptance

- For any default-path change, compare against a same-pod baseline with the same model, binary feature set, GPU lock state, dataset, prompt/output lengths, and runtime config except for the intended change.
- M3 default-path throughput must not regress materially:
  - c=1,4,16,32 full sweep with `n_repeats >= 3`;
  - no cell may regress by more than `3%` versus baseline unless CI95 intervals overlap and the regression is explicitly classified as noise;
  - c32 must not regress by more than `2%` for changes touching attention, MoE, graph, scheduler, or sampling hot paths.
- Latency must not trade away product usability:
  - TTFT p50 may not regress by more than `10%`;
  - ITL p95 may not regress by more than `10%`;
  - TPOT p50 may not regress by more than `5%`;
  - any intentional latency tradeoff must be opt-in, documented, and excluded from default-path completion.
- Build-loop changes must not regress build speed:
  - attention-only rebuild p95 remains `<= 90s`;
  - no unrelated static library changes from `cache_hit` to `built` in the attention-only rebuild gate.
- API-only changes must not regress M3 c32 throughput by more than `1%` on a same-binary smoke, or must prove the touched path is not exercised by the benchmark.
- Opt-in experiments may regress default-path metrics only if the default path is proven unchanged by a forced-off A/B row.

### Publishable Artifact Requirements

Every publishable validation artifact must include:

- baseline and candidate git SHAs;
- git dirty status;
- binary SHA256;
- runtime config snapshot;
- env hash;
- GPU process preflight and cleanup status;
- correctness gate results;
- throughput/latency regression table;
- explicit verdict: `pass`, `fail`, or `diagnostic-only`.

## Final Completion Criteria

This goal is complete when all milestones A through I meet their quantitative acceptance criteria and one full validation packet is committed under `docs/bench/` or `docs/status/` containing:

- build timing table before/after;
- structured profile sample and schema validation result;
- migrated runner example artifact;
- env registry, named preset, and runtime config snapshot example;
- auto-config decision trace and selector validation report;
- OpenAI API compatibility matrix;
- strict JSON/schema validation report;
- code-size and ownership summary;
- correctness and performance non-regression report;
- exact commands used for local and GPU validation.

## Non-Goals

- This goal does not require a new M3 throughput record.
- This goal does not require FA2 source to become default.
- This goal does not require implementing multimodal chat input beyond rejecting unsupported parts correctly.
- This goal does not require full JSON Schema support; unsupported strict schemas may be rejected.
- This goal does not require deleting all env vars; it requires typed resolution, registry ownership, source attribution, and artifact visibility.
