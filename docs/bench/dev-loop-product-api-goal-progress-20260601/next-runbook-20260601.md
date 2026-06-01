# Next Runbook — Dev Loop Product API Goal (2026-06-01)

This runbook is the executor-facing plan for continuing
[`docs/dev-loop-product-api-goal-2026-05-30.md`](../../dev-loop-product-api-goal-2026-05-30.md).

## 0) Preconditions

```bash
cd /Users/chejinxuan/rust_ws/ferrum-infer-rs
python3 scripts/check_ferrum_env_registry.py --self-test
python3 scripts/check_ferrum_env_registry.py --fail-on-registry-gap
python3 scripts/m3_validate_runner_artifact.py --self-test
python3 scripts/m3_collect_allcell_runner_artifacts.py --self-test
python3 scripts/validate_real_model_api_smoke.py --self-test
python3 scripts/check_fa2_source_native.py --self-test
python3 scripts/check_fa2_source_native.py
python3 scripts/check_runtime_snapshot_boundary.py --self-test
python3 scripts/check_runtime_snapshot_boundary.py
python3 scripts/m3_cuda_build_boundary_probe.py --self-test
python3 scripts/validate_cuda_build_summary.py --self-test
python3 scripts/validate_cuda_build_boundary_manifest.py --self-test
which nvcc
which nvidia-smi
```

If `nvcc` / `nvidia-smi` are missing, run this runbook on the restored CUDA pod.

## 1) Milestone A — 5-run release rebuild boundary probe

```bash
OUT_A=/workspace/m3-release-touch-probe-20260601-$(date +%Y%m%d_%H%M%S)
python3 scripts/m3_cuda_build_boundary_probe.py \
  --iterations 5 \
  --out "$OUT_A" \
  --fail-on-limit
python3 scripts/validate_cuda_build_boundary_manifest.py \
  --require-limits-pass \
  "$OUT_A/build_boundary_manifest.json"
```

Acceptance for Milestone A:

- Artifact exists: `"$OUT_A/build_boundary_manifest.json"`
- Manifest `limits_pass` is true
- `p50_ms <= 75_000` and `p95_ms <= 90_000`
- `status=built` only occurs for the edited target in the isolated scope

## 2) Milestone I — default-path all-cell same-pod packet

Use the dedicated all-cell wrapper to run `c=1/4/16/32` in one packet. The
candidate below is the restored native in-repo `fa2-source` path, not the old
external FlashAttention-source build and not the runtime vLLM/Torch direct FFI
shim. Do not satisfy this step with pre-2026-06-01 source-linked FA2 smoke
artifacts.

```bash
OUT_I=/workspace/m3-default-path-allcells-20260601-$(date +%Y%m%d_%H%M%S)
BASELINE_ENV_JSON='{"FERRUM_FA_LAYOUT_VARLEN":"1"}'
CANDIDATE_ENV_JSON='{"FERRUM_FA_LAYOUT_VARLEN":"1","FERRUM_FA2_SOURCE":"1"}'
OUT_ROOT="$OUT_I" \
  BASELINE_ENV_JSON="$BASELINE_ENV_JSON" \
  CANDIDATE_ENV_JSON="$CANDIDATE_ENV_JSON" \
  bash scripts/m3_default_path_allcells_ab.sh
```

Optional validation-only dry run:

```bash
BASELINE_ENV_JSON='{"FERRUM_FA_LAYOUT_VARLEN":"1"}'
CANDIDATE_ENV_JSON='{"FERRUM_FA_LAYOUT_VARLEN":"1","FERRUM_FA2_SOURCE":"1"}'
BASELINE_ENV_JSON="$BASELINE_ENV_JSON" \
  CANDIDATE_ENV_JSON="$CANDIDATE_ENV_JSON" \
  REPEATS=1 VALIDATE_ONLY=1 \
  bash scripts/m3_default_path_allcells_ab.sh
```

Acceptance for Milestone I:

- Aggregate summary shows all cells: `c1`, `c4`, `c16`, `c32`
- `concurrency_cells_ok == true`
- Baseline/candidate rows both present with `not_publishable == false`
- Validation checklist includes `bench_completion` and `paris`
- Candidate manifests show `FERRUM_FA2_SOURCE=1` without requiring
  `FERRUM_FA2_DIRECT_FFI_SHIM`, external FlashAttention source, or CUTLASS
  build inputs.

## 3) Milestone E — startup selector ownership verification

```bash
cargo test -q -p ferrum-types auto_config -- --nocapture
cargo test -q -p ferrum-types runtime_config -- --nocapture
cargo test -q -p ferrum-cli config -- --nocapture
cargo test -q -p ferrum-cli runtime-env -- --nocapture
cargo test -q -p ferrum-cli source-resolver -- --nocapture
cargo test -q -p ferrum-cli commands::serve -- --nocapture
cargo test -q -p ferrum-server route_health_includes_runtime_config_snapshot -- --nocapture
```

Acceptance for Milestone E:

- `ferrum` artifacts (`effective_config.json`, `decision_trace.jsonl`) show source/effect metadata from builder defaults and explicit override precedence (`CLI > env > config_file > default`).
- No unexpected fallback to legacy env-bundle defaults for benchmark/model/admin paths being validated.

## 4) Milestone F/G — real-model API evidence

Open
[`docs/status/openai-api-compat-2026-05-30.md`](../../status/openai-api-compat-2026-05-30.md)
for the ignored SDK smoke list and run those tests against a real model on GPU.

On a real-model host with `Qwen/Qwen3-0.6B` available, run. Use
`CARGO_FEATURES=metal` on the local Metal path, or the appropriate CUDA feature
set on the restored CUDA pod.

```bash
OUT_F=${OUT_F:-/workspace/m3-real-model-api-smoke-20260601-$(date +%Y%m%d_%H%M%S)}
MODEL=qwen3:0.6b \
FERRUM_BIN=ferrum \
CARGO_FEATURES=metal \
OUT_ROOT="$OUT_F" \
bash scripts/m3_real_model_api_smoke.sh

# quick local assertion before commit
test -f "$OUT_F/commands.md"
test -f "$OUT_F/run_summary.json"
python3 scripts/validate_real_model_api_smoke.py "$OUT_F"
```

Optional knobs:

```bash
# Only async-openai tests
ASYNC_TESTS=1 PYTHON_TEST=0 OUT_ROOT=$OUT_F bash scripts/m3_real_model_api_smoke.sh

# Skip model pull when cache is already warm
PULL_MODEL=0 OUT_ROOT=$OUT_F bash scripts/m3_real_model_api_smoke.sh

# Use an explicit local/release binary for model pull
FERRUM_BIN=target/release/ferrum OUT_ROOT=$OUT_F bash scripts/m3_real_model_api_smoke.sh

# Run python openai smoke only
ASYNC_TESTS=0 PYTHON_TEST=1 OUT_ROOT=$OUT_F bash scripts/m3_real_model_api_smoke.sh
```

The resulting artifact root should be committed under `docs/bench/` with:

- `commands.md` (command-by-command transcript and return codes)
- `run_summary.json` (`all_passed` flag and command status list)
- individual `cargo-test-*.log` logs
- `python3 scripts/validate_real_model_api_smoke.py "$OUT_F"` passing output
- committed under
  `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-real-model-api-smoke-<ts>/`

The artifact should additionally capture real-model strict-schema and streaming behavior
and include any strict/non-strict latency notes needed for the completion packet.

Acceptance for Milestone F/G:

- `run_summary.json` exists and includes `all_passed == true` after all required runs.
- `scripts/validate_real_model_api_smoke.py "$OUT_F"` passes with the full
  required command set.
- Real-model strict-schema and streaming behavior are observed on a real server path, not stub-only.
- The evidence artifact includes both command logs and failure/success traces.

## 5) Post-run finalization

```bash
python3 scripts/m3_validate_runner_artifact.py "$OUT_I" \
  --require-bench
python3 scripts/validate_cuda_build_boundary_manifest.py --fail-on-limit \
  "$OUT_A/build_boundary_manifest.json"
python3 scripts/check_fa2_source_native.py
python3 scripts/check_runtime_snapshot_boundary.py
```
