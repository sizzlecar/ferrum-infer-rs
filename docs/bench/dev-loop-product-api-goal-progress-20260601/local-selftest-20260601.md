# dev-loop-product-api-goal local self-test evidence (2026-06-01)

## Scope

- Objective anchor: [docs/dev-loop-product-api-goal-2026-05-30.md](../dev-loop-product-api-goal-2026-05-30.md)
- Workspace: `/Users/chejinxuan/rust_ws/ferrum-infer-rs`
- Execution time: `2026-06-01 10:19:37 +0800`

## Commands run (and result)

| Command | Result |
|---|---|
| `python3 scripts/m3_cuda_build_boundary_probe.py --self-test` | `m3_cuda_build_boundary_probe self-test ok` |
| `python3 scripts/validate_cuda_build_summary.py --self-test` | `validate_cuda_build_summary self-test ok` |
| `python3 scripts/validate_cuda_build_boundary_manifest.py --self-test` | `validate_cuda_build_boundary_manifest self-test ok` |
| `python3 scripts/m3_validate_runner_artifact.py --self-test` | `m3_validate_runner_artifact self-test ok` |
| `python3 scripts/m3_collect_allcell_runner_artifacts.py --self-test` | `m3_collect_allcell_runner_artifacts self-test ok` |
| `python3 scripts/check_ferrum_env_registry.py --self-test` | `check_ferrum_env_registry self-test ok` |

## Env registry summary snapshot

A full JSON summary was generated with:

```bash
python3 scripts/check_ferrum_env_registry.py --json --fail-on-registry-gap
```

Key summary points from that run:

- `candidate_names`: 152
- `registered_entries`: 146 (`146/146` covered)
- `unique_ferrum_env_candidates`: 151
- `direct_env_reads`: 75
- `hot_direct_env_reads`: 4
- `hot_direct_env_reads_classified`: 4
- `hot_direct_env_reads_unclassified`: 0

Additional machine-readable artifact persisted with:

```bash
python3 scripts/check_ferrum_env_registry.py --json --fail-on-registry-gap \
  > docs/bench/dev-loop-product-api-goal-progress-20260601/registry-json-snapshot-20260601.json
```

Saved snapshot summary:

- `files_scanned`: 583
- `candidate_names`: 151
- `unique_names`: 152
- `hot_unique_names`: 120
- `direct_env_reads`: 75
- `process_env_writes`: 24
- `non_test_process_env_writes`: 1
- `hot_direct_env_reads_classified`: 4
- `hot_direct_env_reads_unclassified`: 0

For the full machine-readable output, run the same command again or capture into an artifact file in the same format.

## Current status relative to objective blockers

- This run confirms local tooling gates remain green.
- No new implementation artifacts were produced in this run.
- Binding blockers remain unchanged (`A`/`E`/`I`/`F-G`) requiring GPU-backed or broader benchmark/API evidence.

## Milestone-A local execution status

Attempted Milestone A real probe with CUDA build path:

- Command: `python3 scripts/m3_cuda_build_boundary_probe.py --iterations 5 --out /tmp/m3-release-touch-probe-20260601-01 --fail-on-limit --no-cargo-verbose`
- Result: failed on run 1 due missing CUDA tooling (`nvcc` / `nvidia-smi`).
- Failure evidence: `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-release-touch-probe-20260601-01-run1-build.log`

## Threshold check

- `python3 scripts/check_ferrum_env_registry.py --json --fail-on-registry-gap --max-direct-env-reads 75 --max-process-env-writes 24 --max-non-test-process-env-writes 1 --max-hot-direct-env-reads 4 > /tmp/registry-threshold-check-20260601.json` -> `pass` (artifact: `/tmp/registry-threshold-check-20260601.json`)
