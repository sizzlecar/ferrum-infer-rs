# m3_real_model_api_smoke.sh local script validation (2026-06-01)

## Commands

```bash
cd /Users/chejinxuan/rust_ws/ferrum-infer-rs
bash -n scripts/m3_real_model_api_smoke.sh
OUT_ROOT=/tmp/m3-real-model-api-smoke-empty-check \
  PULL_MODEL=0 RUN_PYTHON_CHECK=0 ASYNC_TESTS=0 PYTHON_TEST=0 \
  scripts/m3_real_model_api_smoke.sh
python3 scripts/validate_real_model_api_smoke.py --self-test
OUT_ROOT=/tmp/m3-real-model-api-smoke-partial-check \
  FERRUM_BIN=/usr/bin/true PULL_MODEL=1 RUN_PYTHON_CHECK=0 \
  ASYNC_TESTS=0 PYTHON_TEST=0 \
  scripts/m3_real_model_api_smoke.sh
python3 scripts/validate_real_model_api_smoke.py \
  --allow-partial \
  /tmp/m3-real-model-api-smoke-partial-check
python3 scripts/validate_real_model_api_smoke.py \
  /tmp/m3-real-model-api-smoke-partial-check
```

## Result

- `bash -n` exit code: `0`
- Empty-command guard exit code: `1`
- The empty-command guard still wrote
  `/tmp/m3-real-model-api-smoke-empty-check/run_summary.json`.
- The generated summary had no commands and `all_passed=false`.
- `python3 scripts/validate_real_model_api_smoke.py --self-test` exit code:
  `0`.
- Partial one-command smoke using `FERRUM_BIN=/usr/bin/true` exit code: `0`.
- Partial one-command artifact validation with `--allow-partial` exit code:
  `0`.
- Full-suite validation for the same partial artifact exit code: `1`, with
  `missing required smoke commands`, proving partial debug packets cannot
  satisfy the F/G completion validator.

## Evidence notes

- This is not real-model API evidence.
- This verifies only the executor script's local artifact behavior:
  `run_summary.json` is written even when the run is not acceptable, and the
  script returns non-zero for an invalid/no-command evidence packet.
- The script now uses Python `time.time_ns()` for millisecond timing instead
  of GNU-specific `date +%s%3N`, so `elapsed_ms` generation works on macOS and
  Linux executor hosts.
