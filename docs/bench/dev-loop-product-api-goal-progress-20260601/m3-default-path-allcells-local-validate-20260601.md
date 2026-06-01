# m3_default_path_allcells_ab.sh local validate-only run (2026-06-01)

## Command

```bash
cd /Users/chejinxuan/rust_ws/ferrum-infer-rs
OUT_ROOT=/tmp/m3-default-path-allcells-local-validate-20260601-$(date +%Y%m%d_%H%M%S)
BASELINE_ENV_JSON="$(python3 - <<'PY'
import json
print(json.dumps({'FERRUM_FA_LAYOUT_VARLEN':'1'}))
PY
)"
CANDIDATE_ENV_JSON="$(python3 - <<'PY'
import json
print(json.dumps({'FERRUM_FA_LAYOUT_VARLEN':'1','FERRUM_FA2_SOURCE':'1'}))
PY
)"
MODEL_DIR=/tmp BIN=/bin/echo REPEATS=1 WARMUP_REQUESTS=0 NUM_PROMPTS=1 \
VALIDATE_ONLY=1 OUT_ROOT="$OUT_ROOT" \
BASELINE_ENV_JSON="$BASELINE_ENV_JSON" CANDIDATE_ENV_JSON="$CANDIDATE_ENV_JSON" \
bash scripts/m3_default_path_allcells_ab.sh
```

## Result

- Exit code: `0`
- Validation-only output: per-cell config checks passed for all four concurrency cells (`c1`, `c4`, `c16`, `c32`).
- Observed stdout marker: `config ok` for each cell.

## Evidence notes

- This run exercises the local config-generation/validation path only (`VALIDATE_ONLY=1`) and does not launch `ferrum`.
- This also verifies the default touched-area set is now accepted by the shared validation vocabulary.
