#!/usr/bin/env bash
#
# All-cell A/B wrapper for the source-linked FA2 path.
#
# Usage on a GPU pod from the repo root:
#   OUT_ROOT=/workspace/m3-fa2-source-allcells-n3 \
#   REPEATS=3 \
#   bash scripts/m3_fa2_source_allcells_ab.sh

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/workspace/m3-fa2-source-allcells-ab-$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-10}"
PORT_BASE="${PORT_BASE:-18600}"
CONCURRENCIES="${CONCURRENCIES:-1 4 16 32}"
VALIDATION_CHANGE_TYPE="${VALIDATION_CHANGE_TYPE:-opt_in_experiment}"

mkdir -p "$OUT_ROOT"

echo "OUT_ROOT=$OUT_ROOT"
echo "REPEATS=$REPEATS"
echo "NUM_PROMPTS=$NUM_PROMPTS"
echo "WARMUP_REQUESTS=$WARMUP_REQUESTS"
echo "CONCURRENCIES=$CONCURRENCIES"
echo "VALIDATION_CHANGE_TYPE=$VALIDATION_CHANGE_TYPE"

{
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "repo=$(pwd)"
    echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_status_short_begin"
    git status --short 2>/dev/null || true
    echo "git_status_short_end"
    echo "repeats=$REPEATS"
    echo "num_prompts=$NUM_PROMPTS"
    echo "warmup_requests=$WARMUP_REQUESTS"
    echo "concurrencies=$CONCURRENCIES"
    echo "validation_change_type=$VALIDATION_CHANGE_TYPE"
} >"$OUT_ROOT/metadata.txt"

index=0
for concurrency in $CONCURRENCIES; do
    cell_root="$OUT_ROOT/c${concurrency}"
    cell_port_base=$((PORT_BASE + index * 10))
    echo "=== c=${concurrency} port_base=${cell_port_base} ==="
    OUT_ROOT="$cell_root" \
        FA2_SOURCE=1 \
        FA2_EXTRA_LD_LIBRARY_PATH="" \
        REPEATS="$REPEATS" \
        NUM_PROMPTS="$NUM_PROMPTS" \
        WARMUP_REQUESTS="$WARMUP_REQUESTS" \
        CONCURRENCY="$concurrency" \
        PORT_BASE="$cell_port_base" \
        VALIDATION_CHANGE_TYPE="$VALIDATION_CHANGE_TYPE" \
        bash scripts/m3_fa2_direct_ffi_ab.sh
    index=$((index + 1))
done

if [[ "${VALIDATE_ONLY:-0}" != "1" ]]; then
    python3 scripts/m3_collect_allcell_runner_artifacts.py "$OUT_ROOT" \
        --baseline-case fa_layout \
        --candidate fa2_source \
        --change-type "$VALIDATION_CHANGE_TYPE"
    if [[ "${VALIDATE_ARTIFACT:-1}" == "1" ]]; then
        python3 scripts/m3_validate_runner_artifact.py "$OUT_ROOT"
    fi
fi

python3 - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
print("ALLCELL_SUMMARY_BEGIN")
for cell in sorted(root.glob("c*"), key=lambda p: int(p.name[1:])):
    rows = {}
    for path in sorted(cell.glob("*_c*_n*/bench.json")):
        data = json.load(open(path))
        throughput = data.get("output_throughput_tps") or {}
        name = path.parent.name.split("_c", 1)[0]
        rows[name] = (
            throughput.get("mean", data.get("output_throughput")),
            throughput.get("ci95_hw"),
        )
    source = rows.get("fa2_source", (None, None))[0]
    layout = rows.get("fa_layout", (None, None))[0]
    delta = None if not source or not layout else (source / layout - 1.0) * 100.0
    print(
        f"ALLCELL {cell.name}: "
        f"fa2_source={source} fa_layout={layout} delta_pct={delta}"
    )
print("ALLCELL_SUMMARY_END")
PY
