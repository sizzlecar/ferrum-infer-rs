#!/usr/bin/env bash
#
# Thin wrapper for a default-path all-cell (c=1/4/16/32) A/B sweep.
#
# Usage:
#   OUT_ROOT=/workspace/m3-default-path-allcells-20260601 \
#   BASELINE_ENV_JSON='{"FERRUM_FA_LAYOUT_VARLEN":"1"}' \
#   CANDIDATE_ENV_JSON='{"FERRUM_FA_LAYOUT_VARLEN":"1","FERRUM_FA2_SOURCE":"1"}' \
#   bash scripts/m3_default_path_allcells_ab.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
BIN="${BIN:-./target/release/ferrum}"
OUT_ROOT="${OUT_ROOT:-/workspace/m3-default-path-allcells-ab-$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-10}"
PORT_BASE="${PORT_BASE:-18600}"
CONCURRENCIES="${CONCURRENCIES:-1 4 16 32}"

VALIDATION_CHANGE_TYPE="${VALIDATION_CHANGE_TYPE:-default_path}"
BASELINE_NAME="${BASELINE_NAME:-baseline}"
CANDIDATE_NAME="${CANDIDATE_NAME:-candidate}"
FEATURES="${FEATURES:-cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source}"
DEFAULT_BASELINE_ENV_JSON='{"FERRUM_FA_LAYOUT_VARLEN":"1"}'
DEFAULT_CANDIDATE_ENV_JSON='{"FERRUM_FA_LAYOUT_VARLEN":"1"}'
BASELINE_ENV_JSON="${BASELINE_ENV_JSON:-$DEFAULT_BASELINE_ENV_JSON}"
CANDIDATE_ENV_JSON="${CANDIDATE_ENV_JSON:-$DEFAULT_CANDIDATE_ENV_JSON}"
VALIDATION_TOUCHED_AREAS="${VALIDATION_TOUCHED_AREAS:-model_forward,attention_prefill_mixed_path,fa2_runtime_path}"
VALIDATION_PROFILE_ENV="${VALIDATION_PROFILE_ENV:-}"

mkdir -p "$OUT_ROOT"

echo "OUT_ROOT=$OUT_ROOT"
echo "CONCURRENCIES=$CONCURRENCIES"
echo "REPEATS=$REPEATS"
echo "NUM_PROMPTS=$NUM_PROMPTS"
echo "WARMUP_REQUESTS=$WARMUP_REQUESTS"
echo "VALIDATION_CHANGE_TYPE=$VALIDATION_CHANGE_TYPE"
echo "BASELINE_NAME=$BASELINE_NAME"
echo "CANDIDATE_NAME=$CANDIDATE_NAME"

index=0
for concurrency in $CONCURRENCIES; do
    CELL_ROOT="$OUT_ROOT/c${concurrency}"
    PORT="$((PORT_BASE + index * 10))"
    mkdir -p "$CELL_ROOT"

    CONCURRENCY="$concurrency" PORT="$PORT" REPEATS="$REPEATS" \
        NUM_PROMPTS="$NUM_PROMPTS" WARMUP_REQUESTS="$WARMUP_REQUESTS" \
        MODEL_DIR="$MODEL_DIR" HF_MODEL="$HF_MODEL" BIN="$BIN" FEATURES="$FEATURES" \
        BASELINE_NAME="$BASELINE_NAME" CANDIDATE_NAME="$CANDIDATE_NAME" \
        VALIDATION_CHANGE_TYPE="$VALIDATION_CHANGE_TYPE" \
        BASELINE_ENV_JSON="$BASELINE_ENV_JSON" CANDIDATE_ENV_JSON="$CANDIDATE_ENV_JSON" \
        VALIDATION_TOUCHED_AREAS="$VALIDATION_TOUCHED_AREAS" \
        VALIDATION_PROFILE_ENV="$VALIDATION_PROFILE_ENV" \
        python3 - "$CELL_ROOT" <<'PY'
import json
import os
import pathlib
import sys

cell_root = pathlib.Path(sys.argv[1])
concurrency = int(os.environ["CONCURRENCY"])
repeats = int(os.environ["REPEATS"])
num_prompts = int(os.environ["NUM_PROMPTS"])
warmup = int(os.environ["WARMUP_REQUESTS"])
port = int(os.environ["PORT"])


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


config = {
    "name": "m3-default-path-allcells-ab",
    "out_root": str(cell_root),
    "model_dir": os.environ["MODEL_DIR"],
    "hf_model": os.environ["HF_MODEL"],
    "bin": os.environ["BIN"],
    "features": os.environ["FEATURES"],
    "build": False,
    "preset": "m3_qwen3_30b_a3b_int4",
    "port_base": port,
    "repeats": repeats,
    "num_prompts": num_prompts,
    "warmup_requests": warmup,
    "concurrency": concurrency,
    "baseline_case": os.environ["BASELINE_NAME"],
    "gates": {"paris": True, "multi_turn": True},
    "validation": {
        "change_type": os.environ["VALIDATION_CHANGE_TYPE"],
        "touched_areas": split_csv(os.environ["VALIDATION_TOUCHED_AREAS"]),
        "performance_regression_required": True,
    },
    "cases": [
        {
            "name": os.environ["BASELINE_NAME"],
            "port": port,
            "env": json.loads(os.environ["BASELINE_ENV_JSON"]),
        },
        {
            "name": os.environ["CANDIDATE_NAME"],
            "port": port + 1,
            "env": json.loads(os.environ["CANDIDATE_ENV_JSON"]),
        },
    ],
}

validation_profile_env = (os.environ.get("VALIDATION_PROFILE_ENV") or "").strip()
if validation_profile_env:
    config["validation"]["benchmark_impact"] = {
        "m3_benchmark_exercised": True,
        "reason": "default-path benchmark path is exercised",
        "evidence": validation_profile_env,
    }

with open(cell_root / "runner_config.json", "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

    RUNNER_FLAGS=()
    if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
        RUNNER_FLAGS+=(--validate-only)
    fi
    python3 scripts/m3_ab_runner.py --config "$CELL_ROOT/runner_config.json" "${RUNNER_FLAGS[@]}"
    index=$((index + 1))
done

if [[ "${VALIDATE_ONLY:-0}" != "1" ]]; then
    python3 scripts/m3_collect_allcell_runner_artifacts.py \
        "$OUT_ROOT" \
        --baseline-case "$BASELINE_NAME" \
        --candidate "$CANDIDATE_NAME" \
        --change-type "$VALIDATION_CHANGE_TYPE"

    if [[ "${VALIDATE_ARTIFACT:-1}" == "1" ]]; then
        python3 scripts/m3_validate_runner_artifact.py --require-bench "$OUT_ROOT"
    fi
fi
