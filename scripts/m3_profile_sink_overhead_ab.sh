#!/usr/bin/env bash
#
# Thin wrapper around scripts/m3_ab_runner.py for profile sink overhead A/B.
# The profile_sink row only enables the structured JSONL sink metadata path;
# it does not enable diagnostic timers, dumps, or profile gates.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
BIN="${BIN:-./target/release/ferrum}"
OUT_ROOT="${OUT_ROOT:-/workspace/m3-profile-sink-overhead-ab-$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-1}"
PORT_BASE="${PORT_BASE:-18740}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
CONCURRENCY="${CONCURRENCY:-32}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-256}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"
FEATURES="${FEATURES:-cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin}"
ARTIFACT_VERDICT="${ARTIFACT_VERDICT:-diagnostic-only}"

export MODEL_DIR HF_MODEL BIN OUT_ROOT REPEATS PORT_BASE NUM_PROMPTS WARMUP_REQUESTS
export CONCURRENCY RANDOM_INPUT_LEN RANDOM_OUTPUT_LEN FEATURES BUILD="${BUILD:-0}"
export ARTIFACT_VERDICT

mkdir -p "$OUT_ROOT"
CONFIG="$OUT_ROOT/runner_config.json"

python3 - "$CONFIG" <<'PY'
import json
import os
import sys

config_path = sys.argv[1]
port_base = int(os.environ.get("PORT_BASE", "18740"))
artifact_verdict = os.environ["ARTIFACT_VERDICT"]
config = {
    "name": "m3-profile-sink-overhead-ab",
    "out_root": os.environ["OUT_ROOT"],
    "model_dir": os.environ["MODEL_DIR"],
    "hf_model": os.environ["HF_MODEL"],
    "bin": os.environ["BIN"],
    "features": os.environ["FEATURES"],
    "build": os.environ.get("BUILD", "0") == "1",
    "artifact_verdict": artifact_verdict,
    "preset": "m3_qwen3_30b_a3b_int4",
    "port_base": port_base,
    "repeats": int(os.environ["REPEATS"]),
    "num_prompts": int(os.environ["NUM_PROMPTS"]),
    "warmup_requests": int(os.environ["WARMUP_REQUESTS"]),
    "concurrency": int(os.environ["CONCURRENCY"]),
    "random_input_len": int(os.environ["RANDOM_INPUT_LEN"]),
    "random_output_len": int(os.environ["RANDOM_OUTPUT_LEN"]),
    "baseline_case": "default",
    "gates": {"paris": True, "multi_turn": False},
    "validation": {
        "change_type": "diagnostic",
        "touched_areas": ["profile_output", "benchmark_harness"],
        "performance_regression_required": True,
    },
    "profile": {"profile_env_cases": ["profile_sink"]},
    "cases": [
        {"name": "default", "port": port_base, "env": {}},
        {"name": "profile_sink", "port": port_base + 1, "env": {}},
    ],
}
if artifact_verdict != "pass":
    config["not_publishable_reason"] = (
        "profile-sink-only overhead smoke; use N>=3 before publishing"
    )
reason = os.environ.get("NOT_PUBLISHABLE_REASON")
if reason:
    config["not_publishable_reason"] = reason

with open(config_path, "w") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "OUT_ROOT=$OUT_ROOT"
RUNNER_FLAGS=()
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
    RUNNER_FLAGS+=(--validate-only)
fi
python3 scripts/m3_ab_runner.py --config "$CONFIG" "${RUNNER_FLAGS[@]}"
