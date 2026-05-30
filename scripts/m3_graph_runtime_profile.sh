#!/usr/bin/env bash
#
# Thin wrapper around scripts/m3_ab_runner.py for a low-intrusion M3 c=32
# graph-on runtime profile. This remains a diagnostic profile run, not a
# throughput confirmation sweep.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
BIN="${BIN:-./target/release/ferrum}"
OUT_ROOT="${OUT_ROOT:-/workspace/m3-graph-runtime-profile-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-18164}"
CONCURRENCY="${CONCURRENCY:-32}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-10}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-256}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"
FEATURES="${FEATURES:-cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin}"

export MODEL_DIR HF_MODEL BIN OUT_ROOT PORT CONCURRENCY NUM_PROMPTS WARMUP_REQUESTS
export RANDOM_INPUT_LEN RANDOM_OUTPUT_LEN FEATURES BUILD="${BUILD:-0}"

mkdir -p "$OUT_ROOT"
CONFIG="$OUT_ROOT/runner_config.json"

python3 - "$CONFIG" <<'PY'
import json
import os
import sys

config_path = sys.argv[1]
port = int(os.environ.get("PORT", "18164"))
config = {
    "name": "m3-graph-runtime-profile",
    "out_root": os.environ["OUT_ROOT"],
    "model_dir": os.environ["MODEL_DIR"],
    "hf_model": os.environ["HF_MODEL"],
    "bin": os.environ["BIN"],
    "features": os.environ["FEATURES"],
    "build": os.environ.get("BUILD", "0") == "1",
    "artifact_verdict": "diagnostic-only",
    "not_publishable_reason": (
        "graph-on diagnostic profile run with runtime/profile timers enabled; "
        "throughput is diagnostic-only"
    ),
    "preset": "m3_qwen3_30b_a3b_int4",
    "port_base": port,
    "repeats": 1,
    "num_prompts": int(os.environ["NUM_PROMPTS"]),
    "warmup_requests": int(os.environ["WARMUP_REQUESTS"]),
    "concurrency": int(os.environ["CONCURRENCY"]),
    "random_input_len": int(os.environ["RANDOM_INPUT_LEN"]),
    "random_output_len": int(os.environ["RANDOM_OUTPUT_LEN"]),
    "gates": {"paris": True, "multi_turn": False},
    "validation": {
        "change_type": "diagnostic",
        "touched_areas": ["profile_output", "model_forward"],
        "performance_regression_required": False,
    },
    "base_env": {
        "FERRUM_UNIFIED_POST_PROF": "1",
        "FERRUM_BATCH_DECODE_PROF": "1",
        "FERRUM_RBD_PROF": "1",
        "FERRUM_NEXT_BATCH_PROF": "1",
        "FERRUM_GRAPH_PROF": "1",
    },
    "cases": [{"name": "graph_runtime_profile", "port": port, "env": {}}],
    "profile": {
        "structured": True,
        "required_events": ["unified_prof", "iter_prof", "graph_prof"],
        "required_any_events": [["batched_decode_prof", "unified_layer_prof"]],
    },
}
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
