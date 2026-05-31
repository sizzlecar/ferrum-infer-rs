#!/usr/bin/env bash
#
# Thin wrapper around scripts/m3_ab_runner.py for the c=32 route-shape
# and unified-engine profile gate. This remains a diagnostic profile run,
# not a throughput confirmation sweep.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
BIN="${BIN:-./target/release/ferrum}"
OUT_ROOT="${OUT_ROOT:-/workspace/m3-route-unified-profile-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-18143}"
CONCURRENCY="${CONCURRENCY:-32}"
TOP_K="${TOP_K:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-256}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"
FEATURES="${FEATURES:-cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin}"

export MODEL_DIR HF_MODEL BIN OUT_ROOT PORT CONCURRENCY TOP_K NUM_PROMPTS WARMUP_REQUESTS
export RANDOM_INPUT_LEN RANDOM_OUTPUT_LEN FEATURES BUILD="${BUILD:-0}"

mkdir -p "$OUT_ROOT"
CONFIG="$OUT_ROOT/runner_config.json"

python3 - "$CONFIG" <<'PY'
import json
import os
import sys

config_path = sys.argv[1]
port = int(os.environ.get("PORT", "18143"))
concurrency = int(os.environ["CONCURRENCY"])
top_k = int(os.environ["TOP_K"])
target_pairs = str(concurrency * top_k)
config = {
    "name": "m3-route-unified-profile",
    "out_root": os.environ["OUT_ROOT"],
    "model_dir": os.environ["MODEL_DIR"],
    "hf_model": os.environ["HF_MODEL"],
    "bin": os.environ["BIN"],
    "features": os.environ["FEATURES"],
    "build": os.environ.get("BUILD", "0") == "1",
    "artifact_verdict": "diagnostic-only",
    "not_publishable_reason": "graph-off sync-timer route/profile run; throughput is diagnostic-only",
    "preset": "m3_qwen3_30b_a3b_int4",
    "port_base": port,
    "repeats": 1,
    "num_prompts": int(os.environ["NUM_PROMPTS"]),
    "warmup_requests": int(os.environ["WARMUP_REQUESTS"]),
    "concurrency": concurrency,
    "random_input_len": int(os.environ["RANDOM_INPUT_LEN"]),
    "random_output_len": int(os.environ["RANDOM_OUTPUT_LEN"]),
    "gates": {"paris": True, "multi_turn": False},
    "validation": {
        "change_type": "diagnostic",
        "touched_areas": ["profile_output", "model_forward", "moe_route_dump"],
        "performance_regression_required": False,
    },
    "base_env": {
        "FERRUM_MOE_GRAPH": "0",
        "FERRUM_VLLM_PAGED_ATTN_V1_SHORT": "0",
        "FERRUM_MOE_DUMP": "1",
        "FERRUM_MOE_DUMP_BATCH_X_TOPK": target_pairs,
        "FERRUM_VLLM_MOE_LOG_CONFIG": "1",
        "FERRUM_VLLM_MOE_LOG_CONFIG_MIN_PAIRS": os.environ.get("FERRUM_VLLM_MOE_LOG_CONFIG_MIN_PAIRS", target_pairs),
        "FERRUM_VLLM_MOE_LOG_CONFIG_MAX_PAIRS": os.environ.get("FERRUM_VLLM_MOE_LOG_CONFIG_MAX_PAIRS", target_pairs),
        "FERRUM_VLLM_MOE_LOG_CONFIG_LIMIT": os.environ.get("FERRUM_VLLM_MOE_LOG_CONFIG_LIMIT", "32"),
        "FERRUM_UNIFIED_POST_PROF": "1",
        "FERRUM_UNIFIED_LAYER_PROF": "1",
        "FERRUM_UNIFIED_LAYER_PROF_EVERY": os.environ.get("FERRUM_UNIFIED_LAYER_PROF_EVERY", "16"),
        "FERRUM_UNIFIED_LAYER_PROF_MAX_M": os.environ.get("FERRUM_UNIFIED_LAYER_PROF_MAX_M", "64"),
        "FERRUM_UNIFIED_LAYER_PROF_MIN_SEQS": os.environ.get("FERRUM_UNIFIED_LAYER_PROF_MIN_SEQS", str(concurrency)),
        "FERRUM_DECODE_OP_PROFILE": "1",
        "FERRUM_BATCH_DECODE_PROF": "1",
        "FERRUM_NEXT_BATCH_PROF": "1",
        "FERRUM_MOE_PROFILE": "1",
    },
    "cases": [{"name": "route_profile", "port": port, "env": {}}],
    "profile": {
        "structured": True,
        "required_events": ["moe_dump", "vllm_moe_config", "unified_prof", "iter_prof", "bucket_prof"],
        "required_any_events": [["unified_layer_prof", "batched_decode_prof"]],
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
