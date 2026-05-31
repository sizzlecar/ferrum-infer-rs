#!/usr/bin/env bash
#
# Thin wrapper around scripts/m3_ab_runner.py for the FA-compatible
# dual-layout K/V A/B.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
BIN="${BIN:-./target/release/ferrum}"
OUT_ROOT="${OUT_ROOT:-/workspace/m3-fa-layout-varlen-ab-$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-1}"
PORT_BASE="${PORT_BASE:-18380}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-10}"
CONCURRENCY="${CONCURRENCY:-32}"
FEATURES="${FEATURES:-cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin}"

export MODEL_DIR HF_MODEL BIN OUT_ROOT REPEATS PORT_BASE NUM_PROMPTS WARMUP_REQUESTS CONCURRENCY FEATURES BUILD="${BUILD:-0}"

mkdir -p "$OUT_ROOT"
CONFIG="$OUT_ROOT/runner_config.json"

python3 - "$CONFIG" <<'PY'
import json
import os
import sys

config_path = sys.argv[1]
port_base = int(os.environ.get("PORT_BASE", "18380"))
config = {
    "name": "m3-fa-layout-varlen-ab",
    "out_root": os.environ["OUT_ROOT"],
    "model_dir": os.environ["MODEL_DIR"],
    "hf_model": os.environ["HF_MODEL"],
    "bin": os.environ["BIN"],
    "features": os.environ["FEATURES"],
    "build": os.environ.get("BUILD", "0") == "1",
    "preset": "m3_qwen3_30b_a3b_int4",
    "port_base": port_base,
    "repeats": int(os.environ["REPEATS"]),
    "num_prompts": int(os.environ["NUM_PROMPTS"]),
    "warmup_requests": int(os.environ["WARMUP_REQUESTS"]),
    "concurrency": int(os.environ["CONCURRENCY"]),
    "baseline_case": "default",
    "gates": {"paris": True, "multi_turn": False},
    "validation": {
        "change_type": "opt_in_experiment",
        "touched_areas": ["model_forward", "attention_prefill_mixed_path"],
        "performance_regression_required": True,
    },
    "cases": [
        {"name": "fa_layout", "port": port_base, "env": {"FERRUM_FA_LAYOUT_VARLEN": "1"}},
        {"name": "default", "port": port_base + 1, "env": {"FERRUM_FA_LAYOUT_VARLEN": "0"}},
    ],
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
