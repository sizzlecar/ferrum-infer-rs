#!/usr/bin/env bash
#
# Thin wrapper around scripts/m3_ab_runner.py for FA2 direct/source A/B.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
BIN="${BIN:-./target/release/ferrum}"
OUT_ROOT="${OUT_ROOT:-/workspace/m3-fa2-direct-ffi-ab-$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-1}"
PORT_BASE="${PORT_BASE:-18480}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-10}"
CONCURRENCY="${CONCURRENCY:-32}"
VALIDATION_CHANGE_TYPE="${VALIDATION_CHANGE_TYPE:-opt_in_experiment}"
FA2_SHIM="${FA2_SHIM:-/workspace/libferrum_fa2_shim.so}"
FA2_SOURCE="${FA2_SOURCE:-0}"
TORCH_LIB="${TORCH_LIB:-/workspace/vllm-venv/lib/python3.12/site-packages/torch/lib}"
FA2_DIR="${FA2_DIR:-/workspace/vllm-venv/lib/python3.12/site-packages/vllm/vllm_flash_attn}"
if [[ -z "${FA2_EXTRA_LD_LIBRARY_PATH+x}" ]]; then
    FA2_EXTRA_LD_LIBRARY_PATH="${TORCH_LIB}:${FA2_DIR}"
fi

FEATURES="${FEATURES:-cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin}"
if [[ "$FA2_SOURCE" == "1" ]]; then
    FEATURES="${FEATURES},fa2-source"
elif [[ ! -r "$FA2_SHIM" ]]; then
    echo "FA2_SHIM is not readable: $FA2_SHIM" >&2
    echo "build it with: bash scripts/microbenches/build_fa2_ferrum_shim.sh" >&2
    exit 1
fi

export MODEL_DIR HF_MODEL BIN OUT_ROOT REPEATS PORT_BASE NUM_PROMPTS WARMUP_REQUESTS CONCURRENCY
export VALIDATION_CHANGE_TYPE FA2_SHIM FA2_SOURCE TORCH_LIB FA2_DIR FA2_EXTRA_LD_LIBRARY_PATH FEATURES BUILD="${BUILD:-0}"

mkdir -p "$OUT_ROOT"
CONFIG="$OUT_ROOT/runner_config.json"

python3 - "$CONFIG" <<'PY'
import json
import os
import sys

config_path = sys.argv[1]
port_base = int(os.environ.get("PORT_BASE", "18480"))
extra_ld = os.environ.get("FA2_EXTRA_LD_LIBRARY_PATH", "")
ld_library_path = f"{extra_ld + ':' if extra_ld else ''}/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
fa2_source = os.environ.get("FA2_SOURCE") == "1"
candidate_name = "fa2_source" if fa2_source else "fa2_direct"
candidate_env = {
    "FERRUM_FA_LAYOUT_VARLEN": "1",
    "FERRUM_FA2_SOURCE": "1",
} if fa2_source else {
    "FERRUM_FA_LAYOUT_VARLEN": "1",
    "FERRUM_FA2_DIRECT_FFI_SHIM": os.environ["FA2_SHIM"],
}

config = {
    "name": "m3-fa2-direct-ffi-ab",
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
    "baseline_case": "fa_layout",
    "gates": {"paris": True, "multi_turn": True, "multi_turn_3round": True},
    "validation": {
        "change_type": os.environ["VALIDATION_CHANGE_TYPE"],
        "touched_areas": ["model_forward", "attention_prefill_mixed_path", "fa2_runtime_path"],
        "performance_regression_required": True,
    },
    "base_env": {
        "LD_LIBRARY_PATH": ld_library_path
    },
    "cases": [
        {"name": candidate_name, "port": port_base, "env": candidate_env},
        {
            "name": "fa_layout",
            "port": port_base + 1,
            "env": {"FERRUM_FA_LAYOUT_VARLEN": "1", "FERRUM_FA2_DIRECT_FFI": "0"},
        },
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
