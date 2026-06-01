#!/usr/bin/env bash
#
# Release GGUF-vs-GGUF benchmark packet for the 2026-06-01 release.
#
# Runs Ferrum aliases against vLLM's experimental GGUF loader using the same
# tokenizer-aware random 256/128 bench-serve workload.
#
# Usage on a GPU host from repo root:
#
#   source /workspace/vllm-venv/bin/activate
#   cargo build --release -p ferrum-cli \
#     --features cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source
#   OUT_ROOT=/workspace/release-bench-20260601-gguf-8b \
#     bash scripts/release_gguf_8b_vs_vllm.sh

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/workspace/release-bench-20260601-gguf-8b-$(date +%Y%m%d_%H%M%S)}"
CONCURRENCIES="${CONCURRENCIES:-1,4,16,32}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
WARMUP="${WARMUP:-10}"
N_REPEATS="${N_REPEATS:-3}"
FERRUM_BIN="${FERRUM_BIN:-./target/release/ferrum}"
FERRUM_PORT_BASE="${FERRUM_PORT_BASE:-18800}"
VLLM_PORT_BASE="${VLLM_PORT_BASE:-18900}"

mkdir -p "$OUT_ROOT"

{
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo=$(pwd)"
    echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "concurrencies=$CONCURRENCIES"
    echo "num_prompts=$NUM_PROMPTS"
    echo "warmup=$WARMUP"
    echo "n_repeats=$N_REPEATS"
    echo "ferrum_bin=$FERRUM_BIN"
    echo "vllm=$(command -v vllm || true)"
    vllm --version 2>/dev/null || true
} >"$OUT_ROOT/metadata.txt"

run_model() {
    local name="$1"
    local ferrum_alias="$2"
    local vllm_model="$3"
    local tokenizer_model="$4"
    local ferrum_port="$5"
    local vllm_port="$6"
    local out_dir="$OUT_ROOT/$name"

    mkdir -p "$out_dir"
    echo "=== $name ==="

    "$FERRUM_BIN" pull "$ferrum_alias" >"$out_dir/ferrum_pull.log" 2>&1 || {
        cat "$out_dir/ferrum_pull.log" >&2
        return 1
    }

    if command -v hf >/dev/null 2>&1; then
        hf download "$tokenizer_model" \
            --include tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.json \
            >"$out_dir/tokenizer_pull.log" 2>&1 || true
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$tokenizer_model" \
            --include tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.json \
            >"$out_dir/tokenizer_pull.log" 2>&1 || true
    fi

    FERRUM_BIN="$FERRUM_BIN" \
    VLLM_MODEL_OVERRIDE="$vllm_model" \
    BENCH_MODEL="$vllm_model" \
    TOKENIZER_MODEL="$tokenizer_model" \
    VLLM_TOKENIZER_MODEL="$tokenizer_model" \
    VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-} --hf-config-path $tokenizer_model" \
    VLLM_PREFIX_CACHE="${VLLM_PREFIX_CACHE:-false}" \
    VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-true}" \
    VLLM_GPU_MEM_FRAC="${VLLM_GPU_MEM_FRAC:-0.85}" \
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}" \
    VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}" \
    bash scripts/bench_vs_vllm.sh "$ferrum_alias" "$CONCURRENCIES" \
        --num-prompts "$NUM_PROMPTS" \
        --warmup "$WARMUP" \
        --n-repeats "$N_REPEATS" \
        --ferrum-port "$ferrum_port" \
        --vllm-port "$vllm_port" \
        --out-dir "$out_dir"
}

run_model \
    qwen3-8b-q4_k_m \
    qwen3:8b-q4_k_m \
    Qwen/Qwen3-8B-GGUF:Q4_K_M \
    Qwen/Qwen3-8B \
    "$FERRUM_PORT_BASE" \
    "$VLLM_PORT_BASE"

run_model \
    llama31-8b-q4_k_m \
    llama3.1:8b-q4_k_m \
    bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M \
    NousResearch/Meta-Llama-3.1-8B-Instruct \
    "$((FERRUM_PORT_BASE + 10))" \
    "$((VLLM_PORT_BASE + 10))"

python3 - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
print("GGUF_RELEASE_SUMMARY_BEGIN")
for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    ferrum = model_dir / "ferrum_report.json"
    vllm = model_dir / "vllm_report.json"
    if not ferrum.exists() or not vllm.exists():
        print(f"{model_dir.name}: missing reports")
        continue
    f = json.load(open(ferrum))
    v = json.load(open(vllm))
    f_cells = f.get("cells") or [f]
    v_cells = v.get("cells") or [v]
    print(model_dir.name)
    for fc, vc in zip(f_cells, v_cells):
        c = fc.get("scenario", {}).get("concurrency") or fc.get("concurrency")
        ft = (fc.get("output_throughput_tps") or {}).get("mean", fc.get("output_throughput"))
        vt = (vc.get("output_throughput_tps") or {}).get("mean", vc.get("output_throughput"))
        ratio = None if not ft or not vt else ft / vt
        print(f"  c={c}: ferrum={ft} vllm={vt} ratio={ratio}")
print("GGUF_RELEASE_SUMMARY_END")
PY
