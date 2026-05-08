#!/bin/bash
# Bench MIXED prefill+decode (FERRUM_MIXED_BATCH=1) vs the legacy serial
# path. Tests c=1, 4, 16, 32 against M2 (Llama-3.1-8B-GPTQ-Int4).
#
# Usage: run on the vast.ai pod from /workspace/ferrum-infer-rs.
# Restarts ferrum twice (once per mode) and writes per-cell JSONs into
# results/ferrum__M2__c{C}__r{R}_{MODE}128.json.
set -euo pipefail

PORT=8800
MODEL_DIR=/workspace/models/M2
RESULTS=/workspace/ferrum-infer-rs/bench/v0.2-cuda/results

start_ferrum() {
    local mode="$1"  # "MIXED" or "SERIAL"
    local mb="0"
    if [[ "$mode" == "MIXED" ]]; then mb="1"; fi
    pkill -9 ferrum 2>/dev/null || true
    sleep 3
    local log="/tmp/ferrum_${mode}.log"
    nohup env CUDA_VISIBLE_DEVICES=0 \
        FERRUM_KV_CAPACITY=2048 \
        FERRUM_KV_MAX_BLOCKS=2048 \
        FERRUM_MAX_BATCH=4 \
        FERRUM_GREEDY_ARGMAX=1 \
        FERRUM_UNIFIED_GRAPH=1 \
        FERRUM_SPLIT_K_ATTN=0 \
        FERRUM_MIXED_BATCH="$mb" \
        FERRUM_UNIFIED_TOKEN_BUDGET=512 \
        /workspace/ferrum-infer-rs/target/release/ferrum serve \
            --model "$MODEL_DIR" --port "$PORT" \
        > "$log" 2>&1 < /dev/null &
    for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
        if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            echo "  ferrum READY ($mode) after $((i*4))s" >&2
            return 0
        fi
        sleep 4
    done
    echo "  ERROR: ferrum failed to start ($mode)" >&2
    tail -20 "$log" >&2
    return 1
}

run_cell() {
    local c="$1"
    local mode="$2"
    local out="$RESULTS/ferrum__M2__c${c}__r1_${mode}128.json"
    timeout 240 vllm bench serve \
        --backend openai-chat \
        --base-url "http://127.0.0.1:$PORT" \
        --endpoint /v1/chat/completions \
        --model M2 \
        --tokenizer "$MODEL_DIR" \
        --dataset-name custom \
        --dataset-path prompts.jsonl \
        --num-prompts 128 \
        --max-concurrency "$c" \
        --temperature 0 \
        --result-dir "$RESULTS" \
        --result-filename "ferrum__M2__c${c}__r1_${mode}128.json" \
        --save-result \
        --disable-tqdm 2>&1 | grep -E 'Output token throughput|Mean TPOT|Successful'
    echo
}

cd /workspace/ferrum-infer-rs/bench/v0.2-cuda
for mode in MIXED SERIAL; do
    echo "===== $mode ====="
    start_ferrum "$mode"
    for c in 1 4 16 32; do
        echo "--- c=$c $mode ---"
        run_cell "$c" "$mode" || echo "  cell c=$c $mode failed"
    done
done

pkill -9 ferrum 2>/dev/null || true
echo
echo "===== summary ====="
for mode in MIXED SERIAL; do
    echo "[$mode]"
    for c in 1 4 16 32; do
        f="$RESULTS/ferrum__M2__c${c}__r1_${mode}128.json"
        if [[ -f "$f" ]]; then
            python3 -c "
import json
d = json.load(open('$f'))
tp = d.get('output_throughput', 0)
tpot = d.get('mean_tpot_ms', 0)
ok = d.get('completed', 0)
print(f'  c=${c}: {tp:6.1f} tok/s | TPOT {tpot:5.2f} ms | ok={ok}/128')
"
        fi
    done
done
