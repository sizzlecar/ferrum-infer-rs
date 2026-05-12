#!/bin/bash
# Shared-system-prompt bench: 128 chat requests with same system msg,
# unique user msg. Measures TTFT / throughput with FERRUM_PREFIX_CACHE
# OFF vs ON to quantify the block-level prefix cache win.

set -e
cd /workspace/ferrum-infer-rs

PORT=8801

run_one() {
    local label="$1"
    local prefix_cache="$2"  # "0" or "1"
    local log="bench/v0.2-cuda/results/shared_prefix_${label}.log"
    pkill -9 -f "ferrum.*serve" 2>/dev/null || true
    sleep 3

    echo "=== [$label] FERRUM_PREFIX_CACHE=$prefix_cache ==="
    CUDA_VISIBLE_DEVICES=0 \
    FERRUM_VLLM_MOE=1 \
    FERRUM_KV_CAPACITY=2048 \
    FERRUM_KV_MAX_BLOCKS=4096 \
    FERRUM_PAGED_MAX_SEQS=32 \
    FERRUM_METAL_PAGED_KV=1 \
    FERRUM_GREEDY_ARGMAX=1 \
    FERRUM_MOE_BUCKETED=1 \
    FERRUM_MARLIN_SKIP_WS_ZERO=1 \
    FERRUM_MOE_STREAMS=4 \
    FERRUM_MOE_BATCH_THRESHOLD=4 \
    FERRUM_MOE_GRAPH=1 \
    FERRUM_GRAPH_SKIP_UPLOAD=1 \
    FERRUM_PREFIX_CACHE=$prefix_cache \
      /workspace/ferrum-infer-rs/target/release/ferrum serve \
        --model /workspace/models/M3 --port "$PORT" \
        --gpu-memory-utilization 0.95 \
      > "$log" 2>&1 &
    local fpid=$!
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
        ! kill -0 "$fpid" 2>/dev/null && { echo "[$label] died"; tail -50 "$log"; exit 1; }
        sleep 1
    done
    curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' > /dev/null

    # First request to seed the cache (only relevant when prefix_cache=1)
    if [ "$prefix_cache" = "1" ]; then
        echo "[$label] seeding cache with one prompt..."
        python3 bench/v0.2-cuda/shared_prefix_bench.py \
            --base-url "http://127.0.0.1:$PORT" \
            --concurrency 1 --num-prompts 1 --max-tokens 4 >/dev/null
    fi

    echo "[$label] running 128 prompts c=32..."
    python3 bench/v0.2-cuda/shared_prefix_bench.py \
        --base-url "http://127.0.0.1:$PORT" \
        --concurrency 32 --num-prompts 128 --max-tokens 64

    kill -INT "$fpid" 2>/dev/null
    sleep 3
    pkill -9 -f "ferrum.*serve" 2>/dev/null || true
}

# Ensure aiohttp available
pip install -q aiohttp 2>/dev/null || python3 -m pip install -q --user aiohttp 2>/dev/null || true

run_one "off" "0"
echo
run_one "on" "1"
