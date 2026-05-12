#!/bin/bash
# Apples-to-apples bench comparing prefix cache implementations on
# a SHARED-PREFIX workload. Uses the SAME vllm bench serve client
# the canonical apples bench uses — only the dataset changes.
#
# 4 runs:
#   ferrum_off : ferrum, FERRUM_PREFIX_CACHE=0
#   ferrum_on  : ferrum, FERRUM_PREFIX_CACHE=1
#   vllm_off   : vllm,   --no-enable-prefix-caching
#   vllm_on    : vllm,   --enable-prefix-caching
#
# All use the SAME prompts.jsonl (rewritten to shared-prefix).
# Original is backed up + restored at the end.

set -e
cd /workspace/ferrum-infer-rs
PROMPTS=bench/v0.2-cuda/prompts.jsonl
SHARED=bench/v0.2-cuda/shared_prefix_prompts.jsonl
BACKUP=bench/v0.2-cuda/prompts.jsonl.apples_backup

# 1. Generate the shared-prefix dataset
python3 bench/v0.2-cuda/gen_shared_prefix_jsonl.py "$SHARED"

# 2. Swap dataset (apples client reads this exact path)
[ -f "$BACKUP" ] || cp "$PROMPTS" "$BACKUP"
cp "$SHARED" "$PROMPTS"
trap 'cp "$BACKUP" "$PROMPTS"; pkill -9 -f "ferrum.*serve" 2>/dev/null; pkill -9 -f "vllm serve" 2>/dev/null; true' EXIT

start_ferrum() {
    local prefix_cache="$1"
    pkill -9 -f "ferrum.*serve" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 3
    CUDA_VISIBLE_DEVICES=0 \
    FERRUM_VLLM_MOE=1 FERRUM_KV_CAPACITY=2048 FERRUM_KV_MAX_BLOCKS=4096 \
    FERRUM_PAGED_MAX_SEQS=32 FERRUM_METAL_PAGED_KV=1 FERRUM_GREEDY_ARGMAX=1 \
    FERRUM_MOE_BUCKETED=1 FERRUM_MARLIN_SKIP_WS_ZERO=1 FERRUM_MOE_STREAMS=4 \
    FERRUM_MOE_BATCH_THRESHOLD=4 FERRUM_MOE_GRAPH=1 FERRUM_GRAPH_SKIP_UPLOAD=1 \
    FERRUM_PREFIX_CACHE=$prefix_cache \
      /workspace/ferrum-infer-rs/target/release/ferrum serve \
        --model /workspace/models/M3 --port 8801 \
        --gpu-memory-utilization 0.95 \
      > /tmp/srv.log 2>&1 &
    SRV_PID=$!
    for i in $(seq 1 240); do
        curl -sf http://127.0.0.1:8801/v1/models >/dev/null 2>&1 && return 0
        ! kill -0 $SRV_PID 2>/dev/null && { echo "FERRUM DIED"; tail -50 /tmp/srv.log; exit 1; }
        sleep 1
    done
    return 1
}

start_vllm() {
    local prefix_flag="$1"  # "" for default-off, "--enable-prefix-caching" for on
    pkill -9 -f "ferrum.*serve" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 3
    source /workspace/vllm-venv/bin/activate
    local extra
    if [ -z "$prefix_flag" ]; then
        extra="--no-enable-prefix-caching"
    else
        extra=""  # vLLM default is enable-prefix-caching on, so don't pass --no
    fi
    vllm serve /workspace/models/M3 --port 8801 \
        --max-num-seqs 64 --max-model-len 4096 \
        $extra \
        --no-enable-log-requests \
        --quantization gptq_marlin \
      > /tmp/srv.log 2>&1 &
    SRV_PID=$!
    for i in $(seq 1 240); do
        curl -sf http://127.0.0.1:8801/v1/models >/dev/null 2>&1 && return 0
        ! kill -0 $SRV_PID 2>/dev/null && { echo "VLLM DIED"; tail -50 /tmp/srv.log; exit 1; }
        sleep 1
    done
    return 1
}

run_cell() {
    local cell_id="$1"  # e.g. ferrum_off
    local engine_for_script="$2"  # "ferrum" or "vllm" (just affects results filename)
    # Prewarm with one short request
    curl -sf -m 60 -X POST http://127.0.0.1:8801/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' > /dev/null
    # For prefix-cache hits to materialize the cache must be seeded by
    # at least one full request first. Spend one slot on that.
    curl -sf -m 120 -X POST http://127.0.0.1:8801/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "$(python3 -c '
import json, sys
prefix = open("'"$PROMPTS"'").read().splitlines()[0]
p = json.loads(prefix)["prompt"]
print(json.dumps({"model":"x","messages":[{"role":"user","content":p}],"max_tokens":4,"temperature":0}))
')" > /dev/null
    source /workspace/vllm-venv/bin/activate
    bash bench/v0.2-cuda/run_cell.sh "$engine_for_script" M3 32 "${cell_id}" 8801 2>&1 | tail -3
}

mkdir -p bench/v0.2-cuda/results

echo "=========================================="
echo "[1/4] ferrum FERRUM_PREFIX_CACHE=0"
echo "=========================================="
start_ferrum 0
run_cell "shared_ferrum_off" "ferrum"
kill -INT $SRV_PID 2>/dev/null; sleep 3

echo
echo "=========================================="
echo "[2/4] ferrum FERRUM_PREFIX_CACHE=1"
echo "=========================================="
start_ferrum 1
run_cell "shared_ferrum_on" "ferrum"
kill -INT $SRV_PID 2>/dev/null; sleep 3

echo
echo "=========================================="
echo "[3/4] vllm --no-enable-prefix-caching"
echo "=========================================="
start_vllm ""
run_cell "shared_vllm_off" "vllm"
kill -INT $SRV_PID 2>/dev/null; sleep 3

echo
echo "=========================================="
echo "[4/4] vllm (default prefix-caching ON)"
echo "=========================================="
start_vllm "enable"
run_cell "shared_vllm_on" "vllm"
kill -INT $SRV_PID 2>/dev/null; sleep 3

echo
echo "=========================================="
echo "Summary"
echo "=========================================="
for c in shared_ferrum_off shared_ferrum_on shared_vllm_off shared_vllm_on; do
    F=bench/v0.2-cuda/results/*__M3__c32__${c}.json
    F=$(ls $F 2>/dev/null | head -1)
    if [ -n "$F" ] && [ -f "$F" ]; then
        python3 -c "
import json
d = json.load(open('$F'))
print('${c}'.ljust(22),
      f\"out={d.get('output_throughput',0):7.1f} tok/s\",
      f\"p50={d.get('mean_tpot_ms',0):6.2f}ms\",
      f\"p99={d.get('p99_tpot_ms',0):7.2f}ms\",
      f\"ttft={d.get('mean_ttft_ms',0):6.0f}ms\")
"
    else
        echo "${c}: no result file"
    fi
done
