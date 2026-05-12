#!/bin/bash
# Re-run ferrum's 2 cells WITHOUT seeding the cache, to match vllm's
# default (no warmup-of-cache-only) behavior. The bench client itself
# always does its own prewarm chat completion.
set -e
cd /workspace/ferrum-infer-rs

# Apples dataset (shared-prefix) must be in place
[ -f bench/v0.2-cuda/prompts.jsonl.apples_backup ] || cp bench/v0.2-cuda/prompts.jsonl bench/v0.2-cuda/prompts.jsonl.apples_backup
python3 bench/v0.2-cuda/gen_shared_prefix_jsonl.py bench/v0.2-cuda/shared_prefix_prompts.jsonl
cp bench/v0.2-cuda/shared_prefix_prompts.jsonl bench/v0.2-cuda/prompts.jsonl
trap 'cp bench/v0.2-cuda/prompts.jsonl.apples_backup bench/v0.2-cuda/prompts.jsonl; pkill -9 -f "ferrum.*serve" 2>/dev/null; pkill -9 -f "VLLM" 2>/dev/null; true' EXIT

start_ferrum() {
    local prefix_cache="$1"
    pkill -9 -f "ferrum.*serve" 2>/dev/null || true
    pkill -9 -f "VLLM" 2>/dev/null || true
    sleep 5
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
        ! kill -0 $SRV_PID 2>/dev/null && { echo "DIED"; tail -50 /tmp/srv.log; exit 1; }
        sleep 1
    done
    return 1
}

run_cell_noseed() {
    local cell_id="$1"
    # No application-level seeding — let the bench client's natural
    # prewarm-of-1 (the run_cell.sh prewarm) be the ONLY priming, same
    # as the vllm runs got.
    source /workspace/vllm-venv/bin/activate
    bash bench/v0.2-cuda/run_cell.sh ferrum M3 32 "${cell_id}" 8801 2>&1 | tail -3
}

echo "[1/2] ferrum FERRUM_PREFIX_CACHE=0  (NO SEED — cold start)"
start_ferrum 0
run_cell_noseed "shared_ferrum_off_coldstart"
kill -INT $SRV_PID 2>/dev/null; sleep 3

echo
echo "[2/2] ferrum FERRUM_PREFIX_CACHE=1  (NO SEED — cold start)"
start_ferrum 1
run_cell_noseed "shared_ferrum_on_coldstart"
kill -INT $SRV_PID 2>/dev/null; sleep 3

echo
echo "==========================================="
echo "Summary (4-way fair: no seeding, cold start)"
echo "==========================================="
for c in shared_ferrum_off_coldstart shared_ferrum_on_coldstart shared_vllm_off shared_vllm_on; do
    F=$(ls bench/v0.2-cuda/results/*__M3__c32__r${c}.json 2>/dev/null | head -1)
    if [ -n "$F" ] && [ -f "$F" ]; then
        python3 -c "
import json
d = json.load(open('$F'))
print('${c}'.ljust(30),
      f\"out={d.get('output_throughput',0):7.1f} tok/s\",
      f\"p50={d.get('mean_tpot_ms',0):6.2f}ms\",
      f\"p99={d.get('p99_tpot_ms',0):7.2f}ms\",
      f\"ttft={d.get('mean_ttft_ms',0):6.0f}ms\")
"
    fi
done
