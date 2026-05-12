#!/bin/bash
# vllm-only completion of the 4-way prefix bench. Run AFTER
# test_prefix_apples.sh got ferrum results.
set -e
cd /workspace/ferrum-infer-rs
MODEL=/workspace/models/M3

start_vllm() {
    local prefix_flag="$1"
    pkill -9 -f "ferrum.*serve" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 4
    source /workspace/vllm-venv/bin/activate
    local extra
    if [ -z "$prefix_flag" ]; then
        extra="--no-enable-prefix-caching"
    else
        extra=""
    fi
    vllm serve $MODEL --port 8801 \
        --max-num-seqs 64 --max-model-len 4096 \
        $extra --no-enable-log-requests \
        --quantization gptq_marlin \
      > /tmp/vllm_srv.log 2>&1 &
    SRV_PID=$!
    for i in $(seq 1 240); do
        curl -sf http://127.0.0.1:8801/v1/models >/dev/null 2>&1 && return 0
        ! kill -0 $SRV_PID 2>/dev/null && { echo "DIED"; tail -50 /tmp/vllm_srv.log; exit 1; }
        sleep 1
    done
    return 1
}

run_cell_vllm() {
    local cell_id="$1"
    source /workspace/vllm-venv/bin/activate
    bash bench/v0.2-cuda/run_cell.sh vllm M3 32 "${cell_id}" 8801 2>&1 | tail -3
}

echo "[3/4] vllm --no-enable-prefix-caching"
start_vllm ""
run_cell_vllm "shared_vllm_off"
kill -INT $SRV_PID 2>/dev/null; sleep 3

echo
echo "[4/4] vllm (default prefix-caching ON)"
start_vllm "enable"
run_cell_vllm "shared_vllm_on"
kill -INT $SRV_PID 2>/dev/null; sleep 3

echo
echo "=========================================="
echo "Summary"
echo "=========================================="
for c in shared_ferrum_off shared_ferrum_on shared_vllm_off shared_vllm_on; do
    F=$(ls bench/v0.2-cuda/results/*__M3__c32__r${c}.json 2>/dev/null | head -1)
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
    fi
done
# Restore original prompts.jsonl (apples)
BACKUP=bench/v0.2-cuda/prompts.jsonl.apples_backup
[ -f "$BACKUP" ] && cp "$BACKUP" bench/v0.2-cuda/prompts.jsonl
