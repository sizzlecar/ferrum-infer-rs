#!/bin/bash
# Apples c=32 ferrum with extra env flags. Usage:
#   test_envs.sh <label> [extra env exports]
# Example:
#   test_envs.sh chunked "FERRUM_CHUNKED_PREFILL=128"
#   test_envs.sh device_route "FERRUM_MOE_DEVICE_ROUTE=1"
#   test_envs.sh both "FERRUM_CHUNKED_PREFILL=128 FERRUM_MOE_DEVICE_ROUTE=1"

set -e
cd /workspace/ferrum-infer-rs
LABEL="${1:-baseline}"
EXTRA="${2:-}"

PORT=8801
RUN_R=$(date +%s)
LOG=bench/v0.2-cuda/results/ferrum__M3__c32__${LABEL}_r${RUN_R}.server.log

pkill -9 -f "ferrum.*serve" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3

CMD="CUDA_VISIBLE_DEVICES=0 \
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
$EXTRA \
  /workspace/ferrum-infer-rs/target/release/ferrum serve \
    --model /workspace/models/M3 --port $PORT \
    --gpu-memory-utilization 0.95"
echo "[$LABEL] env: $EXTRA"
eval "$CMD" > "$LOG" 2>&1 &
FPID=$!

for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  ! kill -0 "$FPID" 2>/dev/null && { echo "died"; tail -50 "$LOG"; exit 1; }
  sleep 1
done

curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' > /dev/null

source /workspace/vllm-venv/bin/activate
bash bench/v0.2-cuda/run_cell.sh ferrum M3 32 "${LABEL}_${RUN_R}" "$PORT" 2>&1 | tail -3

kill -INT "$FPID" 2>/dev/null; sleep 3
pkill -9 -f "ferrum.*serve" 2>/dev/null || true
