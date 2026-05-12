#!/usr/bin/env bash
# Profile ferrum at c=32 with FERRUM_DECODE_OP_PROFILE=1.
# Prints per-stage timing (norm/qkv/attn/moe/etc) to identify hotspot.
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cd /workspace/ferrum-infer-rs
source /workspace/vllm-venv/bin/activate

pkill -9 -f "ferrum.*serve" 2>/dev/null || true
sleep 3
PORT=8801
LOG=/tmp/ferrum_prof.log

CUDA_VISIBLE_DEVICES=0 \
FERRUM_VLLM_MOE=1 \
FERRUM_KV_CAPACITY=2048 \
FERRUM_KV_MAX_BLOCKS=4096 \
FERRUM_PAGED_MAX_SEQS=32 \
FERRUM_METAL_PAGED_KV=1 \
FERRUM_MIXED_BATCH=0 \
FERRUM_GREEDY_ARGMAX=1 \
FERRUM_MOE_BUCKETED=1 \
FERRUM_MARLIN_SKIP_WS_ZERO=1 \
FERRUM_MOE_STREAMS=4 \
FERRUM_DECODE_OP_PROFILE=1 \
  /workspace/ferrum-infer-rs/target/release/ferrum serve \
    --model /workspace/models/M3 --port "$PORT" \
    --gpu-memory-utilization 0.95 \
  > "$LOG" 2>&1 &
PID=$!
for i in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  ! kill -0 "$PID" 2>/dev/null && { echo "ferrum died"; tail -50 "$LOG"; exit 1; }
  sleep 1
done
echo "ready in ${i}s"
curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' >/dev/null

echo "=== running c=32 with PROFILE on ==="
bash bench/v0.2-cuda/run_cell.sh ferrum M3 32 99 "$PORT" 2>&1 | tail -5

echo "=== killing server ==="
kill -INT "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true

echo
echo "=== per-stage timing (last 60 lines of log) ==="
grep -E "NORM|MATMUL|ATTN|MOE|prefill|forward.*ms" "$LOG" | tail -40
echo
echo "=== full log: $LOG ==="
