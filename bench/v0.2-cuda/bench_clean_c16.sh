#!/usr/bin/env bash
# Clean bench — NO profiling env vars, just measure throughput.
# Used to confirm whether the bg-loop fix actually moves the needle
# without per-iter sync/timer overhead distorting the comparison.
set -uo pipefail

WORKSPACE=/workspace
MODEL_DIR=$WORKSPACE/models/M2
PORT=8800
LOG_DIR=/tmp/clean_bench_run
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/ferrum_server.log"
BENCH_LOG="$LOG_DIR/vllm_bench.log"

pkill -f 'target/release/ferrum' 2>/dev/null || true
sleep 2
: > "$SERVER_LOG"   # truncate fresh
: > "$BENCH_LOG"

echo "[$(date +%H:%M:%S)] starting ferrum (NO PROF) ..."
CUDA_VISIBLE_DEVICES=0 \
FERRUM_KV_CAPACITY=2048 \
FERRUM_MAX_BATCH=32 \
  $WORKSPACE/ferrum-infer-rs/target/release/ferrum serve \
    --model "$MODEL_DIR" --port $PORT \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" -o /dev/null 2>&1; then
    echo "  ready at ${i}s"; break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[FATAL] server died"; tail -20 "$SERVER_LOG"; exit 1
  fi
  sleep 1
done

echo "[$(date +%H:%M:%S)] warmup ..."
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"M2","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' \
  > /dev/null
sleep 2

echo "[$(date +%H:%M:%S)] running 3 reps c=16 (clean) ..."
for rep in 1 2 3; do
  echo "--- rep $rep ---"
  timeout 240 vllm bench serve \
    --backend openai-chat \
    --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/chat/completions \
    --model "$MODEL_DIR" \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len 128 \
    --num-prompts 32 \
    --max-concurrency 16 \
    --request-rate inf \
    --temperature 0 \
    --top-p 1 \
    --ignore-eos \
    --result-dir "$LOG_DIR" \
    --result-filename "bench_c16_r${rep}.json" \
    --save-result 2>&1 | tee -a "$BENCH_LOG" | grep -E 'Output token|Mean TPOT|P99 TPOT|Successful'
done

kill $SERVER_PID 2>/dev/null
sleep 2
kill -9 $SERVER_PID 2>/dev/null || true

echo ""
echo "=== SUMMARY ALL REPS ==="
for rep in 1 2 3; do
  if [ -f "$LOG_DIR/bench_c16_r${rep}.json" ]; then
    python3 -c "
import json
d = json.load(open('$LOG_DIR/bench_c16_r${rep}.json'))
print(f'  r$rep: out={d.get(\"output_throughput\",0):.1f} tok/s  TPOT_p50={d.get(\"median_tpot_ms\",0):.1f}ms  TPOT_p99={d.get(\"p99_tpot_ms\",0):.1f}ms')
"
  fi
done
