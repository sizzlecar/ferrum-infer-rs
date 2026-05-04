#!/usr/bin/env bash
# Test 4 graph configs to find which combo is fastest.
set -uo pipefail

WORKSPACE=/workspace
MODEL=$WORKSPACE/models/M2
PORT=8800
LOG_DIR=/tmp/graph_ab
mkdir -p "$LOG_DIR"

run_cfg() {
  local name="$1"
  local skip_upload="$2"
  local skip_sync="$3"

  echo "=== $name (SKIP_UPLOAD=$skip_upload SKIP_SYNC=$skip_sync) ==="
  pid=$(pgrep -f "release/ferr" | head -1)
  if [ -n "$pid" ]; then kill $pid; sleep 2; fi

  CUDA_VISIBLE_DEVICES=0 \
  FERRUM_KV_CAPACITY=2048 \
  FERRUM_MAX_BATCH=32 \
  FERRUM_BATCHED_GRAPH=1 \
  FERRUM_GRAPH_PROF=1 \
  FERRUM_GRAPH_SKIP_UPLOAD=$skip_upload \
  FERRUM_GRAPH_SKIP_SYNC=$skip_sync \
    $WORKSPACE/ferrum-infer-rs/target/release/ferrum serve \
      --model "$MODEL" --port $PORT \
      > "$LOG_DIR/server_$name.log" 2>&1 &
  spid=$!

  for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
    if ! kill -0 $spid 2>/dev/null; then echo "  SERVER_DIED"; cat "$LOG_DIR/server_$name.log" | tail -10; return; fi
    sleep 1
  done

  # warmup
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"M2","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' > /dev/null
  sleep 2

  timeout 60 vllm bench serve \
    --backend openai-chat \
    --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len 128 \
    --num-prompts 32 \
    --max-concurrency 16 \
    --request-rate inf \
    --temperature 0 \
    --top-p 1 \
    --ignore-eos 2>&1 | grep -E "Output token|TPOT_p50|Mean TPOT|Successful" | head -5

  echo "  graph-prof samples (last 3):"
  grep "graph-prof" "$LOG_DIR/server_$name.log" | tail -3

  kill $spid 2>/dev/null
  sleep 2
}

run_cfg "default"     "0" "0"
run_cfg "no_sync"     "0" "1"
run_cfg "no_upload"   "1" "0"
run_cfg "no_both"     "1" "1"
