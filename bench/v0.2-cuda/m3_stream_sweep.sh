#!/bin/bash
# M3 multi-stream MoE sweep: probe whether more streams help now that
# dispatches are sorted by m desc (Stage 7) AND host route is on GPU
# (Stage 8). Earlier doc claimed s=4 = s=8 — but with sort + GPU route
# the GPU has less stalling, so a deeper pool MIGHT now win.

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
BENCH_DIR="${WORKSPACE}/ferrum-infer-rs/bench/v0.2-cuda"
RESULTS_DIR="$BENCH_DIR/results_m3_streams_v2"
MODEL_DIR="${WORKSPACE}/models/M3"
mkdir -p "$RESULTS_DIR"

PORT=8800

run_one() {
  local n_streams="$1"
  echo "=== FERRUM_MOE_STREAMS=$n_streams ==="
  local server_log="$RESULTS_DIR/s${n_streams}__server.log"
  CUDA_VISIBLE_DEVICES=0 \
  FERRUM_KV_CAPACITY=2048 \
  FERRUM_KV_MAX_BLOCKS=2048 \
  FERRUM_PAGED_MAX_SEQS=32 \
  FERRUM_METAL_PAGED_KV=0 \
  FERRUM_MIXED_BATCH=0 \
  FERRUM_GREEDY_ARGMAX=1 \
  FERRUM_MOE_BUCKETED=1 \
  FERRUM_MARLIN_SKIP_WS_ZERO=1 \
  FERRUM_MOE_STREAMS=$n_streams \
    "$WORKSPACE/ferrum-infer-rs/target/release/ferrum" serve \
      --model "$MODEL_DIR" --port "$PORT" \
      --gpu-memory-utilization 0.95 \
      > "$server_log" 2>&1 &
  PID=$!

  for i in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
      tail -30 "$server_log"
      return 1
    fi
    sleep 1
  done

  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}' \
    >/dev/null

  vllm bench serve \
    --backend openai-chat \
    --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/chat/completions \
    --model "$MODEL_DIR" \
    --tokenizer "$MODEL_DIR" \
    --dataset-name random \
    --random-input-len 256 --random-output-len 128 \
    --num-prompts 128 \
    --max-concurrency 32 \
    --save-result --result-dir "$RESULTS_DIR" \
    --result-filename "s${n_streams}_c32.json" \
    --percentile-metrics "ttft,tpot,itl" 2>&1 | tee "$RESULTS_DIR/s${n_streams}_c32.log" >/dev/null

  kill -INT "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
  sleep 3
}

for s in 1 2 4 8; do
  run_one $s
done

echo ""
echo "=== headlines ==="
for s in 1 2 4 8; do
  python3 -c "
import json
d = json.load(open('$RESULTS_DIR/s${s}_c32.json'))
print(f's=${s}  output_throughput={d[\"output_throughput\"]:.1f} tok/s   mean_tpot={d[\"mean_tpot_ms\"]:.2f} ms')
"
done
