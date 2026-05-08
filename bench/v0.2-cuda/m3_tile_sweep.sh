#!/bin/bash
# Marlin tile-size sweep at c=32. After Stage 12.1 + 13a defaults ON,
# the (1,8,8) thread tile (128x128) is what prob_m=16 auto-picks. Try
# the other CALL_IF specs to see which actually wins for the MoE
# shape (gate_up n=1536, k=2048; down n=2048, k=768).
#
# FERRUM_MARLIN_TILE accepts: 64x256, 128x128, 64x128, 128x64.
# Only c=32, single rep — quick A/B (~2 min total).

set -uo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
BENCH_DIR="${WORKSPACE}/ferrum-infer-rs/bench/v0.2-cuda"
RESULTS_DIR="$BENCH_DIR/results_m3_tile_sweep"
MODEL_DIR="${WORKSPACE}/models/M3"
mkdir -p "$RESULTS_DIR"

run_one() {
  local tile="$1"
  local PORT=8800
  local SERVER_LOG="$RESULTS_DIR/server_${tile}.log"

  echo "── tile=$tile ────────────────────────────────────"
  pkill -9 -f "target/release/ferrum" 2>/dev/null
  sleep 3

  CUDA_VISIBLE_DEVICES=0 \
  FERRUM_KV_CAPACITY=2048 \
  FERRUM_KV_MAX_BLOCKS=4096 \
  FERRUM_PAGED_MAX_SEQS=32 \
  FERRUM_METAL_PAGED_KV=1 \
  FERRUM_MIXED_BATCH=0 \
  FERRUM_GREEDY_ARGMAX=1 \
  FERRUM_MOE_BUCKETED=1 \
  FERRUM_MARLIN_SKIP_WS_ZERO=1 \
  FERRUM_MOE_STREAMS=4 \
  FERRUM_MARLIN_TILE="$tile" \
    "$WORKSPACE/ferrum-infer-rs/target/release/ferrum" serve \
      --model "$MODEL_DIR" --port "$PORT" \
      --gpu-memory-utilization 0.95 > "$SERVER_LOG" 2>&1 &
  local PID=$!

  for i in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$PID" 2>/dev/null; then echo "died (tile=$tile)"; tail -50 "$SERVER_LOG"; return 1; fi
    sleep 1
  done

  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}' >/dev/null

  vllm bench serve \
    --backend openai-chat --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/chat/completions --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 128 --max-concurrency 32 \
    --save-result --result-dir "$RESULTS_DIR" \
    --result-filename "tile_${tile}_c32.json" \
    --percentile-metrics ttft,tpot,itl 2>&1 | tail -5

  kill -INT "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
}

for tile in 128x128 64x256 64x128 128x64; do
  run_one "$tile"
done

echo ""
echo "=== headlines (c=32) ==="
for tile in 128x128 64x256 64x128 128x64; do
  python3 -c "
import json
d = json.load(open('$RESULTS_DIR/tile_${tile}_c32.json'))
print(f'tile=${tile}  out={d[\"output_throughput\"]:.1f} tok/s  tpot={d[\"mean_tpot_ms\"]:.2f} ms')
" 2>/dev/null || echo "tile=$tile: missing data"
done
