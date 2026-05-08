#!/bin/bash
# Test FERRUM_MIXED_BATCH=1 on Qwen3-MoE: does it produce correct
# output AND give a perf gain? Earlier doc claimed garbled output;
# now retest after Stage 8 changes (GPU route + alloc-free etc).

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
BENCH_DIR="${WORKSPACE}/ferrum-infer-rs/bench/v0.2-cuda"
RESULTS_DIR="$BENCH_DIR/results_m3_mixed"
MODEL_DIR="${WORKSPACE}/models/M3"
mkdir -p "$RESULTS_DIR"

PORT=8800

run_one() {
  local label="$1"
  local mixed="$2"
  local server_log="$RESULTS_DIR/${label}__server.log"
  echo "=== $label (FERRUM_MIXED_BATCH=$mixed) ==="

  CUDA_VISIBLE_DEVICES=0 \
  FERRUM_KV_CAPACITY=2048 \
  FERRUM_KV_MAX_BLOCKS=2048 \
  FERRUM_PAGED_MAX_SEQS=32 \
  FERRUM_METAL_PAGED_KV=0 \
  FERRUM_MIXED_BATCH=$mixed \
  FERRUM_GREEDY_ARGMAX=1 \
  FERRUM_MOE_BUCKETED=1 \
  FERRUM_MARLIN_SKIP_WS_ZERO=1 \
  FERRUM_MOE_STREAMS=4 \
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

  # ── Sanity check: simple "What is 2+2?" should answer "4". ─────────
  echo "--- sanity check chat completion ---"
  local resp
  resp=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":16,"stream":false}')
  echo "  response: $resp" | head -c 400
  echo ""

  # ── Bench at c=32 ─────────────────────────────────────────────────
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
    --result-filename "${label}_c32.json" \
    --percentile-metrics "ttft,tpot,itl" 2>&1 | tee "$RESULTS_DIR/${label}_c32.log" >/dev/null

  kill -INT "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
  sleep 3
}

run_one mixed_off 0
run_one mixed_on  1

echo ""
echo "=== headlines ==="
for label in mixed_off mixed_on; do
  python3 -c "
import json
d = json.load(open('$RESULTS_DIR/${label}_c32.json'))
print('${label}  output_throughput=%.1f tok/s   mean_tpot=%.2f ms' % (d['output_throughput'], d['mean_tpot_ms']))
"
done
