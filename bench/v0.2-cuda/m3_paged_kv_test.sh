#!/bin/bash
# Test FERRUM_METAL_PAGED_KV=1 on Qwen3-MoE — the M3 doc said paged KV
# was disabled for CUDA, but B::supports_paged_kv() returns true on
# CUDA. Earlier MIXED_BATCH note ("paged-KV required for the fix") is
# the strongest signal that this is the actual unlock.

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
BENCH_DIR="${WORKSPACE}/ferrum-infer-rs/bench/v0.2-cuda"
RESULTS_DIR="$BENCH_DIR/results_m3_paged"
MODEL_DIR="${WORKSPACE}/models/M3"
mkdir -p "$RESULTS_DIR"

PORT=8800

run_one() {
  local label="$1"
  local paged="$2"
  local server_log="$RESULTS_DIR/${label}__server.log"
  echo "=== $label (FERRUM_METAL_PAGED_KV=$paged) ==="

  CUDA_VISIBLE_DEVICES=0 \
  FERRUM_KV_CAPACITY=2048 \
  FERRUM_KV_MAX_BLOCKS=2048 \
  FERRUM_PAGED_MAX_SEQS=32 \
  FERRUM_METAL_PAGED_KV=$paged \
  FERRUM_MIXED_BATCH=0 \
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
    -d '{"model":"x","messages":[{"role":"user","content":"What is 2+2? Just reply with the number."}],"max_tokens":24,"stream":false}')
  echo "  response: $resp" | head -c 600
  echo ""

  # ── Bench at c=16 and c=32 ────────────────────────────────────────
  for c in 16 32; do
    vllm bench serve \
      --backend openai-chat \
      --base-url "http://127.0.0.1:$PORT" \
      --endpoint /v1/chat/completions \
      --model "$MODEL_DIR" \
      --tokenizer "$MODEL_DIR" \
      --dataset-name random \
      --random-input-len 256 --random-output-len 128 \
      --num-prompts $((c * 4)) \
      --max-concurrency $c \
      --save-result --result-dir "$RESULTS_DIR" \
      --result-filename "${label}_c${c}.json" \
      --percentile-metrics "ttft,tpot,itl" 2>&1 | tee "$RESULTS_DIR/${label}_c${c}.log" >/dev/null
  done

  kill -INT "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
  sleep 3
}

run_one paged_off 0
run_one paged_on  1

echo ""
echo "=== headlines ==="
for label in paged_off paged_on; do
  for c in 16 32; do
    python3 -c "
import json
d = json.load(open('$RESULTS_DIR/${label}_c${c}.json'))
print('${label} c=${c}  output_throughput=%.1f tok/s   mean_tpot=%.2f ms' % (d['output_throughput'], d['mean_tpot_ms']))
"
  done
done
