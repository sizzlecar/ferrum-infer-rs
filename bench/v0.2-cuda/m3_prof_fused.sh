#!/bin/bash
# c=32 only, FERRUM_DECODE_OP_PROFILE on. Surfaces per-op timings to
# the server log. Used to find the bottleneck driving Stage 12's small
# 5% gain.

set -uo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
BENCH_DIR="${WORKSPACE}/ferrum-infer-rs/bench/v0.2-cuda"
RESULTS_DIR="$BENCH_DIR/results_m3_prof_fused"
MODEL_DIR="${WORKSPACE}/models/M3"
mkdir -p "$RESULTS_DIR"
PORT=8800
SERVER_LOG="$RESULTS_DIR/server_prof.log"

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
FERRUM_MOE_FUSED=1 \
FERRUM_DECODE_OP_PROFILE=1 \
  "$WORKSPACE/ferrum-infer-rs/target/release/ferrum" serve \
    --model "$MODEL_DIR" --port "$PORT" \
    --gpu-memory-utilization 0.95 > "$SERVER_LOG" 2>&1 &
PID=$!
for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "ready in ${i}s"; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "died"; tail -50 "$SERVER_LOG"; exit 1; fi
  sleep 1
done
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}' >/dev/null
vllm bench serve --backend openai-chat --base-url "http://127.0.0.1:$PORT" \
  --endpoint /v1/chat/completions --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
  --dataset-name random --random-input-len 256 --random-output-len 128 \
  --num-prompts 128 --max-concurrency 32 --percentile-metrics ttft,tpot,itl 2>&1 | tail -25
kill -INT "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
echo === DECODE PROFILE TAIL ===
grep -E "\[decode-prof\]|\[bucket-prof\]|\[moe-prof\]|\[attn-prof\]|\[stage-prof\]" "$SERVER_LOG" | tail -80
