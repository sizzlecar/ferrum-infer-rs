#!/bin/bash
# M3 fused MoE Marlin bench (Stage 12). Adds FERRUM_MOE_FUSED=1 to
# m3_clean.sh — same model, same workload, ONE-launch-per-bucket
# Marlin path instead of multi-stream per-expert. Direct A/B vs Stage
# 9 baseline (267.7 tok/s c=32 on 2026-05-08).

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
BENCH_DIR="${WORKSPACE}/ferrum-infer-rs/bench/v0.2-cuda"
RESULTS_DIR="$BENCH_DIR/results_m3_fused"
MODEL_DIR="${WORKSPACE}/models/M3"
mkdir -p "$RESULTS_DIR"

PORT=8800

echo "=== M3 fused MoE Marlin bench (Stage 12) ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
echo ""

SERVER_LOG="$RESULTS_DIR/ferrum_fused__server.log"

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
  "$WORKSPACE/ferrum-infer-rs/target/release/ferrum" serve \
    --model "$MODEL_DIR" --port "$PORT" \
    --gpu-memory-utilization 0.95 \
    > "$SERVER_LOG" 2>&1 &
PID=$!

for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ready in ${i}s"
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "process died"
    tail -50 "$SERVER_LOG"
    exit 1
  fi
  sleep 1
done

curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}' \
  >/dev/null

for c in 1 8 16 32; do
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
    --result-filename "ferrum_fused_c${c}.json" \
    --percentile-metrics "ttft,tpot,itl" 2>&1 | tee "$RESULTS_DIR/ferrum_fused_c${c}.log"
done

kill -INT "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

echo ""
echo "=== headlines (FERRUM_MOE_FUSED=1) ==="
for c in 1 8 16 32; do
  python3 -c "
import json
d = json.load(open('$RESULTS_DIR/ferrum_fused_c${c}.json'))
print(f'c=${c}  output_throughput={d[\"output_throughput\"]:.1f} tok/s   mean_tpot={d[\"mean_tpot_ms\"]:.2f} ms   mean_ttft={d[\"mean_ttft_ms\"]:.0f} ms')
"
done
