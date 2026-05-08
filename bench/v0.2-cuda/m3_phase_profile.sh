#!/bin/bash
# M3 (Qwen3-30B-A3B-INT4) phase-level profile run.
# Runs ferrum at c=32 with FERRUM_DECODE_OP_PROFILE=1 so the new
# [bucket-prof] line in qwen3_moe.rs prints per-phase microsecond
# breakdown across all 48 layers per decode step.
#
# Usage:
#   bash bench/v0.2-cuda/m3_phase_profile.sh
#
# Output:
#   results_m3_phase_prof/ferrum_phase_prof_c32.json   — vllm bench result
#   results_m3_phase_prof/ferrum_phase_prof__server.log — full server log
#                                                         (grep [bucket-prof])

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
BENCH_DIR="${WORKSPACE}/ferrum-infer-rs/bench/v0.2-cuda"
RESULTS_DIR="$BENCH_DIR/results_m3_phase_prof"
MODEL_DIR="${WORKSPACE}/models/M3"
mkdir -p "$RESULTS_DIR"

PORT=8800

echo "=== M3 phase profile (c=32) ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
echo ""

SERVER_LOG="$RESULTS_DIR/ferrum_phase_prof__server.log"

CUDA_VISIBLE_DEVICES=0 \
FERRUM_KV_CAPACITY=2048 \
FERRUM_KV_MAX_BLOCKS=2048 \
FERRUM_PAGED_MAX_SEQS=32 \
FERRUM_METAL_PAGED_KV=0 \
FERRUM_MIXED_BATCH=0 \
FERRUM_GREEDY_ARGMAX=1 \
FERRUM_MOE_BUCKETED=1 \
FERRUM_MARLIN_SKIP_WS_ZERO=1 \
FERRUM_MOE_STREAMS=4 \
FERRUM_DECODE_OP_PROFILE=1 \
  "$WORKSPACE/ferrum-infer-rs/target/release/ferrum" serve \
    --model "$MODEL_DIR" --port "$PORT" \
    --gpu-memory-utilization 0.95 \
    > "$SERVER_LOG" 2>&1 &
PID=$!
echo "ferrum pid=$PID logging to $SERVER_LOG"

# Wait for ready
for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ready in ${i}s"
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "process died before becoming ready"
    tail -50 "$SERVER_LOG"
    exit 1
  fi
  sleep 1
done

# Pre-warm so the first scratch alloc + first MoE forward don't pollute
# the [bucket-prof] line we want to read.
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}' \
  >/dev/null

# Mark in the log so we can find the prof window cleanly later.
echo "===PROFILE_START===" >> "$SERVER_LOG"

vllm bench serve \
  --backend openai-chat \
  --base-url "http://127.0.0.1:$PORT" \
  --endpoint /v1/chat/completions \
  --model "$MODEL_DIR" \
  --tokenizer "$MODEL_DIR" \
  --dataset-name random \
  --random-input-len 256 --random-output-len 128 \
  --num-prompts $((32 * 4)) \
  --max-concurrency 32 \
  --save-result --result-dir "$RESULTS_DIR" \
  --result-filename "ferrum_phase_prof_c32.json" \
  --percentile-metrics "ttft,tpot,itl" 2>&1 | tee "$RESULTS_DIR/ferrum_phase_prof_c32.log"

echo "===PROFILE_END===" >> "$SERVER_LOG"

kill -INT "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

echo ""
echo "=== headline ==="
python3 -c "
import json
d = json.load(open('$RESULTS_DIR/ferrum_phase_prof_c32.json'))
print(f'output_throughput={d[\"output_throughput\"]:.1f} tok/s   mean_tpot={d[\"mean_tpot_ms\"]:.2f} ms   mean_ttft={d[\"mean_ttft_ms\"]:.0f} ms')
"

echo ""
echo "=== bucket-prof lines (last 30) ==="
grep '\[bucket-prof\]' "$SERVER_LOG" | tail -30

echo ""
echo "=== batched-decode-prof lines (last 5) ==="
grep '\[batched-decode-prof\]' "$SERVER_LOG" | tail -5
