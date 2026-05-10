#!/bin/bash
# Phase A apples-to-apples baseline using `ferrum serve` + Python httpx
# async client. Mirrors `vllm bench serve`'s protocol (random input, max
# concurrency, streaming) so numbers compare directly to PR #102's
# c=1=104.5 / c=8=295.6 / c=16=374.6 / c=32=391.1 tok/s baseline.
#
# Lighter-weight than `vllm bench serve` — no vllm install needed,
# only httpx (already on the pod). Tokenizer is loaded via transformers
# if available; if not, prompts use a fixed-size dummy vocab.

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
HF_HOME="${HF_HOME:-$WORKSPACE/.hf_home}"
MODEL_SNAP_DIR="$HF_HOME/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots"
MODEL_DIR="$MODEL_SNAP_DIR/$(ls "$MODEL_SNAP_DIR" 2>/dev/null | head -1)"
RESULTS_DIR="$WORKSPACE/ferrum-infer-rs/bench/v0.2-cuda/results_serve_phase_a_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "ERROR: Qwen3-30B-A3B-GPTQ-Int4 not at $MODEL_DIR" >&2
  exit 1
fi

GIT_HEAD=$(git -C "$WORKSPACE/ferrum-infer-rs" rev-parse --short HEAD)
PORT=8800
CLIENT_PY="$WORKSPACE/ferrum-infer-rs/bench/v0.2-cuda/bench_serve_client.py"
SERVER_LOG="$RESULTS_DIR/ferrum_server.log"

echo "=== Phase A serve baseline @ $GIT_HEAD ===" | tee "$RESULTS_DIR/summary.txt"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | tee -a "$RESULTS_DIR/summary.txt"
echo "Model: $MODEL_DIR" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"

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
  "$WORKSPACE/ferrum-infer-rs/target/release/ferrum" serve \
    --model "$MODEL_DIR" --port "$PORT" \
    --gpu-memory-utilization 0.95 \
    > "$SERVER_LOG" 2>&1 &
PID=$!

echo "ferrum PID: $PID — waiting for /v1/models …" | tee -a "$RESULTS_DIR/summary.txt"
for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ready in ${i}s" | tee -a "$RESULTS_DIR/summary.txt"
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "ferrum died before becoming ready" | tee -a "$RESULTS_DIR/summary.txt"
    tail -50 "$SERVER_LOG"
    exit 1
  fi
  sleep 1
done

# Prewarm: small request to flush tokenizer + initial graph capture.
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}' \
  >/dev/null
echo "prewarm done" | tee -a "$RESULTS_DIR/summary.txt"

for c in 1 8 16 32; do
  echo "" | tee -a "$RESULTS_DIR/summary.txt"
  echo "--- c=$c ---" | tee -a "$RESULTS_DIR/summary.txt"
  python3 "$CLIENT_PY" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL_DIR" \
    --tokenizer "$MODEL_DIR" \
    --random-input-len 256 \
    --random-output-len 128 \
    --num-prompts $((c * 4)) \
    --max-concurrency "$c" \
    --result-file "$RESULTS_DIR/c${c}.json" 2>&1 | tee "$RESULTS_DIR/c${c}.log"
done

kill -INT "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

echo "" | tee -a "$RESULTS_DIR/summary.txt"
echo "=== Phase A serve summary ===" | tee -a "$RESULTS_DIR/summary.txt"
for c in 1 8 16 32; do
  if [[ -f "$RESULTS_DIR/c${c}.json" ]]; then
    python3 -c "
import json
d = json.load(open('$RESULTS_DIR/c${c}.json'))
print(f'c={d[\"max_concurrency\"]:>2}  output_tps={d[\"output_throughput\"]:>7.1f}  '
      f'mean_tpot={d[\"mean_tpot_ms\"]:>7.2f}ms  mean_ttft={d[\"mean_ttft_ms\"]:>6.0f}ms  '
      f'completed={d[\"completed\"]}/{d[\"num_prompts\"]}  fail={d[\"failed\"]}')
" | tee -a "$RESULTS_DIR/summary.txt"
  fi
done
