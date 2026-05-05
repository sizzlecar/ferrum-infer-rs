#!/usr/bin/env bash
# run_cell.sh — run ONE bench cell.
#
# Usage:
#   run_cell.sh <engine> <model_tag> <c> <repeat> <port>
#
# Assumes the engine server is already running on $port and ready.
# Calls vendored benchmark_serving.py against it. Saves:
#   results/<engine>__<model_tag>__c<c>__r<repeat>.json
#   results/<engine>__<model_tag>__c<c>__r<repeat>.bench.log
#
# Resume-safe: if the JSON exists with non-zero throughput, skip.

set -euo pipefail

ENGINE="$1"
MODEL_TAG="$2"
C="$3"
REPEAT="$4"
PORT="$5"

WORKSPACE="${WORKSPACE:-/workspace}"
BENCH_DIR="${WORKSPACE}/ferrum-infer-rs/bench/v0.2-cuda"
RESULTS_DIR="$BENCH_DIR/results"
mkdir -p "$RESULTS_DIR"

CELL="${ENGINE}__${MODEL_TAG}__c${C}__r${REPEAT}"
JSON="$RESULTS_DIR/$CELL.json"
LOG="$RESULTS_DIR/$CELL.bench.log"

# Resume: skip if already completed. vLLM 0.20 calls it
# `output_throughput`; older bench harnesses used
# `output_throughput_tok_s`. Accept either so old result dirs still
# resume cleanly.
if [[ -f "$JSON" ]]; then
  EXISTING=$(python3 -c "import json; d=json.load(open('$JSON')); print(d.get('output_throughput', d.get('output_throughput_tok_s', 0)))" 2>/dev/null || echo 0)
  if [[ "$EXISTING" != "0" && "$EXISTING" != "0.0" ]]; then
    echo "[$CELL] skip (already $EXISTING tok/s)"
    exit 0
  fi
fi

# num_prompts = 4 × c, capped at 128 (the size of our prompt set).
NUM_PROMPTS=$((C * 4))
[[ $NUM_PROMPTS -gt 128 ]] && NUM_PROMPTS=128

# W1 (c=1) → output 512, prompt 128
# W2/W3/W4 (c≥4) → output 256, prompt 512
if [[ "$C" -eq 1 ]]; then
  MAX_OUTPUT=512
  PROMPT_LEN=128
else
  MAX_OUTPUT=256
  PROMPT_LEN=512
fi

# nvidia-smi memory sampler (background)
GPU_CSV="$RESULTS_DIR/$CELL.gpu.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
  --format=csv,noheader,nounits -lms 2000 > "$GPU_CSV" &
SMI_PID=$!
trap "kill $SMI_PID 2>/dev/null || true" EXIT

echo "[$CELL] num_prompts=$NUM_PROMPTS max_output=$MAX_OUTPUT"

# `vllm bench serve --model X` does TWO things: (1) sends X as the
# request body "model" field (must match what the server accepts;
# ferrum accepts any name, vLLM matches its --model), (2) loads
# X as a tokenizer for prompt-length stats — so X needs to be a
# valid HF id or a local path with tokenizer.json. Pointing it at
# the model dir works for all our engines.
MODEL_ARG="$WORKSPACE/models/$MODEL_TAG"

# Run with hard timeout — never let one cell burn 15+ min.
# vLLM 0.20+ moved benchmark_serving.py to the CLI: `vllm bench serve`.
timeout 900 vllm bench serve \
  --backend openai-chat \
  --base-url "http://127.0.0.1:$PORT" \
  --endpoint /v1/chat/completions \
  --model "$MODEL_ARG" \
  --dataset-name custom \
  --dataset-path "$BENCH_DIR/prompts.jsonl" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$C" \
  --request-rate inf \
  --temperature 0 \
  --top-p 1 \
  --result-dir "$RESULTS_DIR" \
  --result-filename "$CELL.json" \
  --save-result \
  > "$LOG" 2>&1 || {
    EC=$?
    echo "[$CELL] FAILED (exit=$EC) — see $LOG"
    kill $SMI_PID 2>/dev/null || true
    return $EC 2>/dev/null || exit $EC
  }

kill $SMI_PID 2>/dev/null || true
trap - EXIT

# Print one-line summary. vLLM 0.20 schema: flat keys
# (output_throughput, median_tpot_ms, p99_ttft_ms, …) — older bench
# harness used nested tpot_ms.{median,p95}. p99 instead of p95.
python3 -c "
import json
d = json.load(open('$JSON'))
def f(x): return f'{x:.1f}' if isinstance(x, (int, float)) else 'n/a'
out_tps = d.get('output_throughput')
print(f'[$CELL] out={f(out_tps)} tok/s  TPOT_p50={f(d.get(\"median_tpot_ms\"))}ms  TPOT_p99={f(d.get(\"p99_tpot_ms\"))}ms  TTFT_p50={f(d.get(\"median_ttft_ms\"))}ms  TTFT_p99={f(d.get(\"p99_ttft_ms\"))}ms  ({d.get(\"completed\", 0)}/{d.get(\"completed\", 0)+d.get(\"failed\", 0)} ok)')
" 2>&1 || echo "[$CELL] (could not parse JSON)"
