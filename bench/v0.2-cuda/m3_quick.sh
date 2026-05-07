#!/bin/bash
# M3 (Qwen3-30B-A3B-GPTQ-INT4) focused bench: ferrum bucketed vs
# ferrum per-pair vs vllm. Smaller matrix than run_sweep.sh — designed
# to validate the bucketed path correctness + ballpark perf in <30 min.
#
# Run on the 4090 pod after build completes:
#   bash bench/v0.2-cuda/m3_quick.sh

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
BENCH_DIR="${WORKSPACE}/ferrum-infer-rs/bench/v0.2-cuda"
RESULTS_DIR="$BENCH_DIR/results_m3_quick"
MODELS_DIR="${WORKSPACE}/models"
MODEL_DIR="$MODELS_DIR/M3"
mkdir -p "$RESULTS_DIR"

PORT=8800

ferrum_start() {
  local mode="$1"  # "bucketed" or "per_pair"
  local server_log="$RESULTS_DIR/ferrum_${mode}__server.log"
  local bucketed_env=""
  case "$mode" in
    bucketed) bucketed_env="FERRUM_MOE_BUCKETED=1" ;;
    per_pair) bucketed_env="FERRUM_MOE_BUCKETED=0" ;;
  esac
  echo "  starting ferrum (${mode}) on M3 ..." >&2
  CUDA_VISIBLE_DEVICES=0 \
  FERRUM_KV_CAPACITY=2048 \
  FERRUM_KV_MAX_BLOCKS=2048 \
  FERRUM_PAGED_MAX_SEQS=64 \
  FERRUM_METAL_PAGED_KV=1 \
  FERRUM_MIXED_BATCH=1 \
  FERRUM_GREEDY_ARGMAX=1 \
  FERRUM_UNIFIED_GRAPH=0 \
  $bucketed_env \
    "$WORKSPACE/ferrum-infer-rs/target/release/ferrum" serve \
      --model "$MODEL_DIR" --port "$PORT" \
      --gpu-memory-utilization 0.95 \
      > "$server_log" 2>&1 &
  ENGINE_PID=$!
}

vllm_start() {
  local server_log="$RESULTS_DIR/vllm__server.log"
  echo "  starting vllm on M3 ..." >&2
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --port "$PORT" \
    --max-num-seqs 32 \
    --max-model-len 4096 \
    --no-enable-prefix-caching \
    --no-enable-log-requests \
    --quantization gptq_marlin \
    > "$server_log" 2>&1 &
  ENGINE_PID=$!
}

wait_ready() {
  local pid="$1"
  for i in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "  ready in ${i}s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "  process died before becoming ready"
      return 1
    fi
    sleep 1
  done
  echo "  timeout (10 min)"
  return 1
}

prewarm() {
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}' \
    >/dev/null
}

run_bench() {
  local label="$1"
  local concurrency="$2"
  local out_file="$RESULTS_DIR/${label}_c${concurrency}.json"
  echo "  bench $label c=$concurrency ..."
  vllm bench serve \
    --backend openai-chat \
    --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/chat/completions \
    --model "$MODEL_DIR" \
    --tokenizer "$MODEL_DIR" \
    --dataset-name random \
    --random-input-len 256 --random-output-len 128 \
    --num-prompts $((concurrency * 8)) \
    --max-concurrency "$concurrency" \
    --save-result --result-dir "$RESULTS_DIR" \
    --result-filename "${label}_c${concurrency}.json" \
    --percentile-metrics "ttft,tpot,itl" 2>&1 | tee "$RESULTS_DIR/${label}_c${concurrency}.log"
}

run_engine() {
  local engine="$1"
  case "$engine" in
    ferrum_bucketed) ferrum_start bucketed ;;
    ferrum_per_pair) ferrum_start per_pair ;;
    vllm)            vllm_start ;;
  esac
  if ! wait_ready "$ENGINE_PID"; then
    echo "[FAIL] $engine never became ready" | tee -a "$RESULTS_DIR/summary.txt"
    kill -9 "$ENGINE_PID" 2>/dev/null
    return 1
  fi
  prewarm
  for c in 1 4 16; do
    run_bench "$engine" "$c"
  done
  kill -INT "$ENGINE_PID" 2>/dev/null
  wait "$ENGINE_PID" 2>/dev/null
  sleep 5
}

echo "=== M3 quick bench ==="
echo "Output dir: $RESULTS_DIR"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
echo ""

for engine in ferrum_bucketed ferrum_per_pair vllm; do
  echo "--- $engine ---"
  run_engine "$engine" || echo "skipping rest of $engine"
  echo ""
done

echo "=== summary ==="
for f in "$RESULTS_DIR"/*.json; do
  [[ -f "$f" ]] || continue
  python3 -c "
import json, sys
d = json.load(open('$f'))
fn = '$f'.split('/')[-1]
print(f'{fn:40s} TPOT={d.get(\"mean_tpot_ms\", \"?\"):.2f}ms output_tps={d.get(\"output_throughput\", \"?\"):.1f}')
" 2>/dev/null || echo "$f: parse failed"
done
