#!/bin/bash
# Phase 3 A/B bench: run `ferrum bench-serve` under 3 env-var variants
# on Qwen3-30B-A3B-GPTQ-Int4 to isolate the contribution of each layer
# of the routing-on-device refactor.
#
# Usage:
#   bash bench/v0.2-cuda/bench_phase3_variants.sh baseline      # VLLM_MOE only
#   bash bench/v0.2-cuda/bench_phase3_variants.sh device_route  # + DEVICE_ROUTE
#   bash bench/v0.2-cuda/bench_phase3_variants.sh graph         # + MOE_GRAPH
#
# Each variant spins up its own `ferrum serve`, runs c=1/8/16/32 sweep
# via `ferrum bench-serve`, then tears down.
#
# Expected post-Phase 3 numbers (RTX 4090, vs vLLM 0.20.1 ~1870 tok/s c=32):
#   baseline      : c=32 ≈ 717.5 tok/s  (PR #173 reference)
#   device_route  : c=32 ≈ 700-740 tok/s  (parity — host path redundant but kept)
#   graph         : c=32 ≈ 880-900 tok/s  (+15-25% TPOT, ~50% of vLLM)

set -uo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
HF_HOME="${HF_HOME:-$WORKSPACE/.hf_home}"
MODEL_SNAP_DIR="$HF_HOME/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots"
MODEL_DIR="$MODEL_SNAP_DIR/$(ls "$MODEL_SNAP_DIR" 2>/dev/null | head -1)"

VARIANT="${1:?Usage: $0 baseline|device_route|graph}"
case "$VARIANT" in
  baseline)
    EXTRA_ENV=""
    LABEL="baseline (VLLM_MOE only)"
    ;;
  device_route)
    EXTRA_ENV="FERRUM_MOE_DEVICE_ROUTE=1"
    LABEL="device_route (+ VLLM_MOE)"
    ;;
  graph)
    EXTRA_ENV="FERRUM_MOE_DEVICE_ROUTE=1 FERRUM_MOE_GRAPH=1"
    LABEL="graph (+ device_route + VLLM_MOE)"
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 1
    ;;
esac

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "ERROR: Qwen3-30B-A3B-GPTQ-Int4 not at $MODEL_DIR" >&2
  exit 1
fi

RESULTS_DIR="$WORKSPACE/ferrum-infer-rs/bench/v0.2-cuda/phase3_${VARIANT}_$(date +%H%M%S)"
mkdir -p "$RESULTS_DIR"
GIT_HEAD=$(git -C "$WORKSPACE/ferrum-infer-rs" rev-parse --short HEAD)
PORT=8801
SERVER_LOG="$RESULTS_DIR/ferrum_server.log"

echo "=== $LABEL @ $GIT_HEAD ===" | tee "$RESULTS_DIR/summary.txt"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | tee -a "$RESULTS_DIR/summary.txt"
echo "Model: $MODEL_DIR" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"

# Always-on baseline env (matches m3_bench_serve.sh defaults).
ALWAYS_ON="CUDA_VISIBLE_DEVICES=0 FERRUM_VLLM_MOE=1 FERRUM_KV_CAPACITY=2048 FERRUM_KV_MAX_BLOCKS=4096 FERRUM_PAGED_MAX_SEQS=32 FERRUM_METAL_PAGED_KV=1 FERRUM_MIXED_BATCH=0 FERRUM_GREEDY_ARGMAX=1 FERRUM_MOE_BUCKETED=1 FERRUM_MARLIN_SKIP_WS_ZERO=1 FERRUM_MOE_STREAMS=4"

eval "$ALWAYS_ON $EXTRA_ENV $WORKSPACE/ferrum-infer-rs/target/release/ferrum serve \
    --model $MODEL_DIR --port $PORT \
    --gpu-memory-utilization 0.95 \
    > $SERVER_LOG 2>&1 &"
PID=$!

echo "ferrum PID: $PID env: $EXTRA_ENV" | tee -a "$RESULTS_DIR/summary.txt"
for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ready in ${i}s" | tee -a "$RESULTS_DIR/summary.txt"
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "FERRUM DIED before ready" | tee -a "$RESULTS_DIR/summary.txt"
    tail -50 "$SERVER_LOG" | tee -a "$RESULTS_DIR/summary.txt"
    exit 1
  fi
  sleep 1
done

curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}' >/dev/null
echo "prewarm done" | tee -a "$RESULTS_DIR/summary.txt"

for c in 1 8 16 32; do
  echo "" | tee -a "$RESULTS_DIR/summary.txt"
  echo "--- c=$c ---" | tee -a "$RESULTS_DIR/summary.txt"
  "$WORKSPACE/ferrum-infer-rs/target/release/ferrum" bench-serve \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL_DIR" \
    --tokenizer "$MODEL_DIR" \
    --random-input-len 256 \
    --random-output-len 128 \
    --num-prompts $((c * 4)) \
    --max-concurrency "$c" \
    --result-file "$RESULTS_DIR/c${c}.json" 2>&1 | tee "$RESULTS_DIR/c${c}.log"
done

kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null || true

# Summary table.
echo "" | tee -a "$RESULTS_DIR/summary.txt"
echo "=== $VARIANT tok/s + TPOT ===" | tee -a "$RESULTS_DIR/summary.txt"
for c in 1 8 16 32; do
  if [[ -f "$RESULTS_DIR/c${c}.json" ]]; then
    tok=$(python3 -c "import json; d=json.load(open('$RESULTS_DIR/c${c}.json')); print(f\"{d.get('output_throughput', d.get('request_throughput', 0)):.1f}\")" 2>/dev/null)
    tpot=$(python3 -c "import json; d=json.load(open('$RESULTS_DIR/c${c}.json')); print(f\"{d.get('mean_tpot_ms', 0):.2f}\")" 2>/dev/null)
    echo "c=$c: ${tok} tok/s, TPOT ${tpot}ms" | tee -a "$RESULTS_DIR/summary.txt"
  fi
done
echo "RESULTS: $RESULTS_DIR"
