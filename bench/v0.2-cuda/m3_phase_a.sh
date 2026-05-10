#!/bin/bash
# Phase A baseline — Qwen3-30B-A3B-GPTQ-Int4 on current main (post #148-#153
# cuda/mod.rs split + simple_engine_config drop). Uses ferrum bench (not
# vllm bench serve) to avoid requiring vllm install on the pod; numbers
# are self-comparable to future Phase B/C/D runs but not directly to
# PR #102's `vllm bench serve` numbers (different harness).
#
# Goal: confirm we haven't regressed at c=1/4/16/32 and lock in a starting
# line for the "close-vLLM-gap-to-80%" perf push.

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
HF_HOME="${HF_HOME:-$WORKSPACE/.hf_home}"
MODEL_SNAP_DIR="$HF_HOME/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots"
MODEL_DIR="$MODEL_SNAP_DIR/$(ls "$MODEL_SNAP_DIR" 2>/dev/null | head -1)"
RESULTS_DIR="$WORKSPACE/ferrum-infer-rs/bench/v0.2-cuda/results_phase_a_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "ERROR: Qwen3-30B-A3B-GPTQ-Int4 not found at $MODEL_DIR" >&2
  exit 1
fi

GIT_HEAD=$(git -C "$WORKSPACE/ferrum-infer-rs" rev-parse --short HEAD)
echo "=== Phase A baseline @ $GIT_HEAD ===" | tee "$RESULTS_DIR/summary.txt"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | tee -a "$RESULTS_DIR/summary.txt"
echo "Model: $MODEL_DIR" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"

# Same env as m3_clean.sh PR #102 baseline.
COMMON_ENV=(
  CUDA_VISIBLE_DEVICES=0
  FERRUM_VLLM_MOE=1
  FERRUM_KV_CAPACITY=2048
  FERRUM_KV_MAX_BLOCKS=4096
  FERRUM_PAGED_MAX_SEQS=32
  FERRUM_METAL_PAGED_KV=1
  FERRUM_MIXED_BATCH=0
  FERRUM_GREEDY_ARGMAX=1
  FERRUM_MOE_BUCKETED=1
  FERRUM_MARLIN_SKIP_WS_ZERO=1
  FERRUM_MOE_STREAMS=4
)

for c in 1 4 16 32; do
  echo "--- c=$c ---" | tee -a "$RESULTS_DIR/summary.txt"
  env "${COMMON_ENV[@]}" \
    timeout 600 "$WORKSPACE/ferrum-infer-rs/target/release/ferrum" bench \
      "$MODEL_DIR" --concurrency "$c" --max-tokens 128 --rounds 3 \
      2>&1 | tee "$RESULTS_DIR/c${c}.log"
  echo "" | tee -a "$RESULTS_DIR/summary.txt"
done

echo "" | tee -a "$RESULTS_DIR/summary.txt"
echo "=== Phase A summary ===" | tee -a "$RESULTS_DIR/summary.txt"
for c in 1 4 16 32; do
  line=$(grep -E "Throughput \(e2e\):" "$RESULTS_DIR/c${c}.log" | head -1)
  tpot=$(grep -E "TPOT:" "$RESULTS_DIR/c${c}.log" | head -1)
  ttft=$(grep -E "TTFT:" "$RESULTS_DIR/c${c}.log" | head -1)
  echo "c=$c  $line  $tpot  $ttft" | tee -a "$RESULTS_DIR/summary.txt"
done
