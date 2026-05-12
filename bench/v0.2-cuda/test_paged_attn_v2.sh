#!/bin/bash
# Quick c=32 bench with FERRUM_USE_VLLM_PAGED_ATTN=1 to validate the
# paged_attention_v2 port lands the expected -2-3 ms TPOT win vs the
# baseline (1038 tok/s at commit 4a17a17, recorded 2026-05-12).
#
# Mirrors apples_m3_drive.sh::ferrum_start env block + adds
# FERRUM_USE_VLLM_PAGED_ATTN=1.

set -e
cd /workspace/ferrum-infer-rs

PORT=8801
RUN_R=$(date +%s)
LOG=bench/v0.2-cuda/results/ferrum__M3__c32__pa2_r${RUN_R}.server.log
mkdir -p bench/v0.2-cuda/results

pkill -9 -f "ferrum.*serve" 2>/dev/null || true
pkill -9 -f "vllm serve"    2>/dev/null || true
sleep 3

echo "=== starting ferrum w/ FERRUM_USE_VLLM_PAGED_ATTN=1 ==="
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
FERRUM_MOE_BATCH_THRESHOLD=4 \
FERRUM_USE_VLLM_PAGED_ATTN=1 \
  /workspace/ferrum-infer-rs/target/release/ferrum serve \
    --model /workspace/models/M3 --port "$PORT" \
    --gpu-memory-utilization 0.95 \
  > "$LOG" 2>&1 &
FPID=$!
echo "FPID=$FPID LOG=$LOG"

for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "[ferrum] ready in ${i}s"; break; }
  ! kill -0 "$FPID" 2>/dev/null && { echo "[ferrum] died"; tail -80 "$LOG"; exit 1; }
  sleep 1
done

echo "=== prewarm ==="
curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' > /dev/null

source /workspace/vllm-venv/bin/activate
echo "=== bench c=32 r=${RUN_R} ==="
rm -f bench/v0.2-cuda/results/ferrum__M3__c32__r${RUN_R}.*
bash bench/v0.2-cuda/run_cell.sh ferrum M3 32 "${RUN_R}" "$PORT" 2>&1 | tail -30

echo
echo "=== result json ==="
F=bench/v0.2-cuda/results/ferrum__M3__c32__r${RUN_R}.json
if [ -f "$F" ]; then
  python3 -c "
import json, sys
d = json.load(open('$F'))
print(f\"output_throughput={d.get('output_throughput',0):.1f} tok/s\")
print(f\"mean_tpot_ms={d.get('mean_tpot_ms',0):.2f}\")
print(f\"p99_tpot_ms={d.get('p99_tpot_ms',0):.2f}\")
print(f\"completed={d.get('completed',0)} failed={d.get('failed',0)}\")
"
else
  echo "MISSING: $F"
fi

kill -INT "$FPID" 2>/dev/null; sleep 3
pkill -9 -f "ferrum.*serve" 2>/dev/null || true
