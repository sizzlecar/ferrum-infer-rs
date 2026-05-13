#!/bin/bash
# Bisect helper: build ferrum at given commit, run M2 c=32 bench, output tok/s.
# Usage: bisect_m2.sh <commit-sha>
set -e
cd /workspace/ferrum-infer-rs
SHA="$1"
[ -z "$SHA" ] && { echo "usage: $0 <sha>"; exit 1; }

echo "[bisect] checkout $SHA"
git stash 2>&1 | tail -1 || true
git checkout "$SHA" 2>&1 | tail -1

echo "[bisect] cargo build (incremental)..."
source /root/.cargo/env
cargo build --release --features cuda,vllm-moe-marlin -p ferrum-cli --bin ferrum 2>&1 | tail -2

# kill any stragglers
pkill -9 -f "ferrum.*serve" 2>/dev/null || true
pkill -9 -f "vllm" 2>/dev/null || true
pkill -9 -f "VLLM" 2>/dev/null || true
sleep 3

echo "[bisect] starting ferrum on M2..."
PORT=8801
LOG=/tmp/bisect_m2.log
CUDA_VISIBLE_DEVICES=0 \
FERRUM_KV_CAPACITY=2048 FERRUM_KV_MAX_BLOCKS=4096 \
FERRUM_PAGED_MAX_SEQS=32 FERRUM_METAL_PAGED_KV=1 FERRUM_GREEDY_ARGMAX=1 \
FERRUM_MARLIN_SKIP_WS_ZERO=1 \
  /workspace/ferrum-infer-rs/target/release/ferrum serve \
    --model /workspace/models/M2 --port "$PORT" \
    --gpu-memory-utilization 0.95 \
  > "$LOG" 2>&1 &
FPID=$!
for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "[bisect] ready in ${i}s"; break; }
    ! kill -0 $FPID 2>/dev/null && { echo "[bisect] FERRUM DIED"; tail -50 "$LOG"; exit 2; }
    sleep 1
done
curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' > /dev/null

source /workspace/vllm-venv/bin/activate
RES=bench/v0.2-cuda/results/bisect__M2__c32__${SHA}.json
rm -f bench/v0.2-cuda/results/bisect__M2__c32__${SHA}.*

bash bench/v0.2-cuda/run_cell.sh ferrum M2 32 "bisect_${SHA}" "$PORT" 2>&1 | tail -3

kill -INT $FPID 2>/dev/null; sleep 3
pkill -9 -f "ferrum.*serve" 2>/dev/null || true

# Print result
F="bench/v0.2-cuda/results/ferrum__M2__c32__rbisect_${SHA}.json"
if [ -f "$F" ]; then
    python3 -c "
import json
d = json.load(open('$F'))
print(f\"[bisect] $SHA: out={d.get('output_throughput',0):.1f} tok/s TPOT_p50={d.get('mean_tpot_ms',0):.2f}ms\")
"
else
    echo "[bisect] NO RESULT FILE"
fi
