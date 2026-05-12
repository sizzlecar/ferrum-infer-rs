Welcome to vast.ai. If authentication fails, try again after a few seconds, and double check your ssh key.
Have fun!
#!/usr/bin/env bash
# apples-to-apples M3 bench v2: with FERRUM_MIXED_BATCH=1
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cd /workspace/ferrum-infer-rs

pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "ferrum.*serve"    2>/dev/null || true
pkill -9 -f "vllm serve"       2>/dev/null || true
sleep 3

source /workspace/vllm-venv/bin/activate

# 1. vLLM 0.20.2 server (unchanged baseline)
echo "=== [vllm] starting ==="
PORT=8800
SERVER_LOG=bench/v0.2-cuda/results/vllm__M3__r1.server.log
CUDA_VISIBLE_DEVICES=0 \
  vllm serve /workspace/models/M3 --port "$PORT" \
    --max-num-seqs 64 --max-model-len 4096 \
    --no-enable-prefix-caching --no-enable-log-requests \
    --quantization gptq_marlin \
  > "$SERVER_LOG" 2>&1 &
VPID=$!
for i in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "[vllm] ready in ${i}s"; break; }
  ! kill -0 "$VPID" 2>/dev/null && { echo "[vllm] died"; tail -80 "$SERVER_LOG"; exit 1; }
  sleep 1
done
curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' > /dev/null
for c in 1 4 16 32; do
  echo "--- [vllm] cell c=$c ---"
  rm -f bench/v0.2-cuda/results/vllm__M3__c${c}__r1.*
  bash bench/v0.2-cuda/run_cell.sh vllm M3 "$c" 1 "$PORT" || echo "  (cell failed)"
done
kill -INT "$VPID"; wait "$VPID" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
deactivate 2>/dev/null || true
sleep 5

# 2. ferrum (MIXED_BATCH=1 — NEW)
echo "=== [ferrum] starting with FERRUM_MIXED_BATCH=1 ==="
PORT=8801
SERVER_LOG=bench/v0.2-cuda/results/ferrum__M3__r1.server.log
CUDA_VISIBLE_DEVICES=0 \
FERRUM_VLLM_MOE=1 \
FERRUM_KV_CAPACITY=2048 \
FERRUM_KV_MAX_BLOCKS=4096 \
FERRUM_PAGED_MAX_SEQS=32 \
FERRUM_METAL_PAGED_KV=1 \
FERRUM_MIXED_BATCH=1 \
FERRUM_GREEDY_ARGMAX=1 \
FERRUM_MOE_BUCKETED=1 \
FERRUM_MARLIN_SKIP_WS_ZERO=1 \
FERRUM_MOE_STREAMS=4 \
  /workspace/ferrum-infer-rs/target/release/ferrum serve \
    --model /workspace/models/M3 --port "$PORT" \
    --gpu-memory-utilization 0.95 \
  > "$SERVER_LOG" 2>&1 &
FPID=$!
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "[ferrum] ready in ${i}s"; break; }
  ! kill -0 "$FPID" 2>/dev/null && { echo "[ferrum] died"; tail -80 "$SERVER_LOG"; exit 1; }
  sleep 1
done
curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' > /dev/null
source /workspace/vllm-venv/bin/activate
for c in 1 4 16 32; do
  echo "--- [ferrum] cell c=$c ---"
  rm -f bench/v0.2-cuda/results/ferrum__M3__c${c}__r1.*
  bash bench/v0.2-cuda/run_cell.sh ferrum M3 "$c" 1 "$PORT" || echo "  (cell failed)"
done
kill -INT "$FPID"; wait "$FPID" 2>/dev/null || true
pkill -9 -f "ferrum.*serve" 2>/dev/null || true

echo
echo "=== summary (with MIXED_BATCH=1) ==="
printf '%-8s %-3s %12s %12s\n' engine c "out_tps" "TPOT_p50_ms"
for engine in vllm ferrum; do
  for c in 1 4 16 32; do
    F=bench/v0.2-cuda/results/${engine}__M3__c${c}__r1.json
    [ -f "$F" ] && python3 -c "
import json, sys
d=json.load(open(sys.argv[1]))
print('%-8s %-3s %12.1f %12.2f' % ('${engine}', '${c}', d['output_throughput'], d.get('median_tpot_ms', float('nan'))))
" "$F"
  done
done
echo "ALL DONE"
