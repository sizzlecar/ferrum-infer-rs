Welcome to vast.ai. If authentication fails, try again after a few seconds, and double check your ssh key.
Have fun!
#!/usr/bin/env bash
# Same as apples_m3_drive.sh, ferrum side runs with FERRUM_MOE_GRAPH=1 +
# FERRUM_GRAPH_SKIP_UPLOAD=1 to test graph capture impact at all c.
# Skip the vllm half (already have its baseline).
set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cd /workspace/ferrum-infer-rs
source /workspace/vllm-venv/bin/activate

pkill -9 -f "ferrum.*serve" 2>/dev/null || true
sleep 3
PORT=8801
SERVER_LOG=bench/v0.2-cuda/results/ferrum__M3_graph__r1.server.log
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
FERRUM_MOE_GRAPH=1 \
FERRUM_GRAPH_SKIP_UPLOAD=1 \
  /workspace/ferrum-infer-rs/target/release/ferrum serve \
    --model /workspace/models/M3 --port "$PORT" \
    --gpu-memory-utilization 0.95 \
  > "$SERVER_LOG" 2>&1 &
FPID=$!
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "[ferrum-graph] ready in ${i}s"; break; }
  ! kill -0 "$FPID" 2>/dev/null && { echo "[ferrum-graph] died"; tail -50 "$SERVER_LOG"; exit 1; }
  sleep 1
done
curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' > /dev/null
for c in 1 4 16 32; do
  echo "--- [ferrum-graph] cell c=$c ---"
  rm -f bench/v0.2-cuda/results/ferrum_graph__M3__c${c}__r1.*
  bash bench/v0.2-cuda/run_cell.sh ferrum_graph M3 "$c" 1 "$PORT" || echo "  (cell failed)"
done
kill -INT "$FPID" 2>/dev/null; wait "$FPID" 2>/dev/null || true
pkill -9 -f "ferrum.*serve" 2>/dev/null || true

echo
echo "=== ferrum-graph summary ==="
printf '%-12s %-3s %12s %12s\n' engine c "out_tps" "TPOT_p50_ms"
for c in 1 4 16 32; do
  F=bench/v0.2-cuda/results/ferrum_graph__M3__c${c}__r1.json
  [ -f "$F" ] && python3 -c "
import json, sys
d=json.load(open(sys.argv[1]))
print('%-12s %-3s %12.1f %12.2f' % ('ferrum-graph', '${c}', d['output_throughput'], d.get('median_tpot_ms', float('nan'))))
" "$F"
done
echo "ALL DONE"
