#!/bin/bash
# nsys profile of vllm serve + apples c=32 bench. Captures the bench
# window (--delay skips vllm load + warmup). Output:
#   /tmp/vllm_bench.nsys-rep
#   /tmp/vllm_kernels.csv

set -e
cd /workspace/ferrum-infer-rs

OUT_REP=/tmp/vllm_bench.nsys-rep
OUT_KER=/tmp/vllm_kernels.csv
PORT=8800

pkill -9 -f "vllm serve"   2>/dev/null || true
pkill -9 -f "ferrum.*serve" 2>/dev/null || true
sleep 3

source /workspace/vllm-venv/bin/activate

# Start vllm under nsys with --delay 60 (skip vllm boot, ~30-50s) +
# --duration 60 (covers the apples bench client window).
# IMPORTANT: nsys profile takes `--` before the command.
echo "[nsys_vllm] launching vllm under nsys..."
nohup nsys profile \
    --output "$OUT_REP" \
    --trace cuda \
    --delay 60 \
    --duration 60 \
    --force-overwrite true \
    -- vllm serve /workspace/models/M3 --port "$PORT" \
        --max-num-seqs 64 --max-model-len 4096 \
        --no-enable-prefix-caching --no-enable-log-requests \
        --quantization gptq_marlin \
    > /tmp/vllm_nsys_server.log 2>&1 &
NSYS_PID=$!
echo "  NSYS_PID=$NSYS_PID"

# Wait for vllm ready (typically 30-50s on M3)
echo "[nsys_vllm] waiting for vllm ready..."
for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "  vllm ready in ${i}s"
        break
    fi
    ! kill -0 $NSYS_PID 2>/dev/null && {
        echo "  nsys/vllm died"; tail -50 /tmp/vllm_nsys_server.log; exit 1
    }
    sleep 1
done

echo "[nsys_vllm] firing apples c=32 bench (cell)..."
bash bench/v0.2-cuda/run_cell.sh vllm M3 32 "nsys$(date +%s)" "$PORT" 2>&1 | tail -10 || true

echo "[nsys_vllm] waiting for nsys to finalize..."
# Wait up to 60s past --duration end for finalize
wait $NSYS_PID 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 5

echo
echo "[nsys_vllm] generating kernel CSV..."
nsys stats --report cuda_gpu_kern_sum --format csv "$OUT_REP" > "$OUT_KER" 2>/dev/null

echo
echo "[nsys_vllm] done"
echo "  $OUT_REP"
echo "  $OUT_KER"
echo
echo "=== top 25 kernels by total time ==="
head -1 "$OUT_KER"
tail -n +2 "$OUT_KER" | sort -t',' -k2 -n -r | head -25
