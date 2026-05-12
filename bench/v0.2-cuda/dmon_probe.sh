#!/bin/bash
# Fast bottleneck localization via nvidia-smi dmon.
# Runs the bench, samples DRAM + SM util at 1s intervals during steady-state
# decode, averages the middle window (skips warmup + tail).
#
# Usage:  bash bench/v0.2-cuda/dmon_probe.sh <ferrum|vllm>
#
# Output:
#   /tmp/<engine>_dmon.csv     raw 1s samples
#   /tmp/<engine>_dmon.summary mean of [warmup_end .. tail_skip] window

set -e
ENGINE="${1:-ferrum}"
OUT_RAW="/tmp/${ENGINE}_dmon.csv"
OUT_SUM="/tmp/${ENGINE}_dmon.summary"

# Apples env
export CUDA_VISIBLE_DEVICES=0
export FERRUM_VLLM_MOE=1
export FERRUM_KV_CAPACITY=2048
export FERRUM_KV_MAX_BLOCKS=4096
export FERRUM_PAGED_MAX_SEQS=32
export FERRUM_METAL_PAGED_KV=1
export FERRUM_MIXED_BATCH=0
export FERRUM_GREEDY_ARGMAX=1
export FERRUM_MOE_BUCKETED=1
export FERRUM_MARLIN_SKIP_WS_ZERO=1
export FERRUM_MOE_STREAMS=4
export FERRUM_MOE_BATCH_THRESHOLD=4
export FERRUM_MOE_GRAPH=1
export FERRUM_GRAPH_SKIP_UPLOAD=1

# Start dmon in background, csv with timestamps
# fields: gpuidx pwr gtemp sm mem enc dec mclk pclk
# we want: sm (SM %), mem (DRAM %), pwr (W), mclk (mem clock), pclk (gpu clock)
echo "[dmon] starting nvidia-smi dmon 90s sample @ 1s..."
nvidia-smi dmon -s pumct -i 0 -d 1 -c 90 -o T > "$OUT_RAW" 2>&1 &
DMON_PID=$!

# Brief delay so dmon header is captured first
sleep 2

case "$ENGINE" in
    ferrum)
        # Long enough bench to span ~60s of steady-state decode
        echo "[dmon] launching ferrum bench (rounds=10, max_tokens=256)..."
        /workspace/ferrum-infer-rs/target/release/ferrum bench \
            /workspace/models/M3 \
            --concurrency 32 \
            --max-tokens 256 \
            --rounds 10 \
            2>&1 | tail -25 &
        BENCH_PID=$!
        wait $BENCH_PID
        ;;
    vllm)
        # vllm: start server, fire 3 bench cells back to back
        echo "[dmon] launching vllm server..."
        cd /workspace/ferrum-infer-rs
        source /workspace/vllm-venv/bin/activate
        pkill -9 -f "vllm serve" 2>/dev/null || true
        sleep 3
        nohup vllm serve /workspace/models/M3 --port 8800 \
            --max-num-seqs 64 --max-model-len 4096 \
            --no-enable-prefix-caching --no-enable-log-requests \
            --quantization gptq_marlin \
            > /tmp/vllm_dmon_server.log 2>&1 &
        VLLM_PID=$!
        # Wait for ready
        for i in $(seq 1 180); do
            if curl -sf "http://127.0.0.1:8800/v1/models" >/dev/null 2>&1; then
                echo "  vllm ready in ${i}s"
                break
            fi
            ! kill -0 $VLLM_PID 2>/dev/null && { echo "vllm died"; tail -50 /tmp/vllm_dmon_server.log; kill $DMON_PID 2>/dev/null; exit 1; }
            sleep 1
        done
        # Fire 3 bench cells (each ~17s) for ~50s of steady-state
        for cell in a b c; do
            echo "[dmon] vllm bench cell $cell..."
            bash bench/v0.2-cuda/run_cell.sh vllm M3 32 "dmon_$cell" 8800 2>&1 | tail -3 || true
        done
        kill -INT $VLLM_PID 2>/dev/null || true
        sleep 3
        ;;
    *)
        echo "unknown engine: $ENGINE"; exit 1;;
esac

# Wait for dmon to finish its window
echo "[dmon] waiting for dmon to finish..."
wait $DMON_PID 2>/dev/null || true

echo "[dmon] sampling done. Computing steady-state averages..."

# Parse dmon csv. Format (col positions after header):
#   #Date       Time         gpu   pwr  gtemp  sm   mem   enc  dec  jpg  ofa  mclk  pclk
# We skip first 15 samples (warmup) and last 5 (tail), compute mean of sm + mem.
python3 - "$OUT_RAW" "$OUT_SUM" <<'PYEOF'
import sys
inp, out = sys.argv[1], sys.argv[2]
with open(inp) as f:
    lines = [l.rstrip() for l in f if l.strip()]
# data lines have leading date "yyyymmdd" or 8-char date — skip header lines (start with '#')
rows = []
for l in lines:
    if l.startswith('#') or not l[0].isdigit():
        continue
    parts = l.split()
    if len(parts) < 13:
        continue
    # Columns by inspection: date, time, gpu, pwr, gtemp, sm, mem, enc, dec, jpg, ofa, mclk, pclk
    try:
        sm = int(parts[5]); mem = int(parts[6]); pwr = float(parts[3]); pclk = int(parts[11]); mclk = int(parts[10])
        rows.append((sm, mem, pwr, mclk, pclk))
    except ValueError:
        continue
n = len(rows)
if n < 30:
    open(out, 'w').write(f"too few samples: {n}\n")
    sys.exit(0)
# skip first 15 (warmup) and last 5
window = rows[15:n-5]
sm_avg = sum(r[0] for r in window) / len(window)
mem_avg = sum(r[1] for r in window) / len(window)
pwr_avg = sum(r[2] for r in window) / len(window)
mclk_avg = sum(r[3] for r in window) / len(window)
pclk_avg = sum(r[4] for r in window) / len(window)
# Peak sm/mem observed
sm_peak = max(r[0] for r in window)
mem_peak = max(r[1] for r in window)
with open(out, 'w') as f:
    f.write(f"samples_total={n}\n")
    f.write(f"samples_window={len(window)}\n")
    f.write(f"SM_util_avg={sm_avg:.1f}%   peak={sm_peak}%\n")
    f.write(f"DRAM_util_avg={mem_avg:.1f}%  peak={mem_peak}%\n")
    f.write(f"power_avg={pwr_avg:.1f}W\n")
    f.write(f"clock_GPU_avg={pclk_avg:.0f}MHz\n")
    f.write(f"clock_MEM_avg={mclk_avg:.0f}MHz\n")
print(open(out).read())
PYEOF

echo
echo "[dmon] done."
echo "  raw: $OUT_RAW"
echo "  summary: $OUT_SUM"
