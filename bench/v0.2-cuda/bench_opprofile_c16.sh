#!/usr/bin/env bash
# Run ferrum with FERRUM_DECODE_OP_PROFILE=1 and bench at c=16.
# Captures batched-op-profile lines from server.log, summarizes.
set -uo pipefail

WORKSPACE=/workspace
MODEL_DIR=$WORKSPACE/models/M2
PORT=8800
LOG_DIR=/tmp/opprofile_run
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/ferrum_server.log"
BENCH_LOG="$LOG_DIR/vllm_bench.log"

# Kill any leftover ferrum
pkill -f 'target/release/ferrum' 2>/dev/null || true
sleep 2

# Start server with op-profile enabled
echo "[$(date +%H:%M:%S)] starting ferrum (FERRUM_DECODE_OP_PROFILE=1) ..." | tee -a "$SERVER_LOG"
CUDA_VISIBLE_DEVICES=0 \
FERRUM_KV_CAPACITY=2048 \
FERRUM_MAX_BATCH=32 \
FERRUM_DECODE_OP_PROFILE=1 \
  $WORKSPACE/ferrum-infer-rs/target/release/ferrum serve \
    --model "$MODEL_DIR" --port $PORT \
    >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "  server pid=$SERVER_PID"

# Wait for ready (up to 90s)
echo "[$(date +%H:%M:%S)] waiting for ready ..."
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" -o /dev/null 2>&1; then
    echo "  ready at ${i}s"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[FATAL] server died — last 20 lines:"
    tail -20 "$SERVER_LOG"
    exit 1
  fi
  sleep 1
done

# Warmup with a tiny request
echo "[$(date +%H:%M:%S)] warmup ..."
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"M2","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' \
  > /dev/null
sleep 2

# Reset op-profile counters by reading and discarding pre-bench lines
PRE_PROFILE_LINES=$(grep -c 'batched-op-profile' "$SERVER_LOG" 2>/dev/null)
PRE_PROFILE_LINES=${PRE_PROFILE_LINES:-0}
echo "[$(date +%H:%M:%S)] pre-bench profile lines: $PRE_PROFILE_LINES"

# Smaller workload — 32 prompts × 128 output tokens to fit timeout
# comfortably and produce many clean per-iter profile lines.
echo "[$(date +%H:%M:%S)] running vllm bench c=16 ..."
timeout 240 vllm bench serve \
  --backend openai-chat \
  --base-url "http://127.0.0.1:$PORT" \
  --endpoint /v1/chat/completions \
  --model "$MODEL_DIR" \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 128 \
  --num-prompts 32 \
  --max-concurrency 16 \
  --request-rate inf \
  --temperature 0 \
  --top-p 1 \
  --ignore-eos \
  --result-dir "$LOG_DIR" \
  --result-filename bench_c16.json \
  --save-result \
  > "$BENCH_LOG" 2>&1
EC=$?
echo "[$(date +%H:%M:%S)] bench exit=$EC"

# Stop server (graceful)
kill $SERVER_PID 2>/dev/null
sleep 3
kill -9 $SERVER_PID 2>/dev/null || true

# Print bench summary
echo ""
echo "=== BENCH SUMMARY ==="
grep -E 'Output token throughput|Mean TPOT|Median TPOT|P99 TPOT|Successful requests' "$BENCH_LOG" | head -20

# Extract op-profile lines AFTER warmup
echo ""
echo "=== OP-PROFILE (post-warmup, only m>1 batched iters) ==="
TOTAL_PROF_LINES=$(grep -c 'batched-op-profile' "$SERVER_LOG" 2>/dev/null)
TOTAL_PROF_LINES=${TOTAL_PROF_LINES:-0}
POST_LINES=$((TOTAL_PROF_LINES - PRE_PROFILE_LINES))
echo "captured $POST_LINES post-warmup batched-op-profile lines (warmup=$PRE_PROFILE_LINES, total=$TOTAL_PROF_LINES)"

# Group by m, average each component
grep 'batched-op-profile' "$SERVER_LOG" | tail -n $POST_LINES | python3 - <<'PY'
import sys, re
from collections import defaultdict
buckets = defaultdict(lambda: defaultdict(list))
pat = re.compile(r"m=(\d+) total=(\d+)us  matmul=(\d+)us\((\d+)\) attn=(\d+)us\((\d+)\) qkr=(\d+)us\((\d+)\) norm=(\d+)us\((\d+)\) other=(\d+)us\((\d+)\)  unwrapped=(\d+)us")
for line in sys.stdin:
    m = pat.search(line)
    if not m: continue
    M = int(m.group(1))
    buckets[M]['total'].append(int(m.group(2)))
    buckets[M]['matmul'].append(int(m.group(3))); buckets[M]['matmul_n'].append(int(m.group(4)))
    buckets[M]['attn'].append(int(m.group(5)));   buckets[M]['attn_n'].append(int(m.group(6)))
    buckets[M]['qkr'].append(int(m.group(7)));    buckets[M]['qkr_n'].append(int(m.group(8)))
    buckets[M]['norm'].append(int(m.group(9)));   buckets[M]['norm_n'].append(int(m.group(10)))
    buckets[M]['other'].append(int(m.group(11))); buckets[M]['other_n'].append(int(m.group(12)))
    buckets[M]['unwr'].append(int(m.group(13)))

if not buckets:
    print("NO DATA — patterns didn't match. First 3 lines:")
    sys.exit(0)

print(f"{'m':>4} {'iters':>6} {'total_avg_us':>12} {'mm':>10} {'attn':>10} {'qkr':>10} {'norm':>9} {'other':>9} {'unwr':>9}  --  pct (mm/attn/qkr/other)")
def avg(xs): return sum(xs)//len(xs) if xs else 0
def pct(x, t): return f"{(100*x/t):.1f}%" if t else "0.0%"
for M in sorted(buckets):
    b = buckets[M]
    n = len(b['total']); t = avg(b['total'])
    mm = avg(b['matmul']); at = avg(b['attn']); qk = avg(b['qkr']); nm = avg(b['norm']); ot = avg(b['other']); un = avg(b['unwr'])
    print(f"{M:>4} {n:>6} {t:>12} {mm:>10} {at:>10} {qk:>10} {nm:>9} {ot:>9} {un:>9}   {pct(mm,t)}/{pct(at,t)}/{pct(qk,t)}/{pct(ot,t)}")
    # per-call avg for matmuls
    if b['matmul_n']:
        mm_n = sum(b['matmul_n']) // len(b['matmul_n'])
        if mm_n:
            mm_per = avg(b['matmul']) // mm_n
            print(f"     mm calls/iter≈{mm_n}  mm_per_call≈{mm_per}us")
    if b['attn_n']:
        at_n = sum(b['attn_n']) // len(b['attn_n'])
        if at_n:
            at_per = avg(b['attn']) // at_n
            print(f"     attn calls/iter≈{at_n}  attn_per_call≈{at_per}us")
PY

echo ""
echo "=== FILES ==="
echo "  server log: $SERVER_LOG (size $(wc -c < $SERVER_LOG))"
echo "  bench log:  $BENCH_LOG"
echo "  bench json: $LOG_DIR/bench_c16.json"
