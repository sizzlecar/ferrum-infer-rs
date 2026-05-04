#!/usr/bin/env bash
# Engine-level profiling: FERRUM_BATCH_DECODE_PROF=1 prints
# [iter-prof] (sched/process) and [batch-decode-prof] (prep/decode/post +
# logits/lock/sample/emit) every 32 iters from continuous_engine.rs.
# No GPU sync barriers added (host-side wall-clock timers only) so the
# instrumentation is non-perturbing — measured throughput should match
# Phase 8's 484 tok/s baseline closely.
set -uo pipefail

WORKSPACE=/workspace
MODEL_DIR=$WORKSPACE/models/M2
PORT=8800
LOG_DIR=/tmp/engine_prof_run
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/ferrum_server.log"
BENCH_LOG="$LOG_DIR/vllm_bench.log"

pkill -f 'target/release/ferrum' 2>/dev/null || true
sleep 2

echo "[$(date +%H:%M:%S)] starting ferrum (FERRUM_BATCH_DECODE_PROF=1) ..." | tee -a "$SERVER_LOG"
CUDA_VISIBLE_DEVICES=0 \
FERRUM_KV_CAPACITY=2048 \
FERRUM_MAX_BATCH=32 \
FERRUM_BATCH_DECODE_PROF=1 \
  $WORKSPACE/ferrum-infer-rs/target/release/ferrum serve \
    --model "$MODEL_DIR" --port $PORT \
    >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "  server pid=$SERVER_PID"

echo "[$(date +%H:%M:%S)] waiting for ready ..."
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" -o /dev/null 2>&1; then
    echo "  ready at ${i}s"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[FATAL] server died — last 20 lines:"; tail -20 "$SERVER_LOG"; exit 1
  fi
  sleep 1
done

echo "[$(date +%H:%M:%S)] warmup ..."
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"M2","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' \
  > /dev/null
sleep 2

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

kill $SERVER_PID 2>/dev/null
sleep 3
kill -9 $SERVER_PID 2>/dev/null || true

echo ""
echo "=== BENCH SUMMARY ==="
grep -E 'Output token throughput|Mean TPOT|Median TPOT|P99 TPOT|Successful requests' "$BENCH_LOG" | head -20

echo ""
echo "=== iter-prof samples ==="
grep '\[iter-prof\]' "$SERVER_LOG" | head -10

echo ""
echo "=== batch-decode-prof samples ==="
grep '\[batch-decode-prof\]' "$SERVER_LOG" | head -10

echo ""
echo "=== aggregate (iter-prof) ==="
grep '\[iter-prof\]' "$SERVER_LOG" | python3 - <<'PY'
import sys, re
samples = []
pat = re.compile(r"iter#(\d+) total=(\d+)us sched=(\d+)us process=(\d+)us batch_size=(\d+)")
for line in sys.stdin:
    m = pat.search(line)
    if m:
        samples.append([int(g) for g in m.groups()])
if not samples:
    print("NO iter-prof samples (check raw lines below)")
    sys.exit(0)
n = len(samples)
def avg(i): return sum(s[i] for s in samples)//n
def med(i):
    sorted_v = sorted(s[i] for s in samples)
    return sorted_v[n//2]
total, sched, proc, bs = avg(1), avg(2), avg(3), avg(4)
print(f"n={n}  avg iter total={total}us  sched={sched}us({100*sched/total:.1f}%)  process={proc}us({100*proc/total:.1f}%)  avg_batch={bs}")
print(f"   medians: total={med(1)}us sched={med(2)}us process={med(3)}us")
PY

echo ""
echo "=== aggregate (batch-decode-prof) ==="
grep '\[batch-decode-prof\]' "$SERVER_LOG" | python3 - <<'PY'
import sys, re
samples = []
pat = re.compile(r"call#(\d+) m=(\d+) total=(\d+)us prep=(\d+)us decode=(\d+)us post=(\d+)us \(logits=(\d+)us lock=(\d+)us sample=(\d+)us emit=(\d+)us\)")
for line in sys.stdin:
    m = pat.search(line)
    if m:
        samples.append([int(g) for g in m.groups()])
if not samples:
    print("NO batch-decode-prof samples (check raw)")
    sys.exit(0)
n = len(samples)
def avg(i): return sum(s[i] for s in samples)//n
def pct(x,t): return f'{100*x/t:5.1f}%' if t else '   0%'
mm = avg(1); total = avg(2); prep = avg(3); dec = avg(4); post = avg(5)
log = avg(6); lock = avg(7); samp = avg(8); emit = avg(9)
print(f"n={n}  m={mm}  total/call={total}us")
print(f"   prep={prep}us({pct(prep,total)})  decode={dec}us({pct(dec,total)})  post={post}us({pct(post,total)})")
print(f"   post breakdown (totals across {mm} seqs):")
print(f"     logits.to_vec_f32 = {log}us({pct(log,total)})  ← per-seq sync device→host")
print(f"     write-lock acquire= {lock}us({pct(lock,total)})  ← per-seq")
print(f"     sample            = {samp}us({pct(samp,total)})  ← per-seq host work")
print(f"     send_stream_update= {emit}us({pct(emit,total)})  ← per-seq tokio send")
PY

echo ""
echo "=== FILES ==="
echo "  server log: $SERVER_LOG (size $(wc -c < $SERVER_LOG))"
echo "  bench log:  $BENCH_LOG"
echo "  bench json: $LOG_DIR/bench_c16.json"
