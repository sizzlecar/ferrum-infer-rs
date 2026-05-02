#!/bin/bash
# Capture xctrace Metal System Trace of ferrum-server at c=16 batched
# decode. Output: /tmp/ferrum_trace_$TS.trace + an exported per-kernel
# CSV via xctrace export.
#
# Usage: ./xctrace_ferrum.sh [output_dir]
#   output_dir defaults to ../bench/group-a-moe-batched-decode-2026-05-02/

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FERRUM_BIN="${FERRUM_BIN:-$ROOT/target/release/ferrum}"
GGUF="${GGUF:-/Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf}"
BENCH_PY="$ROOT/bench/scripts/bench_serving.py"
OUTDIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
PORT=8783
TS="$(date +%Y%m%d_%H%M%S)"
TRACE_PATH="/tmp/ferrum_xctrace_${TS}.trace"
TIME_LIMIT="${TIME_LIMIT:-15s}"

if [ ! -f "$FERRUM_BIN" ]; then
  echo "Missing $FERRUM_BIN" >&2; exit 1
fi
if [ ! -f "$GGUF" ]; then
  echo "Missing $GGUF" >&2; exit 1
fi

# Pre-warm the macOS page cache so xctrace doesn't capture mmap fault
# storms instead of actual Metal work.
echo "prewarm cache..."
cat "$GGUF" > /dev/null

# Start ferrum-server with the experimental batched-decode path engaged.
# FERRUM_MOE_BATCHED=1 + threshold=2 → engine-level batching kicks in
# at m≥2. FERRUM_MOE_BATCHED_DECODE=1 → uses the new GEMV-based path
# (the fused gate+up+silu kernel + threshold decoupling fix from 3b68b91).
echo "starting ferrum-server..."
FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS=16 FERRUM_KV_CAPACITY=1024 FERRUM_MAX_BATCH=16 \
  FERRUM_MOE_BATCHED=1 FERRUM_MOE_BATCH_THRESHOLD=2 FERRUM_MOE_BATCHED_DECODE=1 \
  "$FERRUM_BIN" serve --model "$GGUF" --port "$PORT" \
  > /tmp/ferrum_xctrace_server.log 2>&1 &
SERVER_PID=$!
echo "ferrum-server pid=$SERVER_PID"

# Wait for the server to be ready.
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ready after ${i}s"
    break
  fi
  sleep 1
done

# One warmup request — ensures Metal pipelines are JIT-compiled and any
# scratch allocations have happened, so the trace captures steady-state
# decode work, not setup.
echo "warmup..."
curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-30B-A3B-Q4_K_M","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":false,"temperature":0.0}' \
  > /dev/null 2>&1

# Start xctrace attached to ferrum-server. Runs in foreground for
# TIME_LIMIT, capturing the Metal System Trace into TRACE_PATH.
# We launch the bench load in background so it overlaps with the
# capture window.
echo "starting xctrace recording for $TIME_LIMIT into $TRACE_PATH ..."
(
  sleep 2  # let xctrace warm up before bench starts
  echo "driving c=16 bench burst..."
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT" \
    --model Qwen3-30B-A3B-Q4_K_M \
    --num-prompts 16 --max-concurrency 16 --max-tokens 64 \
    --deterministic-prompts \
    > /tmp/ferrum_xctrace_bench.log 2>&1 &
  echo "bench pid=$!"
) &
DRIVER_PID=$!

xcrun xctrace record \
  --template 'Metal System Trace' \
  --output "$TRACE_PATH" \
  --time-limit "$TIME_LIMIT" \
  --attach "$SERVER_PID" \
  --no-prompt 2>&1 | tail -10

echo "xctrace done; killing server..."
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
wait "$DRIVER_PID" 2>/dev/null

if [ -d "$TRACE_PATH" ] || [ -f "$TRACE_PATH" ]; then
  echo "trace saved: $TRACE_PATH"
  echo ""
  echo "--- table of contents ---"
  xcrun xctrace export --input "$TRACE_PATH" --toc 2>&1 | head -60
else
  echo "ERROR: trace file not created"
  exit 1
fi
