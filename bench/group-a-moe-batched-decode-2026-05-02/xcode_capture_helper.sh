#!/bin/bash
# Helper for the user's Xcode GPU Frame Capture session.
#
# Usage: ./xcode_capture_helper.sh
#
# Steps performed:
#   1. Pre-warm the GGUF page cache.
#   2. Start ferrum-server with the experimental batched-decode path
#      and the Metal HUD env (gives Xcode visibility into Metal events).
#   3. Print the ferrum PID and wait for "ATTACH NOW" key from user.
#   4. After user attaches Xcode and starts capture, send 1 curl
#      request → triggers ONE decode_batch_internal at m=N (configurable).
#   5. Wait for capture to finish (user signals).
#   6. Clean up server.
#
# Why c=4 not c=16:
#   - GPU Frame Capture in Xcode buffers all Metal events; at c=16
#     with 48 layers × ~14 dispatches/layer × 16 tokens, that's
#     ~10K dispatches per round → may overwhelm capture.
#   - c=4 still hits the new batched path (m≥2) with all the same
#     kernels, just smaller pairs, so the per-kernel timing is
#     representative.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FERRUM_BIN="${FERRUM_BIN:-$ROOT/target/release/ferrum}"
GGUF="${GGUF:-/Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf}"
PORT="${PORT:-8783}"
C="${CONCURRENCY:-4}"

echo "=== prewarming page cache for $(basename "$GGUF") ==="
cat "$GGUF" > /dev/null

echo "=== starting ferrum-server with batched-decode opt-in ==="
# MTL_HUD_ENABLED=1 → makes Metal events more visible to Xcode.
# MTL_DEBUG_LAYER=1 → enables Metal debug layer (slows down a bit but
#   Xcode capture shows kernel names cleanly).
# MTL_CAPTURE_ENABLED=1 → required for programmatic capture; harmless
#   for Xcode-side capture too.
MTL_DEBUG_LAYER=1 MTL_HUD_ENABLED=1 MTL_CAPTURE_ENABLED=1 \
  FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS="$C" FERRUM_KV_CAPACITY=1024 FERRUM_MAX_BATCH="$C" \
  FERRUM_MOE_BATCHED=1 FERRUM_MOE_BATCH_THRESHOLD=2 FERRUM_MOE_BATCHED_DECODE=1 \
  FERRUM_DECODE_OP_PROFILE=1 \
  "$FERRUM_BIN" serve --model "$GGUF" --port "$PORT" \
  > /tmp/ferrum_xcode_capture.log 2>&1 &
SERVER_PID=$!
echo ""
echo "================================================================"
echo "  ferrum PID = $SERVER_PID"
echo "  port       = $PORT"
echo "  log        = /tmp/ferrum_xcode_capture.log"
echo "  c          = $C"
echo "================================================================"
echo ""

# Wait for ready.
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[READY] server ready after ${i}s"
    break
  fi
  sleep 1
done

# Warmup
echo ""
echo "=== warmup (one throwaway request to JIT-compile kernels) ==="
curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-30B-A3B-Q4_K_M","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":false,"temperature":0.0}' \
  > /dev/null 2>&1
echo "warmup done"

echo ""
echo "================================================================"
echo "  STEP 1: Open Xcode, attach to PID $SERVER_PID"
echo "    Xcode menu: Debug → Attach to Process → '$SERVER_PID'"
echo ""
echo "  STEP 2: Start GPU Frame Capture"
echo "    Xcode menu: Debug → Capture GPU Workload"
echo "    OR press the camera icon in the Debug bar."
echo ""
echo "  STEP 3: Press ENTER here once capture is armed"
echo "================================================================"
read -p "Press ENTER when GPU capture is armed > "

echo ""
echo "=== driving $C concurrent requests (single decode round per request) ==="
# Fire C concurrent requests with max_tokens=8 (small to keep capture
# focused on one or two decode_batch rounds).
for i in $(seq 1 "$C"); do
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"Qwen3-30B-A3B-Q4_K_M\",\"messages\":[{\"role\":\"user\",\"content\":\"Probe $i\"}],\"max_tokens\":8,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1 &
done
wait
echo "drive done"

echo ""
echo "================================================================"
echo "  STEP 4: Stop GPU Frame Capture in Xcode."
echo "  STEP 5: Save the .gputrace and tell me the path."
echo ""
echo "  Press ENTER to kill the ferrum-server"
echo "================================================================"
read -p "Press ENTER to clean up > "

kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
echo ""
echo "=== done ==="
echo ""
echo "Profile log tail (last batched-decode-prof lines):"
grep batched-decode-prof /tmp/ferrum_xcode_capture.log | tail -5
