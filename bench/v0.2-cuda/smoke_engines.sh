#!/usr/bin/env bash
# smoke_engines.sh — verify each engine boots + serves on M2 BEFORE
# launching the long sweep. ~1-2 min. Catches the "engine doesn't
# even start" failures cheaply (vs discovering at cell 30 of the
# sweep).
#
# Run AFTER setup.sh finishes and BEFORE run_sweep.sh.

set -uo pipefail

BENCH_DIR="${BENCH_DIR:-/workspace/ferrum-infer-rs/bench/v0.2-cuda}"
RESULTS_DIR="$BENCH_DIR/results"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/_smoke.log"
M2=/workspace/models/M2

# Kill any leftover engines
pkill -9 -f "ferrum.*serve"  2>/dev/null
pkill -9 -f "vllm"           2>/dev/null
sleep 3

probe() {
  local name="$1" port="$2" max="${3:-180}"
  for i in $(seq 1 "$max"); do
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "  $name ready in ${i}s" | tee -a "$LOG"
      return 0
    fi
    sleep 1
  done
  echo "  $name TIMEOUT (${max}s)" | tee -a "$LOG"
  return 1
}

chat() {
  local port="$1" model="$2"
  local resp
  resp=$(curl -sf -m 60 -X POST "http://127.0.0.1:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"reply OK\"}],\"max_tokens\":10,\"stream\":false,\"temperature\":0}")
  if [[ "$resp" == *"choices"* ]]; then
    local content
    content=$(echo "$resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"][:80])')
    echo "  chat ok: $content" | tee -a "$LOG"
    return 0
  fi
  echo "  chat FAIL: ${resp:0:300}" | tee -a "$LOG"
  return 1
}

echo "=== smoke_engines on M2 — $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "$LOG"

# ── ferrum ──
echo "" | tee -a "$LOG"
echo "=== ferrum ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 nohup /workspace/ferrum-infer-rs/target/release/ferrum serve \
  --model "$M2" --port 9201 \
  > "$RESULTS_DIR/_smoke_ferrum.log" 2>&1 &
PID=$!
FAIL=0
probe ferrum 9201 120 || FAIL=1
[[ $FAIL -eq 0 ]] && { chat 9201 default || FAIL=1; }
kill -9 $PID 2>/dev/null
wait $PID 2>/dev/null
sleep 5

# ── vllm ──
echo "" | tee -a "$LOG"
echo "=== vllm 0.20 ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$M2" --port 9202 --max-num-seqs 4 --quantization gptq_marlin \
  --no-enable-prefix-caching \
  > "$RESULTS_DIR/_smoke_vllm.log" 2>&1 &
PID=$!
probe vllm 9202 240 || FAIL=$((FAIL+1))
[[ $FAIL -lt 2 ]] && { chat 9202 "$M2" || FAIL=$((FAIL+1)); }
kill -9 $PID 2>/dev/null
wait $PID 2>/dev/null
sleep 3

echo "" | tee -a "$LOG"
echo "=== smoke done — $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

if [[ $FAIL -gt 0 ]] ; then
  echo "" | tee -a "$LOG"
  echo "✗ smoke had $FAIL failure(s) — check _smoke_*.log; do NOT launch run_sweep.sh yet" | tee -a "$LOG"
  exit 1
fi
echo "✓ both engines smoke-passed; safe to run bash $BENCH_DIR/run_sweep.sh" | tee -a "$LOG"
