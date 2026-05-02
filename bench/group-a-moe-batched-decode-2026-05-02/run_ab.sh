#!/bin/bash
# A/B bench: ferrum serve on Qwen3-30B-A3B Q4_K_M with the new
# batched-decode MoE path (one indirect-dispatch GEMV per linear, fixed
# dispatch count) vs the legacy per-token loop. Runs c=1, 4, 8, 16 in
# burst mode using the standard vLLM-style bench_serving.py harness.
#
# Mode toggle:
#   batched   : default (auto-on when backend supports batched MoE GEMV)
#   per_token : FERRUM_MOE_BATCHED=0 (engine kill switch — per-token loop)

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FERRUM_BIN="${FERRUM_BIN:-$ROOT/target/release/ferrum}"
GGUF="${GGUF:-/Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf}"
BENCH_PY="$ROOT/bench/scripts/bench_serving.py"
OUTDIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8783}"

# Concurrencies and their per-c parameters.
# Format: c:num_prompts:max_tokens
DEFAULT_CONFIGS=(
  "1:8:64"
  "4:16:64"
  "8:24:64"
  "16:32:64"
)

if [ "$#" -ge 1 ]; then
  # Filter to a single concurrency.
  WANT_C="$1"
  CONFIGS=()
  for cfg in "${DEFAULT_CONFIGS[@]}"; do
    c="${cfg%%:*}"
    if [ "$c" = "$WANT_C" ]; then
      CONFIGS+=("$cfg")
    fi
  done
  if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "Unknown concurrency $WANT_C" >&2
    exit 1
  fi
else
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

if [ ! -f "$FERRUM_BIN" ]; then
  echo "Missing $FERRUM_BIN — build with cargo build --release --features metal -p ferrum-cli" >&2
  exit 1
fi
if [ ! -f "$GGUF" ]; then
  echo "Missing GGUF $GGUF" >&2
  exit 1
fi
if [ ! -f "$BENCH_PY" ]; then
  echo "Missing $BENCH_PY" >&2
  exit 1
fi

run_one_phase() {
  local mode="$1"      # "batched" or "per_token"
  local c="$2"
  local num_prompts="$3"
  local max_tokens="$4"

  local tag="moe_30b_a3b__${mode}__c${c}"
  local server_log="$OUTDIR/${tag}.server.log"
  local result_json="$OUTDIR/${tag}.json"
  local bench_log="$OUTDIR/${tag}.bench.log"

  echo ""
  echo "=== ${tag} ==="
  date

  # Kill any straggler server.
  pkill -f "ferrum.*serve.*$PORT" 2>/dev/null || true
  sleep 1

  # Mode env.
  #   batched  : opt into both the engine-level batched gate
  #              (FERRUM_MOE_BATCHED=1 with threshold lowered to 2)
  #              AND the experimental batched-decode MoE FFN path
  #              (FERRUM_MOE_BATCHED_DECODE=1).
  #   per_token: legacy per-token loop, no batching at any layer.
  local mode_env=""
  if [ "$mode" = "batched" ]; then
    mode_env="FERRUM_MOE_BATCHED=1 FERRUM_MOE_BATCH_THRESHOLD=2 FERRUM_MOE_BATCHED_DECODE=1"
  fi

  echo "starting ferrum serve (env: $mode_env) → $server_log"
  env \
    FERRUM_METAL_PAGED_KV=1 \
    FERRUM_PAGED_MAX_SEQS="$c" \
    FERRUM_KV_CAPACITY=1024 \
    FERRUM_MAX_BATCH="$c" \
    $mode_env \
    "$FERRUM_BIN" serve --model "$GGUF" --port "$PORT" \
    > "$server_log" 2>&1 &
  local server_pid=$!

  # Wait for the server to be ready (model load).
  echo "waiting for server (pid=$server_pid) to be ready ..."
  local deadline=$((SECONDS + 180))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "server died — last 20 lines of log:"
      tail -20 "$server_log"
      return 1
    fi
    sleep 1
  done
  if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "server failed to come up within 180s"
    kill "$server_pid" 2>/dev/null
    return 1
  fi
  echo "server ready (took ${SECONDS}s)"

  # Warmup: a single throwaway request — pays the mmap cold-cache cost
  # for the 18.6 GB GGUF and primes Metal pipelines. Without this, the
  # first prompt of the real bench inflates TTFT/TPOT by 30-60s on M1
  # Max with the model freshly loaded.
  echo "warmup ..."
  curl -sf -m 180 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$(basename "$GGUF" .gguf)"'","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":false,"temperature":0.0}' \
    > /dev/null 2>&1 || echo "warmup request failed (continuing)"

  # Run bench.
  echo "running bench c=$c num_prompts=$num_prompts max_tokens=$max_tokens ..."
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$(basename "$GGUF" .gguf)" \
    --num-prompts "$num_prompts" \
    --max-concurrency "$c" \
    --max-tokens "$max_tokens" \
    --deterministic-prompts \
    --result-file "$result_json" \
    > "$bench_log" 2>&1

  # Stop server.
  kill "$server_pid" 2>/dev/null
  wait "$server_pid" 2>/dev/null

  echo "done — results in $result_json"
  # Tail key metrics.
  if [ -f "$result_json" ]; then
    python3 -c "
import json
d = json.load(open('$result_json'))
def f(x): return f'{x:.1f}' if x is not None else 'n/a'
ttft = d.get('ttft_ms', {})
tpot = d.get('tpot_ms', {})
itl  = d.get('itl_ms',  {})
print(f\"  output_tok/s={f(d.get('output_throughput_tok_s'))}  req/s={f(d.get('request_throughput_rps'))}\")
print(f\"  TTFT median={f(ttft.get('median'))} p99={f(ttft.get('p99'))} ms\")
print(f\"  TPOT median={f(tpot.get('median'))} p99={f(tpot.get('p99'))} ms\")
print(f\"  ITL  median={f(itl.get('median'))} p99={f(itl.get('p99'))} ms\")
print(f\"  successful={d.get('successful_requests')}/{d.get('num_prompts')}\")
"
  fi
}

# ── Page-cache prewarm ────────────────────────────────────────────────
# 18.6 GB MoE GGUF won't fit cold-cache on 32 GB Mac without paging,
# which inflates whichever phase runs first. `cat > /dev/null` hydrates
# the macOS unified buffer cache so both A/B phases start warm.
echo "prewarming macOS page cache for $(basename "$GGUF") ..."
cat "$GGUF" > /dev/null
echo "prewarm done"

# ── Sweep ──
for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r c num_prompts max_tokens <<< "$cfg"
  for mode in batched per_token; do
    run_one_phase "$mode" "$c" "$num_prompts" "$max_tokens" || \
      echo "phase $mode c=$c failed"
    sleep 3
  done
done

echo ""
echo "=== ALL DONE ==="
date
