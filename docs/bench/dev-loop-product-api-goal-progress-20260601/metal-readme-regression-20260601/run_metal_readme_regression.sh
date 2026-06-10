#!/usr/bin/env bash
set -euo pipefail
OUT="docs/bench/dev-loop-product-api-goal-progress-20260601/metal-readme-regression-20260601"
ROOT="$PWD"
FERRUM_BIN="$ROOT/target/release/ferrum"
BENCH_PY="$ROOT/bench/scripts/bench_serving.py"
GGUF_DIR="/Users/chejinxuan/ferrum-bench/models"
PORT=18181
BASE="http://127.0.0.1:${PORT}"
mkdir -p "$OUT"

pkill -9 -f 'ferrum.*serve|llama-server|mistralrs.*serve' 2>/dev/null || true
sleep 5
{
  date '+%Y-%m-%d %H:%M:%S %z'
  sw_vers 2>/dev/null || true
  sysctl vm.swapusage || true
  vm_stat | head -12 || true
  "$FERRUM_BIN" --version || true
} > "$OUT/env_start.txt"

SERVER_PID=""
stop_server() {
  local label="$1"
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  pkill -9 -f 'ferrum.*serve' 2>/dev/null || true
  sleep 8
  sysctl vm.swapusage > "$OUT/${label}.swap_after_server.txt" || true
}

start_server() {
  local model_path="$1" label="$2" moe="$3"
  pkill -9 -f 'ferrum.*serve' 2>/dev/null || true
  sleep 8
  sysctl vm.swapusage > "$OUT/${label}.swap_before_server.txt" || true
  if [ "$moe" = "1" ]; then
    FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS=32 FERRUM_KV_CAPACITY=512 FERRUM_MAX_BATCH=16 \
    FERRUM_MOE_BATCHED=1 FERRUM_MOE_BATCHED_DECODE=1 FERRUM_MOE_BATCH_THRESHOLD=2 \
      "$FERRUM_BIN" serve --model "$model_path" --host 127.0.0.1 --port "$PORT" \
      > "$OUT/${label}.server.stdout" 2> "$OUT/${label}.server.stderr" &
  else
    FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS=32 FERRUM_KV_CAPACITY=512 FERRUM_MAX_BATCH=16 \
      "$FERRUM_BIN" serve --model "$model_path" --host 127.0.0.1 --port "$PORT" \
      > "$OUT/${label}.server.stdout" 2> "$OUT/${label}.server.stderr" &
  fi
  SERVER_PID=$!
  echo "$SERVER_PID" > "$OUT/${label}.server.pid"
  for _ in $(seq 1 180); do
    if curl -fsS --noproxy '*' "$BASE/v1/models" > "$OUT/${label}.models.json" 2> "$OUT/${label}.models.stderr"; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited before ready for ${label}" >&2
      return 1
    fi
    sleep 1
  done
  echo "server did not become ready for ${label}" >&2
  return 1
}

json_chat() {
  local payload="$1" response="$2"
  curl -fsS --noproxy '*' -H 'Content-Type: application/json' -d @"$payload" "$BASE/v1/chat/completions" > "$response"
}

chat_check() {
  local label="$1" model="$2"
  cat > "$OUT/${label}.paris_payload.json" <<JSON
{"model":"${model}","messages":[{"role":"user","content":"What is the capital of France? Reply with just the city name."}],"max_tokens":16,"temperature":0,"stream":false}
JSON
  json_chat "$OUT/${label}.paris_payload.json" "$OUT/${label}.paris_response.json"
  python3 - <<'PY' "$OUT/${label}.paris_response.json" "$OUT/${label}.paris_verdict.txt"
import json, sys
body=json.load(open(sys.argv[1], encoding='utf-8'))
content=body['choices'][0]['message']['content']
passed='paris' in content.lower()
open(sys.argv[2],'w',encoding='utf-8').write(f"content={content}\npassed={str(passed).lower()}\n")
if not passed: raise SystemExit(1)
PY
  cat > "$OUT/${label}.multiturn_payload.json" <<JSON
{"model":"${model}","messages":[{"role":"system","content":"You are concise and follow instructions exactly."},{"role":"user","content":"Remember this code word: basalt. Reply only: remembered."},{"role":"assistant","content":"remembered"},{"role":"user","content":"What code word did I ask you to remember? Reply with just the word."}],"max_tokens":16,"temperature":0,"stream":false}
JSON
  json_chat "$OUT/${label}.multiturn_payload.json" "$OUT/${label}.multiturn_response.json"
  python3 - <<'PY' "$OUT/${label}.multiturn_response.json" "$OUT/${label}.multiturn_verdict.txt"
import json, sys
body=json.load(open(sys.argv[1], encoding='utf-8'))
content=body['choices'][0]['message']['content']
passed='basalt' in content.lower()
open(sys.argv[2],'w',encoding='utf-8').write(f"content={content}\npassed={str(passed).lower()}\n")
if not passed: raise SystemExit(1)
PY
}

bench_cell() {
  local label="$1" model="$2" c="$3" prompts="$4" max_tokens=64
  sysctl vm.swapusage > "$OUT/${label}.c${c}.swap_before_bench.txt" || true
  python3 "$BENCH_PY" \
    --base-url "$BASE" \
    --model "$model" \
    --num-prompts "$prompts" \
    --max-concurrency "$c" \
    --max-tokens "$max_tokens" \
    --deterministic-prompts \
    --result-file "$OUT/${label}.c${c}.json" \
    > "$OUT/${label}.c${c}.bench.log" 2>&1
  sysctl vm.swapusage > "$OUT/${label}.c${c}.swap_after_bench.txt" || true
  python3 - <<'PY' "$OUT/${label}.c${c}.json" "$OUT/${label}.c${c}.summary.txt"
import json, sys
j=json.load(open(sys.argv[1], encoding='utf-8'))
out=j.get('output_throughput_tok_s')
tpot=j.get('tpot_ms',{}).get('median')
ttft=j.get('ttft_ms',{}).get('median')
open(sys.argv[2],'w',encoding='utf-8').write(f"output_throughput_tok_s={out}\ntpot_median_ms={tpot}\nttft_median_ms={ttft}\n")
print(f"{sys.argv[1]} throughput={out:.3f} tok/s tpot_med={tpot:.3f}ms ttft_med={ttft:.3f}ms")
PY
}

run_model() {
  local label="$1" model_file="$2" model="$3" moe="$4"
  local model_path="$GGUF_DIR/$model_file"
  start_server "$model_path" "$label" "$moe"
  chat_check "$label" "$model"
  case "$label" in
    llama31_8b)
      bench_cell "$label" "$model" 1 8
      bench_cell "$label" "$model" 8 24
      bench_cell "$label" "$model" 16 32
      ;;
    qwen3_8b)
      bench_cell "$label" "$model" 16 32
      ;;
    qwen3_30b_a3b)
      bench_cell "$label" "$model" 16 32
      ;;
  esac
  stop_server "$label"
}

run_model llama31_8b "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" "Llama-3.1-8B" 0
run_model qwen3_8b "Qwen3-8B-Q4_K_M.gguf" "Qwen3-8B" 0
run_model qwen3_30b_a3b "Qwen3-30B-A3B-Q4_K_M.gguf" "Qwen3-30B-A3B" 1

sysctl vm.swapusage > "$OUT/env_end_swap.txt" || true
printf 'done: %s\n' "$OUT"
