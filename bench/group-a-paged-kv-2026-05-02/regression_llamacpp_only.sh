#!/bin/bash
# Same-day llama.cpp baseline on dense Group A models. The doc's 74.8 /
# 71.7 numbers are from a few days ago — confirm whether ferrum's
# 96/93 today is a real win or just llama.cpp also moved.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GGUF_DIR="${GGUF_DIR:-/Users/chejinxuan/ferrum-bench/models}"
BENCH_PY="$ROOT/bench/scripts/bench_serving.py"
OUTDIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8001}"

MODELS=(
  "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|Llama-3.1-8B-Q4_K_M"
  "Qwen3-8B-Q4_K_M.gguf|Qwen3-8B-Q4_K_M"
)

C=16
NUM_PROMPTS=32
MAX_TOKENS=64

for entry in "${MODELS[@]}"; do
  IFS='|' read -r gguf_name model_label <<< "$entry"
  GGUF="$GGUF_DIR/$gguf_name"
  if [ ! -f "$GGUF" ]; then continue; fi

  echo ""
  echo "▶ llama-server $model_label"
  pkill -9 -f "llama-server" 2>/dev/null; sleep 1
  cat "$GGUF" > /dev/null

  llama-server --model "$GGUF" --port "$PORT" \
    --ctx-size 4096 --parallel "$C" --batch-size 2048 --jinja \
    > "$OUTDIR/regression__${model_label}__llamacpp.server.log" 2>&1 &
  LPID=$!
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "  ready in ${i}s"
      break
    fi
    sleep 1
  done
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_label\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$model_label" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$C" --max-tokens "$MAX_TOKENS" \
    --deterministic-prompts \
    --result-file "$OUTDIR/regression__${model_label}__llamacpp.json" \
    > "$OUTDIR/regression__${model_label}__llamacpp.bench.log" 2>&1
  python3 -c "
import json
d = json.load(open('$OUTDIR/regression__${model_label}__llamacpp.json'))
def f(x): return f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
ttft=d.get('ttft_ms',{}); tpot=d.get('tpot_ms',{})
print(f\"  out_tok/s={f(d.get('output_throughput_tok_s'))}  TPOT_med={f(tpot.get('median'))}  TTFT_med={f(ttft.get('median'))}\")
" 2>/dev/null

  kill $LPID 2>/dev/null; wait $LPID 2>/dev/null
  sleep 2
done
