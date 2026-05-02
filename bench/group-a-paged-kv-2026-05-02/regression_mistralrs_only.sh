#!/bin/bash
# Just the mistralrs half of the Group A regression — ferrum already
# has clean results. mistralrs CLI takes --port AFTER `serve`, which
# the original regression script got wrong; this fixes it.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GGUF_DIR="${GGUF_DIR:-/Users/chejinxuan/ferrum-bench/models}"
TOK_DIR="${TOK_DIR:-/Users/chejinxuan/ferrum-bench/tokenizers}"
BENCH_PY="$ROOT/bench/scripts/bench_serving.py"
OUTDIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8002}"

MODELS=(
  "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|Meta-Llama-3.1-8B-Instruct.tokenizer.json|Llama-3.1-8B-Q4_K_M"
  "Qwen3-8B-Q4_K_M.gguf|Qwen3-8B.tokenizer.json|Qwen3-8B-Q4_K_M"
  "Qwen3-30B-A3B-Q4_K_M.gguf|Qwen3-30B-A3B.tokenizer.json|Qwen3-30B-A3B-Q4_K_M"
)

C=16
NUM_PROMPTS=32
MAX_TOKENS=64

for entry in "${MODELS[@]}"; do
  IFS='|' read -r gguf_name tok_name model_label <<< "$entry"
  GGUF="$GGUF_DIR/$gguf_name"
  TOK="$TOK_DIR/$tok_name"
  if [ ! -f "$GGUF" ]; then continue; fi

  echo ""
  echo "▶ mistralrs $model_label"
  pkill -9 -f "mistralrs.*serve" 2>/dev/null; sleep 1
  cat "$GGUF" > /dev/null

  mistralrs serve --port "$PORT" --max-seqs $C text \
    --format gguf -m "$GGUF_DIR" -f "$gguf_name" -t "$TOK" \
    > "$OUTDIR/regression__${model_label}__mistralrs.server.log" 2>&1 &
  MPID=$!
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "  ready in ${i}s"
      break
    fi
    sleep 1
  done
  if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "  FAILED to come up"
    tail -3 "$OUTDIR/regression__${model_label}__mistralrs.server.log"
    kill $MPID 2>/dev/null; wait $MPID 2>/dev/null
    continue
  fi
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT" \
    --model default \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$C" --max-tokens "$MAX_TOKENS" \
    --deterministic-prompts \
    --result-file "$OUTDIR/regression__${model_label}__mistralrs.json" \
    > "$OUTDIR/regression__${model_label}__mistralrs.bench.log" 2>&1
  python3 -c "
import json
d = json.load(open('$OUTDIR/regression__${model_label}__mistralrs.json'))
def f(x): return f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
ttft=d.get('ttft_ms',{}); tpot=d.get('tpot_ms',{})
print(f\"  out_tok/s={f(d.get('output_throughput_tok_s'))}  TPOT_med={f(tpot.get('median'))}  TTFT_med={f(ttft.get('median'))}\")
" 2>/dev/null

  kill $MPID 2>/dev/null; wait $MPID 2>/dev/null
  sleep 2
done
