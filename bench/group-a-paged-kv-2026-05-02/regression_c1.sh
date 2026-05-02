#!/bin/bash
# c=1 regression — verify the m=1 paged-KV branch doesn't regress
# vs the contig path. Doc baseline c=1 was 43.7 (Qwen3-30B-A3B
# default per-token). Compare today's m=1 paged + per_token contig.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FERRUM_BIN="${FERRUM_BIN:-$ROOT/target/release/ferrum}"
GGUF_DIR="${GGUF_DIR:-/Users/chejinxuan/ferrum-bench/models}"
BENCH_PY="$ROOT/bench/scripts/bench_serving.py"
OUTDIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8783

MODELS=(
  "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|Llama-3.1-8B"
  "Qwen3-8B-Q4_K_M.gguf|Qwen3-8B"
  "Qwen3-30B-A3B-Q4_K_M.gguf|Qwen3-30B-A3B"
)

C=1
NUM_PROMPTS=8
MAX_TOKENS=64

for entry in "${MODELS[@]}"; do
  IFS='|' read -r gguf_name model_label <<< "$entry"
  GGUF="$GGUF_DIR/$gguf_name"
  if [ ! -f "$GGUF" ]; then continue; fi
  cat "$GGUF" > /dev/null

  for mode in "paged" "contig"; do
    pkill -9 -f "ferrum.*serve" 2>/dev/null; sleep 1
    if [ "$mode" = "paged" ]; then
      env_str="FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS=4 FERRUM_KV_CAPACITY=512"
    else
      env_str=""
    fi
    echo ""
    echo "▶ $model_label / $mode"
    env $env_str FERRUM_MAX_BATCH=1 \
      "$FERRUM_BIN" serve --model "$GGUF" --port "$PORT" \
      > "$OUTDIR/c1__${model_label}__${mode}.server.log" 2>&1 &
    FPID=$!
    for i in $(seq 1 60); do
      if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
      sleep 1
    done
    curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$model_label\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
      > /dev/null 2>&1
    python3 "$BENCH_PY" \
      --base-url "http://127.0.0.1:$PORT" \
      --model "$model_label" \
      --num-prompts $NUM_PROMPTS --max-concurrency $C --max-tokens $MAX_TOKENS \
      --deterministic-prompts \
      --result-file "$OUTDIR/c1__${model_label}__${mode}.json" \
      > "$OUTDIR/c1__${model_label}__${mode}.bench.log" 2>&1
    python3 -c "
import json
d = json.load(open('$OUTDIR/c1__${model_label}__${mode}.json'))
def f(x): return f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
ttft=d.get('ttft_ms',{}); tpot=d.get('tpot_ms',{})
print(f'  out_tok/s={f(d.get(\"output_throughput_tok_s\"))}  TPOT_med={f(tpot.get(\"median\"))}  TTFT_med={f(ttft.get(\"median\"))}')
" 2>/dev/null
    kill $FPID 2>/dev/null; wait $FPID 2>/dev/null
  done
done
