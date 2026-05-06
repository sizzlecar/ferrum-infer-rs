#!/bin/bash
# Fill the per-c curve for dense Group A models (Llama-3.1-8B and
# Qwen3-8B). Already have c=1 (regression_c1.sh) and c=16
# (regression_group_a.sh); this script adds c=4 and c=8.
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
)

CONFIGS=(
  "4:16:64"
  "8:24:64"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r gguf_name model_label <<< "$entry"
  GGUF="$GGUF_DIR/$gguf_name"
  if [ ! -f "$GGUF" ]; then continue; fi
  cat "$GGUF" > /dev/null

  for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r c num_prompts max_tokens <<< "$cfg"

    pkill -9 -f "ferrum.*serve" 2>/dev/null; sleep 1
    echo ""
    echo "▶ $model_label / c=$c"
    FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS=$((c * 2)) FERRUM_KV_CAPACITY=512 \
      FERRUM_MAX_BATCH=$c \
      "$FERRUM_BIN" serve --model "$GGUF" --port "$PORT" \
      > "$OUTDIR/dense__${model_label}__c${c}.server.log" 2>&1 &
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
      --num-prompts $num_prompts --max-concurrency $c --max-tokens $max_tokens \
      --deterministic-prompts \
      --result-file "$OUTDIR/dense__${model_label}__c${c}.json" \
      > "$OUTDIR/dense__${model_label}__c${c}.bench.log" 2>&1
    python3 -c "
import json
d = json.load(open('$OUTDIR/dense__${model_label}__c${c}.json'))
def f(x): return f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
ttft=d.get('ttft_ms',{}); tpot=d.get('tpot_ms',{})
print(f'  out_tok/s={f(d.get(\"output_throughput_tok_s\"))}  TPOT_med={f(tpot.get(\"median\"))}  TTFT_med={f(ttft.get(\"median\"))}')
" 2>/dev/null
    kill $FPID 2>/dev/null; wait $FPID 2>/dev/null
  done
done
