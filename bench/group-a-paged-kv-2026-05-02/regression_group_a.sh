#!/bin/bash
# Group A regression: ferrum (paged batched) + mistralrs CLI (serve)
# on the 3 Group A models @ c=16, validating
#   - the paged-KV mirror in Qwen3MoeModel didn't break dense models
#   - ferrum's c=16 throughput on each model vs mistralrs and the
#     historical baseline.
#
# Group A models:
#   - Llama-3.1-8B-Instruct Q4_K_M (dense, 4.6 GB)
#   - Qwen3-8B Q4_K_M (dense, 4.7 GB)
#   - Qwen3-30B-A3B Q4_K_M (MoE, 17 GB)

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FERRUM_BIN="${FERRUM_BIN:-$ROOT/target/release/ferrum}"
GGUF_DIR="${GGUF_DIR:-/Users/chejinxuan/ferrum-bench/models}"
TOK_DIR="${TOK_DIR:-/Users/chejinxuan/ferrum-bench/tokenizers}"
BENCH_PY="$ROOT/bench/scripts/bench_serving.py"
OUTDIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8783}"
MISTRAL_PORT="${MISTRAL_PORT:-8002}"

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

  if [ ! -f "$GGUF" ]; then echo "SKIP: $GGUF missing" >&2; continue; fi

  echo ""
  echo "=========================================================="
  echo "  $model_label  (gguf=$gguf_name)"
  echo "=========================================================="

  # Prewarm page cache for this model.
  cat "$GGUF" > /dev/null

  # ── ferrum paged batched ──────────────────────────────────────────
  pkill -9 -f "ferrum.*serve.*$PORT" 2>/dev/null; sleep 1
  echo ""
  echo "▶ ferrum paged batched @ c=$C"
  FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS=$((C * 2)) \
    FERRUM_KV_CAPACITY=512 FERRUM_MAX_BATCH=$C \
    FERRUM_MOE_BATCHED=1 FERRUM_MOE_BATCH_THRESHOLD=2 FERRUM_MOE_BATCHED_DECODE=1 \
    "$FERRUM_BIN" serve --model "$GGUF" --port "$PORT" \
    > "$OUTDIR/regression__${model_label}__ferrum.server.log" 2>&1 &
  FPID=$!
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  curl -sf -m 90 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_label\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$model_label" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$C" --max-tokens "$MAX_TOKENS" \
    --deterministic-prompts \
    --result-file "$OUTDIR/regression__${model_label}__ferrum.json" \
    > "$OUTDIR/regression__${model_label}__ferrum.bench.log" 2>&1
  python3 -c "
import json
d = json.load(open('$OUTDIR/regression__${model_label}__ferrum.json'))
def f(x): return f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
ttft=d.get('ttft_ms',{}); tpot=d.get('tpot_ms',{})
print(f\"  out_tok/s={f(d.get('output_throughput_tok_s'))}  TPOT_med={f(tpot.get('median'))}  TTFT_med={f(ttft.get('median'))}\")
" 2>/dev/null || echo "  (no result)"
  kill $FPID 2>/dev/null; wait $FPID 2>/dev/null

  # ── mistralrs serve ───────────────────────────────────────────────
  pkill -9 -f "mistralrs.*serve.*$MISTRAL_PORT" 2>/dev/null; sleep 1
  echo ""
  echo "▶ mistralrs serve @ c=$C"
  mistralrs serve --port "$MISTRAL_PORT" --max-seqs $C text \
    --format gguf -m "$GGUF_DIR" -f "$gguf_name" -t "$TOK" \
    > "$OUTDIR/regression__${model_label}__mistralrs.server.log" 2>&1 &
  MPID=$!
  # mistralrs takes longer to load.
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$MISTRAL_PORT/v1/models" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  curl -sf -m 90 -X POST "http://127.0.0.1:$MISTRAL_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"any\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$MISTRAL_PORT" \
    --model any \
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
" 2>/dev/null || echo "  (no result)"
  kill $MPID 2>/dev/null; wait $MPID 2>/dev/null

  sleep 3
done

echo ""
echo "=========================================================="
echo "  ALL DONE"
echo "=========================================================="
