#!/bin/bash
# Re-run only the c=16 row of the Group A suite with a clean system.
#
# Why: the original suite ran 36 cells back-to-back. By the time the
# Qwen3-30B-A3B row started, vm.swapusage had climbed to ~7.5 GB on the
# 32 GB Mac (mmap'd GGUFs across three engines + lingering allocations),
# which depressed ferrum's MoE c=16 number from its true ~80 tok/s to
# ~48 tok/s. Same suite, same env, just memory pressure.
#
# This script re-runs only the c=16 row of all three models with a
# pkill + cooldown between every cell so each engine starts fresh.

set -u

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FERRUM_BIN="${FERRUM_BIN:-$ROOT/target/release/ferrum}"
GGUF_DIR="${GGUF_DIR:-/Users/chejinxuan/ferrum-bench/models}"
TOK_DIR="${TOK_DIR:-/Users/chejinxuan/ferrum-bench/tokenizers}"
BENCH_PY="$ROOT/bench/scripts/bench_serving.py"
OUTDIR="$(cd "$(dirname "$0")" && pwd)"

PORT_FERRUM=8783
PORT_LLAMACPP=8001
PORT_MISTRALRS=8002

C=16
NUM_PROMPTS=32
MAX_TOKENS=64
COOLDOWN=15

MODELS=(
  "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|Meta-Llama-3.1-8B-Instruct.tokenizer.json|Llama-3.1-8B"
  "Qwen3-8B-Q4_K_M.gguf|Qwen3-8B.tokenizer.json|Qwen3-8B"
  "Qwen3-30B-A3B-Q4_K_M.gguf|Qwen3-30B-A3B.tokenizer.json|Qwen3-30B-A3B"
)

cooldown() {
  pkill -9 -f "ferrum.*serve|llama-server|mistralrs.*serve" 2>/dev/null
  sleep "$COOLDOWN"
  echo "[mem] $(sysctl -n vm.swapusage | awk '{print $4, $5, $6}') | $(vm_stat | awk '/Pages free/ {print "free="$3}')"
}

run_ferrum() {
  local gguf="$1" model_label="$2"
  cooldown
  local cell="rerun_ferrum__${model_label}__c${C}"
  echo ""
  echo "▶ $cell"
  FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS=$((C * 2)) FERRUM_KV_CAPACITY=512 FERRUM_MAX_BATCH=$C \
    "$FERRUM_BIN" serve --model "$gguf" --port "$PORT_FERRUM" \
    > "$OUTDIR/${cell}.server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT_FERRUM/v1/models" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT_FERRUM/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_label\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT_FERRUM" \
    --model "$model_label" \
    --num-prompts $NUM_PROMPTS --max-concurrency $C --max-tokens $MAX_TOKENS \
    --deterministic-prompts \
    --result-file "$OUTDIR/${cell}.json" \
    > "$OUTDIR/${cell}.bench.log" 2>&1
  python3 -c "
import json
try:
    d = json.load(open('$OUTDIR/${cell}.json'))
    f = lambda x: f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
    tpot=d.get('tpot_ms',{})
    print(f'  out_tok/s={f(d.get(\"output_throughput_tok_s\"))}  TPOT_med={f(tpot.get(\"median\"))}')
except Exception as e:
    print(f'  ERROR: {e}')
" 2>/dev/null
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}

run_llamacpp() {
  local gguf="$1" model_label="$2"
  cooldown
  local cell="rerun_llamacpp__${model_label}__c${C}"
  echo ""
  echo "▶ $cell"
  llama-server --model "$gguf" --port "$PORT_LLAMACPP" \
    --ctx-size 4096 --parallel $C --batch-size 2048 --jinja \
    > "$OUTDIR/${cell}.server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT_LLAMACPP/v1/models" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT_LLAMACPP/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_label\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT_LLAMACPP" \
    --model "$model_label" \
    --num-prompts $NUM_PROMPTS --max-concurrency $C --max-tokens $MAX_TOKENS \
    --deterministic-prompts \
    --result-file "$OUTDIR/${cell}.json" \
    > "$OUTDIR/${cell}.bench.log" 2>&1
  python3 -c "
import json
try:
    d = json.load(open('$OUTDIR/${cell}.json'))
    f = lambda x: f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
    tpot=d.get('tpot_ms',{})
    print(f'  out_tok/s={f(d.get(\"output_throughput_tok_s\"))}  TPOT_med={f(tpot.get(\"median\"))}')
except Exception as e:
    print(f'  ERROR: {e}')
" 2>/dev/null
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}

run_mistralrs() {
  local gguf_name="$1" tok_name="$2" model_label="$3"
  cooldown
  local cell="rerun_mistralrs__${model_label}__c${C}"
  echo ""
  echo "▶ $cell"
  mistralrs serve --port "$PORT_MISTRALRS" --max-seqs $C text \
    --format gguf -m "$GGUF_DIR" -f "$gguf_name" -t "$TOK_DIR/$tok_name" \
    > "$OUTDIR/${cell}.server.log" 2>&1 &
  local pid=$!
  local came_up=0
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT_MISTRALRS/v1/models" >/dev/null 2>&1; then
      came_up=1; break
    fi
    sleep 1
  done
  if [ "$came_up" = "0" ]; then
    echo "  FAILED to come up"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    return
  fi
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT_MISTRALRS/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT_MISTRALRS" \
    --model "default" \
    --num-prompts $NUM_PROMPTS --max-concurrency $C --max-tokens $MAX_TOKENS \
    --deterministic-prompts \
    --result-file "$OUTDIR/${cell}.json" \
    > "$OUTDIR/${cell}.bench.log" 2>&1
  python3 -c "
import json
try:
    d = json.load(open('$OUTDIR/${cell}.json'))
    f = lambda x: f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
    tpot=d.get('tpot_ms',{})
    print(f'  out_tok/s={f(d.get(\"output_throughput_tok_s\"))}  TPOT_med={f(tpot.get(\"median\"))}')
except Exception as e:
    print(f'  ERROR: {e}')
" 2>/dev/null
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}

# Run only c=16 for each model, with cooldown between every engine.
for entry in "${MODELS[@]}"; do
  IFS='|' read -r gguf_name tok_name model_label <<< "$entry"
  GGUF="$GGUF_DIR/$gguf_name"
  if [ ! -f "$GGUF" ]; then continue; fi
  echo ""
  echo "================================================================"
  echo "  $model_label (rerun, clean state)"
  echo "================================================================"
  cat "$GGUF" > /dev/null  # prewarm page cache
  run_ferrum    "$GGUF" "$model_label"
  run_llamacpp  "$GGUF" "$model_label"
  run_mistralrs "$gguf_name" "$tok_name" "$model_label"
done

echo ""
echo "[done] $(date '+%Y-%m-%d %H:%M:%S')"
sysctl vm.swapusage
