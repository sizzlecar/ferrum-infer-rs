#!/bin/bash
# Master Group A benchmark suite for the macOS 2026-05-02 report.
#
# Runs 3 engines × 3 models × 4 concurrencies = 36 cells, capturing
# per-cell raw JSON (bench_serving.py format) + per-cell server log.
# Plus a top-level environment fingerprint captured at suite start.
#
# Engines:
#   ferrum (paged-batched on M ≥ 2; per-token on M=1; default behaviour
#           for FERRUM_METAL_PAGED_KV=1)
#   llama.cpp llama-server (--parallel = c)
#   mistralrs serve (--max-seqs = c)
#
# Models (Group A): Llama-3.1-8B / Qwen3-8B / Qwen3-30B-A3B (all Q4_K_M)
# Concurrencies: c=1, 4, 8, 16
#
# Bench harness: bench/scripts/bench_serving.py — vLLM benchmark_serving.py
# style, deterministic prompt round-robin, max_tokens=64, temperature=0.0,
# OpenAI /v1/chat/completions SSE.

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

MODELS=(
  "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|Meta-Llama-3.1-8B-Instruct.tokenizer.json|Llama-3.1-8B"
  "Qwen3-8B-Q4_K_M.gguf|Qwen3-8B.tokenizer.json|Qwen3-8B"
  "Qwen3-30B-A3B-Q4_K_M.gguf|Qwen3-30B-A3B.tokenizer.json|Qwen3-30B-A3B"
)

# Concurrency levels and their per-c (num_prompts:max_tokens) pairs.
CONFIGS=(
  "1:8:64"
  "4:16:64"
  "8:24:64"
  "16:32:64"
)

# Snapshot environment + memory + swap state at suite start.
{
  echo "===================================================="
  echo "macOS Group A bench suite — start: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "===================================================="
  echo ""
  echo "[hardware/os]"
  sw_vers
  echo ""
  echo "Model: $(sysctl -n hw.model)"
  echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
  echo "Cores (logical): $(sysctl -n hw.ncpu)"
  echo "Cores (physical): $(sysctl -n hw.physicalcpu)"
  echo "RAM: $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024)) GB"
  system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset|Total Number of Cores|Metal" | head
  echo ""
  echo "[software]"
  echo "ferrum: 0.7.0 @ $(git -C "$ROOT" log -1 --pretty='%h %s')"
  echo "llama.cpp (homebrew): $(brew list --versions llama.cpp 2>/dev/null | head -1)"
  echo "ggml (homebrew): $(brew list --versions ggml 2>/dev/null | head -1)"
  echo "mistralrs: $(mistralrs --version 2>&1 | head -1)"
  echo "Python: $(python3 --version)"
  echo ""
  echo "[memory state at suite start]"
  vm_stat | head -7
  sysctl vm.swapusage
  echo ""
  echo "[bench harness]"
  echo "Script: bench/scripts/bench_serving.py (vLLM benchmark_serving.py style)"
  echo "Deterministic prompts, temperature=0.0, max_tokens=64"
  echo "Per-c (num_prompts, max_tokens):"
  for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r c np mt <<< "$cfg"
    echo "  c=$c → num_prompts=$np, max_tokens=$mt"
  done
  echo ""
  echo "[ferrum env]"
  echo "FERRUM_METAL_PAGED_KV=1"
  echo "FERRUM_PAGED_MAX_SEQS=\$((c*2))    # 2× concurrency for pool headroom"
  echo "FERRUM_KV_CAPACITY=512             # per-seq cap, keeps pool ≈3.1 GB at MAX_SEQS=32"
  echo "FERRUM_MAX_BATCH=\$c"
  echo ""
  echo "[llama.cpp env]"
  echo "llama-server --ctx-size 4096 --parallel \$c --batch-size 2048 --jinja"
  echo ""
  echo "[mistralrs env]"
  echo "mistralrs serve --port \$port --max-seqs \$c text --format gguf -m \$GGUF_DIR -f \$gguf -t \$tokenizer"
  echo "Note: mistralrs HTTP API requires literal model='default' in request body"
} > "$OUTDIR/_suite_env.txt"
cat "$OUTDIR/_suite_env.txt"

run_ferrum() {
  local gguf="$1" model_label="$2" c="$3" num_prompts="$4" max_tokens="$5"
  pkill -9 -f "ferrum.*serve" 2>/dev/null; sleep 1
  local cell="ferrum__${model_label}__c${c}"
  echo ""
  echo "▶ $cell"
  FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS=$((c * 2)) FERRUM_KV_CAPACITY=512 FERRUM_MAX_BATCH=$c \
    "$FERRUM_BIN" serve --model "$gguf" --port "$PORT_FERRUM" \
    > "$OUTDIR/${cell}.server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT_FERRUM/v1/models" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT_FERRUM/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_label\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT_FERRUM" \
    --model "$model_label" \
    --num-prompts $num_prompts --max-concurrency $c --max-tokens $max_tokens \
    --deterministic-prompts \
    --result-file "$OUTDIR/${cell}.json" \
    > "$OUTDIR/${cell}.bench.log" 2>&1
  python3 -c "
import json
try:
    d = json.load(open('$OUTDIR/${cell}.json'))
    f = lambda x: f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
    ttft=d.get('ttft_ms',{}); tpot=d.get('tpot_ms',{})
    print(f'  out_tok/s={f(d.get(\"output_throughput_tok_s\"))}  TPOT_med={f(tpot.get(\"median\"))}  TTFT_med={f(ttft.get(\"median\"))}')
except Exception as e:
    print(f'  ERROR: {e}')
" 2>/dev/null
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}

run_llamacpp() {
  local gguf="$1" model_label="$2" c="$3" num_prompts="$4" max_tokens="$5"
  pkill -9 -f "llama-server" 2>/dev/null; sleep 1
  local cell="llamacpp__${model_label}__c${c}"
  echo ""
  echo "▶ $cell"
  llama-server --model "$gguf" --port "$PORT_LLAMACPP" \
    --ctx-size 4096 --parallel $c --batch-size 2048 --jinja \
    > "$OUTDIR/${cell}.server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT_LLAMACPP/v1/models" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT_LLAMACPP/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_label\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT_LLAMACPP" \
    --model "$model_label" \
    --num-prompts $num_prompts --max-concurrency $c --max-tokens $max_tokens \
    --deterministic-prompts \
    --result-file "$OUTDIR/${cell}.json" \
    > "$OUTDIR/${cell}.bench.log" 2>&1
  python3 -c "
import json
try:
    d = json.load(open('$OUTDIR/${cell}.json'))
    f = lambda x: f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
    ttft=d.get('ttft_ms',{}); tpot=d.get('tpot_ms',{})
    print(f'  out_tok/s={f(d.get(\"output_throughput_tok_s\"))}  TPOT_med={f(tpot.get(\"median\"))}  TTFT_med={f(ttft.get(\"median\"))}')
except Exception as e:
    print(f'  ERROR: {e}')
" 2>/dev/null
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}

run_mistralrs() {
  local gguf_name="$1" tok_name="$2" model_label="$3" c="$4" num_prompts="$5" max_tokens="$6"
  pkill -9 -f "mistralrs.*serve" 2>/dev/null; sleep 1
  local cell="mistralrs__${model_label}__c${c}"
  echo ""
  echo "▶ $cell"
  mistralrs serve --port "$PORT_MISTRALRS" --max-seqs $c text \
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
  # mistralrs requires literal model='default' in request body.
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT_MISTRALRS/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT_MISTRALRS" \
    --model "default" \
    --num-prompts $num_prompts --max-concurrency $c --max-tokens $max_tokens \
    --deterministic-prompts \
    --result-file "$OUTDIR/${cell}.json" \
    > "$OUTDIR/${cell}.bench.log" 2>&1
  python3 -c "
import json
try:
    d = json.load(open('$OUTDIR/${cell}.json'))
    f = lambda x: f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
    ttft=d.get('ttft_ms',{}); tpot=d.get('tpot_ms',{})
    print(f'  out_tok/s={f(d.get(\"output_throughput_tok_s\"))}  TPOT_med={f(tpot.get(\"median\"))}  TTFT_med={f(ttft.get(\"median\"))}')
except Exception as e:
    print(f'  ERROR: {e}')
" 2>/dev/null
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}

# ── Run the full grid ──────────────────────────────────────────────────
for entry in "${MODELS[@]}"; do
  IFS='|' read -r gguf_name tok_name model_label <<< "$entry"
  GGUF="$GGUF_DIR/$gguf_name"
  if [ ! -f "$GGUF" ]; then
    echo "SKIP (missing gguf): $GGUF" >&2
    continue
  fi

  echo ""
  echo "================================================================"
  echo "  $model_label"
  echo "================================================================"

  # Prewarm page cache once per model so cold-load doesn't pollute
  # the first cell of the model.
  echo "prewarming $gguf_name page cache..."
  cat "$GGUF" > /dev/null

  for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r c num_prompts max_tokens <<< "$cfg"
    run_ferrum    "$GGUF" "$model_label" "$c" "$num_prompts" "$max_tokens"
    run_llamacpp  "$GGUF" "$model_label" "$c" "$num_prompts" "$max_tokens"
    run_mistralrs "$gguf_name" "$tok_name" "$model_label" "$c" "$num_prompts" "$max_tokens"
  done
done

# Memory state at suite end.
{
  echo ""
  echo "===================================================="
  echo "macOS Group A bench suite — end:   $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "===================================================="
  echo "[memory state at suite end]"
  vm_stat | head -7
  sysctl vm.swapusage
} >> "$OUTDIR/_suite_env.txt"
cat "$OUTDIR/_suite_env.txt" | tail -10
echo ""
echo "All cells written under $OUTDIR/"
