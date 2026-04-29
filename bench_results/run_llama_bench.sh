#!/usr/bin/env bash
set -uo pipefail
MODELS_DIR="$HOME/ferrum-bench/models"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"

run() {
  local label="$1"; local file="$2"
  echo "=== $label ===" | tee -a "$OUT_DIR/bench_runner.log"
  date "+%Y-%m-%d %H:%M:%S" | tee -a "$OUT_DIR/bench_runner.log"
  /opt/homebrew/bin/llama-bench \
    -m "$MODELS_DIR/$file" \
    -p 512 -n 128 -t 8 -ngl 99 -r 5 \
    -o md \
    > "$OUT_DIR/llamacpp_$label.md" \
    2> "$OUT_DIR/llamacpp_$label.stderr.log"
  echo "exit=$?" | tee -a "$OUT_DIR/bench_runner.log"
  cat "$OUT_DIR/llamacpp_$label.md" | tee -a "$OUT_DIR/bench_runner.log"
  echo "" | tee -a "$OUT_DIR/bench_runner.log"
}

run "qwen3_8b"        "Qwen3-8B-Q4_K_M.gguf"
run "llama31_8b"      "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
run "qwen3_30b_a3b"   "Qwen3-30B-A3B-Q4_K_M.gguf"
echo "ALL DONE" | tee -a "$OUT_DIR/bench_runner.log"
