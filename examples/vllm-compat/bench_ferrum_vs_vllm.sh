#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
MODEL="${2:-Qwen/Qwen3-0.6B}"
TOKENIZER="${3:?usage: $0 <base-url> <model> <tokenizer-dir>}"

ferrum bench-serve \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --dataset random \
  --random-input-len 256 \
  --random-output-len 128 \
  --num-prompts 64 \
  --warmup-requests 8 \
  --concurrency 8 \
  --n-repeats 3 \
  --fail-on-error
