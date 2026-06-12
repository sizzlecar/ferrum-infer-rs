#!/usr/bin/env bash
# W1 L5 batch: per verified model — mirror-pull, serve, bench-serve
# c=1/4/16 (n=3, random 256/128) with an --out report, then delete the
# weights to keep the disk inside its budget. 30B-class models add c=32
# (run separately). See docs/goals/model-coverage-2026-06-12/GOAL.md L5.
set -u
cd "$(dirname "$0")/.."
BIN=target/release/ferrum
OUTDIR=docs/goals/model-coverage-2026-06-12/artifacts
PORT=18230

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
unset HTTPS_PROXY https_proxy HTTP_PROXY http_proxy ALL_PROXY all_proxy

# alias | cache repo dir | gguf filename | report id
MODELS=(
  "mistral-small:24b-q4_k_m|models--bartowski--mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF|mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf|mistral-small-24b"
  "magistral:24b-q4_k_m|models--bartowski--mistralai_Magistral-Small-2509-GGUF|mistralai_Magistral-Small-2509-Q4_K_M.gguf|magistral-24b"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r alias repo file rid <<<"$entry"
  echo "=== [$rid] pull $(date +%H:%M:%S)"
  ok=0
  for i in 1 2 3 4 5; do
    "$BIN" pull "$alias" >/dev/null 2>&1
    if ls ~/.cache/huggingface/hub/"$repo"/snapshots/*/"$file" >/dev/null 2>&1; then ok=1; break; fi
    sleep 5
  done
  if [ "$ok" != "1" ]; then echo "=== [$rid] PULL FAILED, skipping"; continue; fi

  TOKDIR=$(dirname "$(ls ~/.cache/huggingface/hub/"$repo"/snapshots/*/tokenizer.json | head -1)")
  PORT=$((PORT + 1))
  echo "=== [$rid] serve on $PORT + bench $(date +%H:%M:%S)"
  pkill -f "ferrum serve" 2>/dev/null; sleep 1
  "$BIN" serve "$alias" --port "$PORT" --kv-capacity 4096 --max-num-seqs 4 >"/tmp/ferrum_l5_${rid}.log" 2>&1 &
  SRV=$!
  healthy=0
  for i in $(seq 1 200); do
    if curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then healthy=1; break; fi
    if ! kill -0 "$SRV" 2>/dev/null; then break; fi
    sleep 3
  done
  if [ "$healthy" != "1" ]; then
    echo "=== [$rid] SERVER NOT HEALTHY, skipping"; kill "$SRV" 2>/dev/null; continue
  fi
  "$BIN" bench-serve --base-url "http://127.0.0.1:$PORT" --model "$alias" \
    --tokenizer "$TOKDIR" --concurrency-sweep 1,4,16 --num-prompts 100 \
    --n-repeats 3 --out "$OUTDIR/l5_${rid}_metal_2026-06-12.json" 2>&1 | tail -2
  kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
  rm -rf ~/.cache/huggingface/hub/"$repo"
  echo "=== [$rid] DONE $(date +%H:%M:%S)"
done
echo "=== L5 BATCH COMPLETE"
