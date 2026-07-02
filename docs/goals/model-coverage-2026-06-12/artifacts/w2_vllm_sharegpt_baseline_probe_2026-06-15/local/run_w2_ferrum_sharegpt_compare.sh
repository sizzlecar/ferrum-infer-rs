#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ferrum-infer-rs

if [ -f /root/.cargo/env ]; then
  . /root/.cargo/env
fi
export PATH="/root/.cargo/bin:${PATH}"

OUT=/workspace/w2_vllm_sharegpt_baseline_probe_2026-06-15/ferrum_compare
DATASET=/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
PORT=8475

mkdir -p "$OUT/server" "$OUT/bench" "$OUT/remote"
date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short.txt" || true
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"
sha256sum target/release/ferrum > "$OUT/remote/ferrum.sha256"

pkill -f "target/release/ferrum serve" 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 3

cat > "$OUT/server/serve.command.json" <<JSON
{"cmd":["target/release/ferrum","serve","--model","gemma3:27b-gptq","--port","$PORT","--kv-capacity","512","--max-num-seqs","16","--effective-config-json","$OUT/server/serve_effective_config.json","--decision-trace-jsonl","$OUT/server/serve_decision_trace.jsonl"]}
JSON
target/release/ferrum serve \
  --model gemma3:27b-gptq \
  --port "$PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/server/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl" \
  > "$OUT/server/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server/server.pid"

cleanup_server() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup_server EXIT

ready=0
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
    > "$OUT/server/models.json" 2> "$OUT/server/models.err"; then
    ready=1
    echo "$i" > "$OUT/server/ready_poll.txt"
    break
  fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  echo FAIL > "$OUT/run.status"
  tail -n 200 "$OUT/server/server.log" > "$OUT/server/server_tail_on_fail.log" || true
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit 1
fi

cat > "$OUT/bench/bench-serve.command.json" <<JSON
{"cmd":["timeout","1800","target/release/ferrum","bench-serve","--base-url","http://127.0.0.1:$PORT","--model","gemma3:27b-gptq","--tokenizer","$TOKENIZER","--dataset","sharegpt","--sharegpt-path","$DATASET","--random-output-len","64","--concurrency-sweep","16,32","--num-prompts","16","--n-repeats","1","--fail-on-error","--seed","9271","--out","$OUT/bench/bench_ferrum_sharegpt_c16_c32_16x1.json"]}
JSON
set +e
timeout 1800 target/release/ferrum bench-serve \
  --base-url "http://127.0.0.1:${PORT}" \
  --model gemma3:27b-gptq \
  --tokenizer "$TOKENIZER" \
  --dataset sharegpt \
  --sharegpt-path "$DATASET" \
  --random-output-len 64 \
  --concurrency-sweep 16,32 \
  --num-prompts 16 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271 \
  --out "$OUT/bench/bench_ferrum_sharegpt_c16_c32_16x1.json" \
  > "$OUT/bench/bench-serve.stdout" 2> "$OUT/bench/bench-serve.stderr"
BENCH_RC=$?
set -e
echo "$BENCH_RC" > "$OUT/bench/bench-serve.rc"
nvidia-smi > "$OUT/remote/nvidia_smi_after_bench.txt" || true

if [ "$BENCH_RC" -eq 0 ]; then
  echo PASS > "$OUT/run.status"
else
  echo FAIL > "$OUT/run.status"
fi
date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
exit "$BENCH_RC"
