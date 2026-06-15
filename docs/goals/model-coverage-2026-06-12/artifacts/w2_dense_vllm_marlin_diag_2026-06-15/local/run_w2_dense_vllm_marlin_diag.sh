#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ferrum-infer-rs

if [ -f /root/.cargo/env ]; then
  . /root/.cargo/env
fi
export PATH="/root/.cargo/bin:${PATH}"

OUT=/workspace/w2_dense_vllm_marlin_diag_2026-06-15
DATASET=/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
PORT=8467

mkdir -p "$OUT" "$OUT/build" "$OUT/remote" "$OUT/correctness" "$OUT/server" "$OUT/profile"
date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short_after_source_sync.txt" || true
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"
rustc --version > "$OUT/remote/rustc_version.txt"
cargo --version > "$OUT/remote/cargo_version.txt"
nvcc --version > "$OUT/remote/nvcc_version.txt"

cat > "$OUT/gpu_contract.md" <<TXT
lane: W2 Gemma3 CUDA dense vLLM Marlin first-fail diagnostic
expected_runtime_cost: 15-35min, hard cap 45min, reused Vast 40826362 1x RTX 4090 at about USD 0.425/hr
stop_condition: start/SSH/CUDA/source sync/build/server readiness, vLLM dense Marlin load, ferrum run smoke, or c16/c32 small sample first failure; diagnostic complete and copied; or 45min cap
correctness_gate: release build plus FERRUM_VLLM_MARLIN=1 ferrum run smoke plus server readiness plus bench-serve --fail-on-error zero-error diagnostic
performance_command: bench-serve sharegpt natural ASCII, FERRUM_VLLM_MARLIN=1, c16/c32, num_prompts=16, n_repeats=1, random-output-len=64, seed 9271, diagnostic only
TXT

cat > "$OUT/build/release_build.command.txt" <<TXT
CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
TXT
if ! CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$OUT/build/release_build.log" 2>&1; then
  echo FAIL > "$OUT/run.status"
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit 1
fi
sha256sum target/release/ferrum > "$OUT/build/ferrum.sha256"

cat > "$OUT/correctness/run.command.json" <<JSON
{"env":{"FERRUM_VLLM_MARLIN":"1"},"cmd":["timeout","1200","target/release/ferrum","run","gemma3:27b-gptq","--backend","cuda","--prompt","What is 2+3? Answer with just the number.","--max-tokens","64","--temperature","0","--kv-capacity","2560","--max-num-seqs","2","--output-format","jsonl","--effective-config-json","$OUT/correctness/run_effective_config.json","--decision-trace-jsonl","$OUT/correctness/run_decision_trace.jsonl"]}
JSON
set +e
FERRUM_VLLM_MARLIN=1 timeout 1200 target/release/ferrum run \
  gemma3:27b-gptq \
  --backend cuda \
  --prompt "What is 2+3? Answer with just the number." \
  --max-tokens 64 \
  --temperature 0 \
  --kv-capacity 2560 \
  --max-num-seqs 2 \
  --output-format jsonl \
  --effective-config-json "$OUT/correctness/run_effective_config.json" \
  --decision-trace-jsonl "$OUT/correctness/run_decision_trace.jsonl" \
  > "$OUT/correctness/run.stdout" 2> "$OUT/correctness/run.stderr"
RUN_RC=$?
set -e
echo "$RUN_RC" > "$OUT/correctness/run.rc"
if [ "$RUN_RC" -ne 0 ]; then
  echo FAIL > "$OUT/run.status"
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit "$RUN_RC"
fi

pkill -f "target/release/ferrum serve" 2>/dev/null || true
sleep 2

cat > "$OUT/server/serve.command.json" <<JSON
{"env":{"FERRUM_VLLM_MARLIN":"1"},"cmd":["target/release/ferrum","serve","--model","gemma3:27b-gptq","--port","$PORT","--kv-capacity","512","--max-num-seqs","16","--effective-config-json","$OUT/server/serve_effective_config.json","--decision-trace-jsonl","$OUT/server/serve_decision_trace.jsonl"]}
JSON

FERRUM_VLLM_MARLIN=1 target/release/ferrum serve \
  --model gemma3:27b-gptq \
  --port "$PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/server/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl" \
  > "$OUT/server/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server/server.pid"

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
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit 1
fi

cat > "$OUT/profile/bench-serve.command.json" <<JSON
{"cmd":["timeout","1800","target/release/ferrum","bench-serve","--base-url","http://127.0.0.1:$PORT","--model","gemma3:27b-gptq","--tokenizer","$TOKENIZER","--dataset","sharegpt","--sharegpt-path","$DATASET","--random-output-len","64","--concurrency-sweep","16,32","--num-prompts","16","--n-repeats","1","--fail-on-error","--seed","9271","--out","$OUT/profile/bench_sharegpt_c16_c32_16x1_dense_vllm_marlin.json"]}
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
  --out "$OUT/profile/bench_sharegpt_c16_c32_16x1_dense_vllm_marlin.json" \
  > "$OUT/profile/bench-serve.stdout" 2> "$OUT/profile/bench-serve.stderr"
BENCH_RC=$?
set -e
echo "$BENCH_RC" > "$OUT/profile/bench-serve.rc"
nvidia-smi > "$OUT/remote/nvidia_smi_after_bench.txt" || true

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

if [ "$BENCH_RC" -eq 0 ]; then
  echo PASS > "$OUT/run.status"
else
  echo FAIL > "$OUT/run.status"
fi
date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
exit "$BENCH_RC"
