#!/usr/bin/env bash
set -euo pipefail

if [ -f /root/.cargo/env ]; then
  # shellcheck disable=SC1091
  . /root/.cargo/env
fi
export PATH="/root/.cargo/bin:${PATH}"
export CARGO_TARGET_DIR=/workspace/ferrum-infer-rs/target

REPO=/workspace/ferrum-w2-prompt-token-admission
OUT=/workspace/w2_prompt_token_admission_c16_ab_2026-06-15
BIN="${CARGO_TARGET_DIR}/release/ferrum"
DATASET=/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
PORT=8497

cd "$REPO"
mkdir -p "$OUT"/{bench,build,correctness,remote,server,smoke}

date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short.txt"
rustc --version > "$OUT/remote/rustc_version.txt"
cargo --version > "$OUT/remote/cargo_version.txt"
nvcc --version > "$OUT/remote/nvcc_version.txt"
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"

printf '%q ' cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source > "$OUT/build/release_build.command.txt"
printf '\n' >> "$OUT/build/release_build.command.txt"
set +e
cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$OUT/build/release_build.stdout" 2> "$OUT/build/release_build.stderr"
BUILD_RC=$?
set -e
echo "$BUILD_RC" > "$OUT/build/release_build.rc"
if [ "$BUILD_RC" -ne 0 ]; then
  echo FAIL > "$OUT/run.status"
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit "$BUILD_RC"
fi
sha256sum "$BIN" > "$OUT/build/ferrum.sha256"

cat > "$OUT/correctness/run.command.json" <<JSON
{"cmd":["timeout","1200","$BIN","run","gemma3:27b-gptq","--backend","cuda","--prompt","What is 2+3? Answer with just the number.","--max-tokens","64","--temperature","0","--kv-capacity","2560","--max-num-seqs","2","--output-format","jsonl","--effective-config-json","$OUT/correctness/run_effective_config.json","--decision-trace-jsonl","$OUT/correctness/run_decision_trace.jsonl"]}
JSON
set +e
timeout 1200 "$BIN" run gemma3:27b-gptq \
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
if [ "$RUN_RC" -ne 0 ] || ! grep -q '"content":"5"' "$OUT/correctness/run.stdout"; then
  echo FAIL > "$OUT/run.status"
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit 1
fi

pkill -f "ferrum serve" 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 3

cat > "$OUT/server/serve.command.json" <<JSON
{"cmd":["$BIN","serve","--model","gemma3:27b-gptq","--port","$PORT","--kv-capacity","512","--max-num-seqs","16","--effective-config-json","$OUT/server/serve_effective_config.json","--decision-trace-jsonl","$OUT/server/serve_decision_trace.jsonl"]}
JSON
"$BIN" serve \
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

cat > "$OUT/smoke/chat_request.json" <<'JSON'
{"model":"gemma3:27b-gptq","messages":[{"role":"user","content":"What is 2+3? Answer with just the number."}],"max_tokens":8,"temperature":0,"stream":false}
JSON
set +e
curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$OUT/smoke/chat_request.json" \
  > "$OUT/smoke/chat_response.json" 2> "$OUT/smoke/chat_response.err"
SMOKE_RC=$?
set -e
echo "$SMOKE_RC" > "$OUT/smoke/chat_response.rc"
python3 - "$OUT/smoke/chat_response.json" > "$OUT/smoke/chat_validation.json" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path))
content = data["choices"][0]["message"].get("content", "")
usage = data.get("usage") or {}
ok = bool(content.strip()) and usage.get("prompt_tokens", 0) > 0 and usage.get("completion_tokens", 0) > 0
print(json.dumps({"ok": ok, "content": content, "usage": usage}, ensure_ascii=False))
sys.exit(0 if ok else 1)
PY

cat > "$OUT/bench/bench-serve.command.json" <<JSON
{"cmd":["timeout","1200","$BIN","bench-serve","--base-url","http://127.0.0.1:$PORT","--model","gemma3:27b-gptq","--tokenizer","$TOKENIZER","--dataset","sharegpt","--sharegpt-path","$DATASET","--random-output-len","64","--concurrency-sweep","16","--num-prompts","16","--n-repeats","1","--fail-on-error","--seed","9271","--out","$OUT/bench/bench_ferrum_prompt_token_admission_c16_16x1.json"]}
JSON
set +e
timeout 1200 "$BIN" bench-serve \
  --base-url "http://127.0.0.1:${PORT}" \
  --model gemma3:27b-gptq \
  --tokenizer "$TOKENIZER" \
  --dataset sharegpt \
  --sharegpt-path "$DATASET" \
  --random-output-len 64 \
  --concurrency-sweep 16 \
  --num-prompts 16 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271 \
  --out "$OUT/bench/bench_ferrum_prompt_token_admission_c16_16x1.json" \
  > "$OUT/bench/bench-serve.stdout" 2> "$OUT/bench/bench-serve.stderr"
BENCH_RC=$?
set -e
echo "$BENCH_RC" > "$OUT/bench/bench-serve.rc"
nvidia-smi > "$OUT/remote/nvidia_smi_after_bench.txt" || true

python3 - "$OUT/bench/bench_ferrum_prompt_token_admission_c16_16x1.json" > "$OUT/summary.json" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1]))
row = rows[0]
summary = {
    "concurrency": row["concurrency"],
    "completed": row["completed_per_run"],
    "errored": row["errored_per_run"],
    "bad_output": row["bad_output_per_run"],
    "output_throughput_tps": row["output_throughput_tps"]["mean"],
    "request_throughput_rps": row["request_throughput_rps"]["mean"],
    "ttft_p50_ms": row["ttft_ms"]["p50"]["mean"],
    "tpot_p50_ms": row["tpot_ms"]["p50"]["mean"],
    "output_token_count_source": row["output_token_count_source"],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if [ "$BENCH_RC" -eq 0 ]; then
  echo PASS > "$OUT/run.status"
else
  echo FAIL > "$OUT/run.status"
fi
date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
exit "$BENCH_RC"
