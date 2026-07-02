#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ferrum-infer-rs

if [ -f /root/.cargo/env ]; then
  . /root/.cargo/env
fi
export PATH="/root/.cargo/bin:${PATH}"

OUT=/workspace/w2_vllm_sharegpt_baseline_probe_2026-06-15
MODEL_PATH=/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
TOKENIZER="$MODEL_PATH"
DATASET=/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
VLLM_PY=/workspace/vllm-venv-0101-cu126/bin/python
PORT=8405

mkdir -p "$OUT/env" "$OUT/server" "$OUT/smoke" "$OUT/bench" "$OUT/remote"
date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short.txt" || true
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"
rustc --version > "$OUT/remote/rustc_version.txt" || true
cargo --version > "$OUT/remote/cargo_version.txt" || true
nvcc --version > "$OUT/remote/nvcc_version.txt" || true

cat > "$OUT/gpu_contract.md" <<'TXT'
lane: W2 Gemma3 CUDA vLLM ShareGPT baseline-cleanliness probe
instance: Vast 40826362, 1x RTX 4090, cache-retained native CUDA machine
expected_runtime_cost: 20-45min, hard cap 60min, reused instance at about USD 0.425/hr when running
stop_condition: start/SSH/CUDA/vLLM server first failure, baseline smoke failure, c16/c32 ShareGPT diagnostic complete and artifacts copied, or 60min hard cap
correctness_gate: vLLM OpenAI /v1/models plus non-stream chat smoke before bench-serve
performance_command: diagnostic-only natural ASCII ShareGPT c16/c32 with bench-serve --fail-on-error, seed 9271, n_repeats=1, num_prompts=16
release_grade_status: not release-grade evidence; this only tests whether vLLM is baseline-clean on natural prompts after random-prompt invalid-UTF8 failures
TXT

if [ ! -x "$VLLM_PY" ]; then
  echo "missing_vllm_venv" > "$OUT/run.status"
  exit 1
fi
if [ ! -f "$DATASET" ]; then
  echo "missing_dataset" > "$OUT/run.status"
  exit 1
fi
if [ ! -x target/release/ferrum ]; then
  echo "missing_ferrum_binary" > "$OUT/run.status"
  exit 1
fi

"$VLLM_PY" - <<'PY' > "$OUT/env/vllm_version.txt" 2>&1
import vllm
print(vllm.__version__)
PY
"$VLLM_PY" -m pip freeze > "$OUT/env/vllm_pip_freeze.txt" 2>&1 || true

pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -f "target/release/ferrum serve" 2>/dev/null || true
sleep 3

cat > "$OUT/server/vllm_server.command.json" <<JSON
{"cmd":["$VLLM_PY","-m","vllm.entrypoints.openai.api_server","--host","127.0.0.1","--port","$PORT","--model","$MODEL_PATH","--served-model-name","gemma3:27b-gptq","--max-model-len","512","--max-num-seqs","16","--gpu-memory-utilization","0.92"],"dependency_pins":{"transformers":"4.55.4","fastapi":"0.116.1","starlette":"0.47.2","prometheus-fastapi-instrumentator":"7.1.0"}}
JSON
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model "$MODEL_PATH" \
  --served-model-name gemma3:27b-gptq \
  --max-model-len 512 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.92 \
  > "$OUT/server/vllm_server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server/vllm_server.pid"

cleanup_server() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup_server EXIT

ready=0
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
    > "$OUT/server/v1_models.json" 2> "$OUT/server/v1_models.err"; then
    ready=1
    echo "$i" > "$OUT/server/ready_poll.txt"
    break
  fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  echo "server_not_ready" > "$OUT/run.status"
  tail -n 200 "$OUT/server/vllm_server.log" > "$OUT/server/vllm_server_tail_on_fail.log" || true
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit 1
fi

cat > "$OUT/smoke/chat_request.json" <<'JSON'
{
  "model": "gemma3:27b-gptq",
  "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
  "max_tokens": 16,
  "temperature": 0
}
JSON
if ! curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @"$OUT/smoke/chat_request.json" \
  > "$OUT/smoke/chat_response.json" 2> "$OUT/smoke/chat_response.err"; then
  echo "chat_smoke_failed" > "$OUT/run.status"
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit 1
fi
python3 - <<PY
import json, pathlib, sys
resp = json.loads(pathlib.Path("$OUT/smoke/chat_response.json").read_text())
content = resp["choices"][0]["message"].get("content") or ""
usage = resp.get("usage") or {}
if not content.strip():
    raise SystemExit("empty smoke content")
if int(usage.get("completion_tokens") or 0) <= 0:
    raise SystemExit("missing completion usage")
PY

cat > "$OUT/bench/bench-serve.command.json" <<JSON
{"cmd":["timeout","1800","target/release/ferrum","bench-serve","--base-url","http://127.0.0.1:$PORT","--model","gemma3:27b-gptq","--tokenizer","$TOKENIZER","--dataset","sharegpt","--sharegpt-path","$DATASET","--random-output-len","64","--concurrency-sweep","16,32","--num-prompts","16","--n-repeats","1","--fail-on-error","--seed","9271","--out","$OUT/bench/bench_vllm_sharegpt_c16_c32_16x1.json"],"engine":"vllm","version_file":"$OUT/env/vllm_version.txt"}
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
  --out "$OUT/bench/bench_vllm_sharegpt_c16_c32_16x1.json" \
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
