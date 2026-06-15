#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/ferrum-infer-rs
FERRUM="$REPO/target/release/ferrum"
OUT=/workspace/w2_vllm_paged_attn_gemma_diag_2026-06-15
DATASET=/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
PORT=8477

cd "$REPO"
if [ -f /root/.cargo/env ]; then
  . /root/.cargo/env
fi
export PATH="/root/.cargo/bin:${PATH}"

mkdir -p "$OUT/server" "$OUT/smoke" "$OUT/bench" "$OUT/remote" "$OUT/config"
date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short.txt" || true
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"
sha256sum "$FERRUM" > "$OUT/remote/ferrum.sha256"

cat > "$OUT/gpu_contract.md" <<'TXT'
lane: W2 Gemma3 CUDA typed vLLM paged-attn ShareGPT diagnostic
instance: Vast 40826362, 1x RTX 4090, cache-retained native CUDA machine
expected_runtime_cost: 10-25min, hard cap 35min, reused instance at about USD 0.425/hr when running
stop_condition: start/SSH/CUDA/server readiness first failure, typed attention-selection assertion failure, chat smoke failure, c16/c32 ShareGPT diagnostic complete and artifacts copied, or 35min hard cap
correctness_gate: ferrum serve from an artifact-local ferrum.toml with runtime.use_vllm_paged_attn=true, readiness, decision-trace assertion, and non-stream chat smoke before bench-serve
performance_command: diagnostic-only natural ASCII ShareGPT c16/c32 with bench-serve --fail-on-error, seed 9271, n_repeats=1, num_prompts=16
baseline_engine_version_build: vLLM 0.10.1.1 CUDA12 baseline from w2_vllm_sharegpt_baseline_probe_2026-06-15; same RTX 4090, same HF/safetensors GPTQ model, same ShareGPT dataset
release_grade_status: not release-grade evidence; tests whether a typed product config for vLLM paged attention closes the clean vLLM ShareGPT gap
TXT

cat > "$OUT/ferrum.toml" <<'TOML'
[client]
base_url = "http://localhost:8080"
timeout_seconds = 300

[client.retry]
max_attempts = 3
initial_delay_ms = 100
max_delay_ms = 5000
backoff_multiplier = 2.0

[models]
model_dir = "./models"
cache_dir = "./model_cache"
default_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

[models.aliases]

[models.download]
hf_cache_dir = "/root/.cache/huggingface"
timeout_seconds = 300
max_concurrent = 4
retry_attempts = 3

[benchmark]
num_requests = 100
concurrency = 10
prompt_length = 512
max_tokens = 256
warmup_requests = 10
output_dir = "./benchmark_results"

[server]
host = "127.0.0.1"
port = 8000
config_path = "server.toml"
log_level = "info"
hot_reload = false

[dev]
debug = false
profile_memory = false
profile_gpu = false
mock_backends = false
test_data_dir = "./test_data"

[runtime]
use_vllm_paged_attn = true
TOML
cp "$OUT/ferrum.toml" "$OUT/config/ferrum.toml"

pkill -f "target/release/ferrum serve" 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 3

cat > "$OUT/server/serve.command.json" <<JSON
{"cwd":"$OUT","cmd":["$FERRUM","serve","--model","gemma3:27b-gptq","--port","$PORT","--kv-capacity","512","--max-num-seqs","16","--effective-config-json","$OUT/server/serve_effective_config.json","--decision-trace-jsonl","$OUT/server/serve_decision_trace.jsonl"]}
JSON
(
  cd "$OUT"
  "$FERRUM" serve \
    --model gemma3:27b-gptq \
    --port "$PORT" \
    --kv-capacity 512 \
    --max-num-seqs 16 \
    --effective-config-json "$OUT/server/serve_effective_config.json" \
    --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl"
) > "$OUT/server/server.log" 2>&1 &
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

python3 - <<PY
import json, pathlib
cfg_path = pathlib.Path("$OUT/server/serve_effective_config.json")
cfg = json.loads(cfg_path.read_text())
decisions = {item.get("selection"): item for item in cfg.get("decisions", [])}
prefill = decisions.get("attention_prefill_mixed_backend", {}).get("selected")
decode = decisions.get("attention_decode_backend", {}).get("selected")
if prefill != "vllm_paged_varlen":
    raise SystemExit(f"unexpected prefill backend: {prefill}")
if decode not in {"vllm_paged_attn_v1_short", "vllm_paged_attn_v2"}:
    raise SystemExit(f"unexpected decode backend: {decode}")
entries = {item.get("key"): item for item in cfg.get("entries", [])}
if entries.get("FERRUM_USE_VLLM_PAGED_ATTN", {}).get("effective_value") != "1":
    raise SystemExit("FERRUM_USE_VLLM_PAGED_ATTN not materialized as 1")
pathlib.Path("$OUT/server/attention_selection.status").write_text(
    f"PASS prefill={prefill} decode={decode}\\n"
)
PY

curl -fsS "http://127.0.0.1:${PORT}/health" \
  > "$OUT/server/health_before.json" 2> "$OUT/server/health_before.err" || true

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
  echo FAIL > "$OUT/run.status"
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit 1
fi
python3 - <<PY
import json, pathlib
resp = json.loads(pathlib.Path("$OUT/smoke/chat_response.json").read_text())
content = resp["choices"][0]["message"].get("content") or ""
usage = resp.get("usage") or {}
if not content.strip():
    raise SystemExit("empty smoke content")
if int(usage.get("completion_tokens") or 0) <= 0:
    raise SystemExit("missing completion usage")
PY

cat > "$OUT/bench/bench-serve.command.json" <<JSON
{"cmd":["timeout","1800","$FERRUM","bench-serve","--base-url","http://127.0.0.1:$PORT","--model","gemma3:27b-gptq","--tokenizer","$TOKENIZER","--dataset","sharegpt","--sharegpt-path","$DATASET","--random-output-len","64","--concurrency-sweep","16,32","--num-prompts","16","--n-repeats","1","--fail-on-error","--seed","9271","--out","$OUT/bench/bench_ferrum_vllm_paged_attn_sharegpt_c16_c32_16x1.json"]}
JSON
set +e
timeout 1800 "$FERRUM" bench-serve \
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
  --out "$OUT/bench/bench_ferrum_vllm_paged_attn_sharegpt_c16_c32_16x1.json" \
  > "$OUT/bench/bench-serve.stdout" 2> "$OUT/bench/bench-serve.stderr"
BENCH_RC=$?
set -e
echo "$BENCH_RC" > "$OUT/bench/bench-serve.rc"
curl -fsS "http://127.0.0.1:${PORT}/health" \
  > "$OUT/server/health_after.json" 2> "$OUT/server/health_after.err" || true
nvidia-smi > "$OUT/remote/nvidia_smi_after_bench.txt" || true

if [ "$BENCH_RC" -eq 0 ]; then
  echo PASS > "$OUT/run.status"
else
  echo FAIL > "$OUT/run.status"
fi
date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
exit "$BENCH_RC"
