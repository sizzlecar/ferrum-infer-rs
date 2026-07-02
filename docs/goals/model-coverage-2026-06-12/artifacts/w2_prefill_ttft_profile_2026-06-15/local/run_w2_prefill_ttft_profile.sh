#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/ferrum-infer-rs
FERRUM="$REPO/target/release/ferrum"
OUT=/workspace/w2_prefill_ttft_profile_2026-06-15
DATASET=/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
PORT=8478

cd "$REPO"
if [ -f /root/.cargo/env ]; then
  . /root/.cargo/env
fi
export PATH="/root/.cargo/bin:${PATH}"

mkdir -p "$OUT/server" "$OUT/smoke" "$OUT/bench" "$OUT/remote" "$OUT/profile"
date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short.txt" || true
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"
sha256sum "$FERRUM" > "$OUT/remote/ferrum.sha256"

cat > "$OUT/gpu_contract.md" <<'TXT'
lane: W2 Gemma3 CUDA prefill/TTFT profile diagnostic
instance: Vast 40826362, 1x RTX 4090, cache-retained native CUDA machine
expected_runtime_cost: 8-20min, hard cap 30min, reused instance at about USD 0.425/hr when running
stop_condition: start/SSH/CUDA/server readiness first failure, chat smoke failure, c16 ShareGPT diagnostic complete and artifacts copied, or 30min hard cap
correctness_gate: ferrum serve readiness plus non-stream chat smoke before bench-serve; bench-serve must use --fail-on-error
performance_command: diagnostic-only natural ASCII ShareGPT c16 with bench-serve --fail-on-error, seed 9271, n_repeats=1, num_prompts=16
profile_scope: server runs with FERRUM_PREFILL_OP_PROFILE=1 to split first-token prefill work; not release-grade evidence and not a product behavior validation path
baseline_engine_version_build: vLLM 0.10.1.1 CUDA12 baseline from w2_vllm_sharegpt_baseline_probe_2026-06-15; same RTX 4090, same HF/safetensors GPTQ model, same ShareGPT dataset
release_grade_status: not release-grade evidence; diagnostic only
TXT

pkill -f "target/release/ferrum serve" 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 3

cat > "$OUT/server/serve.command.json" <<JSON
{"env":{"FERRUM_PREFILL_OP_PROFILE":"1"},"cmd":["$FERRUM","serve","--model","gemma3:27b-gptq","--port","$PORT","--kv-capacity","512","--max-num-seqs","16","--effective-config-json","$OUT/server/serve_effective_config.json","--decision-trace-jsonl","$OUT/server/serve_decision_trace.jsonl"]}
JSON
FERRUM_PREFILL_OP_PROFILE=1 "$FERRUM" serve \
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
{"cmd":["timeout","1200","$FERRUM","bench-serve","--base-url","http://127.0.0.1:$PORT","--model","gemma3:27b-gptq","--tokenizer","$TOKENIZER","--dataset","sharegpt","--sharegpt-path","$DATASET","--random-output-len","64","--concurrency-sweep","16","--num-prompts","16","--n-repeats","1","--fail-on-error","--seed","9271","--out","$OUT/bench/bench_ferrum_prefill_profile_sharegpt_c16_16x1.json"]}
JSON
set +e
timeout 1200 "$FERRUM" bench-serve \
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
  --out "$OUT/bench/bench_ferrum_prefill_profile_sharegpt_c16_16x1.json" \
  > "$OUT/bench/bench-serve.stdout" 2> "$OUT/bench/bench-serve.stderr"
BENCH_RC=$?
set -e
echo "$BENCH_RC" > "$OUT/bench/bench-serve.rc"
nvidia-smi > "$OUT/remote/nvidia_smi_after_bench.txt" || true

grep '^[[]prefill-profile[]]' "$OUT/server/server.log" > "$OUT/profile/prefill_profile.log" || true
python3 - <<PY
import json, pathlib, re, statistics
path = pathlib.Path("$OUT/profile/prefill_profile.log")
lines = path.read_text().splitlines() if path.exists() else []
calls = []
current = None
total_re = re.compile(r"tokens=(\\d+) layers total=(\\d+) ms")
bucket_re = re.compile(r"\\[prefill-profile\\] ([^:]+): (\\d+) calls (\\d+) ms \\(avg (\\d+) us\\)")
for line in lines:
    if " layers total=" in line:
        m = total_re.search(line)
        if m:
            current = {"tokens": int(m.group(1)), "total_ms": int(m.group(2)), "buckets": {}}
            calls.append(current)
        continue
    m = bucket_re.search(line)
    if m and current is not None:
        current["buckets"][m.group(1)] = {
            "calls": int(m.group(2)),
            "ms": int(m.group(3)),
            "avg_us": int(m.group(4)),
        }
summary = {"num_prefills": len(calls), "calls": calls}
if calls:
    totals = [c["total_ms"] for c in calls]
    summary["total_ms"] = {
        "min": min(totals),
        "max": max(totals),
        "mean": statistics.mean(totals),
        "median": statistics.median(totals),
    }
    bucket_names = sorted({b for c in calls for b in c["buckets"]})
    summary["bucket_ms_mean"] = {
        b: statistics.mean([c["buckets"].get(b, {}).get("ms", 0) for c in calls])
        for b in bucket_names
    }
pathlib.Path("$OUT/profile/prefill_profile_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\\n"
)
PY

if [ "$BENCH_RC" -eq 0 ]; then
  echo PASS > "$OUT/run.status"
else
  echo FAIL > "$OUT/run.status"
fi
date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
exit "$BENCH_RC"
