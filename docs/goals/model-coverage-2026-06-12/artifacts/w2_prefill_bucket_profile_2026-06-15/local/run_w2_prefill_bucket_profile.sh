#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/ferrum-infer-rs
FERRUM="$REPO/target/release/ferrum"
OUT=/workspace/w2_prefill_bucket_profile_2026-06-15
DATASET=/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
PORT=8482

cd "$REPO"
if [ -f /root/.cargo/env ]; then
  . /root/.cargo/env
fi
export PATH="/root/.cargo/bin:${PATH}"
export CUDA_COMPUTE_CAP=89

mkdir -p "$OUT/build" "$OUT/server" "$OUT/smoke" "$OUT/bench" "$OUT/remote" "$OUT/profile"
date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short.txt" || true
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"

cat > "$OUT/build/cargo_build.command.txt" <<'TXT'
CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
TXT
set +e
cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$OUT/build/cargo_build.stdout" 2> "$OUT/build/cargo_build.stderr"
BUILD_RC=$?
set -e
echo "$BUILD_RC" > "$OUT/build/cargo_build.rc"
if [ "$BUILD_RC" -ne 0 ]; then
  echo FAIL_BUILD > "$OUT/run.status"
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit "$BUILD_RC"
fi
sha256sum "$FERRUM" > "$OUT/remote/ferrum.sha256"

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
  echo FAIL_READY > "$OUT/run.status"
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
  echo FAIL_SMOKE > "$OUT/run.status"
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
{"cmd":["timeout","1200","$FERRUM","bench-serve","--base-url","http://127.0.0.1:$PORT","--model","gemma3:27b-gptq","--tokenizer","$TOKENIZER","--dataset","sharegpt","--sharegpt-path","$DATASET","--random-output-len","64","--concurrency-sweep","16","--num-prompts","16","--n-repeats","1","--fail-on-error","--seed","9271","--out","$OUT/bench/bench_ferrum_prefill_bucket_sharegpt_c16_16x1.json"]}
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
  --out "$OUT/bench/bench_ferrum_prefill_bucket_sharegpt_c16_16x1.json" \
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
  echo FAIL_BENCH > "$OUT/run.status"
fi
date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
exit "$BENCH_RC"
