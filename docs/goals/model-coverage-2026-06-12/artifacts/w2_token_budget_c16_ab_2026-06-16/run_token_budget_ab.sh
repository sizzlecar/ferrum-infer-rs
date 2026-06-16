#!/usr/bin/env bash
set -euo pipefail

OUT=${OUT:-/workspace/artifacts/w2_token_budget_c16_ab_2026-06-16}
REPO=${REPO:-/workspace/ferrum-infer-rs-run}
BIN=${BIN:-/workspace/ferrum-infer-rs/target/release/ferrum}
MODEL_PATH=${MODEL_PATH:-/workspace/hf-cache/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2}
PORT=${PORT:-18143}
export OUT

mkdir -p "$OUT/meta"
cd "$OUT"
git -C "$REPO" rev-parse HEAD >"$OUT/meta/git_sha.txt"
git -C "$REPO" status --short --untracked-files=no >"$OUT/meta/git_status_short.txt"
sha256sum "$BIN" >"$OUT/meta/ferrum.sha256"
nvidia-smi >"$OUT/meta/nvidia_smi_before.txt" || true

cp "$REPO/ferrum.toml" "$OUT/ferrum.base.toml"
cat >"$OUT/ferrum.toml" <<EOF
$(cat "$OUT/ferrum.base.toml")
batch_decode_prof = true
next_batch_prof = true
unified_post_prof = true
EOF

stop_server() {
  local pid_file="$1"
  if [ -f "$pid_file" ]; then
    local pid
    pid=$(cat "$pid_file" || true)
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
      for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" || true
      fi
    fi
  fi
}

cleanup() {
  stop_server "$OUT/server.pid"
}
trap cleanup EXIT

wait_ready() {
  local dir="$1"
  for i in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
      >"$dir/models.json" 2>"$dir/models_curl.err"; then
      echo "$i" >"$dir/ready_at_poll.txt"
      return 0
    fi
    if ! kill -0 "$(cat "$dir/server.pid")" 2>/dev/null; then
      echo "server exited while loading" >&2
      tail -n 180 "$dir/server.stderr" >&2 || true
      return 20
    fi
    sleep 1
  done
  echo "server ready timeout" >&2
  tail -n 180 "$dir/server.stderr" >&2 || true
  return 21
}

start_server() {
  local dir="$1"
  local token_budget="$2"
  stop_server "$OUT/server.pid"
  nvidia-smi >"$dir/nvidia_smi_before.txt" || true
  HF_HOME=/workspace/hf-cache "$BIN" serve \
    --model gemma3:27b-gptq \
    --backend cuda \
    --port "$PORT" \
    --kv-capacity 512 \
    --max-num-seqs 16 \
    --max-num-batched-tokens "$token_budget" \
    --effective-config-json "$dir/effective_config.json" \
    --decision-trace-jsonl "$dir/decision_trace.jsonl" \
    --profile-jsonl "$dir/profile.jsonl" \
    --profile-commit-sha "$(cat "$OUT/meta/git_sha.txt")" \
    --profile-model gemma3-27b-gptq \
    --profile-concurrency 16 \
    >"$dir/server.stdout" 2>"$dir/server.stderr" &
  echo $! >"$dir/server.pid"
  cp "$dir/server.pid" "$OUT/server.pid"
  wait_ready "$dir"
}

run_cell() {
  local token_budget="$1"
  export TOKEN_BUDGET="$token_budget"
  local cell_dir="$OUT/mbt_${token_budget}"
  local smoke_dir="$cell_dir/serve_smoke"
  local bench_dir="$cell_dir/c16_random64_16"
  mkdir -p "$smoke_dir" "$bench_dir"

  start_server "$smoke_dir" "$token_budget"

  cat >"$smoke_dir/chat_payload.json" <<'JSON'
{
  "model": "gemma3:27b-gptq",
  "messages": [
    {"role": "user", "content": "What is 2+3? Answer with only the number."}
  ],
  "max_tokens": 8,
  "temperature": 0,
  "stream": true,
  "stream_options": {"include_usage": true}
}
JSON

  curl -fsS -N \
    -H 'Content-Type: application/json' \
    -d @"$smoke_dir/chat_payload.json" \
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    >"$smoke_dir/chat_stream.sse" 2>"$smoke_dir/chat_stream.err"

  python3 - <<'PY'
import os
import pathlib
import sys

root = pathlib.Path(os.environ["OUT"])
token_budget = os.environ["TOKEN_BUDGET"]
p = root / f"mbt_{token_budget}/serve_smoke/chat_stream.sse"
s = p.read_text(errors="replace")
ok = "data: [DONE]" in s and "5" in s
(root / f"mbt_{token_budget}/serve_smoke/smoke.ok").write_text(str(ok))
print("SMOKE_OK", token_budget, ok)
if not ok:
    print(s[:2000])
    sys.exit(2)
PY

  HF_HOME=/workspace/hf-cache timeout 900 "$BIN" bench-serve \
    --base-url "http://127.0.0.1:${PORT}" \
    --model gemma3:27b-gptq \
    --tokenizer "$MODEL_PATH" \
    --dataset random \
    --random-input-len 64 \
    --random-output-len 16 \
    --concurrency 16 \
    --num-prompts 16 \
    --warmup-requests 4 \
    --n-repeats 1 \
    --fail-on-error \
    --seed 9271 \
    --output json \
    --out "$bench_dir/bench.json" \
    >"$bench_dir/bench.stdout" 2>"$bench_dir/bench.stderr"
  echo $? >"$bench_dir/bench.rc"

  nvidia-smi >"$bench_dir/nvidia_smi_after.txt" || true
  grep -nE \
    'first-token-prof|stream-ttft-prof|unified-prof|iter-prof|unified-decode|panic|CUDA_ERROR|illegal|OOM|out of memory|error' \
    "$smoke_dir/server.stderr" \
    "$bench_dir/bench.stderr" \
    "$smoke_dir/server.stdout" \
    >"$cell_dir/log_scan.txt" 2>/dev/null || true

  stop_server "$OUT/server.pid"
}

for budget in 1024 512; do
  TOKEN_BUDGET="$budget" run_cell "$budget"
done

python3 - <<'PY'
import json
import os
import pathlib

root = pathlib.Path(os.environ["OUT"])
rows = []
for budget in (1024, 512):
    p = root / f"mbt_{budget}/c16_random64_16/bench.json"
    data = json.loads(p.read_text())
    rows.append({
        "max_num_batched_tokens": budget,
        "completed_per_run": data.get("completed_per_run"),
        "errored_per_run": data.get("errored_per_run"),
        "request_throughput_rps": data.get("request_throughput_rps", {}).get("mean"),
        "output_throughput_tps": data.get("output_throughput_tps", {}).get("mean"),
        "ttft_p50_ms": data.get("ttft_ms", {}).get("p50", {}).get("mean"),
        "ttft_p95_ms": data.get("ttft_ms", {}).get("p95", {}).get("mean"),
        "tpot_p50_ms": data.get("tpot_ms", {}).get("p50", {}).get("mean"),
        "itl_p95_ms": data.get("itl_ms", {}).get("p95", {}).get("mean"),
        "output_token_count_source": data.get("output_token_count_source"),
    })
(root / "summary.json").write_text(json.dumps({"rows": rows}, indent=2))
print(json.dumps({"rows": rows}, indent=2))
PY

cleanup
trap - EXIT
