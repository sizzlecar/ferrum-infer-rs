#!/usr/bin/env bash
set -euo pipefail

OUT=/workspace/w2_ttft_profile_c16
REPO=/workspace/ferrum-infer-rs-run
BIN=/workspace/ferrum-infer-rs/target/release/ferrum
MODEL_PATH=/workspace/hf-cache/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
PORT=18142

cd "$OUT"
cp "$REPO/ferrum.toml" "$OUT/ferrum.toml"
cat >>"$OUT/ferrum.toml" <<'EOF'
batch_decode_prof = true
next_batch_prof = true
unified_post_prof = true
EOF

SMOKE_DIR="$OUT/serve_smoke_kv512"
BENCH_DIR="$OUT/c16_profile_kv512"

mkdir -p "$SMOKE_DIR" "$BENCH_DIR"

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

wait_ready() {
  local dir="$1"
  for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
      >"$dir/models.json" 2>"$dir/models_curl.err"; then
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
  stop_server "$OUT/server.pid"
  nvidia-smi >"$dir/nvidia_smi_before.txt" || true
  HF_HOME=/workspace/hf-cache "$BIN" serve \
    --model gemma3:27b-gptq \
    --backend cuda \
    --port "$PORT" \
    --kv-capacity 512 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 1024 \
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

start_server "$SMOKE_DIR"

cat >"$SMOKE_DIR/chat_payload.json" <<'JSON'
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
  -d @"$SMOKE_DIR/chat_payload.json" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  >"$SMOKE_DIR/chat_stream.sse" 2>"$SMOKE_DIR/chat_stream.err"

python3 - <<'PY'
import pathlib
import sys

p = pathlib.Path("/workspace/w2_ttft_profile_c16/serve_smoke_kv512/chat_stream.sse")
s = p.read_text(errors="replace")
ok = "data: [DONE]" in s and "5" in s
pathlib.Path("/workspace/w2_ttft_profile_c16/serve_smoke_kv512/smoke.ok").write_text(str(ok))
print("SMOKE_OK", ok)
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
  --out "$BENCH_DIR/bench.json" \
  >"$BENCH_DIR/bench.stdout" 2>"$BENCH_DIR/bench.stderr"
echo $? >"$BENCH_DIR/bench.rc"

nvidia-smi >"$BENCH_DIR/nvidia_smi_after.txt" || true
grep -nE \
  'first-token-prof|stream-ttft-prof|unified-prof|iter-prof|unified-decode|panic|CUDA_ERROR|illegal|OOM|out of memory|error' \
  "$SMOKE_DIR/server.stderr" \
  "$BENCH_DIR/bench.stderr" \
  "$SMOKE_DIR/server.stdout" \
  >"$OUT/log_scan_kv512.txt" 2>/dev/null || true
grep -nE \
  'first-token-prof|stream-ttft-prof|unified-prof|iter-prof|unified-decode|panic|CUDA_ERROR|illegal|OOM|out of memory|error' \
  "$SMOKE_DIR/server.stderr" \
  >"$OUT/profile_log_extract_kv512.txt" 2>/dev/null || true

stop_server "$OUT/server.pid"
