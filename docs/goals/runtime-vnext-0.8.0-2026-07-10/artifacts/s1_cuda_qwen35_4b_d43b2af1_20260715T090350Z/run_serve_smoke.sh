#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/ferrum-infer-rs
artifact=/workspace/artifacts/s1_cuda_qwen35_4b_d43b2af1_20260715T090350Z
model=/workspace/hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
port=8000
server_log="$artifact/serve.server.log"

mkdir -p "$artifact"
cd "$repo"

server_command=(
  target/release/ferrum serve "$model"
  --host 127.0.0.1
  --port "$port"
  --backend cuda
  --profile-jsonl "$artifact/serve.profile.jsonl"
  --profile-detail basic
  --scheduler-trace-jsonl "$artifact/serve.scheduler-trace.jsonl"
)

{
  date -u +started_at=%Y-%m-%dT%H:%M:%SZ
  printf 'git_sha='
  git rev-parse HEAD
  printf 'git_status='
  git status --short | tr '\n' ','
  printf '\n'
  sha256sum target/release/ferrum
  printf 'server_command='
  printf '%q ' "${server_command[@]}"
  printf '\n'
} >"$artifact/serve.command.log"

"${server_command[@]}" >"$server_log" 2>&1 &
server_pid=$!
cleanup() {
  kill -TERM "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 180); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" \
      >"$artifact/serve.health.json" 2>"$artifact/serve.health.stderr"; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo 1 >"$artifact/serve.exit"
  exit 1
fi

payload=$(jq -n --arg model "$model" '{
  model: $model,
  messages: [{role: "user", content: "What is the capital of France? Answer in one word."}],
  temperature: 0,
  max_tokens: 16
}')
stream_payload=$(jq -n --arg model "$model" '{
  model: $model,
  messages: [{role: "user", content: "What is the capital of France? Answer in one word."}],
  temperature: 0,
  max_tokens: 16,
  stream: true,
  stream_options: {include_usage: true}
}')

set +e
curl --fail-with-body -sS --max-time 180 \
  -H 'Content-Type: application/json' \
  -d "$payload" \
  "http://127.0.0.1:$port/v1/chat/completions" \
  >"$artifact/serve.nonstream.json" 2>"$artifact/serve.nonstream.stderr"
nonstream_status=$?
curl --fail-with-body -sSN --max-time 180 \
  -H 'Content-Type: application/json' \
  -d "$stream_payload" \
  "http://127.0.0.1:$port/v1/chat/completions" \
  >"$artifact/serve.stream.sse" 2>"$artifact/serve.stream.stderr"
stream_status=$?
set -e

status=0
if [[ "$nonstream_status" -ne 0 || "$stream_status" -ne 0 ]]; then
  status=1
else
  set +e
  python3 - "$artifact/serve.nonstream.json" "$artifact/serve.stream.sse" <<'PY' \
    >"$artifact/serve.validation.log" 2>&1
import json
import pathlib
import sys

nonstream = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
message = nonstream["choices"][0]["message"]
text = "".join(str(message.get(key) or "") for key in ("content", "reasoning", "reasoning_content"))
if not text.strip():
    raise SystemExit("non-stream response contains no output text")

sse_lines = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
data = [line[6:] for line in sse_lines if line.startswith("data: ")]
if data.count("[DONE]") != 1:
    raise SystemExit(f"expected exactly one [DONE], got {data.count('[DONE]')}")
chunks = [json.loads(item) for item in data if item != "[DONE]"]
stream_text = ""
usage = []
for chunk in chunks:
    if chunk.get("usage") is not None:
        usage.append(chunk["usage"])
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        stream_text += "".join(str(delta.get(key) or "") for key in ("content", "reasoning", "reasoning_content"))
if not stream_text.strip():
    raise SystemExit("stream response contains no output text")
if len(usage) != 1 or int(usage[0].get("completion_tokens") or 0) < 1:
    raise SystemExit(f"stream usage is missing or invalid: {usage!r}")
for marker in ("\ufffd", "<unk>", "[PAD"):
    if marker in text or marker in stream_text:
        raise SystemExit(f"response contains blocker marker {marker!r}")
print(json.dumps({"status": "PASS", "nonstream_text": text, "stream_text": stream_text, "usage": usage}, ensure_ascii=False))
PY
  status=$?
  set -e
fi

date -u +finished_at=%Y-%m-%dT%H:%M:%SZ >>"$artifact/serve.command.log"
printf '%s\n' "$status" >"$artifact/serve.exit"
exit "$status"
