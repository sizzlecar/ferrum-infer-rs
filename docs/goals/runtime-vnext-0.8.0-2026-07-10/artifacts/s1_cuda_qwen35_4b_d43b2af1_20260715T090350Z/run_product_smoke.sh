#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/ferrum-infer-rs
artifact=/workspace/artifacts/s1_cuda_qwen35_4b_d43b2af1_20260715T090350Z
model=/workspace/hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
stdout="$artifact/run.stdout.jsonl"
stderr="$artifact/run.stderr.log"
profile="$artifact/run.profile.jsonl"
scheduler_trace="$artifact/run.scheduler-trace.jsonl"

mkdir -p "$artifact"
cd "$repo"

command=(
  target/release/ferrum run "$model"
  --backend cuda
  --prompt 'What is the capital of France? Answer in one word.'
  --max-tokens 16
  --output-format jsonl
  --profile-jsonl "$profile"
  --profile-detail basic
  --scheduler-trace-jsonl "$scheduler_trace"
)

{
  date -u +started_at=%Y-%m-%dT%H:%M:%SZ
  printf 'git_sha='
  git rev-parse HEAD
  printf 'git_status='
  git status --short | tr '\n' ','
  printf '\n'
  sha256sum target/release/ferrum
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$artifact/run.command.log"

set +e
timeout 300s "${command[@]}" >"$stdout" 2>"$stderr"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  set +e
  python3 - "$stdout" <<'PY' >"$artifact/run.validation.log" 2>&1
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
assistant = [record for record in records if record.get("event") == "assistant"]
if len(assistant) != 1:
    raise SystemExit(f"expected exactly one assistant record, got {len(assistant)}")
record = assistant[0]
content = record.get("content")
if not isinstance(content, str) or not content.strip():
    raise SystemExit("assistant content is empty")
if int(record.get("n_tokens") or 0) < 1:
    raise SystemExit("assistant emitted no tokens")
for marker in ("\ufffd", "<unk>", "[PAD", "data: [DONE]"):
    if marker in content:
        raise SystemExit(f"assistant content contains blocker marker {marker!r}")
print(json.dumps({"status": "PASS", "assistant": record}, ensure_ascii=False))
PY
  validation_status=$?
  set -e
  if [[ "$validation_status" -ne 0 ]]; then
    status="$validation_status"
  fi
fi

date -u +finished_at=%Y-%m-%dT%H:%M:%SZ >>"$artifact/run.command.log"
printf '%s\n' "$status" >"$artifact/run.exit"
exit "$status"
