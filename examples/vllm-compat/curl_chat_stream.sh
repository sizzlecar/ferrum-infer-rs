#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${OPENAI_MODEL:-Qwen/Qwen3-0.6B}"

curl -N "$BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @- <<JSON
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "Say hi in one short sentence."}],
  "max_tokens": 128,
  "temperature": 0,
  "stream": true,
  "stream_options": {"include_usage": true}
}
JSON
