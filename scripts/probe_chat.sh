#!/usr/bin/env bash
# One-shot chat probe: waits for health, sends a known-answer request,
# prints content + finish_reason. Usage: probe_chat.sh PORT [MAX_TOKENS]
set -u
PORT="${1:?usage: probe_chat.sh PORT [MAX_TOKENS]}"
MT="${2:-32}"
for i in $(seq 1 150); do
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 3
done
curl -s --max-time 180 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"probe\",\"temperature\":0,\"max_tokens\":$MT,\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+3? Answer with just the number.\"}]}" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d["choices"][0]; print("content:", repr(c["message"].get("content"))); print("reasoning_len:", len(c["message"].get("reasoning") or "")); print("finish:", c.get("finish_reason"))'
