#!/usr/bin/env bash
# W1 model-coverage smoke ladder (L2/L3/L4 subset runnable against a local
# `ferrum serve`). See docs/goals/model-coverage-2026-06-12/GOAL.md.
#
# Usage:
#   scripts/model_coverage_smoke.sh <model-alias> [--reasoning] [--port N]
#
# Gates covered:
#   L2/L3: known-answer, natural EOS stop, multi-turn recall,
#          stream==non-stream content (temp 0), reasoning extraction
#          (--reasoning models must populate message.reasoning)
#   L4:    required tool call (10x), strict json_schema (10x)
#
# Prints `FERRUM W1 SMOKE PASS: <alias>` on success; non-zero exit on any
# failure. The server log is kept at /tmp/ferrum_w1_smoke_<port>.log.

set -u
MODEL="${1:?usage: model_coverage_smoke.sh <alias> [--reasoning] [--port N]}"
shift
REASONING=0
PORT=18200
while [ $# -gt 0 ]; do
  case "$1" in
    --reasoning) REASONING=1 ;;
    --port) shift; PORT="$1" ;;
  esac
  shift
done

BIN="${FERRUM_BIN:-target/release/ferrum}"
LOG="/tmp/ferrum_w1_smoke_${PORT}.log"
BASE="http://127.0.0.1:${PORT}"

"$BIN" serve "$MODEL" --port "$PORT" >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[smoke] waiting for $BASE/health (model: $MODEL)"
for i in $(seq 1 150); do
  if curl -s --max-time 2 "$BASE/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[smoke] FAIL: server exited during startup; tail of log:"; tail -20 "$LOG"; exit 1
  fi
  sleep 2
done
if [ "${HEALTHY:-0}" != "1" ]; then
  echo "[smoke] FAIL: server not healthy after 300s"; tail -20 "$LOG"; exit 1
fi

MODEL="$MODEL" BASE="$BASE" REASONING="$REASONING" python3 - <<'PY'
import json, os, sys, urllib.request

BASE = os.environ["BASE"]
MODEL = os.environ["MODEL"]
REASONING = os.environ["REASONING"] == "1"
failures = []

def chat(payload, stream=False):
    payload = {"model": MODEL, "temperature": 0, **payload}
    if stream:
        payload["stream"] = True
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            return json.load(r)
        content, tool_names, finish = "", [], None
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            for ch in chunk.get("choices", []):
                delta = ch.get("delta", {})
                content += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    name = (tc.get("function") or {}).get("name")
                    if name:
                        tool_names.append(name)
                finish = ch.get("finish_reason") or finish
        return {"content": content, "tool_names": tool_names, "finish": finish}

def check(name, ok, detail=""):
    print(f"[smoke] {'ok  ' if ok else 'FAIL'} {name}{(' — ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(name)

# L2/L3 — known answer + natural EOS
r = chat({"messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}], "max_tokens": 2048})
msg = r["choices"][0]["message"]
content, finish = msg.get("content") or "", r["choices"][0].get("finish_reason")
check("known-answer contains 5", "5" in content, repr(content[-200:]))
check("natural EOS stop", finish == "stop", f"finish={finish}")
if REASONING:
    check("reasoning extracted", bool(msg.get("reasoning")), "message.reasoning empty")
    check("no think tags leak into content", "<think>" not in content, repr(content[:120]))

# L3 — multi-turn recall
r = chat({"messages": [
    {"role": "user", "content": "My name is Bob. Remember it."},
    {"role": "assistant", "content": "Understood, Bob."},
    {"role": "user", "content": "What is my name? Answer with just the name."},
], "max_tokens": 1024})
content = r["choices"][0]["message"].get("content") or ""
check("multi-turn recall Bob", "Bob" in content, repr(content[-200:]))

# L3 — stream vs non-stream content identity (temp 0)
prompt = {"messages": [{"role": "user", "content": "Name three colors, comma separated, nothing else."}], "max_tokens": 1024}
ns = chat(prompt)["choices"][0]["message"].get("content") or ""
st = chat(prompt, stream=True)["content"]
check("stream == non-stream", ns.strip() == st.strip(), f"non-stream={ns!r} stream={st!r}")

# L4 — required tool call (10x)
tool = {"type": "function", "function": {
    "name": "calc", "description": "Evaluate an arithmetic expression.",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}
ok_calls = 0
for _ in range(10):
    r = chat({"messages": [{"role": "user", "content": "Use calc to compute 123+456."}],
              "tools": [tool], "tool_choice": "required", "max_tokens": 2048})
    calls = r["choices"][0]["message"].get("tool_calls") or []
    if calls and calls[0]["function"]["name"] == "calc":
        try:
            json.loads(calls[0]["function"]["arguments"]); ok_calls += 1
        except Exception:
            pass
check("required tool-call 10/10", ok_calls == 10, f"{ok_calls}/10")

# L4 — strict json_schema (10x)
schema = {"type": "json_schema", "json_schema": {"name": "Answer", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}}}
ok_schema = 0
for _ in range(10):
    r = chat({"messages": [{"role": "user", "content": "Return the sum of 123+456."}],
              "response_format": schema, "max_tokens": 2048})
    content = r["choices"][0]["message"].get("content") or ""
    try:
        if isinstance(json.loads(content).get("answer"), int):
            ok_schema += 1
    except Exception:
        pass
check("strict json_schema 10/10", ok_schema == 10, f"{ok_schema}/10")

sys.exit(1 if failures else 0)
PY
RC=$?
if [ "$RC" -eq 0 ]; then
  echo "FERRUM W1 SMOKE PASS: $MODEL"
else
  echo "FERRUM W1 SMOKE FAIL: $MODEL (see $LOG)"
fi
exit "$RC"
