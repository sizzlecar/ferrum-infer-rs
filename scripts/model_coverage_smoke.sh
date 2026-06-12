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
KV_CAPACITY=""
MAX_SEQS=4
while [ $# -gt 0 ]; do
  case "$1" in
    --reasoning) REASONING=1 ;;
    --port) shift; PORT="$1" ;;
    --kv-capacity) shift; KV_CAPACITY="$1" ;;
    --max-seqs) shift; MAX_SEQS="$1" ;;
  esac
  shift
done

# The server-profile autosizer prefers high concurrency and can pick a
# 512-1024 token per-sequence budget that 400s this ladder's max_tokens
# (recorded as a W1 product gap in docs/goals/model-coverage-2026-06-12/).
# Pin deterministic serve conditions instead: the ladder is single-request,
# so trade concurrency for per-seq capacity. The KV pool is max_num_seqs ×
# kv_capacity — the capacity bump MUST come with the concurrency cut
# (32 seqs × 8192 tokens is a ~36 GB pool on an 8B model, which thrashes
# a 32 GB Mac). 4 × 4096/8192 stays in the default pool's memory class.
if [ -z "$KV_CAPACITY" ]; then
  if [ "$REASONING" = "1" ]; then
    # Reasoning models think for hundreds-to-thousands of tokens per reply.
    KV_CAPACITY=8192
  else
    KV_CAPACITY=4096
  fi
fi
SERVE_ARGS=(--kv-capacity "$KV_CAPACITY" --max-num-seqs "$MAX_SEQS")
# SMOKE_NO_KV_PIN=1 drops the pinning entirely — on CUDA pods the
# autosizer manages pools correctly and explicit pins stack a second
# allocation on top (OOMs a 24GB card on 30B-class GPTQ).
if [ -n "${SMOKE_NO_KV_PIN:-}" ]; then
  SERVE_ARGS=()
fi
# Extra serve args (e.g. `--gpu-devices 0,1` on dual-GPU pods).
if [ -n "${FERRUM_SMOKE_EXTRA_SERVE_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  SERVE_ARGS+=(${FERRUM_SMOKE_EXTRA_SERVE_ARGS})
fi

BIN="${FERRUM_BIN:-target/release/ferrum}"
LOG="/tmp/ferrum_w1_smoke_${PORT}.log"
BASE="http://127.0.0.1:${PORT}"

"$BIN" serve "$MODEL" --port "$PORT" ${SERVE_ARGS[@]+"${SERVE_ARGS[@]}"} >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[smoke] waiting for $BASE/health (model: $MODEL)"
for i in $(seq 1 150); do
  if curl -sf --noproxy '*' --max-time 2 "$BASE/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
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

# Talk to the local server directly — never through a system proxy.
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
urllib.request.install_opener(opener)

def chat(payload, stream=False):
    payload = {"model": MODEL, "temperature": 0, **payload}
    if stream:
        payload["stream"] = True
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=int(os.environ.get("SMOKE_REQ_TIMEOUT", "600"))) as r:
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

# L2 — known answer N=10 (GOAL gate: 10/10 semantically correct) + L3
# natural-EOS / reasoning assertions on the first reply.
KNOWN_ANSWERS = [
    ("What is 2+3? Answer with just the number.", "5"),
    ("What is 7*6? Answer with just the number.", "42"),
    ("Spell the English word for the number 9. Answer with just the word.", "nine"),
    ("What is the capital of France? Answer with just the city name.", "Paris"),
    ("What is 100-58? Answer with just the number.", "42"),
    ("Which planet do humans live on? Answer with just the planet name.", "Earth"),
    ("What is 12/4? Answer with just the number.", "3"),
    ("What color is a stop sign? Answer with just the color.", "red"),
    ("What is 2 to the power of 5? Answer with just the number.", "32"),
    ("How many days are in a week? Answer with just the number.", "7"),
]
ok_known, first = 0, None
for q, expected in KNOWN_ANSWERS:
    r = chat({"messages": [{"role": "user", "content": q}], "max_tokens": 2048})
    if first is None:
        first = r
    content = r["choices"][0]["message"].get("content") or ""
    if expected.lower() in content.lower():
        ok_known += 1
    else:
        print(f"[smoke]      known-answer miss: {q!r} -> {content[-120:]!r}")
check("known-answer 10/10", ok_known == 10, f"{ok_known}/10")

msg = first["choices"][0]["message"]
content, finish = msg.get("content") or "", first["choices"][0].get("finish_reason")
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

# L3 — custom stop sequence mechanics: finish_reason=stop, stop string
# never leaks into content. (For reasoning models the stop may hit inside
# the think block — content emptiness is fine, the wire mechanics are
# what's pinned here.)
r = chat({"messages": [{"role": "user", "content": "Count from 1 to 20 as plain comma-separated numbers, nothing else."}],
          "max_tokens": 1024, "stop": ["7"]})
content = r["choices"][0]["message"].get("content") or ""
finish = r["choices"][0].get("finish_reason")
check("custom stop finish_reason", finish == "stop", f"finish={finish}")
check("custom stop does not leak", "7" not in content, repr(content[-120:]))

# L3 — max_tokens cutoff: finish_reason=length and budget respected
r = chat({"messages": [{"role": "user", "content": "Write a long story about the sea."}], "max_tokens": 8})
finish = r["choices"][0].get("finish_reason")
used = (r.get("usage") or {}).get("completion_tokens")
check("max_tokens finish_reason length", finish == "length", f"finish={finish}")
check("max_tokens budget respected", used is not None and used <= 8, f"completion_tokens={used}")

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

# L4 — strict json_schema (20x, GOAL gate criterion)
schema = {"type": "json_schema", "json_schema": {"name": "Answer", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}}}
ok_schema = 0
for _ in range(20):
    r = chat({"messages": [{"role": "user", "content": "Return the sum of 123+456."}],
              "response_format": schema, "max_tokens": 2048})
    content = r["choices"][0]["message"].get("content") or ""
    try:
        if isinstance(json.loads(content).get("answer"), int):
            ok_schema += 1
    except Exception:
        pass
check("strict json_schema 20/20", ok_schema == 20, f"{ok_schema}/20")

sys.exit(1 if failures else 0)
PY
RC=$?
if [ "$RC" -eq 0 ]; then
  echo "FERRUM W1 SMOKE PASS: $MODEL"
else
  echo "FERRUM W1 SMOKE FAIL: $MODEL (see $LOG)"
fi
exit "$RC"
