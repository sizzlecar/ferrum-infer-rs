import json, os, pathlib, sys, urllib.request
art = pathlib.Path(os.environ["ART"])
base = os.environ["BASE_URL"].rstrip("/")
model = os.environ["MODEL_ID"]
reqdir = art / "requests"
reqdir.mkdir(exist_ok=True)
BAD = ["<unk>", "[PAD", "\ufffd"]

def post(payload, name, stream=False):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base + "/v1/chat/completions", data=data, headers={"content-type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read().decode("utf-8", "replace")
            (reqdir / f"{name}.status.txt").write_text(str(response.status) + "\n")
            (reqdir / f"{name}.body.txt").write_text(body, errors="replace")
            if response.status != 200:
                raise SystemExit(f"{name}: HTTP {response.status}: {body[:500]}")
            return body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        (reqdir / f"{name}.status.txt").write_text(str(exc.code) + "\n")
        (reqdir / f"{name}.body.txt").write_text(body, errors="replace")
        raise SystemExit(f"{name}: HTTP {exc.code}: {body[:1000]}")

def check_text(label, text):
    if not str(text).strip():
        raise SystemExit(f"{label}: empty text")
    if str(text).strip() == "</think>":
        raise SystemExit(f"{label}: isolated </think>")
    for token in BAD:
        if token in str(text):
            raise SystemExit(f"{label}: bad token {token!r} in {str(text)[:300]!r}")

nonstream_payload = {
    "model": model,
    "messages": [{"role":"user","content":"What is the capital of France? Answer in one short sentence."}],
    "temperature": 0,
    "max_tokens": 64,
}
body = post(nonstream_payload, "01_nonstream")
data = json.loads(body)
text = data["choices"][0]["message"].get("content") or ""
check_text("nonstream", text)

stream_payload = {
    "model": model,
    "messages": [{"role":"user","content":"Say ferrum-stream-ok in one short sentence."}],
    "temperature": 0,
    "max_tokens": 64,
    "stream": True,
    "stream_options": {"include_usage": True},
}
raw = post(stream_payload, "02_stream")
done = 0
chunks = 0
usage_seen = False
for line in raw.splitlines():
    line = line.strip()
    if not line.startswith("data: "):
        continue
    payload = line[len("data: "):]
    if payload == "[DONE]":
        done += 1
        continue
    obj = json.loads(payload)
    if obj.get("usage"):
        usage_seen = True
    for choice in obj.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content"):
            chunks += 1
if done != 1:
    raise SystemExit(f"stream: DONE count {done}, expected 1")
if chunks < 1:
    raise SystemExit("stream: no content chunks")
if not usage_seen:
    raise SystemExit("stream: include_usage did not produce usage")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
tool_payload = {
    "model": model,
    "messages": [{"role":"user","content":"Call get_weather for Paris. Do not answer in natural language."}],
    "tools": TOOLS,
    "tool_choice": {"type":"function","function":{"name":"get_weather"}},
    "temperature": 0,
    "max_tokens": 160,
}
body = post(tool_payload, "03_tool_call")
data = json.loads(body)
message = data["choices"][0]["message"]
tool_calls = message.get("tool_calls") or []
if not tool_calls:
    raise SystemExit(f"tool_call: missing tool_calls in message {message}")
if tool_calls[0].get("function", {}).get("name") != "get_weather":
    raise SystemExit(f"tool_call: wrong function {tool_calls}")

expected = {"answer":"scenario-ok"}
structured_payload = {
    "model": model,
    "messages": [{"role":"user","content":"Return exactly this JSON object and nothing else: {\"answer\":\"scenario-ok\"}"}],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "ScenarioObject",
            "strict": True,
            "schema": {
                "type":"object",
                "properties":{"answer":{"type":"string"}},
                "required":["answer"],
            },
        },
    },
    "temperature": 0,
    "max_tokens": 192,
}
body = post(structured_payload, "04_structured")
data = json.loads(body)
text = data["choices"][0]["message"].get("content") or ""
parsed = json.loads(text)
if parsed != expected:
    raise SystemExit(f"structured: {parsed!r} != {expected!r}")

summary = {
    "status":"pass",
    "nonstream_text": text[:500],
    "stream_done_count": done,
    "stream_content_chunks": chunks,
    "stream_usage_seen": usage_seen,
    "tool_calls": tool_calls,
    "structured": parsed,
}
(reqdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
