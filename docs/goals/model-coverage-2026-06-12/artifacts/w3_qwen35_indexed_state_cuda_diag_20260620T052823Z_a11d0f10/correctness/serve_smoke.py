import json
import sys
import urllib.request

base = "http://127.0.0.1:18175/v1/chat/completions"
model = "3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b"
out_dir = sys.argv[1]

prompt = [{"role": "user", "content": "Answer with exactly one word: Paris."}]

def post(payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base, data=data, headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=120)

non_stream_payload = {
    "model": model,
    "messages": prompt,
    "temperature": 0,
    "max_tokens": 8,
}
resp = post(non_stream_payload)
raw = resp.read().decode("utf-8")
open(f"{out_dir}/serve_non_stream_response.json", "w", encoding="utf-8").write(raw)
obj = json.loads(raw)
content = obj["choices"][0]["message"].get("content") or ""
if "Paris" not in content:
    raise SystemExit(f"non-stream content missing Paris: {content!r}")
if not obj.get("usage"):
    raise SystemExit("non-stream usage missing")

stream_payload = {
    "model": model,
    "messages": prompt,
    "temperature": 0,
    "max_tokens": 8,
    "stream": True,
    "stream_options": {"include_usage": True},
}
resp = post(stream_payload)
lines = []
done_count = 0
pieces = []
usage_seen = False
for raw_line in resp:
    line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
    lines.append(line)
    if not line.startswith("data: "):
        continue
    payload = line[6:].strip()
    if payload == "[DONE]":
        done_count += 1
        continue
    event = json.loads(payload)
    if event.get("usage"):
        usage_seen = True
    for choice in event.get("choices", []):
        delta = choice.get("delta") or {}
        if delta.get("content"):
            pieces.append(delta["content"])
open(f"{out_dir}/serve_stream_sse.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
stream_content = "".join(pieces)
open(f"{out_dir}/serve_stream_content.txt", "w", encoding="utf-8").write(stream_content)
summary = {
    "non_stream_content": content,
    "stream_content": stream_content,
    "done_count": done_count,
    "usage_seen": usage_seen,
    "stream_piece_count": len(pieces),
}
open(f"{out_dir}/serve_smoke_summary.json", "w", encoding="utf-8").write(json.dumps(summary, indent=2) + "\n")
if done_count != 1:
    raise SystemExit(f"expected exactly one DONE, got {done_count}")
if not stream_content.strip():
    raise SystemExit("stream emitted no content")
if "Paris" not in stream_content:
    raise SystemExit(f"stream content missing Paris: {stream_content!r}")
if not usage_seen:
    raise SystemExit("stream usage missing")
print("SERVE_SMOKE_PASS")
