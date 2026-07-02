#!/usr/bin/env bash
set -euo pipefail
ART=${ART:?}
REPO=/workspace/ferrum-clean
MODEL=/workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b
MODEL_ID=3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b
DATASET=/workspace/artifacts/ferrum_w3_sharegpt_ab_20b8946b_20260619T121844Z_ninja/dataset/ascii_sharegpt_w3_100_58d5721d.jsonl
TARGET_DIR=/workspace/ferrum-w3-qwen35-ac98207/target
PORT=58101
BASE_URL=http://127.0.0.1:${PORT}
SERVER_PID=""
export STATUS=fail
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
on_exit() {
  local rc=$?
  cleanup
  if [[ $rc -eq 0 ]]; then
    STATUS=pass
    export STATUS
    export STATUS
    echo "W3 QWEN35 PREFILL PROFILE SMOKE PASS: ${ART}" | tee -a "${ART}/smoke.log"
  else
    STATUS=fail
    export STATUS
    echo "W3 QWEN35 PREFILL PROFILE SMOKE FAIL rc=${rc}: ${ART}" | tee -a "${ART}/smoke.log" >&2
  fi
  python3 - <<PY || true
import json, os, pathlib, datetime
art = pathlib.Path(os.environ["ART"])
summary = {
    "schema_version": 1,
    "status": os.environ.get("STATUS", "fail"),
    "artifact_dir": str(art),
    "finished_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
}
(art / "summary.status.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
}
trap on_exit EXIT

cd "${REPO}"
if [[ -f /root/.cargo/env ]]; then
  source /root/.cargo/env
fi
export PATH="/root/.cargo/bin:/usr/local/cuda/bin:${PATH}"
mkdir -p "${ART}" "${ART}/requests"
# Ensure this smoke uses only typed/default product config and not inherited hidden FERRUM_* env.
while IFS='=' read -r name _; do
  case "${name}" in
    FERRUM_*) unset "${name}" ;;
  esac
done < <(env)
export ART
export HF_HOME=/workspace/hf-cache
export CARGO_TARGET_DIR="${TARGET_DIR}"
export RUST_BACKTRACE=1
export CUDA_VISIBLE_DEVICES=0
export NO_COLOR=1
export FERRUM_QWEN35_PREFILL_PROFILE=1
export FERRUM_PROFILE_JSONL="${ART}/qwen35_prefill_profile.jsonl"
export FERRUM_PROFILE_COMMIT_SHA="050d73f3a11cea757a53fb4e91d9cd236a4a62e0"
export FERRUM_PROFILE_ENV_HASH="sha256:qwen35-prefill-profile-050d73f3"
export FERRUM_PROFILE_MODEL="3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b"
export FERRUM_PROFILE_CONCURRENCY=32
export FERRUM_PROFILE_RUNTIME_FLAGS_JSON='{"diagnostic":"qwen35_prefill_profile","release_evidence":false}'

cat > "${ART}/lane_contract.json" <<JSON
{
  "schema_version": 1,
  "lane": "W3 Qwen35 CUDA prefill-profile smoke",
  "commit": "$(git rev-parse HEAD)",
  "source_fix_commit": "adf70f90",
  "post_validation": "finish_reason length/repetition/tool/structured hard checks",
  "expected_runtime_minutes": "10-25",
  "hourly_cost_usd": 0.662962962962963,
  "stop_condition": "CUDA build failure, product smoke failure, diagnostic bench failure, profiling JSONL missing, or PASS artifact copied back",
  "correctness_gate": "release CUDA build plus ferrum run and ferrum serve smoke; effective config must select vllm_paged_attn_v2 with FERRUM_USE_VLLM_PAGED_ATTN=1",
  "performance_command": "ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1 --concurrency-sweep 1,32 (diagnostic only)",
  "model_path": "${MODEL}",
  "dataset": "${DATASET}"
}
JSON

{
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=${REPO}"
  echo "artifact=${ART}"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_status_short_begin"
  git status --short --branch
  echo "git_status_short_end"
  echo "nvidia_smi_query_begin"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader
  echo "nvidia_smi_query_end"
  echo "nvcc_begin"
  nvcc --version
  echo "nvcc_end"
  rustc --version
  cargo --version
  test -d "${MODEL}"
  test -f "${MODEL}/config.json"
  test -f "${MODEL}/tokenizer.json"
} | tee "${ART}/environment.log"

BUILD_CMD=(cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source)
printf '%q ' "${BUILD_CMD[@]}" > "${ART}/build.command.txt"
printf '\n' >> "${ART}/build.command.txt"
"${BUILD_CMD[@]}" 2>&1 | tee "${ART}/build.log"
BIN="${TARGET_DIR}/release/ferrum"
test -x "${BIN}"
sha256sum "${BIN}" | tee "${ART}/ferrum.sha256"

RUN_CMD=("${BIN}" run "${MODEL}" --backend cuda --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 4096 --max-tokens 64 --temperature 0 --disable-thinking --output-format jsonl --effective-config-json "${ART}/run_effective_config.json" --decision-trace-jsonl "${ART}/run_decision_trace.jsonl" --prompt "Return one short sentence containing the exact word ferrum-ok.")
printf '%q ' "${RUN_CMD[@]}" > "${ART}/run.command.txt"
printf '\n' >> "${ART}/run.command.txt"
"${RUN_CMD[@]}" > "${ART}/run_stdout.jsonl" 2> "${ART}/run_stderr.log"

python3 - <<'PY'
import json, pathlib, sys, os
art = pathlib.Path(os.environ["ART"])
bad = ["<unk>", "[PAD", "\ufffd"]
assistants = []
for line in (art / "run_stdout.jsonl").read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    if obj.get("event") == "assistant":
        assistants.append(obj)
if not assistants:
    raise SystemExit("no assistant event in ferrum run output")
text = str(assistants[-1].get("content") or "")
if not text.strip():
    raise SystemExit("empty ferrum run assistant text")
if any(token in text for token in bad):
    raise SystemExit(f"bad token in ferrum run text: {text[:200]!r}")
if text.strip() == "</think>":
    raise SystemExit("isolated </think> in ferrum run output")
(art / "run_validation.json").write_text(json.dumps({"status":"pass","assistant_text":text[:1000]}, indent=2) + "\n")
PY

python3 - <<'PY'
import json, pathlib, sys, os
art = pathlib.Path(os.environ["ART"])

def validate(path, label):
    doc = json.loads(path.read_text())
    entries = {entry["key"]: entry for entry in doc.get("entries", [])}
    def expect(key, value):
        actual = entries.get(key, {}).get("effective_value")
        if actual != value:
            raise SystemExit(f"{label}: {key}={actual!r}, expected {value!r}")
    expect("FERRUM_USE_VLLM_PAGED_ATTN", "1")
    expect("FERRUM_VLLM_PAGED_ATTN_V1_SHORT", "0")
    if doc.get("selected_attention_impl") != "vllm_paged_attn_v2":
        raise SystemExit(f"{label}: selected_attention_impl={doc.get('selected_attention_impl')!r}")
    model = doc.get("model_capabilities") or {}
    if model.get("architecture") != "qwen3_5_moe":
        raise SystemExit(f"{label}: architecture={model.get('architecture')!r}")
    if model.get("head_dim") != 256:
        raise SystemExit(f"{label}: head_dim={model.get('head_dim')!r}")
    return {"label": label, "selected_attention_impl": doc.get("selected_attention_impl"), "entries": {k: entries[k] for k in ["FERRUM_USE_VLLM_PAGED_ATTN", "FERRUM_VLLM_PAGED_ATTN_V1_SHORT", "FERRUM_VLLM_MOE"] if k in entries}}
summary = [validate(art / "run_effective_config.json", "run")]
(art / "effective_config_validation_run.json").write_text(json.dumps({"status":"pass","checks":summary}, indent=2) + "\n")
PY

SERVE_CMD=("${BIN}" serve "${MODEL}" --backend cuda --host 127.0.0.1 --port "${PORT}" --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 4096 --effective-config-json "${ART}/serve_effective_config.json" --decision-trace-jsonl "${ART}/serve_decision_trace.jsonl")
printf '%q ' "${SERVE_CMD[@]}" > "${ART}/serve.command.txt"
printf '\n' >> "${ART}/serve.command.txt"
"${SERVE_CMD[@]}" > "${ART}/serve.log" 2>&1 &
SERVER_PID=$!
export SERVER_PID BASE_URL MODEL_ID

python3 - <<'PY'
import json, os, pathlib, time, urllib.request
art = pathlib.Path(os.environ["ART"])
base = os.environ["BASE_URL"].rstrip("/")
deadline = time.time() + 900
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as r:
            body = r.read().decode("utf-8", "replace")
            if r.status == 200:
                data = json.loads(body)
                (art / "serve_health.json").write_text(json.dumps(data, indent=2) + "\n")
                break
            last = f"status={r.status} body={body[:200]}"
    except Exception as exc:
        last = repr(exc)
    time.sleep(1)
else:
    raise SystemExit(f"server did not become healthy: {last}")
PY

python3 - <<'PY'
import json, pathlib, sys, os
art = pathlib.Path(os.environ["ART"])

def validate(path, label):
    doc = json.loads(path.read_text())
    entries = {entry["key"]: entry for entry in doc.get("entries", [])}
    def expect(key, value):
        actual = entries.get(key, {}).get("effective_value")
        if actual != value:
            raise SystemExit(f"{label}: {key}={actual!r}, expected {value!r}")
    expect("FERRUM_USE_VLLM_PAGED_ATTN", "1")
    expect("FERRUM_VLLM_PAGED_ATTN_V1_SHORT", "0")
    if doc.get("selected_attention_impl") != "vllm_paged_attn_v2":
        raise SystemExit(f"{label}: selected_attention_impl={doc.get('selected_attention_impl')!r}")
    model = doc.get("model_capabilities") or {}
    if model.get("architecture") != "qwen3_5_moe":
        raise SystemExit(f"{label}: architecture={model.get('architecture')!r}")
    if model.get("head_dim") != 256:
        raise SystemExit(f"{label}: head_dim={model.get('head_dim')!r}")
    return {"label": label, "selected_attention_impl": doc.get("selected_attention_impl"), "entries": {k: entries[k] for k in ["FERRUM_USE_VLLM_PAGED_ATTN", "FERRUM_VLLM_PAGED_ATTN_V1_SHORT", "FERRUM_VLLM_MOE"] if k in entries}}
summary = [validate(art / "serve_effective_config.json", "serve")]
(art / "effective_config_validation_serve.json").write_text(json.dumps({"status":"pass","checks":summary}, indent=2) + "\n")
PY

cat > "${ART}/requests/smoke_requests.py" <<'PY'
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
PY
python3 "${ART}/requests/smoke_requests.py" 2>&1 | tee "${ART}/requests/smoke_requests.log"


python3 - <<'POSTPY'
import json, os, pathlib, re
art = pathlib.Path(os.environ["ART"])
reqdir = art / "requests"

def read_status(name):
    path = reqdir / f"{name}.status.txt"
    if not path.exists():
        raise SystemExit(f"{name}: missing status file")
    status = path.read_text().strip()
    if status != "200":
        body_path = reqdir / f"{name}.body.txt"
        body = body_path.read_text(errors="replace") if body_path.exists() else ""
        raise SystemExit(f"{name}: HTTP {status}: {body[:500]}")

def reject_repetition(label, text):
    words = re.findall(r"[A-Za-z0-9_'-]+", str(text).lower())
    if len(words) >= 8:
        for size in (1, 2, 3):
            repeated = 1
            last = None
            for i in range(0, len(words) - size + 1, size):
                chunk = tuple(words[i:i + size])
                if chunk == last:
                    repeated += 1
                    if repeated >= 5:
                        raise SystemExit(f"{label}: repeated token chunk {chunk!r} in {text[:300]!r}")
                else:
                    repeated = 1
                    last = chunk

def reject_length_choice(label, choice):
    reason = choice.get("finish_reason")
    if reason == "length":
        raise SystemExit(f"{label}: finish_reason=length")

read_status("01_nonstream")
nonstream = json.loads((reqdir / "01_nonstream.body.txt").read_text(errors="replace"))
choice = nonstream["choices"][0]
reject_length_choice("nonstream", choice)
reject_repetition("nonstream", (choice.get("message") or {}).get("content") or "")

read_status("02_stream")
done = 0
usage_seen = False
content_parts = []
finish_reasons = []
for raw_line in (reqdir / "02_stream.body.txt").read_text(errors="replace").splitlines():
    line = raw_line.strip()
    if not line.startswith("data: "):
        continue
    payload = line[len("data: "):]
    if payload == "[DONE]":
        done += 1
        continue
    obj = json.loads(payload)
    if obj.get("usage"):
        usage_seen = True
    for stream_choice in obj.get("choices") or []:
        if stream_choice.get("finish_reason"):
            finish_reasons.append(stream_choice.get("finish_reason"))
        delta = stream_choice.get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
if done != 1:
    raise SystemExit(f"stream: DONE count {done}, expected 1")
if not usage_seen:
    raise SystemExit("stream: missing usage chunk")
if "length" in finish_reasons:
    raise SystemExit(f"stream: finish_reason=length in {finish_reasons!r}")
reject_repetition("stream", "".join(content_parts))

read_status("03_tool_call")
tool = json.loads((reqdir / "03_tool_call.body.txt").read_text(errors="replace"))
choice = tool["choices"][0]
reject_length_choice("tool_call", choice)
message = choice.get("message") or {}
if not message.get("tool_calls"):
    raise SystemExit(f"tool_call: missing tool_calls in {message}")

read_status("04_structured")
structured = json.loads((reqdir / "04_structured.body.txt").read_text(errors="replace"))
choice = structured["choices"][0]
reject_length_choice("structured", choice)
text = (choice.get("message") or {}).get("content") or ""
parsed = json.loads(text)
if parsed != {"answer": "scenario-ok"}:
    raise SystemExit(f"structured: parsed {parsed!r}")

(reqdir / "post_validation.json").write_text(json.dumps({
    "status": "pass",
    "stream_done_count": done,
    "stream_usage_seen": usage_seen,
    "stream_finish_reasons": finish_reasons,
    "nonstream_finish_reason": nonstream["choices"][0].get("finish_reason"),
    "tool_finish_reason": tool["choices"][0].get("finish_reason"),
    "structured_finish_reason": structured["choices"][0].get("finish_reason"),
}, indent=2) + "\n")
POSTPY

if [[ -f "${DATASET}" ]]; then
  BENCH_CMD=("${BIN}" bench-serve --base-url "${BASE_URL}" --model "${MODEL_ID}" --tokenizer "${MODEL}" --dataset sharegpt --sharegpt-path "${DATASET}" --num-prompts 8 --warmup-requests 1 --n-repeats 1 --concurrency-sweep 1,32 --fail-on-error --seed 9271 --output json --out "${ART}/bench_sharegpt_c1_c32.json")
else
  BENCH_CMD=("${BIN}" bench-serve --base-url "${BASE_URL}" --model "${MODEL_ID}" --tokenizer "${MODEL}" --dataset random --random-input-len 128 --random-output-len 32 --num-prompts 8 --warmup-requests 1 --n-repeats 1 --concurrency-sweep 1,32 --fail-on-error --seed 9271 --output json --out "${ART}/bench_random_c1_c32.json")
fi
printf '%q ' "${BENCH_CMD[@]}" > "${ART}/bench.command.txt"
printf '\n' >> "${ART}/bench.command.txt"
"${BENCH_CMD[@]}" 2>&1 | tee "${ART}/bench.log"

python3 - <<'PY'
import json, os, pathlib, datetime
art = pathlib.Path(os.environ["ART"])
summary = {
  "schema_version": 1,
  "status": "pass",
  "artifact_dir": str(art),
  "git_head": (art / "environment.log").read_text(errors="replace").split("git_head=")[1].splitlines()[0],
  "finished_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
  "run_effective_config": "run_effective_config.json",
  "serve_effective_config": "serve_effective_config.json",
  "requests_summary": "requests/summary.json",
}
(art / "summary.status.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

python3 - <<PY
import json, os, pathlib
art = pathlib.Path(os.environ["ART"])
path = art / "qwen35_prefill_profile.jsonl"
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit(f"missing or empty profile JSONL: {path}")
count = 0
slow_layers = []
for line in path.read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    if obj.get("event") == "qwen35_prefill_prof":
        count += 1
        stage = obj.get("stage_us") or {}
        for layer in stage.get("slow_layers") or []:
            slow_layers.append(layer)
if count < 1:
    raise SystemExit("profile JSONL has no qwen35_prefill_prof events")
(art / "profile_validation.json").write_text(json.dumps({
    "status": "pass",
    "qwen35_prefill_prof_events": count,
    "top_slow_layers": slow_layers[:16],
}, indent=2) + "\n")
PY
