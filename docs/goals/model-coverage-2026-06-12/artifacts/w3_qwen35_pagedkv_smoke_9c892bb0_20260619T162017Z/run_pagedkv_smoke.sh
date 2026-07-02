#!/usr/bin/env bash
set -euo pipefail
ART="$(cd "$(dirname "$0")" && pwd)"
REPO=/workspace/ferrum-w3-qwen35-8cfa422c
TARGET_DIR=/workspace/ferrum-w3-qwen35-ac98207/target
MODEL=/workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b
MODEL_ID=3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b
DATASET=/workspace/artifacts/ferrum_w3_sharegpt_ab_20b8946b_20260619T121844Z_ninja/dataset/ascii_sharegpt_w3_100_58d5721d.jsonl
PORT=58081
SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
if [[ -f /root/.cargo/env ]]; then
  source /root/.cargo/env
fi
export CARGO_TARGET_DIR="$TARGET_DIR"
export HF_HOME=/workspace/hf-cache
export RUST_BACKTRACE=1
export RUST_LOG=${RUST_LOG:-info}
cat > "$ART/gpu_contract.json" <<JSON
{
  "lane": "W3 Qwen3.5 GPTQ-Int4 paged-KV architecture smoke on 1x RTX 4090",
  "expected_runtime": "20-45 minutes",
  "expected_cost": "existing Vast instance runtime only; stop after PASS or first actionable failure",
  "stop_condition": "CUDA build failure, ferrum run failure, ferrum serve/stream failure, bench-serve failure, or diagnostic smoke PASS",
  "correctness_gate": "CUDA release build plus ferrum run one-shot plus ferrum serve non-stream and streaming include_usage smoke",
  "performance_command": "diagnostic-only ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1 --concurrency-sweep 1,32; no release performance claim"
}
JSON
{
  echo "artifact=$ART"
  date -u +%Y-%m-%dT%H:%M:%SZ
  echo "repo=$REPO"
  echo "target_dir=$TARGET_DIR"
  echo "model=$MODEL"
  echo "dataset=$DATASET"
} > "$ART/remote/start.txt"
cd "$REPO"
git rev-parse HEAD > "$ART/remote/git_head.txt"
git status --short > "$ART/remote/git_status_short.txt"
rustc --version > "$ART/remote/rustc_version.txt"
cargo --version > "$ART/remote/cargo_version.txt"
nvcc --version > "$ART/remote/nvcc_version.txt" 2>&1
nvidia-smi > "$ART/remote/nvidia_smi_before.txt"
sha256sum "$DATASET" > "$ART/remote/dataset.sha256"
wc -l "$DATASET" > "$ART/remote/dataset.wc"
BUILD_CMD=(cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source)
printf "%q " "${BUILD_CMD[@]}" > "$ART/build/release_build.command.txt"
echo >> "$ART/build/release_build.command.txt"
set +e
"${BUILD_CMD[@]}" > "$ART/build/release_build.log" 2>&1
build_rc=$?
set -e
echo "$build_rc" > "$ART/build/release_build.rc"
if [[ "$build_rc" -ne 0 ]]; then
  echo "W3 QWEN35 PAGEDKV SMOKE FAIL(build): $ART" > "$ART/run.status"
  tail -240 "$ART/build/release_build.log" > "$ART/build/release_build.tail.txt" || true
  exit "$build_rc"
fi
BIN="$TARGET_DIR/release/ferrum"
sha256sum "$BIN" > "$ART/build/ferrum.sha256"
RUN_CMD=(timeout 900 "$BIN" run "$MODEL" --backend cuda --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 4096 --max-tokens 32 --prompt "Answer in one short sentence: Paris is the capital of which country?" --temperature 0 --repeat-penalty 1.0 --output-format jsonl --effective-config-json "$ART/run/run_effective_config.json" --decision-trace-jsonl "$ART/run/run_decision_trace.jsonl")
printf "%q " "${RUN_CMD[@]}" > "$ART/run/ferrum_run.command.txt"
echo >> "$ART/run/ferrum_run.command.txt"
set +e
"${RUN_CMD[@]}" > "$ART/run/ferrum_run.stdout" 2> "$ART/run/ferrum_run.stderr"
run_rc=$?
set -e
echo "$run_rc" > "$ART/run/ferrum_run.rc"
python3 - "$ART/run/ferrum_run.stdout" "$ART/run/run_validation.json" <<'PY'
import json, pathlib, sys
stdout = pathlib.Path(sys.argv[1]).read_text(errors="replace")
out = {"non_empty": bool(stdout.strip()), "bad_patterns": []}
for pat in ["<unk>", "[PAD]", "panic", "invalid utf-8", "mojibake"]:
    if pat.lower() in stdout.lower():
        out["bad_patterns"].append(pat)
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
if not out["non_empty"] or out["bad_patterns"]:
    raise SystemExit(2)
PY
if [[ "$run_rc" -ne 0 ]]; then
  echo "W3 QWEN35 PAGEDKV SMOKE FAIL(run): $ART" > "$ART/run.status"
  exit "$run_rc"
fi
SERVE_CMD=("$BIN" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --backend cuda --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 4096 --greedy-argmax --effective-config-json "$ART/server/serve_effective_config.json" --decision-trace-jsonl "$ART/server/serve_decision_trace.jsonl")
printf "%q " "${SERVE_CMD[@]}" > "$ART/server/ferrum_server.command.txt"
echo >> "$ART/server/ferrum_server.command.txt"
"${SERVE_CMD[@]}" > "$ART/server/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$ART/server/server.pid"
ready=0
for i in $(seq 1 420); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > "$ART/server/models.json" 2> "$ART/server/models.err"; then
    ready=1
    echo "$i" > "$ART/server/ready_poll.txt"
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "W3 QWEN35 PAGEDKV SMOKE FAIL(server-exit): $ART" > "$ART/run.status"
    tail -240 "$ART/server/server.log" > "$ART/server/server.tail.txt" || true
    exit 20
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  echo "W3 QWEN35 PAGEDKV SMOKE FAIL(server-not-ready): $ART" > "$ART/run.status"
  tail -240 "$ART/server/server.log" > "$ART/server/server.tail.txt" || true
  exit 21
fi
cat > "$ART/smoke/nonstream_request.json" <<JSON
{
  "model": "$MODEL_ID",
  "messages": [{"role": "user", "content": "Answer in one short sentence: What is the capital of France?"}],
  "max_tokens": 32,
  "temperature": 0
}
JSON
curl -sS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" --data-binary "@$ART/smoke/nonstream_request.json" > "$ART/smoke/nonstream_response.json" 2> "$ART/smoke/nonstream_curl.err"
python3 - "$ART/smoke/nonstream_response.json" "$ART/smoke/nonstream_validation.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
data = json.loads(p.read_text())
text = json.dumps(data, ensure_ascii=False)
content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
out = {"has_content": bool(content.strip()), "content_preview": content[:200], "bad_patterns": []}
for pat in ["<unk>", "[PAD]", "panic", "mojibake"]:
    if pat.lower() in text.lower():
        out["bad_patterns"].append(pat)
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
if not out["has_content"] or out["bad_patterns"]:
    raise SystemExit(2)
PY
cat > "$ART/smoke/stream_request.json" <<JSON
{
  "model": "$MODEL_ID",
  "messages": [{"role": "user", "content": "Answer in one short sentence: What color is the sky on a clear day?"}],
  "max_tokens": 32,
  "temperature": 0,
  "stream": true,
  "stream_options": {"include_usage": true}
}
JSON
curl -sS -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" --data-binary "@$ART/smoke/stream_request.json" > "$ART/smoke/stream_response.sse" 2> "$ART/smoke/stream_curl.err"
python3 - "$ART/smoke/stream_response.sse" "$ART/smoke/stream_validation.json" <<'PY'
import json, pathlib, sys
text = pathlib.Path(sys.argv[1]).read_text(errors="replace")
done = 0
chunks = 0
content = []
usage = None
malformed = 0
for line in text.splitlines():
    if not line.startswith("data: "):
        continue
    payload = line[6:]
    if payload == "[DONE]":
        done += 1
        continue
    try:
        obj = json.loads(payload)
    except Exception:
        malformed += 1
        continue
    chunks += 1
    if obj.get("usage") is not None:
        usage = obj.get("usage")
    for choice in obj.get("choices", []):
        delta = choice.get("delta", {})
        if isinstance(delta.get("content"), str):
            content.append(delta["content"])
out_text = "".join(content)
out = {
    "done_count": done,
    "chunks": chunks,
    "has_output": bool(out_text.strip()),
    "usage_present": usage is not None,
    "malformed": malformed,
    "content_preview": out_text[:200],
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
if done != 1 or not out["has_output"] or not out["usage_present"] or malformed:
    raise SystemExit(2)
PY
nvidia-smi > "$ART/perf/nvidia_smi_before_bench.txt"
BENCH_CMD=(timeout 1800 "$BIN" bench-serve --base-url "http://127.0.0.1:$PORT" --model "$MODEL_ID" --tokenizer "$MODEL" --dataset sharegpt --sharegpt-path "$DATASET" --random-output-len 128 --concurrency-sweep 1,32 --num-prompts 100 --warmup-requests 10 --n-repeats 1 --fail-on-error --seed 9271 --output json --out "$ART/perf/bench_ferrum_sharegpt_c1_c32_100x1.json")
printf "%q " "${BENCH_CMD[@]}" > "$ART/perf/bench-ferrum.command.txt"
echo >> "$ART/perf/bench-ferrum.command.txt"
set +e
"${BENCH_CMD[@]}" > "$ART/perf/bench-ferrum.stdout" 2> "$ART/perf/bench-ferrum.stderr"
bench_rc=$?
set -e
echo "$bench_rc" > "$ART/perf/bench-ferrum.rc"
curl -sf "http://127.0.0.1:$PORT/health" > "$ART/server/health_after_bench.json" 2> "$ART/server/health_after_bench.err" || true
nvidia-smi > "$ART/perf/nvidia_smi_after_bench.txt"
python3 - "$ART" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
out = {"artifact": str(root)}
for name in ["run/ferrum_run.rc", "perf/bench-ferrum.rc"]:
    p = root / name
    out[name] = int(p.read_text().strip()) if p.exists() else None
for name in ["run/run_validation.json", "smoke/nonstream_validation.json", "smoke/stream_validation.json"]:
    p = root / name
    if p.exists():
        out[name] = json.loads(p.read_text())
bench = root / "perf/bench_ferrum_sharegpt_c1_c32_100x1.json"
if bench.exists():
    data = json.loads(bench.read_text())
    out["bench_keys"] = sorted(data)[:40]
    out["bench"] = data
path = root / "perf/diagnostic_summary.json"
path.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
PY
if [[ "$bench_rc" -ne 0 ]]; then
  echo "W3 QWEN35 PAGEDKV SMOKE FAIL(bench): $ART" > "$ART/run.status"
  exit "$bench_rc"
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$ART/remote/end_utc.txt"
echo "W3 QWEN35 PAGEDKV DIAG SMOKE PASS: $ART" > "$ART/run.status"
