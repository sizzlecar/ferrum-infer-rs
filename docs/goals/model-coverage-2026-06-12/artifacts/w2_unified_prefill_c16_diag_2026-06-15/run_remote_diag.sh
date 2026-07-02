#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/ferrum-infer-rs}"
OUT="${OUT:-/workspace/w2_unified_prefill_c16_diag_2026-06-15}"
FERRUM="${FERRUM:-$REPO/target/release/ferrum}"
MODEL="${MODEL:-gemma3:27b-gptq}"
PORT="${PORT:-8492}"
TOKENIZER="${TOKENIZER:-/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2}"
DATASET="${DATASET:-/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl}"
BASE="http://127.0.0.1:${PORT}"
VLLM_C16_TPS="${VLLM_C16_TPS:-518.7959572662905}"

mkdir -p \
  "$OUT/build" \
  "$OUT/server" \
  "$OUT/smoke" \
  "$OUT/bench" \
  "$OUT/remote"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT/remote/start_utc.txt"
cd "$REPO"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short.txt"
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"
nvcc --version > "$OUT/remote/nvcc_version.txt" 2>&1 || true

if [[ -f /root/.cargo/env ]]; then
  # shellcheck disable=SC1091
  source /root/.cargo/env
fi

set +e
timeout 1500 cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$OUT/build/cargo_build.stdout" \
  2> "$OUT/build/cargo_build.stderr"
BUILD_RC=$?
set -e
echo "$BUILD_RC" > "$OUT/build/cargo_build.rc"
if [[ "$BUILD_RC" -ne 0 ]]; then
  echo "FAIL" > "$OUT/run.status"
  exit "$BUILD_RC"
fi

sha256sum "$FERRUM" > "$OUT/remote/ferrum.sha256"
"$FERRUM" --version > "$OUT/remote/ferrum_version.txt" 2>&1 || true

pkill -f "ferrum serve.*${PORT}" >/dev/null 2>&1 || true
sleep 1

python3 - "$OUT/server/serve.command.json" "$FERRUM" "$MODEL" "$PORT" "$OUT" <<'PY'
import json
import sys

path, ferrum, model, port, out = sys.argv[1:]
cmd = [
    ferrum,
    "serve",
    "--model",
    model,
    "--port",
    port,
    "--kv-capacity",
    "512",
    "--max-num-seqs",
    "16",
    "--effective-config-json",
    f"{out}/server/serve_effective_config.json",
    "--decision-trace-jsonl",
    f"{out}/server/serve_decision_trace.jsonl",
]
with open(path, "w", encoding="utf-8") as f:
    json.dump({"cmd": cmd}, f)
    f.write("\n")
PY

"$FERRUM" serve \
  --model "$MODEL" \
  --port "$PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/server/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl" \
  > "$OUT/server/server.log" \
  2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server/server.pid"

: > "$OUT/server/ready_poll.txt"
READY=0
for i in $(seq 1 300); do
  if curl -sf --noproxy '*' --max-time 2 "$BASE/health" > /dev/null 2>&1; then
    READY=1
    echo "ready_poll=$i" >> "$OUT/server/ready_poll.txt"
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "server exited during startup at poll=$i" >> "$OUT/server/ready_poll.txt"
    echo "FAIL" > "$OUT/run.status"
    exit 1
  fi
  sleep 2
done
if [[ "$READY" -ne 1 ]]; then
  echo "server not ready" >> "$OUT/server/ready_poll.txt"
  echo "FAIL" > "$OUT/run.status"
  exit 1
fi

curl -sf --noproxy '*' --max-time 10 "$BASE/v1/models" \
  > "$OUT/server/models.json" \
  2> "$OUT/server/models.err"

BASE="$BASE" MODEL="$MODEL" OUT="$OUT" python3 - <<'PY'
import json
import os
import sys
import urllib.request

base = os.environ["BASE"]
model = os.environ["MODEL"]
out = os.environ["OUT"]

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
urllib.request.install_opener(opener)

payload = {
    "model": model,
    "temperature": 0,
    "max_tokens": 16,
    "messages": [
        {
            "role": "user",
            "content": "What is 2+3? Answer with just the number.",
        }
    ],
}
req = urllib.request.Request(
    f"{base}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as r:
    data = json.load(r)

with open(f"{out}/smoke/chat_response.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=True)
    f.write("\n")

choice = (data.get("choices") or [{}])[0]
message = choice.get("message") or {}
content = message.get("content") or ""
usage = data.get("usage") or {}
bad_markers = ["<unk>", "[PAD", "\ufffd"]
validation = {
    "content": content,
    "finish_reason": choice.get("finish_reason"),
    "completion_tokens": usage.get("completion_tokens"),
    "usage_present": bool(usage),
    "bad_output": any(m in content for m in bad_markers),
}
validation["pass"] = (
    bool(content.strip())
    and not validation["bad_output"]
    and isinstance(validation["completion_tokens"], int)
    and validation["completion_tokens"] > 0
)
with open(f"{out}/smoke/chat_validation.json", "w", encoding="utf-8") as f:
    json.dump(validation, f, indent=2, ensure_ascii=True)
    f.write("\n")

if not validation["pass"]:
    sys.exit(1)
PY

python3 - "$OUT/bench/bench-serve.command.json" "$FERRUM" "$BASE" "$MODEL" "$TOKENIZER" "$DATASET" "$OUT" <<'PY'
import json
import sys

path, ferrum, base, model, tokenizer, dataset, out = sys.argv[1:]
cmd = [
    "timeout",
    "1200",
    ferrum,
    "bench-serve",
    "--base-url",
    base,
    "--model",
    model,
    "--tokenizer",
    tokenizer,
    "--dataset",
    "sharegpt",
    "--sharegpt-path",
    dataset,
    "--random-output-len",
    "64",
    "--concurrency-sweep",
    "16",
    "--num-prompts",
    "16",
    "--n-repeats",
    "1",
    "--fail-on-error",
    "--seed",
    "9271",
    "--out",
    f"{out}/bench/bench_ferrum_unified_prefill_c16_16x1.json",
]
with open(path, "w", encoding="utf-8") as f:
    json.dump({"cmd": cmd}, f)
    f.write("\n")
PY

set +e
timeout 1200 "$FERRUM" bench-serve \
  --base-url "$BASE" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --dataset sharegpt \
  --sharegpt-path "$DATASET" \
  --random-output-len 64 \
  --concurrency-sweep 16 \
  --num-prompts 16 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271 \
  --out "$OUT/bench/bench_ferrum_unified_prefill_c16_16x1.json" \
  > "$OUT/bench/bench-serve.stdout" \
  2> "$OUT/bench/bench-serve.stderr"
BENCH_RC=$?
set -e
echo "$BENCH_RC" > "$OUT/bench/bench-serve.rc"

nvidia-smi > "$OUT/remote/nvidia_smi_after_bench.txt" || true

python3 - "$OUT" "$VLLM_C16_TPS" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
vllm = float(sys.argv[2])
bench_path = out / "bench" / "bench_ferrum_unified_prefill_c16_16x1.json"
bench_rc = int((out / "bench" / "bench-serve.rc").read_text().strip())
chat_validation = json.loads((out / "smoke" / "chat_validation.json").read_text())
server_log = (out / "server" / "server.log").read_text(errors="replace")
reports = json.loads(bench_path.read_text()) if bench_path.exists() else []
if isinstance(reports, dict):
    reports = [reports]
report = reports[0] if reports else {}
tps = (((report.get("output_throughput_tps") or {}).get("mean")) or 0.0)
ratio = (tps / vllm) if vllm else 0.0
summary = {
    "status": "PASS" if bench_rc == 0 and chat_validation.get("pass") else "FAIL",
    "bench_rc": bench_rc,
    "chat_pass": bool(chat_validation.get("pass")),
    "varlen_unified_in_log": "varlen_unified=true" in server_log,
    "concurrency": report.get("concurrency"),
    "n_repeats": report.get("n_repeats"),
    "n_requests_per_run": report.get("n_requests_per_run"),
    "completed_per_run": report.get("completed_per_run"),
    "errored_per_run": report.get("errored_per_run"),
    "bad_output_per_run": report.get("bad_output_per_run"),
    "output_token_count_source": report.get("output_token_count_source"),
    "output_throughput_tps_mean": tps,
    "vllm_c16_tps_baseline": vllm,
    "ferrum_vs_vllm_ratio": ratio,
    "ferrum_vs_vllm_percent": ratio * 100.0,
}
with (out / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=True)
    f.write("\n")
(out / "run.status").write_text(summary["status"] + "\n")
if summary["status"] != "PASS":
    sys.exit(1)
PY

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT/remote/end_utc.txt"
exit "$BENCH_RC"
