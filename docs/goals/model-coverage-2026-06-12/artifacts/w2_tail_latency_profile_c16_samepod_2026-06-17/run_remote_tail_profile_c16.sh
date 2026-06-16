#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/ferrum-infer-rs}"
OUT="${OUT:-/workspace/w2_tail_latency_profile_c16_samepod_2026-06-17}"
MODEL="${MODEL:-gemma3:27b-gptq}"
DATASET="${DATASET:-/workspace/ascii_sharegpt_w2_100.jsonl}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
PORT="${PORT:-18149}"
BIN="$ROOT/target/release/ferrum"

mkdir -p "$OUT"/{env,server,smoke,perf,analysis}
exec > >(tee -a "$OUT/run.log") 2>&1

SERVER_PID=""
cleanup_server() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT

wait_http() {
  local base="$1"
  local log_file="$2"
  local label="$3"
  for _ in $(seq 1 240); do
    if curl -sf --noproxy '*' "$base/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      echo "[tail-profile] $label server exited before readiness" >&2
      tail -160 "$log_file" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "[tail-profile] timed out waiting for $label server" >&2
  tail -160 "$log_file" >&2 || true
  return 1
}

run_stream_smoke() {
  local base="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  python3 - "$MODEL" "$out_dir/stream_request.json" <<'PY'
import json
import sys
from pathlib import Path

model = sys.argv[1]
path = Path(sys.argv[2])
payload = {
    "model": model,
    "messages": [
        {"role": "user", "content": "What is 2+3? Answer with just the number."}
    ],
    "max_tokens": 64,
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  timeout 300 curl -sS --no-buffer --noproxy '*' \
    -H 'content-type: application/json' \
    -X POST "$base/v1/chat/completions" \
    --data @"$out_dir/stream_request.json" \
    > "$out_dir/stream_response.sse"
  python3 - "$out_dir" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
content = []
done_count = 0
usage = None
malformed = []
for raw in (out / "stream_response.sse").read_text(errors="replace").splitlines():
    line = raw.strip()
    if not line.startswith("data:"):
        continue
    data = line[5:].strip()
    if data == "[DONE]":
        done_count += 1
        continue
    if not data:
        continue
    try:
        obj = json.loads(data)
    except Exception as exc:
        malformed.append({"line": data, "error": str(exc)})
        continue
    if obj.get("usage") is not None:
        usage = obj.get("usage")
    for choice in obj.get("choices") or []:
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            content.append(delta["content"])
        message = choice.get("message") or {}
        if isinstance(message.get("content"), str):
            content.append(message["content"])

text = "".join(content)
summary = {
    "content": text,
    "done_count": done_count,
    "usage": usage,
    "malformed_count": len(malformed),
    "malformed": malformed[:5],
}
(out / "stream_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
if malformed:
    raise SystemExit("stream smoke had malformed SSE JSON")
if done_count != 1:
    raise SystemExit(f"expected exactly one DONE, got {done_count}")
if "5" not in text.strip():
    raise SystemExit(f"expected answer containing 5, got {text!r}")
if not usage or int(usage.get("completion_tokens") or 0) <= 0:
    raise SystemExit(f"expected positive usage completion tokens, got {usage!r}")
PY
}

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT/env/start_utc.txt"
nvidia-smi > "$OUT/env/nvidia_smi_before.txt"
nvcc --version > "$OUT/env/nvcc_version.txt" 2>&1 || true
df -h /workspace / > "$OUT/env/df_before.txt" 2>&1 || true
sha256sum "$DATASET" > "$OUT/env/dataset.sha256"
wc -l "$DATASET" > "$OUT/env/dataset.wc"

cd "$ROOT"
git rev-parse HEAD > "$OUT/env/git_sha.txt"
git status --short > "$OUT/env/git_status_short.txt"
test -x "$BIN"
"$BIN" --version > "$OUT/env/ferrum_version.txt" 2>&1 || true
sha256sum "$BIN" > "$OUT/env/ferrum.sha256"

MODEL_ROOT="$HF_HOME/hub/models--circulus--gemma-3-27b-it-gptq"
MODEL_PATH="$(find "$MODEL_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
  echo "[tail-profile] missing model snapshot under $MODEL_ROOT/snapshots" >&2
  exit 20
fi
printf '%s\n' "$MODEL_PATH" > "$OUT/env/model_path.txt"
find "$MODEL_PATH" -maxdepth 1 -type f -printf '%f\n' | sort > "$OUT/env/model_files.txt"
du -sh "$HF_HOME" > "$OUT/env/hf_home_du.txt" 2>&1 || true

echo "[tail-profile] starting Ferrum serve with FERRUM_DECODE_OP_PROFILE=1"
HF_HOME="$HF_HOME" FERRUM_DECODE_OP_PROFILE=1 "$BIN" serve "$MODEL" \
  --backend cuda \
  --port "$PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/server/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl" \
  > "$OUT/server/server.log" 2>&1 &
SERVER_PID=$!
BASE="http://127.0.0.1:$PORT"
wait_http "$BASE" "$OUT/server/server.log" "Ferrum profile"
curl -sf --noproxy '*' "$BASE/v1/models" > "$OUT/server/models.json"
python3 - <<'PY' "$OUT/server/serve_effective_config.json" "$OUT/server/active_chunk_check.json"
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
matches = []
def walk(node):
    if isinstance(node, dict):
        if node.get("key") == "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK":
            matches.append(node)
        for value in node.values():
            walk(value)
    elif isinstance(node, list):
        for value in node:
            walk(value)
walk(data)
Path(sys.argv[2]).write_text(json.dumps({"matches": matches}, indent=2, sort_keys=True) + "\n")
if not any(str(m.get("effective_value")) == "16" for m in matches):
    raise SystemExit("expected typed/default active decode prefill chunk value 16")
PY

run_stream_smoke "$BASE" "$OUT/smoke"

BENCH="$OUT/perf/bench_ferrum_profile_c16_100x1.json"
BENCH_CMD=(
  timeout 7200 "$BIN" bench-serve
  --base-url "$BASE"
  --model "$MODEL"
  --tokenizer "$MODEL_PATH"
  --dataset sharegpt
  --sharegpt-path "$DATASET"
  --random-output-len 128
  --concurrency-sweep 16
  --num-prompts 100
  --n-repeats 1
  --fail-on-error
  --seed 9271
  --out "$BENCH"
)
printf '%q ' "${BENCH_CMD[@]}" > "$OUT/perf/bench-ferrum-profile.command.txt"
printf '\n' >> "$OUT/perf/bench-ferrum-profile.command.txt"
"${BENCH_CMD[@]}" > "$OUT/perf/bench-ferrum-profile.stdout" 2> "$OUT/perf/bench-ferrum-profile.stderr"
nvidia-smi > "$OUT/perf/nvidia_smi_after_bench.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT/env/end_utc.txt"
df -h /workspace / > "$OUT/env/df_after.txt" 2>&1 || true

: <<'PY'
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
bench_path = Path(sys.argv[2])
log_path = Path(sys.argv[3])
smoke_path = Path(sys.argv[4])

bench = json.loads(bench_path.read_text())
smoke = json.loads(smoke_path.read_text())
row_re = re.compile(r"\[batched-op-profile\]\s+m=(\d+)\s+total=(\d+)us\s+(.*)$")
kv_re = re.compile(r"([a-z_]+)=(\d+)us(?:\((\d+)\))?")
rows = []
for line in log_path.read_text(errors="replace").splitlines():
    m = row_re.search(line)
    if not m:
        continue
    row = {"m": int(m.group(1)), "total_us": int(m.group(2))}
    for key, val, count in kv_re.findall(m.group(3)):
        row[f"{key}_us"] = int(val)
        if count:
            row[f"{key}_calls"] = int(count)
    rows.append(row)

def read_rel(path):
    p = out / path
    return p.read_text().strip() if p.exists() else None

def percentile(sorted_values, pct):
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, math.ceil((pct / 100.0) * len(sorted_values)) - 1))
    return sorted_values[idx]

def stat(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return {
        "count": len(vals),
        "mean": statistics.fmean(vals),
        "p50": percentile(vals, 50),
        "p95": percentile(vals, 95),
        "max": vals[-1],
    }

keys = [
    "total",
    "matmul",
    "attn",
    "qkr",
    "norm",
    "other",
    "tail_norm",
    "tail_mlp",
    "tail_gate_up",
    "tail_down",
    "marlin_ws_zero",
    "marlin_gather",
    "marlin_kernel",
    "tail_act",
    "tail_resid",
    "unwrapped",
]

by_m = defaultdict(list)
for row in rows:
    by_m[str(row["m"])].append(row)

profile_by_m = {}
for m, group in sorted(by_m.items(), key=lambda item: int(item[0])):
    entry = {"count": len(group)}
    for key in keys:
        field = "total_us" if key == "total" else f"{key}_us"
        entry[f"{key}_us"] = stat([row.get(field) for row in group])
    total_sum = sum(row.get("total_us", 0) for row in group)
    if total_sum > 0:
        entry["share_of_total"] = {
            key: sum(row.get(f"{key}_us", 0) for row in group) / total_sum
            for key in keys
            if key != "total"
        }
    profile_by_m[m] = entry

def metric_mean(name):
    node = bench.get(name)
    return node.get("mean") if isinstance(node, dict) else None

def p95_itl():
    node = bench.get("itl_ms")
    if isinstance(node, dict):
        p95 = node.get("p95")
        if isinstance(p95, dict):
            return p95.get("mean")
    return None

quality = bench.get("quality_issues_per_run") or []
quality_nonzero = []
for idx, item in enumerate(quality):
    if isinstance(item, dict):
        nz = {k: v for k, v in item.items() if v}
        if nz:
            quality_nonzero.append({"run": idx, "issues": nz})

summary = {
    "lane": "w2_tail_latency_profile_c16_samepod",
    "status": "diagnostic_pass",
    "release_gate": False,
    "remote_git_sha": read_rel("env/git_sha.txt"),
    "remote_git_status_short": read_rel("env/git_status_short.txt"),
    "ferrum_binary_sha256": read_rel("env/ferrum.sha256"),
    "dataset_sha256": read_rel("env/dataset.sha256"),
    "model_path": read_rel("env/model_path.txt"),
    "smoke": smoke,
    "bench_completed_per_run": bench.get("completed_per_run"),
    "bench_errored_per_run": bench.get("errored_per_run"),
    "bench_quality_issues_per_run": bench.get("quality_issues_per_run"),
    "bench_quality_nonzero": quality_nonzero,
    "output_token_count_source": bench.get("output_token_count_source"),
    "output_tps_mean": metric_mean("output_throughput_tps"),
    "itl_p95_ms_mean": p95_itl(),
    "profile_rows": len(rows),
    "profile_by_m": profile_by_m,
    "same_pod_reference": {
        "vllm_c16_output_tps_mean": 500.67038762731977,
        "vllm_c16_output_tps_lcb": 478.39462812583776,
        "vllm_c16_p95_itl_ms_mean": 33.06958213333332,
        "ferrum_c16_output_tps_mean": 422.34520497237537,
        "ferrum_c16_output_tps_lcb": 414.59153186899397,
        "ferrum_c16_p95_itl_ms_mean": 52.81935383333333,
    },
}

if any(x != 0 for x in bench.get("errored_per_run", [])):
    summary["status"] = "diagnostic_fail"
if quality_nonzero:
    summary["status"] = "diagnostic_fail"
if bench.get("output_token_count_source") != "usage":
    summary["status"] = "diagnostic_fail"
if not rows:
    summary["status"] = "diagnostic_fail"

(out / "analysis/profile_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

m16 = profile_by_m.get("16", {})
shares = m16.get("share_of_total") or {}
tail = m16.get("tail_mlp_us") or {}
total = m16.get("total_us") or {}
lines = [
    "# W2 Tail Profile c16 Diagnostic",
    "",
    f"- Status: `{summary['status']}`",
    "- Release gate: no",
    f"- Profile rows: `{len(rows)}`",
    f"- Bench completed/errors: `{bench.get('completed_per_run')}` / `{bench.get('errored_per_run')}`",
    f"- Output token count source: `{bench.get('output_token_count_source')}`",
    f"- Output throughput mean: `{summary['output_tps_mean']}` tok/s",
    f"- p95 ITL mean: `{summary['itl_p95_ms_mean']}` ms",
    f"- m=16 total_us mean/p95/max: `{total.get('mean')}` / `{total.get('p95')}` / `{total.get('max')}`",
    f"- m=16 tail_mlp_us mean/p95/max: `{tail.get('mean')}` / `{tail.get('p95')}` / `{tail.get('max')}`",
    f"- m=16 tail_mlp share: `{shares.get('tail_mlp')}`",
    f"- m=16 tail_gate_up share: `{shares.get('tail_gate_up')}`",
    f"- m=16 tail_down share: `{shares.get('tail_down')}`",
    f"- m=16 attention share: `{shares.get('attn')}`",
    "",
    "This artifact is diagnostic only because profile logging changes runtime cost.",
]
(out / "analysis/summary.md").write_text("\n".join(lines) + "\n")

if summary["status"] != "diagnostic_pass":
    raise SystemExit("tail profile diagnostic failed; see analysis/profile_summary.json")
PY

POSTPROCESS="${POSTPROCESS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/postprocess_tail_profile_c16.py}"
python3 "$POSTPROCESS" "$OUT"
echo "W2 TAIL PROFILE C16 DIAGNOSTIC PASS: $OUT"
