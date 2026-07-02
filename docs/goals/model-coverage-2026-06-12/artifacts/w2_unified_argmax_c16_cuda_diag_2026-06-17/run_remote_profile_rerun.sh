#!/usr/bin/env bash
set -euo pipefail

OUT=${OUT:-/workspace/artifacts/w2_unified_argmax_c16_cuda_diag_2026-06-17}
REPO=${REPO:-/workspace/ferrum-infer-rs-run}
BIN=${BIN:-/workspace/ferrum-target/release/ferrum}
MODEL_PATH=${MODEL_PATH:-/workspace/hf-cache/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2}
PORT=${PORT:-18145}
export HF_HOME=${HF_HOME:-/workspace/hf-cache}

PROFILE_OUT="$OUT/profile_rerun_c16"
SERVER_DIR="$PROFILE_OUT/server"
BENCH_DIR="$PROFILE_OUT/c16_profile"
mkdir -p "$SERVER_DIR" "$BENCH_DIR"

stop_server() {
  local pid_file="$1"
  if [ -f "$pid_file" ]; then
    local pid
    pid=$(cat "$pid_file" || true)
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
      for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" || true
      fi
    fi
  fi
}

cleanup() {
  stop_server "$SERVER_DIR/server.pid"
}
trap cleanup EXIT

if [ ! -x "$BIN" ]; then
  echo "missing ferrum binary: $BIN" >&2
  exit 10
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "missing model path: $MODEL_PATH" >&2
  exit 11
fi
if [ ! -f "$OUT/ferrum.toml" ]; then
  echo "missing artifact ferrum.toml: $OUT/ferrum.toml" >&2
  exit 12
fi

date -u +"%Y-%m-%dT%H:%M:%SZ" >"$PROFILE_OUT/start_utc.txt"
printf '%s\n' "$PORT" >"$PROFILE_OUT/port.txt"
sha256sum "$BIN" >"$PROFILE_OUT/ferrum.sha256"
nvidia-smi >"$PROFILE_OUT/nvidia_smi_before.txt" || true
cp "$OUT/ferrum.toml" "$PROFILE_OUT/ferrum.cwd.toml"

if [ -d "$REPO/.git" ]; then
  git -C "$REPO" rev-parse HEAD >"$PROFILE_OUT/source_git_sha.txt"
  git -C "$REPO" status --short --untracked-files=no >"$PROFILE_OUT/source_git_status_short.txt" || true
fi

cd "$OUT"
pwd >"$PROFILE_OUT/cwd.txt"

if command -v ss >/dev/null 2>&1; then
  if ss -ltnp | grep -E ":${PORT}[[:space:]]" >"$PROFILE_OUT/port_busy.txt" 2>/dev/null; then
    echo "port already busy: $PORT" >&2
    exit 13
  fi
fi

commit_sha="unknown"
if [ -f "$PROFILE_OUT/source_git_sha.txt" ]; then
  commit_sha=$(cat "$PROFILE_OUT/source_git_sha.txt")
elif [ -f "$OUT/meta/git_sha.txt" ]; then
  commit_sha=$(cat "$OUT/meta/git_sha.txt")
fi

HF_HOME="$HF_HOME" "$BIN" serve \
  --model gemma3:27b-gptq \
  --backend cuda \
  --port "$PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 1024 \
  --effective-config-json "$SERVER_DIR/effective_config.json" \
  --decision-trace-jsonl "$SERVER_DIR/decision_trace.jsonl" \
  --profile-jsonl "$SERVER_DIR/profile.jsonl" \
  --profile-commit-sha "$commit_sha" \
  --profile-model gemma3-27b-gptq \
  --profile-concurrency 16 \
  >"$SERVER_DIR/server.stdout" 2>"$SERVER_DIR/server.stderr" &
echo $! >"$SERVER_DIR/server.pid"

for i in $(seq 1 300); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
    >"$SERVER_DIR/models.json" 2>"$SERVER_DIR/models_curl.err"; then
    echo "$i" >"$SERVER_DIR/ready_at_poll.txt"
    break
  fi
  if ! kill -0 "$(cat "$SERVER_DIR/server.pid")" 2>/dev/null; then
    echo "server exited while loading" >&2
    tail -n 180 "$SERVER_DIR/server.stderr" >&2 || true
    exit 20
  fi
  sleep 1
done
if [ ! -f "$SERVER_DIR/ready_at_poll.txt" ]; then
  echo "server ready timeout" >&2
  tail -n 180 "$SERVER_DIR/server.stderr" >&2 || true
  exit 21
fi

python3 - <<'PY'
import json
import pathlib
import sys

cfg_path = pathlib.Path("profile_rerun_c16/server/effective_config.json")
cfg = json.loads(cfg_path.read_text())
entries = {entry["key"]: entry for entry in cfg.get("entries", [])}
required = [
    "FERRUM_BATCH_DECODE_PROF",
    "FERRUM_NEXT_BATCH_PROF",
    "FERRUM_UNIFIED_POST_PROF",
    "FERRUM_DECODE_OP_PROFILE",
    "FERRUM_MARLIN_PROFILE",
    "FERRUM_GREEDY_ARGMAX",
]
missing = [key for key in required if entries.get(key, {}).get("effective_value") != "1"]
out = pathlib.Path("profile_rerun_c16/server/effective_profile_entries.json")
out.write_text(json.dumps({key: entries.get(key) for key in required}, indent=2, sort_keys=True))
if missing:
    pathlib.Path("profile_rerun_c16/server/profile_config_error.json").write_text(
        json.dumps({"missing_or_not_enabled": missing}, indent=2, sort_keys=True)
    )
    sys.exit(22)
PY

cat >"$SERVER_DIR/chat_payload.json" <<'JSON'
{
  "model": "gemma3:27b-gptq",
  "messages": [
    {"role": "user", "content": "What is 2+3? Answer with only the number."}
  ],
  "max_tokens": 8,
  "temperature": 0,
  "stream": true,
  "stream_options": {"include_usage": true}
}
JSON

curl -fsS -N \
  -H 'Content-Type: application/json' \
  -d @"$SERVER_DIR/chat_payload.json" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  >"$SERVER_DIR/chat_stream.sse" 2>"$SERVER_DIR/chat_stream.err"

python3 - <<'PY'
import pathlib
import sys

s = pathlib.Path("profile_rerun_c16/server/chat_stream.sse").read_text(errors="replace")
ok = s.count("data: [DONE]") == 1 and "5" in s and '"usage"' in s
pathlib.Path("profile_rerun_c16/server/smoke.ok").write_text(str(ok))
print("SERVE_SMOKE_OK", ok)
if not ok:
    print(s[:2000])
    sys.exit(32)
PY

set +e
HF_HOME="$HF_HOME" timeout 900 "$BIN" bench-serve \
  --base-url "http://127.0.0.1:${PORT}" \
  --model gemma3:27b-gptq \
  --tokenizer "$MODEL_PATH" \
  --dataset random \
  --random-input-len 64 \
  --random-output-len 16 \
  --concurrency 16 \
  --num-prompts 16 \
  --warmup-requests 4 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271 \
  --output json \
  --out "$BENCH_DIR/bench.json" \
  >"$BENCH_DIR/bench.stdout" 2>"$BENCH_DIR/bench.stderr"
bench_rc=$?
set -e
echo "$bench_rc" >"$BENCH_DIR/bench.rc"

nvidia-smi >"$BENCH_DIR/nvidia_smi_after.txt" || true
grep -nE \
  'first-token-prof|stream-ttft-prof|unified-op-profile|unified-prof|iter-prof|unified-decode|panic|CUDA_ERROR|illegal|OOM|out of memory|error|argmax|readback' \
  "$SERVER_DIR/server.stderr" \
  "$SERVER_DIR/server.stdout" \
  "$BENCH_DIR/bench.stderr" \
  "$BENCH_DIR/bench.stdout" \
  >"$PROFILE_OUT/log_scan.txt" 2>/dev/null || true
grep -nE \
  'unified-op-profile|readback|marlin_|panic|CUDA_ERROR|illegal|OOM|out of memory|error' \
  "$SERVER_DIR/server.stderr" \
  "$SERVER_DIR/server.stdout" \
  >"$PROFILE_OUT/profile_log_extract.txt" 2>/dev/null || true

python3 - <<'PY'
import json
import pathlib
import re

root = pathlib.Path("profile_rerun_c16")
bench = json.loads((root / "c16_profile/bench.json").read_text())
if isinstance(bench, list):
    bench = bench[0] if bench else {}
profile = (root / "profile_log_extract.txt").read_text(errors="replace")
rows = []
keys = [
    "m_total",
    "num_seqs",
    "prefill",
    "decode",
    "sampled",
    "total",
    "qkv",
    "qkr",
    "attn",
    "o_proj",
    "norm",
    "gate_up",
    "act",
    "down",
    "resid",
    "final_norm",
    "final_copy",
    "lm_head",
    "readback",
    "marlin_ws_zero",
    "marlin_gather",
    "marlin_kernel",
    "marlin_qkv_kernel",
    "marlin_o_proj_kernel",
    "marlin_gate_up_kernel",
    "marlin_down_kernel",
    "marlin_lm_head_kernel",
    "unwrapped",
]
for line in profile.splitlines():
    if "[unified-op-profile]" not in line:
        continue
    fields = {"line": line}
    call = re.search(r"call#(\d+)", line)
    if call:
        fields["call"] = int(call.group(1))
    for key in keys:
        match = re.search(rf"(?<![A-Za-z0-9_]){key}=([0-9]+)", line)
        if match:
            fields[key] = int(match.group(1))
    rows.append(fields)

target = None
for row in rows:
    if row.get("num_seqs") == 16 and row.get("prefill", 0) > 0 and row.get("decode", 0) > 0:
        if target is None or row.get("m_total", 0) > target.get("m_total", 0):
            target = row
if target is None:
    for row in rows:
        if row.get("num_seqs") == 16:
            if target is None or row.get("m_total", 0) > target.get("m_total", 0):
                target = row

summary = {
    "bench_rc": int((root / "c16_profile/bench.rc").read_text().strip()),
    "completed_per_run": bench.get("completed_per_run"),
    "errored_per_run": bench.get("errored_per_run"),
    "output_token_count_source": bench.get("output_token_count_source"),
    "request_throughput_rps": (bench.get("request_throughput_rps") or {}).get("mean"),
    "output_throughput_tps": (bench.get("output_throughput_tps") or {}).get("mean"),
    "ttft_p50_ms": ((bench.get("ttft_ms") or {}).get("p50") or {}).get("mean"),
    "ttft_p95_ms": ((bench.get("ttft_ms") or {}).get("p95") or {}).get("mean"),
    "itl_p95_ms": ((bench.get("itl_ms") or {}).get("p95") or {}).get("mean"),
    "unified_profile_row_count": len(rows),
    "unified_profile_rows": rows[-20:],
    "target_mixed_frame": target,
    "previous_target_readback_us": 22039,
    "previous_output_tps": 158.877,
}
if target and isinstance(target.get("readback"), int):
    summary["readback_delta_us_vs_previous"] = target["readback"] - 22039
    summary["readback_ratio_vs_previous"] = target["readback"] / 22039
if isinstance(summary["output_throughput_tps"], (int, float)):
    summary["throughput_delta_tps_vs_previous"] = summary["output_throughput_tps"] - 158.877
    summary["throughput_ratio_vs_previous"] = summary["output_throughput_tps"] / 158.877
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))

if summary["bench_rc"] != 0:
    raise SystemExit(summary["bench_rc"])
if summary["unified_profile_row_count"] == 0:
    raise SystemExit(33)
PY

date -u +"%Y-%m-%dT%H:%M:%SZ" >"$PROFILE_OUT/end_utc.txt"
cleanup
trap - EXIT
