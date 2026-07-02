#!/usr/bin/env bash
set -euo pipefail

ART="${ART:-/workspace/w2_typed_prefill_profile_2026-06-16}"
SRC="${SRC:-/workspace/ferrum-infer-rs}"
RUN_CWD="$ART/run_cwd"
FERRUM="$SRC/target/release/ferrum"
MODEL="${MODEL:-gemma3:27b-gptq}"
TOKENIZER="${TOKENIZER:-/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2}"
DATASET="${DATASET:-/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl}"
PORT="${PORT:-18117}"
BASE="http://127.0.0.1:${PORT}"
VLLM_C16_TPS="${VLLM_C16_TPS:-518.7959572662905}"
SERVER_PID=""

cleanup() {
  set +e
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    pgrep -af "$FERRUM serve .*--port $PORT" > "$ART/server/server_ps_before_stop.txt" || true
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
  fi
  pkill -f "$FERRUM serve .*--port $PORT" 2>/dev/null || true
  pgrep -af "$FERRUM serve .*--port $PORT" > "$ART/server/server_ps_after_stop.txt" || true
}

finish() {
  rc=$?
  echo "$rc" > "$ART/run_profile.rc"
  cleanup
  exit "$rc"
}
trap finish EXIT

mkdir -p "$ART"/{bench,profile,remote,server,smoke} "$RUN_CWD"
printf "%s\n" "W2 Gemma3 typed-config ShareGPT prefill profile" > "$ART/lane.txt"
printf "%s\n" "$MODEL" > "$ART/model_id.txt"
printf "%s\n" "$TOKENIZER" > "$ART/tokenizer_path.txt"
printf "%s\n" "$DATASET" > "$ART/dataset_path.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$ART/remote/start_utc.txt"

cd "$SRC"
git rev-parse HEAD > "$ART/remote/git_head.txt"
git status --short --untracked-files=no -- . ":!docs/goals/model-coverage-2026-06-12/artifacts" \
  > "$ART/remote/git_status_source_short.txt"
nvidia-smi > "$ART/remote/nvidia_smi_before_profile.txt"
nvcc --version > "$ART/remote/nvcc_version.txt" 2>&1 || true
sha256sum "$FERRUM" > "$ART/remote/ferrum.sha256"
"$FERRUM" --version > "$ART/remote/ferrum_version.txt" 2>&1 || true

if [[ ! -s "$DATASET" ]]; then
  echo "missing dataset: $DATASET" >&2
  exit 20
fi

cp "$SRC/ferrum.toml" "$RUN_CWD/ferrum.toml"
cat >> "$RUN_CWD/ferrum.toml" <<EOF
batch_decode_prof = true
batch_prefill_prof = true
decode_op_profile = true
prefill_op_profile = true
unified_post_prof = true
next_batch_prof = true
batched_graph = true
EOF

cd "$RUN_CWD"
pkill -f "$FERRUM serve .*--port $PORT" 2>/dev/null || true
sleep 1

SERVE_CMD=(
  "$FERRUM" serve
  --model "$MODEL"
  --backend cuda
  --host 127.0.0.1
  --port "$PORT"
  --batched-graph
  --max-num-seqs 16
  --max-num-batched-tokens 2048
  --kv-capacity 512
  --effective-config-json "$ART/server/serve_effective_config.json"
  --decision-trace-jsonl "$ART/server/serve_decision_trace.jsonl"
)
printf "%q " "${SERVE_CMD[@]}" > "$ART/server/serve.command.txt"
printf "\n" >> "$ART/server/serve.command.txt"
"${SERVE_CMD[@]}" > "$ART/server/server.stdout" 2> "$ART/server/server.log" &
SERVER_PID=$!
echo "$SERVER_PID" > "$ART/server/server.pid"

ready=0
for i in $(seq 1 300); do
  if curl -fsS --noproxy '*' "$BASE/v1/models" > "$ART/server/models.json.tmp" \
    2> "$ART/server/models_curl_last.err"; then
    mv "$ART/server/models.json.tmp" "$ART/server/models.json"
    echo "$i" > "$ART/server/ready_poll_count.txt"
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited during startup at poll=$i" > "$ART/server/server_exited.txt"
    tail -200 "$ART/server/server.log" > "$ART/server/server_tail_on_exit.log" || true
    exit 21
  fi
  sleep 1
done
if [[ "$ready" != "1" ]]; then
  tail -200 "$ART/server/server.log" > "$ART/server/server_tail_on_not_ready.log" || true
  exit 22
fi

cat > "$ART/smoke/chat_request.json" <<EOF
{"model":"$MODEL","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"max_tokens":16,"temperature":0,"stream":false}
EOF
curl -fsS --noproxy '*' -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary "@$ART/smoke/chat_request.json" \
  > "$ART/smoke/chat_response.json" \
  2> "$ART/smoke/chat_curl.err"
python3 - "$ART/smoke/chat_response.json" "$ART/smoke/chat_validation.json" <<'PY'
import json
import sys

response_path, validation_path = sys.argv[1:]
with open(response_path, "r", encoding="utf-8") as f:
    data = json.load(f)
choice = (data.get("choices") or [{}])[0]
message = choice.get("message") or {}
content = message.get("content") or ""
usage = data.get("usage") or {}
completion = usage.get("completion_tokens")
bad_markers = ["<unk>", "[PAD", "\ufffd"]
validation = {
    "content": content,
    "finish_reason": choice.get("finish_reason"),
    "completion_tokens": completion,
    "usage_present": bool(usage),
    "bad_output": any(marker in content for marker in bad_markers),
}
validation["pass"] = (
    content.strip() == "5"
    and isinstance(completion, int)
    and completion > 0
    and not validation["bad_output"]
)
with open(validation_path, "w", encoding="utf-8") as f:
    json.dump(validation, f, indent=2, ensure_ascii=True)
    f.write("\n")
if not validation["pass"]:
    raise SystemExit(f"chat smoke failed: {validation}")
print(f"CHAT_SMOKE_PASS content={content.strip()!r} completion_tokens={completion}")
PY

BENCH_CMD=(
  timeout 1200 "$FERRUM" bench-serve
  --base-url "$BASE"
  --model "$MODEL"
  --tokenizer "$TOKENIZER"
  --dataset sharegpt
  --sharegpt-path "$DATASET"
  --random-output-len 64
  --concurrency-sweep 16
  --num-prompts 16
  --n-repeats 1
  --fail-on-error
  --seed 9271
  --out "$ART/bench/bench_sharegpt_c16.json"
)
python3 - "$ART/bench/bench_sharegpt_c16.command.json" "${BENCH_CMD[@]}" <<'PY'
import json
import sys

path = sys.argv[1]
cmd = sys.argv[2:]
with open(path, "w", encoding="utf-8") as f:
    json.dump({"cmd": cmd}, f)
    f.write("\n")
PY

set +e
"${BENCH_CMD[@]}" > "$ART/bench/bench_sharegpt_c16.stdout" \
  2> "$ART/bench/bench_sharegpt_c16.stderr"
bench_rc=$?
set -e
echo "$bench_rc" > "$ART/bench/bench_sharegpt_c16.rc"
if [[ "$bench_rc" -ne 0 ]]; then
  tail -200 "$ART/server/server.log" > "$ART/server/server_tail_on_bench_fail.log" || true
  exit "$bench_rc"
fi

grep -aE "\\[batch-prefill\\]|\\[prefill-profile\\]|\\[op-profile\\]|\\[batched-op-profile\\]|\\[unified-prof\\]|\\[iter-prof\\]|\\[nb-prof\\]" \
  "$ART/server/server.log" > "$ART/profile/profile_extract.log" || true
grep -aE "ConfigFile|FERRUM_BATCH_PREFILL_PROF|FERRUM_PREFILL_OP_PROFILE|FERRUM_DECODE_OP_PROFILE|FERRUM_BATCH_DECODE_PROF|FERRUM_UNIFIED_POST_PROF|FERRUM_NEXT_BATCH_PROF|FERRUM_BATCHED_GRAPH" \
  "$ART/server/serve_decision_trace.jsonl" > "$ART/profile/runtime_decision_trace_extract.log" || true
nvidia-smi > "$ART/remote/nvidia_smi_after_profile.txt" || true

python3 - "$ART" "$VLLM_C16_TPS" <<'PY'
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
vllm = float(sys.argv[2])
bench = json.loads((out / "bench" / "bench_sharegpt_c16.json").read_text())
if isinstance(bench, list):
    report = bench[0] if bench else {}
else:
    report = bench
log = (out / "server" / "server.log").read_text(errors="replace")
effective = json.loads((out / "server" / "serve_effective_config.json").read_text())
runtime_sources = {
    entry.get("key"): entry.get("source")
    for entry in effective.get("entries", [])
    if entry.get("key", "").endswith("_PROF")
    or entry.get("key") in {"FERRUM_DECODE_OP_PROFILE", "FERRUM_PREFILL_OP_PROFILE"}
}
batch_prefill_lines = re.findall(r"\[batch-prefill\].*", log)
fallback_true = [line for line in batch_prefill_lines if "fallback=true" in line]
fallback_false = [line for line in batch_prefill_lines if "fallback=false" in line]
tps = ((report.get("output_throughput_tps") or {}).get("mean")) or 0.0
summary = {
    "status": "PASS",
    "bench_rc": int((out / "bench" / "bench_sharegpt_c16.rc").read_text().strip()),
    "chat_pass": json.loads((out / "smoke" / "chat_validation.json").read_text()).get("pass"),
    "concurrency": report.get("concurrency"),
    "n_repeats": report.get("n_repeats"),
    "n_requests_per_run": report.get("n_requests_per_run"),
    "completed_per_run": report.get("completed_per_run"),
    "errored_per_run": report.get("errored_per_run"),
    "bad_output_per_run": report.get("bad_output_per_run"),
    "zero_output_tokens_per_run": report.get("zero_output_tokens_per_run"),
    "output_token_count_source": report.get("output_token_count_source"),
    "output_throughput_tps_mean": tps,
    "ttft_ms_p50": (report.get("time_to_first_token_ms") or {}).get("p50"),
    "tpot_ms_p50": (report.get("time_per_output_token_ms") or {}).get("p50"),
    "output_tokens_per_request": report.get("output_tokens_per_request"),
    "vllm_c16_tps_baseline": vllm,
    "ferrum_vs_vllm_ratio": (tps / vllm) if vllm else 0.0,
    "ferrum_vs_vllm_percent": ((tps / vllm) * 100.0) if vllm else 0.0,
    "varlen_unified_in_log": "varlen_unified=true" in log,
    "batch_prefill_line_count": len(batch_prefill_lines),
    "batch_prefill_fallback_true_count": len(fallback_true),
    "batch_prefill_fallback_false_count": len(fallback_false),
    "batch_prefill_sample": batch_prefill_lines[:8],
    "runtime_profiler_sources": runtime_sources,
}
if summary["bench_rc"] != 0 or not summary["chat_pass"]:
    summary["status"] = "FAIL"
with (out / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=True)
    f.write("\n")
with (out / "summary.md").open("w", encoding="utf-8") as f:
    f.write("# W2 typed-config ShareGPT prefill profile\n\n")
    for key in [
        "status",
        "bench_rc",
        "chat_pass",
        "completed_per_run",
        "errored_per_run",
        "bad_output_per_run",
        "zero_output_tokens_per_run",
        "output_token_count_source",
        "output_throughput_tps_mean",
        "ttft_ms_p50",
        "tpot_ms_p50",
        "ferrum_vs_vllm_percent",
        "varlen_unified_in_log",
        "batch_prefill_fallback_true_count",
        "batch_prefill_fallback_false_count",
    ]:
        f.write(f"- {key}: {summary.get(key)}\n")
    f.write("\n## Batch Prefill Sample\n\n")
    for line in summary["batch_prefill_sample"]:
        f.write(f"- `{line}`\n")
if summary["status"] != "PASS":
    raise SystemExit(1)
PY

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$ART/remote/end_utc.txt"
