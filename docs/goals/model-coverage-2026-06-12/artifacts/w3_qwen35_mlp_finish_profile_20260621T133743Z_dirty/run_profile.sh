#!/usr/bin/env bash
set -euo pipefail
ART=$(cd "$(dirname "$0")" && pwd)
cd /workspace/ferrum-w3-clean-b6
export PATH=/root/.cargo/bin:$PATH
export HF_HOME=/workspace/hf-cache
export FERRUM_QWEN35_LAYER_DETAIL_PROFILE=1
MODEL=/workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b
DATA=/workspace/artifacts/ferrum_w3_sharegpt_ab_20b8946b_20260619T121844Z_ninja/dataset/ascii_sharegpt_w3_100_58d5721d.jsonl
PORT=18266
cat > "$ART/runtime_flags.json" <<'JSON'
{"gpu_memory_utilization":0.9,"max_model_len":2048,"max_num_seqs":32,"max_num_batched_tokens":8192,"kv_capacity":2048,"scheduler_prefill_first_until_active":32,"scheduler_active_decode_prefill_chunk":8192,"greedy_argmax":true,"disable_batched_graph":true,"disable_unified_graph":true,"profile":"qwen35_layer_detail"}
JSON
RUNTIME_FLAGS=$(tr -d '\n' < "$ART/runtime_flags.json")
{
  echo "lane=W3 Qwen35 c32 mlp-finish profile diagnostic"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "head=$(git rev-parse HEAD)"
  echo "status_short_begin<<STATUS"
  git status --short crates/ferrum-models/src/models/qwen35.rs
  echo "STATUS"
  echo "model=$MODEL"
  echo "dataset=$DATA"
  echo "port=$PORT"
} > "$ART/run_context.txt"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version --format=csv > "$ART/gpu_before.csv"
sha256sum target/release/ferrum > "$ART/binary.sha256"
target/release/ferrum serve \
  "$MODEL" \
  --backend cuda \
  --gpu-devices 0 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 2048 \
  --kv-capacity 2048 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --scheduler-prefill-first-until-active 32 \
  --scheduler-active-decode-prefill-chunk 8192 \
  --greedy-argmax \
  --disable-batched-graph \
  --disable-unified-graph \
  --effective-config-json "$ART/serve_effective_config.json" \
  --decision-trace-jsonl "$ART/serve_decision_trace.jsonl" \
  --profile-jsonl "$ART/profile_detail.jsonl" \
  --profile-commit-sha "690ba923-dirty-linear-scratch" \
  --profile-model "qwen35-gptq-int4" \
  --profile-concurrency 32 \
  --profile-runtime-flags-json "$RUNTIME_FLAGS" \
  > "$ART/serve.log" 2>&1 &
SERVE_PID=$!
echo "$SERVE_PID" > "$ART/serve.pid"
cleanup() {
  if kill -0 "$SERVE_PID" 2>/dev/null; then
    kill "$SERVE_PID" || true
    wait "$SERVE_PID" || true
  fi
}
trap cleanup EXIT
python3 - <<PY > "$ART/wait_health.log" 2>&1
import time, urllib.request, sys
url = "http://127.0.0.1:${PORT}/health"
last = None
for _ in range(240):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            print(r.read().decode())
        sys.exit(0)
    except Exception as e:
        last = repr(e)
        time.sleep(2)
print("health timeout", last)
sys.exit(1)
PY
python3 - <<PY > "$ART/serve_smoke.log" 2>&1
import json, urllib.request
url = "http://127.0.0.1:${PORT}/v1/chat/completions"
payload = {"model":"$MODEL","messages":[{"role":"user","content":"What is the capital of France? Answer with one word."}],"max_tokens":8,"temperature":0,"stream":False}
req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    body = r.read().decode()
print(body)
text = json.loads(body)["choices"][0]["message"].get("content", "")
if "Paris" not in text:
    raise SystemExit(f"missing Paris: {text!r}")
print("SERVE_SMOKE_PASS")
PY
target/release/ferrum bench-serve \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --dataset sharegpt \
  --sharegpt-path "$DATA" \
  --random-output-len 128 \
  --num-prompts 20 \
  --warmup-requests 4 \
  --n-repeats 1 \
  --concurrency 32 \
  --fail-on-error \
  --seed 9271 \
  --hw-id rtx-4090 \
  --commit-sha 690ba923-dirty-linear-scratch \
  --out "$ART/bench_c32_20x1.json" > "$ART/bench_c32_20x1.log" 2>&1
python3 - <<PY > "$ART/profile_summary.json"
import collections, json, pathlib, statistics
art = pathlib.Path("$ART")
summary = {"artifact": str(art)}
bench = json.loads((art / "bench_c32_20x1.json").read_text())
summary["bench"] = {
  "output_tps": bench.get("output_throughput_tps"),
  "completed": bench.get("completed_per_run"),
  "errored": bench.get("errored_per_run"),
  "bad_output": bench.get("bad_output_per_run"),
  "missing_done": bench.get("missing_done_per_run"),
  "duplicate_done": bench.get("duplicate_done_per_run"),
  "zero_output_tokens": bench.get("zero_output_tokens_per_run"),
  "output_token_count_source": bench.get("output_token_count_source"),
}
events = collections.defaultdict(list)
profile_path = art / "profile_detail.jsonl"
if profile_path.exists():
    for line in profile_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        ev = row.get("event")
        if ev in {"qwen35_mlp_finish_detail", "qwen35_sparse_moe_detail", "qwen35_linear_decode_detail"}:
            events[ev].append(row)
summary["event_counts"] = {k: len(v) for k, v in sorted(events.items())}
for ev, rows in events.items():
    agg = {}
    by_stage = collections.defaultdict(list)
    for row in rows:
        for k, v in (row.get("stage_us") or {}).items():
            if isinstance(v, (int, float)):
                by_stage[k].append(v)
    for k, vals in by_stage.items():
        agg[k] = {"mean": statistics.mean(vals), "p50": statistics.median(vals), "sum": sum(vals)}
    summary[ev] = dict(sorted(agg.items(), key=lambda kv: -kv[1]["sum"]))
print(json.dumps(summary, indent=2, sort_keys=True))
PY
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version --format=csv > "$ART/gpu_after.csv"
echo "QWEN35_MLP_FINISH_PROFILE_DIAG_PASS $ART" | tee "$ART/result.txt"
