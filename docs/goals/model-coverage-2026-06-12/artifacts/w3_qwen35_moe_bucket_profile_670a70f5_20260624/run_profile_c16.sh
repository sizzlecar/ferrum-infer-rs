#!/usr/bin/env bash
set -Eeuo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_HOME=/workspace/hf-cache
export NO_COLOR=1
export RUST_BACKTRACE=1
ROOT=/workspace/artifacts/w3_qwen35_moe_bucket_profile_670a70f5_20260624
REPO=/workspace/ferrum-infer-rs-git
MODEL=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
PORT=58693
DATASET=$REPO/docs/goals/model-coverage-2026-06-12/artifacts/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=$(find /workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots -mindepth 1 -maxdepth 1 -type d | head -n 1)
PROFILE_FLAGS='{"FERRUM_QWEN35_DECODE_PROFILE":"1","FERRUM_QWEN35_LAYER_DETAIL_PROFILE":"1","FERRUM_MARLIN_PROFILE":"1","diagnostic":"w3_qwen35_moe_bucket_profile_c16","no_live_vllm":true}'
mkdir -p "$ROOT/logs" "$ROOT/build" "$ROOT/run_smoke" "$ROOT/server" "$ROOT/perf" "$ROOT/env" "$ROOT/hardware"
cd "$REPO"
{
  echo started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo lane=W3_QWEN35_MOE_BUCKET_PROFILE_C16_DIAG
  echo git_sha=$(git rev-parse HEAD)
  echo stop_condition=ssh_cuda_source_sync_build_run_smoke_serve_smoke_bench_profile_parse_first_failure_or_completion
  echo correctness_gate=ferrum_run_smoke_and_ferrum_serve_chat_smoke_and_bench_fail_on_error_zero_errors
  echo diagnostic_command=ferrum_bench_serve_c16_num_prompts16_output32_n_repeats1_fail_on_error_seed9271_ignore_eos_profile_jsonl
  echo no_live_vllm=true
  echo tokenizer=$TOKENIZER
  git status --short --untracked-files=no
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader,nounits
  nvcc --version | tail -n 1 || true
} > "$ROOT/logs/preflight.log" 2>&1
git rev-parse HEAD > "$ROOT/env/git_sha.txt"
git status --short --untracked-files=no > "$ROOT/env/git_status_short.txt"
nvidia-smi > "$ROOT/hardware/nvidia_smi_before.txt" 2>&1 || true
BUILD_CMD=(cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source)
printf "%q " "${BUILD_CMD[@]}" > "$ROOT/build/build.command.txt"; echo >> "$ROOT/build/build.command.txt"
"${BUILD_CMD[@]}" > "$ROOT/build/build.stdout.log" 2> "$ROOT/build/build.stderr.log"
echo $? > "$ROOT/build/build.exit"
sha256sum target/release/ferrum > "$ROOT/env/ferrum.sha256"
sha256sum "$DATASET" > "$ROOT/env/dataset.sha256"
RUN_CMD=(target/release/ferrum run "$MODEL" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --disable-thinking --output-format jsonl --temperature 0 --max-tokens 8 --effective-config-json "$ROOT/run_smoke/effective_config.json" --decision-trace-jsonl "$ROOT/run_smoke/decision_trace.jsonl" --prompt "What is 2+3? Answer with only the number.")
printf "%q " "${RUN_CMD[@]}" > "$ROOT/run_smoke/ferrum_run.command.txt"; echo >> "$ROOT/run_smoke/ferrum_run.command.txt"
"${RUN_CMD[@]}" > "$ROOT/run_smoke/ferrum_run.stdout.log" 2> "$ROOT/run_smoke/ferrum_run.stderr.log"
echo $? > "$ROOT/run_smoke/ferrum_run.exit"
python3 - "$ROOT/run_smoke/ferrum_run.stdout.log" <<PY
import sys
text=open(sys.argv[1]).read()
if "5" not in text:
    raise SystemExit("ferrum run smoke missing expected 5")
PY
export FERRUM_QWEN35_DECODE_PROFILE=1
export FERRUM_QWEN35_LAYER_DETAIL_PROFILE=1
export FERRUM_MARLIN_PROFILE=1
SERVE_CMD=(target/release/ferrum serve "$MODEL" --host 127.0.0.1 --port "$PORT" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --effective-config-json "$ROOT/server/effective_config.json" --decision-trace-jsonl "$ROOT/server/decision_trace.jsonl" --scheduler-trace-jsonl "$ROOT/server/scheduler_trace.jsonl" --profile-jsonl "$ROOT/server/profile.jsonl" --profile-commit-sha "$(git rev-parse HEAD)" --profile-model "$MODEL" --profile-concurrency 16 --profile-runtime-flags-json "$PROFILE_FLAGS")
printf "%q " "${SERVE_CMD[@]}" > "$ROOT/server/serve.command.txt"; echo >> "$ROOT/server/serve.command.txt"
"${SERVE_CMD[@]}" > "$ROOT/server/serve.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$ROOT/server/server.pid"
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" > "$ROOT/server/server.wait.stdout" 2> "$ROOT/server/server.wait.stderr" || true
  nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader,nounits > "$ROOT/logs/nvidia_after.csv" 2>&1 || true
  echo finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ) >> "$ROOT/logs/preflight.log"
}
trap cleanup EXIT
READY=0
for i in $(seq 1 1200); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo server_exited_before_ready > "$ROOT/server/serve.exit"
    tail -n 240 "$ROOT/server/serve.log" > "$ROOT/server/serve.failure.tail.txt" 2>/dev/null || true
    exit 31
  fi
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > "$ROOT/server/models.json" 2> "$ROOT/server/models.curl.err"; then
    READY=1
    break
  fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo server_not_ready > "$ROOT/server/serve.exit"
  exit 32
fi
python3 - "$PORT" "$ROOT/server/chat_smoke_response.json" <<PY
import json, sys, urllib.request
port, out = sys.argv[1:]
payload = {"model":"Qwen/Qwen3.5-35B-A3B-GPTQ-Int4","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"temperature":0,"max_tokens":8,"stream":False}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=90) as resp:
    text=resp.read().decode()
open(out,"w").write(text)
content=json.loads(text)["choices"][0]["message"].get("content","")
print(content)
if "5" not in content:
    raise SystemExit(40)
PY
BENCH_REPORT=$ROOT/perf/bench_ferrum_c16_16x1_o32.json
BENCH_CMD=(target/release/ferrum bench-serve --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --tokenizer "$TOKENIZER" --dataset sharegpt --sharegpt-path "$DATASET" --random-output-len 32 --ignore-eos --concurrency 16 --num-prompts 16 --warmup-requests 0 --n-repeats 1 --fail-on-error --seed 9271 --timeout 600 --out "$BENCH_REPORT")
printf "%q " "${BENCH_CMD[@]}" > "$ROOT/perf/bench-ferrum-c16.command.txt"; echo >> "$ROOT/perf/bench-ferrum-c16.command.txt"
"${BENCH_CMD[@]}" > "$ROOT/perf/bench.stdout.log" 2> "$ROOT/perf/bench.stderr.log"
echo $? > "$ROOT/perf/bench.exit"
python3 - <<PY | tee "$ROOT/perf/bench_metrics.txt"
import json
from pathlib import Path
p=Path("$BENCH_REPORT")
data=json.loads(p.read_text())
rows=data if isinstance(data, list) else [data]
for r in rows:
    print("DIAG_METRIC", "c", r.get("concurrency"), "output_tps", r.get("output_throughput_tps",{}).get("mean"), "completed", r.get("completed_per_run"), "errored", r.get("errored_per_run"), "ttft_p95", r.get("ttft_ms",{}).get("p95",{}).get("mean"), "itl_p95", r.get("itl_ms",{}).get("p95",{}).get("mean"))
PY
python3 - "$ROOT/server/profile.jsonl" "$ROOT/server/profile_summary.json" <<PY
import json, statistics, sys
from pathlib import Path
profile_path=Path(sys.argv[1]); out_path=Path(sys.argv[2])
rows=[]
event_counts={}
for line in profile_path.read_text().splitlines():
    if not line.strip():
        continue
    ev=json.loads(line)
    name=ev.get("event")
    event_counts[name]=event_counts.get(name,0)+1
    if name != "qwen35_sparse_moe_detail":
        continue
    shape=ev.get("shape") or {}
    stage=ev.get("stage_us") or {}
    bucket=stage.get("routed_bucket")
    if not isinstance(bucket, dict):
        continue
    row={"layer": shape.get("layer"), "tokens": shape.get("tokens"), "top_k": shape.get("top_k")}
    for key in ["total","route","plan","gather","gemm1","silu","gemm3","combine","sync","d2h"]:
        val=bucket.get(key)
        if isinstance(val, (int,float)):
            row[key]=float(val)
    rows.append(row)
def pct(vals, q):
    if not vals:
        return None
    vals=sorted(vals)
    idx=min(len(vals)-1, max(0, round((len(vals)-1)*q)))
    return vals[idx]
summary={"event_counts": event_counts, "bucket_rows": len(rows), "by_tokens": {}, "overall": {}}
for label, subset in [("overall", rows)]:
    stats={}
    for key in ["total","route","plan","gather","gemm1","silu","gemm3","combine","sync","d2h"]:
        vals=[r[key] for r in subset if key in r]
        if vals:
            stats[key]={"mean": statistics.fmean(vals), "p50": pct(vals,0.50), "p95": pct(vals,0.95), "n": len(vals)}
    summary[label]=stats
for tokens in sorted({r.get("tokens") for r in rows if r.get("tokens") is not None}):
    subset=[r for r in rows if r.get("tokens")==tokens]
    stats={}
    for key in ["total","route","plan","gather","gemm1","silu","gemm3","combine","sync","d2h"]:
        vals=[r[key] for r in subset if key in r]
        if vals:
            stats[key]={"mean": statistics.fmean(vals), "p50": pct(vals,0.50), "p95": pct(vals,0.95), "n": len(vals)}
    summary["by_tokens"][str(tokens)]=stats
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
print(json.dumps({"bucket_rows": summary["bucket_rows"], "events": event_counts}, sort_keys=True))
if summary["bucket_rows"] == 0:
    raise SystemExit("no qwen35 routed_bucket profile rows")
PY
python3 - "$ROOT" <<PY
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
def read(p):
    try: return p.read_text().strip()
    except FileNotFoundError: return None
bench=json.loads((root/"perf/bench_ferrum_c16_16x1_o32.json").read_text())
rows=bench if isinstance(bench,list) else [bench]
profile_summary=json.loads((root/"server/profile_summary.json").read_text())
summary={
  "lane":"W3_QWEN35_MOE_BUCKET_PROFILE_C16_DIAG",
  "status":"passed",
  "git_sha":read(root/"env/git_sha.txt"),
  "binary_sha256":read(root/"env/ferrum.sha256"),
  "no_live_vllm":True,
  "bench_exit":read(root/"perf/bench.exit"),
  "run_smoke_exit":read(root/"run_smoke/ferrum_run.exit"),
  "serve_chat_smoke_response_exists":(root/"server/chat_smoke_response.json").exists(),
  "bench": {
    "concurrency": rows[0].get("concurrency"),
    "completed_per_run": rows[0].get("completed_per_run"),
    "errored_per_run": rows[0].get("errored_per_run"),
    "output_throughput_tps_mean": rows[0].get("output_throughput_tps",{}).get("mean"),
    "ttft_ms_p95_mean": rows[0].get("ttft_ms",{}).get("p95",{}).get("mean"),
    "itl_ms_p95_mean": rows[0].get("itl_ms",{}).get("p95",{}).get("mean"),
    "random_output_len": 32,
    "num_prompts": 16
  },
  "profile_summary": profile_summary
}
(root/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
(root/"summary.txt").write_text("W3 Qwen35 MoE bucket profile c16 diagnostic: run/serve smoke passed, c16 short bench completed, routed_bucket profile rows captured. Diagnostic only; no release performance claim.\n")
PY
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader,nounits > "$ROOT/logs/nvidia_final.csv" 2>&1 || true
