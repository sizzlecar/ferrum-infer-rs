#!/usr/bin/env bash
set -Eeuo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_HOME=/workspace/hf-cache
export NO_COLOR=1
ART="/workspace/artifacts/w3_qwen35_fused_gate_merge_c16_quick_a4bbc933_20260624T123416Z"
REPO=/workspace/ferrum-infer-rs-git
MODEL=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
PORT=57048
DATASET=$REPO/docs/goals/model-coverage-2026-06-12/artifacts/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=$(find /workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots -mindepth 1 -maxdepth 1 -type d | head -n 1)
cd "$REPO"
{
  echo started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo lane=W3_QWEN35_FUSED_GATE_MERGE_C16_QUICK
  echo release_evidence=false
  echo no_live_vllm=true
  echo port=$PORT
  echo git_head=$(git rev-parse HEAD)
  echo git_status_start
  git status --short --untracked-files=no
  echo git_status_end
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader,nounits
  echo tokenizer=$TOKENIZER
  echo dataset=$DATASET
} > "$ART/logs/preflight.log" 2>&1
sha256sum target/release/ferrum > "$ART/env/ferrum.sha256"
git rev-parse HEAD > "$ART/env/git_sha.txt"
git status --short --untracked-files=no > "$ART/env/git_status_short.txt"
nvidia-smi > "$ART/hardware/nvidia_smi_before.txt" 2>&1 || true
SERVE_CMD=(target/release/ferrum serve "$MODEL" --host 127.0.0.1 --port "$PORT" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --effective-config-json "$ART/server/effective_config.json" --decision-trace-jsonl "$ART/server/decision_trace.jsonl" --scheduler-trace-jsonl "$ART/server/scheduler_trace.jsonl")
printf "%q " "${SERVE_CMD[@]}" > "$ART/server/serve.command.txt"; echo >> "$ART/server/serve.command.txt"
"${SERVE_CMD[@]}" > "$ART/server/serve.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$ART/server/server.pid"
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" > "$ART/server/server.wait.stdout" 2> "$ART/server/server.wait.stderr" || true
  nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader,nounits > "$ART/logs/nvidia_after.csv" 2>&1 || true
  echo finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ) >> "$ART/logs/preflight.log"
}
trap cleanup EXIT
READY=0
for i in $(seq 1 900); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo server_exited_before_ready > "$ART/server/serve.exit"
    tail -n 260 "$ART/server/serve.log" > "$ART/server/serve.failure.tail.txt" 2>/dev/null || true
    exit 31
  fi
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > "$ART/server/models.json" 2> "$ART/server/models.curl.err"; then
    READY=1; break
  fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then echo server_not_ready > "$ART/server/serve.exit"; exit 32; fi
python3 - "$PORT" "$ART/server/chat_smoke_response.json" <<PY
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
BENCH_REPORT=$ART/perf/bench_ferrum_c16_32x1.json
BENCH_CMD=(target/release/ferrum bench-serve --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --tokenizer "$TOKENIZER" --dataset sharegpt --sharegpt-path "$DATASET" --random-output-len 128 --ignore-eos --concurrency 16 --num-prompts 32 --warmup-requests 4 --n-repeats 1 --fail-on-error --seed 9271 --timeout 600 --out "$BENCH_REPORT")
printf "%q " "${BENCH_CMD[@]}" > "$ART/perf/bench-ferrum-c16.command.txt"; echo >> "$ART/perf/bench-ferrum-c16.command.txt"
"${BENCH_CMD[@]}" > "$ART/perf/bench.stdout.log" 2> "$ART/perf/bench.stderr.log"
echo $? > "$ART/perf/bench.exit"
python3 - <<PY | tee "$ART/perf/bench_metrics.txt"
import json
from pathlib import Path
p=Path("$BENCH_REPORT")
data=json.loads(p.read_text())
rows=data if isinstance(data, list) else [data]
for r in rows:
    print("FUSED_GATE_MERGE_C16_DIAG", "c", r.get("concurrency"), "output_tps", r.get("output_throughput_tps",{}).get("mean"), "completed", r.get("completed_per_run"), "errored", r.get("errored_per_run"), "ttft_p95", r.get("ttft_ms",{}).get("p95",{}).get("mean"), "itl_p95", r.get("itl_ms",{}).get("p95",{}).get("mean"))
PY
echo "FERRUM W3 QWEN35 FUSED GATE MERGE C16 QUICK PASS: $ART" | tee "$ART/summary.txt"
