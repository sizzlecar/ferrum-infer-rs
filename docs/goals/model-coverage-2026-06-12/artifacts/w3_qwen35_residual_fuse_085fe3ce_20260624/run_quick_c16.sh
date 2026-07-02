#!/usr/bin/env bash
set -Eeuo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_HOME=/workspace/hf-cache
export NO_COLOR=1
export RUST_BACKTRACE=1
ROOT=/workspace/artifacts/w3_qwen35_residual_fuse_085fe3ce_20260624
REPO=/workspace/ferrum-infer-rs-git
MODEL=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
PORT=58691
DATASET=$REPO/docs/goals/model-coverage-2026-06-12/artifacts/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=$(find /workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots -mindepth 1 -maxdepth 1 -type d | head -n 1)
mkdir -p "$ROOT/logs" "$ROOT/run_smoke" "$ROOT/server" "$ROOT/perf" "$ROOT/env" "$ROOT/hardware" "$ROOT/commands"
cd "$REPO"
{
  echo started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo lane=W3_QWEN35_RESIDUAL_FUSE_C16_QUICK_DIAG
  echo git_sha=$(git rev-parse HEAD)
  echo stop_condition=run_smoke_failure_or_server_failure_or_c16_bench_failure_or_completion
  echo correctness_gate=ferrum_run_smoke_and_ferrum_serve_chat_smoke
  echo performance_command=ferrum_bench_serve_c16_num_prompts32_n_repeats1_fail_on_error_seed9271_ignore_eos
  echo no_live_vllm=true
  echo tokenizer=$TOKENIZER
  git status --short --untracked-files=no
  sha256sum target/release/ferrum
  sha256sum "$DATASET"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader,nounits
  nvcc --version | tail -n 1 || true
} > "$ROOT/logs/preflight.log" 2>&1
sha256sum target/release/ferrum > "$ROOT/env/ferrum.sha256"
git rev-parse HEAD > "$ROOT/env/git_sha.txt"
nvidia-smi > "$ROOT/hardware/nvidia_smi_before.txt" 2>&1 || true
RUN_CMD=(target/release/ferrum run "$MODEL" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --disable-thinking --output-format jsonl --temperature 0 --max-tokens 8 --effective-config-json "$ROOT/run_smoke/effective_config.json" --decision-trace-jsonl "$ROOT/run_smoke/decision_trace.jsonl" --prompt "What is 2+3? Answer with only the number.")
printf "%q " "${RUN_CMD[@]}" > "$ROOT/run_smoke/ferrum_run.command.txt"; echo >> "$ROOT/run_smoke/ferrum_run.command.txt"
"${RUN_CMD[@]}" > "$ROOT/run_smoke/ferrum_run.stdout.log" 2> "$ROOT/run_smoke/ferrum_run.stderr.log"
echo $? > "$ROOT/run_smoke/ferrum_run.exit"
python3 - "$ROOT/run_smoke/ferrum_run.stdout.log" <<PY
import json, sys
text=open(sys.argv[1]).read()
if "5" not in text:
    raise SystemExit("ferrum run smoke missing expected 5")
PY
SERVE_CMD=(target/release/ferrum serve "$MODEL" --host 127.0.0.1 --port "$PORT" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --effective-config-json "$ROOT/server/effective_config.json" --decision-trace-jsonl "$ROOT/server/decision_trace.jsonl" --scheduler-trace-jsonl "$ROOT/server/scheduler_trace.jsonl")
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
BENCH_REPORT=$ROOT/perf/bench_ferrum_c16_32x1.json
BENCH_CMD=(target/release/ferrum bench-serve --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --tokenizer "$TOKENIZER" --dataset sharegpt --sharegpt-path "$DATASET" --random-output-len 128 --ignore-eos --concurrency 16 --num-prompts 32 --warmup-requests 4 --n-repeats 1 --fail-on-error --seed 9271 --timeout 600 --out "$BENCH_REPORT")
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
