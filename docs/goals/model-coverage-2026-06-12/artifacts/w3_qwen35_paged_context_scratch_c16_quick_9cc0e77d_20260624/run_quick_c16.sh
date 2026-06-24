#!/usr/bin/env bash
set -Eeuo pipefail

export PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_HOME=/workspace/hf-cache
export NO_COLOR=1
export RUST_BACKTRACE=1

ROOT=/workspace/artifacts/w3_qwen35_paged_context_scratch_c16_quick_9cc0e77d_20260624
REPO=/workspace/ferrum-infer-rs-git
MODEL=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
PORT=58710
DATASET=$REPO/docs/goals/model-coverage-2026-06-12/artifacts/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=$(find /workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots -mindepth 1 -maxdepth 1 -type d | head -n 1)

mkdir -p "$ROOT"/{logs,run_smoke,server,perf,env,hardware,build,vast}
cd "$REPO"

{
  echo started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo lane=W3_QWEN35_PAGED_CONTEXT_SCRATCH_C16_QUICK_9CC0E77D
  echo git_sha=$(git rev-parse HEAD)
  echo stop_condition=build_failure_or_run_smoke_failure_or_server_failure_or_c16_bench_failure_or_completion
  echo correctness_gate=ferrum_run_smoke_and_ferrum_serve_chat_smoke
  echo performance_command=ferrum_bench_serve_c16_num_prompts32_n_repeats1_fail_on_error_seed9271_ignore_eos
  echo diagnostic_only=true
  echo no_live_vllm=true
  echo tokenizer=$TOKENIZER
  git status --short --untracked-files=no
  sha256sum "$DATASET"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader,nounits
  nvcc --version | tail -n 1 || true
} > "$ROOT/logs/preflight.log" 2>&1

git rev-parse HEAD > "$ROOT/env/git_sha.txt"
git status --short --untracked-files=no > "$ROOT/env/git_status_short.txt"
sha256sum "$DATASET" > "$ROOT/env/dataset.sha256"
nvidia-smi > "$ROOT/hardware/nvidia_smi_before.txt" 2>&1 || true

EXPECTED_SHA=9cc0e77d562ca94659c15bc7fb61c439d7d588b2
ACTUAL_SHA=$(git rev-parse HEAD)
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "wrong git sha: expected $EXPECTED_SHA got $ACTUAL_SHA" >&2
  exit 20
fi

pkill -f 'target/release/ferrum serve' >/dev/null 2>&1 || true

BUILD_CMD=(cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source)
printf "%q " "${BUILD_CMD[@]}" > "$ROOT/build/build.command.txt"; echo >> "$ROOT/build/build.command.txt"
if "${BUILD_CMD[@]}" > "$ROOT/build/build.stdout.log" 2> "$ROOT/build/build.stderr.log"; then
  echo 0 > "$ROOT/build/build.exit"
else
  code=$?
  echo "$code" > "$ROOT/build/build.exit"
  exit "$code"
fi
sha256sum target/release/ferrum > "$ROOT/env/ferrum.sha256"

RUN_CMD=(target/release/ferrum run "$MODEL" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --disable-thinking --output-format jsonl --temperature 0 --max-tokens 8 --effective-config-json "$ROOT/run_smoke/effective_config.json" --decision-trace-jsonl "$ROOT/run_smoke/decision_trace.jsonl" --prompt "What is 2+3? Answer with only the number.")
printf "%q " "${RUN_CMD[@]}" > "$ROOT/run_smoke/ferrum_run.command.txt"; echo >> "$ROOT/run_smoke/ferrum_run.command.txt"
if "${RUN_CMD[@]}" > "$ROOT/run_smoke/ferrum_run.stdout.log" 2> "$ROOT/run_smoke/ferrum_run.stderr.log"; then
  echo 0 > "$ROOT/run_smoke/ferrum_run.exit"
else
  code=$?
  echo "$code" > "$ROOT/run_smoke/ferrum_run.exit"
  exit "$code"
fi
python3 - "$ROOT/run_smoke/ferrum_run.stdout.log" <<'PY'
import sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
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
for _ in $(seq 1 1200); do
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

python3 - "$PORT" "$ROOT/server/chat_smoke_response.json" <<'PY'
import json
import sys
import urllib.request

port, out = sys.argv[1:]
payload = {
    "model": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with only the number."}],
    "temperature": 0,
    "max_tokens": 8,
    "stream": False,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=90) as resp:
    text = resp.read().decode()
open(out, "w", encoding="utf-8").write(text)
content = json.loads(text)["choices"][0]["message"].get("content", "")
print(content)
if "5" not in content:
    raise SystemExit(40)
PY

BENCH_REPORT=$ROOT/perf/bench_ferrum_c16_32x1.json
BENCH_CMD=(target/release/ferrum bench-serve --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --tokenizer "$TOKENIZER" --dataset sharegpt --sharegpt-path "$DATASET" --random-output-len 128 --ignore-eos --concurrency 16 --num-prompts 32 --warmup-requests 4 --n-repeats 1 --fail-on-error --seed 9271 --timeout 600 --out "$BENCH_REPORT")
printf "%q " "${BENCH_CMD[@]}" > "$ROOT/perf/bench-ferrum-c16.command.txt"; echo >> "$ROOT/perf/bench-ferrum-c16.command.txt"
if "${BENCH_CMD[@]}" > "$ROOT/perf/bench.stdout.log" 2> "$ROOT/perf/bench.stderr.log"; then
  echo 0 > "$ROOT/perf/bench.exit"
else
  code=$?
  echo "$code" > "$ROOT/perf/bench.exit"
  exit "$code"
fi

python3 - "$ROOT" "$BENCH_REPORT" <<'PY' | tee "$ROOT/perf/bench_metrics.txt"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = Path(sys.argv[2])
data = json.loads(report.read_text())
rows = data if isinstance(data, list) else [data]
row = rows[0]
metric = {
    "output_throughput_tps": row.get("output_throughput_tps", {}).get("mean"),
    "completed_per_run": row.get("completed_per_run"),
    "errored_per_run": row.get("errored_per_run"),
    "itl_p95_ms": row.get("itl_ms", {}).get("p95", {}).get("mean"),
    "ttft_p95_ms": row.get("ttft_ms", {}).get("p95", {}).get("mean"),
    "output_token_count_source": row.get("output_token_count_source"),
}
previous = 688.1409470636319
metric["previous_zzz111_output_tps"] = previous
if isinstance(metric["output_throughput_tps"], (int, float)):
    metric["delta_vs_zzz111_output_tps"] = metric["output_throughput_tps"] - previous
    metric["ratio_vs_zzz111_output_tps"] = metric["output_throughput_tps"] / previous
print(
    "DIAG_METRIC",
    "c", row.get("concurrency"),
    "output_tps", metric["output_throughput_tps"],
    "completed", metric["completed_per_run"],
    "errored", metric["errored_per_run"],
    "itl_p95", metric["itl_p95_ms"],
    "ratio_vs_zzz111", metric.get("ratio_vs_zzz111_output_tps"),
)
summary = {
    "lane": "W3_QWEN35_PAGED_CONTEXT_SCRATCH_C16_QUICK_9CC0E77D",
    "status": "passed",
    "diagnostic_only": True,
    "no_live_vllm": True,
    "git_sha": (root / "env/git_sha.txt").read_text().strip(),
    "binary_sha256": (root / "env/ferrum.sha256").read_text().split()[0],
    "bench": metric,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(root / "summary.txt").write_text("W3 QWEN35 PAGED CONTEXT SCRATCH C16 QUICK DIAG PASS: " + str(root) + "\n")
PY

nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader,nounits > "$ROOT/logs/nvidia_final.csv" 2>&1 || true
