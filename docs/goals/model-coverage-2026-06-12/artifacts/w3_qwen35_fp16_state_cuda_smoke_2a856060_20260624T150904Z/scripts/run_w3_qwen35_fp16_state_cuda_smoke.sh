#!/usr/bin/env bash
set -Eeuo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_HOME=/workspace/hf-cache
export NO_COLOR=1
SHA=2a8560609053387de0ad36f786cfb41d04b820bb
MODEL=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
REPO=/workspace/ferrum-infer-rs
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ART=/workspace/artifacts/w3_qwen35_fp16_state_cuda_smoke_${SHA:0:8}_${STAMP}
PORT=55988
SERVER_PID=""
mkdir -p "$ART"/{env,hardware,logs,run,server,perf,scripts}
cp "$0" "$ART/scripts/$(basename "$0")" 2>/dev/null || true
log_step() { printf "%s %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$ART/logs/lane.log"; }
stop_server() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" > "$ART/server/server.wait.stdout" 2> "$ART/server/server.wait.stderr" || true
  fi
  SERVER_PID=""
}
cleanup() {
  local exit_code=$?
  stop_server
  nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits > "$ART/logs/nvidia_after.csv" 2>&1 || true
  echo finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ) >> "$ART/logs/lane.log" || true
  echo exit_code=$exit_code >> "$ART/logs/lane.log" || true
  exit "$exit_code"
}
trap cleanup EXIT
log_step "lane_start art=$ART sha=$SHA release_evidence=false no_live_vllm=true"
if ! command -v cmake >/dev/null 2>&1; then
  log_step "install_cmake_start"
  DEBIAN_FRONTEND=noninteractive apt-get update > "$ART/logs/apt-update.log" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y cmake > "$ART/logs/apt-install-cmake.log" 2>&1
  log_step "install_cmake_done"
fi
if ! command -v cargo >/dev/null 2>&1; then
  log_step "install_rust_start"
  curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable > "$ART/logs/rustup.log" 2>&1
  log_step "install_rust_done"
fi
source /root/.cargo/env
rustc --version > "$ART/env/rustc_version.txt"
cargo --version > "$ART/env/cargo_version.txt"
nvcc --version > "$ART/env/nvcc_version.txt" 2>&1 || true
nvidia-smi > "$ART/hardware/nvidia_smi_before.txt" 2>&1 || true
cd "$REPO"
log_step "git_sync_start"
git fetch origin goal/w2-w3-release-grade > "$ART/logs/git-fetch.log" 2>&1
git checkout goal/w2-w3-release-grade > "$ART/logs/git-checkout.log" 2>&1
git reset --hard "$SHA" > "$ART/logs/git-reset.log" 2>&1
git rev-parse HEAD > "$ART/env/git_sha.txt"
git status --short --branch --untracked-files=no > "$ART/env/git_status_short.txt"
if [ "$(git rev-parse HEAD)" != "$SHA" ]; then
  log_step "git_sha_mismatch"
  exit 10
fi
if [ -n "$(git status --short --untracked-files=no)" ]; then
  log_step "git_dirty_after_reset"
  exit 11
fi
log_step "git_sync_done"
DATASET=$REPO/docs/goals/model-coverage-2026-06-12/artifacts/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=$(find /workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots -mindepth 1 -maxdepth 1 -type d | sort | head -n 1 || true)
if [ -z "${TOKENIZER:-}" ] || [ ! -d "$TOKENIZER" ]; then
  log_step "tokenizer_missing"
  exit 12
fi
{
  echo lane=w3-qwen35-fp16-recurrent-state-cuda-build-smoke
  echo expected_runtime_cost='20-45min at dph_total about 0.4777777778/hr'
  echo stop_condition='stop after CUDA build/smoke/diagnostic pass or first failure'
  echo model=$MODEL
  echo tokenizer=$TOKENIZER
  echo dataset=$DATASET
  echo repo=$REPO
  echo art=$ART
  echo port=$PORT
  echo git_head=$(git rev-parse HEAD)
  echo git_status_start
  git status --short --branch --untracked-files=no
  echo git_status_end
  rustc --version
  cargo --version
  nvcc --version | tail -n 1 || true
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader,nounits
} > "$ART/logs/preflight.log" 2>&1
CHECK_CMD=(cargo check -p ferrum-models --features cuda)
printf "%q " "${CHECK_CMD[@]}" > "$ART/env/cargo-check.command.txt"; echo >> "$ART/env/cargo-check.command.txt"
log_step "cargo_check_start"
set +e
"${CHECK_CMD[@]}" > "$ART/logs/cargo-check.stdout.log" 2> "$ART/logs/cargo-check.stderr.log"
CHECK_EXIT=$?
set -e
echo "$CHECK_EXIT" > "$ART/env/cargo-check.exit"
if [ "$CHECK_EXIT" -ne 0 ]; then
  tail -n 260 "$ART/logs/cargo-check.stderr.log" > "$ART/logs/cargo-check.failure.tail.txt" 2>/dev/null || true
  log_step "cargo_check_failed exit=$CHECK_EXIT"
  exit 20
fi
log_step "cargo_check_done"
BUILD_CMD=(cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source)
printf "%q " "${BUILD_CMD[@]}" > "$ART/env/build.command.txt"; echo >> "$ART/env/build.command.txt"
log_step "release_build_start"
set +e
"${BUILD_CMD[@]}" > "$ART/logs/build.stdout.log" 2> "$ART/logs/build.stderr.log"
BUILD_EXIT=$?
set -e
echo "$BUILD_EXIT" > "$ART/env/build.exit"
if [ "$BUILD_EXIT" -ne 0 ]; then
  tail -n 260 "$ART/logs/build.stderr.log" > "$ART/logs/build.failure.tail.txt" 2>/dev/null || true
  log_step "release_build_failed exit=$BUILD_EXIT"
  exit 21
fi
sha256sum target/release/ferrum > "$ART/env/ferrum.sha256"
log_step "release_build_done"
RUN_CMD=(target/release/ferrum run "$MODEL" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --max-num-seqs 32 --kv-capacity 512 --max-num-batched-tokens 192 --temperature 0 --max-tokens 8 --output-format jsonl --prompt "What is 2+3? Answer with only the number." --effective-config-json "$ART/run/effective_config.json" --decision-trace-jsonl "$ART/run/decision_trace.jsonl")
printf "%q " "${RUN_CMD[@]}" > "$ART/run/run.command.txt"; echo >> "$ART/run/run.command.txt"
log_step "ferrum_run_smoke_start"
set +e
"${RUN_CMD[@]}" > "$ART/run/run.stdout.jsonl" 2> "$ART/run/run.stderr.log"
RUN_EXIT=$?
set -e
echo "$RUN_EXIT" > "$ART/run/run.exit"
if [ "$RUN_EXIT" -ne 0 ]; then
  tail -n 260 "$ART/run/run.stderr.log" > "$ART/run/run.failure.tail.txt" 2>/dev/null || true
  log_step "ferrum_run_smoke_failed exit=$RUN_EXIT"
  exit 30
fi
python3 - "$ART/run/run.stdout.jsonl" <<'PY'
import json, sys
content=""
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    line=line.strip()
    if not line:
        continue
    obj=json.loads(line)
    if obj.get("event") == "assistant":
        content += obj.get("content", "")
if "5" not in content:
    raise SystemExit(f"run smoke missing expected answer, content={content!r}")
PY
log_step "ferrum_run_smoke_done"
SERVE_CMD=(target/release/ferrum serve "$MODEL" --host 127.0.0.1 --port "$PORT" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --max-num-seqs 32 --kv-capacity 512 --max-num-batched-tokens 192 --scheduler-prefill-first-until-active 32 --scheduler-prefill-step-chunk 6 --effective-config-json "$ART/server/effective_config.json" --decision-trace-jsonl "$ART/server/decision_trace.jsonl" --scheduler-trace-jsonl "$ART/server/scheduler_trace.jsonl")
printf "%q " "${SERVE_CMD[@]}" > "$ART/server/serve.command.txt"; echo >> "$ART/server/serve.command.txt"
log_step "serve_start"
"${SERVE_CMD[@]}" > "$ART/server/serve.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$ART/server/server.pid"
READY=0
for i in $(seq 1 900); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo server_exited_before_ready > "$ART/server/serve.exit"
    tail -n 260 "$ART/server/serve.log" > "$ART/server/serve.failure.tail.txt" 2>/dev/null || true
    log_step "serve_exited_before_ready"
    exit 31
  fi
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > "$ART/server/models.json" 2> "$ART/server/models.curl.err"; then
    READY=1
    break
  fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo server_not_ready > "$ART/server/serve.exit"
  log_step "serve_not_ready"
  exit 32
fi
log_step "serve_ready"
jq -e '.selected_max_sequences == 32 and .selected_recurrent_state_max_slots == 32 and .selected_admission_limit == 32 and .admission.memory_estimate.recurrent_state_bytes_per_sequence == 32931840' "$ART/server/effective_config.json" > "$ART/server/effective_config.assert.txt" 2>&1 || {
  cp "$ART/server/effective_config.json" "$ART/server/effective_config.assert.failure.json"
  log_step "effective_config_assert_failed"
  exit 33
}
python3 - "$PORT" "$ART/server/chat_smoke_response.json" <<'PY'
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
log_step "serve_chat_smoke_done"
BENCH_REPORT=$ART/perf/bench_ferrum_c32_32x1.json
BENCH_CMD=(target/release/ferrum bench-serve --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --tokenizer "$TOKENIZER" --dataset sharegpt --sharegpt-path "$DATASET" --random-output-len 128 --ignore-eos --concurrency 32 --num-prompts 32 --warmup-requests 4 --n-repeats 1 --fail-on-error --seed 9271 --timeout 600 --out "$BENCH_REPORT")
printf "%q " "${BENCH_CMD[@]}" > "$ART/perf/bench-ferrum-c32.command.txt"; echo >> "$ART/perf/bench-ferrum-c32.command.txt"
log_step "bench_c32_start"
set +e
"${BENCH_CMD[@]}" > "$ART/perf/bench.stdout.log" 2> "$ART/perf/bench.stderr.log"
BENCH_EXIT=$?
set -e
echo "$BENCH_EXIT" > "$ART/perf/bench.exit"
if [ "$BENCH_EXIT" -ne 0 ]; then
  tail -n 260 "$ART/perf/bench.stderr.log" > "$ART/perf/bench.failure.tail.txt" 2>/dev/null || true
  log_step "bench_c32_failed exit=$BENCH_EXIT"
  exit 50
fi
log_step "bench_c32_done"
stop_server
python3 - <<PY | tee "$ART/perf/bench_metrics.txt"
import json
from pathlib import Path
p=Path("$BENCH_REPORT")
data=json.loads(p.read_text())
rows=data if isinstance(data, list) else [data]
for r in rows:
    print("FP16_STATE_C32_DIAG", "c", r.get("concurrency"), "output_tps", r.get("output_throughput_tps",{}).get("mean"), "completed", r.get("completed_per_run"), "errored", r.get("errored_per_run"), "ttft_p95", r.get("ttft_ms",{}).get("p95",{}).get("mean"), "itl_p95", r.get("itl_ms",{}).get("p95",{}).get("mean"))
PY
python3 - <<PY
import json
from collections import Counter
from pathlib import Path
trace=Path("$ART/server/scheduler_trace.jsonl")
serve=Path("$ART/server/serve.log")
summary={"trace": str(trace), "lines": 0, "result_counts": {}, "max_active_len": 0, "max_admitted_total": 0, "max_cancelled_total": 0, "max_completed_total": 0, "max_generated_tokens_seen_in_prefill": 0, "prefill_with_generated_tokens_iterations": 0, "prefill_with_generated_tokens_observations": 0, "serve_log_counts": {}}
results=Counter()
if trace.exists():
    with trace.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj=json.loads(line)
            except json.JSONDecodeError:
                continue
            summary["lines"] += 1
            results[str(obj.get("result"))] += 1
            for key in ("scheduler_before", "scheduler_after_schedule", "scheduler_after_process"):
                s=obj.get(key) or {}
                summary["max_active_len"]=max(summary["max_active_len"], int(s.get("active_len") or 0))
                summary["max_admitted_total"]=max(summary["max_admitted_total"], int(s.get("admitted_total") or 0))
                summary["max_cancelled_total"]=max(summary["max_cancelled_total"], int(s.get("cancelled_total") or 0))
                summary["max_completed_total"]=max(summary["max_completed_total"], int(s.get("completed_total") or 0))
            saw=False
            for req in ((obj.get("plan") or {}).get("requests") or []):
                if req.get("phase") == "Prefilling":
                    gen=int(req.get("generated_tokens") or 0)
                    if gen > 0:
                        saw=True
                        summary["prefill_with_generated_tokens_observations"] += 1
                        summary["max_generated_tokens_seen_in_prefill"] = max(summary["max_generated_tokens_seen_in_prefill"], gen)
            if saw:
                summary["prefill_with_generated_tokens_iterations"] += 1
summary["result_counts"] = dict(sorted(results.items()))
if serve.exists():
    needles=("cancelled during decode", "recurrent-state alloc deferred", "Unified prefill alloc deferred", "ResourceExhausted", "OOM", "out of memory", "panicked")
    counts=Counter()
    with serve.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for n in needles:
                if n in line:
                    counts[n] += 1
    summary["serve_log_counts"] = dict(counts)
Path("$ART/server/scheduler_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
gzip -f "$ART/server/scheduler_trace.jsonl" 2>/dev/null || true
gzip -f "$ART/server/serve.log" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits > "$ART/logs/nvidia_final.csv" 2>&1 || true
echo "FERRUM W3 QWEN35 FP16 RECURRENT STATE C32 DIAG PASS: $ART" | tee "$ART/summary.txt"
log_step "lane_pass"
