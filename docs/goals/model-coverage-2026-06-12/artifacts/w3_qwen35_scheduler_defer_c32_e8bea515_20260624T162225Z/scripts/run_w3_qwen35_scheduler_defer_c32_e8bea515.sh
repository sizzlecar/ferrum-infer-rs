#!/usr/bin/env bash
set -Eeuo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HF_HOME=/workspace/hf-cache
export NO_COLOR=1
SHA=e8bea515f257bc6545abcee34b96e92db4d4ce65
MODEL=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
REPO=/workspace/ferrum-infer-rs
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ART=/workspace/artifacts/w3_qwen35_scheduler_defer_c32_${SHA:0:8}_${STAMP}
PORT=55989
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
  echo lane=w3-qwen35-e8bea515-scheduler-defer-c32-diagnostic
  echo expected_runtime_cost="15-35min at dph_total about 0.4777777778/hr if target cache is retained"
  echo stop_condition="stop after build/run/serve/c32 diagnostic pass or first failure/stall evidence"
  echo model=$MODEL
  echo tokenizer=$TOKENIZER
  echo dataset=$DATASET
  echo repo=$REPO
  echo art=$ART
  echo port=$PORT
} > "$ART/logs/preflight.log"
log_step "cargo_check_start"
printf "%q " cargo check -p ferrum-engine -p ferrum-scheduler > "$ART/env/cargo-check.command.txt"; echo >> "$ART/env/cargo-check.command.txt"
set +e
cargo check -p ferrum-engine -p ferrum-scheduler > "$ART/logs/cargo-check.stdout.log" 2> "$ART/logs/cargo-check.stderr.log"
CHECK_EXIT=$?
set -e
echo "$CHECK_EXIT" > "$ART/env/cargo-check.exit"
if [ "$CHECK_EXIT" -ne 0 ]; then
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
  log_step "ferrum_run_smoke_failed exit=$RUN_EXIT"
  exit 30
fi
if ! grep -q '"content":"5"' "$ART/run/run.stdout.jsonl"; then
  log_step "ferrum_run_smoke_wrong_answer"
  exit 31
fi
jq -e '.selected_max_sequences == 32 and .selected_recurrent_state_max_slots == 32 and .selected_admission_limit == 32' "$ART/run/effective_config.json" > "$ART/run/effective_config.assert.txt"
log_step "ferrum_run_smoke_done"
SERVE_CMD=(target/release/ferrum serve "$MODEL" --host 127.0.0.1 --port "$PORT" --backend cuda --gpu-devices 0 --gpu-memory-utilization 0.90 --max-num-seqs 32 --kv-capacity 512 --max-num-batched-tokens 192 --scheduler-prefill-first-until-active 32 --scheduler-prefill-step-chunk 6 --effective-config-json "$ART/server/effective_config.json" --decision-trace-jsonl "$ART/server/decision_trace.jsonl" --scheduler-trace-jsonl "$ART/server/scheduler_trace.jsonl")
printf "%q " "${SERVE_CMD[@]}" > "$ART/server/serve.command.txt"; echo >> "$ART/server/serve.command.txt"
log_step "serve_start"
"${SERVE_CMD[@]}" > "$ART/server/serve.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$ART/server/server.pid"
for i in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log_step "serve_died_before_ready"
    exit 40
  fi
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$ART/server/models.json" 2> "$ART/server/models.curl.err"; then
    break
  fi
  sleep 1
done
if ! curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$ART/server/models.json" 2> "$ART/server/models.curl.err"; then
  log_step "serve_models_failed"
  exit 41
fi
jq -e '.selected_max_sequences == 32 and .selected_recurrent_state_max_slots == 32 and .selected_admission_limit == 32 and .admission.memory_estimate.recurrent_state_bytes_per_sequence == 32931840' "$ART/server/effective_config.json" > "$ART/server/effective_config.assert.txt"
log_step "serve_ready"
CHAT_PAYLOAD='{"model":"Qwen/Qwen3.5-35B-A3B-GPTQ-Int4","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"temperature":0,"max_tokens":8}'
curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' --data "$CHAT_PAYLOAD" > "$ART/server/chat_smoke_response.json"
if ! jq -e '.choices[0].message.content | contains("5")' "$ART/server/chat_smoke_response.json" >/dev/null; then
  log_step "serve_chat_smoke_wrong_answer"
  exit 42
fi
log_step "serve_chat_smoke_done"
BENCH_REPORT="$ART/perf/bench_ferrum_c32_32x1.json"
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
    print("SCHED_DEFER_C32_DIAG", "c", r.get("concurrency"), "output_tps", r.get("output_throughput_tps",{}).get("mean"), "completed", r.get("completed_per_run"), "errored", r.get("errored_per_run"), "ttft_p95", r.get("ttft_ms",{}).get("p95",{}).get("mean"), "itl_p95", r.get("itl_ms",{}).get("p95",{}).get("mean"))
PY
python3 - <<PY
import json
from collections import Counter
from pathlib import Path
trace=Path("$ART/server/scheduler_trace.jsonl")
serve=Path("$ART/server/serve.log")
summary={"trace": str(trace), "lines": 0, "result_counts": {}, "max_active_len": 0, "max_waiting_queue_len": 0, "max_prefill_queue_len": 0, "max_decode_queue_len": 0, "max_admitted_total": 0, "max_cancelled_total": 0, "max_completed_total": 0, "max_failed_total": 0, "last_completed_total": None, "last_cancelled_total": None, "last_admitted_total": None, "last_prefill_delta": None, "last_decode_delta": None, "prefill_with_generated_tokens_iterations": 0, "serve_log_counts": {}}
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
                summary["max_waiting_queue_len"]=max(summary["max_waiting_queue_len"], int(s.get("waiting_queue_len") or 0))
                summary["max_prefill_queue_len"]=max(summary["max_prefill_queue_len"], int(s.get("prefill_queue_len") or 0))
                summary["max_decode_queue_len"]=max(summary["max_decode_queue_len"], int(s.get("decode_queue_len") or 0))
                summary["max_admitted_total"]=max(summary["max_admitted_total"], int(s.get("admitted_total") or 0))
                summary["max_cancelled_total"]=max(summary["max_cancelled_total"], int(s.get("cancelled_total") or 0))
                summary["max_completed_total"]=max(summary["max_completed_total"], int(s.get("completed_total") or 0))
                summary["max_failed_total"]=max(summary["max_failed_total"], int(s.get("failed_total") or 0))
            s=obj.get("scheduler_after_process") or {}
            summary["last_completed_total"]=int(s.get("completed_total") or 0)
            summary["last_cancelled_total"]=int(s.get("cancelled_total") or 0)
            summary["last_admitted_total"]=int(s.get("admitted_total") or 0)
            counters=obj.get("engine_counters") or {}
            summary["last_prefill_delta"]=counters.get("prefill_tokens_delta")
            summary["last_decode_delta"]=counters.get("decode_tokens_delta")
            saw=False
            for req in ((obj.get("plan") or {}).get("requests") or []):
                if req.get("phase") == "Prefilling" and int(req.get("generated_tokens") or 0) > 0:
                    saw=True
            if saw:
                summary["prefill_with_generated_tokens_iterations"] += 1
summary["result_counts"] = dict(sorted(results.items()))
if serve.exists():
    needles=("Unified prefill alloc deferred", "recurrent-state alloc deferred", "ResourceExhausted", "OOM", "out of memory", "panicked", "ERROR", "WARN", "cancelled during decode", "cancelled during prefill")
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
gzip -f "$ART/logs/build.stderr.log" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits > "$ART/logs/nvidia_final.csv" 2>&1 || true
echo "FERRUM W3 QWEN35 SCHEDULER DEFER C32 DIAG PASS: $ART" | tee "$ART/summary.txt"
log_step "lane_pass"
