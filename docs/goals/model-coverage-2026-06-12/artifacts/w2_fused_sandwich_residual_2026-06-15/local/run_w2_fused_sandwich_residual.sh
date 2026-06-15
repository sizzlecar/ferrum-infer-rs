#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ferrum-infer-rs

if [ -f /root/.cargo/env ]; then
  . /root/.cargo/env
fi
export PATH="/root/.cargo/bin:${PATH}"

OUT=/workspace/w2_fused_sandwich_residual_2026-06-15
DATASET=/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
PORT=8474

mkdir -p "$OUT/build" "$OUT/run" "$OUT/server" "$OUT/profile" "$OUT/remote"
date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short_after_source_sync.txt" || true
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"
rustc --version > "$OUT/remote/rustc_version.txt"
cargo --version > "$OUT/remote/cargo_version.txt"
nvcc --version > "$OUT/remote/nvcc_version.txt"

cat > "$OUT/gpu_contract.md" <<'TXT'
lane: W2 Gemma3 CUDA fused sandwich residual-add minimal validation
instance: Vast 40826362, 1x RTX 4090, cache-retained native CUDA machine
expected_runtime_cost: 15-35min, hard cap 45min, reused instance at about USD 0.425/hr when running
stop_condition: start/SSH/CUDA/source sync/build first failure, ferrum run correctness failure, serve/bench diagnostic complete and artifacts copied, or 45min hard cap
correctness_gate: CUDA release build plus product ferrum run smoke before any performance diagnostic; serve readiness and bench-serve --fail-on-error for product serve path
performance_command: diagnostic-only natural ASCII ShareGPT c16/c32 small sample with bench-serve --fail-on-error, seed 9271, n_repeats=1, FERRUM_DECODE_OP_PROFILE=1
release_grade_status: not release-grade evidence; no MODEL_RELEASE_GRADE_W2 PASS is expected from this checkpoint
TXT

cat > "$OUT/build/release_build.command.txt" <<'TXT'
CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
TXT
if ! CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$OUT/build/release_build.log" 2>&1; then
  echo FAIL > "$OUT/run.status"
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit 1
fi
sha256sum target/release/ferrum > "$OUT/build/ferrum.sha256"

cat > "$OUT/run/ferrum-run.command.txt" <<TXT
timeout 1200 target/release/ferrum run gemma3:27b-gptq \
  --backend cuda \
  --prompt "What is 2+3? Answer with just the number." \
  --max-tokens 64 \
  --temperature 0 \
  --kv-capacity 2560 \
  --max-num-seqs 2 \
  --output-format jsonl \
  --effective-config-json "$OUT/run/run_effective_config.json" \
  --decision-trace-jsonl "$OUT/run/run_decision_trace.jsonl"
TXT
set +e
timeout 1200 target/release/ferrum run gemma3:27b-gptq \
  --backend cuda \
  --prompt "What is 2+3? Answer with just the number." \
  --max-tokens 64 \
  --temperature 0 \
  --kv-capacity 2560 \
  --max-num-seqs 2 \
  --output-format jsonl \
  --effective-config-json "$OUT/run/run_effective_config.json" \
  --decision-trace-jsonl "$OUT/run/run_decision_trace.jsonl" \
  > "$OUT/run/ferrum-run.stdout" 2> "$OUT/run/ferrum-run.stderr"
RUN_RC=$?
set -e
echo "$RUN_RC" > "$OUT/run/ferrum-run.rc"
if [ "$RUN_RC" -ne 0 ]; then
  echo FAIL > "$OUT/run.status"
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit "$RUN_RC"
fi

pkill -f "target/release/ferrum serve" 2>/dev/null || true
sleep 2

cat > "$OUT/server/serve.command.json" <<JSON
{"env":{"FERRUM_DECODE_OP_PROFILE":"1"},"cmd":["target/release/ferrum","serve","--model","gemma3:27b-gptq","--port","$PORT","--kv-capacity","512","--max-num-seqs","16","--effective-config-json","$OUT/server/serve_effective_config.json","--decision-trace-jsonl","$OUT/server/serve_decision_trace.jsonl"]}
JSON
FERRUM_DECODE_OP_PROFILE=1 target/release/ferrum serve \
  --model gemma3:27b-gptq \
  --port "$PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/server/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl" \
  > "$OUT/server/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server/server.pid"

cleanup_server() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup_server EXIT

ready=0
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
    > "$OUT/server/models.json" 2> "$OUT/server/models.err"; then
    ready=1
    echo "$i" > "$OUT/server/ready_poll.txt"
    break
  fi
  sleep 2
done

if [ "$ready" != 1 ]; then
  echo FAIL > "$OUT/run.status"
  tail -n 200 "$OUT/server/server.log" > "$OUT/server/server_tail_on_fail.log" || true
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit 1
fi

cat > "$OUT/profile/bench-serve.command.json" <<JSON
{"cmd":["timeout","1800","target/release/ferrum","bench-serve","--base-url","http://127.0.0.1:$PORT","--model","gemma3:27b-gptq","--tokenizer","$TOKENIZER","--dataset","sharegpt","--sharegpt-path","$DATASET","--random-output-len","64","--concurrency-sweep","16,32","--num-prompts","16","--n-repeats","1","--fail-on-error","--seed","9271","--out","$OUT/profile/bench_sharegpt_c16_c32_16x1_fused_sandwich.json"]}
JSON
set +e
timeout 1800 target/release/ferrum bench-serve \
  --base-url "http://127.0.0.1:${PORT}" \
  --model gemma3:27b-gptq \
  --tokenizer "$TOKENIZER" \
  --dataset sharegpt \
  --sharegpt-path "$DATASET" \
  --random-output-len 64 \
  --concurrency-sweep 16,32 \
  --num-prompts 16 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271 \
  --out "$OUT/profile/bench_sharegpt_c16_c32_16x1_fused_sandwich.json" \
  > "$OUT/profile/bench-serve.stdout" 2> "$OUT/profile/bench-serve.stderr"
BENCH_RC=$?
set -e
echo "$BENCH_RC" > "$OUT/profile/bench-serve.rc"
nvidia-smi > "$OUT/remote/nvidia_smi_after_bench.txt" || true

python3 - <<PY > "$OUT/profile/op_profile_summary.json"
import json, pathlib, re, statistics

out = pathlib.Path("$OUT")
text = (out / "server/server.log").read_text(errors="replace")
header_pat = re.compile(r"\\[batched-op-profile\\] m=(\\d+) total=(\\d+)us")
metric_pat = re.compile(r"([a-z_]+)=([0-9]+)us\\(([0-9]+)\\)")
rows = []
for line in text.splitlines():
    if "[batched-op-profile]" not in line:
        continue
    header = header_pat.search(line)
    if not header:
        continue
    row = {"batch": int(header.group(1)), "total_us": int(header.group(2))}
    for name, us, calls in metric_pat.findall(line):
        row[f"{name}_us"] = int(us)
        row[f"{name}_calls"] = int(calls)
    rows.append(row)

keys = sorted({key for row in rows for key in row if key.endswith("_us") and key != "total_us"})
summary = {"bench_rc": $BENCH_RC, "rows": len(rows), "keys": keys, "by_batch": {}}
for batch in sorted({row["batch"] for row in rows}):
    group = [row for row in rows if row["batch"] == batch]
    totals = [row["total_us"] for row in group]
    item = {
        "count": len(group),
        "total_us_mean": statistics.mean(totals),
        "total_us_p95": sorted(totals)[int(0.95 * (len(totals) - 1))],
    }
    total = item["total_us_mean"] or 1
    for key in keys:
        vals = [row.get(key, 0) for row in group]
        item[f"{key}_mean"] = statistics.mean(vals)
        item[f"{key}_p95"] = sorted(vals)[int(0.95 * (len(vals) - 1))]
        item[f"{key.removesuffix('_us')}_share_mean"] = item[f"{key}_mean"] / total
    summary["by_batch"][str(batch)] = item

print(json.dumps(summary, indent=2, sort_keys=True))
PY

if [ "$BENCH_RC" -eq 0 ]; then
  echo PASS > "$OUT/run.status"
else
  echo FAIL > "$OUT/run.status"
fi
date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
exit "$BENCH_RC"
