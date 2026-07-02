#!/usr/bin/env bash
set -euo pipefail
ART="$(dirname "$0")"
REPO=/workspace/ferrum-w3-qwen35-8cfa422c
TARGET_DIR=/workspace/ferrum-w3-qwen35-ac98207/target
MODEL=/workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b
MODEL_ID=3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b
DATASET=/workspace/artifacts/ferrum_w3_sharegpt_ab_20b8946b_20260619T121844Z_ninja/dataset/ascii_sharegpt_w3_100_58d5721d.jsonl
PORT=58080
SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
if [[ -f /root/.cargo/env ]]; then
  source /root/.cargo/env
fi
export CARGO_TARGET_DIR="$TARGET_DIR"
export HF_HOME=/workspace/hf-cache
export RUST_BACKTRACE=1
{
  echo "artifact=$ART"
  date -u +%Y-%m-%dT%H:%M:%SZ
  echo "repo=$REPO"
  echo "target_dir=$TARGET_DIR"
} > "$ART/remote/start.txt"
cd "$REPO"
git rev-parse HEAD > "$ART/remote/git_head.txt"
git status --short > "$ART/remote/git_status_short.txt"
rustc --version > "$ART/remote/rustc_version.txt"
cargo --version > "$ART/remote/cargo_version.txt"
nvcc --version > "$ART/remote/nvcc_version.txt" 2>&1
nvidia-smi > "$ART/remote/nvidia_smi_before.txt"
BUILD_CMD=(cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source)
printf '%q ' "${BUILD_CMD[@]}" > "$ART/build/release_build.command.txt"
echo >> "$ART/build/release_build.command.txt"
"${BUILD_CMD[@]}" > "$ART/build/release_build.log" 2>&1
BIN="$TARGET_DIR/release/ferrum"
sha256sum "$BIN" > "$ART/build/ferrum.sha256"
SERVE_CMD=("$BIN" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --backend cuda --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 4096 --effective-config-json "$ART/server/serve_effective_config.json" --decision-trace-jsonl "$ART/server/serve_decision_trace.jsonl")
printf '%q ' "${SERVE_CMD[@]}" > "$ART/server/ferrum_server.command.txt"
echo >> "$ART/server/ferrum_server.command.txt"
"${SERVE_CMD[@]}" > "$ART/server/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$ART/server/server.pid"
ready=0
for i in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > "$ART/server/models.json" 2> "$ART/server/models.err"; then
    ready=1
    echo "ready_poll=$i" > "$ART/server/ready_poll.txt"
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "server exited before ready at poll $i" > "$ART/run.status"
    tail -200 "$ART/server/server.log" > "$ART/server/server.tail.txt" || true
    exit 20
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  echo "server not ready after 600s" > "$ART/run.status"
  tail -200 "$ART/server/server.log" > "$ART/server/server.tail.txt" || true
  exit 21
fi
nvidia-smi > "$ART/perf/nvidia_smi_before_bench.txt"
BENCH_CMD=("$BIN" bench-serve --base-url "http://127.0.0.1:$PORT" --model "$MODEL_ID" --tokenizer "$MODEL" --dataset sharegpt --sharegpt-path "$DATASET" --random-output-len 128 --concurrency-sweep 1 --num-prompts 100 --warmup-requests 10 --n-repeats 1 --fail-on-error --seed 9271 --output json --out "$ART/perf/bench_ferrum_sharegpt_c1_100x1.json")
printf '%q ' "${BENCH_CMD[@]}" > "$ART/perf/bench-ferrum.command.txt"
echo >> "$ART/perf/bench-ferrum.command.txt"
set +e
"${BENCH_CMD[@]}" > "$ART/perf/bench-ferrum.stdout" 2> "$ART/perf/bench-ferrum.stderr"
bench_rc=$?
set -e
echo "$bench_rc" > "$ART/perf/bench-ferrum.rc"
curl -sf "http://127.0.0.1:$PORT/health" > "$ART/server/health_after_bench.json" 2> "$ART/server/health_after_bench.err" || true
nvidia-smi > "$ART/perf/nvidia_smi_after_bench.txt"
python3 - "$ART" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
out = {"artifact": str(root)}
rc = int((root / "perf/bench-ferrum.rc").read_text().strip())
out["bench_rc"] = rc
p = root / "perf/bench_ferrum_sharegpt_c1_100x1.json"
if p.exists():
    data = json.loads(p.read_text())
    out["top_level_keys"] = sorted(data)[:40]
    text = json.dumps(data)
    out["contains_error_word"] = "error" in text.lower()
    if isinstance(data, dict):
        for key in ("summary", "results", "runs", "measurements"):
            if key in data:
                out[key] = data[key]
with open(root / "perf/diagnostic_summary.json", "w") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print(json.dumps(out, indent=2, sort_keys=True))
PY
if [[ "$bench_rc" -eq 0 ]]; then
  echo "W3 QWEN35 ARGMAX DIAG SMOKE PASS: $ART" > "$ART/run.status"
else
  echo "W3 QWEN35 ARGMAX DIAG SMOKE FAIL: $ART" > "$ART/run.status"
  exit "$bench_rc"
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$ART/remote/end_utc.txt"
