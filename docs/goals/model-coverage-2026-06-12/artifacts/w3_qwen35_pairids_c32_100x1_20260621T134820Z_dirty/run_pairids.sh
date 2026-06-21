#!/usr/bin/env bash
set -euo pipefail
ART=$(cd "$(dirname "$0")" && pwd)
cd /workspace/ferrum-w3-clean-b6
export PATH=/root/.cargo/bin:$PATH
export HF_HOME=/workspace/hf-cache
export FERRUM_VLLM_MOE_PAIR_IDS=1
MODEL=/workspace/hf-cache/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b
DATA=/workspace/artifacts/ferrum_w3_sharegpt_ab_20b8946b_20260619T121844Z_ninja/dataset/ascii_sharegpt_w3_100_58d5721d.jsonl
PORT=18267
{
  echo "lane=W3 Qwen35 c32 pair-id MoE diagnostic"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "head=$(git rev-parse HEAD)"
  echo "status_short_begin<<STATUS"
  git status --short crates/ferrum-models/src/models/qwen35.rs
  echo "STATUS"
  echo "FERRUM_VLLM_MOE_PAIR_IDS=$FERRUM_VLLM_MOE_PAIR_IDS"
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
  --num-prompts 100 \
  --warmup-requests 10 \
  --n-repeats 1 \
  --concurrency 32 \
  --fail-on-error \
  --seed 9271 \
  --hw-id rtx-4090 \
  --commit-sha 690ba923-dirty-linear-scratch-pairids \
  --out "$ART/bench_c32_100x1.json" > "$ART/bench_c32_100x1.log" 2>&1
python3 - <<PY > "$ART/summary.json"
import json, pathlib
art = pathlib.Path("$ART")
d = json.loads((art / "bench_c32_100x1.json").read_text())
summary = {
  "artifact": str(art),
  "output_tps": d.get("output_throughput_tps"),
  "completed": d.get("completed_per_run"),
  "errored": d.get("errored_per_run"),
  "bad_output": d.get("bad_output_per_run"),
  "missing_done": d.get("missing_done_per_run"),
  "duplicate_done": d.get("duplicate_done_per_run"),
  "zero_output_tokens": d.get("zero_output_tokens_per_run"),
  "output_token_count_source": d.get("output_token_count_source"),
  "itl_p50_ms": (((d.get("itl_ms") or {}).get("p50") or {}).get("mean")),
  "tpot_p50_ms": (((d.get("tpot_ms") or {}).get("p50") or {}).get("mean")),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version --format=csv > "$ART/gpu_after.csv"
echo "QWEN35_PAIRIDS_C32_100X1_DIAG_PASS $ART" | tee "$ART/result.txt"
