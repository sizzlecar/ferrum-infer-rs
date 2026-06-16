#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/ferrum-infer-rs}"
OUT="${OUT:-/workspace/w2_active_decode_prefill_budget_c16_cuda_2026-06-17}"
MODEL="${MODEL:-gemma3:27b-gptq}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
DATASET="${DATASET:-/workspace/ascii_sharegpt_w2_100.jsonl}"
RUN_PORT="${RUN_PORT:-18150}"
BENCH_PORT="${BENCH_PORT:-18151}"

VLLM_REF_TPS_LCB="${VLLM_REF_TPS_LCB:-478.39462812583776}"
VLLM_REF_TPS_MEAN="${VLLM_REF_TPS_MEAN:-500.67038762731977}"
VLLM_REF_P95_ITL="${VLLM_REF_P95_ITL:-33.06958213333332}"
PREV_FERRUM_TPS_LCB="${PREV_FERRUM_TPS_LCB:-414.59153186899397}"
PREV_FERRUM_TPS_MEAN="${PREV_FERRUM_TPS_MEAN:-422.34520497237537}"
PREV_FERRUM_P95_ITL="${PREV_FERRUM_P95_ITL:-52.81935383333333}"

mkdir -p "$OUT"/{env,build,correctness,server,perf}
exec > >(tee -a "$OUT/run.log") 2>&1

echo "[w2-post-sched-c16] start $(date -u +%FT%TZ)"
echo "[w2-post-sched-c16] root=$ROOT out=$OUT model=$MODEL hf_home=$HF_HOME dataset=$DATASET"
date -u +%FT%TZ > "$OUT/env/start_utc.txt"

SERVER_PID=""
cleanup_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap cleanup_server EXIT

wait_http() {
  local base="$1"
  local log_file="$2"
  local label="$3"
  for _ in $(seq 1 360); do
    if curl -sf --noproxy '*' --max-time 2 "$base/v1/models" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[w2-post-sched-c16] $label server exited" >&2
      tail -120 "$log_file" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "[w2-post-sched-c16] timed out waiting for $label server" >&2
  tail -120 "$log_file" >&2 || true
  return 1
}

run_stream_smoke() {
  local base="$1"
  local model="$2"
  local out_dir="$3"
  mkdir -p "$out_dir"
  python3 - <<'PY' "$base" "$model" "$out_dir"
import json
import sys
import urllib.request
from pathlib import Path

base, model, out_dir = sys.argv[1:4]
out = Path(out_dir)
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "What is 2+3? Answer with only the number."}],
    "max_tokens": 16,
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
}
out.joinpath("stream_request.json").write_text(json.dumps(payload, indent=2) + "\n")
req = urllib.request.Request(
    f"{base}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
done_count = 0
content = ""
usage = None
sse_lines = []
with urllib.request.urlopen(req, timeout=600) as resp:
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        if not line:
            continue
        sse_lines.append(line)
        if line == "data: [DONE]":
            done_count += 1
            continue
        if not line.startswith("data: "):
            raise SystemExit(f"malformed SSE line: {line!r}")
        chunk = json.loads(line[6:])
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            content += (choice.get("delta") or {}).get("content") or ""
out.joinpath("stream_response.sse").write_text("\n".join(sse_lines) + "\n")
summary = {"done_count": done_count, "content": content, "usage": usage}
out.joinpath("stream_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
if done_count != 1:
    raise SystemExit(f"expected exactly one DONE, got {done_count}")
if not content.strip():
    raise SystemExit("expected at least one output token")
if not usage or usage.get("completion_tokens", 0) <= 0:
    raise SystemExit(f"missing/invalid usage: {usage!r}")
PY
}

nvidia-smi > "$OUT/env/nvidia_smi_before.txt"
nvcc --version > "$OUT/env/nvcc_version.txt" 2>&1 || true
df -h /workspace / > "$OUT/env/df_before.txt" 2>&1 || true
sha256sum "$DATASET" > "$OUT/env/dataset.sha256"
wc -l "$DATASET" > "$OUT/env/dataset.wc"

MODEL_ROOT="$HF_HOME/hub/models--circulus--gemma-3-27b-it-gptq"
MODEL_PATH="$(find "$MODEL_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
  echo "[w2-post-sched-c16] missing model snapshot under $MODEL_ROOT/snapshots" >&2
  exit 20
fi
printf '%s\n' "$MODEL_PATH" > "$OUT/env/model_path.txt"
find "$MODEL_PATH" -maxdepth 1 -type f -printf '%f\n' | sort > "$OUT/env/model_files.txt"
du -sh "$HF_HOME" > "$OUT/env/hf_home_du.txt" 2>&1 || true

cd "$ROOT"
git rev-parse HEAD > "$OUT/env/git_sha.txt"
git status --short > "$OUT/env/git_status_short.txt"

if ! command -v cargo >/dev/null 2>&1; then
  echo "[w2-post-sched-c16] installing rustup minimal toolchain"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal > "$OUT/env/rustup_install.log" 2>&1
fi
# shellcheck disable=SC1090
source "$HOME/.cargo/env"
rustc -V > "$OUT/env/rustc_version.txt"
cargo -V > "$OUT/env/cargo_version.txt"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

printf '%q ' cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$OUT/build/build.command.txt"
printf '\n' >> "$OUT/build/build.command.txt"
time cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$OUT/build/cargo_build_release.log" 2>&1
BIN="$ROOT/target/release/ferrum"
sha256sum "$BIN" > "$OUT/env/ferrum.sha256"

echo "[w2-post-sched-c16] Ferrum run correctness"
python3 - <<PY
import json
from pathlib import Path
cmd = [
    "timeout", "1200", "$BIN", "run", "$MODEL",
    "--backend", "cuda",
    "--prompt", "What is 2+3? Answer with just the number.",
    "--max-tokens", "64",
    "--temperature", "0",
    "--kv-capacity", "2560",
    "--max-num-seqs", "2",
    "--output-format", "jsonl",
    "--effective-config-json", "$OUT/correctness/run_effective_config.json",
    "--decision-trace-jsonl", "$OUT/correctness/run_decision_trace.jsonl",
]
Path("$OUT/correctness/run.command.json").write_text(json.dumps({"cmd": cmd}, indent=2) + "\n")
PY
HF_HOME="$HF_HOME" timeout 1200 "$BIN" run "$MODEL" \
  --backend cuda \
  --prompt "What is 2+3? Answer with just the number." \
  --max-tokens 64 \
  --temperature 0 \
  --kv-capacity 2560 \
  --max-num-seqs 2 \
  --output-format jsonl \
  --effective-config-json "$OUT/correctness/run_effective_config.json" \
  --decision-trace-jsonl "$OUT/correctness/run_decision_trace.jsonl" \
  > "$OUT/correctness/run.stdout" 2> "$OUT/correctness/run.stderr"
python3 - <<'PY' "$OUT/correctness/run.stdout" "$OUT/correctness/run_summary.json"
import json
import sys
from pathlib import Path
lines = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
assistant = [line for line in lines if line.get("event") == "assistant"]
summary = {
    "assistant_events": len(assistant),
    "content": "".join(line.get("content", "") for line in assistant),
    "n_tokens": sum(int(line.get("n_tokens", 0)) for line in assistant),
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
if not summary["content"].strip():
    raise SystemExit("ferrum run produced no assistant content")
if summary["n_tokens"] <= 0:
    raise SystemExit("ferrum run reported zero output tokens")
PY

echo "[w2-post-sched-c16] Ferrum serve correctness"
HF_HOME="$HF_HOME" "$BIN" serve "$MODEL" \
  --backend cuda \
  --port "$RUN_PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/correctness/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/correctness/serve_decision_trace.jsonl" \
  > "$OUT/correctness/serve.log" 2>&1 &
SERVER_PID=$!
RUN_BASE="http://127.0.0.1:$RUN_PORT"
wait_http "$RUN_BASE" "$OUT/correctness/serve.log" "Ferrum correctness"
curl -sf --noproxy '*' "$RUN_BASE/v1/models" > "$OUT/correctness/models.json"
run_stream_smoke "$RUN_BASE" "$MODEL" "$OUT/correctness/smoke"
cleanup_server

echo "[w2-post-sched-c16] Ferrum serve + c16 bench"
HF_HOME="$HF_HOME" "$BIN" serve "$MODEL" \
  --backend cuda \
  --port "$BENCH_PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/server/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl" \
  > "$OUT/server/server.log" 2>&1 &
SERVER_PID=$!
BENCH_BASE="http://127.0.0.1:$BENCH_PORT"
wait_http "$BENCH_BASE" "$OUT/server/server.log" "Ferrum bench"
curl -sf --noproxy '*' "$BENCH_BASE/v1/models" > "$OUT/server/models.json"
python3 - <<'PY' "$OUT/server/serve_effective_config.json" "$OUT/server/active_chunk_check.json"
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
matches = []
def walk(node):
    if isinstance(node, dict):
        if node.get("key") == "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK":
            matches.append(node)
        for value in node.values():
            walk(value)
    elif isinstance(node, list):
        for value in node:
            walk(value)
walk(data)
Path(sys.argv[2]).write_text(json.dumps({"matches": matches}, indent=2, sort_keys=True) + "\n")
if not any(str(m.get("effective_value")) == "16" for m in matches):
    raise SystemExit("expected typed/default active decode prefill chunk value 16")
PY

BENCH_JSON="$OUT/perf/bench_ferrum_sharegpt_c16_100x3.json"
BENCH_CMD=(
  timeout 10800 "$BIN" bench-serve
  --base-url "$BENCH_BASE"
  --model "$MODEL"
  --tokenizer "$MODEL_PATH"
  --dataset sharegpt
  --sharegpt-path "$DATASET"
  --random-output-len 128
  --concurrency-sweep 16
  --num-prompts 100
  --n-repeats 3
  --fail-on-error
  --require-ci
  --seed 9271
  --out "$BENCH_JSON"
)
printf '%q ' "${BENCH_CMD[@]}" > "$OUT/perf/bench-ferrum.command.txt"
printf '\n' >> "$OUT/perf/bench-ferrum.command.txt"
"${BENCH_CMD[@]}" > "$OUT/perf/bench-ferrum.stdout" 2> "$OUT/perf/bench-ferrum.stderr"
cleanup_server
nvidia-smi > "$OUT/perf/nvidia_smi_after_bench.txt"

python3 - <<'PY' "$OUT" "$BENCH_JSON" "$VLLM_REF_TPS_MEAN" "$VLLM_REF_TPS_LCB" "$VLLM_REF_P95_ITL" "$PREV_FERRUM_TPS_MEAN" "$PREV_FERRUM_TPS_LCB" "$PREV_FERRUM_P95_ITL"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
bench = json.loads(Path(sys.argv[2]).read_text())
vllm_mean = float(sys.argv[3])
vllm_lcb = float(sys.argv[4])
vllm_p95 = float(sys.argv[5])
prev_mean = float(sys.argv[6])
prev_lcb = float(sys.argv[7])
prev_p95 = float(sys.argv[8])

def read(path):
    p = out / path
    return p.read_text().strip() if p.exists() else None

def lcb(bench):
    metric = bench["output_throughput_tps"]
    return metric["mean"] - metric.get("ci95_hw", 0.0)

def p95_itl(bench):
    return bench["itl_ms"]["p95"]["mean"]

def clean(bench):
    if any(x != 100 for x in bench.get("completed_per_run", [])):
        return False
    if any(x != 0 for x in bench.get("errored_per_run", [])):
        return False
    for issue in bench.get("quality_issues_per_run", []):
        if any(issue.get(k, 0) for k in issue):
            return False
    return bench.get("output_token_count_source") == "usage"

current_mean = bench["output_throughput_tps"]["mean"]
current_lcb = lcb(bench)
current_p95 = p95_itl(bench)
summary = {
    "lane": "w2_active_decode_prefill_budget_c16_cuda",
    "status": "diagnostic_pass" if clean(bench) else "diagnostic_fail_quality",
    "release_gate": False,
    "remote_git_sha": read("env/git_sha.txt"),
    "remote_git_status_short": read("env/git_status_short.txt"),
    "ferrum_binary_sha256": read("env/ferrum.sha256"),
    "dataset_sha256": read("env/dataset.sha256"),
    "model_path": read("env/model_path.txt"),
    "run_smoke": json.loads((out / "correctness/run_summary.json").read_text()),
    "serve_smoke": json.loads((out / "correctness/smoke/stream_summary.json").read_text()),
    "ferrum_completed_per_run": bench.get("completed_per_run"),
    "ferrum_errored_per_run": bench.get("errored_per_run"),
    "ferrum_quality_issues_per_run": bench.get("quality_issues_per_run"),
    "ferrum_output_token_count_source": bench.get("output_token_count_source"),
    "ferrum_output_tps_mean": current_mean,
    "ferrum_output_tps_lcb": current_lcb,
    "ferrum_p95_itl_ms_mean": current_p95,
    "same_pod_vllm_reference_output_tps_mean": vllm_mean,
    "same_pod_vllm_reference_output_tps_lcb": vllm_lcb,
    "same_pod_vllm_reference_p95_itl_ms_mean": vllm_p95,
    "ferrum_lcb_vs_vllm_lcb_ratio": current_lcb / vllm_lcb if vllm_lcb else None,
    "ferrum_p95_itl_vs_vllm_p95_itl_ratio": current_p95 / vllm_p95 if vllm_p95 else None,
    "previous_ferrum_output_tps_mean": prev_mean,
    "previous_ferrum_output_tps_lcb": prev_lcb,
    "previous_ferrum_p95_itl_ms_mean": prev_p95,
    "delta_output_tps_lcb": current_lcb - prev_lcb,
    "delta_p95_itl_ms": current_p95 - prev_p95,
    "w2_throughput_pass_at_c16": (current_lcb / vllm_lcb) >= 0.8 if vllm_lcb else False,
    "w2_p95_itl_pass_at_c16": (current_p95 / vllm_p95) <= 1.25 if vllm_p95 else False,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(out / "summary.md").write_text(
    "# W2 active-decode prefill budget c16 diagnostic\n\n"
    "This is diagnostic evidence only and did not produce `MODEL_RELEASE_GRADE_W2 PASS`.\n\n"
    f"- Status: `{summary['status']}`\n"
    f"- Ferrum completed/errors/source: {summary['ferrum_completed_per_run']} / {summary['ferrum_errored_per_run']} / {summary['ferrum_output_token_count_source']}\n"
    f"- Ferrum output TPS mean/LCB: {current_mean:.3f} / {current_lcb:.3f}\n"
    f"- Same-pod vLLM reference TPS mean/LCB: {vllm_mean:.3f} / {vllm_lcb:.3f}\n"
    f"- Ferrum LCB / vLLM LCB: {summary['ferrum_lcb_vs_vllm_lcb_ratio']:.4f}\n"
    f"- Ferrum p95 ITL: {current_p95:.3f} ms\n"
    f"- Same-pod vLLM reference p95 ITL: {vllm_p95:.3f} ms\n"
    f"- Ferrum p95 ITL / vLLM p95 ITL: {summary['ferrum_p95_itl_vs_vllm_p95_itl_ratio']:.4f}\n"
    f"- Delta vs previous Ferrum LCB: {summary['delta_output_tps_lcb']:.3f} tok/s\n"
    f"- Delta vs previous Ferrum p95 ITL: {summary['delta_p95_itl_ms']:.3f} ms\n"
    f"- c16 throughput diagnostic pass: {summary['w2_throughput_pass_at_c16']}\n"
    f"- c16 p95 ITL diagnostic pass: {summary['w2_p95_itl_pass_at_c16']}\n"
)
PY

nvidia-smi > "$OUT/env/nvidia_smi_after.txt"
df -h /workspace / > "$OUT/env/df_after.txt" 2>&1 || true
date -u +%FT%TZ > "$OUT/env/end_utc.txt"
echo "[w2-post-sched-c16] done $(date -u +%FT%TZ)"
