#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/ferrum-infer-rs}"
OUT="${OUT:-/workspace/w2_vllm_ferrum_same_hw_c16_ab_2026-06-17}"
MODEL="${MODEL:-gemma3:27b-gptq}"
HF_REPO="${HF_REPO:-circulus/gemma-3-27b-it-gptq}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
DATASET="${DATASET:-/workspace/ascii_sharegpt_w2_100.jsonl}"
VLLM_VENV="${VLLM_VENV:-/workspace/vllm-venv-0101-cu126}"
VLLM_PORT="${VLLM_PORT:-18145}"
FERRUM_CORRECTNESS_PORT="${FERRUM_CORRECTNESS_PORT:-18146}"
FERRUM_BENCH_PORT="${FERRUM_BENCH_PORT:-18147}"

mkdir -p "$OUT"/{env,install,prefetch,build,vllm,correctness,server,perf}
exec > >(tee -a "$OUT/run.log") 2>&1

echo "[w2-c16-ab] start $(date -u +%FT%TZ)"
echo "[w2-c16-ab] root=$ROOT out=$OUT model=$MODEL hf_home=$HF_HOME dataset=$DATASET"

SERVER_PID=""
BUILD_PID=""
cleanup_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  if [ -n "${BUILD_PID:-}" ] && kill -0 "$BUILD_PID" 2>/dev/null; then
    kill "$BUILD_PID" 2>/dev/null || true
    wait "$BUILD_PID" 2>/dev/null || true
    BUILD_PID=""
  fi
}
trap cleanup_server EXIT

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

wait_http() {
  local base="$1"
  local log_file="$2"
  local label="$3"
  for _ in $(seq 1 360); do
    if curl -sf --noproxy '*' --max-time 2 "$base/v1/models" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[w2-c16-ab] $label server exited" >&2
      tail -120 "$log_file" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "[w2-c16-ab] timed out waiting for $label server" >&2
  tail -120 "$log_file" >&2 || true
  return 1
}

nvidia-smi > "$OUT/env/nvidia_smi_before.txt"
nvcc --version > "$OUT/env/nvcc_version.txt" 2>&1 || true
df -h /workspace / > "$OUT/env/df_before.txt" 2>&1 || true
sha256sum "$DATASET" > "$OUT/env/dataset.sha256"
wc -l "$DATASET" > "$OUT/env/dataset.wc"

cd "$ROOT"
git rev-parse HEAD > "$OUT/env/git_sha.txt"
git status --short > "$OUT/env/git_status_short.txt"

if ! command -v cargo >/dev/null 2>&1; then
  echo "[w2-c16-ab] installing rustup minimal toolchain"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal > "$OUT/env/rustup_install.log" 2>&1
fi
# shellcheck disable=SC1090
source "$HOME/.cargo/env"
rustc -V > "$OUT/env/rustc_version.txt"
cargo -V > "$OUT/env/cargo_version.txt"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

echo "[w2-c16-ab] starting Ferrum CUDA release build in background"
(
  set -euo pipefail
  cd "$ROOT"
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
  printf '%q ' cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
    > "$OUT/build/build.command.txt"
  printf '\n' >> "$OUT/build/build.command.txt"
  time cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
  sha256sum target/release/ferrum > "$OUT/env/ferrum.sha256"
) > "$OUT/build/cargo_build_release.log" 2>&1 &
BUILD_PID=$!

echo "[w2-c16-ab] ensuring vLLM venv"
if [ ! -x "$VLLM_VENV/bin/python" ]; then
  rm -rf "$VLLM_VENV"
  python3 -m venv "$VLLM_VENV"
  "$VLLM_VENV/bin/python" -m pip install --upgrade pip wheel setuptools \
    > "$OUT/install/pip_upgrade.log" 2>&1
  "$VLLM_VENV/bin/python" -m pip install vllm==0.10.1.1 \
    > "$OUT/install/vllm0101_install.log" 2>&1
else
  echo "[w2-c16-ab] reusing existing vLLM venv"
fi
"$VLLM_VENV/bin/python" -m pip install transformers==4.55.4 \
  > "$OUT/install/pin_transformers_4554.log" 2>&1
"$VLLM_VENV/bin/python" -m pip install fastapi==0.116.1 starlette==0.47.2 prometheus-fastapi-instrumentator==7.1.0 \
  > "$OUT/install/pin_api_stack.log" 2>&1
"$VLLM_VENV/bin/python" -m pip freeze > "$OUT/env/vllm_pip_freeze.txt"
"$VLLM_VENV/bin/python" - <<'PY' > "$OUT/env/vllm_versions.json"
import importlib.metadata as md
import json
mods = ["vllm", "torch", "transformers", "fastapi", "starlette", "prometheus-fastapi-instrumentator"]
print(json.dumps({m: md.version(m) for m in mods}, indent=2, sort_keys=True))
PY
"$VLLM_VENV/bin/python" - <<'PY' > "$OUT/env/torch_cuda.json"
import json
import torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2, sort_keys=True))
PY

MODEL_ROOT="$HF_HOME/hub/models--circulus--gemma-3-27b-it-gptq"
MODEL_PATH="$(find "$MODEL_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
  echo "[w2-c16-ab] prefetching $HF_REPO"
  HF_HOME="$HF_HOME" HF_XET_HIGH_PERFORMANCE=1 "$VLLM_VENV/bin/python" - <<'PY' > "$OUT/prefetch/snapshot_download.log" 2>&1
import os
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=os.environ.get("HF_REPO", "circulus/gemma-3-27b-it-gptq"))
print(path)
PY
fi
MODEL_PATH="$(find "$MODEL_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
  echo "[w2-c16-ab] missing model snapshot under $MODEL_ROOT/snapshots" >&2
  exit 20
fi
printf '%s\n' "$MODEL_PATH" > "$OUT/env/model_path.txt"
find "$MODEL_PATH" -maxdepth 1 -type f -printf '%f\n' | sort > "$OUT/env/model_files.txt"
du -sh "$HF_HOME" > "$OUT/env/hf_home_du.txt" 2>&1 || true

echo "[w2-c16-ab] waiting for Ferrum build"
wait "$BUILD_PID"
BUILD_PID=""
BIN="$ROOT/target/release/ferrum"
test -x "$BIN"

echo "[w2-c16-ab] vLLM server + smoke + c16 bench"
SERVER_CMD=(
  "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server
  --host 127.0.0.1
  --port "$VLLM_PORT"
  --model "$MODEL_PATH"
  --served-model-name "$MODEL"
  --max-model-len 512
  --max-num-seqs 16
  --gpu-memory-utilization 0.92
)
python3 - "$OUT/vllm/vllm_server.command.json" "${SERVER_CMD[@]}" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "cmd": sys.argv[2:],
    "dependency_pins": {
        "vllm": "0.10.1.1",
        "transformers": "4.55.4",
        "fastapi": "0.116.1",
        "starlette": "0.47.2",
        "prometheus-fastapi-instrumentator": "7.1.0",
    },
}, indent=2, sort_keys=True) + "\n")
PY
HF_HOME="$HF_HOME" "${SERVER_CMD[@]}" > "$OUT/vllm/vllm_server.log" 2>&1 &
SERVER_PID=$!
VLLM_BASE="http://127.0.0.1:$VLLM_PORT"
wait_http "$VLLM_BASE" "$OUT/vllm/vllm_server.log" "vLLM"
curl -sf --noproxy '*' "$VLLM_BASE/v1/models" > "$OUT/vllm/models.json"
run_stream_smoke "$VLLM_BASE" "$MODEL" "$OUT/vllm/smoke"
VLLM_BENCH="$OUT/perf/bench_vllm_sharegpt_c16_100x3.json"
VLLM_BENCH_CMD=(
  timeout 10800 "$BIN" bench-serve
  --base-url "$VLLM_BASE"
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
  --out "$VLLM_BENCH"
)
printf '%q ' "${VLLM_BENCH_CMD[@]}" > "$OUT/perf/bench-vllm.command.txt"
printf '\n' >> "$OUT/perf/bench-vllm.command.txt"
"${VLLM_BENCH_CMD[@]}" > "$OUT/perf/bench-vllm.stdout" 2> "$OUT/perf/bench-vllm.stderr"
cleanup_server

echo "[w2-c16-ab] Ferrum run correctness"
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

echo "[w2-c16-ab] Ferrum serve correctness"
SERVER_PID=""
HF_HOME="$HF_HOME" "$BIN" serve "$MODEL" \
  --backend cuda \
  --port "$FERRUM_CORRECTNESS_PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/correctness/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/correctness/serve_decision_trace.jsonl" \
  > "$OUT/correctness/serve.log" 2>&1 &
SERVER_PID=$!
FERRUM_CORRECTNESS_BASE="http://127.0.0.1:$FERRUM_CORRECTNESS_PORT"
wait_http "$FERRUM_CORRECTNESS_BASE" "$OUT/correctness/serve.log" "Ferrum correctness"
curl -sf --noproxy '*' "$FERRUM_CORRECTNESS_BASE/v1/models" > "$OUT/correctness/models.json"
run_stream_smoke "$FERRUM_CORRECTNESS_BASE" "$MODEL" "$OUT/correctness/smoke"
cleanup_server

echo "[w2-c16-ab] Ferrum serve + c16 bench"
SERVER_PID=""
HF_HOME="$HF_HOME" "$BIN" serve "$MODEL" \
  --backend cuda \
  --port "$FERRUM_BENCH_PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/server/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl" \
  > "$OUT/server/server.log" 2>&1 &
SERVER_PID=$!
FERRUM_BENCH_BASE="http://127.0.0.1:$FERRUM_BENCH_PORT"
wait_http "$FERRUM_BENCH_BASE" "$OUT/server/server.log" "Ferrum bench"
curl -sf --noproxy '*' "$FERRUM_BENCH_BASE/v1/models" > "$OUT/server/models.json"
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
FERRUM_BENCH="$OUT/perf/bench_ferrum_sharegpt_c16_100x3.json"
FERRUM_BENCH_CMD=(
  timeout 10800 "$BIN" bench-serve
  --base-url "$FERRUM_BENCH_BASE"
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
  --out "$FERRUM_BENCH"
)
printf '%q ' "${FERRUM_BENCH_CMD[@]}" > "$OUT/perf/bench-ferrum.command.txt"
printf '\n' >> "$OUT/perf/bench-ferrum.command.txt"
"${FERRUM_BENCH_CMD[@]}" > "$OUT/perf/bench-ferrum.stdout" 2> "$OUT/perf/bench-ferrum.stderr"
cleanup_server
nvidia-smi > "$OUT/perf/nvidia_smi_after_bench.txt"

python3 - <<'PY' "$OUT" "$VLLM_BENCH" "$FERRUM_BENCH"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
vllm = json.loads(Path(sys.argv[2]).read_text())
ferrum = json.loads(Path(sys.argv[3]).read_text())

def read(path):
    p = out / path
    return p.read_text().strip() if p.exists() else None

def lcb(bench):
    metric = bench["output_throughput_tps"]
    return metric["mean"] - metric.get("ci95_hw", 0.0)

def p95_itl(bench):
    return bench["itl_ms"]["p95"]["mean"]

def clean(bench):
    ok = True
    if any(x != 100 for x in bench.get("completed_per_run", [])):
        ok = False
    if any(x != 0 for x in bench.get("errored_per_run", [])):
        ok = False
    for issue in bench.get("quality_issues_per_run", []):
        if any(issue.get(k, 0) for k in issue):
            ok = False
    if bench.get("output_token_count_source") != "usage":
        ok = False
    return ok

vllm_lcb = lcb(vllm)
ferrum_lcb = lcb(ferrum)
vllm_p95 = p95_itl(vllm)
ferrum_p95 = p95_itl(ferrum)
summary = {
    "lane": "w2_vllm_ferrum_same_hw_c16_sharegpt_ab",
    "status": "diagnostic_pass",
    "remote_git_sha": read("env/git_sha.txt"),
    "remote_git_status_short": read("env/git_status_short.txt"),
    "ferrum_binary_sha256": read("env/ferrum.sha256"),
    "dataset_sha256": read("env/dataset.sha256"),
    "model_path": read("env/model_path.txt"),
    "vllm_versions": json.loads((out / "env/vllm_versions.json").read_text()),
    "torch_cuda": json.loads((out / "env/torch_cuda.json").read_text()),
    "vllm_smoke": json.loads((out / "vllm/smoke/stream_summary.json").read_text()),
    "ferrum_smoke": json.loads((out / "correctness/smoke/stream_summary.json").read_text()),
    "vllm_completed_per_run": vllm.get("completed_per_run"),
    "vllm_errored_per_run": vllm.get("errored_per_run"),
    "vllm_quality_issues_per_run": vllm.get("quality_issues_per_run"),
    "vllm_output_token_count_source": vllm.get("output_token_count_source"),
    "ferrum_completed_per_run": ferrum.get("completed_per_run"),
    "ferrum_errored_per_run": ferrum.get("errored_per_run"),
    "ferrum_quality_issues_per_run": ferrum.get("quality_issues_per_run"),
    "ferrum_output_token_count_source": ferrum.get("output_token_count_source"),
    "vllm_output_tps_mean": vllm["output_throughput_tps"]["mean"],
    "vllm_output_tps_lcb": vllm_lcb,
    "ferrum_output_tps_mean": ferrum["output_throughput_tps"]["mean"],
    "ferrum_output_tps_lcb": ferrum_lcb,
    "ferrum_lcb_vs_vllm_lcb_ratio": ferrum_lcb / vllm_lcb if vllm_lcb else None,
    "vllm_p95_itl_ms_mean": vllm_p95,
    "ferrum_p95_itl_ms_mean": ferrum_p95,
    "ferrum_p95_itl_vs_vllm_p95_itl_ratio": ferrum_p95 / vllm_p95 if vllm_p95 else None,
    "w2_throughput_pass_at_c16": (ferrum_lcb / vllm_lcb) >= 0.8 if vllm_lcb else False,
    "w2_p95_itl_pass_at_c16": (ferrum_p95 / vllm_p95) <= 1.25 if vllm_p95 else False,
}
if not clean(vllm):
    summary["status"] = "diagnostic_fail_vllm_quality"
if not clean(ferrum):
    summary["status"] = "diagnostic_fail_ferrum_quality"

(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(out / "summary.md").write_text(
    "# W2 same-pod c16 vLLM/Ferrum ShareGPT diagnostic\n\n"
    f"- Status: {summary['status']}\n"
    f"- vLLM output TPS mean/LCB: {summary['vllm_output_tps_mean']:.3f} / {summary['vllm_output_tps_lcb']:.3f}\n"
    f"- Ferrum output TPS mean/LCB: {summary['ferrum_output_tps_mean']:.3f} / {summary['ferrum_output_tps_lcb']:.3f}\n"
    f"- Ferrum LCB / vLLM LCB: {summary['ferrum_lcb_vs_vllm_lcb_ratio']:.4f}\n"
    f"- vLLM p95 ITL: {summary['vllm_p95_itl_ms_mean']:.3f} ms\n"
    f"- Ferrum p95 ITL: {summary['ferrum_p95_itl_ms_mean']:.3f} ms\n"
    f"- Ferrum p95 ITL / vLLM p95 ITL: {summary['ferrum_p95_itl_vs_vllm_p95_itl_ratio']:.4f}\n"
    f"- c16 throughput gate diagnostic: {summary['w2_throughput_pass_at_c16']}\n"
    f"- c16 p95 ITL gate diagnostic: {summary['w2_p95_itl_pass_at_c16']}\n"
    f"- vLLM completed/errors/source: {summary['vllm_completed_per_run']} / {summary['vllm_errored_per_run']} / {summary['vllm_output_token_count_source']}\n"
    f"- Ferrum completed/errors/source: {summary['ferrum_completed_per_run']} / {summary['ferrum_errored_per_run']} / {summary['ferrum_output_token_count_source']}\n"
)
PY

nvidia-smi > "$OUT/env/nvidia_smi_after.txt"
df -h /workspace / > "$OUT/env/df_after.txt" 2>&1 || true
date -u +%FT%TZ > "$OUT/env/end_utc.txt"
echo "[w2-c16-ab] done $(date -u +%FT%TZ)"
