#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/ferrum-infer-rs}"
OUT="${OUT:-/workspace/w2_vllm_same_hw_c16_sharegpt_2026-06-17}"
MODEL="${MODEL:-gemma3:27b-gptq}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
DATASET="${DATASET:-/workspace/ascii_sharegpt_w2_100.jsonl}"
VLLM_VENV="${VLLM_VENV:-/workspace/vllm-venv-0101-cu126}"
PORT="${PORT:-18145}"

mkdir -p "$OUT"/{env,install,server,correctness,perf}
exec > >(tee -a "$OUT/run.log") 2>&1

echo "[w2-vllm-c16] start $(date -u +%FT%TZ)"
echo "[w2-vllm-c16] root=$ROOT out=$OUT model=$MODEL hf_home=$HF_HOME dataset=$DATASET venv=$VLLM_VENV port=$PORT"

SERVER_PID=""
cleanup_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap cleanup_server EXIT

nvidia-smi > "$OUT/env/nvidia_smi_before.txt"
nvcc --version > "$OUT/env/nvcc_version.txt" 2>&1 || true
df -h /workspace / > "$OUT/env/df_before.txt" 2>&1 || true
sha256sum "$DATASET" > "$OUT/env/dataset.sha256"
wc -l "$DATASET" > "$OUT/env/dataset.wc"

cd "$ROOT"
git rev-parse HEAD > "$OUT/env/git_sha.txt"
git status --short > "$OUT/env/git_status_short.txt"

if [ ! -x target/release/ferrum ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "[w2-vllm-c16] installing rustup minimal toolchain"
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal > "$OUT/env/rustup_install.log" 2>&1
  fi
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
  echo "[w2-vllm-c16] building ferrum bench client"
  printf '%q ' cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
    > "$OUT/env/ferrum_build.command.txt"
  printf '\n' >> "$OUT/env/ferrum_build.command.txt"
  time cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
    > "$OUT/env/ferrum_build.log" 2>&1
fi

if command -v rustc >/dev/null 2>&1; then rustc -V > "$OUT/env/rustc_version.txt"; fi
if command -v cargo >/dev/null 2>&1; then cargo -V > "$OUT/env/cargo_version.txt"; fi
sha256sum target/release/ferrum > "$OUT/env/ferrum.sha256"

MODEL_ROOT="$HF_HOME/hub/models--circulus--gemma-3-27b-it-gptq"
MODEL_PATH="$(find "$MODEL_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
  echo "[w2-vllm-c16] missing model snapshot under $MODEL_ROOT/snapshots" >&2
  exit 20
fi
printf '%s\n' "$MODEL_PATH" > "$OUT/env/model_path.txt"
find "$MODEL_PATH" -maxdepth 1 -type f -printf '%f\n' | sort > "$OUT/env/model_files.txt"

if [ ! -x "$VLLM_VENV/bin/python" ]; then
  echo "[w2-vllm-c16] installing vLLM 0.10.1.1 into $VLLM_VENV"
  rm -rf "$VLLM_VENV"
  python3 -m venv "$VLLM_VENV"
  "$VLLM_VENV/bin/python" -m pip install --upgrade pip wheel setuptools \
    > "$OUT/install/pip_upgrade.log" 2>&1
  "$VLLM_VENV/bin/python" -m pip install vllm==0.10.1.1 \
    > "$OUT/install/vllm0101_install.log" 2>&1
  "$VLLM_VENV/bin/python" -m pip install transformers==4.55.4 \
    > "$OUT/install/pin_transformers_4554.log" 2>&1
  "$VLLM_VENV/bin/python" -m pip install fastapi==0.116.1 starlette==0.47.2 prometheus-fastapi-instrumentator==7.1.0 \
    > "$OUT/install/pin_api_stack.log" 2>&1
else
  echo "[w2-vllm-c16] reusing existing vLLM venv"
fi

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

SERVER_CMD=(
  "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server
  --host 127.0.0.1
  --port "$PORT"
  --model "$MODEL_PATH"
  --served-model-name "$MODEL"
  --max-model-len 512
  --max-num-seqs 16
  --gpu-memory-utilization 0.92
)
python3 - "$OUT/server/vllm_server.command.json" "${SERVER_CMD[@]}" <<'PY'
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
}, indent=2) + "\n")
PY

echo "[w2-vllm-c16] starting vLLM server"
HF_HOME="$HF_HOME" "${SERVER_CMD[@]}" > "$OUT/server/vllm_server.log" 2>&1 &
SERVER_PID=$!
BASE="http://127.0.0.1:$PORT"
for _ in $(seq 1 300); do
  if curl -sf --noproxy '*' --max-time 2 "$BASE/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[w2-vllm-c16] server exited" >&2
    tail -120 "$OUT/server/vllm_server.log" >&2 || true
    exit 21
  fi
  sleep 2
done
curl -sf --noproxy '*' "$BASE/v1/models" > "$OUT/correctness/models.json"

echo "[w2-vllm-c16] streaming smoke"
python3 - <<'PY' "$BASE" "$MODEL" "$OUT/correctness"
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

echo "[w2-vllm-c16] running ShareGPT c16 baseline"
BENCH_JSON="$OUT/perf/bench_vllm_sharegpt_c16_100x3.json"
BENCH_CMD=(
  timeout 10800 target/release/ferrum bench-serve
  --base-url "$BASE"
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
printf '%q ' "${BENCH_CMD[@]}" > "$OUT/perf/bench-serve.command.txt"
printf '\n' >> "$OUT/perf/bench-serve.command.txt"
"${BENCH_CMD[@]}" > "$OUT/perf/bench-serve.stdout" 2> "$OUT/perf/bench-serve.stderr"
nvidia-smi > "$OUT/perf/nvidia_smi_after_bench.txt"

python3 - <<'PY' "$OUT" "$BENCH_JSON"
import json
import math
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
bench_path = Path(sys.argv[2])
bench = json.loads(bench_path.read_text())

def read(path):
    p = out / path
    return p.read_text().strip() if p.exists() else None

output = bench["output_throughput_tps"]
itl = bench["itl_ms"]
ferrum_lcb = 397.9845379429117
ferrum_p95_itl = 57.781574733333336
baseline_lcb = output["mean"] - output.get("ci95_hw", 0.0)
baseline_p95 = itl["p95"]["mean"]
summary = {
    "lane": "w2_vllm_same_hw_c16_sharegpt",
    "status": "diagnostic_pass",
    "engine": "vllm",
    "model": bench.get("model"),
    "remote_git_sha": read("env/git_sha.txt"),
    "remote_git_status_short": read("env/git_status_short.txt"),
    "ferrum_binary_sha256": read("env/ferrum.sha256"),
    "dataset_sha256": read("env/dataset.sha256"),
    "model_path": read("env/model_path.txt"),
    "vllm_versions": json.loads((out / "env/vllm_versions.json").read_text()),
    "torch_cuda": json.loads((out / "env/torch_cuda.json").read_text()),
    "server_command": json.loads((out / "server/vllm_server.command.json").read_text()),
    "smoke": json.loads((out / "correctness/stream_summary.json").read_text()),
    "completed_per_run": bench.get("completed_per_run"),
    "errored_per_run": bench.get("errored_per_run"),
    "quality_issues_per_run": bench.get("quality_issues_per_run"),
    "output_token_count_source": bench.get("output_token_count_source"),
    "vllm_output_tps_mean": output["mean"],
    "vllm_output_tps_ci95_hw": output.get("ci95_hw"),
    "vllm_output_tps_lcb": baseline_lcb,
    "vllm_p95_itl_ms_mean": baseline_p95,
    "vllm_p95_itl_ms_ci95_hw": itl["p95"].get("ci95_hw"),
    "ferrum_same_iteration_c16_lcb": ferrum_lcb,
    "ferrum_same_iteration_c16_p95_itl_ms": ferrum_p95_itl,
    "ferrum_lcb_vs_vllm_lcb_ratio": ferrum_lcb / baseline_lcb if baseline_lcb else None,
    "ferrum_p95_itl_vs_vllm_p95_itl_ratio": ferrum_p95_itl / baseline_p95 if baseline_p95 else None,
    "w2_throughput_threshold_lcb_80pct": baseline_lcb * 0.8,
    "w2_p95_itl_threshold_125pct": baseline_p95 * 1.25,
}
if any(x != 100 for x in bench.get("completed_per_run", [])):
    summary["status"] = "diagnostic_fail_incomplete_requests"
if any(x != 0 for x in bench.get("errored_per_run", [])):
    summary["status"] = "diagnostic_fail_errors"
for issue in bench.get("quality_issues_per_run", []):
    if any(issue.get(k, 0) for k in issue):
        summary["status"] = "diagnostic_fail_quality"
if bench.get("output_token_count_source") != "usage":
    summary["status"] = "diagnostic_fail_non_usage_tokens"

(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(out / "summary.md").write_text(
    "# W2 vLLM same-hardware c16 ShareGPT diagnostic\n\n"
    f"- Status: {summary['status']}\n"
    f"- vLLM output TPS mean/LCB: {summary['vllm_output_tps_mean']:.3f} / {summary['vllm_output_tps_lcb']:.3f}\n"
    f"- vLLM p95 ITL mean: {summary['vllm_p95_itl_ms_mean']:.3f} ms\n"
    f"- Ferrum c16 LCB / vLLM c16 LCB: {summary['ferrum_lcb_vs_vllm_lcb_ratio']:.4f}\n"
    f"- Ferrum p95 ITL / vLLM p95 ITL: {summary['ferrum_p95_itl_vs_vllm_p95_itl_ratio']:.4f}\n"
    f"- Completed per run: {summary['completed_per_run']}\n"
    f"- Errored per run: {summary['errored_per_run']}\n"
    f"- Output token count source: {summary['output_token_count_source']}\n"
)
PY

cleanup_server
nvidia-smi > "$OUT/env/nvidia_smi_after.txt"
date -u +%FT%TZ > "$OUT/env/end_utc.txt"
echo "[w2-vllm-c16] done $(date -u +%FT%TZ)"
