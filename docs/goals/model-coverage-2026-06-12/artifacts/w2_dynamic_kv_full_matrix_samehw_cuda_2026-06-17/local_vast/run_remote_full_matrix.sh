#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/ferrum-infer-rs}"
OUT="${OUT:-/workspace/w2_dynamic_kv_full_matrix_samehw_cuda_2026-06-17}"
MODEL="${MODEL:-gemma3:27b-gptq}"
HF_REPO="${HF_REPO:-circulus/gemma-3-27b-it-gptq}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
DATASET="${DATASET:-/workspace/ascii_sharegpt_w2_100.jsonl}"
VLLM_VERSION="${VLLM_VERSION:-0.23.0}"
VLLM_VENV="${VLLM_VENV:-/workspace/vllm-venv-${VLLM_VERSION//./_}}"
VLLM_PORT="${VLLM_PORT:-18245}"
FERRUM_CORRECTNESS_PORT="${FERRUM_CORRECTNESS_PORT:-18246}"
FERRUM_BENCH_PORT="${FERRUM_BENCH_PORT:-18247}"
SWEEP="${SWEEP:-1,4,16,32}"

mkdir -p "$OUT"/{env,install,prefetch,build,vllm,correctness,server,perf}
exec > >(tee -a "$OUT/run.log") 2>&1

echo "[w2-full-matrix] start $(date -u +%FT%TZ)"
echo "[w2-full-matrix] root=$ROOT out=$OUT model=$MODEL hf_home=$HF_HOME dataset=$DATASET sweep=$SWEEP"
date -u +%FT%TZ > "$OUT/env/start_utc.txt"

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

wait_http() {
  local base="$1"
  local log_file="$2"
  local label="$3"
  for _ in $(seq 1 420); do
    if curl -sf --noproxy '*' --max-time 2 "$base/v1/models" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[w2-full-matrix] $label server exited" >&2
      tail -160 "$log_file" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "[w2-full-matrix] timed out waiting for $label server" >&2
  tail -160 "$log_file" >&2 || true
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

capture_env() {
  nvidia-smi > "$OUT/env/nvidia_smi_before.txt"
  nvcc --version > "$OUT/env/nvcc_version.txt" 2>&1 || true
  df -h /workspace / > "$OUT/env/df_before.txt" 2>&1 || true
  sha256sum "$DATASET" > "$OUT/env/dataset.sha256"
  wc -l "$DATASET" > "$OUT/env/dataset.wc"
}

ensure_rust() {
  if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.cargo/env"
  fi
  if ! command -v cargo >/dev/null 2>&1; then
    echo "[w2-full-matrix] installing rustup minimal toolchain"
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal > "$OUT/env/rustup_install.log" 2>&1
    # shellcheck disable=SC1090
    source "$HOME/.cargo/env"
  fi
  rustc -V > "$OUT/env/rustc_version.txt"
  cargo -V > "$OUT/env/cargo_version.txt"
}

ensure_model() {
  local model_root="$HF_HOME/hub/models--circulus--gemma-3-27b-it-gptq"
  MODEL_PATH="$(find "$model_root/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
  if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
    echo "[w2-full-matrix] prefetching $HF_REPO"
    HF_HOME="$HF_HOME" HF_REPO="$HF_REPO" HF_XET_HIGH_PERFORMANCE=1 "$VLLM_VENV/bin/python" - <<'PY' > "$OUT/prefetch/snapshot_download.log" 2>&1
import os
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=os.environ["HF_REPO"])
print(path)
PY
  fi
  MODEL_PATH="$(find "$model_root/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
  if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
    echo "[w2-full-matrix] missing model snapshot under $model_root/snapshots" >&2
    exit 20
  fi
  printf '%s\n' "$MODEL_PATH" > "$OUT/env/model_path.txt"
  find "$MODEL_PATH" -maxdepth 1 -type f -printf '%f\n' | sort > "$OUT/env/model_files.txt"
  du -sh "$HF_HOME" > "$OUT/env/hf_home_du.txt" 2>&1 || true
}

ensure_vllm() {
  echo "[w2-full-matrix] ensuring vLLM venv"
  if [ -x "$VLLM_VENV/bin/python" ] && "$VLLM_VENV/bin/python" -m pip show vllm 2>/dev/null | grep -q "Version: $VLLM_VERSION"; then
    echo "[w2-full-matrix] reusing existing vLLM venv"
  else
    rm -rf "$VLLM_VENV"
    if ! python3 -m venv "$VLLM_VENV" > "$OUT/install/python_venv_initial.log" 2>&1; then
      echo "[w2-full-matrix] installing python3-venv after venv bootstrap failure"
      apt-get update -qq > "$OUT/install/apt_update_python3_venv.log" 2>&1
      apt-get install -y -qq python3-venv > "$OUT/install/apt_install_python3_venv.log" 2>&1
      rm -rf "$VLLM_VENV"
      python3 -m venv "$VLLM_VENV"
    fi
    "$VLLM_VENV/bin/python" -m pip install --upgrade pip wheel setuptools \
      > "$OUT/install/pip_upgrade.log" 2>&1
    "$VLLM_VENV/bin/python" -m pip install "vllm==$VLLM_VERSION" \
      > "$OUT/install/vllm_install.log" 2>&1
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
}

start_ferrum_build() {
  echo "[w2-full-matrix] starting Ferrum CUDA release build in background"
  (
    set -euo pipefail
    cd "$ROOT"
    if [ -f "$HOME/.cargo/env" ]; then
      # shellcheck disable=SC1090
      source "$HOME/.cargo/env"
    fi
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="$CUDA_HOME/bin:$PATH"
    printf '%q ' cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
      > "$OUT/build/build.command.txt"
    printf '\n' >> "$OUT/build/build.command.txt"
    time cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
    sha256sum target/release/ferrum > "$OUT/env/ferrum.sha256"
  ) > "$OUT/build/cargo_build_release.log" 2>&1 &
  BUILD_PID=$!
}

run_ferrum_run_smoke() {
  echo "[w2-full-matrix] Ferrum run correctness"
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
}

write_summary() {
  local ferrum_rc="$1"
  local vllm_rc="$2"
  python3 - <<'PY' "$OUT" "$ferrum_rc" "$vllm_rc"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
ferrum_rc = int(sys.argv[2])
vllm_rc = int(sys.argv[3])

def read(path):
    p = out / path
    return p.read_text().strip() if p.exists() else None

def load_json(path, default=None):
    p = out / path
    if not p.exists():
        return default
    return json.loads(p.read_text())

def normalize_reports(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("reports"), list):
        return value["reports"]
    if isinstance(value, dict):
        return [value]
    return []

def cell_key(report):
    return int(report.get("concurrency") or 0)

def lcb(report):
    metric = report["output_throughput_tps"]
    return metric["mean"] - metric.get("ci95_hw", 0.0)

def p95_itl(report):
    return report["itl_ms"]["p95"]["mean"]

def clean(report):
    if report.get("output_token_count_source") != "usage":
        return False
    if any(x != 100 for x in report.get("completed_per_run", [])):
        return False
    if any(x != 0 for x in report.get("errored_per_run", [])):
        return False
    for issue in report.get("quality_issues_per_run", []):
        if any(issue.get(k, 0) for k in issue):
            return False
    return True

vllm_reports = {cell_key(r): r for r in normalize_reports(load_json("perf/bench_vllm_sharegpt_sweep_100x3.json"))}
ferrum_reports = {cell_key(r): r for r in normalize_reports(load_json("perf/bench_ferrum_sharegpt_sweep_100x3.json"))}
cells = [1, 4, 16, 32]
cell_rows = []
all_pass = True
for c in cells:
    vr = vllm_reports.get(c)
    fr = ferrum_reports.get(c)
    row = {"concurrency": c, "vllm_present": vr is not None, "ferrum_present": fr is not None}
    if vr is None or fr is None:
        row["status"] = "missing_report"
        all_pass = False
        cell_rows.append(row)
        continue
    v_lcb = lcb(vr)
    f_lcb = lcb(fr)
    v_p95 = p95_itl(vr)
    f_p95 = p95_itl(fr)
    throughput_ratio = f_lcb / v_lcb if v_lcb else None
    p95_ratio = f_p95 / v_p95 if v_p95 else None
    row.update({
        "vllm_output_tps_mean": vr["output_throughput_tps"]["mean"],
        "vllm_output_tps_lcb": v_lcb,
        "ferrum_output_tps_mean": fr["output_throughput_tps"]["mean"],
        "ferrum_output_tps_lcb": f_lcb,
        "throughput_lcb_ratio": throughput_ratio,
        "vllm_p95_itl_ms": v_p95,
        "ferrum_p95_itl_ms": f_p95,
        "p95_itl_ratio": p95_ratio,
        "vllm_completed_per_run": vr.get("completed_per_run"),
        "vllm_errored_per_run": vr.get("errored_per_run"),
        "ferrum_completed_per_run": fr.get("completed_per_run"),
        "ferrum_errored_per_run": fr.get("errored_per_run"),
        "vllm_output_token_count_source": vr.get("output_token_count_source"),
        "ferrum_output_token_count_source": fr.get("output_token_count_source"),
        "vllm_clean": clean(vr),
        "ferrum_clean": clean(fr),
        "throughput_pass": throughput_ratio is not None and throughput_ratio >= 0.8,
        "p95_itl_pass": p95_ratio is not None and p95_ratio <= 1.25,
    })
    row["status"] = "pass" if row["vllm_clean"] and row["ferrum_clean"] and row["throughput_pass"] and row["p95_itl_pass"] else "fail"
    if row["status"] != "pass":
        all_pass = False
    cell_rows.append(row)

status = "diagnostic_pass" if all_pass and ferrum_rc == 0 and vllm_rc == 0 else "diagnostic_fail"
summary = {
    "lane": "w2_dynamic_kv_full_matrix_samehw_cuda",
    "status": status,
    "release_gate": "pending_final_validator",
    "required_validator_pass_line": "MODEL_RELEASE_GRADE_W2 PASS: <out_dir>",
    "remote_git_sha": read("env/git_sha.txt"),
    "remote_git_status_short": read("env/git_status_short.txt"),
    "ferrum_binary_sha256": read("env/ferrum.sha256"),
    "dataset_sha256": read("env/dataset.sha256"),
    "model_path": read("env/model_path.txt"),
    "vllm_versions": load_json("env/vllm_versions.json"),
    "torch_cuda": load_json("env/torch_cuda.json"),
    "vllm_smoke": load_json("vllm/smoke/stream_summary.json"),
    "ferrum_run_smoke": load_json("correctness/run_summary.json"),
    "ferrum_serve_smoke": load_json("correctness/smoke/stream_summary.json"),
    "vllm_bench_exit_code": vllm_rc,
    "ferrum_bench_exit_code": ferrum_rc,
    "concurrency_cells": cell_rows,
    "max_num_seqs": 32,
    "kv_capacity": 512,
    "active_decode_prefill_chunk_check": load_json("server/active_chunk_check.json"),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

lines = [
    "# W2 dynamic-KV same-hardware full matrix pre-gate summary",
    "",
    "This summary is pre-gate evidence. Final release status is determined by ",
    "`model_release_grade_goal_gate.manifest.json` when present.",
    "",
    f"- Status: `{status}`",
    f"- Remote git SHA: `{summary['remote_git_sha']}`",
    f"- Remote git status: `{summary['remote_git_status_short'] or 'clean'}`",
    f"- vLLM bench exit code: `{vllm_rc}`",
    f"- Ferrum bench exit code: `{ferrum_rc}`",
    f"- Server max-num-seqs: `32`",
    f"- Server kv-capacity: `512`",
    "",
    "| c | Ferrum LCB | vLLM LCB | ratio | Ferrum p95 ITL | vLLM p95 ITL | p95 ratio | status |",
    "|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in cell_rows:
    if row.get("status") == "missing_report":
        lines.append(f"| {row['concurrency']} | n/a | n/a | n/a | n/a | n/a | n/a | missing |")
    else:
        lines.append(
            f"| {row['concurrency']} | {row['ferrum_output_tps_lcb']:.3f} | "
            f"{row['vllm_output_tps_lcb']:.3f} | {row['throughput_lcb_ratio']:.4f} | "
            f"{row['ferrum_p95_itl_ms']:.3f} | {row['vllm_p95_itl_ms']:.3f} | "
            f"{row['p95_itl_ratio']:.4f} | {row['status']} |"
        )
(out / "summary.md").write_text("\n".join(lines) + "\n")
PY
}

capture_env
cd "$ROOT"
git rev-parse HEAD > "$OUT/env/git_sha.txt"
git status --short > "$OUT/env/git_status_short.txt"

ensure_rust
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

start_ferrum_build
ensure_vllm
ensure_model

echo "[w2-full-matrix] waiting for Ferrum build"
wait "$BUILD_PID"
BUILD_PID=""
BIN="$ROOT/target/release/ferrum"
test -x "$BIN"

echo "[w2-full-matrix] vLLM server + smoke + sweep"
SERVER_CMD=(
  "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server
  --host 127.0.0.1
  --port "$VLLM_PORT"
  --model "$MODEL_PATH"
  --served-model-name "$MODEL"
  --max-model-len 512
  --max-num-seqs 32
  --gpu-memory-utilization 0.92
)
python3 - "$OUT/vllm/vllm_server.command.json" "${SERVER_CMD[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "cmd": sys.argv[2:],
    "dependency_pins": {
        "vllm": os.environ.get("VLLM_VERSION", "unknown"),
        "dependency_policy": "vLLM release resolver; no manual transformers/api downgrades",
    },
}, indent=2, sort_keys=True) + "\n")
PY
HF_HOME="$HF_HOME" "${SERVER_CMD[@]}" > "$OUT/vllm/vllm_server.log" 2>&1 &
SERVER_PID=$!
VLLM_BASE="http://127.0.0.1:$VLLM_PORT"
wait_http "$VLLM_BASE" "$OUT/vllm/vllm_server.log" "vLLM"
curl -sf --noproxy '*' "$VLLM_BASE/v1/models" > "$OUT/vllm/models.json"
run_stream_smoke "$VLLM_BASE" "$MODEL" "$OUT/vllm/smoke"

VLLM_BENCH="$OUT/perf/bench_vllm_sharegpt_sweep_100x3.json"
VLLM_BENCH_CMD=(
  timeout 14400 "$BIN" bench-serve
  --base-url "$VLLM_BASE"
  --model "$MODEL"
  --tokenizer "$MODEL_PATH"
  --dataset sharegpt
  --sharegpt-path "$DATASET"
  --random-output-len 128
  --concurrency-sweep "$SWEEP"
  --num-prompts 100
  --n-repeats 3
  --fail-on-error
  --require-ci
  --seed 9271
  --out "$VLLM_BENCH"
)
printf '%q ' "${VLLM_BENCH_CMD[@]}" > "$OUT/perf/bench-vllm.command.txt"
printf '\n' >> "$OUT/perf/bench-vllm.command.txt"
set +e
"${VLLM_BENCH_CMD[@]}" > "$OUT/perf/bench-vllm.stdout" 2> "$OUT/perf/bench-vllm.stderr"
VLLM_RC=$?
set -e
cleanup_server
if [ "$VLLM_RC" -ne 0 ]; then
  echo "[w2-full-matrix] vLLM baseline failed with rc=$VLLM_RC"
  write_summary 99 "$VLLM_RC" || true
  exit "$VLLM_RC"
fi

run_ferrum_run_smoke

echo "[w2-full-matrix] Ferrum serve correctness"
HF_HOME="$HF_HOME" "$BIN" serve "$MODEL" \
  --backend cuda \
  --port "$FERRUM_CORRECTNESS_PORT" \
  --kv-capacity 512 \
  --max-num-seqs 32 \
  --effective-config-json "$OUT/correctness/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/correctness/serve_decision_trace.jsonl" \
  > "$OUT/correctness/serve.log" 2>&1 &
SERVER_PID=$!
FERRUM_CORRECTNESS_BASE="http://127.0.0.1:$FERRUM_CORRECTNESS_PORT"
wait_http "$FERRUM_CORRECTNESS_BASE" "$OUT/correctness/serve.log" "Ferrum correctness"
curl -sf --noproxy '*' "$FERRUM_CORRECTNESS_BASE/v1/models" > "$OUT/correctness/models.json"
run_stream_smoke "$FERRUM_CORRECTNESS_BASE" "$MODEL" "$OUT/correctness/smoke"
cleanup_server

echo "[w2-full-matrix] Ferrum serve + sweep"
HF_HOME="$HF_HOME" "$BIN" serve "$MODEL" \
  --backend cuda \
  --port "$FERRUM_BENCH_PORT" \
  --kv-capacity 512 \
  --max-num-seqs 32 \
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

FERRUM_BENCH="$OUT/perf/bench_ferrum_sharegpt_sweep_100x3.json"
FERRUM_BENCH_CMD=(
  timeout 14400 "$BIN" bench-serve
  --base-url "$FERRUM_BENCH_BASE"
  --model "$MODEL"
  --tokenizer "$MODEL_PATH"
  --dataset sharegpt
  --sharegpt-path "$DATASET"
  --random-output-len 128
  --concurrency-sweep "$SWEEP"
  --num-prompts 100
  --n-repeats 3
  --fail-on-error
  --require-ci
  --seed 9271
  --out "$FERRUM_BENCH"
)
printf '%q ' "${FERRUM_BENCH_CMD[@]}" > "$OUT/perf/bench-ferrum.command.txt"
printf '\n' >> "$OUT/perf/bench-ferrum.command.txt"
set +e
"${FERRUM_BENCH_CMD[@]}" > "$OUT/perf/bench-ferrum.stdout" 2> "$OUT/perf/bench-ferrum.stderr"
FERRUM_RC=$?
set -e
cleanup_server

nvidia-smi > "$OUT/perf/nvidia_smi_after_bench.txt"
write_summary "$FERRUM_RC" "$VLLM_RC"
nvidia-smi > "$OUT/env/nvidia_smi_after.txt"
df -h /workspace / > "$OUT/env/df_after.txt" 2>&1 || true
date -u +%FT%TZ > "$OUT/env/end_utc.txt"

if [ "$FERRUM_RC" -ne 0 ]; then
  echo "[w2-full-matrix] Ferrum bench failed with rc=$FERRUM_RC"
  exit "$FERRUM_RC"
fi

echo "[w2-full-matrix] generating W2 release-grade manifest"
set +e
python3 "$ROOT/scripts/release/model_release_grade_manifest.py" w2 \
  --source "$OUT" \
  --out "$OUT" \
  2>&1 | tee "$OUT/model_release_grade_manifest.log"
GATE_RC=${PIPESTATUS[0]}
set -e
if [ "$GATE_RC" -ne 0 ]; then
  echo "[w2-full-matrix] release-grade manifest/gate failed with rc=$GATE_RC"
  exit "$GATE_RC"
fi

echo "[w2-full-matrix] done $(date -u +%FT%TZ)"
