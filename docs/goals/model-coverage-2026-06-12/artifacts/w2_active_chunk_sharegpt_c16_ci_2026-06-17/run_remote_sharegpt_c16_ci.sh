#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/ferrum-infer-rs}"
OUT="${OUT:-/workspace/w2_active_chunk_sharegpt_c16_ci_2026-06-17}"
MODEL="${MODEL:-gemma3:27b-gptq}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
DATASET="${DATASET:-/workspace/ascii_sharegpt_w2_100.jsonl}"
PORT_CORRECTNESS="${PORT_CORRECTNESS:-18134}"
PORT_BENCH="${PORT_BENCH:-18135}"

mkdir -p "$OUT"/{env,build,correctness,server,perf}
exec > >(tee -a "$OUT/run.log") 2>&1

echo "[w2-c16] start $(date -u +%FT%TZ)"
echo "[w2-c16] root=$ROOT out=$OUT model=$MODEL hf_home=$HF_HOME dataset=$DATASET"

cleanup_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap cleanup_server EXIT

nvidia-smi > "$OUT/env/nvidia_smi_before.txt"
nvcc --version > "$OUT/env/nvcc_version.txt"
df -h /workspace / > "$OUT/env/df_before.txt" 2>&1 || true
sha256sum "$DATASET" > "$OUT/env/dataset.sha256"
wc -l "$DATASET" > "$OUT/env/dataset.wc"

cd "$ROOT"
git rev-parse HEAD > "$OUT/env/git_sha.txt"
git status --short > "$OUT/env/git_status_short.txt"

if ! command -v cargo >/dev/null 2>&1; then
  echo "[w2-c16] installing rustup minimal toolchain"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal > "$OUT/env/rustup_install.log" 2>&1
fi
# shellcheck disable=SC1090
source "$HOME/.cargo/env"
rustc -V > "$OUT/env/rustc_version.txt"
cargo -V > "$OUT/env/cargo_version.txt"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

echo "[w2-c16] CUDA release build"
printf '%q ' cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$OUT/build/build.command.txt"
printf '\n' >> "$OUT/build/build.command.txt"
time cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$OUT/build/cargo_build_release.log" 2>&1
sha256sum target/release/ferrum > "$OUT/env/ferrum.sha256"

MODEL_ROOT="$HF_HOME/hub/models--circulus--gemma-3-27b-it-gptq"
MODEL_PATH="$(find "$MODEL_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
  echo "[w2-c16] missing model snapshot under $MODEL_ROOT/snapshots" >&2
  exit 20
fi
printf '%s\n' "$MODEL_PATH" > "$OUT/env/model_path.txt"
find "$MODEL_PATH" -maxdepth 1 -type f -printf '%f\n' | sort > "$OUT/env/model_files.txt"

BIN="$ROOT/target/release/ferrum"

echo "[w2-c16] correctness: ferrum run"
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

echo "[w2-c16] correctness: ferrum serve streaming smoke"
SERVER_PID=""
HF_HOME="$HF_HOME" "$BIN" serve "$MODEL" \
  --backend cuda \
  --port "$PORT_CORRECTNESS" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/correctness/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/correctness/serve_decision_trace.jsonl" \
  > "$OUT/correctness/serve.log" 2>&1 &
SERVER_PID=$!
BASE="http://127.0.0.1:$PORT_CORRECTNESS"
for _ in $(seq 1 300); do
  if curl -sf --noproxy '*' --max-time 2 "$BASE/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[w2-c16] correctness server exited" >&2
    tail -80 "$OUT/correctness/serve.log" >&2 || true
    exit 21
  fi
  sleep 2
done
curl -sf --noproxy '*' "$BASE/v1/models" > "$OUT/correctness/models.json"
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
cleanup_server

echo "[w2-c16] performance: ferrum serve + ShareGPT c16 CI"
SERVER_PID=""
HF_HOME="$HF_HOME" "$BIN" serve "$MODEL" \
  --backend cuda \
  --port "$PORT_BENCH" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/server/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl" \
  > "$OUT/server/server.log" 2>&1 &
SERVER_PID=$!
BASE="http://127.0.0.1:$PORT_BENCH"
for _ in $(seq 1 300); do
  if curl -sf --noproxy '*' --max-time 2 "$BASE/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[w2-c16] bench server exited" >&2
    tail -80 "$OUT/server/server.log" >&2 || true
    exit 22
  fi
  sleep 2
done
curl -sf --noproxy '*' "$BASE/v1/models" > "$OUT/server/models.json"
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
    raise SystemExit("FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16 not found in effective config")
PY

BENCH_OUT="$OUT/perf/bench_ferrum_sharegpt_c16_100x3.json"
printf '%q ' timeout 7200 "$BIN" bench-serve \
  --base-url "$BASE" \
  --model "$MODEL" \
  --tokenizer "$MODEL_PATH" \
  --dataset sharegpt \
  --sharegpt-path "$DATASET" \
  --random-output-len 128 \
  --concurrency-sweep 16 \
  --num-prompts 100 \
  --n-repeats 3 \
  --fail-on-error \
  --require-ci \
  --seed 9271 \
  --out "$BENCH_OUT" > "$OUT/perf/bench-serve.command.txt"
printf '\n' >> "$OUT/perf/bench-serve.command.txt"
timeout 7200 "$BIN" bench-serve \
  --base-url "$BASE" \
  --model "$MODEL" \
  --tokenizer "$MODEL_PATH" \
  --dataset sharegpt \
  --sharegpt-path "$DATASET" \
  --random-output-len 128 \
  --concurrency-sweep 16 \
  --num-prompts 100 \
  --n-repeats 3 \
  --fail-on-error \
  --require-ci \
  --seed 9271 \
  --out "$BENCH_OUT" \
  > "$OUT/perf/bench-serve.stdout" 2> "$OUT/perf/bench-serve.stderr"
nvidia-smi > "$OUT/perf/nvidia_smi_after_bench.txt"
cleanup_server

python3 - <<'PY' "$OUT" "$BENCH_OUT"
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
bench = json.loads(Path(sys.argv[2]).read_text())
mean = float(bench["output_throughput_tps"]["mean"])
ci = float(bench["output_throughput_tps"].get("ci95_hw") or 0.0)
lcb = mean - ci
historical_vllm_lcb = 491.150
historical_threshold = historical_vllm_lcb * 0.8
summary = {
    "status": "pass",
    "diagnostic_only": True,
    "reason_not_release_grade": (
        "single Ferrum c16 diagnostic; W2 requires c=1/4/16/32, same-hardware "
        "vLLM baseline, full L0-L5 correctness, and MODEL_RELEASE_GRADE_W2 PASS"
    ),
    "git_sha": out.joinpath("env/git_sha.txt").read_text().strip(),
    "git_status_short": out.joinpath("env/git_status_short.txt").read_text().splitlines(),
    "binary_sha256": out.joinpath("env/ferrum.sha256").read_text().split()[0],
    "dataset_sha256": out.joinpath("env/dataset.sha256").read_text().split()[0],
    "completed_per_run": bench.get("completed_per_run"),
    "errored_per_run": bench.get("errored_per_run"),
    "quality_counts": {
        key: bench.get(key)
        for key in [
            "bad_output_per_run",
            "malformed_stream_per_run",
            "missing_done_per_run",
            "duplicate_done_per_run",
            "zero_output_tokens_per_run",
            "stream_bulk_flush_per_run",
            "http_500_per_run",
            "panic_per_run",
        ]
    },
    "output_token_count_source": bench.get("output_token_count_source"),
    "output_throughput_tps_mean": mean,
    "output_throughput_tps_ci95_hw": ci,
    "output_throughput_tps_lcb": lcb,
    "historical_vllm_c16_lcb_same_dataset": historical_vllm_lcb,
    "historical_vllm_80pct_threshold": historical_threshold,
    "diagnostic_ratio_vs_historical_vllm_lcb": lcb / historical_vllm_lcb,
    "gap_to_historical_80pct_threshold": historical_threshold - lcb,
    "itl_ms": bench.get("itl_ms"),
    "actual_input_tokens": bench.get("actual_input_tokens"),
}
out.joinpath("summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
lines = [
    "# W2 Active Chunk ShareGPT c16 CI Diagnostic",
    "",
    f"- Git SHA: `{summary['git_sha']}`",
    f"- Binary SHA256: `{summary['binary_sha256']}`",
    f"- Dataset SHA256: `{summary['dataset_sha256']}`",
    f"- completed_per_run: `{summary['completed_per_run']}`",
    f"- errored_per_run: `{summary['errored_per_run']}`",
    f"- output_token_count_source: `{summary['output_token_count_source']}`",
    f"- output throughput mean: `{mean:.3f} tok/s`",
    f"- output throughput LCB: `{lcb:.3f} tok/s`",
    f"- diagnostic ratio vs historical vLLM LCB: `{(lcb / historical_vllm_lcb) * 100:.2f}%`",
    f"- gap to historical 80% threshold: `{historical_threshold - lcb:.3f} tok/s`",
    "",
    "This is a diagnostic artifact, not a W2 release-grade PASS.",
]
out.joinpath("summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

date -u +%FT%TZ > "$OUT/env/end_utc.txt"
nvidia-smi > "$OUT/env/nvidia_smi_after.txt"
echo "[w2-c16] PASS diagnostic artifact: $OUT"
