#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/ferrum-infer-rs}"
OUT="${OUT:-/workspace/w2_active_decode_chunk_c16_cuda_diag_2026-06-17}"
MODEL="${MODEL:-gemma3:27b-gptq}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
HF_REPO="${HF_REPO:-circulus/gemma-3-27b-it-gptq}"
PORT_CORRECTNESS="${PORT_CORRECTNESS:-18124}"
PORT_PROFILE="${PORT_PROFILE:-18125}"

mkdir -p "$OUT"/{env,build,prefetch,correctness,server,bench,profile}
exec > >(tee -a "$OUT/run.log") 2>&1

echo "[diag] start $(date -u +%FT%TZ)"
echo "[diag] root=$ROOT out=$OUT model=$MODEL hf_home=$HF_HOME"

cleanup_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
cleanup_prefetch() {
  if [ -n "${PREFETCH_PID:-}" ]; then
    if kill -0 "$PREFETCH_PID" 2>/dev/null; then
      kill "$PREFETCH_PID" 2>/dev/null || true
      wait "$PREFETCH_PID" 2>/dev/null || true
    fi
  fi
}
trap 'cleanup_server; cleanup_prefetch' EXIT

nvidia-smi > "$OUT/env/nvidia_smi_before.txt"
nvcc --version > "$OUT/env/nvcc_version.txt"
df -h /workspace / > "$OUT/env/df_before.txt" 2>&1 || true

cd "$ROOT"
git rev-parse HEAD > "$OUT/env/git_sha.txt"
git status --short > "$OUT/env/git_status_short.txt"

if ! command -v cargo >/dev/null 2>&1; then
  echo "[diag] installing rustup minimal toolchain"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal > "$OUT/env/rustup_install.log" 2>&1
fi
# shellcheck disable=SC1090
source "$HOME/.cargo/env"
rustc -V > "$OUT/env/rustc_version.txt"
cargo -V > "$OUT/env/cargo_version.txt"

echo "[diag] ensuring native build/python dependencies"
DEBIAN_FRONTEND=noninteractive apt-get install -y clang libclang-dev python3-venv > "$OUT/env/apt_extra.log" 2>&1

if [ ! -x /workspace/hf-venv/bin/python ]; then
  echo "[diag] creating HF helper venv"
  python3 -m venv /workspace/hf-venv
  /workspace/hf-venv/bin/python -m pip install -U pip > "$OUT/env/pip_install.log" 2>&1
  /workspace/hf-venv/bin/python -m pip install -U "huggingface_hub[hf_xet]" >> "$OUT/env/pip_install.log" 2>&1
fi

MODEL_ROOT="$HF_HOME/hub/models--circulus--gemma-3-27b-it-gptq"
if [ ! -d "$MODEL_ROOT/snapshots" ] || [ -z "$(find "$MODEL_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)" ]; then
  echo "[diag] starting anonymous HF prefetch for $HF_REPO"
  mkdir -p "$HF_HOME"
  (
    export HF_HOME
    export HF_XET_HIGH_PERFORMANCE=1
    /workspace/hf-venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("circulus/gemma-3-27b-it-gptq", cache_dir="/workspace/hf-cache/hub")
PY
  ) > "$OUT/prefetch/hf_snapshot_download.log" 2>&1 &
  PREFETCH_PID=$!
else
  echo "[diag] HF snapshot already present"
  PREFETCH_PID=""
fi

echo "[diag] starting CUDA release build"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
time cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$OUT/build/cargo_build_release.log" 2>&1
sha256sum target/release/ferrum > "$OUT/env/ferrum.sha256"

if [ -n "${PREFETCH_PID:-}" ]; then
  echo "[diag] waiting for HF prefetch pid=$PREFETCH_PID"
  wait "$PREFETCH_PID"
  PREFETCH_PID=""
fi

MODEL_PATH="$(find "$MODEL_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
  echo "[diag] missing model snapshot under $MODEL_ROOT/snapshots" >&2
  exit 20
fi
printf '%s\n' "$MODEL_PATH" > "$OUT/env/model_path.txt"
find "$MODEL_PATH" -maxdepth 1 -type f -printf '%f\n' | sort > "$OUT/env/model_files.txt"

BIN="$ROOT/target/release/ferrum"

echo "[diag] correctness: ferrum run"
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

echo "[diag] correctness: ferrum serve streaming smoke"
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
    echo "[diag] correctness server exited" >&2
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

echo "[diag] profile: ferrum serve + c16 bench"
SERVER_PID=""
FERRUM_BATCH_DECODE_PROF=1 \
FERRUM_NEXT_BATCH_PROF=1 \
FERRUM_UNIFIED_POST_PROF=1 \
FERRUM_DECODE_OP_PROFILE=1 \
FERRUM_MARLIN_PROFILE=1 \
HF_HOME="$HF_HOME" "$BIN" serve "$MODEL" \
  --backend cuda \
  --port "$PORT_PROFILE" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/server/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl" \
  > "$OUT/server/serve_profile.log" 2>&1 &
SERVER_PID=$!
BASE="http://127.0.0.1:$PORT_PROFILE"
for _ in $(seq 1 300); do
  if curl -sf --noproxy '*' --max-time 2 "$BASE/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[diag] profile server exited" >&2
    tail -80 "$OUT/server/serve_profile.log" >&2 || true
    exit 22
  fi
  sleep 2
done
curl -sf --noproxy '*' "$BASE/v1/models" > "$OUT/server/models.json"

python3 - <<'PY' "$OUT/server/serve_effective_config.json" "$OUT/profile/effective_config_chunk_check.json"
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
out = {"matches": matches}
Path(sys.argv[2]).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
if not any(str(m.get("effective_value")) == "16" for m in matches):
    raise SystemExit("FERRUM_ACTIVE_DECODE_PREFILL_CHUNK=16 not found in effective config")
PY

python3 - <<PY
import json
from pathlib import Path
cmd = [
    "timeout", "1800", "$BIN", "bench-serve",
    "--base-url", "$BASE",
    "--model", "$MODEL",
    "--tokenizer", "$MODEL_PATH",
    "--dataset", "random",
    "--random-input-len", "256",
    "--random-output-len", "128",
    "--concurrency-sweep", "16",
    "--num-prompts", "16",
    "--n-repeats", "1",
    "--fail-on-error",
    "--seed", "9271",
    "--out", "$OUT/bench/bench_random_c16.json",
]
Path("$OUT/bench/bench-serve.command.json").write_text(json.dumps({"cmd": cmd}, indent=2) + "\n")
PY
timeout 1800 "$BIN" bench-serve \
  --base-url "$BASE" \
  --model "$MODEL" \
  --tokenizer "$MODEL_PATH" \
  --dataset random \
  --random-input-len 256 \
  --random-output-len 128 \
  --concurrency-sweep 16 \
  --num-prompts 16 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271 \
  --out "$OUT/bench/bench_random_c16.json" \
  > "$OUT/bench/bench-serve.stdout" 2> "$OUT/bench/bench-serve.stderr"
cleanup_server

nvidia-smi > "$OUT/env/nvidia_smi_after.txt"
df -h /workspace / > "$OUT/env/df_after.txt" 2>&1 || true

python3 - <<'PY' "$OUT"
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
bench = json.loads((out / "bench/bench_random_c16.json").read_text())
server_log = (out / "server/serve_profile.log").read_text(errors="replace")
profile_lines = [
    line for line in server_log.splitlines()
    if "decode_op_profile" in line
    or "marlin" in line.lower()
    or "batch_decode" in line
    or "next_batch" in line
    or "unified_post" in line
]
mixed = [
    line for line in server_log.splitlines()
    if re.search(r"prefill=1[0-9].*decode=4|prefill=12.*decode=4", line)
]
summary = {
    "artifact_dir": str(out),
    "model": "gemma3:27b-gptq",
    "bench_top_level_keys": sorted(bench.keys()),
    "profile_line_count": len(profile_lines),
    "large_mixed_prefill_decode_lines": mixed[:20],
    "release_claim": False,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[diag] complete $(date -u +%FT%TZ)"
