#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/ferrum-smoke-99859ce6}"
OUT="${OUT:-/workspace/w2_autosize_serve_smoke_cuda_2026-06-17}"
MODEL="${MODEL:-gemma3:27b-gptq}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
PORT="${PORT:-18456}"

mkdir -p "$OUT"/{env,build,correctness}
exec > >(tee -a "$OUT/run.log") 2>&1

echo "[autosize-smoke] start $(date -u +%FT%TZ)"
echo "[autosize-smoke] root=$ROOT out=$OUT model=$MODEL hf_home=$HF_HOME port=$PORT"
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
  for _ in $(seq 1 420); do
    if curl -sf --noproxy '*' --max-time 2 "$base/v1/models" >/dev/null; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[autosize-smoke] server exited before health" >&2
      tail -200 "$log_file" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "[autosize-smoke] timed out waiting for server" >&2
  tail -200 "$log_file" >&2 || true
  return 1
}

run_stream_smoke() {
  local base="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  python3 - <<'PY' "$base" "$MODEL" "$out_dir"
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

extract_effective_config() {
  python3 - <<'PY' "$OUT/correctness/serve_effective_config.json" "$OUT/correctness/effective_config_keys.json"
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
out = Path(sys.argv[2])
data = json.loads(source.read_text())
wanted = {
    "FERRUM_PAGED_MAX_SEQS",
    "FERRUM_KV_CAPACITY",
    "FERRUM_MAX_BATCHED_TOKENS",
    "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK",
}
matches = []

def walk(node):
    if isinstance(node, dict):
        if node.get("key") in wanted:
            matches.append(node)
        for value in node.values():
            walk(value)
    elif isinstance(node, list):
        for value in node:
            walk(value)

walk(data)
out.write_text(json.dumps({"matches": matches}, indent=2, sort_keys=True) + "\n")
PY
}

cd "$ROOT"
git rev-parse HEAD > "$OUT/env/git_head.txt"
git status --short > "$OUT/env/git_status_short.txt"
nvidia-smi > "$OUT/env/nvidia_smi_before.txt"
nvcc --version > "$OUT/env/nvcc_version.txt" 2>&1 || true
df -h /workspace / > "$OUT/env/df_before.txt" 2>&1 || true

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "[autosize-smoke] installing rustup minimal toolchain"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal > "$OUT/env/rustup_install.log" 2>&1
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi
rustc -V > "$OUT/env/rustc_version.txt"
cargo -V > "$OUT/env/cargo_version.txt"

echo "[autosize-smoke] build"
(
  set -x
  time cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
) > "$OUT/build/cargo_build_release.log" 2>&1
BIN="$ROOT/target/release/ferrum"
test -x "$BIN"
sha256sum "$BIN" > "$OUT/env/ferrum.sha256"

echo "[autosize-smoke] serve correctness"
BASE="http://127.0.0.1:$PORT"
python3 - <<PY
import json
from pathlib import Path
cmd = [
    "$BIN", "serve", "$MODEL",
    "--backend", "cuda",
    "--port", "$PORT",
    "--kv-capacity", "512",
    "--max-num-seqs", "32",
    "--effective-config-json", "$OUT/correctness/serve_effective_config.json",
    "--decision-trace-jsonl", "$OUT/correctness/serve_decision_trace.jsonl",
]
Path("$OUT/correctness/serve.command.json").write_text(json.dumps({"cmd": cmd}, indent=2) + "\\n")
PY
HF_HOME="$HF_HOME" "$BIN" serve "$MODEL" \
  --backend cuda \
  --port "$PORT" \
  --kv-capacity 512 \
  --max-num-seqs 32 \
  --effective-config-json "$OUT/correctness/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/correctness/serve_decision_trace.jsonl" \
  > "$OUT/correctness/serve.log" 2>&1 &
SERVER_PID=$!
wait_http "$BASE" "$OUT/correctness/serve.log"
curl -sf --noproxy '*' "$BASE/v1/models" > "$OUT/correctness/models.json"
extract_effective_config
run_stream_smoke "$BASE" "$OUT/correctness/smoke"
cleanup_server

set +e
grep -Eai 'out of memory|oom|panic|cuda_error_out_of_memory' "$OUT/correctness/serve.log" > "$OUT/correctness/log_blocker_scan.txt"
SCAN_RC=$?
set -e
if [ "$SCAN_RC" -eq 0 ]; then
  echo "[autosize-smoke] blocker strings found in serve.log" >&2
  cat "$OUT/correctness/log_blocker_scan.txt" >&2
  exit 31
fi

nvidia-smi > "$OUT/env/nvidia_smi_after.txt"
date -u +%FT%TZ > "$OUT/env/end_utc.txt"
python3 - <<'PY' "$OUT"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = {
    "status": "pass",
    "git_head": out.joinpath("env/git_head.txt").read_text().strip(),
    "binary_sha256": out.joinpath("env/ferrum.sha256").read_text().strip(),
    "stream_summary": json.loads(out.joinpath("correctness/smoke/stream_summary.json").read_text()),
    "effective_config_keys": json.loads(out.joinpath("correctness/effective_config_keys.json").read_text()),
}
out.joinpath("summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
echo "W2 AUTOSIZE SERVE SMOKE PASS: $OUT"
