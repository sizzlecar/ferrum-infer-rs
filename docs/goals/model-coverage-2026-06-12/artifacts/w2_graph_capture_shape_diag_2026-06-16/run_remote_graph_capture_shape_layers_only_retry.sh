#!/usr/bin/env bash
set -euo pipefail

SRC=/workspace/ferrum-infer-rs
OUT=/workspace/w2_graph_capture_shape_layers_only_diag_2026-06-16
BIN_TARGET=/workspace/ferrum-target
MODEL=gemma3:27b-gptq
PORT=18142
SERVER_PID=""

export HF_HOME=/workspace/hf-cache
export HF_XET_HIGH_PERFORMANCE=1
export CARGO_TARGET_DIR="$BIN_TARGET"
export CUDA_COMPUTE_CAP=89
export PATH="/root/.cargo/bin:${PATH}"

mkdir -p "$OUT"/{bench,build,correctness,env,failure,profile,remote,server,smoke}

cleanup() {
  set +e
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    pgrep -af "ferrum serve .*--port ${PORT}" > "$OUT/server/server_ps_before_stop.txt" || true
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
  fi
  pkill -f "ferrum serve .*--port ${PORT}" 2>/dev/null || true
  pgrep -af "ferrum serve .*--port ${PORT}" > "$OUT/server/server_ps_after_stop.txt" || true
}

finish() {
  rc=$?
  echo "$rc" > "$OUT/run_remote.rc"
  cleanup
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  nvidia-smi > "$OUT/remote/nvidia_smi_after.txt" 2>&1 || true
  grep -aE "CUDA_ERROR|cuGraph|\[unified-graph\]|panic|illegal|OOM|out of memory|nodes=|node_types=|mem_free_mib" \
    "$OUT/server/server.log" > "$OUT/failure/error_scan.log" 2>/dev/null || true
  tail -260 "$OUT/server/server.log" > "$OUT/failure/server_tail.log" 2>/dev/null || true
  exit "$rc"
}
trap finish EXIT

cd "$SRC"
date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short --untracked-files=no -- crates scripts Cargo.toml Cargo.lock ferrum.toml \
  > "$OUT/remote/git_status_source_short.txt" || true
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"
nvcc --version > "$OUT/remote/nvcc_version.txt"
df -h / /workspace > "$OUT/remote/df_before.txt" 2>&1 || true

if [ ! -x "$BIN_TARGET/release/ferrum" ]; then
  echo "missing built ferrum binary: $BIN_TARGET/release/ferrum" >&2
  exit 10
fi
sha256sum "$BIN_TARGET/release/ferrum" > "$OUT/build/ferrum.sha256"
"$BIN_TARGET/release/ferrum" --version > "$OUT/build/ferrum.version.txt" 2>&1 || true

find "$HF_HOME/hub" -maxdepth 4 -name tokenizer.json -print > "$OUT/env/tokenizer_candidates.txt" || true
TOKENIZER_JSON=$(grep 'models--circulus--gemma-3-27b-it-gptq/snapshots/.*/tokenizer.json' "$OUT/env/tokenizer_candidates.txt" | head -n 1)
TOKENIZER_DIR=$(dirname "$TOKENIZER_JSON")
if [ -z "$TOKENIZER_JSON" ] || [ "$TOKENIZER_DIR" = "." ]; then
  echo "tokenizer path not found in $HF_HOME" >&2
  exit 20
fi
echo "$TOKENIZER_DIR" > "$OUT/env/tokenizer_dir.txt"

cat > "$OUT/correctness/run.command.txt" <<TXT
$BIN_TARGET/release/ferrum run $MODEL --backend cuda --prompt 'What is 2+3? Answer with only the number.' --max-tokens 8 --temperature 0 --kv-capacity 512 --max-num-seqs 2 --unified-graph --unified-graph-layers-only --output-format jsonl
TXT
timeout 900 "$BIN_TARGET/release/ferrum" run "$MODEL" \
  --backend cuda \
  --prompt "What is 2+3? Answer with only the number." \
  --max-tokens 8 \
  --temperature 0 \
  --kv-capacity 512 \
  --max-num-seqs 2 \
  --unified-graph \
  --unified-graph-layers-only \
  --output-format jsonl \
  > "$OUT/correctness/run.stdout" 2> "$OUT/correctness/run.stderr"
python3 - "$OUT/correctness/run.stdout" > "$OUT/correctness/run_validation.txt" <<'PY'
import json
import re
import sys

content = ""
tokens = 0
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    if obj.get("event") == "assistant":
        content += obj.get("content") or ""
        tokens += int(obj.get("n_tokens") or 0)
if not re.search(r"\b5\b", content) or tokens <= 0:
    raise SystemExit(f"RUN_SMOKE_FAIL content={content!r} tokens={tokens}")
print(f"RUN_SMOKE_PASS content={content.strip()!r} tokens={tokens}")
PY

cat > "$OUT/server/serve.command.txt" <<TXT
$BIN_TARGET/release/ferrum serve --model $MODEL --backend cuda --host 127.0.0.1 --port $PORT --kv-capacity 512 --max-num-seqs 16 --max-num-batched-tokens 2048 --unified-graph --unified-graph-layers-only
TXT
"$BIN_TARGET/release/ferrum" serve \
  --model "$MODEL" \
  --backend cuda \
  --host 127.0.0.1 \
  --port "$PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --unified-graph \
  --unified-graph-layers-only \
  --effective-config-json "$OUT/server/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl" \
  > "$OUT/server/server.stdout" 2> "$OUT/server/server.log" &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server/server.pid"

ready=0
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$OUT/server/models.json.tmp" 2> "$OUT/server/models.err"; then
    mv "$OUT/server/models.json.tmp" "$OUT/server/models.json"
    echo "$i" > "$OUT/server/ready_poll_count.txt"
    ready=1
    break
  fi
  sleep 2
done
if [ "$ready" != "1" ]; then
  echo "server did not become ready" >&2
  exit 30
fi

cat > "$OUT/smoke/chat_request.json" <<JSON
{"model":"$MODEL","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"max_tokens":8,"temperature":0,"stream":false}
JSON
curl -fsS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary "@$OUT/smoke/chat_request.json" \
  > "$OUT/smoke/chat_response.json" 2> "$OUT/smoke/chat_curl.err"
python3 - "$OUT/smoke/chat_response.json" > "$OUT/smoke/chat_validation.txt" <<'PY'
import json
import re
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
content = data["choices"][0]["message"].get("content") or ""
usage = data.get("usage") or {}
completion = int(usage.get("completion_tokens") or 0)
if not re.search(r"\b5\b", content) or completion <= 0:
    raise SystemExit(f"SERVE_SMOKE_FAIL content={content!r} completion_tokens={completion}")
print(f"SERVE_SMOKE_PASS content={content.strip()!r} completion_tokens={completion}")
PY

cat > "$OUT/bench/bench_c16.command.txt" <<TXT
$BIN_TARGET/release/ferrum bench-serve --base-url http://127.0.0.1:$PORT --model $MODEL --tokenizer $TOKENIZER_DIR --dataset random --random-input-len 128 --random-output-len 128 --concurrency 16 --num-prompts 64 --warmup-requests 0 --n-repeats 1 --fail-on-error --seed 9271 --out $OUT/bench/bench_c16.json
TXT
timeout 1200 "$BIN_TARGET/release/ferrum" bench-serve \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER_DIR" \
  --dataset random \
  --random-input-len 128 \
  --random-output-len 128 \
  --concurrency 16 \
  --num-prompts 64 \
  --warmup-requests 0 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271 \
  --out "$OUT/bench/bench_c16.json" \
  > "$OUT/bench/bench_c16.stdout" 2> "$OUT/bench/bench_c16.stderr"

grep -aE "cuGraph|\[unified-graph\]|CUDA_ERROR|graph|nodes=|node_types=|mem_free_mib" "$OUT/server/server.log" \
  > "$OUT/profile/graph_extract.log" || true

python3 - "$OUT" <<'PY' > "$OUT/summary.json"
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
bench_path = out / "bench/bench_c16.json"
bench = json.load(open(bench_path, encoding="utf-8")) if bench_path.exists() else {}
summary = {
    "status": "PASS",
    "release_grade": False,
    "lane": "W2 Gemma3 unified-graph layers-only capture-shape CUDA diagnostic",
    "artifact_dir": str(out),
    "git_head": (out / "remote/git_head.txt").read_text().strip(),
    "git_status_source_short": (out / "remote/git_status_source_short.txt").read_text(),
    "binary_sha256": (out / "build/ferrum.sha256").read_text().strip(),
    "run_validation": (out / "correctness/run_validation.txt").read_text().strip(),
    "serve_validation": (out / "smoke/chat_validation.txt").read_text().strip(),
    "bench_completed_per_run": bench.get("completed_per_run"),
    "bench_errored_per_run": bench.get("errored_per_run"),
    "output_token_count_source": bench.get("output_token_count_source"),
    "output_throughput_tps": bench.get("output_throughput_tps"),
    "graph_extract_tail": (out / "profile/graph_extract.log").read_text(errors="ignore")[-4000:]
        if (out / "profile/graph_extract.log").exists()
        else "",
    "pass_line": None,
}
json.dump(summary, sys.stdout, indent=2, sort_keys=True)
print()
PY

echo "W2 GRAPH CAPTURE SHAPE LAYERS-ONLY DIAG COMPLETE: $OUT"
