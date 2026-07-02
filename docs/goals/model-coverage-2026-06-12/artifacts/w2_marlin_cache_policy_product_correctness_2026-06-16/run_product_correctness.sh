#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/.cargo/bin:${PATH}"
ART="/workspace/w2_marlin_cache_policy_product_correctness_2026-06-16"
SRC_MAIN="/workspace/ferrum-infer-rs"
CLEAN="/workspace/ferrum-product-correctness-212b2bf9"
TARGET_DIR="/workspace/ferrum-infer-rs/target"
MODEL="gemma3:27b-gptq"
MODEL_PATH="/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2"
PORT=18087
BIN="$TARGET_DIR/release/ferrum"
mkdir -p "$ART" "$ART/remote" "$ART/build" "$ART/run" "$ART/server" "$ART/serve_smoke"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ART/remote/started_at_utc.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$ART/remote/nvidia_smi_before.txt"
nvcc --version > "$ART/remote/nvcc_version.txt"
rm -rf "$CLEAN"
git -C "$SRC_MAIN" worktree prune || true
git -C "$SRC_MAIN" worktree add --detach "$CLEAN" HEAD > "$ART/remote/worktree_add.stdout" 2> "$ART/remote/worktree_add.stderr"
cd "$CLEAN"
git rev-parse HEAD > "$ART/remote/git_head.txt"
git status --short --untracked-files=no > "$ART/remote/git_status_short.txt"
set +e
CARGO_TARGET_DIR="$TARGET_DIR" cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source > "$ART/build/build.stdout" 2> "$ART/build/build.stderr"
build_rc=$?
set -e
echo "$build_rc" > "$ART/build/build.rc"
if [ "$build_rc" -ne 0 ]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$ART/remote/finished_at_utc.txt"
  exit 0
fi
sha256sum "$BIN" > "$ART/build/ferrum.sha256"
"$BIN" --version > "$ART/build/ferrum.version.txt" 2>&1 || true
cat > "$ART/run/run.command.txt" <<EOF
$BIN run $MODEL --backend cuda --max-tokens 8 --temperature 0 --kv-capacity 2560 --max-num-seqs 2 --output-format jsonl --effective-config-json "$ART/run/run_effective_config.json" --decision-trace-jsonl "$ART/run/run_decision_trace.jsonl" --prompt "What is 2+3? Answer with only the number."
EOF
set +e
timeout 900 "$BIN" run "$MODEL" \
  --backend cuda \
  --max-tokens 8 \
  --temperature 0 \
  --kv-capacity 2560 \
  --max-num-seqs 2 \
  --output-format jsonl \
  --effective-config-json "$ART/run/run_effective_config.json" \
  --decision-trace-jsonl "$ART/run/run_decision_trace.jsonl" \
  --prompt "What is 2+3? Answer with only the number." \
  > "$ART/run/run.stdout" 2> "$ART/run/run.stderr"
run_rc=$?
set -e
echo "$run_rc" > "$ART/run/run.rc"
cat > "$ART/server/serve.command.txt" <<EOF
timeout 1200 $BIN serve --model $MODEL --backend cuda --host 127.0.0.1 --port $PORT --max-num-seqs 16 --max-num-batched-tokens 2048 --kv-capacity 512 --effective-config-json "$ART/server/serve_effective_config.json" --decision-trace-jsonl "$ART/server/serve_decision_trace.jsonl"
EOF
set +e
timeout 1200 "$BIN" serve \
  --model "$MODEL" \
  --backend cuda \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --kv-capacity 512 \
  --effective-config-json "$ART/server/serve_effective_config.json" \
  --decision-trace-jsonl "$ART/server/serve_decision_trace.jsonl" \
  > "$ART/server/server.stdout" 2> "$ART/server/server.log" &
server_pid=$!
set -e
echo "$server_pid" > "$ART/server/server.pid"
ready=0
for i in $(seq 1 180); do
  if curl -sf --noproxy "*" --max-time 2 "http://127.0.0.1:$PORT/v1/models" > "$ART/server/models.json" 2> "$ART/server/models_curl_last.err"; then
    ready=1
    echo "$i" > "$ART/server/ready_poll_count.txt"
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "server exited during readiness poll $i" > "$ART/server/ready_failure.txt"
    break
  fi
  sleep 2
done
echo "$ready" > "$ART/server/ready"
cat > "$ART/serve_smoke/chat_request.json" <<EOF
{"model":"$MODEL","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"max_tokens":1,"temperature":0,"stream":false}
EOF
chat_rc=99
if [ "$ready" = "1" ]; then
  set +e
  curl -sS --noproxy "*" --max-time 600 \
    -H "Content-Type: application/json" \
    -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    --data @"$ART/serve_smoke/chat_request.json" \
    > "$ART/serve_smoke/chat_response.json" \
    2> "$ART/serve_smoke/chat_curl.err"
  chat_rc=$?
  set -e
fi
echo "$chat_rc" > "$ART/serve_smoke/chat.rc"
curl -sS --noproxy "*" --max-time 5 "http://127.0.0.1:$PORT/health" > "$ART/server/health_after_chat.json" 2> "$ART/server/health_after_chat.err" || true
nvidia-smi > "$ART/server/nvidia_smi_before_stop.txt" || true
if kill -0 "$server_pid" 2>/dev/null; then
  kill "$server_pid" 2> "$ART/server/kill_server.err" || true
  wait "$server_pid" > "$ART/server/stop_server.stdout" 2> "$ART/server/stop_server.stderr" || true
fi
nvidia-smi > "$ART/server/nvidia_smi_after_server_stop.txt" || true
set +e
grep -E -i "panic|error|nan|<unk>|\\[PAD\\]|invalid utf|mojibake|illegal address|cuda error" "$ART/server/server.log" > "$ART/server/error_scan.txt"
scan_rc=$?
set -e
echo "$scan_rc" > "$ART/server/error_scan.grep_rc"
python3 - <<PY > "$ART/correctness_check.json"
import json, pathlib, sys
art = pathlib.Path("$ART")
result = {"ok": False, "checks": {}}
try:
    run_rc = int((art/"run/run.rc").read_text().strip())
except Exception:
    run_rc = -1
result["checks"]["run_rc"] = run_rc
run_content = ""
for line in (art/"run/run.stdout").read_text(errors="replace").splitlines():
    try:
        row = json.loads(line)
    except Exception:
        continue
    if row.get("event") == "assistant":
        run_content = row.get("content") or ""
result["checks"]["run_content"] = run_content
try:
    chat_rc = int((art/"serve_smoke/chat.rc").read_text().strip())
except Exception:
    chat_rc = -1
result["checks"]["chat_rc"] = chat_rc
serve_content = ""
usage = None
try:
    resp = json.loads((art/"serve_smoke/chat_response.json").read_text())
    serve_content = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    usage = resp.get("usage")
except Exception as exc:
    result["checks"]["serve_parse_error"] = str(exc)
result["checks"]["serve_content"] = serve_content
result["checks"]["usage"] = usage
error_lines = (art/"server/error_scan.txt").read_text(errors="replace").splitlines() if (art/"server/error_scan.txt").exists() else []
result["checks"]["error_scan_lines"] = len(error_lines)
result["ok"] = (
    run_rc == 0 and run_content.strip() == "5" and
    chat_rc == 0 and serve_content.strip() == "5" and
    len(error_lines) == 0
)
print(json.dumps(result, indent=2, sort_keys=True))
sys.exit(0 if result["ok"] else 1)
PY
check_rc=$?
echo "$check_rc" > "$ART/correctness_check.rc"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ART/remote/finished_at_utc.txt"
exit 0
