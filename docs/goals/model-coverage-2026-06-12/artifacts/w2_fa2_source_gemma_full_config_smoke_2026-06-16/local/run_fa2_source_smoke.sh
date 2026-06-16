#!/usr/bin/env bash
set -euo pipefail

ART="${ART:-/workspace/w2_fa2_source_gemma_full_config_smoke_2026-06-16}"
WORKTREE="${WORKTREE:-/workspace/ferrum-fa2-source-smoke-d6fe78d6}"
FERRUM="${FERRUM:-/workspace/ferrum-infer-rs/target/release/ferrum}"
MODEL_PATH="${MODEL_PATH:-/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2}"
DATASET="${DATASET:-/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl}"
PORT="${PORT:-18479}"

mkdir -p "$ART"/{bench,config,remote,server,smoke}

date -u +%FT%TZ > "$ART/remote/start_utc.txt"
(
  cd "$WORKTREE"
  git rev-parse HEAD > "$ART/remote/git_head.txt"
  git status --short > "$ART/remote/git_status_short.txt"
)
sha256sum "$FERRUM" > "$ART/remote/ferrum.sha256"
"$FERRUM" --version > "$ART/remote/ferrum_version.txt" 2>&1 || true
rustc --version > "$ART/remote/rustc_version.txt" 2>&1 || true
cargo --version > "$ART/remote/cargo_version.txt" 2>&1 || true
nvcc --version > "$ART/remote/nvcc_version.txt" 2>&1 || true
nvidia-smi > "$ART/remote/nvidia_smi_before.txt" 2>&1 || true

cp "$WORKTREE/ferrum.toml" "$ART/ferrum.toml"
cat >> "$ART/ferrum.toml" <<'TOML'

# Diagnostic override for W2 FA2-source product smoke.
use_vllm_paged_attn = true
fa2_source = true
fa2_direct_ffi = false
greedy_argmax = true
TOML
cp "$ART/ferrum.toml" "$ART/config/ferrum.toml"

python3 - <<PY > "$ART/server/serve.command.json"
import json
print(json.dumps({
  "cwd": "$ART",
  "cmd": [
    "$FERRUM", "serve",
    "--model", "gemma3:27b-gptq",
    "--backend", "cuda",
    "--host", "127.0.0.1",
    "--port", "$PORT",
    "--kv-capacity", "512",
    "--max-num-seqs", "16",
    "--effective-config-json", "$ART/server/serve_effective_config.json",
    "--decision-trace-jsonl", "$ART/server/serve_decision_trace.jsonl"
  ],
  "note": "artifact-local ferrum.toml selects runtime.fa2_source=true; diagnostic only"
}, indent=2))
PY

(
  cd "$ART"
  "$FERRUM" serve \
    --model gemma3:27b-gptq \
    --backend cuda \
    --host 127.0.0.1 \
    --port "$PORT" \
    --kv-capacity 512 \
    --max-num-seqs 16 \
    --effective-config-json "$ART/server/serve_effective_config.json" \
    --decision-trace-jsonl "$ART/server/serve_decision_trace.jsonl"
) > "$ART/server/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$ART/server/server.pid"

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
  fi
}
trap cleanup EXIT

READY=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" > "$ART/server/health_before.json" 2>"$ART/server/health_before.err"; then
    READY=1
    echo "ready_at_poll=$i" > "$ART/server/ready_poll.txt"
    break
  fi
  sleep 2
done
if [[ "$READY" != "1" ]]; then
  echo "server_not_ready" > "$ART/run.status"
  exit 20
fi

python3 - <<'PY' "$ART/server/serve_decision_trace.jsonl" "$ART/server/attention_decision_check.json"
import json, sys
trace_path, out_path = sys.argv[1], sys.argv[2]
selected = None
for line in open(trace_path):
    row = json.loads(line)
    if row.get("selection") == "attention_prefill_mixed_backend":
        selected = row.get("selected")
        break
ok = selected == "fa2_source"
json.dump({"ok": ok, "selected": selected}, open(out_path, "w"), indent=2, sort_keys=True)
if not ok:
    raise SystemExit(21)
PY

cat > "$ART/smoke/chat_request.json" <<'JSON'
{"model":"gemma3:27b-gptq","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"max_tokens":1,"temperature":0.0,"stream":false}
JSON
curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @"$ART/smoke/chat_request.json" \
  > "$ART/smoke/chat_response.json" \
  2> "$ART/smoke/chat_response.err"

python3 - <<'PY' "$ART/smoke/chat_response.json" "$ART/smoke/chat_check.json"
import json, sys
resp = json.load(open(sys.argv[1]))
content = resp["choices"][0]["message"].get("content", "").strip()
usage = resp.get("usage")
ok = content == "5" and bool(usage)
json.dump({"ok": ok, "content": content, "usage": usage}, open(sys.argv[2], "w"), indent=2, sort_keys=True)
if not ok:
    raise SystemExit(22)
PY

python3 - <<PY > "$ART/bench/bench-serve.command.json"
import json
print(json.dumps({
  "cmd": [
    "$FERRUM", "bench-serve",
    "--base-url", "http://127.0.0.1:${PORT}",
    "--model", "gemma3:27b-gptq",
    "--tokenizer", "$MODEL_PATH",
    "--dataset", "sharegpt",
    "--sharegpt-path", "$DATASET",
    "--random-output-len", "64",
    "--concurrency-sweep", "16",
    "--num-prompts", "16",
    "--n-repeats", "1",
    "--fail-on-error",
    "--seed", "9271",
    "--out", "$ART/bench/bench_fa2_source_sharegpt_c16_16x1.json"
  ],
  "release_grade": False,
  "reason": "minimal typed FA2-source product smoke; no --require-ci"
}, indent=2))
PY

set +e
timeout 900 "$FERRUM" bench-serve \
  --base-url "http://127.0.0.1:${PORT}" \
  --model gemma3:27b-gptq \
  --tokenizer "$MODEL_PATH" \
  --dataset sharegpt \
  --sharegpt-path "$DATASET" \
  --random-output-len 64 \
  --concurrency-sweep 16 \
  --num-prompts 16 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271 \
  --out "$ART/bench/bench_fa2_source_sharegpt_c16_16x1.json" \
  > "$ART/bench/bench-serve.stdout" \
  2> "$ART/bench/bench-serve.stderr"
BENCH_RC=$?
set -e
echo "$BENCH_RC" > "$ART/bench/bench-serve.rc"
curl -sf "http://127.0.0.1:${PORT}/health" > "$ART/server/health_after_bench.json" 2>"$ART/server/health_after_bench.err" || true

python3 - <<'PY' "$ART"
import json, pathlib, sys
art = pathlib.Path(sys.argv[1])
bench_rc = int((art / "bench/bench-serve.rc").read_text().strip())
rows = []
if (art / "bench/bench_fa2_source_sharegpt_c16_16x1.json").exists():
    data = json.load(open(art / "bench/bench_fa2_source_sharegpt_c16_16x1.json"))
    raw_rows = data if isinstance(data, list) else data.get("results", data.get("rows", []))
    for row in raw_rows:
        c = row.get("concurrency") or row.get("concurrency_level")
        completed = row.get("completed_requests", row.get("completed", 0))
        errored = row.get("errored_requests", row.get("errored", 0))
        bad = sum(row.get("bad_output_per_run") or [])
        zero = sum(row.get("zero_output_tokens_per_run") or [])
        tps = row.get("output_throughput_tps_mean") or row.get("output_tokens_per_second")
        rows.append({
            "concurrency": c,
            "completed": completed,
            "errored": errored,
            "bad_output": bad,
            "zero_output_tokens": zero,
            "output_throughput_tps_mean": tps,
            "output_token_count_source": row.get("output_token_count_source"),
        })
scan_terms = ("panic", "ERROR", "NaN", "<unk>", "[PAD]", "invalid UTF", "mojibake")
scan = []
server_log = art / "server/server.log"
if server_log.exists():
    for line in server_log.read_text(errors="ignore").splitlines():
        if any(term in line for term in scan_terms):
            scan.append(line)
(art / "server/error_scan.txt").write_text("\n".join(scan) + ("\n" if scan else ""))
ok = bench_rc == 0 and rows and all(
    r["completed"] == 16
    and r["errored"] == 0
    and r["bad_output"] == 0
    and r["zero_output_tokens"] == 0
    and r["output_token_count_source"] == "usage"
    for r in rows
) and len(scan) == 0
json.dump({
    "ok": ok,
    "bench_rc": bench_rc,
    "rows": rows,
    "server_error_scan_lines": len(scan),
    "reason": "diagnostic n_repeats=1; no --require-ci; final W2 validator not run",
    "release_grade": False,
}, open(art / "diagnostic_check.json", "w"), indent=2, sort_keys=True)
if not ok:
    raise SystemExit(23)
PY

nvidia-smi > "$ART/remote/nvidia_smi_after_bench.txt" 2>&1 || true
date -u +%FT%TZ > "$ART/remote/end_utc.txt"
echo PASS > "$ART/run.status"
