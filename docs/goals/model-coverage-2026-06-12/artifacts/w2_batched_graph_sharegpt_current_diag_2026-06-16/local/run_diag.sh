#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/.cargo/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ART="/workspace/w2_batched_graph_sharegpt_current_diag_2026-06-16"
SRC="/workspace/ferrum-batched-graph-diag-01730042"
BIN="/workspace/ferrum-infer-rs/target/release/ferrum"
MODEL="/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2"
TOKENIZER="$MODEL"
DATASET="/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl"
PORT=18477
mkdir -p "$ART/remote" "$ART/server" "$ART/smoke" "$ART/bench" "$ART/local"
date -u +%FT%TZ > "$ART/remote/start_utc.txt"
cd "$SRC"
git rev-parse HEAD > "$ART/remote/git_head.txt"
git status --short --untracked-files=no > "$ART/remote/git_status_short.txt"
nvidia-smi > "$ART/remote/nvidia_smi_before.txt"
nvcc --version > "$ART/remote/nvcc_version.txt"
rustc --version > "$ART/remote/rustc_version.txt"
cargo --version > "$ART/remote/cargo_version.txt"
test -x "$BIN"
sha256sum "$BIN" > "$ART/remote/ferrum.sha256"
"$BIN" --version > "$ART/remote/ferrum_version.txt" 2>&1 || true
test -d "$MODEL"
test -f "$DATASET"
pkill -f "ferrum serve.*--port ${PORT}" || true
sleep 2
cat > "$ART/server/serve.command.json" <<EOF
{
  "command": ["$BIN", "serve", "--model", "gemma3:27b-gptq", "--backend", "cuda", "--host", "127.0.0.1", "--port", "$PORT", "--batched-graph", "--kv-capacity", "512", "--max-num-seqs", "16", "--effective-config-json", "$ART/server/serve_effective_config.json", "--decision-trace-jsonl", "$ART/server/serve_decision_trace.jsonl"],
  "cwd": "$SRC",
  "note": "product CLI --batched-graph diagnostic; no hidden performance env"
}
EOF
(
  cd "$SRC"
  "$BIN" serve \
    --model gemma3:27b-gptq \
    --backend cuda \
    --host 127.0.0.1 \
    --port "$PORT" \
    --batched-graph \
    --kv-capacity 512 \
    --max-num-seqs 16 \
    --effective-config-json "$ART/server/serve_effective_config.json" \
    --decision-trace-jsonl "$ART/server/serve_decision_trace.jsonl"
) > "$ART/server/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$ART/server/server.pid"
ready=0
for i in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "$ART/server/models.json" 2> "$ART/server/models.err"; then
    echo "ready_at_poll=$i" > "$ART/server/ready_poll.txt"
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server_exited_at_poll=$i" > "$ART/server/ready_poll.txt"
    break
  fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  tail -n 200 "$ART/server/server.log" > "$ART/server/server_tail_on_not_ready.log" || true
  exit 20
fi
cat > "$ART/smoke/chat_request.json" <<'EOF'
{"model":"gemma3:27b-gptq","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"max_tokens":1,"temperature":0.0,"stream":false}
EOF
curl -fsS -H 'Content-Type: application/json' \
  -d @"$ART/smoke/chat_request.json" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  > "$ART/smoke/chat_response.json" 2> "$ART/smoke/chat_response.err"
cat > "$ART/bench/bench-serve.command.json" <<EOF
{
  "command": ["$BIN", "bench-serve", "--base-url", "http://127.0.0.1:${PORT}", "--model", "gemma3:27b-gptq", "--tokenizer", "$TOKENIZER", "--dataset", "sharegpt", "--sharegpt-path", "$DATASET", "--random-output-len", "64", "--concurrency-sweep", "16,32", "--num-prompts", "16", "--n-repeats", "1", "--fail-on-error", "--seed", "9271", "--out", "$ART/bench/bench_ferrum_batched_graph_sharegpt_c16_c32_16x1.json"],
  "release_grade": false,
  "reason": "n_repeats=1 diagnostic to isolate --batched-graph on current HEAD; no --require-ci"
}
EOF
set +e
timeout 1800 "$BIN" bench-serve \
  --base-url "http://127.0.0.1:${PORT}" \
  --model gemma3:27b-gptq \
  --tokenizer "$TOKENIZER" \
  --dataset sharegpt \
  --sharegpt-path "$DATASET" \
  --random-output-len 64 \
  --concurrency-sweep 16,32 \
  --num-prompts 16 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271 \
  --out "$ART/bench/bench_ferrum_batched_graph_sharegpt_c16_c32_16x1.json" \
  > "$ART/bench/bench-serve.stdout" 2> "$ART/bench/bench-serve.stderr"
bench_rc=$?
set -e
echo "$bench_rc" > "$ART/bench/bench-serve.rc"
curl -fsS "http://127.0.0.1:${PORT}/health" > "$ART/server/health_after_bench.json" 2> "$ART/server/health_after_bench.err" || true
nvidia-smi > "$ART/remote/nvidia_smi_after_bench.txt" || true
grep -E -i "panic|error|nan|<unk>|\[PAD\]|invalid utf|mojibake|illegal address|cuda error|graph-failed|capture-failed|fallback" "$ART/server/server.log" > "$ART/server/error_scan.txt" || true
python3 - <<'PY' > "$ART/diagnostic_check.json"
import json, pathlib
art=pathlib.Path('/workspace/w2_batched_graph_sharegpt_current_diag_2026-06-16')
bench_rc=int((art/'bench/bench-serve.rc').read_text().strip())
rows=json.loads((art/'bench/bench_ferrum_batched_graph_sharegpt_c16_c32_16x1.json').read_text()) if (art/'bench/bench_ferrum_batched_graph_sharegpt_c16_c32_16x1.json').exists() else []
baseline={16:518.796,32:524.128}
default_ferrum={16:339.93061418630174,32:340.55536966852867}
summary=[]
ok=bench_rc==0 and len(rows)==2
for row in rows:
    c=int(row['concurrency'])
    tps=float(row['output_throughput_tps']['mean'])
    completed=sum(row.get('completed_per_run') or [])
    errored=sum(row.get('errored_per_run') or [])
    bad=sum(row.get('bad_output_per_run') or [])
    zero=sum(row.get('zero_output_tokens_per_run') or [])
    panic=sum(row.get('panic_per_run') or [])
    http500=sum(row.get('http_500_per_run') or [])
    if completed != 16 or errored or bad or zero or panic or http500:
        ok=False
    summary.append({
        'concurrency': c,
        'completed': completed,
        'errored': errored,
        'bad_output': bad,
        'zero_output_tokens': zero,
        'panic': panic,
        'http_500': http500,
        'output_throughput_tps_mean': tps,
        'vllm_baseline_tps': baseline[c],
        'ratio_to_vllm': tps/baseline[c],
        'graph_disabled_current_tps': default_ferrum[c],
        'delta_vs_graph_disabled_tps': tps-default_ferrum[c],
        'delta_vs_graph_disabled_pct': (tps-default_ferrum[c])/default_ferrum[c],
        'ttft_p50_ms': ((row.get('ttft_ms') or {}).get('p50') or {}).get('mean'),
        'tpot_p50_ms': ((row.get('tpot_ms') or {}).get('p50') or {}).get('mean'),
        'itl_p50_ms': ((row.get('itl_ms') or {}).get('p50') or {}).get('mean'),
        'output_token_count_source': row.get('output_token_count_source'),
        'n_repeats': row.get('n_repeats'),
    })
error_scan_lines=0
p=art/'server/error_scan.txt'
if p.exists(): error_scan_lines=len([ln for ln in p.read_text(errors='replace').splitlines() if ln.strip()])
if error_scan_lines: ok=False
config={}
p=art/'server/serve_effective_config.json'
if p.exists():
    obj=json.loads(p.read_text())
    config={'selected_graph_mode': obj.get('selected_graph_mode')}
print(json.dumps({'ok':ok,'release_grade':False,'reason':'diagnostic n_repeats=1; no --require-ci; final W2 validator not run','bench_rc':bench_rc,'rows':summary,'server_error_scan_lines':error_scan_lines,**config}, indent=2, sort_keys=True))
PY
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
date -u +%FT%TZ > "$ART/remote/end_utc.txt"
if [ "$bench_rc" -ne 0 ]; then exit "$bench_rc"; fi
python3 - <<'PY'
import json, pathlib
obj=json.loads(pathlib.Path('/workspace/w2_batched_graph_sharegpt_current_diag_2026-06-16/diagnostic_check.json').read_text())
if not obj.get('ok'): raise SystemExit(30)
print('DIAGNOSTIC PASS: /workspace/w2_batched_graph_sharegpt_current_diag_2026-06-16')
PY
