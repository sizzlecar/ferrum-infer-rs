#!/usr/bin/env bash
set -euo pipefail
ART=/workspace/w2_typed_decode_profile_2026-06-16
SRC=/workspace/ferrum-infer-rs
RUN_CWD="$ART/run_cwd"
MODEL=/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
MODEL_ID=70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
PORT=18116
SERVER_PID=""
cleanup() {
  set +e
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    pgrep -af "$SRC/target/release/ferrum serve .*--port $PORT" > "$ART/server/server_ps_before_stop.txt" || true
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
  fi
  pkill -f "$SRC/target/release/ferrum serve .*--port $PORT" 2>/dev/null || true
  pgrep -af "$SRC/target/release/ferrum serve .*--port $PORT" > "$ART/server/server_ps_after_stop.txt" || true
}
finish() {
  rc=$?
  echo "$rc" > "$ART/run_profile.rc"
  cleanup
  exit "$rc"
}
trap finish EXIT
mkdir -p "$ART/server" "$ART/smoke" "$ART/bench" "$ART/profile" "$ART/remote" "$RUN_CWD"
printf "%s\n" "W2 Gemma3 typed-config decode integration profile" > "$ART/lane.txt"
printf "%s\n" "$MODEL" > "$ART/model_path.txt"
printf "%s\n" "$MODEL_ID" > "$ART/model_id.txt"
cp "$SRC/ferrum.toml" "$RUN_CWD/ferrum.toml"
cat >> "$RUN_CWD/ferrum.toml" <<EOF
batch_decode_prof = true
unified_post_prof = true
next_batch_prof = true
batched_graph = true
EOF
nvidia-smi > "$ART/remote/nvidia_smi_before_profile.txt"
cd "$RUN_CWD"
SERVE_CMD=("$SRC/target/release/ferrum" serve --model "$MODEL" --backend cuda --host 127.0.0.1 --port "$PORT" --batched-graph --max-num-seqs 16 --max-num-batched-tokens 1024 --kv-capacity 256 --effective-config-json "$ART/server/serve_effective_config.json" --decision-trace-jsonl "$ART/server/serve_decision_trace.jsonl")
printf "%q " "${SERVE_CMD[@]}" > "$ART/server/serve.command.txt"
printf "\n" >> "$ART/server/serve.command.txt"
"${SERVE_CMD[@]}" > "$ART/server/server.stdout" 2> "$ART/server/server.log" &
SERVER_PID=$!
echo "$SERVER_PID" > "$ART/server/server.pid"
ready=0
for i in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$ART/server/models.json.tmp" 2> "$ART/server/models_curl_last.err"; then
    mv "$ART/server/models.json.tmp" "$ART/server/models.json"
    echo "$i" > "$ART/server/ready_poll_count.txt"
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != "1" ]]; then
  tail -200 "$ART/server/server.log" > "$ART/server/server_tail_on_not_ready.log" || true
  exit 10
fi
cat > "$ART/smoke/chat_request.json" <<EOF
{"model":"$MODEL_ID","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"max_tokens":16,"temperature":0,"stream":false}
EOF
curl -fsS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" --data-binary "@$ART/smoke/chat_request.json" > "$ART/smoke/chat_response.json" 2> "$ART/smoke/chat_curl.err"
python3 - "$ART/smoke/chat_response.json" <<PY
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
content = data["choices"][0]["message"].get("content") or ""
usage = data.get("usage") or {}
completion = int(usage.get("completion_tokens") or 0)
if content.strip() != "5" or completion <= 0:
    raise SystemExit(f"chat smoke failed: content={content!r} completion={completion}")
print(f"CHAT_SMOKE_PASS content={content.strip()!r} completion_tokens={completion}")
PY
BENCH_CMD=("$SRC/target/release/ferrum" bench-serve --base-url "http://127.0.0.1:$PORT" --model "$MODEL_ID" --tokenizer "$MODEL" --concurrency 16 --num-prompts 16 --warmup-requests 0 --random-input-len 32 --random-output-len 32 --fail-on-error --seed 9271 --output json --out "$ART/bench/bench_c16.json")
printf "%q " "${BENCH_CMD[@]}" > "$ART/bench/bench_c16.command.txt"
printf "\n" >> "$ART/bench/bench_c16.command.txt"
set +e
"${BENCH_CMD[@]}" > "$ART/bench/bench_c16.stdout" 2> "$ART/bench/bench_c16.stderr"
bench_rc=$?
set -e
echo "$bench_rc" > "$ART/bench/bench_c16.rc"
if [[ "$bench_rc" -ne 0 ]]; then
  tail -200 "$ART/server/server.log" > "$ART/server/server_tail_on_bench_fail.log" || true
  exit "$bench_rc"
fi
grep -aE "\\[iter-prof\\]|\\[unified-prof\\]|\\[nb-prof\\]" "$ART/server/server.log" > "$ART/profile/profile_extract.log" || true
grep -aE "ConfigFile|FERRUM_BATCH_DECODE_PROF|FERRUM_UNIFIED_POST_PROF|FERRUM_NEXT_BATCH_PROF|FERRUM_BATCHED_GRAPH" "$ART/server/serve_decision_trace.jsonl" > "$ART/profile/runtime_decision_trace_extract.log" || true
python3 - "$ART/bench/bench_c16.json" <<PY > "$ART/bench/bench_summary.txt"
import json, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
print("completed_requests", d.get("completed_requests"))
print("errored_requests", d.get("errored_requests"))
print("output_throughput_tps_mean", (d.get("output_throughput_tps") or {}).get("mean"))
print("ttft_ms_p50", ((d.get("time_to_first_token_ms") or {}).get("p50")))
print("tpot_ms_p50", ((d.get("time_per_output_token_ms") or {}).get("p50")))
print("output_tokens_per_request", d.get("output_tokens_per_request"))
PY
nvidia-smi > "$ART/remote/nvidia_smi_after_profile.txt"
