#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ferrum-infer-rs

if [ -f /root/.cargo/env ]; then
  . /root/.cargo/env
fi
export PATH="/root/.cargo/bin:${PATH}"

OUT=/workspace/w2_unified_graph_typed_c16_2026-06-16
MODEL=gemma3:27b-gptq
MODEL_PATH=/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
DATASET=/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
PORT=18123
SERVER_PID=""

mkdir -p "$OUT" "$OUT/build" "$OUT/remote" "$OUT/correctness" "$OUT/server" "$OUT/bench" "$OUT/profile"

cleanup() {
  set +e
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    pgrep -af "target/release/ferrum serve .*--port $PORT" > "$OUT/server/server_ps_before_stop.txt" || true
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
  fi
  pkill -f "target/release/ferrum serve .*--port $PORT" 2>/dev/null || true
  pgrep -af "target/release/ferrum serve .*--port $PORT" > "$OUT/server/server_ps_after_stop.txt" || true
}

finish() {
  rc=$?
  echo "$rc" > "$OUT/run_remote.rc"
  cleanup
  date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
  exit "$rc"
}
trap finish EXIT

date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short.txt" || true
git status --short --untracked-files=no -- crates scripts Cargo.toml Cargo.lock ferrum.toml > "$OUT/remote/git_status_source_short.txt" || true
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"
rustc --version > "$OUT/remote/rustc_version.txt"
cargo --version > "$OUT/remote/cargo_version.txt"
nvcc --version > "$OUT/remote/nvcc_version.txt"

if [ -d /usr/local/cuda/compat ]; then
  mkdir -p /usr/local/cuda/compat.disabled-ferrum
  mv /usr/local/cuda/compat/libcuda* /usr/local/cuda/compat.disabled-ferrum/ 2>/dev/null || true
  ldconfig 2>/dev/null || true
fi

cat > "$OUT/build/release_build.command.txt" <<'TXT'
CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
TXT
CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$OUT/build/release_build.stdout" 2> "$OUT/build/release_build.stderr"
sha256sum target/release/ferrum > "$OUT/build/ferrum.sha256"
target/release/ferrum --version > "$OUT/build/ferrum_version.txt" 2>&1 || true

cat > "$OUT/correctness/run.command.json" <<JSON
{"cmd":["timeout","1200","target/release/ferrum","run","$MODEL","--backend","cuda","--prompt","What is 2+3? Answer with just the number.","--max-tokens","64","--temperature","0","--kv-capacity","2560","--max-num-seqs","2","--unified-graph","--output-format","jsonl","--effective-config-json","$OUT/correctness/run_effective_config.json","--decision-trace-jsonl","$OUT/correctness/run_decision_trace.jsonl"]}
JSON
timeout 1200 target/release/ferrum run "$MODEL" \
  --backend cuda \
  --prompt "What is 2+3? Answer with just the number." \
  --max-tokens 64 \
  --temperature 0 \
  --kv-capacity 2560 \
  --max-num-seqs 2 \
  --unified-graph \
  --output-format jsonl \
  --effective-config-json "$OUT/correctness/run_effective_config.json" \
  --decision-trace-jsonl "$OUT/correctness/run_decision_trace.jsonl" \
  > "$OUT/correctness/run.stdout" 2> "$OUT/correctness/run.stderr"
python3 - "$OUT/correctness/run.stdout" > "$OUT/correctness/run_validation.txt" <<'PY'
import json
import sys

content = ""
tokens = 0
for line in open(sys.argv[1], "r", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    if obj.get("event") == "assistant":
        content += obj.get("content") or ""
        tokens += int(obj.get("n_tokens") or 0)
if content.strip() != "5" or tokens <= 0:
    raise SystemExit(f"RUN_SMOKE_FAIL content={content!r} tokens={tokens}")
print(f"RUN_SMOKE_PASS content={content.strip()!r} tokens={tokens}")
PY

target/release/ferrum serve \
  --model "$MODEL" \
  --backend cuda \
  --host 127.0.0.1 \
  --port "$PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --unified-graph \
  --effective-config-json "$OUT/server/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl" \
  > "$OUT/server/server.stdout" 2> "$OUT/server/server.log" &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server/server.pid"
printf "%s\n" "target/release/ferrum serve --model $MODEL --backend cuda --host 127.0.0.1 --port $PORT --kv-capacity 512 --max-num-seqs 16 --max-num-batched-tokens 2048 --unified-graph --effective-config-json $OUT/server/serve_effective_config.json --decision-trace-jsonl $OUT/server/serve_decision_trace.jsonl" > "$OUT/server/serve.command.txt"

ready=0
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$OUT/server/models.json.tmp" 2> "$OUT/server/models_curl_last.err"; then
    mv "$OUT/server/models.json.tmp" "$OUT/server/models.json"
    echo "$i" > "$OUT/server/ready_poll_count.txt"
    ready=1
    break
  fi
  sleep 2
done
if [ "$ready" != "1" ]; then
  tail -200 "$OUT/server/server.log" > "$OUT/server/server_tail_on_not_ready.log" || true
  exit 10
fi

cat > "$OUT/correctness/serve_chat_request.json" <<JSON
{"model":"$MODEL","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"max_tokens":16,"temperature":0,"stream":false}
JSON
curl -fsS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary "@$OUT/correctness/serve_chat_request.json" \
  > "$OUT/correctness/serve_chat_response.json" 2> "$OUT/correctness/serve_chat_curl.err"
python3 - "$OUT/correctness/serve_chat_response.json" > "$OUT/correctness/serve_chat_validation.txt" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
content = data["choices"][0]["message"].get("content") or ""
usage = data.get("usage") or {}
completion = int(usage.get("completion_tokens") or 0)
if content.strip() != "5" or completion <= 0:
    raise SystemExit(f"SERVE_SMOKE_FAIL content={content!r} completion_tokens={completion}")
print(f"SERVE_SMOKE_PASS content={content.strip()!r} completion_tokens={completion}")
PY

cat > "$OUT/bench/bench_sharegpt_c16.command.txt" <<TXT
target/release/ferrum bench-serve --base-url http://127.0.0.1:$PORT --model $MODEL --tokenizer $MODEL_PATH --dataset sharegpt --sharegpt-path $DATASET --random-output-len 64 --concurrency 16 --num-prompts 16 --n-repeats 3 --fail-on-error --require-ci --seed 9271 --out $OUT/bench/bench_sharegpt_c16.json
TXT
target/release/ferrum bench-serve \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" \
  --tokenizer "$MODEL_PATH" \
  --dataset sharegpt \
  --sharegpt-path "$DATASET" \
  --random-output-len 64 \
  --concurrency 16 \
  --num-prompts 16 \
  --n-repeats 3 \
  --fail-on-error \
  --require-ci \
  --seed 9271 \
  --out "$OUT/bench/bench_sharegpt_c16.json" \
  > "$OUT/bench/bench_sharegpt_c16.stdout" 2> "$OUT/bench/bench_sharegpt_c16.stderr"

grep -aE "\\[unified-prof\\]|\\[batched-op-profile\\]|\\[prefill-profile\\]|\\[unified-decode\\]|graph" "$OUT/server/server.log" > "$OUT/profile/profile_extract.log" || true
grep -aE "FERRUM_UNIFIED_GRAPH|FERRUM_BATCHED_GRAPH|decode_graph_policy|ConfigFile|Cli" "$OUT/server/serve_decision_trace.jsonl" > "$OUT/profile/runtime_decision_trace_extract.log" || true
nvidia-smi > "$OUT/remote/nvidia_smi_after.txt"

python3 - "$OUT" <<'PY' > "$OUT/summary.json"
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
bench = json.load(open(out / "bench/bench_sharegpt_c16.json", "r", encoding="utf-8"))
runtime = json.load(open(out / "server/serve_effective_config.json", "r", encoding="utf-8"))
entries = {e.get("key"): e for e in runtime.get("runtime_config", {}).get("entries", [])}

def stat(name):
    value = bench.get(name)
    return value if isinstance(value, (int, float, str)) else value

throughput = (bench.get("output_throughput_tps") or {}).get("mean")
baseline = 518.7959572662905
summary = {
    "status": "PASS",
    "release_grade": False,
    "lane": "W2 Gemma3-27B CUDA typed unified graph c16 diagnostic",
    "artifact_dir": str(out),
    "git_head": (out / "remote/git_head.txt").read_text().strip(),
    "git_status_short": (out / "remote/git_status_short.txt").read_text().strip(),
    "git_status_source_short": (out / "remote/git_status_source_short.txt").read_text().strip(),
    "binary_sha256": (out / "build/ferrum.sha256").read_text().strip(),
    "run_validation": (out / "correctness/run_validation.txt").read_text().strip(),
    "serve_validation": (out / "correctness/serve_chat_validation.txt").read_text().strip(),
    "runtime_unified_graph": entries.get("FERRUM_UNIFIED_GRAPH", {}),
    "selected_graph_mode": runtime.get("selected_graph_mode"),
    "bench_completed_per_run": bench.get("completed_per_run"),
    "bench_errored_per_run": bench.get("errored_per_run"),
    "output_token_count_source": bench.get("output_token_count_source"),
    "output_throughput_tps_mean": throughput,
    "vllm_c16_tps_baseline_orientation": baseline,
    "ferrum_vs_vllm_ratio_orientation": throughput / baseline if isinstance(throughput, (int, float)) else None,
    "n_repeats": bench.get("n_repeats"),
    "num_prompts": bench.get("num_prompts"),
    "concurrency": bench.get("concurrency"),
    "pass_line": None,
}
json.dump(summary, sys.stdout, indent=2, sort_keys=True)
print()
PY
