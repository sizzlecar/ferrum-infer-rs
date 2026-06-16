#!/usr/bin/env bash
set -euo pipefail

SRC="${SRC:-/workspace/ferrum-active-chunk-8bc7cf08}"
ART="${ART:-/workspace/w2_active_decode_prefill_chunk_c16_diag_2026-06-16}"
MODEL="${MODEL:-gemma3:27b-gptq}"
TOKENIZER="${TOKENIZER:-/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2}"
DATASET="${DATASET:-/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl}"
BIN="$SRC/target/release/ferrum"

mkdir -p "$ART"/{bench,build,correctness,profile,remote,run_cwd,server,smoke}
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$ART/remote/start_utc.txt"
nvidia-smi > "$ART/remote/nvidia_smi_before.txt"
nvcc --version > "$ART/remote/nvcc_version.txt"

cd "$SRC"
if [ -f "$HOME/.cargo/env" ]; then
  # Vast startup shells are not always login shells; keep this explicit.
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi
git rev-parse HEAD > "$ART/remote/git_head.txt" 2>/dev/null || true
git status --short > "$ART/remote/git_status_short.txt" 2>/dev/null || true

cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$ART/build/cargo_build.stdout" \
  2> "$ART/build/cargo_build.stderr"
printf "0\n" > "$ART/build/cargo_build.rc"
sha256sum "$BIN" > "$ART/build/ferrum.sha256"
"$BIN" --version > "$ART/build/ferrum_version.txt" 2>&1 || true

timeout 1200 "$BIN" run "$MODEL" \
  --backend cuda \
  --prompt "What is 2+3? Answer with only the number." \
  --max-tokens 8 \
  --temperature 0 \
  --kv-capacity 2560 \
  --max-num-seqs 2 \
  --output-format jsonl \
  > "$ART/correctness/run.stdout" \
  2> "$ART/correctness/run.stderr"
printf "0\n" > "$ART/correctness/run.rc"
python3 - "$ART/correctness/run.stdout" > "$ART/correctness/run_validation.json" <<'PY'
import json
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
assistant = None
for line in text.splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if event.get("event") == "assistant":
        assistant = event
if assistant is None:
    raise SystemExit("missing assistant event")
content = str(assistant.get("content", ""))
bad = any(marker in content for marker in ["<unk>", "[PAD]", "\ufffd"])
ok = ("5" in content) and not bad
print(json.dumps({"pass": ok, "content": content, "bad_output": bad}, sort_keys=True))
if not ok:
    raise SystemExit("run validation failed")
PY

cp "$SRC/ferrum.toml" "$ART/run_cwd/ferrum.toml"
cat >> "$ART/run_cwd/ferrum.toml" <<'EOF'
batch_decode_prof = true
batch_prefill_prof = true
decode_op_profile = true
prefill_op_profile = true
unified_post_prof = true
next_batch_prof = true
EOF

wait_ready() {
  local port="$1"
  local out="$2"
  : > "$out"
  for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" > "${out}.models.json" 2> "${out}.models.err"; then
      echo "$i" > "${out}.ready_poll_count"
      return 0
    fi
    sleep 2
  done
  return 1
}

run_arm() {
  local name="$1"
  local port="$2"
  shift 2
  local server_dir="$ART/server/$name"
  local smoke_dir="$ART/smoke/$name"
  local bench_dir="$ART/bench/$name"
  local profile_dir="$ART/profile/$name"
  mkdir -p "$server_dir" "$smoke_dir" "$bench_dir" "$profile_dir"

  (
    cd "$ART/run_cwd"
    echo "$BIN serve --model $MODEL --backend cuda --host 127.0.0.1 --port $port --max-num-seqs 16 --max-num-batched-tokens 2048 --kv-capacity 512 $* --effective-config-json $server_dir/serve_effective_config.json --decision-trace-jsonl $server_dir/serve_decision_trace.jsonl" \
      > "$server_dir/serve.command.txt"
    "$BIN" serve \
      --model "$MODEL" \
      --backend cuda \
      --host 127.0.0.1 \
      --port "$port" \
      --max-num-seqs 16 \
      --max-num-batched-tokens 2048 \
      --kv-capacity 512 \
      "$@" \
      --effective-config-json "$server_dir/serve_effective_config.json" \
      --decision-trace-jsonl "$server_dir/serve_decision_trace.jsonl" \
      > "$server_dir/server.stdout" \
      2> "$server_dir/server.log" &
    echo $! > "$server_dir/server.pid"
  )

  local pid
  pid="$(cat "$server_dir/server.pid")"
  trap 'kill "$pid" >/dev/null 2>&1 || true' RETURN
  wait_ready "$port" "$server_dir/ready" || {
    ps -fp "$pid" > "$server_dir/server_ps_after_ready_fail.txt" 2>&1 || true
    return 1
  }

  cat > "$smoke_dir/chat_request.json" <<'JSON'
{"model":"gemma3:27b-gptq","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"temperature":0,"max_tokens":8,"stream":false}
JSON
  curl -fsS "http://127.0.0.1:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary @"$smoke_dir/chat_request.json" \
    > "$smoke_dir/chat_response.json" \
    2> "$smoke_dir/chat_curl.err"
  python3 - "$smoke_dir/chat_response.json" > "$smoke_dir/chat_validation.json" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
content = data["choices"][0]["message"].get("content") or ""
bad = any(marker in content for marker in ["<unk>", "[PAD]", "\ufffd"])
ok = ("5" in content) and not bad
print(json.dumps({"pass": ok, "content": content, "bad_output": bad}, sort_keys=True))
if not ok:
    raise SystemExit("serve smoke validation failed")
PY

  cat > "$bench_dir/bench-serve.command.json" <<JSON
{"cmd":["timeout","2400","$BIN","bench-serve","--base-url","http://127.0.0.1:$port","--model","$MODEL","--tokenizer","$TOKENIZER","--dataset","sharegpt","--sharegpt-path","$DATASET","--random-output-len","128","--concurrency-sweep","16","--num-prompts","64","--n-repeats","1","--fail-on-error","--seed","9271","--out","$bench_dir/bench_sharegpt_c16.json"]}
JSON
  timeout 2400 "$BIN" bench-serve \
    --base-url "http://127.0.0.1:$port" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --dataset sharegpt \
    --sharegpt-path "$DATASET" \
    --random-output-len 128 \
    --concurrency-sweep 16 \
    --num-prompts 64 \
    --n-repeats 1 \
    --fail-on-error \
    --seed 9271 \
    --out "$bench_dir/bench_sharegpt_c16.json" \
    > "$bench_dir/bench-serve.stdout" \
    2> "$bench_dir/bench-serve.stderr"
  printf "0\n" > "$bench_dir/bench-serve.rc"

  rg -n "\\[unified-prof\\]|\\[batched-op-profile\\]|\\[prefill-profile\\]|\\[nb-prof\\]" \
    "$server_dir/server.log" > "$profile_dir/profile_extract.log" || true
  nvidia-smi > "$server_dir/nvidia_smi_after_bench.txt"
  ps -fp "$pid" > "$server_dir/server_ps_before_stop.txt" 2>&1 || true
  kill "$pid" >/dev/null 2>&1 || true
  wait "$pid" 2>/dev/null || true
  ps -fp "$pid" > "$server_dir/server_ps_after_stop.txt" 2>&1 || true
  trap - RETURN
}

run_arm default 18141
run_arm chunk32 18142 --scheduler-active-decode-prefill-chunk 32

nvidia-smi > "$ART/remote/nvidia_smi_after.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$ART/remote/end_utc.txt"
printf "0\n" > "$ART/run_remote.rc"
