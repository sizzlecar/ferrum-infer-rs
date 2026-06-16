#!/usr/bin/env bash
set -euo pipefail

OUT=${OUT:-/workspace/artifacts/w2_unified_argmax_c16_cuda_diag_2026-06-17}
REPO=${REPO:-/workspace/ferrum-infer-rs-run}
BUILD_REPO=${BUILD_REPO:-/workspace/ferrum-infer-rs}
TARGET_DIR=${TARGET_DIR:-/workspace/ferrum-target}
BIN=${BIN:-${TARGET_DIR}/release/ferrum}
MODEL_PATH=${MODEL_PATH:-/workspace/hf-cache/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2}
PORT=${PORT:-18144}

export OUT
export CARGO_TARGET_DIR="$TARGET_DIR"
export HF_HOME=${HF_HOME:-/workspace/hf-cache}

mkdir -p "$OUT"/{meta,build,run_smoke,serve_smoke,c16_profile}

stop_server() {
  local pid_file="$1"
  if [ -f "$pid_file" ]; then
    local pid
    pid=$(cat "$pid_file" || true)
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
      for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" || true
      fi
    fi
  fi
}

cleanup() {
  stop_server "$OUT/server.pid"
}
trap cleanup EXIT

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal \
    >"$OUT/build/rustup_install.log" 2>&1
fi
if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi

cd "$REPO"
git rev-parse HEAD >"$OUT/meta/git_sha.txt"
git status --short --untracked-files=no >"$OUT/meta/git_status_short.txt" || true
git status --short --untracked-files=no -- \
  Cargo.toml Cargo.lock crates scripts ferrum.toml \
  >"$OUT/meta/source_git_status_short.txt" || true
nvidia-smi >"$OUT/meta/nvidia_smi_before.txt" || true
nvcc --version >"$OUT/meta/nvcc_version.txt" 2>&1 || true
rustc --version >"$OUT/meta/rustc_version.txt" 2>&1 || true
cargo --version >"$OUT/meta/cargo_version.txt" 2>&1 || true

prefetch_pid=""
if [ ! -d "$MODEL_PATH" ]; then
  (
    python3 -m pip install -q -U huggingface_hub hf_xet >"$OUT/build/hf_pip_install.log" 2>&1
    HF_HOME="$HF_HOME" HF_XET_HIGH_PERFORMANCE=1 python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("circulus/gemma-3-27b-it-gptq")
PY
  ) >"$OUT/build/model_prefetch.log" 2>&1 &
  prefetch_pid=$!
fi
printf '%s\n' "$MODEL_PATH" >"$OUT/meta/model_path.txt"

cp "$REPO/ferrum.toml" "$OUT/ferrum.base.toml"
cat "$OUT/ferrum.base.toml" >"$OUT/ferrum.toml"
cat >>"$OUT/ferrum.toml" <<'EOF'
batch_decode_prof = true
next_batch_prof = true
unified_post_prof = true
decode_op_profile = true
marlin_profile = true
EOF

if [ ! -x "$BIN" ] || [ "$REPO/Cargo.lock" -nt "$BIN" ] || [ "$REPO/Cargo.toml" -nt "$BIN" ]; then
  cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
    >"$OUT/build/cargo_build_release_cuda.log" 2>&1
else
  echo "reusing existing binary $BIN" >"$OUT/build/cargo_build_release_cuda.log"
fi
if [ -n "$prefetch_pid" ]; then
  wait "$prefetch_pid"
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "model path missing after prefetch: $MODEL_PATH" | tee "$OUT/meta/model_path_missing.txt" >&2
  find "$HF_HOME/hub" -maxdepth 4 -name tokenizer.json -print >"$OUT/meta/tokenizer_candidates.txt" 2>/dev/null || true
  exit 30
fi
sha256sum "$BIN" >"$OUT/meta/ferrum.sha256"

set +e
HF_HOME="$HF_HOME" "$BIN" run gemma3:27b-gptq \
  --backend cuda \
  --max-tokens 8 \
  --temperature 0 \
  --prompt "What is 2+3? Answer with only the number." \
  --effective-config-json "$OUT/run_smoke/effective_config.json" \
  --decision-trace-jsonl "$OUT/run_smoke/decision_trace.jsonl" \
  >"$OUT/run_smoke/run.stdout" 2>"$OUT/run_smoke/run.stderr"
run_rc=$?
set -e
echo "$run_rc" >"$OUT/run_smoke/run.rc"
if [ "$run_rc" -ne 0 ] || ! grep -q '5' "$OUT/run_smoke/run.stdout"; then
  echo "ferrum run smoke failed" >&2
  tail -n 120 "$OUT/run_smoke/run.stderr" >&2 || true
  exit 31
fi

wait_ready() {
  local dir="$1"
  for i in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
      >"$dir/models.json" 2>"$dir/models_curl.err"; then
      echo "$i" >"$dir/ready_at_poll.txt"
      return 0
    fi
    if ! kill -0 "$(cat "$dir/server.pid")" 2>/dev/null; then
      echo "server exited while loading" >&2
      tail -n 180 "$dir/server.stderr" >&2 || true
      return 20
    fi
    sleep 1
  done
  echo "server ready timeout" >&2
  tail -n 180 "$dir/server.stderr" >&2 || true
  return 21
}

start_server() {
  local dir="$1"
  stop_server "$OUT/server.pid"
  nvidia-smi >"$dir/nvidia_smi_before.txt" || true
  HF_HOME="$HF_HOME" "$BIN" serve \
    --model gemma3:27b-gptq \
    --backend cuda \
    --port "$PORT" \
    --kv-capacity 512 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 1024 \
    --effective-config-json "$dir/effective_config.json" \
    --decision-trace-jsonl "$dir/decision_trace.jsonl" \
    --profile-jsonl "$dir/profile.jsonl" \
    --profile-commit-sha "$(cat "$OUT/meta/git_sha.txt")" \
    --profile-model gemma3-27b-gptq \
    --profile-concurrency 16 \
    >"$dir/server.stdout" 2>"$dir/server.stderr" &
  echo $! >"$dir/server.pid"
  cp "$dir/server.pid" "$OUT/server.pid"
  wait_ready "$dir"
}

start_server "$OUT/serve_smoke"

cat >"$OUT/serve_smoke/chat_payload.json" <<'JSON'
{
  "model": "gemma3:27b-gptq",
  "messages": [
    {"role": "user", "content": "What is 2+3? Answer with only the number."}
  ],
  "max_tokens": 8,
  "temperature": 0,
  "stream": true,
  "stream_options": {"include_usage": true}
}
JSON

curl -fsS -N \
  -H 'Content-Type: application/json' \
  -d @"$OUT/serve_smoke/chat_payload.json" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  >"$OUT/serve_smoke/chat_stream.sse" 2>"$OUT/serve_smoke/chat_stream.err"

python3 - <<'PY'
import os
import pathlib
import sys

root = pathlib.Path(os.environ["OUT"])
s = (root / "serve_smoke/chat_stream.sse").read_text(errors="replace")
done = s.count("data: [DONE]")
ok = done == 1 and "5" in s and '"usage"' in s
(root / "serve_smoke/smoke.ok").write_text(str(ok))
print("SERVE_SMOKE_OK", ok)
if not ok:
    print(s[:2000])
    sys.exit(32)
PY

HF_HOME="$HF_HOME" timeout 900 "$BIN" bench-serve \
  --base-url "http://127.0.0.1:${PORT}" \
  --model gemma3:27b-gptq \
  --tokenizer "$MODEL_PATH" \
  --dataset random \
  --random-input-len 64 \
  --random-output-len 16 \
  --concurrency 16 \
  --num-prompts 16 \
  --warmup-requests 4 \
  --n-repeats 1 \
  --fail-on-error \
  --seed 9271 \
  --output json \
  --out "$OUT/c16_profile/bench.json" \
  >"$OUT/c16_profile/bench.stdout" 2>"$OUT/c16_profile/bench.stderr"
echo $? >"$OUT/c16_profile/bench.rc"

nvidia-smi >"$OUT/c16_profile/nvidia_smi_after.txt" || true
grep -nE \
  'first-token-prof|stream-ttft-prof|unified-op-profile|unified-prof|iter-prof|unified-decode|panic|CUDA_ERROR|illegal|OOM|out of memory|error|argmax|readback' \
  "$OUT/serve_smoke/server.stderr" \
  "$OUT/c16_profile/bench.stderr" \
  "$OUT/serve_smoke/server.stdout" \
  >"$OUT/log_scan.txt" 2>/dev/null || true
grep -nE \
  'unified-op-profile|readback|marlin_|panic|CUDA_ERROR|illegal|OOM|out of memory|error' \
  "$OUT/serve_smoke/server.stderr" \
  >"$OUT/profile_log_extract.txt" 2>/dev/null || true

python3 - <<'PY'
import json
import os
import pathlib
import re

root = pathlib.Path(os.environ["OUT"])
bench = json.loads((root / "c16_profile/bench.json").read_text())
if isinstance(bench, list):
    bench = bench[0] if bench else {}
profile = (root / "profile_log_extract.txt").read_text(errors="replace") if (root / "profile_log_extract.txt").exists() else ""
rows = []
for line in profile.splitlines():
    if "[unified-op-profile]" not in line:
        continue
    fields = {}
    for key in ("call", "m_total", "num_seqs", "prefill", "decode", "sampled", "total", "gate_up", "down", "lm_head", "readback", "marlin_kernel", "marlin_gate_up_kernel", "marlin_down_kernel"):
        pattern = r"call#(\d+)" if key == "call" else rf"{key}=([0-9]+)"
        match = re.search(pattern, line)
        if match:
            fields[key] = int(match.group(1))
    fields["line"] = line
    rows.append(fields)
target = None
for row in rows:
    if row.get("num_seqs") == 16 and row.get("prefill", 0) > 0 and row.get("decode", 0) > 0:
        if target is None or row.get("m_total", 0) > target.get("m_total", 0):
            target = row
summary = {
    "bench_rc": int((root / "c16_profile/bench.rc").read_text().strip()),
    "completed_per_run": bench.get("completed_per_run"),
    "errored_per_run": bench.get("errored_per_run"),
    "output_token_count_source": bench.get("output_token_count_source"),
    "request_throughput_rps": (bench.get("request_throughput_rps") or {}).get("mean"),
    "output_throughput_tps": (bench.get("output_throughput_tps") or {}).get("mean"),
    "ttft_p50_ms": ((bench.get("ttft_ms") or {}).get("p50") or {}).get("mean"),
    "ttft_p95_ms": ((bench.get("ttft_ms") or {}).get("p95") or {}).get("mean"),
    "itl_p95_ms": ((bench.get("itl_ms") or {}).get("p95") or {}).get("mean"),
    "unified_profile_rows": rows[-20:],
    "target_mixed_frame": target,
    "previous_target_readback_us": 22039,
    "previous_output_tps": 158.877,
}
if target and isinstance(target.get("readback"), int):
    summary["readback_delta_us_vs_previous"] = target["readback"] - 22039
    summary["readback_ratio_vs_previous"] = target["readback"] / 22039
if isinstance(summary["output_throughput_tps"], (int, float)):
    summary["throughput_delta_tps_vs_previous"] = summary["output_throughput_tps"] - 158.877
    summary["throughput_ratio_vs_previous"] = summary["output_throughput_tps"] / 158.877
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

cleanup
trap - EXIT
