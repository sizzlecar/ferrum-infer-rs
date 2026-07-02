#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ferrum-infer-rs

if [ -f /root/.cargo/env ]; then
  . /root/.cargo/env
fi
export PATH="/root/.cargo/bin:${PATH}"

OUT=/workspace/w2_marlin_nested_profile_2026-06-15
DATASET=/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
PORT=8465

mkdir -p "$OUT" "$OUT/build" "$OUT/remote" "$OUT/server" "$OUT/profile"
date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short_after_source_sync.txt" || true
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"

cat > "$OUT/gpu_contract.md" <<TXT
lane: W2 Gemma3 CUDA dense-Marlin nested profile diagnostic
expected_runtime_cost: 15-35min, hard cap 45min, reused Vast 40826362 1x RTX 4090 at about USD 0.402/hr
stop_condition: start/SSH/CUDA/source sync/build/server readiness first failure, dense Marlin nested profile c16/c32 small sample complete and copied, or 45min cap
correctness_gate: release build plus server readiness plus bench-serve --fail-on-error zero-error diagnostic
performance_command: bench-serve sharegpt natural ASCII, FERRUM_DECODE_OP_PROFILE=1 and FERRUM_MARLIN_PROFILE=1, c16/c32, num_prompts=16, n_repeats=1, random-output-len=64, seed 9271, diagnostic only
TXT

cat > "$OUT/build/release_build.command.txt" <<TXT
CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
TXT
if ! CUDA_COMPUTE_CAP=89 cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source \
  > "$OUT/build/release_build.log" 2>&1; then
  echo FAIL > "$OUT/run.status"
  exit 1
fi
sha256sum target/release/ferrum > "$OUT/build/ferrum.sha256"

pkill -f "target/release/ferrum serve" 2>/dev/null || true
sleep 2

cat > "$OUT/server/serve.command.json" <<JSON
{"env":{"FERRUM_DECODE_OP_PROFILE":"1","FERRUM_MARLIN_PROFILE":"1"},"cmd":["target/release/ferrum","serve","--model","gemma3:27b-gptq","--port","$PORT","--kv-capacity","512","--max-num-seqs","16","--effective-config-json","$OUT/server/serve_effective_config.json","--decision-trace-jsonl","$OUT/server/serve_decision_trace.jsonl"]}
JSON

FERRUM_DECODE_OP_PROFILE=1 FERRUM_MARLIN_PROFILE=1 target/release/ferrum serve \
  --model gemma3:27b-gptq \
  --port "$PORT" \
  --kv-capacity 512 \
  --max-num-seqs 16 \
  --effective-config-json "$OUT/server/serve_effective_config.json" \
  --decision-trace-jsonl "$OUT/server/serve_decision_trace.jsonl" \
  > "$OUT/server/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server/server.pid"

ready=0
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
    > "$OUT/server/models.json" 2> "$OUT/server/models.err"; then
    ready=1
    echo "$i" > "$OUT/server/ready_poll.txt"
    break
  fi
  sleep 2
done

if [ "$ready" != 1 ]; then
  echo FAIL > "$OUT/run.status"
  tail -n 200 "$OUT/server/server.log" > "$OUT/server/server_tail_on_fail.log" || true
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

cat > "$OUT/profile/bench-serve.command.json" <<JSON
{"cmd":["timeout","1800","target/release/ferrum","bench-serve","--base-url","http://127.0.0.1:$PORT","--model","gemma3:27b-gptq","--tokenizer","$TOKENIZER","--dataset","sharegpt","--sharegpt-path","$DATASET","--random-output-len","64","--concurrency-sweep","16,32","--num-prompts","16","--n-repeats","1","--fail-on-error","--seed","9271","--out","$OUT/profile/bench_sharegpt_c16_c32_16x1_marlin_nested_profile.json"]}
JSON

set +e
timeout 1800 target/release/ferrum bench-serve \
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
  --out "$OUT/profile/bench_sharegpt_c16_c32_16x1_marlin_nested_profile.json" \
  > "$OUT/profile/bench-serve.stdout" 2> "$OUT/profile/bench-serve.stderr"
BENCH_RC=$?
set -e
echo "$BENCH_RC" > "$OUT/profile/bench-serve.rc"
nvidia-smi > "$OUT/remote/nvidia_smi_after_bench.txt" || true

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

python3 - <<PY > "$OUT/profile/op_profile_summary.json"
import json, pathlib, re, statistics

out = pathlib.Path("$OUT")
text = (out / "server/server.log").read_text(errors="replace")
pat = re.compile(
    r"\\[batched-op-profile\\] m=(\\d+) total=(\\d+)us  "
    r"matmul=(\\d+)us\\((\\d+)\\) attn=(\\d+)us\\((\\d+)\\) "
    r"qkr=(\\d+)us\\((\\d+)\\) norm=(\\d+)us\\((\\d+)\\) "
    r"other=(\\d+)us\\((\\d+)\\) tail_norm=(\\d+)us\\((\\d+)\\) "
    r"tail_mlp=(\\d+)us\\((\\d+)\\) tail_gate_up=(\\d+)us\\((\\d+)\\) "
    r"tail_down=(\\d+)us\\((\\d+)\\) marlin_ws_zero=(\\d+)us\\((\\d+)\\) "
    r"marlin_kernel=(\\d+)us\\((\\d+)\\) tail_act=(\\d+)us\\((\\d+)\\) "
    r"tail_resid=(\\d+)us\\((\\d+)\\)  unwrapped=(\\d+)us"
)
rows = []
for match in pat.finditer(text):
    rows.append(
        {
            "batch": int(match.group(1)),
            "total_us": int(match.group(2)),
            "matmul_us": int(match.group(3)),
            "matmul_calls": int(match.group(4)),
            "attn_us": int(match.group(5)),
            "attn_calls": int(match.group(6)),
            "qkr_us": int(match.group(7)),
            "qkr_calls": int(match.group(8)),
            "norm_us": int(match.group(9)),
            "norm_calls": int(match.group(10)),
            "other_us": int(match.group(11)),
            "other_calls": int(match.group(12)),
            "tail_norm_us": int(match.group(13)),
            "tail_norm_calls": int(match.group(14)),
            "tail_mlp_us": int(match.group(15)),
            "tail_mlp_calls": int(match.group(16)),
            "tail_gate_up_us": int(match.group(17)),
            "tail_gate_up_calls": int(match.group(18)),
            "tail_down_us": int(match.group(19)),
            "tail_down_calls": int(match.group(20)),
            "marlin_ws_zero_us": int(match.group(21)),
            "marlin_ws_zero_calls": int(match.group(22)),
            "marlin_kernel_us": int(match.group(23)),
            "marlin_kernel_calls": int(match.group(24)),
            "tail_act_us": int(match.group(25)),
            "tail_act_calls": int(match.group(26)),
            "tail_resid_us": int(match.group(27)),
            "tail_resid_calls": int(match.group(28)),
            "unwrapped_us": int(match.group(29)),
        }
    )

summary = {"bench_rc": $BENCH_RC, "rows": len(rows), "by_batch": {}}
keys = [
    "total_us",
    "matmul_us",
    "attn_us",
    "qkr_us",
    "norm_us",
    "other_us",
    "tail_norm_us",
    "tail_mlp_us",
    "tail_gate_up_us",
    "tail_down_us",
    "marlin_ws_zero_us",
    "marlin_kernel_us",
    "tail_act_us",
    "tail_resid_us",
    "unwrapped_us",
]
for batch in sorted({row["batch"] for row in rows}):
    group = [row for row in rows if row["batch"] == batch]
    item = {"count": len(group)}
    for key in keys:
        vals = [row[key] for row in group]
        item[f"{key}_mean"] = statistics.mean(vals)
        item[f"{key}_p95"] = sorted(vals)[int(0.95 * (len(vals) - 1))]
    total = item["total_us_mean"] or 1
    for key in keys:
        if key != "total_us":
            item[f"{key.removesuffix('_us')}_share_mean"] = item[f"{key}_mean"] / total
    summary["by_batch"][str(batch)] = item

print(json.dumps(summary, indent=2, sort_keys=True))
PY

if [ "$BENCH_RC" -eq 0 ]; then
  echo PASS > "$OUT/run.status"
else
  echo FAIL > "$OUT/run.status"
fi
date -u +%FT%TZ > "$OUT/remote/end_utc.txt"
exit "$BENCH_RC"
