#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ferrum-infer-rs

OUT=/workspace/w2_decode_op_profile_2026-06-15
DATASET=/workspace/w2_natural_prompt_baseline_probe_2026-06-15/dataset/ascii_sharegpt.jsonl
TOKENIZER=/root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/70d89a3a6b401b5f56558cb5d4c0f1fd158980b2
PORT=8462

mkdir -p "$OUT" "$OUT/remote" "$OUT/server" "$OUT/profile"
date -u +%FT%TZ > "$OUT/remote/start_utc.txt"
git rev-parse HEAD > "$OUT/remote/git_head.txt"
git status --short > "$OUT/remote/git_status_short.txt" || true
sha256sum target/release/ferrum > "$OUT/remote/ferrum.sha256"
nvidia-smi > "$OUT/remote/nvidia_smi_before.txt"

cat > "$OUT/gpu_contract.md" <<TXT
lane: W2 Gemma3 CUDA decode-op-profile diagnostic
expected_runtime_cost: 10-25min, hard cap 40min, reused Vast 40826362 1x RTX 4090 at about USD 0.402/hr
stop_condition: startup/SSH/CUDA/server readiness first failure, profile c16/c32 small sample complete and copied, or 40min cap
correctness_gate: server readiness plus bench-serve --fail-on-error zero-error diagnostic
performance_command: bench-serve sharegpt natural ASCII, c16/c32, num_prompts=16, n_repeats=1, seed 9271, diagnostic only
TXT

pkill -f "target/release/ferrum serve" 2>/dev/null || true
sleep 2

cat > "$OUT/server/serve.command.json" <<JSON
{"env":{"FERRUM_DECODE_OP_PROFILE":"1"},"cmd":["target/release/ferrum","serve","--model","gemma3:27b-gptq","--port","$PORT","--kv-capacity","512","--max-num-seqs","16","--effective-config-json","$OUT/server/serve_effective_config.json","--decision-trace-jsonl","$OUT/server/serve_decision_trace.jsonl"]}
JSON

FERRUM_DECODE_OP_PROFILE=1 target/release/ferrum serve \
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
{"cmd":["timeout","1800","target/release/ferrum","bench-serve","--base-url","http://127.0.0.1:$PORT","--model","gemma3:27b-gptq","--tokenizer","$TOKENIZER","--dataset","sharegpt","--sharegpt-path","$DATASET","--random-output-len","64","--concurrency-sweep","16,32","--num-prompts","16","--n-repeats","1","--fail-on-error","--seed","9271","--out","$OUT/profile/bench_sharegpt_c16_c32_16x1_profile.json"]}
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
  --out "$OUT/profile/bench_sharegpt_c16_c32_16x1_profile.json" \
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
    r"other=(\\d+)us\\((\\d+)\\)  unwrapped=(\\d+)us"
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
            "unwrapped_us": int(match.group(13)),
        }
    )

summary = {"bench_rc": $BENCH_RC, "rows": len(rows), "by_batch": {}}
for batch in sorted({row["batch"] for row in rows}):
    group = [row for row in rows if row["batch"] == batch]
    item = {"count": len(group)}
    for key in [
        "total_us",
        "matmul_us",
        "attn_us",
        "qkr_us",
        "norm_us",
        "other_us",
        "unwrapped_us",
    ]:
        vals = [row[key] for row in group]
        item[f"{key}_mean"] = statistics.mean(vals)
        item[f"{key}_p95"] = sorted(vals)[int(0.95 * (len(vals) - 1))]
    total = item["total_us_mean"] or 1
    for key in ["matmul_us", "attn_us", "qkr_us", "norm_us", "other_us", "unwrapped_us"]:
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
