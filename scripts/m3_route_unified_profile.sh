#!/usr/bin/env bash
#
# Run the scoped M3 c=32 route-shape + unified-engine profile gate.
#
# This is not a throughput confirmation sweep. It exists to answer the next
# bottleneck question before changing kernels:
#   - what is the real decode MoE routing shape at c=32?
#   - how much wall time is model vs unified decode post-process?
#
# Usage on a GPU pod from the repo root:
#   MODEL_DIR=/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/<rev> \
#   OUT_ROOT=/workspace/m3-route-unified-profile \
#   bash scripts/m3_route_unified_profile.sh
#
# Optional:
#   BUILD=1 bash scripts/m3_route_unified_profile.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
BIN="${BIN:-./target/release/ferrum}"
OUT_ROOT="${OUT_ROOT:-/workspace/m3-route-unified-profile-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-18143}"
CONCURRENCY="${CONCURRENCY:-32}"
TOP_K="${TOP_K:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-256}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "MODEL_DIR does not exist: $MODEL_DIR" >&2
    exit 1
fi

if [[ "${BUILD:-0}" == "1" ]]; then
    cargo build --release -p ferrum-cli --features cuda,marlin,vllm-paged-attn-v2,vllm-moe-marlin
fi

if [[ ! -x "$BIN" ]]; then
    echo "ferrum binary not executable: $BIN" >&2
    echo "set BUILD=1 or BIN=/path/to/ferrum" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "MODEL_DIR=$MODEL_DIR"
echo "CONCURRENCY=$CONCURRENCY NUM_PROMPTS=$NUM_PROMPTS RANDOM=${RANDOM_INPUT_LEN}/${RANDOM_OUTPUT_LEN}"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -INT "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_health() {
    local log_file="$1"
    for _ in $(seq 1 180); do
        curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server exited before health on port $PORT" >&2
            tail -200 "$log_file" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "server health timeout on port $PORT" >&2
    tail -200 "$log_file" >&2 || true
    return 1
}

SERVER_LOG="$OUT_ROOT/server.log"
BENCH_LOG="$OUT_ROOT/bench.log"
BENCH_JSON="$OUT_ROOT/bench.json"

env \
    HF_HOME=/workspace/hf-cache \
    FERRUM_BACKEND=cuda \
    FERRUM_MOE_DEVICE_ROUTE=1 \
    FERRUM_MOE_STREAMS=4 \
    FERRUM_GREEDY_ARGMAX=1 \
    FERRUM_KV_MAX_BLOCKS=2048 \
    FERRUM_PAGED_MAX_SEQS=32 \
    FERRUM_MOE_GRAPH=0 \
    FERRUM_VLLM_MOE=1 \
    FERRUM_VLLM_MOE_PAIR_IDS=1 \
    FERRUM_USE_VLLM_PAGED_ATTN=1 \
    FERRUM_VLLM_PAGED_ATTN_V1_SHORT=0 \
    FERRUM_PREFIX_CACHE=0 \
    FERRUM_MOE_DUMP=1 \
    FERRUM_MOE_DUMP_BATCH_X_TOPK="$((CONCURRENCY * TOP_K))" \
    FERRUM_VLLM_MOE_LOG_CONFIG=1 \
    FERRUM_VLLM_MOE_LOG_CONFIG_MIN_PAIRS="$((CONCURRENCY * TOP_K))" \
    FERRUM_VLLM_MOE_LOG_CONFIG_LIMIT="${FERRUM_VLLM_MOE_LOG_CONFIG_LIMIT:-32}" \
    FERRUM_UNIFIED_POST_PROF=1 \
    FERRUM_UNIFIED_LAYER_PROF=1 \
    FERRUM_UNIFIED_LAYER_PROF_EVERY="${FERRUM_UNIFIED_LAYER_PROF_EVERY:-16}" \
    FERRUM_BATCH_DECODE_PROF=1 \
    FERRUM_NEXT_BATCH_PROF=1 \
    FERRUM_MOE_PROFILE=1 \
    "$BIN" serve "$MODEL_DIR" --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_health "$SERVER_LOG"

cat >"$OUT_ROOT/paris_req.json" <<JSON
{"model":"$HF_MODEL","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64,"temperature":0.0}
JSON
curl -fsS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "@$OUT_ROOT/paris_req.json" >"$OUT_ROOT/paris.json"
python3 - "$OUT_ROOT/paris.json" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path))
content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
print("PARIS_CONTENT=", content)
if "Paris" not in content:
    raise SystemExit("Paris gate failed")
PY

"$BIN" bench-serve \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "$HF_MODEL" \
    --tokenizer "$MODEL_DIR" \
    --dataset random \
    --random-input-len "$RANDOM_INPUT_LEN" \
    --random-output-len "$RANDOM_OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --n-repeats 1 \
    --concurrency "$CONCURRENCY" \
    --output json \
    --out "$BENCH_JSON" \
    >"$BENCH_LOG" 2>&1

grep -nE "MOE_DUMP|vllm-moe-config|unified-prof|unified-layer-prof|iter-prof|bg-loop-gap|nb-prof|batched-decode-prof|bucket-prof" \
    "$SERVER_LOG" >"$OUT_ROOT/profile_snippets.log" || true

if ! grep -q "MOE_DUMP" "$OUT_ROOT/profile_snippets.log"; then
    echo "missing MOE_DUMP output; route shape was not captured" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
fi

if ! grep -q "unified-prof" "$OUT_ROOT/profile_snippets.log"; then
    echo "missing unified-prof output; increase NUM_PROMPTS or RANDOM_OUTPUT_LEN" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
fi

if ! grep -q "unified-layer-prof" "$OUT_ROOT/profile_snippets.log"; then
    echo "missing unified-layer-prof output; verify the binary includes FERRUM_UNIFIED_LAYER_PROF" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
fi

if ! grep -q "vllm-moe-config" "$OUT_ROOT/profile_snippets.log"; then
    echo "missing vllm-moe-config output; verify the binary includes the filtered config logger" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
fi

python3 - "$BENCH_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path))
throughput = data.get("output_throughput_tps") or {}
tpot = data.get("tpot_ms") or {}
print("BENCH_FILE=", path)
print("OUTPUT_TPS=", throughput.get("mean", data.get("output_throughput")))
print("TPOT_P50=", (tpot.get("p50") or {}).get("mean", data.get("median_tpot_ms")))
PY

python3 - "$OUT_ROOT/profile_snippets.log" "$OUT_ROOT/profile_summary.json" <<'PY'
import json
import re
import statistics
import sys

snippets_path, out_path = sys.argv[1:3]
text = open(snippets_path).read().splitlines()


def kvs(line):
    out = {}
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(-?\d+(?:\.\d+)?)", line):
        out[key] = float(value) if "." in value else int(value)
    return out


summary = {
    "moe_dump": [],
    "unified_prof": {},
    "unified_layer_prof": {},
    "iter_prof": {},
    "batched_decode_prof": {},
    "bucket_prof": {},
    "vllm_moe_config": {},
}
unified = []
unified_layer = []
iters = []
batched_decode = []
buckets = []
configs = []

for line in text:
    if "MOE_DUMP" in line and "batch_x_topk=" in line:
        summary["moe_dump"].append(kvs(line))
    elif "unified-layer-prof" in line:
        unified_layer.append(kvs(line))
    elif "unified-prof" in line:
        unified.append(kvs(line))
    elif "iter-prof" in line:
        iters.append(kvs(line))
    elif "batched-decode-prof" in line:
        batched_decode.append(kvs(line))
    elif "bucket-prof" in line:
        buckets.append(kvs(line))
    elif "vllm-moe-config" in line:
        configs.append(kvs(line))


def med(rows, key):
    vals = [row[key] for row in rows if key in row]
    if not vals:
        return None
    return statistics.median(vals)


for name, rows, keys in [
    ("unified_prof", unified, ["total", "model", "decode_post", "sample", "sched", "stream", "stop", "complete"]),
    ("unified_layer_prof", unified_layer, ["m", "seqs", "sampled", "layers", "layer_sum", "final_sum", "input_norm", "qkv", "split_cache", "attn", "o_proj", "post_norm", "router", "moe", "residual_add", "final_norm", "sample_gather", "lm_head", "readback"]),
    ("iter_prof", iters, ["total", "sched", "process", "batch_size"]),
    ("batched_decode_prof", batched_decode, ["m", "layers", "total", "dense", "attn_peritem", "moe", "route", "gate", "up", "silu", "down", "wsum", "other"]),
    ("bucket_prof", buckets, ["bk_total", "sync", "d2h", "host_route", "plan", "gather", "gemm1", "silu", "gemm3", "combine"]),
    ("vllm_moe_config", configs, ["batch_x_topk", "prob_m", "prob_n", "prob_k", "block", "top_k", "thread_k", "thread_n", "threads", "blocks_per_sm", "sh_cache", "max_shared"]),
]:
    summary[name]["samples"] = len(rows)
    for key in keys:
        value = med(rows, key)
        if value is not None:
            summary[name][f"{key}_median"] = value

if unified:
    model = summary["unified_prof"].get("model_median")
    total = summary["unified_prof"].get("total_median")
    if model and total:
        summary["unified_prof"]["model_share"] = model / total

if batched_decode:
    total = summary["batched_decode_prof"].get("total_median")
    for key in ["dense", "attn_peritem", "moe", "other"]:
        value = summary["batched_decode_prof"].get(f"{key}_median")
        if value is not None and total:
            summary["batched_decode_prof"][f"{key}_share"] = value / total

if unified_layer:
    layer_sum = summary["unified_layer_prof"].get("layer_sum_median")
    for key in ["input_norm", "qkv", "split_cache", "attn", "o_proj", "post_norm", "router", "moe", "residual_add"]:
        value = summary["unified_layer_prof"].get(f"{key}_median")
        if value is not None and layer_sum:
            summary["unified_layer_prof"][f"{key}_share"] = value / layer_sum

if buckets:
    gemm1 = summary["bucket_prof"].get("gemm1_median")
    gemm3 = summary["bucket_prof"].get("gemm3_median")
    bk_total = summary["bucket_prof"].get("bk_total_median")
    if gemm1 is not None and gemm3 is not None and bk_total:
        # bk_total is logged in ms; phase counters are logged in us.
        summary["bucket_prof"]["gemm13_share"] = (gemm1 + gemm3) / (bk_total * 1000.0)

if configs:
    summary["vllm_moe_config"]["rows"] = configs[:16]

open(out_path, "w").write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print("PROFILE_SUMMARY=", out_path)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "PROFILE_SNIPPETS=$OUT_ROOT/profile_snippets.log"
tail -80 "$OUT_ROOT/profile_snippets.log"
