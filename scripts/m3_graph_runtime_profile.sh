#!/usr/bin/env bash
#
# Run a low-intrusion M3 c=32 graph-on runtime profile on a GPU pod.
#
# This is not an A/B sweep. It keeps the production graph path enabled and
# avoids FERRUM_DECODE_OP_PROFILE because that flag inserts sync-heavy layer
# timers that distort graph replay. The goal is to split c=32 wall time across:
#   - scheduler/process iteration time
#   - unified model vs decode post-process
#   - decode-only model/post-process
#   - CUDA graph upload/launch/sync
#
# Usage from the repo root on a GPU pod:
#   MODEL_DIR=/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/<rev> \
#   OUT_ROOT=/workspace/m3-graph-runtime-profile \
#   bash scripts/m3_graph_runtime_profile.sh
#
# Optional:
#   BUILD=1 bash scripts/m3_graph_runtime_profile.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
BIN="${BIN:-./target/release/ferrum}"
OUT_ROOT="${OUT_ROOT:-/workspace/m3-graph-runtime-profile-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-18164}"
CONCURRENCY="${CONCURRENCY:-32}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-10}"
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

{
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "repo=$(pwd)"
    echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_status_short_begin"
    git status --short 2>/dev/null || true
    echo "git_status_short_end"
    echo "nvidia_smi_begin"
    nvidia-smi 2>/dev/null || true
    echo "nvidia_smi_end"
    echo "model_dir=$MODEL_DIR"
    echo "hf_model=$HF_MODEL"
    echo "bin=$BIN"
    echo "num_prompts=$NUM_PROMPTS"
    echo "warmup_requests=$WARMUP_REQUESTS"
    echo "concurrency=$CONCURRENCY"
    echo "random_input_len=$RANDOM_INPUT_LEN"
    echo "random_output_len=$RANDOM_OUTPUT_LEN"
} >"$OUT_ROOT/metadata.txt"

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
    FERRUM_MOE_GRAPH=1 \
    FERRUM_VLLM_MOE=1 \
    FERRUM_VLLM_MOE_PAIR_IDS=1 \
    FERRUM_USE_VLLM_PAGED_ATTN=1 \
    FERRUM_PREFIX_CACHE=0 \
    FERRUM_UNIFIED_POST_PROF=1 \
    FERRUM_BATCH_DECODE_PROF=1 \
    FERRUM_RBD_PROF=1 \
    FERRUM_NEXT_BATCH_PROF=1 \
    FERRUM_GRAPH_PROF=1 \
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

grep -nE "unified-prof|iter-prof|rbd-prof|graph-prof|nb-prof|moe-graph|decode_batch_internal" \
    "$SERVER_LOG" >"$OUT_ROOT/profile_snippets.log" || true

if ! grep -q "unified-prof" "$OUT_ROOT/profile_snippets.log"; then
    echo "missing unified-prof output; increase NUM_PROMPTS or RANDOM_OUTPUT_LEN" >&2
    tail -200 "$SERVER_LOG" >&2 || true
    exit 1
fi

if ! grep -q "graph-prof" "$OUT_ROOT/profile_snippets.log"; then
    echo "missing graph-prof output; graph replay may not have been reached" >&2
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
itl = data.get("itl_ms") or {}
ttft = data.get("ttft_ms") or {}
print("BENCH_FILE=", path)
print("OUTPUT_TPS=", throughput.get("mean", data.get("output_throughput")))
print("TPOT_P50=", (tpot.get("p50") or {}).get("mean", data.get("median_tpot_ms")))
print("ITL_P95=", (itl.get("p95") or {}).get("mean"))
print("TTFT_P50=", (ttft.get("p50") or {}).get("mean", data.get("median_ttft_ms")))
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


def med(rows, key):
    vals = [row[key] for row in rows if key in row]
    if not vals:
        return None
    return statistics.median(vals)


groups = {
    "unified_prof": [],
    "iter_prof": [],
    "rbd_prof": [],
    "graph_prof": [],
    "nb_prof": [],
}

for line in text:
    if "unified-prof" in line:
        groups["unified_prof"].append(kvs(line))
    elif "iter-prof" in line:
        groups["iter_prof"].append(kvs(line))
    elif "rbd-prof" in line:
        groups["rbd_prof"].append(kvs(line))
    elif "graph-prof" in line:
        groups["graph_prof"].append(kvs(line))
    elif "nb-prof" in line:
        groups["nb_prof"].append(kvs(line))

summary = {}
for name, rows in groups.items():
    keys = sorted({key for row in rows for key in row})
    summary[name] = {"samples": len(rows)}
    for key in keys:
        value = med(rows, key)
        if value is not None:
            summary[name][f"{key}_median"] = value

unified = summary.get("unified_prof", {})
if unified.get("total_median") and unified.get("model_median") is not None:
    unified["model_share"] = unified["model_median"] / unified["total_median"]
if unified.get("total_median") and unified.get("decode_post_median") is not None:
    unified["decode_post_share"] = unified["decode_post_median"] / unified["total_median"]

rbd = summary.get("rbd_prof", {})
if rbd.get("decode_median") and rbd.get("post_median") is not None:
    rbd["post_vs_decode"] = rbd["post_median"] / rbd["decode_median"]

graph = summary.get("graph_prof", {})
if graph.get("total_median"):
    for key in ("upload", "launch", "sync"):
        value = graph.get(f"{key}_median")
        if value is not None:
            graph[f"{key}_share"] = value / graph["total_median"]

open(out_path, "w").write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print("PROFILE_SUMMARY=", out_path)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "PROFILE_SNIPPETS=$OUT_ROOT/profile_snippets.log"
tail -80 "$OUT_ROOT/profile_snippets.log"
