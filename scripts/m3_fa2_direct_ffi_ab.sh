#!/usr/bin/env bash
#
# Scoped M3 A/B for the opt-in vLLM FlashAttention-2 direct FFI path.
#
# Compares:
#   fa2_direct: FERRUM_FA2_DIRECT_FFI_SHIM points at the C ABI shim
#   fa_layout : existing FA-compatible KV layout with Ferrum's own varlen reader
#
# Usage on a GPU pod from the repo root:
#   FA2_SHIM=/workspace/libferrum_fa2_shim.so \
#   OUT_ROOT=/workspace/m3-fa2-direct-ffi-ab \
#   REPEATS=1 \
#   bash scripts/m3_fa2_direct_ffi_ab.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
BIN="${BIN:-./target/release/ferrum}"
OUT_ROOT="${OUT_ROOT:-/workspace/m3-fa2-direct-ffi-ab-$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-1}"
PORT_BASE="${PORT_BASE:-18480}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-10}"
CONCURRENCY="${CONCURRENCY:-32}"
FA2_SHIM="${FA2_SHIM:-/workspace/libferrum_fa2_shim.so}"
TORCH_LIB="${TORCH_LIB:-/workspace/vllm-venv/lib/python3.12/site-packages/torch/lib}"
FA2_DIR="${FA2_DIR:-/workspace/vllm-venv/lib/python3.12/site-packages/vllm/vllm_flash_attn}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "MODEL_DIR does not exist: $MODEL_DIR" >&2
    exit 1
fi
if [[ ! -r "$FA2_SHIM" ]]; then
    echo "FA2_SHIM is not readable: $FA2_SHIM" >&2
    echo "build it with: bash scripts/microbenches/build_fa2_ferrum_shim.sh" >&2
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
echo "REPEATS=$REPEATS"
echo "FA2_SHIM=$FA2_SHIM"

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
    echo "repeats=$REPEATS"
    echo "num_prompts=$NUM_PROMPTS"
    echo "warmup_requests=$WARMUP_REQUESTS"
    echo "concurrency=$CONCURRENCY"
    echo "fa2_shim=$FA2_SHIM"
    echo "torch_lib=$TORCH_LIB"
    echo "fa2_dir=$FA2_DIR"
} >"$OUT_ROOT/metadata.txt"

BASE_ENV=(
    HF_HOME=/workspace/hf-cache
    LD_LIBRARY_PATH="${TORCH_LIB}:${FA2_DIR}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    FERRUM_BACKEND=cuda
    FERRUM_MOE_DEVICE_ROUTE=1
    FERRUM_MOE_STREAMS=4
    FERRUM_GREEDY_ARGMAX=1
    FERRUM_KV_MAX_BLOCKS=2048
    FERRUM_PAGED_MAX_SEQS=32
    FERRUM_MOE_GRAPH=1
    FERRUM_VLLM_MOE=1
    FERRUM_VLLM_MOE_PAIR_IDS=1
    FERRUM_USE_VLLM_PAGED_ATTN=1
    FERRUM_PREFIX_CACHE=0
)

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
    local port="$1"
    local log_file="$2"
    for _ in $(seq 1 180); do
        curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && return 0
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server exited before health on port $port" >&2
            tail -160 "$log_file" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "server health timeout on port $port" >&2
    tail -160 "$log_file" >&2 || true
    return 1
}

run_case() {
    local name="$1"
    local port="$2"
    shift 2
    local dir="$OUT_ROOT/${name}_c${CONCURRENCY}_n${REPEATS}"
    mkdir -p "$dir"

    echo "=== $name ==="
    env "${BASE_ENV[@]}" "$@" "$BIN" serve "$MODEL_DIR" --port "$port" >"$dir/server.log" 2>&1 &
    SERVER_PID=$!
    wait_health "$port" "$dir/server.log"

    cat >"$dir/paris_req.json" <<JSON
{"model":"$HF_MODEL","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64,"temperature":0.0}
JSON
    curl -fsS -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "@$dir/paris_req.json" >"$dir/paris.json"
    python3 - "$dir/paris.json" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
print("PARIS_CONTENT=", content)
if "Paris" not in content:
    raise SystemExit("Paris gate failed")
PY

    "$BIN" bench-serve \
        --base-url "http://127.0.0.1:${port}" \
        --model "$HF_MODEL" \
        --tokenizer "$MODEL_DIR" \
        --dataset random \
        --random-input-len 256 \
        --random-output-len 128 \
        --num-prompts "$NUM_PROMPTS" \
        --warmup-requests "$WARMUP_REQUESTS" \
        --n-repeats "$REPEATS" \
        --concurrency "$CONCURRENCY" \
        --output json \
        --out "$dir/bench.json" \
        >"$dir/bench.log" 2>&1

    python3 - "$dir/bench.json" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
throughput = data.get("output_throughput_tps") or {}
mean = throughput.get("mean", data.get("output_throughput"))
ci95 = throughput.get("ci95_hw")
tpot = (data.get("tpot_ms") or {}).get("p50", {}).get("mean", data.get("median_tpot_ms"))
itl = (data.get("itl_ms") or {}).get("p95", {}).get("mean")
ttft = (data.get("ttft_ms") or {}).get("p50", {}).get("mean", data.get("median_ttft_ms"))
print(f"RESULT throughput={mean} ci95={ci95} tpot_p50={tpot} itl_p95={itl} ttft_p50={ttft} file={path}")
PY

    cleanup
    SERVER_PID=""
}

run_case fa2_direct "$PORT_BASE" \
    FERRUM_FA_LAYOUT_VARLEN=1 \
    FERRUM_FA2_DIRECT_FFI_SHIM="$FA2_SHIM"
run_case fa_layout "$((PORT_BASE + 1))" \
    FERRUM_FA_LAYOUT_VARLEN=1 \
    FERRUM_FA2_DIRECT_FFI=0

python3 - "$OUT_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = {}
for path in sorted(root.glob("*_c*_n*/bench.json")):
    data = json.load(open(path))
    throughput = data.get("output_throughput_tps") or {}
    name = path.parent.name.split("_c", 1)[0]
    rows[name] = (throughput.get("mean", data.get("output_throughput")), throughput.get("ci95_hw"))
for name in sorted(rows):
    mean, ci95 = rows[name]
    print(f"SUMMARY {name}: throughput={mean} ci95={ci95}")
if rows.get("fa2_direct", (None, None))[0] and rows.get("fa_layout", (None, None))[0]:
    candidate = rows["fa2_direct"][0]
    baseline = rows["fa_layout"][0]
    print(f"DELTA fa2_direct vs fa_layout: {(candidate / baseline - 1.0) * 100.0:.2f}%")
PY
