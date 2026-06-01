#!/usr/bin/env bash
#
# ferrum vs vLLM apples-to-apples bench — PLAYBOOK § 4.B + § 0.6/0.7.
#
# Drives both engines on the same host with identical workload params,
# dumps the effective-config parity table BEFORE running any bench, and
# emits one ferrum BenchReport + one vllm BenchReport per scenario cell.
#
# Phase 0 deliverable: one credible parity-confirmed report under
# `docs/bench/<hw>-<date>-parity/`. The output of this script feeds the
# § 8 decision point (gap < 15% → Phase 1 deferred; > 40% → full Phase 1).
#
# Usage:
#   scripts/bench_vs_vllm.sh <model> <concurrency_sweep_csv> [--request-rate R]
#
# Example:
#   scripts/bench_vs_vllm.sh qwen3:0.6b 1,4,16,32
#   scripts/bench_vs_vllm.sh Qwen/Qwen3-30B-A3B-GPTQ-Int4 32 --request-rate 20

set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,18p' "$0" | sed 's/^# \?//'
    exit 1
fi

MODEL_ALIAS="$1"
SWEEP_CSV="$2"
shift 2
REQUEST_RATE=""
NUM_PROMPTS=100
WARMUP=10
N_REPEATS=3
SLO_TTFT=500
SLO_TPOT=50
SLO_E2E=30000
FERRUM_PORT=8800
VLLM_PORT=8801
OUT_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --request-rate) REQUEST_RATE="$2"; shift 2 ;;
        --num-prompts)  NUM_PROMPTS="$2"; shift 2 ;;
        --n-repeats)    N_REPEATS="$2"; shift 2 ;;
        --warmup)       WARMUP="$2"; shift 2 ;;
        --slo-ttft)     SLO_TTFT="$2"; shift 2 ;;
        --slo-tpot)     SLO_TPOT="$2"; shift 2 ;;
        --slo-e2e)      SLO_E2E="$2"; shift 2 ;;
        --out-dir)      OUT_DIR="$2"; shift 2 ;;
        --ferrum-port)  FERRUM_PORT="$2"; shift 2 ;;
        --vllm-port)    VLLM_PORT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$OUT_DIR" ]; then
    HW_TAG="cuda-rtx4090"
    [ "$(uname -s)" = "Darwin" ] && HW_TAG="metal-$(uname -m)"
    OUT_DIR="docs/bench/${HW_TAG}-$(date +%Y-%m-%d)-parity"
fi
mkdir -p "$OUT_DIR"

# ─────────────────────────────────────────────────────────────────────
# Resolve ferrum binary + vllm HF id
# ─────────────────────────────────────────────────────────────────────

FERRUM_BIN="${FERRUM_BIN:-./target/release/ferrum}"
if [ ! -x "$FERRUM_BIN" ]; then
    echo "ERROR: ferrum binary not found at $FERRUM_BIN" >&2
    echo "  build: cargo build --release -p ferrum-cli --features cuda,vllm-moe-marlin" >&2
    exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: 'vllm' not on PATH. Install in a venv:" >&2
    echo "  pip install vllm==0.20.2" >&2
    exit 1
fi

# alias resolution: ferrum accepts both 'qwen3:0.6b' and full HF IDs.
# For vllm we need the HF ID — translate common aliases.
case "$MODEL_ALIAS" in
    qwen3:0.6b)        VLLM_MODEL="Qwen/Qwen3-0.6B" ;;
    qwen3:4b)          VLLM_MODEL="Qwen/Qwen3-4B" ;;
    qwen3:8b)          VLLM_MODEL="Qwen/Qwen3-8B" ;;
    llama3.1:8b-int4)  VLLM_MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" ;;
    qwen3-moe-30b-int4) VLLM_MODEL="Qwen/Qwen3-30B-A3B-GPTQ-Int4" ;;
    *) VLLM_MODEL="$MODEL_ALIAS" ;;  # caller passed an HF ID directly
esac
VLLM_MODEL="${VLLM_MODEL_OVERRIDE:-$VLLM_MODEL}"
BENCH_MODEL="${BENCH_MODEL:-$VLLM_MODEL}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$VLLM_MODEL}"

# ─────────────────────────────────────────────────────────────────────
# Tokenizer path (needed by ferrum bench-serve --tokenizer)
# ─────────────────────────────────────────────────────────────────────

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -n "${TOKENIZER_PATH_OVERRIDE:-}" ]; then
    TOK_SNAPSHOT="$TOKENIZER_PATH_OVERRIDE"
else
    TOK_REPO_DIR=$(echo "$TOKENIZER_MODEL" | sed 's#/#--#g')
    TOK_DIR=$(find "$HF_HOME/hub" -type d -name "models--${TOK_REPO_DIR}" 2>/dev/null | head -1)
    if [ -z "$TOK_DIR" ]; then
        echo "ERROR: tokenizer not found in $HF_HOME/hub/ for $TOKENIZER_MODEL" >&2
        echo "  run: huggingface-cli download $TOKENIZER_MODEL" >&2
        exit 1
    fi
    TOK_SNAPSHOT=$(find "$TOK_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
fi
if [ -z "$TOK_SNAPSHOT" ] || [ ! -f "$TOK_SNAPSHOT/tokenizer.json" ]; then
    echo "ERROR: tokenizer.json not found under $TOK_SNAPSHOT" >&2
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# vLLM args we'll dump and use (PLAYBOOK § 0.6 parity table)
# ─────────────────────────────────────────────────────────────────────

# Defaults chosen to match ferrum's defaults as closely as possible.
# Override via env var if needed.
VLLM_GPU_MEM_FRAC="${VLLM_GPU_MEM_FRAC:-0.85}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
VLLM_DTYPE="${VLLM_DTYPE:-fp16}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-true}"
# Ferrum's prefix cache defaults OFF (PR #204) — match here for parity.
VLLM_PREFIX_CACHE="${VLLM_PREFIX_CACHE:-false}"
VLLM_TOKENIZER_MODEL="${VLLM_TOKENIZER_MODEL:-}"

VLLM_ARGS=(
    "$VLLM_MODEL"
    --port "$VLLM_PORT"
    --gpu-memory-utilization "$VLLM_GPU_MEM_FRAC"
    --max-num-seqs "$VLLM_MAX_NUM_SEQS"
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
    --dtype "$VLLM_DTYPE"
)
if [ -n "$VLLM_TOKENIZER_MODEL" ]; then
    VLLM_ARGS+=(--tokenizer "$VLLM_TOKENIZER_MODEL")
fi
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    VLLM_EXTRA_ARGS_ARRAY=($VLLM_EXTRA_ARGS)
    VLLM_ARGS+=("${VLLM_EXTRA_ARGS_ARRAY[@]}")
fi
if [ "$VLLM_ENABLE_CHUNKED_PREFILL" = "true" ]; then
    VLLM_ARGS+=(--enable-chunked-prefill)
fi
if [ "$VLLM_PREFIX_CACHE" != "true" ]; then
    VLLM_ARGS+=(--no-enable-prefix-caching)
fi

# Ferrum-side parity knobs — these are env, dumped into the report.
export FERRUM_PREFIX_CACHE="${FERRUM_PREFIX_CACHE:-0}"
export FERRUM_PAGED_MAX_SEQS="${FERRUM_PAGED_MAX_SEQS:-$VLLM_MAX_NUM_SEQS}"

# ─────────────────────────────────────────────────────────────────────
# Emit parity table BEFORE running any bench
# ─────────────────────────────────────────────────────────────────────

PARITY_FILE="$OUT_DIR/parity.md"
{
    echo "# Config parity — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "Model:    $MODEL_ALIAS (vllm sees \`$VLLM_MODEL\`, bench sends \`$BENCH_MODEL\`)"
    echo "Sweep:    $SWEEP_CSV"
    [ -n "$REQUEST_RATE" ] && echo "Mode:     open-loop @ ${REQUEST_RATE} req/s"
    echo "Repeats:  $N_REPEATS"
    echo "Warmup:   $WARMUP"
    echo "Prompts:  $NUM_PROMPTS per cell"
    echo "SLO:      ttft=$SLO_TTFT  tpot=$SLO_TPOT  e2el=$SLO_E2E (ms)"
    echo
    echo "| knob | ferrum | vllm | parity |"
    echo "|---|---|---|---|"
    printf "| gpu_memory_utilization | (computed at load) | %s | ⚠ inspect after run |\n" "$VLLM_GPU_MEM_FRAC"
    printf "| max_seqs | %s | %s | %s |\n" \
        "$FERRUM_PAGED_MAX_SEQS" "$VLLM_MAX_NUM_SEQS" \
        "$([ "$FERRUM_PAGED_MAX_SEQS" = "$VLLM_MAX_NUM_SEQS" ] && echo ✓ || echo ⚠ DIFF)"
    printf "| max_num_batched_tokens | (engine constant) | %s | ⚠ inspect |\n" "$VLLM_MAX_NUM_BATCHED_TOKENS"
    printf "| chunked_prefill | on (Phase 3 path) | %s | %s |\n" \
        "$VLLM_ENABLE_CHUNKED_PREFILL" \
        "$([ "$VLLM_ENABLE_CHUNKED_PREFILL" = "true" ] && echo ✓ || echo ⚠ DIFF)"
    printf "| prefix_caching | %s | %s | %s |\n" \
        "$([ "$FERRUM_PREFIX_CACHE" = "1" ] && echo on || echo off)" \
        "$([ "$VLLM_PREFIX_CACHE" = "true" ] && echo on || echo off)" \
        "$( [ "$FERRUM_PREFIX_CACHE" = "1" ] && [ "$VLLM_PREFIX_CACHE" = "true" ] && echo ✓ \
            || [ "$FERRUM_PREFIX_CACHE" != "1" ] && [ "$VLLM_PREFIX_CACHE" != "true" ] && echo ✓ \
            || echo "⚠ DIFF" )"
    printf "| dtype | (model config) | %s | ⚠ inspect after run |\n" "$VLLM_DTYPE"
    echo
    echo "**Action required**: confirm the ⚠ rows match before trusting any ratio."
} | tee "$PARITY_FILE"

echo
echo "✓ parity → $PARITY_FILE"
echo

# ─────────────────────────────────────────────────────────────────────
# Helpers: start / stop a server, wait for /health
# ─────────────────────────────────────────────────────────────────────

wait_for_health() {
    local url="$1"
    local label="$2"
    local timeout=300
    local start
    start=$(date +%s)
    while ! curl -fsS "$url" >/dev/null 2>&1; do
        if [ $(($(date +%s) - start)) -gt "$timeout" ]; then
            echo "ERROR: $label did not become healthy within ${timeout}s" >&2
            return 1
        fi
        sleep 2
    done
    echo "  $label ready ($(($(date +%s) - start))s)"
}

cleanup() {
    [ -n "${FERRUM_PID:-}" ] && kill "$FERRUM_PID" 2>/dev/null || true
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ─────────────────────────────────────────────────────────────────────
# Run one engine — ferrum then vllm — collect per-cell reports
# ─────────────────────────────────────────────────────────────────────

run_one_engine() {
    local engine="$1"           # ferrum | vllm
    local base_url="$2"
    local report_file="$3"

    local sweep_arg
    if [ -n "$REQUEST_RATE" ]; then
        sweep_arg="--request-rate $REQUEST_RATE"
    else
        sweep_arg="--concurrency-sweep $SWEEP_CSV"
    fi

    # shellcheck disable=SC2086
    timeout 1800 "$FERRUM_BIN" bench-serve \
        --base-url "$base_url" \
        --model "$BENCH_MODEL" \
        --tokenizer "$TOK_SNAPSHOT" \
        --dataset random \
        --random-input-len 256 --random-output-len 128 \
        --num-prompts "$NUM_PROMPTS" \
        --warmup-requests "$WARMUP" \
        --n-repeats "$N_REPEATS" \
        --goodput "ttft:$SLO_TTFT tpot:$SLO_TPOT e2el:$SLO_E2E" \
        --tag "$engine" \
        --output json --out "$report_file" \
        $sweep_arg
}

# ──── ferrum ────
echo "▶ starting ferrum on port $FERRUM_PORT"
"$FERRUM_BIN" serve "$MODEL_ALIAS" --port "$FERRUM_PORT" \
    > "$OUT_DIR/ferrum.log" 2>&1 &
FERRUM_PID=$!
wait_for_health "http://127.0.0.1:$FERRUM_PORT/health" ferrum

echo "▶ bench-serve → ferrum"
run_one_engine ferrum "http://127.0.0.1:$FERRUM_PORT" "$OUT_DIR/ferrum_report.json"

kill "$FERRUM_PID" 2>/dev/null || true
wait "$FERRUM_PID" 2>/dev/null || true
unset FERRUM_PID

# ──── vllm ────
echo "▶ starting vllm on port $VLLM_PORT"
vllm serve "${VLLM_ARGS[@]}" > "$OUT_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
wait_for_health "http://127.0.0.1:$VLLM_PORT/health" vllm

echo "▶ bench-serve → vllm"
run_one_engine vllm "http://127.0.0.1:$VLLM_PORT" "$OUT_DIR/vllm_report.json"

kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
unset VLLM_PID

# ─────────────────────────────────────────────────────────────────────
# Done. Comparison report generation lands in Phase 2.4 — for now,
# point the operator at the two JSONs.
# ─────────────────────────────────────────────────────────────────────

cat <<EOF

✓ parity-confirmed bench complete

  parity table:      $PARITY_FILE
  ferrum report:     $OUT_DIR/ferrum_report.json
  vllm report:       $OUT_DIR/vllm_report.json

Phase 2.4 ships a markdown comparison generator; for now diff the two
JSONs by hand or pipe into jq:

  jq '.output_throughput_tps' $OUT_DIR/{ferrum,vllm}_report.json

PLAYBOOK § 8 decision point:
  - gap (ferrum/vllm) < 0.85 → consider Phase 1 (per-op profiling)
  - gap ≥ 0.85             → defer Phase 1; Phase 3 correctness next
EOF
