#!/usr/bin/env bash
#
# Bottleneck localization on a specific (model, concurrency) cell.
#
# Workflow (PLAYBOOK § 5 escalation path):
#   1. ferrum bench-serve baseline (no profiling overhead)
#   2. ferrum bench-serve again with FERRUM_DECODE_OP_PROFILE=1 to
#      capture per-op breakdown via the migrated BackendTimer probes
#      (Phase 1.1 + 1.2 — CUDA-event-accurate, no Instant::now() lies)
#   3. vLLM bench-serve at the same cell for parity comparison
#   4. Parse ferrum's per-op stderr → markdown breakdown
#
# Output:
#   docs/bench/bottleneck-<date>-<model>/
#     ferrum_baseline.json     ferrum bench-serve (n_repeats=N)
#     ferrum_profile.log       FERRUM_DECODE_OP_PROFILE stderr capture
#     ferrum_profile.json      bench-serve metrics during profile run
#     vllm_baseline.json       vLLM bench-serve same cell
#     parity.md                config-parity table (from bench_vs_vllm.sh)
#     breakdown.md             rendered per-op % table
#
# Usage:
#   bash scripts/bottleneck_localize.sh llama3.1:8b-int4 32
#   bash scripts/bottleneck_localize.sh qwen3-moe-30b-int4 32

set -euo pipefail

MODEL_ALIAS="${1:-llama3.1:8b-int4}"
CONCURRENCY="${2:-32}"
N_REPEATS="${N_REPEATS:-3}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
WARMUP="${WARMUP:-10}"

# Resolve model alias → HF id (matches bench_vs_vllm.sh table).
case "$MODEL_ALIAS" in
    qwen3:0.6b)        HF_MODEL="Qwen/Qwen3-0.6B" ;;
    qwen3:4b)          HF_MODEL="Qwen/Qwen3-4B" ;;
    qwen3:8b)          HF_MODEL="Qwen/Qwen3-8B" ;;
    llama3.1:8b-int4)  HF_MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" ;;
    qwen3-moe-30b-int4) HF_MODEL="Qwen/Qwen3-30B-A3B-GPTQ-Int4" ;;
    *)                 HF_MODEL="$MODEL_ALIAS" ;;
esac

SAFE_TAG=$(echo "$MODEL_ALIAS" | tr '/:' '__')
OUT_DIR="docs/bench/bottleneck-$(date +%Y-%m-%d-%H%M)-${SAFE_TAG}"
mkdir -p "$OUT_DIR"

FERRUM_BIN="${FERRUM_BIN:-./target/release/ferrum}"
FERRUM_PORT="${FERRUM_PORT:-8800}"
VLLM_PORT="${VLLM_PORT:-8801}"

# ─── Required tooling check ──────────────────────────────────────────
if [ ! -x "$FERRUM_BIN" ]; then
    echo "ERROR: ferrum binary missing at $FERRUM_BIN" >&2
    exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: vllm not on PATH (activate venv?)" >&2
    exit 1
fi

# ─── HF model + tokenizer locate ─────────────────────────────────────
echo "▶ Ensuring $HF_MODEL is in HF cache"
# huggingface-cli is deprecated in hf-hub 1.16+; use `hf download` if
# available, fall back to old name otherwise.
if command -v hf >/dev/null 2>&1; then
    hf download "$HF_MODEL" >/dev/null 2>&1 || true
else
    huggingface-cli download "$HF_MODEL" --quiet
fi

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# HF cache layout: models--<org>--<repo>/snapshots/<hash>/ — repo path
# slashes become "--" (double-dash), so sed not tr.
HF_DIR_NAME="models--$(echo "$HF_MODEL" | sed 's,/,--,g')"
TOK_DIR=$(find "$HF_HOME/hub" -maxdepth 1 -type d -name "$HF_DIR_NAME" 2>/dev/null | head -1)
if [ -z "$TOK_DIR" ]; then
    echo "ERROR: tokenizer dir not found for $HF_MODEL (expected $HF_HOME/hub/$HF_DIR_NAME)" >&2
    echo "Available dirs:" >&2
    ls "$HF_HOME/hub" 2>&1 | head -10 >&2
    exit 1
fi
TOK_SNAP=$(find "$TOK_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)
if [ -z "$TOK_SNAP" ] || [ ! -f "$TOK_SNAP/tokenizer.json" ]; then
    echo "ERROR: tokenizer.json not found at $TOK_SNAP" >&2
    exit 1
fi
echo "  tokenizer: $TOK_SNAP"

# ─── GPU lock (CUDA only; no-op on Mac) ──────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "▶ Locking GPU"
    bash scripts/lock_gpu.sh || true
fi

cleanup() {
    pkill -f "$FERRUM_BIN serve" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    if command -v nvidia-smi >/dev/null 2>&1; then
        bash scripts/unlock_gpu.sh 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_health() {
    local url="$1"
    local label="$2"
    for i in $(seq 1 120); do
        curl -fsS "$url" >/dev/null 2>&1 && echo "  $label ready" && return
        sleep 2
    done
    echo "ERROR: $label not healthy after 240s" >&2
    return 1
}

# ─────────────────────────────────────────────────────────────────────
# Step 1 — ferrum baseline (no profile env, clean throughput)
# ─────────────────────────────────────────────────────────────────────
echo
echo "▶ STEP 1: ferrum baseline at c=$CONCURRENCY"
"$FERRUM_BIN" serve "$MODEL_ALIAS" --port "$FERRUM_PORT" \
    > "$OUT_DIR/ferrum_baseline.log" 2>&1 &
FERRUM_PID=$!
wait_for_health "http://127.0.0.1:$FERRUM_PORT/health" "ferrum baseline"

"$FERRUM_BIN" bench-serve \
    --base-url "http://127.0.0.1:$FERRUM_PORT" \
    --model "$HF_MODEL" --tokenizer "$TOK_SNAP" \
    --dataset random --random-input-len 256 --random-output-len 128 \
    --num-prompts "$NUM_PROMPTS" --warmup-requests "$WARMUP" \
    --n-repeats "$N_REPEATS" --concurrency "$CONCURRENCY" \
    --output json --out "$OUT_DIR/ferrum_baseline.json" \
    --tag "baseline"

kill "$FERRUM_PID" 2>/dev/null || true
wait "$FERRUM_PID" 2>/dev/null || true
sleep 2

# ─────────────────────────────────────────────────────────────────────
# Step 2 — ferrum with per-op profile (the new BackendTimer probes)
# ─────────────────────────────────────────────────────────────────────
echo
echo "▶ STEP 2: ferrum with FERRUM_DECODE_OP_PROFILE=1"
FERRUM_DECODE_OP_PROFILE=1 "$FERRUM_BIN" serve "$MODEL_ALIAS" --port "$FERRUM_PORT" \
    > "$OUT_DIR/ferrum_profile.log" 2>&1 &
FERRUM_PID=$!
wait_for_health "http://127.0.0.1:$FERRUM_PORT/health" "ferrum profile"

# Short bench — profile output is per-decode-token, so 50 prompts × 128
# output tokens × c=32 already gives ~200 profile lines (plenty).
"$FERRUM_BIN" bench-serve \
    --base-url "http://127.0.0.1:$FERRUM_PORT" \
    --model "$HF_MODEL" --tokenizer "$TOK_SNAP" \
    --dataset random --random-input-len 256 --random-output-len 128 \
    --num-prompts "$NUM_PROMPTS" --warmup-requests "$WARMUP" \
    --n-repeats 1 --concurrency "$CONCURRENCY" \
    --output json --out "$OUT_DIR/ferrum_profile.json" \
    --tag "profile"

kill "$FERRUM_PID" 2>/dev/null || true
wait "$FERRUM_PID" 2>/dev/null || true
sleep 2

# ─────────────────────────────────────────────────────────────────────
# Step 3 — vLLM at same cell for parity
# ─────────────────────────────────────────────────────────────────────
echo
echo "▶ STEP 3: vLLM at c=$CONCURRENCY"
# vLLM args match bench_vs_vllm.sh defaults (apples-to-apples).
VLLM_ARGS=(
    "$HF_MODEL"
    --port "$VLLM_PORT"
    --gpu-memory-utilization 0.85
    --max-num-seqs 32
    --max-model-len 4096
    --dtype float16
    --no-enable-prefix-caching
    --enable-chunked-prefill
)
# INT4 models need --quantization gptq_marlin
if [[ "$HF_MODEL" == *INT4* ]] || [[ "$HF_MODEL" == *Int4* ]] || [[ "$HF_MODEL" == *GPTQ* ]]; then
    VLLM_ARGS+=(--quantization gptq_marlin)
fi

vllm serve "${VLLM_ARGS[@]}" > "$OUT_DIR/vllm_baseline.log" 2>&1 &
VLLM_PID=$!
wait_for_health "http://127.0.0.1:$VLLM_PORT/health" "vllm"

"$FERRUM_BIN" bench-serve \
    --base-url "http://127.0.0.1:$VLLM_PORT" \
    --model "$HF_MODEL" --tokenizer "$TOK_SNAP" \
    --dataset random --random-input-len 256 --random-output-len 128 \
    --num-prompts "$NUM_PROMPTS" --warmup-requests "$WARMUP" \
    --n-repeats "$N_REPEATS" --concurrency "$CONCURRENCY" \
    --output json --out "$OUT_DIR/vllm_baseline.json" \
    --tag "vllm"

kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────
# Step 4 — parse + render breakdown
# ─────────────────────────────────────────────────────────────────────
echo
echo "▶ STEP 4: parse per-op breakdown"
python3 scripts/parse_decode_profile.py \
    "$OUT_DIR/ferrum_profile.log" \
    "$OUT_DIR/ferrum_baseline.json" \
    "$OUT_DIR/vllm_baseline.json" \
    > "$OUT_DIR/breakdown.md"

cat <<EOF

✓ Bottleneck localization complete

  results:    $OUT_DIR/
    ferrum_baseline.json   apples-to-apples ferrum throughput
    vllm_baseline.json     apples-to-apples vLLM throughput
    ferrum_profile.log     per-op breakdown (FERRUM_DECODE_OP_PROFILE=1)
    breakdown.md           rendered markdown table — which op dominates

Open breakdown.md to see which op category is the bottleneck.
EOF
