#!/usr/bin/env bash
#
# 4-cell bottleneck sweep — captures bench data + chrome trace + nsys
# kernel profile for c=1, c=4, c=16, c=32 on a single model.
#
# Produces docs/bench/m3-bottleneck-<date>/{c1,c4,c16,c32}/:
#   ferrum_baseline.json   ferrum bench-serve metrics
#   vllm_baseline.json     vLLM bench-serve metrics (apples-to-apples)
#   ferrum_trace.json      chrome trace from BackendTimer probes
#   ferrum_nsys.csv        nsys kernel-level breakdown (ground truth)
#   bench_serve.log        raw stderr
#
# Usage:
#   bash scripts/sweep_bottleneck.sh <model> [c1,c2,c3,...]
#   bash scripts/sweep_bottleneck.sh qwen3-moe-30b-int4
#   bash scripts/sweep_bottleneck.sh qwen3-moe-30b-int4 1,4,16,32

set -euo pipefail

MODEL_ALIAS="${1:-qwen3-moe-30b-int4}"
SWEEP_CSV="${2:-1,4,16,32}"
N_REPEATS="${N_REPEATS:-1}"  # for trace + nsys captures n=1 is fine
NUM_PROMPTS="${NUM_PROMPTS:-30}"
WARMUP="${WARMUP:-5}"

# Resolve alias → HF id (matches bench_vs_vllm.sh)
case "$MODEL_ALIAS" in
    qwen3:0.6b)        HF_MODEL="Qwen/Qwen3-0.6B" ;;
    qwen3:4b)          HF_MODEL="Qwen/Qwen3-4B" ;;
    qwen3:8b)          HF_MODEL="Qwen/Qwen3-8B" ;;
    llama3.1:8b-int4)  HF_MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" ;;
    qwen3-moe-30b-int4) HF_MODEL="Qwen/Qwen3-30B-A3B-GPTQ-Int4" ;;
    *)                 HF_MODEL="$MODEL_ALIAS" ;;
esac

SAFE=$(echo "$MODEL_ALIAS" | tr '/:' '__')
OUT_ROOT="docs/bench/sweep-$(date +%Y-%m-%d-%H%M)-${SAFE}"
mkdir -p "$OUT_ROOT"

FERRUM_BIN="${FERRUM_BIN:-./target/release/ferrum}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Locate tokenizer dir
HF_DIR_NAME="models--$(echo "$HF_MODEL" | sed 's,/,--,g')"
TOK_DIR=$(find "$HF_HOME/hub" -maxdepth 1 -type d -name "$HF_DIR_NAME" | head -1)
TOK_SNAP=$(find "$TOK_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)
[ -f "$TOK_SNAP/tokenizer.json" ] || { echo "tokenizer.json missing at $TOK_SNAP" >&2; exit 1; }

# Lock GPU (idempotent — bench_vs_vllm.sh path)
if command -v nvidia-smi >/dev/null 2>&1; then
    bash scripts/lock_gpu.sh 2>&1 | tail -3 || true
fi

cleanup() {
    pkill -f "$FERRUM_BIN serve" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Standard iter-3 env knobs (best non-marlin baseline).
# 2026-05-27 cleanup after explicit per-knob bisect on RTX 4090 c=32:
#   - FERRUM_GRAPH=1: no-op placebo (only set by scripts; not read by
#     any source file). Real graph capture is FERRUM_MOE_GRAPH=1 which
#     the autosizer (gpu_mem_autosize.rs:282) now sets by default.
#   - FERRUM_GRAPH_SKIP_UPLOAD=1: measured -3.8% at c=32 vs default
#     (per-replay re-upload). graph.rs:156 comment said the re-upload
#     is required for c=4 perf; turns out it helps c=32 too.
#   - FERRUM_USE_VLLM_PAGED_ATTN=1: REGRESSED post-2026-05-27 build —
#     produces garbage on the non-split-K varlen path (separate from
#     the moe_align fix). Hold OFF until fixed; see session report.
export FERRUM_MOE_DEVICE_ROUTE=1
export FERRUM_MOE_STREAMS=4
export FERRUM_GREEDY_ARGMAX=1
export FERRUM_KV_MAX_BLOCKS=2048
export FERRUM_PAGED_MAX_SEQS=32
# DECODE_OP_PROFILE inflates cuStreamSynchronize count (~4 stage_end ×
# 48 layers × per-iter eager work) — cost ~10% c=32 in graph-on
# regime, ~16% in eager regime. Off by default; opt in for per-stage
# host timer breakdown. Verified 2026-05-27 (iter-1 A/B on RTX 4090).
[ "${FERRUM_PROFILE_STAGES:-0}" = "1" ] && export FERRUM_DECODE_OP_PROFILE=1

# Per-cell sweep
IFS=',' read -ra CELLS <<< "$SWEEP_CSV"
for c in "${CELLS[@]}"; do
    cell_dir="$OUT_ROOT/c${c}"
    mkdir -p "$cell_dir"
    echo
    echo "═══════════════════════════════════════════════════════════════"
    echo " c=$c  →  $cell_dir"
    echo "═══════════════════════════════════════════════════════════════"

    # ── ferrum: chrome trace via FERRUM_TRACE_OUT + bench in one shot ──
    export FERRUM_TRACE_OUT="$(pwd)/$cell_dir/ferrum_trace.json"
    rm -f "$FERRUM_TRACE_OUT"

    FPORT=$((18000 + c))
    # nsys wraps ferrum serve (the process with GPU work) — wrapping bench-serve
    # (HTTP client, no CUDA) was a bug: nsys-rep contained zero kernel data.
    # Skip model-load region via --capture-range=cudaProfilerApi when ferrum
    # supports cudaProfilerStart/Stop; until then we just trim model-load µs
    # post-hoc with nsys stats --start-time.
    if [ "$c" = "32" ] && command -v nsys >/dev/null 2>&1; then
        nsys profile -o "$(pwd)/$cell_dir/ferrum_nsys" \
            --trace=cuda,nvtx,cublas --force-overwrite=true --sample=none \
            "$FERRUM_BIN" serve "$HF_MODEL" --port "$FPORT" \
            > "$cell_dir/ferrum_serve.log" 2>&1 &
        FPID=$!
    else
        "$FERRUM_BIN" serve "$HF_MODEL" --port "$FPORT" \
            > "$cell_dir/ferrum_serve.log" 2>&1 &
        FPID=$!
    fi
    for i in $(seq 1 90); do
        curl -fsS "http://127.0.0.1:$FPORT/health" >/dev/null 2>&1 && break
        sleep 2
    done

    "$FERRUM_BIN" bench-serve \
        --base-url "http://127.0.0.1:$FPORT" \
        --model "$HF_MODEL" --tokenizer "$TOK_SNAP" \
        --dataset random --random-input-len 256 --random-output-len 128 \
        --num-prompts "$NUM_PROMPTS" --warmup-requests "$WARMUP" \
        --n-repeats "$N_REPEATS" --concurrency "$c" \
        --output json --out "$cell_dir/ferrum_baseline.json" 2>&1 \
        | tail -6
    # SIGINT (not SIGTERM) so ferrum serve's tokio ctrl_c handler fires,
    # which is what flushes the global TraceWriter buffered events.
    # When under nsys, the wrapper process is FPID — nsys forwards SIGINT.
    kill -INT "$FPID" 2>/dev/null || true
    # Wait up to 30s for graceful flush (nsys post-processing takes longer)
    for _ in $(seq 1 60); do kill -0 "$FPID" 2>/dev/null || break; sleep 0.5; done
    kill -KILL "$FPID" 2>/dev/null || true
    wait "$FPID" 2>/dev/null || true
    sleep 2

    # Extract nsys kernel summary CSV (c=32 only)
    if [ "$c" = "32" ] && [ -f "$cell_dir/ferrum_nsys.nsys-rep" ]; then
        nsys stats "$cell_dir/ferrum_nsys.nsys-rep" \
            --force-export=true --report cuda_gpu_kern_sum --format csv 2>&1 \
            | grep -E "^[0-9]" > "$cell_dir/ferrum_nsys.csv" || true
    fi

    # ── vLLM bench (no env vars; clean apples-to-apples) ──
    # Skip when SKIP_VLLM=1 (e.g. pod has no vllm install, or only ferrum
    # numbers needed). Cell still produces ferrum_baseline.json + trace.
    if [ "${SKIP_VLLM:-0}" = "1" ] || ! command -v vllm >/dev/null 2>&1; then
        echo "  (vllm skipped: SKIP_VLLM=${SKIP_VLLM:-0}, command=$(command -v vllm || echo absent))"
        python3 -c "
import json
f = json.load(open('$cell_dir/ferrum_baseline.json'))
ft = f.get('output_throughput_tps', {}).get('mean', 0)
print(f'  c=$c  ferrum={ft:.1f} tok/s  (vllm skipped)')
"
        continue
    fi
    VPORT=$((19000 + c))
    VLLM_ARGS=(
        "$HF_MODEL"
        --port "$VPORT"
        --gpu-memory-utilization 0.85
        --max-num-seqs 32
        --max-model-len 4096
        --dtype float16
        --no-enable-prefix-caching
        --enable-chunked-prefill
    )
    if [[ "$HF_MODEL" == *INT4* ]] || [[ "$HF_MODEL" == *Int4* ]] || [[ "$HF_MODEL" == *GPTQ* ]]; then
        VLLM_ARGS+=(--quantization gptq_marlin)
    fi
    vllm serve "${VLLM_ARGS[@]}" > "$cell_dir/vllm_serve.log" 2>&1 &
    VPID=$!
    for i in $(seq 1 90); do
        curl -fsS "http://127.0.0.1:$VPORT/health" >/dev/null 2>&1 && break
        sleep 2
    done
    "$FERRUM_BIN" bench-serve \
        --base-url "http://127.0.0.1:$VPORT" \
        --model "$HF_MODEL" --tokenizer "$TOK_SNAP" \
        --dataset random --random-input-len 256 --random-output-len 128 \
        --num-prompts "$NUM_PROMPTS" --warmup-requests "$WARMUP" \
        --n-repeats "$N_REPEATS" --concurrency "$c" \
        --output json --out "$cell_dir/vllm_baseline.json" \
        --tag "vllm" 2>&1 | tail -6
    kill "$VPID" 2>/dev/null || true
    wait "$VPID" 2>/dev/null || true
    sleep 2

    # Quick summary print
    python3 -c "
import json
f = json.load(open('$cell_dir/ferrum_baseline.json'))
v = json.load(open('$cell_dir/vllm_baseline.json'))
ft = f.get('output_throughput_tps', {}).get('mean', 0)
vt = v.get('output_throughput_tps', {}).get('mean', 0)
r = ft/vt if vt > 0 else 0
print(f'  c=$c  ferrum={ft:.1f} tok/s  vllm={vt:.1f} tok/s  ratio={r:.3f}')
"
done

if command -v nvidia-smi >/dev/null 2>&1; then
    bash scripts/unlock_gpu.sh 2>&1 || true
fi

echo
echo "✓ sweep complete: $OUT_ROOT/"
ls -la "$OUT_ROOT"
