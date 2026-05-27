#!/usr/bin/env bash
#
# paris_bisect.sh — 4-env smoke test for the MoE device-route path.
#
# Verifies whether ferrum's vllm-moe-marlin device-route emits correct
# token output (the canary "Paris" prompt). Without this gate, the M3
# sweep would happily report tok/s for garbled output, as it did in
# session-2026-05-25 (see docs/bench/m3-80pct-goal-2026-05-25/
# session-2026-05-26-corrected/SESSION-REPORT.md).
#
# Each cell runs:
#   ferrum run Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
#       --prompt "What is the capital of France?" \
#       --output-format jsonl --max-tokens 64 --temperature 0.0
# under different env combinations, then greps the assistant content for
# "Paris". A cell that *should* produce Paris but does not is a regression.
#
# Exit codes:
#   0  — all expected cells produce "Paris"
#   1  — at least one expected cell missing "Paris" (fix is broken)
#   2  — ferrum binary missing or model not downloaded
#
# Usage:
#   bash scripts/paris_bisect.sh [FERRUM_BIN] [OUT_DIR]
#     FERRUM_BIN defaults to ./target/release/ferrum
#     OUT_DIR    defaults to /tmp/paris-bisect-<unix-ts>

set -uo pipefail

FERRUM_BIN="${1:-./target/release/ferrum}"
OUT_DIR="${2:-/tmp/paris-bisect-$(date +%s)}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}"
PROMPT="${PROMPT:-What is the capital of France?}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TIMEOUT="${TIMEOUT:-180}"  # per-cell wall-clock budget

mkdir -p "$OUT_DIR"

if [ ! -x "$FERRUM_BIN" ]; then
    echo "FATAL: ferrum binary not found at $FERRUM_BIN" >&2
    exit 2
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Run one cell: name, expected_outcome (pass/garbled), env-kv pairs...
run_cell() {
    local name="$1"; shift
    local expect="$1"; shift  # "pass" if must contain Paris, "any" otherwise
    local env_args=("$@")

    local log_file="$OUT_DIR/${name}.log"
    local out_file="$OUT_DIR/${name}.out"
    local content=""

    log "▶ cell $name (expect=$expect): ${env_args[*]}"

    # Clear all ferrum env first, then apply this cell's
    # Note: env -i wipes everything; we re-inject HF_HOME, PATH, CUDA_*
    timeout "$TIMEOUT" env -i \
        HOME="$HOME" \
        PATH="$PATH" \
        HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
        CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
        LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        "${env_args[@]}" \
        "$FERRUM_BIN" run "$MODEL" \
            --prompt "$PROMPT" \
            --output-format jsonl \
            --max-tokens "$MAX_TOKENS" \
            --temperature 0.0 \
            > "$out_file" 2> "$log_file"
    local rc=$?

    if [ $rc -ne 0 ]; then
        log "  ✗ exit $rc (see $log_file)"
        echo "EXIT_$rc" > "$OUT_DIR/${name}.status"
        return 1
    fi

    # Extract the assistant content from the jsonl stream
    content=$(grep '"event":"assistant"' "$out_file" | head -1 | python3 -c '
import json,sys
try:
    d=json.loads(sys.stdin.read())
    print(d.get("content","").replace("\n"," ")[:200])
except Exception as e:
    print(f"PARSE_ERR:{e}")
' 2>/dev/null)

    log "  content: $content"
    echo "$content" > "$OUT_DIR/${name}.content"

    case "$expect" in
        pass)
            if echo "$content" | grep -qi "paris"; then
                log "  ✓ Paris found"
                echo "PASS" > "$OUT_DIR/${name}.status"
                return 0
            else
                log "  ✗ Paris MISSING — fix is broken or this path is regressed"
                echo "FAIL_NO_PARIS" > "$OUT_DIR/${name}.status"
                return 1
            fi
            ;;
        any)
            echo "OBSERVED" > "$OUT_DIR/${name}.status"
            return 0
            ;;
        *)
            log "  ? unknown expect=$expect"
            return 2
            ;;
    esac
}

log "Paris bisect → $OUT_DIR"
log "  model:   $MODEL"
log "  binary:  $FERRUM_BIN"

# Common env knobs (iter-3 baseline, sans the bisect-controlled ones).
# 2026-05-27 cleanup:
#   - FERRUM_GRAPH_SKIP_UPLOAD=1 dropped: measured -3.8% at c=32.
#     graph.rs:156's "re-upload needed for c=4 perf" rationale holds
#     at c=32 too; default-unset (= re-upload) is the right value.
#   - FERRUM_USE_VLLM_PAGED_ATTN=1 dropped: regression in current
#     branch produces garbage on non-split-K varlen path. Unrelated
#     to moe_align fix; track separately.
#   - FERRUM_GRAPH=X kept in per-cell envs as cell labels (real
#     gating is FERRUM_MOE_GRAPH; FERRUM_GRAPH is a no-op placebo).
COMMON=(
    FERRUM_MOE_DEVICE_ROUTE=1
    FERRUM_MOE_STREAMS=4
    FERRUM_GREEDY_ARGMAX=1
    FERRUM_KV_MAX_BLOCKS=2048
    FERRUM_PAGED_MAX_SEQS=32
)

# Track failures
fails=0

# Cell A — SAFE baseline (no vllm-moe-marlin, no graph): must always work
run_cell "A_safe" "pass" \
    "${COMMON[@]}" \
    FERRUM_MOE_GRAPH=0 FERRUM_VLLM_MOE=0 \
    || fails=$((fails+1))

# Cell B — VLLM_MOE only (graph off): THE primary bisect cell.
# Before the moe_align_block_size.cu fix this emits garbage.
run_cell "B_vllm_moe" "pass" \
    "${COMMON[@]}" \
    FERRUM_MOE_GRAPH=0 FERRUM_VLLM_MOE=1 \
    || fails=$((fails+1))

# Cell C — VLLM_MOE + GRAPH on: must also produce Paris (graph is innocent
# per the corrected diagnosis — B and C should both pass or both fail)
run_cell "C_vllm_moe_graph" "pass" \
    "${COMMON[@]}" \
    FERRUM_MOE_GRAPH=1 FERRUM_VLLM_MOE=1 \
    || fails=$((fails+1))

# Cell D — VLLM_MOE + HOST_ROUTE workaround: unchanged by the kernel fix,
# must still produce Paris (this was the only ON-path that worked pre-fix)
run_cell "D_vllm_moe_host_route" "pass" \
    "${COMMON[@]}" \
    FERRUM_MOE_GRAPH=0 FERRUM_VLLM_MOE=1 FERRUM_MOE_HOST_ROUTE=1 \
    || fails=$((fails+1))

log ""
log "═════ Paris bisect summary ═════"
for cell in A_safe B_vllm_moe C_vllm_moe_graph D_vllm_moe_host_route; do
    status=$(cat "$OUT_DIR/${cell}.status" 2>/dev/null || echo "MISSING")
    log "  $cell: $status"
done

if [ "$fails" -eq 0 ]; then
    log "✓ ALL CELLS PASS — fix verified, safe to proceed with sweep"
    exit 0
else
    log "✗ $fails CELL(S) FAILED — abort sweep, investigate $OUT_DIR"
    exit 1
fi
