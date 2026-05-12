#!/bin/bash
# Phase A: per-kernel ncu profile for c=32 M3 hot path
#
# Usage:  bash bench/v0.2-cuda/ncu_profile.sh <ferrum|vllm>
#
# Strategy: ncu wrap mode around a self-contained bench (no separate
# server/client orchestration). Captures Speed-Of-Light + memory-throughput
# + scheduler-stats sections for the top decode kernels — comparable across
# the two engines once we map kernel names.
#
# Output:
#   /tmp/<engine>_profile.ncu-rep    — full Nsight Compute report (binary)
#   /tmp/<engine>_profile_summary.csv — extracted per-kernel metrics
#
# Note: ncu replay mode is 20-100× slower per captured kernel. A typical run
# captures ~30 launches across 3 kernels → ~100 kernel-replays, ~30-60 s of
# extra wall time. Total run < 5 min.

set -e
ENGINE="${1:-ferrum}"
LAUNCH_SKIP="${LAUNCH_SKIP:-500}"
LAUNCH_COUNT="${LAUNCH_COUNT:-30}"
OUT_REP="/tmp/${ENGINE}_profile.ncu-rep"
OUT_CSV="/tmp/${ENGINE}_profile_summary.csv"

if ! command -v ncu >/dev/null 2>&1; then
    if [ -x /usr/local/cuda/bin/ncu ]; then
        NCU=/usr/local/cuda/bin/ncu
    else
        echo "[ncu_profile] ncu not found"; exit 1
    fi
else
    NCU=ncu
fi
echo "[ncu_profile] using $($NCU --version | head -1)"

# Apples env block — same as apples_m3_drive.sh ferrum_start (the 1038 tok/s
# baseline config). We profile the baseline path, NOT paged_attn_v2.
export CUDA_VISIBLE_DEVICES=0
export FERRUM_VLLM_MOE=1
export FERRUM_KV_CAPACITY=2048
export FERRUM_KV_MAX_BLOCKS=4096
export FERRUM_PAGED_MAX_SEQS=32
export FERRUM_METAL_PAGED_KV=1
export FERRUM_MIXED_BATCH=0
export FERRUM_GREEDY_ARGMAX=1
export FERRUM_MOE_BUCKETED=1
export FERRUM_MARLIN_SKIP_WS_ZERO=1
export FERRUM_MOE_STREAMS=4
export FERRUM_MOE_BATCH_THRESHOLD=4
export FERRUM_MOE_GRAPH=1
export FERRUM_GRAPH_SKIP_UPLOAD=1

case "$ENGINE" in
    ferrum)
        # Use the self-contained `ferrum bench` subcommand — no server/
        # client split needed under ncu. c=32, short decode for speed.
        # Kernel regex captures ferrum's decode hot path:
        #   paged_batched_flash_decode_attn_f16  (attn dominant)
        #   marlin_moe_wna16_*                   (MoE GEMM)
        #   Marlin_3*                            (vLLM-marlin dense qkv/o)
        KERNEL_REGEX='regex:paged_batched_flash_decode_attn_f16|marlin_moe_wna16|Marlin_3'
        CMD="/workspace/ferrum-infer-rs/target/release/ferrum bench /workspace/models/M3 --concurrency 32 --max-tokens 64 --rounds 1"
        ;;
    vllm)
        # vLLM bench: serve + client are separate processes — ncu wraps
        # `vllm bench serve` which spawns the engine internally? Actually
        # safer to wrap the vllm engine directly. For now use `vllm serve`
        # under ncu and fire bench from a parallel ssh session.
        # TODO(phase-A.2): split this path properly.
        echo "[ncu_profile] vllm path requires manual orchestration (see TODO)"
        exit 2
        ;;
    *)
        echo "[ncu_profile] unknown engine: $ENGINE"; exit 1
        ;;
esac

echo "[ncu_profile] CMD: $CMD"
echo "[ncu_profile] kernel filter: $KERNEL_REGEX"
echo "[ncu_profile] skip=$LAUNCH_SKIP count=$LAUNCH_COUNT"

# --set basic keeps overhead reasonable while still emitting
# SOL+MemoryWorkloadAnalysis+ComputeWorkloadAnalysis.
$NCU \
    --target-processes application-only \
    --launch-skip "$LAUNCH_SKIP" \
    --launch-count "$LAUNCH_COUNT" \
    --kernel-name "$KERNEL_REGEX" \
    --set basic \
    --csv \
    --log-file "$OUT_CSV" \
    -o "$OUT_REP" \
    $CMD

echo
echo "[ncu_profile] done"
echo "  binary: $OUT_REP"
echo "  csv   : $OUT_CSV"
echo
echo "  Quick summary (top kernels by gpu__time_duration.sum):"
$NCU --import "$OUT_REP" --csv --print-summary per-kernel 2>/dev/null | \
    awk -F',' 'NR>1{print $0}' | head -20
