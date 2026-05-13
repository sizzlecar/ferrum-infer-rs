#!/bin/bash
# Phase A v2: nsys per-kernel time profile for c=32 M3.
#
# Usage:  bash bench/v0.2-cuda/nsys_profile.sh <ferrum|vllm>
#
# Output:
#   /tmp/<engine>_bench.nsys-rep       — nsys binary report
#   /tmp/<engine>_kernels.csv          — per-kernel time summary (sorted)
#   /tmp/<engine>_apis.csv             — per-CUDA-API time summary
#
# Why nsys instead of ncu:  Vast multi-tenant kernel module sets
# `NVreg_RestrictProfilingToAdminUsers=1`, blocking ncu's GPU PM counters
# (ERR_NVGPUCTRPERM). nsys uses CUPTI activity API which has different
# perm requirements and works in restricted containers.

set -e
ENGINE="${1:-ferrum}"
OUT_REP="/tmp/${ENGINE}_bench.nsys-rep"
OUT_KER="/tmp/${ENGINE}_kernels.csv"
OUT_API="/tmp/${ENGINE}_apis.csv"

if ! command -v nsys >/dev/null 2>&1; then
    echo "[nsys_profile] nsys not found in PATH; install nsight-systems-2024.X"
    exit 1
fi
echo "[nsys_profile] $(nsys --version | head -1)"

# Apples env (matches apples_m3_drive.sh, the 1038 tok/s baseline config)
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
        # Self-contained — no client needed. c=32, short decode keeps the
        # capture file small. --delay 15 skips model load + warmup; ferrum
        # bench's actual GPU activity starts ~13s after process launch.
        nsys profile \
            --output "$OUT_REP" \
            --trace cuda \
            --sample none \
            --delay 13 \
            --duration 20 \
            --force-overwrite true \
            /workspace/ferrum-infer-rs/target/release/ferrum bench \
                /workspace/models/M3 \
                --concurrency 32 \
                --max-tokens 64 \
                --rounds 1
        ;;
    vllm)
        # vLLM: server + client in separate processes. Profile only the
        # server (where kernels actually run). Bench client fires from a
        # parallel `nohup` invocation; nsys --duration window captures the
        # bench window.
        cd /workspace/ferrum-infer-rs
        rm -f "$OUT_REP"
        # Start vllm server under nsys
        echo "[nsys_profile] launching vllm server (under nsys)..."
        source /workspace/vllm-venv/bin/activate
        nohup nsys profile \
            --output "$OUT_REP" \
            --trace cuda \
            --sample none \
            --delay 60 \
            --duration 30 \
            --force-overwrite true \
            vllm serve /workspace/models/M3 --port 8800 \
                --max-num-seqs 64 --max-model-len 4096 \
                --no-enable-prefix-caching --no-enable-log-requests \
                --quantization gptq_marlin \
            > /tmp/vllm_server.out 2>&1 &
        SERVER_PID=$!
        echo "[nsys_profile] SERVER_PID=$SERVER_PID"

        # Wait for vllm ready (server outputs "Application startup complete")
        for i in $(seq 1 180); do
            if curl -sf http://127.0.0.1:8800/v1/models >/dev/null 2>&1; then
                echo "[nsys_profile] vllm ready in ${i}s"
                break
            fi
            ! kill -0 $SERVER_PID 2>/dev/null && { echo "vllm died"; tail -50 /tmp/vllm_server.out; exit 1; }
            sleep 1
        done

        # Fire bench (now overlapping nsys delay/duration window)
        echo "[nsys_profile] firing bench c=32..."
        bash bench/v0.2-cuda/run_cell.sh vllm M3 32 nsys 8800 2>&1 | tail -10 || true

        # Give nsys time to finish its duration window
        echo "[nsys_profile] waiting for nsys to complete capture window..."
        wait $SERVER_PID 2>/dev/null || true
        pkill -9 -f "vllm serve" 2>/dev/null || true
        ;;
    *)
        echo "[nsys_profile] unknown engine: $ENGINE"; exit 1
        ;;
esac

echo
echo "[nsys_profile] generating CSV reports..."
nsys stats --report cuda_gpu_kern_sum --format csv "$OUT_REP" > "$OUT_KER" 2>/dev/null
nsys stats --report cuda_api_sum     --format csv "$OUT_REP" > "$OUT_API" 2>/dev/null

echo
echo "[nsys_profile] done"
echo "  $OUT_REP"
echo "  $OUT_KER"
echo "  $OUT_API"
echo
echo "=== top 15 kernels by total time ==="
# CSV columns:  Time(%),Total Time(ns),Instances,Avg(ns),Med(ns),Min,Max,StdDev,Name
head -1 "$OUT_KER"
tail -n +2 "$OUT_KER" | sort -t',' -k2 -n -r | head -15
