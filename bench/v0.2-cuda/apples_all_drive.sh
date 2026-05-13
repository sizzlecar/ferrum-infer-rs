#!/usr/bin/env bash
# Apples-to-apples sweep across M1 (FP16) / M2 (INT4 dense) / M3 (INT4 MoE).
# Uses the same env-rich ferrum config as apples_m3_drive.sh, but parameterized
# per model (MoE-specific knobs only when arch=MoE).
#
# Models selected via MODELS env: "M1 M2 M3" by default.
# Concurrencies fixed at 1/4/16/32, 1 repeat per cell.

set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cd /workspace/ferrum-infer-rs

MODELS="${MODELS:-M2 M1 M3}"  # M2 first (smallest), M3 last (largest)
CONCURRENCIES="${CONCURRENCIES:-1 4 16 32}"

pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "ferrum.*serve"    2>/dev/null || true
pkill -9 -f "vllm serve"       2>/dev/null || true
pkill -9 -f "VLLM"             2>/dev/null || true
sleep 5

vllm_args_for_model() {
    local tag="$1"
    # M1 = FP16 (no quantization flag), M2/M3 = GPTQ marlin
    case "$tag" in
        M1) echo "--max-num-seqs 64 --max-model-len 4096 --no-enable-prefix-caching --no-enable-log-requests" ;;
        M2|M3) echo "--max-num-seqs 64 --max-model-len 4096 --no-enable-prefix-caching --no-enable-log-requests --quantization gptq_marlin" ;;
        *) echo ""; return 1 ;;
    esac
}

ferrum_env_for_model() {
    local tag="$1"
    # Common env. MoE-specific knobs only matter for M3 but are harmless on dense.
    cat <<EOF
CUDA_VISIBLE_DEVICES=0
FERRUM_KV_CAPACITY=2048
FERRUM_KV_MAX_BLOCKS=4096
FERRUM_PAGED_MAX_SEQS=32
FERRUM_METAL_PAGED_KV=1
FERRUM_GREEDY_ARGMAX=1
FERRUM_MARLIN_SKIP_WS_ZERO=1
EOF
    if [ "$tag" = "M3" ]; then
        cat <<EOF
FERRUM_VLLM_MOE=1
FERRUM_MOE_BUCKETED=1
FERRUM_MOE_STREAMS=4
FERRUM_MOE_BATCH_THRESHOLD=4
FERRUM_MOE_GRAPH=1
FERRUM_GRAPH_SKIP_UPLOAD=1
EOF
    fi
}

run_model() {
    local TAG="$1"
    local MODEL_DIR="/workspace/models/$TAG"
    if [ ! -e "$MODEL_DIR" ]; then
        echo "[!! ] $TAG not found at $MODEL_DIR — skipping"
        return 0
    fi

    # ── vLLM half ───────────────────────────────────────────
    if [ -n "${SKIP_VLLM:-}" ]; then
        echo "[$TAG] SKIP_VLLM=1 — using existing vllm results from bench/v0.2-cuda/results/"
    else
    echo
    echo "============================================="
    echo "[$TAG] vllm"
    echo "============================================="
    pkill -9 -f "vllm" 2>/dev/null || true
    pkill -9 -f "VLLM" 2>/dev/null || true
    sleep 5
    source /workspace/vllm-venv/bin/activate
    PORT=8800
    SERVER_LOG="bench/v0.2-cuda/results/vllm__${TAG}__r1.server.log"
    local vargs="$(vllm_args_for_model "$TAG")"
    CUDA_VISIBLE_DEVICES=0 \
      vllm serve "$MODEL_DIR" --port "$PORT" $vargs \
      > "$SERVER_LOG" 2>&1 &
    VPID=$!
    for i in $(seq 1 600); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "[vllm] ready in ${i}s"; break; }
        ! kill -0 "$VPID" 2>/dev/null && { echo "[vllm] died"; tail -50 "$SERVER_LOG"; return 1; }
        sleep 1
    done
    curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_DIR\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"temperature\":0}" > /dev/null || true
    for c in $CONCURRENCIES; do
        echo "--- [vllm/$TAG] cell c=$c ---"
        rm -f bench/v0.2-cuda/results/vllm__${TAG}__c${c}__r1.*
        bash bench/v0.2-cuda/run_cell.sh vllm "$TAG" "$c" 1 "$PORT" 2>&1 | tail -3 || echo "  (cell failed)"
    done
    kill -INT "$VPID" 2>/dev/null; wait "$VPID" 2>/dev/null || true
    pkill -9 -f "vllm" 2>/dev/null || true
    pkill -9 -f "VLLM" 2>/dev/null || true
    deactivate 2>/dev/null || true
    sleep 5
    fi  # SKIP_VLLM gate

    # ── ferrum half ───────────────────────────────────────────
    echo
    echo "============================================="
    echo "[$TAG] ferrum"
    echo "============================================="
    PORT=8801
    SERVER_LOG="bench/v0.2-cuda/results/ferrum__${TAG}__r1.server.log"
    local env_vars=""
    while IFS= read -r line; do
        [ -n "$line" ] && env_vars="$env_vars $line"
    done < <(ferrum_env_for_model "$TAG")
    eval "$env_vars /workspace/ferrum-infer-rs/target/release/ferrum serve \
        --model \"$MODEL_DIR\" --port \"$PORT\" \
        --gpu-memory-utilization 0.95" > "$SERVER_LOG" 2>&1 &
    FPID=$!
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "[ferrum] ready in ${i}s"; break; }
        ! kill -0 "$FPID" 2>/dev/null && { echo "[ferrum] died"; tail -50 "$SERVER_LOG"; return 1; }
        sleep 1
    done
    curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' > /dev/null
    source /workspace/vllm-venv/bin/activate
    for c in $CONCURRENCIES; do
        echo "--- [ferrum/$TAG] cell c=$c ---"
        rm -f bench/v0.2-cuda/results/ferrum__${TAG}__c${c}__r1.*
        bash bench/v0.2-cuda/run_cell.sh ferrum "$TAG" "$c" 1 "$PORT" 2>&1 | tail -3 || echo "  (cell failed)"
    done
    # Hard kill — ferrum's SIGINT shutdown has a futex_wait_queue_me hang
    # that strands `wait $FPID` indefinitely (observed apples 2026-05-13
    # M2 c=32). Until that's fixed upstream, the iter loop uses SIGKILL.
    kill -9 "$FPID" 2>/dev/null || true
    pkill -9 -f "ferrum.*serve" 2>/dev/null || true
    sleep 3
}

for TAG in $MODELS; do
    run_model "$TAG"
done

echo
echo "============================================="
echo "Summary"
echo "============================================="
printf '%-8s %-3s %12s %12s\n' engine c "out_tps" "TPOT_p50_ms"
for engine in vllm ferrum; do
    for TAG in $MODELS; do
        for c in $CONCURRENCIES; do
            F="bench/v0.2-cuda/results/${engine}__${TAG}__c${c}__r1.json"
            if [ -f "$F" ]; then
                python3 -c "
import json
d = json.load(open('$F'))
c_str = '${c}'; print(f\"{'${engine}':8s} ${TAG} c={c_str:>2s} out={d.get('output_throughput',0):8.1f} tok/s  p50={d.get('mean_tpot_ms',0):6.2f}ms\")
"
            fi
        done
    done
done
