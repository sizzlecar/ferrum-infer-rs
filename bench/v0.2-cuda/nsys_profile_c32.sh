#!/usr/bin/env bash
# Profile ferrum c=32 with nsys — kernel-level breakdown.
#
# Strategy: launch ferrum under nsys profile with --delay/--duration to
# capture a steady-state window (~20s of bench after warmup). The
# resulting .nsys-rep can be opened in nsight-systems GUI, or summarised
# via `"$NSYS" stats --report cuda_kernel_sum ferrum_c32.nsys-rep`.
#
# Use the data to answer:
#   - Of attn_peritem=5 ms: how split between split_qkv_norm_rope and
#     paged_decode_attention and o_proj?
#   - Of moe=6 ms: actual launches × time per launch for gemm1 (gate+up
#     Marlin) and gemm3 (down Marlin)?
#   - Of "other"=5 ms (or 2 ms with greedy): lm_head, final norm,
#     embedding_lookup, B::sync waits?
#   - Are there any unexpected long-tail kernels?
#
# Compare side-by-side with `nsys_profile_vllm_c32.sh` (TBD) for same-
# kernel timing diff.

set -uo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cd /workspace/ferrum-infer-rs

NSYS=""
if command -v nsys >/dev/null 2>&1; then
  NSYS=nsys
else
  # Fall back to the nsys bundled inside Nsight Compute (Vast cuda-devel
  # ships nsight-compute but not standalone nsight-systems).
  for candidate in /opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys; do
    if [ -x "$candidate" ]; then
      NSYS="$candidate"
      break
    fi
  done
fi
if [ -z "$NSYS" ]; then
  echo "nsys not found. Install Nsight Systems CLI." >&2
  exit 1
fi
echo "[nsys] using: $NSYS"

OUT_DIR=/workspace/ferrum-infer-rs/bench/v0.2-cuda/results_nsys_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_DIR"
NSYS_FILE="$OUT_DIR/ferrum_c32.nsys-rep"

pkill -9 -f "ferrum.*serve" 2>/dev/null || true
sleep 3
PORT=8801

# Launch ferrum under nsys. --delay=30 skips the ~30s startup (model
# load + first prewarm); --duration=20 captures ~20s of bench work
# which covers ~500 decode iters at c=32 + a few prefills.
CUDA_VISIBLE_DEVICES=0 \
FERRUM_VLLM_MOE=1 \
FERRUM_KV_CAPACITY=2048 \
FERRUM_KV_MAX_BLOCKS=4096 \
FERRUM_PAGED_MAX_SEQS=32 \
FERRUM_METAL_PAGED_KV=1 \
FERRUM_MIXED_BATCH=0 \
FERRUM_GREEDY_ARGMAX=1 \
FERRUM_MOE_BUCKETED=1 \
FERRUM_MARLIN_SKIP_WS_ZERO=1 \
FERRUM_MOE_STREAMS=4 \
FERRUM_MOE_BATCH_THRESHOLD=4 \
  "$NSYS" profile \
    --trace=cuda,nvtx,cudnn,osrt \
    --delay=15 --duration=20 \
    --force-overwrite=true \
    --output="$NSYS_FILE" \
    /workspace/ferrum-infer-rs/target/release/ferrum serve \
      --model /workspace/models/M3 --port "$PORT" \
      --gpu-memory-utilization 0.95 \
  > "$OUT_DIR/server.log" 2>&1 &
PID=$!

# Wait for ready
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "[ferrum] ready in ${i}s"; break; }
  ! kill -0 "$PID" 2>/dev/null && { echo "[ferrum] died"; tail -80 "$OUT_DIR/server.log"; exit 1; }
  sleep 1
done

# Prewarm
curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' \
  >/dev/null

source /workspace/vllm-venv/bin/activate

# Model load on Vast 4090 takes ~12 s (8 s safetensors + 4 s init); nsys
# delay=15 fires the capture window right when we hit `ready`. Bench
# runs ~12 s, fully inside the 20 s capture window (15..35 from process
# start). No host-side sleep needed — go straight from prewarm to bench.
echo "[nsys] capture window opens shortly; launching c=32 bench"
bash bench/v0.2-cuda/run_cell.sh ferrum_nsys M3 32 99 "$PORT" 2>&1 | tee "$OUT_DIR/bench.log"

# Wait for nsys to finish capturing
sleep 5
echo "[ferrum] stopping"
kill -INT "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
pkill -9 -f "ferrum.*serve" 2>/dev/null || true

echo
echo "=== nsys file: $NSYS_FILE ==="
ls -la "$NSYS_FILE"
echo
echo "=== top 20 kernels by total time ==="
"$NSYS" stats --report cuda_kernel_sum --format csv "$NSYS_FILE" 2>/dev/null \
  | head -22

echo
echo "=== top 10 cuda API calls ==="
"$NSYS" stats --report cuda_api_sum --format csv "$NSYS_FILE" 2>/dev/null \
  | head -12

echo
echo "Done. Inspect interactively: nsys-ui $NSYS_FILE"
