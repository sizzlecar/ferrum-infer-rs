#!/usr/bin/env bash
# run_sweep.sh — outer loop over the 144-cell matrix.
#
# Strategy (saves ~2 hr vs naive nested loop):
#   - For each (engine, model) pair, start the server ONCE with
#     --max-seqs=32 (covers all c values), prewarm with one small
#     request, then iterate (c × repeat) without restarting.
#   - 12 server starts total instead of 144.
#
# Order of (engine, model) pairs is fail-fast:
#   M2 (Llama-INT4)   first  — smallest, lowest blast radius
#   M1 (Llama-FP16)   second — biggest dense memory pressure
#   M3 (Qwen3-MoE)    third  — first MoE; mistralrs may panic here
#   M4 (Qwen3-Coder)  last   — same shape as M3, infrastructure proven
#
# Within each pair: c=1 first (catches correctness issues cheaply),
# c=32 last (catches resource issues), 3 repeats per c.
#
# Resume-safe: every completed cell is detected by results/*.json
# existing with output_throughput_tok_s > 0.

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
BENCH_DIR="${WORKSPACE}/ferrum-infer-rs/bench/v0.2-cuda"
RESULTS_DIR="$BENCH_DIR/results"
MODELS_DIR="${WORKSPACE}/models"
mkdir -p "$RESULTS_DIR"

# Matrix axes (v0.2 scope, 2026-05-03):
#   - mistralrs dropped (PoisonError on MoE, finicky setup, not the
#     primary target of comparison)
#   - M4 (Qwen3-Coder-30B-A3B GPTQ) dropped (community pack only)
ENGINES=(ferrum vllm)
MODEL_ORDER=(M2 M1 M3)            # cheapest blast radius first (8B INT4 → 8B FP16 → 30B MoE INT4)
CONCURRENCIES=(1 4 16 32)
REPEATS=(1 2 3)
# Total cells: 3 models × 2 engines × 4 c × 3 reps = 72

# Per-pair max-seqs sized for c=32 max + headroom
MAX_SEQS=64

# Read models.txt into associative arrays
declare -A MODEL_REPO MODEL_PRECISION MODEL_SIZE
while IFS='|' read -r tag repo precision size; do
  [[ -z "$tag" || "$tag" =~ ^# ]] && continue
  MODEL_REPO[$tag]="$repo"
  MODEL_PRECISION[$tag]="$precision"
  MODEL_SIZE[$tag]="$size"
done < "$BENCH_DIR/models.txt"

# Engine launch / kill helpers (one fn per engine).
PORT=8800

ferrum_start() {
  # Bash with `set -u` doesn't guarantee left-to-right evaluation of
  # `local a=… b=…$a` on a single line, so split the dependent vars.
  local model_tag="$1"
  local model_dir="$MODELS_DIR/$model_tag"
  local server_log="$RESULTS_DIR/ferrum__${model_tag}__server.log"
  echo "  starting ferrum on $model_tag ..." >&2
  # No paged-KV / max-seqs env vars — those are Metal-only.
  # ferrum's CUDA path uses its own decode runner config; matches
  # what `smoke_engines.sh` proved boots cleanly.
  CUDA_VISIBLE_DEVICES=0 \
    "$WORKSPACE/ferrum-infer-rs/target/release/ferrum" serve \
      --model "$model_dir" --port "$PORT" \
      > "$server_log" 2>&1 &
  ENGINE_PID=$!
}

vllm_start() {
  local model_tag="$1"
  local model_dir="$MODELS_DIR/$model_tag"
  local precision="${MODEL_PRECISION[$model_tag]}"
  local server_log="$RESULTS_DIR/vllm__${model_tag}__server.log"
  local quant_args=""
  if [[ "$precision" == "GPTQ_INT4" ]]; then
    quant_args="--quantization gptq_marlin"
  fi
  echo "  starting vllm on $model_tag (quant=$precision) ..." >&2
  # max-model-len 4096: our bench's worst case is prompt 512 +
  # output 512 = 1024; 4096 leaves headroom. Default model max
  # (e.g. 131072 for Llama-3.1) demands a 16 GB KV pool by itself,
  # which doesn't fit on a 24 GB 4090 alongside the weights.
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$model_dir" \
    --port "$PORT" \
    --max-num-seqs $MAX_SEQS \
    --max-model-len 4096 \
    --no-enable-prefix-caching \
    --disable-log-requests \
    $quant_args \
    > "$server_log" 2>&1 &
  ENGINE_PID=$!
}

# mistralrs_start removed in v0.2 scope — see ENGINES list above.

# Wait for the engine to be ready (up to 5 min). Kill if not up.
wait_ready() {
  local pid="$1"
  for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "  ready in ${i}s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "  process died before becoming ready"
      return 1
    fi
    sleep 1
  done
  echo "  timeout (5 min) waiting for ready"
  return 1
}

# Prewarm with one small chat completion (drops cold-start variance
# from the very first cell of each pair).
prewarm() {
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"default","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"stream":false}' \
    > /dev/null 2>&1 || true
}

# Cleanly kill the engine process tree.
kill_engine() {
  local pid="$1"
  if [[ -n "$pid" ]]; then
    kill -TERM "$pid" 2>/dev/null || true
    sleep 2
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "mistralrs-server" 2>/dev/null || true
  pkill -9 -f "ferrum.*serve" 2>/dev/null || true
  sleep 3
}

# ── outer loop ──────────────────────────────────────────────────────
START_T=$SECONDS
for engine in "${ENGINES[@]}"; do
  for model in "${MODEL_ORDER[@]}"; do
    PAIR="${engine}__${model}"
    PAIR_LOG="$RESULTS_DIR/${PAIR}__pair.log"

    # Skip pair if all 12 cells already exist.
    REMAINING=0
    for c in "${CONCURRENCIES[@]}"; do
      for r in "${REPEATS[@]}"; do
        cell="$RESULTS_DIR/${engine}__${model}__c${c}__r${r}.json"
        if [[ ! -f "$cell" ]] || [[ "$(python3 -c "import json,sys; print(json.load(open('$cell')).get('output_throughput_tok_s',0))" 2>/dev/null || echo 0)" == "0.0" ]]; then
          REMAINING=$((REMAINING+1))
        fi
      done
    done
    if [[ $REMAINING -eq 0 ]]; then
      echo "── skip $PAIR (12/12 cells already done) ──"
      continue
    fi

    echo
    echo "════════════════════════════════════════════════════════════"
    echo " $PAIR — $REMAINING cells remaining"
    echo "════════════════════════════════════════════════════════════"

    # Start server. We avoid `$(engine_start ...)` capture because
    # `&` inside $() runs in a subshell — $! is the subshell-local
    # background PID, and subshell exit can confuse downstream
    # `kill -0` checks. Use a global ENGINE_PID set by the start fn.
    case "$engine" in
      ferrum)    ferrum_start "$model"    ;;
      vllm)      vllm_start "$model"      ;;
    esac
    PID="$ENGINE_PID"

    if ! wait_ready "$PID" >> "$PAIR_LOG" 2>&1; then
      echo "  $PAIR FAILED to come up (see $PAIR_LOG and ${PAIR}__server.log) — skip 12 cells"
      kill_engine "$PID"
      continue
    fi
    prewarm

    # Run all (c × repeat) cells against this server
    for c in "${CONCURRENCIES[@]}"; do
      for r in "${REPEATS[@]}"; do
        bash "$BENCH_DIR/run_cell.sh" "$engine" "$model" "$c" "$r" "$PORT" || true
      done
    done

    kill_engine "$PID"

    # Disk space + time check
    DISK_USED=$(df -h "$WORKSPACE" | tail -1 | awk '{print $5}')
    ELAPSED=$((SECONDS - START_T))
    echo "── $PAIR done; total elapsed: $((ELAPSED/60)) min; disk: $DISK_USED ──"
  done
done

echo
echo "═══════════════════════════════════════════════════════════"
echo "  sweep complete in $(( (SECONDS - START_T) / 60 )) min"
echo "  results: $RESULTS_DIR"
echo "  exfil:   bash bench/v0.2-cuda/pull_results.sh   (run from local)"
echo "═══════════════════════════════════════════════════════════"
