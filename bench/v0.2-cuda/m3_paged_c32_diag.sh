#!/bin/bash
# Stage 9 paged-KV c=32 OOB diagnostic. Three configs to isolate
# whether the bug is (a) KV pool block exhaustion (more seqs spawn
# more cumulative block allocs), (b) concurrency-specific kernel bug,
# or (c) something else in the long-tail of the bench.

set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
BENCH_DIR="${WORKSPACE}/ferrum-infer-rs/bench/v0.2-cuda"
RESULTS_DIR="$BENCH_DIR/results_m3_paged_diag"
MODEL_DIR="${WORKSPACE}/models/M3"
mkdir -p "$RESULTS_DIR"

PORT=8800

run_one() {
  local label="$1"; local kv_blocks="$2"; local conc="$3"; local nprompts="$4"
  local server_log="$RESULTS_DIR/${label}__server.log"
  echo "=== $label  (KV_MAX_BLOCKS=$kv_blocks  c=$conc  n=$nprompts) ==="

  CUDA_VISIBLE_DEVICES=0 \
  FERRUM_KV_CAPACITY=2048 \
  FERRUM_KV_MAX_BLOCKS=$kv_blocks \
  FERRUM_PAGED_MAX_SEQS=$conc \
  FERRUM_METAL_PAGED_KV=1 \
  FERRUM_MIXED_BATCH=0 \
  FERRUM_GREEDY_ARGMAX=1 \
  FERRUM_MOE_BUCKETED=1 \
  FERRUM_MARLIN_SKIP_WS_ZERO=1 \
  FERRUM_MOE_STREAMS=4 \
    "$WORKSPACE/ferrum-infer-rs/target/release/ferrum" serve \
      --model "$MODEL_DIR" --port "$PORT" \
      --gpu-memory-utilization 0.95 \
      > "$server_log" 2>&1 &
  PID=$!

  for i in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
      tail -30 "$server_log"
      return 1
    fi
    sleep 1
  done

  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}' \
    >/dev/null

  vllm bench serve \
    --backend openai-chat \
    --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/chat/completions \
    --model "$MODEL_DIR" \
    --tokenizer "$MODEL_DIR" \
    --dataset-name random \
    --random-input-len 256 --random-output-len 128 \
    --num-prompts $nprompts \
    --max-concurrency $conc \
    --save-result --result-dir "$RESULTS_DIR" \
    --result-filename "${label}.json" \
    --percentile-metrics "ttft,tpot,itl" 2>&1 | tee "$RESULTS_DIR/${label}.log" >/dev/null \
    || echo "[$label] bench failed/hung"

  kill -INT "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
  sleep 3
}

# A: c=32 with smaller pool. Should fail FAST if exhaustion.
# B: c=32 with smaller num_prompts (64 vs 128). If A fails, B works → exhaustion.
# C: c=24 to find concurrency threshold. If c=24 works but c=32 breaks → kernel bug above some threshold.
# D: c=32 with bigger pool (8192 blocks).

run_one a_c32_blocks2048_n64  2048 32 64    || true
run_one b_c24_blocks2048_n128 2048 24 128   || true
run_one c_c32_blocks4096_n128 4096 32 128   || true
run_one d_c32_blocks8192_n128 8192 32 128   || true

echo ""
echo "=== diag headlines ==="
for f in "$RESULTS_DIR"/*.json; do
  [[ -f "$f" ]] || continue
  python3 -c "
import json
d = json.load(open('$f'))
n = '$f'.split('/')[-1].replace('.json','')
print('%-30s  tps=%.1f  TPOT=%.2fms' % (n, d.get('output_throughput', 0.0), d.get('mean_tpot_ms', 0.0)))
"
done
echo ""
echo "=== panic check ==="
grep -l 'panicked\|ILLEGAL\|aborted' "$RESULTS_DIR"/*server.log 2>/dev/null | xargs -I {} basename {}
