#!/bin/bash
# Re-run the Qwen3-30B-A3B row with the correct MoE-batched env vars.
#
# Why this exists: the original run_suite.sh ran ferrum on the MoE model
# WITHOUT `FERRUM_MOE_BATCHED=1 FERRUM_MOE_BATCHED_DECODE=1`. The batched
# MoE path is opt-in by default (see qwen3_moe.rs::moe_forward_dispatch
# line 2496 — `opted_in = std::env::var("FERRUM_MOE_BATCHED") == "1"`).
# Without the opt-in, the engine falls back to the per-token loop on MoE.
# Per-token at c=16 lands at ~48 tok/s; batched at c=16 lands at ~80
# tok/s. We want the latter — that's the engine's actual capability.
#
# Dense (Llama-3.1-8B, Qwen3-8B) doesn't need these env vars and the
# original suite numbers are correct for them.
#
# This script also runs llama.cpp's MoE row again so we have a clean
# back-to-back comparison instead of cells run hours apart on different
# memory pressure states.
#
# mistralrs is skipped — it `PoisonError`-panics on Qwen3-30B-A3B-Q4_K_M
# (mistralrs-core 0.8.1 add_request.rs:466), so a re-run produces 0 tok/s.

set -u

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FERRUM_BIN="${FERRUM_BIN:-$ROOT/target/release/ferrum}"
GGUF_DIR="${GGUF_DIR:-/Users/chejinxuan/ferrum-bench/models}"
BENCH_PY="$ROOT/bench/scripts/bench_serving.py"
OUTDIR="$(cd "$(dirname "$0")" && pwd)"

PORT_FERRUM=8783
PORT_LLAMACPP=8001

GGUF_NAME="Qwen3-30B-A3B-Q4_K_M.gguf"
MODEL_LABEL="Qwen3-30B-A3B"
GGUF="$GGUF_DIR/$GGUF_NAME"

CONFIGS=(
  "1:8:64"
  "4:16:64"
  "8:24:64"
  "16:32:64"
)
COOLDOWN=20

cooldown() {
  pkill -9 -f "ferrum.*serve|llama-server|mistralrs.*serve" 2>/dev/null
  sleep "$COOLDOWN"
  echo "[mem] $(sysctl -n vm.swapusage | awk '{print $4, $5, $6}') | $(vm_stat | awk '/Pages free/ {print "free="$3}')"
}

run_ferrum_moe_batched() {
  local c="$1" num_prompts="$2" max_tokens="$3"
  cooldown
  local cell="ferrum_moebatched__${MODEL_LABEL}__c${c}"
  echo ""
  echo "▶ $cell"
  FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS=$((c * 2)) FERRUM_KV_CAPACITY=512 \
  FERRUM_MAX_BATCH=$c \
  FERRUM_MOE_BATCHED=1 FERRUM_MOE_BATCHED_DECODE=1 FERRUM_MOE_BATCH_THRESHOLD=2 \
    "$FERRUM_BIN" serve --model "$GGUF" --port "$PORT_FERRUM" \
    > "$OUTDIR/${cell}.server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT_FERRUM/v1/models" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT_FERRUM/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_LABEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT_FERRUM" \
    --model "$MODEL_LABEL" \
    --num-prompts $num_prompts --max-concurrency $c --max-tokens $max_tokens \
    --deterministic-prompts \
    --result-file "$OUTDIR/${cell}.json" \
    > "$OUTDIR/${cell}.bench.log" 2>&1
  python3 -c "
import json
try:
    d = json.load(open('$OUTDIR/${cell}.json'))
    f = lambda x: f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
    tpot=d.get('tpot_ms',{})
    print(f'  out_tok/s={f(d.get(\"output_throughput_tok_s\"))}  TPOT_med={f(tpot.get(\"median\"))}')
except Exception as e:
    print(f'  ERROR: {e}')
" 2>/dev/null
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}

run_llamacpp_moe() {
  local c="$1" num_prompts="$2" max_tokens="$3"
  cooldown
  local cell="llamacpp_moe__${MODEL_LABEL}__c${c}"
  echo ""
  echo "▶ $cell"
  llama-server --model "$GGUF" --port "$PORT_LLAMACPP" \
    --ctx-size 4096 --parallel $c --batch-size 2048 --jinja \
    > "$OUTDIR/${cell}.server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT_LLAMACPP/v1/models" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf -m 60 -X POST "http://127.0.0.1:$PORT_LLAMACPP/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_LABEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":false,\"temperature\":0.0}" \
    > /dev/null 2>&1
  python3 "$BENCH_PY" \
    --base-url "http://127.0.0.1:$PORT_LLAMACPP" \
    --model "$MODEL_LABEL" \
    --num-prompts $num_prompts --max-concurrency $c --max-tokens $max_tokens \
    --deterministic-prompts \
    --result-file "$OUTDIR/${cell}.json" \
    > "$OUTDIR/${cell}.bench.log" 2>&1
  python3 -c "
import json
try:
    d = json.load(open('$OUTDIR/${cell}.json'))
    f = lambda x: f'{x:.1f}' if isinstance(x,(int,float)) else 'n/a'
    tpot=d.get('tpot_ms',{})
    print(f'  out_tok/s={f(d.get(\"output_throughput_tok_s\"))}  TPOT_med={f(tpot.get(\"median\"))}')
except Exception as e:
    print(f'  ERROR: {e}')
" 2>/dev/null
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}

cat "$GGUF" > /dev/null  # prewarm page cache

for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r c num_prompts max_tokens <<< "$cfg"
  run_ferrum_moe_batched "$c" "$num_prompts" "$max_tokens"
  run_llamacpp_moe       "$c" "$num_prompts" "$max_tokens"
done

echo ""
echo "[done] $(date '+%Y-%m-%d %H:%M:%S')"
sysctl vm.swapusage
