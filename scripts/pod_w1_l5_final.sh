#!/usr/bin/env bash
# Final W1 micro-session: L5 benches for the three 32B GPTQ models on one
# pod (mutual-spread = the amendment-#4 same-pod perf evidence). Serve
# pins: 512 tokens/seq x 32 seqs (random 256/128 fits; ~4.3GB pool beside
# ~18GB weights on a 24GB card).
set -ux
cd "$(dirname "$0")/.."
. "$HOME/.cargo/env" 2>/dev/null || true
BIN=target/release/ferrum
W=/workspace/w1
G=$W/gates
mkdir -p "$G"
export HF_HUB_OFFLINE=1

l5f() { # alias rid repodir
  local alias="$1" rid="$2" repodir="$3"
  pkill -f "[f]errum serve" 2>/dev/null; sleep 2
  local tokjson tokdir
  tokjson=$(ls /root/.cache/huggingface/hub/"$repodir"/snapshots/*/tokenizer.json 2>/dev/null | head -1)
  if [ -z "$tokjson" ]; then echo "=== STAGE l5f_${rid} FAIL (no tokenizer)"; return; fi
  tokdir=$(dirname "$tokjson")
  nohup "$BIN" serve "$alias" --kv-capacity 512 --max-num-seqs 32 --port 8300 \
    > "$W/serve_l5f_${rid}.log" 2>&1 &
  local srv=$!
  local healthy=0
  for i in $(seq 1 150); do
    if curl -sf --max-time 2 http://127.0.0.1:8300/health >/dev/null 2>&1; then healthy=1; break; fi
    if ! kill -0 "$srv" 2>/dev/null; then break; fi
    sleep 3
  done
  if [ "$healthy" != "1" ]; then
    echo "=== STAGE l5f_${rid} FAIL (server)"; tail -3 "$W/serve_l5f_${rid}.log"; return
  fi
  "$BIN" bench-serve --base-url http://127.0.0.1:8300 --model "$alias" \
    --tokenizer "$tokdir" --concurrency-sweep 1,4,16,32 --num-prompts 100 \
    --n-repeats 3 --out "$G/l5f_${rid}_cuda.json" > "$G/l5f_${rid}_run.log" 2>&1 \
    && echo "=== STAGE l5f_${rid} PASS" || echo "=== STAGE l5f_${rid} FAIL"
  kill "$srv" 2>/dev/null; sleep 2
}
l5f deepseek-r1:32b-gptq r1-32b models--OPEA--DeepSeek-R1-Distill-Qwen-32B-int4-gptq-sym-inc
l5f qwen3:32b-gptq qwen3-32b models--JunHowie--Qwen3-32B-GPTQ-Int4
l5f qwen2.5-coder:32b-gptq qwen25-coder-32b models--Qwen--Qwen2.5-Coder-32B-Instruct-GPTQ-Int4
pkill -f "[f]errum serve" 2>/dev/null
echo "=== L5FINAL COMPLETE"
