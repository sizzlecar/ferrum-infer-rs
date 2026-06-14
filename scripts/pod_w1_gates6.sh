#!/usr/bin/env bash
# W1 single-pod final smoke pass: per-model KV pins sized to fit a 24GB
# card alongside each model's weights (the CUDA autosizer either picks
# 512/seq — 400s the ladder's max_tokens — or computes zero blocks for
# 32B-GPTQ; both recorded as the autosizer W1 follow-up).
set -ux
cd "$(dirname "$0")/.."
. "$HOME/.cargo/env" 2>/dev/null || true
BIN=target/release/ferrum
G=/workspace/w1/gates
mkdir -p "$G"
export HF_HUB_OFFLINE=1

s6() { # alias rid kv seqs extra...
  local alias="$1" rid="$2" kv="$3" seqs="$4"; shift 4
  pkill -f "[f]errum serve" 2>/dev/null; sleep 2
  FERRUM_BIN=$BIN SMOKE_REQ_TIMEOUT=900 bash scripts/model_coverage_smoke.sh \
    "$alias" "$@" --kv-capacity "$kv" --max-seqs "$seqs" --port 8240 \
    > "$G/smoke6_${rid}_cuda.log" 2>&1 \
    && echo "=== STAGE smoke6_${rid} PASS" || echo "=== STAGE smoke6_${rid} FAIL"
}
s6 qwen3-coder:30b-gptq coder-30b-gptq 4096 2
s6 deepseek-r1:32b-gptq r1-32b-gptq 4096 2 --reasoning
s6 qwen3:32b-gptq qwen3-32b-gptq 4096 2
s6 deepseek-r1:8b r1-8b-bf16 4096 1 --reasoning
pkill -f "[f]errum serve" 2>/dev/null
echo "=== GATES6 COMPLETE"
