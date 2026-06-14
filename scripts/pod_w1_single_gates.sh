#!/usr/bin/env bash
# W1 single-4090 gate sequence. Run from the repo root after
# pod_w1_bootstrap.sh single has staged everything (build.ok + 6 dl_*.ok).
# Artifacts land in /workspace/w1/gates/. Each stage prints a
# `=== STAGE <name> PASS|FAIL` marker; failures continue to the next model
# (GOAL: 首败即停该模型,不阻塞同批).
set -ux
cd "$(dirname "$0")/.."
. "$HOME/.cargo/env" 2>/dev/null || true
BIN=target/release/ferrum
W=/workspace/w1
G=$W/gates
mkdir -p "$G"
export HF_HUB_OFFLINE=1

serve_wait() { # alias-or-path port extra...
  local model="$1" port="$2"; shift 2
  pkill -f "ferrum serve" 2>/dev/null; sleep 2
  nohup "$BIN" serve "$model" --port "$port" "$@" >"$W/serve_${port}.log" 2>&1 &
  for i in $(seq 1 150); do
    curl -sf --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "SERVER FAILED on $port"; tail -5 "$W/serve_${port}.log"; return 1
}

# ---- Stage 1: L1 representative (R1-8B BF16 vs HF transformers) ----
if [ ! -f "$G/l1_refs.json" ]; then
  python3 scripts/pod_l1_reference_match.py gen \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out "$G/l1_refs.json" \
    > "$G/l1_gen.log" 2>&1 && echo "=== STAGE l1_gen PASS" || echo "=== STAGE l1_gen FAIL"
fi
if [ -f "$G/l1_refs.json" ]; then
  # BF16 8B = 16GB weights; the KV pool lives in VRAM on CUDA, so the
  # reasoning-default 8192x4 pool (~4.8GB) OOMs a 24GB card. 4096x2 fits.
  if serve_wait deepseek-r1:8b 8201 --kv-capacity 4096 --max-num-seqs 2; then
    python3 scripts/pod_l1_reference_match.py verify --refs "$G/l1_refs.json" \
      --base-url http://127.0.0.1:8201 --alias deepseek-r1:8b --out "$G/l1_report.json" \
      > "$G/l1_verify.log" 2>&1 && echo "=== STAGE l1_verify PASS" || echo "=== STAGE l1_verify FAIL"
    # CUDA BF16 smoke for R1-8B rides the same weights
    FERRUM_BIN=$BIN SMOKE_REQ_TIMEOUT=900 bash scripts/model_coverage_smoke.sh \
      deepseek-r1:8b --reasoning --kv-capacity 4096 --max-seqs 2 --port 8202 \
      > "$G/smoke_r1-8b-bf16_cuda.log" 2>&1 \
      && echo "=== STAGE smoke_r1-8b PASS" || echo "=== STAGE smoke_r1-8b FAIL"
  fi
fi

# ---- Stage 2: GPTQ smokes ----
smoke() { # alias rid extra-flags...
  local alias="$1" rid="$2"; shift 2
  pkill -f "ferrum serve" 2>/dev/null; sleep 2
  FERRUM_BIN=$BIN SMOKE_REQ_TIMEOUT=900 bash scripts/model_coverage_smoke.sh \
    "$alias" "$@" --port 8210 > "$G/smoke_${rid}_cuda.log" 2>&1 \
    && echo "=== STAGE smoke_${rid} PASS" || echo "=== STAGE smoke_${rid} FAIL"
}
smoke qwen3-coder:30b-gptq coder-30b-gptq
smoke deepseek-r1:32b-gptq r1-32b-gptq --reasoning
smoke qwen3:32b-gptq qwen3-32b-gptq
smoke qwen2.5-coder:32b-gptq qwen25-coder-32b-gptq

# ---- Stage 3: L5 per model (c=1/4/16; 30B-class +32) ----
l5() { # alias rid hfrepo csweep
  local alias="$1" rid="$2" repo="$3" csweep="$4"
  local tokdir
  tokdir=$(dirname "$(ls /root/.cache/huggingface/hub/models--${repo}/snapshots/*/tokenizer.json | head -1)")
  if serve_wait "$alias" 8220; then
    "$BIN" bench-serve --base-url http://127.0.0.1:8220 --model "$alias" \
      --tokenizer "$tokdir" --concurrency-sweep "$csweep" --num-prompts 100 \
      --n-repeats 3 --out "$G/l5_${rid}_cuda.json" > "$G/l5_${rid}.log" 2>&1 \
      && echo "=== STAGE l5_${rid} PASS" || echo "=== STAGE l5_${rid} FAIL"
  fi
}
l5 deepseek-r1:8b r1-8b-bf16 deepseek-ai--DeepSeek-R1-0528-Qwen3-8B 1,4,16
l5 qwen3-coder:30b-gptq coder-30b-gptq jart25--Qwen3-Coder-30B-A3B-Instruct-Int4-gptq 1,4,16,32
l5 deepseek-r1:32b-gptq r1-32b-gptq OPEA--DeepSeek-R1-Distill-Qwen-32B-int4-gptq-sym-inc 1,4,16,32
l5 qwen3:32b-gptq qwen3-32b-gptq JunHowie--Qwen3-32B-GPTQ-Int4 1,4,16,32
l5 qwen2.5-coder:32b-gptq qwen25-coder-32b-gptq Qwen--Qwen2.5-Coder-32B-Instruct-GPTQ-Int4 1,4,16,32

# ---- Stage 4: C7/G0 baseline guard (M3 = Qwen3-30B-A3B-GPTQ) ----
l5 Qwen/Qwen3-30B-A3B-GPTQ-Int4 m3-baseline Qwen--Qwen3-30B-A3B-GPTQ-Int4 32 || true

pkill -f "ferrum serve" 2>/dev/null
echo "=== ALL SINGLE-POD GATES DONE"
ls -la "$G"
