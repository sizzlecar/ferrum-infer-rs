#!/usr/bin/env bash
# W2 Gemma3-27B gate ladder. First failure stops the ladder (Paris rule:
# never benchmark known-garbage output). Run from repo root on the pod.
#
# Outputs under /workspace/w2/gates/:
#   smoke_gemma3-27b-gptq.log   L2 known-answer + L3 mechanics + L4 agent
#   l5_gemma3-27b-gptq_cuda.json  bench-serve sweep c=1/4/16/32
#   llamacpp_tg128.txt          llama-bench decode reference
#   ratio.txt                   ferrum c=1 decode vs llama.cpp tg128
set -uxo pipefail
W=/workspace/w2
G=$W/gates
mkdir -p "$G"
cd "$(dirname "$0")/.."

ALIAS=gemma3:27b-gptq
RID=gemma3-27b-gptq
GGUF=$(ls /root/.cache/huggingface/hub/models--unsloth--gemma-3-27b-it-GGUF/snapshots/*/gemma-3-27b-it-Q4_K_M.gguf 2>/dev/null || ls /root/.cache/huggingface/hub/models--unsloth--gemma-3-27b-it-GGUF/snapshots/*/*.gguf | head -1)

# ── L2+L3+L4: certification smoke ladder (serve + known-answer 10x +
#    stop/stream mechanics + tools 10x + schema 20x) ──────────────────────
SMOKE_REQ_TIMEOUT=180 bash scripts/model_coverage_smoke.sh "$ALIAS" \
  --port 8400 --kv-capacity 512 --max-seqs 32 \
  2>&1 | tee "$G/smoke_${RID}.log"
grep -q "SMOKE PASS" "$G/smoke_${RID}.log" || { echo "=== W2 GATE FAIL at smoke ladder"; exit 1; }

# ── L5: bench-serve sweep ────────────────────────────────────────────────
setsid target/release/ferrum serve --model "$ALIAS" --port 8400 \
  --kv-capacity 512 --max-num-seqs 32 > "$G/serve_l5_${RID}.log" 2>&1 &
SERVE_PID=$!
for i in $(seq 1 120); do
  curl -s --max-time 3 http://127.0.0.1:8400/v1/models | grep -q gemma && break
  sleep 5
done
target/release/ferrum bench-serve --base-url http://127.0.0.1:8400 \
  --model "$ALIAS" \
  --tokenizer "$(ls -d /root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/*)" \
  --concurrency-sweep 1,4,16,32 --num-prompts 100 --n-repeats 3 \
  --out "$G/l5_${RID}_cuda.json" 2>&1 | tail -5
kill "$SERVE_PID" 2>/dev/null || true
sleep 5
python3 - "$G/l5_${RID}_cuda.json" <<'EOF' || { echo "=== W2 GATE FAIL at L5"; exit 1; }
import json, sys
cells = json.load(open(sys.argv[1]))
for c in cells:
    errs = sum(c["errored_per_run"])
    if errs or any(x != 100 for x in c["completed_per_run"]):
        print(f"c={c['concurrency']} unclean: errors={errs} completed={c['completed_per_run']}")
        sys.exit(1)
print("L5 clean:", [(c["concurrency"], round(c["output_throughput_tps"]["mean"],1)) for c in cells])
EOF

# ── llama.cpp same-card decode reference ────────────────────────────────
/workspace/llama.cpp/build/bin/llama-bench -m "$GGUF" -p 0 -n 128 -r 3 -o json \
  > "$G/llamacpp_tg128.json" 2> "$G/llamacpp_tg128.err"
python3 - "$G/l5_${RID}_cuda.json" "$G/llamacpp_tg128.json" <<'EOF' | tee "$G/ratio.txt"
import json, sys
cells = json.load(open(sys.argv[1]))
c1 = next(c for c in cells if c["concurrency"] == 1)
ferrum = c1["output_throughput_tps"]["mean"]
lb = json.load(open(sys.argv[2]))
tg = next(r for r in lb if r.get("n_gen", 0) > 0)
lc = tg["avg_ts"]
ratio = ferrum / lc
verdict = "PASS" if ratio >= 0.5 else "FAIL"
gap = " (known-gap 0.5-0.8x)" if 0.5 <= ratio < 0.8 else ""
print(f"ferrum c=1 decode {ferrum:.1f} tok/s vs llama.cpp tg128 {lc:.1f} tok/s -> ratio {ratio:.3f} {verdict}{gap}")
EOF

echo "=== W2 GATES COMPLETE"
