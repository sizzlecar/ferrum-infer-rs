#!/usr/bin/env bash
# W2 Gemma3-27B gate ladder. First failure stops the ladder (Paris rule:
# never benchmark known-garbage output). Run from repo root on the pod.
#
# Outputs under /workspace/w2/gates/:
#   smoke_gemma3-27b-gptq.log      L2 known-answer + L3 mechanics + L4 agent
#   l5_gemma3-27b-gptq_cuda.json   bench-serve sweep c=1/4/16/32
#   llamacpp_tg128.json            llama-bench decode reference
#   ratio.txt                      ferrum c=1 decode vs llama.cpp tg128
set -uxo pipefail
W=/workspace/w2
G=$W/gates
mkdir -p "$G"
cd "$(dirname "$0")/.."

ALIAS=gemma3:27b-gptq
RID=gemma3-27b-gptq
GGUF=$(ls /root/.cache/huggingface/hub/models--unsloth--gemma-3-27b-it-GGUF/snapshots/*/gemma-3-27b-it-Q4_K_M.gguf 2>/dev/null || ls /root/.cache/huggingface/hub/models--unsloth--gemma-3-27b-it-GGUF/snapshots/*/*.gguf | head -1)

python3 - "$G/session_metadata.json" <<'EOF'
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    p = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": p.returncode,
        "stdout": p.stdout.strip(),
        "stderr": p.stderr.strip(),
    }


meta = {
    "schema_version": 1,
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    "lane": "MODEL_COVERAGE_W2 gemma3-27b cuda-gptq",
    "stop_condition": "correctness failure, 8h without gate progress, or PASS/failure artifact collected",
    "correctness_gate": "model_coverage_smoke known-answer 10/10 + L3/L4 smoke PASS",
    "performance_gate": "bench-serve random 256/128 c=1/4/16/32 zero-error + llama.cpp ratio >= 0.5",
    "git_rev_parse_head": run(["git", "rev-parse", "HEAD"]),
    "git_status_short": run(["git", "status", "--short"]),
    "gpu": run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]),
    "nvcc": run(["nvcc", "--version"]),
}
Path(sys.argv[1]).write_text(json.dumps(meta, indent=2) + "\n")
EOF

# ── L2+L3+L4: certification smoke ladder (serve + known-answer 10x +
#    stop/stream mechanics + tools 10x + schema 20x) ──────────────────────
# Gemma3-27B KV is heavy (62 layers x 16 kv heads = 508KB/token fp16);
# the autosizer's pool budget on 24GB is ~2.6GB = ~5.4k token-slots.
# Smoke requests default to max_tokens 2048, so pin a 2-seq x 2560-slot
# layout that exactly fits the pool instead of the 8x512 default split.
FERRUM_SMOKE_LOGIT_DUMP_DIR="$G/logit_dump_smoke" \
SMOKE_REQ_TIMEOUT=180 bash scripts/model_coverage_smoke.sh "$ALIAS" \
  --port 8400 --kv-capacity 2560 --max-seqs 2 \
  2>&1 | tee "$G/smoke_${RID}.log"
cp /tmp/ferrum_w1_smoke_8400.log "$G/serve_smoke_${RID}.log" 2>/dev/null || true
grep -q "SMOKE PASS" "$G/smoke_${RID}.log" || {
  echo "=== W2 GATE FAIL at smoke ladder"
  tail -80 "$G/serve_smoke_${RID}.log" 2>/dev/null || true
  exit 1
}

# ── L5: bench-serve sweep ────────────────────────────────────────────────
# Split into separate serve runs because c=32 needs a tighter memory envelope
# on a 24GB card. The c=32 lane keeps client concurrency at 32 but caps server
# active admission at 16 via --max-num-seqs. Native CUDA evidence showed
# admission caps 31 and 30 still OOM at the same 393-token fp16 KV allocation,
# while cap 16 completes the required 100x3 c=32 client-concurrency cell with
# zero errors. Both runs are required W2 gate evidence.
TOK="$(ls -d /root/.cache/huggingface/hub/models--circulus--gemma-3-27b-it-gptq/snapshots/*)"
l5_run() {
  local sweep="$1" kvcap="$2" seqs="$3" out="$4"
  setsid target/release/ferrum serve --model "$ALIAS" --port 8400 \
    --kv-capacity "$kvcap" --max-num-seqs "$seqs" > "$G/serve_l5_${RID}_${sweep//,/}.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 120); do
    curl -s --max-time 3 http://127.0.0.1:8400/v1/models | grep -q gemma && break
    sleep 5
  done
  target/release/ferrum bench-serve --base-url http://127.0.0.1:8400 \
    --model "$ALIAS" --tokenizer "$TOK" \
    --random-input-len 256 --random-output-len 128 \
    --concurrency-sweep "$sweep" --num-prompts 100 --n-repeats 3 \
    --fail-on-error --require-ci --seed 9271 \
    --out "$out" 2>&1 | tail -3
  kill "$pid" 2>/dev/null || true
  sleep 8
  pkill -f "ferrum serve --model gemma3" 2>/dev/null || true
  sleep 5
}
l5_run "1,4,16" 512 16 "$G/l5_${RID}_cuda_1_4_16.json"
l5_run "32" 400 16 "$G/l5_${RID}_cuda_c32.json"
python3 - "$G/l5_${RID}_cuda_1_4_16.json" "$G/l5_${RID}_cuda_c32.json" "$G/l5_${RID}_cuda.json" <<'EOF' || { echo "=== W2 GATE FAIL at L5"; exit 1; }
import json, sys
cells = json.load(open(sys.argv[1])) + json.load(open(sys.argv[2]))
cells.sort(key=lambda c: c["concurrency"])
json.dump(cells, open(sys.argv[3], "w"), indent=2)
seen = [c["concurrency"] for c in cells]
if seen != [1, 4, 16, 32]:
    print(f"missing required L5 cells: got {seen}")
    sys.exit(1)
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
python3 - "$G/l5_${RID}_cuda.json" "$G/llamacpp_tg128.json" <<'EOF' | tee "$G/ratio.txt" || { echo "=== W2 GATE FAIL at perf floor"; exit 1; }
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
if ratio < 0.5:
    sys.exit(1)
EOF

echo "=== W2 GATES COMPLETE"
