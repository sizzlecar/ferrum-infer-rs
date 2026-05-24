#!/usr/bin/env bash
#
# lm-eval-light — PLAYBOOK § 3 L4.
#
# Drives EleutherAI's lm-evaluation-harness against a running ferrum
# server's /v1/completions endpoint. Runs three small task slices:
#
#   - mmlu     100 questions, 5-shot   (general knowledge)
#   - arc_easy 100 questions, 0-shot   (commonsense reasoning)
#   - gsm8k    50  questions, 5-shot   (chain-of-thought math)
#
# Total wall-clock ≈ 10 min on Metal qwen3:0.6b. Compares accuracy vs a
# committed baseline at `fixtures/lm_eval_baseline.json` with rtol=0.05
# (matches vLLM's `.buildkite/lm-eval-harness/` thresholding pattern).
#
# Usage:
#   scripts/lm_eval_light.sh <model>
#   scripts/lm_eval_light.sh qwen3:0.6b
#   scripts/lm_eval_light.sh --baseline path/to/baseline.json qwen3:0.6b
#   scripts/lm_eval_light.sh --refresh-baseline qwen3:0.6b      # write new baseline
#
# Env:
#   LM_EVAL_PORT     port to start ferrum serve on (default: random in 18000-19000)
#   LM_EVAL_RTOL     relative tolerance for baseline drift (default: 0.05)
#   FERRUM_BIN       ferrum binary path (default: ./target/release/ferrum)
#   PYTHON           python3 binary (default: python3)

set -euo pipefail

REFRESH_BASELINE=0
BASELINE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --refresh-baseline) REFRESH_BASELINE=1; shift ;;
        --baseline) BASELINE="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) break ;;
    esac
done

if [ $# -lt 1 ]; then
    echo "usage: $0 <model> [--baseline PATH] [--refresh-baseline]" >&2
    exit 1
fi

MODEL="$1"
RTOL="${LM_EVAL_RTOL:-0.05}"
PYTHON="${PYTHON:-python3}"
FERRUM_BIN="${FERRUM_BIN:-./target/release/ferrum}"
REPO_ROOT="$(git rev-parse --show-toplevel)"

if [ -z "$BASELINE" ]; then
    BASELINE="$REPO_ROOT/fixtures/lm_eval_baseline.json"
fi

if [ ! -x "$FERRUM_BIN" ]; then
    echo "ERROR: ferrum binary missing at $FERRUM_BIN" >&2
    echo "  build: cargo build --release -p ferrum-cli --features metal" >&2
    exit 1
fi

# Confirm lm-eval is installed.
if ! "$PYTHON" -c "import lm_eval" 2>/dev/null; then
    echo "ERROR: lm_eval not installed in $PYTHON. Run:" >&2
    echo "  pip install lm-eval" >&2
    exit 1
fi

PORT="${LM_EVAL_PORT:-$((RANDOM % 1000 + 18000))}"
OUT_DIR="$REPO_ROOT/docs/bench/lm_eval/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

echo "▶ starting ferrum serve $MODEL on port $PORT"
"$FERRUM_BIN" serve "$MODEL" --port "$PORT" \
    > "$OUT_DIR/serve.log" 2>&1 &
SERVE_PID=$!

cleanup() {
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for /health
i=0
while ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 180 ]; then
        echo "ERROR: server did not become healthy within 3 minutes" >&2
        exit 1
    fi
    sleep 1
done
echo "  ready ($i s)"

# Run the three task slices. `lm-eval` writes one results.json per
# --tasks invocation; we run them separately to keep the wall-clock
# breakdown legible.
declare -a tasks=(
    "mmlu:5:100"
    "arc_easy:0:100"
    "gsm8k:5:50"
)

# Aggregate results into a single JSON for baseline comparison.
RESULTS="$OUT_DIR/results.json"
echo "{}" > "$RESULTS"

for spec in "${tasks[@]}"; do
    IFS=":" read -r task n_shot limit <<< "$spec"
    echo
    echo "▶ lm-eval task=$task n_shot=$n_shot limit=$limit"
    "$PYTHON" -m lm_eval \
        --model local-completions \
        --model_args "base_url=http://127.0.0.1:${PORT}/v1/completions,model=${MODEL},tokenizer_backend=huggingface" \
        --tasks "$task" \
        --num_fewshot "$n_shot" \
        --limit "$limit" \
        --output_path "$OUT_DIR/${task}" \
        --apply_chat_template 2>&1 \
        | tee -a "$OUT_DIR/eval.log"
done

# Aggregate accuracies into one JSON for compare/baseline.
"$PYTHON" - "$OUT_DIR" "$RESULTS" <<'PY'
import glob, json, os, sys
out_dir, results_path = sys.argv[1], sys.argv[2]
agg = {}
# lm-eval writes results to <out_dir>/<task>/results_*.json or <task>/results.json
for task_dir in glob.glob(os.path.join(out_dir, "*")):
    if not os.path.isdir(task_dir):
        continue
    task = os.path.basename(task_dir)
    if task == "results":
        continue
    # Find the most recent results json.
    candidates = sorted(glob.glob(os.path.join(task_dir, "**", "results*.json"), recursive=True))
    if not candidates:
        continue
    with open(candidates[-1]) as f:
        data = json.load(f)
    # Typical key path: data["results"][<task_name>]["acc,none"] or similar.
    task_results = data.get("results", {})
    for tname, metrics in task_results.items():
        # Prefer normalized accuracy if available, else accuracy.
        for k in ("acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,flexible-extract"):
            if k in metrics:
                agg[f"{task}/{tname}/{k}"] = metrics[k]
with open(results_path, "w") as f:
    json.dump(agg, f, indent=2, sort_keys=True)
PY

echo
echo "▶ aggregated → $RESULTS"
cat "$RESULTS"

# Compare or refresh baseline.
if [ "$REFRESH_BASELINE" = "1" ]; then
    mkdir -p "$(dirname "$BASELINE")"
    cp "$RESULTS" "$BASELINE"
    echo
    echo "✓ refreshed baseline: $BASELINE"
    exit 0
fi

if [ ! -f "$BASELINE" ]; then
    echo
    echo "  no baseline at $BASELINE — first run."
    echo "  to commit this run as the baseline:"
    echo "    cp $RESULTS $BASELINE"
    exit 0
fi

echo
echo "▶ diff vs baseline ($BASELINE, rtol=$RTOL)"
"$PYTHON" - "$BASELINE" "$RESULTS" "$RTOL" <<'PY'
import json, sys
base = json.load(open(sys.argv[1]))
meas = json.load(open(sys.argv[2]))
rtol = float(sys.argv[3])

drifted = []
for k, v in base.items():
    if k not in meas:
        drifted.append((k, v, None, "missing"))
        continue
    m = meas[k]
    if v == 0:
        continue
    rel = abs(m - v) / abs(v)
    if rel > rtol:
        drifted.append((k, v, m, f"rel={rel:.3f} > rtol={rtol}"))

if drifted:
    print(f"\n  ⚠ {len(drifted)} task(s) drifted beyond rtol={rtol}:")
    for k, v, m, why in drifted:
        print(f"    {k}: baseline={v} measured={m} ({why})")
    sys.exit(1)
print(f"\n✓ all {len(base)} task slices within rtol={rtol}.")
PY

echo
echo "  results dir: $OUT_DIR"
