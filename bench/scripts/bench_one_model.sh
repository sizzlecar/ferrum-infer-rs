#!/bin/bash
# Run all 3 engines (ferrum / mistral.rs / llama.cpp) sequentially on ONE model.
# Strict serial — only one inference process at a time. No background work.
# Each engine's raw output is written under $OUTDIR/<model>__<engine>.txt.
#
# Args:
#   $1 = gguf filename (e.g. Qwen3-8B-Q4_K_M.gguf)
#   $2 = tokenizer filename (e.g. Qwen3-8B.tokenizer.json)
#   $3 = output dir (will be created)
#
# Env vars (with defaults from the original M1 Max run):
#   GGUF_DIR      — directory holding the GGUF files
#   TOK_DIR       — directory holding tokenizer.json files
#   FERRUM_BIN    — path to the ferrum CLI release binary
#   MISTRAL_PY    — Python interpreter inside the venv with mistralrs-metal installed
#   BENCH_PY      — path to bench_mistral.py (defaults to sibling of this script)

set -u

MODEL="$1"
TOK="$2"
OUTDIR="$3"

GGUF_DIR="${GGUF_DIR:-/Users/chejinxuan/ferrum-bench/models}"
TOK_DIR="${TOK_DIR:-/Users/chejinxuan/ferrum-bench/tokenizers}"
FERRUM_BIN="${FERRUM_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/target/release/ferrum}"
MISTRAL_PY="${MISTRAL_PY:-/tmp/mistral_bench/bin/python}"
BENCH_PY="${BENCH_PY:-$(cd "$(dirname "$0")" && pwd)/bench_mistral.py}"
CAPTURE_ENV="$(cd "$(dirname "$0")" && pwd)/capture_env.sh"

GGUF="$GGUF_DIR/$MODEL"
TOK_PATH="$TOK_DIR/$TOK"

mkdir -p "$OUTDIR"
TAG="${MODEL%.gguf}"

# What actually matters for bench accuracy is "did paging happen
# *during* this run", not "is there old swap from before". macOS keeps
# pages in swap once written even if the owning process doesn't touch
# them — those stay there indefinitely until reboot or `sudo purge`.
# As long as the bench's working set fits in (free + inactive) memory,
# old swap is harmless.
#
# So we capture a swap baseline now and compare against the post-run
# value at the bottom. A growth > FERRUM_BENCH_SWAP_GROWTH_MB during
# the run flags the bench as paging-affected (default 256 MB, generous
# enough to absorb noise from background daemons). The script doesn't
# refuse upfront — it records the delta into the result file so
# whoever reads the report can see whether to trust the numbers.
swap_baseline_mb=$(sysctl -n vm.swapusage | awk -F'used = ' '{print $2}' | awk '{print int($1)}')
swap_growth_threshold_mb="${FERRUM_BENCH_SWAP_GROWTH_MB:-256}"
echo "swap baseline at bench start: ${swap_baseline_mb} MB (will warn if grows by >${swap_growth_threshold_mb} MB during the run)" >&2

# Free up inactive pages before the bench so the model load doesn't
# evict them under pressure (which IS the slow path on M1 Max). This
# is a no-op if the system has no inactive pages to compress.
sync 2>/dev/null || true

# Prompt of N space-separated "the" tokens — most BPE tokenizers split this 1-token-per-word.
# Trailing pp output prints actual tokenised length, which is what we use.
PROMPT_50=$(printf 'the %.0s' $(seq 1 50))
PROMPT_512=$(printf 'the %.0s' $(seq 1 512))

echo "=== ${MODEL} ==="
date

# ── llama.cpp (llama-bench: pp + tg in one shot, 3 trials internal) ──
echo ">>> llama.cpp (pp50, pp512, tg128 × 3 trials)"
{
  echo "# llama-bench output for $MODEL"
  date
} > "$OUTDIR/${TAG}__llamacpp.txt"
"$CAPTURE_ENV" "before llama.cpp" "$OUTDIR/${TAG}__llamacpp.txt"
{
  llama-bench -m "$GGUF" -p 50,512 -n 128 -r 3 2>&1
} >> "$OUTDIR/${TAG}__llamacpp.txt"
"$CAPTURE_ENV" "after llama.cpp" "$OUTDIR/${TAG}__llamacpp.txt"
echo "    written → ${TAG}__llamacpp.txt"

sleep 2

# ── ferrum (3 separate runs per op — fresh process for clean memory) ──
echo ">>> ferrum (9 runs: pp50×3, pp512×3, tg128×3)"
{
  echo "# ferrum bench output for $MODEL"
  date
} > "$OUTDIR/${TAG}__ferrum.txt"
"$CAPTURE_ENV" "before ferrum" "$OUTDIR/${TAG}__ferrum.txt"
{
  for i in 1 2 3; do
    echo "--- pp50 trial $i ---"
    "$FERRUM_BIN" run "$GGUF" --tokenizer "$TOK_PATH" \
      --prompt "$PROMPT_50" --max-tokens 1 --temperature 0.0 --bench-mode 2>&1 \
      | grep -E "prefill:|throughput|model ready"
  done

  for i in 1 2 3; do
    echo "--- pp512 trial $i ---"
    "$FERRUM_BIN" run "$GGUF" --tokenizer "$TOK_PATH" \
      --prompt "$PROMPT_512" --max-tokens 1 --temperature 0.0 --bench-mode 2>&1 \
      | grep -E "prefill:|throughput|model ready"
  done

  for i in 1 2 3; do
    echo "--- tg128 trial $i ---"
    "$FERRUM_BIN" run "$GGUF" --tokenizer "$TOK_PATH" \
      --prompt "Once upon a time" --max-tokens 128 --temperature 0.0 --bench-mode 2>&1 \
      | grep -E "prefill:|throughput|model ready"
  done
} >> "$OUTDIR/${TAG}__ferrum.txt"
"$CAPTURE_ENV" "after ferrum" "$OUTDIR/${TAG}__ferrum.txt"
echo "    written → ${TAG}__ferrum.txt"

sleep 2

# ── mistral.rs (Python wheel, 3 separate processes for clean memory) ──
echo ">>> mistral.rs (3 runs: pp50, pp512, tg128 — each runs 3 trials internally)"
{
  echo "# mistral.rs Python bench output for $MODEL"
  date
} > "$OUTDIR/${TAG}__mistralrs.txt"
"$CAPTURE_ENV" "before mistral.rs" "$OUTDIR/${TAG}__mistralrs.txt"
{
  echo "--- pp50 ---"
  GGUF_DIR="$GGUF_DIR" "$MISTRAL_PY" "$BENCH_PY" "$MODEL" pp 50 1 3 2>/dev/null

  echo "--- pp512 ---"
  GGUF_DIR="$GGUF_DIR" "$MISTRAL_PY" "$BENCH_PY" "$MODEL" pp 512 1 3 2>/dev/null

  echo "--- tg128 ---"
  GGUF_DIR="$GGUF_DIR" "$MISTRAL_PY" "$BENCH_PY" "$MODEL" tg 0 128 3 2>/dev/null
} >> "$OUTDIR/${TAG}__mistralrs.txt"
"$CAPTURE_ENV" "after mistral.rs" "$OUTDIR/${TAG}__mistralrs.txt"
echo "    written → ${TAG}__mistralrs.txt"

echo "=== ${MODEL} done ==="
date

# Post-run swap delta check: did the bench cause new paging?
swap_after_mb=$(sysctl -n vm.swapusage | awk -F'used = ' '{print $2}' | awk '{print int($1)}')
swap_delta_mb=$((swap_after_mb - swap_baseline_mb))
echo "swap delta over bench: ${swap_delta_mb} MB (baseline=${swap_baseline_mb}, after=${swap_after_mb})"
if [ "$swap_delta_mb" -gt "$swap_growth_threshold_mb" ]; then
  echo "⚠ ⚠ ⚠ swap grew by ${swap_delta_mb} MB during the run (>${swap_growth_threshold_mb} MB threshold)"
  echo "  Bench numbers in this output are likely paging-affected."
  echo "  → Re-run after closing apps / \`sudo purge\` / reboot for clean numbers."
  # Append a banner to every result file so re-aggregating doesn't
  # silently treat this run as clean data.
  for f in "$OUTDIR/${TAG}__llamacpp.txt" "$OUTDIR/${TAG}__ferrum.txt" "$OUTDIR/${TAG}__mistralrs.txt"; do
    if [ -f "$f" ]; then
      printf "\n## ⚠ PAGING-AFFECTED RUN: swap grew by %s MB (baseline %s → %s)\n" \
        "$swap_delta_mb" "$swap_baseline_mb" "$swap_after_mb" >> "$f"
    fi
  done
fi
