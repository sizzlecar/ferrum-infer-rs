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

GGUF="$GGUF_DIR/$MODEL"
TOK_PATH="$TOK_DIR/$TOK"

mkdir -p "$OUTDIR"
TAG="${MODEL%.gguf}"

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
  llama-bench -m "$GGUF" -p 50,512 -n 128 -r 3 2>&1
} | tee "$OUTDIR/${TAG}__llamacpp.txt" >/dev/null
echo "    written → ${TAG}__llamacpp.txt"

sleep 2

# ── ferrum (3 separate runs per op — fresh process for clean memory) ──
echo ">>> ferrum (9 runs: pp50×3, pp512×3, tg128×3)"
{
  echo "# ferrum bench output for $MODEL"
  date

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
} | tee "$OUTDIR/${TAG}__ferrum.txt" >/dev/null
echo "    written → ${TAG}__ferrum.txt"

sleep 2

# ── mistral.rs (Python wheel, 3 separate processes for clean memory) ──
echo ">>> mistral.rs (3 runs: pp50, pp512, tg128 — each runs 3 trials internally)"
{
  echo "# mistral.rs Python bench output for $MODEL"
  date

  echo "--- pp50 ---"
  GGUF_DIR="$GGUF_DIR" "$MISTRAL_PY" "$BENCH_PY" "$MODEL" pp 50 1 3 2>/dev/null

  echo "--- pp512 ---"
  GGUF_DIR="$GGUF_DIR" "$MISTRAL_PY" "$BENCH_PY" "$MODEL" pp 512 1 3 2>/dev/null

  echo "--- tg128 ---"
  GGUF_DIR="$GGUF_DIR" "$MISTRAL_PY" "$BENCH_PY" "$MODEL" tg 0 128 3 2>/dev/null
} | tee "$OUTDIR/${TAG}__mistralrs.txt" >/dev/null
echo "    written → ${TAG}__mistralrs.txt"

echo "=== ${MODEL} done ==="
date
