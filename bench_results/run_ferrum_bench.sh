#!/usr/bin/env bash
# Run ferrum's pp512 / tg128 equivalents on the three Q4_K_M models we
# already have llama.cpp baselines for. Mirrors llama-bench's protocol:
#   pp512 — feed a ~512 token prompt, measure prompt-processing rate.
#   tg128 — short prompt, generate 128 tokens, measure decode rate.
# 1 warm-up + 5 timed runs per cell. Reports mean ± stddev.
set -uo pipefail

MODELS_DIR="$HOME/ferrum-bench/models"
TOKENIZERS_DIR="$HOME/ferrum-bench/tokenizers"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"
FERRUM="$(cd "$OUT_DIR/.." && pwd)/target/release/ferrum"

# 512-token-ish English text (paragraph repeated to overflow a single
# 512-token window for any common tokenizer). Each model's tokenizer
# slices it slightly differently; ferrum reports `prompt: N tokens` so
# we trust that and divide by the true N.
LONG_PROMPT="The history of artificial intelligence dates back to antiquity, with myths and stories of artificial beings endowed with intelligence by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. The field of AI research was founded at a workshop held on the campus of Dartmouth College, USA during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. Eventually, it became obvious that they had grossly underestimated the difficulty of the project. In 1973, in response to the criticism from James Lighthill and ongoing pressure from congress, the U.S. and British Governments stopped funding undirected research into artificial intelligence, and the difficult years that followed would later be known as an AI winter. Seven years later, a visionary initiative by the Japanese Government inspired governments and industry to provide AI with billions of dollars, but by the late 1980s the investors became disillusioned and withdrew funding again."

run_one() {
    local label="$1"; local model_file="$2"; local tokenizer="$3"
    local mode="$4"; local prompt="$5"; local max_tokens="$6"

    local out="$OUT_DIR/ferrum_${label}_${mode}.txt"
    : > "$out"
    echo "## $label / $mode"
    echo "## $label / $mode" >> "$out"

    # Pre-warmup once (not timed)
    FERRUM_KV_CAPACITY=1024 "$FERRUM" run "$MODELS_DIR/$model_file" \
        --tokenizer "$TOKENIZERS_DIR/$tokenizer" \
        --prompt "$prompt" \
        --max-tokens "$max_tokens" --temperature 0 --backend metal \
        --bench-mode > /dev/null 2>> "$out"

    # 5 timed runs
    for rep in 1 2 3 4 5; do
        local raw
        raw=$(FERRUM_KV_CAPACITY=1024 "$FERRUM" run "$MODELS_DIR/$model_file" \
            --tokenizer "$TOKENIZERS_DIR/$tokenizer" \
            --prompt "$prompt" \
            --max-tokens "$max_tokens" --temperature 0 --backend metal \
            --bench-mode 2>&1)
        # Pull `prompt: N tokens`, prefill seconds, decode seconds from output
        local p_n p_s d_n d_s
        p_n=$(echo "$raw" | sed -n 's/^→ prompt: \([0-9]*\) tokens$/\1/p' | head -1)
        p_s=$(echo "$raw" | sed -n 's/^✓ prefill: [0-9]* tok in \([0-9.]*\)s.*$/\1/p' | head -1)
        d_n=$(echo "$raw" | sed -n 's/^tokens: [0-9]* prompt + \([0-9]*\) generated tok$/\1/p' | head -1)
        d_s=$(echo "$raw" | sed -n 's/^time: [0-9.]*s prefill + \([0-9.]*\)s decode$/\1/p' | head -1)
        echo "rep=$rep p_n=$p_n p_s=$p_s d_n=$d_n d_s=$d_s" | tee -a "$out"
    done

    echo ""
}

# pp512: long prompt, decode just 1 token
run_one "qwen3_8b"        "Qwen3-8B-Q4_K_M.gguf"                  "Qwen3-8B.tokenizer.json"                       "pp512" "$LONG_PROMPT" 1
run_one "qwen3_8b"        "Qwen3-8B-Q4_K_M.gguf"                  "Qwen3-8B.tokenizer.json"                       "tg128" "Hi" 128

run_one "llama31_8b"      "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" "Meta-Llama-3.1-8B-Instruct.tokenizer.json"   "pp512" "$LONG_PROMPT" 1
run_one "llama31_8b"      "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" "Meta-Llama-3.1-8B-Instruct.tokenizer.json"   "tg128" "Hi" 128

# Qwen3-30B-A3B (MoE) — wired through Qwen3MoeModel<MetalBackend> as
# of PR #35. Per-(token, expert) gemv loop, target ~40-50 tok/s decode
# vs llama.cpp's 44.52 tok/s baseline.
run_one "qwen3_30b_a3b"   "Qwen3-30B-A3B-Q4_K_M.gguf"             "Qwen3-30B-A3B.tokenizer.json"                  "pp512" "$LONG_PROMPT" 1 || true
run_one "qwen3_30b_a3b"   "Qwen3-30B-A3B-Q4_K_M.gguf"             "Qwen3-30B-A3B.tokenizer.json"                  "tg128" "Hi" 128 || true

echo "ALL DONE"
