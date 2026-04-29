#!/usr/bin/env bash
# scripts/bench_m1_q4_k_m.sh
#
# M1 Max Q4_K_M head-to-head: ferrum vs mistral.rs vs llama.cpp.
# Single-stream decode benchmark (the sanity check before more workloads).
#
# Layout:
#   scripts/bench_m1_q4_k_m.sh /path/to/models_dir /path/to/tokenizers_dir
#
# `models_dir` is expected to contain (one or more of) the three target
# GGUFs:
#   - Qwen3-8B-Q4_K_M.gguf
#   - Llama-3.1-8B-Instruct-Q4_K_M.gguf
#   - Qwen3-30B-A3B-Q4_K_M.gguf
#
# `tokenizers_dir` should mirror the same names with `.tokenizer.json`
# suffix (download from the matching HF model card):
#   - Qwen3-8B.tokenizer.json
#   - Llama-3.1-8B-Instruct.tokenizer.json
#   - Qwen3-30B-A3B.tokenizer.json
#
# Output:
#   - bench_results/<UTC-timestamp>/<model>-<engine>.log  (raw output)
#   - bench_results/<UTC-timestamp>/summary.md            (markdown table)
#
# Engines that aren't installed are skipped with a clear message — the
# script never fails because mistralrs or llama-cli isn't present.

set -uo pipefail

MODELS_DIR="${1:?usage: bench_m1_q4_k_m.sh <models_dir> <tokenizers_dir>}"
TOKENIZERS_DIR="${2:?usage: bench_m1_q4_k_m.sh <models_dir> <tokenizers_dir>}"

# Workload defaults — keep small enough that 30B-A3B finishes in <5min.
PROMPT="${BENCH_PROMPT:-Explain the theory of relativity in two sentences.}"
MAX_TOKENS="${BENCH_MAX_TOKENS:-256}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FERRUM_BIN="${FERRUM_BIN:-$REPO_ROOT/target/release/ferrum}"

if [ ! -x "$FERRUM_BIN" ]; then
  echo "Building ferrum CLI (release, --features metal) ..."
  (cd "$REPO_ROOT" && cargo build -p ferrum-cli --bin ferrum --features metal --release)
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="$REPO_ROOT/bench_results/$stamp"
mkdir -p "$out_dir"

summary="$out_dir/summary.md"
{
  echo "# M1 Max Q4_K_M benchmark — $stamp"
  echo
  echo "Prompt: \`$PROMPT\`"
  echo "Max tokens: $MAX_TOKENS"
  echo
  echo "| Model | Engine | Decode tok/s | Prefill (s) | Decode (s) |"
  echo "|---|---|---|---|---|"
} > "$summary"

models=(
  "Qwen3-8B-Q4_K_M.gguf::Qwen3-8B.tokenizer.json"
  "Llama-3.1-8B-Instruct-Q4_K_M.gguf::Llama-3.1-8B-Instruct.tokenizer.json"
  "Qwen3-30B-A3B-Q4_K_M.gguf::Qwen3-30B-A3B.tokenizer.json"
)

# Helper: extract "throughput: NN.N tok/s" from ferrum stderr.
extract_ferrum_decode_tps() { awk '/throughput:/ { for (i=1;i<=NF;i++) if ($i+0 == $i && $i+0 > 0) { print $i; exit } }' "$1"; }
extract_ferrum_prefill_secs() { awk '/time:/ { match($0, /([0-9.]+)s prefill/, m); if (m[1]) { print m[1]; exit } }' "$1"; }
extract_ferrum_decode_secs() { awk '/time:/ { match($0, /([0-9.]+)s decode/, m); if (m[1]) { print m[1]; exit } }' "$1"; }

run_ferrum() {
  local gguf="$1"
  local tok="$2"
  local model_label="$3"
  local log="$out_dir/$model_label-ferrum.log"
  echo
  echo "=== ferrum: $model_label ==="
  if "$FERRUM_BIN" run "$gguf" --tokenizer "$tok" --prompt "$PROMPT" \
      --max-tokens "$MAX_TOKENS" --backend metal --bench-mode --temperature 0 \
      2>"$log" 1>/dev/null; then
    local tps prefill decode
    tps=$(extract_ferrum_decode_tps "$log")
    prefill=$(extract_ferrum_prefill_secs "$log")
    decode=$(extract_ferrum_decode_secs "$log")
    printf "| %s | ferrum | %s | %s | %s |\n" "$model_label" "${tps:-?}" "${prefill:-?}" "${decode:-?}" >> "$summary"
    echo "  tok/s = ${tps:-?}, prefill = ${prefill:-?}s, decode = ${decode:-?}s"
  else
    echo "  ferrum FAILED — see $log"
    printf "| %s | ferrum | FAIL | — | — |\n" "$model_label" >> "$summary"
  fi
}

run_mistralrs() {
  local gguf="$1"
  local tok="$2"
  local model_label="$3"
  if ! command -v mistralrs >/dev/null 2>&1; then
    echo "  (mistralrs not installed — skipping)"
    printf "| %s | mistral.rs | skipped | — | — |\n" "$model_label" >> "$summary"
    return
  fi
  local log="$out_dir/$model_label-mistralrs.log"
  echo
  echo "=== mistral.rs: $model_label ==="
  # mistral.rs CLI has its own subcommand surface; this is a placeholder
  # — the user will plug in the correct invocation once mistralrs is set
  # up locally. Keeping the slot in the summary table.
  printf "| %s | mistral.rs | TODO | — | — |\n" "$model_label" >> "$summary"
  echo "  TODO: wire mistralrs gguf invocation"
}

run_llamacpp() {
  local gguf="$1"
  local _tok="$2"  # llama.cpp uses the embedded tokenizer
  local model_label="$3"
  if ! command -v llama-cli >/dev/null 2>&1; then
    echo "  (llama-cli not installed — \`brew install llama.cpp\` to enable; skipping)"
    printf "| %s | llama.cpp | skipped | — | — |\n" "$model_label" >> "$summary"
    return
  fi
  local log="$out_dir/$model_label-llamacpp.log"
  echo
  echo "=== llama.cpp: $model_label ==="
  # llama-cli prints timing in a `print_timings` block at the end;
  # capture stderr, look for "eval time" and "prompt eval time".
  if llama-cli -m "$gguf" -p "$PROMPT" -n "$MAX_TOKENS" --no-display-prompt 2>"$log" 1>/dev/null; then
    local decode_tps prompt_tps
    decode_tps=$(awk '/eval time/ && !/prompt eval time/ { for (i=1;i<=NF;i++) if ($i ~ /^[0-9.]+$/ && $(i+1) ~ /tokens\/s/) { print $i; exit } }' "$log")
    prompt_tps=$(awk '/prompt eval time/ { for (i=1;i<=NF;i++) if ($i ~ /^[0-9.]+$/ && $(i+1) ~ /tokens\/s/) { print $i; exit } }' "$log")
    printf "| %s | llama.cpp | %s | (prompt %s tok/s) | — |\n" "$model_label" "${decode_tps:-?}" "${prompt_tps:-?}" >> "$summary"
    echo "  decode tok/s = ${decode_tps:-?}, prompt tok/s = ${prompt_tps:-?}"
  else
    echo "  llama-cli FAILED — see $log"
    printf "| %s | llama.cpp | FAIL | — | — |\n" "$model_label" >> "$summary"
  fi
}

for entry in "${models[@]}"; do
  gguf_name="${entry%%::*}"
  tok_name="${entry##*::}"
  gguf="$MODELS_DIR/$gguf_name"
  tok="$TOKENIZERS_DIR/$tok_name"

  if [ ! -f "$gguf" ]; then
    echo "skipping $gguf_name (not found at $gguf)"
    continue
  fi
  if [ ! -f "$tok" ]; then
    echo "skipping $gguf_name (tokenizer missing at $tok)"
    continue
  fi

  model_label="${gguf_name%.gguf}"

  run_ferrum "$gguf" "$tok" "$model_label"
  run_mistralrs "$gguf" "$tok" "$model_label"
  run_llamacpp "$gguf" "$tok" "$model_label"
done

echo
echo "Done. Summary at: $summary"
echo
cat "$summary"
