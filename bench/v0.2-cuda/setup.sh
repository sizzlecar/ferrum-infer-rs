#!/usr/bin/env bash
# v0.2 CUDA bench — one-shot pod setup.
#
# Idempotent: re-running is safe (skips work that's done).
#
# Usage on a fresh RunPod / vast.ai RTX 4090 pod with PyTorch+CUDA:
#   export HF_TOKEN=hf_...      # required for Llama-3.1-8B (gated)
#   bash bench/v0.2-cuda/setup.sh
#
# Stages:
#   1. apt deps (build-essential, git, jq, rsync) — for source builds
#   2. Rust 1.91 — ferrum builds against this
#   3. Clone ferrum, cargo build --release --features cuda
#   4. pip install vllm==<pin>
#   5. cargo install mistralrs-server --version <pin>
#   6. parallel HF download of all 4 models to /workspace/models
#   7. download ShareGPT v3 + generate prompts.json
#   8. download vllm benchmark_serving.py
#
# Estimated wall: 40-60 min on a fresh pod. The dominant phases are
# ferrum CUDA kernel compile (~20 min) and the 4 parallel model
# downloads (~10-30 min depending on bandwidth).

set -euo pipefail

# ── pins (synced with versions.txt) ──────────────────────────────────
FERRUM_REF="${FERRUM_REF:-bench/v0.2-cuda}"
VLLM_VERSION="${VLLM_VERSION:-0.20.0}"
# mistralrs dropped from v0.2 scope — see bench/v0.2-cuda/models.txt
# VLLM_BENCH_TAG no longer needed: 0.10+ ships `vllm bench serve` CLI

# ── paths ────────────────────────────────────────────────────────────
WORKSPACE="${WORKSPACE:-/workspace}"
MODELS_DIR="${WORKSPACE}/models"
DATASETS_DIR="${WORKSPACE}/datasets"
FERRUM_DIR="${WORKSPACE}/ferrum-infer-rs"
BENCH_DIR="${FERRUM_DIR}/bench/v0.2-cuda"
mkdir -p "$MODELS_DIR" "$DATASETS_DIR"

log()  { printf '\n\033[1;36m[setup] %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  ⚠ %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m  ✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ── 1. apt deps ──────────────────────────────────────────────────────
log "[1/8] apt deps"
if ! command -v jq >/dev/null || ! command -v git >/dev/null || ! command -v rsync >/dev/null; then
  apt-get update -q
  apt-get install -y --no-install-recommends \
    build-essential git jq rsync curl ca-certificates pkg-config libssl-dev cmake
fi
ok "apt deps in place"

# ── 2. Rust ──────────────────────────────────────────────────────────
log "[2/8] rust toolchain"
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
ok "$(cargo --version)"

# ── 3. ferrum clone + build ──────────────────────────────────────────
log "[3/8] ferrum clone + cargo build --release --features cuda (~20 min CUDA kernel compile)"
if [[ ! -d "$FERRUM_DIR" ]]; then
  git clone https://github.com/sizzlecar/ferrum-infer-rs.git "$FERRUM_DIR"
fi
cd "$FERRUM_DIR"
git fetch --tags --quiet
git checkout "$FERRUM_REF" --quiet

# CUDA_HOME is required by ferrum-kernels' build.rs (NVCC location).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
[[ -d "$CUDA_HOME" ]] || die "CUDA_HOME=$CUDA_HOME does not exist; image is probably wrong"
export PATH="$CUDA_HOME/bin:$PATH"

if [[ ! -x "$FERRUM_DIR/target/release/ferrum" ]]; then
  cargo build --release -p ferrum-cli --bin ferrum --features cuda
fi
"$FERRUM_DIR/target/release/ferrum" --version | tee /dev/stderr
ok "ferrum built"

# ── 4. vLLM ──────────────────────────────────────────────────────────
log "[4/8] vLLM ${VLLM_VERSION}"
pip install --quiet "vllm==${VLLM_VERSION}"
python -c "import vllm; print('  vllm:', vllm.__version__)"
ok "vllm installed"

# ── 5. (mistralrs dropped from v0.2 scope) ───────────────────────────
log "[5/7] mistralrs skipped (dropped from v0.2 scope)"

# ── 6. parallel HF model downloads ───────────────────────────────────
log "[6/7] HF model downloads (parallel, 3 jobs)"
[[ -n "${HF_TOKEN:-}" ]] || warn "HF_TOKEN not set — gated repos will fail"
pip install --quiet -U "huggingface_hub[cli]"

download_one() {
  local tag="$1" repo="$2"
  local target="$MODELS_DIR/$tag"
  if [[ -d "$target" && -n "$(ls "$target" 2>/dev/null)" ]]; then
    echo "[$tag] $repo — already downloaded, skip"
    return 0
  fi
  mkdir -p "$target"
  echo "[$tag] downloading $repo to $target"
  # `hf` CLI (huggingface_hub v1.x). Also caches under HF_HOME by
  # hardlink, so deleting `--local-dir` doesn't waste cache space.
  hf download "$repo" --local-dir "$target" --quiet \
    ${HF_TOKEN:+--token "$HF_TOKEN"} || {
      echo "[$tag] FAILED" >&2
      return 1
    }
  echo "[$tag] done ($(du -sh "$target" | cut -f1))"
}
export -f download_one
export MODELS_DIR HF_TOKEN

# Read pinned models from models.txt (skip comments / blank lines).
DL_PIDS=()
while IFS='|' read -r tag repo precision size; do
  [[ -z "$tag" || "$tag" =~ ^# ]] && continue
  download_one "$tag" "$repo" &
  DL_PIDS+=($!)
done < "$BENCH_DIR/models.txt"

DL_FAILED=0
for pid in "${DL_PIDS[@]}"; do
  wait "$pid" || DL_FAILED=$((DL_FAILED+1))
done
[[ $DL_FAILED -gt 0 ]] && die "$DL_FAILED model download(s) failed"
df -h "$WORKSPACE" | tail -1
ok "all models downloaded"

# ── 7. ShareGPT + prompts.json + bench harness ──────────────────────
log "[7/7] ShareGPT subset + verify \`vllm bench serve\`"
SG_FILE="$DATASETS_DIR/ShareGPT_V3_unfiltered_cleaned_split.json"
if [[ ! -f "$SG_FILE" ]]; then
  hf download anon8231489123/ShareGPT_Vicuna_unfiltered \
    ShareGPT_V3_unfiltered_cleaned_split.json \
    --repo-type dataset --local-dir "$DATASETS_DIR" --quiet
fi
SEED="$(git -C "$FERRUM_DIR" rev-parse --short HEAD)"
python3 "$BENCH_DIR/prompts_subset.py" \
  --input "$SG_FILE" \
  --output "$BENCH_DIR/prompts.json" \
  --seed "$SEED"
ok "prompts.json built (seed=$SEED)"

# vLLM 0.10+ ships `vllm bench serve` CLI (replaces the deprecated
# standalone benchmark_serving.py). We pip-installed vllm above, just
# verify the CLI is wired.
vllm bench serve --help > /tmp/vllm_bench_help.txt 2>&1 || \
  die "\`vllm bench serve --help\` failed — vllm install is broken"
HELP_LINES=$(wc -l < /tmp/vllm_bench_help.txt)
[[ "$HELP_LINES" -lt 30 ]] && die "vllm bench serve help output suspiciously short ($HELP_LINES lines)"
ok "vllm bench serve CLI available ($HELP_LINES help lines)"

# ── final environment fingerprint ────────────────────────────────────
log "writing _env.txt"
{
  echo "=== v0.2 CUDA bench environment fingerprint ==="
  echo "captured: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "--- hardware ---"
  nvidia-smi -q 2>&1 | grep -E "Product Name|Driver Version|CUDA Version|Total " | head
  echo
  echo "--- cuda toolkit ---"
  nvcc --version 2>&1 | tail -3 || true
  echo
  echo "--- cpu / ram ---"
  grep -m1 "model name" /proc/cpuinfo | awk -F: '{print $2}' | xargs
  echo "cpu cores: $(nproc)"
  echo "ram: $(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')"
  echo
  echo "--- engine versions ---"
  echo "ferrum: $($FERRUM_DIR/target/release/ferrum --version)"
  echo "ferrum git: $(git -C $FERRUM_DIR rev-parse HEAD) ($FERRUM_REF)"
  echo "vllm: $(python -c 'import vllm; print(vllm.__version__)')"
  echo "rust: $(cargo --version)"
  echo
  echo "--- model HF revisions ---"
  for d in "$MODELS_DIR"/*/; do
    if [[ -f "$d/config.json" ]]; then
      echo "$(basename "$d"): $(du -sh "$d" | cut -f1) — $(jq -r '._name_or_path // "?"' "$d/config.json" 2>/dev/null)"
    fi
  done
  echo
  echo "--- bench harness ---"
  echo "bench harness: vllm bench serve (CLI from installed vLLM ${VLLM_VERSION})"
  echo "prompts.json: $(jq -r '.count' "$BENCH_DIR/prompts.json") prompts, seed=$(jq -r '.seed' "$BENCH_DIR/prompts.json")"
} > "$BENCH_DIR/_env.txt"
cat "$BENCH_DIR/_env.txt"

echo
echo "─────────────────────────────────────────────────────"
echo " setup complete in $SECONDS s"
echo " next: bash $BENCH_DIR/run_sweep.sh"
echo "─────────────────────────────────────────────────────"
