#!/usr/bin/env bash
#
# pod_setup.sh — Vast 4090 pod setup for ferrum vs vLLM parity bench.
#
# Run this once per pod. Idempotent — safe to re-run after partial
# failures (apt + cargo + pip all support resume / skip-if-done).
#
# Wraps each long-running step in `timeout` so a stuck step never
# burns rented GPU minutes (PLAYBOOK / feedback_bench_hang_guard).
#
# Usage:
#   bash pod_setup.sh
#
# Expects:
#   - Image: nvidia/cuda:12.4.0-devel-ubuntu22.04
#   - $BRANCH env var (default: phase-0-testing-system)

set -euo pipefail

BRANCH="${BRANCH:-phase-0-testing-system}"
REPO_URL="${REPO_URL:-https://github.com/sizzlecar/ferrum-infer-rs.git}"

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

log "▶ apt update + tooling"
export DEBIAN_FRONTEND=noninteractive
timeout 600 apt-get update -qq
timeout 1200 apt-get install -y -qq \
    curl git build-essential pkg-config libssl-dev cmake \
    python3-venv python3-pip jq bc \
    nvidia-utils-535 || true   # nvidia-smi sometimes missing in :devel images

log "▶ Rust toolchain"
if [ ! -d "$HOME/.cargo" ]; then
    timeout 300 curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
rustc --version

log "▶ git clone"
mkdir -p /workspace
cd /workspace
if [ ! -d ferrum-infer-rs ]; then
    timeout 300 git clone "$REPO_URL"
fi
cd ferrum-infer-rs
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
log "  on $(git rev-parse --short HEAD)"

log "▶ Python venv + vLLM"
if [ ! -d "/workspace/vllm-venv" ]; then
    python3 -m venv /workspace/vllm-venv
fi
# shellcheck disable=SC1091
source /workspace/vllm-venv/bin/activate
timeout 60 pip install -U -q pip wheel
# vllm 0.20.x matches the baseline in CLAUDE.md
timeout 1800 pip install -q 'vllm==0.20.2' 'huggingface_hub[cli]'
vllm --version || true

log "▶ HF model download (qwen3:0.6b)"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
mkdir -p "$HF_HOME"
timeout 600 huggingface-cli download Qwen/Qwen3-0.6B --quiet || true

log "▶ ferrum release build (cuda)"
cd /workspace/ferrum-infer-rs
# CUDA 13+ toolchains don't infer the GPU's compute capability from
# `nvidia-smi`; candle-kernels' build script needs it explicit. 4090 is
# sm_89. Without this, the build fails at `flash_fwd_hdim*_sm80.cu`.
export CUDA_COMPUTE_CAP=89
# Don't pre-fetch all features — just cuda for the bench. vllm-moe-marlin
# requires more build time and we're benching qwen3:0.6b (no MoE).
timeout 1800 cargo build --release -p ferrum-cli --features cuda 2>&1 \
    | grep -E "Compiling|Finished|error" || true

if [ ! -x ./target/release/ferrum ]; then
    log "ERROR: ferrum binary not built"
    exit 1
fi

log "✓ pod setup complete"
log "  ferrum: $(./target/release/ferrum --version)"
log "  vllm:   $(vllm --version 2>/dev/null | head -1)"
log ""
log "next: bash scripts/pod_bench.sh"
