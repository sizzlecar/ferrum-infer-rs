#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /workspace/ferrum-infer-rs-git
cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
sha256sum target/release/ferrum > /workspace/artifacts/w3_qwen35_moe_zero_skip_55368e57_20260624/setup/ferrum_after.sha256
