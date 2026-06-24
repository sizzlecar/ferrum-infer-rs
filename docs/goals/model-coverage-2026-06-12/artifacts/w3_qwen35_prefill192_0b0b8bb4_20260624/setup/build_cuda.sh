#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /workspace/ferrum-infer-rs-git
cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
sha256sum target/release/ferrum
