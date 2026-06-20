#!/usr/bin/env bash
set -euo pipefail
if [ -f "$HOME/.cargo/env" ]; then
  . "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:$PATH"
cd /workspace/ferrum-clean
ART="${ART:?}"
{
  echo "== git =="
  git rev-parse HEAD
  git status --short --branch
  echo "== toolchain =="
  rustc --version
  cargo --version
  nvcc --version | tail -n 1
  echo "== cargo check ferrum-kernels cuda =="
  cargo check -p ferrum-kernels --features cuda --all-targets
  echo "== cargo check ferrum-cli cuda product features =="
  cargo check -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
  echo "== focused qwen35 varlen reference test =="
  cargo test -p ferrum-models recurrent_delta_rule_varlen_backend_matches_per_sequence_reference -- --nocapture
  echo "W3 QWEN35 VARLEN GDN CUDA BUILD SMOKE PASS: $ART"
} 2>&1 | tee "$ART/command.log"
