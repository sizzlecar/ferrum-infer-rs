#!/usr/bin/env bash
set -euo pipefail
ART="/workspace/artifacts/w3_qwen35_cuda_source_check_a4bbc933_20260624T121458Z"
REPO=/workspace/ferrum-infer-rs-git
cd "$REPO"
mkdir -p "$ART"/{logs,build,env,hardware}
{
  echo started_at=$(date -Is)
  echo repo=$REPO
  echo art=$ART
  echo head=$(git rev-parse HEAD)
  echo status_start
  git status --short --branch
  echo status_end
  echo nvidia_smi
  nvidia-smi || true
  echo nvcc
  nvcc --version || true
  echo rustc
  source "$HOME/.cargo/env" 2>/dev/null || true
  rustc --version
  cargo --version
} | tee "$ART/logs/preflight.log"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv > "$ART/hardware/nvidia_before.csv" || true
nvcc --version > "$ART/hardware/nvcc_version.txt" 2>&1 || true
git rev-parse HEAD > "$ART/env/git_sha.txt"
git status --short --branch > "$ART/env/git_status_short.txt"
source "$HOME/.cargo/env" 2>/dev/null || true
export CARGO_TERM_COLOR=never
export NO_COLOR=1
set +e
(
  set -x
  cargo check -p ferrum-models --features cuda
) > "$ART/build/cargo_check_ferrum_models_cuda.stdout.log" 2> "$ART/build/cargo_check_ferrum_models_cuda.stderr.log"
RC=$?
echo $RC > "$ART/build/cargo_check_ferrum_models_cuda.exit"
if [ "$RC" -ne 0 ]; then
  echo "CUDA_SOURCE_CHECK_FAIL cargo_check_ferrum_models_cuda rc=$RC art=$ART" | tee "$ART/summary.txt"
  exit "$RC"
fi
(
  set -x
  cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
) > "$ART/build/cargo_build_release_ferrum_cuda.stdout.log" 2> "$ART/build/cargo_build_release_ferrum_cuda.stderr.log"
RC=$?
echo $RC > "$ART/build/cargo_build_release_ferrum_cuda.exit"
if [ "$RC" -ne 0 ]; then
  echo "CUDA_SOURCE_CHECK_FAIL cargo_build_release_ferrum_cuda rc=$RC art=$ART" | tee "$ART/summary.txt"
  exit "$RC"
fi
sha256sum target/release/ferrum > "$ART/env/ferrum.sha256"
(
  set -x
  cargo test -p ferrum-models --features cuda sparse_moe_decode_fused_gate_merge_gates_shared_and_adds_routed -- --nocapture
) > "$ART/build/cargo_test_fused_gate_cuda_feature.stdout.log" 2> "$ART/build/cargo_test_fused_gate_cuda_feature.stderr.log"
RC=$?
echo $RC > "$ART/build/cargo_test_fused_gate_cuda_feature.exit"
if [ "$RC" -ne 0 ]; then
  echo "CUDA_SOURCE_CHECK_FAIL cargo_test_fused_gate_cuda_feature rc=$RC art=$ART" | tee "$ART/summary.txt"
  exit "$RC"
fi
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv > "$ART/hardware/nvidia_after.csv" || true
echo "CUDA_SOURCE_CHECK_PASS: $ART" | tee "$ART/summary.txt"
