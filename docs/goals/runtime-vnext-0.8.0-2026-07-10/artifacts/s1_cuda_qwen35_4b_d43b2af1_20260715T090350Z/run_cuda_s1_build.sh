#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/ferrum-infer-rs
artifact=/workspace/artifacts/s1_cuda_qwen35_4b_d43b2af1_20260715T090350Z
log="$artifact/cuda-s1-build.log"

mkdir -p "$artifact"
cd "$repo"
. /root/.cargo/env

{
  date -u +started_at=%Y-%m-%dT%H:%M:%SZ
  printf 'git_sha='
  git rev-parse HEAD
  printf 'git_status='
  git status --short | tr '\n' ','
  printf '\n'
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  nvcc --version | tail -1
  echo 'command=CARGO_BUILD_JOBS=16 cargo build --release -p ferrum-cli --bin ferrum --features cuda'
} >"$log"

set +e
CARGO_BUILD_JOBS=16 cargo build --release -p ferrum-cli --bin ferrum \
  --features cuda >>"$log" 2>&1
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  sha256sum target/release/ferrum >"$artifact/cuda-s1-binary.sha256"
fi
date -u +finished_at=%Y-%m-%dT%H:%M:%SZ >>"$log"
printf '%s\n' "$status" >"$artifact/cuda-s1-build.exit"
exit "$status"
