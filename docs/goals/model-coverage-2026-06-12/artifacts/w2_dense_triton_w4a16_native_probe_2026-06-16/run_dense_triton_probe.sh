#!/usr/bin/env bash
set -euo pipefail
REMOTE_REPO=/workspace/ferrum-infer-rs
REMOTE_ART=/workspace/ferrum-artifacts/w2_dense_triton_w4a16_native_probe_2026-06-16
mkdir -p "$REMOTE_ART" "$REMOTE_ART/remote"
cd "$REMOTE_REPO"
date -u +%Y-%m-%dT%H:%M:%SZ > "$REMOTE_ART/remote/start_utc.txt"
git rev-parse HEAD > "$REMOTE_ART/remote/git_head.txt"
git status --short --untracked-files=no > "$REMOTE_ART/remote/git_status_short.txt"
nvidia-smi > "$REMOTE_ART/remote/nvidia_smi_before.txt"
nvcc --version > "$REMOTE_ART/remote/nvcc_version.txt"
OUT_BIN="$REMOTE_ART/dense_triton_w4a16_gemma3_perf" \
  timeout 20m bash scripts/microbenches/build_and_run_dense_triton_w4a16_gemma3_perf.sh \
  > "$REMOTE_ART/probe.stdout" 2> "$REMOTE_ART/probe.stderr"
rc=$?
printf '%s\n' "$rc" > "$REMOTE_ART/probe.rc"
if [ "$rc" -eq 0 ]; then
  sha256sum "$REMOTE_ART/dense_triton_w4a16_gemma3_perf" > "$REMOTE_ART/dense_triton_w4a16_gemma3_perf.sha256"
fi
nvidia-smi > "$REMOTE_ART/remote/nvidia_smi_after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$REMOTE_ART/remote/end_utc.txt"
exit "$rc"
