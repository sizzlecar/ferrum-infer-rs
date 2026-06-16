#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ferrum-infer-rs
ART="docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_cache_policy_native_probe_2026-06-16"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ART/remote/started_at_utc.txt"
git rev-parse HEAD > "$ART/remote/git_head.txt"
git status --short --untracked-files=no > "$ART/remote/git_status_short.txt" || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$ART/remote/nvidia_smi_before.txt"
nvcc --version > "$ART/remote/nvcc_version.txt"
set +e
bash scripts/microbenches/build_and_run_gemma3_marlin_cache_policy_perf.sh > "$ART/probe/stdout.txt" 2> "$ART/probe/stderr.txt"
rc=$?
set -e
echo "$rc" > "$ART/probe/rc"
if [ -x /tmp/gemma3_tail_mlp_cache_policy_baseline ]; then
    sha256sum /tmp/gemma3_tail_mlp_cache_policy_baseline > "$ART/probe/baseline.sha256"
fi
if [ -x /tmp/gemma3_tail_mlp_cache_policy_evict_first ]; then
    sha256sum /tmp/gemma3_tail_mlp_cache_policy_evict_first > "$ART/probe/evict_first.sha256"
fi
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > "$ART/remote/nvidia_smi_after.txt" || true
date -u +%Y-%m-%dT%H:%M:%SZ > "$ART/remote/finished_at_utc.txt"
exit 0
