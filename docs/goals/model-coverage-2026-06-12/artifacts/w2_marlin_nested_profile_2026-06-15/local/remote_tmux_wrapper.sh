#!/usr/bin/env bash
set -euo pipefail

OUT=/workspace/w2_marlin_nested_profile_2026-06-15
mkdir -p "$OUT"

set +e
bash /workspace/run_w2_marlin_nested_profile.sh > "$OUT/nohup.log" 2>&1
rc=$?
set -e

printf '%s\n' "$rc" > "$OUT/nohup.rc"
exit "$rc"
