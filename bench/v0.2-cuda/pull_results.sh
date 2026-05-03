#!/usr/bin/env bash
# pull_results.sh — exfil the bench results from the rented pod.
# Run this LOCALLY (not on the pod). Test it mid-bench too — don't
# wait until the end to discover SSH is misconfigured.
#
# Usage:
#   POD_HOST=root@1.2.3.4 POD_PORT=22 bash pull_results.sh
# Or set POD=user@host:port (alias):
#   POD=root@1.2.3.4:22 bash pull_results.sh
#
# Pulls everything under the pod's /workspace/ferrum-infer-rs/bench/v0.2-cuda/
# (results/, _env.txt, prompts.json) into a fresh local directory
# named docs/bench/cuda-rtx4090-<today>/.

set -euo pipefail

if [[ -n "${POD:-}" ]]; then
  POD_HOST="${POD%:*}"
  POD_PORT="${POD##*:}"
fi

POD_HOST="${POD_HOST:?must set POD_HOST=user@host or POD=user@host:port}"
POD_PORT="${POD_PORT:-22}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATE="$(date +%Y-%m-%d)"
DEST="$ROOT/docs/bench/cuda-rtx4090-${DATE}"
mkdir -p "$DEST"

echo "rsyncing from $POD_HOST:$POD_PORT into $DEST/"
rsync -avz -e "ssh -p $POD_PORT -o StrictHostKeyChecking=no" \
  --exclude='*.gpu.csv'  `# can be huge; pull on demand` \
  "$POD_HOST:/workspace/ferrum-infer-rs/bench/v0.2-cuda/" \
  "$DEST/"

echo
echo "─── pulled artifacts ───"
ls -lh "$DEST/" | head -10
echo
echo "─── result count ───"
RESULTS_OK=$(find "$DEST/results" -name "*.json" 2>/dev/null | wc -l | xargs)
echo "  JSON results: $RESULTS_OK / 144"
echo
echo "next: write the report at $DEST/README.md"
