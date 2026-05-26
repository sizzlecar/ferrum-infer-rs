#!/usr/bin/env bash
#
# pod_collect_m3_80pct.sh — fetch m3-80pct session results from pod to
# local docs/bench/m3-80pct-goal-2026-05-25/session-2026-05-26/.
#
# Usage:
#   bash scripts/pod_collect_m3_80pct.sh <ssh_host> <ssh_port>
#   bash scripts/pod_collect_m3_80pct.sh ssh9.vast.ai 13406

set -euo pipefail

SSH_HOST="${1:?need ssh host}"
SSH_PORT="${2:?need ssh port}"

LOCAL_DIR="docs/bench/m3-80pct-goal-2026-05-25/session-2026-05-26"
mkdir -p "$LOCAL_DIR"

# Pull the sweep output dir + orchestrator logs
echo "▶ pulling sweep artifacts..."
rsync -av -e "ssh -p $SSH_PORT" \
    "root@$SSH_HOST:/workspace/ferrum-infer-rs/docs/bench/m3-80pct-session-2026-05-26/" \
    "$LOCAL_DIR/" \
    --include='*.json' --include='*.csv' --include='*.md' --include='*.log' \
    --include='c*/' --include='c*/**' --exclude='*.nsys-rep' \
    2>&1 | tail -5

echo "▶ pulling nsys-rep separately (larger file)..."
scp -P "$SSH_PORT" \
    "root@$SSH_HOST:/workspace/ferrum-infer-rs/docs/bench/m3-80pct-session-2026-05-26/c32/ferrum_nsys.nsys-rep" \
    "$LOCAL_DIR/c32/" 2>&1 | tail -3 || echo "  (nsys-rep not yet ready or missing)"

echo "▶ pulling orchestrator logs..."
mkdir -p "$LOCAL_DIR/orchestrator"
scp -P "$SSH_PORT" \
    "root@$SSH_HOST:/workspace/m3-80pct-session/orchestrator.log" \
    "root@$SSH_HOST:/workspace/m3-80pct-session/build.log" \
    "root@$SSH_HOST:/workspace/m3-80pct-session/vllm_install.log" \
    "root@$SSH_HOST:/workspace/m3-80pct-session/hf_download.log" \
    "root@$SSH_HOST:/workspace/m3-80pct-session/sweep.log" \
    "$LOCAL_DIR/orchestrator/" 2>&1 | tail -5 || true

echo "▶ summary"
for c in 1 4 16 32; do
    cell_dir="$LOCAL_DIR/c${c}"
    if [ ! -d "$cell_dir" ]; then
        echo "  c=$c  missing"
        continue
    fi
    python3 - "$cell_dir" <<'PY'
import json, sys, os
d = sys.argv[1]
def load(p):
    if not os.path.exists(p): return None
    return json.load(open(p))
f = load(f"{d}/ferrum_baseline.json")
v = load(f"{d}/vllm_baseline.json")
fts = (f or {}).get('output_throughput_tps', {})
vts = (v or {}).get('output_throughput_tps', {})
ft = fts.get('mean', 0); ftsd = fts.get('std', 0)
vt = vts.get('mean', 0); vtsd = vts.get('std', 0)
c = os.path.basename(d).lstrip('c')
ratio = ft / vt if vt > 0 else 0
print(f"  c={c}  ferrum={ft:.1f}±{ftsd:.1f}  vllm={vt:.1f}±{vtsd:.1f}  ratio={ratio:.3f}")
PY
done

echo "▶ pulled into $LOCAL_DIR/"
ls -la "$LOCAL_DIR"
