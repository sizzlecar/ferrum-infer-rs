#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
scripts/release.sh is no longer a release source of truth.

Use the unified gate runner and G0 release goal documents instead, for example:

  python3 scripts/release/run_gate.py --list-lanes
  python3 scripts/release/run_gate.py unit --out <out_dir>

This compatibility wrapper intentionally fails so stale release automation does
not bypass the required source, binary, and summary gates.
EOF

exit 2
