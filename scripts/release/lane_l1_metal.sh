#!/usr/bin/env bash
# L1-metal lane for the test-architecture goal (Gate C2).
#
# Runs the Metal-validated portion of the regression tiers on an Apple
# Silicon host (Metal is the local GPU): cross-backend op parity on Metal,
# the tiny-model full-stack engine suite under the metal feature, and the
# in-process server wire contract. Times the run and emits a lane artifact
# the final validator aggregates.
#
# Usage: scripts/release/lane_l1_metal.sh [OUT_DIR]
# PASS line: TEST_ARCH L1_METAL PASS: <out_dir>

set -euo pipefail

OUT_DIR="${1:-$(pwd)/.l1-metal-out}"
mkdir -p "$OUT_DIR"

# Share the main target to avoid duplicating the candle/metal build, and keep
# incremental compilation off so the cache does not balloon.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$(git rev-parse --show-toplevel)/target}"
export CARGO_INCREMENTAL=0

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "L1-metal lane requires macOS (Metal). Host is $(uname -s)." >&2
  exit 2
fi

start_total=$(date +%s)

run_step() {
  local name="$1"; shift
  echo "=== L1-metal step: $name ==="
  local t0 t1
  t0=$(date +%s)
  if ! "$@" > "$OUT_DIR/$name.log" 2>&1; then
    echo "FAIL: $name (see $OUT_DIR/$name.log)" >&2
    tail -20 "$OUT_DIR/$name.log" >&2
    exit 1
  fi
  t1=$(date +%s)
  echo "  ok ($((t1 - t0))s)"
  if ! grep -q "test result: ok" "$OUT_DIR/$name.log"; then
    echo "FAIL: $name did not report a passing test result" >&2
    exit 1
  fi
}

run_step op_parity_metal \
  cargo test -p ferrum-testkit --features metal --test op_diff
run_step tiny_stack_metal \
  cargo test -p ferrum-engine --features metal --test tiny_stack
run_step server_wire \
  cargo test -p ferrum-server --test tiny_stack_wire

end_total=$(date +%s)
elapsed=$((end_total - start_total))

cat > "$OUT_DIR/l1_metal.json" <<JSON
{
  "lane": "l1-metal",
  "status": "pass",
  "l1_metal_seconds": $elapsed,
  "steps": ["op_parity_metal", "tiny_stack_metal", "server_wire"],
  "host": "$(uname -sm)"
}
JSON

echo "l1_metal_seconds=$elapsed"
echo "TEST_ARCH L1_METAL PASS: $OUT_DIR"
