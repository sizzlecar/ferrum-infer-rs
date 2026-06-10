#!/usr/bin/env python3
"""Explosion-radius classifier: changed paths -> required regression lanes.

Encodes the GOAL.md stage-4 routing rules so a PR's regression scope is a
deterministic function of its diff, not a judgement call:

  - only backend/cuda/**            -> L0 (+ recorded into the L1-cuda batch)
  - only backend/metal/**           -> L0 + L1-metal
  - shared LLM surface              -> L0 + L1-metal (+ L1-cuda batch)
    (ferrum-models, ferrum-engine, ferrum-scheduler, ferrum-sampler,
     ferrum-server, ferrum-interfaces, ferrum-types, ferrum-kv)
  - anything else                   -> L0

L1-cuda is never required synchronously (batched-execution decision, GOAL.md);
when a change touches CUDA or the shared surface this classifier records that
the L1-cuda batch range is non-empty rather than demanding a pod per PR.

Usage:
  explosion_radius.py --self-test
  explosion_radius.py --paths a.rs b.rs ...      # prints lanes JSON
  git diff --name-only main... | explosion_radius.py --stdin
"""

from __future__ import annotations

import argparse
import json
import sys

sys.dont_write_bytecode = True

SELFTEST_PASS = "EXPLOSION_RADIUS SELFTEST PASS"

L0 = "L0"
L1_METAL = "L1-metal"
L1_CUDA_BATCH = "L1-cuda-batch"

CUDA_BACKEND = "crates/ferrum-kernels/src/backend/cuda/"
METAL_BACKEND = "crates/ferrum-kernels/src/backend/metal/"

# Shared LLM surface: a change here can affect every backend's behavior, so it
# requires both platform lanes.
SHARED_PREFIXES = (
    "crates/ferrum-models/",
    "crates/ferrum-engine/",
    "crates/ferrum-scheduler/",
    "crates/ferrum-sampler/",
    "crates/ferrum-server/",
    "crates/ferrum-interfaces/",
    "crates/ferrum-types/",
    "crates/ferrum-kv/",
)


def classify(paths: list[str]) -> dict[str, object]:
    """Return required lanes for a set of changed paths.

    L0 is always required. Returns a dict with the sorted lane list plus the
    reasons that drove each non-L0 lane (for transparency in CI output).
    """
    lanes: set[str] = {L0}
    reasons: dict[str, list[str]] = {}

    def add(lane: str, path: str) -> None:
        lanes.add(lane)
        reasons.setdefault(lane, [])
        if path not in reasons[lane]:
            reasons[lane].append(path)

    for raw in paths:
        path = raw.strip()
        if not path:
            continue
        if path.startswith(CUDA_BACKEND):
            # CUDA-only kernel change: L0 locally; defer real GPU run to batch.
            add(L1_CUDA_BATCH, path)
        elif path.startswith(METAL_BACKEND):
            add(L1_METAL, path)
        elif any(path.startswith(p) for p in SHARED_PREFIXES):
            add(L1_METAL, path)
            add(L1_CUDA_BATCH, path)

    return {
        "lanes": sorted(lanes),
        "reasons": reasons,
        "l1_cuda_batch_required": L1_CUDA_BATCH in lanes,
    }


def run_self_test() -> None:
    # CUDA-only kernel change: L0 + cuda batch, no metal.
    r = classify([f"{CUDA_BACKEND}moe.rs"])
    assert r["lanes"] == [L0, L1_CUDA_BATCH], r
    assert r["l1_cuda_batch_required"] is True

    # Metal-only change: L0 + metal, no cuda batch.
    r = classify([f"{METAL_BACKEND}paged.rs"])
    assert r["lanes"] == [L0, L1_METAL], r
    assert r["l1_cuda_batch_required"] is False

    # Shared surface: both platform lanes.
    r = classify(["crates/ferrum-engine/src/continuous_engine.rs"])
    assert r["lanes"] == [L0, L1_CUDA_BATCH, L1_METAL], r
    assert r["l1_cuda_batch_required"] is True

    # Docs / scripts only: L0 only.
    r = classify(["docs/goals/x.md", "scripts/release/run_gate.py"])
    assert r["lanes"] == [L0], r

    # Mixed cuda + metal: both lanes.
    r = classify([f"{CUDA_BACKEND}moe.rs", f"{METAL_BACKEND}paged.rs"])
    assert r["lanes"] == [L0, L1_CUDA_BATCH, L1_METAL], r

    # Shared + cuda: still just both (no double counting).
    r = classify(
        ["crates/ferrum-models/src/models/llama_family.rs", f"{CUDA_BACKEND}quant.rs"]
    )
    assert r["lanes"] == [L0, L1_CUDA_BATCH, L1_METAL], r

    # Empty / whitespace paths ignored.
    r = classify(["", "  "])
    assert r["lanes"] == [L0], r

    # Non-cuda kernel code (shared kernels crate root) is not auto-CUDA.
    r = classify(["crates/ferrum-kernels/src/backend/traits.rs"])
    assert r["lanes"] == [L0], r

    print(SELFTEST_PASS)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--stdin", action="store_true", help="read paths from stdin")
    parser.add_argument("--paths", nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        try:
            run_self_test()
        except AssertionError as exc:
            print(f"SELFTEST FAIL: {exc}")
            return 1
        return 0
    paths = list(args.paths)
    if args.stdin:
        paths.extend(line for line in sys.stdin.read().splitlines())
    print(json.dumps(classify(paths), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
