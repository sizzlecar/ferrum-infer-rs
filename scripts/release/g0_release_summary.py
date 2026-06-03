#!/usr/bin/env python3
"""Aggregate G0 release gate artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_GROUPS = {
    "unit": ["unit.gate.json", "source/unit.gate.json"],
    "metal-source": ["metal.gate.json", "source/metal.gate.json"],
    "metal-tarball": ["metal-tarball/gate.json"],
    "cuda-tarball": ["cuda-tarball/gate.json"],
    "homebrew-cuda-fetch": ["homebrew-cuda-fetch/gate.json"],
}
OPTIONAL = ["homebrew-metal/gate.json", "g0_cuda4090_smoke.gate.json", "g0_cuda4090_full.gate.json"]


def load_gate(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing {path}"
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return False, f"invalid json {path}: {e}"
    if data.get("status") != "pass":
        return False, f"gate not pass {path}: {data}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root
    errors: list[str] = []
    gates: list[str] = []
    for name, alternatives in REQUIRED_GROUPS.items():
        group_errors = []
        for rel in alternatives:
            ok, msg = load_gate(root / rel)
            if ok:
                gates.append(rel)
                break
            group_errors.append(msg)
        else:
            errors.append(f"{name}: " + " OR ".join(group_errors))
    for rel in OPTIONAL:
        path = root / rel
        if path.exists():
            ok, msg = load_gate(path)
            if ok:
                gates.append(rel)
            else:
                errors.append(msg)
    if errors:
        for err in errors:
            print(f"G0 RELEASE FAIL: {err}", file=sys.stderr)
        return 1
    (root / "g0_release_summary.json").write_text(json.dumps({"status": "pass", "gates": gates}, indent=2) + "\n")
    print(f"G0 RELEASE PASS: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
