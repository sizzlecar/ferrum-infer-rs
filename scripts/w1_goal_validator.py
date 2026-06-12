#!/usr/bin/env python3
"""MODEL_COVERAGE_W1 goal validator.

Reads docs/goals/model-coverage-2026-06-12/w1_matrix.json and prints
`MODEL_COVERAGE_W1 GOAL PASS: <goal_dir>` iff every cell of every model
is `pass`, or `waived` with a non-empty reason. Referenced artifacts must
exist on disk. Anything else prints a per-model TODO list and exits 1.

This is the only thing allowed to declare W1 done — see GOAL.md.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOAL_DIR = REPO_ROOT / "docs/goals/model-coverage-2026-06-12"
MATRIX = GOAL_DIR / "w1_matrix.json"

VALID_STATUSES = {"pass", "fail", "pending", "waived"}


def main() -> int:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    problems: list[str] = []
    cells_total = 0
    cells_pass = 0

    for model in matrix["models"]:
        mid = model["id"]
        for cell_name, cell in model["cells"].items():
            cells_total += 1
            status = cell.get("status")
            if status not in VALID_STATUSES:
                problems.append(f"{mid}/{cell_name}: invalid status {status!r}")
                continue
            artifact = cell.get("artifact")
            if artifact and not (REPO_ROOT / artifact).exists():
                problems.append(f"{mid}/{cell_name}: artifact missing on disk: {artifact}")
                continue
            if status == "pass":
                if not artifact:
                    problems.append(f"{mid}/{cell_name}: pass without artifact")
                    continue
                cells_pass += 1
            elif status == "waived":
                if not cell.get("reason"):
                    problems.append(f"{mid}/{cell_name}: waived without reason")
                    continue
                cells_pass += 1
            else:
                note = cell.get("note", "")
                problems.append(
                    f"{mid}/{cell_name}: {status}" + (f" — {note}" if note else "")
                )

    print(f"[w1-validator] {cells_pass}/{cells_total} cells satisfied")
    if problems:
        print(f"[w1-validator] {len(problems)} cells block the goal:")
        for p in problems:
            print(f"  - {p}")
        print(f"MODEL_COVERAGE_W1 GOAL FAIL ({len(problems)} blocking cells)")
        return 1

    print(f"MODEL_COVERAGE_W1 GOAL PASS: {GOAL_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
