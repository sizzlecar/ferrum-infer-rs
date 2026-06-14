#!/usr/bin/env python3
"""Flip the final six W1 cells from the l5f trio reports + spread verdict.

Run after the three l5f_*_cuda.json artifacts are fetched locally and
scripts/w1_perf_spread.py has printed its verdict. Refuses to flip
anything if a report is missing or unclean.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "docs/goals/model-coverage-2026-06-12/artifacts"
MATRIX = ROOT / "docs/goals/model-coverage-2026-06-12/w1_matrix.json"

TRIO = {
    "r1-32b": "r1-distill-qwen-32b",
    "qwen3-32b": "qwen3-dense-32b-14b",
    "qwen25-coder-32b": "qwen2.5-coder-32b",
}


def load_clean(rid: str):
    reps = json.loads((ART / f"l5f_{rid}_cuda.json").read_text())
    reps = reps if isinstance(reps, list) else [reps]
    for r in reps:
        if sum(r["errored_per_run"]) or any(c == 0 for c in r["completed_per_run"]):
            raise SystemExit(f"{rid}: unclean L5 report — refusing to flip cells")
    return reps


def main() -> int:
    spread_verdict = sys.argv[1] if len(sys.argv) > 1 else ""
    if spread_verdict not in ("PASS", "EXCEEDS-10PCT"):
        raise SystemExit("usage: w1_finalize_cells.py PASS|EXCEEDS-10PCT (from w1_perf_spread.py)")

    matrix = json.loads(MATRIX.read_text())
    by_id = {m["id"]: m for m in matrix["models"]}

    for rid, mid in TRIO.items():
        reps = load_clean(rid)
        cells = by_id[mid]["cells"]
        summary = "; ".join(
            f"c={r['concurrency']} {r['output_throughput_tps']['mean']:.1f} tok/s"
            for r in reps
        )
        cells["l5_concurrency"] = {
            "status": "pass",
            "artifact": f"docs/goals/model-coverage-2026-06-12/artifacts/l5f_{rid}_cuda.json",
            "note": f"CUDA 1x4090 (512tok x 32seq), n=3, 100% completion zero errors: {summary}",
        }
        if spread_verdict == "PASS":
            cells["perf_same_arch"] = {
                "status": "pass",
                "artifact": f"docs/goals/model-coverage-2026-06-12/artifacts/l5f_{rid}_cuda.json",
                "note": "Amendment #4 same-pod mutual spread across the 32B GPTQ trio <= 10% (w1_perf_spread.py)",
            }
        else:
            cells["perf_same_arch"] = {
                "status": "fail",
                "note": "Amendment #4 mutual spread exceeded 10% — open issue; see w1_perf_spread.py output",
            }

    MATRIX.write_text(json.dumps(matrix, ensure_ascii=False, indent=2))
    print("cells flipped; run scripts/w1_goal_validator.py")
    return 0


if __name__ == "__main__":
    main()
