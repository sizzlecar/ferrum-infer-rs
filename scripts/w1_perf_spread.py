#!/usr/bin/env python3
"""Amendment-#4 same-pod mutual-spread check for the 32B GPTQ trio.

Reads the three l5f reports, compares c=32 output throughput, and prints
the spread relative to the trio mean. Spread <= 10% = the same-arch perf
criterion holds (identical code path, three same-class models, one pod).
"""

import json
import sys
from pathlib import Path

ART = Path("docs/goals/model-coverage-2026-06-12/artifacts")
MODELS = ["r1-32b", "qwen3-32b", "qwen25-coder-32b"]


def main() -> int:
    rows = []
    for rid in MODELS:
        reps = json.loads((ART / f"l5f_{rid}_cuda.json").read_text())
        reps = reps if isinstance(reps, list) else [reps]
        for r in reps:
            errs = sum(r["errored_per_run"])
            comp = r["completed_per_run"]
            if errs or any(c != comp[0] or c == 0 for c in comp):
                print(f"[spread] {rid} c={r['concurrency']}: NOT CLEAN "
                      f"(completed={comp}, errored={r['errored_per_run']})")
                return 1
        c32 = next(r for r in reps if r["concurrency"] == 32)
        tput = c32["output_throughput_tps"]["mean"]
        rows.append((rid, tput))
        print(f"[spread] {rid}: c=32 {tput:.1f} tok/s (all cells clean)")

    mean = sum(t for _, t in rows) / len(rows)
    worst = max(abs(t - mean) / mean for _, t in rows)
    print(f"[spread] trio mean {mean:.1f} tok/s; worst deviation {worst*100:.1f}%")
    verdict = "PASS" if worst <= 0.10 else "EXCEEDS-10PCT"
    print(f"W1_PERF_SPREAD {verdict}")
    return 0 if worst <= 0.10 else 2


if __name__ == "__main__":
    sys.exit(main())
