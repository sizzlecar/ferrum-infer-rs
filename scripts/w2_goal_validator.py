#!/usr/bin/env python3
"""MODEL_COVERAGE_W2 goal validator.

Reads docs/goals/model-coverage-2026-06-12/w2_matrix.json and prints
`MODEL_COVERAGE_W2 GOAL PASS: <goal_dir>` iff every cell of every model
is `pass`, or `waived` with a non-empty reason. Referenced artifacts must
exist on disk. Anything else prints a per-model TODO list and exits 1.

This is the only thing allowed to declare W2 done — see GOAL.md.
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOAL_DIR = REPO_ROOT / "docs/goals/model-coverage-2026-06-12"
MATRIX = GOAL_DIR / "w2_matrix.json"

VALID_STATUSES = {"pass", "fail", "pending", "waived"}
REQUIRED_L5_CONCURRENCY = [1, 4, 16, 32]
REQUIRED_RANDOM_INPUT_LEN = 256
REQUIRED_RANDOM_OUTPUT_LEN = 128
REQUIRED_REQUESTS_PER_RUN = 100
MIN_NEW_FAMILY_PERF_RATIO = 0.5


def validate_l5_artifact(path: Path, mid: str, cell_name: str) -> list[str]:
    problems: list[str] = []
    try:
        cells = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{mid}/{cell_name}: invalid L5 json {path}: {exc}"]
    if not isinstance(cells, list):
        return [f"{mid}/{cell_name}: L5 artifact must be a JSON list: {path}"]

    by_concurrency = {}
    for cell in cells:
        if not isinstance(cell, dict):
            problems.append(f"{mid}/{cell_name}: malformed L5 cell in {path}")
            continue
        by_concurrency[cell.get("concurrency")] = cell

    seen = sorted(k for k in by_concurrency if isinstance(k, int))
    if seen != REQUIRED_L5_CONCURRENCY:
        problems.append(
            f"{mid}/{cell_name}: L5 must contain c={REQUIRED_L5_CONCURRENCY}, got {seen}"
        )
        return problems

    for concurrency in REQUIRED_L5_CONCURRENCY:
        cell = by_concurrency[concurrency]
        completed = cell.get("completed_per_run")
        errored = cell.get("errored_per_run")
        output_source = cell.get("output_token_count_source")
        n_repeats = cell.get("n_repeats", 0)
        if cell.get("n_prompt") != REQUIRED_RANDOM_INPUT_LEN:
            problems.append(
                f"{mid}/{cell_name}: c={concurrency} n_prompt={cell.get('n_prompt')}, "
                f"expected {REQUIRED_RANDOM_INPUT_LEN}"
            )
        if cell.get("n_gen") != REQUIRED_RANDOM_OUTPUT_LEN:
            problems.append(
                f"{mid}/{cell_name}: c={concurrency} n_gen={cell.get('n_gen')}, "
                f"expected {REQUIRED_RANDOM_OUTPUT_LEN}"
            )
        if n_repeats < 3:
            problems.append(f"{mid}/{cell_name}: c={concurrency} n_repeats < 3")
        if output_source != "usage":
            problems.append(
                f"{mid}/{cell_name}: c={concurrency} output_token_count_source={output_source!r}"
            )
        if not isinstance(completed, list) or len(completed) != n_repeats:
            problems.append(
                f"{mid}/{cell_name}: c={concurrency} completed_per_run length does not "
                f"match n_repeats={n_repeats}: {completed}"
            )
        elif completed != [REQUIRED_REQUESTS_PER_RUN] * n_repeats:
            problems.append(
                f"{mid}/{cell_name}: c={concurrency} completed_per_run={completed}, "
                f"expected all {REQUIRED_REQUESTS_PER_RUN}"
            )
        if not isinstance(errored, list) or len(errored) != n_repeats:
            problems.append(
                f"{mid}/{cell_name}: c={concurrency} errored_per_run length does not "
                f"match n_repeats={n_repeats}: {errored}"
            )
        elif any(x != 0 for x in errored):
            problems.append(f"{mid}/{cell_name}: c={concurrency} errored_per_run={errored}")
    return problems


def validate_ratio_artifact(path: Path, mid: str, cell_name: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"ratio\s+([0-9]+(?:\.[0-9]+)?)\s+(PASS|FAIL)", text)
    if not match:
        return [f"{mid}/{cell_name}: ratio artifact does not contain a PASS/FAIL ratio line"]
    ratio = float(match.group(1))
    verdict = match.group(2)
    if verdict != "PASS" or ratio < MIN_NEW_FAMILY_PERF_RATIO:
        return [
            f"{mid}/{cell_name}: ratio {ratio:.3f} verdict={verdict}, "
            f"expected >= {MIN_NEW_FAMILY_PERF_RATIO}"
        ]
    return []


def validate_smoke_artifact(path: Path, mid: str, cell_name: str) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "SMOKE PASS" not in text:
        return [f"{mid}/{cell_name}: smoke artifact lacks SMOKE PASS line"]
    return []


def validate_pass_artifact(path: Path, mid: str, cell_name: str) -> list[str]:
    if cell_name == "l5_concurrency":
        return validate_l5_artifact(path, mid, cell_name)
    if cell_name == "perf_vs_llamacpp":
        return validate_ratio_artifact(path, mid, cell_name)
    if cell_name in {"l2_gptq_cuda", "l3_behavior", "l4_agent"}:
        return validate_smoke_artifact(path, mid, cell_name)
    return []


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
                artifact_path = REPO_ROOT / artifact
                artifact_problems = validate_pass_artifact(artifact_path, mid, cell_name)
                if artifact_problems:
                    problems.extend(artifact_problems)
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

    print(f"[w2-validator] {cells_pass}/{cells_total} cells satisfied")
    if problems:
        print(f"[w2-validator] {len(problems)} cells block the goal:")
        for p in problems:
            print(f"  - {p}")
        print(f"MODEL_COVERAGE_W2 GOAL FAIL ({len(problems)} blocking cells)")
        return 1

    print(f"MODEL_COVERAGE_W2 GOAL PASS: {GOAL_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
