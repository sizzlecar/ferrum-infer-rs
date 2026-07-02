#!/usr/bin/env python3
"""W3-S0 design artifact gate.

This gate turns the W3 recurrent-state/cache design contract into a structured
artifact consumable by model_release_grade_goal_gate.py as `w3_s0_design`.
It is intentionally metadata-only: no GPU, model weights, or performance
claims are involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL_DOC = REPO_ROOT / "docs/goals/model-coverage-2026-06-12/RELEASE_GRADE_GOAL.md"
W3_CHARTER = REPO_ROOT / "docs/goals/model-coverage-2026-06-12/W3_CHARTER.md"
MANIFEST_NAME = "w3_s0_design.json"
PASS_LINE = "W3 S0 DESIGN PASS"
SELFTEST_PASS_LINE = "W3 S0 DESIGN SELFTEST PASS"
REQUIRED_GOAL_PHRASES = [
    "recurrent state cache trait",
    "paged-KV",
    "ContinuousBatch",
    "preemption/release",
]
REQUIRED_CHARTER_PHRASES = [
    "recurrent state cache",
    "paged-KV",
    "ContinuousBatch",
    "抢占/恢复",
]


class DesignGateError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git_output(args: list[str], *, default: str = "unknown") -> str:
    try:
        proc = run_command(["git", *args])
    except OSError:
        return default
    if proc.returncode != 0:
        return default
    return proc.stdout.strip()


def git_summary() -> dict[str, Any]:
    tracked = [
        line
        for line in git_output(["status", "--short", "--untracked-files=no"], default="").splitlines()
        if line.strip()
    ]
    untracked = [
        line
        for line in git_output(["ls-files", "--others", "--exclude-standard"], default="").splitlines()
        if line.strip()
    ]
    return {
        "sha": git_output(["rev-parse", "HEAD"]),
        "is_dirty": bool(tracked or untracked),
        "tracked_status_short": tracked,
        "untracked_count": len(untracked),
        "untracked_sample": untracked[:20],
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_text(path: Path, label: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DesignGateError(f"{label} missing: {path}") from exc
    if not text.strip():
        raise DesignGateError(f"{label} is empty: {path}")
    return text


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_ref(path: Path, out_dir: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    try:
        return path.relative_to(out_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def require_phrases(text: str, phrases: list[str], label: str) -> None:
    missing = [phrase for phrase in phrases if phrase not in text]
    if missing:
        joined = ", ".join(repr(phrase) for phrase in missing)
        raise DesignGateError(f"{label} missing required phrase(s): {joined}")


def design_manifest(out_dir: Path) -> dict[str, Any]:
    goal_text = load_text(GOAL_DOC, "release-grade goal doc")
    charter_text = load_text(W3_CHARTER, "W3 charter")
    require_phrases(goal_text, REQUIRED_GOAL_PHRASES, "release-grade goal doc")
    require_phrases(charter_text, REQUIRED_CHARTER_PHRASES, "W3 charter")

    pass_line = f"{PASS_LINE}: {out_dir}"
    return {
        "schema_version": 1,
        "status": "pass",
        "lane": "w3_s0_design",
        "created_at": iso_now(),
        "command_line": command_line(),
        "pass_line": pass_line,
        "hidden_env": [],
        "source_docs": {
            "release_grade_goal": {
                "path": artifact_ref(GOAL_DOC, out_dir),
                "sha256": sha256_text(goal_text),
            },
            "w3_charter": {
                "path": artifact_ref(W3_CHARTER, out_dir),
                "sha256": sha256_text(charter_text),
            },
        },
        "recurrent_state_cache": {
            "trait": "ferrum_interfaces::RecurrentStateManager",
            "handle": "ferrum_interfaces::RecurrentStateHandle",
            "state_spec": "ferrum_interfaces::RecurrentStateSpec",
            "tensor_spec": "ferrum_interfaces::RecurrentStateTensorSpec",
            "allocation_owner": "request admission",
            "lifetime_owner": "request lifecycle",
            "state_kind": "O(1) recurrent DeltaNet state, not sequence KV pages",
            "required_operations": [
                "allocate(request_id, spec)",
                "can_allocate(spec)",
                "get_handle(request_id)",
                "release(request_id)",
                "clone_handle_for_preemption(request_id)",
                "stats()",
            ],
        },
        "coexistence": {
            "paged_kv": (
                "Recurrent state is a separate allocation domain from paged-KV; "
                "mixed-attention models may carry both a KV handle and recurrent handle."
            ),
            "continuous_batch": (
                "ContinuousBatch admission allocates recurrent state once per request "
                "and passes the handle through prefill/decode batch items."
            ),
            "preemption": (
                "Preemption must preserve or clone the recurrent handle with the same "
                "request identity; dropped requests release both KV and recurrent state."
            ),
            "release": (
                "Completion, cancellation, or admission failure must release recurrent "
                "state even when no paged-KV blocks were allocated."
            ),
        },
        "scheduler_contract": {
            "prefill": "prefill receives a freshly allocated recurrent handle when the model advertises a spec",
            "decode": "decode reuses the request's recurrent handle and mutates state in token order",
            "multi_turn": "chat continuations retain request-local history only through explicit handles",
            "failure_mode": "missing recurrent state for a W3 model is a correctness failure, not a fallback",
        },
        "acceptance": {
            "design_covers_paged_kv": True,
            "design_covers_continuous_batch": True,
            "design_covers_preemption": True,
            "design_covers_release": True,
            "design_uses_typed_interfaces": True,
            "hidden_env_required": False,
        },
        "git": git_summary(),
    }


def validate_manifest(data: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if data.get("schema_version") != 1:
        problems.append("schema_version must be 1")
    if data.get("status") != "pass":
        problems.append("status must be pass")
    if data.get("lane") != "w3_s0_design":
        problems.append("lane must be w3_s0_design")
    pass_line = data.get("pass_line")
    if not isinstance(pass_line, str) or not pass_line.startswith(f"{PASS_LINE}:"):
        problems.append(f"pass_line must start with {PASS_LINE}:")
    if data.get("hidden_env") != []:
        problems.append("hidden_env must be empty")
    recurrent = data.get("recurrent_state_cache")
    if not isinstance(recurrent, dict):
        problems.append("recurrent_state_cache must be an object")
    else:
        for key in ["trait", "state_spec", "allocation_owner", "lifetime_owner", "state_kind"]:
            if not isinstance(recurrent.get(key), str) or not recurrent[key].strip():
                problems.append(f"recurrent_state_cache.{key} must be non-empty")
        ops = recurrent.get("required_operations")
        if not isinstance(ops, list) or len(ops) < 4 or not all(isinstance(item, str) and item for item in ops):
            problems.append("recurrent_state_cache.required_operations must list required operations")
    coexistence = data.get("coexistence")
    if not isinstance(coexistence, dict):
        problems.append("coexistence must be an object")
    else:
        for key in ["paged_kv", "continuous_batch", "preemption", "release"]:
            if not isinstance(coexistence.get(key), str) or not coexistence[key].strip():
                problems.append(f"coexistence.{key} must be non-empty")
    acceptance = data.get("acceptance")
    if not isinstance(acceptance, dict):
        problems.append("acceptance must be an object")
    else:
        for key in [
            "design_covers_paged_kv",
            "design_covers_continuous_batch",
            "design_covers_preemption",
            "design_covers_release",
            "design_uses_typed_interfaces",
        ]:
            if acceptance.get(key) is not True:
                problems.append(f"acceptance.{key} must be true")
        if acceptance.get("hidden_env_required") is not False:
            problems.append("acceptance.hidden_env_required must be false")
    return problems


def run_gate(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = design_manifest(out_dir)
    problems = validate_manifest(data)
    if problems:
        print(f"W3 S0 DESIGN FAIL ({len(problems)} problems)", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    write_json(out_dir / MANIFEST_NAME, data)
    print(data["pass_line"])
    return 0


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrum-w3-s0-design-") as tmp:
        out_dir = Path(tmp) / "good"
        data = design_manifest(out_dir)
        problems = validate_manifest(data)
        if problems:
            raise AssertionError("good S0 design selftest failed: " + "; ".join(problems))

        bad = dict(data)
        bad["coexistence"] = dict(data["coexistence"])
        bad["coexistence"]["preemption"] = ""
        bad_problems = validate_manifest(bad)
        if not any("coexistence.preemption" in problem for problem in bad_problems):
            raise AssertionError("bad S0 design selftest did not fail as expected")
    print(SELFTEST_PASS_LINE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/w3_s0_design")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            return run_selftest()
        return run_gate(Path(args.out))
    except DesignGateError as exc:
        print(f"W3 S0 DESIGN FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
