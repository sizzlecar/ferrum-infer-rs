#!/usr/bin/env python3
"""Plan release gates from changed files and checked-in impact rules."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL = "release-regression-hardening-2026-06-28"
DEFAULT_RULES = REPO_ROOT / "scripts/release/change_impact_rules.json"
DEFAULT_FIXTURES = REPO_ROOT / "scripts/release/fixtures/change_impact/planner_fixtures.json"
PASS_LINE = "CHANGE IMPACT GATE PLAN PASS"
SELFTEST_PASS_LINE = "CHANGE IMPACT GATE PLAN SELFTEST PASS"
FINAL_STAGE_GATES = {
    "actual_model_regression",
    "model_contract",
    "native_operator",
    "observability_profile",
    "product_sentinel",
    "resource_invariant",
    "support_matrix_contract",
}
SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTH", "CREDENTIAL", "KEY")
SAFE_ENV_NAMES = {"CI", "CARGO_HOME", "HF_HOME", "HOME", "PATH", "RUSTFLAGS", "RUST_BACKTRACE", "RUST_LOG"}
SAFE_ENV_PREFIXES = ("CARGO_", "FERRUM_", "HF_", "RUST_")


class PlannerError(RuntimeError):
    pass


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def run_git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise PlannerError(f"git {' '.join(args)} failed rc={proc.returncode}\n{proc.stderr}")
    return proc.stdout


def git_value(args: list[str], default: str = "unknown") -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return default
    return proc.stdout.strip() or default


def resolve_git_rev(value: str) -> str:
    if not value:
        return value
    resolved = git_value(["rev-parse", value], default="")
    return resolved or value


def git_changed_files(base: str, head: str) -> list[str]:
    out = run_git(["diff", "--name-only", f"{base}..{head}"])
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def git_dirty() -> bool:
    return bool(run_git(["status", "--short"]).strip())


def git_dirty_files() -> list[str]:
    return [line for line in run_git(["status", "--short"]).splitlines() if line.strip()]


def sanitized_env() -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in os.environ.items():
        if any(marker in key.upper() for marker in SECRET_ENV_MARKERS):
            continue
        if key in SAFE_ENV_NAMES or any(key.startswith(prefix) for prefix in SAFE_ENV_PREFIXES):
            safe[key] = value
    return dict(sorted(safe.items()))


def normalize_changed_files(files: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in files:
        rel = repo_rel(Path(item)) if item.startswith("/") else item
        rel = rel.strip().lstrip("./")
        if rel and rel not in seen:
            seen.add(rel)
            out.append(rel)
    return sorted(out)


def load_rules(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise PlannerError(f"{path}: schema_version must be 1")
    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        raise PlannerError(f"{path}: rules must be a non-empty list")
    required = {
        "id",
        "path_globs",
        "domains",
        "required_gates",
        "release_invalidation",
        "exceptions",
        "owner",
        "reason",
    }
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise PlannerError(f"{path}: rules[{idx}] must be an object")
        missing = sorted(required - set(rule))
        if missing:
            raise PlannerError(f"{path}: rules[{idx}] missing {', '.join(missing)}")
        for key in ("path_globs", "domains", "required_gates", "release_invalidation"):
            if not isinstance(rule[key], list):
                raise PlannerError(f"{path}: rules[{idx}].{key} must be a list")
        if not isinstance(rule["exceptions"], list):
            raise PlannerError(f"{path}: rules[{idx}].exceptions must be a list")
    return rules


def matches_any(path: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in globs)


def apply_exceptions(
    path: str,
    rule: dict[str, Any],
    required_gates: set[str],
    decision_log: list[dict[str, Any]],
) -> None:
    for exception in rule.get("exceptions", []):
        if not isinstance(exception, dict):
            continue
        globs = exception.get("path_globs")
        if not isinstance(globs, list) or not matches_any(path, globs):
            continue
        removed = sorted(set(exception.get("remove_required_gates") or []) & required_gates)
        required_gates.difference_update(removed)
        decision_log.append(
            {
                "path": path,
                "rule_id": rule["id"],
                "exception_id": exception.get("id"),
                "removed_required_gates": removed,
                "reason": exception.get("reason"),
            }
        )


def artifact_id(artifact: dict[str, Any], index: int) -> str:
    raw = artifact.get("id") or artifact.get("gate") or artifact.get("artifact_dir") or f"artifact-{index}"
    return str(raw)


def stage_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise PlannerError("--stage-artifact entries must use GATE=ARTIFACT_DIR")
    gate, raw_path = value.split("=", 1)
    gate = gate.strip()
    if not gate:
        raise PlannerError("--stage-artifact gate name must be non-empty")
    if gate not in FINAL_STAGE_GATES:
        raise PlannerError(f"--stage-artifact gate must be one of {sorted(FINAL_STAGE_GATES)}: {gate}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return gate, path


def read_stage_manifest(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "gate.manifest.json"
    if not manifest_path.exists():
        raise PlannerError(f"{artifact_dir}: missing gate.manifest.json")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PlannerError(f"{manifest_path}: expected JSON object")
    return data


def normalize_stage_artifact(value: str, head_sha: str) -> dict[str, Any]:
    gate, artifact_dir = stage_spec(value)
    if not artifact_dir.is_dir():
        raise PlannerError(f"{artifact_dir}: stage artifact directory does not exist")
    manifest = read_stage_manifest(artifact_dir)
    pass_line = str(manifest.get("pass_line") or "")
    if " PASS:" not in pass_line:
        raise PlannerError(f"{artifact_dir}: gate.manifest.json pass_line must contain ' PASS:'")
    git_sha = str(manifest.get("git_sha") or "")
    artifact_dirty = bool(manifest.get("git_dirty"))
    artifact = {
        "id": gate,
        "gate": gate,
        "artifact_dir": str(artifact_dir),
        "pass_line": pass_line,
        "git_sha": git_sha,
        "git_dirty": artifact_dirty,
        "impact_domains": list(manifest.get("impact_domains") or []),
        "strict_current": True,
        "manifest": str(artifact_dir / "gate.manifest.json"),
    }
    if git_sha != head_sha:
        artifact["stale_reason"] = "stage artifact git_sha does not match planned head"
    if artifact_dirty:
        artifact["stale_reason"] = "stage artifact was produced from a dirty tree"
    return artifact


def stale_artifact_invalidations(
    previous_artifacts: list[dict[str, Any]],
    impact_domains: set[str],
    head_sha: str,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    invalidated: list[str] = []
    stale: list[dict[str, Any]] = []
    satisfied: list[dict[str, Any]] = []
    for index, artifact in enumerate(previous_artifacts):
        if not isinstance(artifact, dict):
            continue
        artifact_domains = set(str(item) for item in artifact.get("impact_domains", []) if item)
        intersects = bool(artifact_domains & impact_domains)
        artifact_sha = str(artifact.get("git_sha") or "")
        artifact_dirty = bool(artifact.get("git_dirty"))
        strict_current = bool(artifact.get("strict_current"))
        aid = artifact_id(artifact, index)
        if (strict_current and artifact_sha != head_sha) or (strict_current and artifact_dirty):
            invalidated.append(f"artifact:{aid}")
            reason = artifact.get("stale_reason") or "strict stage artifact is not current"
            stale.append({**artifact, "id": aid, "stale_reason": reason})
        elif intersects and artifact_sha != head_sha:
            invalidated.append(f"artifact:{aid}")
            stale.append({**artifact, "id": aid, "stale_reason": "impact domain changed after artifact"})
        else:
            satisfied.append({**artifact, "id": aid})
    return invalidated, stale, satisfied


def plan_from_files(
    *,
    changed_files: list[str],
    base_sha: str,
    head_sha: str,
    dirty: bool,
    rules: list[dict[str, Any]],
    previous_artifacts: list[dict[str, Any]] | None = None,
    required_gate_overrides: set[str] | None = None,
) -> dict[str, Any]:
    changed_files = normalize_changed_files(changed_files)
    impact_domains: set[str] = set()
    required_gates: set[str] = set()
    invalidated: set[str] = set()
    optional_diagnostic_gates: set[str] = set()
    unknown_files: list[str] = []
    decision_log: list[dict[str, Any]] = []

    for changed in changed_files:
        matched = False
        for rule in rules:
            if not matches_any(changed, rule["path_globs"]):
                continue
            matched = True
            impact_domains.update(str(domain) for domain in rule["domains"])
            before_gates = set(required_gates)
            required_gates.update(str(gate) for gate in rule["required_gates"])
            invalidated.update(str(gate) for gate in rule["release_invalidation"])
            apply_exceptions(changed, rule, required_gates, decision_log)
            decision_log.append(
                {
                    "path": changed,
                    "rule_id": rule["id"],
                    "domains": rule["domains"],
                    "required_gates_added": sorted(required_gates - before_gates),
                    "release_invalidation": rule["release_invalidation"],
                    "owner": rule["owner"],
                    "reason": rule["reason"],
                }
            )
        if not matched:
            unknown_files.append(changed)
            decision_log.append(
                {
                    "path": changed,
                    "rule_id": None,
                    "domains": ["unknown"],
                    "reason": "no change-impact rule matched this path",
                }
            )

    artifact_invalidations, stale_artifacts, satisfied_artifacts = stale_artifact_invalidations(
        previous_artifacts or [],
        impact_domains,
        head_sha,
    )
    required_gates.update(required_gate_overrides or set())
    invalidated.update(artifact_invalidations)
    status = "fail" if unknown_files else "pass"
    return {
        "schema_version": 1,
        "status": status,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "dirty": dirty,
        "changed_files": changed_files,
        "impact_domains": sorted(impact_domains),
        "required_gates": sorted(required_gates),
        "optional_diagnostic_gates": sorted(optional_diagnostic_gates),
        "invalidated_previous_gates": sorted(invalidated),
        "unknown_files": unknown_files,
        "decision_log": decision_log,
        "previous_artifacts": previous_artifacts or [],
        "satisfied_artifacts": satisfied_artifacts,
        "stale_artifacts": stale_artifacts,
    }


def markdown_plan(plan: dict[str, Any]) -> str:
    lines = [
        "# Gate Plan",
        "",
        f"- status: `{plan['status']}`",
        f"- base_sha: `{plan['base_sha']}`",
        f"- head_sha: `{plan['head_sha']}`",
        f"- dirty: `{plan['dirty']}`",
        f"- impact domains: {', '.join(plan['impact_domains']) or '(none)'}",
        f"- required gates: {', '.join(plan['required_gates']) or '(none)'}",
        f"- invalidated gates: {', '.join(plan['invalidated_previous_gates']) or '(none)'}",
        "",
        "## Changed Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in plan["changed_files"])
    if plan["unknown_files"]:
        lines.extend(["", "## Unknown Files", ""])
        lines.extend(f"- `{path}`" for path in plan["unknown_files"])
    lines.extend(["", "## Decisions", "", "| path | rule | domains | reason |", "|---|---|---|---|"])
    for decision in plan["decision_log"]:
        lines.append(
            "| {path} | {rule} | {domains} | {reason} |".format(
                path=decision.get("path"),
                rule=decision.get("rule_id"),
                domains=", ".join(decision.get("domains") or []),
                reason=str(decision.get("reason", "")).replace("|", "\\|"),
            )
        )
    return "\n".join(lines) + "\n"


def release_candidate_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base_sha": plan["base_sha"],
        "head_sha": plan["head_sha"],
        "dirty": plan["dirty"],
        "changed_files": plan["changed_files"],
        "impact_domains": plan["impact_domains"],
        "required_gates": plan["required_gates"],
        "satisfied_gates": [
            artifact.get("gate") for artifact in plan["satisfied_artifacts"] if artifact.get("gate")
        ],
        "invalidated_gates": plan["invalidated_previous_gates"],
        "invalidation_reason": "derived from change-impact rules and stale artifact impact domains",
        "artifact_paths": [
            artifact.get("artifact_dir")
            for artifact in plan["satisfied_artifacts"]
            if artifact.get("artifact_dir")
        ],
        "pass_lines": [
            artifact.get("pass_line") for artifact in plan["satisfied_artifacts"] if artifact.get("pass_line")
        ],
        "stale_artifacts": plan["stale_artifacts"],
    }


def write_outputs(out: Path, plan: dict[str, Any], selfcheck: dict[str, Any] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (out / "gate_plan.md").write_text(markdown_plan(plan))
    (out / "changed_files.json").write_text(
        json.dumps({"schema_version": 1, "changed_files": plan["changed_files"]}, indent=2, sort_keys=True)
        + "\n"
    )
    (out / "release_candidate_manifest.json").write_text(
        json.dumps(release_candidate_manifest(plan), indent=2, sort_keys=True) + "\n"
    )
    (out / "planner_selfcheck.json").write_text(
        json.dumps(selfcheck or {"schema_version": 1, "status": "not_run"}, indent=2, sort_keys=True)
        + "\n"
    )


def write_standard_artifact_files(
    out: Path,
    *,
    plan: dict[str, Any],
    selfcheck: dict[str, Any] | None,
    started_at: int,
    ended_at: int,
    pass_line: str,
    command: list[str],
    rules: Path,
    fixtures: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "failures").mkdir(exist_ok=True)
    (out / "diagnostics").mkdir(exist_ok=True)
    dirty_files = git_dirty_files()
    (out / "pass_line.txt").write_text(pass_line + "\n", encoding="utf-8")
    (out / "command.log").write_text(" ".join(command) + "\n", encoding="utf-8")
    (out / "git_status.txt").write_text(run_git(["status", "--short"]), encoding="utf-8")
    (out / "sanitized_env.json").write_text(
        json.dumps(sanitized_env(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "goal": GOAL,
        "phase": "change_impact",
        "status": plan["status"],
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_sec": ended_at - started_at,
        "repo_root": str(REPO_ROOT),
        "git_sha": plan["head_sha"],
        "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "command": command,
        "artifact_dir": str(out),
        "pass_line": pass_line,
        "inputs": {
            "rules": str(rules),
            "fixtures": str(fixtures),
            "base_sha": plan["base_sha"],
            "head_sha": plan["head_sha"],
            "changed_files": plan["changed_files"],
            "previous_artifact_count": len(plan["previous_artifacts"]),
        },
        "outputs": {
            "gate_plan": str(out / "gate_plan.json"),
            "gate_plan_markdown": str(out / "gate_plan.md"),
            "release_candidate_manifest": str(out / "release_candidate_manifest.json"),
            "planner_selfcheck": str(out / "planner_selfcheck.json"),
            "changed_files": str(out / "changed_files.json"),
        },
        "validation_summary": {
            "impact_domains": plan["impact_domains"],
            "required_gates": plan["required_gates"],
            "unknown_file_count": len(plan["unknown_files"]),
            "stale_artifact_count": len(plan["stale_artifacts"]),
            "satisfied_artifact_count": len(plan["satisfied_artifacts"]),
            "planner_selfcheck_status": (selfcheck or {}).get("status", "not_run"),
        },
    }
    (out / "gate.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def assert_contains(label: str, actual: list[str], expected: list[str]) -> list[str]:
    missing = sorted(set(expected) - set(actual))
    return [f"{label} missing expected values: {missing}"] if missing else []


def assert_not_contains(label: str, actual: list[str], forbidden: list[str]) -> list[str]:
    present = sorted(set(forbidden) & set(actual))
    return [f"{label} unexpectedly contained values: {present}"] if present else []


def load_fixture_data(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise PlannerError(f"{path}: schema_version must be 1")
    fixtures = data.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise PlannerError(f"{path}: fixtures must be a non-empty list")
    return fixtures


def run_selftest(rules: list[dict[str, Any]], fixtures_path: Path) -> dict[str, Any]:
    fixture_results: list[dict[str, Any]] = []
    failures: list[str] = []
    fixtures = load_fixture_data(fixtures_path)
    for fixture in fixtures:
        fid = str(fixture.get("id"))
        plan = plan_from_files(
            changed_files=list(fixture.get("changed_files") or []),
            base_sha="fixture-base",
            head_sha="fixture-head",
            dirty=False,
            rules=rules,
            previous_artifacts=list(fixture.get("previous_artifacts") or []),
            required_gate_overrides=FINAL_STAGE_GATES
            if fixture.get("require_final_stage_gates")
            else set(),
        )
        release_candidate = release_candidate_manifest(plan)
        fixture_failures: list[str] = []
        expected_status = fixture.get("expect_status", "pass")
        if plan["status"] != expected_status:
            fixture_failures.append(f"status {plan['status']!r} != {expected_status!r}")
        fixture_failures += assert_contains(
            "impact_domains", plan["impact_domains"], list(fixture.get("expect_domains") or [])
        )
        fixture_failures += assert_contains(
            "required_gates", plan["required_gates"], list(fixture.get("expect_required_gates") or [])
        )
        fixture_failures += assert_not_contains(
            "required_gates", plan["required_gates"], list(fixture.get("forbid_required_gates") or [])
        )
        if "expect_unknown_files" in fixture and plan["unknown_files"] != fixture["expect_unknown_files"]:
            fixture_failures.append(
                f"unknown_files {plan['unknown_files']!r} != {fixture['expect_unknown_files']!r}"
            )
        if "expect_invalidated" in fixture and plan["invalidated_previous_gates"] != fixture["expect_invalidated"]:
            fixture_failures.append(
                "invalidated_previous_gates "
                f"{plan['invalidated_previous_gates']!r} != {fixture['expect_invalidated']!r}"
            )
        fixture_failures += assert_contains(
            "invalidated_previous_gates",
            plan["invalidated_previous_gates"],
            list(fixture.get("expect_invalidated_contains") or []),
        )
        fixture_failures += assert_contains(
            "satisfied_gates",
            release_candidate["satisfied_gates"],
            list(fixture.get("expect_satisfied_gates") or []),
        )
        if "expect_artifact_path_count" in fixture:
            actual_count = len(release_candidate["artifact_paths"])
            expected_count = int(fixture["expect_artifact_path_count"])
            if actual_count != expected_count:
                fixture_failures.append(f"artifact_paths count {actual_count} != {expected_count}")
        if fixture_failures:
            failures.extend(f"{fid}: {failure}" for failure in fixture_failures)
        fixture_results.append(
            {
                "id": fid,
                "status": "pass" if not fixture_failures else "fail",
                "plan_status": plan["status"],
                "impact_domains": plan["impact_domains"],
                "required_gates": plan["required_gates"],
                "unknown_files": plan["unknown_files"],
                "invalidated_previous_gates": plan["invalidated_previous_gates"],
                "failures": fixture_failures,
            }
        )
    return {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "fixture_count": len(fixtures),
        "fixtures": fixture_results,
        "failures": failures,
    }


def load_previous_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            artifacts.append(data)
        else:
            raise PlannerError(f"{path}: previous artifact manifest must be a JSON object")
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--previous-artifact", action="append", type=Path, default=[])
    parser.add_argument(
        "--stage-artifact",
        action="append",
        default=[],
        help="final goal stage evidence as GATE=ARTIFACT_DIR; reads ARTIFACT_DIR/gate.manifest.json",
    )
    parser.add_argument(
        "--require-final-stage-gates",
        action="store_true",
        help="force the final goal stage gates into required_gates for release candidate aggregation",
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    started_at = int(time.time())
    args = parse_args()
    rules = load_rules(args.rules)
    if args.self_test:
        selfcheck = run_selftest(rules, args.fixtures)
        if args.out:
            plan = plan_from_files(
                changed_files=[],
                base_sha="self-test",
                head_sha="self-test",
                dirty=False,
                rules=rules,
            )
            write_outputs(
                args.out,
                plan,
                selfcheck,
            )
            write_standard_artifact_files(
                args.out,
                plan=plan,
                selfcheck=selfcheck,
                started_at=started_at,
                ended_at=int(time.time()),
                pass_line=f"{SELFTEST_PASS_LINE}: {args.out}",
                command=sys.argv,
                rules=args.rules,
                fixtures=args.fixtures,
            )
        if selfcheck["status"] != "pass":
            raise PlannerError("\n".join(selfcheck["failures"]))
        suffix = f": {args.out}" if args.out else ""
        print(f"{SELFTEST_PASS_LINE}{suffix}")
        return 0

    if args.changed_file:
        changed_files = normalize_changed_files(args.changed_file)
        base_sha = resolve_git_rev(args.base) if args.base else "manual-base"
        head_sha = resolve_git_rev(args.head) if args.head else "manual-head"
    else:
        if not args.base or not args.head:
            raise PlannerError("provide --base and --head, or one or more --changed-file entries")
        changed_files = git_changed_files(args.base, args.head)
        base_sha = resolve_git_rev(args.base)
        head_sha = resolve_git_rev(args.head)
    previous_artifacts = load_previous_artifacts(args.previous_artifact)
    previous_artifacts.extend(normalize_stage_artifact(value, head_sha) for value in args.stage_artifact)
    plan = plan_from_files(
        changed_files=changed_files,
        base_sha=base_sha,
        head_sha=head_sha,
        dirty=git_dirty(),
        rules=rules,
        previous_artifacts=previous_artifacts,
        required_gate_overrides=FINAL_STAGE_GATES if args.require_final_stage_gates else set(),
    )
    if args.out is None:
        raise PlannerError("--out is required for non-self-test runs")
    write_outputs(args.out, plan)
    pass_line = f"{PASS_LINE}: {args.out}"
    write_standard_artifact_files(
        args.out,
        plan=plan,
        selfcheck=None,
        started_at=started_at,
        ended_at=int(time.time()),
        pass_line=pass_line,
        command=sys.argv,
        rules=args.rules,
        fixtures=args.fixtures,
    )
    if plan["status"] != "pass":
        raise PlannerError(f"gate plan failed: unknown_files={plan['unknown_files']}")
    print(pass_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
