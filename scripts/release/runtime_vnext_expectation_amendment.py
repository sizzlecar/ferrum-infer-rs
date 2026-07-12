#!/usr/bin/env python3
"""Generate reviewed Runtime vNext expectation candidates from discovery evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import runtime_vnext_baseline_scenarios as baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_REPO_PATH = SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()
PASS_PREFIX = "RUNTIME VNEXT EXPECTATION CANDIDATE PASS"
SELFTEST_PASS_LINE = "RUNTIME VNEXT EXPECTATION AMENDMENT SELFTEST PASS"
SPLIT_FIELDS = ("variant", "preset", "entrypoint")
SELECTOR_FIELDS = ("scenario_id", "variant", "preset", "entrypoint", "case_id")
OUTCOME_STATUSES = {"pass", "known-fail", "blocked"}


class AmendmentError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AmendmentError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AmendmentError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON document must be an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_text(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def generator_identity() -> dict[str, Any]:
    require(not git_text(["status", "--short", "--untracked-files=all"]), "candidate generation requires a clean worktree")
    git_sha = git_text(["rev-parse", "HEAD"])
    tree_sha = git_text(["rev-parse", "HEAD^{tree}"])
    blob_sha = git_text(["rev-parse", f"HEAD:{SCRIPT_REPO_PATH}"])
    checked_in = subprocess.run(
        ["git", "show", f"HEAD:{SCRIPT_REPO_PATH}"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(checked_in.returncode == 0, f"candidate generator is not checked in: {checked_in.stderr.decode(errors='replace').strip()}")
    require(checked_in.stdout == SCRIPT_PATH.read_bytes(), "candidate generator differs from its checked-in blob")
    return {
        "path": SCRIPT_REPO_PATH,
        "git_sha": git_sha,
        "source_tree_sha": tree_sha,
        "git_blob_sha": blob_sha,
        "sha256": hashlib.sha256(checked_in.stdout).hexdigest(),
        "dirty_status": {"is_dirty": False, "status_short": []},
    }


def normalized_value(case: dict[str, Any], field: str) -> str:
    value = case.get(field)
    if field == "preset" and value is None:
        return "none"
    require(isinstance(value, str) and value, f"case {case.get('case_id')} has invalid {field}")
    return value


def outcome(case: dict[str, Any]) -> tuple[str, str | None]:
    observed = case.get("observed_outcome")
    require(isinstance(observed, dict), f"case {case.get('case_id')} lacks observed_outcome")
    status = observed.get("status")
    failure_class = observed.get("failure_class")
    require(status in OUTCOME_STATUSES, f"case {case.get('case_id')} observed status is invalid")
    if status == "pass":
        require(failure_class is None, f"passing case {case.get('case_id')} declares a failure class")
    else:
        require(isinstance(failure_class, str) and failure_class, f"non-passing case {case.get('case_id')} lacks failure class")
    require(case.get("status") == status, f"case {case.get('case_id')} status differs from observed_outcome")
    return str(status), failure_class


def selector_from(path: dict[str, str], *, case_id: str = "*") -> dict[str, str]:
    return {
        "scenario_id": path["scenario_id"],
        "variant": path.get("variant", "*"),
        "preset": path.get("preset", "*"),
        "entrypoint": path.get("entrypoint", "*"),
        "case_id": case_id,
    }


def proposal(path: dict[str, str], result: tuple[str, str | None], count: int, *, case_id: str = "*") -> dict[str, Any]:
    return {
        "selector": selector_from(path, case_id=case_id),
        "expected_status": result[0],
        "failure_class": result[1],
        "observation_count": count,
    }


def proposal_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    selector = row["selector"]
    scenario_number = int(selector["scenario_id"][1:])
    specificity = sum(selector[field] != "*" for field in SELECTOR_FIELDS)
    return (
        scenario_number,
        specificity,
        *(selector[field] for field in SELECTOR_FIELDS[1:]),
        row["expected_status"],
        row["failure_class"] or "",
    )


def plan_key(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    ordered = sorted(rows, key=proposal_sort_key)
    exact_count = sum(row["selector"]["case_id"] != "*" for row in ordered)
    specificity = sum(
        sum(row["selector"][field] != "*" for field in SELECTOR_FIELDS)
        for row in ordered
    )
    stable = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    return len(ordered), exact_count, specificity, stable


def compress_group(
    cases: list[dict[str, Any]],
    path: dict[str, str],
    available_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    counts = Counter(outcome(case) for case in cases)
    if len(counts) == 1:
        result = next(iter(counts))
        return [proposal(path, result, len(cases))]

    default_result = sorted(counts, key=lambda item: (-counts[item], item[0], item[1] or ""))[0]
    default_plan = [proposal(path, default_result, counts[default_result])]
    default_plan.extend(
        proposal(path, outcome(case), 1, case_id=str(case["case_id"]))
        for case in sorted(cases, key=lambda item: str(item["case_id"]))
        if outcome(case) != default_result
    )
    options = [default_plan]

    for field in available_fields:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for case in cases:
            buckets.setdefault(normalized_value(case, field), []).append(case)
        if len(buckets) <= 1:
            continue
        remaining = tuple(candidate for candidate in available_fields if candidate != field)
        split_plan: list[dict[str, Any]] = []
        for value in sorted(buckets):
            split_plan.extend(compress_group(buckets[value], {**path, field: value}, remaining))
        options.append(split_plan)
    return min(options, key=plan_key)


def compress_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        scenario_id = case.get("scenario_id")
        require(scenario_id in baseline.SCENARIO_IDS, f"case {case.get('case_id')} scenario is invalid")
        by_scenario.setdefault(str(scenario_id), []).append(case)
    require(set(by_scenario) == set(baseline.SCENARIO_IDS), "discovery cases do not cover C01-C21")
    proposals: list[dict[str, Any]] = []
    for scenario_id in baseline.SCENARIO_IDS:
        proposals.extend(compress_group(by_scenario[scenario_id], {"scenario_id": scenario_id}, SPLIT_FIELDS))
    return sorted(proposals, key=proposal_sort_key)


def materialize_rules(
    proposals: list[dict[str, Any]],
    *,
    model_key: str,
    backend: str,
    report_sha256: str,
    downstream_goal: str,
    owner: str,
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for row in proposals:
        status = row["expected_status"]
        count = row["observation_count"]
        evidence_basis = (
            f"Frozen cff4 {model_key}/{backend} discovery report sha256:{report_sha256} "
            f"observed {count} matching case{'s' if count != 1 else ''}."
        )
        next_action = (
            f"Preserve this frozen legacy behavior through {downstream_goal} migration."
            if status == "pass"
            else f"Reproduce this frozen failure in G00 and make the final contract pass in {downstream_goal}."
        )
        rules.append(
            {
                "selector": row["selector"],
                "expected_status": status,
                "failure_class": row["failure_class"],
                "downstream_goal": downstream_goal,
                "owner": owner,
                "evidence_basis": evidence_basis,
                "next_action": next_action,
            }
        )
    return rules


def safe_artifact_path(root: Path, raw: Any, label: str) -> Path:
    require(isinstance(raw, str) and raw, f"{label} path is invalid")
    relative = Path(raw)
    require(not relative.is_absolute() and ".." not in relative.parts, f"{label} path is not a safe relative path")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), f"{label} path crosses a symlink: {cursor}")
    path = cursor.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AmendmentError(f"{label} escapes artifact root") from exc
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked: {path}")
    return path


def load_discovery(root: Path, report_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    root = root.resolve()
    report_input = report_path.expanduser().absolute()
    try:
        report_relative = report_input.relative_to(root)
    except ValueError as exc:
        raise AmendmentError("discovery report is outside artifact root") from exc
    report_path = safe_artifact_path(root, report_relative.as_posix(), "discovery report")
    report = read_json(report_path)
    report_sha = file_sha256(report_path)
    require(report.get("schema_version") == baseline.SCHEMA_VERSION, "discovery report schema_version mismatch")
    require(report.get("status") == "discovery", "input report is not discovery evidence")
    require(report.get("formal_pass_allowed") is False, "discovery report incorrectly allows formal PASS")
    require(report.get("source_git_sha") == baseline.FROZEN_LEGACY_SHA, "discovery report source SHA drift")
    require(report.get("pass_line") is None, "discovery report must not carry a PASS line")
    model_key = report.get("model_key")
    backend = report.get("backend")
    require(model_key in {"m1-qwen35-4b", "m2-qwen35-35b-a3b", "m3-qwen3-30b-a3b"}, "discovery model key invalid")
    require(backend in {"cuda", "metal"}, "discovery backend invalid")

    snapshot_path = safe_artifact_path(root, "legacy-correctness-expectations.json", "discovery expectation snapshot")
    snapshot = read_json(snapshot_path)
    require(snapshot.get("schema_version") == baseline.SCHEMA_VERSION, "discovery expectation snapshot schema drift")
    require(snapshot.get("source_git_sha") == baseline.FROZEN_LEGACY_SHA, "discovery expectation snapshot source SHA drift")
    snapshot_lanes = snapshot.get("lanes")
    expected_lanes = {
        f"{model}/{lane_backend}"
        for model in ("m1-qwen35-4b", "m2-qwen35-35b-a3b", "m3-qwen3-30b-a3b")
        for lane_backend in ("cuda", "metal")
    }
    require(isinstance(snapshot_lanes, dict) and set(snapshot_lanes) == expected_lanes, "discovery expectation snapshot lane matrix drift")
    snapshot_sha = file_sha256(snapshot_path)
    require(report.get("expectations_catalog_sha256") == snapshot_sha, "discovery report expectation snapshot drift")

    invocation_ref = report.get("executor_invocation")
    require(isinstance(invocation_ref, dict), "discovery executor invocation ref is invalid")
    require(invocation_ref.get("kind") == "raw-json", "discovery executor invocation kind is invalid")
    invocation_path = safe_artifact_path(root, invocation_ref.get("path"), "discovery executor invocation")
    require(invocation_ref.get("sha256") == file_sha256(invocation_path), "discovery executor invocation SHA256 mismatch")
    invocation = read_json(invocation_path)
    require(invocation.get("mode") == "discover", "discovery executor invocation mode drift")
    require(invocation.get("runner_path") == baseline.RUNNER_REPO_PATH, "discovery executor runner path drift")
    invocation_argv = invocation.get("argv")
    require(isinstance(invocation_argv, list) and "--discover" in invocation_argv, "discovery executor argv lacks --discover")

    current_catalog = baseline.validate_expectations_catalog(read_json(baseline.EXPECTATIONS_PATH))
    planned = baseline.planned_case_rows(str(model_key), str(backend), current_catalog)
    planned_by_id = {row["case_id"]: row for row in planned}
    observations = report.get("observations")
    require(isinstance(observations, list), "discovery observations must be an array")
    require(report.get("case_count") == len(observations) == len(planned), "discovery case count differs from the current matrix")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for index, ref in enumerate(observations):
        require(isinstance(ref, dict), f"observation[{index}] ref is invalid")
        require(ref.get("kind") == "raw-json", f"observation[{index}] kind is invalid")
        path = safe_artifact_path(root, ref.get("path"), f"observation[{index}]")
        require(path not in seen_paths, f"observation path is reused: {path}")
        seen_paths.add(path)
        require(ref.get("sha256") == file_sha256(path), f"observation[{index}] SHA256 mismatch")
        case = read_json(path)
        baseline.reject_forbidden_markers(case, f"observation[{index}]", allow_internal_fixture=False)
        case_id = case.get("case_id")
        require(isinstance(case_id, str) and case_id in planned_by_id, f"observation[{index}] case id is not planned")
        require(case_id not in seen_ids, f"duplicate discovery case id: {case_id}")
        seen_ids.add(case_id)
        row = planned_by_id[case_id]
        require(case.get("schema_version") == baseline.SCHEMA_VERSION, f"case {case_id} schema drift")
        require(case.get("source_git_sha") == baseline.FROZEN_LEGACY_SHA, f"case {case_id} source SHA drift")
        require(case.get("model_key") == model_key and case.get("backend") == backend, f"case {case_id} lane drift")
        require(case.get("expectations_catalog_sha256") == snapshot_sha, f"case {case_id} expectation snapshot drift")
        expected_identity = (row["scenario_id"], row["variant"], row["preset"], row["entrypoint"])
        actual_identity = (case.get("scenario_id"), case.get("variant"), case.get("preset"), case.get("entrypoint"))
        require(actual_identity == expected_identity, f"case {case_id} identity differs from the current matrix")
        outcome(case)
        cases.append(case)
    require(seen_ids == set(planned_by_id), "discovery observations do not cover the complete case-id matrix")
    return report, cases, report_sha


def validate_candidate(
    cases: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    *,
    model_key: str,
    backend: str,
) -> dict[str, Any]:
    catalog = baseline.validate_expectations_catalog(read_json(baseline.EXPECTATIONS_PATH))
    trial = copy.deepcopy(catalog)
    lane_key = f"{model_key}/{backend}"
    trial["lanes"][lane_key]["rules"] = rules
    baseline.validate_expectations_catalog(trial)
    case_by_id = {str(case["case_id"]): case for case in cases}
    rows = baseline.planned_case_rows(model_key, backend, trial)
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        actual = outcome(case_by_id[row["case_id"]])
        expectation = row["expectation"]
        resolved = (expectation["expected_status"], expectation["failure_class"])
        if actual != resolved:
            mismatches.append({"case_id": row["case_id"], "observed": actual, "resolved": resolved})
    require(not mismatches, f"candidate rules mismatch discovery: {mismatches[:5]}")
    statuses = Counter(outcome(case)[0] for case in cases)
    failures = Counter(outcome(case)[1] for case in cases if outcome(case)[1] is not None)
    return {
        "case_count": len(rows),
        "mismatch_count": 0,
        "status_counts": dict(sorted(statuses.items())),
        "failure_class_counts": dict(sorted(failures.items())),
        "resolved_lane_sha256": canonical_json_sha256(trial["lanes"][lane_key]),
    }


def generate(
    root: Path,
    report_path: Path,
    out: Path,
    *,
    downstream_goal: str,
    owner: str,
) -> dict[str, Any]:
    require(downstream_goal in {"G08A", "G08B", "G08C"}, "downstream goal must be G08A, G08B, or G08C")
    require(owner.strip() == owner and owner, "owner must be a non-empty canonical string")
    report, cases, report_sha = load_discovery(root, report_path)
    model_key = str(report["model_key"])
    backend = str(report["backend"])
    proposals = compress_cases(cases)
    rules = materialize_rules(
        proposals,
        model_key=model_key,
        backend=backend,
        report_sha256=report_sha,
        downstream_goal=downstream_goal,
        owner=owner,
    )
    validation = validate_candidate(cases, rules, model_key=model_key, backend=backend)
    out = out.expanduser().resolve()
    try:
        out.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise AmendmentError("candidate output must live outside the source worktree")
    result = {
        "schema_version": 1,
        "status": "candidate",
        "formal_pass_allowed": False,
        "model_key": model_key,
        "backend": backend,
        "source_report": {"path": str(report_path.resolve()), "sha256": report_sha},
        "source_expectations_catalog_sha256": report["expectations_catalog_sha256"],
        "generator": generator_identity(),
        "rule_count": len(rules),
        "validation": validation,
        "rules": rules,
        "next_gate": "Review and commit this lane together with every remaining discovery amendment, then rerun the full canonical lane.",
        "pass_line": f"{PASS_PREFIX}: {out.resolve()}",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, out)
    return result


def resolve_proposal(rows: list[dict[str, Any]], case: dict[str, Any]) -> tuple[str, str | None]:
    values = {
        "scenario_id": str(case["scenario_id"]),
        "variant": normalized_value(case, "variant"),
        "preset": normalized_value(case, "preset"),
        "entrypoint": normalized_value(case, "entrypoint"),
        "case_id": str(case["case_id"]),
    }
    matches: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        selector = row["selector"]
        if all(selector[field] == "*" or selector[field] == values[field] for field in SELECTOR_FIELDS):
            matches.append((sum(selector[field] != "*" for field in SELECTOR_FIELDS), row))
    require(matches, f"self-test case {case['case_id']} is uncovered")
    score = max(item[0] for item in matches)
    winners = [row for specificity, row in matches if specificity == score]
    require(len(winners) == 1, f"self-test case {case['case_id']} is ambiguous")
    return winners[0]["expected_status"], winners[0]["failure_class"]


def fixture_case(
    scenario_id: str,
    ordinal: int,
    *,
    variant: str,
    preset: str | None,
    entrypoint: str,
    status: str,
) -> dict[str, Any]:
    return {
        "case_id": f"{scenario_id.lower()}-{ordinal:03d}",
        "scenario_id": scenario_id,
        "variant": variant,
        "preset": preset,
        "entrypoint": entrypoint,
        "status": status,
        "observed_outcome": {
            "status": status,
            "failure_class": None if status == "pass" else f"{scenario_id.lower()}-contract-violation",
        },
    }


def self_test() -> int:
    uniform = [
        fixture_case("C01", index, variant="config", preset=None, entrypoint="run", status="known-fail")
        for index in range(1, 21)
    ]
    mixed_exact = [
        fixture_case(
            "C03",
            index,
            variant="all",
            preset="P_DETERMINISTIC",
            entrypoint="run",
            status="known-fail" if index in {4, 5, 7, 9} else "pass",
        )
        for index in range(1, 11)
    ]
    preset_split = [
        fixture_case(
            "C14",
            index,
            variant=("required", "type", "enum", "additional-properties")[(index - 1) % 4],
            preset="P_NO_THINKING" if index <= 50 else "P_THINKING",
            entrypoint="serve",
            status="pass" if index <= 50 else "known-fail",
        )
        for index in range(1, 71)
    ]
    expectations = {
        "C01": (uniform, 1),
        "C03": (mixed_exact, 5),
        "C14": (preset_split, 2),
    }
    for scenario_id, (cases, expected_rule_count) in expectations.items():
        rules = compress_group(cases, {"scenario_id": scenario_id}, SPLIT_FIELDS)
        require(len(rules) == expected_rule_count, f"{scenario_id} compression count drift")
        for case in cases:
            require(resolve_proposal(rules, case) == outcome(case), f"{scenario_id} compressed outcome drift")
    ambiguous = compress_group(uniform, {"scenario_id": "C01"}, SPLIT_FIELDS)
    ambiguous.append(copy.deepcopy(ambiguous[0]))
    try:
        resolve_proposal(ambiguous, uniform[0])
    except AmendmentError as exc:
        require("ambiguous" in str(exc), f"ambiguity fixture failed for unexpected reason: {exc}")
    else:
        raise AssertionError("duplicate selector unexpectedly resolved")
    print(SELFTEST_PASS_LINE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--downstream-goal")
    parser.add_argument("--owner")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if None in (args.artifact_root, args.report, args.out, args.downstream_goal, args.owner):
        parser.error("--artifact-root, --report, --out, --downstream-goal, and --owner are required")
    try:
        result = generate(
            args.artifact_root,
            args.report,
            args.out,
            downstream_goal=args.downstream_goal,
            owner=args.owner,
        )
    except (AmendmentError, baseline.ScenarioError, OSError, ValueError) as exc:
        print(f"RUNTIME VNEXT EXPECTATION CANDIDATE FAIL: {args.out}: {exc}", file=sys.stderr)
        return 1
    print(result["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
