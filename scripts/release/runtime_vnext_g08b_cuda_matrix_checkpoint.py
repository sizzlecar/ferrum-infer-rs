#!/usr/bin/env python3
"""Validate the canonical G08B Qwen3.5-35B CUDA C01-C21 candidate report."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import runtime_vnext_baseline_scenarios as matrix


SCHEMA_VERSION = 1
CHECKPOINT_ID = "runtime-vnext-g08b-cuda-model-matrix"
MODEL_KEY = "m2-qwen35-35b-a3b"
BACKEND = "cuda"
EXPECTED_CASE_COUNT = 703
REQUIRED_CLIENT_CONCURRENCY = 32
REQUIRED_ACTIVE_FLOOR = 16
REQUIRED_ACTIVE_DUTY_CYCLE = 0.80
MODEL_LOCK_PATH = (
    Path(__file__).resolve().parent
    / "configs/runtime_vnext_g08b_m2_cuda.models.lock.json"
)
PASS_PREFIX = "FERRUM RUNTIME VNEXT G08B CUDA MODEL MATRIX PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G08B CUDA MODEL MATRIX SELFTEST PASS"
DOES_NOT_PROVE = [
    "G08B Metal Q4_K_S product path",
    "G08B legacy/reference parity",
    "G08B mutation and legacy-deletion acceptance",
    "G08B CUDA/Metal performance smoke",
    "G08B final PASS",
    "G09 formal performance",
    "G10 release readiness",
]


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def require_count(value: Any, label: str, *, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def expected_case_count(source_git_sha: str) -> int:
    catalog = matrix.candidate_expectations_catalog(source_git_sha)
    return len(matrix.planned_case_rows(MODEL_KEY, BACKEND, catalog))


def summarize_matrix(report: dict[str, Any]) -> dict[str, Any]:
    require(
        report.get("execution_contract") == matrix.G08_EXECUTION_CONTRACT,
        "scenario report is not a G08 candidate execution",
    )
    require(report.get("status") == "pass", "scenario report status is not pass")
    require(report.get("model_key") == MODEL_KEY, f"scenario report model must be {MODEL_KEY}")
    require(report.get("backend") == BACKEND, "scenario report backend must be cuda")
    require(
        report.get("models_lock_sha256") == file_sha256(MODEL_LOCK_PATH),
        "scenario report is not bound to the checked-in M2 CUDA model lock",
    )
    source_git_sha = report.get("source_git_sha")
    require(
        isinstance(source_git_sha, str) and matrix.GIT_SHA_RE.fullmatch(source_git_sha),
        "scenario report source_git_sha is invalid",
    )
    planned_count = expected_case_count(source_git_sha)
    require(
        planned_count == EXPECTED_CASE_COUNT,
        f"canonical M2 CUDA planner drifted from {EXPECTED_CASE_COUNT} to {planned_count}",
    )

    scenarios = require_list(report.get("scenarios"), "scenario report scenarios")
    require(
        [item.get("id") for item in scenarios if isinstance(item, dict)]
        == list(matrix.SCENARIO_IDS),
        "scenario report must contain ordered C01-C21 exactly once",
    )
    total_cases = 0
    for raw in scenarios:
        scenario = require_object(raw, "scenario")
        scenario_id = scenario.get("id")
        count = require_count(scenario.get("case_count"), f"{scenario_id}.case_count", minimum=1)
        require(scenario.get("status") == "pass", f"{scenario_id} status is not pass")
        require(scenario.get("passed_count") == count, f"{scenario_id} did not pass every case")
        for field in (
            "known_failed_count",
            "blocked_count",
            "failed_count",
            "error_count",
            "unexpected_count",
        ):
            require(scenario.get(field) == 0, f"{scenario_id}.{field} must be 0")
        total_cases += count
    require(
        total_cases == EXPECTED_CASE_COUNT,
        f"scenario report must pass exactly {EXPECTED_CASE_COUNT} cases, got {total_cases}",
    )

    commands = require_list(report.get("commands"), "scenario report commands")
    entrypoints = {
        command.get("entrypoint")
        for command in commands
        if isinstance(command, dict)
    }
    require(entrypoints == {"run", "serve"}, "scenario report must contain real run and serve commands")

    c18 = require_object(scenarios[17], "C18")
    cells = require_list(c18.get("concurrency_cells"), "C18 concurrency cells")
    require(
        [cell.get("requested_concurrency") for cell in cells if isinstance(cell, dict)]
        == [1, 4, 16, 32],
        "C18 CUDA cells must be ordered c1/c4/c16/c32",
    )
    for raw_cell in cells:
        cell = require_object(raw_cell, "C18 concurrency cell")
        requested = require_count(cell.get("requested_concurrency"), "C18 requested concurrency", minimum=1)
        matrix.validate_c18_resource_balance_summary(
            cell.get("resource_balance"),
            expected_request_count=requested,
            label=f"C18.c{requested}.resource_balance",
        )
    c32 = require_object(cells[-1], "C18.c32")
    cap = require_count(c32.get("typed_admission_cap"), "C18.c32.typed_admission_cap", minimum=1)
    floor = require_count(c32.get("active_floor"), "C18.c32.active_floor", minimum=1)
    observed = require_count(c32.get("observed_max_active"), "C18.c32.observed_max_active", minimum=1)
    timeline = require_object(c32.get("active_timeline"), "C18.c32.active_timeline")
    duty = timeline.get("active_duty_cycle")
    require(cap >= REQUIRED_ACTIVE_FLOOR, "C18.c32 typed admission cap is below 16")
    require(floor == REQUIRED_ACTIVE_FLOOR, "C18.c32 active floor must equal 16")
    require(
        REQUIRED_ACTIVE_FLOOR <= observed <= cap,
        "C18.c32 observed max-active does not reach the typed active floor",
    )
    require(
        isinstance(duty, (int, float))
        and not isinstance(duty, bool)
        and math.isfinite(float(duty))
        and float(duty) >= REQUIRED_ACTIVE_DUTY_CYCLE,
        "C18.c32 active duty-cycle is below 0.80",
    )
    require(
        c32.get("request_count") == REQUIRED_CLIENT_CONCURRENCY
        and c32.get("completed_request_count") == REQUIRED_CLIENT_CONCURRENCY
        and c32.get("completion_rate") == 1.0,
        "C18.c32 request completion is not 32/32",
    )
    for field in (
        "error_count",
        "bad_output_count",
        "crosstalk_count",
        "bad_checksum_count",
        "server_500_count",
        "panic_count",
        "oom_count",
    ):
        require(c32.get(field) == 0, f"C18.c32.{field} must be 0")

    return {
        "scenario_count": len(scenarios),
        "case_count": total_cases,
        "passed_case_count": total_cases,
        "known_failed_count": 0,
        "blocked_count": 0,
        "error_count": 0,
        "unexpected_count": 0,
        "entrypoints": sorted(entrypoints),
        "c18": {
            "requested_concurrency": REQUIRED_CLIENT_CONCURRENCY,
            "typed_admission_cap": cap,
            "active_floor": floor,
            "observed_max_active": observed,
            "active_duty_cycle": float(duty),
            "resource_balance": copy.deepcopy(c32["resource_balance"]),
        },
    }


def validate_report(artifact_root: Path, report_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_root = artifact_root.resolve()
    report_path = report_path.resolve()
    require(artifact_root.is_dir(), f"artifact root does not exist: {artifact_root}")
    try:
        report_path.relative_to(artifact_root)
    except ValueError as error:
        raise ValidationError("scenario report is outside its artifact root") from error
    report = matrix.read_json(report_path)
    try:
        matrix.validate_report_document(
            report,
            artifact_root,
            report_path=report_path,
            allow_internal_fixture=False,
            require_current_output_path=True,
        )
    except matrix.ScenarioError as error:
        raise ValidationError(f"canonical scenario report validation failed: {error}") from error
    return report, summarize_matrix(report)


def write_checkpoint(
    artifact_root: Path,
    report_path: Path,
    out: Path,
) -> dict[str, Any]:
    report, summary = validate_report(artifact_root, report_path)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "manifest.json").exists(), "checkpoint output already contains manifest.json")
    pass_line = f"{PASS_PREFIX}: {out}"
    validation = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g08b_cuda_model_matrix_validation",
        "checkpoint_id": CHECKPOINT_ID,
        "status": "pass",
        "validated_at": iso_now(),
        "execution_contract": matrix.G08_EXECUTION_CONTRACT,
        "source_git_sha": report["source_git_sha"],
        "source_tree_sha": report["source_tree_sha"],
        "model_key": MODEL_KEY,
        "backend": BACKEND,
        "model_revision": report["model_revision"],
        "model_files": report["model_files"],
        "binary_sha256": report["binary_sha256"],
        "models_lock_sha256": report["models_lock_sha256"],
        "scenario_report": {
            "path": str(report_path.resolve()),
            "sha256": file_sha256(report_path.resolve()),
        },
        "summary": summary,
        "does_not_prove": DOES_NOT_PROVE,
        "pass_line": pass_line,
    }
    validation_path = out / "validation.json"
    write_json(validation_path, validation)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g08b_cuda_model_matrix_manifest",
        "lane": CHECKPOINT_ID,
        "checkpoint_id": "G08B-CUDA-MATRIX",
        "status": "pass",
        "canonical": True,
        "source_git_sha": report["source_git_sha"],
        "source_tree_sha": report["source_tree_sha"],
        "dirty": False,
        "artifact_dir": str(out),
        "scenario_report": validation["scenario_report"],
        "validation": {
            "path": str(validation_path),
            "sha256": file_sha256(validation_path),
        },
        "summary": summary,
        "does_not_prove": DOES_NOT_PROVE,
        "pass_line": pass_line,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def fixture_report() -> dict[str, Any]:
    source_git_sha = "1" * 40
    rows = matrix.planned_case_rows(
        MODEL_KEY,
        BACKEND,
        matrix.candidate_expectations_catalog(source_git_sha),
    )
    counts = Counter(row["scenario_id"] for row in rows)
    scenarios: list[dict[str, Any]] = []
    for scenario_id in matrix.SCENARIO_IDS:
        count = counts[scenario_id]
        scenario: dict[str, Any] = {
            "id": scenario_id,
            "status": "pass",
            "case_count": count,
            "passed_count": count,
            "known_failed_count": 0,
            "blocked_count": 0,
            "failed_count": 0,
            "error_count": 0,
            "unexpected_count": 0,
        }
        if scenario_id == "C18":
            cells = []
            typed_cap = REQUIRED_ACTIVE_FLOOR
            for ordinal, requested in enumerate((1, 4, 16, 32), start=1):
                case_id = f"c18-{ordinal:03d}"
                floor = matrix.required_active_floor(MODEL_KEY, BACKEND, requested)
                trace = matrix.c18_fixture_trace_rows(case_id, requested, typed_cap)
                timeline = matrix.derive_active_timeline(
                    trace,
                    requested_concurrency=requested,
                    typed_active_cap=typed_cap,
                    active_floor=floor,
                    expected_request_count=requested,
                )
                cells.append(
                    {
                        "requested_concurrency": requested,
                        "case_count": 1,
                        "passed_count": 1,
                        "request_count": requested,
                        "completed_request_count": requested,
                        "completion_rate": 1.0,
                        "typed_admission_cap": typed_cap,
                        "active_floor": floor,
                        "observed_max_active": timeline["observed_max_active"],
                        "active_timeline": timeline,
                        "active_timeline_error": None,
                        "resource_balance": matrix.c18_resource_balance(
                            trace,
                            expected_request_count=requested,
                        ),
                        "error_count": 0,
                        "bad_output_count": 0,
                        "crosstalk_count": 0,
                        "bad_checksum_count": 0,
                        "server_500_count": 0,
                        "panic_count": 0,
                        "oom_count": 0,
                    }
                )
            scenario["concurrency_cells"] = cells
        scenarios.append(scenario)
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_contract": matrix.G08_EXECUTION_CONTRACT,
        "status": "pass",
        "source_git_sha": source_git_sha,
        "models_lock_sha256": file_sha256(MODEL_LOCK_PATH),
        "model_key": MODEL_KEY,
        "backend": BACKEND,
        "commands": [
            {"id": "actual-run", "entrypoint": "run"},
            {"id": "actual-serve", "entrypoint": "serve"},
        ],
        "scenarios": scenarios,
    }


def expect_reject(
    report: dict[str, Any],
    name: str,
    mutate: Callable[[dict[str, Any]], None],
    marker: str,
) -> None:
    candidate = copy.deepcopy(report)
    mutate(candidate)
    try:
        summarize_matrix(candidate)
    except (ValidationError, matrix.ScenarioError) as error:
        require(marker.lower() in str(error).lower(), f"{name} rejected unexpectedly: {error}")
        return
    raise AssertionError(f"{name} unexpectedly passed")


def self_test() -> int:
    locked_sources = matrix.locked_execution_sources(
        matrix.read_json(MODEL_LOCK_PATH),
        MODEL_KEY,
        BACKEND,
    )
    require(len(locked_sources["weight_files"]) == 19, "checked-in M2 CUDA weight lock is incomplete")
    require(len(locked_sources["semantic_source"]["files"]) == 5, "checked-in M2 semantic lock is incomplete")
    require(
        locked_sources["weight_revision"] == "3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b",
        "checked-in M2 CUDA revision drift",
    )
    report = fixture_report()
    summary = summarize_matrix(report)
    require(summary["scenario_count"] == 21, "fixture scenario count mismatch")
    require(summary["case_count"] == EXPECTED_CASE_COUNT, "fixture case count mismatch")
    require(summary["c18"]["active_duty_cycle"] >= 0.80, "fixture c32 duty-cycle mismatch")
    expect_reject(
        report,
        "wrong model lock",
        lambda value: value.update({"models_lock_sha256": "0" * 64}),
        "checked-in M2 CUDA model lock",
    )
    expect_reject(
        report,
        "known failure",
        lambda value: value["scenarios"][0].update(
            {"status": "known-fail", "known_failed_count": 1}
        ),
        "status is not pass",
    )
    expect_reject(
        report,
        "missing scenario",
        lambda value: value["scenarios"].pop(),
        "ordered C01-C21",
    )
    expect_reject(
        report,
        "missing case",
        lambda value: value["scenarios"][0].update(
            {
                "case_count": value["scenarios"][0]["case_count"] - 1,
                "passed_count": value["scenarios"][0]["passed_count"] - 1,
            }
        ),
        "exactly 703 cases",
    )
    expect_reject(
        report,
        "missing serve",
        lambda value: value.update({"commands": [{"id": "actual-run", "entrypoint": "run"}]}),
        "run and serve",
    )
    expect_reject(
        report,
        "low c32 duty",
        lambda value: value["scenarios"][17]["concurrency_cells"][-1]["active_timeline"].update(
            {"active_duty_cycle": 0.79}
        ),
        "below 0.80",
    )
    expect_reject(
        report,
        "low c32 cap",
        lambda value: value["scenarios"][17]["concurrency_cells"][-1].update(
            {"typed_admission_cap": 15}
        ),
        "below 16",
    )
    expect_reject(
        report,
        "resource leak",
        lambda value: value["scenarios"][17]["concurrency_cells"][-1]["resource_balance"].update(
            {"leaked_resource_count": 1}
        ),
        "leaked_resource_count",
    )
    print(SELFTEST_PASS_LINE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--scenario-report", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.artifact_root is None or args.scenario_report is None or args.out is None:
        parser.error("--artifact-root, --scenario-report, and --out are required")
    try:
        manifest = write_checkpoint(args.artifact_root, args.scenario_report, args.out)
    except (ValidationError, matrix.ScenarioError, OSError, ValueError) as error:
        print(f"FERRUM RUNTIME VNEXT G08B CUDA MODEL MATRIX FAIL: {args.out}: {error}", file=sys.stderr)
        return 1
    print(manifest["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
