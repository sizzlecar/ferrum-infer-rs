#!/usr/bin/env python3
"""Validate G08B model matrices; the path remains CUDA-named for compatibility."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import runtime_vnext_baseline_scenarios as matrix
from runtime_vnext_g08b_cuda_matrix_prepare import BACKEND_SPECS as PREPARATION_SPECS


SCHEMA_VERSION = 1
MODEL_KEY = "m2-qwen35-35b-a3b"


@dataclass(frozen=True)
class CheckpointSpec:
    backend: str
    checkpoint_id: str
    checkpoint_label: str
    expected_case_count: int
    required_client_concurrency: int
    required_active_floor: int
    required_active_duty_cycle: float
    concurrency_cells: tuple[int, ...]
    pass_prefix: str
    selftest_pass_line: str
    does_not_prove: tuple[str, ...]

    @property
    def model_lock_path(self) -> Path:
        return PREPARATION_SPECS[self.backend].model_lock_path

    @property
    def artifact_type_prefix(self) -> str:
        return f"runtime_vnext_g08b_{self.backend}_model_matrix"


CUDA_SPEC = CheckpointSpec(
    backend="cuda",
    checkpoint_id="runtime-vnext-g08b-cuda-model-matrix",
    checkpoint_label="G08B-CUDA-MATRIX",
    expected_case_count=703,
    required_client_concurrency=32,
    required_active_floor=16,
    required_active_duty_cycle=0.80,
    concurrency_cells=(1, 4, 16, 32),
    pass_prefix="FERRUM RUNTIME VNEXT G08B CUDA MODEL MATRIX PASS",
    selftest_pass_line="FERRUM RUNTIME VNEXT G08B CUDA MODEL MATRIX SELFTEST PASS",
    does_not_prove=(
        "G08B Metal Q4_K_S product path",
        "G08B legacy/reference parity",
        "G08B mutation and legacy-deletion acceptance",
        "G08B CUDA/Metal performance smoke",
        "G08B final PASS",
        "G09 formal performance",
        "G10 release readiness",
    ),
)

METAL_SPEC = CheckpointSpec(
    backend="metal",
    checkpoint_id="runtime-vnext-g08b-metal-model-matrix",
    checkpoint_label="G08B-METAL-MATRIX",
    expected_case_count=702,
    required_client_concurrency=16,
    required_active_floor=4,
    required_active_duty_cycle=0.80,
    concurrency_cells=(1, 4, 16),
    pass_prefix="FERRUM RUNTIME VNEXT G08B METAL MODEL MATRIX PASS",
    selftest_pass_line="FERRUM RUNTIME VNEXT G08B METAL MODEL MATRIX SELFTEST PASS",
    does_not_prove=(
        "current-HEAD G08B CUDA GPTQ-Int4 product path",
        "G08B legacy/reference parity",
        "G08B mutation and legacy-deletion acceptance",
        "G08B CUDA/Metal performance smoke",
        "G08B final PASS",
        "G09 formal performance",
        "G10 release readiness",
    ),
)

CHECKPOINT_SPECS = {
    CUDA_SPEC.backend: CUDA_SPEC,
    METAL_SPEC.backend: METAL_SPEC,
}


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


def expected_case_count(source_git_sha: str, spec: CheckpointSpec) -> int:
    catalog = matrix.candidate_expectations_catalog(source_git_sha)
    return len(matrix.planned_case_rows(MODEL_KEY, spec.backend, catalog))


def summarize_matrix(
    report: dict[str, Any],
    spec: CheckpointSpec = CUDA_SPEC,
) -> dict[str, Any]:
    require(
        report.get("execution_contract") == matrix.G08_EXECUTION_CONTRACT,
        "scenario report is not a G08 candidate execution",
    )
    require(report.get("status") == "pass", "scenario report status is not pass")
    require(report.get("model_key") == MODEL_KEY, f"scenario report model must be {MODEL_KEY}")
    require(
        report.get("backend") == spec.backend,
        f"scenario report backend must be {spec.backend}",
    )
    require(
        report.get("models_lock_sha256") == file_sha256(spec.model_lock_path),
        (
            "scenario report is not bound to the checked-in "
            f"M2 {spec.backend.upper()} model lock"
        ),
    )
    source_git_sha = report.get("source_git_sha")
    require(
        isinstance(source_git_sha, str) and matrix.GIT_SHA_RE.fullmatch(source_git_sha),
        "scenario report source_git_sha is invalid",
    )
    planned_count = expected_case_count(source_git_sha, spec)
    require(
        planned_count == spec.expected_case_count,
        (
            f"canonical M2 {spec.backend.upper()} planner drifted from "
            f"{spec.expected_case_count} to {planned_count}"
        ),
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
        total_cases == spec.expected_case_count,
        (
            f"scenario report must pass exactly {spec.expected_case_count} "
            f"cases, got {total_cases}"
        ),
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
        == list(spec.concurrency_cells),
        (
            f"C18 {spec.backend.upper()} cells must be ordered "
            + "/".join(f"c{value}" for value in spec.concurrency_cells)
        ),
    )
    for raw_cell in cells:
        cell = require_object(raw_cell, "C18 concurrency cell")
        requested = require_count(cell.get("requested_concurrency"), "C18 requested concurrency", minimum=1)
        matrix.validate_c18_resource_balance_summary(
            cell.get("resource_balance"),
            expected_request_count=requested,
            label=f"C18.c{requested}.resource_balance",
        )
    top_label = f"C18.c{spec.required_client_concurrency}"
    top_cell = require_object(cells[-1], top_label)
    cap = require_count(
        top_cell.get("typed_admission_cap"),
        f"{top_label}.typed_admission_cap",
        minimum=1,
    )
    floor = require_count(
        top_cell.get("active_floor"),
        f"{top_label}.active_floor",
        minimum=1,
    )
    observed = require_count(
        top_cell.get("observed_max_active"),
        f"{top_label}.observed_max_active",
        minimum=1,
    )
    timeline = require_object(
        top_cell.get("active_timeline"),
        f"{top_label}.active_timeline",
    )
    duty = timeline.get("active_duty_cycle")
    require(
        cap >= spec.required_active_floor,
        (
            f"{top_label} typed admission cap is below "
            f"{spec.required_active_floor}"
        ),
    )
    require(
        floor == spec.required_active_floor,
        f"{top_label} active floor must equal {spec.required_active_floor}",
    )
    require(
        spec.required_active_floor <= observed <= cap,
        f"{top_label} observed max-active does not reach the typed active floor",
    )
    require(
        isinstance(duty, (int, float))
        and not isinstance(duty, bool)
        and math.isfinite(float(duty))
        and float(duty) >= spec.required_active_duty_cycle,
        (
            f"{top_label} active duty-cycle is below "
            f"{spec.required_active_duty_cycle:.2f}"
        ),
    )
    require(
        top_cell.get("request_count") == spec.required_client_concurrency
        and top_cell.get("completed_request_count")
        == spec.required_client_concurrency
        and top_cell.get("completion_rate") == 1.0,
        (
            f"{top_label} request completion is not "
            f"{spec.required_client_concurrency}/{spec.required_client_concurrency}"
        ),
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
        require(top_cell.get(field) == 0, f"{top_label}.{field} must be 0")

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
            "requested_concurrency": spec.required_client_concurrency,
            "typed_admission_cap": cap,
            "active_floor": floor,
            "observed_max_active": observed,
            "active_duty_cycle": float(duty),
            "resource_balance": copy.deepcopy(top_cell["resource_balance"]),
        },
    }


def validate_report(
    artifact_root: Path,
    report_path: Path,
    spec: CheckpointSpec = CUDA_SPEC,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    return report, summarize_matrix(report, spec)


def write_checkpoint(
    artifact_root: Path,
    report_path: Path,
    out: Path,
    spec: CheckpointSpec = CUDA_SPEC,
) -> dict[str, Any]:
    report, summary = validate_report(artifact_root, report_path, spec)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "manifest.json").exists(), "checkpoint output already contains manifest.json")
    pass_line = f"{spec.pass_prefix}: {out}"
    validation = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": f"{spec.artifact_type_prefix}_validation",
        "checkpoint_id": spec.checkpoint_id,
        "status": "pass",
        "validated_at": iso_now(),
        "execution_contract": matrix.G08_EXECUTION_CONTRACT,
        "source_git_sha": report["source_git_sha"],
        "source_tree_sha": report["source_tree_sha"],
        "model_key": MODEL_KEY,
        "backend": spec.backend,
        "model_revision": report["model_revision"],
        "model_files": report["model_files"],
        "binary_sha256": report["binary_sha256"],
        "models_lock_sha256": report["models_lock_sha256"],
        "scenario_report": {
            "path": str(report_path.resolve()),
            "sha256": file_sha256(report_path.resolve()),
        },
        "summary": summary,
        "does_not_prove": list(spec.does_not_prove),
        "pass_line": pass_line,
    }
    validation_path = out / "validation.json"
    write_json(validation_path, validation)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": f"{spec.artifact_type_prefix}_manifest",
        "lane": spec.checkpoint_id,
        "checkpoint_id": spec.checkpoint_label,
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
        "does_not_prove": list(spec.does_not_prove),
        "pass_line": pass_line,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def fixture_report(spec: CheckpointSpec = CUDA_SPEC) -> dict[str, Any]:
    source_git_sha = "1" * 40
    rows = matrix.planned_case_rows(
        MODEL_KEY,
        spec.backend,
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
            typed_cap = spec.required_active_floor
            for ordinal, requested in enumerate(spec.concurrency_cells, start=1):
                case_id = f"c18-{ordinal:03d}"
                floor = matrix.required_active_floor(
                    MODEL_KEY,
                    spec.backend,
                    requested,
                )
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
        "models_lock_sha256": file_sha256(spec.model_lock_path),
        "model_key": MODEL_KEY,
        "backend": spec.backend,
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
    spec: CheckpointSpec = CUDA_SPEC,
) -> None:
    candidate = copy.deepcopy(report)
    mutate(candidate)
    try:
        summarize_matrix(candidate, spec)
    except (ValidationError, matrix.ScenarioError) as error:
        require(marker.lower() in str(error).lower(), f"{name} rejected unexpectedly: {error}")
        return
    raise AssertionError(f"{name} unexpectedly passed")


def self_test(spec: CheckpointSpec = CUDA_SPEC) -> int:
    preparation_spec = PREPARATION_SPECS[spec.backend]
    locked_sources = matrix.locked_execution_sources(
        matrix.read_json(spec.model_lock_path),
        MODEL_KEY,
        spec.backend,
    )
    require(
        len(locked_sources["weight_files"]) == preparation_spec.weight_file_count,
        f"checked-in M2 {spec.backend.upper()} weight lock is incomplete",
    )
    require(
        len(locked_sources["semantic_source"]["files"])
        == preparation_spec.semantic_file_count,
        "checked-in M2 semantic lock is incomplete",
    )
    require(
        locked_sources["weight_revision"] == preparation_spec.weight_revision,
        f"checked-in M2 {spec.backend.upper()} revision drift",
    )
    report = fixture_report(spec)
    summary = summarize_matrix(report, spec)
    require(summary["scenario_count"] == 21, "fixture scenario count mismatch")
    require(
        summary["case_count"] == spec.expected_case_count,
        "fixture case count mismatch",
    )
    require(
        summary["c18"]["active_duty_cycle"]
        >= spec.required_active_duty_cycle,
        "fixture top concurrency duty-cycle mismatch",
    )
    expect_reject(
        report,
        "wrong model lock",
        lambda value: value.update({"models_lock_sha256": "0" * 64}),
        f"checked-in M2 {spec.backend.upper()} model lock",
        spec,
    )
    expect_reject(
        report,
        "known failure",
        lambda value: value["scenarios"][0].update(
            {"status": "known-fail", "known_failed_count": 1}
        ),
        "status is not pass",
        spec,
    )
    expect_reject(
        report,
        "missing scenario",
        lambda value: value["scenarios"].pop(),
        "ordered C01-C21",
        spec,
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
        f"exactly {spec.expected_case_count} cases",
        spec,
    )
    expect_reject(
        report,
        "missing serve",
        lambda value: value.update({"commands": [{"id": "actual-run", "entrypoint": "run"}]}),
        "run and serve",
        spec,
    )
    expect_reject(
        report,
        "low top-cell duty",
        lambda value: value["scenarios"][17]["concurrency_cells"][-1]["active_timeline"].update(
            {"active_duty_cycle": spec.required_active_duty_cycle - 0.01}
        ),
        f"below {spec.required_active_duty_cycle:.2f}",
        spec,
    )
    expect_reject(
        report,
        "low top-cell cap",
        lambda value: value["scenarios"][17]["concurrency_cells"][-1].update(
            {"typed_admission_cap": spec.required_active_floor - 1}
        ),
        f"below {spec.required_active_floor}",
        spec,
    )
    expect_reject(
        report,
        "resource leak",
        lambda value: value["scenarios"][17]["concurrency_cells"][-1]["resource_balance"].update(
            {"leaked_resource_count": 1}
        ),
        "leaked_resource_count",
        spec,
    )
    print(spec.selftest_pass_line)
    return 0


def main(
    *,
    default_backend: str = "cuda",
    fixed_backend: bool = False,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--scenario-report", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    if fixed_backend:
        parser.set_defaults(backend=default_backend)
    else:
        parser.add_argument(
            "--backend",
            choices=tuple(CHECKPOINT_SPECS),
            default=default_backend,
        )
    args = parser.parse_args()
    spec = CHECKPOINT_SPECS[args.backend]
    if args.self_test:
        return self_test(spec)
    if args.artifact_root is None or args.scenario_report is None or args.out is None:
        parser.error("--artifact-root, --scenario-report, and --out are required")
    try:
        manifest = write_checkpoint(
            args.artifact_root,
            args.scenario_report,
            args.out,
            spec,
        )
    except (ValidationError, matrix.ScenarioError, OSError, ValueError) as error:
        print(
            (
                "FERRUM RUNTIME VNEXT G08B "
                f"{spec.backend.upper()} MODEL MATRIX FAIL: {args.out}: {error}"
            ),
            file=sys.stderr,
        )
        return 1
    print(manifest["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
