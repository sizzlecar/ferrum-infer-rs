#!/usr/bin/env python3
"""Aggregate canonical G08A evidence into the Qwen3.5-4B migration gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PASS_PREFIX = "FERRUM RUNTIME VNEXT G08A QWEN35 4B PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G08A CHECKPOINT SELFTEST PASS"
MODEL_KEY = "m1-qwen35-4b"
HISTORICAL_CASE_IDS = ("H02.1", "H12.1", "H12.2", "H12.3", "H12.4")
SOURCE_SUMMARY = {
    "maximum_provider_files": 8,
    "maximum_provider_glue_loc": 1500,
    "minimum_scaffolding_reduction_ratio": 0.60,
    "required_lifecycle_categories": 5,
    "required_lifecycle_owner_count": 1,
    "required_legacy_selection_count": 0,
}
MATRIX_REQUIREMENTS = {
    "cuda": {
        "case_count": 703,
        "concurrency": 32,
        "active_floor": 32,
        "duty_cycle": 0.80,
        "artifact_type": "runtime_vnext_g08a_cuda_model_matrix_manifest",
        "child_lane": "runtime-vnext-g08a-cuda-model-matrix",
    },
    "metal": {
        "case_count": 702,
        "concurrency": 16,
        "active_floor": 16,
        "duty_cycle": 0.80,
        "artifact_type": "runtime_vnext_g08a_metal_model_matrix_manifest",
        "child_lane": "runtime-vnext-g08a-metal-model-matrix",
    },
}
OUTER_GATE_FIELDS = {
    "artifact_dir",
    "binary",
    "child_artifacts",
    "child_execution_artifacts",
    "child_pass_line",
    "child_returncode",
    "command_line",
    "delegated_command_line",
    "dirty_status",
    "duration_sec",
    "error",
    "finished_at",
    "git_sha",
    "lane",
    "model",
    "pass_line",
    "sanitized_env",
    "schema_version",
    "started_at",
    "status",
}


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class DependencySpec:
    lane: str
    child_relative: str
    child_pass_prefix: str
    kind: str
    backend: str | None = None


DEPENDENCY_SPECS = {
    "source": DependencySpec(
        "vnext-g08a-source",
        "manifest.json",
        "FERRUM RUNTIME VNEXT G08A SOURCE OWNERSHIP PASS",
        "source",
    ),
    "cuda": DependencySpec(
        "vnext-g08a-cuda",
        "manifest.json",
        "FERRUM RUNTIME VNEXT G08A CUDA MODEL MATRIX PASS",
        "matrix",
        "cuda",
    ),
    "metal": DependencySpec(
        "vnext-g08a-metal",
        "manifest.json",
        "FERRUM RUNTIME VNEXT G08A METAL MODEL MATRIX PASS",
        "matrix",
        "metal",
    ),
    "numerics": DependencySpec(
        "vnext-g08a-numerics",
        "manifest.json",
        "FERRUM RUNTIME VNEXT G08A NUMERICS PASS",
        "numerics",
    ),
    "s2": DependencySpec(
        "vnext-s2",
        "s2-product-contract/manifest.json",
        "FERRUM RUNTIME VNEXT S2 CUDA PRODUCT CONTRACT PASS",
        "s2",
    ),
    "cuda_performance": DependencySpec(
        "vnext-g08-performance-smoke",
        "manifest.json",
        "FERRUM RUNTIME VNEXT G08 PERFORMANCE SMOKE PASS",
        "performance",
        "cuda",
    ),
    "metal_performance": DependencySpec(
        "vnext-g08-performance-smoke",
        "manifest.json",
        "FERRUM RUNTIME VNEXT G08 PERFORMANCE SMOKE PASS",
        "performance",
        "metal",
    ),
}

DEPENDENCY_SCRIPTS = {
    "source": "scripts/release/runtime_vnext_g08a_source_contract.py",
    "cuda": "scripts/release/runtime_vnext_g08a_cuda_matrix_checkpoint.py",
    "metal": "scripts/release/runtime_vnext_g08a_metal_matrix_checkpoint.py",
    "numerics": "scripts/release/runtime_vnext_g08a_numerics.py",
    "s2": "scripts/release/runtime_vnext_s2_cuda_product_contract.py",
    "cuda_performance": "scripts/release/runtime_vnext_g08_performance_smoke.py",
    "metal_performance": "scripts/release/runtime_vnext_g08_performance_smoke.py",
}

PERFORMANCE_THRESHOLDS = {
    "legacy": 0.90,
    "external": 0.70,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"invalid {label} JSON {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(resolved.is_file() and not resolved.is_symlink(), f"artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_ref(value: Any, label: str) -> Path:
    require(isinstance(value, dict), f"{label} reference is missing")
    require(set(value) >= {"path", "sha256"}, f"{label} reference fields are incomplete")
    path = Path(str(value["path"])).expanduser().resolve()
    require(path.is_file() and not path.is_symlink(), f"{label} file is missing: {path}")
    require(sha256(path) == value["sha256"], f"{label} SHA256 mismatch")
    if "size_bytes" in value:
        require(path.stat().st_size == value["size_bytes"], f"{label} size mismatch")
    return path


def command_flag(command: list[str], flag: str, label: str) -> str:
    require(command.count(flag) == 1, f"{label} must contain {flag} exactly once")
    index = command.index(flag)
    require(index + 1 < len(command), f"{label} {flag} value is missing")
    value = command[index + 1]
    require(value and not value.startswith("--"), f"{label} {flag} value is invalid")
    return value


def validate_delegated_command(
    command: Any,
    *,
    key: str,
    declared_root: Path,
) -> list[str]:
    require(
        isinstance(command, list)
        and len(command) >= 4
        and all(isinstance(value, str) and value for value in command),
        f"{key} delegated command is invalid",
    )
    require(
        command[1] == DEPENDENCY_SCRIPTS[key],
        f"{key} delegated script mismatch",
    )
    require(
        Path(command_flag(command, "--out", f"{key} delegated command"))
        .expanduser()
        .resolve()
        == declared_root,
        f"{key} delegated output mismatch",
    )
    return command


def validate_outer_command(command: Any, spec: DependencySpec, root: Path) -> None:
    require(
        isinstance(command, list)
        and len(command) >= 5
        and all(isinstance(value, str) and value for value in command),
        f"{spec.lane} outer command is invalid",
    )
    require(
        command[1:3] == ["scripts/release/run_gate.py", spec.lane],
        f"{spec.lane} outer command identity mismatch",
    )
    require(
        Path(command_flag(command, "--out", f"{spec.lane} outer command"))
        .expanduser()
        .resolve()
        == root,
        f"{spec.lane} outer command output mismatch",
    )


def git_text(source_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=source_root,
        text=True,
        capture_output=True,
        check=False,
    )
    require(process.returncode == 0, f"git {' '.join(args)} failed: {process.stderr.strip()}")
    return process.stdout.strip()


def source_identity(source_root: Path) -> dict[str, Any]:
    status = [line for line in git_text(source_root, "status", "--short").splitlines() if line]
    require(not status, f"G08A source must be clean: {status[:8]}")
    return {
        "git_sha": git_text(source_root, "rev-parse", "HEAD"),
        "git_tree_sha": git_text(source_root, "rev-parse", "HEAD^{tree}"),
        "dirty": False,
    }


def validate_validation_ref(root: Path, child: dict[str, Any], source: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    path = validate_ref(child.get("validation"), f"{label} validation")
    require(path.parent == root, f"{label} validation must be inside its child root")
    value = read_json(path, f"{label} validation")
    require(value.get("status") == "pass", f"{label} validation status is not pass")
    require(value.get("source_git_sha") == source["git_sha"], f"{label} validation source SHA is stale")
    require(value.get("source_tree_sha") == source["git_tree_sha"], f"{label} validation source tree is stale")
    return path, value


def validate_source_child(root: Path, child: dict[str, Any], source: dict[str, Any], *, verify_checkout: bool) -> dict[str, Any]:
    require(
        child.get("artifact_type") == "runtime_vnext_g08a_source_ownership_manifest"
        and child.get("lane") == "runtime-vnext-g08a-source-ownership",
        "G08A source child identity mismatch",
    )
    summary = child.get("summary")
    require(isinstance(summary, dict), "G08A source summary is missing")
    require(summary.get("provider_file_count", 10**9) <= SOURCE_SUMMARY["maximum_provider_files"], "G08A provider file limit failed")
    require(summary.get("provider_glue_production_loc", 10**9) <= SOURCE_SUMMARY["maximum_provider_glue_loc"], "G08A provider LOC limit failed")
    require(summary.get("scaffolding_reduction_ratio", -1.0) >= SOURCE_SUMMARY["minimum_scaffolding_reduction_ratio"], "G08A scaffolding reduction failed")
    require(summary.get("lifecycle_ownership_categories") == SOURCE_SUMMARY["required_lifecycle_categories"], "G08A lifecycle category ownership failed")
    require(summary.get("lifecycle_implementation_owner_count") == SOURCE_SUMMARY["required_lifecycle_owner_count"], "G08A lifecycle owner count failed")
    require(summary.get("legacy_source_selection_count") == SOURCE_SUMMARY["required_legacy_selection_count"], "G08A legacy selection remains")
    validation_path = validate_ref(child.get("validation"), "G08A source validation")
    require(validation_path.parent == root, "G08A source validation escaped its root")
    if verify_checkout:
        import runtime_vnext_g08a_source_contract as source_contract

        verified = source_contract.verify_manifest(root / "manifest.json", verify_checkout=True)
        require(
            verified.get("source_git_sha") == source["git_sha"]
            and verified.get("source_tree_sha") == source["git_tree_sha"]
            and verified.get("dirty") is False,
            "G08A source deep verification source mismatch",
        )
    return {"summary": copy.deepcopy(summary), "validation": file_ref(validation_path)}


def validate_matrix_child(
    root: Path,
    child: dict[str, Any],
    source: dict[str, Any],
    backend: str,
    *,
    delegated_command: list[str],
    verify_checkout: bool,
) -> dict[str, Any]:
    requirement = MATRIX_REQUIREMENTS[backend]
    require(
        child.get("artifact_type") == requirement["artifact_type"]
        and child.get("lane") == requirement["child_lane"]
        and child.get("model_key", MODEL_KEY) == MODEL_KEY,
        f"G08A {backend} matrix child identity mismatch",
    )
    _, validation = validate_validation_ref(root, child, source, f"G08A {backend} matrix")
    require(validation.get("model_key") == MODEL_KEY and validation.get("backend") == backend, f"G08A {backend} model/backend mismatch")
    summary = child.get("summary")
    require(isinstance(summary, dict) and summary == validation.get("summary"), f"G08A {backend} summary binding mismatch")
    require(summary.get("scenario_count") == 21, f"G08A {backend} scenario denominator mismatch")
    require(summary.get("case_count") == summary.get("passed_case_count") == requirement["case_count"], f"G08A {backend} case matrix incomplete")
    for field in ("known_failed_count", "blocked_count", "error_count", "unexpected_count"):
        require(summary.get(field) == 0, f"G08A {backend} {field} must be zero")
    require(summary.get("entrypoints") == ["run", "serve"], f"G08A {backend} lacks run/serve coverage")
    c18 = summary.get("c18")
    require(isinstance(c18, dict), f"G08A {backend} C18 summary is missing")
    require(c18.get("requested_concurrency") == requirement["concurrency"], f"G08A {backend} C18 concurrency mismatch")
    require(c18.get("active_floor") == requirement["active_floor"], f"G08A {backend} C18 active floor mismatch")
    require(c18.get("observed_max_active", 0) >= requirement["active_floor"], f"G08A {backend} C18 active floor was not reached")
    require(c18.get("active_duty_cycle", 0.0) >= requirement["duty_cycle"], f"G08A {backend} C18 duty cycle failed")
    scenario_path = validate_ref(child.get("scenario_report"), f"G08A {backend} scenario report")
    report = read_json(scenario_path, f"G08A {backend} scenario report")
    require(report.get("status") == "pass" and report.get("source_git_sha") == source["git_sha"] and report.get("source_tree_sha") == source["git_tree_sha"], f"G08A {backend} scenario report is stale")
    require(report.get("model_key") == MODEL_KEY and report.get("backend") == backend, f"G08A {backend} scenario report identity mismatch")
    artifact_root = Path(
        command_flag(
            delegated_command,
            "--artifact-root",
            f"G08A {backend} delegated command",
        )
    ).expanduser().resolve()
    command_report = Path(
        command_flag(
            delegated_command,
            "--scenario-report",
            f"G08A {backend} delegated command",
        )
    ).expanduser().resolve()
    require(command_report == scenario_path, f"G08A {backend} command/report binding mismatch")
    require(
        command_report.is_relative_to(artifact_root),
        f"G08A {backend} scenario report escaped its raw artifact root",
    )
    if verify_checkout:
        import runtime_vnext_g08a_matrix_specs as matrix_specs
        import runtime_vnext_g08b_cuda_matrix_checkpoint as matrix_checkpoint

        matrix_specs.validate_model_lock_contract(backend)
        deep_report, deep_summary = matrix_checkpoint.validate_report(
            artifact_root,
            command_report,
            matrix_specs.CHECKPOINT_SPECS[backend],
        )
        require(deep_report == report, f"G08A {backend} deep report binding mismatch")
        require(deep_summary == summary, f"G08A {backend} deep summary mismatch")
    return {"summary": copy.deepcopy(summary), "scenario_report": file_ref(scenario_path)}


def validate_numerics_child(
    root: Path,
    child: dict[str, Any],
    source: dict[str, Any],
    *,
    delegated_command: list[str],
    verify_checkout: bool,
) -> dict[str, Any]:
    require(
        child.get("artifact_type") == "runtime_vnext_g08a_numerics_manifest"
        and child.get("lane") == "runtime-vnext-g08a-numerics",
        "G08A numerics child identity mismatch",
    )
    _, validation = validate_validation_ref(root, child, source, "G08A numerics")
    require(validation.get("artifact_local_tolerance_count") == 0, "G08A numerics contains artifact-local tolerances")
    require(validation.get("operation_state_row_count", 0) >= 27, "G08A operation/state numerical coverage is incomplete")
    require(validation.get("layer_checkpoint_count") == 2, "G08A layer numerical coverage is incomplete")
    require(validation.get("full_model_checkpoint_count") == 1, "G08A full-model numerical checkpoint is missing")
    require(validation.get("full_vocabulary_logits_checkpoint_count") == 1, "G08A full-vocabulary logits checkpoint is missing")
    parity = validation.get("token_parity")
    require(isinstance(parity, dict), "G08A token parity summary is missing")
    require(parity.get("case_count") == 20, "G08A token parity must pass 20/20 prompts")
    require(parity.get("token_count_per_case") == 64, "G08A token parity token denominator mismatch")
    require(parity.get("matched_token_count") == 1280, "G08A token parity matched-token denominator mismatch")
    require(parity.get("exception_count") == 0, "G08A token parity contains generated-token exceptions")
    require(parity.get("waiver_count") == 0, "G08A token parity contains waivers")
    expected_child_summary = {
        "catalog_row_count": validation.get("catalog_row_count"),
        "operation_state_row_count": validation.get("operation_state_row_count"),
        **parity,
    }
    require(
        child.get("summary") == expected_child_summary,
        "G08A numerical child/validation summary mismatch",
    )
    inputs = validation.get("inputs")
    require(isinstance(inputs, dict) and len(inputs) == 5, "G08A numerical input set is incomplete")
    input_paths = {
        key: validate_ref(value, f"G08A numerics input {key}")
        for key, value in inputs.items()
    }
    require(child.get("inputs") == inputs, "G08A numerical child/input binding mismatch")
    expected_flags = {
        "metal_op_numerics": "--g08a-op-numerics",
        "linear_attention": "--g08a-linear-attention",
        "full_attention": "--g08a-full-attention",
        "full_model": "--g08a-full-model",
        "token_parity": "--g08a-token-parity",
    }
    command_paths = {
        key: Path(
            command_flag(
                delegated_command,
                flag,
                "G08A numerics delegated command",
            )
        )
        .expanduser()
        .resolve()
        for key, flag in expected_flags.items()
    }
    require(
        command_paths["metal_op_numerics"] / "metal-op-numerics.json"
        == input_paths["metal_op_numerics"],
        "G08A op numerics command/input binding mismatch",
    )
    for key in ("linear_attention", "full_attention", "full_model", "token_parity"):
        require(command_paths[key] == input_paths[key], f"G08A {key} command/input binding mismatch")
    if verify_checkout:
        import runtime_vnext_g08a_numerics as numerics

        with tempfile.TemporaryDirectory(prefix="g08a-numerics-revalidate-") as tmp:
            deep_manifest = numerics.validate_and_write(
                op_artifact=command_paths["metal_op_numerics"],
                linear_gate_path=command_paths["linear_attention"],
                full_attention_gate_path=command_paths["full_attention"],
                model_gate_path=command_paths["full_model"],
                token_parity_path=command_paths["token_parity"],
                out=Path(tmp) / "out",
                require_clean=True,
            )
        require(
            deep_manifest.get("source_git_sha") == source["git_sha"]
            and deep_manifest.get("source_tree_sha") == source["git_tree_sha"],
            "G08A numerics deep source mismatch",
        )
        require(deep_manifest.get("summary") == child.get("summary"), "G08A numerics deep summary mismatch")
    return {
        "operation_state_row_count": validation["operation_state_row_count"],
        "layer_checkpoint_count": validation["layer_checkpoint_count"],
        "token_parity": copy.deepcopy(parity),
    }


def validate_s2_child(root: Path, child: dict[str, Any], source: dict[str, Any], *, verify_checkout: bool) -> dict[str, Any]:
    require(
        child.get("artifact_type") == "runtime_vnext_s2_cuda_product_contract_manifest"
        and child.get("lane") == "runtime-vnext-s2"
        and child.get("checkpoint_id") == "S2",
        "G08A S2 child identity mismatch",
    )
    child_source = child.get("source")
    require(isinstance(child_source, dict), "G08A S2 source is missing")
    require(child_source.get("git_sha") == source["git_sha"] and child_source.get("git_tree_sha") == source["git_tree_sha"], "G08A S2 source is stale")
    acceptance = child.get("acceptance")
    require(isinstance(acceptance, dict) and acceptance and all(value is True for value in acceptance.values()), "G08A S2 acceptance is incomplete")
    children = child.get("children")
    require(isinstance(children, dict), "G08A S2 child matrix is missing")
    historical = children.get("historical_resource_source")
    require(isinstance(historical, dict), "G08A S2 historical resource child is missing")
    deep = historical.get("deep_validation")
    require(isinstance(deep, dict) and deep.get("case_count") == len(HISTORICAL_CASE_IDS), "G08A historical resource case denominator mismatch")
    require(deep.get("source_test_count") == 7, "G08A historical source test denominator mismatch")
    if verify_checkout:
        import runtime_vnext_s2_cuda_product_contract as s2_contract

        verified = s2_contract.verify_checkpoint_manifest(root / "manifest.json", verify_checkout=True)
        verified_source = verified.get("source")
        require(isinstance(verified_source, dict) and verified_source.get("git_sha") == source["git_sha"] and verified_source.get("git_tree_sha") == source["git_tree_sha"], "G08A S2 deep verification source mismatch")
    return {
        "historical_case_ids": list(HISTORICAL_CASE_IDS),
        "historical_case_count": deep["case_count"],
        "historical_source_test_count": deep["source_test_count"],
        "acceptance": copy.deepcopy(acceptance),
    }


def validate_performance_child(
    root: Path,
    child: dict[str, Any],
    source: dict[str, Any],
    backend: str,
    *,
    delegated_command: list[str],
    verify_checkout: bool,
    expected_model_key: str = MODEL_KEY,
) -> dict[str, Any]:
    require(
        child.get("artifact_type") == "runtime_vnext_g08_performance_smoke_manifest"
        and child.get("lane") == "runtime-vnext-g08-performance-smoke"
        and child.get("model_key") == expected_model_key
        and child.get("backend") == backend,
        f"G08A {backend} performance child identity mismatch",
    )
    _, validation = validate_validation_ref(root, child, source, f"G08A {backend} performance")
    require(validation.get("model_key") == expected_model_key and validation.get("backend") == backend, f"G08A {backend} performance validation identity mismatch")
    summary = validation.get("summary")
    require(isinstance(summary, dict), f"G08A {backend} performance summary is missing")
    require(child.get("summary") == summary, f"G08A {backend} performance child/validation summary mismatch")
    ratios = summary.get("ratios")
    require(isinstance(ratios, list) and len(ratios) == 2, f"G08A {backend} performance must contain two cells")
    expected_cells = [1, 32 if backend == "cuda" else 16]
    require([row.get("concurrency") for row in ratios if isinstance(row, dict)] == expected_cells, f"G08A {backend} performance cells mismatch")
    baseline_kind = summary.get("baseline_kind")
    require(baseline_kind in PERFORMANCE_THRESHOLDS, f"G08A {backend} performance baseline kind is invalid")
    threshold = PERFORMANCE_THRESHOLDS[baseline_kind]
    ratio_key = f"candidate_over_{baseline_kind}"
    for row in ratios:
        require(isinstance(row, dict) and row.get("passes") is True, f"G08A {backend} performance ratio failed")
        require(
            set(row) == {"concurrency", ratio_key, "threshold", "passes"}
            and row.get("threshold") == threshold,
            f"G08A {backend} performance threshold contract mismatch",
        )
        ratio = row.get(ratio_key)
        require(
            isinstance(ratio, (int, float))
            and not isinstance(ratio, bool)
            and math.isfinite(float(ratio))
            and float(ratio) >= threshold,
            f"G08A {backend} performance is below threshold",
        )
    artifact_root = Path(
        command_flag(
            delegated_command,
            "--artifact-root",
            f"G08A {backend} performance delegated command",
        )
    ).expanduser().resolve()
    require(
        Path(str(validation.get("artifact_root", ""))).expanduser().resolve()
        == artifact_root,
        f"G08A {backend} performance raw artifact binding mismatch",
    )
    if verify_checkout:
        import runtime_vnext_g08_performance_smoke as performance

        deep = performance.validate_artifact(artifact_root)
        contract = deep.pop("contract")
        require(
            contract.get("source_git_sha") == source["git_sha"]
            and contract.get("source_tree_sha") == source["git_tree_sha"]
            and contract.get("model_key") == expected_model_key
            and contract.get("backend") == backend,
            f"G08A {backend} performance deep contract mismatch",
        )
        require(deep == summary, f"G08A {backend} performance deep summary mismatch")
    return {"baseline_kind": summary.get("baseline_kind"), "ratios": copy.deepcopy(ratios)}


def validate_child(
    root: Path,
    child: dict[str, Any],
    spec: DependencySpec,
    source: dict[str, Any],
    *,
    declared_root: Path,
    delegated_command: list[str],
    verify_checkout: bool,
    expected_model_key: str = MODEL_KEY,
) -> dict[str, Any]:
    require(child.get("schema_version") == 1 and child.get("status") == "pass" and child.get("canonical") is True, f"{spec.lane} child status is not canonical PASS")
    require(Path(str(child.get("artifact_dir", ""))).expanduser().resolve() == root, f"{spec.lane} child artifact_dir mismatch")
    require(
        child.get("pass_line") == f"{spec.child_pass_prefix}: {declared_root}",
        f"{spec.lane} child PASS line mismatch",
    )
    require(child.get("source_git_sha", source["git_sha"]) == source["git_sha"], f"{spec.lane} child source SHA is stale")
    require(child.get("source_tree_sha", source["git_tree_sha"]) == source["git_tree_sha"], f"{spec.lane} child source tree is stale")
    require(child.get("dirty", False) is False, f"{spec.lane} child is dirty")
    if spec.kind == "source":
        return validate_source_child(root, child, source, verify_checkout=verify_checkout)
    if spec.kind == "matrix":
        assert spec.backend is not None
        return validate_matrix_child(
            root,
            child,
            source,
            spec.backend,
            delegated_command=delegated_command,
            verify_checkout=verify_checkout,
        )
    if spec.kind == "numerics":
        return validate_numerics_child(
            root,
            child,
            source,
            delegated_command=delegated_command,
            verify_checkout=verify_checkout,
        )
    if spec.kind == "s2":
        return validate_s2_child(root, child, source, verify_checkout=verify_checkout)
    if spec.kind == "performance":
        assert spec.backend is not None
        return validate_performance_child(
            root,
            child,
            source,
            spec.backend,
            delegated_command=delegated_command,
            verify_checkout=verify_checkout,
            expected_model_key=expected_model_key,
        )
    raise CheckpointError(f"unknown G08A dependency kind: {spec.kind}")


def validate_outer(
    path: Path,
    key: str,
    spec: DependencySpec,
    source: dict[str, Any],
    *,
    verify_checkout: bool,
) -> dict[str, Any]:
    outer_path = path.expanduser().resolve()
    outer = read_json(outer_path, f"{spec.lane} outer manifest")
    require(set(outer) == OUTER_GATE_FIELDS, f"{spec.lane} outer field set mismatch")
    root = outer_path.parent
    require(
        outer.get("schema_version") == 1
        and outer.get("lane") == spec.lane
        and outer.get("status") == "pass"
        and outer.get("child_returncode") == 0
        and outer.get("error") is None
        and Path(str(outer.get("artifact_dir", ""))).expanduser().resolve() == root,
        f"{spec.lane} outer identity/status mismatch",
    )
    require(outer.get("git_sha") == source["git_sha"] and outer.get("dirty_status") == {"is_dirty": False, "status_short": []}, f"{spec.lane} outer source is stale or dirty")
    validate_outer_command(outer.get("command_line"), spec, root)
    delegated_command = validate_delegated_command(
        outer.get("delegated_command_line"),
        key=key,
        declared_root=root,
    )
    expected_child_pass = f"{spec.child_pass_prefix}: {root}"
    require(outer.get("child_pass_line") == expected_child_pass, f"{spec.lane} outer child PASS mismatch")
    require(outer.get("pass_line") == f"FERRUM GATE {spec.lane} PASS: {root}", f"{spec.lane} outer PASS mismatch")
    child_path = root / spec.child_relative
    child_artifacts = outer.get("child_artifacts")
    require(isinstance(child_artifacts, dict), f"{spec.lane} outer child provenance is missing")
    require(
        child_artifacts.get("kind") == spec.lane
        and child_artifacts.get("source") == source,
        f"{spec.lane} outer child provenance identity mismatch",
    )
    child_ref = child_artifacts.get("child_manifest")
    require(isinstance(child_ref, dict), f"{spec.lane} outer lacks child manifest binding")
    require(validate_ref(child_ref, f"{spec.lane} child manifest") == child_path, f"{spec.lane} child manifest path mismatch")
    execution_refs = outer.get("child_execution_artifacts")
    require(isinstance(execution_refs, list) and len(execution_refs) == 3, f"{spec.lane} child execution receipt set mismatch")
    expected_execution_paths = {"run_gate.child.command.json", "run_gate.child.stdout", "run_gate.child.stderr"}
    require({row.get("path") for row in execution_refs if isinstance(row, dict)} == expected_execution_paths, f"{spec.lane} child execution receipt names mismatch")
    execution_paths: dict[str, Path] = {}
    for row in execution_refs:
        require(isinstance(row, dict), f"{spec.lane} child execution receipt is invalid")
        execution_paths[str(row["path"])] = validate_ref(
            {**row, "path": str(root / str(row["path"]))},
            f"{spec.lane} {row['path']}",
        )
    command_receipt = read_json(
        execution_paths["run_gate.child.command.json"],
        f"{spec.lane} child command receipt",
    )
    duration = command_receipt.get("duration_sec")
    require(
        set(command_receipt) == {"cmd", "duration_sec", "env_overrides"}
        and command_receipt.get("cmd") == delegated_command
        and isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(float(duration))
        and float(duration) >= 0.0
        and isinstance(command_receipt.get("env_overrides"), dict),
        f"{spec.lane} child command receipt mismatch",
    )
    stdout = (root / "run_gate.child.stdout").read_text(encoding="utf-8")
    require(stdout.splitlines().count(expected_child_pass) == 1, f"{spec.lane} child stdout lacks exactly one PASS line")
    child = read_json(child_path, f"{spec.lane} child manifest")
    summary = validate_child(
        child_path.parent,
        child,
        spec,
        source,
        declared_root=root,
        delegated_command=delegated_command,
        verify_checkout=verify_checkout,
    )
    return {
        "outer_manifest": file_ref(outer_path),
        "child_manifest": file_ref(child_path),
        "summary": summary,
    }


def verify_dependency_manifest(
    key: str,
    manifest_path: Path,
    delegated_command: list[str],
    *,
    source_root: Path = REPO_ROOT,
    verify_checkout: bool = True,
    expected_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    require(key in DEPENDENCY_SPECS, f"unknown G08A dependency: {key}")
    spec = DEPENDENCY_SPECS[key]
    path = manifest_path.expanduser().resolve()
    child_root = path.parent
    declared_root = child_root.parent if key == "s2" else child_root
    source = source_identity(source_root) if verify_checkout else expected_source
    require(isinstance(source, dict), f"{key} expected source is missing")
    command = validate_delegated_command(
        delegated_command,
        key=key,
        declared_root=declared_root,
    )
    child = read_json(path, f"{key} child manifest")
    expected_model_key = child.get("model_key", MODEL_KEY)
    if spec.kind == "performance":
        import runtime_vnext_g08_performance_smoke as performance

        require(
            expected_model_key in performance.MODEL_KEYS,
            f"{key} performance model key is unsupported",
        )
    summary = validate_child(
        child_root,
        child,
        spec,
        source,
        declared_root=declared_root,
        delegated_command=command,
        verify_checkout=verify_checkout,
        expected_model_key=expected_model_key,
    )
    return {
        "kind": spec.lane,
        "child_manifest": file_ref(path),
        "source": source,
        "summary": summary,
    }


def validate_dependencies(paths: dict[str, Path], source: dict[str, Any], *, verify_checkout: bool) -> dict[str, Any]:
    require(set(paths) == set(DEPENDENCY_SPECS), "G08A dependency path set mismatch")
    dependencies = {
        key: validate_outer(
            paths[key],
            key,
            spec,
            source,
            verify_checkout=verify_checkout,
        )
        for key, spec in DEPENDENCY_SPECS.items()
    }
    require(dependencies["cuda_performance"]["outer_manifest"]["path"] != dependencies["metal_performance"]["outer_manifest"]["path"], "G08A CUDA and Metal performance artifacts must be distinct")
    return dependencies


def acceptance(dependencies: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_key": MODEL_KEY,
        "source_ownership": "pass",
        "cuda_cases": "703/703",
        "metal_cases": "702/702",
        "waiver_count": 0,
        "historical_cases": f"{len(HISTORICAL_CASE_IDS)}/{len(HISTORICAL_CASE_IDS)}",
        "historical_case_ids": list(HISTORICAL_CASE_IDS),
        "numerical_operation_state_rows": dependencies["numerics"]["summary"]["operation_state_row_count"],
        "numerical_layer_checkpoints": dependencies["numerics"]["summary"]["layer_checkpoint_count"],
        "token_parity": "20/20",
        "performance_smoke_backends": ["cuda", "metal"],
        "product_entrypoints": ["run", "serve"],
    }


def build_manifest(output: Path, source: dict[str, Any], dependencies: dict[str, Any], validation_ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g08a_qwen35_4b_checkpoint",
        "checkpoint_id": "G08A",
        "lane": "runtime-vnext-g08a",
        "status": "pass",
        "canonical": True,
        "artifact_dir": str(output),
        "source": source,
        "dependencies": dependencies,
        "acceptance": acceptance(dependencies),
        "validation": validation_ref,
        "unlocks": ["S4", "G08B"],
        "does_not_prove": ["G08", "G09", "G10", "release readiness"],
        "pass_line": f"{PASS_PREFIX}: {output}",
    }


def verify_checkpoint_manifest(path: Path, *, source_root: Path = REPO_ROOT, verify_checkout: bool = True, expected_source: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    root = manifest_path.parent
    manifest = read_json(manifest_path, "G08A checkpoint manifest")
    required = {"schema_version", "artifact_type", "checkpoint_id", "lane", "status", "canonical", "artifact_dir", "source", "dependencies", "acceptance", "validation", "unlocks", "does_not_prove", "pass_line"}
    require(set(manifest) == required, "G08A checkpoint field set mismatch")
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type") == "runtime_vnext_g08a_qwen35_4b_checkpoint"
        and manifest.get("checkpoint_id") == "G08A"
        and manifest.get("lane") == "runtime-vnext-g08a"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True
        and Path(str(manifest.get("artifact_dir", ""))).expanduser().resolve() == root
        and manifest.get("unlocks") == ["S4", "G08B"]
        and manifest.get("does_not_prove") == ["G08", "G09", "G10", "release readiness"]
        and manifest.get("pass_line") == f"{PASS_PREFIX}: {root}",
        "G08A checkpoint identity/status/PASS mismatch",
    )
    source = source_identity(source_root) if verify_checkout else expected_source
    require(isinstance(source, dict) and manifest.get("source") == source, "G08A checkpoint source is stale")
    dependency_values = manifest.get("dependencies")
    require(isinstance(dependency_values, dict) and set(dependency_values) == set(DEPENDENCY_SPECS), "G08A checkpoint dependency set mismatch")
    paths = {
        key: Path(str(dependency_values[key]["outer_manifest"]["path"]))
        for key in DEPENDENCY_SPECS
    }
    dependencies = validate_dependencies(paths, source, verify_checkout=verify_checkout)
    require(dependency_values == dependencies, "G08A checkpoint dependencies drifted")
    require(manifest.get("acceptance") == acceptance(dependencies), "G08A checkpoint acceptance drifted")
    validation_path = validate_ref(manifest.get("validation"), "G08A aggregate validation")
    require(validation_path == root / "validation.json", "G08A aggregate validation path mismatch")
    validation = read_json(validation_path, "G08A aggregate validation")
    require(validation.get("status") == "pass" and validation.get("source") == source and validation.get("dependencies") == dependencies and validation.get("acceptance") == manifest["acceptance"] and validation.get("pass_line") == manifest["pass_line"], "G08A aggregate validation content mismatch")
    return {
        "kind": "vnext-g08a",
        "child_manifest": file_ref(manifest_path),
        "source": source,
        "dependency_count": len(dependencies),
        "acceptance": manifest["acceptance"],
    }


def build_checkpoint(paths: dict[str, Path], output: Path, *, source_root: Path = REPO_ROOT, verify_checkout: bool = True, expected_source: dict[str, Any] | None = None) -> str:
    source = source_identity(source_root) if verify_checkout else expected_source
    require(isinstance(source, dict), "G08A expected source is missing")
    output = output.expanduser().resolve()
    require(not output.is_relative_to(source_root.resolve()), "G08A output must be outside the source tree")
    require(not output.exists(), f"G08A output already exists: {output}")
    dependencies = validate_dependencies(paths, source, verify_checkout=verify_checkout)
    if verify_checkout:
        require(source_identity(source_root) == source, "G08A source changed during dependency validation")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        pass_line = f"{PASS_PREFIX}: {output}"
        validation = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_g08a_qwen35_4b_validation",
            "status": "pass",
            "validated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "source": source,
            "dependencies": dependencies,
            "acceptance": acceptance(dependencies),
            "pass_line": pass_line,
        }
        write_json(staging / "validation.json", validation, exclusive=True)
        manifest = build_manifest(output, source, dependencies, file_ref(staging / "validation.json"))
        manifest["validation"]["path"] = str(output / "validation.json")
        write_json(staging / "manifest.json", manifest, exclusive=True)
        os.replace(staging, output)
        verify_checkpoint_manifest(output / "manifest.json", source_root=source_root, verify_checkout=verify_checkout, expected_source=source)
        return pass_line
    except BaseException:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        if output.exists() and output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        raise


def fixture_child(root: Path, key: str, source: dict[str, Any]) -> None:
    spec = DEPENDENCY_SPECS[key]
    child_root = root / ("s2-product-contract" if key == "s2" else "")
    child_root.mkdir(parents=True, exist_ok=True)
    validation: dict[str, Any] = {"status": "pass", "source_git_sha": source["git_sha"], "source_tree_sha": source["git_tree_sha"]}
    child: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "canonical": True,
        "artifact_dir": str(child_root.resolve()),
        "source_git_sha": source["git_sha"],
        "source_tree_sha": source["git_tree_sha"],
        "dirty": False,
        "pass_line": f"{spec.child_pass_prefix}: {root.resolve()}",
    }
    delegated: list[str]
    if key == "source":
        child.update({"artifact_type": "runtime_vnext_g08a_source_ownership_manifest", "lane": "runtime-vnext-g08a-source-ownership", "summary": {"provider_file_count": 1, "provider_glue_production_loc": 499, "scaffolding_reduction_ratio": 1.0, "lifecycle_ownership_categories": 5, "lifecycle_implementation_owner_count": 1, "legacy_source_selection_count": 0}})
    elif key in {"cuda", "metal"}:
        backend = key
        requirement = MATRIX_REQUIREMENTS[backend]
        summary = {"scenario_count": 21, "case_count": requirement["case_count"], "passed_case_count": requirement["case_count"], "known_failed_count": 0, "blocked_count": 0, "error_count": 0, "unexpected_count": 0, "entrypoints": ["run", "serve"], "c18": {"requested_concurrency": requirement["concurrency"], "typed_admission_cap": requirement["active_floor"], "active_floor": requirement["active_floor"], "observed_max_active": requirement["active_floor"], "active_duty_cycle": 0.9}}
        report_path = root / "scenario-report.json"
        write_json(report_path, {"status": "pass", "source_git_sha": source["git_sha"], "source_tree_sha": source["git_tree_sha"], "model_key": MODEL_KEY, "backend": backend})
        validation.update({"model_key": MODEL_KEY, "backend": backend, "summary": summary})
        write_json(child_root / "validation.json", validation)
        child.update({"artifact_type": requirement["artifact_type"], "lane": requirement["child_lane"], "model_key": MODEL_KEY, "summary": summary, "scenario_report": file_ref(report_path), "validation": file_ref(child_root / "validation.json")})
    elif key == "numerics":
        op_root = root / "op-numerics"
        op_root.mkdir(exist_ok=True)
        numeric_paths = {
            "metal_op_numerics": op_root / "metal-op-numerics.json",
            "linear_attention": root / "linear-attention.json",
            "full_attention": root / "full-attention.json",
            "full_model": root / "full-model.json",
            "token_parity": root / "token-parity.json",
        }
        for ordinal, path in enumerate(numeric_paths.values()):
            write_json(path, {"ordinal": ordinal})
        inputs = {name: file_ref(path) for name, path in numeric_paths.items()}
        parity = {"case_count": 20, "token_count_per_case": 64, "matched_token_count": 1280, "exception_count": 0, "waiver_count": 0}
        validation.update({"catalog_row_count": 33, "artifact_local_tolerance_count": 0, "operation_state_row_count": 27, "layer_checkpoint_count": 2, "full_model_checkpoint_count": 1, "full_vocabulary_logits_checkpoint_count": 1, "token_parity": parity, "inputs": inputs})
        write_json(child_root / "validation.json", validation)
        child.update({"artifact_type": "runtime_vnext_g08a_numerics_manifest", "lane": "runtime-vnext-g08a-numerics", "validation": file_ref(child_root / "validation.json"), "summary": {"catalog_row_count": 33, "operation_state_row_count": 27, **parity}, "inputs": inputs})
    elif key == "s2":
        child.pop("source_git_sha")
        child.pop("source_tree_sha")
        child.pop("dirty")
        child.update({"artifact_type": "runtime_vnext_s2_cuda_product_contract_manifest", "lane": "runtime-vnext-s2", "checkpoint_id": "S2", "source": {"git_sha": source["git_sha"], "git_tree_sha": source["git_tree_sha"]}, "acceptance": {"all_required_children_present": True}, "children": {"historical_resource_source": {"deep_validation": {"case_count": 5, "source_test_count": 7}}}})
    else:
        backend = spec.backend
        assert backend is not None
        high = 32 if backend == "cuda" else 16
        ratios = [{"concurrency": value, "candidate_over_external": 0.75, "threshold": 0.70, "passes": True} for value in (1, high)]
        raw_root = root / "raw-performance"
        raw_root.mkdir(exist_ok=True)
        performance_summary = {"baseline_kind": "external", "ratios": ratios}
        validation.update({"model_key": MODEL_KEY, "backend": backend, "artifact_root": str(raw_root.resolve()), "summary": performance_summary})
        write_json(child_root / "validation.json", validation)
        child.update({"artifact_type": "runtime_vnext_g08_performance_smoke_manifest", "lane": "runtime-vnext-g08-performance-smoke", "model_key": MODEL_KEY, "backend": backend, "validation": file_ref(child_root / "validation.json"), "summary": performance_summary})
    if key == "source":
        write_json(child_root / "validation.json", validation)
        child["validation"] = file_ref(child_root / "validation.json")
    write_json(child_root / "manifest.json", child)
    delegated = [sys.executable, DEPENDENCY_SCRIPTS[key]]
    if key in {"cuda", "metal"}:
        delegated.extend(["--artifact-root", str(root.resolve()), "--scenario-report", str((root / "scenario-report.json").resolve())])
    elif key == "numerics":
        delegated.extend(
            [
                "--g08a-op-numerics",
                str((root / "op-numerics").resolve()),
                "--g08a-linear-attention",
                str((root / "linear-attention.json").resolve()),
                "--g08a-full-attention",
                str((root / "full-attention.json").resolve()),
                "--g08a-full-model",
                str((root / "full-model.json").resolve()),
                "--g08a-token-parity",
                str((root / "token-parity.json").resolve()),
            ]
        )
    elif key.endswith("performance"):
        delegated.extend(["--artifact-root", str((root / "raw-performance").resolve())])
    delegated.extend(["--out", str(root.resolve())])
    write_json(
        root / "run_gate.child.command.json",
        {"cmd": delegated, "duration_sec": 1.0, "env_overrides": {}},
    )
    (root / "run_gate.child.stdout").write_text(
        f"{spec.child_pass_prefix}: {root.resolve()}\n",
        encoding="utf-8",
    )
    (root / "run_gate.child.stderr").write_text("", encoding="utf-8")
    execution = []
    for name in ("run_gate.child.command.json", "run_gate.child.stdout", "run_gate.child.stderr"):
        ref = file_ref(root / name)
        ref["path"] = name
        execution.append(ref)
    outer = {
        "schema_version": 1,
        "lane": spec.lane,
        "status": "pass",
        "command_line": [sys.executable, "scripts/release/run_gate.py", spec.lane, "--out", str(root.resolve())],
        "delegated_command_line": delegated,
        "child_returncode": 0,
        "child_pass_line": f"{spec.child_pass_prefix}: {root.resolve()}",
        "child_artifacts": {"kind": spec.lane, "source": source, "child_manifest": file_ref(child_root / "manifest.json")},
        "child_execution_artifacts": execution,
        "git_sha": source["git_sha"],
        "dirty_status": {"is_dirty": False, "status_short": []},
        "artifact_dir": str(root.resolve()),
        "started_at": "2026-08-06T00:00:00+00:00",
        "finished_at": "2026-08-06T00:00:01+00:00",
        "duration_sec": 1.0,
        "binary": {"path": None, "sha256": None},
        "model": None,
        "sanitized_env": {},
        "pass_line": f"FERRUM GATE {spec.lane} PASS: {root.resolve()}",
        "error": None,
    }
    write_json(root / "gate.manifest.json", outer)


def rehash_fixture(root: Path, key: str) -> None:
    spec = DEPENDENCY_SPECS[key]
    child_root = root / ("s2-product-contract" if key == "s2" else "")
    child = read_json(child_root / "manifest.json", "fixture child")
    if (child_root / "validation.json").is_file():
        validation = read_json(child_root / "validation.json", "fixture validation")
        child["validation"] = file_ref(child_root / "validation.json")
        if "summary" in child and isinstance(validation.get("summary"), dict):
            child["summary"] = validation["summary"]
    write_json(child_root / "manifest.json", child)
    outer = read_json(root / "gate.manifest.json", "fixture outer")
    outer["child_artifacts"]["child_manifest"] = file_ref(child_root / "manifest.json")
    write_json(root / "gate.manifest.json", outer)


def self_test() -> int:
    source = {"git_sha": "1" * 40, "git_tree_sha": "2" * 40, "dirty": False}
    with tempfile.TemporaryDirectory(prefix="g08a-checkpoint-selftest-") as tmp:
        root = Path(tmp)
        paths: dict[str, Path] = {}
        for key in DEPENDENCY_SPECS:
            dependency_root = root / key
            dependency_root.mkdir()
            fixture_child(dependency_root, key, source)
            paths[key] = dependency_root / "gate.manifest.json"
        cuda_outer = read_json(paths["cuda"], "CUDA outer fixture")
        standalone = verify_dependency_manifest(
            "cuda",
            root / "cuda" / "manifest.json",
            cuda_outer["delegated_command_line"],
            verify_checkout=False,
            expected_source=source,
        )
        require(
            standalone.get("kind") == "vnext-g08a-cuda"
            and standalone.get("source") == source,
            "G08A standalone dependency provenance mismatch",
        )
        import run_gate

        run_gate_lane = run_gate.LaneCommand(
            cmd=cuda_outer["delegated_command_line"],
            expected_child_pass_line=cuda_outer["child_pass_line"],
            child_manifest_path=root / "cuda" / "manifest.json",
            provenance_kind="vnext-g08a-cuda",
        )
        run_gate_provenance = run_gate.verify_child_pass_line(
            run_gate_lane,
            (root / "cuda" / "run_gate.child.stdout").read_text(encoding="utf-8"),
            verify_checkout=False,
        )
        require(
            isinstance(run_gate_provenance, dict)
            and run_gate_provenance.get("kind") == "vnext-g08a-cuda"
            and run_gate_provenance.get("source") == source,
            "G08A run_gate dependency provenance mismatch",
        )
        output = root / "aggregate"
        line = build_checkpoint(paths, output, source_root=REPO_ROOT, verify_checkout=False, expected_source=source)
        require(line == f"{PASS_PREFIX}: {output.resolve()}", "G08A self-test PASS line mismatch")
        verify_checkpoint_manifest(output / "manifest.json", verify_checkout=False, expected_source=source)

        cuda_child = root / "cuda" / "manifest.json"
        cuda = read_json(cuda_child, "CUDA fixture")
        cuda["source_git_sha"] = "3" * 40
        write_json(cuda_child, cuda)
        rehash_fixture(root / "cuda", "cuda")
        try:
            validate_dependencies(paths, source, verify_checkout=False)
        except CheckpointError as error:
            require("stale" in str(error), f"stale source mutation failed unexpectedly: {error}")
        else:
            raise AssertionError("stale CUDA source mutation unexpectedly passed")
        fixture_child(root / "cuda", "cuda", source)

        perf_root = root / "metal_performance"
        validation_path = perf_root / "validation.json"
        validation = read_json(validation_path, "performance fixture")
        validation["summary"]["ratios"][0]["candidate_over_external"] = 0.69
        write_json(validation_path, validation)
        rehash_fixture(perf_root, "metal_performance")
        try:
            validate_dependencies(paths, source, verify_checkout=False)
        except CheckpointError as error:
            require("below threshold" in str(error), f"performance mutation failed unexpectedly: {error}")
        else:
            raise AssertionError("below-threshold performance mutation unexpectedly passed")

        fixture_child(perf_root, "metal_performance", source)
        validation_path = perf_root / "validation.json"
        validation = read_json(validation_path, "performance fixture")
        validation["summary"]["ratios"][0].update(
            {"candidate_over_external": 0.01, "threshold": 0.0}
        )
        write_json(validation_path, validation)
        rehash_fixture(perf_root, "metal_performance")
        try:
            validate_dependencies(paths, source, verify_checkout=False)
        except CheckpointError as error:
            require(
                "threshold contract" in str(error),
                f"threshold downgrade failed unexpectedly: {error}",
            )
        else:
            raise AssertionError("downgraded performance threshold unexpectedly passed")

        fixture_child(perf_root, "metal_performance", source)
        command_path = perf_root / "run_gate.child.command.json"
        command_receipt = read_json(command_path, "command fixture")
        command_receipt["cmd"].append("--forged")
        write_json(command_path, command_receipt)
        outer_path = perf_root / "gate.manifest.json"
        outer = read_json(outer_path, "outer fixture")
        for row in outer["child_execution_artifacts"]:
            if row["path"] == "run_gate.child.command.json":
                ref = file_ref(command_path)
                row.update({"sha256": ref["sha256"], "size_bytes": ref["size_bytes"]})
        write_json(outer_path, outer)
        try:
            validate_dependencies(paths, source, verify_checkout=False)
        except CheckpointError as error:
            require(
                "command receipt mismatch" in str(error),
                f"forged command receipt failed unexpectedly: {error}",
            )
        else:
            raise AssertionError("forged command receipt unexpectedly passed")

        fixture_child(perf_root, "metal_performance", source)
        outer = read_json(outer_path, "outer fixture")
        outer["child_artifacts"]["kind"] = "delegated-manifest"
        write_json(outer_path, outer)
        try:
            validate_dependencies(paths, source, verify_checkout=False)
        except CheckpointError as error:
            require(
                "provenance identity mismatch" in str(error),
                f"forged outer provenance failed unexpectedly: {error}",
            )
        else:
            raise AssertionError("forged outer provenance unexpectedly passed")
    print(SELFTEST_PASS_LINE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cuda", type=Path)
    parser.add_argument("--metal", type=Path)
    parser.add_argument("--numerics", type=Path)
    parser.add_argument("--s2", type=Path)
    parser.add_argument("--cuda-performance", type=Path)
    parser.add_argument("--metal-performance", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    values = {
        "source": args.source,
        "cuda": args.cuda,
        "metal": args.metal,
        "numerics": args.numerics,
        "s2": args.s2,
        "cuda_performance": args.cuda_performance,
        "metal_performance": args.metal_performance,
    }
    missing = [key for key, value in values.items() if value is None]
    if missing or args.out is None:
        parser.error("missing required G08A arguments: " + ", ".join(missing + (["out"] if args.out is None else [])))
    try:
        line = build_checkpoint(
            {key: value for key, value in values.items() if value is not None},
            args.out,
            source_root=args.source_root,
        )
    except (CheckpointError, OSError, ValueError) as error:
        print(f"FERRUM RUNTIME VNEXT G08A QWEN35 4B FAIL: {error}", file=os.sys.stderr)
        return 1
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
