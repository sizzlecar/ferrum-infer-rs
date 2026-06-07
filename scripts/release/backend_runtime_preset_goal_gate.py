#!/usr/bin/env python3
"""Final validator for the backend runtime preset fast-iteration goal."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_PREFIX = "BACKEND RUNTIME PRESET GOAL PASS"
REQUIRED_SNAPSHOT_CASES = {
    "metal_llama_8b_dense_gguf",
    "metal_qwen3_30b_a3b_moe_gguf",
    "cuda_llama_8b_dense_gptq",
    "cuda_qwen3_30b_a3b_moe_gptq",
}
REQUIRED_SNAPSHOT_GROUPS = {
    "attention_impls",
    "graph_modes",
    "kv_layouts",
    "kv_dtypes",
    "moe_decode_paths",
    "cache_modes",
}
ADMISSION_FIELDS = {
    "effective_max_concurrent",
    "queue_depth",
    "active_prefill",
    "active_decode",
    "current_batch_size",
    "rejected_requests_total",
    "failed_requests_total",
    "completed_requests_total",
}
RUN_GATE_MANIFEST_REQUIRED_FIELDS = {
    "schema_version",
    "lane",
    "status",
    "command_line",
    "delegated_command_line",
    "child_returncode",
    "child_pass_line",
    "git_sha",
    "dirty_status",
    "artifact_dir",
    "started_at",
    "finished_at",
    "duration_sec",
    "binary",
    "model",
    "sanitized_env",
    "pass_line",
    "error",
}
RUN_GATE_BINARY_LANES = {"metal", "cuda-smoke", "cuda-full", "cuda-llama-dense"}
REQUIRED_PRODUCT_SCENARIO_TYPES = {
    "run_multiturn",
    "run_first_token_ux",
    "serve_concurrency_quality",
    "serve_multiturn_recall",
    "serve_stateful_loop",
    "serve_stream",
    "serve_tool_call",
    "serve_structured_output",
    "serve_context_limit",
    "serve_python_openai_sdk",
}
REQUIRED_PRODUCT_CONCURRENCY_CELLS = {1, 4, 16, 32}


class ValidationError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValidationError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def resolve_path(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else root / path


def require_dir(path: Path, label: str) -> Path:
    if not path.exists():
        raise ValidationError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise ValidationError(f"{label} must be a directory: {path}")
    return path


def status_pass(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValidationError(f"{path} must contain a JSON object")
    if data.get("status") != "pass":
        raise ValidationError(f"{path} status is not pass: {data.get('status')!r}")
    return data


def validate_gate_manifest(path: Path, expected_lanes: set[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = status_pass(path)
    missing = sorted(RUN_GATE_MANIFEST_REQUIRED_FIELDS - set(data))
    if missing:
        raise ValidationError(f"{path} missing manifest fields: {', '.join(missing)}")
    if data.get("schema_version") != 1:
        raise ValidationError(f"{path} schema_version must be 1")
    lane = data.get("lane")
    if lane not in expected_lanes:
        raise ValidationError(
            f"{path} lane {lane!r} not in expected lanes {sorted(expected_lanes)}"
        )
    artifact_dir = data.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValidationError(f"{path} artifact_dir must be a non-empty string")
    pass_line = data.get("pass_line")
    expected_pass = f"FERRUM GATE {lane} PASS: {artifact_dir}"
    if pass_line != expected_pass:
        raise ValidationError(f"{path} missing unified FERRUM GATE pass_line")
    expected_child = expected_child_pass_line_for_manifest(str(lane), artifact_dir)
    child_pass_line = data.get("child_pass_line")
    if child_pass_line != expected_child:
        raise ValidationError(
            f"{path} child_pass_line {child_pass_line!r} != {expected_child!r}"
        )
    binary = data.get("binary")
    if not isinstance(binary, dict):
        raise ValidationError(f"{path} binary must be an object")
    if lane in RUN_GATE_BINARY_LANES:
        digest = binary.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValidationError(f"{path} missing binary sha256 for {lane}")
    for key in ["command_line", "delegated_command_line"]:
        if not isinstance(data.get(key), list) or not data[key]:
            raise ValidationError(f"{path} {key} must be a non-empty list")
    if not isinstance(data.get("dirty_status"), dict):
        raise ValidationError(f"{path} dirty_status must be an object")
    if not isinstance(data.get("sanitized_env"), dict):
        raise ValidationError(f"{path} sanitized_env must be an object")
    return {
        "kind": "run_gate_manifest",
        "path": str(path),
        "lane": lane,
        "child_pass_line": child_pass_line,
    }


def expected_child_pass_line_for_manifest(lane: str, artifact_dir: str) -> str:
    source_lanes = {
        "unit": "unit",
        "metal": "metal",
        "cuda-smoke": "g0_cuda4090_smoke",
        "cuda-full": "g0_cuda4090_full",
        "cuda-llama-dense": "g0_cuda4090_llama_dense",
    }
    if lane in source_lanes:
        return f"G0 SOURCE {source_lanes[lane]} PASS: {artifact_dir}"
    binary_lanes = {
        "metal-tarball": "METAL TARBALL GATE",
        "cuda-tarball": "CUDA TARBALL GATE",
        "homebrew-metal": "HOMEBREW METAL GATE",
        "homebrew-cuda-fetch": "HOMEBREW CUDA FETCH GATE",
    }
    if lane in binary_lanes:
        return f"{binary_lanes[lane]} PASS: {artifact_dir}"
    if lane == "release-complete":
        return f"FERRUM RELEASE COMPLETION PASS: {artifact_dir}"
    raise ValidationError(f"cannot infer child PASS line for run_gate lane {lane!r}")


def validate_source_gate(
    label: str,
    artifact: Path,
    *,
    expected_lanes: set[str],
    gate_files: list[str],
) -> dict[str, Any]:
    artifact = require_dir(artifact, f"{label} artifact")
    manifest_evidence = validate_gate_manifest(artifact / "gate.manifest.json", expected_lanes)
    if manifest_evidence is not None:
        return manifest_evidence

    errors: list[str] = []
    for rel in gate_files:
        path = artifact / rel
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        data = status_pass(path)
        return {
            "kind": "source_gate_json",
            "path": str(path),
            "lane": data.get("lane"),
        }
    raise ValidationError(f"{label}: no passing source gate artifact found: {'; '.join(errors)}")


def discover_unit_artifact(provided: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for root in provided:
        candidates.extend(
            [
                root,
                root.parent,
                root.parent / "unit",
                root.parent / "source-unit",
                root.parent / "source" / "unit",
            ]
        )
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "unit.gate.json").is_file() or (candidate / "gate.manifest.json").is_file():
            return candidate
    return None


def run_checked(
    label: str,
    cmd: list[str],
    *,
    cwd: Path,
    out_dir: Path,
    timeout: int | None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    duration = time.monotonic() - started
    (out_dir / f"{label}.stdout").write_text(proc.stdout, errors="replace")
    (out_dir / f"{label}.stderr").write_text(proc.stderr, errors="replace")
    command_path = out_dir / f"{label}.command.json"
    write_json(
        command_path,
        {
            "cmd": cmd,
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "duration_sec": duration,
        },
    )
    if proc.returncode != 0:
        raise ValidationError(
            f"{label} failed rc={proc.returncode}; see {out_dir / (label + '.stderr')}"
        )
    return {
        "kind": "command",
        "label": label,
        "cmd": cmd,
        "stdout": str(out_dir / f"{label}.stdout"),
        "stderr": str(out_dir / f"{label}.stderr"),
        "command": str(command_path),
        "duration_sec": duration,
    }


def validate_boundary_artifact(path: Path) -> dict[str, Any]:
    path = require_dir(path, "backend boundary artifact")
    result = status_pass(path / "backend_boundary_audit.json")
    if result.get("violation_count") != 0:
        raise ValidationError(f"backend boundary violations != 0 in {path}")
    return {
        "kind": "backend_boundary_audit",
        "path": str(path / "backend_boundary_audit.json"),
        "finding_count": result.get("finding_count"),
    }


def validate_snapshot_artifact(path: Path) -> dict[str, Any]:
    path = require_dir(path, "backend preset snapshot artifact")
    summary = status_pass(path / "summary.json")
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise ValidationError(f"{path / 'summary.json'} missing cases list")
    seen = set(str(case) for case in cases)
    if seen != REQUIRED_SNAPSHOT_CASES:
        raise ValidationError(
            f"backend preset snapshot cases mismatch missing="
            f"{sorted(REQUIRED_SNAPSHOT_CASES - seen)} extra={sorted(seen - REQUIRED_SNAPSHOT_CASES)}"
        )
    for case in sorted(REQUIRED_SNAPSHOT_CASES):
        snapshot = load_json(path / f"{case}.json")
        groups = snapshot.get("candidate_groups")
        if not isinstance(groups, list):
            raise ValidationError(f"{case}: candidate_groups must be a list")
        group_names = {group.get("name") for group in groups if isinstance(group, dict)}
        missing = sorted(REQUIRED_SNAPSHOT_GROUPS - group_names)
        if missing:
            raise ValidationError(f"{case}: missing candidate groups {missing}")
        if not isinstance(snapshot.get("model_capabilities"), dict):
            raise ValidationError(f"{case}: missing ModelCapabilities")
        for group in groups:
            if not isinstance(group, dict):
                raise ValidationError(f"{case}: candidate group must be object")
            selected = group.get("selected")
            candidates = group.get("candidates")
            rejected = group.get("rejected")
            if not isinstance(selected, str) or not selected:
                raise ValidationError(f"{case}.{group.get('name')}: selected missing")
            if not isinstance(candidates, list) or candidates.count(selected) != 1:
                raise ValidationError(
                    f"{case}.{group.get('name')}: candidates must contain selected once"
                )
            non_selected = [candidate for candidate in candidates if candidate != selected]
            if len(non_selected) != len(rejected or []):
                raise ValidationError(f"{case}.{group.get('name')}: rejected count mismatch")
            for item in rejected or []:
                if not isinstance(item, dict) or not str(item.get("reason", "")).strip():
                    raise ValidationError(f"{case}.{group.get('name')}: rejected reason missing")
    return {"kind": "backend_preset_snapshot", "path": str(path), "cases": sorted(seen)}


def validate_scenario_artifact(path: Path) -> dict[str, Any]:
    path = require_dir(path, "scenario artifact")
    summary = status_pass(path / "summary.json")
    if summary.get("failed") not in (0, None):
        raise ValidationError(f"scenario failed count is not zero in {path / 'summary.json'}")
    pass_line = summary.get("pass_line")
    if not isinstance(pass_line, str) or not pass_line.startswith("BACKEND REGRESSION SMOKE PASS:"):
        raise ValidationError(f"scenario summary missing BACKEND REGRESSION SMOKE pass_line")
    scenarios = summary.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValidationError("scenario summary must include at least one scenario result")
    failed = [item for item in scenarios if isinstance(item, dict) and item.get("status") == "fail"]
    if failed:
        raise ValidationError(f"scenario artifact contains failed scenarios: {failed}")
    return {
        "kind": "scenario_regression_smoke",
        "path": str(path / "summary.json"),
        "scenario_count": summary.get("scenario_count"),
    }


def validate_product_scenario_manifest(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValidationError(f"{path} must contain a JSON object")
    if data.get("schema_version") != 1:
        raise ValidationError(f"{path} schema_version must be 1")
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValidationError(f"{path} scenarios must be a non-empty list")
    by_type: dict[str, list[dict[str, Any]]] = {}
    for idx, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValidationError(f"{path} scenarios[{idx}] must be an object")
        name = scenario.get("name")
        typ = scenario.get("type")
        if not isinstance(name, str) or not name:
            raise ValidationError(f"{path} scenarios[{idx}].name must be non-empty")
        if not isinstance(typ, str) or not typ:
            raise ValidationError(f"{path} scenarios[{idx}].type must be non-empty")
        by_type.setdefault(typ, []).append(scenario)
    missing = sorted(REQUIRED_PRODUCT_SCENARIO_TYPES - set(by_type))
    if missing:
        raise ValidationError(f"{path} missing scenario types: {', '.join(missing)}")
    concurrency_cells: set[int] = set()
    for scenario in by_type["serve_concurrency_quality"]:
        cells = scenario.get("concurrency_cells")
        if not isinstance(cells, list):
            raise ValidationError(f"{path} serve_concurrency_quality missing concurrency_cells")
        for cell in cells:
            if isinstance(cell, bool) or not isinstance(cell, int):
                raise ValidationError(f"{path} invalid concurrency cell {cell!r}")
            concurrency_cells.add(cell)
    missing_cells = sorted(REQUIRED_PRODUCT_CONCURRENCY_CELLS - concurrency_cells)
    if missing_cells:
        raise ValidationError(f"{path} missing concurrency cells: {missing_cells}")
    run_multiturn = by_type["run_multiturn"][0]
    if int(run_multiturn.get("min_assistant_turns", 0)) < 2:
        raise ValidationError(f"{path} run_multiturn min_assistant_turns must be >= 2")
    stateful = by_type["serve_stateful_loop"][0]
    if int(stateful.get("min_non_empty_assistant", 0)) < 4:
        raise ValidationError(
            f"{path} serve_stateful_loop min_non_empty_assistant must be >= 4"
        )
    first_token = by_type["run_first_token_ux"][0]
    if int(first_token.get("hint_timeout_ms", 0)) > 1000:
        raise ValidationError(f"{path} first-token hint_timeout_ms must be <= 1000")
    if int(first_token.get("max_hint_len", 10**9)) > 32:
        raise ValidationError(f"{path} first-token max_hint_len must be <= 32")
    sdk = by_type["serve_python_openai_sdk"][0]
    if sdk.get("optional") is not True:
        raise ValidationError(f"{path} serve_python_openai_sdk must be optional")
    return {
        "kind": "product_scenario_manifest_contract",
        "path": str(path),
        "scenario_types": sorted(by_type),
        "concurrency_cells": sorted(concurrency_cells),
    }


def validate_effective_config_schema(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if data.get("schema_version") != 1:
        raise ValidationError(f"{path}: schema_version must be 1")
    for field in [
        "entries",
        "model_capabilities",
        "hardware_capabilities",
        "workload_profile",
        "admission",
        "decisions",
    ]:
        if field not in data:
            raise ValidationError(f"{path}: missing {field}")
    admission = data["admission"]
    if not isinstance(admission, dict):
        raise ValidationError(f"{path}: admission must be object")
    missing_admission = sorted(field for field in ADMISSION_FIELDS if field not in admission)
    if missing_admission:
        raise ValidationError(f"{path}: admission missing {missing_admission}")
    decisions = data["decisions"]
    if not isinstance(decisions, list) or not decisions:
        raise ValidationError(f"{path}: decisions must be a non-empty list")
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValidationError(f"{path}: decision must be object")
        for key in ["selection", "selected", "candidates", "rejected"]:
            if key not in decision:
                raise ValidationError(f"{path}: decision missing {key}")
    return {"kind": "effective_config_schema", "path": str(path)}


def validate_runtime_schema_from_snapshots(snapshot_artifact: Path) -> dict[str, Any]:
    evidence = []
    for case in sorted(REQUIRED_SNAPSHOT_CASES):
        evidence.append(validate_effective_config_schema(snapshot_artifact / f"{case}.json"))
    return {"kind": "runtime_preset_schema", "snapshots": evidence}


def validate_metal_artifact(root: Path, repo: Path, out_dir: Path, timeout: int | None) -> dict[str, Any]:
    evidence = {
        "source_gate": validate_source_gate(
            "metal",
            root,
            expected_lanes={"metal"},
            gate_files=["metal.gate.json", "source-metal/metal.gate.json"],
        )
    }
    metal_readme = root / "metal-readme"
    if not metal_readme.is_dir():
        raise ValidationError(f"metal artifact missing metal-readme directory: {metal_readme}")
    evidence["metal_readme_validator"] = run_checked(
        "validate-metal-readme",
        [sys.executable, "scripts/release/validate_metal_readme_regression.py", str(metal_readme)],
        cwd=repo,
        out_dir=out_dir,
        timeout=timeout,
    )
    summary = load_json(metal_readme / "summary.json")
    model_keys = {str(model.get("key", "")).lower() for model in summary.get("models", [])}
    if not any("llama" in key for key in model_keys):
        raise ValidationError("metal artifact missing Llama 8B-class evidence")
    if not any("qwen3_30b" in key or "qwen3-30b" in key for key in model_keys):
        raise ValidationError("metal artifact missing Qwen3-30B-A3B evidence")
    evidence["model_keys"] = sorted(model_keys)
    return evidence


def validate_cuda_source_artifact(label: str, root: Path) -> dict[str, Any]:
    gate_files = {
        "cuda-smoke": ["g0_cuda4090_smoke.gate.json", "cuda-smoke/g0_cuda4090_smoke.gate.json"],
        "cuda-full": ["g0_cuda4090_full.gate.json", "cuda-full/g0_cuda4090_full.gate.json"],
        "cuda-llama-dense": [
            "g0_cuda4090_llama_dense.gate.json",
            "cuda-llama-dense/g0_cuda4090_llama_dense.gate.json",
        ],
    }[label]
    evidence = validate_source_gate(
        label,
        root,
        expected_lanes={label},
        gate_files=gate_files,
    )
    if label == "cuda-llama-dense":
        dense_gate = root / "gate.json"
        if dense_gate.is_file():
            data = status_pass(dense_gate)
            checks = data.get("checks")
            if not isinstance(checks, dict):
                raise ValidationError(f"{dense_gate}: missing checks")
            for check in ["run", "serve", "serve_health", "bench_serve"]:
                if check not in checks:
                    raise ValidationError(f"{dense_gate}: missing {check} check")
    return evidence


def validate_completion_artifact(path: Path) -> dict[str, Any]:
    path = require_dir(path, "release completion artifact")
    data = status_pass(path / "release_completion_gate.json")
    return {"kind": "release_completion", "path": str(path), "tag": data.get("tag")}


def validate_goal(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = iso_now()
    started = time.monotonic()
    timeout = args.timeout

    metal = resolve_path(repo, args.metal_artifact)
    cuda_smoke = resolve_path(repo, args.cuda_smoke_artifact)
    cuda_full = resolve_path(repo, args.cuda_full_artifact)
    cuda_dense = resolve_path(repo, args.cuda_dense_artifact)
    assert metal is not None and cuda_smoke is not None and cuda_full is not None and cuda_dense is not None
    provided = [metal, cuda_smoke, cuda_full, cuda_dense]

    evidence: dict[str, Any] = {}
    unit = resolve_path(repo, args.unit_artifact) or discover_unit_artifact(provided)
    if unit is None:
        unit = out_dir / "unit"
        evidence["unit"] = run_checked(
            "run-gate-unit",
            [sys.executable, "scripts/release/run_gate.py", "unit", "--out", str(unit)],
            cwd=repo,
            out_dir=out_dir / "commands",
            timeout=timeout,
        )
    evidence["unit_gate"] = validate_source_gate(
        "unit",
        unit,
        expected_lanes={"unit"},
        gate_files=["unit.gate.json", "source-unit/unit.gate.json"],
    )

    evidence["run_gate_selftest"] = run_checked(
        "run-gate-selftest",
        [sys.executable, "scripts/release/run_gate.py", "--self-test"],
        cwd=repo,
        out_dir=out_dir / "commands",
        timeout=timeout,
    )

    scenario = resolve_path(repo, args.scenario_artifact)
    if scenario is None:
        scenario = out_dir / "scenario-smoke"
        evidence["scenario_command"] = run_checked(
            "run-scenario-smoke",
            [
                sys.executable,
                "scripts/release/run_scenarios.py",
                "--manifest",
                "scripts/release/scenarios/product_regression_smoke.json",
                "--out",
                str(scenario),
            ],
            cwd=repo,
            out_dir=out_dir / "commands",
            timeout=timeout,
        )
    evidence["scenario_smoke"] = validate_scenario_artifact(scenario)
    evidence["scenario_manifest_contract"] = validate_product_scenario_manifest(
        repo / "scripts/release/scenarios/product_regression.json"
    )

    boundary = resolve_path(repo, args.backend_boundary_artifact)
    if boundary is None:
        boundary = out_dir / "backend-boundary"
        evidence["backend_boundary_command"] = run_checked(
            "backend-boundary-audit",
            [
                sys.executable,
                "scripts/release/backend_boundary_audit.py",
                "--out",
                str(boundary),
            ],
            cwd=repo,
            out_dir=out_dir / "commands",
            timeout=timeout,
        )
    evidence["backend_boundary"] = validate_boundary_artifact(boundary)

    snapshot = resolve_path(repo, args.preset_snapshot_artifact)
    if snapshot is None:
        snapshot = out_dir / "preset-snapshots"
        evidence["preset_snapshot_command"] = run_checked(
            "backend-runtime-preset-snapshot",
            [
                sys.executable,
                "scripts/release/backend_runtime_preset_snapshot.py",
                "--out",
                str(snapshot),
            ],
            cwd=repo,
            out_dir=out_dir / "commands",
            timeout=timeout,
        )
    evidence["preset_snapshot"] = validate_snapshot_artifact(snapshot)
    evidence["runtime_preset_schema"] = validate_runtime_schema_from_snapshots(snapshot)

    evidence["metal"] = validate_metal_artifact(metal, repo, out_dir / "commands", timeout)
    evidence["cuda_smoke"] = validate_cuda_source_artifact("cuda-smoke", cuda_smoke)
    evidence["cuda_full"] = validate_cuda_source_artifact("cuda-full", cuda_full)
    evidence["cuda_dense"] = validate_cuda_source_artifact("cuda-llama-dense", cuda_dense)

    completion_artifact = resolve_path(repo, args.release_completion_artifact)
    if completion_artifact is not None:
        evidence["release_completion"] = validate_completion_artifact(completion_artifact)
    completion_manifest = resolve_path(repo, args.completion_manifest)
    if completion_manifest is not None:
        completion_out = out_dir / "release-complete"
        evidence["release_completion_command"] = run_checked(
            "release-completion-validator",
            [
                sys.executable,
                "scripts/release/validate_release_completion_manifest.py",
                "--manifest",
                str(completion_manifest),
                "--out",
                str(completion_out),
            ],
            cwd=repo,
            out_dir=out_dir / "commands",
            timeout=timeout,
        )
        evidence["release_completion"] = validate_completion_artifact(completion_out)

    return {
        "schema_version": 1,
        "status": "pass",
        "started_at": started_at,
        "finished_at": iso_now(),
        "duration_sec": time.monotonic() - started,
        "artifact_dir": str(out_dir),
        "inputs": {
            "unit_artifact": str(unit),
            "metal_artifact": str(metal),
            "cuda_smoke_artifact": str(cuda_smoke),
            "cuda_full_artifact": str(cuda_full),
            "cuda_dense_artifact": str(cuda_dense),
            "scenario_artifact": str(scenario),
            "backend_boundary_artifact": str(boundary),
            "preset_snapshot_artifact": str(snapshot),
        },
        "evidence": evidence,
        "pass_line": f"{PASS_PREFIX}: {out_dir}",
    }


def make_gate(root: Path, rel: str, lane: str = "selftest") -> None:
    write_json(root / rel, {"status": "pass", "lane": lane})


def make_run_gate_manifest(root: Path, lane: str, child_pass_line: str, *, binary: bool) -> None:
    write_json(
        root / "gate.manifest.json",
        {
            "schema_version": 1,
            "lane": lane,
            "status": "pass",
            "command_line": ["python3", "scripts/release/run_gate.py", lane],
            "delegated_command_line": ["delegated", lane],
            "child_returncode": 0,
            "child_pass_line": child_pass_line,
            "git_sha": "selftest",
            "dirty_status": {"is_dirty": False, "status_short": []},
            "artifact_dir": str(root),
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:01+00:00",
            "duration_sec": 1.0,
            "binary": {
                "path": "target/release/ferrum" if binary else None,
                "sha256": "1" * 64 if binary else None,
            },
            "model": "selftest-model" if binary else None,
            "sanitized_env": {},
            "pass_line": f"FERRUM GATE {lane} PASS: {root}",
            "error": None,
        },
    )


def metal_model(key: str) -> dict[str, Any]:
    return {
        "key": key,
        "default_startup": {
            "passed": True,
            "max_sequences": 16,
            "min_required_max_sequences": 1,
            "max_allowed_max_sequences": 32,
        },
        "server_ready": True,
        "serve_startup": {"passed": True, "max_sequences": 16},
        "chat": {
            "paris": {"passed": True},
            "multiturn": {"passed": True},
            "stream": {"passed": True},
            "stateful_loop": {
                "passed": True,
                "length_finishes": 0,
                "repeated_prefixes": 0,
            },
        },
        "tool_call": {
            "status": "pass",
            "checks": {
                "omitted_tool_choice": {"passed": True},
                "explicit_auto_tool_choice": {"passed": True},
                "required_tool_choice": {"passed": True},
                "tool_result_fill": {"passed": True},
            },
        },
        "run": {"passed": True},
        "moe": "qwen3_30b" in key,
        "cells": [
            {
                "concurrency": 16,
                "prompts": 1,
                "completed": 1,
                "failed": 0,
                "output_throughput_tok_s": 1.0,
                "ratio_to_readme": 1.0,
                "not_regressed_90pct": True,
                "quality": {
                    "passed": True,
                    "requests": 1,
                    "status_200": 1,
                    "marker_ok": 1,
                    "square_ok": 1,
                    "crosstalk": 0,
                    "length_finishes": 0,
                },
            }
        ],
    }


def make_selftest_artifacts(root: Path) -> dict[str, Path]:
    unit = root / "unit"
    make_run_gate_manifest(
        unit,
        "unit",
        f"G0 SOURCE unit PASS: {unit}",
        binary=False,
    )

    metal = root / "metal"
    make_run_gate_manifest(
        metal,
        "metal",
        f"G0 SOURCE metal PASS: {metal}",
        binary=True,
    )
    metal_readme = metal / "metal-readme"
    write_json(
        metal_readme / "summary.json",
        {"models": [metal_model("llama31_8b"), metal_model("qwen3_30b_a3b")]},
    )
    for key in ["llama31_8b", "qwen3_30b_a3b"]:
        (metal_readme / f"{key}.server.stdout").write_text("ok\n")
        (metal_readme / f"{key}.run.stdout").write_text("ok\n")

    cuda_smoke = root / "cuda-smoke"
    make_run_gate_manifest(
        cuda_smoke,
        "cuda-smoke",
        f"G0 SOURCE g0_cuda4090_smoke PASS: {cuda_smoke}",
        binary=True,
    )
    cuda_full = root / "cuda-full"
    make_run_gate_manifest(
        cuda_full,
        "cuda-full",
        f"G0 SOURCE g0_cuda4090_full PASS: {cuda_full}",
        binary=True,
    )
    cuda_dense = root / "cuda-dense"
    make_run_gate_manifest(
        cuda_dense,
        "cuda-llama-dense",
        f"G0 SOURCE g0_cuda4090_llama_dense PASS: {cuda_dense}",
        binary=True,
    )
    write_json(
        cuda_dense / "gate.json",
        {
            "status": "pass",
            "lane": "g0_cuda4090_llama_dense",
            "checks": {
                "run": {"passed": True},
                "serve": {"math": {"passed": True}},
                "serve_health": {"passed": True},
                "bench_serve": {"passed": True},
            },
        },
    )

    scenario = root / "scenario"
    write_json(
        scenario / "summary.json",
        {
            "schema_version": 1,
            "status": "pass",
            "failed": 0,
            "scenario_count": 2,
            "scenarios": [
                {"name": "run", "type": "run_multiturn", "status": "pass"},
                {"name": "serve", "type": "serve_stream", "status": "pass"},
            ],
            "pass_line": f"BACKEND REGRESSION SMOKE PASS: {scenario}",
        },
    )

    boundary = root / "boundary"
    write_json(
        boundary / "backend_boundary_audit.json",
        {
            "schema_version": 1,
            "status": "pass",
            "finding_count": 0,
            "violation_count": 0,
            "findings": [],
            "violations": [],
        },
    )

    snapshot = root / "snapshot"
    write_json(
        snapshot / "summary.json",
        {
            "schema_version": 1,
            "status": "pass",
            "cases": sorted(REQUIRED_SNAPSHOT_CASES),
        },
    )
    for case in REQUIRED_SNAPSHOT_CASES:
        write_json(
            snapshot / f"{case}.json",
            {
                "schema_version": 1,
                "entries": [],
                "model_capabilities": {"architecture": "selftest"},
                "hardware_capabilities": {"backend": "selftest"},
                "workload_profile": {"target_concurrency": 1},
                "admission": {field: 0 for field in ADMISSION_FIELDS},
                "decisions": [
                    {
                        "selection": "attention_decode_backend",
                        "selected": "selftest",
                        "candidates": ["selftest"],
                        "rejected": [],
                    }
                ],
                "candidate_groups": [
                    {
                        "name": group,
                        "selected": "selected",
                        "candidates": ["selected", "rejected"],
                        "rejected": [{"value": "rejected", "reason": "selftest"}],
                    }
                    for group in sorted(REQUIRED_SNAPSHOT_GROUPS)
                ],
            },
        )
    return {
        "unit": unit,
        "metal": metal,
        "cuda_smoke": cuda_smoke,
        "cuda_full": cuda_full,
        "cuda_dense": cuda_dense,
        "scenario": scenario,
        "boundary": boundary,
        "snapshot": snapshot,
    }


def self_test() -> int:
    import tempfile

    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="ferrum-backend-runtime-goal-") as tmp:
        root = Path(tmp)
        artifacts = make_selftest_artifacts(root / "artifacts")
        validate_gate_manifest(artifacts["metal"] / "gate.manifest.json", {"metal"})
        bad_manifest = load_json(artifacts["metal"] / "gate.manifest.json")
        bad_manifest["child_pass_line"] = "wrong pass line"
        bad_manifest_path = root / "bad-metal-manifest" / "gate.manifest.json"
        write_json(bad_manifest_path, bad_manifest)
        try:
            validate_gate_manifest(bad_manifest_path, {"metal"})
            raise AssertionError("bad child PASS line unexpectedly passed")
        except ValidationError as exc:
            if "child_pass_line" not in str(exc):
                raise
        bad_scenarios = root / "bad-product-regression.json"
        write_json(
            bad_scenarios,
            {
                "schema_version": 1,
                "scenarios": [
                    {
                        "name": "concurrency",
                        "type": "serve_concurrency_quality",
                        "concurrency_cells": [1, 4],
                    }
                ],
            },
        )
        try:
            validate_product_scenario_manifest(bad_scenarios)
            raise AssertionError("incomplete scenario manifest unexpectedly passed")
        except ValidationError as exc:
            if "missing scenario types" not in str(exc):
                raise
        out = root / "out"
        args = argparse.Namespace(
            root=repo,
            out=out,
            unit_artifact=artifacts["unit"],
            metal_artifact=artifacts["metal"],
            cuda_smoke_artifact=artifacts["cuda_smoke"],
            cuda_full_artifact=artifacts["cuda_full"],
            cuda_dense_artifact=artifacts["cuda_dense"],
            scenario_artifact=artifacts["scenario"],
            backend_boundary_artifact=artifacts["boundary"],
            preset_snapshot_artifact=artifacts["snapshot"],
            release_completion_artifact=None,
            completion_manifest=None,
            timeout=120,
        )
        result = validate_goal(args)
        write_json(out / "backend_runtime_preset_goal_gate.json", result)
        if result.get("status") != "pass":
            raise AssertionError(result)
    print("BACKEND RUNTIME PRESET GOAL SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--unit-artifact", type=Path)
    parser.add_argument("--metal-artifact", type=Path)
    parser.add_argument("--cuda-smoke-artifact", type=Path)
    parser.add_argument("--cuda-full-artifact", type=Path)
    parser.add_argument("--cuda-dense-artifact", type=Path)
    parser.add_argument("--scenario-artifact", type=Path)
    parser.add_argument("--backend-boundary-artifact", type=Path)
    parser.add_argument("--preset-snapshot-artifact", type=Path)
    parser.add_argument("--release-completion-artifact", type=Path)
    parser.add_argument("--completion-manifest", type=Path)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.out is None:
        parser.error("--out is required")
    for name in ["metal_artifact", "cuda_smoke_artifact", "cuda_full_artifact", "cuda_dense_artifact"]:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")

    out_dir = args.out.resolve()
    try:
        result = validate_goal(args)
        write_json(out_dir / "backend_runtime_preset_goal_gate.json", result)
    except (ValidationError, subprocess.TimeoutExpired) as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            out_dir / "backend_runtime_preset_goal_gate.json",
            {
                "schema_version": 1,
                "status": "fail",
                "error": str(exc),
                "artifact_dir": str(out_dir),
            },
        )
        print(f"BACKEND RUNTIME PRESET GOAL FAIL: {out_dir}: {exc}", file=sys.stderr)
        return 1

    print(result["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
