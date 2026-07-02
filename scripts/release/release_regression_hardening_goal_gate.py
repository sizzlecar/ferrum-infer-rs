#!/usr/bin/env python3
"""Final aggregator for the release regression hardening goal.

This gate does not create product evidence. It proves that the already-produced
WP artifacts are current, pass their own gates, and cover the final hardening
requirements before printing the only goal-completion PASS line.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL = "release-regression-hardening-2026-06-28"
PASS_LINE = "RELEASE_REGRESSION_HARDENING GOAL PASS"
SELFTEST_PASS_LINE = "RELEASE_REGRESSION_HARDENING GOAL SELFTEST PASS"
SCHEMA_VERSION = 1
GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTH", "CREDENTIAL", "KEY")
SAFE_ENV_NAMES = {
    "CI",
    "CARGO_HOME",
    "HF_HOME",
    "HOME",
    "PATH",
    "RUSTFLAGS",
    "RUST_BACKTRACE",
    "RUST_LOG",
    "SHELL",
    "USER",
}
SAFE_ENV_PREFIXES = ("CARGO_", "FERRUM_", "HF_", "RUST_", "VAST_")
OBSERVABILITY_SUMMARY_FIELDS = {
    "request_count",
    "failed_count",
    "corrupted_count",
    "bad_text_count",
    "oom_prevented_count",
    "silent_oom_count",
    "latency_p50_p95_p99",
    "memory_high_water_bytes",
    "resource_leak_count",
    "top_slow_phases",
    "first_failure_event",
    "replay_commands",
}
REQUIRED_L2_ENTRYPOINTS = {"run", "serve", "stream", "basic_concurrency"}
ALLOWED_L2_ARCHITECTURES_BY_BACKEND = {
    "cuda": {"llama_dense", "qwen3_moe"},
    "metal": {"llama_dense", "qwen3", "qwen3_moe"},
}
REQUIRED_RESOURCE_SCENARIOS = {
    "kv_allocate_release_success",
    "kv_capacity_reject",
    "recurrent_slot_limit_reject",
    "scheduler_defer_reopen_after_capacity",
    "scheduler_cancel_releases_capacity",
    "engine_prefill_failure_rolls_back",
    "engine_decode_failure_rolls_back",
    "serve_client_disconnect_cleans_up",
    "run_multiturn_resource_balance",
    "mixed_batch_partial_failure",
    "oom_prevented_by_admission",
    "trace_replay_selftest",
}
REQUIRED_RESOURCE_KINDS = {
    "backend_workspace",
    "kv_block",
    "recurrent_state_slot",
    "scheduler_admission_slot",
}
RESOURCE_CACHE_KINDS = {"model_cache_ref", "session_cache_ref"}
REQUIRED_NATIVE_PASS_FIXTURES = {"dummy_manifest.json", "fa2_manifest.json"}
REQUIRED_NATIVE_RESOLVER_FAIL_CLOSED_CASES = {
    "abi_mismatch",
    "binary_sha256_mismatch",
    "compute_capability_mismatch",
    "missing_manifest",
    "operator_mismatch",
}
REQUIRED_PRODUCT_SENTINEL_SCENARIO_TYPES = {
    "native_op_manifest",
    "profile_artifact",
    "profile_replay_link",
    "replay_bundle",
    "resource_trace",
    "sse_fixture",
}
REQUIRED_RELEASE_CANDIDATE_STAGE_GATES = {
    "actual_model_regression",
    "model_contract",
    "native_operator",
    "observability_profile",
    "product_sentinel",
    "resource_invariant",
    "support_matrix_contract",
}


class GoalGateError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoalGateError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GoalGateError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GoalGateError(f"{path}: expected JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def head_sha() -> str:
    value = git_value(["rev-parse", "HEAD"])
    if not GIT_SHA_RE.match(value):
        raise GoalGateError(f"current HEAD is not a git SHA: {value!r}")
    return value


def current_dirty_files() -> list[str]:
    return git_value(["status", "--short"], default="").splitlines()


def resolve_path(raw: str, *, base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = (base / path).resolve()
    if candidate.exists():
        return candidate
    return (REPO_ROOT / path).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GoalGateError(message)


def require_status_pass(data: dict[str, Any], label: str) -> None:
    require(data.get("status") == "pass", f"{label}.status must be pass")


def require_pass_line(data: dict[str, Any], label: str, prefix: str) -> str:
    value = data.get("pass_line")
    require(
        isinstance(value, str) and value.startswith(f"{prefix}:"),
        f"{label}.pass_line must start with {prefix}:",
    )
    pass_prefix = value.split(":", 1)[0].upper()
    require(
        "SELFTEST" not in pass_prefix and "SELF-TEST" not in pass_prefix,
        f"{label}.pass_line must not be selftest evidence",
    )
    return value


def require_git_sha(data: dict[str, Any], label: str, expected_sha: str) -> str:
    value = data.get("git_sha")
    require(isinstance(value, str) and GIT_SHA_RE.match(value), f"{label}.git_sha must be a 40-character SHA")
    require(value.lower() == expected_sha.lower(), f"{label}.git_sha {value} is stale vs HEAD {expected_sha}")
    return value


def require_clean_manifest(data: dict[str, Any], label: str) -> None:
    require(
        isinstance(data.get("git_dirty"), bool),
        f"{label}.git_dirty must be boolean",
    )
    dirty_files = require_string_list(data.get("dirty_files", []), f"{label}.dirty_files")
    require(data["git_dirty"] is False, f"{label}.git_dirty must be false for final PASS")
    require(not dirty_files, f"{label}.dirty_files must be empty for final PASS")


def require_real_pass_line(value: Any, label: str) -> str:
    require(isinstance(value, str) and " PASS:" in value, f"{label}.pass_line must be a gate PASS line")
    prefix = value.split(":", 1)[0].upper()
    require("SELFTEST" not in prefix and "SELF-TEST" not in prefix, f"{label}.pass_line must not be selftest evidence")
    return value


def require_string_list(value: Any, label: str) -> list[str]:
    require(isinstance(value, list), f"{label} must be a list")
    require(all(isinstance(item, str) for item in value), f"{label} entries must be strings")
    return list(value)


def normalize_command(value: Any, label: str) -> list[str]:
    if isinstance(value, str):
        require(value.strip(), f"{label} must be non-empty")
        return [value]
    if isinstance(value, list):
        require(value, f"{label} must be non-empty")
        require(all(isinstance(item, str) and item.strip() for item in value), f"{label} entries must be non-empty strings")
        return list(value)
    raise GoalGateError(f"{label} must be a non-empty string or string array")


def load_stage_manifest(root: Path, label: str, pass_prefix: str, expected_sha: str) -> dict[str, Any]:
    manifest = read_json(root / "gate.manifest.json")
    require_status_pass(manifest, f"{label}.manifest")
    pass_line = require_pass_line(manifest, f"{label}.manifest", pass_prefix)
    require_git_sha(manifest, f"{label}.manifest", expected_sha)
    require_clean_manifest(manifest, f"{label}.manifest")
    return {
        "label": label,
        "artifact_dir": str(root),
        "manifest_path": str(root / "gate.manifest.json"),
        "pass_line": pass_line,
        "git_sha": manifest.get("git_sha"),
        "manifest": manifest,
    }


def load_summary_from_manifest(stage: dict[str, Any], fallback_name: str) -> dict[str, Any]:
    root = Path(stage["artifact_dir"])
    manifest = stage["manifest"]
    raw = None
    outputs = manifest.get("outputs")
    if isinstance(outputs, dict):
        raw = outputs.get("summary")
    if raw is None:
        raw = manifest.get("summary")
    path = resolve_path(str(raw), base=root) if isinstance(raw, str) and raw.strip() else root / fallback_name
    summary = read_json(path)
    require_status_pass(summary, f"{stage['label']}.summary")
    return {**summary, "_summary_path": str(path)}


def validate_resource_fixture_coverage(summary: dict[str, Any]) -> None:
    fixture_summary = summary.get("fixture_summary")
    require(
        isinstance(fixture_summary, dict),
        "resource_invariant.summary.fixture_summary must be an object",
    )
    scenario_count = fixture_summary.get("scenario_count")
    require(
        isinstance(scenario_count, int) and scenario_count >= len(REQUIRED_RESOURCE_SCENARIOS),
        f"resource_invariant.summary.fixture_summary.scenario_count must be >= {len(REQUIRED_RESOURCE_SCENARIOS)}",
    )
    required_scenarios = set(
        require_string_list(
            fixture_summary.get("required_scenarios", []),
            "resource_invariant.summary.fixture_summary.required_scenarios",
        )
    )
    missing = sorted(REQUIRED_RESOURCE_SCENARIOS - required_scenarios)
    require(not missing, f"resource_invariant.summary.fixture_summary.required_scenarios missing {missing}")
    trace = summary.get("trace")
    require(isinstance(trace, dict), "resource_invariant.summary.trace must be an object")
    trace_scenarios = set(
        require_string_list(
            trace.get("scenarios", []),
            "resource_invariant.summary.trace.scenarios",
        )
    )
    missing_trace = sorted(REQUIRED_RESOURCE_SCENARIOS - trace_scenarios)
    require(not missing_trace, f"resource_invariant.summary.trace.scenarios missing {missing_trace}")


def validate_resource_summary(summary: dict[str, Any]) -> None:
    resource_summary = summary.get("resource_summary")
    require(
        isinstance(resource_summary, dict) and resource_summary,
        "resource_invariant.summary.resource_summary must be a non-empty object",
    )
    resource_kinds = set(resource_summary)
    missing_kinds = sorted(REQUIRED_RESOURCE_KINDS - resource_kinds)
    require(not missing_kinds, f"resource_invariant.summary.resource_summary missing resource kinds {missing_kinds}")
    require(
        resource_kinds & RESOURCE_CACHE_KINDS,
        "resource_invariant.summary.resource_summary must include model_cache_ref or session_cache_ref",
    )
    for resource_kind, bucket in sorted(resource_summary.items()):
        require(
            isinstance(bucket, dict),
            f"resource_invariant.summary.resource_summary.{resource_kind} must be an object",
        )
        for key in ("capacity", "reserved", "committed", "released", "leaked"):
            value = bucket.get(key)
            require(
                isinstance(value, int) and value >= 0,
                f"resource_invariant.summary.resource_summary.{resource_kind}.{key} must be a non-negative integer",
            )
        require(
            bucket["leaked"] == 0,
            f"resource_invariant.summary.resource_summary.{resource_kind}.leaked must be 0",
        )


def validate_resource_invariant(root: Path, expected_sha: str) -> dict[str, Any]:
    stage = load_stage_manifest(root, "resource_invariant", "RESOURCE INVARIANT GATE PASS", expected_sha)
    summary = read_json(root / "invariant_report.json")
    require_status_pass(summary, "resource_invariant.summary")
    for key in ("leaked_resources", "underflow_count", "silent_oom_count", "panic_count"):
        require(summary.get(key) == 0, f"resource_invariant.summary.{key} must be 0")
    validate_resource_fixture_coverage(summary)
    validate_resource_summary(summary)
    stage["summary"] = {**summary, "_summary_path": str(root / "invariant_report.json")}
    return stage


def validate_release_candidate_invalidation(
    gate_plan: dict[str, Any],
    release_candidate: dict[str, Any],
) -> None:
    impact_domains = require_string_list(gate_plan.get("impact_domains", []), "change_impact.gate_plan.impact_domains")
    require("unknown" not in impact_domains, "change_impact.gate_plan.impact_domains must not contain unknown")
    invalidated_gates = set(
        require_string_list(
            release_candidate.get("invalidated_gates", []),
            "release_candidate_manifest.invalidated_gates",
        )
    )
    satisfied_gates = set(
        require_string_list(
            release_candidate.get("satisfied_gates", []),
            "release_candidate_manifest.satisfied_gates",
        )
    )
    overlap = sorted(invalidated_gates & satisfied_gates)
    require(not overlap, f"release_candidate_manifest counts invalidated gates as satisfied: {overlap}")

    artifact_paths = set(
        require_string_list(
            release_candidate.get("artifact_paths", []),
            "release_candidate_manifest.artifact_paths",
        )
    )
    pass_lines = set(
        require_string_list(
            release_candidate.get("pass_lines", []),
            "release_candidate_manifest.pass_lines",
        )
    )
    stale_artifacts = release_candidate.get("stale_artifacts", [])
    require(isinstance(stale_artifacts, list), "release_candidate_manifest.stale_artifacts must be a list")
    counted_stale: list[str] = []
    for index, artifact in enumerate(stale_artifacts):
        require(isinstance(artifact, dict), f"release_candidate_manifest.stale_artifacts[{index}] must be an object")
        artifact_id = str(artifact.get("id") or artifact.get("gate") or artifact.get("artifact_dir") or f"stale-{index}")
        if artifact_id in satisfied_gates or f"artifact:{artifact_id}" in satisfied_gates:
            counted_stale.append(artifact_id)
        artifact_dir = artifact.get("artifact_dir")
        if isinstance(artifact_dir, str) and artifact_dir in artifact_paths:
            counted_stale.append(artifact_id)
        pass_line = artifact.get("pass_line")
        if isinstance(pass_line, str) and pass_line in pass_lines:
            counted_stale.append(artifact_id)
    require(
        not counted_stale,
        f"release_candidate_manifest counts stale artifacts as pass evidence: {sorted(set(counted_stale))}",
    )


def candidate_path_variants(raw: str, *, base: Path) -> set[str]:
    variants = {raw}
    path = Path(raw)
    if path.is_absolute():
        variants.add(str(path.resolve()))
    else:
        variants.add(str((base / path).resolve()))
        variants.add(str((REPO_ROOT / path).resolve()))
    return variants


def validate_release_candidate_stage_evidence(
    release_candidate: dict[str, Any],
    change_root: Path,
    stage_artifacts: dict[str, dict[str, str]],
) -> None:
    required_gates = set(
        require_string_list(
            release_candidate.get("required_gates", []),
            "release_candidate_manifest.required_gates",
        )
    )
    satisfied_gates = set(
        require_string_list(
            release_candidate.get("satisfied_gates", []),
            "release_candidate_manifest.satisfied_gates",
        )
    )
    artifact_paths = require_string_list(
        release_candidate.get("artifact_paths", []),
        "release_candidate_manifest.artifact_paths",
    )
    pass_lines = set(
        require_string_list(
            release_candidate.get("pass_lines", []),
            "release_candidate_manifest.pass_lines",
        )
    )

    missing_required = sorted(REQUIRED_RELEASE_CANDIDATE_STAGE_GATES - required_gates)
    require(
        not missing_required,
        f"release_candidate_manifest.required_gates missing final gates {missing_required}",
    )
    missing_satisfied = sorted(REQUIRED_RELEASE_CANDIDATE_STAGE_GATES - satisfied_gates)
    require(
        not missing_satisfied,
        f"release_candidate_manifest.satisfied_gates missing final gates {missing_satisfied}",
    )

    artifact_path_variants: set[str] = set()
    for raw_path in artifact_paths:
        artifact_path_variants.update(candidate_path_variants(raw_path, base=change_root))

    missing_artifact_paths: list[str] = []
    missing_pass_lines: list[str] = []
    for gate in sorted(REQUIRED_RELEASE_CANDIDATE_STAGE_GATES):
        artifact = stage_artifacts.get(gate)
        require(artifact is not None, f"final gate missing internal stage artifact entry for {gate}")
        expected_path = artifact["artifact_dir"]
        expected_path_variants = candidate_path_variants(expected_path, base=REPO_ROOT)
        if not (expected_path_variants & artifact_path_variants):
            missing_artifact_paths.append(gate)
        if artifact["pass_line"] not in pass_lines:
            missing_pass_lines.append(gate)

    require(
        not missing_artifact_paths,
        "release_candidate_manifest.artifact_paths missing final stage artifacts "
        f"{missing_artifact_paths}",
    )
    require(
        not missing_pass_lines,
        f"release_candidate_manifest.pass_lines missing final stage PASS lines {missing_pass_lines}",
    )


def validate_required_planner_fixture(
    fixtures: dict[str, dict[str, Any]],
    fixture_id: str,
    *,
    required_gates: set[str],
    forbidden_gates: set[str] | None = None,
) -> None:
    fixture = fixtures.get(fixture_id)
    require(fixture is not None, f"planner_selfcheck missing fixture {fixture_id}")
    require(fixture.get("status") == "pass", f"planner_selfcheck.{fixture_id}.status must be pass")
    actual_required = set(
        require_string_list(
            fixture.get("required_gates", []),
            f"planner_selfcheck.{fixture_id}.required_gates",
        )
    )
    missing = sorted(required_gates - actual_required)
    require(not missing, f"planner_selfcheck.{fixture_id}.required_gates missing {missing}")
    forbidden = sorted((forbidden_gates or set()) & actual_required)
    require(not forbidden, f"planner_selfcheck.{fixture_id}.required_gates unexpectedly contained {forbidden}")


def validate_planner_selfcheck(root: Path) -> dict[str, Any]:
    path = root / "planner_selfcheck.json"
    selfcheck = read_json(path)
    require_status_pass(selfcheck, "change_impact.planner_selfcheck")
    fixtures = selfcheck.get("fixtures")
    require(isinstance(fixtures, list) and fixtures, "change_impact.planner_selfcheck.fixtures must be non-empty")
    fixture_map: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(fixtures):
        require(isinstance(fixture, dict), f"change_impact.planner_selfcheck.fixtures[{index}] must be an object")
        fixture_id = fixture.get("id")
        require(
            isinstance(fixture_id, str) and fixture_id.strip(),
            f"change_impact.planner_selfcheck.fixtures[{index}].id must be non-empty",
        )
        fixture_map[fixture_id] = fixture
    validate_required_planner_fixture(
        fixture_map,
        "engine_shared_runtime",
        required_gates={"metal_sentinel", "cuda_sentinel"},
    )
    validate_required_planner_fixture(
        fixture_map,
        "cuda_kernel_local",
        required_gates={"cuda_sentinel", "metal_boundary_smoke"},
        forbidden_gates={"metal_full"},
    )
    validate_required_planner_fixture(
        fixture_map,
        "metal_kernel_local",
        required_gates={"metal_sentinel", "cuda_boundary_smoke"},
        forbidden_gates={"cuda_full"},
    )
    return {**selfcheck, "_summary_path": str(path)}


def validate_change_impact(root: Path, expected_sha: str) -> dict[str, Any]:
    gate_plan = read_json(root / "gate_plan.json")
    require_status_pass(gate_plan, "change_impact.gate_plan")
    require(gate_plan.get("unknown_files") == [], "change_impact.gate_plan.unknown_files must be empty")
    require(
        gate_plan.get("head_sha") == expected_sha,
        f"change_impact.gate_plan.head_sha {gate_plan.get('head_sha')!r} is stale vs HEAD {expected_sha}",
    )
    release_candidate = read_json(root / "release_candidate_manifest.json")
    require(
        release_candidate.get("head_sha") == expected_sha,
        "release_candidate_manifest.head_sha must match current HEAD",
    )
    require(
        isinstance(release_candidate.get("required_gates"), list),
        "release_candidate_manifest.required_gates must be a list",
    )
    validate_release_candidate_invalidation(gate_plan, release_candidate)
    planner_selfcheck = validate_planner_selfcheck(root)
    pass_line = f"CHANGE IMPACT GATE PLAN PASS: {root}"
    return {
        "label": "change_impact",
        "artifact_dir": str(root),
        "manifest_path": None,
        "pass_line": pass_line,
        "git_sha": gate_plan.get("head_sha"),
        "gate_plan": gate_plan,
        "release_candidate_manifest": release_candidate,
        "planner_selfcheck": planner_selfcheck,
    }


def validate_product_scenario_coverage(summary: dict[str, Any]) -> None:
    required = summary.get("required_stage2_fixture_count")
    require(
        isinstance(required, int) and required >= 12,
        "product_sentinel.summary must cover >=12 stage-2 fixtures",
    )
    scenario_count = summary.get("scenario_count")
    require(
        isinstance(scenario_count, int) and scenario_count >= required,
        "product_sentinel.summary.scenario_count must be >= required_stage2_fixture_count",
    )
    scenarios = summary.get("scenarios")
    require(isinstance(scenarios, list), "product_sentinel.summary.scenarios must be a list")
    require(
        len(scenarios) == scenario_count,
        "product_sentinel.summary.scenarios length must match scenario_count",
    )
    scenario_types: set[str] = set()
    for index, scenario in enumerate(scenarios):
        require(
            isinstance(scenario, dict),
            f"product_sentinel.summary.scenarios[{index}] must be an object",
        )
        require(
            scenario.get("status") == "pass",
            f"product_sentinel.summary.scenarios[{index}].status must be pass",
        )
        scenario_type = scenario.get("type")
        require(
            isinstance(scenario_type, str) and scenario_type.strip(),
            f"product_sentinel.summary.scenarios[{index}].type must be non-empty",
        )
        scenario_types.add(scenario_type)
    missing_types = sorted(REQUIRED_PRODUCT_SENTINEL_SCENARIO_TYPES - scenario_types)
    require(
        not missing_types,
        f"product_sentinel.summary.scenarios missing scenario types {missing_types}",
    )


def validate_product_sentinel(root: Path, expected_sha: str) -> dict[str, Any]:
    stage = load_stage_manifest(root, "product_sentinel", "PRODUCT BACKEND SENTINEL PASS", expected_sha)
    summary = load_summary_from_manifest(stage, "product_backend_sentinel_summary.json")
    require(summary.get("failed") == 0, "product_sentinel.summary.failed must be 0")
    validate_product_scenario_coverage(summary)
    failure_links = summary.get("failure_links")
    require(isinstance(failure_links, list), "product_sentinel.summary.failure_links must be a list")
    required_failure_kinds = {"bad_output", "oom_admission", "panic_error"}
    seen_failure_kinds = set()
    for index, link in enumerate(failure_links):
        require(isinstance(link, dict), f"product_sentinel.summary.failure_links[{index}] must be an object")
        failure_kind = link.get("failure_kind")
        require(
            isinstance(failure_kind, str) and failure_kind.strip(),
            f"product_sentinel.summary.failure_links[{index}].failure_kind must be non-empty",
        )
        seen_failure_kinds.add(failure_kind)
        for key in ("profile_event_id", "request_id", "replay_command", "bundle_dir"):
            require(
                isinstance(link.get(key), str) and link[key].strip(),
                f"product_sentinel.summary.failure_links[{index}].{key} must be non-empty",
            )
        if failure_kind != "bad_output":
            require(
                isinstance(link.get("failure_diagnostics"), str) and link["failure_diagnostics"].strip(),
                f"product_sentinel.summary.failure_links[{index}].failure_diagnostics must be non-empty",
            )
    missing = sorted(required_failure_kinds - seen_failure_kinds)
    require(not missing, f"product_sentinel.summary.failure_links missing {missing}")
    stage["summary"] = summary
    return stage


def validate_model_contract(root: Path, expected_sha: str) -> dict[str, Any]:
    stage = load_stage_manifest(root, "model_contract", "MODEL ONBOARDING CONTRACT PASS", expected_sha)
    summary = load_summary_from_manifest(stage, "model_onboarding_contract_summary.json")
    contracts = summary.get("contracts")
    require(isinstance(contracts, list) and contracts, "model_contract.summary.contracts must be non-empty")
    seen_contract_ids: set[str] = set()
    for index, contract in enumerate(contracts):
        require(isinstance(contract, dict), f"model_contract.summary.contracts[{index}] must be an object")
        require(contract.get("status") == "pass", f"model_contract.summary.contracts[{index}].status must be pass")
        contract_id = contract.get("contract_id")
        require(
            isinstance(contract_id, str) and contract_id.strip(),
            f"model_contract.summary.contracts[{index}].contract_id must be non-empty",
        )
        require(
            contract_id not in seen_contract_ids,
            f"model_contract.summary.contracts duplicate contract_id {contract_id!r}",
        )
        seen_contract_ids.add(contract_id)
    stage["summary"] = summary
    return stage


def validate_model_support_linkage(
    model_summary: dict[str, Any],
    support_summary: dict[str, Any],
) -> None:
    contracts = model_summary.get("contracts")
    require(isinstance(contracts, list), "model_contract.summary.contracts must be a list")
    contract_ids = {
        contract["contract_id"]
        for contract in contracts
        if isinstance(contract, dict)
        and isinstance(contract.get("contract_id"), str)
        and contract["contract_id"].strip()
    }
    require(contract_ids, "model_contract.summary must expose at least one contract_id")
    rows = support_summary.get("rows")
    require(isinstance(rows, list), "support_matrix_contract.summary.rows must be a list")
    missing: list[str] = []
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"support_matrix_contract.summary.rows[{index}] must be an object")
        contract_id = row.get("contract_id")
        require(
            isinstance(contract_id, str) and contract_id.strip(),
            f"support_matrix_contract.summary.rows[{index}].contract_id must be non-empty",
        )
        if contract_id not in contract_ids:
            missing.append(f"{row.get('model_id', f'row-{index}')}->{contract_id}")
    require(
        not missing,
        "support_matrix_contract.summary references contracts missing from model_contract.summary: "
        + ", ".join(missing),
    )


def validate_support_matrix_contract(root: Path, expected_sha: str) -> dict[str, Any]:
    stage = load_stage_manifest(root, "support_matrix_contract", "SUPPORT MATRIX CONTRACT PASS", expected_sha)
    summary = load_summary_from_manifest(stage, "support_matrix_contract_summary.json")
    rows = summary.get("rows")
    require(isinstance(rows, list) and rows, "support_matrix_contract.summary.rows must be non-empty")
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"support_matrix_contract.summary.rows[{index}] must be an object")
        require(
            isinstance(row.get("contract_id"), str) and row["contract_id"],
            f"support_matrix_contract.summary.rows[{index}].contract_id must be non-empty",
        )
    stage["summary"] = summary
    return stage


def validate_vertical_entrypoint(entry: Any, *, entrypoint: str, artifact_root: Path) -> None:
    require(
        isinstance(entry, dict),
        f"observability_vertical_slice.summary.entrypoints.{entrypoint} must be an object",
    )
    require(
        entry.get("schema_version") == SCHEMA_VERSION,
        f"observability_vertical_slice.summary.entrypoints.{entrypoint}.schema_version must be {SCHEMA_VERSION}",
    )
    event_count = entry.get("event_count")
    require(
        isinstance(event_count, int) and event_count > 0,
        f"observability_vertical_slice.summary.entrypoints.{entrypoint}.event_count must be positive",
    )
    for key in ("profile_jsonl", "request_dump", "summary"):
        value = entry.get(key)
        require(
            isinstance(value, str) and value.strip(),
            f"observability_vertical_slice.summary.entrypoints.{entrypoint}.{key} must be non-empty",
        )
        resolved = resolve_path(value, base=artifact_root)
        require(
            resolved.is_file(),
            f"observability_vertical_slice.summary.entrypoints.{entrypoint}.{key} must exist: {resolved}",
        )
    replay_command = entry.get("replay_command")
    require(
        isinstance(replay_command, str) and replay_command.strip(),
        f"observability_vertical_slice.summary.entrypoints.{entrypoint}.replay_command must be non-empty",
    )


def validate_vertical_slice_summary(summary: dict[str, Any], *, artifact_root: Path) -> None:
    require(
        summary.get("gate") == "observability_vertical_slice",
        "observability_vertical_slice.summary.gate must be observability_vertical_slice",
    )
    require(summary.get("l0_only") is True, "observability_vertical_slice.summary.l0_only must be true")
    require(summary.get("same_schema_version") is True, "observability_vertical_slice.summary.same_schema_version must be true")
    entrypoints = summary.get("entrypoints")
    require(isinstance(entrypoints, dict), "observability_vertical_slice.summary.entrypoints must be an object")
    for entrypoint in ("run", "serve"):
        validate_vertical_entrypoint(
            entrypoints.get(entrypoint),
            entrypoint=entrypoint,
            artifact_root=artifact_root,
        )
    analyzer = summary.get("analyzer")
    require(isinstance(analyzer, dict), "observability_vertical_slice.summary.analyzer must be an object")
    stdout = analyzer.get("stdout")
    require(
        isinstance(stdout, str) and "FERRUM PROFILE ANALYZER PASS" in stdout,
        "observability_vertical_slice.summary.analyzer.stdout must include FERRUM PROFILE ANALYZER PASS",
    )


def load_vertical_slice(path_value: Any, observability_root: Path, expected_sha: str) -> dict[str, Any]:
    require(isinstance(path_value, str) and path_value.strip(), "observability_profile.summary.vertical_slice_artifact is required")
    path = resolve_path(path_value, base=observability_root)
    require(path.is_dir(), f"observability_profile.summary.vertical_slice_artifact must be an artifact directory: {path}")
    manifest_path = path / "observability_vertical_slice_manifest.json"
    manifest = read_json(manifest_path)
    require_status_pass(manifest, "observability_vertical_slice.manifest")
    require_pass_line(manifest, "observability_vertical_slice.manifest", "OBSERVABILITY VERTICAL SLICE PASS")
    require_git_sha(manifest, "observability_vertical_slice.manifest", expected_sha)
    require_clean_manifest(manifest, "observability_vertical_slice.manifest")
    summary_path = resolve_path(str(manifest.get("summary")), base=path) if manifest.get("summary") else path / "observability_profile_summary.json"
    summary = read_json(summary_path)
    require_status_pass(summary, "observability_vertical_slice.summary")
    validate_vertical_slice_summary(summary, artifact_root=path)
    return {"manifest": manifest, "summary": summary, "summary_path": str(summary_path)}


def require_zero_count(summary: dict[str, Any], field: str, label: str) -> None:
    value = summary.get(field)
    require(
        isinstance(value, int) and value == 0,
        f"{label}.{field} must be 0 for final PASS",
    )


def require_non_negative_int(value: Any, label: str) -> int:
    require(isinstance(value, int) and value >= 0, f"{label} must be a non-negative integer")
    return value


def validate_observability_diagnostic_shape(summary: dict[str, Any], label: str) -> None:
    latency = summary.get("latency_p50_p95_p99")
    require(isinstance(latency, dict), f"{label}.latency_p50_p95_p99 must be an object")
    duration = latency.get("duration_us")
    require(
        isinstance(duration, dict),
        f"{label}.latency_p50_p95_p99.duration_us must be an object",
    )
    p50 = require_non_negative_int(duration.get("p50"), f"{label}.latency_p50_p95_p99.duration_us.p50")
    p95 = require_non_negative_int(duration.get("p95"), f"{label}.latency_p50_p95_p99.duration_us.p95")
    p99 = require_non_negative_int(duration.get("p99"), f"{label}.latency_p50_p95_p99.duration_us.p99")
    sample_count = require_non_negative_int(
        duration.get("sample_count"),
        f"{label}.latency_p50_p95_p99.duration_us.sample_count",
    )
    require(sample_count > 0, f"{label}.latency_p50_p95_p99.duration_us.sample_count must be positive")
    require(p50 <= p95 <= p99, f"{label}.latency_p50_p95_p99.duration_us percentiles must be ordered")

    memory = summary.get("memory_high_water_bytes")
    require(isinstance(memory, dict), f"{label}.memory_high_water_bytes must be an object")
    memory_max = require_non_negative_int(memory.get("max"), f"{label}.memory_high_water_bytes.max")
    by_backend_scope = memory.get("by_backend_scope")
    require(
        isinstance(by_backend_scope, dict) and by_backend_scope,
        f"{label}.memory_high_water_bytes.by_backend_scope must be a non-empty object",
    )
    scope_values: list[int] = []
    for scope, value in sorted(by_backend_scope.items()):
        require(
            isinstance(scope, str) and scope.strip(),
            f"{label}.memory_high_water_bytes.by_backend_scope keys must be non-empty",
        )
        scope_values.append(
            require_non_negative_int(
                value,
                f"{label}.memory_high_water_bytes.by_backend_scope.{scope}",
            )
        )
    require(memory_max == max(scope_values), f"{label}.memory_high_water_bytes.max must equal by_backend_scope maximum")

    slow_phases = summary.get("top_slow_phases")
    require(
        isinstance(slow_phases, list) and slow_phases,
        f"{label}.top_slow_phases must contain at least one timed phase",
    )
    for index, phase in enumerate(slow_phases):
        require(isinstance(phase, dict), f"{label}.top_slow_phases[{index}] must be an object")
        for key in ("event_id", "request_id", "entrypoint", "backend", "phase"):
            value = phase.get(key)
            require(
                isinstance(value, str) and value.strip(),
                f"{label}.top_slow_phases[{index}].{key} must be non-empty",
            )
        require_non_negative_int(
            phase.get("duration_us"),
            f"{label}.top_slow_phases[{index}].duration_us",
        )


def validate_replay_commands(value: Any, label: str) -> None:
    require(isinstance(value, list) and value, f"{label} must contain at least one replay command")
    for index, command in enumerate(value):
        require(isinstance(command, dict), f"{label}[{index}] must be an object")
        for key in ("event_id", "request_id", "command", "bundle_dir"):
            item = command.get(key)
            require(
                isinstance(item, str) and item.strip(),
                f"{label}[{index}].{key} must be non-empty",
            )


def validate_observability_profile(root: Path, expected_sha: str) -> dict[str, Any]:
    stage = load_stage_manifest(root, "observability_profile", "OBSERVABILITY PROFILE GATE PASS", expected_sha)
    summary = load_summary_from_manifest(stage, "observability_profile_summary.json")
    missing = sorted(OBSERVABILITY_SUMMARY_FIELDS - set(summary))
    require(not missing, f"observability_profile.summary missing fields: {missing}")
    request_count = summary.get("request_count")
    require(
        isinstance(request_count, int) and request_count > 0,
        "observability_profile.summary.request_count must be a positive integer",
    )
    validate_observability_diagnostic_shape(summary, "observability_profile.summary")
    for field in (
        "failed_count",
        "corrupted_count",
        "bad_text_count",
        "silent_oom_count",
        "resource_leak_count",
    ):
        require_zero_count(summary, field, "observability_profile.summary")
    require(
        summary.get("first_failure_event") is None,
        "observability_profile.summary.first_failure_event must be null when failed_count is 0",
    )
    validate_replay_commands(
        summary.get("replay_commands"),
        "observability_profile.summary.replay_commands",
    )
    entrypoints = set(summary.get("entrypoints") or [])
    require({"run", "serve"} <= entrypoints, "observability_profile.summary must include run and serve entrypoints")
    stage["summary"] = summary
    stage["vertical_slice"] = load_vertical_slice(summary.get("vertical_slice_artifact"), root, expected_sha)
    return stage


def validate_native_operator(root: Path, expected_sha: str) -> dict[str, Any]:
    stage = load_stage_manifest(root, "native_operator", "NATIVE OP ARTIFACT PASS", expected_sha)
    summary = load_summary_from_manifest(stage, "native_operator_artifact_summary.json")
    bulk = summary.get("bulk_source")
    third_party = summary.get("unregistered_third_party_source")
    require(isinstance(bulk, dict), "native_operator.summary.bulk_source must be an object")
    require(isinstance(third_party, dict), "native_operator.summary.unregistered_third_party_source must be an object")
    require(bulk.get("count") == 0, "native_operator.summary.bulk_source.count must be 0")
    require(third_party.get("count") == 0, "native_operator.summary.unregistered_third_party_source.count must be 0")
    manifests = summary.get("manifests")
    require(isinstance(manifests, list) and manifests, "native_operator.summary.manifests must be non-empty")
    operators = set()
    for index, manifest in enumerate(manifests):
        require(isinstance(manifest, dict), f"native_operator.summary.manifests[{index}] must be an object")
        operator = manifest.get("operator")
        require(
            isinstance(operator, str) and operator.strip(),
            f"native_operator.summary.manifests[{index}].operator must be non-empty",
        )
        operators.add(operator)
        for key in ("backend", "compute_capabilities", "linkage", "binary_sha256", "resolution"):
            require(
                key in manifest,
                f"native_operator.summary.manifests[{index}].{key} is required",
            )
    require("fa2" in operators, "native_operator.summary.manifests must include fa2")
    selftest_summary = summary.get("selftest_summary")
    require(isinstance(selftest_summary, dict), "native_operator.summary.selftest_summary must be an object")
    require_status_pass(selftest_summary, "native_operator.summary.selftest_summary")
    pass_fixtures = set(
        require_string_list(
            selftest_summary.get("pass_fixtures", []),
            "native_operator.summary.selftest_summary.pass_fixtures",
        )
    )
    missing_pass_fixtures = sorted(REQUIRED_NATIVE_PASS_FIXTURES - pass_fixtures)
    require(
        not missing_pass_fixtures,
        f"native_operator.summary.selftest_summary.pass_fixtures missing {missing_pass_fixtures}",
    )
    fail_closed_cases = set(
        require_string_list(
            selftest_summary.get("resolver_fail_closed_cases", []),
            "native_operator.summary.selftest_summary.resolver_fail_closed_cases",
        )
    )
    missing_fail_closed = sorted(REQUIRED_NATIVE_RESOLVER_FAIL_CLOSED_CASES - fail_closed_cases)
    require(
        not missing_fail_closed,
        "native_operator.summary.selftest_summary.resolver_fail_closed_cases missing "
        + str(missing_fail_closed),
    )
    require(
        selftest_summary.get("python_runtime_dependency") == "none",
        "native_operator.summary.selftest_summary.python_runtime_dependency must be none",
    )
    stage["summary"] = summary
    stage["fa2_source_removal_inventory"] = {
        "bulk_source": bulk,
        "unregistered_third_party_source": third_party,
    }
    return stage


def artifact_git_sha(artifact: dict[str, Any], label: str, expected_sha: str) -> None:
    require(artifact.get("status") == "pass", f"{label}.status must be pass")
    require_git_sha(artifact, label, expected_sha)


def validate_backend_resolution(artifact: dict[str, Any], label: str, backend: str) -> None:
    requested = artifact.get("requested_backend")
    effective = artifact.get("effective_backend")
    require(
        isinstance(requested, str) and requested.strip(),
        f"{label}.requested_backend must be non-empty",
    )
    require(
        isinstance(effective, str) and effective.strip(),
        f"{label}.effective_backend must be non-empty",
    )
    require(effective == backend, f"{label}.effective_backend must be {backend}")
    require(
        requested == "auto" or requested == effective,
        f"{label}.requested_backend {requested!r} does not match effective_backend {effective!r}",
    )


def validate_replay_index(value: Any, *, artifact_dir: Path, label: str) -> None:
    require(isinstance(value, list), f"{label}.replay_bundle_index must be a list")
    for index, entry in enumerate(value):
        entry_label = f"{label}.replay_bundle_index[{index}]"
        require(isinstance(entry, dict), f"{entry_label} must be an object")
        for key in ("request_id", "replay_command", "bundle_dir"):
            require(
                isinstance(entry.get(key), str) and entry[key].strip(),
                f"{entry_label}.{key} must be non-empty",
            )
        bundle_dir = resolve_path(entry["bundle_dir"], base=artifact_dir)
        require(bundle_dir.is_dir(), f"{entry_label}.bundle_dir must exist: {bundle_dir}")


def artifact_from_reference(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return read_json(path)
    manifest = read_json(path / "gate.manifest.json")
    summary_path = None
    outputs = manifest.get("outputs")
    if isinstance(outputs, dict):
        summary_path = outputs.get("summary")
    if summary_path is None:
        summary_path = manifest.get("summary")
    extra: dict[str, Any] = {}
    if isinstance(summary_path, str) and summary_path.strip():
        candidate = resolve_path(summary_path, base=path)
        if candidate.is_file():
            extra = read_json(candidate)
    return {
        **extra,
        "status": manifest.get("status", extra.get("status")),
        "git_sha": manifest.get("git_sha", extra.get("git_sha")),
        "artifact_dir": manifest.get("artifact_dir", extra.get("artifact_dir", str(path))),
        "pass_line": manifest.get("pass_line", extra.get("pass_line")),
    }


def validate_l2_artifact(
    data: dict[str, Any],
    key: str,
    backend: str,
    expected_sha: str,
    *,
    summary_path: Path,
) -> None:
    artifact = data.get(key)
    require(isinstance(artifact, dict), f"actual_model_regression.{key} must be an object")
    artifact_git_sha(artifact, f"actual_model_regression.{key}", expected_sha)
    require(artifact.get("backend") == backend, f"actual_model_regression.{key}.backend must be {backend}")
    validate_backend_resolution(artifact, f"actual_model_regression.{key}", backend)
    entrypoints = set(artifact.get("entrypoints") or [])
    missing = sorted(REQUIRED_L2_ENTRYPOINTS - entrypoints)
    require(not missing, f"actual_model_regression.{key}.entrypoints missing {missing}")
    artifact_dir = artifact.get("artifact_dir")
    require(isinstance(artifact_dir, str) and artifact_dir.strip(), f"actual_model_regression.{key}.artifact_dir must be non-empty")
    resolved_artifact_dir = resolve_path(artifact_dir, base=summary_path.parent)
    require(
        resolved_artifact_dir.is_dir(),
        f"actual_model_regression.{key}.artifact_dir must exist and be a directory: {resolved_artifact_dir}",
    )
    require_real_pass_line(artifact.get("pass_line"), f"actual_model_regression.{key}")
    require(
        isinstance(artifact.get("model_id"), str) and artifact["model_id"].strip(),
        f"actual_model_regression.{key}.model_id must be non-empty",
    )
    require(
        isinstance(artifact.get("architecture"), str) and artifact["architecture"].strip(),
        f"actual_model_regression.{key}.architecture must be non-empty",
    )
    allowed_architectures = ALLOWED_L2_ARCHITECTURES_BY_BACKEND[backend]
    require(
        artifact["architecture"] in allowed_architectures,
        f"actual_model_regression.{key}.architecture must be one of {sorted(allowed_architectures)}",
    )
    require(
        isinstance(artifact.get("git_dirty"), bool),
        f"actual_model_regression.{key}.git_dirty must be boolean",
    )
    dirty_files = require_string_list(
        artifact.get("dirty_files", []),
        f"actual_model_regression.{key}.dirty_files",
    )
    require(
        artifact["git_dirty"] is False,
        f"actual_model_regression.{key}.git_dirty must be false for final PASS",
    )
    require(
        not dirty_files,
        f"actual_model_regression.{key}.dirty_files must be empty for final PASS",
    )
    normalize_command(
        artifact.get("command") or artifact.get("command_line"),
        f"actual_model_regression.{key}.command",
    )
    profile_detail = artifact.get("profile_detail") or artifact.get("observability_profile_detail")
    require(
        isinstance(profile_detail, str) and profile_detail.strip(),
        f"actual_model_regression.{key}.profile_detail must be non-empty",
    )
    replay_index = artifact.get("replay_bundle_index", [])
    validate_replay_index(
        replay_index,
        artifact_dir=resolved_artifact_dir,
        label=f"actual_model_regression.{key}",
    )


def validate_selected_native_cuda_artifact(
    selection: dict[str, Any],
    *,
    expected_sha: str,
    summary_path: Path,
) -> None:
    raw = selection.get("cuda_artifact")
    require(
        isinstance(raw, str) and raw.strip(),
        "native operator selected path requires cuda_artifact",
    )
    artifact_path = resolve_path(raw, base=summary_path.parent)
    require(
        artifact_path.exists(),
        f"actual_model_regression.native_operator_selection.cuda_artifact must exist: {artifact_path}",
    )
    artifact = artifact_from_reference(artifact_path)
    label = "actual_model_regression.native_operator_selection.cuda_artifact"
    require(artifact.get("status") == "pass", f"{label}.status must be pass")
    require_git_sha(artifact, label, expected_sha)
    require(artifact.get("backend") == "cuda", f"{label}.backend must be cuda")
    require_real_pass_line(artifact.get("pass_line"), label)
    artifact_dir = artifact.get("artifact_dir")
    require(isinstance(artifact_dir, str) and artifact_dir.strip(), f"{label}.artifact_dir must be non-empty")
    resolved_artifact_dir = resolve_path(artifact_dir, base=artifact_path if artifact_path.is_dir() else artifact_path.parent)
    require(resolved_artifact_dir.is_dir(), f"{label}.artifact_dir must exist: {resolved_artifact_dir}")


def find_actual_model_regression_path(args: argparse.Namespace) -> Path | None:
    if args.actual_model_regression_summary is not None:
        if args.actual_model_regression_summary.is_dir():
            return args.actual_model_regression_summary / "actual_model_regression_summary.json"
        return args.actual_model_regression_summary
    for root in [args.product_sentinel, args.observability_profile]:
        for name in [
            "actual_model_regression_summary.json",
            "l2_actual_model_regression_summary.json",
        ]:
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None


def validate_actual_model_regression(args: argparse.Namespace, expected_sha: str) -> dict[str, Any]:
    path = find_actual_model_regression_path(args)
    require(path is not None, "actual model regression summary is required for final PASS")
    summary = read_json(path)
    require_status_pass(summary, "actual_model_regression")
    require_pass_line(summary, "actual_model_regression", "ACTUAL MODEL REGRESSION SUMMARY PASS")
    require_git_sha(summary, "actual_model_regression", expected_sha)
    validate_l2_artifact(
        summary,
        "metal_l2_artifact",
        "metal",
        expected_sha,
        summary_path=path,
    )
    validate_l2_artifact(
        summary,
        "cuda_l2_artifact",
        "cuda",
        expected_sha,
        summary_path=path,
    )
    selection = summary.get("native_operator_selection")
    require(isinstance(selection, dict), "actual_model_regression.native_operator_selection must be an object")
    require(selection.get("status") == "pass", "actual_model_regression.native_operator_selection.status must be pass")
    selected = selection.get("selected")
    if selected is True:
        validate_selected_native_cuda_artifact(
            selection,
            expected_sha=expected_sha,
            summary_path=path,
        )
    elif selected is False:
        require(isinstance(selection.get("non_selected_reason"), str) and selection["non_selected_reason"].strip(), "native operator non-selected path requires non_selected_reason")
    else:
        raise GoalGateError("actual_model_regression.native_operator_selection.selected must be boolean")
    return {**summary, "_summary_path": str(path)}


def append_replay_index_entry(entries: list[dict[str, Any]], source: str, entry: dict[str, Any]) -> None:
    replay_command = entry.get("replay_command") or entry.get("command")
    request_id = entry.get("request_id")
    bundle_dir = entry.get("bundle_dir")
    if not (
        isinstance(replay_command, str)
        and replay_command.strip()
        and isinstance(request_id, str)
        and request_id.strip()
        and isinstance(bundle_dir, str)
        and bundle_dir.strip()
    ):
        return
    normalized = {
        "source": source,
        "request_id": request_id,
        "replay_command": replay_command,
        "bundle_dir": bundle_dir,
    }
    for key in (
        "profile_event_id",
        "event_id",
        "failure_kind",
        "entrypoint",
        "backend",
        "model_id",
        "artifact_dir",
        "failure_diagnostics",
    ):
        if key in entry:
            normalized[key] = entry[key]
    entries.append(normalized)


def build_replay_bundle_index(
    product_summary: dict[str, Any],
    observability_summary: dict[str, Any],
    actual_model_summary: dict[str, Any],
    out: Path,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for link in product_summary.get("failure_links") or []:
        if isinstance(link, dict):
            append_replay_index_entry(entries, "product_sentinel.failure_links", link)
    for command in observability_summary.get("replay_commands") or []:
        if isinstance(command, dict):
            append_replay_index_entry(entries, "observability_profile.replay_commands", command)
    for key in ("metal_l2_artifact", "cuda_l2_artifact"):
        artifact = actual_model_summary.get(key)
        if not isinstance(artifact, dict):
            continue
        for entry in artifact.get("replay_bundle_index") or []:
            if isinstance(entry, dict):
                enriched = {
                    **entry,
                    "backend": artifact.get("backend"),
                    "model_id": artifact.get("model_id"),
                    "artifact_dir": artifact.get("artifact_dir"),
                }
                append_replay_index_entry(entries, f"actual_model_regression.{key}", enriched)
    require(entries, "replay bundle index must contain at least one replay entry")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "goal": GOAL,
        "path": str(out / "replay_bundle_index.json"),
        "entry_count": len(entries),
        "entries": entries,
    }


def sanitized_env() -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        upper = key.upper()
        if any(marker in upper for marker in SECRET_ENV_MARKERS):
            continue
        if key in SAFE_ENV_NAMES or any(key.startswith(prefix) for prefix in SAFE_ENV_PREFIXES):
            result[key] = value
    return result


def validate_binary_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    require(SHA256_RE.match(value), "--binary-sha256 must be a 64-character hex digest")
    return value


def build_goal_manifest(args: argparse.Namespace) -> dict[str, Any]:
    started_at = int(time.time())
    expected_sha = head_sha()
    dirty_files = current_dirty_files()
    if args.require_clean:
        tracked_dirty = [line for line in dirty_files if not line.startswith("?? ")]
        require(not tracked_dirty, f"--require-clean failed; tracked dirty files: {tracked_dirty}")
    resource = validate_resource_invariant(args.resource_invariant, expected_sha)
    change_impact = validate_change_impact(args.change_impact, expected_sha)
    product = validate_product_sentinel(args.product_sentinel, expected_sha)
    model_contract = validate_model_contract(args.model_contract, expected_sha)
    support_matrix_contract = validate_support_matrix_contract(args.support_matrix_contract, expected_sha)
    validate_model_support_linkage(model_contract["summary"], support_matrix_contract["summary"])
    observability = validate_observability_profile(args.observability_profile, expected_sha)
    native_operator = validate_native_operator(args.native_operator, expected_sha)
    actual_model = validate_actual_model_regression(args, expected_sha)
    stage_artifacts = {
        "resource_invariant": {
            "artifact_dir": resource["artifact_dir"],
            "pass_line": resource["pass_line"],
            "git_sha": resource["git_sha"],
        },
        "change_impact": {
            "artifact_dir": change_impact["artifact_dir"],
            "pass_line": change_impact["pass_line"],
            "git_sha": change_impact["git_sha"],
        },
        "product_sentinel": {
            "artifact_dir": product["artifact_dir"],
            "pass_line": product["pass_line"],
            "git_sha": product["git_sha"],
        },
        "model_contract": {
            "artifact_dir": model_contract["artifact_dir"],
            "pass_line": model_contract["pass_line"],
            "git_sha": model_contract["git_sha"],
        },
        "support_matrix_contract": {
            "artifact_dir": support_matrix_contract["artifact_dir"],
            "pass_line": support_matrix_contract["pass_line"],
            "git_sha": support_matrix_contract["git_sha"],
        },
        "observability_profile": {
            "artifact_dir": observability["artifact_dir"],
            "pass_line": observability["pass_line"],
            "git_sha": observability["git_sha"],
        },
        "native_operator": {
            "artifact_dir": native_operator["artifact_dir"],
            "pass_line": native_operator["pass_line"],
            "git_sha": native_operator["git_sha"],
        },
        "actual_model_regression": {
            "artifact_dir": str(Path(actual_model["_summary_path"]).parent),
            "pass_line": actual_model["pass_line"],
            "git_sha": actual_model["git_sha"],
        },
    }
    validate_release_candidate_stage_evidence(
        change_impact["release_candidate_manifest"],
        Path(change_impact["artifact_dir"]),
        stage_artifacts,
    )
    replay_bundle_index = build_replay_bundle_index(
        product["summary"],
        observability["summary"],
        actual_model,
        args.out,
    )
    ended_at = int(time.time())
    pass_line = f"{PASS_LINE}: {args.out}"
    return {
        "schema_version": SCHEMA_VERSION,
        "goal": GOAL,
        "status": "pass",
        "pass_line": pass_line,
        "artifact_dir": str(args.out),
        "started_at_unix": started_at,
        "ended_at_unix": ended_at,
        "duration_sec": ended_at - started_at,
        "repo_root": str(REPO_ROOT),
        "git_sha": expected_sha,
        "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(dirty_files),
        "changed_files": dirty_files,
        "binary_sha256": validate_binary_sha256(args.binary_sha256),
        "command": sys.argv,
        "sanitized_env": sanitized_env(),
        "stage_artifacts": stage_artifacts,
        "gate_plan": change_impact["gate_plan"],
        "release_candidate_manifest": change_impact["release_candidate_manifest"],
        "resource_invariant_summary": resource["summary"],
        "product_scenario_summary": product["summary"],
        "model_contract_summary": model_contract["summary"],
        "support_matrix_contract_summary": support_matrix_contract["summary"],
        "observability_profile_summary": observability["summary"],
        "observability_vertical_slice_summary": observability["vertical_slice"],
        "actual_model_regression_summary": actual_model,
        "native_operator_artifact_summary": native_operator["summary"],
        "fa2_source_removal_inventory": native_operator["fa2_source_removal_inventory"],
        "replay_bundle_index": replay_bundle_index,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = build_goal_manifest(args)
    write_json(args.out / "replay_bundle_index.json", manifest["replay_bundle_index"])
    write_json(args.out / "goal_manifest.json", manifest)
    write_json(
        args.out / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "goal": GOAL,
            "phase": "final_aggregator",
            "status": "pass",
            "artifact_dir": str(args.out),
            "pass_line": manifest["pass_line"],
            "git_sha": manifest["git_sha"],
            "git_dirty": manifest["git_dirty"],
            "changed_files": manifest["changed_files"],
            "goal_manifest": str(args.out / "goal_manifest.json"),
            "replay_bundle_index": str(args.out / "replay_bundle_index.json"),
        },
    )
    (args.out / "pass_line.txt").write_text(manifest["pass_line"] + "\n", encoding="utf-8")
    return manifest


def make_gate_manifest(root: Path, prefix: str, sha: str, summary_name: str, summary: dict[str, Any]) -> None:
    write_json(root / summary_name, summary)
    write_json(
        root / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "artifact_dir": str(root),
            "pass_line": f"{prefix}: {root}",
            "git_sha": sha,
            "git_dirty": False,
            "dirty_files": [],
            "outputs": {"summary": str(root / summary_name)},
            "summary": str(root / summary_name),
        },
    )


def selftest_artifacts(root: Path, sha: str) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    resource = root / "resource"
    resource.mkdir()
    resource_summary = {
        "schema_version": 1,
        "status": "pass",
        "leaked_resources": 0,
        "underflow_count": 0,
        "silent_oom_count": 0,
        "panic_count": 0,
        "trace": {
            "source": str(resource / "resource_trace.jsonl"),
            "events": 72,
            "scenarios": sorted(REQUIRED_RESOURCE_SCENARIOS),
            "failures": [],
            "failure_counts": {},
        },
        "fixture_summary": {
            "scenario_count": len(REQUIRED_RESOURCE_SCENARIOS),
            "required_scenarios": sorted(REQUIRED_RESOURCE_SCENARIOS),
            "pass_fixtures": [],
            "fail_fixtures": [],
        },
        "resource_summary": {
            "backend_workspace": {
                "capacity": 2,
                "reserved": 3,
                "committed": 3,
                "released": 3,
                "leaked": 0,
            },
            "kv_block": {
                "capacity": 8,
                "reserved": 4,
                "committed": 4,
                "released": 4,
                "leaked": 0,
            },
            "model_cache_ref": {
                "capacity": 4,
                "reserved": 1,
                "committed": 1,
                "released": 1,
                "leaked": 0,
            },
            "recurrent_state_slot": {
                "capacity": 16,
                "reserved": 1,
                "committed": 1,
                "released": 1,
                "leaked": 0,
            },
            "scheduler_admission_slot": {
                "capacity": 2,
                "reserved": 2,
                "committed": 2,
                "released": 2,
                "leaked": 0,
            },
        },
    }
    write_json(resource / "invariant_report.json", resource_summary)
    write_json(
        resource / "gate.manifest.json",
        {
            "schema_version": 1,
            "status": "pass",
            "artifact_dir": str(resource),
            "pass_line": f"RESOURCE INVARIANT GATE PASS: {resource}",
            "git_sha": sha,
            "git_dirty": False,
            "dirty_files": [],
            "invariant_report": str(resource / "invariant_report.json"),
        },
    )

    change = root / "change"
    change.mkdir()
    release_candidate_stage_artifacts = {
        "actual_model_regression": {
            "artifact_dir": str(root),
            "pass_line": f"ACTUAL MODEL REGRESSION SUMMARY PASS: {root}",
        },
        "model_contract": {
            "artifact_dir": str(root / "model"),
            "pass_line": f"MODEL ONBOARDING CONTRACT PASS: {root / 'model'}",
        },
        "native_operator": {
            "artifact_dir": str(root / "native"),
            "pass_line": f"NATIVE OP ARTIFACT PASS: {root / 'native'}",
        },
        "observability_profile": {
            "artifact_dir": str(root / "observability"),
            "pass_line": f"OBSERVABILITY PROFILE GATE PASS: {root / 'observability'}",
        },
        "product_sentinel": {
            "artifact_dir": str(root / "product"),
            "pass_line": f"PRODUCT BACKEND SENTINEL PASS: {root / 'product'}",
        },
        "resource_invariant": {
            "artifact_dir": str(resource),
            "pass_line": f"RESOURCE INVARIANT GATE PASS: {resource}",
        },
        "support_matrix_contract": {
            "artifact_dir": str(root / "support-matrix"),
            "pass_line": f"SUPPORT MATRIX CONTRACT PASS: {root / 'support-matrix'}",
        },
    }
    release_candidate_required_gates = sorted({"unit"} | REQUIRED_RELEASE_CANDIDATE_STAGE_GATES)
    write_json(
        change / "gate_plan.json",
        {
            "schema_version": 1,
            "status": "pass",
            "head_sha": sha,
            "unknown_files": [],
            "required_gates": release_candidate_required_gates,
        },
    )
    write_json(
        change / "release_candidate_manifest.json",
        {
            "schema_version": 1,
            "head_sha": sha,
            "required_gates": release_candidate_required_gates,
            "invalidated_gates": [],
            "satisfied_gates": sorted(REQUIRED_RELEASE_CANDIDATE_STAGE_GATES),
            "artifact_paths": [
                artifact["artifact_dir"]
                for _, artifact in sorted(release_candidate_stage_artifacts.items())
            ],
            "pass_lines": [
                artifact["pass_line"]
                for _, artifact in sorted(release_candidate_stage_artifacts.items())
            ],
            "stale_artifacts": [],
        },
    )
    write_json(
        change / "planner_selfcheck.json",
        {
            "schema_version": 1,
            "status": "pass",
            "fixture_count": 3,
            "failures": [],
            "fixtures": [
                {
                    "id": "engine_shared_runtime",
                    "status": "pass",
                    "required_gates": ["unit", "resource_invariant", "metal_sentinel", "cuda_sentinel"],
                },
                {
                    "id": "cuda_kernel_local",
                    "status": "pass",
                    "required_gates": ["cuda_sentinel", "metal_boundary_smoke"],
                },
                {
                    "id": "metal_kernel_local",
                    "status": "pass",
                    "required_gates": ["metal_sentinel", "cuda_boundary_smoke"],
                },
            ],
        },
    )

    product = root / "product"
    product_scenarios = [
        {"name": "profile_run_serve", "type": "profile_artifact", "status": "pass"},
        {"name": "profile_replay_link", "type": "profile_replay_link", "status": "pass"},
        {"name": "resource_trace", "type": "resource_trace", "status": "pass"},
        {"name": "replay_normal", "type": "replay_bundle", "status": "pass"},
        {"name": "replay_bad_output", "type": "replay_bundle", "status": "pass"},
        {"name": "replay_oom_admission", "type": "replay_bundle", "status": "pass"},
        {"name": "replay_panic_error", "type": "replay_bundle", "status": "pass"},
        {"name": "serve_stream_done_once_pass", "type": "sse_fixture", "status": "pass"},
        {"name": "serve_stream_missing_done_fail", "type": "sse_fixture", "status": "pass"},
        {"name": "serve_stream_duplicate_done_fail", "type": "sse_fixture", "status": "pass"},
        {"name": "serve_stream_malformed_json_fail", "type": "sse_fixture", "status": "pass"},
        {"name": "native_op_manifest", "type": "native_op_manifest", "status": "pass"},
    ]
    make_gate_manifest(
        product,
        "PRODUCT BACKEND SENTINEL PASS",
        sha,
        "product_backend_sentinel_summary.json",
        {
            "schema_version": 1,
            "status": "pass",
            "gate": "product_backend_sentinel",
            "failed": 0,
            "scenario_count": len(product_scenarios),
            "required_stage2_fixture_count": 12,
            "scenarios": product_scenarios,
            "failure_links": [
                {
                    "failure_kind": "bad_output",
                    "profile_event_id": "evt-bad-output-blocker",
                    "request_id": "req-fixture",
                    "replay_command": "ferrum run synthetic/no-weight",
                    "bundle_dir": "fixtures/bad-output",
                    "failure_diagnostics": None,
                },
                {
                    "failure_kind": "oom_admission",
                    "profile_event_id": "evt-oom-admission-blocker",
                    "request_id": "req-fixture",
                    "replay_command": "ferrum run synthetic/no-weight",
                    "bundle_dir": "fixtures/oom-admission",
                    "failure_diagnostics": "fixtures/oom-admission/failure_diagnostics.json",
                },
                {
                    "failure_kind": "panic_error",
                    "profile_event_id": "evt-panic-error-blocker",
                    "request_id": "req-fixture",
                    "replay_command": "ferrum run synthetic/no-weight",
                    "bundle_dir": "fixtures/panic-error",
                    "failure_diagnostics": "fixtures/panic-error/failure_diagnostics.json",
                },
            ],
        },
    )

    model = root / "model"
    make_gate_manifest(
        model,
        "MODEL ONBOARDING CONTRACT PASS",
        sha,
        "model_onboarding_contract_summary.json",
        {
            "schema_version": 1,
            "status": "pass",
            "gate": "model_onboarding_contract",
            "contracts": [{"contract_id": "fixture", "status": "pass"}],
        },
    )

    vertical = root / "vertical"
    vertical_entrypoints: dict[str, dict[str, Any]] = {}
    for entrypoint in ("run", "serve"):
        entry_dir = vertical / entrypoint
        request_dump = entry_dir / "request_dump" / "request.json"
        profile = entry_dir / "profile.jsonl"
        entry_summary = entry_dir / "observability_profile_summary.json"
        request_dump.parent.mkdir(parents=True, exist_ok=True)
        profile.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_id": f"evt-{entrypoint}-complete",
                    "request_id": f"req-{entrypoint}-fixture",
                    "entrypoint": entrypoint,
                    "backend": "synthetic",
                    "phase": "request_complete",
                    "event_kind": "instant",
                    "timestamp": "2026-07-02T00:00:00Z",
                    "status": "ok",
                    "model": "synthetic/no-weight",
                    "replay": {
                        "command": f"ferrum {entrypoint} synthetic/no-weight",
                        "bundle_dir": f"request_dumps/req-{entrypoint}-fixture",
                    },
                    "attributes": {},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_json(
            request_dump,
            {
                "schema_version": SCHEMA_VERSION,
                "entrypoint": entrypoint,
                "request_id": f"req-{entrypoint}-fixture",
                "l0_only": True,
                "sanitized": True,
            },
        )
        write_json(
            entry_summary,
            {
                "schema_version": SCHEMA_VERSION,
                "entrypoint": entrypoint,
                "l0_only": True,
                "status": "pass",
            },
        )
        vertical_entrypoints[entrypoint] = {
            "entrypoint": entrypoint,
            "schema_version": SCHEMA_VERSION,
            "profile_jsonl": str(profile),
            "event_count": 1,
            "request_dump": str(request_dump),
            "replay_command": f"ferrum {entrypoint} synthetic/no-weight",
            "summary": str(entry_summary),
        }
    vertical_summary = {
        "schema_version": 1,
        "gate": "observability_vertical_slice",
        "status": "pass",
        "l0_only": True,
        "same_schema_version": True,
        "entrypoints": vertical_entrypoints,
        "analyzer": {
            "out": str(vertical / "analyzer"),
            "stdout": "FERRUM PROFILE ANALYZER PASS: fixture\n",
        },
    }
    write_json(vertical / "observability_profile_summary.json", vertical_summary)
    write_json(
        vertical / "observability_vertical_slice_manifest.json",
        {
            "schema_version": 1,
            "status": "pass",
            "artifact_dir": str(vertical),
            "pass_line": f"OBSERVABILITY VERTICAL SLICE PASS: {vertical}",
            "git_sha": sha,
            "git_dirty": False,
            "dirty_files": [],
            "summary": str(vertical / "observability_profile_summary.json"),
        },
    )

    observability = root / "observability"
    make_gate_manifest(
        observability,
        "OBSERVABILITY PROFILE GATE PASS",
        sha,
        "observability_profile_summary.json",
        {
            "schema_version": 1,
            "status": "pass",
            "gate": "observability_profile",
            "entrypoints": ["run", "serve"],
            "request_count": 2,
            "failed_count": 0,
            "corrupted_count": 0,
            "bad_text_count": 0,
            "oom_prevented_count": 0,
            "silent_oom_count": 0,
            "latency_p50_p95_p99": {
                "duration_us": {"p50": 10, "p95": 20, "p99": 30, "sample_count": 2}
            },
            "memory_high_water_bytes": {
                "max": 2048,
                "by_backend_scope": {"synthetic:process": 2048},
            },
            "resource_leak_count": 0,
            "top_slow_phases": [
                {
                    "event_id": "evt-serve-complete",
                    "request_id": "req-serve-fixture",
                    "entrypoint": "serve",
                    "backend": "synthetic",
                    "phase": "request_complete",
                    "duration_us": 30,
                }
            ],
            "first_failure_event": None,
            "replay_commands": [
                {
                    "event_id": "evt-run-complete",
                    "request_id": "req-run-fixture",
                    "command": "ferrum run synthetic/no-weight",
                    "bundle_dir": "fixtures/run-request-dump",
                },
                {
                    "event_id": "evt-serve-complete",
                    "request_id": "req-serve-fixture",
                    "command": "curl -s http://127.0.0.1:8000/v1/chat/completions",
                    "bundle_dir": "fixtures/serve-request-dump",
                },
            ],
            "vertical_slice_artifact": str(vertical),
            "actual_smoke_artifact": "fixture-l1-smoke",
        },
    )

    support_matrix = root / "support-matrix"
    make_gate_manifest(
        support_matrix,
        "SUPPORT MATRIX CONTRACT PASS",
        sha,
        "support_matrix_contract_summary.json",
        {
            "schema_version": 1,
            "status": "pass",
            "gate": "support_matrix_contract",
            "git_sha": sha,
            "rows": [
                {
                    "model_id": "fixture/qwen3-moe",
                    "contract_id": "fixture",
                    "claims": {"cuda": True, "metal": False, "int4_gptq": True},
                }
            ],
        },
    )

    native = root / "native"
    make_gate_manifest(
        native,
        "NATIVE OP ARTIFACT PASS",
        sha,
        "native_operator_artifact_summary.json",
        {
            "schema_version": 1,
            "status": "pass",
            "gate": "native_operator_artifact",
            "manifests": [
                {
                    "manifest": "fixtures/fa2_manifest.json",
                    "operator": "fa2",
                    "backend": "cuda",
                    "compute_capabilities": ["sm_89"],
                    "linkage": "static",
                    "binary_artifact": "fixtures/libferrum_native_fa2.a",
                    "binary_sha256": "c" * 64,
                    "resolution": {
                        "operator": "fa2",
                        "backend": "cuda",
                        "linkage": "static",
                        "binary_sha256": "c" * 64,
                    },
                }
            ],
            "selftest_summary": {
                "schema_version": 1,
                "status": "pass",
                "pass_fixtures": ["dummy_manifest.json", "fa2_manifest.json"],
                "fail_fixtures": ["missing_binary_sha256.json"],
                "resolver_fail_closed_cases": [
                    "abi_mismatch",
                    "binary_sha256_mismatch",
                    "compute_capability_mismatch",
                    "missing_manifest",
                    "operator_mismatch",
                ],
                "python_runtime_dependency": "none",
            },
            "bulk_source": {"count": 0, "samples": []},
            "unregistered_third_party_source": {"count": 0, "samples": []},
        },
    )

    for path in (
        root / "fixtures/metal-l2/request_dump",
        root / "fixtures/cuda-l2/request_dump",
    ):
        path.mkdir(parents=True, exist_ok=True)
    native_cuda_dir = root / "fixtures/cuda-native-op"
    native_cuda_dir.mkdir(parents=True, exist_ok=True)
    native_cuda_artifact = native_cuda_dir / "native_cuda.json"
    write_json(
        native_cuda_artifact,
        {
            "schema_version": 1,
            "status": "pass",
            "backend": "cuda",
            "git_sha": sha,
            "artifact_dir": str(native_cuda_dir),
            "pass_line": "NATIVE OP ARTIFACT PASS: fixtures/cuda-native-op",
        },
    )

    actual = root / "actual_model_regression_summary.json"
    write_json(
        actual,
        {
            "schema_version": 1,
            "status": "pass",
            "git_sha": sha,
            "pass_line": f"ACTUAL MODEL REGRESSION SUMMARY PASS: {root}",
            "metal_l2_artifact": {
                "status": "pass",
                "backend": "metal",
                "requested_backend": "metal",
                "effective_backend": "metal",
                "git_sha": sha,
                "git_dirty": False,
                "dirty_files": [],
                "artifact_dir": str(root / "fixtures/metal-l2"),
                "pass_line": "METAL L2 ACTUAL MODEL PASS: fixtures/metal-l2",
                "model_id": "fixture/metal-model",
                "architecture": "llama_dense",
                "entrypoints": ["run", "serve", "stream", "basic_concurrency"],
                "command": ["ferrum", "run", "fixture/metal-model", "--profile-detail", "basic"],
                "profile_detail": "basic",
                "replay_bundle_index": [
                    {
                        "request_id": "req-metal-fixture",
                        "entrypoint": "run",
                        "replay_command": "ferrum run fixture/metal-model",
                        "bundle_dir": str(root / "fixtures/metal-l2/request_dump"),
                    }
                ],
            },
            "cuda_l2_artifact": {
                "status": "pass",
                "backend": "cuda",
                "requested_backend": "cuda",
                "effective_backend": "cuda",
                "git_sha": sha,
                "git_dirty": False,
                "dirty_files": [],
                "artifact_dir": str(root / "fixtures/cuda-l2"),
                "pass_line": "CUDA L2 ACTUAL MODEL PASS: fixtures/cuda-l2",
                "model_id": "fixture/cuda-model",
                "architecture": "qwen3_moe",
                "entrypoints": ["run", "serve", "stream", "basic_concurrency"],
                "command": ["ferrum", "run", "fixture/cuda-model", "--profile-detail", "basic"],
                "profile_detail": "basic",
                "replay_bundle_index": [
                    {
                        "request_id": "req-cuda-fixture",
                        "entrypoint": "run",
                        "replay_command": "ferrum run fixture/cuda-model",
                        "bundle_dir": str(root / "fixtures/cuda-l2/request_dump"),
                    }
                ],
            },
            "native_operator_selection": {
                "status": "pass",
                "selected": True,
                "cuda_artifact": str(native_cuda_artifact),
                "cuda_artifact_dir": str(native_cuda_dir),
                "cuda_artifact_pass_line": "NATIVE OP ARTIFACT PASS: fixtures/cuda-native-op",
            },
        },
    )
    return {
        "resource": resource,
        "change": change,
        "product": product,
        "model": model,
        "support_matrix": support_matrix,
        "observability": observability,
        "native": native,
        "actual": actual,
    }


def run_selftest() -> dict[str, Any]:
    sha = head_sha()
    with tempfile.TemporaryDirectory(prefix="ferrum-hardening-goal-selftest-") as tmp:
        root = Path(tmp)
        artifacts = selftest_artifacts(root, sha)
        out = root / "out"
        args = argparse.Namespace(
            out=out,
            resource_invariant=artifacts["resource"],
            change_impact=artifacts["change"],
            product_sentinel=artifacts["product"],
            model_contract=artifacts["model"],
            support_matrix_contract=artifacts["support_matrix"],
            observability_profile=artifacts["observability"],
            native_operator=artifacts["native"],
            actual_model_regression_summary=artifacts["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        manifest = run_gate(args)
        require((out / "goal_manifest.json").is_file(), "selftest missing goal_manifest.json")
        bad_resource_coverage = root / "bad-resource-coverage"
        artifacts_bad_resource_coverage = selftest_artifacts(
            root / "bad-resource-coverage-fixtures",
            sha,
        )
        bad_resource_report_path = artifacts_bad_resource_coverage["resource"] / "invariant_report.json"
        bad_resource_report = read_json(bad_resource_report_path)
        bad_resource_report["fixture_summary"]["required_scenarios"] = [
            scenario
            for scenario in bad_resource_report["fixture_summary"]["required_scenarios"]
            if scenario != "oom_prevented_by_admission"
        ]
        bad_resource_report["trace"]["scenarios"] = [
            scenario
            for scenario in bad_resource_report["trace"]["scenarios"]
            if scenario != "oom_prevented_by_admission"
        ]
        bad_resource_report["fixture_summary"]["scenario_count"] = len(
            bad_resource_report["fixture_summary"]["required_scenarios"]
        )
        write_json(bad_resource_report_path, bad_resource_report)
        args_bad_resource_coverage = argparse.Namespace(
            out=bad_resource_coverage,
            resource_invariant=artifacts_bad_resource_coverage["resource"],
            change_impact=artifacts_bad_resource_coverage["change"],
            product_sentinel=artifacts_bad_resource_coverage["product"],
            model_contract=artifacts_bad_resource_coverage["model"],
            support_matrix_contract=artifacts_bad_resource_coverage["support_matrix"],
            observability_profile=artifacts_bad_resource_coverage["observability"],
            native_operator=artifacts_bad_resource_coverage["native"],
            actual_model_regression_summary=artifacts_bad_resource_coverage["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_resource_coverage)
            raise AssertionError("missing resource invariant scenario unexpectedly passed final gate")
        except GoalGateError as exc:
            require(
                "resource_invariant.summary.fixture_summary" in str(exc),
                f"unexpected resource coverage error: {exc}",
            )
        bad_native = root / "bad-native"
        # Reuse a full artifact set but mutate native source inventory to prove fail-closed behavior.
        artifacts_bad = selftest_artifacts(root / "bad", sha)
        bad_summary = read_json(artifacts_bad["native"] / "native_operator_artifact_summary.json")
        bad_summary["bulk_source"]["count"] = 1
        write_json(artifacts_bad["native"] / "native_operator_artifact_summary.json", bad_summary)
        args_bad = argparse.Namespace(
            out=bad_native,
            resource_invariant=artifacts_bad["resource"],
            change_impact=artifacts_bad["change"],
            product_sentinel=artifacts_bad["product"],
            model_contract=artifacts_bad["model"],
            support_matrix_contract=artifacts_bad["support_matrix"],
            observability_profile=artifacts_bad["observability"],
            native_operator=artifacts_bad["native"],
            actual_model_regression_summary=artifacts_bad["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad)
            raise AssertionError("bulk source regression unexpectedly passed final gate")
        except GoalGateError as exc:
            require("bulk_source.count" in str(exc), f"unexpected bulk source error: {exc}")
        bad_native_selftest = root / "bad-native-selftest"
        artifacts_bad_native_selftest = selftest_artifacts(
            root / "bad-native-selftest-fixtures",
            sha,
        )
        bad_native_selftest_summary_path = (
            artifacts_bad_native_selftest["native"] / "native_operator_artifact_summary.json"
        )
        bad_native_selftest_summary = read_json(bad_native_selftest_summary_path)
        bad_native_selftest_summary["selftest_summary"]["pass_fixtures"] = ["fa2_manifest.json"]
        write_json(bad_native_selftest_summary_path, bad_native_selftest_summary)
        args_bad_native_selftest = argparse.Namespace(
            out=bad_native_selftest,
            resource_invariant=artifacts_bad_native_selftest["resource"],
            change_impact=artifacts_bad_native_selftest["change"],
            product_sentinel=artifacts_bad_native_selftest["product"],
            model_contract=artifacts_bad_native_selftest["model"],
            support_matrix_contract=artifacts_bad_native_selftest["support_matrix"],
            observability_profile=artifacts_bad_native_selftest["observability"],
            native_operator=artifacts_bad_native_selftest["native"],
            actual_model_regression_summary=artifacts_bad_native_selftest["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_native_selftest)
            raise AssertionError("missing native-op dummy fixture unexpectedly passed final gate")
        except GoalGateError as exc:
            require(
                "native_operator.summary.selftest_summary.pass_fixtures" in str(exc),
                f"unexpected native-op selftest error: {exc}",
            )
        bad_product = root / "bad-product"
        artifacts_bad_product = selftest_artifacts(root / "bad-product-fixtures", sha)
        bad_product_summary = read_json(
            artifacts_bad_product["product"] / "product_backend_sentinel_summary.json"
        )
        bad_product_summary["failure_links"] = []
        write_json(
            artifacts_bad_product["product"] / "product_backend_sentinel_summary.json",
            bad_product_summary,
        )
        args_bad_product = argparse.Namespace(
            out=bad_product,
            resource_invariant=artifacts_bad_product["resource"],
            change_impact=artifacts_bad_product["change"],
            product_sentinel=artifacts_bad_product["product"],
            model_contract=artifacts_bad_product["model"],
            support_matrix_contract=artifacts_bad_product["support_matrix"],
            observability_profile=artifacts_bad_product["observability"],
            native_operator=artifacts_bad_product["native"],
            actual_model_regression_summary=artifacts_bad_product["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_product)
            raise AssertionError("missing product failure links unexpectedly passed final gate")
        except GoalGateError as exc:
            require("failure_links missing" in str(exc), f"unexpected product sentinel error: {exc}")
        bad_product_scenario = root / "bad-product-scenario"
        artifacts_bad_product_scenario = selftest_artifacts(
            root / "bad-product-scenario-fixtures",
            sha,
        )
        bad_product_scenario_summary_path = (
            artifacts_bad_product_scenario["product"] / "product_backend_sentinel_summary.json"
        )
        bad_product_scenario_summary = read_json(bad_product_scenario_summary_path)
        bad_product_scenario_summary["scenarios"][0]["status"] = "fail"
        write_json(bad_product_scenario_summary_path, bad_product_scenario_summary)
        args_bad_product_scenario = argparse.Namespace(
            out=bad_product_scenario,
            resource_invariant=artifacts_bad_product_scenario["resource"],
            change_impact=artifacts_bad_product_scenario["change"],
            product_sentinel=artifacts_bad_product_scenario["product"],
            model_contract=artifacts_bad_product_scenario["model"],
            support_matrix_contract=artifacts_bad_product_scenario["support_matrix"],
            observability_profile=artifacts_bad_product_scenario["observability"],
            native_operator=artifacts_bad_product_scenario["native"],
            actual_model_regression_summary=artifacts_bad_product_scenario["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_product_scenario)
            raise AssertionError("failed product sentinel scenario unexpectedly passed final gate")
        except GoalGateError as exc:
            require(
                "product_sentinel.summary.scenarios[0].status" in str(exc),
                f"unexpected product scenario error: {exc}",
            )
        bad_support_contract_link = root / "bad-support-contract-link"
        artifacts_bad_support_contract_link = selftest_artifacts(
            root / "bad-support-contract-link-fixtures",
            sha,
        )
        bad_support_summary_path = (
            artifacts_bad_support_contract_link["support_matrix"]
            / "support_matrix_contract_summary.json"
        )
        bad_support_summary = read_json(bad_support_summary_path)
        bad_support_summary["rows"][0]["contract_id"] = "missing-contract"
        write_json(bad_support_summary_path, bad_support_summary)
        args_bad_support_contract_link = argparse.Namespace(
            out=bad_support_contract_link,
            resource_invariant=artifacts_bad_support_contract_link["resource"],
            change_impact=artifacts_bad_support_contract_link["change"],
            product_sentinel=artifacts_bad_support_contract_link["product"],
            model_contract=artifacts_bad_support_contract_link["model"],
            support_matrix_contract=artifacts_bad_support_contract_link["support_matrix"],
            observability_profile=artifacts_bad_support_contract_link["observability"],
            native_operator=artifacts_bad_support_contract_link["native"],
            actual_model_regression_summary=artifacts_bad_support_contract_link["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_support_contract_link)
            raise AssertionError("support matrix row with missing contract unexpectedly passed final gate")
        except GoalGateError as exc:
            require(
                "references contracts missing" in str(exc),
                f"unexpected support/model contract linkage error: {exc}",
            )
        bad_dirty_stage = root / "bad-dirty-stage"
        artifacts_bad_dirty_stage = selftest_artifacts(root / "bad-dirty-stage-fixtures", sha)
        bad_dirty_manifest_path = artifacts_bad_dirty_stage["product"] / "gate.manifest.json"
        bad_dirty_manifest = read_json(bad_dirty_manifest_path)
        bad_dirty_manifest["git_dirty"] = True
        bad_dirty_manifest["dirty_files"] = [" M scripts/release/product_backend_sentinel_gate.py"]
        write_json(bad_dirty_manifest_path, bad_dirty_manifest)
        args_bad_dirty_stage = argparse.Namespace(
            out=bad_dirty_stage,
            resource_invariant=artifacts_bad_dirty_stage["resource"],
            change_impact=artifacts_bad_dirty_stage["change"],
            product_sentinel=artifacts_bad_dirty_stage["product"],
            model_contract=artifacts_bad_dirty_stage["model"],
            support_matrix_contract=artifacts_bad_dirty_stage["support_matrix"],
            observability_profile=artifacts_bad_dirty_stage["observability"],
            native_operator=artifacts_bad_dirty_stage["native"],
            actual_model_regression_summary=artifacts_bad_dirty_stage["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_dirty_stage)
            raise AssertionError("dirty stage artifact unexpectedly passed final gate")
        except GoalGateError as exc:
            require("git_dirty" in str(exc), f"unexpected dirty stage error: {exc}")
        bad_dirty_vertical = root / "bad-dirty-vertical"
        artifacts_bad_dirty_vertical = selftest_artifacts(root / "bad-dirty-vertical-fixtures", sha)
        bad_dirty_observability_summary = read_json(
            artifacts_bad_dirty_vertical["observability"] / "observability_profile_summary.json"
        )
        bad_dirty_vertical_manifest_path = (
            Path(bad_dirty_observability_summary["vertical_slice_artifact"])
            / "observability_vertical_slice_manifest.json"
        )
        bad_dirty_vertical_manifest = read_json(bad_dirty_vertical_manifest_path)
        bad_dirty_vertical_manifest["git_dirty"] = True
        bad_dirty_vertical_manifest["dirty_files"] = [
            " M scripts/release/observability_vertical_slice_gate.py"
        ]
        write_json(bad_dirty_vertical_manifest_path, bad_dirty_vertical_manifest)
        args_bad_dirty_vertical = argparse.Namespace(
            out=bad_dirty_vertical,
            resource_invariant=artifacts_bad_dirty_vertical["resource"],
            change_impact=artifacts_bad_dirty_vertical["change"],
            product_sentinel=artifacts_bad_dirty_vertical["product"],
            model_contract=artifacts_bad_dirty_vertical["model"],
            support_matrix_contract=artifacts_bad_dirty_vertical["support_matrix"],
            observability_profile=artifacts_bad_dirty_vertical["observability"],
            native_operator=artifacts_bad_dirty_vertical["native"],
            actual_model_regression_summary=artifacts_bad_dirty_vertical["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_dirty_vertical)
            raise AssertionError("dirty observability vertical slice unexpectedly passed final gate")
        except GoalGateError as exc:
            require(
                "observability_vertical_slice.manifest.git_dirty" in str(exc),
                f"unexpected dirty vertical slice error: {exc}",
            )
        bad_observability_shape = root / "bad-observability-shape"
        artifacts_bad_observability_shape = selftest_artifacts(
            root / "bad-observability-shape-fixtures",
            sha,
        )
        bad_observability_shape_summary_path = (
            artifacts_bad_observability_shape["observability"] / "observability_profile_summary.json"
        )
        bad_observability_shape_summary = read_json(bad_observability_shape_summary_path)
        bad_observability_shape_summary["latency_p50_p95_p99"]["duration_us"][
            "sample_count"
        ] = 0
        write_json(bad_observability_shape_summary_path, bad_observability_shape_summary)
        args_bad_observability_shape = argparse.Namespace(
            out=bad_observability_shape,
            resource_invariant=artifacts_bad_observability_shape["resource"],
            change_impact=artifacts_bad_observability_shape["change"],
            product_sentinel=artifacts_bad_observability_shape["product"],
            model_contract=artifacts_bad_observability_shape["model"],
            support_matrix_contract=artifacts_bad_observability_shape["support_matrix"],
            observability_profile=artifacts_bad_observability_shape["observability"],
            native_operator=artifacts_bad_observability_shape["native"],
            actual_model_regression_summary=artifacts_bad_observability_shape["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_observability_shape)
            raise AssertionError("observability summary with no latency samples unexpectedly passed final gate")
        except GoalGateError as exc:
            require(
                "latency_p50_p95_p99.duration_us.sample_count" in str(exc),
                f"unexpected observability shape error: {exc}",
            )
        bad_invalidated_counted = root / "bad-invalidated-counted"
        artifacts_bad_invalidated_counted = selftest_artifacts(
            root / "bad-invalidated-counted-fixtures",
            sha,
        )
        bad_release_candidate_path = (
            artifacts_bad_invalidated_counted["change"] / "release_candidate_manifest.json"
        )
        bad_release_candidate = read_json(bad_release_candidate_path)
        bad_release_candidate["invalidated_gates"] = ["product_sentinel"]
        bad_release_candidate["satisfied_gates"] = ["product_sentinel"]
        write_json(bad_release_candidate_path, bad_release_candidate)
        args_bad_invalidated_counted = argparse.Namespace(
            out=bad_invalidated_counted,
            resource_invariant=artifacts_bad_invalidated_counted["resource"],
            change_impact=artifacts_bad_invalidated_counted["change"],
            product_sentinel=artifacts_bad_invalidated_counted["product"],
            model_contract=artifacts_bad_invalidated_counted["model"],
            support_matrix_contract=artifacts_bad_invalidated_counted["support_matrix"],
            observability_profile=artifacts_bad_invalidated_counted["observability"],
            native_operator=artifacts_bad_invalidated_counted["native"],
            actual_model_regression_summary=artifacts_bad_invalidated_counted["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_invalidated_counted)
            raise AssertionError("invalidated-but-satisfied gate unexpectedly passed final gate")
        except GoalGateError as exc:
            require("invalidated" in str(exc), f"unexpected invalidated gate error: {exc}")
        bad_stale_counted = root / "bad-stale-counted"
        artifacts_bad_stale_counted = selftest_artifacts(root / "bad-stale-counted-fixtures", sha)
        bad_stale_release_candidate_path = (
            artifacts_bad_stale_counted["change"] / "release_candidate_manifest.json"
        )
        bad_stale_release_candidate = read_json(bad_stale_release_candidate_path)
        stale_artifact = {
            "id": "old-product-sentinel",
            "gate": "product_sentinel",
            "artifact_dir": "docs/release/old-product-sentinel",
            "pass_line": "PRODUCT BACKEND SENTINEL PASS: docs/release/old-product-sentinel",
        }
        bad_stale_release_candidate["stale_artifacts"] = [stale_artifact]
        bad_stale_release_candidate["artifact_paths"] = [stale_artifact["artifact_dir"]]
        bad_stale_release_candidate["pass_lines"] = [stale_artifact["pass_line"]]
        write_json(bad_stale_release_candidate_path, bad_stale_release_candidate)
        args_bad_stale_counted = argparse.Namespace(
            out=bad_stale_counted,
            resource_invariant=artifacts_bad_stale_counted["resource"],
            change_impact=artifacts_bad_stale_counted["change"],
            product_sentinel=artifacts_bad_stale_counted["product"],
            model_contract=artifacts_bad_stale_counted["model"],
            support_matrix_contract=artifacts_bad_stale_counted["support_matrix"],
            observability_profile=artifacts_bad_stale_counted["observability"],
            native_operator=artifacts_bad_stale_counted["native"],
            actual_model_regression_summary=artifacts_bad_stale_counted["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_stale_counted)
            raise AssertionError("stale artifact counted as pass unexpectedly passed final gate")
        except GoalGateError as exc:
            require("stale artifacts" in str(exc), f"unexpected stale-counted error: {exc}")
        bad_release_candidate_missing_stage = root / "bad-release-candidate-missing-stage"
        artifacts_bad_release_candidate_missing_stage = selftest_artifacts(
            root / "bad-release-candidate-missing-stage-fixtures",
            sha,
        )
        bad_release_candidate_missing_stage_path = (
            artifacts_bad_release_candidate_missing_stage["change"] / "release_candidate_manifest.json"
        )
        bad_release_candidate_missing_stage_manifest = read_json(
            bad_release_candidate_missing_stage_path
        )
        bad_release_candidate_missing_stage_manifest["satisfied_gates"] = [
            gate
            for gate in bad_release_candidate_missing_stage_manifest["satisfied_gates"]
            if gate != "observability_profile"
        ]
        write_json(
            bad_release_candidate_missing_stage_path,
            bad_release_candidate_missing_stage_manifest,
        )
        args_bad_release_candidate_missing_stage = argparse.Namespace(
            out=bad_release_candidate_missing_stage,
            resource_invariant=artifacts_bad_release_candidate_missing_stage["resource"],
            change_impact=artifacts_bad_release_candidate_missing_stage["change"],
            product_sentinel=artifacts_bad_release_candidate_missing_stage["product"],
            model_contract=artifacts_bad_release_candidate_missing_stage["model"],
            support_matrix_contract=artifacts_bad_release_candidate_missing_stage["support_matrix"],
            observability_profile=artifacts_bad_release_candidate_missing_stage["observability"],
            native_operator=artifacts_bad_release_candidate_missing_stage["native"],
            actual_model_regression_summary=artifacts_bad_release_candidate_missing_stage["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_release_candidate_missing_stage)
            raise AssertionError("release candidate missing final stage unexpectedly passed final gate")
        except GoalGateError as exc:
            require(
                "release_candidate_manifest.satisfied_gates missing final gates" in str(exc),
                f"unexpected release candidate stage evidence error: {exc}",
            )
        bad_planner_selfcheck = root / "bad-planner-selfcheck"
        artifacts_bad_planner_selfcheck = selftest_artifacts(
            root / "bad-planner-selfcheck-fixtures",
            sha,
        )
        bad_planner_selfcheck_path = (
            artifacts_bad_planner_selfcheck["change"] / "planner_selfcheck.json"
        )
        bad_planner_selfcheck_data = read_json(bad_planner_selfcheck_path)
        bad_planner_selfcheck_data["fixtures"] = [
            fixture
            for fixture in bad_planner_selfcheck_data["fixtures"]
            if fixture["id"] != "engine_shared_runtime"
        ]
        write_json(bad_planner_selfcheck_path, bad_planner_selfcheck_data)
        args_bad_planner_selfcheck = argparse.Namespace(
            out=bad_planner_selfcheck,
            resource_invariant=artifacts_bad_planner_selfcheck["resource"],
            change_impact=artifacts_bad_planner_selfcheck["change"],
            product_sentinel=artifacts_bad_planner_selfcheck["product"],
            model_contract=artifacts_bad_planner_selfcheck["model"],
            support_matrix_contract=artifacts_bad_planner_selfcheck["support_matrix"],
            observability_profile=artifacts_bad_planner_selfcheck["observability"],
            native_operator=artifacts_bad_planner_selfcheck["native"],
            actual_model_regression_summary=artifacts_bad_planner_selfcheck["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_planner_selfcheck)
            raise AssertionError("missing planner shared-runtime fixture unexpectedly passed final gate")
        except GoalGateError as exc:
            require("planner_selfcheck" in str(exc), f"unexpected planner selfcheck error: {exc}")
        missing_actual = argparse.Namespace(
            **{
                **args.__dict__,
                "out": root / "missing-actual",
                "actual_model_regression_summary": None,
            }
        )
        try:
            run_gate(missing_actual)
            raise AssertionError("missing actual model regression unexpectedly passed final gate")
        except GoalGateError as exc:
            require("actual model regression summary" in str(exc), f"unexpected missing actual error: {exc}")
        bad_dirty = root / "bad-dirty"
        artifacts_bad_dirty = selftest_artifacts(root / "bad-dirty-fixtures", sha)
        bad_actual = read_json(artifacts_bad_dirty["actual"])
        bad_actual["metal_l2_artifact"]["git_dirty"] = True
        bad_actual["metal_l2_artifact"]["dirty_files"] = [
            " M crates/ferrum-cli/src/commands/run.rs"
        ]
        write_json(artifacts_bad_dirty["actual"], bad_actual)
        args_bad_dirty = argparse.Namespace(
            out=bad_dirty,
            resource_invariant=artifacts_bad_dirty["resource"],
            change_impact=artifacts_bad_dirty["change"],
            product_sentinel=artifacts_bad_dirty["product"],
            model_contract=artifacts_bad_dirty["model"],
            support_matrix_contract=artifacts_bad_dirty["support_matrix"],
            observability_profile=artifacts_bad_dirty["observability"],
            native_operator=artifacts_bad_dirty["native"],
            actual_model_regression_summary=artifacts_bad_dirty["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_dirty)
            raise AssertionError("dirty actual-model artifact unexpectedly passed final gate")
        except GoalGateError as exc:
            require("git_dirty" in str(exc), f"unexpected dirty actual-model error: {exc}")
        bad_architecture = root / "bad-architecture"
        artifacts_bad_architecture = selftest_artifacts(root / "bad-architecture-fixtures", sha)
        bad_architecture_actual = read_json(artifacts_bad_architecture["actual"])
        bad_architecture_actual["cuda_l2_artifact"]["architecture"] = "unknown_architecture"
        write_json(artifacts_bad_architecture["actual"], bad_architecture_actual)
        args_bad_architecture = argparse.Namespace(
            out=bad_architecture,
            resource_invariant=artifacts_bad_architecture["resource"],
            change_impact=artifacts_bad_architecture["change"],
            product_sentinel=artifacts_bad_architecture["product"],
            model_contract=artifacts_bad_architecture["model"],
            support_matrix_contract=artifacts_bad_architecture["support_matrix"],
            observability_profile=artifacts_bad_architecture["observability"],
            native_operator=artifacts_bad_architecture["native"],
            actual_model_regression_summary=artifacts_bad_architecture["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_architecture)
            raise AssertionError("unknown actual-model L2 architecture unexpectedly passed final gate")
        except GoalGateError as exc:
            require("architecture" in str(exc), f"unexpected actual-model architecture error: {exc}")
        bad_backend = root / "bad-backend"
        artifacts_bad_backend = selftest_artifacts(root / "bad-backend-fixtures", sha)
        bad_backend_actual = read_json(artifacts_bad_backend["actual"])
        bad_backend_actual["metal_l2_artifact"]["requested_backend"] = "metal"
        bad_backend_actual["metal_l2_artifact"]["effective_backend"] = "cpu"
        write_json(artifacts_bad_backend["actual"], bad_backend_actual)
        args_bad_backend = argparse.Namespace(
            out=bad_backend,
            resource_invariant=artifacts_bad_backend["resource"],
            change_impact=artifacts_bad_backend["change"],
            product_sentinel=artifacts_bad_backend["product"],
            model_contract=artifacts_bad_backend["model"],
            support_matrix_contract=artifacts_bad_backend["support_matrix"],
            observability_profile=artifacts_bad_backend["observability"],
            native_operator=artifacts_bad_backend["native"],
            actual_model_regression_summary=artifacts_bad_backend["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_backend)
            raise AssertionError("backend fallback actual-model artifact unexpectedly passed final gate")
        except GoalGateError as exc:
            require("effective_backend" in str(exc), f"unexpected backend mismatch error: {exc}")
        bad_artifact_dir = root / "bad-artifact-dir"
        artifacts_bad_artifact_dir = selftest_artifacts(root / "bad-artifact-dir-fixtures", sha)
        bad_artifact_dir_actual = read_json(artifacts_bad_artifact_dir["actual"])
        bad_artifact_dir_actual["metal_l2_artifact"]["artifact_dir"] = str(
            root / "missing-metal-l2-artifact-dir"
        )
        write_json(artifacts_bad_artifact_dir["actual"], bad_artifact_dir_actual)
        args_bad_artifact_dir = argparse.Namespace(
            out=bad_artifact_dir,
            resource_invariant=artifacts_bad_artifact_dir["resource"],
            change_impact=artifacts_bad_artifact_dir["change"],
            product_sentinel=artifacts_bad_artifact_dir["product"],
            model_contract=artifacts_bad_artifact_dir["model"],
            support_matrix_contract=artifacts_bad_artifact_dir["support_matrix"],
            observability_profile=artifacts_bad_artifact_dir["observability"],
            native_operator=artifacts_bad_artifact_dir["native"],
            actual_model_regression_summary=artifacts_bad_artifact_dir["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_artifact_dir)
            raise AssertionError("missing actual-model artifact_dir unexpectedly passed final gate")
        except GoalGateError as exc:
            require("artifact_dir" in str(exc), f"unexpected artifact_dir error: {exc}")
        bad_replay_bundle = root / "bad-replay-bundle"
        artifacts_bad_replay_bundle = selftest_artifacts(
            root / "bad-replay-bundle-fixtures",
            sha,
        )
        bad_replay_bundle_actual = read_json(artifacts_bad_replay_bundle["actual"])
        bad_replay_bundle_actual["metal_l2_artifact"]["replay_bundle_index"][0][
            "bundle_dir"
        ] = str(root / "missing-replay-bundle")
        write_json(artifacts_bad_replay_bundle["actual"], bad_replay_bundle_actual)
        args_bad_replay_bundle = argparse.Namespace(
            out=bad_replay_bundle,
            resource_invariant=artifacts_bad_replay_bundle["resource"],
            change_impact=artifacts_bad_replay_bundle["change"],
            product_sentinel=artifacts_bad_replay_bundle["product"],
            model_contract=artifacts_bad_replay_bundle["model"],
            support_matrix_contract=artifacts_bad_replay_bundle["support_matrix"],
            observability_profile=artifacts_bad_replay_bundle["observability"],
            native_operator=artifacts_bad_replay_bundle["native"],
            actual_model_regression_summary=artifacts_bad_replay_bundle["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_replay_bundle)
            raise AssertionError("missing L2 replay bundle unexpectedly passed final gate")
        except GoalGateError as exc:
            require("bundle_dir" in str(exc), f"unexpected replay bundle error: {exc}")
        bad_native_artifact = root / "bad-native-selection-artifact"
        artifacts_bad_native_artifact = selftest_artifacts(
            root / "bad-native-selection-artifact-fixtures",
            sha,
        )
        bad_native_actual = read_json(artifacts_bad_native_artifact["actual"])
        bad_native_actual["native_operator_selection"]["cuda_artifact"] = str(
            root / "missing-native-cuda-artifact.json"
        )
        write_json(artifacts_bad_native_artifact["actual"], bad_native_actual)
        args_bad_native_artifact = argparse.Namespace(
            out=bad_native_artifact,
            resource_invariant=artifacts_bad_native_artifact["resource"],
            change_impact=artifacts_bad_native_artifact["change"],
            product_sentinel=artifacts_bad_native_artifact["product"],
            model_contract=artifacts_bad_native_artifact["model"],
            support_matrix_contract=artifacts_bad_native_artifact["support_matrix"],
            observability_profile=artifacts_bad_native_artifact["observability"],
            native_operator=artifacts_bad_native_artifact["native"],
            actual_model_regression_summary=artifacts_bad_native_artifact["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        try:
            run_gate(args_bad_native_artifact)
            raise AssertionError("missing selected native CUDA artifact unexpectedly passed final gate")
        except GoalGateError as exc:
            require("cuda_artifact" in str(exc), f"unexpected native CUDA artifact error: {exc}")
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "goal_manifest_keys": sorted(manifest.keys()),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resource-invariant", type=Path)
    parser.add_argument("--change-impact", type=Path)
    parser.add_argument("--product-sentinel", type=Path)
    parser.add_argument("--model-contract", type=Path)
    parser.add_argument("--support-matrix-contract", type=Path)
    parser.add_argument("--observability-profile", type=Path)
    parser.add_argument("--native-operator", type=Path)
    parser.add_argument("--actual-model-regression-summary", type=Path)
    parser.add_argument("--binary-sha256")
    parser.add_argument("--require-clean", action="store_true")
    return parser.parse_args()


def require_normal_args(args: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in [
            ("--out", args.out),
            ("--resource-invariant", args.resource_invariant),
            ("--change-impact", args.change_impact),
            ("--product-sentinel", args.product_sentinel),
            ("--model-contract", args.model_contract),
            ("--support-matrix-contract", args.support_matrix_contract),
            ("--observability-profile", args.observability_profile),
            ("--native-operator", args.native_operator),
        ]
        if value is None
    ]
    if missing:
        raise GoalGateError(f"missing required args: {', '.join(missing)}")


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_selftest()
            print(SELFTEST_PASS_LINE)
            return 0
        require_normal_args(args)
        manifest = run_gate(args)
        print(manifest["pass_line"])
        return 0
    except GoalGateError as exc:
        print(f"RELEASE_REGRESSION_HARDENING GOAL FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
