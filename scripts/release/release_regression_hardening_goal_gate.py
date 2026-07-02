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


def load_stage_manifest(root: Path, label: str, pass_prefix: str, expected_sha: str) -> dict[str, Any]:
    manifest = read_json(root / "gate.manifest.json")
    require_status_pass(manifest, f"{label}.manifest")
    pass_line = require_pass_line(manifest, f"{label}.manifest", pass_prefix)
    require_git_sha(manifest, f"{label}.manifest", expected_sha)
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


def validate_resource_invariant(root: Path, expected_sha: str) -> dict[str, Any]:
    stage = load_stage_manifest(root, "resource_invariant", "RESOURCE INVARIANT GATE PASS", expected_sha)
    summary = read_json(root / "invariant_report.json")
    require_status_pass(summary, "resource_invariant.summary")
    for key in ("leaked_resources", "underflow_count", "silent_oom_count", "panic_count"):
        require(summary.get(key) == 0, f"resource_invariant.summary.{key} must be 0")
    stage["summary"] = {**summary, "_summary_path": str(root / "invariant_report.json")}
    return stage


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
    pass_line = f"CHANGE IMPACT GATE PLAN PASS: {root}"
    return {
        "label": "change_impact",
        "artifact_dir": str(root),
        "manifest_path": None,
        "pass_line": pass_line,
        "git_sha": gate_plan.get("head_sha"),
        "gate_plan": gate_plan,
        "release_candidate_manifest": release_candidate,
    }


def validate_product_sentinel(root: Path, expected_sha: str) -> dict[str, Any]:
    stage = load_stage_manifest(root, "product_sentinel", "PRODUCT BACKEND SENTINEL PASS", expected_sha)
    summary = load_summary_from_manifest(stage, "product_backend_sentinel_summary.json")
    require(summary.get("failed") == 0, "product_sentinel.summary.failed must be 0")
    required = summary.get("required_stage2_fixture_count")
    require(isinstance(required, int) and required >= 12, "product_sentinel.summary must cover >=12 stage-2 fixtures")
    stage["summary"] = summary
    return stage


def validate_model_contract(root: Path, expected_sha: str) -> dict[str, Any]:
    stage = load_stage_manifest(root, "model_contract", "MODEL ONBOARDING CONTRACT PASS", expected_sha)
    summary = load_summary_from_manifest(stage, "model_onboarding_contract_summary.json")
    contracts = summary.get("contracts")
    require(isinstance(contracts, list) and contracts, "model_contract.summary.contracts must be non-empty")
    for index, contract in enumerate(contracts):
        require(isinstance(contract, dict), f"model_contract.summary.contracts[{index}] must be an object")
        require(contract.get("status") == "pass", f"model_contract.summary.contracts[{index}].status must be pass")
    stage["summary"] = summary
    return stage


def load_vertical_slice(path_value: Any, observability_root: Path) -> dict[str, Any]:
    require(isinstance(path_value, str) and path_value.strip(), "observability_profile.summary.vertical_slice_artifact is required")
    path = resolve_path(path_value, base=observability_root)
    if path.is_dir():
        manifest_path = path / "observability_vertical_slice_manifest.json"
        manifest = read_json(manifest_path)
        require_status_pass(manifest, "observability_vertical_slice.manifest")
        require_pass_line(manifest, "observability_vertical_slice.manifest", "OBSERVABILITY VERTICAL SLICE PASS")
        summary_path = resolve_path(str(manifest.get("summary")), base=path) if manifest.get("summary") else path / "observability_profile_summary.json"
        summary = read_json(summary_path)
        require_status_pass(summary, "observability_vertical_slice.summary")
        return {"manifest": manifest, "summary": summary, "summary_path": str(summary_path)}
    summary = read_json(path)
    require_status_pass(summary, "observability_vertical_slice.summary")
    return {"manifest": None, "summary": summary, "summary_path": str(path)}


def validate_observability_profile(root: Path, expected_sha: str) -> dict[str, Any]:
    stage = load_stage_manifest(root, "observability_profile", "OBSERVABILITY PROFILE GATE PASS", expected_sha)
    summary = load_summary_from_manifest(stage, "observability_profile_summary.json")
    missing = sorted(OBSERVABILITY_SUMMARY_FIELDS - set(summary))
    require(not missing, f"observability_profile.summary missing fields: {missing}")
    require(summary.get("bad_text_count") == 0, "observability_profile.summary.bad_text_count must be 0")
    require(summary.get("silent_oom_count") == 0, "observability_profile.summary.silent_oom_count must be 0")
    require(summary.get("resource_leak_count") == 0, "observability_profile.summary.resource_leak_count must be 0")
    entrypoints = set(summary.get("entrypoints") or [])
    require({"run", "serve"} <= entrypoints, "observability_profile.summary must include run and serve entrypoints")
    stage["summary"] = summary
    stage["vertical_slice"] = load_vertical_slice(summary.get("vertical_slice_artifact"), root)
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
    require(isinstance(summary.get("manifests"), list) and summary["manifests"], "native_operator.summary.manifests must be non-empty")
    stage["summary"] = summary
    stage["fa2_source_removal_inventory"] = {
        "bulk_source": bulk,
        "unregistered_third_party_source": third_party,
    }
    return stage


def artifact_git_sha(artifact: dict[str, Any], label: str, expected_sha: str) -> None:
    require(artifact.get("status") == "pass", f"{label}.status must be pass")
    require_git_sha(artifact, label, expected_sha)


def validate_l2_artifact(data: dict[str, Any], key: str, backend: str, expected_sha: str) -> None:
    artifact = data.get(key)
    require(isinstance(artifact, dict), f"actual_model_regression.{key} must be an object")
    artifact_git_sha(artifact, f"actual_model_regression.{key}", expected_sha)
    require(artifact.get("backend") == backend, f"actual_model_regression.{key}.backend must be {backend}")
    entrypoints = set(artifact.get("entrypoints") or [])
    missing = sorted(REQUIRED_L2_ENTRYPOINTS - entrypoints)
    require(not missing, f"actual_model_regression.{key}.entrypoints missing {missing}")
    artifact_dir = artifact.get("artifact_dir")
    require(isinstance(artifact_dir, str) and artifact_dir.strip(), f"actual_model_regression.{key}.artifact_dir must be non-empty")


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
    validate_l2_artifact(summary, "metal_l2_artifact", "metal", expected_sha)
    validate_l2_artifact(summary, "cuda_l2_artifact", "cuda", expected_sha)
    selection = summary.get("native_operator_selection")
    require(isinstance(selection, dict), "actual_model_regression.native_operator_selection must be an object")
    require(selection.get("status") == "pass", "actual_model_regression.native_operator_selection.status must be pass")
    selected = selection.get("selected")
    if selected is True:
        require(isinstance(selection.get("cuda_artifact"), str) and selection["cuda_artifact"].strip(), "native operator selected path requires cuda_artifact")
    elif selected is False:
        require(isinstance(selection.get("non_selected_reason"), str) and selection["non_selected_reason"].strip(), "native operator non-selected path requires non_selected_reason")
    else:
        raise GoalGateError("actual_model_regression.native_operator_selection.selected must be boolean")
    return {**summary, "_summary_path": str(path)}


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
    observability = validate_observability_profile(args.observability_profile, expected_sha)
    native_operator = validate_native_operator(args.native_operator, expected_sha)
    actual_model = validate_actual_model_regression(args, expected_sha)
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
        "stage_artifacts": {
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
        },
        "gate_plan": change_impact["gate_plan"],
        "release_candidate_manifest": change_impact["release_candidate_manifest"],
        "resource_invariant_summary": resource["summary"],
        "product_scenario_summary": product["summary"],
        "model_contract_summary": model_contract["summary"],
        "observability_profile_summary": observability["summary"],
        "observability_vertical_slice_summary": observability["vertical_slice"],
        "actual_model_regression_summary": actual_model,
        "native_operator_artifact_summary": native_operator["summary"],
        "fa2_source_removal_inventory": native_operator["fa2_source_removal_inventory"],
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = build_goal_manifest(args)
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
        "resource_summary": {},
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
            "invariant_report": str(resource / "invariant_report.json"),
        },
    )

    change = root / "change"
    change.mkdir()
    write_json(
        change / "gate_plan.json",
        {
            "schema_version": 1,
            "status": "pass",
            "head_sha": sha,
            "unknown_files": [],
            "required_gates": ["unit"],
        },
    )
    write_json(
        change / "release_candidate_manifest.json",
        {
            "schema_version": 1,
            "head_sha": sha,
            "required_gates": ["unit"],
            "invalidated_gates": [],
        },
    )

    product = root / "product"
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
            "scenario_count": 12,
            "required_stage2_fixture_count": 12,
            "scenarios": [],
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
    vertical_summary = {
        "schema_version": 1,
        "gate": "observability_vertical_slice",
        "status": "pass",
        "entrypoints": {"run": {}, "serve": {}},
    }
    write_json(vertical / "observability_profile_summary.json", vertical_summary)
    write_json(
        vertical / "observability_vertical_slice_manifest.json",
        {
            "schema_version": 1,
            "status": "pass",
            "artifact_dir": str(vertical),
            "pass_line": f"OBSERVABILITY VERTICAL SLICE PASS: {vertical}",
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
            "latency_p50_p95_p99": {"p50": 1, "p95": 1, "p99": 1},
            "memory_high_water_bytes": {"cpu:process": 1},
            "resource_leak_count": 0,
            "top_slow_phases": [],
            "first_failure_event": None,
            "replay_commands": [],
            "vertical_slice_artifact": str(vertical),
            "actual_smoke_artifact": "fixture-l1-smoke",
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
            "manifests": [{"operator": "fa2", "backend": "cuda"}],
            "bulk_source": {"count": 0, "samples": []},
            "unregistered_third_party_source": {"count": 0, "samples": []},
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
                "git_sha": sha,
                "artifact_dir": "fixtures/metal-l2",
                "entrypoints": ["run", "serve", "stream", "basic_concurrency"],
            },
            "cuda_l2_artifact": {
                "status": "pass",
                "backend": "cuda",
                "git_sha": sha,
                "artifact_dir": "fixtures/cuda-l2",
                "entrypoints": ["run", "serve", "stream", "basic_concurrency"],
            },
            "native_operator_selection": {
                "status": "pass",
                "selected": True,
                "cuda_artifact": "fixtures/cuda-native-op",
            },
        },
    )
    return {
        "resource": resource,
        "change": change,
        "product": product,
        "model": model,
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
            observability_profile=artifacts["observability"],
            native_operator=artifacts["native"],
            actual_model_regression_summary=artifacts["actual"],
            binary_sha256=None,
            require_clean=False,
        )
        manifest = run_gate(args)
        require((out / "goal_manifest.json").is_file(), "selftest missing goal_manifest.json")
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
