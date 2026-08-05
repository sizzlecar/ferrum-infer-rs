#!/usr/bin/env python3
"""Capture and validate a source-scoped CUDA release execution-plan reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    import runtime_vnext_native_operator_set as native_operator_set
except ModuleNotFoundError:
    from scripts.release import runtime_vnext_native_operator_set as native_operator_set


LEGACY_SCHEMA_VERSION = 1
SCHEMA_VERSION = 2
ARTIFACT_TYPE = "runtime-vnext-cuda-release-plan-reference"
READY_PREFIX = "FERRUM CUDA RELEASE PLAN REFERENCE READY"
CANDIDATE_BUILD_RECEIPT_TYPE = "runtime_vnext_candidate_build_receipt"
EXECUTION_CONTRACT = "g08-model-matrix-v1"
MODEL_KEY = "m2-qwen35-35b-a3b"
CASE_ID = "c13-022"
BACKEND = "cuda"
FEATURES = [
    "cuda",
    "vllm-moe-marlin",
    "vllm-paged-attn-v2",
]
RELEASE_BUILD_COMMAND = [
    "cargo",
    "build",
    "--release",
    "--locked",
    "--jobs",
    "4",
    "-p",
    "ferrum-cli",
    "--bin",
    "ferrum",
    "--features",
    ",".join(FEATURES),
]
NATIVE_OPERATOR_SET_LOCK_ENV = "FERRUM_NATIVE_OPERATOR_SET_LOCK"
NATIVE_OPERATOR_SET_LOCK_INPUT = "native-operator-set.lock.json"
CANONICAL_NATIVE_OPERATOR_SET_LOCK_REL = Path(
    "build/candidate/native-operator-set.lock.json"
)
REQUIRED_CUDA_NATIVE_OPERATORS = (
    "ferrum.cuda.marlin",
    "ferrum.cuda.vllm_marlin",
    "ferrum.cuda.vllm_moe_marlin",
    "ferrum.cuda.vllm_paged_attention_v2",
)
GIT_SHA_LENGTH = 40
SHA256_LENGTH = 64
MAX_PORTABLE_ARTIFACT_BYTES = 512 * 1024 * 1024
SELFTEST_PASS_LINE = "FERRUM CUDA RELEASE PLAN REFERENCE SELFTEST PASS"


class PlanReferenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanReferenceError(message)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def require_git_sha(value: Any, label: str) -> str:
    require(is_lower_hex(value, GIT_SHA_LENGTH), f"{label} must be a lowercase Git SHA")
    return value


def require_sha256(value: Any, label: str) -> str:
    require(is_lower_hex(value, SHA256_LENGTH), f"{label} must be a lowercase SHA256")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanReferenceError(f"{label} is not valid UTF-8 JSON: {path}: {error}") from error
    require(isinstance(value, dict) and value, f"{label} must be a non-empty object")
    return value


def file_ref(path: Path, artifact_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(artifact_root).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def execution_artifact_ref(
    path: Path,
    artifact_root: Path,
    kind: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": path.relative_to(artifact_root).as_posix(),
        "sha256": sha256(path),
    }


def resolve_file_ref(artifact_root: Path, raw: Any, label: str) -> Path:
    require(isinstance(raw, dict), f"{label} reference must be an object")
    require(
        set(raw) == {"path", "sha256", "size_bytes"},
        f"{label} reference must contain path, sha256, and size_bytes",
    )
    relative = raw.get("path")
    require(
        isinstance(relative, str)
        and relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        f"{label} reference path is invalid",
    )
    unresolved = artifact_root / relative
    require(
        not unresolved.is_symlink(),
        f"{label} reference must not be a symlink",
    )
    path = unresolved.resolve()
    try:
        path.relative_to(artifact_root.resolve())
    except ValueError as error:
        raise PlanReferenceError(f"{label} reference escapes its artifact root") from error
    require(path.is_file() and not path.is_symlink(), f"{label} file is missing: {path}")
    require(
        raw.get("sha256") == sha256(path)
        and raw.get("size_bytes") == path.stat().st_size,
        f"{label} reference identity mismatch",
    )
    return path


def resolve_execution_artifact_ref(
    artifact_root: Path,
    raw: Any,
    label: str,
    *,
    expected_kind: str,
) -> Path:
    require(isinstance(raw, dict), f"{label} artifact reference must be an object")
    require(
        set(raw) == {"kind", "path", "sha256"},
        f"{label} artifact reference must contain exactly kind, path, and sha256",
    )
    require(
        raw.get("kind") == expected_kind,
        f"{label} artifact kind must be {expected_kind}",
    )
    relative = raw.get("path")
    require(
        isinstance(relative, str)
        and relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        f"{label} artifact reference path is invalid",
    )
    unresolved = artifact_root / relative
    require(
        not unresolved.is_symlink(),
        f"{label} artifact reference must not be a symlink",
    )
    path = unresolved.resolve()
    try:
        path.relative_to(artifact_root.resolve())
    except ValueError as error:
        raise PlanReferenceError(
            f"{label} artifact reference escapes its artifact root"
        ) from error
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
        f"{label} artifact is missing or empty: {path}",
    )
    require(
        raw.get("sha256") == sha256(path),
        f"{label} artifact reference identity mismatch",
    )
    return path


def require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise PlanReferenceError(
            f"{label} must be under execution artifact root {root}: {path}"
        ) from error


def plan_events_from_trace(trace_path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, raw_line in enumerate(
        trace_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise PlanReferenceError(
                f"scheduler trace line {line_number} is invalid JSON: {error}"
            ) from error
        require(
            isinstance(event, dict),
            f"scheduler trace line {line_number} must be an object",
        )
        if event.get("phase") != "vnext.plan_built":
            continue
        attributes = event.get("attributes")
        require(
            isinstance(attributes, dict),
            f"plan_built line {line_number} attributes are missing",
        )
        plan_hash = require_sha256(
            attributes.get("plan_hash"),
            f"plan_built line {line_number} plan_hash",
        )
        plan_id = attributes.get("plan_id")
        require(
            plan_id == f"plan/sha256/{plan_hash}",
            f"plan_built line {line_number} plan_id does not derive from plan_hash",
        )
        require(
            event.get("backend") == "actual"
            and event.get("entrypoint") == "serve"
            and event.get("status") == "ok"
            and attributes.get("execution_trace_source") == "vnext",
            f"plan_built line {line_number} is not actual vNext serve evidence",
        )
        events.append(
            {
                "line_number": line_number,
                "request_id": event.get("request_id"),
                "model": event.get("model"),
                "plan_hash": plan_hash,
                "plan_id": plan_id,
            }
        )
    require(events, "scheduler trace has no vnext.plan_built events")
    return events


def validate_focused_report(
    focused: dict[str, Any],
    *,
    expected: dict[str, Any],
) -> None:
    for key in (
        "source_git_sha",
        "source_tree_sha",
        "binary_sha256",
        "backend",
        "model_key",
        "model_revision",
        "models_lock_sha256",
        "hardware_id",
    ):
        require(
            focused.get(key) == expected.get(key),
            f"focused report {key} differs from its execution manifest",
        )
    dirty = focused.get("dirty_status")
    require(
        isinstance(dirty, dict)
        and dirty.get("is_dirty") is False
        and dirty.get("status_short") == [],
        "focused report requires a clean source",
    )
    scope = focused.get("scope")
    require(
        isinstance(scope, dict)
        and scope.get("kind") == "focused-diagnostic"
        and scope.get("requested_case_ids") == [CASE_ID]
        and scope.get("requested_scenario_ids") == [],
        f"focused report scope must be exactly {CASE_ID}",
    )
    require(
        focused.get("decision") in {"KEEP", "REJECT", "PASS"},
        "focused report decision is missing",
    )


def validate_native_operator_set_lock_identity(
    raw: Any,
    *,
    lock_path: Path | None,
    require_recorded_path: bool,
) -> dict[str, Any]:
    require(
        isinstance(raw, dict)
        and set(raw) == {"path", "sha256", "size_bytes", "mtime_ns"},
        "release build native operator set lock identity is invalid",
    )
    recorded_path_raw = raw.get("path")
    recorded_path = Path(recorded_path_raw) if isinstance(recorded_path_raw, str) else Path()
    require(
        isinstance(recorded_path_raw, str)
        and recorded_path_raw
        and recorded_path.is_absolute()
        and ".." not in recorded_path.parts
        and str(recorded_path) == recorded_path_raw,
        "release build native operator set lock path is invalid",
    )
    require_sha256(raw.get("sha256"), "release build native operator set lock sha256")
    for key in ("size_bytes", "mtime_ns"):
        value = raw.get(key)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"release build native operator set lock {key} is invalid",
        )
    if lock_path is not None:
        require(
            lock_path.is_absolute()
            and not lock_path.is_symlink()
            and lock_path.is_file()
            and lock_path.stat().st_size > 0,
            f"release build native operator set lock is missing: {lock_path}",
        )
        require(
            not require_recorded_path or recorded_path == lock_path,
            "release build command and native operator set lock path differ",
        )
        require(
            raw["sha256"] == sha256(lock_path)
            and raw["size_bytes"] == lock_path.stat().st_size,
            "release build native operator set lock content identity differs",
        )
        if require_recorded_path:
            require(
                raw["mtime_ns"] == lock_path.stat().st_mtime_ns,
                "release build native operator set lock mtime differs",
            )
    return raw


def validate_native_operator_set_closure(lock_path: Path) -> dict[str, Any]:
    try:
        return native_operator_set.validate_native_operator_set(
            lock_path,
            REQUIRED_CUDA_NATIVE_OPERATORS,
        )
    except native_operator_set.NativeOperatorSetEvidenceError as error:
        raise PlanReferenceError(
            f"release build native operator set closure is invalid: {error}"
        ) from error


def validate_release_build_command(
    receipt: dict[str, Any],
    *,
    allow_legacy: bool,
    execution_root: Path | None = None,
    portable_lock_path: Path | None = None,
) -> Path | None:
    command = receipt.get("command")
    if command == RELEASE_BUILD_COMMAND:
        require(
            allow_legacy and receipt.get("native_operator_set_lock") is None,
            "legacy release command is only valid for a schema v1 reference",
        )
        return None

    require(
        isinstance(command, list)
        and len(command) == len(RELEASE_BUILD_COMMAND) + 2
        and command[0] == "env"
        and command[2:] == RELEASE_BUILD_COMMAND,
        "reference build did not use the exact official CUDA release command",
    )
    assignment = command[1]
    prefix = f"{NATIVE_OPERATOR_SET_LOCK_ENV}="
    require(
        isinstance(assignment, str)
        and assignment.startswith(prefix)
        and assignment.count("=") == 1
        and len(assignment) > len(prefix),
        "release build command has an invalid native operator set lock binding",
    )
    command_lock_path = Path(assignment[len(prefix) :])
    identity = validate_native_operator_set_lock_identity(
        receipt.get("native_operator_set_lock"),
        lock_path=None,
        require_recorded_path=False,
    )
    require(
        command_lock_path == Path(identity["path"]),
        "release build command and native operator set lock path differ",
    )

    if execution_root is not None:
        require_under(command_lock_path, execution_root, "native operator set lock")
        require(
            command_lock_path.resolve()
            == (execution_root / CANONICAL_NATIVE_OPERATOR_SET_LOCK_REL).resolve(),
            "release build native operator set lock path is not canonical",
        )
        validate_native_operator_set_lock_identity(
            identity,
            lock_path=command_lock_path,
            require_recorded_path=True,
        )
        validate_native_operator_set_closure(command_lock_path)
    if portable_lock_path is not None:
        validate_native_operator_set_lock_identity(
            identity,
            lock_path=portable_lock_path,
            require_recorded_path=False,
        )
        validate_native_operator_set_closure(portable_lock_path)
    return command_lock_path


def validate_release_build_receipt(
    execution_root: Path,
    execution: dict[str, Any],
) -> tuple[Path, dict[str, Any], Path, Path]:
    receipt_path = resolve_execution_artifact_ref(
        execution_root,
        execution.get("binary_build_receipt"),
        "release build receipt",
        expected_kind="raw-json",
    )
    receipt = read_object(receipt_path, "release build receipt")
    require(
        receipt.get("schema_version") == 1
        and receipt.get("artifact_type") == CANDIDATE_BUILD_RECEIPT_TYPE
        and receipt.get("status") == "pass"
        and receipt.get("execution_contract") == EXECUTION_CONTRACT,
        "reference build receipt is not a passing G08 candidate receipt",
    )
    require(
        receipt.get("backend") == BACKEND
        and receipt.get("build_mode", "release") == "release",
        "reference build did not use the official CUDA release profile",
    )
    native_lock = validate_release_build_command(
        receipt,
        allow_legacy=False,
        execution_root=execution_root,
    )
    require(native_lock is not None, "current release build requires a native operator set lock")
    for key in ("source_git_sha", "source_tree_sha", "hardware_id", "binary_sha256"):
        require(
            receipt.get(key) == execution.get(key),
            f"release build receipt {key} differs from execution",
        )
    dirty = receipt.get("dirty_status")
    require(
        isinstance(dirty, dict)
        and dirty.get("is_dirty") is False
        and dirty.get("status_short") == [],
        "release build receipt requires a clean source",
    )
    binary_path = resolve_execution_artifact_ref(
        execution_root,
        execution.get("binary_artifact"),
        "reference execution binary",
        expected_kind="binary",
    )
    build_binary_path = resolve_execution_artifact_ref(
        execution_root,
        receipt.get("binary_artifact"),
        "reference build binary",
        expected_kind="binary",
    )
    require(
        sha256(binary_path)
        == sha256(build_binary_path)
        == execution.get("binary_sha256")
        == receipt.get("binary_sha256"),
        "release build and execution binary identities differ",
    )
    return receipt_path, receipt, binary_path, native_lock


def capture(
    *,
    out_root: Path,
    execution_manifest_path: Path,
    focused_report_path: Path,
    trace_path: Path,
    actual_effective_config_path: Path,
    hardware_id: str,
) -> dict[str, Any]:
    root = out_root.expanduser().resolve()
    require(not root.exists() or not any(root.iterdir()), f"output root must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    execution_manifest_path = execution_manifest_path.expanduser().resolve()
    focused_report_path = focused_report_path.expanduser().resolve()
    trace_path = trace_path.expanduser().resolve()
    actual_effective_config_path = actual_effective_config_path.expanduser().resolve()
    for path, label in (
        (execution_manifest_path, "execution manifest"),
        (focused_report_path, "focused report"),
        (trace_path, "scheduler trace"),
        (actual_effective_config_path, "actual effective config"),
    ):
        require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    execution_root = execution_manifest_path.parent
    for path, label in (
        (focused_report_path, "focused report"),
        (trace_path, "scheduler trace"),
        (actual_effective_config_path, "actual effective config"),
    ):
        require_under(path, execution_root, label)
    require(
        trace_path.name.endswith(".scheduler-trace.jsonl"),
        "reference input is not a scheduler-trace JSONL file",
    )

    execution = read_object(execution_manifest_path, "execution manifest")
    focused = read_object(focused_report_path, "focused report")
    require(
        execution.get("schema_version") == 1
        and execution.get("execution_contract") == EXECUTION_CONTRACT
        and execution.get("backend") == BACKEND
        and execution.get("model_key") == MODEL_KEY,
        "reference execution is not the M2 CUDA G08 product lane",
    )
    require_git_sha(execution.get("source_git_sha"), "reference execution source_git_sha")
    require_git_sha(execution.get("source_tree_sha"), "reference execution source_tree_sha")
    require_sha256(execution.get("binary_sha256"), "reference execution binary_sha256")
    require_sha256(
        execution.get("models_lock_sha256"),
        "reference execution models_lock_sha256",
    )
    dirty = execution.get("dirty_status")
    require(
        isinstance(dirty, dict)
        and dirty.get("is_dirty") is False
        and dirty.get("status_short") == [],
        "reference execution requires a clean source",
    )
    require(
        hardware_id.strip() == hardware_id
        and hardware_id
        and execution.get("hardware_id") == hardware_id,
        "reference execution hardware differs from the requested hardware",
    )
    validate_focused_report(focused, expected=execution)
    (
        build_receipt_path,
        _,
        release_binary_path,
        native_operator_set_lock_path,
    ) = validate_release_build_receipt(execution_root, execution)
    effective_config_path = resolve_execution_artifact_ref(
        execution_root,
        execution.get("effective_config"),
        "reference typed effective config",
        expected_kind="raw-json",
    )
    actual_config = read_object(
        actual_effective_config_path,
        "actual effective config",
    )
    require(
        actual_config.get("backend") == BACKEND
        or any(
            isinstance(entry, dict)
            and entry.get("key") == "FERRUM_BACKEND"
            and entry.get("effective_value") == BACKEND
            for entry in actual_config.get("entries", [])
        ),
        "actual effective config lacks typed CUDA evidence",
    )
    events = plan_events_from_trace(trace_path)
    observed_hashes = sorted({event["plan_hash"] for event in events})
    require(
        len(observed_hashes) == 1,
        f"release reference trace contains multiple plan identities: {observed_hashes}",
    )
    plan_hash = observed_hashes[0]
    model_files = execution.get("model_files")
    require(
        isinstance(model_files, dict)
        and model_files
        and all(
            isinstance(name, str)
            and name
            and is_lower_hex(digest, SHA256_LENGTH)
            for name, digest in model_files.items()
        ),
        "reference execution model file lock is invalid",
    )

    input_dir = root / "inputs"
    input_dir.mkdir(parents=True)
    try:
        portable_native_lock, _ = native_operator_set.stage_native_operator_set(
            native_operator_set_lock_path,
            input_dir / "native-operator-set",
            REQUIRED_CUDA_NATIVE_OPERATORS,
        )
    except native_operator_set.NativeOperatorSetEvidenceError as error:
        raise PlanReferenceError(
            f"failed to stage the release native operator set: {error}"
        ) from error
    copied: dict[str, Any] = {
        NATIVE_OPERATOR_SET_LOCK_INPUT: file_ref(portable_native_lock, root)
    }
    for label, source in (
        ("release-build-receipt.json", build_receipt_path),
        ("execution-manifest.json", execution_manifest_path),
        ("focused-report.json", focused_report_path),
        ("scheduler-trace.jsonl", trace_path),
        ("typed-effective-config.json", effective_config_path),
        ("actual-effective-config.json", actual_effective_config_path),
        ("release-binary", release_binary_path),
    ):
        destination = input_dir / label
        shutil.copy2(source, destination)
        copied[label] = file_ref(destination, root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "reference-ready",
        "created_at": now_iso(),
        "scope": {
            "execution_contract": EXECUTION_CONTRACT,
            "case_id": CASE_ID,
            "model_key": MODEL_KEY,
            "backend": BACKEND,
            "entrypoint": "serve",
        },
        "profile": "release",
        "features": FEATURES,
        "source_git_sha": execution["source_git_sha"],
        "source_tree_sha": execution["source_tree_sha"],
        "dirty_status": execution["dirty_status"],
        "binary_sha256": execution["binary_sha256"],
        "models_lock_sha256": execution["models_lock_sha256"],
        "model_revision": execution["model_revision"],
        "model_files_sha256": canonical_json_sha256(model_files),
        "hardware_id": execution["hardware_id"],
        "typed_effective_config_sha256": sha256(effective_config_path),
        "actual_effective_config_sha256": sha256(actual_effective_config_path),
        "plan_identity": {
            "plan_hash": plan_hash,
            "plan_id": f"plan/sha256/{plan_hash}",
            "plan_built_event_count": len(events),
            "model_values": sorted(
                {
                    event["model"]
                    for event in events
                    if isinstance(event["model"], str)
                }
            ),
        },
        "source_compatibility_policy": (
            "reference commit must equal or be an ancestor of the candidate; "
            "the canonical plan identity must still match exactly"
        ),
        "inputs": copied,
        "ready_line": f"{READY_PREFIX}: {root}",
        "pass_line": None,
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def validate(
    manifest_path: Path,
    *,
    expected_hardware_id: str,
    candidate_source_git_sha: str | None = None,
    candidate_source_tree_sha: str | None = None,
    repository_root: Path | None = None,
    allow_internal_fixture: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        f"release plan reference manifest is missing: {manifest_path}",
    )
    root = manifest_path.parent
    manifest = read_object(manifest_path, "release plan reference manifest")
    schema_version = manifest.get("schema_version")
    require(
        schema_version in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}
        and manifest.get("artifact_type") == ARTIFACT_TYPE
        and manifest.get("status") == "reference-ready",
        "release plan reference schema, type, or status is invalid",
    )
    require(
        manifest.get("scope")
        == {
            "execution_contract": EXECUTION_CONTRACT,
            "case_id": CASE_ID,
            "model_key": MODEL_KEY,
            "backend": BACKEND,
            "entrypoint": "serve",
        },
        "release plan reference scope is invalid",
    )
    require(
        manifest.get("profile") == "release"
        and manifest.get("features") == FEATURES
        and manifest.get("hardware_id") == expected_hardware_id,
        "release plan reference profile, features, or hardware differs",
    )
    reference_sha = require_git_sha(
        manifest.get("source_git_sha"),
        "release plan reference source_git_sha",
    )
    reference_tree = require_git_sha(
        manifest.get("source_tree_sha"),
        "release plan reference source_tree_sha",
    )
    for key in (
        "binary_sha256",
        "models_lock_sha256",
        "model_files_sha256",
        "typed_effective_config_sha256",
        "actual_effective_config_sha256",
    ):
        require_sha256(manifest.get(key), f"release plan reference {key}")
    dirty = manifest.get("dirty_status")
    require(
        isinstance(dirty, dict)
        and dirty.get("is_dirty") is False
        and dirty.get("status_short") == [],
        "release plan reference requires a clean source",
    )
    identity = manifest.get("plan_identity")
    require(isinstance(identity, dict), "release plan identity is missing")
    plan_hash = require_sha256(identity.get("plan_hash"), "release plan identity plan_hash")
    require(
        identity.get("plan_id") == f"plan/sha256/{plan_hash}"
        and isinstance(identity.get("plan_built_event_count"), int)
        and identity["plan_built_event_count"] >= 1,
        "release plan identity is invalid",
    )
    inputs = manifest.get("inputs")
    require(isinstance(inputs, dict), "release plan reference inputs are missing")
    expected_inputs = {
        "release-build-receipt.json",
        "execution-manifest.json",
        "focused-report.json",
        "scheduler-trace.jsonl",
        "typed-effective-config.json",
        "actual-effective-config.json",
        "release-binary",
    }
    if schema_version == SCHEMA_VERSION:
        expected_inputs.add(NATIVE_OPERATOR_SET_LOCK_INPUT)
    require(set(inputs) == expected_inputs, "release plan reference input set is invalid")
    resolved = {
        label: resolve_file_ref(root, inputs[label], f"release plan reference {label}")
        for label in sorted(expected_inputs)
    }
    execution = read_object(
        resolved["execution-manifest.json"],
        "release reference execution manifest",
    )
    focused = read_object(
        resolved["focused-report.json"],
        "release reference focused report",
    )
    build_receipt = read_object(
        resolved["release-build-receipt.json"],
        "release reference build receipt",
    )
    for key in (
        "source_git_sha",
        "source_tree_sha",
        "binary_sha256",
        "models_lock_sha256",
        "model_revision",
        "hardware_id",
    ):
        require(
            execution.get(key) == manifest.get(key),
            f"release reference execution {key} differs from manifest",
        )
    require(
        execution.get("backend") == BACKEND
        and execution.get("model_key") == MODEL_KEY
        and canonical_json_sha256(execution.get("model_files"))
        == manifest["model_files_sha256"],
        "release reference execution model binding differs from manifest",
    )
    require(
        build_receipt.get("schema_version") == 1
        and build_receipt.get("artifact_type") == CANDIDATE_BUILD_RECEIPT_TYPE
        and build_receipt.get("status") == "pass"
        and build_receipt.get("execution_contract") == EXECUTION_CONTRACT
        and build_receipt.get("backend") == BACKEND
        and build_receipt.get("build_mode", "release") == "release"
        and build_receipt.get("binary_sha256") == manifest["binary_sha256"],
        "release reference build receipt no longer proves the official release profile",
    )
    validated_native_lock = validate_release_build_command(
        build_receipt,
        allow_legacy=schema_version == LEGACY_SCHEMA_VERSION,
        portable_lock_path=(
            resolved[NATIVE_OPERATOR_SET_LOCK_INPUT]
            if schema_version == SCHEMA_VERSION
            else None
        ),
    )
    require(
        (schema_version == SCHEMA_VERSION and validated_native_lock is not None)
        or (
            schema_version == LEGACY_SCHEMA_VERSION
            and validated_native_lock is None
        ),
        "release plan reference schema does not match its native lock contract",
    )
    validate_focused_report(focused, expected=execution)
    require(
        sha256(resolved["typed-effective-config.json"])
        == manifest["typed_effective_config_sha256"]
        and sha256(resolved["actual-effective-config.json"])
        == manifest["actual_effective_config_sha256"],
        "release reference effective config identity differs from manifest",
    )
    require(
        sha256(resolved["release-binary"]) == manifest["binary_sha256"],
        "release reference binary content differs from its recorded SHA256",
    )
    trace_events = plan_events_from_trace(resolved["scheduler-trace.jsonl"])
    require(
        sorted({event["plan_hash"] for event in trace_events}) == [plan_hash]
        and len(trace_events) == identity["plan_built_event_count"],
        "release reference trace differs from its recorded plan identity",
    )
    require(
        manifest.get("pass_line") is None
        and isinstance(manifest.get("ready_line"), str)
        and manifest["ready_line"].startswith(f"{READY_PREFIX}: "),
        "release plan reference readiness was confused with PASS",
    )

    if candidate_source_git_sha is not None:
        require_git_sha(candidate_source_git_sha, "candidate source_git_sha")
        require_git_sha(candidate_source_tree_sha, "candidate source_tree_sha")
        if reference_sha == candidate_source_git_sha:
            require(
                reference_tree == candidate_source_tree_sha,
                "same-commit release plan reference has a different source tree",
            )
        elif not allow_internal_fixture:
            require(
                repository_root is not None,
                "repository root is required for plan-reference ancestry validation",
            )
            ancestor = subprocess.run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    reference_sha,
                    candidate_source_git_sha,
                ],
                cwd=repository_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            require(
                ancestor.returncode == 0,
                "stale-plan-reference: reference source is not an ancestor of the candidate",
            )
            tree = subprocess.run(
                ["git", "rev-parse", f"{reference_sha}^{{tree}}"],
                cwd=repository_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            require(
                tree.returncode == 0 and tree.stdout.strip() == reference_tree,
                "release plan reference source tree does not belong to its commit",
            )
    return manifest


def copy_validated(
    *,
    source_manifest: Path,
    destination_root: Path,
    expected_hardware_id: str,
    candidate_source_git_sha: str,
    candidate_source_tree_sha: str,
    repository_root: Path,
) -> tuple[dict[str, Any], Path]:
    source_manifest = source_manifest.expanduser().resolve()
    reference = validate(
        source_manifest,
        expected_hardware_id=expected_hardware_id,
        candidate_source_git_sha=candidate_source_git_sha,
        candidate_source_tree_sha=candidate_source_tree_sha,
        repository_root=repository_root,
    )
    source_root = source_manifest.parent
    destination_root = destination_root.resolve()
    require(not destination_root.exists(), f"reference destination exists: {destination_root}")
    total_bytes = 0
    for path in source_root.rglob("*"):
        require(not path.is_symlink(), f"release plan reference contains a symlink: {path}")
        if path.is_file():
            total_bytes += path.stat().st_size
    require(
        total_bytes <= MAX_PORTABLE_ARTIFACT_BYTES,
        "release plan reference exceeds the 512 MiB portable artifact limit",
    )
    shutil.copytree(source_root, destination_root, symlinks=False)
    imported_manifest = destination_root / source_manifest.relative_to(source_root)
    imported = validate(
        imported_manifest,
        expected_hardware_id=expected_hardware_id,
        candidate_source_git_sha=candidate_source_git_sha,
        candidate_source_tree_sha=candidate_source_tree_sha,
        repository_root=repository_root,
    )
    require(imported == reference, "imported release plan reference changed during copy")
    return imported, imported_manifest


def make_fixture(root: Path) -> tuple[Path, dict[str, str]]:
    source_git_sha = "1" * 40
    source_tree_sha = "2" * 40
    binary_bytes = b"release-plan-reference-binary"
    plan_hash = "a" * 64
    hardware_id = "fixture-rtx4090-24564mib"
    execution_root = root / "execution"
    binary = execution_root / "build/candidate/ferrum"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(binary_bytes)
    native_lock = native_operator_set.create_selftest_native_operator_set(
        binary.parent,
        REQUIRED_CUDA_NATIVE_OPERATORS,
    )
    native_lock_stat = native_lock.stat()
    build_receipt = execution_root / "build/candidate/candidate-build-receipt.json"
    write_json(
        build_receipt,
        {
            "schema_version": 1,
            "artifact_type": CANDIDATE_BUILD_RECEIPT_TYPE,
            "status": "pass",
            "execution_contract": EXECUTION_CONTRACT,
            "source_git_sha": source_git_sha,
            "source_tree_sha": source_tree_sha,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "hardware_id": hardware_id,
            "backend": BACKEND,
            "command": [
                "env",
                f"{NATIVE_OPERATOR_SET_LOCK_ENV}={native_lock}",
                *RELEASE_BUILD_COMMAND,
            ],
            "native_operator_set_lock": {
                "path": str(native_lock),
                "sha256": sha256(native_lock),
                "size_bytes": native_lock_stat.st_size,
                "mtime_ns": native_lock_stat.st_mtime_ns,
            },
            "binary_sha256": sha256(binary),
            "binary_artifact": execution_artifact_ref(
                binary,
                execution_root,
                "binary",
            ),
        },
    )
    effective = execution_root / "effective-config.json"
    actual = execution_root / "actual-effective-config.json"
    write_json(effective, {"schema_version": 1, "execution_contract": EXECUTION_CONTRACT})
    write_json(
        actual,
        {
            "schema_version": 1,
            "backend": BACKEND,
            "entries": [{"key": "FERRUM_BACKEND", "effective_value": BACKEND}],
        },
    )
    execution = execution_root / "execution-manifest.json"
    model_files = {"config.json": "5" * 64}
    write_json(
        execution,
        {
            "schema_version": 1,
            "execution_contract": EXECUTION_CONTRACT,
            "backend": BACKEND,
            "model_key": MODEL_KEY,
            "model_revision": "3" * 40,
            "model_files": model_files,
            "models_lock_sha256": "4" * 64,
            "hardware_id": hardware_id,
            "source_git_sha": source_git_sha,
            "source_tree_sha": source_tree_sha,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "binary_sha256": sha256(binary),
            "binary_artifact": execution_artifact_ref(
                binary,
                execution_root,
                "binary",
            ),
            "binary_build_receipt": execution_artifact_ref(
                build_receipt,
                execution_root,
                "raw-json",
            ),
            "effective_config": execution_artifact_ref(
                effective,
                execution_root,
                "raw-json",
            ),
        },
    )
    focused = execution_root / "correctness/focused-c13-022-report.json"
    write_json(
        focused,
        {
            "schema_version": 1,
            "backend": BACKEND,
            "model_key": MODEL_KEY,
            "model_revision": "3" * 40,
            "models_lock_sha256": "4" * 64,
            "hardware_id": hardware_id,
            "source_git_sha": source_git_sha,
            "source_tree_sha": source_tree_sha,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "binary_sha256": sha256(binary),
            "decision": "KEEP",
            "scope": {
                "kind": "focused-diagnostic",
                "requested_case_ids": [CASE_ID],
                "requested_scenario_ids": [],
            },
        },
    )
    trace = execution_root / "commands/serve-01.scheduler-trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "phase": "vnext.plan_built",
                "backend": "actual",
                "entrypoint": "serve",
                "status": "ok",
                "request_id": "request.fixture",
                "model": "fixture",
                "attributes": {
                    "execution_trace_source": "vnext",
                    "plan_hash": plan_hash,
                    "plan_id": f"plan/sha256/{plan_hash}",
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return execution_root, {
        "execution": str(execution),
        "focused": str(focused),
        "trace": str(trace),
        "actual": str(actual),
        "hardware_id": hardware_id,
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "plan_hash": plan_hash,
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-plan-reference-") as raw:
        root = Path(raw)
        execution_root, fixture = make_fixture(root)
        execution = read_object(
            Path(fixture["execution"]),
            "fixture execution manifest",
        )
        fixture_receipt_path = resolve_execution_artifact_ref(
            execution_root,
            execution["binary_build_receipt"],
            "fixture build receipt",
            expected_kind="raw-json",
        )
        fixture_receipt = read_object(fixture_receipt_path, "fixture build receipt")
        fixture_native_lock = validate_release_build_command(
            fixture_receipt,
            allow_legacy=False,
            execution_root=execution_root,
        )
        require(
            fixture_native_lock is not None,
            "canonical fixture release command did not bind a native operator set",
        )

        hostile_commands = {
            "extra-env": [
                "env",
                "FERRUM_HOSTILE=1",
                *fixture_receipt["command"][1:],
            ],
            "wrong-env-key": [
                "env",
                f"FERRUM_NATIVE_LOCK={fixture_native_lock}",
                *RELEASE_BUILD_COMMAND,
            ],
            "relative-lock": [
                "env",
                f"{NATIVE_OPERATOR_SET_LOCK_ENV}=relative.lock.json",
                *RELEASE_BUILD_COMMAND,
            ],
        }
        for name, command in hostile_commands.items():
            hostile_receipt = json.loads(json.dumps(fixture_receipt))
            hostile_receipt["command"] = command
            try:
                validate_release_build_command(
                    hostile_receipt,
                    allow_legacy=False,
                    execution_root=execution_root,
                )
            except PlanReferenceError:
                pass
            else:
                raise PlanReferenceError(f"hostile {name} release command was accepted")

        wrong_identity = json.loads(json.dumps(fixture_receipt))
        wrong_identity["native_operator_set_lock"]["path"] = str(
            execution_root / "wrong-native.lock.json"
        )
        try:
            validate_release_build_command(
                wrong_identity,
                allow_legacy=False,
                execution_root=execution_root,
            )
        except PlanReferenceError as error:
            require(
                "command and native operator set lock path differ" in str(error),
                f"wrong native lock path used an unexpected rejection: {error}",
            )
        else:
            raise PlanReferenceError("wrong native lock identity path was accepted")

        wrong_lock_sha = json.loads(json.dumps(fixture_receipt))
        wrong_lock_sha["native_operator_set_lock"]["sha256"] = "f" * SHA256_LENGTH
        try:
            validate_release_build_command(
                wrong_lock_sha,
                allow_legacy=False,
                execution_root=execution_root,
            )
        except PlanReferenceError as error:
            require(
                "content identity differs" in str(error),
                f"wrong native lock SHA used an unexpected rejection: {error}",
            )
        else:
            raise PlanReferenceError("wrong native lock SHA was accepted")

        legacy_receipt = json.loads(json.dumps(fixture_receipt))
        legacy_receipt["command"] = list(RELEASE_BUILD_COMMAND)
        legacy_receipt.pop("native_operator_set_lock")
        require(
            validate_release_build_command(legacy_receipt, allow_legacy=True) is None,
            "schema v1 release command compatibility did not validate",
        )
        try:
            validate_release_build_command(legacy_receipt, allow_legacy=False)
        except PlanReferenceError as error:
            require(
                "schema v1" in str(error),
                f"current schema legacy command used an unexpected rejection: {error}",
            )
        else:
            raise PlanReferenceError("current schema accepted a legacy release command")

        binary_ref = execution["binary_artifact"]
        binary_path = resolve_execution_artifact_ref(
            execution_root,
            binary_ref,
            "fixture execution binary",
            expected_kind="binary",
        )
        require(
            binary_path.is_file(),
            "typed execution artifact reference did not resolve",
        )
        hostile_refs = {
            "missing-kind": (
                {
                    "path": binary_ref["path"],
                    "sha256": binary_ref["sha256"],
                },
                "must contain exactly",
            ),
            "wrong-kind": (
                {**binary_ref, "kind": "raw-json"},
                "artifact kind must be binary",
            ),
            "extra-size": (
                {**binary_ref, "size_bytes": binary_path.stat().st_size},
                "must contain exactly",
            ),
            "wrong-sha": (
                {**binary_ref, "sha256": "f" * SHA256_LENGTH},
                "identity mismatch",
            ),
            "path-escape": (
                {**binary_ref, "path": "../ferrum"},
                "path is invalid",
            ),
        }
        for name, (hostile_ref, expected_error) in hostile_refs.items():
            try:
                resolve_execution_artifact_ref(
                    execution_root,
                    hostile_ref,
                    f"hostile {name}",
                    expected_kind="binary",
                )
            except PlanReferenceError as error:
                require(
                    expected_error in str(error),
                    f"{name} used an unexpected rejection: {error}",
                )
            else:
                raise PlanReferenceError(f"{name} execution artifact ref was accepted")
        symlink_path = execution_root / "symlinked-ferrum"
        symlink_path.symlink_to(binary_path)
        try:
            resolve_execution_artifact_ref(
                execution_root,
                {
                    "kind": "binary",
                    "path": symlink_path.relative_to(execution_root).as_posix(),
                    "sha256": binary_ref["sha256"],
                },
                "hostile symlink",
                expected_kind="binary",
            )
        except PlanReferenceError as error:
            require(
                "must not be a symlink" in str(error),
                f"symlink used an unexpected rejection: {error}",
            )
        else:
            raise PlanReferenceError("symlinked execution artifact ref was accepted")

        artifact = root / "reference"
        captured = capture(
            out_root=artifact,
            execution_manifest_path=Path(fixture["execution"]),
            focused_report_path=Path(fixture["focused"]),
            trace_path=Path(fixture["trace"]),
            actual_effective_config_path=Path(fixture["actual"]),
            hardware_id=fixture["hardware_id"],
        )
        validated = validate(
            artifact / "manifest.json",
            expected_hardware_id=fixture["hardware_id"],
            candidate_source_git_sha=fixture["source_git_sha"],
            candidate_source_tree_sha=fixture["source_tree_sha"],
            allow_internal_fixture=True,
        )
        require(
            captured == validated
            and validated["plan_identity"]["plan_hash"] == fixture["plan_hash"],
            "matching release plan reference did not round-trip",
        )
        portable_lock = resolve_file_ref(
            artifact,
            validated["inputs"][NATIVE_OPERATOR_SET_LOCK_INPUT],
            "fixture portable native operator set lock",
        )
        portable_set = validate_native_operator_set_closure(portable_lock)
        member_path = portable_lock.parent / portable_set["_members"][0]["path"]
        member_bytes = member_path.read_bytes()
        member_path.write_bytes(member_bytes + b"-tampered")
        try:
            validate(
                artifact / "manifest.json",
                expected_hardware_id=fixture["hardware_id"],
                candidate_source_git_sha=fixture["source_git_sha"],
                candidate_source_tree_sha=fixture["source_tree_sha"],
                allow_internal_fixture=True,
            )
        except PlanReferenceError as error:
            require(
                "native operator set closure is invalid" in str(error)
                and "mismatch" in str(error),
                f"tampered native closure member used an unexpected rejection: {error}",
            )
        else:
            raise PlanReferenceError("tampered native operator closure member was accepted")
        member_path.write_bytes(member_bytes)

        release_binary = artifact / "inputs/release-binary"
        release_binary_bytes = release_binary.read_bytes()
        release_binary.write_bytes(release_binary_bytes + b"-tampered")
        try:
            validate(
                artifact / "manifest.json",
                expected_hardware_id=fixture["hardware_id"],
                candidate_source_git_sha=fixture["source_git_sha"],
                candidate_source_tree_sha=fixture["source_tree_sha"],
                allow_internal_fixture=True,
            )
        except PlanReferenceError as error:
            require(
                "release-binary" in str(error)
                and "identity mismatch" in str(error),
                f"tampered release binary used an unexpected rejection: {error}",
            )
        else:
            raise PlanReferenceError("tampered release binary was accepted")
        release_binary.write_bytes(release_binary_bytes)

        hostile = read_object(artifact / "manifest.json", "hostile manifest")
        hostile["plan_identity"]["plan_hash"] = "b" * 64
        hostile["plan_identity"]["plan_id"] = f"plan/sha256/{'b' * 64}"
        write_json(artifact / "manifest.json", hostile)
        try:
            validate(
                artifact / "manifest.json",
                expected_hardware_id=fixture["hardware_id"],
                candidate_source_git_sha=fixture["source_git_sha"],
                candidate_source_tree_sha=fixture["source_tree_sha"],
                allow_internal_fixture=True,
            )
        except PlanReferenceError as error:
            require(
                "trace differs" in str(error),
                f"hostile plan hash used an unexpected rejection: {error}",
            )
        else:
            raise PlanReferenceError("hostile plan hash was accepted")

        build_mode_hostile = json.loads(json.dumps(validated))
        receipt_path = artifact / "inputs/release-build-receipt.json"
        receipt = read_object(receipt_path, "hostile release build receipt")
        receipt["build_mode"] = "debug"
        write_json(receipt_path, receipt)
        build_mode_hostile["inputs"]["release-build-receipt.json"] = file_ref(
            receipt_path,
            artifact,
        )
        write_json(artifact / "manifest.json", build_mode_hostile)
        try:
            validate(
                artifact / "manifest.json",
                expected_hardware_id=fixture["hardware_id"],
                candidate_source_git_sha=fixture["source_git_sha"],
                candidate_source_tree_sha=fixture["source_tree_sha"],
                allow_internal_fixture=True,
            )
        except PlanReferenceError as error:
            require(
                "official release profile" in str(error),
                f"hostile build mode used an unexpected rejection: {error}",
            )
        else:
            raise PlanReferenceError("hostile release build mode was accepted")
    print(SELFTEST_PASS_LINE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--focused-report", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--actual-effective-config", type=Path)
    parser.add_argument("--hardware-id")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    for key in (
        "out",
        "execution_manifest",
        "focused_report",
        "trace",
        "actual_effective_config",
        "hardware_id",
    ):
        require(getattr(args, key) is not None, f"--{key.replace('_', '-')} is required")
    return args


def main() -> int:
    try:
        args = parse_args()
        if args.self_test:
            self_test()
            return 0
        manifest = capture(
            out_root=args.out,
            execution_manifest_path=args.execution_manifest,
            focused_report_path=args.focused_report,
            trace_path=args.trace,
            actual_effective_config_path=args.actual_effective_config,
            hardware_id=args.hardware_id,
        )
        print(manifest["ready_line"])
        return 0
    except (OSError, PlanReferenceError, subprocess.TimeoutExpired) as error:
        print(f"FERRUM CUDA RELEASE PLAN REFERENCE REJECT: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
