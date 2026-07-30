#!/usr/bin/env python3
"""Portable native-operator artifact-set closure validation and staging."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable


LOCK_SCHEMA_VERSION = 5
LOCK_FILE_NAME = "native-operator-set.lock.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SINGLE_EVIDENCE_FIELDS = (
    "manifest",
    "package_spec",
    "g03_catalog",
    "abi_contract",
    "source_build_receipt",
    "source_build_plan",
    "package_receipt",
)
LIST_EVIDENCE_FIELDS = (
    "source_build_inputs",
    "source_build_logs",
    "package_build_logs",
    "license_files",
)
ARTIFACT_FIELDS = {
    "operator",
    "backend",
    "manifest_path",
    "manifest",
    "artifact_path",
    "operator_abi_version",
    "ferrum_native_abi_version",
    "source_package_sha256",
    "inputs_sha256",
    "package_spec",
    "g03_catalog",
    "abi_contract",
    "source_build_receipt",
    "source_build_plan",
    "source_build_inputs",
    "source_build_logs",
    "source_archive_sha256",
    "package_receipt",
    "package_build_logs",
    "license_files",
    "binary_sha256",
    "abi_contract_sha256",
    "descriptor_export",
    "required_exports",
    "operation_bindings",
    "system_libraries",
}


class NativeOperatorSetEvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeOperatorSetEvidenceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def cuda_build_inputs_hash(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def cuda_native_set_signature(
    build_lock_path: str,
    native_identity: dict[str, Any],
    build_units: Iterable[tuple[str, str, str]],
) -> str:
    units = tuple(build_units)
    operator_binaries = ",".join(
        f"{row['operator']}={row['sha256']}"
        for row in native_identity["binary_sha256_by_operator"]
    )
    return (
        f"lock={build_lock_path}:"
        f"lock_sha256={native_identity['lock_sha256']}:"
        f"catalog={native_identity['g03_catalog_sha256']}:"
        f"operators={len(units)}:"
        f"operator_binaries={operator_binaries}:"
        f"build_units={','.join(unit_name for _, unit_name, _ in units)}"
    )


def validate_cuda_build_summary(
    path: Path,
    build_lock_path: str,
    native_identity: dict[str, Any],
    build_units: Iterable[tuple[str, str, str]],
) -> dict[str, Any]:
    units = tuple(build_units)
    require(path.is_file() and not path.is_symlink(), "CUDA build summary receipt is missing")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeOperatorSetEvidenceError(
            f"CUDA build summary receipt is unreadable: {path}: {error}"
        ) from error
    require(
        isinstance(data, dict)
        and set(data) == {"schema_version", "artifact_type", "rows"}
        and data.get("schema_version") == 1
        and data.get("artifact_type") == "ferrum_cuda_build_summary_receipt",
        "CUDA build summary receipt schema mismatch",
    )
    rows = data.get("rows")
    require(isinstance(rows, list), "CUDA build summary rows are missing")
    by_artifact: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        require(
            isinstance(raw, dict)
            and set(raw)
            == {"artifact", "status", "reason", "elapsed_ms", "inputs_hash"},
            f"CUDA build summary row {index} field set is invalid",
        )
        artifact = raw.get("artifact")
        require(
            isinstance(artifact, str) and artifact and artifact not in by_artifact,
            f"CUDA build summary row {index} has an invalid or duplicate artifact",
        )
        require(
            isinstance(raw.get("status"), str)
            and raw["status"] not in {"rejected", "failed"}
            and isinstance(raw.get("reason"), str)
            and raw["reason"]
            and isinstance(raw.get("elapsed_ms"), int)
            and not isinstance(raw["elapsed_ms"], bool)
            and raw["elapsed_ms"] >= 0
            and re.fullmatch(r"sha256:[0-9a-f]{64}", str(raw.get("inputs_hash")))
            is not None,
            f"CUDA build summary row {index} is malformed or rejected",
        )
        by_artifact[artifact] = raw

    set_row = by_artifact.get("native_operator_artifact_set")
    expected_set_signature = cuda_native_set_signature(
        build_lock_path,
        native_identity,
        units,
    )
    require(
        isinstance(set_row, dict)
        and set_row.get("status") == "linked"
        and set_row.get("reason") == "manifest-v3-artifact-set-v5-validated"
        and set_row.get("inputs_hash")
        == cuda_build_inputs_hash(expected_set_signature),
        "CUDA build did not validate and link the native operator artifact set",
    )
    for summary_artifact, _unit_name, operator in units:
        row = by_artifact.get(summary_artifact)
        require(
            isinstance(row, dict)
            and row.get("status") == "artifact"
            and row.get("reason") == "native-operator-artifact-set"
            and row.get("inputs_hash") == cuda_build_inputs_hash(operator),
            f"CUDA build did not bind {summary_artifact} through the native operator artifact set",
        )
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": len(rows),
        "native_operator_artifact_set_status": "linked",
        "native_operator_artifact_set_inputs_hash": set_row["inputs_hash"],
    }


def _absolute_without_resolution(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _safe_relative_path(value: Any, label: str) -> Path:
    require(
        isinstance(value, str) and value and "\\" not in value,
        f"{label} path is invalid",
    )
    relative = Path(value)
    require(
        not relative.is_absolute()
        and relative.as_posix() == value
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"{label} path is unsafe",
    )
    return relative


def _read_lock(lock_path: Path) -> tuple[Path, dict[str, Any]]:
    expanded = _absolute_without_resolution(lock_path.expanduser())
    require(
        not expanded.is_symlink(),
        f"native operator set lock must not be a symlink: {expanded}",
    )
    resolved = expanded.resolve()
    require(
        resolved.is_file() and not resolved.is_symlink(),
        f"native operator set lock is not a regular file: {resolved}",
    )
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeOperatorSetEvidenceError(
            f"native operator set lock is unreadable: {resolved}: {error}"
        ) from error
    require(isinstance(raw, dict), "native operator set lock root is not an object")
    require(
        set(raw) == {"schema_version", "g03_catalog_sha256", "artifacts"},
        "native operator set lock field set mismatch",
    )
    require(
        raw.get("schema_version") == LOCK_SCHEMA_VERSION,
        f"native operator set lock schema must be {LOCK_SCHEMA_VERSION}",
    )
    require(
        isinstance(raw.get("g03_catalog_sha256"), str)
        and SHA256_RE.fullmatch(raw["g03_catalog_sha256"]) is not None,
        "native operator set lock catalog SHA256 is invalid",
    )
    require(
        isinstance(raw.get("artifacts"), list) and raw["artifacts"],
        "native operator set lock has no artifacts",
    )
    return resolved, raw


def _regular_member(root: Path, relative: Path, label: str) -> Path:
    candidate = root
    for part in relative.parts:
        candidate /= part
        require(not candidate.is_symlink(), f"{label} contains a symlink: {candidate}")
    require(
        candidate.is_file()
        and not candidate.is_symlink()
        and candidate.resolve().is_relative_to(root),
        f"{label} file is missing or escapes the lock root: {candidate}",
    )
    return candidate


def validate_native_operator_set(
    lock_path: Path,
    required_operators: Iterable[str],
) -> dict[str, Any]:
    resolved, lock = _read_lock(lock_path)
    root = resolved.parent.resolve()
    required = sorted(required_operators)
    entries: dict[str, dict[str, Any]] = {}
    operators: list[str] = []
    operator_binaries: list[dict[str, str]] = []
    operation_binding_count = 0

    def add_entry(
        raw_path: Any,
        raw_sha256: Any,
        raw_size: Any,
        label: str,
    ) -> None:
        relative = _safe_relative_path(raw_path, label)
        require(
            isinstance(raw_sha256, str)
            and SHA256_RE.fullmatch(raw_sha256) is not None,
            f"{label} SHA256 is invalid",
        )
        path = _regular_member(root, relative, label)
        size = path.stat().st_size
        require(
            raw_size is None
            or (
                isinstance(raw_size, int)
                and not isinstance(raw_size, bool)
                and raw_size == size
            ),
            f"{label} size mismatch",
        )
        require(sha256_file(path) == raw_sha256, f"{label} SHA256 mismatch")
        row = {
            "path": relative.as_posix(),
            "sha256": raw_sha256,
            "size_bytes": size,
        }
        previous = entries.get(row["path"])
        require(
            previous is None or previous == row,
            f"{label} has conflicting duplicate evidence: {row['path']}",
        )
        entries[row["path"]] = row

    def add_evidence(raw: Any, label: str) -> None:
        require(
            isinstance(raw, dict)
            and set(raw) == {"path", "sha256", "size_bytes"},
            f"{label} evidence record is invalid",
        )
        add_entry(raw["path"], raw["sha256"], raw["size_bytes"], label)

    for index, raw_artifact in enumerate(lock["artifacts"]):
        label = f"native operator artifact {index}"
        require(isinstance(raw_artifact, dict), f"{label} is invalid")
        require(set(raw_artifact) == ARTIFACT_FIELDS, f"{label} field set mismatch")
        operator = raw_artifact.get("operator")
        require(isinstance(operator, str) and operator, f"{label} operator is invalid")
        operators.append(operator)
        operator_binaries.append(
            {
                "operator": operator,
                "sha256": str(raw_artifact.get("binary_sha256")),
            }
        )
        require(raw_artifact.get("backend") == "cuda", f"{label} backend is not CUDA")
        for field in (
            "source_package_sha256",
            "inputs_sha256",
            "source_archive_sha256",
            "binary_sha256",
            "abi_contract_sha256",
        ):
            require(
                isinstance(raw_artifact.get(field), str)
                and SHA256_RE.fullmatch(raw_artifact[field]) is not None,
                f"{label}.{field} is not a SHA256 digest",
            )
        for field in (
            "operator_abi_version",
            "ferrum_native_abi_version",
            "descriptor_export",
        ):
            require(
                isinstance(raw_artifact.get(field), str) and raw_artifact[field],
                f"{label}.{field} is empty",
            )
        for field in ("required_exports", "operation_bindings", "system_libraries"):
            require(isinstance(raw_artifact.get(field), list), f"{label}.{field} is not a list")
        require(raw_artifact["required_exports"], f"{label}.required_exports is empty")
        operation_binding_count += len(raw_artifact["operation_bindings"])

        for field in SINGLE_EVIDENCE_FIELDS:
            add_evidence(raw_artifact.get(field), f"{label}.{field}")
        for field in LIST_EVIDENCE_FIELDS:
            rows = raw_artifact.get(field)
            require(isinstance(rows, list), f"{label}.{field} is not a list")
            for evidence_index, evidence in enumerate(rows):
                add_evidence(evidence, f"{label}.{field}[{evidence_index}]")

        manifest_relative = _safe_relative_path(
            raw_artifact.get("manifest_path"),
            f"{label}.manifest_path",
        )
        require(
            manifest_relative.as_posix() == raw_artifact["manifest"]["path"],
            f"{label}.manifest_path differs from manifest evidence",
        )
        add_entry(
            raw_artifact.get("artifact_path"),
            raw_artifact.get("binary_sha256"),
            None,
            f"{label}.artifact",
        )

    require(
        len(operators) == len(set(operators)),
        "native operator set lock contains duplicate operators",
    )
    require(
        operators == sorted(operators),
        "native operator set lock operators are not deterministically sorted",
    )
    require(
        operators == required,
        "native operator set lock does not cover the required operator set",
    )
    require(
        operation_binding_count > 0,
        "native operator set must bind at least one live operation/provider",
    )
    members = [entries[path] for path in sorted(entries)]
    require(
        len(members) >= len(operators) * 9,
        "native operator set evidence closure is unexpectedly small",
    )
    return {
        "lock_path": str(resolved),
        "lock_sha256": sha256_file(resolved),
        "lock_size_bytes": resolved.stat().st_size,
        "schema_version": LOCK_SCHEMA_VERSION,
        "g03_catalog_sha256": lock["g03_catalog_sha256"],
        "operators": operators,
        "binary_sha256_by_operator": operator_binaries,
        "closure": {
            "member_count": len(members),
            "total_bytes": sum(row["size_bytes"] for row in members),
            "index_sha256": canonical_json_sha256(members),
        },
        "_members": members,
    }


def public_identity(validated: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in validated.items()
        if key not in {"lock_path", "_members"}
    }


def cuda_native_build_cache_contract(
    build_units: Iterable[tuple[str, str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "ferrum_cuda_native_build_cache_contract",
        "build_summary_schema_version": 1,
        "build_units": [list(unit) for unit in build_units],
    }


def native_operator_set_cache_key(
    validated: dict[str, Any],
    cache_contract: dict[str, Any],
) -> str:
    return canonical_json_sha256(
        {
            "schema_version": 1,
            "artifact_type": "ferrum_native_operator_set_build_cache_key",
            "native_operator_set": public_identity(validated),
            "cache_contract": cache_contract,
        }
    )


def stage_native_operator_set(
    source_lock: Path,
    destination_root: Path,
    required_operators: Iterable[str],
) -> tuple[Path, dict[str, Any]]:
    source = validate_native_operator_set(source_lock, required_operators)
    destination = _absolute_without_resolution(destination_root)
    require(
        not destination.exists() and not destination.is_symlink(),
        f"native operator set staging directory already exists: {destination}",
    )
    destination.mkdir(parents=True)
    source_root = Path(source["lock_path"]).parent
    for row in source["_members"]:
        relative = Path(row["path"])
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root.joinpath(*relative.parts), target)
        require(
            target.stat().st_size == row["size_bytes"]
            and sha256_file(target) == row["sha256"],
            f"staged native operator member changed during copy: {row['path']}",
        )
    staged_lock = destination / LOCK_FILE_NAME
    require(
        LOCK_FILE_NAME not in {row["path"] for row in source["_members"]},
        "native operator evidence closure collides with the staged lock name",
    )
    shutil.copy2(Path(source["lock_path"]), staged_lock)

    source_after = validate_native_operator_set(source_lock, required_operators)
    staged = validate_native_operator_set(staged_lock, required_operators)
    require(
        public_identity(source_after) == public_identity(source)
        and source_after["_members"] == source["_members"],
        "native operator set source changed while it was staged",
    )
    require(
        public_identity(staged) == public_identity(source)
        and staged["_members"] == source["_members"],
        "staged native operator set closure differs from its source",
    )
    return staged_lock.resolve(), staged


def ensure_content_addressed_native_operator_set(
    source_lock: Path,
    cache_root: Path,
    required_operators: Iterable[str],
    cache_contract: dict[str, Any],
) -> tuple[Path, dict[str, Any], str, bool]:
    operators = tuple(required_operators)
    source = validate_native_operator_set(source_lock, operators)
    cache_key = native_operator_set_cache_key(source, cache_contract)
    root = _absolute_without_resolution(cache_root.expanduser())
    require(
        not root.is_symlink(),
        f"native operator build cache root must not be a symlink: {root}",
    )
    root.mkdir(parents=True, exist_ok=True)
    require(
        root.is_dir() and not root.is_symlink(),
        f"native operator build cache root is not a directory: {root}",
    )
    destination = root / cache_key
    cached_lock = destination / LOCK_FILE_NAME

    if destination.exists() or destination.is_symlink():
        require(
            destination.is_dir() and not destination.is_symlink(),
            f"native operator build cache entry is not a directory: {destination}",
        )
        cached = validate_native_operator_set(cached_lock, operators)
        require(
            public_identity(cached) == public_identity(source)
            and cached["_members"] == source["_members"],
            f"native operator build cache identity mismatch: {destination}",
        )
        return cached_lock.resolve(), cached, cache_key, True

    temporary = root / f".{cache_key}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        staged_lock, staged = stage_native_operator_set(
            source_lock,
            temporary,
            operators,
        )
        try:
            temporary.rename(destination)
        except OSError:
            if not destination.is_dir() or destination.is_symlink():
                raise
            shutil.rmtree(temporary)
        cached = validate_native_operator_set(cached_lock, operators)
        require(
            public_identity(cached) == public_identity(source)
            and cached["_members"] == source["_members"]
            and public_identity(cached) == public_identity(staged),
            f"native operator build cache changed while it was published: {destination}",
        )
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    return cached_lock.resolve(), cached, cache_key, False


def create_selftest_native_operator_set(
    root: Path,
    required_operators: Iterable[str],
) -> Path:
    """Create a structurally complete closure fixture for release-script self-tests."""
    root.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []

    def evidence(path: Path, content: bytes) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    for index, operator in enumerate(sorted(required_operators)):
        package = root / "packages" / str(index)
        manifest = evidence(package / "manifest.json", b'{"schema_version":3}\n')
        package_spec = evidence(package / "package.spec.json", b'{"fixture":true}\n')
        catalog = evidence(package / "g03-catalog.json", b'{"fixture":true}\n')
        abi = evidence(package / "abi.json", b'{"fixture":true}\n')
        source_receipt = evidence(package / "source-receipt.json", b'{"fixture":true}\n')
        source_plan = evidence(package / "source-plan.json", b'{"fixture":true}\n')
        source_input = evidence(package / "source-input.cu", b"// fixture\n")
        source_log = evidence(package / "source.log", b"fixture\n")
        package_receipt = evidence(package / "package-receipt.json", b'{"fixture":true}\n')
        package_log = evidence(package / "package.log", b"fixture\n")
        license_file = evidence(package / "LICENSE", b"fixture license\n")
        archive = package / "libfixture.a"
        archive.write_bytes(f"native-{index}\n".encode("ascii"))
        artifacts.append(
            {
                "operator": operator,
                "backend": "cuda",
                "manifest_path": manifest["path"],
                "manifest": manifest,
                "artifact_path": archive.relative_to(root).as_posix(),
                "operator_abi_version": "fixture-v1",
                "ferrum_native_abi_version": "2",
                "source_package_sha256": "1" * 64,
                "inputs_sha256": "2" * 64,
                "package_spec": package_spec,
                "g03_catalog": catalog,
                "abi_contract": abi,
                "source_build_receipt": source_receipt,
                "source_build_plan": source_plan,
                "source_build_inputs": [source_input],
                "source_build_logs": [source_log],
                "source_archive_sha256": "3" * 64,
                "package_receipt": package_receipt,
                "package_build_logs": [package_log],
                "license_files": [license_file],
                "binary_sha256": sha256_file(archive),
                "abi_contract_sha256": "4" * 64,
                "descriptor_export": f"ferrum_fixture_descriptor_{index}",
                "required_exports": [f"ferrum_fixture_{index}"],
                "operation_bindings": []
                if index == 0
                else [
                    {
                        "operation_id": f"fixture.operation.{index}",
                        "provider_id": f"fixture.provider.{index}",
                    }
                ],
                "system_libraries": [],
            }
        )
    lock = root / LOCK_FILE_NAME
    lock.write_text(
        json.dumps(
            {
                "schema_version": LOCK_SCHEMA_VERSION,
                "g03_catalog_sha256": "5" * 64,
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_native_operator_set(lock, required_operators)
    return lock
