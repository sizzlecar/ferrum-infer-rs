#!/usr/bin/env python3
"""Independently verify a copied G07B native-operator chain artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


SCHEMA_VERSION = 1
RECEIPT_SCHEMA = "ferrum.bounded-command-receipt.v1"
SOURCE_BUILD_RECEIPT_SCHEMA_VERSION = 5
DEPENDENCY_DOMAIN_ORDER = {
    "source": 0,
    "backend_toolchain": 1,
    "host_toolchain": 2,
}
REQUIRED_CUDA_TOOLKIT_FILES = {
    "bin/bin2c",
    "bin/cudafe++",
    "bin/fatbinary",
    "bin/nvcc",
    "bin/nvlink",
    "bin/ptxas",
}
REQUIRED_CUDA_TOOLKIT_SCOPES = (
    "bin/crt/",
    "include/",
    "nvvm/bin/",
    "nvvm/libdevice/",
)
PACKAGES = {
    "marlin": "ferrum.cuda.marlin",
    "vllm-marlin": "ferrum.cuda.vllm_marlin",
    "vllm-moe-marlin": "ferrum.cuda.vllm_moe_marlin",
    "vllm-paged-attention-v2": "ferrum.cuda.vllm_paged_attention_v2",
}
SOURCE_FEATURES = [
    "cuda",
    "vllm-marlin",
    "vllm-moe-marlin",
    "vllm-paged-attn-v2",
]
ARTIFACT_FEATURES = [*SOURCE_FEATURES, "native-op-artifact"]
EXPECTED_BUILD_UNITS = {
    "marlin",
    "vllm_marlin",
    "vllm_moe_marlin",
    "vllm_paged_attn",
}
EXPECTED_STEPS = {
    "builder-build",
    "bootstrap-example-build",
    "bootstrap-catalog-export",
    "assemble-artifact-set",
    "artifact-example-build",
    "artifact-catalog-export",
    *(f"materialize-{name}" for name in PACKAGES),
    *(f"source-build-{name}" for name in PACKAGES),
    *(f"package-{name}" for name in PACKAGES),
}
DOES_NOT_PROVE = {
    "canonical G03 PASS",
    "canonical G07A PASS",
    "canonical G07B PASS",
    "G07 aggregate PASS",
    "model correctness",
    "model performance",
    "release readiness",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CUDA_SUMMARY_RE = re.compile(
    r"\[cuda-build-summary\]\s+artifact=(\S+)\s+status=(\S+)\s+reason=(\S+)"
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def require_sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA256",
    )
    return value


def read_json(path: Path, label: str) -> Any:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {label} {path}: {error}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(cwd: Path, command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    require(
        result.returncode == 0,
        f"command failed ({result.returncode}): {command!r}: {result.stderr[-1000:]}",
    )
    return result.stdout.strip()


def resolve_relative_file(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise VerificationError(f"{label}.path must be non-empty")
    require("\\" not in raw, f"{label}.path must use portable separators")
    relative = Path(raw)
    require(
        not relative.is_absolute()
        and raw == relative.as_posix()
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"{label}.path is not a safe relative path: {raw!r}",
    )
    canonical_root = root.resolve()
    candidate = root.joinpath(*relative.parts)
    require(
        candidate.is_file() and not candidate.is_symlink(),
        f"{label} is missing or is a symlink: {candidate}",
    )
    require(
        candidate.resolve().is_relative_to(canonical_root),
        f"{label} escapes its artifact root: {candidate}",
    )
    return candidate


def verify_file_identity(path: Path, identity: Any, label: str) -> None:
    row = require_dict(identity, label)
    require(set(row) == {"path", "sha256", "size_bytes"}, f"{label} shape mismatch")
    require_sha(row["sha256"], f"{label}.sha256")
    require(
        isinstance(row["size_bytes"], int) and row["size_bytes"] >= 0,
        f"{label}.size_bytes is invalid",
    )
    require(sha256(path) == row["sha256"], f"{label} SHA256 mismatch")
    require(path.stat().st_size == row["size_bytes"], f"{label} size mismatch")


def verify_evidence(root: Path, evidence: Any, label: str) -> Path:
    row = require_dict(evidence, label)
    path = resolve_relative_file(root, row.get("path"), label)
    verify_file_identity(path, row, label)
    return path


def verify_evidence_list(root: Path, values: Any, label: str, *, nonempty: bool) -> list[Path]:
    rows = require_list(values, label)
    require(not nonempty or bool(rows), f"{label} must not be empty")
    paths = [verify_evidence(root, row, f"{label}[{index}]") for index, row in enumerate(rows)]
    relative = [path.relative_to(root.resolve()).as_posix() for path in paths]
    require(relative == sorted(set(relative)), f"{label} paths must be sorted and unique")
    return paths


DependencyIdentity = tuple[str, str, str]


def require_absolute_posix_path(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be non-empty")
    require("\\" not in value, f"{label} must use POSIX separators")
    path = PurePosixPath(value)
    require(
        path.is_absolute()
        and value == path.as_posix()
        and all(part not in {"", ".", ".."} for part in path.parts[1:]),
        f"{label} must be a normalized absolute path: {value!r}",
    )
    return value


def require_relative_posix_path(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be non-empty")
    require("\\" not in value, f"{label} must use POSIX separators")
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and value == path.as_posix()
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} must be a normalized relative path: {value!r}",
    )
    return value


def normalize_depfile_relative_path(value: str, label: str) -> str:
    require("\\" not in value, f"{label} contains a non-portable separator")
    parts: list[str] = []
    for part in value.split("/"):
        if part in {"", "."}:
            continue
        require(part != "..", f"{label} escapes its working directory: {value!r}")
        parts.append(part)
    require(bool(parts), f"{label} is empty after normalization")
    return require_relative_posix_path("/".join(parts), label)


def verify_tool_file_identity(value: Any, label: str) -> dict[str, Any]:
    row = require_dict(value, label)
    require(set(row) == {"path", "sha256", "size_bytes"}, f"{label} shape mismatch")
    require_absolute_posix_path(row.get("path"), f"{label}.path")
    require_sha(row.get("sha256"), f"{label}.sha256")
    require(
        isinstance(row.get("size_bytes"), int) and row["size_bytes"] > 0,
        f"{label}.size_bytes must be positive",
    )
    return row


def dependency_identity(value: Any, label: str) -> DependencyIdentity:
    row = require_dict(value, label)
    require(set(row) == {"domain", "path", "sha256"}, f"{label} shape mismatch")
    domain = row.get("domain")
    require(domain in DEPENDENCY_DOMAIN_ORDER, f"{label}.domain is unsupported: {domain!r}")
    if domain == "host_toolchain":
        path = require_absolute_posix_path(row.get("path"), f"{label}.path")
    else:
        path = require_relative_posix_path(row.get("path"), f"{label}.path")
    digest = require_sha(row.get("sha256"), f"{label}.sha256")
    return (domain, path, digest)


def dependency_sort_key(value: DependencyIdentity) -> tuple[int, str, str]:
    return (DEPENDENCY_DOMAIN_ORDER[value[0]], value[1], value[2])


def verify_dependency_rows(values: Any, label: str) -> list[DependencyIdentity]:
    rows = require_list(values, label)
    require(bool(rows), f"{label} must not be empty")
    identities = [
        dependency_identity(row, f"{label}[{index}]") for index, row in enumerate(rows)
    ]
    require(
        identities == sorted(set(identities), key=dependency_sort_key),
        f"{label} must be strictly sorted and unique in dependency-domain order",
    )
    return identities


def insert_toolchain_owner(
    owners: dict[str, DependencyIdentity],
    absolute_path: str,
    identity: DependencyIdentity,
    label: str,
) -> None:
    path = require_absolute_posix_path(absolute_path, label)
    previous = owners.get(path)
    require(
        previous is None or previous == identity,
        f"toolchain manifests ambiguously own {path}: {previous!r} versus {identity!r}",
    )
    owners[path] = identity


def posix_join(root: str, relative: str) -> str:
    return (PurePosixPath(root) / PurePosixPath(relative)).as_posix()


def verify_toolchain_manifests(
    static_identity: Any,
    cuda_manifest_value: Any,
    host_manifest_value: Any,
    label: str,
) -> tuple[dict[str, DependencyIdentity], set[DependencyIdentity]]:
    static = require_dict(static_identity, f"{label}.static_identity")
    require(
        set(static)
        == {
            "backend",
            "compiler_driver",
            "cuda_toolkit",
            "host_toolchain",
            "archiver",
        },
        f"{label}.static_identity shape mismatch",
    )
    require(
        static.get("backend") == "cuda"
        and static.get("compiler_driver") == "cuda_nvcc",
        f"{label} must use the explicit CUDA nvcc compiler driver",
    )
    verify_tool_file_identity(static.get("archiver"), f"{label}.archiver")

    cuda = require_dict(static.get("cuda_toolkit"), f"{label}.cuda_toolkit")
    require(
        set(cuda) == {"canonical_root", "release_version", "nvcc", "manifest"},
        f"{label}.cuda_toolkit shape mismatch",
    )
    cuda_root = require_absolute_posix_path(
        cuda.get("canonical_root"), f"{label}.cuda_toolkit.canonical_root"
    )
    require(
        isinstance(cuda.get("release_version"), str)
        and re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", cuda["release_version"]) is not None,
        f"{label}.cuda_toolkit.release_version is invalid",
    )
    nvcc = verify_tool_file_identity(cuda.get("nvcc"), f"{label}.cuda_toolkit.nvcc")
    cuda_manifest = require_dict(cuda_manifest_value, f"{label}.cuda_manifest")
    require(
        set(cuda_manifest) == {"schema_version", "canonical_root", "entries"}
        and cuda_manifest.get("schema_version") == 1
        and cuda_manifest.get("canonical_root") == cuda_root,
        f"{label}.cuda_manifest identity mismatch",
    )
    cuda_entries = require_list(cuda_manifest.get("entries"), f"{label}.cuda_manifest.entries")
    require(bool(cuda_entries), f"{label}.cuda_manifest.entries must not be empty")

    owners: dict[str, DependencyIdentity] = {}
    allowed: set[DependencyIdentity] = set()
    logical_paths: list[str] = []
    selected_nvcc: dict[str, Any] | None = None
    covered_scopes: set[str] = set()
    for index, raw in enumerate(cuda_entries):
        row = require_dict(raw, f"{label}.cuda_manifest.entries[{index}]")
        require(
            set(row) == {"logical_path", "resolved_path", "sha256", "size_bytes"},
            f"{label}.cuda_manifest.entries[{index}] shape mismatch",
        )
        logical = require_relative_posix_path(
            row.get("logical_path"), f"{label}.cuda_manifest.entries[{index}].logical_path"
        )
        resolved = require_relative_posix_path(
            row.get("resolved_path"), f"{label}.cuda_manifest.entries[{index}].resolved_path"
        )
        digest = require_sha(
            row.get("sha256"), f"{label}.cuda_manifest.entries[{index}].sha256"
        )
        require(
            isinstance(row.get("size_bytes"), int) and row["size_bytes"] >= 0,
            f"{label}.cuda_manifest.entries[{index}].size_bytes is invalid",
        )
        logical_paths.append(logical)
        identity = ("backend_toolchain", logical, digest)
        allowed.add(identity)
        for relative in (logical, resolved):
            absolute = posix_join(cuda_root, relative)
            insert_toolchain_owner(
                owners,
                absolute,
                identity,
                f"{label}.cuda_manifest.entries[{index}]",
            )
            if absolute == nvcc["path"]:
                selected_nvcc = row
        for scope in REQUIRED_CUDA_TOOLKIT_SCOPES:
            if logical.startswith(scope):
                covered_scopes.add(scope)
    require(
        logical_paths == sorted(set(logical_paths)),
        f"{label}.cuda_manifest entries must be strictly sorted by logical_path",
    )
    require(
        REQUIRED_CUDA_TOOLKIT_FILES.issubset(logical_paths)
        and covered_scopes == set(REQUIRED_CUDA_TOOLKIT_SCOPES),
        f"{label}.cuda_manifest does not cover the required compiler inputs",
    )
    require(
        selected_nvcc is not None
        and selected_nvcc.get("sha256") == nvcc["sha256"]
        and selected_nvcc.get("size_bytes") == nvcc["size_bytes"],
        f"{label}.cuda_manifest does not bind the selected nvcc",
    )

    host = require_dict(static.get("host_toolchain"), f"{label}.host_toolchain")
    require(
        set(host) == {"compiler", "compiler_version", "target", "manifest"},
        f"{label}.host_toolchain shape mismatch",
    )
    compiler = verify_tool_file_identity(host.get("compiler"), f"{label}.host_toolchain.compiler")
    require(
        isinstance(host.get("compiler_version"), str) and bool(host["compiler_version"].strip()),
        f"{label}.host_toolchain.compiler_version is missing",
    )
    require(
        isinstance(host.get("target"), str)
        and bool(host["target"])
        and len(host["target"]) <= 256
        and not any(character.isspace() for character in host["target"]),
        f"{label}.host_toolchain.target is invalid",
    )
    host_manifest = require_dict(host_manifest_value, f"{label}.host_manifest")
    require(
        set(host_manifest)
        == {
            "schema_version",
            "compiler",
            "compiler_version",
            "target",
            "executable_inputs",
            "include_roots",
            "include_probe_sha256",
            "driver_probe_sha256",
            "discovery_roots",
            "files",
        }
        and host_manifest.get("schema_version") == 2
        and host_manifest.get("compiler") == compiler
        and host_manifest.get("compiler_version") == host["compiler_version"]
        and host_manifest.get("target") == host["target"],
        f"{label}.host_manifest identity mismatch",
    )
    require_sha(
        host_manifest.get("include_probe_sha256"), f"{label}.host_manifest.include_probe_sha256"
    )
    require_sha(
        host_manifest.get("driver_probe_sha256"), f"{label}.host_manifest.driver_probe_sha256"
    )
    executable_inputs = [
        verify_tool_file_identity(row, f"{label}.host_manifest.executable_inputs[{index}]")
        for index, row in enumerate(
            require_list(
                host_manifest.get("executable_inputs"),
                f"{label}.host_manifest.executable_inputs",
            )
        )
    ]
    executable_paths = [row["path"] for row in executable_inputs]
    require(
        executable_paths == sorted(set(executable_paths)) and compiler in executable_inputs,
        f"{label}.host_manifest executable inputs are unordered or omit the compiler",
    )
    include_roots = [
        require_absolute_posix_path(row, f"{label}.host_manifest.include_roots[{index}]")
        for index, row in enumerate(
            require_list(host_manifest.get("include_roots"), f"{label}.host_manifest.include_roots")
        )
    ]
    discovery_roots = [
        require_absolute_posix_path(row, f"{label}.host_manifest.discovery_roots[{index}]")
        for index, row in enumerate(
            require_list(
                host_manifest.get("discovery_roots"),
                f"{label}.host_manifest.discovery_roots",
            )
        )
    ]
    require(
        bool(include_roots) and len(include_roots) == len(set(include_roots)),
        f"{label}.host_manifest include roots must be non-empty and unique",
    )
    require(
        discovery_roots == sorted(set(discovery_roots)) and bool(discovery_roots),
        f"{label}.host_manifest discovery roots must be sorted, unique, and non-empty",
    )
    require(
        all(
            PurePosixPath(path).parent.as_posix() in discovery_roots
            for path in executable_paths
        )
        and all(
            any(PurePosixPath(path).parent.as_posix() == root for path in executable_paths)
            for root in discovery_roots
        ),
        f"{label}.host_manifest does not bind executable inputs to discovery roots",
    )

    host_files = require_list(host_manifest.get("files"), f"{label}.host_manifest.files")
    require(bool(host_files), f"{label}.host_manifest.files must not be empty")
    host_logical_paths: list[str] = []
    ownership_roots = [PurePosixPath(root) for root in (*include_roots, *discovery_roots)]
    for index, raw in enumerate(host_files):
        row = require_dict(raw, f"{label}.host_manifest.files[{index}]")
        require(
            set(row) == {"logical_path", "resolved_path", "sha256", "size_bytes"},
            f"{label}.host_manifest.files[{index}] shape mismatch",
        )
        logical = require_absolute_posix_path(
            row.get("logical_path"), f"{label}.host_manifest.files[{index}].logical_path"
        )
        resolved = require_absolute_posix_path(
            row.get("resolved_path"), f"{label}.host_manifest.files[{index}].resolved_path"
        )
        digest = require_sha(row.get("sha256"), f"{label}.host_manifest.files[{index}].sha256")
        require(
            isinstance(row.get("size_bytes"), int) and row["size_bytes"] >= 0,
            f"{label}.host_manifest.files[{index}].size_bytes is invalid",
        )
        require(
            any(PurePosixPath(logical).is_relative_to(root) for root in ownership_roots),
            f"{label}.host_manifest.files[{index}] is outside declared roots",
        )
        host_logical_paths.append(logical)
        identity = ("host_toolchain", logical, digest)
        allowed.add(identity)
        for absolute in (logical, resolved):
            insert_toolchain_owner(
                owners,
                absolute,
                identity,
                f"{label}.host_manifest.files[{index}]",
            )
    require(
        host_logical_paths == sorted(set(host_logical_paths)),
        f"{label}.host_manifest files must be strictly sorted by logical_path",
    )
    return owners, allowed


def verify_source_plan_dependencies(
    plan: Any, label: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[set[DependencyIdentity]]]:
    value = require_dict(plan, label)
    translation_units = [
        require_dict(row, f"{label}.translation_units[{index}]")
        for index, row in enumerate(
            require_list(value.get("translation_units"), f"{label}.translation_units")
        )
    ]
    headers = [
        require_dict(row, f"{label}.headers[{index}]")
        for index, row in enumerate(require_list(value.get("headers"), f"{label}.headers"))
    ]
    for collection_name, rows in (
        ("translation_units", translation_units),
        ("headers", headers),
    ):
        for index, row in enumerate(rows):
            require(
                set(row) == {"path", "sha256"},
                f"{label}.{collection_name}[{index}] shape mismatch",
            )
            require_relative_posix_path(
                row.get("path"), f"{label}.{collection_name}[{index}].path"
            )
            require_sha(row.get("sha256"), f"{label}.{collection_name}[{index}].sha256")
        paths = [row["path"] for row in rows]
        require(
            paths == sorted(set(paths)) and (collection_name != "translation_units" or bool(paths)),
            f"{label}.{collection_name} must be sorted and unique",
        )
    header_by_path = {row["path"]: row for row in headers}
    closures = [
        require_dict(row, f"{label}.dependency_closures[{index}]")
        for index, row in enumerate(
            require_list(value.get("dependency_closures"), f"{label}.dependency_closures")
        )
    ]
    require(
        len(closures) == len(translation_units),
        f"{label} must contain one dependency closure per translation unit",
    )
    expected_by_unit: list[set[DependencyIdentity]] = []
    for index, (translation_unit, closure) in enumerate(zip(translation_units, closures)):
        require(
            set(closure) == {
                "translation_unit",
                "headers",
                "closure_sha256",
            }
            and closure.get("translation_unit") == translation_unit["path"],
            f"{label}.dependency_closures[{index}] identity mismatch",
        )
        closure_headers = [
            require_dict(row, f"{label}.dependency_closures[{index}].headers[{header_index}]")
            for header_index, row in enumerate(
                require_list(
                    closure.get("headers"),
                    f"{label}.dependency_closures[{index}].headers",
                )
            )
        ]
        closure_paths = [row.get("path") for row in closure_headers]
        require(
            closure_paths == sorted(set(closure_paths))
            and all(header_by_path.get(row.get("path")) == row for row in closure_headers),
            f"{label}.dependency_closures[{index}] contains an unknown or unordered header",
        )
        closure_payload = {
            "translation_unit": translation_unit,
            "headers": closure_headers,
        }
        closure_digest = hashlib.sha256(
            json.dumps(
                closure_payload,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        require(
            closure.get("closure_sha256") == closure_digest,
            f"{label}.dependency_closures[{index}] SHA256 mismatch",
        )
        expected = {
            ("source", translation_unit["path"], translation_unit["sha256"]),
            *(
                ("source", row["path"], row["sha256"])
                for row in closure_headers
            ),
        }
        expected_by_unit.append(expected)
    return translation_units, closures, expected_by_unit


def parse_make_words(value: str, label: str) -> list[str]:
    words: list[str] = []
    word: list[str] = []
    escaped = False
    for character in value:
        if escaped:
            word.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character.isspace():
            if word:
                words.append("".join(word))
                word = []
        else:
            word.append(character)
    require(not escaped, f"{label} ends with an incomplete escape")
    if word:
        words.append("".join(word))
    return words


def parse_make_depfile(raw: str, label: str) -> tuple[str, list[str]]:
    require(bool(raw.strip()) and "\0" not in raw, f"{label} is empty or contains NUL")
    normalized = raw.replace("\\\r\n", "").replace("\\\n", "").rstrip("\r\n")
    require("\n" not in normalized and "\r" not in normalized, f"{label} has multiple rules")
    escaped = False
    delimiter: int | None = None
    for index, character in enumerate(normalized):
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ":":
            delimiter = index
            break
    require(delimiter is not None, f"{label} has no target delimiter")
    targets = parse_make_words(normalized[:delimiter], label)
    dependencies = parse_make_words(normalized[delimiter + 1 :], label)
    require(
        len(targets) == 1 and bool(dependencies),
        f"{label} must contain exactly one target and at least one dependency",
    )
    return targets[0], dependencies


def verify_translation_unit_dependency_evidence(
    command: dict[str, Any],
    depfile_path: Path,
    expected_source: set[DependencyIdentity],
    toolchain_owners: dict[str, DependencyIdentity],
    allowed_toolchain: set[DependencyIdentity],
    label: str,
) -> list[DependencyIdentity]:
    working_directory = require_absolute_posix_path(
        command.get("depfile_producer_working_directory"),
        f"{label}.depfile_producer_working_directory",
    )
    producer_object = require_absolute_posix_path(
        command.get("depfile_producer_object_file"),
        f"{label}.depfile_producer_object_file",
    )
    current_object = require_absolute_posix_path(command.get("object_file"), f"{label}.object_file")
    require(
        PurePosixPath(producer_object).name == PurePosixPath(current_object).name,
        f"{label} producer object basename differs from the current object",
    )
    observed = verify_dependency_rows(command.get("observed_dependencies"), f"{label}.observed")
    observed_source = {row for row in observed if row[0] == "source"}
    require(
        observed_source == expected_source,
        f"{label} source dependencies differ from the exact translation-unit closure",
    )
    require(
        all(row[0] == "source" or row in allowed_toolchain for row in observed),
        f"{label} contains a dependency absent from the toolchain manifests",
    )

    try:
        raw = depfile_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise VerificationError(f"cannot read {label} depfile: {error}") from error
    target, dependencies = parse_make_depfile(raw, f"{label}.depfile")
    require(target == producer_object, f"{label} depfile target differs from its producer object")

    expected_by_path = {row[1]: row for row in expected_source}
    working = PurePosixPath(working_directory)
    parsed: set[DependencyIdentity] = set()
    for raw_dependency in dependencies:
        path = PurePosixPath(raw_dependency)
        if path.is_absolute() and not path.is_relative_to(working):
            identity = toolchain_owners.get(path.as_posix())
            require(
                identity is not None,
                f"{label} depfile contains an unmanifested external dependency: {raw_dependency}",
            )
        else:
            if path.is_absolute():
                relative_raw = path.relative_to(working).as_posix()
            else:
                relative_raw = raw_dependency
            relative = normalize_depfile_relative_path(
                relative_raw, f"{label}.depfile dependency"
            )
            identity = expected_by_path.get(relative)
            require(
                identity is not None,
                f"{label} depfile contains an undeclared source dependency: {relative}",
            )
        require(identity not in parsed, f"{label} depfile contains a duplicate dependency")
        parsed.add(identity)
    require(
        {row for row in parsed if row[0] == "source"} == expected_source,
        f"{label} depfile differs from the exact translation-unit closure",
    )
    parsed_rows = sorted(parsed, key=dependency_sort_key)
    require(
        parsed_rows == observed,
        f"{label} raw depfile semantics differ from typed receipt evidence",
    )
    return observed


def collect_artifact_index(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"artifact tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"chain.manifest.json", "failure.json"}:
            continue
        rows.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def verify_artifact_index(root: Path, manifest: dict[str, Any]) -> None:
    expected = require_list(manifest.get("artifacts"), "manifest.artifacts")
    require(
        manifest.get("artifact_count") == len(expected),
        "manifest artifact_count differs from its index",
    )
    require(expected == collect_artifact_index(root), "artifact directory index mismatch")


def verify_source(root: Path, source_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    source = require_dict(manifest.get("source"), "manifest.source")
    require(source == read_json(root / "source.json", "source identity"), "source identity copies differ")
    require(source.get("dirty") is False and source.get("status_short") == [], "artifact source is dirty")
    require(
        GIT_SHA_RE.fullmatch(str(source.get("git_sha", ""))) is not None,
        "artifact source Git SHA is invalid",
    )
    require(
        GIT_SHA_RE.fullmatch(str(source.get("git_tree_sha", ""))) is not None,
        "artifact source tree SHA is invalid",
    )
    require(source_root.is_dir(), f"source root is missing: {source_root}")
    actual = {
        "git_sha": run_text(source_root, ["git", "rev-parse", "HEAD"]),
        "git_tree_sha": run_text(source_root, ["git", "rev-parse", "HEAD^{tree}"]),
        "dirty": False,
        "status_short": run_text(
            source_root, ["git", "status", "--short", "--untracked-files=all"]
        ).splitlines(),
    }
    require(not actual["status_short"], f"verification source is dirty: {actual['status_short']}")
    require(actual == source, "artifact source differs from the verification checkout")
    lane_plan = require_dict(read_json(root / "lane-plan.json", "lane plan"), "lane plan")
    require(lane_plan.get("schema_version") == 1, "lane plan schema mismatch")
    require(lane_plan.get("lane") == "runtime-vnext-g07b-native-chain", "lane identity mismatch")
    require(lane_plan.get("source") == source, "lane plan source identity mismatch")
    require(
        isinstance(lane_plan.get("expected_runtime_seconds"), int)
        and lane_plan["expected_runtime_seconds"] > 0,
        "lane expected runtime is invalid",
    )
    require(
        isinstance(lane_plan.get("hard_deadline_seconds"), int)
        and lane_plan["hard_deadline_seconds"] >= lane_plan["expected_runtime_seconds"],
        "lane hard deadline is invalid",
    )
    for field in ("hard_stop", "correctness_gate", "performance_command", "progress_signal"):
        require(
            isinstance(lane_plan.get(field), str) and lane_plan[field].strip(),
            f"lane plan {field} is missing",
        )
    return source


def verify_hardware(root: Path, manifest: dict[str, Any]) -> None:
    hardware = require_dict(manifest.get("hardware"), "manifest.hardware")
    require(hardware == read_json(root / "hardware.json", "hardware identity"), "hardware copies differ")
    require(hardware.get("gpu_count") == 1, "hardware must contain exactly one GPU")
    require("RTX 4090" in str(hardware.get("gpu", "")), "hardware is not an RTX 4090")
    require("RTX 4090" in str(hardware.get("nvidia_smi", "")), "nvidia-smi evidence is not RTX 4090")
    require("release" in str(hardware.get("nvcc_version", "")).lower(), "nvcc version evidence is missing")
    tools = require_dict(hardware.get("tools"), "hardware.tools")
    require(set(tools) == {"nvcc", "ccbin", "cc", "ar"}, "hardware tool set mismatch")
    for name, identity in tools.items():
        row = require_dict(identity, f"hardware.tools.{name}")
        require(isinstance(row.get("path"), str) and Path(row["path"]).is_absolute(), f"{name} path is invalid")
        require_sha(row.get("sha256"), f"hardware.tools.{name}.sha256")


def verify_step(root: Path, step_id: str) -> None:
    step_root = root / "steps" / step_id
    require(step_root.is_dir() and not step_root.is_symlink(), f"step is missing: {step_id}")
    plan = require_dict(read_json(step_root / "plan.json", f"{step_id} plan"), f"{step_id} plan")
    receipt = require_dict(
        read_json(step_root / "bounded.receipt.json", f"{step_id} receipt"),
        f"{step_id} receipt",
    )
    require(plan.get("schema_version") == 1 and plan.get("step_id") == step_id, f"{step_id} plan identity mismatch")
    command = plan.get("command")
    require(
        isinstance(command, list)
        and bool(command)
        and all(isinstance(part, str) and bool(part) for part in command),
        f"{step_id} command is invalid",
    )
    expected = plan.get("expected_duration_seconds")
    deadline = plan.get("hard_deadline_seconds")
    if (
        not isinstance(expected, int)
        or expected <= 0
        or not isinstance(deadline, int)
        or deadline < expected
    ):
        raise VerificationError(f"{step_id} duration contract is invalid")
    require(
        isinstance(plan.get("progress_signal"), str)
        and bool(plan["progress_signal"].strip()),
        f"{step_id} progress signal is missing",
    )
    require(receipt.get("schema") == RECEIPT_SCHEMA, f"{step_id} receipt schema mismatch")
    require(
        receipt.get("command") == command and receipt.get("cwd") == plan.get("cwd"),
        f"{step_id} receipt command/cwd differs from its plan",
    )
    require(
        receipt.get("status") == "pass"
        and receipt.get("rc") == 0
        and receipt.get("violation") is None
        and receipt.get("cleanup") == {"process_group_gone": True},
        f"{step_id} bounded execution did not pass cleanly",
    )
    duration = receipt.get("duration_seconds")
    if not isinstance(duration, (int, float)):
        raise VerificationError(f"{step_id} duration is invalid")
    require(0 <= duration <= deadline + 5, f"{step_id} duration exceeds its deadline")
    limits = require_dict(receipt.get("limits"), f"{step_id}.limits")
    require(
        limits.get("max_processes") == 96
        and limits.get("max_group_threads") == 256
        and limits.get("max_per_process_threads") == 64
        and limits.get("wall_timeout_seconds") == float(deadline),
        f"{step_id} worker bounds differ from the lane contract",
    )
    require(
        isinstance(receipt.get("sampling_error_count"), int)
        and receipt["sampling_error_count"] <= limits.get("max_sampling_errors", -1),
        f"{step_id} sampling errors exceed their bound",
    )
    for stream in ("stdout", "stderr"):
        path = step_root / f"{stream}.log"
        verify_file_identity(path, receipt.get(stream), f"{step_id}.{stream}")


def verify_steps(root: Path) -> None:
    steps_root = root / "steps"
    require(steps_root.is_dir() and not steps_root.is_symlink(), "steps directory is missing")
    actual = {path.name for path in steps_root.iterdir() if path.is_dir()}
    require(actual == EXPECTED_STEPS, f"step set mismatch: {sorted(actual ^ EXPECTED_STEPS)}")
    for step_id in sorted(EXPECTED_STEPS):
        verify_step(root, step_id)


def verify_version(value: Any, label: str) -> None:
    version = require_dict(value, label)
    require(set(version) == {"major", "minor"}, f"{label} shape mismatch")
    require(
        isinstance(version["major"], int)
        and version["major"] > 0
        and isinstance(version["minor"], int)
        and version["minor"] >= 0,
        f"{label} is invalid",
    )


def verify_provider_catalog(catalog: Any) -> dict[tuple[str, str], dict[str, Any]]:
    value = require_dict(catalog, "provider catalog")
    require(value.get("schema_version") == 1 and value.get("backend") == "cuda", "provider catalog identity mismatch")
    providers = require_list(value.get("providers"), "provider catalog providers")
    require(bool(providers), "provider catalog providers must not be empty")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    keys: list[tuple[str, str]] = []
    for index, raw in enumerate(providers):
        row = require_dict(raw, f"providers[{index}]")
        operation_id = row.get("operation_id")
        provider_id = row.get("provider_id")
        if (
            not isinstance(operation_id, str)
            or not operation_id.startswith("operation.")
            or not isinstance(provider_id, str)
            or not provider_id.startswith("provider.cuda.")
        ):
            raise VerificationError(f"providers[{index}] identifiers are invalid")
        verify_version(row.get("operation_contract_version"), f"providers[{index}].operation_contract_version")
        verify_version(row.get("provider_version"), f"providers[{index}].provider_version")
        require_sha(row.get("operation_fingerprint"), f"providers[{index}].operation_fingerprint")
        require_sha(
            row.get("provider_implementation_fingerprint"),
            f"providers[{index}].provider_implementation_fingerprint",
        )
        key = (operation_id, provider_id)
        require(key not in result, f"duplicate provider row: {key}")
        result[key] = row
        keys.append(key)
    require(keys == sorted(keys), "provider catalog is not sorted")
    return result


def verify_binding(binding: Any, providers: dict[tuple[str, str], dict[str, Any]], label: str) -> None:
    row = require_dict(binding, label)
    operation_id = row.get("operation_id")
    provider_id = row.get("provider_id")
    if not isinstance(operation_id, str) or not isinstance(provider_id, str):
        raise VerificationError(f"{label} identifiers are invalid")
    key = (operation_id, provider_id)
    require(key in providers, f"{label} is absent from the live provider catalog: {key}")
    provider = providers[key]
    for field in (
        "operation_contract_version",
        "provider_version",
        "provider_implementation_fingerprint",
    ):
        require(row.get(field) == provider.get(field), f"{label}.{field} differs from the live catalog")
    entrypoints = require_list(row.get("entrypoints"), f"{label}.entrypoints")
    require(
        bool(entrypoints)
        and all(isinstance(value, str) and bool(value) for value in entrypoints)
        and entrypoints == sorted(set(entrypoints)),
        f"{label}.entrypoints must be sorted, unique, and non-empty",
    )


def verify_source_build(
    root: Path,
    source_root: Path,
    name: str,
    operator: str,
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    build_root = root / "source-builds" / name
    receipt = require_dict(
        read_json(build_root / "source-build.receipt.json", f"{name} source receipt"),
        f"{name} source receipt",
    )
    plan_path = source_root / f"native-operators/cuda/source-locks/{name}.plan.json"
    plan = require_dict(read_json(plan_path, f"{name} source plan"), f"{name} source plan")
    require(
        receipt.get("schema_version") == SOURCE_BUILD_RECEIPT_SCHEMA_VERSION
        and receipt.get("status") == "pass"
        and receipt.get("plan_only") is False
        and receipt.get("failure_class") is None,
        f"{name} source build is not a terminal schema-v5 PASS",
    )
    require(plan.get("schema_version") == 3, f"{name} source plan schema mismatch")
    require(receipt.get("operator") == operator == plan.get("operator"), f"{name} source operator mismatch")
    require(receipt.get("plan_sha256") == sha256(plan_path), f"{name} source plan pin mismatch")
    require(receipt.get("source_package") == plan.get("source_package"), f"{name} source package mismatch")
    require(receipt.get("builder_sha") == source["git_sha"], f"{name} builder SHA mismatch")
    require(receipt.get("compute_capability") == "sm_89", f"{name} compute capability mismatch")
    require(
        receipt.get("nvcc_threads") == 4
        and isinstance(receipt.get("elapsed_ms"), int)
        and receipt["elapsed_ms"] >= 0,
        f"{name} source build execution metadata is invalid",
    )
    require(isinstance(receipt.get("toolchain"), dict), f"{name} source toolchain evidence is missing")
    static_identity = require_dict(receipt["toolchain"].get("static_identity"), f"{name}.toolchain.static_identity")
    cuda_toolkit = require_dict(static_identity.get("cuda_toolkit"), f"{name}.toolchain.cuda_toolkit")
    host_toolchain = require_dict(static_identity.get("host_toolchain"), f"{name}.toolchain.host_toolchain")
    cuda_manifest_path = verify_evidence(
        build_root, cuda_toolkit.get("manifest"), f"{name}.cuda_toolkit.manifest"
    )
    host_manifest_path = verify_evidence(
        build_root, host_toolchain.get("manifest"), f"{name}.host_toolchain.manifest"
    )
    toolchain_owners, allowed_toolchain = verify_toolchain_manifests(
        static_identity,
        read_json(cuda_manifest_path, f"{name} CUDA toolkit manifest"),
        read_json(host_manifest_path, f"{name} host toolchain manifest"),
        f"{name}.toolchain",
    )
    translation_units, closures, expected_dependencies = verify_source_plan_dependencies(
        plan, f"{name}.plan"
    )
    locked_source_root = source_root / "crates/ferrum-kernels"
    for collection in (translation_units, require_list(plan.get("headers"), f"{name}.plan.headers")):
        for row in collection:
            path = resolve_relative_file(
                locked_source_root,
                row.get("path"),
                f"{name}.locked_source.{row.get('path')}",
            )
            require(
                sha256(path) == row.get("sha256"),
                f"{name} locked source SHA mismatch: {row.get('path')}",
            )

    commands = require_list(receipt.get("commands"), f"{name}.commands")
    require(
        len(commands) == len(translation_units) + 1,
        f"{name} must contain one command per translation unit plus one archive command",
    )
    observed_compiled: list[str] = []
    observed_hits: list[str] = []
    expected_working_directory: str | None = None
    for index, (translation_unit, closure, expected_source) in enumerate(
        zip(translation_units, closures, expected_dependencies)
    ):
        raw = commands[index]
        command = require_dict(raw, f"{name}.commands[{index}]")
        require(
            command.get("translation_unit") == translation_unit["path"]
            and command.get("dependency_closure_sha256") == closure["closure_sha256"],
            f"{name}.commands[{index}] does not bind its exact translation-unit closure",
        )
        working_directory = require_absolute_posix_path(
            command.get("working_directory"), f"{name}.commands[{index}].working_directory"
        )
        if expected_working_directory is None:
            expected_working_directory = working_directory
        require(
            working_directory == expected_working_directory,
            f"{name}.commands[{index}] working directory differs from the build plan",
        )
        object_file = command.get("object_file")
        require_absolute_posix_path(object_file, f"{name}.commands[{index}].object_file")
        require_sha(command.get("object_cache_key"), f"{name}.commands[{index}].object_cache_key")
        require(
            isinstance(command.get("object_cache_entry"), str)
            and bool(command["object_cache_entry"]),
            f"{name}.commands[{index}] object cache entry is missing",
        )
        require_sha(command.get("object_sha256"), f"{name}.commands[{index}].object_sha256")
        require(
            isinstance(command.get("object_size_bytes"), int)
            and command["object_size_bytes"] > 0
            and isinstance(command.get("object_identity"), dict)
            and isinstance(command.get("elapsed_ms"), int)
            and command["elapsed_ms"] >= 0,
            f"{name}.commands[{index}] object identity/timing evidence is incomplete",
        )
        if command.get("compiler_executed") is True:
            require(
                command.get("return_code") == 0
                and command.get("object_cache_status") == "published"
                and command.get("dependency_validation") == "depfile",
                f"{name}.commands[{index}] compiler execution mismatch",
            )
            observed_compiled.append(translation_unit["path"])
        else:
            require(
                command.get("compiler_executed") is False
                and command.get("return_code") is None
                and command.get("object_cache_status") == "hit"
                and command.get("dependency_validation") == "cache_proof",
                f"{name}.commands[{index}] cache-hit execution mismatch",
            )
            observed_hits.append(translation_unit["path"])
        for stream in ("stdout_log", "stderr_log"):
            path = resolve_relative_file(build_root, command.get(stream), f"{name}.commands[{index}].{stream}")
            require(path.stat().st_size > 0, f"{name}.commands[{index}].{stream} is empty")
        depfile = command.get("depfile")
        depfile_path = resolve_relative_file(
            build_root, depfile, f"{name}.commands[{index}].depfile"
        )
        require_sha(command.get("depfile_sha256"), f"{name}.commands[{index}].depfile_sha256")
        require(
            sha256(depfile_path) == command["depfile_sha256"],
            f"{name}.commands[{index}] depfile SHA mismatch",
        )
        verify_translation_unit_dependency_evidence(
            command,
            depfile_path,
            expected_source,
            toolchain_owners,
            allowed_toolchain,
            f"{name}.commands[{index}]",
        )
        if command.get("object_cache_status") == "published":
            require(
                command.get("depfile_producer_working_directory") == working_directory
                and command.get("depfile_producer_object_file") == object_file,
                f"{name}.commands[{index}] published depfile producer identity mismatch",
            )
        object_path = build_root / "objects" / PurePosixPath(object_file).name
        require(
            object_path.is_file() and not object_path.is_symlink(),
            f"{name} object is missing: {object_path}",
        )
        require(
            sha256(object_path) == command["object_sha256"],
            f"{name}.commands[{index}] object SHA mismatch",
        )
        require(
            object_path.stat().st_size == command.get("object_size_bytes"),
            f"{name}.commands[{index}] object size mismatch",
        )

    archive_command = require_dict(commands[-1], f"{name}.commands.archive")
    require(
        archive_command.get("translation_unit") is None
        and archive_command.get("object_file") is None
        and archive_command.get("object_cache_key") is None
        and archive_command.get("object_cache_status") is None
        and archive_command.get("object_cache_entry") is None
        and archive_command.get("object_sha256") is None
        and archive_command.get("object_size_bytes") is None
        and archive_command.get("object_identity") is None
        and archive_command.get("dependency_closure_sha256") is None
        and archive_command.get("dependency_validation") is None
        and archive_command.get("depfile") is None
        and archive_command.get("depfile_sha256") is None
        and archive_command.get("depfile_producer_working_directory") is None
        and archive_command.get("depfile_producer_object_file") is None
        and archive_command.get("observed_dependencies") == []
        and archive_command.get("compiler_executed") is False
        and archive_command.get("return_code") == 0
        and isinstance(archive_command.get("elapsed_ms"), int)
        and archive_command["elapsed_ms"] >= 0,
        f"{name} archive command is not terminal",
    )
    require(
        archive_command.get("working_directory") == expected_working_directory,
        f"{name} archive command working directory mismatch",
    )
    for stream in ("stdout_log", "stderr_log"):
        path = resolve_relative_file(
            build_root, archive_command.get(stream), f"{name}.commands.archive.{stream}"
        )
        require(path.stat().st_size > 0, f"{name}.commands.archive.{stream} is empty")

    compiled = require_list(receipt.get("compiled_translation_units"), f"{name}.compiled_translation_units")
    hits = require_list(receipt.get("cache_hit_translation_units"), f"{name}.cache_hit_translation_units")
    require(
        compiled == observed_compiled
        and hits == observed_hits
        and compiled == sorted(set(compiled))
        and hits == sorted(set(hits)),
        f"{name} compiled/cache-hit summaries differ from command evidence",
    )
    miss_probe = receipt["toolchain"].get("miss_probe")
    if compiled:
        probe = require_dict(miss_probe, f"{name}.toolchain.miss_probe")
        require(
            probe.get("probed_for_misses") == compiled
            and all(
                isinstance(probe.get(field), str) and bool(probe[field].strip())
                for field in (
                    "nvcc_version",
                    "host_compiler_version",
                    "host_target",
                    "archiver_version",
                )
            ),
            f"{name} miss-only toolchain probe differs from compiled commands",
        )
    else:
        require(miss_probe is None, f"{name} cache-only build unexpectedly ran a miss probe")
    archive_file = receipt.get("archive_file")
    require(archive_file == plan.get("archive_file"), f"{name} archive filename mismatch")
    archive = resolve_relative_file(build_root, archive_file, f"{name}.archive")
    require_sha(receipt.get("archive_sha256"), f"{name}.archive_sha256")
    require(sha256(archive) == receipt["archive_sha256"], f"{name} source archive SHA mismatch")
    return receipt, plan


def verify_package(
    root: Path,
    source_root: Path,
    name: str,
    operator: str,
    source_receipt: dict[str, Any],
    source_plan: dict[str, Any],
    provider_bytes: bytes,
    provider_sha: str,
    abi_bytes: bytes,
    abi_sha: str,
    providers: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    package_root = root / "packages" / name
    receipt = require_dict(
        read_json(package_root / "package.receipt.json", f"{name} package receipt"),
        f"{name} package receipt",
    )
    spec_path = root / "package-specs" / f"{name}.json"
    spec = require_dict(read_json(spec_path, f"{name} package spec"), f"{name} package spec")
    require(
        spec.get("schema_version") == 3
        and spec.get("operator") == operator
        and spec.get("backend") == "cuda"
        and spec.get("compute_capabilities") == ["sm_89"],
        f"{name} materialized package spec identity mismatch",
    )
    bindings = require_list(spec.get("operation_bindings"), f"{name}.operation_bindings")
    for index, binding in enumerate(bindings):
        verify_binding(binding, providers, f"{name}.operation_bindings[{index}]")

    require(receipt.get("schema_version") == 5 and receipt.get("operator") == operator, f"{name} package receipt identity mismatch")
    require(
        receipt.get("g03_catalog_sha256") == provider_sha
        and receipt.get("abi_contract_sha256") == abi_sha,
        f"{name} package catalog/ABI pins mismatch",
    )
    required_evidence = (
        "package_spec",
        "g03_catalog",
        "abi_contract",
        "source_build_receipt",
        "source_build_plan",
        "source_archive_verification",
        "final_archive_verification",
    )
    resolved = {
        field: verify_evidence(package_root, receipt.get(field), f"{name}.{field}")
        for field in required_evidence
    }
    for field in ("source_build_inputs", "source_build_logs", "package_build_logs", "license_files"):
        verify_evidence_list(package_root, receipt.get(field), f"{name}.{field}", nonempty=True)
    require(resolved["package_spec"].read_bytes() == spec_path.read_bytes(), f"{name} packaged spec differs")
    require(resolved["g03_catalog"].read_bytes() == provider_bytes, f"{name} packaged provider catalog differs")
    require(resolved["abi_contract"].read_bytes() == abi_bytes, f"{name} packaged ABI contract differs")
    require(
        read_json(resolved["source_build_receipt"], f"{name} packaged source receipt") == source_receipt,
        f"{name} packaged source receipt differs",
    )
    require(
        read_json(resolved["source_build_plan"], f"{name} packaged source plan") == source_plan,
        f"{name} packaged source plan differs",
    )
    require(
        receipt.get("source_archive_sha256") == source_receipt.get("archive_sha256"),
        f"{name} package source archive pin mismatch",
    )
    manifest_path = resolve_relative_file(package_root, receipt.get("manifest_file"), f"{name}.manifest")
    artifact_path = resolve_relative_file(package_root, receipt.get("artifact_file"), f"{name}.artifact")
    require_sha(receipt.get("manifest_sha256"), f"{name}.manifest_sha256")
    require_sha(receipt.get("binary_sha256"), f"{name}.binary_sha256")
    require(sha256(manifest_path) == receipt["manifest_sha256"], f"{name} manifest SHA mismatch")
    require(sha256(artifact_path) == receipt["binary_sha256"], f"{name} artifact SHA mismatch")
    package_commands = require_list(receipt.get("package_commands"), f"{name}.package_commands")
    require(len(package_commands) == 2, f"{name} must contain exactly two package commands")
    require(all(require_dict(row, f"{name}.package_command").get("return_code") == 0 for row in package_commands), f"{name} package command failed")

    manifest = require_dict(read_json(manifest_path, f"{name} manifest"), f"{name} manifest")
    require(
        manifest.get("schema_version") == 3
        and manifest.get("operator") == operator
        and manifest.get("backend") == "cuda"
        and manifest.get("linkage") == "static"
        and manifest.get("compute_capabilities") == ["sm_89"],
        f"{name} native manifest identity mismatch",
    )
    require(
        manifest.get("ferrum_native_abi_version") == "2"
        and manifest.get("g03_catalog_sha256") == provider_sha
        and manifest.get("abi_contract_sha256") == abi_sha,
        f"{name} native manifest ABI/catalog mismatch",
    )
    require(
        manifest.get("source_package") == source_receipt.get("source_package")
        and manifest.get("inputs_sha256") == source_receipt.get("inputs_sha256")
        and manifest.get("binary_sha256") == receipt.get("binary_sha256")
        and manifest.get("operation_bindings") == bindings,
        f"{name} manifest is not the source/spec projection",
    )
    return receipt, spec, manifest


def verify_lock_and_inventory(
    root: Path,
    packages: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    provider_sha: str,
    abi_sha: str,
    artifact_inventory: list[Any],
) -> None:
    lock_path = root / "native-operators.lock.json"
    lock = require_dict(read_json(lock_path, "artifact-set lock"), "artifact-set lock")
    artifacts = require_list(lock.get("artifacts"), "artifact-set artifacts")
    require(
        lock.get("schema_version") == 5
        and lock.get("g03_catalog_sha256") == provider_sha
        and len(artifacts) == 4,
        "artifact-set identity mismatch",
    )
    lock_by_operator: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(artifacts):
        row = require_dict(raw, f"artifact-set[{index}]")
        operator = row.get("operator")
        if (
            not isinstance(operator, str)
            or operator not in PACKAGES.values()
            or operator in lock_by_operator
        ):
            raise VerificationError(f"invalid lock operator: {operator}")
        lock_by_operator[operator] = row
        for field in (
            "manifest",
            "package_spec",
            "g03_catalog",
            "abi_contract",
            "source_build_receipt",
            "source_build_plan",
            "package_receipt",
        ):
            verify_evidence(root, row.get(field), f"lock.{operator}.{field}")
        for field in (
            "source_build_inputs",
            "source_build_logs",
            "package_build_logs",
            "license_files",
        ):
            verify_evidence_list(root, row.get(field), f"lock.{operator}.{field}", nonempty=True)
        manifest_path = resolve_relative_file(root, row.get("manifest_path"), f"lock.{operator}.manifest_path")
        artifact_path = resolve_relative_file(root, row.get("artifact_path"), f"lock.{operator}.artifact_path")
        name = next(package_name for package_name, expected in PACKAGES.items() if expected == operator)
        receipt, spec, manifest = packages[name]
        require(read_json(manifest_path, f"lock {operator} manifest") == manifest, f"lock {operator} manifest differs")
        require(sha256(artifact_path) == receipt["binary_sha256"], f"lock {operator} artifact SHA mismatch")
        require(
            row.get("binary_sha256") == receipt["binary_sha256"]
            and row.get("abi_contract_sha256") == abi_sha
            and row.get("operation_bindings") == spec["operation_bindings"],
            f"lock {operator} semantic pins mismatch",
        )
    require(set(lock_by_operator) == set(PACKAGES.values()), "artifact-set operator coverage mismatch")
    require([row["operator"] for row in artifacts] == sorted(PACKAGES.values()), "artifact-set is not sorted")

    inventory_by_operator: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(artifact_inventory):
        row = require_dict(raw, f"artifact inventory[{index}]")
        operator = row.get("operator")
        if (
            not isinstance(operator, str)
            or operator not in lock_by_operator
            or operator in inventory_by_operator
        ):
            raise VerificationError(f"invalid compiled operator: {operator}")
        inventory_by_operator[operator] = row
        lock_row = lock_by_operator[operator]
        for field in (
            "operator_abi_version",
            "ferrum_native_abi_version",
            "backend",
            "linkage",
            "g03_catalog_sha256",
            "abi_contract_sha256",
            "descriptor_export",
            "operation_bindings",
            "required_exports",
            "source_package_sha256",
            "inputs_sha256",
            "binary_sha256",
        ):
            inventory_field = "exports" if field == "required_exports" else field
            require(
                row.get(inventory_field) == lock_row.get(field),
                f"compiled inventory {operator}.{inventory_field} differs from the lock",
            )
        require(row.get("schema_version") == 3, f"compiled inventory {operator} schema mismatch")
    require(set(inventory_by_operator) == set(PACKAGES.values()), "compiled inventory coverage mismatch")


def verify_catalogs_and_packages(
    root: Path,
    source_root: Path,
    manifest: dict[str, Any],
    source: dict[str, Any],
) -> None:
    bootstrap_provider = root / "bootstrap/provider-catalog.json"
    artifact_provider = root / "artifact/provider-catalog.json"
    bootstrap_capability = root / "bootstrap/capability-catalog.json"
    artifact_capability = root / "artifact/capability-catalog.json"
    bootstrap_inventory_path = root / "bootstrap/compiled-native-operators.json"
    artifact_inventory_path = root / "artifact/compiled-native-operators.json"
    for path in (
        bootstrap_provider,
        artifact_provider,
        bootstrap_capability,
        artifact_capability,
        bootstrap_inventory_path,
        artifact_inventory_path,
    ):
        require(path.is_file() and not path.is_symlink(), f"runtime export is missing: {path}")
    require(bootstrap_provider.read_bytes() == artifact_provider.read_bytes(), "provider catalogs differ")
    require(bootstrap_capability.read_bytes() == artifact_capability.read_bytes(), "capability catalogs differ")
    provider_bytes = bootstrap_provider.read_bytes()
    provider_sha = sha256(bootstrap_provider)
    providers = verify_provider_catalog(read_json(bootstrap_provider, "bootstrap provider catalog"))
    bootstrap_inventory = require_list(
        read_json(bootstrap_inventory_path, "bootstrap compiled inventory"),
        "bootstrap compiled inventory",
    )
    artifact_inventory = require_list(
        read_json(artifact_inventory_path, "artifact compiled inventory"),
        "artifact compiled inventory",
    )
    require(bootstrap_inventory == [], "bootstrap binary unexpectedly contains native artifacts")
    require(len(artifact_inventory) == 4, "artifact binary must contain exactly four native artifacts")

    abi_path = root / "contracts/ferrum-native-abi-v2.json"
    abi_bytes = abi_path.read_bytes()
    abi_sha = sha256(abi_path)
    source_abi = source_root / "native-operators/abi/ferrum-native-abi-v2.json"
    require(abi_bytes == source_abi.read_bytes(), "artifact ABI contract differs from the source checkout")
    abi = require_dict(read_json(abi_path, "ABI contract"), "ABI contract")
    require(
        abi
        == {
            "schema_version": 1,
            "ferrum_native_abi_version": "2",
            "descriptor_struct": "FerrumNativeOperatorDescriptorV2",
            "descriptor_symbol_policy": "operator_namespaced",
            "descriptor_fields": [
                {"name": "struct_size", "c_type": "uint32_t"},
                {"name": "ferrum_native_abi_version", "c_type": "uint32_t"},
                {"name": "operator_name", "c_type": "const char *"},
                {"name": "operator_abi_version", "c_type": "const char *"},
                {"name": "g03_catalog_sha256", "c_type": "const char *"},
                {"name": "abi_contract_sha256", "c_type": "const char *"},
            ],
        },
        "native ABI contract shape mismatch",
    )

    package_results: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for name, operator in PACKAGES.items():
        source_receipt, source_plan = verify_source_build(root, source_root, name, operator, source)
        package_results[name] = verify_package(
            root,
            source_root,
            name,
            operator,
            source_receipt,
            source_plan,
            provider_bytes,
            provider_sha,
            abi_bytes,
            abi_sha,
            providers,
        )
    verify_lock_and_inventory(
        root,
        package_results,
        provider_sha,
        abi_sha,
        artifact_inventory,
    )

    catalog = require_dict(manifest.get("catalog"), "manifest.catalog")
    require(
        catalog
        == {
            "provider_sha256": provider_sha,
            "provider_identity_unchanged": True,
            "capability_identity_unchanged": True,
            "bootstrap_native_operator_count": 0,
            "artifact_native_operator_count": 4,
        },
        "manifest catalog summary mismatch",
    )
    abi_summary = require_dict(manifest.get("abi_contract"), "manifest.abi_contract")
    require(abi_summary.get("sha256") == abi_sha, "manifest ABI summary mismatch")
    lock_summary = require_dict(manifest.get("artifact_set"), "manifest.artifact_set")
    require(
        lock_summary.get("sha256") == sha256(root / "native-operators.lock.json")
        and lock_summary.get("schema_version") == 5
        and lock_summary.get("operator_count") == 4,
        "manifest artifact-set summary mismatch",
    )


def verify_build_summaries(root: Path, manifest: dict[str, Any]) -> None:
    step_root = root / "steps/artifact-example-build"
    text = (
        (step_root / "stdout.log").read_text(encoding="utf-8", errors="replace")
        + "\n"
        + (step_root / "stderr.log").read_text(encoding="utf-8", errors="replace")
    )
    summaries = [
        {"artifact": artifact, "status": status, "reason": reason}
        for artifact, status, reason in CUDA_SUMMARY_RE.findall(text)
    ]
    require(manifest.get("artifact_build_summaries") == summaries, "manifest build summaries differ from logs")
    for unit in EXPECTED_BUILD_UNITS:
        rows = [row for row in summaries if row["artifact"] == unit]
        require(len(rows) == 1, f"artifact build summary count mismatch for {unit}")
        require(
            rows[0]["status"] == "artifact"
            and rows[0]["reason"] == "native-operator-artifact-set",
            f"native artifact {unit} fell back to source",
        )
    for step in ("bootstrap-catalog-export", "artifact-catalog-export"):
        output = (root / "steps" / step / "stdout.log").read_text(encoding="utf-8", errors="replace")
        require(
            output.count("FERRUM RUNTIME VNEXT CUDA LIVE CATALOG READY:") == 1,
            f"{step} did not emit exactly one runtime readiness line",
        )


def verify_manifest(root: Path, source_root: Path) -> None:
    root = root.resolve()
    source_root = source_root.resolve()
    require(not (root / "failure.json").exists(), "KEEP artifact also contains failure.json")
    manifest = require_dict(read_json(root / "chain.manifest.json", "chain manifest"), "chain manifest")
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type") == "runtime_vnext_g07b_native_chain_manifest"
        and manifest.get("status") == "keep",
        "chain manifest identity/status mismatch",
    )
    require(set(require_list(manifest.get("does_not_prove"), "manifest.does_not_prove")) == DOES_NOT_PROVE, "chain proof boundary mismatch")
    scope = require_dict(manifest.get("scope"), "manifest.scope")
    require(
        scope
        == {
            "backend": "cuda",
            "gpu_count": 1,
            "gpu_model": "RTX 4090",
            "compute_capability": "sm_89",
            "source_features": SOURCE_FEATURES,
            "artifact_features": ARTIFACT_FEATURES,
            "operators": sorted(PACKAGES.values()),
        },
        "chain scope mismatch",
    )
    verify_artifact_index(root, manifest)
    source = verify_source(root, source_root, manifest)
    verify_hardware(root, manifest)
    verify_steps(root)
    verify_catalogs_and_packages(root, source_root, manifest, source)
    verify_build_summaries(root, manifest)
    binaries = require_dict(manifest.get("binaries"), "manifest.binaries")
    for kind in ("bootstrap", "artifact"):
        path = root / f"binaries/{kind}/runtime_vnext_cuda_catalog"
        identity = require_dict(binaries.get(kind), f"manifest.binaries.{kind}")
        require(path.is_file() and not path.is_symlink(), f"{kind} binary is missing")
        require(sha256(path) == identity.get("sha256"), f"{kind} binary SHA mismatch")


def expect_reject(action: Any, label: str) -> None:
    try:
        action()
    except VerificationError:
        return
    raise AssertionError(f"{label}: verifier accepted tampered evidence")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="g07b-native-chain-validator-") as temporary:
        root = Path(temporary)
        payload = root / "payload.txt"
        payload.write_text("original\n", encoding="utf-8")
        manifest = {
            "artifacts": collect_artifact_index(root),
            "artifact_count": 1,
        }
        verify_artifact_index(root, manifest)
        payload.write_text("tampered\n", encoding="utf-8")
        expect_reject(lambda: verify_artifact_index(root, manifest), "artifact-index tamper")
        payload.write_text("original\n", encoding="utf-8")
        link = root / "payload-link"
        os.symlink(payload.name, link)
        expect_reject(lambda: collect_artifact_index(root), "artifact symlink")

        def digest(value: str) -> str:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()

        cuda_root = "/cuda"
        cuda_paths = sorted(
            {
                *REQUIRED_CUDA_TOOLKIT_FILES,
                "bin/crt/link.stub",
                "include/cuda.h",
                "nvvm/bin/cicc",
                "nvvm/libdevice/libdevice.10.bc",
            }
        )
        cuda_entries = [
            {
                "logical_path": path,
                "resolved_path": path,
                "sha256": digest(f"cuda:{path}"),
                "size_bytes": 1,
            }
            for path in cuda_paths
        ]
        nvcc_entry = next(row for row in cuda_entries if row["logical_path"] == "bin/nvcc")
        host_compiler = {
            "path": "/host/bin/c++",
            "sha256": digest("host:c++"),
            "size_bytes": 1,
        }
        host_cc1plus = {
            "path": "/host/bin/cc1plus",
            "sha256": digest("host:cc1plus"),
            "size_bytes": 1,
        }
        static_identity = {
            "backend": "cuda",
            "compiler_driver": "cuda_nvcc",
            "cuda_toolkit": {
                "canonical_root": cuda_root,
                "release_version": "12.4",
                "nvcc": {
                    "path": "/cuda/bin/nvcc",
                    "sha256": nvcc_entry["sha256"],
                    "size_bytes": 1,
                },
                "manifest": {"path": "toolchain/cuda.json", "sha256": digest("cuda"), "size_bytes": 1},
            },
            "host_toolchain": {
                "compiler": host_compiler,
                "compiler_version": "fixture cc 1.0",
                "target": "x86_64-ferrum-linux-gnu",
                "manifest": {"path": "toolchain/host.json", "sha256": digest("host"), "size_bytes": 1},
            },
            "archiver": {
                "path": "/usr/bin/ar",
                "sha256": digest("ar"),
                "size_bytes": 1,
            },
        }
        cuda_manifest = {
            "schema_version": 1,
            "canonical_root": cuda_root,
            "entries": cuda_entries,
        }
        host_files = [
            {
                "logical_path": "/host/bin/c++",
                "resolved_path": "/host/bin/c++",
                "sha256": host_compiler["sha256"],
                "size_bytes": 1,
            },
            {
                "logical_path": "/host/bin/cc1plus",
                "resolved_path": "/host/bin/cc1plus",
                "sha256": host_cc1plus["sha256"],
                "size_bytes": 1,
            },
            {
                "logical_path": "/host/include/stddef.h",
                "resolved_path": "/host/include/stddef.h",
                "sha256": digest("host:stddef"),
                "size_bytes": 1,
            },
        ]
        host_manifest = {
            "schema_version": 2,
            "compiler": host_compiler,
            "compiler_version": "fixture cc 1.0",
            "target": "x86_64-ferrum-linux-gnu",
            "executable_inputs": [host_compiler, host_cc1plus],
            "include_roots": ["/host/include"],
            "include_probe_sha256": digest("include-probe"),
            "driver_probe_sha256": digest("driver-probe"),
            "discovery_roots": ["/host/bin"],
            "files": host_files,
        }
        owners, allowed = verify_toolchain_manifests(
            static_identity,
            cuda_manifest,
            host_manifest,
            "selftest.toolchain",
        )

        source_unit = ("source", "kernels/a.cu", digest("a.cu"))
        source_header = ("source", "kernels/a.h", digest("a.h"))
        backend_header = (
            "backend_toolchain",
            "include/cuda.h",
            digest("cuda:include/cuda.h"),
        )
        host_header = (
            "host_toolchain",
            "/host/include/stddef.h",
            digest("host:stddef"),
        )
        identities = [source_unit, source_header, backend_header, host_header]
        command = {
            "object_file": "/new/objects/00000000-a.o",
            "depfile_producer_working_directory": "/src",
            "depfile_producer_object_file": "/old/objects/00000000-a.o",
            "observed_dependencies": [
                {"domain": domain, "path": path, "sha256": dependency_sha}
                for domain, path, dependency_sha in identities
            ],
        }
        depfile = root / "dependency.d"
        depfile.write_text(
            "/old/objects/00000000-a.o: /src/kernels/a.cu /src/kernels/a.h "
            "/cuda/include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        expected_source = {source_unit, source_header}
        verify_translation_unit_dependency_evidence(
            command,
            depfile,
            expected_source,
            owners,
            allowed,
            "selftest.command",
        )

        tampered_identity = copy.deepcopy(command)
        tampered_identity["observed_dependencies"][2]["sha256"] = digest("forged")
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                tampered_identity,
                depfile,
                expected_source,
                owners,
                allowed,
                "selftest.tampered_identity",
            ),
            "typed dependency SHA forgery",
        )
        unsorted = copy.deepcopy(command)
        unsorted["observed_dependencies"].reverse()
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                unsorted,
                depfile,
                expected_source,
                owners,
                allowed,
                "selftest.unsorted",
            ),
            "typed dependency order",
        )
        unknown_domain = copy.deepcopy(command)
        unknown_domain["observed_dependencies"][0]["domain"] = "rocm_guess"
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                unknown_domain,
                depfile,
                expected_source,
                owners,
                allowed,
                "selftest.unknown_domain",
            ),
            "unknown dependency domain",
        )
        relative_producer = copy.deepcopy(command)
        relative_producer["depfile_producer_working_directory"] = "src"
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                relative_producer,
                depfile,
                expected_source,
                owners,
                allowed,
                "selftest.relative_producer",
            ),
            "relative producer root",
        )
        depfile.write_text(
            "/wrong/objects/00000000-a.o: /src/kernels/a.cu /src/kernels/a.h "
            "/cuda/include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                command,
                depfile,
                expected_source,
                owners,
                allowed,
                "selftest.wrong_target",
            ),
            "same-basename wrong-directory target",
        )
        depfile.write_text(
            "/old/objects/00000000-a.o: /src/kernels/a.cu /src/kernels/a.h "
            "/cuda/include/cuda.h /host/include/stddef.h /tmp/generated.h\n",
            encoding="utf-8",
        )
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                command,
                depfile,
                expected_source,
                owners,
                allowed,
                "selftest.unmanifested_external",
            ),
            "unmanifested external dependency",
        )
        depfile.write_text(
            "/old/objects/00000000-a.o: /src/kernels/a.cu /src/kernels/a.h "
            "/cuda/include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                command,
                depfile,
                {source_unit},
                owners,
                allowed,
                "selftest.wrong_tu_closure",
            ),
            "wrong translation-unit closure",
        )

        wrong_driver = copy.deepcopy(static_identity)
        wrong_driver["compiler_driver"] = "hip_clang"
        expect_reject(
            lambda: verify_toolchain_manifests(
                wrong_driver,
                cuda_manifest,
                host_manifest,
                "selftest.wrong_driver",
            ),
            "backend/compiler-driver rewrite",
        )
        colliding_host = copy.deepcopy(host_manifest)
        colliding_host["include_roots"].append("/cuda/include")
        colliding_host["files"].append(
            {
                "logical_path": "/cuda/include/cuda.h",
                "resolved_path": "/cuda/include/cuda.h",
                "sha256": digest("host-owned-cuda-header"),
                "size_bytes": 1,
            }
        )
        colliding_host["files"].sort(key=lambda row: row["logical_path"])
        expect_reject(
            lambda: verify_toolchain_manifests(
                static_identity,
                cuda_manifest,
                colliding_host,
                "selftest.cross_domain_alias",
            ),
            "CUDA/host ownership collision",
        )
    print("FERRUM RUNTIME VNEXT G07B NATIVE CHAIN VALIDATOR SELFTEST PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--artifact-root", type=Path)
    result.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        try:
            self_test()
        except (AssertionError, OSError, VerificationError) as error:
            print(
                f"FERRUM RUNTIME VNEXT G07B NATIVE CHAIN VALIDATOR SELFTEST REJECT: {error}",
                file=sys.stderr,
            )
            return 1
        return 0
    if args.artifact_root is None:
        print("--artifact-root is required unless --self-test is used", file=sys.stderr)
        return 2
    root = args.artifact_root.expanduser().resolve()
    try:
        verify_manifest(root, args.source_root.expanduser().resolve())
    except (OSError, subprocess.SubprocessError, VerificationError, ValueError) as error:
        print(
            f"FERRUM RUNTIME VNEXT G07B NATIVE CHAIN KEEP REJECTED: {root}: {error}",
            file=sys.stderr,
        )
        return 1
    print(f"FERRUM RUNTIME VNEXT G07B NATIVE CHAIN KEEP VERIFIED: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
