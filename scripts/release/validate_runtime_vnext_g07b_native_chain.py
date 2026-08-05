#!/usr/bin/env python3
"""Independently verify a copied G07B native-operator chain artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


SCHEMA_VERSION = 3
RECEIPT_SCHEMA = "ferrum.bounded-command-receipt.v1"
SOURCE_BUILD_RECEIPT_SCHEMA_VERSION = 7
SOURCE_OBJECT_BUILD_CONTRACT_VERSION = 7
MAX_DEPFILE_BYTES = 16 * 1024 * 1024
MAX_DEPFILE_DEPENDENCIES = 250_000
MAX_DEPFILE_WORD_BYTES = 16 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
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
ARTIFACT_FEATURES = [
    "cuda",
    "vllm-marlin",
    "vllm-moe-marlin",
    "vllm-paged-attn-v2",
    "native-op-artifact",
]
EXPECTED_BUILD_UNITS = {
    "marlin",
    "vllm_marlin",
    "vllm_moe_marlin",
    "vllm_paged_attn",
}
CUDA_BUILD_SUMMARY_RECEIPT_SCHEMA_VERSION = 1
EXPECTED_STEPS = {
    "builder-build",
    "assemble-artifact-set",
    "artifact-example-build",
    "artifact-catalog-export",
    *(f"materialize-{name}" for name in PACKAGES),
    *(f"source-build-{name}" for name in PACKAGES),
    *(f"package-{name}" for name in PACKAGES),
}
DOES_NOT_PROVE = {
    "canonical G07B PASS",
    "G07 aggregate PASS",
    "model correctness",
    "model performance",
    "release readiness",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CUDA_INPUTS_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


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


def read_bounded_regular_file(path: Path, max_bytes: int, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VerificationError(f"cannot open {label} {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        require(
            stat.S_ISREG(metadata.st_mode) and metadata.st_size <= max_bytes,
            f"{label} is not a regular file or exceeds {max_bytes} bytes: {path}",
        )
        chunks: list[bytes] = []
        total = 0
        while total <= max_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        require(total <= max_bytes, f"{label} grew beyond {max_bytes} bytes: {path}")
        return b"".join(chunks)
    except OSError as error:
        raise VerificationError(f"cannot read {label} {path}: {error}") from error
    finally:
        os.close(descriptor)


def read_json(path: Path, label: str) -> Any:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    try:
        raw = read_bounded_regular_file(path, MAX_JSON_BYTES, label)
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {label} {path}: {error}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rust_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def normalize_absolute_depfile_path(value: str, label: str) -> str:
    require(
        bool(value)
        and value.startswith("/")
        and "\\" not in value
        and not any(character in value for character in ("\0", "\n", "\r")),
        f"{label} must be an absolute POSIX path",
    )
    components: list[str] = []
    for component in value.split("/"):
        if component in {"", "."}:
            continue
        if component == "..":
            require(bool(components), f"{label} escapes the filesystem root")
            components.pop()
        else:
            components.append(component)
    return "/" + "/".join(components)


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
        if part == "..":
            require(bool(parts), f"{label} escapes its working directory: {value!r}")
            parts.pop()
        else:
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


def canonical_tool_file_identity(value: Any, label: str) -> dict[str, Any]:
    row = verify_tool_file_identity(value, label)
    return {
        "path": row["path"],
        "sha256": row["sha256"],
        "size_bytes": row["size_bytes"],
    }


def canonical_evidence_file(value: Any, label: str) -> dict[str, Any]:
    row = require_dict(value, label)
    require(set(row) == {"path", "sha256", "size_bytes"}, f"{label} shape mismatch")
    path = require_relative_posix_path(row.get("path"), f"{label}.path")
    digest = require_sha(row.get("sha256"), f"{label}.sha256")
    require(
        isinstance(row.get("size_bytes"), int) and row["size_bytes"] > 0,
        f"{label}.size_bytes must be positive",
    )
    return {"path": path, "sha256": digest, "size_bytes": row["size_bytes"]}


def canonical_static_toolchain_identity(value: Any, label: str) -> dict[str, Any]:
    static = require_dict(value, label)
    require(
        set(static)
        == {
            "backend",
            "compiler_driver",
            "cuda_toolkit",
            "host_toolchain",
            "archiver",
        },
        f"{label} shape mismatch",
    )
    require(
        static.get("backend") == "cuda"
        and static.get("compiler_driver") == "cuda_nvcc",
        f"{label} must use the explicit CUDA nvcc compiler driver",
    )

    cuda = require_dict(static.get("cuda_toolkit"), f"{label}.cuda_toolkit")
    require(
        set(cuda)
        == {
            "canonical_root",
            "invocation_root",
            "release_version",
            "nvcc",
            "manifest",
        },
        f"{label}.cuda_toolkit shape mismatch",
    )
    canonical_root = require_absolute_posix_path(
        cuda.get("canonical_root"), f"{label}.cuda_toolkit.canonical_root"
    )
    invocation_root = require_absolute_posix_path(
        cuda.get("invocation_root"), f"{label}.cuda_toolkit.invocation_root"
    )
    require(
        isinstance(cuda.get("release_version"), str)
        and re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", cuda["release_version"]) is not None,
        f"{label}.cuda_toolkit.release_version is invalid",
    )

    host = require_dict(static.get("host_toolchain"), f"{label}.host_toolchain")
    require(
        set(host) == {"compiler", "compiler_version", "target", "manifest"},
        f"{label}.host_toolchain shape mismatch",
    )
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
    return {
        "backend": "cuda",
        "compiler_driver": "cuda_nvcc",
        "cuda_toolkit": {
            "canonical_root": canonical_root,
            "invocation_root": invocation_root,
            "release_version": cuda["release_version"],
            "nvcc": canonical_tool_file_identity(
                cuda.get("nvcc"), f"{label}.cuda_toolkit.nvcc"
            ),
            "manifest": canonical_evidence_file(
                cuda.get("manifest"), f"{label}.cuda_toolkit.manifest"
            ),
        },
        "host_toolchain": {
            "compiler": canonical_tool_file_identity(
                host.get("compiler"), f"{label}.host_toolchain.compiler"
            ),
            "compiler_version": host["compiler_version"],
            "target": host["target"],
            "manifest": canonical_evidence_file(
                host.get("manifest"), f"{label}.host_toolchain.manifest"
            ),
        },
        "archiver": canonical_tool_file_identity(
            static.get("archiver"), f"{label}.archiver"
        ),
    }


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
    static = canonical_static_toolchain_identity(
        static_identity, f"{label}.static_identity"
    )
    cuda = static["cuda_toolkit"]
    cuda_root = require_absolute_posix_path(
        cuda.get("canonical_root"), f"{label}.cuda_toolkit.canonical_root"
    )
    invocation_root = require_absolute_posix_path(
        cuda.get("invocation_root"), f"{label}.cuda_toolkit.invocation_root"
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
        for root in (cuda_root, invocation_root):
            for relative in (logical, resolved):
                absolute = posix_join(root, relative)
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

    host = static["host_toolchain"]
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
        identity = ("host_toolchain", resolved, digest)
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


def canonical_source_file_lock(value: Any, label: str) -> dict[str, str]:
    row = require_dict(value, label)
    require(set(row) == {"path", "sha256"}, f"{label} shape mismatch")
    return {
        "path": require_relative_posix_path(row.get("path"), f"{label}.path"),
        "sha256": require_sha(row.get("sha256"), f"{label}.sha256"),
    }


def canonical_nvcc_policy(value: Any, label: str) -> dict[str, Any]:
    row = require_dict(value, label)
    fields = (
        "cpp_standard",
        "optimization",
        "use_fast_math",
        "relaxed_constexpr",
        "extended_lambda",
        "host_position_independent_code",
        "host_default_visibility",
    )
    require(set(row) == set(fields), f"{label} shape mismatch")
    require(
        row.get("cpp_standard") == "cpp17" and row.get("optimization") == "o3",
        f"{label} compiler language/optimization policy is unsupported",
    )
    require(
        all(isinstance(row.get(field), bool) for field in fields[2:]),
        f"{label} boolean policy fields are invalid",
    )
    return {field: row[field] for field in fields}


def expected_nvcc_policy_flags(policy: dict[str, Any]) -> list[str]:
    result = ["-std=c++17", "-O3"]
    if policy["use_fast_math"]:
        result.append("--use_fast_math")
    if policy["relaxed_constexpr"]:
        result.append("--expt-relaxed-constexpr")
    if policy["extended_lambda"]:
        result.append("--expt-extended-lambda")
    if policy["host_position_independent_code"]:
        result.extend(["-Xcompiler", "-fPIC"])
    if policy["host_default_visibility"]:
        result.extend(["-Xcompiler", "-fvisibility=default"])
    return result


def expected_architecture_argument(plan: dict[str, Any], compute_capability: str, label: str) -> str:
    architecture = plan.get("architecture")
    require(
        architecture in {"device_compute_capability", "compute80_ptx"},
        f"{label}.architecture is unsupported",
    )
    return (
        f"-arch={compute_capability}"
        if architecture == "device_compute_capability"
        else "-arch=compute_80"
    )


def expected_effective_environment(
    static_identity: dict[str, Any], label: str
) -> dict[str, str]:
    tool_paths = (
        static_identity["cuda_toolkit"]["nvcc"]["path"],
        static_identity["host_toolchain"]["compiler"]["path"],
        static_identity["archiver"]["path"],
    )
    parents = {
        require_absolute_posix_path(path, f"{label}.tool_path")
        for path in tool_paths
    }
    path_entries = {
        PurePosixPath(path).parent.as_posix() for path in parents
    } | {"/bin", "/usr/bin"}
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": ":".join(sorted(path_entries)),
        "SOURCE_DATE_EPOCH": "0",
        "TMPDIR": "/tmp",
        "TZ": "UTC",
        "ZERO_AR_DATE": "1",
    }


def safe_component(value: str) -> str:
    return "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in value
    )


def source_object_file_name(index: int, translation_unit: dict[str, str]) -> str:
    stem = PurePosixPath(translation_unit["path"]).stem or "translation_unit"
    return (
        f"{index:08}_{safe_component(stem)}_"
        f"{translation_unit['sha256'][:8]}.o"
    )


def expected_source_command_argv(
    *,
    static_identity: dict[str, Any],
    translation_unit: dict[str, str],
    object_file: str,
    architecture_argument: str,
    compiler_depfile_output: str,
    include_dirs: list[Any],
    defines: list[Any],
    nvcc_policy: dict[str, Any],
    nvcc_threads: int,
) -> list[str]:
    return [
        static_identity["cuda_toolkit"]["nvcc"]["path"],
        "-c",
        translation_unit["path"],
        "-o",
        object_file,
        architecture_argument,
        "-ccbin",
        static_identity["host_toolchain"]["compiler"]["path"],
        "-MMD",
        "-MF",
        compiler_depfile_output,
        "-MT",
        object_file,
        *(f"-I{value}" for value in include_dirs),
        *(f"-D{value}" for value in defines),
        *expected_nvcc_policy_flags(nvcc_policy),
        "--threads",
        str(nvcc_threads),
    ]


def verify_exact_list(value: Any, expected: list[Any], label: str) -> None:
    require(isinstance(value, list) and value == expected, f"{label} differs from its typed contract")


def verify_source_build_input_identity(
    receipt: dict[str, Any],
    plan: dict[str, Any],
    static_identity: dict[str, Any],
    label: str,
) -> tuple[dict[str, str], dict[str, Any], str]:
    source_package = require_dict(plan.get("source_package"), f"{label}.source_package")
    require(
        set(source_package) == {"kind", "revision", "sha256"},
        f"{label}.source_package shape mismatch",
    )
    source_package_sha256 = require_sha(
        source_package.get("sha256"), f"{label}.source_package.sha256"
    )
    compute_capability = receipt.get("compute_capability")
    require(
        isinstance(compute_capability, str)
        and re.fullmatch(r"sm_[0-9]+", compute_capability) is not None,
        f"{label}.compute_capability is invalid",
    )
    architecture_argument = expected_architecture_argument(
        plan, compute_capability, label
    )
    require(
        receipt.get("architecture_argument") == architecture_argument,
        f"{label}.architecture_argument differs from the locked plan",
    )
    effective_environment = expected_effective_environment(
        static_identity, f"{label}.effective_environment"
    )
    require(
        receipt.get("effective_environment") == effective_environment,
        f"{label}.effective_environment differs from the typed tool paths",
    )
    plan_sha256 = require_sha(receipt.get("plan_sha256"), f"{label}.plan_sha256")
    inputs = {
        "plan_sha256": plan_sha256,
        "source_package_sha256": source_package_sha256,
        "builder_contract_version": SOURCE_OBJECT_BUILD_CONTRACT_VERSION,
        "architecture_argument": architecture_argument,
        "effective_environment": effective_environment,
        "toolchain": static_identity,
    }
    require(
        receipt.get("inputs_sha256") == rust_json_sha256(inputs),
        f"{label}.inputs_sha256 does not match independently rebuilt inputs",
    )
    return effective_environment, canonical_nvcc_policy(
        plan.get("nvcc_policy"), f"{label}.nvcc_policy"
    ), architecture_argument


def expected_source_object_cache_key(
    *,
    operator: str,
    translation_unit: dict[str, str],
    closure: dict[str, Any],
    plan: dict[str, Any],
    nvcc_policy: dict[str, Any],
    architecture_argument: str,
    effective_environment: dict[str, str],
    static_identity: dict[str, Any],
) -> str:
    headers = [
        canonical_source_file_lock(
            row, f"{operator}.{translation_unit['path']}.closure_header"
        )
        for row in require_list(closure.get("headers"), "dependency closure headers")
    ]
    identity = {
        "schema_version": SOURCE_OBJECT_BUILD_CONTRACT_VERSION,
        "operator": operator,
        "translation_unit": canonical_source_file_lock(
            translation_unit, f"{operator}.translation_unit"
        ),
        "dependency_closure_sha256": require_sha(
            closure.get("closure_sha256"), "dependency closure SHA256"
        ),
        "headers": headers,
        "include_dirs": require_list(plan.get("include_dirs"), "source plan include_dirs"),
        "defines": require_list(plan.get("defines"), "source plan defines"),
        "nvcc_policy": nvcc_policy,
        "architecture_argument": architecture_argument,
        "builder_contract_version": SOURCE_OBJECT_BUILD_CONTRACT_VERSION,
        "effective_environment": effective_environment,
        "toolchain": static_identity,
    }
    return rust_json_sha256(identity)


def verify_source_object_cache_key(
    value: Any,
    *,
    label: str,
    operator: str,
    translation_unit: dict[str, str],
    closure: dict[str, Any],
    plan: dict[str, Any],
    nvcc_policy: dict[str, Any],
    architecture_argument: str,
    effective_environment: dict[str, str],
    static_identity: dict[str, Any],
) -> str:
    expected = expected_source_object_cache_key(
        operator=operator,
        translation_unit=translation_unit,
        closure=closure,
        plan=plan,
        nvcc_policy=nvcc_policy,
        architecture_argument=architecture_argument,
        effective_environment=effective_environment,
        static_identity=static_identity,
    )
    require(value == expected, f"{label} differs from independently rebuilt inputs")
    return expected


def parse_make_words(value: str, label: str, max_words: int) -> list[str]:
    words: list[str] = []
    word: list[str] = []
    word_bytes = 0
    escaped = False
    for character in value:
        if escaped:
            word.append(character)
            word_bytes += len(character.encode("utf-8"))
            require(
                word_bytes <= MAX_DEPFILE_WORD_BYTES,
                f"{label} word exceeds {MAX_DEPFILE_WORD_BYTES} bytes",
            )
            escaped = False
        elif character == "\\":
            escaped = True
        elif character.isspace():
            if word:
                require(len(words) < max_words, f"{label} exceeds its word-count limit")
                words.append("".join(word))
                word = []
                word_bytes = 0
        else:
            word.append(character)
            word_bytes += len(character.encode("utf-8"))
            require(
                word_bytes <= MAX_DEPFILE_WORD_BYTES,
                f"{label} word exceeds {MAX_DEPFILE_WORD_BYTES} bytes",
            )
    require(not escaped, f"{label} ends with an incomplete escape")
    if word:
        require(len(words) < max_words, f"{label} exceeds its word-count limit")
        words.append("".join(word))
    return words


def parse_make_depfile(raw: str, label: str) -> tuple[str, list[str]]:
    require(
        len(raw.encode("utf-8")) <= MAX_DEPFILE_BYTES
        and bool(raw.strip())
        and "\0" not in raw,
        f"{label} is too large, empty, or contains NUL",
    )
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
    targets = parse_make_words(normalized[:delimiter], label, 1)
    dependencies = parse_make_words(
        normalized[delimiter + 1 :], label, MAX_DEPFILE_DEPENDENCIES
    )
    require(
        len(targets) == 1 and bool(dependencies),
        f"{label} target/dependency count or word size is invalid",
    )
    return targets[0], dependencies


def escape_make_word(value: str, label: str) -> str:
    require(
        bool(value)
        and len(value.encode("utf-8")) <= MAX_DEPFILE_WORD_BYTES
        and not any(character in value for character in ("\0", "\n", "\r")),
        f"{label} must be a bounded non-empty single-line make word",
    )
    return "".join(
        f"\\{character}" if character in {"\\", " ", "\t", ":", "#", "$"} else character
        for character in value
    )


def serialize_portable_depfile(target: str, dependencies: list[str], label: str) -> str:
    require_absolute_posix_path(target, f"{label}.target")
    require(
        0 < len(dependencies) <= MAX_DEPFILE_DEPENDENCIES,
        f"{label} dependency count is invalid",
    )
    words = [escape_make_word(target, f"{label}.target")]
    for index, dependency in enumerate(dependencies):
        if dependency.startswith("/"):
            require_absolute_posix_path(dependency, f"{label}.dependencies[{index}]")
        else:
            require_relative_posix_path(dependency, f"{label}.dependencies[{index}]")
        words.append(escape_make_word(dependency, f"{label}.dependencies[{index}]"))
    raw = f"{words[0]}: {' '.join(words[1:])}\n"
    require(
        len(raw.encode("utf-8")) <= MAX_DEPFILE_BYTES,
        f"{label} exceeds {MAX_DEPFILE_BYTES} bytes",
    )
    parsed_target, parsed_dependencies = parse_make_depfile(raw, label)
    require(
        parsed_target == target and parsed_dependencies == dependencies,
        f"{label} serialization did not round-trip",
    )
    return raw


def verify_translation_unit_dependency_evidence(
    command: dict[str, Any],
    compiler_depfile_path: Path,
    depfile_path: Path,
    expected_source: set[DependencyIdentity],
    toolchain_owners: dict[str, DependencyIdentity],
    allowed_toolchain: set[DependencyIdentity],
    label: str,
    *,
    expected_compiler_sha256: str | None = None,
    expected_portable_sha256: str | None = None,
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

    compiler_bytes = read_bounded_regular_file(
        compiler_depfile_path, MAX_DEPFILE_BYTES, f"{label}.compiler_depfile"
    )
    portable_bytes = read_bounded_regular_file(
        depfile_path, MAX_DEPFILE_BYTES, f"{label}.depfile"
    )
    if expected_compiler_sha256 is not None:
        require_sha(expected_compiler_sha256, f"{label}.compiler_depfile_sha256")
        require(
            hashlib.sha256(compiler_bytes).hexdigest() == expected_compiler_sha256,
            f"{label} compiler depfile SHA mismatch",
        )
    if expected_portable_sha256 is not None:
        require_sha(expected_portable_sha256, f"{label}.depfile_sha256")
        require(
            hashlib.sha256(portable_bytes).hexdigest() == expected_portable_sha256,
            f"{label} portable depfile SHA mismatch",
        )
    try:
        compiler_raw = compiler_bytes.decode("utf-8")
        portable_raw = portable_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"cannot read {label} depfile evidence: {error}") from error
    compiler_target, compiler_dependencies = parse_make_depfile(
        compiler_raw, f"{label}.compiler_depfile"
    )
    portable_target, portable_dependencies = parse_make_depfile(
        portable_raw, f"{label}.depfile"
    )
    require(
        compiler_target == producer_object and portable_target == producer_object,
        f"{label} compiler/portable depfile target differs from its producer object",
    )

    raw_bindings = require_list(command.get("depfile_bindings"), f"{label}.depfile_bindings")
    require(
        0 < len(raw_bindings) <= MAX_DEPFILE_DEPENDENCIES,
        f"{label}.depfile_bindings count is invalid",
    )
    working = PurePosixPath(working_directory)
    bindings: list[tuple[str, str, DependencyIdentity]] = []
    bound_dependencies: set[DependencyIdentity] = set()
    for index, raw_binding in enumerate(raw_bindings):
        binding = require_dict(raw_binding, f"{label}.depfile_bindings[{index}]")
        require(
            set(binding) == {"producer_path", "portable_path", "dependency"},
            f"{label}.depfile_bindings[{index}] shape mismatch",
        )
        producer_path = binding.get("producer_path")
        portable_path = binding.get("portable_path")
        require(
            isinstance(producer_path, str)
            and bool(producer_path)
            and len(producer_path.encode("utf-8")) <= MAX_DEPFILE_WORD_BYTES
            and not any(character in producer_path for character in ("\0", "\n", "\r")),
            f"{label}.depfile_bindings[{index}].producer_path is invalid",
        )
        require(
            isinstance(portable_path, str)
            and bool(portable_path)
            and len(portable_path.encode("utf-8")) <= MAX_DEPFILE_WORD_BYTES,
            f"{label}.depfile_bindings[{index}].portable_path is invalid",
        )
        dependency = verify_dependency_rows(
            [binding.get("dependency")],
            f"{label}.depfile_bindings[{index}].dependency",
        )[0]
        require(
            dependency not in bound_dependencies,
            f"{label}.depfile_bindings duplicate a typed dependency",
        )
        bound_dependencies.add(dependency)
        producer = PurePosixPath(producer_path)
        if dependency[0] == "source":
            require(
                portable_path == dependency[1],
                f"{label}.depfile_bindings[{index}] source portable path mismatch",
            )
            if producer.is_absolute():
                try:
                    producer_relative = producer.relative_to(working).as_posix()
                except ValueError as error:
                    raise VerificationError(
                        f"{label}.depfile_bindings[{index}] source producer path escapes its working directory"
                    ) from error
            else:
                producer_relative = producer_path
            require(
                normalize_depfile_relative_path(
                    producer_relative,
                    f"{label}.depfile_bindings[{index}].producer_path",
                )
                == dependency[1],
                f"{label}.depfile_bindings[{index}] source producer path mismatch",
            )
        else:
            normalized_producer = normalize_absolute_depfile_path(
                producer_path,
                f"{label}.depfile_bindings[{index}].producer_path",
            )
            require(
                producer.is_absolute()
                and require_absolute_posix_path(
                    portable_path,
                    f"{label}.depfile_bindings[{index}].portable_path",
                )
                and toolchain_owners.get(normalized_producer) == dependency
                and toolchain_owners.get(portable_path) == dependency,
                f"{label}.depfile_bindings[{index}] toolchain identity is unmanifested",
            )
        bindings.append((producer_path, portable_path, dependency))

    require(
        compiler_dependencies == [binding[0] for binding in bindings],
        f"{label} compiler depfile differs from its ordered typed bindings",
    )
    expected_portable_dependencies = [
        portable_path
        for _, portable_path, _ in sorted(
            bindings, key=lambda binding: dependency_sort_key(binding[2])
        )
    ]
    require(
        portable_dependencies == expected_portable_dependencies
        and portable_raw
        == serialize_portable_depfile(
            producer_object, expected_portable_dependencies, f"{label}.canonical_depfile"
        ),
        f"{label} portable depfile differs from its canonical typed bindings",
    )
    require(
        sorted(bound_dependencies, key=dependency_sort_key) == observed,
        f"{label} depfile bindings differ from typed receipt evidence",
    )
    for _, portable_path, dependency in bindings:
        if dependency[0] != "source":
            identity = toolchain_owners.get(portable_path)
            require(
                identity == dependency,
                f"{label} portable depfile contains an unmanifested external dependency: {portable_path}",
            )
    require(
        {row for row in bound_dependencies if row[0] == "source"} == expected_source,
        f"{label} depfile differs from the exact translation-unit closure",
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


def verify_lane_plan(lane_plan: dict[str, Any], source: dict[str, Any]) -> None:
    require(
        lane_plan.get("schema_version") == SCHEMA_VERSION,
        "lane plan schema mismatch",
    )
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
    verify_lane_plan(lane_plan, source)
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


def verify_step_plan(plan: dict[str, Any], step_id: str) -> tuple[list[str], int]:
    require(
        plan.get("schema_version") == SCHEMA_VERSION and plan.get("step_id") == step_id,
        f"{step_id} plan identity mismatch",
    )
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
    return command, deadline


def verify_step(root: Path, step_id: str) -> None:
    step_root = root / "steps" / step_id
    require(step_root.is_dir() and not step_root.is_symlink(), f"step is missing: {step_id}")
    plan = require_dict(read_json(step_root / "plan.json", f"{step_id} plan"), f"{step_id} plan")
    receipt = require_dict(
        read_json(step_root / "bounded.receipt.json", f"{step_id} receipt"),
        f"{step_id} receipt",
    )
    command, deadline = verify_step_plan(plan, step_id)
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
    native_source_root: Path,
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
        f"{name} source build is not a terminal schema-v7 PASS",
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
    static_identity = canonical_static_toolchain_identity(
        receipt["toolchain"].get("static_identity"),
        f"{name}.toolchain.static_identity",
    )
    cuda_toolkit = static_identity["cuda_toolkit"]
    host_toolchain = static_identity["host_toolchain"]
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
    effective_environment, nvcc_policy, architecture_argument = (
        verify_source_build_input_identity(receipt, plan, static_identity, name)
    )
    object_cache_root = require_absolute_posix_path(
        receipt.get("object_cache_root"), f"{name}.object_cache_root"
    )
    include_dirs = require_list(plan.get("include_dirs"), f"{name}.plan.include_dirs")
    require(
        all(
            isinstance(value, str)
            and require_relative_posix_path(
                value, f"{name}.plan.include_dirs[{index}]"
            )
            for index, value in enumerate(include_dirs)
        )
        and include_dirs == sorted(set(include_dirs)),
        f"{name}.plan.include_dirs must be sorted, unique relative paths",
    )
    defines = require_list(plan.get("defines"), f"{name}.plan.defines")
    require(
        all(
            isinstance(value, str)
            and bool(value)
            and not any(character.isspace() for character in value)
            for value in defines
        )
        and defines == sorted(set(defines)),
        f"{name}.plan.defines must be sorted, unique single arguments",
    )
    for collection in (translation_units, require_list(plan.get("headers"), f"{name}.plan.headers")):
        for row in collection:
            path = resolve_relative_file(
                native_source_root,
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
    expected_object_files: list[str] = []
    expected_output_root: PurePosixPath | None = None
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
        object_path_identity = PurePosixPath(object_file)
        output_root = object_path_identity.parent.parent
        if expected_output_root is None:
            expected_output_root = output_root
        require(
            output_root == expected_output_root
            and object_file
            == (
                output_root
                / "objects"
                / source_object_file_name(index, translation_unit)
            ).as_posix(),
            f"{name}.commands[{index}] object path differs from the locked translation unit",
        )
        expected_object_files.append(object_file)
        object_cache_key = verify_source_object_cache_key(
            command.get("object_cache_key"),
            label=f"{name}.commands[{index}].object_cache_key",
            operator=operator,
            translation_unit=translation_unit,
            closure=closure,
            plan=plan,
            nvcc_policy=nvcc_policy,
            architecture_argument=architecture_argument,
            effective_environment=effective_environment,
            static_identity=static_identity,
        )
        expected_cache_entry = (
            PurePosixPath(object_cache_root)
            / f"{operator}.object.{index:02}"
            / object_cache_key
        ).as_posix()
        require(
            command.get("object_cache_entry") == expected_cache_entry,
            f"{name}.commands[{index}] object cache entry differs from its content key",
        )
        stem = PurePosixPath(translation_unit["path"]).stem or "translation_unit"
        expected_depfile = f"depfiles/{index:08}-{stem}.d"
        expected_compiler_depfile = (
            f"depfiles/{index:08}-{stem}.compiler.raw.d"
        )
        require(
            command.get("compiler_depfile") == expected_compiler_depfile
            and command.get("depfile") == expected_depfile
            and command.get("stdout_log") == f"logs/{index:02}-{stem}.stdout.log"
            and command.get("stderr_log") == f"logs/{index:02}-{stem}.stderr.log",
            f"{name}.commands[{index}] output paths differ from the source-build contract",
        )
        expected_argv = expected_source_command_argv(
            static_identity=static_identity,
            translation_unit=translation_unit,
            object_file=object_file,
            architecture_argument=architecture_argument,
            compiler_depfile_output=(
                output_root / expected_compiler_depfile
            ).as_posix(),
            include_dirs=include_dirs,
            defines=defines,
            nvcc_policy=nvcc_policy,
            nvcc_threads=receipt["nvcc_threads"],
        )
        verify_exact_list(
            command.get("argv"),
            expected_argv,
            f"{name}.commands[{index}].argv",
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
        compiler_depfile = command.get("compiler_depfile")
        compiler_depfile_path = resolve_relative_file(
            build_root,
            compiler_depfile,
            f"{name}.commands[{index}].compiler_depfile",
        )
        require_sha(
            command.get("compiler_depfile_sha256"),
            f"{name}.commands[{index}].compiler_depfile_sha256",
        )
        depfile = command.get("depfile")
        depfile_path = resolve_relative_file(
            build_root, depfile, f"{name}.commands[{index}].depfile"
        )
        require_sha(command.get("depfile_sha256"), f"{name}.commands[{index}].depfile_sha256")
        verify_translation_unit_dependency_evidence(
            command,
            compiler_depfile_path,
            depfile_path,
            expected_source,
            toolchain_owners,
            allowed_toolchain,
            f"{name}.commands[{index}]",
            expected_compiler_sha256=command["compiler_depfile_sha256"],
            expected_portable_sha256=command["depfile_sha256"],
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
        and archive_command.get("compiler_depfile") is None
        and archive_command.get("compiler_depfile_sha256") is None
        and archive_command.get("depfile") is None
        and archive_command.get("depfile_sha256") is None
        and archive_command.get("depfile_producer_working_directory") is None
        and archive_command.get("depfile_producer_object_file") is None
        and archive_command.get("depfile_bindings") == []
        and archive_command.get("observed_dependencies") == []
        and archive_command.get("compiler_executed") is False
        and archive_command.get("return_code") == 0
        and isinstance(archive_command.get("elapsed_ms"), int)
        and archive_command["elapsed_ms"] >= 0,
        f"{name} archive command is not terminal",
    )
    require(expected_output_root is not None, f"{name} has no source-build output root")
    archive_file = require_relative_posix_path(
        receipt.get("archive_file"), f"{name}.archive_file"
    )
    require(archive_file == plan.get("archive_file"), f"{name} archive filename mismatch")
    require(
        PurePosixPath(archive_file).parent == PurePosixPath(".")
        and archive_file.startswith("lib")
        and archive_file.endswith(".a"),
        f"{name} archive filename is invalid",
    )
    expected_archive_argv = [
        static_identity["archiver"]["path"],
        "rcs",
        (expected_output_root / archive_file).as_posix(),
        *expected_object_files,
    ]
    verify_exact_list(
        archive_command.get("argv"), expected_archive_argv, f"{name}.commands.archive.argv"
    )
    require(
        archive_command.get("working_directory") == expected_working_directory
        and archive_command.get("stdout_log") == "logs/archive.stdout.log"
        and archive_command.get("stderr_log") == "logs/archive.stderr.log",
        f"{name} archive command differs from the typed archiver contract",
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

    manifest_by_operator = {
        manifest["operator"]: manifest for _, _, manifest in packages.values()
    }
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
        package_manifest = manifest_by_operator[operator]
        source_package = require_dict(
            package_manifest.get("source_package"),
            f"package manifest {operator}.source_package",
        )
        expected_inventory = {
            "schema_version": package_manifest.get("schema_version"),
            "operator": package_manifest.get("operator"),
            "operator_abi_version": package_manifest.get("operator_abi_version"),
            "ferrum_native_abi_version": package_manifest.get("ferrum_native_abi_version"),
            "backend": package_manifest.get("backend"),
            "linkage": package_manifest.get("linkage"),
            "g03_catalog_sha256": package_manifest.get("g03_catalog_sha256"),
            "abi_contract_sha256": package_manifest.get("abi_contract_sha256"),
            "descriptor_export": package_manifest.get("descriptor_export"),
            "operation_bindings": package_manifest.get("operation_bindings"),
            "exports": package_manifest.get("exports"),
            "source_package_sha256": source_package.get("sha256"),
            "inputs_sha256": package_manifest.get("inputs_sha256"),
            "binary_sha256": package_manifest.get("binary_sha256"),
        }
        require(
            row == expected_inventory,
            f"compiled inventory {operator} differs from the package manifest bound by the lock",
        )
    require(set(inventory_by_operator) == set(PACKAGES.values()), "compiled inventory coverage mismatch")


def verify_catalogs_and_packages(
    root: Path,
    source_root: Path,
    native_source_root: Path,
    manifest: dict[str, Any],
    source: dict[str, Any],
) -> None:
    catalog_input = root / "catalog-input/provider-catalog.json"
    artifact_provider = root / "artifact/provider-catalog.json"
    artifact_capability = root / "artifact/capability-catalog.json"
    artifact_inventory_path = root / "artifact/compiled-native-operators.json"
    for path in (
        catalog_input,
        artifact_provider,
        artifact_capability,
        artifact_inventory_path,
    ):
        require(path.is_file() and not path.is_symlink(), f"runtime export is missing: {path}")
    require(catalog_input.read_bytes() == artifact_provider.read_bytes(), "provider catalogs differ")
    provider_bytes = catalog_input.read_bytes()
    provider_sha = sha256(catalog_input)
    providers = verify_provider_catalog(read_json(catalog_input, "G03 provider catalog input"))
    artifact_inventory = require_list(
        read_json(artifact_inventory_path, "artifact compiled inventory"),
        "artifact compiled inventory",
    )
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
        source_receipt, source_plan = verify_source_build(
            root,
            source_root,
            native_source_root,
            name,
            operator,
            source,
        )
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
            "input_kind": "canonical-g03-live-catalog-gate",
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
    path = root / "artifact-build-summary.receipt.json"
    receipt = require_dict(
        read_json(path, "artifact build summary receipt"),
        "artifact build summary receipt",
    )
    require(
        receipt.get("schema_version") == CUDA_BUILD_SUMMARY_RECEIPT_SCHEMA_VERSION
        and receipt.get("artifact_type") == "ferrum_cuda_build_summary_receipt",
        "artifact build summary receipt identity mismatch",
    )
    summaries = require_list(receipt.get("rows"), "artifact build summary receipt.rows")
    for index, raw_row in enumerate(summaries):
        row = require_dict(raw_row, f"artifact build summary row {index}")
        require(
            set(row) == {"artifact", "status", "reason", "elapsed_ms", "inputs_hash"},
            f"artifact build summary row {index} shape mismatch",
        )
        require(
            all(isinstance(row.get(key), str) and row[key] for key in ("artifact", "status", "reason")),
            f"artifact build summary row {index} text fields are invalid",
        )
        require(
            isinstance(row.get("elapsed_ms"), int) and row["elapsed_ms"] >= 0,
            f"artifact build summary row {index} elapsed_ms is invalid",
        )
        require(
            isinstance(row.get("inputs_hash"), str)
            and CUDA_INPUTS_HASH_RE.fullmatch(row["inputs_hash"]) is not None,
            f"artifact build summary row {index} inputs_hash is invalid",
        )
    require(
        manifest.get("artifact_build_summaries") == summaries,
        "manifest build summaries differ from receipt",
    )
    receipt_summary = require_dict(
        manifest.get("artifact_build_summary_receipt"),
        "manifest.artifact_build_summary_receipt",
    )
    require(
        receipt_summary
        == {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "schema_version": CUDA_BUILD_SUMMARY_RECEIPT_SCHEMA_VERSION,
        },
        "manifest build summary receipt identity mismatch",
    )
    for unit in EXPECTED_BUILD_UNITS:
        rows = [row for row in summaries if row["artifact"] == unit]
        require(len(rows) == 1, f"artifact build summary count mismatch for {unit}")
        require(
            rows[0]["status"] == "artifact"
            and rows[0]["reason"] == "native-operator-artifact-set",
            f"native artifact {unit} fell back to source",
        )
    artifact_set_rows = [
        row for row in summaries if row["artifact"] == "native_operator_artifact_set"
    ]
    require(
        len(artifact_set_rows) == 1
        and artifact_set_rows[0]["status"] == "linked",
        "artifact build did not link exactly one native operator artifact set",
    )
    require(
        not any(row["status"] == "rejected" for row in summaries),
        "artifact build summary contains a rejected build decision",
    )
    output = (root / "steps/artifact-catalog-export/stdout.log").read_text(
        encoding="utf-8", errors="replace"
    )
    require(
        output.count("FERRUM RUNTIME VNEXT CUDA LIVE CATALOG READY:") == 1,
        "artifact-catalog-export did not emit exactly one runtime readiness line",
    )


def verify_dependency_bindings(
    root: Path, manifest: dict[str, Any], source: dict[str, Any]
) -> None:
    dependencies = require_dict(manifest.get("dependencies"), "manifest.dependencies")
    require(set(dependencies) == {"g03", "g07a"}, "chain dependency set mismatch")
    g03 = require_dict(dependencies["g03"], "manifest.dependencies.g03")
    g07a = require_dict(dependencies["g07a"], "manifest.dependencies.g07a")
    require(
        set(g03)
        == {
            "outer_manifest",
            "child_manifest",
            "source",
            "provider_catalog",
            "capability_catalog",
            "catalogs",
        }
        and set(g07a)
        == {
            "outer_manifest",
            "child_manifest",
            "source",
            "hardware_fingerprint",
            "scenario_targets",
            "semantic_plan_hash",
        },
        "chain dependency field set mismatch",
    )
    require(
        g03["source"] == g07a["source"] == source,
        "chain dependency source identity forked",
    )

    def verify_external_identity(value: Any, label: str) -> dict[str, Any]:
        identity = require_dict(value, label)
        raw_path = Path(identity["path"]) if isinstance(identity.get("path"), str) else Path()
        require(
            set(identity) == {"path", "sha256", "size_bytes"}
            and raw_path.is_absolute()
            and require_sha(identity["sha256"], f"{label}.sha256")
            and isinstance(identity["size_bytes"], int)
            and not isinstance(identity["size_bytes"], bool)
            and identity["size_bytes"] >= 0,
            f"{label} identity is invalid",
        )
        path = raw_path.resolve()
        require(
            path.is_file()
            and not path.is_symlink()
            and sha256(path) == identity["sha256"]
            and path.stat().st_size == identity["size_bytes"],
            f"{label} external artifact drifted",
        )
        return identity

    for dependency_name, dependency in (("g03", g03), ("g07a", g07a)):
        verify_external_identity(
            dependency["outer_manifest"],
            f"manifest.dependencies.{dependency_name}.outer_manifest",
        )
        verify_external_identity(
            dependency["child_manifest"],
            f"manifest.dependencies.{dependency_name}.child_manifest",
        )
    provider = verify_external_identity(
        g03["provider_catalog"], "manifest.dependencies.g03.provider_catalog"
    )
    verify_external_identity(
        g03["capability_catalog"], "manifest.dependencies.g03.capability_catalog"
    )
    catalog_input = root / "catalog-input/provider-catalog.json"
    require(
        provider["sha256"] == sha256(catalog_input)
        and provider["size_bytes"] == catalog_input.stat().st_size,
        "chain catalog input differs from its canonical G03 dependency identity",
    )
    catalogs = require_dict(g03["catalogs"], "manifest.dependencies.g03.catalogs")
    provider_summary = require_dict(
        catalogs.get("provider"), "manifest.dependencies.g03.catalogs.provider"
    )
    require(
        provider_summary.get("sha256") == provider["sha256"],
        "chain G03 catalog summary differs from the provider identity",
    )
    require(
        isinstance(g07a["hardware_fingerprint"], str)
        and SHA256_RE.fullmatch(g07a["hardware_fingerprint"]) is not None
        and isinstance(g07a["scenario_targets"], dict)
        and isinstance(g07a["semantic_plan_hash"], str)
        and SHA256_RE.fullmatch(g07a["semantic_plan_hash"]) is not None,
        "chain G07A dependency summary is invalid",
    )


def verify_manifest(root: Path, source_root: Path, native_source_root: Path) -> None:
    root = root.resolve()
    source_root = source_root.resolve()
    native_source_root = native_source_root.resolve()
    require(
        native_source_root.is_dir()
        and not native_source_root.is_relative_to(source_root),
        "native source root must be an external directory",
    )
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
            "source_build_units": list(PACKAGES),
            "artifact_features": ARTIFACT_FEATURES,
            "operators": sorted(PACKAGES.values()),
        },
        "chain scope mismatch",
    )
    require(
        manifest.get("native_source")
        == {
            "root": str(native_source_root),
            "external_to_repository": True,
        },
        "chain native source root binding mismatch",
    )
    verify_artifact_index(root, manifest)
    source = verify_source(root, source_root, manifest)
    verify_dependency_bindings(root, manifest, source)
    verify_hardware(root, manifest)
    verify_steps(root)
    verify_catalogs_and_packages(
        root,
        source_root,
        native_source_root,
        manifest,
        source,
    )
    verify_build_summaries(root, manifest)
    binaries = require_dict(manifest.get("binaries"), "manifest.binaries")
    require(set(binaries) == {"artifact"}, "manifest binary set mismatch")
    path = root / "binaries/artifact/runtime_vnext_cuda_catalog"
    identity = require_dict(binaries.get("artifact"), "manifest.binaries.artifact")
    require(path.is_file() and not path.is_symlink(), "artifact binary is missing")
    require(sha256(path) == identity.get("sha256"), "artifact binary SHA mismatch")


def expect_reject(action: Any, label: str) -> None:
    try:
        action()
    except VerificationError:
        return
    raise AssertionError(f"{label}: verifier accepted tampered evidence")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="g07b-native-chain-validator-") as temporary:
        root = Path(temporary)
        source = {
            "git_sha": "0" * 40,
            "git_tree_sha": "1" * 40,
            "dirty": False,
            "status_short": [],
        }
        dependency_root = root / "dependency-bindings"
        (dependency_root / "catalog-input").mkdir(parents=True)

        def write_dependency_file(name: str, content: str) -> Path:
            path = dependency_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return path.resolve()

        def external_identity(path: Path) -> dict[str, Any]:
            return {
                "path": str(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }

        g03_outer = write_dependency_file("external/g03-gate.json", "g03 outer\n")
        g03_child = write_dependency_file("external/g03-child.json", "g03 child\n")
        g07a_outer = write_dependency_file("external/g07a-gate.json", "g07a outer\n")
        g07a_child = write_dependency_file("external/g07a-child.json", "g07a child\n")
        provider = write_dependency_file("external/provider.json", "provider\n")
        capability = write_dependency_file("external/capability.json", "capability\n")
        catalog_input = write_dependency_file(
            "catalog-input/provider-catalog.json", "provider\n"
        )
        dependency_manifest = {
            "dependencies": {
                "g03": {
                    "outer_manifest": external_identity(g03_outer),
                    "child_manifest": external_identity(g03_child),
                    "source": source,
                    "provider_catalog": external_identity(provider),
                    "capability_catalog": external_identity(capability),
                    "catalogs": {
                        "provider": {"sha256": sha256(provider)},
                        "capability": {"sha256": sha256(capability)},
                    },
                },
                "g07a": {
                    "outer_manifest": external_identity(g07a_outer),
                    "child_manifest": external_identity(g07a_child),
                    "source": source,
                    "hardware_fingerprint": "2" * 64,
                    "scenario_targets": {"no_op": 30.0},
                    "semantic_plan_hash": "3" * 64,
                },
            }
        }
        verify_dependency_bindings(dependency_root, dependency_manifest, source)
        catalog_input.write_text("forked provider\n", encoding="utf-8")
        expect_reject(
            lambda: verify_dependency_bindings(
                dependency_root, dependency_manifest, source
            ),
            "canonical provider catalog fork",
        )
        catalog_input.write_text("provider\n", encoding="utf-8")
        forked_summary = copy.deepcopy(dependency_manifest)
        forked_summary["dependencies"]["g03"]["catalogs"]["provider"][
            "sha256"
        ] = "4" * 64
        expect_reject(
            lambda: verify_dependency_bindings(
                dependency_root, forked_summary, source
            ),
            "G03 provider summary fork",
        )
        g07a_child.write_text("tampered child\n", encoding="utf-8")
        expect_reject(
            lambda: verify_dependency_bindings(
                dependency_root, dependency_manifest, source
            ),
            "G07A child manifest tamper",
        )
        lane_plan = {
            "schema_version": SCHEMA_VERSION,
            "lane": "runtime-vnext-g07b-native-chain",
            "source": source,
            "expected_runtime_seconds": 1,
            "hard_deadline_seconds": 1,
            "hard_stop": "first failure",
            "correctness_gate": "catalog identity",
            "performance_command": "not applicable",
            "progress_signal": "bounded receipts",
        }
        verify_lane_plan(lane_plan, source)
        legacy_lane_plan = copy.deepcopy(lane_plan)
        legacy_lane_plan["schema_version"] = SCHEMA_VERSION - 1
        expect_reject(
            lambda: verify_lane_plan(legacy_lane_plan, source),
            "legacy lane plan schema",
        )
        step_plan = {
            "schema_version": SCHEMA_VERSION,
            "step_id": "builder-build",
            "command": ["cargo", "build"],
            "cwd": "/workspace/ferrum",
            "expected_duration_seconds": 1,
            "hard_deadline_seconds": 1,
            "progress_signal": "Cargo log growth",
        }
        verify_step_plan(step_plan, "builder-build")
        legacy_step_plan = copy.deepcopy(step_plan)
        legacy_step_plan["schema_version"] = SCHEMA_VERSION - 1
        expect_reject(
            lambda: verify_step_plan(legacy_step_plan, "builder-build"),
            "legacy step plan schema",
        )
        summary_root = root / "build-summary"
        summary_path = summary_root / "artifact-build-summary.receipt.json"
        summary_path.parent.mkdir(parents=True)
        summary_rows = [
            {
                "artifact": unit,
                "status": "artifact",
                "reason": "native-operator-artifact-set",
                "elapsed_ms": 0,
                "inputs_hash": f"sha256:{index:064x}",
            }
            for index, unit in enumerate(sorted(EXPECTED_BUILD_UNITS), start=1)
        ]
        summary_rows.append(
            {
                "artifact": "native_operator_artifact_set",
                "status": "linked",
                "reason": "manifest-v3-artifact-set-v5-validated",
                "elapsed_ms": 1,
                "inputs_hash": f"sha256:{9:064x}",
            }
        )
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": CUDA_BUILD_SUMMARY_RECEIPT_SCHEMA_VERSION,
                    "artifact_type": "ferrum_cuda_build_summary_receipt",
                    "rows": summary_rows,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for step in ("artifact-catalog-export",):
            step_root = summary_root / "steps" / step
            step_root.mkdir(parents=True)
            (step_root / "stdout.log").write_text(
                "FERRUM RUNTIME VNEXT CUDA LIVE CATALOG READY: fixture\n",
                encoding="utf-8",
            )
        summary_manifest = {
            "artifact_build_summaries": summary_rows,
            "artifact_build_summary_receipt": {
                "path": summary_path.relative_to(summary_root).as_posix(),
                "sha256": sha256(summary_path),
                "schema_version": CUDA_BUILD_SUMMARY_RECEIPT_SCHEMA_VERSION,
            },
        }
        verify_build_summaries(summary_root, summary_manifest)
        tampered_summary = copy.deepcopy(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )
        tampered_summary["rows"][0]["status"] = "built"
        summary_path.write_text(
            json.dumps(tampered_summary, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_reject(
            lambda: verify_build_summaries(summary_root, summary_manifest),
            "artifact build summary receipt tamper",
        )

        artifact_index_root = root / "artifact-index"
        artifact_index_root.mkdir()
        payload = artifact_index_root / "payload.txt"
        payload.write_text("original\n", encoding="utf-8")
        manifest = {
            "artifacts": collect_artifact_index(artifact_index_root),
            "artifact_count": 1,
        }
        verify_artifact_index(artifact_index_root, manifest)
        payload.write_text("tampered\n", encoding="utf-8")
        expect_reject(
            lambda: verify_artifact_index(artifact_index_root, manifest),
            "artifact-index tamper",
        )
        payload.write_text("original\n", encoding="utf-8")
        link = artifact_index_root / "payload-link"
        os.symlink(payload.name, link)
        expect_reject(
            lambda: collect_artifact_index(artifact_index_root),
            "artifact symlink",
        )

        def digest(value: str) -> str:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()

        require(
            normalize_depfile_relative_path(
                "kernels/vllm_marlin_moe/core/../vllm_torch_shim.h",
                "selftest.source_parent_segment",
            )
            == "kernels/vllm_marlin_moe/vllm_torch_shim.h",
            "source depfile parent segment was not normalized",
        )
        for escaped in ("../outside.h", "kernels/../../outside.h"):
            expect_reject(
                lambda escaped=escaped: normalize_depfile_relative_path(
                    escaped, "selftest.escaped_source"
                ),
                f"source depfile escape {escaped}",
            )

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
                "invocation_root": "/cuda-alias",
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
        require(
            owners["/cuda/include/cuda.h"]
            == owners["/cuda-alias/include/cuda.h"],
            "canonical and invocation CUDA roots do not share one typed owner",
        )
        canonical_static = canonical_static_toolchain_identity(
            static_identity, "selftest.cache_identity.static_identity"
        )
        fixture_plan = {
            "source_package": {
                "kind": "ferrum-git-tree",
                "revision": "7" * 40,
                "sha256": digest("source-package"),
            },
            "architecture": "compute80_ptx",
            "include_dirs": ["include"],
            "defines": ["FEATURE=1"],
            "nvcc_policy": {
                "cpp_standard": "cpp17",
                "optimization": "o3",
                "use_fast_math": True,
                "relaxed_constexpr": True,
                "extended_lambda": False,
                "host_position_independent_code": True,
                "host_default_visibility": False,
            },
        }
        fixture_environment = expected_effective_environment(
            canonical_static, "selftest.cache_identity.environment"
        )
        fixture_receipt = {
            "compute_capability": "sm_89",
            "architecture_argument": "-arch=compute_80",
            "effective_environment": fixture_environment,
            "plan_sha256": digest("plan"),
        }
        fixture_inputs = {
            "plan_sha256": fixture_receipt["plan_sha256"],
            "source_package_sha256": fixture_plan["source_package"]["sha256"],
            "builder_contract_version": SOURCE_OBJECT_BUILD_CONTRACT_VERSION,
            "architecture_argument": fixture_receipt["architecture_argument"],
            "effective_environment": fixture_environment,
            "toolchain": canonical_static,
        }
        fixture_receipt["inputs_sha256"] = rust_json_sha256(fixture_inputs)
        effective_environment, fixture_policy, fixture_architecture = (
            verify_source_build_input_identity(
                fixture_receipt,
                fixture_plan,
                canonical_static,
                "selftest.cache_identity",
            )
        )
        tampered_inputs = copy.deepcopy(fixture_receipt)
        tampered_inputs["inputs_sha256"] = digest("forged-inputs")
        expect_reject(
            lambda: verify_source_build_input_identity(
                tampered_inputs,
                fixture_plan,
                canonical_static,
                "selftest.tampered_inputs",
            ),
            "source-build inputs SHA forgery",
        )
        fixture_translation_unit = {
            "path": "kernels/a.cu",
            "sha256": digest("a.cu"),
        }
        fixture_closure = {
            "translation_unit": "kernels/a.cu",
            "headers": [{"path": "kernels/a.h", "sha256": digest("a.h")}],
            "closure_sha256": digest("closure"),
        }
        fixture_cache_key = expected_source_object_cache_key(
            operator="ferrum.cuda.fixture",
            translation_unit=fixture_translation_unit,
            closure=fixture_closure,
            plan=fixture_plan,
            nvcc_policy=fixture_policy,
            architecture_argument=fixture_architecture,
            effective_environment=effective_environment,
            static_identity=canonical_static,
        )
        verify_source_object_cache_key(
            fixture_cache_key,
            label="selftest.cache_identity.object_cache_key",
            operator="ferrum.cuda.fixture",
            translation_unit=fixture_translation_unit,
            closure=fixture_closure,
            plan=fixture_plan,
            nvcc_policy=fixture_policy,
            architecture_argument=fixture_architecture,
            effective_environment=effective_environment,
            static_identity=canonical_static,
        )
        expect_reject(
            lambda: verify_source_object_cache_key(
                digest("forged-object-key"),
                label="selftest.tampered_object_cache_key",
                operator="ferrum.cuda.fixture",
                translation_unit=fixture_translation_unit,
                closure=fixture_closure,
                plan=fixture_plan,
                nvcc_policy=fixture_policy,
                architecture_argument=fixture_architecture,
                effective_environment=effective_environment,
                static_identity=canonical_static,
            ),
            "object cache key forgery",
        )
        fixture_object = (
            PurePosixPath("/out/objects")
            / source_object_file_name(0, fixture_translation_unit)
        ).as_posix()
        fixture_argv = expected_source_command_argv(
            static_identity=canonical_static,
            translation_unit=fixture_translation_unit,
            object_file=fixture_object,
            architecture_argument=fixture_architecture,
            compiler_depfile_output="/out/depfiles/00000000-a.compiler.raw.d",
            include_dirs=fixture_plan["include_dirs"],
            defines=fixture_plan["defines"],
            nvcc_policy=fixture_policy,
            nvcc_threads=4,
        )
        verify_exact_list(fixture_argv, fixture_argv, "selftest.fixture_argv")
        for index, replacement, label in (
            (0, "/forged/nvcc", "nvcc argv"),
            (7, "/forged/c++", "host compiler argv"),
            (10, "/forged/dependency.d", "depfile argv"),
            (12, "/forged/object.o", "object target argv"),
            (-1, "8", "nvcc worker bound"),
        ):
            tampered_argv = fixture_argv.copy()
            tampered_argv[index] = replacement
            expect_reject(
                lambda tampered_argv=tampered_argv: verify_exact_list(
                    tampered_argv, fixture_argv, f"selftest.{label}"
                ),
                label,
            )
        reordered_flags = fixture_argv.copy()
        flag_index = reordered_flags.index("--use_fast_math")
        reordered_flags[flag_index], reordered_flags[flag_index + 1] = (
            reordered_flags[flag_index + 1],
            reordered_flags[flag_index],
        )
        expect_reject(
            lambda: verify_exact_list(
                reordered_flags, fixture_argv, "selftest.reordered_policy_flags"
            ),
            "policy flag order",
        )
        injected_flag = fixture_argv.copy()
        injected_flag.insert(-2, "--forged")
        expect_reject(
            lambda: verify_exact_list(
                injected_flag, fixture_argv, "selftest.injected_flag"
            ),
            "injected compiler flag",
        )
        fixture_archive_argv = [
            canonical_static["archiver"]["path"],
            "rcs",
            "/out/libfixture.a",
            fixture_object,
            "/out/objects/00000001-b.o",
        ]
        reordered_archive = fixture_archive_argv.copy()
        reordered_archive[-2:] = reversed(reordered_archive[-2:])
        expect_reject(
            lambda: verify_exact_list(
                reordered_archive,
                fixture_archive_argv,
                "selftest.reordered_archive_members",
            ),
            "archive member order",
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
            "depfile_bindings": [
                {
                    "producer_path": producer_path,
                    "portable_path": portable_path,
                    "dependency": {
                        "domain": identity[0],
                        "path": identity[1],
                        "sha256": identity[2],
                    },
                }
                for producer_path, portable_path, identity in (
                    ("/src/kernels/a.cu", "kernels/a.cu", source_unit),
                    ("/src/kernels/core/../a.h", "kernels/a.h", source_header),
                    (
                        "/cuda/bin/../include/cuda.h",
                        "/cuda/include/cuda.h",
                        backend_header,
                    ),
                    (
                        "/host/include/stddef.h",
                        "/host/include/stddef.h",
                        host_header,
                    ),
                )
            ],
            "observed_dependencies": [
                {"domain": domain, "path": path, "sha256": dependency_sha}
                for domain, path, dependency_sha in identities
            ],
        }
        compiler_depfile = root / "compiler-dependency.raw.d"
        depfile = root / "dependency.d"
        compiler_depfile.write_text(
            "/old/objects/00000000-a.o: /src/kernels/a.cu /src/kernels/core/../a.h "
            "/cuda/bin/../include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        depfile.write_text(
            "/old/objects/00000000-a.o: kernels/a.cu kernels/a.h "
            "/cuda/include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        expected_source = {source_unit, source_header}
        verify_translation_unit_dependency_evidence(
            command,
            compiler_depfile,
            depfile,
            expected_source,
            owners,
            allowed,
            "selftest.command",
        )
        forged_producer = copy.deepcopy(command)
        forged_producer["depfile_bindings"][2]["producer_path"] = "/forged/include/cuda.h"
        compiler_depfile.write_text(
            "/old/objects/00000000-a.o: /src/kernels/a.cu /src/kernels/core/../a.h "
            "/forged/include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                forged_producer,
                compiler_depfile,
                depfile,
                expected_source,
                owners,
                allowed,
                "selftest.forged_producer",
            ),
            "unmanifested raw producer path",
        )
        compiler_depfile.write_text(
            "/old/objects/00000000-a.o: /src/kernels/a.cu /src/kernels/core/../a.h "
            "/cuda/bin/../include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        depfile.write_text(
            "/old/objects/00000000-a.o:  kernels/a.cu kernels/a.h "
            "/cuda/include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                command,
                compiler_depfile,
                depfile,
                expected_source,
                owners,
                allowed,
                "selftest.noncanonical_portable",
            ),
            "noncanonical portable depfile bytes",
        )
        depfile.write_text(
            "/old/objects/00000000-a.o: kernels/a.cu kernels/a.h "
            "/cuda/include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        invocation_alias = copy.deepcopy(command)
        invocation_alias["depfile_bindings"][2]["producer_path"] = (
            "/cuda-alias/bin/../include/cuda.h"
        )
        invocation_alias["depfile_bindings"][2]["portable_path"] = (
            "/cuda-alias/include/cuda.h"
        )
        compiler_depfile.write_text(
            "/old/objects/00000000-a.o: /src/kernels/a.cu /src/kernels/core/../a.h "
            "/cuda-alias/bin/../include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        depfile.write_text(
            "/old/objects/00000000-a.o: kernels/a.cu kernels/a.h "
            "/cuda-alias/include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        verify_translation_unit_dependency_evidence(
            invocation_alias,
            compiler_depfile,
            depfile,
            expected_source,
            owners,
            allowed,
            "selftest.invocation_alias",
        )
        compiler_depfile.write_text(
            "/old/objects/00000000-a.o: /src/kernels/a.cu /src/kernels/core/../a.h "
            "/cuda/bin/../include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        depfile.write_text(
            "/old/objects/00000000-a.o: kernels/a.cu kernels/a.h "
            "/cuda/include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )

        tampered_identity = copy.deepcopy(command)
        tampered_identity["observed_dependencies"][2]["sha256"] = digest("forged")
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                tampered_identity,
                compiler_depfile,
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
                compiler_depfile,
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
                compiler_depfile,
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
                compiler_depfile,
                depfile,
                expected_source,
                owners,
                allowed,
                "selftest.relative_producer",
            ),
            "relative producer root",
        )
        depfile.write_text(
            "/wrong/objects/00000000-a.o: kernels/a.cu kernels/a.h "
            "/cuda/include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                command,
                compiler_depfile,
                depfile,
                expected_source,
                owners,
                allowed,
                "selftest.wrong_target",
            ),
            "same-basename wrong-directory target",
        )
        depfile.write_text(
            "/old/objects/00000000-a.o: kernels/a.cu kernels/a.h "
            "/cuda/include/cuda.h /host/include/stddef.h /tmp/generated.h\n",
            encoding="utf-8",
        )
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                command,
                compiler_depfile,
                depfile,
                expected_source,
                owners,
                allowed,
                "selftest.unmanifested_external",
            ),
            "unmanifested external dependency",
        )
        depfile.write_text(
            "/old/objects/00000000-a.o: kernels/a.cu kernels/a.h "
            "/cuda/include/cuda.h /host/include/stddef.h\n",
            encoding="utf-8",
        )
        expect_reject(
            lambda: verify_translation_unit_dependency_evidence(
                command,
                compiler_depfile,
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
        wrong_invocation_root = copy.deepcopy(static_identity)
        wrong_invocation_root["cuda_toolkit"]["invocation_root"] = "cuda-alias"
        expect_reject(
            lambda: verify_toolchain_manifests(
                wrong_invocation_root,
                cuda_manifest,
                host_manifest,
                "selftest.wrong_invocation_root",
            ),
            "relative CUDA invocation root",
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
    result.add_argument("--native-source-root", type=Path)
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
    if args.native_source_root is None:
        print("--native-source-root is required unless --self-test is used", file=sys.stderr)
        return 2
    root = args.artifact_root.expanduser().resolve()
    try:
        verify_manifest(
            root,
            args.source_root.expanduser().resolve(),
            args.native_source_root.expanduser().resolve(),
        )
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
