#!/usr/bin/env python3
"""Fail-closed public prerelease download gate for Ferrum v0.8.4.

The backend lanes intentionally exercise the documented first-run path from the
public GitHub prerelease asset through a fresh Hugging Face cache.  The
aggregate lane binds the two independently collected backend receipts to the
same immutable GitHub release/asset snapshot and to their staged SHA256 values.

This validator is fixed to v0.8.4.  It is deliberately standalone so the
frozen v0.8.0 release validators keep their historical contract.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterable


VERSION = "0.8.4"
TAG = "v0.8.4"
REPOSITORY = "sizzlecar/ferrum-infer-rs"
RELEASE_API_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{TAG}"
VALIDATOR_VERSION = "1.0.0"
SCHEMA_VERSION = 1
USER_AGENT = f"ferrum-v084-prerelease-download-gate/{VALIDATOR_VERSION}"
SUMMARY_PREFIX = f"ferrum-{VERSION}-prerelease-download"
GOAL_E2E_ARTIFACT_TYPE = "ferrum_v084_readme_e2e_summary"
GOAL_PRERELEASE_ARTIFACT_TYPE = "ferrum_v084_prerelease_manifest"
GOAL_EVIDENCE_KEYS = {
    "binary_version",
    "binary_help",
    "doctor",
    "download",
    "run",
    "serve",
    "models",
    "chat",
    "stream",
    "logs",
}
GOAL_LOG_SCAN_PATTERNS = [
    "panic",
    "oom",
    "cuda error",
    "metal error",
    "invalid utf-8",
    "<unk>",
    "[pad]",
    "control-token",
]
MAX_API_BYTES = 16 * 1024 * 1024
MAX_HTTP_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_TAR_MEMBERS = 64
MAX_TAR_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RC_TAG_RE = re.compile(r"^v0\.8\.4-rc\.[1-9][0-9]*$")
GOAL_SIDECAR_SUFFIXES = (
    ".sha256",
    ".binary.sha256",
    ".version.json",
    ".dependency.json",
    ".abi.json",
)
GOAL_ASSET_NAMES = {
    "cpu": "ferrum-linux-x86_64.tar.gz",
    "metal": "ferrum-macos-aarch64.tar.gz",
    "cuda": "ferrum-linux-x86_64-cuda-sm89.tar.gz",
}
GOAL_EXPECTED_ASSETS = {
    name
    for asset in GOAL_ASSET_NAMES.values()
    for name in (
        asset,
        *(asset + suffix for suffix in GOAL_SIDECAR_SUFFIXES),
        asset.removesuffix(".tar.gz") + ".dependencies.txt",
    )
}


@dataclass(frozen=True)
class BackendSpec:
    backend: str
    pass_label: str
    asset_name: str
    dependency_audit_name: str
    target_triple: str
    model_alias: str
    model_repositories: tuple[str, ...]
    required_model_files: tuple[tuple[str, tuple[str, ...]], ...]
    download_size_marker: str
    default_port: int

    @property
    def companion_names(self) -> tuple[str, ...]:
        return (
            f"{self.asset_name}.sha256",
            f"{self.asset_name}.binary.sha256",
            f"{self.asset_name}.version.json",
            f"{self.asset_name}.dependency.json",
            f"{self.asset_name}.abi.json",
            self.dependency_audit_name,
        )

    @property
    def required_release_asset_names(self) -> tuple[str, ...]:
        return (self.asset_name, *self.companion_names)


BACKEND_SPECS = {
    "metal": BackendSpec(
        backend="metal",
        pass_label="METAL",
        asset_name="ferrum-macos-aarch64.tar.gz",
        dependency_audit_name="ferrum-macos-aarch64.dependencies.txt",
        target_triple="aarch64-apple-darwin",
        model_alias="qwen3.5:4b-q4_k_m",
        model_repositories=("unsloth/Qwen3.5-4B-GGUF", "Qwen/Qwen3.5-4B"),
        required_model_files=(
            ("unsloth/Qwen3.5-4B-GGUF", ("Qwen3.5-4B-Q4_K_M.gguf",)),
            ("Qwen/Qwen3.5-4B", ("config.json", "tokenizer.json")),
        ),
        download_size_marker="2.55 GiB",
        default_port=18484,
    ),
    "cuda": BackendSpec(
        backend="cuda",
        pass_label="CUDA",
        asset_name="ferrum-linux-x86_64-cuda-sm89.tar.gz",
        dependency_audit_name="ferrum-linux-x86_64-cuda-sm89.dependencies.txt",
        target_triple="x86_64-unknown-linux-gnu",
        model_alias="qwen3.5:4b",
        model_repositories=("Qwen/Qwen3.5-4B",),
        required_model_files=(
            (
                "Qwen/Qwen3.5-4B",
                (
                    "config.json",
                    "tokenizer.json",
                    "model.safetensors.index.json",
                    "model.safetensors-00001-of-00002.safetensors",
                    "model.safetensors-00002-of-00002.safetensors",
                ),
            ),
        ),
        download_size_marker="8.7 GiB",
        default_port=28484,
    ),
}


def backend_download_asset_names(spec: BackendSpec) -> tuple[str, ...]:
    """Return the public assets whose HTTP receipts are owned by a backend lane.

    Metal owns the CPU bundle as well as its native bundle so the two real
    platform lanes together download the exact 21-asset release denominator.
    """

    names = list(spec.required_release_asset_names)
    if spec.backend == "metal":
        cpu_asset = GOAL_ASSET_NAMES["cpu"]
        names.extend(
            (
                cpu_asset,
                *(cpu_asset + suffix for suffix in GOAL_SIDECAR_SUFFIXES),
                cpu_asset.removesuffix(".tar.gz") + ".dependencies.txt",
            )
        )
    require(len(names) == len(set(names)), f"{spec.backend} download asset names collide")
    return tuple(names)


FORBIDDEN_TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("panic", re.compile(r"\bpanic(?:ked)?\b", re.IGNORECASE)),
    ("out-of-memory", re.compile(r"\b(?:out of memory|oom)\b", re.IGNORECASE)),
    (
        "cuda-error",
        re.compile(
            r"(?:CUDA[_ ]ERROR|CUBLAS_STATUS_[A-Z_]+|CUSPARSE_STATUS_[A-Z_]+|"
            r"NCCL\s+(?:WARN|ERROR)|illegal memory access)",
            re.IGNORECASE,
        ),
    ),
    (
        "metal-error",
        re.compile(
            r"(?:MTLCommandBufferError|command buffer execution failed|"
            r"Metal[^\n]{0,80}(?:error|fault)|GPU fault)",
            re.IGNORECASE,
        ),
    ),
    ("invalid-utf8-report", re.compile(r"invalid utf-?8", re.IGNORECASE)),
    ("unk-token", re.compile(r"<unk>", re.IGNORECASE)),
    ("pad-token", re.compile(r"\[PAD\]", re.IGNORECASE)),
    (
        "internal-control-token",
        re.compile(
            r"(?:<\|(?:im_start|im_end|endoftext|assistant|user|system)\|>|"
            r"</?think>|<\|channel\|>|<\|message\|>)",
            re.IGNORECASE,
        ),
    ),
)


class GateError(RuntimeError):
    """A fail-closed gate assertion failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def pretty_json_sha256(value: Any) -> str:
    return sha256_bytes(
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def strict_utf8(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise GateError(f"{label} is not strict UTF-8: {error}") from error


def reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_bytes(data: bytes, label: str) -> Any:
    text = strict_utf8(data, label)
    try:
        return json.loads(text, object_pairs_hook=reject_duplicate_json_pairs)
    except GateError:
        raise
    except json.JSONDecodeError as error:
        raise GateError(f"{label} is not valid JSON: {error}") from error


def read_json_file(path: Path, label: str) -> Any:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise GateError(f"cannot read {label} {path}: {error}") from error
    return parse_json_bytes(data, label)


def file_ref(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    display = path
    if relative_to is not None:
        try:
            display = path.resolve().relative_to(relative_to.resolve())
        except ValueError:
            display = path.resolve()
    return {
        "path": str(display),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def resolve_saved_ref(
    raw: Any,
    *,
    root: Path,
    label: str,
    require_nonempty: bool = False,
) -> Path:
    require(isinstance(raw, dict), f"{label} reference is not an object")
    require(set(raw) == {"path", "sha256", "size_bytes"}, f"{label} reference fields differ")
    text = raw.get("path")
    require(isinstance(text, str) and text, f"{label} reference path is missing")
    pure = PurePosixPath(text)
    require(
        not pure.is_absolute() and "\\" not in text and ".." not in pure.parts,
        f"{label} reference path escapes its artifact root",
    )
    root_resolved = root.resolve()
    candidate = root.joinpath(*pure.parts)
    require(not candidate.is_symlink(), f"{label} reference is a symlink")
    path = candidate.resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as error:
        raise GateError(f"{label} reference escapes its artifact root") from error
    require(path.is_file() and not path.is_symlink(), f"{label} reference is not a regular file")
    size = raw.get("size_bytes")
    digest = raw.get("sha256")
    require(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0,
        f"{label} reference size is invalid",
    )
    require(
        isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
        f"{label} reference SHA256 is invalid",
    )
    require(path.stat().st_size == size, f"{label} referenced size changed")
    require(sha256_file(path) == digest, f"{label} referenced SHA256 changed")
    if require_nonempty:
        require(size > 0, f"{label} evidence is empty")
    return path


def backend_total_deadline_seconds(args: argparse.Namespace, spec: BackendSpec) -> int:
    """Return the declared aggregate budget backed by every bounded child step."""

    seconds = (
        args.api_timeout_seconds
        + len(backend_download_asset_names(spec)) * args.asset_download_timeout_seconds
        + 4 * args.command_timeout_seconds
        + args.model_command_timeout_seconds
        + args.server_total_timeout_seconds
        + 600.0
    )
    return max(1, math.ceil(seconds))


def goal_model_identity_from_cache(
    cache: Any,
    spec: BackendSpec,
) -> dict[str, Any]:
    require(isinstance(cache, dict), "model cache receipt is missing")
    repositories = cache.get("repositories")
    require(isinstance(repositories, list) and repositories, "model cache repositories are missing")
    primary = next(
        (
            row
            for row in repositories
            if isinstance(row, dict) and row.get("repository") == spec.model_repositories[0]
        ),
        None,
    )
    require(isinstance(primary, dict), "primary model cache receipt is missing")
    revision = primary.get("revision")
    require(
        isinstance(revision, str) and GIT_SHA_RE.fullmatch(revision) is not None,
        "primary model revision is not immutable",
    )
    raw_files = primary.get("files")
    require(isinstance(raw_files, list) and raw_files, "primary model file inventory is missing")
    files: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_files):
        require(isinstance(raw, dict), f"primary model file {index} is not an object")
        name = raw.get("path")
        size = raw.get("size_bytes")
        require(isinstance(name, str) and name, f"primary model file {index} name is missing")
        require(name not in names, f"duplicate primary model file: {name}")
        require(
            isinstance(size, int) and not isinstance(size, bool) and size > 0,
            f"primary model file {name} size is invalid",
        )
        row: dict[str, Any] = {"name": name, "size_bytes": size}
        recorded_sha = raw.get("sha256")
        if recorded_sha is not None:
            require(
                isinstance(recorded_sha, str) and SHA256_RE.fullmatch(recorded_sha) is not None,
                f"primary model file {name} recorded SHA256 is invalid",
            )
            row["sha256"] = recorded_sha
        files.append(row)
        names.add(name)
    return {"alias": spec.model_alias, "revision": revision, "files": files}


def portable_model_repositories(cache: Any, spec: BackendSpec) -> list[dict[str, Any]]:
    """Project every required repository without retaining cache-file refs."""

    require(isinstance(cache, dict), "model cache receipt is missing")
    raw_repositories = cache.get("repositories")
    require(isinstance(raw_repositories, list), "model cache repositories are missing")
    by_name = {
        row.get("repository"): row
        for row in raw_repositories
        if isinstance(row, dict) and isinstance(row.get("repository"), str)
    }
    require(
        set(by_name) == set(spec.model_repositories),
        "model cache repository denominator differs",
    )
    projected: list[dict[str, Any]] = []
    for repository in spec.model_repositories:
        row = by_name[repository]
        revision = row.get("revision")
        files = row.get("files")
        require(
            isinstance(revision, str) and GIT_SHA_RE.fullmatch(revision) is not None,
            f"{repository} revision is not immutable",
        )
        require(isinstance(files, list) and files, f"{repository} file inventory is empty")
        portable_files: list[dict[str, Any]] = []
        for index, raw_file in enumerate(files):
            require(isinstance(raw_file, dict), f"{repository} file {index} is invalid")
            name = raw_file.get("path")
            size = raw_file.get("size_bytes")
            require(isinstance(name, str) and name, f"{repository} file {index} name is missing")
            require(
                isinstance(size, int) and not isinstance(size, bool) and size > 0,
                f"{repository} file {name} size is invalid",
            )
            portable_file: dict[str, Any] = {"name": name, "size_bytes": size}
            for key in ("sha256", "blob_id"):
                value = raw_file.get(key)
                if isinstance(value, str) and value:
                    portable_file[key] = value
            portable_files.append(portable_file)
        projected.append(
            {
                "repository": repository,
                "revision": revision,
                "files": portable_files,
                "files_metadata_sha256": canonical_json_sha256(portable_files),
            }
        )
    return projected


def write_model_download_receipt(
    *,
    out: Path,
    spec: BackendSpec,
    model_cache: dict[str, Any],
    run_process: dict[str, Any],
    download_size_marker: str,
) -> dict[str, Any]:
    command_path = resolve_saved_ref(
        run_process.get("command"),
        root=out,
        label=f"{spec.backend} model-download run command",
        require_nonempty=True,
    )
    command = read_json_file(command_path, f"{spec.backend} model-download run command")
    require(isinstance(command, dict), "model-download run command receipt is invalid")
    require(
        command.get("status") == "pass"
        and command.get("returncode") == 0
        and isinstance(command.get("started_at"), str)
        and isinstance(command.get("finished_at"), str),
        "model-download run timing/status receipt differs",
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ferrum_v084_cold_cache_model_download_receipt",
        "status": "pass",
        "backend": spec.backend,
        "model_alias": spec.model_alias,
        "source": "https://huggingface.co",
        "cache_root": str(model_cache.get("root")),
        "fresh_cache": True,
        "download_complete": model_cache.get("incomplete_downloads") == [],
        "download_size_marker": download_size_marker,
        "repositories": portable_model_repositories(model_cache, spec),
        "execution": {
            "started_at": command["started_at"],
            "finished_at": command["finished_at"],
            "timeout_seconds": command.get("timeout_seconds"),
            "progress_signal": "run log bytes and fresh model-cache growth",
        },
        "run_process": run_process,
    }
    path = out / "cold-cache-model-download.json"
    write_json_atomic(path, receipt)
    return file_ref(path, relative_to=out)


def validate_process_receipt(
    raw: Any,
    *,
    root: Path,
    label: str,
    expected_status: str,
    expected_returncode: int | None,
) -> dict[str, Any]:
    require(isinstance(raw, dict), f"{label} process receipt is missing")
    for key in ("command", "stdout", "stderr", "progress"):
        resolve_saved_ref(
            raw.get(key),
            root=root,
            label=f"{label} {key}",
            require_nonempty=key in {"command", "progress"},
        )
    stdin = raw.get("stdin")
    if stdin is not None:
        resolve_saved_ref(stdin, root=root, label=f"{label} stdin", require_nonempty=True)
    command_path = resolve_saved_ref(
        raw["command"], root=root, label=f"{label} command", require_nonempty=True
    )
    command = read_json_file(command_path, f"{label} command receipt")
    require(isinstance(command, dict), f"{label} command receipt root is not an object")
    require(command.get("status") == expected_status, f"{label} command status differs")
    if expected_returncode is not None:
        require(
            command.get("returncode") == expected_returncode,
            f"{label} command return code differs",
        )
    require(
        isinstance(command.get("started_at"), str)
        and isinstance(command.get("finished_at"), str)
        and isinstance(command.get("timeout_seconds"), (int, float))
        and command["timeout_seconds"] > 0,
        f"{label} command timing receipt is incomplete",
    )
    require(
        command.get("stdout") == raw.get("stdout")
        and command.get("stderr") == raw.get("stderr"),
        f"{label} command/output references differ",
    )
    progress_path = resolve_saved_ref(
        raw["progress"], root=root, label=f"{label} progress", require_nonempty=True
    )
    previous_elapsed = -1.0
    rows = progress_path.read_bytes().splitlines()
    require(rows, f"{label} progress is empty")
    for index, line in enumerate(rows):
        row = parse_json_bytes(line, f"{label} progress row {index}")
        require(isinstance(row, dict), f"{label} progress row is invalid")
        elapsed = row.get("elapsed_seconds")
        require(
            isinstance(elapsed, (int, float)) and elapsed >= previous_elapsed,
            f"{label} progress elapsed time is not monotonic",
        )
        previous_elapsed = float(elapsed)
    if expected_status == "terminated":
        cleanup = command.get("cleanup_precondition")
        require(
            isinstance(cleanup, dict)
            and cleanup.get("process_alive") is True
            and isinstance(cleanup.get("observed_at"), str),
            f"{label} lacks live-process evidence before active cleanup",
        )
    return command


def write_portable_process_receipt(
    *,
    out: Path,
    label: str,
    raw: dict[str, Any],
    expected_status: str,
    expected_returncode: int | None,
    extracted_binary: dict[str, Any],
    network_environment: dict[str, Any],
) -> dict[str, Any]:
    command = validate_process_receipt(
        raw,
        root=out,
        label=label,
        expected_status=expected_status,
        expected_returncode=expected_returncode,
    )
    resolve_saved_ref(
        extracted_binary,
        root=out,
        label=f"{label} extracted binary",
        require_nonempty=True,
    )
    network_path = resolve_saved_ref(
        network_environment,
        root=out,
        label=f"{label} network environment",
        require_nonempty=True,
    )
    network_document = read_json_file(network_path, f"{label} network environment")
    validate_network_environment_document(
        network_document, consumer="ferrum-child-processes"
    )
    require(
        isinstance(command.get("environment"), dict)
        and command["environment"].get("network_routing") == network_document,
        f"{label} command/network environment receipts differ",
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ferrum_v084_portable_process_receipt",
        "label": label,
        "status": command["status"],
        "returncode": command.get("returncode"),
        "command": raw["command"],
        "stdout": raw["stdout"],
        "stderr": raw["stderr"],
        "progress": raw["progress"],
        "stdin": raw.get("stdin"),
        "extracted_binary": extracted_binary,
        "network_environment": network_environment,
    }
    destination = out / "portable-process-receipts" / f"{label}.json"
    write_json_atomic(destination, receipt)
    validate_portable_process_receipt(
        destination,
        root=out,
        label=label,
        expected_status=expected_status,
        expected_returncode=expected_returncode,
    )
    return file_ref(destination, relative_to=out)


def validate_portable_process_receipt(
    path: Path,
    *,
    root: Path,
    label: str,
    expected_status: str,
    expected_returncode: int | None,
) -> dict[str, Any]:
    receipt = read_json_file(path, f"{label} portable process receipt")
    require(
        isinstance(receipt, dict)
        and set(receipt)
        == {
            "schema_version",
            "artifact_type",
            "label",
            "status",
            "returncode",
            "command",
            "stdout",
            "stderr",
            "progress",
            "stdin",
            "extracted_binary",
            "network_environment",
        }
        and receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("artifact_type") == "ferrum_v084_portable_process_receipt"
        and receipt.get("label") == label
        and receipt.get("status") == expected_status,
        f"{label} portable process receipt schema/status differs",
    )
    if expected_returncode is not None:
        require(
            receipt.get("returncode") == expected_returncode,
            f"{label} portable process return code differs",
        )
    command = validate_process_receipt(
        {key: receipt[key] for key in ("command", "stdout", "stderr", "progress", "stdin")},
        root=root,
        label=label,
        expected_status=expected_status,
        expected_returncode=expected_returncode,
    )
    require(
        receipt.get("returncode") == command.get("returncode"),
        f"{label} portable process/command return codes differ",
    )
    resolve_saved_ref(
        receipt.get("extracted_binary"),
        root=root,
        label=f"{label} extracted binary",
        require_nonempty=True,
    )
    network_path = resolve_saved_ref(
        receipt.get("network_environment"),
        root=root,
        label=f"{label} network environment",
        require_nonempty=True,
    )
    network_document = read_json_file(network_path, f"{label} network environment")
    validate_network_environment_document(
        network_document, consumer="ferrum-child-processes"
    )
    require(
        isinstance(command.get("environment"), dict)
        and command["environment"].get("network_routing") == network_document,
        f"{label} portable command/network environment differs",
    )
    return receipt


def write_goal_log_scan(
    *,
    out: Path,
    process_receipts: list[tuple[str, dict[str, Any]]],
    response_refs: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    scanned: list[dict[str, Any]] = []
    for label, receipt in process_receipts:
        for stream in ("stdout", "stderr"):
            ref = receipt.get(stream)
            path = resolve_saved_ref(ref, root=out, label=f"{label} {stream}")
            text = strict_utf8(path.read_bytes(), f"{label} {stream}")
            scan_forbidden_text(f"{label} {stream}", text)
            scanned.append({"label": f"{label} {stream}", "file": ref})
    for label, ref in response_refs:
        path = resolve_saved_ref(ref, root=out, label=label, require_nonempty=True)
        text = strict_utf8(path.read_bytes(), label)
        scan_forbidden_text(label, text)
        scanned.append({"label": label, "file": ref})
    require(scanned, "goal E2E log scan has no evidence files")
    receipt_path = out / "readme-e2e-log-scan.json"
    write_json_atomic(
        receipt_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "ferrum_v084_readme_e2e_log_scan",
            "status": "pass",
            "forbidden_patterns": GOAL_LOG_SCAN_PATTERNS,
            "found": [],
            "files": scanned,
        },
    )
    return file_ref(receipt_path, relative_to=out)


def emit_goal_e2e_summary(
    *,
    out: Path,
    spec: BackendSpec,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    deadline_seconds: int,
    candidate_sha: str,
    asset_sha256: str,
    binary_sha256: str,
    extracted_binary: Path,
    readme_contract: dict[str, Any],
    environment_receipt: dict[str, Any],
    model_cache: dict[str, Any],
    identity_checks: dict[str, Any],
    doctor: dict[str, Any],
    run_result: dict[str, Any],
    serve: dict[str, Any],
    download_receipt: dict[str, Any],
    network_environment: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    require(elapsed_seconds <= deadline_seconds, "backend E2E exceeded its declared total deadline")
    allowed_overrides = {"HF_HOME"} | ({"LD_LIBRARY_PATH"} if spec.backend == "cuda" else set())
    require(
        isinstance(environment_receipt, dict)
        and set(environment_receipt.get("overrides", {})) == allowed_overrides
        and environment_receipt.get("effective_override_keys") == sorted(allowed_overrides)
        and environment_receipt.get("model_source_base_url") == "https://huggingface.co"
        and environment_receipt.get("hf_endpoint_removed") is True
        and environment_receipt.get("credentials_removed") is True,
        "backend E2E used an undocumented behavior-changing environment override",
    )
    require(
        readme_contract.get("download_size_announced_before_run") is True,
        "packaged README download size/order contract is missing",
    )
    require(
        isinstance(doctor.get("cache_before"), dict)
        and doctor["cache_before"].get("exists") is False
        and doctor.get("cache_unchanged") is True,
        "doctor did not preserve the absent fresh cache",
    )
    require(
        model_cache.get("incomplete_downloads") == [],
        "model cache contains incomplete downloads",
    )

    version_process = identity_checks.get("version")
    help_process = identity_checks.get("help")
    doctor_process = doctor.get("doctor")
    doctor_model_process = doctor.get("doctor_model")
    run_process = run_result.get("process")
    serve_process = serve.get("process")
    process_rows = (
        ("binary version", version_process, "pass", 0),
        ("binary help", help_process, "pass", 0),
        ("doctor", doctor_process, "pass", 0),
        ("doctor model", doctor_model_process, "pass", 0),
        ("README run", run_process, "pass", 0),
        ("README serve", serve_process, "terminated", None),
    )
    for label, receipt, status, returncode in process_rows:
        validate_process_receipt(
            receipt,
            root=out,
            label=label,
            expected_status=status,
            expected_returncode=returncode,
        )
    extracted_binary_ref = file_ref(extracted_binary, relative_to=out)
    child_network_ref = network_environment["child_processes"]
    portable_processes = {
        "binary_version": write_portable_process_receipt(
            out=out, label="binary-version", raw=version_process,
            expected_status="pass", expected_returncode=0,
            extracted_binary=extracted_binary_ref, network_environment=child_network_ref,
        ),
        "binary_help": write_portable_process_receipt(
            out=out, label="binary-help", raw=help_process,
            expected_status="pass", expected_returncode=0,
            extracted_binary=extracted_binary_ref, network_environment=child_network_ref,
        ),
        "doctor": write_portable_process_receipt(
            out=out, label="doctor-model", raw=doctor_model_process,
            expected_status="pass", expected_returncode=0,
            extracted_binary=extracted_binary_ref, network_environment=child_network_ref,
        ),
        "run": write_portable_process_receipt(
            out=out, label="readme-run", raw=run_process,
            expected_status="pass", expected_returncode=0,
            extracted_binary=extracted_binary_ref, network_environment=child_network_ref,
        ),
        "serve": write_portable_process_receipt(
            out=out, label="readme-serve", raw=serve_process,
            expected_status="terminated", expected_returncode=None,
            extracted_binary=extracted_binary_ref, network_environment=child_network_ref,
        ),
    }

    readiness = serve.get("readiness")
    readme_equivalence = serve.get("readme_equivalence")
    models = serve.get("models")
    nonstream = serve.get("nonstream")
    stream = serve.get("stream")
    require(isinstance(readiness, dict) and readiness.get("attempts", 0) > 0, "serve was not ready")
    require(
        isinstance(readme_equivalence, dict)
        and readme_equivalence.get("equivalent") is True
        and readme_equivalence.get("documented_argv") == readme_contract.get("serve_argv"),
        "serve execution was not explicitly normalized to the README command",
    )
    require(isinstance(models, dict), "models receipt is missing")
    require(isinstance(nonstream, dict), "non-stream chat receipt is missing")
    require(isinstance(stream, dict), "stream chat receipt is missing")
    models_exchange = models.get("exchange")
    chat_exchange = nonstream.get("exchange")
    stream_exchange = stream.get("exchange")
    for label, exchange in (
        ("models", models_exchange),
        ("non-stream chat", chat_exchange),
        ("stream chat", stream_exchange),
    ):
        require(isinstance(exchange, dict), f"{label} exchange receipt is missing")
        require(exchange.get("status") == 200, f"{label} HTTP status differs")
        resolve_saved_ref(
            exchange.get("exchange"), root=out, label=f"{label} exchange", require_nonempty=True
        )
        resolve_saved_ref(
            exchange.get("response"), root=out, label=f"{label} response", require_nonempty=True
        )

    stream_validation = stream.get("validation")
    require(isinstance(stream_validation, dict), "stream validation receipt is missing")
    usage = stream_validation.get("usage")
    require(isinstance(usage, dict), "stream usage receipt is missing")
    output_tokens = usage.get("completion_tokens")
    require(
        stream_validation.get("done_count") == 1
        and stream_validation.get("usage_object_count") == 1
        and isinstance(output_tokens, int)
        and not isinstance(output_tokens, bool)
        and output_tokens > 0,
        "stream usage/[DONE] receipt differs",
    )
    require(
        run_result.get("disable_thinking") is True
        and run_result.get("objective_response_nonempty") is True,
        "README run receipt differs",
    )
    nonstream_validation = nonstream.get("validation")
    require(
        isinstance(nonstream_validation, dict)
        and isinstance(nonstream_validation.get("content"), str)
        and bool(nonstream_validation["content"].strip()),
        "non-stream assistant content is empty",
    )
    served_model = models.get("served_model")
    require(isinstance(served_model, dict) and served_model.get("id") == "ferrum", "served model differs")

    tarball_download_evidence = download_receipt.get("receipt")
    resolve_saved_ref(
        tarball_download_evidence,
        root=out,
        label=f"{spec.backend} public tarball download receipt",
        require_nonempty=True,
    )
    model_download_evidence = write_model_download_receipt(
        out=out,
        spec=spec,
        model_cache=model_cache,
        run_process=run_process,
        download_size_marker=readme_contract["download_size_marker"],
    )
    log_scan_ref = write_goal_log_scan(
        out=out,
        process_receipts=[
            (label, receipt)
            for label, receipt, _status, _returncode in process_rows
            if isinstance(receipt, dict)
        ],
        response_refs=[
            ("models response", models_exchange["response"]),
            ("non-stream chat response", chat_exchange["response"]),
            ("stream chat response", stream_exchange["response"]),
        ],
    )
    evidence = {
        "binary_version": portable_processes["binary_version"],
        "binary_help": portable_processes["binary_help"],
        "doctor": portable_processes["doctor"],
        "download": model_download_evidence,
        "run": portable_processes["run"],
        "serve": portable_processes["serve"],
        "models": models_exchange["exchange"],
        "chat": chat_exchange["exchange"],
        "stream": stream_exchange["exchange"],
        "logs": log_scan_ref,
    }
    require(set(evidence) == GOAL_EVIDENCE_KEYS, "goal E2E evidence denominator differs")
    for label, ref in evidence.items():
        resolve_saved_ref(ref, root=out, label=f"goal E2E {label}", require_nonempty=True)

    artifact_dir = str(out)
    pass_line = f"FERRUM {VERSION} README E2E PASS: {spec.backend} {artifact_dir}"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": GOAL_E2E_ARTIFACT_TYPE,
        "status": "pass",
        "version": VERSION,
        "backend": spec.backend,
        "source_git_sha": candidate_sha,
        "asset_name": spec.asset_name,
        "asset_sha256": asset_sha256,
        "binary_sha256": binary_sha256,
        "model": goal_model_identity_from_cache(model_cache, spec),
        "cold_cache": {
            "fresh_cache": True,
            "cache_root": str(model_cache.get("root")),
            "undocumented_behavior_env": {
                "behavior_overrides": [],
                "network_routing_is_behavior_override": False,
                "network_environment": network_environment,
            },
            "download_size_announced": True,
            "download_complete": True,
        },
        "execution": {
            "started_at": started_at,
            "finished_at": finished_at,
            "deadline_seconds": deadline_seconds,
            "progress_signal": "download bytes, process log bytes, and model-cache growth",
        },
        "network_environment": network_environment,
        "checks": {
            "binary_version": True,
            "binary_help": True,
            "doctor": True,
            "run": {"exit_code": 0, "non_empty": True, "disable_thinking": True},
            "serve": {"ready": True},
            "models": {"http_status": 200, "model_present": True},
            "chat": {"http_status": 200, "non_empty_content": True},
            "stream": {
                "http_status": 200,
                "done_count": 1,
                "usage_chunks": 1,
                "output_tokens": output_tokens,
            },
            "log_scan": {
                "forbidden_patterns": GOAL_LOG_SCAN_PATTERNS,
                "found": [],
            },
        },
        "evidence": evidence,
        "artifact_dir": artifact_dir,
        "pass_line": pass_line,
    }
    summary_path = out / f"ferrum-{VERSION}-readme-e2e-{spec.backend}.json"
    write_json_atomic(summary_path, summary)
    return summary_path, summary


def scan_forbidden_text(label: str, text: str) -> None:
    for name, pattern in FORBIDDEN_TEXT_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            excerpt = text[max(0, match.start() - 40) : match.end() + 40]
            raise GateError(
                f"forbidden {name} pattern in {label}: {excerpt!r}"
            )


def read_and_scan_utf8(path: Path, label: str) -> str:
    text = strict_utf8(path.read_bytes(), label)
    scan_forbidden_text(label, text)
    return text


def positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not (0.1 <= parsed <= 86_400):
        raise argparse.ArgumentTypeError("must be between 0.1 and 86400 seconds")
    return parsed


def valid_port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def valid_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError("must be a lowercase 64-character SHA256")
    return normalized


def prepare_fresh_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    require(not resolved.exists(), f"output directory must be fresh: {resolved}")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def github_digest(value: Any, label: str) -> str:
    require(isinstance(value, str), f"{label} GitHub digest is missing")
    prefix = "sha256:"
    require(value.startswith(prefix), f"{label} GitHub digest is not sha256")
    digest = value[len(prefix) :].lower()
    require(SHA256_RE.fullmatch(digest) is not None, f"{label} GitHub digest is invalid")
    return digest


def expected_public_asset_url(name: str) -> str:
    return f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{name}"


def validate_https_url(url: str, *, initial_asset: bool = False) -> None:
    parsed = urllib.parse.urlparse(url)
    require(parsed.scheme == "https", f"URL is not HTTPS: {url}")
    host = (parsed.hostname or "").lower()
    allowed = (
        host in {"github.com", "api.github.com"}
        or host.endswith(".githubusercontent.com")
    )
    require(allowed, f"URL host is not a GitHub public host: {url}")
    if initial_asset:
        require(
            url == expected_public_asset_url(PurePosixPath(parsed.path).name),
            f"asset URL is not the fixed public {TAG} path: {url}",
        )


class SafeGithubRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        validate_https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


HTTP_OPENER = urllib.request.build_opener(SafeGithubRedirectHandler())


def release_identity_projection(release: dict[str, Any]) -> dict[str, Any]:
    assets = release.get("assets")
    require(isinstance(assets, list), "GitHub release assets must be an array")
    projected_assets: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_ids: set[int] = set()
    for index, raw in enumerate(assets):
        require(isinstance(raw, dict), f"GitHub release asset {index} is not an object")
        name = raw.get("name")
        asset_id = raw.get("id")
        require(isinstance(name, str) and name, f"GitHub asset {index} name is invalid")
        require(name not in seen_names, f"duplicate GitHub asset name: {name}")
        require(
            isinstance(asset_id, int) and not isinstance(asset_id, bool) and asset_id > 0,
            f"GitHub asset {name} id is invalid",
        )
        require(asset_id not in seen_ids, f"duplicate GitHub asset id: {asset_id}")
        seen_names.add(name)
        seen_ids.add(asset_id)
        size = raw.get("size")
        require(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            f"GitHub asset {name} size is invalid",
        )
        projected_assets.append(
            {
                "id": asset_id,
                "name": name,
                "size": size,
                "digest": raw.get("digest"),
                "state": raw.get("state"),
                "content_type": raw.get("content_type"),
                "browser_download_url": raw.get("browser_download_url"),
                "created_at": raw.get("created_at"),
                "updated_at": raw.get("updated_at"),
            }
        )
    projected_assets.sort(key=lambda value: value["name"])
    return {
        "id": release.get("id"),
        "tag_name": release.get("tag_name"),
        "target_commitish": release.get("target_commitish"),
        "name": release.get("name"),
        "draft": release.get("draft"),
        "prerelease": release.get("prerelease"),
        "created_at": release.get("created_at"),
        "published_at": release.get("published_at"),
        "html_url": release.get("html_url"),
        "assets": projected_assets,
    }


def validate_release_snapshot(
    raw: Any, spec: BackendSpec
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(isinstance(raw, dict), "GitHub release API root is not an object")
    release_id = raw.get("id")
    require(
        isinstance(release_id, int) and not isinstance(release_id, bool) and release_id > 0,
        "GitHub release id is invalid",
    )
    require(raw.get("tag_name") == TAG, f"GitHub release tag is not {TAG}")
    require(raw.get("draft") is False, "GitHub release must be publicly visible, not draft")
    require(raw.get("prerelease") is True, "GitHub release must have prerelease=true")
    assets = raw.get("assets")
    require(isinstance(assets, list), "GitHub release assets must be an array")
    by_name: dict[str, dict[str, Any]] = {}
    for item in assets:
        require(isinstance(item, dict), "GitHub release asset is not an object")
        name = item.get("name")
        require(isinstance(name, str) and name, "GitHub release asset name is invalid")
        require(name not in by_name, f"duplicate GitHub release asset: {name}")
        by_name[name] = item

    selected: dict[str, dict[str, Any]] = {}
    for name in backend_download_asset_names(spec):
        require(name in by_name, f"required public release asset is missing: {name}")
        item = by_name[name]
        asset_id = item.get("id")
        size = item.get("size")
        require(
            isinstance(asset_id, int) and not isinstance(asset_id, bool) and asset_id > 0,
            f"GitHub asset {name} id is invalid",
        )
        require(
            isinstance(size, int) and not isinstance(size, bool) and size > 0,
            f"GitHub asset {name} size must be positive",
        )
        require(item.get("state") == "uploaded", f"GitHub asset {name} is not uploaded")
        digest = github_digest(item.get("digest"), name)
        public_url = item.get("browser_download_url")
        require(isinstance(public_url, str), f"GitHub asset {name} URL is missing")
        require(
            public_url == expected_public_asset_url(name),
            f"GitHub asset {name} is not on the fixed public {TAG} path",
        )
        validate_https_url(public_url, initial_asset=True)
        selected[name] = {
            "id": asset_id,
            "name": name,
            "size_bytes": size,
            "digest": f"sha256:{digest}",
            "sha256": digest,
            "state": "uploaded",
            "content_type": item.get("content_type"),
            "browser_download_url": public_url,
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
        }

    projection = release_identity_projection(raw)
    identity = {
        "id": release_id,
        "tag_name": TAG,
        "draft": False,
        "prerelease": True,
        "html_url": raw.get("html_url"),
        "target_commitish": raw.get("target_commitish"),
        "created_at": raw.get("created_at"),
        "published_at": raw.get("published_at"),
        "immutable_snapshot": projection,
        "immutable_snapshot_sha256": canonical_json_sha256(projection),
    }
    return identity, selected


def fetch_public_release_snapshot(out: Path, timeout: float) -> tuple[Any, dict[str, Any]]:
    raw_path = out / "github-release-api.json"
    meta_path = out / "github-release-api.fetch.json"
    started_at = iso_now()
    started = time.monotonic()
    deadline = started + timeout
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "github_release_api_fetch",
        "url": RELEASE_API_URL,
        "started_at": started_at,
        "timeout_seconds": timeout,
        "deadline_monotonic": deadline,
        "status": "running",
    }
    write_json_atomic(meta_path, metadata)
    request = urllib.request.Request(
        RELEASE_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with HTTP_OPENER.open(request, timeout=min(timeout, 30.0)) as response:
            require(response.status == 200, f"GitHub release API returned HTTP {response.status}")
            data = read_bounded_http_body(
                response,
                deadline=deadline,
                max_bytes=MAX_API_BYTES,
                label="GitHub release API",
            )
            effective_url = response.geturl()
            validate_https_url(effective_url)
        raw_path.write_bytes(data)
        parsed = parse_json_bytes(data, "GitHub release API response")
        metadata.update(
            {
                "status": "pass",
                "http_status": 200,
                "effective_url": effective_url,
                "response": file_ref(raw_path, relative_to=out),
                "finished_at": iso_now(),
                "duration_seconds": time.monotonic() - started,
            }
        )
        write_json_atomic(meta_path, metadata)
        return parsed, metadata
    except Exception as error:
        metadata.update(
            {
                "status": "fail",
                "error": str(error),
                "finished_at": iso_now(),
                "duration_seconds": time.monotonic() - started,
            }
        )
        write_json_atomic(meta_path, metadata)
        raise


def read_bounded_http_body(
    response: Any,
    *,
    deadline: float,
    max_bytes: int,
    label: str,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        require(time.monotonic() <= deadline, f"{label} exceeded its hard deadline")
        chunk = response.read(min(DOWNLOAD_CHUNK_BYTES, max_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        require(total <= max_bytes, f"{label} exceeded {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def download_public_asset(
    identity: dict[str, Any],
    destination: Path,
    *,
    timeout: float,
    progress_interval: float,
    out: Path,
) -> dict[str, Any]:
    name = identity["name"]
    url = identity["browser_download_url"]
    require(url == expected_public_asset_url(name), f"unexpected public URL for {name}")
    validate_https_url(url, initial_asset=True)
    require(not destination.exists(), f"download destination already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.part")
    require(not temporary.exists(), f"partial download already exists: {temporary}")
    progress_path = out / "downloads" / f"{name}.progress.jsonl"
    metadata_path = out / "downloads" / f"{name}.download.json"
    started = time.monotonic()
    deadline = started + timeout
    started_at = iso_now()
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "public_github_release_asset_download",
        "asset": identity,
        "url": url,
        "started_at": started_at,
        "timeout_seconds": timeout,
        "progress_interval_seconds": progress_interval,
        "progress_log": str(progress_path.relative_to(out)),
        "status": "running",
    }
    write_json_atomic(metadata_path, metadata)
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/octet-stream", "User-Agent": USER_AGENT},
    )
    downloaded = 0
    next_progress = started
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with HTTP_OPENER.open(request, timeout=min(timeout, 30.0)) as response:
            require(response.status == 200, f"download {name} returned HTTP {response.status}")
            http_status = response.status
            effective_url = response.geturl()
            validate_https_url(effective_url)
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    parsed_length = int(content_length)
                except ValueError as error:
                    raise GateError(f"download {name} Content-Length is invalid") from error
                require(
                    parsed_length == identity["size_bytes"],
                    f"download {name} Content-Length differs from GitHub API size",
                )
            digest = hashlib.sha256()
            with temporary.open("xb") as handle:
                while True:
                    now = time.monotonic()
                    require(now <= deadline, f"download {name} exceeded its hard deadline")
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    require(
                        downloaded <= identity["size_bytes"],
                        f"download {name} exceeded GitHub API size",
                    )
                    if now >= next_progress:
                        sample = {
                            "observed_at": iso_now(),
                            "elapsed_seconds": now - started,
                            "bytes_downloaded": downloaded,
                            "expected_bytes": identity["size_bytes"],
                        }
                        append_jsonl(progress_path, sample)
                        print(
                            f"download {name}: {downloaded}/{identity['size_bytes']} bytes",
                            flush=True,
                        )
                        next_progress = now + progress_interval
                handle.flush()
                os.fsync(handle.fileno())
        require(time.monotonic() <= deadline, f"download {name} exceeded its hard deadline")
        final_digest = digest.hexdigest()
        require(
            downloaded == identity["size_bytes"],
            f"download {name} size {downloaded} differs from GitHub API {identity['size_bytes']}",
        )
        require(
            final_digest == identity["sha256"],
            f"download {name} SHA256 differs from GitHub digest",
        )
        os.replace(temporary, destination)
        finished_monotonic = time.monotonic()
        require(finished_monotonic <= deadline, f"download {name} exceeded its hard deadline")
        duration_seconds = finished_monotonic - started
        append_jsonl(
            progress_path,
            {
                "observed_at": iso_now(),
                "elapsed_seconds": duration_seconds,
                "bytes_downloaded": downloaded,
                "expected_bytes": identity["size_bytes"],
                "complete": True,
            },
        )
        metadata.update(
            {
                "status": "pass",
                "http_status": http_status,
                "effective_url": effective_url,
                "download": file_ref(destination, relative_to=out),
                "finished_at": iso_now(),
                "duration_seconds": duration_seconds,
            }
        )
        write_json_atomic(metadata_path, metadata)
        metadata["receipt"] = file_ref(metadata_path, relative_to=out)
        return metadata
    except Exception as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        metadata.update(
            {
                "status": "fail",
                "bytes_downloaded": downloaded,
                "error": str(error),
                "finished_at": iso_now(),
                "duration_seconds": time.monotonic() - started,
            }
        )
        write_json_atomic(metadata_path, metadata)
        raise


def parse_checksum_file(data: bytes, expected_filename: str, label: str) -> str:
    text = strict_utf8(data, label)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    require(len(lines) == 1, f"{label} must contain exactly one checksum line")
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+[*]?(.+)", lines[0])
    require(match is not None, f"{label} has invalid SHA256 syntax")
    assert match is not None
    filename = match.group(2).strip()
    require(filename == expected_filename, f"{label} names {filename!r}, expected {expected_filename!r}")
    return match.group(1).lower()


def safe_extract_tarball(tar_path: Path, destination: Path) -> list[dict[str, Any]]:
    require(not destination.exists(), f"extraction directory must be fresh: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    extracted: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_size = 0
    try:
        with tarfile.open(tar_path, mode="r:gz") as archive:
            members = archive.getmembers()
            require(members, "release tarball is empty")
            require(
                len(members) <= MAX_TAR_MEMBERS,
                f"release tarball has more than {MAX_TAR_MEMBERS} members",
            )
            for member in members:
                name = member.name
                require("\\" not in name and "\x00" not in name, f"unsafe tar member name: {name!r}")
                pure = PurePosixPath(name)
                require(not pure.is_absolute(), f"absolute tar member path: {name!r}")
                parts = tuple(part for part in pure.parts if part not in {"", "."})
                require(parts and ".." not in parts, f"traversal tar member path: {name!r}")
                normalized = PurePosixPath(*parts).as_posix()
                require(normalized not in seen, f"duplicate tar member path: {normalized}")
                seen.add(normalized)
                require(
                    member.isdir() or member.isfile(),
                    f"tar member is not a regular file/directory: {normalized}",
                )
                require(not member.issparse(), f"sparse tar member is forbidden: {normalized}")
                target = root.joinpath(*parts)
                try:
                    target.relative_to(root)
                except ValueError as error:
                    raise GateError(f"tar member escapes extraction root: {normalized}") from error
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=False)
                    target.chmod(member.mode & 0o777)
                    extracted.append({"path": normalized, "type": "directory", "size_bytes": 0})
                    continue
                require(member.size >= 0, f"tar member has negative size: {normalized}")
                total_size += member.size
                require(
                    total_size <= MAX_TAR_EXPANDED_BYTES,
                    "release tarball exceeds expanded-size limit",
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                require(not target.exists(), f"tar extraction target already exists: {normalized}")
                source = archive.extractfile(member)
                require(source is not None, f"cannot read tar member: {normalized}")
                written = 0
                with source, target.open("xb") as handle:
                    while True:
                        chunk = source.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
                        require(written <= member.size, f"tar member exceeds declared size: {normalized}")
                require(written == member.size, f"tar member size mismatch: {normalized}")
                target.chmod(member.mode & 0o777)
                extracted.append(
                    {
                        "path": normalized,
                        "type": "file",
                        "size_bytes": written,
                        "sha256": sha256_file(target),
                        "mode": oct(member.mode & 0o777),
                    }
                )
    except Exception:
        # Leave the bounded partial tree as diagnostic evidence; never execute it.
        raise
    binary = destination / "ferrum"
    require(binary.is_file() and not binary.is_symlink(), "tarball must contain regular root ferrum")
    require(os.access(binary, os.X_OK), "extracted ferrum is not executable")
    return extracted


def validate_manifest_common(
    document: Any,
    *,
    label: str,
    spec: BackendSpec,
    asset_sha256: str,
    binary_sha256: str,
) -> dict[str, Any]:
    require(isinstance(document, dict), f"{label} root must be an object")
    require(document.get("schema_version") == 1, f"{label} schema_version must be 1")
    require(document.get("asset_name") == spec.asset_name, f"{label} asset_name differs")
    require(document.get("asset_sha256") == asset_sha256, f"{label} asset SHA256 differs")
    require(document.get("binary_name") == "ferrum", f"{label} binary_name differs")
    require(document.get("binary_sha256") == binary_sha256, f"{label} binary SHA256 differs")
    candidate_sha = document.get("release_candidate_sha")
    candidate_tag = document.get("release_candidate_tag")
    require(
        isinstance(candidate_sha, str) and GIT_SHA_RE.fullmatch(candidate_sha) is not None,
        f"{label} release_candidate_sha is invalid",
    )
    require(
        isinstance(candidate_tag, str) and RC_TAG_RE.fullmatch(candidate_tag) is not None,
        f"{label} release_candidate_tag is not v0.8.4-rc.N",
    )
    staging_label = document.get("staging_label")
    require(
        isinstance(staging_label, str) and staging_label.startswith("v0.8.4-rc"),
        f"{label} staging_label is invalid",
    )
    for field in ("workflow_run_id", "workflow_run_attempt"):
        value = document.get(field)
        require(
            (isinstance(value, str) and value.strip() != "")
            or (isinstance(value, int) and not isinstance(value, bool) and value > 0),
            f"{label} {field} is invalid",
        )
    return {
        "release_candidate_sha": candidate_sha,
        "release_candidate_tag": candidate_tag,
        "staging_label": staging_label,
        "workflow_run_id": str(document["workflow_run_id"]),
        "workflow_run_attempt": str(document["workflow_run_attempt"]),
    }


def validate_adjacent_bundle(
    download_dir: Path,
    spec: BackendSpec,
    *,
    asset_sha256: str,
    binary_sha256: str,
) -> dict[str, Any]:
    checksum_path = download_dir / f"{spec.asset_name}.sha256"
    binary_checksum_path = download_dir / f"{spec.asset_name}.binary.sha256"
    adjacent_asset_sha = parse_checksum_file(
        checksum_path.read_bytes(), spec.asset_name, checksum_path.name
    )
    adjacent_binary_sha = parse_checksum_file(
        binary_checksum_path.read_bytes(), "ferrum", binary_checksum_path.name
    )
    require(adjacent_asset_sha == asset_sha256, "adjacent asset checksum differs")
    require(adjacent_binary_sha == binary_sha256, "adjacent binary checksum differs")

    manifest_paths = {
        "version": download_dir / f"{spec.asset_name}.version.json",
        "dependency": download_dir / f"{spec.asset_name}.dependency.json",
        "abi": download_dir / f"{spec.asset_name}.abi.json",
    }
    documents = {
        name: read_json_file(path, f"{name} manifest")
        for name, path in manifest_paths.items()
    }
    common_rows = {
        name: validate_manifest_common(
            document,
            label=f"{name} manifest",
            spec=spec,
            asset_sha256=asset_sha256,
            binary_sha256=binary_sha256,
        )
        for name, document in documents.items()
    }
    common_identities = {
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in common_rows.values()
    }
    require(len(common_identities) == 1, "adjacent manifest release identities differ")
    common = next(iter(common_rows.values()))

    version_manifest = documents["version"]
    require(version_manifest.get("version") == VERSION, "version manifest is not 0.8.4")

    dependency_manifest = documents["dependency"]
    require(
        dependency_manifest.get("audit_file") == spec.dependency_audit_name,
        "dependency manifest audit_file differs",
    )
    audit_path = download_dir / spec.dependency_audit_name
    audit_sha = sha256_file(audit_path)
    require(
        dependency_manifest.get("audit_sha256") == audit_sha,
        "dependency manifest audit SHA256 differs",
    )
    require(
        dependency_manifest.get("forbidden_runtime_linkage_found") is False,
        "dependency manifest reports forbidden runtime linkage",
    )
    linkage = dependency_manifest.get("forbidden_runtime_linkage")
    require(isinstance(linkage, list), "dependency manifest linkage list is missing")
    audit_text = strict_utf8(audit_path.read_bytes(), "dependency audit")
    for name in ("python", "torch", "vllm"):
        require(name in linkage, f"dependency manifest does not forbid {name}")
        require(
            re.search(rf"\b{re.escape(name)}\b", audit_text, re.IGNORECASE) is None,
            f"dependency audit contains forbidden linkage {name}",
        )

    abi_manifest = documents["abi"]
    require(abi_manifest.get("backend") == spec.backend, "ABI manifest backend differs")
    require(
        abi_manifest.get("target_triple") == spec.target_triple,
        "ABI manifest target_triple differs",
    )
    require(
        abi_manifest.get("dependency_audit_sha256") == audit_sha,
        "ABI manifest dependency audit SHA256 differs",
    )
    if spec.backend == "cuda":
        require(
            str(abi_manifest.get("cuda_compute_capability")) == "89",
            "CUDA ABI manifest compute capability is not 89",
        )

    return {
        "asset_checksum": file_ref(checksum_path, relative_to=download_dir.parent),
        "binary_checksum": file_ref(binary_checksum_path, relative_to=download_dir.parent),
        "version_manifest": file_ref(manifest_paths["version"], relative_to=download_dir.parent),
        "dependency_manifest": file_ref(
            manifest_paths["dependency"], relative_to=download_dir.parent
        ),
        "abi_manifest": file_ref(manifest_paths["abi"], relative_to=download_dir.parent),
        "dependency_audit": file_ref(audit_path, relative_to=download_dir.parent),
        "release_candidate": common,
    }


def cache_snapshot(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {
            "exists": False,
            "entries": [],
            "entry_count": 0,
            "metadata_sha256": canonical_json_sha256([]),
        }
    require(root.is_dir() and not root.is_symlink(), f"model cache is not a regular directory: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        stat = path.lstat()
        if path.is_symlink():
            entries.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "target": os.readlink(path),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        elif path.is_dir():
            entries.append(
                {
                    "path": relative,
                    "type": "directory",
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        elif path.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        else:
            raise GateError(f"model cache contains unsupported entry: {relative}")
    return {
        "exists": True,
        "entries": entries,
        "entry_count": len(entries),
        "metadata_sha256": canonical_json_sha256(entries),
    }


def cache_progress(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"cache_exists": False, "entry_count": 0, "regular_file_bytes": 0}
    count = 0
    regular_bytes = 0
    for path in root.rglob("*"):
        count += 1
        if path.is_file() and not path.is_symlink():
            regular_bytes += path.stat().st_size
    return {
        "cache_exists": True,
        "entry_count": count,
        "regular_file_bytes": regular_bytes,
    }


def assert_cache_unchanged(before: dict[str, Any], after: dict[str, Any], label: str) -> None:
    require(before == after, f"{label} mutated the fresh model cache")


def repository_cache_dir(cache_root: Path, repository: str) -> Path:
    return cache_root / "hub" / f"models--{repository.replace('/', '--')}"


def model_cache_receipt(cache_root: Path, spec: BackendSpec) -> dict[str, Any]:
    require(
        cache_root.is_dir() and not cache_root.is_symlink(),
        "model run did not create a regular explicit cache root",
    )
    cache_root_resolved = cache_root.resolve()
    incomplete = sorted(
        path.relative_to(cache_root).as_posix()
        for path in cache_root.rglob("*.incomplete")
    )
    require(not incomplete, f"model cache contains incomplete downloads: {incomplete}")
    requirements = dict(spec.required_model_files)
    repositories: list[dict[str, Any]] = []
    for repository in spec.model_repositories:
        repo_dir = repository_cache_dir(cache_root, repository)
        require(
            repo_dir.is_dir()
            and not repo_dir.is_symlink()
            and repo_dir.resolve().is_relative_to(cache_root_resolved),
            f"model repository cache is missing, linked, or escaping: {repository}",
        )
        ref_path = repo_dir / "refs" / "main"
        require(
            ref_path.parent.is_dir()
            and not ref_path.parent.is_symlink()
            and ref_path.is_file()
            and not ref_path.is_symlink()
            and ref_path.resolve().is_relative_to(repo_dir.resolve()),
            f"{repository} refs/main is missing, linked, or escaping",
        )
        revision = strict_utf8(ref_path.read_bytes(), f"{repository} refs/main").strip()
        require(
            GIT_SHA_RE.fullmatch(revision) is not None,
            f"{repository} refs/main is not a full immutable revision",
        )
        snapshot = repo_dir / "snapshots" / revision
        require(
            snapshot.parent.is_dir()
            and not snapshot.parent.is_symlink()
            and snapshot.is_dir()
            and not snapshot.is_symlink()
            and snapshot.resolve().is_relative_to(repo_dir.resolve()),
            f"{repository} immutable snapshot is missing, linked, or escaping",
        )
        blobs = repo_dir / "blobs"
        require(
            blobs.is_dir() and not blobs.is_symlink(),
            f"{repository} blob directory is missing or linked",
        )
        files: list[dict[str, Any]] = []
        for path in sorted(snapshot.rglob("*"), key=lambda item: item.relative_to(snapshot).as_posix()):
            if path.is_dir():
                require(not path.is_symlink(), f"cached model directory is linked: {path}")
                continue
            require(path.is_file(), f"unsupported cached model entry: {path}")
            relative = path.relative_to(snapshot).as_posix()
            resolved = path.resolve(strict=True)
            require(
                resolved.is_file() and resolved.is_relative_to(repo_dir.resolve()),
                f"cached model file escapes repository: {repository}/{relative}",
            )
            stat = path.stat()
            row: dict[str, Any] = {
                "path": relative,
                "size_bytes": stat.st_size,
                "is_symlink": path.is_symlink(),
            }
            if path.is_symlink():
                target = os.readlink(path)
                require(
                    not Path(target).is_absolute() and resolved.is_relative_to(blobs.resolve()),
                    f"cached model symlink does not resolve to repository blobs: {repository}/{relative}",
                )
                row["blob_target"] = target
                row["blob_id"] = resolved.name
            elif stat.st_size <= 16 * 1024 * 1024:
                row["sha256"] = sha256_file(path)
            files.append(row)
        present = {row["path"] for row in files}
        for required_file in requirements[repository]:
            require(
                required_file in present,
                f"{repository}@{revision} is missing {required_file}",
            )
        repositories.append(
            {
                "repository": repository,
                "revision": revision,
                "ref": file_ref(ref_path, relative_to=cache_root),
                "snapshot_path": str(snapshot.relative_to(cache_root)),
                "files": files,
                "file_count": len(files),
                "files_metadata_sha256": canonical_json_sha256(files),
            }
        )
    progress = cache_progress(cache_root)
    return {
        "root": str(cache_root),
        "repositories": repositories,
        "incomplete_downloads": [],
        **progress,
    }


def validate_readme_contract_texts(text: str, chinese: str, spec: BackendSpec) -> dict[str, Any]:
    global_commands = ("ferrum --version", "ferrum --help", "ferrum doctor")
    doctor_command = f"ferrum doctor {spec.model_alias}"
    run_command = f"ferrum run {spec.model_alias} --disable-thinking"
    serve_command = (
        f"ferrum serve --model {spec.model_alias} --served-model-name ferrum "
        "--disable-thinking --port 8000"
    )
    normalized_structures: dict[str, list[str]] = {}
    for language, document in (("english", text), ("chinese", chinese)):
        positions = [document.find(command) for command in global_commands]
        platform_doctor_position = document.find(doctor_command)
        platform_run_position = document.find(run_command)
        platform_serve_position = document.find(serve_command)
        require(all(position >= 0 for position in positions), f"{language} README omits the global version/help/doctor validation block")
        require(positions == sorted(positions), f"{language} README global validation block order differs")
        require(
            positions[-1] < platform_doctor_position < platform_run_position < platform_serve_position,
            f"{language} README global validation block is not before the platform model flow",
        )
        normalized_structures[language] = [
            "global-version", "global-help", "global-doctor", "platform-doctor", "platform-run", "platform-serve"
        ]
    require(
        normalized_structures["english"] == normalized_structures["chinese"],
        "English and Chinese README validation/model-flow structures differ",
    )
    size_position = text.find(spec.download_size_marker)
    run_position = text.find(run_command)
    require(size_position >= 0, f"packaged README omits {spec.download_size_marker} download size")
    require(run_position >= 0, f"packaged README omits exact run command: {run_command}")
    require(size_position < run_position, "packaged README states download size after the run command")
    require(doctor_command in text, f"packaged README omits exact doctor command: {doctor_command}")
    require(serve_command in text, f"packaged README omits exact serve command: {serve_command}")
    require("hung" in text.lower(), "packaged README omits the first-download hung explanation")
    require(
        "Omit the flag" in text and "default reasoning" in text,
        "packaged README omits how to restore default reasoning",
    )
    chinese_size_position = chinese.find(spec.download_size_marker)
    chinese_run_position = chinese.find(run_command)
    require(
        chinese_size_position >= 0 and chinese_size_position < chinese_run_position,
        "README_zh.md omits the download size before the exact run command",
    )
    require(
        doctor_command in chinese and serve_command in chinese and "卡住" in chinese,
        "README_zh.md quick-start commands or hung explanation differ",
    )
    require(
        "删除该参数" in chinese and "默认" in chinese and "推理" in chinese,
        "README_zh.md omits how to restore default reasoning",
    )
    return {
        "doctor_command": doctor_command,
        "run_command": run_command,
        "serve_command": serve_command,
        "serve_argv": shlex.split(serve_command),
        "global_validation_commands": list(global_commands),
        "normalized_structure": normalized_structures["english"],
        "languages": {"english": "packaged README.md", "chinese": "repository README_zh.md"},
        "download_size_marker": spec.download_size_marker,
        "download_size_announced_before_run": True,
    }


def validate_packaged_readme_contract(runtime_dir: Path, spec: BackendSpec) -> dict[str, Any]:
    readme_path = runtime_dir / "README.md"
    require(
        readme_path.is_file() and not readme_path.is_symlink(),
        "release tarball does not contain a regular root README.md",
    )
    chinese_path = Path(__file__).resolve().parents[2] / "README_zh.md"
    require(
        chinese_path.is_file() and not chinese_path.is_symlink(),
        "repository README_zh.md is missing or linked",
    )
    receipt = validate_readme_contract_texts(
        strict_utf8(readme_path.read_bytes(), "packaged README.md"),
        strict_utf8(chinese_path.read_bytes(), "README_zh.md"),
        spec,
    )
    receipt["file"] = file_ref(readme_path, relative_to=runtime_dir.parent)
    return receipt


NETWORK_ENV_KEYS = (
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy",
    "NO_PROXY", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
)
CUSTOM_CA_KEYS = {"SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"}


def network_environment_document(environment: dict[str, str], *, consumer: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key in NETWORK_ENV_KEYS:
        value = environment.get(key)
        if value is None:
            continue
        loopback = False
        if "PROXY" in key.upper():
            if key.upper() == "NO_PROXY":
                hosts = {part.strip().lower() for part in value.split(",")}
                loopback = bool(hosts & {"localhost", "127.0.0.1", "::1"})
            else:
                host = (urllib.parse.urlsplit(value).hostname or "").lower()
                loopback = host in {"localhost", "127.0.0.1", "::1"}
        rows.append(
            {
                "key": key,
                "value_sha256": sha256_bytes(value.encode("utf-8")),
                "loopback": loopback,
                "custom_ca": key in CUSTOM_CA_KEYS,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ferrum_v084_sanitized_network_environment_receipt",
        "consumer": consumer,
        "secret_values_recorded": False,
        "variables": rows,
    }


def validate_network_environment_document(raw: Any, *, consumer: str) -> None:
    require(isinstance(raw, dict), f"{consumer} network environment receipt is missing")
    require(
        set(raw)
        == {
            "schema_version",
            "artifact_type",
            "consumer",
            "secret_values_recorded",
            "variables",
        }
        and raw.get("schema_version") == SCHEMA_VERSION
        and raw.get("artifact_type")
        == "ferrum_v084_sanitized_network_environment_receipt"
        and raw.get("consumer") == consumer
        and raw.get("secret_values_recorded") is False,
        f"{consumer} network environment receipt schema differs",
    )
    rows = raw.get("variables")
    require(isinstance(rows, list), f"{consumer} network environment variables differ")
    seen: set[str] = set()
    for row in rows:
        require(
            isinstance(row, dict)
            and set(row) == {"key", "value_sha256", "loopback", "custom_ca"},
            f"{consumer} network environment row schema differs",
        )
        key = row.get("key")
        require(
            isinstance(key, str)
            and key in NETWORK_ENV_KEYS
            and key not in seen
            and isinstance(row.get("value_sha256"), str)
            and SHA256_RE.fullmatch(row["value_sha256"]) is not None
            and isinstance(row.get("loopback"), bool)
            and isinstance(row.get("custom_ca"), bool)
            and row["custom_ca"] == (key in CUSTOM_CA_KEYS),
            f"{consumer} network environment row differs",
        )
        seen.add(key)


def write_network_environment_receipt(
    out: Path, *, label: str, environment: dict[str, str], consumer: str
) -> dict[str, Any]:
    path = out / "network-environment" / f"{label}.json"
    document = network_environment_document(environment, consumer=consumer)
    validate_network_environment_document(document, consumer=consumer)
    write_json_atomic(path, document)
    return file_ref(path, relative_to=out)


def sanitized_child_environment(cache_root: Path, spec: BackendSpec) -> tuple[dict[str, str], dict[str, Any]]:
    network_keys = {
        "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"
    }
    operational_keys = {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TMP", "TEMP",
        "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    }
    inherited_allowlist = operational_keys | network_keys
    environment = {
        key: value for key, value in os.environ.items() if key in inherited_allowlist
    }
    removed = sorted(set(os.environ) - set(environment))
    behavior_keys = {
        "HF_ENDPOINT",
        "TRANSFORMERS_CACHE",
        "CUDA_VISIBLE_DEVICES",
        "RUST_LOG",
    }
    removed_behavior = sorted(
        key for key in removed if key in behavior_keys or key.startswith("FERRUM_")
    )
    environment["HF_HOME"] = str(cache_root)
    overrides: dict[str, str] = {"HF_HOME": str(cache_root)}
    if spec.backend == "cuda":
        value = "/usr/local/cuda/lib64"
        environment["LD_LIBRARY_PATH"] = value
        overrides["LD_LIBRARY_PATH"] = value
    inherited_network_keys = sorted(
        key for key in network_keys
        if key in environment
    )
    return environment, {
        "overrides": overrides,
        "effective_override_keys": sorted(overrides),
        "effective_environment_keys": sorted(environment),
        "inherited_allowlist": sorted(inherited_allowlist),
        "removed_environment_keys": removed,
        "removed_behavior_keys": removed_behavior,
        "allowed_inherited_network_keys": inherited_network_keys,
        "model_source_base_url": "https://huggingface.co",
        "hf_endpoint_removed": True,
        "credentials_removed": True,
        "network_routing": network_environment_document(
            environment, consumer="ferrum-child-processes"
        ),
    }


def terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=grace_seconds)


def require_process_alive_before_cleanup(child: "ManagedChild") -> None:
    require(
        child.process is not None and child.process.poll() is None,
        f"{child.label} process exited before active gate cleanup",
    )


def record_process_alive_before_cleanup(child: "ManagedChild") -> None:
    require_process_alive_before_cleanup(child)
    metadata = read_json_file(child.command_path, f"{child.label} pre-cleanup command metadata")
    require(isinstance(metadata, dict) and metadata.get("status") == "running", f"{child.label} was not running before cleanup")
    metadata["cleanup_precondition"] = {"process_alive": True, "observed_at": iso_now()}
    write_json_atomic(child.command_path, metadata)


class ManagedChild:
    def __init__(
        self,
        *,
        label: str,
        command: list[str],
        cwd: Path,
        environment: dict[str, str],
        environment_receipt: dict[str, Any],
        out: Path,
        timeout: float,
        progress_interval: float,
        stdin_data: bytes | None = None,
        progress_observer: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.label = label
        self.command = command
        self.cwd = cwd
        self.environment = environment
        self.environment_receipt = environment_receipt
        self.timeout = timeout
        self.progress_interval = progress_interval
        self.stdin_data = stdin_data
        self.progress_observer = progress_observer
        self.out = out
        self.root = out / "processes" / label
        self.root.mkdir(parents=True, exist_ok=False)
        self.stdout_path = self.root / "stdout.log"
        self.stderr_path = self.root / "stderr.log"
        self.progress_path = self.root / "progress.jsonl"
        self.command_path = self.root / "command.json"
        self.stdin_path = self.root / "stdin.txt"
        self.stdout_handle: BinaryIO | None = None
        self.stderr_handle: BinaryIO | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.started = 0.0
        self.deadline = 0.0
        self.started_at = ""
        self.last_progress = 0.0
        self.finished = False

    def start(self) -> None:
        require(self.process is None, f"child {self.label} was already started")
        self.started = time.monotonic()
        self.deadline = self.started + self.timeout
        self.started_at = iso_now()
        if self.stdin_data is not None:
            self.stdin_path.write_bytes(self.stdin_data)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "kind": "bounded_child_process",
            "label": self.label,
            "command": self.command,
            "cwd": str(self.cwd),
            "environment": self.environment_receipt,
            "stdin": file_ref(self.stdin_path, relative_to=self.root)
            if self.stdin_data is not None
            else None,
            "stdout_log": str(self.stdout_path.relative_to(self.root)),
            "stderr_log": str(self.stderr_path.relative_to(self.root)),
            "progress_log": str(self.progress_path.relative_to(self.root)),
            "progress_signal": "process state, log byte counts, and optional model-cache growth",
            "started_at": self.started_at,
            "timeout_seconds": self.timeout,
            "status": "running",
        }
        write_json_atomic(self.command_path, metadata)
        self.stdout_handle = self.stdout_path.open("xb")
        self.stderr_handle = self.stderr_path.open("xb")
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=self.environment,
                stdin=subprocess.PIPE if self.stdin_data is not None else subprocess.DEVNULL,
                stdout=self.stdout_handle,
                stderr=self.stderr_handle,
                start_new_session=True,
            )
            if self.stdin_data is not None:
                assert self.process.stdin is not None
                self.process.stdin.write(self.stdin_data)
                self.process.stdin.flush()
                self.process.stdin.close()
            self.sample(force=True)
        except BaseException as error:
            if self.process is not None and self.process.poll() is None:
                terminate_process_group(self.process)
            if self.stdout_handle is not None:
                self.stdout_handle.close()
            if self.stderr_handle is not None:
                self.stderr_handle.close()
            self.finish_metadata("fail", error=f"child process start failed: {error}")
            raise

    def sample(self, *, force: bool = False, extra: dict[str, Any] | None = None) -> None:
        require(self.process is not None, f"child {self.label} is not started")
        now = time.monotonic()
        if not force and now < self.last_progress + self.progress_interval:
            return
        sample: dict[str, Any] = {
            "observed_at": iso_now(),
            "elapsed_seconds": now - self.started,
            "pid": self.process.pid,
            "returncode": self.process.poll(),
            "stdout_bytes": self.stdout_path.stat().st_size if self.stdout_path.exists() else 0,
            "stderr_bytes": self.stderr_path.stat().st_size if self.stderr_path.exists() else 0,
        }
        if self.progress_observer is not None:
            sample["observable_progress"] = self.progress_observer()
        if extra:
            sample.update(extra)
        append_jsonl(self.progress_path, sample)
        self.last_progress = now

    def ensure_before_deadline(self) -> None:
        require(time.monotonic() <= self.deadline, f"child {self.label} exceeded its hard deadline")

    def wait(self) -> int:
        require(self.process is not None, f"child {self.label} is not started")
        try:
            while self.process.poll() is None:
                if time.monotonic() > self.deadline:
                    terminate_process_group(self.process)
                    self.finish_metadata("timeout", error=f"hard timeout after {self.timeout} seconds")
                    raise GateError(f"child {self.label} timed out after {self.timeout} seconds")
                self.sample()
                time.sleep(min(1.0, max(0.05, self.deadline - time.monotonic())))
            self.sample(force=True)
            returncode = int(self.process.returncode)
            self.finish_metadata("pass" if returncode == 0 else "fail")
            return returncode
        except BaseException as error:
            if self.process.poll() is None:
                terminate_process_group(self.process)
            if not self.finished:
                status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "fail"
                self.finish_metadata(status, error=f"child wait aborted: {error}")
            raise

    def terminate(self) -> int:
        require(self.process is not None, f"child {self.label} is not started")
        terminate_process_group(self.process)
        self.sample(force=True)
        returncode = int(self.process.returncode)
        self.finish_metadata("terminated")
        return returncode

    def finish_metadata(self, status: str, error: str | None = None) -> None:
        if self.finished:
            return
        self.finished = True
        if self.stdout_handle is not None and not self.stdout_handle.closed:
            self.stdout_handle.flush()
            self.stdout_handle.close()
        if self.stderr_handle is not None and not self.stderr_handle.closed:
            self.stderr_handle.flush()
            self.stderr_handle.close()
        metadata = read_json_file(self.command_path, f"{self.label} command metadata")
        assert isinstance(metadata, dict)
        metadata.update(
            {
                "status": status,
                "returncode": (
                    None
                    if status == "terminated"
                    else self.process.returncode if self.process is not None else None
                ),
                "cleanup_returncode": (
                    self.process.returncode
                    if status == "terminated" and self.process is not None
                    else None
                ),
                "finished_at": iso_now(),
                "duration_seconds": time.monotonic() - self.started,
                "stdout": file_ref(self.stdout_path, relative_to=self.out),
                "stderr": file_ref(self.stderr_path, relative_to=self.out),
                "error": error,
            }
        )
        write_json_atomic(self.command_path, metadata)

    def checked_logs(self) -> tuple[str, str]:
        if not self.finished:
            raise GateError(f"child {self.label} logs requested before finish")
        return (
            read_and_scan_utf8(self.stdout_path, f"{self.label} stdout"),
            read_and_scan_utf8(self.stderr_path, f"{self.label} stderr"),
        )

    def receipt(self, out: Path) -> dict[str, Any]:
        return {
            "command": file_ref(self.command_path, relative_to=out),
            "stdout": file_ref(self.stdout_path, relative_to=out),
            "stderr": file_ref(self.stderr_path, relative_to=out),
            "progress": file_ref(self.progress_path, relative_to=out),
            "stdin": file_ref(self.stdin_path, relative_to=out)
            if self.stdin_path.exists()
            else None,
        }


def run_checked_child(
    *,
    label: str,
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    environment_receipt: dict[str, Any],
    out: Path,
    timeout: float,
    progress_interval: float,
    stdin_data: bytes | None = None,
    progress_observer: Callable[[], dict[str, Any]] | None = None,
) -> tuple[ManagedChild, str, str]:
    child = ManagedChild(
        label=label,
        command=command,
        cwd=cwd,
        environment=environment,
        environment_receipt=environment_receipt,
        out=out,
        timeout=timeout,
        progress_interval=progress_interval,
        stdin_data=stdin_data,
        progress_observer=progress_observer,
    )
    child.start()
    returncode = child.wait()
    stdout, stderr = child.checked_logs()
    require(returncode == 0, f"child {label} failed with return code {returncode}")
    return child, stdout, stderr


def check_platform(spec: BackendSpec) -> dict[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if spec.backend == "metal":
        require(system == "darwin", "Metal prerelease gate requires macOS")
        require(machine in {"arm64", "aarch64"}, "Metal prerelease gate requires Apple Silicon")
    else:
        require(system == "linux", "CUDA prerelease gate requires Linux")
        require(machine in {"x86_64", "amd64"}, "CUDA prerelease gate requires x86_64")
    return {"system": system, "machine": machine}


def assert_port_available(port: int) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise GateError(f"port {port} is unavailable: {error}") from error
    finally:
        probe.close()


def localhost_request(
    *,
    method: str,
    url: str,
    timeout: float,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    max_bytes: int = MAX_HTTP_RESPONSE_BYTES,
) -> tuple[int, dict[str, str], bytes]:
    parsed = urllib.parse.urlparse(url)
    require(parsed.scheme == "http", f"local request scheme must be HTTP: {url}")
    require(parsed.hostname in {"127.0.0.1", "localhost"}, f"request is not local: {url}")
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    deadline = time.monotonic() + timeout
    try:
        with urllib.request.urlopen(request, timeout=min(timeout, 15.0)) as response:
            data = read_bounded_http_body(
                response,
                deadline=deadline,
                max_bytes=max_bytes,
                label=url,
            )
            return response.status, dict(response.headers.items()), data
    except urllib.error.HTTPError as error:
        data = read_bounded_http_body(
            error,
            deadline=deadline,
            max_bytes=max_bytes,
            label=url,
        )
        return error.code, dict(error.headers.items()), data


def save_http_exchange(
    *,
    label: str,
    method: str,
    url: str,
    out: Path,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, str], bytes, dict[str, Any]]:
    root = out / "http" / label
    root.mkdir(parents=True, exist_ok=False)
    request_body = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if payload is not None
        else None
    )
    if request_body is not None:
        (root / "request.json").write_bytes(request_body + b"\n")
    started = time.monotonic()
    started_at = iso_now()
    status, response_headers, response_body = localhost_request(
        method=method,
        url=url,
        timeout=timeout,
        body=request_body,
        headers={"Content-Type": "application/json"} if request_body is not None else None,
    )
    response_path = root / "response.body"
    response_path.write_bytes(response_body)
    text = read_and_scan_utf8(response_path, f"{label} HTTP response")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "bounded_local_http_exchange",
        "label": label,
        "method": method,
        "url": url,
        "timeout_seconds": timeout,
        "started_at": started_at,
        "finished_at": iso_now(),
        "duration_seconds": time.monotonic() - started,
        "status": status,
        "request": file_ref(root / "request.json", relative_to=out)
        if request_body is not None
        else None,
        "response": file_ref(response_path, relative_to=out),
        "response_content_type": next(
            (value for key, value in response_headers.items() if key.lower() == "content-type"),
            None,
        ),
    }
    write_json_atomic(root / "exchange.json", metadata)
    metadata["exchange"] = file_ref(root / "exchange.json", relative_to=out)
    metadata["response_text"] = text
    return status, response_headers, response_body, metadata


def wait_for_server(
    server: ManagedChild,
    *,
    port: int,
    startup_timeout: float,
    out: Path,
) -> dict[str, Any]:
    progress_path = out / "http" / "readiness.progress.jsonl"
    started = time.monotonic()
    deadline = min(server.deadline, started + startup_timeout)
    attempts = 0
    while time.monotonic() <= deadline:
        assert server.process is not None
        if server.process.poll() is not None:
            server.sample(force=True, extra={"readiness": "server-exited"})
            raise GateError(f"serve process exited before readiness: {server.process.returncode}")
        attempts += 1
        attempt: dict[str, Any] = {
            "attempt": attempts,
            "observed_at": iso_now(),
            "elapsed_seconds": time.monotonic() - started,
        }
        try:
            status, _headers, body = localhost_request(
                method="GET",
                url=f"http://127.0.0.1:{port}/health",
                timeout=min(5.0, max(0.1, deadline - time.monotonic())),
                max_bytes=1024 * 1024,
            )
            text = strict_utf8(body, "health response")
            scan_forbidden_text("health response", text)
            attempt.update({"status": status, "response_bytes": len(body)})
            append_jsonl(progress_path, attempt)
            server.sample(extra={"readiness_attempt": attempts, "health_status": status})
            if status == 200:
                return {
                    "attempts": attempts,
                    "duration_seconds": time.monotonic() - started,
                    "progress": file_ref(progress_path, relative_to=out),
                }
        except Exception as error:
            attempt["error"] = str(error)
            append_jsonl(progress_path, attempt)
            server.sample(extra={"readiness_attempt": attempts, "health_error": str(error)})
        time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))
    raise GateError(f"server was not ready within {startup_timeout} seconds")


def parse_json_response(data: bytes, label: str) -> dict[str, Any]:
    parsed = parse_json_bytes(data, label)
    require(isinstance(parsed, dict), f"{label} root must be an object")
    return parsed


def validate_models_response(status: int, data: bytes) -> dict[str, Any]:
    require(status == 200, f"/v1/models returned HTTP {status}")
    document = parse_json_response(data, "/v1/models response")
    require(document.get("object") == "list", "/v1/models object is not list")
    models = document.get("data")
    require(isinstance(models, list), "/v1/models data is not an array")
    matches = [row for row in models if isinstance(row, dict) and row.get("id") == "ferrum"]
    require(len(matches) == 1, f"/v1/models must expose exactly one ferrum entry, got {len(matches)}")
    return matches[0]


def validate_nonstream_response(status: int, data: bytes) -> dict[str, Any]:
    require(status == 200, f"nonstream Chat Completions returned HTTP {status}")
    document = parse_json_response(data, "nonstream Chat Completions response")
    choices = document.get("choices")
    require(isinstance(choices, list) and choices, "nonstream response choices are empty")
    first = choices[0]
    require(isinstance(first, dict), "nonstream first choice is not an object")
    message = first.get("message")
    require(isinstance(message, dict), "nonstream response message is not an object")
    content = message.get("content")
    require(isinstance(content, str) and content.strip(), "nonstream assistant content is empty")
    return {
        "content": content,
        "finish_reason": first.get("finish_reason"),
        "usage": document.get("usage"),
    }


def parse_sse_stream(data: bytes, label: str = "stream Chat Completions response") -> dict[str, Any]:
    text = strict_utf8(data, label)
    scan_forbidden_text(label, text)
    events: list[str] = []
    for line in text.splitlines():
        if not line or line.startswith(":"):
            continue
        require(line.startswith("data:"), f"{label} contains non-data SSE line: {line!r}")
        events.append(line[len("data:") :].strip())
    require(events, f"{label} contains no SSE events")
    done_count = events.count("[DONE]")
    require(done_count == 1, f"{label} [DONE] count is {done_count}, expected 1")
    require(events[-1] == "[DONE]", f"{label} has data after [DONE]")
    usage_objects: list[dict[str, Any]] = []
    content_parts: list[str] = []
    json_event_count = 0
    for ordinal, event in enumerate(events[:-1]):
        require(event != "[DONE]", f"{label} has premature [DONE] at event {ordinal}")
        try:
            document = json.loads(event, object_pairs_hook=reject_duplicate_json_pairs)
        except GateError:
            raise
        except json.JSONDecodeError as error:
            raise GateError(f"{label} event {ordinal} is invalid JSON: {error}") from error
        require(isinstance(document, dict), f"{label} event {ordinal} is not an object")
        json_event_count += 1
        usage = document.get("usage")
        if usage is not None:
            require(isinstance(usage, dict), f"{label} event {ordinal} usage is not an object")
            usage_objects.append(usage)
        choices = document.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    content_parts.append(delta["content"])
    require(json_event_count > 0, f"{label} contains no JSON events")
    require(
        len(usage_objects) == 1,
        f"{label} usage object count is {len(usage_objects)}, expected 1",
    )
    usage = usage_objects[0]
    completion_tokens = usage.get("completion_tokens")
    require(
        isinstance(completion_tokens, int)
        and not isinstance(completion_tokens, bool)
        and completion_tokens > 0,
        f"{label} completion_tokens must be a positive integer",
    )
    content = "".join(content_parts)
    require(content.strip(), f"{label} assistant content is empty")
    return {
        "event_count": len(events),
        "json_event_count": json_event_count,
        "done_count": 1,
        "usage_object_count": 1,
        "usage": usage,
        "content": content,
    }


def execute_binary_identity_checks(
    binary: Path,
    *,
    runtime_dir: Path,
    environment: dict[str, str],
    environment_receipt: dict[str, Any],
    out: Path,
    timeout: float,
    progress_interval: float,
) -> dict[str, Any]:
    version_child, version_stdout, version_stderr = run_checked_child(
        label="binary-version",
        command=[str(binary), "--version"],
        cwd=runtime_dir,
        environment=environment,
        environment_receipt=environment_receipt,
        out=out,
        timeout=timeout,
        progress_interval=progress_interval,
    )
    version_text = (version_stdout + "\n" + version_stderr).strip()
    require(
        any(line.strip() == f"ferrum {VERSION}" for line in version_text.splitlines()),
        f"extracted ferrum does not report exact version {VERSION}: {version_text!r}",
    )
    help_child, help_stdout, help_stderr = run_checked_child(
        label="binary-help",
        command=[str(binary), "--help"],
        cwd=runtime_dir,
        environment=environment,
        environment_receipt=environment_receipt,
        out=out,
        timeout=timeout,
        progress_interval=progress_interval,
    )
    help_text = help_stdout + "\n" + help_stderr
    require(help_text.strip(), "extracted ferrum --help output is empty")
    for command in ("doctor", "run", "serve"):
        require(re.search(rf"\b{command}\b", help_text) is not None, f"--help omits {command}")
    return {
        "version": version_child.receipt(out),
        "help": help_child.receipt(out),
        "reported_version": VERSION,
    }


def execute_doctor_checks(
    binary: Path,
    spec: BackendSpec,
    *,
    runtime_dir: Path,
    cache_root: Path,
    environment: dict[str, str],
    environment_receipt: dict[str, Any],
    out: Path,
    timeout: float,
    progress_interval: float,
) -> dict[str, Any]:
    require(not cache_root.exists(), f"model cache must be absent before doctor: {cache_root}")
    before = cache_snapshot(cache_root)
    plain_child, plain_stdout, plain_stderr = run_checked_child(
        label="doctor",
        command=[str(binary), "doctor"],
        cwd=runtime_dir,
        environment=environment,
        environment_receipt=environment_receipt,
        out=out,
        timeout=timeout,
        progress_interval=progress_interval,
    )
    after_plain = cache_snapshot(cache_root)
    assert_cache_unchanged(before, after_plain, "ferrum doctor")
    require(f"Ferrum {VERSION}" in plain_stdout + plain_stderr, "doctor omits exact version")

    model_child, model_stdout, model_stderr = run_checked_child(
        label="doctor-model",
        command=[str(binary), "doctor", spec.model_alias],
        cwd=runtime_dir,
        environment=environment,
        environment_receipt=environment_receipt,
        out=out,
        timeout=timeout,
        progress_interval=progress_interval,
    )
    after_model = cache_snapshot(cache_root)
    assert_cache_unchanged(before, after_model, "ferrum doctor MODEL")
    doctor_text = model_stdout + "\n" + model_stderr
    require(spec.model_alias in doctor_text, "doctor MODEL output omits requested alias")
    require("No model was downloaded" in doctor_text, "doctor MODEL does not state no download")
    return {
        "cache_before": before,
        "cache_after": after_model,
        "cache_unchanged": True,
        "doctor": plain_child.receipt(out),
        "doctor_model": model_child.receipt(out),
    }


def execute_run_check(
    binary: Path,
    spec: BackendSpec,
    *,
    runtime_dir: Path,
    cache_root: Path,
    environment: dict[str, str],
    environment_receipt: dict[str, Any],
    out: Path,
    timeout: float,
    progress_interval: float,
) -> dict[str, Any]:
    marker = "FERRUM-084-OK"
    prompt = f"Reply with exactly the ASCII marker {marker} and nothing else.\n/bye\n"
    child, stdout, stderr = run_checked_child(
        label="readme-run",
        command=[str(binary), "run", spec.model_alias, "--disable-thinking"],
        cwd=runtime_dir,
        environment=environment,
        environment_receipt=environment_receipt,
        out=out,
        timeout=timeout,
        progress_interval=progress_interval,
        stdin_data=prompt.encode("utf-8"),
        progress_observer=lambda: cache_progress(cache_root),
    )
    require(marker in stdout, "README run response did not contain the objective marker")
    return {
        "command": child.command,
        "disable_thinking": True,
        "objective_marker": marker,
        "objective_response_nonempty": True,
        "process": child.receipt(out),
        "stderr_nonempty": bool(stderr.strip()),
    }


def execute_serve_checks(
    binary: Path,
    spec: BackendSpec,
    *,
    runtime_dir: Path,
    cache_root: Path,
    environment: dict[str, str],
    environment_receipt: dict[str, Any],
    out: Path,
    port: int,
    startup_timeout: float,
    total_timeout: float,
    request_timeout: float,
    progress_interval: float,
    documented_serve_argv: list[str],
) -> dict[str, Any]:
    assert_port_available(port)
    server = ManagedChild(
        label="readme-serve",
        command=[
            str(binary),
            "serve",
            "--model",
            spec.model_alias,
            "--served-model-name",
            "ferrum",
            "--disable-thinking",
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ],
        cwd=runtime_dir,
        environment=environment,
        environment_receipt=environment_receipt,
        out=out,
        timeout=total_timeout,
        progress_interval=progress_interval,
        progress_observer=lambda: cache_progress(cache_root),
    )
    server.start()
    primary_error: Exception | None = None
    result: dict[str, Any] = {}
    try:
        readiness = wait_for_server(
            server,
            port=port,
            startup_timeout=startup_timeout,
            out=out,
        )
        server.ensure_before_deadline()
        status, _headers, body, models_exchange = save_http_exchange(
            label="models",
            method="GET",
            url=f"http://127.0.0.1:{port}/v1/models",
            out=out,
            timeout=min(request_timeout, max(0.1, server.deadline - time.monotonic())),
        )
        model_entry = validate_models_response(status, body)

        nonstream_payload = {
            "model": "ferrum",
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with a short hello from Ferrum.",
                }
            ],
            "max_tokens": 32,
        }
        server.ensure_before_deadline()
        status, _headers, body, nonstream_exchange = save_http_exchange(
            label="chat-nonstream",
            method="POST",
            url=f"http://127.0.0.1:{port}/v1/chat/completions",
            out=out,
            timeout=min(request_timeout, max(0.1, server.deadline - time.monotonic())),
            payload=nonstream_payload,
        )
        nonstream = validate_nonstream_response(status, body)

        stream_payload = {
            "model": "ferrum",
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with a short hello from Ferrum.",
                }
            ],
            "max_tokens": 32,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        server.ensure_before_deadline()
        status, headers, body, stream_exchange = save_http_exchange(
            label="chat-stream",
            method="POST",
            url=f"http://127.0.0.1:{port}/v1/chat/completions",
            out=out,
            timeout=min(request_timeout, max(0.1, server.deadline - time.monotonic())),
            payload=stream_payload,
        )
        require(status == 200, f"stream Chat Completions returned HTTP {status}")
        content_type = next(
            (value for key, value in headers.items() if key.lower() == "content-type"),
            "",
        )
        require(
            content_type.lower().startswith("text/event-stream"),
            f"stream Content-Type is not text/event-stream: {content_type!r}",
        )
        stream = parse_sse_stream(body)
        normalized = ["ferrum", *server.command[1:]]
        host_index = normalized.index("--host")
        require(normalized[host_index + 1] == "127.0.0.1", "gate serve host override differs")
        del normalized[host_index : host_index + 2]
        port_index = normalized.index("--port")
        normalized[port_index + 1] = "8000"
        require(normalized == documented_serve_argv, "actual serve argv is not README-equivalent")
        result = {
            "command": server.command,
            "readiness": readiness,
            "models": {
                "served_model": model_entry,
                "exchange": {key: value for key, value in models_exchange.items() if key != "response_text"},
            },
            "nonstream": {
                "validation": nonstream,
                "exchange": {
                    key: value for key, value in nonstream_exchange.items() if key != "response_text"
                },
            },
            "stream": {
                "validation": stream,
                "exchange": {key: value for key, value in stream_exchange.items() if key != "response_text"},
            },
            "readme_equivalence": {
                "equivalent": True,
                "documented_argv": documented_serve_argv,
                "actual_argv": server.command,
                "normalized_argv": normalized,
                "allowed_gate_overrides": {
                    "host": "127.0.0.1",
                    "port": {"documented": 8000, "actual": port},
                },
            },
        }
    except Exception as error:
        primary_error = error
    finally:
        try:
            record_process_alive_before_cleanup(server)
            server.terminate()
            server.checked_logs()
            result["process"] = server.receipt(out)
        except Exception as cleanup_error:
            if primary_error is None:
                primary_error = cleanup_error
            else:
                primary_error = GateError(
                    f"{primary_error}; serve cleanup/log validation also failed: {cleanup_error}"
                )
    if primary_error is not None:
        raise primary_error
    return result


def backend_summary_name(backend: str) -> str:
    return f"{SUMMARY_PREFIX}-{backend}.json"


def aggregate_summary_name() -> str:
    return f"{SUMMARY_PREFIX}-aggregate.json"


def run_backend(args: argparse.Namespace) -> int:
    spec = BACKEND_SPECS[args.mode]
    out = prepare_fresh_output(args.out)
    summary_path = out / backend_summary_name(spec.backend)
    started_at = iso_now()
    started = time.monotonic()
    total_deadline_seconds = backend_total_deadline_seconds(args, spec)
    partial: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "artifact_type": "ferrum_v084_prerelease_download_backend",
        "version": VERSION,
        "tag": TAG,
        "backend": spec.backend,
        "status": "running",
        "artifact_dir": str(out),
        "started_at": started_at,
        "execution_budget": {
            "deadline_seconds": total_deadline_seconds,
            "progress_signal": "download bytes, process log bytes, and model-cache growth",
        },
        "command_line": [sys.executable, *sys.argv],
    }
    write_json_atomic(summary_path, partial)
    try:
        urllib_network_ref = write_network_environment_receipt(
            out,
            label="urllib-public-downloads",
            environment=dict(os.environ),
            consumer="urllib-public-github-downloads",
        )
        host = check_platform(spec)
        raw_release, api_fetch = fetch_public_release_snapshot(out, args.api_timeout_seconds)
        release, api_assets = validate_release_snapshot(raw_release, spec)
        raw_release_path = out / "github-release-api.json"
        release["raw_api_snapshot"] = file_ref(raw_release_path, relative_to=out)

        download_dir = out / "public-assets"
        download_dir.mkdir(parents=True, exist_ok=False)
        downloads: dict[str, Any] = {}
        for name in backend_download_asset_names(spec):
            downloads[name] = download_public_asset(
                api_assets[name],
                download_dir / name,
                timeout=args.asset_download_timeout_seconds,
                progress_interval=args.progress_interval_seconds,
                out=out,
            )

        asset_path = download_dir / spec.asset_name
        asset_sha = sha256_file(asset_path)
        require(asset_sha == api_assets[spec.asset_name]["sha256"], "tarball GitHub digest differs")
        staged_match: bool | None = None
        if args.expected_asset_sha256 is not None:
            staged_match = asset_sha == args.expected_asset_sha256
            require(staged_match, "public tarball SHA256 differs from staged expected SHA256")

        extraction_dir = out / "extracted"
        extracted = safe_extract_tarball(asset_path, extraction_dir)
        binary = extraction_dir / "ferrum"
        binary_sha = sha256_file(binary)
        readme_contract = validate_packaged_readme_contract(extraction_dir, spec)
        adjacent = validate_adjacent_bundle(
            download_dir,
            spec,
            asset_sha256=asset_sha,
            binary_sha256=binary_sha,
            extracted_binary=binary,
        )

        cache_root = out / "model-cache"
        environment, environment_receipt = sanitized_child_environment(cache_root, spec)
        child_network_ref = write_network_environment_receipt(
            out,
            label="ferrum-child-processes",
            environment=environment,
            consumer="ferrum-child-processes",
        )
        network_environment = {
            "urllib_public_downloads": urllib_network_ref,
            "child_processes": child_network_ref,
        }
        identity_checks = execute_binary_identity_checks(
            binary,
            runtime_dir=extraction_dir,
            environment=environment,
            environment_receipt=environment_receipt,
            out=out,
            timeout=args.command_timeout_seconds,
            progress_interval=args.progress_interval_seconds,
            documented_serve_argv=readme_contract["serve_argv"],
        )
        doctor = execute_doctor_checks(
            binary,
            spec,
            runtime_dir=extraction_dir,
            cache_root=cache_root,
            environment=environment,
            environment_receipt=environment_receipt,
            out=out,
            timeout=args.command_timeout_seconds,
            progress_interval=args.progress_interval_seconds,
        )
        run_result = execute_run_check(
            binary,
            spec,
            runtime_dir=extraction_dir,
            cache_root=cache_root,
            environment=environment,
            environment_receipt=environment_receipt,
            out=out,
            timeout=args.model_command_timeout_seconds,
            progress_interval=args.progress_interval_seconds,
        )
        model_cache = model_cache_receipt(cache_root, spec)
        serve = execute_serve_checks(
            binary,
            spec,
            runtime_dir=extraction_dir,
            cache_root=cache_root,
            environment=environment,
            environment_receipt=environment_receipt,
            out=out,
            port=args.port,
            startup_timeout=args.serve_startup_timeout_seconds,
            total_timeout=args.server_total_timeout_seconds,
            request_timeout=args.request_timeout_seconds,
            progress_interval=args.progress_interval_seconds,
        )

        e2e_finished_at = iso_now()
        e2e_summary_path, e2e_summary = emit_goal_e2e_summary(
            out=out,
            spec=spec,
            started_at=started_at,
            finished_at=e2e_finished_at,
            elapsed_seconds=time.monotonic() - started,
            deadline_seconds=total_deadline_seconds,
            candidate_sha=adjacent["release_candidate"]["release_candidate_sha"],
            asset_sha256=asset_sha,
            binary_sha256=binary_sha,
            readme_contract=readme_contract,
            environment_receipt=environment_receipt,
            model_cache=model_cache,
            identity_checks=identity_checks,
            doctor=doctor,
            run_result=run_result,
            serve=serve,
            download_receipt=downloads[spec.asset_name],
            network_environment=network_environment,
        )

        pass_line = f"FERRUM {VERSION} PRERELEASE DOWNLOAD {spec.pass_label} PASS: {out}"
        summary = {
            **partial,
            "status": "pass",
            "pass_line": pass_line,
            "finished_at": iso_now(),
            "duration_seconds": time.monotonic() - started,
            "host": host,
            "timeouts": {
                "api_seconds": args.api_timeout_seconds,
                "asset_download_seconds_each": args.asset_download_timeout_seconds,
                "short_command_seconds": args.command_timeout_seconds,
                "model_command_seconds": args.model_command_timeout_seconds,
                "serve_startup_seconds": args.serve_startup_timeout_seconds,
                "server_total_seconds": args.server_total_timeout_seconds,
                "request_seconds_each": args.request_timeout_seconds,
                "progress_interval_seconds": args.progress_interval_seconds,
            },
            "release": release,
            "release_api_fetch": api_fetch,
            "api_assets": api_assets,
            "downloads": downloads,
            "asset": {
                **api_assets[spec.asset_name],
                "public_sha256": asset_sha,
                "expected_staged_sha256": args.expected_asset_sha256,
                "staged_public_sha256_equal": staged_match,
                "file": file_ref(asset_path, relative_to=out),
            },
            "adjacent_bundle": adjacent,
            "extraction": {
                "root": str(extraction_dir),
                "members": extracted,
                "members_metadata_sha256": canonical_json_sha256(extracted),
            },
            "binary": {
                "path": str(binary),
                "sha256": binary_sha,
                "size_bytes": binary.stat().st_size,
                "identity_checks": identity_checks,
            },
            "readme_contract": readme_contract,
            "environment": environment_receipt,
            "network_environment": network_environment,
            "model": {
                "alias": spec.model_alias,
                "cache": model_cache,
                "identity": e2e_summary["model"],
            },
            "goal_e2e_summary": file_ref(e2e_summary_path, relative_to=out),
            "checks": {
                "doctor": doctor,
                "run": run_result,
                "serve": serve,
            },
        }
        write_json_atomic(summary_path, summary)
        print(pass_line)
        return 0
    except Exception as error:
        failed = {
            **partial,
            "status": "fail",
            "pass_line": None,
            "finished_at": iso_now(),
            "duration_seconds": time.monotonic() - started,
            "error": str(error),
        }
        write_json_atomic(summary_path, failed)
        print(f"FERRUM {VERSION} PRERELEASE DOWNLOAD {spec.pass_label} FAIL: {error}", file=sys.stderr)
        return 1


def load_backend_summary(path: Path, backend: str) -> dict[str, Any]:
    document = read_json_file(path, f"{backend} backend summary")
    require(isinstance(document, dict), f"{backend} summary root is not an object")
    require(document.get("schema_version") == SCHEMA_VERSION, f"{backend} summary schema differs")
    require(document.get("validator_version") == VALIDATOR_VERSION, f"{backend} validator differs")
    require(
        document.get("artifact_type") == "ferrum_v084_prerelease_download_backend",
        f"{backend} summary artifact type differs",
    )
    require(document.get("version") == VERSION and document.get("tag") == TAG, f"{backend} version/tag differs")
    require(document.get("backend") == backend, f"{backend} summary backend differs")
    require(document.get("status") == "pass", f"{backend} backend gate is not pass")
    artifact_dir = document.get("artifact_dir")
    require(isinstance(artifact_dir, str) and artifact_dir, f"{backend} artifact_dir is missing")
    expected_pass = (
        f"FERRUM {VERSION} PRERELEASE DOWNLOAD {BACKEND_SPECS[backend].pass_label} PASS: "
        f"{artifact_dir}"
    )
    require(document.get("pass_line") == expected_pass, f"{backend} PASS line binding differs")
    release = document.get("release")
    require(isinstance(release, dict), f"{backend} release receipt is missing")
    require(release.get("tag_name") == TAG, f"{backend} release tag differs")
    require(release.get("draft") is False and release.get("prerelease") is True, f"{backend} release state differs")
    release_snapshot_sha = release.get("immutable_snapshot_sha256")
    require(
        isinstance(release_snapshot_sha, str) and SHA256_RE.fullmatch(release_snapshot_sha) is not None,
        f"{backend} release snapshot SHA256 is invalid",
    )
    require(
        canonical_json_sha256(release.get("immutable_snapshot")) == release_snapshot_sha,
        f"{backend} release immutable snapshot digest differs",
    )
    asset = document.get("asset")
    require(isinstance(asset, dict), f"{backend} asset receipt is missing")
    expected_asset_name = BACKEND_SPECS[backend].asset_name
    require(asset.get("name") == expected_asset_name, f"{backend} asset name differs")
    public_sha = asset.get("public_sha256")
    expected_sha = asset.get("expected_staged_sha256")
    require(
        isinstance(public_sha, str) and SHA256_RE.fullmatch(public_sha) is not None,
        f"{backend} public asset SHA256 is invalid",
    )
    require(
        isinstance(expected_sha, str) and SHA256_RE.fullmatch(expected_sha) is not None,
        f"{backend} aggregate requires --expected-asset-sha256 evidence",
    )
    require(asset.get("staged_public_sha256_equal") is True, f"{backend} staged/public equality is not true")
    require(public_sha == expected_sha == asset.get("sha256"), f"{backend} staged/public/API SHA256 differs")
    adjacent = document.get("adjacent_bundle")
    require(isinstance(adjacent, dict), f"{backend} adjacent bundle is missing")
    candidate = adjacent.get("release_candidate")
    require(isinstance(candidate, dict), f"{backend} release candidate identity is missing")
    require(
        isinstance(candidate.get("release_candidate_sha"), str)
        and GIT_SHA_RE.fullmatch(candidate["release_candidate_sha"]) is not None,
        f"{backend} candidate SHA is invalid",
    )
    binary = document.get("binary")
    require(isinstance(binary, dict), f"{backend} binary identity is missing")
    require(
        isinstance(binary.get("sha256"), str) and SHA256_RE.fullmatch(binary["sha256"]) is not None,
        f"{backend} binary SHA256 is invalid",
    )
    checks = document.get("checks")
    require(isinstance(checks, dict) and set(checks) == {"doctor", "run", "serve"}, f"{backend} README checks are incomplete")
    return document


def aggregate_documents(metal: dict[str, Any], cuda: dict[str, Any]) -> dict[str, Any]:
    metal_release = metal["release"]
    cuda_release = cuda["release"]
    require(metal_release["id"] == cuda_release["id"], "backend release ids differ")
    require(
        metal_release["immutable_snapshot_sha256"]
        == cuda_release["immutable_snapshot_sha256"],
        "backend GitHub release snapshots differ",
    )
    metal_candidate = metal["adjacent_bundle"]["release_candidate"]
    cuda_candidate = cuda["adjacent_bundle"]["release_candidate"]
    candidate_fields = (
        "release_candidate_sha",
        "release_candidate_tag",
        "staging_label",
    )
    metal_candidate_identity = {
        field: metal_candidate[field] for field in candidate_fields
    }
    cuda_candidate_identity = {
        field: cuda_candidate[field] for field in candidate_fields
    }
    require(
        metal_candidate_identity == cuda_candidate_identity,
        "Metal and CUDA adjacent manifests bind different release candidates",
    )
    assets = {
        "metal": {
            "name": metal["asset"]["name"],
            "id": metal["asset"]["id"],
            "size_bytes": metal["asset"]["size_bytes"],
            "sha256": metal["asset"]["public_sha256"],
            "expected_staged_sha256": metal["asset"]["expected_staged_sha256"],
        },
        "cuda": {
            "name": cuda["asset"]["name"],
            "id": cuda["asset"]["id"],
            "size_bytes": cuda["asset"]["size_bytes"],
            "sha256": cuda["asset"]["public_sha256"],
            "expected_staged_sha256": cuda["asset"]["expected_staged_sha256"],
        },
    }
    require(assets["metal"]["id"] != assets["cuda"]["id"], "backend tarball asset ids collide")
    return {
        "release": {
            "id": metal_release["id"],
            "tag_name": TAG,
            "draft": False,
            "prerelease": True,
            "immutable_snapshot_sha256": metal_release["immutable_snapshot_sha256"],
        },
        "release_candidate": {
            **metal_candidate_identity,
            "backend_workflows": {
                "metal": {
                    "workflow_run_id": metal_candidate["workflow_run_id"],
                    "workflow_run_attempt": metal_candidate["workflow_run_attempt"],
                },
                "cuda": {
                    "workflow_run_id": cuda_candidate["workflow_run_id"],
                    "workflow_run_attempt": cuda_candidate["workflow_run_attempt"],
                },
            },
        },
        "assets": assets,
    }


def exact_asset_directory(path: Path, label: str) -> dict[str, Path]:
    root = path.expanduser().resolve()
    require(root.is_dir() and not root.is_symlink(), f"{label} is not a regular directory")
    children = list(root.iterdir())
    names = {child.name for child in children}
    require(names == GOAL_EXPECTED_ASSETS, f"{label} asset denominator differs")
    result: dict[str, Path] = {}
    for child in children:
        require(
            child.is_file() and not child.is_symlink(),
            f"{label} contains a non-regular asset: {child.name}",
        )
        require(child.stat().st_size > 0, f"{label} contains an empty asset: {child.name}")
        result[child.name] = child
    return result


def validate_goal_release_snapshot(raw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(isinstance(raw, dict), "goal release snapshot root is not an object")
    release_id = raw.get("id")
    require(
        isinstance(release_id, int) and not isinstance(release_id, bool) and release_id > 0,
        "goal release snapshot id is invalid",
    )
    require(raw.get("tag_name") == TAG, "goal release snapshot tag differs")
    require(raw.get("draft") is False, "goal release snapshot is draft")
    require(raw.get("prerelease") is True, "goal release snapshot is not a prerelease")
    created_at = raw.get("created_at")
    published_at = raw.get("published_at")
    require(isinstance(created_at, str), "goal release snapshot created_at is missing")
    require(isinstance(published_at, str), "goal release snapshot published_at is missing")
    try:
        created_timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        published_timestamp = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise GateError("goal release snapshot published_at is invalid") from error
    require(created_timestamp.tzinfo is not None, "goal release created_at lacks timezone")
    require(published_timestamp.tzinfo is not None and created_timestamp <= published_timestamp, "goal release publication timing differs")
    assets = raw.get("assets")
    require(isinstance(assets, list), "goal release snapshot assets are missing")
    require(len(assets) == len(GOAL_EXPECTED_ASSETS), "goal release snapshot asset count differs")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    ids: set[int] = set()
    for index, asset in enumerate(assets):
        require(isinstance(asset, dict), f"goal release asset {index} is not an object")
        name = asset.get("name")
        asset_id = asset.get("id")
        size = asset.get("size")
        require(isinstance(name, str) and name in GOAL_EXPECTED_ASSETS, f"goal release asset {index} name differs")
        require(name not in names, f"duplicate goal release asset: {name}")
        require(
            isinstance(asset_id, int)
            and not isinstance(asset_id, bool)
            and asset_id > 0
            and asset_id not in ids,
            f"goal release asset {name} id differs",
        )
        require(
            isinstance(size, int) and not isinstance(size, bool) and size > 0,
            f"goal release asset {name} size differs",
        )
        digest = github_digest(asset.get("digest"), name)
        url = asset.get("browser_download_url")
        require(url == expected_public_asset_url(name), f"goal release asset {name} URL differs")
        if "state" in asset:
            require(asset.get("state") == "uploaded", f"goal release asset {name} is not uploaded")
        rows.append(
            {
                "id": asset_id,
                "name": name,
                "size": size,
                "digest": f"sha256:{digest}",
            }
        )
        names.add(name)
        ids.add(asset_id)
    require(names == GOAL_EXPECTED_ASSETS, "goal release asset-name denominator differs")
    rows.sort(key=lambda row: (row["name"], row["id"]))
    projection = release_identity_projection(raw)
    return {
        "id": release_id,
        "snapshot_sha256": canonical_json_sha256(projection),
        "asset_set_sha256": canonical_json_sha256(rows),
        "published_at": published_at,
        "published_timestamp": published_timestamp.timestamp(),
        "created_at": created_at,
        "created_timestamp": created_timestamp.timestamp(),
    }, rows


def validate_goal_tag_snapshot(raw: Any, *, candidate_sha: str) -> None:
    require(isinstance(raw, dict), "annotated tag snapshot root is not an object")
    require(raw.get("tag") == TAG, "annotated tag snapshot tag differs")
    require(
        isinstance(raw.get("sha"), str) and GIT_SHA_RE.fullmatch(raw["sha"]) is not None,
        "annotated tag object SHA is invalid",
    )
    peeled = raw.get("object")
    require(isinstance(peeled, dict), "annotated tag peeled object is missing")
    require(
        peeled.get("type") == "commit" and peeled.get("sha") == candidate_sha,
        "annotated tag does not peel to the release candidate",
    )


def validate_goal_tag_ref_snapshot(raw: Any, *, annotated_tag_sha: str) -> None:
    require(isinstance(raw, dict), "tag ref snapshot root is not an object")
    require(raw.get("ref") == f"refs/tags/{TAG}", "tag ref snapshot ref differs")
    target = raw.get("object")
    require(isinstance(target, dict), "tag ref snapshot object is missing")
    require(
        target.get("type") == "tag" and target.get("sha") == annotated_tag_sha,
        "refs/tags/v0.8.4 does not point to the annotated tag object",
    )


def validate_rc_tag_chain(ref_raw: Any, tag_raw: Any, *, rc_tag: str, candidate_sha: str) -> None:
    require(isinstance(tag_raw, dict) and tag_raw.get("tag") == rc_tag, "RC annotated tag differs")
    tag_sha = tag_raw.get("sha")
    require(isinstance(tag_sha, str) and GIT_SHA_RE.fullmatch(tag_sha) is not None, "RC tag object SHA is invalid")
    peeled = tag_raw.get("object")
    require(isinstance(peeled, dict) and peeled.get("type") == "commit" and peeled.get("sha") == candidate_sha, "RC tag does not peel to candidate")
    require(isinstance(ref_raw, dict) and ref_raw.get("ref") == f"refs/tags/{rc_tag}", "RC tag ref differs")
    target = ref_raw.get("object")
    require(isinstance(target, dict) and target.get("type") == "tag" and target.get("sha") == tag_sha, "RC ref does not bind annotated tag object")


def iso_timestamp(value: Any, label: str) -> float:
    require(isinstance(value, str), f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise GateError(f"{label} is invalid") from error
    require(parsed.tzinfo is not None, f"{label} lacks timezone")
    return parsed.timestamp()


def command_option(command: Any, option: str, *, label: str) -> str:
    require(isinstance(command, list), f"{label} command line is missing")
    positions = [index for index, value in enumerate(command) if value == option]
    require(len(positions) == 1, f"{label} must contain exactly one {option}")
    index = positions[0]
    require(index + 1 < len(command), f"{label} {option} value is missing")
    value = command[index + 1]
    require(isinstance(value, str) and value, f"{label} {option} value is invalid")
    return value


def validate_prepublication_binary_gate_pair(
    *,
    backend: str,
    outer_path: Path,
    child_path: Path,
    candidate_sha: str,
    staged_asset: Path,
    release_published_timestamp: float,
) -> dict[str, Path]:
    lane = f"{backend}-tarball"
    pass_prefix = "METAL TARBALL GATE PASS: " if backend == "metal" else "CUDA TARBALL GATE PASS: "
    expected_asset_name = BACKEND_SPECS[backend].asset_name
    outer = read_json_file(outer_path, f"{backend} prepublication outer gate")
    child = read_json_file(child_path, f"{backend} prepublication child gate")
    require(isinstance(outer, dict), f"{backend} prepublication outer gate is invalid")
    require(isinstance(child, dict), f"{backend} prepublication child gate is invalid")
    require(
        outer.get("schema_version") == SCHEMA_VERSION
        and outer.get("lane") == lane
        and outer.get("status") == "pass"
        and outer.get("child_returncode") == 0,
        f"{backend} prepublication outer status/lane differs",
    )
    require(outer.get("git_sha") == candidate_sha, f"{backend} prepublication candidate differs")
    require(
        outer.get("dirty_status") == {"is_dirty": False, "status_short": []},
        f"{backend} prepublication gate used a dirty candidate",
    )
    artifact_dir = outer.get("artifact_dir")
    require(isinstance(artifact_dir, str) and artifact_dir, f"{backend} artifact_dir is missing")
    require(
        outer.get("pass_line") == f"FERRUM GATE {lane} PASS: {artifact_dir}"
        and outer.get("child_pass_line") == pass_prefix + artifact_dir,
        f"{backend} prepublication exact PASS lines differ",
    )
    try:
        from release_binary_gate import (
            GateError as BinaryGateError,
            resolve_evidence_ref,
            validate_gate_data,
            validate_progress_jsonl,
        )

        validate_gate_data(child, root=child_path.parent)
    except (BinaryGateError, OSError, ValueError) as error:
        raise GateError(f"{backend} prepublication child deep validation failed: {error}") from error
    require(
        child.get("status") == "pass" and child.get("mode") == lane and child.get("version") == VERSION,
        f"{backend} prepublication child status/mode/version differs",
    )
    child_started = iso_timestamp(child.get("started_at"), f"{backend} child started_at")
    child_finished = iso_timestamp(child.get("finished_at"), f"{backend} child finished_at")
    require(child_started <= child_finished <= release_published_timestamp, f"{backend} child timing is not prepublication")
    asset_evidence = child.get("evidence", {}).get("asset")
    require(isinstance(asset_evidence, dict), f"{backend} child asset evidence is missing")
    expected_sha = sha256_file(staged_asset)
    expected_size = staged_asset.stat().st_size
    require(
        asset_evidence.get("source") == "asset-path"
        and asset_evidence.get("classification") == "local-prepublication"
        and asset_evidence.get("requested_url") is None
        and PurePosixPath(str(asset_evidence.get("requested_path"))).name == expected_asset_name
        and asset_evidence.get("sha256") == expected_sha
        and asset_evidence.get("size_bytes") == expected_size,
        f"{backend} child local staged-asset provenance differs",
    )
    archive_path = resolve_evidence_ref(child_path.parent, asset_evidence.get("archive"), f"{backend} child archive")
    source_receipt_path = resolve_evidence_ref(child_path.parent, asset_evidence.get("source_receipt"), f"{backend} child source receipt")
    resolve_evidence_ref(child_path.parent, asset_evidence.get("unpacked_binary"), f"{backend} child unpacked binary")
    resolve_evidence_ref(child_path.parent, asset_evidence.get("extraction_receipt"), f"{backend} child extraction receipt")
    require(sha256_file(archive_path) == expected_sha and archive_path.stat().st_size == expected_size, f"{backend} child archive differs from staged bytes")
    source_receipt = read_json_file(source_receipt_path, f"{backend} child source receipt")
    require(
        isinstance(source_receipt, dict)
        and source_receipt.get("source_sha256") == expected_sha
        and source_receipt.get("source_size_bytes") == expected_size
        and source_receipt.get("copied_sha256") == expected_sha
        and source_receipt.get("copied_size_bytes") == expected_size
        and source_receipt.get("http_performed") is False
        and all(source_receipt.get(key) is None for key in ("requested_url", "effective_url", "effective_url_sha256", "http_status")),
        f"{backend} child source receipt does not explicitly prove local non-HTTP bytes",
    )
    try:
        validate_progress_jsonl(
            child_path.parent,
            source_receipt.get("progress"),
            f"{backend} local asset-copy progress",
            expected_size=expected_size,
        )
        extraction_receipt_path = resolve_evidence_ref(
            child_path.parent,
            asset_evidence.get("extraction_receipt"),
            f"{backend} child extraction receipt",
        )
        extraction_receipt = read_json_file(
            extraction_receipt_path, f"{backend} child extraction receipt"
        )
        validate_progress_jsonl(
            child_path.parent,
            extraction_receipt.get("progress"),
            f"{backend} bounded extraction progress",
            expected_size=extraction_receipt.get("extracted_size_bytes"),
        )
    except (BinaryGateError, OSError, ValueError) as error:
        raise GateError(f"{backend} slow-operation progress evidence differs: {error}") from error
    try:
        outer_started = datetime.fromisoformat(str(outer.get("started_at")).replace("Z", "+00:00"))
        outer_finished = datetime.fromisoformat(str(outer.get("finished_at")).replace("Z", "+00:00"))
    except ValueError as error:
        raise GateError(f"{backend} outer timing is invalid") from error
    require(
        outer_started.tzinfo is not None
        and outer_finished.tzinfo is not None
        and outer_started <= outer_finished
        and outer_finished.timestamp() <= release_published_timestamp,
        f"{backend} binary gate was not completed before prerelease publication",
    )
    child_artifacts = outer.get("child_artifacts")
    require(isinstance(child_artifacts, dict), f"{backend} child artifact binding is missing")
    child_manifest = child_artifacts.get("child_manifest")
    require(isinstance(child_manifest, dict), f"{backend} child manifest binding is missing")
    bound_path = child_manifest.get("path")
    require(
        isinstance(bound_path, str)
        and PurePosixPath(bound_path) == PurePosixPath(artifact_dir) / child_path.name,
        f"{backend} outer binds a different child path",
    )
    require(
        child_manifest.get("sha256") == sha256_file(child_path)
        and child_manifest.get("size_bytes") == child_path.stat().st_size,
        f"{backend} outer child SHA256 binding differs",
    )
    delegated = outer.get("delegated_command_line")
    require(
        isinstance(delegated, list)
        and len(delegated) >= 3
        and delegated[1].endswith("release_binary_gate.py")
        and delegated[2] == lane,
        f"{backend} delegated gate program/lane differs",
    )
    require(command_option(delegated, "--version", label=f"{backend} delegated command") == VERSION, f"{backend} delegated version differs")
    require(command_option(delegated, "--out", label=f"{backend} delegated command") == artifact_dir, f"{backend} delegated output differs")
    asset_argument = command_option(delegated, "--asset-path", label=f"{backend} delegated command")
    sha_argument = command_option(delegated, "--sha256", label=f"{backend} delegated command")
    require(
        PurePosixPath(asset_argument).name == expected_asset_name,
        f"{backend} prepublication --asset-path basename differs",
    )
    require(sha_argument == expected_sha, f"{backend} prepublication --sha256 differs")
    execution_rows = outer.get("child_execution_artifacts")
    require(isinstance(execution_rows, list) and len(execution_rows) == 3, f"{backend} child execution artifacts differ")
    execution_files: dict[str, Path] = {}
    for row in execution_rows:
        require(isinstance(row, dict) and set(row) == {"path", "sha256", "size_bytes"}, f"{backend} child execution artifact schema differs")
        name = row.get("path")
        require(name in {"run_gate.child.command.json", "run_gate.child.stdout", "run_gate.child.stderr"} and name not in execution_files, f"{backend} child execution artifact name differs")
        source = outer_path.parent / name
        require(source.is_file() and not source.is_symlink(), f"{backend} child execution artifact is missing: {name}")
        require(row.get("sha256") == sha256_file(source) and row.get("size_bytes") == source.stat().st_size, f"{backend} child execution artifact binding differs: {name}")
        execution_files[name] = source
    command_document = read_json_file(execution_files["run_gate.child.command.json"], f"{backend} child execution command")
    require(
        isinstance(command_document, dict)
        and set(command_document) == {"cmd", "cwd", "timeout_seconds", "started_at", "finished_at", "duration_seconds", "returncode", "env_overrides"}
        and command_document.get("cmd") == delegated
        and isinstance(command_document.get("cwd"), str)
        and isinstance(command_document.get("timeout_seconds"), (int, float))
        and command_document["timeout_seconds"] > 0
        and iso_timestamp(command_document.get("started_at"), f"{backend} child execution started_at")
        <= iso_timestamp(command_document.get("finished_at"), f"{backend} child execution finished_at")
        and isinstance(command_document.get("duration_seconds"), (int, float))
        and 0 <= command_document["duration_seconds"] <= command_document["timeout_seconds"]
        and command_document.get("returncode") == 0
        and command_document.get("env_overrides") == {"PYTHONDONTWRITEBYTECODE": "1"},
        f"{backend} child execution command evidence differs",
    )
    stdout = strict_utf8(execution_files["run_gate.child.stdout"].read_bytes(), f"{backend} child stdout")
    require(pass_prefix + artifact_dir in stdout.splitlines(), f"{backend} child stdout lacks exact PASS line")
    from run_gate import standard_g0_artifact_tree

    computed_tree = standard_g0_artifact_tree(outer_path.parent)
    require(child_artifacts.get("artifact_tree") == computed_tree, f"{backend} outer artifact-tree binding differs")
    tree_files: dict[str, Path] = {}
    for row in computed_tree["files"]:
        relative = row["path"]
        source = outer_path.parent.joinpath(*PurePosixPath(relative).parts)
        tree_files[relative] = source
    require(child_path.resolve() in {path.resolve() for path in tree_files.values()}, f"{backend} artifact tree omits child gate")
    require(set(execution_files).issubset(tree_files), f"{backend} artifact tree omits child execution evidence")
    return tree_files


SOURCE_GATE_LANES = {
    "unit": "unit",
    "metal": "metal",
    "cuda_full": "cuda-full",
    "cuda_llama_dense": "cuda-llama-dense",
}
SOURCE_CHILD_PASS_NAMES = {
    "unit": "unit",
    "metal": "metal",
    "cuda_full": "g0_cuda4090_full",
    "cuda_llama_dense": "g0_cuda4090_llama_dense",
}


def validate_source_gate_pair(
    *, label: str, outer_path: Path, child_path: Path, candidate_sha: str, publication_cutoff: float
) -> list[Path]:
    lane = SOURCE_GATE_LANES[label]
    outer = read_json_file(outer_path, f"{label} source outer")
    child = read_json_file(child_path, f"{label} source child")
    require(isinstance(outer, dict) and isinstance(child, dict), f"{label} source gate documents differ")
    artifact_dir = outer.get("artifact_dir")
    require(
        outer.get("schema_version") == SCHEMA_VERSION
        and outer.get("lane") == lane
        and outer.get("status") == "pass"
        and outer.get("child_returncode") == 0
        and outer.get("git_sha") == candidate_sha
        and outer.get("dirty_status") == {"is_dirty": False, "status_short": []}
        and isinstance(artifact_dir, str)
        and outer.get("pass_line") == f"FERRUM GATE {lane} PASS: {artifact_dir}",
        f"{label} source outer identity/PASS differs",
    )
    require(
        outer.get("child_pass_line")
        == f"G0 SOURCE {SOURCE_CHILD_PASS_NAMES[label]} PASS: {artifact_dir}",
        f"{label} source child exact PASS line differs",
    )
    require(child.get("status") == "pass", f"{label} source child is not PASS")
    started = iso_timestamp(outer.get("started_at"), f"{label} source started_at")
    finished = iso_timestamp(outer.get("finished_at"), f"{label} source finished_at")
    require(started <= finished <= publication_cutoff, f"{label} source gate timing is not prepublication")
    binding = outer.get("child_artifacts")
    require(isinstance(binding, dict), f"{label} source child binding is missing")
    child_manifest = binding.get("child_manifest")
    require(isinstance(child_manifest, dict), f"{label} source child manifest binding is missing")
    require(
        child_manifest.get("sha256") == sha256_file(child_path)
        and child_manifest.get("size_bytes") == child_path.stat().st_size
        and PurePosixPath(str(child_manifest.get("path"))) == PurePosixPath(artifact_dir) / child_path.name,
        f"{label} source child path/SHA binding differs",
    )
    tree = binding.get("artifact_tree")
    require(isinstance(tree, dict) and set(tree) == {"schema_version", "kind", "file_count", "total_size_bytes", "files", "sha256"}, f"{label} source artifact tree schema differs")
    rows = tree.get("files")
    require(tree.get("schema_version") == 1 and tree.get("kind") == "standard-g0-regular-file-tree" and isinstance(rows, list) and rows, f"{label} source artifact tree differs")
    paths: list[Path] = []
    normalized_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"path", "sha256", "size_bytes"}, f"{label} source artifact row differs")
        relative = row.get("path")
        require(isinstance(relative, str) and relative not in seen, f"{label} source artifact path differs")
        pure = PurePosixPath(relative)
        require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in relative, f"{label} source artifact escapes")
        require(not ({part.lower() for part in pure.parts} & {"cache", "huggingface", "hub", "models", "model-cache"}) and pure.suffix.lower() not in {".gguf", ".safetensors", ".pt", ".pth"}, f"{label} source tree contains model/cache bytes")
        source = outer_path.parent.joinpath(*pure.parts)
        require(source.is_file() and not source.is_symlink(), f"{label} source artifact is missing: {relative}")
        require(row.get("sha256") == sha256_file(source) and row.get("size_bytes") == source.stat().st_size, f"{label} source artifact binding differs: {relative}")
        seen.add(relative)
        normalized_rows.append(row)
        paths.append(source)
    require(tree.get("file_count") == len(rows) and tree.get("total_size_bytes") == sum(row["size_bytes"] for row in rows) and tree.get("sha256") == pretty_json_sha256(normalized_rows), f"{label} source artifact tree aggregate differs")
    require(child_path.resolve() in {path.resolve() for path in paths}, f"{label} source tree omits supplied child")
    return paths


def copy_portable_manifest_closure(source_manifest: Path, destination_root: Path, label: str) -> Path:
    source_root = source_manifest.parent.resolve()
    destination_root.mkdir(parents=True, exist_ok=False)
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve()
        require(resolved.is_relative_to(source_root) and path.is_file() and not path.is_symlink(), f"{label} evidence escapes or is linked")
        if resolved in visited:
            return
        visited.add(resolved)
        relative = resolved.relative_to(source_root)
        destination = destination_root / relative
        copy_evidence_file(resolved, destination)
        if resolved.suffix != ".json":
            return
        document = read_json_file(resolved, f"{label} {relative}")

        def scan(value: Any, where: str) -> None:
            if is_saved_file_ref(value):
                # Source-bundle manifests use the same byte-identity shape for
                # archive-internal members; those are not filesystem refs.
                if ".members[" in where:
                    return
                nested = resolve_nested_saved_ref(value, backend_root=source_root, containing_file=resolved, label=where)
                visit(nested)
            elif isinstance(value, dict):
                for key, child in value.items():
                    scan(child, f"{where}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    scan(child, f"{where}[{index}]")

        scan(document, label)

    visit(source_manifest)
    return destination_root / source_manifest.name


def copy_evidence_file(source: Path, destination: Path) -> None:
    require(source.is_file() and not source.is_symlink(), f"evidence input is not regular: {source}")
    require(not destination.exists(), f"evidence destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    require(
        destination.stat().st_size == source.stat().st_size
        and sha256_file(destination) == sha256_file(source),
        f"copied evidence bytes differ: {source}",
    )


def validate_portable_public_source_receipt(
    path: Path,
    *,
    root: Path,
    expected_name: str,
) -> dict[str, Any]:
    receipt = read_json_file(path, f"portable public source receipt {expected_name}")
    require(
        isinstance(receipt, dict)
        and set(receipt)
        == {
            "schema_version",
            "artifact_type",
            "status",
            "asset_name",
            "asset",
            "url",
            "effective_url",
            "http_status",
            "started_at",
            "finished_at",
            "duration_seconds",
            "timeout_seconds",
            "progress_interval_seconds",
            "download",
            "progress",
            "source_receipt_sha256",
        }
        and receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("artifact_type")
        == "ferrum_v084_portable_public_asset_source_receipt"
        and receipt.get("status") == "pass"
        and receipt.get("asset_name") == expected_name
        and receipt.get("url") == expected_public_asset_url(expected_name)
        and receipt.get("http_status") == 200
        and isinstance(receipt.get("effective_url"), str)
        and isinstance(receipt.get("source_receipt_sha256"), str)
        and SHA256_RE.fullmatch(receipt["source_receipt_sha256"]) is not None,
        f"portable public source receipt schema/provenance differs: {expected_name}",
    )
    validate_https_url(receipt["effective_url"])
    asset = receipt.get("asset")
    require(
        isinstance(asset, dict)
        and asset.get("name") == expected_name
        and isinstance(asset.get("size_bytes"), int)
        and not isinstance(asset.get("size_bytes"), bool)
        and asset["size_bytes"] > 0
        and isinstance(asset.get("sha256"), str)
        and SHA256_RE.fullmatch(asset["sha256"]) is not None,
        f"portable public source asset identity differs: {expected_name}",
    )
    download = resolve_saved_ref(
        receipt.get("download"),
        root=root,
        label=f"portable public download {expected_name}",
        require_nonempty=True,
    )
    require(
        download.name == expected_name
        and download.stat().st_size == asset["size_bytes"]
        and sha256_file(download) == asset["sha256"],
        f"portable public downloaded bytes differ: {expected_name}",
    )
    started = iso_timestamp(receipt.get("started_at"), f"portable {expected_name} started_at")
    finished = iso_timestamp(receipt.get("finished_at"), f"portable {expected_name} finished_at")
    timeout = receipt.get("timeout_seconds")
    duration = receipt.get("duration_seconds")
    require(
        finished >= started
        and isinstance(timeout, (int, float))
        and timeout > 0
        and isinstance(duration, (int, float))
        and 0 <= duration <= timeout,
        f"portable public source timing differs: {expected_name}",
    )
    progress = resolve_saved_ref(
        receipt.get("progress"),
        root=root,
        label=f"portable public progress {expected_name}",
        require_nonempty=True,
    )
    previous_elapsed = -1.0
    previous_bytes = -1
    rows = progress.read_bytes().splitlines()
    require(rows, f"portable public progress is empty: {expected_name}")
    complete_count = 0
    for index, line in enumerate(rows):
        row = parse_json_bytes(line, f"portable public progress {expected_name} row {index}")
        expected_fields = {"observed_at", "elapsed_seconds", "bytes_downloaded", "expected_bytes"}
        if isinstance(row, dict) and row.get("complete") is True:
            expected_fields.add("complete")
            complete_count += 1
        require(
            isinstance(row, dict) and set(row) == expected_fields,
            f"portable public progress row schema differs: {expected_name}",
        )
        elapsed = row.get("elapsed_seconds")
        byte_count = row.get("bytes_downloaded")
        require(
            isinstance(elapsed, (int, float))
            and elapsed >= previous_elapsed
            and elapsed <= timeout
            and isinstance(byte_count, int)
            and not isinstance(byte_count, bool)
            and byte_count >= previous_bytes
            and byte_count <= asset["size_bytes"]
            and row.get("expected_bytes") == asset["size_bytes"],
            f"portable public progress monotonicity differs: {expected_name}",
        )
        previous_elapsed = float(elapsed)
        previous_bytes = byte_count
    require(
        complete_count == 1
        and isinstance(row, dict)
        and row.get("complete") is True
        and row.get("bytes_downloaded") == asset["size_bytes"],
        f"portable public progress completion differs: {expected_name}",
    )
    return receipt


def validate_public_download_provenance_wrapper(
    path: Path,
    *,
    root: Path,
    expected_name: str,
) -> dict[str, Any]:
    wrapper = read_json_file(path, f"public provenance wrapper {expected_name}")
    require(
        isinstance(wrapper, dict)
        and set(wrapper)
        == {
            "schema_version",
            "artifact_type",
            "status",
            "backend_lane",
            "asset_name",
            "url",
            "effective_url",
            "http_status",
            "started_at",
            "finished_at",
            "duration_seconds",
            "timeout_seconds",
            "download",
            "progress",
            "source_receipt",
        }
        and wrapper.get("schema_version") == SCHEMA_VERSION
        and wrapper.get("artifact_type") == "ferrum_v084_public_asset_download_provenance"
        and wrapper.get("status") == "pass"
        and wrapper.get("asset_name") == expected_name,
        f"public provenance wrapper schema differs: {expected_name}",
    )
    source_path = resolve_saved_ref(
        wrapper.get("source_receipt"),
        root=root,
        label=f"portable source receipt {expected_name}",
        require_nonempty=True,
    )
    source = validate_portable_public_source_receipt(
        source_path, root=root, expected_name=expected_name
    )
    require(
        wrapper.get("download") == source.get("download")
        and wrapper.get("progress") == source.get("progress")
        and all(wrapper.get(key) == source.get(key) for key in (
            "url",
            "effective_url",
            "http_status",
            "started_at",
            "finished_at",
            "duration_seconds",
            "timeout_seconds",
        )),
        f"public provenance wrapper/source binding differs: {expected_name}",
    )
    return wrapper


def validate_backend_public_downloads(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    backend: str,
    release_published_timestamp: float,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    spec = BACKEND_SPECS[backend]
    downloads = summary.get("downloads")
    require(isinstance(downloads, dict), f"{backend} public download receipts are missing")
    require(
        set(downloads) == set(backend_download_asset_names(spec)),
        f"{backend} public download receipt denominator differs",
    )
    validated_downloads: dict[str, dict[str, Any]] = {}
    for name in backend_download_asset_names(spec):
        receipt = downloads[name]
        require(isinstance(receipt, dict), f"{backend} download receipt {name} is missing")
        require(receipt.get("status") == "pass", f"{backend} download receipt {name} is not pass")
        require(receipt.get("url") == expected_public_asset_url(name), f"{backend} download URL differs: {name}")
        require(receipt.get("http_status") == 200, f"{backend} download HTTP status differs: {name}")
        downloaded = resolve_saved_ref(
            receipt.get("download"),
            root=summary_path.parent,
            label=f"{backend} downloaded {name}",
            require_nonempty=True,
        )
        receipt_path = resolve_saved_ref(
            receipt.get("receipt"),
            root=summary_path.parent,
            label=f"{backend} HTTP download receipt {name}",
            require_nonempty=True,
        )
        receipt_document = read_json_file(receipt_path, f"{backend} HTTP download receipt {name}")
        require(isinstance(receipt_document, dict), f"{backend} HTTP receipt {name} is invalid")
        require(
            set(receipt_document)
            == {
                "schema_version",
                "kind",
                "asset",
                "url",
                "started_at",
                "timeout_seconds",
                "progress_interval_seconds",
                "progress_log",
                "status",
                "http_status",
                "effective_url",
                "download",
                "finished_at",
                "duration_seconds",
            },
            f"{backend} HTTP receipt schema differs: {name}",
        )
        require(
            receipt_document.get("kind") == "public_github_release_asset_download"
            and receipt_document.get("status") == "pass"
            and receipt_document.get("http_status") == 200
            and receipt_document.get("url") == expected_public_asset_url(name),
            f"{backend} HTTP provenance differs: {name}",
        )
        effective_url = receipt_document.get("effective_url")
        require(isinstance(effective_url, str), f"{backend} effective URL is missing: {name}")
        validate_https_url(effective_url)
        receipt_downloaded = resolve_saved_ref(
            receipt_document.get("download"),
            root=summary_path.parent,
            label=f"{backend} HTTP receipt downloaded file {name}",
            require_nonempty=True,
        )
        require(
            downloaded == receipt_downloaded,
            f"{backend} summary and HTTP receipt bind different bytes: {name}",
        )
        asset_identity = receipt_document.get("asset")
        require(
            isinstance(asset_identity, dict)
            and asset_identity == summary["api_assets"][name],
            f"{backend} HTTP receipt asset identity differs: {name}",
        )
        started_at = receipt_document.get("started_at")
        finished_at = receipt_document.get("finished_at")
        timeout_seconds = receipt_document.get("timeout_seconds")
        duration_seconds = receipt_document.get("duration_seconds")
        require(
            isinstance(started_at, str)
            and isinstance(finished_at, str)
            and isinstance(timeout_seconds, (int, float))
            and timeout_seconds > 0
            and isinstance(duration_seconds, (int, float))
            and 0 <= duration_seconds <= timeout_seconds,
            f"{backend} HTTP receipt timing/deadline differs: {name}",
        )
        require(
            iso_timestamp(started_at, f"{backend} HTTP {name} started_at") >= release_published_timestamp
            and iso_timestamp(finished_at, f"{backend} HTTP {name} finished_at") >= iso_timestamp(started_at, f"{backend} HTTP {name} started_at"),
            f"{backend} public download did not occur after prerelease publication: {name}",
        )
        progress_text = receipt_document.get("progress_log")
        require(isinstance(progress_text, str) and progress_text, f"{backend} progress path is missing")
        progress_pure = PurePosixPath(progress_text)
        require(
            not progress_pure.is_absolute() and ".." not in progress_pure.parts and "\\" not in progress_text,
            f"{backend} progress path escapes artifact root: {name}",
        )
        progress_path = summary_path.parent.joinpath(*progress_pure.parts).resolve()
        require(
            progress_path.is_relative_to(summary_path.parent.resolve())
            and progress_path.is_file()
            and not progress_path.is_symlink(),
            f"{backend} progress evidence is missing: {name}",
        )
        progress_rows: list[dict[str, Any]] = []
        previous_elapsed = -1.0
        previous_bytes = -1
        for index, line in enumerate(progress_path.read_bytes().splitlines()):
            require(line.strip(), f"{backend} progress contains an empty row: {name}")
            row = parse_json_bytes(line, f"{backend} progress {name} row {index}")
            require(isinstance(row, dict), f"{backend} progress row is not an object: {name}")
            expected_fields = {"observed_at", "elapsed_seconds", "bytes_downloaded", "expected_bytes"}
            if row.get("complete") is True:
                expected_fields.add("complete")
            require(set(row) == expected_fields, f"{backend} progress row schema differs: {name}")
            elapsed = row.get("elapsed_seconds")
            downloaded_bytes = row.get("bytes_downloaded")
            expected_bytes = row.get("expected_bytes")
            require(
                isinstance(elapsed, (int, float))
                and elapsed >= previous_elapsed
                and elapsed <= timeout_seconds
                and isinstance(downloaded_bytes, int)
                and not isinstance(downloaded_bytes, bool)
                and downloaded_bytes >= previous_bytes
                and isinstance(expected_bytes, int)
                and expected_bytes == asset_identity["size_bytes"]
                and downloaded_bytes <= expected_bytes,
                f"{backend} progress monotonicity/deadline differs: {name}",
            )
            previous_elapsed = float(elapsed)
            previous_bytes = downloaded_bytes
            progress_rows.append(row)
        require(
            progress_rows
            and sum(row.get("complete") is True for row in progress_rows) == 1
            and progress_rows[-1].get("complete") is True
            and progress_rows[-1]["bytes_downloaded"] == asset_identity["size_bytes"],
            f"{backend} progress completion receipt differs: {name}",
        )
        validated_downloads[name] = {
            "path": downloaded,
            "receipt_path": receipt_path,
            "receipt": receipt_document,
            "progress_path": progress_path,
        }
    goal_ref = summary.get("goal_e2e_summary")
    goal_path = resolve_saved_ref(
        goal_ref,
        root=summary_path.parent,
        label=f"{backend} goal E2E summary",
        require_nonempty=True,
    )
    goal = read_json_file(goal_path, f"{backend} goal E2E summary")
    require(isinstance(goal, dict), f"{backend} goal E2E summary root is not an object")
    require(
        goal.get("artifact_type") == GOAL_E2E_ARTIFACT_TYPE
        and goal.get("status") == "pass"
        and goal.get("version") == VERSION
        and goal.get("backend") == backend,
        f"{backend} goal E2E identity/status differs",
    )
    candidate = summary["adjacent_bundle"]["release_candidate"]
    require(
        goal.get("source_git_sha") == candidate["release_candidate_sha"],
        f"{backend} goal E2E candidate differs",
    )
    require(
        goal.get("asset_name") == spec.asset_name
        and goal.get("asset_sha256") == summary["asset"]["public_sha256"]
        and goal.get("binary_sha256") == summary["binary"]["sha256"],
        f"{backend} goal E2E binary/asset binding differs",
    )
    execution = goal.get("execution")
    require(
        isinstance(execution, dict)
        and iso_timestamp(execution.get("started_at"), f"{backend} E2E started_at") >= release_published_timestamp
        and iso_timestamp(execution.get("finished_at"), f"{backend} E2E finished_at") >= iso_timestamp(execution.get("started_at"), f"{backend} E2E started_at"),
        f"{backend} README E2E did not occur after prerelease publication",
    )
    require(
        isinstance(summary.get("readme_contract"), dict)
        and summary["readme_contract"].get("download_size_announced_before_run") is True,
        f"{backend} packaged README receipt is missing",
    )
    return goal_path, validated_downloads


def is_saved_file_ref(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"path", "sha256", "size_bytes"}


def resolve_nested_saved_ref(
    raw: dict[str, Any],
    *,
    backend_root: Path,
    containing_file: Path,
    label: str,
) -> Path:
    errors: list[str] = []
    for base in (containing_file.parent, backend_root):
        try:
            path = resolve_saved_ref(raw, root=base, label=label)
            path.relative_to(backend_root.resolve())
            return path
        except (GateError, ValueError) as error:
            errors.append(str(error))
    raise GateError(f"cannot resolve portable nested reference {label}: {'; '.join(errors)}")


def relocate_reachable_evidence_file(
    source: Path,
    *,
    backend_root: Path,
    destination_root: Path,
    relocated: dict[Path, Path],
    active: set[Path],
) -> Path:
    source = source.resolve()
    require(source.is_relative_to(backend_root.resolve()), "E2E evidence escapes backend root")
    if source in relocated:
        destination = relocated[source]
        require(destination.is_file(), f"relocated evidence is incomplete: {destination}")
        return destination
    require(source not in active, f"cyclic evidence reference: {source}")
    relative = source.relative_to(backend_root.resolve())
    # Preserve the backend artifact's root-relative graph so recorded command
    # paths can be replayed against the portable root without a hidden prefix.
    destination = destination_root / relative
    require(not destination.exists(), f"portable evidence destination collides: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    active.add(source)
    try:
        if source.suffix == ".json":
            document = read_json_file(source, f"portable E2E evidence {relative}")

            def rewrite(value: Any, where: str) -> Any:
                if is_saved_file_ref(value):
                    nested = resolve_nested_saved_ref(
                        value,
                        backend_root=backend_root,
                        containing_file=source,
                        label=where,
                    )
                    copied = relocate_reachable_evidence_file(
                        nested,
                        backend_root=backend_root,
                        destination_root=destination_root,
                        relocated=relocated,
                        active=active,
                    )
                    return file_ref(copied, relative_to=destination_root)
                if isinstance(value, dict):
                    return {
                        key: rewrite(child, f"{where}.{key}")
                        for key, child in value.items()
                    }
                if isinstance(value, list):
                    return [rewrite(child, f"{where}[{index}]") for index, child in enumerate(value)]
                return value

            write_json_atomic(destination, rewrite(document, relative.as_posix()))
        else:
            copy_evidence_file(source, destination)
        relocated[source] = destination
        return destination
    finally:
        active.discard(source)


def relocate_goal_e2e_summary(
    source_path: Path,
    *,
    backend: str,
    destination_root: Path,
) -> Path:
    document = read_json_file(source_path, f"{backend} source goal E2E summary")
    require(isinstance(document, dict), f"{backend} source goal E2E root is not an object")
    evidence = document.get("evidence")
    require(
        isinstance(evidence, dict) and set(evidence) == GOAL_EVIDENCE_KEYS,
        f"{backend} source goal E2E evidence denominator differs",
    )
    destination_root.mkdir(parents=True, exist_ok=False)
    backend_root = source_path.parent.resolve()
    relocated_paths: dict[Path, Path] = {}
    relocated_refs: dict[str, Any] = {}
    process_expectations = {
        "binary_version": ("binary-version", "pass", 0),
        "binary_help": ("binary-help", "pass", 0),
        "doctor": ("doctor-model", "pass", 0),
        "run": ("readme-run", "pass", 0),
        "serve": ("readme-serve", "terminated", None),
    }
    for label in sorted(GOAL_EVIDENCE_KEYS):
        source = resolve_saved_ref(
            evidence[label],
            root=backend_root,
            label=f"{backend} goal E2E evidence {label}",
            require_nonempty=True,
        )
        if label in process_expectations:
            process_label, status, returncode = process_expectations[label]
            validate_portable_process_receipt(
                source,
                root=backend_root,
                label=process_label,
                expected_status=status,
                expected_returncode=returncode,
            )
        destination = relocate_reachable_evidence_file(
            source,
            backend_root=backend_root,
            destination_root=destination_root,
            relocated=relocated_paths,
            active=set(),
        )
        relocated_refs[label] = file_ref(destination, relative_to=destination_root)
    document["evidence"] = relocated_refs
    network = document.get("network_environment")
    require(
        isinstance(network, dict)
        and set(network) == {"urllib_public_downloads", "child_processes"},
        f"{backend} source goal E2E network-environment denominator differs",
    )
    relocated_network: dict[str, Any] = {}
    network_consumers = {
        "urllib_public_downloads": "urllib-public-github-downloads",
        "child_processes": "ferrum-child-processes",
    }
    for label, raw_ref in network.items():
        source = resolve_saved_ref(
            raw_ref,
            root=backend_root,
            label=f"{backend} network environment {label}",
            require_nonempty=True,
        )
        validate_network_environment_document(
            read_json_file(source, f"{backend} network environment {label}"),
            consumer=network_consumers[label],
        )
        destination = relocate_reachable_evidence_file(
            source,
            backend_root=backend_root,
            destination_root=destination_root,
            relocated=relocated_paths,
            active=set(),
        )
        relocated_network[label] = file_ref(destination, relative_to=destination_root)
    document["network_environment"] = relocated_network
    cold_cache = document.get("cold_cache")
    require(isinstance(cold_cache, dict), f"{backend} cold-cache receipt differs")
    undocumented = cold_cache.get("undocumented_behavior_env")
    require(
        isinstance(undocumented, dict)
        and undocumented.get("behavior_overrides") == []
        and undocumented.get("network_routing_is_behavior_override") is False,
        f"{backend} network routing was misclassified as a behavior override",
    )
    undocumented["network_environment"] = relocated_network
    destination = destination_root / "summary.json"
    write_json_atomic(destination, document)
    return destination


def run_assemble(args: argparse.Namespace) -> int:
    assembly_started_at = iso_now()
    out = prepare_fresh_output(args.out)
    manifest_path = out / f"ferrum-{VERSION}-prerelease-manifest.json"
    try:
        metal_path = args.metal_summary.expanduser().resolve()
        cuda_path = args.cuda_summary.expanduser().resolve()
        require(metal_path != cuda_path, "Metal and CUDA backend summary paths must differ")
        metal = load_backend_summary(metal_path, "metal")
        cuda = load_backend_summary(cuda_path, "cuda")
        aggregate = aggregate_documents(metal, cuda)

        staged_inputs = exact_asset_directory(args.staged_assets_dir, "staged asset directory")
        release_input = args.release_snapshot.expanduser().resolve()
        tag_input = args.tag_snapshot.expanduser().resolve()
        tag_ref_input = args.tag_ref_snapshot.expanduser().resolve()
        rc_tag_input = args.rc_tag_snapshot.expanduser().resolve()
        rc_tag_ref_input = args.rc_tag_ref_snapshot.expanduser().resolve()
        require(
            release_input.is_file() and not release_input.is_symlink(),
            "release snapshot is not a regular file",
        )
        require(tag_input.is_file() and not tag_input.is_symlink(), "tag snapshot is not a regular file")
        require(
            tag_ref_input.is_file() and not tag_ref_input.is_symlink(),
            "tag ref snapshot is not a regular file",
        )
        release_raw = read_json_file(release_input, "goal release snapshot")
        release_identity, release_rows = validate_goal_release_snapshot(release_raw)
        require(
            release_identity["id"] == aggregate["release"]["id"]
            and release_identity["snapshot_sha256"]
            == aggregate["release"]["immutable_snapshot_sha256"],
            "backend E2E receipts and supplied release snapshot differ",
        )
        candidate = aggregate["release_candidate"]
        candidate_sha = candidate["release_candidate_sha"]
        rc_tag = candidate["release_candidate_tag"]
        tag_raw = read_json_file(tag_input, "annotated tag snapshot")
        validate_goal_tag_snapshot(tag_raw, candidate_sha=candidate_sha)
        tag_ref_raw = read_json_file(tag_ref_input, "tag ref snapshot")
        validate_goal_tag_ref_snapshot(tag_ref_raw, annotated_tag_sha=tag_raw["sha"])
        for label, path in (("RC tag", rc_tag_input), ("RC tag ref", rc_tag_ref_input)):
            require(path.is_file() and not path.is_symlink(), f"{label} snapshot is not regular")
        rc_tag_raw = read_json_file(rc_tag_input, "RC annotated tag snapshot")
        rc_tag_ref_raw = read_json_file(rc_tag_ref_input, "RC tag ref snapshot")
        validate_rc_tag_chain(rc_tag_ref_raw, rc_tag_raw, rc_tag=rc_tag, candidate_sha=candidate_sha)

        publication_cutoff = min(release_identity["created_timestamp"], release_identity["published_timestamp"])
        source_inputs = {
            "unit": {"outer": args.unit_outer.expanduser().resolve(), "child": args.unit_child.expanduser().resolve()},
            "metal": {"outer": args.metal_source_outer.expanduser().resolve(), "child": args.metal_source_child.expanduser().resolve()},
            "cuda_full": {"outer": args.cuda_full_outer.expanduser().resolve(), "child": args.cuda_full_child.expanduser().resolve()},
            "cuda_llama_dense": {"outer": args.cuda_llama_dense_outer.expanduser().resolve(), "child": args.cuda_llama_dense_child.expanduser().resolve()},
        }
        source_tree_inputs: dict[str, list[Path]] = {}
        for label, paths in source_inputs.items():
            require(all(path.is_file() and not path.is_symlink() for path in paths.values()), f"{label} source gate pair is not regular")
            source_tree_inputs[label] = validate_source_gate_pair(
                label=label,
                outer_path=paths["outer"],
                child_path=paths["child"],
                candidate_sha=candidate_sha,
                publication_cutoff=publication_cutoff,
            )

        workflow_input = args.workflow_policy_manifest.expanduser().resolve()
        native_input = args.native_set_manifest.expanduser().resolve()
        require(workflow_input.is_file() and not workflow_input.is_symlink(), "workflow policy manifest is not regular")
        require(native_input.is_file() and not native_input.is_symlink(), "native-set manifest is not regular")
        from v084_workflow_native_gate import validate_native_set_manifest, validate_workflow_policy_manifest
        workflow_validated = validate_workflow_policy_manifest(workflow_input)
        native_validated = validate_native_set_manifest(native_input)
        for label, path, document, validated_document in (
            ("workflow policy", workflow_input, read_json_file(workflow_input, "workflow policy manifest"), workflow_validated),
            ("native operator set", native_input, read_json_file(native_input, "native-set manifest"), native_validated),
        ):
            require(isinstance(validated_document, dict), f"{label} validator returned no identity")
            require(isinstance(document, dict), f"{label} manifest differs")
            evidence_candidate = document.get("evidence", {}).get("candidate")
            require(isinstance(evidence_candidate, dict) and evidence_candidate.get("git_sha") == candidate_sha and evidence_candidate.get("tag") == rc_tag, f"{label} candidate differs")
            started = iso_timestamp(document.get("started_at"), f"{label} started_at")
            finished = iso_timestamp(document.get("finished_at"), f"{label} finished_at")
            require(started <= finished <= publication_cutoff, f"{label} timing is not prepublication")

        gate_input_paths = {
            "metal": {
                "outer": args.metal_tarball_outer.expanduser().resolve(),
                "child": args.metal_tarball_child.expanduser().resolve(),
            },
            "cuda": {
                "outer": args.cuda_tarball_outer.expanduser().resolve(),
                "child": args.cuda_tarball_child.expanduser().resolve(),
            },
        }
        gate_execution_inputs: dict[str, dict[str, Path]] = {}
        for backend, paths in gate_input_paths.items():
            for role, path in paths.items():
                require(
                    path.is_file() and not path.is_symlink(),
                    f"{backend} prepublication {role} gate is not a regular file",
                )
            gate_execution_inputs[backend] = validate_prepublication_binary_gate_pair(
                backend=backend,
                outer_path=paths["outer"],
                child_path=paths["child"],
                candidate_sha=candidate_sha,
                staged_asset=staged_inputs[BACKEND_SPECS[backend].asset_name],
                release_published_timestamp=publication_cutoff,
            )

        release_by_name = {row["name"]: row for row in release_rows}
        backend_results = {
            "metal": validate_backend_public_downloads(
                metal,
                summary_path=metal_path,
                backend="metal",
                release_published_timestamp=release_identity["published_timestamp"],
            ),
            "cuda": validate_backend_public_downloads(
                cuda,
                summary_path=cuda_path,
                backend="cuda",
                release_published_timestamp=release_identity["published_timestamp"],
            ),
        }
        goal_paths = {backend: result[0] for backend, result in backend_results.items()}
        public_downloads: dict[str, dict[str, Any]] = {}
        for backend, (_goal_path, rows) in backend_results.items():
            for name, row in rows.items():
                require(name not in public_downloads, f"duplicate public download owner: {name}")
                public_downloads[name] = {**row, "backend": backend}
        require(
            set(public_downloads) == GOAL_EXPECTED_ASSETS,
            "real public download receipt denominator differs",
        )
        staged_refs: dict[str, Any] = {}
        public_receipts: dict[str, Any] = {}
        staged_out = out / "staged-assets"
        public_out = out / "public-downloads"
        provenance_out = out / "public-download-provenance"
        staged_out.mkdir()
        public_out.mkdir()
        provenance_out.mkdir()
        for name in sorted(GOAL_EXPECTED_ASSETS):
            staged_source = staged_inputs[name]
            public_row = public_downloads[name]
            public_source = public_row["path"]
            staged_sha = sha256_file(staged_source)
            public_sha = sha256_file(public_source)
            row = release_by_name[name]
            require(
                staged_source.stat().st_size == public_source.stat().st_size == row["size"]
                and staged_sha == public_sha == row["digest"][7:],
                f"staged/public/GitHub bytes differ: {name}",
            )
            staged_destination = staged_out / name
            public_destination = public_out / name
            copy_evidence_file(staged_source, staged_destination)
            copy_evidence_file(public_source, public_destination)
            staged_refs[name] = file_ref(staged_destination, relative_to=out)
            source_receipt = public_row["receipt"]
            portable_source_receipt_path = provenance_out / f"{name}.source-receipt.json"
            copied_progress = provenance_out / f"{name}.progress.jsonl"
            copy_evidence_file(public_row["progress_path"], copied_progress)
            portable_download_ref = file_ref(public_destination, relative_to=out)
            portable_progress_ref = file_ref(copied_progress, relative_to=out)
            portable_source_receipt = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "ferrum_v084_portable_public_asset_source_receipt",
                "status": "pass",
                "asset_name": name,
                "asset": source_receipt["asset"],
                "url": source_receipt["url"],
                "effective_url": source_receipt["effective_url"],
                "http_status": source_receipt["http_status"],
                "started_at": source_receipt["started_at"],
                "finished_at": source_receipt["finished_at"],
                "duration_seconds": source_receipt["duration_seconds"],
                "timeout_seconds": source_receipt["timeout_seconds"],
                "progress_interval_seconds": source_receipt["progress_interval_seconds"],
                "download": portable_download_ref,
                "progress": portable_progress_ref,
                "source_receipt_sha256": sha256_file(public_row["receipt_path"]),
            }
            write_json_atomic(portable_source_receipt_path, portable_source_receipt)
            validate_portable_public_source_receipt(
                portable_source_receipt_path,
                root=out,
                expected_name=name,
            )
            provenance = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "ferrum_v084_public_asset_download_provenance",
                "status": "pass",
                "backend_lane": public_row["backend"],
                "asset_name": name,
                "url": source_receipt["url"],
                "effective_url": source_receipt["effective_url"],
                "http_status": source_receipt["http_status"],
                "started_at": source_receipt.get("started_at"),
                "finished_at": source_receipt.get("finished_at"),
                "duration_seconds": source_receipt.get("duration_seconds"),
                "timeout_seconds": source_receipt.get("timeout_seconds"),
                "download": portable_download_ref,
                "progress": portable_progress_ref,
                "source_receipt": file_ref(portable_source_receipt_path, relative_to=out),
            }
            provenance_path = provenance_out / f"{name}.json"
            write_json_atomic(provenance_path, provenance)
            validate_public_download_provenance_wrapper(
                provenance_path,
                root=out,
                expected_name=name,
            )
            public_receipts[name] = {
                "url": source_receipt["url"],
                "http_status": source_receipt["http_status"],
                "file": file_ref(public_destination, relative_to=out),
                "receipt": file_ref(provenance_path, relative_to=out),
                "progress": file_ref(copied_progress, relative_to=out),
            }
        e2e_refs: dict[str, Any] = {}
        for backend in ("metal", "cuda"):
            relocated = relocate_goal_e2e_summary(
                goal_paths[backend],
                backend=backend,
                destination_root=out / "readme-e2e" / backend,
            )
            e2e_refs[backend] = file_ref(relocated, relative_to=out)

        snapshots = out / "snapshots"
        release_copy = snapshots / "release.json"
        tag_copy = snapshots / "annotated-tag.json"
        tag_ref_copy = snapshots / "tag-ref.json"
        rc_tag_copy = snapshots / "rc-annotated-tag.json"
        rc_tag_ref_copy = snapshots / "rc-tag-ref.json"
        copy_evidence_file(release_input, release_copy)
        copy_evidence_file(tag_input, tag_copy)
        copy_evidence_file(tag_ref_input, tag_ref_copy)
        copy_evidence_file(rc_tag_input, rc_tag_copy)
        copy_evidence_file(rc_tag_ref_input, rc_tag_ref_copy)
        source_refs: dict[str, Any] = {}
        for label, paths in source_inputs.items():
            destination_root = out / "source-gates" / label
            for source in source_tree_inputs[label]:
                relative = source.resolve().relative_to(paths["outer"].parent.resolve())
                copy_evidence_file(source, destination_root / relative)
            outer_destination = destination_root / "gate.manifest.json"
            if not outer_destination.exists():
                copy_evidence_file(paths["outer"], outer_destination)
            child_relative = paths["child"].resolve().relative_to(paths["outer"].parent.resolve())
            child_destination = destination_root / child_relative
            require(child_destination.is_file(), f"{label} copied source tree omits child")
            source_refs[label] = {
                "outer": file_ref(outer_destination, relative_to=out),
                "child": file_ref(child_destination, relative_to=out),
            }
        workflow_copy = copy_portable_manifest_closure(workflow_input, out / "workflow-policy", "workflow policy")
        native_copy = copy_portable_manifest_closure(native_input, out / "native-operator-set", "native operator set")
        gate_refs: dict[str, Any] = {}
        for backend, paths in gate_input_paths.items():
            destination_root = out / "prepublication-binary-gates" / backend
            for name, path in gate_execution_inputs[backend].items():
                destination = destination_root.joinpath(*PurePosixPath(name).parts)
                copy_evidence_file(path, destination)
            outer_destination = destination_root / "gate.manifest.json"
            copy_evidence_file(paths["outer"], outer_destination)
            child_destination = destination_root / paths["child"].resolve().relative_to(paths["outer"].parent.resolve())
            require(child_destination.is_file(), f"{backend} copied binary artifact tree omits child")
            gate_refs[backend] = {
                "outer": file_ref(outer_destination, relative_to=out),
                "child": file_ref(child_destination, relative_to=out),
            }
        artifact_dir = str(out)
        pass_line = f"FERRUM {VERSION} PRERELEASE DOWNLOAD PASS: {artifact_dir}"
        assembly_finished_at = iso_now()
        terminal_evidence_times = [release_identity["published_timestamp"]]
        terminal_evidence_times.extend(
            iso_timestamp(read_json_file(paths["outer"], f"{label} source outer timing").get("finished_at"), f"{label} source finished_at")
            for label, paths in source_inputs.items()
        )
        terminal_evidence_times.extend(
            iso_timestamp(read_json_file(paths["outer"], f"{backend} prepublication timing").get("finished_at"), f"{backend} prepublication finished_at")
            for backend, paths in gate_input_paths.items()
        )
        terminal_evidence_times.extend(
            iso_timestamp(read_json_file(path, f"{label} timing").get("finished_at"), f"{label} finished_at")
            for label, path in (("workflow policy", workflow_input), ("native operator set", native_input))
        )
        for backend, goal_path in goal_paths.items():
            goal_document = read_json_file(goal_path, f"{backend} goal E2E timing")
            terminal_evidence_times.append(iso_timestamp(goal_document.get("execution", {}).get("finished_at"), f"{backend} E2E finished_at"))
        terminal_evidence_times.extend(
            iso_timestamp(row["receipt"].get("finished_at"), f"public {name} finished_at")
            for name, row in public_downloads.items()
        )
        require(
            iso_timestamp(assembly_started_at, "assembly started_at")
            <= iso_timestamp(assembly_finished_at, "assembly finished_at")
            and iso_timestamp(assembly_finished_at, "assembly finished_at") >= max(terminal_evidence_times),
            "assembly timing does not cover all prerequisite evidence",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": GOAL_PRERELEASE_ARTIFACT_TYPE,
            "status": "pass",
            "version": VERSION,
            "started_at": assembly_started_at,
            "finished_at": assembly_finished_at,
            "source": {"git_sha": candidate_sha, "dirty": False},
            "release": {
                "id": release_identity["id"],
                "tag": TAG,
                "release_candidate_tag": rc_tag,
                "asset_set_sha256": release_identity["asset_set_sha256"],
            },
            "evidence": {
                "release_snapshot": file_ref(release_copy, relative_to=out),
                "tag_snapshot": file_ref(tag_copy, relative_to=out),
                "tag_ref_snapshot": file_ref(tag_ref_copy, relative_to=out),
                "rc_tag_ref_snapshot": file_ref(rc_tag_ref_copy, relative_to=out),
                "rc_tag_snapshot": file_ref(rc_tag_copy, relative_to=out),
                "source_gates": source_refs,
                "workflow_policy": file_ref(workflow_copy, relative_to=out),
                "native_operator_set": file_ref(native_copy, relative_to=out),
                "staged_assets": staged_refs,
                "public_downloads": public_receipts,
                "readme_e2e": e2e_refs,
                "prepublication_binary_gates": gate_refs,
            },
            "artifact_dir": artifact_dir,
            "pass_line": pass_line,
        }
        write_json_atomic(manifest_path, manifest)

        from v084_release_goal_gate import ValidationError, validate_prerelease_manifest

        try:
            validated = validate_prerelease_manifest(manifest_path)
        except ValidationError as error:
            raise GateError(f"assembled prerelease manifest failed goal validation: {error}") from error
        require(validated.get("pass_line") == pass_line, "assembled prerelease PASS binding differs")
        print(pass_line)
        return 0
    except Exception as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": GOAL_PRERELEASE_ARTIFACT_TYPE,
            "status": "fail",
            "version": VERSION,
            "error": str(error),
        }
        write_json_atomic(out / "assembly-failure.json", failure)
        print(f"FERRUM {VERSION} PRERELEASE ASSEMBLY FAIL: {error}", file=sys.stderr)
        return 1


def run_aggregate(args: argparse.Namespace) -> int:
    out = prepare_fresh_output(args.out)
    summary_path = out / aggregate_summary_name()
    started_at = iso_now()
    started = time.monotonic()
    partial = {
        "schema_version": SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "artifact_type": "ferrum_v084_prerelease_download_aggregate",
        "version": VERSION,
        "tag": TAG,
        "status": "running",
        "artifact_dir": str(out),
        "started_at": started_at,
        "command_line": [sys.executable, *sys.argv],
    }
    write_json_atomic(summary_path, partial)
    try:
        metal_path = args.metal_summary.expanduser().resolve()
        cuda_path = args.cuda_summary.expanduser().resolve()
        require(metal_path != cuda_path, "Metal and CUDA summary paths must differ")
        metal = load_backend_summary(metal_path, "metal")
        cuda = load_backend_summary(cuda_path, "cuda")
        aggregate = aggregate_documents(metal, cuda)
        pass_line = f"FERRUM {VERSION} PRERELEASE DOWNLOAD AGGREGATE PASS: {out}"
        summary = {
            **partial,
            "status": "pass",
            "pass_line": pass_line,
            "finished_at": iso_now(),
            "duration_seconds": time.monotonic() - started,
            "children": {
                "metal": {
                    "summary": file_ref(metal_path),
                    "pass_line": metal["pass_line"],
                },
                "cuda": {
                    "summary": file_ref(cuda_path),
                    "pass_line": cuda["pass_line"],
                },
            },
            **aggregate,
        }
        write_json_atomic(summary_path, summary)
        print(pass_line)
        return 0
    except Exception as error:
        write_json_atomic(
            summary_path,
            {
                **partial,
                "status": "fail",
                "pass_line": None,
                "finished_at": iso_now(),
                "duration_seconds": time.monotonic() - started,
                "error": str(error),
            },
        )
        print(f"FERRUM {VERSION} PRERELEASE DOWNLOAD AGGREGATE FAIL: {error}", file=sys.stderr)
        return 1


def selftest_asset(name: str, content: bytes, asset_id: int) -> dict[str, Any]:
    return {
        "id": asset_id,
        "name": name,
        "size": len(content),
        "digest": f"sha256:{sha256_bytes(content)}",
        "state": "uploaded",
        "content_type": "application/octet-stream",
        "browser_download_url": expected_public_asset_url(name),
        "created_at": "2026-09-02T00:00:00Z",
        "updated_at": "2026-09-02T00:00:00Z",
    }


def selftest_release(spec: BackendSpec) -> dict[str, Any]:
    assets = [
        selftest_asset(name, f"fixture:{name}".encode(), index + 100)
        for index, name in enumerate(backend_download_asset_names(spec))
    ]
    return {
        "id": 84,
        "tag_name": TAG,
        "target_commitish": "1" * 40,
        "name": "Ferrum 0.8.4",
        "draft": False,
        "prerelease": True,
        "created_at": "2026-09-02T00:00:00Z",
        "published_at": "2026-09-02T00:01:00Z",
        "html_url": f"https://github.com/{REPOSITORY}/releases/tag/{TAG}",
        "assets": assets,
    }


def write_tar_fixture(path: Path, members: Iterable[tuple[str, bytes, str]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(content)
                info.mode = 0o755 if PurePosixPath(name).name == "ferrum" else 0o644
                archive.addfile(info, io.BytesIO(content))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = content.decode()
                archive.addfile(info)
            else:
                raise AssertionError(kind)


def selftest_manifest_document(
    spec: BackendSpec, asset_sha: str, binary_sha: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "asset_name": spec.asset_name,
        "asset_sha256": asset_sha,
        "binary_name": "ferrum",
        "binary_sha256": binary_sha,
        "release_candidate_sha": "1" * 40,
        "release_candidate_tag": "v0.8.4-rc.1",
        "staging_label": "v0.8.4-rc",
        "workflow_run_id": "123",
        "workflow_run_attempt": "1",
    }


def make_selftest_adjacent_bundle(root: Path, spec: BackendSpec) -> tuple[str, str]:
    binary = root / "ferrum"
    binary.write_bytes(b"fixture-binary")
    binary_sha = sha256_file(binary)
    asset = root / spec.asset_name
    asset.write_bytes(b"fixture-asset")
    asset_sha = sha256_file(asset)
    (root / f"{spec.asset_name}.sha256").write_text(
        f"{asset_sha}  {spec.asset_name}\n", encoding="utf-8"
    )
    (root / f"{spec.asset_name}.binary.sha256").write_text(
        f"{binary_sha}  ferrum\n", encoding="utf-8"
    )
    audit = root / spec.dependency_audit_name
    audit.write_text("fixture dependency audit\n", encoding="utf-8")
    audit_sha = sha256_file(audit)
    common = selftest_manifest_document(spec, asset_sha, binary_sha)
    version = {**common, "version": VERSION}
    dependency = {
        **common,
        "audit_file": spec.dependency_audit_name,
        "audit_sha256": audit_sha,
        "forbidden_runtime_linkage": ["python", "torch", "vllm"],
        "forbidden_runtime_linkage_found": False,
    }
    abi: dict[str, Any] = {
        **common,
        "target_triple": spec.target_triple,
        "backend": spec.backend,
        "dependency_audit_sha256": audit_sha,
    }
    if spec.backend == "cuda":
        abi["cuda_compute_capability"] = "89"
    for suffix, document in (
        ("version.json", version),
        ("dependency.json", dependency),
        ("abi.json", abi),
    ):
        write_json_atomic(root / f"{spec.asset_name}.{suffix}", document)
    return asset_sha, binary_sha


def expect_gate_error(label: str, function: Callable[[], Any]) -> None:
    try:
        function()
    except GateError:
        return
    raise AssertionError(f"negative self-test did not fail: {label}")


def fake_backend_summary(
    backend: str,
    release_snapshot_sha: str = "a" * 64,
    candidate_sha: str = "1" * 40,
) -> dict[str, Any]:
    spec = BACKEND_SPECS[backend]
    asset_sha = ("b" if backend == "metal" else "c") * 64
    artifact_dir = f"/tmp/{backend}"
    candidate = {
        "release_candidate_sha": candidate_sha,
        "release_candidate_tag": "v0.8.4-rc.1",
        "staging_label": "v0.8.4-rc",
        "workflow_run_id": "123" if backend == "metal" else "456",
        "workflow_run_attempt": "1",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "artifact_type": "ferrum_v084_prerelease_download_backend",
        "version": VERSION,
        "tag": TAG,
        "backend": backend,
        "status": "pass",
        "artifact_dir": artifact_dir,
        "pass_line": f"FERRUM {VERSION} PRERELEASE DOWNLOAD {spec.pass_label} PASS: {artifact_dir}",
        "release": {
            "id": 84,
            "tag_name": TAG,
            "draft": False,
            "prerelease": True,
            "immutable_snapshot": {"fixture": "release"},
            "immutable_snapshot_sha256": release_snapshot_sha,
        },
        "asset": {
            "id": 1 if backend == "metal" else 2,
            "name": spec.asset_name,
            "size_bytes": 10,
            "sha256": asset_sha,
            "public_sha256": asset_sha,
            "expected_staged_sha256": asset_sha,
            "staged_public_sha256_equal": True,
        },
        "adjacent_bundle": {"release_candidate": candidate},
        "binary": {"sha256": "d" * 64},
        "checks": {"doctor": {}, "run": {}, "serve": {}},
    }


def make_assembly_selftest_inputs(root: Path) -> argparse.Namespace:
    from v084_release_goal_gate import build_selftest_fixture

    prerelease_path, _promotion_path, _final_path = build_selftest_fixture(root)
    prerelease = read_json_file(prerelease_path, "assembly fixture prerelease")
    candidate_sha = prerelease["source"]["git_sha"]
    release_path = root / prerelease["evidence"]["release_snapshot"]["path"]
    tag_path = root / prerelease["evidence"]["tag_snapshot"]["path"]
    tag = read_json_file(tag_path, "assembly fixture annotated tag")
    tag_ref_path = root / "tag-ref.json"
    write_json_atomic(
        tag_ref_path,
        {"ref": f"refs/tags/{TAG}", "object": {"type": "tag", "sha": tag["sha"]}},
    )
    release = read_json_file(release_path, "assembly fixture release")
    release_changed = False
    if "created_at" not in release:
        release["created_at"] = "2026-09-02T00:00:09+00:00"
        release_changed = True
    if "published_at" not in release:
        release["published_at"] = "2026-09-02T00:00:10+00:00"
        release_changed = True
    for asset in release.get("assets", []):
        if asset.get("state") != "uploaded":
            asset["state"] = "uploaded"
            release_changed = True
    if release_changed:
        write_json_atomic(release_path, release)
        prerelease["evidence"]["release_snapshot"] = file_ref(release_path, relative_to=root)
        write_json_atomic(prerelease_path, prerelease)
    projection = release_identity_projection(release)
    snapshot_sha = canonical_json_sha256(projection)
    release_assets = {row["name"]: row for row in release["assets"]}
    for backend in ("metal", "cuda"):
        spec = BACKEND_SPECS[backend]
        summary = fake_backend_summary(backend, snapshot_sha, candidate_sha)
        summary["release"]["id"] = release["id"]
        summary["release"]["immutable_snapshot"] = projection
        _release_identity, api_assets = validate_release_snapshot(release, spec)
        summary["api_assets"] = api_assets
        e2e_ref = prerelease["evidence"]["readme_e2e"][backend]
        e2e_path = root / e2e_ref["path"]
        e2e = read_json_file(e2e_path, f"assembly fixture {backend} E2E")
        asset_row = release_assets[spec.asset_name]
        summary["asset"].update(
            {
                "id": asset_row["id"],
                "size_bytes": asset_row["size"],
                "sha256": e2e["asset_sha256"],
                "public_sha256": e2e["asset_sha256"],
                "expected_staged_sha256": e2e["asset_sha256"],
            }
        )
        summary["binary"]["sha256"] = e2e["binary_sha256"]
        summary["readme_contract"] = {"download_size_announced_before_run": True}
        summary["goal_e2e_summary"] = file_ref(e2e_path, relative_to=root)
        summary["downloads"] = {}
        for name in backend_download_asset_names(spec):
            downloaded = root / "files" / name
            receipt_path = root / "download-receipts" / backend / f"{name}.json"
            progress_path = root / "download-receipts" / backend / f"{name}.progress.jsonl"
            expected_bytes = downloaded.stat().st_size
            append_jsonl(
                progress_path,
                {
                    "observed_at": "2026-09-02T00:00:12+00:00",
                    "elapsed_seconds": 0.0,
                    "bytes_downloaded": expected_bytes,
                    "expected_bytes": expected_bytes,
                },
            )
            append_jsonl(
                progress_path,
                {
                    "observed_at": "2026-09-02T00:00:13+00:00",
                    "elapsed_seconds": 1.0,
                    "bytes_downloaded": expected_bytes,
                    "expected_bytes": expected_bytes,
                    "complete": True,
                },
            )
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "kind": "public_github_release_asset_download",
                "status": "pass",
                "asset": api_assets[name],
                "url": expected_public_asset_url(name),
                "effective_url": expected_public_asset_url(name),
                "http_status": 200,
                "started_at": "2026-09-02T00:00:12+00:00",
                "finished_at": "2026-09-02T00:00:13+00:00",
                "duration_seconds": 1.0,
                "timeout_seconds": 60.0,
                "progress_interval_seconds": 10.0,
                "progress_log": str(progress_path.relative_to(root)),
                "download": file_ref(downloaded, relative_to=root),
            }
            write_json_atomic(receipt_path, receipt)
            summary["downloads"][name] = {
                **receipt,
                "receipt": file_ref(receipt_path, relative_to=root),
            }
        write_json_atomic(root / f"backend-{backend}.json", summary)

    gate_args: dict[str, Path] = {}
    for backend in ("metal", "cuda"):
        from release_binary_gate import (
            evidence_ref as binary_evidence_ref,
            prepare_tarball as prepare_binary_tarball,
            selftest_checks as binary_selftest_checks,
            selftest_command_bundle as binary_selftest_command_bundle,
            selftest_serve_evidence as binary_selftest_serve_evidence,
            selftest_progress_ref as binary_selftest_progress_ref,
            validate_gate_data as validate_binary_gate_data,
        )

        lane = f"{backend}-tarball"
        artifact_dir = f"/remote/prepublication/{lane}"
        child_root = root / "prepublication-input" / backend
        child_path = child_root / "gate.json"
        outer_path = root / "prepublication-input" / backend / "gate.manifest.json"
        asset = root / "staged" / BACKEND_SPECS[backend].asset_name
        pass_prefix = "METAL TARBALL GATE PASS: " if backend == "metal" else "CUDA TARBALL GATE PASS: "
        asset_evidence: dict[str, Any] = {}
        unpacked_binary = prepare_binary_tarball(
            VERSION,
            asset.name,
            child_root,
            sha256_file(asset),
            asset,
            asset_evidence,
        )
        receipt_ref = asset_evidence["source_receipt"]
        resolve_saved_ref(receipt_ref, root=child_root, label=f"{backend} fixture source receipt")
        receipt_path = child_root.joinpath(*PurePosixPath(receipt_ref["path"]).parts)
        receipt = read_json_file(receipt_path, f"{backend} fixture source receipt")
        receipt["progress"] = binary_selftest_progress_ref(
            child_root, "asset.local-copy.progress.jsonl", asset.stat().st_size
        )
        write_json_atomic(receipt_path, receipt)
        asset_evidence["source_receipt"] = binary_evidence_ref(child_root, receipt_path)
        commands: dict[str, Any] = {
            "version": binary_selftest_command_bundle(
                child_root,
                "version",
                stdout=f"ferrum {VERSION}\n",
                command=[str(unpacked_binary), "--version"],
            ),
            "cli": binary_selftest_command_bundle(
                child_root,
                "cli",
                command=[str(unpacked_binary), "run", "selftest-model", "--disable-thinking"],
            ),
            "serve": binary_selftest_serve_evidence(
                child_root, "serve", binary_path=str(unpacked_binary)
            ),
        }
        if backend == "cuda":
            commands["ldd"] = binary_selftest_command_bundle(
                child_root,
                "ldd",
                command=["ldd", str(unpacked_binary)],
            )
        child = {
            "schema_version": 2,
            "artifact_type": "ferrum_release_binary_gate",
            "status": "pass",
            "mode": lane,
            "version": VERSION,
            "artifact_dir": artifact_dir,
            "started_at": "2026-09-01T23:59:50+00:00",
            "finished_at": "2026-09-01T23:59:51+00:00",
            "deadline_at": "2026-09-02T00:00:00+00:00",
            "duration_sec": 1.0,
            "rc": 0,
            "pass_line": pass_prefix + artifact_dir,
            "checks": binary_selftest_checks(),
            "evidence": {"asset": asset_evidence, "commands": commands},
        }
        validate_binary_gate_data(child, root=child_root)
        write_json_atomic(child_path, child)
        delegated = [
            sys.executable,
            "scripts/release/release_binary_gate.py",
            lane,
            "--version",
            VERSION,
            "--out",
            artifact_dir,
            "--asset-path",
            f"/staged/{asset.name}",
            "--sha256",
            sha256_file(asset),
        ]
        execution_dir = outer_path.parent
        command_path = execution_dir / "run_gate.child.command.json"
        stdout_path = execution_dir / "run_gate.child.stdout"
        stderr_path = execution_dir / "run_gate.child.stderr"
        write_json_atomic(
            command_path,
            {
                "cmd": delegated,
                "cwd": "/remote/ferrum",
                "timeout_seconds": 120,
                "started_at": "2026-09-01T23:59:50+00:00",
                "finished_at": "2026-09-01T23:59:51+00:00",
                "duration_seconds": 1.0,
                "returncode": 0,
                "env_overrides": {"PYTHONDONTWRITEBYTECODE": "1"},
            },
        )
        stdout_path.write_text(pass_prefix + artifact_dir + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        execution_rows = [
            {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (command_path, stdout_path, stderr_path)
        ]
        from run_gate import standard_g0_artifact_tree

        artifact_tree = standard_g0_artifact_tree(child_root)
        write_json_atomic(
            outer_path,
            {
                "schema_version": SCHEMA_VERSION,
                "lane": lane,
                "status": "pass",
                "child_returncode": 0,
                "git_sha": candidate_sha,
                "dirty_status": {"is_dirty": False, "status_short": []},
                "artifact_dir": artifact_dir,
                "started_at": "2026-09-02T00:00:00+00:00",
                "finished_at": "2026-09-02T00:00:01+00:00",
                "pass_line": f"FERRUM GATE {lane} PASS: {artifact_dir}",
                "child_pass_line": pass_prefix + artifact_dir,
                "child_artifacts": {
                    "child_manifest": {
                        "path": f"{artifact_dir}/gate.json",
                        "sha256": sha256_file(child_path),
                        "size_bytes": child_path.stat().st_size,
                    },
                    "artifact_tree": artifact_tree,
                },
                "delegated_command_line": delegated,
                "child_execution_artifacts": execution_rows,
            },
        )
        gate_args[f"{backend}_tarball_outer"] = outer_path
        gate_args[f"{backend}_tarball_child"] = child_path
    source_gate_args: dict[str, Path] = {}
    source_cli_names = {
        "unit": ("unit_outer", "unit_child"),
        "metal": ("metal_source_outer", "metal_source_child"),
        "cuda_full": ("cuda_full_outer", "cuda_full_child"),
        "cuda_llama_dense": ("cuda_llama_dense_outer", "cuda_llama_dense_child"),
    }
    for label, (outer_name, child_name) in source_cli_names.items():
        pair = prerelease["evidence"]["source_gates"][label]
        source_gate_args[outer_name] = root / pair["outer"]["path"]
        source_gate_args[child_name] = root / pair["child"]["path"]
    return argparse.Namespace(
        metal_summary=root / "backend-metal.json",
        cuda_summary=root / "backend-cuda.json",
        staged_assets_dir=root / "staged",
        release_snapshot=release_path,
        tag_snapshot=tag_path,
        tag_ref_snapshot=tag_ref_path,
        rc_tag_ref_snapshot=root / prerelease["evidence"]["rc_tag_ref_snapshot"]["path"],
        rc_tag_snapshot=root / prerelease["evidence"]["rc_tag_snapshot"]["path"],
        workflow_policy_manifest=root / prerelease["evidence"]["workflow_policy"]["path"],
        native_set_manifest=root / prerelease["evidence"]["native_operator_set"]["path"],
        **source_gate_args,
        **gate_args,
        out=root / "assembled",
    )


def run_self_tests() -> None:
    cases: list[tuple[str, Callable[[], None]]] = []

    def quiet_assemble(args: argparse.Namespace) -> int:
        previous_stdout, previous_stderr = sys.stdout, sys.stderr
        previous_iso_now = globals()["iso_now"]
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            globals()["iso_now"] = lambda: "2026-09-02T01:00:00+00:00"
            return run_assemble(args)
        finally:
            globals()["iso_now"] = previous_iso_now
            sys.stdout, sys.stderr = previous_stdout, previous_stderr

    def release_positive() -> None:
        release, assets = validate_release_snapshot(selftest_release(BACKEND_SPECS["metal"]), BACKEND_SPECS["metal"])
        assert release["prerelease"] is True
        assert set(assets) == set(backend_download_asset_names(BACKEND_SPECS["metal"]))

    cases.append(("release-positive", release_positive))

    def release_not_prerelease() -> None:
        release = selftest_release(BACKEND_SPECS["cuda"])
        release["prerelease"] = False
        expect_gate_error(
            "release prerelease=false",
            lambda: validate_release_snapshot(release, BACKEND_SPECS["cuda"]),
        )

    cases.append(("release-negative-prerelease", release_not_prerelease))

    def release_digest_missing() -> None:
        release = selftest_release(BACKEND_SPECS["metal"])
        release["assets"][0]["digest"] = None
        expect_gate_error(
            "release digest missing",
            lambda: validate_release_snapshot(release, BACKEND_SPECS["metal"]),
        )

    cases.append(("release-negative-digest", release_digest_missing))

    def tar_positive() -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "asset.tar.gz"
            write_tar_fixture(
                archive,
                [("ferrum", b"binary", "file"), ("README.md", b"readme", "file")],
            )
            members = safe_extract_tarball(archive, root / "extract")
            assert any(row["path"] == "ferrum" for row in members)

    cases.append(("safe-tar-positive", tar_positive))

    def tar_traversal() -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "asset.tar.gz"
            write_tar_fixture(
                archive,
                [("ferrum", b"binary", "file"), ("../escape", b"bad", "file")],
            )
            expect_gate_error(
                "tar traversal", lambda: safe_extract_tarball(archive, root / "extract")
            )

    cases.append(("safe-tar-negative-traversal", tar_traversal))

    def tar_symlink() -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "asset.tar.gz"
            write_tar_fixture(
                archive,
                [("ferrum", b"binary", "file"), ("link", b"ferrum", "symlink")],
            )
            expect_gate_error(
                "tar symlink", lambda: safe_extract_tarball(archive, root / "extract")
            )

    cases.append(("safe-tar-negative-symlink", tar_symlink))

    def adjacent_positive() -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            asset_sha, binary_sha = make_selftest_adjacent_bundle(root, BACKEND_SPECS["cuda"])
            receipt = validate_adjacent_bundle(
                root,
                BACKEND_SPECS["cuda"],
                asset_sha256=asset_sha,
                binary_sha256=binary_sha,
            )
            assert receipt["release_candidate"]["release_candidate_sha"] == "1" * 40

    cases.append(("adjacent-bundle-positive", adjacent_positive))

    def adjacent_tamper() -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            asset_sha, binary_sha = make_selftest_adjacent_bundle(root, BACKEND_SPECS["metal"])
            path = root / f"{BACKEND_SPECS['metal'].asset_name}.binary.sha256"
            path.write_text(f"{'0' * 64}  ferrum\n", encoding="utf-8")
            expect_gate_error(
                "tampered binary checksum",
                lambda: validate_adjacent_bundle(
                    root,
                    BACKEND_SPECS["metal"],
                    asset_sha256=asset_sha,
                    binary_sha256=binary_sha,
                ),
            )

    cases.append(("adjacent-bundle-negative", adjacent_tamper))

    def sse_positive() -> None:
        events = [
            {"choices": [{"delta": {"content": "hello"}}], "usage": None},
            {
                "choices": [],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        ]
        body = "".join(f"data: {json.dumps(row)}\n\n" for row in events) + "data: [DONE]\n\n"
        parsed = parse_sse_stream(body.encode())
        assert parsed["done_count"] == 1 and parsed["usage"]["completion_tokens"] == 1

    cases.append(("sse-positive", sse_positive))

    def sse_duplicate_done() -> None:
        body = (
            'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
            'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
            "data: [DONE]\n\ndata: [DONE]\n\n"
        )
        expect_gate_error("duplicate DONE", lambda: parse_sse_stream(body.encode()))

    cases.append(("sse-negative-done", sse_duplicate_done))

    def strict_utf8_negative() -> None:
        expect_gate_error("invalid utf8", lambda: strict_utf8(b"\xff", "fixture"))

    cases.append(("utf8-negative", strict_utf8_negative))

    def forbidden_control_token() -> None:
        expect_gate_error(
            "control token",
            lambda: scan_forbidden_text("fixture", "hello <|im_end|>"),
        )

    cases.append(("forbidden-token-negative", forbidden_control_token))

    def cache_mutation() -> None:
        before = {"exists": False, "entries": []}
        after = {"exists": True, "entries": []}
        expect_gate_error(
            "doctor cache mutation", lambda: assert_cache_unchanged(before, after, "doctor")
        )

    cases.append(("cache-mutation-negative", cache_mutation))

    def cache_rejects_escaping_symlink() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-cache-escape-") as raw:
            root = Path(raw)
            cache = root / "cache"
            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            spec = BACKEND_SPECS["metal"]
            revision = "1" * 40
            for repository, required_files in spec.required_model_files:
                repo = repository_cache_dir(cache, repository)
                (repo / "refs").mkdir(parents=True)
                (repo / "refs" / "main").write_text(revision + "\n", encoding="utf-8")
                snapshot = repo / "snapshots" / revision
                blobs = repo / "blobs"
                snapshot.mkdir(parents=True)
                blobs.mkdir()
                for index, name in enumerate(required_files):
                    target = blobs / f"blob-{index}"
                    target.write_bytes(b"model")
                    destination = snapshot / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.symlink_to(os.path.relpath(target, destination.parent))
            escaped = repository_cache_dir(cache, spec.model_repositories[0]) / "snapshots" / revision / spec.required_model_files[0][1][0]
            escaped.unlink()
            escaped.symlink_to(outside)
            expect_gate_error("escaping model-cache symlink", lambda: model_cache_receipt(cache, spec))

    cases.append(("cache-negative-symlink-escape", cache_rejects_escaping_symlink))

    def environment_sanitization() -> None:
        hostile = {
            "FERRUM_BACKEND": "bogus",
            "HF_ENDPOINT": "https://example.invalid",
            "HF_TOKEN": "secret",
            "TRANSFORMERS_CACHE": "/tmp/other",
            "CUDA_VISIBLE_DEVICES": "99",
            "RUST_LOG": "trace",
        }
        allowed_network = {
            "HTTPS_PROXY": "http://127.0.0.1:3128",
            "SSL_CERT_FILE": "/private/custom-ca.pem",
        }
        changed = {**hostile, **allowed_network}
        previous = {key: os.environ.get(key) for key in changed}
        try:
            os.environ.update(changed)
            environment, receipt = sanitized_child_environment(Path("/tmp/ferrum-v084-cache"), BACKEND_SPECS["metal"])
            assert not (set(hostile) & set(environment))
            assert set(hostile).issubset(receipt["removed_environment_keys"])
            assert (set(hostile) - {"HF_TOKEN"}).issubset(receipt["removed_behavior_keys"])
            assert all(environment[key] == value for key, value in allowed_network.items())
            assert receipt["effective_override_keys"] == ["HF_HOME"]
            network = receipt["network_routing"]
            validate_network_environment_document(network, consumer="ferrum-child-processes")
            rows = {row["key"]: row for row in network["variables"]}
            assert rows["HTTPS_PROXY"]["loopback"] is True
            assert rows["SSL_CERT_FILE"]["custom_ca"] is True
            assert "value" not in rows["HTTPS_PROXY"] and "value" not in rows["SSL_CERT_FILE"]
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    cases.append(("environment-sanitization", environment_sanitization))

    def readme_bilingual_contract() -> None:
        repository_root = Path(__file__).resolve().parents[2]
        english = strict_utf8((repository_root / "README.md").read_bytes(), "README.md")
        chinese = strict_utf8((repository_root / "README_zh.md").read_bytes(), "README_zh.md")
        receipt = validate_readme_contract_texts(english, chinese, BACKEND_SPECS["metal"])
        assert receipt["download_size_announced_before_run"] is True
        expect_gate_error(
            "Chinese reasoning recovery missing",
            lambda: validate_readme_contract_texts(
                english,
                chinese.replace("删除该参数", "保留该参数"),
                BACKEND_SPECS["metal"],
            ),
        )
        expect_gate_error(
            "English global help missing",
            lambda: validate_readme_contract_texts(
                english.replace("ferrum --help", "ferrum help", 1),
                chinese,
                BACKEND_SPECS["metal"],
            ),
        )
        expect_gate_error(
            "Chinese global version missing",
            lambda: validate_readme_contract_texts(
                english,
                chinese.replace("ferrum --version", "ferrum version", 1),
                BACKEND_SPECS["metal"],
            ),
        )

    cases.append(("readme-bilingual-contract", readme_bilingual_contract))

    def child_interrupt_cleanup() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-child-cleanup-") as raw:
            root = Path(raw)
            observations = 0

            def interrupt_observer() -> dict[str, Any]:
                nonlocal observations
                observations += 1
                if observations >= 2:
                    raise KeyboardInterrupt("self-test interrupt")
                return {"observation": observations}

            child = ManagedChild(
                label="interrupt-cleanup",
                command=[sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=root,
                environment=os.environ.copy(),
                environment_receipt={"overrides": {}, "removed_keys": []},
                out=root,
                timeout=5.0,
                progress_interval=0.05,
                progress_observer=interrupt_observer,
            )
            child.start()
            try:
                child.wait()
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("interrupt cleanup self-test did not interrupt")
            assert child.process is not None and child.process.poll() is not None
            command = read_json_file(child.command_path, "interrupt cleanup command")
            assert command["status"] == "interrupted"

    cases.append(("child-interrupt-cleanup", child_interrupt_cleanup))

    def server_self_exit_rejected() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-server-exit-") as raw:
            root = Path(raw)
            child = ManagedChild(
                label="self-exited-server",
                command=[sys.executable, "-c", "pass"],
                cwd=root,
                environment=os.environ.copy(),
                environment_receipt={"overrides": {}},
                out=root,
                timeout=2.0,
                progress_interval=0.05,
            )
            child.start()
            assert child.process is not None
            child.process.wait(timeout=2.0)
            expect_gate_error("server self exit", lambda: require_process_alive_before_cleanup(child))

    cases.append(("server-negative-self-exit", server_self_exit_rejected))

    def make_portable_process_fixture(
        root: Path, *, label: str, status: str, returncode: int | None
    ) -> tuple[Path, Path]:
        binary = root / "extracted" / "ferrum"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"portable binary")
        network_ref = write_network_environment_receipt(
            root,
            label="ferrum-child-processes",
            environment={"HTTPS_PROXY": "http://127.0.0.1:3128"},
            consumer="ferrum-child-processes",
        )
        network_document = read_json_file(
            resolve_saved_ref(network_ref, root=root, label="fixture child network"),
            "fixture child network",
        )
        process_root = root / "processes" / label
        process_root.mkdir(parents=True)
        stdout = process_root / "stdout.log"
        stderr = process_root / "stderr.log"
        progress = process_root / "progress.jsonl"
        stdout.write_text("fixture output\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        progress.write_text(
            json.dumps({"observed_at": iso_now(), "elapsed_seconds": 0.1}) + "\n",
            encoding="utf-8",
        )
        command_path = process_root / "command.json"
        command: dict[str, Any] = {
            "status": status,
            "returncode": returncode,
            "started_at": "2026-09-02T00:00:00+00:00",
            "finished_at": "2026-09-02T00:00:01+00:00",
            "timeout_seconds": 60,
            "duration_seconds": 1.0,
            "environment": {"network_routing": network_document},
            "stdout": file_ref(stdout, relative_to=root),
            "stderr": file_ref(stderr, relative_to=root),
        }
        if status == "terminated":
            command["cleanup_precondition"] = {
                "process_alive": True,
                "observed_at": "2026-09-02T00:00:00+00:00",
            }
        write_json_atomic(command_path, command)
        wrapper_ref = write_portable_process_receipt(
            out=root,
            label=label,
            raw={
                "command": file_ref(command_path, relative_to=root),
                "stdout": file_ref(stdout, relative_to=root),
                "stderr": file_ref(stderr, relative_to=root),
                "progress": file_ref(progress, relative_to=root),
                "stdin": None,
            },
            expected_status=status,
            expected_returncode=returncode,
            extracted_binary=file_ref(binary, relative_to=root),
            network_environment=network_ref,
        )
        return (
            resolve_saved_ref(wrapper_ref, root=root, label=f"{label} wrapper"),
            progress,
        )

    def portable_process_missing_progress_rejected() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-process-progress-") as raw:
            root = Path(raw)
            wrapper, progress = make_portable_process_fixture(
                root, label="readme-run", status="pass", returncode=0
            )
            progress.unlink()
            expect_gate_error(
                "portable process missing progress",
                lambda: validate_portable_process_receipt(
                    wrapper,
                    root=root,
                    label="readme-run",
                    expected_status="pass",
                    expected_returncode=0,
                ),
            )

    cases.append(("portable-process-negative-missing-progress", portable_process_missing_progress_rejected))

    def portable_serve_pass_status_rejected() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-serve-status-") as raw:
            root = Path(raw)
            wrapper, _progress = make_portable_process_fixture(
                root, label="readme-serve", status="terminated", returncode=None
            )
            document = read_json_file(wrapper, "serve wrapper")
            document["status"] = "pass"
            write_json_atomic(wrapper, document)
            expect_gate_error(
                "portable serve pass status",
                lambda: validate_portable_process_receipt(
                    wrapper,
                    root=root,
                    label="readme-serve",
                    expected_status="terminated",
                    expected_returncode=None,
                ),
            )

    cases.append(("portable-serve-negative-pass-status", portable_serve_pass_status_rejected))

    def portable_public_source_dangling_ref_rejected() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-source-dangling-") as raw:
            root = Path(raw)
            name = GOAL_ASSET_NAMES["cpu"]
            download = root / "public-assets" / name
            download.parent.mkdir(parents=True)
            download.write_bytes(b"public bytes")
            progress = root / "provenance" / f"{name}.progress.jsonl"
            append_jsonl(
                progress,
                {
                    "observed_at": "2026-09-02T00:00:00+00:00",
                    "elapsed_seconds": 0.0,
                    "bytes_downloaded": download.stat().st_size,
                    "expected_bytes": download.stat().st_size,
                    "complete": True,
                },
            )
            receipt_path = root / "provenance" / f"{name}.source-receipt.json"
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "ferrum_v084_portable_public_asset_source_receipt",
                "status": "pass",
                "asset_name": name,
                "asset": {
                    "name": name,
                    "size_bytes": download.stat().st_size,
                    "sha256": sha256_file(download),
                },
                "url": expected_public_asset_url(name),
                "effective_url": expected_public_asset_url(name),
                "http_status": 200,
                "started_at": "2026-09-02T00:00:00+00:00",
                "finished_at": "2026-09-02T00:00:01+00:00",
                "duration_seconds": 1.0,
                "timeout_seconds": 60.0,
                "progress_interval_seconds": 10.0,
                "download": file_ref(download, relative_to=root),
                "progress": file_ref(progress, relative_to=root),
                "source_receipt_sha256": "1" * 64,
            }
            write_json_atomic(receipt_path, receipt)
            validate_portable_public_source_receipt(
                receipt_path, root=root, expected_name=name
            )
            receipt["download"] = {
                **receipt["download"],
                "path": f"missing/{name}",
            }
            write_json_atomic(receipt_path, receipt)
            expect_gate_error(
                "portable public source dangling download",
                lambda: validate_portable_public_source_receipt(
                    receipt_path, root=root, expected_name=name
                ),
            )

    cases.append(("portable-public-source-negative-dangling-ref", portable_public_source_dangling_ref_rejected))

    def portable_evidence_graph() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-portable-evidence-") as raw:
            root = Path(raw)
            binary = root / "extracted" / "ferrum"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"portable binary")
            network_ref = write_network_environment_receipt(
                root,
                label="ferrum-child-processes",
                environment={"HTTPS_PROXY": "http://127.0.0.1:3128"},
                consumer="ferrum-child-processes",
            )
            urllib_network_ref = write_network_environment_receipt(
                root,
                label="urllib-public-downloads",
                environment={"SSL_CERT_FILE": "/private/custom-ca.pem"},
                consumer="urllib-public-github-downloads",
            )
            network_document = read_json_file(
                resolve_saved_ref(network_ref, root=root, label="fixture child network"),
                "fixture child network",
            )
            process_refs: dict[str, dict[str, Any]] = {}
            raw_processes: dict[str, dict[str, Any]] = {}
            definitions = {
                "binary_version": ("binary-version", "pass", 0),
                "binary_help": ("binary-help", "pass", 0),
                "doctor": ("doctor-model", "pass", 0),
                "run": ("readme-run", "pass", 0),
                "serve": ("readme-serve", "terminated", None),
            }
            for evidence_label, (process_label, status, returncode) in definitions.items():
                process_root = root / "processes" / process_label
                process_root.mkdir(parents=True)
                stdout = process_root / "stdout.log"
                stderr = process_root / "stderr.log"
                progress = process_root / "progress.jsonl"
                stdout.write_text("objective response\n", encoding="utf-8")
                stderr.write_text("", encoding="utf-8")
                progress.write_text(
                    json.dumps({"observed_at": iso_now(), "elapsed_seconds": 0.1}) + "\n",
                    encoding="utf-8",
                )
                command = process_root / "command.json"
                command_document: dict[str, Any] = {
                    "status": status,
                    "returncode": returncode,
                    "started_at": "2026-09-02T00:00:00+00:00",
                    "finished_at": "2026-09-02T00:00:01+00:00",
                    "timeout_seconds": 60,
                    "duration_seconds": 1.0,
                    "environment": {"network_routing": network_document},
                    "stdout": file_ref(stdout, relative_to=root),
                    "stderr": file_ref(stderr, relative_to=root),
                }
                if status == "terminated":
                    command_document["cleanup_precondition"] = {
                        "process_alive": True,
                        "observed_at": "2026-09-02T00:00:00+00:00",
                    }
                write_json_atomic(command, command_document)
                raw_receipt = {
                    "command": file_ref(command, relative_to=root),
                    "stdout": file_ref(stdout, relative_to=root),
                    "stderr": file_ref(stderr, relative_to=root),
                    "progress": file_ref(progress, relative_to=root),
                    "stdin": None,
                }
                raw_processes[evidence_label] = raw_receipt
                process_refs[evidence_label] = write_portable_process_receipt(
                    out=root,
                    label=process_label,
                    raw=raw_receipt,
                    expected_status=status,
                    expected_returncode=returncode,
                    extracted_binary=file_ref(binary, relative_to=root),
                    network_environment=network_ref,
                )
            model_receipt = root / "cold-cache-model-download.json"
            write_json_atomic(
                model_receipt,
                {
                    "artifact_type": "ferrum_v084_cold_cache_model_download_receipt",
                    "repositories": [
                        {"repository": repository, "revision": "1" * 40, "files": []}
                        for repository in BACKEND_SPECS["metal"].model_repositories
                    ],
                    "run_process": raw_processes["run"],
                    "cache_root": str(root / "model-cache"),
                },
            )
            plain = root / "plain.txt"
            plain.write_text("evidence\n", encoding="utf-8")
            evidence = {label: file_ref(plain, relative_to=root) for label in GOAL_EVIDENCE_KEYS}
            evidence.update(process_refs)
            evidence["download"] = file_ref(model_receipt, relative_to=root)
            summary = root / "goal.json"
            network = {
                "urllib_public_downloads": urllib_network_ref,
                "child_processes": network_ref,
            }
            write_json_atomic(
                summary,
                {
                    "evidence": evidence,
                    "network_environment": network,
                    "cold_cache": {
                        "undocumented_behavior_env": {
                            "behavior_overrides": [],
                            "network_routing_is_behavior_override": False,
                            "network_environment": network,
                        }
                    },
                },
            )
            destination_root = root / "portable"
            relocated_summary = relocate_goal_e2e_summary(
                summary,
                backend="metal",
                destination_root=destination_root,
            )
            relocated = read_json_file(relocated_summary, "portable summary")
            receipt_path = resolve_saved_ref(
                relocated["evidence"]["download"],
                root=destination_root,
                label="portable model download receipt",
            )
            receipt = read_json_file(receipt_path, "portable model download receipt")
            command_path = resolve_saved_ref(
                receipt["run_process"]["command"],
                root=destination_root,
                label="portable run command",
            )
            copied_command = read_json_file(command_path, "portable run command")
            resolve_saved_ref(
                copied_command["stdout"],
                root=destination_root,
                label="portable run stdout",
                require_nonempty=True,
            )
            assert not any("model-cache" in path.as_posix() for path in destination_root.rglob("*"))

    cases.append(("portable-evidence-graph", portable_evidence_graph))

    def aggregate_positive() -> None:
        snapshot = canonical_json_sha256({"fixture": "release"})
        combined = aggregate_documents(
            fake_backend_summary("metal", snapshot),
            fake_backend_summary("cuda", snapshot),
        )
        assert combined["release"]["id"] == 84

    cases.append(("aggregate-positive", aggregate_positive))

    def aggregate_snapshot_mismatch() -> None:
        expect_gate_error(
            "release snapshot mismatch",
            lambda: aggregate_documents(
                fake_backend_summary("metal", "a" * 64),
                fake_backend_summary("cuda", "e" * 64),
            ),
        )

    cases.append(("aggregate-negative-snapshot", aggregate_snapshot_mismatch))

    def aggregate_candidate_mismatch() -> None:
        metal = fake_backend_summary("metal", "a" * 64)
        cuda = fake_backend_summary("cuda", "a" * 64)
        cuda["adjacent_bundle"]["release_candidate"]["release_candidate_sha"] = "2" * 40
        expect_gate_error(
            "release candidate mismatch",
            lambda: aggregate_documents(metal, cuda),
        )

    cases.append(("aggregate-negative-candidate", aggregate_candidate_mismatch))

    def assembly_roundtrip() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-assembly-") as raw:
            root = Path(raw)
            args = make_assembly_selftest_inputs(root)
            assert quiet_assemble(args) == 0
            from v084_release_goal_gate import validate_prerelease_manifest

            manifest = args.out / f"ferrum-{VERSION}-prerelease-manifest.json"
            validated = validate_prerelease_manifest(manifest)
            expected_candidate = read_json_file(
                args.tag_snapshot, "assembly fixture tag"
            )["object"]["sha"]
            assert validated["source"]["git_sha"] == expected_candidate

    cases.append(("assembly-roundtrip", assembly_roundtrip))

    def aggregate_uses_diagnostic_pass_line() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-aggregate-line-") as raw:
            root = Path(raw)
            assembly_args = make_assembly_selftest_inputs(root)
            aggregate_args = argparse.Namespace(
                metal_summary=assembly_args.metal_summary,
                cuda_summary=assembly_args.cuda_summary,
                out=root / "aggregate",
            )
            previous_stdout = sys.stdout
            capture = io.StringIO()
            try:
                sys.stdout = capture
                assert run_aggregate(aggregate_args) == 0
            finally:
                sys.stdout = previous_stdout
            output = capture.getvalue()
            assert f"FERRUM {VERSION} PRERELEASE DOWNLOAD AGGREGATE PASS:" in output
            assert f"FERRUM {VERSION} PRERELEASE DOWNLOAD PASS:" not in output

    cases.append(("aggregate-diagnostic-pass-line", aggregate_uses_diagnostic_pass_line))

    def assembly_rejects_missing_cpu_http_receipt() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-cpu-provenance-") as raw:
            root = Path(raw)
            args = make_assembly_selftest_inputs(root)
            summary = read_json_file(args.metal_summary, "Metal summary")
            cpu_name = GOAL_ASSET_NAMES["cpu"]
            receipt_ref = summary["downloads"][cpu_name]["receipt"]
            receipt_path = resolve_saved_ref(
                receipt_ref,
                root=args.metal_summary.parent,
                label="CPU receipt",
            )
            receipt_path.unlink()
            assert quiet_assemble(args) == 1

    cases.append(("assembly-negative-cpu-http-provenance", assembly_rejects_missing_cpu_http_receipt))

    def assembly_rejects_tag_ref_drift() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-tag-ref-") as raw:
            root = Path(raw)
            args = make_assembly_selftest_inputs(root)
            tag_ref = read_json_file(args.tag_ref_snapshot, "tag ref")
            tag_ref["object"]["sha"] = "f" * 40
            write_json_atomic(args.tag_ref_snapshot, tag_ref)
            assert quiet_assemble(args) == 1

    cases.append(("assembly-negative-tag-ref", assembly_rejects_tag_ref_drift))

    def assembly_rejects_prepublication_child_drift() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-prepub-child-") as raw:
            root = Path(raw)
            args = make_assembly_selftest_inputs(root)
            child = read_json_file(args.cuda_tarball_child, "CUDA prepublication child")
            child["tamper"] = True
            write_json_atomic(args.cuda_tarball_child, child)
            assert quiet_assemble(args) == 1

    cases.append(("assembly-negative-prepublication-child", assembly_rejects_prepublication_child_drift))

    def rebind_prepublication_fixture(args: argparse.Namespace, backend: str) -> None:
        from run_gate import standard_g0_artifact_tree

        child_path = getattr(args, f"{backend}_tarball_child")
        outer_path = getattr(args, f"{backend}_tarball_outer")
        outer = read_json_file(outer_path, f"{backend} outer fixture")
        outer["child_artifacts"]["child_manifest"].update(
            {"sha256": sha256_file(child_path), "size_bytes": child_path.stat().st_size}
        )
        outer["child_artifacts"]["artifact_tree"] = standard_g0_artifact_tree(outer_path.parent)
        write_json_atomic(outer_path, outer)

    def assembly_rejects_prepublication_public_classification() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-prepub-classification-") as raw:
            args = make_assembly_selftest_inputs(Path(raw))
            child = read_json_file(args.metal_tarball_child, "Metal prepublication child")
            child["evidence"]["asset"]["source"] = "public-url"
            child["evidence"]["asset"]["classification"] = "canonical-public-release"
            write_json_atomic(args.metal_tarball_child, child)
            rebind_prepublication_fixture(args, "metal")
            assert quiet_assemble(args) == 1

    cases.append(("assembly-negative-prepublication-classification", assembly_rejects_prepublication_public_classification))

    def assembly_rejects_prepublication_http_receipt() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-prepub-http-") as raw:
            args = make_assembly_selftest_inputs(Path(raw))
            child = read_json_file(args.cuda_tarball_child, "CUDA prepublication child")
            child_root = args.cuda_tarball_child.parent
            source_ref = child["evidence"]["asset"]["source_receipt"]
            source_path = resolve_saved_ref(source_ref, root=child_root, label="CUDA local source receipt")
            source = read_json_file(source_path, "CUDA local source receipt")
            source["http_performed"] = True
            source["http_status"] = 200
            write_json_atomic(source_path, source)
            child["evidence"]["asset"]["source_receipt"] = file_ref(source_path, relative_to=child_root)
            write_json_atomic(args.cuda_tarball_child, child)
            rebind_prepublication_fixture(args, "cuda")
            assert quiet_assemble(args) == 1

    cases.append(("assembly-negative-prepublication-http", assembly_rejects_prepublication_http_receipt))

    def assembly_rejects_prepublication_source_progress() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-prepub-source-progress-") as raw:
            args = make_assembly_selftest_inputs(Path(raw))
            child_root = args.metal_tarball_child.parent
            child = read_json_file(args.metal_tarball_child, "Metal prepublication child")
            asset = child["evidence"]["asset"]
            source_path = resolve_saved_ref(asset["source_receipt"], root=child_root, label="Metal source receipt")
            source = read_json_file(source_path, "Metal source receipt")
            progress_path = resolve_saved_ref(source["progress"], root=child_root, label="Metal source progress")
            rows = [parse_json_bytes(line, "Metal source progress row") for line in progress_path.read_bytes().splitlines()]
            rows[-1]["bytes"] = 0
            progress_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            source["progress"] = file_ref(progress_path, relative_to=child_root)
            write_json_atomic(source_path, source)
            asset["source_receipt"] = file_ref(source_path, relative_to=child_root)
            write_json_atomic(args.metal_tarball_child, child)
            rebind_prepublication_fixture(args, "metal")
            assert quiet_assemble(args) == 1

    cases.append(("assembly-negative-prepublication-source-progress", assembly_rejects_prepublication_source_progress))

    def assembly_rejects_prepublication_extraction_progress() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-prepub-extraction-progress-") as raw:
            args = make_assembly_selftest_inputs(Path(raw))
            child_root = args.cuda_tarball_child.parent
            child = read_json_file(args.cuda_tarball_child, "CUDA prepublication child")
            asset = child["evidence"]["asset"]
            receipt_path = resolve_saved_ref(asset["extraction_receipt"], root=child_root, label="CUDA extraction receipt")
            receipt = read_json_file(receipt_path, "CUDA extraction receipt")
            progress_path = resolve_saved_ref(receipt["progress"], root=child_root, label="CUDA extraction progress")
            rows = [parse_json_bytes(line, "CUDA extraction progress row") for line in progress_path.read_bytes().splitlines()]
            rows[-1]["complete"] = False
            progress_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            receipt["progress"] = file_ref(progress_path, relative_to=child_root)
            write_json_atomic(receipt_path, receipt)
            asset["extraction_receipt"] = file_ref(receipt_path, relative_to=child_root)
            write_json_atomic(args.cuda_tarball_child, child)
            rebind_prepublication_fixture(args, "cuda")
            assert quiet_assemble(args) == 1

    cases.append(("assembly-negative-prepublication-extraction-progress", assembly_rejects_prepublication_extraction_progress))

    def assembly_rejects_public_tamper() -> None:
        with tempfile.TemporaryDirectory(prefix="ferrum-v084-assembly-tamper-") as raw:
            root = Path(raw)
            args = make_assembly_selftest_inputs(root)
            target = root / "files" / GOAL_ASSET_NAMES["cpu"]
            target.write_bytes(target.read_bytes() + b"tamper")
            assert quiet_assemble(args) == 1

    cases.append(("assembly-negative-public-tamper", assembly_rejects_public_tamper))

    for name, case in cases:
        case()
        print(f"self-test {name}: PASS")
    print(f"FERRUM {VERSION} PRERELEASE DOWNLOAD SELFTEST PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Ferrum v0.8.4 public prerelease downloads and README flows"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run focused offline validator self-tests",
    )
    subparsers = parser.add_subparsers(dest="mode")
    for backend, spec in BACKEND_SPECS.items():
        child = subparsers.add_parser(backend, help=f"run the {spec.pass_label} public download gate")
        child.add_argument("--out", required=True, type=Path)
        child.add_argument("--expected-asset-sha256", type=valid_sha256)
        child.add_argument("--port", type=valid_port, default=spec.default_port)
        child.add_argument("--api-timeout-seconds", type=positive_seconds, default=60.0)
        child.add_argument(
            "--asset-download-timeout-seconds", type=positive_seconds, default=900.0
        )
        child.add_argument("--command-timeout-seconds", type=positive_seconds, default=60.0)
        child.add_argument(
            "--model-command-timeout-seconds", type=positive_seconds, default=7200.0
        )
        child.add_argument(
            "--serve-startup-timeout-seconds", type=positive_seconds, default=1800.0
        )
        child.add_argument(
            "--server-total-timeout-seconds", type=positive_seconds, default=2400.0
        )
        child.add_argument("--request-timeout-seconds", type=positive_seconds, default=180.0)
        child.add_argument("--progress-interval-seconds", type=positive_seconds, default=10.0)
    aggregate = subparsers.add_parser("aggregate", help="aggregate Metal and CUDA backend PASS summaries")
    aggregate.add_argument("--metal-summary", required=True, type=Path)
    aggregate.add_argument("--cuda-summary", required=True, type=Path)
    aggregate.add_argument("--out", required=True, type=Path)
    assemble = subparsers.add_parser(
        "assemble",
        help="assemble the portable 21-asset prerelease goal manifest",
    )
    assemble.add_argument("--metal-summary", required=True, type=Path)
    assemble.add_argument("--cuda-summary", required=True, type=Path)
    assemble.add_argument("--staged-assets-dir", required=True, type=Path)
    assemble.add_argument("--release-snapshot", required=True, type=Path)
    assemble.add_argument("--tag-snapshot", required=True, type=Path)
    assemble.add_argument("--tag-ref-snapshot", required=True, type=Path)
    assemble.add_argument("--rc-tag-ref-snapshot", required=True, type=Path)
    assemble.add_argument("--rc-tag-snapshot", required=True, type=Path)
    assemble.add_argument("--unit-outer", required=True, type=Path)
    assemble.add_argument("--unit-child", required=True, type=Path)
    assemble.add_argument("--metal-source-outer", required=True, type=Path)
    assemble.add_argument("--metal-source-child", required=True, type=Path)
    assemble.add_argument("--cuda-full-outer", required=True, type=Path)
    assemble.add_argument("--cuda-full-child", required=True, type=Path)
    assemble.add_argument("--cuda-llama-dense-outer", required=True, type=Path)
    assemble.add_argument("--cuda-llama-dense-child", required=True, type=Path)
    assemble.add_argument("--workflow-policy-manifest", required=True, type=Path)
    assemble.add_argument("--native-set-manifest", required=True, type=Path)
    assemble.add_argument("--metal-tarball-outer", required=True, type=Path)
    assemble.add_argument("--metal-tarball-child", required=True, type=Path)
    assemble.add_argument("--cuda-tarball-outer", required=True, type=Path)
    assemble.add_argument("--cuda-tarball-child", required=True, type=Path)
    assemble.add_argument("--out", required=True, type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        require(args.mode is None, "--self-test cannot be combined with a lane")
        try:
            run_self_tests()
            return 0
        except Exception as error:
            print(f"FERRUM {VERSION} PRERELEASE DOWNLOAD SELFTEST FAIL: {error}", file=sys.stderr)
            return 1
    if args.mode is None:
        parser.error("choose metal, cuda, aggregate, or --self-test")
    if args.mode in BACKEND_SPECS:
        return run_backend(args)
    if args.mode == "aggregate":
        return run_aggregate(args)
    if args.mode == "assemble":
        return run_assemble(args)
    parser.error(f"unknown mode: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
