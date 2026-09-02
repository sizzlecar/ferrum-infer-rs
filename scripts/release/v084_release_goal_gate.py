#!/usr/bin/env python3
"""Fail-closed, read-only final validators for the Ferrum 0.8.4 release.

The validator never calls GitHub, crates.io, Homebrew, or a product binary and
never writes an artifact.  Collectors must first save immutable JSON snapshots
and evidence files.  Every file consumed through a manifest reference uses the
portable schema below and is re-read for both byte size and SHA256::

    {"path": "relative/path", "size_bytes": 123, "sha256": "<64 hex>"}

Reference paths are relative to the JSON file that owns the reference, must
stay below that directory, and may not be symlinks.  The three top-level input
manifest artifact types are:

* ``ferrum_v084_prerelease_manifest``
* ``ferrum_v084_promotion_manifest``
* ``ferrum_v084_final_manifest``

Use ``--self-test`` for hermetic positive and negative fixtures, or one of::

    v084_release_goal_gate.py prerelease --manifest <manifest.json>
    v084_release_goal_gate.py promotion  --manifest <manifest.json>
    v084_release_goal_gate.py final      --manifest <manifest.json>

GitHub release snapshots are the JSON bodies returned by the REST release API.
The tag snapshot is the JSON body returned by ``GET /git/tags/<tag-object-sha>``
and therefore proves that ``v0.8.4`` is annotated and peels to the candidate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlparse


RELEASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = RELEASE_DIR.parent
if str(RELEASE_DIR) not in sys.path:
    sys.path.insert(0, str(RELEASE_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import g0_cuda_llama_dense_gate as dense_gate  # noqa: E402
import m3_validate_runner_artifact as m3_validator  # noqa: E402
import release_binary_gate as binary_gate  # noqa: E402
import v084_crates_io_release as crates_release  # noqa: E402
import v084_workflow_native_gate as workflow_native  # noqa: E402


VERSION = "0.8.4"
TAG = "v0.8.4"
SCHEMA_VERSION = 1
REPOSITORY = "sizzlecar/ferrum-infer-rs"
FINAL_ARTIFACT_DIR = "docs/release/g0/0.8.4"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RC_TAG_RE = re.compile(r"^v0\.8\.4-rc\.[1-9][0-9]*$")

BACKENDS: dict[str, dict[str, Any]] = {
    "cpu": {
        "asset": "ferrum-linux-x86_64.tar.gz",
        "target": "x86_64-unknown-linux-gnu",
        "tar_members": {"ferrum", "LICENSE", "README.md"},
    },
    "metal": {
        "asset": "ferrum-macos-aarch64.tar.gz",
        "target": "aarch64-apple-darwin",
        "tar_members": {"ferrum", "LICENSE", "README.md"},
    },
    "cuda": {
        "asset": "ferrum-linux-x86_64-cuda-sm89.tar.gz",
        "target": "x86_64-unknown-linux-gnu",
        "tar_members": {"ferrum", "LICENSE", "README.md", "CUDA-BUILD.txt"},
    },
}
SIDECAR_SUFFIXES = (
    ".sha256",
    ".binary.sha256",
    ".version.json",
    ".dependency.json",
    ".abi.json",
)
EXPECTED_ASSETS = {
    name
    for spec in BACKENDS.values()
    for name in (spec["asset"], *(spec["asset"] + suffix for suffix in SIDECAR_SUFFIXES))
} | {
    spec["asset"].removesuffix(".tar.gz") + ".dependencies.txt"
    for spec in BACKENDS.values()
}
EXPECTED_CRATES = {
    "ferrum-bench-core",
    "ferrum-cli",
    "ferrum-engine",
    "ferrum-interfaces",
    "ferrum-kernels",
    "ferrum-kv",
    "ferrum-models",
    "ferrum-native-ops",
    "ferrum-native-ops-builder",
    "ferrum-quantization",
    "ferrum-sampler",
    "ferrum-scheduler",
    "ferrum-server",
    "ferrum-testkit",
    "ferrum-tokenizer",
    "ferrum-types",
}
E2E_EVIDENCE_KEYS = {
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


class ValidationError(Exception):
    """A saved release claim is incomplete, mutable, or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_object(value: Any, where: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{where} must be an object")
    return value


def exact_fields(value: Any, fields: set[str], where: str) -> dict[str, Any]:
    obj = require_object(value, where)
    require(set(obj) == fields, f"{where} fields differ: expected {sorted(fields)}, got {sorted(obj)}")
    return obj


def nonempty(value: Any, where: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{where} must be a non-empty string")
    return value


def require_sha256(value: Any, where: str) -> str:
    text = nonempty(value, where)
    require(SHA256_RE.fullmatch(text) is not None, f"{where} must be lowercase SHA256")
    return text


def require_git_sha(value: Any, where: str) -> str:
    text = nonempty(value, where)
    require(GIT_SHA_RE.fullmatch(text) is not None, f"{where} must be a full lowercase git SHA")
    return text


def normalized_positive_int(value: Any, where: str) -> int:
    require(
        (type(value) is int and value > 0)
        or (isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) is not None),
        f"{where} must be a positive integer",
    )
    return int(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, where: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON for {where}: {path}: {exc}") from exc


def resolve_ref(raw: Any, *, root: Path, where: str) -> tuple[dict[str, Any], Path]:
    ref = exact_fields(raw, {"path", "size_bytes", "sha256"}, where)
    rel_text = nonempty(ref["path"], f"{where}.path")
    pure = PurePosixPath(rel_text)
    require(not pure.is_absolute(), f"{where}.path must be relative")
    require("\\" not in rel_text and ".." not in pure.parts, f"{where}.path may not escape its manifest directory")
    candidate = root.joinpath(*pure.parts)
    require(not candidate.is_symlink(), f"{where} may not reference a symlink")
    root_real = root.resolve()
    path = candidate.resolve()
    require(path.is_relative_to(root_real), f"{where} escapes its manifest directory")
    require(path.is_file() and not path.is_symlink(), f"{where} is not a regular file: {path}")
    size = ref["size_bytes"]
    require(type(size) is int and size >= 0, f"{where}.size_bytes must be a non-negative integer")
    expected_sha = require_sha256(ref["sha256"], f"{where}.sha256")
    require(path.stat().st_size == size, f"{where}.size_bytes changed")
    require(file_sha256(path) == expected_sha, f"{where}.sha256 changed")
    return copy.deepcopy(ref), path


def read_json_ref(raw: Any, *, root: Path, where: str) -> tuple[dict[str, Any], Path, Any]:
    ref, path = resolve_ref(raw, root=root, where=where)
    return ref, path, read_json(path, where)


def validate_source(raw: Any, where: str = "source") -> dict[str, Any]:
    source = exact_fields(raw, {"git_sha", "dirty"}, where)
    require_git_sha(source["git_sha"], f"{where}.git_sha")
    require(source["dirty"] is False, f"{where}.dirty must be false")
    return copy.deepcopy(source)


def validate_tag_snapshot(raw: Any, *, candidate_sha: str, where: str, expected_tag: str = TAG) -> dict[str, Any]:
    tag = require_object(raw, where)
    require(tag.get("tag") == expected_tag, f"{where}.tag differs")
    require_git_sha(tag.get("sha"), f"{where}.sha")
    peeled = require_object(tag.get("object"), f"{where}.object")
    require(peeled.get("type") == "commit", f"{where} does not peel to a commit")
    require(peeled.get("sha") == candidate_sha, f"{where} peels to a different candidate")
    return {"tag": expected_tag, "tag_object_sha": tag["sha"], "commit_sha": candidate_sha}


def validate_tag_chain(
    ref_raw: Any,
    tag_raw: Any,
    *,
    candidate_sha: str,
    where: str,
    expected_ref: str = f"refs/tags/{TAG}",
    expected_tag: str = TAG,
) -> dict[str, Any]:
    ref = require_object(ref_raw, f"{where} ref")
    require(ref.get("ref") == expected_ref, f"{where} ref name differs")
    ref_object = require_object(ref.get("object"), f"{where} ref.object")
    require(ref_object.get("type") == "tag", f"{where} is not an annotated tag ref")
    ref_sha = require_git_sha(ref_object.get("sha"), f"{where} ref tag-object SHA")
    tag = validate_tag_snapshot(
        tag_raw,
        candidate_sha=candidate_sha,
        where=f"{where} tag object",
        expected_tag=expected_tag,
    )
    require(
        ref_sha == tag["tag_object_sha"],
        f"{where} ref does not point at the saved tag object",
    )
    return tag


def github_asset_rows(release: Any, *, prerelease: bool, where: str) -> list[dict[str, Any]]:
    obj = require_object(release, where)
    require(type(obj.get("id")) is int and obj["id"] > 0, f"{where}.id differs")
    require(obj.get("tag_name") == TAG, f"{where}.tag_name differs")
    require(obj.get("draft") is False, f"{where}.draft must be false")
    require(obj.get("prerelease") is prerelease, f"{where}.prerelease differs")
    raw_assets = obj.get("assets")
    require(isinstance(raw_assets, list) and len(raw_assets) == len(EXPECTED_ASSETS), f"{where} must contain exactly {len(EXPECTED_ASSETS)} assets")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    ids: set[int] = set()
    for index, raw in enumerate(raw_assets):
        asset = require_object(raw, f"{where}.assets[{index}]")
        name = nonempty(asset.get("name"), f"{where}.assets[{index}].name")
        asset_id = asset.get("id")
        size = asset.get("size")
        digest = nonempty(asset.get("digest"), f"{where}.assets[{index}].digest")
        require(type(asset_id) is int and asset_id > 0 and asset_id not in ids, f"{where} asset id differs")
        require(type(size) is int and size > 0, f"{where} asset {name} size differs")
        require(digest.startswith("sha256:") and SHA256_RE.fullmatch(digest[7:]) is not None, f"{where} asset {name} digest differs")
        require(name not in names, f"{where} has duplicate asset name {name}")
        download_url = nonempty(asset.get("browser_download_url"), f"{where} asset {name} URL")
        parsed = urlparse(download_url)
        expected_path = f"/{REPOSITORY}/releases/download/{TAG}/{quote(name)}"
        require(parsed.scheme == "https" and parsed.netloc == "github.com" and parsed.path == expected_path and not parsed.query and not parsed.fragment, f"{where} asset {name} is not the canonical public URL")
        ids.add(asset_id)
        names.add(name)
        rows.append({"id": asset_id, "name": name, "size": size, "digest": digest})
    require(names == EXPECTED_ASSETS, f"{where} asset-name denominator differs")
    return sorted(rows, key=lambda row: (row["name"], row["id"]))


def asset_set_sha256(rows: list[dict[str, Any]]) -> str:
    return canonical_sha256(rows)


def parse_checksum(path: Path, *, expected_name: str, where: str) -> str:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read {where}: {exc}") from exc
    require(len(lines) == 1, f"{where} must contain exactly one non-empty line")
    parts = lines[0].split()
    require(len(parts) == 2 and parts[1].lstrip("*") == expected_name, f"{where} filename differs")
    return require_sha256(parts[0], f"{where} digest")


def tar_binary_sha(path: Path, *, expected_members: set[str], where: str) -> str:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            require(len(names) == len(set(names)) and set(names) == expected_members, f"{where} members differ")
            require(all(member.isfile() for member in members), f"{where} contains a non-regular member")
            binary_member = next(member for member in members if member.name == "ferrum")
            stream = archive.extractfile(binary_member)
            require(stream is not None, f"{where} ferrum member cannot be read")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()
    except (tarfile.TarError, OSError) as exc:
        raise ValidationError(f"invalid tarball for {where}: {exc}") from exc


def require_common_sidecar(
    raw: Any,
    *,
    asset_name: str,
    asset_sha: str,
    binary_sha: str,
    candidate_sha: str,
    rc_tag: str,
    where: str,
) -> dict[str, Any]:
    obj = require_object(raw, where)
    require(obj.get("schema_version") == 1, f"{where}.schema_version differs")
    require(obj.get("asset_name") == asset_name and obj.get("asset_sha256") == asset_sha, f"{where} asset binding differs")
    require(obj.get("binary_name") == "ferrum" and obj.get("binary_sha256") == binary_sha, f"{where} binary binding differs")
    require(obj.get("release_candidate_sha") == candidate_sha and obj.get("release_candidate_tag") == rc_tag, f"{where} candidate binding differs")
    nonempty(obj.get("staging_label"), f"{where}.staging_label")
    nonempty(str(obj.get("workflow_run_id", "")), f"{where}.workflow_run_id")
    nonempty(str(obj.get("workflow_run_attempt", "")), f"{where}.workflow_run_attempt")
    return obj


def validate_downloaded_backend(
    backend: str,
    *,
    paths: dict[str, Path],
    candidate_sha: str,
    rc_tag: str,
) -> dict[str, str]:
    spec = BACKENDS[backend]
    asset_name = spec["asset"]
    tarball = paths[asset_name]
    asset_sha = file_sha256(tarball)
    checksum_sha = parse_checksum(paths[asset_name + ".sha256"], expected_name=asset_name, where=f"{backend} tarball checksum")
    require(checksum_sha == asset_sha, f"{backend} adjacent tarball checksum differs")
    binary_sha = parse_checksum(paths[asset_name + ".binary.sha256"], expected_name="ferrum", where=f"{backend} binary checksum")
    require(tar_binary_sha(tarball, expected_members=spec["tar_members"], where=f"{backend} tarball") == binary_sha, f"{backend} extracted binary checksum differs")

    version = read_json(paths[asset_name + ".version.json"], f"{backend} version sidecar")
    common = require_common_sidecar(version, asset_name=asset_name, asset_sha=asset_sha, binary_sha=binary_sha, candidate_sha=candidate_sha, rc_tag=rc_tag, where=f"{backend} version sidecar")
    require(version.get("version") == VERSION, f"{backend} version sidecar version differs")

    dependency = read_json(paths[asset_name + ".dependency.json"], f"{backend} dependency sidecar")
    require_common_sidecar(dependency, asset_name=asset_name, asset_sha=asset_sha, binary_sha=binary_sha, candidate_sha=candidate_sha, rc_tag=rc_tag, where=f"{backend} dependency sidecar")
    expected_audit = asset_name.removesuffix(".tar.gz") + ".dependencies.txt"
    audit_path = paths[expected_audit]
    try:
        audit_text = audit_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read {backend} dependency audit: {exc}") from exc
    audit_sha = file_sha256(audit_path)
    forbidden_match = re.search(r"python|torch|vllm", audit_text, re.IGNORECASE)
    require(forbidden_match is None, f"{backend} dependency audit contains forbidden runtime linkage: {forbidden_match.group(0) if forbidden_match else ''}")
    require(dependency.get("audit_file") == expected_audit, f"{backend} dependency audit filename differs")
    require(dependency.get("audit_sha256") == audit_sha, f"{backend} dependency audit SHA differs")
    require(dependency.get("forbidden_runtime_linkage") == ["python", "torch", "vllm"] and dependency.get("forbidden_runtime_linkage_found") is False, f"{backend} dependency audit did not fail-close")

    abi = read_json(paths[asset_name + ".abi.json"], f"{backend} ABI sidecar")
    require_common_sidecar(abi, asset_name=asset_name, asset_sha=asset_sha, binary_sha=binary_sha, candidate_sha=candidate_sha, rc_tag=rc_tag, where=f"{backend} ABI sidecar")
    require(abi.get("target_triple") == spec["target"] and abi.get("backend") == backend, f"{backend} ABI identity differs")
    require(abi.get("dependency_audit_sha256") == audit_sha, f"{backend} ABI/dependency audit binding differs")
    if backend == "cuda":
        require(abi.get("cuda_compute_capability") == "89", "CUDA compute capability differs")
        require(abi.get("cuda_toolkit_image") == "nvidia/cuda:12.4.0-devel-ubuntu22.04", "CUDA toolkit image differs")
        require(abi.get("cargo_features") == ["cuda", "vllm-moe-marlin", "vllm-paged-attn-v2"], "CUDA Cargo feature set differs")
    return {
        "asset_sha256": asset_sha,
        "binary_sha256": binary_sha,
        "workflow_run_id": normalized_positive_int(
            common["workflow_run_id"], f"{backend} workflow_run_id"
        ),
        "workflow_run_attempt": normalized_positive_int(
            common["workflow_run_attempt"], f"{backend} workflow_run_attempt"
        ),
        "staging_label": str(common["staging_label"]),
    }


def parse_time(value: Any, where: str) -> datetime:
    text = nonempty(value, where)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{where} is not an ISO-8601 timestamp") from exc
    require(parsed.tzinfo is not None, f"{where} must include a timezone")
    return parsed


def read_jsonl(path: Path, where: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for ordinal, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            rows.append(require_object(value, f"{where} line {ordinal}"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {where}: {exc}") from exc
    require(rows, f"{where} is empty")
    return rows


def validate_progress_jsonl(
    path: Path,
    *,
    where: str,
    expected_bytes: int | None = None,
) -> None:
    rows = read_jsonl(path, where)
    elapsed = -1.0
    downloaded = -1
    complete_count = 0
    for index, row in enumerate(rows):
        current_elapsed = row.get("elapsed_seconds")
        require(
            isinstance(current_elapsed, (int, float))
            and not isinstance(current_elapsed, bool)
            and current_elapsed >= elapsed >= -1,
            f"{where} elapsed time is not monotonic",
        )
        elapsed = float(current_elapsed)
        if "bytes_downloaded" in row:
            current_bytes = row.get("bytes_downloaded")
            require(type(current_bytes) is int and current_bytes >= downloaded, f"{where} byte progress is not monotonic")
            downloaded = current_bytes
            if expected_bytes is not None:
                require(row.get("expected_bytes") == expected_bytes and current_bytes <= expected_bytes, f"{where} expected byte denominator differs")
        if row.get("complete") is True:
            complete_count += 1
            require(index == len(rows) - 1, f"{where} completion is not the final sample")
    if expected_bytes is not None:
        require(complete_count == 1 and downloaded == expected_bytes, f"{where} final download completion differs")


def validate_public_download_provenance(
    entry: dict[str, Any],
    *,
    root: Path,
    name: str,
    published: dict[str, Any],
    downloaded_ref: dict[str, Any],
) -> datetime:
    receipt_ref, receipt_path, receipt = read_json_ref(entry["receipt"], root=root, where=f"public download receipt {name}")
    wrapper = exact_fields(
        receipt,
        {
            "schema_version", "artifact_type", "status", "backend_lane", "asset_name",
            "url", "effective_url", "http_status", "started_at", "finished_at",
            "duration_seconds", "timeout_seconds", "download", "progress", "source_receipt",
        },
        f"public download provenance {name}",
    )
    require(
        wrapper["schema_version"] == 1
        and wrapper["artifact_type"] == "ferrum_v084_public_asset_download_provenance"
        and wrapper["status"] == "pass"
        and wrapper["backend_lane"] in {"metal", "cuda"}
        and wrapper["asset_name"] == name
        and wrapper["url"] == entry["url"]
        and wrapper["http_status"] == 200,
        f"public download provenance identity differs: {name}",
    )
    parsed_effective = urlparse(nonempty(wrapper["effective_url"], f"public effective URL {name}"))
    require(parsed_effective.scheme == "https" and parsed_effective.hostname is not None and (parsed_effective.hostname == "github.com" or parsed_effective.hostname.endswith(".githubusercontent.com")), f"public effective URL host differs: {name}")
    started = parse_time(wrapper["started_at"], f"public download {name} started_at")
    finished = parse_time(wrapper["finished_at"], f"public download {name} finished_at")
    duration = wrapper["duration_seconds"]
    timeout = wrapper["timeout_seconds"]
    require(
        started <= finished
        and isinstance(duration, (int, float)) and not isinstance(duration, bool) and 0 <= duration <= (finished - started).total_seconds() + 5
        and isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and duration <= timeout,
        f"public download timing/deadline differs: {name}",
    )
    wrapper_download, wrapper_download_path = resolve_ref(wrapper["download"], root=root, where=f"public wrapper download {name}")
    require(wrapper_download == downloaded_ref, f"public wrapper binds different downloaded bytes: {name}")
    progress_ref, progress_path = resolve_ref(wrapper["progress"], root=root, where=f"public wrapper progress {name}")
    entry_progress, entry_progress_path = resolve_ref(entry["progress"], root=root, where=f"public entry progress {name}")
    require(progress_ref == entry_progress and progress_path == entry_progress_path, f"public progress refs differ: {name}")
    validate_progress_jsonl(progress_path, where=f"public progress {name}", expected_bytes=published["size"])
    _, source_path, source = read_json_ref(wrapper["source_receipt"], root=root, where=f"public source receipt {name}")
    source_obj = exact_fields(
        source,
        {
            "schema_version", "artifact_type", "status", "asset_name", "asset",
            "url", "effective_url", "http_status", "started_at", "finished_at",
            "duration_seconds", "timeout_seconds", "progress_interval_seconds",
            "download", "progress", "source_receipt_sha256",
        },
        f"public source receipt {name}",
    )
    require(
        source_obj.get("schema_version") == 1
        and source_obj.get("artifact_type") == "ferrum_v084_portable_public_asset_source_receipt"
        and source_obj.get("status") == "pass"
        and source_obj.get("asset_name") == name
        and source_obj.get("url") == entry["url"]
        and source_obj.get("effective_url") == wrapper["effective_url"]
        and source_obj.get("http_status") == 200
        and source_obj.get("started_at") == wrapper["started_at"]
        and source_obj.get("finished_at") == wrapper["finished_at"]
        and source_obj.get("duration_seconds") == wrapper["duration_seconds"]
        and source_obj.get("timeout_seconds") == wrapper["timeout_seconds"]
        and isinstance(source_obj.get("progress_interval_seconds"), (int, float))
        and source_obj["progress_interval_seconds"] > 0,
        f"public source receipt identity/timing differs: {name}",
    )
    asset = require_object(source_obj.get("asset"), f"public source receipt asset {name}")
    require(
        asset.get("name") == name
        and asset.get("size_bytes") == published["size"]
        and asset.get("sha256") == downloaded_ref["sha256"]
        and asset.get("browser_download_url") == entry["url"],
        f"public source receipt GitHub asset identity differs: {name}",
    )
    source_download, source_download_path = resolve_ref(source_obj.get("download"), root=root, where=f"public source receipt download {name}")
    source_progress, source_progress_path = resolve_ref(source_obj.get("progress"), root=root, where=f"public source receipt progress {name}")
    require(
        source_download == downloaded_ref
        and source_download_path == wrapper_download_path
        and source_progress == progress_ref
        and source_progress_path == progress_path
        and require_sha256(source_obj.get("source_receipt_sha256"), f"public source original receipt SHA {name}"),
        f"public source receipt download/progress closure differs: {name}",
    )
    require(receipt_ref["size_bytes"] > 0 and source_path != receipt_path, f"public provenance closure differs: {name}")
    return finished


NETWORK_ENV_KEYS = {
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy",
    "NO_PROXY", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
}
CUSTOM_CA_KEYS = {"SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"}


def validate_network_environment_ref(
    raw_ref: Any, *, root: Path, consumer: str, where: str
) -> tuple[dict[str, Any], Path]:
    _, path, document = read_json_ref(raw_ref, root=root, where=where)
    receipt = exact_fields(
        document,
        {"schema_version", "artifact_type", "consumer", "secret_values_recorded", "variables"},
        where,
    )
    require(
        receipt["schema_version"] == 1
        and receipt["artifact_type"] == "ferrum_v084_sanitized_network_environment_receipt"
        and receipt["consumer"] == consumer
        and receipt["secret_values_recorded"] is False,
        f"{where} identity differs",
    )
    variables = receipt["variables"]
    require(isinstance(variables, list), f"{where} variables differ")
    seen: set[str] = set()
    for index, raw in enumerate(variables):
        row = exact_fields(raw, {"key", "value_sha256", "loopback", "custom_ca"}, f"{where} variable {index}")
        key = nonempty(row["key"], f"{where} variable key")
        require(
            key in NETWORK_ENV_KEYS
            and key not in seen
            and require_sha256(row["value_sha256"], f"{where} {key} value hash")
            and type(row["loopback"]) is bool
            and type(row["custom_ca"]) is bool
            and row["custom_ca"] == (key in CUSTOM_CA_KEYS),
            f"{where} variable differs",
        )
        seen.add(key)
    return receipt, path


def validate_process_progress(path: Path, *, where: str) -> None:
    rows = read_jsonl(path, where)
    require(rows, f"{where} is empty")
    elapsed = -1.0
    for index, raw in enumerate(rows):
        row = require_object(raw, f"{where} row {index}")
        value = row.get("elapsed_seconds")
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and value >= elapsed, f"{where} elapsed time is not monotonic")
        elapsed = float(value)


def validate_e2e_command(
    raw_ref: Any,
    *,
    root: Path,
    backend: str,
    label: str,
    alias: str,
    package_binary_sha256: str,
    recorded_artifact_dir: str,
    network_environment_ref: dict[str, Any],
) -> dict[str, Any]:
    _, _, wrapper_raw = read_json_ref(raw_ref, root=root, where=f"{backend} E2E {label} process wrapper")
    expected_label = {
        "binary_version": "binary-version", "binary_help": "binary-help",
        "doctor": "doctor-model", "run": "readme-run", "serve": "readme-serve",
    }[label]
    wrapper = exact_fields(
        wrapper_raw,
        {"schema_version", "artifact_type", "label", "status", "returncode", "command", "stdout", "stderr", "progress", "stdin", "extracted_binary", "network_environment"},
        f"{backend} E2E {label} process wrapper",
    )
    expected_status = "terminated" if label == "serve" else "pass"
    require(
        wrapper["schema_version"] == 1
        and wrapper["artifact_type"] == "ferrum_v084_portable_process_receipt"
        and wrapper["label"] == expected_label
        and wrapper["status"] == expected_status
        and (wrapper["returncode"] is None if label == "serve" else wrapper["returncode"] == 0)
        and wrapper["network_environment"] == network_environment_ref,
        f"{backend} E2E {label} wrapper identity differs",
    )
    _, extracted_binary = resolve_ref(wrapper["extracted_binary"], root=root, where=f"{backend} E2E {label} extracted binary")
    require(extracted_binary.name == "ferrum" and file_sha256(extracted_binary) == package_binary_sha256, f"{backend} E2E {label} extracted binary differs")
    _, command_path, command = read_json_ref(wrapper["command"], root=root, where=f"{backend} E2E {label} command")
    document = require_object(command, f"{backend} E2E {label} command")
    require(document.get("stdout") == wrapper["stdout"] and document.get("stderr") == wrapper["stderr"], f"{backend} E2E {label} wrapper/output refs differ")
    _, progress_path = resolve_ref(wrapper["progress"], root=root, where=f"{backend} E2E {label} progress")
    validate_process_progress(progress_path, where=f"{backend} E2E {label} progress")
    network_document, _ = validate_network_environment_ref(
        wrapper["network_environment"], root=root, consumer="ferrum-child-processes",
        where=f"{backend} E2E {label} network environment",
    )
    environment = require_object(document.get("environment"), f"{backend} E2E {label} environment")
    require(environment.get("network_routing") == network_document, f"{backend} E2E {label} command/network receipt differs")
    argv = document.get("command")
    require(isinstance(argv, list) and all(isinstance(item, str) and item for item in argv), f"{backend} E2E {label} argv differs")
    expected = {
        "binary_version": "--version",
        "binary_help": "--help",
        "doctor": "doctor",
        "run": "run",
        "serve": "serve",
    }[label]
    require(expected in argv, f"{backend} E2E {label} argv differs")
    recorded_root = Path(recorded_artifact_dir)
    recorded_binary = Path(argv[0])
    require(recorded_root.is_absolute() and recorded_binary.is_absolute(), f"{backend} E2E {label} recorded binary path differs")
    try:
        binary_relative = recorded_binary.relative_to(recorded_root)
    except ValueError as exc:
        raise ValidationError(f"{backend} E2E {label} binary escapes recorded artifact root") from exc
    require((root / binary_relative).resolve() == extracted_binary.resolve(), f"{backend} E2E {label} argv[0]/extracted binary mapping differs")
    if label in {"doctor", "run"}:
        position = argv.index(expected)
        require(position + 1 < len(argv) and argv[position + 1] == alias, f"{backend} E2E {label} alias differs")
    if label == "serve":
        require(command_option(argv, "--model", f"{backend} E2E serve") == alias, f"{backend} E2E serve alias differs")
    if label in {"run", "serve"}:
        require(argv.count("--disable-thinking") == 1, f"{backend} E2E {label} disable-thinking differs")
    status = document.get("status")
    returncode = document.get("returncode")
    if label == "serve":
        cleanup = require_object(document.get("cleanup_precondition"), f"{backend} E2E serve cleanup precondition")
        require(status == "terminated" and returncode is None and cleanup.get("process_alive") is True and isinstance(cleanup.get("observed_at"), str), f"{backend} E2E serve terminal/cleanup status differs")
    else:
        require(status == "pass" and returncode == 0, f"{backend} E2E {label} return code differs")
    started = parse_time(document.get("started_at"), f"{backend} E2E {label} started_at")
    finished = parse_time(document.get("finished_at"), f"{backend} E2E {label} finished_at")
    timeout = document.get("timeout_seconds")
    duration = document.get("duration_seconds")
    require(started <= finished and isinstance(timeout, (int, float)) and timeout > 0 and isinstance(duration, (int, float)) and 0 <= duration <= timeout, f"{backend} E2E {label} timing differs")
    for stream in ("stdout", "stderr"):
        _, stream_path = resolve_ref(document.get(stream), root=root, where=f"{backend} E2E {label} {stream}")
        try:
            text = stream_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValidationError(f"cannot read {backend} E2E {label} {stream}: {exc}") from exc
        lowered = text.lower()
        require(not any(pattern in lowered for pattern in ("panic", "oom", "invalid utf-8", "<unk>", "[pad]")), f"{backend} E2E {label} log contains forbidden output")
    stdout_path = resolve_ref(wrapper["stdout"], root=root, where=f"{backend} E2E {label} stdout")[1]
    stdout_text = stdout_path.read_text(encoding="utf-8")
    if label == "binary_version":
        require(re.search(rf"(?m)^ferrum\s+{re.escape(VERSION)}(?:\s|$)", stdout_text) is not None, f"{backend} E2E version stdout differs")
    elif label == "binary_help":
        lowered = stdout_text.lower()
        require("ferrum" in lowered and ("usage" in lowered or "commands" in lowered), f"{backend} E2E help stdout differs")
    elif label == "run":
        assistant: list[str] = []
        for line in stdout_text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("event") == "assistant" and isinstance(row.get("content"), str):
                assistant.append(row["content"])
        objective = "\n".join(assistant) if assistant else re.sub(r"<think>.*?</think>", "", stdout_text, flags=re.S)
        require(bool(objective.strip()), f"{backend} E2E run objective stdout is empty")
    return {"finished_at": finished, "wrapper": wrapper, "command": document}


def validate_e2e_http_exchange(
    raw_ref: Any,
    *,
    root: Path,
    backend: str,
    label: str,
) -> datetime:
    _, _, exchange = read_json_ref(raw_ref, root=root, where=f"{backend} E2E {label} exchange")
    row = require_object(exchange, f"{backend} E2E {label} exchange")
    require(
        row.get("schema_version") == 1
        and row.get("kind") == "bounded_local_http_exchange"
        and row.get("status") == 200
        and str(row.get("url", "")).startswith("http://127.0.0.1:")
        and isinstance(row.get("timeout_seconds"), (int, float))
        and row["timeout_seconds"] > 0,
        f"{backend} E2E {label} HTTP exchange identity differs",
    )
    started = parse_time(row.get("started_at"), f"{backend} E2E {label} HTTP started_at")
    finished = parse_time(row.get("finished_at"), f"{backend} E2E {label} HTTP finished_at")
    require(started <= finished and 0 <= row.get("duration_seconds", -1) <= row["timeout_seconds"], f"{backend} E2E {label} HTTP timing differs")
    _, response_path = resolve_ref(row.get("response"), root=root, where=f"{backend} E2E {label} response")
    response = response_path.read_text(encoding="utf-8")
    if label == "models":
        payload = json.loads(response)
        require(payload.get("object") == "list" and sum(isinstance(item, dict) and item.get("id") == "ferrum" for item in payload.get("data", [])) == 1, f"{backend} /v1/models response differs")
    elif label == "chat":
        payload = json.loads(response)
        choices = payload.get("choices")
        require(isinstance(choices, list) and choices and isinstance(choices[0].get("message", {}).get("content"), str) and choices[0]["message"]["content"].strip(), f"{backend} chat response differs")
    else:
        events = [line[5:].strip() for line in response.splitlines() if line.startswith("data:")]
        require(events.count("[DONE]") == 1 and events[-1:] == ["[DONE]"], f"{backend} stream [DONE] differs")
        usage = []
        content = []
        for event in events[:-1]:
            payload = json.loads(event)
            if isinstance(payload.get("usage"), dict):
                usage.append(payload["usage"])
            for choice in payload.get("choices", []):
                value = choice.get("delta", {}).get("content")
                if isinstance(value, str):
                    content.append(value)
        require(len(usage) == 1 and type(usage[0].get("completion_tokens")) is int and usage[0]["completion_tokens"] > 0 and "".join(content).strip(), f"{backend} stream usage/content differs")
    return finished


def validate_e2e_model_download(raw_ref: Any, *, root: Path, backend: str, alias: str) -> dict[str, Any]:
    _, _, receipt = read_json_ref(raw_ref, root=root, where=f"{backend} E2E model download")
    row = require_object(receipt, f"{backend} E2E model download")
    require(
        row.get("schema_version") == 1
        and row.get("artifact_type") == "ferrum_v084_cold_cache_model_download_receipt"
        and row.get("status") == "pass"
        and row.get("backend") == backend
        and row.get("model_alias") == alias
        and row.get("source") == "https://huggingface.co"
        and row.get("fresh_cache") is True
        and row.get("download_complete") is True,
        f"{backend} E2E model download identity differs",
    )
    repositories = row.get("repositories")
    require(isinstance(repositories, list) and repositories, f"{backend} E2E model repositories differ")
    for repo in repositories:
        repo_obj = require_object(repo, f"{backend} E2E model repository")
        require_git_sha(repo_obj.get("revision"), f"{backend} E2E model revision")
        files = repo_obj.get("files")
        require(isinstance(files, list) and files and all(isinstance(item, dict) and isinstance(item.get("name"), str) and type(item.get("size_bytes")) is int and item["size_bytes"] > 0 for item in files), f"{backend} E2E model file inventory differs")
        require(repo_obj.get("files_metadata_sha256") == canonical_sha256(files), f"{backend} E2E model file inventory digest differs")
    execution = exact_fields(row.get("execution"), {"started_at", "finished_at", "timeout_seconds", "progress_signal"}, f"{backend} E2E model download execution")
    started = parse_time(execution["started_at"], f"{backend} E2E model download started_at")
    finished = parse_time(execution["finished_at"], f"{backend} E2E model download finished_at")
    require(started <= finished and isinstance(execution["timeout_seconds"], (int, float)) and execution["timeout_seconds"] > 0 and nonempty(execution["progress_signal"], f"{backend} E2E model progress"), f"{backend} E2E model download timing differs")
    run_process = require_object(row.get("run_process"), f"{backend} E2E model run process")
    exact_fields(run_process, {"command", "stdout", "stderr", "progress", "stdin"}, f"{backend} E2E model run process")
    for key in ("command", "stdout", "stderr", "progress"):
        resolve_ref(run_process[key], root=root, where=f"{backend} E2E model run {key}")
    return {"finished_at": finished, "run_process": run_process}


def validate_readme_e2e(
    raw: Any,
    *,
    path: Path,
    backend: str,
    source_sha: str,
    package: dict[str, str],
) -> dict[str, Any]:
    fields = {
        "schema_version", "artifact_type", "status", "version", "backend",
        "source_git_sha", "asset_name", "asset_sha256", "binary_sha256",
        "model", "cold_cache", "execution", "network_environment", "checks", "evidence",
        "artifact_dir", "pass_line",
    }
    summary = exact_fields(raw, fields, f"{backend} README E2E summary")
    require(summary["schema_version"] == SCHEMA_VERSION and summary["artifact_type"] == "ferrum_v084_readme_e2e_summary", f"{backend} README E2E schema/type differs")
    require(summary["status"] == "pass" and summary["version"] == VERSION and summary["backend"] == backend, f"{backend} README E2E status/version/backend differs")
    require(summary["source_git_sha"] == source_sha, f"{backend} README E2E candidate differs")
    require(summary["asset_name"] == BACKENDS[backend]["asset"], f"{backend} README E2E asset name differs")
    require(summary["asset_sha256"] == package["asset_sha256"] and summary["binary_sha256"] == package["binary_sha256"], f"{backend} README E2E downloaded-byte binding differs")

    model = exact_fields(summary["model"], {"alias", "revision", "files"}, f"{backend} README E2E model")
    expected_alias = "qwen3.5:4b-q4_k_m" if backend == "metal" else "qwen3.5:4b"
    require(model["alias"] == expected_alias, f"{backend} README alias differs")
    require_git_sha(model["revision"], f"{backend} model revision")
    files = model["files"]
    require(isinstance(files, list) and bool(files), f"{backend} model files are absent")
    model_names: set[str] = set()
    for index, raw_file in enumerate(files):
        row = require_object(raw_file, f"{backend} model.files[{index}]")
        require(
            set(row) in ({"name", "size_bytes"}, {"name", "size_bytes", "sha256"}),
            f"{backend} model.files[{index}] fields differ",
        )
        name = nonempty(row["name"], f"{backend} model.files[{index}].name")
        require(name not in model_names, f"{backend} model file is duplicated: {name}")
        require(type(row["size_bytes"]) is int and row["size_bytes"] > 0, f"{backend} model file size differs")
        if "sha256" in row:
            require_sha256(row["sha256"], f"{backend} model file SHA")
        model_names.add(name)

    network_refs = exact_fields(summary["network_environment"], {"urllib_public_downloads", "child_processes"}, f"{backend} network environment")
    validate_network_environment_ref(network_refs["urllib_public_downloads"], root=path.parent, consumer="urllib-public-github-downloads", where=f"{backend} urllib network environment")
    validate_network_environment_ref(network_refs["child_processes"], root=path.parent, consumer="ferrum-child-processes", where=f"{backend} child network environment")
    cold = exact_fields(summary["cold_cache"], {"fresh_cache", "cache_root", "undocumented_behavior_env", "download_size_announced", "download_complete"}, f"{backend} cold cache")
    require(cold["fresh_cache"] is True and cold["download_size_announced"] is True and cold["download_complete"] is True, f"{backend} cold-cache proof differs")
    nonempty(cold["cache_root"], f"{backend} cache root")
    undocumented = exact_fields(cold["undocumented_behavior_env"], {"behavior_overrides", "network_routing_is_behavior_override", "network_environment"}, f"{backend} undocumented behavior environment")
    require(undocumented["behavior_overrides"] == [] and undocumented["network_routing_is_behavior_override"] is False and undocumented["network_environment"] == network_refs, f"{backend} network routing/behavior classification differs")

    execution = exact_fields(summary["execution"], {"started_at", "finished_at", "deadline_seconds", "progress_signal"}, f"{backend} execution")
    started = parse_time(execution["started_at"], f"{backend} start time")
    finished = parse_time(execution["finished_at"], f"{backend} finish time")
    deadline = execution["deadline_seconds"]
    require(type(deadline) is int and deadline > 0 and started <= finished and (finished - started).total_seconds() <= deadline, f"{backend} execution deadline/timestamps differ")
    nonempty(execution["progress_signal"], f"{backend} progress signal")

    checks = exact_fields(summary["checks"], {"binary_version", "binary_help", "doctor", "run", "serve", "models", "chat", "stream", "log_scan"}, f"{backend} README checks")
    require(checks["binary_version"] is True and checks["binary_help"] is True and checks["doctor"] is True, f"{backend} basic CLI checks differ")
    run = exact_fields(checks["run"], {"exit_code", "non_empty", "disable_thinking"}, f"{backend} run check")
    require(run == {"exit_code": 0, "non_empty": True, "disable_thinking": True}, f"{backend} run contract differs")
    serve = exact_fields(checks["serve"], {"ready"}, f"{backend} serve check")
    require(serve["ready"] is True, f"{backend} serve readiness differs")
    models = exact_fields(checks["models"], {"http_status", "model_present"}, f"{backend} models check")
    require(models == {"http_status": 200, "model_present": True}, f"{backend} /v1/models contract differs")
    chat = exact_fields(checks["chat"], {"http_status", "non_empty_content"}, f"{backend} chat check")
    require(chat == {"http_status": 200, "non_empty_content": True}, f"{backend} non-stream chat contract differs")
    stream = exact_fields(checks["stream"], {"http_status", "done_count", "usage_chunks", "output_tokens"}, f"{backend} stream check")
    require(stream["http_status"] == 200 and stream["done_count"] == 1 and stream["usage_chunks"] == 1 and type(stream["output_tokens"]) is int and stream["output_tokens"] > 0, f"{backend} stream contract differs")
    scan = exact_fields(checks["log_scan"], {"forbidden_patterns", "found"}, f"{backend} log scan")
    required_patterns = {"panic", "oom", "cuda error", "metal error", "invalid utf-8", "<unk>", "[pad]", "control-token"}
    require(isinstance(scan["forbidden_patterns"], list) and {str(item).lower() for item in scan["forbidden_patterns"]} == required_patterns and scan["found"] == [], f"{backend} log scan denominator/result differs")

    evidence = exact_fields(summary["evidence"], E2E_EVIDENCE_KEYS, f"{backend} README evidence")
    evidence_finished: list[datetime] = []
    process_validations: dict[str, dict[str, Any]] = {}
    for label in ("binary_version", "binary_help", "doctor", "run", "serve"):
        validated_process = validate_e2e_command(
            evidence[label], root=path.parent, backend=backend, label=label,
            alias=expected_alias, package_binary_sha256=package["binary_sha256"],
            recorded_artifact_dir=nonempty(summary["artifact_dir"], f"{backend} README artifact_dir"),
            network_environment_ref=network_refs["child_processes"],
        )
        process_validations[label] = validated_process
        evidence_finished.append(validated_process["finished_at"])
    model_download = validate_e2e_model_download(
        evidence["download"], root=path.parent, backend=backend, alias=expected_alias
    )
    evidence_finished.append(model_download["finished_at"])
    require(
        model_download["run_process"]["command"] == process_validations["run"]["wrapper"]["command"]
        and model_download["run_process"]["stdout"] == process_validations["run"]["wrapper"]["stdout"]
        and model_download["run_process"]["stderr"] == process_validations["run"]["wrapper"]["stderr"]
        and model_download["run_process"]["progress"] == process_validations["run"]["wrapper"]["progress"],
        f"{backend} model download/run process closure differs",
    )
    for label in ("models", "chat", "stream"):
        evidence_finished.append(
            validate_e2e_http_exchange(
                evidence[label], root=path.parent, backend=backend, label=label
            )
        )
    _, _, logs = read_json_ref(evidence["logs"], root=path.parent, where=f"{backend} README logs")
    log_receipt = exact_fields(logs, {"schema_version", "artifact_type", "status", "forbidden_patterns", "found", "files"}, f"{backend} README log scan")
    require(log_receipt["schema_version"] == 1 and log_receipt["artifact_type"] == "ferrum_v084_readme_e2e_log_scan" and log_receipt["status"] == "pass" and log_receipt["found"] == [] and {str(item).lower() for item in log_receipt["forbidden_patterns"]} == required_patterns, f"{backend} README log scan receipt differs")
    scanned_files = log_receipt["files"]
    require(isinstance(scanned_files, list) and scanned_files, f"{backend} README log scan file denominator differs")
    for index, raw_file in enumerate(scanned_files):
        row = exact_fields(raw_file, {"label", "file"}, f"{backend} README scanned file {index}")
        _, scanned_path = resolve_ref(row["file"], root=path.parent, where=f"{backend} README scanned file {index}")
        scanned_text = scanned_path.read_text(encoding="utf-8").lower()
        require(not any(pattern in scanned_text for pattern in required_patterns), f"{backend} README scanned file contains forbidden output")
    artifact_dir = nonempty(summary["artifact_dir"], f"{backend} README artifact_dir")
    expected_pass = f"FERRUM 0.8.4 README E2E PASS: {backend} {artifact_dir}"
    require(summary["pass_line"] == expected_pass, f"{backend} README exact PASS line differs")
    require(max(evidence_finished) <= finished, f"{backend} README evidence finished after its summary")
    return {"backend": backend, "asset_sha256": summary["asset_sha256"], "binary_sha256": summary["binary_sha256"], "finished_at": finished}


def command_option(argv: Any, option: str, where: str) -> str:
    require(isinstance(argv, list) and all(isinstance(item, str) for item in argv), f"{where} command line differs")
    positions = [index for index, item in enumerate(argv) if item == option]
    require(len(positions) == 1 and positions[0] + 1 < len(argv), f"{where} must contain exactly one {option}")
    return nonempty(argv[positions[0] + 1], f"{where} {option}")


def validate_prerelease_manifest(path: Path) -> dict[str, Any]:
    data = exact_fields(read_json(path, "prerelease manifest"), {"schema_version", "artifact_type", "status", "version", "started_at", "finished_at", "source", "release", "evidence", "artifact_dir", "pass_line"}, "prerelease manifest")
    require(data["schema_version"] == SCHEMA_VERSION and data["artifact_type"] == "ferrum_v084_prerelease_manifest", "prerelease schema/type differs")
    require(data["status"] == "pass" and data["version"] == VERSION, "prerelease status/version differs")
    source = validate_source(data["source"], "prerelease.source")
    release_identity = exact_fields(data["release"], {"id", "tag", "release_candidate_tag", "asset_set_sha256"}, "prerelease.release")
    require(type(release_identity["id"]) is int and release_identity["id"] > 0 and release_identity["tag"] == TAG, "prerelease release id/tag differs")
    rc_tag = nonempty(release_identity["release_candidate_tag"], "prerelease RC tag")
    require(RC_TAG_RE.fullmatch(rc_tag) is not None, "prerelease RC tag differs")
    require_sha256(release_identity["asset_set_sha256"], "prerelease asset set SHA")
    manifest_started = parse_time(data["started_at"], "prerelease started_at")
    manifest_finished = parse_time(data["finished_at"], "prerelease finished_at")
    require(manifest_started <= manifest_finished, "prerelease manifest timestamps differ")
    evidence = exact_fields(
        data["evidence"],
        {
            "release_snapshot",
            "tag_ref_snapshot",
            "tag_snapshot",
            "rc_tag_ref_snapshot",
            "rc_tag_snapshot",
            "source_gates",
            "workflow_policy",
            "native_operator_set",
            "staged_assets",
            "public_downloads",
            "readme_e2e",
            "prepublication_binary_gates",
        },
        "prerelease.evidence",
    )
    _, _, release_snapshot = read_json_ref(evidence["release_snapshot"], root=path.parent, where="prerelease release snapshot")
    rows = github_asset_rows(release_snapshot, prerelease=True, where="prerelease release snapshot")
    fingerprint = asset_set_sha256(rows)
    require(release_snapshot["id"] == release_identity["id"] and fingerprint == release_identity["asset_set_sha256"], "prerelease release identity/fingerprint differs")
    _, _, tag_ref_snapshot = read_json_ref(evidence["tag_ref_snapshot"], root=path.parent, where="prerelease tag ref snapshot")
    _, _, tag_snapshot = read_json_ref(evidence["tag_snapshot"], root=path.parent, where="prerelease annotated tag snapshot")
    validate_tag_chain(
        tag_ref_snapshot,
        tag_snapshot,
        candidate_sha=source["git_sha"],
        where="prerelease annotated tag",
    )
    _, _, rc_ref_snapshot = read_json_ref(evidence["rc_tag_ref_snapshot"], root=path.parent, where="prerelease RC tag ref snapshot")
    _, _, rc_tag_snapshot = read_json_ref(evidence["rc_tag_snapshot"], root=path.parent, where="prerelease RC annotated tag snapshot")
    require(rc_tag_snapshot.get("tag") == rc_tag, "prerelease RC annotated tag name differs")
    validate_tag_chain(
        rc_ref_snapshot,
        rc_tag_snapshot,
        candidate_sha=source["git_sha"],
        where="prerelease RC annotated tag",
        expected_ref=f"refs/tags/{rc_tag}",
        expected_tag=rc_tag,
    )
    release_created = parse_time(release_snapshot.get("created_at"), "prerelease release created_at")
    release_published = parse_time(release_snapshot.get("published_at"), "prerelease release published_at")
    publication_cutoff = min(release_created, release_published)

    source_gates = exact_fields(evidence["source_gates"], {"unit", "metal", "cuda_full", "cuda_llama_dense"}, "prerelease source gates")
    source_specs = {
        "unit": ("unit", "unit.gate.json", "G0 SOURCE unit PASS: "),
        "metal": ("metal", "metal.gate.json", "G0 SOURCE metal PASS: "),
        "cuda_full": ("cuda-full", "g0_cuda4090_full.gate.json", "G0 SOURCE g0_cuda4090_full PASS: "),
        "cuda_llama_dense": ("cuda-llama-dense", "g0_cuda4090_llama_dense.gate.json", "G0 SOURCE g0_cuda4090_llama_dense PASS: "),
    }
    prerequisite_finishes: list[datetime] = []
    for key, (lane, child_filename, child_prefix) in source_specs.items():
        child, outer_path, child_path = validate_outer_child_gate(
            source_gates[key], root=path.parent, g0_gate_paths=None, lane=lane,
            child_filename=child_filename, child_pass_prefix=child_prefix,
            source_sha=source["git_sha"], where=f"prerelease source {key}",
        )
        validate_lane_gate(child, lane=lane if key in {"unit", "metal"} else ("g0_cuda4090_full" if key == "cuda_full" else "g0_cuda4090_llama_dense"), source_sha=source["git_sha"], where=f"prerelease source {key} child")
        deep_validate_source_gate(key, child, outer_path=outer_path, source_sha=source["git_sha"])
        outer = read_json(outer_path, f"prerelease source {key} outer")
        prerequisite_finishes.append(parse_time(outer.get("finished_at"), f"prerelease source {key} finished_at"))

    for key, lane, artifact_type, pass_prefix in (
        ("workflow_policy", "release-workflow-policy", "ferrum_v084_release_workflow_policy_manifest", "FERRUM 0.8.4 RELEASE WORKFLOW POLICY PASS"),
        ("native_operator_set", "native-operator-set", "ferrum_v084_native_operator_set_manifest", "FERRUM 0.8.4 NATIVE OPERATOR SET PASS"),
    ):
        _, gate_path, gate = read_json_ref(evidence[key], root=path.parent, where=f"prerelease {key}")
        validate_workflow_native_gate(gate, path=gate_path, lane=lane, artifact_type=artifact_type, pass_prefix=pass_prefix, source_sha=source["git_sha"], candidate_tag=rc_tag, where=f"prerelease {key}")
        prerequisite_finishes.append(parse_time(gate.get("finished_at"), f"prerelease {key} finished_at"))

    staged = exact_fields(evidence["staged_assets"], EXPECTED_ASSETS, "prerelease staged assets")
    downloads = exact_fields(evidence["public_downloads"], EXPECTED_ASSETS, "prerelease public downloads")
    release_by_name = {row["name"]: row for row in rows}
    paths: dict[str, Path] = {}
    terminal_finishes: list[datetime] = []
    for name, raw_download in downloads.items():
        download = exact_fields(raw_download, {"url", "http_status", "file", "receipt", "progress"}, f"public download receipt {name}")
        expected_url = f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{quote(name)}"
        require(download["url"] == expected_url and download["http_status"] == 200, f"public download request differs: {name}")
        ref, downloaded_path = resolve_ref(download["file"], root=path.parent, where=f"public download {name}")
        staged_ref, staged_path = resolve_ref(staged[name], root=path.parent, where=f"staged asset {name}")
        require(downloaded_path != staged_path, f"public download reused the staged path: {name}")
        published = release_by_name[name]
        require(ref["size_bytes"] == published["size"] and "sha256:" + ref["sha256"] == published["digest"], f"public download bytes differ from GitHub digest: {name}")
        require(ref["size_bytes"] == staged_ref["size_bytes"] and ref["sha256"] == staged_ref["sha256"], f"public download bytes differ from staged bytes: {name}")
        terminal_finishes.append(validate_public_download_provenance(download, root=path.parent, name=name, published=published, downloaded_ref=ref))
        paths[name] = downloaded_path
    packages = {backend: validate_downloaded_backend(backend, paths=paths, candidate_sha=source["git_sha"], rc_tag=rc_tag) for backend in BACKENDS}

    prepublication = exact_fields(
        evidence["prepublication_binary_gates"],
        {"metal", "cuda"},
        "prerelease prepublication binary gates",
    )
    prepublication_specs = {
        "metal": ("metal-tarball", "METAL TARBALL GATE PASS: "),
        "cuda": ("cuda-tarball", "CUDA TARBALL GATE PASS: "),
    }
    for backend, (lane, child_prefix) in prepublication_specs.items():
        child, outer_path, child_path = validate_outer_child_gate(
            prepublication[backend],
            root=path.parent,
            g0_gate_paths=None,
            lane=lane,
            child_filename="gate.json",
            child_pass_prefix=child_prefix,
            source_sha=source["git_sha"],
            where=f"prerelease {backend} prepublication gate",
        )
        validate_simple_gate(
            child,
            path=child_path,
            mode=lane,
            expected_asset_source="asset-path",
            where=f"prerelease {backend} prepublication child",
        )
        child_asset = require_object(
            require_object(child.get("evidence"), f"prerelease {backend} evidence").get("asset"),
            f"prerelease {backend} asset",
        )
        _, child_binary_path = resolve_ref(
            child_asset.get("unpacked_binary"),
            root=child_path.parent,
            where=f"prerelease {backend} unpacked binary",
        )
        require(
            child_asset.get("sha256") == packages[backend]["asset_sha256"]
            and file_sha256(child_binary_path) == packages[backend]["binary_sha256"],
            f"prerelease {backend} exercised tarball/binary bytes differ from staged/public bytes",
        )
        outer = require_object(read_json(outer_path, f"prerelease {backend} prepublication outer"), f"prerelease {backend} prepublication outer")
        delegated = outer.get("delegated_command_line")
        expected_asset = BACKENDS[backend]["asset"]
        asset_path = command_option(delegated, "--asset-path", f"prerelease {backend} prepublication")
        require(PurePosixPath(asset_path).name == expected_asset, f"prerelease {backend} --asset-path differs")
        require(
            command_option(delegated, "--sha256", f"prerelease {backend} prepublication")
            == packages[backend]["asset_sha256"],
            f"prerelease {backend} --sha256 differs from staged/public tarball",
        )
        outer = read_json(outer_path, f"prerelease {backend} prepublication outer")
        prerequisite_finishes.append(parse_time(outer.get("finished_at"), f"prerelease {backend} prepublication finished_at"))

    e2e = exact_fields(evidence["readme_e2e"], {"metal", "cuda"}, "prerelease README E2E")
    for backend in ("metal", "cuda"):
        _, e2e_path, summary = read_json_ref(e2e[backend], root=path.parent, where=f"{backend} README E2E summary")
        validated_e2e = validate_readme_e2e(summary, path=e2e_path, backend=backend, source_sha=source["git_sha"], package=packages[backend])
        terminal_finishes.append(validated_e2e["finished_at"])
    require(prerequisite_finishes and max(prerequisite_finishes) <= publication_cutoff, "prerelease required gate finished after GitHub release creation/publication")
    require(terminal_finishes and min(terminal_finishes) >= release_published and manifest_finished >= max(terminal_finishes), "prerelease download/E2E timing closure differs")
    artifact_dir = nonempty(data["artifact_dir"], "prerelease artifact_dir")
    pass_line = f"FERRUM 0.8.4 PRERELEASE DOWNLOAD PASS: {artifact_dir}"
    require(data["pass_line"] == pass_line, "prerelease exact PASS line differs")
    return {
        "source": source,
        "release": {"id": release_identity["id"], "tag": TAG, "asset_set_sha256": fingerprint},
        "release_candidate_tag": rc_tag,
        "packages": packages,
        "started_at": manifest_started,
        "finished_at": manifest_finished,
        "pass_line": pass_line,
    }


def promotion_target_identity(raw: Any, *, prerelease: bool, where: str) -> dict[str, Any]:
    """Return only the release identity the 0.8.4 goal requires to stay fixed.

    GitHub updates service-owned fields such as asset ``download_count`` while
    a public release is being observed.  Those fields are deliberately outside
    the identity: promotion binds the release id/tag/draft state and each
    asset's id/name/size/digest, while allowing unrelated counters and display
    metadata to change concurrently.
    """

    release = require_object(raw, where)
    rows = github_asset_rows(release, prerelease=prerelease, where=where)
    return {
        "id": release["id"],
        "tag_name": release["tag_name"],
        "draft": release["draft"],
        "assets": rows,
    }


def validate_promotion_mutation_receipt(
    raw: Any, *, release_id: int, where: str
) -> dict[str, Any]:
    receipt = exact_fields(
        raw,
        {
            "schema_version",
            "artifact_type",
            "status",
            "method",
            "endpoint",
            "body",
            "body_sha256",
            "release_id",
            "attempted_at",
            "confirmed_at",
            "confirmation",
            "ambiguous_outcome_recovered",
        },
        where,
    )
    body = {"prerelease": False}
    require(
        receipt["schema_version"] == SCHEMA_VERSION
        and receipt["artifact_type"] == "ferrum_v084_github_promotion_mutation_receipt"
        and receipt["status"] == "confirmed"
        and receipt["method"] == "PATCH"
        and receipt["endpoint"] == f"/repos/{REPOSITORY}/releases/{release_id}"
        and receipt["body"] == body
        and receipt["body_sha256"] == canonical_sha256(body)
        and receipt["release_id"] == release_id,
        f"{where} immutable PATCH identity differs",
    )
    attempted = parse_time(receipt["attempted_at"], f"{where}.attempted_at")
    confirmed = parse_time(receipt["confirmed_at"], f"{where}.confirmed_at")
    require(attempted <= confirmed, f"{where} timestamps differ")
    confirmation = receipt["confirmation"]
    require(
        confirmation
        in {"patch-response", "saved-patch-response", "live-state-recovery"},
        f"{where} confirmation result differs",
    )
    require(
        receipt["ambiguous_outcome_recovered"]
        is (confirmation == "live-state-recovery"),
        f"{where} ambiguity state differs",
    )
    return receipt


def validate_promotion_manifest(path: Path) -> dict[str, Any]:
    data = exact_fields(read_json(path, "promotion manifest"), {"schema_version", "artifact_type", "status", "version", "source", "release", "evidence", "artifact_dir", "pass_line"}, "promotion manifest")
    require(data["schema_version"] == SCHEMA_VERSION and data["artifact_type"] == "ferrum_v084_promotion_manifest", "promotion schema/type differs")
    require(data["status"] == "pass" and data["version"] == VERSION, "promotion status/version differs")
    source = validate_source(data["source"], "promotion.source")
    identity = exact_fields(data["release"], {"id", "tag", "asset_set_sha256"}, "promotion.release")
    require(type(identity["id"]) is int and identity["id"] > 0 and identity["tag"] == TAG, "promotion release id/tag differs")
    require_sha256(identity["asset_set_sha256"], "promotion asset set SHA")
    evidence = exact_fields(
        data["evidence"],
        {
            "prerelease_manifest",
            "mutation_receipt",
            "release_before",
            "release_after",
            "latest_release",
            "tag_ref_snapshot",
            "tag_snapshot",
        },
        "promotion.evidence",
    )
    _, prerelease_path = resolve_ref(evidence["prerelease_manifest"], root=path.parent, where="promotion prerelease manifest")
    prerelease = validate_prerelease_manifest(prerelease_path)
    require(prerelease["source"] == source and prerelease["release"] == {"id": identity["id"], "tag": TAG, "asset_set_sha256": identity["asset_set_sha256"]}, "promotion/prerelease identity differs")

    _, _, mutation = read_json_ref(
        evidence["mutation_receipt"],
        root=path.parent,
        where="promotion mutation receipt",
    )
    validate_promotion_mutation_receipt(
        mutation,
        release_id=identity["id"],
        where="promotion mutation receipt",
    )

    _, _, before = read_json_ref(evidence["release_before"], root=path.parent, where="promotion release-before snapshot")
    _, _, after = read_json_ref(evidence["release_after"], root=path.parent, where="promotion release-after snapshot")
    _, _, latest = read_json_ref(evidence["latest_release"], root=path.parent, where="promotion latest-release snapshot")
    before_rows = github_asset_rows(before, prerelease=True, where="promotion release-before snapshot")
    after_rows = github_asset_rows(after, prerelease=False, where="promotion release-after snapshot")
    latest_rows = github_asset_rows(latest, prerelease=False, where="promotion latest-release snapshot")
    before_identity = promotion_target_identity(
        before, prerelease=True, where="promotion release-before snapshot"
    )
    after_identity = promotion_target_identity(
        after, prerelease=False, where="promotion release-after snapshot"
    )
    latest_identity = promotion_target_identity(
        latest, prerelease=False, where="promotion latest-release snapshot"
    )
    require(before_identity == after_identity == latest_identity, "promotion/latest target identity changed")
    require(before["id"] == after["id"] == latest["id"] == identity["id"], "promotion/latest release id changed")
    require(before_rows == after_rows == latest_rows, "promotion changed asset id/name/size/digest set")
    require(asset_set_sha256(after_rows) == identity["asset_set_sha256"], "promotion asset fingerprint differs")
    promoted_at = parse_time(after.get("updated_at"), "promotion release updated_at")
    require(promoted_at >= prerelease["finished_at"], "promotion occurred before prerelease PASS completion")
    _, _, tag_ref_snapshot = read_json_ref(evidence["tag_ref_snapshot"], root=path.parent, where="promotion tag ref snapshot")
    _, _, tag_snapshot = read_json_ref(evidence["tag_snapshot"], root=path.parent, where="promotion annotated tag snapshot")
    validate_tag_chain(
        tag_ref_snapshot,
        tag_snapshot,
        candidate_sha=source["git_sha"],
        where="promotion annotated tag",
    )
    artifact_dir = nonempty(data["artifact_dir"], "promotion artifact_dir")
    pass_line = f"FERRUM 0.8.4 PROMOTION PASS: {artifact_dir}"
    require(data["pass_line"] == pass_line, "promotion exact PASS line differs")
    return {"source": source, "release": {"id": identity["id"], "tag": TAG, "asset_set_sha256": identity["asset_set_sha256"]}, "promoted_at": promoted_at, "pass_line": pass_line}


def validate_simple_gate(
    raw: Any,
    *,
    path: Path,
    mode: str,
    expected_asset_source: str | None,
    where: str,
) -> dict[str, Any]:
    gate = exact_fields(
        raw,
        {
            "schema_version", "artifact_type", "status", "mode", "version",
            "artifact_dir", "started_at", "finished_at", "deadline_at",
            "duration_sec", "rc", "pass_line", "checks", "evidence",
        },
        where,
    )
    require(gate.get("status") == "pass" and gate.get("mode") == mode and gate.get("version") == VERSION, f"{where} status/mode/version differs")
    try:
        if mode in {"metal-tarball", "cuda-tarball"}:
            require(
                gate.get("schema_version") == binary_gate.SCHEMA_VERSION
                and gate.get("artifact_type") == "ferrum_release_binary_gate"
                and gate.get("rc") == 0,
                f"{where} binary v2 identity differs",
            )
            binary_gate.validate_timing(gate, where, expected_rc=0)
            artifact_dir = nonempty(gate.get("artifact_dir"), f"{where}.artifact_dir")
            require(
                gate.get("pass_line") == binary_gate.PASS_PREFIXES[mode] + artifact_dir,
                f"{where} exact PASS line differs",
            )
            checks = require_object(gate.get("checks"), f"{where}.checks")
            require(set(checks) == {"version", "cli", "serve"} and checks.get("version") is True, f"{where} check denominator differs")
            binary_gate.validate_product_checks(checks)
            evidence = exact_fields(gate.get("evidence"), {"asset", "commands"}, f"{where}.evidence")
            binary_gate.validate_asset_evidence(path.parent, evidence["asset"], version=VERSION, mode=mode)
            command_names = {"version", "cli", "serve"} | ({"ldd"} if mode == "cuda-tarball" else set())
            commands = exact_fields(evidence["commands"], command_names, f"{where}.commands")
            version_receipt = binary_gate.validate_command_bundle(path.parent, commands["version"], f"{where} version")
            cli_receipt = binary_gate.validate_command_bundle(path.parent, commands["cli"], f"{where} cli")
            binary_gate.validate_serve_evidence(path.parent, commands["serve"])
            serve_receipt_path = binary_gate.resolve_evidence_ref(path.parent, commands["serve"]["command"]["receipt"], f"{where} serve receipt")
            serve_receipt = binary_gate.read_json(serve_receipt_path, f"{where} serve receipt")
            recorded_binary = version_receipt.get("command", [None])[0]
            require(
                isinstance(recorded_binary, str)
                and PurePosixPath(recorded_binary).name == "ferrum"
                and version_receipt.get("command") == [recorded_binary, "--version"]
                and cli_receipt.get("command", [None])[0] == recorded_binary
                and "run" in cli_receipt.get("command", [])
                and "--disable-thinking" in cli_receipt.get("command", [])
                and serve_receipt.get("command", [None])[0] == recorded_binary
                and serve_receipt.get("command", [None, None])[1] == "serve"
                and "--disable-thinking" in serve_receipt.get("command", []),
                f"{where} recorded binary command identity differs",
            )
            if mode == "cuda-tarball":
                ldd_receipt = binary_gate.validate_command_bundle(path.parent, commands["ldd"], f"{where} ldd")
                require(ldd_receipt.get("command") == ["ldd", recorded_binary], f"{where} ldd binary identity differs")
            version_text = binary_gate.referenced_text(path.parent, commands["version"]["stdout"], f"{where} version stdout") + binary_gate.referenced_text(path.parent, commands["version"]["stderr"], f"{where} version stderr")
            require(f"ferrum {VERSION}" in version_text, f"{where} version output differs")
        else:
            binary_gate.validate_gate_data(gate, root=path.parent)
    except binary_gate.GateError as exc:
        raise ValidationError(f"{where} failed binary v2 deep validation: {exc}") from exc
    if expected_asset_source is not None:
        evidence = require_object(gate.get("evidence"), f"{where}.evidence")
        asset = require_object(evidence.get("asset"), f"{where}.evidence.asset")
        require(
            asset.get("source") == expected_asset_source,
            f"{where} asset source differs",
        )
    return gate


def validate_lane_gate(raw: Any, *, lane: str, source_sha: str, where: str) -> None:
    gate = require_object(raw, where)
    require(gate.get("status") == "pass" and gate.get("lane") == lane, f"{where} status/lane differs")
    source = gate.get("source")
    if isinstance(source, dict):
        if "git_sha" in source:
            require(source.get("git_sha") == source_sha, f"{where} candidate differs")
        if "dirty" in source:
            require(source.get("dirty") is False, f"{where} source is dirty")
        if "dirty_status" in source:
            require(
                source.get("dirty_status") == {"is_dirty": False, "status_short": []},
                f"{where} source is dirty",
            )


def validate_unit_artifact(child: dict[str, Any], *, root: Path, source_sha: str) -> None:
    require(child.get("artifact_type") == "g0_source_unit_bounded_gate" and child.get("receipt_schema") == "ferrum.bounded-command-receipt.v1", "unit bounded gate identity differs")
    require(child.get("command") == ["env", "PYTHONDONTWRITEBYTECODE=1", "CARGO_BUILD_JOBS=8", "RUST_TEST_THREADS=8", "cargo", "test", "--workspace", "--all-targets"], "unit bounded command differs")
    source = require_object(child.get("source"), "unit bounded source")
    require(source.get("git_sha") == source_sha and source.get("dirty_status") == {"is_dirty": False, "status_short": []}, "unit bounded source differs")
    refs: dict[str, tuple[dict[str, Any], Path]] = {}
    for key in ("bounded_receipt", "source_receipt", "stdout_log", "stderr_log"):
        refs[key] = resolve_ref(child.get(key), root=root, where=f"unit {key}")
    receipt = read_json(refs["bounded_receipt"][1], "unit bounded receipt")
    require(receipt.get("schema") == "ferrum.bounded-command-receipt.v1" and receipt.get("status") == "pass" and receipt.get("rc") == 0 and receipt.get("reason") == "command_completed" and receipt.get("cleanup", {}).get("process_group_gone") is True and receipt.get("violation") is None, "unit bounded receipt differs")
    receipt_source = read_json(refs["source_receipt"][1], "unit source receipt")
    require(receipt_source.get("git_sha") == source_sha and receipt_source.get("dirty_status") == {"is_dirty": False, "status_short": []}, "unit source receipt differs")
    for key, receipt_key in (("stdout_log", "stdout"), ("stderr_log", "stderr")):
        inner = require_object(receipt.get(receipt_key), f"unit bounded receipt {receipt_key}")
        require(inner.get("sha256") == refs[key][0]["sha256"] and inner.get("size_bytes") == refs[key][0]["size_bytes"], f"unit bounded {receipt_key} binding differs")


def validate_accelerator_artifact(lane: str, *, root: Path, source_sha: str) -> None:
    if lane == "metal":
        command = [sys.executable, str(RELEASE_DIR / "validate_metal_readme_regression.py"), str(root / "metal-readme"), "--require-release-matrix"]
        process = __import__("subprocess").run(command, text=True, stdout=__import__("subprocess").PIPE, stderr=__import__("subprocess").PIPE, check=False)
        require(process.returncode == 0 and f"METAL README GATE PASS: {root / 'metal-readme'}" in process.stdout, f"Metal strict artifact validator failed: {process.stderr.strip()}")
    elif lane == "cuda-full":
        try:
            result = m3_validator.validate_artifact(
                root,
                require_bench=True,
                require_profile_events=False,
                expected_candidate_sha=source_sha,
            )
        except m3_validator.ValidationError as exc:
            raise ValidationError(f"CUDA full m3 artifact validator failed: {exc}") from exc
        require(result.get("ok") is True, "CUDA full m3 artifact validator did not pass")
    elif lane == "cuda-llama-dense":
        try:
            result = dense_gate.validate_artifact(root, expected_git_sha=source_sha)
        except Exception as exc:
            raise ValidationError(f"CUDA dense artifact validator failed: {exc}") from exc
        require(result.get("status") == "pass", "CUDA dense artifact validator did not pass")


def deep_validate_source_gate(
    key: str, child: dict[str, Any], *, outer_path: Path, source_sha: str
) -> None:
    if key == "unit":
        validate_unit_artifact(child, root=outer_path.parent, source_sha=source_sha)
    else:
        validate_accelerator_artifact(
            {"metal": "metal", "cuda_full": "cuda-full", "cuda_llama_dense": "cuda-llama-dense"}[key],
            root=outer_path.parent,
            source_sha=source_sha,
        )


def validate_crates_gate(path: Path, *, source_sha: str) -> datetime:
    try:
        gate = crates_release.validate_publish_manifest(path)
    except crates_release.ReleaseError as exc:
        raise ValidationError(f"crates.io publish manifest failed deep validation: {exc}") from exc
    candidate = require_object(gate.get("release_candidate"), "crates.io release candidate")
    require(candidate.get("git_sha") == source_sha and candidate.get("dirty") is False, "crates.io gate candidate differs")
    return parse_time(gate.get("created_at"), "crates.io gate created_at")


def validate_workflow_native_gate(
    raw: Any,
    *,
    path: Path,
    lane: str,
    artifact_type: str,
    pass_prefix: str,
    source_sha: str,
    candidate_tag: str,
    where: str,
) -> dict[str, Any]:
    gate = exact_fields(
        raw,
        {
            "schema_version",
            "artifact_type",
            "status",
            "version",
            "lane",
            "started_at",
            "finished_at",
            "artifact_dir",
            "pass_line",
            "evidence",
        },
        where,
    )
    require(
        gate["schema_version"] == SCHEMA_VERSION
        and gate["artifact_type"] == artifact_type
        and gate["status"] == "pass"
        and gate["version"] == VERSION
        and gate["lane"] == lane,
        f"{where} schema/type/status/version/lane differs",
    )
    require(
        parse_time(gate["started_at"], f"{where}.started_at")
        <= parse_time(gate["finished_at"], f"{where}.finished_at"),
        f"{where} timestamps differ",
    )
    evidence = require_object(gate["evidence"], f"{where}.evidence")
    candidate = require_object(evidence.get("candidate"), f"{where}.evidence.candidate")
    require(
        candidate.get("git_sha") == source_sha
        and candidate.get("tag") == candidate_tag,
        f"{where} candidate SHA/tag differs",
    )
    require_git_sha(candidate.get("git_tree_sha"), f"{where} candidate git tree SHA")
    artifact_dir = nonempty(gate["artifact_dir"], f"{where}.artifact_dir")
    require(
        gate["pass_line"] == f"{pass_prefix}: {artifact_dir}",
        f"{where} exact PASS line/artifact_dir binding differs",
    )
    try:
        if lane == "release-workflow-policy":
            deep = workflow_native.validate_workflow_policy_manifest(path)
        else:
            deep = workflow_native.validate_native_set_manifest(path)
    except Exception as exc:
        raise ValidationError(f"{where} failed deep validation: {exc}") from exc
    return deep


def validate_g0_summary(raw: Any, *, path: Path, source_sha: str) -> set[Path]:
    summary = exact_fields(
        raw,
        {
            "schema_version",
            "status",
            "gates",
            "artifact_dir",
            "pass_line",
            "release_candidate_sha",
        },
        "G0 summary",
    )
    require(summary["schema_version"] == SCHEMA_VERSION, "G0 summary schema differs")
    require(summary["status"] == "pass", "G0 summary status differs")
    require(summary["release_candidate_sha"] == source_sha, "G0 summary candidate differs")
    artifact_dir = nonempty(summary["artifact_dir"], "G0 summary artifact_dir")
    require(artifact_dir == FINAL_ARTIFACT_DIR, "G0 summary artifact_dir differs")
    require(summary["pass_line"] == f"G0 RELEASE PASS: {artifact_dir}", "G0 summary exact PASS line differs")
    gates = summary["gates"]
    require(isinstance(gates, list) and len(gates) >= 8, "G0 summary gate list is incomplete")
    gate_paths: set[Path] = set()
    summary_root = path.parent.resolve()
    for index, value in enumerate(gates):
        text = nonempty(value, f"G0 summary gates[{index}]")
        pure = PurePosixPath(text)
        require(
            not pure.is_absolute()
            and "\\" not in text
            and ".." not in pure.parts
            and pure.name == "gate.manifest.json",
            f"G0 summary gates[{index}] is not a canonical relative gate manifest path",
        )
        candidate = path.parent.joinpath(*pure.parts)
        require(not candidate.is_symlink(), f"G0 summary gate is a symlink: {text}")
        resolved = candidate.resolve()
        require(resolved.is_relative_to(summary_root), f"G0 summary gate escapes its root: {text}")
        require(resolved.is_file() and not resolved.is_symlink(), f"G0 summary gate is missing: {text}")
        require(resolved not in gate_paths, f"G0 summary has duplicate gate path {text}")
        gate_paths.add(resolved)
    return gate_paths


def recorded_posix_path(value: Any, *, where: str) -> PurePosixPath:
    text = nonempty(value, where)
    pure = PurePosixPath(text)
    require("\\" not in text and ".." not in pure.parts, f"{where} is not a canonical recorded path")
    return pure


def pretty_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_outer_artifact_tree(
    outer: dict[str, Any], *, outer_path: Path, where: str
) -> dict[str, dict[str, Any]]:
    child_artifacts = require_object(outer.get("child_artifacts"), f"{where} child_artifacts")
    tree = exact_fields(
        child_artifacts.get("artifact_tree"),
        {"schema_version", "kind", "file_count", "total_size_bytes", "files", "sha256"},
        f"{where} artifact tree",
    )
    require(tree["schema_version"] == 1 and tree["kind"] == "standard-g0-regular-file-tree", f"{where} artifact tree identity differs")
    root = outer_path.parent.resolve()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    forbidden_parts = {".cache", "cache", "huggingface", "hub", "models", "model-cache"}
    forbidden_suffixes = {".gguf", ".safetensors", ".pt", ".pth"}
    for candidate in sorted(outer_path.parent.rglob("*"), key=lambda item: item.as_posix()):
        require(not candidate.is_symlink(), f"{where} artifact tree contains symlink")
        resolved = candidate.resolve()
        require(resolved.is_relative_to(root), f"{where} artifact tree escapes its root")
        if candidate.is_dir():
            continue
        require(candidate.is_file(), f"{where} artifact tree contains non-regular entry")
        relative = candidate.relative_to(outer_path.parent).as_posix()
        if relative == outer_path.name:
            continue
        pure = PurePosixPath(relative)
        require(not ({part.lower() for part in pure.parts} & forbidden_parts) and pure.suffix.lower() not in forbidden_suffixes, f"{where} artifact tree contains model/cache bytes")
        require(relative not in seen, f"{where} artifact tree contains duplicate path")
        seen.add(relative)
        rows.append({"path": relative, "sha256": file_sha256(candidate), "size_bytes": candidate.stat().st_size})
    require(
        tree["files"] == rows
        and tree["file_count"] == len(rows)
        and tree["total_size_bytes"] == sum(row["size_bytes"] for row in rows)
        and tree["sha256"] == pretty_json_sha256(rows),
        f"{where} artifact tree denominator/bytes differ",
    )
    return {row["path"]: row for row in rows}


def validate_child_execution_artifacts(
    outer: dict[str, Any], *, outer_path: Path, tree: dict[str, dict[str, Any]], where: str
) -> None:
    rows = outer.get("child_execution_artifacts")
    require(isinstance(rows, list) and len(rows) == 3, f"{where} child execution artifact denominator differs")
    expected = {"run_gate.child.command.json", "run_gate.child.stdout", "run_gate.child.stderr"}
    by_name: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = exact_fields(raw, {"path", "sha256", "size_bytes"}, f"{where} child execution artifact")
        name = nonempty(row["path"], f"{where} child execution path")
        require(name in expected and name not in by_name and tree.get(name) == row, f"{where} child execution artifact binding differs")
        by_name[name] = row
    require(set(by_name) == expected, f"{where} child execution artifact denominator differs")
    command = read_json(outer_path.parent / "run_gate.child.command.json", f"{where} child command")
    command_obj = exact_fields(command, {"cmd", "cwd", "timeout_seconds", "started_at", "finished_at", "duration_seconds", "returncode", "env_overrides"}, f"{where} child command")
    require(command_obj["cmd"] == outer.get("delegated_command_line") and command_obj["returncode"] == 0, f"{where} child command/return code differs")
    started = parse_time(command_obj["started_at"], f"{where} child command started_at")
    finished = parse_time(command_obj["finished_at"], f"{where} child command finished_at")
    require(started <= finished and 0 <= command_obj["duration_seconds"] <= command_obj["timeout_seconds"], f"{where} child command timing differs")


def validate_outer_child_gate(
    raw: Any,
    *,
    root: Path,
    g0_gate_paths: set[Path] | None,
    lane: str,
    child_filename: str,
    child_pass_prefix: str,
    source_sha: str,
    where: str,
) -> tuple[dict[str, Any], Path, Path]:
    pair = exact_fields(raw, {"outer", "child"}, where)
    _, outer_path, outer = read_json_ref(pair["outer"], root=root, where=f"{where}.outer")
    _, child_path, child = read_json_ref(pair["child"], root=root, where=f"{where}.child")
    if g0_gate_paths is not None:
        require(outer_path in g0_gate_paths, f"G0 summary does not name {where} outer manifest")
    require(
        outer_path.name == "gate.manifest.json"
        and child_path == outer_path.parent / child_filename,
        f"{where} outer/child layout differs",
    )
    outer_obj = require_object(outer, f"{where} outer")
    require(
        outer_obj.get("schema_version") == SCHEMA_VERSION
        and outer_obj.get("status") == "pass"
        and outer_obj.get("lane") == lane
        and outer_obj.get("child_returncode") == 0,
        f"{where} outer status/lane/child return code differs",
    )
    require(outer_obj.get("git_sha") == source_sha, f"{where} outer candidate differs")
    require(
        outer_obj.get("dirty_status") == {"is_dirty": False, "status_short": []},
        f"{where} outer source is dirty",
    )
    artifact_dir = nonempty(outer_obj.get("artifact_dir"), f"{where} outer artifact_dir")
    recorded_artifact_dir = recorded_posix_path(artifact_dir, where=f"{where} artifact_dir")
    expected_outer_pass = f"FERRUM GATE {lane} PASS: {artifact_dir}"
    expected_child_pass = child_pass_prefix + artifact_dir
    require(outer_obj.get("pass_line") == expected_outer_pass, f"{where} outer exact PASS line differs")
    require(outer_obj.get("child_pass_line") == expected_child_pass, f"{where} delegated exact PASS line differs")

    child_artifacts = require_object(outer_obj.get("child_artifacts"), f"{where} child_artifacts")
    bound = exact_fields(
        child_artifacts.get("child_manifest"),
        {"path", "sha256", "size_bytes"},
        f"{where} bound child manifest",
    )
    bound_path = recorded_posix_path(bound.get("path"), where=f"{where} bound child path")
    require(
        bound_path == recorded_artifact_dir / child_filename,
        f"{where} outer binds a different recorded child manifest",
    )
    require(
        require_sha256(bound.get("sha256"), f"{where} bound child SHA256")
        == file_sha256(child_path)
        and bound.get("size_bytes") == child_path.stat().st_size,
        f"{where} bound child byte identity differs",
    )
    tree = validate_outer_artifact_tree(outer_obj, outer_path=outer_path, where=where)
    validate_child_execution_artifacts(outer_obj, outer_path=outer_path, tree=tree, where=where)
    return require_object(child, f"{where} child"), outer_path, child_path


def validate_final_manifest(path: Path) -> dict[str, Any]:
    data = exact_fields(read_json(path, "final manifest"), {"schema_version", "artifact_type", "status", "version", "source", "release", "evidence", "artifact_dir", "pass_line"}, "final manifest")
    require(data["schema_version"] == SCHEMA_VERSION and data["artifact_type"] == "ferrum_v084_final_manifest", "final schema/type differs")
    require(data["status"] == "pass" and data["version"] == VERSION, "final status/version differs")
    source = validate_source(data["source"], "final.source")
    release = exact_fields(data["release"], {"id", "tag", "asset_set_sha256"}, "final.release")
    require(type(release["id"]) is int and release["id"] > 0 and release["tag"] == TAG, "final release id/tag differs")
    require_sha256(release["asset_set_sha256"], "final asset set SHA")
    fields = {
        "prerelease_manifest",
        "promotion_manifest",
        "metal_tarball",
        "cuda_tarball",
        "crates_io",
        "homebrew_metal",
        "homebrew_cuda_fetch",
        "workflow_policy",
        "native_operator_set",
        "g0_summary",
        "g0_gates",
    }
    evidence = exact_fields(data["evidence"], fields, "final.evidence")

    _, prerelease_path = resolve_ref(evidence["prerelease_manifest"], root=path.parent, where="final prerelease manifest")
    _, promotion_path = resolve_ref(evidence["promotion_manifest"], root=path.parent, where="final promotion manifest")
    prerelease = validate_prerelease_manifest(prerelease_path)
    promotion = validate_promotion_manifest(promotion_path)
    expected_identity = {"id": release["id"], "tag": TAG, "asset_set_sha256": release["asset_set_sha256"]}
    require(prerelease["source"] == promotion["source"] == source and prerelease["release"] == promotion["release"] == expected_identity, "final prerelease/promotion/source identity differs")

    _, g0_summary_path, g0_summary = read_json_ref(evidence["g0_summary"], root=path.parent, where="final G0 summary")
    g0_gate_paths = validate_g0_summary(
        g0_summary,
        path=g0_summary_path,
        source_sha=source["git_sha"],
    )
    simple_modes = {
        "metal_tarball": ("metal-tarball", "METAL TARBALL GATE PASS: "),
        "cuda_tarball": ("cuda-tarball", "CUDA TARBALL GATE PASS: "),
        "homebrew_metal": ("homebrew-metal", "HOMEBREW METAL GATE PASS: "),
        "homebrew_cuda_fetch": ("homebrew-cuda-fetch", "HOMEBREW CUDA FETCH GATE PASS: "),
    }
    for key, (mode, child_pass_prefix) in simple_modes.items():
        child, outer_path, child_path = validate_outer_child_gate(
            evidence[key],
            root=path.parent,
            g0_gate_paths=g0_gate_paths,
            lane=mode,
            child_filename="gate.json",
            child_pass_prefix=child_pass_prefix,
            source_sha=source["git_sha"],
            where=f"final {key} gate",
        )
        validate_simple_gate(
            child,
            path=child_path,
            mode=mode,
            expected_asset_source=("public-url" if key in {"metal_tarball", "cuda_tarball"} else None),
            where=f"final {key} child gate",
        )
        child_evidence = require_object(child.get("evidence"), f"final {key} evidence")
        backend = "metal" if key in {"metal_tarball", "homebrew_metal"} else "cuda"
        package = prerelease["packages"][backend]
        if key in {"metal_tarball", "cuda_tarball"}:
            child_asset = require_object(child_evidence.get("asset"), f"final {key} asset")
            _, child_binary_path = resolve_ref(child_asset.get("unpacked_binary"), root=child_path.parent, where=f"final {key} unpacked binary")
            require(
                child_asset.get("sha256") == package["asset_sha256"]
                and file_sha256(child_binary_path) == package["binary_sha256"],
                f"final {key} public tarball/binary bytes differ from prerelease bytes",
            )
        elif key == "homebrew_metal":
            formula = require_object(child_evidence.get("formula"), "final Homebrew Metal formula")
            identity = require_object(formula.get("identity"), "final Homebrew Metal formula identity")
            _, installed_path = resolve_ref(child_evidence.get("installed_binary"), root=child_path.parent, where="final Homebrew Metal installed identity")
            installed = require_object(read_json(installed_path, "final Homebrew Metal installed identity"), "final Homebrew Metal installed identity")
            require(
                identity.get("stable_checksum") == package["asset_sha256"]
                and installed.get("sha256") == package["binary_sha256"],
                "final Homebrew Metal formula/installed bytes differ from prerelease bytes",
            )
        else:
            formula = require_object(child_evidence.get("formula"), "final Homebrew CUDA formula")
            identity = require_object(formula.get("identity"), "final Homebrew CUDA formula identity")
            _, fetched_path = resolve_ref(child_evidence.get("fetched_archive"), root=child_path.parent, where="final Homebrew CUDA fetched identity")
            fetched = require_object(read_json(fetched_path, "final Homebrew CUDA fetched identity"), "final Homebrew CUDA fetched identity")
            require(
                identity.get("stable_checksum") == package["asset_sha256"]
                and fetched.get("sha256") == package["asset_sha256"],
                "final Homebrew CUDA formula/fetched bytes differ from prerelease bytes",
            )
        outer = read_json(outer_path, f"final {key} outer")
        require(parse_time(outer.get("started_at"), f"final {key} started_at") >= promotion["promoted_at"], f"final {key} ran before promotion")
        delegated = outer.get("delegated_command_line")
        require(isinstance(delegated, list), f"final {key} delegated command differs")
        if key in {"metal_tarball", "cuda_tarball"}:
            require("--asset-path" not in delegated and "--sha256" not in delegated, f"final {key} did not use the public release URL path")
    _, crates_path, _ = read_json_ref(evidence["crates_io"], root=path.parent, where="final crates.io gate")
    require(validate_crates_gate(crates_path, source_sha=source["git_sha"]) >= promotion["promoted_at"], "crates.io publish occurred before promotion")
    workflow_native_specs = {
        "workflow_policy": (
            "release-workflow-policy",
            "ferrum_v084_release_workflow_policy_manifest",
            "FERRUM 0.8.4 RELEASE WORKFLOW POLICY PASS",
        ),
        "native_operator_set": (
            "native-operator-set",
            "ferrum_v084_native_operator_set_manifest",
            "FERRUM 0.8.4 NATIVE OPERATOR SET PASS",
        ),
    }
    workflow_native_evidence: dict[str, dict[str, Any]] = {}
    workflow_native_paths: dict[str, Path] = {}
    for key, (lane, artifact_type, pass_prefix) in workflow_native_specs.items():
        _, gate_path, gate = read_json_ref(evidence[key], root=path.parent, where=f"final {key} gate")
        workflow_native_paths[key] = gate_path
        workflow_native_evidence[key] = validate_workflow_native_gate(
            gate,
            path=gate_path,
            lane=lane,
            artifact_type=artifact_type,
            pass_prefix=pass_prefix,
            source_sha=source["git_sha"],
            candidate_tag=prerelease["release_candidate_tag"],
            where=f"final {key} gate",
        )
    workflow_evidence = workflow_native_evidence["workflow_policy"]
    native_evidence = workflow_native_evidence["native_operator_set"]
    bundles = require_object(workflow_evidence.get("bundles"), "final workflow bundles")
    for backend in BACKENDS:
        bundle = require_object(bundles.get(backend), f"final workflow bundle {backend}")
        package = prerelease["packages"][backend]
        require(
            bundle.get("asset_sha256") == package["asset_sha256"]
            and bundle.get("binary_sha256") == package["binary_sha256"]
            and bundle.get("workflow_run_id") == package["workflow_run_id"]
            and bundle.get("workflow_run_attempt") == package["workflow_run_attempt"]
            and bundle.get("staging_label") == package["staging_label"],
            f"final workflow/public {backend} byte/run identity differs",
        )
    workflow_runs = require_object(workflow_evidence.get("runs"), "final workflow runs")
    require(
        native_evidence.get("cuda_run") == workflow_runs.get("cuda"),
        "final native gate uses a different CUDA workflow run",
    )
    native_source = require_object(native_evidence.get("source_bundle"), "final native source bundle")
    native_set = require_object(native_evidence.get("native_set"), "final native set")
    require(
        native_set.get("source_revisions") == [native_source.get("bundle_id")],
        "final native set does not bind the checked source bundle",
    )
    native_root = workflow_native_paths["native_operator_set"].parent
    native_abi_path = workflow_native.resolve_portable_ref(
        native_evidence.get("cuda_abi"), native_root, "final native CUDA ABI"
    )
    native_abi = require_object(read_json(native_abi_path, "final native CUDA ABI"), "final native CUDA ABI")
    workflow_root = workflow_native_paths["workflow_policy"].parent
    cuda_bundle = require_object(bundles.get("cuda"), "final workflow CUDA bundle")
    cuda_zip_path = workflow_native.resolve_portable_ref(
        cuda_bundle.get("zip"), workflow_root, "final workflow CUDA Actions ZIP"
    )
    cuda_abi_name = f"{BACKENDS['cuda']['asset']}.abi.json"
    try:
        with zipfile.ZipFile(cuda_zip_path) as archive:
            names = archive.namelist()
            require(names.count(cuda_abi_name) == 1, "final workflow CUDA ABI member differs")
            workflow_cuda_abi = archive.read(cuda_abi_name)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ValidationError(f"cannot read final workflow CUDA ABI member: {exc}") from exc
    require(
        hashlib.sha256(workflow_cuda_abi).hexdigest() == file_sha256(native_abi_path),
        "final native CUDA ABI bytes differ from the Actions ZIP ABI",
    )
    cuda_package = prerelease["packages"]["cuda"]
    require(
        native_abi.get("asset_sha256") == cuda_package["asset_sha256"]
        and native_abi.get("binary_sha256") == cuda_package["binary_sha256"]
        and normalized_positive_int(
            native_abi.get("workflow_run_id"), "final native CUDA ABI workflow_run_id"
        )
        == cuda_package["workflow_run_id"]
        and normalized_positive_int(
            native_abi.get("workflow_run_attempt"),
            "final native CUDA ABI workflow_run_attempt",
        )
        == cuda_package["workflow_run_attempt"]
        and native_abi.get("release_candidate_sha") == source["git_sha"],
        "final native CUDA ABI differs from the staged/public CUDA asset",
    )
    g0_refs = exact_fields(evidence["g0_gates"], {"unit", "metal", "cuda_full", "cuda_llama_dense"}, "final G0 source/accelerator gates")
    lanes = {
        "unit": ("unit", "unit.gate.json", "unit", "G0 SOURCE unit PASS: "),
        "metal": ("metal", "metal.gate.json", "metal", "G0 SOURCE metal PASS: "),
        "cuda_full": ("cuda-full", "g0_cuda4090_full.gate.json", "g0_cuda4090_full", "G0 SOURCE g0_cuda4090_full PASS: "),
        "cuda_llama_dense": ("cuda-llama-dense", "g0_cuda4090_llama_dense.gate.json", "g0_cuda4090_llama_dense", "G0 SOURCE g0_cuda4090_llama_dense PASS: "),
    }
    for key, (lane, child_filename, child_lane, child_pass_prefix) in lanes.items():
        child, outer_path, _ = validate_outer_child_gate(
            g0_refs[key],
            root=path.parent,
            g0_gate_paths=g0_gate_paths,
            lane=lane,
            child_filename=child_filename,
            child_pass_prefix=child_pass_prefix,
            source_sha=source["git_sha"],
            where=f"final G0 {lane} gate",
        )
        validate_lane_gate(
            child,
            lane=child_lane,
            source_sha=source["git_sha"],
            where=f"final G0 {lane} child gate",
        )
        deep_validate_source_gate(key, child, outer_path=outer_path, source_sha=source["git_sha"])
    artifact_dir = nonempty(data["artifact_dir"], "final artifact_dir")
    require(artifact_dir == FINAL_ARTIFACT_DIR, "final artifact_dir is not the goal's canonical directory")
    pass_line = f"FERRUM 0.8.4 RELEASE PASS: {FINAL_ARTIFACT_DIR}"
    require(data["pass_line"] == pass_line, "final exact terminal PASS line differs")
    return {"source": source, "release": expected_identity, "pass_line": pass_line}


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_json(path: Path, payload: Any) -> None:
    write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def make_ref(path: Path, root: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def write_outer_gate_fixture(
    outer_path: Path,
    child_path: Path,
    document: dict[str, Any],
    *,
    started_at: str = "2026-09-02T00:00:00+00:00",
    finished_at: str = "2026-09-02T00:00:01+00:00",
) -> None:
    root = outer_path.parent
    delegated = document.setdefault("delegated_command_line", ["fixture-child"])
    execution = {
        "cmd": delegated,
        "cwd": "/remote/ferrum",
        "timeout_seconds": 120,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": 1.0,
        "returncode": 0,
        "env_overrides": {"PYTHONDONTWRITEBYTECODE": "1"},
    }
    write_json(root / "run_gate.child.command.json", execution)
    write_bytes(root / "run_gate.child.stdout", (str(document.get("child_pass_line", "PASS")) + "\n").encode())
    write_bytes(root / "run_gate.child.stderr", b"")
    document["started_at"] = started_at
    document["finished_at"] = finished_at
    document["child_execution_artifacts"] = [
        {"path": name, "sha256": file_sha256(root / name), "size_bytes": (root / name).stat().st_size}
        for name in ("run_gate.child.command.json", "run_gate.child.stdout", "run_gate.child.stderr")
    ]
    artifact_dir = document["artifact_dir"]
    child_name = child_path.name
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path != outer_path:
            rows.append({"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
    document["child_artifacts"] = {
        "child_manifest": {
            "path": str(PurePosixPath(artifact_dir) / child_name),
            "sha256": file_sha256(child_path),
            "size_bytes": child_path.stat().st_size,
        },
        "artifact_tree": {
            "schema_version": 1,
            "kind": "standard-g0-regular-file-tree",
            "file_count": len(rows),
            "total_size_bytes": sum(row["size_bytes"] for row in rows),
            "files": rows,
            "sha256": pretty_json_sha256(rows),
        },
    }
    write_json(outer_path, document)


def fixture_binary_checks() -> dict[str, Any]:
    return {
        "version": True,
        "cli": {"passed": True, "has_context": True, "has_math": True, "disable_thinking": True},
        "serve": {
            "math": [200, "579"],
            "multiturn": [200, "ferrum-blue"],
            "boundary_status": 400,
            "disable_thinking": True,
            "strict_json": [200, '{"answer":579}'],
            "tool_call": [200, "tool_calls"],
            "stream": [200, 1],
        },
    }


def build_tar_binary_child_fixture(
    gate_dir: Path,
    *,
    mode: str,
    source_asset: Path,
    public: bool,
    recorded_artifact_dir: str,
) -> dict[str, Any]:
    asset_name = source_asset.name
    expected_sha = file_sha256(source_asset)
    metadata: dict[str, Any] = {}
    binary = binary_gate.prepare_tarball(
        VERSION,
        asset_name,
        gate_dir,
        expected_sha,
        source_asset,
        metadata,
    )
    if public:
        canonical = binary_gate.official_asset(VERSION, asset_name)
        now = time.time()
        source_progress = binary_gate.selftest_progress_ref(
            gate_dir, "asset.download.progress.jsonl", (gate_dir / asset_name).stat().st_size
        )
        source_receipt = gate_dir / "asset.source.receipt.json"
        attempt = {
            "attempt": 1,
            "requested_url": canonical,
            "effective_url": canonical,
            "effective_url_sha256": None,
            "http_status": 200,
            "response_headers": {},
            "received_size_bytes": (gate_dir / asset_name).stat().st_size,
            "error": None,
            **binary_gate.timing_receipt(now, now + 10, now, 0),
        }
        binary_gate.write_json(
            source_receipt,
            {
                "source": "public-url",
                "http_performed": True,
                "requested_url": canonical,
                "requested_path": None,
                "effective_url": canonical,
                "effective_url_sha256": None,
                "http_status": 200,
                "response_headers": {},
                "received_size_bytes": (gate_dir / asset_name).stat().st_size,
                "attempts": [attempt],
                "progress": source_progress,
                **binary_gate.timing_receipt(now, now + 10, now, 0),
            },
        )
        checksum_url = binary_gate.official_asset(VERSION, f"{asset_name}.sha256")
        checksum_progress = binary_gate.selftest_progress_ref(
            gate_dir,
            "asset.checksum.download.progress.jsonl",
            (gate_dir / f"{asset_name}.sha256").stat().st_size,
        )
        checksum_receipt = gate_dir / "asset.checksum.receipt.json"
        checksum_attempt = dict(attempt)
        checksum_attempt.update(
            {
                "requested_url": checksum_url,
                "effective_url": checksum_url,
                "received_size_bytes": (gate_dir / f"{asset_name}.sha256").stat().st_size,
            }
        )
        binary_gate.write_json(
            checksum_receipt,
            {
                "source": "public-url",
                "http_performed": True,
                "requested_url": checksum_url,
                "requested_path": None,
                "effective_url": checksum_url,
                "effective_url_sha256": None,
                "http_status": 200,
                "response_headers": {},
                "received_size_bytes": (gate_dir / f"{asset_name}.sha256").stat().st_size,
                "attempts": [checksum_attempt],
                "progress": checksum_progress,
                **binary_gate.timing_receipt(now, now + 10, now, 0),
            },
        )
        metadata.update(
            {
                "source": "public-url",
                "classification": "canonical-public-release",
                "requested_url": canonical,
                "requested_path": None,
                "source_receipt": binary_gate.evidence_ref(gate_dir, source_receipt),
            }
        )
        metadata["checksum"]["source"] = "public-url"
        metadata["checksum"]["receipt"] = binary_gate.evidence_ref(
            gate_dir, checksum_receipt
        )
    commands = {
        "version": binary_gate.selftest_command_bundle(
            gate_dir,
            "version",
            stdout=f"ferrum {VERSION}\n",
            command=[str(binary), "--version"],
        ),
        "cli": binary_gate.selftest_command_bundle(
            gate_dir,
            "cli",
            command=[str(binary), "run", "selftest-model", "--disable-thinking"],
        ),
        "serve": binary_gate.selftest_serve_evidence(
            gate_dir, "tar", binary_path=str(binary)
        ),
    }
    if mode == "cuda-tarball":
        commands["ldd"] = binary_gate.selftest_command_bundle(
            gate_dir, "ldd", command=["ldd", str(binary)]
        )
    started = time.time()
    gate = binary_gate.write_gate(
        gate_dir,
        mode,
        VERSION,
        binary_gate.selftest_checks(),
        evidence={"asset": metadata, "commands": commands},
        started=started,
        deadline=started + 60,
    )
    gate["artifact_dir"] = recorded_artifact_dir
    gate["pass_line"] = binary_gate.PASS_PREFIXES[mode] + recorded_artifact_dir
    return gate


def build_homebrew_binary_child_fixture(
    gate_dir: Path,
    *,
    mode: str,
    source_asset: Path,
    recorded_artifact_dir: str,
) -> dict[str, Any]:
    gate_dir.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(source_asset)
    formula_document = binary_gate.selftest_formula_document(mode, VERSION, digest)
    formula_path = gate_dir / "brew_info.json"
    write_json(formula_path, formula_document)
    formula = {
        "brew_info": binary_gate.evidence_ref(gate_dir, formula_path),
        "identity": binary_gate.parse_formula_info(
            formula_document, mode=mode, version=VERSION
        ),
    }
    spec = binary_gate.FORMULAE[mode]
    if mode == "homebrew-metal":
        captured = gate_dir / "installed/ferrum"
        with tarfile.open(source_asset, "r:gz") as archive:
            member = archive.extractfile("ferrum")
            require(member is not None, "Homebrew fixture source lacks ferrum")
            write_bytes(captured, member.read())
        command_v_path = "/opt/homebrew/bin/ferrum"
        identity_path = gate_dir / "installed_binary.json"
        write_json(
            identity_path,
            {
                "command_v_path": command_v_path,
                "resolved_path": "/opt/homebrew/Cellar/ferrum/0.8.4/bin/ferrum",
                "command_v_is_symlink": True,
                "sha256": file_sha256(captured),
                "size_bytes": captured.stat().st_size,
                "captured_binary": binary_gate.evidence_ref(gate_dir, captured),
            },
        )
        commands = {
            "reinstall": binary_gate.selftest_command_bundle(
                gate_dir, "brew-reinstall", command=["brew", "reinstall", spec["formula"]]
            ),
            "brew_info": binary_gate.selftest_command_bundle(
                gate_dir,
                "brew-info",
                stdout=json.dumps(formula_document),
                command=["brew", "info", "--json=v2", spec["formula"]],
                stdout_path=formula_path,
            ),
            "command_v": binary_gate.selftest_command_bundle(
                gate_dir,
                "command-v",
                stdout=command_v_path + "\n",
                command=["/bin/sh", "-c", "command -v ferrum"],
            ),
            "version": binary_gate.selftest_command_bundle(
                gate_dir,
                "brew-version",
                stdout=f"ferrum {VERSION}\n",
                command=[command_v_path, "--version"],
            ),
            "help": binary_gate.selftest_command_bundle(
                gate_dir,
                "brew-help",
                stdout="Ferrum usage\n",
                command=[command_v_path, "--help"],
            ),
            "cli": binary_gate.selftest_command_bundle(
                gate_dir,
                "brew-cli",
                command=[command_v_path, "run", "selftest-model", "--disable-thinking"],
            ),
            "serve": binary_gate.selftest_serve_evidence(
                gate_dir, "brew", binary_path=command_v_path
            ),
        }
        formula["brew_info"] = binary_gate.evidence_ref(gate_dir, formula_path)
        checks = binary_gate.selftest_checks(homebrew=True)
        evidence = {
            "formula": formula,
            "installed_binary": binary_gate.evidence_ref(gate_dir, identity_path),
            "commands": commands,
        }
    else:
        captured = gate_dir / spec["asset"]
        write_bytes(captured, source_asset.read_bytes())
        reported_path = f"/home/linuxbrew/.cache/Homebrew/downloads/{spec['asset']}"
        identity_path = gate_dir / "fetched_archive.json"
        write_json(
            identity_path,
            {
                "reported_path": reported_path,
                "resolved_path": reported_path,
                "sha256": file_sha256(captured),
                "size_bytes": captured.stat().st_size,
                "captured_archive": binary_gate.evidence_ref(gate_dir, captured),
            },
        )
        commands = {
            "fetch": binary_gate.selftest_command_bundle(
                gate_dir, "brew-fetch", command=["brew", "fetch", "--force", spec["formula"]]
            ),
            "brew_info": binary_gate.selftest_command_bundle(
                gate_dir,
                "brew-info",
                stdout=json.dumps(formula_document),
                command=["brew", "info", "--json=v2", spec["formula"]],
                stdout_path=formula_path,
            ),
            "brew_cache": binary_gate.selftest_command_bundle(
                gate_dir,
                "brew-cache",
                stdout=reported_path + "\n",
                command=["brew", "--cache", spec["formula"]],
            ),
        }
        formula["brew_info"] = binary_gate.evidence_ref(gate_dir, formula_path)
        checks = {"fetch": True, "formula_version": VERSION}
        evidence = {
            "formula": formula,
            "fetched_archive": binary_gate.evidence_ref(gate_dir, identity_path),
            "commands": commands,
        }
    started = time.time()
    gate = binary_gate.write_gate(
        gate_dir,
        mode,
        VERSION,
        checks,
        evidence=evidence,
        started=started,
        deadline=started + 60,
    )
    gate["artifact_dir"] = recorded_artifact_dir
    gate["pass_line"] = binary_gate.PASS_PREFIXES[mode] + recorded_artifact_dir
    return gate


def build_e2e_evidence_fixture(e2e_dir: Path, *, backend: str, alias: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    evidence: dict[str, Any] = {}
    recorded_root = f"/remote/readme-e2e-{backend}"
    extracted_binary = e2e_dir / "runtime/ferrum"
    write_bytes(extracted_binary, f"fixture-ferrum-{backend}".encode())
    network_refs: dict[str, Any] = {}
    child_network_document: dict[str, Any] | None = None
    for network_label, consumer in {
        "urllib_public_downloads": "urllib-public-github-downloads",
        "child_processes": "ferrum-child-processes",
    }.items():
        network_path = e2e_dir / "network-environment" / f"{network_label}.json"
        variables = []
        if network_label == "child_processes":
            variables = [{"key": "HTTPS_PROXY", "value_sha256": "8" * 64, "loopback": False, "custom_ca": False}]
        document = {
            "schema_version": 1,
            "artifact_type": "ferrum_v084_sanitized_network_environment_receipt",
            "consumer": consumer,
            "secret_values_recorded": False,
            "variables": variables,
        }
        write_json(network_path, document)
        network_refs[network_label] = make_ref(network_path, e2e_dir)
        if network_label == "child_processes":
            child_network_document = document
    require(child_network_document is not None, "fixture child network receipt missing")
    raw_process_refs: dict[str, dict[str, Any]] = {}
    for label, argv in {
        "binary_version": [f"{recorded_root}/runtime/ferrum", "--version"],
        "binary_help": [f"{recorded_root}/runtime/ferrum", "--help"],
        "doctor": [f"{recorded_root}/runtime/ferrum", "doctor", alias],
        "run": [f"{recorded_root}/runtime/ferrum", "run", alias, "--disable-thinking"],
        "serve": [f"{recorded_root}/runtime/ferrum", "serve", "--model", alias, "--disable-thinking"],
    }.items():
        command_root = e2e_dir / "commands" / label
        stdout = command_root / "stdout.log"
        stderr = command_root / "stderr.log"
        progress = command_root / "progress.jsonl"
        stdout_payload = {
            "binary_version": f"ferrum {VERSION}\n",
            "binary_help": "Ferrum usage and commands\n",
            "doctor": f"model {alias} is ready\n",
            "run": json.dumps({"event": "assistant", "content": "hello from Ferrum"}) + "\n",
            "serve": "server ready\n",
        }[label]
        write_bytes(stdout, stdout_payload.encode())
        write_bytes(stderr, b"")
        write_bytes(progress, (json.dumps({"observed_at": "2026-09-02T00:00:14+00:00", "elapsed_seconds": 0.1, "returncode": 0}) + "\n").encode())
        command_path = command_root / "command.json"
        write_json(command_path, {
            "schema_version": 1, "kind": "bounded_child_process", "label": label,
            "command": argv, "cwd": recorded_root, "environment": {"overrides": {}, "network_routing": child_network_document}, "stdin": None,
            "stdout_log": "stdout.log", "stderr_log": "stderr.log", "progress_log": "progress.jsonl",
            "progress_signal": "log bytes", "started_at": "2026-09-02T00:00:14+00:00",
            "timeout_seconds": 60, "status": "terminated" if label == "serve" else "pass",
            "returncode": None if label == "serve" else 0, "finished_at": "2026-09-02T00:00:15+00:00", "duration_seconds": 1.0,
            "stdout": make_ref(stdout, e2e_dir), "stderr": make_ref(stderr, e2e_dir), "error": None,
            **({"cleanup_precondition": {"process_alive": True, "observed_at": "2026-09-02T00:00:15+00:00"}} if label == "serve" else {}),
        })
        raw_process_refs[label] = {
            "command": make_ref(command_path, e2e_dir),
            "stdout": make_ref(stdout, e2e_dir),
            "stderr": make_ref(stderr, e2e_dir),
            "progress": make_ref(progress, e2e_dir),
            "stdin": None,
        }
        wrapper_path = e2e_dir / "portable-process-receipts" / f"{label}.json"
        write_json(
            wrapper_path,
            {
                "schema_version": 1,
                "artifact_type": "ferrum_v084_portable_process_receipt",
                "label": {"binary_version": "binary-version", "binary_help": "binary-help", "doctor": "doctor-model", "run": "readme-run", "serve": "readme-serve"}[label],
                "status": "terminated" if label == "serve" else "pass",
                "returncode": None if label == "serve" else 0,
                **raw_process_refs[label],
                "extracted_binary": make_ref(extracted_binary, e2e_dir),
                "network_environment": network_refs["child_processes"],
            },
        )
        evidence[label] = make_ref(wrapper_path, e2e_dir)

    model_files = [{"name": "model.safetensors", "size_bytes": 8}]
    model_download = e2e_dir / "cold-cache-model-download.json"
    write_json(model_download, {
        "schema_version": 1, "artifact_type": "ferrum_v084_cold_cache_model_download_receipt",
        "status": "pass", "backend": backend, "model_alias": alias, "source": "https://huggingface.co",
        "cache_root": f"/tmp/{backend}-cache", "fresh_cache": True, "download_complete": True,
        "download_size_marker": "2.55 GiB" if backend == "metal" else "8.7 GiB",
        "repositories": [{"repository": "fixture/model", "revision": "3" * 40, "files": model_files, "files_metadata_sha256": canonical_sha256(model_files)}],
        "execution": {"started_at": "2026-09-02T00:00:14+00:00", "finished_at": "2026-09-02T00:00:15+00:00", "timeout_seconds": 60, "progress_signal": "cache growth"},
        "run_process": raw_process_refs["run"],
    })
    evidence["download"] = make_ref(model_download, e2e_dir)

    responses = {
        "models": json.dumps({"object": "list", "data": [{"id": "ferrum"}]}),
        "chat": json.dumps({"choices": [{"message": {"content": "hello"}}]}),
        "stream": 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n',
    }
    for label, response in responses.items():
        http_root = e2e_dir / "http" / label
        response_path = http_root / "response.body"
        write_bytes(response_path, response.encode())
        exchange_path = http_root / "exchange.json"
        write_json(exchange_path, {
            "schema_version": 1, "kind": "bounded_local_http_exchange", "label": label,
            "method": "GET" if label == "models" else "POST", "url": f"http://127.0.0.1:18080/{label}",
            "timeout_seconds": 30, "started_at": "2026-09-02T00:00:14+00:00",
            "finished_at": "2026-09-02T00:00:15+00:00", "duration_seconds": 1.0,
            "status": 200, "request": None, "response": make_ref(response_path, e2e_dir),
            "response_content_type": "text/event-stream" if label == "stream" else "application/json",
        })
        evidence[label] = make_ref(exchange_path, e2e_dir)

    log_path = e2e_dir / "readme-e2e-log-scan.json"
    scanned = e2e_dir / "commands/run/stdout.log"
    write_json(log_path, {
        "schema_version": 1, "artifact_type": "ferrum_v084_readme_e2e_log_scan", "status": "pass",
        "forbidden_patterns": ["panic", "oom", "cuda error", "metal error", "invalid utf-8", "<unk>", "[pad]", "control-token"],
        "found": [], "files": [{"label": "run stdout", "file": make_ref(scanned, e2e_dir)}],
    })
    evidence["logs"] = make_ref(log_path, e2e_dir)
    return evidence, network_refs, recorded_root


def make_tar(path: Path, *, members: set[str], binary: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name in sorted(members):
            payload = binary if name == "ferrum" else f"fixture {name}\n".encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o755 if name == "ferrum" else 0o644
            archive.addfile(info, io.BytesIO(payload))


def expect_failure(callable_: Any, contains: str) -> None:
    try:
        callable_()
    except ValidationError as exc:
        require(contains in str(exc), f"negative self-test failed for wrong reason: {exc}")
    else:
        raise ValidationError(f"negative self-test unexpectedly passed: {contains}")


def build_crates_publish_fixture(root: Path, candidate_sha: str) -> Path:
    prepublish_root = root / "prepublish"
    prepublish_path = crates_release.build_fixture_prepublish(prepublish_root)
    prepublish, _ = crates_release.validate_prepublish_manifest(prepublish_path)
    require(
        prepublish["release_candidate"]["git_sha"] == candidate_sha,
        "crates fixture candidate differs",
    )
    out = root / "publish"
    out.mkdir(parents=True)
    receipt = crates_release.new_publish_receipt(prepublish, prepublish_path)
    binding, _copied_prepublish = crates_release.copy_prepublish_evidence(
        prepublish=prepublish, prepublish_path=prepublish_path, out=out
    )
    receipt["prepublish"] = copy.deepcopy(binding)
    for package, row in zip(prepublish["packages"], receipt["packages"]):
        checksum = package["archive"]["sha256"]
        row["state"] = "visible"
        row["disposition"] = "offline-selftest-visible"
        row["visibility"] = {
            "visible": True,
            "api": {"checksum": checksum},
            "index": {"checksum": checksum},
        }
        row["visibility_observations"] = [copy.deepcopy(row["visibility"])]
    receipt["status"] = "pass"

    resolution_root = out / "clean-resolution/attempt-1"
    resolution_root.mkdir(parents=True)
    lock_path = resolution_root / "Cargo.lock"
    write_bytes(lock_path, b"# offline crates.io lock fixture\n")
    metadata_path = resolution_root / "metadata.json"
    resolved = [
        {
            "name": name,
            "version": VERSION,
            "source": "registry+https://github.com/rust-lang/crates.io-index",
        }
        for name in sorted(EXPECTED_CRATES)
    ]
    write_json(metadata_path, {"packages": resolved})
    resolution = {
        "status": "pass",
        "cargo_lock": crates_release.artifact_ref(lock_path, root=out),
        "metadata": crates_release.artifact_ref(metadata_path, root=out),
        "resolved": resolved,
        "commands": {
            "generate_lockfile": crates_release.fixture_command_receipt(
                out, name="generate-lockfile", argv=["cargo", "generate-lockfile"]
            ),
            "metadata": crates_release.fixture_command_receipt(
                out, name="registry-metadata", argv=["cargo", "metadata", "--locked", "--format-version", "1"]
            ),
        },
    }
    receipt["clean_resolution"] = {"state": "pass", "result": resolution}

    install_root = out / "clean-install/attempt-1"
    install_root.mkdir(parents=True)
    binary = install_root / "root/bin/ferrum"
    version_stdout = install_root / "version.stdout"
    help_stdout = install_root / "help.stdout"
    write_bytes(binary, b"offline ferrum binary fixture\n")
    write_bytes(version_stdout, f"ferrum {VERSION}\n".encode())
    write_bytes(help_stdout, b"Usage: ferrum [COMMAND]\n")
    install = {
        "status": "pass",
        "command": ["cargo", "install", "ferrum-cli", "--version", VERSION, "--locked", "--root", str((install_root / "root").resolve()), "--target-dir", str((install_root / "target").resolve())],
        "binary": crates_release.artifact_ref(binary, root=out),
        "binary_sha256": file_sha256(binary),
        "version_stdout": crates_release.artifact_ref(version_stdout, root=out),
        "help_stdout": crates_release.artifact_ref(help_stdout, root=out),
        "commands": {
            "install": crates_release.fixture_command_receipt(
                out, name="install", argv=["cargo", "install", "ferrum-cli", "--version", VERSION, "--locked", "--root", str((install_root / "root").resolve()), "--target-dir", str((install_root / "target").resolve())]
            ),
            "version": crates_release.fixture_command_receipt(
                out, name="installed-version", argv=[str(binary.resolve()), "--version"]
            ),
            "help": crates_release.fixture_command_receipt(
                out, name="installed-help", argv=[str(binary.resolve()), "--help"]
            ),
        },
    }
    receipt["install"] = {"state": "pass", "result": install}
    crates_release.persist_receipt(out, receipt)
    receipt_path = out / "publish.resume.json"

    publish_rows = []
    for package, row in zip(prepublish["packages"], receipt["packages"]):
        checksum = package["archive"]["sha256"]
        publish_rows.append(
            {
                "position": package["position"],
                "name": package["name"],
                "version": VERSION,
                "archive_sha256": checksum,
                "crates_io_visible": True,
                "disposition": row["disposition"],
                "api_checksum": checksum,
                "index_checksum": checksum,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "ferrum_v084_crates_io_publish_manifest",
        "status": "pass",
        "lane": "runtime-vnext-crates-io",
        "version": VERSION,
        "canonical": True,
        "release_candidate": prepublish["release_candidate"],
        "prepublish": binding,
        "publish_order": prepublish["topology"]["order"],
        "packages": publish_rows,
        "cargo_workspace_crates": [
            {"name": row["name"], "version": VERSION, "crates_io_visible": True}
            for row in publish_rows
        ],
        "clean_resolution": resolution,
        "install": install,
        "resume_receipt": crates_release.artifact_ref(receipt_path, root=out),
        "created_at": "2026-09-02T00:04:00+00:00",
        "credential_policy": {
            "source": "existing-cargo-config-or-environment",
            "secret_values_recorded": False,
            "token_cli_arguments": False,
        },
        "artifact_dir": str(out.resolve()),
        "manifest_id": "",
        "pass_line": f"FERRUM CRATES IO V0.8.4 PASS: {out.resolve()}",
    }
    manifest["manifest_id"] = crates_release.manifest_identity(
        manifest,
        (
            "schema_version",
            "artifact_type",
            "version",
            "release_candidate",
            "prepublish",
            "publish_order",
            "packages",
            "clean_resolution",
            "install",
            "artifact_dir",
        ),
    )
    path = out / "crates-io.manifest.json"
    write_json(path, manifest)
    crates_release.validate_publish_manifest(path)
    return path


def build_workflow_native_fixture(
    root: Path,
    candidate_sha: str,
    candidate_tag: str,
    public_files: Path,
) -> dict[str, Path]:
    source_root = root / "source"
    source_root.mkdir(parents=True)
    release_run_id = 8401
    cuda_run_id = 8402
    zip_paths: dict[str, Path] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for backend in workflow_native.BACKENDS:
        run_id = release_run_id if backend in {"cpu", "metal"} else cuda_run_id
        directory = source_root / "staged" / backend
        directory.mkdir(parents=True)
        for name in workflow_native.expected_bundle_names(backend):
            write_bytes(directory / name, (public_files / name).read_bytes())
        zip_path = source_root / f"{backend}.zip"
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member in sorted(directory.iterdir()):
                archive.write(member, arcname=member.name)
        asset = str(workflow_native.BACKENDS[backend]["asset"])
        zip_paths[backend] = zip_path
        artifacts[backend] = {
            "id": 8000 + list(workflow_native.BACKENDS).index(backend),
            "name": f"{asset.removesuffix('.tar.gz')}-v084-selftest-{candidate_sha}",
            "size_in_bytes": zip_path.stat().st_size,
            "digest": f"sha256:{file_sha256(zip_path)}",
            "expired": False,
            "workflow_run": {"id": run_id, "head_sha": candidate_sha},
        }
    documents = {
        "release_run": workflow_native.fake_run(
            release_run_id,
            str(workflow_native.BACKENDS["cpu"]["workflow"]),
            candidate_sha,
        ),
        "cuda_run": workflow_native.fake_run(
            cuda_run_id,
            str(workflow_native.BACKENDS["cuda"]["workflow"]),
            candidate_sha,
        ),
        "release_jobs": workflow_native.fake_jobs(
            ("cpu", "metal"), run_id=release_run_id, candidate_sha=candidate_sha
        ),
        "cuda_jobs": workflow_native.fake_jobs(
            ("cuda",), run_id=cuda_run_id, candidate_sha=candidate_sha
        ),
        "release_artifacts": {
            "total_count": 2,
            "artifacts": [artifacts["cpu"], artifacts["metal"]],
        },
        "cuda_artifacts": {"total_count": 1, "artifacts": [artifacts["cuda"]]},
    }
    for name, document in documents.items():
        write_json(source_root / f"{name}.json", document)
    workflow_args = argparse.Namespace(
        candidate_sha=candidate_sha,
        candidate_tag=candidate_tag,
        release_run=source_root / "release_run.json",
        release_jobs=source_root / "release_jobs.json",
        release_artifacts=source_root / "release_artifacts.json",
        cuda_run=source_root / "cuda_run.json",
        cuda_jobs=source_root / "cuda_jobs.json",
        cuda_artifacts=source_root / "cuda_artifacts.json",
        cpu_zip=zip_paths["cpu"],
        metal_zip=zip_paths["metal"],
        cuda_zip=zip_paths["cuda"],
        staged_root=source_root / "staged",
    )
    workflow_out = root / "workflow-policy"
    workflow_evidence = workflow_native.materialize_workflow_evidence(
        workflow_out,
        workflow_args,
        workflow_native.validate_workflow_policy(workflow_args, verify_checkout=False),
    )
    workflow_pass = f"{workflow_native.WORKFLOW_PASS_PREFIX}: {workflow_out}"
    workflow_native.write_gate_manifest(
        workflow_out,
        lane="release-workflow-policy",
        pass_line=workflow_pass,
        evidence=workflow_evidence,
        started_at="2026-09-02T00:00:00+00:00",
        finished_at="2026-09-02T00:00:01+00:00",
    )
    workflow_path = workflow_out / "gate.manifest.json"
    workflow_native.validate_workflow_policy_manifest(workflow_path)

    bundle_source_root = root / "native-source"
    definition_root = root / "native-definitions"
    (bundle_source_root / "kernels").mkdir(parents=True)
    definition_root.mkdir()
    write_bytes(bundle_source_root / "kernels/fixture.cu", b"// fixture native source\n")
    write_json(
        definition_root / "fixture.json",
        {
            "schema_version": 3,
            "operator": "ferrum.cuda.fixture",
            "translation_units": ["kernels/fixture.cu"],
            "headers": [],
        },
    )
    source_archive = root / "ferrum-native-cuda-sources-fixture.tar.gz"
    source_manifest_path = root / "native-source-manifest.json"
    workflow_native.source_bundle.create(
        argparse.Namespace(
            source_root=bundle_source_root,
            definition_root=definition_root,
            archive=source_archive,
            manifest=source_manifest_path,
            github_repository=workflow_native.GITHUB_REPOSITORY,
            github_tag="ferrum-native-cuda12.4-sm89-v6",
        )
    )
    source_manifest = read_json(source_manifest_path, "fixture native source manifest")
    native_root = root / "native-archive-root/inputs"
    lock = workflow_native.native_set.create_selftest_native_operator_set(
        native_root,
        workflow_native.REQUIRED_CUDA_NATIVE_OPERATORS,
    )
    lock_document = read_json(lock, "fixture native lock")
    for row in lock_document["artifacts"]:
        manifest_path = native_root / row["manifest_path"]
        write_json(
            manifest_path,
            {
                "schema_version": 3,
                "source_package": {"revision": source_manifest["bundle_id"]},
            },
        )
        row["manifest"] = {
            "path": row["manifest"]["path"],
            "sha256": file_sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }
    write_json(lock, lock_document)
    workflow_native.native_set.validate_native_operator_set(
        lock,
        workflow_native.REQUIRED_CUDA_NATIVE_OPERATORS,
    )
    raw_tar = root / "native-set.tar"
    with tarfile.open(raw_tar, mode="w") as archive:
        archive.add(native_root, arcname="inputs")
    native_archive = root / "native-operator-set-cuda12.4-sm89-v6.tar.zst"
    compressed = __import__("subprocess").run(
        ["zstd", "--quiet", "--force", str(raw_tar), "-o", str(native_archive)],
        text=True,
        stdout=__import__("subprocess").PIPE,
        stderr=__import__("subprocess").PIPE,
        check=False,
    )
    require(compressed.returncode == 0, f"native fixture zstd failed: {compressed.stderr}")
    cuda_workflow_text = re.sub(
        r"(?m)^(\s*NATIVE_OPERATOR_SET_ARCHIVE_SHA256:\s*)[0-9a-f]{64}\s*$",
        rf"\g<1>{file_sha256(native_archive)}",
        (RELEASE_DIR.parent.parent / ".github/workflows/release-cuda.yml").read_text(
            encoding="utf-8"
        ),
    )
    fixture_workflow_path = root / "release-cuda.fixture.yml"
    write_bytes(fixture_workflow_path, cuda_workflow_text.encode())
    workflow_document = workflow_native.workflow_policy.parse_workflow_yaml(
        cuda_workflow_text, "release-cuda.yml"
    )
    native_args = argparse.Namespace(
        candidate_sha=candidate_sha,
        candidate_tag=candidate_tag,
        source_bundle_manifest=source_manifest_path,
        source_bundle_archive=source_archive,
        native_set_archive=native_archive,
        native_set_lock=lock,
        cuda_run=source_root / "cuda_run.json",
        cuda_jobs=source_root / "cuda_jobs.json",
        cuda_abi_manifest=(
            source_root
            / "staged/cuda"
            / f"{workflow_native.BACKENDS['cuda']['asset']}.abi.json"
        ),
        workflow_path=fixture_workflow_path,
    )
    native_out = root / "native-set"
    native_evidence = workflow_native.materialize_native_evidence(
        native_out,
        native_args,
        workflow_native.validate_native_set(
            native_args,
            verify_checkout=False,
            validate_static_workflow=False,
            workflow_document=workflow_document,
        ),
    )
    native_pass = f"{workflow_native.NATIVE_PASS_PREFIX}: {native_out}"
    workflow_native.write_gate_manifest(
        native_out,
        lane="native-operator-set",
        pass_line=native_pass,
        evidence=native_evidence,
        started_at="2026-09-02T00:00:00+00:00",
        finished_at="2026-09-02T00:00:01+00:00",
    )
    native_path = native_out / "gate.manifest.json"
    workflow_native.validate_native_set_manifest(native_path)
    return {"workflow_policy": workflow_path, "native_operator_set": native_path}


def build_selftest_fixture(root: Path) -> tuple[Path, Path, Path]:
    candidate_sha = "a" * 40
    rc_tag = "v0.8.4-rc.1"
    release_id = 804
    files = root / "files"
    staged_files = root / "staged"
    public_refs: dict[str, Any] = {}
    staged_refs: dict[str, Any] = {}
    packages: dict[str, dict[str, str]] = {}
    for backend, spec in BACKENDS.items():
        asset = spec["asset"]
        binary = f"fixture-ferrum-{backend}".encode()
        tar_path = files / asset
        make_tar(tar_path, members=spec["tar_members"], binary=binary)
        asset_sha = file_sha256(tar_path)
        binary_sha = hashlib.sha256(binary).hexdigest()
        write_bytes(files / f"{asset}.sha256", f"{asset_sha}  {asset}\n".encode())
        write_bytes(files / f"{asset}.binary.sha256", f"{binary_sha}  ferrum\n".encode())
        common = {
            "schema_version": 1, "asset_name": asset, "asset_sha256": asset_sha,
            "binary_name": "ferrum", "binary_sha256": binary_sha,
            "release_candidate_sha": candidate_sha, "release_candidate_tag": rc_tag,
            "staging_label": "v084-selftest",
            "workflow_run_id": "8402" if backend == "cuda" else "8401",
            "workflow_run_attempt": "1",
        }
        audit_name = asset.removesuffix(".tar.gz") + ".dependencies.txt"
        write_bytes(files / audit_name, f"fixture native dependencies for {backend}\nlibc\n".encode())
        audit_sha = file_sha256(files / audit_name)
        write_json(files / f"{asset}.version.json", {**common, "version": VERSION})
        write_json(files / f"{asset}.dependency.json", {**common, "audit_file": audit_name, "audit_sha256": audit_sha, "forbidden_runtime_linkage": ["python", "torch", "vllm"], "forbidden_runtime_linkage_found": False})
        abi = {**common, "target_triple": spec["target"], "backend": backend, "dependency_audit_sha256": audit_sha}
        if backend == "cuda":
            abi.update({"cuda_compute_capability": "89", "cuda_toolkit_image": "nvidia/cuda:12.4.0-devel-ubuntu22.04", "cargo_features": ["cuda", "vllm-moe-marlin", "vllm-paged-attn-v2"]})
        write_json(files / f"{asset}.abi.json", abi)
        for name in (asset, *(asset + suffix for suffix in SIDECAR_SUFFIXES), audit_name):
            public_path = files / name
            staged_path = staged_files / name
            write_bytes(staged_path, public_path.read_bytes())
            public_refs[name] = {
                "url": f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{quote(name)}",
                "http_status": 200,
                "file": make_ref(public_path, root),
            }
            staged_refs[name] = make_ref(staged_path, root)
        packages[backend] = {"asset_sha256": asset_sha, "binary_sha256": binary_sha}

    asset_rows = []
    for index, name in enumerate(sorted(EXPECTED_ASSETS), 1):
        file_path = files / name
        asset_rows.append({"id": 1000 + index, "name": name, "size": file_path.stat().st_size, "digest": "sha256:" + file_sha256(file_path), "browser_download_url": f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{quote(name)}"})
    before = {"id": release_id, "tag_name": TAG, "draft": False, "prerelease": True, "name": "Ferrum 0.8.4", "created_at": "2026-09-02T00:00:10+00:00", "published_at": "2026-09-02T00:00:11+00:00", "updated_at": "2026-09-02T00:00:11+00:00", "assets": asset_rows}
    release_path = root / "release-before.json"
    write_json(release_path, before)
    tag_path = root / "tag.json"
    write_json(tag_path, {"sha": "2" * 40, "tag": TAG, "object": {"type": "commit", "sha": candidate_sha}})
    tag_ref_path = root / "tag-ref.json"
    write_json(
        tag_ref_path,
        {"ref": f"refs/tags/{TAG}", "object": {"type": "tag", "sha": "2" * 40}},
    )
    rc_tag_path = root / "rc-tag.json"
    rc_tag_ref_path = root / "rc-tag-ref.json"
    write_json(rc_tag_path, {"sha": "3" * 40, "tag": rc_tag, "object": {"type": "commit", "sha": candidate_sha}})
    write_json(rc_tag_ref_path, {"ref": f"refs/tags/{rc_tag}", "object": {"type": "tag", "sha": "3" * 40}})

    release_by_name = {row["name"]: row for row in asset_rows}
    for name, entry in public_refs.items():
        downloaded = entry["file"]
        progress_path = root / "public-provenance" / f"{name}.progress.jsonl"
        write_bytes(
            progress_path,
            (json.dumps({"observed_at": "2026-09-02T00:00:12+00:00", "elapsed_seconds": 0.1, "bytes_downloaded": downloaded["size_bytes"], "expected_bytes": downloaded["size_bytes"]}) + "\n" + json.dumps({"observed_at": "2026-09-02T00:00:13+00:00", "elapsed_seconds": 1.0, "bytes_downloaded": downloaded["size_bytes"], "expected_bytes": downloaded["size_bytes"], "complete": True}) + "\n").encode(),
        )
        source_path = root / "public-provenance" / f"{name}.source.json"
        source = {
            "schema_version": 1, "artifact_type": "ferrum_v084_portable_public_asset_source_receipt",
            "status": "pass", "asset_name": name,
            "asset": {"name": name, "size_bytes": downloaded["size_bytes"], "sha256": downloaded["sha256"], "browser_download_url": entry["url"]},
            "url": entry["url"], "effective_url": entry["url"], "http_status": 200,
            "started_at": "2026-09-02T00:00:12+00:00", "finished_at": "2026-09-02T00:00:13+00:00",
            "duration_seconds": 1.0, "timeout_seconds": 60, "progress_interval_seconds": 1,
            "download": downloaded, "progress": make_ref(progress_path, root),
            "source_receipt_sha256": "9" * 64,
        }
        write_json(source_path, source)
        wrapper_path = root / "public-provenance" / f"{name}.json"
        wrapper = {
            "schema_version": 1, "artifact_type": "ferrum_v084_public_asset_download_provenance",
            "status": "pass", "backend_lane": "cuda" if "cuda" in name else "metal",
            "asset_name": name, "url": entry["url"], "effective_url": entry["url"], "http_status": 200,
            "started_at": "2026-09-02T00:00:12+00:00", "finished_at": "2026-09-02T00:00:13+00:00",
            "duration_seconds": 1.0, "timeout_seconds": 60,
            "download": downloaded, "progress": make_ref(progress_path, root), "source_receipt": make_ref(source_path, root),
        }
        write_json(wrapper_path, wrapper)
        entry["receipt"] = make_ref(wrapper_path, root)
        entry["progress"] = make_ref(progress_path, root)

    prepublication_pairs: dict[str, dict[str, Path]] = {}
    for backend, mode in (("metal", "metal-tarball"), ("cuda", "cuda-tarball")):
        gate_dir = root / f"prepublication-{backend}"
        recorded_dir = f"/remote/release-evidence/{gate_dir.name}"
        child_path = gate_dir / "gate.json"
        outer_path = gate_dir / "gate.manifest.json"
        child_prefix = "METAL TARBALL GATE PASS: " if backend == "metal" else "CUDA TARBALL GATE PASS: "
        asset = BACKENDS[backend]["asset"]
        write_json(
            child_path,
            build_tar_binary_child_fixture(
                gate_dir,
                mode=mode,
                source_asset=staged_files / asset,
                public=False,
                recorded_artifact_dir=recorded_dir,
            ),
        )
        write_outer_gate_fixture(
            outer_path,
            child_path,
            {
                "schema_version": 1,
                "status": "pass",
                "lane": mode,
                "child_returncode": 0,
                "child_pass_line": child_prefix + recorded_dir,
                "delegated_command_line": [
                    "python3",
                    "scripts/release/release_binary_gate.py",
                    mode,
                    "--version",
                    VERSION,
                    "--out",
                    recorded_dir,
                    "--asset-path",
                    f"/staged/{asset}",
                    "--sha256",
                    packages[backend]["asset_sha256"],
                ],
                "git_sha": candidate_sha,
                "dirty_status": {"is_dirty": False, "status_short": []},
                "artifact_dir": recorded_dir,
                "pass_line": f"FERRUM GATE {mode} PASS: {recorded_dir}",
            },
        )
        prepublication_pairs[backend] = {"outer": outer_path, "child": child_path}

    e2e_refs: dict[str, Any] = {}
    for backend in ("metal", "cuda"):
        e2e_dir = root / f"e2e-{backend}"
        alias = "qwen3.5:4b-q4_k_m" if backend == "metal" else "qwen3.5:4b"
        evidence, network_environment, recorded_e2e_root = build_e2e_evidence_fixture(e2e_dir, backend=backend, alias=alias)
        model_file: dict[str, Any] = {
            "name": "model.safetensors",
            "size_bytes": 8,
        }
        if backend == "metal":
            model_file["sha256"] = "4" * 64
        summary = {
            "schema_version": 1, "artifact_type": "ferrum_v084_readme_e2e_summary",
            "status": "pass", "version": VERSION, "backend": backend,
            "source_git_sha": candidate_sha, "asset_name": BACKENDS[backend]["asset"],
            "asset_sha256": packages[backend]["asset_sha256"], "binary_sha256": packages[backend]["binary_sha256"],
            "model": {"alias": alias, "revision": "3" * 40, "files": [model_file]},
            "cold_cache": {"fresh_cache": True, "cache_root": f"/tmp/{backend}-cache", "undocumented_behavior_env": {"behavior_overrides": [], "network_routing_is_behavior_override": False, "network_environment": network_environment}, "download_size_announced": True, "download_complete": True},
            "execution": {"started_at": "2026-09-02T00:00:12+00:00", "finished_at": "2026-09-02T00:00:16+00:00", "deadline_seconds": 120, "progress_signal": "new log bytes and cache bytes"},
            "network_environment": network_environment,
            "checks": {"binary_version": True, "binary_help": True, "doctor": True, "run": {"exit_code": 0, "non_empty": True, "disable_thinking": True}, "serve": {"ready": True}, "models": {"http_status": 200, "model_present": True}, "chat": {"http_status": 200, "non_empty_content": True}, "stream": {"http_status": 200, "done_count": 1, "usage_chunks": 1, "output_tokens": 7}, "log_scan": {"forbidden_patterns": ["panic", "oom", "cuda error", "metal error", "invalid utf-8", "<unk>", "[pad]", "control-token"], "found": []}},
            "evidence": evidence, "artifact_dir": recorded_e2e_root,
            "pass_line": f"FERRUM 0.8.4 README E2E PASS: {backend} {recorded_e2e_root}",
        }
        summary_path = e2e_dir / "summary.json"
        write_json(summary_path, summary)
        e2e_refs[backend] = make_ref(summary_path, root)

    rows = github_asset_rows(before, prerelease=True, where="selftest release")
    prerelease = {
        "schema_version": 1, "artifact_type": "ferrum_v084_prerelease_manifest", "status": "pass", "version": VERSION,
        "started_at": "2026-09-02T00:00:00+00:00",
        "finished_at": "2026-09-02T00:00:20+00:00",
        "source": {"git_sha": candidate_sha, "dirty": False},
        "release": {"id": release_id, "tag": TAG, "release_candidate_tag": rc_tag, "asset_set_sha256": asset_set_sha256(rows)},
        "evidence": {
            "release_snapshot": make_ref(release_path, root),
            "tag_ref_snapshot": make_ref(tag_ref_path, root),
            "tag_snapshot": make_ref(tag_path, root),
            "staged_assets": staged_refs,
            "public_downloads": public_refs,
            "readme_e2e": e2e_refs,
            "prepublication_binary_gates": {
                backend: {name: make_ref(gate_path, root) for name, gate_path in pair.items()}
                for backend, pair in prepublication_pairs.items()
            },
        },
        "artifact_dir": "fixture/prerelease", "pass_line": "FERRUM 0.8.4 PRERELEASE DOWNLOAD PASS: fixture/prerelease",
    }
    prerelease_path = root / "prerelease.json"
    write_json(prerelease_path, prerelease)

    after = copy.deepcopy(before)
    after["prerelease"] = False
    after["updated_at"] = "2026-09-02T00:02:00Z"
    after_path = root / "release-after.json"
    latest_path = root / "release-latest.json"
    write_json(after_path, after)
    write_json(latest_path, after)
    mutation_path = root / "promotion-mutation.json"
    mutation_body = {"prerelease": False}
    write_json(
        mutation_path,
        {
            "schema_version": 1,
            "artifact_type": "ferrum_v084_github_promotion_mutation_receipt",
            "status": "confirmed",
            "method": "PATCH",
            "endpoint": f"/repos/{REPOSITORY}/releases/{release_id}",
            "body": mutation_body,
            "body_sha256": canonical_sha256(mutation_body),
            "release_id": release_id,
            "attempted_at": "2026-09-02T00:01:59Z",
            "confirmed_at": "2026-09-02T00:02:00Z",
            "confirmation": "patch-response",
            "ambiguous_outcome_recovered": False,
        },
    )
    promotion = {
        "schema_version": 1, "artifact_type": "ferrum_v084_promotion_manifest", "status": "pass", "version": VERSION,
        "source": {"git_sha": candidate_sha, "dirty": False},
        "release": {"id": release_id, "tag": TAG, "asset_set_sha256": asset_set_sha256(rows)},
        "evidence": {
            "prerelease_manifest": make_ref(prerelease_path, root),
            "mutation_receipt": make_ref(mutation_path, root),
            "release_before": make_ref(release_path, root),
            "release_after": make_ref(after_path, root),
            "latest_release": make_ref(latest_path, root),
            "tag_ref_snapshot": make_ref(tag_ref_path, root),
            "tag_snapshot": make_ref(tag_path, root),
        },
        "artifact_dir": "fixture/promotion", "pass_line": "FERRUM 0.8.4 PROMOTION PASS: fixture/promotion",
    }
    promotion_path = root / "promotion.json"
    write_json(promotion_path, promotion)

    gate_pairs: dict[str, dict[str, Path]] = {}
    simple_specs = {
        "metal_tarball": ("metal-tarball", "METAL TARBALL GATE PASS: "),
        "cuda_tarball": ("cuda-tarball", "CUDA TARBALL GATE PASS: "),
        "homebrew_metal": ("homebrew-metal", "HOMEBREW METAL GATE PASS: "),
        "homebrew_cuda_fetch": ("homebrew-cuda-fetch", "HOMEBREW CUDA FETCH GATE PASS: "),
    }
    for key, (mode, child_prefix) in simple_specs.items():
        gate_dir = root / f"gate-{key}"
        recorded_dir = f"/remote/release-evidence/{gate_dir.name}"
        child_path = gate_dir / "gate.json"
        outer_path = gate_dir / "gate.manifest.json"
        if mode in {"metal-tarball", "cuda-tarball"}:
            backend = "metal" if mode == "metal-tarball" else "cuda"
            write_json(
                child_path,
                build_tar_binary_child_fixture(
                    gate_dir,
                    mode=mode,
                    source_asset=files / BACKENDS[backend]["asset"],
                    public=True,
                    recorded_artifact_dir=recorded_dir,
                ),
            )
        else:
            backend = "metal" if mode == "homebrew-metal" else "cuda"
            write_json(
                child_path,
                build_homebrew_binary_child_fixture(
                    gate_dir,
                    mode=mode,
                    source_asset=files / BACKENDS[backend]["asset"],
                    recorded_artifact_dir=recorded_dir,
                ),
            )
        write_outer_gate_fixture(
            outer_path,
            child_path,
            {
                "schema_version": 1,
                "status": "pass",
                "lane": mode,
                "child_returncode": 0,
                "child_pass_line": child_prefix + recorded_dir,
                "delegated_command_line": ["python3", "scripts/release/release_binary_gate.py", mode, "--version", VERSION, "--out", recorded_dir],
                "git_sha": candidate_sha,
                "dirty_status": {"is_dirty": False, "status_short": []},
                "artifact_dir": recorded_dir,
                "pass_line": f"FERRUM GATE {mode} PASS: {recorded_dir}",
            },
            started_at="2026-09-02T00:03:00+00:00",
            finished_at="2026-09-02T00:03:01+00:00",
        )
        gate_pairs[key] = {"outer": outer_path, "child": child_path}
    crates_path = build_crates_publish_fixture(root / "crates", candidate_sha)
    workflow_native_paths = build_workflow_native_fixture(
        root / "workflow-native", candidate_sha, rc_tag, files
    )
    lane_pairs: dict[str, dict[str, Path]] = {}
    lane_specs = {
        "unit": ("unit", "unit.gate.json", "unit", "G0 SOURCE unit PASS: "),
        "metal": ("metal", "metal.gate.json", "metal", "G0 SOURCE metal PASS: "),
        "cuda_full": ("cuda-full", "g0_cuda4090_full.gate.json", "g0_cuda4090_full", "G0 SOURCE g0_cuda4090_full PASS: "),
        "cuda_llama_dense": ("cuda-llama-dense", "g0_cuda4090_llama_dense.gate.json", "g0_cuda4090_llama_dense", "G0 SOURCE g0_cuda4090_llama_dense PASS: "),
    }
    for key, (lane, child_name, child_lane, child_prefix) in lane_specs.items():
        gate_dir = root / f"g0-{key}"
        recorded_dir = f"/remote/release-evidence/{gate_dir.name}"
        child_path = gate_dir / child_name
        outer_path = gate_dir / "gate.manifest.json"
        if key == "cuda_full":
            shutil.copytree(
                RELEASE_DIR.parents[1] / "docs/release/g0/0.7.7/cuda-full",
                gate_dir,
                dirs_exist_ok=True,
            )
            recorded_prefix = "docs/release/g0/0.7.7/cuda-full/"

            def localize_cuda_fixture(value: Any) -> Any:
                if isinstance(value, str) and value.startswith(recorded_prefix):
                    return value.removeprefix(recorded_prefix)
                if isinstance(value, list):
                    return [localize_cuda_fixture(item) for item in value]
                if isinstance(value, dict):
                    localized = {
                        name: localize_cuda_fixture(item)
                        for name, item in value.items()
                    }
                    if "git_head" in localized:
                        localized["git_head"] = candidate_sha
                    if "git_status_short" in localized:
                        localized["git_status_short"] = []
                    return localized
                return value

            for json_path in gate_dir.rglob("*.json"):
                try:
                    document = read_json(json_path, "CUDA full fixture JSON")
                except ValidationError:
                    continue
                write_json(json_path, localize_cuda_fixture(document))
        elif key == "cuda_llama_dense":
            shutil.copytree(
                RELEASE_DIR.parents[1] / "docs/release/g0/0.7.7/cuda-llama-dense",
                gate_dir,
                dirs_exist_ok=True,
            )
            metadata_path = gate_dir / "metadata.json"
            metadata = read_json(metadata_path, "dense fixture metadata")
            metadata["git_dirty"] = False
            metadata["git_sha"] = candidate_sha
            write_json(metadata_path, metadata)
            for command_name in ("run.command.json", "serve.command.json"):
                command_path = gate_dir / command_name
                command = read_json(command_path, f"dense fixture {command_name}")
                require(isinstance(command, list), "dense fixture command differs")
                if "--disable-thinking" not in command:
                    command.append("--disable-thinking")
                write_json(command_path, command)
        elif key == "metal":
            models = []
            for model_key, cells in {
                "llama31_8b": (1, 8, 16),
                "qwen3_30b_a3b": (16,),
            }.items():
                model_cells = [
                    {
                        "concurrency": concurrency,
                        "quality": {
                            "passed": True,
                            "requests": 4,
                            "status_200": 4,
                            "marker_ok": 4,
                            "square_ok": 4,
                            "crosstalk": 0,
                            "length_finishes": 0,
                        },
                        "completed": 4,
                        "prompts": 4,
                        "failed": 0,
                        "output_throughput_tok_s": 1.0,
                        "ratio_to_readme": 1.0,
                        "not_regressed_90pct": True,
                    }
                    for concurrency in cells
                ]
                models.append(
                    {
                        "key": model_key,
                        "moe": model_key == "qwen3_30b_a3b",
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
                        "unsafe_batch_probe": {
                            "enabled": False,
                            "product_default": False,
                            "startup": {},
                            "quality": {},
                        },
                        "run": {"passed": True},
                        "cells": model_cells,
                    }
                )
            write_json(gate_dir / "metal-readme/summary.json", {"models": models})
        child = {"status": "pass", "lane": child_lane}
        if key == "unit":
            stdout_path = gate_dir / "unit.stdout"
            stderr_path = gate_dir / "unit.stderr"
            source_receipt_path = gate_dir / "unit.source.json"
            bounded_receipt_path = gate_dir / "unit.bounded.json"
            write_bytes(stdout_path, b"cargo test pass\n")
            write_bytes(stderr_path, b"")
            write_json(
                source_receipt_path,
                {
                    "git_sha": candidate_sha,
                    "dirty_status": {"is_dirty": False, "status_short": []},
                },
            )
            write_json(
                bounded_receipt_path,
                {
                    "schema": "ferrum.bounded-command-receipt.v1",
                    "status": "pass",
                    "rc": 0,
                    "reason": "command_completed",
                    "cleanup": {"process_group_gone": True},
                    "violation": None,
                    "stdout": {
                        "sha256": file_sha256(stdout_path),
                        "size_bytes": stdout_path.stat().st_size,
                    },
                    "stderr": {
                        "sha256": file_sha256(stderr_path),
                        "size_bytes": stderr_path.stat().st_size,
                    },
                },
            )
            child = {
                "status": "pass",
                "lane": child_lane,
                "artifact_type": "g0_source_unit_bounded_gate",
                "receipt_schema": "ferrum.bounded-command-receipt.v1",
                "command": ["env", "PYTHONDONTWRITEBYTECODE=1", "CARGO_BUILD_JOBS=8", "RUST_TEST_THREADS=8", "cargo", "test", "--workspace", "--all-targets"],
                "bounded_receipt": make_ref(bounded_receipt_path, gate_dir),
                "source_receipt": make_ref(source_receipt_path, gate_dir),
                "stdout_log": make_ref(stdout_path, gate_dir),
                "stderr_log": make_ref(stderr_path, gate_dir),
                "source": {
                "git_sha": candidate_sha,
                "dirty_status": {"is_dirty": False, "status_short": []},
                },
            }
        write_json(child_path, child)
        write_outer_gate_fixture(
            outer_path,
            child_path,
            {
                "schema_version": 1,
                "status": "pass",
                "lane": lane,
                "child_returncode": 0,
                "child_pass_line": child_prefix + recorded_dir,
                "git_sha": candidate_sha,
                "dirty_status": {"is_dirty": False, "status_short": []},
                "artifact_dir": recorded_dir,
                "pass_line": f"FERRUM GATE {lane} PASS: {recorded_dir}",
            },
        )
        lane_pairs[key] = {"outer": outer_path, "child": child_path}

    prerelease["evidence"].update(
        {
            "rc_tag_ref_snapshot": make_ref(rc_tag_ref_path, root),
            "rc_tag_snapshot": make_ref(rc_tag_path, root),
            "source_gates": {
                key: {name: make_ref(gate_path, root) for name, gate_path in pair.items()}
                for key, pair in lane_pairs.items()
            },
            "workflow_policy": make_ref(workflow_native_paths["workflow_policy"], root),
            "native_operator_set": make_ref(workflow_native_paths["native_operator_set"], root),
        }
    )
    write_json(prerelease_path, prerelease)
    promotion["evidence"]["prerelease_manifest"] = make_ref(prerelease_path, root)
    write_json(promotion_path, promotion)
    g0_path = root / "g0-summary.json"
    fingerprint = asset_set_sha256(rows)
    g0_inputs = [
        *(pair["outer"] for pair in lane_pairs.values()),
        *(pair["outer"] for pair in gate_pairs.values()),
    ]
    write_json(
        g0_path,
        {
            "schema_version": 1,
            "status": "pass",
            "gates": [path.relative_to(root).as_posix() for path in g0_inputs],
            "artifact_dir": FINAL_ARTIFACT_DIR,
            "pass_line": f"G0 RELEASE PASS: {FINAL_ARTIFACT_DIR}",
            "release_candidate_sha": candidate_sha,
        },
    )
    final_evidence = {
        "prerelease_manifest": make_ref(prerelease_path, root),
        "promotion_manifest": make_ref(promotion_path, root),
        **{
            key: {name: make_ref(gate_path, root) for name, gate_path in pair.items()}
            for key, pair in gate_pairs.items()
        },
        "crates_io": make_ref(crates_path, root),
        **{
            key: make_ref(gate_path, root)
            for key, gate_path in workflow_native_paths.items()
        },
        "g0_summary": make_ref(g0_path, root),
        "g0_gates": {
            key: {name: make_ref(gate_path, root) for name, gate_path in pair.items()}
            for key, pair in lane_pairs.items()
        },
    }
    final = {"schema_version": 1, "artifact_type": "ferrum_v084_final_manifest", "status": "pass", "version": VERSION, "source": {"git_sha": candidate_sha, "dirty": False}, "release": {"id": release_id, "tag": TAG, "asset_set_sha256": fingerprint}, "evidence": final_evidence, "artifact_dir": FINAL_ARTIFACT_DIR, "pass_line": f"FERRUM 0.8.4 RELEASE PASS: {FINAL_ARTIFACT_DIR}"}
    final_path = root / "final.json"
    write_json(final_path, final)
    return prerelease_path, promotion_path, final_path


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-v084-release-goal-") as temporary:
        base = Path(temporary)
        fixture_index = 0

        def fresh_fixture() -> tuple[Path, Path, Path, Path]:
            nonlocal fixture_index
            fixture_index += 1
            fixture_root = base / f"case-{fixture_index}"
            prerelease, promotion, final = build_selftest_fixture(fixture_root)
            return fixture_root, prerelease, promotion, final

        root, prerelease_path, promotion_path, final_path = fresh_fixture()
        validated_prerelease = validate_prerelease_manifest(prerelease_path)
        validate_promotion_manifest(promotion_path)
        copied_final = read_json(final_path, "copied remote-path final fixture")
        copied_outer_path = root / copied_final["evidence"]["g0_gates"]["unit"]["outer"]["path"]
        copied_outer = read_json(copied_outer_path, "copied remote-path outer fixture")
        require(
            str(copied_outer.get("artifact_dir", "")).startswith("/remote/release-evidence/"),
            "copied remote-path fixture did not retain its remote artifact_dir",
        )
        validate_final_manifest(final_path)

        # Portable README E2E evidence must reject locally forged process,
        # binary, progress, and network-environment receipts independently of
        # the assembler that copied them.
        prerelease_fixture = read_json(prerelease_path, "selftest prerelease fixture")
        metal_summary_path = root / prerelease_fixture["evidence"]["readme_e2e"]["metal"]["path"]
        metal_summary = read_json(metal_summary_path, "selftest Metal README E2E summary")
        metal_evidence = require_object(metal_summary["evidence"], "selftest Metal README E2E evidence")
        metal_network = require_object(metal_summary["network_environment"], "selftest Metal network environment")
        metal_package = validated_prerelease["packages"]["metal"]
        metal_alias = "qwen3.5:4b-q4_k_m"

        run_wrapper_path = metal_summary_path.parent / metal_evidence["run"]["path"]
        run_wrapper_original = run_wrapper_path.read_bytes()
        run_wrapper = read_json(run_wrapper_path, "selftest run wrapper")
        wrong_binary_path = metal_summary_path.parent / "runtime/wrong-ferrum"
        write_bytes(wrong_binary_path, b"not-the-downloaded-ferrum")
        run_wrapper["extracted_binary"] = make_ref(wrong_binary_path, metal_summary_path.parent)
        write_json(run_wrapper_path, run_wrapper)
        expect_failure(
            lambda: validate_e2e_command(
                make_ref(run_wrapper_path, metal_summary_path.parent),
                root=metal_summary_path.parent,
                backend="metal",
                label="run",
                alias=metal_alias,
                package_binary_sha256=metal_package["binary_sha256"],
                recorded_artifact_dir=metal_summary["artifact_dir"],
                network_environment_ref=metal_network["child_processes"],
            ),
            "extracted binary differs",
        )
        run_wrapper_path.write_bytes(run_wrapper_original)

        run_wrapper = read_json(run_wrapper_path, "selftest run wrapper")
        empty_progress_path = metal_summary_path.parent / "commands/run/empty-progress.jsonl"
        write_bytes(empty_progress_path, b"")
        run_wrapper["progress"] = make_ref(empty_progress_path, metal_summary_path.parent)
        write_json(run_wrapper_path, run_wrapper)
        expect_failure(
            lambda: validate_e2e_command(
                make_ref(run_wrapper_path, metal_summary_path.parent),
                root=metal_summary_path.parent,
                backend="metal",
                label="run",
                alias=metal_alias,
                package_binary_sha256=metal_package["binary_sha256"],
                recorded_artifact_dir=metal_summary["artifact_dir"],
                network_environment_ref=metal_network["child_processes"],
            ),
            "is empty",
        )
        run_wrapper_path.write_bytes(run_wrapper_original)

        child_network_path = metal_summary_path.parent / metal_network["child_processes"]["path"]
        child_network = read_json(child_network_path, "selftest child network environment")
        child_network["variables"][0]["value"] = "plaintext-secret"
        bad_network_path = metal_summary_path.parent / "network-environment/plaintext.json"
        write_json(bad_network_path, child_network)
        expect_failure(
            lambda: validate_network_environment_ref(
                make_ref(bad_network_path, metal_summary_path.parent),
                root=metal_summary_path.parent,
                consumer="ferrum-child-processes",
                where="selftest plaintext network environment",
            ),
            "fields differ",
        )

        serve_wrapper_path = metal_summary_path.parent / metal_evidence["serve"]["path"]
        serve_wrapper_original = serve_wrapper_path.read_bytes()
        serve_wrapper = read_json(serve_wrapper_path, "selftest serve wrapper")
        serve_command_path = metal_summary_path.parent / serve_wrapper["command"]["path"]
        serve_command_original = serve_command_path.read_bytes()
        serve_command = read_json(serve_command_path, "selftest serve command")
        del serve_command["cleanup_precondition"]
        write_json(serve_command_path, serve_command)
        serve_wrapper["command"] = make_ref(serve_command_path, metal_summary_path.parent)
        write_json(serve_wrapper_path, serve_wrapper)
        expect_failure(
            lambda: validate_e2e_command(
                make_ref(serve_wrapper_path, metal_summary_path.parent),
                root=metal_summary_path.parent,
                backend="metal",
                label="serve",
                alias=metal_alias,
                package_binary_sha256=metal_package["binary_sha256"],
                recorded_artifact_dir=metal_summary["artifact_dir"],
                network_environment_ref=metal_network["child_processes"],
            ),
            "cleanup precondition",
        )
        serve_command_path.write_bytes(serve_command_original)
        serve_wrapper_path.write_bytes(serve_wrapper_original)

        # Dependency-audit negatives: the downloaded audit is authoritative,
        # both for its digest binding and for forbidden runtime linkage.
        cpu_asset = BACKENDS["cpu"]["asset"]
        audit_path = root / "files" / (cpu_asset.removesuffix(".tar.gz") + ".dependencies.txt")
        dependency_path = root / "files" / f"{cpu_asset}.dependency.json"
        abi_path = root / "files" / f"{cpu_asset}.abi.json"
        backend_paths = {name: root / "files" / name for name in EXPECTED_ASSETS}
        audit_original = audit_path.read_bytes()
        dependency_original = dependency_path.read_bytes()
        abi_original = abi_path.read_bytes()
        audit_path.write_bytes(b"fixture native dependencies changed\nlibc\n")
        expect_failure(
            lambda: validate_downloaded_backend(
                "cpu", paths=backend_paths, candidate_sha="a" * 40, rc_tag="v0.8.4-rc.1"
            ),
            "dependency audit SHA differs",
        )
        audit_path.write_bytes(b"libpython3.14.dylib\n")
        changed_sha = file_sha256(audit_path)
        dependency = read_json(dependency_path, "selftest dependency sidecar")
        dependency["audit_sha256"] = changed_sha
        write_json(dependency_path, dependency)
        abi = read_json(abi_path, "selftest ABI sidecar")
        abi["dependency_audit_sha256"] = changed_sha
        write_json(abi_path, abi)
        expect_failure(
            lambda: validate_downloaded_backend(
                "cpu", paths=backend_paths, candidate_sha="a" * 40, rc_tag="v0.8.4-rc.1"
            ),
            "forbidden runtime linkage",
        )
        audit_path.write_bytes(audit_original)
        dependency_path.write_bytes(dependency_original)
        abi_path.write_bytes(abi_original)

        # Reference integrity negative: a downloaded byte changes after capture.
        target = root / "files" / (BACKENDS["cpu"]["asset"] + ".sha256")
        original = target.read_bytes()
        target.write_bytes(original + b"tamper")
        expect_failure(lambda: validate_prerelease_manifest(prerelease_path), "size_bytes changed")
        target.write_bytes(original)

        # Prerelease semantic negative: streaming usage is absent, with a fresh ref.
        pre = read_json(prerelease_path, "selftest prerelease")
        e2e_path = root / pre["evidence"]["readme_e2e"]["metal"]["path"]
        e2e = read_json(e2e_path, "selftest E2E")
        e2e["checks"]["stream"]["output_tokens"] = 0
        write_json(e2e_path, e2e)
        pre["evidence"]["readme_e2e"]["metal"] = make_ref(e2e_path, root)
        negative_pre = root / "negative-prerelease.json"
        write_json(negative_pre, pre)
        expect_failure(lambda: validate_prerelease_manifest(negative_pre), "stream contract differs")

        # Restore the positive fixture before promotion/final recursion.
        root, prerelease_path, promotion_path, final_path = fresh_fixture()
        promotion = read_json(promotion_path, "selftest promotion")
        after_path = root / promotion["evidence"]["release_after"]["path"]
        latest_path = root / promotion["evidence"]["latest_release"]["path"]
        after = read_json(after_path, "selftest release after")
        after["name"] = "display name changed concurrently"
        after["assets"][0]["download_count"] = 99
        write_json(after_path, after)
        write_json(latest_path, after)
        promotion["evidence"]["release_after"] = make_ref(after_path, root)
        promotion["evidence"]["latest_release"] = make_ref(latest_path, root)
        mutable_promotion = root / "mutable-service-fields-promotion.json"
        write_json(mutable_promotion, promotion)
        validate_promotion_manifest(mutable_promotion)

        after["draft"] = True
        write_json(after_path, after)
        write_json(latest_path, after)
        promotion["evidence"]["release_after"] = make_ref(after_path, root)
        promotion["evidence"]["latest_release"] = make_ref(latest_path, root)
        negative_promotion = root / "negative-promotion.json"
        write_json(negative_promotion, promotion)
        expect_failure(lambda: validate_promotion_manifest(negative_promotion), "draft")

        # Restore again and make one final aggregate child fail closed.
        root, prerelease_path, promotion_path, final_path = fresh_fixture()
        final = read_json(final_path, "selftest final")
        crates_path = root / final["evidence"]["crates_io"]["path"]
        crates = read_json(crates_path, "selftest crates")
        crates["cargo_workspace_crates"][0]["crates_io_visible"] = False
        write_json(crates_path, crates)
        final["evidence"]["crates_io"] = make_ref(crates_path, root)
        negative_final = root / "negative-final.json"
        write_json(negative_final, final)
        expect_failure(lambda: validate_final_manifest(negative_final), "workspace version/visibility differs")

        # A child gate cannot be substituted independently of its canonical
        # run_gate outer directory, even when the substitute has valid bytes.
        root, prerelease_path, promotion_path, final_path = fresh_fixture()
        final = read_json(final_path, "selftest final outer/child")
        metal_pair = final["evidence"]["metal_tarball"]
        original_child = root / metal_pair["child"]["path"]
        substituted_child = root / "substituted-metal-gate.json"
        write_bytes(substituted_child, original_child.read_bytes())
        metal_pair["child"] = make_ref(substituted_child, root)
        negative_pair = root / "negative-final-child-substitution.json"
        write_json(negative_pair, final)
        expect_failure(lambda: validate_final_manifest(negative_pair), "outer/child layout differs")

        # Where run_gate records a child digest, changing the explicit child
        # ref cannot bypass that outer-manifest binding.
        root, prerelease_path, promotion_path, final_path = fresh_fixture()
        final = read_json(final_path, "selftest final bound child")
        unit_pair = final["evidence"]["g0_gates"]["unit"]
        unit_child_path = root / unit_pair["child"]["path"]
        unit_child = read_json(unit_child_path, "selftest unit child")
        unit_child["fixture_note"] = "tampered"
        write_json(unit_child_path, unit_child)
        unit_pair["child"] = make_ref(unit_child_path, root)
        negative_bound = root / "negative-final-bound-child.json"
        write_json(negative_bound, final)
        expect_failure(lambda: validate_final_manifest(negative_bound), "unit.child.size_bytes changed")

        # Workflow/native PASS artifacts must bind the same RC candidate as
        # the prerelease bytes and final source identity.
        root, prerelease_path, promotion_path, final_path = fresh_fixture()
        final = read_json(final_path, "selftest workflow candidate")
        workflow_path = root / final["evidence"]["workflow_policy"]["path"]
        workflow = read_json(workflow_path, "selftest workflow policy")
        workflow["evidence"]["candidate"]["git_sha"] = "6" * 40
        write_json(workflow_path, workflow)
        final["evidence"]["workflow_policy"] = make_ref(workflow_path, root)
        negative_candidate = root / "negative-final-workflow-candidate.json"
        write_json(negative_candidate, final)
        expect_failure(
            lambda: validate_final_manifest(negative_candidate),
            "prerelease workflow_policy.sha256 changed",
        )

        root, prerelease_path, promotion_path, final_path = fresh_fixture()
        final = read_json(final_path, "selftest native PASS binding")
        native_path = root / final["evidence"]["native_operator_set"]["path"]
        native = read_json(native_path, "selftest native operator set")
        native["pass_line"] = "FERRUM 0.8.4 NATIVE OPERATOR SET PASS: /wrong/artifact"
        write_json(native_path, native)
        final["evidence"]["native_operator_set"] = make_ref(native_path, root)
        negative_pass = root / "negative-final-native-pass.json"
        write_json(negative_pass, final)
        expect_failure(
            lambda: validate_final_manifest(negative_pass),
            "prerelease native_operator_set.size_bytes changed",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run hermetic positive and negative tests")
    subparsers = parser.add_subparsers(dest="mode")
    for mode in ("prerelease", "promotion", "final"):
        child = subparsers.add_parser(mode)
        child.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.self_test and args.mode is None:
        parser.error("choose --self-test or a validation mode")
    if args.self_test and args.mode is not None:
        parser.error("--self-test cannot be combined with a validation mode")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            self_test()
            print("FERRUM 0.8.4 RELEASE GOAL GATE SELFTEST PASS")
            return 0
        path = args.manifest.resolve()
        require(path.is_file() and not path.is_symlink(), f"manifest is not a regular non-symlink file: {path}")
        if args.mode == "prerelease":
            result = validate_prerelease_manifest(path)
        elif args.mode == "promotion":
            result = validate_promotion_manifest(path)
        else:
            result = validate_final_manifest(path)
        print(result["pass_line"])
        return 0
    except (ValidationError, OSError) as exc:
        print(f"FERRUM 0.8.4 RELEASE GOAL GATE FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
