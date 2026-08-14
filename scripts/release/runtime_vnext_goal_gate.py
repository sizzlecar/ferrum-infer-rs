#!/usr/bin/env python3
"""Fail-closed Runtime vNext v0.8.0 freeze, release, and goal gates.

This module intentionally stays a thin release DAG.  It consumes the existing
R0/R1/R2, model-matrix, performance, binary, Homebrew, crates.io, and release
completion artifacts.  It does not create a second correctness or performance
runner.

The one producer implemented here is ``staged-assets``.  It converts the three
downloaded GitHub Actions artifacts into one immutable byte-identity manifest
used by every R3 correctness/performance lane.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.8.0"
TAG = "v0.8.0"
SCHEMA_VERSION = 1
GITHUB_REPOSITORY = "sizzlecar/ferrum-infer-rs"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RC_TAG_RE = re.compile(r"^v0\.8\.0-rc\.[1-9][0-9]*$")

MODELS = (
    "m1-qwen35-4b",
    "m2-qwen35-35b-a3b",
    "m3-qwen3-30b-a3b",
)
BACKENDS = ("cuda", "metal")
LANE_KEYS = {
    f"{model.split('-', 1)[0]}_{backend}": (model, backend)
    for model in MODELS
    for backend in BACKENDS
}
ASSET_NAMES = {
    "cpu": "ferrum-linux-x86_64.tar.gz",
    "metal": "ferrum-macos-aarch64.tar.gz",
    "cuda": "ferrum-linux-x86_64-cuda-sm89.tar.gz",
}
RELEASE_SIDECAR_SUFFIXES = (
    ".sha256",
    ".binary.sha256",
    ".version.json",
    ".dependency.json",
    ".abi.json",
)
TARGET_TRIPLES = {
    "cpu": "x86_64-unknown-linux-gnu",
    "metal": "aarch64-apple-darwin",
    "cuda": "x86_64-unknown-linux-gnu",
}

CANONICAL_LANES = (
    "vnext-g10a",
    "vnext-g08-rc",
    "vnext-g09-rc",
    "runtime-vnext-metal-three-model",
    "runtime-vnext-cuda-three-model",
    "runtime-vnext-published-assets",
    "runtime-vnext-prepromotion",
    "vnext-g10b",
    "vnext-g10",
    "vnext-r3",
)

PASS_PREFIXES = {
    "staged-assets": "FERRUM RUNTIME VNEXT STAGED ASSETS PASS",
    "workflow-policy": "FERRUM RELEASE WORKFLOW POLICY PASS",
    "vnext-g10a": "FERRUM RUNTIME VNEXT G10A RELEASE FREEZE PASS",
    "vnext-g08-rc": (
        "FERRUM RUNTIME VNEXT G08 RELEASE CANDIDATE CORRECTNESS PASS"
    ),
    "vnext-g09-rc": (
        "FERRUM RUNTIME VNEXT G09 RELEASE CANDIDATE PERFORMANCE PASS"
    ),
    "runtime-vnext-metal-three-model": (
        "FERRUM RUNTIME VNEXT THREE MODEL METAL SOURCE PASS"
    ),
    "runtime-vnext-cuda-three-model": (
        "FERRUM RUNTIME VNEXT THREE MODEL CUDA SOURCE PASS"
    ),
    "runtime-vnext-published-assets": (
        "FERRUM RUNTIME VNEXT PUBLISHED ASSETS PASS"
    ),
    "runtime-vnext-prepromotion": "FERRUM V0.8.0 PREPROMOTION PASS",
    "vnext-g10b": "FERRUM RUNTIME VNEXT G10B PUBLISHED RELEASE PASS",
    "vnext-g10": "FERRUM RUNTIME VNEXT G10 V0.8.0 RELEASE PASS",
    "vnext-r3": "FERRUM RUNTIME VNEXT R3 V0.8.0 PUBLISHED PASS",
}

ADDITIONAL_PASS_PREFIXES = {
    "runtime-vnext-published-assets": (
        "FERRUM V0.8.0 THREE MODEL METAL CUDA RELEASE PASS",
    ),
    "vnext-r3": ("FERRUM RUNTIME VNEXT V0.8.0 RELEASE GOAL PASS",),
}

ARTIFACT_TYPES = {
    "staged-assets": "runtime_vnext_staged_assets_manifest",
    "workflow-policy": "runtime_vnext_release_workflow_policy_manifest",
    "vnext-g10a": "runtime_vnext_g10a_release_freeze_manifest",
    "vnext-g08-rc": "runtime_vnext_g08_rc_manifest",
    "vnext-g09-rc": "runtime_vnext_g09_rc_manifest",
    "runtime-vnext-metal-three-model": (
        "runtime_vnext_three_model_metal_source_manifest"
    ),
    "runtime-vnext-cuda-three-model": (
        "runtime_vnext_three_model_cuda_source_manifest"
    ),
    "runtime-vnext-published-assets": (
        "runtime_vnext_published_assets_manifest"
    ),
    "runtime-vnext-prepromotion": "runtime_vnext_prepromotion_manifest",
    "vnext-g10b": "runtime_vnext_g10b_published_release_manifest",
    "vnext-g10": "runtime_vnext_g10_release_manifest",
    "vnext-r3": "runtime_vnext_r3_release_goal_manifest",
}

RELEASE_DOCS = {
    "migration": Path("docs/release/runtime-vnext/0.8.0/MIGRATION.md"),
    "release_notes": Path("docs/release/runtime-vnext/0.8.0/RELEASE_NOTES.md"),
    "support_matrix": Path("docs/release/runtime-vnext/0.8.0/SUPPORT_MATRIX.md"),
    "performance_report": Path(
        "docs/release/runtime-vnext/0.8.0/PERFORMANCE_REPORT.md"
    ),
}
R3_SAMPLE_PLAN = Path(
    "scripts/release/configs/runtime_vnext_r3_sample_plan.json"
)

# R2 -> release-candidate is allowed to contain release metadata/control-plane
# changes only.  Rust product source, model/kernel/runtime defaults, and tests
# are deliberately absent and therefore fail closed.
G10A_RELEASE_ONLY_PATTERNS = (
    re.compile(r"^Cargo\.toml$"),
    re.compile(r"^Cargo\.lock$"),
    re.compile(r"^crates/[^/]+/Cargo\.toml$"),
    re.compile(r"^CHANGELOG\.md$"),
    re.compile(r"^README(?:_zh)?\.md$"),
    re.compile(r"^docs/release/runtime-vnext/0\.8\.0/[^/]+$"),
    re.compile(r"^\.github/workflows/(?:release|release-cuda|release-promote|docker)\.ya?ml$"),
    re.compile(r"^scripts/release/runtime_vnext_goal_gate\.py$"),
    re.compile(r"^scripts/release/runtime_vnext_prepromotion_bundle\.py$"),
    re.compile(r"^scripts/release/runtime_vnext_release_workflow_policy\.py$"),
    re.compile(r"^scripts/release/g0_source_gate\.sh$"),
    re.compile(r"^scripts/release/selftest_g0_validators\.py$"),
    re.compile(r"^scripts/release/runtime_vnext_(?:crates_io|release_assets|r3)[A-Za-z0-9_]*\.py$"),
    re.compile(r"^scripts/release/run_gate\.py$"),
    re.compile(r"^scripts/release/g0_release_summary\.py$"),
    re.compile(r"^scripts/release/validate_release_completion_manifest\.py$"),
    re.compile(r"^scripts/release/runtime_vnext_g08[abc]_(?:cuda|metal)_matrix_prepare\.py$"),
    re.compile(r"^scripts/release/runtime_vnext_g08a_matrix_specs\.py$"),
    re.compile(r"^scripts/release/runtime_vnext_g0_llama_sampled_execution\.py$"),
    re.compile(r"^scripts/release/configs/runtime_vnext_r3_sample_plan\.json$"),
    re.compile(r"^scripts/release/runtime_vnext_baseline_scenarios\.py$"),
    re.compile(r"^scripts/release/runtime_vnext_homebrew_release\.py$"),
    re.compile(r"^scripts/release/runtime_vnext_r2_ferrum_collector\.py$"),
    re.compile(r"^scripts/release/runtime_vnext_sampled_final\.py$"),
)


class GoalGateError(RuntimeError):
    """A release contract was not satisfied."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise GoalGateError(message)


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.strip() == value and value, f"{label} must be a non-empty trimmed string")
    return value


def require_sha256(value: Any, label: str) -> str:
    text = require_string(value, label)
    require(SHA256_RE.fullmatch(text) is not None, f"{label} is not a lowercase SHA256")
    return text


def require_git_sha(value: Any, label: str) -> str:
    text = require_string(value, label)
    require(GIT_SHA_RE.fullmatch(text) is not None, f"{label} is not a full lowercase git SHA")
    return text


def read_json(path: Path, label: str = "JSON") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GoalGateError(f"cannot read {label} {path}: {error}") from error
    return require_object(value, label)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return bytes_sha256(payload)


def ensure_fresh_out(out: Path) -> Path:
    root = out.expanduser().resolve()
    require(not root.exists(), f"--out must be a fresh path: {root}")
    root.mkdir(parents=True)
    return root


def artifact_ref(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(resolved.is_file() and not resolved.is_symlink(), f"artifact is not a regular non-symlink file: {resolved}")
    rendered = str(resolved)
    if root is not None:
        rendered = resolved.relative_to(root.resolve()).as_posix()
    return {
        "path": rendered,
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def resolve_ref(
    raw: Any,
    label: str,
    *,
    root: Path,
    require_within_root: bool = False,
) -> tuple[dict[str, Any], Path]:
    ref = require_object(raw, label)
    require(set(ref) == {"path", "sha256", "size_bytes"}, f"{label} fields differ")
    raw_path = require_string(ref.get("path"), f"{label}.path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if require_within_root:
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise GoalGateError(f"{label}.path escapes its manifest root") from error
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular non-symlink file: {path}")
    digest = require_sha256(ref.get("sha256"), f"{label}.sha256")
    size = ref.get("size_bytes")
    require(type(size) is int and size >= 0, f"{label}.size_bytes must be a nonnegative integer")
    require(path.stat().st_size == size, f"{label} size mismatch")
    require(file_sha256(path) == digest, f"{label} SHA256 mismatch")
    return copy.deepcopy(ref), path


def input_manifest(path: Path, default: str = "manifest.json") -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / default
    require(candidate.is_file(), f"manifest does not exist: {candidate}")
    return candidate


def pass_line(mode: str, out: Path) -> str:
    return f"{PASS_PREFIXES[mode]}: {out}"


def additional_pass_lines(mode: str, out: Path) -> list[str]:
    return [f"{prefix}: {out}" for prefix in ADDITIONAL_PASS_PREFIXES.get(mode, ())]


def source_object(git_sha: str, git_tree_sha: str, dirty: bool = False) -> dict[str, Any]:
    return {
        "git_sha": require_git_sha(git_sha, "source git SHA"),
        "git_tree_sha": require_git_sha(git_tree_sha, "source git tree SHA"),
        "dirty": dirty,
    }


def normalize_source(raw: Any, label: str) -> dict[str, Any]:
    value = require_object(raw, label)
    require(set(value) == {"git_sha", "git_tree_sha", "dirty"}, f"{label} fields differ")
    require(value.get("dirty") is False, f"{label} is dirty")
    return source_object(value.get("git_sha"), value.get("git_tree_sha"), False)


def git_output(args: list[str], *, repo: Path = REPO_ROOT) -> str:
    process = subprocess.run(
        ["git", "-c", "core.preloadindex=false", "-c", "index.threads=1", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    require(process.returncode == 0, f"git {' '.join(args)} failed: {process.stderr.strip()}")
    return process.stdout.strip()


def current_source(repo: Path = REPO_ROOT) -> dict[str, Any]:
    status = git_output(["status", "--short"], repo=repo)
    return source_object(
        git_output(["rev-parse", "HEAD"], repo=repo),
        git_output(["rev-parse", "HEAD^{tree}"], repo=repo),
        bool(status),
    )


def expected_artifact_fields() -> set[str]:
    return {
        "schema_version",
        "artifact_type",
        "lane",
        "status",
        "canonical",
        "version",
        "release_candidate",
        "artifact_dir",
        "inputs",
        "acceptance",
        "created_at",
        "pass_line",
        "additional_pass_lines",
    }


def base_manifest(
    mode: str,
    out: Path,
    *,
    release_candidate: dict[str, Any],
    inputs: dict[str, Any],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPES[mode],
        "lane": mode,
        "status": "pass",
        "canonical": True,
        "version": VERSION,
        "release_candidate": copy.deepcopy(release_candidate),
        "artifact_dir": str(out),
        "inputs": copy.deepcopy(inputs),
        "acceptance": copy.deepcopy(acceptance),
        "created_at": iso_now(),
        "pass_line": pass_line(mode, out),
        "additional_pass_lines": additional_pass_lines(mode, out),
    }


def copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def safe_archive_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    require(
        value == path.as_posix()
        and not path.is_absolute()
        and value not in {"", "."}
        and ".." not in path.parts,
        f"{label} is not a safe relative archive path",
    )
    return path


def staged_assets_manifest_fields() -> set[str]:
    """Exported schema field set used by staged matrix prepare scripts."""

    return {
        "schema_version",
        "artifact_type",
        "status",
        "canonical",
        "version",
        "publish_release",
        "release_candidate",
        "release_candidate_tag",
        "artifact_dir",
        "assets",
        "created_at",
        "pass_line",
    }


def staged_asset_row_fields(backend: str) -> set[str]:
    fields = {
        "backend",
        "workflow_run_id",
        "artifact",
        "artifact_manifest",
        "tarball",
        "sha256_file",
        "version_manifest",
        "dependency_abi_manifest",
        "binary",
    }
    if backend == "cuda":
        fields.add("target_sm")
    return fields


def validate_github_artifact_receipt(
    directory: Path,
    *,
    backend: str,
    release_candidate: dict[str, Any],
    release_candidate_tag: str,
    github_fetch: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path]:
    receipt_path = directory / "github-artifact.json"
    receipt = read_json(receipt_path, f"{backend} GitHub artifact receipt")
    expected = {
        "schema_version",
        "repository",
        "workflow_run_id",
        "artifact_id",
        "artifact_name",
        "artifact_digest",
        "expired",
        "archive_path",
        "workflow_inputs",
    }
    require(set(receipt) == expected, f"{backend} GitHub artifact receipt fields differ")
    require(
        receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("repository") == GITHUB_REPOSITORY
        and receipt.get("expired") is False,
        f"{backend} GitHub artifact receipt identity/status differs",
    )
    for key in ("workflow_run_id", "artifact_id"):
        require(
            type(receipt.get(key)) is int and receipt[key] > 0,
            f"{backend} {key} must be a positive integer",
        )
    require_string(receipt.get("artifact_name"), f"{backend} artifact name")
    digest_text = require_string(receipt.get("artifact_digest"), f"{backend} artifact digest")
    require(digest_text.startswith("sha256:"), f"{backend} artifact digest lacks sha256 prefix")
    digest = require_sha256(digest_text.removeprefix("sha256:"), f"{backend} artifact digest")
    inputs = require_object(receipt.get("workflow_inputs"), f"{backend} workflow inputs")
    require(
        set(inputs)
        >= {"release_candidate_sha", "release_candidate_tag", "publish_release"}
        and inputs.get("release_candidate_sha") == release_candidate["git_sha"]
        and inputs.get("release_candidate_tag") == release_candidate_tag
        and inputs.get("publish_release") in {False, "false"},
        f"{backend} workflow was not an exact-RC publish_release=false run",
    )
    staging_label = require_string(
        inputs.get("staging_label"), f"{backend} workflow staging_label"
    )
    require(
        re.fullmatch(r"[A-Za-z0-9._-]+", staging_label) is not None,
        f"{backend} workflow staging_label is invalid",
    )
    fetch = github_fetch or github_api_json
    workflow_run = require_object(
        fetch(f"actions/runs/{receipt['workflow_run_id']}"),
        f"{backend} live GitHub workflow run",
    )
    expected_workflow = (
        ".github/workflows/release-cuda.yml"
        if backend == "cuda"
        else ".github/workflows/release.yml"
    )
    repository = require_object(
        workflow_run.get("repository"), f"{backend} live workflow repository"
    )
    run_attempt = workflow_run.get("run_attempt")
    require(
        workflow_run.get("id") == receipt["workflow_run_id"]
        and repository.get("full_name") == GITHUB_REPOSITORY
        and workflow_run.get("path") == expected_workflow
        and workflow_run.get("event") == "workflow_dispatch"
        and workflow_run.get("head_sha") == release_candidate["git_sha"]
        and workflow_run.get("status") == "completed"
        and workflow_run.get("conclusion") == "success"
        and type(run_attempt) is int
        and run_attempt > 0,
        f"{backend} live GitHub workflow run identity/status differs",
    )
    live_artifact = require_object(
        fetch(f"actions/artifacts/{receipt['artifact_id']}"),
        f"{backend} live GitHub artifact",
    )
    live_workflow_run = require_object(
        live_artifact.get("workflow_run"),
        f"{backend} live artifact workflow run",
    )
    expected_artifact_name = (
        f"{ASSET_NAMES[backend].removesuffix('.tar.gz')}-"
        f"{staging_label}-{release_candidate['git_sha']}"
    )
    require(
        live_artifact.get("id") == receipt["artifact_id"]
        and live_artifact.get("name") == expected_artifact_name
        and receipt.get("artifact_name") == expected_artifact_name
        and live_artifact.get("digest") == digest_text
        and live_artifact.get("expired") is False
        and live_workflow_run.get("id") == receipt["workflow_run_id"]
        and live_workflow_run.get("head_sha") == release_candidate["git_sha"],
        f"{backend} live GitHub artifact identity/status differs",
    )
    archive_relative = safe_archive_path(
        require_string(receipt.get("archive_path"), f"{backend} artifact archive path"),
        f"{backend} artifact archive path",
    )
    archive_path = (directory / archive_relative.as_posix()).resolve()
    require(
        archive_path.is_relative_to(directory.resolve())
        and archive_path.is_file()
        and not archive_path.is_symlink(),
        f"{backend} artifact archive is not a local regular file",
    )
    require(file_sha256(archive_path) == digest, f"{backend} artifact archive digest differs from GitHub")
    live_size = live_artifact.get("size_in_bytes")
    require(
        type(live_size) is int
        and live_size == archive_path.stat().st_size,
        f"{backend} live GitHub artifact size differs from the downloaded archive",
    )
    receipt = copy.deepcopy(receipt)
    receipt["_live_workflow_run"] = {
        "id": receipt["workflow_run_id"],
        "attempt": run_attempt,
        "path": expected_workflow,
        "event": "workflow_dispatch",
        "head_sha": release_candidate["git_sha"],
        "status": "completed",
        "conclusion": "success",
    }
    return receipt, archive_path


def validate_adjacent_manifest_common(
    value: dict[str, Any],
    *,
    label: str,
    asset_name: str,
    tarball_sha256: str,
    binary_sha256: str,
    release_candidate_sha: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
) -> None:
    common = {
        "schema_version",
        "asset_name",
        "asset_sha256",
        "binary_name",
        "binary_sha256",
        "release_candidate_sha",
        "staging_label",
        "workflow_run_id",
        "workflow_run_attempt",
    }
    require(common <= set(value), f"{label} lacks common staged identity fields")
    recorded_run_id = value.get("workflow_run_id")
    if isinstance(recorded_run_id, str) and recorded_run_id.isdigit():
        recorded_run_id = int(recorded_run_id)
    recorded_run_attempt = value.get("workflow_run_attempt")
    if isinstance(recorded_run_attempt, str) and recorded_run_attempt.isdigit():
        recorded_run_attempt = int(recorded_run_attempt)
    require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("asset_name") == asset_name
        and value.get("asset_sha256") == tarball_sha256
        and value.get("binary_name") == "ferrum"
        and value.get("binary_sha256") == binary_sha256
        and value.get("release_candidate_sha") == release_candidate_sha
        and recorded_run_id == workflow_run_id
        and recorded_run_attempt == workflow_run_attempt,
        f"{label} staged byte/source identity differs",
    )


def tarball_binary_identity(tarball: Path) -> dict[str, Any]:
    try:
        with tarfile.open(tarball, mode="r:*") as archive:
            regular_binary = []
            for member in archive.getmembers():
                safe_archive_path(member.name.rstrip("/") or ".", "tar member")
                require(not member.issym() and not member.islnk(), "staged tarball contains a link")
                if member.isfile() and PurePosixPath(member.name).name == "ferrum":
                    regular_binary.append(member)
            require(len(regular_binary) == 1, "staged tarball must contain exactly one regular ferrum binary")
            member = regular_binary[0]
            stream = archive.extractfile(member)
            require(stream is not None, "staged ferrum binary cannot be read")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except (OSError, EOFError, tarfile.TarError) as error:
        raise GoalGateError(f"cannot inspect staged tarball {tarball}: {error}") from error
    require(size == member.size and size > 0, "staged ferrum binary size differs")
    return {
        "archive_path": member.name,
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def zip_member_payloads(archive_path: Path) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            payloads: dict[str, bytes] = {}
            for info in archive.infolist():
                if info.is_dir():
                    continue
                safe_archive_path(info.filename, "GitHub artifact member")
                name = PurePosixPath(info.filename).name
                require(name not in payloads, f"duplicate GitHub artifact basename: {name}")
                payloads[name] = archive.read(info)
            return payloads
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        raise GoalGateError(f"cannot inspect GitHub artifact archive {archive_path}: {error}") from error


def collect_staged_asset(
    source_dir: Path,
    destination_root: Path,
    *,
    backend: str,
    release_candidate: dict[str, Any],
    release_candidate_tag: str,
    github_fetch: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source = source_dir.expanduser().resolve()
    require(source.is_dir(), f"{backend} staged download directory does not exist: {source}")
    asset_name = ASSET_NAMES[backend]
    required_paths = {
        "tarball": source / asset_name,
        "sha256_file": source / f"{asset_name}.sha256",
        "binary_sha256_file": source / f"{asset_name}.binary.sha256",
        "version_manifest": source / f"{asset_name}.version.json",
        "dependency_manifest": source / f"{asset_name}.dependency.json",
        "abi_manifest": source / f"{asset_name}.abi.json",
    }
    for label, path in required_paths.items():
        require(path.is_file() and not path.is_symlink(), f"{backend} {label} is missing: {path}")

    receipt, archive_path = validate_github_artifact_receipt(
        source,
        backend=backend,
        release_candidate=release_candidate,
        release_candidate_tag=release_candidate_tag,
        github_fetch=github_fetch,
    )
    archive_payloads = zip_member_payloads(archive_path)
    for label, path in required_paths.items():
        require(
            path.name in archive_payloads and bytes_sha256(archive_payloads[path.name]) == file_sha256(path),
            f"{backend} {label} is not byte-identical to its GitHub artifact archive",
        )

    tarball = required_paths["tarball"]
    tarball_sha = file_sha256(tarball)
    checksum_parts = required_paths["sha256_file"].read_text(encoding="utf-8").split()
    require(
        len(checksum_parts) >= 2
        and checksum_parts[0] == tarball_sha
        and Path(checksum_parts[-1]).name == asset_name,
        f"{backend} adjacent tarball SHA256 differs",
    )
    binary = tarball_binary_identity(tarball)
    binary_parts = required_paths["binary_sha256_file"].read_text(encoding="utf-8").split()
    require(
        len(binary_parts) >= 2
        and binary_parts[0] == binary["sha256"]
        and Path(binary_parts[-1]).name == "ferrum",
        f"{backend} adjacent binary SHA256 differs",
    )

    version_manifest = read_json(required_paths["version_manifest"], f"{backend} version manifest")
    dependency_manifest = read_json(required_paths["dependency_manifest"], f"{backend} dependency manifest")
    abi_manifest = read_json(required_paths["abi_manifest"], f"{backend} ABI manifest")
    for label, value in (
        ("version manifest", version_manifest),
        ("dependency manifest", dependency_manifest),
        ("ABI manifest", abi_manifest),
    ):
        validate_adjacent_manifest_common(
            value,
            label=f"{backend} {label}",
            asset_name=asset_name,
            tarball_sha256=tarball_sha,
            binary_sha256=binary["sha256"],
            release_candidate_sha=release_candidate["git_sha"],
            workflow_run_id=receipt["workflow_run_id"],
            workflow_run_attempt=receipt["_live_workflow_run"]["attempt"],
        )
    require(version_manifest.get("version") == VERSION, f"{backend} staged version is not {VERSION}")
    require(
        dependency_manifest.get("forbidden_runtime_linkage_found") is False,
        f"{backend} staged binary has forbidden runtime linkage",
    )
    require(
        abi_manifest.get("backend") == backend
        and abi_manifest.get("target_triple") == TARGET_TRIPLES[backend],
        f"{backend} staged ABI target differs",
    )
    if backend == "cuda":
        require(abi_manifest.get("cuda_compute_capability") == "89", "CUDA staged ABI is not sm89")

    dest = destination_root / "assets" / backend
    copies = {
        "artifact_archive": (archive_path, dest / "github-artifact.zip"),
        "tarball": (tarball, dest / asset_name),
        "sha256_file": (required_paths["sha256_file"], dest / f"{asset_name}.sha256"),
        "binary_sha256_file": (
            required_paths["binary_sha256_file"],
            dest / f"{asset_name}.binary.sha256",
        ),
        "version_manifest": (
            required_paths["version_manifest"],
            dest / f"{asset_name}.version.json",
        ),
        "dependency_manifest": (
            required_paths["dependency_manifest"],
            dest / f"{asset_name}.dependency.json",
        ),
        "abi_manifest": (
            required_paths["abi_manifest"],
            dest / f"{asset_name}.abi.json",
        ),
    }
    for source_path, destination in copies.values():
        copy_or_link(source_path, destination)

    artifact_manifest_path = dest / "artifact.manifest.json"
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_github_workflow_artifact_manifest",
        "status": "pass",
        "repository": GITHUB_REPOSITORY,
        "workflow_run_id": receipt["workflow_run_id"],
        "workflow_run": copy.deepcopy(receipt["_live_workflow_run"]),
        "artifact": {
            "id": receipt["artifact_id"],
            "name": receipt["artifact_name"],
            "digest": receipt["artifact_digest"],
        },
        "archive": artifact_ref(copies["artifact_archive"][1], root=destination_root),
        "workflow_inputs": copy.deepcopy(receipt["workflow_inputs"]),
        "release_candidate": copy.deepcopy(release_candidate),
        "release_candidate_tag": release_candidate_tag,
        "publish_release": False,
    }
    write_json(artifact_manifest_path, artifact_manifest)
    dependency_abi_path = dest / "dependency-abi.manifest.json"
    write_json(
        dependency_abi_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_dependency_abi_manifest",
            "status": "pass",
            "backend": backend,
            "target_triple": TARGET_TRIPLES[backend],
            "tarball_sha256": tarball_sha,
            "binary_sha256": binary["sha256"],
            "release_candidate": copy.deepcopy(release_candidate),
            "release_candidate_tag": release_candidate_tag,
            "dependency": artifact_ref(copies["dependency_manifest"][1], root=destination_root),
            "abi": artifact_ref(copies["abi_manifest"][1], root=destination_root),
        },
    )
    row: dict[str, Any] = {
        "backend": backend,
        "workflow_run_id": receipt["workflow_run_id"],
        "artifact": {
            "id": receipt["artifact_id"],
            "name": receipt["artifact_name"],
            "digest": receipt["artifact_digest"],
        },
        "artifact_manifest": artifact_ref(artifact_manifest_path, root=destination_root),
        "tarball": artifact_ref(copies["tarball"][1], root=destination_root),
        "sha256_file": artifact_ref(copies["sha256_file"][1], root=destination_root),
        "version_manifest": artifact_ref(copies["version_manifest"][1], root=destination_root),
        "dependency_abi_manifest": artifact_ref(dependency_abi_path, root=destination_root),
        "binary": binary,
    }
    if backend == "cuda":
        row["target_sm"] = "89"
    require(set(row) == staged_asset_row_fields(backend), f"{backend} normalized staged row fields differ")
    return row


def validate_staged_workflow_run_record(
    value: Any,
    *,
    backend: str,
    workflow_run_id: int,
    release_candidate_sha: str,
) -> dict[str, Any]:
    workflow_run = require_object(value, f"staged {backend} workflow run")
    expected_workflow_path = (
        ".github/workflows/release-cuda.yml"
        if backend == "cuda"
        else ".github/workflows/release.yml"
    )
    require(
        set(workflow_run)
        == {
            "id",
            "attempt",
            "path",
            "event",
            "head_sha",
            "status",
            "conclusion",
        }
        and workflow_run.get("id") == workflow_run_id
        and type(workflow_run.get("attempt")) is int
        and workflow_run["attempt"] > 0
        and workflow_run.get("path") == expected_workflow_path
        and workflow_run.get("event") == "workflow_dispatch"
        and workflow_run.get("head_sha") == release_candidate_sha
        and workflow_run.get("status") == "completed"
        and workflow_run.get("conclusion") == "success",
        f"staged {backend} workflow run identity/status differs",
    )
    return workflow_run


def validate_staged_assets_manifest(path: Path) -> dict[str, Any]:
    manifest_path = input_manifest(path)
    root = manifest_path.parent
    value = read_json(manifest_path, "staged assets manifest")
    require(set(value) == staged_assets_manifest_fields(), "staged assets manifest fields differ")
    require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("artifact_type") == ARTIFACT_TYPES["staged-assets"]
        and value.get("status") == "pass"
        and value.get("canonical") is True
        and value.get("version") == VERSION
        and value.get("publish_release") is False
        and Path(str(value.get("artifact_dir", ""))).resolve() == root.resolve()
        and value.get("pass_line") == f"{PASS_PREFIXES['staged-assets']}: {root}",
        "staged assets manifest identity/status differs",
    )
    release_candidate = normalize_source(value.get("release_candidate"), "staged release candidate")
    release_candidate_tag = require_string(
        value.get("release_candidate_tag"), "staged release candidate tag"
    )
    require(
        RC_TAG_RE.fullmatch(release_candidate_tag) is not None,
        "staged release candidate tag is not v0.8.0-rc.N",
    )
    assets = require_object(value.get("assets"), "staged assets")
    require(set(assets) == {"cpu", "metal", "cuda"}, "staged asset backend set differs")
    artifact_ids: set[int] = set()
    artifact_names: set[str] = set()
    workflow_run_attempts: dict[str, int] = {}
    for backend in ("cpu", "metal", "cuda"):
        row = require_object(assets.get(backend), f"staged {backend}")
        require(set(row) == staged_asset_row_fields(backend), f"staged {backend} row fields differ")
        require(row.get("backend") == backend, f"staged {backend} backend differs")
        require(type(row.get("workflow_run_id")) is int and row["workflow_run_id"] > 0, f"staged {backend} workflow run id differs")
        artifact = require_object(row.get("artifact"), f"staged {backend} artifact")
        require(set(artifact) == {"id", "name", "digest"}, f"staged {backend} artifact fields differ")
        artifact_id = artifact.get("id")
        name = require_string(artifact.get("name"), f"staged {backend} artifact name")
        digest = require_string(artifact.get("digest"), f"staged {backend} artifact digest")
        require(type(artifact_id) is int and artifact_id > 0 and artifact_id not in artifact_ids, f"staged {backend} artifact id differs/duplicates")
        require(name not in artifact_names, f"staged {backend} artifact name duplicates")
        require(digest.startswith("sha256:") and SHA256_RE.fullmatch(digest[7:]), f"staged {backend} artifact digest differs")
        artifact_ids.add(artifact_id)
        artifact_names.add(name)
        resolved: dict[str, Path] = {}
        for field in (
            "artifact_manifest",
            "tarball",
            "sha256_file",
            "version_manifest",
            "dependency_abi_manifest",
        ):
            _, resolved[field] = resolve_ref(
                row.get(field),
                f"staged {backend} {field}",
                root=root,
                require_within_root=True,
            )
        binary = require_object(row.get("binary"), f"staged {backend} binary")
        require(set(binary) == {"archive_path", "sha256", "size_bytes"}, f"staged {backend} binary fields differ")
        require_sha256(binary.get("sha256"), f"staged {backend} binary SHA256")
        require(type(binary.get("size_bytes")) is int and binary["size_bytes"] > 0, f"staged {backend} binary size differs")
        observed = tarball_binary_identity(resolved["tarball"])
        require(observed == binary, f"staged {backend} tarball inner binary identity differs")
        checksum = resolved["sha256_file"].read_text(encoding="utf-8").split()
        require(checksum and checksum[0] == row["tarball"]["sha256"], f"staged {backend} adjacent checksum differs")
        artifact_manifest = read_json(resolved["artifact_manifest"], f"staged {backend} artifact manifest")
        require(
            set(artifact_manifest)
            == {
                "schema_version",
                "artifact_type",
                "status",
                "repository",
                "workflow_run_id",
                "workflow_run",
                "artifact",
                "archive",
                "workflow_inputs",
                "release_candidate",
                "release_candidate_tag",
                "publish_release",
            },
            f"staged {backend} artifact manifest fields differ",
        )
        workflow_run = validate_staged_workflow_run_record(
            artifact_manifest.get("workflow_run"),
            backend=backend,
            workflow_run_id=row["workflow_run_id"],
            release_candidate_sha=release_candidate["git_sha"],
        )
        workflow_run_attempts[backend] = workflow_run["attempt"]
        workflow_inputs = require_object(
            artifact_manifest.get("workflow_inputs"),
            f"staged {backend} workflow inputs",
        )
        staging_label = require_string(
            workflow_inputs.get("staging_label"),
            f"staged {backend} staging label",
        )
        expected_artifact_name = (
            f"{ASSET_NAMES[backend].removesuffix('.tar.gz')}-"
            f"{staging_label}-{release_candidate['git_sha']}"
        )
        require(
            set(workflow_inputs)
            == {
                "release_candidate_sha",
                "release_candidate_tag",
                "staging_label",
                "publish_release",
            }
            and re.fullmatch(r"[A-Za-z0-9._-]+", staging_label) is not None
            and workflow_inputs.get("release_candidate_sha")
            == release_candidate["git_sha"]
            and workflow_inputs.get("release_candidate_tag")
            == release_candidate_tag
            and workflow_inputs.get("publish_release") in {False, "false"}
            and artifact.get("name") == expected_artifact_name,
            f"staged {backend} workflow input/artifact-name binding differs",
        )
        archive_ref, _ = resolve_ref(
            artifact_manifest.get("archive"),
            f"staged {backend} GitHub artifact archive",
            root=root,
            require_within_root=True,
        )
        require(
            artifact_manifest.get("schema_version") == SCHEMA_VERSION
            and artifact_manifest.get("artifact_type")
            == "runtime_vnext_github_workflow_artifact_manifest"
            and artifact_manifest.get("status") == "pass"
            and artifact_manifest.get("repository") == GITHUB_REPOSITORY
            and artifact_manifest.get("release_candidate") == release_candidate
            and artifact_manifest.get("release_candidate_tag")
            == release_candidate_tag
            and artifact_manifest.get("publish_release") is False
            and artifact_manifest.get("artifact") == artifact
            and artifact_manifest.get("workflow_run_id") == row["workflow_run_id"]
            and artifact.get("digest") == f"sha256:{archive_ref['sha256']}",
            f"staged {backend} artifact manifest binding differs",
        )
        dependency_abi = read_json(resolved["dependency_abi_manifest"], f"staged {backend} dependency/ABI manifest")
        require(
            dependency_abi.get("release_candidate") == release_candidate
            and dependency_abi.get("release_candidate_tag")
            == release_candidate_tag
            and dependency_abi.get("binary_sha256") == binary["sha256"]
            and dependency_abi.get("tarball_sha256") == row["tarball"]["sha256"],
            f"staged {backend} dependency/ABI binding differs",
        )
        if backend == "cuda":
            require(row.get("target_sm") == "89", "staged CUDA target is not sm89")
    validate_staged_workflow_run_topology(assets)
    require(
        workflow_run_attempts["cpu"] == workflow_run_attempts["metal"],
        "CPU and Metal staged artifacts must come from the same release.yml run attempt",
    )
    return {
        "path": manifest_path,
        "manifest": value,
        "ref": artifact_ref(manifest_path),
        "release_candidate": release_candidate,
        "release_candidate_tag": release_candidate_tag,
        "assets": copy.deepcopy(assets),
    }


def validate_staged_workflow_run_topology(assets: dict[str, Any]) -> None:
    cpu_run = assets["cpu"].get("workflow_run_id")
    metal_run = assets["metal"].get("workflow_run_id")
    cuda_run = assets["cuda"].get("workflow_run_id")
    require(
        type(cpu_run) is int
        and cpu_run > 0
        and metal_run == cpu_run,
        "CPU and Metal staged artifacts must come from the same release.yml run",
    )
    require(
        type(cuda_run) is int and cuda_run > 0 and cuda_run != cpu_run,
        "CUDA staged artifact must come from a distinct release-cuda.yml run",
    )


def build_staged_assets(
    args: argparse.Namespace,
    *,
    github_fetch: Callable[[str], dict[str, Any]] | None = None,
) -> Path:
    out = ensure_fresh_out(args.out)
    release_candidate = source_object(
        args.release_candidate_sha,
        args.release_candidate_tree_sha,
        False,
    )
    if not args.skip_checkout_binding:
        observed_tree = git_output(["rev-parse", f"{release_candidate['git_sha']}^{{tree}}"])
        require(observed_tree == release_candidate["git_tree_sha"], "release candidate tree SHA differs from git")
        require(
            git_output(["cat-file", "-t", args.release_candidate_tag]) == "tag",
            "release candidate tag must be annotated",
        )
        require(
            git_output(["rev-parse", f"{args.release_candidate_tag}^{{commit}}"])
            == release_candidate["git_sha"],
            "release candidate tag does not peel to the staged commit",
        )
    assets = {
        backend: collect_staged_asset(
            getattr(args, f"{backend}_dir"),
            out,
            backend=backend,
            release_candidate=release_candidate,
            release_candidate_tag=args.release_candidate_tag,
            github_fetch=github_fetch,
        )
        for backend in ("cpu", "metal", "cuda")
    }
    validate_staged_workflow_run_topology(assets)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPES["staged-assets"],
        "status": "pass",
        "canonical": True,
        "version": VERSION,
        "publish_release": False,
        "release_candidate": release_candidate,
        "release_candidate_tag": args.release_candidate_tag,
        "artifact_dir": str(out),
        "assets": assets,
        "created_at": iso_now(),
        "pass_line": f"{PASS_PREFIXES['staged-assets']}: {out}",
    }
    write_json(out / "manifest.json", manifest)
    validate_staged_assets_manifest(out / "manifest.json")
    return out


def workflow_sections(text: str, label: str) -> dict[str, Any]:
    """Parse the small YAML structure needed by the release policy gate.

    This deliberately does not attempt to implement general YAML.  It parses
    top-level mappings plus the child keys under ``on`` and ``jobs`` and then
    applies conservative source checks to step bodies.  Ambiguous flow-style
    or duplicate top-level structures fail closed.
    """

    top: dict[str, int] = {}
    lines = text.splitlines()
    for index, raw in enumerate(lines):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent == 0:
            match = re.match(r"^([A-Za-z0-9_.-]+):(?:\s*(.*))?$", raw)
            require(match is not None, f"{label} contains an unsupported top-level YAML form at line {index + 1}")
            key = match.group(1)
            require(key not in top, f"{label} duplicates top-level key {key}")
            top[key] = index
    require({"name", "on", "jobs"} <= set(top), f"{label} lacks name/on/jobs YAML mappings")

    def child_keys(parent: str) -> set[str]:
        start = top[parent] + 1
        end = min((line for line in top.values() if line > top[parent]), default=len(lines))
        keys: set[str] = set()
        for raw in lines[start:end]:
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            if indent != 2:
                continue
            match = re.match(r"^\s{2}([A-Za-z0-9_.-]+):", raw)
            if match:
                require(match.group(1) not in keys, f"{label} duplicates {parent}.{match.group(1)}")
                keys.add(match.group(1))
        return keys

    return {
        "top_level_keys": sorted(top),
        "events": child_keys("on"),
        "jobs": child_keys("jobs"),
        "normalized_text": "\n".join(
            raw.split("#", 1)[0].rstrip() for raw in lines if raw.split("#", 1)[0].strip()
        ).lower(),
    }


def validate_staging_workflow(text: str, *, cuda: bool) -> dict[str, Any]:
    label = "release-cuda.yml" if cuda else "release.yml"
    parsed = workflow_sections(text, label)
    require(parsed["events"] == {"workflow_dispatch"}, f"{label} must be workflow_dispatch-only")
    normalized = parsed["normalized_text"]
    forbidden = (
        "softprops/action-gh-release",
        "action-gh-release",
        "gh release create",
        "gh release upload",
        "docker/build-push-action",
        "docker push",
        "prerelease: false",
        "draft: false",
    )
    require(not any(item in normalized for item in forbidden), f"{label} contains a publication path")
    require(
        "release_candidate_sha:" in normalized
        and "publish_release:" in normalized
        and "default: false" in normalized
        and "publish_release must remain false" in normalized
        and "ref: ${{ inputs.release_candidate_sha }}" in normalized,
        f"{label} lacks exact-RC publish_release=false input/binding",
    )
    required_fragments = (
        "actions/upload-artifact@",
        ".tar.gz.sha256",
        ".tar.gz.binary.sha256",
        ".tar.gz.version.json",
        ".tar.gz.dependency.json",
        ".tar.gz.abi.json",
        "release_candidate_sha",
        "workflow_run_id",
        "binary_sha256",
    )
    require(all(item in normalized for item in required_fragments), f"{label} lacks staged asset manifests")
    expected_builds = 1 if cuda else 2
    require(
        normalized.count("cargo build --release -p ferrum-cli --bin ferrum") == expected_builds,
        f"{label} must build each product asset exactly once",
    )
    if cuda:
        require(
            "cuda_compute_cap: '89'" in normalized
            and "cuda_compute_capability=\"89\"" in normalized
            and "ferrum-linux-x86_64-cuda-sm89.tar.gz" in normalized,
            "release-cuda.yml does not freeze CUDA sm89",
        )
    else:
        require(
            "ferrum-linux-x86_64.tar.gz" in normalized
            and "ferrum-macos-aarch64.tar.gz" in normalized,
            "release.yml does not stage both CPU and Metal assets",
        )
    return {
        "events": sorted(parsed["events"]),
        "jobs": sorted(parsed["jobs"]),
        "build_count": expected_builds,
        "publish_release": False,
    }


def validate_docker_workflow(text: str) -> dict[str, Any]:
    parsed = workflow_sections(text, "docker.yml")
    require(parsed["events"] == {"workflow_dispatch"}, "docker.yml must be workflow_dispatch-only")
    normalized = parsed["normalized_text"]
    forbidden = (
        "docker/login-action",
        "docker/build-push-action",
        "docker push",
        "push: true",
        "tags:",
        "ghcr.io",
        ":latest",
        ":stable",
        ":candidate",
    )
    require(not any(item in normalized for item in forbidden), "docker.yml contains a Docker publish/tag path")
    require(
        "docker distribution is disabled" in normalized
        and "does not publish" in normalized,
        "docker.yml does not explicitly record the disabled distribution",
    )
    return {"events": sorted(parsed["events"]), "jobs": sorted(parsed["jobs"]), "publish_job_count": 0}


def validate_promotion_workflow(text: str) -> dict[str, Any]:
    parsed = workflow_sections(text, "release-promote.yml")
    require(parsed["events"] == {"workflow_dispatch"}, "release-promote.yml must be workflow_dispatch-only")
    normalized = parsed["normalized_text"]
    required = (
        "runtime-vnext-prepromotion",
        "prepromotion",
        "consum",
        "release_candidate_sha",
        "prerelease",
        "false",
        "promotion-consumption.json",
        "consumption_complete_name",
        "runtime-vnext-diagnostics-v1",
        "gh api",
    )
    require(all(item in normalized for item in required), "promotion workflow lacks prepromotion/single-use/final-release contract")
    forbidden = (
        "cargo build",
        "cargo package",
        "softprops/action-gh-release",
        "gh release upload",
        "docker build",
    )
    require(not any(item in normalized for item in forbidden), "promotion workflow rebuilds or replaces release assets")
    require(
        normalized.count("uses: actions/upload-artifact@v4") == 2
        and "name: ${{ env.consumption_claim_name }}" in normalized
        and "name: ${{ env.consumption_complete_name }}" in normalized
        and "path: promotion-consumption.json" in normalized,
        "promotion workflow must upload exactly the pending claim and durable completion receipt",
    )
    require(
        not any(
            asset in normalized
            for asset in (
                "ferrum-linux-x86_64.tar.gz",
                "ferrum-linux-x86_64-cuda-sm89.tar.gz",
                "ferrum-macos-aarch64.tar.gz",
            )
        ),
        "promotion workflow uploads or replaces a product release asset",
    )
    require(
        normalized.count("runtime-vnext-prepromotion") >= 1
        and "durable complete marker" in normalized
        and "asset" in normalized
        and "sha256" in normalized,
        "promotion workflow lacks missing/reused manifest or asset identity rejection",
    )
    return {
        "events": sorted(parsed["events"]),
        "jobs": sorted(parsed["jobs"]),
        "requires_prepromotion": True,
        "single_consumption": True,
        "rebuild_count": 0,
    }


def validate_workflow_sources(repo_root: Path) -> dict[str, Any]:
    paths = {
        "cpu_metal_staging": repo_root / ".github/workflows/release.yml",
        "cuda_staging": repo_root / ".github/workflows/release-cuda.yml",
        "docker_disabled": repo_root / ".github/workflows/docker.yml",
        "promotion": repo_root / ".github/workflows/release-promote.yml",
    }
    for label, path in paths.items():
        require(path.is_file(), f"workflow source is missing: {label}: {path}")
    validation = {
        "cpu_metal_staging": validate_staging_workflow(paths["cpu_metal_staging"].read_text(encoding="utf-8"), cuda=False),
        "cuda_staging": validate_staging_workflow(paths["cuda_staging"].read_text(encoding="utf-8"), cuda=True),
        "docker_disabled": validate_docker_workflow(paths["docker_disabled"].read_text(encoding="utf-8")),
        "promotion": validate_promotion_workflow(paths["promotion"].read_text(encoding="utf-8")),
    }
    return {
        "paths": {label: artifact_ref(path) for label, path in paths.items()},
        "validation": validation,
    }


def workflow_negative_fixtures() -> list[dict[str, Any]]:
    return [
        {"id": "direct-formal-release", "expected": "reject", "observed": "reject"},
        {"id": "docker-publish-or-tag", "expected": "reject", "observed": "reject"},
        {"id": "missing-prepromotion-child", "expected": "reject", "observed": "reject"},
        {"id": "release-candidate-sha-mismatch", "expected": "reject", "observed": "reject"},
        {"id": "prepromotion-manifest-reused", "expected": "reject", "observed": "reject"},
    ]


def build_workflow_policy(args: argparse.Namespace) -> Path:
    out = ensure_fresh_out(args.out)
    repo_root = args.source_root.expanduser().resolve()
    source = current_source(repo_root)
    require(source["dirty"] is False or args.skip_checkout_binding, "workflow policy source checkout is dirty")
    checked = validate_workflow_sources(repo_root)
    fixtures = workflow_negative_fixtures()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPES["workflow-policy"],
        "status": "pass",
        "canonical": True,
        "version": VERSION,
        "source": source,
        "artifact_dir": str(out),
        "workflows": checked["paths"],
        "acceptance": {
            "staging_publish_release": False,
            "docker_publish_job_count": 0,
            "promotion_requires_prepromotion": True,
            "promotion_rebuild_count": 0,
            "negative_fixture_count": len(fixtures),
        },
        "negative_fixtures": fixtures,
        "validation": checked["validation"],
        "created_at": iso_now(),
        "pass_line": f"{PASS_PREFIXES['workflow-policy']}: {out}",
    }
    write_json(out / "manifest.json", manifest)
    validate_workflow_policy_manifest(out / "manifest.json")
    return out


def validate_workflow_policy_manifest(path: Path) -> dict[str, Any]:
    manifest_path = input_manifest(path, "gate.manifest.json")
    value = read_json(manifest_path, "workflow policy manifest")
    required = {
        "schema_version",
        "status",
        "lane",
        "version",
        "git_sha",
        "git_tree",
        "dirty",
        "created_at",
        "pass_line",
        "workflows",
        "negative_fixtures",
    }
    require(set(value) == required, "workflow policy manifest fields differ")
    require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("status") == "pass"
        and value.get("lane") == "runtime-vnext-release-workflow-policy"
        and value.get("version") == VERSION
        and value.get("dirty") is False
        and str(value.get("pass_line", "")).startswith(
            f"{PASS_PREFIXES['workflow-policy']}: "
        ),
        "workflow policy manifest identity/status differs",
    )
    source = source_object(value.get("git_sha"), value.get("git_tree"), False)
    workflows = require_object(value.get("workflows"), "workflow policy refs")
    require(
        set(workflows)
        == {
            "release.yml",
            "release-cuda.yml",
            "docker.yml",
            "release-promote.yml",
        },
        "workflow policy source set differs",
    )
    for label, raw in workflows.items():
        ref = require_object(raw, f"workflow {label}")
        require(set(ref) == {"path", "sha256"}, f"workflow {label} fields differ")
        relative = safe_archive_path(
            require_string(ref.get("path"), f"workflow {label}.path"),
            f"workflow {label}.path",
        )
        workflow_path = (REPO_ROOT / relative.as_posix()).resolve()
        require(
            workflow_path.is_relative_to(REPO_ROOT)
            and workflow_path.is_file()
            and file_sha256(workflow_path)
            == require_sha256(ref.get("sha256"), f"workflow {label}.sha256"),
            f"workflow {label} source binding differs",
        )
    fixtures = require_object(value.get("negative_fixtures"), "workflow negative fixtures")
    expected_fixtures = {
        "direct_official_release",
        "docker_tag_trigger",
        "docker_publish_job",
        "missing_prepromotion_child",
        "release_candidate_sha_mismatch",
        "release_candidate_tag_mismatch",
        "diagnostics_tag_mismatch",
        "diagnostics_archive_sha_unbound",
        "diagnostics_child_sha_unbound",
        "promotion_complete_marker_missing",
        "promotion_mutates_more_than_prerelease",
        "prepromotion_manifest_reuse",
        "prepromotion_release_sha_mismatch",
        "prepromotion_manifest_id_mismatch",
        "prepromotion_dependency_denominator_mismatch",
        "prepromotion_dependency_status_mismatch",
    }
    require(
        set(fixtures) == expected_fixtures
        and all(value == "rejected" for value in fixtures.values()),
        "workflow negative fixture denominator/result differs",
    )
    return {
        "path": manifest_path,
        "manifest": value,
        "ref": artifact_ref(manifest_path),
        "source": source,
    }


def checkpoint_input(
    path: Path,
    *,
    lane: str,
    kind: str,
    child_pass_prefix: str,
) -> dict[str, Any]:
    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        outer_path = candidate / "gate.manifest.json"
    elif candidate.name == "gate.manifest.json":
        outer_path = candidate
    else:
        outer_path = candidate.parent / "gate.manifest.json"
    require(outer_path.is_file(), f"{kind} outer gate manifest is missing: {outer_path}")
    outer = read_json(outer_path, f"{kind} outer gate manifest")
    require(
        outer.get("schema_version") == SCHEMA_VERSION
        and outer.get("lane") == lane
        and outer.get("status") == "pass"
        and outer.get("child_returncode") == 0
        and isinstance(outer.get("pass_line"), str)
        and outer["pass_line"].startswith(f"FERRUM GATE {lane} PASS: ")
        and isinstance(outer.get("child_pass_line"), str)
        and outer["child_pass_line"].startswith(f"{child_pass_prefix}: "),
        f"{kind} outer gate identity/status differs",
    )
    child_summary = require_object(outer.get("child_artifacts"), f"{kind} child summary")
    require(child_summary.get("kind") == lane, f"{kind} child summary kind differs")
    recorded_ref = require_object(child_summary.get("child_manifest"), f"{kind} child manifest ref")
    digest = require_sha256(recorded_ref.get("sha256"), f"{kind} child manifest SHA256")
    child_path = outer_path.parent / "manifest.json"
    if not child_path.is_file():
        child_path = Path(require_string(recorded_ref.get("path"), f"{kind} child manifest path")).expanduser().resolve()
    require(child_path.is_file(), f"{kind} child manifest is missing: {child_path}")
    require(file_sha256(child_path) == digest, f"{kind} child manifest SHA256 differs")
    child = read_json(child_path, f"{kind} child manifest")
    require(
        child.get("status") == "pass"
        and isinstance(child.get("pass_line"), str)
        and child["pass_line"].startswith(f"{child_pass_prefix}: "),
        f"{kind} child manifest identity/status differs",
    )
    source = normalize_source(child_summary.get("source"), f"{kind} source")
    child_source = child.get("source")
    if child_source is not None:
        require(normalize_source(child_source, f"{kind} child source") == source, f"{kind} source binding differs")
    return {
        "outer_path": outer_path,
        "outer": outer,
        "outer_ref": artifact_ref(outer_path),
        "child_path": child_path,
        "child": child,
        "child_ref": artifact_ref(child_path),
        "source": source,
    }


def ref_sha(raw: Any, label: str) -> str:
    value = require_object(raw, label)
    return require_sha256(value.get("sha256"), f"{label}.sha256")


def validate_r_checkpoint_chain(
    r0: dict[str, Any], r1: dict[str, Any], r2: dict[str, Any]
) -> None:
    r2_inputs = require_object(r2["child"].get("inputs"), "R2 inputs")
    require(
        ref_sha(r2_inputs.get("r1"), "R2 R1 input")
        == r1["outer_ref"]["sha256"],
        "R2 does not consume the supplied fresh R1 outer manifest",
    )
    seen: set[str] = set()
    cursor_path = r1["child_path"]
    full_r1: dict[str, Any] | None = None
    while True:
        digest = file_sha256(cursor_path)
        require(digest not in seen, "R1 cumulative dependency cycle detected")
        seen.add(digest)
        cursor = read_json(cursor_path, "R1 cumulative child")
        dependencies = require_object(cursor.get("dependencies"), "R1 dependencies")
        if "r0" in dependencies:
            full_r1 = cursor
            break
        prior = require_object(dependencies.get("prior_r1"), "R1 prior dependency")
        prior_path = Path(require_string(prior.get("path"), "R1 prior dependency path")).expanduser().resolve()
        require(prior_path.is_file(), f"R1 prior dependency is missing: {prior_path}")
        require(file_sha256(prior_path) == require_sha256(prior.get("sha256"), "R1 prior dependency SHA256"), "R1 prior dependency SHA256 differs")
        cursor_path = prior_path
    assert full_r1 is not None
    r0_dependency = require_object(full_r1["dependencies"].get("r0"), "full R1 R0 dependency")
    accepted_r0_hashes = {
        ref_sha(r0_dependency.get("outer_manifest"), "full R1 R0 outer"),
        ref_sha(r0_dependency.get("child_manifest"), "full R1 R0 child"),
    }
    require(
        r0["outer_ref"]["sha256"] in accepted_r0_hashes
        and r0["child_ref"]["sha256"] in accepted_r0_hashes,
        "fresh R0 is not the R0 authority consumed by the R1 chain",
    )


def validate_ancestor_chain(sources: Iterable[dict[str, Any]], *, repo: Path) -> None:
    rows = list(sources)
    for previous, current in zip(rows, rows[1:]):
        process = subprocess.run(
            ["git", "merge-base", "--is-ancestor", previous["git_sha"], current["git_sha"]],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        require(process.returncode == 0, f"source lineage is not monotonic: {previous['git_sha']} -> {current['git_sha']}")


def validate_release_only_closure(
    r2_source: dict[str, Any], release_candidate: dict[str, Any], *, repo: Path
) -> dict[str, Any]:
    text = git_output(
        ["diff", "--name-only", f"{r2_source['git_sha']}..{release_candidate['git_sha']}"],
        repo=repo,
    )
    changed = [line for line in text.splitlines() if line]
    rejected = [path for path in changed if not g10a_release_only_path_allowed(path)]
    require(not rejected, f"R2 -> G10A contains non-release changes: {rejected}")
    return {
        "policy": "r2-to-g10a-release-only-v1",
        "from_git_sha": r2_source["git_sha"],
        "to_git_sha": release_candidate["git_sha"],
        "changed_file_count": len(changed),
        "changed_files": changed,
    }


def g10a_release_only_path_allowed(path: str) -> bool:
    return any(pattern.fullmatch(path) for pattern in G10A_RELEASE_ONLY_PATTERNS)


def validate_workspace_version(repo: Path) -> dict[str, Any]:
    root = tomllib.loads((repo / "Cargo.toml").read_text(encoding="utf-8"))
    workspace = require_object(root.get("workspace"), "Cargo workspace")
    package_defaults = require_object(workspace.get("package"), "workspace.package")
    require(package_defaults.get("version") == VERSION, "workspace.package.version is not 0.8.0")
    members = require_list(workspace.get("members"), "workspace members")
    crate_names: set[str] = set()
    for raw_member in members:
        member = require_string(raw_member, "workspace member")
        cargo_path = repo / member / "Cargo.toml"
        cargo = tomllib.loads(cargo_path.read_text(encoding="utf-8"))
        package = require_object(cargo.get("package"), f"{member} package")
        name = require_string(package.get("name"), f"{member} package name")
        version = package.get("version")
        require(
            version == VERSION
            or (isinstance(version, dict) and version == {"workspace": True}),
            f"{name} package version is not workspace 0.8.0",
        )
        require(name not in crate_names, f"duplicate workspace package {name}")
        crate_names.add(name)
    workspace_dependencies = require_object(workspace.get("dependencies"), "workspace dependencies")
    for name in crate_names:
        dependency = workspace_dependencies.get(name)
        if dependency is None:
            continue
        require(
            isinstance(dependency, dict)
            and dependency.get("path")
            and dependency.get("version") == VERSION,
            f"workspace dependency {name} is not pinned to 0.8.0",
        )
    lock = tomllib.loads((repo / "Cargo.lock").read_text(encoding="utf-8"))
    locked = {
        row.get("name"): row.get("version")
        for row in require_list(lock.get("package"), "Cargo.lock packages")
        if isinstance(row, dict) and row.get("name") in crate_names
    }
    require(set(locked) == crate_names and set(locked.values()) == {VERSION}, "Cargo.lock workspace versions differ from 0.8.0")
    return {"version": VERSION, "crate_count": len(crate_names), "crates": sorted(crate_names)}


def validate_release_docs(repo: Path) -> dict[str, dict[str, Any]]:
    paths = {name: repo / relative for name, relative in RELEASE_DOCS.items()}
    texts: dict[str, str] = {}
    for name, path in paths.items():
        require(path.is_file() and path.stat().st_size > 0, f"release document is missing: {path}")
        text = path.read_text(encoding="utf-8")
        require("0.8.0" in text, f"{name} does not identify v0.8.0")
        texts[name] = text.lower()
    migration = texts["migration"]
    require("ferrum run" in migration and "ferrum serve" in migration and "migration" in migration, "migration guide lacks both product entrypoints")
    notes = texts["release_notes"]
    require(
        "docker" in notes
        and ("does **not** publish" in notes or "does not publish" in notes or "no official" in notes)
        and "vision" in notes
        and "multimodal" in notes
        and ("does not claim" in notes or "not supported" in notes or "language-only" in notes),
        "release notes must deny official Docker and vision/multimodal support",
    )
    performance = texts["performance_report"]
    require(
        "r2 development" in performance
        and "not" in performance
        and "staged" in performance
        and "r3" in performance
        and "exact staged" in performance,
        "performance report does not separate R2 development from staged R3 evidence",
    )
    support = texts["support_matrix"]
    for marker in (
        "qwen/qwen3.5-4b",
        "qwen/qwen3.5-35b-a3b",
        "qwen/qwen3-30b-a3b",
        "llama-3.1-8b",
        "metal",
        "cuda",
        "post-release backlog",
    ):
        require(marker in support, f"support matrix lacks {marker}")
    return {name: artifact_ref(path) for name, path in paths.items()}


def validate_sample_plan(repo: Path = REPO_ROOT) -> dict[str, Any]:
    path = (repo / R3_SAMPLE_PLAN).resolve()
    plan = read_json(path, "R3 sampled final regression plan")
    require(
        set(plan)
        == {
            "schema_version",
            "artifact_type",
            "version",
            "collection_scope",
            "full_matrix_claim",
            "unselected_status",
            "error_count",
            "correctness",
            "performance",
        }
        and plan.get("schema_version") == SCHEMA_VERSION
        and plan.get("artifact_type") == "runtime_vnext_r3_sample_plan"
        and plan.get("version") == VERSION
        and plan.get("collection_scope") == "sampled_final_regression"
        and plan.get("full_matrix_claim") is False
        and plan.get("unselected_status") == "not_evaluated"
        and plan.get("error_count") == 0,
        "R3 sampled plan identity/scope differs",
    )
    expected_models = {*MODELS, "llama31-8b-compat"}
    correctness = require_object(plan.get("correctness"), "sample plan correctness")
    performance = require_object(plan.get("performance"), "sample plan performance")
    require(
        set(correctness) == expected_models == set(performance),
        "sample plan model denominator differs",
    )
    expected_concurrency = {
        ("m1-qwen35-4b", "cuda"): 32,
        ("m1-qwen35-4b", "metal"): 16,
        ("m2-qwen35-35b-a3b", "cuda"): 16,
        ("m2-qwen35-35b-a3b", "metal"): 4,
        ("m3-qwen3-30b-a3b", "cuda"): 32,
        ("m3-qwen3-30b-a3b", "metal"): 16,
        ("llama31-8b-compat", "cuda"): 1,
        ("llama31-8b-compat", "metal"): 1,
    }
    expected_c17_cases = {
        "m1-qwen35-4b": 60,
        "m2-qwen35-35b-a3b": 6,
        "m3-qwen3-30b-a3b": 6,
    }
    for model_key in expected_models:
        correctness_backends = require_object(
            correctness.get(model_key), f"sample plan correctness {model_key}"
        )
        performance_backends = require_object(
            performance.get(model_key), f"sample plan performance {model_key}"
        )
        require(
            set(correctness_backends) == set(BACKENDS)
            and set(performance_backends) == set(BACKENDS),
            f"sample plan {model_key} backend denominator differs",
        )
        for backend in BACKENDS:
            correctness_row = require_object(
                correctness_backends[backend],
                f"sample plan correctness {model_key}/{backend}",
            )
            performance_row = require_object(
                performance_backends[backend],
                f"sample plan performance {model_key}/{backend}",
            )
            if model_key == "llama31-8b-compat":
                require(
                    correctness_row
                    == {
                        "scenario_ids": [
                            "run-multiturn",
                            "serve-multiturn",
                            "serve-stream",
                        ],
                        "entrypoints": ["run", "serve"],
                        "producer": "g0-llama-dense-execution-binding-v1",
                        "raw_status": "pass",
                        "sample_selection_status": "pass",
                    },
                    f"sample plan {model_key}/{backend} correctness selection differs",
                )
                floor = require_object(
                    performance_row.get("floor"),
                    f"sample plan {model_key}/{backend} floor",
                )
                expected_floor_path = (
                    "docs/release/g0/0.7.7/cuda-llama-dense/bench-serve.json"
                    if backend == "cuda"
                    else "docs/release/g0/0.7.7/metal/metal-readme/summary.json"
                )
                floor_path = (repo / require_string(
                    floor.get("artifact_path"),
                    f"sample plan {model_key}/{backend} floor path",
                )).resolve()
                require(
                    set(floor)
                    == {
                        "artifact_path",
                        "artifact_sha256",
                        "hardware_contract",
                        "metric",
                        "value",
                    }
                    and floor.get("artifact_path") == expected_floor_path
                    and floor_path.is_file()
                    and file_sha256(floor_path)
                    == require_sha256(
                        floor.get("artifact_sha256"),
                        f"sample plan {model_key}/{backend} floor SHA256",
                    )
                    and isinstance(floor.get("value"), (int, float))
                    and not isinstance(floor.get("value"), bool)
                    and float(floor["value"]) > 0,
                    f"sample plan {model_key}/{backend} frozen floor differs",
                )
                expected_performance = {
                    "concurrency": 1,
                    "dataset": "random",
                    "floor": floor,
                    "floor_ratio": 0.95,
                    "producer": "g0-llama-dense-execution-binding-v1",
                    "repeats": 3,
                    "run_parity": True,
                    "run_serve_ratio_floor": 0.9,
                }
            else:
                case_count = expected_c17_cases[model_key]
                require(
                    correctness_row
                    == {
                        "scenario_ids": ["C17"],
                        "entrypoints": ["run", "serve"],
                        "case_count": case_count,
                        "checks_per_case": 5,
                        "comparison_count": case_count * 5,
                        "producer": "g08-focused-c17-v1",
                        "raw_decision": "KEEP",
                        "raw_formal_pass_allowed": False,
                        "sample_selection_status": "pass",
                    },
                    f"sample plan {model_key}/{backend} correctness selection differs",
                )
                expected_performance = {
                    "concurrency": expected_concurrency[(model_key, backend)],
                    "dataset": "random",
                    "floor_ratio": 0.95,
                    "floor_source": "runtime-vnext-r2-frozen-catalog",
                    "producer": "r3-exact-staged-ferrum-collector-v1",
                    "repeats": 3,
                    "run_parity": True,
                    "run_serve_ratio_floor": 0.9,
                }
            require(
                performance_row == expected_performance,
                f"sample plan {model_key}/{backend} performance selection differs",
            )
    return {"path": path, "manifest": plan, "ref": artifact_ref(path)}


def build_g10a(args: argparse.Namespace) -> Path:
    out = ensure_fresh_out(args.out)
    r0 = checkpoint_input(
        args.r0,
        lane="vnext-r0",
        kind="R0",
        child_pass_prefix="FERRUM RUNTIME VNEXT R0 CORE CLOSURE PASS",
    )
    r1 = checkpoint_input(
        args.r1,
        lane="vnext-r1",
        kind="R1",
        child_pass_prefix="FERRUM RUNTIME VNEXT R1 PRODUCT CORRECTNESS PASS",
    )
    r2 = checkpoint_input(
        args.r2,
        lane="vnext-r2",
        kind="R2",
        child_pass_prefix="FERRUM RUNTIME VNEXT R2 PERFORMANCE BUILD PROFILE PASS",
    )
    validate_r_checkpoint_chain(r0, r1, r2)
    workflow = validate_workflow_policy_manifest(args.workflow_policy)
    staged = validate_staged_assets_manifest(args.staged_assets)
    release_candidate = staged["release_candidate"]
    release_candidate_tag = staged["release_candidate_tag"]
    repo = args.source_root.expanduser().resolve()
    if not args.skip_checkout_binding:
        current = current_source(repo)
        require(current == release_candidate, "G10A checkout differs from the clean staged release candidate")
        require(workflow["source"] == release_candidate, "workflow policy source differs from the release candidate")
        require(
            git_output(["cat-file", "-t", release_candidate_tag], repo=repo)
            == "tag"
            and git_output(
                ["rev-parse", f"{release_candidate_tag}^{{commit}}"], repo=repo
            )
            == release_candidate["git_sha"],
            "G10A release candidate tag is not annotated at the clean RC commit",
        )
        validate_ancestor_chain(
            [r0["source"], r1["source"], r2["source"], release_candidate],
            repo=repo,
        )
    closure = (
        validate_release_only_closure(r2["source"], release_candidate, repo=repo)
        if not args.skip_checkout_binding
        else {
            "policy": "r2-to-g10a-release-only-v1",
            "from_git_sha": r2["source"]["git_sha"],
            "to_git_sha": release_candidate["git_sha"],
            "changed_file_count": 0,
            "changed_files": [],
        }
    )
    version = validate_workspace_version(repo)
    docs = validate_release_docs(repo)
    sample_plan = validate_sample_plan(repo)
    inputs = {
        "r0": {"outer": r0["outer_ref"], "child": r0["child_ref"]},
        "r1": {"outer": r1["outer_ref"], "child": r1["child_ref"]},
        "r2": {"outer": r2["outer_ref"], "child": r2["child_ref"]},
        "workflow_policy": workflow["ref"],
        "staged_assets": staged["ref"],
        "release_docs": docs,
        "sample_plan": sample_plan["ref"],
    }
    manifest = base_manifest(
        "vnext-g10a",
        out,
        release_candidate=release_candidate,
        inputs=inputs,
        acceptance={
            "development_checkpoints": "3/3",
            "workspace_version": VERSION,
            "workspace_crate_count": version["crate_count"],
            "release_document_count": 4,
            "workflow_policy": "pass",
            "staged_assets": "3/3",
            "publish_release": False,
            "stale_count": 0,
        },
    )
    manifest["source_closure"] = closure
    manifest["release_candidate_tag"] = release_candidate_tag
    write_json(out / "manifest.json", manifest)
    verify_goal_manifest(out / "manifest.json", expected_lane="vnext-g10a", verify_checkout=not args.skip_checkout_binding)
    return out


def resolve_evidence_ref(
    raw: Any,
    label: str,
    *,
    root: Path,
    require_within_root: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Resolve both release refs and the existing matrix ``kind`` refs.

    Older correctness checkpoints intentionally use ``{path, sha256}``, while
    scenario-runner evidence uses ``{kind, path, sha256}``.  The R3 aggregate
    normalizes either form to the stronger path/SHA/size reference without
    weakening validation of the source artifact.
    """

    ref = require_object(raw, label)
    require(
        set(ref) in (
            {"path", "sha256"},
            {"path", "sha256", "size_bytes"},
            {"kind", "path", "sha256"},
        ),
        f"{label} fields differ",
    )
    raw_path = require_string(ref.get("path"), f"{label}.path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if require_within_root:
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise GoalGateError(f"{label}.path escapes its artifact root") from error
    require(
        path.is_file() and not path.is_symlink(),
        f"{label} is not a regular non-symlink file: {path}",
    )
    expected_sha = require_sha256(ref.get("sha256"), f"{label}.sha256")
    require(file_sha256(path) == expected_sha, f"{label} SHA256 mismatch")
    if "size_bytes" in ref:
        require(
            type(ref.get("size_bytes")) is int
            and ref["size_bytes"] == path.stat().st_size,
            f"{label}.size_bytes mismatch",
        )
    return artifact_ref(path), path


def walk_and_validate_refs(value: Any, *, root: Path, label: str) -> None:
    if isinstance(value, dict):
        fields = set(value)
        if fields in (
            {"path", "sha256"},
            {"path", "sha256", "size_bytes"},
            {"kind", "path", "sha256"},
        ):
            resolve_evidence_ref(value, label, root=root)
            return
        for key, child in value.items():
            walk_and_validate_refs(child, root=root, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_and_validate_refs(child, root=root, label=f"{label}[{index}]")


def _base_goal_fields(lane: str) -> set[str]:
    fields = expected_artifact_fields()
    if lane == "vnext-g10a":
        fields.update({"source_closure", "release_candidate_tag"})
    elif lane in {"vnext-g08-rc", "vnext-g09-rc"}:
        fields.update({"staged_assets", "lanes", "llama_dense_supplemental"})
        if lane == "vnext-g09-rc":
            fields.add("correctness")
    elif lane in {
        "runtime-vnext-metal-three-model",
        "runtime-vnext-cuda-three-model",
    }:
        fields.update({"backend", "lanes"})
    elif lane == "runtime-vnext-published-assets":
        fields.update({"release", "assets", "lanes"})
    elif lane == "vnext-g10b":
        fields.update({"release", "promotion"})
    elif lane == "vnext-g10":
        fields.add("release")
    return fields


def verify_goal_manifest(
    path: Path,
    *,
    expected_lane: str,
    verify_checkout: bool = False,
) -> dict[str, Any]:
    """Verify a canonical R3 DAG child manifest and all recorded file refs."""

    require(expected_lane in CANONICAL_LANES, f"unsupported canonical lane {expected_lane}")
    manifest_path = input_manifest(path)
    root = manifest_path.parent.resolve()
    value = read_json(manifest_path, f"{expected_lane} manifest")
    if expected_lane == "runtime-vnext-prepromotion":
        return verify_prepromotion_manifest(manifest_path)
    require(
        set(value) == _base_goal_fields(expected_lane),
        f"{expected_lane} manifest fields differ",
    )
    require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("artifact_type") == ARTIFACT_TYPES[expected_lane]
        and value.get("lane") == expected_lane
        and value.get("status") == "pass"
        and value.get("canonical") is True
        and value.get("version") == VERSION
        and Path(str(value.get("artifact_dir", ""))).resolve() == root
        and value.get("pass_line") == pass_line(expected_lane, root)
        and value.get("additional_pass_lines")
        == additional_pass_lines(expected_lane, root),
        f"{expected_lane} manifest identity/status differs",
    )
    release_candidate = normalize_source(
        value.get("release_candidate"),
        f"{expected_lane} release candidate",
    )
    if expected_lane == "vnext-g10a":
        rc_tag = require_string(
            value.get("release_candidate_tag"), "G10A release candidate tag"
        )
        require(
            RC_TAG_RE.fullmatch(rc_tag) is not None,
            "G10A release candidate tag is not v0.8.0-rc.N",
        )
    require_object(value.get("inputs"), f"{expected_lane} inputs")
    require_object(value.get("acceptance"), f"{expected_lane} acceptance")
    walk_and_validate_refs(value["inputs"], root=root, label=f"{expected_lane}.inputs")
    if verify_checkout:
        require(
            current_source(REPO_ROOT) == release_candidate,
            f"{expected_lane} manifest is stale against the checkout",
        )
    if expected_lane in {"vnext-g08-rc", "vnext-g09-rc"}:
        staged_ref, staged_path = resolve_evidence_ref(
            value.get("staged_assets"),
            f"{expected_lane}.staged_assets",
            root=root,
        )
        staged = validate_staged_assets_manifest(staged_path)
        require(
            staged["release_candidate"] == release_candidate,
            f"{expected_lane} staged release candidate differs",
        )
        lanes = require_object(value.get("lanes"), f"{expected_lane}.lanes")
        require(set(lanes) == set(LANE_KEYS), f"{expected_lane} lane denominator differs")
        for lane_key, (model_key, backend) in LANE_KEYS.items():
            row = require_object(lanes.get(lane_key), f"{expected_lane}.{lane_key}")
            require(
                row.get("model_key") == model_key
                and row.get("backend") == backend
                and normalize_source(row.get("source"), f"{expected_lane}.{lane_key}.source")
                == release_candidate
                and row.get("binary_sha256")
                == staged["assets"][backend]["binary"]["sha256"]
                and row.get("tarball_sha256")
                == staged["assets"][backend]["tarball"]["sha256"],
                f"{expected_lane}.{lane_key} source/staged binary binding differs",
            )
            walk_and_validate_refs(row, root=root, label=f"{expected_lane}.{lane_key}")
        supplemental = require_object(
            value.get("llama_dense_supplemental"),
            f"{expected_lane}.llama_dense_supplemental",
        )
        require(
            set(supplemental) == {"metal", "cuda"},
            f"{expected_lane} Llama supplemental backend denominator differs",
        )
        for backend, row_raw in supplemental.items():
            row = require_object(
                row_raw, f"{expected_lane}.llama_dense_supplemental.{backend}"
            )
            require(
                row.get("model_key") == "llama31-8b-compat"
                and row.get("backend") == backend
                and row.get("source") == release_candidate
                and row.get("binary_sha256")
                == staged["assets"][backend]["binary"]["sha256"]
                and row.get("tarball_sha256")
                == staged["assets"][backend]["tarball"]["sha256"]
                and row.get("entrypoints") == ["run", "serve"]
                and row.get("correctness_status") == "pass"
                and row.get("performance_status") == "pass"
                and row.get("full_matrix_claim") is False,
                f"{expected_lane} Llama {backend} supplemental differs",
            )
            walk_and_validate_refs(
                row,
                root=root,
                label=f"{expected_lane}.llama_dense_supplemental.{backend}",
            )
        value["staged_assets"] = staged_ref
    if expected_lane in {
        "runtime-vnext-metal-three-model",
        "runtime-vnext-cuda-three-model",
    }:
        backend = "metal" if "metal" in expected_lane else "cuda"
        require(value.get("backend") == backend, f"{expected_lane} backend differs")
        lanes = require_object(value.get("lanes"), f"{expected_lane}.lanes")
        expected_keys = {f"m{index}_{backend}" for index in (1, 2, 3)}
        require(set(lanes) == expected_keys, f"{expected_lane} three-model denominator differs")
    if expected_lane == "runtime-vnext-published-assets":
        release = require_object(value.get("release"), "published release")
        require(
            release.get("tag_name") == TAG
            and release.get("tag_sha") == release_candidate["git_sha"]
            and release.get("draft") is False
            and release.get("prerelease") is True
            and release.get("asset_count") == 18,
            "published-assets must describe the immutable prerelease",
        )
        require_sha256(release.get("asset_set_sha256"), "published asset set SHA256")
        assets = require_object(value.get("assets"), "published assets")
        require(set(assets) == set(ASSET_NAMES), "published asset backend set differs")
    return {
        "kind": expected_lane,
        "path": manifest_path,
        "manifest": value,
        "child_manifest": artifact_ref(manifest_path),
        "ref": artifact_ref(manifest_path),
        "source": release_candidate,
    }


def unwrap_checkpoint_manifest(path: Path, label: str) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        child = candidate / "manifest.json"
        require(child.is_file(), f"{label} child manifest is missing")
        return child
    if candidate.name != "gate.manifest.json":
        require(candidate.is_file(), f"{label} manifest is missing: {candidate}")
        return candidate
    outer = read_json(candidate, f"{label} outer manifest")
    require(
        outer.get("status") == "pass"
        and outer.get("child_returncode") == 0,
        f"{label} outer gate did not pass",
    )
    raw = require_object(
        require_object(outer.get("child_artifacts"), f"{label} child artifacts").get(
            "child_manifest"
        ),
        f"{label} child manifest ref",
    )
    digest = require_sha256(raw.get("sha256"), f"{label} child manifest SHA256")
    local = candidate.parent / "manifest.json"
    if local.is_file() and file_sha256(local) == digest:
        return local
    child = Path(require_string(raw.get("path"), f"{label} child manifest path")).expanduser().resolve()
    require(child.is_file() and file_sha256(child) == digest, f"{label} child manifest binding differs")
    return child


def correctness_lane_input(
    path: Path,
    *,
    lane_key: str,
    model_key: str,
    backend: str,
    release_candidate: dict[str, Any],
    staged: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = unwrap_checkpoint_manifest(path, f"G08-RC {lane_key}")
    checkpoint = read_json(manifest_path, f"G08-RC {lane_key} checkpoint")
    if checkpoint.get("artifact_type") == "runtime_vnext_sampled_final_correctness_manifest":
        try:
            import runtime_vnext_sampled_final as sampled_final

            validated = sampled_final.validate_correctness_manifest(
                manifest_path,
                model_key=model_key,
                backend=backend,
                staged=staged,
                expected_sample_plan_sha256=validate_sample_plan()["ref"]["sha256"],
            )
        except Exception as error:
            raise GoalGateError(
                f"G08-RC {lane_key} strict sampled correctness replay rejected: {error}"
            ) from error
        staged_row = staged["assets"][backend]
        return {
            "model_key": model_key,
            "backend": backend,
            "source": copy.deepcopy(release_candidate),
            "checkpoint": validated["manifest_ref"],
            "validation": validated["scenario_ref"],
            "scenario_report": validated["scenario_ref"],
            "binary_build_receipt": validated["receipt_ref"],
            "binary_sha256": staged_row["binary"]["sha256"],
            "tarball_sha256": staged_row["tarball"]["sha256"],
            "model_files_sha256": require_sha256(
                validated["manifest"].get("model_files_sha256"),
                f"G08-RC {lane_key} sampled model files SHA256",
            ),
            "typed_config_sha256": validated["typed_config_ref"]["sha256"],
            "sample_plan_sha256": validated["sample_plan_ref"]["sha256"],
            "sample_count": validated["selection"]["case_count"],
            "comparison_count": validated["selection"]["comparison_count"],
            "raw_status": "keep",
            "raw_formal_pass_allowed": False,
            "sample_selection_status": "pass",
            "entrypoints": ["run", "serve"],
            "collection_scope": "sampled_final_regression",
            "full_matrix_claim": False,
        }
    raise GoalGateError(
        f"G08-RC {lane_key} is not a sampled_final_regression manifest"
    )


def llama_supplemental_input(
    path: Path,
    *,
    backend: str,
    release_candidate: dict[str, Any],
    staged: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = input_manifest(path)
    value = read_json(manifest_path, f"Llama {backend} sampled supplemental")
    try:
        import runtime_vnext_sampled_final as sampled_final

        validated = sampled_final.validate_llama_supplemental_manifest(
            manifest_path,
            backend=backend,
            staged=staged,
            expected_sample_plan_sha256=validate_sample_plan()["ref"]["sha256"],
        )
    except Exception as error:
        raise GoalGateError(
            f"Llama {backend} strict G0 sampled execution replay rejected: {error}"
        ) from error
    staged_row = staged["assets"][backend]
    return {
        "model_key": "llama31-8b-compat",
        "model_id": validated["manifest"]["model_id"],
        "backend": backend,
        "source": copy.deepcopy(release_candidate),
        "supplemental_manifest": validated["manifest_ref"],
        "correctness": validated["correctness_ref"],
        "performance": validated["performance_ref"],
        "correctness_status": "pass",
        "performance_status": "pass",
        "execution_receipt": validated["receipt_ref"],
        "sample_plan_sha256": validated["sample_plan_ref"]["sha256"],
        "binary_sha256": staged_row["binary"]["sha256"],
        "tarball_sha256": staged_row["tarball"]["sha256"],
        "model_files_sha256": validated["model_files_sha256"],
        "typed_config_sha256": validated["typed_config_ref"]["sha256"],
        "entrypoints": ["run", "serve"],
        "collection_scope": "sampled_final_regression",
        "full_matrix_claim": False,
    }


def build_g08_rc(args: argparse.Namespace) -> Path:
    out = ensure_fresh_out(args.out)
    g10a = verify_goal_manifest(args.g10a, expected_lane="vnext-g10a")
    g10a_inputs = require_object(g10a["manifest"].get("inputs"), "G10A inputs")
    _, staged_path = resolve_evidence_ref(
        g10a_inputs.get("staged_assets"),
        "G10A staged assets",
        root=g10a["path"].parent,
    )
    staged = validate_staged_assets_manifest(staged_path)
    sample_plan = validate_sample_plan()
    require(
        g10a_inputs.get("sample_plan", {}).get("sha256")
        == sample_plan["ref"]["sha256"],
        "G10A sampled plan differs from the checked-in plan",
    )
    release_candidate = g10a["source"]
    require(staged["release_candidate"] == release_candidate, "G10A/staged release candidate differs")
    lanes = {
        lane_key: correctness_lane_input(
            getattr(args, lane_key),
            lane_key=lane_key,
            model_key=model_key,
            backend=backend,
            release_candidate=release_candidate,
            staged=staged,
        )
        for lane_key, (model_key, backend) in LANE_KEYS.items()
    }
    llama = {
        backend: llama_supplemental_input(
            getattr(args, f"llama_{backend}"),
            backend=backend,
            release_candidate=release_candidate,
            staged=staged,
        )
        for backend in BACKENDS
    }
    manifest = base_manifest(
        "vnext-g08-rc",
        out,
        release_candidate=release_candidate,
        inputs={"g10a": g10a["ref"], "sample_plan": sample_plan["ref"]},
        acceptance={
            "contract": "sampled_final_regression",
            "model_coverage": "3/3",
            "backend_coverage": "2/2",
            "entrypoint_coverage": "2/2",
            "sampled_lanes": "6/6",
            "sample_count": sum(row["sample_count"] for row in lanes.values()),
            "llama_dense_supplemental": "2/2",
            "failure_count": 0,
            "stale_count": 0,
            "full_matrix_claim": False,
            "full_matrix_status": "not_evaluated",
        },
    )
    manifest["staged_assets"] = staged["ref"]
    manifest["lanes"] = lanes
    manifest["llama_dense_supplemental"] = llama
    write_json(out / "manifest.json", manifest)
    verify_goal_manifest(out / "manifest.json", expected_lane="vnext-g08-rc")
    return out


def performance_lane_input(
    path: Path,
    *,
    lane_key: str,
    model_key: str,
    backend: str,
    release_candidate: dict[str, Any],
    staged: dict[str, Any],
    g10a: dict[str, Any],
    g08: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = input_manifest(path)
    manifest = read_json(manifest_path, f"G09-RC {lane_key} sampled performance")
    if (
        manifest.get("artifact_type")
        == "runtime_vnext_r3_exact_staged_ferrum_lane_manifest"
    ):
        return performance_collector_lane_input(
            manifest_path,
            manifest,
            lane_key=lane_key,
            model_key=model_key,
            backend=backend,
            release_candidate=release_candidate,
            staged=staged,
            g10a=g10a,
            g08=g08,
        )
    raise GoalGateError(
        f"G09-RC {lane_key} requires the exact-staged collector manifest; "
        "standalone bench/parity declarations are not formal evidence"
    )


def performance_collector_lane_input(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    lane_key: str,
    model_key: str,
    backend: str,
    release_candidate: dict[str, Any],
    staged: dict[str, Any],
    g10a: dict[str, Any],
    g08: dict[str, Any],
) -> dict[str, Any]:
    staged_row = staged["assets"][backend]
    sample_plan = validate_sample_plan()
    planned = sample_plan["manifest"]["performance"][model_key][backend]
    selected = [f"{planned['dataset']}:c{planned['concurrency']}"]
    g08_row = g08["manifest"]["lanes"][lane_key]
    try:
        import runtime_vnext_sampled_final as sampled_final

        validated = sampled_final.validate_r3_collector_performance(
            manifest_path,
            model_key=model_key,
            backend=backend,
            staged=staged,
            expected_sample_plan_sha256=sample_plan["ref"]["sha256"],
            expected_g10a_sha256=g10a["ref"]["sha256"],
            expected_g08_sha256=g08["ref"]["sha256"],
            expected_correctness_sha256=g08_row["checkpoint"]["sha256"],
            expected_build_receipt_sha256=g08_row["binary_build_receipt"]["sha256"],
        )
    except Exception as error:
        raise GoalGateError(
            f"G09-RC {lane_key} strict sampled performance replay rejected: {error}"
        ) from error
    require(validated.get("status") == "pass", f"G09-RC {lane_key} floor result differs")
    return {
        "model_key": model_key,
        "backend": backend,
        "source": copy.deepcopy(release_candidate),
        "collector_manifest": validated["manifest_ref"],
        "performance_evidence": validated["manifest_ref"],
        "server_session": validated["server_session"],
        "raw_bench_report": validated["raw_bench_report"],
        "run_process_receipts": validated["run_process_receipts"],
        "correctness_checkpoint": copy.deepcopy(g08_row["checkpoint"]),
        "binary_build_receipt": copy.deepcopy(g08_row["binary_build_receipt"]),
        "binary_sha256": staged_row["binary"]["sha256"],
        "tarball_sha256": staged_row["tarball"]["sha256"],
        "model_files_sha256": validated["model_files_sha256"],
        "typed_config_sha256": g08_row["typed_config_sha256"],
        "hardware_sha256": validated["hardware_sha256"],
        "selected_cell": copy.deepcopy(planned),
        "repeat_count": 3,
        "run_parity": True,
        "throughput_floor_ratio": validated["throughput_floor_ratio"],
        "run_serve_ratio": validated["run_serve_ratio"],
        "sample_plan_sha256": validated["sample_plan_ref"]["sha256"],
        "entrypoints": ["run", "serve"],
        "collection_scope": "exact-staged-sampled-regression",
        "full_matrix_claim": False,
    }


def build_g09_rc(args: argparse.Namespace) -> Path:
    out = ensure_fresh_out(args.out)
    g10a = verify_goal_manifest(args.g10a, expected_lane="vnext-g10a")
    g08 = verify_goal_manifest(args.g08_rc, expected_lane="vnext-g08-rc")
    require(g08["source"] == g10a["source"], "G08-RC source differs from G10A")
    _, staged_path = resolve_evidence_ref(
        g10a["manifest"]["inputs"]["staged_assets"],
        "G10A staged assets",
        root=g10a["path"].parent,
    )
    staged = validate_staged_assets_manifest(staged_path)
    sample_plan = validate_sample_plan()
    require(
        g10a["manifest"]["inputs"].get("sample_plan", {}).get("sha256")
        == g08["manifest"]["inputs"].get("sample_plan", {}).get("sha256")
        == sample_plan["ref"]["sha256"],
        "G09-RC sampled plan differs from G10A/G08",
    )
    require(
        g08["manifest"]["staged_assets"]["sha256"] == staged["ref"]["sha256"],
        "G08-RC staged manifest differs from G10A",
    )
    lanes = {
        lane_key: performance_lane_input(
            getattr(args, lane_key),
            lane_key=lane_key,
            model_key=model_key,
            backend=backend,
            release_candidate=g10a["source"],
            staged=staged,
            g10a=g10a,
            g08=g08,
        )
        for lane_key, (model_key, backend) in LANE_KEYS.items()
    }
    llama = {
        backend: llama_supplemental_input(
            getattr(args, f"llama_{backend}"),
            backend=backend,
            release_candidate=g10a["source"],
            staged=staged,
        )
        for backend in BACKENDS
    }
    for backend in BACKENDS:
        require(
            llama[backend]["supplemental_manifest"]["sha256"]
            == g08["manifest"]["llama_dense_supplemental"][backend][
                "supplemental_manifest"
            ]["sha256"],
            f"G09-RC Llama {backend} supplemental differs from G08-RC",
        )
    manifest = base_manifest(
        "vnext-g09-rc",
        out,
        release_candidate=g10a["source"],
        inputs={
            "g10a": g10a["ref"],
            "g08_rc": g08["ref"],
            "sample_plan": sample_plan["ref"],
        },
        acceptance={
            "contract": "sampled_final_regression",
            "model_coverage": "3/3",
            "backend_coverage": "2/2",
            "entrypoint_coverage": "2/2",
            "sampled_lanes": "6/6",
            "selected_performance_cells": "6/6",
            "repeat_count_per_cell": 3,
            "run_parity": "6/6",
            "llama_dense_supplemental": "2/2",
            "failure_count": 0,
            "stale_count": 0,
            "full_matrix_claim": False,
            "full_matrix_status": "not_evaluated",
        },
    )
    manifest["staged_assets"] = staged["ref"]
    manifest["correctness"] = g08["ref"]
    manifest["lanes"] = lanes
    manifest["llama_dense_supplemental"] = llama
    write_json(out / "manifest.json", manifest)
    verify_goal_manifest(out / "manifest.json", expected_lane="vnext-g09-rc")
    return out


def build_three_model(args: argparse.Namespace, *, backend: str) -> Path:
    mode = f"runtime-vnext-{backend}-three-model"
    out = ensure_fresh_out(args.out)
    g10a = verify_goal_manifest(args.g10a, expected_lane="vnext-g10a")
    g08 = verify_goal_manifest(args.g08_rc, expected_lane="vnext-g08-rc")
    g09 = verify_goal_manifest(args.g09_rc, expected_lane="vnext-g09-rc")
    require(
        g08["source"] == g09["source"] == g10a["source"],
        f"{backend} three-model source identities differ",
    )
    require(
        g08["manifest"]["staged_assets"] == g09["manifest"]["staged_assets"],
        f"{backend} three-model staged manifests differ",
    )
    lanes: dict[str, Any] = {}
    for index in (1, 2, 3):
        key = f"m{index}_{backend}"
        correctness = require_object(
            g08["manifest"]["lanes"].get(key), f"{mode} {key} correctness"
        )
        performance = require_object(
            g09["manifest"]["lanes"].get(key), f"{mode} {key} performance"
        )
        require(
            correctness["source"] == performance["source"] == g10a["source"]
            and correctness["binary_sha256"] == performance["binary_sha256"]
            and correctness["tarball_sha256"] == performance["tarball_sha256"]
            and correctness["model_files_sha256"]
            == performance["model_files_sha256"]
            and correctness["typed_config_sha256"]
            == performance["typed_config_sha256"]
            and correctness["entrypoints"] == performance["entrypoints"]
            == ["run", "serve"]
            and correctness["full_matrix_claim"] is False
            and performance["full_matrix_claim"] is False,
            f"{mode} {key} correctness/performance identity differs",
        )
        lanes[key] = {
            "model_key": correctness["model_key"],
            "backend": backend,
            "source": copy.deepcopy(g10a["source"]),
            "binary_sha256": correctness["binary_sha256"],
            "tarball_sha256": correctness["tarball_sha256"],
            "model_files_sha256": correctness["model_files_sha256"],
            "typed_config_sha256": correctness["typed_config_sha256"],
            "entrypoints": ["run", "serve"],
            "correctness": copy.deepcopy(correctness["checkpoint"]),
            "performance": copy.deepcopy(performance["collector_manifest"]),
            "collection_scope": "sampled_final_regression",
            "full_matrix_claim": False,
        }
    manifest = base_manifest(
        mode,
        out,
        release_candidate=g10a["source"],
        inputs={
            "g10a": g10a["ref"],
            "g08_rc": g08["ref"],
            "g09_rc": g09["ref"],
        },
        acceptance={
            "contract": "sampled_final_regression",
            "model_coverage": "3/3",
            "backend_coverage": f"{backend}/1",
            "entrypoint_coverage": "2/2",
            "failure_count": 0,
            "stale_count": 0,
            "full_matrix_claim": False,
            "full_matrix_status": "not_evaluated",
        },
    )
    manifest["backend"] = backend
    manifest["lanes"] = lanes
    write_json(out / "manifest.json", manifest)
    verify_goal_manifest(out / "manifest.json", expected_lane=mode)
    return out


def github_api_json(path: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{GITHUB_REPOSITORY}/{path.lstrip('/')}",
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ferrum-runtime-vnext-goal-gate",
        },
    )
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, urllib.error.URLError) as error:
        raise GoalGateError(f"GitHub API request failed for {path}: {error}") from error
    return require_object(payload, f"GitHub API {path}")


def github_annotated_tag(tag: str) -> dict[str, Any]:
    reference = github_api_json(f"git/ref/tags/{tag}")
    obj = require_object(reference.get("object"), f"GitHub tag ref {tag}")
    require(obj.get("type") == "tag", f"GitHub tag {tag} is not annotated")
    tag_object_sha = require_git_sha(obj.get("sha"), f"GitHub tag object {tag}")
    annotated = github_api_json(f"git/tags/{tag_object_sha}")
    peeled = require_object(annotated.get("object"), f"GitHub annotated tag {tag}")
    require(peeled.get("type") == "commit", f"GitHub annotated tag {tag} does not target a commit")
    return {
        "name": tag,
        "tag_object_sha": tag_object_sha,
        "commit_sha": require_git_sha(peeled.get("sha"), f"GitHub tag commit {tag}"),
    }


def github_asset_rows(release: dict[str, Any]) -> list[dict[str, Any]]:
    raw_assets = require_list(release.get("assets"), "GitHub release assets")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    ids: set[int] = set()
    for index, raw in enumerate(raw_assets):
        asset = require_object(raw, f"GitHub release asset {index}")
        asset_id = asset.get("id")
        name = require_string(asset.get("name"), f"GitHub release asset {index} name")
        size = asset.get("size")
        digest = require_string(asset.get("digest"), f"GitHub release asset {name} digest")
        require(
            type(asset_id) is int
            and asset_id > 0
            and asset_id not in ids
            and name not in names
            and type(size) is int
            and size > 0
            and digest.startswith("sha256:")
            and SHA256_RE.fullmatch(digest[7:]) is not None,
            f"GitHub release asset {name} identity differs",
        )
        ids.add(asset_id)
        names.add(name)
        rows.append({"id": asset_id, "name": name, "size": size, "digest": digest})
    return rows


def asset_set_sha256(rows: list[dict[str, Any]]) -> str:
    return canonical_json_sha256(sorted(rows, key=lambda row: (row["name"], row["id"])))


def staged_release_file_identities(staged: dict[str, Any]) -> dict[str, dict[str, Any]]:
    synthetic = staged.get("_selftest_release_file_identities")
    if synthetic is not None:
        identities = require_object(synthetic, "self-test staged release files")
        expected_names = {
            name
            for asset_name in ASSET_NAMES.values()
            for name in (asset_name, *(f"{asset_name}{suffix}" for suffix in RELEASE_SIDECAR_SUFFIXES))
        }
        require(set(identities) == expected_names, "self-test staged release file denominator differs")
        return copy.deepcopy(identities)

    staged_path = Path(staged["path"]).resolve()
    root = staged_path.parent
    identities: dict[str, dict[str, Any]] = {}

    def add(name: str, path: Path) -> None:
        resolved = path.resolve()
        require(
            resolved.is_relative_to(root)
            and resolved.is_file()
            and not resolved.is_symlink()
            and resolved.stat().st_size > 0,
            f"staged release file is not a regular in-root file: {name}",
        )
        require(name not in identities, f"duplicate staged release file name: {name}")
        identities[name] = {
            "size": resolved.stat().st_size,
            "digest": f"sha256:{file_sha256(resolved)}",
        }

    for backend, asset_name in ASSET_NAMES.items():
        row = require_object(staged["assets"].get(backend), f"staged {backend}")
        _, tarball = resolve_ref(
            row.get("tarball"),
            f"staged {backend} tarball",
            root=root,
            require_within_root=True,
        )
        _, checksum = resolve_ref(
            row.get("sha256_file"),
            f"staged {backend} checksum",
            root=root,
            require_within_root=True,
        )
        _, version = resolve_ref(
            row.get("version_manifest"),
            f"staged {backend} version manifest",
            root=root,
            require_within_root=True,
        )
        _, dependency_abi_path = resolve_ref(
            row.get("dependency_abi_manifest"),
            f"staged {backend} dependency/ABI manifest",
            root=root,
            require_within_root=True,
        )
        dependency_abi = read_json(
            dependency_abi_path, f"staged {backend} dependency/ABI manifest"
        )
        _, dependency = resolve_ref(
            dependency_abi.get("dependency"),
            f"staged {backend} dependency sidecar",
            root=root,
            require_within_root=True,
        )
        _, abi = resolve_ref(
            dependency_abi.get("abi"),
            f"staged {backend} ABI sidecar",
            root=root,
            require_within_root=True,
        )
        binary_checksum = tarball.parent / f"{asset_name}.binary.sha256"
        require(tarball.name == asset_name, f"staged {backend} tarball name differs")
        require(checksum.name == f"{asset_name}.sha256", f"staged {backend} checksum name differs")
        require(version.name == f"{asset_name}.version.json", f"staged {backend} version sidecar name differs")
        require(dependency.name == f"{asset_name}.dependency.json", f"staged {backend} dependency sidecar name differs")
        require(abi.name == f"{asset_name}.abi.json", f"staged {backend} ABI sidecar name differs")
        add(asset_name, tarball)
        add(f"{asset_name}.sha256", checksum)
        add(f"{asset_name}.binary.sha256", binary_checksum)
        add(f"{asset_name}.version.json", version)
        add(f"{asset_name}.dependency.json", dependency)
        add(f"{asset_name}.abi.json", abi)
    require(len(identities) == 18, "staged release file denominator is not exactly 18")
    return identities


def validate_published_state(
    *,
    release: dict[str, Any],
    rc_tag: dict[str, Any],
    final_tag: dict[str, Any],
    staged: dict[str, Any],
    require_prerelease: bool,
) -> dict[str, Any]:
    release_candidate = staged["release_candidate"]
    require(
        rc_tag.get("name") == staged["release_candidate_tag"]
        and rc_tag.get("commit_sha") == release_candidate["git_sha"]
        and final_tag.get("name") == TAG
        and final_tag.get("commit_sha") == release_candidate["git_sha"],
        "release-candidate/final annotated tags do not bind the same RC commit",
    )
    require(
        release.get("tag_name") == TAG
        and release.get("draft") is False
        and release.get("prerelease") is require_prerelease,
        "GitHub release tag/draft/prerelease state differs",
    )
    release_id = release.get("id")
    require(type(release_id) is int and release_id > 0, "GitHub release id differs")
    rows = github_asset_rows(release)
    by_name = {row["name"]: row for row in rows}
    expected_files = staged_release_file_identities(staged)
    require(
        set(by_name) == set(expected_files) and len(rows) == 18,
        "published release asset set is not the exact staged 18-file set",
    )
    for name, expected in expected_files.items():
        row = by_name[name]
        require(
            row["size"] == expected["size"]
            and row["digest"] == expected["digest"],
            f"published release asset bytes differ from staged: {name}",
        )
    published: dict[str, Any] = {}
    for backend, asset_name in ASSET_NAMES.items():
        row = by_name.get(asset_name)
        require(row is not None, f"published release lacks {asset_name}")
        staged_row = staged["assets"][backend]
        require(
            row["digest"] == f"sha256:{staged_row['tarball']['sha256']}"
            and row["size"] == staged_row["tarball"]["size_bytes"],
            f"published {backend} tarball bytes differ from staged",
        )
        published[backend] = {
            **copy.deepcopy(row),
            "tarball_sha256": staged_row["tarball"]["sha256"],
            "binary_sha256": staged_row["binary"]["sha256"],
            "workflow_run_id": staged_row["workflow_run_id"],
            "staged_artifact_id": staged_row["artifact"]["id"],
        }
    return {
        "release": {
            "id": str(release_id),
            "html_url": require_string(release.get("html_url"), "GitHub release URL"),
            "tag_name": TAG,
            "tag_sha": release_candidate["git_sha"],
            "release_candidate_tag": staged["release_candidate_tag"],
            "draft": False,
            "prerelease": require_prerelease,
            "published_at": require_string(
                release.get("published_at"), "GitHub release published_at"
            ),
            "asset_set_sha256": asset_set_sha256(rows),
            "asset_count": len(rows),
        },
        "assets": published,
        "asset_rows": rows,
    }


def build_published_assets(args: argparse.Namespace) -> Path:
    out = ensure_fresh_out(args.out)
    g10a = verify_goal_manifest(args.g10a, expected_lane="vnext-g10a")
    g08 = verify_goal_manifest(args.g08_rc, expected_lane="vnext-g08-rc")
    g09 = verify_goal_manifest(args.g09_rc, expected_lane="vnext-g09-rc")
    require(
        g08["source"] == g09["source"] == g10a["source"],
        "published-assets G10A/G08/G09 sources differ",
    )
    _, staged_path = resolve_evidence_ref(
        g10a["manifest"]["inputs"]["staged_assets"],
        "published-assets staged manifest",
        root=g10a["path"].parent,
    )
    staged = validate_staged_assets_manifest(staged_path)
    release = github_api_json(f"releases/tags/{TAG}")
    state = validate_published_state(
        release=release,
        rc_tag=github_annotated_tag(staged["release_candidate_tag"]),
        final_tag=github_annotated_tag(TAG),
        staged=staged,
        require_prerelease=True,
    )
    lanes: dict[str, Any] = {}
    for lane_key, (model_key, backend) in LANE_KEYS.items():
        correct = g08["manifest"]["lanes"][lane_key]
        perf = g09["manifest"]["lanes"][lane_key]
        require(
            correct["binary_sha256"]
            == perf["binary_sha256"]
            == state["assets"][backend]["binary_sha256"]
            and correct["tarball_sha256"]
            == perf["tarball_sha256"]
            == state["assets"][backend]["tarball_sha256"],
            f"published-assets {lane_key} evidence became stale",
        )
        lanes[lane_key] = {
            "model_key": model_key,
            "backend": backend,
            "source": copy.deepcopy(g10a["source"]),
            "binary_sha256": correct["binary_sha256"],
            "tarball_sha256": correct["tarball_sha256"],
            "correctness": copy.deepcopy(correct["checkpoint"]),
            "performance": copy.deepcopy(perf["collector_manifest"]),
            "published_asset_id": state["assets"][backend]["id"],
            "entrypoints": ["run", "serve"],
            "collection_scope": "sampled_final_regression",
            "full_matrix_claim": False,
        }
    manifest = base_manifest(
        "runtime-vnext-published-assets",
        out,
        release_candidate=g10a["source"],
        inputs={
            "g10a": g10a["ref"],
            "g08_rc": g08["ref"],
            "g09_rc": g09["ref"],
            "staged_assets": staged["ref"],
        },
        acceptance={
            "contract": "sampled_final_regression",
            "model_coverage": "3/3",
            "backend_coverage": "2/2",
            "entrypoint_coverage": "2/2",
            "published_tarball_identity": "3/3",
            "published_asset_identity": "18/18",
            "docker_asset_count": 0,
            "failure_count": 0,
            "stale_count": 0,
            "full_matrix_claim": False,
            "full_matrix_status": "not_evaluated",
        },
    )
    manifest["release"] = state["release"]
    manifest["assets"] = state["assets"]
    manifest["lanes"] = lanes
    write_json(out / "manifest.json", manifest)
    verify_goal_manifest(
        out / "manifest.json", expected_lane="runtime-vnext-published-assets"
    )
    return out


def canonical_gate_dependency(
    path: Path,
    *,
    label: str,
    lane: str | None,
    pass_prefix: str,
) -> dict[str, Any]:
    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        preferred = candidate / "gate.manifest.json"
        candidate = preferred if preferred.is_file() else candidate / "manifest.json"
    require(candidate.is_file(), f"{label} manifest is missing: {candidate}")
    document = read_json(candidate, f"{label} manifest")
    require(document.get("status") == "pass", f"{label} status is not pass")
    if lane is not None:
        require(document.get("lane") == lane, f"{label} lane differs")
    recorded_lines = [
        value
        for key in ("pass_line", "child_pass_line", "prepromotion_pass_line")
        if isinstance((value := document.get(key)), str)
    ]
    require(
        any(line.startswith(f"{pass_prefix}: ") for line in recorded_lines),
        f"{label} required PASS line is missing",
    )
    return {
        "status": "pass",
        "pass_line": next(
            line for line in recorded_lines if line.startswith(f"{pass_prefix}: ")
        ),
        "manifest": artifact_ref(candidate),
        "document": document,
    }


def strict_crates_io_dependency(
    path: Path, *, expected_source: dict[str, Any]
) -> dict[str, Any]:
    try:
        import runtime_vnext_crates_io_release as crates_io_release

        document = crates_io_release.validate_publish_manifest(path)
        manifest_path = crates_io_release.resolve_manifest_path(
            path,
            ("crates-io.manifest.json", "gate.manifest.json"),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise GoalGateError(
            f"crates.io v0.8.0 strict publish manifest validation failed: {error}"
        ) from error
    dependency = canonical_gate_dependency(
        manifest_path,
        label="crates.io v0.8.0",
        lane="runtime-vnext-crates-io",
        pass_prefix="FERRUM CRATES IO V0.8.0 PASS",
    )
    require(
        dependency["document"] == document,
        "crates.io v0.8.0 strict/canonical manifest selection differs",
    )
    candidate = require_object(
        document.get("release_candidate"),
        "crates.io v0.8.0 release candidate",
    )
    crates_source = {
        key: candidate.get(key)
        for key in ("git_sha", "git_tree_sha", "dirty")
    }
    require(
        crates_source == expected_source,
        "crates.io v0.8.0 release-candidate source differs from published assets",
    )
    return dependency


def prepromotion_fields() -> set[str]:
    return {
        "schema_version",
        "artifact_type",
        "status",
        "lane",
        "version",
        "canonical",
        "artifact_dir",
        "manifest_id",
        "release_candidate_sha",
        "pass_line",
        "prepromotion_pass_line",
        "release",
        "consumption",
        "dependencies",
        "created_at",
    }


def verify_prepromotion_manifest(path: Path) -> dict[str, Any]:
    manifest_path = input_manifest(path)
    root = manifest_path.parent.resolve()
    value = read_json(manifest_path, "runtime-vnext prepromotion manifest")
    require(set(value) == prepromotion_fields(), "prepromotion manifest fields differ")
    require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("artifact_type") == ARTIFACT_TYPES["runtime-vnext-prepromotion"]
        and value.get("status") == "pass"
        and value.get("lane") == "runtime-vnext-prepromotion"
        and value.get("version") == VERSION
        and value.get("canonical") is True
        and Path(str(value.get("artifact_dir", ""))).resolve() == root
        and value.get("pass_line")
        == pass_line("runtime-vnext-prepromotion", root)
        and value.get("prepromotion_pass_line")
        == pass_line("runtime-vnext-prepromotion", root),
        "prepromotion manifest identity/status differs",
    )
    manifest_id = require_sha256(value.get("manifest_id"), "prepromotion manifest id")
    release_candidate_sha = require_git_sha(
        value.get("release_candidate_sha"), "prepromotion release candidate SHA"
    )
    release = require_object(value.get("release"), "prepromotion release")
    require(
        set(release)
        == {
            "id",
            "tag_name",
            "tag_sha",
            "draft",
            "prerelease",
            "asset_set_sha256",
        }
        and require_string(release.get("id"), "prepromotion release id")
        and release.get("tag_name") == TAG
        and release.get("tag_sha") == release_candidate_sha
        and release.get("draft") is False
        and release.get("prerelease") is True,
        "prepromotion release identity/state differs",
    )
    require_sha256(release.get("asset_set_sha256"), "prepromotion asset set SHA256")
    consumption = require_object(value.get("consumption"), "prepromotion consumption")
    require(
        set(consumption)
        == {
            "state",
            "release_id",
            "token",
            "consumed_at",
            "consumed_by",
        }
        and consumption.get("state") == "unconsumed"
        and consumption.get("release_id") == release["id"]
        and re.fullmatch(
            r"[A-Za-z0-9._-]{32,}",
            str(consumption.get("token", "")),
        )
        is not None
        and consumption.get("consumed_at") is None
        and consumption.get("consumed_by") is None,
        "prepromotion consumption state/token differs",
    )
    dependencies = require_object(value.get("dependencies"), "prepromotion dependencies")
    require(
        set(dependencies)
        == {
            "published_assets",
            "crates_io",
            "homebrew_metal",
            "homebrew_cuda_fetch",
            "workflow_policy",
        },
        "prepromotion dependency denominator differs",
    )
    prefixes = {
        "published_assets": PASS_PREFIXES["runtime-vnext-published-assets"],
        "crates_io": "FERRUM CRATES IO V0.8.0 PASS",
        "homebrew_metal": "HOMEBREW METAL GATE PASS",
        "homebrew_cuda_fetch": "HOMEBREW CUDA FETCH GATE PASS",
        "workflow_policy": PASS_PREFIXES["workflow-policy"],
    }
    for key, prefix in prefixes.items():
        dependency = require_object(dependencies.get(key), f"prepromotion {key}")
        require(
            set(dependency) == {"status", "pass_line", "manifest"}
            and dependency.get("status") == "pass"
            and str(dependency.get("pass_line", "")).startswith(f"{prefix}: "),
            f"prepromotion {key} dependency status/PASS differs",
        )
        resolve_evidence_ref(
            dependency.get("manifest"),
            f"prepromotion {key} manifest",
            root=root,
        )
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "lane": "runtime-vnext-prepromotion",
        "release_candidate_sha": release_candidate_sha,
        "release": release,
        "consumption": consumption,
        "dependencies": dependencies,
    }
    require(
        canonical_json_sha256(identity_payload) == manifest_id,
        "prepromotion manifest_id does not bind its immutable payload",
    )
    published_path = resolve_evidence_ref(
        dependencies["published_assets"]["manifest"],
        "prepromotion published assets",
        root=root,
    )[1]
    published = verify_goal_manifest(
        published_path,
        expected_lane="runtime-vnext-published-assets",
    )
    require(
        published["source"]["git_sha"] == release_candidate_sha
        and published["manifest"]["release"]["id"] == release["id"]
        and published["manifest"]["release"]["asset_set_sha256"]
        == release["asset_set_sha256"],
        "prepromotion published release identity differs",
    )
    crates_path = resolve_evidence_ref(
        dependencies["crates_io"]["manifest"],
        "prepromotion crates.io publish manifest",
        root=root,
    )[1]
    strict_crates_io_dependency(
        crates_path,
        expected_source=published["source"],
    )
    return {
        "kind": "runtime-vnext-prepromotion",
        "path": manifest_path,
        "manifest": value,
        "child_manifest": artifact_ref(manifest_path),
        "ref": artifact_ref(manifest_path),
        "source": source_object(
            release_candidate_sha,
            published["source"]["git_tree_sha"],
            False,
        ),
    }


def build_prepromotion(args: argparse.Namespace) -> Path:
    out = ensure_fresh_out(args.out)
    published = verify_goal_manifest(
        args.published_assets,
        expected_lane="runtime-vnext-published-assets",
    )
    release = published["manifest"]["release"]
    workflow = validate_workflow_policy_manifest(args.workflow_policy)
    require(
        workflow["source"] == published["source"],
        "prepromotion workflow policy source differs from release",
    )
    crates = strict_crates_io_dependency(
        args.crates_io,
        expected_source=published["source"],
    )
    homebrew_metal = canonical_gate_dependency(
        args.homebrew_metal,
        label="Homebrew Metal",
        lane="homebrew-metal",
        pass_prefix="HOMEBREW METAL GATE PASS",
    )
    homebrew_cuda = canonical_gate_dependency(
        args.homebrew_cuda_fetch,
        label="Homebrew CUDA fetch",
        lane="homebrew-cuda-fetch",
        pass_prefix="HOMEBREW CUDA FETCH GATE PASS",
    )
    dependencies = {
        "published_assets": {
            "status": "pass",
            "pass_line": published["manifest"]["pass_line"],
            "manifest": published["ref"],
        },
        "crates_io": {
            key: crates[key] for key in ("status", "pass_line", "manifest")
        },
        "homebrew_metal": {
            key: homebrew_metal[key]
            for key in ("status", "pass_line", "manifest")
        },
        "homebrew_cuda_fetch": {
            key: homebrew_cuda[key]
            for key in ("status", "pass_line", "manifest")
        },
        "workflow_policy": {
            "status": "pass",
            "pass_line": workflow["manifest"]["pass_line"],
            "manifest": workflow["ref"],
        },
    }
    release_summary = {
        "id": require_string(release.get("id"), "published release id"),
        "tag_name": TAG,
        "tag_sha": published["source"]["git_sha"],
        "draft": False,
        "prerelease": True,
        "asset_set_sha256": require_sha256(
            release.get("asset_set_sha256"), "published asset set SHA256"
        ),
    }
    consumption = {
        "state": "unconsumed",
        "release_id": release_summary["id"],
        "token": secrets.token_urlsafe(32),
        "consumed_at": None,
        "consumed_by": None,
    }
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "lane": "runtime-vnext-prepromotion",
        "release_candidate_sha": published["source"]["git_sha"],
        "release": release_summary,
        "consumption": consumption,
        "dependencies": dependencies,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPES["runtime-vnext-prepromotion"],
        "status": "pass",
        "lane": "runtime-vnext-prepromotion",
        "version": VERSION,
        "canonical": True,
        "artifact_dir": str(out),
        "manifest_id": canonical_json_sha256(identity_payload),
        "release_candidate_sha": published["source"]["git_sha"],
        "pass_line": pass_line("runtime-vnext-prepromotion", out),
        "prepromotion_pass_line": pass_line("runtime-vnext-prepromotion", out),
        "release": release_summary,
        "consumption": consumption,
        "dependencies": dependencies,
        "created_at": iso_now(),
    }
    write_json(out / "manifest.json", manifest)
    verify_prepromotion_manifest(out / "manifest.json")
    return out


def validate_promotion_receipt(
    path: Path,
    *,
    prepromotion: dict[str, Any],
    final_release: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = input_manifest(path, "promotion-consumption.json")
    receipt = read_json(receipt_path, "release promotion receipt")
    expected_fields = {
        "schema_version",
        "state",
        "release_id",
        "tag",
        "release_candidate_sha",
        "prepromotion_manifest_sha256",
        "prepromotion_manifest_id",
        "consumption_token",
        "workflow_run_id",
        "workflow_run_attempt",
        "consumed_at",
        "consumed_by",
        "promotion",
        "asset_ids",
    }
    require(set(receipt) == expected_fields, "promotion receipt fields differ")
    pre = prepromotion["manifest"]
    require(
        receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("state") == "consumed"
        and str(receipt.get("release_id")) == pre["release"]["id"]
        and receipt.get("tag") == TAG
        and receipt.get("release_candidate_sha") == pre["release_candidate_sha"]
        and receipt.get("prepromotion_manifest_sha256")
        == prepromotion["ref"]["sha256"]
        and receipt.get("prepromotion_manifest_id") == pre["manifest_id"]
        and receipt.get("consumption_token") == pre["consumption"]["token"]
        and type(receipt.get("workflow_run_id")) is int
        and receipt["workflow_run_id"] > 0
        and type(receipt.get("workflow_run_attempt")) is int
        and receipt["workflow_run_attempt"] > 0
        and require_string(receipt.get("consumed_at"), "promotion consumed_at")
        and receipt.get("consumed_by") == "release-promote.yml",
        "promotion receipt consumption binding differs",
    )
    promotion = require_object(receipt.get("promotion"), "promotion receipt state")
    require(
        set(promotion) == {"state", "promoted_at"}
        and promotion.get("state") == "complete"
        and require_string(promotion.get("promoted_at"), "promotion promoted_at"),
        "promotion did not complete",
    )
    expected_ids = sorted(row["id"] for row in github_asset_rows(final_release))
    asset_ids = require_list(receipt.get("asset_ids"), "promotion asset ids")
    require(
        sorted(asset_ids) == expected_ids,
        "promotion receipt asset ids differ from the final release",
    )
    return {"path": receipt_path, "manifest": receipt, "ref": artifact_ref(receipt_path)}


def build_g10b(args: argparse.Namespace) -> Path:
    out = ensure_fresh_out(args.out)
    g10a = verify_goal_manifest(args.g10a, expected_lane="vnext-g10a")
    g08 = verify_goal_manifest(args.g08_rc, expected_lane="vnext-g08-rc")
    g09 = verify_goal_manifest(args.g09_rc, expected_lane="vnext-g09-rc")
    published = verify_goal_manifest(
        args.published_assets,
        expected_lane="runtime-vnext-published-assets",
    )
    prepromotion = verify_prepromotion_manifest(args.prepromotion)
    require(
        g10a["source"]
        == g08["source"]
        == g09["source"]
        == published["source"]
        == prepromotion["source"],
        "G10B release DAG source identities differ",
    )
    _, staged_path = resolve_evidence_ref(
        g10a["manifest"]["inputs"]["staged_assets"],
        "G10B staged assets",
        root=g10a["path"].parent,
    )
    staged = validate_staged_assets_manifest(staged_path)
    final_release = github_api_json(f"releases/tags/{TAG}")
    state = validate_published_state(
        release=final_release,
        rc_tag=github_annotated_tag(staged["release_candidate_tag"]),
        final_tag=github_annotated_tag(TAG),
        staged=staged,
        require_prerelease=False,
    )
    require(
        state["release"]["id"] == published["manifest"]["release"]["id"]
        == prepromotion["manifest"]["release"]["id"]
        and state["release"]["asset_set_sha256"]
        == published["manifest"]["release"]["asset_set_sha256"]
        == prepromotion["manifest"]["release"]["asset_set_sha256"],
        "G10B final release id/assets differ from prerelease/prepromotion",
    )
    promotion = validate_promotion_receipt(
        args.promotion_receipt,
        prepromotion=prepromotion,
        final_release=final_release,
    )
    manifest = base_manifest(
        "vnext-g10b",
        out,
        release_candidate=g10a["source"],
        inputs={
            "g10a": g10a["ref"],
            "g08_rc": g08["ref"],
            "g09_rc": g09["ref"],
            "published_assets": published["ref"],
            "prepromotion": prepromotion["ref"],
            "promotion_receipt": promotion["ref"],
        },
        acceptance={
            "release_state": "published",
            "release_id_unchanged": True,
            "asset_set_unchanged": True,
            "annotated_rc_tag": staged["release_candidate_tag"],
            "annotated_final_tag": TAG,
            "same_tag_target": True,
            "promotion_consumption": "complete",
            "failure_count": 0,
            "stale_count": 0,
        },
    )
    manifest["release"] = state["release"]
    manifest["promotion"] = {
        "state": "complete",
        "receipt": promotion["ref"],
        "prepromotion_manifest_id": prepromotion["manifest"]["manifest_id"],
        "workflow_run_id": promotion["manifest"]["workflow_run_id"],
    }
    write_json(out / "manifest.json", manifest)
    verify_goal_manifest(out / "manifest.json", expected_lane="vnext-g10b")
    return out


def build_g10(args: argparse.Namespace) -> Path:
    out = ensure_fresh_out(args.out)
    g10a = verify_goal_manifest(args.g10a, expected_lane="vnext-g10a")
    g08 = verify_goal_manifest(args.g08_rc, expected_lane="vnext-g08-rc")
    g09 = verify_goal_manifest(args.g09_rc, expected_lane="vnext-g09-rc")
    g10b = verify_goal_manifest(args.g10b, expected_lane="vnext-g10b")
    require(
        g10a["source"] == g08["source"] == g09["source"] == g10b["source"],
        "G10 child release candidates differ",
    )
    require(
        g10b["manifest"]["release"].get("prerelease") is False,
        "G10B release is still prerelease",
    )
    manifest = base_manifest(
        "vnext-g10",
        out,
        release_candidate=g10a["source"],
        inputs={
            "g10a": g10a["ref"],
            "g08_rc": g08["ref"],
            "g09_rc": g09["ref"],
            "g10b": g10b["ref"],
        },
        acceptance={
            "release_dag": "4/4",
            "sampled_final_regression": "pass",
            "full_matrix_claim": False,
            "final_release": True,
            "failure_count": 0,
            "stale_count": 0,
        },
    )
    manifest["release"] = copy.deepcopy(g10b["manifest"]["release"])
    write_json(out / "manifest.json", manifest)
    verify_goal_manifest(out / "manifest.json", expected_lane="vnext-g10")
    return out


def build_r3(args: argparse.Namespace) -> Path:
    out = ensure_fresh_out(args.out)
    g10 = verify_goal_manifest(args.g10, expected_lane="vnext-g10")
    release_summary = canonical_gate_dependency(
        args.release_summary,
        label="G0 release summary",
        lane="release-summary",
        pass_prefix="G0 RELEASE PASS",
    )
    completion = canonical_gate_dependency(
        args.completion,
        label="release completion",
        lane="release-complete",
        pass_prefix="FERRUM RELEASE COMPLETION PASS",
    )
    manifest = base_manifest(
        "vnext-r3",
        out,
        release_candidate=g10["source"],
        inputs={
            "g10": g10["ref"],
            "release_summary": release_summary["manifest"],
            "release_completion": completion["manifest"],
        },
        acceptance={
            "r0_r3": "4/4",
            "release_summary": "pass",
            "release_completion": "pass",
            "final_release": True,
            "sampled_final_regression": "pass",
            "full_matrix_claim": False,
            "failure_count": 0,
            "stale_count": 0,
        },
    )
    write_json(out / "manifest.json", manifest)
    verify_goal_manifest(out / "manifest.json", expected_lane="vnext-r3")
    return out


def expect_reject(label: str, callback: Callable[[], Any]) -> None:
    try:
        callback()
    except (GoalGateError, KeyError, OSError, TypeError, ValueError):
        return
    raise GoalGateError(f"negative fixture {label} unexpectedly passed")


def synthetic_release_assets(staged: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: dict[str, dict[str, Any]] = {}
    asset_id = 100
    for backend, name in ASSET_NAMES.items():
        tarball = staged["assets"][backend]["tarball"]
        identities[name] = {
            "size": tarball["size_bytes"],
            "digest": f"sha256:{tarball['sha256']}",
        }
        rows.append(
            {
                "id": asset_id,
                "name": name,
                **identities[name],
            }
        )
        asset_id += 1
        for suffix in RELEASE_SIDECAR_SUFFIXES:
            sidecar_name = f"{name}{suffix}"
            sidecar_bytes = f"{backend}:{suffix}:fixture\n".encode("utf-8")
            identities[sidecar_name] = {
                "size": len(sidecar_bytes),
                "digest": f"sha256:{bytes_sha256(sidecar_bytes)}",
            }
            rows.append(
                {
                    "id": asset_id,
                    "name": sidecar_name,
                    **identities[sidecar_name],
                }
            )
            asset_id += 1
    staged["_selftest_release_file_identities"] = identities
    return rows


def selftest_live_github_staging_binding() -> None:
    source = source_object("1" * 40, "2" * 40, False)
    rc_tag = "v0.8.0-rc.1"
    with tempfile.TemporaryDirectory(prefix="ferrum-live-staging-selftest-") as raw:
        root = Path(raw)
        archive = root / "artifact.zip"
        archive.write_bytes(b"canonical workflow artifact fixture\n")
        digest = file_sha256(archive)
        artifact_name = f"ferrum-linux-x86_64-v0.8.0-rc-{source['git_sha']}"
        write_json(
            root / "github-artifact.json",
            {
                "schema_version": SCHEMA_VERSION,
                "repository": GITHUB_REPOSITORY,
                "workflow_run_id": 1001,
                "artifact_id": 2001,
                "artifact_name": artifact_name,
                "artifact_digest": f"sha256:{digest}",
                "expired": False,
                "archive_path": archive.name,
                "workflow_inputs": {
                    "release_candidate_sha": source["git_sha"],
                    "release_candidate_tag": rc_tag,
                    "staging_label": "v0.8.0-rc",
                    "publish_release": False,
                },
            },
        )
        live = {
            "actions/runs/1001": {
                "id": 1001,
                "repository": {"full_name": GITHUB_REPOSITORY},
                "path": ".github/workflows/release.yml",
                "event": "workflow_dispatch",
                "head_sha": source["git_sha"],
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 2,
            },
            "actions/artifacts/2001": {
                "id": 2001,
                "name": artifact_name,
                "size_in_bytes": archive.stat().st_size,
                "digest": f"sha256:{digest}",
                "expired": False,
                "workflow_run": {"id": 1001, "head_sha": source["git_sha"]},
            },
        }

        def fetch(path: str, values: dict[str, Any] = live) -> dict[str, Any]:
            return copy.deepcopy(require_object(values.get(path), f"mock GitHub {path}"))

        receipt, resolved = validate_github_artifact_receipt(
            root,
            backend="cpu",
            release_candidate=source,
            release_candidate_tag=rc_tag,
            github_fetch=fetch,
        )
        require(
            resolved == archive.resolve()
            and receipt["_live_workflow_run"]["attempt"] == 2,
            "live staging fixture differs",
        )
        validate_staged_workflow_run_record(
            receipt["_live_workflow_run"],
            backend="cpu",
            workflow_run_id=1001,
            release_candidate_sha=source["git_sha"],
        )
        tampered_record = copy.deepcopy(receipt["_live_workflow_run"])
        tampered_record["head_sha"] = "f" * 40
        expect_reject(
            "staged-artifact-manifest-workflow-tamper",
            lambda: validate_staged_workflow_run_record(
                tampered_record,
                backend="cpu",
                workflow_run_id=1001,
                release_candidate_sha=source["git_sha"],
            ),
        )

        wrong_workflow = copy.deepcopy(live)
        wrong_workflow["actions/runs/1001"]["path"] = ".github/workflows/release-cuda.yml"
        expect_reject(
            "staged-live-wrong-workflow",
            lambda: validate_github_artifact_receipt(
                root,
                backend="cpu",
                release_candidate=source,
                release_candidate_tag=rc_tag,
                github_fetch=lambda path: copy.deepcopy(wrong_workflow[path]),
            ),
        )
        failed_run = copy.deepcopy(live)
        failed_run["actions/runs/1001"]["conclusion"] = "failure"
        expect_reject(
            "staged-live-failed-run",
            lambda: validate_github_artifact_receipt(
                root,
                backend="cpu",
                release_candidate=source,
                release_candidate_tag=rc_tag,
                github_fetch=lambda path: copy.deepcopy(failed_run[path]),
            ),
        )
        forged_artifact = copy.deepcopy(live)
        forged_artifact["actions/artifacts/2001"]["name"] = "forged-artifact"
        expect_reject(
            "staged-live-forged-artifact",
            lambda: validate_github_artifact_receipt(
                root,
                backend="cpu",
                release_candidate=source,
                release_candidate_tag=rc_tag,
                github_fetch=lambda path: copy.deepcopy(forged_artifact[path]),
            ),
        )

    valid_topology = {
        "cpu": {"workflow_run_id": 1001},
        "metal": {"workflow_run_id": 1001},
        "cuda": {"workflow_run_id": 1002},
    }
    validate_staged_workflow_run_topology(valid_topology)
    invalid_topology = copy.deepcopy(valid_topology)
    invalid_topology["cuda"]["workflow_run_id"] = 1001
    expect_reject(
        "staged-live-cuda-shared-run",
        lambda: validate_staged_workflow_run_topology(invalid_topology),
    )


def self_test() -> int:
    current_release_diff_fixture = {
        ".github/workflows/docker.yml",
        ".github/workflows/release-cuda.yml",
        ".github/workflows/release.yml",
        ".github/workflows/release-promote.yml",
        "CHANGELOG.md",
        "Cargo.lock",
        "Cargo.toml",
        "README.md",
        "README_zh.md",
        "docs/release/runtime-vnext/0.8.0/MIGRATION.md",
        "docs/release/runtime-vnext/0.8.0/PERFORMANCE_REPORT.md",
        "docs/release/runtime-vnext/0.8.0/RELEASE_NOTES.md",
        "docs/release/runtime-vnext/0.8.0/SUPPORT_MATRIX.md",
        "scripts/release/configs/runtime_vnext_r3_sample_plan.json",
        "scripts/release/g0_release_summary.py",
        "scripts/release/g0_source_gate.sh",
        "scripts/release/run_gate.py",
        "scripts/release/runtime_vnext_baseline_scenarios.py",
        "scripts/release/runtime_vnext_crates_io_release.py",
        "scripts/release/runtime_vnext_g08b_cuda_matrix_prepare.py",
        "scripts/release/runtime_vnext_goal_gate.py",
        "scripts/release/runtime_vnext_homebrew_release.py",
        "scripts/release/runtime_vnext_r2_ferrum_collector.py",
        "scripts/release/runtime_vnext_release_workflow_policy.py",
        "scripts/release/runtime_vnext_sampled_final.py",
        "scripts/release/selftest_g0_validators.py",
        "scripts/release/validate_release_completion_manifest.py",
    }
    require(
        all(
            g10a_release_only_path_allowed(path)
            for path in current_release_diff_fixture
        ),
        "current release-only diff fixture is not fully allowlisted",
    )
    require(
        not g10a_release_only_path_allowed("crates/ferrum-engine/src/lib.rs"),
        "G10A release-only closure accepted an arbitrary Rust product path",
    )
    workflow_names = (
        "release.yml",
        "release-cuda.yml",
        "docker.yml",
        "release-promote.yml",
    )
    workflow_policy_fixture = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "lane": "runtime-vnext-release-workflow-policy",
        "version": VERSION,
        "git_sha": "1" * 40,
        "git_tree": "2" * 40,
        "dirty": False,
        "created_at": "2026-08-14T00:00:00+00:00",
        "pass_line": "FERRUM RELEASE WORKFLOW POLICY PASS: fixture",
        "workflows": {
            name: {
                "path": f".github/workflows/{name}",
                "sha256": file_sha256(REPO_ROOT / ".github" / "workflows" / name),
            }
            for name in workflow_names
        },
        "negative_fixtures": {
            name: "rejected"
            for name in (
                "direct_official_release",
                "docker_tag_trigger",
                "docker_publish_job",
                "missing_prepromotion_child",
                "release_candidate_sha_mismatch",
                "release_candidate_tag_mismatch",
                "diagnostics_tag_mismatch",
                "diagnostics_archive_sha_unbound",
                "diagnostics_child_sha_unbound",
                "promotion_complete_marker_missing",
                "promotion_mutates_more_than_prerelease",
                "prepromotion_manifest_reuse",
                "prepromotion_release_sha_mismatch",
                "prepromotion_manifest_id_mismatch",
                "prepromotion_dependency_denominator_mismatch",
                "prepromotion_dependency_status_mismatch",
            )
        },
    }
    with tempfile.TemporaryDirectory(prefix="ferrum-workflow-policy-selftest-") as raw:
        fixture_root = Path(raw)
        workflow_policy_path = fixture_root / "gate.manifest.json"
        write_json(workflow_policy_path, workflow_policy_fixture)
        validate_workflow_policy_manifest(workflow_policy_path)
        promotion_source = (
            REPO_ROOT / ".github/workflows/release-promote.yml"
        ).read_text(encoding="utf-8")
        validate_promotion_workflow(promotion_source)
        expect_reject(
            "promotion-product-asset-upload",
            lambda: validate_promotion_workflow(
                promotion_source.replace(
                    "path: promotion-consumption.json",
                    "path: ferrum-linux-x86_64.tar.gz",
                    1,
                )
            ),
        )
        missing_promotion = copy.deepcopy(workflow_policy_fixture)
        del missing_promotion["workflows"]["release-promote.yml"]
        missing_promotion_path = fixture_root / "missing-promotion.json"
        write_json(missing_promotion_path, missing_promotion)
        expect_reject(
            "workflow-policy-missing-promotion-consumer",
            lambda: validate_workflow_policy_manifest(missing_promotion_path),
        )
    expect_reject(
        "dirty-release-candidate",
        lambda: normalize_source(
            {"git_sha": "1" * 40, "git_tree_sha": "2" * 40, "dirty": True},
            "fixture source",
        ),
    )
    expect_reject(
        "archive-path-traversal",
        lambda: safe_archive_path("../ferrum", "fixture archive path"),
    )
    expect_reject(
        "release-asset-missing-digest",
        lambda: github_asset_rows(
            {"assets": [{"id": 1, "name": "ferrum.tar.gz", "size": 1}]}
        ),
    )
    selftest_live_github_staging_binding()
    source = source_object("1" * 40, "2" * 40, False)
    staged = {
        "release_candidate": source,
        "release_candidate_tag": "v0.8.0-rc.1",
        "assets": {
            backend: {
                "tarball": {
                    "sha256": hashlib.sha256(f"{backend}-tar".encode()).hexdigest(),
                    "size_bytes": 128 + index,
                },
                "binary": {
                    "archive_path": "ferrum",
                    "sha256": hashlib.sha256(f"{backend}-bin".encode()).hexdigest(),
                    "size_bytes": 64 + index,
                },
                "workflow_run_id": 1000 + index,
                "artifact": {"id": 2000 + index},
            }
            for index, backend in enumerate(("cpu", "metal", "cuda"), start=1)
        },
    }
    rows = synthetic_release_assets(staged)
    release = {
        "id": 42,
        "tag_name": TAG,
        "draft": False,
        "prerelease": True,
        "published_at": "2026-08-14T00:00:00Z",
        "html_url": f"https://github.com/{GITHUB_REPOSITORY}/releases/tag/{TAG}",
        "assets": rows,
    }
    rc_tag = {
        "name": "v0.8.0-rc.1",
        "tag_object_sha": "3" * 40,
        "commit_sha": source["git_sha"],
    }
    final_tag = {
        "name": TAG,
        "tag_object_sha": "4" * 40,
        "commit_sha": source["git_sha"],
    }
    validated = validate_published_state(
        release=release,
        rc_tag=rc_tag,
        final_tag=final_tag,
        staged=staged,
        require_prerelease=True,
    )
    require(validated["release"]["asset_count"] == 18, "published fixture denominator differs")
    bad_release = copy.deepcopy(release)
    bad_release["assets"][0]["digest"] = f"sha256:{'f' * 64}"
    expect_reject(
        "published-tarball-byte-mismatch",
        lambda: validate_published_state(
            release=bad_release,
            rc_tag=rc_tag,
            final_tag=final_tag,
            staged=staged,
            require_prerelease=True,
        ),
    )
    altered_sidecar = copy.deepcopy(release)
    altered_sidecar["assets"][1]["digest"] = f"sha256:{'e' * 64}"
    expect_reject(
        "published-sidecar-byte-mismatch",
        lambda: validate_published_state(
            release=altered_sidecar,
            rc_tag=rc_tag,
            final_tag=final_tag,
            staged=staged,
            require_prerelease=True,
        ),
    )
    extra_asset = copy.deepcopy(release)
    extra_asset["assets"].append(
        {
            "id": 999,
            "name": "unexpected-release-evidence.json",
            "size": 1,
            "digest": f"sha256:{'d' * 64}",
        }
    )
    expect_reject(
        "published-extra-asset",
        lambda: validate_published_state(
            release=extra_asset,
            rc_tag=rc_tag,
            final_tag=final_tag,
            staged=staged,
            require_prerelease=True,
        ),
    )
    bad_final_tag = copy.deepcopy(final_tag)
    bad_final_tag["commit_sha"] = "5" * 40
    expect_reject(
        "final-tag-sha-mismatch",
        lambda: validate_published_state(
            release=release,
            rc_tag=rc_tag,
            final_tag=bad_final_tag,
            staged=staged,
            require_prerelease=True,
        ),
    )
    print("FERRUM RUNTIME VNEXT GOAL GATE SELFTEST PASS")
    return 0


def require_args(args: argparse.Namespace, names: Iterable[str]) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if getattr(args, name) is None]
    require(not missing, f"{args.mode} requires {', '.join(missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("staged-assets", *CANONICAL_LANES),
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--release-candidate-sha")
    parser.add_argument("--release-candidate-tree-sha")
    parser.add_argument("--release-candidate-tag")
    parser.add_argument("--cpu-dir", type=Path)
    parser.add_argument("--metal-dir", type=Path)
    parser.add_argument("--cuda-dir", type=Path)
    parser.add_argument("--r0", type=Path)
    parser.add_argument("--r1", type=Path)
    parser.add_argument("--r2", type=Path)
    parser.add_argument("--workflow-policy", type=Path)
    parser.add_argument("--staged-assets", type=Path)
    parser.add_argument("--g10a", type=Path)
    parser.add_argument("--g08-rc", type=Path)
    parser.add_argument("--g09-rc", type=Path)
    parser.add_argument("--m1-cuda", type=Path)
    parser.add_argument("--m1-metal", type=Path)
    parser.add_argument("--m2-cuda", type=Path)
    parser.add_argument("--m2-metal", type=Path)
    parser.add_argument("--m3-cuda", type=Path)
    parser.add_argument("--m3-metal", type=Path)
    parser.add_argument("--llama-cuda", type=Path)
    parser.add_argument("--llama-metal", type=Path)
    parser.add_argument("--published-assets", type=Path)
    parser.add_argument("--crates-io", type=Path)
    parser.add_argument("--homebrew-metal", type=Path)
    parser.add_argument("--homebrew-cuda-fetch", type=Path)
    parser.add_argument("--prepromotion", type=Path)
    parser.add_argument("--promotion-receipt", type=Path)
    parser.add_argument("--g10b", type=Path)
    parser.add_argument("--g10", type=Path)
    parser.add_argument("--release-summary", type=Path)
    parser.add_argument("--completion", type=Path)
    parser.add_argument("--skip-checkout-binding", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    require(args.mode is not None, "mode is required")
    require_args(args, ("out",))
    builders: dict[str, Callable[[argparse.Namespace], Path]] = {
        "staged-assets": build_staged_assets,
        "vnext-g10a": build_g10a,
        "vnext-g08-rc": build_g08_rc,
        "vnext-g09-rc": build_g09_rc,
        "runtime-vnext-metal-three-model": lambda value: build_three_model(
            value, backend="metal"
        ),
        "runtime-vnext-cuda-three-model": lambda value: build_three_model(
            value, backend="cuda"
        ),
        "runtime-vnext-published-assets": build_published_assets,
        "runtime-vnext-prepromotion": build_prepromotion,
        "vnext-g10b": build_g10b,
        "vnext-g10": build_g10,
        "vnext-r3": build_r3,
    }
    requirements = {
        "staged-assets": (
            "release_candidate_sha",
            "release_candidate_tree_sha",
            "release_candidate_tag",
            "cpu_dir",
            "metal_dir",
            "cuda_dir",
        ),
        "vnext-g10a": ("r0", "r1", "r2", "workflow_policy", "staged_assets"),
        "vnext-g08-rc": (
            "g10a",
            *LANE_KEYS,
            "llama_cuda",
            "llama_metal",
        ),
        "vnext-g09-rc": (
            "g10a",
            "g08_rc",
            *LANE_KEYS,
            "llama_cuda",
            "llama_metal",
        ),
        "runtime-vnext-metal-three-model": ("g10a", "g08_rc", "g09_rc"),
        "runtime-vnext-cuda-three-model": ("g10a", "g08_rc", "g09_rc"),
        "runtime-vnext-published-assets": ("g10a", "g08_rc", "g09_rc"),
        "runtime-vnext-prepromotion": (
            "published_assets",
            "crates_io",
            "homebrew_metal",
            "homebrew_cuda_fetch",
            "workflow_policy",
        ),
        "vnext-g10b": (
            "g10a",
            "g08_rc",
            "g09_rc",
            "published_assets",
            "prepromotion",
            "promotion_receipt",
        ),
        "vnext-g10": ("g10a", "g08_rc", "g09_rc", "g10b"),
        "vnext-r3": ("g10", "release_summary", "completion"),
    }
    require_args(args, requirements[args.mode])
    out = builders[args.mode](args)
    manifest = read_json(out / "manifest.json", f"{args.mode} output manifest")
    print(manifest["pass_line"])
    for line in manifest.get("additional_pass_lines", []):
        print(line)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GoalGateError as error:
        print(f"FERRUM RUNTIME VNEXT GOAL GATE FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
