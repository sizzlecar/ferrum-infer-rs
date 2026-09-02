#!/usr/bin/env python3
"""Validate Ferrum 0.8.4 staging workflows and the pinned native set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = Path(__file__).resolve().parent
if str(RELEASE_DIR) not in sys.path:
    sys.path.insert(0, str(RELEASE_DIR))

import native_operator_source_bundle as source_bundle  # noqa: E402
import runtime_vnext_native_operator_set as native_set  # noqa: E402
import runtime_vnext_release_workflow_policy as workflow_policy  # noqa: E402
from runtime_vnext_plan_reference import REQUIRED_CUDA_NATIVE_OPERATORS  # noqa: E402


VERSION = "0.8.4"
BASE_SHA = "84be21f06dcd8b625de00ca5d62ace1e3046db47"
GITHUB_REPOSITORY = "sizzlecar/ferrum-infer-rs"
WORKFLOW_PASS_PREFIX = "FERRUM 0.8.4 RELEASE WORKFLOW POLICY PASS"
NATIVE_PASS_PREFIX = "FERRUM 0.8.4 NATIVE OPERATOR SET PASS"
SELFTEST_PASS_LINE = "FERRUM 0.8.4 WORKFLOW NATIVE GATE SELFTEST PASS"
NATIVE_RELEASE_TAG = "ferrum-native-cuda12.4-sm89-v6"
NATIVE_ARCHIVE_NAME = "native-operator-set-cuda12.4-sm89-v6.tar.zst"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RC_TAG_RE = re.compile(r"^v0\.8\.4-rc\.[1-9][0-9]*$")
MAX_NATIVE_ARCHIVE_MEMBERS = 10_000
MAX_NATIVE_ARCHIVE_MEMBER_BYTES = 512 * 1024 * 1024
MAX_NATIVE_ARCHIVE_TOTAL_BYTES = 1024 * 1024 * 1024


BACKENDS: dict[str, dict[str, Any]] = {
    "cpu": {
        "asset": "ferrum-linux-x86_64.tar.gz",
        "audit": "ferrum-linux-x86_64.dependencies.txt",
        "backend": "cpu",
        "target": "x86_64-unknown-linux-gnu",
        "workflow": ".github/workflows/release.yml",
        "job": "Stage Linux x86_64 CPU",
    },
    "metal": {
        "asset": "ferrum-macos-aarch64.tar.gz",
        "audit": "ferrum-macos-aarch64.dependencies.txt",
        "backend": "metal",
        "target": "aarch64-apple-darwin",
        "workflow": ".github/workflows/release.yml",
        "job": "Stage macOS aarch64 Metal",
    },
    "cuda": {
        "asset": "ferrum-linux-x86_64-cuda-sm89.tar.gz",
        "audit": "ferrum-linux-x86_64-cuda-sm89.dependencies.txt",
        "backend": "cuda",
        "target": "x86_64-unknown-linux-gnu",
        "workflow": ".github/workflows/release-cuda.yml",
        "job": "Stage Linux x86_64 CUDA sm89",
    },
}

REQUIRED_JOB_STEPS: dict[str, tuple[str, ...]] = {
    "cpu": (
        "Validate immutable staging inputs",
        "Run actions/checkout@v4",
        "Verify exact clean release candidate",
        "Verify workspace release version",
        "Build release CPU binary exactly once",
        "Strip and audit binary",
        "Package staged CPU asset and adjacent manifests",
        "Smoke test staged CPU binary",
        "Upload staged CPU asset",
    ),
    "metal": (
        "Validate immutable staging inputs",
        "Run actions/checkout@v4",
        "Verify exact clean release candidate",
        "Verify workspace release version",
        "Build release Metal binary exactly once",
        "Strip and audit binary",
        "Package staged Metal asset and adjacent manifests",
        "Smoke test staged Metal binary",
        "Upload staged Metal asset",
    ),
    "cuda": (
        "Validate immutable staging inputs",
        "Run actions/checkout@v4",
        "Verify exact clean release candidate",
        "Verify native operator source boundary",
        "Materialize pinned CUDA native operator set",
        "Verify workspace release version",
        "Build release CUDA sm89 binary exactly once",
        "Strip and audit CUDA binary",
        "Package staged CUDA asset and adjacent manifests",
        "Deferred CUDA smoke contract",
        "Upload staged CUDA asset",
    ),
}


class GateError(RuntimeError):
    """A required release-evidence invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def portable_ref(path: Path, root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    require(
        resolved.is_file()
        and not path.is_symlink()
        and resolved.is_relative_to(resolved_root),
        f"portable evidence is missing or escapes its artifact: {path}",
    )
    return {
        "path": resolved.relative_to(resolved_root).as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def resolve_portable_ref(raw: Any, root: Path, label: str) -> Path:
    require(
        isinstance(raw, dict) and set(raw) == {"path", "sha256", "size_bytes"},
        f"{label} reference shape differs",
    )
    text = raw.get("path")
    require(isinstance(text, str) and text, f"{label} path is missing")
    pure = PurePosixPath(text)
    require(
        not pure.is_absolute() and "\\" not in text and ".." not in pure.parts,
        f"{label} path is not portable",
    )
    candidate = root.joinpath(*pure.parts)
    require(not candidate.is_symlink(), f"{label} is a symlink")
    resolved_root = root.resolve()
    path = candidate.resolve()
    require(
        path.is_relative_to(resolved_root) and path.is_file() and not path.is_symlink(),
        f"{label} is missing or escapes its artifact",
    )
    size = raw.get("size_bytes")
    digest = raw.get("sha256")
    require(
        isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
        and isinstance(digest, str)
        and SHA256_RE.fullmatch(digest) is not None
        and path.stat().st_size == size
        and sha256_file(path) == digest,
        f"{label} byte identity differs",
    )
    return path


def copy_evidence(source: Path, destination: Path, root: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    require(source.is_file() and not source.is_symlink(), f"evidence input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists() and not destination.is_symlink(), f"evidence output exists: {destination}")
    shutil.copyfile(source, destination)
    require(sha256_file(destination) == sha256_file(source), f"evidence copy differs: {source}")
    return portable_ref(destination, root)


def validate_readme_contract(
    paths: dict[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    specs = {
        "english": {
            "path": (paths or {}).get("english", REPO_ROOT / "README.md"),
            "metal_size": "2.55 GiB",
            "cuda_size": "8.7 GiB",
            "restore": "Omit the flag to preserve the model template's default reasoning",
        },
        "chinese": {
            "path": (paths or {}).get("chinese", REPO_ROOT / "README_zh.md"),
            "metal_size": "2.55 GiB",
            "cuda_size": "8.7 GiB",
            "restore": "删除该参数即可恢复模型模板默认的",
        },
    }
    result: dict[str, dict[str, Any]] = {}
    normalized: dict[str, dict[str, Any]] = {}
    for language, spec in specs.items():
        path = Path(spec["path"])
        text = path.read_text(encoding="utf-8")
        quick_heading = "## Quick Start" if language == "english" else "## 快速开始"
        next_heading = "## Features" if language == "english" else "## 功能"
        start = text.find(quick_heading)
        finish = text.find(next_heading, start + len(quick_heading))
        require(start >= 0 and finish > start, f"{language} README Quick Start block differs")
        quick = text[start:finish]
        global_commands = ("ferrum --version", "ferrum --help", "ferrum doctor")
        global_positions = [
            quick.find(f"\n{command}\n") for command in global_commands
        ]
        require(
            all(position >= 0 for position in global_positions)
            and global_positions == sorted(global_positions),
            f"{language} README global version/help/doctor commands differ",
        )
        block_headings = ("### macOS Apple Silicon", "### Linux NVIDIA CUDA")
        positions = [quick.find(heading) for heading in block_headings]
        require(positions[0] >= 0 and positions[1] > positions[0], f"{language} README backend block order differs")
        aliases = {"metal": "qwen3.5:4b-q4_k_m", "cuda": "qwen3.5:4b"}
        sizes = {"metal": str(spec["metal_size"]), "cuda": str(spec["cuda_size"])}
        structure: dict[str, Any] = {}
        for backend_index, backend in enumerate(("metal", "cuda")):
            alias = aliases[backend]
            doctor = f"ferrum doctor {alias}"
            run = f"ferrum run {alias} --disable-thinking"
            serve = (
                f"ferrum serve --model {alias} --served-model-name ferrum "
                "--disable-thinking --port 8000"
            )
            block_start = positions[backend_index]
            serve_end = quick.find("\n", quick.find(serve, block_start))
            require(serve_end > block_start, f"{language} README {backend} serve block differs")
            block = quick[block_start:serve_end]
            doctor_pos = block.find(doctor)
            run_pos = block.find(run)
            serve_pos = block.find(serve)
            size_pos = block.find(sizes[backend])
            require(
                0 <= size_pos < doctor_pos < run_pos < serve_pos,
                f"{language} README {backend} size/doctor/run/serve order differs",
            )
            first_run = re.search(r"(?m)^ferrum run[^\r\n]*$", block)
            require(
                first_run is not None and first_run.group(0) == run,
                f"{language} README {backend} promotes a different/smaller primary model",
            )
            structure[backend] = {
                "alias": alias,
                "size": sizes[backend],
                "doctor": doctor,
                "run": run,
                "serve": serve,
                "disable_thinking": True,
            }
        structure["global_commands"] = list(global_commands)
        hung_markers = ("progress output", "hung") if language == "english" else ("进度输出", "卡住")
        require(all(marker in quick for marker in hung_markers), f"{language} README does not distinguish download progress from a hung process")
        require(str(spec["restore"]) in text, f"{language} README does not explain restoring default reasoning")
        result[language] = file_ref(path)
        normalized[language] = structure
    require(normalized["english"] == normalized["chinese"], "English/Chinese README Quick Start structures differ")
    return result


def normalized_positive_int(value: Any, label: str) -> int:
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return value


def git_output(*arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        process.returncode == 0,
        f"git {' '.join(arguments)} failed: {process.stderr.strip()}",
    )
    return process.stdout.strip()


def validate_candidate_checkout(candidate_sha: str, candidate_tag: str) -> dict[str, str]:
    require(GIT_SHA_RE.fullmatch(candidate_sha) is not None, "candidate SHA is invalid")
    require(RC_TAG_RE.fullmatch(candidate_tag) is not None, "candidate RC tag is invalid")
    require(git_output("rev-parse", "HEAD") == candidate_sha, "HEAD is not the candidate SHA")
    require(not git_output("status", "--short", "--untracked-files=all"), "candidate checkout is dirty")
    require(
        git_output("merge-base", "--is-ancestor", BASE_SHA, candidate_sha) == "",
        "candidate does not descend from the required main baseline",
    )
    require(git_output("cat-file", "-t", candidate_tag) == "tag", "RC tag is not annotated")
    require(
        git_output("rev-parse", f"{candidate_tag}^{{commit}}") == candidate_sha,
        "RC tag does not peel to the candidate SHA",
    )
    return {
        "git_sha": candidate_sha,
        "git_tree_sha": git_output("rev-parse", f"{candidate_sha}^{{tree}}"),
        "tag": candidate_tag,
    }


def validate_static_staging_workflows() -> dict[str, dict[str, str]]:
    release_path = REPO_ROOT / ".github/workflows/release.yml"
    cuda_path = REPO_ROOT / ".github/workflows/release-cuda.yml"
    release_text = release_path.read_text(encoding="utf-8")
    cuda_text = cuda_path.read_text(encoding="utf-8")
    release_document = workflow_policy.parse_workflow_yaml(release_text, "release.yml")
    cuda_document = workflow_policy.parse_workflow_yaml(cuda_text, "release-cuda.yml")
    workflow_policy.validate_release_workflow(release_document)
    workflow_policy.validate_cuda_workflow(cuda_document)
    return {
        "release": {"path": str(release_path), "sha256": sha256_bytes(release_text.encode())},
        "cuda": {"path": str(cuda_path), "sha256": sha256_bytes(cuda_text.encode())},
    }


def validate_run_snapshot(
    document: dict[str, Any],
    *,
    label: str,
    expected_path: str,
    candidate_sha: str,
) -> dict[str, Any]:
    run_id = normalized_positive_int(document.get("id"), f"{label}.id")
    attempt = normalized_positive_int(document.get("run_attempt"), f"{label}.run_attempt")
    repository = document.get("repository")
    require(
        isinstance(repository, dict)
        and repository.get("full_name") == GITHUB_REPOSITORY,
        f"{label} repository differs",
    )
    require(
        document.get("path") == expected_path
        and document.get("event") == "workflow_dispatch"
        and document.get("head_sha") == candidate_sha
        and document.get("status") == "completed"
        and document.get("conclusion") == "success",
        f"{label} identity/status differs",
    )
    return {
        "id": run_id,
        "attempt": attempt,
        "path": expected_path,
        "head_sha": candidate_sha,
        "status": "completed",
        "conclusion": "success",
    }


def validate_jobs_snapshot(
    document: dict[str, Any],
    *,
    backends: tuple[str, ...],
    run: dict[str, Any],
    candidate_sha: str,
) -> dict[str, dict[str, Any]]:
    jobs = document.get("jobs")
    require(isinstance(jobs, list), "jobs snapshot lacks jobs")
    total = document.get("total_count")
    if total is not None:
        require(total == len(jobs), "jobs snapshot total_count differs")
    by_name: dict[str, dict[str, Any]] = {}
    for job in jobs:
        require(isinstance(job, dict), "jobs snapshot contains a non-object job")
        name = job.get("name")
        require(isinstance(name, str) and name not in by_name, "job name is missing or duplicated")
        by_name[name] = job
    expected_names = {str(BACKENDS[backend]["job"]) for backend in backends}
    require(set(by_name) == expected_names, f"workflow job set differs: {sorted(by_name)}")
    result: dict[str, dict[str, Any]] = {}
    for backend in backends:
        name = str(BACKENDS[backend]["job"])
        job = by_name[name]
        require(
            job.get("status") == "completed" and job.get("conclusion") == "success",
            f"{name} did not complete successfully",
        )
        require(
            job.get("head_sha") == candidate_sha
            and normalized_positive_int(job.get("run_attempt"), f"{name}.run_attempt")
            == run["attempt"],
            f"{name} candidate/run attempt differs",
        )
        run_url = job.get("run_url")
        require(
            isinstance(run_url, str)
            and run_url.endswith(f"/actions/runs/{run['id']}"),
            f"{name} run URL differs",
        )
        steps = job.get("steps")
        require(isinstance(steps, list), f"{name} steps are missing")
        step_by_name: dict[str, dict[str, Any]] = {}
        ordered_names: list[str] = []
        for step in steps:
            require(isinstance(step, dict), f"{name} has a non-object step")
            step_name = step.get("name")
            require(
                isinstance(step_name, str) and step_name not in step_by_name,
                f"{name} step name is missing or duplicated",
            )
            ordered_names.append(step_name)
            step_by_name[step_name] = step
        required_steps = REQUIRED_JOB_STEPS[backend]
        for step_name in required_steps:
            step = step_by_name.get(step_name)
            require(
                isinstance(step, dict)
                and step.get("status") == "completed"
                and step.get("conclusion") == "success",
                f"{name} required step did not pass: {step_name}",
            )
        positions = [ordered_names.index(step_name) for step_name in required_steps]
        require(positions == sorted(positions), f"{name} required step order differs")
        result[backend] = {
            "id": normalized_positive_int(job.get("id"), f"{name}.id"),
            "name": name,
            "required_steps": list(required_steps),
        }
    return result


def expected_bundle_names(backend: str) -> set[str]:
    asset = str(BACKENDS[backend]["asset"])
    return {
        asset,
        f"{asset}.sha256",
        f"{asset}.binary.sha256",
        f"{asset}.version.json",
        f"{asset}.dependency.json",
        f"{asset}.abi.json",
        str(BACKENDS[backend]["audit"]),
    }


def zip_payloads(path: Path, backend: str) -> dict[str, bytes]:
    require(path.is_file() and not path.is_symlink(), f"{backend} artifact ZIP is missing")
    try:
        with zipfile.ZipFile(path) as archive:
            payloads: dict[str, bytes] = {}
            for info in archive.infolist():
                require(not info.is_dir(), f"{backend} artifact ZIP contains a directory")
                pure = PurePosixPath(info.filename)
                require(
                    not pure.is_absolute()
                    and ".." not in pure.parts
                    and pure.name == info.filename
                    and pure.name not in payloads,
                    f"{backend} artifact ZIP member is unsafe or duplicated: {info.filename}",
                )
                payloads[pure.name] = archive.read(info)
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        raise GateError(f"cannot read {backend} artifact ZIP: {error}") from error
    require(set(payloads) == expected_bundle_names(backend), f"{backend} artifact ZIP is not exactly 7 files")
    return payloads


def parse_checksum(payload: bytes, expected_name: str, label: str) -> str:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise GateError(f"{label} is not UTF-8") from error
    parts = text.split()
    require(
        len(parts) == 2
        and SHA256_RE.fullmatch(parts[0]) is not None
        and Path(parts[1]).name == expected_name,
        f"{label} format differs",
    )
    return parts[0]


def parse_json_payload(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read {label}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def tarball_binary_sha256(payload: bytes, label: str) -> str:
    with tempfile.TemporaryDirectory(prefix="ferrum-v084-tar-") as raw:
        archive_path = Path(raw) / "asset.tar.gz"
        archive_path.write_bytes(payload)
        try:
            with tarfile.open(archive_path, mode="r:gz") as archive:
                binaries = []
                for member in archive.getmembers():
                    pure = PurePosixPath(member.name)
                    require(
                        not pure.is_absolute() and ".." not in pure.parts,
                        f"{label} contains an unsafe member",
                    )
                    require(not member.issym() and not member.islnk(), f"{label} contains a link")
                    if member.isfile() and pure.name == "ferrum":
                        binaries.append(member)
                require(len(binaries) == 1, f"{label} must contain exactly one ferrum binary")
                stream = archive.extractfile(binaries[0])
                require(stream is not None, f"{label} ferrum binary cannot be read")
                return sha256_bytes(stream.read())
        except (OSError, EOFError, tarfile.TarError) as error:
            raise GateError(f"cannot inspect {label}: {error}") from error


def common_sidecar_identity(
    document: dict[str, Any],
    *,
    backend: str,
    asset_sha: str,
    binary_sha: str,
    candidate_sha: str,
    candidate_tag: str,
    run: dict[str, Any],
) -> str:
    asset = str(BACKENDS[backend]["asset"])
    require(
        document.get("schema_version") == 1
        and document.get("asset_name") == asset
        and document.get("asset_sha256") == asset_sha
        and document.get("binary_name") == "ferrum"
        and document.get("binary_sha256") == binary_sha
        and document.get("release_candidate_sha") == candidate_sha
        and document.get("release_candidate_tag") == candidate_tag
        and normalized_positive_int(document.get("workflow_run_id"), "workflow_run_id")
        == run["id"]
        and normalized_positive_int(document.get("workflow_run_attempt"), "workflow_run_attempt")
        == run["attempt"],
        f"{backend} sidecar candidate/run/byte identity differs",
    )
    staging_label = document.get("staging_label")
    require(
        isinstance(staging_label, str)
        and re.fullmatch(r"[A-Za-z0-9._-]+", staging_label) is not None,
        f"{backend} staging label is invalid",
    )
    return staging_label


def validate_staged_bundle(
    *,
    backend: str,
    directory: Path,
    zip_path: Path,
    candidate_sha: str,
    candidate_tag: str,
    run: dict[str, Any],
) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(), f"{backend} staged directory is missing")
    local_paths = [path for path in directory.iterdir()]
    require(
        all(path.is_file() and not path.is_symlink() for path in local_paths)
        and {path.name for path in local_paths} == expected_bundle_names(backend),
        f"{backend} staged directory is not exactly 7 regular files",
    )
    payloads = zip_payloads(zip_path, backend)
    for path in local_paths:
        require(path.read_bytes() == payloads[path.name], f"{backend} staged file differs from artifact ZIP: {path.name}")

    asset = str(BACKENDS[backend]["asset"])
    asset_payload = payloads[asset]
    asset_sha = sha256_bytes(asset_payload)
    binary_sha = tarball_binary_sha256(asset_payload, f"{backend} tarball")
    require(
        parse_checksum(payloads[f"{asset}.sha256"], asset, f"{backend} asset checksum") == asset_sha,
        f"{backend} adjacent asset SHA256 differs",
    )
    require(
        parse_checksum(payloads[f"{asset}.binary.sha256"], "ferrum", f"{backend} binary checksum")
        == binary_sha,
        f"{backend} adjacent binary SHA256 differs",
    )
    version = parse_json_payload(payloads[f"{asset}.version.json"], f"{backend} version sidecar")
    dependency = parse_json_payload(
        payloads[f"{asset}.dependency.json"],
        f"{backend} dependency sidecar",
    )
    abi = parse_json_payload(payloads[f"{asset}.abi.json"], f"{backend} ABI sidecar")
    documents = (version, dependency, abi)
    require(all(isinstance(row, dict) for row in documents), f"{backend} sidecar root differs")
    labels = {
        common_sidecar_identity(
            row,
            backend=backend,
            asset_sha=asset_sha,
            binary_sha=binary_sha,
            candidate_sha=candidate_sha,
            candidate_tag=candidate_tag,
            run=run,
        )
        for row in documents
    }
    require(len(labels) == 1, f"{backend} sidecar staging labels differ")
    staging_label = next(iter(labels))
    audit_name = str(BACKENDS[backend]["audit"])
    audit_sha = sha256_bytes(payloads[audit_name])
    require(version.get("version") == VERSION, f"{backend} staged version differs")
    require(
        dependency.get("audit_file") == audit_name
        and dependency.get("audit_sha256") == audit_sha
        and dependency.get("forbidden_runtime_linkage_found") is False,
        f"{backend} dependency audit differs",
    )
    forbidden = dependency.get("forbidden_runtime_linkage")
    require(
        isinstance(forbidden, list)
        and all(name in forbidden for name in ("python", "torch", "vllm")),
        f"{backend} forbidden runtime linkage denominator differs",
    )
    require(
        abi.get("backend") == BACKENDS[backend]["backend"]
        and abi.get("target_triple") == BACKENDS[backend]["target"]
        and abi.get("dependency_audit_sha256") == audit_sha,
        f"{backend} ABI sidecar differs",
    )
    if backend == "cuda":
        require(
            str(abi.get("cuda_compute_capability")) == "89"
            and abi.get("cargo_features")
            == ["cuda", "vllm-moe-marlin", "vllm-paged-attn-v2"],
            "CUDA ABI/native feature identity differs",
        )
    return {
        "backend": backend,
        "asset_sha256": asset_sha,
        "binary_sha256": binary_sha,
        "staging_label": staging_label,
        "workflow_run_id": run["id"],
        "workflow_run_attempt": run["attempt"],
        "zip": file_ref(zip_path),
    }


def artifact_rows(document: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = document.get("artifacts")
    require(isinstance(rows, list), f"{label} lacks artifacts")
    total = document.get("total_count")
    if total is not None:
        require(total == len(rows), f"{label} total_count differs")
    require(all(isinstance(row, dict) for row in rows), f"{label} contains a non-object artifact")
    return rows


def validate_artifacts_snapshot(
    document: dict[str, Any],
    *,
    backends: tuple[str, ...],
    bundles: dict[str, dict[str, Any]],
    run: dict[str, Any],
    candidate_sha: str,
) -> dict[str, dict[str, Any]]:
    rows = artifact_rows(document, "workflow artifacts snapshot")
    require(len(rows) == len(backends), "workflow artifact count differs")
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("name")
        require(isinstance(name, str) and name not in by_name, "artifact name is missing or duplicated")
        by_name[name] = row
    result: dict[str, dict[str, Any]] = {}
    for backend in backends:
        bundle = bundles[backend]
        expected_name = (
            f"{str(BACKENDS[backend]['asset']).removesuffix('.tar.gz')}-"
            f"{bundle['staging_label']}-{candidate_sha}"
        )
        row = by_name.get(expected_name)
        require(isinstance(row, dict), f"{backend} workflow artifact is missing")
        digest = row.get("digest")
        require(
            row.get("expired") is False
            and digest == f"sha256:{bundle['zip']['sha256']}"
            and row.get("size_in_bytes") == bundle["zip"]["size_bytes"],
            f"{backend} workflow artifact byte/status identity differs",
        )
        workflow_run = row.get("workflow_run")
        require(
            isinstance(workflow_run, dict)
            and workflow_run.get("id") == run["id"]
            and workflow_run.get("head_sha") == candidate_sha,
            f"{backend} artifact workflow run differs",
        )
        result[backend] = {
            "id": normalized_positive_int(row.get("id"), f"{backend} artifact id"),
            "name": expected_name,
            "digest": digest,
        }
    require(set(by_name) == {row["name"] for row in result.values()}, "workflow has unexpected artifacts")
    return result


def ensure_fresh_out(path: Path) -> Path:
    out = path.expanduser().resolve()
    require(not out.exists() and not out.is_symlink(), f"output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def validate_workflow_policy(
    args: argparse.Namespace,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    candidate = (
        validate_candidate_checkout(args.candidate_sha, args.candidate_tag)
        if verify_checkout
        else {"git_sha": args.candidate_sha, "git_tree_sha": "2" * 40, "tag": args.candidate_tag}
    )
    workflows = validate_static_staging_workflows()
    readmes = validate_readme_contract()
    release_run_doc = read_json(args.release_run, "release workflow run snapshot")
    cuda_run_doc = read_json(args.cuda_run, "CUDA workflow run snapshot")
    release_run = validate_run_snapshot(
        release_run_doc,
        label="release workflow run",
        expected_path=str(BACKENDS["cpu"]["workflow"]),
        candidate_sha=args.candidate_sha,
    )
    cuda_run = validate_run_snapshot(
        cuda_run_doc,
        label="CUDA workflow run",
        expected_path=str(BACKENDS["cuda"]["workflow"]),
        candidate_sha=args.candidate_sha,
    )
    require(release_run["id"] != cuda_run["id"], "CUDA and CPU/Metal must use distinct workflow runs")
    jobs = {
        "release": validate_jobs_snapshot(
            read_json(args.release_jobs, "release jobs snapshot"),
            backends=("cpu", "metal"),
            run=release_run,
            candidate_sha=args.candidate_sha,
        ),
        "cuda": validate_jobs_snapshot(
            read_json(args.cuda_jobs, "CUDA jobs snapshot"),
            backends=("cuda",),
            run=cuda_run,
            candidate_sha=args.candidate_sha,
        ),
    }
    staged_root = args.staged_root.expanduser().resolve()
    bundle_args = {
        "cpu": args.cpu_zip,
        "metal": args.metal_zip,
        "cuda": args.cuda_zip,
    }
    bundles = {
        backend: validate_staged_bundle(
            backend=backend,
            directory=staged_root / backend,
            zip_path=zip_path.expanduser().resolve(),
            candidate_sha=args.candidate_sha,
            candidate_tag=args.candidate_tag,
            run=release_run if backend in {"cpu", "metal"} else cuda_run,
        )
        for backend, zip_path in bundle_args.items()
    }
    labels = {row["staging_label"] for row in bundles.values()}
    require(len(labels) == 1, "CPU/Metal/CUDA staging labels differ")
    artifacts = {
        "release": validate_artifacts_snapshot(
            read_json(args.release_artifacts, "release artifacts snapshot"),
            backends=("cpu", "metal"),
            bundles=bundles,
            run=release_run,
            candidate_sha=args.candidate_sha,
        ),
        "cuda": validate_artifacts_snapshot(
            read_json(args.cuda_artifacts, "CUDA artifacts snapshot"),
            backends=("cuda",),
            bundles=bundles,
            run=cuda_run,
            candidate_sha=args.candidate_sha,
        ),
    }
    return {
        "candidate": candidate,
        "workflows": workflows,
        "readmes": readmes,
        "runs": {"release": release_run, "cuda": cuda_run},
        "jobs": jobs,
        "artifacts": artifacts,
        "bundles": bundles,
        "staging_label": next(iter(labels)),
        "snapshots": {
            "release_run": file_ref(args.release_run),
            "release_jobs": file_ref(args.release_jobs),
            "release_artifacts": file_ref(args.release_artifacts),
            "cuda_run": file_ref(args.cuda_run),
            "cuda_jobs": file_ref(args.cuda_jobs),
            "cuda_artifacts": file_ref(args.cuda_artifacts),
        },
    }


def materialize_workflow_evidence(
    out: Path,
    args: argparse.Namespace,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    portable = json.loads(json.dumps(evidence))
    inputs = out / "inputs"
    portable["workflows"] = {
        "release": copy_evidence(
            REPO_ROOT / ".github/workflows/release.yml",
            inputs / "workflows/release.yml",
            out,
        ),
        "cuda": copy_evidence(
            REPO_ROOT / ".github/workflows/release-cuda.yml",
            inputs / "workflows/release-cuda.yml",
            out,
        ),
    }
    portable["readmes"] = {
        "english": copy_evidence(REPO_ROOT / "README.md", inputs / "readmes/README.md", out),
        "chinese": copy_evidence(
            REPO_ROOT / "README_zh.md", inputs / "readmes/README_zh.md", out
        ),
    }
    snapshot_inputs = {
        "release_run": args.release_run,
        "release_jobs": args.release_jobs,
        "release_artifacts": args.release_artifacts,
        "cuda_run": args.cuda_run,
        "cuda_jobs": args.cuda_jobs,
        "cuda_artifacts": args.cuda_artifacts,
    }
    portable["snapshots"] = {
        name: copy_evidence(path, inputs / f"snapshots/{name}.json", out)
        for name, path in snapshot_inputs.items()
    }
    zip_inputs = {"cpu": args.cpu_zip, "metal": args.metal_zip, "cuda": args.cuda_zip}
    for backend, source in zip_inputs.items():
        portable["bundles"][backend]["zip"] = copy_evidence(
            source, inputs / f"actions-artifacts/{backend}.zip", out
        )
    return portable


def _validate_gate_header(
    path: Path,
    *,
    lane: str,
    artifact_type: str,
    pass_prefix: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest = read_json(path, f"{lane} gate manifest")
    require(
        set(manifest)
        == {
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
        f"{lane} gate manifest fields differ",
    )
    require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact_type") == artifact_type
        and manifest.get("status") == "pass"
        and manifest.get("version") == VERSION
        and manifest.get("lane") == lane,
        f"{lane} gate manifest identity differs",
    )
    try:
        started = datetime.fromisoformat(str(manifest.get("started_at")).replace("Z", "+00:00"))
        finished = datetime.fromisoformat(str(manifest.get("finished_at")).replace("Z", "+00:00"))
    except ValueError as error:
        raise GateError(f"{lane} gate timestamps differ") from error
    require(
        started.tzinfo is not None and finished.tzinfo is not None and started <= finished,
        f"{lane} gate timestamps differ",
    )
    artifact_dir = manifest.get("artifact_dir")
    require(
        isinstance(artifact_dir, str)
        and artifact_dir
        and manifest.get("pass_line") == f"{pass_prefix}: {artifact_dir}",
        f"{lane} gate PASS binding differs",
    )
    root = path.parent.resolve()
    evidence = manifest.get("evidence")
    require(isinstance(evidence, dict), f"{lane} gate evidence is missing")
    return manifest, root, evidence


def validate_workflow_policy_manifest(path: Path) -> dict[str, Any]:
    _, root, evidence = _validate_gate_header(
        path.resolve(),
        lane="release-workflow-policy",
        artifact_type="ferrum_v084_release_workflow_policy_manifest",
        pass_prefix=WORKFLOW_PASS_PREFIX,
    )
    candidate = evidence.get("candidate")
    require(
        isinstance(candidate, dict)
        and GIT_SHA_RE.fullmatch(str(candidate.get("git_sha", ""))) is not None
        and GIT_SHA_RE.fullmatch(str(candidate.get("git_tree_sha", ""))) is not None
        and RC_TAG_RE.fullmatch(str(candidate.get("tag", ""))) is not None,
        "workflow-policy candidate identity differs",
    )
    workflows = evidence.get("workflows")
    require(isinstance(workflows, dict) and set(workflows) == {"release", "cuda"}, "workflow refs differ")
    release_workflow = resolve_portable_ref(workflows["release"], root, "release workflow")
    cuda_workflow = resolve_portable_ref(workflows["cuda"], root, "CUDA workflow")
    workflow_policy.validate_release_workflow(
        workflow_policy.parse_workflow_yaml(release_workflow.read_text(encoding="utf-8"), "release.yml")
    )
    workflow_policy.validate_cuda_workflow(
        workflow_policy.parse_workflow_yaml(cuda_workflow.read_text(encoding="utf-8"), "release-cuda.yml")
    )
    readmes = evidence.get("readmes")
    require(isinstance(readmes, dict) and set(readmes) == {"english", "chinese"}, "README refs differ")
    validate_readme_contract(
        {
            language: resolve_portable_ref(readmes[language], root, f"{language} README")
            for language in ("english", "chinese")
        }
    )
    snapshots = evidence.get("snapshots")
    expected_snapshot_names = {
        "release_run",
        "release_jobs",
        "release_artifacts",
        "cuda_run",
        "cuda_jobs",
        "cuda_artifacts",
    }
    require(
        isinstance(snapshots, dict) and set(snapshots) == expected_snapshot_names,
        "workflow snapshot denominator differs",
    )
    snapshot_paths = {
        name: resolve_portable_ref(snapshots[name], root, f"workflow snapshot {name}")
        for name in expected_snapshot_names
    }
    release_run = validate_run_snapshot(
        read_json(snapshot_paths["release_run"], "release run snapshot"),
        label="release workflow run",
        expected_path=str(BACKENDS["cpu"]["workflow"]),
        candidate_sha=candidate["git_sha"],
    )
    cuda_run = validate_run_snapshot(
        read_json(snapshot_paths["cuda_run"], "CUDA run snapshot"),
        label="CUDA workflow run",
        expected_path=str(BACKENDS["cuda"]["workflow"]),
        candidate_sha=candidate["git_sha"],
    )
    require(release_run["id"] != cuda_run["id"], "workflow run topology differs")
    expected_jobs = {
        "release": validate_jobs_snapshot(
            read_json(snapshot_paths["release_jobs"], "release jobs snapshot"),
            backends=("cpu", "metal"),
            run=release_run,
            candidate_sha=candidate["git_sha"],
        ),
        "cuda": validate_jobs_snapshot(
            read_json(snapshot_paths["cuda_jobs"], "CUDA jobs snapshot"),
            backends=("cuda",),
            run=cuda_run,
            candidate_sha=candidate["git_sha"],
        ),
    }
    require(evidence.get("jobs") == expected_jobs, "recorded workflow jobs differ")
    bundles = evidence.get("bundles")
    require(isinstance(bundles, dict) and set(bundles) == set(BACKENDS), "workflow bundle denominator differs")
    rebuilt: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="ferrum-v084-workflow-revalidate-") as raw:
        temporary = Path(raw)
        for backend in BACKENDS:
            zip_path = resolve_portable_ref(bundles[backend].get("zip"), root, f"{backend} Actions ZIP")
            directory = temporary / backend
            directory.mkdir()
            for name, payload in zip_payloads(zip_path, backend).items():
                (directory / name).write_bytes(payload)
            rebuilt[backend] = validate_staged_bundle(
                backend=backend,
                directory=directory,
                zip_path=zip_path,
                candidate_sha=candidate["git_sha"],
                candidate_tag=candidate["tag"],
                run=release_run if backend in {"cpu", "metal"} else cuda_run,
            )
            recorded = bundles[backend]
            require(isinstance(recorded, dict), f"{backend} recorded bundle differs")
            comparable = dict(rebuilt[backend])
            comparable["zip"] = recorded.get("zip")
            require(comparable == recorded, f"{backend} recorded bundle differs")
    labels = {row["staging_label"] for row in rebuilt.values()}
    require(labels == {evidence.get("staging_label")}, "workflow staging label differs")
    expected_artifacts = {
        "release": validate_artifacts_snapshot(
            read_json(snapshot_paths["release_artifacts"], "release artifacts snapshot"),
            backends=("cpu", "metal"),
            bundles=rebuilt,
            run=release_run,
            candidate_sha=candidate["git_sha"],
        ),
        "cuda": validate_artifacts_snapshot(
            read_json(snapshot_paths["cuda_artifacts"], "CUDA artifacts snapshot"),
            backends=("cuda",),
            bundles=rebuilt,
            run=cuda_run,
            candidate_sha=candidate["git_sha"],
        ),
    }
    require(evidence.get("artifacts") == expected_artifacts, "recorded workflow artifacts differ")
    require(evidence.get("runs") == {"release": release_run, "cuda": cuda_run}, "recorded workflow runs differ")
    return evidence


def materialize_native_evidence(
    out: Path,
    args: argparse.Namespace,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    portable = json.loads(json.dumps(evidence))
    inputs = out / "inputs"
    portable["source_bundle"]["manifest"] = copy_evidence(
        args.source_bundle_manifest, inputs / "source-bundle/manifest.json", out
    )
    portable["source_bundle"]["archive"] = copy_evidence(
        args.source_bundle_archive,
        inputs / f"source-bundle/{Path(args.source_bundle_archive).name}",
        out,
    )
    portable["native_set"]["archive"] = copy_evidence(
        args.native_set_archive,
        inputs / f"native-set/{Path(args.native_set_archive).name}",
        out,
    )
    portable["cuda_run_snapshot"] = copy_evidence(
        args.cuda_run, inputs / "snapshots/cuda-run.json", out
    )
    portable["cuda_jobs_snapshot"] = copy_evidence(
        args.cuda_jobs, inputs / "snapshots/cuda-jobs.json", out
    )
    portable["cuda_abi"] = copy_evidence(
        args.cuda_abi_manifest, inputs / "staged/cuda.abi.json", out
    )
    portable["workflow"] = copy_evidence(
        getattr(args, "workflow_path", REPO_ROOT / ".github/workflows/release-cuda.yml"),
        inputs / "workflows/release-cuda.yml",
        out,
    )
    portable["workflow_policy_snapshot"] = copy_evidence(
        REPO_ROOT / ".github/workflows/release-cuda.yml",
        inputs / "workflows/policy-release-cuda.yml",
        out,
    )
    return portable


def validate_native_set_manifest(path: Path) -> dict[str, Any]:
    _, root, evidence = _validate_gate_header(
        path.resolve(),
        lane="native-operator-set",
        artifact_type="ferrum_v084_native_operator_set_manifest",
        pass_prefix=NATIVE_PASS_PREFIX,
    )
    candidate = evidence.get("candidate")
    require(
        isinstance(candidate, dict)
        and GIT_SHA_RE.fullmatch(str(candidate.get("git_sha", ""))) is not None
        and GIT_SHA_RE.fullmatch(str(candidate.get("git_tree_sha", ""))) is not None
        and RC_TAG_RE.fullmatch(str(candidate.get("tag", ""))) is not None,
        "native-set candidate identity differs",
    )
    source = evidence.get("source_bundle")
    native = evidence.get("native_set")
    require(isinstance(source, dict) and isinstance(native, dict), "native/source evidence is missing")
    source_manifest = resolve_portable_ref(source.get("manifest"), root, "native source manifest")
    source_archive = resolve_portable_ref(source.get("archive"), root, "native source archive")
    native_archive = resolve_portable_ref(native.get("archive"), root, "native-set archive")
    cuda_run = resolve_portable_ref(evidence.get("cuda_run_snapshot"), root, "native CUDA run snapshot")
    cuda_jobs = resolve_portable_ref(evidence.get("cuda_jobs_snapshot"), root, "native CUDA jobs snapshot")
    cuda_abi = resolve_portable_ref(evidence.get("cuda_abi"), root, "native CUDA ABI")
    workflow_path = resolve_portable_ref(evidence.get("workflow"), root, "native CUDA workflow")
    policy_workflow_path = resolve_portable_ref(
        evidence.get("workflow_policy_snapshot"), root, "native CUDA workflow policy snapshot"
    )
    workflow_document = workflow_policy.parse_workflow_yaml(
        workflow_path.read_text(encoding="utf-8"), "release-cuda.yml"
    )
    policy_document = workflow_policy.parse_workflow_yaml(
        policy_workflow_path.read_text(encoding="utf-8"), "release-cuda.yml"
    )
    environment = workflow_document.get("env")
    expected_url = (
        f"https://github.com/{GITHUB_REPOSITORY}/releases/download/"
        f"{NATIVE_RELEASE_TAG}/{NATIVE_ARCHIVE_NAME}"
    )
    require(
        isinstance(environment, dict)
        and environment.get("NATIVE_OPERATOR_SET_ARCHIVE_URL") == expected_url
        and SHA256_RE.fullmatch(
            str(environment.get("NATIVE_OPERATOR_SET_ARCHIVE_SHA256", ""))
        )
        is not None,
        "captured CUDA workflow native v6 pin differs",
    )
    # The policy validator freezes the current public archive digest.  Recheck
    # every other workflow contract against that canonical policy while the
    # copied workflow's digest remains bound below to its copied native bytes.
    workflow_policy.validate_cuda_workflow(policy_document)
    normalized_document = json.loads(json.dumps(workflow_document))
    normalized_environment = normalized_document.get("env")
    canonical_environment = policy_document.get("env")
    require(
        isinstance(normalized_environment, dict)
        and isinstance(canonical_environment, dict),
        "captured CUDA workflow env differs",
    )
    for key in (
        "NATIVE_OPERATOR_SET_ARCHIVE_URL",
        "NATIVE_OPERATOR_SET_ARCHIVE_SHA256",
    ):
        normalized_environment[key] = canonical_environment[key]
    workflow_policy.validate_cuda_workflow(normalized_document)
    with tempfile.TemporaryDirectory(prefix="ferrum-v084-native-revalidate-") as raw:
        lock = safe_extract_native_archive(native_archive, Path(raw))
        args = argparse.Namespace(
            candidate_sha=candidate["git_sha"],
            candidate_tag=candidate["tag"],
            source_bundle_manifest=source_manifest,
            source_bundle_archive=source_archive,
            native_set_archive=native_archive,
            native_set_lock=lock,
            cuda_run=cuda_run,
            cuda_jobs=cuda_jobs,
            cuda_abi_manifest=cuda_abi,
        )
        rebuilt = validate_native_set(
            args,
            verify_checkout=False,
            validate_static_workflow=False,
            workflow_document=workflow_document,
        )
    require(source.get("bundle_id") == rebuilt["source_bundle"]["bundle_id"], "native source bundle id differs")
    require(native.get("workflow_url") == rebuilt["native_set"]["workflow_url"], "native workflow URL differs")
    require(native.get("workflow_sha256") == rebuilt["native_set"]["workflow_sha256"], "native workflow SHA differs")
    require(native.get("identity") == rebuilt["native_set"]["identity"], "native-set identity differs")
    require(native.get("source_revisions") == rebuilt["native_set"]["source_revisions"], "native source revisions differ")
    require(evidence.get("cuda_run") == rebuilt["cuda_run"], "native CUDA run differs")
    require(evidence.get("cuda_job") == rebuilt["cuda_job"], "native CUDA job differs")
    return evidence


def write_gate_manifest(
    out: Path,
    *,
    lane: str,
    pass_line: str,
    evidence: dict[str, Any],
    started_at: str,
    finished_at: str,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "gate.manifest.json").exists(), f"gate manifest already exists: {out}")
    manifest = {
        "schema_version": 1,
        "artifact_type": f"ferrum_v084_{lane.replace('-', '_')}_manifest",
        "status": "pass",
        "version": VERSION,
        "lane": lane,
        "started_at": started_at,
        "finished_at": finished_at,
        "artifact_dir": str(out),
        "pass_line": pass_line,
        "evidence": evidence,
    }
    write_json(out / "gate.manifest.json", manifest)
    (out / "pass_line.txt").write_text(pass_line + "\n", encoding="utf-8")


def safe_extract_native_archive(archive: Path, destination: Path) -> Path:
    require(archive.is_file() and not archive.is_symlink(), "native-set archive is missing")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.Popen(
            ["zstd", "--decompress", "--stdout", str(archive)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise GateError("zstd is required to validate the native-set archive") from error
    assert process.stdout is not None and process.stderr is not None
    total = 0
    count = 0
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
            for member in tar:
                count += 1
                require(count <= MAX_NATIVE_ARCHIVE_MEMBERS, "native-set archive has too many members")
                pure = PurePosixPath(member.name.rstrip("/"))
                require(
                    pure.parts
                    and not pure.is_absolute()
                    and ".." not in pure.parts
                    and pure.as_posix() not in seen,
                    f"native-set archive member is unsafe or duplicated: {member.name}",
                )
                seen.add(pure.as_posix())
                require(
                    member.isdir() or member.isfile(),
                    f"native-set archive contains a non-regular member: {member.name}",
                )
                target = destination.joinpath(*pure.parts)
                require(target.resolve().is_relative_to(destination.resolve()), "native-set member escapes output")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                require(
                    0 <= member.size <= MAX_NATIVE_ARCHIVE_MEMBER_BYTES,
                    f"native-set member size is invalid: {member.name}",
                )
                total += member.size
                require(total <= MAX_NATIVE_ARCHIVE_TOTAL_BYTES, "native-set archive is too large")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                require(source is not None, f"cannot read native-set member: {member.name}")
                written = 0
                with target.open("xb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
                        written += len(chunk)
                require(written == member.size, f"native-set member size changed: {member.name}")
    except Exception as error:
        if process.poll() is None:
            process.kill()
        process.wait()
        if isinstance(error, GateError):
            raise
        if isinstance(error, (OSError, EOFError, tarfile.TarError)):
            raise GateError(f"cannot extract native-set archive: {error}") from error
        raise
    process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    returncode = process.wait()
    require(returncode == 0, f"zstd failed while reading native-set archive: {stderr[-2000:]}")
    lock = destination / "inputs/native-operator-set.lock.json"
    require(lock.is_file() and not lock.is_symlink(), "native-set archive lacks its canonical lock")
    return lock


def native_source_revisions(lock_path: Path) -> set[str]:
    lock = read_json(lock_path, "native operator set lock")
    artifacts = lock.get("artifacts")
    require(isinstance(artifacts, list) and artifacts, "native operator set lock has no artifacts")
    revisions: set[str] = set()
    for index, row in enumerate(artifacts):
        require(isinstance(row, dict), f"native artifact {index} is invalid")
        relative = row.get("manifest_path")
        require(isinstance(relative, str), f"native artifact {index} manifest path is invalid")
        manifest = read_json(lock_path.parent / relative, f"native artifact {index} manifest")
        package = manifest.get("source_package")
        require(isinstance(package, dict), f"native artifact {index} source package is missing")
        revision = package.get("revision")
        require(isinstance(revision, str) and revision, f"native artifact {index} revision is missing")
        revisions.add(revision)
    return revisions


def validate_native_set(
    args: argparse.Namespace,
    *,
    verify_checkout: bool = True,
    validate_static_workflow: bool = True,
    workflow_document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = (
        validate_candidate_checkout(args.candidate_sha, args.candidate_tag)
        if verify_checkout
        else {"git_sha": args.candidate_sha, "git_tree_sha": "2" * 40, "tag": args.candidate_tag}
    )
    source_manifest_path = args.source_bundle_manifest.expanduser().resolve()
    if verify_checkout:
        require(
            source_manifest_path
            == (REPO_ROOT / "native-operators/cuda/source-bundles/ferrum-native-cuda-v1.json").resolve(),
            "source-bundle manifest is not the checked-in canonical manifest",
        )
    source_manifest = source_bundle.validate_manifest(
        source_bundle.read_json(source_manifest_path, "native source-bundle manifest")
    )
    source_archive = args.source_bundle_archive.expanduser().resolve()
    source_bundle.verify_archive(source_manifest, source_archive)

    cuda_workflow_path = REPO_ROOT / ".github/workflows/release-cuda.yml"
    if workflow_document is None:
        cuda_text = cuda_workflow_path.read_text(encoding="utf-8")
        workflow_document = workflow_policy.parse_workflow_yaml(cuda_text, "release-cuda.yml")
    if validate_static_workflow:
        workflow_policy.validate_cuda_workflow(workflow_document)
    environment = workflow_document.get("env")
    require(isinstance(environment, dict), "CUDA workflow env is missing")
    pinned_sha = environment.get("NATIVE_OPERATOR_SET_ARCHIVE_SHA256")
    pinned_url = environment.get("NATIVE_OPERATOR_SET_ARCHIVE_URL")
    require(
        isinstance(pinned_sha, str) and SHA256_RE.fullmatch(pinned_sha) is not None,
        "CUDA workflow native-set SHA pin is invalid",
    )
    require(
        isinstance(pinned_url, str)
        and pinned_url.startswith(
            f"https://github.com/{GITHUB_REPOSITORY}/releases/download/"
        ),
        "CUDA workflow native-set URL pin is invalid",
    )
    native_archive = args.native_set_archive.expanduser().resolve()
    require(sha256_file(native_archive) == pinned_sha, "public native-set archive differs from workflow pin")
    require(Path(pinned_url).name == native_archive.name, "native-set archive filename differs from workflow pin")

    provided_lock = args.native_set_lock.expanduser().resolve()
    provided = native_set.validate_native_operator_set(
        provided_lock,
        REQUIRED_CUDA_NATIVE_OPERATORS,
    )
    with tempfile.TemporaryDirectory(prefix="ferrum-v084-native-set-") as raw:
        archived_lock = safe_extract_native_archive(native_archive, Path(raw))
        archived = native_set.validate_native_operator_set(
            archived_lock,
            REQUIRED_CUDA_NATIVE_OPERATORS,
        )
        require(
            native_set.public_identity(provided) == native_set.public_identity(archived),
            "provided native-set closure differs from the pinned public archive",
        )
        revisions = native_source_revisions(archived_lock)
    require(
        revisions == {source_manifest["bundle_id"]},
        "native-set manifests do not all bind the checked-in source bundle",
    )

    cuda_run = validate_run_snapshot(
        read_json(args.cuda_run, "CUDA workflow run snapshot"),
        label="CUDA workflow run",
        expected_path=str(BACKENDS["cuda"]["workflow"]),
        candidate_sha=args.candidate_sha,
    )
    cuda_jobs = validate_jobs_snapshot(
        read_json(args.cuda_jobs, "CUDA jobs snapshot"),
        backends=("cuda",),
        run=cuda_run,
        candidate_sha=args.candidate_sha,
    )
    abi = read_json(args.cuda_abi_manifest, "staged CUDA ABI manifest")
    require(
        abi.get("schema_version") == 1
        and abi.get("backend") == "cuda"
        and abi.get("target_triple") == BACKENDS["cuda"]["target"]
        and str(abi.get("cuda_compute_capability")) == "89"
        and abi.get("release_candidate_sha") == args.candidate_sha
        and abi.get("release_candidate_tag") == args.candidate_tag
        and normalized_positive_int(abi.get("workflow_run_id"), "CUDA ABI workflow_run_id")
        == cuda_run["id"]
        and normalized_positive_int(
            abi.get("workflow_run_attempt"), "CUDA ABI workflow_run_attempt"
        )
        == cuda_run["attempt"]
        and abi.get("cargo_features")
        == ["cuda", "vllm-moe-marlin", "vllm-paged-attn-v2"],
        "staged CUDA ABI does not bind the native-set workflow run/candidate",
    )
    return {
        "candidate": candidate,
        "source_bundle": {
            "manifest": file_ref(source_manifest_path),
            "archive": file_ref(source_archive),
            "bundle_id": source_manifest["bundle_id"],
        },
        "native_set": {
            "archive": file_ref(native_archive),
            "workflow_url": pinned_url,
            "workflow_sha256": pinned_sha,
            "identity": native_set.public_identity(provided),
            "source_revisions": sorted(revisions),
        },
        "cuda_run": cuda_run,
        "cuda_job": cuda_jobs["cuda"],
        "cuda_abi": file_ref(args.cuda_abi_manifest),
        "workflow": {
            "path": str(cuda_workflow_path),
            "sha256": sha256_file(cuda_workflow_path),
        },
    }


def run_workflow_policy(args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    out = ensure_fresh_out(args.out)
    evidence = materialize_workflow_evidence(out, args, validate_workflow_policy(args))
    pass_line = f"{WORKFLOW_PASS_PREFIX}: {out}"
    write_gate_manifest(
        out,
        lane="release-workflow-policy",
        pass_line=pass_line,
        evidence=evidence,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    validate_workflow_policy_manifest(out / "gate.manifest.json")
    print(pass_line)
    return 0


def run_native_set(args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    out = ensure_fresh_out(args.out)
    evidence = materialize_native_evidence(out, args, validate_native_set(args))
    pass_line = f"{NATIVE_PASS_PREFIX}: {out}"
    write_gate_manifest(
        out,
        lane="native-operator-set",
        pass_line=pass_line,
        evidence=evidence,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    validate_native_set_manifest(out / "gate.manifest.json")
    print(pass_line)
    return 0


def fake_run(run_id: int, path: str, candidate_sha: str) -> dict[str, Any]:
    return {
        "id": run_id,
        "run_attempt": 1,
        "path": path,
        "event": "workflow_dispatch",
        "head_sha": candidate_sha,
        "status": "completed",
        "conclusion": "success",
        "repository": {"full_name": GITHUB_REPOSITORY},
    }


def fake_jobs(
    backends: tuple[str, ...],
    *,
    run_id: int,
    candidate_sha: str,
) -> dict[str, Any]:
    return {
        "total_count": len(backends),
        "jobs": [
            {
                "id": 9000 + index,
                "name": BACKENDS[backend]["job"],
                "status": "completed",
                "conclusion": "success",
                "head_sha": candidate_sha,
                "run_attempt": 1,
                "run_url": f"https://api.github.com/repos/{GITHUB_REPOSITORY}/actions/runs/{run_id}",
                "steps": [
                    {
                        "name": name,
                        "status": "completed",
                        "conclusion": "success",
                    }
                    for name in REQUIRED_JOB_STEPS[backend]
                ],
            }
            for index, backend in enumerate(backends)
        ],
    }


def create_tarball(path: Path, binary: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w:gz") as archive:
        for name, payload, mode in (
            ("ferrum", binary, 0o755),
            ("LICENSE", b"fixture license\n", 0o644),
            ("README.md", b"fixture readme\n", 0o644),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = mode
            archive.addfile(info, __import__("io").BytesIO(payload))
    return sha256_file(path)


def create_staged_fixture(
    root: Path,
    *,
    backend: str,
    candidate_sha: str,
    candidate_tag: str,
    run_id: int,
) -> tuple[Path, dict[str, Any]]:
    directory = root / "staged" / backend
    directory.mkdir(parents=True)
    asset = str(BACKENDS[backend]["asset"])
    audit_name = str(BACKENDS[backend]["audit"])
    binary = f"fixture-{backend}-binary".encode()
    asset_sha = create_tarball(directory / asset, binary)
    binary_sha = sha256_bytes(binary)
    audit = f"fixture {backend} dependencies\nlibc\n".encode()
    (directory / audit_name).write_bytes(audit)
    audit_sha = sha256_bytes(audit)
    (directory / f"{asset}.sha256").write_text(f"{asset_sha}  {asset}\n")
    (directory / f"{asset}.binary.sha256").write_text(f"{binary_sha}  ferrum\n")
    common = {
        "schema_version": 1,
        "asset_name": asset,
        "asset_sha256": asset_sha,
        "binary_name": "ferrum",
        "binary_sha256": binary_sha,
        "release_candidate_sha": candidate_sha,
        "release_candidate_tag": candidate_tag,
        "staging_label": "v0.8.4-rc",
        "workflow_run_id": str(run_id),
        "workflow_run_attempt": "1",
    }
    write_json(directory / f"{asset}.version.json", {**common, "version": VERSION})
    write_json(
        directory / f"{asset}.dependency.json",
        {
            **common,
            "audit_file": audit_name,
            "audit_sha256": audit_sha,
            "forbidden_runtime_linkage": ["python", "torch", "vllm"],
            "forbidden_runtime_linkage_found": False,
        },
    )
    abi = {
        **common,
        "backend": BACKENDS[backend]["backend"],
        "target_triple": BACKENDS[backend]["target"],
        "dependency_audit_sha256": audit_sha,
    }
    if backend == "cuda":
        abi.update(
            {
                "cuda_compute_capability": "89",
                "cargo_features": ["cuda", "vllm-moe-marlin", "vllm-paged-attn-v2"],
            }
        )
    write_json(directory / f"{asset}.abi.json", abi)
    zip_path = root / f"{backend}.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.iterdir()):
            archive.write(path, arcname=path.name)
    artifact = {
        "id": 8000 + list(BACKENDS).index(backend),
        "name": f"{asset.removesuffix('.tar.gz')}-v0.8.4-rc-{candidate_sha}",
        "size_in_bytes": zip_path.stat().st_size,
        "digest": f"sha256:{sha256_file(zip_path)}",
        "expired": False,
        "workflow_run": {"id": run_id, "head_sha": candidate_sha},
    }
    return zip_path, artifact


def expect_failure(label: str, action: Any) -> None:
    try:
        action()
    except (GateError, source_bundle.BundleError, native_set.NativeOperatorSetEvidenceError):
        return
    raise GateError(f"self-test mutation unexpectedly passed: {label}")


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-v084-workflow-native-") as raw:
        root = Path(raw)
        readme_paths = {
            "english": root / "README.md",
            "chinese": root / "README_zh.md",
        }
        shutil.copy2(REPO_ROOT / "README.md", readme_paths["english"])
        shutil.copy2(REPO_ROOT / "README_zh.md", readme_paths["chinese"])
        validate_readme_contract(paths=readme_paths)
        readme_english = readme_paths["english"].read_text(encoding="utf-8")
        readme_paths["english"].write_text(
            readme_english.replace(
                "ferrum run qwen3.5:4b-q4_k_m --disable-thinking",
                "ferrum run qwen3.5:0.8b --disable-thinking",
                1,
            ),
            encoding="utf-8",
        )
        expect_failure(
            "README smaller primary model substitution",
            lambda: validate_readme_contract(paths=readme_paths),
        )
        readme_paths["english"].write_text(
            readme_english.replace(
                "ferrum run qwen3.5:4b-q4_k_m --disable-thinking",
                "ferrum run qwen3.5:4b-q4_k_m --disable-thinking --max-tokens 1",
                1,
            ),
            encoding="utf-8",
        )
        expect_failure(
            "README first run exact command",
            lambda: validate_readme_contract(paths=readme_paths),
        )
        readme_paths["english"].write_text(
            readme_english.replace("ferrum --help\n", "", 1), encoding="utf-8"
        )
        expect_failure(
            "README global help command",
            lambda: validate_readme_contract(paths=readme_paths),
        )
        readme_paths["english"].write_text(readme_english, encoding="utf-8")
        candidate_sha = "1" * 40
        candidate_tag = "v0.8.4-rc.9"
        release_run_id = 8041
        cuda_run_id = 8042
        paths: dict[str, Path] = {}
        artifacts: dict[str, dict[str, Any]] = {}
        for backend in BACKENDS:
            run_id = release_run_id if backend in {"cpu", "metal"} else cuda_run_id
            paths[backend], artifacts[backend] = create_staged_fixture(
                root,
                backend=backend,
                candidate_sha=candidate_sha,
                candidate_tag=candidate_tag,
                run_id=run_id,
            )
        documents = {
            "release_run": fake_run(release_run_id, str(BACKENDS["cpu"]["workflow"]), candidate_sha),
            "cuda_run": fake_run(cuda_run_id, str(BACKENDS["cuda"]["workflow"]), candidate_sha),
            "release_jobs": fake_jobs(
                ("cpu", "metal"),
                run_id=release_run_id,
                candidate_sha=candidate_sha,
            ),
            "cuda_jobs": fake_jobs(
                ("cuda",),
                run_id=cuda_run_id,
                candidate_sha=candidate_sha,
            ),
            "release_artifacts": {"total_count": 2, "artifacts": [artifacts["cpu"], artifacts["metal"]]},
            "cuda_artifacts": {"total_count": 1, "artifacts": [artifacts["cuda"]]},
        }
        for name, document in documents.items():
            write_json(root / f"{name}.json", document)
        workflow_args = argparse.Namespace(
            candidate_sha=candidate_sha,
            candidate_tag=candidate_tag,
            release_run=root / "release_run.json",
            release_jobs=root / "release_jobs.json",
            release_artifacts=root / "release_artifacts.json",
            cuda_run=root / "cuda_run.json",
            cuda_jobs=root / "cuda_jobs.json",
            cuda_artifacts=root / "cuda_artifacts.json",
            cpu_zip=paths["cpu"],
            metal_zip=paths["metal"],
            cuda_zip=paths["cuda"],
            staged_root=root / "staged",
        )
        workflow_result = validate_workflow_policy(workflow_args, verify_checkout=False)
        require(len(workflow_result["bundles"]) == 3, "workflow self-test bundle count differs")
        bad_run = read_json(root / "cuda_run.json", "fixture CUDA run")
        bad_run["head_sha"] = "3" * 40
        write_json(root / "cuda_run.bad.json", bad_run)
        bad_args = argparse.Namespace(**vars(workflow_args))
        bad_args.cuda_run = root / "cuda_run.bad.json"
        expect_failure(
            "workflow candidate mismatch",
            lambda: validate_workflow_policy(bad_args, verify_checkout=False),
        )
        substituted_jobs = read_json(root / "cuda_jobs.json", "fixture CUDA jobs")
        substituted_jobs["jobs"][0]["head_sha"] = "4" * 40
        substituted_jobs["jobs"][0]["run_url"] = (
            f"https://api.github.com/repos/{GITHUB_REPOSITORY}/actions/runs/999999"
        )
        write_json(root / "cuda_jobs.substituted.json", substituted_jobs)
        substituted_args = argparse.Namespace(**vars(workflow_args))
        substituted_args.cuda_jobs = root / "cuda_jobs.substituted.json"
        expect_failure(
            "workflow jobs snapshot substitution",
            lambda: validate_workflow_policy(substituted_args, verify_checkout=False),
        )
        extra_file = root / "staged/cpu/unexpected.txt"
        extra_file.write_text("unexpected fixture member\n", encoding="utf-8")
        expect_failure(
            "workflow extracted file count",
            lambda: validate_workflow_policy(workflow_args, verify_checkout=False),
        )
        extra_file.unlink()

        source_root = root / "source"
        definition_root = root / "definitions"
        (source_root / "kernels").mkdir(parents=True)
        definition_root.mkdir()
        (source_root / "kernels/fixture.cu").write_text("// fixture\n")
        write_json(
            definition_root / "fixture.json",
            {
                "schema_version": 3,
                "operator": "ferrum.cuda.fixture",
                "translation_units": ["kernels/fixture.cu"],
                "headers": [],
            },
        )
        source_archive = root / "source.tar.gz"
        source_manifest_path = root / "source.json"
        source_bundle.create(
            argparse.Namespace(
                source_root=source_root,
                definition_root=definition_root,
                archive=source_archive,
                manifest=source_manifest_path,
                github_repository=GITHUB_REPOSITORY,
                github_tag="fixture",
            )
        )
        source_manifest = read_json(source_manifest_path, "fixture source manifest")
        archive_root = root / "native-archive-root"
        lock_root = archive_root / "inputs"
        lock = native_set.create_selftest_native_operator_set(
            lock_root,
            REQUIRED_CUDA_NATIVE_OPERATORS,
        )
        lock_document = read_json(lock, "fixture native lock")
        for row in lock_document["artifacts"]:
            manifest_path = lock_root / row["manifest_path"]
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "source_package": {"revision": source_manifest["bundle_id"]},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            row["manifest"] = {
                "path": row["manifest"]["path"],
                "sha256": sha256_file(manifest_path),
                "size_bytes": manifest_path.stat().st_size,
            }
        write_json(lock, lock_document)
        native_set.validate_native_operator_set(lock, REQUIRED_CUDA_NATIVE_OPERATORS)
        raw_tar = root / "native-set.tar"
        with tarfile.open(raw_tar, mode="w") as archive:
            archive.add(lock_root, arcname="inputs")
        native_archive = root / "native-set.tar.zst"
        compressed = subprocess.run(
            ["zstd", "--quiet", "--force", str(raw_tar), "-o", str(native_archive)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        require(compressed.returncode == 0, f"self-test zstd failed: {compressed.stderr}")
        workflow_document = {
            "env": {
                "NATIVE_OPERATOR_SET_ARCHIVE_URL": (
                    f"https://github.com/{GITHUB_REPOSITORY}/releases/download/fixture/"
                    f"{native_archive.name}"
                ),
                "NATIVE_OPERATOR_SET_ARCHIVE_SHA256": sha256_file(native_archive),
            }
        }
        native_args = argparse.Namespace(
            candidate_sha=candidate_sha,
            candidate_tag=candidate_tag,
            source_bundle_manifest=source_manifest_path,
            source_bundle_archive=source_archive,
            native_set_archive=native_archive,
            native_set_lock=lock,
            cuda_run=root / "cuda_run.json",
            cuda_jobs=root / "cuda_jobs.json",
            cuda_abi_manifest=(
                root / "staged/cuda" / f"{BACKENDS['cuda']['asset']}.abi.json"
            ),
        )
        native_result = validate_native_set(
            native_args,
            verify_checkout=False,
            validate_static_workflow=False,
            workflow_document=workflow_document,
        )
        require(
            native_result["native_set"]["identity"]["operators"]
            == sorted(REQUIRED_CUDA_NATIVE_OPERATORS),
            "native self-test operator set differs",
        )
        bad_workflow = json.loads(json.dumps(workflow_document))
        bad_workflow["env"]["NATIVE_OPERATOR_SET_ARCHIVE_SHA256"] = "f" * 64
        expect_failure(
            "native archive pin mismatch",
            lambda: validate_native_set(
                native_args,
                verify_checkout=False,
                validate_static_workflow=False,
                workflow_document=bad_workflow,
            ),
        )
    print(SELFTEST_PASS_LINE)


def add_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-sha", required=True, help="exact clean 40-character candidate commit")
    parser.add_argument("--candidate-tag", required=True, help="annotated v0.8.4-rc.N tag")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run hermetic positive and fail-closed fixtures")
    subparsers = parser.add_subparsers(dest="mode")
    workflow = subparsers.add_parser("workflow-policy", help="validate workflow policy, live snapshots, and staged bundles")
    add_candidate_arguments(workflow)
    workflow.add_argument("--release-run", required=True, type=Path)
    workflow.add_argument("--release-jobs", required=True, type=Path)
    workflow.add_argument("--release-artifacts", required=True, type=Path)
    workflow.add_argument("--cuda-run", required=True, type=Path)
    workflow.add_argument("--cuda-jobs", required=True, type=Path)
    workflow.add_argument("--cuda-artifacts", required=True, type=Path)
    workflow.add_argument("--cpu-zip", required=True, type=Path)
    workflow.add_argument("--metal-zip", required=True, type=Path)
    workflow.add_argument("--cuda-zip", required=True, type=Path)
    workflow.add_argument("--staged-root", required=True, type=Path, help="root containing cpu/, metal/, cuda/ with exactly 7 files each")
    workflow.add_argument("--out", required=True, type=Path)
    native = subparsers.add_parser("native-set", help="validate the checked-in source bundle and pinned public native set")
    add_candidate_arguments(native)
    native.add_argument("--source-bundle-manifest", required=True, type=Path)
    native.add_argument("--source-bundle-archive", required=True, type=Path)
    native.add_argument("--native-set-archive", required=True, type=Path)
    native.add_argument("--native-set-lock", required=True, type=Path)
    native.add_argument("--cuda-run", required=True, type=Path)
    native.add_argument("--cuda-jobs", required=True, type=Path)
    native.add_argument("--cuda-abi-manifest", required=True, type=Path)
    native.add_argument("--out", required=True, type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.self_test:
            require(args.mode is None, "--self-test cannot be combined with a subcommand")
            run_self_test()
            return 0
        if args.mode == "workflow-policy":
            return run_workflow_policy(args)
        if args.mode == "native-set":
            return run_native_set(args)
        parser.error("choose workflow-policy, native-set, or --self-test")
    except (
        GateError,
        workflow_policy.PolicyError,
        source_bundle.BundleError,
        native_set.NativeOperatorSetEvidenceError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"FERRUM 0.8.4 WORKFLOW NATIVE GATE FAIL: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
