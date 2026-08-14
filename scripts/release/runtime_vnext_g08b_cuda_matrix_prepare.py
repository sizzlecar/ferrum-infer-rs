#!/usr/bin/env python3
"""Prepare G08B model-matrix inputs; the path remains CUDA-named for compatibility."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import bounded_command
import runtime_vnext_baseline_scenarios as matrix
import runtime_vnext_native_operator_set as native_operator_set


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_REPO_PATH = SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()
G08B_MODEL_KEY = "m2-qwen35-35b-a3b"
G08B_SERVED_MODEL_NAME = "m2-qwen35-35b-a3b"
BUILD_DIR = Path("build/candidate")
BUILD_RECEIPT_REL = BUILD_DIR / "candidate-build-receipt.json"
BUILD_BINARY_REL = BUILD_DIR / "ferrum"
STAGED_MANIFEST_REL = BUILD_DIR / "staged-assets-manifest.json"
STAGED_METADATA_DIR = BUILD_DIR / "staged-metadata"
CUDA_CORRECTNESS_IMPORT_REL = BUILD_DIR / "cuda-correctness-artifact"
MODELS_LOCK_REL = Path("models.lock.json")
EXECUTION_MANIFEST_REL = Path("execution-manifest.json")
BUILD_ENV_KEYS = (
    "CARGO_BUILD_JOBS",
    "CARGO_HOME",
    "CARGO_TARGET_DIR",
    "CC",
    "CUDA_HOME",
    "CUDA_PATH",
    "CXX",
    "LD_LIBRARY_PATH",
    "NVCC",
    "PATH",
    "RUSTFLAGS",
    "RUSTUP_HOME",
    "MACOSX_DEPLOYMENT_TARGET",
    "SDKROOT",
)


@dataclass(frozen=True)
class BackendSpec:
    backend: str
    model_key: str
    model_label: str
    model_lock_path: Path
    lock_id: str
    weight_revision: str
    weight_format: str
    weight_file_count: int
    semantic_file_count: int
    build_ready_prefix: str
    manifest_ready_prefix: str
    prepare_selftest_pass_line: str
    probe_commands: dict[str, list[str]]
    typed_run_config: dict[str, Any]
    typed_serve_config: dict[str, Any]
    run_extra_args: tuple[str, ...]
    serve_extra_args: tuple[str, ...]
    source_repo_paths: tuple[str, ...]
    selftest_temp_prefix: str


CUDA_SPEC = BackendSpec(
    backend="cuda",
    model_key=G08B_MODEL_KEY,
    model_label="G08B M2",
    model_lock_path=SCRIPT_PATH.parent
    / "configs/runtime_vnext_g08b_m2_cuda.models.lock.json",
    lock_id="runtime-vnext-g08b-m2-cuda-v1",
    weight_revision="3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b",
    weight_format="gptq_int4_moe_wna16",
    weight_file_count=19,
    semantic_file_count=6,
    build_ready_prefix="FERRUM RUNTIME VNEXT G08B CUDA BUILD READY",
    manifest_ready_prefix="FERRUM RUNTIME VNEXT G08B CUDA MANIFEST READY",
    prepare_selftest_pass_line="FERRUM RUNTIME VNEXT G08B CUDA PREPARE SELFTEST PASS",
    probe_commands={
        "cargo": ["cargo", "--version"],
        "rustc": ["rustc", "--version", "--verbose"],
        "nvcc": ["nvcc", "--version"],
        "nvidia_smi": ["nvidia-smi"],
    },
    typed_run_config={
        "backend": "cuda",
        "gpu_devices": [0],
        "gpu_memory_utilization": 0.9,
    },
    typed_serve_config={
        "backend": "cuda",
        "gpu_devices": [0],
        "gpu_memory_utilization": 0.9,
        "served_model_name": G08B_SERVED_MODEL_NAME,
    },
    run_extra_args=(
        "--gpu-devices",
        "0",
        "--gpu-memory-utilization",
        "0.90",
    ),
    serve_extra_args=(
        "--gpu-devices",
        "0",
        "--gpu-memory-utilization",
        "0.90",
        "--served-model-name",
        G08B_SERVED_MODEL_NAME,
    ),
    source_repo_paths=(SCRIPT_REPO_PATH,),
    selftest_temp_prefix="ferrum-g08b-cuda-prepare-",
)

METAL_SPEC = BackendSpec(
    backend="metal",
    model_key=G08B_MODEL_KEY,
    model_label="G08B M2",
    model_lock_path=SCRIPT_PATH.parent
    / "configs/runtime_vnext_g08b_m2_metal.models.lock.json",
    lock_id="runtime-vnext-g08b-m2-metal-v1",
    weight_revision="bc014a17be43adabd7066b7a86075ff935c6a4e2",
    weight_format="gguf_q4_k_s",
    weight_file_count=1,
    semantic_file_count=6,
    build_ready_prefix="FERRUM RUNTIME VNEXT G08B METAL BUILD READY",
    manifest_ready_prefix="FERRUM RUNTIME VNEXT G08B METAL MANIFEST READY",
    prepare_selftest_pass_line="FERRUM RUNTIME VNEXT G08B METAL PREPARE SELFTEST PASS",
    probe_commands={
        "cargo": ["cargo", "--version"],
        "rustc": ["rustc", "--version", "--verbose"],
        "xcodebuild": ["xcodebuild", "-version"],
        "system_profiler": [
            "system_profiler",
            "SPDisplaysDataType",
        ],
    },
    typed_run_config={
        "backend": "metal",
        "gpu_memory_utilization": 0.9,
    },
    typed_serve_config={
        "backend": "metal",
        "gpu_memory_utilization": 0.9,
        "served_model_name": G08B_SERVED_MODEL_NAME,
    },
    run_extra_args=(
        "--gpu-memory-utilization",
        "0.90",
    ),
    serve_extra_args=(
        "--gpu-memory-utilization",
        "0.90",
        "--served-model-name",
        G08B_SERVED_MODEL_NAME,
    ),
    source_repo_paths=(
        SCRIPT_REPO_PATH,
        "scripts/release/runtime_vnext_g08b_metal_matrix_prepare.py",
    ),
    selftest_temp_prefix="ferrum-g08b-metal-prepare-",
)

BACKEND_SPECS = {
    CUDA_SPEC.backend: CUDA_SPEC,
    METAL_SPEC.backend: METAL_SPEC,
}


class PreparationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_text(argv: Sequence[str], *, timeout: float = 30.0) -> str:
    result = subprocess.run(
        list(argv),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    require(
        result.returncode == 0,
        f"command failed ({result.returncode}): {list(argv)!r}: {result.stderr.strip()[:512]}",
    )
    return result.stdout.strip()


def source_observation(spec: BackendSpec) -> dict[str, Any]:
    status = run_text(["git", "status", "--short", "--untracked-files=all"])
    require(not status, "candidate preparation requires a clean worktree")
    source_git_sha = run_text(["git", "rev-parse", "HEAD"])
    source_tree_sha = run_text(["git", "rev-parse", "HEAD^{tree}"])
    require(matrix.GIT_SHA_RE.fullmatch(source_git_sha) is not None, "candidate source SHA is invalid")
    require(matrix.GIT_SHA_RE.fullmatch(source_tree_sha) is not None, "candidate source tree SHA is invalid")
    for source_repo_path in spec.source_repo_paths:
        require(
            run_text(
                ["git", "cat-file", "-e", f"{source_git_sha}:{source_repo_path}"]
            )
            == "",
            (
                "candidate prepare source is not checked in at the source SHA: "
                f"{source_repo_path}"
            ),
        )
    return {
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "dirty_status": {"is_dirty": False, "status_short": []},
    }


def artifact_root(raw: str) -> Path:
    root = Path(raw).expanduser().resolve()
    try:
        root.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise PreparationError("artifact root must be outside the Git worktree")
    root.mkdir(parents=True, exist_ok=True)
    require(root.is_dir(), f"artifact root is not a directory: {root}")
    return root


def capture_probe(root: Path, name: str, argv: Sequence[str]) -> dict[str, str]:
    result = subprocess.run(
        list(argv),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30.0,
        check=False,
    )
    path = root / BUILD_DIR / f"{name}.log"
    text = (
        f"command={json.dumps(list(argv), separators=(',', ':'))}\n"
        f"returncode={result.returncode}\n"
        f"stdout:\n{result.stdout.rstrip()}\n"
        f"stderr:\n{result.stderr.rstrip()}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    require(result.returncode == 0, f"required build probe failed: {name}")
    require(path.stat().st_size >= 16, f"required build probe is empty: {name}")
    return matrix.existing_artifact_ref(root, path, "runtime-log")


def validate_checked_in_lock(spec: BackendSpec) -> dict[str, Any]:
    require(
        spec.model_lock_path.is_file(),
        f"checked-in model lock is missing: {spec.model_lock_path}",
    )
    document = matrix.require_object(
        matrix.read_json(spec.model_lock_path),
        f"{spec.model_label} {spec.backend.upper()} model lock",
    )
    require(
        document.get("schema_version") == matrix.SCHEMA_VERSION,
        f"{spec.model_label} model lock schema mismatch",
    )
    require(
        document.get("lock_id") == spec.lock_id,
        f"{spec.model_label} model lock id mismatch",
    )
    sources = matrix.locked_execution_sources(
        document,
        spec.model_key,
        spec.backend,
    )
    require(
        sources["weight_revision"] == spec.weight_revision,
        f"{spec.model_label} weight revision drift",
    )
    require(
        sources["weight_format"] == spec.weight_format,
        f"{spec.model_label} weight format drift",
    )
    require(
        len(sources["weight_files"]) == spec.weight_file_count,
        (
            f"{spec.model_label} weight lock must contain exactly "
            f"{spec.weight_file_count} files"
        ),
    )
    require(
        len(sources["semantic_source"]["files"]) == spec.semantic_file_count,
        (
            f"{spec.model_label} semantic lock must contain exactly "
            f"{spec.semantic_file_count} files"
        ),
    )
    return {"document": document, "sources": sources}


def build_candidate(
    root: Path,
    hardware_id: str,
    spec: BackendSpec,
    native_operator_set_lock: Path | None,
) -> Path:
    require(hardware_id.strip() == hardware_id and hardware_id, "hardware id must be non-empty and trimmed")
    receipt_path = root / BUILD_RECEIPT_REL
    binary_path = root / BUILD_BINARY_REL
    require(not receipt_path.exists(), f"candidate build receipt already exists: {receipt_path}")
    require(not binary_path.exists(), f"candidate build binary already exists: {binary_path}")
    if spec.backend == "cuda":
        require(
            native_operator_set_lock is not None,
            "CUDA candidate build requires --native-operator-set-lock",
        )
        require(
            not native_operator_set_lock.is_symlink()
            and native_operator_set_lock.is_file(),
            "CUDA native operator set lock must be a regular non-symlink file",
        )
        source_native_operator_set_lock = native_operator_set_lock.resolve()
        canonical_native_operator_set_lock = (
            root / matrix.CANDIDATE_NATIVE_OPERATOR_SET_LOCK_REL
        ).resolve()
        try:
            native_operator_set_lock, staged_native_operator_set = (
                native_operator_set.stage_native_operator_set(
                    source_native_operator_set_lock,
                    canonical_native_operator_set_lock.parent,
                    matrix.CANDIDATE_REQUIRED_CUDA_NATIVE_OPERATORS,
                )
            )
        except native_operator_set.NativeOperatorSetEvidenceError as error:
            raise PreparationError(
                f"cannot stage CUDA native operator set: {error}"
            ) from error
        require(
            native_operator_set_lock == canonical_native_operator_set_lock,
            "staged CUDA native operator set lock path is not canonical",
        )
        native_operator_set_lock_identity = matrix.native_operator_set_lock_identity(
            native_operator_set_lock
        )
    else:
        require(
            native_operator_set_lock is None,
            "--native-operator-set-lock is CUDA-only",
        )
        native_operator_set_lock_identity = None
    build_command = matrix.candidate_build_command(
        spec.backend,
        native_operator_set_lock,
    )
    before = source_observation(spec)
    probes = {
        name: capture_probe(root, name, argv)
        for name, argv in spec.probe_commands.items()
    }
    bounded_path = root / BUILD_DIR / "bounded-command-receipt.json"
    stdout_path = root / BUILD_DIR / "stdout.log"
    stderr_path = root / BUILD_DIR / "stderr.log"
    wrapper_rc, bounded_receipt = bounded_command.run_bounded_command(
        command=build_command,
        cwd=REPO_ROOT,
        receipt_path=bounded_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        limits=bounded_command.Limits(
            wall_timeout_seconds=2700.0,
            max_processes=64,
            max_group_threads=256,
            max_per_process_threads=64,
            sample_interval_seconds=0.2,
            max_sampling_errors=3,
            term_grace_seconds=2.0,
        ),
    )
    require(
        wrapper_rc == 0
        and bounded_receipt.get("status") == "pass"
        and bounded_receipt.get("rc") == 0,
        f"bounded {spec.backend.upper()} build failed; inspect {bounded_path}",
    )
    after = source_observation(spec)
    require(
        after == before,
        f"candidate source changed during the {spec.backend.upper()} build",
    )
    if native_operator_set_lock_identity is not None:
        require(
            matrix.native_operator_set_lock_identity(native_operator_set_lock)
            == native_operator_set_lock_identity,
            "staged CUDA native operator set lock changed during the build",
        )
        try:
            staged_after = native_operator_set.validate_native_operator_set(
                native_operator_set_lock,
                matrix.CANDIDATE_REQUIRED_CUDA_NATIVE_OPERATORS,
            )
        except native_operator_set.NativeOperatorSetEvidenceError as error:
            raise PreparationError(
                f"staged CUDA native operator set changed during the build: {error}"
            ) from error
        require(
            native_operator_set.public_identity(staged_after)
            == native_operator_set.public_identity(staged_native_operator_set),
            "staged CUDA native operator set identity changed during the build",
        )
    cargo_metadata = json.loads(
        run_text(["cargo", "metadata", "--format-version", "1", "--no-deps"])
    )
    target_directory = Path(
        matrix.require_string(
            cargo_metadata.get("target_directory"),
            "cargo metadata target_directory",
        )
    )
    built_binary = target_directory / "release/ferrum"
    require(
        built_binary.is_file(),
        f"{spec.backend.upper()} build did not produce {built_binary}",
    )
    require(
        os.access(built_binary, os.X_OK),
        f"{spec.backend.upper()} build output is not executable",
    )
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_binary, binary_path)
    require(os.access(binary_path, os.X_OK), "copied candidate binary is not executable")
    binary_ref = matrix.existing_artifact_ref(root, binary_path, "binary")
    stdout_ref = matrix.existing_artifact_ref(
        root,
        stdout_path,
        "stdout-log",
        allow_empty=True,
    )
    stderr_ref = matrix.existing_artifact_ref(root, stderr_path, "stderr-log")
    receipt = {
        "schema_version": matrix.SCHEMA_VERSION,
        "artifact_type": matrix.CANDIDATE_BUILD_RECEIPT_TYPE,
        "status": "pass",
        "execution_contract": matrix.G08_EXECUTION_CONTRACT,
        **before,
        "hardware_id": hardware_id,
        "backend": spec.backend,
        "artifact_root": str(root),
        "repository_root": str(REPO_ROOT),
        "source_observations": {"before": before, "after": after},
        "command": build_command,
        "build_environment": {
            key: os.environ[key]
            for key in BUILD_ENV_KEYS
            if key in os.environ
        },
        "cargo_target_directory": str(target_directory),
        "returncode": 0,
        "started_at": bounded_receipt["started_at"],
        "finished_at": bounded_receipt["ended_at"],
        "duration_sec": bounded_receipt["duration_seconds"],
        "binary_artifact": binary_ref,
        "binary_sha256": binary_ref["sha256"],
        "bounded_receipt": matrix.existing_artifact_ref(root, bounded_path, "raw-json"),
        "stdout": stdout_ref,
        "stderr": stderr_ref,
        "probe_artifacts": probes,
    }
    if native_operator_set_lock_identity is not None:
        receipt["native_operator_set_lock"] = native_operator_set_lock_identity
    write_json(receipt_path, receipt)
    matrix.validate_candidate_build_receipt(
        root,
        matrix.existing_artifact_ref(root, receipt_path, "raw-json"),
        expected={
            "source_git_sha": before["source_git_sha"],
            "source_tree_sha": before["source_tree_sha"],
            "hardware_id": hardware_id,
            "backend": spec.backend,
            "binary_sha256": binary_ref["sha256"],
            "binary_path": binary_path,
        },
        allow_internal_fixture=False,
    )
    print(f"{spec.build_ready_prefix}: {receipt_path}")
    return receipt_path


def bind_cuda_correctness_artifact(
    root: Path,
    *,
    correctness_build_manifest: Path,
    hardware_id: str,
    spec: BackendSpec,
) -> Path:
    require(spec.backend == "cuda", "CUDA correctness artifact binding is CUDA-only")
    require(
        hardware_id.strip() == hardware_id and hardware_id,
        "hardware id must be non-empty and trimmed",
    )
    receipt_path = root / BUILD_RECEIPT_REL
    import_root = root / CUDA_CORRECTNESS_IMPORT_REL
    candidate_binary = root / BUILD_BINARY_REL
    require(not receipt_path.exists(), f"candidate build receipt already exists: {receipt_path}")
    require(not import_root.exists(), f"CUDA correctness import already exists: {import_root}")
    require(not candidate_binary.exists(), f"candidate build binary already exists: {candidate_binary}")
    source_manifest = correctness_build_manifest.expanduser().resolve()
    require(source_manifest.is_file(), f"CUDA correctness build manifest is missing: {source_manifest}")
    before = source_observation(spec)
    source_correctness, _, _ = matrix.validate_cuda_correctness_build_artifact(
        source_manifest,
        expected={
            "source_git_sha": before["source_git_sha"],
            "source_tree_sha": before["source_tree_sha"],
        },
        allow_internal_fixture=False,
    )
    source_native_lock = matrix.validate_plain_artifact_ref(
        source_manifest.parent,
        source_correctness.get("native_operator_set_artifact"),
        "CUDA correctness portable native operator set lock",
    )
    staged_native_lock, staged_native_set = (
        native_operator_set.stage_native_operator_set(
            source_native_lock,
            root / BUILD_DIR,
            matrix.CANDIDATE_REQUIRED_CUDA_NATIVE_OPERATORS,
        )
    )
    shutil.copytree(source_manifest.parent, import_root, symlinks=False)
    copied_manifest = import_root / source_manifest.relative_to(source_manifest.parent)
    imported, binary_path, binary_sha = matrix.validate_cuda_correctness_build_artifact(
        copied_manifest,
        expected={
            "source_git_sha": before["source_git_sha"],
            "source_tree_sha": before["source_tree_sha"],
        },
        allow_internal_fixture=False,
    )
    require(
        imported["native_source_policy"] == "cache-only",
        "imported CUDA correctness artifact is not cache-only",
    )
    imported_native_lock = matrix.validate_plain_artifact_ref(
        import_root,
        imported.get("native_operator_set_artifact"),
        "imported CUDA correctness portable native operator set lock",
    )
    imported_native_set = native_operator_set.validate_native_operator_set(
        imported_native_lock,
        matrix.CANDIDATE_REQUIRED_CUDA_NATIVE_OPERATORS,
    )
    require(
        native_operator_set.public_identity(imported_native_set)
        == native_operator_set.public_identity(staged_native_set),
        "candidate native operator closure differs from the correctness artifact",
    )
    probes = {
        name: capture_probe(root, name, argv)
        for name, argv in spec.probe_commands.items()
    }
    after = source_observation(spec)
    require(after == before, "candidate source changed during CUDA correctness artifact binding")
    shutil.copy2(binary_path, candidate_binary)
    require(
        candidate_binary.is_file()
        and matrix.file_sha256(candidate_binary) == binary_sha,
        "canonical candidate binary differs from the correctness artifact",
    )
    binary_ref = matrix.existing_artifact_ref(root, candidate_binary, "binary")
    require(binary_ref["sha256"] == binary_sha, "imported CUDA correctness binary SHA drift")
    receipt = {
        "schema_version": matrix.SCHEMA_VERSION,
        "artifact_type": matrix.CANDIDATE_BUILD_RECEIPT_TYPE,
        "status": "pass",
        "execution_contract": matrix.G08_EXECUTION_CONTRACT,
        **before,
        "hardware_id": hardware_id,
        "backend": "cuda",
        "artifact_root": str(root),
        "repository_root": str(REPO_ROOT),
        "build_mode": matrix.CUDA_CORRECTNESS_BUILD_MODE,
        "bound_at": matrix.iso_now(),
        "source_observations": {"before": before, "after": after},
        "binary_artifact": binary_ref,
        "binary_sha256": binary_sha,
        "native_operator_set_lock": matrix.native_operator_set_lock_identity(
            staged_native_lock
        ),
        "correctness_build_manifest": matrix.existing_artifact_ref(
            root,
            copied_manifest,
            "raw-json",
        ),
        "probe_artifacts": probes,
    }
    write_json(receipt_path, receipt)
    matrix.validate_candidate_build_receipt(
        root,
        matrix.existing_artifact_ref(root, receipt_path, "raw-json"),
        expected={
            "source_git_sha": before["source_git_sha"],
            "source_tree_sha": before["source_tree_sha"],
            "hardware_id": hardware_id,
            "backend": "cuda",
            "binary_sha256": binary_sha,
            "binary_path": candidate_binary,
        },
        allow_internal_fixture=False,
    )
    print(f"FERRUM RUNTIME VNEXT G08B CUDA CORRECTNESS BUILD BOUND: {receipt_path}")
    return receipt_path


def safe_relative_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be a non-empty path")
    path = Path(value)
    require(
        not path.is_absolute() and ".." not in path.parts,
        f"{label} must be a safe relative path",
    )
    return path


def staged_file_ref(
    manifest_root: Path,
    raw: Any,
    label: str,
) -> tuple[dict[str, Any], Path]:
    require(isinstance(raw, dict), f"{label} must be an object")
    require(
        set(raw) == {"path", "sha256", "size_bytes"},
        f"{label} must contain exactly path, sha256, and size_bytes",
    )
    relative = safe_relative_path(raw.get("path"), f"{label}.path")
    path = (manifest_root / relative).resolve()
    require(
        path.is_relative_to(manifest_root.resolve()),
        f"{label}.path escapes the staged asset root",
    )
    require(
        path.is_file() and not path.is_symlink(),
        f"{label} is not a regular non-symlink file: {path}",
    )
    digest = matrix.require_sha256(raw.get("sha256"), f"{label}.sha256")
    size = raw.get("size_bytes")
    require(
        type(size) is int and size > 0,
        f"{label}.size_bytes must be a positive integer",
    )
    require(path.stat().st_size == size, f"{label} size mismatch")
    require(matrix.file_sha256(path) == digest, f"{label} SHA256 mismatch")
    return copy.deepcopy(raw), path


def validate_staged_asset_input(
    staged_assets_manifest: Path,
    *,
    backend: str,
    expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    require(backend in {"cuda", "metal"}, "staged matrix backend must be CUDA or Metal")
    manifest_path = staged_assets_manifest.expanduser().resolve()
    require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        f"staged assets manifest is not a regular file: {manifest_path}",
    )
    document = matrix.require_object(
        matrix.read_json(manifest_path),
        "staged assets manifest",
    )
    require(document.get("schema_version") == 1, "staged assets schema_version mismatch")
    require(
        document.get("artifact_type") == "runtime_vnext_staged_assets_manifest",
        "staged assets artifact_type mismatch",
    )
    require(document.get("status") == "pass", "staged assets status is not pass")
    require(document.get("version") == "0.8.0", "staged assets version is not 0.8.0")
    require(
        document.get("publish_release") is False,
        "staged assets must come from publish_release=false",
    )
    release_candidate = matrix.require_object(
        document.get("release_candidate"),
        "staged assets release_candidate",
    )
    require(
        release_candidate.get("dirty") is False,
        "staged assets release candidate is dirty",
    )
    require(
        release_candidate.get("git_sha") == expected_source.get("source_git_sha"),
        "staged assets release candidate source SHA mismatch",
    )
    require(
        release_candidate.get("git_tree_sha")
        == expected_source.get("source_tree_sha"),
        "staged assets release candidate source tree mismatch",
    )
    assets = matrix.require_object(document.get("assets"), "staged assets map")
    require(
        set(assets) == {"cpu", "metal", "cuda"},
        "staged assets must contain exactly cpu, metal, and cuda",
    )
    selected = matrix.require_object(
        assets.get(backend),
        f"staged assets {backend} row",
    )
    require(
        selected.get("backend") == backend,
        f"staged assets selected backend is not {backend}",
    )
    workflow_run_id = selected.get("workflow_run_id")
    require(
        type(workflow_run_id) is int and workflow_run_id > 0,
        "staged asset workflow_run_id must be a positive integer",
    )
    artifact = matrix.require_object(selected.get("artifact"), "staged workflow artifact")
    require(
        set(artifact) == {"id", "name", "digest"}
        and type(artifact.get("id")) is int
        and artifact["id"] > 0
        and isinstance(artifact.get("name"), str)
        and bool(artifact["name"].strip())
        and isinstance(artifact.get("digest"), str)
        and artifact["digest"].startswith("sha256:")
        and len(artifact["digest"]) == 71,
        "staged workflow artifact identity is malformed",
    )
    matrix.require_sha256(
        artifact["digest"].removeprefix("sha256:"),
        "staged workflow artifact digest",
    )
    if backend == "cuda":
        require(selected.get("target_sm") == "89", "staged CUDA target_sm is not 89")

    manifest_root = manifest_path.parent
    resolved_refs: dict[str, dict[str, Any]] = {}
    resolved_paths: dict[str, Path] = {}
    for name in (
        "artifact_manifest",
        "tarball",
        "sha256_file",
        "version_manifest",
        "dependency_abi_manifest",
    ):
        resolved_refs[name], resolved_paths[name] = staged_file_ref(
            manifest_root,
            selected.get(name),
            f"staged {backend} {name}",
        )
    tarball_ref = resolved_refs["tarball"]
    sha_text = resolved_paths["sha256_file"].read_text(
        encoding="utf-8", errors="strict"
    )
    sha_parts = sha_text.split()
    require(
        sha_parts and sha_parts[0] == tarball_ref["sha256"],
        "staged tarball adjacent SHA256 file mismatch",
    )

    binary = matrix.require_object(selected.get("binary"), "staged asset binary")
    require(
        set(binary) == {"archive_path", "sha256", "size_bytes"},
        "staged asset binary identity fields mismatch",
    )
    archive_path = safe_relative_path(
        binary.get("archive_path"),
        "staged asset binary archive_path",
    )
    require(archive_path.name == "ferrum", "staged asset binary basename is not ferrum")
    binary_sha = matrix.require_sha256(
        binary.get("sha256"),
        "staged asset binary SHA256",
    )
    binary_size = binary.get("size_bytes")
    require(
        type(binary_size) is int and binary_size > 0,
        "staged asset binary size must be positive",
    )
    try:
        with tarfile.open(resolved_paths["tarball"], mode="r:*") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.name == archive_path.as_posix()
            ]
            require(
                len(members) == 1,
                "staged tarball must contain the exact binary archive_path once",
            )
            member = members[0]
            require(
                member.isfile() and not member.issym() and not member.islnk(),
                "staged tarball binary is not a regular file",
            )
            require(member.size == binary_size, "staged tarball binary size mismatch")
            stream = archive.extractfile(member)
            require(stream is not None, "staged tarball binary cannot be read")
            digest = hashlib.sha256()
            observed_size = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                observed_size += len(chunk)
                digest.update(chunk)
            require(observed_size == binary_size, "staged tarball binary read size mismatch")
            require(digest.hexdigest() == binary_sha, "staged tarball binary SHA256 mismatch")
    except (tarfile.TarError, EOFError) as error:
        raise PreparationError(f"staged tarball cannot be read: {error}") from error
    return {
        "manifest_path": manifest_path,
        "document": document,
        "selected": copy.deepcopy(selected),
        "resolved_refs": resolved_refs,
        "resolved_paths": resolved_paths,
        "archive_path": archive_path,
        "binary_sha256": binary_sha,
        "binary_size_bytes": binary_size,
    }


def bind_staged_asset(
    root: Path,
    *,
    staged_assets_manifest: Path,
    hardware_id: str,
    spec: BackendSpec,
) -> Path:
    require(
        hardware_id.strip() == hardware_id and hardware_id,
        "hardware id must be non-empty and trimmed",
    )
    receipt_path = root / BUILD_RECEIPT_REL
    candidate_binary = root / BUILD_BINARY_REL
    copied_manifest = root / STAGED_MANIFEST_REL
    require(not receipt_path.exists(), f"candidate build receipt already exists: {receipt_path}")
    require(not candidate_binary.exists(), f"candidate binary already exists: {candidate_binary}")
    require(not copied_manifest.exists(), f"staged assets manifest already exists: {copied_manifest}")
    before = source_observation(spec)
    validated = validate_staged_asset_input(
        staged_assets_manifest,
        backend=spec.backend,
        expected_source=before,
    )
    copied_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(validated["manifest_path"], copied_manifest)
    require(
        matrix.file_sha256(copied_manifest)
        == matrix.file_sha256(validated["manifest_path"]),
        "copied staged assets manifest differs from its source",
    )
    metadata_refs: dict[str, dict[str, str]] = {}
    for name in (
        "artifact_manifest",
        "sha256_file",
        "version_manifest",
        "dependency_abi_manifest",
    ):
        source_path = validated["resolved_paths"][name]
        suffix = "".join(source_path.suffixes)
        destination = root / STAGED_METADATA_DIR / f"{name}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        require(
            matrix.file_sha256(destination) == validated["resolved_refs"][name]["sha256"]
            and destination.stat().st_size
            == validated["resolved_refs"][name]["size_bytes"],
            f"copied staged metadata {name} differs from its source",
        )
        metadata_refs[name] = matrix.existing_artifact_ref(
            root,
            destination,
            "staged-metadata",
        )
    with tarfile.open(validated["resolved_paths"]["tarball"], mode="r:*") as archive:
        member = next(
            member
            for member in archive.getmembers()
            if member.name == validated["archive_path"].as_posix()
        )
        stream = archive.extractfile(member)
        require(stream is not None, "staged binary cannot be extracted")
        candidate_binary.parent.mkdir(parents=True, exist_ok=True)
        with candidate_binary.open("xb") as destination:
            shutil.copyfileobj(stream, destination, length=1024 * 1024)
        candidate_binary.chmod(candidate_binary.stat().st_mode | 0o111)
    require(
        candidate_binary.stat().st_size == validated["binary_size_bytes"]
        and matrix.file_sha256(candidate_binary) == validated["binary_sha256"],
        "extracted staged candidate binary identity mismatch",
    )
    after = source_observation(spec)
    require(after == before, "candidate source changed during staged asset binding")
    binary_ref = matrix.existing_artifact_ref(root, candidate_binary, "binary")
    receipt = {
        "schema_version": matrix.SCHEMA_VERSION,
        "artifact_type": matrix.CANDIDATE_BUILD_RECEIPT_TYPE,
        "status": "pass",
        "execution_contract": matrix.G08_EXECUTION_CONTRACT,
        **before,
        "hardware_id": hardware_id,
        "backend": spec.backend,
        "artifact_root": str(root),
        "repository_root": str(REPO_ROOT),
        "build_mode": matrix.STAGED_RELEASE_ASSET_BUILD_MODE,
        "bound_at": matrix.iso_now(),
        "source_observations": {"before": before, "after": after},
        "release_version": "0.8.0",
        "staged_assets_manifest": matrix.existing_artifact_ref(
            root,
            copied_manifest,
            "raw-json",
        ),
        "selected_staged_asset": validated["selected"],
        "staged_metadata_artifacts": metadata_refs,
        "binary_artifact": binary_ref,
        "binary_sha256": binary_ref["sha256"],
    }
    write_json(receipt_path, receipt)
    matrix.validate_candidate_build_receipt(
        root,
        matrix.existing_artifact_ref(root, receipt_path, "raw-json"),
        expected={
            "source_git_sha": before["source_git_sha"],
            "source_tree_sha": before["source_tree_sha"],
            "hardware_id": hardware_id,
            "backend": spec.backend,
            "binary_sha256": binary_ref["sha256"],
            "binary_path": candidate_binary,
        },
        allow_internal_fixture=False,
    )
    print(
        "FERRUM RUNTIME VNEXT STAGED BINARY BOUND "
        f"{spec.backend.upper()}: {receipt_path}"
    )
    return receipt_path


def materialize_exact(path: Path, payload: bytes, label: str) -> None:
    if path.exists():
        require(path.is_file() and path.read_bytes() == payload, f"existing {label} differs from the canonical bytes")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def normalized_model_arg(
    path: Path,
    *,
    weight_format: str,
    weight_files: Mapping[str, str],
) -> Path:
    # A GGUF lane binds one exact quantization file. Passing its snapshot
    # directory would make the product resolver treat that directory as an
    # unknown model source and fall through to a remote repository lookup.
    # Preserve the locked snapshot filename instead of guessing among GGUFs.
    normalized = path.expanduser().absolute()
    if normalized.is_dir() and weight_format.startswith("gguf_"):
        require(
            len(weight_files) == 1,
            "GGUF execution requires exactly one locked weight file",
        )
        relative_path = Path(next(iter(weight_files)))
        require(
            relative_path.suffix.lower() == ".gguf",
            "GGUF execution lock does not name a .gguf file",
        )
        return normalized / relative_path
    return normalized


def self_test_staged_asset_binding(root: Path, spec: BackendSpec) -> None:
    staged_root = root / "staged-assets"
    staged_root.mkdir(parents=True)
    binary_bytes = b"staged ferrum v0.8.0 fixture bytes\n"
    tarball = staged_root / f"ferrum-{spec.backend}.tar.gz"
    archive_path = "package/ferrum"
    with tarfile.open(tarball, mode="w:gz") as archive:
        member = tarfile.TarInfo(archive_path)
        member.mode = 0o755
        member.size = len(binary_bytes)
        archive.addfile(member, io.BytesIO(binary_bytes))

    metadata_paths = {
        "artifact_manifest": staged_root / "artifact-manifest.json",
        "version_manifest": staged_root / "version.json",
        "dependency_abi_manifest": staged_root / "dependency-abi.json",
    }
    write_json(metadata_paths["artifact_manifest"], {"status": "pass"})
    write_json(metadata_paths["version_manifest"], {"version": "0.8.0"})
    write_json(metadata_paths["dependency_abi_manifest"], {"backend": spec.backend})
    sha_path = staged_root / f"{tarball.name}.sha256"
    sha_path.write_text(
        f"{matrix.file_sha256(tarball)}  {tarball.name}\n",
        encoding="utf-8",
    )

    def plain_ref(path: Path) -> dict[str, Any]:
        return {
            "path": path.relative_to(staged_root).as_posix(),
            "sha256": matrix.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }

    source = {
        "source_git_sha": matrix.FROZEN_LEGACY_SHA,
        "source_tree_sha": matrix.frozen_tree_sha(),
        "dirty_status": {"is_dirty": False, "status_short": []},
    }
    artifact_digest = hashlib.sha256(b"workflow artifact fixture").hexdigest()
    selected: dict[str, Any] = {
        "backend": spec.backend,
        "workflow_run_id": 42,
        "artifact": {
            "id": 43,
            "name": f"ferrum-{spec.backend}-v0.8.0",
            "digest": f"sha256:{artifact_digest}",
        },
        "artifact_manifest": plain_ref(metadata_paths["artifact_manifest"]),
        "tarball": plain_ref(tarball),
        "sha256_file": plain_ref(sha_path),
        "version_manifest": plain_ref(metadata_paths["version_manifest"]),
        "dependency_abi_manifest": plain_ref(
            metadata_paths["dependency_abi_manifest"]
        ),
        "binary": {
            "archive_path": archive_path,
            "sha256": hashlib.sha256(binary_bytes).hexdigest(),
            "size_bytes": len(binary_bytes),
        },
    }
    if spec.backend == "cuda":
        selected["target_sm"] = "89"
    document = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_staged_assets_manifest",
        "status": "pass",
        "version": "0.8.0",
        "publish_release": False,
        "release_candidate": {
            "git_sha": source["source_git_sha"],
            "git_tree_sha": source["source_tree_sha"],
            "dirty": False,
        },
        "assets": {
            "cpu": {},
            "metal": copy.deepcopy(selected) if spec.backend == "metal" else {},
            "cuda": copy.deepcopy(selected) if spec.backend == "cuda" else {},
        },
    }
    manifest_path = staged_root / "manifest.json"
    write_json(manifest_path, document)
    validated = validate_staged_asset_input(
        manifest_path,
        backend=spec.backend,
        expected_source=source,
    )
    require(
        validated["binary_sha256"] == hashlib.sha256(binary_bytes).hexdigest(),
        "staged binary self-test identity drifted",
    )

    bound_root = root / f"staged-bound-{spec.backend}"
    bound_binary = bound_root / BUILD_BINARY_REL
    bound_binary.parent.mkdir(parents=True)
    bound_binary.write_bytes(binary_bytes)
    bound_binary.chmod(0o755)
    bound_manifest = bound_root / STAGED_MANIFEST_REL
    bound_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, bound_manifest)
    bound_metadata: dict[str, dict[str, str]] = {}
    for name in (
        "artifact_manifest",
        "sha256_file",
        "version_manifest",
        "dependency_abi_manifest",
    ):
        destination = bound_root / STAGED_METADATA_DIR / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(validated["resolved_paths"][name], destination)
        bound_metadata[name] = matrix.existing_artifact_ref(
            bound_root,
            destination,
            "staged-metadata",
        )
    receipt_path = bound_root / BUILD_RECEIPT_REL
    receipt = {
        "schema_version": matrix.SCHEMA_VERSION,
        "artifact_type": matrix.CANDIDATE_BUILD_RECEIPT_TYPE,
        "status": "pass",
        "execution_contract": matrix.G08_EXECUTION_CONTRACT,
        **source,
        "hardware_id": f"{spec.backend}-staged-fixture",
        "backend": spec.backend,
        "artifact_root": str(bound_root.resolve()),
        "repository_root": str(REPO_ROOT),
        "build_mode": matrix.STAGED_RELEASE_ASSET_BUILD_MODE,
        "bound_at": matrix.iso_now(),
        "source_observations": {"before": source, "after": source},
        "release_version": "0.8.0",
        "staged_assets_manifest": matrix.existing_artifact_ref(
            bound_root,
            bound_manifest,
            "raw-json",
        ),
        "selected_staged_asset": validated["selected"],
        "staged_metadata_artifacts": bound_metadata,
        "binary_artifact": matrix.existing_artifact_ref(
            bound_root,
            bound_binary,
            "binary",
        ),
        "binary_sha256": hashlib.sha256(binary_bytes).hexdigest(),
    }
    write_json(receipt_path, receipt)
    matrix.validate_candidate_build_receipt(
        bound_root,
        matrix.existing_artifact_ref(
            bound_root,
            receipt_path,
            "raw-json",
        ),
        expected={
            "source_git_sha": source["source_git_sha"],
            "source_tree_sha": source["source_tree_sha"],
            "hardware_id": receipt["hardware_id"],
            "backend": spec.backend,
            "binary_sha256": receipt["binary_sha256"],
            "binary_path": bound_binary.resolve(),
        },
        allow_internal_fixture=True,
    )

    def expect_reject(
        name: str,
        mutate: Any,
        marker: str,
    ) -> None:
        hostile = copy.deepcopy(document)
        mutate(hostile)
        hostile_path = staged_root / f"hostile-{name}.json"
        write_json(hostile_path, hostile)
        try:
            validate_staged_asset_input(
                hostile_path,
                backend=spec.backend,
                expected_source=source,
            )
        except PreparationError as error:
            require(
                marker.lower() in str(error).lower(),
                f"staged {name} fixture failed unexpectedly: {error}",
            )
            return
        raise AssertionError(f"staged {name} fixture unexpectedly passed")

    def asset_row(value: dict[str, Any]) -> dict[str, Any]:
        return value["assets"][spec.backend]

    expect_reject(
        "wrong-tar",
        lambda value: asset_row(value)["tarball"].update({"sha256": "0" * 64}),
        "tarball SHA256 mismatch",
    )
    expect_reject(
        "wrong-binary",
        lambda value: asset_row(value)["binary"].update({"sha256": "0" * 64}),
        "binary SHA256 mismatch",
    )
    expect_reject(
        "wrong-source",
        lambda value: value["release_candidate"].update({"git_sha": "0" * 40}),
        "source SHA mismatch",
    )
    expect_reject(
        "wrong-backend",
        lambda value: asset_row(value).update(
            {"backend": "metal" if spec.backend == "cuda" else "cuda"}
        ),
        "selected backend",
    )
    expect_reject(
        "wrong-version",
        lambda value: value.update({"version": "0.8.1"}),
        "version is not 0.8.0",
    )
    expect_reject(
        "dirty-source",
        lambda value: value["release_candidate"].update({"dirty": True}),
        "candidate is dirty",
    )


def prepare_manifest(
    root: Path,
    *,
    model_dir: Path,
    semantic_source_root: Path,
    port: int,
    spec: BackendSpec,
) -> Path:
    require(1 <= port <= 65535, "execution port must be in 1..65535")
    manifest_path = root / EXECUTION_MANIFEST_REL
    require(not manifest_path.exists(), f"execution manifest already exists: {manifest_path}")
    source = source_observation(spec)
    checked_lock = validate_checked_in_lock(spec)
    lock_path = root / MODELS_LOCK_REL
    materialize_exact(lock_path, spec.model_lock_path.read_bytes(), "models.lock")
    build_receipt_path = root / BUILD_RECEIPT_REL
    require(build_receipt_path.is_file(), f"candidate build receipt is missing: {build_receipt_path}")
    build_receipt_ref = matrix.existing_artifact_ref(root, build_receipt_path, "raw-json")
    build_receipt = matrix.require_object(matrix.read_json(build_receipt_path), "candidate build receipt")
    binary_ref = matrix.require_object(build_receipt.get("binary_artifact"), "candidate build binary artifact")
    binary_path = root / matrix.require_string(binary_ref.get("path"), "candidate build binary path")
    binary_sha256 = matrix.require_sha256(build_receipt.get("binary_sha256"), "candidate build binary SHA")
    require(build_receipt.get("source_git_sha") == source["source_git_sha"], "candidate build source SHA is stale")
    require(build_receipt.get("source_tree_sha") == source["source_tree_sha"], "candidate build source tree is stale")
    require(
        build_receipt.get("backend") == spec.backend,
        f"candidate build backend is not {spec.backend.upper()}",
    )
    sources = checked_lock["sources"]
    model_dir = normalized_model_arg(
        model_dir,
        weight_format=sources["weight_format"],
        weight_files=sources["weight_files"],
    )
    semantic_source_root = semantic_source_root.expanduser().resolve()
    effective_path = (
        root
        / f"correctness/{spec.model_key}/{spec.backend}/effective-config.json"
    )
    effective = {
        "schema_version": matrix.SCHEMA_VERSION,
        "execution_contract": matrix.G08_EXECUTION_CONTRACT,
        **source,
        "models_lock_sha256": matrix.file_sha256(lock_path),
        "binary_sha256": binary_sha256,
        "model_key": spec.model_key,
        "backend": spec.backend,
        "model_revision": sources["weight_revision"],
        "model_files": sources["weight_files"],
        "hardware_id": build_receipt["hardware_id"],
        "typed_effective_config": {
            "composition_contract": "resolved-model-plan-vnext",
            "run": copy.deepcopy(spec.typed_run_config),
            "serve": copy.deepcopy(spec.typed_serve_config),
        },
    }
    write_json(effective_path, effective)
    manifest = {
        "schema_version": matrix.SCHEMA_VERSION,
        "execution_contract": matrix.G08_EXECUTION_CONTRACT,
        **source,
        "models_lock_sha256": matrix.file_sha256(lock_path),
        "binary_sha256": binary_sha256,
        "model_key": spec.model_key,
        "backend": spec.backend,
        "model_revision": sources["weight_revision"],
        "model_files": sources["weight_files"],
        "hardware_id": build_receipt["hardware_id"],
        "binary_artifact": binary_ref,
        "binary_build_receipt": build_receipt_ref,
        "models_lock": matrix.existing_artifact_ref(root, lock_path, "raw-json"),
        "effective_config": matrix.existing_artifact_ref(root, effective_path, "raw-json"),
        "execution": {
            "model_arg": str(model_dir),
            "semantic_source_root": str(semantic_source_root),
            "host": "127.0.0.1",
            "port": port,
            "startup_timeout_sec": 900,
            "case_timeout_sec": 900,
            "run_extra_args": list(spec.run_extra_args),
            "serve_extra_args": list(spec.serve_extra_args),
        },
    }
    matrix.validate_execution_manifest(manifest, root, allow_internal_fixture=False)
    write_json(manifest_path, manifest)
    print(f"{spec.manifest_ready_prefix}: {manifest_path}")
    return manifest_path


def self_test(spec: BackendSpec) -> None:
    checked = validate_checked_in_lock(spec)
    require(
        checked["sources"]["weight_format"] == spec.weight_format,
        f"{spec.model_label} format drift",
    )
    require(
        set(spec.probe_commands) == matrix.CANDIDATE_BUILD_PROBES[spec.backend],
        f"canonical {spec.backend.upper()} build probes drift",
    )
    expected_features = (
        "cuda,vllm-moe-marlin,vllm-paged-attn-v2"
        if spec.backend == "cuda"
        else "metal"
    )
    require(
        matrix.CANDIDATE_BUILD_COMMANDS[spec.backend][-2:]
        == ["--features", expected_features],
        f"canonical {spec.backend.upper()} build command drift",
    )
    require(
        ("--gpu-devices" in spec.run_extra_args) is (spec.backend == "cuda"),
        "GPU device selection must remain CUDA-only",
    )
    require(
        "SPHardwareDataType"
        not in spec.probe_commands.get("system_profiler", []),
        "Metal probe must not capture host serial or hardware UUID fields",
    )
    require(
        "--served-model-name" in spec.serve_extra_args,
        "serve model name is missing from the product command",
    )
    with tempfile.TemporaryDirectory(
        prefix=spec.selftest_temp_prefix
    ) as tmp:
        root = Path(tmp)
        if spec.backend == "cuda":
            try:
                matrix.candidate_build_command("cuda")
            except matrix.ScenarioError as error:
                require(
                    "requires a native operator set lock" in str(error),
                    f"missing CUDA lock failed for an unexpected reason: {error}",
                )
            else:
                raise AssertionError("CUDA candidate build accepted a missing native lock")
            source_native_lock = native_operator_set.create_selftest_native_operator_set(
                root / "native-source",
                matrix.CANDIDATE_REQUIRED_CUDA_NATIVE_OPERATORS,
            )
            canonical_native_root = (
                root.resolve() / matrix.CANDIDATE_NATIVE_OPERATOR_SET_LOCK_REL
            ).parent
            native_lock, staged_native = native_operator_set.stage_native_operator_set(
                source_native_lock,
                canonical_native_root,
                matrix.CANDIDATE_REQUIRED_CUDA_NATIVE_OPERATORS,
            )
            require(
                native_lock
                == (root / matrix.CANDIDATE_NATIVE_OPERATOR_SET_LOCK_REL).resolve()
                and len(staged_native["_members"]) > 1,
                "CUDA self-test native operator closure was not staged canonically",
            )
            require(
                matrix.native_operator_set_lock_identity(native_lock)["sha256"]
                == matrix.file_sha256(native_lock),
                "CUDA self-test native operator lock identity drifted",
            )
            broken_native_root = root / "broken-native"
            shutil.copytree(native_lock.parent, broken_native_root)
            broken_native_lock = broken_native_root / native_operator_set.LOCK_FILE_NAME
            broken_member = Path(staged_native["_members"][0]["path"])
            broken_native_root.joinpath(*broken_member.parts).unlink()
            try:
                native_operator_set.validate_native_operator_set(
                    broken_native_lock,
                    matrix.CANDIDATE_REQUIRED_CUDA_NATIVE_OPERATORS,
                )
            except native_operator_set.NativeOperatorSetEvidenceError as error:
                require(
                    "missing" in str(error).lower(),
                    f"missing staged CUDA member failed ambiguously: {error}",
                )
            else:
                raise AssertionError(
                    "CUDA native operator closure accepted a missing staged member"
                )
            candidate_command = matrix.candidate_build_command("cuda", native_lock)
            require(
                candidate_command[:2]
                == [
                    "env",
                    f"FERRUM_NATIVE_OPERATOR_SET_LOCK={native_lock}",
                ]
                and candidate_command[2:] == matrix.CANDIDATE_BUILD_COMMANDS["cuda"],
                "CUDA candidate build command did not bind its explicit native lock",
            )
        else:
            require(
                matrix.candidate_build_command("metal")
                == matrix.CANDIDATE_BUILD_COMMANDS["metal"],
                "Metal candidate build command drifted",
            )
        blob = root / "blobs" / ("a" * 64)
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"locked GGUF bytes\n")
        snapshot_file = root / "snapshots/revision/Model-Q4_K_S.gguf"
        snapshot_file.parent.mkdir(parents=True)
        snapshot_file.symlink_to(blob)
        normalized_snapshot = normalized_model_arg(
            snapshot_file,
            weight_format=spec.weight_format,
            weight_files={"Model-Q4_K_S.gguf": matrix.file_sha256(blob)},
        )
        require(
            normalized_snapshot.name == snapshot_file.name
            and normalized_snapshot.is_symlink(),
            "model argument normalization resolved away the HF snapshot filename",
        )
        normalized_snapshot_root = normalized_model_arg(
            snapshot_file.parent,
            weight_format="gguf_q4_k_s",
            weight_files={"Model-Q4_K_S.gguf": matrix.file_sha256(blob)},
        )
        require(
            normalized_snapshot_root == snapshot_file.absolute()
            and normalized_snapshot_root.is_symlink(),
            "GGUF snapshot directory did not bind the exact locked weight file",
        )
        manifest = matrix.make_execution_fixture_manifest(root)
        lock_path = root / manifest["models_lock"]["path"]
        lock = matrix.read_json(lock_path)
        fixture_model = lock["models"][0]
        fixture_lane = fixture_model["lanes"].pop("cuda")
        fixture_model["lanes"][spec.backend] = fixture_lane
        write_json(lock_path, lock)
        manifest["backend"] = spec.backend
        manifest["hardware_id"] = f"{spec.backend}-fixture"
        manifest["models_lock_sha256"] = matrix.file_sha256(lock_path)
        manifest["models_lock"] = matrix.existing_artifact_ref(
            root,
            lock_path,
            "raw-json",
        )
        manifest["execution_contract"] = matrix.G08_EXECUTION_CONTRACT
        effective_path = root / manifest["effective_config"]["path"]
        effective = matrix.read_json(effective_path)
        effective["backend"] = spec.backend
        effective["hardware_id"] = manifest["hardware_id"]
        effective["models_lock_sha256"] = manifest["models_lock_sha256"]
        effective["execution_contract"] = matrix.G08_EXECUTION_CONTRACT
        effective["typed_effective_config"] = {
            "run": copy.deepcopy(spec.typed_run_config),
            "serve": copy.deepcopy(spec.typed_serve_config),
        }
        write_json(effective_path, effective)
        manifest["effective_config"] = matrix.existing_artifact_ref(root, effective_path, "raw-json")
        manifest["execution"]["run_extra_args"] = list(spec.run_extra_args)
        manifest["execution"]["serve_extra_args"] = list(spec.serve_extra_args)
        manifest["binary_build_receipt"] = matrix.make_candidate_build_receipt_fixture(root, manifest)
        validated = matrix.validate_execution_manifest(manifest, root, allow_internal_fixture=True)
        require(validated["build_receipt_path"] is not None, "candidate build receipt was not validated")
        hostile = copy.deepcopy(manifest)
        hostile_receipt = matrix.read_json(root / hostile["binary_build_receipt"]["path"])
        hostile_receipt["binary_sha256"] = "0" * 64
        hostile_path = root / "build/candidate/hostile-receipt.json"
        write_json(hostile_path, hostile_receipt)
        hostile["binary_build_receipt"] = matrix.existing_artifact_ref(root, hostile_path, "raw-json")
        try:
            matrix.validate_execution_manifest(hostile, root, allow_internal_fixture=True)
        except matrix.ScenarioError as error:
            require("binary SHA mismatch" in str(error), f"hostile receipt failed for an unexpected reason: {error}")
        else:
            raise AssertionError("candidate build receipt accepted a changed binary SHA")
        self_test_staged_asset_binding(root, spec)
    print(spec.prepare_selftest_pass_line)


def parse_args(
    *,
    default_backend: str = "cuda",
    fixed_backend: bool = False,
    backend_specs: Mapping[str, BackendSpec] = BACKEND_SPECS,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    if fixed_backend:
        parser.set_defaults(backend=default_backend)
    else:
        parser.add_argument(
            "--backend",
            choices=tuple(backend_specs),
            default=default_backend,
        )
    subparsers = parser.add_subparsers(dest="command")
    build_parser = subparsers.add_parser(
        "build",
        help="build and bind the current accelerator candidate",
    )
    build_parser.add_argument("--artifact-root", required=True)
    build_parser.add_argument("--hardware-id", required=True)
    build_parser.add_argument("--native-operator-set-lock", type=Path)
    bind_parser = subparsers.add_parser(
        "bind-correctness",
        help="import and bind a cache-only CUDA correctness binary",
    )
    bind_parser.add_argument("--artifact-root", required=True)
    bind_parser.add_argument("--correctness-build-manifest", required=True)
    bind_parser.add_argument("--hardware-id", required=True)
    staged_parser = subparsers.add_parser(
        "bind-staged",
        help="extract and bind the exact binary from a frozen staged release tarball",
    )
    staged_parser.add_argument("--artifact-root", required=True)
    staged_parser.add_argument("--staged-assets-manifest", required=True)
    staged_parser.add_argument("--hardware-id", required=True)
    manifest_parser = subparsers.add_parser("manifest", help="validate model snapshots and write the execution manifest")
    manifest_parser.add_argument("--artifact-root", required=True)
    manifest_parser.add_argument("--model-dir", required=True)
    manifest_parser.add_argument("--semantic-source-root", required=True)
    manifest_parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    require(
        args.self_test or args.command is not None,
        "choose --self-test, build, bind-correctness, bind-staged, or manifest",
    )
    require(not (args.self_test and args.command is not None), "--self-test cannot be combined with a command")
    return args


def main(
    *,
    default_backend: str = "cuda",
    fixed_backend: bool = False,
    backend_specs: Mapping[str, BackendSpec] = BACKEND_SPECS,
    error_label: str = "runtime_vnext_g08b_matrix_prepare",
) -> int:
    try:
        args = parse_args(
            default_backend=default_backend,
            fixed_backend=fixed_backend,
            backend_specs=backend_specs,
        )
        spec = backend_specs[args.backend]
        if args.self_test:
            self_test(spec)
        elif args.command == "build":
            build_candidate(
                artifact_root(args.artifact_root),
                args.hardware_id,
                spec,
                args.native_operator_set_lock,
            )
        elif args.command == "bind-correctness":
            bind_cuda_correctness_artifact(
                artifact_root(args.artifact_root),
                correctness_build_manifest=Path(args.correctness_build_manifest),
                hardware_id=args.hardware_id,
                spec=spec,
            )
        elif args.command == "bind-staged":
            bind_staged_asset(
                artifact_root(args.artifact_root),
                staged_assets_manifest=Path(args.staged_assets_manifest),
                hardware_id=args.hardware_id,
                spec=spec,
            )
        elif args.command == "manifest":
            prepare_manifest(
                artifact_root(args.artifact_root),
                model_dir=Path(args.model_dir),
                semantic_source_root=Path(args.semantic_source_root),
                port=args.port,
                spec=spec,
            )
        else:
            raise PreparationError(f"unsupported command: {args.command}")
    except (
        PreparationError,
        matrix.ScenarioError,
        native_operator_set.NativeOperatorSetEvidenceError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"{error_label}: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
