#!/usr/bin/env python3
"""Prepare G08B model-matrix inputs; the path remains CUDA-named for compatibility."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import bounded_command
import runtime_vnext_baseline_scenarios as matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_REPO_PATH = SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()
MODEL_KEY = "m2-qwen35-35b-a3b"
SERVED_MODEL_NAME = "m2-qwen35-35b-a3b"
BUILD_DIR = Path("build/candidate")
BUILD_RECEIPT_REL = BUILD_DIR / "candidate-build-receipt.json"
BUILD_BINARY_REL = BUILD_DIR / "ferrum"
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


CUDA_SPEC = BackendSpec(
    backend="cuda",
    model_lock_path=SCRIPT_PATH.parent
    / "configs/runtime_vnext_g08b_m2_cuda.models.lock.json",
    lock_id="runtime-vnext-g08b-m2-cuda-v1",
    weight_revision="3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b",
    weight_format="gptq_int4_moe_wna16",
    weight_file_count=19,
    semantic_file_count=5,
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
        "served_model_name": SERVED_MODEL_NAME,
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
        SERVED_MODEL_NAME,
    ),
)

METAL_SPEC = BackendSpec(
    backend="metal",
    model_lock_path=SCRIPT_PATH.parent
    / "configs/runtime_vnext_g08b_m2_metal.models.lock.json",
    lock_id="runtime-vnext-g08b-m2-metal-v1",
    weight_revision="bc014a17be43adabd7066b7a86075ff935c6a4e2",
    weight_format="gguf_q4_k_s",
    weight_file_count=1,
    semantic_file_count=5,
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
        "served_model_name": SERVED_MODEL_NAME,
    },
    run_extra_args=(
        "--gpu-memory-utilization",
        "0.90",
    ),
    serve_extra_args=(
        "--gpu-memory-utilization",
        "0.90",
        "--served-model-name",
        SERVED_MODEL_NAME,
    ),
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


def source_observation() -> dict[str, Any]:
    status = run_text(["git", "status", "--short", "--untracked-files=all"])
    require(not status, "candidate preparation requires a clean worktree")
    source_git_sha = run_text(["git", "rev-parse", "HEAD"])
    source_tree_sha = run_text(["git", "rev-parse", "HEAD^{tree}"])
    require(matrix.GIT_SHA_RE.fullmatch(source_git_sha) is not None, "candidate source SHA is invalid")
    require(matrix.GIT_SHA_RE.fullmatch(source_tree_sha) is not None, "candidate source tree SHA is invalid")
    require(
        run_text(["git", "cat-file", "-e", f"{source_git_sha}:{SCRIPT_REPO_PATH}"]) == "",
        "candidate prepare script is not checked in at the source SHA",
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
        f"G08B M2 {spec.backend.upper()} model lock",
    )
    require(document.get("schema_version") == matrix.SCHEMA_VERSION, "G08B model lock schema mismatch")
    require(document.get("lock_id") == spec.lock_id, "G08B model lock id mismatch")
    sources = matrix.locked_execution_sources(document, MODEL_KEY, spec.backend)
    require(
        sources["weight_revision"] == spec.weight_revision,
        "G08B weight revision drift",
    )
    require(
        sources["weight_format"] == spec.weight_format,
        "G08B weight format drift",
    )
    require(
        len(sources["weight_files"]) == spec.weight_file_count,
        f"G08B weight lock must contain exactly {spec.weight_file_count} files",
    )
    require(
        len(sources["semantic_source"]["files"]) == spec.semantic_file_count,
        f"G08B semantic lock must contain exactly {spec.semantic_file_count} files",
    )
    return {"document": document, "sources": sources}


def build_candidate(root: Path, hardware_id: str, spec: BackendSpec) -> Path:
    require(hardware_id.strip() == hardware_id and hardware_id, "hardware id must be non-empty and trimmed")
    receipt_path = root / BUILD_RECEIPT_REL
    binary_path = root / BUILD_BINARY_REL
    require(not receipt_path.exists(), f"candidate build receipt already exists: {receipt_path}")
    require(not binary_path.exists(), f"candidate build binary already exists: {binary_path}")
    before = source_observation()
    probes = {
        name: capture_probe(root, name, argv)
        for name, argv in spec.probe_commands.items()
    }
    bounded_path = root / BUILD_DIR / "bounded-command-receipt.json"
    stdout_path = root / BUILD_DIR / "stdout.log"
    stderr_path = root / BUILD_DIR / "stderr.log"
    wrapper_rc, bounded_receipt = bounded_command.run_bounded_command(
        command=matrix.CANDIDATE_BUILD_COMMANDS[spec.backend],
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
    after = source_observation()
    require(
        after == before,
        f"candidate source changed during the {spec.backend.upper()} build",
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
        "command": matrix.CANDIDATE_BUILD_COMMANDS[spec.backend],
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


def materialize_exact(path: Path, payload: bytes, label: str) -> None:
    if path.exists():
        require(path.is_file() and path.read_bytes() == payload, f"existing {label} differs from the canonical bytes")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


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
    source = source_observation()
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
    model_dir = model_dir.expanduser().resolve()
    semantic_source_root = semantic_source_root.expanduser().resolve()
    sources = checked_lock["sources"]
    effective_path = (
        root / f"correctness/{MODEL_KEY}/{spec.backend}/effective-config.json"
    )
    effective = {
        "schema_version": matrix.SCHEMA_VERSION,
        "execution_contract": matrix.G08_EXECUTION_CONTRACT,
        **source,
        "models_lock_sha256": matrix.file_sha256(lock_path),
        "binary_sha256": binary_sha256,
        "model_key": MODEL_KEY,
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
        "model_key": MODEL_KEY,
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
        "G08B format drift",
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
        prefix=f"ferrum-g08b-{spec.backend}-prepare-"
    ) as tmp:
        root = Path(tmp)
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
    print(spec.prepare_selftest_pass_line)


def parse_args(
    *,
    default_backend: str = "cuda",
    fixed_backend: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    if fixed_backend:
        parser.set_defaults(backend=default_backend)
    else:
        parser.add_argument(
            "--backend",
            choices=tuple(BACKEND_SPECS),
            default=default_backend,
        )
    subparsers = parser.add_subparsers(dest="command")
    build_parser = subparsers.add_parser(
        "build",
        help="build and bind the current accelerator candidate",
    )
    build_parser.add_argument("--artifact-root", required=True)
    build_parser.add_argument("--hardware-id", required=True)
    manifest_parser = subparsers.add_parser("manifest", help="validate model snapshots and write the execution manifest")
    manifest_parser.add_argument("--artifact-root", required=True)
    manifest_parser.add_argument("--model-dir", required=True)
    manifest_parser.add_argument("--semantic-source-root", required=True)
    manifest_parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    require(args.self_test or args.command is not None, "choose --self-test, build, or manifest")
    require(not (args.self_test and args.command is not None), "--self-test cannot be combined with a command")
    return args


def main(
    *,
    default_backend: str = "cuda",
    fixed_backend: bool = False,
) -> int:
    try:
        args = parse_args(
            default_backend=default_backend,
            fixed_backend=fixed_backend,
        )
        spec = BACKEND_SPECS[args.backend]
        if args.self_test:
            self_test(spec)
        elif args.command == "build":
            build_candidate(
                artifact_root(args.artifact_root),
                args.hardware_id,
                spec,
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
    except (PreparationError, matrix.ScenarioError, OSError, subprocess.SubprocessError) as error:
        print(f"runtime_vnext_g08b_matrix_prepare: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
