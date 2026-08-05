#!/usr/bin/env python3
"""Build a CUDA diagnostic binary and validate its product execution-plan trace.

Binary readiness and semantic trace acceptance are intentionally separate. The
build mode cannot print a PASS line; only validation of actual `ferrum serve`
evidence can do that.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import bounded_command
import runtime_vnext_native_operator_set as native_operator_set
import runtime_vnext_plan_reference as plan_reference

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 on retained CUDA build hosts.
    tomllib = None


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "cuda-correctness"
FEATURES = ",".join(plan_reference.FEATURES)
READY_PREFIX = "FERRUM CUDA CORRECTNESS BINARY READY"
PLAN_READY_PREFIX = "FERRUM CUDA CORRECTNESS IMPORT INVENTORY READY"
SEMANTIC_PASS_PREFIX = "FERRUM CUDA CORRECTNESS SEMANTIC TRACE PASS"
SELFTEST_PASS_LINE = "FERRUM CUDA CORRECTNESS BUILD SELFTEST PASS"
SCHEMA_VERSION = 6
SEMANTIC_SCHEMA_VERSION = 6
REFERENCE_ARTIFACT_TYPE = plan_reference.ARTIFACT_TYPE
CORRECTNESS_ARTIFACT_TYPE = "runtime-vnext-cuda-correctness-binary"
REFERENCE_MODEL_KEY = plan_reference.MODEL_KEY
REFERENCE_CASE_ID = plan_reference.CASE_ID
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CORE_PTX_BLOCK_RE = re.compile(
    r"const CORE_PTX_KERNELS:\s*&\[&str\]\s*=\s*&\[(?P<body>.*?)\];",
    re.DOTALL,
)
QUOTED_PATH_RE = re.compile(r'"([^"\r\n]+)"')
CACHE_EVENT_RE = re.compile(
    r"\[cuda-native-build-cache\]\s+artifact=(?P<artifact>\S+)\s+"
    r"status=(?P<status>published|cache_hit|imported)\s+"
    r"sha256=(?P<sha256>[0-9a-f]{64})"
)
BUILD_SUMMARY_RE = re.compile(
    r"\[cuda-build-summary\]\s+artifact=(?P<artifact>\S+)\s+"
    r"status=(?P<status>\S+)\s+reason=(?P<reason>\S+)"
)
FORBIDDEN_OVERRIDE_PREFIXES = (
    "CARGO_PROFILE_CUDA_CORRECTNESS_",
    "CARGO_PROFILE_RELEASE_",
)
FORBIDDEN_OVERRIDE_KEYS = (
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
)
UNMANAGED_CUDA_BUILD_PACKAGES = ("candle-kernels",)


class CorrectnessBuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectnessBuildError(message)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_file_identity(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ),
        f"file changed while it was inventoried: {path}",
    )
    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
    }


def file_ref(path: Path, artifact_root: Path) -> dict[str, Any]:
    resolved_root = artifact_root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise CorrectnessBuildError(
            f"artifact file is outside its root: {resolved_path}"
        ) from error
    require(
        path.is_file() and not path.is_symlink() and resolved_path.is_file(),
        f"artifact file is not regular: {resolved_path}",
    )
    return {
        "path": relative.as_posix(),
        "sha256": sha256(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
    }


def resolve_file_ref(
    artifact_root: Path,
    raw: Any,
    label: str,
) -> Path:
    require(isinstance(raw, dict), f"{label} reference must be an object")
    relative = raw.get("path")
    require(
        isinstance(relative, str)
        and relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        f"{label} reference path is invalid",
    )
    path = (artifact_root / relative).resolve()
    try:
        path.relative_to(artifact_root.resolve())
    except ValueError as error:
        raise CorrectnessBuildError(
            f"{label} reference escapes its artifact root"
        ) from error
    require(path.is_file() and not path.is_symlink(), f"{label} file is missing: {path}")
    require(
        raw.get("sha256") == sha256(path)
        and raw.get("size_bytes", path.stat().st_size) == path.stat().st_size,
        f"{label} reference identity mismatch",
    )
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_text(source_root: Path, argv: Sequence[str], timeout: float = 30.0) -> str:
    result = subprocess.run(
        list(argv),
        cwd=source_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    require(
        result.returncode == 0,
        f"command failed ({result.returncode}): {list(argv)!r}: {result.stderr[-1000:]}",
    )
    return result.stdout.strip()


def command_probe(source_root: Path, argv: Sequence[str]) -> dict[str, Any]:
    executable = shutil.which(argv[0])
    result = subprocess.run(
        list(argv),
        cwd=source_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    combined = (result.stdout + "\n" + result.stderr).strip()
    return {
        "command": list(argv),
        "resolved_executable": executable,
        "returncode": result.returncode,
        "output_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        "output": combined,
    }


def toolchain_probe(source_root: Path) -> dict[str, Any]:
    cuda_home = next(
        (
            Path(value)
            for key in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR")
            if (value := os.environ.get(key))
        ),
        None,
    )
    nvcc = (
        str(cuda_home / "bin/nvcc")
        if cuda_home is not None and (cuda_home / "bin/nvcc").is_file()
        else "nvcc"
    )
    ccbin = os.environ.get("NVCC_CCBIN") or os.environ.get("CC") or "cc"
    cxx = os.environ.get("CXX") or "c++"
    probes = {
        "nvcc": command_probe(source_root, [nvcc, "--version"]),
        "ccbin": command_probe(source_root, [ccbin, "--version"]),
        "cxx": command_probe(source_root, [cxx, "--version"]),
        "ar": command_probe(source_root, ["ar", "--version"]),
        "rustc": command_probe(source_root, ["rustc", "-vV"]),
    }
    require(probes["nvcc"]["returncode"] == 0, "nvcc --version failed")
    require(probes["ar"]["returncode"] == 0, "ar --version failed")
    return {
        "host_platform": platform.platform(),
        "host_machine": platform.machine(),
        "ambient_compiler_env": {
            key: os.environ.get(key)
            for key in ("NVCC_CCBIN", "CC", "CXX")
            if os.environ.get(key)
        },
        "probes": probes,
    }


def source_identity(source_root: Path, *, require_clean: bool) -> dict[str, Any]:
    git_sha = run_text(source_root, ["git", "rev-parse", "HEAD"])
    tree_sha = run_text(source_root, ["git", "rev-parse", "HEAD^{tree}"])
    status = run_text(
        source_root,
        ["git", "status", "--short", "--untracked-files=all"],
    ).splitlines()
    require(GIT_SHA_RE.fullmatch(git_sha) is not None, "source git SHA is invalid")
    require(GIT_SHA_RE.fullmatch(tree_sha) is not None, "source tree SHA is invalid")
    if require_clean:
        require(not status, f"CUDA correctness build requires a clean source tree: {status}")
    return {
        "source_git_sha": git_sha,
        "source_tree_sha": tree_sha,
        "dirty_status": {
            "is_dirty": bool(status),
            "status_short": status,
        },
    }


def parse_flat_toml_table(document: str, table_name: str) -> dict[str, Any]:
    current_table = None
    result = {}
    for line_number, raw_line in enumerate(document.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_table = line[1:-1].strip()
            continue
        if current_table != table_name:
            continue
        require("=" in line, f"invalid {table_name} TOML line {line_number}")
        key, raw_value = (part.strip() for part in line.split("=", 1))
        require(
            re.fullmatch(r"[A-Za-z0-9_-]+", key) is not None,
            f"invalid {table_name} key at line {line_number}",
        )
        if raw_value in {"true", "false"}:
            value: Any = raw_value == "true"
        elif re.fullmatch(r"-?[0-9]+", raw_value):
            value = int(raw_value)
        elif raw_value.startswith('"') and raw_value.endswith('"'):
            value = json.loads(raw_value)
        else:
            raise CorrectnessBuildError(
                f"unsupported {table_name} scalar at line {line_number}: {raw_value!r}"
            )
        require(key not in result, f"duplicate {table_name} key: {key}")
        result[key] = value
    require(result, f"Cargo table [{table_name}] is missing or empty")
    return result


def load_profile_tables(document: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if tomllib is not None:
        parsed = tomllib.loads(document)
        profile_root = parsed.get("profile", {})
        return (
            profile_root.get(PROFILE),
            profile_root.get("release"),
        )
    return (
        parse_flat_toml_table(document, f"profile.{PROFILE}"),
        parse_flat_toml_table(document, "profile.release"),
    )


def validate_profile(source_root: Path) -> dict[str, Any]:
    cargo_toml = source_root / "Cargo.toml"
    profile, release = load_profile_tables(cargo_toml.read_text(encoding="utf-8"))
    require(isinstance(profile, dict), f"Cargo profile {PROFILE!r} is missing")
    expected = {
        "inherits": "release",
        "lto": False,
        "codegen-units": 16,
        "incremental": True,
        "strip": False,
    }
    require(profile == expected, f"Cargo profile {PROFILE!r} drift: {profile!r}")
    require(isinstance(release, dict), "Cargo release profile is missing")
    require(release.get("opt-level") == 3, "release opt-level must remain 3")
    return {
        "name": PROFILE,
        "settings": profile,
        "inherited_opt_level": release["opt-level"],
        "semantic_delta_from_release": [
            "lto",
            "codegen-units",
            "incremental",
            "strip",
        ],
    }


def core_ptx_outputs(source_root: Path) -> list[tuple[str, str, str]]:
    build_rs = source_root / "crates/ferrum-kernels/build.rs"
    source = build_rs.read_text(encoding="utf-8")
    match = CORE_PTX_BLOCK_RE.search(source)
    require(match is not None, "cannot locate CORE_PTX_KERNELS in ferrum-kernels/build.rs")
    paths = QUOTED_PATH_RE.findall(match.group("body"))
    require(paths, "CORE_PTX_KERNELS is empty")
    outputs = []
    seen = set()
    for path in paths:
        stem = Path(path).stem
        require(stem not in seen, f"duplicate core PTX output stem: {stem}")
        seen.add(stem)
        outputs.append(
            (
                f"core_ptx.{stem}",
                f"{stem}.ptx",
                f"{stem}.ptx.stamp",
            )
        )
    return outputs


def discover_import_dirs(roots: Sequence[Path]) -> list[Path]:
    discovered = set()
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        require(root.is_dir(), f"native import target root is not a directory: {root}")
        candidates = [root] if root.name == "out" else []
        candidates.extend(root.glob("*/build/ferrum-kernels-*/out"))
        candidates.extend(root.glob("build/ferrum-kernels-*/out"))
        for candidate in candidates:
            if candidate.is_dir():
                discovered.add(candidate.resolve())
    return sorted(discovered)


def inventory_imports(
    source_root: Path,
    import_dirs: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    requirements = core_ptx_outputs(source_root)
    inventory = []
    missing = []
    for artifact_id, file_name, stamp_name in requirements:
        providers = []
        for directory in import_dirs:
            artifact = directory / file_name
            stamp = directory / stamp_name
            if artifact.is_file() and stamp.is_file():
                artifact_identity = stable_file_identity(artifact)
                stamp_identity = stable_file_identity(stamp)
                providers.append(
                    {
                        "out_dir": str(directory),
                        "artifact": artifact_identity,
                        "stamp": stamp_identity,
                    }
                )
        if not providers:
            missing.append(artifact_id)
        inventory.append(
            {
                "artifact_id": artifact_id,
                "file_name": file_name,
                "stamp_file_name": stamp_name,
                "providers": providers,
            }
        )
    return inventory, missing


def reject_hidden_overrides() -> None:
    rejected = []
    for key, value in os.environ.items():
        if not value:
            continue
        if key in FORBIDDEN_OVERRIDE_KEYS or key.startswith(FORBIDDEN_OVERRIDE_PREFIXES):
            rejected.append(key)
    require(
        not rejected,
        f"hidden compiler/profile overrides are forbidden: {sorted(rejected)}",
    )


def make_build_command(
    *,
    target_dir: Path,
    native_cache: Path,
    import_dirs: Sequence[Path],
    native_operator_set_lock: Path,
    compute_capability: str,
    cargo_jobs: int,
    nvcc_threads: int,
) -> list[str]:
    return [
        "env",
        f"CARGO_TARGET_DIR={target_dir}",
        f"CARGO_BUILD_JOBS={cargo_jobs}",
        f"CUDA_COMPUTE_CAP={compute_capability}",
        f"FERRUM_NVCC_THREADS={nvcc_threads}",
        f"FERRUM_CUDA_NATIVE_BUILD_CACHE={native_cache}",
        f"FERRUM_CUDA_NATIVE_IMPORT_DIRS={os.pathsep.join(map(str, import_dirs))}",
        "FERRUM_CUDA_NATIVE_SOURCE_POLICY=cache-only",
        f"FERRUM_NATIVE_OPERATOR_SET_LOCK={native_operator_set_lock}",
        "cargo",
        "build",
        "--profile",
        PROFILE,
        "-p",
        "ferrum-cli",
        "--bin",
        "ferrum",
        "--features",
        FEATURES,
        "--message-format=json-render-diagnostics",
        "-vv",
    ]


def parse_native_build_log(log: str) -> dict[str, Any]:
    cache_events = [
        {
            "artifact_id": match.group("artifact"),
            "status": match.group("status"),
            "sha256": match.group("sha256"),
        }
        for match in CACHE_EVENT_RE.finditer(log)
    ]
    summaries = [
        {
            "artifact": match.group("artifact"),
            "status": match.group("status"),
            "reason": match.group("reason"),
        }
        for match in BUILD_SUMMARY_RE.finditer(log)
    ]
    rebuilt = [
        row
        for row in summaries
        if row["status"] == "built"
        and (
            row["artifact"].startswith("core-ptx:")
            or row["artifact"]
            in {"marlin", "vllm_marlin", "vllm_moe_marlin", "vllm_paged_attn"}
        )
    ]
    compiler_lines = [
        line
        for line in log.splitlines()
        if re.search(r"\[(?:vllm-marlin|vllm-moe-marlin|vllm-paged-attn-v2)\]\s+compiling", line)
    ]
    rejected = [row for row in summaries if row["status"] == "rejected"]
    unmanaged_build_scripts = []
    direct_nvcc_invocations = []
    for line in log.splitlines():
        stripped = line.strip()
        if "Running `" not in stripped:
            continue
        if re.search(r"(?:^|[\s/])nvcc(?:\s|`|$)", stripped):
            direct_nvcc_invocations.append(stripped)
        if not stripped.endswith("/build-script-build`"):
            continue
        for package in UNMANAGED_CUDA_BUILD_PACKAGES:
            if (
                f"CARGO_PKG_NAME={package}" in stripped
                or f"CARGO_PKG_NAME='{package}'" in stripped
            ):
                unmanaged_build_scripts.append(
                    {"package": package, "command": stripped}
                )
                break
    managed_recompile_count = len(rebuilt) + len(compiler_lines)
    unmanaged_recompile_count = len(unmanaged_build_scripts) + len(
        direct_nvcc_invocations
    )
    return {
        "cache_events": cache_events,
        "build_summaries": summaries,
        "rebuilt_native_artifacts": rebuilt,
        "rejected_native_artifacts": rejected,
        "native_compiler_lines": compiler_lines,
        "unmanaged_native_build_scripts": unmanaged_build_scripts,
        "direct_nvcc_invocations": direct_nvcc_invocations,
        "managed_native_recompile_count": managed_recompile_count,
        "unmanaged_native_recompile_count": unmanaged_recompile_count,
        "native_recompile_count": managed_recompile_count
        + unmanaged_recompile_count,
    }


def parse_ferrum_compiler_artifacts(log: str) -> list[dict[str, Any]]:
    artifacts = []
    for raw_line in log.splitlines():
        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact":
            continue
        target = message.get("target")
        if not isinstance(target, dict) or target.get("name") != "ferrum":
            continue
        kinds = target.get("kind")
        if not isinstance(kinds, list) or "bin" not in kinds:
            continue
        executable = message.get("executable")
        if not isinstance(executable, str):
            continue
        artifacts.append(
            {
                "executable": executable,
                "fresh": message.get("fresh"),
                "target_name": target.get("name"),
                "target_kind": kinds,
                "profile": message.get("profile"),
            }
        )
    return artifacts


def required_native_artifact_ids(source_root: Path) -> set[str]:
    return {
        artifact_id
        for artifact_id, _, _ in core_ptx_outputs(source_root)
    }


def prepare_target_for_verified_build(target_dir: Path) -> dict[str, Any]:
    binary = target_dir / PROFILE / "ferrum"
    removed = []
    if binary.exists():
        binary.unlink()
        removed.append(str(binary))
    build_root = target_dir / PROFILE / "build"
    if build_root.is_dir():
        for path in sorted(build_root.glob("ferrum-kernels-*")):
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path))
    return {
        "removed_outputs": removed,
        "forced_ferrum_relink": True,
        "forced_ferrum_kernels_build_script": True,
    }


def artifact_root(raw: Path, source_root: Path) -> Path:
    root = raw.expanduser().resolve()
    try:
        root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise CorrectnessBuildError("artifact root must be outside the source tree")
    root.mkdir(parents=True, exist_ok=True)
    return root


def import_release_plan_reference(
    args: argparse.Namespace,
    root: Path,
    *,
    source_root: Path,
    current_identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        reference, imported_manifest = plan_reference.copy_validated(
            source_manifest=args.reference_plan_manifest,
            destination_root=root / "release-plan-reference",
            expected_hardware_id=args.hardware_id,
            candidate_source_git_sha=current_identity["source_git_sha"],
            candidate_source_tree_sha=current_identity["source_tree_sha"],
            repository_root=source_root,
        )
    except plan_reference.PlanReferenceError as error:
        raise CorrectnessBuildError(str(error)) from error
    return reference, file_ref(imported_manifest, root)


def create_plan(
    args: argparse.Namespace,
    root: Path,
    *,
    require_clean: bool,
) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve()
    require((source_root / "Cargo.toml").is_file(), f"invalid source root: {source_root}")
    reject_hidden_overrides()
    identity = source_identity(source_root, require_clean=require_clean)
    reference, reference_ref = import_release_plan_reference(
        args,
        root,
        source_root=source_root,
        current_identity=identity,
    )
    profile = validate_profile(source_root)
    toolchain = toolchain_probe(source_root)
    import_dirs = discover_import_dirs(args.import_target_root)
    inventory, missing = inventory_imports(source_root, import_dirs)
    native_cache = args.native_cache.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    source_native_operator_set_lock = (
        args.native_operator_set_lock.expanduser().resolve()
    )
    require(native_cache.is_absolute(), "native cache must be absolute")
    require(target_dir.is_absolute(), "Cargo target directory must be absolute")
    require(
        source_native_operator_set_lock.is_file()
        and not source_native_operator_set_lock.is_symlink(),
        "native operator set lock must be a regular file: "
        f"{source_native_operator_set_lock}",
    )
    native_operator_set_lock, _ = native_operator_set.stage_native_operator_set(
        source_native_operator_set_lock,
        root / "native-operator-set",
        plan_reference.REQUIRED_CUDA_NATIVE_OPERATORS,
    )
    native_operator_set_identity = stable_file_identity(native_operator_set_lock)
    native_operator_set_artifact = file_ref(native_operator_set_lock, root)
    require(
        re.fullmatch(r"[0-9]{2,3}", args.compute_capability) is not None,
        "compute capability must be numeric, for example 89",
    )
    require(1 <= args.cargo_jobs <= 16, "cargo jobs must be in [1, 16]")
    command = make_build_command(
        target_dir=target_dir,
        native_cache=native_cache,
        import_dirs=import_dirs,
        native_operator_set_lock=native_operator_set_lock,
        compute_capability=args.compute_capability,
        cargo_jobs=args.cargo_jobs,
        nvcc_threads=args.nvcc_threads,
    )
    source_inputs = {}
    for relative in (
        "Cargo.toml",
        "crates/ferrum-kernels/build.rs",
        "crates/ferrum-native-ops/src/build_cache.rs",
        "scripts/release/runtime_vnext_cuda_correctness_build.py",
        "scripts/release/runtime_vnext_native_operator_set.py",
    ):
        path = source_root / relative
        require(path.is_file(), f"required correctness-build source is missing: {relative}")
        source_inputs[relative] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime-vnext-cuda-correctness-build-plan",
        "status": "import-inventory-ready" if not missing else "import-inventory-incomplete",
        "created_at": now_iso(),
        **identity,
        "profile": profile,
        "features": FEATURES.split(","),
        "compute_capability": args.compute_capability,
        "hardware_id": args.hardware_id,
        "cargo_jobs": args.cargo_jobs,
        "nvcc_threads": args.nvcc_threads,
        "native_source_policy": "cache-only",
        "target_dir": str(target_dir),
        "native_build_cache": str(native_cache),
        "native_import_dirs": [str(path) for path in import_dirs],
        "native_operator_set_lock": native_operator_set_identity,
        "native_operator_set_artifact": native_operator_set_artifact,
        "native_import_inventory": inventory,
        "missing_native_imports": missing,
        "ready": not missing,
        "build_command": command,
        "toolchain": toolchain,
        "source_inputs": source_inputs,
        "hard_deadline_seconds": args.wall_timeout_seconds,
        "progress_signal": "bounded build stdout/stderr log byte growth",
        "stop_condition": "first native rebuild, build failure, deadline, or runnable binary",
        "semantic_plan_contract": {
            "status": "pending-product-load",
            "reference_kind": REFERENCE_ARTIFACT_TYPE,
            "reference_plan_manifest": reference_ref,
            "reference_source_git_sha": reference["source_git_sha"],
            "reference_source_tree_sha": reference["source_tree_sha"],
            "reference_plan_hash": reference["plan_identity"]["plan_hash"],
            "reference_binary_sha256": reference["binary_sha256"],
            "reference_profile": reference["profile"],
            "hardware_id": reference["hardware_id"],
            "model_key": reference["scope"]["model_key"],
            "case_id": reference["scope"]["case_id"],
            "require_exact_match_before_focused_result": True,
        },
        "limitations": [
            "inventory providers are candidates; exact signatures are resolved fail-closed by the build script",
            "binary readiness does not prove product execution-plan equivalence",
        ],
    }


def run_build(args: argparse.Namespace, plan: dict[str, Any], root: Path) -> dict[str, Any]:
    require(plan["ready"], f"native import inventory is incomplete: {plan['missing_native_imports']}")
    source_root = args.source_root.expanduser().resolve()
    target_dir = Path(plan["target_dir"])
    native_cache = Path(plan["native_build_cache"])
    target_dir.mkdir(parents=True, exist_ok=True)
    native_cache.mkdir(parents=True, exist_ok=True)
    target_preparation = prepare_target_for_verified_build(target_dir)
    build_dir = root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    build_receipt = build_dir / "bounded.receipt.json"
    build_stdout = build_dir / "stdout.log"
    build_stderr = build_dir / "stderr.log"
    wrapper_rc, receipt = bounded_command.run_bounded_command(
        command=plan["build_command"],
        cwd=source_root,
        receipt_path=build_receipt,
        stdout_path=build_stdout,
        stderr_path=build_stderr,
        limits=bounded_command.Limits(
            wall_timeout_seconds=float(args.wall_timeout_seconds),
            max_processes=64,
            max_group_threads=256,
            max_per_process_threads=64,
            sample_interval_seconds=0.2,
            max_sampling_errors=3,
            term_grace_seconds=2.0,
        ),
    )
    stdout_text = (
        build_stdout.read_text(encoding="utf-8") if build_stdout.is_file() else ""
    )
    stderr_text = (
        build_stderr.read_text(encoding="utf-8") if build_stderr.is_file() else ""
    )
    native_signal = parse_native_build_log(stdout_text + "\n" + stderr_text)
    write_json(build_dir / "native-build-signal.json", native_signal)
    require(
        wrapper_rc == 0 and receipt.get("status") == "pass" and receipt.get("rc") == 0,
        f"bounded CUDA correctness build failed: {build_receipt}",
    )
    require(
        native_signal["native_recompile_count"] == 0,
        "CUDA correctness build compiled native PTX/TU instead of reusing signed outputs",
    )
    artifact_set_summaries = [
        row
        for row in native_signal["build_summaries"]
        if row["artifact"] == "native_operator_artifact_set"
    ]
    require(
        len(artifact_set_summaries) == 1
        and artifact_set_summaries[0]["status"] == "linked",
        "CUDA correctness build did not resolve and link exactly one native operator artifact set",
    )
    binary = target_dir / PROFILE / "ferrum"
    require(binary.is_file() and os.access(binary, os.X_OK), f"build output is missing: {binary}")
    compiler_artifacts = parse_ferrum_compiler_artifacts(stdout_text)
    current_binary_artifacts = [
        row
        for row in compiler_artifacts
        if Path(row["executable"]).resolve() == binary.resolve()
    ]
    require(
        any(row["fresh"] is False for row in current_binary_artifacts),
        "Cargo did not emit a non-fresh ferrum compiler-artifact for the built executable",
    )
    expected_native = required_native_artifact_ids(source_root)
    observed_native = {
        event["artifact_id"] for event in native_signal["cache_events"]
    }
    missing_native_evidence = sorted(expected_native - observed_native)
    require(
        not missing_native_evidence,
        f"native cache evidence is incomplete: {missing_native_evidence}",
    )

    copied_binary = root / "binary/ferrum"
    copied_binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, copied_binary)
    smoke_dir = root / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_receipt = smoke_dir / "bounded.receipt.json"
    smoke_stdout = smoke_dir / "stdout.log"
    smoke_stderr = smoke_dir / "stderr.log"
    smoke_rc, smoke = bounded_command.run_bounded_command(
        command=[str(copied_binary), "--version"],
        cwd=source_root,
        receipt_path=smoke_receipt,
        stdout_path=smoke_stdout,
        stderr_path=smoke_stderr,
        limits=bounded_command.Limits(
            wall_timeout_seconds=30,
            max_processes=8,
            max_group_threads=32,
            max_per_process_threads=16,
        ),
    )
    require(
        smoke_rc == 0 and smoke.get("status") == "pass" and smoke.get("rc") == 0,
        f"CUDA correctness binary smoke failed: {smoke_receipt}",
    )
    after = source_identity(source_root, require_clean=True)
    require(
        after["source_git_sha"] == plan["source_git_sha"]
        and after["source_tree_sha"] == plan["source_tree_sha"],
        "source identity changed during CUDA correctness build",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": CORRECTNESS_ARTIFACT_TYPE,
        "status": "binary-ready",
        "created_at": now_iso(),
        "source_git_sha": plan["source_git_sha"],
        "source_tree_sha": plan["source_tree_sha"],
        "dirty_status": plan["dirty_status"],
        "profile": plan["profile"],
        "features": plan["features"],
        "compute_capability": plan["compute_capability"],
        "hardware_id": plan["hardware_id"],
        "cargo_jobs": plan["cargo_jobs"],
        "nvcc_threads": plan["nvcc_threads"],
        "native_source_policy": plan["native_source_policy"],
        "target_dir": plan["target_dir"],
        "native_build_cache": plan["native_build_cache"],
        "native_import_dirs": plan["native_import_dirs"],
        "native_operator_set_lock": plan["native_operator_set_lock"],
        "native_operator_set_artifact": plan["native_operator_set_artifact"],
        "source_inputs": plan["source_inputs"],
        "toolchain": plan["toolchain"],
        "target_preparation": target_preparation,
        "compiler_artifacts": current_binary_artifacts,
        "native_build_signal": native_signal,
        "plan": file_ref(root / "plan.json", root),
        "native_build_signal_artifact": file_ref(
            build_dir / "native-build-signal.json", root
        ),
        "semantic_plan_contract": {
            **plan["semantic_plan_contract"],
            "status": "pending-product-trace-validation",
        },
        "binary": file_ref(copied_binary, root),
        "build_receipt": file_ref(build_receipt, root),
        "build_stdout": file_ref(build_stdout, root),
        "build_stderr": file_ref(build_stderr, root),
        "smoke_receipt": file_ref(smoke_receipt, root),
        "smoke_stdout": file_ref(smoke_stdout, root),
        "smoke_stderr": file_ref(smoke_stderr, root),
        "ready_line": f"{READY_PREFIX}: {root}",
        "pass_line": None,
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def read_typed_effective_config(path: Path, label: str) -> dict[str, Any]:
    document = read_json_object(path, label)
    typed = document.get("typed_effective_config")
    require(
        isinstance(typed, dict) and typed,
        f"{label} typed_effective_config must be a non-empty JSON object",
    )
    return typed


def verify_semantic_trace_artifact(
    artifact_root: Path,
    *,
    source_root: Path | None = None,
    verify_checkout: bool = False,
    allow_internal_fixture: bool = False,
) -> dict[str, Any]:
    validation_path = artifact_root.expanduser().resolve()
    if validation_path.name != "validation.json":
        validation_path = validation_path / "validation.json"
    root = validation_path.parent
    validation = read_json_object(validation_path, "semantic validation")
    require(
        validation.get("schema_version") == SEMANTIC_SCHEMA_VERSION
        and validation.get("artifact_type")
        == "runtime-vnext-cuda-correctness-semantic-trace"
        and validation.get("status") == "pass",
        "semantic validation schema, type, or status is invalid",
    )
    source_git_sha = validation.get("source_git_sha")
    source_tree_sha = validation.get("source_tree_sha")
    require(
        isinstance(source_git_sha, str)
        and GIT_SHA_RE.fullmatch(source_git_sha) is not None
        and isinstance(source_tree_sha, str)
        and GIT_SHA_RE.fullmatch(source_tree_sha) is not None,
        "semantic validation source identity is invalid",
    )
    require(
        validation.get("dirty_status")
        == {"is_dirty": False, "status_short": []},
        "semantic validation does not prove a clean source",
    )
    require(
        isinstance(validation.get("pass_line"), str)
        and validation["pass_line"].startswith(
            f"{SEMANTIC_PASS_PREFIX}: "
        ),
        "semantic validation PASS line is invalid",
    )
    inputs = validation.get("inputs")
    expected_inputs = {
        "build-manifest.json",
        "execution-manifest.json",
        "focused-report.json",
        "scheduler-trace.jsonl",
        "correctness-binary",
        "correctness-typed-effective-config.json",
        "release-plan-reference.json",
    }
    require(
        isinstance(inputs, dict) and set(inputs) == expected_inputs,
        "semantic validation input set is invalid",
    )
    resolved = {
        label: resolve_file_ref(
            root,
            inputs[label],
            f"semantic validation {label}",
        )
        for label in sorted(expected_inputs)
    }

    build = read_json_object(
        resolved["build-manifest.json"],
        "semantic correctness build manifest",
    )
    execution = read_json_object(
        resolved["execution-manifest.json"],
        "semantic correctness execution manifest",
    )
    focused = read_json_object(
        resolved["focused-report.json"],
        "semantic correctness focused report",
    )
    hardware_id = build.get("hardware_id")
    require(
        build.get("artifact_type") == CORRECTNESS_ARTIFACT_TYPE
        and build.get("status") == "binary-ready"
        and build.get("source_git_sha") == source_git_sha
        and build.get("source_tree_sha") == source_tree_sha
        and build.get("dirty_status")
        == {"is_dirty": False, "status_short": []}
        and build.get("profile", {}).get("name") == PROFILE
        and build.get("profile", {}).get("settings")
        == {
            "inherits": "release",
            "lto": False,
            "codegen-units": 16,
            "incremental": True,
            "strip": False,
        }
        and build.get("profile", {}).get("inherited_opt_level") == 3
        and build.get("features") == plan_reference.FEATURES
        and build.get("native_build_signal", {}).get(
            "native_recompile_count"
        )
        == 0
        and isinstance(hardware_id, str)
        and hardware_id,
        "semantic correctness build boundary is invalid",
    )
    binary_path = resolved["correctness-binary"]
    binary_sha256 = sha256(binary_path)
    build_binary_ref = build.get("binary")
    execution_binary_ref = execution.get("binary_artifact")
    require(
        isinstance(build_binary_ref, dict)
        and isinstance(execution_binary_ref, dict)
        and build_binary_ref.get("sha256") == binary_sha256
        and build_binary_ref.get("size_bytes") == binary_path.stat().st_size
        and execution_binary_ref.get("kind") == "binary"
        and execution_binary_ref.get("sha256") == binary_sha256
        and execution.get("binary_sha256") == binary_sha256
        and focused.get("binary_sha256") == binary_sha256
        and validation.get("binary_sha256") == binary_sha256,
        "semantic correctness binary identities differ",
    )
    require(
        execution.get("source_git_sha")
        == focused.get("source_git_sha")
        == source_git_sha
        and execution.get("source_tree_sha")
        == focused.get("source_tree_sha")
        == source_tree_sha
        and execution.get("dirty_status")
        == focused.get("dirty_status")
        == {"is_dirty": False, "status_short": []}
        and execution.get("backend") == focused.get("backend") == "cuda"
        and execution.get("hardware_id")
        == focused.get("hardware_id")
        == hardware_id,
        "semantic execution/focused source or hardware boundary differs",
    )
    try:
        reference = plan_reference.validate(
            resolved["release-plan-reference.json"],
            expected_hardware_id=hardware_id,
            candidate_source_git_sha=source_git_sha,
            candidate_source_tree_sha=source_tree_sha,
            repository_root=source_root,
            allow_internal_fixture=allow_internal_fixture,
        )
        plan_reference.validate_focused_report(
            focused,
            expected=execution,
        )
    except plan_reference.PlanReferenceError as error:
        raise CorrectnessBuildError(
            f"semantic release reference failed: {error}"
        ) from error
    reference_plan_hash = reference["plan_identity"]["plan_hash"]
    contract = build.get("semantic_plan_contract")
    require(
        isinstance(contract, dict)
        and contract.get("reference_kind") == REFERENCE_ARTIFACT_TYPE
        and contract.get("reference_plan_hash") == reference_plan_hash
        and contract.get("reference_source_git_sha")
        == reference["source_git_sha"]
        and contract.get("reference_source_tree_sha")
        == reference["source_tree_sha"]
        and contract.get("reference_binary_sha256")
        == reference["binary_sha256"]
        and contract.get("reference_profile") == "release"
        and contract.get("hardware_id") == hardware_id
        and contract.get("model_key") == REFERENCE_MODEL_KEY
        and contract.get("case_id") == REFERENCE_CASE_ID
        and contract.get("require_exact_match_before_focused_result")
        is True,
        "semantic correctness build is not bound to the copied release reference",
    )
    require(
        execution.get("models_lock_sha256")
        == reference.get("models_lock_sha256")
        and execution.get("model_revision")
        == reference.get("model_revision")
        and plan_reference.canonical_json_sha256(
            execution.get("model_files")
        )
        == reference.get("model_files_sha256"),
        "semantic release/correctness model locks differ",
    )
    correctness_config_path = resolved[
        "correctness-typed-effective-config.json"
    ]
    correctness_config_ref = execution.get("effective_config")
    require(
        isinstance(correctness_config_ref, dict)
        and correctness_config_ref.get("kind") == "raw-json"
        and correctness_config_ref.get("sha256")
        == sha256(correctness_config_path),
        "semantic correctness typed config identity differs from execution",
    )
    reference_config_path = plan_reference.resolve_file_ref(
        resolved["release-plan-reference.json"].parent,
        reference["inputs"]["typed-effective-config.json"],
        "semantic release typed effective config",
    )
    require(
        read_typed_effective_config(
            correctness_config_path,
            "semantic correctness typed effective config",
        )
        == read_typed_effective_config(
            reference_config_path,
            "semantic release typed effective config",
        ),
        "semantic release/correctness typed effective configs differ",
    )
    try:
        events = plan_reference.plan_events_from_trace(
            resolved["scheduler-trace.jsonl"]
        )
    except plan_reference.PlanReferenceError as error:
        raise CorrectnessBuildError(
            f"semantic scheduler trace failed: {error}"
        ) from error
    observed_hashes = sorted({event["plan_hash"] for event in events})
    request_ids = sorted(
        {
            event["request_id"]
            for event in events
            if isinstance(event["request_id"], str)
        }
    )
    model_values = sorted(
        {
            event["model"]
            for event in events
            if isinstance(event["model"], str)
        }
    )
    require(
        observed_hashes == [reference_plan_hash]
        and validation.get("reference_plan_hash")
        == reference_plan_hash
        and validation.get("reference_source_git_sha")
        == reference["source_git_sha"]
        and validation.get("reference_source_tree_sha")
        == reference["source_tree_sha"]
        and validation.get("reference_binary_sha256")
        == reference["binary_sha256"]
        and validation.get("observed_plan_hashes") == observed_hashes
        and validation.get("plan_built_event_count") == len(events)
        and validation.get("request_ids") == request_ids
        and validation.get("model_values") == model_values
        and validation.get("focused_decision") == focused.get("decision")
        and focused.get("decision") in {"KEEP", "PASS"},
        "semantic validation summary is not reproducible from copied inputs",
    )
    if verify_checkout:
        require(
            source_root is not None,
            "semantic checkout verification requires a source root",
        )
        current = source_identity(
            source_root.expanduser().resolve(),
            require_clean=True,
        )
        require(
            current["source_git_sha"] == source_git_sha
            and current["source_tree_sha"] == source_tree_sha,
            "semantic artifact differs from the current source checkout",
        )
    return {
        "validation_path": str(validation_path),
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "hardware_id": hardware_id,
        "binary_sha256": binary_sha256,
        "reference_plan_hash": reference_plan_hash,
        "reference_binary_sha256": reference["binary_sha256"],
        "plan_built_event_count": len(events),
        "focused_decision": focused["decision"],
    }


def validate_semantic_trace(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    build_manifest_path = args.build_manifest.expanduser().resolve()
    execution_manifest_path = args.execution_manifest.expanduser().resolve()
    focused_report_path = args.focused_report.expanduser().resolve()
    trace_path = args.validate_semantic_trace.expanduser().resolve()
    for path, label in (
        (build_manifest_path, "build manifest"),
        (execution_manifest_path, "execution manifest"),
        (focused_report_path, "focused report"),
        (trace_path, "scheduler trace"),
    ):
        require(path.is_file(), f"{label} is missing: {path}")

    build = read_json_object(build_manifest_path, "build manifest")
    execution = read_json_object(execution_manifest_path, "execution manifest")
    focused = read_json_object(focused_report_path, "focused report")
    require(
        build.get("artifact_type") == CORRECTNESS_ARTIFACT_TYPE
        and build.get("status") == "binary-ready",
        "build manifest is not a CUDA correctness binary-ready artifact",
    )
    contract = build.get("semantic_plan_contract")
    require(isinstance(contract, dict), "build semantic_plan_contract is missing")
    reference_ref = contract.get("reference_plan_manifest")
    reference_path = resolve_file_ref(
        build_manifest_path.parent,
        reference_ref,
        "build release plan reference",
    )
    reference = plan_reference.validate(
        reference_path,
        expected_hardware_id=str(build.get("hardware_id", "")),
    )
    reference_plan_hash = reference["plan_identity"]["plan_hash"]
    require(
        contract.get("reference_kind") == REFERENCE_ARTIFACT_TYPE
        and contract.get("reference_plan_hash") == reference_plan_hash
        and contract.get("reference_source_git_sha") == reference["source_git_sha"]
        and contract.get("reference_source_tree_sha") == reference["source_tree_sha"]
        and contract.get("reference_binary_sha256") == reference["binary_sha256"]
        and contract.get("reference_profile") == "release"
        and contract.get("hardware_id") == reference["hardware_id"]
        and contract.get("model_key") == REFERENCE_MODEL_KEY
        and contract.get("case_id") == REFERENCE_CASE_ID
        and contract.get("require_exact_match_before_focused_result") is True,
        "build manifest release plan reference binding is incomplete",
    )
    build_binary_ref = build.get("binary")
    require(isinstance(build_binary_ref, dict), "build binary reference is missing")
    build_binary = build_manifest_path.parent / str(build_binary_ref.get("path", ""))
    require(build_binary.is_file(), f"build binary is missing: {build_binary}")
    build_binary_sha256 = sha256(build_binary)
    require(
        build_binary_sha256 == build_binary_ref.get("sha256"),
        "build binary SHA256 differs from its manifest",
    )

    execution_root = execution_manifest_path.parent
    plan_reference.require_under(focused_report_path, execution_root, "focused report")
    plan_reference.require_under(trace_path, execution_root, "scheduler trace")
    require(
        trace_path.name.endswith(".scheduler-trace.jsonl"),
        "semantic input is not a scheduler-trace JSONL file",
    )
    execution_binary_ref = execution.get("binary_artifact")
    execution_binary = plan_reference.resolve_execution_artifact_ref(
        execution_root,
        execution_binary_ref,
        "correctness execution binary",
        expected_kind="binary",
    )
    execution_binary_sha256 = sha256(execution_binary)
    require(
        execution_binary_sha256
        == execution_binary_ref.get("sha256")
        == execution.get("binary_sha256")
        == focused.get("binary_sha256")
        == build_binary_sha256,
        "build, execution, and focused-report binary SHA256 identities differ",
    )
    require(
        execution.get("backend") == focused.get("backend") == "cuda",
        "semantic validation requires CUDA execution evidence",
    )
    require(
        build.get("hardware_id")
        == execution.get("hardware_id")
        == focused.get("hardware_id")
        == reference.get("hardware_id"),
        "release reference, build, execution, and focused-report hardware identities differ",
    )
    require(
        execution.get("source_git_sha")
        == focused.get("source_git_sha")
        == build.get("source_git_sha"),
        "build, execution, and focused-report Git SHAs differ",
    )
    require(
        execution.get("source_tree_sha")
        == focused.get("source_tree_sha")
        == build.get("source_tree_sha"),
        "build, execution, and focused-report source tree SHAs differ",
    )
    require(
        not build.get("dirty_status", {}).get("is_dirty")
        and not execution.get("dirty_status", {}).get("is_dirty")
        and not focused.get("dirty_status", {}).get("is_dirty"),
        "semantic validation requires clean build and execution sources",
    )
    plan_reference.validate_focused_report(focused, expected=execution)
    require(
        execution.get("models_lock_sha256") == reference.get("models_lock_sha256")
        and execution.get("model_revision") == reference.get("model_revision")
        and plan_reference.canonical_json_sha256(execution.get("model_files"))
        == reference.get("model_files_sha256"),
        "release reference and correctness execution model locks differ",
    )
    execution_effective = plan_reference.resolve_execution_artifact_ref(
        execution_root,
        execution.get("effective_config"),
        "correctness typed effective config",
        expected_kind="raw-json",
    )
    reference_effective = plan_reference.resolve_file_ref(
        reference_path.parent,
        reference["inputs"]["typed-effective-config.json"],
        "release reference typed effective config",
    )
    require(
        read_typed_effective_config(
            execution_effective,
            "correctness typed effective config",
        )
        == read_typed_effective_config(
            reference_effective,
            "release reference typed effective config",
        ),
        "release reference and correctness typed effective configs differ",
    )

    plan_events = plan_reference.plan_events_from_trace(trace_path)
    observed_hashes = sorted({event["plan_hash"] for event in plan_events})
    require(
        observed_hashes == [reference_plan_hash],
        "stale-plan-reference: scheduler trace plan hash mismatch: "
        f"reference {reference_plan_hash}, observed {observed_hashes}",
    )

    input_dir = root / "inputs"
    require(
        not input_dir.exists(),
        f"semantic validation input directory already exists: {input_dir}",
    )
    input_dir.mkdir(parents=True)
    copied: dict[str, Any] = {}
    for label, source in (
        ("build-manifest.json", build_manifest_path),
        ("execution-manifest.json", execution_manifest_path),
        ("focused-report.json", focused_report_path),
        ("scheduler-trace.jsonl", trace_path),
        ("correctness-binary", build_binary),
        (
            "correctness-typed-effective-config.json",
            execution_effective,
        ),
    ):
        destination = input_dir / label
        shutil.copy2(source, destination)
        copied[label] = file_ref(destination, root)
    reference_copy_root = input_dir / "release-plan-reference"
    shutil.copytree(
        reference_path.parent,
        reference_copy_root,
        symlinks=False,
    )
    copied_reference_path = reference_copy_root / reference_path.name
    copied["release-plan-reference.json"] = file_ref(
        copied_reference_path,
        root,
    )
    result = {
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "artifact_type": "runtime-vnext-cuda-correctness-semantic-trace",
        "status": "pass",
        "created_at": now_iso(),
        "source_git_sha": build["source_git_sha"],
        "source_tree_sha": build["source_tree_sha"],
        "dirty_status": {"is_dirty": False, "status_short": []},
        "binary_sha256": build_binary_sha256,
        "reference_plan_hash": reference_plan_hash,
        "reference_source_git_sha": reference["source_git_sha"],
        "reference_source_tree_sha": reference["source_tree_sha"],
        "reference_binary_sha256": reference["binary_sha256"],
        "observed_plan_hashes": observed_hashes,
        "plan_built_event_count": len(plan_events),
        "request_ids": sorted(
            {
                event["request_id"]
                for event in plan_events
                if isinstance(event["request_id"], str)
            }
        ),
        "model_values": sorted(
            {
                event["model"]
                for event in plan_events
                if isinstance(event["model"], str)
            }
        ),
        "focused_decision": focused["decision"],
        "inputs": copied,
        "pass_line": f"{SEMANTIC_PASS_PREFIX}: {root}",
    }
    write_json(root / "validation.json", result)
    verify_semantic_trace_artifact(
        root,
        source_root=ROOT,
        verify_checkout=False,
    )
    return result


def self_test() -> None:
    command = make_build_command(
        target_dir=Path("/tmp/target"),
        native_cache=Path("/tmp/native-cache"),
        import_dirs=[Path("/tmp/release/out")],
        native_operator_set_lock=Path("/tmp/native-operators.lock.json"),
        compute_capability="89",
        cargo_jobs=4,
        nvcc_threads=4,
    )
    require(
        "FERRUM_NVCC_THREADS=4" in command,
        "typed NVCC worker bound was not written into the build command",
    )
    require(
        "FERRUM_CUDA_NATIVE_SOURCE_POLICY=cache-only" in command,
        "cache-only native source policy was not written into the build command",
    )
    require(
        "FERRUM_NATIVE_OPERATOR_SET_LOCK=/tmp/native-operators.lock.json" in command,
        "typed native operator set lock was not written into the build command",
    )
    clean_log = "\n".join(
        [
            "[cuda-native-build-cache] artifact=static.vllm_marlin status=imported "
            f"sha256={'a' * 64} import_dir=/tmp/release/out",
            "[cuda-build-summary] artifact=vllm_marlin status=cache_hit "
            f"reason=shared-native-build-cache elapsed_ms=1 inputs_hash=sha256:{'b' * 64}",
        ]
    )
    clean = parse_native_build_log(clean_log)
    require(clean["native_recompile_count"] == 0, "cache-hit log was classified as rebuild")
    rebuilt = parse_native_build_log(
        "[cuda-build-summary] artifact=vllm_marlin status=built "
        f"reason=missing-lib elapsed_ms=1 inputs_hash=sha256:{'b' * 64}"
    )
    require(rebuilt["native_recompile_count"] == 1, "native rebuild was not rejected")
    unmanaged = parse_native_build_log(
        "     Running `CARGO_PKG_NAME=candle-kernels "
        "OUT_DIR=/tmp/candle/out /tmp/candle/build-script-build`"
    )
    require(
        unmanaged["unmanaged_native_recompile_count"] == 1
        and unmanaged["native_recompile_count"] == 1
        and unmanaged["unmanaged_native_build_scripts"][0]["package"]
        == "candle-kernels",
        "unmanaged Candle CUDA build-script execution was not rejected",
    )
    compiler_log = json.dumps(
        {
            "reason": "compiler-artifact",
            "target": {"name": "ferrum", "kind": ["bin"]},
            "executable": "/tmp/target/cuda-correctness/ferrum",
            "fresh": False,
            "profile": {"opt_level": "3"},
        }
    )
    compiler_artifacts = parse_ferrum_compiler_artifacts(compiler_log)
    require(
        len(compiler_artifacts) == 1 and compiler_artifacts[0]["fresh"] is False,
        "non-fresh ferrum compiler artifact was not parsed",
    )

    with tempfile.TemporaryDirectory(prefix="ferrum-cuda-correctness-build-") as raw:
        root = Path(raw)
        release_effective = root / "release-effective.json"
        correctness_effective = root / "correctness-effective.json"
        typed_config = {
            "composition_contract": "resolved-model-plan-vnext",
            "run": {"backend": "cuda", "gpu_devices": [0]},
            "serve": {
                "backend": "cuda",
                "gpu_devices": [0],
                "served_model_name": REFERENCE_MODEL_KEY,
            },
        }
        write_json(
            release_effective,
            {
                "binary_sha256": "a" * 64,
                "typed_effective_config": typed_config,
            },
        )
        write_json(
            correctness_effective,
            {
                "binary_sha256": "b" * 64,
                "typed_effective_config": typed_config,
            },
        )
        require(
            read_typed_effective_config(
                release_effective,
                "release effective fixture",
            )
            == read_typed_effective_config(
                correctness_effective,
                "correctness effective fixture",
            ),
            "binary identity leaked into the typed effective config contract",
        )
        changed_typed_config = copy.deepcopy(typed_config)
        changed_typed_config["serve"]["gpu_devices"] = [1]
        write_json(
            correctness_effective,
            {
                "binary_sha256": "b" * 64,
                "typed_effective_config": changed_typed_config,
            },
        )
        require(
            read_typed_effective_config(
                release_effective,
                "release effective fixture",
            )
            != read_typed_effective_config(
                correctness_effective,
                "changed correctness effective fixture",
            ),
            "typed effective config drift was normalized away",
        )
        source = root / "source"
        source.mkdir()
        (source / "Cargo.toml").write_text(
            """
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true

[profile.cuda-correctness]
inherits = "release"
lto = false
codegen-units = 16
incremental = true
strip = false
""".strip()
            + "\n",
            encoding="utf-8",
        )
        kernel_dir = source / "crates/ferrum-kernels"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "build.rs").write_text(
            'const CORE_PTX_KERNELS: &[&str] = &["kernels/a.cu", "kernels/b.cu"];\n',
            encoding="utf-8",
        )
        outputs = core_ptx_outputs(source)
        require(
            outputs
            == [
                ("core_ptx.a", "a.ptx", "a.ptx.stamp"),
                ("core_ptx.b", "b.ptx", "b.ptx.stamp"),
            ],
            "core PTX inventory parsing drift",
        )
        validate_profile(source)
        fallback_document = (source / "Cargo.toml").read_text(encoding="utf-8")
        require(
            parse_flat_toml_table(
                fallback_document, f"profile.{PROFILE}"
            )
            == {
                "inherits": "release",
                "lto": False,
                "codegen-units": 16,
                "incremental": True,
                "strip": False,
            },
            "Python 3.10 profile parser drift",
        )
        import_out = root / "target/release/build/ferrum-kernels-fixture/out"
        import_out.mkdir(parents=True)
        for _, file_name, stamp_name in outputs:
            (import_out / file_name).write_bytes(b"artifact")
            (import_out / stamp_name).write_text("signature", encoding="utf-8")
        discovered = discover_import_dirs([root / "target"])
        inventory, missing = inventory_imports(source, discovered)
        require(not missing, f"complete import fixture was rejected: {missing}")
        require(
            len(inventory) == len(outputs),
            "native import inventory cardinality drift",
        )
        (import_out / "b.ptx.stamp").unlink()
        _, missing = inventory_imports(source, discovered)
        require(
            missing == ["core_ptx.b"],
            f"missing native import was not classified: {missing}",
        )

        source_native_lock = native_operator_set.create_selftest_native_operator_set(
            root / "native-source",
            plan_reference.REQUIRED_CUDA_NATIVE_OPERATORS,
        )
        portable_native_lock, portable_native_set = (
            native_operator_set.stage_native_operator_set(
                source_native_lock,
                root / "native-operator-set",
                plan_reference.REQUIRED_CUDA_NATIVE_OPERATORS,
            )
        )
        portable_ref = file_ref(portable_native_lock, root)
        require(
            portable_ref["path"]
            == "native-operator-set/native-operator-set.lock.json"
            and portable_ref["sha256"]
            == portable_native_set["lock_sha256"],
            "portable native operator set artifact binding drifted",
        )

        execution_root, fixture = plan_reference.make_fixture(root)
        execution_manifest_path = Path(fixture["execution"])
        focused_report = Path(fixture["focused"])
        trace = Path(fixture["trace"])
        actual_effective_config = Path(fixture["actual"])
        hardware_id = fixture["hardware_id"]
        source_git_sha = fixture["source_git_sha"]
        source_tree_sha = fixture["source_tree_sha"]
        plan_hash = fixture["plan_hash"]
        fixture_effective = execution_root / "effective-config.json"
        write_json(
            fixture_effective,
            {
                "schema_version": 1,
                "execution_contract": plan_reference.EXECUTION_CONTRACT,
                "typed_effective_config": typed_config,
            },
        )
        fixture_execution = read_json_object(
            execution_manifest_path,
            "fixture execution manifest",
        )
        fixture_execution["effective_config"] = (
            plan_reference.execution_artifact_ref(
                fixture_effective,
                execution_root,
                "raw-json",
            )
        )
        write_json(execution_manifest_path, fixture_execution)
        execution_document = read_json_object(
            execution_manifest_path,
            "fixture execution manifest",
        )
        execution_binary = plan_reference.resolve_execution_artifact_ref(
            execution_root,
            execution_document["binary_artifact"],
            "fixture execution binary",
            expected_kind="binary",
        )
        binary_bytes = execution_binary.read_bytes()
        reference_artifact = root / "release-plan-reference"
        reference = plan_reference.capture(
            out_root=reference_artifact,
            execution_manifest_path=execution_manifest_path,
            focused_report_path=focused_report,
            trace_path=trace,
            actual_effective_config_path=actual_effective_config,
            hardware_id=hardware_id,
        )
        validated_reference = plan_reference.validate(
            reference_artifact / "manifest.json",
            expected_hardware_id=hardware_id,
            candidate_source_git_sha=source_git_sha,
            candidate_source_tree_sha=source_tree_sha,
            allow_internal_fixture=True,
        )
        require(
            reference == validated_reference
            and reference["plan_identity"]["plan_hash"] == plan_hash,
            "matching release plan reference was not captured and validated",
        )

        build_artifact = root / "build-artifact"
        build_binary = build_artifact / "binary/ferrum"
        build_binary.parent.mkdir(parents=True)
        build_binary.write_bytes(binary_bytes)
        imported_reference_root = build_artifact / "release-plan-reference"
        shutil.copytree(reference_artifact, imported_reference_root)
        imported_reference_manifest = imported_reference_root / "manifest.json"
        reference_ref = file_ref(imported_reference_manifest, build_artifact)
        write_json(
            build_artifact / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": CORRECTNESS_ARTIFACT_TYPE,
                "status": "binary-ready",
                "source_git_sha": source_git_sha,
                "source_tree_sha": source_tree_sha,
                "dirty_status": {"is_dirty": False, "status_short": []},
                "hardware_id": hardware_id,
                "profile": {
                    "name": PROFILE,
                    "settings": {
                        "inherits": "release",
                        "lto": False,
                        "codegen-units": 16,
                        "incremental": True,
                        "strip": False,
                    },
                    "inherited_opt_level": 3,
                    "semantic_delta_from_release": [
                        "lto",
                        "codegen-units",
                        "incremental",
                        "strip",
                    ],
                },
                "features": copy.deepcopy(plan_reference.FEATURES),
                "native_build_signal": {
                    "native_recompile_count": 0,
                },
                "binary": file_ref(build_binary, build_artifact),
                "semantic_plan_contract": {
                    "status": "pending-product-trace-validation",
                    "reference_kind": REFERENCE_ARTIFACT_TYPE,
                    "reference_plan_manifest": reference_ref,
                    "reference_source_git_sha": source_git_sha,
                    "reference_source_tree_sha": source_tree_sha,
                    "reference_plan_hash": plan_hash,
                    "reference_binary_sha256": sha256(execution_binary),
                    "reference_profile": "release",
                    "hardware_id": hardware_id,
                    "model_key": REFERENCE_MODEL_KEY,
                    "case_id": REFERENCE_CASE_ID,
                    "require_exact_match_before_focused_result": True,
                },
            },
        )
        semantic_args = argparse.Namespace(
            build_manifest=build_artifact / "manifest.json",
            execution_manifest=execution_root / "execution-manifest.json",
            focused_report=focused_report,
            validate_semantic_trace=trace,
        )
        semantic = validate_semantic_trace(
            semantic_args, root / "semantic-validation"
        )
        require(
            semantic["status"] == "pass"
            and semantic["observed_plan_hashes"] == [plan_hash],
            "matching semantic trace was not accepted",
        )
        verified_semantic = verify_semantic_trace_artifact(
            root / "semantic-validation",
            allow_internal_fixture=True,
        )
        require(
            verified_semantic["binary_sha256"] == sha256(build_binary)
            and verified_semantic["reference_plan_hash"] == plan_hash,
            "self-contained semantic artifact did not round-trip",
        )

        def expect_semantic_artifact_reject(
            name: str,
            mutate: Any,
            expected_error: str,
        ) -> None:
            hostile_root = root / f"semantic-hostile-{name}"
            shutil.copytree(root / "semantic-validation", hostile_root)
            mutate(hostile_root)
            try:
                verify_semantic_trace_artifact(
                    hostile_root,
                    allow_internal_fixture=True,
                )
            except CorrectnessBuildError as error:
                require(
                    expected_error in str(error),
                    f"{name} used an unexpected rejection: {error}",
                )
                return
            raise CorrectnessBuildError(
                f"hostile semantic artifact was accepted: {name}"
            )

        expect_semantic_artifact_reject(
            "correctness-binary",
            lambda hostile: (
                hostile / "inputs/correctness-binary"
            ).write_bytes(b"tampered-correctness-binary"),
            "reference identity mismatch",
        )

        def mutate_typed_config(hostile: Path) -> None:
            config_path = (
                hostile
                / "inputs/correctness-typed-effective-config.json"
            )
            config = read_json_object(config_path, "hostile typed config")
            config["typed_effective_config"]["serve"]["gpu_devices"] = [1]
            write_json(config_path, config)
            execution_path = hostile / "inputs/execution-manifest.json"
            execution_doc = read_json_object(
                execution_path,
                "hostile execution",
            )
            execution_doc["effective_config"]["sha256"] = sha256(config_path)
            write_json(execution_path, execution_doc)
            validation_path = hostile / "validation.json"
            validation_doc = read_json_object(
                validation_path,
                "hostile semantic validation",
            )
            validation_doc["inputs"][
                "correctness-typed-effective-config.json"
            ] = file_ref(config_path, hostile)
            validation_doc["inputs"]["execution-manifest.json"] = file_ref(
                execution_path,
                hostile,
            )
            write_json(validation_path, validation_doc)

        expect_semantic_artifact_reject(
            "typed-config",
            mutate_typed_config,
            "typed effective configs differ",
        )

        def mutate_model_lock(hostile: Path) -> None:
            execution_path = hostile / "inputs/execution-manifest.json"
            execution_doc = read_json_object(
                execution_path,
                "hostile model-lock execution",
            )
            execution_doc["model_revision"] = "9" * 40
            write_json(execution_path, execution_doc)
            validation_path = hostile / "validation.json"
            validation_doc = read_json_object(
                validation_path,
                "hostile semantic validation",
            )
            validation_doc["inputs"]["execution-manifest.json"] = file_ref(
                execution_path,
                hostile,
            )
            write_json(validation_path, validation_doc)

        expect_semantic_artifact_reject(
            "model-lock",
            mutate_model_lock,
            "model_revision differs",
        )
        expect_semantic_artifact_reject(
            "release-binary",
            lambda hostile: (
                hostile
                / "inputs/release-plan-reference/inputs/release-binary"
            ).write_bytes(b"tampered-release-binary"),
            "release-binary reference identity mismatch",
        )

        trace.write_text(
            trace.read_text(encoding="utf-8").replace(plan_hash, "b" * 64),
            encoding="utf-8",
        )
        try:
            validate_semantic_trace(semantic_args, root / "semantic-reject")
        except CorrectnessBuildError:
            pass
        else:
            raise CorrectnessBuildError("mismatched semantic trace was accepted")
    print(SELFTEST_PASS_LINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--native-cache", type=Path)
    parser.add_argument("--native-operator-set-lock", type=Path)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--import-target-root", type=Path, action="append", default=[])
    parser.add_argument("--compute-capability", default="89")
    parser.add_argument("--hardware-id")
    parser.add_argument("--cargo-jobs", type=int, default=4)
    parser.add_argument("--nvcc-threads", type=int, default=4)
    parser.add_argument("--wall-timeout-seconds", type=float, default=600)
    parser.add_argument("--reference-plan-manifest", type=Path)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--focused-report", type=Path)
    parser.add_argument("--validate-semantic-trace", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    require(args.out is not None, "--out is required")
    require(
        isinstance(args.hardware_id, str)
        and args.hardware_id.strip() == args.hardware_id
        and args.hardware_id,
        "--hardware-id must be non-empty and trimmed",
    )
    if args.validate_semantic_trace is not None:
        require(args.build_manifest is not None, "--build-manifest is required")
        require(args.execution_manifest is not None, "--execution-manifest is required")
        require(args.focused_report is not None, "--focused-report is required")
        require(
            args.reference_plan_manifest is None and not args.plan_only,
            "semantic trace validation consumes the build-bound reference only",
        )
        return args
    require(
        args.reference_plan_manifest is not None,
        "--reference-plan-manifest is required",
    )
    require(args.native_cache is not None, "--native-cache is required")
    require(
        args.native_operator_set_lock is not None,
        "--native-operator-set-lock is required",
    )
    require(args.target_dir is not None, "--target-dir is required")
    require(args.import_target_root, "--import-target-root is required")
    require(
        1 <= args.nvcc_threads <= 8,
        "--nvcc-threads must be in [1, 8]",
    )
    require(
        1 <= args.wall_timeout_seconds <= 1800,
        "--wall-timeout-seconds must be in [1, 1800]",
    )
    return args


def main() -> int:
    try:
        args = parse_args()
        if args.self_test:
            self_test()
            return 0
        source_root = args.source_root.expanduser().resolve()
        root = artifact_root(args.out, source_root)
        require(not any(root.iterdir()), f"artifact root must be empty: {root}")
        if args.validate_semantic_trace is not None:
            validate_semantic_trace(args, root)
            print(f"{SEMANTIC_PASS_PREFIX}: {root}")
            return 0
        plan = create_plan(args, root, require_clean=not args.plan_only)
        write_json(root / "plan.json", plan)
        if args.plan_only:
            require(plan["ready"], f"native import inventory is incomplete: {plan['missing_native_imports']}")
            print(f"{PLAN_READY_PREFIX}: {root}")
            return 0
        run_build(args, plan, root)
        print(f"{READY_PREFIX}: {root}")
        return 0
    except (
        CorrectnessBuildError,
        native_operator_set.NativeOperatorSetEvidenceError,
        OSError,
        ValueError,
        plan_reference.PlanReferenceError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"FERRUM CUDA CORRECTNESS BUILD REJECT: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
