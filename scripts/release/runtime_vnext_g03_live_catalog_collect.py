#!/usr/bin/env python3
"""Collect a live CUDA operation/provider catalog for the G03 checkpoint.

The collector deliberately emits only EVIDENCE READY.  A separate checkpoint
validator binds these raw files to canonical S1 evidence before it can print a
G03 live-catalog PASS line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import bounded_command
from runtime_vnext_native_operator_set import (
    NativeOperatorSetEvidenceError,
    public_identity as native_operator_set_public_identity,
    stage_native_operator_set,
    validate_cuda_build_summary,
    validate_native_operator_set,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_PATH = Path(__file__).resolve()
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "runtime_vnext_g03_live_catalog_raw_collection"
EVIDENCE_READY_LINE = "FERRUM RUNTIME VNEXT G03 LIVE CATALOG EVIDENCE READY"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G03 LIVE CATALOG COLLECTOR SELFTEST PASS"
REQUIRED_CUDA_NATIVE_OPERATORS = (
    "ferrum.cuda.marlin",
    "ferrum.cuda.vllm_marlin",
    "ferrum.cuda.vllm_moe_marlin",
    "ferrum.cuda.vllm_paged_attention_v2",
)
REQUIRED_CUDA_NATIVE_BUILD_UNITS = (
    ("marlin", "marlin", "ferrum.cuda.marlin"),
    ("vllm_marlin", "vllm_marlin", "ferrum.cuda.vllm_marlin"),
    (
        "vllm_moe_marlin",
        "vllm_moe_marlin",
        "ferrum.cuda.vllm_moe_marlin",
    ),
    (
        "vllm_paged_attn",
        "vllm_paged_attention_v2",
        "ferrum.cuda.vllm_paged_attention_v2",
    ),
)
PRODUCT_FEATURES = (
    "cuda",
    "vllm-moe-marlin",
    "vllm-paged-attn-v2",
)
PROVIDER_FIELDS = {
    "operation_id",
    "operation_contract_version",
    "operation_fingerprint",
    "provider_id",
    "provider_version",
    "provider_implementation_fingerprint",
}
CAPABILITY_FIELDS = {
    "device",
    "operations",
    "providers",
    "engine_providers",
    "weight_materializers",
}
EXPORT_READY_RE = re.compile(
    r"^FERRUM RUNTIME VNEXT CUDA NATIVE CATALOG INPUT READY: "
    r"provider=(?P<provider>\S+) capability=(?P<capability>\S+) "
    r"provider_count=(?P<provider_count>[1-9][0-9]*) "
    r"capability_fingerprint=(?P<capability_fingerprint>[0-9a-f]{64})$"
)
CORE_PTX_BLOCK_RE = re.compile(
    r"const CORE_PTX_KERNELS:\s*&\[&str\]\s*=\s*&\[(?P<body>.*?)\];",
    re.DOTALL,
)
QUOTED_RUST_PATH_RE = re.compile(r'"([^"\r\n]+)"')
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
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class CollectionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectionError(message)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


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


def serde_json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reject_hidden_build_overrides() -> None:
    rejected = sorted(
        key
        for key, value in os.environ.items()
        if value
        and (
            key in FORBIDDEN_OVERRIDE_KEYS
            or any(key.startswith(prefix) for prefix in FORBIDDEN_OVERRIDE_PREFIXES)
        )
    )
    require(
        not rejected,
        f"hidden compiler/profile overrides are forbidden: {rejected}",
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path, label: str) -> Any:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CollectionError(f"cannot read {label} {path}: {error}") from error


def run_text(cwd: Path, command: Sequence[str], timeout: int = 60) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    require(
        result.returncode == 0,
        f"command failed ({result.returncode}): {list(command)!r}: "
        f"{result.stderr[-2000:]}",
    )
    return result.stdout.strip()


def resolve_tool(raw: str, label: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        executable = candidate.absolute()
    else:
        found = shutil.which(raw)
        require(found is not None, f"{label} is not on PATH: {raw}")
        executable = Path(found).absolute()  # type: ignore[arg-type]
    require(
        executable.is_file() and os.access(executable, os.X_OK),
        f"{label} is not executable: {executable}",
    )
    resolved = executable.resolve()
    if label in {"cargo", "rustc"} and resolved.name == "rustup":
        actual = Path(
            run_text(REPO_ROOT, [str(resolved), "which", label])
        ).expanduser().resolve()
        require(
            actual.is_file() and os.access(actual, os.X_OK) and actual.name == label,
            f"rustup did not resolve an executable {label}: {actual}",
        )
        return actual
    return resolved


def source_identity(source_root: Path) -> dict[str, Any]:
    git_sha = run_text(source_root, ["git", "rev-parse", "HEAD"])
    tree_sha = run_text(source_root, ["git", "rev-parse", "HEAD^{tree}"])
    status = run_text(
        source_root,
        ["git", "status", "--short", "--untracked-files=all"],
    ).splitlines()
    require(GIT_SHA_RE.fullmatch(git_sha) is not None, "invalid source Git SHA")
    require(GIT_SHA_RE.fullmatch(tree_sha) is not None, "invalid source tree SHA")
    require(not status, f"G03 live catalog requires clean source: {status}")
    return {
        "git_sha": git_sha,
        "git_tree_sha": tree_sha,
        "dirty": False,
        "status_short": [],
    }


def hardware_identity(
    source_root: Path,
    *,
    nvidia_smi: Path,
    nvcc: Path,
    cargo: Path,
    rustc: Path,
) -> dict[str, Any]:
    rows = run_text(
        source_root,
        [
            str(nvidia_smi),
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ],
    ).splitlines()
    require(len(rows) == 1, f"G03 live catalog requires exactly one GPU: {rows}")
    require("RTX 4090" in rows[0], f"G03 live catalog requires RTX 4090: {rows[0]!r}")
    return {
        "policy": "cuda-g0-1x-rtx4090",
        "gpu_count": 1,
        "gpu": rows[0],
        "nvidia_smi": run_text(source_root, [str(nvidia_smi)]),
        "nvcc": run_text(source_root, [str(nvcc), "--version"]),
        "cargo": run_text(source_root, [str(cargo), "-V"]),
        "rustc": run_text(source_root, [str(rustc), "-vV"]),
        "tools": {
            "nvidia_smi": file_identity(nvidia_smi),
            "nvcc": file_identity(nvcc),
            "cargo": file_identity(cargo),
            "rustc": file_identity(rustc),
        },
    }


def file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display = (
        resolved.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None and resolved.is_relative_to(relative_to.resolve())
        else str(resolved)
    )
    return {
        "path": display,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_version(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == {"major", "minor"}, f"{label} field set mismatch")
    for field in ("major", "minor"):
        require(
            isinstance(value[field], int)
            and not isinstance(value[field], bool)
            and value[field] >= 0,
            f"{label}.{field} is invalid",
        )
    require(value["major"] >= 1, f"{label}.major must be at least 1")


def provider_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation_id": row["operation_id"],
        "operation_contract_version": row["operation_contract_version"],
        "operation_fingerprint": row["operation_fingerprint"],
        "provider_id": row["provider_id"],
        "provider_version": row["provider_version"],
        "provider_implementation_fingerprint": row[
            "provider_implementation_fingerprint"
        ],
    }


def provider_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["operation_id"],
        row["operation_contract_version"]["major"],
        row["operation_contract_version"]["minor"],
        row["operation_fingerprint"],
        row["provider_id"],
        row["provider_version"]["major"],
        row["provider_version"]["minor"],
        row["provider_implementation_fingerprint"],
    )


def validate_provider_catalog(path: Path) -> dict[str, Any]:
    value = read_json(path, "provider catalog")
    require(isinstance(value, dict), "provider catalog must be an object")
    require(
        set(value) == {"schema_version", "backend", "providers"},
        "provider catalog field set mismatch",
    )
    require(value.get("schema_version") == 1, "provider catalog schema must be 1")
    require(value.get("backend") == "cuda", "provider catalog backend must be CUDA")
    providers = value.get("providers")
    require(isinstance(providers, list) and providers, "provider catalog is empty")
    provider_ids: list[str] = []
    operation_ids: set[str] = set()
    for index, row in enumerate(providers):
        require(isinstance(row, dict), f"provider row {index} must be an object")
        require(set(row) == PROVIDER_FIELDS, f"provider row {index} field set mismatch")
        for field in ("operation_id", "provider_id"):
            require(
                isinstance(row[field], str) and row[field],
                f"provider row {index}.{field} is invalid",
            )
        require(
            row["operation_id"].startswith("operation."),
            f"provider row {index}.operation_id prefix is invalid",
        )
        require(
            row["provider_id"].startswith("provider.cuda."),
            f"provider row {index}.provider_id prefix is invalid",
        )
        for field in ("operation_fingerprint", "provider_implementation_fingerprint"):
            require(
                isinstance(row[field], str)
                and SHA256_RE.fullmatch(row[field]) is not None,
                f"provider row {index}.{field} is invalid",
            )
        validate_version(row["operation_contract_version"], f"provider row {index} operation version")
        validate_version(row["provider_version"], f"provider row {index} provider version")
        provider_ids.append(row["provider_id"])
        operation_ids.add(row["operation_id"])
    require(len(provider_ids) == len(set(provider_ids)), "provider IDs are not unique")
    require(
        providers == sorted(providers, key=provider_sort_key),
        "provider catalog is not deterministically sorted",
    )
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "canonical_fingerprint": canonical_json_sha256(value),
        "provider_count": len(providers),
        "operation_count": len(operation_ids),
        "provider_ids": provider_ids,
        "operation_ids": sorted(operation_ids),
        "projection": [provider_projection(row) for row in providers],
    }


def validate_capability_catalog(path: Path, *, cuda_ordinal: int) -> dict[str, Any]:
    value = read_json(path, "capability catalog")
    require(isinstance(value, dict), "capability catalog must be an object")
    require(set(value) == CAPABILITY_FIELDS, "capability catalog field set mismatch")
    device = value.get("device")
    require(isinstance(device, dict), "capability catalog device must be an object")
    require(device.get("id") == f"cuda:{cuda_ordinal}", "capability device ID mismatch")
    require(device.get("class") == "accelerator", "capability device class mismatch")
    require(device.get("ordinal") == cuda_ordinal, "capability device ordinal mismatch")
    require(
        isinstance(device.get("total_memory_bytes"), int)
        and not isinstance(device["total_memory_bytes"], bool)
        and device["total_memory_bytes"] > 0,
        "capability device memory is invalid",
    )
    require(
        isinstance(device.get("runtime_implementation_fingerprint"), str)
        and SHA256_RE.fullmatch(device["runtime_implementation_fingerprint"]) is not None,
        "capability runtime implementation fingerprint is invalid",
    )
    capabilities = device.get("capabilities")
    require(
        isinstance(capabilities, list)
        and capabilities
        and capabilities == sorted(set(capabilities)),
        "device capabilities must be a non-empty sorted unique list",
    )
    dynamic_profiles = device.get("dynamic_storage_profiles")
    require(
        isinstance(dynamic_profiles, list) and dynamic_profiles,
        "device dynamic storage profiles are empty",
    )
    counts: dict[str, int] = {}
    for field in ("operations", "providers", "engine_providers", "weight_materializers"):
        rows = value.get(field)
        require(isinstance(rows, dict) and rows, f"capability catalog {field} is empty")
        require(list(rows) == sorted(rows), f"capability catalog {field} is not sorted")
        counts[f"{field}_count"] = len(rows)
    operations = value["operations"]
    provider_groups = value["providers"]
    projected: list[dict[str, Any]] = []
    provider_ids: list[str] = []
    for operation_id, operation in operations.items():
        require(
            isinstance(operation, dict)
            and operation.get("id") == operation_id
            and operation_id.startswith("operation."),
            f"capability operation identity mismatch: {operation_id}",
        )
        operation_version = operation.get("version")
        validate_version(operation_version, f"capability operation {operation_id} version")
        rows = provider_groups.get(operation_id)
        require(
            isinstance(rows, list) and rows,
            f"capability operation has no providers: {operation_id}",
        )
        operation_fingerprint = serde_json_fingerprint(operation)
        for index, provider in enumerate(rows):
            label = f"capability provider {operation_id}[{index}]"
            require(isinstance(provider, dict), f"{label} must be an object")
            require(
                provider.get("operation_id") == operation_id,
                f"{label} operation ID mismatch",
            )
            provider_id = provider.get("provider_id")
            require(
                isinstance(provider_id, str)
                and provider_id.startswith("provider.cuda."),
                f"{label} provider ID is invalid",
            )
            require(
                provider.get("operation_fingerprint") == operation_fingerprint,
                f"{label} operation fingerprint mismatch",
            )
            provider_fingerprint = provider.get(
                "provider_implementation_fingerprint"
            )
            require(
                isinstance(provider_fingerprint, str)
                and SHA256_RE.fullmatch(provider_fingerprint) is not None,
                f"{label} implementation fingerprint is invalid",
            )
            provider_version = provider.get("version")
            validate_version(provider_version, f"{label} version")
            projected.append(
                {
                    "operation_id": operation_id,
                    "operation_contract_version": operation_version,
                    "operation_fingerprint": operation_fingerprint,
                    "provider_id": provider_id,
                    "provider_version": provider_version,
                    "provider_implementation_fingerprint": provider_fingerprint,
                }
            )
            provider_ids.append(provider_id)
    require(
        set(provider_groups) == set(operations),
        "capability provider group keys differ from operation keys",
    )
    require(
        len(provider_ids) == len(set(provider_ids)),
        "capability provider IDs are not unique",
    )
    projected.sort(key=provider_sort_key)
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "canonical_fingerprint": canonical_json_sha256(value),
        "runtime_fingerprint": serde_json_fingerprint(value),
        "device_id": device["id"],
        "runtime_implementation_fingerprint": device[
            "runtime_implementation_fingerprint"
        ],
        **counts,
        "provider_count": len(projected),
        "provider_ids": sorted(provider_ids),
        "projection": projected,
    }


def validate_exporter_stdout(
    stdout_path: Path,
    *,
    provider_catalog: Path,
    capability_catalog: Path,
    provider_count: int,
    capability_fingerprint: str,
) -> dict[str, Any]:
    lines = stdout_path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 1, "catalog exporter stdout must contain exactly one line")
    match = EXPORT_READY_RE.fullmatch(lines[0])
    require(match is not None, "catalog exporter readiness line is malformed")
    assert match is not None
    require(
        Path(match.group("provider")).resolve() == provider_catalog.resolve(),
        "catalog exporter provider path differs from its argv",
    )
    require(
        Path(match.group("capability")).resolve() == capability_catalog.resolve(),
        "catalog exporter capability path differs from its argv",
    )
    require(
        int(match.group("provider_count")) == provider_count,
        "catalog exporter provider count differs from the provider catalog",
    )
    require(
        match.group("capability_fingerprint") == capability_fingerprint,
        "catalog exporter capability fingerprint differs from the capability catalog",
    )
    return {
        "line": lines[0],
        "provider_count": provider_count,
        "capability_fingerprint": capability_fingerprint,
    }


def expected_core_ptx_artifacts(source_root: Path) -> set[str]:
    build_rs = source_root / "crates/ferrum-kernels/build.rs"
    source = build_rs.read_text(encoding="utf-8")
    match = CORE_PTX_BLOCK_RE.search(source)
    require(match is not None, "cannot parse CORE_PTX_KERNELS from ferrum-kernels/build.rs")
    assert match is not None
    paths = QUOTED_RUST_PATH_RE.findall(match.group("body"))
    require(paths and len(paths) == len(set(paths)), "CORE_PTX_KERNELS is empty or duplicated")
    return {f"core-ptx:{path}" for path in paths}


def validate_cache_only_build_summary(
    path: Path,
    *,
    source_root: Path,
) -> dict[str, Any]:
    value = read_json(path, "CUDA build summary")
    require(
        isinstance(value, dict)
        and set(value) == {"schema_version", "artifact_type", "rows"}
        and value.get("schema_version") == 1
        and value.get("artifact_type") == "ferrum_cuda_build_summary_receipt",
        "CUDA build summary schema mismatch",
    )
    rows = value.get("rows")
    require(isinstance(rows, list), "CUDA build summary rows are missing")
    by_artifact: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"CUDA build summary row {index} is invalid")
        artifact = row.get("artifact")
        require(
            isinstance(artifact, str) and artifact and artifact not in by_artifact,
            f"CUDA build summary row {index} artifact is invalid or duplicated",
        )
        by_artifact[artifact] = row
    expected_core = expected_core_ptx_artifacts(source_root)
    observed_core = {key for key in by_artifact if key.startswith("core-ptx:")}
    require(observed_core == expected_core, "CUDA build summary core PTX coverage mismatch")
    require(
        all(
            by_artifact[key].get("status") == "cache_hit"
            and by_artifact[key].get("reason") == "signature-match"
            for key in expected_core
        ),
        "CUDA live catalog build compiled or reused stale core PTX",
    )
    expected_non_core = {
        "native_operator_artifact_set",
        *(unit[0] for unit in REQUIRED_CUDA_NATIVE_BUILD_UNITS),
    }
    require(
        set(by_artifact) == expected_core | expected_non_core,
        "CUDA build summary contains an unexpected build unit",
    )
    return {
        "row_count": len(rows),
        "core_ptx_count": len(expected_core),
        "core_ptx_status": "cache_hit",
        "core_ptx_artifacts_sha256": canonical_json_sha256(sorted(expected_core)),
    }


def bounded_step(
    *,
    root: Path,
    step_id: str,
    cwd: Path,
    command: Sequence[str],
    expected_duration_seconds: int,
    hard_deadline_seconds: int,
    progress_signal: str,
    max_processes: int,
    max_group_threads: int,
    max_per_process_threads: int,
) -> dict[str, Any]:
    step_root = root / step_id
    step_root.mkdir(parents=True, exist_ok=False)
    write_json(
        step_root / "plan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "step_id": step_id,
            "command": list(command),
            "cwd": str(cwd),
            "expected_duration_seconds": expected_duration_seconds,
            "hard_deadline_seconds": hard_deadline_seconds,
            "progress_signal": progress_signal,
            "worker_limits": {
                "max_processes": max_processes,
                "max_group_threads": max_group_threads,
                "max_per_process_threads": max_per_process_threads,
            },
            "started_at": now_iso(),
        },
    )
    wrapper_rc, receipt = bounded_command.run_bounded_command(
        command=list(command),
        cwd=cwd,
        receipt_path=step_root / "bounded.receipt.json",
        stdout_path=step_root / "stdout.log",
        stderr_path=step_root / "stderr.log",
        limits=bounded_command.Limits(
            wall_timeout_seconds=float(hard_deadline_seconds),
            max_processes=max_processes,
            max_group_threads=max_group_threads,
            max_per_process_threads=max_per_process_threads,
            sample_interval_seconds=0.2,
            max_sampling_errors=3,
            term_grace_seconds=3.0,
        ),
    )
    require(
        wrapper_rc == 0
        and receipt.get("status") == "pass"
        and receipt.get("rc") == 0
        and receipt.get("cleanup", {}).get("process_group_gone") is True,
        f"bounded step failed: {step_id}: {step_root / 'bounded.receipt.json'}",
    )
    return receipt


def build_command(
    *,
    cargo: Path,
    rustc: Path,
    nvcc: Path,
    target_dir: Path,
    native_operator_set_lock: Path,
    build_summary_receipt: Path,
    native_build_cache: Path,
    native_import_dirs: Sequence[Path],
    cargo_jobs: int,
) -> list[str]:
    path_entries = [str(cargo.parent), str(rustc.parent), str(nvcc.parent)]
    path_entries.extend(os.environ.get("PATH", "").split(os.pathsep))
    build_path = os.pathsep.join(dict.fromkeys(entry for entry in path_entries if entry))
    return [
        "/usr/bin/env",
        f"PATH={build_path}",
        f"RUSTC={rustc}",
        f"CARGO_TARGET_DIR={target_dir}",
        f"CARGO_BUILD_JOBS={cargo_jobs}",
        "CUDA_COMPUTE_CAP=89",
        "FERRUM_NVCC_THREADS=4",
        f"FERRUM_CUDA_NATIVE_BUILD_CACHE={native_build_cache}",
        "FERRUM_CUDA_NATIVE_IMPORT_DIRS="
        + os.pathsep.join(str(path) for path in native_import_dirs),
        f"FERRUM_NATIVE_OPERATOR_SET_LOCK={native_operator_set_lock}",
        "FERRUM_CUDA_NATIVE_SOURCE_POLICY=cache-only",
        f"FERRUM_CUDA_BUILD_SUMMARY_RECEIPT={build_summary_receipt}",
        str(cargo),
        "build",
        "--profile",
        "cuda-correctness",
        "--locked",
        "--jobs",
        str(cargo_jobs),
        "-p",
        "ferrum-kernels",
        "--example",
        "runtime_vnext_cuda_catalog_input",
        "--features",
        ",".join(PRODUCT_FEATURES),
    ]


def portable_command(
    command: Sequence[str],
    *,
    source_root: Path,
    out: Path,
    target_dir: Path,
    native_build_cache: Path,
    native_import_dirs: Sequence[Path],
) -> list[str]:
    replacements = [
        (str(source_root), "<source-root>"),
        (str(out), "<artifact-root>"),
        (str(target_dir), "<target-dir>"),
        (str(native_build_cache), "<native-build-cache>"),
    ]
    replacements.extend(
        (str(path), f"<native-import-dir-{index}>")
        for index, path in enumerate(native_import_dirs)
    )
    replacements.sort(key=lambda row: len(row[0]), reverse=True)
    result = []
    for argument in command:
        portable = argument
        for raw, replacement in replacements:
            portable = portable.replace(raw, replacement)
        result.append(portable)
    return result


def artifact_index(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"artifact contains symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"raw.manifest.json", "failure.json"}:
            continue
        rows.append(file_identity(path, relative_to=root))
    return rows


def collect(args: argparse.Namespace) -> Path:
    source_root = args.source_root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    native_build_cache = args.native_build_cache.expanduser().resolve()
    native_import_dirs = tuple(
        path.expanduser().resolve() for path in args.native_import_dir
    )
    lock_input = args.native_operator_set_lock.expanduser().absolute()
    require(source_root.is_dir() and (source_root / "Cargo.toml").is_file(), "invalid source root")
    require(not out.exists(), f"output already exists: {out}")
    require(not out.is_relative_to(source_root), "--out must be outside the source tree")
    require(not target_dir.is_relative_to(source_root), "--target-dir must be outside the source tree")
    require(
        not out.is_relative_to(target_dir) and not target_dir.is_relative_to(out),
        "--out and --target-dir must not overlap",
    )
    require(
        native_build_cache.is_dir() and not native_build_cache.is_symlink(),
        f"native build cache is missing or unsafe: {native_build_cache}",
    )
    require(
        all(path.is_dir() and not path.is_symlink() for path in native_import_dirs),
        "every --native-import-dir must be an existing regular directory",
    )
    require(
        len(native_import_dirs) == len(set(native_import_dirs)),
        "--native-import-dir contains duplicates",
    )
    require(1 <= args.cargo_jobs <= 8, "--cargo-jobs must be in [1, 8]")
    require(args.cuda_ordinal == 0, "canonical G03 live catalog requires CUDA ordinal 0")
    require(
        args.attention_policy == "native-adaptive",
        "canonical G03 live catalog requires native-adaptive attention",
    )
    require(
        args.build_timeout_seconds >= args.build_expected_seconds,
        "build timeout must be at least expected duration",
    )
    require(
        args.export_timeout_seconds >= args.export_expected_seconds,
        "export timeout must be at least expected duration",
    )
    reject_hidden_build_overrides()

    source = source_identity(source_root)
    cargo = resolve_tool(args.cargo, "cargo")
    rustc = resolve_tool(args.rustc, "rustc")
    nvcc = resolve_tool(args.nvcc, "nvcc")
    nvidia_smi = resolve_tool(args.nvidia_smi, "nvidia-smi")
    hardware = hardware_identity(
        source_root,
        nvidia_smi=nvidia_smi,
        nvcc=nvcc,
        cargo=cargo,
        rustc=rustc,
    )
    validated_input_lock = validate_native_operator_set(
        lock_input, REQUIRED_CUDA_NATIVE_OPERATORS
    )

    out.mkdir(parents=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = out / "source-snapshot"
    source_snapshot.mkdir()
    collector_snapshot = source_snapshot / COLLECTOR_PATH.name
    shutil.copy2(COLLECTOR_PATH, collector_snapshot)
    staged_lock, staged_set = stage_native_operator_set(
        lock_input,
        out / "bootstrap-native-operator-set",
        REQUIRED_CUDA_NATIVE_OPERATORS,
    )
    require(
        native_operator_set_public_identity(staged_set)
        == native_operator_set_public_identity(validated_input_lock),
        "staged native operator set differs from its input",
    )
    write_json(out / "source.json", source)
    write_json(out / "hardware.json", hardware)
    write_json(
        out / "lane-plan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "lane": "runtime-vnext-g03-live-catalog-raw",
            "source": source,
            "expected_runtime_seconds": (
                args.build_expected_seconds + args.export_expected_seconds
            ),
            "hard_deadline_seconds": (
                args.build_timeout_seconds + args.export_timeout_seconds
            ),
            "hard_stop": "first failed bounded step, source drift, malformed catalog, or native source fallback",
            "correctness_gate": "independent checkpoint must bind both live catalogs to canonical same-SHA S1 evidence",
            "performance_command": "not applicable; this lane collects catalog correctness evidence",
            "progress_signal": "Cargo/export log growth, bounded receipts, and catalog file creation",
            "native_build_cache": str(native_build_cache),
            "native_import_dirs": [str(path) for path in native_import_dirs],
            "does_not_prove": [
                "canonical G03 PASS",
                "canonical G07B PASS",
                "model correctness",
                "model performance",
                "release readiness",
            ],
        },
    )

    build_summary = out / "build/cuda-build-summary.receipt.json"
    command = build_command(
        cargo=cargo,
        rustc=rustc,
        nvcc=nvcc,
        target_dir=target_dir,
        native_operator_set_lock=staged_lock,
        build_summary_receipt=build_summary,
        native_build_cache=native_build_cache,
        native_import_dirs=native_import_dirs,
        cargo_jobs=args.cargo_jobs,
    )
    bounded_step(
        root=out,
        step_id="build",
        cwd=source_root,
        command=command,
        expected_duration_seconds=args.build_expected_seconds,
        hard_deadline_seconds=args.build_timeout_seconds,
        progress_signal="Cargo log bytes or rustc/nvcc/linker CPU activity",
        max_processes=16,
        max_group_threads=192,
        max_per_process_threads=64,
    )
    validated_summary = validate_cuda_build_summary(
        build_summary,
        str(staged_lock),
        staged_set,
        REQUIRED_CUDA_NATIVE_BUILD_UNITS,
    )
    cache_only_summary = validate_cache_only_build_summary(
        build_summary,
        source_root=source_root,
    )
    binary_source = (
        target_dir
        / "cuda-correctness/examples/runtime_vnext_cuda_catalog_input"
    )
    require(binary_source.is_file(), f"catalog exporter binary is missing: {binary_source}")
    binary = out / "binary/runtime_vnext_cuda_catalog_input"
    binary.parent.mkdir(parents=True)
    shutil.copy2(binary_source, binary)

    catalog_root = out / "catalog"
    catalog_root.mkdir()
    provider_catalog = catalog_root / "provider-catalog.json"
    capability_catalog = catalog_root / "capability-catalog.json"
    export_command = [
        str(binary),
        str(args.cuda_ordinal),
        args.attention_policy,
        str(provider_catalog),
        str(capability_catalog),
    ]
    export_receipt = bounded_step(
        root=out,
        step_id="catalog-export",
        cwd=source_root,
        command=export_command,
        expected_duration_seconds=args.export_expected_seconds,
        hard_deadline_seconds=args.export_timeout_seconds,
        progress_signal="CUDA process activity and both catalog files becoming non-empty",
        max_processes=2,
        max_group_threads=32,
        max_per_process_threads=32,
    )
    provider = validate_provider_catalog(provider_catalog)
    capability = validate_capability_catalog(
        capability_catalog, cuda_ordinal=args.cuda_ordinal
    )
    require(
        provider["operation_count"] == capability["operations_count"],
        "provider and capability catalog operation counts differ",
    )
    require(
        provider["operation_ids"]
        == sorted(read_json(capability_catalog, "capability catalog")["operations"]),
        "provider and capability catalog operation IDs differ",
    )
    require(
        provider["projection"] == capability["projection"],
        "provider catalog differs from capability catalog provider projection",
    )
    exporter = validate_exporter_stdout(
        out / "catalog-export/stdout.log",
        provider_catalog=provider_catalog,
        capability_catalog=capability_catalog,
        provider_count=provider["provider_count"],
        capability_fingerprint=capability["runtime_fingerprint"],
    )

    source_after = source_identity(source_root)
    require(source_after == source, "source changed during live catalog collection")
    lock_after = validate_native_operator_set(staged_lock, REQUIRED_CUDA_NATIVE_OPERATORS)
    require(
        native_operator_set_public_identity(lock_after)
        == native_operator_set_public_identity(staged_set),
        "bootstrap native operator set changed during live catalog collection",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "ready",
        "created_at": now_iso(),
        "source": source,
        "hardware": hardware,
        "collector": {
            "source_path": COLLECTOR_PATH.relative_to(source_root).as_posix(),
            **file_identity(collector_snapshot, relative_to=out),
        },
        "scope": {
            "backend": "cuda",
            "gpu_count": 1,
            "gpu_model": "RTX 4090",
            "cuda_ordinal": args.cuda_ordinal,
            "attention_policy": args.attention_policy,
            "cargo_profile": "cuda-correctness",
            "cargo_jobs": args.cargo_jobs,
            "features": list(PRODUCT_FEATURES),
        },
        "bootstrap_native_operator_set": {
            "role": "build bootstrap only; G07B must rebuild artifacts against the exported live catalog",
            "lock": file_identity(staged_lock, relative_to=out),
            **native_operator_set_public_identity(staged_set),
        },
        "build": {
            "command": command,
            "portable_command": portable_command(
                command,
                source_root=source_root,
                out=out,
                target_dir=target_dir,
                native_build_cache=native_build_cache,
                native_import_dirs=native_import_dirs,
            ),
            "receipt": file_identity(out / "build/bounded.receipt.json", relative_to=out),
            "summary": {**validated_summary, **cache_only_summary},
            "summary_receipt": file_identity(build_summary, relative_to=out),
            "native_build_cache": str(native_build_cache),
            "native_import_dirs": [str(path) for path in native_import_dirs],
        },
        "export": {
            "command": export_command,
            "portable_command": portable_command(
                export_command,
                source_root=source_root,
                out=out,
                target_dir=target_dir,
                native_build_cache=native_build_cache,
                native_import_dirs=native_import_dirs,
            ),
            "binary": file_identity(binary, relative_to=out),
            "receipt": file_identity(
                out / "catalog-export/bounded.receipt.json", relative_to=out
            ),
            "receipt_status": export_receipt.get("status"),
            "readiness": exporter,
            "provider_catalog": {
                "path": provider_catalog.relative_to(out).as_posix(),
                "sha256": provider["sha256"],
                "size_bytes": provider["size_bytes"],
                "canonical_fingerprint": provider["canonical_fingerprint"],
                "provider_count": provider["provider_count"],
                "operation_count": provider["operation_count"],
                "provider_ids": provider["provider_ids"],
                "operation_ids": provider["operation_ids"],
            },
            "capability_catalog": {
                "path": capability_catalog.relative_to(out).as_posix(),
                "sha256": capability["sha256"],
                "size_bytes": capability["size_bytes"],
                "canonical_fingerprint": capability["canonical_fingerprint"],
                "device_id": capability["device_id"],
                "runtime_implementation_fingerprint": capability[
                    "runtime_implementation_fingerprint"
                ],
                "operations_count": capability["operations_count"],
                "providers_count": capability["providers_count"],
                "engine_providers_count": capability[
                    "engine_providers_count"
                ],
                "weight_materializers_count": capability[
                    "weight_materializers_count"
                ],
            },
        },
        "does_not_prove": [
            "canonical G03 PASS",
            "full G03 CPU/CUDA/Metal conformance",
            "canonical G07B PASS",
            "model correctness",
            "model performance",
            "release readiness",
        ],
    }
    manifest["artifacts"] = artifact_index(out)
    manifest["artifact_count"] = len(manifest["artifacts"])
    write_json(out / "raw.manifest.json", manifest)
    return out


def self_test() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ferrum-g03-live-catalog-") as raw:
        root = Path(raw)
        tool_root = root / "tools"
        tool_root.mkdir()
        actual_cargo = tool_root / "cargo"
        actual_cargo.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        actual_cargo.chmod(0o755)
        rustup = tool_root / "rustup"
        rustup.write_text(
            "#!/bin/sh\n"
            "test \"$1\" = which || exit 2\n"
            "test \"$2\" = cargo || exit 3\n"
            f"printf '%s\\n' {actual_cargo}\n",
            encoding="utf-8",
        )
        rustup.chmod(0o755)
        cargo_proxy = tool_root / "cargo-proxy"
        cargo_proxy.symlink_to(rustup)
        require(
            resolve_tool(str(cargo_proxy), "cargo") == actual_cargo.resolve(),
            "rustup cargo proxy did not resolve to the actual toolchain binary",
        )
        provider_path = root / "provider.json"
        capability_path = root / "capability.json"
        operation = {
            "id": "operation.fixture",
            "version": {"major": 1, "minor": 0},
        }
        operation_fingerprint = serde_json_fingerprint(operation)
        provider = {
            "schema_version": 1,
            "backend": "cuda",
            "providers": [
                {
                    "operation_id": "operation.fixture",
                    "operation_contract_version": {"major": 1, "minor": 0},
                    "operation_fingerprint": operation_fingerprint,
                    "provider_id": "provider.cuda.fixture",
                    "provider_version": {"major": 1, "minor": 0},
                    "provider_implementation_fingerprint": "2" * 64,
                }
            ],
        }
        capability = {
            "device": {
                "id": "cuda:0",
                "class": "accelerator",
                "ordinal": 0,
                "total_memory_bytes": 1024,
                "runtime_implementation_fingerprint": "3" * 64,
                "capabilities": ["capability.fixture"],
                "dynamic_storage_profiles": [
                    {"allocator": "linear_arena", "view": "contiguous"}
                ],
            },
            "operations": {"operation.fixture": operation},
            "providers": {
                "operation.fixture": [
                    {
                        "provider_id": "provider.cuda.fixture",
                        "operation_id": "operation.fixture",
                        "operation_fingerprint": operation_fingerprint,
                        "provider_implementation_fingerprint": "2" * 64,
                        "version": {"major": 1, "minor": 0},
                    }
                ]
            },
            "engine_providers": {"provider.engine.cuda.vnext": {"provider_id": "provider.engine.cuda.vnext"}},
            "weight_materializers": {"weight-materializer.identity": {"id": "weight-materializer.identity"}},
        }
        write_json(provider_path, provider)
        write_json(capability_path, capability)
        provider_identity = validate_provider_catalog(provider_path)
        capability_identity = validate_capability_catalog(capability_path, cuda_ordinal=0)
        require(provider_identity["provider_count"] == 1, "self-test provider count mismatch")
        require(capability_identity["operations_count"] == 1, "self-test operation count mismatch")
        require(
            provider_identity["projection"] == capability_identity["projection"],
            "self-test provider projection mismatch",
        )

        export_stdout = root / "export.stdout"
        export_stdout.write_text(
            "FERRUM RUNTIME VNEXT CUDA NATIVE CATALOG INPUT READY: "
            f"provider={provider_path} capability={capability_path} "
            "provider_count=1 "
            f"capability_fingerprint={capability_identity['runtime_fingerprint']}\n",
            encoding="utf-8",
        )
        validate_exporter_stdout(
            export_stdout,
            provider_catalog=provider_path,
            capability_catalog=capability_path,
            provider_count=1,
            capability_fingerprint=capability_identity["runtime_fingerprint"],
        )
        export_stdout.write_text(
            export_stdout.read_text(encoding="utf-8").replace(
                capability_identity["runtime_fingerprint"], "9" * 64
            ),
            encoding="utf-8",
        )
        try:
            validate_exporter_stdout(
                export_stdout,
                provider_catalog=provider_path,
                capability_catalog=capability_path,
                provider_count=1,
                capability_fingerprint=capability_identity["runtime_fingerprint"],
            )
        except CollectionError:
            pass
        else:
            raise CollectionError("exporter fingerprint mutation unexpectedly passed")

        duplicate = json.loads(json.dumps(provider))
        duplicate["providers"].append(dict(duplicate["providers"][0]))
        write_json(root / "duplicate.json", duplicate)
        try:
            validate_provider_catalog(root / "duplicate.json")
        except CollectionError:
            pass
        else:
            raise CollectionError("duplicate provider mutation unexpectedly passed")

        zero_major = json.loads(json.dumps(provider))
        zero_major["providers"][0]["provider_version"]["major"] = 0
        write_json(root / "zero-major.json", zero_major)
        try:
            validate_provider_catalog(root / "zero-major.json")
        except CollectionError:
            pass
        else:
            raise CollectionError("zero-major provider mutation unexpectedly passed")

        changed_capability = json.loads(json.dumps(capability))
        changed_capability["providers"]["operation.fixture"][0][
            "provider_implementation_fingerprint"
        ] = "8" * 64
        write_json(root / "changed-capability.json", changed_capability)
        changed_identity = validate_capability_catalog(
            root / "changed-capability.json", cuda_ordinal=0
        )
        require(
            provider_identity["projection"] != changed_identity["projection"],
            "capability provider mutation was not visible in the projection",
        )

        source_root = root / "source"
        build_rs = source_root / "crates/ferrum-kernels/build.rs"
        build_rs.parent.mkdir(parents=True)
        build_rs.write_text(
            'const CORE_PTX_KERNELS: &[&str] = &["kernels/fixture.cu"];\n',
            encoding="utf-8",
        )
        build_rows = [
            {
                "artifact": "native_operator_artifact_set",
                "status": "linked",
                "reason": "manifest-v3-artifact-set-v5-validated",
                "elapsed_ms": 0,
                "inputs_hash": "sha256:" + "1" * 64,
            },
            {
                "artifact": "core-ptx:kernels/fixture.cu",
                "status": "cache_hit",
                "reason": "signature-match",
                "elapsed_ms": 0,
                "inputs_hash": "sha256:" + "2" * 64,
            },
        ]
        for summary_artifact, _unit, _operator in REQUIRED_CUDA_NATIVE_BUILD_UNITS:
            build_rows.append(
                {
                    "artifact": summary_artifact,
                    "status": "artifact",
                    "reason": "native-operator-artifact-set",
                    "elapsed_ms": 0,
                    "inputs_hash": "sha256:" + "3" * 64,
                }
            )
        summary_path = root / "build-summary.json"
        write_json(
            summary_path,
            {
                "schema_version": 1,
                "artifact_type": "ferrum_cuda_build_summary_receipt",
                "rows": build_rows,
            },
        )
        validate_cache_only_build_summary(summary_path, source_root=source_root)
        build_rows[1]["status"] = "built"
        build_rows[1]["reason"] = "missing-ptx"
        write_json(
            root / "built-summary.json",
            {
                "schema_version": 1,
                "artifact_type": "ferrum_cuda_build_summary_receipt",
                "rows": build_rows,
            },
        )
        try:
            validate_cache_only_build_summary(
                root / "built-summary.json", source_root=source_root
            )
        except CollectionError:
            pass
        else:
            raise CollectionError("built core PTX mutation unexpectedly passed")

        command = build_command(
            cargo=Path("/fixture/cargo"),
            rustc=Path("/fixture/rustc"),
            nvcc=Path("/fixture/nvcc"),
            target_dir=Path("/fixture/target"),
            native_operator_set_lock=Path("/fixture/native.lock.json"),
            build_summary_receipt=Path("/fixture/build-summary.json"),
            native_build_cache=Path("/fixture/native-cache"),
            native_import_dirs=(Path("/fixture/import"),),
            cargo_jobs=4,
        )
        require("FERRUM_CUDA_NATIVE_SOURCE_POLICY=cache-only" in command, "cache-only policy missing")
        require(
            "FERRUM_CUDA_NATIVE_BUILD_CACHE=/fixture/native-cache" in command
            and "FERRUM_CUDA_NATIVE_IMPORT_DIRS=/fixture/import" in command,
            "explicit native cache/import contract is missing",
        )
        require("--locked" in command and "--jobs" in command, "bounded Cargo arguments missing")
        require(
            command[-1] == ",".join(PRODUCT_FEATURES),
            "product feature set changed",
        )
        require(SHA256_RE.fullmatch(sha256_file(COLLECTOR_PATH)) is not None, "collector SHA invalid")
    print(SELFTEST_PASS_LINE)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--source-root", type=Path, default=REPO_ROOT)
    result.add_argument("--native-operator-set-lock", type=Path)
    result.add_argument("--out", type=Path)
    result.add_argument("--target-dir", type=Path)
    result.add_argument("--native-build-cache", type=Path)
    result.add_argument(
        "--native-import-dir",
        type=Path,
        action="append",
        default=[],
        help="optional populated core-PTX import directory; may be repeated",
    )
    result.add_argument("--cuda-ordinal", type=int, default=0)
    result.add_argument(
        "--attention-policy",
        choices=("auto", "portable", "native-adaptive"),
        default="native-adaptive",
    )
    result.add_argument("--cargo", default="cargo")
    result.add_argument("--rustc", default="rustc")
    result.add_argument("--nvcc", default="nvcc")
    result.add_argument("--nvidia-smi", default="nvidia-smi")
    result.add_argument("--cargo-jobs", type=int, default=4)
    result.add_argument("--build-expected-seconds", type=int, default=480)
    result.add_argument("--build-timeout-seconds", type=int, default=900)
    result.add_argument("--export-expected-seconds", type=int, default=20)
    result.add_argument("--export-timeout-seconds", type=int, default=120)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    require(args.native_operator_set_lock is not None, "--native-operator-set-lock is required")
    require(args.out is not None, "--out is required")
    require(args.target_dir is not None, "--target-dir is required")
    require(args.native_build_cache is not None, "--native-build-cache is required")
    out = args.out.expanduser().resolve()
    try:
        result = collect(args)
    except (
        CollectionError,
        NativeOperatorSetEvidenceError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        if out.exists() and out.is_dir():
            try:
                rows = artifact_index(out)
            except CollectionError:
                rows = []
            write_json(
                out / "failure.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": f"{ARTIFACT_TYPE}_failure",
                    "status": "reject",
                    "created_at": now_iso(),
                    "error": str(error),
                    "artifacts": rows,
                },
            )
        print(f"FERRUM RUNTIME VNEXT G03 LIVE CATALOG COLLECTOR REJECT: {out}: {error}", file=sys.stderr)
        return 1
    print(f"{EVIDENCE_READY_LINE}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
