#!/usr/bin/env python3
"""Build and verify the G07B native-operator chain on one RTX 4090.

This runner produces a KEEP/REJECT diagnostic artifact. It deliberately does
not print the canonical G07B PASS line because G07B also consumes canonical
G03 and G07A child manifests.
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import bounded_command
import validate_runtime_vnext_g07b_native_chain as independent_validator


SCHEMA_VERSION = independent_validator.SCHEMA_VERSION
ARTIFACT_FEATURES = (
    "cuda",
    "vllm-marlin",
    "vllm-moe-marlin",
    "vllm-paged-attn-v2",
    "native-op-artifact",
)
PACKAGES = (
    "marlin",
    "vllm-marlin",
    "vllm-moe-marlin",
    "vllm-paged-attention-v2",
)
EXPECTED_OPERATORS = {
    "ferrum.cuda.marlin",
    "ferrum.cuda.vllm_marlin",
    "ferrum.cuda.vllm_moe_marlin",
    "ferrum.cuda.vllm_paged_attention_v2",
}
EXPECTED_ARTIFACT_BUILD_UNITS = {
    "marlin",
    "vllm_marlin",
    "vllm_moe_marlin",
    "vllm_paged_attn",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CUDA_BUILD_SUMMARY_RECEIPT_SCHEMA_VERSION = 1
CUDA_INPUTS_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ChainError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ChainError(message)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path, label: str) -> Any:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ChainError(f"cannot read {label} {path}: {error}") from error


def run_text(cwd: Path, command: Sequence[str], timeout: int = 30) -> str:
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
        f"command failed ({result.returncode}): {command!r}: {result.stderr[-1000:]}",
    )
    return result.stdout.strip()


def resolve_tool(raw: str, label: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        found = shutil.which(raw)
        if found is None:
            raise ChainError(f"{label} is not on PATH: {raw}")
        resolved = Path(found).resolve()
    require(resolved.is_file() and os.access(resolved, os.X_OK), f"{label} is not executable: {resolved}")
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
    require(not status, f"G07B native chain requires clean source: {status}")
    return {
        "git_sha": git_sha,
        "git_tree_sha": tree_sha,
        "dirty": False,
        "status_short": [],
    }


def hardware_identity(source_root: Path, nvcc: Path, tools: dict[str, Path]) -> dict[str, Any]:
    names = run_text(
        source_root,
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    ).splitlines()
    require(len(names) == 1, f"G07B requires exactly one GPU, found {names}")
    require("RTX 4090" in names[0], f"G07B requires one RTX 4090, found {names[0]!r}")
    query = run_text(
        source_root,
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ],
    )
    return {
        "gpu_count": 1,
        "gpu": query,
        "nvidia_smi": run_text(source_root, ["nvidia-smi"]),
        "nvcc_version": run_text(source_root, [str(nvcc), "--version"]),
        "rustc": run_text(source_root, ["rustc", "-vV"]),
        "cargo": run_text(source_root, ["cargo", "-V"]),
        "tools": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for name, path in tools.items()
        },
    }


def run_step(
    *,
    root: Path,
    step_id: str,
    cwd: Path,
    command: Sequence[str],
    expected_duration_seconds: int,
    deadline_seconds: int,
    progress_signal: str,
    lane_deadline_monotonic: float,
) -> dict[str, Any]:
    remaining_seconds = int(lane_deadline_monotonic - time.monotonic())
    require(
        remaining_seconds >= expected_duration_seconds,
        f"G07B lane has insufficient time for {step_id}: "
        f"remaining={remaining_seconds}s expected={expected_duration_seconds}s",
    )
    deadline_seconds = min(deadline_seconds, remaining_seconds)
    step_root = root / "steps" / step_id
    step_root.mkdir(parents=True, exist_ok=False)
    write_json(
        step_root / "plan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "step_id": step_id,
            "command": list(command),
            "cwd": str(cwd),
            "expected_duration_seconds": expected_duration_seconds,
            "hard_deadline_seconds": deadline_seconds,
            "progress_signal": progress_signal,
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
            wall_timeout_seconds=float(deadline_seconds),
            max_processes=96,
            max_group_threads=256,
            max_per_process_threads=64,
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
        f"G07B step failed: {step_id}: {step_root / 'bounded.receipt.json'}",
    )
    return receipt


def cargo_build_command(
    *,
    target_dir: Path,
    cargo_jobs: int,
    features: Sequence[str],
    artifact_lock: Path,
    native_cache: Path,
    compute_capability: str,
    nvcc_threads: int,
    build_summary_receipt: Path | None,
) -> list[str]:
    environment = [
        "env",
        f"CARGO_TARGET_DIR={target_dir}",
        f"CARGO_BUILD_JOBS={cargo_jobs}",
        f"CUDA_COMPUTE_CAP={compute_capability.removeprefix('sm_')}",
        f"FERRUM_NVCC_THREADS={nvcc_threads}",
        f"FERRUM_CUDA_NATIVE_BUILD_CACHE={native_cache}",
    ]
    environment.extend(
        [
            f"FERRUM_NATIVE_OPERATOR_SET_LOCK={artifact_lock}",
            "FERRUM_CUDA_NATIVE_SOURCE_POLICY=cache-only",
        ]
    )
    if build_summary_receipt is not None:
        environment.append(f"FERRUM_CUDA_BUILD_SUMMARY_RECEIPT={build_summary_receipt}")
    return [
        *environment,
        "cargo",
        "build",
        "--profile",
        "cuda-correctness",
        "-p",
        "ferrum-kernels",
        "--example",
        "runtime_vnext_cuda_catalog",
        "--features",
        ",".join(features),
    ]


def run_catalog_export(
    *,
    root: Path,
    step_id: str,
    source_root: Path,
    binary: Path,
    output_root: Path,
    cuda_ordinal: int,
    attention_policy: str,
    lane_deadline_monotonic: float,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "provider": output_root / "provider-catalog.json",
        "capability": output_root / "capability-catalog.json",
        "inventory": output_root / "compiled-native-operators.json",
    }
    run_step(
        root=root,
        step_id=step_id,
        cwd=source_root,
        command=[
            str(binary),
            str(cuda_ordinal),
            attention_policy,
            str(paths["provider"]),
            str(paths["capability"]),
            str(paths["inventory"]),
        ],
        expected_duration_seconds=20,
        deadline_seconds=120,
        progress_signal="catalog files are created and CUDA process activity remains visible",
        lane_deadline_monotonic=lane_deadline_monotonic,
    )
    for label, path in paths.items():
        require(path.is_file() and path.stat().st_size > 0, f"{label} output is missing: {path}")
    return paths


def artifact_index(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"G07B artifact tree contains a symlink: {path}")
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


def validate_chain(
    *,
    root: Path,
    source: dict[str, Any],
    hardware: dict[str, Any],
    native_source_root: Path,
    catalog_input_path: Path,
    artifact_paths: dict[str, Path],
    lock_path: Path,
    abi_contract_path: Path,
    artifact_binary: Path,
) -> dict[str, Any]:
    artifact_inventory = read_json(artifact_paths["inventory"], "artifact native inventory")
    require(isinstance(artifact_inventory, list), "artifact native inventory must be an array")
    operators = {
        row.get("operator")
        for row in artifact_inventory
        if isinstance(row, dict)
    }
    require(operators == EXPECTED_OPERATORS, f"artifact operator set mismatch: {operators}")

    provider_sha = sha256(catalog_input_path)
    require(
        catalog_input_path.read_bytes() == artifact_paths["provider"].read_bytes(),
        "artifact runtime provider catalog differs from the explicit G03 catalog input",
    )
    for row in artifact_inventory:
        require(isinstance(row, dict), "artifact inventory row must be an object")
        require(row.get("schema_version") == 3, "artifact inventory manifest schema must be 3")
        require(
            row.get("g03_catalog_sha256") == provider_sha,
            f"artifact {row.get('operator')} does not bind the live provider catalog",
        )

    lock = read_json(lock_path, "native operator artifact-set lock")
    require(isinstance(lock, dict), "native operator artifact-set lock must be an object")
    require(lock.get("schema_version") == 5, "native operator artifact-set schema must be 5")
    require(lock.get("g03_catalog_sha256") == provider_sha, "artifact-set catalog pin mismatch")
    lock_artifacts = lock.get("artifacts")
    require(isinstance(lock_artifacts, list) and len(lock_artifacts) == 4, "artifact-set must contain four packages")
    require(
        {row.get("operator") for row in lock_artifacts if isinstance(row, dict)}
        == EXPECTED_OPERATORS,
        "artifact-set operator set mismatch",
    )

    abi_sha = sha256(abi_contract_path)
    require(SHA256_RE.fullmatch(abi_sha) is not None, "ABI contract SHA256 is invalid")
    require(
        all(row.get("abi_contract_sha256") == abi_sha for row in artifact_inventory),
        "compiled native inventory ABI contract pin mismatch",
    )

    build_summary_path = root / "artifact-build-summary.receipt.json"
    build_summary_receipt = read_json(
        build_summary_path, "artifact build summary receipt"
    )
    require(
        isinstance(build_summary_receipt, dict)
        and build_summary_receipt.get("schema_version")
        == CUDA_BUILD_SUMMARY_RECEIPT_SCHEMA_VERSION
        and build_summary_receipt.get("artifact_type")
        == "ferrum_cuda_build_summary_receipt",
        "artifact build summary receipt identity mismatch",
    )
    raw_summaries = build_summary_receipt.get("rows")
    require(isinstance(raw_summaries, list), "artifact build summary rows must be an array")
    summaries = []
    for index, raw_row in enumerate(raw_summaries):
        require(isinstance(raw_row, dict), f"artifact build summary row {index} must be an object")
        require(
            set(raw_row)
            == {"artifact", "status", "reason", "elapsed_ms", "inputs_hash"},
            f"artifact build summary row {index} shape mismatch",
        )
        require(
            all(isinstance(raw_row.get(key), str) and raw_row[key] for key in ("artifact", "status", "reason")),
            f"artifact build summary row {index} text fields are invalid",
        )
        require(
            isinstance(raw_row.get("elapsed_ms"), int) and raw_row["elapsed_ms"] >= 0,
            f"artifact build summary row {index} elapsed_ms is invalid",
        )
        require(
            isinstance(raw_row.get("inputs_hash"), str)
            and CUDA_INPUTS_HASH_RE.fullmatch(raw_row["inputs_hash"]) is not None,
            f"artifact build summary row {index} inputs_hash is invalid",
        )
        summaries.append(raw_row)
    artifact_rows = {
        row["artifact"]: row
        for row in summaries
        if row["artifact"] in EXPECTED_ARTIFACT_BUILD_UNITS
    }
    require(
        set(artifact_rows) == EXPECTED_ARTIFACT_BUILD_UNITS,
        f"artifact build summary coverage mismatch: {sorted(artifact_rows)}",
    )
    require(
        all(
            sum(row["artifact"] == unit for row in summaries) == 1
            for unit in EXPECTED_ARTIFACT_BUILD_UNITS
        ),
        "artifact build summary contains duplicate native build-unit decisions",
    )
    require(
        all(
            row["status"] == "artifact"
            and row["reason"] == "native-operator-artifact-set"
            for row in artifact_rows.values()
        ),
        f"artifact build unexpectedly used native source: {artifact_rows}",
    )
    artifact_set_rows = [
        row for row in summaries if row["artifact"] == "native_operator_artifact_set"
    ]
    require(
        len(artifact_set_rows) == 1 and artifact_set_rows[0]["status"] == "linked",
        "artifact build did not resolve and link exactly one native operator artifact set",
    )
    require(
        not any(row["status"] == "rejected" for row in summaries),
        "artifact build summary contains a rejected build decision",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g07b_native_chain_manifest",
        "status": "keep",
        "created_at": now_iso(),
        "source": source,
        "hardware": hardware,
        "native_source": {
            "root": str(native_source_root),
            "external_to_repository": True,
        },
        "scope": {
            "backend": "cuda",
            "gpu_count": 1,
            "gpu_model": "RTX 4090",
            "compute_capability": "sm_89",
            "source_build_units": list(PACKAGES),
            "artifact_features": list(ARTIFACT_FEATURES),
            "operators": sorted(EXPECTED_OPERATORS),
        },
        "catalog": {
            "provider_sha256": provider_sha,
            "provider_identity_unchanged": True,
            "input_kind": "explicit-g03-provider-catalog",
            "artifact_native_operator_count": len(artifact_inventory),
        },
        "abi_contract": {
            "path": str(abi_contract_path),
            "sha256": abi_sha,
        },
        "artifact_set": {
            "path": str(lock_path),
            "sha256": sha256(lock_path),
            "schema_version": 5,
            "operator_count": len(lock_artifacts),
        },
        "binaries": {
            "artifact": {
                "path": str(artifact_binary),
                "sha256": sha256(artifact_binary),
            },
        },
        "artifact_build_summaries": summaries,
        "artifact_build_summary_receipt": {
            "path": build_summary_path.relative_to(root).as_posix(),
            "sha256": sha256(build_summary_path),
            "schema_version": CUDA_BUILD_SUMMARY_RECEIPT_SCHEMA_VERSION,
        },
        "does_not_prove": [
            "canonical G03 PASS",
            "canonical G07A PASS",
            "canonical G07B PASS",
            "G07 aggregate PASS",
            "model correctness",
            "model performance",
            "release readiness",
        ],
    }


def execute(args: argparse.Namespace) -> Path:
    source_root = args.source_root.expanduser().resolve()
    native_source_root = args.native_source_root.expanduser().resolve()
    g03_provider_catalog = args.g03_provider_catalog.expanduser().resolve()
    out = args.out.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    object_cache = args.object_cache.expanduser().resolve()
    native_cache = args.native_cache.expanduser().resolve()
    require(source_root.is_dir(), f"source root is missing: {source_root}")
    require(
        native_source_root.is_dir()
        and not native_source_root.is_relative_to(source_root),
        "--native-source-root must be an existing directory outside the Git source tree",
    )
    require(
        g03_provider_catalog.is_file() and not g03_provider_catalog.is_symlink(),
        f"G03 provider catalog is missing: {g03_provider_catalog}",
    )
    require(not out.exists(), f"output already exists: {out}")
    require(not out.is_relative_to(source_root), "--out must be outside the Git source tree")
    require(args.compute_capability == "sm_89", "G07B fixed lane requires --compute-capability sm_89")
    require(1 <= args.cargo_jobs <= 16, "--cargo-jobs must be in [1, 16]")
    require(args.nvcc_threads == 4, "G07B fixed lane requires --nvcc-threads 4")
    require(
        args.hard_timeout_seconds >= args.expected_runtime_seconds,
        "--hard-timeout-seconds must be at least --expected-runtime-seconds",
    )
    lane_deadline_monotonic = time.monotonic() + args.hard_timeout_seconds
    out.mkdir(parents=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    object_cache.mkdir(parents=True, exist_ok=True)
    native_cache.mkdir(parents=True, exist_ok=True)

    source = source_identity(source_root)
    tools = {
        "nvcc": resolve_tool(args.nvcc, "nvcc"),
        "ccbin": resolve_tool(args.ccbin, "CUDA host compiler"),
        "cc": resolve_tool(args.cc, "descriptor compiler"),
        "ar": resolve_tool(args.ar, "archiver"),
    }
    cuda_toolkit_root = args.cuda_toolkit_root.expanduser().resolve()
    require(cuda_toolkit_root.is_dir(), f"CUDA toolkit root is missing: {cuda_toolkit_root}")
    hardware = hardware_identity(source_root, tools["nvcc"], tools)
    write_json(out / "source.json", source)
    write_json(out / "hardware.json", hardware)
    abi_contract = out / "contracts/ferrum-native-abi-v2.json"
    abi_contract.parent.mkdir(parents=True)
    shutil.copy2(
        source_root / "native-operators/abi/ferrum-native-abi-v2.json",
        abi_contract,
    )
    catalog_input = out / "catalog-input/provider-catalog.json"
    catalog_input.parent.mkdir(parents=True)
    shutil.copy2(g03_provider_catalog, catalog_input)
    write_json(
        out / "lane-plan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "lane": "runtime-vnext-g07b-native-chain",
            "source": source,
            "expected_runtime_seconds": args.expected_runtime_seconds,
            "hard_deadline_seconds": args.hard_timeout_seconds,
            "hard_stop": "first failed bounded step, catalog drift, source fallback, or runtime binding rejection",
            "correctness_gate": "artifact catalog must equal the explicit G03 input; runtime inventory must contain exactly four verified operators",
            "performance_command": "not applicable; G07B validates build/runtime selection, not model throughput",
            "progress_signal": "per-step bounded receipts, log byte growth, and newly materialized package/artifact files",
        },
    )

    run_step(
        root=out,
        step_id="builder-build",
        cwd=source_root,
        command=[
            "env",
            f"CARGO_TARGET_DIR={target_dir}",
            f"CARGO_BUILD_JOBS={args.cargo_jobs}",
            "cargo",
            "build",
            "--release",
            "-p",
            "ferrum-native-ops-builder",
            "--bin",
            "ferrum-native-ops-builder",
        ],
        expected_duration_seconds=120,
        deadline_seconds=600,
        progress_signal="Cargo stdout/stderr log growth and rustc CPU activity",
        lane_deadline_monotonic=lane_deadline_monotonic,
    )
    builder = target_dir / "release/ferrum-native-ops-builder"
    require(builder.is_file() and os.access(builder, os.X_OK), f"builder binary is missing: {builder}")

    specs_root = out / "package-specs"
    specs_root.mkdir()
    for name in PACKAGES:
        run_step(
            root=out,
            step_id=f"materialize-{name}",
            cwd=source_root,
            command=[
                str(builder),
                "materialize-package-spec",
                "--definition",
                str(source_root / f"native-operators/cuda/package-definitions/{name}.json"),
                "--g03-catalog",
                str(catalog_input),
                "--out",
                str(specs_root / f"{name}.json"),
            ],
            expected_duration_seconds=5,
            deadline_seconds=60,
            progress_signal="package spec output creation",
            lane_deadline_monotonic=lane_deadline_monotonic,
        )

    source_build_root = out / "source-builds"
    packages_root = out / "packages"
    source_build_root.mkdir()
    packages_root.mkdir()
    for name in PACKAGES:
        build_out = source_build_root / name
        run_step(
            root=out,
            step_id=f"source-build-{name}",
            cwd=source_root,
            command=[
                str(builder),
                "source-build",
                "--plan",
                str(source_root / f"native-operators/cuda/source-locks/{name}.plan.json"),
                "--source-root",
                str(native_source_root),
                "--compute-capability",
                args.compute_capability,
                "--builder-sha",
                source["git_sha"],
                "--nvcc",
                str(tools["nvcc"]),
                "--cuda-toolkit-root",
                str(cuda_toolkit_root),
                "--ccbin",
                str(tools["ccbin"]),
                "--ar",
                str(tools["ar"]),
                "--nvcc-threads",
                str(args.nvcc_threads),
                "--object-cache",
                str(object_cache),
                "--out",
                str(build_out),
            ],
            expected_duration_seconds=args.source_build_expected_seconds,
            deadline_seconds=args.source_build_timeout_seconds,
            progress_signal="source-build log growth, nvcc CPU activity, or object-cache file growth",
            lane_deadline_monotonic=lane_deadline_monotonic,
        )
        package_out = packages_root / name
        run_step(
            root=out,
            step_id=f"package-{name}",
            cwd=source_root,
            command=[
                str(builder),
                "package",
                "--spec",
                str(specs_root / f"{name}.json"),
                "--source-root",
                str(native_source_root),
                "--license-root",
                str(source_root),
                "--source-build-receipt",
                str(build_out / "source-build.receipt.json"),
                "--source-build-plan",
                str(source_root / f"native-operators/cuda/source-locks/{name}.plan.json"),
                "--g03-catalog",
                str(catalog_input),
                "--abi-contract",
                str(source_root / "native-operators/abi/ferrum-native-abi-v2.json"),
                "--out",
                str(package_out),
                "--cc",
                str(tools["cc"]),
                "--ar",
                str(tools["ar"]),
            ],
            expected_duration_seconds=30,
            deadline_seconds=180,
            progress_signal="package receipt, descriptor object, archive, and provenance file creation",
            lane_deadline_monotonic=lane_deadline_monotonic,
        )

    receipt_paths = [packages_root / name / "package.receipt.json" for name in PACKAGES]
    receipt_hashes = [sha256(path) for path in receipt_paths]
    provider_sha = sha256(catalog_input)
    lock_path = out / "native-operators.lock.json"
    assemble_command = [str(builder), "assemble-set"]
    for receipt in receipt_paths:
        assemble_command.extend(["--receipt", str(receipt)])
    for receipt_sha in receipt_hashes:
        assemble_command.extend(["--receipt-sha256", receipt_sha])
    assemble_command.extend(
        [
            "--g03-catalog-sha256",
            provider_sha,
            "--compute-capability",
            args.compute_capability,
            "--out",
            str(lock_path),
        ]
    )
    build_summary_receipt = out / "artifact-build-summary.receipt.json"
    run_step(
        root=out,
        step_id="assemble-artifact-set",
        cwd=source_root,
        command=assemble_command,
        expected_duration_seconds=15,
        deadline_seconds=120,
        progress_signal="artifact-set lock creation",
        lane_deadline_monotonic=lane_deadline_monotonic,
    )

    run_step(
        root=out,
        step_id="artifact-example-build",
        cwd=source_root,
        command=cargo_build_command(
            target_dir=target_dir,
            cargo_jobs=args.cargo_jobs,
            features=ARTIFACT_FEATURES,
            artifact_lock=lock_path,
            native_cache=native_cache,
            compute_capability=args.compute_capability,
            nvcc_threads=args.nvcc_threads,
            build_summary_receipt=build_summary_receipt,
        ),
        expected_duration_seconds=args.artifact_build_expected_seconds,
        deadline_seconds=args.artifact_build_timeout_seconds,
        progress_signal="Cargo log growth and linker activity; native source compilation is a stop condition",
        lane_deadline_monotonic=lane_deadline_monotonic,
    )
    built_example = target_dir / "cuda-correctness/examples/runtime_vnext_cuda_catalog"
    require(built_example.is_file(), f"artifact catalog example is missing: {built_example}")
    artifact_binary = out / "binaries/artifact/runtime_vnext_cuda_catalog"
    artifact_binary.parent.mkdir(parents=True)
    shutil.copy2(built_example, artifact_binary)
    artifact_paths = run_catalog_export(
        root=out,
        step_id="artifact-catalog-export",
        source_root=source_root,
        binary=artifact_binary,
        output_root=out / "artifact",
        cuda_ordinal=args.cuda_ordinal,
        attention_policy=args.attention_policy,
        lane_deadline_monotonic=lane_deadline_monotonic,
    )

    current_source = source_identity(source_root)
    require(current_source == source, "source identity changed during G07B native chain")
    manifest = validate_chain(
        root=out,
        source=source,
        hardware=hardware,
        native_source_root=native_source_root,
        catalog_input_path=catalog_input,
        artifact_paths=artifact_paths,
        lock_path=lock_path,
        abi_contract_path=abi_contract,
        artifact_binary=artifact_binary,
    )
    manifest["artifacts"] = artifact_index(out)
    manifest["artifact_count"] = len(manifest["artifacts"])
    write_json(out / "chain.manifest.json", manifest)
    independent_validator.verify_manifest(out, source_root, native_source_root)
    return out


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--native-source-root", type=Path, required=True)
    result.add_argument("--g03-provider-catalog", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--target-dir", type=Path, required=True)
    result.add_argument("--object-cache", type=Path, required=True)
    result.add_argument("--native-cache", type=Path, required=True)
    result.add_argument("--cuda-toolkit-root", type=Path, default=Path("/usr/local/cuda"))
    result.add_argument("--nvcc", default="nvcc")
    result.add_argument("--ccbin", default="g++")
    result.add_argument("--cc", default="cc")
    result.add_argument("--ar", default="ar")
    result.add_argument("--compute-capability", default="sm_89")
    result.add_argument("--cuda-ordinal", type=int, default=0)
    result.add_argument("--attention-policy", choices=("auto", "portable", "native-adaptive"), default="auto")
    result.add_argument("--cargo-jobs", type=int, default=4)
    result.add_argument("--nvcc-threads", type=int, choices=(4,), default=4)
    result.add_argument("--expected-runtime-seconds", type=int, default=3600)
    result.add_argument("--hard-timeout-seconds", type=int, default=5400)
    result.add_argument("--source-build-expected-seconds", type=int, default=300)
    result.add_argument("--source-build-timeout-seconds", type=int, default=1200)
    result.add_argument("--artifact-build-expected-seconds", type=int, default=120)
    result.add_argument("--artifact-build-timeout-seconds", type=int, default=900)
    return result


def main() -> int:
    args = parser().parse_args()
    out = args.out.expanduser().resolve()
    try:
        result = execute(args)
    except (
        ChainError,
        OSError,
        independent_validator.VerificationError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        if out.exists() and out.is_dir():
            failure_index_error = None
            try:
                failure_artifacts = artifact_index(out)
            except ChainError as index_error:
                failure_artifacts = []
                failure_index_error = str(index_error)
            write_json(
                out / "failure.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "runtime_vnext_g07b_native_chain_failure",
                    "status": "reject",
                    "created_at": now_iso(),
                    "error": str(error),
                    "artifact_index_error": failure_index_error,
                    "artifacts": failure_artifacts,
                },
            )
        print(f"FERRUM RUNTIME VNEXT G07B NATIVE CHAIN REJECT: {out}: {error}", file=sys.stderr)
        return 1
    print(f"FERRUM RUNTIME VNEXT G07B NATIVE CHAIN KEEP: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
