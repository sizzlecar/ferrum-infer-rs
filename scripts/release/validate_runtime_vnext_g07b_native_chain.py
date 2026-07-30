#!/usr/bin/env python3
"""Independently verify a copied G07B native-operator chain artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 1
RECEIPT_SCHEMA = "ferrum.bounded-command-receipt.v1"
PACKAGES = {
    "marlin": "ferrum.cuda.marlin",
    "vllm-marlin": "ferrum.cuda.vllm_marlin",
    "vllm-moe-marlin": "ferrum.cuda.vllm_moe_marlin",
    "vllm-paged-attention-v2": "ferrum.cuda.vllm_paged_attention_v2",
}
SOURCE_FEATURES = [
    "cuda",
    "vllm-marlin",
    "vllm-moe-marlin",
    "vllm-paged-attn-v2",
]
ARTIFACT_FEATURES = [*SOURCE_FEATURES, "native-op-artifact"]
EXPECTED_BUILD_UNITS = {
    "marlin",
    "vllm_marlin",
    "vllm_moe_marlin",
    "vllm_paged_attn",
}
EXPECTED_STEPS = {
    "builder-build",
    "bootstrap-example-build",
    "bootstrap-catalog-export",
    "assemble-artifact-set",
    "artifact-example-build",
    "artifact-catalog-export",
    *(f"materialize-{name}" for name in PACKAGES),
    *(f"source-build-{name}" for name in PACKAGES),
    *(f"package-{name}" for name in PACKAGES),
}
DOES_NOT_PROVE = {
    "canonical G03 PASS",
    "canonical G07A PASS",
    "canonical G07B PASS",
    "G07 aggregate PASS",
    "model correctness",
    "model performance",
    "release readiness",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CUDA_SUMMARY_RE = re.compile(
    r"\[cuda-build-summary\]\s+artifact=(\S+)\s+status=(\S+)\s+reason=(\S+)"
)


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


def read_json(path: Path, label: str) -> Any:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {label} {path}: {error}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    require(lane_plan.get("schema_version") == 1, "lane plan schema mismatch")
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


def verify_step(root: Path, step_id: str) -> None:
    step_root = root / "steps" / step_id
    require(step_root.is_dir() and not step_root.is_symlink(), f"step is missing: {step_id}")
    plan = require_dict(read_json(step_root / "plan.json", f"{step_id} plan"), f"{step_id} plan")
    receipt = require_dict(
        read_json(step_root / "bounded.receipt.json", f"{step_id} receipt"),
        f"{step_id} receipt",
    )
    require(plan.get("schema_version") == 1 and plan.get("step_id") == step_id, f"{step_id} plan identity mismatch")
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
        receipt.get("schema_version") == 4
        and receipt.get("status") == "pass"
        and receipt.get("plan_only") is False
        and receipt.get("failure_class") is None,
        f"{name} source build is not a terminal schema-v4 PASS",
    )
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
    static_identity = require_dict(receipt["toolchain"].get("static_identity"), f"{name}.toolchain.static_identity")
    cuda_toolkit = require_dict(static_identity.get("cuda_toolkit"), f"{name}.toolchain.cuda_toolkit")
    host_toolchain = require_dict(static_identity.get("host_toolchain"), f"{name}.toolchain.host_toolchain")
    verify_evidence(build_root, cuda_toolkit.get("manifest"), f"{name}.cuda_toolkit.manifest")
    verify_evidence(build_root, host_toolchain.get("manifest"), f"{name}.host_toolchain.manifest")

    commands = require_list(receipt.get("commands"), f"{name}.commands")
    require(bool(commands), f"{name} source build commands must not be empty")
    for index, raw in enumerate(commands):
        command = require_dict(raw, f"{name}.commands[{index}]")
        object_file = command.get("object_file")
        if object_file is None:
            require(
                command.get("translation_unit") is None
                and command.get("compiler_executed") is False
                and command.get("return_code") == 0
                and command.get("object_cache_status") is None,
                f"{name}.commands[{index}] archive execution mismatch",
            )
        elif command.get("compiler_executed") is True:
            require(
                command.get("return_code") == 0
                and command.get("object_cache_status") == "published"
                and command.get("dependency_validation") == "depfile",
                f"{name}.commands[{index}] compiler execution mismatch",
            )
        else:
            require(
                command.get("compiler_executed") is False
                and command.get("return_code") is None
                and command.get("object_cache_status") == "hit"
                and command.get("dependency_validation") == "cache_proof",
                f"{name}.commands[{index}] cache-hit execution mismatch",
            )
        for stream in ("stdout_log", "stderr_log"):
            path = resolve_relative_file(build_root, command.get(stream), f"{name}.commands[{index}].{stream}")
            require(path.stat().st_size > 0, f"{name}.commands[{index}].{stream} is empty")
        depfile = command.get("depfile")
        if depfile is not None:
            depfile_path = resolve_relative_file(build_root, depfile, f"{name}.commands[{index}].depfile")
            require_sha(command.get("depfile_sha256"), f"{name}.commands[{index}].depfile_sha256")
            require(sha256(depfile_path) == command["depfile_sha256"], f"{name}.commands[{index}] depfile SHA mismatch")
        if object_file is not None:
            object_path = build_root / "objects" / Path(object_file).name
            require(object_path.is_file() and not object_path.is_symlink(), f"{name} object is missing: {object_path}")
            require_sha(command.get("object_sha256"), f"{name}.commands[{index}].object_sha256")
            require(sha256(object_path) == command["object_sha256"], f"{name}.commands[{index}] object SHA mismatch")
            require(
                object_path.stat().st_size == command.get("object_size_bytes"),
                f"{name}.commands[{index}] object size mismatch",
            )

    translation_units = [row["path"] for row in require_list(plan.get("translation_units"), f"{name}.plan.translation_units")]
    compiled = require_list(receipt.get("compiled_translation_units"), f"{name}.compiled_translation_units")
    hits = require_list(receipt.get("cache_hit_translation_units"), f"{name}.cache_hit_translation_units")
    require(not set(compiled) & set(hits), f"{name} translation unit is both compiled and cache-hit")
    require(set(compiled) | set(hits) == set(translation_units), f"{name} source-build coverage mismatch")
    archive_file = receipt.get("archive_file")
    require(archive_file == plan.get("archive_file"), f"{name} archive filename mismatch")
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
        lock_row = lock_by_operator[operator]
        for field in (
            "operator_abi_version",
            "ferrum_native_abi_version",
            "backend",
            "linkage",
            "g03_catalog_sha256",
            "abi_contract_sha256",
            "descriptor_export",
            "operation_bindings",
            "required_exports",
            "source_package_sha256",
            "inputs_sha256",
            "binary_sha256",
        ):
            inventory_field = "exports" if field == "required_exports" else field
            require(
                row.get(inventory_field) == lock_row.get(field),
                f"compiled inventory {operator}.{inventory_field} differs from the lock",
            )
        require(row.get("schema_version") == 3, f"compiled inventory {operator} schema mismatch")
    require(set(inventory_by_operator) == set(PACKAGES.values()), "compiled inventory coverage mismatch")


def verify_catalogs_and_packages(
    root: Path,
    source_root: Path,
    manifest: dict[str, Any],
    source: dict[str, Any],
) -> None:
    bootstrap_provider = root / "bootstrap/provider-catalog.json"
    artifact_provider = root / "artifact/provider-catalog.json"
    bootstrap_capability = root / "bootstrap/capability-catalog.json"
    artifact_capability = root / "artifact/capability-catalog.json"
    bootstrap_inventory_path = root / "bootstrap/compiled-native-operators.json"
    artifact_inventory_path = root / "artifact/compiled-native-operators.json"
    for path in (
        bootstrap_provider,
        artifact_provider,
        bootstrap_capability,
        artifact_capability,
        bootstrap_inventory_path,
        artifact_inventory_path,
    ):
        require(path.is_file() and not path.is_symlink(), f"runtime export is missing: {path}")
    require(bootstrap_provider.read_bytes() == artifact_provider.read_bytes(), "provider catalogs differ")
    require(bootstrap_capability.read_bytes() == artifact_capability.read_bytes(), "capability catalogs differ")
    provider_bytes = bootstrap_provider.read_bytes()
    provider_sha = sha256(bootstrap_provider)
    providers = verify_provider_catalog(read_json(bootstrap_provider, "bootstrap provider catalog"))
    bootstrap_inventory = require_list(
        read_json(bootstrap_inventory_path, "bootstrap compiled inventory"),
        "bootstrap compiled inventory",
    )
    artifact_inventory = require_list(
        read_json(artifact_inventory_path, "artifact compiled inventory"),
        "artifact compiled inventory",
    )
    require(bootstrap_inventory == [], "bootstrap binary unexpectedly contains native artifacts")
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
        source_receipt, source_plan = verify_source_build(root, source_root, name, operator, source)
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
            "capability_identity_unchanged": True,
            "bootstrap_native_operator_count": 0,
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
    step_root = root / "steps/artifact-example-build"
    text = (
        (step_root / "stdout.log").read_text(encoding="utf-8", errors="replace")
        + "\n"
        + (step_root / "stderr.log").read_text(encoding="utf-8", errors="replace")
    )
    summaries = [
        {"artifact": artifact, "status": status, "reason": reason}
        for artifact, status, reason in CUDA_SUMMARY_RE.findall(text)
    ]
    require(manifest.get("artifact_build_summaries") == summaries, "manifest build summaries differ from logs")
    for unit in EXPECTED_BUILD_UNITS:
        rows = [row for row in summaries if row["artifact"] == unit]
        require(len(rows) == 1, f"artifact build summary count mismatch for {unit}")
        require(
            rows[0]["status"] == "artifact"
            and rows[0]["reason"] == "native-operator-artifact-set",
            f"native artifact {unit} fell back to source",
        )
    for step in ("bootstrap-catalog-export", "artifact-catalog-export"):
        output = (root / "steps" / step / "stdout.log").read_text(encoding="utf-8", errors="replace")
        require(
            output.count("FERRUM RUNTIME VNEXT CUDA LIVE CATALOG READY:") == 1,
            f"{step} did not emit exactly one runtime readiness line",
        )


def verify_manifest(root: Path, source_root: Path) -> None:
    root = root.resolve()
    source_root = source_root.resolve()
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
            "source_features": SOURCE_FEATURES,
            "artifact_features": ARTIFACT_FEATURES,
            "operators": sorted(PACKAGES.values()),
        },
        "chain scope mismatch",
    )
    verify_artifact_index(root, manifest)
    source = verify_source(root, source_root, manifest)
    verify_hardware(root, manifest)
    verify_steps(root)
    verify_catalogs_and_packages(root, source_root, manifest, source)
    verify_build_summaries(root, manifest)
    binaries = require_dict(manifest.get("binaries"), "manifest.binaries")
    for kind in ("bootstrap", "artifact"):
        path = root / f"binaries/{kind}/runtime_vnext_cuda_catalog"
        identity = require_dict(binaries.get(kind), f"manifest.binaries.{kind}")
        require(path.is_file() and not path.is_symlink(), f"{kind} binary is missing")
        require(sha256(path) == identity.get("sha256"), f"{kind} binary SHA mismatch")


def expect_reject(action: Any, label: str) -> None:
    try:
        action()
    except VerificationError:
        return
    raise AssertionError(f"{label}: verifier accepted tampered evidence")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="g07b-native-chain-validator-") as temporary:
        root = Path(temporary)
        payload = root / "payload.txt"
        payload.write_text("original\n", encoding="utf-8")
        manifest = {
            "artifacts": collect_artifact_index(root),
            "artifact_count": 1,
        }
        verify_artifact_index(root, manifest)
        payload.write_text("tampered\n", encoding="utf-8")
        expect_reject(lambda: verify_artifact_index(root, manifest), "artifact-index tamper")
        payload.write_text("original\n", encoding="utf-8")
        link = root / "payload-link"
        os.symlink(payload.name, link)
        expect_reject(lambda: collect_artifact_index(root), "artifact symlink")
    print("FERRUM RUNTIME VNEXT G07B NATIVE CHAIN VALIDATOR SELFTEST PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--artifact-root", type=Path)
    result.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
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
    root = args.artifact_root.expanduser().resolve()
    try:
        verify_manifest(root, args.source_root.expanduser().resolve())
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
