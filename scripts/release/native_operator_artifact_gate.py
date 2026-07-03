#!/usr/bin/env python3
"""Validate Ferrum native operator manifests and artifact resolver invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES = REPO_ROOT / "scripts/release/fixtures/native_operator"
PASS_LINE = "NATIVE OP ARTIFACT PASS"
SELFTEST_PASS_LINE = "NATIVE OP ARTIFACT SELFTEST PASS"
SCHEMA_VERSION = 1
FERRUM_NATIVE_ABI_VERSION = "1"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SM_RE = re.compile(r"^sm_[0-9]+$")

BULK_SOURCE_ROOTS = [
    REPO_ROOT / "crates/ferrum-kernels/kernels/fa2_source/flash_attn",
    REPO_ROOT / "crates/ferrum-kernels/kernels/fa2_source/cutlass",
]
CPP_CUDA_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp", ".hxx"}
BUILD_RS = REPO_ROOT / "crates/ferrum-kernels/build.rs"
RELEASE_CONFIG_AUDIT_FILES = [
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / ".github/workflows/release-cuda.yml",
    REPO_ROOT / "scripts/release/g0_source_gate.sh",
    REPO_ROOT / "scripts/release/run_gate.py",
]
RELEASE_CONFIG_AUDIT_GLOBS = [
    REPO_ROOT / "scripts/release/configs",
]
RELEASE_CONFIG_FORBIDDEN_PATTERNS = {
    "release_build_feature_fa2_source": re.compile(r"cuda[,\w-]*,fa2-source|fa2-source[,\w-]*", re.IGNORECASE),
    "vendored_flash_attn_probe": re.compile(r"fa2_source/flash_attn|flash_attn/src", re.IGNORECASE),
    "vendored_cutlass_probe": re.compile(r"fa2_source/cutlass|cutlass/include", re.IGNORECASE),
}
NATIVE_OP_SOURCE_COMPILE_PATTERNS = {
    "compile_fa2": re.compile(r"\bcompile_[a-z0-9_]*fa2[a-z0-9_]*\s*\(", re.IGNORECASE),
    "build_fa2": re.compile(r"\bbuild_[a-z0-9_]*fa2[a-z0-9_]*\s*\(", re.IGNORECASE),
    "fa2_cc_build": re.compile(r"fa2[\s\S]{0,240}cc::Build", re.IGNORECASE),
    "fa2_nvcc_spawn": re.compile(r"fa2[\s\S]{0,240}Command::new\(&nvcc\)", re.IGNORECASE),
}


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolverRequirement:
    operator: str
    backend: str
    operator_abi_version: str = FERRUM_NATIVE_ABI_VERSION
    ferrum_native_abi_version: str = FERRUM_NATIVE_ABI_VERSION
    compute_capability: str | None = None
    source_package_sha256: str | None = None
    inputs_sha256: str | None = None
    binary_sha256: str | None = None


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise GateError(f"{path}: expected JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_keys(data: dict[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise GateError(f"{context}: missing required keys: {', '.join(missing)}")


def require_no_extra_keys(data: dict[str, Any], allowed: set[str], context: str) -> None:
    extra = sorted(set(data) - allowed)
    if extra:
        raise GateError(f"{context}: unexpected keys: {', '.join(extra)}")


def require_string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"{context}.{key} must be a non-empty string")
    return value


def require_string_or_null(data: dict[str, Any], key: str, context: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"{context}.{key} must be a non-empty string or null")
    return value


def require_sha256(value: str, context: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.match(value):
        raise GateError(f"{context} must be a 64-character hex sha256 digest")


def require_string_list(data: dict[str, Any], key: str, context: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list):
        raise GateError(f"{context}.{key} must be an array")
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise GateError(f"{context}.{key}[{index}] must be a non-empty string")
        out.append(item)
    return out


def validate_manifest(manifest: dict[str, Any], *, context: str) -> dict[str, Any]:
    required = {
        "schema_version",
        "operator",
        "operator_abi_version",
        "ferrum_native_abi_version",
        "backend",
        "compute_capabilities",
        "source_package",
        "inputs_sha256",
        "binary_sha256",
        "linkage",
        "exports",
        "license_files",
        "build_summary",
    }
    allowed = required | {"cuda_toolkit", "cuda_runtime_min"}
    require_keys(manifest, required, context)
    require_no_extra_keys(manifest, allowed, context)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise GateError(f"{context}.schema_version must be {SCHEMA_VERSION}")
    require_string(manifest, "operator", context)
    require_string(manifest, "operator_abi_version", context)
    require_string(manifest, "ferrum_native_abi_version", context)
    backend = require_string(manifest, "backend", context)
    if backend not in {"cuda", "metal", "cpu"}:
        raise GateError(f"{context}.backend must be cuda, metal, or cpu")
    require_string_or_null(manifest, "cuda_toolkit", context)
    require_string_or_null(manifest, "cuda_runtime_min", context)
    compute_capabilities = require_string_list(manifest, "compute_capabilities", context)
    if backend == "cuda" and not compute_capabilities:
        raise GateError(f"{context}.compute_capabilities must be non-empty for cuda")
    for capability in compute_capabilities:
        if not SM_RE.match(capability):
            raise GateError(f"{context}.compute_capabilities must use sm_xx values")

    source_package = manifest.get("source_package")
    if not isinstance(source_package, dict):
        raise GateError(f"{context}.source_package must be an object")
    require_keys(source_package, {"kind", "revision", "sha256"}, f"{context}.source_package")
    require_no_extra_keys(source_package, {"kind", "revision", "sha256"}, f"{context}.source_package")
    require_string(source_package, "kind", f"{context}.source_package")
    require_string(source_package, "revision", f"{context}.source_package")
    require_sha256(source_package.get("sha256"), f"{context}.source_package.sha256")

    require_sha256(manifest.get("inputs_sha256"), f"{context}.inputs_sha256")
    require_sha256(manifest.get("binary_sha256"), f"{context}.binary_sha256")
    linkage = require_string(manifest, "linkage", context)
    if linkage not in {"static", "dynamic"}:
        raise GateError(f"{context}.linkage must be static or dynamic")
    exports = require_string_list(manifest, "exports", context)
    if "ferrum_native_op_init" not in exports:
        raise GateError(f"{context}.exports must include ferrum_native_op_init")
    require_string_list(manifest, "license_files", context)

    build_summary = manifest.get("build_summary")
    if not isinstance(build_summary, dict):
        raise GateError(f"{context}.build_summary must be an object")
    require_keys(
        build_summary,
        {"builder_sha", "elapsed_ms", "nvcc_version", "host_compiler"},
        f"{context}.build_summary",
    )
    require_no_extra_keys(
        build_summary,
        {"builder_sha", "elapsed_ms", "nvcc_version", "host_compiler"},
        f"{context}.build_summary",
    )
    require_string(build_summary, "builder_sha", f"{context}.build_summary")
    elapsed_ms = build_summary.get("elapsed_ms")
    if not isinstance(elapsed_ms, int) or elapsed_ms < 0:
        raise GateError(f"{context}.build_summary.elapsed_ms must be a non-negative integer")
    require_string_or_null(build_summary, "nvcc_version", f"{context}.build_summary")
    require_string(build_summary, "host_compiler", f"{context}.build_summary")
    return manifest


def resolve_manifest(
    manifest: dict[str, Any] | None,
    requirement: ResolverRequirement,
    *,
    context: str,
) -> dict[str, Any]:
    if manifest is None:
        raise GateError(f"{context}: native operator manifest is missing")
    validate_manifest(manifest, context=context)
    if manifest["operator"] != requirement.operator:
        raise GateError(
            f"{context}: operator mismatch manifest={manifest['operator']} required={requirement.operator}"
        )
    if manifest["backend"] != requirement.backend:
        raise GateError(
            f"{context}: backend mismatch manifest={manifest['backend']} required={requirement.backend}"
        )
    if manifest["operator_abi_version"] != requirement.operator_abi_version:
        raise GateError(
            f"{context}: operator ABI mismatch manifest={manifest['operator_abi_version']} "
            f"required={requirement.operator_abi_version}"
        )
    if manifest["ferrum_native_abi_version"] != requirement.ferrum_native_abi_version:
        raise GateError(
            f"{context}: Ferrum native ABI mismatch manifest={manifest['ferrum_native_abi_version']} "
            f"required={requirement.ferrum_native_abi_version}"
        )
    if requirement.compute_capability and requirement.compute_capability not in manifest["compute_capabilities"]:
        raise GateError(
            f"{context}: compute capability mismatch manifest={manifest['compute_capabilities']} "
            f"required={requirement.compute_capability}"
        )
    expected = {
        "source_package.sha256": (
            manifest["source_package"]["sha256"],
            requirement.source_package_sha256,
        ),
        "inputs_sha256": (manifest["inputs_sha256"], requirement.inputs_sha256),
        "binary_sha256": (manifest["binary_sha256"], requirement.binary_sha256),
    }
    for field, (actual, wanted) in expected.items():
        if wanted is None:
            continue
        require_sha256(wanted, f"{context}.expected.{field}")
        if actual.lower() != wanted.lower():
            raise GateError(f"{context}: {field} mismatch manifest={actual} expected={wanted}")
    return {
        "operator": manifest["operator"],
        "backend": manifest["backend"],
        "linkage": manifest["linkage"],
        "binary_sha256": manifest["binary_sha256"],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_operator_value(items: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise GateError(f"{flag} value must use operator=value form: {item}")
        operator, value = item.split("=", 1)
        if not operator or not value:
            raise GateError(f"{flag} value must use non-empty operator=value form: {item}")
        out[operator] = value
    return out


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def fixture_marker(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return normalized in {"fixture", "fixtures"} or normalized.startswith("fixture-")


def reject_fixture_release_inputs(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    artifact_path: Path,
) -> None:
    if is_relative_to(manifest_path, DEFAULT_FIXTURES):
        raise GateError(
            f"{manifest_path}: normal native operator artifact gate must not use fixture manifests"
        )
    if is_relative_to(artifact_path, DEFAULT_FIXTURES):
        raise GateError(
            f"{manifest_path}: normal native operator artifact gate must not use fixture binaries"
        )
    if artifact_path.suffix.lower() == ".whl":
        raise GateError(f"{manifest_path}: native operator artifact must not be a Python wheel")
    source_package = manifest.get("source_package")
    if isinstance(source_package, dict) and (
        fixture_marker(source_package.get("kind")) or fixture_marker(source_package.get("revision"))
    ):
        raise GateError(
            f"{manifest_path}: normal native operator artifact gate must not use fixture source_package metadata"
        )


def count_bulk_source_files() -> tuple[int, list[str]]:
    samples: list[str] = []
    count = 0
    for root in BULK_SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            count += 1
            if len(samples) < 20:
                samples.append(path.relative_to(REPO_ROOT).as_posix())
    return count, samples


def find_unregistered_third_party_sources() -> tuple[int, list[str]]:
    samples: list[str] = []
    count = 0
    for root in (REPO_ROOT / "crates").glob("**/third_party"):
        if "target" in root.parts:
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in CPP_CUDA_EXTENSIONS:
                count += 1
                if len(samples) < 20:
                    samples.append(path.relative_to(REPO_ROOT).as_posix())
    return count, samples


def native_operator_dev_build_audit() -> dict[str, Any]:
    try:
        build_rs = BUILD_RS.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise GateError(f"missing native operator build audit input: {BUILD_RS}") from exc
    matches: list[dict[str, str]] = []
    for name, pattern in sorted(NATIVE_OP_SOURCE_COMPILE_PATTERNS.items()):
        for match in pattern.finditer(build_rs):
            line = build_rs.count("\n", 0, match.start()) + 1
            matches.append(
                {
                    "pattern": name,
                    "file": BUILD_RS.relative_to(REPO_ROOT).as_posix(),
                    "line": str(line),
                }
            )
    obsolete_marker = (
        "feature fa2-source is obsolete; use a Ferrum native operator artifact for FA2" in build_rs
        and '"fa2_source"' in build_rs
        and '"skipped"' in build_rs
        and "obsolete-native-operator-artifact-required" in build_rs
    )
    return {
        "status": "pass" if not matches and obsolete_marker else "fail",
        "source_compile_count": len(matches),
        "source_compile_matches": matches,
        "fa2_source_feature_behavior": "obsolete_warning_only" if obsolete_marker else "missing_obsolete_marker",
        "inspected_files": [BUILD_RS.relative_to(REPO_ROOT).as_posix()],
    }


def release_config_audit_from_texts(texts: dict[str, str]) -> dict[str, Any]:
    matches: list[dict[str, str]] = []
    for label, text in sorted(texts.items()):
        for pattern_name, pattern in sorted(RELEASE_CONFIG_FORBIDDEN_PATTERNS.items()):
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                matches.append(
                    {
                        "pattern": pattern_name,
                        "file": label,
                        "line": str(line),
                        "match": match.group(0),
                    }
                )
    return {
        "status": "pass" if not matches else "fail",
        "forbidden_reference_count": len(matches),
        "forbidden_references": matches,
        "inspected_files": sorted(texts),
    }


def native_operator_release_config_audit() -> dict[str, Any]:
    texts: dict[str, str] = {}
    files = list(RELEASE_CONFIG_AUDIT_FILES)
    for root in RELEASE_CONFIG_AUDIT_GLOBS:
        if root.is_dir():
            files.extend(sorted(path for path in root.rglob("*") if path.is_file()))
    for path in files:
        if not path.exists():
            continue
        texts[path.relative_to(REPO_ROOT).as_posix()] = path.read_text(encoding="utf-8")
    return release_config_audit_from_texts(texts)


def git_output(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise GateError(f"git {' '.join(args)} failed rc={proc.returncode}: {proc.stderr}")
    return proc.stdout.strip()


def run_selftest() -> dict[str, Any]:
    pass_dir = DEFAULT_FIXTURES / "pass"
    fail_dir = DEFAULT_FIXTURES / "fail"
    pass_paths = sorted(pass_dir.glob("*.json"))
    fail_paths = sorted(fail_dir.glob("*.json"))
    if not pass_paths:
        raise GateError(f"missing pass fixtures under {pass_dir}")
    if not fail_paths:
        raise GateError(f"missing fail fixtures under {fail_dir}")

    resolved: list[str] = []
    for path in pass_paths:
        manifest = validate_manifest(read_json(path), context=str(path))
        requirement = ResolverRequirement(
            operator=str(manifest["operator"]),
            backend=str(manifest["backend"]),
            compute_capability=manifest["compute_capabilities"][0]
            if manifest["compute_capabilities"]
            else None,
            source_package_sha256=str(manifest["source_package"]["sha256"]),
            inputs_sha256=str(manifest["inputs_sha256"]),
            binary_sha256=str(manifest["binary_sha256"]),
        )
        resolve_manifest(manifest, requirement, context=str(path))
        resolved.append(path.name)

    rejected_validation: list[str] = []
    for path in fail_paths:
        try:
            validate_manifest(read_json(path), context=str(path))
        except GateError:
            rejected_validation.append(path.name)
        else:
            raise GateError(f"{path} was expected to fail validation")

    fa2_fixture = validate_manifest(read_json(pass_dir / "fa2_manifest.json"), context="fa2_fixture")
    mutations = {
        "missing_manifest": lambda: resolve_manifest(
            None,
            ResolverRequirement(operator="fa2", backend="cuda", compute_capability="sm_89"),
            context="missing_manifest",
        ),
        "binary_sha256_mismatch": lambda: resolve_manifest(
            fa2_fixture,
            ResolverRequirement(
                operator="fa2",
                backend="cuda",
                compute_capability="sm_89",
                binary_sha256="d" * 64,
            ),
            context="binary_sha256_mismatch",
        ),
        "abi_mismatch": lambda: resolve_manifest(
            {**fa2_fixture, "operator_abi_version": "2"},
            ResolverRequirement(operator="fa2", backend="cuda", compute_capability="sm_89"),
            context="abi_mismatch",
        ),
        "compute_capability_mismatch": lambda: resolve_manifest(
            fa2_fixture,
            ResolverRequirement(operator="fa2", backend="cuda", compute_capability="sm_90"),
            context="compute_capability_mismatch",
        ),
        "operator_mismatch": lambda: resolve_manifest(
            fa2_fixture,
            ResolverRequirement(operator="dummy", backend="cuda", compute_capability="sm_89"),
            context="operator_mismatch",
        ),
    }
    rejected_resolution: list[str] = []
    for name, action in mutations.items():
        try:
            action()
        except GateError:
            rejected_resolution.append(name)
        else:
            raise GateError(f"resolver mutation {name} was expected to fail closed")

    dev_build_audit = native_operator_dev_build_audit()
    if dev_build_audit["status"] != "pass":
        raise GateError("native operator dev-build source compile audit failed")
    bulk_count, bulk_samples = count_bulk_source_files()
    if bulk_count:
        raise GateError(
            "native operator self-test requires FA2/CUTLASS bulk source count = 0; "
            f"found {bulk_count}: {bulk_samples}"
        )
    third_party_count, third_party_samples = find_unregistered_third_party_sources()
    if third_party_count:
        raise GateError(
            "native operator self-test rejects unregistered crates/**/third_party C++/CUDA source; "
            f"found {third_party_count}: {third_party_samples}"
        )
    release_config_audit = native_operator_release_config_audit()
    if release_config_audit["status"] != "pass":
        raise GateError("native operator release-config audit failed")
    fixture_release_rejections: list[str] = []
    for label, manifest_path, artifact_path in [
        (
            "fixture_manifest_path",
            pass_dir / "fa2_manifest.json",
            DEFAULT_FIXTURES / "artifacts/fa2/libferrum_native_fa2.a",
        ),
        (
            "fixture_source_package",
            REPO_ROOT / "external/native/fa2_manifest.json",
            REPO_ROOT / "external/native/libferrum_native_fa2.a",
        ),
    ]:
        try:
            reject_fixture_release_inputs(
                manifest_path=manifest_path,
                manifest=fa2_fixture,
                artifact_path=artifact_path,
            )
        except GateError:
            fixture_release_rejections.append(label)
        else:
            raise GateError(f"normal gate fixture rejection {label} unexpectedly passed")
    bad_release_config = release_config_audit_from_texts(
        {
            "bad-release.yml": (
                "cargo build --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source\n"
                "test -f crates/ferrum-kernels/kernels/fa2_source/flash_attn/src/flash.h\n"
            )
        }
    )
    if bad_release_config["status"] != "fail":
        raise GateError("native operator release-config audit failed to reject bad fixture")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "pass_fixtures": resolved,
        "fail_fixtures": rejected_validation,
        "resolver_fail_closed_cases": rejected_resolution,
        "normal_gate_fixture_rejections": fixture_release_rejections,
        "python_runtime_dependency": "none",
        "bulk_source": {"count": bulk_count, "samples": bulk_samples},
        "unregistered_third_party_source": {
            "count": third_party_count,
            "samples": third_party_samples,
        },
        "normal_cuda_dev_build": dev_build_audit,
        "release_config": release_config_audit,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.manifest:
        raise GateError("normal native operator artifact gate requires at least one --manifest")
    started_at = datetime.now(timezone.utc)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    binary_artifacts = {
        operator: Path(value)
        for operator, value in parse_operator_value(args.binary_artifact, "--binary-artifact").items()
    }
    source_hashes = parse_operator_value(args.source_package_sha256, "--source-package-sha256")
    input_hashes = parse_operator_value(args.inputs_sha256, "--inputs-sha256")

    bulk_count, bulk_samples = count_bulk_source_files()
    third_party_count, third_party_samples = find_unregistered_third_party_sources()
    dev_build_audit = native_operator_dev_build_audit()
    release_config_audit = native_operator_release_config_audit()
    if bulk_count:
        raise GateError(
            "native operator artifact gate requires FA2/CUTLASS bulk source removal before PASS; "
            f"found {bulk_count} files"
        )
    if third_party_count:
        raise GateError(
            "native operator artifact gate rejects unregistered crates/**/third_party C++/CUDA source; "
            f"found {third_party_count} files"
        )
    if dev_build_audit["status"] != "pass":
        raise GateError(
            "native operator artifact gate requires normal CUDA dev build native-op source compile count = 0; "
            f"found {dev_build_audit['source_compile_count']} hooks"
        )
    if release_config_audit["status"] != "pass":
        raise GateError(
            "native operator artifact gate requires release CUDA configs to avoid fa2-source and vendored FA2 probes; "
            f"found {release_config_audit['forbidden_reference_count']} references"
        )

    manifest_summaries: list[dict[str, Any]] = []
    for path in args.manifest:
        manifest = validate_manifest(read_json(path), context=str(path))
        operator = str(manifest["operator"])
        artifact_path = binary_artifacts.get(operator)
        if artifact_path is None:
            raise GateError(f"{path}: missing --binary-artifact {operator}=<path>")
        reject_fixture_release_inputs(
            manifest_path=path,
            manifest=manifest,
            artifact_path=artifact_path,
        )
        if not artifact_path.is_file():
            raise GateError(f"{path}: binary artifact does not exist: {artifact_path}")
        binary_sha256 = sha256_file(artifact_path)
        requirement = ResolverRequirement(
            operator=operator,
            backend=str(manifest["backend"]),
            compute_capability=manifest["compute_capabilities"][0] if manifest["compute_capabilities"] else None,
            source_package_sha256=source_hashes.get(operator),
            inputs_sha256=input_hashes.get(operator),
            binary_sha256=binary_sha256,
        )
        if requirement.source_package_sha256 is None:
            raise GateError(f"{path}: missing --source-package-sha256 {operator}=<sha256>")
        if requirement.inputs_sha256 is None:
            raise GateError(f"{path}: missing --inputs-sha256 {operator}=<sha256>")
        resolution = resolve_manifest(manifest, requirement, context=str(path))
        manifest_summaries.append(
            {
                "manifest": str(path),
                "operator": operator,
                "backend": manifest["backend"],
                "compute_capabilities": manifest["compute_capabilities"],
                "linkage": manifest["linkage"],
                "source_package": manifest["source_package"],
                "inputs_sha256": manifest["inputs_sha256"],
                "binary_artifact": str(artifact_path),
                "binary_sha256": binary_sha256,
                "resolution": resolution,
            }
        )

    selftest_summary = run_selftest()
    dirty_files = git_output(["status", "--short"]).splitlines()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "gate": "native_operator_artifact",
        "manifests": manifest_summaries,
        "selftest_summary": selftest_summary,
        "bulk_source": {"count": bulk_count, "samples": bulk_samples},
        "unregistered_third_party_source": {
            "count": third_party_count,
            "samples": third_party_samples,
        },
        "normal_cuda_dev_build": dev_build_audit,
        "release_config": release_config_audit,
    }
    write_json(out / "native_operator_artifact_summary.json", summary)
    pass_line = f"{PASS_LINE}: {out}"
    ended_at = datetime.now(timezone.utc)
    (out / "pass_line.txt").write_text(pass_line + "\n", encoding="utf-8")
    (out / "command.log").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (out / "git_status.txt").write_text(
        git_output(["status", "--short", "--branch"]) + "\n",
        encoding="utf-8",
    )
    write_json(out / "sanitized_env.json", {"schema_version": SCHEMA_VERSION, "env": {}})
    write_json(
        out / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "goal": "release-regression-hardening-2026-06-28",
            "phase": "native_operator_artifact",
            "status": "pass",
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_sec": (ended_at - started_at).total_seconds(),
            "repo_root": str(REPO_ROOT),
            "git_sha": git_output(["rev-parse", "HEAD"]),
            "git_branch": git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_dirty": bool(dirty_files),
            "dirty_files": dirty_files,
            "command": sys.argv,
            "artifact_dir": str(out),
            "pass_line": pass_line,
            "outputs": {
                "summary": str(out / "native_operator_artifact_summary.json"),
                "pass_line": str(out / "pass_line.txt"),
                "command_log": str(out / "command.log"),
                "git_status": str(out / "git_status.txt"),
                "sanitized_env": str(out / "sanitized_env.json"),
            },
            "validation_summary": summary,
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--binary-artifact", action="append", default=[])
    parser.add_argument("--source-package-sha256", action="append", default=[])
    parser.add_argument("--inputs-sha256", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_selftest()
            print(SELFTEST_PASS_LINE)
            return 0
        if args.out is None:
            raise GateError("--out is required unless --self-test is set")
        run_gate(args)
        print(f"{PASS_LINE}: {args.out}")
        return 0
    except GateError as exc:
        print(f"NATIVE OP ARTIFACT FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
