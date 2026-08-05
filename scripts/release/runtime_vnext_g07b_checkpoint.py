#!/usr/bin/env python3
"""Freeze a canonical G07B checkpoint from verified G03/G07A dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import runtime_vnext_g03_live_catalog_checkpoint as g03_checkpoint
import runtime_vnext_g07a_checkpoint as g07a_checkpoint
import validate_runtime_vnext_g07b_native_chain as chain_validator


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PASS_PREFIX = "FERRUM RUNTIME VNEXT G07B NATIVE OPERATORS PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G07B CHECKPOINT SELFTEST PASS"
OUTER_GATE_FIELDS = {
    "artifact_dir",
    "binary",
    "child_artifacts",
    "child_execution_artifacts",
    "child_pass_line",
    "child_returncode",
    "command_line",
    "delegated_command_line",
    "dirty_status",
    "duration_sec",
    "error",
    "finished_at",
    "git_sha",
    "lane",
    "model",
    "pass_line",
    "sanitized_env",
    "schema_version",
    "started_at",
    "status",
}
UNLOCKS = ["G07"]
DOES_NOT_PROVE = ["G07", "G08", "G09", "G10", "release readiness"]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CHECKPOINT_CONTROL_FILES = {
    "manifest.json",
    "gate.manifest.json",
    "run_gate.child.command.json",
    "run_gate.child.stdout",
    "run_gate.child.stderr",
}


class CheckpointError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def write_json_create_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("x", encoding="ascii") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def source_identity(source_root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=source_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        require(result.returncode == 0, f"git {' '.join(arguments)} failed")
        return result.stdout.strip()

    status = git("status", "--short", "--untracked-files=all").splitlines()
    require(not status, f"G07B requires clean source: {status}")
    return validate_source_identity(
        {
            "git_sha": git("rev-parse", "HEAD"),
            "git_tree_sha": git("rev-parse", "HEAD^{tree}"),
            "dirty": False,
            "status_short": [],
        },
        "current source",
    )


def validate_source_identity(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} identity must be an object")
    require(
        set(value) == {"git_sha", "git_tree_sha", "dirty", "status_short"}
        and isinstance(value.get("git_sha"), str)
        and GIT_SHA_RE.fullmatch(value["git_sha"]) is not None
        and isinstance(value.get("git_tree_sha"), str)
        and GIT_SHA_RE.fullmatch(value["git_tree_sha"]) is not None
        and value.get("dirty") is False
        and value.get("status_short") == [],
        f"{label} identity is invalid or dirty",
    )
    return value


def file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    rendered = (
        resolved.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": rendered,
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_external_identity(value: Any, label: str) -> tuple[Path, dict[str, Any]]:
    require(isinstance(value, dict), f"{label} reference must be an object")
    require(
        set(value) in ({"path", "sha256"}, {"path", "sha256", "size_bytes"}),
        f"{label} reference field set mismatch",
    )
    raw_path = Path(str(value.get("path", ""))).expanduser()
    require(raw_path.is_absolute(), f"{label} path must be absolute")
    path = raw_path.resolve()
    expected_sha = value.get("sha256")
    require(
        isinstance(expected_sha, str)
        and SHA256_RE.fullmatch(expected_sha) is not None
        and sha256(path) == expected_sha,
        f"{label} SHA256 mismatch",
    )
    if "size_bytes" in value:
        require(value["size_bytes"] == path.stat().st_size, f"{label} size mismatch")
    return path, file_identity(path)


def _validate_outer_gate(
    path: Path,
    *,
    expected_lane: str,
    expected_child_prefix: str,
    source: dict[str, Any],
    child_verifier: Callable[[Path], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    outer_path = path.expanduser().resolve()
    outer = read_json(outer_path, f"{expected_lane} outer gate manifest")
    require(set(outer) == OUTER_GATE_FIELDS, f"{expected_lane} outer field set mismatch")
    root = outer_path.parent.resolve()
    require(
        outer["schema_version"] == 1
        and outer["lane"] == expected_lane
        and outer["status"] == "pass"
        and outer["child_returncode"] == 0
        and outer["error"] is None
        and Path(str(outer["artifact_dir"])).expanduser().resolve() == root
        and outer["pass_line"] == f"FERRUM GATE {expected_lane} PASS: {root}"
        and outer["child_pass_line"] == f"{expected_child_prefix}: {root}",
        f"{expected_lane} outer identity/status/PASS mismatch",
    )
    dirty = outer.get("dirty_status")
    require(
        dirty == {"is_dirty": False, "status_short": []}
        and outer.get("git_sha") == source["git_sha"],
        f"{expected_lane} outer source is stale or dirty",
    )
    child_artifacts = outer.get("child_artifacts")
    require(isinstance(child_artifacts, dict), f"{expected_lane} child artifacts missing")
    child_ref = child_artifacts.get("child_manifest")
    child_path, normalized_child_ref = validate_external_identity(
        child_ref, f"{expected_lane} child manifest"
    )
    require(child_path == root / "manifest.json", f"{expected_lane} child path mismatch")
    child_summary = child_verifier(child_path)
    require(child_summary.get("source") == source, f"{expected_lane} child source mismatch")
    return (
        {
            "outer_manifest": file_identity(outer_path),
            "child_manifest": normalized_child_ref,
            "source": source,
        },
        child_summary,
        child_path,
    )


def validate_g03_outer(
    path: Path,
    *,
    source_root: Path,
    source: dict[str, Any],
    verify_checkout: bool = True,
) -> dict[str, Any]:
    dependency, child, child_path = _validate_outer_gate(
        path,
        expected_lane="vnext-g03-live-catalog",
        expected_child_prefix="FERRUM RUNTIME VNEXT G03 LIVE CATALOG PASS",
        source=source,
        child_verifier=lambda candidate: g03_checkpoint.verify_checkpoint_manifest(
            candidate, source_root=source_root, verify_checkout=verify_checkout
        ),
    )
    provider_path = child_path.parent / "provider-catalog.json"
    capability_path = child_path.parent / "capability-catalog.json"
    catalogs = child.get("catalogs")
    require(isinstance(catalogs, dict), "G03 child catalogs are missing")
    dependency.update(
        {
            "provider_catalog": file_identity(provider_path),
            "capability_catalog": file_identity(capability_path),
            "catalogs": catalogs,
        }
    )
    return dependency


def validate_g07a_outer(
    path: Path, *, source: dict[str, Any], verify_checkout: bool = True
) -> dict[str, Any]:
    dependency, summary, _ = _validate_outer_gate(
        path,
        expected_lane="vnext-g07a",
        expected_child_prefix="FERRUM RUNTIME VNEXT G07A BUILD ITERATION PASS",
        source=source,
        child_verifier=lambda candidate: g07a_checkpoint.verify_checkpoint_manifest(
            candidate, verify_checkout=verify_checkout
        ),
    )
    dependency.update(
        {
            "hardware_fingerprint": summary["hardware_fingerprint"],
            "scenario_targets": summary["scenario_targets"],
            "semantic_plan_hash": summary["semantic_plan_hash"],
        }
    )
    return dependency


def validate_dependencies(
    g03_path: Path,
    g07a_path: Path,
    *,
    source_root: Path,
    verify_checkout: bool = True,
    expected_source: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root = source_root.expanduser().resolve()
    source = (
        source_identity(source_root)
        if verify_checkout
        else validate_source_identity(expected_source, "expected source")
    )
    dependencies = {
        "g03": validate_g03_outer(
            g03_path,
            source_root=source_root,
            source=source,
            verify_checkout=verify_checkout,
        ),
        "g07a": validate_g07a_outer(
            g07a_path,
            source=source,
            verify_checkout=verify_checkout,
        ),
    }
    require(
        dependencies["g03"]["source"] == dependencies["g07a"]["source"] == source,
        "G03/G07A/current source identity forked",
    )
    return source, dependencies


def checkpoint_artifact_index(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"checkpoint artifact contains symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in CHECKPOINT_CONTROL_FILES:
            rows.append(file_identity(path, relative_to=root))
    return rows


def validate_chain(
    chain_root: Path,
    *,
    source_root: Path,
    source: dict[str, Any],
    dependencies: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    root = chain_root.expanduser().resolve()
    manifest = read_json(root / "chain.manifest.json", "G07B chain manifest")
    native_source = manifest.get("native_source")
    require(isinstance(native_source, dict), "G07B chain native source is missing")
    native_source_root = Path(str(native_source.get("root", ""))).expanduser().resolve()
    try:
        chain_validator.verify_manifest(root, source_root, native_source_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointError(f"G07B chain verification failed: {error}") from error
    manifest = read_json(root / "chain.manifest.json", "G07B chain manifest")
    require(manifest.get("source") == source, "G07B chain source is stale")
    require(manifest.get("dependencies") == dependencies, "G07B chain dependency binding drifted")
    canonical_provider = Path(dependencies["g03"]["provider_catalog"]["path"])
    chain_provider = root / "catalog-input/provider-catalog.json"
    require(
        chain_provider.read_bytes() == canonical_provider.read_bytes(),
        "G07B chain provider catalog differs from canonical G03",
    )
    return manifest, native_source_root


def _copy_checkpoint_artifacts(chain_root: Path, staging: Path) -> None:
    copies = {
        chain_root / "artifact/compiled-native-operators.json": staging
        / "native-operator-catalog.json",
        chain_root / "catalog-input/provider-catalog.json": staging
        / "resolver-fixtures/g03-provider-catalog.json",
        chain_root / "artifact/provider-catalog.json": staging
        / "resolver-fixtures/runtime-provider-catalog.json",
        chain_root / "artifact/capability-catalog.json": staging
        / "resolver-fixtures/capability-catalog.json",
        chain_root / "native-operators.lock.json": staging
        / "resolver-fixtures/native-operators.lock.json",
        chain_root / "artifact-build-summary.receipt.json": staging
        / "resolver-fixtures/artifact-build-summary.receipt.json",
    }
    for source, destination in copies.items():
        require(source.is_file() and not source.is_symlink(), f"chain artifact missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for step in sorted(chain_validator.EXPECTED_STEPS):
        for name in ("stdout.log", "stderr.log"):
            source = chain_root / "steps" / step / name
            require(source.is_file() and not source.is_symlink(), f"chain log missing: {source}")
            destination = staging / "build-logs" / f"{step}.{name}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def build_manifest(
    *,
    artifact_root: Path,
    declared_root: Path,
    source: dict[str, Any],
    dependencies: dict[str, Any],
    chain_root: Path,
    chain_manifest: dict[str, Any],
    native_source_root: Path,
) -> dict[str, Any]:
    index = checkpoint_artifact_index(artifact_root)
    pass_line = f"{PASS_PREFIX}: {declared_root}"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g07b_native_operators_checkpoint",
        "checkpoint_id": "G07B",
        "lane": "runtime-vnext-g07b",
        "status": "pass",
        "canonical": True,
        "artifact_dir": str(declared_root),
        "source": source,
        "dependencies": dependencies,
        "chain_evidence": {
            "root": str(chain_root),
            "manifest": file_identity(chain_root / "chain.manifest.json"),
            "native_source_root": str(native_source_root),
            "artifact_index_sha256": canonical_json_sha256(chain_manifest["artifacts"]),
            "artifact_count": chain_manifest["artifact_count"],
        },
        "native_operator_catalog": file_identity(
            artifact_root / "native-operator-catalog.json", relative_to=artifact_root
        ),
        "artifact_index": index,
        "artifact_index_sha256": canonical_json_sha256(index),
        "unlocks": UNLOCKS,
        "does_not_prove": DOES_NOT_PROVE,
        "pass_line": pass_line,
    }


def verify_checkpoint_manifest(
    path: Path, *, source_root: Path = REPO_ROOT, verify_checkout: bool = True
) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    root = manifest_path.parent.resolve()
    manifest = read_json(manifest_path, "G07B checkpoint manifest")
    required_fields = {
        "schema_version",
        "artifact_type",
        "checkpoint_id",
        "lane",
        "status",
        "canonical",
        "artifact_dir",
        "source",
        "dependencies",
        "chain_evidence",
        "native_operator_catalog",
        "artifact_index",
        "artifact_index_sha256",
        "unlocks",
        "does_not_prove",
        "pass_line",
    }
    require(set(manifest) == required_fields, "G07B checkpoint field set mismatch")
    declared_root = Path(str(manifest.get("artifact_dir", ""))).expanduser().resolve()
    require(
        manifest["schema_version"] == SCHEMA_VERSION
        and manifest["artifact_type"] == "runtime_vnext_g07b_native_operators_checkpoint"
        and manifest["checkpoint_id"] == "G07B"
        and manifest["lane"] == "runtime-vnext-g07b"
        and manifest["status"] == "pass"
        and manifest["canonical"] is True
        and declared_root == root
        and manifest["unlocks"] == UNLOCKS
        and manifest["does_not_prove"] == DOES_NOT_PROVE
        and manifest["pass_line"] == f"{PASS_PREFIX}: {root}",
        "G07B checkpoint identity/status/PASS mismatch",
    )
    source = (
        source_identity(source_root)
        if verify_checkout
        else validate_source_identity(manifest["source"], "G07B checkpoint source")
    )
    require(manifest["source"] == source, "G07B checkpoint source is stale")
    dependency_refs = manifest["dependencies"]
    require(
        isinstance(dependency_refs, dict) and set(dependency_refs) == {"g03", "g07a"},
        "G07B checkpoint dependency set mismatch",
    )
    g03_ref = dependency_refs.get("g03")
    g07a_ref = dependency_refs.get("g07a")
    require(
        isinstance(g03_ref, dict)
        and isinstance(g07a_ref, dict)
        and isinstance(g03_ref.get("outer_manifest"), dict)
        and isinstance(g07a_ref.get("outer_manifest"), dict),
        "G07B checkpoint dependency manifest references are invalid",
    )
    g03_outer = Path(str(g03_ref["outer_manifest"].get("path", "")))
    g07a_outer = Path(str(g07a_ref["outer_manifest"].get("path", "")))
    _, dependencies = validate_dependencies(
        g03_outer,
        g07a_outer,
        source_root=source_root,
        verify_checkout=verify_checkout,
        expected_source=source,
    )
    require(manifest["dependencies"] == dependencies, "G07B checkpoint dependencies drifted")
    chain_summary = manifest["chain_evidence"]
    require(isinstance(chain_summary, dict), "G07B chain summary is missing")
    chain_root = Path(str(chain_summary.get("root", ""))).expanduser().resolve()
    chain_manifest, native_source_root = validate_chain(
        chain_root,
        source_root=source_root,
        source=source,
        dependencies=dependencies,
    )
    expected = build_manifest(
        artifact_root=root,
        declared_root=root,
        source=source,
        dependencies=dependencies,
        chain_root=chain_root,
        chain_manifest=chain_manifest,
        native_source_root=native_source_root,
    )
    for field in (
        "chain_evidence",
        "native_operator_catalog",
        "artifact_index",
        "artifact_index_sha256",
    ):
        require(manifest[field] == expected[field], f"G07B checkpoint {field} drifted")
    require(
        (root / "native-operator-catalog.json").read_bytes()
        == (chain_root / "artifact/compiled-native-operators.json").read_bytes(),
        "G07B checkpoint native operator catalog differs from chain evidence",
    )
    return {
        "kind": "vnext-g07b",
        "child_manifest": file_identity(manifest_path),
        "source": source,
        "g03_catalog_sha256": dependencies["g03"]["provider_catalog"]["sha256"],
        "g07a_hardware_fingerprint": dependencies["g07a"]["hardware_fingerprint"],
        "artifact_count": len(manifest["artifact_index"]),
    }


def execute(args: argparse.Namespace) -> str:
    source_root = args.source_root.expanduser().resolve()
    output = args.out.expanduser().resolve()
    require(not output.is_relative_to(source_root), "G07B output must be outside source root")
    require(not output.exists(), f"G07B output already exists: {output}")
    source, dependencies = validate_dependencies(
        args.g03, args.g07a, source_root=source_root
    )
    chain_root = args.chain_artifact_root.expanduser().resolve()
    chain_manifest, native_source_root = validate_chain(
        chain_root,
        source_root=source_root,
        source=source,
        dependencies=dependencies,
    )
    require(source_identity(source_root) == source, "source changed while validating G07B")
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.parent.is_symlink(), "G07B output parent must not be a symlink")
    staging = output.parent / f".{output.name}.{os.getpid()}.tmp"
    require(not staging.exists(), f"G07B staging path exists: {staging}")
    staging.mkdir()
    try:
        _copy_checkpoint_artifacts(chain_root, staging)
        manifest = build_manifest(
            artifact_root=staging,
            declared_root=output,
            source=source,
            dependencies=dependencies,
            chain_root=chain_root,
            chain_manifest=chain_manifest,
            native_source_root=native_source_root,
        )
        write_json_create_new(staging / "manifest.json", manifest)
        staging.replace(output)
        verify_checkpoint_manifest(output / "manifest.json", source_root=source_root)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        if output.exists() and output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        raise
    return f"{PASS_PREFIX}: {output}"


def expect_reject(action: Callable[[], Any], label: str) -> None:
    try:
        action()
    except CheckpointError:
        return
    raise CheckpointError(f"self-test mutation unexpectedly passed: {label}")


def self_test() -> str:
    with tempfile.TemporaryDirectory(prefix="ferrum-g07b-checkpoint-") as temporary:
        root = Path(temporary).resolve()
        source = {
            "git_sha": "1" * 40,
            "git_tree_sha": "2" * 40,
            "dirty": False,
            "status_short": [],
        }
        gate_root = root / "gate"
        gate_root.mkdir()
        child = gate_root / "manifest.json"
        write_json_create_new(child, {"source": source})
        outer = {
            "artifact_dir": str(gate_root),
            "binary": {"path": None, "sha256": None},
            "child_artifacts": {
                "kind": "fixture",
                "child_manifest": file_identity(child),
            },
            "child_execution_artifacts": [],
            "child_pass_line": f"CHILD PASS: {gate_root}",
            "child_returncode": 0,
            "command_line": ["fixture"],
            "delegated_command_line": ["fixture-child"],
            "dirty_status": {"is_dirty": False, "status_short": []},
            "duration_sec": 1.0,
            "error": None,
            "finished_at": "2026-08-05T00:00:01Z",
            "git_sha": source["git_sha"],
            "lane": "fixture-lane",
            "model": None,
            "pass_line": f"FERRUM GATE fixture-lane PASS: {gate_root}",
            "sanitized_env": {},
            "schema_version": 1,
            "started_at": "2026-08-05T00:00:00Z",
            "status": "pass",
        }
        outer_path = gate_root / "gate.manifest.json"
        write_json_create_new(outer_path, outer)
        dependency, child_summary, _ = _validate_outer_gate(
            outer_path,
            expected_lane="fixture-lane",
            expected_child_prefix="CHILD PASS",
            source=source,
            child_verifier=lambda candidate: read_json(candidate, "fixture child"),
        )
        require(
            dependency["source"] == source and child_summary["source"] == source,
            "fixture dependency mismatch",
        )
        original = child.read_bytes()
        child.write_bytes(original + b" ")
        expect_reject(
            lambda: _validate_outer_gate(
                outer_path,
                expected_lane="fixture-lane",
                expected_child_prefix="CHILD PASS",
                source=source,
                child_verifier=lambda candidate: read_json(candidate, "fixture child"),
            ),
            "child digest tamper",
        )
        child.write_bytes(original)
        outer["lane"] = "wrong-lane"
        outer_path.unlink()
        write_json_create_new(outer_path, outer)
        expect_reject(
            lambda: _validate_outer_gate(
                outer_path,
                expected_lane="fixture-lane",
                expected_child_prefix="CHILD PASS",
                source=source,
                child_verifier=lambda candidate: read_json(candidate, "fixture child"),
            ),
            "outer lane tamper",
        )
    return SELFTEST_PASS_LINE


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-root", type=Path, default=REPO_ROOT)
    result.add_argument("--g03", type=Path)
    result.add_argument("--g07a", type=Path)
    result.add_argument("--chain-artifact-root", type=Path)
    result.add_argument("--out", type=Path)
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.self_test:
            require(
                all(
                    value is None
                    for value in (
                        args.g03,
                        args.g07a,
                        args.chain_artifact_root,
                        args.out,
                    )
                ),
                "--self-test cannot be combined with artifact arguments",
            )
            print(self_test())
            return 0
        missing = [
            flag
            for flag, value in (
                ("--g03", args.g03),
                ("--g07a", args.g07a),
                ("--chain-artifact-root", args.chain_artifact_root),
                ("--out", args.out),
            )
            if value is None
        ]
        require(not missing, "missing required arguments: " + ", ".join(missing))
        print(execute(args))
        return 0
    except (CheckpointError, OSError, RuntimeError, ValueError) as error:
        output = args.out.expanduser().resolve() if args.out is not None else Path("<unset>")
        print(f"{PASS_PREFIX} FAIL: {output}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
