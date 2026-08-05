#!/usr/bin/env python3
"""Aggregate canonical G00P, G07A, and G07B checkpoints into the G07 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import runtime_vnext_g07a_checkpoint as g07a_checkpoint
import runtime_vnext_g07b_checkpoint as g07b_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PASS_PREFIX = "FERRUM RUNTIME VNEXT G07 BUILD NATIVE OPS PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G07 CHECKPOINT SELFTEST PASS"
G00P_PASS_PREFIX = "FERRUM RUNTIME VNEXT G00 BASELINE PASS"
G00P_FULL_SELFTEST_PASS_LINE = (
    "FERRUM RUNTIME VNEXT G00 BASELINE FULL SELFTEST PASS"
)
G00P_FROZEN_LEGACY_SHA = "cff4c47765ef3259b8a04890187d99c60da86394"
UNLOCKS = ["G08", "G09", "G10"]
DOES_NOT_PROVE = ["G08", "G09", "G10", "release readiness"]
CHECKPOINT_CONTROL_FILES = {
    "manifest.json",
    "gate.manifest.json",
    "run_gate.child.command.json",
    "run_gate.child.stdout",
    "run_gate.child.stderr",
}
CHILD_EXECUTION_FILES = (
    "run_gate.child.command.json",
    "run_gate.child.stdout",
    "run_gate.child.stderr",
)


class CheckpointError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return g07b_checkpoint.read_json(path, label)
    except g07b_checkpoint.CheckpointError as error:
        raise CheckpointError(str(error)) from error


def file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    try:
        return g07b_checkpoint.file_identity(path, relative_to=relative_to)
    except g07b_checkpoint.CheckpointError as error:
        raise CheckpointError(str(error)) from error


def source_identity(source_root: Path) -> dict[str, Any]:
    try:
        return g07b_checkpoint.source_identity(source_root)
    except g07b_checkpoint.CheckpointError as error:
        raise CheckpointError(str(error)) from error


def validate_source_identity(value: Any, label: str) -> dict[str, Any]:
    try:
        return g07b_checkpoint.validate_source_identity(value, label)
    except g07b_checkpoint.CheckpointError as error:
        raise CheckpointError(str(error)) from error


def validate_external_identity(value: Any, label: str) -> tuple[Path, dict[str, Any]]:
    try:
        return g07b_checkpoint.validate_external_identity(value, label)
    except g07b_checkpoint.CheckpointError as error:
        raise CheckpointError(str(error)) from error


def write_json_create_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("x", encoding="ascii") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def artifact_index(root: Path, *, include_manifest: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"G07 artifact contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if not include_manifest and relative in CHECKPOINT_CONTROL_FILES:
            continue
        rows.append(file_identity(path, relative_to=root))
    return rows


def tree_identity(path: Path, *, relative_to: Path) -> dict[str, Any]:
    root = path.resolve()
    require(root.is_dir() and not root.is_symlink(), f"artifact tree is missing: {root}")
    rows = artifact_index(root, include_manifest=True)
    return {
        "path": root.relative_to(relative_to.resolve()).as_posix(),
        "member_count": len(rows),
        "sha256": canonical_json_sha256(rows),
    }


def validate_outer_gate(
    path: Path,
    *,
    expected_lane: str,
    expected_child_prefix: str,
    source: dict[str, Any],
    child_verifier: Callable[[Path], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    outer_path = path.expanduser().resolve()
    outer = read_json(outer_path, f"{expected_lane} outer manifest")
    require(
        set(outer) == g07b_checkpoint.OUTER_GATE_FIELDS,
        f"{expected_lane} outer field set mismatch",
    )
    root = outer_path.parent.resolve()
    require(
        outer.get("schema_version") == 1
        and outer.get("lane") == expected_lane
        and outer.get("status") == "pass"
        and outer.get("child_returncode") == 0
        and outer.get("error") is None
        and Path(str(outer.get("artifact_dir", ""))).expanduser().resolve() == root
        and outer.get("pass_line") == f"FERRUM GATE {expected_lane} PASS: {root}"
        and outer.get("child_pass_line") == f"{expected_child_prefix}: {root}",
        f"{expected_lane} outer identity/status/PASS mismatch",
    )
    require(
        outer.get("dirty_status") == {"is_dirty": False, "status_short": []}
        and outer.get("git_sha") == source["git_sha"],
        f"{expected_lane} outer source is stale or dirty",
    )
    child_artifacts = outer.get("child_artifacts")
    require(isinstance(child_artifacts, dict), f"{expected_lane} child artifacts missing")
    child_path, normalized_child = validate_external_identity(
        child_artifacts.get("child_manifest"), f"{expected_lane} child manifest"
    )
    require(child_path == root / "manifest.json", f"{expected_lane} child path mismatch")
    try:
        summary = child_verifier(child_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointError(f"{expected_lane} child verification failed: {error}") from error
    require(summary.get("source") == source, f"{expected_lane} child source mismatch")
    require(
        summary.get("child_manifest", {}).get("sha256")
        == normalized_child["sha256"],
        f"{expected_lane} child summary binding mismatch",
    )
    dependency = {
        "outer_manifest": file_identity(outer_path),
        "child_manifest": normalized_child,
        "source": source,
        "summary": summary,
    }
    return dependency, summary, read_json(child_path, f"{expected_lane} child"), child_path


def validate_child_execution_artifacts(
    root: Path, value: Any, *, label: str
) -> list[dict[str, Any]]:
    require(isinstance(value, list), f"{label} child execution artifacts are missing")
    expected = [
        file_identity(root / name, relative_to=root) for name in CHILD_EXECUTION_FILES
    ]
    require(value == expected, f"{label} child execution artifact binding drifted")
    return expected


def validate_g00p_outer(
    path: Path,
    *,
    source: dict[str, Any],
    verify_checkout: bool = True,
    child_verifier: Callable[[Path, str, list[str]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    outer_path = path.expanduser().resolve()
    outer = read_json(outer_path, "vnext-g00 outer manifest")
    require(
        set(outer) == g07b_checkpoint.OUTER_GATE_FIELDS,
        "vnext-g00 outer field set mismatch",
    )
    root = outer_path.parent.resolve()
    expected_child_pass = f"{G00P_PASS_PREFIX}: {root}"
    require(
        outer.get("schema_version") == 1
        and outer.get("lane") == "vnext-g00"
        and outer.get("status") == "pass"
        and outer.get("child_returncode") == 0
        and outer.get("error") is None
        and Path(str(outer.get("artifact_dir", ""))).expanduser().resolve() == root
        and outer.get("pass_line") == f"FERRUM GATE vnext-g00 PASS: {root}"
        and outer.get("child_pass_line") == expected_child_pass,
        "vnext-g00 outer identity/status/PASS mismatch",
    )
    require(
        outer.get("dirty_status") == {"is_dirty": False, "status_short": []}
        and outer.get("git_sha") == source["git_sha"],
        "vnext-g00 validator source is stale or dirty",
    )
    validate_child_execution_artifacts(
        root, outer.get("child_execution_artifacts"), label="vnext-g00"
    )
    delegated = outer.get("delegated_command_line")
    require(
        isinstance(delegated, list)
        and len(delegated) == 5
        and all(isinstance(part, str) and part for part in delegated)
        and delegated[1] == "scripts/release/runtime_vnext_baseline_gate.py"
        and delegated[2] == "--out"
        and Path(delegated[3]).expanduser().resolve() == root
        and delegated[4] == "--require-full-self-test",
        "vnext-g00 delegated command is not the canonical full validator",
    )
    child_path = root / "manifest.json"
    child = read_json(child_path, "vnext-g00 child manifest")
    require(
        child.get("schema_version") == 1
        and child.get("status") == "pass"
        and child.get("source_git_sha") == G00P_FROZEN_LEGACY_SHA
        and child.get("validator_git_sha") == source["git_sha"]
        and child.get("validator_dirty_status") == []
        and Path(str(child.get("artifact_dir", ""))).expanduser().resolve() == root
        and child.get("waiver_count") == 0
        and child.get("pass_line") == expected_child_pass,
        "vnext-g00 child identity/status/PASS mismatch",
    )
    stdout = (root / "run_gate.child.stdout").read_text(
        encoding="utf-8", errors="strict"
    )
    require(
        expected_child_pass in stdout.splitlines()
        and G00P_FULL_SELFTEST_PASS_LINE in stdout.splitlines(),
        "vnext-g00 child stdout lacks the canonical PASS lines",
    )
    if child_verifier is None:
        import run_gate as unified_gate

        lane_command = unified_gate.LaneCommand(
            cmd=delegated,
            expected_child_pass_line=expected_child_pass,
            child_manifest_path=child_path,
            expected_source_git_sha=G00P_FROZEN_LEGACY_SHA,
            provenance_kind="vnext-g00",
        )
        try:
            summary = unified_gate.verify_child_pass_line(
                lane_command,
                stdout,
                verify_checkout=verify_checkout,
            )
        except unified_gate.GateError as error:
            raise CheckpointError(
                f"vnext-g00 canonical provenance failed: {error}"
            ) from error
    else:
        summary = child_verifier(child_path, stdout, delegated)
    require(isinstance(summary, dict), "vnext-g00 verifier summary is missing")
    require(
        summary.get("kind") == "vnext-g00"
        and summary.get("child_manifest", {}).get("sha256")
        == g07b_checkpoint.sha256(child_path)
        and summary.get("artifact_index_sha256")
        == canonical_json_sha256(child.get("artifact_index")),
        "vnext-g00 verifier summary binding mismatch",
    )
    require(
        outer.get("child_artifacts") == summary,
        "vnext-g00 outer child provenance differs from full revalidation",
    )
    return {
        "outer_manifest": file_identity(outer_path),
        "child_manifest": file_identity(child_path),
        "source": source,
        "frozen_legacy_git_sha": G00P_FROZEN_LEGACY_SHA,
        "artifact_index_sha256": summary["artifact_index_sha256"],
        "summary": summary,
    }


def validate_cross_binding(
    *,
    source: dict[str, Any],
    g00p_dependency: dict[str, Any],
    g07a_dependency: dict[str, Any],
    g07a_summary: dict[str, Any],
    g07a_manifest: dict[str, Any],
    g07b_summary: dict[str, Any],
    g07b_manifest: dict[str, Any],
) -> dict[str, Any]:
    require(
        g07a_manifest.get("source") == g07b_manifest.get("source") == source,
        "G07A/G07B/current source identity forked",
    )
    require(
        g00p_dependency.get("source") == source
        and g00p_dependency.get("frozen_legacy_git_sha")
        == G00P_FROZEN_LEGACY_SHA,
        "G00P validator source or frozen legacy source drifted",
    )
    g07b_dependencies = g07b_manifest.get("dependencies")
    require(isinstance(g07b_dependencies, dict), "G07B dependencies are missing")
    bound_g07a = g07b_dependencies.get("g07a")
    require(isinstance(bound_g07a, dict), "G07B does not bind G07A")
    for field in ("outer_manifest", "child_manifest", "source"):
        require(
            bound_g07a.get(field) == g07a_dependency.get(field),
            f"G07B binds a different G07A {field}",
        )
    require(
        bound_g07a.get("hardware_fingerprint")
        == g07a_summary.get("hardware_fingerprint")
        == g07b_summary.get("g07a_hardware_fingerprint"),
        "G07A/G07B build-host fingerprint forked",
    )
    require(
        bound_g07a.get("scenario_targets") == g07a_summary.get("scenario_targets")
        and bound_g07a.get("semantic_plan_hash")
        == g07a_summary.get("semantic_plan_hash"),
        "G07B G07A timing or semantic-plan binding drifted",
    )
    g07a_artifacts = g07a_manifest.get("artifacts")
    require(isinstance(g07a_artifacts, dict), "G07A artifacts are missing")
    g03 = g07b_dependencies.get("g03")
    require(isinstance(g03, dict), "G07B G03 dependency is missing")
    provider_catalog = g03.get("provider_catalog")
    require(isinstance(provider_catalog, dict), "G07B G03 provider catalog is missing")
    native_catalog = g07b_manifest.get("native_operator_catalog")
    chain = g07b_manifest.get("chain_evidence")
    require(isinstance(native_catalog, dict), "G07B native operator catalog is missing")
    require(isinstance(chain, dict), "G07B native chain evidence is missing")
    require(
        provider_catalog.get("sha256") == g07b_summary.get("g03_catalog_sha256"),
        "G07B summary and canonical G03 catalog forked",
    )
    return {
        "source_git_sha": source["git_sha"],
        "source_tree_sha": source["git_tree_sha"],
        "g00p_outer_sha256": g00p_dependency["outer_manifest"]["sha256"],
        "g00p_child_sha256": g00p_dependency["child_manifest"]["sha256"],
        "g00p_artifact_index_sha256": g00p_dependency[
            "artifact_index_sha256"
        ],
        "g07a_outer_sha256": g07a_dependency["outer_manifest"]["sha256"],
        "g07a_child_sha256": g07a_dependency["child_manifest"]["sha256"],
        "crate_graph_sha256": g07a_artifacts["crate_graph"]["sha256"],
        "invalidation_report_sha256": g07a_artifacts["invalidation_report"]["sha256"],
        "build_timing_summary_sha256": g07a_artifacts["build_timing_summary"]["sha256"],
        "semantic_plan_hash": g07a_summary["semantic_plan_hash"],
        "g03_provider_catalog_sha256": provider_catalog["sha256"],
        "native_operator_catalog_sha256": native_catalog["sha256"],
        "native_chain_manifest_sha256": chain["manifest"]["sha256"],
    }


def validate_dependencies(
    g00p_path: Path,
    g07a_path: Path,
    g07b_path: Path,
    *,
    source_root: Path,
    verify_checkout: bool = True,
    expected_source: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    source = (
        source_identity(source_root)
        if verify_checkout
        else validate_source_identity(expected_source, "expected source")
    )
    g00p_dependency = validate_g00p_outer(
        g00p_path,
        source=source,
        verify_checkout=verify_checkout,
    )
    g07a_dependency, g07a_summary, g07a_manifest, g07a_child = validate_outer_gate(
        g07a_path,
        expected_lane="vnext-g07a",
        expected_child_prefix="FERRUM RUNTIME VNEXT G07A BUILD ITERATION PASS",
        source=source,
        child_verifier=lambda candidate: g07a_checkpoint.verify_checkpoint_manifest(
            candidate, verify_checkout=verify_checkout
        ),
    )
    g07b_dependency, g07b_summary, g07b_manifest, g07b_child = validate_outer_gate(
        g07b_path,
        expected_lane="vnext-g07b",
        expected_child_prefix="FERRUM RUNTIME VNEXT G07B NATIVE OPERATORS PASS",
        source=source,
        child_verifier=lambda candidate: g07b_checkpoint.verify_checkpoint_manifest(
            candidate,
            source_root=source_root,
            verify_checkout=verify_checkout,
        ),
    )
    freshness = validate_cross_binding(
        source=source,
        g00p_dependency=g00p_dependency,
        g07a_dependency=g07a_dependency,
        g07a_summary=g07a_summary,
        g07a_manifest=g07a_manifest,
        g07b_summary=g07b_summary,
        g07b_manifest=g07b_manifest,
    )
    dependencies = {
        "g00p": g00p_dependency,
        "g07a": g07a_dependency,
        "g07b": g07b_dependency,
    }
    return source, dependencies, freshness, g07a_child.parent, g07b_child.parent


def copy_tree(source: Path, destination: Path) -> None:
    require(source.is_dir() and not source.is_symlink(), f"source tree is missing: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.rglob("*")):
        require(not path.is_symlink(), f"source tree contains a symlink: {path}")
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def copy_checkpoint_artifacts(g07a_root: Path, g07b_root: Path, staging: Path) -> None:
    files = {
        g07a_root / "crate-graph.json": staging / "crate-graph.json",
        g07a_root / "invalidation-report.json": staging / "invalidation-report.json",
        g07b_root / "native-operator-catalog.json": staging / "native-operator-catalog.json",
    }
    for source, destination in files.items():
        require(
            source.is_file() and not source.is_symlink(),
            f"child artifact is missing: {source}",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    trees = {
        g07a_root / "build-timings": staging / "build-timings",
        g07b_root / "resolver-fixtures": staging / "resolver-fixtures",
        g07b_root / "build-logs": staging / "build-logs",
    }
    for source, destination in trees.items():
        copy_tree(source, destination)


def validate_checkpoint_copies(root: Path, g07a_root: Path, g07b_root: Path) -> None:
    pairs = {
        root / "crate-graph.json": g07a_root / "crate-graph.json",
        root / "invalidation-report.json": g07a_root / "invalidation-report.json",
        root / "native-operator-catalog.json": g07b_root / "native-operator-catalog.json",
    }
    for copied, source in pairs.items():
        require(
            copied.read_bytes() == source.read_bytes(),
            f"aggregate copy drifted: {copied.name}",
        )
    for name, source in (
        ("build-timings", g07a_root / "build-timings"),
        ("resolver-fixtures", g07b_root / "resolver-fixtures"),
        ("build-logs", g07b_root / "build-logs"),
    ):
        require(
            artifact_index(root / name, include_manifest=True)
            == artifact_index(source, include_manifest=True),
            f"aggregate {name} tree differs from child checkpoint",
        )


def build_manifest(
    *,
    artifact_root: Path,
    declared_root: Path,
    source: dict[str, Any],
    dependencies: dict[str, Any],
    freshness: dict[str, Any],
) -> dict[str, Any]:
    index = artifact_index(artifact_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g07_build_native_ops_checkpoint",
        "checkpoint_id": "G07",
        "lane": "runtime-vnext-g07",
        "status": "pass",
        "canonical": True,
        "artifact_dir": str(declared_root),
        "source": source,
        "dependencies": dependencies,
        "freshness": freshness,
        "artifacts": {
            "crate_graph": file_identity(
                artifact_root / "crate-graph.json", relative_to=artifact_root
            ),
            "invalidation_report": file_identity(
                artifact_root / "invalidation-report.json", relative_to=artifact_root
            ),
            "native_operator_catalog": file_identity(
                artifact_root / "native-operator-catalog.json", relative_to=artifact_root
            ),
            "build_timings": tree_identity(
                artifact_root / "build-timings", relative_to=artifact_root
            ),
            "resolver_fixtures": tree_identity(
                artifact_root / "resolver-fixtures", relative_to=artifact_root
            ),
            "build_logs": tree_identity(
                artifact_root / "build-logs", relative_to=artifact_root
            ),
        },
        "artifact_index": index,
        "artifact_index_sha256": canonical_json_sha256(index),
        "unlocks": UNLOCKS,
        "does_not_prove": DOES_NOT_PROVE,
        "pass_line": f"{PASS_PREFIX}: {declared_root}",
    }


def verify_checkpoint_manifest(
    path: Path, *, source_root: Path = REPO_ROOT, verify_checkout: bool = True
) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    root = manifest_path.parent.resolve()
    manifest = read_json(manifest_path, "G07 checkpoint manifest")
    required = {
        "schema_version",
        "artifact_type",
        "checkpoint_id",
        "lane",
        "status",
        "canonical",
        "artifact_dir",
        "source",
        "dependencies",
        "freshness",
        "artifacts",
        "artifact_index",
        "artifact_index_sha256",
        "unlocks",
        "does_not_prove",
        "pass_line",
    }
    require(set(manifest) == required, "G07 checkpoint field set mismatch")
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type")
        == "runtime_vnext_g07_build_native_ops_checkpoint"
        and manifest.get("checkpoint_id") == "G07"
        and manifest.get("lane") == "runtime-vnext-g07"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True
        and Path(str(manifest.get("artifact_dir", ""))).expanduser().resolve() == root
        and manifest.get("unlocks") == UNLOCKS
        and manifest.get("does_not_prove") == DOES_NOT_PROVE
        and manifest.get("pass_line") == f"{PASS_PREFIX}: {root}",
        "G07 checkpoint identity/status/PASS mismatch",
    )
    dependencies_value = manifest.get("dependencies")
    require(
        isinstance(dependencies_value, dict)
        and set(dependencies_value) == {"g00p", "g07a", "g07b"},
        "G07 dependency set mismatch",
    )
    g00p_path = Path(str(dependencies_value["g00p"]["outer_manifest"]["path"]))
    g07a_path = Path(str(dependencies_value["g07a"]["outer_manifest"]["path"]))
    g07b_path = Path(str(dependencies_value["g07b"]["outer_manifest"]["path"]))
    source, dependencies, freshness, g07a_root, g07b_root = validate_dependencies(
        g00p_path,
        g07a_path,
        g07b_path,
        source_root=source_root,
        verify_checkout=verify_checkout,
        expected_source=manifest.get("source"),
    )
    require(manifest.get("source") == source, "G07 checkpoint source is stale")
    require(manifest.get("dependencies") == dependencies, "G07 dependencies drifted")
    require(manifest.get("freshness") == freshness, "G07 freshness binding drifted")
    validate_checkpoint_copies(root, g07a_root, g07b_root)
    expected = build_manifest(
        artifact_root=root,
        declared_root=root,
        source=source,
        dependencies=dependencies,
        freshness=freshness,
    )
    for field in ("artifacts", "artifact_index", "artifact_index_sha256"):
        require(manifest.get(field) == expected[field], f"G07 checkpoint {field} drifted")
    return {
        "kind": "vnext-g07",
        "child_manifest": file_identity(manifest_path),
        "source": source,
        "g00p_child_sha256": freshness["g00p_child_sha256"],
        "g07a_child_sha256": freshness["g07a_child_sha256"],
        "g03_provider_catalog_sha256": freshness["g03_provider_catalog_sha256"],
        "native_operator_catalog_sha256": freshness["native_operator_catalog_sha256"],
        "artifact_count": len(manifest["artifact_index"]),
    }


def execute(args: argparse.Namespace) -> str:
    source_root = args.source_root.expanduser().resolve()
    output = args.out.expanduser().resolve()
    require(not output.is_relative_to(source_root), "G07 output must be outside source root")
    require(not output.exists(), f"G07 output already exists: {output}")
    source, dependencies, freshness, g07a_root, g07b_root = validate_dependencies(
        args.g00p,
        args.g07a,
        args.g07b,
        source_root=source_root,
    )
    require(source_identity(source_root) == source, "source changed while validating G07")
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.parent.is_symlink(), "G07 output parent must not be a symlink")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        copy_checkpoint_artifacts(g07a_root, g07b_root, staging)
        validate_checkpoint_copies(staging, g07a_root, g07b_root)
        manifest = build_manifest(
            artifact_root=staging,
            declared_root=output,
            source=source,
            dependencies=dependencies,
            freshness=freshness,
        )
        write_json_create_new(staging / "manifest.json", manifest)
        os.replace(staging, output)
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
    with tempfile.TemporaryDirectory(prefix="ferrum-g07-checkpoint-") as temporary:
        root = Path(temporary).resolve()
        source = {
            "git_sha": "1" * 40,
            "git_tree_sha": "2" * 40,
            "dirty": False,
            "status_short": [],
        }
        g07a_root = root / "g07a"
        g07b_root = root / "g07b"
        (g07a_root / "build-timings").mkdir(parents=True)
        (g07b_root / "resolver-fixtures").mkdir(parents=True)
        (g07b_root / "build-logs").mkdir(parents=True)
        for path, payload in (
            (g07a_root / "crate-graph.json", b"crate"),
            (g07a_root / "invalidation-report.json", b"invalidation"),
            (g07a_root / "build-timings/summary.json", b"timing"),
            (g07b_root / "native-operator-catalog.json", b"native"),
            (g07b_root / "resolver-fixtures/catalog.json", b"catalog"),
            (g07b_root / "build-logs/build.stdout.log", b"build"),
        ):
            path.write_bytes(payload)
        staging = root / "aggregate"
        staging.mkdir()
        copy_checkpoint_artifacts(g07a_root, g07b_root, staging)
        validate_checkpoint_copies(staging, g07a_root, g07b_root)
        (staging / "nested").mkdir()
        for name in CHECKPOINT_CONTROL_FILES:
            (staging / name).write_text(f"control {name}\n", encoding="utf-8")
        (staging / "nested/manifest.json").write_text("nested\n", encoding="utf-8")
        indexed_paths = {row["path"] for row in artifact_index(staging)}
        g07b_indexed_paths = {
            row["path"] for row in g07b_checkpoint.checkpoint_artifact_index(staging)
        }
        require(
            "nested/manifest.json" in indexed_paths
            and "nested/manifest.json" in g07b_indexed_paths
            and not (CHECKPOINT_CONTROL_FILES & indexed_paths)
            and not (CHECKPOINT_CONTROL_FILES & g07b_indexed_paths),
            "checkpoint control-file exclusion or nested manifest coverage drifted",
        )

        g00p_root = root / "g00p"
        g00p_root.mkdir()
        g00p_child = {
            "schema_version": 1,
            "status": "pass",
            "source_git_sha": G00P_FROZEN_LEGACY_SHA,
            "validator_git_sha": source["git_sha"],
            "validator_dirty_status": [],
            "artifact_dir": str(g00p_root),
            "artifact_index": [],
            "waiver_count": 0,
            "pass_line": f"{G00P_PASS_PREFIX}: {g00p_root}",
        }
        write_json_create_new(g00p_root / "manifest.json", g00p_child)
        (g00p_root / "run_gate.child.stdout").write_text(
            f"{G00P_FULL_SELFTEST_PASS_LINE}\n"
            f"{G00P_PASS_PREFIX}: {g00p_root}\n",
            encoding="utf-8",
        )
        (g00p_root / "run_gate.child.stderr").write_text("", encoding="utf-8")
        (g00p_root / "run_gate.child.command.json").write_text(
            "{\"fixture\": true}\n", encoding="utf-8"
        )
        g00p_summary = {
            "kind": "vnext-g00",
            "child_manifest": {
                "path": "manifest.json",
                "sha256": g07b_checkpoint.sha256(g00p_root / "manifest.json"),
                "artifact_count": 0,
                "contract_sha256": "e" * 64,
            },
            "artifact_index_sha256": canonical_json_sha256([]),
            "full_redteam": {"pass_line": G00P_FULL_SELFTEST_PASS_LINE},
        }
        g00p_outer = {
            "artifact_dir": str(g00p_root),
            "binary": {"path": None, "sha256": None},
            "child_artifacts": g00p_summary,
            "child_execution_artifacts": [
                file_identity(g00p_root / name, relative_to=g00p_root)
                for name in CHILD_EXECUTION_FILES
            ],
            "child_pass_line": f"{G00P_PASS_PREFIX}: {g00p_root}",
            "child_returncode": 0,
            "command_line": ["run_gate.py", "vnext-g00"],
            "delegated_command_line": [
                sys.executable,
                "scripts/release/runtime_vnext_baseline_gate.py",
                "--out",
                str(g00p_root),
                "--require-full-self-test",
            ],
            "dirty_status": {"is_dirty": False, "status_short": []},
            "duration_sec": 1.0,
            "error": None,
            "finished_at": "2026-08-05T00:00:01Z",
            "git_sha": source["git_sha"],
            "lane": "vnext-g00",
            "model": None,
            "pass_line": f"FERRUM GATE vnext-g00 PASS: {g00p_root}",
            "sanitized_env": {},
            "schema_version": 1,
            "started_at": "2026-08-05T00:00:00Z",
            "status": "pass",
        }
        write_json_create_new(g00p_root / "gate.manifest.json", g00p_outer)
        g00p_dependency = validate_g00p_outer(
            g00p_root / "gate.manifest.json",
            source=source,
            verify_checkout=False,
            child_verifier=lambda _path, _stdout, _command: g00p_summary,
        )
        g00p_stdout = g00p_root / "run_gate.child.stdout"
        original_stdout = g00p_stdout.read_bytes()
        g00p_stdout.write_bytes(original_stdout + b"tampered\n")
        expect_reject(
            lambda: validate_g00p_outer(
                g00p_root / "gate.manifest.json",
                source=source,
                verify_checkout=False,
                child_verifier=lambda _path, _stdout, _command: g00p_summary,
            ),
            "G00P wrapper stdout tamper",
        )
        g00p_stdout.write_bytes(original_stdout)
        outer_ref = {
            "path": "/fixture/g07a/gate.manifest.json",
            "sha256": "3" * 64,
            "size_bytes": 1,
        }
        child_ref = {
            "path": "/fixture/g07a/manifest.json",
            "sha256": "4" * 64,
            "size_bytes": 1,
        }
        summary = {
            "hardware_fingerprint": "5" * 64,
            "scenario_targets": {"noop": 30},
            "semantic_plan_hash": "6" * 64,
        }
        g07a_dependency = {
            "outer_manifest": outer_ref,
            "child_manifest": child_ref,
            "source": source,
            "summary": summary,
        }
        g07a_manifest = {
            "source": source,
            "artifacts": {
                "crate_graph": {"sha256": "7" * 64},
                "invalidation_report": {"sha256": "8" * 64},
                "build_timing_summary": {"sha256": "9" * 64},
            },
        }
        g07b_summary = {
            "g07a_hardware_fingerprint": "5" * 64,
            "g03_catalog_sha256": "a" * 64,
        }
        g07b_manifest = {
            "source": source,
            "dependencies": {
                "g07a": {
                    "outer_manifest": outer_ref,
                    "child_manifest": child_ref,
                    "source": source,
                    "hardware_fingerprint": "5" * 64,
                    "scenario_targets": {"noop": 30},
                    "semantic_plan_hash": "6" * 64,
                },
                "g03": {"provider_catalog": {"sha256": "a" * 64}},
            },
            "native_operator_catalog": {"sha256": "b" * 64},
            "chain_evidence": {"manifest": {"sha256": "c" * 64}},
        }
        freshness = validate_cross_binding(
            source=source,
            g00p_dependency=g00p_dependency,
            g07a_dependency=g07a_dependency,
            g07a_summary=summary,
            g07a_manifest=g07a_manifest,
            g07b_summary=g07b_summary,
            g07b_manifest=g07b_manifest,
        )
        require(freshness["g03_provider_catalog_sha256"] == "a" * 64, "catalog binding drifted")
        stale = json.loads(json.dumps(g07b_manifest))
        stale["dependencies"]["g07a"]["child_manifest"]["sha256"] = "d" * 64
        expect_reject(
            lambda: validate_cross_binding(
                source=source,
                g00p_dependency=g00p_dependency,
                g07a_dependency=g07a_dependency,
                g07a_summary=summary,
                g07a_manifest=g07a_manifest,
                g07b_summary=g07b_summary,
                g07b_manifest=stale,
            ),
            "G07B G07A child fork",
        )
        (staging / "crate-graph.json").write_bytes(b"tampered")
        expect_reject(
            lambda: validate_checkpoint_copies(staging, g07a_root, g07b_root),
            "aggregate crate graph tamper",
        )
    return SELFTEST_PASS_LINE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--g00p", type=Path)
    parser.add_argument("--g07a", type=Path)
    parser.add_argument("--g07b", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        try:
            print(self_test())
        except (OSError, RuntimeError, ValueError) as error:
            print(f"{SELFTEST_PASS_LINE} REJECT: {error}", file=sys.stderr)
            return 1
        return 0
    missing = [
        name
        for name in ("g00p", "g07a", "g07b", "out")
        if getattr(args, name) is None
    ]
    if missing:
        print(
            f"{PASS_PREFIX} FAIL: missing {', '.join('--' + name for name in missing)}",
            file=sys.stderr,
        )
        return 2
    try:
        print(execute(args))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"{PASS_PREFIX} FAIL: {args.out}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
