#!/usr/bin/env python3
"""Freeze canonical G07A build-iteration evidence as a DAG checkpoint."""

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
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_gate  # noqa: E402
import runtime_vnext_cuda_correctness_build as correctness_build  # noqa: E402
import runtime_vnext_g07a_build_iteration as timing_collector  # noqa: E402
import runtime_vnext_s1_cuda_checkpoint as s1_checkpoint  # noqa: E402
import validate_runtime_vnext_g07a_build_iteration as timing_validator  # noqa: E402


SCHEMA_VERSION = 1
PASS_PREFIX = "FERRUM RUNTIME VNEXT G07A BUILD ITERATION PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G07A CHECKPOINT SELFTEST PASS"
UNLOCKS = ["G07B", "S4"]
DOES_NOT_PROVE = [
    "G07B",
    "G07",
    "G08",
    "G09",
    "G10",
    "release readiness",
]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class CheckpointError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def sha256(path: Path) -> str:
    require(
        path.is_file() and not path.is_symlink(),
        f"required regular file is missing: {path}",
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(
        path.is_file() and not path.is_symlink(),
        f"{label} is not a regular file: {path}",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(
        result.returncode == 0,
        f"git {' '.join(args)} failed: {result.stderr.strip()}",
    )
    return result.stdout.strip()


def clean_source(*, verify_checkout: bool = True) -> dict[str, Any]:
    if not verify_checkout:
        return {}
    status = [
        line
        for line in git_text(
            "status",
            "--short",
            "--untracked-files=all",
        ).splitlines()
        if line
    ]
    require(not status, f"G07A requires a clean checkout: {status}")
    return {
        "git_sha": git_text("rev-parse", "HEAD"),
        "git_tree_sha": git_text("rev-parse", "HEAD^{tree}"),
        "dirty": False,
        "status_short": [],
    }


def require_source(
    value: Any,
    expected: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} source must be an object")
    source = dict(value)
    require(
        set(source) == {"git_sha", "git_tree_sha", "dirty", "status_short"}
        and GIT_SHA_RE.fullmatch(str(source.get("git_sha"))) is not None
        and GIT_SHA_RE.fullmatch(str(source.get("git_tree_sha"))) is not None
        and source.get("dirty") is False
        and source.get("status_short") == [],
        f"{label} source identity is invalid",
    )
    if expected:
        require(source == expected, f"{label} source differs from current checkout")
    return source


def resolve_ref(
    root: Path,
    value: Any,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    require(isinstance(value, dict), f"{label} reference must be an object")
    ref = dict(value)
    require(
        set(ref) == {"path", "sha256", "size_bytes"},
        f"{label} reference field set mismatch",
    )
    relative = ref.get("path")
    require(
        isinstance(relative, str)
        and relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        f"{label} path is not a safe relative path",
    )
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise CheckpointError(f"{label} escapes its artifact root") from error
    size = ref.get("size_bytes")
    require(
        isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
        and path.stat().st_size == size,
        f"{label} size mismatch",
    )
    require(
        SHA256_RE.fullmatch(str(ref.get("sha256"))) is not None
        and sha256(path) == ref["sha256"],
        f"{label} SHA256 mismatch",
    )
    return path, ref


def external_ref(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_external_ref(value: Any, label: str) -> tuple[Path, dict[str, Any]]:
    require(isinstance(value, dict), f"{label} reference must be an object")
    ref = dict(value)
    require(
        set(ref) == {"path", "sha256", "size_bytes"},
        f"{label} reference field set mismatch",
    )
    path = Path(str(ref.get("path", ""))).expanduser().resolve()
    require(path.is_absolute(), f"{label} path must be absolute")
    size = ref.get("size_bytes")
    require(
        isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
        and path.stat().st_size == size,
        f"{label} size mismatch",
    )
    require(
        SHA256_RE.fullmatch(str(ref.get("sha256"))) is not None
        and sha256(path) == ref["sha256"],
        f"{label} SHA256 mismatch",
    )
    return path, ref


def validate_g00f(
    manifest_path: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    manifest = read_json(path, "G00F manifest")
    lane = run_gate.LaneCommand(
        cmd=[],
        expected_child_pass_line=manifest.get("pass_line"),
        child_manifest_path=path,
        provenance_kind="vnext-g00f",
    )
    try:
        provenance = run_gate.validate_vnext_g00f_provenance(
            lane,
            manifest,
            sha256(path),
            verify_checkout=True,
        )
    except run_gate.GateError as error:
        raise CheckpointError(f"G00F provenance failed: {error}") from error
    require_source(manifest.get("source"), source, "G00F")
    return {
        "manifest": external_ref(path),
        "provenance": provenance,
    }


def validate_s1(
    manifest_path: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    manifest = read_json(path, "S1 manifest")
    require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact_type")
        == "runtime_vnext_s1_cuda_basic_slice_manifest"
        and manifest.get("checkpoint_id") == "S1-CUDA-basic"
        and manifest.get("lane") == "runtime-vnext-s1-cuda"
        and manifest.get("status") == "pass"
        and manifest.get("backend") == "cuda"
        and manifest.get("entrypoints") == ["ferrum run", "ferrum serve"]
        and str(manifest.get("pass_line", "")).startswith(
            "FERRUM RUNTIME VNEXT S1 CUDA BASIC SLICE PASS:"
        ),
        "S1 manifest identity/status mismatch",
    )
    require(
        manifest.get("source_git_sha") == source["git_sha"],
        "S1 source differs from current checkout",
    )
    artifact_root = Path(str(manifest.get("artifact_dir", ""))).resolve()
    require(
        path == artifact_root / "manifest.json",
        "S1 manifest is not under its declared artifact_dir",
    )
    validation_path, validation_ref = resolve_ref(
        artifact_root,
        manifest.get("validation"),
        "S1 validation",
    )
    validation = read_json(validation_path, "S1 validation")
    raw_root = Path(str(manifest.get("raw_artifact_dir", ""))).resolve()
    require(
        validation.get("schema_version") == 1
        and validation.get("artifact_type")
        == "runtime_vnext_s1_cuda_basic_slice_validation"
        and validation.get("status") == "pass"
        and Path(str(validation.get("raw_artifact_dir", ""))).resolve()
        == raw_root
        and validation.get("source_git_sha") == source["git_sha"]
        and validation.get("binary_sha256") == manifest.get("binary_sha256")
        and validation.get("hardware") == manifest.get("hardware"),
        "S1 validation binding mismatch",
    )
    try:
        correctness = s1_checkpoint.validate(raw_root, source["git_sha"])
        performance = s1_checkpoint.validate_profile_overhead(raw_root)
        product = s1_checkpoint.validate_product_commands(raw_root)
        raw_index = s1_checkpoint.artifact_index(raw_root)
    except s1_checkpoint.ValidationError as error:
        raise CheckpointError(f"S1 raw evidence failed: {error}") from error
    require(
        validation.get("correctness")
        == {
            "run": correctness["run"],
            "serve": correctness["serve"],
        }
        and validation.get("profile_overhead") == performance
        and validation.get("product") == product
        and validation.get("raw_artifact_count") == len(raw_index)
        and validation.get("raw_artifact_index") == raw_index
        and validation.get("raw_artifact_index_sha256")
        == canonical_json_sha256(raw_index),
        "S1 validation is not independently reproducible",
    )
    gpu_uuid = manifest.get("metrics", {}).get("gpu_uuid")
    require(
        isinstance(gpu_uuid, str) and gpu_uuid.startswith("GPU-"),
        "S1 GPU UUID is invalid",
    )
    return {
        "manifest": external_ref(path),
        "validation": validation_ref,
        "raw_artifact_dir": str(raw_root),
        "raw_artifact_index_sha256": validation[
            "raw_artifact_index_sha256"
        ],
        "binary_sha256": manifest["binary_sha256"],
        "gpu_uuid": gpu_uuid,
    }


def validate_source_gate(
    outer_path: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    path = outer_path.expanduser().resolve()
    outer = read_json(path, "workspace source gate outer manifest")
    require(
        path.name == "gate.manifest.json"
        and outer.get("schema_version") == 1
        and outer.get("lane") == "unit"
        and outer.get("status") == "pass"
        and outer.get("git_sha") == source["git_sha"]
        and outer.get("dirty_status")
        == {"is_dirty": False, "status_short": []}
        and outer.get("pass_line")
        == f"FERRUM GATE unit PASS: {path.parent}",
        "workspace source gate outer manifest is stale or invalid",
    )
    child_ref = outer.get("child_artifacts", {}).get("child_manifest")
    require(
        isinstance(child_ref, dict),
        "workspace source gate lacks child manifest identity",
    )
    child_path = Path(str(child_ref.get("path", ""))).resolve()
    require(
        child_path.parent == path.parent
        and sha256(child_path) == child_ref.get("sha256"),
        "workspace source gate child manifest identity mismatch",
    )
    child = read_json(child_path, "workspace source gate child manifest")
    lane = run_gate.LaneCommand(
        cmd=[],
        expected_child_pass_line=outer.get("child_pass_line"),
        child_manifest_path=child_path,
        provenance_kind="g0-source-unit",
    )
    try:
        provenance = run_gate.validate_g0_source_unit_provenance(
            lane,
            child,
            sha256(child_path),
            verify_checkout=True,
        )
    except run_gate.GateError as error:
        raise CheckpointError(
            f"workspace source gate provenance failed: {error}"
        ) from error
    require(
        provenance.get("source")
        == {
            "git_sha": source["git_sha"],
            "git_tree_sha": source["git_tree_sha"],
            "dirty_status": {"is_dirty": False, "status_short": []},
        },
        "workspace source gate did not test the candidate clean source tree",
    )
    return {
        "outer_manifest": external_ref(path),
        "child_manifest": external_ref(child_path),
        "provenance": provenance,
    }


def validate_semantic_plan(
    artifact_root: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    try:
        summary = correctness_build.verify_semantic_trace_artifact(
            artifact_root,
            source_root=REPO_ROOT,
            verify_checkout=True,
        )
    except correctness_build.CorrectnessBuildError as error:
        raise CheckpointError(f"semantic-plan verification failed: {error}") from error
    require(
        summary.get("source_git_sha") == source["git_sha"]
        and summary.get("source_tree_sha") == source["git_tree_sha"]
        and isinstance(summary.get("hardware_id"), str)
        and "rtx4090" in summary["hardware_id"].lower(),
        "semantic-plan artifact is stale or not fixed-host RTX 4090 evidence",
    )
    validation_path = Path(summary["validation_path"])
    return {
        "validation": external_ref(validation_path),
        "source_git_sha": source["git_sha"],
        "source_tree_sha": source["git_tree_sha"],
        "hardware_id": summary["hardware_id"],
        "binary_sha256": summary["binary_sha256"],
        "reference_binary_sha256": summary[
            "reference_binary_sha256"
        ],
        "plan_hash": summary["reference_plan_hash"],
        "plan_built_event_count": summary["plan_built_event_count"],
        "focused_decision": summary["focused_decision"],
    }


def validate_timing_evidence(
    artifact_root: Path,
    source: dict[str, Any],
    g00f: dict[str, Any],
    s1: dict[str, Any],
) -> dict[str, Any]:
    root = artifact_root.expanduser().resolve()
    try:
        summary = timing_validator.verify_manifest(
            root,
            REPO_ROOT,
            require_canonical=True,
            verify_checkout=True,
        )
    except timing_validator.VerificationError as error:
        raise CheckpointError(f"G07A timing evidence failed: {error}") from error
    manifest_path = root / "evidence.manifest.json"
    manifest = read_json(manifest_path, "G07A timing evidence manifest")
    timing_source = require_source(manifest.get("source"), source, "G07A timing")
    g00f_copy, _ = resolve_ref(
        root,
        manifest.get("inputs", {}).get("g00f"),
        "timing G00F input",
    )
    s1_copy, _ = resolve_ref(
        root,
        manifest.get("inputs", {}).get("s1"),
        "timing S1 input",
    )
    require(
        sha256(g00f_copy) == g00f["manifest"]["sha256"]
        and sha256(s1_copy) == s1["manifest"]["sha256"],
        "timing evidence dependency copies differ from checkpoint inputs",
    )
    require(
        summary.get("mode") == "canonical"
        and summary.get("source_git_sha") == source["git_sha"]
        and all(
            row.get("target_met") is True
            for row in summary.get("scenario_targets", {}).values()
        ),
        "canonical G07A timing threshold summary mismatch",
    )
    return {
        "artifact_root": str(root),
        "manifest": external_ref(manifest_path),
        "artifact_index_sha256": manifest["artifact_index_sha256"],
        "source": timing_source,
        "hardware": manifest["hardware"],
        "scenario_targets": summary["scenario_targets"],
    }


def copied_artifact_ref(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def checkpoint_artifact_index(root: Path) -> dict[str, dict[str, Any]]:
    excluded = {
        "manifest.json",
        "gate.manifest.json",
        "run_gate.child.command.json",
        "run_gate.child.stdout",
        "run_gate.child.stderr",
    }
    return {
        str(path.relative_to(root)): {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and str(path.relative_to(root)) not in excluded
    }


def verify_checkpoint_manifest(
    manifest_path: Path,
    *,
    verify_checkout: bool,
) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    root = path.parent
    manifest = read_json(path, "G07A checkpoint manifest")
    require(
        set(manifest)
        == {
            "schema_version",
            "artifact_type",
            "checkpoint_id",
            "lane",
            "status",
            "canonical",
            "artifact_dir",
            "source",
            "dependencies",
            "timing_evidence",
            "artifacts",
            "artifact_index",
            "artifact_index_sha256",
            "unlocks",
            "does_not_prove",
            "pass_line",
        },
        "G07A checkpoint manifest field set mismatch",
    )
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type")
        == "runtime_vnext_g07a_build_iteration_checkpoint"
        and manifest.get("checkpoint_id") == "G07A"
        and manifest.get("lane") == "runtime-vnext-g07a"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True
        and Path(str(manifest.get("artifact_dir", ""))).resolve() == root
        and manifest.get("pass_line") == f"{PASS_PREFIX}: {root}"
        and manifest.get("unlocks") == UNLOCKS
        and manifest.get("does_not_prove") == DOES_NOT_PROVE,
        "G07A checkpoint identity/status mismatch",
    )
    current = clean_source(verify_checkout=verify_checkout)
    source = require_source(manifest.get("source"), current, "G07A checkpoint")
    dependencies = manifest.get("dependencies")
    require(
        isinstance(dependencies, dict)
        and set(dependencies)
        == {"g00f", "s1", "source_gate", "semantic_plan"},
        "G07A checkpoint dependency set mismatch",
    )
    g00f_path, _ = validate_external_ref(
        dependencies["g00f"].get("manifest"),
        "G00F manifest",
    )
    g00f = validate_g00f(g00f_path, source)
    require(g00f == dependencies["g00f"], "G00F summary drifted")
    s1_path, _ = validate_external_ref(
        dependencies["s1"].get("manifest"),
        "S1 manifest",
    )
    s1 = validate_s1(s1_path, source)
    require(s1 == dependencies["s1"], "S1 summary drifted")
    source_gate_path, _ = validate_external_ref(
        dependencies["source_gate"].get("outer_manifest"),
        "workspace source gate outer manifest",
    )
    source_gate = validate_source_gate(source_gate_path, source)
    require(
        source_gate == dependencies["source_gate"],
        "workspace source gate summary drifted",
    )
    semantic_path, _ = validate_external_ref(
        dependencies["semantic_plan"].get("validation"),
        "semantic-plan validation",
    )
    semantic = validate_semantic_plan(semantic_path, source)
    require(
        semantic == dependencies["semantic_plan"],
        "semantic-plan summary drifted",
    )
    timing_value = manifest.get("timing_evidence")
    require(
        isinstance(timing_value, dict)
        and isinstance(timing_value.get("artifact_root"), str),
        "G07A timing evidence reference is invalid",
    )
    timing = validate_timing_evidence(
        Path(timing_value["artifact_root"]),
        source,
        g00f,
        s1,
    )
    require(timing == timing_value, "G07A timing summary drifted")
    raw_root = Path(timing["artifact_root"])
    raw_manifest = read_json(
        raw_root / "evidence.manifest.json",
        "G07A raw timing manifest",
    )
    raw_crate_graph, _ = resolve_ref(
        raw_root,
        raw_manifest.get("crate_graph"),
        "raw crate graph",
    )
    raw_invalidation, _ = resolve_ref(
        raw_root,
        raw_manifest.get("invalidation_report"),
        "raw invalidation report",
    )
    artifacts = manifest.get("artifacts")
    require(
        isinstance(artifacts, dict)
        and set(artifacts)
        == {"crate_graph", "invalidation_report", "build_timing_summary"},
        "G07A checkpoint artifact set mismatch",
    )
    copied_crate_graph, _ = resolve_ref(
        root,
        artifacts["crate_graph"],
        "checkpoint crate graph",
    )
    copied_invalidation, _ = resolve_ref(
        root,
        artifacts["invalidation_report"],
        "checkpoint invalidation report",
    )
    timing_summary_path, _ = resolve_ref(
        root,
        artifacts["build_timing_summary"],
        "checkpoint build timing summary",
    )
    require(
        copied_crate_graph.read_bytes() == raw_crate_graph.read_bytes()
        and copied_invalidation.read_bytes() == raw_invalidation.read_bytes(),
        "checkpoint copies differ from canonical timing evidence",
    )
    timing_summary = read_json(
        timing_summary_path,
        "checkpoint build timing summary",
    )
    require(
        timing_summary
        == {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_g07a_build_timing_summary",
            "source": source,
            "hardware": timing["hardware"],
            "scenario_targets": timing["scenario_targets"],
            "raw_evidence": timing["manifest"],
            "raw_artifact_index_sha256": timing[
                "artifact_index_sha256"
            ],
        },
        "checkpoint build timing summary mismatch",
    )
    index = checkpoint_artifact_index(root)
    require(
        manifest.get("artifact_index") == index
        and manifest.get("artifact_index_sha256")
        == canonical_json_sha256(index),
        "G07A checkpoint artifact index mismatch",
    )
    return {
        "kind": "vnext-g07a",
        "child_manifest": external_ref(path),
        "source": source,
        "hardware_fingerprint": timing["hardware"]["fingerprint"],
        "scenario_targets": timing["scenario_targets"],
        "semantic_plan_hash": semantic["plan_hash"],
        "artifact_count": len(index),
    }


def build_checkpoint(
    *,
    g00f_path: Path,
    s1_path: Path,
    timing_root: Path,
    source_gate_path: Path,
    semantic_root: Path,
    out_dir: Path,
) -> str:
    source = clean_source()
    output = out_dir.expanduser().resolve()
    require(
        output != REPO_ROOT and REPO_ROOT not in output.parents,
        "G07A output must be outside the source tree",
    )
    require(
        not output.exists() or not any(output.iterdir()),
        f"G07A output is not empty: {output}",
    )
    g00f = validate_g00f(g00f_path, source)
    s1 = validate_s1(s1_path, source)
    source_gate = validate_source_gate(source_gate_path, source)
    semantic = validate_semantic_plan(semantic_root, source)
    timing = validate_timing_evidence(
        timing_root,
        source,
        g00f,
        s1,
    )
    raw_root = Path(timing["artifact_root"])
    raw_manifest = read_json(
        raw_root / "evidence.manifest.json",
        "G07A raw timing manifest",
    )
    raw_crate_graph, _ = resolve_ref(
        raw_root,
        raw_manifest.get("crate_graph"),
        "raw crate graph",
    )
    raw_invalidation, _ = resolve_ref(
        raw_root,
        raw_manifest.get("invalidation_report"),
        "raw invalidation report",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        crate_graph = staging / "crate-graph.json"
        invalidation = staging / "invalidation-report.json"
        shutil.copy2(raw_crate_graph, crate_graph)
        shutil.copy2(raw_invalidation, invalidation)
        timing_summary_path = staging / "build-timings/summary.json"
        write_json(
            timing_summary_path,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "runtime_vnext_g07a_build_timing_summary",
                "source": source,
                "hardware": timing["hardware"],
                "scenario_targets": timing["scenario_targets"],
                "raw_evidence": timing["manifest"],
                "raw_artifact_index_sha256": timing[
                    "artifact_index_sha256"
                ],
            },
        )
        artifacts = {
            "crate_graph": copied_artifact_ref(staging, crate_graph),
            "invalidation_report": copied_artifact_ref(
                staging,
                invalidation,
            ),
            "build_timing_summary": copied_artifact_ref(
                staging,
                timing_summary_path,
            ),
        }
        pass_line = f"{PASS_PREFIX}: {output}"
        index = checkpoint_artifact_index(staging)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_g07a_build_iteration_checkpoint",
            "checkpoint_id": "G07A",
            "lane": "runtime-vnext-g07a",
            "status": "pass",
            "canonical": True,
            "artifact_dir": str(output),
            "source": source,
            "dependencies": {
                "g00f": g00f,
                "s1": s1,
                "source_gate": source_gate,
                "semantic_plan": semantic,
            },
            "timing_evidence": timing,
            "artifacts": artifacts,
            "artifact_index": index,
            "artifact_index_sha256": canonical_json_sha256(index),
            "unlocks": UNLOCKS,
            "does_not_prove": DOES_NOT_PROVE,
            "pass_line": pass_line,
        }
        write_json(staging / "manifest.json", manifest)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
        verify_checkpoint_manifest(
            output / "manifest.json",
            verify_checkout=True,
        )
        return pass_line
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        raise


def expect_reject(action: Any, label: str) -> None:
    try:
        action()
    except CheckpointError:
        return
    raise AssertionError(f"{label} unexpectedly passed")


def self_test() -> int:
    require(
        timing_collector.EXPECTED_SCENARIOS
        == tuple(row[0] for row in timing_validator.EXPECTED_SCENARIOS),
        "G07A collector/validator scenario matrix drifted",
    )
    require(
        UNLOCKS == ["G07B", "S4"]
        and set(DOES_NOT_PROVE)
        == {"G07B", "G07", "G08", "G09", "G10", "release readiness"},
        "G07A checkpoint proof boundary drifted",
    )
    with tempfile.TemporaryDirectory(prefix="g07a-checkpoint-selftest-") as raw:
        root = Path(raw)
        payload = root / "payload.json"
        payload.write_text('{"ok":true}\n', encoding="ascii")
        ref = {
            "path": "payload.json",
            "sha256": sha256(payload),
            "size_bytes": payload.stat().st_size,
        }
        resolved, _ = resolve_ref(root, ref, "selftest payload")
        require(
            resolved == payload.resolve(),
            "selftest payload did not resolve",
        )
        bad_sha = dict(ref)
        bad_sha["sha256"] = "0" * 64
        expect_reject(
            lambda: resolve_ref(root, bad_sha, "tampered payload"),
            "tampered payload",
        )
        bad_path = dict(ref)
        bad_path["path"] = "../payload.json"
        expect_reject(
            lambda: resolve_ref(root, bad_path, "escaping payload"),
            "escaping payload",
        )
        source = {
            "git_sha": "1" * 40,
            "git_tree_sha": "2" * 40,
            "dirty": False,
            "status_short": [],
        }
        require_source(source, source, "selftest")
        stale = dict(source)
        stale["git_sha"] = "3" * 40
        expect_reject(
            lambda: require_source(stale, source, "stale"),
            "stale source",
        )
        dirty = dict(source)
        dirty["dirty"] = True
        expect_reject(
            lambda: require_source(dirty, {}, "dirty"),
            "dirty source",
        )
        (root / "run_gate.child.stdout").write_text(
            "outer runner output\n",
            encoding="ascii",
        )
        require(
            "run_gate.child.stdout" not in checkpoint_artifact_index(root),
            "outer runner artifact leaked into the child artifact index",
        )
    print(SELFTEST_PASS_LINE)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--g00f", type=Path)
    result.add_argument("--s1-artifact", type=Path)
    result.add_argument("--g07a-evidence-root", type=Path)
    result.add_argument("--source-gate", type=Path)
    result.add_argument("--semantic-plan-equivalence", type=Path)
    result.add_argument("--out", type=Path)
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        try:
            return self_test()
        except (AssertionError, CheckpointError, OSError) as error:
            print(
                f"{SELFTEST_PASS_LINE} REJECT: {error}",
                file=sys.stderr,
            )
            return 1
    required = {
        "--g00f": args.g00f,
        "--s1-artifact": args.s1_artifact,
        "--g07a-evidence-root": args.g07a_evidence_root,
        "--source-gate": args.source_gate,
        "--semantic-plan-equivalence": args.semantic_plan_equivalence,
        "--out": args.out,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser().error(f"required arguments are missing: {', '.join(missing)}")
    try:
        print(
            build_checkpoint(
                g00f_path=args.g00f,
                s1_path=args.s1_artifact,
                timing_root=args.g07a_evidence_root,
                source_gate_path=args.source_gate,
                semantic_root=args.semantic_plan_equivalence,
                out_dir=args.out,
            )
        )
        return 0
    except (
        CheckpointError,
        OSError,
        subprocess.SubprocessError,
        timing_validator.VerificationError,
    ) as error:
        print(
            f"{PASS_PREFIX} FAIL: {args.out}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
