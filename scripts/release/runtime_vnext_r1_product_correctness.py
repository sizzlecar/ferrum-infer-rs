#!/usr/bin/env python3
"""Aggregate the release-blocking R1 product correctness evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import runtime_vnext_g08a_matrix_specs as g08a_specs
import runtime_vnext_g08b_cuda_matrix_checkpoint as matrix_checkpoint
import runtime_vnext_g08c_cuda_matrix_checkpoint as g08c_specs
import runtime_vnext_r0_core_closure as r0_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PASS_PREFIX = "FERRUM RUNTIME VNEXT R1 PRODUCT CORRECTNESS PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT R1 PRODUCT CORRECTNESS SELFTEST PASS"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VNEXT_STARTUP_MARKER = "Building registered model from a typed vNext execution plan"
LLAMA_SCENARIOS = (
    ("run_multiturn_recall", "run_multiturn"),
    ("serve_multiturn_recall", "serve_multiturn_recall"),
    ("serve_stream_done_usage", "serve_stream"),
)
LLAMA_MANIFESTS = {
    "cuda": REPO_ROOT
    / "scripts/release/scenarios/runtime_vnext_r1_llama_dense_cuda.json",
    "metal": REPO_ROOT
    / "scripts/release/scenarios/runtime_vnext_r1_llama_dense_metal.json",
}
LLAMA_MODEL_MARKERS = {
    "cuda": "Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
    "metal": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
}
CONTROL_PLANE_FILES = frozenset(
    {
        "scripts/release/run_gate.py",
        "scripts/release/runtime_vnext_r1_product_correctness.py",
        "scripts/release/scenarios/runtime_vnext_r1_llama_dense_cuda.json",
        "scripts/release/scenarios/runtime_vnext_r1_llama_dense_metal.json",
    }
)
MATRIX_EVIDENCE_CLOSURE_FILES = frozenset(
    {
        "docs/goals/runtime-vnext-0.8.0-2026-07-10/CORRECTNESS_ACCEPTANCE_AMENDMENT_2026-08-07.md",
        "docs/goals/runtime-vnext-0.8.0-2026-07-10/MODEL_MATRIX.md",
        "docs/goals/runtime-vnext-0.8.0-2026-07-10/RELEASE_ACCELERATION_AMENDMENT_2026-08-06.md",
        "scripts/release/runtime_vnext_baseline_scenarios.py",
        "scripts/release/runtime_vnext_g08b_cuda_matrix_checkpoint.py",
        "scripts/release/runtime_vnext_g08c_cuda_matrix_checkpoint.py",
        "scripts/release/runtime_vnext_r1_product_correctness.py",
    }
)
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
MATRIX_CHILD_FIELDS = {
    "artifact_dir",
    "artifact_type",
    "canonical",
    "checkpoint_id",
    "dirty",
    "does_not_prove",
    "lane",
    "pass_line",
    "scenario_report",
    "schema_version",
    "source_git_sha",
    "source_tree_sha",
    "status",
    "summary",
    "validation",
}
DOES_NOT_PROVE = [
    "R2 performance, profile overhead, or build-time acceptance",
    "R3 release assets, publication, or installed regression",
    "v0.8.0 release readiness",
]


class R1Error(RuntimeError):
    pass


@dataclass(frozen=True)
class MatrixLane:
    key: str
    lane: str
    script: str
    model_key: str
    backend: str
    spec: matrix_checkpoint.CheckpointSpec


MATRIX_LANES = {
    "m1_cuda": MatrixLane(
        "m1_cuda",
        "vnext-g08a-cuda",
        "scripts/release/runtime_vnext_g08a_cuda_matrix_checkpoint.py",
        "m1-qwen35-4b",
        "cuda",
        g08a_specs.CHECKPOINT_SPECS["cuda"],
    ),
    "m1_metal": MatrixLane(
        "m1_metal",
        "vnext-g08a-metal",
        "scripts/release/runtime_vnext_g08a_metal_matrix_checkpoint.py",
        "m1-qwen35-4b",
        "metal",
        g08a_specs.CHECKPOINT_SPECS["metal"],
    ),
    "m2_cuda": MatrixLane(
        "m2_cuda",
        "vnext-g08b-cuda",
        "scripts/release/runtime_vnext_g08b_cuda_matrix_checkpoint.py",
        "m2-qwen35-35b-a3b",
        "cuda",
        matrix_checkpoint.CHECKPOINT_SPECS["cuda"],
    ),
    "m2_metal": MatrixLane(
        "m2_metal",
        "vnext-g08b-metal",
        "scripts/release/runtime_vnext_g08b_metal_matrix_checkpoint.py",
        "m2-qwen35-35b-a3b",
        "metal",
        matrix_checkpoint.CHECKPOINT_SPECS["metal"],
    ),
    "m3_cuda": MatrixLane(
        "m3_cuda",
        "vnext-g08c-cuda",
        "scripts/release/runtime_vnext_g08c_cuda_matrix_checkpoint.py",
        "m3-qwen3-30b-a3b",
        "cuda",
        g08c_specs.CHECKPOINT_SPECS["cuda"],
    ),
    "m3_metal": MatrixLane(
        "m3_metal",
        "vnext-g08c-metal",
        "scripts/release/runtime_vnext_g08c_metal_matrix_checkpoint.py",
        "m3-qwen3-30b-a3b",
        "metal",
        g08c_specs.CHECKPOINT_SPECS["metal"],
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R1Error(message)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise R1Error(f"invalid {label} JSON {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(
        resolved.is_file() and not resolved.is_symlink(),
        f"artifact is missing: {resolved}",
    )
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_ref(value: Any, label: str) -> Path:
    require(
        isinstance(value, dict)
        and set(value) == {"path", "sha256", "size_bytes"},
        f"{label} reference fields differ",
    )
    path = Path(str(value["path"])).expanduser().resolve()
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    require(path.stat().st_size == value["size_bytes"], f"{label} size mismatch")
    require(sha256(path) == value["sha256"], f"{label} SHA256 mismatch")
    return path


def require_sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{label} is invalid")
    return value


def git_text(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(
        process.returncode == 0,
        f"git {' '.join(args)} failed: {process.stderr.strip()}",
    )
    return process.stdout.strip()


def current_source() -> dict[str, Any]:
    status = [line for line in git_text("status", "--short").splitlines() if line]
    require(not status, f"R1 source must be clean: {status[:8]}")
    return {
        "git_sha": git_text("rev-parse", "HEAD"),
        "git_tree_sha": git_text("rev-parse", "HEAD^{tree}"),
        "dirty": False,
    }


def normalize_source(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} source is missing")
    source = {
        "git_sha": value.get("git_sha"),
        "git_tree_sha": value.get("git_tree_sha"),
        "dirty": value.get("dirty"),
    }
    require(
        isinstance(source["git_sha"], str)
        and GIT_SHA_RE.fullmatch(source["git_sha"]) is not None
        and isinstance(source["git_tree_sha"], str)
        and GIT_SHA_RE.fullmatch(source["git_tree_sha"]) is not None
        and source["dirty"] is False,
        f"{label} source identity is invalid",
    )
    return source


def source_closure(recorded: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    recorded_sha = recorded["git_sha"]
    require(
        git_text("rev-parse", f"{recorded_sha}^{{tree}}")
        == recorded["git_tree_sha"],
        "R0 recorded source tree differs from git",
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", recorded_sha, current["git_sha"]],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    require(ancestor.returncode == 0, "R0 source is not an ancestor of R1 source")
    changed = [
        line
        for line in git_text(
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            f"{recorded_sha}..{current['git_sha']}",
        ).splitlines()
        if line
    ]
    rejected = [
        path
        for path in changed
        if not path.startswith("docs/") and path not in CONTROL_PLANE_FILES
    ]
    require(
        not rejected,
        f"R0 is stale after product or non-R1 validator changes: {rejected[:8]}",
    )
    return {
        "from_git_sha": recorded_sha,
        "to_git_sha": current["git_sha"],
        "changed_files": changed,
        "changed_file_count": len(changed),
        "policy": "r1-control-plane-only",
    }


def matrix_source_closure(
    recorded: dict[str, Any],
    current: dict[str, Any],
    lane: MatrixLane,
) -> dict[str, Any]:
    recorded_sha = recorded["git_sha"]
    require(
        git_text("rev-parse", f"{recorded_sha}^{{tree}}")
        == recorded["git_tree_sha"],
        f"{lane.key} recorded source tree differs from git",
    )
    if recorded == current:
        return {
            "from_git_sha": recorded_sha,
            "to_git_sha": current["git_sha"],
            "changed_files": [],
            "changed_file_count": 0,
            "policy": "exact-source",
        }
    require(
        lane.model_key == "m1-qwen35-4b",
        f"{lane.key} matrix evidence must match current source exactly",
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", recorded_sha, current["git_sha"]],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    require(
        ancestor.returncode == 0,
        f"{lane.key} recorded source is not an ancestor of current source",
    )
    changed = [
        line
        for line in git_text(
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            f"{recorded_sha}..{current['git_sha']}",
        ).splitlines()
        if line
    ]
    rejected = [
        path for path in changed if path not in MATRIX_EVIDENCE_CLOSURE_FILES
    ]
    require(
        not rejected,
        f"{lane.key} matrix evidence is stale after non-control-plane changes: {rejected[:8]}",
    )
    require(
        changed,
        f"{lane.key} source identity differs without an observable git diff",
    )
    return {
        "from_git_sha": recorded_sha,
        "to_git_sha": current["git_sha"],
        "changed_files": changed,
        "changed_file_count": len(changed),
        "policy": "m1-r1-matrix-control-plane-only",
    }


def resolve_member(root: Path, recorded_root: Path, raw: Any, label: str) -> Path:
    require(isinstance(raw, str) and raw, f"{label} path is invalid")
    path = Path(raw)
    if path.is_absolute():
        try:
            relative = path.relative_to(recorded_root)
        except ValueError as error:
            raise R1Error(f"{label} escaped recorded artifact root: {path}") from error
    else:
        relative = path
    require(not relative.is_absolute() and ".." not in relative.parts, f"{label} path escapes")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise R1Error(f"{label} escaped artifact root: {resolved}") from error
    return resolved


def validate_portable_ref(
    value: Any, root: Path, recorded_root: Path, label: str
) -> Path:
    require(isinstance(value, dict), f"{label} reference is missing")
    require(
        set(value) >= {"path", "sha256"}, f"{label} reference fields are incomplete"
    )
    path = resolve_member(root, recorded_root, value["path"], label)
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    require(sha256(path) == value["sha256"], f"{label} SHA256 mismatch")
    if "size_bytes" in value:
        require(path.stat().st_size == value["size_bytes"], f"{label} size mismatch")
    return path


def validate_relative_ref(value: Any, root: Path, label: str) -> Path:
    require(isinstance(value, dict), f"{label} reference is missing")
    require(set(value) >= {"path", "sha256"}, f"{label} reference fields differ")
    raw = Path(str(value["path"]))
    require(not raw.is_absolute() and ".." not in raw.parts, f"{label} path must be relative")
    path = (root / raw).resolve()
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    require(sha256(path) == value["sha256"], f"{label} SHA256 mismatch")
    return path


def command_flag(command: list[str], flag: str, label: str) -> str:
    require(command.count(flag) == 1, f"{label} must contain {flag} exactly once")
    index = command.index(flag)
    require(index + 1 < len(command), f"{label} {flag} value is missing")
    value = command[index + 1]
    require(value and not value.startswith("--"), f"{label} {flag} value is invalid")
    return value


def recorded_and_actual_roots(
    actual_out: Path, recorded_out: Path, recorded_artifact: Path
) -> tuple[Path, Path]:
    try:
        relative_out = recorded_out.relative_to(recorded_artifact)
    except ValueError as error:
        raise R1Error("matrix gate output is outside its recorded artifact root") from error
    actual_artifact = actual_out
    for _ in relative_out.parts:
        actual_artifact = actual_artifact.parent
    require(
        (actual_artifact / relative_out).resolve() == actual_out.resolve(),
        "matrix artifact relocation is inconsistent",
    )
    return recorded_artifact, actual_artifact.resolve()


def validate_outer_receipts(
    outer: dict[str, Any], actual_root: Path, delegated: list[str], expected_pass: str
) -> None:
    refs = outer.get("child_execution_artifacts")
    require(isinstance(refs, list) and len(refs) == 3, "outer execution receipt set mismatch")
    expected = {
        "run_gate.child.command.json",
        "run_gate.child.stdout",
        "run_gate.child.stderr",
    }
    require(
        {row.get("path") for row in refs if isinstance(row, dict)} == expected,
        "outer execution receipt names mismatch",
    )
    paths: dict[str, Path] = {}
    for row in refs:
        require(isinstance(row, dict), "outer execution receipt is invalid")
        paths[str(row["path"])] = validate_relative_ref(
            row, actual_root, f"outer {row['path']}"
        )
    command = read_json(paths["run_gate.child.command.json"], "child command receipt")
    require(command.get("cmd") == delegated, "child command receipt differs")
    stdout = paths["run_gate.child.stdout"].read_text(encoding="utf-8")
    require(
        stdout.splitlines().count(expected_pass) == 1,
        "child stdout lacks exactly one required PASS line",
    )


def validate_r0(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    outer_path = path.expanduser().resolve()
    outer = read_json(outer_path, "R0 outer manifest")
    require(set(outer) == OUTER_GATE_FIELDS, "R0 outer field set mismatch")
    actual_root = outer_path.parent
    recorded_root = Path(str(outer.get("artifact_dir", "")))
    require(recorded_root.is_absolute(), "R0 recorded artifact root is invalid")
    require(
        outer.get("schema_version") == 1
        and outer.get("lane") == "vnext-r0"
        and outer.get("status") == "pass"
        and outer.get("child_returncode") == 0
        and outer.get("error") is None
        and outer.get("dirty_status") == {"is_dirty": False, "status_short": []},
        "R0 outer identity/status mismatch",
    )
    expected_child_pass = f"FERRUM RUNTIME VNEXT R0 CORE CLOSURE PASS: {recorded_root}"
    require(outer.get("child_pass_line") == expected_child_pass, "R0 child PASS mismatch")
    require(
        outer.get("pass_line") == f"FERRUM GATE vnext-r0 PASS: {recorded_root}",
        "R0 outer PASS mismatch",
    )
    delegated = outer.get("delegated_command_line")
    require(isinstance(delegated, list), "R0 delegated command is missing")
    require(
        len(delegated) >= 3
        and Path(str(delegated[1])).as_posix().endswith(
            "scripts/release/runtime_vnext_r0_core_closure.py"
        )
        and Path(command_flag(delegated, "--out", "R0 delegated command"))
        == recorded_root,
        "R0 delegated command identity mismatch",
    )
    validate_outer_receipts(outer, actual_root, delegated, expected_child_pass)
    artifacts = outer.get("child_artifacts")
    require(isinstance(artifacts, dict), "R0 child provenance is missing")
    child_ref = artifacts.get("child_manifest")
    child_path = validate_portable_ref(
        child_ref, actual_root, recorded_root, "R0 child manifest"
    )
    require(child_path == actual_root / "manifest.json", "R0 child manifest path mismatch")
    child = read_json(child_path, "R0 child manifest")
    child_source = normalize_source(child.get("source"), "R0 aggregate")
    require(
        child.get("schema_version") == 1
        and child.get("artifact_type") == "runtime_vnext_r0_core_closure_manifest"
        and child.get("checkpoint_id") == "R0"
        and child.get("lane") == "runtime-vnext-r0"
        and child.get("status") == "pass"
        and child.get("canonical") is True
        and child.get("artifact_dir") == str(recorded_root)
        and child.get("pass_line") == expected_child_pass
        and child.get("unlocks") == ["R1"],
        "R0 child identity/status mismatch",
    )
    dependencies = child.get("dependencies")
    require(
        isinstance(dependencies, dict)
        and set(dependencies) == set(r0_checkpoint.DEPENDENCY_KEYS),
        "R0 child dependency set mismatch",
    )
    require(
        child.get("acceptance") == r0_checkpoint.acceptance(dependencies),
        "R0 child acceptance drifted",
    )
    require(
        artifacts.get("source") == child_source
        and artifacts.get("acceptance") == child["acceptance"],
        "R0 outer provenance summary differs from child",
    )
    closure = source_closure(child_source, source)
    return {
        "outer_manifest": file_ref(outer_path),
        "child_manifest": file_ref(child_path),
        "source": child_source,
        "source_closure": closure,
        "acceptance": copy.deepcopy(child["acceptance"]),
    }


def expected_matrix_summary(spec: matrix_checkpoint.CheckpointSpec) -> dict[str, Any]:
    return {
        "case_count": spec.expected_case_count,
        "client_concurrency": spec.required_client_concurrency,
        "active_floor": spec.required_active_floor,
    }


def scenario_by_id(report: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    scenarios = report.get("scenarios")
    require(isinstance(scenarios, list), "matrix scenarios are missing")
    matches = [row for row in scenarios if isinstance(row, dict) and row.get("id") == scenario_id]
    require(len(matches) == 1, f"matrix must contain exactly one {scenario_id}")
    return matches[0]


def validate_startup_identity(
    report: dict[str, Any], artifact_root: Path, lane: MatrixLane
) -> dict[str, Any]:
    commands = report.get("commands")
    require(isinstance(commands, list) and len(commands) == 7, f"{lane.key} command topology differs")
    identities: set[tuple[str, str, str, str]] = set()
    marker_count = 0
    for command in commands:
        require(isinstance(command, dict), f"{lane.key} command is invalid")
        stderr_path = validate_relative_ref(
            command.get("stderr"), artifact_root, f"{lane.key} command stderr"
        )
        text = stderr_path.read_text(encoding="utf-8", errors="replace")
        lines = [line for line in text.splitlines() if VNEXT_STARTUP_MARKER in line]
        require(len(lines) == 1, f"{lane.key} command lacks one typed vNext startup marker")
        line = lines[0]
        require(f'backend="{lane.backend}"' in line, f"{lane.key} startup backend differs")
        external = re.search(r"external_metadata_id=([^ ]+)", line)
        family = re.search(r"family_id=([^ ]+)", line)
        family_fp = re.search(r'family_fingerprint="([0-9a-f]{64})"', line)
        program_fp = re.search(r'program_fingerprint="([0-9a-f]{64})"', line)
        require(
            external is not None
            and family is not None
            and family_fp is not None
            and program_fp is not None
            and "prepared_model_reused=true" in line,
            f"{lane.key} typed vNext startup identity is incomplete",
        )
        identities.add(
            (external.group(1), family.group(1), family_fp.group(1), program_fp.group(1))
        )
        marker_count += 1
    require(len(identities) == 1, f"{lane.key} run/serve resolved different typed plans")
    identity = next(iter(identities))
    return {
        "entrypoints": ["run", "serve"],
        "command_count": len(commands),
        "typed_vnext_startup_marker_count": marker_count,
        "external_metadata_id": identity[0],
        "family_id": identity[1],
        "family_fingerprint": identity[2],
        "program_fingerprint": identity[3],
        "production_legacy_selection_count": 0,
    }


def validate_provider_execution(
    report: dict[str, Any], artifact_root: Path, lane: MatrixLane
) -> dict[str, Any]:
    c18 = scenario_by_id(report, "C18")
    c18_ref = next(
        (
            ref
            for ref in c18.get("artifacts", [])
            if isinstance(ref, dict) and ref.get("kind") == "raw-json"
        ),
        None,
    )
    c18_path = validate_relative_ref(c18_ref, artifact_root, f"{lane.key} C18 raw")
    c18_raw = read_json(c18_path, f"{lane.key} C18 raw")
    cases = c18_raw.get("cases")
    require(isinstance(cases, list) and cases, f"{lane.key} C18 cases are missing")
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for case_ref in cases:
        case_path = validate_relative_ref(case_ref, artifact_root, f"{lane.key} C18 case")
        case = read_json(case_path, f"{lane.key} C18 case")
        observed = case.get("observed")
        require(isinstance(observed, dict), f"{lane.key} C18 observed result is missing")
        concurrency = observed.get("requested_concurrency")
        require(isinstance(concurrency, int) and concurrency > 0, f"{lane.key} C18 concurrency is invalid")
        candidates.append((concurrency, case_path, case))
    concurrency, case_path, case = max(candidates, key=lambda row: row[0])
    require(
        concurrency == lane.spec.required_client_concurrency,
        f"{lane.key} C18 top concurrency differs",
    )
    artifacts = case.get("artifacts")
    require(isinstance(artifacts, dict), f"{lane.key} C18 case artifacts are missing")
    transcript_path = validate_relative_ref(
        artifacts.get("http_transcript"), artifact_root, f"{lane.key} C18 transcript"
    )
    transcript = read_json(transcript_path, f"{lane.key} C18 transcript")
    trace_rows = transcript.get("scheduler_trace_rows")
    require(isinstance(trace_rows, list) and trace_rows, f"{lane.key} C18 trace is missing")
    phase_sets = {
        "vnext.node_started": set(),
        "vnext.operation_submitted": set(),
        "vnext.node_retired": set(),
    }
    operations: set[str] = set()
    event_count = 0
    for wrapper in trace_rows:
        require(isinstance(wrapper, dict), f"{lane.key} trace wrapper is invalid")
        row = wrapper.get("raw")
        if not isinstance(row, dict) or row.get("phase") not in phase_sets:
            continue
        attributes = row.get("attributes")
        require(isinstance(attributes, dict), f"{lane.key} provider event lacks attributes")
        provider = attributes.get("provider_id")
        require(
            isinstance(provider, str)
            and provider.startswith(f"provider.{lane.backend}."),
            f"{lane.key} provider identity differs: {provider!r}",
        )
        require(
            row.get("backend") == "actual"
            and row.get("status") == "ok"
            and attributes.get("execution_trace_source") == "vnext"
            and attributes.get("actual_model_smoke") is True
            and attributes.get("diagnostic_only") is False
            and attributes.get("l0_only") is False,
            f"{lane.key} provider event is not actual vNext execution",
        )
        phase_sets[str(row["phase"])].add(provider)
        operation = attributes.get("operation_id")
        if row["phase"] == "vnext.operation_submitted":
            require(isinstance(operation, str) and operation, f"{lane.key} submitted operation lacks identity")
            operations.add(operation)
        event_count += 1
    selected = phase_sets["vnext.node_started"]
    submitted = phase_sets["vnext.operation_submitted"]
    retired = phase_sets["vnext.node_retired"]
    require(
        len(selected) >= 4 and selected == submitted == retired,
        f"{lane.key} selected/submitted/retired provider sets differ",
    )
    return {
        "provider_ids": sorted(submitted),
        "provider_count": len(submitted),
        "operation_ids": sorted(operations),
        "operation_count": len(operations),
        "provider_event_count": event_count,
        "conformance": "selected-submitted-retired",
        "c18_raw": file_ref(c18_path),
        "c18_case": file_ref(case_path),
        "c18_transcript": file_ref(transcript_path),
    }


def validate_resource_contract(
    report: dict[str, Any], artifact_root: Path, lane: MatrixLane
) -> dict[str, Any]:
    c09 = scenario_by_id(report, "C09")
    raw_ref = next(
        (
            ref
            for ref in c09.get("artifacts", [])
            if isinstance(ref, dict) and ref.get("kind") == "raw-json"
        ),
        None,
    )
    raw_path = validate_relative_ref(raw_ref, artifact_root, f"{lane.key} C09 raw")
    raw = read_json(raw_path, f"{lane.key} C09 raw")
    assertions = raw.get("assertions")
    require(isinstance(assertions, dict), f"{lane.key} C09 assertions are missing")
    case_count = raw.get("case_count")
    require(isinstance(case_count, int) and case_count > 0, f"{lane.key} C09 denominator is invalid")
    require(
        raw.get("status") == "pass"
        and raw.get("blocked_count") == 0
        and assertions.get("resource_final_state") == "released"
        and assertions.get("admitted_released_count") == case_count
        and assertions.get("post_capacity_success_count") == case_count
        and assertions.get("post_capacity_failure_count") == 0
        and assertions.get("leak_count") == 0
        and assertions.get("double_release_count") == 0,
        f"{lane.key} cancel/release/capacity contract failed",
    )
    return {
        "cancel_case_count": case_count,
        "admitted_released_count": case_count,
        "post_capacity_success_count": case_count,
        "resource_final_state": "released",
        "leak_count": 0,
        "double_release_count": 0,
        "c09_raw": file_ref(raw_path),
    }


def validate_matrix(path: Path, lane: MatrixLane, source: dict[str, Any]) -> dict[str, Any]:
    outer_path = path.expanduser().resolve()
    outer = read_json(outer_path, f"{lane.key} outer manifest")
    require(set(outer) == OUTER_GATE_FIELDS, f"{lane.key} outer field set mismatch")
    actual_out = outer_path.parent
    recorded_out = Path(str(outer.get("artifact_dir", "")))
    require(recorded_out.is_absolute(), f"{lane.key} recorded output is invalid")
    require(
        outer.get("schema_version") == 1
        and outer.get("lane") == lane.lane
        and outer.get("status") == "pass"
        and outer.get("child_returncode") == 0
        and outer.get("error") is None
        and isinstance(outer.get("git_sha"), str)
        and GIT_SHA_RE.fullmatch(outer["git_sha"]) is not None
        and outer.get("dirty_status") == {"is_dirty": False, "status_short": []},
        f"{lane.key} outer identity/source/status mismatch",
    )
    recorded_git_sha = outer["git_sha"]
    delegated = outer.get("delegated_command_line")
    require(isinstance(delegated, list) and len(delegated) >= 8, f"{lane.key} delegated command is missing")
    require(
        Path(str(delegated[1])).as_posix().endswith(lane.script),
        f"{lane.key} delegated validator identity differs",
    )
    recorded_artifact = Path(
        command_flag(delegated, "--artifact-root", f"{lane.key} delegated command")
    )
    recorded_report = Path(
        command_flag(delegated, "--scenario-report", f"{lane.key} delegated command")
    )
    require(
        Path(command_flag(delegated, "--out", f"{lane.key} delegated command"))
        == recorded_out,
        f"{lane.key} delegated output differs",
    )
    _, actual_artifact = recorded_and_actual_roots(
        actual_out, recorded_out, recorded_artifact
    )
    try:
        report_relative = recorded_report.relative_to(recorded_artifact)
    except ValueError as error:
        raise R1Error(f"{lane.key} report escaped recorded artifact root") from error
    actual_report = (actual_artifact / report_relative).resolve()
    child_prefix = lane.spec.pass_prefix
    expected_child_pass = f"{child_prefix}: {recorded_out}"
    require(outer.get("child_pass_line") == expected_child_pass, f"{lane.key} child PASS mismatch")
    require(
        outer.get("pass_line") == f"FERRUM GATE {lane.lane} PASS: {recorded_out}",
        f"{lane.key} outer PASS mismatch",
    )
    validate_outer_receipts(outer, actual_out, delegated, expected_child_pass)
    child_artifacts = outer.get("child_artifacts")
    require(isinstance(child_artifacts, dict), f"{lane.key} child provenance is missing")
    child_path = validate_portable_ref(
        child_artifacts.get("child_manifest"),
        actual_out,
        recorded_out,
        f"{lane.key} child manifest",
    )
    require(child_path == actual_out / "manifest.json", f"{lane.key} child path differs")
    child = read_json(child_path, f"{lane.key} child manifest")
    require(set(child) == MATRIX_CHILD_FIELDS, f"{lane.key} child field set mismatch")
    require(
        child.get("schema_version") == 1
        and child.get("status") == "pass"
        and child.get("canonical") is True
        and child.get("artifact_dir") == str(recorded_out)
        and child.get("source_git_sha") == recorded_git_sha
        and isinstance(child.get("source_tree_sha"), str)
        and GIT_SHA_RE.fullmatch(child["source_tree_sha"]) is not None
        and child.get("dirty") is False
        and child.get("pass_line") == expected_child_pass,
        f"{lane.key} child identity/source/status mismatch",
    )
    recorded_source = {
        "git_sha": recorded_git_sha,
        "git_tree_sha": child["source_tree_sha"],
        "dirty": False,
    }
    closure = matrix_source_closure(recorded_source, source, lane)
    report_path = validate_portable_ref(
        child.get("scenario_report"),
        actual_artifact,
        recorded_artifact,
        f"{lane.key} scenario report",
    )
    require(report_path == actual_report, f"{lane.key} scenario report command binding differs")
    report = read_json(report_path, f"{lane.key} scenario report")
    summary = child.get("summary")
    requirement = expected_matrix_summary(lane.spec)
    require(
        report.get("status") == "pass"
        and report.get("execution_contract") == matrix_checkpoint.matrix.G08_EXECUTION_CONTRACT
        and report.get("source_git_sha") == recorded_source["git_sha"]
        and report.get("source_tree_sha") == recorded_source["git_tree_sha"]
        and report.get("dirty_status") == {"is_dirty": False, "status_short": []}
        and report.get("model_key") == lane.model_key
        and report.get("backend") == lane.backend,
        f"{lane.key} report identity/source/status mismatch",
    )
    require(isinstance(summary, dict), f"{lane.key} child summary is missing")
    require(
        summary.get("scenario_count") == 21
        and summary.get("case_count") == requirement["case_count"]
        and summary.get("passed_case_count") == requirement["case_count"]
        and summary.get("entrypoints") == ["run", "serve"],
        f"{lane.key} matrix denominator or entrypoints differ",
    )
    for field in ("known_failed_count", "blocked_count", "error_count", "unexpected_count"):
        require(summary.get(field) == 0, f"{lane.key} {field} must be zero")
    c18 = summary.get("c18")
    require(isinstance(c18, dict), f"{lane.key} C18 summary is missing")
    balance = c18.get("resource_balance")
    require(
        c18.get("requested_concurrency") == requirement["client_concurrency"]
        and c18.get("active_floor") == requirement["active_floor"]
        and c18.get("observed_max_active", 0) >= requirement["active_floor"]
        and c18.get("active_duty_cycle", 0.0) >= lane.spec.required_active_duty_cycle
        and isinstance(balance, dict)
        and balance.get("leaked_resource_count") == 0
        and balance.get("underflow_count") == 0,
        f"{lane.key} C18 admission/resource contract failed",
    )
    validation_path = validate_portable_ref(
        child.get("validation"), actual_out, recorded_out, f"{lane.key} validation"
    )
    validation = read_json(validation_path, f"{lane.key} validation")
    require(
        validation.get("status") == "pass"
        and validation.get("source_git_sha") == recorded_source["git_sha"]
        and validation.get("source_tree_sha") == recorded_source["git_tree_sha"]
        and validation.get("model_key") == lane.model_key
        and validation.get("backend") == lane.backend
        and validation.get("summary") == summary
        and validation.get("binary_sha256") == report.get("binary_sha256")
        and validation.get("scenario_report", {}).get("sha256") == sha256(report_path),
        f"{lane.key} validation/child/report binding mismatch",
    )
    require(
        report.get("models_lock_sha256") == sha256(lane.spec.model_lock_path),
        f"{lane.key} model lock differs from current source",
    )
    startup = validate_startup_identity(report, actual_artifact, lane)
    providers = validate_provider_execution(report, actual_artifact, lane)
    resources = validate_resource_contract(report, actual_artifact, lane)
    return {
        "lane": lane.lane,
        "model_key": lane.model_key,
        "backend": lane.backend,
        "outer_manifest": file_ref(outer_path),
        "child_manifest": file_ref(child_path),
        "validation": file_ref(validation_path),
        "scenario_report": file_ref(report_path),
        "binary_sha256": require_sha(report.get("binary_sha256"), f"{lane.key} binary SHA"),
        "hardware_id": report.get("hardware_id"),
        "summary": copy.deepcopy(summary),
        "product_execution_identity": startup,
        "provider_execution": providers,
        "resource_contract": resources,
        "source_closure": closure,
    }


def validate_self_hash(value: dict[str, Any], label: str) -> None:
    digest = value.get("canonical_sha256")
    require_sha(digest, f"{label} canonical SHA")
    candidate = copy.deepcopy(value)
    candidate.pop("canonical_sha256")
    require(
        candidate.get("canonical_sha256_scope")
        == "document_without_canonical_sha256_fields"
        and json_sha256(candidate) == digest,
        f"{label} canonical SHA mismatch",
    )


def validate_artifact_tree(root: Path, recorded: Path) -> dict[str, Any]:
    tree_path = root / "artifact_tree.json"
    tree = read_json(tree_path, "Llama artifact tree")
    validate_self_hash(tree, "Llama artifact tree")
    require(
        tree.get("schema_version") == 1
        and tree.get("artifact_root") == str(recorded),
        "Llama artifact tree identity mismatch",
    )
    rows = tree.get("files")
    require(isinstance(rows, list) and tree.get("file_count") == len(rows), "Llama artifact tree count mismatch")
    indexed: set[str] = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"path", "size", "sha256"}, "Llama artifact tree row is invalid")
        relative = row.get("path")
        require(isinstance(relative, str) and relative not in indexed, "Llama artifact tree path is invalid")
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"Llama artifact is missing: {relative}")
        require(path.stat().st_size == row["size"] and sha256(path) == row["sha256"], f"Llama artifact binding mismatch: {relative}")
        indexed.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "artifact_tree.json"
    }
    require(indexed == actual, "Llama artifact tree coverage mismatch")
    return file_ref(tree_path)


def validate_llama_manifest(backend: str) -> dict[str, Any]:
    path = LLAMA_MANIFESTS[backend]
    manifest = read_json(path, f"Llama {backend} scenario manifest")
    require(
        manifest.get("schema_version") == 1
        and manifest.get("backend") == backend
        and LLAMA_MODEL_MARKERS[backend] in str(manifest.get("model"))
        and manifest.get("server") == {"args": ["--backend", backend], "mode": "start"},
        f"Llama {backend} manifest identity/default path mismatch",
    )
    scenarios = manifest.get("scenarios")
    require(isinstance(scenarios, list), f"Llama {backend} manifest scenarios are missing")
    require(
        [(row.get("name"), row.get("type")) for row in scenarios if isinstance(row, dict)]
        == list(LLAMA_SCENARIOS),
        f"Llama {backend} manifest scenario order differs",
    )
    run = scenarios[0]
    require(
        run.get("use_default_max_tokens") is True
        and "max_tokens" not in run
        and run.get("min_assistant_turns") == 2,
        f"Llama {backend} run must exercise the visible default token budget",
    )
    return manifest


def validate_llama(path: Path, backend: str, source: dict[str, Any]) -> dict[str, Any]:
    candidate = path.expanduser().resolve()
    summary_path = candidate / "summary.json" if candidate.is_dir() else candidate
    root = summary_path.parent
    summary = read_json(summary_path, f"Llama {backend} summary")
    recorded = Path(str(summary.get("artifact_dir", "")))
    require(recorded.is_absolute(), f"Llama {backend} recorded root is invalid")
    validate_llama_manifest(backend)
    require(
        summary.get("schema_version") == 1
        and summary.get("status") == "pass"
        and summary.get("backend") == backend
        and LLAMA_MODEL_MARKERS[backend] in str(summary.get("model"))
        and summary.get("git_sha") == source["git_sha"]
        and summary.get("dirty_status") == {"is_dirty": False, "status_short": []}
        and summary.get("scenario_count") == 3
        and summary.get("manifest_scenario_count") == 3
        and summary.get("requested_scenarios") == []
        and summary.get("selected_scenarios") == [name for name, _ in LLAMA_SCENARIOS]
        and summary.get("failed") == 0
        and summary.get("skipped") == 0
        and summary.get("pass_line") == f"BACKEND REGRESSION SMOKE PASS: {recorded}",
        f"Llama {backend} summary identity/source/result mismatch",
    )
    rows = summary.get("scenarios")
    require(isinstance(rows, list) and len(rows) == 3, f"Llama {backend} scenario rows are missing")
    require(
        [(row.get("name"), row.get("type")) for row in rows if isinstance(row, dict)]
        == list(LLAMA_SCENARIOS),
        f"Llama {backend} result scenario order differs",
    )
    for row in rows:
        require(isinstance(row, dict) and row.get("status") == "pass", f"Llama {backend} scenario failed")
        result_path = resolve_member(root, recorded, row.get("artifact"), f"Llama {backend} result")
        require(read_json(result_path, f"Llama {backend} result") == row, f"Llama {backend} result binding mismatch")
    run, multi, stream = rows
    require(
        run.get("assistant_turns", 0) >= 2
        and run.get("length_finishes") == 0
        and run.get("used_default_max_tokens") is True,
        f"Llama {backend} ferrum run multi-turn failed",
    )
    require(
        multi.get("assistant_turns") == 2 and multi.get("recalled_secret") is True,
        f"Llama {backend} ferrum serve multi-turn failed",
    )
    require(
        stream.get("done_count") == 1
        and stream.get("content_delta_count", 0) > 0
        and stream.get("malformed_json") == 0
        and stream.get("usage_chunks") == 1
        and stream.get("errors") == []
        and isinstance(stream.get("output_text"), str)
        and bool(stream["output_text"].strip()),
        f"Llama {backend} stream contract failed",
    )
    receipt_path = root / "execution_receipt.json"
    receipt = read_json(receipt_path, f"Llama {backend} execution receipt")
    validate_self_hash(receipt, f"Llama {backend} execution receipt")
    require(
        receipt.get("schema_version") == 1
        and receipt.get("mode") == "start"
        and receipt.get("backend") == backend
        and receipt.get("model") == summary.get("model")
        and receipt.get("git_sha") == source["git_sha"]
        and receipt.get("dirty_status") == summary["dirty_status"]
        and receipt.get("selected_scenarios") == summary["selected_scenarios"]
        and receipt.get("scenario_count") == 3
        and receipt.get("failed") == 0
        and receipt.get("skipped") == 0,
        f"Llama {backend} execution receipt mismatch",
    )
    child_env = receipt.get("child_env")
    require(
        isinstance(child_env, dict)
        and not any(str(key).startswith("FERRUM_") for key in child_env),
        f"Llama {backend} used hidden FERRUM environment",
    )
    inputs = receipt.get("input_artifacts")
    require(isinstance(inputs, dict) and set(inputs) == {"runner", "manifest"}, f"Llama {backend} input set differs")
    expected_inputs = {
        "runner": REPO_ROOT / "scripts/release/run_scenarios.py",
        "manifest": LLAMA_MANIFESTS[backend],
    }
    for key, current in expected_inputs.items():
        item = inputs[key]
        require(isinstance(item, dict) and set(item) == {"path", "sha256"}, f"Llama {backend} {key} input invalid")
        copied = resolve_member(root, recorded, item["path"], f"Llama {backend} {key} input")
        require(
            item["sha256"] == sha256(copied) == sha256(current),
            f"Llama {backend} {key} input differs from current source",
        )
    binary_sha = require_sha(receipt.get("binary_sha256"), f"Llama {backend} binary SHA")
    summary_receipt = summary.get("execution_receipt")
    require(
        isinstance(summary_receipt, dict)
        and summary_receipt.get("artifact_sha256") == sha256(receipt_path)
        and summary_receipt.get("canonical_sha256") == receipt["canonical_sha256"]
        and summary_receipt.get("manifest_sha256") == sha256(LLAMA_MANIFESTS[backend])
        and summary_receipt.get("runner_sha256")
        == sha256(REPO_ROOT / "scripts/release/run_scenarios.py")
        and summary_receipt.get("binary_sha256") == binary_sha,
        f"Llama {backend} summary/receipt binding mismatch",
    )
    server_argv = receipt.get("server_argv")
    require(isinstance(server_argv, list), f"Llama {backend} server argv is missing")
    require(
        server_argv.count("--backend") == 1
        and command_flag(server_argv, "--backend", f"Llama {backend} server argv")
        == backend
        and server_argv[-1] == summary.get("model"),
        f"Llama {backend} serve did not use the explicit product backend/model",
    )
    run_command = read_json(
        root / "run_multiturn_recall/command.json", f"Llama {backend} run command"
    )
    require(
        run_command.get("binary_sha256") == binary_sha
        and run_command.get("env_policy") == "remove_FERRUM_prefix"
        and isinstance(run_command.get("argv"), list)
        and run_command["argv"][1:4] == ["run", "--backend", backend]
        and run_command["argv"][-1] == summary.get("model"),
        f"Llama {backend} run command/binary binding mismatch",
    )
    if backend == "cuda":
        hardware = receipt.get("hardware")
        require(
            isinstance(hardware, dict)
            and hardware.get("returncode") == 0
            and len([line for line in str(hardware.get("stdout", "")).splitlines() if line.strip()]) == 1
            and "RTX 4090" in str(hardware.get("stdout")),
            "Llama CUDA evidence must use exactly one RTX 4090",
        )
    tree_ref = validate_artifact_tree(root, recorded)
    return {
        "backend": backend,
        "model": summary["model"],
        "summary": file_ref(summary_path),
        "execution_receipt": file_ref(receipt_path),
        "artifact_tree": tree_ref,
        "binary_sha256": binary_sha,
        "scenario_count": 3,
        "entrypoints": ["run", "serve"],
        "stream_done_count": 1,
        "stream_usage_count": 1,
    }


def acceptance(
    matrices: dict[str, Any], llamas: dict[str, Any]
) -> dict[str, Any]:
    require(set(matrices) == set(MATRIX_LANES), "R1 matrix lane set is incomplete")
    require(set(llamas) == {"cuda", "metal"}, "R1 Llama backend set is incomplete")
    by_backend: dict[str, list[dict[str, Any]]] = {"cuda": [], "metal": []}
    model_rows: dict[str, Any] = {}
    provider_count = 0
    cancel_count = 0
    for key, lane in MATRIX_LANES.items():
        row = matrices[key]
        require(
            row.get("model_key") == lane.model_key
            and row.get("backend") == lane.backend
            and row.get("summary", {}).get("case_count") == lane.spec.expected_case_count
            and row.get("summary", {}).get("passed_case_count") == lane.spec.expected_case_count,
            f"R1 {key} denominator differs",
        )
        identity = row.get("product_execution_identity")
        provider = row.get("provider_execution")
        resources = row.get("resource_contract")
        require(
            isinstance(identity, dict)
            and identity.get("production_legacy_selection_count") == 0
            and identity.get("typed_vnext_startup_marker_count") == 7,
            f"R1 {key} vNext/legacy selection failed",
        )
        require(
            isinstance(provider, dict)
            and provider.get("provider_count", 0) >= 4
            and provider.get("conformance") == "selected-submitted-retired",
            f"R1 {key} provider conformance failed",
        )
        require(
            isinstance(resources, dict)
            and resources.get("resource_final_state") == "released"
            and resources.get("admitted_released_count")
            == resources.get("cancel_case_count")
            and resources.get("post_capacity_success_count")
            == resources.get("cancel_case_count")
            and resources.get("leak_count") == 0
            and resources.get("double_release_count") == 0,
            f"R1 {key} resource/capacity contract failed",
        )
        by_backend[lane.backend].append(row)
        provider_count += provider["provider_count"]
        cancel_count += resources["cancel_case_count"]
        model_rows[key] = {
            "model_key": lane.model_key,
            "backend": lane.backend,
            "cases": f"{lane.spec.expected_case_count}/{lane.spec.expected_case_count}",
            "provider_count": provider["provider_count"],
            "cancel_capacity_cases": resources["cancel_case_count"],
        }
    backend_binaries: dict[str, str] = {}
    backend_hardware: dict[str, str] = {}
    for backend, rows in by_backend.items():
        binaries = {row["binary_sha256"] for row in rows}
        hardware = {row["hardware_id"] for row in rows}
        require(len(binaries) == 1, f"R1 {backend} matrices used different binaries")
        require(len(hardware) == 1, f"R1 {backend} matrices used different hardware")
        binary = next(iter(binaries))
        require(llamas[backend].get("binary_sha256") == binary, f"R1 Llama {backend} used a different binary")
        require(
            llamas[backend].get("scenario_count") == 3
            and llamas[backend].get("entrypoints") == ["run", "serve"]
            and llamas[backend].get("stream_done_count") == 1
            and llamas[backend].get("stream_usage_count") == 1,
            f"R1 Llama {backend} product contract failed",
        )
        backend_binaries[backend] = binary
        backend_hardware[backend] = next(iter(hardware))
    return {
        "matrix_lanes": "6/6",
        "models": model_rows,
        "total_matrix_case_count": sum(
            lane.spec.expected_case_count for lane in MATRIX_LANES.values()
        ),
        "matrix_failure_count": 0,
        "product_entrypoints": ["run", "serve"],
        "provider_conformance_lane_count": 6,
        "provider_count_sum": provider_count,
        "resource_cancel_capacity_case_count": cancel_count,
        "resource_leak_count": 0,
        "production_legacy_selection_count": 0,
        "llama_dense_backends": "2/2",
        "llama_dense_scenarios": "6/6",
        "backend_binary_sha256": backend_binaries,
        "backend_hardware_id": backend_hardware,
        "waiver_count": 0,
        "exception_count": 0,
    }


def validate_inputs(paths: dict[str, Path], source: dict[str, Any]) -> dict[str, Any]:
    require(
        set(paths)
        == {"r0", *MATRIX_LANES, "llama_cuda", "llama_metal"},
        "R1 input path set differs",
    )
    r0 = validate_r0(paths["r0"], source)
    matrices = {
        key: validate_matrix(paths[key], lane, source)
        for key, lane in MATRIX_LANES.items()
    }
    llamas = {
        "cuda": validate_llama(paths["llama_cuda"], "cuda", source),
        "metal": validate_llama(paths["llama_metal"], "metal", source),
    }
    return {
        "r0": r0,
        "matrices": matrices,
        "llama_dense": llamas,
        "acceptance": acceptance(matrices, llamas),
    }


def build(paths: dict[str, Path], out: Path) -> str:
    output = out.expanduser().resolve()
    require(
        REPO_ROOT not in output.parents and output != REPO_ROOT,
        "R1 output must be outside the source tree",
    )
    require(
        not output.exists() or not any(output.iterdir()),
        f"R1 output must be absent or empty: {output}",
    )
    source = current_source()
    dependencies = validate_inputs(paths, source)
    require(current_source() == source, "R1 source changed during validation")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        pass_line = f"{PASS_PREFIX}: {output}"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_r1_product_correctness_manifest",
            "checkpoint_id": "R1",
            "lane": "runtime-vnext-r1",
            "status": "pass",
            "canonical": True,
            "artifact_dir": str(output),
            "source": source,
            "dependencies": dependencies,
            "acceptance": dependencies["acceptance"],
            "unlocks": ["R2"],
            "does_not_prove": DOES_NOT_PROVE,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "pass_line": pass_line,
        }
        write_json(staging / "manifest.json", manifest, exclusive=True)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
        verify_manifest(output / "manifest.json", verify_checkout=True)
        return pass_line
    except BaseException:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        if output.exists() and output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        raise


def verify_manifest(
    manifest_path: Path,
    *,
    verify_checkout: bool = True,
    expected_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    root = path.parent
    manifest = read_json(path, "R1 manifest")
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
        "acceptance",
        "unlocks",
        "does_not_prove",
        "created_at",
        "pass_line",
    }
    require(set(manifest) == required, "R1 manifest field set mismatch")
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type")
        == "runtime_vnext_r1_product_correctness_manifest"
        and manifest.get("checkpoint_id") == "R1"
        and manifest.get("lane") == "runtime-vnext-r1"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True
        and Path(str(manifest.get("artifact_dir", ""))).resolve() == root
        and manifest.get("unlocks") == ["R2"]
        and manifest.get("does_not_prove") == DOES_NOT_PROVE
        and manifest.get("pass_line") == f"{PASS_PREFIX}: {root}",
        "R1 manifest identity/status/PASS mismatch",
    )
    source = normalize_source(manifest.get("source"), "R1")
    expected = current_source() if verify_checkout else expected_source
    if expected is not None:
        require(source == expected, "R1 aggregate source is stale")
    dependencies = manifest.get("dependencies")
    require(
        isinstance(dependencies, dict)
        and set(dependencies) == {"r0", "matrices", "llama_dense", "acceptance"},
        "R1 dependency set mismatch",
    )
    matrices = dependencies["matrices"]
    llamas = dependencies["llama_dense"]
    require(isinstance(matrices, dict) and isinstance(llamas, dict), "R1 evidence maps are missing")
    for key, row in matrices.items():
        require(isinstance(row, dict), f"R1 {key} evidence is invalid")
        for ref_name in ("outer_manifest", "child_manifest", "validation", "scenario_report"):
            validate_ref(row[ref_name], f"R1 {key} {ref_name}")
        provider = row.get("provider_execution")
        resources = row.get("resource_contract")
        require(isinstance(provider, dict) and isinstance(resources, dict), f"R1 {key} detailed evidence is missing")
        for ref_name in ("c18_raw", "c18_case", "c18_transcript"):
            validate_ref(provider[ref_name], f"R1 {key} {ref_name}")
        validate_ref(resources["c09_raw"], f"R1 {key} c09_raw")
    for backend, row in llamas.items():
        require(isinstance(row, dict), f"R1 Llama {backend} evidence is invalid")
        for ref_name in ("summary", "execution_receipt", "artifact_tree"):
            validate_ref(row[ref_name], f"R1 Llama {backend} {ref_name}")
    r0 = dependencies["r0"]
    require(isinstance(r0, dict), "R1 R0 dependency is missing")
    validate_ref(r0["outer_manifest"], "R1 R0 outer manifest")
    validate_ref(r0["child_manifest"], "R1 R0 child manifest")
    require(
        r0["source_closure"] == source_closure(r0["source"], source),
        "R1 R0 source closure drifted",
    )
    accepted = acceptance(matrices, llamas)
    require(
        dependencies.get("acceptance") == accepted
        and manifest.get("acceptance") == accepted,
        "R1 acceptance summary drifted",
    )
    return {
        "kind": "vnext-r1",
        "child_manifest": {"path": str(path), "sha256": sha256(path)},
        "source": source,
        "acceptance": copy.deepcopy(accepted),
    }


def fixture_matrix(key: str, binary: str, hardware: str, root: Path) -> dict[str, Any]:
    lane = MATRIX_LANES[key]
    refs = {}
    for name in (
        "outer_manifest",
        "child_manifest",
        "validation",
        "scenario_report",
        "c18_raw",
        "c18_case",
        "c18_transcript",
        "c09_raw",
    ):
        path = root / f"{key}.{name}.json"
        write_json(path, {"key": key, "name": name})
        refs[name] = file_ref(path)
    return {
        "lane": lane.lane,
        "model_key": lane.model_key,
        "backend": lane.backend,
        "outer_manifest": refs["outer_manifest"],
        "child_manifest": refs["child_manifest"],
        "validation": refs["validation"],
        "scenario_report": refs["scenario_report"],
        "binary_sha256": binary,
        "hardware_id": hardware,
        "summary": {
            "case_count": lane.spec.expected_case_count,
            "passed_case_count": lane.spec.expected_case_count,
        },
        "product_execution_identity": {
            "typed_vnext_startup_marker_count": 7,
            "production_legacy_selection_count": 0,
        },
        "provider_execution": {
            "provider_count": 4,
            "conformance": "selected-submitted-retired",
            "c18_raw": refs["c18_raw"],
            "c18_case": refs["c18_case"],
            "c18_transcript": refs["c18_transcript"],
        },
        "resource_contract": {
            "cancel_case_count": 3,
            "admitted_released_count": 3,
            "post_capacity_success_count": 3,
            "resource_final_state": "released",
            "leak_count": 0,
            "double_release_count": 0,
            "c09_raw": refs["c09_raw"],
        },
    }


def expect_reject(action: Any, marker: str) -> None:
    try:
        action()
    except R1Error:
        return
    raise R1Error(f"self-test mutation was accepted: {marker}")


def self_test() -> int:
    source = {
        "git_sha": git_text("rev-parse", "HEAD"),
        "git_tree_sha": git_text("rev-parse", "HEAD^{tree}"),
        "dirty": False,
    }
    exact_closure = matrix_source_closure(
        source,
        source,
        MATRIX_LANES["m1_cuda"],
    )
    require(
        exact_closure["policy"] == "exact-source"
        and exact_closure["changed_file_count"] == 0,
        "matrix exact-source closure differs",
    )
    with tempfile.TemporaryDirectory(prefix="ferrum-r1-selftest-") as temporary:
        root = Path(temporary).resolve()
        binaries = {"cuda": "1" * 64, "metal": "2" * 64}
        hardware = {"cuda": "rtx4090-fixture", "metal": "m1-max-fixture"}
        matrices = {
            key: fixture_matrix(key, binaries[lane.backend], hardware[lane.backend], root)
            for key, lane in MATRIX_LANES.items()
        }
        llamas = {
            backend: {
                "backend": backend,
                "model": LLAMA_MODEL_MARKERS[backend],
                "binary_sha256": binaries[backend],
                "scenario_count": 3,
                "entrypoints": ["run", "serve"],
                "stream_done_count": 1,
                "stream_usage_count": 1,
            }
            for backend in ("cuda", "metal")
        }
        accepted = acceptance(matrices, llamas)
        require(
            accepted["matrix_lanes"] == "6/6"
            and accepted["llama_dense_scenarios"] == "6/6"
            and accepted["production_legacy_selection_count"] == 0,
            "R1 fixture acceptance differs",
        )
        bad = copy.deepcopy(matrices)
        bad.pop("m3_metal")
        expect_reject(lambda: acceptance(bad, llamas), "missing matrix lane")
        bad = copy.deepcopy(matrices)
        bad["m2_cuda"]["binary_sha256"] = "3" * 64
        expect_reject(lambda: acceptance(bad, llamas), "backend binary mismatch")
        bad = copy.deepcopy(matrices)
        bad["m1_cuda"]["product_execution_identity"]["production_legacy_selection_count"] = 1
        expect_reject(lambda: acceptance(bad, llamas), "legacy selection")
        bad = copy.deepcopy(matrices)
        bad["m3_metal"]["resource_contract"]["leak_count"] = 1
        expect_reject(lambda: acceptance(bad, llamas), "resource leak")
        bad_llama = copy.deepcopy(llamas)
        bad_llama["cuda"]["stream_done_count"] = 0
        expect_reject(lambda: acceptance(matrices, bad_llama), "Llama stream")
    print(SELFTEST_PASS_LINE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0", type=Path)
    for key in MATRIX_LANES:
        parser.add_argument(f"--{key.replace('_', '-')}", type=Path)
    parser.add_argument("--llama-cuda", type=Path)
    parser.add_argument("--llama-metal", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test()
        values = {
            "r0": args.r0,
            **{key: getattr(args, key) for key in MATRIX_LANES},
            "llama_cuda": args.llama_cuda,
            "llama_metal": args.llama_metal,
        }
        missing = [key for key, value in values.items() if value is None]
        require(not missing and args.out is not None, "missing required inputs: " + ", ".join(missing))
        paths = {key: value for key, value in values.items() if value is not None}
        assert args.out is not None
        print(build(paths, args.out))
        return 0
    except (OSError, R1Error, RuntimeError, ValueError) as error:
        print(f"FERRUM RUNTIME VNEXT R1 PRODUCT CORRECTNESS FAIL: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
