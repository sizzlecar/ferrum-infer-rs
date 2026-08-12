#!/usr/bin/env python3
"""Aggregate the release-blocking R0 source, numerics, and CUDA S2 evidence."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import runtime_vnext_g08a_checkpoint as g08a
import runtime_vnext_s2_cuda_product_contract as s2_contract


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PASS_PREFIX = "FERRUM RUNTIME VNEXT R0 CORE CLOSURE PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT R0 CORE CLOSURE SELFTEST PASS"
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
DEPENDENCY_KEYS = ("source", "numerics", "s2")
G02_ROSTER_BRIDGE_ID = "g02-roster-only-05a5d2f8-v1"
G02_ROSTER_BRIDGE_BASE_GIT_SHA = "05a5d2f8611ed3a3fedb5c69ff3ba11e533bc4c7"
G02_ROSTER_BRIDGE_COMMIT_GIT_SHA = "a609cac8099e0190004a7f6523166f281c6b9ad2"
G02_ROSTER_BRIDGE_COMMIT_TREE_SHA = "d2fa2ffd22d322cf4a7188562f121a3a8babc0c7"
G02_ROSTER_BRIDGE_PATH = "scripts/release/runtime_vnext_g02_core.py"
G02_ROSTER_BRIDGE_OLD_BLOB = "38b832c95ecee833240a1477678fb5ce350f52fb"
# Exact post-fix Git blob sealed into the one permitted bridge commit.
G02_ROSTER_BRIDGE_NEW_BLOB = "fa369a3ee52535ead59aefb4b3f675844feb09b8"
G02_ROSTER_BRIDGE_CHANGED_FILES = frozenset(
    {
        "docs/goals/runtime-vnext-0.8.0-2026-07-10/CORRECTNESS_ACCEPTANCE_AMENDMENT_2026-08-07.md",
        G02_ROSTER_BRIDGE_PATH,
        "scripts/release/runtime_vnext_s2_cuda_product_contract.py",
        "scripts/release/runtime_vnext_r0_core_closure.py",
        "scripts/release/runtime_vnext_r1_product_correctness.py",
    }
)
ARTIFACT_EVIDENCE_BRIDGE_ID = "artifact-evidence-only-a609cac8-v1"
ARTIFACT_EVIDENCE_BRIDGE_BASE_GIT_SHA = G02_ROSTER_BRIDGE_COMMIT_GIT_SHA
ARTIFACT_EVIDENCE_BRIDGE_BASE_TREE_SHA = G02_ROSTER_BRIDGE_COMMIT_TREE_SHA
ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_PATH = (
    "docs/goals/runtime-vnext-0.8.0-2026-07-10/"
    "HOST_SUSPEND_EVIDENCE_AMENDMENT_2026-08-12.md"
)
ARTIFACT_EVIDENCE_BRIDGE_BASELINE_PATH = (
    "scripts/release/runtime_vnext_baseline_scenarios.py"
)
ARTIFACT_EVIDENCE_BRIDGE_R0_PATH = "scripts/release/runtime_vnext_r0_core_closure.py"
ARTIFACT_EVIDENCE_BRIDGE_R1_PATH = (
    "scripts/release/runtime_vnext_r1_product_correctness.py"
)
ARTIFACT_EVIDENCE_BRIDGE_CHANGED_FILES = (
    ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_PATH,
    ARTIFACT_EVIDENCE_BRIDGE_BASELINE_PATH,
    ARTIFACT_EVIDENCE_BRIDGE_R0_PATH,
    ARTIFACT_EVIDENCE_BRIDGE_R1_PATH,
)
ARTIFACT_EVIDENCE_BRIDGE_OLD_BLOBS = {
    ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_PATH: "0" * 40,
    ARTIFACT_EVIDENCE_BRIDGE_BASELINE_PATH: "e667cef1b2bad37d439be472abd09d2203bd42c1",
    ARTIFACT_EVIDENCE_BRIDGE_R0_PATH: "d86d3cb9719b5d669802bbf22b81cafc9d060360",
    ARTIFACT_EVIDENCE_BRIDGE_R1_PATH: "e23b242414afee16b0435099900bf78a4e832d12",
}
# These two blobs depend on sibling integration. They intentionally fail closed until
# the final amendment and baseline assembler bytes stop moving and are sealed here.
ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_NEW_BLOB = "bdf2082acae76dc2475329baefd434abce5ec35b"
ARTIFACT_EVIDENCE_BRIDGE_BASELINE_NEW_BLOB = "8907606f0fdbf58720e84fcf8fdc2d18d4d8da76"
CONTROL_PLANE_PREFIXES = ("docs/",)
SAME_HISTORY_COLLECTOR = "scripts/release/runtime_vnext_g08a_same_history_collector.py"
R1_AGGREGATOR = "scripts/release/runtime_vnext_r1_product_correctness.py"
S0A_CONTRACT_SPLIT = "scripts/release/runtime_vnext_s0a_contract_split.py"
CONTROL_PLANE_FILES = frozenset(
    {
        "scripts/release/runtime_vnext_r0_core_closure.py",
        R1_AGGREGATOR,
        "scripts/release/run_gate.py",
        "scripts/release/change_impact_rules.json",
        "scripts/release/fixtures/change_impact/planner_fixtures.json",
    }
)
S2_MULTITURN_SCENARIO = (
    "scripts/release/scenarios/runtime_vnext_s2_multiturn_concurrency_cuda.json"
)
DEPENDENCY_LOCAL_FILES = {
    "source": frozenset(
        {
            S0A_CONTRACT_SPLIT,
            "scripts/release/runtime_vnext_s2_cuda_product_contract.py",
            "scripts/release/runtime_vnext_s2_multiturn_concurrency_checkpoint.py",
            S2_MULTITURN_SCENARIO,
        }
    ),
    "numerics": frozenset(
        {
            S0A_CONTRACT_SPLIT,
            "scripts/release/runtime_vnext_s2_cuda_product_contract.py",
            "scripts/release/runtime_vnext_s2_multiturn_concurrency_checkpoint.py",
            S2_MULTITURN_SCENARIO,
        }
    ),
    "s2": frozenset({SAME_HISTORY_COLLECTOR}),
}
DOES_NOT_PROVE = [
    "R1 three-model CUDA and Metal correctness",
    "R2 performance, profile, or build acceptance",
    "R3 release assets or installed regression",
    "v0.8.0 release readiness",
]


class R0Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R0Error(message)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise R0Error(f"invalid {label} JSON {path}: {error}") from error
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


def file_ref(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(resolved.is_file() and not resolved.is_symlink(), f"artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_ref(value: Any, label: str) -> Path:
    require(
        isinstance(value, dict) and set(value) == {"path", "sha256", "size_bytes"},
        f"{label} reference fields differ",
    )
    path = Path(str(value["path"])).expanduser().resolve()
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    require(path.stat().st_size == value["size_bytes"], f"{label} size mismatch")
    require(sha256(path) == value["sha256"], f"{label} SHA256 mismatch")
    return path


def git_text(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(process.returncode == 0, f"git {' '.join(args)} failed: {process.stderr.strip()}")
    return process.stdout.strip()


def current_source() -> dict[str, Any]:
    status = [line for line in git_text("status", "--short").splitlines() if line]
    require(not status, f"R0 source must be clean: {status[:8]}")
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
        GIT_SHA_RE.fullmatch(str(source["git_sha"])) is not None
        and GIT_SHA_RE.fullmatch(str(source["git_tree_sha"])) is not None
        and source["dirty"] is False,
        f"{label} source identity is invalid",
    )
    return source


def source_from_outer(path: Path, key: str) -> dict[str, Any]:
    outer = read_json(path.expanduser().resolve(), f"R0 {key} outer manifest")
    child_artifacts = outer.get("child_artifacts")
    require(isinstance(child_artifacts, dict), f"R0 {key} child provenance is missing")
    return normalize_source(child_artifacts.get("source"), f"R0 {key}")


def control_plane_only(paths: list[str], key: str) -> tuple[list[str], list[str]]:
    require(key in DEPENDENCY_LOCAL_FILES, f"unknown R0 dependency closure: {key}")
    allowed_files = CONTROL_PLANE_FILES | DEPENDENCY_LOCAL_FILES[key]
    allowed = []
    rejected = []
    for path in paths:
        if path in allowed_files or path.startswith(CONTROL_PLANE_PREFIXES):
            allowed.append(path)
        else:
            rejected.append(path)
    return allowed, rejected


def _validate_g02_roster_bridge_facts(
    *,
    recorded_sha: str,
    current_sha: str,
    parent_shas: list[str],
    changed_files: list[str],
    old_blob: str,
    new_blob: str,
    expected_new_blob: str,
) -> dict[str, Any]:
    """Validate the immutable facts for the one-time G02 roster-only bridge."""
    require(
        GIT_SHA_RE.fullmatch(expected_new_blob) is not None
        and expected_new_blob != "0" * 40,
        "G02 roster bridge new blob is not sealed",
    )
    require(
        GIT_SHA_RE.fullmatch(recorded_sha) is not None
        and GIT_SHA_RE.fullmatch(current_sha) is not None
        and all(GIT_SHA_RE.fullmatch(parent) is not None for parent in parent_shas)
        and GIT_SHA_RE.fullmatch(old_blob) is not None
        and GIT_SHA_RE.fullmatch(new_blob) is not None,
        "G02 roster bridge Git identity is invalid",
    )
    require(
        recorded_sha == G02_ROSTER_BRIDGE_BASE_GIT_SHA,
        "G02 roster bridge evidence is not the sealed 05a source",
    )
    require(
        current_sha == G02_ROSTER_BRIDGE_COMMIT_GIT_SHA,
        "G02 roster bridge is not the sealed a609 checkpoint",
    )
    require(
        parent_shas == [G02_ROSTER_BRIDGE_BASE_GIT_SHA],
        "G02 roster bridge current source is not the unique direct child of 05a",
    )
    require(
        len(changed_files) == len(G02_ROSTER_BRIDGE_CHANGED_FILES)
        and changed_files == sorted(G02_ROSTER_BRIDGE_CHANGED_FILES),
        "G02 roster bridge changed-file order or set differs",
    )
    require(
        old_blob == G02_ROSTER_BRIDGE_OLD_BLOB,
        "G02 roster bridge old validator blob differs",
    )
    require(new_blob == expected_new_blob, "G02 roster bridge new validator blob differs")
    return {
        "bridge_id": G02_ROSTER_BRIDGE_ID,
        "base_git_sha": G02_ROSTER_BRIDGE_BASE_GIT_SHA,
        "commit_git_sha": current_sha,
        "parent_git_shas": copy.deepcopy(parent_shas),
        "changed_files": sorted(changed_files),
        "g02_validator": {
            "path": G02_ROSTER_BRIDGE_PATH,
            "old_blob": old_blob,
            "new_blob": new_blob,
        },
    }


def _validate_artifact_evidence_bridge_facts(
    *,
    base_sha: str,
    base_tree_sha: str,
    current_sha: str,
    current_tree_sha: str,
    parent_shas: list[str],
    changes: list[dict[str, str]],
    current_blobs: dict[str, str],
    expected_final_blobs: dict[str, str],
) -> dict[str, Any]:
    """Validate the immutable facts for the one-time a609 evidence-only hop."""
    require(
        set(expected_final_blobs)
        == {
            ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_PATH,
            ARTIFACT_EVIDENCE_BRIDGE_BASELINE_PATH,
        }
        and all(
            GIT_SHA_RE.fullmatch(blob) is not None and blob != "0" * 40
            for blob in expected_final_blobs.values()
        ),
        "artifact-evidence bridge final blobs are not sealed",
    )
    require(
        tuple(current_blobs) == ARTIFACT_EVIDENCE_BRIDGE_CHANGED_FILES
        and all(GIT_SHA_RE.fullmatch(blob) is not None for blob in current_blobs.values()),
        "artifact-evidence bridge current blob closure differs",
    )
    require(
        GIT_SHA_RE.fullmatch(base_sha) is not None
        and GIT_SHA_RE.fullmatch(base_tree_sha) is not None
        and GIT_SHA_RE.fullmatch(current_sha) is not None
        and GIT_SHA_RE.fullmatch(current_tree_sha) is not None
        and all(GIT_SHA_RE.fullmatch(parent) is not None for parent in parent_shas),
        "artifact-evidence bridge Git identity is invalid",
    )
    require(
        base_sha == ARTIFACT_EVIDENCE_BRIDGE_BASE_GIT_SHA,
        "artifact-evidence bridge does not start at the sealed a609 checkpoint",
    )
    require(
        base_tree_sha == ARTIFACT_EVIDENCE_BRIDGE_BASE_TREE_SHA,
        "artifact-evidence bridge a609 tree is not sealed",
    )
    require(current_sha != base_sha, "artifact-evidence bridge requires one new commit")
    require(
        parent_shas == [ARTIFACT_EVIDENCE_BRIDGE_BASE_GIT_SHA],
        "artifact-evidence bridge current source is not the unique direct child of a609",
    )
    require(
        isinstance(changes, list)
        and len(changes) == len(ARTIFACT_EVIDENCE_BRIDGE_CHANGED_FILES),
        "artifact-evidence bridge changed-file count differs",
    )
    expected_modes = {
        ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_PATH: ("000000", "100644", "A"),
        ARTIFACT_EVIDENCE_BRIDGE_BASELINE_PATH: ("100644", "100644", "M"),
        ARTIFACT_EVIDENCE_BRIDGE_R0_PATH: ("100644", "100644", "M"),
        ARTIFACT_EVIDENCE_BRIDGE_R1_PATH: ("100755", "100755", "M"),
    }
    normalized_changes: list[dict[str, str]] = []
    for index, path in enumerate(ARTIFACT_EVIDENCE_BRIDGE_CHANGED_FILES):
        change = changes[index]
        require(
            isinstance(change, dict)
            and set(change)
            == {"path", "status", "old_mode", "new_mode", "old_blob", "new_blob"},
            f"artifact-evidence bridge change row {index} fields differ",
        )
        require(
            change.get("path") == path,
            "artifact-evidence bridge changed-file order or set differs",
        )
        old_mode, new_mode, status = expected_modes[path]
        require(
            (change.get("old_mode"), change.get("new_mode"), change.get("status"))
            == (old_mode, new_mode, status),
            f"artifact-evidence bridge mode/status differs for {path}",
        )
        old_blob = str(change.get("old_blob"))
        new_blob = str(change.get("new_blob"))
        require(
            old_blob == ARTIFACT_EVIDENCE_BRIDGE_OLD_BLOBS[path],
            f"artifact-evidence bridge a609 blob differs for {path}",
        )
        require(
            GIT_SHA_RE.fullmatch(new_blob) is not None
            and new_blob != "0" * 40
            and new_blob != old_blob,
            f"artifact-evidence bridge final blob is invalid for {path}",
        )
        require(
            new_blob == current_blobs[path],
            f"artifact-evidence bridge final tree blob differs for {path}",
        )
        if path in expected_final_blobs:
            require(
                new_blob == expected_final_blobs[path],
                f"artifact-evidence bridge sealed final blob differs for {path}",
            )
        normalized_changes.append(copy.deepcopy(change))
    return {
        "bridge_id": ARTIFACT_EVIDENCE_BRIDGE_ID,
        "base_git_sha": ARTIFACT_EVIDENCE_BRIDGE_BASE_GIT_SHA,
        "base_git_tree_sha": ARTIFACT_EVIDENCE_BRIDGE_BASE_TREE_SHA,
        "commit_git_sha": current_sha,
        "commit_git_tree_sha": current_tree_sha,
        "parent_git_shas": copy.deepcopy(parent_shas),
        "changed_files": list(ARTIFACT_EVIDENCE_BRIDGE_CHANGED_FILES),
        "changes": normalized_changes,
        "final_blobs": copy.deepcopy(current_blobs),
        "sealed_final_blobs": copy.deepcopy(expected_final_blobs),
    }


def _parse_raw_git_changes(raw: str) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    for index, line in enumerate(raw.splitlines()):
        try:
            metadata, path = line.split("\t", 1)
        except ValueError as error:
            raise R0Error(
                f"artifact-evidence bridge raw diff row {index} is malformed"
            ) from error
        fields = metadata.split()
        require(
            len(fields) == 5 and fields[0].startswith(":"),
            f"artifact-evidence bridge raw diff row {index} metadata differs",
        )
        changes.append(
            {
                "path": path,
                "status": fields[4],
                "old_mode": fields[0][1:],
                "new_mode": fields[1],
                "old_blob": fields[2],
                "new_blob": fields[3],
            }
        )
    return changes


def g02_roster_bridge(recorded: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Resolve and validate the sealed bridge facts from Git, without heuristics."""
    recorded_sha = str(recorded["git_sha"])
    current_sha = str(current["git_sha"])
    require(
        git_text("rev-parse", f"{current_sha}^{{tree}}") == current["git_tree_sha"],
        "G02 roster bridge current source tree differs from git",
    )
    commit_and_parents = git_text("rev-list", "--parents", "-n", "1", current_sha).split()
    require(
        commit_and_parents and commit_and_parents[0] == current_sha,
        "G02 roster bridge commit topology is unavailable",
    )
    changed_files = [
        line
        for line in git_text(
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            f"{recorded_sha}..{current_sha}",
        ).splitlines()
        if line
    ]
    return _validate_g02_roster_bridge_facts(
        recorded_sha=recorded_sha,
        current_sha=current_sha,
        parent_shas=commit_and_parents[1:],
        changed_files=changed_files,
        old_blob=git_text("rev-parse", f"{recorded_sha}:{G02_ROSTER_BRIDGE_PATH}"),
        new_blob=git_text("rev-parse", f"{current_sha}:{G02_ROSTER_BRIDGE_PATH}"),
        expected_new_blob=G02_ROSTER_BRIDGE_NEW_BLOB,
    )


def artifact_evidence_bridge(
    recorded: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the exact a609 -> direct-child evidence-only hop from Git."""
    recorded_sha = str(recorded["git_sha"])
    current_sha = str(current["git_sha"])
    require(
        recorded
        == {
            "git_sha": ARTIFACT_EVIDENCE_BRIDGE_BASE_GIT_SHA,
            "git_tree_sha": ARTIFACT_EVIDENCE_BRIDGE_BASE_TREE_SHA,
            "dirty": False,
        },
        "artifact-evidence bridge recorded source is not the sealed a609 tree",
    )
    require(
        git_text("rev-parse", f"{recorded_sha}^{{tree}}")
        == ARTIFACT_EVIDENCE_BRIDGE_BASE_TREE_SHA,
        "artifact-evidence bridge a609 tree differs from git",
    )
    require(
        git_text("rev-parse", f"{current_sha}^{{tree}}") == current["git_tree_sha"],
        "artifact-evidence bridge current source tree differs from git",
    )
    commit_and_parents = git_text("rev-list", "--parents", "-n", "1", current_sha).split()
    require(
        commit_and_parents and commit_and_parents[0] == current_sha,
        "artifact-evidence bridge commit topology is unavailable",
    )
    changes = _parse_raw_git_changes(
        git_text(
            "diff",
            "--raw",
            "--no-renames",
            "--abbrev=40",
            f"{recorded_sha}..{current_sha}",
        )
    )
    return _validate_artifact_evidence_bridge_facts(
        base_sha=recorded_sha,
        base_tree_sha=str(recorded["git_tree_sha"]),
        current_sha=current_sha,
        current_tree_sha=str(current["git_tree_sha"]),
        parent_shas=commit_and_parents[1:],
        changes=changes,
        current_blobs={
            path: git_text("rev-parse", f"{current_sha}:{path}")
            for path in ARTIFACT_EVIDENCE_BRIDGE_CHANGED_FILES
        },
        expected_final_blobs={
            ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_PATH: (
                ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_NEW_BLOB
            ),
            ARTIFACT_EVIDENCE_BRIDGE_BASELINE_PATH: (
                ARTIFACT_EVIDENCE_BRIDGE_BASELINE_NEW_BLOB
            ),
        },
    )


def _single_artifact_evidence_closure(
    recorded_sha: str, current_sha: str, bridge: dict[str, Any]
) -> dict[str, Any]:
    require(
        recorded_sha == ARTIFACT_EVIDENCE_BRIDGE_BASE_GIT_SHA
        and bridge.get("bridge_id") == ARTIFACT_EVIDENCE_BRIDGE_ID
        and bridge.get("base_git_sha") == recorded_sha
        and bridge.get("commit_git_sha") == current_sha,
        "artifact-evidence bridge hop identity/order differs",
    )
    changed_files = bridge.get("changed_files")
    require(
        changed_files == list(ARTIFACT_EVIDENCE_BRIDGE_CHANGED_FILES),
        "artifact-evidence bridge hop file order differs",
    )
    return {
        "from_git_sha": recorded_sha,
        "to_git_sha": current_sha,
        "changed_files_by_hop": [copy.deepcopy(changed_files)],
        "changed_file_count": len(changed_files),
        "hop_count": 1,
        "policy": "artifact-evidence-only-a609-direct-child-bridge",
        "bridge_hops": [copy.deepcopy(bridge)],
    }


def _ordered_g02_artifact_evidence_closure(
    recorded_sha: str,
    current_sha: str,
    g02_bridge: dict[str, Any],
    artifact_bridge: dict[str, Any],
) -> dict[str, Any]:
    require(
        recorded_sha == G02_ROSTER_BRIDGE_BASE_GIT_SHA
        and g02_bridge.get("bridge_id") == G02_ROSTER_BRIDGE_ID
        and g02_bridge.get("base_git_sha") == recorded_sha
        and g02_bridge.get("commit_git_sha") == G02_ROSTER_BRIDGE_COMMIT_GIT_SHA,
        "ordered evidence bridge first hop is not sealed 05a -> a609 G02",
    )
    require(
        artifact_bridge.get("bridge_id") == ARTIFACT_EVIDENCE_BRIDGE_ID
        and artifact_bridge.get("base_git_sha") == G02_ROSTER_BRIDGE_COMMIT_GIT_SHA
        and artifact_bridge.get("commit_git_sha") == current_sha,
        "ordered evidence bridge second hop is not sealed a609 -> final",
    )
    first_files = g02_bridge.get("changed_files")
    second_files = artifact_bridge.get("changed_files")
    require(
        first_files == sorted(G02_ROSTER_BRIDGE_CHANGED_FILES)
        and second_files == list(ARTIFACT_EVIDENCE_BRIDGE_CHANGED_FILES),
        "ordered evidence bridge hop file order differs",
    )
    return {
        "from_git_sha": recorded_sha,
        "to_git_sha": current_sha,
        "changed_files_by_hop": [
            copy.deepcopy(first_files),
            copy.deepcopy(second_files),
        ],
        "changed_file_count": len(first_files) + len(second_files),
        "hop_count": 2,
        "policy": "ordered-g02-roster-then-artifact-evidence-bridge",
        "bridge_hops": [copy.deepcopy(g02_bridge), copy.deepcopy(artifact_bridge)],
    }


def artifact_evidence_source_closure(
    recorded: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    bridge = artifact_evidence_bridge(recorded, current)
    return _single_artifact_evidence_closure(
        str(recorded["git_sha"]), str(current["git_sha"]), bridge
    )


def g02_then_artifact_evidence_source_closure(
    recorded: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    a609_source = {
        "git_sha": G02_ROSTER_BRIDGE_COMMIT_GIT_SHA,
        "git_tree_sha": G02_ROSTER_BRIDGE_COMMIT_TREE_SHA,
        "dirty": False,
    }
    first_hop = g02_roster_bridge(recorded, a609_source)
    second_hop = artifact_evidence_bridge(a609_source, current)
    return _ordered_g02_artifact_evidence_closure(
        str(recorded["git_sha"]),
        str(current["git_sha"]),
        first_hop,
        second_hop,
    )


def source_closure(
    source: dict[str, Any], current: dict[str, Any], key: str, label: str
) -> dict[str, Any]:
    recorded_sha = str(source["git_sha"])
    require(
        git_text("rev-parse", f"{recorded_sha}^{{tree}}") == source["git_tree_sha"],
        f"{label} recorded source tree differs from git",
    )
    if source == current:
        return {
            "from_git_sha": recorded_sha,
            "to_git_sha": current["git_sha"],
            "changed_files": [],
            "changed_file_count": 0,
            "policy": "exact-source",
        }
    require(
        key in {"source", "numerics"},
        f"{label} must use current-source evidence",
    )
    if (
        recorded_sha == G02_ROSTER_BRIDGE_BASE_GIT_SHA
        and current["git_sha"] == G02_ROSTER_BRIDGE_COMMIT_GIT_SHA
    ):
        bridge = g02_roster_bridge(source, current)
        return {
            "from_git_sha": recorded_sha,
            "to_git_sha": current["git_sha"],
            "changed_files": copy.deepcopy(bridge["changed_files"]),
            "changed_file_count": len(bridge["changed_files"]),
            "policy": "g02-roster-only-evidence-bridge",
            "bridge": bridge,
        }
    if recorded_sha == ARTIFACT_EVIDENCE_BRIDGE_BASE_GIT_SHA:
        return artifact_evidence_source_closure(source, current)
    require(
        recorded_sha == G02_ROSTER_BRIDGE_BASE_GIT_SHA,
        f"{label} evidence does not start at sealed 05a or a609",
    )
    return g02_then_artifact_evidence_source_closure(source, current)


def validate_dependencies(
    paths: dict[str, Path], current: dict[str, Any]
) -> dict[str, Any]:
    require(set(paths) == set(DEPENDENCY_KEYS), "R0 dependency path set mismatch")
    dependencies: dict[str, Any] = {}
    for key in DEPENDENCY_KEYS:
        path = paths[key].expanduser().resolve()
        source = source_from_outer(path, key)
        closure = source_closure(source, current, key, f"R0 {key}")
        validated = g08a.validate_outer(
            path,
            key,
            g08a.DEPENDENCY_SPECS[key],
            source,
            verify_checkout=False,
        )
        if key == "s2":
            child_path = Path(validated["child_manifest"]["path"])
            deep = s2_contract.verify_checkpoint_manifest(
                child_path,
                verify_checkout=False,
            )
            require(deep.get("source") == source, "R0 S2 deep source binding mismatch")
            bindings = deep.get("bindings")
            require(isinstance(bindings, dict), "R0 S2 deep bindings are missing")
            validated["summary"]["product_execution_identity"] = copy.deepcopy(
                bindings.get("product_execution_identity")
            )
        dependencies[key] = {
            "source": source,
            "source_closure": closure,
            "outer_manifest": validated["outer_manifest"],
            "child_manifest": validated["child_manifest"],
            "summary": validated["summary"],
        }
    require(
        dependencies["source"]["source"] == dependencies["numerics"]["source"],
        "R0 G08A source and numerics evidence must share one source identity",
    )
    return dependencies


def acceptance(dependencies: dict[str, Any]) -> dict[str, Any]:
    source_summary = dependencies["source"]["summary"]["summary"]
    require(source_summary.get("lifecycle_ownership_categories") == 5, "R0 lifecycle category coverage differs")
    require(source_summary.get("lifecycle_implementation_owner_count") == 1, "R0 lifecycle owner count differs")
    require(source_summary.get("legacy_source_selection_count") == 0, "R0 legacy selection is nonzero")
    require(source_summary.get("provider_file_count", 10**9) <= 8, "R0 provider file limit failed")
    require(source_summary.get("provider_glue_production_loc", 10**9) <= 1500, "R0 provider glue limit failed")

    numerics = dependencies["numerics"]["summary"]
    parity = numerics.get("token_parity")
    same_history = numerics.get("same_history")
    require(isinstance(parity, dict) and isinstance(same_history, dict), "R0 numerical summaries are missing")
    require(numerics.get("operation_state_row_count", 0) >= 27, "R0 operation/state coverage is incomplete")
    require(numerics.get("layer_checkpoint_count") == 2, "R0 layer checkpoint coverage differs")
    require(parity.get("case_count") == parity.get("prompt_token_match_count") == 20, "R0 prompt token parity is incomplete")
    require(parity.get("product_output_token_count_per_runtime") == 1280, "R0 token observation denominator differs")
    require(parity.get("exception_count") == parity.get("waiver_count") == 0, "R0 token parity contains an exception or waiver")
    require(same_history.get("case_count") == 20, "R0 same-history prompt denominator differs")
    require(same_history.get("validated_decision_count") == 1280, "R0 same-history decision denominator differs")
    require(same_history.get("ferrum_oracle_exact_count") == 1280, "R0 Ferrum/oracle exact decisions are incomplete")
    require(same_history.get("exception_count") == same_history.get("waiver_count") == 0, "R0 same-history contains an exception or waiver")

    s2 = dependencies["s2"]["summary"]
    s2_acceptance = s2.get("acceptance")
    require(
        isinstance(s2_acceptance, dict)
        and s2_acceptance
        and all(value is True for value in s2_acceptance.values()),
        "R0 CUDA S2 acceptance is incomplete",
    )
    require(s2.get("historical_case_count") == 5, "R0 historical resource case denominator differs")
    require(s2.get("historical_source_test_count") == 7, "R0 historical source test denominator differs")
    product_identity = s2.get("product_execution_identity")
    require(isinstance(product_identity, dict), "R0 CUDA product execution identity is missing")
    require(
        product_identity.get("entrypoints") == ["run", "serve"]
        and product_identity.get("same_resolved_execution_plan") is True
        and product_identity.get("same_runtime_implementation") is True
        and product_identity.get("production_legacy_selection_count") == 0,
        "R0 CUDA run/serve plan or runtime identity failed",
    )
    return {
        "source_ownership": {
            "lifecycle_categories": 5,
            "lifecycle_owner_count": 1,
            "legacy_selection_count": 0,
            "provider_file_count": source_summary["provider_file_count"],
            "provider_glue_production_loc": source_summary["provider_glue_production_loc"],
        },
        "numerics": {
            "operation_state_rows": numerics["operation_state_row_count"],
            "layer_checkpoints": numerics["layer_checkpoint_count"],
            "prompt_token_parity": "20/20",
            "same_history_decisions": "1280/1280",
            "free_run_exact_sequences_diagnostic": f"{parity.get('generated_sequence_match_count')}/20",
            "waiver_count": 0,
            "exception_count": 0,
        },
        "cuda_s2": {
            "acceptance_count": len(s2_acceptance),
            "historical_resource_cases": "5/5",
            "historical_source_tests": 7,
            "product_entrypoints": ["run", "serve"],
            "resolved_execution_plan_hash": product_identity[
                "resolved_execution_plan_hash"
            ],
            "runtime_implementation_fingerprint": product_identity[
                "runtime_implementation_fingerprint"
            ],
            "same_resolved_execution_plan": True,
            "same_runtime_implementation": True,
            "production_legacy_selection_count": 0,
        },
    }


def dependency_checkpoint(dependency: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": copy.deepcopy(dependency["source"]),
        "source_closure": copy.deepcopy(dependency["source_closure"]),
        "outer_manifest": copy.deepcopy(dependency["outer_manifest"]),
        "child_manifest": copy.deepcopy(dependency["child_manifest"]),
        "summary": copy.deepcopy(dependency["summary"]),
    }


def verify_manifest(manifest_path: Path, *, verify_checkout: bool = True) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    manifest = read_json(path, "R0 manifest")
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
            "acceptance",
            "unlocks",
            "does_not_prove",
            "created_at",
            "pass_line",
        },
        "R0 manifest field set mismatch",
    )
    root = path.parent
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type") == "runtime_vnext_r0_core_closure_manifest"
        and manifest.get("checkpoint_id") == "R0"
        and manifest.get("lane") == "runtime-vnext-r0"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True
        and Path(str(manifest.get("artifact_dir", ""))).expanduser().resolve() == root,
        "R0 manifest identity/status mismatch",
    )
    require(manifest.get("pass_line") == f"{PASS_PREFIX}: {root}", "R0 manifest PASS line mismatch")
    source = normalize_source(manifest.get("source"), "R0 current")
    if verify_checkout:
        require(source == current_source(), "R0 aggregate source is stale")
    dependencies = manifest.get("dependencies")
    require(isinstance(dependencies, dict) and set(dependencies) == set(DEPENDENCY_KEYS), "R0 dependency set mismatch")
    for key in DEPENDENCY_KEYS:
        dependency = dependencies[key]
        require(
            isinstance(dependency, dict)
            and set(dependency)
            == {"source", "source_closure", "outer_manifest", "child_manifest", "summary"},
            f"R0 {key} dependency fields differ",
        )
        dependency_source = normalize_source(dependency["source"], f"R0 {key}")
        validate_ref(dependency["outer_manifest"], f"R0 {key} outer manifest")
        validate_ref(dependency["child_manifest"], f"R0 {key} child manifest")
        expected_closure = source_closure(
            dependency_source, source, key, f"R0 {key}"
        )
        require(dependency["source_closure"] == expected_closure, f"R0 {key} source closure mismatch")
    require(manifest.get("acceptance") == acceptance(dependencies), "R0 acceptance summary mismatch")
    require(manifest.get("unlocks") == ["R1"], "R0 unlock set mismatch")
    require(manifest.get("does_not_prove") == DOES_NOT_PROVE, "R0 does_not_prove mismatch")
    return {
        "kind": "vnext-r0",
        "child_manifest": {"path": str(path), "sha256": sha256(path)},
        "source": source,
        "acceptance": copy.deepcopy(manifest["acceptance"]),
    }


def build(paths: dict[str, Path], out: Path) -> str:
    output = out.expanduser().resolve()
    require(REPO_ROOT not in output.parents and output != REPO_ROOT, "R0 output must be outside the source tree")
    require(not output.exists() or not any(output.iterdir()), f"R0 output must be absent or empty: {output}")
    source = current_source()
    dependencies = validate_dependencies(paths, source)
    accepted = acceptance(dependencies)
    staging_parent = output.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=staging_parent))
    try:
        pass_line = f"{PASS_PREFIX}: {output}"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_r0_core_closure_manifest",
            "checkpoint_id": "R0",
            "lane": "runtime-vnext-r0",
            "status": "pass",
            "canonical": True,
            "artifact_dir": str(output),
            "source": source,
            "dependencies": {
                key: dependency_checkpoint(dependencies[key]) for key in DEPENDENCY_KEYS
            },
            "acceptance": accepted,
            "unlocks": ["R1"],
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


def expect_reject(action: Any, marker: str) -> None:
    try:
        action()
    except R0Error:
        return
    raise R0Error(f"self-test mutation was accepted: {marker}")


def fixture_dependencies() -> dict[str, Any]:
    return {
        "source": {
            "summary": {
                "summary": {
                    "lifecycle_ownership_categories": 5,
                    "lifecycle_implementation_owner_count": 1,
                    "legacy_source_selection_count": 0,
                    "provider_file_count": 1,
                    "provider_glue_production_loc": 568,
                }
            }
        },
        "numerics": {
            "summary": {
                "operation_state_row_count": 27,
                "layer_checkpoint_count": 2,
                "token_parity": {
                    "case_count": 20,
                    "prompt_token_match_count": 20,
                    "product_output_token_count_per_runtime": 1280,
                    "generated_sequence_match_count": 19,
                    "exception_count": 0,
                    "waiver_count": 0,
                },
                "same_history": {
                    "case_count": 20,
                    "validated_decision_count": 1280,
                    "ferrum_oracle_exact_count": 1280,
                    "exception_count": 0,
                    "waiver_count": 0,
                },
            }
        },
        "s2": {
            "summary": {
                "acceptance": copy.deepcopy(s2_contract.ACCEPTANCE),
                "historical_case_count": 5,
                "historical_source_test_count": 7,
                "product_execution_identity": {
                    "entrypoints": ["run", "serve"],
                    "resolved_execution_plan_hash": "1" * 64,
                    "runtime_implementation_fingerprint": "2" * 64,
                    "same_resolved_execution_plan": True,
                    "same_runtime_implementation": True,
                    "production_legacy_selection_count": 0,
                },
            }
        },
    }


def fixture_artifact_evidence_changes(
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str]]:
    new_blobs = {
        ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_PATH: "1" * 40,
        ARTIFACT_EVIDENCE_BRIDGE_BASELINE_PATH: "2" * 40,
        ARTIFACT_EVIDENCE_BRIDGE_R0_PATH: "3" * 40,
        ARTIFACT_EVIDENCE_BRIDGE_R1_PATH: "4" * 40,
    }
    modes = {
        ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_PATH: ("000000", "100644", "A"),
        ARTIFACT_EVIDENCE_BRIDGE_BASELINE_PATH: ("100644", "100644", "M"),
        ARTIFACT_EVIDENCE_BRIDGE_R0_PATH: ("100644", "100644", "M"),
        ARTIFACT_EVIDENCE_BRIDGE_R1_PATH: ("100755", "100755", "M"),
    }
    changes = []
    for path in ARTIFACT_EVIDENCE_BRIDGE_CHANGED_FILES:
        old_mode, new_mode, status = modes[path]
        changes.append(
            {
                "path": path,
                "status": status,
                "old_mode": old_mode,
                "new_mode": new_mode,
                "old_blob": ARTIFACT_EVIDENCE_BRIDGE_OLD_BLOBS[path],
                "new_blob": new_blobs[path],
            }
        )
    return changes, new_blobs, {
        path: new_blobs[path]
        for path in (
            ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_PATH,
            ARTIFACT_EVIDENCE_BRIDGE_BASELINE_PATH,
        )
    }


def self_test() -> int:
    bridge_facts = {
        "recorded_sha": G02_ROSTER_BRIDGE_BASE_GIT_SHA,
        "current_sha": G02_ROSTER_BRIDGE_COMMIT_GIT_SHA,
        "parent_shas": [G02_ROSTER_BRIDGE_BASE_GIT_SHA],
        "changed_files": sorted(G02_ROSTER_BRIDGE_CHANGED_FILES),
        "old_blob": G02_ROSTER_BRIDGE_OLD_BLOB,
        "new_blob": G02_ROSTER_BRIDGE_NEW_BLOB,
        "expected_new_blob": G02_ROSTER_BRIDGE_NEW_BLOB,
    }
    sealed_bridge = _validate_g02_roster_bridge_facts(**bridge_facts)
    require(
        sealed_bridge["bridge_id"] == G02_ROSTER_BRIDGE_ID
        and sealed_bridge["g02_validator"]["new_blob"]
        == G02_ROSTER_BRIDGE_NEW_BLOB,
        "R0 sealed G02 roster bridge fixture differs",
    )
    for field, value, marker in (
        ("old_blob", "d" * 40, "wrong old G02 blob"),
        ("new_blob", "e" * 40, "wrong new G02 blob"),
        ("parent_shas", ["f" * 40], "second commit after 05a"),
        ("current_sha", "c" * 40, "unsealed direct child of 05a"),
        (
            "changed_files",
            list(reversed(sorted(G02_ROSTER_BRIDGE_CHANGED_FILES))),
            "wrong G02 file order",
        ),
    ):
        mutated = copy.deepcopy(bridge_facts)
        mutated[field] = value
        expect_reject(
            lambda mutated=mutated: _validate_g02_roster_bridge_facts(**mutated),
            marker,
        )
    artifact_changes, current_blobs, expected_final_blobs = (
        fixture_artifact_evidence_changes()
    )
    artifact_facts = {
        "base_sha": ARTIFACT_EVIDENCE_BRIDGE_BASE_GIT_SHA,
        "base_tree_sha": ARTIFACT_EVIDENCE_BRIDGE_BASE_TREE_SHA,
        "current_sha": "c" * 40,
        "current_tree_sha": "d" * 40,
        "parent_shas": [ARTIFACT_EVIDENCE_BRIDGE_BASE_GIT_SHA],
        "changes": artifact_changes,
        "current_blobs": current_blobs,
        "expected_final_blobs": expected_final_blobs,
    }
    sealed_artifact_bridge = _validate_artifact_evidence_bridge_facts(
        **artifact_facts
    )
    require(
        sealed_artifact_bridge["bridge_id"] == ARTIFACT_EVIDENCE_BRIDGE_ID
        and sealed_artifact_bridge["changed_files"]
        == list(ARTIFACT_EVIDENCE_BRIDGE_CHANGED_FILES),
        "R0 sealed artifact-evidence bridge fixture differs",
    )
    raw_fixture = "\n".join(
        (
            f":{row['old_mode']} {row['new_mode']} {row['old_blob']} "
            f"{row['new_blob']} {row['status']}\t{row['path']}"
        )
        for row in artifact_changes
    )
    require(
        _parse_raw_git_changes(raw_fixture) == artifact_changes,
        "R0 artifact-evidence raw diff parser fixture differs",
    )
    for field, value, marker in (
        ("base_sha", G02_ROSTER_BRIDGE_BASE_GIT_SHA, "flattened 05a to final hop"),
        ("base_tree_sha", "e" * 40, "wrong a609 tree"),
        ("current_tree_sha", "unsealed", "invalid final tree"),
        ("parent_shas", ["f" * 40], "wrong a609 parent"),
        (
            "parent_shas",
            [ARTIFACT_EVIDENCE_BRIDGE_BASE_GIT_SHA, "f" * 40],
            "merge child of a609",
        ),
        ("expected_final_blobs", {}, "unsealed final blobs"),
        ("current_blobs", {}, "missing current blob closure"),
    ):
        mutated = copy.deepcopy(artifact_facts)
        mutated[field] = value
        expect_reject(
            lambda mutated=mutated: _validate_artifact_evidence_bridge_facts(
                **mutated
            ),
            marker,
        )
    mutated = copy.deepcopy(artifact_facts)
    mutated["changes"][0], mutated["changes"][1] = (
        mutated["changes"][1],
        mutated["changes"][0],
    )
    expect_reject(
        lambda: _validate_artifact_evidence_bridge_facts(**mutated),
        "wrong artifact-evidence changed-file order",
    )
    mutated = copy.deepcopy(artifact_facts)
    mutated["changes"].pop()
    expect_reject(
        lambda: _validate_artifact_evidence_bridge_facts(**mutated),
        "missing artifact-evidence changed file",
    )
    for field, value, marker in (
        ("status", "A", "wrong baseline status"),
        ("old_mode", "100755", "wrong baseline old mode"),
    ):
        mutated = copy.deepcopy(artifact_facts)
        mutated["changes"][1][field] = value
        expect_reject(
            lambda mutated=mutated: _validate_artifact_evidence_bridge_facts(
                **mutated
            ),
            marker,
        )
    mutated = copy.deepcopy(artifact_facts)
    mutated["expected_final_blobs"] = {
        ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_PATH: (
            ARTIFACT_EVIDENCE_BRIDGE_AMENDMENT_NEW_BLOB
        ),
        ARTIFACT_EVIDENCE_BRIDGE_BASELINE_PATH: (
            ARTIFACT_EVIDENCE_BRIDGE_BASELINE_NEW_BLOB
        ),
    }
    expect_reject(
        lambda: _validate_artifact_evidence_bridge_facts(**mutated),
        "integration-dependent final blobs remain unsealed",
    )
    for path, marker in (
        ("crates/ferrum-engine/src/lib.rs", "crate change"),
        ("Cargo.toml", "Cargo change"),
        (
            "scripts/release/configs/runtime_vnext_g08a_m1_cuda.models.lock.json",
            "model lock change",
        ),
        (
            "scripts/release/configs/runtime_vnext_g08a_source_contract.json",
            "release config change",
        ),
        ("ferrum.toml", "runtime config change"),
        (
            "scripts/release/scenarios/runtime_vnext_r1_llama_dense_cuda.json",
            "product scenario change",
        ),
    ):
        mutated = copy.deepcopy(artifact_facts)
        extra = copy.deepcopy(mutated["changes"][-1])
        extra["path"] = path
        mutated["changes"].append(extra)
        expect_reject(
            lambda mutated=mutated: _validate_artifact_evidence_bridge_facts(
                **mutated
            ),
            marker,
        )
    for index, change in enumerate(artifact_changes):
        mutated = copy.deepcopy(artifact_facts)
        mutated["changes"][index]["old_blob"] = "e" * 40
        expect_reject(
            lambda mutated=mutated: _validate_artifact_evidence_bridge_facts(
                **mutated
            ),
            f"wrong a609 blob for {change['path']}",
        )
        mutated = copy.deepcopy(artifact_facts)
        mutated["changes"][index]["new_blob"] = "f" * 40
        expect_reject(
            lambda mutated=mutated: _validate_artifact_evidence_bridge_facts(
                **mutated
            ),
            f"wrong final tree blob for {change['path']}",
        )
    for path in expected_final_blobs:
        index = ARTIFACT_EVIDENCE_BRIDGE_CHANGED_FILES.index(path)
        mutated = copy.deepcopy(artifact_facts)
        mutated["changes"][index]["new_blob"] = "f" * 40
        expect_reject(
            lambda mutated=mutated: _validate_artifact_evidence_bridge_facts(
                **mutated
            ),
            f"wrong sealed final blob for {path}",
        )
    g02_hop = _validate_g02_roster_bridge_facts(**bridge_facts)
    ordered = _ordered_g02_artifact_evidence_closure(
        G02_ROSTER_BRIDGE_BASE_GIT_SHA,
        artifact_facts["current_sha"],
        g02_hop,
        sealed_artifact_bridge,
    )
    require(
        ordered["hop_count"] == 2
        and [hop["bridge_id"] for hop in ordered["bridge_hops"]]
        == [G02_ROSTER_BRIDGE_ID, ARTIFACT_EVIDENCE_BRIDGE_ID],
        "R0 ordered 05a -> a609 -> final bridge differs",
    )
    expect_reject(
        lambda: _ordered_g02_artifact_evidence_closure(
            G02_ROSTER_BRIDGE_BASE_GIT_SHA,
            artifact_facts["current_sha"],
            sealed_artifact_bridge,
            g02_hop,
        ),
        "reversed evidence bridge hops",
    )
    single = _single_artifact_evidence_closure(
        ARTIFACT_EVIDENCE_BRIDGE_BASE_GIT_SHA,
        artifact_facts["current_sha"],
        sealed_artifact_bridge,
    )
    require(
        single["hop_count"] == 1
        and single["bridge_hops"][0]["bridge_id"]
        == ARTIFACT_EVIDENCE_BRIDGE_ID,
        "R0 a609 -> final evidence bridge differs",
    )
    expect_reject(
        lambda: _single_artifact_evidence_closure(
            G02_ROSTER_BRIDGE_BASE_GIT_SHA,
            artifact_facts["current_sha"],
            sealed_artifact_bridge,
        ),
        "flattened single-hop 05a evidence",
    )
    for extra_path, marker in (
        ("crates/ferrum-engine/src/lib.rs", "extra product change"),
        (
            "scripts/release/scenarios/runtime_vnext_s2_multiturn_concurrency_cuda.json",
            "extra scenario change",
        ),
    ):
        mutated = copy.deepcopy(bridge_facts)
        mutated["changed_files"].append(extra_path)
        expect_reject(
            lambda mutated=mutated: _validate_g02_roster_bridge_facts(**mutated),
            marker,
        )
    allowed, rejected = control_plane_only(
        [
            "docs/goals/r0.md",
            "scripts/release/runtime_vnext_r0_core_closure.py",
            "scripts/release/change_impact_rules.json",
            "scripts/release/runtime_vnext_s2_cuda_product_contract.py",
            S2_MULTITURN_SCENARIO,
        ],
        "numerics",
    )
    require(len(allowed) == 5 and not rejected, "R0 control-plane closure rejected allowed paths")
    patch_control_plane = [
        "docs/goals/runtime-vnext-0.8.0-2026-07-10/CORRECTNESS_ACCEPTANCE_AMENDMENT_2026-08-07.md",
        "scripts/release/runtime_vnext_r0_core_closure.py",
        R1_AGGREGATOR,
    ]
    for key in DEPENDENCY_KEYS:
        allowed, rejected = control_plane_only(patch_control_plane, key)
        require(
            allowed == patch_control_plane and not rejected,
            f"R0 {key} closure rejected aggregate control-plane patch paths",
        )
    s2_patch_paths = [SAME_HISTORY_COLLECTOR, *patch_control_plane]
    allowed, rejected = control_plane_only(s2_patch_paths, "s2")
    require(
        allowed == s2_patch_paths and not rejected,
        "R0 S2 closure rejected numerics-only collector containment",
    )
    allowed, rejected = control_plane_only(s2_patch_paths, "numerics")
    require(
        allowed == patch_control_plane and rejected == [SAME_HISTORY_COLLECTOR],
        "R0 numerics closure accepted stale same-history collector evidence",
    )
    _, rejected = control_plane_only([S2_MULTITURN_SCENARIO], "s2")
    require(
        rejected == [S2_MULTITURN_SCENARIO],
        "R0 S2 closure accepted a changed product scenario",
    )
    for key in ("source", "numerics"):
        allowed, rejected = control_plane_only([S0A_CONTRACT_SPLIT], key)
        require(
            allowed == [S0A_CONTRACT_SPLIT] and not rejected,
            f"R0 {key} closure rejected the G01A control-plane validator",
        )
    _, rejected = control_plane_only([S0A_CONTRACT_SPLIT], "s2")
    require(
        rejected == [S0A_CONTRACT_SPLIT],
        "R0 S2 closure accepted a changed G01A control-plane validator",
    )
    _, rejected = control_plane_only(["crates/ferrum-engine/src/lib.rs"], "numerics")
    require(rejected == ["crates/ferrum-engine/src/lib.rs"], "R0 product change was not rejected")
    dependencies = fixture_dependencies()
    accepted = acceptance(dependencies)
    require(
        accepted["numerics"]["same_history_decisions"] == "1280/1280"
        and accepted["cuda_s2"]["historical_resource_cases"] == "5/5",
        "R0 fixture acceptance differs",
    )
    bad = copy.deepcopy(dependencies)
    bad["source"]["summary"]["summary"]["lifecycle_implementation_owner_count"] = 2
    expect_reject(lambda: acceptance(bad), "multiple lifecycle owners")
    bad = copy.deepcopy(dependencies)
    bad["numerics"]["summary"]["same_history"]["ferrum_oracle_exact_count"] = 1279
    expect_reject(lambda: acceptance(bad), "numerical decision miss")
    bad = copy.deepcopy(dependencies)
    bad["s2"]["summary"]["acceptance"]["run_and_serve_product_paths_covered"] = False
    expect_reject(lambda: acceptance(bad), "S2 product path miss")
    source = {
        "git_sha": git_text("rev-parse", "HEAD"),
        "git_tree_sha": git_text("rev-parse", "HEAD^{tree}"),
        "dirty": False,
    }
    with tempfile.TemporaryDirectory(prefix="ferrum-r0-selftest-") as temporary:
        root = Path(temporary).resolve()
        sealed_dependencies = fixture_dependencies()
        for key in DEPENDENCY_KEYS:
            outer = root / f"{key}.outer.json"
            child = root / f"{key}.child.json"
            write_json(outer, {"key": key, "kind": "outer"})
            write_json(child, {"key": key, "kind": "child"})
            sealed_dependencies[key].update(
                {
                    "source": source,
                    "source_closure": source_closure(source, source, key, key),
                    "outer_manifest": file_ref(outer),
                    "child_manifest": file_ref(child),
                }
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_r0_core_closure_manifest",
            "checkpoint_id": "R0",
            "lane": "runtime-vnext-r0",
            "status": "pass",
            "canonical": True,
            "artifact_dir": str(root),
            "source": source,
            "dependencies": sealed_dependencies,
            "acceptance": acceptance(sealed_dependencies),
            "unlocks": ["R1"],
            "does_not_prove": DOES_NOT_PROVE,
            "created_at": "2026-08-06T00:00:00+00:00",
            "pass_line": f"{PASS_PREFIX}: {root}",
        }
        write_json(root / "manifest.json", manifest)
        verified = verify_manifest(root / "manifest.json", verify_checkout=False)
        require(verified.get("kind") == "vnext-r0", "R0 manifest fixture failed")
        manifest["acceptance"]["cuda_s2"]["production_legacy_selection_count"] = 1
        write_json(root / "manifest.json", manifest)
        expect_reject(
            lambda: verify_manifest(root / "manifest.json", verify_checkout=False),
            "manifest acceptance mutation",
        )
    print(SELFTEST_PASS_LINE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g08a-source", type=Path)
    parser.add_argument("--g08a-numerics", type=Path)
    parser.add_argument("--s2", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test()
        required = {
            "--g08a-source": args.g08a_source,
            "--g08a-numerics": args.g08a_numerics,
            "--s2": args.s2,
            "--out": args.out,
        }
        missing = [flag for flag, value in required.items() if value is None]
        require(not missing, "missing required arguments: " + ", ".join(missing))
        assert args.g08a_source is not None
        assert args.g08a_numerics is not None
        assert args.s2 is not None
        assert args.out is not None
        pass_line = build(
            {
                "source": args.g08a_source,
                "numerics": args.g08a_numerics,
                "s2": args.s2,
            },
            args.out,
        )
        print(pass_line)
        return 0
    except (OSError, R0Error, RuntimeError, ValueError) as error:
        print(f"FERRUM RUNTIME VNEXT R0 CORE CLOSURE FAIL: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
