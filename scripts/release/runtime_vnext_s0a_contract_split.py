#!/usr/bin/env python3
"""Produce the canonical Runtime vNext S0A contract-split artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL_ROOT = REPO_ROOT / "docs/goals/runtime-vnext-0.8.0-2026-07-10"
BASELINE_COMMIT = "b5377b12464b60203a3fe57a6de4c9952ed2474b"
PASS_PREFIX = "FERRUM RUNTIME VNEXT G01A CONTRACT SPLIT PASS"
OWNER_MAP_PASS_PREFIX = "VNEXT PUBLIC OWNER MAP PASS"
BOUNDED_COMMAND = REPO_ROOT / "scripts/release/bounded_command.py"
OWNER_MAP_EXAMPLE = "vnext_public_owner_map"
INVENTORY_DOCUMENT = REPO_ROOT / "docs/release/cleanup/20260714-inventory.md"
ADR_DOCUMENT = GOAL_ROOT / "S0A_CONTRACT_SPLIT_MAP.md"

PRODUCTION_GROUPS = {
    "resource": {
        "topological_order": [
            "contracts",
            "ledger",
            "capacity",
            "allocation",
            "dynamic_pool",
            "provisioning",
            "static_lease",
            "plan_runtime",
            "transaction",
            "work",
            "recovery",
            "sequence",
            "batch",
            "invocation",
        ],
        "audit": "S0A_RESOURCE_DEPENDENCY_AUDIT.md",
    },
    "execution": {
        "topological_order": [
            "foundation",
            "binding",
            "work",
            "workspace",
            "provider_resource",
            "contracts",
            "storage",
            "allocation",
            "solver",
            "memory",
            "provider",
            "policy",
            "plan",
            "resolution",
            "validation",
            "planner",
        ],
        "audit": "S0A_EXECUTION_DEPENDENCY_AUDIT.md",
    },
    "event": {
        "topological_order": [
            "foundation",
            "identity",
            "topology",
            "sequence_binding",
            "execution_event",
            "resource_pool",
            "replay",
            "sink",
        ],
        "audit": "S0A_EVENT_DEPENDENCY_AUDIT.md",
    },
}

TEST_TARGET_GROUPS = {
    "resource": [
        "vnext_resource_capacity_contract_tests",
        "vnext_resource_transaction_lifecycle_tests",
        "vnext_resource_transaction_evidence_tests",
        "vnext_resource_sequence_activation_tests",
        "vnext_resource_sequence_recovery_tests",
        "vnext_resource_recovery_authority_tests",
        "vnext_resource_runtime_close_tests",
    ],
    "event": [
        "vnext_event_execution_contract_tests",
        "vnext_event_sink_contract_tests",
        "vnext_event_resource_pool_contract_tests",
        "vnext_event_recovery_contract_tests",
        "vnext_event_replay_contract_tests",
    ],
    "core": [
        "vnext_planning_resource_contract_tests",
        "vnext_plan_wire_contract_tests",
        "vnext_provider_selection_contract_tests",
        "vnext_weight_layout_contract_tests",
        "vnext_resolution_contract_tests",
        "vnext_execution_graph_contract_tests",
        "vnext_source_audit_contract_tests",
    ],
    "device_operation": [
        "vnext_device_operation_batch_contract_tests",
        "vnext_device_operation_cancel_contract_tests",
        "vnext_device_operation_completion_contract_tests",
        "vnext_device_operation_dispatch_contract_tests",
        "vnext_device_operation_legacy_authority_contract_tests",
    ],
}

SHARED_TEST_SUPPORT = [
    "crates/ferrum-interfaces/tests/vnext_resource_contract/support.rs",
    "crates/ferrum-interfaces/tests/vnext_core_contract/mod.rs",
    "crates/ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs",
    "crates/ferrum-interfaces/tests/vnext_event_contract/mod.rs",
    "crates/ferrum-interfaces/tests/vnext_event_contract/event_fixture.rs",
    "crates/ferrum-interfaces/tests/vnext_event_contract/execution_fixture.rs",
    "crates/ferrum-interfaces/tests/vnext_event_contract/model.rs",
    "crates/ferrum-interfaces/tests/vnext_event_contract/resolution.rs",
    "crates/ferrum-interfaces/tests/vnext_event_contract/resource_fixture.rs",
    "crates/ferrum-interfaces/tests/vnext_event_contract/runtime.rs",
]

REMOVED_OVERSIZED_TARGETS = [
    "crates/ferrum-interfaces/tests/vnext_contract_tests.rs",
    "crates/ferrum-interfaces/tests/vnext_resource_contract_tests.rs",
    "crates/ferrum-interfaces/tests/vnext_event_contract_tests.rs",
    "crates/ferrum-interfaces/tests/vnext_device_operation_contract_tests.rs",
]


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"required regular file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_result(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git_bytes(*args: str) -> bytes:
    result = git_result(*args)
    require(
        result.returncode == 0,
        f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}",
    )
    return result.stdout


def git_text(*args: str) -> str:
    return git_bytes(*args).decode().strip()


def clean_source() -> dict[str, Any]:
    status = [line for line in git_text("status", "--short").splitlines() if line]
    require(not status, f"S0A gate requires a clean checkout: {status}")
    require(
        git_result("merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD").returncode == 0,
        f"S0A baseline {BASELINE_COMMIT} is not an ancestor of HEAD",
    )
    return {
        "git_sha": git_text("rev-parse", "HEAD"),
        "git_tree_sha": git_text("rev-parse", "HEAD^{tree}"),
        "dirty": False,
        "status_short": [],
    }


def bind_g00f(g00f_outer_path: Path, source: dict[str, Any]) -> dict[str, Any]:
    outer_path = g00f_outer_path.resolve()
    require(REPO_ROOT not in outer_path.parents, "G00F artifact must be outside the source tree")
    outer = read_json(outer_path, "G00F outer manifest")
    require(
        outer.get("lane") == "vnext-g00f"
        and outer.get("status") == "pass"
        and outer.get("git_sha") == source["git_sha"]
        and outer.get("dirty_status") == {"is_dirty": False, "status_short": []},
        "G00F outer manifest is stale or invalid",
    )
    child_ref = outer.get("child_artifacts", {}).get("child_manifest")
    require(isinstance(child_ref, dict), "G00F outer manifest lacks child identity")
    child_path = Path(str(child_ref.get("path", ""))).resolve()
    child_digest = sha256(child_path)
    require(child_ref.get("sha256") == child_digest, "G00F child manifest SHA256 mismatch")
    child = read_json(child_path, "G00F child manifest")

    sys.path.insert(0, str(REPO_ROOT / "scripts/release"))
    import run_gate  # pylint: disable=import-outside-toplevel

    provenance = run_gate.validate_vnext_g00f_provenance(
        run_gate.LaneCommand(
            cmd=[],
            expected_child_pass_line=child.get("pass_line"),
            child_manifest_path=child_path,
            provenance_kind="vnext-g00f",
        ),
        child,
        child_digest,
        verify_checkout=True,
    )
    return {
        "outer_manifest": {"path": str(outer_path), "sha256": sha256(outer_path)},
        "child_manifest": {"path": str(child_path), "sha256": child_digest},
        "source": provenance["source"],
        "g00a": provenance["g00a"],
    }


def physical_and_logical_lines(payload: bytes) -> tuple[int, int]:
    text = payload.decode("utf-8")
    physical = len(text.splitlines())
    logical = 0
    in_block = False
    for raw in text.splitlines():
        line = raw.strip()
        if in_block:
            if "*/" in line:
                line = line.split("*/", 1)[1].strip()
                in_block = False
            else:
                continue
        while line.startswith("/*"):
            if "*/" in line[2:]:
                line = line.split("*/", 1)[1].strip()
            else:
                in_block = True
                line = ""
        if line and not line.startswith("//"):
            logical += 1
    return physical, logical


def source_row(path: Path, category: str) -> dict[str, Any]:
    relative = path.relative_to(REPO_ROOT).as_posix()
    payload = path.read_bytes()
    physical, logical = physical_and_logical_lines(payload)
    return {
        "path": relative,
        "category": category,
        "physical_lines": physical,
        "logical_lines": logical,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "git_blob": git_text("rev-parse", f"HEAD:{relative}"),
    }


def collect_split_inventory(source: dict[str, Any]) -> dict[str, Any]:
    production_rows: list[dict[str, Any]] = []
    facade_rows: list[dict[str, Any]] = []
    for group in PRODUCTION_GROUPS:
        facade = REPO_ROOT / f"crates/ferrum-interfaces/src/vnext/{group}.rs"
        facade_rows.append(source_row(facade, "production_facade"))
        owner_dir = REPO_ROOT / f"crates/ferrum-interfaces/src/vnext/{group}"
        for path in sorted(owner_dir.glob("*.rs")):
            if path.stem == "tests" or path.stem.endswith("_tests"):
                continue
            production_rows.append(source_row(path, "production_owner"))

    target_rows = []
    for targets in TEST_TARGET_GROUPS.values():
        for target in targets:
            target_rows.append(
                source_row(
                    REPO_ROOT / f"crates/ferrum-interfaces/tests/{target}.rs",
                    "contract_test_target",
                )
            )
    support_rows = [source_row(REPO_ROOT / path, "shared_test_support_owner") for path in SHARED_TEST_SUPPORT]

    require(
        max(row["physical_lines"] for row in facade_rows) <= 500,
        "production facade exceeds 500 physical lines",
    )
    require(
        max(row["physical_lines"] for row in production_rows) <= 2500,
        "production owner exceeds 2500 physical lines",
    )
    require(
        max(row["physical_lines"] for row in target_rows + support_rows) <= 2000,
        "contract test or reusable support owner exceeds 2000 physical lines",
    )
    for removed in REMOVED_OVERSIZED_TARGETS:
        require(not (REPO_ROOT / removed).exists(), f"removed oversized target reappeared: {removed}")

    split_paths = [row["path"] for row in facade_rows + production_rows + target_rows + support_rows]
    for relative in split_paths:
        payload = (REPO_ROOT / relative).read_text()
        require("include!(" not in payload, f"split source uses include!: {relative}")
        if "/src/vnext/" in relative and not relative.endswith(("dynamic_pool_tests.rs", "sequence_session_frame_tests.rs")):
            require("use super::*" not in payload, f"production owner uses wildcard parent import: {relative}")

    baseline_rows = []
    for group in PRODUCTION_GROUPS:
        relative = f"crates/ferrum-interfaces/src/vnext/{group}.rs"
        payload = git_bytes("show", f"{BASELINE_COMMIT}:{relative}")
        physical, logical = physical_and_logical_lines(payload)
        baseline_rows.append(
            {
                "path": relative,
                "git_commit": BASELINE_COMMIT,
                "physical_lines": physical,
                "logical_lines": logical,
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )

    rows = sorted(
        facade_rows + production_rows + target_rows + support_rows,
        key=lambda row: row["path"],
    )
    return {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s0a_split_inventory",
        "source": source,
        "baseline": {"git_commit": BASELINE_COMMIT, "monoliths": baseline_rows},
        "threshold_policy": {
            "production_facade_max_physical_lines": 500,
            "production_owner_max_physical_lines": 2500,
            "contract_test_target_or_support_owner_max_physical_lines": 2000,
            "shared_support_accounting": "each reusable fixture module is one explicit owner and is not duplicated into every consumer target",
            "physical_lines_are_a_conservative_upper_bound_for_logical_lines": True,
        },
        "files": rows,
        "removed_oversized_targets": REMOVED_OVERSIZED_TARGETS,
        "summary": {
            "file_count": len(rows),
            "facade_count": len(facade_rows),
            "production_owner_count": len(production_rows),
            "contract_test_target_count": len(target_rows),
            "shared_test_support_owner_count": len(support_rows),
            "maximum_facade_physical_lines": max(row["physical_lines"] for row in facade_rows),
            "maximum_production_owner_physical_lines": max(row["physical_lines"] for row in production_rows),
            "maximum_contract_test_or_support_owner_physical_lines": max(
                row["physical_lines"] for row in target_rows + support_rows
            ),
            "include_macro_count": 0,
            "production_wildcard_parent_import_count": 0,
            "removed_oversized_target_present_count": 0,
        },
    }


def contract_map(split_inventory: dict[str, Any]) -> dict[str, Any]:
    groups = []
    inventory_by_path = {row["path"]: row for row in split_inventory["files"]}
    for group, policy in PRODUCTION_GROUPS.items():
        facade_path = f"crates/ferrum-interfaces/src/vnext/{group}.rs"
        owner_prefix = f"crates/ferrum-interfaces/src/vnext/{group}/"
        owners = [
            row
            for path, row in inventory_by_path.items()
            if path.startswith(owner_prefix) and row["category"] == "production_owner"
        ]
        audit_path = GOAL_ROOT / policy["audit"]
        require(audit_path.is_file(), f"dependency audit is missing: {audit_path}")
        groups.append(
            {
                "group": group,
                "facade": inventory_by_path[facade_path],
                "owners": sorted(owners, key=lambda row: row["path"]),
                "owner_count": len(owners),
                "dependency_audit": {
                    "path": audit_path.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256(audit_path),
                    "multi_module_scc_count": 0,
                    "topological_order": policy["topological_order"],
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s0a_contract_map",
        "baseline_commit": BASELINE_COMMIT,
        "production_groups": groups,
        "test_target_groups": TEST_TARGET_GROUPS,
        "shared_test_support": SHARED_TEST_SUPPORT,
        "preserved_invariants": [
            "capacity is published only from committed backing",
            "admission remains capacity-derived and distinguishes defer from impossible",
            "request/sequence/session/step/invocation authorities retain exact lifetimes",
            "possibly-submitted work retains ownership until typed fence terminal",
            "capacity waiting preserves register/recheck and avoids global head-of-line blocking",
        ],
        "summary": {
            "production_group_count": 3,
            "multi_module_scc_count": 0,
            "test_target_count": sum(len(targets) for targets in TEST_TARGET_GROUPS.values()),
            "public_path_policy": "facade re-export preserves ferrum_interfaces::vnext::* paths",
            "semantic_change_count": 0,
        },
    }


def run_public_owner_map(checkpoint_root: Path) -> dict[str, Any]:
    output_path = checkpoint_root / "public-owner-map.json"
    logs = checkpoint_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-owner-baseline-") as temporary:
        baseline_dir = Path(temporary)
        for group in PRODUCTION_GROUPS:
            relative = f"crates/ferrum-interfaces/src/vnext/{group}.rs"
            (baseline_dir / f"{group}.rs").write_bytes(
                git_bytes("show", f"{BASELINE_COMMIT}:{relative}")
            )
        command = [
            "cargo",
            "run",
            "-q",
            "-p",
            "ferrum-interfaces",
            "--example",
            OWNER_MAP_EXAMPLE,
            "--",
            BASELINE_COMMIT,
            str(baseline_dir),
            "crates/ferrum-interfaces/src/vnext",
            str(output_path),
        ]
        started = time.monotonic()
        env = os.environ.copy()
        env.update({"CARGO_BUILD_JOBS": "4", "RUST_TEST_THREADS": "2"})
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        duration = time.monotonic() - started
    if result.stdout:
        (logs / "public-owner-map.stdout").write_text(result.stdout)
    if result.stderr:
        (logs / "public-owner-map.stderr").write_text(result.stderr)
    write_json(
        logs / "public-owner-map.command.json",
        {
            "command": command,
            "cwd": str(REPO_ROOT),
            "env_overrides": {"CARGO_BUILD_JOBS": "4", "RUST_TEST_THREADS": "2"},
            "returncode": result.returncode,
            "duration_seconds": duration,
        },
    )
    require(result.returncode == 0, "public owner map command failed")
    expected_line = f"{OWNER_MAP_PASS_PREFIX}: mapped=1490/1490 lost=0 ambiguous=0 inaccessible=0 added=0 unsupported=0 output={output_path}"
    require(result.stdout.splitlines().count(expected_line) == 1, "public owner map exact PASS line mismatch")
    owner_map = read_json(output_path, "public owner map")
    require(
        owner_map.get("summary")
        == {
            "baseline_items": 1490,
            "mapped_items": 1490,
            "lost_items": 0,
            "ambiguous_items": 0,
            "inaccessible_items": 0,
            "added_items": 0,
            "excluded_non_public_owner_members": 1,
            "unsupported_syntax_count": 0,
            "coverage_percent": 100.0,
            "pass": True,
        },
        "public owner map acceptance summary mismatch",
    )
    return {
        "command": command,
        "duration_seconds": duration,
        "pass_line": expected_line,
        "summary": owner_map["summary"],
    }


def expected_machine_proof_lines() -> list[str]:
    sys.path.insert(0, str(REPO_ROOT / "scripts/release"))
    import runtime_vnext_g01a_checkpoint as historical  # pylint: disable=import-outside-toplevel

    lines = [
        "VNEXT PLAN DETERMINISM PASS: 100/100",
        "VNEXT PLAN ROUNDTRIP PASS: 100/100",
        "VNEXT BREAKING VERSION REJECT PASS: 100/100",
        f"VNEXT FAIL CLOSED PASS: {historical.EXPECTED_FAIL_CLOSED_CASES}/{historical.EXPECTED_FAIL_CLOSED_CASES}",
        f"VNEXT MODEL IDENTITY PASS: {historical.EXPECTED_MODEL_IDENTITY_CASES}/{historical.EXPECTED_MODEL_IDENTITY_CASES}",
        f"VNEXT OPERATION ORACLE PASS: {historical.EXPECTED_ORACLE_CASES}/{historical.EXPECTED_ORACLE_CASES}",
        f"VNEXT MODEL WIRE PASS: {historical.EXPECTED_MODEL_WIRE_CASES}/{historical.EXPECTED_MODEL_WIRE_CASES}",
        "VNEXT LEGACY MAP PASS: 82/82",
    ]
    for proofs in historical.RESOURCE_PROOF_LINES.values():
        lines.extend(f"{prefix}: {count}/{count}" for prefix, count in proofs)
    lines.extend(
        f"{prefix}: {count}/{count}"
        for prefix, count in historical.EVENT_PROOF_LINES.values()
    )
    lines.extend(
        f"{prefix}: {count}/{count}"
        for prefix, count in historical.DEVICE_OPERATION_PROOF_LINES.values()
    )
    return sorted(lines)


def run_bounded_aggregate(checkpoint_root: Path) -> dict[str, Any]:
    logs = checkpoint_root / "logs"
    stdout_path = logs / "all-targets.stdout"
    stderr_path = logs / "all-targets.stderr"
    receipt_path = logs / "all-targets.receipt.json"
    command = [
        sys.executable,
        str(BOUNDED_COMMAND),
        "--receipt",
        str(receipt_path),
        "--stdout-log",
        str(stdout_path),
        "--stderr-log",
        str(stderr_path),
        "--cwd",
        str(REPO_ROOT),
        "--wall-timeout-seconds",
        "1800",
        "--max-processes",
        "24",
        "--max-group-threads",
        "128",
        "--max-per-process-threads",
        "32",
        "--sample-interval-seconds",
        "0.05",
        "--max-sampling-errors",
        "3",
        "--term-grace-seconds",
        "2",
        "--",
        "cargo",
        "test",
        "-p",
        "ferrum-interfaces",
        "--all-targets",
        "--",
        "--test-threads=2",
        "--nocapture",
    ]
    env = os.environ.copy()
    env.update(
        {
            "CARGO_BUILD_JOBS": "4",
            "RUST_TEST_THREADS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    started_at = iso_now()
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    duration = time.monotonic() - started
    finished_at = iso_now()
    if result.stdout:
        (logs / "all-targets.runner.stdout").write_text(result.stdout)
    if result.stderr:
        (logs / "all-targets.runner.stderr").write_text(result.stderr)
    require(receipt_path.is_file(), "bounded aggregate receipt is missing")
    receipt = read_json(receipt_path, "bounded aggregate receipt")
    require(
        result.returncode == 0
        and receipt.get("schema") == "ferrum.bounded-command-receipt.v1"
        and receipt.get("status") == "pass"
        and receipt.get("rc") == 0
        and receipt.get("reason") == "command_completed"
        and receipt.get("violation") is None
        and receipt.get("sampling_errors") == []
        and receipt.get("termination") == {"signals": [], "errors": []}
        and receipt.get("cleanup") == {"process_group_gone": True},
        "bounded aggregate command or receipt failed",
    )
    stdout = stdout_path.read_text(errors="replace")
    stderr = stderr_path.read_text(errors="replace")
    combined = f"{stdout}\n{stderr}"
    require("test result: FAILED" not in combined, "aggregate output contains a failed test result")
    proof_lines = expected_machine_proof_lines()
    combined_lines = [line.strip() for line in combined.splitlines()]
    for line in proof_lines:
        require(combined_lines.count(line) == 1, f"missing or duplicate aggregate proof line: {line}")

    expected_targets = {
        path.stem
        for path in (REPO_ROOT / "crates/ferrum-interfaces/tests").glob("*.rs")
    }
    observed_targets = set(
        re.findall(r"Running tests/([A-Za-z0-9_]+)\.rs", combined)
    )
    require(
        expected_targets <= observed_targets,
        f"aggregate did not execute every integration target: {sorted(expected_targets - observed_targets)}",
    )
    summaries = [
        int(match.group(1))
        for match in re.finditer(
            r"test result: ok\. ([0-9]+) passed; 0 failed; [0-9]+ ignored;",
            combined,
        )
    ]
    require(summaries and sum(summaries) >= 100, "aggregate test summary is unexpectedly empty")
    return {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s0a_compile_unit_trybuild_evidence",
        "command": command,
        "cargo_command": command[-10:],
        "cwd": str(REPO_ROOT),
        "env_overrides": {
            "CARGO_BUILD_JOBS": "4",
            "RUST_TEST_THREADS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration,
        "returncode": result.returncode,
        "receipt": {
            "path": receipt_path.relative_to(checkpoint_root).as_posix(),
            "sha256": sha256(receipt_path),
            "limits": receipt["limits"],
            "peaks": receipt["peaks"],
            "cleanup": receipt["cleanup"],
        },
        "logs": {
            "stdout": {
                "path": stdout_path.relative_to(checkpoint_root).as_posix(),
                "sha256": sha256(stdout_path),
                "size_bytes": stdout_path.stat().st_size,
            },
            "stderr": {
                "path": stderr_path.relative_to(checkpoint_root).as_posix(),
                "sha256": sha256(stderr_path),
                "size_bytes": stderr_path.stat().st_size,
            },
        },
        "tests": {
            "expected_integration_targets": sorted(expected_targets),
            "observed_integration_targets": sorted(observed_targets),
            "test_result_summary_count": len(summaries),
            "reported_passed_test_sum": sum(summaries),
            "machine_proof_lines": proof_lines,
            "machine_proof_line_count": len(proof_lines),
        },
        "status": "pass",
    }


def artifact_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "role": "bounded-log" if "logs" in path.relative_to(root).parts else "contract-split-evidence",
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def build_gate(g00f_path: Path, output_root: Path) -> str:
    source = clean_source()
    output = output_root.resolve()
    require(REPO_ROOT not in output.parents and output != REPO_ROOT, "S0A output must be outside the source tree")
    checkpoint_root = output / "g01a-contract-split"
    require(not checkpoint_root.exists(), f"S0A checkpoint output already exists: {checkpoint_root}")
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    started_at = iso_now()
    started = time.monotonic()
    try:
        g00f = bind_g00f(g00f_path, source)
        split_inventory = collect_split_inventory(source)
        write_json(checkpoint_root / "split-inventory.json", split_inventory)
        write_json(checkpoint_root / "contract-map.json", contract_map(split_inventory))
        require(INVENTORY_DOCUMENT.is_file(), "required pre-move inventory document is missing")
        require(ADR_DOCUMENT.is_file(), "S0A ADR/source map is missing")
        (checkpoint_root / "adr.md").write_bytes(ADR_DOCUMENT.read_bytes())
        owner_evidence = run_public_owner_map(checkpoint_root)
        compile_evidence = run_bounded_aggregate(checkpoint_root)
        write_json(checkpoint_root / "compile-unit-trybuild.json", compile_evidence)
        pass_line = f"{PASS_PREFIX}: {output}"
        rows = artifact_rows(checkpoint_root)
        required_core = {
            "adr.md",
            "contract-map.json",
            "public-owner-map.json",
            "split-inventory.json",
            "compile-unit-trybuild.json",
        }
        require(required_core <= {row["path"] for row in rows}, "S0A core artifact set is incomplete")
        manifest = {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_g01a_contract_split_manifest",
            "checkpoint_id": "G01A-S0A",
            "lane": "runtime-vnext-g01a-contract-split",
            "status": "pass",
            "canonical": True,
            "artifact_dir": str(checkpoint_root),
            "output_root": str(output),
            "source": source,
            "baseline_commit": BASELINE_COMMIT,
            "g00f": g00f,
            "inventory_document": {
                "path": INVENTORY_DOCUMENT.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(INVENTORY_DOCUMENT),
            },
            "adr_source": {
                "path": ADR_DOCUMENT.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(ADR_DOCUMENT),
            },
            "public_owner_evidence": owner_evidence,
            "compile_evidence": {
                "path": "compile-unit-trybuild.json",
                "sha256": sha256(checkpoint_root / "compile-unit-trybuild.json"),
                "reported_passed_test_sum": compile_evidence["tests"]["reported_passed_test_sum"],
                "machine_proof_line_count": compile_evidence["tests"]["machine_proof_line_count"],
            },
            "artifact_count": len(rows),
            "artifact_index": rows,
            "unlocks": ["G01B", "S1"],
            "does_not_prove": [
                "G01",
                "G01B",
                "model_migration",
                "performance",
                "production_wiring",
                "release",
            ],
            "started_at": started_at,
            "finished_at": iso_now(),
            "duration_seconds": time.monotonic() - started,
            "pass_line": pass_line,
        }
        write_json(checkpoint_root / "manifest.json", manifest)
        return pass_line
    except Exception as error:
        write_json(
            checkpoint_root / "failure.json",
            {
                "schema_version": 1,
                "artifact_type": "runtime_vnext_g01a_contract_split_failure",
                "source": source,
                "started_at": started_at,
                "finished_at": iso_now(),
                "duration_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def self_test() -> int:
    require(sum(len(targets) for targets in TEST_TARGET_GROUPS.values()) == 24, "S0A target matrix drifted")
    require(len(SHARED_TEST_SUPPORT) == 10, "S0A shared test support matrix drifted")
    require(set(PRODUCTION_GROUPS) == {"resource", "execution", "event"}, "S0A production scope drifted")
    lines = expected_machine_proof_lines()
    require(len(lines) == len(set(lines)) and len(lines) >= 20, "S0A machine proof matrix drifted")
    print("FERRUM RUNTIME VNEXT G01A CONTRACT SPLIT SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g00f", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.g00f is None or args.out is None:
        parser.error("--g00f and --out are required")
    try:
        print(build_gate(args.g00f, args.out))
        return 0
    except GateError as error:
        print(f"{PASS_PREFIX} FAIL: {args.out}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
