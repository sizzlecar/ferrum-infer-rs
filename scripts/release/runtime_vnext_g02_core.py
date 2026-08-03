#!/usr/bin/env python3
"""Collect and validate the S2-scoped G02 L0/L1 source evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BOUNDED_COMMAND = REPO_ROOT / "scripts/release/bounded_command.py"
GOAL_DOC = (
    "docs/goals/runtime-vnext-0.8.0-2026-07-10/"
    "EXECUTION_STRATEGY_AMENDMENT_2026-07-14.md"
)
PASS_PREFIX = "FERRUM RUNTIME VNEXT G02 CORE L0 L1 PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT G02 CORE L0 L1 FAIL"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G02 CORE L0 L1 SELFTEST PASS"
SCHEMA = "ferrum.runtime-vnext-g02-core.v1"
RECEIPT_SCHEMA = "ferrum.bounded-command-receipt.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SUMMARY_RE = re.compile(
    r"test result: ok\. (?P<passed>\d+) passed; (?P<failed>\d+) failed; "
    r"(?P<ignored>\d+) ignored; (?P<measured>\d+) measured; "
    r"(?P<filtered>\d+) filtered out;"
)
TEST_NAME_RE = re.compile(r"^test (?P<name>[^ ]+) \.\.\.", re.MULTILINE)
L1_PROOF_RE = re.compile(
    r"FERRUM RUNTIME VNEXT G02 L1 TEST PASS: "
    r"weight_sha256=(?P<weight>[0-9a-f]{64}) "
    r"output_sha256=(?P<output>[0-9a-f]{64}) "
    r"allocations=(?P<allocations>\d+) "
    r"released_static_resources=(?P<released_static_resources>\d+) "
    r"live_allocations_after_close=(?P<live_allocations_after_close>\d+) "
    r"submissions=(?P<submissions>\d+) commands=(?P<commands>\d+)"
)
L0_WARM_LIMIT_SECONDS = 60.0
L1_WARM_LIMIT_SECONDS = 300.0
PROCESS_LIMIT = 16
GROUP_THREAD_LIMIT = 64
PER_PROCESS_THREAD_LIMIT = 16


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class TestSpec:
    name: str
    command: tuple[str, ...]
    expected_tests: frozenset[str]
    layer: str


L0_TESTS = (
    TestSpec(
        "reference-runtime-negative",
        (
            "cargo",
            "test",
            "-p",
            "ferrum-kernels",
            "--lib",
            "backend::reference::runtime::tests",
            "--",
            "--test-threads=1",
            "--nocapture",
        ),
        frozenset(
            {
                "backend::reference::runtime::tests::aligned_reference_storage_fulfills_descriptor_contract",
                "backend::reference::runtime::tests::descriptor_cannot_overclaim_device_class_or_capabilities",
                "backend::reference::runtime::tests::foreign_stream_state_fails_closed",
                "backend::reference::runtime::tests::readback_honors_host_offset_and_element_type",
                "backend::reference::runtime::tests::submission_requirements_fail_closed_before_reference_execution",
            }
        ),
        "l0",
    ),
    TestSpec(
        "program-plan",
        (
            "cargo",
            "test",
            "-p",
            "ferrum-interfaces",
            "--test",
            "vnext_program_plan_compiler_contract_tests",
            "--",
            "--test-threads=1",
            "--nocapture",
        ),
        frozenset(
            {
                "approximate_materializer_is_not_authorized_by_capability_registration",
                "compilation_rejects_missing_or_guessed_product_input_capacity",
                "compilation_reports_the_exact_tensor_binding_on_signature_mismatch",
                "completion_retention_binds_one_typed_output_and_requires_expected_wire_policy",
                "completion_retention_rejects_inputs_weights_and_unknown_values_before_planning",
                "dense_binding_in_mixed_checkpoint_does_not_require_the_container_format",
                "determinism_compile_option_retains_every_operation_output",
                "materializer_cannot_change_the_prepared_logical_weight_contract",
                "quantized_binding_still_requires_its_format_and_abi_before_allocation",
                "semantic_program_compiles_through_the_registered_provider_authority",
                "trusted_materializer_changes_physical_plan_memory_and_wire_requires_its_witness",
                "weight_arena_reaches_provider_alignment_fixed_point",
            }
        ),
        "l0",
    ),
    TestSpec(
        "operation-oracle",
        (
            "cargo",
            "test",
            "-p",
            "ferrum-interfaces",
            "--test",
            "vnext_oracle_contract_tests",
            "--",
            "--test-threads=1",
            "--nocapture",
        ),
        frozenset(
            {
                "descriptor_and_request_result_wire_require_explicit_revalidation",
                "exact_absolute_and_relative_comparison_are_fail_closed",
                "external_trait_object_and_registry_bound_handle_invoke",
                "host_tensor_rejects_noncanonical_nonfinite_and_overflowing_inputs",
                "independently_anchored_descriptor_rejects_impostor_and_registry_never_accepts_call_oracle",
                "operation_oracle_contract_proof_line",
                "reference_operation_chain_resolves_to_one_terminal_oracle",
                "registry_rejects_missing_duplicate_contract_signature_and_fingerprint_mismatches",
                "request_result_count_and_attribute_bounds_are_enforced",
            }
        ),
        "l0",
    ),
    TestSpec(
        "planning-resource",
        (
            "cargo",
            "test",
            "-p",
            "ferrum-interfaces",
            "--test",
            "vnext_planning_resource_contract_tests",
            "--",
            "--test-threads=1",
            "--nocapture",
        ),
        frozenset(
            {
                "attention_provider_policy_is_sealed_into_runtime_fingerprint",
                "execution_memory_is_core_owned_and_exact",
                "maximum_active_sequence_ceiling_is_nonzero_and_o_graph",
                "minimum_runnable_sums_lifetime_minima_and_sequential_invocation_peak",
                "operation_resource_contract_requires_explicit_presence_and_alignment",
                "provider_formula_is_policy_invariant_and_core_binds_token_ceiling",
                "provider_workspace_formulas_are_actual_shape_checked_and_wire_closed",
                "reusable_execution_workspace_is_core_derived_plan_data",
                "runtime_capacity_reserve_and_concurrency_are_typed_planning_inputs",
                "state_capacity_demand_is_explicit_checked_and_wire_closed",
                "theoretical_ceiling_over_u64_is_canonical_evidence_not_capacity_policy",
            }
        ),
        "l0",
    ),
    TestSpec(
        "resource-transaction",
        (
            "cargo",
            "test",
            "-p",
            "ferrum-interfaces",
            "--test",
            "vnext_resource_transaction_lifecycle_tests",
            "--",
            "--test-threads=1",
            "--nocapture",
        ),
        frozenset({"transaction_lifecycle_contracts_are_exhaustive"}),
        "l0",
    ),
)

L1_TEST = TestSpec(
    "tiny-real-weights",
    (
        "cargo",
        "test",
        "-p",
        "ferrum-models",
        "--test",
        "vnext_l1_reference_runtime",
        "tiny_real_safetensors_executes_through_reference_vnext_runtime",
        "--",
        "--test-threads=1",
        "--nocapture",
    ),
    frozenset({"tiny_real_safetensors_executes_through_reference_vnext_runtime"}),
    "l1",
)

WARMUP_COMMANDS = (
    (
        "cargo",
        "test",
        "-p",
        "ferrum-kernels",
        "--lib",
        "--no-run",
    ),
    (
        "cargo",
        "test",
        "-p",
        "ferrum-interfaces",
        "--test",
        "vnext_program_plan_compiler_contract_tests",
        "--test",
        "vnext_oracle_contract_tests",
        "--test",
        "vnext_planning_resource_contract_tests",
        "--test",
        "vnext_resource_transaction_lifecycle_tests",
        "--no-run",
    ),
    (
        "cargo",
        "test",
        "-p",
        "ferrum-models",
        "--test",
        "vnext_l1_reference_runtime",
        "--no-run",
    ),
)

AUDITED_SOURCE_PATHS = (
    "crates/ferrum-kernels/src/backend/mod.rs",
    "crates/ferrum-kernels/src/backend/reference/mod.rs",
    "crates/ferrum-kernels/src/backend/reference/composition.rs",
    "crates/ferrum-kernels/src/backend/reference/dense_linear.rs",
    "crates/ferrum-kernels/src/backend/reference/runtime.rs",
    "crates/ferrum-models/tests/vnext_l1_reference_runtime.rs",
)
FORBIDDEN_CRITICAL_PATH_MARKERS = ("MockKv", "StubLlm", "MockTensor")
REQUIRED_L1_MARKERS = (
    "SafetensorsArchive::open",
    "serialize_to_file",
    "ReferenceVNextComposition::create",
    "RuntimeResourceDriver::new",
    "ResourceTransaction::begin",
    "OperationDispatch::encode_and_submit_wave_with_inputs",
    "PlanRuntimeResources::close",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"invalid JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.preloadindex=false", "-c", "index.threads=1", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(proc.returncode == 0, f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def clean_source_identity() -> dict[str, Any]:
    status = [line for line in git("status", "--short").splitlines() if line]
    require(not status, f"G02 core requires a clean checkout: {status}")
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_tree_sha": git("rev-parse", "HEAD^{tree}"),
        "dirty": False,
        "status_short": [],
    }


def source_locks() -> list[dict[str, Any]]:
    rows = []
    for relative in (*AUDITED_SOURCE_PATHS, "scripts/release/runtime_vnext_g02_core.py"):
        path = REPO_ROOT / relative
        rows.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def audit_critical_path() -> dict[str, Any]:
    texts = {
        relative: (REPO_ROOT / relative).read_text(encoding="utf-8")
        for relative in AUDITED_SOURCE_PATHS
    }
    counts = {
        marker: sum(text.count(marker) for text in texts.values())
        for marker in FORBIDDEN_CRITICAL_PATH_MARKERS
    }
    require(all(count == 0 for count in counts.values()), f"mock marker found: {counts}")
    l1_text = texts["crates/ferrum-models/tests/vnext_l1_reference_runtime.rs"]
    missing = [marker for marker in REQUIRED_L1_MARKERS if marker not in l1_text]
    require(not missing, f"L1 real-path markers are missing: {missing}")
    return {
        "audited_paths": list(AUDITED_SOURCE_PATHS),
        "forbidden_marker_counts": counts,
        "required_l1_markers": list(REQUIRED_L1_MARKERS),
        "missing_required_l1_markers": [],
    }


def child_environment() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("FERRUM_")}
    env.update(
        {
            "CARGO_BUILD_JOBS": "4",
            "RUST_TEST_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def validate_receipt(
    receipt_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    expected_command: tuple[str, ...],
    wall_timeout_seconds: float,
) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    require(receipt.get("schema") == RECEIPT_SCHEMA, "bounded receipt schema mismatch")
    require(receipt.get("command") == list(expected_command), "bounded receipt command mismatch")
    require(Path(str(receipt.get("cwd"))).resolve() == REPO_ROOT, "bounded receipt cwd mismatch")
    require(receipt.get("status") == "pass", "bounded receipt status is not pass")
    require(receipt.get("rc") == 0, "bounded receipt rc is not zero")
    require(receipt.get("reason") == "command_completed", "bounded receipt reason mismatch")
    require(receipt.get("violation") is None, "bounded receipt reports a limit violation")
    require(receipt.get("sampling_error_count") == 0, "bounded receipt has sampling errors")
    cleanup = receipt.get("cleanup")
    require(isinstance(cleanup, dict) and cleanup.get("process_group_gone") is True,
            "bounded process group cleanup failed")
    limits = receipt.get("limits")
    require(isinstance(limits, dict), "bounded receipt limits are missing")
    require(limits.get("max_processes") == PROCESS_LIMIT, "process limit mismatch")
    require(limits.get("max_group_threads") == GROUP_THREAD_LIMIT, "group thread limit mismatch")
    require(limits.get("max_per_process_threads") == PER_PROCESS_THREAD_LIMIT,
            "per-process thread limit mismatch")
    require(float(limits.get("wall_timeout_seconds", -1)) == wall_timeout_seconds,
            "wall timeout mismatch")
    peaks = receipt.get("peaks")
    require(isinstance(peaks, dict), "bounded receipt peaks are missing")
    require(0 <= int(peaks.get("processes", -1)) <= PROCESS_LIMIT, "process peak exceeds limit")
    require(0 <= int(peaks.get("group_threads", -1)) <= GROUP_THREAD_LIMIT,
            "group thread peak exceeds limit")
    require(0 <= int(peaks.get("per_process_threads", -1)) <= PER_PROCESS_THREAD_LIMIT,
            "per-process thread peak exceeds limit")
    duration = receipt.get("duration_seconds")
    require(isinstance(duration, (int, float)) and math.isfinite(duration) and 0 <= duration,
            "bounded duration is invalid")
    for label, expected_path in (("stdout", stdout_path), ("stderr", stderr_path)):
        row = receipt.get(label)
        require(isinstance(row, dict), f"bounded {label} evidence is missing")
        require(Path(str(row.get("path"))).resolve() == expected_path.resolve(),
                f"bounded {label} path mismatch")
        require(row.get("sha256") == sha256(expected_path), f"bounded {label} SHA mismatch")
        require(row.get("size_bytes") == expected_path.stat().st_size,
                f"bounded {label} size mismatch")
    return receipt


def validate_test_output(text: str, spec: TestSpec) -> dict[str, Any]:
    summaries = list(SUMMARY_RE.finditer(text))
    require(len(summaries) == 1, f"{spec.name} must contain one test summary")
    summary = {key: int(value) for key, value in summaries[0].groupdict().items()}
    require(summary["passed"] == len(spec.expected_tests), f"{spec.name} passed count mismatch")
    require(summary["failed"] == 0 and summary["ignored"] == 0 and summary["measured"] == 0,
            f"{spec.name} has failed, ignored, or measured tests")
    observed = frozenset(match.group("name") for match in TEST_NAME_RE.finditer(text))
    require(observed == spec.expected_tests,
            f"{spec.name} test set mismatch: missing={sorted(spec.expected_tests - observed)} "
            f"extra={sorted(observed - spec.expected_tests)}")
    return {"summary": summary, "tests": sorted(observed)}


def run_bounded(
    artifact_root: Path,
    name: str,
    command: tuple[str, ...],
    wall_timeout_seconds: float,
) -> dict[str, Any]:
    command_root = artifact_root / "commands" / name
    command_root.mkdir(parents=True, exist_ok=False)
    receipt_path = command_root / "receipt.json"
    stdout_path = command_root / "stdout.log"
    stderr_path = command_root / "stderr.log"
    driver_stdout = command_root / "bounded-driver.stdout.log"
    driver_stderr = command_root / "bounded-driver.stderr.log"
    bounded = [
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
        str(int(wall_timeout_seconds)),
        "--max-processes",
        str(PROCESS_LIMIT),
        "--max-group-threads",
        str(GROUP_THREAD_LIMIT),
        "--max-per-process-threads",
        str(PER_PROCESS_THREAD_LIMIT),
        "--sample-interval-seconds",
        "1",
        "--",
        *command,
    ]
    proc = subprocess.run(
        bounded,
        cwd=REPO_ROOT,
        env=child_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    driver_stdout.write_text(proc.stdout, encoding="utf-8")
    driver_stderr.write_text(proc.stderr, encoding="utf-8")
    require(proc.returncode == 0, f"{name} bounded command failed rc={proc.returncode}")
    receipt = validate_receipt(
        receipt_path,
        stdout_path,
        stderr_path,
        command,
        wall_timeout_seconds,
    )
    return {
        "name": name,
        "command": list(command),
        "receipt": str(receipt_path.relative_to(artifact_root)),
        "receipt_sha256": sha256(receipt_path),
        "stdout": str(stdout_path.relative_to(artifact_root)),
        "stdout_sha256": sha256(stdout_path),
        "stderr": str(stderr_path.relative_to(artifact_root)),
        "stderr_sha256": sha256(stderr_path),
        "duration_seconds": receipt["duration_seconds"],
        "peaks": receipt["peaks"],
    }


def collect(out: Path) -> dict[str, Any]:
    out = out.resolve()
    try:
        out.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise GateError("G02 core artifact root must be outside the source checkout")
    source = clean_source_identity()
    require(not out.exists(), f"output already exists: {out}")
    out.mkdir(parents=True)
    audit = audit_critical_path()
    warmups = [
        run_bounded(out, f"warmup-{index}", command, L1_WARM_LIMIT_SECONDS)
        for index, command in enumerate(WARMUP_COMMANDS, start=1)
    ]
    l0_rows = []
    for spec in L0_TESTS:
        row = run_bounded(out, spec.name, spec.command, L0_WARM_LIMIT_SECONDS)
        output = (out / row["stdout"]).read_text(encoding="utf-8")
        row["test_evidence"] = validate_test_output(output, spec)
        l0_rows.append(row)
    l0_duration = sum(float(row["duration_seconds"]) for row in l0_rows)
    require(l0_duration <= L0_WARM_LIMIT_SECONDS,
            f"L0 warm duration {l0_duration:.3f}s exceeds {L0_WARM_LIMIT_SECONDS:.0f}s")
    l1_row = run_bounded(out, L1_TEST.name, L1_TEST.command, L1_WARM_LIMIT_SECONDS)
    l1_output = (out / l1_row["stdout"]).read_text(encoding="utf-8")
    l1_row["test_evidence"] = validate_test_output(l1_output, L1_TEST)
    proofs = list(L1_PROOF_RE.finditer(l1_output))
    require(len(proofs) == 1, "L1 output must contain exactly one proof line")
    proof = proofs[0].groupdict()
    counters = {
        name: int(proof[name])
        for name in (
            "allocations",
            "released_static_resources",
            "live_allocations_after_close",
            "submissions",
            "commands",
        )
    }
    require(counters["allocations"] >= 2 and counters["submissions"] >= 2
            and counters["commands"] >= 2, "L1 runtime counters do not prove execution")
    require(counters["released_static_resources"] == 1,
            "L1 did not release its exact static weight allocation")
    require(counters["live_allocations_after_close"] == 0,
            "L1 retained a reference runtime allocation after close")
    l1_row["numerical_proof"] = {
        "weight_sha256": proof["weight"],
        "output_sha256": proof["output"],
        **counters,
    }
    require(float(l1_row["duration_seconds"]) <= L1_WARM_LIMIT_SECONDS,
            "L1 warm duration exceeds five minutes")
    pass_line = f"{PASS_PREFIX}: {out}"
    manifest = {
        "schema": SCHEMA,
        "status": "pass",
        "lane": "runtime-vnext-g02-core-l0-l1",
        "goal_doc": GOAL_DOC,
        "created_at": iso_now(),
        "source": source,
        "source_locks": source_locks(),
        "bounded_command": {
            "path": str(BOUNDED_COMMAND.relative_to(REPO_ROOT)),
            "sha256": sha256(BOUNDED_COMMAND),
        },
        "warmups": warmups,
        "l0": {
            "warm_limit_seconds": L0_WARM_LIMIT_SECONDS,
            "duration_seconds": l0_duration,
            "test_count": sum(len(spec.expected_tests) for spec in L0_TESTS),
            "commands": l0_rows,
        },
        "l1": {
            "warm_limit_seconds": L1_WARM_LIMIT_SECONDS,
            **l1_row,
        },
        "critical_path_audit": audit,
        "full_g02_claimed": False,
        "pass_line": pass_line,
    }
    write_json(out / "manifest.json", manifest)
    validate_artifact(out)
    return manifest


def validate_artifact(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = read_json(root / "manifest.json")
    require(manifest.get("schema") == SCHEMA, "G02 core manifest schema mismatch")
    require(manifest.get("status") == "pass", "G02 core manifest status mismatch")
    require(manifest.get("full_g02_claimed") is False, "S2 core must not claim full G02")
    expected_pass = f"{PASS_PREFIX}: {root}"
    require(manifest.get("pass_line") == expected_pass, "G02 core PASS line mismatch")
    source = manifest.get("source")
    require(isinstance(source, dict) and source.get("dirty") is False,
            "G02 core source identity is missing or dirty")
    require(source.get("git_sha") == git("rev-parse", "HEAD"), "G02 core git SHA is stale")
    require(source.get("git_tree_sha") == git("rev-parse", "HEAD^{tree}"),
            "G02 core git tree is stale")
    locks = manifest.get("source_locks")
    require(isinstance(locks, list) and locks, "G02 core source locks are missing")
    for row in locks:
        require(isinstance(row, dict), "G02 core source lock is malformed")
        path = REPO_ROOT / str(row.get("path"))
        require(row.get("sha256") == sha256(path), f"G02 core source lock is stale: {path}")
        require(row.get("size_bytes") == path.stat().st_size, f"G02 core source size is stale: {path}")
    bounded = manifest.get("bounded_command")
    require(isinstance(bounded, dict), "G02 core bounded command lock is missing")
    require(bounded.get("sha256") == sha256(BOUNDED_COMMAND), "bounded command lock is stale")
    audit = manifest.get("critical_path_audit")
    require(isinstance(audit, dict), "G02 core critical-path audit is missing")
    require(audit == audit_critical_path(), "G02 core critical-path audit is stale")
    warmups = manifest.get("warmups")
    require(isinstance(warmups, list) and len(warmups) == len(WARMUP_COMMANDS),
            "G02 core warmup matrix mismatch")
    l0 = manifest.get("l0")
    require(isinstance(l0, dict), "G02 core L0 evidence is missing")
    l0_rows = l0.get("commands")
    require(isinstance(l0_rows, list) and len(l0_rows) == len(L0_TESTS),
            "G02 core L0 command matrix mismatch")
    require(l0.get("test_count") == sum(len(spec.expected_tests) for spec in L0_TESTS),
            "G02 core L0 test count mismatch")
    require(float(l0.get("duration_seconds", math.inf)) <= L0_WARM_LIMIT_SECONDS,
            "G02 core L0 timing exceeds target")
    l1 = manifest.get("l1")
    require(isinstance(l1, dict), "G02 core L1 evidence is missing")
    require(float(l1.get("duration_seconds", math.inf)) <= L1_WARM_LIMIT_SECONDS,
            "G02 core L1 timing exceeds target")
    command_rows = [
        *((row, command, L1_WARM_LIMIT_SECONDS)
          for row, command in zip(warmups, WARMUP_COMMANDS, strict=True)),
        *((row, spec.command, L0_WARM_LIMIT_SECONDS)
          for row, spec in zip(l0_rows, L0_TESTS, strict=True)),
        (l1, L1_TEST.command, L1_WARM_LIMIT_SECONDS),
    ]
    for row, expected_command, timeout in command_rows:
        require(isinstance(row, dict) and row.get("command") == list(expected_command),
                "G02 core persisted command mismatch")
        receipt = root / str(row.get("receipt"))
        stdout = root / str(row.get("stdout"))
        stderr = root / str(row.get("stderr"))
        validate_receipt(receipt, stdout, stderr, expected_command, timeout)
        require(row.get("receipt_sha256") == sha256(receipt), "G02 core receipt SHA mismatch")
        require(row.get("stdout_sha256") == sha256(stdout), "G02 core stdout SHA mismatch")
        require(row.get("stderr_sha256") == sha256(stderr), "G02 core stderr SHA mismatch")
    for row, spec in zip(l0_rows, L0_TESTS, strict=True):
        observed = validate_test_output((root / row["stdout"]).read_text(encoding="utf-8"), spec)
        require(row.get("test_evidence") == observed, f"{spec.name} persisted evidence mismatch")
    l1_observed = validate_test_output((root / l1["stdout"]).read_text(encoding="utf-8"), L1_TEST)
    require(l1.get("test_evidence") == l1_observed, "L1 persisted test evidence mismatch")
    proof_matches = list(L1_PROOF_RE.finditer((root / l1["stdout"]).read_text(encoding="utf-8")))
    require(len(proof_matches) == 1, "L1 persisted numerical proof is missing")
    proof = proof_matches[0].groupdict()
    expected_proof = {
        "weight_sha256": proof["weight"],
        "output_sha256": proof["output"],
        "allocations": int(proof["allocations"]),
        "released_static_resources": int(proof["released_static_resources"]),
        "live_allocations_after_close": int(proof["live_allocations_after_close"]),
        "submissions": int(proof["submissions"]),
        "commands": int(proof["commands"]),
    }
    require(l1.get("numerical_proof") == expected_proof, "L1 numerical proof mismatch")
    return manifest


def self_test() -> int:
    sample = """
running 1 test
test tiny_real_safetensors_executes_through_reference_vnext_runtime ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;
"""
    validate_test_output(sample, L1_TEST)
    try:
        validate_test_output(sample.replace("1 passed", "0 passed"), L1_TEST)
    except GateError as error:
        require("passed count" in str(error), f"unexpected count rejection: {error}")
    else:
        raise AssertionError("G02 core self-test accepted a forged test count")
    proof = (
        "FERRUM RUNTIME VNEXT G02 L1 TEST PASS: "
        f"weight_sha256={'a' * 64} output_sha256={'b' * 64} "
        "allocations=2 released_static_resources=1 live_allocations_after_close=0 "
        "submissions=2 commands=6"
    )
    require(L1_PROOF_RE.fullmatch(proof) is not None, "valid L1 proof fixture was rejected")
    require(SHA256_RE.fullmatch("a" * 64) is not None, "SHA fixture was rejected")
    with tempfile.TemporaryDirectory(prefix="ferrum-g02-core-selftest-") as temporary:
        path = Path(temporary) / "payload"
        path.write_bytes(b"reference")
        require(sha256(path) == hashlib.sha256(b"reference").hexdigest(), "SHA helper drift")
    audit = audit_critical_path()
    require(all(value == 0 for value in audit["forbidden_marker_counts"].values()),
            "critical-path audit did not prove zero mocks")
    print(SELFTEST_PASS_LINE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--validate-only", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    try:
        if args.validate_only is not None:
            manifest = validate_artifact(args.validate_only)
            print(manifest["pass_line"])
            return 0
        require(args.out is not None, "--out is required")
        manifest = collect(args.out)
        print(manifest["pass_line"])
        return 0
    except GateError as error:
        target = (args.validate_only or args.out or Path("<missing-out>")).resolve()
        print(f"{FAIL_PREFIX}: {target}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
