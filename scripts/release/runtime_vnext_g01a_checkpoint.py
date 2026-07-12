#!/usr/bin/env python3
"""Validate and freeze the Runtime vNext G01A pure-contract checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
RUN_GATE_PATH = REPO_ROOT / "scripts/release/run_gate.py"
BOUNDED_COMMAND_PATH = REPO_ROOT / "scripts/release/bounded_command.py"
GOAL_ROOT = REPO_ROOT / "docs/goals/runtime-vnext-0.8.0-2026-07-10"
GOAL_PATH = GOAL_ROOT / "G01_CORE_CONTRACTS.md"
ADR_PATH = GOAL_ROOT / "G01A_CONTRACT_ADR.md"
MAP_PATH = GOAL_ROOT / "G01A_LEGACY_CONTRACT_MAP.json"
INTERFACES_ROOT = REPO_ROOT / "crates/ferrum-interfaces"
CHECKPOINT_SUBDIR = "g01a-contract-checkpoint"
CHECKPOINT_ID = "G01A"
SCHEMA_VERSION = 1
PASS_PREFIX = "FERRUM RUNTIME VNEXT G01A CONTRACT CHECKPOINT PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G01A CONTRACT CHECKPOINT SELFTEST PASS"

REQUIRED_CONTRACTS = (
    "DeviceRuntime",
    "OperationContract",
    "ModelFamilyProvider",
    "ModelProgram",
    "ExecutionPlanner",
    "ExecutionPlan",
    "ResourceTransaction",
    "ExecutionEventSink",
    "ResolvedModelPlan",
)
G01A_EXECUTION_IDENTITY_VERSION = (3, 0)
G01A_EVENT_REQUIRED_TRANSITION = "NodeRetired"
G01A_EVENT_FORBIDDEN_TRANSITION = "NodeCompleted"
G01A_SEMANTIC_TYPE_KINDS = {
    "BatchWorkShape": "struct",
    "ClaimedBackingTransaction": "struct",
    "ParticipantNodeKey": "struct",
    "BatchOperationIdentity": "struct",
    "DefinitelyNotSubmittedRetryAuthority": "struct",
    "BatchedOperationInvocation": "struct",
}
G01A_DNF_RETRY_AUTHORITY_TYPE = "DefinitelyNotSubmittedRetryAuthority"
G01A_MULTIPARTICIPANT_DISPATCH_MARKERS = {
    "batch_identity_parameter": "BatchOperationIdentity",
    "participant_bindings_parameter": "active_bindings",
    "participant_binding_type": "TrustedActiveSequenceBinding",
    "batched_invocation_body": "BatchedOperationInvocation",
}
REQUIRED_UNIT_TESTS = {
    "blocked_tensor_storage_requires_explicit_exact_or_zero_fill_padding",
    "blocked_weight_layout_requires_explicit_exact_or_zero_fill_padding",
    "breaking_schema_versions_are_rejected_100_of_100",
    "dynamic_descriptor_and_memory_plan_standalone_wire_are_checked",
    "execution_alias_effect_wire_mutations_are_rejected",
    "execution_alias_may_alias_supports_distinct_or_exact_storage",
    "execution_alias_must_alias_builds_exact_equivalence_and_single_allocation",
    "execution_alias_rejects_overwrite_before_last_consumer",
    "execution_alias_rejects_partial_and_wrong_input_overlap",
    "execution_memory_is_core_owned_and_exact",
    "execution_plan_is_deterministic_100_of_100",
    "execution_plan_schema_round_trip_100_of_100",
    "execution_state_effect_graph_orders_raw_war_waw",
    "execution_state_read_only_nodes_remain_independent",
    "externally_trusted_node_resolution_cannot_be_replaced_by_wire_data",
    "failure_envelope_wire_limit_precedes_deserialization",
    "forged_self_hashed_plan_is_rejected_by_semantic_rebuild",
    "generic_contracts_have_zero_architecture_names",
    "mandatory_object_safe_contracts_accept_trait_objects",
    "maximum_active_sequence_ceiling_is_nonzero_and_o_graph",
    "minimum_runnable_sums_lifetime_minima_and_sequential_invocation_peak",
    "model_program_rejects_duplicate_declared_outputs",
    "operation_resource_contract_requires_explicit_presence_and_alignment",
    "physical_weight_layout_tree_accepts_dense_fixture",
    "physical_weight_layout_tree_accepts_grouped_quantized_axis_index_fixture",
    "physical_weight_layout_tree_accepts_recursive_quantized_expert_stack_fixture",
    "physical_weight_layout_tree_rejects_invalid_shape_reuse_padding_overflow_and_limits",
    "planning_registry_missing_duplicate_and_mismatched_entries_fail_before_plan",
    "prepared_family_wire_requires_typed_registry_reconstruction",
    "preferred_provider_is_only_a_core_validated_preference",
    "provider_catalog_and_reference_oracle_fail_closed",
    "provider_implementation_fingerprint_is_plan_hashed_and_revalidated",
    "provider_raw_estimate_identity_input_and_output_are_revalidated_by_core",
    "provider_workspace_formulas_are_actual_shape_checked_and_wire_closed",
    "resolution_source_matrix_rejects_forbidden_binding_before_plan",
    "resolved_external_device_catalog_runtime_and_node_resolution_are_exact",
    "resolved_model_family_identity_is_unique_and_fail_closed",
    "resolved_model_plan_closes_all_contract_links",
    "resolved_model_plan_initial_construction_requires_verified_evidence_context",
    "resolved_source_evidence_rejects_raw_bytes_and_provenance_tampering",
    "resolved_source_parser_identity_and_determinism_are_enforced",
    "runtime_capacity_reserve_and_concurrency_are_typed_planning_inputs",
    "self_consistent_wire_provider_selection_is_rejected",
    "self_consistent_wire_resource_estimate_and_memory_mutation_is_rejected",
    "silent_success_defaults_are_absent",
    "state_capacity_demand_is_explicit_checked_and_wire_closed",
    "storage_incompatible_preference_falls_back_with_canonical_evidence",
    "theoretical_ceiling_over_u64_is_canonical_evidence_not_capacity_policy",
    "typed_planning_registry_invokes_real_contract_and_estimator_once",
    "unknown_inputs_fail_closed",
    "weight_schema_order_is_normalized_before_fingerprinting",
}
REQUIRED_RESOURCE_TESTS = {
    "closing_root_rejects_every_parent_to_child_derivation",
    "plan_runtime_close_recovery_is_ownership_safe",
    "poisoned_bound_stream_retains_sequence_until_stream_drop",
    "resource_capacity_concurrency_is_bounded",
    "resource_transaction_abandon_panic_child",
    "resource_transaction_contract_is_exhaustive",
    "sequence_owner_drop_defers_blocking_backend_recovery",
}
REQUIRED_EVENT_TESTS = {"vnext_event_replay_v5_contract"}
REQUIRED_RESOLUTION_LIMITS_TESTS = {
    "field_path_count_and_total_bytes_are_bounded_before_parser",
    "json_parser_checks_source_bytes_when_called_directly",
    "json_parser_preflight_enforces_node_budget_before_building_a_value_tree",
    "json_parser_preflight_handles_escaped_structure_and_rejects_trailing_roots",
    "json_parser_preflight_reports_depth_budget_before_serde_recursion_failure",
    "parsed_json_depth_node_and_text_budgets_are_enforced_for_each_result",
    "parser_descriptor_deserialization_and_each_verification_read_fail_closed",
    "provenance_limit_accepts_max_and_rejects_max_plus_one_before_parser",
    "public_resolution_availability_limits_are_stable",
    "repeated_parser_results_must_be_canonically_deterministic",
    "resolved_wire_limit_accepts_max_and_rejects_max_plus_one_before_serde",
    "source_byte_limit_accepts_max_and_rejects_max_plus_one_before_parser",
}
REQUIRED_DEVICE_OPERATION_TESTS = {
    "completion_reaper_drop_defers_blocking_backend_recovery",
    "device_and_operation_contract_is_exhaustive",
}
REQUIRED_ORACLE_TESTS = {
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
REQUIRED_MODEL_WIRE_TESTS = {
    "prepared_family_wire_accepts_max_and_rejects_max_plus_one_before_serde",
    "prepared_family_wire_rejects_unknown_fields_and_typed_drift",
    "prepared_family_wire_round_trip_requires_external_typed_registry",
    "prepared_model_family_wire_proof_line",
    "typed_family_config_and_registry_identity_fail_closed",
    "typed_config_is_serialized_once_and_signed_external_identity_is_replayed",
}
REQUIRED_COMPILE_TESTS = {"vnext_compile"}
REQUIRED_LEGACY_TESTS = {"legacy_backend_methods_are_mapped_82_of_82"}
REQUIRED_TESTS_BY_TARGET = {
    "vnext_contract_tests": REQUIRED_UNIT_TESTS,
    "vnext_resource_contract_tests": REQUIRED_RESOURCE_TESTS,
    "vnext_event_contract_tests": REQUIRED_EVENT_TESTS,
    "vnext_resolution_limits_contract_tests": REQUIRED_RESOLUTION_LIMITS_TESTS,
    "vnext_device_operation_contract_tests": REQUIRED_DEVICE_OPERATION_TESTS,
    "vnext_oracle_contract_tests": REQUIRED_ORACLE_TESTS,
    "vnext_model_wire_contract_tests": REQUIRED_MODEL_WIRE_TESTS,
    "vnext_compile": REQUIRED_COMPILE_TESTS,
    "vnext_legacy_map": REQUIRED_LEGACY_TESTS,
}
REQUIRED_ADMISSION_LIB_TESTS = {
    "vnext::admission::tests::allocator_availability_change_advances_capacity_epoch_without_recounting_units",
    "vnext::admission::tests::batch_child_claim_charges_once_and_binds_every_parent",
    "vnext::admission::tests::batch_child_defer_and_reject_have_zero_partial_parent_effect",
    "vnext::admission::tests::batch_child_rejects_duplicate_and_foreign_parents_atomically",
    "vnext::admission::tests::batch_child_rejects_stale_later_parent_without_partial_effect",
    "vnext::admission::tests::batch_child_unwind_releases_all_parent_edges",
    "vnext::admission::tests::child_claim_is_counted_in_future_request_epoch_headroom",
    "vnext::admission::tests::child_claim_rejects_foreign_parent_and_epoch_exhaustion_atomically",
    "vnext::admission::tests::child_claim_rejects_stale_sequence_generation_without_side_effect",
    "vnext::admission::tests::child_claim_uses_parent_authority_without_consuming_sequence_slot",
    "vnext::admission::tests::child_multi_domain_defer_and_reject_have_zero_partial_effect",
    "vnext::admission::tests::child_unwind_releases_without_poisoning_parent",
    "vnext::admission::tests::concurrent_admission_never_exceeds_ceiling",
    "vnext::admission::tests::concurrent_children_preserve_global_and_per_parent_counts",
    "vnext::admission::tests::early_release_of_any_batch_parent_fails_closed_and_retains_shared_claim",
    "vnext::admission::tests::early_request_release_with_live_sequence_fails_closed_and_retains_claims",
    "vnext::admission::tests::early_sequence_release_with_active_child_fails_closed",
    "vnext::admission::tests::epoch_exhaustion_rejects_admission_before_claim",
    "vnext::admission::tests::exact_fit_claims_only_immediate_and_release_retries",
    "vnext::admission::tests::growth_defer_and_permanent_reject_are_distinct",
    "vnext::admission::tests::lease_authority_is_bound_to_the_exact_coordinator",
    "vnext::admission::tests::lease_release_uses_preallocated_reuse_storage",
    "vnext::admission::tests::multi_domain_failure_has_zero_partial_effect",
    "vnext::admission::tests::multi_domain_growth_validates_all_before_one_epoch_commit",
    "vnext::admission::tests::mutation_unwind_wakes_parked_waiter_with_terminal_error",
    "vnext::admission::tests::overlapping_batch_children_track_each_parent_without_double_charging",
    "vnext::admission::tests::poisoned_child_drop_retains_capacity_and_child_counts",
    "vnext::admission::tests::poisoned_drop_retains_claim_and_is_observable",
    "vnext::admission::tests::request_authority_reuses_sparse_storage_with_new_generation",
    "vnext::admission::tests::request_capacity_is_shared_once_across_multiple_sequences",
    "vnext::admission::tests::request_defer_and_reject_are_atomic_and_do_not_consume_sequence_slots",
    "vnext::admission::tests::request_unwind_releases_request_capacity_without_poisoning_coordinator",
    "vnext::admission::tests::sequence_issue_failure_has_zero_side_effect",
    "vnext::admission::tests::sequence_unwind_releases_lease_without_poisoning_coordinator",
    "vnext::admission::tests::snapshot_counts_live_authorities_independently_from_active_counter",
    "vnext::admission::tests::sparse_sequence_ids_reuse_storage_and_change_generation",
    "vnext::admission::tests::temporary_multi_domain_shortfall_has_zero_partial_effect",
    "vnext::admission::tests::waiter_listener_closes_recheck_to_park_race",
    "vnext::admission::tests::waiter_recheck_closes_release_and_growth_races",
    "vnext::admission::tests::zero_domain_plan_still_uses_dynamic_sequence_authority",
}
REQUIRED_TEST_DRIVERS = {
    f"crates/ferrum-interfaces/tests/{target}.rs"
    for target in REQUIRED_TESTS_BY_TARGET
}
EXPECTED_RESOURCE_CASES = 311
EXPECTED_FAIL_CLOSED_CASES = 62
EXPECTED_EVENT_REPLAY_V5_CASES = 161
EXPECTED_DEVICE_OPERATION_CASES = 299
EXPECTED_ORACLE_CASES = 26
EXPECTED_MODEL_WIRE_CASES = 24
EXPECTED_MODEL_IDENTITY_CASES = 5
EXPECTED_DYNAMIC_ADMISSION_CASES = 40
EXPECTED_TRYBUILD_PASS_CASES = 2
EXPECTED_TRYBUILD_FAIL_CASES = 78
TEST_THREADS_ARG = "--test-threads=1"
BOUNDED_RECEIPT_SCHEMA = "ferrum.bounded-command-receipt.v1"
BOUNDED_TEST_COMMAND_COUNT = 20
BOUNDED_TEST_ENV_OVERRIDES = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "CARGO_BUILD_JOBS": "2",
}
# These process-group limits include cargo and rustc workers as well as the
# test binary. The regular profiles' 8-process/32-group-thread/16-per-process
# ceiling admits the observed 5-process/28-thread cold-compile peak while still
# terminating runaway test concurrency hundreds of times before 8192 threads.
BOUNDED_TEST_PROFILES = {
    "regular": {
        "wall_timeout_seconds": 120.0,
        "max_processes": 8,
        "max_group_threads": 32,
        "max_per_process_threads": 16,
        "sample_interval_seconds": 0.05,
        "max_sampling_errors": 3,
        "term_grace_seconds": 1.0,
    },
    # The no-default-features admission target may require a fresh rustc link
    # after trybuild changes the target cache. Keep that compile bounded without
    # mistaking rustc's normal internal workers for a runaway test process.
    "admission": {
        "wall_timeout_seconds": 120.0,
        "max_processes": 8,
        "max_group_threads": 32,
        "max_per_process_threads": 16,
        "sample_interval_seconds": 0.05,
        "max_sampling_errors": 3,
        "term_grace_seconds": 1.0,
    },
    "resource": {
        "wall_timeout_seconds": 60.0,
        "max_processes": 8,
        "max_group_threads": 32,
        "max_per_process_threads": 16,
        "sample_interval_seconds": 0.05,
        "max_sampling_errors": 3,
        "term_grace_seconds": 1.0,
    },
    "trybuild": {
        "wall_timeout_seconds": 300.0,
        "max_processes": 8,
        # trybuild launches its own bounded Cargo build. With
        # CARGO_BUILD_JOBS=2 the observed cold compile uses seven processes and
        # can transiently cross 32 group threads while each process remains at
        # or below 16. Keep the independent process/per-process ceilings and a
        # group ceiling that is still 128x below the historical 8192-thread
        # failure.
        "max_group_threads": 64,
        "max_per_process_threads": 16,
        "sample_interval_seconds": 0.05,
        "max_sampling_errors": 3,
        "term_grace_seconds": 1.0,
    },
}
QUALITY_COMMANDS = (
    ("cargo", "check", "-p", "ferrum-interfaces", "--no-default-features"),
    ("cargo", "fmt", "--all", "--", "--check"),
    (
        "cargo",
        "clippy",
        "-p",
        "ferrum-interfaces",
        "--all-targets",
        "--no-default-features",
        "--",
        "-A",
        "warnings",
    ),
)
MAP_CLASSIFICATIONS = {
    "stable_device_primitive",
    "versioned_operation",
    "model_semantic",
    "dead_code",
}
G00A_DOES_NOT_PROVE = {"G00", "G01B", "model_migration", "performance", "release"}
G01A_DOES_NOT_PROVE = {
    "G00",
    "G01B",
    "G01",
    "runtime_migration",
    "product_routing",
    "model_migration",
    "performance",
    "release",
}
G00A_ARTIFACTS = {
    "coupling-inventory.json",
    "generation-presets.catalog.json",
    "historical-bugs.catalog.json",
    "inventory-review.catalog.json",
    "model-facts.lock.json",
    "model-resolution.input.json",
    "model-resolution.json",
    "models.catalog.json",
}
G01A_ARTIFACTS = {"adr.md", "contract-map.json", "compile-unit-trybuild.json"}
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
GIT_BLOB_RE = re.compile(r"^[0-9a-f]{40,64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
KNOWN_ARCHITECTURE_RE = re.compile(
    r"\b(?:qwen|llama|deepseek|mistral|mixtral|gemma|phi|chatglm|internlm|baichuan|yi)"
    r"(?!eld)[a-z0-9_.-]*\b",
    re.IGNORECASE,
)
FORBIDDEN_VNEXT_CHECKS = {
    "Any/downcast",
    "backend feature cfg",
    "product/runtime import",
    "hidden environment read",
}
GIT_OVERRIDE_ENV_KEYS = {
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_DIR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_WORK_TREE",
}


class CheckpointError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a JSON array")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty string")
    return value.strip()


def require_sha256(value: Any, label: str) -> str:
    digest = require_string(value, label).lower()
    require(SHA256_RE.fullmatch(digest) is not None, f"{label} must be a lowercase SHA256")
    return digest


def require_git_sha(value: Any, label: str) -> str:
    digest = require_string(value, label).lower()
    require(GIT_SHA_RE.fullmatch(digest) is not None, f"{label} must be a full lowercase Git SHA")
    return digest


def require_exact_string_set(value: Any, expected: set[str], label: str) -> None:
    rows = require_list(value, label)
    require(all(isinstance(row, str) and row for row in rows), f"{label} must contain strings")
    require(len(rows) == len(set(rows)), f"{label} contains duplicates")
    require(set(rows) == expected, f"{label} mismatch: expected={sorted(expected)} actual={sorted(rows)}")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CheckpointError(f"invalid {label}: {exc}") from exc
    return require_object(value, label)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return strict_json_bytes(path.read_bytes(), label)
    except OSError as exc:
        raise CheckpointError(f"cannot read {label} {path}: {exc}") from exc


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def clean_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in GIT_OVERRIDE_ENV_KEYS:
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_git(args: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        env=clean_subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise CheckpointError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def source_identity() -> dict[str, Any]:
    status = [line for line in run_git(["status", "--short"]).splitlines() if line.strip()]
    require(not status, f"G01A checkpoint requires a clean checkout: {status}")
    return {
        "git_sha": require_git_sha(run_git(["rev-parse", "HEAD"]), "current HEAD"),
        "git_tree_sha": require_git_sha(
            run_git(["rev-parse", "HEAD^{tree}"]), "current source tree"
        ),
        "dirty": False,
        "status_short": [],
    }


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def require_external_file(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    require(not is_within(resolved, REPO_ROOT), f"{label} must be outside the Git source tree")
    require(resolved.is_file() and not resolved.is_symlink(), f"{label} must be a regular non-symlink file")
    return resolved


def require_external_output(path: Path) -> tuple[Path, Path]:
    out = path.resolve()
    require(not is_within(out, REPO_ROOT), "--out must resolve outside the Git source tree")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / CHECKPOINT_SUBDIR
    require(not checkpoint.exists(), f"G01A checkpoint output must be fresh: {checkpoint}")
    return out, checkpoint


def safe_relative_path(raw: Any, label: str) -> str:
    relative = require_string(raw, label)
    path = Path(relative)
    require(not path.is_absolute(), f"{label} must be relative")
    require(path.as_posix() == relative and ".." not in path.parts and "." not in path.parts, f"{label} is not normalized")
    return relative


def validate_indexed_files(
    root: Path,
    rows_raw: Any,
    expected_paths: set[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    rows = require_list(rows_raw, label)
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = require_object(raw, f"{label}[{index}]")
        relative = safe_relative_path(row.get("path"), f"{label}[{index}].path")
        require(relative not in indexed, f"{label} contains duplicate path: {relative}")
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"{label} missing regular artifact: {relative}")
        size = row.get("size_bytes")
        require(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"{label}[{relative}].size_bytes invalid")
        require(path.stat().st_size == size, f"{label}[{relative}] size mismatch")
        require(file_sha256(path) == require_sha256(row.get("sha256"), f"{label}[{relative}].sha256"), f"{label}[{relative}] SHA256 mismatch")
        require_string(row.get("role"), f"{label}[{relative}].role")
        indexed[relative] = row
    require(set(indexed) == expected_paths, f"{label} path set mismatch: expected={sorted(expected_paths)} actual={sorted(indexed)}")
    return indexed


def validate_g00a_checkpoint(
    outer_path: Path,
    current: dict[str, Any],
) -> dict[str, Any]:
    outer_path = outer_path.resolve()
    outer_bytes = outer_path.read_bytes()
    outer = strict_json_bytes(outer_bytes, "G00a outer gate manifest")
    root = outer_path.parent.resolve()
    require(outer_path == root / "gate.manifest.json", "--g00a must point to the outer gate.manifest.json")
    require(outer.get("schema_version") == 1, "G00a outer schema_version mismatch")
    require(outer.get("lane") == "vnext-g00a" and outer.get("status") == "pass", "G00a outer lane/status mismatch")
    require(outer.get("child_returncode") == 0 and outer.get("error") is None, "G00a outer child result mismatch")
    require(Path(require_string(outer.get("artifact_dir"), "G00a outer artifact_dir")).resolve() == root, "G00a outer artifact_dir mismatch")
    expected_outer_pass = f"FERRUM GATE vnext-g00a PASS: {root}"
    expected_child_pass = f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {root}"
    require(outer.get("pass_line") == expected_outer_pass, "G00a outer pass_line mismatch")
    require(outer.get("child_pass_line") == expected_child_pass, "G00a outer child_pass_line mismatch")
    dirty = require_object(outer.get("dirty_status"), "G00a outer dirty_status")
    require(dirty == {"is_dirty": False, "status_short": []}, "G00a outer dirty status mismatch")
    require(outer.get("git_sha") == current["git_sha"], "G00a outer manifest is stale against current HEAD")

    execution_rows = require_list(outer.get("child_execution_artifacts"), "G00a child execution artifacts")
    expected_execution = {
        "run_gate.child.command.json",
        "run_gate.child.stdout",
        "run_gate.child.stderr",
    }
    execution_by_path: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(execution_rows):
        row = require_object(raw, f"G00a child execution artifacts[{index}]")
        relative = safe_relative_path(row.get("path"), f"G00a child execution artifacts[{index}].path")
        require(relative not in execution_by_path, f"duplicate G00a child execution artifact: {relative}")
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"missing G00a execution artifact: {relative}")
        require(path.stat().st_size == row.get("size_bytes"), f"G00a execution artifact size mismatch: {relative}")
        require(file_sha256(path) == require_sha256(row.get("sha256"), f"G00a execution artifact {relative}.sha256"), f"G00a execution artifact SHA mismatch: {relative}")
        execution_by_path[relative] = row
    require(set(execution_by_path) == expected_execution, "G00a execution artifact set mismatch")

    outer_child = require_object(outer.get("child_artifacts"), "G00a outer child_artifacts")
    require(outer_child.get("kind") == "vnext-g00a", "G00a outer child provenance kind mismatch")
    collector_ref = require_object(outer_child.get("collector"), "G00a outer collector")
    require(collector_ref.get("git_sha") == current["git_sha"], "G00a collector SHA is stale")
    require(collector_ref.get("git_tree_sha") == current["git_tree_sha"], "G00a collector tree is stale")
    checkpoint_ref = require_object(outer_child.get("checkpoint"), "G00a outer checkpoint")
    require(checkpoint_ref.get("id") == "G00a", "G00a outer checkpoint id mismatch")
    require_exact_string_set(checkpoint_ref.get("unlocks"), {"G01A"}, "G00a outer checkpoint.unlocks")
    require_exact_string_set(checkpoint_ref.get("does_not_prove"), G00A_DOES_NOT_PROVE, "G00a outer checkpoint.does_not_prove")

    child_ref = require_object(outer_child.get("child_manifest"), "G00a outer child manifest ref")
    child_path = Path(require_string(child_ref.get("path"), "G00a child manifest path")).resolve()
    require(child_path == root / "manifest.json", "G00a child manifest path mismatch")
    require(child_path.is_file() and not child_path.is_symlink(), "G00a child manifest missing")
    child_bytes = child_path.read_bytes()
    child_sha = bytes_sha256(child_bytes)
    require(child_sha == require_sha256(child_ref.get("sha256"), "G00a child manifest SHA256"), "G00a child manifest SHA mismatch")
    child = strict_json_bytes(child_bytes, "G00a child manifest")
    require(child.get("schema_version") == 1, "G00a child schema_version mismatch")
    require(child.get("artifact_type") == "runtime_vnext_g00a_fact_checkpoint_manifest", "G00a child artifact_type mismatch")
    require(child.get("checkpoint_id") == "G00a" and child.get("lane") == "runtime-vnext-g00a", "G00a child checkpoint/lane mismatch")
    require(child.get("status") == "pass" and child.get("canonical") is True and child.get("dirty") is False, "G00a child status/canonical/dirty mismatch")
    require(child.get("pass_line") == expected_child_pass, "G00a child pass_line mismatch")
    require(Path(require_string(child.get("artifact_dir"), "G00a child artifact_dir")).resolve() == root, "G00a child artifact_dir mismatch")
    require_exact_string_set(child.get("unlocks"), {"G01A"}, "G00a child.unlocks")
    require_exact_string_set(child.get("does_not_prove"), G00A_DOES_NOT_PROVE, "G00a child.does_not_prove")
    collector = require_object(child.get("collector"), "G00a child collector")
    require(collector.get("git_sha") == current["git_sha"] and child.get("git_sha") == current["git_sha"], "G00a child SHA is stale")
    require(collector.get("git_tree_sha") == current["git_tree_sha"] and child.get("git_tree_sha") == current["git_tree_sha"], "G00a child tree is stale")
    require(collector_ref.get("contracts_sha256") == collector.get("contracts_sha256"), "G00a outer/child contract digest mismatch")

    artifact_index = validate_indexed_files(root, child.get("artifact_index"), G00A_ARTIFACTS, "G00a artifact_index")
    require(child.get("artifact_count") == len(G00A_ARTIFACTS), "G00a child artifact_count mismatch")
    index_sha = canonical_json_sha256(child["artifact_index"])
    require(index_sha == require_sha256(outer_child.get("artifact_index_sha256"), "G00a outer artifact_index_sha256"), "G00a outer/child artifact index digest mismatch")

    inventory_path = root / "coupling-inventory.json"
    inventory = read_json(inventory_path, "G00a coupling inventory")
    findings = require_list(require_object(inventory.get("coupling"), "G00a inventory coupling").get("findings"), "G00a inventory coupling.findings")
    backend_methods: list[dict[str, str]] = []
    for raw in findings:
        row = require_object(raw, "G00a coupling finding")
        if row.get("category") != "backend_trait_method":
            continue
        backend_methods.append(
            {
                "legacy_trait": require_string(row.get("trait"), "G00a backend trait"),
                "legacy_method": require_string(row.get("symbol"), "G00a backend method"),
            }
        )
    method_keys = {(row["legacy_trait"], row["legacy_method"]) for row in backend_methods}
    require(len(backend_methods) == 82 and len(method_keys) == 82, "G00a backend trait method inventory must contain exactly 82 unique methods")
    summary = require_object(inventory.get("summary"), "G00a inventory summary")
    counts = require_object(summary.get("coupling_count_by_category"), "G00a inventory category counts")
    require(counts.get("backend_trait_method") == 82, "G00a inventory summary backend method count mismatch")

    return {
        "outer_manifest": {"path": str(outer_path), "sha256": bytes_sha256(outer_bytes)},
        "child_manifest": {"path": str(child_path), "sha256": child_sha},
        "artifact_index_sha256": index_sha,
        "source": {"git_sha": current["git_sha"], "git_tree_sha": current["git_tree_sha"]},
        "coupling_inventory": {
            "path": str(inventory_path),
            "sha256": artifact_index["coupling-inventory.json"]["sha256"],
            "backend_trait_method_count": 82,
        },
        "backend_methods": sorted(backend_methods, key=lambda row: (row["legacy_trait"], row["legacy_method"])),
    }


def validate_contract_map(document: dict[str, Any], backend_methods: list[dict[str, str]]) -> dict[str, Any]:
    require(set(document) == {"schema_version", "artifact_type", "source", "mappings", "summary"}, "contract map top-level field set mismatch")
    require(document.get("schema_version") == 1, "contract map schema_version mismatch")
    require(document.get("artifact_type") == "runtime_vnext_g01a_legacy_contract_map", "contract map artifact_type mismatch")
    source = require_object(document.get("source"), "contract map source")
    require(set(source) == {"g00a_checkpoint_id", "category", "expected_method_count"}, "contract map source field set mismatch")
    require(source == {"g00a_checkpoint_id": "G00a", "category": "backend_trait_method", "expected_method_count": 82}, "contract map source mismatch")
    mappings = require_list(document.get("mappings"), "contract map mappings")
    require(len(mappings) == 82, "contract map must contain exactly 82 mappings")
    expected = {(row["legacy_trait"], row["legacy_method"]) for row in backend_methods}
    actual: set[tuple[str, str]] = set()
    classification_counts: dict[str, int] = {key: 0 for key in sorted(MAP_CLASSIFICATIONS)}
    for index, raw in enumerate(mappings):
        row = require_object(raw, f"contract map mappings[{index}]")
        require(set(row) == {"legacy_trait", "legacy_method", "classification", "owner", "disposition"}, f"contract map mapping[{index}] field set mismatch")
        key = (
            require_string(row.get("legacy_trait"), f"contract map mapping[{index}].legacy_trait"),
            require_string(row.get("legacy_method"), f"contract map mapping[{index}].legacy_method"),
        )
        require(key not in actual, f"duplicate contract map method: {key[0]}::{key[1]}")
        actual.add(key)
        classification = require_string(row.get("classification"), f"contract map mapping[{index}].classification")
        require(classification in MAP_CLASSIFICATIONS, f"unsupported contract map classification: {classification}")
        owner = require_string(row.get("owner"), f"contract map mapping[{index}].owner")
        disposition = require_string(row.get("disposition"), f"contract map mapping[{index}].disposition")
        require("special" not in owner.lower() and "special" not in disposition.lower(), f"contract map special-case escape hatch is forbidden: {key[0]}::{key[1]}")
        classification_counts[classification] += 1
    require(actual == expected, f"contract map method coverage mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    summary = require_object(document.get("summary"), "contract map summary")
    require(summary == {"mapped": 82, "unmapped": 0, "missing_owner": 0, "special_case": 0}, "contract map summary mismatch")
    require(sum(classification_counts.values()) == 82, "contract map classification count mismatch")
    return {
        "mapped": 82,
        "unmapped": 0,
        "missing_owner": 0,
        "special_case": 0,
        "classification_counts": classification_counts,
    }


def require_any_token(text: str, tokens: tuple[str, ...], label: str) -> None:
    lowered = text.lower()
    require(any(token.lower() in lowered for token in tokens), f"ADR is missing {label}")


def validate_adr(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CheckpointError(f"ADR is not UTF-8: {exc}") from exc
    require(len(text.strip()) >= 1000, "ADR is too short to substantiate the G01A decision")
    require_any_token(text, ("capability trait", "capability traits", "capability 小 trait", "小 capability trait"), "aggregated small capability-trait alternative")
    require_any_token(text, ("typed operation registry", "typed operation", "类型化 operation registry"), "typed operation registry alternative")
    require_any_token(text, ("compile-time", "compile time", "编译期"), "compile-time comparison")
    require_any_token(text, ("runtime overhead", "运行时开销", "运行期开销"), "runtime-overhead comparison")
    require_any_token(text, ("object safety", "对象安全"), "object-safety comparison")
    require_any_token(text, ("error localization", "错误定位"), "error-localization comparison")
    require_any_token(text, ("extension cost", "扩展成本"), "extension-cost comparison")
    require_any_token(text, ("decision", "决策"), "recorded decision")
    require("G01A" in text and "G01B" in text, "ADR must distinguish G01A from G01B")
    return {"utf8": True, "size_bytes": len(payload), "required_topics": 8}


def tracked_paths(patterns: list[str]) -> list[str]:
    output = run_git(["ls-files", "--", *patterns])
    return sorted({line for line in output.splitlines() if line.strip()})


def discover_contract_paths() -> dict[str, list[str]]:
    vnext_sources = tracked_paths(
        [
            "crates/ferrum-interfaces/src/vnext.rs",
            "crates/ferrum-interfaces/src/vnext/*.rs",
            "crates/ferrum-interfaces/src/vnext/**/*.rs",
        ]
    )
    require(vnext_sources, "no tracked ferrum-interfaces vnext source files found")
    tests = tracked_paths(["crates/ferrum-interfaces/tests/vnext_*.rs"])
    validate_test_driver_set(set(tests))
    fixtures = tracked_paths(
        [
            "crates/ferrum-interfaces/tests/ui/vnext/*.rs",
            "crates/ferrum-interfaces/tests/ui/vnext/*.stderr",
            "crates/ferrum-interfaces/tests/ui/vnext/**/*.rs",
            "crates/ferrum-interfaces/tests/ui/vnext/**/*.stderr",
        ]
    )
    pass_cases = [path for path in fixtures if "/pass/" in path and path.endswith(".rs")]
    fail_cases = [path for path in fixtures if "/fail/" in path and path.endswith(".rs")]
    require(pass_cases, "G01A trybuild suite has no compile/pass case")
    require(fail_cases, "G01A trybuild suite has no compile/fail case")
    fixture_set = set(fixtures)
    for case in fail_cases:
        require(case.removesuffix(".rs") + ".stderr" in fixture_set, f"compile/fail fixture lacks checked-in stderr: {case}")
    fixed = [
        "Cargo.lock",
        "crates/ferrum-interfaces/Cargo.toml",
        "crates/ferrum-interfaces/src/lib.rs",
        GOAL_PATH.relative_to(REPO_ROOT).as_posix(),
        ADR_PATH.relative_to(REPO_ROOT).as_posix(),
        MAP_PATH.relative_to(REPO_ROOT).as_posix(),
        SCRIPT_PATH.relative_to(REPO_ROOT).as_posix(),
        RUN_GATE_PATH.relative_to(REPO_ROOT).as_posix(),
        BOUNDED_COMMAND_PATH.relative_to(REPO_ROOT).as_posix(),
        "scripts/release/selftest_g0_validators.py",
    ]
    all_paths = sorted(set(fixed + vnext_sources + tests + fixtures))
    return {
        "all": all_paths,
        "vnext_sources": vnext_sources,
        "tests": tests,
        "fixtures": fixtures,
        "pass_cases": sorted(pass_cases),
        "fail_cases": sorted(fail_cases),
    }


def validate_test_driver_set(paths: set[str]) -> None:
    missing = REQUIRED_TEST_DRIVERS - paths
    require(not missing, f"missing required vnext test drivers: {sorted(missing)}")


def contract_file_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in paths:
        path = REPO_ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"contract path must be a regular non-symlink file: {relative}")
        blob = run_git(["rev-parse", f"HEAD:{relative}"])
        require(GIT_BLOB_RE.fullmatch(blob) is not None, f"contract path is not tracked at HEAD: {relative}")
        rows.append(
            {
                "path": relative,
                "git_blob": blob,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    validate_contract_file_rows(rows, set(paths))
    return rows


def validate_contract_file_rows(
    rows_raw: Any,
    expected_paths: set[str],
) -> dict[str, dict[str, Any]]:
    rows = require_list(rows_raw, "G01A contract files")
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = require_object(raw, f"G01A contract files[{index}]")
        require(set(row) == {"path", "git_blob", "sha256", "size_bytes"}, f"G01A contract files[{index}] field set mismatch")
        relative = safe_relative_path(row.get("path"), f"G01A contract files[{index}].path")
        require(relative not in indexed, f"duplicate G01A contract path: {relative}")
        path = REPO_ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"missing G01A contract path: {relative}")
        size = row.get("size_bytes")
        require(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"G01A contract size invalid: {relative}")
        require(path.stat().st_size == size, f"G01A contract size mismatch: {relative}")
        require(file_sha256(path) == require_sha256(row.get("sha256"), f"G01A contract {relative}.sha256"), f"G01A contract SHA256 mismatch: {relative}")
        blob = require_string(row.get("git_blob"), f"G01A contract {relative}.git_blob")
        require(GIT_BLOB_RE.fullmatch(blob) is not None, f"G01A contract Git blob invalid: {relative}")
        require(run_git(["rev-parse", f"HEAD:{relative}"]) == blob, f"G01A contract Git blob mismatch: {relative}")
        indexed[relative] = row
    require(set(indexed) == expected_paths, f"G01A contract file set mismatch: expected={sorted(expected_paths)} actual={sorted(indexed)}")
    return indexed


RustToken = tuple[str, str]


def _decode_rust_escape(source: str, index: int, label: str) -> tuple[str, int]:
    require(index < len(source), f"unterminated escape in {label}")
    escape = source[index]
    simple = {
        "0": "\0",
        "t": "\t",
        "n": "\n",
        "r": "\r",
        '"': '"',
        "'": "'",
        "\\": "\\",
    }
    if escape in simple:
        return simple[escape], index + 1
    if escape == "x":
        digits = source[index + 1 : index + 3]
        require(
            len(digits) == 2 and re.fullmatch(r"[0-9a-fA-F]{2}", digits) is not None,
            f"invalid hexadecimal escape in {label}",
        )
        return chr(int(digits, 16)), index + 3
    if escape == "u":
        require(index + 1 < len(source) and source[index + 1] == "{", f"invalid Unicode escape in {label}")
        end = source.find("}", index + 2)
        require(end >= 0, f"unterminated Unicode escape in {label}")
        digits = source[index + 2 : end].replace("_", "")
        require(
            1 <= len(digits) <= 6 and re.fullmatch(r"[0-9a-fA-F]+", digits) is not None,
            f"invalid Unicode escape in {label}",
        )
        value = int(digits, 16)
        require(value <= 0x10FFFF and not 0xD800 <= value <= 0xDFFF, f"invalid Unicode scalar in {label}")
        return chr(value), end + 1
    if escape in {"\n", "\r"}:
        next_index = index + 1
        if escape == "\r" and next_index < len(source) and source[next_index] == "\n":
            next_index += 1
        while next_index < len(source) and source[next_index] in " \t\n\r":
            next_index += 1
        return "", next_index
    raise CheckpointError(f"unsupported escape in {label}: \\{escape}")


def _raw_string_start(source: str, index: int) -> tuple[int, int] | None:
    for prefix in ("br", "cr", "r"):
        if not source.startswith(prefix, index):
            continue
        cursor = index + len(prefix)
        hashes = 0
        while cursor < len(source) and source[cursor] == "#":
            hashes += 1
            cursor += 1
        if cursor < len(source) and source[cursor] == '"':
            return cursor + 1, hashes
    return None


def _normal_string_start(source: str, index: int) -> int | None:
    if source[index] == '"':
        return index + 1
    if source[index] in {"b", "c"} and index + 1 < len(source) and source[index + 1] == '"':
        return index + 2
    return None


def rust_tokens(source: str, label: str = "Rust source") -> list[RustToken]:
    tokens: list[RustToken] = []
    index = 0
    length = len(source)
    while index < length:
        character = source[index]
        if character.isspace():
            index += 1
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = length if newline < 0 else newline + 1
            continue
        if source.startswith("/*", index):
            depth = 1
            index += 2
            while index < length and depth:
                if source.startswith("/*", index):
                    depth += 1
                    index += 2
                elif source.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            require(depth == 0, f"unterminated block comment in {label}")
            continue

        raw_start = _raw_string_start(source, index)
        if raw_start is not None:
            content_start, hashes = raw_start
            terminator = '"' + ("#" * hashes)
            end = source.find(terminator, content_start)
            require(end >= 0, f"unterminated raw string in {label}")
            tokens.append(("string", source[content_start:end]))
            index = end + len(terminator)
            continue

        string_start = _normal_string_start(source, index)
        if string_start is not None:
            cursor = string_start
            decoded: list[str] = []
            while cursor < length and source[cursor] != '"':
                if source[cursor] == "\\":
                    value, cursor = _decode_rust_escape(source, cursor + 1, label)
                    decoded.append(value)
                else:
                    decoded.append(source[cursor])
                    cursor += 1
            require(cursor < length, f"unterminated string in {label}")
            tokens.append(("string", "".join(decoded)))
            index = cursor + 1
            continue

        if character == "'":
            if index + 2 < length and source[index + 2] == "'":
                index += 3
                continue
            if index + 1 < length and source[index + 1] == "\\":
                _, cursor = _decode_rust_escape(source, index + 2, label)
                require(cursor < length and source[cursor] == "'", f"unterminated character literal in {label}")
                index = cursor + 1
                continue
            tokens.append(("punct", character))
            index += 1
            continue
        if character == "b" and index + 3 < length and source[index + 1] == "'":
            cursor = index + 2
            if source[cursor] == "\\":
                _, cursor = _decode_rust_escape(source, cursor + 1, label)
            else:
                cursor += 1
            require(cursor < length and source[cursor] == "'", f"unterminated byte literal in {label}")
            index = cursor + 1
            continue

        if source.startswith("r#", index) and index + 2 < length:
            next_character = source[index + 2]
            if next_character == "_" or next_character.isalpha():
                cursor = index + 3
                while cursor < length and (source[cursor] == "_" or source[cursor].isalnum()):
                    cursor += 1
                tokens.append(("ident", source[index + 2 : cursor]))
                index = cursor
                continue
        if character == "_" or character.isalpha():
            cursor = index + 1
            while cursor < length and (source[cursor] == "_" or source[cursor].isalnum()):
                cursor += 1
            tokens.append(("ident", source[index:cursor]))
            index = cursor
            continue
        if character.isdigit():
            cursor = index + 1
            while cursor < length and (source[cursor].isalnum() or source[cursor] in "_."):
                cursor += 1
            tokens.append(("number", source[index:cursor]))
            index = cursor
            continue
        if source.startswith("::", index):
            tokens.append(("punct", "::"))
            index += 2
            continue
        tokens.append(("punct", character))
        index += 1
    return tokens


def _matching_token(tokens: list[RustToken], start: int, opening: str, closing: str) -> int:
    require(start < len(tokens) and tokens[start] == ("punct", opening), f"missing {opening} token")
    depth = 0
    for index in range(start, len(tokens)):
        if tokens[index] == ("punct", opening):
            depth += 1
        elif tokens[index] == ("punct", closing):
            depth -= 1
            if depth == 0:
                return index
    raise CheckpointError(f"unbalanced {opening}{closing} tokens in vnext source")


def _token_sequence_count(tokens: list[RustToken], sequence: tuple[RustToken, ...]) -> int:
    width = len(sequence)
    return sum(
        tokens[index : index + width] == list(sequence)
        for index in range(0, len(tokens) - width + 1)
    )


def _named_item_bodies(
    tokens: list[RustToken], kind: str, name: str
) -> list[list[RustToken]]:
    bodies: list[list[RustToken]] = []
    for index in range(len(tokens) - 1):
        if tokens[index] != ("ident", kind) or tokens[index + 1] != ("ident", name):
            continue
        opening = _item_body_opening(tokens, index + 2, f"{kind} {name}")
        closing = _matching_token(tokens, opening, "{", "}")
        bodies.append(tokens[opening + 1 : closing])
    return bodies


def _named_impl_bodies(tokens: list[RustToken], name: str) -> list[list[RustToken]]:
    bodies: list[list[RustToken]] = []
    for index, token in enumerate(tokens):
        if token != ("ident", "impl"):
            continue
        # `impl` also appears in argument and return-position `impl Trait`.
        # Reject those before asking the item parser to find a body: treating a
        # function's generic `<...>` as an impl-item header can make the scanner
        # consume the surrounding function and report bogus delimiter failures.
        if not _impl_header_mentions_name(tokens, index + 1, name):
            continue
        opening = _item_body_opening(tokens, index + 1, f"impl {name}")
        header = tokens[index + 1 : opening]
        if ("ident", name) not in header or ("ident", "for") in header:
            continue
        closing = _matching_token(tokens, opening, "{", "}")
        bodies.append(tokens[opening + 1 : closing])
    return bodies


def _impl_header_mentions_name(
    tokens: list[RustToken], start: int, name: str
) -> bool:
    angle_depth = 0
    bracket_depth = 0
    cursor = start
    while cursor < len(tokens):
        token = tokens[cursor]
        if token == ("ident", name):
            return True
        if token == ("punct", "<"):
            angle_depth += 1
        elif token == ("punct", ">") and angle_depth:
            angle_depth -= 1
        elif token == ("punct", "["):
            bracket_depth += 1
        elif token == ("punct", "]"):
            bracket_depth -= 1
        elif angle_depth == bracket_depth == 0 and token in {
            ("punct", ")"),
            ("punct", ";"),
            ("punct", "="),
            ("punct", "{"),
        }:
            return False
        require(
            angle_depth >= 0 and bracket_depth >= 0,
            "unbalanced delimiters while classifying impl header",
        )
        cursor += 1
    return False


def _unrestricted_public_functions(
    tokens: list[RustToken],
) -> list[dict[str, Any]]:
    functions: list[dict[str, Any]] = []
    modifiers = {"async", "const", "default", "extern", "unsafe"}
    for index, token in enumerate(tokens):
        if token != ("ident", "pub"):
            continue
        cursor = index + 1
        if cursor < len(tokens) and tokens[cursor] == ("punct", "("):
            continue
        while cursor < len(tokens) and (
            tokens[cursor][0] == "ident" and tokens[cursor][1] in modifiers
        ):
            cursor += 1
        if cursor >= len(tokens) or tokens[cursor] != ("ident", "fn"):
            continue
        require(
            cursor + 1 < len(tokens) and tokens[cursor + 1][0] == "ident",
            "public function lacks a name in vnext source",
        )
        name = tokens[cursor + 1][1]
        parameters_open = next(
            (
                candidate
                for candidate in range(cursor + 2, len(tokens))
                if tokens[candidate] == ("punct", "(")
            ),
            None,
        )
        require(parameters_open is not None, f"public function {name} lacks parameters")
        parameters_close = _matching_token(tokens, parameters_open, "(", ")")
        signature_end = parameters_close + 1
        angle_depth = 0
        bracket_depth = 0
        while signature_end < len(tokens):
            current = tokens[signature_end]
            if current == ("punct", "<"):
                angle_depth += 1
            elif current == ("punct", ">") and angle_depth:
                angle_depth -= 1
            elif current == ("punct", "["):
                bracket_depth += 1
            elif current == ("punct", "]"):
                bracket_depth -= 1
            elif angle_depth == bracket_depth == 0 and current in {
                ("punct", "{"),
                ("punct", ";"),
            }:
                break
            signature_end += 1
        require(signature_end < len(tokens), f"public function {name} lacks a terminator")
        body: list[RustToken] = []
        if tokens[signature_end] == ("punct", "{"):
            body_end = _matching_token(tokens, signature_end, "{", "}")
            body = tokens[signature_end + 1 : body_end]
        functions.append(
            {
                "name": name,
                "parameters": tokens[parameters_open + 1 : parameters_close],
                "signature": tokens[cursor:signature_end],
                "body": body,
            }
        )
    return functions


def _execution_identity_version(event_tokens: list[RustToken]) -> tuple[int, int, int]:
    initializers: list[list[RustToken]] = []
    for index in range(len(event_tokens) - 1):
        if (
            event_tokens[index] != ("ident", "const")
            or event_tokens[index + 1] != ("ident", "EXECUTION_IDENTITY_VERSION")
        ):
            continue
        require(
            index > 0 and event_tokens[index - 1] == ("ident", "pub"),
            "EXECUTION_IDENTITY_VERSION must be an unrestricted public constant",
        )
        equals = next(
            (
                cursor
                for cursor in range(index + 2, len(event_tokens))
                if event_tokens[cursor] == ("punct", "=")
            ),
            None,
        )
        require(equals is not None, "EXECUTION_IDENTITY_VERSION lacks an initializer")
        semicolon = next(
            (
                cursor
                for cursor in range(equals + 1, len(event_tokens))
                if event_tokens[cursor] == ("punct", ";")
            ),
            None,
        )
        require(semicolon is not None, "EXECUTION_IDENTITY_VERSION lacks a terminator")
        initializers.append(event_tokens[equals + 1 : semicolon])
    expected = [
        ("ident", "ContractVersion"),
        ("punct", "::"),
        ("ident", "new"),
        ("punct", "("),
        ("number", str(G01A_EXECUTION_IDENTITY_VERSION[0])),
        ("punct", ","),
        ("number", str(G01A_EXECUTION_IDENTITY_VERSION[1])),
        ("punct", ")"),
    ]
    require(
        len(initializers) == 1 and initializers[0] == expected,
        "execution identity version must be exactly 3.0",
    )
    return (
        G01A_EXECUTION_IDENTITY_VERSION[0],
        G01A_EXECUTION_IDENTITY_VERSION[1],
        len(initializers),
    )


def validate_g01a_semantic_contracts(
    source_text_by_path: dict[str, str],
) -> dict[str, Any]:
    event_sources = [
        source
        for relative, source in source_text_by_path.items()
        if Path(relative).name == "event.rs"
    ]
    require(len(event_sources) == 1, "G01A source set must contain exactly one event.rs")
    event_tokens = rust_tokens(event_sources[0], "vnext event contract")
    combined_tokens: list[RustToken] = []
    for relative in sorted(source_text_by_path):
        combined_tokens.extend(
            rust_tokens(source_text_by_path[relative], f"vnext source {relative}")
        )

    major, minor, version_definition_count = _execution_identity_version(event_tokens)
    event_bodies = _named_item_bodies(event_tokens, "enum", "ExecutionEventKind")
    require(
        len(event_bodies) == 1,
        "vnext event contract must define exactly one ExecutionEventKind enum",
    )
    required_variant_count = sum(
        token == ("ident", G01A_EVENT_REQUIRED_TRANSITION)
        for token in event_bodies[0]
    )
    forbidden_identifier_count = sum(
        token == ("ident", G01A_EVENT_FORBIDDEN_TRANSITION)
        for token in event_tokens
    )
    require(
        required_variant_count == 1,
        f"ExecutionEventKind must define exactly one {G01A_EVENT_REQUIRED_TRANSITION}",
    )
    require(
        forbidden_identifier_count == 0,
        f"vnext event contract must contain zero {G01A_EVENT_FORBIDDEN_TRANSITION} identifiers",
    )

    semantic_definition_counts: dict[str, int] = {}
    for name, kind in G01A_SEMANTIC_TYPE_KINDS.items():
        count = sum(
            combined_tokens[index] == ("ident", kind)
            and index + 1 < len(combined_tokens)
            and combined_tokens[index + 1] == ("ident", name)
            for index in range(len(combined_tokens))
        )
        require(
            count == 1,
            f"required G01A semantic type {name} must have exactly one {kind} definition, found {count}",
        )
        semantic_definition_counts[name] = count

    retry_bodies = _named_item_bodies(
        combined_tokens, "struct", G01A_DNF_RETRY_AUTHORITY_TYPE
    )
    require(
        len(retry_bodies) == 1 and any(token == ("punct", ":") for token in retry_bodies[0]),
        f"{G01A_DNF_RETRY_AUTHORITY_TYPE} must be a field-sealed struct",
    )
    require(
        ("ident", "pub") not in retry_bodies[0],
        f"{G01A_DNF_RETRY_AUTHORITY_TYPE} must have no public fields",
    )
    retry_public_associated_constructors = sum(
        ("ident", "self") not in function["parameters"]
        for body in _named_impl_bodies(combined_tokens, G01A_DNF_RETRY_AUTHORITY_TYPE)
        for function in _unrestricted_public_functions(body)
    )
    require(
        retry_public_associated_constructors == 0,
        f"{G01A_DNF_RETRY_AUTHORITY_TYPE} must have zero public associated constructors",
    )

    dispatch_functions = [
        function
        for body in _named_impl_bodies(combined_tokens, "OperationDispatch")
        for function in _unrestricted_public_functions(body)
    ]
    observed_dispatch_markers = {
        "batch_identity_parameter": any(
            ("ident", G01A_MULTIPARTICIPANT_DISPATCH_MARKERS["batch_identity_parameter"])
            in function["parameters"]
            for function in dispatch_functions
        ),
        "participant_bindings_parameter": any(
            ("ident", G01A_MULTIPARTICIPANT_DISPATCH_MARKERS["participant_bindings_parameter"])
            in function["parameters"]
            for function in dispatch_functions
        ),
        "participant_binding_type": any(
            ("ident", G01A_MULTIPARTICIPANT_DISPATCH_MARKERS["participant_binding_type"])
            in function["parameters"]
            for function in dispatch_functions
        ),
        "batched_invocation_body": any(
            ("ident", G01A_MULTIPARTICIPANT_DISPATCH_MARKERS["batched_invocation_body"])
            in function["body"]
            for function in dispatch_functions
        ),
    }
    for label, observed in observed_dispatch_markers.items():
        require(observed, f"multi-participant dispatch marker missing: {label}")
    matching_dispatch_methods = sum(
        all(
            (
                ("ident", marker) in function["body"]
                if label == "batched_invocation_body"
                else ("ident", marker) in function["parameters"]
            )
            for label, marker in G01A_MULTIPARTICIPANT_DISPATCH_MARKERS.items()
        )
        for function in dispatch_functions
    )
    require(
        matching_dispatch_methods == 1,
        "exactly one public OperationDispatch method must close over all multi-participant markers",
    )

    raw_shape_public_type_count = _token_sequence_count(
        combined_tokens,
        (
            ("ident", "pub"),
            ("ident", "struct"),
            ("ident", "DynamicResourceShape"),
        ),
    )
    raw_shape_public_impl_method_count = sum(
        len(_unrestricted_public_functions(body))
        for body in _named_impl_bodies(combined_tokens, "DynamicResourceShape")
    )
    raw_shape_public_parameter_path_count = sum(
        ("ident", "DynamicResourceShape") in function["parameters"]
        for function in _unrestricted_public_functions(combined_tokens)
    )
    raw_shape_counts = {
        "unrestricted_public_type_count": raw_shape_public_type_count,
        "unrestricted_public_impl_method_count": raw_shape_public_impl_method_count,
        "unrestricted_public_parameter_path_count": raw_shape_public_parameter_path_count,
    }
    for label, count in raw_shape_counts.items():
        require(
            count == 0,
            f"public raw DynamicResourceShape {label} must be zero, found {count}",
        )

    return {
        "schema_version": 1,
        "execution_identity": {
            "constant": "EXECUTION_IDENTITY_VERSION",
            "major": major,
            "minor": minor,
            "definition_count": version_definition_count,
        },
        "event_transition": {
            "enum": "ExecutionEventKind",
            "required_variant": G01A_EVENT_REQUIRED_TRANSITION,
            "required_variant_count": required_variant_count,
            "forbidden_identifier": G01A_EVENT_FORBIDDEN_TRANSITION,
            "forbidden_identifier_count": forbidden_identifier_count,
        },
        "required_type_kinds": copy.deepcopy(G01A_SEMANTIC_TYPE_KINDS),
        "definition_counts": semantic_definition_counts,
        "dnf_retry_authority": {
            "type_name": G01A_DNF_RETRY_AUTHORITY_TYPE,
            "field_sealed": True,
            "public_associated_constructor_count": retry_public_associated_constructors,
        },
        "multi_participant_dispatch": {
            "owner": "OperationDispatch",
            "required_markers": copy.deepcopy(G01A_MULTIPARTICIPANT_DISPATCH_MARKERS),
            "observed_markers": observed_dispatch_markers,
            "matching_public_method_count": matching_dispatch_methods,
        },
        "public_raw_dynamic_resource_shape": {
            "type_name": "DynamicResourceShape",
            **raw_shape_counts,
        },
    }


def _backend_feature_cfg_count(tokens: list[RustToken]) -> int:
    count = 0
    for index, token in enumerate(tokens[:-1]):
        if token != ("ident", "cfg") or tokens[index + 1] != ("punct", "("):
            continue
        end = _matching_token(tokens, index + 1, "(", ")")
        body = tokens[index + 2 : end]
        for offset in range(len(body) - 2):
            if (
                body[offset] == ("ident", "feature")
                and body[offset + 1] == ("punct", "=")
                and body[offset + 2][0] == "string"
                and body[offset + 2][1] in {"cuda", "metal"}
            ):
                count += 1
    return count


def _item_body_opening(tokens: list[RustToken], start: int, label: str) -> int:
    cursor = start
    paren_depth = 0
    bracket_depth = 0
    angle_depth = 0
    while cursor < len(tokens):
        token = tokens[cursor]
        if token == ("punct", "("):
            paren_depth += 1
        elif token == ("punct", ")"):
            paren_depth -= 1
        elif token == ("punct", "["):
            bracket_depth += 1
        elif token == ("punct", "]"):
            bracket_depth -= 1
        elif token == ("punct", "<"):
            angle_depth += 1
        elif token == ("punct", ">") and angle_depth:
            angle_depth -= 1
        elif token == ("punct", "{"):
            if paren_depth == bracket_depth == angle_depth == 0:
                return cursor
            cursor = _matching_token(tokens, cursor, "{", "}")
        require(
            paren_depth >= 0 and bracket_depth >= 0 and angle_depth >= 0,
            f"unbalanced delimiters in {label}",
        )
        cursor += 1
    raise CheckpointError(f"{label} lacks a body in vnext source")


def _silent_success_default_count(tokens: list[RustToken]) -> int:
    count = 0
    for index, token in enumerate(tokens):
        if token != ("ident", "trait"):
            continue
        opening = _item_body_opening(tokens, index + 1, "trait declaration")
        closing = _matching_token(tokens, opening, "{", "}")
        cursor = opening + 1
        body_depth = 0
        while cursor < closing:
            current = tokens[cursor]
            if current == ("punct", "{"):
                body_depth += 1
            elif current == ("punct", "}"):
                body_depth -= 1
            elif body_depth == 0 and current == ("ident", "fn"):
                signature_cursor = cursor + 1
                paren_depth = 0
                bracket_depth = 0
                angle_depth = 0
                while signature_cursor < closing:
                    signature_token = tokens[signature_cursor]
                    if signature_token == ("punct", "("):
                        paren_depth += 1
                    elif signature_token == ("punct", ")"):
                        paren_depth -= 1
                    elif signature_token == ("punct", "["):
                        bracket_depth += 1
                    elif signature_token == ("punct", "]"):
                        bracket_depth -= 1
                    elif signature_token == ("punct", "<"):
                        angle_depth += 1
                    elif signature_token == ("punct", ">") and angle_depth:
                        angle_depth -= 1
                    elif (
                        paren_depth == bracket_depth == angle_depth == 0
                        and signature_token == ("punct", ";")
                    ):
                        cursor = signature_cursor
                        break
                    elif signature_token == ("punct", "{"):
                        if paren_depth or bracket_depth or angle_depth:
                            signature_cursor = _matching_token(
                                tokens, signature_cursor, "{", "}"
                            )
                            signature_cursor += 1
                            continue
                        method_end = _matching_token(tokens, signature_cursor, "{", "}")
                        method_body = tokens[signature_cursor + 1 : method_end]
                        if any(
                            token_kind == "ident"
                            and token_value in {"Ok", "Some", "true", "default"}
                            for token_kind, token_value in method_body
                        ):
                            count += 1
                        cursor = method_end
                        break
                    signature_cursor += 1
                else:
                    raise CheckpointError("unterminated trait method in vnext source")
            cursor += 1
    return count


def validate_vnext_static_contracts(source_paths: list[str]) -> dict[str, Any]:
    source_text_by_path: dict[str, str] = {}
    for relative in source_paths:
        try:
            source_text_by_path[relative] = (REPO_ROOT / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise CheckpointError(f"cannot inspect vnext source {relative}: {exc}") from exc
    combined = "\n".join(source_text_by_path[relative] for relative in sorted(source_text_by_path))
    tokens = rust_tokens(combined, "combined vnext source")
    definitions: dict[str, int] = {}
    for name in REQUIRED_CONTRACTS:
        count = sum(
            tokens[index][0] == "ident"
            and tokens[index][1] in {"trait", "struct", "enum"}
            and index + 1 < len(tokens)
            and tokens[index + 1] == ("ident", name)
            for index in range(len(tokens))
        )
        require(count == 1, f"required contract {name} must have exactly one definition, found {count}")
        definitions[name] = count
    architecture_symbols = sorted(
        {
            value
            for kind, value in tokens
            if kind == "ident" and KNOWN_ARCHITECTURE_RE.fullmatch(value) is not None
        }
    )
    require(not architecture_symbols, f"generic vnext contracts contain architecture names: {architecture_symbols}")
    identifiers = [value for kind, value in tokens if kind == "ident"]
    forbidden_counts = {
        "Any/downcast": sum(
            value in {"Any", "TypeId", "downcast", "downcast_ref", "downcast_mut"}
            for value in identifiers
        ),
        "backend feature cfg": _backend_feature_cfg_count(tokens),
        "product/runtime import": sum(
            value
            in {
                "ferrum_cli",
                "ferrum_server",
                "ferrum_engine",
                "ferrum_models",
                "ferrum_kernels",
            }
            for value in identifiers
        ),
        "hidden environment read": (
            _token_sequence_count(
                tokens,
                (("ident", "std"), ("punct", "::"), ("ident", "env")),
            )
            + _token_sequence_count(
                tokens,
                (("ident", "env"), ("punct", "::"), ("ident", "var")),
            )
            + _token_sequence_count(
                tokens,
                (("ident", "env"), ("punct", "::"), ("ident", "var_os")),
            )
        ),
    }
    require(
        set(forbidden_counts) == FORBIDDEN_VNEXT_CHECKS,
        "token-aware forbidden contract matrix drift",
    )
    for label, count in forbidden_counts.items():
        require(count == 0, f"generic vnext contracts contain forbidden {label}: count={count}")
    silent_success_defaults = _silent_success_default_count(tokens)
    require(
        silent_success_defaults == 0,
        f"generic vnext traits contain silent success defaults: count={silent_success_defaults}",
    )
    semantic_contracts = validate_g01a_semantic_contracts(source_text_by_path)
    return {
        "required_contract_count": len(REQUIRED_CONTRACTS),
        "required_contracts": list(REQUIRED_CONTRACTS),
        "definition_counts": definitions,
        "architecture_named_symbol_count": 0,
        "silent_success_default_count": silent_success_defaults,
        "forbidden_pattern_counts": forbidden_counts,
        "semantic_contracts": semantic_contracts,
    }


def validate_repository_isolation(vnext_source_paths: list[str]) -> dict[str, Any]:
    production_paths = tracked_paths(["crates/*/src/*.rs", "crates/*/src/**/*.rs"])
    vnext_sources = set(vnext_source_paths)
    integration_path = "crates/ferrum-interfaces/src/lib.rs"
    integration_text = (REPO_ROOT / integration_path).read_text(encoding="utf-8")
    declaration_count = len(re.findall(r"(?m)^\s*pub\s+mod\s+vnext\s*;\s*$", integration_text))
    require(declaration_count == 1, f"ferrum-interfaces must expose exactly one `pub mod vnext;`, found {declaration_count}")
    reference_pattern = re.compile(r"(?:\b[A-Za-z_][A-Za-z0-9_]*::)*\bvnext::|::vnext\b")
    outside_references: list[dict[str, Any]] = []
    for relative in production_paths:
        if relative in vnext_sources:
            continue
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        if relative == integration_path:
            text = re.sub(r"(?m)^\s*pub\s+mod\s+vnext\s*;\s*$", "", text)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if reference_pattern.search(line):
                outside_references.append(
                    {"path": relative, "line": line_number, "text": line.strip()}
                )
    require(
        not outside_references,
        f"production vnext references escape the isolated interfaces module: {outside_references[:10]}",
    )
    return {
        "production_source_file_count": len(production_paths),
        "allowed_module_declaration_count": declaration_count,
        "outside_vnext_reference_count": 0,
    }


def bounded_profile_for_command(command: list[str] | tuple[str, ...]) -> str:
    require(
        tuple(command[:2]) == ("cargo", "test"),
        "bounded G01A evidence command must be cargo test",
    )
    require(
        command.count(TEST_THREADS_ARG) == 1
        and "--" in command
        and command.index(TEST_THREADS_ARG) > command.index("--"),
        f"bounded cargo test must contain exactly one {TEST_THREADS_ARG}",
    )
    if "--test" in command:
        target_index = command.index("--test") + 1
        require(target_index < len(command), "bounded cargo test target is missing")
        target = command[target_index]
        if target == "vnext_resource_contract_tests":
            return "resource"
        if target == "vnext_compile":
            return "trybuild"
    if "vnext::admission" in command and "--lib" in command:
        return "admission"
    return "regular"


def validate_bounded_receipt(
    receipt_raw: Any,
    command: list[str] | tuple[str, ...],
    profile_name: str,
    stdout_bytes: bytes,
    stderr_bytes: bytes,
    label: str,
) -> dict[str, Any]:
    receipt = require_object(receipt_raw, f"{label} receipt")
    expected_fields = {
        "schema",
        "command",
        "cwd",
        "pid",
        "pgid",
        "limits",
        "peaks",
        "started_at",
        "ended_at",
        "duration_seconds",
        "reason",
        "rc",
        "status",
        "successful_samples",
        "sampling_error_count",
        "sampling_errors",
        "violation",
        "termination",
        "cleanup",
        "stdout",
        "stderr",
    }
    require(set(receipt) == expected_fields, f"{label} receipt field set mismatch")
    require(receipt.get("schema") == BOUNDED_RECEIPT_SCHEMA, f"{label} receipt schema mismatch")
    receipt_command = require_list(receipt.get("command"), f"{label} receipt command")
    require(receipt_command == list(command), f"{label} receipt command mismatch")
    require(
        Path(require_string(receipt.get("cwd"), f"{label} receipt cwd")).resolve()
        == REPO_ROOT.resolve(),
        f"{label} receipt cwd mismatch",
    )
    pid = receipt.get("pid")
    pgid = receipt.get("pgid")
    require(
        isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid > 0
        and pgid == pid,
        f"{label} receipt pid/pgid mismatch",
    )
    require(profile_name in BOUNDED_TEST_PROFILES, f"{label} bounded profile is unknown")
    expected_limits = BOUNDED_TEST_PROFILES[profile_name]
    limits = require_object(receipt.get("limits"), f"{label} receipt limits")
    require(set(limits) == set(expected_limits), f"{label} receipt limit field set mismatch")
    require(
        all(
            isinstance(limits.get(key), (int, float))
            and not isinstance(limits.get(key), bool)
            and limits[key] == value
            for key, value in expected_limits.items()
        ),
        f"{label} receipt limits mismatch",
    )
    rc = receipt.get("rc")
    require(
        receipt.get("status") == "pass"
        and receipt.get("reason") == "command_completed"
        and isinstance(rc, int)
        and not isinstance(rc, bool)
        and rc == 0,
        f"{label} receipt command status mismatch",
    )
    successful_samples = receipt.get("successful_samples")
    require(
        isinstance(successful_samples, int)
        and not isinstance(successful_samples, bool)
        and successful_samples >= 1,
        f"{label} receipt successful_samples invalid",
    )
    sampling_error_count = receipt.get("sampling_error_count")
    require(
        isinstance(sampling_error_count, int)
        and not isinstance(sampling_error_count, bool)
        and sampling_error_count == 0
        and receipt.get("sampling_errors") == [],
        f"{label} receipt contains sampling errors",
    )
    require(receipt.get("violation") is None, f"{label} receipt contains a resource violation")
    require(
        receipt.get("termination") == {"signals": [], "errors": []},
        f"{label} receipt termination is not clean",
    )
    cleanup = require_object(receipt.get("cleanup"), f"{label} receipt cleanup")
    require(
        set(cleanup) == {"process_group_gone"}
        and cleanup.get("process_group_gone") is True,
        f"{label} receipt process group cleanup failed",
    )
    require(
        bool(require_string(receipt.get("started_at"), f"{label} receipt started_at"))
        and bool(require_string(receipt.get("ended_at"), f"{label} receipt ended_at"))
        and isinstance(receipt.get("duration_seconds"), (int, float))
        and not isinstance(receipt.get("duration_seconds"), bool)
        and receipt["duration_seconds"] >= 0,
        f"{label} receipt timing evidence invalid",
    )
    peaks = require_object(receipt.get("peaks"), f"{label} receipt peaks")
    require(
        set(peaks)
        == {
            "processes",
            "group_threads",
            "per_process_threads",
            "per_process_threads_pid",
        },
        f"{label} receipt peak field set mismatch",
    )
    for key in ("processes", "group_threads", "per_process_threads"):
        value = peaks.get(key)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 1,
            f"{label} receipt peak {key} invalid",
        )
    peak_pid = peaks.get("per_process_threads_pid")
    require(
        isinstance(peak_pid, int) and not isinstance(peak_pid, bool) and peak_pid > 0,
        f"{label} receipt peak pid invalid",
    )
    require(
        peaks["processes"] <= limits["max_processes"]
        and peaks["group_threads"] <= limits["max_group_threads"]
        and peaks["per_process_threads"] <= limits["max_per_process_threads"]
        and peaks["group_threads"] >= peaks["processes"]
        and peaks["group_threads"] >= peaks["per_process_threads"],
        f"{label} receipt peak exceeds its fixed bound",
    )
    identities = (("stdout", stdout_bytes), ("stderr", stderr_bytes))
    paths: list[str] = []
    for stream, payload in identities:
        identity = require_object(receipt.get(stream), f"{label} receipt {stream}")
        require(
            set(identity) == {"path", "sha256", "size_bytes"},
            f"{label} receipt {stream} identity field set mismatch",
        )
        paths.append(require_string(identity.get("path"), f"{label} receipt {stream}.path"))
        size_bytes = identity.get("size_bytes")
        require(
            isinstance(size_bytes, int)
            and not isinstance(size_bytes, bool)
            and size_bytes == len(payload)
            and require_sha256(identity.get("sha256"), f"{label} receipt {stream}.sha256")
            == bytes_sha256(payload),
            f"{label} receipt {stream} identity mismatch",
        )
    require(len(set(paths)) == 2, f"{label} receipt output paths are not distinct")
    return receipt


def validate_command_execution(row: dict[str, Any], label: str) -> dict[str, Any] | None:
    command = require_list(row.get("command"), f"{label}.command")
    require(
        all(isinstance(part, str) and part for part in command),
        f"{label}.command contains an invalid argv item",
    )
    execution = require_object(row.get("execution"), f"{label}.execution")
    require(
        set(execution) == {"kind", "profile", "receipt", "receipt_sha256"},
        f"{label}.execution field set mismatch",
    )
    stdout = row.get("stdout")
    stderr = row.get("stderr")
    require(isinstance(stdout, str) and isinstance(stderr, str), f"{label} output type mismatch")
    if tuple(command) in QUALITY_COMMANDS:
        require(
            execution
            == {
                "kind": "direct",
                "profile": None,
                "receipt": None,
                "receipt_sha256": None,
            },
            f"{label} direct quality execution metadata mismatch",
        )
        require(
            row.get("env_overrides") == {"PYTHONDONTWRITEBYTECODE": "1"},
            f"{label} direct quality environment mismatch",
        )
        return None
    profile_name = bounded_profile_for_command(command)
    require(execution.get("kind") == "bounded-command", f"{label} cargo test is not bounded")
    require(execution.get("profile") == profile_name, f"{label} bounded profile mismatch")
    require(
        row.get("env_overrides") == BOUNDED_TEST_ENV_OVERRIDES,
        f"{label} bounded environment mismatch",
    )
    receipt = validate_bounded_receipt(
        execution.get("receipt"),
        command,
        profile_name,
        stdout.encode("utf-8"),
        stderr.encode("utf-8"),
        label,
    )
    require(
        require_sha256(execution.get("receipt_sha256"), f"{label}.receipt_sha256")
        == canonical_json_sha256(receipt),
        f"{label} bounded receipt SHA256 mismatch",
    )
    return receipt


def summarize_bounded_execution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bounded: list[tuple[str, dict[str, Any]]] = []
    direct_count = 0
    profile_counts = {name: 0 for name in BOUNDED_TEST_PROFILES}
    for index, row in enumerate(rows):
        receipt = validate_command_execution(row, f"G01A evidence command[{index}]")
        if receipt is None:
            direct_count += 1
            continue
        profile_name = require_string(
            require_object(row["execution"], "G01A evidence execution").get("profile"),
            "G01A bounded profile",
        )
        profile_counts[profile_name] += 1
        bounded.append((profile_name, receipt))
    require(direct_count == len(QUALITY_COMMANDS), "G01A direct quality command count mismatch")
    require(
        len(bounded) == BOUNDED_TEST_COMMAND_COUNT,
        f"G01A must contain exactly {BOUNDED_TEST_COMMAND_COUNT} bounded cargo test commands",
    )
    require(
        profile_counts
        == {"regular": 14, "admission": 2, "resource": 2, "trybuild": 2},
        f"G01A bounded profile command counts mismatch: {profile_counts}",
    )
    return {
        "runner": BOUNDED_COMMAND_PATH.relative_to(REPO_ROOT).as_posix(),
        "receipt_schema": BOUNDED_RECEIPT_SCHEMA,
        "required_command_count": BOUNDED_TEST_COMMAND_COUNT,
        "passed_command_count": len(bounded),
        "all_process_groups_gone": all(
            receipt["cleanup"]["process_group_gone"] for _, receipt in bounded
        ),
        "peak_processes": max(receipt["peaks"]["processes"] for _, receipt in bounded),
        "peak_group_threads": max(
            receipt["peaks"]["group_threads"] for _, receipt in bounded
        ),
        "peak_per_process_threads": max(
            receipt["peaks"]["per_process_threads"] for _, receipt in bounded
        ),
        "profile_counts": profile_counts,
    }


def run_evidence_command(command: list[str], index: int) -> dict[str, Any]:
    if tuple(command[:2]) == ("cargo", "test"):
        profile_name = bounded_profile_for_command(command)
        profile = BOUNDED_TEST_PROFILES[profile_name]
        env = clean_subprocess_env()
        env["CARGO_BUILD_JOBS"] = "2"
        with tempfile.TemporaryDirectory(prefix=f"ferrum-g01a-command-{index:02d}-") as raw_tmp:
            root = Path(raw_tmp)
            receipt_path = root / "receipt.json"
            stdout_path = root / "stdout.log"
            stderr_path = root / "stderr.log"
            wrapper_command = [
                sys.executable,
                str(BOUNDED_COMMAND_PATH),
                "--receipt",
                str(receipt_path),
                "--stdout-log",
                str(stdout_path),
                "--stderr-log",
                str(stderr_path),
                "--cwd",
                str(REPO_ROOT),
                "--wall-timeout-seconds",
                str(profile["wall_timeout_seconds"]),
                "--max-processes",
                str(profile["max_processes"]),
                "--max-group-threads",
                str(profile["max_group_threads"]),
                "--max-per-process-threads",
                str(profile["max_per_process_threads"]),
                "--sample-interval-seconds",
                str(profile["sample_interval_seconds"]),
                "--max-sampling-errors",
                str(profile["max_sampling_errors"]),
                "--term-grace-seconds",
                str(profile["term_grace_seconds"]),
                "--",
                *command,
            ]
            wrapper = subprocess.run(
                wrapper_command,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            require(receipt_path.is_file(), f"bounded command produced no receipt: {' '.join(command)}")
            receipt = read_json(receipt_path, f"bounded receipt for command {index}")
            stdout_bytes = stdout_path.read_bytes()
            stderr_bytes = stderr_path.read_bytes()
            stdout = stdout_bytes.decode("utf-8")
            stderr = stderr_bytes.decode("utf-8")
            receipt = validate_bounded_receipt(
                receipt,
                command,
                profile_name,
                stdout_bytes,
                stderr_bytes,
                f"G01A evidence command[{index}]",
            )
            expected_wrapper_line = (
                f"BOUNDED COMMAND PASS: command_completed: {receipt_path.resolve()}"
            )
            require(
                wrapper.returncode == 0
                and wrapper.stdout.splitlines().count(expected_wrapper_line) == 1
                and not wrapper.stderr,
                f"bounded command wrapper failed: {' '.join(command)}: "
                f"{wrapper.stderr.strip() or wrapper.stdout.strip()}",
            )
            row = {
                "command": command,
                "cwd": str(REPO_ROOT),
                "env_overrides": copy.deepcopy(BOUNDED_TEST_ENV_OVERRIDES),
                "started_at": receipt["started_at"],
                "finished_at": receipt["ended_at"],
                "duration_sec": receipt["duration_seconds"],
                "returncode": receipt["rc"],
                "stdout": stdout,
                "stderr": stderr,
                "stdout_sha256": bytes_sha256(stdout_bytes),
                "stderr_sha256": bytes_sha256(stderr_bytes),
                "execution": {
                    "kind": "bounded-command",
                    "profile": profile_name,
                    "receipt": receipt,
                    "receipt_sha256": canonical_json_sha256(receipt),
                },
            }
            validate_command_execution(row, f"G01A evidence command[{index}]")
            return row

    started_at = iso_now()
    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=clean_subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    finished_at = iso_now()
    duration = time.monotonic() - started
    row = {
        "command": command,
        "cwd": str(REPO_ROOT),
        "env_overrides": {"PYTHONDONTWRITEBYTECODE": "1"},
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": duration,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_sha256": bytes_sha256(proc.stdout.encode("utf-8")),
        "stderr_sha256": bytes_sha256(proc.stderr.encode("utf-8")),
        "execution": {
            "kind": "direct",
            "profile": None,
            "receipt": None,
            "receipt_sha256": None,
        },
    }
    require(proc.returncode == 0, f"compile/test command failed: {' '.join(command)}")
    validate_command_execution(row, f"G01A evidence command[{index}]")
    return row


def listed_tests(command_row: dict[str, Any], label: str) -> set[str]:
    tests: set[str] = set()
    for line in require_string(command_row.get("stdout"), f"{label}.stdout").splitlines():
        match = re.fullmatch(r"([^:][^:]*): test", line.strip())
        if match:
            tests.add(match.group(1))
    require(tests, f"{label} listed no tests")
    return tests


def validate_test_run(row: dict[str, Any], label: str, expected_count: int) -> None:
    stdout = require_string(row.get("stdout"), f"{label}.stdout")
    summaries = re.findall(
        r"test result: ok\. (\d+) passed; 0 failed; (\d+) ignored; "
        r"(\d+) measured; (\d+) filtered out;",
        stdout,
    )
    require(len(summaries) == 1, f"{label} must contain exactly one Rust test summary")
    passed, ignored, measured, filtered = (int(value) for value in summaries[0])
    require(
        passed == expected_count,
        f"{label} passed test count mismatch: expected={expected_count} actual={passed}",
    )
    require(
        ignored == measured == filtered == 0,
        f"{label} contains ignored, measured, or filtered tests",
    )


def validate_resource_test_run(
    row: dict[str, Any], label: str, expected_count: int
) -> None:
    stdout = require_string(row.get("stdout"), f"{label}.stdout")
    summaries = [
        tuple(int(value) for value in match)
        for match in re.findall(
            r"test result: ok\. (\d+) passed; 0 failed; (\d+) ignored; "
            r"(\d+) measured; (\d+) filtered out;",
            stdout,
        )
    ]
    expected = [
        (1, 0, 0, expected_count - 1),
        (expected_count, 0, 0, 0),
    ]
    require(
        summaries == expected,
        f"{label} must contain one exact panic-isolation child summary and one parent summary: "
        f"expected={expected} actual={summaries}",
    )
    require(
        stdout.count("test resource_transaction_abandon_panic_child ... ok") == 2,
        f"{label} must run the panic-isolation test once in the parent and once in its child",
    )


def validate_admission_test_list(row: dict[str, Any], label: str) -> set[str]:
    stdout = require_string(row.get("stdout"), f"{label}.stdout")
    tests = {
        match.group(1)
        for line in stdout.splitlines()
        if (
            match := re.fullmatch(
                r"(vnext::admission::tests::[A-Za-z0-9_]+): test", line.strip()
            )
        )
    }
    require(tests, f"{label} listed no admission tests")
    require(
        tests == REQUIRED_ADMISSION_LIB_TESTS,
        f"{label} exact test set mismatch: "
        f"expected={sorted(REQUIRED_ADMISSION_LIB_TESTS)} actual={sorted(tests)}",
    )
    counts = re.findall(r"(?m)^(\d+) tests, (\d+) benchmarks$", stdout)
    require(
        counts == [(str(EXPECTED_DYNAMIC_ADMISSION_CASES), "0")],
        f"{label} must list exactly {EXPECTED_DYNAMIC_ADMISSION_CASES} tests and 0 benchmarks",
    )
    return tests


def validate_admission_test_run(row: dict[str, Any], label: str) -> None:
    stdout = require_string(row.get("stdout"), f"{label}.stdout")
    summaries = re.findall(
        r"test result: ok\. (\d+) passed; 0 failed; (\d+) ignored; "
        r"(\d+) measured; (\d+) filtered out;",
        stdout,
    )
    expected_prefix = (
        str(EXPECTED_DYNAMIC_ADMISSION_CASES),
        "0",
        "0",
    )
    require(
        len(summaries) == 1 and summaries[0][:3] == expected_prefix,
        f"{label} exact admission summary mismatch: "
        f"expected_prefix={expected_prefix} actual={summaries}",
    )


def test_command(target: str, mode: str) -> tuple[str, ...]:
    require(target in REQUIRED_TESTS_BY_TARGET, f"unknown G01A test target: {target}")
    require(mode in {"--list", "--nocapture"}, f"unsupported G01A test mode: {mode}")
    return (
        "cargo",
        "test",
        "-p",
        "ferrum-interfaces",
        "--test",
        target,
        "--",
        mode,
        TEST_THREADS_ARG,
    )


def admission_test_command(mode: str) -> tuple[str, ...]:
    require(mode in {"--list", "--nocapture"}, f"unsupported G01A admission test mode: {mode}")
    return (
        "cargo",
        "test",
        "-p",
        "ferrum-interfaces",
        "--no-default-features",
        "vnext::admission",
        "--lib",
        "--",
        mode,
        TEST_THREADS_ARG,
    )


def evidence_command_matrix() -> list[list[str]]:
    commands = [list(command) for command in QUALITY_COMMANDS]
    commands.extend(list(test_command(target, "--list")) for target in REQUIRED_TESTS_BY_TARGET)
    commands.extend(
        list(test_command(target, "--nocapture")) for target in REQUIRED_TESTS_BY_TARGET
    )
    commands.append(list(admission_test_command("--list")))
    commands.append(list(admission_test_command("--nocapture")))
    return commands


def parse_machine_proofs(target_stdout: dict[str, str]) -> dict[str, int]:
    require(
        set(target_stdout) == set(REQUIRED_TESTS_BY_TARGET),
        "machine proof target output set mismatch",
    )
    lines_by_target = {
        target: [line.strip() for line in stdout.splitlines()]
        for target, stdout in target_stdout.items()
    }
    contract_lines = lines_by_target["vnext_contract_tests"]
    exact_contract = {
        "deterministic_plan_cases": "VNEXT PLAN DETERMINISM PASS: 100/100",
        "schema_round_trip_cases": "VNEXT PLAN ROUNDTRIP PASS: 100/100",
        "breaking_version_reject_cases": "VNEXT BREAKING VERSION REJECT PASS: 100/100",
    }
    for label, line in exact_contract.items():
        require(contract_lines.count(line) == 1, f"missing or duplicate machine proof line for {label}: {line}")
    ratio_proofs = {
        "resource_transaction_cases": (
            "vnext_resource_contract_tests",
            "VNEXT RESOURCE TRANSACTION PASS",
            EXPECTED_RESOURCE_CASES,
        ),
        "fail_closed_cases": (
            "vnext_contract_tests",
            "VNEXT FAIL CLOSED PASS",
            EXPECTED_FAIL_CLOSED_CASES,
        ),
        "model_identity_cases": (
            "vnext_contract_tests",
            "VNEXT MODEL IDENTITY PASS",
            EXPECTED_MODEL_IDENTITY_CASES,
        ),
        "event_replay_v5_contract_cases": (
            "vnext_event_contract_tests",
            "VNEXT EVENT/REPLAY V5 PASS",
            EXPECTED_EVENT_REPLAY_V5_CASES,
        ),
        "device_operation_contract_cases": (
            "vnext_device_operation_contract_tests",
            "VNEXT DEVICE OPERATION PASS",
            EXPECTED_DEVICE_OPERATION_CASES,
        ),
        "operation_oracle_contract_cases": (
            "vnext_oracle_contract_tests",
            "VNEXT OPERATION ORACLE PASS",
            EXPECTED_ORACLE_CASES,
        ),
        "model_wire_contract_cases": (
            "vnext_model_wire_contract_tests",
            "VNEXT MODEL WIRE PASS",
            EXPECTED_MODEL_WIRE_CASES,
        ),
        "legacy_backend_methods_mapped": (
            "vnext_legacy_map",
            "VNEXT LEGACY MAP PASS",
            82,
        ),
    }
    ratios: dict[str, int] = {}
    for label, (target, prefix, expected) in ratio_proofs.items():
        pattern = re.compile(
            rf"^{re.escape(prefix)}: ([1-9][0-9]*)/([1-9][0-9]*)$"
        )
        lines = lines_by_target[target]
        matches = [pattern.fullmatch(line) for line in lines]
        matches = [match for match in matches if match is not None]
        require(len(matches) == 1, f"missing or duplicate machine proof line for {label}")
        passed, total = (int(value) for value in matches[0].groups())
        require(
            passed == total == expected,
            f"machine proof count changed for {label}: expected {expected}/{expected}, actual {passed}/{total}",
        )
        ratios[label] = total
    return {
        "deterministic_plan_cases": 100,
        "schema_round_trip_cases": 100,
        "breaking_version_reject_cases": 100,
        **ratios,
    }


def collect_compile_evidence(
    source: dict[str, Any],
    discovery: dict[str, list[str]],
    static_contracts: dict[str, Any],
    isolation: dict[str, Any],
    map_summary: dict[str, Any],
) -> dict[str, Any]:
    commands = evidence_command_matrix()
    rows = [run_evidence_command(command, index) for index, command in enumerate(commands)]
    rows_by_command = {tuple(row["command"]): row for row in rows}
    require(len(rows_by_command) == len(rows), "G01A evidence command matrix contains duplicates")
    tests_by_target: dict[str, set[str]] = {}
    stdout_by_target: dict[str, str] = {}
    for target, expected_tests in REQUIRED_TESTS_BY_TARGET.items():
        listed = listed_tests(
            rows_by_command[test_command(target, "--list")],
            f"{target} test list",
        )
        require(
            listed == expected_tests,
            f"{target} exact test set mismatch: expected={sorted(expected_tests)} actual={sorted(listed)}",
        )
        tests_by_target[target] = listed
        run_row = rows_by_command[test_command(target, "--nocapture")]
        if target == "vnext_resource_contract_tests":
            validate_resource_test_run(
                run_row, f"{target} tests", len(expected_tests)
            )
        else:
            validate_test_run(run_row, f"{target} tests", len(expected_tests))
        stdout_by_target[target] = run_row["stdout"]
    admission_tests = validate_admission_test_list(
        rows_by_command[admission_test_command("--list")],
        "vnext admission lib test list",
    )
    validate_admission_test_run(
        rows_by_command[admission_test_command("--nocapture")],
        "vnext admission lib tests",
    )
    require(
        len(discovery["pass_cases"]) == EXPECTED_TRYBUILD_PASS_CASES
        and len(discovery["fail_cases"]) == EXPECTED_TRYBUILD_FAIL_CASES,
        "G01A trybuild fixture count differs from the frozen 2-pass/78-fail matrix",
    )
    proofs = parse_machine_proofs(stdout_by_target)
    bounded_execution = summarize_bounded_execution(rows)
    return {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_g01a_compile_unit_trybuild_evidence",
        "source": copy.deepcopy(source),
        "command_count": len(rows),
        "commands_passed": len(rows),
        "commands": rows,
        "bounded_execution": bounded_execution,
        "tests": {
            "test_targets": {
                target: sorted(tests) for target, tests in tests_by_target.items()
            },
            "required_test_targets": {
                target: sorted(tests)
                for target, tests in REQUIRED_TESTS_BY_TARGET.items()
            },
            "admission_lib_tests": sorted(admission_tests),
            "required_admission_lib_tests": sorted(REQUIRED_ADMISSION_LIB_TESTS),
            "trybuild_pass_cases": discovery["pass_cases"],
            "trybuild_fail_cases": discovery["fail_cases"],
            "trybuild_pass_case_count": len(discovery["pass_cases"]),
            "trybuild_fail_case_count": len(discovery["fail_cases"]),
        },
        "assertions": {
            "deterministic_plan_cases": proofs["deterministic_plan_cases"],
            "schema_round_trip_cases": proofs["schema_round_trip_cases"],
            "breaking_version_reject_cases": proofs["breaking_version_reject_cases"],
            "resource_transaction_cases": proofs["resource_transaction_cases"],
            "fail_closed_cases": proofs["fail_closed_cases"],
            "model_identity_cases": proofs["model_identity_cases"],
            "event_replay_v5_contract_cases": proofs["event_replay_v5_contract_cases"],
            "device_operation_contract_cases": proofs["device_operation_contract_cases"],
            "operation_oracle_contract_cases": proofs["operation_oracle_contract_cases"],
            "model_wire_contract_cases": proofs["model_wire_contract_cases"],
            "dynamic_admission_cases": EXPECTED_DYNAMIC_ADMISSION_CASES,
            "legacy_backend_methods_mapped": proofs["legacy_backend_methods_mapped"],
            "legacy_backend_methods_unmapped": map_summary["unmapped"],
            "architecture_named_symbol_count": static_contracts["architecture_named_symbol_count"],
            "required_contract_count": static_contracts["required_contract_count"],
            "silent_success_default_count": 0,
            "unknown_fallback_success_count": 0,
            "outside_vnext_production_reference_count": isolation[
                "outside_vnext_reference_count"
            ],
        },
    }


def artifact_row(path: Path, root: Path, role: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "role": role,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def validate_g01a_manifest_shape(manifest: dict[str, Any], out: Path, checkpoint: Path) -> None:
    expected_fields = {
        "schema_version",
        "artifact_type",
        "checkpoint_id",
        "lane",
        "canonical",
        "status",
        "artifact_dir",
        "output_root",
        "artifact_count",
        "artifact_index",
        "pass_line",
        "source",
        "g00a",
        "contract_files",
        "contract_files_sha256",
        "contracts",
        "isolation",
        "legacy_mapping",
        "adr",
        "compile_evidence",
        "freshness",
        "unlocks",
        "does_not_prove",
    }
    require(set(manifest) == expected_fields, "G01A manifest field set mismatch")
    require(manifest.get("schema_version") == 1, "G01A manifest schema_version mismatch")
    require(manifest.get("artifact_type") == "runtime_vnext_g01a_contract_checkpoint_manifest", "G01A manifest artifact_type mismatch")
    require(manifest.get("checkpoint_id") == "G01A" and manifest.get("lane") == "runtime-vnext-g01a", "G01A manifest checkpoint/lane mismatch")
    require(manifest.get("canonical") is True and manifest.get("status") == "pass", "G01A manifest canonical/status mismatch")
    require(Path(require_string(manifest.get("artifact_dir"), "G01A manifest artifact_dir")).resolve() == checkpoint.resolve(), "G01A manifest artifact_dir mismatch")
    require(Path(require_string(manifest.get("output_root"), "G01A manifest output_root")).resolve() == out.resolve(), "G01A manifest output_root mismatch")
    require(manifest.get("pass_line") == f"{PASS_PREFIX}: {out.resolve()}", "G01A manifest pass_line mismatch")
    require_exact_string_set(manifest.get("unlocks"), {"G01B"}, "G01A manifest unlocks")
    require_exact_string_set(manifest.get("does_not_prove"), G01A_DOES_NOT_PROVE, "G01A manifest does_not_prove")


def snapshot_regular_file(source: Path, destination: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise CheckpointError(f"cannot open {label} as a regular non-symlink file: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    require(payload and len(payload) == metadata.st_size, f"{label} snapshot is empty or truncated")
    destination.write_bytes(payload)
    require(destination.read_bytes() == payload, f"{label} snapshot verification failed")
    return payload


def create_checkpoint(g00a_arg: Path, out_arg: Path) -> Path:
    current = source_identity()
    g00a_path = require_external_file(g00a_arg, "--g00a")
    out, checkpoint = require_external_output(out_arg)
    require(not is_within(out, g00a_path.parent), "G01A output must not be nested inside the G00a artifact")
    g00a = validate_g00a_checkpoint(g00a_path, current)

    discovery = discover_contract_paths()
    contract_rows = contract_file_rows(discovery["all"])
    static_contracts = validate_vnext_static_contracts(discovery["vnext_sources"])
    isolation = validate_repository_isolation(discovery["vnext_sources"])
    map_document = read_json(MAP_PATH, "G01A legacy contract map")
    map_summary = validate_contract_map(map_document, g00a.pop("backend_methods"))
    adr_payload = ADR_PATH.read_bytes()
    adr_summary = validate_adr(adr_payload)
    compile_evidence = collect_compile_evidence(
        current, discovery, static_contracts, isolation, map_summary
    )
    require(source_identity() == current, "source checkout changed while G01A evidence was collected")

    staging = Path(tempfile.mkdtemp(prefix=f".{CHECKPOINT_SUBDIR}.staging-", dir=out))
    try:
        snapshot_regular_file(ADR_PATH, staging / "adr.md", "G01A ADR")
        snapshot_regular_file(MAP_PATH, staging / "contract-map.json", "G01A contract map")
        write_json(staging / "compile-unit-trybuild.json", compile_evidence)
        roles = {
            "adr.md": "checked-in-g01a-architecture-decision",
            "contract-map.json": "checked-in-legacy-backend-method-map",
            "compile-unit-trybuild.json": "current-head-compile-unit-trybuild-evidence",
        }
        index = [artifact_row(staging / name, staging, role) for name, role in sorted(roles.items())]
        validate_indexed_files(staging, index, G01A_ARTIFACTS, "G01A artifact_index")
        indexed = {row["path"]: row for row in index}
        require(indexed["adr.md"]["sha256"] == file_sha256(ADR_PATH), "copied ADR digest mismatch")
        require(indexed["contract-map.json"]["sha256"] == file_sha256(MAP_PATH), "copied contract map digest mismatch")

        pass_line = f"{PASS_PREFIX}: {out}"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_g01a_contract_checkpoint_manifest",
            "checkpoint_id": CHECKPOINT_ID,
            "lane": "runtime-vnext-g01a",
            "canonical": True,
            "status": "pass",
            "artifact_dir": str(checkpoint),
            "output_root": str(out),
            "artifact_count": len(index),
            "artifact_index": index,
            "pass_line": pass_line,
            "source": current,
            "g00a": g00a,
            "contract_files": contract_rows,
            "contract_files_sha256": canonical_json_sha256(contract_rows),
            "contracts": static_contracts,
            "isolation": isolation,
            "legacy_mapping": map_summary,
            "adr": adr_summary,
            "compile_evidence": copy.deepcopy(indexed["compile-unit-trybuild.json"]),
            "freshness": {
                "checkout_clean": True,
                "g00a_outer_matches_current_head": True,
                "g00a_child_matches_current_head_and_tree": True,
                "g00a_artifact_index_verified": True,
                "contract_files_match_current_head_blobs": True,
                "compile_tests_ran_on_current_head": True,
            },
            "unlocks": ["G01B"],
            "does_not_prove": sorted(G01A_DOES_NOT_PROVE),
        }
        validate_g01a_manifest_shape(manifest, out, checkpoint)
        write_json(staging / "manifest.json", manifest)
        written_manifest = read_json(staging / "manifest.json", "written G01A manifest")
        require(written_manifest == manifest, "G01A manifest round-trip mismatch")
        validate_g01a_manifest_shape(written_manifest, out, checkpoint)
        require(source_identity() == current, "source checkout changed before G01A publication")
        os.replace(staging, checkpoint)
        staging = Path()
    finally:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)
    return out


def expect_rejected(label: str, action: Any, marker: str) -> None:
    try:
        action()
    except CheckpointError as exc:
        require(marker in str(exc), f"self-test {label} rejected for unexpected reason: {exc}")
        return
    raise CheckpointError(f"self-test mutation unexpectedly passed: {label}")


def selftest_bounded_row(command: list[str]) -> dict[str, Any]:
    profile_name = bounded_profile_for_command(command)
    stdout = "synthetic bounded stdout\n"
    stderr = ""
    receipt = {
        "schema": BOUNDED_RECEIPT_SCHEMA,
        "command": copy.deepcopy(command),
        "cwd": str(REPO_ROOT),
        "pid": 4242,
        "pgid": 4242,
        "limits": copy.deepcopy(BOUNDED_TEST_PROFILES[profile_name]),
        "peaks": {
            "processes": 1,
            "group_threads": 2,
            "per_process_threads": 2,
            "per_process_threads_pid": 4242,
        },
        "started_at": "2026-07-11T00:00:00.000Z",
        "ended_at": "2026-07-11T00:00:00.100Z",
        "duration_seconds": 0.1,
        "reason": "command_completed",
        "rc": 0,
        "status": "pass",
        "successful_samples": 2,
        "sampling_error_count": 0,
        "sampling_errors": [],
        "violation": None,
        "termination": {"signals": [], "errors": []},
        "cleanup": {"process_group_gone": True},
        "stdout": {
            "path": "/tmp/ferrum-g01a-selftest/stdout.log",
            "sha256": bytes_sha256(stdout.encode()),
            "size_bytes": len(stdout.encode()),
        },
        "stderr": {
            "path": "/tmp/ferrum-g01a-selftest/stderr.log",
            "sha256": bytes_sha256(stderr.encode()),
            "size_bytes": len(stderr.encode()),
        },
    }
    return {
        "command": copy.deepcopy(command),
        "cwd": str(REPO_ROOT),
        "env_overrides": copy.deepcopy(BOUNDED_TEST_ENV_OVERRIDES),
        "started_at": receipt["started_at"],
        "finished_at": receipt["ended_at"],
        "duration_sec": receipt["duration_seconds"],
        "returncode": 0,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_sha256": bytes_sha256(stdout.encode()),
        "stderr_sha256": bytes_sha256(stderr.encode()),
        "execution": {
            "kind": "bounded-command",
            "profile": profile_name,
            "receipt": receipt,
            "receipt_sha256": canonical_json_sha256(receipt),
        },
    }


def refresh_selftest_receipt_sha(row: dict[str, Any]) -> None:
    execution = require_object(row.get("execution"), "self-test execution")
    execution["receipt_sha256"] = canonical_json_sha256(
        require_object(execution.get("receipt"), "self-test receipt")
    )


def selftest_inventory() -> tuple[dict[str, Any], list[dict[str, str]]]:
    methods = [{"legacy_trait": "Backend", "legacy_method": f"method_{index:02d}"} for index in range(82)]
    inventory = {
        "coupling": {
            "findings": [
                {"category": "backend_trait_method", "trait": row["legacy_trait"], "symbol": row["legacy_method"]}
                for row in methods
            ]
        },
        "summary": {"coupling_count_by_category": {"backend_trait_method": 82}},
    }
    return inventory, methods


def selftest_map(methods: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_g01a_legacy_contract_map",
        "source": {"g00a_checkpoint_id": "G00a", "category": "backend_trait_method", "expected_method_count": 82},
        "mappings": [
            {
                **row,
                "classification": "versioned_operation",
                "owner": "OperationContract",
                "disposition": "map to the versioned operation catalog",
            }
            for row in methods
        ],
        "summary": {"mapped": 82, "unmapped": 0, "missing_owner": 0, "special_case": 0},
    }


def make_selftest_g00a(root: Path, current: dict[str, Any]) -> Path:
    root.mkdir(parents=True)
    root = root.resolve()
    inventory, _ = selftest_inventory()
    write_json(root / "coupling-inventory.json", inventory)
    for name in sorted(G00A_ARTIFACTS - {"coupling-inventory.json"}):
        write_json(root / name, {"selftest": name})
    index = [
        artifact_row(root / name, root, "selftest-g00a-fact")
        for name in sorted(G00A_ARTIFACTS)
    ]
    child_pass = f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {root}"
    collector = {
        "git_sha": current["git_sha"],
        "git_tree_sha": current["git_tree_sha"],
        "contracts_sha256": "c" * 64,
    }
    child = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_g00a_fact_checkpoint_manifest",
        "checkpoint_id": "G00a",
        "lane": "runtime-vnext-g00a",
        "canonical": True,
        "status": "pass",
        "dirty": False,
        "artifact_dir": str(root),
        "artifact_count": len(index),
        "artifact_index": index,
        "collector": collector,
        "git_sha": current["git_sha"],
        "git_tree_sha": current["git_tree_sha"],
        "unlocks": ["G01A"],
        "does_not_prove": sorted(G00A_DOES_NOT_PROVE),
        "pass_line": child_pass,
    }
    write_json(root / "manifest.json", child)
    for name, payload in {
        "run_gate.child.command.json": b"{}\n",
        "run_gate.child.stdout": (child_pass + "\n").encode(),
        "run_gate.child.stderr": b"",
    }.items():
        (root / name).write_bytes(payload)
    execution = [
        {
            "path": name,
            "sha256": file_sha256(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in (
            "run_gate.child.command.json",
            "run_gate.child.stdout",
            "run_gate.child.stderr",
        )
    ]
    child_bytes = (root / "manifest.json").read_bytes()
    outer = {
        "schema_version": 1,
        "lane": "vnext-g00a",
        "status": "pass",
        "child_returncode": 0,
        "error": None,
        "artifact_dir": str(root),
        "pass_line": f"FERRUM GATE vnext-g00a PASS: {root}",
        "child_pass_line": child_pass,
        "dirty_status": {"is_dirty": False, "status_short": []},
        "git_sha": current["git_sha"],
        "child_execution_artifacts": execution,
        "child_artifacts": {
            "kind": "vnext-g00a",
            "collector": collector,
            "checkpoint": {
                "id": "G00a",
                "unlocks": ["G01A"],
                "does_not_prove": sorted(G00A_DOES_NOT_PROVE),
            },
            "child_manifest": {
                "path": str(root / "manifest.json"),
                "sha256": bytes_sha256(child_bytes),
            },
            "artifact_index_sha256": canonical_json_sha256(index),
        },
    }
    write_json(root / "gate.manifest.json", outer)
    return root / "gate.manifest.json"


def retarget_selftest_g00a(root: Path) -> tuple[Path, dict[str, Any]]:
    root = root.resolve()
    child_path = root / "manifest.json"
    child = read_json(child_path, "self-test copied G00a child")
    child["artifact_dir"] = str(root)
    child["pass_line"] = f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {root}"
    write_json(child_path, child)
    outer_path = root / "gate.manifest.json"
    outer = read_json(outer_path, "self-test copied G00a outer")
    outer["artifact_dir"] = str(root)
    outer["pass_line"] = f"FERRUM GATE vnext-g00a PASS: {root}"
    outer["child_pass_line"] = child["pass_line"]
    outer["child_artifacts"]["child_manifest"] = {
        "path": str(child_path),
        "sha256": file_sha256(child_path),
    }
    write_json(outer_path, outer)
    return outer_path, outer


def run_self_test() -> None:
    _, methods = selftest_inventory()
    valid_map = selftest_map(methods)
    summary = validate_contract_map(valid_map, methods)
    require(summary["mapped"] == 82, "self-test valid map summary mismatch")
    missing = copy.deepcopy(valid_map)
    missing["mappings"].pop()
    expect_rejected("map count", lambda: validate_contract_map(missing, methods), "exactly 82")
    special = copy.deepcopy(valid_map)
    special["mappings"][0]["disposition"] = "special case"
    expect_rejected("special case", lambda: validate_contract_map(special, methods), "special-case")
    stale_methods = copy.deepcopy(methods)
    stale_methods[0]["legacy_method"] = "changed_upstream"
    expect_rejected("inventory freshness", lambda: validate_contract_map(valid_map, stale_methods), "coverage mismatch")

    valid_source = (
        "\n".join(f"pub trait {name} {{}}" for name in REQUIRED_CONTRACTS)
        + r'''
pub const EXECUTION_IDENTITY_VERSION: ContractVersion = ContractVersion::new(3, 0);
pub enum ExecutionEventKind { NodeRetired }
pub(crate) struct DynamicResourceShape { sequences: u32 }
impl DynamicResourceShape {
    pub(crate) fn new(sequences: u32) -> Self { Self { sequences } }
}
pub(crate) fn admit_dynamic_shape(_shape: DynamicResourceShape) {}
pub struct BatchWorkShape { seal: () }
struct ClaimedBackingTransaction { seal: () }
struct ParticipantNodeKey { seal: () }
pub struct BatchOperationIdentity { seal: () }
fn accepts_impl_trait(reason: impl Into<String>) { let _ = reason.into(); }
pub struct DefinitelyNotSubmittedRetryAuthority { seal: () }
impl DefinitelyNotSubmittedRetryAuthority {
    pub fn consume(self) {}
}
pub struct BatchedOperationInvocation<'a> { seal: &'a () }
pub struct OperationDispatch;
pub fn yield_now() {}
impl OperationDispatch {
    pub fn encode_and_submit(
        _batch_identity: &BatchOperationIdentity,
        active_bindings: &[TrustedActiveSequenceBinding],
    ) {
        let _ = active_bindings;
        let _invocation: Option<BatchedOperationInvocation<'_>> = None;
    }
}
'''
    )
    lexical_decoys = r'''
// pub trait DeviceRuntime {} QwenSpecialCase Any std::env::var("HOME")
/* outer Qwen Any cfg(feature = "cuda") { /* nested ferrum_engine */ } */
const COMMENT_LIKE: &str = "Qwen Any downcast_ref std::env::var cfg(feature = \"cuda\")";
const RAW_COMMENT_LIKE: &str = r###"Llama TypeId ferrum_server cfg(feature = "metal")"###;
const BYTE_COMMENT_LIKE: &[u8] = br##"DeepSeek ferrum_models env::var_os"##;
pub trait LexicalDecoySafeCombinator {
    fn required(&self) -> u64;
    fn doubled(&self) -> u64 { self.required() + self.required() }
}
'''
    with tempfile.TemporaryDirectory(prefix="ferrum-g01a-selftest-") as raw_tmp:
        root = Path(raw_tmp)
        source = root / "event.rs"
        source.write_text(valid_source + lexical_decoys, encoding="utf-8")
        original_root = globals()["REPO_ROOT"]
        try:
            globals()["REPO_ROOT"] = root
            static = validate_vnext_static_contracts(["event.rs"])
            require(static["required_contract_count"] == 9, "self-test contract count mismatch")
            require(
                static["silent_success_default_count"] == 0
                and all(value == 0 for value in static["forbidden_pattern_counts"].values()),
                "self-test lexical decoys affected static contract counts",
            )
            semantic = static["semantic_contracts"]
            require(
                semantic["execution_identity"] == {
                    "constant": "EXECUTION_IDENTITY_VERSION",
                    "major": 3,
                    "minor": 0,
                    "definition_count": 1,
                }
                and semantic["event_transition"]["required_variant_count"] == 1
                and semantic["event_transition"]["forbidden_identifier_count"] == 0
                and all(
                    count == 1 for count in semantic["definition_counts"].values()
                )
                and all(
                    semantic["multi_participant_dispatch"]["observed_markers"].values()
                )
                and all(
                    count == 0
                    for label, count in semantic[
                        "public_raw_dynamic_resource_shape"
                    ].items()
                    if label.endswith("_count")
                ),
                "self-test G01A semantic contract summary mismatch",
            )
            rejection_fixtures = (
                ("architecture name", "pub struct QwenSpecialCase;", "architecture names"),
                ("Yi architecture name", "pub struct YiModel;", "architecture names"),
                ("raw architecture name", "pub struct r#LlamaSpecialCase;", "architecture names"),
                ("Any", "use std::any::Any;", "Any/downcast"),
                ("downcast", "fn cast(value: &dyn Any) { value.downcast_ref::<u8>(); }", "Any/downcast"),
                ("product import", "use ferrum_engine::Engine;", "product/runtime import"),
                ("environment read", 'fn read() { std::env::var("HOME"); }', "hidden environment read"),
                (
                    "escaped backend cfg",
                    r'#[cfg(feature = "cu\x64a")] pub struct AcceleratorOnly;',
                    "backend feature cfg",
                ),
                (
                    "qualified Ok default",
                    "pub trait UnsafeDefault { fn maybe(&self) -> Result<(), E> { Result::Ok(()) } }",
                    "silent success defaults",
                ),
                (
                    "Default default",
                    "pub trait UnsafeDefault { fn maybe(&self) -> Value { Default::default() } }",
                    "silent success defaults",
                ),
                (
                    "const-brace signature default",
                    "pub trait UnsafeDefault { fn maybe(&self) -> [u8; { 1 }] { Default::default() } }",
                    "silent success defaults",
                ),
                (
                    "Some default",
                    "pub trait UnsafeDefault { fn maybe(&self) -> Option<u8> { Some(1) } }",
                    "silent success defaults",
                ),
                (
                    "true default",
                    "pub trait UnsafeDefault { fn maybe(&self) -> bool { true } }",
                    "silent success defaults",
                ),
            )
            for label, production_tokens, marker in rejection_fixtures:
                source.write_text(
                    valid_source + lexical_decoys + "\n" + production_tokens + "\n",
                    encoding="utf-8",
                )
                expect_rejected(
                    label,
                    lambda: validate_vnext_static_contracts(["event.rs"]),
                    marker,
                )

            def expect_semantic_rejected(
                label: str, mutated_source: str, marker: str
            ) -> None:
                source.write_text(mutated_source + lexical_decoys, encoding="utf-8")
                expect_rejected(
                    label,
                    lambda: validate_vnext_static_contracts(["event.rs"]),
                    marker,
                )

            expect_semantic_rejected(
                "execution identity 2.0",
                valid_source.replace(
                    "ContractVersion::new(3, 0)", "ContractVersion::new(2, 0)", 1
                ),
                "exactly 3.0",
            )
            expect_semantic_rejected(
                "missing NodeRetired",
                valid_source.replace("NodeRetired", "MissingNodeRetired", 1),
                "exactly one NodeRetired",
            )
            expect_semantic_rejected(
                "forbidden NodeCompleted",
                valid_source.replace(
                    "ExecutionEventKind { NodeRetired }",
                    "ExecutionEventKind { NodeRetired, NodeCompleted }",
                    1,
                ),
                "zero NodeCompleted",
            )
            for type_name, kind in G01A_SEMANTIC_TYPE_KINDS.items():
                expect_semantic_rejected(
                    f"missing semantic type {type_name}",
                    valid_source.replace(
                        f"{kind} {type_name}", f"{kind} Missing{type_name}", 1
                    ),
                    f"semantic type {type_name}",
                )
                expect_semantic_rejected(
                    f"wrong semantic type kind {type_name}",
                    valid_source.replace(
                        f"{kind} {type_name}", f"enum {type_name}", 1
                    ),
                    f"semantic type {type_name}",
                )
            expect_semantic_rejected(
                "public DNF retry field",
                valid_source.replace("{ seal: () }\nimpl DefinitelyNotSubmittedRetryAuthority", "{ pub seal: () }\nimpl DefinitelyNotSubmittedRetryAuthority", 1),
                "no public fields",
            )
            expect_semantic_rejected(
                "public DNF retry constructor",
                valid_source.replace(
                    "impl DefinitelyNotSubmittedRetryAuthority {\n    pub fn consume(self) {}",
                    "impl DefinitelyNotSubmittedRetryAuthority {\n    pub fn new() -> Self { Self { seal: () } }\n    pub fn consume(self) {}",
                    1,
                ),
                "zero public associated constructors",
            )
            dispatch_marker_mutations = {
                "batch_identity_parameter": valid_source.replace(
                    "_batch_identity: &BatchOperationIdentity",
                    "_batch_identity: &MissingBatchOperationIdentity",
                    1,
                ),
                "participant_bindings_parameter": valid_source.replace(
                    "active_bindings: &[TrustedActiveSequenceBinding]",
                    "bindings: &[TrustedActiveSequenceBinding]",
                    1,
                ),
                "participant_binding_type": valid_source.replace(
                    "active_bindings: &[TrustedActiveSequenceBinding]",
                    "active_bindings: &[MissingTrustedActiveSequenceBinding]",
                    1,
                ),
                "batched_invocation_body": valid_source.replace(
                    "Option<BatchedOperationInvocation<'_>>",
                    "Option<MissingBatchedOperationInvocation<'_>>",
                    1,
                ),
            }
            for label, mutated_source in dispatch_marker_mutations.items():
                expect_semantic_rejected(
                    f"multi-participant dispatch {label}",
                    mutated_source,
                    f"marker missing: {label}",
                )
            duplicate_dispatch = r'''
impl OperationDispatch {
    pub fn duplicate_batch_dispatch(
        _batch_identity: &BatchOperationIdentity,
        active_bindings: &[TrustedActiveSequenceBinding],
    ) {
        let _ = active_bindings;
        let _invocation: Option<BatchedOperationInvocation<'_>> = None;
    }
}
'''
            expect_semantic_rejected(
                "duplicate multi-participant dispatch method",
                valid_source + duplicate_dispatch,
                "exactly one public OperationDispatch method",
            )
            expect_semantic_rejected(
                "public DynamicResourceShape type",
                valid_source.replace(
                    "pub(crate) struct DynamicResourceShape",
                    "pub struct DynamicResourceShape",
                    1,
                ),
                "unrestricted_public_type_count",
            )
            expect_semantic_rejected(
                "public DynamicResourceShape method",
                valid_source.replace("pub(crate) fn new", "pub fn new", 1),
                "unrestricted_public_impl_method_count",
            )
            expect_semantic_rejected(
                "public DynamicResourceShape admission parameter",
                valid_source
                + "\npub fn forged_admission(_shape: DynamicResourceShape) {}\n",
                "unrestricted_public_parameter_path_count",
            )
        finally:
            globals()["REPO_ROOT"] = original_root

    valid_adr = (
        "# G01A Decision\n\n"
        "Compare aggregated capability traits with a typed operation registry. "
        "The compile-time, runtime overhead, object safety, error localization, and extension cost dimensions are recorded. "
        "Decision: use typed operation registry with small capability traits. G01A freezes contracts; G01B measures and implements.\n"
        + ("Evidence boundary and rejected alternatives. " * 40)
    ).encode("utf-8")
    validate_adr(valid_adr)
    expect_rejected("ADR topic", lambda: validate_adr(valid_adr.replace(b"object safety", b"object semantics")), "object-safety")

    contract_proofs = "\n".join(
        [
            "VNEXT PLAN DETERMINISM PASS: 100/100",
            "VNEXT PLAN ROUNDTRIP PASS: 100/100",
            "VNEXT BREAKING VERSION REJECT PASS: 100/100",
            f"VNEXT FAIL CLOSED PASS: {EXPECTED_FAIL_CLOSED_CASES}/{EXPECTED_FAIL_CLOSED_CASES}",
            f"VNEXT MODEL IDENTITY PASS: {EXPECTED_MODEL_IDENTITY_CASES}/{EXPECTED_MODEL_IDENTITY_CASES}",
        ]
    )
    proof_stdout = {target: "" for target in REQUIRED_TESTS_BY_TARGET}
    proof_stdout.update(
        {
            "vnext_contract_tests": contract_proofs,
            "vnext_resource_contract_tests": (
                f"VNEXT RESOURCE TRANSACTION PASS: "
                f"{EXPECTED_RESOURCE_CASES}/{EXPECTED_RESOURCE_CASES}\n"
            ),
            "vnext_event_contract_tests": (
                f"VNEXT EVENT/REPLAY V5 PASS: "
                f"{EXPECTED_EVENT_REPLAY_V5_CASES}/{EXPECTED_EVENT_REPLAY_V5_CASES}\n"
            ),
            "vnext_device_operation_contract_tests": (
                f"VNEXT DEVICE OPERATION PASS: "
                f"{EXPECTED_DEVICE_OPERATION_CASES}/{EXPECTED_DEVICE_OPERATION_CASES}\n"
            ),
            "vnext_oracle_contract_tests": (
                f"VNEXT OPERATION ORACLE PASS: {EXPECTED_ORACLE_CASES}/{EXPECTED_ORACLE_CASES}\n"
            ),
            "vnext_model_wire_contract_tests": (
                f"VNEXT MODEL WIRE PASS: "
                f"{EXPECTED_MODEL_WIRE_CASES}/{EXPECTED_MODEL_WIRE_CASES}\n"
            ),
            "vnext_legacy_map": "VNEXT LEGACY MAP PASS: 82/82\n",
        }
    )
    proofs = parse_machine_proofs(proof_stdout)
    require(
        proofs["resource_transaction_cases"] == EXPECTED_RESOURCE_CASES
        and proofs["fail_closed_cases"] == EXPECTED_FAIL_CLOSED_CASES
        and proofs["model_identity_cases"] == EXPECTED_MODEL_IDENTITY_CASES
        and proofs["event_replay_v5_contract_cases"]
        == EXPECTED_EVENT_REPLAY_V5_CASES
        and proofs["device_operation_contract_cases"]
        == EXPECTED_DEVICE_OPERATION_CASES
        and proofs["operation_oracle_contract_cases"] == EXPECTED_ORACLE_CASES
        and proofs["model_wire_contract_cases"] == EXPECTED_MODEL_WIRE_CASES,
        "self-test machine proof parse mismatch",
    )
    missing_roundtrip = copy.deepcopy(proof_stdout)
    missing_roundtrip["vnext_contract_tests"] = missing_roundtrip[
        "vnext_contract_tests"
    ].replace("VNEXT PLAN ROUNDTRIP PASS: 100/100\n", "")
    expect_rejected(
        "missing proof line",
        lambda: parse_machine_proofs(missing_roundtrip),
        "schema_round_trip_cases",
    )
    missing_legacy = copy.deepcopy(proof_stdout)
    missing_legacy["vnext_legacy_map"] = ""
    expect_rejected(
        "missing legacy proof",
        lambda: parse_machine_proofs(missing_legacy),
        "legacy_backend_methods_mapped",
    )
    resource_drift = copy.deepcopy(proof_stdout)
    resource_drift["vnext_resource_contract_tests"] = resource_drift[
        "vnext_resource_contract_tests"
    ].replace(
        f"{EXPECTED_RESOURCE_CASES}/{EXPECTED_RESOURCE_CASES}",
        f"{EXPECTED_RESOURCE_CASES - 1}/{EXPECTED_RESOURCE_CASES - 1}",
    )
    expect_rejected(
        "resource proof count drift",
        lambda: parse_machine_proofs(resource_drift),
        "machine proof count changed for resource_transaction_cases",
    )
    missing_event = copy.deepcopy(proof_stdout)
    missing_event["vnext_event_contract_tests"] = ""
    expect_rejected(
        "event proof missing",
        lambda: parse_machine_proofs(missing_event),
        "event_replay_v5_contract_cases",
    )
    commands = evidence_command_matrix()
    require(
        len(commands) == 23 and len({tuple(command) for command in commands}) == 23,
        "self-test G01A command matrix must contain 23 unique commands",
    )
    cargo_test_commands = [
        command for command in commands if tuple(command[:2]) == ("cargo", "test")
    ]
    require(
        len(cargo_test_commands) == BOUNDED_TEST_COMMAND_COUNT
        and all(command.count(TEST_THREADS_ARG) == 1 for command in cargo_test_commands),
        "self-test G01A command matrix must contain 20 single-threaded cargo tests",
    )
    require(
        bounded_profile_for_command(admission_test_command("--list")) == "admission"
        and bounded_profile_for_command(admission_test_command("--nocapture"))
        == "admission",
        "self-test admission commands must use the bounded cold-compile profile",
    )
    resource_summary_row = {
        "stdout": "\n".join(
            [
                "test resource_transaction_abandon_panic_child ... ok",
                "test resource_transaction_abandon_panic_child ... ok",
                "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 6 filtered out;",
                "test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;",
            ]
        )
    }
    validate_resource_test_run(
        resource_summary_row,
        "self-test resource child summary",
        len(REQUIRED_RESOURCE_TESTS),
    )
    missing_resource_child = copy.deepcopy(resource_summary_row)
    missing_resource_child["stdout"] = missing_resource_child["stdout"].replace(
        "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 6 filtered out;\n",
        "",
    )
    expect_rejected(
        "missing resource panic-isolation child summary",
        lambda: validate_resource_test_run(
            missing_resource_child,
            "self-test missing resource child summary",
            len(REQUIRED_RESOURCE_TESTS),
        ),
        "must contain one exact panic-isolation child summary",
    )
    bounded_row = selftest_bounded_row(
        list(test_command("vnext_resource_contract_tests", "--nocapture"))
    )
    validate_command_execution(bounded_row, "self-test valid bounded row")

    missing_receipt = copy.deepcopy(bounded_row)
    missing_receipt["execution"]["receipt"] = None
    missing_receipt["execution"]["receipt_sha256"] = None
    expect_rejected(
        "missing bounded receipt",
        lambda: validate_command_execution(missing_receipt, "self-test missing receipt"),
        "receipt must be a JSON object",
    )

    cleanup_false = copy.deepcopy(bounded_row)
    cleanup_false["execution"]["receipt"]["cleanup"]["process_group_gone"] = False
    refresh_selftest_receipt_sha(cleanup_false)
    expect_rejected(
        "bounded cleanup false",
        lambda: validate_command_execution(cleanup_false, "self-test cleanup false"),
        "process group cleanup failed",
    )

    peak_exceeded = copy.deepcopy(bounded_row)
    peak_exceeded["execution"]["receipt"]["peaks"]["group_threads"] = (
        BOUNDED_TEST_PROFILES["resource"]["max_group_threads"] + 1
    )
    refresh_selftest_receipt_sha(peak_exceeded)
    expect_rejected(
        "bounded peak exceeded",
        lambda: validate_command_execution(peak_exceeded, "self-test peak exceeded"),
        "peak exceeds",
    )

    sampling_error = copy.deepcopy(bounded_row)
    sampling_error["execution"]["receipt"]["sampling_error_count"] = 1
    sampling_error["execution"]["receipt"]["sampling_errors"] = [
        {"at": "2026-07-11T00:00:00.050Z", "type": "Synthetic", "error": "sample"}
    ]
    refresh_selftest_receipt_sha(sampling_error)
    expect_rejected(
        "bounded sampling error",
        lambda: validate_command_execution(sampling_error, "self-test sampling error"),
        "contains sampling errors",
    )

    command_mismatch = copy.deepcopy(bounded_row)
    command_mismatch["execution"]["receipt"]["command"].append("--unexpected")
    refresh_selftest_receipt_sha(command_mismatch)
    expect_rejected(
        "bounded command mismatch",
        lambda: validate_command_execution(command_mismatch, "self-test command mismatch"),
        "receipt command mismatch",
    )

    missing_test_threads = copy.deepcopy(bounded_row)
    missing_test_threads["command"].remove(TEST_THREADS_ARG)
    missing_test_threads["execution"]["receipt"]["command"].remove(TEST_THREADS_ARG)
    refresh_selftest_receipt_sha(missing_test_threads)
    expect_rejected(
        "bounded missing test threads",
        lambda: validate_command_execution(
            missing_test_threads, "self-test missing test threads"
        ),
        "must contain exactly one --test-threads=1",
    )
    validate_test_driver_set(set(REQUIRED_TEST_DRIVERS))
    expect_rejected(
        "missing legacy driver",
        lambda: validate_test_driver_set(
            REQUIRED_TEST_DRIVERS
            - {"crates/ferrum-interfaces/tests/vnext_legacy_map.rs"}
        ),
        "vnext_legacy_map.rs",
    )
    validate_test_run(
        {"stdout": "test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;"},
        "self-test clean run",
        3,
    )
    expect_rejected(
        "partial test run",
        lambda: validate_test_run(
            {
                "stdout": "test result: ok. 2 passed; 0 failed; 0 ignored; "
                "0 measured; 0 filtered out;"
            },
            "self-test partial run",
            3,
        ),
        "passed test count mismatch",
    )
    expect_rejected(
        "ignored test",
        lambda: validate_test_run(
            {"stdout": "test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out;"},
            "self-test ignored run",
            2,
        ),
        "ignored, measured",
    )
    admission_list_stdout = "\n".join(
        [f"{name}: test" for name in sorted(REQUIRED_ADMISSION_LIB_TESTS)]
        + [f"{EXPECTED_DYNAMIC_ADMISSION_CASES} tests, 0 benchmarks"]
    )
    validate_admission_test_list(
        {"stdout": admission_list_stdout},
        "self-test admission list",
    )
    validate_admission_test_run(
        {
            "stdout": (
                f"test result: ok. {EXPECTED_DYNAMIC_ADMISSION_CASES} passed; 0 failed; "
                "0 ignored; 0 measured; 2 filtered out;"
            )
        },
        "self-test admission run",
    )
    validate_admission_test_run(
        {
            "stdout": (
                f"test result: ok. {EXPECTED_DYNAMIC_ADMISSION_CASES} passed; 0 failed; "
                "0 ignored; 0 measured; 97 filtered out;"
            )
        },
        "self-test admission run with unrelated lib tests",
    )
    expect_rejected(
        "admission 38-case mutation",
        lambda: validate_admission_test_run(
            {
                "stdout": (
                    f"test result: ok. {EXPECTED_DYNAMIC_ADMISSION_CASES - 1} passed; 0 failed; "
                    "0 ignored; 0 measured; 2 filtered out;"
                )
            },
            "self-test admission 38-case run",
        ),
        "exact admission summary mismatch",
    )
    expect_rejected(
        "admission ignored-count mutation",
        lambda: validate_admission_test_run(
            {
                "stdout": (
                    f"test result: ok. {EXPECTED_DYNAMIC_ADMISSION_CASES} passed; 0 failed; "
                    "1 ignored; 0 measured; 2 filtered out;"
                )
            },
            "self-test admission ignored-count run",
        ),
        "exact admission summary mismatch",
    )

    fixture_source = {
        "git_sha": "a" * 40,
        "git_tree_sha": "b" * 40,
        "dirty": False,
        "status_short": [],
    }
    with tempfile.TemporaryDirectory(prefix="ferrum-g01a-g00a-selftest-") as raw_tmp:
        fixtures = Path(raw_tmp)
        valid_root = fixtures / "valid"
        valid_outer = make_selftest_g00a(valid_root, fixture_source)
        binding = validate_g00a_checkpoint(valid_outer, fixture_source)
        require(binding["coupling_inventory"]["backend_trait_method_count"] == 82, "self-test G00a binding count mismatch")

        stale_root = fixtures / "stale"
        shutil.copytree(valid_root, stale_root)
        stale_outer_path, stale_outer = retarget_selftest_g00a(stale_root)
        stale_outer["git_sha"] = "d" * 40
        write_json(stale_outer_path, stale_outer)
        expect_rejected(
            "stale G00a source",
            lambda: validate_g00a_checkpoint(stale_outer_path, fixture_source),
            "stale against current HEAD",
        )

        tamper_root = fixtures / "artifact-tamper"
        shutil.copytree(valid_root, tamper_root)
        tamper_outer_path, _ = retarget_selftest_g00a(tamper_root)
        write_json(tamper_root / "models.catalog.json", {"tampered": True})
        expect_rejected(
            "G00a artifact tamper",
            lambda: validate_g00a_checkpoint(tamper_outer_path, fixture_source),
            "size mismatch",
        )

        digest_root = fixtures / "index-digest"
        shutil.copytree(valid_root, digest_root)
        digest_outer_path, digest_outer = retarget_selftest_g00a(digest_root)
        digest_outer["child_artifacts"]["artifact_index_sha256"] = "e" * 64
        write_json(digest_outer_path, digest_outer)
        expect_rejected(
            "G00a artifact-index digest",
            lambda: validate_g00a_checkpoint(digest_outer_path, fixture_source),
            "artifact index digest mismatch",
        )

    require_exact_string_set(["G01B"], {"G01B"}, "self-test G01A unlocks")
    expect_rejected(
        "forbidden unlock",
        lambda: require_exact_string_set(["G01B", "runtime"], {"G01B"}, "self-test G01A unlocks"),
        "mismatch",
    )
    expect_rejected(
        "missing does_not_prove",
        lambda: require_exact_string_set(
            sorted(G01A_DOES_NOT_PROVE - {"performance"}),
            G01A_DOES_NOT_PROVE,
            "self-test G01A does_not_prove",
        ),
        "mismatch",
    )

    goal_relative = GOAL_PATH.relative_to(REPO_ROOT).as_posix()
    goal_row = {
        "path": goal_relative,
        "git_blob": run_git(["rev-parse", f"HEAD:{goal_relative}"]),
        "sha256": file_sha256(GOAL_PATH),
        "size_bytes": GOAL_PATH.stat().st_size,
    }
    validate_contract_file_rows([goal_row], {goal_relative})
    bad_blob = copy.deepcopy(goal_row)
    bad_blob["git_blob"] = "f" * len(goal_row["git_blob"])
    expect_rejected(
        "current Git blob mismatch",
        lambda: validate_contract_file_rows([bad_blob], {goal_relative}),
        "Git blob mismatch",
    )

    with tempfile.TemporaryDirectory(prefix="ferrum-g01a-copy-selftest-") as raw_tmp:
        copy_root = Path(raw_tmp)
        for name in G01A_ARTIFACTS:
            (copy_root / name).write_text(f"clean {name}\n", encoding="utf-8")
        copy_index = [
            artifact_row(copy_root / name, copy_root, "selftest")
            for name in sorted(G01A_ARTIFACTS)
        ]
        validate_indexed_files(copy_root, copy_index, G01A_ARTIFACTS, "self-test copied artifacts")
        (copy_root / "adr.md").write_text("bad!! adr.md\n", encoding="utf-8")
        expect_rejected(
            "copied artifact tamper",
            lambda: validate_indexed_files(
                copy_root, copy_index, G01A_ARTIFACTS, "self-test copied artifacts"
            ),
            "SHA256 mismatch",
        )

    with tempfile.TemporaryDirectory(prefix="ferrum-g01a-manifest-selftest-") as raw_tmp:
        manifest_out = Path(raw_tmp).resolve()
        manifest_checkpoint = manifest_out / CHECKPOINT_SUBDIR
        manifest = {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_g01a_contract_checkpoint_manifest",
            "checkpoint_id": "G01A",
            "lane": "runtime-vnext-g01a",
            "canonical": True,
            "status": "pass",
            "artifact_dir": str(manifest_checkpoint),
            "output_root": str(manifest_out),
            "artifact_count": 3,
            "artifact_index": [],
            "pass_line": f"{PASS_PREFIX}: {manifest_out}",
            "source": {},
            "g00a": {},
            "contract_files": [],
            "contract_files_sha256": "0" * 64,
            "contracts": {},
            "isolation": {},
            "legacy_mapping": {},
            "adr": {},
            "compile_evidence": {},
            "freshness": {},
            "unlocks": ["G01B"],
            "does_not_prove": sorted(G01A_DOES_NOT_PROVE),
        }
        validate_g01a_manifest_shape(manifest, manifest_out, manifest_checkpoint)
        bad_checkpoint = copy.deepcopy(manifest)
        bad_checkpoint["checkpoint_id"] = "G01"
        expect_rejected(
            "child checkpoint field",
            lambda: validate_g01a_manifest_shape(
                bad_checkpoint, manifest_out, manifest_checkpoint
            ),
            "checkpoint/lane mismatch",
        )
        extra_field = copy.deepcopy(manifest)
        extra_field["performance_ready"] = True
        expect_rejected(
            "child extra claim field",
            lambda: validate_g01a_manifest_shape(
                extra_field, manifest_out, manifest_checkpoint
            ),
            "field set mismatch",
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g00a", type=Path, help="fresh outer vnext-g00a gate.manifest.json")
    parser.add_argument("--out", type=Path, help="fresh external output root")
    parser.add_argument("--self-test", action="store_true", help="run deterministic acceptance and mutation tests")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.self_test:
            require(args.g00a is None and args.out is None, "--self-test does not accept canonical checkpoint arguments")
            run_self_test()
            print(SELFTEST_PASS_LINE)
            return 0
        require(args.g00a is not None, "--g00a is required")
        require(args.out is not None, "--out is required")
        out = create_checkpoint(args.g00a, args.out)
        print(f"{PASS_PREFIX}: {out}")
        return 0
    except (CheckpointError, OSError) as exc:
        print(f"G01A checkpoint ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
