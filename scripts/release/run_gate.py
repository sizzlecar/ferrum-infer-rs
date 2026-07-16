#!/usr/bin/env python3
"""Unified release gate runner.

This is the product-facing release entrypoint. It delegates to the existing
source, binary, and summary validators, then writes one normalized
`gate.manifest.json` and prints the unified PASS line.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_LANES = {
    "unit": "unit",
    "metal": "metal",
    "cuda-smoke": "cuda-smoke",
    "cuda-full": "cuda-full",
    "cuda-llama-dense": "cuda-llama-dense",
    "cuda-llama33-70b-4bit-2x4090-smoke": "cuda-llama33-70b-4bit-2x4090-smoke",
    "cuda-llama33-70b-4bit-2x4090": "cuda-llama33-70b-4bit-2x4090",
}
BINARY_LANES = {
    "metal-tarball",
    "cuda-tarball",
    "homebrew-metal",
    "homebrew-cuda-fetch",
}
LANES = (
    "vnext-g00a",
    "vnext-g00f",
    "vnext-g00",
    "vnext-g01a",
    "vnext-s1-cuda",
    "vnext-s1-cuda-capacity",
    "unit",
    "metal",
    "cuda-smoke",
    "cuda-full",
    "cuda-llama-dense",
    "cuda-llama33-70b-4bit-2x4090-smoke",
    "cuda-llama33-70b-4bit-2x4090",
    "metal-tarball",
    "cuda-tarball",
    "homebrew-metal",
    "homebrew-cuda-fetch",
    "release-summary",
    "release-complete",
)
ENV_ALLOW_PREFIXES = ("FERRUM_",)
ENV_ALLOW_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "LD_LIBRARY_PATH",
    "RUST_LOG",
)
SECRET_KEY_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SAFETENSORS_SHARD_RE = re.compile(
    r"-(\d{5,6})-of-(\d{5,6})\.safetensors$"
)
VNEXT_FROZEN_LEGACY_SHA = "cff4c47765ef3259b8a04890187d99c60da86394"
VNEXT_S0A_PUBLIC_API_ADDED_SHA256 = (
    "3a6e1f97b7cefba2c9792a7dd46955ef20c178535c104e82f9efd1070e2380a9"
)
VNEXT_G00_FULL_SELFTEST_PASS = (
    "FERRUM RUNTIME VNEXT G00 BASELINE FULL SELFTEST PASS"
)
VNEXT_G00_SELFTEST_SUMMARY_PREFIX = (
    "FERRUM RUNTIME VNEXT G00 BASELINE SELFTEST SUMMARY:"
)
VNEXT_G00_REDTEAM_MUTATION_COUNT = 115
VNEXT_G00_REDTEAM_MUTATION_MATRIX_SHA256 = "54a1cb0ffd4742f26c416b1c40f13803840d65fe7c7ba51c4866725fca9db3eb"
VNEXT_G00_REDTEAM_MUTATION_NAMES = (
    "dirty",
    "stale",
    "preset-policy-drift",
    "expectations-lock-sha",
    "preset-semantic-chain-forgery",
    "chat-template-chain-forgery",
    "generation-config-chain-forgery",
    "llama-official-chain-forgery",
    "model-resolution-revision",
    "model-resolution-resolver-sha",
    "model-resolution-shard-index",
    "model-resolution-expected-sha-non-lfs",
    "model-resolution-extra-metadata-request",
    "model-resolution-lfs-tree-classification",
    "model-resolution-extra-source-request",
    "model-resolution-sha-source-enum",
    "model-resolution-model-url-shape",
    "model-resolution-official-tree-binding",
    "model-resolution-license-duplicate",
    "hardware-derived-fingerprint",
    "hardware-probe-command",
    "hardware-probe-raw-derivation",
    "historical-identical-mutation",
    "historical-missing-signature",
    "historical-success-returncode",
    "model-sha",
    "cuda-primary-blocked",
    "scenario-no-ferrum-argv",
    "scenario-missing-tools",
    "scenario-missing-schema",
    "scenario-missing-utf8",
    "scenario-missing-thinking",
    "scenario-missing-cancel",
    "scenario-artifact-sha",
    "scenario-fake-pass",
    "cross-hardware",
    "repeat-count",
    "expected-request-accounting",
    "expected-requests-absolute",
    "expected-requests-missing",
    "errors",
    "usage",
    "ab-identity-swap",
    "ab-request-model-alias",
    "duplicate-server-session",
    "server-session-same-lane-overlap",
    "cross-lane-session-id-conflict",
    "server-cell-window-overlap",
    "report-outside-cell-window",
    "server-process-start-marker",
    "server-process-receipt-env",
    "ready-probe-returncode",
    "loaded-model-probe",
    "external-identity-probe",
    "external-active-cap-argv",
    "external-server-bind-argv",
    "external-vllm-positional-model",
    "performance-collector-plan",
    "performance-collector-config-fingerprint",
    "server-effective-config-model",
    "server-product-config-cap",
    "server-effective-config-argv",
    "benchmark-client-tree-binding",
    "benchmark-client-rust-allowlist",
    "bench-canonical-argv",
    "bench-http-connection-argv",
    "dataset-sha",
    "tokenizer-sha",
    "config-sha",
    "hardware-fingerprint",
    "active-cap",
    "observed-active",
    "zero-observed-active",
    "resource-observation-pid",
    "resource-observation-process-start",
    "resource-summary-forgery",
    "resource-http-process-probe",
    "resource-http-exit-reason",
    "raw-report-sha",
    "raw-report-metric",
    "raw-report-usage",
    "raw-report-quality",
    "itl-evidence-missing",
    "itl-evidence-cardinality",
    "itl-source-forged",
    "itl-usage-event-claimed-eligible",
    "itl-interval-claimed-eligible",
    "itl-coalesced-claimed-eligible",
    "itl-repeat-counts-forged",
    "itl-repeat-intervals-forged",
    "itl-ineligible-partial-percentiles",
    "swap-growth",
    "duplicate-repeat-ordinal",
    "warmup-error",
    "bench-thinking-payload",
    "bench-env-hash",
    "bench-http-connection-env",
    "run-real-command",
    "run-process-receipt-missing",
    "run-session-global-overlap",
    "run-command-window-binding",
    "inventory-source-coverage",
    "inventory-review-binding",
    "artifact-index-empty-file",
    "artifact-index-symlink",
    "build-real-command",
    "build-raw-summary",
    "build-finished-failure",
    "build-content-evidence",
    "build-repair-base-binding",
    "build-native-log-derivation",
    "build-restore-fresh",
    "build-restore-binary",
    "build-restore-mtime",
    "malformed-artifact-type",
)
VNEXT_G00A_DOES_NOT_PROVE = {"G00", "G01B", "model_migration", "performance", "release"}
VNEXT_G00A_CONTRACT_PATHS = {
    "docs/goals/runtime-vnext-0.8.0-2026-07-10/G00_BASELINE.md",
    "docs/goals/runtime-vnext-0.8.0-2026-07-10/G01_CORE_CONTRACTS.md",
    "docs/goals/runtime-vnext-0.8.0-2026-07-10/GOAL.md",
    "docs/goals/runtime-vnext-0.8.0-2026-07-10/MODEL_MATRIX.md",
    "scripts/release/configs/runtime_vnext_generation_presets.json",
    "scripts/release/configs/runtime_vnext_historical_bugs.json",
    "scripts/release/configs/runtime_vnext_inventory_review.json",
    "scripts/release/configs/runtime_vnext_models.json",
    "scripts/release/runtime_vnext_g00a_checkpoint.py",
    "scripts/release/runtime_vnext_inventory.py",
    "scripts/release/runtime_vnext_model_resolver.py",
}
VNEXT_G01A_REQUIRED_CONTRACTS = {
    "DeviceRuntime",
    "OperationContract",
    "ModelFamilyProvider",
    "ModelProgram",
    "ExecutionPlanner",
    "ExecutionPlan",
    "ResourceTransaction",
    "ExecutionEventSink",
    "ResolvedModelPlan",
}
VNEXT_G01A_EXECUTION_IDENTITY_VERSION = (3, 0)
VNEXT_G01A_EVENT_REQUIRED_TRANSITION = "NodeRetired"
VNEXT_G01A_EVENT_FORBIDDEN_TRANSITION = "NodeCompleted"
VNEXT_G01A_SEMANTIC_TYPE_KINDS = {
    "BatchWorkShape": "struct",
    "ClaimedBackingTransaction": "struct",
    "ParticipantNodeKey": "struct",
    "BatchOperationIdentity": "struct",
    "DefinitelyNotSubmittedRetryAuthority": "struct",
    "BatchedOperationInvocation": "struct",
}
VNEXT_G01A_DNF_RETRY_AUTHORITY_TYPE = "DefinitelyNotSubmittedRetryAuthority"
VNEXT_G01A_MULTIPARTICIPANT_DISPATCH_MARKERS = {
    "batch_identity_parameter": "BatchOperationIdentity",
    "participant_bindings_parameter": "active_bindings",
    "participant_binding_type": "TrustedActiveSequenceBinding",
    "batched_invocation_body": "BatchedOperationInvocation",
}
VNEXT_G01A_REQUIRED_UNIT_TESTS = {
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
VNEXT_G01A_REQUIRED_CORE_TESTS_BY_TARGET = {
    "vnext_planning_resource_contract_tests": {
        "operation_resource_contract_requires_explicit_presence_and_alignment",
        "execution_memory_is_core_owned_and_exact",
        "minimum_runnable_sums_lifetime_minima_and_sequential_invocation_peak",
        "runtime_capacity_reserve_and_concurrency_are_typed_planning_inputs",
        "maximum_active_sequence_ceiling_is_nonzero_and_o_graph",
        "theoretical_ceiling_over_u64_is_canonical_evidence_not_capacity_policy",
        "state_capacity_demand_is_explicit_checked_and_wire_closed",
        "provider_workspace_formulas_are_actual_shape_checked_and_wire_closed",
    },
    "vnext_plan_wire_contract_tests": {
        "dynamic_descriptor_and_memory_plan_standalone_wire_are_checked",
        "execution_plan_is_deterministic_100_of_100",
        "execution_plan_schema_round_trip_100_of_100",
        "breaking_schema_versions_are_rejected_100_of_100",
        "forged_self_hashed_plan_is_rejected_by_semantic_rebuild",
        "externally_trusted_node_resolution_cannot_be_replaced_by_wire_data",
        "self_consistent_wire_resource_estimate_and_memory_mutation_is_rejected",
        "self_consistent_wire_provider_selection_is_rejected",
        "typed_planning_registry_invokes_real_contract_and_estimator_once",
    },
    "vnext_provider_selection_contract_tests": {
        "provider_implementation_fingerprint_is_plan_hashed_and_revalidated",
        "planning_registry_missing_duplicate_and_mismatched_entries_fail_before_plan",
        "provider_raw_estimate_identity_input_and_output_are_revalidated_by_core",
        "preferred_provider_is_only_a_core_validated_preference",
        "storage_incompatible_preference_falls_back_with_canonical_evidence",
    },
    "vnext_weight_layout_contract_tests": {
        "physical_weight_layout_tree_accepts_dense_fixture",
        "physical_weight_layout_tree_accepts_grouped_quantized_axis_index_fixture",
        "physical_weight_layout_tree_accepts_recursive_quantized_expert_stack_fixture",
        "weight_schema_order_is_normalized_before_fingerprinting",
        "blocked_weight_layout_requires_explicit_exact_or_zero_fill_padding",
        "physical_weight_layout_tree_rejects_invalid_shape_reuse_padding_overflow_and_limits",
        "blocked_tensor_storage_requires_explicit_exact_or_zero_fill_padding",
        "model_program_rejects_duplicate_declared_outputs",
    },
    "vnext_resolution_contract_tests": {
        "resolved_model_plan_closes_all_contract_links",
        "resolved_model_plan_initial_construction_requires_verified_evidence_context",
        "resolved_source_evidence_rejects_raw_bytes_and_provenance_tampering",
        "resolved_source_parser_identity_and_determinism_are_enforced",
        "resolved_external_device_catalog_runtime_and_node_resolution_are_exact",
        "resolved_model_family_identity_is_unique_and_fail_closed",
        "resolution_source_matrix_rejects_forbidden_binding_before_plan",
        "unknown_inputs_fail_closed",
        "provider_catalog_and_reference_oracle_fail_closed",
        "prepared_family_wire_requires_typed_registry_reconstruction",
        "mandatory_object_safe_contracts_accept_trait_objects",
    },
    "vnext_execution_graph_contract_tests": {
        "execution_alias_must_alias_builds_exact_equivalence_and_single_allocation",
        "execution_alias_may_alias_supports_distinct_or_exact_storage",
        "execution_alias_rejects_partial_and_wrong_input_overlap",
        "execution_alias_rejects_overwrite_before_last_consumer",
        "execution_state_effect_graph_orders_raw_war_waw",
        "execution_state_read_only_nodes_remain_independent",
        "execution_alias_effect_wire_mutations_are_rejected",
    },
    "vnext_source_audit_contract_tests": {
        "generic_contracts_have_zero_architecture_names",
        "silent_success_defaults_are_absent",
        "failure_envelope_wire_limit_precedes_deserialization",
    },
}
VNEXT_G01A_REQUIRED_RESOURCE_TESTS_BY_TARGET = {
    "vnext_resource_capacity_contract_tests": {
        "resource_capacity_concurrency_is_bounded",
        "runtime_implementation_authority_is_exact",
    },
    "vnext_resource_transaction_lifecycle_tests": {
        "transaction_lifecycle_contracts_are_exhaustive",
    },
    "vnext_resource_transaction_evidence_tests": {
        "resource_transaction_abandon_panic_child",
        "transaction_evidence_contracts_are_exhaustive",
    },
    "vnext_resource_sequence_activation_tests": {
        "sequence_activation_contracts_are_exhaustive",
    },
    "vnext_resource_sequence_recovery_tests": {
        "sequence_recovery_contracts_are_exhaustive",
    },
    "vnext_resource_recovery_authority_tests": {
        "recovery_authority_contracts_are_exhaustive",
    },
    "vnext_resource_runtime_close_tests": {
        "closing_root_rejects_every_parent_to_child_derivation",
        "plan_runtime_close_recovery_is_ownership_safe",
        "poisoned_bound_stream_retains_sequence_until_stream_drop",
        "sequence_owner_drop_defers_blocking_backend_recovery",
    },
}
VNEXT_G01A_RESOURCE_PANIC_ISOLATION_TARGET = "vnext_resource_transaction_evidence_tests"
VNEXT_G01A_RESOURCE_PROOF_LINES = {
    "vnext_resource_capacity_contract_tests": (
        ("VNEXT RUNTIME IMPLEMENTATION AUTHORITY PASS", 13),
        ("VNEXT RESOURCE CAPACITY THREAD BOUND PASS", 20),
    ),
    "vnext_resource_transaction_lifecycle_tests": (
        ("VNEXT TRANSACTION LIFECYCLE PASS", 70),
    ),
    "vnext_resource_transaction_evidence_tests": (
        ("VNEXT TRANSACTION EVIDENCE PASS", 69),
    ),
    "vnext_resource_sequence_activation_tests": (
        ("VNEXT SEQUENCE ACTIVATION PASS", 53),
    ),
    "vnext_resource_sequence_recovery_tests": (
        ("VNEXT SEQUENCE RECOVERY PASS", 48),
    ),
    "vnext_resource_recovery_authority_tests": (
        ("VNEXT RECOVERY AUTHORITY PASS", 38),
    ),
    "vnext_resource_runtime_close_tests": (),
}
VNEXT_G01A_REQUIRED_EVENT_TESTS_BY_TARGET = {
    "vnext_event_execution_contract_tests": {"vnext_event_execution_contract"},
    "vnext_event_sink_contract_tests": {"vnext_event_sink_contract"},
    "vnext_event_resource_pool_contract_tests": {
        "vnext_event_resource_pool_contract"
    },
    "vnext_event_recovery_contract_tests": {"vnext_event_recovery_contract"},
    "vnext_event_replay_contract_tests": {"vnext_event_replay_contract"},
}
VNEXT_G01A_EVENT_PROOF_LINES = {
    "vnext_event_execution_contract_tests": ("VNEXT EVENT EXECUTION PASS", 54),
    "vnext_event_sink_contract_tests": ("VNEXT EVENT SINK PASS", 13),
    "vnext_event_resource_pool_contract_tests": (
        "VNEXT EVENT RESOURCE POOL PASS",
        27,
    ),
    "vnext_event_recovery_contract_tests": ("VNEXT EVENT RECOVERY PASS", 20),
    "vnext_event_replay_contract_tests": ("VNEXT EVENT REPLAY PASS", 47),
}
VNEXT_G01A_REQUIRED_RESOLUTION_LIMITS_TESTS = {
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
VNEXT_G01A_REQUIRED_DEVICE_OPERATION_TESTS_BY_TARGET = {
    "vnext_device_operation_batch_contract_tests": {
        "thirty_two_participant_dispatch_is_one_physical_submission"
    },
    "vnext_device_operation_cancel_contract_tests": {
        "device_operation_cancel_contract_is_exhaustive"
    },
    "vnext_device_operation_completion_contract_tests": {
        "completion_reaper_drop_defers_blocking_backend_recovery",
        "device_operation_completion_contract_is_exhaustive",
    },
    "vnext_device_operation_dispatch_contract_tests": {
        "device_operation_dispatch_contract_is_exhaustive"
    },
    "vnext_device_operation_legacy_authority_contract_tests": {
        "device_operation_legacy_authority_contract_is_exhaustive"
    },
}
VNEXT_G01A_DEVICE_OPERATION_PROOF_LINES = {
    "vnext_device_operation_cancel_contract_tests": (
        "VNEXT DEVICE OPERATION CANCEL PASS",
        16,
    ),
    "vnext_device_operation_completion_contract_tests": (
        "VNEXT DEVICE OPERATION COMPLETION PASS",
        200,
    ),
    "vnext_device_operation_dispatch_contract_tests": (
        "VNEXT DEVICE OPERATION DISPATCH PASS",
        70,
    ),
    "vnext_device_operation_legacy_authority_contract_tests": (
        "VNEXT DEVICE OPERATION LEGACY AUTHORITY PASS",
        13,
    ),
}
VNEXT_G01A_REQUIRED_ORACLE_TESTS = {
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
VNEXT_G01A_REQUIRED_MODEL_WIRE_TESTS = {
    "prepared_family_wire_accepts_max_and_rejects_max_plus_one_before_serde",
    "prepared_family_wire_rejects_unknown_fields_and_typed_drift",
    "prepared_family_wire_round_trip_requires_external_typed_registry",
    "prepared_model_family_wire_proof_line",
    "typed_family_config_and_registry_identity_fail_closed",
    "typed_config_is_serialized_once_and_signed_external_identity_is_replayed",
}
VNEXT_G01A_REQUIRED_COMPILE_TESTS = {"vnext_compile"}
VNEXT_G01A_REQUIRED_LEGACY_TESTS = {"legacy_backend_methods_are_mapped_82_of_82"}
VNEXT_G01A_REQUIRED_TESTS_BY_TARGET = {
    **VNEXT_G01A_REQUIRED_CORE_TESTS_BY_TARGET,
    **VNEXT_G01A_REQUIRED_RESOURCE_TESTS_BY_TARGET,
    **VNEXT_G01A_REQUIRED_EVENT_TESTS_BY_TARGET,
    "vnext_resolution_limits_contract_tests": VNEXT_G01A_REQUIRED_RESOLUTION_LIMITS_TESTS,
    **VNEXT_G01A_REQUIRED_DEVICE_OPERATION_TESTS_BY_TARGET,
    "vnext_oracle_contract_tests": VNEXT_G01A_REQUIRED_ORACLE_TESTS,
    "vnext_model_wire_contract_tests": VNEXT_G01A_REQUIRED_MODEL_WIRE_TESTS,
    "vnext_compile": VNEXT_G01A_REQUIRED_COMPILE_TESTS,
    "vnext_legacy_map": VNEXT_G01A_REQUIRED_LEGACY_TESTS,
}
VNEXT_G01A_REQUIRED_ADMISSION_LIB_TESTS = {
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
VNEXT_G01A_QUALITY_COMMANDS = (
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
VNEXT_G01A_EXPECTED_RESOURCE_CASES = 311
VNEXT_G01A_EXPECTED_FAIL_CLOSED_CASES = 62
VNEXT_G01A_EXPECTED_EVENT_REPLAY_V5_CASES = 161
VNEXT_G01A_EXPECTED_DEVICE_OPERATION_CASES = 299
VNEXT_G01A_EXPECTED_ORACLE_CASES = 26
VNEXT_G01A_EXPECTED_MODEL_WIRE_CASES = 24
VNEXT_G01A_EXPECTED_MODEL_IDENTITY_CASES = 5
VNEXT_G01A_EXPECTED_DYNAMIC_ADMISSION_CASES = 40
VNEXT_G01A_EXPECTED_TRYBUILD_PASS_CASES = 2
VNEXT_G01A_EXPECTED_TRYBUILD_FAIL_CASES = 78
VNEXT_G01A_TEST_THREADS_ARG = "--test-threads=1"
VNEXT_G01A_BOUNDED_RECEIPT_SCHEMA = "ferrum.bounded-command-receipt.v1"
VNEXT_G01A_BOUNDED_TEST_COMMAND_COUNT = 60
VNEXT_G01A_BOUNDED_TEST_ENV_OVERRIDES = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "CARGO_BUILD_JOBS": "2",
}
# These bounds cover the complete cargo/rustc/test process group. Regular cold
# builds use up to five processes and 28 group threads. trybuild launches a
# nested Cargo build which can transiently cross 32 group threads even with
# CARGO_BUILD_JOBS=2, so it keeps the same process/per-process ceilings with a
# 64-thread group ceiling, still 128x below the prior 8192-thread failure.
VNEXT_G01A_BOUNDED_TEST_PROFILES = {
    "regular": {
        "wall_timeout_seconds": 120.0,
        "max_processes": 8,
        "max_group_threads": 32,
        "max_per_process_threads": 16,
        "sample_interval_seconds": 0.05,
        "max_sampling_errors": 3,
        "term_grace_seconds": 1.0,
    },
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
        "max_group_threads": 64,
        "max_per_process_threads": 16,
        "sample_interval_seconds": 0.05,
        "max_sampling_errors": 3,
        "term_grace_seconds": 1.0,
    },
}
G0_UNIT_BOUNDED_COMMAND = [
    "env",
    "PYTHONDONTWRITEBYTECODE=1",
    "CARGO_BUILD_JOBS=2",
    "RUST_TEST_THREADS=1",
    "cargo",
    "test",
    "--workspace",
    "--all-targets",
]
G0_UNIT_BOUNDED_ENV_OVERRIDES = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "CARGO_BUILD_JOBS": "2",
    "RUST_TEST_THREADS": "1",
}
G0_UNIT_BENCH_CASES = (
    "single_request/tokens/1",
    "single_request/tokens/5",
    "single_request/tokens/10",
    "single_request/tokens/20",
    "single_request/tokens/50",
    "concurrent_throughput/concurrency/1",
    "concurrent_throughput/concurrency/2",
    "concurrent_throughput/concurrency/4",
    "concurrent_throughput/concurrency/8",
    "concurrent_throughput/concurrency/16",
    "scheduling_overhead/single_request_overhead",
    "scheduling_overhead/sequential_10_requests",
)
G0_UNIT_BOUNDED_LIMITS = {
    "wall_timeout_seconds": 1800.0,
    "max_processes": 16,
    "max_group_threads": 96,
    "max_per_process_threads": 48,
    "sample_interval_seconds": 0.05,
    "max_sampling_errors": 3,
    "term_grace_seconds": 1.0,
}
VNEXT_G01A_DOES_NOT_PROVE = {
    "G00",
    "G01B",
    "G01",
    "runtime_migration",
    "product_routing",
    "model_migration",
    "performance",
    "release",
}
VNEXT_LEGACY_EXPECTATIONS_PATH = (
    "scripts/release/configs/runtime_vnext_legacy_correctness_expectations.json"
)
VNEXT_LEGACY_EXPECTATIONS_SNAPSHOT = "legacy-correctness-expectations.json"
VNEXT_PRIMARY_MODELS = {
    "m1-qwen35-4b": "Qwen/Qwen3.5-4B",
    "m2-qwen35-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
    "m3-qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
}
VNEXT_SUPPLEMENTAL_MODELS = {
    "qwen3-coder-30b-a3b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "deepseek-r1-qwen3-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "llama31-8b-compat": "meta-llama/Llama-3.1-8B-Instruct",
}
VNEXT_RESOLUTION_MODEL_IDS = {
    "m1-qwen35-4b": "M1",
    "m2-qwen35-35b-a3b": "M2",
    "m3-qwen3-30b-a3b": "M3",
    "qwen3-coder-30b-a3b": "Qwen3-Coder-30B-A3B-Instruct",
    "deepseek-r1-qwen3-8b": "DeepSeek-R1-0528-Qwen3-8B",
    "llama31-8b-compat": "Llama-3.1-8B-Instruct",
}
CHILD_INDEX_EXCLUDED = {
    "manifest.json",
    "gate.manifest.json",
    "run_gate.child.stdout",
    "run_gate.child.stderr",
    "run_gate.child.command.json",
}


@dataclass(frozen=True)
class LaneCommand:
    cmd: list[str]
    binary_path: Path | None = None
    model: str | None = None
    expected_child_pass_line: str | None = None
    child_manifest_path: Path | None = None
    expected_source_git_sha: str | None = None
    provenance_kind: str | None = None


class GateError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def git_output(args: list[str], *, default: str = "unknown") -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return default
    if proc.returncode != 0:
        return default
    return proc.stdout.strip()


def git_sha() -> str:
    return git_output(["rev-parse", "HEAD"])


def git_dirty_status() -> dict[str, Any]:
    text = git_output(["status", "--short"], default="")
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "is_dirty": bool(lines),
        "status_short": lines,
    }


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def require_external_vnext_g00_output(path: Path) -> None:
    require_gate(
        not is_within(path, REPO_ROOT),
        f"vnext-g00 --out must resolve outside the Git source tree: {path}",
    )


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitized_env_summary() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if not (key in ENV_ALLOW_KEYS or any(key.startswith(prefix) for prefix in ENV_ALLOW_PREFIXES)):
            continue
        if any(fragment in key.upper() for fragment in SECRET_KEY_FRAGMENTS):
            out[key] = "<redacted>"
        elif len(value) > 512:
            out[key] = f"{value[:512]}...<truncated>"
        else:
            out[key] = value
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def build_lane_command(args: argparse.Namespace, out_dir: Path) -> LaneCommand:
    lane = args.lane
    if lane == "vnext-g00a":
        if args.coupling_inventory is None:
            raise GateError("vnext-g00a requires --coupling-inventory")
        if args.model_resolution is None:
            raise GateError("vnext-g00a requires --model-resolution")
        return LaneCommand(
            cmd=[
                sys.executable,
                "scripts/release/runtime_vnext_g00a_checkpoint.py",
                "--coupling-inventory",
                str(args.coupling_inventory),
                "--model-resolution",
                str(args.model_resolution),
                "--out",
                str(out_dir),
            ],
            expected_child_pass_line=f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {out_dir}",
            child_manifest_path=out_dir / "manifest.json",
            provenance_kind="vnext-g00a",
        )
    if lane == "vnext-g00":
        return LaneCommand(
            cmd=[
                sys.executable,
                "scripts/release/runtime_vnext_baseline_gate.py",
                "--out",
                str(out_dir),
                "--require-full-self-test",
            ],
            expected_child_pass_line=f"FERRUM RUNTIME VNEXT G00 BASELINE PASS: {out_dir}",
            child_manifest_path=out_dir / "manifest.json",
            expected_source_git_sha=VNEXT_FROZEN_LEGACY_SHA,
            provenance_kind="vnext-g00",
        )
    if lane == "vnext-g00f":
        if args.g00a is None:
            raise GateError("vnext-g00f requires --g00a")
        return LaneCommand(
            cmd=[
                sys.executable,
                "scripts/release/runtime_vnext_g00f_checkpoint.py",
                "--g00a",
                str(args.g00a),
                "--out",
                str(out_dir),
            ],
            expected_child_pass_line=(
                f"FERRUM RUNTIME VNEXT G00F FACTS PASS: {out_dir}"
            ),
            child_manifest_path=out_dir / "manifest.json",
            provenance_kind="vnext-g00f",
        )
    if lane == "vnext-g01a":
        if args.g00f is None:
            raise GateError("vnext-g01a requires --g00f")
        return LaneCommand(
            cmd=[
                sys.executable,
                "scripts/release/runtime_vnext_s0a_contract_split.py",
                "--g00f",
                str(args.g00f),
                "--out",
                str(out_dir),
            ],
            expected_child_pass_line=(
                f"FERRUM RUNTIME VNEXT G01A CONTRACT SPLIT PASS: {out_dir}"
            ),
            child_manifest_path=(
                out_dir / "g01a-contract-split" / "manifest.json"
            ),
            provenance_kind="vnext-g01a-s0a",
        )
    if lane == "vnext-s1-cuda":
        if args.s1_artifact is None:
            raise GateError("vnext-s1-cuda requires --s1-artifact")
        return LaneCommand(
            cmd=[
                sys.executable,
                "scripts/release/runtime_vnext_s1_cuda_checkpoint.py",
                str(args.s1_artifact.resolve()),
                "--require-bounded-overhead",
                "--out",
                str(out_dir),
            ],
            expected_child_pass_line=(
                f"FERRUM RUNTIME VNEXT S1 CUDA BASIC SLICE PASS: {out_dir}"
            ),
            child_manifest_path=out_dir / "manifest.json",
            provenance_kind="vnext-s1-cuda",
        )
    if lane == "vnext-s1-cuda-capacity":
        if args.s1_artifact is None:
            raise GateError("vnext-s1-cuda-capacity requires --s1-artifact")
        return LaneCommand(
            cmd=[
                sys.executable,
                "scripts/release/runtime_vnext_s1_cuda_capacity.py",
                "validate",
                str(args.s1_artifact.resolve()),
                "--out",
                str(out_dir),
            ],
            expected_child_pass_line=(
                f"FERRUM RUNTIME VNEXT S1 CUDA CAPACITY PRESSURE PASS: {out_dir}"
            ),
            child_manifest_path=out_dir / "manifest.json",
            provenance_kind="vnext-s1-cuda-capacity",
        )
    if lane in SOURCE_LANES:
        source_lane = SOURCE_LANES[lane]
        return LaneCommand(
            cmd=[
                "scripts/release/g0_source_gate.sh",
                source_lane,
                str(out_dir),
            ],
            binary_path=Path("target/release/ferrum")
            if lane.startswith("cuda") or lane == "metal"
            else None,
            model=model_for_source_lane(lane),
            expected_child_pass_line=source_pass_line(lane, out_dir),
            child_manifest_path=(out_dir / "unit.gate.json")
            if source_lane == "unit"
            else None,
            provenance_kind="g0-source-unit" if source_lane == "unit" else None,
        )
    if lane in BINARY_LANES:
        if not args.version:
            raise GateError(f"{lane} requires --version")
        cmd = [
            sys.executable,
            "scripts/release/release_binary_gate.py",
            lane,
            "--version",
            args.version,
            "--out",
            str(out_dir),
        ]
        if args.asset_path is not None:
            cmd.extend(["--asset-path", str(args.asset_path)])
        if args.sha256 is not None:
            cmd.extend(["--sha256", args.sha256])
        if args.model is not None:
            cmd.extend(["--model", args.model])
        if args.model_name is not None:
            cmd.extend(["--model-name", args.model_name])
        if args.port is not None:
            cmd.extend(["--port", str(args.port)])
        return LaneCommand(
            cmd=cmd,
            model=args.model,
            expected_child_pass_line=binary_pass_line(lane, out_dir),
        )
    if lane == "release-summary":
        release_root = args.release_root or out_dir
        return LaneCommand(
            cmd=[sys.executable, "scripts/release/g0_release_summary.py", str(release_root)],
            expected_child_pass_line=f"G0 RELEASE PASS: {release_root}",
        )
    if lane == "release-complete":
        if args.completion_manifest is None:
            raise GateError("release-complete requires --completion-manifest")
        return LaneCommand(
            cmd=[
                sys.executable,
                "scripts/release/validate_release_completion_manifest.py",
                "--manifest",
                str(args.completion_manifest),
                "--out",
                str(out_dir),
            ],
            expected_child_pass_line=f"FERRUM RELEASE COMPLETION PASS: {out_dir}",
        )
    raise GateError(f"unknown lane: {lane}")


def model_for_source_lane(lane: str) -> str | None:
    if lane in {"cuda-smoke", "cuda-full"}:
        return "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
    if lane == "cuda-llama-dense":
        return "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    if lane in {
        "cuda-llama33-70b-4bit-2x4090-smoke",
        "cuda-llama33-70b-4bit-2x4090",
    }:
        return "clowman/Llama-3.3-70B-Instruct-GPTQ-Int4"
    return None


def source_pass_line(lane: str, out_dir: Path) -> str:
    delegated = {
        "unit": "unit",
        "metal": "metal",
        "cuda-smoke": "g0_cuda4090_smoke",
        "cuda-full": "g0_cuda4090_full",
        "cuda-llama-dense": "g0_cuda4090_llama_dense",
        "cuda-llama33-70b-4bit-2x4090-smoke": "g0_cuda2x4090_llama33_70b_4bit_smoke",
        "cuda-llama33-70b-4bit-2x4090": "g0_cuda2x4090_llama33_70b_4bit",
    }[lane]
    return f"G0 SOURCE {delegated} PASS: {out_dir}"


def binary_pass_line(lane: str, out_dir: Path) -> str:
    delegated = {
        "metal-tarball": "METAL TARBALL GATE",
        "cuda-tarball": "CUDA TARBALL GATE",
        "homebrew-metal": "HOMEBREW METAL GATE",
        "homebrew-cuda-fetch": "HOMEBREW CUDA FETCH GATE",
    }[lane]
    return f"{delegated} PASS: {out_dir}"


def require_gate(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def validate_safetensors_shard_paths(paths: set[str], label: str) -> bool:
    shards = sorted(path for path in paths if path.endswith(".safetensors"))
    matches = [(path, SAFETENSORS_SHARD_RE.search(path)) for path in shards]
    sharded = len(shards) > 1 or any(match is not None for _, match in matches)
    if not sharded:
        return False
    require_gate(shards, f"{label} sharded safetensors set is empty")
    require_gate(
        all(match is not None for _, match in matches),
        f"{label} sharded safetensors path lacks canonical numbering",
    )
    numbered = [
        (int(match.group(1)), int(match.group(2)), len(match.group(1)), len(match.group(2)))
        for _, match in matches
        if match is not None
    ]
    require_gate(
        len({number_width for _, _, number_width, _ in numbered}) == 1,
        f"{label} sharded safetensors number width differs",
    )
    require_gate(
        len({total_width for _, _, _, total_width in numbered}) == 1,
        f"{label} sharded safetensors total width differs",
    )
    totals = {total for _, total, _, _ in numbered}
    require_gate(len(totals) == 1, f"{label} safetensors shards disagree on total count")
    total = totals.pop()
    require_gate(total == len(numbered), f"{label} safetensors shard count differs from numbered total")
    require_gate(
        {number for number, _, _, _ in numbered} == set(range(1, total + 1)),
        f"{label} safetensors shard numbering is incomplete",
    )
    return True


def validate_catalog_weight_paths(
    catalog_lane: dict[str, Any],
    paths: set[str],
    label: str,
) -> bool:
    sharded = validate_safetensors_shard_paths(paths, label)
    selectors = require_list(catalog_lane.get("files"), f"{label}.catalog.files")
    conditional_paths = {
        str(selector["path"])
        for selector in selectors
        if isinstance(selector, dict)
        and selector.get("required_if_sharded") is True
        and "path" in selector
    }
    require_gate(
        sharded or paths.isdisjoint(conditional_paths),
        f"{label} unsharded model contains a conditional weight file",
    )
    active_selectors: list[dict[str, Any]] = []
    for index, raw in enumerate(selectors):
        selector = require_object(raw, f"{label}.catalog.files[{index}]")
        active = selector.get("required") is True or (
            selector.get("required_if_sharded") is True and sharded
        )
        if not active:
            continue
        active_selectors.append(selector)
        if "path" in selector:
            expected = require_string(
                selector.get("path"),
                f"{label}.catalog.files[{index}].path",
            )
            require_gate(expected in paths, f"{label} missing required weight file {expected}")
        elif "glob" in selector:
            pattern = require_string(
                selector.get("glob"),
                f"{label}.catalog.files[{index}].glob",
            )
            require_gate(
                any(fnmatch.fnmatchcase(path, pattern) for path in paths),
                f"{label} missing required weight glob {pattern}",
            )
        else:
            raise GateError(f"{label}.catalog.files[{index}] needs path or glob")
    for path in paths:
        selected = any(
            ("path" in selector and selector["path"] == path)
            or (
                "glob" in selector
                and fnmatch.fnmatchcase(path, str(selector["glob"]))
            )
            for selector in active_selectors
        )
        require_gate(selected, f"{label} contains an unselected weight file: {path}")
    return sharded


def require_object(value: Any, label: str) -> dict[str, Any]:
    require_gate(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require_gate(isinstance(value, list), f"{label} must be a JSON array")
    return value


def require_string(value: Any, label: str) -> str:
    require_gate(isinstance(value, str) and value.strip(), f"{label} must be a non-empty string")
    return value.strip()


def require_sha256(value: Any, label: str) -> str:
    digest = require_string(value, label).lower()
    require_gate(SHA256_RE.fullmatch(digest) is not None, f"{label} must be a lowercase SHA256")
    return digest


def require_git_sha(value: Any, label: str) -> str:
    value = require_string(value, label).lower()
    require_gate(GIT_SHA_RE.fullmatch(value) is not None, f"{label} must be a 40-character git SHA")
    return value


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pretty_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strict_json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def strict_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_json_object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON number: {value}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GateError(f"invalid {label}: {exc}") from exc


def decoded_request_body(request: dict[str, Any], label: str) -> Any:
    encoded = require_string(request.get("response_body_base64"), f"{label}.response_body_base64")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise GateError(f"invalid {label} response body base64: {exc}") from exc
    require_gate(len(payload) == request.get("response_bytes"), f"{label} response body size mismatch")
    require_gate(hashlib.sha256(payload).hexdigest() == request.get("response_sha256"), f"{label} response body SHA256 mismatch")
    return strict_json_bytes(payload, f"{label} response body JSON")


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GateError(f"invalid {label} {path}: {exc}") from exc
    return require_object(value, label)


def child_artifact_path(root: Path, raw: Any, label: str) -> tuple[Path, str]:
    relative = require_string(raw, label)
    rel_path = Path(relative)
    require_gate(not rel_path.is_absolute(), f"{label} must be relative to the child artifact root")
    resolved = (root / rel_path).resolve()
    try:
        normalized = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise GateError(f"{label} escapes the child artifact root: {relative}") from exc
    require_gate(normalized == rel_path.as_posix(), f"{label} must be a normalized relative path")
    require_gate(resolved.is_file() and not resolved.is_symlink(), f"{label} missing regular artifact: {normalized}")
    return resolved, normalized


def validate_child_artifact_index(
    root: Path,
    child_manifest: dict[str, Any],
    *,
    role_from_top_level_path: bool = True,
) -> dict[str, dict[str, Any]]:
    rows = require_list(child_manifest.get("artifact_index"), "delegated manifest artifact_index")
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = require_object(raw, f"delegated manifest artifact_index[{index}]")
        path, relative = child_artifact_path(root, row.get("path"), f"artifact_index[{index}].path")
        require_gate(relative not in CHILD_INDEX_EXCLUDED, f"artifact_index contains excluded control artifact: {relative}")
        require_gate(relative not in indexed, f"artifact_index contains duplicate path: {relative}")
        digest = require_sha256(row.get("sha256"), f"artifact_index[{relative}].sha256")
        size = row.get("size_bytes")
        require_gate(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"artifact_index[{relative}].size_bytes must be positive")
        require_gate(path.stat().st_size == size, f"artifact_index[{relative}] size mismatch")
        require_gate(sha256(path) == digest, f"artifact_index[{relative}] SHA256 mismatch")
        if role_from_top_level_path:
            expected_role = relative.split("/", 1)[0] if "/" in relative else "root-manifest"
            require_gate(row.get("role") == expected_role, f"artifact_index[{relative}] role mismatch")
        else:
            require_string(row.get("role"), f"artifact_index[{relative}].role")
        indexed[relative] = row
    actual: set[str] = set()
    for path in sorted(root.rglob("*")):
        require_gate(not path.is_symlink(), f"child artifact tree contains forbidden symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in CHILD_INDEX_EXCLUDED:
            actual.add(relative)
    require_gate(set(indexed) == actual, f"delegated artifact_index coverage mismatch: missing={sorted(actual - set(indexed))} extra={sorted(set(indexed) - actual)}")
    require_gate(child_manifest.get("artifact_count") == len(indexed), "delegated manifest artifact_count mismatch")
    return indexed


def require_indexed_artifact(
    root: Path,
    index: dict[str, dict[str, Any]],
    raw_path: Any,
    raw_sha256: Any,
    label: str,
) -> tuple[Path, str, str]:
    path, relative = child_artifact_path(root, raw_path, f"{label}.path")
    digest = require_sha256(raw_sha256, f"{label}.sha256")
    require_gate(relative in index, f"{label} is absent from delegated artifact_index: {relative}")
    require_gate(index[relative].get("sha256") == digest, f"{label} SHA256 differs from delegated artifact_index")
    require_gate(sha256(path) == digest, f"{label} artifact SHA256 mismatch")
    return path, relative, digest


def validate_vnext_g00_expectations_snapshot(
    root: Path,
    artifact_index: dict[str, dict[str, Any]],
    raw: Any,
    *,
    source_path: Path,
    source_sha256: str,
    label: str,
) -> dict[str, Any]:
    ref = require_object(raw, label)
    require_gate(
        set(ref) == {"kind", "path", "sha256"},
        f"{label} must be a canonical artifact snapshot reference",
    )
    require_gate(ref.get("kind") == "raw-json", f"{label}.kind must be raw-json")
    snapshot_path, snapshot_relative, snapshot_sha256 = require_indexed_artifact(
        root,
        artifact_index,
        ref.get("path"),
        ref.get("sha256"),
        label,
    )
    require_gate(
        snapshot_relative == VNEXT_LEGACY_EXPECTATIONS_SNAPSHOT,
        f"{label}.path must be {VNEXT_LEGACY_EXPECTATIONS_SNAPSHOT}",
    )
    require_gate(
        snapshot_sha256 == source_sha256,
        f"{label} SHA256 differs from models.lock source contract",
    )
    require_gate(
        snapshot_path.read_bytes() == source_path.read_bytes(),
        f"{label} bytes differ from the checked-in expectations catalog",
    )
    return {
        "kind": "raw-json",
        "path": snapshot_relative,
        "sha256": snapshot_sha256,
    }


def validate_vnext_g00_runner_identity(
    raw: Any,
    *,
    scenario_runner_path: str,
    scenario_runner_sha256: str,
    validator_git_sha: str,
    contract_by_path: dict[str, dict[str, Any]],
    label: str,
    verify_checkout: bool,
) -> dict[str, Any]:
    runner = require_object(raw, label)
    expected_fields = {
        "path",
        "sha256",
        "git_sha",
        "source_tree_sha",
        "git_blob_sha",
        "dirty_status",
    }
    require_gate(set(runner) == expected_fields, f"{label} field set mismatch")
    require_gate(runner.get("path") == scenario_runner_path, f"{label}.path mismatch")
    runner_sha256 = require_sha256(runner.get("sha256"), f"{label}.sha256")
    require_gate(runner_sha256 == scenario_runner_sha256, f"{label}.sha256 mismatch")
    runner_git_sha = require_git_sha(runner.get("git_sha"), f"{label}.git_sha")
    runner_tree_sha = require_git_sha(runner.get("source_tree_sha"), f"{label}.source_tree_sha")
    runner_blob_sha = require_git_sha(runner.get("git_blob_sha"), f"{label}.git_blob_sha")
    require_gate(runner_git_sha == validator_git_sha, f"{label}.git_sha differs from delegated validator")
    dirty = require_object(runner.get("dirty_status"), f"{label}.dirty_status")
    require_gate(
        dirty == {"is_dirty": False, "status_short": []},
        f"{label}.dirty_status must prove a clean runner checkout",
    )
    runner_contract = require_object(
        contract_by_path.get(scenario_runner_path),
        f"{label} delegated contract",
    )
    require_gate(
        require_sha256(runner_contract.get("sha256"), f"{label} delegated contract SHA256")
        == runner_sha256,
        f"{label} differs from delegated contract",
    )
    if verify_checkout:
        require_gate(git_sha() == validator_git_sha, f"{label} validator SHA is stale against current HEAD")
        require_gate(
            git_output(["rev-parse", "HEAD^{tree}"]) == runner_tree_sha,
            f"{label}.source_tree_sha is stale against current HEAD",
        )
        require_gate(
            git_output(["rev-parse", f"HEAD:{scenario_runner_path}"]) == runner_blob_sha,
            f"{label}.git_blob_sha is stale against current HEAD",
        )
        require_gate(not git_dirty_status()["is_dirty"], f"{label} current checkout is dirty")
        runner_path = REPO_ROOT / scenario_runner_path
        require_gate(
            runner_path.is_file()
            and not runner_path.is_symlink()
            and sha256(runner_path) == runner_sha256,
            f"{label} current runner file differs from its identity",
        )
    return {
        "path": scenario_runner_path,
        "sha256": runner_sha256,
        "git_sha": runner_git_sha,
        "source_tree_sha": runner_tree_sha,
        "git_blob_sha": runner_blob_sha,
        "dirty_status": {"is_dirty": False, "status_short": []},
    }


def normalized_file_locks(raw: Any, label: str) -> list[dict[str, Any]]:
    rows = require_list(raw, label)
    require_gate(rows, f"{label} must not be empty")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item_raw in enumerate(rows):
        item = require_object(item_raw, f"{label}[{index}]")
        path = require_string(item.get("path"), f"{label}[{index}].path")
        require_gate(path not in seen, f"{label} duplicate file path: {path}")
        seen.add(path)
        size = item.get("size_bytes")
        require_gate(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"{label}[{index}].size_bytes must be positive")
        normalized.append(
            {
                "path": path,
                "sha256": require_sha256(item.get("sha256"), f"{label}[{index}].sha256"),
                "size_bytes": size,
            }
        )
    return sorted(normalized, key=lambda row: row["path"])


def normalized_model_source(raw: Any, label: str) -> dict[str, Any]:
    source = require_object(raw, label)
    return {
        "repo": require_string(source.get("repo"), f"{label}.repo"),
        "revision": require_git_sha(source.get("revision"), f"{label}.revision"),
        "files": normalized_file_locks(source.get("files"), f"{label}.files"),
    }


def require_exact_string_set(raw: Any, expected: set[str], label: str) -> set[str]:
    values = require_list(raw, label)
    normalized = [require_string(value, f"{label}[{index}]") for index, value in enumerate(values)]
    require_gate(len(normalized) == len(set(normalized)), f"{label} contains duplicates")
    actual = set(normalized)
    require_gate(actual == expected, f"{label} mismatch: expected={sorted(expected)} actual={sorted(actual)}")
    return actual


def validate_vnext_catalog_expected_weight_facts(
    raw_catalog: Any,
    resolved_by_id: dict[str, dict[str, Any]],
) -> int:
    catalog = require_object(raw_catalog, "vnext-g00a models catalog")
    rows = require_list(catalog.get("models"), "vnext-g00a models catalog.models")
    require_gate(len(rows) == 12, "vnext-g00a models catalog must contain 12 lanes")
    catalog_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = require_object(raw_row, f"vnext-g00a models catalog.models[{index}]")
        lane_id = require_string(row.get("id"), f"vnext-g00a models catalog.models[{index}].id")
        require_gate(lane_id not in catalog_by_id, f"vnext-g00a duplicate catalog lane id: {lane_id}")
        catalog_by_id[lane_id] = row
    require_gate(
        set(catalog_by_id) == set(resolved_by_id),
        "vnext-g00a models catalog/resolution lane set mismatch",
    )

    assertion_count = 0
    for lane_id in sorted(catalog_by_id):
        catalog_lane = catalog_by_id[lane_id]
        resolved_lane = require_object(resolved_by_id[lane_id], f"vnext-g00a resolved lane {lane_id}")
        weight_source = require_object(
            resolved_lane.get("weight_source"),
            f"vnext-g00a resolved lane {lane_id}.weight_source",
        )
        weight_files: dict[str, dict[str, Any]] = {}
        for file_index, raw_file in enumerate(
            require_list(weight_source.get("files"), f"vnext-g00a resolved lane {lane_id}.weight_source.files")
        ):
            file_row = require_object(raw_file, f"vnext-g00a resolved lane {lane_id}.weight_source.files[{file_index}]")
            path = require_string(file_row.get("path"), f"vnext-g00a resolved lane {lane_id}.weight_source.files[{file_index}].path")
            require_gate(path not in weight_files, f"vnext-g00a duplicate resolved weight path: {lane_id}/{path}")
            weight_files[path] = file_row

        selectors = require_list(catalog_lane.get("files"), f"vnext-g00a models catalog {lane_id}.files")
        for selector_index, raw_selector in enumerate(selectors):
            selector = require_object(raw_selector, f"vnext-g00a models catalog {lane_id}.files[{selector_index}]")
            expected_size = selector.get("expected_size_bytes")
            expected_sha_raw = selector.get("expected_sha256")
            if expected_size is None and expected_sha_raw is None:
                continue
            require_gate(
                "path" in selector and "glob" not in selector,
                f"vnext-g00a catalog expected weight identity requires an exact path: {lane_id}",
            )
            path = require_string(selector.get("path"), f"vnext-g00a models catalog {lane_id}.files[{selector_index}].path")
            file_row = require_object(weight_files.get(path), f"vnext-g00a resolved expected weight {lane_id}/{path}")
            if expected_size is not None:
                require_gate(
                    isinstance(expected_size, int) and not isinstance(expected_size, bool) and expected_size > 0,
                    f"vnext-g00a catalog expected size is invalid: {lane_id}/{path}",
                )
                require_gate(
                    file_row.get("size_bytes") == expected_size,
                    f"vnext-g00a catalog expected size mismatch: {lane_id}/{path}",
                )
            if expected_sha_raw is not None:
                require_gate(
                    selector.get("required") is True,
                    f"vnext-g00a catalog expected SHA256 requires required=true: {lane_id}/{path}",
                )
                expected_sha = require_sha256(
                    expected_sha_raw,
                    f"vnext-g00a models catalog {lane_id}.files[{selector_index}].expected_sha256",
                )
                require_gate(
                    file_row.get("sha256") == expected_sha,
                    f"vnext-g00a catalog expected SHA256 mismatch: {lane_id}/{path}",
                )
                require_gate(
                    file_row.get("sha256_source") == "hugging_face_lfs_oid",
                    f"vnext-g00a catalog expected SHA256 lacks Hugging Face LFS identity: {lane_id}/{path}",
                )
                require_gate(
                    file_row.get("lfs_oid") == expected_sha,
                    f"vnext-g00a catalog expected SHA256 differs from LFS OID: {lane_id}/{path}",
                )
            assertion_count += 1
    require_gate(assertion_count > 0, "vnext-g00a models catalog has no expected weight identity assertions")
    return assertion_count


def validate_vnext_g00a_provenance(
    lane_command: LaneCommand,
    child_manifest: dict[str, Any],
    child_manifest_sha256: str,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    manifest_path = lane_command.child_manifest_path
    require_gate(manifest_path is not None, "vnext-g00a delegated manifest path is missing")
    root = manifest_path.parent.resolve()
    require_gate(manifest_path.resolve() == root / "manifest.json", "vnext-g00a manifest path mismatch")
    require_gate(child_manifest.get("schema_version") == 1, "vnext-g00a manifest schema_version mismatch")
    require_gate(child_manifest.get("artifact_type") == "runtime_vnext_g00a_fact_checkpoint_manifest", "vnext-g00a manifest artifact_type mismatch")
    require_gate(child_manifest.get("lane") == "runtime-vnext-g00a", "vnext-g00a manifest lane mismatch")
    require_gate(child_manifest.get("checkpoint_id") == "G00a", "vnext-g00a checkpoint_id mismatch")
    require_gate(child_manifest.get("canonical") is True, "vnext-g00a manifest must be canonical")
    require_gate(child_manifest.get("dirty") is False, "vnext-g00a manifest must be clean")
    artifact_dir = Path(require_string(child_manifest.get("artifact_dir"), "vnext-g00a artifact_dir"))
    require_gate(artifact_dir.resolve() == root, "vnext-g00a artifact_dir mismatch")
    require_exact_string_set(child_manifest.get("unlocks"), {"G01A"}, "vnext-g00a unlocks")
    require_exact_string_set(child_manifest.get("does_not_prove"), VNEXT_G00A_DOES_NOT_PROVE, "vnext-g00a does_not_prove")

    freshness = require_object(child_manifest.get("freshness"), "vnext-g00a freshness")
    expected_freshness = {
        "catalogs_match_collector_head",
        "collector_checkout_clean",
        "inventory_candidate_matches_current_analyzer_recomputation",
        "inventory_frozen_source_clean",
        "model_resolution_catalog_matches_current_head",
        "model_resolution_input_matches_live_facts",
        "model_resolution_live_recomputed",
        "model_resolution_resolver_matches_current_head",
        "model_resolution_source_matches_collector_head",
    }
    require_gate(set(freshness) == expected_freshness, "vnext-g00a freshness field set mismatch")
    require_gate(all(value is True for value in freshness.values()), "vnext-g00a freshness contains a non-true value")

    index = validate_child_artifact_index(root, child_manifest, role_from_top_level_path=False)
    expected_artifacts = {
        "coupling-inventory.json",
        "generation-presets.catalog.json",
        "historical-bugs.catalog.json",
        "inventory-review.catalog.json",
        "model-facts.lock.json",
        "model-resolution.input.json",
        "model-resolution.json",
        "models.catalog.json",
    }
    require_gate(set(index) == expected_artifacts, "vnext-g00a artifact set mismatch")
    require_gate(child_manifest.get("artifact_count") == len(expected_artifacts), "vnext-g00a artifact_count mismatch")
    fact_sources = require_object(child_manifest.get("fact_source_artifacts"), "vnext-g00a fact_source_artifacts")
    expected_fact_sources = {
        "coupling_inventory": "coupling-inventory.json",
        "model_resolution_input": "model-resolution.input.json",
        "model_resolution_live": "model-resolution.json",
    }
    require_gate(set(fact_sources) == set(expected_fact_sources), "vnext-g00a fact source field set mismatch")
    for source_name, artifact_name in expected_fact_sources.items():
        source_ref = require_object(fact_sources.get(source_name), f"vnext-g00a fact source {source_name}")
        require_gate(source_ref == index[artifact_name], f"vnext-g00a fact source/index mismatch: {source_name}")

    collector = require_object(child_manifest.get("collector"), "vnext-g00a collector")
    collector_sha = require_git_sha(collector.get("git_sha"), "vnext-g00a collector.git_sha")
    collector_tree = require_git_sha(collector.get("git_tree_sha"), "vnext-g00a collector.git_tree_sha")
    require_gate(collector.get("dirty") is False and collector.get("status_short") == [], "vnext-g00a collector dirty state mismatch")
    require_gate(child_manifest.get("git_sha") == collector_sha, "vnext-g00a manifest/collector git SHA mismatch")
    require_gate(child_manifest.get("git_tree_sha") == collector_tree, "vnext-g00a manifest/collector tree mismatch")
    contract_rows = require_list(collector.get("contracts"), "vnext-g00a collector.contracts")
    require_gate(pretty_json_sha256(contract_rows) == require_sha256(collector.get("contracts_sha256"), "vnext-g00a collector.contracts_sha256"), "vnext-g00a collector contract-list digest mismatch")
    contracts: dict[str, dict[str, Any]] = {}
    for row_index, raw in enumerate(contract_rows):
        row = require_object(raw, f"vnext-g00a collector.contracts[{row_index}]")
        relative = require_string(row.get("path"), f"vnext-g00a collector.contracts[{row_index}].path")
        rel_path = Path(relative)
        require_gate(not rel_path.is_absolute() and rel_path.as_posix() == relative and ".." not in rel_path.parts, f"invalid vnext-g00a contract path: {relative}")
        require_gate(relative not in contracts, f"duplicate vnext-g00a contract path: {relative}")
        digest = require_sha256(row.get("sha256"), f"vnext-g00a collector contract {relative}.sha256")
        size = row.get("size_bytes")
        require_gate(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"vnext-g00a collector contract {relative}.size_bytes invalid")
        git_blob = require_string(row.get("git_blob"), f"vnext-g00a collector contract {relative}.git_blob")
        require_gate(re.fullmatch(r"[0-9a-f]{40,64}", git_blob) is not None, f"vnext-g00a collector contract {relative}.git_blob invalid")
        contracts[relative] = {"sha256": digest, "size_bytes": size, "git_blob": git_blob}
    require_gate(set(contracts) == VNEXT_G00A_CONTRACT_PATHS, "vnext-g00a collector contract path set mismatch")
    catalog_contract_copies = {
        "generation-presets.catalog.json": "scripts/release/configs/runtime_vnext_generation_presets.json",
        "historical-bugs.catalog.json": "scripts/release/configs/runtime_vnext_historical_bugs.json",
        "inventory-review.catalog.json": "scripts/release/configs/runtime_vnext_inventory_review.json",
        "models.catalog.json": "scripts/release/configs/runtime_vnext_models.json",
    }
    for artifact_name, contract_path in catalog_contract_copies.items():
        require_gate(
            index[artifact_name].get("sha256") == contracts[contract_path]["sha256"],
            f"vnext-g00a copied catalog differs from collector contract: {artifact_name}",
        )
        require_gate(
            index[artifact_name].get("size_bytes") == contracts[contract_path]["size_bytes"],
            f"vnext-g00a copied catalog size differs from collector contract: {artifact_name}",
        )

    if verify_checkout:
        require_gate(git_sha() == collector_sha, "vnext-g00a collector SHA is stale against current HEAD")
        require_gate(git_output(["rev-parse", "HEAD^{tree}"]) == collector_tree, "vnext-g00a collector tree is stale against current HEAD")
        require_gate(not git_dirty_status()["is_dirty"], "vnext-g00a run_gate checkout must remain clean")
        for relative, identity in contracts.items():
            path = REPO_ROOT / relative
            require_gate(path.is_file() and not path.is_symlink(), f"vnext-g00a current contract missing: {relative}")
            require_gate(path.stat().st_size == identity["size_bytes"], f"vnext-g00a current contract size mismatch: {relative}")
            require_gate(sha256(path) == identity["sha256"], f"vnext-g00a current contract SHA256 mismatch: {relative}")
            require_gate(git_output(["rev-parse", f"HEAD:{relative}"]) == identity["git_blob"], f"vnext-g00a current contract Git blob mismatch: {relative}")

    frozen = require_object(child_manifest.get("frozen_source"), "vnext-g00a frozen_source")
    require_gate(require_git_sha(frozen.get("git_sha"), "vnext-g00a frozen_source.git_sha") == VNEXT_FROZEN_LEGACY_SHA, "vnext-g00a frozen legacy SHA mismatch")
    frozen_tree = require_git_sha(frozen.get("git_tree_sha"), "vnext-g00a frozen_source.git_tree_sha")
    if verify_checkout:
        require_gate(
            git_output(["rev-parse", f"{VNEXT_FROZEN_LEGACY_SHA}^{{tree}}"])
            == frozen_tree,
            "vnext-g00a frozen legacy tree mismatch",
        )

    lock_ref = require_object(child_manifest.get("model_facts_lock"), "vnext-g00a model_facts_lock")
    lock_path, lock_rel, lock_digest = require_indexed_artifact(
        root,
        index,
        lock_ref.get("path"),
        lock_ref.get("sha256"),
        "vnext-g00a model_facts_lock",
    )
    require_gate(lock_ref.get("size_bytes") == lock_path.stat().st_size, "vnext-g00a model_facts_lock size mismatch")
    lock = read_json_object(lock_path, "vnext-g00a model facts lock")
    require_gate(lock.get("schema_version") == 1, "vnext-g00a model facts lock schema mismatch")
    require_gate(lock.get("artifact_type") == "runtime_vnext_g00a_model_facts_lock", "vnext-g00a model facts lock artifact_type mismatch")
    require_gate(lock.get("checkpoint_id") == "G00a", "vnext-g00a model facts lock checkpoint mismatch")
    lock_scope = require_object(lock.get("scope"), "vnext-g00a model facts lock scope")
    require_exact_string_set(lock_scope.get("unlocks"), {"G01A"}, "vnext-g00a lock scope.unlocks")
    require_exact_string_set(lock_scope.get("does_not_prove"), VNEXT_G00A_DOES_NOT_PROVE, "vnext-g00a lock scope.does_not_prove")
    require_gate(lock_scope.get("historical_evidence") == "catalog_only", "vnext-g00a historical scope must be catalog_only")
    require_gate(lock.get("collector") == {"contracts_sha256": collector["contracts_sha256"], "git_sha": collector_sha, "git_tree_sha": collector_tree}, "vnext-g00a lock collector mismatch")
    require_gate(lock.get("frozen_legacy_source") == frozen, "vnext-g00a lock frozen source mismatch")

    def indexed_digest(relative: str) -> str:
        require_gate(relative in index, f"vnext-g00a missing indexed artifact: {relative}")
        return require_sha256(index[relative].get("sha256"), f"vnext-g00a index {relative}.sha256")

    model_catalog = require_object(lock.get("model_catalog"), "vnext-g00a lock model_catalog")
    require_gate(model_catalog.get("lane_count") == 12, "vnext-g00a model catalog lane_count must be 12")
    require_gate(model_catalog.get("catalog_sha256") == indexed_digest("models.catalog.json"), "vnext-g00a model catalog copy mismatch")
    presets = require_object(lock.get("generation_presets"), "vnext-g00a lock generation_presets")
    require_gate(presets.get("catalog_sha256") == indexed_digest("generation-presets.catalog.json"), "vnext-g00a generation preset catalog copy mismatch")
    preset_facts = require_object(presets.get("facts"), "vnext-g00a generation preset facts")
    preset_models = require_object(preset_facts.get("models"), "vnext-g00a generation preset models")
    require_gate(set(preset_models) == set(VNEXT_PRIMARY_MODELS), "vnext-g00a generation presets must cover exactly the three primary models")
    expected_presets = {"P_DETERMINISTIC", "P_NO_THINKING", "P_THINKING", "P_OFFICIAL_DEFAULT"}
    for model_key, raw in preset_models.items():
        model_presets = require_object(require_object(raw, f"vnext-g00a generation preset {model_key}").get("presets"), f"vnext-g00a generation preset {model_key}.presets")
        require_gate(set(model_presets) == expected_presets, f"vnext-g00a generation preset matrix mismatch: {model_key}")

    history = require_object(lock.get("historical_bug_catalog"), "vnext-g00a lock historical_bug_catalog")
    require_gate(history.get("catalog_sha256") == indexed_digest("historical-bugs.catalog.json"), "vnext-g00a historical catalog copy mismatch")
    history_facts = require_object(history.get("facts"), "vnext-g00a historical facts")
    require_gate(history_facts.get("catalog_scope") == "catalog_only" and history_facts.get("full_historical_corpus_complete") is False, "vnext-g00a historical facts overclaim corpus completion")
    require_gate(history_facts.get("family_count") == 15 and history_facts.get("concrete_case_count") == 28, "vnext-g00a historical fact counts mismatch")
    families = require_list(history_facts.get("families"), "vnext-g00a historical families")
    require_gate(len(families) == 15 and sum(len(require_list(require_object(family, "vnext-g00a historical family").get("cases"), "vnext-g00a historical family cases")) for family in families) == 28, "vnext-g00a historical family/case matrix mismatch")

    inventory = require_object(lock.get("inventory"), "vnext-g00a lock inventory")
    inventory_document = read_json_object(root / "coupling-inventory.json", "vnext-g00a coupling inventory")
    normalized_inventory = dict(inventory_document)
    normalized_inventory.pop("root", None)
    require_gate(pretty_json_sha256(normalized_inventory) == inventory.get("normalized_inventory_sha256"), "vnext-g00a normalized inventory digest mismatch")
    analyzer_contract = require_object(inventory.get("analyzer_contract"), "vnext-g00a inventory analyzer_contract")
    expected_analyzer_contract = next(row for row in contract_rows if row.get("path") == "scripts/release/runtime_vnext_inventory.py")
    require_gate(analyzer_contract == expected_analyzer_contract, "vnext-g00a inventory analyzer contract mismatch")
    analyzer = require_object(inventory_document.get("analyzer"), "vnext-g00a coupling inventory analyzer")
    require_gate(analyzer.get("path") == "scripts/release/runtime_vnext_inventory.py", "vnext-g00a coupling inventory analyzer path mismatch")
    review = require_object(inventory.get("review"), "vnext-g00a lock inventory review")
    require_gate(review.get("sha256") == indexed_digest("inventory-review.catalog.json"), "vnext-g00a inventory review copy mismatch")
    require_gate(review.get("unresolved_count") == 0, "vnext-g00a inventory review has unresolved classifications")

    resolution_ref = require_object(lock.get("model_resolution"), "vnext-g00a lock model_resolution")
    require_gate(resolution_ref.get("live_recomputed") is True, "vnext-g00a model resolution was not live-recomputed")
    require_sha256(resolution_ref.get("live_facts_sha256"), "vnext-g00a live model facts SHA256")
    resolution_path = root / "model-resolution.json"
    resolution = read_json_object(resolution_path, "vnext-g00a model resolution")
    require_gate(resolution.get("schema_version") == 1 and resolution.get("artifact_type") == "runtime_vnext_model_resolution", "vnext-g00a model resolution schema mismatch")
    require_gate(resolution.get("catalog_sha256") == indexed_digest("models.catalog.json"), "vnext-g00a model resolution catalog mismatch")
    require_gate(resolution.get("source") == resolution_ref.get("source"), "vnext-g00a model resolution source lock mismatch")
    require_gate(resolution.get("resolver") == resolution_ref.get("resolver"), "vnext-g00a model resolution resolver lock mismatch")
    resolution_source = require_object(resolution.get("source"), "vnext-g00a model resolution source")
    require_gate(resolution_source.get("git_sha") == collector_sha and resolution_source.get("dirty") is False and resolution_source.get("status_short") == [], "vnext-g00a model resolution source is stale or dirty")
    resolution_resolver = require_object(resolution.get("resolver"), "vnext-g00a model resolution resolver")
    require_gate(resolution_resolver.get("path") == "scripts/release/runtime_vnext_model_resolver.py", "vnext-g00a resolver path mismatch")
    require_gate(resolution_resolver.get("sha256") == contracts["scripts/release/runtime_vnext_model_resolver.py"]["sha256"], "vnext-g00a resolver contract SHA mismatch")
    resolution_policy = require_object(resolution.get("policy"), "vnext-g00a model resolution policy")
    require_gate(resolution_policy.get("transport") == "network_huggingface_https", "vnext-g00a model resolution transport mismatch")
    require_gate(resolution_policy.get("raw_response_body_kinds") == ["model-info", "repo-tree"], "vnext-g00a raw response body policy mismatch")
    require_gate(
        resolution_policy.get("lfs_metadata_download")
        == {
            "allowed_suffixes": [".safetensors.index.json"],
            "max_bytes": 32 * 1024 * 1024,
            "selector_requirement": "weight_source_exact_path_required_if_sharded",
            "sha256_must_match_lfs_oid": True,
        },
        "vnext-g00a LFS metadata download policy mismatch",
    )
    requests = require_list(resolution.get("requests"), "vnext-g00a model resolution requests")
    require_gate(requests, "vnext-g00a model resolution has no live request provenance")
    request_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    expected_request_keys: set[tuple[str, str]] = set()
    expected_metadata_urls: set[str] = set()
    for request_index, raw in enumerate(requests):
        request = require_object(raw, f"vnext-g00a model resolution requests[{request_index}]")
        require_gate(request.get("method") == "GET", f"vnext-g00a request method mismatch at {request_index}")
        kind = require_string(request.get("kind"), f"vnext-g00a request[{request_index}].kind")
        require_gate(
            kind in {"model-info", "repo-tree", "metadata-file"},
            f"vnext-g00a request kind mismatch at {request_index}: {kind}",
        )
        url = require_string(request.get("url"), f"vnext-g00a request[{request_index}].url")
        require_gate(url.startswith("https://huggingface.co/"), f"vnext-g00a request URL mismatch at {request_index}")
        status = request.get("status")
        require_gate(isinstance(status, int) and not isinstance(status, bool) and 200 <= status < 300, f"vnext-g00a request status mismatch at {request_index}")
        response_bytes = request.get("response_bytes")
        require_gate(isinstance(response_bytes, int) and not isinstance(response_bytes, bool) and response_bytes > 0, f"vnext-g00a request size mismatch at {request_index}")
        if kind == "metadata-file":
            require_gate(
                response_bytes <= 32 * 1024 * 1024,
                f"vnext-g00a metadata response exceeds download limit at {request_index}",
            )
        require_sha256(request.get("response_sha256"), f"vnext-g00a request[{request_index}].response_sha256")
        key = (kind, url)
        require_gate(key not in request_lookup, f"vnext-g00a duplicate request provenance: {kind} {url}")
        request_lookup[key] = request
        if kind in {"model-info", "repo-tree"}:
            decoded_request_body(request, f"vnext-g00a request[{request_index}]")
        else:
            require_gate("response_body_base64" not in request, f"vnext-g00a metadata request unexpectedly embeds its body: {url}")

    def validate_live_source_response(raw_source: Any, label: str) -> None:
        source = require_object(raw_source, label)
        model_url = require_string(source.get("model_request_url"), f"{label}.model_request_url")
        requested_revision = require_object(
            source.get("requested_revision"),
            f"{label}.requested_revision",
        )
        requested_status = requested_revision.get("status")
        require_gate(
            requested_status in {"pinned", "resolution_required", "same_as_weight_revision"},
            f"{label}.requested_revision.status is invalid",
        )
        if requested_status == "resolution_required":
            require_gate(
                requested_revision.get("value") is None,
                f"{label}.requested_revision.value must be null when resolution is required",
            )
        else:
            require_gate(
                requested_revision.get("value") == source.get("revision"),
                f"{label}.requested_revision.value differs from the resolved revision",
            )
        expected_model_url = f"https://huggingface.co/api/models/{source['repo']}"
        if requested_status != "resolution_required":
            expected_model_url += f"/revision/{source['revision']}"
        require_gate(
            model_url == expected_model_url,
            f"{label}.model_request_url is not canonical for its requested revision",
        )
        expected_request_keys.add(("model-info", model_url))
        model_request = require_object(request_lookup.get(("model-info", model_url)), f"{label} model request")
        model_body = require_object(decoded_request_body(model_request, f"{label} model request"), f"{label} model response")
        require_gate(model_body.get("sha") == source.get("revision"), f"{label} model response revision mismatch")
        tree_entries: dict[str, dict[str, Any]] = {}
        for tree_url_raw in require_list(source.get("tree_request_urls"), f"{label}.tree_request_urls"):
            tree_url = require_string(tree_url_raw, f"{label}.tree_request_url")
            expected_request_keys.add(("repo-tree", tree_url))
            tree_request = require_object(request_lookup.get(("repo-tree", tree_url)), f"{label} tree request")
            tree_body = require_list(decoded_request_body(tree_request, f"{label} tree request"), f"{label} tree response")
            for entry_index, raw_entry in enumerate(tree_body):
                entry = require_object(raw_entry, f"{label} tree response[{entry_index}]")
                if entry.get("type") not in {"file", None}:
                    continue
                path = require_string(entry.get("path"), f"{label} tree response[{entry_index}].path")
                require_gate(not Path(path).is_absolute() and ".." not in Path(path).parts, f"{label} unsafe tree path: {path}")
                require_gate(path not in tree_entries, f"{label} duplicate tree path: {path}")
                tree_entries[path] = entry
        primary_files = require_list(source.get("files"), f"{label}.files")
        license_files = require_list(require_object(source.get("license"), f"{label}.license").get("files"), f"{label}.license.files")
        primary_paths = {
            require_string(require_object(row, f"{label}.file").get("path"), f"{label}.file.path")
            for row in primary_files
        }
        license_paths = {
            require_string(require_object(row, f"{label}.license.file").get("path"), f"{label}.license.file.path")
            for row in license_files
        }
        require_gate(
            len(primary_paths) == len(primary_files),
            f"{label}.files contains duplicate paths",
        )
        require_gate(
            len(license_paths) == len(license_files),
            f"{label}.license.files contains duplicate paths",
        )
        require_gate(
            primary_paths.isdisjoint(license_paths),
            f"{label}.files and license.files contain duplicate paths",
        )
        allowed_license_basenames = {
            "copying",
            "license",
            "license.md",
            "license.txt",
            "notice",
            "notice.txt",
        }
        require_gate(
            all(Path(path).name.lower() in allowed_license_basenames for path in license_paths),
            f"{label}.license.files contains a non-license path",
        )
        for file_index, raw_file in enumerate([*primary_files, *license_files]):
            file_row = require_object(raw_file, f"{label}.file[{file_index}]")
            path = require_string(file_row.get("path"), f"{label}.file[{file_index}].path")
            tree_entry = require_object(tree_entries.get(path), f"{label} tree fact {path}")
            require_gate(tree_entry.get("oid") == file_row.get("git_oid"), f"{label} tree Git OID mismatch: {path}")
            require_gate(tree_entry.get("size") == file_row.get("size_bytes"), f"{label} tree size mismatch: {path}")
            identity_source = require_string(
                file_row.get("sha256_source"),
                f"{label} file SHA256 source {path}",
            )
            require_gate(
                identity_source in {"downloaded_content", "hugging_face_lfs_oid"},
                f"{label} file SHA256 source is invalid: {path}",
            )
            if identity_source == "hugging_face_lfs_oid":
                lfs = require_object(tree_entry.get("lfs"), f"{label} tree LFS fact {path}")
                lfs_oid = require_string(lfs.get("oid"), f"{label} tree LFS OID {path}").lower()
                if lfs_oid.startswith("sha256:"):
                    lfs_oid = lfs_oid.removeprefix("sha256:")
                require_gate(lfs_oid == file_row.get("sha256"), f"{label} tree LFS SHA256 mismatch: {path}")
                require_gate(lfs.get("size") == file_row.get("size_bytes"), f"{label} tree LFS size mismatch: {path}")
                if path.endswith(".safetensors.index.json"):
                    require_gate(file_row.get("size_bytes") <= 32 * 1024 * 1024, f"{label} LFS index exceeds metadata limit: {path}")
                    content_url = require_string(file_row.get("content_request_url"), f"{label} LFS index URL {path}")
                    expected_url = f"https://huggingface.co/{source['repo']}/resolve/{source['revision']}/{path}"
                    require_gate(file_row.get("lfs_metadata_downloaded") is True and content_url == expected_url, f"{label} LFS index download evidence mismatch: {path}")
                    expected_metadata_urls.add(content_url)
                    expected_request_keys.add(("metadata-file", content_url))
                    content_request = require_object(request_lookup.get(("metadata-file", content_url)), f"{label} LFS index request {path}")
                    require_gate(content_request.get("response_sha256") == file_row.get("sha256"), f"{label} LFS index SHA256 mismatch: {path}")
                    require_gate(content_request.get("response_bytes") == file_row.get("size_bytes"), f"{label} LFS index size mismatch: {path}")
                else:
                    require_gate("content_request_url" not in file_row and "lfs_metadata_downloaded" not in file_row, f"{label} non-index LFS file has download evidence: {path}")
            else:
                require_gate(
                    tree_entry.get("lfs") is None,
                    f"{label} downloaded metadata is LFS-backed in the authoritative tree: {path}",
                )
                require_gate(
                    "lfs_oid" not in file_row
                    and "lfs_metadata_downloaded" not in file_row,
                    f"{label} downloaded metadata carries LFS identity or flags: {path}",
                )
                content_url = require_string(file_row.get("content_request_url"), f"{label} metadata URL {path}")
                expected_metadata_urls.add(content_url)
                expected_request_keys.add(("metadata-file", content_url))
                content_request = require_object(request_lookup.get(("metadata-file", content_url)), f"{label} metadata request {path}")
                require_gate(content_request.get("response_sha256") == file_row.get("sha256"), f"{label} metadata SHA256 mismatch: {path}")
                require_gate(content_request.get("response_bytes") == file_row.get("size_bytes"), f"{label} metadata size mismatch: {path}")

    resolved_lanes = require_list(resolution.get("lanes"), "vnext-g00a model resolution lanes")
    input_resolution = read_json_object(root / "model-resolution.input.json", "vnext-g00a input model resolution")
    require_gate(input_resolution.get("schema_version") == 1 and input_resolution.get("artifact_type") == "runtime_vnext_model_resolution", "vnext-g00a input model resolution schema mismatch")
    require_gate(input_resolution.get("catalog_sha256") == resolution.get("catalog_sha256"), "vnext-g00a input/live model catalog mismatch")
    input_source = require_object(input_resolution.get("source"), "vnext-g00a input model resolution source")
    require_gate(input_source.get("git_sha") == collector_sha and input_source.get("dirty") is False and input_source.get("status_short") == [], "vnext-g00a input model resolution source is stale or dirty")
    input_resolver = require_object(input_resolution.get("resolver"), "vnext-g00a input model resolution resolver")
    require_gate(input_resolver == resolution_resolver, "vnext-g00a input/live resolver identity mismatch")
    input_lanes = require_list(input_resolution.get("lanes"), "vnext-g00a input model resolution lanes")
    locked_lanes = require_list(lock.get("models"), "vnext-g00a locked model lanes")
    require_gate(len(input_lanes) == 12 and len(resolved_lanes) == 12 and len(locked_lanes) == 12, "vnext-g00a model lane count mismatch")
    require_gate(pretty_json_sha256(locked_lanes) == resolution_ref.get("live_facts_sha256"), "vnext-g00a live model facts digest mismatch")
    input_by_id = {require_string(require_object(row, "vnext-g00a input lane").get("catalog_lane_id"), "vnext-g00a input lane id"): row for row in input_lanes}
    resolved_by_id = {require_string(require_object(row, "vnext-g00a resolved lane").get("catalog_lane_id"), "vnext-g00a resolved lane id"): row for row in resolved_lanes}
    locked_by_id = {require_string(require_object(row, "vnext-g00a locked lane").get("catalog_lane_id"), "vnext-g00a locked lane id"): row for row in locked_lanes}
    require_gate(len(input_by_id) == 12 and len(resolved_by_id) == 12 and set(input_by_id) == set(resolved_by_id) == set(locked_by_id), "vnext-g00a model lane identity mismatch")
    models_catalog = read_json_object(root / "models.catalog.json", "vnext-g00a models catalog")
    catalog_lanes_by_id = {
        require_string(require_object(row, "vnext-g00a catalog lane").get("id"), "vnext-g00a catalog lane id"): row
        for row in require_list(models_catalog.get("models"), "vnext-g00a models catalog lanes")
    }
    expected_weight_identity_count = validate_vnext_catalog_expected_weight_facts(
        models_catalog,
        resolved_by_id,
    )
    expected_pairs = {(model_id, backend) for model_id in VNEXT_RESOLUTION_MODEL_IDS.values() for backend in ("cuda", "metal")}
    actual_pairs: set[tuple[str, str]] = set()
    for lane_id in sorted(locked_by_id):
        locked_lane = require_object(locked_by_id[lane_id], f"vnext-g00a locked lane {lane_id}")
        resolved_lane = require_object(resolved_by_id[lane_id], f"vnext-g00a resolved lane {lane_id}")
        input_lane = require_object(input_by_id[lane_id], f"vnext-g00a input lane {lane_id}")
        catalog_lane = require_object(catalog_lanes_by_id.get(lane_id), f"vnext-g00a catalog lane {lane_id}")
        catalog_reference = require_object(
            catalog_lane.get("reference"),
            f"vnext-g00a catalog lane {lane_id}.reference",
        )
        weight_revision_rule = require_object(
            catalog_lane.get("revision"),
            f"vnext-g00a catalog lane {lane_id}.revision",
        )
        expected_requested_revisions: dict[str, dict[str, Any]] = {
            "weight_source": {
                "status": weight_revision_rule.get("status"),
                "value": weight_revision_rule.get("value"),
            }
        }
        semantic_revision_rule = require_object(
            catalog_reference.get("semantic_revision"),
            f"vnext-g00a catalog lane {lane_id}.reference.semantic_revision",
        )
        expected_requested_revisions["semantic_source"] = (
            copy.deepcopy(expected_requested_revisions["weight_source"])
            if semantic_revision_rule.get("status") == "same_as_weight_revision"
            else {
                "status": semantic_revision_rule.get("status"),
                "value": semantic_revision_rule.get("value"),
            }
        )
        if catalog_reference.get("tokenizer_repo") is not None:
            tokenizer_revision_rule = require_object(
                catalog_reference.get("tokenizer_revision"),
                f"vnext-g00a catalog lane {lane_id}.reference.tokenizer_revision",
            )
            expected_requested_revisions["tokenizer_source"] = {
                "status": tokenizer_revision_rule.get("status"),
                "value": tokenizer_revision_rule.get("value"),
            }
        allowed_lfs_index_paths = {
            str(selector["path"])
            for selector in require_list(catalog_lane.get("files"), f"vnext-g00a catalog lane {lane_id}.files")
            if isinstance(selector, dict)
            and selector.get("required_if_sharded") is True
            and "path" in selector
            and str(selector["path"]).endswith(".safetensors.index.json")
        }
        resolved_weight = require_object(resolved_lane.get("weight_source"), f"vnext-g00a live {lane_id}.weight_source")
        resolved_weight_paths = {
            require_string(require_object(row, f"vnext-g00a live {lane_id}.weight file").get("path"), f"vnext-g00a live {lane_id}.weight path")
            for row in require_list(resolved_weight.get("files"), f"vnext-g00a live {lane_id}.weight files")
        }
        sharded_weight = validate_catalog_weight_paths(
            catalog_lane,
            resolved_weight_paths,
            f"vnext-g00a live {lane_id}.weight_source",
        )
        for source_name in ("weight_source", "semantic_source", "tokenizer_source"):
            if resolved_lane.get(source_name) is not None:
                resolved_source = require_object(resolved_lane.get(source_name), f"vnext-g00a live {lane_id}.{source_name}")
                require_gate(
                    resolved_source.get("requested_revision")
                    == expected_requested_revisions.get(source_name),
                    f"vnext-g00a live {lane_id}.{source_name}.requested_revision differs from catalog",
                )
                source_rows = [
                    *require_list(resolved_source.get("files"), f"vnext-g00a live {lane_id}.{source_name}.files"),
                    *require_list(require_object(resolved_source.get("license"), f"vnext-g00a live {lane_id}.{source_name}.license").get("files"), f"vnext-g00a live {lane_id}.{source_name}.license.files"),
                ]
                for raw_file in source_rows:
                    file_row = require_object(raw_file, f"vnext-g00a live {lane_id}.{source_name} file")
                    if file_row.get("lfs_metadata_downloaded") is True:
                        require_gate(
                            source_name == "weight_source"
                            and sharded_weight
                            and file_row.get("path") in allowed_lfs_index_paths,
                            f"vnext-g00a LFS metadata download is outside exact weight index selectors: {lane_id}.{source_name}",
                        )
                validate_live_source_response(
                    resolved_lane.get(source_name),
                    f"vnext-g00a live {lane_id}.{source_name}",
                )
        official_rule_raw = catalog_reference.get("official_upstream")
        official_raw = resolved_lane.get("official_upstream")
        require_gate(
            (official_rule_raw is None) == (official_raw is None),
            f"vnext-g00a live {lane_id}.official_upstream catalog presence mismatch",
        )
        if official_rule_raw is not None:
            official_rule = require_object(
                official_rule_raw,
                f"vnext-g00a catalog lane {lane_id}.official_upstream",
            )
            official = require_object(
                official_raw,
                f"vnext-g00a live {lane_id}.official_upstream",
            )
            official_revision = require_string(
                require_object(
                    official_rule.get("revision"),
                    f"vnext-g00a catalog lane {lane_id}.official_upstream.revision",
                ).get("value"),
                f"vnext-g00a catalog lane {lane_id}.official_upstream.revision.value",
            )
            official_repo = require_string(
                official_rule.get("repo"),
                f"vnext-g00a catalog lane {lane_id}.official_upstream.repo",
            )
            require_gate(
                official.get("repo") == official_repo
                and official.get("revision") == official_revision,
                f"vnext-g00a live {lane_id}.official_upstream repo/revision mismatch",
            )
            require_gate(
                official_rule.get("required_gated") is True
                and official.get("gated") not in {None, False},
                f"vnext-g00a live {lane_id}.official_upstream gated evidence mismatch",
            )
            require_gate(
                official.get("access_note") == official_rule.get("access_note")
                and official.get("verification_method")
                == "mirror_content_sha256_and_official_git_blob_oid",
                f"vnext-g00a live {lane_id}.official_upstream verification policy mismatch",
            )
            semantic_source = require_object(
                resolved_lane.get("semantic_source"),
                f"vnext-g00a live {lane_id}.semantic_source",
            )
            require_gate(
                official.get("mirror_repo") == semantic_source.get("repo")
                and official.get("mirror_revision") == semantic_source.get("revision"),
                f"vnext-g00a live {lane_id}.official_upstream mirror source mismatch",
            )
            official_model_url = require_string(
                official.get("model_request_url"),
                f"vnext-g00a live {lane_id}.official_upstream.model_request_url",
            )
            require_gate(
                official_model_url
                == f"https://huggingface.co/api/models/{official_repo}/revision/{official_revision}",
                f"vnext-g00a live {lane_id}.official_upstream model URL mismatch",
            )
            expected_request_keys.add(("model-info", official_model_url))
            official_model_request = require_object(
                request_lookup.get(("model-info", official_model_url)),
                f"vnext-g00a live {lane_id}.official_upstream model request",
            )
            official_model_body = require_object(
                decoded_request_body(
                    official_model_request,
                    f"vnext-g00a live {lane_id}.official_upstream model request",
                ),
                f"vnext-g00a live {lane_id}.official_upstream model response",
            )
            require_gate(
                official_model_body.get("sha") == official.get("revision"),
                f"vnext-g00a live {lane_id}.official_upstream model response revision mismatch",
            )
            official_tree_entries: dict[str, dict[str, Any]] = {}
            for official_tree_url_raw in require_list(
                official.get("tree_request_urls"),
                f"vnext-g00a live {lane_id}.official_upstream.tree_request_urls",
            ):
                official_tree_url = require_string(
                    official_tree_url_raw,
                    f"vnext-g00a live {lane_id}.official_upstream.tree_request_url",
                )
                expected_request_keys.add(("repo-tree", official_tree_url))
                require_gate(
                    official_tree_url.startswith(
                        f"https://huggingface.co/api/models/{official_repo}/tree/{official_revision}?"
                    ),
                    f"vnext-g00a live {lane_id}.official_upstream tree URL mismatch",
                )
                official_tree_request = require_object(
                    request_lookup.get(("repo-tree", official_tree_url)),
                    f"vnext-g00a live {lane_id}.official_upstream tree request",
                )
                official_tree_body = require_list(
                    decoded_request_body(
                        official_tree_request,
                        f"vnext-g00a live {lane_id}.official_upstream tree request",
                    ),
                    f"vnext-g00a live {lane_id}.official_upstream tree response",
                )
                for tree_index, tree_raw in enumerate(official_tree_body):
                    tree_row = require_object(
                        tree_raw,
                        f"vnext-g00a live {lane_id}.official_upstream tree response[{tree_index}]",
                    )
                    if tree_row.get("type") not in {"file", None}:
                        continue
                    tree_path = require_string(
                        tree_row.get("path"),
                        f"vnext-g00a live {lane_id}.official_upstream tree response[{tree_index}].path",
                    )
                    require_gate(
                        not Path(tree_path).is_absolute()
                        and ".." not in Path(tree_path).parts
                        and tree_path not in official_tree_entries,
                        f"vnext-g00a live {lane_id}.official_upstream tree path is unsafe or duplicate: {tree_path}",
                    )
                    official_tree_entries[tree_path] = tree_row
            expected_match_paths = require_list(
                official_rule.get("blob_oid_match_files"),
                f"vnext-g00a catalog lane {lane_id}.official_upstream.blob_oid_match_files",
            )
            expected_oids = require_object(
                official_rule.get("expected_git_oids"),
                f"vnext-g00a catalog lane {lane_id}.official_upstream.expected_git_oids",
            )
            expected_hashes = require_object(
                official_rule.get("expected_content_sha256"),
                f"vnext-g00a catalog lane {lane_id}.official_upstream.expected_content_sha256",
            )
            expected_sizes = require_object(
                official_rule.get("expected_size_bytes"),
                f"vnext-g00a catalog lane {lane_id}.official_upstream.expected_size_bytes",
            )
            matches = require_list(
                official.get("mirror_blob_oid_matches"),
                f"vnext-g00a live {lane_id}.official_upstream.mirror_blob_oid_matches",
            )
            require_gate(
                [row.get("path") if isinstance(row, dict) else None for row in matches]
                == expected_match_paths,
                f"vnext-g00a live {lane_id}.official_upstream match path matrix mismatch",
            )
            for match_raw in matches:
                match = require_object(
                    match_raw,
                    f"vnext-g00a live {lane_id}.official_upstream match",
                )
                match_path = require_string(
                    match.get("path"),
                    f"vnext-g00a live {lane_id}.official_upstream match.path",
                )
                require_gate(
                    match.get("git_oid") == expected_oids.get(match_path)
                    and match.get("content_sha256") == expected_hashes.get(match_path)
                    and match.get("size_bytes") == expected_sizes.get(match_path),
                    f"vnext-g00a live {lane_id}.official_upstream catalog match mismatch for {match_path}",
                )
                tree_row = require_object(
                    official_tree_entries.get(match_path),
                    f"vnext-g00a live {lane_id}.official_upstream tree fact {match_path}",
                )
                require_gate(
                    tree_row.get("oid") == match.get("git_oid")
                    and tree_row.get("size") == match.get("size_bytes"),
                    f"vnext-g00a live {lane_id}.official_upstream tree identity mismatch for {match_path}",
                )
        pair = (
            require_string(locked_lane.get("model_id"), f"vnext-g00a locked lane {lane_id}.model_id"),
            require_string(locked_lane.get("backend"), f"vnext-g00a locked lane {lane_id}.backend"),
        )
        actual_pairs.add(pair)
        require_gate(pair == (resolved_lane.get("model_id"), resolved_lane.get("backend")), f"vnext-g00a model/backend drift: {lane_id}")
        require_gate(pair == (input_lane.get("model_id"), input_lane.get("backend")), f"vnext-g00a input/live model/backend drift: {lane_id}")
        require_gate(locked_lane.get("format") == resolved_lane.get("format"), f"vnext-g00a format drift: {lane_id}")
        require_gate(input_lane.get("format") == resolved_lane.get("format"), f"vnext-g00a input/live format drift: {lane_id}")
        for source_name in ("weight_source", "semantic_source"):
            require_gate(normalized_model_source(locked_lane.get(source_name), f"vnext-g00a locked {lane_id}.{source_name}") == normalized_model_source(resolved_lane.get(source_name), f"vnext-g00a resolved {lane_id}.{source_name}"), f"vnext-g00a source drift: {lane_id}.{source_name}")
            require_gate(normalized_model_source(input_lane.get(source_name), f"vnext-g00a input {lane_id}.{source_name}") == normalized_model_source(resolved_lane.get(source_name), f"vnext-g00a live {lane_id}.{source_name}"), f"vnext-g00a input/live source drift: {lane_id}.{source_name}")
        if locked_lane.get("tokenizer_source") is None or resolved_lane.get("tokenizer_source") is None:
            require_gate(locked_lane.get("tokenizer_source") is None and resolved_lane.get("tokenizer_source") is None, f"vnext-g00a tokenizer source nullability drift: {lane_id}")
        else:
            require_gate(normalized_model_source(locked_lane.get("tokenizer_source"), f"vnext-g00a locked {lane_id}.tokenizer_source") == normalized_model_source(resolved_lane.get("tokenizer_source"), f"vnext-g00a resolved {lane_id}.tokenizer_source"), f"vnext-g00a tokenizer source drift: {lane_id}")
        if input_lane.get("tokenizer_source") is None or resolved_lane.get("tokenizer_source") is None:
            require_gate(input_lane.get("tokenizer_source") is None and resolved_lane.get("tokenizer_source") is None, f"vnext-g00a input/live tokenizer source nullability drift: {lane_id}")
        else:
            require_gate(normalized_model_source(input_lane.get("tokenizer_source"), f"vnext-g00a input {lane_id}.tokenizer_source") == normalized_model_source(resolved_lane.get("tokenizer_source"), f"vnext-g00a live {lane_id}.tokenizer_source"), f"vnext-g00a input/live tokenizer source drift: {lane_id}")
        for field in ("chat_template", "generation_config", "official_upstream"):
            require_gate(locked_lane.get(field) == resolved_lane.get(field), f"vnext-g00a resolved field drift: {lane_id}.{field}")
            require_gate(input_lane.get(field) == resolved_lane.get(field), f"vnext-g00a input/live field drift: {lane_id}.{field}")
    require_gate(actual_pairs == expected_pairs, "vnext-g00a exact six-model CUDA/Metal matrix mismatch")
    actual_metadata_urls = {
        url for kind, url in request_lookup if kind == "metadata-file"
    }
    require_gate(
        actual_metadata_urls == expected_metadata_urls,
        "vnext-g00a metadata request provenance differs from selected metadata files",
    )
    require_gate(
        set(request_lookup) == expected_request_keys,
        "vnext-g00a network request provenance differs from selected sources and files",
    )

    return {
        "kind": "vnext-g00a",
        "child_manifest": {"path": str(manifest_path), "sha256": require_sha256(child_manifest_sha256, "vnext-g00a delegated manifest SHA256")},
        "checkpoint": {"id": "G00a", "unlocks": ["G01A"], "does_not_prove": sorted(VNEXT_G00A_DOES_NOT_PROVE)},
        "collector": {"git_sha": collector_sha, "git_tree_sha": collector_tree, "contracts_sha256": collector["contracts_sha256"]},
        "model_facts_lock": {"path": lock_rel, "sha256": lock_digest},
        "model_lane_count": len(locked_lanes),
        "catalog_expected_weight_identity_count": expected_weight_identity_count,
        "historical_bug_counts": {"families": 15, "cases": 28},
        "artifact_index_sha256": canonical_json_sha256(child_manifest["artifact_index"]),
    }


def validate_vnext_g00f_provenance(
    lane_command: LaneCommand,
    child_manifest: dict[str, Any],
    child_manifest_sha256: str,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    manifest_path = lane_command.child_manifest_path
    require_gate(manifest_path is not None, "vnext-g00f delegated manifest path is missing")
    checkpoint_root = manifest_path.parent.resolve()
    require_gate(
        manifest_path.resolve() == checkpoint_root / "manifest.json",
        "vnext-g00f child manifest path mismatch",
    )
    require_gate(
        set(child_manifest)
        == {
            "schema_version",
            "artifact_type",
            "checkpoint_id",
            "lane",
            "status",
            "canonical",
            "artifact_dir",
            "source",
            "g00a",
            "unlocks",
            "does_not_prove",
            "pass_line",
        },
        "vnext-g00f manifest field set mismatch",
    )
    require_gate(
        child_manifest.get("schema_version") == 1
        and child_manifest.get("artifact_type")
        == "runtime_vnext_g00f_facts_manifest"
        and child_manifest.get("checkpoint_id") == "G00F"
        and child_manifest.get("lane") == "runtime-vnext-g00f"
        and child_manifest.get("status") == "pass"
        and child_manifest.get("canonical") is True,
        "vnext-g00f identity/status mismatch",
    )
    require_gate(
        Path(require_string(child_manifest.get("artifact_dir"), "vnext-g00f artifact_dir")).resolve()
        == checkpoint_root,
        "vnext-g00f artifact_dir mismatch",
    )
    expected_pass = f"FERRUM RUNTIME VNEXT G00F FACTS PASS: {checkpoint_root}"
    require_gate(child_manifest.get("pass_line") == expected_pass, "vnext-g00f pass_line mismatch")
    require_exact_string_set(child_manifest.get("unlocks"), {"S0A", "S1"}, "vnext-g00f unlocks")
    require_exact_string_set(
        child_manifest.get("does_not_prove"),
        {
            "G00P",
            "G01",
            "G01B",
            "model_migration",
            "performance",
            "production_wiring",
            "release",
        },
        "vnext-g00f does_not_prove",
    )

    source = require_object(child_manifest.get("source"), "vnext-g00f source")
    require_gate(
        set(source) == {"git_sha", "git_tree_sha", "dirty", "status_short"}
        and source.get("dirty") is False
        and source.get("status_short") == [],
        "vnext-g00f source shape/dirty state mismatch",
    )
    source_sha = require_git_sha(source.get("git_sha"), "vnext-g00f source.git_sha")
    source_tree = require_git_sha(source.get("git_tree_sha"), "vnext-g00f source.git_tree_sha")
    if verify_checkout:
        require_gate(git_sha() == source_sha, "vnext-g00f source SHA is stale")
        require_gate(
            git_output(["rev-parse", "HEAD^{tree}"]) == source_tree,
            "vnext-g00f source tree is stale",
        )
        require_gate(not git_dirty_status()["is_dirty"], "vnext-g00f checkout is dirty")

    g00a = require_object(child_manifest.get("g00a"), "vnext-g00f G00a binding")
    require_gate(
        set(g00a)
        == {
            "outer_manifest",
            "child_manifest",
            "artifact_index_sha256",
            "model_lane_count",
            "historical_bug_counts",
            "facts_reused_without_copy",
        }
        and g00a.get("facts_reused_without_copy") is True
        and g00a.get("model_lane_count") == 12
        and g00a.get("historical_bug_counts") == {"families": 15, "cases": 28},
        "vnext-g00f G00a binding summary mismatch",
    )
    outer_ref = require_object(g00a.get("outer_manifest"), "vnext-g00f G00a outer ref")
    child_ref = require_object(g00a.get("child_manifest"), "vnext-g00f G00a child ref")
    require_gate(
        set(outer_ref) == {"path", "sha256"} and set(child_ref) == {"path", "sha256"},
        "vnext-g00f G00a reference field set mismatch",
    )
    outer_path = Path(require_string(outer_ref.get("path"), "vnext-g00f G00a outer path")).resolve()
    child_path = Path(require_string(child_ref.get("path"), "vnext-g00f G00a child path")).resolve()
    require_gate(
        outer_path.parent == child_path.parent
        and outer_path.name == "gate.manifest.json"
        and child_path.name == "manifest.json",
        "vnext-g00f G00a outer/child paths mismatch",
    )
    require_gate(
        sha256(outer_path) == require_sha256(outer_ref.get("sha256"), "vnext-g00f G00a outer SHA256")
        and sha256(child_path) == require_sha256(child_ref.get("sha256"), "vnext-g00f G00a child SHA256"),
        "vnext-g00f G00a manifest identity mismatch",
    )
    outer = read_json_object(outer_path, "vnext-g00f bound G00a outer manifest")
    child = read_json_object(child_path, "vnext-g00f bound G00a child manifest")
    require_gate(
        outer.get("lane") == "vnext-g00a"
        and outer.get("status") == "pass"
        and outer.get("git_sha") == source_sha
        and outer.get("dirty_status") == {"is_dirty": False, "status_short": []},
        "vnext-g00f bound G00a outer manifest is stale",
    )
    outer_child = require_object(outer.get("child_artifacts"), "vnext-g00f G00a outer child artifacts")
    require_gate(
        outer_child.get("child_manifest") == child_ref
        and outer_child.get("artifact_index_sha256") == g00a.get("artifact_index_sha256"),
        "vnext-g00f G00a outer/child binding mismatch",
    )
    g00a_provenance = validate_vnext_g00a_provenance(
        LaneCommand(
            cmd=[],
            expected_child_pass_line=child.get("pass_line"),
            child_manifest_path=child_path,
            provenance_kind="vnext-g00a",
        ),
        child,
        require_sha256(child_ref.get("sha256"), "vnext-g00f G00a child SHA256"),
        verify_checkout=verify_checkout,
    )
    require_gate(
        g00a_provenance["artifact_index_sha256"]
        == require_sha256(g00a.get("artifact_index_sha256"), "vnext-g00f G00a artifact index SHA256"),
        "vnext-g00f G00a artifact index mismatch",
    )
    return {
        "kind": "vnext-g00f",
        "child_manifest": {
            "path": str(manifest_path),
            "sha256": require_sha256(child_manifest_sha256, "vnext-g00f manifest SHA256"),
        },
        "checkpoint": {
            "id": "G00F",
            "unlocks": ["S0A", "S1"],
            "does_not_prove": sorted(
                {
                    "G00P",
                    "G01",
                    "G01B",
                    "model_migration",
                    "performance",
                    "production_wiring",
                    "release",
                }
            ),
        },
        "source": {"git_sha": source_sha, "git_tree_sha": source_tree},
        "g00a": g00a,
    }


def validate_vnext_g01a_s0a_provenance(
    lane_command: LaneCommand,
    child_manifest: dict[str, Any],
    child_manifest_sha256: str,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    manifest_path = lane_command.child_manifest_path
    require_gate(manifest_path is not None, "vnext-g01a S0A manifest path is missing")
    checkpoint_root = manifest_path.parent.resolve()
    output_root = checkpoint_root.parent.resolve()
    require_gate(
        manifest_path.resolve() == output_root / "g01a-contract-split/manifest.json",
        "vnext-g01a S0A child manifest path mismatch",
    )
    require_gate(
        set(child_manifest)
        == {
            "schema_version",
            "artifact_type",
            "checkpoint_id",
            "lane",
            "status",
            "canonical",
            "artifact_dir",
            "output_root",
            "source",
            "baseline_commit",
            "g00f",
            "inventory_document",
            "adr_source",
            "public_owner_evidence",
            "compile_evidence",
            "artifact_count",
            "artifact_index",
            "unlocks",
            "does_not_prove",
            "started_at",
            "finished_at",
            "duration_seconds",
            "pass_line",
        },
        "vnext-g01a S0A manifest field set mismatch",
    )
    require_gate(
        child_manifest.get("schema_version") == 1
        and child_manifest.get("artifact_type")
        == "runtime_vnext_g01a_contract_split_manifest"
        and child_manifest.get("checkpoint_id") == "G01A-S0A"
        and child_manifest.get("lane") == "runtime-vnext-g01a-contract-split"
        and child_manifest.get("status") == "pass"
        and child_manifest.get("canonical") is True,
        "vnext-g01a S0A identity/status mismatch",
    )
    require_gate(
        Path(require_string(child_manifest.get("artifact_dir"), "vnext-g01a S0A artifact_dir")).resolve()
        == checkpoint_root
        and Path(require_string(child_manifest.get("output_root"), "vnext-g01a S0A output_root")).resolve()
        == output_root,
        "vnext-g01a S0A output path mismatch",
    )
    expected_pass = f"FERRUM RUNTIME VNEXT G01A CONTRACT SPLIT PASS: {output_root}"
    require_gate(child_manifest.get("pass_line") == expected_pass, "vnext-g01a S0A pass_line mismatch")
    require_gate(
        child_manifest.get("baseline_commit")
        == "b5377b12464b60203a3fe57a6de4c9952ed2474b",
        "vnext-g01a S0A baseline commit mismatch",
    )
    require_exact_string_set(child_manifest.get("unlocks"), {"G01B", "S1"}, "vnext-g01a S0A unlocks")
    require_exact_string_set(
        child_manifest.get("does_not_prove"),
        {
            "G01",
            "G01B",
            "model_migration",
            "performance",
            "production_wiring",
            "release",
        },
        "vnext-g01a S0A does_not_prove",
    )
    source = require_object(child_manifest.get("source"), "vnext-g01a S0A source")
    require_gate(
        set(source) == {"git_sha", "git_tree_sha", "dirty", "status_short"}
        and source.get("dirty") is False
        and source.get("status_short") == [],
        "vnext-g01a S0A source shape/dirty state mismatch",
    )
    source_sha = require_git_sha(source.get("git_sha"), "vnext-g01a S0A source.git_sha")
    source_tree = require_git_sha(source.get("git_tree_sha"), "vnext-g01a S0A source.git_tree_sha")
    if verify_checkout:
        require_gate(git_sha() == source_sha, "vnext-g01a S0A source SHA is stale")
        require_gate(
            git_output(["rev-parse", "HEAD^{tree}"]) == source_tree,
            "vnext-g01a S0A source tree is stale",
        )
        require_gate(not git_dirty_status()["is_dirty"], "vnext-g01a S0A checkout is dirty")

    artifact_index = validate_child_artifact_index(
        checkpoint_root, child_manifest, role_from_top_level_path=False
    )
    required_artifacts = {
        "adr.md",
        "contract-map.json",
        "public-api-migrations.json",
        "public-owner-map.json",
        "split-inventory.json",
        "compile-unit-trybuild.json",
    }
    require_gate(
        required_artifacts <= set(artifact_index),
        "vnext-g01a S0A required artifact set mismatch",
    )

    inventory_ref = require_object(
        child_manifest.get("inventory_document"), "vnext-g01a S0A inventory document"
    )
    inventory_source_path = REPO_ROOT / require_string(
        inventory_ref.get("path"), "vnext-g01a S0A inventory source path"
    )
    require_gate(
        inventory_source_path
        == REPO_ROOT / "docs/release/cleanup/20260714-inventory.md"
        and sha256(inventory_source_path)
        == require_sha256(inventory_ref.get("sha256"), "vnext-g01a S0A inventory SHA256"),
        "vnext-g01a S0A inventory source mismatch",
    )
    adr_ref = require_object(child_manifest.get("adr_source"), "vnext-g01a S0A ADR source")
    adr_source_path = REPO_ROOT / require_string(
        adr_ref.get("path"), "vnext-g01a S0A ADR source path"
    )
    adr_digest = require_sha256(adr_ref.get("sha256"), "vnext-g01a S0A ADR source SHA256")
    require_gate(
        adr_source_path
        == REPO_ROOT
        / "docs/goals/runtime-vnext-0.8.0-2026-07-10/S0A_CONTRACT_SPLIT_MAP.md"
        and sha256(adr_source_path) == adr_digest
        and artifact_index["adr.md"]["sha256"] == adr_digest,
        "vnext-g01a S0A ADR binding mismatch",
    )

    migration_ref = require_object(
        child_manifest.get("public_api_migration_source"),
        "vnext-g01a S0A public API migration source",
    )
    migration_source_path = REPO_ROOT / require_string(
        migration_ref.get("path"), "vnext-g01a S0A public API migration source path"
    )
    migration_digest = require_sha256(
        migration_ref.get("sha256"), "vnext-g01a S0A public API migration SHA256"
    )
    require_gate(
        migration_source_path
        == REPO_ROOT
        / "docs/goals/runtime-vnext-0.8.0-2026-07-10/S0A_PUBLIC_API_MIGRATIONS.json"
        and sha256(migration_source_path) == migration_digest
        and artifact_index["public-api-migrations.json"]["sha256"] == migration_digest,
        "vnext-g01a S0A public API migration binding mismatch",
    )

    owner_map = read_json_object(
        checkpoint_root / "public-owner-map.json", "vnext-g01a S0A public owner map"
    )
    require_gate(
        owner_map.get("summary")
        == {
            "baseline_items": 1490,
            "mapped_items": 1481,
            "migrated_items": 9,
            "lost_items": 0,
            "ambiguous_items": 0,
            "inaccessible_items": 0,
            "added_items": 248,
            "added_items_sha256": VNEXT_S0A_PUBLIC_API_ADDED_SHA256,
            "excluded_non_public_owner_members": 1,
            "unsupported_syntax_count": 0,
            "coverage_percent": 100.0,
            "pass": True,
        },
        "vnext-g01a S0A public owner map acceptance mismatch",
    )
    owner_evidence = require_object(
        child_manifest.get("public_owner_evidence"), "vnext-g01a S0A public owner evidence"
    )
    owner_migration = require_object(
        owner_evidence.get("migration_manifest"),
        "vnext-g01a S0A public owner migration evidence",
    )
    owner_map_migration = require_object(
        owner_map.get("migration_manifest"),
        "vnext-g01a S0A public owner migration manifest",
    )
    require_gate(
        owner_evidence.get("summary") == owner_map.get("summary")
        and isinstance(owner_evidence.get("pass_line"), str)
        and owner_evidence["pass_line"].startswith(
            "VNEXT PUBLIC OWNER MAP PASS: mapped=1481/1490 migrated=9"
        ),
        "vnext-g01a S0A public owner evidence binding mismatch",
    )
    require_gate(
        owner_migration
        == {"path": "public-api-migrations.json", "sha256": migration_digest}
        and owner_map_migration.get("sha256") == migration_digest
        and owner_map_migration.get("migration_count") == 9
        and owner_map_migration.get("expected_added_items") == 248
        and owner_map_migration.get("expected_added_items_sha256")
        == VNEXT_S0A_PUBLIC_API_ADDED_SHA256,
        "vnext-g01a S0A public owner migration evidence mismatch",
    )

    split_inventory = read_json_object(
        checkpoint_root / "split-inventory.json", "vnext-g01a S0A split inventory"
    )
    inventory_summary = require_object(
        split_inventory.get("summary"), "vnext-g01a S0A split inventory summary"
    )
    require_gate(
        split_inventory.get("schema_version") == 1
        and split_inventory.get("artifact_type")
        == "runtime_vnext_s0a_split_inventory"
        and split_inventory.get("source") == source
        and split_inventory.get("baseline", {}).get("git_commit")
        == child_manifest.get("baseline_commit")
        and inventory_summary.get("facade_count") == 3
        and inventory_summary.get("production_owner_count") == 42
        and inventory_summary.get("contract_test_target_count") == 24
        and inventory_summary.get("shared_test_support_owner_count") == 10
        and inventory_summary.get("maximum_facade_physical_lines", 501) <= 500
        and inventory_summary.get("maximum_production_owner_physical_lines", 2501) <= 2500
        and inventory_summary.get("maximum_contract_test_or_support_owner_physical_lines", 2001)
        <= 2000
        and inventory_summary.get("include_macro_count") == 0
        and inventory_summary.get("production_wildcard_parent_import_count") == 0
        and inventory_summary.get("removed_oversized_target_present_count") == 0,
        "vnext-g01a S0A split inventory acceptance mismatch",
    )
    contract_map = read_json_object(
        checkpoint_root / "contract-map.json", "vnext-g01a S0A contract map"
    )
    contract_summary = require_object(
        contract_map.get("summary"), "vnext-g01a S0A contract map summary"
    )
    require_gate(
        contract_map.get("schema_version") == 1
        and contract_map.get("artifact_type") == "runtime_vnext_s0a_contract_map"
        and contract_map.get("baseline_commit") == child_manifest.get("baseline_commit")
        and contract_summary.get("production_group_count") == 3
        and contract_summary.get("multi_module_scc_count") == 0
        and contract_summary.get("test_target_count") == 24
        and contract_summary.get("semantic_change_count") == 9
        and contract_summary.get("added_public_item_count") == 248
        and contract_summary.get("added_public_item_sha256")
        == VNEXT_S0A_PUBLIC_API_ADDED_SHA256,
        "vnext-g01a S0A contract map acceptance mismatch",
    )
    production_groups = require_list(
        contract_map.get("production_groups"), "vnext-g01a S0A production groups"
    )
    require_gate(
        {row.get("group") for row in production_groups}
        == {"resource", "execution", "event"}
        and all(
            require_object(row.get("dependency_audit"), "vnext-g01a S0A dependency audit").get(
                "multi_module_scc_count"
            )
            == 0
            for row in production_groups
        ),
        "vnext-g01a S0A dependency SCC matrix mismatch",
    )

    compile_path = checkpoint_root / "compile-unit-trybuild.json"
    compile_evidence = read_json_object(compile_path, "vnext-g01a S0A compile evidence")
    tests = require_object(compile_evidence.get("tests"), "vnext-g01a S0A compile tests")
    require_gate(
        compile_evidence.get("schema_version") == 1
        and compile_evidence.get("artifact_type")
        == "runtime_vnext_s0a_compile_unit_trybuild_evidence"
        and compile_evidence.get("status") == "pass"
        and compile_evidence.get("returncode") == 0
        and compile_evidence.get("env_overrides")
        == {
            "CARGO_BUILD_JOBS": "4",
            "RUST_TEST_THREADS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        and tests.get("expected_integration_targets") == tests.get("observed_integration_targets")
        and isinstance(tests.get("reported_passed_test_sum"), int)
        and tests["reported_passed_test_sum"] >= 100
        and isinstance(tests.get("machine_proof_line_count"), int)
        and tests["machine_proof_line_count"] >= 20,
        "vnext-g01a S0A aggregate compile/test evidence mismatch",
    )
    receipt_ref = require_object(compile_evidence.get("receipt"), "vnext-g01a S0A receipt ref")
    receipt_path, receipt_relative, receipt_digest = require_indexed_artifact(
        checkpoint_root,
        artifact_index,
        receipt_ref.get("path"),
        receipt_ref.get("sha256"),
        "vnext-g01a S0A bounded receipt",
    )
    receipt = read_json_object(receipt_path, "vnext-g01a S0A bounded receipt")
    require_gate(
        receipt_relative == "logs/all-targets.receipt.json"
        and receipt_digest == artifact_index[receipt_relative]["sha256"]
        and receipt.get("schema") == "ferrum.bounded-command-receipt.v1"
        and receipt.get("status") == "pass"
        and receipt.get("rc") == 0
        and receipt.get("violation") is None
        and receipt.get("sampling_errors") == []
        and receipt.get("cleanup") == {"process_group_gone": True},
        "vnext-g01a S0A bounded receipt acceptance mismatch",
    )
    compile_ref = require_object(
        child_manifest.get("compile_evidence"), "vnext-g01a S0A compile evidence ref"
    )
    require_gate(
        compile_ref.get("path") == "compile-unit-trybuild.json"
        and require_sha256(compile_ref.get("sha256"), "vnext-g01a S0A compile SHA256")
        == artifact_index["compile-unit-trybuild.json"]["sha256"]
        and compile_ref.get("reported_passed_test_sum") == tests["reported_passed_test_sum"]
        and compile_ref.get("machine_proof_line_count") == tests["machine_proof_line_count"],
        "vnext-g01a S0A compile manifest binding mismatch",
    )

    g00f = require_object(child_manifest.get("g00f"), "vnext-g01a S0A G00F binding")
    require_gate(
        set(g00f) == {"outer_manifest", "child_manifest", "source", "g00a"}
        and g00f.get("source") == {"git_sha": source_sha, "git_tree_sha": source_tree},
        "vnext-g01a S0A G00F source binding mismatch",
    )
    g00f_outer_ref = require_object(g00f.get("outer_manifest"), "vnext-g01a S0A G00F outer ref")
    g00f_child_ref = require_object(g00f.get("child_manifest"), "vnext-g01a S0A G00F child ref")
    g00f_outer_path = Path(
        require_string(g00f_outer_ref.get("path"), "vnext-g01a S0A G00F outer path")
    ).resolve()
    g00f_child_path = Path(
        require_string(g00f_child_ref.get("path"), "vnext-g01a S0A G00F child path")
    ).resolve()
    require_gate(
        sha256(g00f_outer_path)
        == require_sha256(g00f_outer_ref.get("sha256"), "vnext-g01a S0A G00F outer SHA256")
        and sha256(g00f_child_path)
        == require_sha256(g00f_child_ref.get("sha256"), "vnext-g01a S0A G00F child SHA256"),
        "vnext-g01a S0A G00F manifest identity mismatch",
    )
    g00f_outer = read_json_object(g00f_outer_path, "vnext-g01a S0A bound G00F outer")
    g00f_child = read_json_object(g00f_child_path, "vnext-g01a S0A bound G00F child")
    require_gate(
        g00f_outer.get("lane") == "vnext-g00f"
        and g00f_outer.get("status") == "pass"
        and g00f_outer.get("git_sha") == source_sha
        and require_object(g00f_outer.get("child_artifacts"), "vnext-g01a S0A G00F outer artifacts").get(
            "child_manifest"
        )
        == g00f_child_ref,
        "vnext-g01a S0A bound G00F outer is stale",
    )
    validate_vnext_g00f_provenance(
        LaneCommand(
            cmd=[],
            expected_child_pass_line=g00f_child.get("pass_line"),
            child_manifest_path=g00f_child_path,
            provenance_kind="vnext-g00f",
        ),
        g00f_child,
        require_sha256(g00f_child_ref.get("sha256"), "vnext-g01a S0A G00F child SHA256"),
        verify_checkout=verify_checkout,
    )
    require_gate(g00f.get("g00a") == g00f_child.get("g00a"), "vnext-g01a S0A G00F/G00a binding drift")

    return {
        "kind": "vnext-g01a-s0a",
        "child_manifest": {
            "path": str(manifest_path),
            "sha256": require_sha256(child_manifest_sha256, "vnext-g01a S0A manifest SHA256"),
        },
        "checkpoint": {
            "id": "G01A-S0A",
            "unlocks": ["G01B", "S1"],
            "does_not_prove": sorted(
                {
                    "G01",
                    "G01B",
                    "model_migration",
                    "performance",
                    "production_wiring",
                    "release",
                }
            ),
        },
        "source": {"git_sha": source_sha, "git_tree_sha": source_tree},
        "public_owner_summary": owner_map["summary"],
        "split_inventory_summary": inventory_summary,
        "contract_map_summary": contract_summary,
        "compile_summary": {
            "reported_passed_test_sum": tests["reported_passed_test_sum"],
            "machine_proof_line_count": tests["machine_proof_line_count"],
            "receipt_peaks": receipt["peaks"],
        },
        "artifact_count": len(artifact_index),
    }


def discover_vnext_g01a_contract_paths() -> set[str]:
    patterns = [
        "crates/ferrum-interfaces/src/vnext.rs",
        "crates/ferrum-interfaces/src/vnext/*.rs",
        "crates/ferrum-interfaces/src/vnext/**/*.rs",
        "crates/ferrum-interfaces/tests/vnext_*.rs",
        "crates/ferrum-interfaces/tests/ui/vnext/*.rs",
        "crates/ferrum-interfaces/tests/ui/vnext/*.stderr",
        "crates/ferrum-interfaces/tests/ui/vnext/**/*.rs",
        "crates/ferrum-interfaces/tests/ui/vnext/**/*.stderr",
    ]
    discovered = {
        line
        for line in git_output(["ls-files", "--", *patterns], default="").splitlines()
        if line.strip()
    }
    fixed = {
        "Cargo.lock",
        "crates/ferrum-interfaces/Cargo.toml",
        "crates/ferrum-interfaces/src/lib.rs",
        "docs/goals/runtime-vnext-0.8.0-2026-07-10/G01_CORE_CONTRACTS.md",
        "docs/goals/runtime-vnext-0.8.0-2026-07-10/G01A_CONTRACT_ADR.md",
        "docs/goals/runtime-vnext-0.8.0-2026-07-10/G01A_LEGACY_CONTRACT_MAP.json",
        "scripts/release/runtime_vnext_g01a_checkpoint.py",
        "scripts/release/run_gate.py",
        "scripts/release/bounded_command.py",
        "scripts/release/selftest_g0_validators.py",
    }
    return discovered | fixed


def validate_bounded_command_receipt(
    receipt_raw: Any,
    *,
    expected_command: list[str] | tuple[str, ...],
    expected_cwd: Path,
    expected_limits: dict[str, int | float],
    stdout_bytes: bytes,
    stderr_bytes: bytes,
    label: str,
) -> dict[str, Any]:
    receipt = require_object(receipt_raw, f"{label} receipt")
    require_gate(
        set(receipt)
        == {
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
        },
        f"{label} receipt field set mismatch",
    )
    require_gate(
        receipt.get("schema") == VNEXT_G01A_BOUNDED_RECEIPT_SCHEMA,
        f"{label} receipt schema mismatch",
    )
    command = require_list(receipt.get("command"), f"{label} receipt command")
    require_gate(command == list(expected_command), f"{label} receipt command mismatch")
    require_gate(
        Path(require_string(receipt.get("cwd"), f"{label} receipt cwd")).resolve()
        == expected_cwd.resolve(),
        f"{label} receipt cwd mismatch",
    )
    pid = receipt.get("pid")
    require_gate(
        isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid > 0
        and receipt.get("pgid") == pid,
        f"{label} receipt pid/pgid mismatch",
    )
    limits = require_object(receipt.get("limits"), f"{label} receipt limits")
    require_gate(
        set(limits) == set(expected_limits)
        and all(
            isinstance(limits.get(key), (int, float))
            and not isinstance(limits.get(key), bool)
            and limits[key] == value
            for key, value in expected_limits.items()
        ),
        f"{label} receipt limits mismatch",
    )
    rc = receipt.get("rc")
    require_gate(
        receipt.get("status") == "pass"
        and receipt.get("reason") == "command_completed"
        and isinstance(rc, int)
        and not isinstance(rc, bool)
        and rc == 0,
        f"{label} receipt command status mismatch",
    )
    successful_samples = receipt.get("successful_samples")
    require_gate(
        isinstance(successful_samples, int)
        and not isinstance(successful_samples, bool)
        and successful_samples >= 1,
        f"{label} receipt successful_samples invalid",
    )
    sampling_error_count = receipt.get("sampling_error_count")
    require_gate(
        isinstance(sampling_error_count, int)
        and not isinstance(sampling_error_count, bool)
        and sampling_error_count == 0
        and receipt.get("sampling_errors") == [],
        f"{label} receipt contains sampling errors",
    )
    require_gate(
        receipt.get("violation") is None,
        f"{label} receipt contains a resource violation",
    )
    require_gate(
        receipt.get("termination") == {"signals": [], "errors": []},
        f"{label} receipt termination is not clean",
    )
    cleanup = require_object(receipt.get("cleanup"), f"{label} receipt cleanup")
    require_gate(
        set(cleanup) == {"process_group_gone"}
        and cleanup.get("process_group_gone") is True,
        f"{label} receipt process group cleanup failed",
    )
    require_gate(
        bool(require_string(receipt.get("started_at"), f"{label} receipt started_at"))
        and bool(require_string(receipt.get("ended_at"), f"{label} receipt ended_at"))
        and isinstance(receipt.get("duration_seconds"), (int, float))
        and not isinstance(receipt.get("duration_seconds"), bool)
        and receipt["duration_seconds"] >= 0,
        f"{label} receipt timing evidence invalid",
    )
    peaks = require_object(receipt.get("peaks"), f"{label} receipt peaks")
    require_gate(
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
        require_gate(
            isinstance(value, int) and not isinstance(value, bool) and value >= 1,
            f"{label} receipt peak {key} invalid",
        )
    peak_pid = peaks.get("per_process_threads_pid")
    require_gate(
        isinstance(peak_pid, int)
        and not isinstance(peak_pid, bool)
        and peak_pid > 0,
        f"{label} receipt peak pid invalid",
    )
    require_gate(
        peaks["processes"] <= limits["max_processes"]
        and peaks["group_threads"] <= limits["max_group_threads"]
        and peaks["per_process_threads"] <= limits["max_per_process_threads"]
        and peaks["group_threads"] >= peaks["processes"]
        and peaks["group_threads"] >= peaks["per_process_threads"],
        f"{label} receipt peak exceeds its fixed bound",
    )
    paths: list[str] = []
    for stream, payload in (("stdout", stdout_bytes), ("stderr", stderr_bytes)):
        identity = require_object(receipt.get(stream), f"{label} receipt {stream}")
        require_gate(
            set(identity) == {"path", "sha256", "size_bytes"},
            f"{label} receipt {stream} identity field set mismatch",
        )
        paths.append(require_string(identity.get("path"), f"{label} receipt {stream}.path"))
        size_bytes = identity.get("size_bytes")
        require_gate(
            isinstance(size_bytes, int)
            and not isinstance(size_bytes, bool)
            and size_bytes == len(payload)
            and require_sha256(identity.get("sha256"), f"{label} receipt {stream}.sha256")
            == hashlib.sha256(payload).hexdigest(),
            f"{label} receipt {stream} identity mismatch",
        )
    require_gate(len(set(paths)) == 2, f"{label} receipt output paths are not distinct")
    return receipt


def vnext_g01a_bounded_profile(command: tuple[str, ...]) -> str:
    require_gate(command[:2] == ("cargo", "test"), "vnext-g01a bounded command is not cargo test")
    require_gate(
        command.count(VNEXT_G01A_TEST_THREADS_ARG) == 1
        and "--" in command
        and command.index(VNEXT_G01A_TEST_THREADS_ARG) > command.index("--"),
        f"vnext-g01a cargo test must contain exactly one {VNEXT_G01A_TEST_THREADS_ARG}",
    )
    if "--test" in command:
        target_index = command.index("--test") + 1
        require_gate(target_index < len(command), "vnext-g01a cargo test target is missing")
        target = command[target_index]
        if target in VNEXT_G01A_REQUIRED_RESOURCE_TESTS_BY_TARGET:
            return "resource"
        if target == "vnext_compile":
            return "trybuild"
    if "vnext::admission" in command and "--lib" in command:
        return "admission"
    return "regular"


def validate_vnext_g01a_command_execution(
    row: dict[str, Any],
    command: tuple[str, ...],
    stdout: str,
    stderr: str,
    index: int,
) -> dict[str, Any] | None:
    label = f"vnext-g01a compile commands[{index}]"
    execution = require_object(row.get("execution"), f"{label}.execution")
    require_gate(
        set(execution) == {"kind", "profile", "receipt", "receipt_sha256"},
        f"{label}.execution field set mismatch",
    )
    if command in VNEXT_G01A_QUALITY_COMMANDS:
        require_gate(
            execution
            == {
                "kind": "direct",
                "profile": None,
                "receipt": None,
                "receipt_sha256": None,
            }
            and row.get("env_overrides") == {"PYTHONDONTWRITEBYTECODE": "1"},
            f"{label} direct quality execution metadata mismatch",
        )
        return None
    profile_name = vnext_g01a_bounded_profile(command)
    require_gate(
        execution.get("kind") == "bounded-command"
        and execution.get("profile") == profile_name
        and row.get("env_overrides") == VNEXT_G01A_BOUNDED_TEST_ENV_OVERRIDES,
        f"{label} bounded execution metadata mismatch",
    )
    receipt = validate_bounded_command_receipt(
        execution.get("receipt"),
        expected_command=command,
        expected_cwd=REPO_ROOT,
        expected_limits=VNEXT_G01A_BOUNDED_TEST_PROFILES[profile_name],
        stdout_bytes=stdout.encode("utf-8"),
        stderr_bytes=stderr.encode("utf-8"),
        label=label,
    )
    require_gate(
        require_sha256(execution.get("receipt_sha256"), f"{label}.receipt_sha256")
        == canonical_json_sha256(receipt),
        f"{label} bounded receipt SHA256 mismatch",
    )
    return receipt


def summarize_vnext_g01a_bounded_execution(
    rows: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    profile_counts = {name: 0 for name in VNEXT_G01A_BOUNDED_TEST_PROFILES}
    receipts: list[dict[str, Any]] = []
    for profile_name, receipt in rows:
        profile_counts[profile_name] += 1
        receipts.append(receipt)
    require_gate(
        len(receipts) == VNEXT_G01A_BOUNDED_TEST_COMMAND_COUNT,
        "vnext-g01a bounded cargo test command count mismatch",
    )
    require_gate(
        profile_counts
        == {"regular": 14, "admission": 2, "resource": 14, "trybuild": 2},
        "vnext-g01a bounded profile command counts mismatch",
    )
    return {
        "runner": "scripts/release/bounded_command.py",
        "receipt_schema": VNEXT_G01A_BOUNDED_RECEIPT_SCHEMA,
        "required_command_count": VNEXT_G01A_BOUNDED_TEST_COMMAND_COUNT,
        "passed_command_count": len(receipts),
        "all_process_groups_gone": all(
            receipt["cleanup"]["process_group_gone"] for receipt in receipts
        ),
        "peak_processes": max(receipt["peaks"]["processes"] for receipt in receipts),
        "peak_group_threads": max(
            receipt["peaks"]["group_threads"] for receipt in receipts
        ),
        "peak_per_process_threads": max(
            receipt["peaks"]["per_process_threads"] for receipt in receipts
        ),
        "profile_counts": profile_counts,
    }


def vnext_g01a_expected_test_summaries(target: str) -> list[tuple[str, str, str, str]]:
    expected_count = len(VNEXT_G01A_REQUIRED_TESTS_BY_TARGET[target])
    if target == VNEXT_G01A_RESOURCE_PANIC_ISOLATION_TARGET:
        return [
            ("1", "0", "0", str(expected_count - 1)),
            (str(expected_count), "0", "0", "0"),
        ]
    return [(str(expected_count), "0", "0", "0")]


def validate_g0_unit_bench_witnesses(
    stdout_text: str, stderr_text: str
) -> dict[str, Any]:
    require_gate(
        "Running benches/engine_bench.rs" in stderr_text,
        "g0 source unit engine_bench execution witness missing",
    )
    for bench_case in G0_UNIT_BENCH_CASES:
        require_gate(
            f"Testing {bench_case}\nSuccess" in stdout_text,
            f"g0 source unit engine_bench case witness missing: {bench_case}",
        )
    return {
        "executed_bench_targets": ["engine_bench"],
        "executed_bench_case_count": len(G0_UNIT_BENCH_CASES),
    }


def validate_g0_source_unit_provenance(
    lane_command: LaneCommand,
    child_manifest: dict[str, Any],
    child_manifest_sha256: str,
) -> dict[str, Any]:
    manifest_path = lane_command.child_manifest_path
    require_gate(manifest_path is not None, "g0 source unit manifest path is missing")
    root = manifest_path.parent.resolve()
    require_gate(
        set(child_manifest)
        == {
            "schema_version",
            "artifact_type",
            "status",
            "lane",
            "pass_line",
            "command",
            "env_overrides",
            "receipt_schema",
            "limits",
            "peaks",
            "cleanup",
            "bounded_receipt",
            "stdout_log",
            "stderr_log",
        },
        "g0 source unit manifest field set mismatch",
    )
    require_gate(
        child_manifest.get("schema_version") == 1
        and child_manifest.get("artifact_type") == "g0_source_unit_bounded_gate"
        and child_manifest.get("lane") == "unit",
        "g0 source unit manifest schema/lane mismatch",
    )
    manifest_limits = require_object(child_manifest.get("limits"), "g0 source unit limits")
    require_gate(
        child_manifest.get("command") == G0_UNIT_BOUNDED_COMMAND
        and child_manifest.get("env_overrides") == G0_UNIT_BOUNDED_ENV_OVERRIDES
        and child_manifest.get("receipt_schema") == VNEXT_G01A_BOUNDED_RECEIPT_SCHEMA
        and set(manifest_limits) == set(G0_UNIT_BOUNDED_LIMITS)
        and all(
            isinstance(manifest_limits.get(key), (int, float))
            and not isinstance(manifest_limits.get(key), bool)
            and manifest_limits[key] == value
            for key, value in G0_UNIT_BOUNDED_LIMITS.items()
        ),
        "g0 source unit bounded command metadata mismatch",
    )

    def artifact_ref(name: str) -> tuple[Path, dict[str, Any]]:
        ref = require_object(child_manifest.get(name), f"g0 source unit {name}")
        require_gate(
            set(ref) == {"path", "sha256", "size_bytes"},
            f"g0 source unit {name} field set mismatch",
        )
        path, _ = child_artifact_path(root, ref.get("path"), f"g0 source unit {name}.path")
        digest = require_sha256(ref.get("sha256"), f"g0 source unit {name}.sha256")
        size = ref.get("size_bytes")
        require_gate(
            isinstance(size, int)
            and not isinstance(size, bool)
            and size >= 0
            and path.stat().st_size == size
            and sha256(path) == digest,
            f"g0 source unit {name} identity mismatch",
        )
        return path, ref

    receipt_path, receipt_ref = artifact_ref("bounded_receipt")
    stdout_path, stdout_ref = artifact_ref("stdout_log")
    stderr_path, stderr_ref = artifact_ref("stderr_log")
    receipt = read_json_object(receipt_path, "g0 source unit bounded receipt")
    stdout_bytes = stdout_path.read_bytes()
    stderr_bytes = stderr_path.read_bytes()
    bench_summary = validate_g0_unit_bench_witnesses(
        stdout_bytes.decode("utf-8"), stderr_bytes.decode("utf-8")
    )
    receipt = validate_bounded_command_receipt(
        receipt,
        expected_command=G0_UNIT_BOUNDED_COMMAND,
        expected_cwd=REPO_ROOT,
        expected_limits=G0_UNIT_BOUNDED_LIMITS,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        label="g0 source unit",
    )
    require_gate(
        Path(require_string(receipt["stdout"].get("path"), "g0 source unit receipt stdout path")).resolve()
        == stdout_path.resolve()
        and Path(require_string(receipt["stderr"].get("path"), "g0 source unit receipt stderr path")).resolve()
        == stderr_path.resolve(),
        "g0 source unit receipt log path mismatch",
    )
    manifest_peaks = require_object(child_manifest.get("peaks"), "g0 source unit peaks")
    manifest_cleanup = require_object(child_manifest.get("cleanup"), "g0 source unit cleanup")
    require_gate(
        manifest_peaks == receipt["peaks"]
        and all(
            isinstance(manifest_peaks.get(key), int)
            and not isinstance(manifest_peaks.get(key), bool)
            for key in (
                "processes",
                "group_threads",
                "per_process_threads",
                "per_process_threads_pid",
            )
        )
        and set(manifest_cleanup) == {"process_group_gone"}
        and manifest_cleanup.get("process_group_gone") is True,
        "g0 source unit bounded summary mismatch",
    )
    return {
        "kind": "g0-source-unit",
        "child_manifest": {"path": str(manifest_path), "sha256": child_manifest_sha256},
        "command": copy.deepcopy(G0_UNIT_BOUNDED_COMMAND),
        "env_overrides": copy.deepcopy(G0_UNIT_BOUNDED_ENV_OVERRIDES),
        **bench_summary,
        "receipt": copy.deepcopy(receipt_ref),
        "stdout": copy.deepcopy(stdout_ref),
        "stderr": copy.deepcopy(stderr_ref),
        "limits": copy.deepcopy(G0_UNIT_BOUNDED_LIMITS),
        "peaks": copy.deepcopy(receipt["peaks"]),
        "cleanup": {"process_group_gone": True},
    }


def validate_vnext_g01a_semantic_summary(raw: Any) -> dict[str, Any]:
    summary = require_object(raw, "vnext-g01a semantic contracts")
    require_gate(
        set(summary)
        == {
            "schema_version",
            "execution_identity",
            "event_transition",
            "required_type_kinds",
            "definition_counts",
            "dnf_retry_authority",
            "multi_participant_dispatch",
            "public_raw_dynamic_resource_shape",
        }
        and summary.get("schema_version") == 1,
        "vnext-g01a semantic contract schema mismatch",
    )

    identity = require_object(
        summary.get("execution_identity"), "vnext-g01a execution identity contract"
    )
    require_gate(
        set(identity) == {"constant", "major", "minor", "definition_count"}
        and identity.get("constant") == "EXECUTION_IDENTITY_VERSION"
        and (identity.get("major"), identity.get("minor"))
        == VNEXT_G01A_EXECUTION_IDENTITY_VERSION
        and identity.get("definition_count") == 1,
        "vnext-g01a execution identity version must be exactly 3.0",
    )

    transition = require_object(
        summary.get("event_transition"), "vnext-g01a event transition contract"
    )
    require_gate(
        set(transition)
        == {
            "enum",
            "required_variant",
            "required_variant_count",
            "forbidden_identifier",
            "forbidden_identifier_count",
        }
        and transition.get("enum") == "ExecutionEventKind"
        and transition.get("required_variant")
        == VNEXT_G01A_EVENT_REQUIRED_TRANSITION
        and transition.get("required_variant_count") == 1,
        f"vnext-g01a must define exactly one {VNEXT_G01A_EVENT_REQUIRED_TRANSITION}",
    )
    require_gate(
        transition.get("forbidden_identifier")
        == VNEXT_G01A_EVENT_FORBIDDEN_TRANSITION
        and transition.get("forbidden_identifier_count") == 0,
        f"vnext-g01a event contract must contain zero {VNEXT_G01A_EVENT_FORBIDDEN_TRANSITION} identifiers",
    )

    type_kinds = require_object(
        summary.get("required_type_kinds"), "vnext-g01a semantic type kinds"
    )
    require_gate(
        type_kinds == VNEXT_G01A_SEMANTIC_TYPE_KINDS,
        "vnext-g01a semantic type/kind matrix mismatch",
    )
    definition_counts = require_object(
        summary.get("definition_counts"), "vnext-g01a semantic definition counts"
    )
    require_gate(
        set(definition_counts) == set(VNEXT_G01A_SEMANTIC_TYPE_KINDS)
        and all(
            isinstance(count, int)
            and not isinstance(count, bool)
            and count == 1
            for count in definition_counts.values()
        ),
        "vnext-g01a semantic type definition count matrix mismatch",
    )

    retry = require_object(
        summary.get("dnf_retry_authority"), "vnext-g01a DNF retry authority"
    )
    require_gate(
        set(retry)
        == {"type_name", "field_sealed", "public_associated_constructor_count"}
        and retry.get("type_name") == VNEXT_G01A_DNF_RETRY_AUTHORITY_TYPE
        and retry.get("field_sealed") is True
        and retry.get("public_associated_constructor_count") == 0,
        "vnext-g01a DefinitelyNotSubmitted retry authority is not sealed",
    )

    dispatch = require_object(
        summary.get("multi_participant_dispatch"),
        "vnext-g01a multi-participant dispatch",
    )
    require_gate(
        set(dispatch)
        == {
            "owner",
            "required_markers",
            "observed_markers",
            "matching_public_method_count",
        }
        and dispatch.get("owner") == "OperationDispatch"
        and dispatch.get("required_markers")
        == VNEXT_G01A_MULTIPARTICIPANT_DISPATCH_MARKERS,
        "vnext-g01a multi-participant dispatch marker matrix mismatch",
    )
    observed_markers = require_object(
        dispatch.get("observed_markers"),
        "vnext-g01a observed multi-participant dispatch markers",
    )
    require_gate(
        set(observed_markers) == set(VNEXT_G01A_MULTIPARTICIPANT_DISPATCH_MARKERS)
        and all(observed is True for observed in observed_markers.values()),
        "vnext-g01a multi-participant dispatch marker is missing",
    )
    require_gate(
        dispatch.get("matching_public_method_count") == 1,
        "vnext-g01a must have exactly one public multi-participant dispatch method",
    )

    raw_shape = require_object(
        summary.get("public_raw_dynamic_resource_shape"),
        "vnext-g01a public raw DynamicResourceShape contract",
    )
    require_gate(
        set(raw_shape)
        == {
            "type_name",
            "unrestricted_public_type_count",
            "unrestricted_public_impl_method_count",
            "unrestricted_public_parameter_path_count",
        }
        and raw_shape.get("type_name") == "DynamicResourceShape",
        "vnext-g01a public raw DynamicResourceShape schema mismatch",
    )
    for count_name in (
        "unrestricted_public_type_count",
        "unrestricted_public_impl_method_count",
        "unrestricted_public_parameter_path_count",
    ):
        count = raw_shape.get(count_name)
        require_gate(
            isinstance(count, int) and not isinstance(count, bool) and count == 0,
            f"vnext-g01a public raw DynamicResourceShape {count_name} must be zero",
        )
    return copy.deepcopy(summary)


def validate_vnext_g01a_provenance(
    lane_command: LaneCommand,
    child_manifest: dict[str, Any],
    child_manifest_sha256: str,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    manifest_path = lane_command.child_manifest_path
    require_gate(manifest_path is not None, "vnext-g01a delegated manifest path is missing")
    checkpoint_root = manifest_path.parent.resolve()
    output_root = checkpoint_root.parent.resolve()
    require_gate(
        manifest_path.resolve() == output_root / "g01a-contract-checkpoint/manifest.json",
        "vnext-g01a child manifest path mismatch",
    )
    require_gate(child_manifest.get("schema_version") == 1, "vnext-g01a schema_version mismatch")
    require_gate(
        child_manifest.get("artifact_type")
        == "runtime_vnext_g01a_contract_checkpoint_manifest",
        "vnext-g01a artifact_type mismatch",
    )
    require_gate(
        child_manifest.get("checkpoint_id") == "G01A"
        and child_manifest.get("lane") == "runtime-vnext-g01a",
        "vnext-g01a checkpoint/lane mismatch",
    )
    require_gate(
        child_manifest.get("canonical") is True and child_manifest.get("status") == "pass",
        "vnext-g01a canonical/status mismatch",
    )
    require_gate(
        Path(require_string(child_manifest.get("artifact_dir"), "vnext-g01a artifact_dir")).resolve()
        == checkpoint_root,
        "vnext-g01a artifact_dir mismatch",
    )
    require_gate(
        Path(require_string(child_manifest.get("output_root"), "vnext-g01a output_root")).resolve()
        == output_root,
        "vnext-g01a output_root mismatch",
    )
    expected_pass = f"FERRUM RUNTIME VNEXT G01A CONTRACT CHECKPOINT PASS: {output_root}"
    require_gate(child_manifest.get("pass_line") == expected_pass, "vnext-g01a pass_line mismatch")
    require_exact_string_set(child_manifest.get("unlocks"), {"G01B"}, "vnext-g01a unlocks")
    require_exact_string_set(
        child_manifest.get("does_not_prove"),
        VNEXT_G01A_DOES_NOT_PROVE,
        "vnext-g01a does_not_prove",
    )

    freshness = require_object(child_manifest.get("freshness"), "vnext-g01a freshness")
    expected_freshness = {
        "checkout_clean",
        "g00a_outer_matches_current_head",
        "g00a_child_matches_current_head_and_tree",
        "g00a_artifact_index_verified",
        "contract_files_match_current_head_blobs",
        "compile_tests_ran_on_current_head",
    }
    require_gate(set(freshness) == expected_freshness, "vnext-g01a freshness field set mismatch")
    require_gate(all(value is True for value in freshness.values()), "vnext-g01a freshness contains a non-true value")

    artifact_index = validate_child_artifact_index(
        checkpoint_root, child_manifest, role_from_top_level_path=False
    )
    require_gate(
        set(artifact_index) == {"adr.md", "contract-map.json", "compile-unit-trybuild.json"},
        "vnext-g01a artifact set mismatch",
    )
    require_gate(child_manifest.get("artifact_count") == 3, "vnext-g01a artifact_count mismatch")

    source = require_object(child_manifest.get("source"), "vnext-g01a source")
    source_sha = require_git_sha(source.get("git_sha"), "vnext-g01a source.git_sha")
    source_tree = require_git_sha(source.get("git_tree_sha"), "vnext-g01a source.git_tree_sha")
    require_gate(source.get("dirty") is False and source.get("status_short") == [], "vnext-g01a source dirty state mismatch")
    if verify_checkout:
        require_gate(git_sha() == source_sha, "vnext-g01a source SHA is stale against current HEAD")
        require_gate(
            git_output(["rev-parse", "HEAD^{tree}"]) == source_tree,
            "vnext-g01a source tree is stale against current HEAD",
        )
        require_gate(not git_dirty_status()["is_dirty"], "vnext-g01a run_gate checkout must remain clean")

    g00a = require_object(child_manifest.get("g00a"), "vnext-g01a G00a binding")
    require_gate(
        set(g00a)
        == {
            "outer_manifest",
            "child_manifest",
            "artifact_index_sha256",
            "source",
            "coupling_inventory",
        },
        "vnext-g01a G00a binding field set mismatch",
    )
    g00a_source = require_object(g00a.get("source"), "vnext-g01a G00a source")
    require_gate(
        g00a_source == {"git_sha": source_sha, "git_tree_sha": source_tree},
        "vnext-g01a G00a/current source binding mismatch",
    )
    for key, expected_name in (("outer_manifest", "gate.manifest.json"), ("child_manifest", "manifest.json")):
        ref = require_object(g00a.get(key), f"vnext-g01a G00a {key}")
        path = Path(require_string(ref.get("path"), f"vnext-g01a G00a {key}.path")).resolve()
        require_gate(path.name == expected_name and path.is_file() and not path.is_symlink(), f"vnext-g01a G00a {key} path mismatch")
        require_gate(
            sha256(path) == require_sha256(ref.get("sha256"), f"vnext-g01a G00a {key}.sha256"),
            f"vnext-g01a G00a {key} SHA mismatch",
        )
    outer_path = Path(g00a["outer_manifest"]["path"]).resolve()
    child_path = Path(g00a["child_manifest"]["path"]).resolve()
    require_gate(outer_path.parent == child_path.parent, "vnext-g01a G00a outer/child roots differ")
    g00a_outer = read_json_object(outer_path, "vnext-g01a bound G00a outer manifest")
    g00a_child = read_json_object(child_path, "vnext-g01a bound G00a child manifest")
    require_gate(
        g00a_outer.get("lane") == "vnext-g00a"
        and g00a_outer.get("status") == "pass"
        and g00a_outer.get("git_sha") == source_sha,
        "vnext-g01a bound G00a outer is stale or invalid",
    )
    g00a_collector = require_object(g00a_child.get("collector"), "vnext-g01a bound G00a collector")
    require_gate(
        g00a_child.get("status") == "pass"
        and g00a_child.get("checkpoint_id") == "G00a"
        and g00a_collector.get("git_sha") == source_sha
        and g00a_collector.get("git_tree_sha") == source_tree,
        "vnext-g01a bound G00a child is stale or invalid",
    )
    bound_outer_child = require_object(
        g00a_outer.get("child_artifacts"),
        "vnext-g01a bound G00a outer child_artifacts",
    )
    require_gate(
        bound_outer_child.get("kind") == "vnext-g00a"
        and require_object(
            bound_outer_child.get("child_manifest"),
            "vnext-g01a bound G00a outer child_manifest",
        )
        == g00a["child_manifest"],
        "vnext-g01a bound G00a outer/child manifest reference mismatch",
    )
    require_gate(
        bound_outer_child.get("artifact_index_sha256")
        == g00a.get("artifact_index_sha256"),
        "vnext-g01a bound G00a outer artifact-index reference mismatch",
    )
    validate_vnext_g00a_provenance(
        LaneCommand(
            cmd=[],
            expected_child_pass_line=g00a_child.get("pass_line"),
            child_manifest_path=child_path,
            provenance_kind="vnext-g00a",
        ),
        g00a_child,
        require_sha256(
            g00a["child_manifest"].get("sha256"),
            "vnext-g01a G00a child manifest SHA256",
        ),
        verify_checkout=verify_checkout,
    )
    require_gate(
        canonical_json_sha256(g00a_child.get("artifact_index"))
        == require_sha256(g00a.get("artifact_index_sha256"), "vnext-g01a G00a artifact_index_sha256"),
        "vnext-g01a bound G00a artifact index digest mismatch",
    )
    inventory_ref = require_object(g00a.get("coupling_inventory"), "vnext-g01a G00a coupling inventory")
    inventory_path = Path(require_string(inventory_ref.get("path"), "vnext-g01a G00a coupling inventory path")).resolve()
    require_gate(inventory_path == outer_path.parent / "coupling-inventory.json", "vnext-g01a coupling inventory path mismatch")
    require_gate(
        inventory_ref.get("backend_trait_method_count") == 82
        and sha256(inventory_path) == require_sha256(inventory_ref.get("sha256"), "vnext-g01a coupling inventory SHA256"),
        "vnext-g01a coupling inventory identity/count mismatch",
    )

    contract_rows = require_list(child_manifest.get("contract_files"), "vnext-g01a contract_files")
    contracts: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(contract_rows):
        row = require_object(raw, f"vnext-g01a contract_files[{index}]")
        require_gate(set(row) == {"path", "git_blob", "sha256", "size_bytes"}, f"vnext-g01a contract_files[{index}] field set mismatch")
        relative = require_string(row.get("path"), f"vnext-g01a contract_files[{index}].path")
        path = REPO_ROOT / relative
        require_gate(relative not in contracts, f"duplicate vnext-g01a contract path: {relative}")
        require_gate(path.is_file() and not path.is_symlink(), f"missing vnext-g01a contract path: {relative}")
        size = row.get("size_bytes")
        require_gate(isinstance(size, int) and not isinstance(size, bool) and size > 0 and path.stat().st_size == size, f"vnext-g01a contract size mismatch: {relative}")
        require_gate(sha256(path) == require_sha256(row.get("sha256"), f"vnext-g01a contract {relative}.sha256"), f"vnext-g01a contract SHA mismatch: {relative}")
        blob = require_string(row.get("git_blob"), f"vnext-g01a contract {relative}.git_blob")
        require_gate(re.fullmatch(r"[0-9a-f]{40,64}", blob) is not None, f"vnext-g01a contract Git blob invalid: {relative}")
        if verify_checkout:
            require_gate(git_output(["rev-parse", f"HEAD:{relative}"]) == blob, f"vnext-g01a contract Git blob mismatch: {relative}")
        contracts[relative] = row
    require_gate(
        set(contracts) == discover_vnext_g01a_contract_paths(),
        "vnext-g01a contract path discovery/binding mismatch",
    )
    require_gate(
        canonical_json_sha256(contract_rows)
        == require_sha256(child_manifest.get("contract_files_sha256"), "vnext-g01a contract_files_sha256"),
        "vnext-g01a contract file list digest mismatch",
    )
    copy_bindings = {
        "adr.md": "docs/goals/runtime-vnext-0.8.0-2026-07-10/G01A_CONTRACT_ADR.md",
        "contract-map.json": "docs/goals/runtime-vnext-0.8.0-2026-07-10/G01A_LEGACY_CONTRACT_MAP.json",
    }
    for artifact, relative in copy_bindings.items():
        require_gate(artifact_index[artifact]["sha256"] == contracts[relative]["sha256"], f"vnext-g01a copied artifact differs from Git-bound source: {artifact}")

    contracts_summary = require_object(child_manifest.get("contracts"), "vnext-g01a contracts summary")
    require_gate(
        set(require_list(contracts_summary.get("required_contracts"), "vnext-g01a required contracts"))
        == VNEXT_G01A_REQUIRED_CONTRACTS
        and contracts_summary.get("required_contract_count") == 9,
        "vnext-g01a required contract matrix mismatch",
    )
    require_gate(contracts_summary.get("architecture_named_symbol_count") == 0, "vnext-g01a architecture-named contract count is nonzero")
    require_gate(contracts_summary.get("silent_success_default_count") == 0, "vnext-g01a silent success default count is nonzero")
    definition_counts = require_object(
        contracts_summary.get("definition_counts"),
        "vnext-g01a contract definition counts",
    )
    require_gate(
        set(definition_counts) == VNEXT_G01A_REQUIRED_CONTRACTS
        and all(value == 1 for value in definition_counts.values()),
        "vnext-g01a contract definition count matrix mismatch",
    )
    forbidden = require_object(contracts_summary.get("forbidden_pattern_counts"), "vnext-g01a forbidden pattern counts")
    require_gate(forbidden and all(value == 0 for value in forbidden.values()), "vnext-g01a forbidden pattern count is nonzero")
    semantic_contracts = validate_vnext_g01a_semantic_summary(
        contracts_summary.get("semantic_contracts")
    )
    isolation = require_object(child_manifest.get("isolation"), "vnext-g01a isolation")
    require_gate(isolation.get("allowed_module_declaration_count") == 1 and isolation.get("outside_vnext_reference_count") == 0, "vnext-g01a isolation audit mismatch")
    mapping = require_object(child_manifest.get("legacy_mapping"), "vnext-g01a legacy mapping")
    require_gate(
        mapping.get("mapped") == 82
        and mapping.get("unmapped") == 0
        and mapping.get("missing_owner") == 0
        and mapping.get("special_case") == 0,
        "vnext-g01a legacy mapping summary mismatch",
    )
    map_document = read_json_object(
        checkpoint_root / "contract-map.json",
        "vnext-g01a copied legacy contract map",
    )
    require_gate(
        set(map_document)
        == {"schema_version", "artifact_type", "source", "mappings", "summary"}
        and map_document.get("schema_version") == 1
        and map_document.get("artifact_type")
        == "runtime_vnext_g01a_legacy_contract_map",
        "vnext-g01a copied legacy contract map schema mismatch",
    )
    require_gate(
        map_document.get("source")
        == {
            "g00a_checkpoint_id": "G00a",
            "category": "backend_trait_method",
            "expected_method_count": 82,
        }
        and map_document.get("summary")
        == {"mapped": 82, "unmapped": 0, "missing_owner": 0, "special_case": 0},
        "vnext-g01a copied legacy contract map source/summary mismatch",
    )
    map_rows = require_list(map_document.get("mappings"), "vnext-g01a legacy mappings")
    require_gate(len(map_rows) == 82, "vnext-g01a copied legacy map must contain 82 rows")
    map_keys: set[tuple[str, str]] = set()
    allowed_classifications = {
        "stable_device_primitive",
        "versioned_operation",
        "model_semantic",
        "dead_code",
    }
    for index, raw in enumerate(map_rows):
        row = require_object(raw, f"vnext-g01a legacy mappings[{index}]")
        require_gate(
            set(row)
            == {"legacy_trait", "legacy_method", "classification", "owner", "disposition"},
            f"vnext-g01a legacy mapping field set mismatch: {index}",
        )
        key = (
            require_string(row.get("legacy_trait"), f"vnext-g01a legacy mappings[{index}].legacy_trait"),
            require_string(row.get("legacy_method"), f"vnext-g01a legacy mappings[{index}].legacy_method"),
        )
        require_gate(key not in map_keys, f"duplicate vnext-g01a legacy mapping: {key}")
        map_keys.add(key)
        require_gate(
            row.get("classification") in allowed_classifications
            and bool(require_string(row.get("owner"), f"vnext-g01a legacy mappings[{index}].owner"))
            and "special" not in require_string(
                row.get("disposition"),
                f"vnext-g01a legacy mappings[{index}].disposition",
            ).lower(),
            f"vnext-g01a legacy mapping classification/owner/disposition invalid: {key}",
        )
    inventory_document = read_json_object(
        inventory_path,
        "vnext-g01a bound G00a coupling inventory",
    )
    inventory_findings = require_list(
        require_object(
            inventory_document.get("coupling"),
            "vnext-g01a bound G00a inventory coupling",
        ).get("findings"),
        "vnext-g01a bound G00a coupling findings",
    )
    inventory_method_keys = {
        (
            require_string(row.get("trait"), "vnext-g01a inventory backend trait"),
            require_string(row.get("symbol"), "vnext-g01a inventory backend method"),
        )
        for raw in inventory_findings
        if (row := require_object(raw, "vnext-g01a inventory finding")).get("category")
        == "backend_trait_method"
    }
    require_gate(
        len(inventory_method_keys) == 82 and map_keys == inventory_method_keys,
        "vnext-g01a copied legacy map differs from bound G00a method inventory",
    )

    evidence_ref = require_object(child_manifest.get("compile_evidence"), "vnext-g01a compile evidence ref")
    require_gate(evidence_ref == artifact_index["compile-unit-trybuild.json"], "vnext-g01a compile evidence/index mismatch")
    evidence = read_json_object(checkpoint_root / "compile-unit-trybuild.json", "vnext-g01a compile evidence")
    require_gate(
        set(evidence)
        == {
            "schema_version",
            "artifact_type",
            "source",
            "command_count",
            "commands_passed",
            "commands",
            "bounded_execution",
            "tests",
            "assertions",
        },
        "vnext-g01a compile evidence field set mismatch",
    )
    require_gate(evidence.get("schema_version") == 1 and evidence.get("artifact_type") == "runtime_vnext_g01a_compile_unit_trybuild_evidence", "vnext-g01a compile evidence schema mismatch")
    require_gate(evidence.get("source") == source, "vnext-g01a compile evidence source mismatch")
    commands = require_list(evidence.get("commands"), "vnext-g01a compile commands")
    require_gate(evidence.get("command_count") == len(commands) and evidence.get("commands_passed") == len(commands), "vnext-g01a compile command count mismatch")

    def test_command(target: str, mode: str) -> tuple[str, ...]:
        return (
            "cargo",
            "test",
            "-p",
            "ferrum-interfaces",
            "--test",
            target,
            "--",
            mode,
            VNEXT_G01A_TEST_THREADS_ARG,
        )

    def admission_test_command(mode: str) -> tuple[str, ...]:
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
            VNEXT_G01A_TEST_THREADS_ARG,
        )

    expected_command_sequence = list(VNEXT_G01A_QUALITY_COMMANDS)
    expected_command_sequence.extend(
        test_command(target, "--list")
        for target in VNEXT_G01A_REQUIRED_TESTS_BY_TARGET
    )
    expected_command_sequence.extend(
        test_command(target, "--nocapture")
        for target in VNEXT_G01A_REQUIRED_TESTS_BY_TARGET
    )
    expected_command_sequence.append(admission_test_command("--list"))
    expected_command_sequence.append(admission_test_command("--nocapture"))
    require_gate(
        len(commands) == len(expected_command_sequence)
        == len(VNEXT_G01A_QUALITY_COMMANDS) + VNEXT_G01A_BOUNDED_TEST_COMMAND_COUNT,
        "vnext-g01a compile evidence command matrix size mismatch",
    )
    command_outputs: dict[tuple[str, ...], str] = {}
    actual_command_sequence: list[tuple[str, ...]] = []
    bounded_rows: list[tuple[str, dict[str, Any]]] = []
    for index, raw in enumerate(commands):
        row = require_object(raw, f"vnext-g01a compile commands[{index}]")
        require_gate(
            set(row)
            == {
                "command",
                "cwd",
                "env_overrides",
                "started_at",
                "finished_at",
                "duration_sec",
                "returncode",
                "stdout",
                "stderr",
                "stdout_sha256",
                "stderr_sha256",
                "execution",
            },
            f"vnext-g01a compile command field set mismatch: {index}",
        )
        require_gate(row.get("returncode") == 0, f"vnext-g01a compile command failed: {index}")
        require_gate(
            Path(require_string(row.get("cwd"), f"vnext-g01a compile commands[{index}].cwd")).resolve()
            == REPO_ROOT.resolve(),
            f"vnext-g01a compile command execution context mismatch: {index}",
        )
        require_gate(
            isinstance(row.get("duration_sec"), (int, float))
            and not isinstance(row.get("duration_sec"), bool)
            and row["duration_sec"] >= 0
            and bool(require_string(row.get("started_at"), f"vnext-g01a compile commands[{index}].started_at"))
            and bool(require_string(row.get("finished_at"), f"vnext-g01a compile commands[{index}].finished_at")),
            f"vnext-g01a compile command timing evidence invalid: {index}",
        )
        stdout = row.get("stdout")
        stderr = row.get("stderr")
        require_gate(isinstance(stdout, str) and isinstance(stderr, str), f"vnext-g01a compile command output type mismatch: {index}")
        require_gate(hashlib.sha256(stdout.encode()).hexdigest() == row.get("stdout_sha256"), f"vnext-g01a compile stdout digest mismatch: {index}")
        require_gate(hashlib.sha256(stderr.encode()).hexdigest() == row.get("stderr_sha256"), f"vnext-g01a compile stderr digest mismatch: {index}")
        command = tuple(require_list(row.get("command"), f"vnext-g01a compile commands[{index}].command"))
        require_gate(all(isinstance(part, str) and part for part in command), f"vnext-g01a compile command argv invalid: {index}")
        require_gate(command not in command_outputs, f"duplicate vnext-g01a compile command: {command}")
        receipt = validate_vnext_g01a_command_execution(
            row,
            command,
            stdout,
            stderr,
            index,
        )
        if receipt is not None:
            bounded_rows.append((vnext_g01a_bounded_profile(command), receipt))
        command_outputs[command] = stdout
        actual_command_sequence.append(command)
    require_gate(
        actual_command_sequence == expected_command_sequence,
        "vnext-g01a compile command sequence mismatch",
    )
    bounded_summary = summarize_vnext_g01a_bounded_execution(bounded_rows)
    recorded_bounded_summary = require_object(
        evidence.get("bounded_execution"), "vnext-g01a bounded execution summary"
    )
    require_gate(
        set(recorded_bounded_summary) == set(bounded_summary)
        and recorded_bounded_summary.get("all_process_groups_gone") is True
        and isinstance(recorded_bounded_summary.get("runner"), str)
        and isinstance(recorded_bounded_summary.get("receipt_schema"), str)
        and isinstance(recorded_bounded_summary.get("profile_counts"), dict)
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in recorded_bounded_summary["profile_counts"].values()
        )
        and all(
            isinstance(recorded_bounded_summary.get(key), int)
            and not isinstance(recorded_bounded_summary.get(key), bool)
            for key in (
                "required_command_count",
                "passed_command_count",
                "peak_processes",
                "peak_group_threads",
                "peak_per_process_threads",
            )
        ),
        "vnext-g01a bounded execution summary types mismatch",
    )
    require_gate(
        recorded_bounded_summary == bounded_summary,
        "vnext-g01a bounded execution summary mismatch",
    )
    for target in VNEXT_G01A_REQUIRED_TESTS_BY_TARGET:
        command = test_command(target, "--nocapture")
        summaries = re.findall(
            r"test result: ok\. (\d+) passed; 0 failed; (\d+) ignored; "
            r"(\d+) measured; (\d+) filtered out;",
            command_outputs[command],
        )
        if target == VNEXT_G01A_RESOURCE_PANIC_ISOLATION_TARGET:
            require_gate(
                command_outputs[command].count(
                    "test resource_transaction_abandon_panic_child ... ok"
                )
                == 2,
                "vnext-g01a resource test must run the panic-isolation case once in the parent and once in its child",
            )
        require_gate(
            summaries == vnext_g01a_expected_test_summaries(target),
            f"vnext-g01a test output exact summary mismatch: {command}",
        )
    admission_run_command = admission_test_command("--nocapture")
    admission_summaries = re.findall(
        r"test result: ok\. (\d+) passed; 0 failed; (\d+) ignored; "
        r"(\d+) measured; (\d+) filtered out;",
        command_outputs[admission_run_command],
    )
    require_gate(
        len(admission_summaries) == 1
        and admission_summaries[0][:3]
        == (
            str(VNEXT_G01A_EXPECTED_DYNAMIC_ADMISSION_CASES),
            "0",
            "0",
        ),
        f"vnext-g01a admission lib test output exact summary mismatch: {admission_run_command}",
    )
    tests = require_object(evidence.get("tests"), "vnext-g01a tests")
    require_gate(
        set(tests)
        == {
            "test_targets",
            "required_test_targets",
            "admission_lib_tests",
            "required_admission_lib_tests",
            "trybuild_pass_cases",
            "trybuild_fail_cases",
            "trybuild_pass_case_count",
            "trybuild_fail_case_count",
        },
        "vnext-g01a tests field set mismatch",
    )
    actual_test_targets = require_object(
        tests.get("test_targets"), "vnext-g01a exact test targets"
    )
    recorded_required_targets = require_object(
        tests.get("required_test_targets"), "vnext-g01a required test targets"
    )
    require_gate(
        set(actual_test_targets) == set(VNEXT_G01A_REQUIRED_TESTS_BY_TARGET)
        and set(recorded_required_targets) == set(VNEXT_G01A_REQUIRED_TESTS_BY_TARGET),
        "vnext-g01a exact test target name set mismatch",
    )
    for target, expected_tests in VNEXT_G01A_REQUIRED_TESTS_BY_TARGET.items():
        actual_tests = require_list(
            actual_test_targets.get(target), f"vnext-g01a {target} exact tests"
        )
        recorded_required = require_list(
            recorded_required_targets.get(target),
            f"vnext-g01a {target} recorded required tests",
        )
        require_gate(
            all(isinstance(name, str) and name for name in actual_tests)
            and len(actual_tests) == len(set(actual_tests))
            and set(actual_tests) == expected_tests
            and actual_tests == sorted(expected_tests)
            and recorded_required == sorted(expected_tests),
            f"vnext-g01a {target} exact test set mismatch",
        )
        listed_lines = command_outputs[test_command(target, "--list")].splitlines()
        listed_from_command = {
            match.group(1)
            for line in listed_lines
            if (match := re.fullmatch(r"([^:][^:]*): test", line.strip()))
        }
        require_gate(
            listed_from_command == expected_tests,
            f"vnext-g01a {target} command output test set mismatch",
        )
    admission_tests = require_list(
        tests.get("admission_lib_tests"), "vnext-g01a exact admission lib tests"
    )
    required_admission_tests = require_list(
        tests.get("required_admission_lib_tests"),
        "vnext-g01a required admission lib tests",
    )
    require_gate(
        all(isinstance(name, str) and name for name in admission_tests)
        and len(admission_tests) == len(set(admission_tests))
        and set(admission_tests) == VNEXT_G01A_REQUIRED_ADMISSION_LIB_TESTS
        and admission_tests == sorted(VNEXT_G01A_REQUIRED_ADMISSION_LIB_TESTS)
        and required_admission_tests
        == sorted(VNEXT_G01A_REQUIRED_ADMISSION_LIB_TESTS),
        "vnext-g01a admission lib exact test set mismatch",
    )
    admission_list_stdout = command_outputs[admission_test_command("--list")]
    admission_listed_from_command = {
        match.group(1)
        for line in admission_list_stdout.splitlines()
        if (
            match := re.fullmatch(
                r"(vnext::admission::tests::[A-Za-z0-9_]+): test", line.strip()
            )
        )
    }
    admission_list_counts = re.findall(
        r"(?m)^(\d+) tests, (\d+) benchmarks$", admission_list_stdout
    )
    require_gate(
        admission_listed_from_command == VNEXT_G01A_REQUIRED_ADMISSION_LIB_TESTS
        and admission_list_counts
        == [(str(VNEXT_G01A_EXPECTED_DYNAMIC_ADMISSION_CASES), "0")],
        "vnext-g01a admission lib command output test set/count mismatch",
    )
    pass_cases = require_list(tests.get("trybuild_pass_cases"), "vnext-g01a trybuild pass cases")
    fail_cases = require_list(tests.get("trybuild_fail_cases"), "vnext-g01a trybuild fail cases")
    expected_pass_cases = sorted(
        relative
        for relative in contracts
        if "/tests/ui/vnext/pass/" in relative and relative.endswith(".rs")
    )
    expected_fail_cases = sorted(
        relative
        for relative in contracts
        if "/tests/ui/vnext/fail/" in relative and relative.endswith(".rs")
    )
    require_gate(
        pass_cases == expected_pass_cases
        and fail_cases == expected_fail_cases
        and tests.get("trybuild_pass_case_count")
        == len(expected_pass_cases)
        == VNEXT_G01A_EXPECTED_TRYBUILD_PASS_CASES
        and tests.get("trybuild_fail_case_count")
        == len(expected_fail_cases)
        == VNEXT_G01A_EXPECTED_TRYBUILD_FAIL_CASES,
        "vnext-g01a trybuild pass/fail coverage differs from Git-bound fixtures",
    )
    require_gate(
        all(case.removesuffix(".rs") + ".stderr" in contracts for case in expected_fail_cases),
        "vnext-g01a trybuild fail fixture lacks Git-bound stderr",
    )
    assertions = require_object(evidence.get("assertions"), "vnext-g01a assertions")
    expected_assertions = {
        "deterministic_plan_cases": 100,
        "schema_round_trip_cases": 100,
        "breaking_version_reject_cases": 100,
        "resource_transaction_cases": VNEXT_G01A_EXPECTED_RESOURCE_CASES,
        "fail_closed_cases": VNEXT_G01A_EXPECTED_FAIL_CLOSED_CASES,
        "model_identity_cases": VNEXT_G01A_EXPECTED_MODEL_IDENTITY_CASES,
        "event_replay_v5_contract_cases": VNEXT_G01A_EXPECTED_EVENT_REPLAY_V5_CASES,
        "device_operation_contract_cases": VNEXT_G01A_EXPECTED_DEVICE_OPERATION_CASES,
        "operation_oracle_contract_cases": VNEXT_G01A_EXPECTED_ORACLE_CASES,
        "model_wire_contract_cases": VNEXT_G01A_EXPECTED_MODEL_WIRE_CASES,
        "dynamic_admission_cases": VNEXT_G01A_EXPECTED_DYNAMIC_ADMISSION_CASES,
        "legacy_backend_methods_mapped": 82,
        "legacy_backend_methods_unmapped": 0,
        "architecture_named_symbol_count": 0,
        "required_contract_count": 9,
        "silent_success_default_count": 0,
        "unknown_fallback_success_count": 0,
        "outside_vnext_production_reference_count": 0,
    }
    require_gate(
        set(assertions) == set(expected_assertions),
        "vnext-g01a compile assertion field set mismatch",
    )
    require_gate(
        all(
            isinstance(assertions.get(key), int)
            and not isinstance(assertions.get(key), bool)
            and assertions[key] == value
            for key, value in expected_assertions.items()
        ),
        "vnext-g01a compile assertion summary mismatch",
    )
    contract_stdout = command_outputs[
        test_command("vnext_plan_wire_contract_tests", "--nocapture")
    ].splitlines()
    for line in (
        "VNEXT PLAN DETERMINISM PASS: 100/100",
        "VNEXT PLAN ROUNDTRIP PASS: 100/100",
        "VNEXT BREAKING VERSION REJECT PASS: 100/100",
    ):
        require_gate(contract_stdout.count(line) == 1, f"vnext-g01a missing or duplicate machine proof line: {line}")
    resource_total = 0
    for target, proofs in VNEXT_G01A_RESOURCE_PROOF_LINES.items():
        proof_stdout = command_outputs[test_command(target, "--nocapture")].splitlines()
        for prefix, expected in proofs:
            pattern = re.compile(
                rf"^{re.escape(prefix)}: ([1-9][0-9]*)/([1-9][0-9]*)$"
            )
            matches = [
                match for line in proof_stdout if (match := pattern.fullmatch(line))
            ]
            require_gate(
                len(matches) == 1,
                f"vnext-g01a missing or duplicate resource machine proof: {target} {prefix}",
            )
            passed, total = (int(value) for value in matches[0].groups())
            require_gate(
                passed == total == expected,
                f"vnext-g01a resource proof mismatch: {target} {prefix}",
            )
            resource_total += total
    require_gate(
        resource_total == assertions["resource_transaction_cases"],
        "vnext-g01a resource proof total/assertion mismatch",
    )
    event_total = 0
    for target, (prefix, expected) in VNEXT_G01A_EVENT_PROOF_LINES.items():
        proof_stdout = command_outputs[test_command(target, "--nocapture")].splitlines()
        pattern = re.compile(
            rf"^{re.escape(prefix)}: ([1-9][0-9]*)/([1-9][0-9]*)$"
        )
        matches = [
            match for line in proof_stdout if (match := pattern.fullmatch(line))
        ]
        require_gate(
            len(matches) == 1,
            f"vnext-g01a missing or duplicate event machine proof: {target} {prefix}",
        )
        passed, total = (int(value) for value in matches[0].groups())
        require_gate(
            passed == total == expected,
            f"vnext-g01a event proof mismatch: {target} {prefix}",
        )
        event_total += total
    require_gate(
        event_total == assertions["event_replay_v5_contract_cases"],
        "vnext-g01a event proof total/assertion mismatch",
    )
    device_operation_total = 0
    for target, (prefix, expected) in VNEXT_G01A_DEVICE_OPERATION_PROOF_LINES.items():
        proof_stdout = command_outputs[test_command(target, "--nocapture")].splitlines()
        pattern = re.compile(
            rf"^{re.escape(prefix)}: ([1-9][0-9]*)/([1-9][0-9]*)$"
        )
        matches = [
            match for line in proof_stdout if (match := pattern.fullmatch(line))
        ]
        require_gate(
            len(matches) == 1,
            "vnext-g01a missing or duplicate device operation machine proof: "
            f"{target} {prefix}",
        )
        passed, total = (int(value) for value in matches[0].groups())
        require_gate(
            passed == total == expected,
            f"vnext-g01a device operation proof mismatch: {target} {prefix}",
        )
        device_operation_total += total
    require_gate(
        device_operation_total == assertions["device_operation_contract_cases"],
        "vnext-g01a device operation proof total/assertion mismatch",
    )
    for assertion_key, target, prefix in (
        (
            "fail_closed_cases",
            "vnext_resolution_contract_tests",
            "VNEXT FAIL CLOSED PASS",
        ),
        (
            "model_identity_cases",
            "vnext_resolution_contract_tests",
            "VNEXT MODEL IDENTITY PASS",
        ),
        (
            "operation_oracle_contract_cases",
            "vnext_oracle_contract_tests",
            "VNEXT OPERATION ORACLE PASS",
        ),
        (
            "model_wire_contract_cases",
            "vnext_model_wire_contract_tests",
            "VNEXT MODEL WIRE PASS",
        ),
        (
            "legacy_backend_methods_mapped",
            "vnext_legacy_map",
            "VNEXT LEGACY MAP PASS",
        ),
    ):
        pattern = re.compile(
            rf"^{re.escape(prefix)}: ([1-9][0-9]*)/([1-9][0-9]*)$"
        )
        proof_stdout = command_outputs[test_command(target, "--nocapture")].splitlines()
        matches = [match for line in proof_stdout if (match := pattern.fullmatch(line))]
        require_gate(len(matches) == 1, f"vnext-g01a missing or duplicate machine proof ratio: {assertion_key}")
        passed, total = (int(value) for value in matches[0].groups())
        require_gate(
            passed == total == assertions[assertion_key],
            f"vnext-g01a machine proof ratio/assertion mismatch: {assertion_key}",
        )

    return {
        "kind": "vnext-g01a",
        "child_manifest": {
            "path": str(manifest_path),
            "sha256": require_sha256(child_manifest_sha256, "vnext-g01a child manifest SHA256"),
        },
        "checkpoint": {
            "id": "G01A",
            "unlocks": ["G01B"],
            "does_not_prove": sorted(VNEXT_G01A_DOES_NOT_PROVE),
        },
        "source": {"git_sha": source_sha, "git_tree_sha": source_tree},
        "g00a": copy.deepcopy(g00a),
        "artifact_index_sha256": canonical_json_sha256(child_manifest["artifact_index"]),
        "contract_files_sha256": child_manifest["contract_files_sha256"],
        "required_contract_count": 9,
        "semantic_contracts": semantic_contracts,
        "legacy_backend_methods_mapped": 82,
        "dynamic_admission_cases": assertions["dynamic_admission_cases"],
        "bounded_execution": copy.deepcopy(bounded_summary),
    }


def validate_vnext_g00_full_redteam(
    lane_command: LaneCommand,
    stdout: str,
) -> dict[str, Any]:
    require_gate(
        "--require-full-self-test" in lane_command.cmd,
        "vnext-g00 delegated command is missing --require-full-self-test",
    )
    lines = stdout.splitlines()
    require_gate(
        VNEXT_G00_FULL_SELFTEST_PASS in lines,
        f"vnext-g00 delegated command did not print exact FULL self-test PASS line: {VNEXT_G00_FULL_SELFTEST_PASS}",
    )
    summary_lines = [
        line.removeprefix(VNEXT_G00_SELFTEST_SUMMARY_PREFIX).strip()
        for line in lines
        if line.startswith(VNEXT_G00_SELFTEST_SUMMARY_PREFIX)
    ]
    require_gate(
        len(summary_lines) == 1,
        "vnext-g00 delegated command must print exactly one full-redteam summary",
    )
    try:
        summary_raw = json.loads(summary_lines[0])
    except json.JSONDecodeError as exc:
        raise GateError(f"vnext-g00 full-redteam summary is invalid JSON: {exc}") from exc
    summary = require_object(summary_raw, "vnext-g00 full-redteam summary")
    require_gate(summary.get("schema_version") == 1, "vnext-g00 full-redteam summary schema mismatch")
    require_gate(summary.get("mode") == "full", "vnext-g00 full-redteam summary mode must be full")
    mutation_count = summary.get("mutation_assertion_count")
    require_gate(
        mutation_count == VNEXT_G00_REDTEAM_MUTATION_COUNT,
        f"vnext-g00 full-redteam mutation count must be {VNEXT_G00_REDTEAM_MUTATION_COUNT}",
    )
    require_gate(
        summary.get("expected_mutation_assertion_count") == VNEXT_G00_REDTEAM_MUTATION_COUNT,
        "vnext-g00 full-redteam locked mutation count mismatch",
    )
    mutation_names = require_list(
        summary.get("mutation_names"),
        "vnext-g00 full-redteam mutation_names",
    )
    require_gate(
        len(mutation_names) == mutation_count
        and all(isinstance(name, str) and name for name in mutation_names),
        "vnext-g00 full-redteam mutation_names are incomplete or malformed",
    )
    require_gate(
        len(set(mutation_names)) == mutation_count,
        "vnext-g00 full-redteam mutation_names contain duplicates",
    )
    mutation_matrix_sha256 = canonical_json_sha256(mutation_names)
    require_gate(
        mutation_matrix_sha256 == VNEXT_G00_REDTEAM_MUTATION_MATRIX_SHA256,
        "vnext-g00 full-redteam mutation matrix SHA256 mismatch",
    )
    validator_counts = require_object(
        summary.get("validator_counts"),
        "vnext-g00 full-redteam validator_counts",
    )
    require_gate(
        validator_counts == {"full-root": mutation_count},
        "vnext-g00 full-redteam validator_counts.full-root must equal mutation count",
    )
    return {
        "pass_line": VNEXT_G00_FULL_SELFTEST_PASS,
        "summary": summary,
        "summary_sha256": canonical_json_sha256(summary),
        "mutation_matrix_sha256": mutation_matrix_sha256,
    }


def validate_vnext_g00_provenance(
    lane_command: LaneCommand,
    child_manifest: dict[str, Any],
    child_manifest_sha256: str,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    manifest_path = lane_command.child_manifest_path
    require_gate(manifest_path is not None, "vnext-g00 delegated manifest path is missing")
    root = manifest_path.parent.resolve()
    require_gate(manifest_path.resolve() == root / "manifest.json", "vnext-g00 delegated manifest must be <artifact_root>/manifest.json")
    manifest_digest = require_sha256(child_manifest_sha256, "vnext-g00 delegated manifest SHA256")
    artifact_dir = Path(require_string(child_manifest.get("artifact_dir"), "delegated manifest artifact_dir"))
    require_gate(artifact_dir.resolve() == root, "delegated manifest artifact_dir mismatch")
    require_gate(child_manifest.get("schema_version") == 1, "delegated manifest schema_version mismatch")
    require_gate(child_manifest.get("waiver_count") == 0, "delegated manifest waiver_count must be zero")
    validator_git_sha = require_git_sha(
        child_manifest.get("validator_git_sha"),
        "delegated manifest validator_git_sha",
    )
    require_gate(child_manifest.get("validator_dirty_status") == [], "delegated manifest validator must be clean")

    artifact_index = validate_child_artifact_index(root, child_manifest)
    models_lock_path, models_lock_rel, models_lock_digest = require_indexed_artifact(
        root,
        artifact_index,
        "models.lock.json",
        child_manifest.get("models_lock_sha256"),
        "models.lock",
    )
    models_lock = read_json_object(models_lock_path, "models.lock")
    require_gate(models_lock.get("schema_version") == 1, "models.lock schema_version mismatch")
    require_gate(models_lock.get("source_git_sha") == VNEXT_FROZEN_LEGACY_SHA, "models.lock frozen source SHA mismatch")
    expectations_binding = require_object(
        models_lock.get("expectations_catalog"),
        "models.lock.expectations_catalog",
    )
    expectations_path = require_string(
        expectations_binding.get("path"),
        "models.lock.expectations_catalog.path",
    )
    expectations_sha = require_sha256(
        expectations_binding.get("sha256"),
        "models.lock.expectations_catalog.sha256",
    )
    require_gate(
        expectations_path == VNEXT_LEGACY_EXPECTATIONS_PATH,
        "models.lock expectations catalog path mismatch",
    )
    expectations_file = REPO_ROOT / expectations_path
    require_gate(
        expectations_file.is_file() and sha256(expectations_file) == expectations_sha,
        "models.lock expectations catalog differs from the clean checkout",
    )
    require_gate(
        child_manifest.get("expectations_catalog") == expectations_binding,
        "delegated manifest expectations catalog mismatch",
    )
    contract_files = require_list(child_manifest.get("contract_files"), "delegated manifest contract_files")
    contract_by_path: dict[str, dict[str, Any]] = {}
    for contract_index, raw_contract in enumerate(contract_files):
        contract = require_object(raw_contract, f"delegated manifest contract_files[{contract_index}]")
        relative = require_string(contract.get("path"), f"delegated manifest contract_files[{contract_index}].path")
        require_gate(relative not in contract_by_path, f"duplicate delegated contract path: {relative}")
        contract_by_path[relative] = contract
    require_gate(
        require_sha256(
            require_object(
                contract_by_path.get(expectations_path),
                "delegated expectations catalog contract",
            ).get("sha256"),
            "delegated expectations catalog contract SHA256",
        )
        == expectations_sha,
        "delegated expectations catalog contract mismatch",
    )

    resolution_ref = require_object(models_lock.get("model_resolution"), "models.lock.model_resolution")
    resolution_path, resolution_rel, resolution_digest = require_indexed_artifact(
        root,
        artifact_index,
        resolution_ref.get("path"),
        resolution_ref.get("sha256"),
        "model-resolution",
    )
    require_gate(resolution_rel == "model-resolution.json", "model-resolution path must be model-resolution.json")
    resolution = read_json_object(resolution_path, "model-resolution")
    require_gate(resolution.get("schema_version") == 1, "model-resolution schema_version mismatch")
    require_gate(resolution.get("artifact_type") == "runtime_vnext_model_resolution", "model-resolution artifact_type mismatch")
    resolver_raw = require_object(resolution.get("resolver"), "model-resolution.resolver")
    resolver_identity = {
        "path": require_string(resolver_raw.get("path"), "model-resolution.resolver.path"),
        "sha256": require_sha256(resolver_raw.get("sha256"), "model-resolution.resolver.sha256"),
    }
    resolution_rows = require_list(resolution.get("lanes"), "model-resolution.lanes")
    resolution_lanes: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(resolution_rows):
        row = require_object(raw, f"model-resolution.lanes[{index}]")
        lane_id = require_string(row.get("catalog_lane_id"), f"model-resolution.lanes[{index}].catalog_lane_id")
        require_gate(lane_id not in resolution_lanes, f"duplicate model-resolution lane {lane_id}")
        resolution_lanes[lane_id] = row

    model_rows = require_list(models_lock.get("models"), "models.lock.models")
    expected_models = {**VNEXT_PRIMARY_MODELS, **VNEXT_SUPPLEMENTAL_MODELS}
    models: dict[str, dict[str, Any]] = {}
    model_identities: list[dict[str, Any]] = []
    expected_resolution_ids: set[str] = set()
    for index, raw in enumerate(model_rows):
        model = require_object(raw, f"models.lock.models[{index}]")
        key = require_string(model.get("key"), f"models.lock.models[{index}].key")
        require_gate(key in expected_models and key not in models, f"unknown or duplicate locked model {key}")
        role = require_string(model.get("role"), f"models[{key}].role")
        expected_role = "primary" if key in VNEXT_PRIMARY_MODELS else "supplemental"
        require_gate(role == expected_role, f"models[{key}].role mismatch")
        official_model_id = require_string(model.get("official_model_id"), f"models[{key}].official_model_id")
        require_gate(official_model_id == expected_models[key], f"models[{key}].official_model_id mismatch")
        lanes = require_object(model.get("lanes"), f"models[{key}].lanes")
        require_gate(set(lanes) == {"cuda", "metal"}, f"models[{key}] must contain CUDA and Metal lanes")
        normalized_lanes: dict[str, Any] = {}
        for backend in ("cuda", "metal"):
            lane = require_object(lanes[backend], f"models[{key}].lanes.{backend}")
            lane_id = require_string(lane.get("catalog_lane_id"), f"models[{key}].lanes.{backend}.catalog_lane_id")
            require_gate(lane_id in resolution_lanes, f"models[{key}].lanes.{backend} missing model-resolution lane")
            expected_resolution_ids.add(lane_id)
            resolved = resolution_lanes[lane_id]
            require_gate(resolved.get("backend") == backend, f"models[{key}].lanes.{backend} resolution backend mismatch")
            require_gate(resolved.get("model_id") == VNEXT_RESOLUTION_MODEL_IDS[key], f"models[{key}].lanes.{backend} resolution model id mismatch")
            require_gate(resolved.get("format") == lane.get("format"), f"models[{key}].lanes.{backend} resolution format mismatch")
            locked_weight = {
                "repo": require_string(lane.get("repo"), f"models[{key}].lanes.{backend}.repo"),
                "revision": require_git_sha(lane.get("revision"), f"models[{key}].lanes.{backend}.revision"),
                "files": normalized_file_locks(lane.get("files"), f"models[{key}].lanes.{backend}.files"),
            }
            require_gate(locked_weight == normalized_model_source(resolved.get("weight_source"), f"model-resolution.{lane_id}.weight_source"), f"models[{key}].lanes.{backend} weight identity differs from model-resolution")
            locked_semantic = normalized_model_source(lane.get("semantic_source"), f"models[{key}].lanes.{backend}.semantic_source")
            require_gate(locked_semantic == normalized_model_source(resolved.get("semantic_source"), f"model-resolution.{lane_id}.semantic_source"), f"models[{key}].lanes.{backend} semantic identity differs from model-resolution")
            locked_tokenizer = lane.get("tokenizer_source")
            resolved_tokenizer = resolved.get("tokenizer_source")
            require_gate((locked_tokenizer is None) == (resolved_tokenizer is None), f"models[{key}].lanes.{backend} tokenizer resolution presence mismatch")
            normalized_tokenizer = None
            if locked_tokenizer is not None:
                normalized_tokenizer = normalized_model_source(locked_tokenizer, f"models[{key}].lanes.{backend}.tokenizer_source")
                require_gate(normalized_tokenizer == normalized_model_source(resolved_tokenizer, f"model-resolution.{lane_id}.tokenizer_source"), f"models[{key}].lanes.{backend} tokenizer identity differs from model-resolution")
            lane_identity = {
                "catalog_lane_id": lane_id,
                "backend": backend,
                "format": require_string(lane.get("format"), f"models[{key}].lanes.{backend}.format"),
                "hardware_id": require_string(lane.get("hardware_id"), f"models[{key}].lanes.{backend}.hardware_id"),
                "weight_source": locked_weight,
                "semantic_source": locked_semantic,
                "tokenizer_source": normalized_tokenizer,
            }
            lane_identity["identity_sha256"] = canonical_json_sha256(lane_identity)
            normalized_lanes[backend] = lane_identity
        identity: dict[str, Any] = {
            "key": key,
            "official_model_id": official_model_id,
            "role": role,
            "lanes": normalized_lanes,
        }
        if role == "primary":
            presets = require_object(model.get("generation_presets"), f"models[{key}].generation_presets")
            require_gate(presets, f"models[{key}].generation_presets must not be empty")
            identity["generation_presets_sha256"] = canonical_json_sha256(presets)
        identity["identity_sha256"] = canonical_json_sha256(identity)
        model_identities.append(identity)
        models[key] = model
    require_gate(set(models) == set(expected_models), "models.lock model matrix is incomplete")
    require_gate(set(resolution_lanes) == expected_resolution_ids, "model-resolution lane matrix differs from models.lock")
    require_gate(child_manifest.get("primary_models") == sorted(VNEXT_PRIMARY_MODELS), "delegated manifest primary_models mismatch")
    require_gate(child_manifest.get("supplemental_models") == sorted(VNEXT_SUPPLEMENTAL_MODELS), "delegated manifest supplemental_models mismatch")

    binaries_path, binaries_rel, binaries_digest = require_indexed_artifact(
        root,
        artifact_index,
        "legacy-binaries.json",
        child_manifest.get("legacy_binaries_sha256"),
        "legacy-binaries",
    )
    binaries_doc = read_json_object(binaries_path, "legacy-binaries")
    require_gate(binaries_doc.get("source_git_sha") == VNEXT_FROZEN_LEGACY_SHA, "legacy-binaries frozen source SHA mismatch")
    binary_rows = require_list(binaries_doc.get("binaries"), "legacy-binaries.binaries")
    binary_identities: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(binary_rows):
        binary = require_object(raw, f"legacy-binaries.binaries[{index}]")
        backend = require_string(binary.get("backend"), f"legacy-binaries.binaries[{index}].backend")
        require_gate(backend in {"cuda", "metal"} and backend not in binary_identities, f"invalid or duplicate legacy binary backend {backend}")
        artifact, artifact_rel, digest = require_indexed_artifact(
            root,
            artifact_index,
            binary.get("artifact_binary"),
            binary.get("binary_sha256"),
            f"legacy-binaries.{backend}.artifact_binary",
        )
        require_gate(sha256(artifact) == digest, f"legacy-binaries.{backend} artifact digest mismatch")
        features = require_list(binary.get("cargo_features"), f"legacy-binaries.{backend}.cargo_features")
        require_gate(features and all(isinstance(item, str) and item for item in features), f"legacy-binaries.{backend}.cargo_features invalid")
        identity = {
            "backend": backend,
            "hardware_id": require_string(binary.get("hardware_id"), f"legacy-binaries.{backend}.hardware_id"),
            "artifact_binary": artifact_rel,
            "binary_sha256": digest,
            "cargo_features": features,
            "build_command": require_list(binary.get("build_command"), f"legacy-binaries.{backend}.build_command"),
        }
        identity["identity_sha256"] = canonical_json_sha256(identity)
        binary_identities[backend] = identity
    require_gate(set(binary_identities) == {"cuda", "metal"}, "legacy-binaries must contain CUDA and Metal identities")

    correctness = require_object(child_manifest.get("correctness_lanes"), "delegated manifest correctness_lanes")
    expected_correctness = {f"{key}/{backend}" for key in VNEXT_PRIMARY_MODELS for backend in ("cuda", "metal")}
    require_gate(set(correctness) == expected_correctness, "delegated correctness lane matrix mismatch")
    require_gate(all(status in {"pass", "blocked"} for status in correctness.values()), "delegated correctness lane status invalid")
    expected_correctness_status = {
        "m1-qwen35-4b/cuda": "pass",
        "m1-qwen35-4b/metal": "blocked",
        "m2-qwen35-35b-a3b/cuda": "pass",
        "m2-qwen35-35b-a3b/metal": "blocked",
        "m3-qwen3-30b-a3b/cuda": "pass",
        "m3-qwen3-30b-a3b/metal": "pass",
    }
    require_gate(correctness == expected_correctness_status, "delegated correctness status matrix mismatch")

    required_scenario_configs: set[str] = set()
    correctness_invocations: dict[str, dict[str, Any]] = {}
    scenario_runner_path = "scripts/release/runtime_vnext_baseline_scenarios.py"
    scenario_runner_sha = sha256(REPO_ROOT / scenario_runner_path)
    require_gate(scenario_runner_sha is not None, "scenario runner is missing from the clean checkout")
    for lane_key, status in sorted(correctness.items()):
        model_key, backend = lane_key.split("/", 1)
        lane_path, _, _ = require_indexed_artifact(
            root,
            artifact_index,
            f"correctness/{model_key}/{backend}/lane.json",
            artifact_index.get(f"correctness/{model_key}/{backend}/lane.json", {}).get("sha256"),
            f"correctness.{model_key}.{backend}.lane",
        )
        lane = read_json_object(lane_path, f"correctness.{model_key}.{backend}.lane")
        require_gate(lane.get("status") == status, f"correctness.{model_key}.{backend} status mismatch")
        model_lane = require_object(require_object(models[model_key]["lanes"], f"models[{model_key}].lanes")[backend], f"models[{model_key}].lanes.{backend}")
        expected_files = {row["path"]: row["sha256"] for row in normalized_file_locks(model_lane.get("files"), f"models[{model_key}].lanes.{backend}.files")}
        require_gate(lane.get("model_key") == model_key and lane.get("backend") == backend, f"correctness.{model_key}.{backend} model/backend identity mismatch")
        require_gate(lane.get("model_revision") == model_lane.get("revision"), f"correctness.{model_key}.{backend} model revision mismatch")
        require_gate(lane.get("model_files") == expected_files, f"correctness.{model_key}.{backend} model files mismatch")
        require_gate(lane.get("hardware_id") == model_lane.get("hardware_id"), f"correctness.{model_key}.{backend} hardware identity mismatch")
        require_gate(lane.get("binary_sha256") == binary_identities[backend]["binary_sha256"], f"correctness.{model_key}.{backend} binary identity mismatch")
        if status == "blocked":
            require_gate(lane.get("current_support") is False and lane.get("comparable") is False and lane.get("waiver") is False, f"correctness.{model_key}.{backend} blocked policy mismatch")
            for field in (
                "failure_class",
                "reason",
                "first_failure",
                "downstream_goal",
                "implementation_path",
                "acceptance_path",
                "downstream_acceptance_pass_line",
            ):
                require_string(lane.get(field), f"correctness.{model_key}.{backend}.{field}")
            attempted = require_list(lane.get("attempted_command"), f"correctness.{model_key}.{backend}.attempted_command")
            require_gate(attempted and Path(str(attempted[0])).name == "ferrum" and any(part in {"run", "serve"} for part in attempted[1:]), f"correctness.{model_key}.{backend} blocked attempt is not a product entrypoint")
            require_gate(isinstance(lane.get("attempted_returncode"), int) and lane.get("attempted_returncode") != 0, f"correctness.{model_key}.{backend} blocked returncode mismatch")
            failure_log = require_string(lane.get("failure_log"), f"correctness.{model_key}.{backend}.failure_log")
            require_gate(failure_log in artifact_index and (root / failure_log).stat().st_size > 0, f"correctness.{model_key}.{backend} blocked failure log missing")
            require_gate("scenario_report" not in lane and "pass_line" not in lane, f"correctness.{model_key}.{backend} blocked lane contains pass evidence")
            continue
        report_ref = require_object(lane.get("scenario_report"), f"correctness.{model_key}.{backend}.scenario_report")
        report_path, _, _ = require_indexed_artifact(
            root,
            artifact_index,
            report_ref.get("path"),
            report_ref.get("sha256"),
            f"correctness.{model_key}.{backend}.scenario_report",
        )
        report = read_json_object(report_path, f"correctness.{model_key}.{backend}.scenario_report")
        for field, expected in {
            "model_key": model_key,
            "backend": backend,
            "model_revision": model_lane.get("revision"),
            "model_files": expected_files,
            "hardware_id": model_lane.get("hardware_id"),
            "binary_sha256": binary_identities[backend]["binary_sha256"],
            "models_lock_sha256": models_lock_digest,
        }.items():
            require_gate(report.get(field) == expected, f"correctness.{model_key}.{backend}.scenario_report {field} mismatch")
        report_expectations = validate_vnext_g00_expectations_snapshot(
            root,
            artifact_index,
            report.get("expectations_catalog"),
            source_path=expectations_file,
            source_sha256=expectations_sha,
            label=f"correctness.{model_key}.{backend}.scenario_report.expectations_catalog",
        )
        require_gate(
            report.get("expectations_catalog_sha256") == report_expectations["sha256"],
            f"correctness.{model_key}.{backend} report expectations SHA mismatch",
        )
        validate_vnext_g00_runner_identity(
            report.get("runner"),
            scenario_runner_path=scenario_runner_path,
            scenario_runner_sha256=scenario_runner_sha,
            validator_git_sha=validator_git_sha,
            contract_by_path=contract_by_path,
            label=f"correctness.{model_key}.{backend}.scenario_report.runner",
            verify_checkout=verify_checkout,
        )
        invocation_ref = require_object(report.get("executor_invocation"), f"correctness.{model_key}.{backend}.executor_invocation")
        invocation_path, invocation_rel, invocation_digest = require_indexed_artifact(
            root,
            artifact_index,
            invocation_ref.get("path"),
            invocation_ref.get("sha256"),
            f"correctness.{model_key}.{backend}.executor_invocation",
        )
        invocation = read_json_object(invocation_path, f"correctness.{model_key}.{backend}.executor_invocation")
        require_gate(invocation.get("mode") == "canonical", f"correctness.{model_key}.{backend} executor mode mismatch")
        require_gate(invocation.get("runner_path") == scenario_runner_path and invocation.get("runner_sha256") == scenario_runner_sha, f"correctness.{model_key}.{backend} executor runner identity mismatch")
        correctness_invocations[lane_key] = {
            "path": invocation_rel,
            "sha256": invocation_digest,
            "runner_path": scenario_runner_path,
            "runner_sha256": scenario_runner_sha,
            "mode": "canonical",
        }
        config_ref = require_object(report.get("effective_config"), f"correctness.{model_key}.{backend}.effective_config")
        _, config_rel, _ = require_indexed_artifact(
            root,
            artifact_index,
            config_ref.get("path"),
            config_ref.get("sha256"),
            f"correctness.{model_key}.{backend}.effective_config",
        )
        required_scenario_configs.add(config_rel)

    require_gate(
        child_manifest.get("correctness_executor_invocations")
        == correctness_invocations,
        "delegated correctness executor invocation matrix mismatch",
    )

    config_refs: dict[str, dict[str, Any]] = {}

    def register_config(raw_path: Any, raw_digest: Any, referenced_by: str) -> None:
        path, relative, digest = require_indexed_artifact(
            root,
            artifact_index,
            raw_path,
            raw_digest,
            f"config reference from {referenced_by}",
        )
        require_gate(path.suffix.lower() == ".json", f"effective config must be JSON: {relative}")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        require_gate(isinstance(parsed, (dict, list)) and bool(parsed), f"effective config is empty: {relative}")
        existing = config_refs.setdefault(relative, {"path": relative, "sha256": digest, "referenced_by": []})
        require_gate(existing["sha256"] == digest, f"effective config has conflicting hashes: {relative}")
        if referenced_by not in existing["referenced_by"]:
            existing["referenced_by"].append(referenced_by)

    def discover_config_refs(value: Any, referenced_by: str) -> None:
        if isinstance(value, dict):
            config = value.get("effective_config")
            if isinstance(config, str) and "effective_config_sha256" in value:
                register_config(config, value.get("effective_config_sha256"), referenced_by)
            elif isinstance(config, dict) and "path" in config and "sha256" in config:
                register_config(config.get("path"), config.get("sha256"), referenced_by)
            for child in value.values():
                discover_config_refs(child, referenced_by)
        elif isinstance(value, list):
            for child in value:
                discover_config_refs(child, referenced_by)

    for relative in sorted(artifact_index):
        if not relative.endswith(".json"):
            continue
        path = root / relative
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GateError(f"indexed JSON artifact is invalid: {relative}: {exc}") from exc
        discover_config_refs(value, relative)
    require_gate(required_scenario_configs <= set(config_refs), "scenario effective config hashes are absent from aggregated config provenance")
    require_gate(config_refs, "vnext-g00 config provenance is empty")

    config_identities = sorted(config_refs.values(), key=lambda row: row["path"])
    for row in config_identities:
        row["referenced_by"].sort()
    model_identities.sort(key=lambda row: row["key"])
    return {
        "kind": "vnext-g00",
        "child_manifest": {
            "path": "manifest.json",
            "sha256": manifest_digest,
            "artifact_count": len(artifact_index),
            "contract_sha256": require_sha256(child_manifest.get("contract_sha256"), "delegated manifest contract_sha256"),
        },
        "models_lock": {
            "path": models_lock_rel,
            "sha256": models_lock_digest,
            "catalog_sha256": require_sha256(models_lock.get("catalog_sha256"), "models.lock.catalog_sha256"),
            "preset_catalog_sha256": require_sha256(models_lock.get("preset_catalog_sha256"), "models.lock.preset_catalog_sha256"),
        },
        "expectations_catalog": {
            "path": expectations_path,
            "sha256": expectations_sha,
        },
        "model_resolution": {
            "path": resolution_rel,
            "sha256": resolution_digest,
            "lane_count": len(resolution_lanes),
            "resolver": resolver_identity,
        },
        "legacy_binaries": {
            "path": binaries_rel,
            "sha256": binaries_digest,
            "identities": [binary_identities[key] for key in sorted(binary_identities)],
        },
        "model_identities": model_identities,
        "config_artifacts": config_identities,
        "correctness_lanes": dict(sorted(correctness.items())),
        "correctness_executor_invocations": correctness_invocations,
        "artifact_index_sha256": canonical_json_sha256(child_manifest["artifact_index"]),
    }


def verify_child_pass_line(
    lane_command: LaneCommand,
    stdout: str,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any] | None:
    expected = lane_command.expected_child_pass_line
    if expected is None:
        return None
    if expected not in stdout.splitlines():
        raise GateError(f"delegated command did not print required PASS line: {expected}")
    full_redteam = None
    if lane_command.provenance_kind == "vnext-g00":
        full_redteam = validate_vnext_g00_full_redteam(lane_command, stdout)
    if lane_command.child_manifest_path is None:
        return None
    try:
        child_manifest_bytes = lane_command.child_manifest_path.read_bytes()
        child_manifest = json.loads(child_manifest_bytes.decode("utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GateError(f"invalid delegated manifest {lane_command.child_manifest_path}: {exc}") from exc
    if not isinstance(child_manifest, dict):
        raise GateError(f"delegated manifest must be a JSON object: {lane_command.child_manifest_path}")
    if child_manifest.get("status") != "pass":
        raise GateError(f"delegated manifest status is not pass: {lane_command.child_manifest_path}")
    if child_manifest.get("pass_line") != expected:
        raise GateError(f"delegated manifest pass_line mismatch: {lane_command.child_manifest_path}")
    if (
        lane_command.expected_source_git_sha is not None
        and child_manifest.get("source_git_sha") != lane_command.expected_source_git_sha
    ):
        raise GateError(f"delegated manifest source_git_sha mismatch: {lane_command.child_manifest_path}")
    child_manifest_digest = hashlib.sha256(child_manifest_bytes).hexdigest()
    if lane_command.provenance_kind == "vnext-g00a":
        return validate_vnext_g00a_provenance(
            lane_command,
            child_manifest,
            child_manifest_digest,
            verify_checkout=verify_checkout,
        )
    if lane_command.provenance_kind == "vnext-g00f":
        return validate_vnext_g00f_provenance(
            lane_command,
            child_manifest,
            child_manifest_digest,
            verify_checkout=verify_checkout,
        )
    if lane_command.provenance_kind == "vnext-g00":
        provenance = validate_vnext_g00_provenance(
            lane_command,
            child_manifest,
            child_manifest_digest,
            verify_checkout=verify_checkout,
        )
        require_gate(full_redteam is not None, "vnext-g00 full-redteam provenance is missing")
        provenance["full_redteam"] = full_redteam
        return provenance
    if lane_command.provenance_kind == "vnext-g01a":
        return validate_vnext_g01a_provenance(
            lane_command,
            child_manifest,
            child_manifest_digest,
            verify_checkout=verify_checkout,
        )
    if lane_command.provenance_kind == "vnext-g01a-s0a":
        return validate_vnext_g01a_s0a_provenance(
            lane_command,
            child_manifest,
            child_manifest_digest,
            verify_checkout=verify_checkout,
        )
    if lane_command.provenance_kind == "g0-source-unit":
        return validate_g0_source_unit_provenance(
            lane_command,
            child_manifest,
            child_manifest_digest,
        )
    return {
        "kind": "delegated-manifest",
        "child_manifest": {
            "path": str(lane_command.child_manifest_path),
            "sha256": child_manifest_digest,
        },
    }


def run_child(
    cmd: list[str],
    out_dir: Path,
    timeout: int | None,
    *,
    prepare_out_dir: bool = True,
) -> subprocess.CompletedProcess[str]:
    if prepare_out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            raise GateError(f"delegated command requires a fresh --out directory: {out_dir}")
    started = time.monotonic()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    duration = time.monotonic() - started
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_gate.child.stdout").write_text(proc.stdout, errors="replace")
    (out_dir / "run_gate.child.stderr").write_text(proc.stderr, errors="replace")
    (out_dir / "run_gate.child.command.json").write_text(
        json.dumps(
            {
                "cmd": cmd,
                "duration_sec": duration,
                "env_overrides": {"PYTHONDONTWRITEBYTECODE": "1"},
            },
            indent=2,
        )
        + "\n"
    )
    return proc


def pass_line(lane: str, out_dir: Path) -> str:
    return f"FERRUM GATE {lane} PASS: {out_dir}"


def child_execution_artifacts(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in (
        "run_gate.child.command.json",
        "run_gate.child.stdout",
        "run_gate.child.stderr",
    ):
        path = out_dir / name
        digest = sha256(path)
        if digest is None:
            continue
        rows.append(
            {
                "path": name,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def manifest(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    lane_command: LaneCommand | None,
    status: str,
    started_at: str,
    finished_at: str,
    duration_sec: float,
    child_returncode: int | None,
    child_pass_line: str | None,
    child_artifacts: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    binary_path = lane_command.binary_path if lane_command else None
    binary_sha = sha256(binary_path) if binary_path else None
    return {
        "schema_version": 1,
        "lane": args.lane,
        "status": status,
        "command_line": command_line(),
        "delegated_command_line": lane_command.cmd if lane_command else None,
        "child_returncode": child_returncode,
        "child_pass_line": child_pass_line,
        "child_artifacts": child_artifacts,
        "child_execution_artifacts": child_execution_artifacts(out_dir),
        "git_sha": git_sha(),
        "dirty_status": git_dirty_status(),
        "artifact_dir": str(out_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": duration_sec,
        "binary": {
            "path": str(binary_path) if binary_path else None,
            "sha256": binary_sha,
        },
        "model": lane_command.model if lane_command else args.model,
        "sanitized_env": sanitized_env_summary(),
        "pass_line": pass_line(args.lane, out_dir) if status == "pass" else None,
        "error": error,
    }


def list_lanes() -> None:
    for lane in LANES:
        print(lane)


def require_selftest(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_selftest_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)


def write_selftest_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def selftest_file_lock(seed: str, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        "size_bytes": 1024,
    }


def selftest_artifact_index(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in CHILD_INDEX_EXCLUDED:
            continue
        digest = sha256(path)
        require_selftest(digest is not None, f"missing selftest artifact digest: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "role": relative.split("/", 1)[0] if "/" in relative else "root-manifest",
            }
        )
    return rows


def make_selftest_vnext_g00_artifact(root: Path) -> LaneCommand:
    root.mkdir(parents=True, exist_ok=True)
    models: list[dict[str, Any]] = []
    resolution_lanes: list[dict[str, Any]] = []
    all_models = {**VNEXT_PRIMARY_MODELS, **VNEXT_SUPPLEMENTAL_MODELS}
    for key, official_model_id in all_models.items():
        lanes: dict[str, Any] = {}
        for backend in ("cuda", "metal"):
            lane_id = f"{key}-{backend}"
            weight = {
                "repo": f"org/{key}-{backend}",
                "revision": hashlib.sha1(f"weight-{lane_id}".encode("utf-8")).hexdigest(),
                "files": [selftest_file_lock(f"weight-file-{lane_id}", f"{key}-{backend}.weights")],
            }
            semantic = {
                "repo": f"org/{key}-semantic",
                "revision": hashlib.sha1(f"semantic-{lane_id}".encode("utf-8")).hexdigest(),
                "files": [selftest_file_lock(f"semantic-file-{lane_id}", "config.json")],
            }
            lanes[backend] = {
                "catalog_lane_id": lane_id,
                "repo": weight["repo"],
                "revision": weight["revision"],
                "format": "safetensors" if backend == "cuda" else "gguf",
                "hardware_id": f"{backend}-selftest",
                "files": weight["files"],
                "semantic_source": semantic,
            }
            resolution_lanes.append(
                {
                    "catalog_lane_id": lane_id,
                    "backend": backend,
                    "model_id": VNEXT_RESOLUTION_MODEL_IDS[key],
                    "format": lanes[backend]["format"],
                    "weight_source": weight,
                    "semantic_source": semantic,
                }
            )
        model: dict[str, Any] = {
            "key": key,
            "official_model_id": official_model_id,
            "role": "primary" if key in VNEXT_PRIMARY_MODELS else "supplemental",
            "lanes": lanes,
        }
        if key in VNEXT_PRIMARY_MODELS:
            model["generation_presets"] = {
                "P_DETERMINISTIC": {"temperature": 0, "seed": 9271},
                "P_NO_THINKING": {"temperature": 0.7, "seed": 9271},
                "P_THINKING": {"temperature": 0.7, "seed": 9271},
                "P_OFFICIAL_DEFAULT": {"temperature": 0.7, "seed": 9271},
            }
        models.append(model)

    resolution_path = root / "model-resolution.json"
    write_selftest_json(
        resolution_path,
        {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_model_resolution",
            "resolver": {
                "path": "scripts/release/runtime_vnext_model_resolver.py",
                "sha256": hashlib.sha256(b"selftest-resolver").hexdigest(),
            },
            "lanes": resolution_lanes,
        },
    )
    resolution_digest = sha256(resolution_path)
    require_selftest(resolution_digest is not None, "selftest model-resolution digest missing")
    expectations_file = REPO_ROOT / VNEXT_LEGACY_EXPECTATIONS_PATH
    expectations_digest = sha256(expectations_file)
    require_selftest(expectations_digest is not None, "selftest expectations catalog digest missing")
    expectations_binding = {
        "path": VNEXT_LEGACY_EXPECTATIONS_PATH,
        "sha256": expectations_digest,
    }
    expectations_snapshot_path = root / VNEXT_LEGACY_EXPECTATIONS_SNAPSHOT
    expectations_snapshot_path.write_bytes(expectations_file.read_bytes())
    expectations_snapshot = {
        "kind": "raw-json",
        "path": VNEXT_LEGACY_EXPECTATIONS_SNAPSHOT,
        "sha256": expectations_digest,
    }
    scenario_runner_relative = "scripts/release/runtime_vnext_baseline_scenarios.py"
    scenario_runner_path = REPO_ROOT / scenario_runner_relative
    scenario_runner_digest = sha256(scenario_runner_path)
    require_selftest(scenario_runner_digest is not None, "selftest scenario runner digest missing")
    selftest_validator_git_sha = "1" * 40
    selftest_runner_identity = {
        "path": scenario_runner_relative,
        "sha256": scenario_runner_digest,
        "git_sha": selftest_validator_git_sha,
        "source_tree_sha": "2" * 40,
        "git_blob_sha": "3" * 40,
        "dirty_status": {"is_dirty": False, "status_short": []},
    }
    models_lock_path = root / "models.lock.json"
    write_selftest_json(
        models_lock_path,
        {
            "schema_version": 1,
            "source_git_sha": VNEXT_FROZEN_LEGACY_SHA,
            "catalog_sha256": hashlib.sha256(b"selftest-model-catalog").hexdigest(),
            "preset_catalog_sha256": hashlib.sha256(b"selftest-preset-catalog").hexdigest(),
            "expectations_catalog": expectations_binding,
            "model_resolution": {"path": "model-resolution.json", "sha256": resolution_digest},
            "models": models,
        },
    )

    binary_rows: list[dict[str, Any]] = []
    binary_sha_by_backend: dict[str, str] = {}
    for backend in ("cuda", "metal"):
        binary_rel = f"binaries/{backend}/ferrum"
        binary_path = root / binary_rel
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        binary_path.write_text(f"{backend} selftest binary\n", encoding="utf-8")
        digest = sha256(binary_path)
        require_selftest(digest is not None, f"selftest {backend} binary digest missing")
        binary_sha_by_backend[backend] = digest
        binary_rows.append(
            {
                "backend": backend,
                "hardware_id": f"{backend}-selftest",
                "artifact_binary": binary_rel,
                "binary_sha256": digest,
                "cargo_features": [backend],
                "build_command": ["cargo", "build", "--release", "--features", backend],
            }
        )
    binaries_path = root / "legacy-binaries.json"
    write_selftest_json(
        binaries_path,
        {
            "schema_version": 1,
            "source_git_sha": VNEXT_FROZEN_LEGACY_SHA,
            "binaries": binary_rows,
        },
    )

    model_by_key = {row["key"]: row for row in models}
    correctness: dict[str, str] = {}
    executor_invocations: dict[str, dict[str, Any]] = {}
    models_lock_digest = sha256(models_lock_path)
    require_selftest(models_lock_digest is not None, "selftest models.lock digest missing")
    for model_key in VNEXT_PRIMARY_MODELS:
        for backend in ("cuda", "metal"):
            lane = model_by_key[model_key]["lanes"][backend]
            correctness[f"{model_key}/{backend}"] = "pass"
            config_rel = f"correctness/{model_key}/{backend}/effective-config.json"
            write_selftest_json(
                root / config_rel,
                {
                    "schema_version": 1,
                    "model_key": model_key,
                    "backend": backend,
                    "typed_effective_config": {"max_tokens": 128},
                },
            )
            config_digest = sha256(root / config_rel)
            require_selftest(config_digest is not None, "selftest correctness config digest missing")
            invocation_rel = f"correctness/{model_key}/{backend}/executor-invocation.json"
            write_selftest_json(
                root / invocation_rel,
                {
                    "schema_version": 1,
                    "mode": "canonical",
                    "runner_path": scenario_runner_relative,
                    "runner_sha256": scenario_runner_digest,
                },
            )
            invocation_digest = sha256(root / invocation_rel)
            require_selftest(invocation_digest is not None, "selftest executor invocation digest missing")
            invocation_identity = {
                "path": invocation_rel,
                "sha256": invocation_digest,
                "runner_path": scenario_runner_relative,
                "runner_sha256": scenario_runner_digest,
                "mode": "canonical",
            }
            executor_invocations[f"{model_key}/{backend}"] = invocation_identity
            report_rel = f"correctness/{model_key}/{backend}/scenario-report.json"
            model_files = {row["path"]: row["sha256"] for row in lane["files"]}
            write_selftest_json(
                root / report_rel,
                {
                    "schema_version": 1,
                    "status": "pass",
                    "model_key": model_key,
                    "backend": backend,
                    "model_revision": lane["revision"],
                    "model_files": model_files,
                    "hardware_id": lane["hardware_id"],
                    "binary_sha256": binary_sha_by_backend[backend],
                    "models_lock_sha256": models_lock_digest,
                    "runner": copy.deepcopy(selftest_runner_identity),
                    "expectations_catalog": copy.deepcopy(expectations_snapshot),
                    "expectations_catalog_sha256": expectations_digest,
                    "executor_invocation": {
                        "path": invocation_rel,
                        "sha256": invocation_digest,
                    },
                    "effective_config": {
                        "path": config_rel,
                        "sha256": config_digest,
                    },
                },
            )
            report_digest = sha256(root / report_rel)
            require_selftest(report_digest is not None, "selftest scenario report digest missing")
            write_selftest_json(
                root / f"correctness/{model_key}/{backend}/lane.json",
                {
                    "schema_version": 1,
                    "status": "pass",
                    "model_key": model_key,
                    "backend": backend,
                    "model_revision": lane["revision"],
                    "model_files": model_files,
                    "hardware_id": lane["hardware_id"],
                    "binary_sha256": binary_sha_by_backend[backend],
                    "scenario_report": {"path": report_rel, "sha256": report_digest},
                },
            )
            if backend == "metal" and model_key in {
                "m1-qwen35-4b",
                "m2-qwen35-35b-a3b",
            }:
                correctness[f"{model_key}/{backend}"] = "blocked"
                executor_invocations.pop(f"{model_key}/{backend}")
                failure_log_rel = f"correctness/{model_key}/{backend}/blocked.log"
                (root / failure_log_rel).write_text("legacy model/backend unsupported\n")
                write_selftest_json(
                    root / f"correctness/{model_key}/{backend}/lane.json",
                    {
                        "schema_version": 1,
                        "status": "blocked",
                        "model_key": model_key,
                        "backend": backend,
                        "model_revision": lane["revision"],
                        "model_files": model_files,
                        "hardware_id": lane["hardware_id"],
                        "binary_sha256": binary_sha_by_backend[backend],
                        "current_support": False,
                        "comparable": False,
                        "waiver": False,
                        "failure_class": "legacy-model-backend-unsupported",
                        "reason": "selftest frozen unsupported lane",
                        "first_failure": "model load rejected before inference",
                        "downstream_goal": "G08A" if model_key == "m1-qwen35-4b" else "G08B",
                        "implementation_path": "vNext model migration",
                        "acceptance_path": "runtime-vNext model lane gate",
                        "downstream_acceptance_pass_line": "FERRUM RUNTIME VNEXT MODEL PASS: fixture",
                        "attempted_command": ["ferrum", "run", "--model", model_key],
                        "attempted_returncode": 1,
                        "failure_log": failure_log_rel,
                    },
                )

    workload_config_rel = "workloads/m3-qwen3-30b-a3b/cuda/random/c1.config.json"
    write_selftest_json(root / workload_config_rel, {"schema_version": 1, "seed": 9271, "concurrency": 1})
    workload_config_digest = sha256(root / workload_config_rel)
    require_selftest(workload_config_digest is not None, "selftest workload config digest missing")
    write_selftest_json(
        root / "performance/m3-qwen3-30b-a3b/cuda/summary.json",
        {
            "schema_version": 1,
            "workload": {
                "effective_config": workload_config_rel,
                "effective_config_sha256": workload_config_digest,
            },
        },
    )

    artifact_index = selftest_artifact_index(root)
    pass_line_value = f"FERRUM RUNTIME VNEXT G00 BASELINE PASS: {root}"
    write_selftest_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "status": "pass",
            "source_git_sha": VNEXT_FROZEN_LEGACY_SHA,
            "validator_git_sha": selftest_validator_git_sha,
            "validator_dirty_status": [],
            "artifact_dir": str(root),
            "contract_sha256": hashlib.sha256(b"selftest-contract").hexdigest(),
            "contract_files": [
                {
                    "path": VNEXT_LEGACY_EXPECTATIONS_PATH,
                    "sha256": expectations_digest,
                    "size_bytes": expectations_file.stat().st_size,
                },
                {
                    "path": scenario_runner_relative,
                    "sha256": scenario_runner_digest,
                    "size_bytes": scenario_runner_path.stat().st_size,
                },
            ],
            "artifact_index": artifact_index,
            "artifact_count": len(artifact_index),
            "models_lock_sha256": models_lock_digest,
            "expectations_catalog": expectations_binding,
            "correctness_executor_invocations": executor_invocations,
            "legacy_binaries_sha256": sha256(binaries_path),
            "primary_models": sorted(VNEXT_PRIMARY_MODELS),
            "supplemental_models": sorted(VNEXT_SUPPLEMENTAL_MODELS),
            "correctness_lanes": correctness,
            "waiver_count": 0,
            "pass_line": pass_line_value,
        },
    )
    return LaneCommand(
        ["selftest", "--require-full-self-test"],
        expected_child_pass_line=pass_line_value,
        child_manifest_path=root / "manifest.json",
        expected_source_git_sha=VNEXT_FROZEN_LEGACY_SHA,
        provenance_kind="vnext-g00",
    )


def refresh_selftest_vnext_manifest(root: Path, *, sync_models_lock: bool = False, sync_binaries: bool = False) -> None:
    manifest_path = root / "manifest.json"
    doc = read_json_object(manifest_path, "selftest delegated manifest")
    doc["artifact_dir"] = str(root)
    doc["pass_line"] = f"FERRUM RUNTIME VNEXT G00 BASELINE PASS: {root}"
    if sync_models_lock:
        doc["models_lock_sha256"] = sha256(root / "models.lock.json")
    if sync_binaries:
        doc["legacy_binaries_sha256"] = sha256(root / "legacy-binaries.json")
    doc["artifact_index"] = selftest_artifact_index(root)
    doc["artifact_count"] = len(doc["artifact_index"])
    write_selftest_json(manifest_path, doc)


def selftest_vnext_lane(root: Path) -> LaneCommand:
    pass_line_value = f"FERRUM RUNTIME VNEXT G00 BASELINE PASS: {root}"
    return LaneCommand(
        ["selftest", "--require-full-self-test"],
        expected_child_pass_line=pass_line_value,
        child_manifest_path=root / "manifest.json",
        expected_source_git_sha=VNEXT_FROZEN_LEGACY_SHA,
        provenance_kind="vnext-g00",
    )


def selftest_vnext_full_summary(**updates: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "mode": "full",
        "mutation_assertion_count": VNEXT_G00_REDTEAM_MUTATION_COUNT,
        "expected_mutation_assertion_count": VNEXT_G00_REDTEAM_MUTATION_COUNT,
        "mutation_names": list(VNEXT_G00_REDTEAM_MUTATION_NAMES),
        "validator_counts": {"full-root": VNEXT_G00_REDTEAM_MUTATION_COUNT},
        "valid_fixture_assertion_count": 3,
    }
    summary.update(updates)
    return summary


def selftest_vnext_stdout(
    lane: LaneCommand,
    *,
    full_pass_line: str = VNEXT_G00_FULL_SELFTEST_PASS,
    summary: dict[str, Any] | None = None,
) -> str:
    require_selftest(lane.expected_child_pass_line is not None, "selftest vnext lane lacks PASS line")
    summary = summary or selftest_vnext_full_summary()
    return "\n".join(
        [
            f"{VNEXT_G00_SELFTEST_SUMMARY_PREFIX} {json.dumps(summary, sort_keys=True, separators=(',', ':'))}",
            full_pass_line,
            lane.expected_child_pass_line,
            "",
        ]
    )


def expect_vnext_provenance_reject(
    valid_root: Path,
    name: str,
    mutate: Any,
    marker: str,
) -> None:
    case = valid_root.parent / f"vnext-reject-{name}"
    shutil.copytree(valid_root, case)
    refresh_selftest_vnext_manifest(case)
    mutate(case)
    lane = selftest_vnext_lane(case)
    try:
        verify_child_pass_line(
            lane,
            selftest_vnext_stdout(lane),
            verify_checkout=False,
        )
    except GateError as exc:
        require_selftest(marker.lower() in str(exc).lower(), f"{name} rejected for unexpected reason: {exc}")
        return
    raise AssertionError(f"vnext provenance mutation {name} unexpectedly passed")


def mutate_selftest_json(path: Path, update: Any) -> None:
    doc = read_json_object(path, "selftest mutation input")
    update(doc)
    write_selftest_json(path, doc)


def tamper_selftest_vnext_invocation_mode(root: Path) -> None:
    lane_key = "m3-qwen3-30b-a3b/cuda"
    base = root / "correctness/m3-qwen3-30b-a3b/cuda"
    invocation_path = base / "executor-invocation.json"
    mutate_selftest_json(invocation_path, lambda data: data.update({"mode": "discover"}))
    invocation_digest = sha256(invocation_path)
    require_selftest(invocation_digest is not None, "tampered invocation digest missing")
    report_path = base / "scenario-report.json"
    mutate_selftest_json(
        report_path,
        lambda data: data["executor_invocation"].update({"sha256": invocation_digest}),
    )
    report_digest = sha256(report_path)
    require_selftest(report_digest is not None, "tampered scenario report digest missing")
    mutate_selftest_json(
        base / "lane.json",
        lambda data: data["scenario_report"].update({"sha256": report_digest}),
    )
    mutate_selftest_json(
        root / "manifest.json",
        lambda data: data["correctness_executor_invocations"][lane_key].update(
            {"sha256": invocation_digest, "mode": "discover"}
        ),
    )
    refresh_selftest_vnext_manifest(root)


def update_selftest_vnext_report(root: Path, lane_key: str, update: Any) -> None:
    model_key, backend = lane_key.split("/", 1)
    base = root / "correctness" / model_key / backend
    report_path = base / "scenario-report.json"
    mutate_selftest_json(report_path, update)
    report_digest = sha256(report_path)
    require_selftest(report_digest is not None, "updated scenario report digest missing")
    mutate_selftest_json(
        base / "lane.json",
        lambda data: data["scenario_report"].update({"sha256": report_digest}),
    )


def minimize_selftest_vnext_runner_identity(root: Path) -> None:
    lane_key = "m3-qwen3-30b-a3b/cuda"
    update_selftest_vnext_report(
        root,
        lane_key,
        lambda data: data.update(
            {
                "runner": {
                    "path": data["runner"]["path"],
                    "sha256": data["runner"]["sha256"],
                }
            }
        ),
    )
    refresh_selftest_vnext_manifest(root)


def tamper_selftest_vnext_runner_git_sha(root: Path) -> None:
    lane_key = "m3-qwen3-30b-a3b/cuda"
    update_selftest_vnext_report(
        root,
        lane_key,
        lambda data: data["runner"].update({"git_sha": "4" * 40}),
    )
    refresh_selftest_vnext_manifest(root)


def tamper_selftest_vnext_expectations_kind(root: Path) -> None:
    lane_key = "m3-qwen3-30b-a3b/cuda"
    update_selftest_vnext_report(
        root,
        lane_key,
        lambda data: data["expectations_catalog"].update({"kind": "source-contract"}),
    )
    refresh_selftest_vnext_manifest(root)


def tamper_selftest_vnext_expectations_snapshot(root: Path) -> None:
    snapshot_path = root / VNEXT_LEGACY_EXPECTATIONS_SNAPSHOT
    write_selftest_json(snapshot_path, {"schema_version": 1, "tampered": True})
    snapshot_digest = sha256(snapshot_path)
    require_selftest(snapshot_digest is not None, "tampered expectations snapshot digest missing")
    manifest = read_json_object(root / "manifest.json", "selftest delegated manifest")
    correctness = require_object(manifest.get("correctness_lanes"), "selftest correctness lanes")
    for lane_key, status in correctness.items():
        if status != "pass":
            continue
        update_selftest_vnext_report(
            root,
            lane_key,
            lambda data, digest=snapshot_digest: (
                data["expectations_catalog"].update({"sha256": digest}),
                data.update({"expectations_catalog_sha256": digest}),
            ),
        )
    refresh_selftest_vnext_manifest(root)


def selftest_g00a_response_request(kind: str, url: str, body: Any) -> dict[str, Any]:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "kind": kind,
        "method": "GET",
        "response_body_base64": base64.b64encode(payload).decode("ascii"),
        "response_bytes": len(payload),
        "response_sha256": hashlib.sha256(payload).hexdigest(),
        "status": 200,
        "url": url,
    }


def set_selftest_g00a_response_body(request: dict[str, Any], body: Any) -> None:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    request["response_body_base64"] = base64.b64encode(payload).decode("ascii")
    request["response_bytes"] = len(payload)
    request["response_sha256"] = hashlib.sha256(payload).hexdigest()


def refresh_selftest_g00a_manifest(root: Path) -> None:
    manifest_path = root / "manifest.json"
    doc = read_json_object(manifest_path, "G00a selftest manifest")
    doc["artifact_dir"] = str(root)
    doc["pass_line"] = f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {root}"
    rows = selftest_artifact_index(root)
    indexed = {row["path"]: row for row in rows}
    doc["artifact_index"] = rows
    doc["artifact_count"] = len(rows)
    doc["fact_source_artifacts"] = {
        "coupling_inventory": copy.deepcopy(indexed["coupling-inventory.json"]),
        "model_resolution_input": copy.deepcopy(indexed["model-resolution.input.json"]),
        "model_resolution_live": copy.deepcopy(indexed["model-resolution.json"]),
    }
    lock_row = indexed["model-facts.lock.json"]
    doc["model_facts_lock"] = {
        "path": "model-facts.lock.json",
        "sha256": lock_row["sha256"],
        "size_bytes": lock_row["size_bytes"],
    }
    write_selftest_json(manifest_path, doc)


def selftest_g00a_lane(root: Path) -> LaneCommand:
    return LaneCommand(
        ["selftest"],
        expected_child_pass_line=f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {root}",
        child_manifest_path=root / "manifest.json",
        provenance_kind="vnext-g00a",
    )


def make_selftest_vnext_g00a_artifact(root: Path) -> LaneCommand:
    root.mkdir(parents=True, exist_ok=True)
    selftest_catalog_lanes: list[dict[str, Any]] = []
    expected_identity_lane = "m2-qwen35-35b-a3b-metal"
    for model_key in {**VNEXT_PRIMARY_MODELS, **VNEXT_SUPPLEMENTAL_MODELS}:
        for backend in ("cuda", "metal"):
            lane_id = f"{model_key}-{backend}"
            seed = f"{lane_id}:weight_source"
            primary_weight_path = (
                "model-00001-of-00001.safetensors"
                if lane_id == "m2-qwen35-35b-a3b-cuda"
                else "weight_source.bin"
            )
            selector: dict[str, Any] = {"path": primary_weight_path, "required": True}
            if lane_id == expected_identity_lane:
                selector.update(
                    {
                        "expected_size_bytes": 1024 + len(seed),
                        "expected_sha256": hashlib.sha256(f"file:{seed}".encode("utf-8")).hexdigest(),
                    }
                )
            selectors = [selector]
            if lane_id == "m2-qwen35-35b-a3b-cuda":
                selectors.append(
                    {
                        "path": "model.safetensors.index.json",
                        "required_if_sharded": True,
                    }
                )
            weight_revision = hashlib.sha1(
                f"revision:{lane_id}:weight_source".encode("utf-8")
            ).hexdigest()
            semantic_revision = hashlib.sha1(
                f"revision:{lane_id}:semantic_source".encode("utf-8")
            ).hexdigest()
            reference: dict[str, Any] = {
                "semantic_revision": {
                    "status": "pinned",
                    "value": semantic_revision,
                }
            }
            if model_key == "llama31-8b-compat":
                reference["tokenizer_repo"] = f"fixture/{lane_id}-tokenizer-source"
                reference["tokenizer_revision"] = {
                    "status": "pinned",
                    "value": hashlib.sha1(
                        f"revision:{lane_id}:tokenizer_source".encode("utf-8")
                    ).hexdigest(),
                }
            catalog_lane: dict[str, Any] = {
                "files": selectors,
                "id": lane_id,
                "reference": reference,
                "revision": {"status": "pinned", "value": weight_revision},
            }
            if model_key == "llama31-8b-compat":
                reference["official_upstream"] = {
                    "access_note": "selftest gated upstream",
                    "blob_oid_match_files": ["config.json"],
                    "expected_content_sha256": {"config.json": "6" * 64},
                    "expected_git_oids": {"config.json": "5" * 40},
                    "expected_size_bytes": {"config.json": 128},
                    "repo": "meta-llama/Llama-3.1-8B-Instruct",
                    "required_gated": True,
                    "revision": {"status": "pinned", "value": "4" * 40},
                }
            selftest_catalog_lanes.append(catalog_lane)
    selftest_catalog_lanes.sort(key=lambda row: row["id"])
    require_selftest(len(selftest_catalog_lanes) == 12, "G00a selftest catalog must contain 12 lanes")
    catalog_payloads = {
        "generation-presets.catalog.json": {
            "catalog_id": "g00a-selftest-generation-presets",
            "model_catalog_id": "g00a-selftest-models",
            "schema_version": 1,
        },
        "historical-bugs.catalog.json": {
            "catalog_id": "g00a-selftest-history",
            "concrete_case_count": 28,
            "family_count": 15,
            "schema_version": 1,
        },
        "inventory-review.catalog.json": {
            "candidate_count": 1,
            "schema_version": 1,
            "unresolved_count": 0,
        },
        "models.catalog.json": {
            "catalog_id": "g00a-selftest-models",
            "lane_count": 12,
            "models": selftest_catalog_lanes,
            "schema_version": 1,
        },
    }
    for name, payload in catalog_payloads.items():
        write_selftest_json(root / name, payload)

    artifact_to_contract = {
        "generation-presets.catalog.json": "scripts/release/configs/runtime_vnext_generation_presets.json",
        "historical-bugs.catalog.json": "scripts/release/configs/runtime_vnext_historical_bugs.json",
        "inventory-review.catalog.json": "scripts/release/configs/runtime_vnext_inventory_review.json",
        "models.catalog.json": "scripts/release/configs/runtime_vnext_models.json",
    }
    contract_payload_by_path = {
        contract_path: (root / artifact_name).read_bytes()
        for artifact_name, contract_path in artifact_to_contract.items()
    }
    for contract_path in VNEXT_G00A_CONTRACT_PATHS - set(contract_payload_by_path):
        contract_payload_by_path[contract_path] = f"G00a selftest contract: {contract_path}\n".encode("utf-8")
    contract_rows = []
    for contract_path in sorted(VNEXT_G00A_CONTRACT_PATHS):
        payload = contract_payload_by_path[contract_path]
        contract_rows.append(
            {
                "git_blob": hashlib.sha1(payload).hexdigest(),
                "path": contract_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    contracts = {row["path"]: row for row in contract_rows}
    collector_sha = "1" * 40
    collector_tree = "2" * 40
    collector = {
        "contracts": contract_rows,
        "contracts_sha256": pretty_json_sha256(contract_rows),
        "dirty": False,
        "git_sha": collector_sha,
        "git_tree_sha": collector_tree,
        "status_short": [],
    }
    frozen = {
        "git_sha": VNEXT_FROZEN_LEGACY_SHA,
        "git_tree_sha": "3" * 40,
    }

    requests: dict[tuple[str, str], dict[str, Any]] = {}

    def model_source(lane_id: str, source_name: str) -> dict[str, Any]:
        seed = f"{lane_id}:{source_name}"
        repo = f"fixture/{lane_id.lower()}-{source_name.replace('_', '-')}"
        revision = hashlib.sha1(f"revision:{seed}".encode("utf-8")).hexdigest()
        file_path = (
            "model-00001-of-00001.safetensors"
            if lane_id == "m2-qwen35-35b-a3b-cuda" and source_name == "weight_source"
            else f"{source_name}.bin"
        )
        file_sha = hashlib.sha256(f"file:{seed}".encode("utf-8")).hexdigest()
        file_size = 1024 + len(seed)
        git_oid = hashlib.sha1(f"git-oid:{seed}".encode("utf-8")).hexdigest()
        file_row = {
            "git_oid": git_oid,
            "lfs_oid": file_sha,
            "path": file_path,
            "sha256": file_sha,
            "sha256_source": "hugging_face_lfs_oid",
            "size_bytes": file_size,
        }
        file_rows = [file_row]
        model_url = f"https://huggingface.co/api/models/{repo}/revision/{revision}"
        tree_url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
        requests[("model-info", model_url)] = selftest_g00a_response_request(
            "model-info",
            model_url,
            {"id": repo, "sha": revision},
        )
        tree_rows = [
            {
                "lfs": {"oid": f"sha256:{file_sha}", "size": file_size},
                "oid": git_oid,
                "path": file_path,
                "size": file_size,
                "type": "file",
            }
        ]
        if lane_id == "m2-qwen35-35b-a3b-cuda" and source_name == "weight_source":
            index_path = "model.safetensors.index.json"
            index_sha = hashlib.sha256(f"index:{seed}".encode("utf-8")).hexdigest()
            index_size = 4096
            index_git_oid = hashlib.sha1(f"index-git:{seed}".encode("utf-8")).hexdigest()
            index_url = f"https://huggingface.co/{repo}/resolve/{revision}/{index_path}"
            file_rows.append(
                {
                    "content_request_url": index_url,
                    "git_oid": index_git_oid,
                    "lfs_metadata_downloaded": True,
                    "lfs_oid": index_sha,
                    "path": index_path,
                    "sha256": index_sha,
                    "sha256_source": "hugging_face_lfs_oid",
                    "size_bytes": index_size,
                }
            )
            tree_rows.append(
                {
                    "lfs": {"oid": f"sha256:{index_sha}", "size": index_size},
                    "oid": index_git_oid,
                    "path": index_path,
                    "size": index_size,
                    "type": "file",
                }
            )
            requests[("metadata-file", index_url)] = {
                "kind": "metadata-file",
                "method": "GET",
                "response_bytes": index_size,
                "response_sha256": index_sha,
                "status": 200,
                "url": index_url,
            }
        requests[("repo-tree", tree_url)] = selftest_g00a_response_request(
            "repo-tree",
            tree_url,
            tree_rows,
        )
        return {
            "files": file_rows,
            "gated": False,
            "license": {"files": [], "hugging_face_id": "apache-2.0"},
            "model_request_url": model_url,
            "repo": repo,
            "requested_revision": {"status": "pinned", "value": revision},
            "revision": revision,
            "tree_request_urls": [tree_url],
        }

    def official_upstream_source(semantic: dict[str, Any]) -> dict[str, Any]:
        repo = "meta-llama/Llama-3.1-8B-Instruct"
        revision = "4" * 40
        model_url = f"https://huggingface.co/api/models/{repo}/revision/{revision}"
        tree_url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
        requests[("model-info", model_url)] = selftest_g00a_response_request(
            "model-info",
            model_url,
            {"id": repo, "sha": revision, "gated": "manual"},
        )
        requests[("repo-tree", tree_url)] = selftest_g00a_response_request(
            "repo-tree",
            tree_url,
            [
                {
                    "oid": "5" * 40,
                    "path": "config.json",
                    "size": 128,
                    "type": "file",
                }
            ],
        )
        return {
            "access_note": "selftest gated upstream",
            "gated": "manual",
            "mirror_blob_oid_matches": [
                {
                    "content_sha256": "6" * 64,
                    "git_oid": "5" * 40,
                    "path": "config.json",
                    "size_bytes": 128,
                }
            ],
            "mirror_repo": semantic["repo"],
            "mirror_revision": semantic["revision"],
            "model_request_url": model_url,
            "repo": repo,
            "revision": revision,
            "tree_request_urls": [tree_url],
            "verification_method": "mirror_content_sha256_and_official_git_blob_oid",
        }

    lanes: list[dict[str, Any]] = []
    for model_key, model_id in {**VNEXT_PRIMARY_MODELS, **VNEXT_SUPPLEMENTAL_MODELS}.items():
        for backend in ("cuda", "metal"):
            lane_id = f"{model_key}-{backend}"
            weight = model_source(lane_id, "weight_source")
            semantic = model_source(lane_id, "semantic_source")
            tokenizer = (
                model_source(lane_id, "tokenizer_source")
                if model_key == "llama31-8b-compat"
                else None
            )
            lanes.append(
                {
                    "backend": backend,
                    "catalog_lane_id": lane_id,
                    "chat_template": {
                        "content_sha256": hashlib.sha256(f"chat:{lane_id}".encode("utf-8")).hexdigest(),
                        "source": "tokenizer_source" if tokenizer is not None else "semantic_source",
                    },
                    "format": "safetensors" if backend == "cuda" else "gguf",
                    "generation_config": {
                        "policy": "fixture",
                        "sha256": hashlib.sha256(f"generation:{lane_id}".encode("utf-8")).hexdigest(),
                    },
                    "hardware_policy": f"{backend}-selftest",
                    "model_id": VNEXT_RESOLUTION_MODEL_IDS[model_key],
                    "official_upstream": (
                        official_upstream_source(semantic)
                        if model_key == "llama31-8b-compat"
                        else None
                    ),
                    "role": "primary" if model_key in VNEXT_PRIMARY_MODELS else "supplemental",
                    "semantic_source": semantic,
                    "tokenizer_source": tokenizer,
                    "weight_source": weight,
                }
            )
    lanes.sort(key=lambda row: row["catalog_lane_id"])
    require_selftest(len(lanes) == 12, "G00a selftest lane matrix must contain 12 lanes")

    models_catalog_sha = sha256(root / "models.catalog.json")
    resolver_identity = {
        "path": "scripts/release/runtime_vnext_model_resolver.py",
        "sha256": contracts["scripts/release/runtime_vnext_model_resolver.py"]["sha256"],
    }
    resolution_source = {
        "dirty": False,
        "git_sha": collector_sha,
        "status_short": [],
    }
    live_resolution = {
        "artifact_type": "runtime_vnext_model_resolution",
        "catalog_id": "g00a-selftest-models",
        "catalog_sha256": models_catalog_sha,
        "lanes": copy.deepcopy(lanes),
        "policy": {
            "lfs_metadata_download": {
                "allowed_suffixes": [".safetensors.index.json"],
                "max_bytes": 32 * 1024 * 1024,
                "selector_requirement": "weight_source_exact_path_required_if_sharded",
                "sha256_must_match_lfs_oid": True,
            },
            "raw_response_body_kinds": ["model-info", "repo-tree"],
            "transport": "network_huggingface_https",
        },
        "requests": [requests[key] for key in sorted(requests)],
        "resolver": resolver_identity,
        "schema_version": 1,
        "source": resolution_source,
    }
    input_resolution = copy.deepcopy(live_resolution)
    write_selftest_json(root / "model-resolution.input.json", input_resolution)
    write_selftest_json(root / "model-resolution.json", live_resolution)

    preset_names = {
        "P_DETERMINISTIC",
        "P_NO_THINKING",
        "P_OFFICIAL_DEFAULT",
        "P_THINKING",
    }
    preset_facts = {
        "models": {
            model_key: {
                "presets": {
                    name: {"seed": 9271, "source": "G00a outer selftest"}
                    for name in sorted(preset_names)
                }
            }
            for model_key in sorted(VNEXT_PRIMARY_MODELS)
        }
    }
    history_families = []
    for family_index in range(1, 16):
        case_count = 2 if family_index <= 13 else 1
        history_families.append(
            {
                "cases": [
                    {"evidence_status": "bound", "id": f"H{family_index:02d}.{case_index}"}
                    for case_index in range(1, case_count + 1)
                ],
                "id": f"H{family_index:02d}",
            }
        )
    require_selftest(
        sum(len(family["cases"]) for family in history_families) == 28,
        "G00a selftest history matrix must contain 28 cases",
    )
    inventory_document = {
        "analyzer": {
            "identity_key": "sha256",
            "path": "scripts/release/runtime_vnext_inventory.py",
        },
        "git": {
            "dirty": False,
            "sha": VNEXT_FROZEN_LEGACY_SHA,
            "status_short": [],
            "tree_sha": frozen["git_tree_sha"],
        },
        "root": "/fixture/frozen-cff4",
        "schema_version": 1,
        "summary": {"coupling_finding_count": 1, "file_count": 1},
    }
    write_selftest_json(root / "coupling-inventory.json", inventory_document)
    normalized_inventory = dict(inventory_document)
    normalized_inventory.pop("root")
    lock = {
        "artifact_type": "runtime_vnext_g00a_model_facts_lock",
        "checkpoint_id": "G00a",
        "collector": {
            "contracts_sha256": collector["contracts_sha256"],
            "git_sha": collector_sha,
            "git_tree_sha": collector_tree,
        },
        "frozen_legacy_source": frozen,
        "generation_presets": {
            "catalog_sha256": sha256(root / "generation-presets.catalog.json"),
            "facts": preset_facts,
        },
        "historical_bug_catalog": {
            "catalog_sha256": sha256(root / "historical-bugs.catalog.json"),
            "facts": {
                "catalog_scope": "catalog_only",
                "concrete_case_count": 28,
                "families": history_families,
                "family_count": 15,
                "full_historical_corpus_complete": False,
            },
        },
        "inventory": {
            "analyzer_contract": contracts["scripts/release/runtime_vnext_inventory.py"],
            "normalized_inventory_sha256": pretty_json_sha256(normalized_inventory),
            "review": {
                "sha256": sha256(root / "inventory-review.catalog.json"),
                "unresolved_count": 0,
            },
        },
        "model_catalog": {
            "catalog_id": "g00a-selftest-models",
            "catalog_sha256": models_catalog_sha,
            "lane_count": 12,
        },
        "model_resolution": {
            "artifact_sha256": sha256(root / "model-resolution.json"),
            "artifact_size_bytes": (root / "model-resolution.json").stat().st_size,
            "input_artifact_sha256": sha256(root / "model-resolution.input.json"),
            "input_artifact_size_bytes": (root / "model-resolution.input.json").stat().st_size,
            "live_facts_sha256": pretty_json_sha256(lanes),
            "live_recomputed": True,
            "resolver": resolver_identity,
            "source": resolution_source,
        },
        "models": copy.deepcopy(lanes),
        "schema_version": 1,
        "scope": {
            "does_not_prove": sorted(VNEXT_G00A_DOES_NOT_PROVE),
            "historical_evidence": "catalog_only",
            "unlocks": ["G01A"],
        },
    }
    write_selftest_json(root / "model-facts.lock.json", lock)

    manifest = {
        "artifact_count": 0,
        "artifact_dir": str(root),
        "artifact_index": [],
        "artifact_index_policy": {
            "manifest_self_digest": "excluded-to-avoid-recursive-digest",
            "non_manifest_artifacts_indexed": True,
        },
        "artifact_type": "runtime_vnext_g00a_fact_checkpoint_manifest",
        "canonical": True,
        "checkpoint_id": "G00a",
        "collector": collector,
        "dirty": False,
        "does_not_prove": sorted(VNEXT_G00A_DOES_NOT_PROVE),
        "fact_source_artifacts": {},
        "freshness": {
            "catalogs_match_collector_head": True,
            "collector_checkout_clean": True,
            "inventory_candidate_matches_current_analyzer_recomputation": True,
            "inventory_frozen_source_clean": True,
            "model_resolution_catalog_matches_current_head": True,
            "model_resolution_input_matches_live_facts": True,
            "model_resolution_live_recomputed": True,
            "model_resolution_resolver_matches_current_head": True,
            "model_resolution_source_matches_collector_head": True,
        },
        "frozen_source": frozen,
        "git_sha": collector_sha,
        "git_tree_sha": collector_tree,
        "lane": "runtime-vnext-g00a",
        "model_facts_lock": {},
        "pass_line": f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {root}",
        "schema_version": 1,
        "status": "pass",
        "unlocks": ["G01A"],
    }
    write_selftest_json(root / "manifest.json", manifest)
    refresh_selftest_g00a_manifest(root)
    return selftest_g00a_lane(root)


def expect_g00a_provenance_reject(
    valid_root: Path,
    name: str,
    mutate: Any,
    marker: str,
    *,
    refresh_after_mutation: bool = True,
) -> None:
    case = valid_root.parent / f"g00a-reject-{name}"
    shutil.copytree(valid_root, case)
    refresh_selftest_g00a_manifest(case)
    mutate(case)
    if refresh_after_mutation:
        refresh_selftest_g00a_manifest(case)
    manifest_path = case / "manifest.json"
    manifest = read_json_object(manifest_path, f"G00a selftest mutation {name}")
    lane = selftest_g00a_lane(case)
    digest = sha256(manifest_path)
    require_selftest(digest is not None, f"G00a selftest mutation {name} manifest digest missing")
    try:
        validate_vnext_g00a_provenance(
            lane,
            manifest,
            digest,
            verify_checkout=False,
        )
    except GateError as exc:
        require_selftest(marker.lower() in str(exc).lower(), f"G00a {name} rejected for unexpected reason: {exc}")
        return
    raise AssertionError(f"G00a provenance mutation {name} unexpectedly passed")


def mutate_g00a_live_lfs(root: Path) -> None:
    path = root / "model-resolution.json"
    doc = read_json_object(path, "G00a live LFS mutation")
    lane = doc["lanes"][0]
    source = lane["weight_source"]
    file_row = source["files"][0]
    replacement = "f" * 64
    file_row["sha256"] = replacement
    file_row["lfs_oid"] = replacement
    tree_url = source["tree_request_urls"][0]
    request = next(row for row in doc["requests"] if row["kind"] == "repo-tree" and row["url"] == tree_url)
    tree_body = decoded_request_body(request, "G00a live LFS mutation tree")
    tree_body[0]["lfs"]["oid"] = f"sha256:{replacement}"
    set_selftest_g00a_response_body(request, tree_body)
    write_selftest_json(path, doc)


def mutate_g00a_expected_sha_coherently(root: Path) -> None:
    lane_id = "m2-qwen35-35b-a3b-metal"
    replacement = "a" * 64
    resolution_documents: dict[str, dict[str, Any]] = {}
    for name in ("model-resolution.input.json", "model-resolution.json"):
        path = root / name
        doc = read_json_object(path, f"G00a coherent expected-SHA mutation {name}")
        lane = next(row for row in doc["lanes"] if row["catalog_lane_id"] == lane_id)
        source = lane["weight_source"]
        file_row = source["files"][0]
        file_row["sha256"] = replacement
        file_row["sha256_source"] = "hugging_face_lfs_oid"
        file_row["lfs_oid"] = replacement
        tree_url = source["tree_request_urls"][0]
        request = next(
            row
            for row in doc["requests"]
            if row["kind"] == "repo-tree" and row["url"] == tree_url
        )
        tree_body = decoded_request_body(request, f"G00a coherent expected-SHA mutation tree {name}")
        tree_body[0]["lfs"]["oid"] = f"sha256:{replacement}"
        set_selftest_g00a_response_body(request, tree_body)
        write_selftest_json(path, doc)
        resolution_documents[name] = doc

    lock_path = root / "model-facts.lock.json"
    lock = read_json_object(lock_path, "G00a coherent expected-SHA mutation lock")
    locked_lane = next(row for row in lock["models"] if row["catalog_lane_id"] == lane_id)
    locked_file = locked_lane["weight_source"]["files"][0]
    locked_file["sha256"] = replacement
    locked_file["sha256_source"] = "hugging_face_lfs_oid"
    locked_file["lfs_oid"] = replacement
    model_resolution = lock["model_resolution"]
    model_resolution["live_facts_sha256"] = pretty_json_sha256(lock["models"])
    model_resolution["artifact_sha256"] = sha256(root / "model-resolution.json")
    model_resolution["artifact_size_bytes"] = (root / "model-resolution.json").stat().st_size
    model_resolution["input_artifact_sha256"] = sha256(root / "model-resolution.input.json")
    model_resolution["input_artifact_size_bytes"] = (root / "model-resolution.input.json").stat().st_size
    require_selftest(
        resolution_documents["model-resolution.input.json"]["lanes"]
        == resolution_documents["model-resolution.json"]["lanes"]
        == lock["models"],
        "G00a coherent expected-SHA mutation did not keep input/live/lock facts aligned",
    )
    write_selftest_json(lock_path, lock)


def mutate_g00a_raw_tree_body(root: Path) -> None:
    path = root / "model-resolution.json"
    doc = read_json_object(path, "G00a raw tree mutation")
    request = next(row for row in doc["requests"] if row["kind"] == "repo-tree")
    tree_body = decoded_request_body(request, "G00a raw tree mutation body")
    tree_body[0]["lfs"]["oid"] = f"sha256:{'e' * 64}"
    set_selftest_g00a_response_body(request, tree_body)
    write_selftest_json(path, doc)


def mutate_g00a_missing_lfs_index_evidence(root: Path) -> None:
    path = root / "model-resolution.json"
    doc = read_json_object(path, "G00a missing LFS index evidence mutation")
    lane = next(
        row for row in doc["lanes"]
        if row["catalog_lane_id"] == "m2-qwen35-35b-a3b-cuda"
    )
    index_file = next(
        row for row in lane["weight_source"]["files"]
        if row["path"] == "model.safetensors.index.json"
    )
    index_file.pop("content_request_url")
    write_selftest_json(path, doc)


def mutate_g00a_extra_metadata_request(
    root: Path,
    *,
    oversized: bool = False,
    kind: str = "metadata-file",
) -> None:
    path = root / "model-resolution.json"
    doc = read_json_object(path, "G00a extra metadata request mutation")
    doc["requests"].append(
        {
            "kind": kind,
            "method": "GET",
            "response_bytes": 32 * 1024 * 1024 + 1 if oversized else 1024,
            "response_sha256": "9" * 64,
            "status": 200,
            "url": "https://huggingface.co/fixture/forged/resolve/" + "8" * 40 + "/model.safetensors",
        }
    )
    write_selftest_json(path, doc)


def mutate_g00a_extra_model_request(root: Path) -> None:
    path = root / "model-resolution.json"
    doc = read_json_object(path, "G00a extra model request mutation")
    url = "https://huggingface.co/api/models/fixture/unselected/revision/" + "8" * 40
    doc["requests"].append(
        selftest_g00a_response_request(
            "model-info",
            url,
            {"id": "fixture/unselected", "sha": "8" * 40},
        )
    )
    write_selftest_json(path, doc)


def mutate_g00a_lfs_tree_as_downloaded(root: Path) -> None:
    path = root / "model-resolution.json"
    doc = read_json_object(path, "G00a LFS tree misclassification mutation")
    lane = next(
        row for row in doc["lanes"]
        if row["catalog_lane_id"] == "m2-qwen35-35b-a3b-cuda"
    )
    index_file = next(
        row for row in lane["weight_source"]["files"]
        if row["path"] == "model.safetensors.index.json"
    )
    index_file["sha256_source"] = "downloaded_content"
    index_file.pop("lfs_oid")
    index_file.pop("lfs_metadata_downloaded")
    write_selftest_json(path, doc)


def mutate_g00a_official_tree_empty(root: Path) -> None:
    path = root / "model-resolution.json"
    doc = read_json_object(path, "G00a official tree mutation")
    official = next(
        lane["official_upstream"]
        for lane in doc["lanes"]
        if lane.get("official_upstream") is not None
    )
    tree_url = official["tree_request_urls"][0]
    request = next(
        row
        for row in doc["requests"]
        if row["kind"] == "repo-tree" and row["url"] == tree_url
    )
    set_selftest_g00a_response_body(request, [])
    write_selftest_json(path, doc)


def mutate_g00a_license_duplicate(root: Path) -> None:
    path = root / "model-resolution.json"
    doc = read_json_object(path, "G00a license duplicate mutation")
    lane = next(
        row for row in doc["lanes"]
        if row["catalog_lane_id"] == "m2-qwen35-35b-a3b-cuda"
    )
    index_file = next(
        row for row in lane["weight_source"]["files"]
        if row["path"] == "model.safetensors.index.json"
    )
    lane["weight_source"]["license"]["files"].append(copy.deepcopy(index_file))
    write_selftest_json(path, doc)


def mutate_g00a_requested_revision_status(root: Path) -> None:
    path = root / "model-resolution.json"
    doc = read_json_object(path, "G00a requested revision mutation")
    source = doc["lanes"][0]["weight_source"]
    old_url = source["model_request_url"]
    new_url = f"https://huggingface.co/api/models/{source['repo']}"
    source["requested_revision"] = {"status": "resolution_required", "value": None}
    source["model_request_url"] = new_url
    request = next(
        row
        for row in doc["requests"]
        if row["kind"] == "model-info" and row["url"] == old_url
    )
    request["url"] = new_url
    write_selftest_json(path, doc)


def make_selftest_release_summary_artifact(root: Path) -> None:
    for rel in [
        "source-unit/unit.gate.json",
        "source-metal/metal.gate.json",
        "source-cuda-full/g0_cuda4090_full.gate.json",
        "source-cuda-llama-dense/g0_cuda4090_llama_dense.gate.json",
        "metal-tarball/gate.json",
        "cuda-tarball/gate.json",
        "homebrew-metal/gate.json",
        "homebrew-cuda-fetch/gate.json",
    ]:
        write_selftest_json(root / rel, {"status": "pass"})


def make_selftest_completion_manifest(path: Path) -> None:
    root = path.parent
    artifacts = {}
    for name in [
        "metal_source_gate_artifact",
        "cuda_full_source_gate_artifact",
        "cuda_dense_source_gate_artifact",
        "metal_tarball_gate_artifact",
        "cuda_tarball_gate_artifact",
        "homebrew_metal_gate_artifact",
        "homebrew_cuda_fetch_gate_artifact",
    ]:
        artifact = root / "artifacts" / name
        artifact.mkdir(parents=True)
        artifacts[name] = str(artifact)
    write_selftest_json(
        path,
        {
            "git_sha": "selftest",
            "dirty_status": {"is_dirty": False, "status_short": []},
            "tag": "v0.0.0-selftest",
            "github_release_url": "https://example.invalid/selftest",
            "release_assets": [
                {
                    "name": "ferrum-selftest.tar.gz",
                    "sha256": "0" * 64,
                }
            ],
            "cargo_workspace_crates": [
                {
                    "name": "ferrum-cli",
                    "version": "0.0.0-selftest",
                    "crates_io_visible": True,
                }
            ],
            **artifacts,
        },
    )


def self_test() -> int:
    import runpy

    this_script = Path(__file__).resolve()
    source_gate_text = (REPO_ROOT / "scripts/release/g0_source_gate.sh").read_text(
        encoding="utf-8"
    )
    shell_command = (
        "    -- env PYTHONDONTWRITEBYTECODE=1 CARGO_BUILD_JOBS=2 "
        "RUST_TEST_THREADS=1 \\\n"
        "      cargo test --workspace --all-targets"
    )
    shell_expected_command = """expected_command = [
    "env",
    "PYTHONDONTWRITEBYTECODE=1",
    "CARGO_BUILD_JOBS=2",
    "RUST_TEST_THREADS=1",
    "cargo",
    "test",
    "--workspace",
    "--all-targets",
]"""
    require_selftest(
        source_gate_text.count(shell_command) == 1
        and source_gate_text.count(shell_expected_command) == 1
        and "-- --test-threads=1" not in source_gate_text,
        "g0 source shell bounded command policy drift",
    )
    g01a_checkpoint = runpy.run_path(
        str(REPO_ROOT / "scripts/release/runtime_vnext_g01a_checkpoint.py")
    )
    require_selftest(
        set(g01a_checkpoint["REQUIRED_CONTRACTS"])
        == VNEXT_G01A_REQUIRED_CONTRACTS,
        "vnext-g01a checkpoint/run_gate required contract matrix drift",
    )
    require_selftest(
        g01a_checkpoint["G01A_EXECUTION_IDENTITY_VERSION"]
        == VNEXT_G01A_EXECUTION_IDENTITY_VERSION
        and g01a_checkpoint["G01A_EVENT_REQUIRED_TRANSITION"]
        == VNEXT_G01A_EVENT_REQUIRED_TRANSITION
        and g01a_checkpoint["G01A_EVENT_FORBIDDEN_TRANSITION"]
        == VNEXT_G01A_EVENT_FORBIDDEN_TRANSITION
        and g01a_checkpoint["G01A_SEMANTIC_TYPE_KINDS"]
        == VNEXT_G01A_SEMANTIC_TYPE_KINDS
        and g01a_checkpoint["G01A_DNF_RETRY_AUTHORITY_TYPE"]
        == VNEXT_G01A_DNF_RETRY_AUTHORITY_TYPE
        and g01a_checkpoint["G01A_MULTIPARTICIPANT_DISPATCH_MARKERS"]
        == VNEXT_G01A_MULTIPARTICIPANT_DISPATCH_MARKERS,
        "vnext-g01a checkpoint/run_gate semantic contract policy drift",
    )

    semantic_summary = {
        "schema_version": 1,
        "execution_identity": {
            "constant": "EXECUTION_IDENTITY_VERSION",
            "major": 3,
            "minor": 0,
            "definition_count": 1,
        },
        "event_transition": {
            "enum": "ExecutionEventKind",
            "required_variant": "NodeRetired",
            "required_variant_count": 1,
            "forbidden_identifier": "NodeCompleted",
            "forbidden_identifier_count": 0,
        },
        "required_type_kinds": copy.deepcopy(VNEXT_G01A_SEMANTIC_TYPE_KINDS),
        "definition_counts": {
            name: 1 for name in VNEXT_G01A_SEMANTIC_TYPE_KINDS
        },
        "dnf_retry_authority": {
            "type_name": VNEXT_G01A_DNF_RETRY_AUTHORITY_TYPE,
            "field_sealed": True,
            "public_associated_constructor_count": 0,
        },
        "multi_participant_dispatch": {
            "owner": "OperationDispatch",
            "required_markers": copy.deepcopy(
                VNEXT_G01A_MULTIPARTICIPANT_DISPATCH_MARKERS
            ),
            "observed_markers": {
                name: True
                for name in VNEXT_G01A_MULTIPARTICIPANT_DISPATCH_MARKERS
            },
            "matching_public_method_count": 1,
        },
        "public_raw_dynamic_resource_shape": {
            "type_name": "DynamicResourceShape",
            "unrestricted_public_type_count": 0,
            "unrestricted_public_impl_method_count": 0,
            "unrestricted_public_parameter_path_count": 0,
        },
    }
    validate_vnext_g01a_semantic_summary(semantic_summary)

    def expect_semantic_summary_reject(
        name: str, mutated: dict[str, Any], marker: str
    ) -> None:
        try:
            validate_vnext_g01a_semantic_summary(mutated)
        except GateError as exc:
            require_selftest(
                marker.lower() in str(exc).lower(),
                f"vnext-g01a semantic mutation {name} rejected for unexpected reason: {exc}",
            )
            return
        raise AssertionError(
            f"vnext-g01a semantic mutation unexpectedly passed: {name}"
        )

    identity_v2 = copy.deepcopy(semantic_summary)
    identity_v2["execution_identity"]["major"] = 2
    expect_semantic_summary_reject("identity v2", identity_v2, "exactly 3.0")
    missing_retired = copy.deepcopy(semantic_summary)
    missing_retired["event_transition"]["required_variant_count"] = 0
    expect_semantic_summary_reject(
        "missing NodeRetired", missing_retired, "exactly one NodeRetired"
    )
    completed_present = copy.deepcopy(semantic_summary)
    completed_present["event_transition"]["forbidden_identifier_count"] = 1
    expect_semantic_summary_reject(
        "NodeCompleted present", completed_present, "zero NodeCompleted"
    )
    for type_name in VNEXT_G01A_SEMANTIC_TYPE_KINDS:
        missing_type = copy.deepcopy(semantic_summary)
        missing_type["definition_counts"][type_name] = 0
        expect_semantic_summary_reject(
            f"missing {type_name}", missing_type, "definition count matrix"
        )
        wrong_kind = copy.deepcopy(semantic_summary)
        wrong_kind["required_type_kinds"][type_name] = "enum"
        expect_semantic_summary_reject(
            f"wrong kind {type_name}", wrong_kind, "type/kind matrix"
        )
    wrong_retry_name = copy.deepcopy(semantic_summary)
    wrong_retry_name["dnf_retry_authority"]["type_name"] = "RetryAuthority"
    expect_semantic_summary_reject(
        "retry authority type name", wrong_retry_name, "not sealed"
    )
    public_retry_field = copy.deepcopy(semantic_summary)
    public_retry_field["dnf_retry_authority"]["field_sealed"] = False
    expect_semantic_summary_reject(
        "public retry field", public_retry_field, "not sealed"
    )
    public_retry_constructor = copy.deepcopy(semantic_summary)
    public_retry_constructor["dnf_retry_authority"][
        "public_associated_constructor_count"
    ] = 1
    expect_semantic_summary_reject(
        "public retry constructor", public_retry_constructor, "not sealed"
    )
    for marker_name in VNEXT_G01A_MULTIPARTICIPANT_DISPATCH_MARKERS:
        missing_marker = copy.deepcopy(semantic_summary)
        missing_marker["multi_participant_dispatch"]["observed_markers"][
            marker_name
        ] = False
        expect_semantic_summary_reject(
            f"dispatch marker {marker_name}", missing_marker, "marker is missing"
        )
    duplicate_dispatch = copy.deepcopy(semantic_summary)
    duplicate_dispatch["multi_participant_dispatch"][
        "matching_public_method_count"
    ] = 2
    expect_semantic_summary_reject(
        "duplicate dispatch", duplicate_dispatch, "exactly one public"
    )
    for count_name in (
        "unrestricted_public_type_count",
        "unrestricted_public_impl_method_count",
        "unrestricted_public_parameter_path_count",
    ):
        public_raw_shape = copy.deepcopy(semantic_summary)
        public_raw_shape["public_raw_dynamic_resource_shape"][count_name] = 1
        expect_semantic_summary_reject(
            f"public raw shape {count_name}", public_raw_shape, count_name
        )
    checkpoint_test_targets = g01a_checkpoint["REQUIRED_TESTS_BY_TARGET"]
    require_selftest(
        checkpoint_test_targets.keys()
        == VNEXT_G01A_REQUIRED_TESTS_BY_TARGET.keys()
        and all(
            checkpoint_test_targets[target]
            == VNEXT_G01A_REQUIRED_TESTS_BY_TARGET[target]
            for target in checkpoint_test_targets
        ),
        "vnext-g01a checkpoint/run_gate exact test matrix drift",
    )
    require_selftest(
        g01a_checkpoint["REQUIRED_ADMISSION_LIB_TESTS"]
        == VNEXT_G01A_REQUIRED_ADMISSION_LIB_TESTS,
        "vnext-g01a checkpoint/run_gate admission lib test matrix drift",
    )
    require_selftest(
        g01a_checkpoint["QUALITY_COMMANDS"] == VNEXT_G01A_QUALITY_COMMANDS
        and g01a_checkpoint["REQUIRED_CORE_TESTS_BY_TARGET"]
        == VNEXT_G01A_REQUIRED_CORE_TESTS_BY_TARGET
        and g01a_checkpoint["REQUIRED_RESOURCE_TESTS_BY_TARGET"]
        == VNEXT_G01A_REQUIRED_RESOURCE_TESTS_BY_TARGET
        and g01a_checkpoint["RESOURCE_PROOF_LINES"]
        == VNEXT_G01A_RESOURCE_PROOF_LINES
        and g01a_checkpoint["REQUIRED_EVENT_TESTS_BY_TARGET"]
        == VNEXT_G01A_REQUIRED_EVENT_TESTS_BY_TARGET
        and g01a_checkpoint["EVENT_PROOF_LINES"]
        == VNEXT_G01A_EVENT_PROOF_LINES
        and g01a_checkpoint["REQUIRED_DEVICE_OPERATION_TESTS_BY_TARGET"]
        == VNEXT_G01A_REQUIRED_DEVICE_OPERATION_TESTS_BY_TARGET
        and g01a_checkpoint["DEVICE_OPERATION_PROOF_LINES"]
        == VNEXT_G01A_DEVICE_OPERATION_PROOF_LINES
        and len(g01a_checkpoint["evidence_command_matrix"]())
        == len(VNEXT_G01A_QUALITY_COMMANDS) + VNEXT_G01A_BOUNDED_TEST_COMMAND_COUNT,
        "vnext-g01a checkpoint/run_gate command matrix drift",
    )
    checkpoint_commands = g01a_checkpoint["evidence_command_matrix"]()
    checkpoint_cargo_tests = [
        command
        for command in checkpoint_commands
        if tuple(command[:2]) == ("cargo", "test")
    ]
    require_selftest(
        g01a_checkpoint["TEST_THREADS_ARG"] == VNEXT_G01A_TEST_THREADS_ARG
        and g01a_checkpoint["BOUNDED_RECEIPT_SCHEMA"]
        == VNEXT_G01A_BOUNDED_RECEIPT_SCHEMA
        and g01a_checkpoint["BOUNDED_TEST_COMMAND_COUNT"]
        == VNEXT_G01A_BOUNDED_TEST_COMMAND_COUNT
        and g01a_checkpoint["BOUNDED_TEST_ENV_OVERRIDES"]
        == VNEXT_G01A_BOUNDED_TEST_ENV_OVERRIDES
        and g01a_checkpoint["BOUNDED_TEST_PROFILES"]
        == VNEXT_G01A_BOUNDED_TEST_PROFILES
        and len(checkpoint_cargo_tests) == VNEXT_G01A_BOUNDED_TEST_COMMAND_COUNT
        and all(
            command.count(VNEXT_G01A_TEST_THREADS_ARG) == 1
            for command in checkpoint_cargo_tests
        ),
        "vnext-g01a checkpoint/run_gate bounded command policy drift",
    )
    require_selftest(
        all(
            vnext_g01a_bounded_profile(
                tuple(g01a_checkpoint["admission_test_command"](mode))
            )
            == "admission"
            for mode in ("--list", "--nocapture")
        ),
        "vnext-g01a outer validator must classify admission cold-compile commands explicitly",
    )
    require_selftest(
        vnext_g01a_expected_test_summaries(
            VNEXT_G01A_RESOURCE_PANIC_ISOLATION_TARGET
        )
        == [("1", "0", "0", "1"), ("2", "0", "0", "0")]
        and all(
            vnext_g01a_expected_test_summaries(target)
            == [("1", "0", "0", "0")]
            for target in VNEXT_G01A_REQUIRED_EVENT_TESTS_BY_TARGET
        ),
        "vnext-g01a outer validator test summary policy drift",
    )
    for checkpoint_name, outer_value in (
        ("EXPECTED_RESOURCE_CASES", VNEXT_G01A_EXPECTED_RESOURCE_CASES),
        ("EXPECTED_FAIL_CLOSED_CASES", VNEXT_G01A_EXPECTED_FAIL_CLOSED_CASES),
        ("EXPECTED_MODEL_IDENTITY_CASES", VNEXT_G01A_EXPECTED_MODEL_IDENTITY_CASES),
        (
            "EXPECTED_EVENT_REPLAY_V5_CASES",
            VNEXT_G01A_EXPECTED_EVENT_REPLAY_V5_CASES,
        ),
        (
            "EXPECTED_DEVICE_OPERATION_CASES",
            VNEXT_G01A_EXPECTED_DEVICE_OPERATION_CASES,
        ),
        ("EXPECTED_ORACLE_CASES", VNEXT_G01A_EXPECTED_ORACLE_CASES),
        ("EXPECTED_MODEL_WIRE_CASES", VNEXT_G01A_EXPECTED_MODEL_WIRE_CASES),
        (
            "EXPECTED_DYNAMIC_ADMISSION_CASES",
            VNEXT_G01A_EXPECTED_DYNAMIC_ADMISSION_CASES,
        ),
        (
            "EXPECTED_TRYBUILD_PASS_CASES",
            VNEXT_G01A_EXPECTED_TRYBUILD_PASS_CASES,
        ),
        (
            "EXPECTED_TRYBUILD_FAIL_CASES",
            VNEXT_G01A_EXPECTED_TRYBUILD_FAIL_CASES,
        ),
    ):
        require_selftest(
            g01a_checkpoint[checkpoint_name] == outer_value,
            f"vnext-g01a checkpoint/run_gate constant drift: {checkpoint_name}",
        )

    bounded_selftest_row = g01a_checkpoint["selftest_bounded_row"](
        list(
            g01a_checkpoint["test_command"](
                VNEXT_G01A_RESOURCE_PANIC_ISOLATION_TARGET, "--nocapture"
            )
        )
    )

    def validate_bounded_selftest_row(row: dict[str, Any]) -> None:
        command = tuple(row["command"])
        validate_vnext_g01a_command_execution(
            row,
            command,
            row["stdout"],
            row["stderr"],
            0,
        )

    def expect_bounded_selftest_reject(
        name: str, row: dict[str, Any], marker: str
    ) -> None:
        try:
            validate_bounded_selftest_row(row)
        except GateError as exc:
            require_selftest(
                marker.lower() in str(exc).lower(),
                f"{name} rejected for unexpected reason: {exc}",
            )
            return
        raise AssertionError(f"vnext-g01a bounded mutation unexpectedly passed: {name}")

    validate_bounded_selftest_row(bounded_selftest_row)
    missing_receipt = copy.deepcopy(bounded_selftest_row)
    missing_receipt["execution"]["receipt"] = None
    missing_receipt["execution"]["receipt_sha256"] = None
    expect_bounded_selftest_reject(
        "missing receipt", missing_receipt, "receipt must be a JSON object"
    )

    cleanup_false = copy.deepcopy(bounded_selftest_row)
    cleanup_false["execution"]["receipt"]["cleanup"]["process_group_gone"] = False
    cleanup_false["execution"]["receipt_sha256"] = canonical_json_sha256(
        cleanup_false["execution"]["receipt"]
    )
    expect_bounded_selftest_reject(
        "cleanup false", cleanup_false, "process group cleanup failed"
    )

    peak_exceeded = copy.deepcopy(bounded_selftest_row)
    peak_exceeded["execution"]["receipt"]["peaks"]["group_threads"] = (
        VNEXT_G01A_BOUNDED_TEST_PROFILES["resource"]["max_group_threads"] + 1
    )
    peak_exceeded["execution"]["receipt_sha256"] = canonical_json_sha256(
        peak_exceeded["execution"]["receipt"]
    )
    expect_bounded_selftest_reject(
        "peak exceeded", peak_exceeded, "peak exceeds"
    )

    sampling_error = copy.deepcopy(bounded_selftest_row)
    sampling_error["execution"]["receipt"]["sampling_error_count"] = 1
    sampling_error["execution"]["receipt"]["sampling_errors"] = [
        {"at": "2026-07-11T00:00:00.050Z", "type": "Synthetic", "error": "sample"}
    ]
    sampling_error["execution"]["receipt_sha256"] = canonical_json_sha256(
        sampling_error["execution"]["receipt"]
    )
    expect_bounded_selftest_reject(
        "sampling error", sampling_error, "contains sampling errors"
    )

    command_mismatch = copy.deepcopy(bounded_selftest_row)
    command_mismatch["execution"]["receipt"]["command"].append("--unexpected")
    command_mismatch["execution"]["receipt_sha256"] = canonical_json_sha256(
        command_mismatch["execution"]["receipt"]
    )
    expect_bounded_selftest_reject(
        "command mismatch", command_mismatch, "receipt command mismatch"
    )

    missing_threads = copy.deepcopy(bounded_selftest_row)
    missing_threads["command"].remove(VNEXT_G01A_TEST_THREADS_ARG)
    missing_threads["execution"]["receipt"]["command"].remove(
        VNEXT_G01A_TEST_THREADS_ARG
    )
    missing_threads["execution"]["receipt_sha256"] = canonical_json_sha256(
        missing_threads["execution"]["receipt"]
    )
    expect_bounded_selftest_reject(
        "missing test threads",
        missing_threads,
        "must contain exactly one --test-threads=1",
    )
    require_selftest(
        validate_safetensors_shard_paths(
            {
                "model-00001-of-000002.safetensors",
                "model-00002-of-000002.safetensors",
            },
            "valid shard fixture",
        ),
        "valid 5/6-width shard fixture was not detected as sharded",
    )
    for paths, needle in (
        (
            {"model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors"},
            "shard count differs",
        ),
        (
            {"model-a.safetensors", "model-b.safetensors"},
            "lacks canonical numbering",
        ),
    ):
        try:
            validate_safetensors_shard_paths(paths, "invalid shard fixture")
        except GateError as exc:
            require_selftest(needle in str(exc), f"invalid shard fixture rejected unexpectedly: {exc}")
        else:
            raise AssertionError(f"invalid shard fixture unexpectedly passed: {sorted(paths)}")
    try:
        validate_catalog_weight_paths(
            {
                "files": [
                    {"glob": "*", "required": True},
                    {
                        "path": "model.safetensors.index.json",
                        "required_if_sharded": True,
                    },
                ]
            },
            {"model.safetensors", "model.safetensors.index.json"},
            "unsharded conditional fixture",
        )
    except GateError as exc:
        require_selftest(
            "unsharded model contains a conditional weight file" in str(exc),
            f"unsharded conditional fixture rejected unexpectedly: {exc}",
        )
    else:
        raise AssertionError("unsharded conditional index unexpectedly passed")
    with tempfile.TemporaryDirectory(prefix="ferrum-run-gate-selftest-") as tmp:
        root = Path(tmp)

        listed = run_selftest_command([sys.executable, str(this_script), "--list-lanes"])
        require_selftest(listed.returncode == 0, listed.stderr or listed.stdout)
        require_selftest(listed.stdout.splitlines() == list(LANES), listed.stdout)

        dry_out = root / "unit-dry-run"
        dry = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "unit",
                "--out",
                str(dry_out),
                "--dry-run",
            ]
        )
        require_selftest(dry.returncode == 0, dry.stderr or dry.stdout)
        dry_manifest = json.loads((dry_out / "gate.manifest.json").read_text())
        require_selftest(dry_manifest["status"] == "dry-run", dry_manifest)
        require_selftest(dry_manifest["lane"] == "unit", dry_manifest)
        require_selftest(
            dry_manifest["delegated_command_line"][0] == "scripts/release/g0_source_gate.sh",
            dry_manifest,
        )
        require_selftest(
            dry_manifest["child_pass_line"] == source_pass_line("unit", dry_out),
            dry_manifest,
        )
        unit_root = root / "unit-bounded-provenance"
        bounded_root = unit_root / "unit-bounded"
        bounded_root.mkdir(parents=True)
        unit_stdout = bounded_root / "stdout.log"
        unit_stderr = bounded_root / "stderr.log"
        unit_receipt_path = bounded_root / "receipt.json"
        unit_stdout.write_text(
            "\n".join(f"Testing {case}\nSuccess" for case in G0_UNIT_BENCH_CASES)
            + "\n",
            encoding="utf-8",
        )
        unit_stderr.write_text(
            "     Running benches/engine_bench.rs (target/debug/deps/engine_bench-selftest)\n",
            encoding="utf-8",
        )
        unit_receipt = {
            "schema": VNEXT_G01A_BOUNDED_RECEIPT_SCHEMA,
            "command": copy.deepcopy(G0_UNIT_BOUNDED_COMMAND),
            "cwd": str(REPO_ROOT),
            "pid": 5252,
            "pgid": 5252,
            "limits": copy.deepcopy(G0_UNIT_BOUNDED_LIMITS),
            "peaks": {
                "processes": 1,
                "group_threads": 2,
                "per_process_threads": 2,
                "per_process_threads_pid": 5252,
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
                "path": str(unit_stdout.resolve()),
                "sha256": hashlib.sha256(unit_stdout.read_bytes()).hexdigest(),
                "size_bytes": unit_stdout.stat().st_size,
            },
            "stderr": {
                "path": str(unit_stderr.resolve()),
                "sha256": hashlib.sha256(unit_stderr.read_bytes()).hexdigest(),
                "size_bytes": unit_stderr.stat().st_size,
            },
        }
        write_selftest_json(unit_receipt_path, unit_receipt)

        def unit_ref(path: Path) -> dict[str, Any]:
            return {
                "path": path.relative_to(unit_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }

        unit_pass_line = f"G0 SOURCE unit PASS: {unit_root}"
        unit_manifest_path = unit_root / "unit.gate.json"
        unit_manifest = {
            "schema_version": 1,
            "artifact_type": "g0_source_unit_bounded_gate",
            "status": "pass",
            "lane": "unit",
            "pass_line": unit_pass_line,
            "command": copy.deepcopy(G0_UNIT_BOUNDED_COMMAND),
            "env_overrides": copy.deepcopy(G0_UNIT_BOUNDED_ENV_OVERRIDES),
            "receipt_schema": VNEXT_G01A_BOUNDED_RECEIPT_SCHEMA,
            "limits": copy.deepcopy(G0_UNIT_BOUNDED_LIMITS),
            "peaks": copy.deepcopy(unit_receipt["peaks"]),
            "cleanup": {"process_group_gone": True},
            "bounded_receipt": unit_ref(unit_receipt_path),
            "stdout_log": unit_ref(unit_stdout),
            "stderr_log": unit_ref(unit_stderr),
        }
        write_selftest_json(unit_manifest_path, unit_manifest)
        unit_lane = LaneCommand(
            cmd=["scripts/release/g0_source_gate.sh", "unit", str(unit_root)],
            expected_child_pass_line=unit_pass_line,
            child_manifest_path=unit_manifest_path,
            provenance_kind="g0-source-unit",
        )
        unit_provenance = verify_child_pass_line(
            unit_lane,
            unit_pass_line + "\n",
            verify_checkout=False,
        )
        require_selftest(
            unit_provenance is not None
            and unit_provenance.get("kind") == "g0-source-unit"
            and unit_provenance.get("cleanup") == {"process_group_gone": True}
            and unit_provenance.get("env_overrides")
            == G0_UNIT_BOUNDED_ENV_OVERRIDES
            and unit_provenance.get("executed_bench_targets") == ["engine_bench"]
            and unit_provenance.get("executed_bench_case_count")
            == len(G0_UNIT_BENCH_CASES),
            "g0 source unit bounded provenance self-test mismatch",
        )

        def expect_unit_manifest_reject(
            name: str, manifest: dict[str, Any], marker: str
        ) -> None:
            try:
                validate_g0_source_unit_provenance(
                    unit_lane, manifest, hashlib.sha256(name.encode()).hexdigest()
                )
            except GateError as exc:
                require_selftest(
                    marker.lower() in str(exc).lower(),
                    f"{name} rejected for unexpected reason: {exc}",
                )
                return
            raise AssertionError(f"g0 unit manifest mutation unexpectedly passed: {name}")

        missing_thread_env = copy.deepcopy(unit_manifest)
        del missing_thread_env["env_overrides"]["RUST_TEST_THREADS"]
        expect_unit_manifest_reject(
            "missing RUST_TEST_THREADS",
            missing_thread_env,
            "bounded command metadata mismatch",
        )
        wrong_thread_env = copy.deepcopy(unit_manifest)
        wrong_thread_env["env_overrides"]["RUST_TEST_THREADS"] = "2"
        expect_unit_manifest_reject(
            "wrong RUST_TEST_THREADS",
            wrong_thread_env,
            "bounded command metadata mismatch",
        )
        legacy_trailing_command = copy.deepcopy(unit_manifest)
        legacy_trailing_command["command"] = [
            "cargo",
            "test",
            "--workspace",
            "--all-targets",
            "--",
            "--test-threads=1",
        ]
        expect_unit_manifest_reject(
            "legacy trailing libtest argument",
            legacy_trailing_command,
            "bounded command metadata mismatch",
        )

        missing_receipt_env = copy.deepcopy(unit_receipt)
        missing_receipt_env["command"].remove("RUST_TEST_THREADS=1")
        try:
            validate_bounded_command_receipt(
                missing_receipt_env,
                expected_command=G0_UNIT_BOUNDED_COMMAND,
                expected_cwd=REPO_ROOT,
                expected_limits=G0_UNIT_BOUNDED_LIMITS,
                stdout_bytes=unit_stdout.read_bytes(),
                stderr_bytes=unit_stderr.read_bytes(),
                label="g0 source unit missing receipt env selftest",
            )
        except GateError as exc:
            require_selftest(
                "command mismatch" in str(exc).lower(),
                f"missing receipt env rejected unexpectedly: {exc}",
            )
        else:
            raise AssertionError("g0 unit receipt without thread env unexpectedly passed")

        missing_bench_stdout = unit_stdout.read_text(encoding="utf-8").replace(
            f"Testing {G0_UNIT_BENCH_CASES[-1]}\nSuccess\n", ""
        )
        try:
            validate_g0_unit_bench_witnesses(
                missing_bench_stdout, unit_stderr.read_text(encoding="utf-8")
            )
        except GateError as exc:
            require_selftest(
                G0_UNIT_BENCH_CASES[-1] in str(exc),
                f"missing bench witness rejected unexpectedly: {exc}",
            )
        else:
            raise AssertionError("g0 unit missing bench witness unexpectedly passed")
        in_tree_vnext_out = REPO_ROOT / (
            f".run-gate-vnext-g00-selftest-{os.getpid()}-{time.monotonic_ns()}"
        )
        in_tree_vnext = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "vnext-g00",
                "--out",
                str(in_tree_vnext_out),
                "--dry-run",
            ]
        )
        require_selftest(in_tree_vnext.returncode != 0, "in-tree vnext-g00 --out unexpectedly passed")
        require_selftest(
            "must resolve outside the Git source tree" in in_tree_vnext.stderr,
            in_tree_vnext.stderr or in_tree_vnext.stdout,
        )
        require_selftest(
            not in_tree_vnext_out.exists(),
            "rejected in-tree vnext-g00 --out created an artifact directory",
        )
        vnext_out = (root / "vnext-g00-dry-run").resolve()
        vnext = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "vnext-g00",
                "--out",
                str(vnext_out),
                "--dry-run",
            ]
        )
        require_selftest(vnext.returncode == 0, vnext.stderr or vnext.stdout)
        vnext_manifest = json.loads((vnext_out / "gate.manifest.json").read_text())
        require_selftest(vnext_manifest["status"] == "dry-run", vnext_manifest)
        require_selftest(vnext_manifest["lane"] == "vnext-g00", vnext_manifest)
        require_selftest(
            vnext_manifest["delegated_command_line"][1]
            == "scripts/release/runtime_vnext_baseline_gate.py",
            vnext_manifest,
        )
        require_selftest(
            "--require-full-self-test" in vnext_manifest["delegated_command_line"],
            vnext_manifest,
        )
        require_selftest(
            vnext_manifest["child_pass_line"]
            == f"FERRUM RUNTIME VNEXT G00 BASELINE PASS: {vnext_out}",
            vnext_manifest,
        )

        g00a_out = (root / "vnext-g00a-dry-run").resolve()
        g00a_inventory = root / "g00a-coupling-inventory.json"
        g00a_resolution = root / "g00a-model-resolution.json"
        g00a = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "vnext-g00a",
                "--coupling-inventory",
                str(g00a_inventory),
                "--model-resolution",
                str(g00a_resolution),
                "--out",
                str(g00a_out),
                "--dry-run",
            ]
        )
        require_selftest(g00a.returncode == 0, g00a.stderr or g00a.stdout)
        g00a_manifest = json.loads((g00a_out / "gate.manifest.json").read_text())
        require_selftest(g00a_manifest["status"] == "dry-run" and g00a_manifest["lane"] == "vnext-g00a", g00a_manifest)
        require_selftest(
            g00a_manifest["delegated_command_line"][1]
            == "scripts/release/runtime_vnext_g00a_checkpoint.py",
            g00a_manifest,
        )
        require_selftest(
            g00a_manifest["child_pass_line"]
            == f"FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: {g00a_out}",
            g00a_manifest,
        )

        g00f_out = (root / "vnext-g00f-dry-run").resolve()
        g00f = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "vnext-g00f",
                "--g00a",
                str(g00a_out / "gate.manifest.json"),
                "--out",
                str(g00f_out),
                "--dry-run",
            ]
        )
        require_selftest(g00f.returncode == 0, g00f.stderr or g00f.stdout)
        g00f_manifest = json.loads((g00f_out / "gate.manifest.json").read_text())
        require_selftest(
            g00f_manifest["status"] == "dry-run"
            and g00f_manifest["lane"] == "vnext-g00f"
            and g00f_manifest["delegated_command_line"][1]
            == "scripts/release/runtime_vnext_g00f_checkpoint.py"
            and g00f_manifest["child_pass_line"]
            == f"FERRUM RUNTIME VNEXT G00F FACTS PASS: {g00f_out}",
            g00f_manifest,
        )

        g01a_out = (root / "vnext-g01a-dry-run").resolve()
        g01a = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "vnext-g01a",
                "--g00f",
                str(g00f_out / "gate.manifest.json"),
                "--out",
                str(g01a_out),
                "--dry-run",
            ]
        )
        require_selftest(g01a.returncode == 0, g01a.stderr or g01a.stdout)
        g01a_manifest = json.loads((g01a_out / "gate.manifest.json").read_text())
        require_selftest(
            g01a_manifest["status"] == "dry-run"
            and g01a_manifest["lane"] == "vnext-g01a",
            g01a_manifest,
        )
        require_selftest(
            g01a_manifest["delegated_command_line"][1]
            == "scripts/release/runtime_vnext_s0a_contract_split.py"
            and g01a_manifest["delegated_command_line"][2:4]
            == ["--g00f", str(g00f_out / "gate.manifest.json")],
            g01a_manifest,
        )
        require_selftest(
            g01a_manifest["child_pass_line"]
            == f"FERRUM RUNTIME VNEXT G01A CONTRACT SPLIT PASS: {g01a_out}",
            g01a_manifest,
        )
        in_tree_g01a_out = REPO_ROOT / (
            f".run-gate-vnext-g01a-selftest-{os.getpid()}-{time.monotonic_ns()}"
        )
        in_tree_g01a = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "vnext-g01a",
                "--g00f",
                str(g00f_out / "gate.manifest.json"),
                "--out",
                str(in_tree_g01a_out),
                "--dry-run",
            ]
        )
        require_selftest(
            in_tree_g01a.returncode != 0,
            "in-tree vnext-g01a --out unexpectedly passed",
        )
        require_selftest(
            "must resolve outside the Git source tree" in in_tree_g01a.stderr
            and not in_tree_g01a_out.exists(),
            in_tree_g01a.stderr or in_tree_g01a.stdout,
        )

        g00a_provenance_root = root / "vnext-g00a-provenance"
        g00a_lane = make_selftest_vnext_g00a_artifact(g00a_provenance_root)
        g00a_child_manifest = read_json_object(
            g00a_provenance_root / "manifest.json",
            "G00a selftest child manifest",
        )
        g00a_manifest_digest = sha256(g00a_provenance_root / "manifest.json")
        require_selftest(g00a_manifest_digest is not None, "G00a selftest manifest digest missing")
        g00a_provenance = validate_vnext_g00a_provenance(
            g00a_lane,
            g00a_child_manifest,
            g00a_manifest_digest,
            verify_checkout=False,
        )
        require_selftest(g00a_provenance["kind"] == "vnext-g00a", str(g00a_provenance))
        require_selftest(g00a_provenance["model_lane_count"] == 12, str(g00a_provenance))
        require_selftest(
            g00a_provenance["catalog_expected_weight_identity_count"] == 1,
            str(g00a_provenance),
        )
        require_selftest(
            g00a_provenance["historical_bug_counts"] == {"families": 15, "cases": 28},
            str(g00a_provenance),
        )
        require_selftest(
            g00a_provenance["child_manifest"]["sha256"] == g00a_manifest_digest,
            str(g00a_provenance),
        )

        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "input-lfs",
            lambda case: mutate_selftest_json(
                case / "model-resolution.input.json",
                lambda data: data["lanes"][0]["weight_source"]["files"][0].update(
                    {"lfs_oid": "d" * 64, "sha256": "d" * 64}
                ),
            ),
            "input/live source drift",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "live-lfs",
            mutate_g00a_live_lfs,
            "source drift",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "catalog-expected-sha-coherent-forgery",
            mutate_g00a_expected_sha_coherently,
            "catalog expected sha256 mismatch",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "raw-tree-body",
            mutate_g00a_raw_tree_body,
            "tree lfs sha256 mismatch",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "missing-lfs-index-evidence",
            mutate_g00a_missing_lfs_index_evidence,
            "lfs index url",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "extra-metadata-request",
            lambda case: mutate_g00a_extra_metadata_request(case),
            "metadata request provenance differs",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "oversized-metadata-request",
            lambda case: mutate_g00a_extra_metadata_request(case, oversized=True),
            "metadata response exceeds download limit",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "unknown-request-kind",
            lambda case: mutate_g00a_extra_metadata_request(case, kind="weight-file"),
            "request kind mismatch",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "extra-model-request",
            mutate_g00a_extra_model_request,
            "network request provenance differs from selected sources and files",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "lfs-tree-as-downloaded",
            mutate_g00a_lfs_tree_as_downloaded,
            "downloaded metadata is LFS-backed in the authoritative tree",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "official-tree-empty",
            mutate_g00a_official_tree_empty,
            "official_upstream tree fact",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "license-duplicate",
            mutate_g00a_license_duplicate,
            "files and license.files contain duplicate paths",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "requested-revision-status",
            mutate_g00a_requested_revision_status,
            "requested_revision differs from catalog",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "lock-live-facts",
            lambda case: mutate_selftest_json(
                case / "model-facts.lock.json",
                lambda data: data["model_resolution"].update(
                    {"live_facts_sha256": "c" * 64}
                ),
            ),
            "live model facts digest mismatch",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "copied-catalog",
            lambda case: mutate_selftest_json(
                case / "generation-presets.catalog.json",
                lambda data: data.update({"tampered": True}),
            ),
            "copied catalog differs from collector contract",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "inventory-normalized-digest",
            lambda case: mutate_selftest_json(
                case / "model-facts.lock.json",
                lambda data: data["inventory"].update(
                    {"normalized_inventory_sha256": "b" * 64}
                ),
            ),
            "normalized inventory digest mismatch",
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "manifest-scope",
            lambda case: mutate_selftest_json(
                case / "manifest.json",
                lambda data: data.update({"unlocks": ["G01B"]}),
            ),
            "unlocks",
            refresh_after_mutation=False,
        )
        expect_g00a_provenance_reject(
            g00a_provenance_root,
            "manifest-index",
            lambda case: mutate_selftest_json(
                case / "manifest.json",
                lambda data: data["artifact_index"][0].update(
                    {"sha256": "a" * 64}
                ),
            ),
            "artifact_index",
            refresh_after_mutation=False,
        )

        vnext_provenance_root = root / "vnext-g00-provenance"
        vnext_lane = make_selftest_vnext_g00_artifact(vnext_provenance_root)
        provenance = verify_child_pass_line(
            vnext_lane,
            selftest_vnext_stdout(vnext_lane),
            verify_checkout=False,
        )
        require_selftest(provenance is not None and provenance["kind"] == "vnext-g00", str(provenance))
        require_selftest(
            provenance["child_manifest"]["sha256"] == sha256(vnext_provenance_root / "manifest.json"),
            str(provenance),
        )
        require_selftest(len(provenance["model_identities"]) == 6, str(provenance))
        require_selftest(len(provenance["legacy_binaries"]["identities"]) == 2, str(provenance))
        require_selftest(provenance["model_resolution"]["lane_count"] == 12, str(provenance))
        require_selftest(len(provenance["config_artifacts"]) >= 7, str(provenance))
        require_selftest(
            provenance["full_redteam"]["summary"]["mutation_assertion_count"]
            == VNEXT_G00_REDTEAM_MUTATION_COUNT,
            str(provenance),
        )
        valid_vnext_stdout = selftest_vnext_stdout(vnext_lane)
        invalid_redteam_outputs = [
            (
                "missing-full-line",
                valid_vnext_stdout.replace(VNEXT_G00_FULL_SELFTEST_PASS + "\n", ""),
                "exact FULL self-test PASS line",
            ),
            (
                "forged-full-line",
                valid_vnext_stdout.replace(
                    VNEXT_G00_FULL_SELFTEST_PASS,
                    VNEXT_G00_FULL_SELFTEST_PASS + " FORGED",
                ),
                "exact FULL self-test PASS line",
            ),
            (
                "missing-summary",
                "\n".join(
                    line
                    for line in valid_vnext_stdout.splitlines()
                    if not line.startswith(VNEXT_G00_SELFTEST_SUMMARY_PREFIX)
                )
                + "\n",
                "exactly one full-redteam summary",
            ),
            (
                "forged-summary-mode",
                selftest_vnext_stdout(
                    vnext_lane,
                    summary=selftest_vnext_full_summary(mode="fast"),
                ),
                "mode must be full",
            ),
            (
                "forged-summary-count",
                selftest_vnext_stdout(
                    vnext_lane,
                    summary=selftest_vnext_full_summary(
                        mutation_assertion_count=VNEXT_G00_REDTEAM_MUTATION_COUNT - 1
                    ),
                ),
                "mutation count must be",
            ),
            (
                "forged-summary-matrix",
                selftest_vnext_stdout(
                    vnext_lane,
                    summary=selftest_vnext_full_summary(
                        mutation_names=[
                            *VNEXT_G00_REDTEAM_MUTATION_NAMES[:-1],
                            "replacement-mutation-with-valid-count",
                        ]
                    ),
                ),
                "mutation matrix SHA256 mismatch",
            ),
        ]
        for name, bad_stdout, marker in invalid_redteam_outputs:
            try:
                verify_child_pass_line(
                    vnext_lane,
                    bad_stdout,
                    verify_checkout=False,
                )
                raise AssertionError(f"vnext-g00 {name} unexpectedly passed")
            except GateError as exc:
                require_selftest(marker in str(exc), f"{name}: {exc}")
        missing_flag_lane = LaneCommand(
            ["selftest"],
            expected_child_pass_line=vnext_lane.expected_child_pass_line,
            child_manifest_path=vnext_lane.child_manifest_path,
            expected_source_git_sha=vnext_lane.expected_source_git_sha,
            provenance_kind="vnext-g00",
        )
        try:
            verify_child_pass_line(
                missing_flag_lane,
                valid_vnext_stdout,
                verify_checkout=False,
            )
            raise AssertionError("vnext-g00 missing formal full-self-test flag unexpectedly passed")
        except GateError as exc:
            require_selftest("missing --require-full-self-test" in str(exc), str(exc))
        top_doc = manifest(
            args=argparse.Namespace(lane="vnext-g00", model=None),
            out_dir=vnext_provenance_root,
            lane_command=vnext_lane,
            status="pass",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            duration_sec=1.0,
            child_returncode=0,
            child_pass_line=vnext_lane.expected_child_pass_line,
            child_artifacts=provenance,
            error=None,
        )
        require_selftest(top_doc["child_artifacts"] == provenance, str(top_doc))

        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "stdout-only-pass",
            lambda case: (case / "manifest.json").unlink(),
            "invalid delegated manifest",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "shallow-pass-manifest",
            lambda case: write_selftest_json(
                case / "manifest.json",
                {
                    "schema_version": 1,
                    "status": "pass",
                    "source_git_sha": VNEXT_FROZEN_LEGACY_SHA,
                    "validator_git_sha": "1" * 40,
                    "validator_dirty_status": [],
                    "artifact_dir": str(case),
                    "waiver_count": 0,
                    "pass_line": f"FERRUM RUNTIME VNEXT G00 BASELINE PASS: {case}",
                },
            ),
            "artifact_index",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "artifact-index-sha",
            lambda case: mutate_selftest_json(
                case / "manifest.json",
                lambda data: data["artifact_index"][0].update({"sha256": "0" * 64}),
            ),
            "artifact_index",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "models-lock-sha",
            lambda case: mutate_selftest_json(
                case / "manifest.json",
                lambda data: data.update({"models_lock_sha256": "0" * 64}),
            ),
            "models.lock",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "expectations-catalog-sha",
            lambda case: mutate_selftest_json(
                case / "models.lock.json",
                lambda data: data["expectations_catalog"].update({"sha256": "0" * 64}),
            ),
            "models.lock",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "minimal-runner-identity",
            minimize_selftest_vnext_runner_identity,
            "runner field set mismatch",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "runner-validator-sha",
            tamper_selftest_vnext_runner_git_sha,
            "git_sha differs from delegated validator",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "expectations-snapshot-kind",
            tamper_selftest_vnext_expectations_kind,
            "kind must be raw-json",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "expectations-snapshot-bytes",
            tamper_selftest_vnext_expectations_snapshot,
            "sha256 differs from models.lock source contract",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "executor-invocation-mode",
            tamper_selftest_vnext_invocation_mode,
            "executor mode mismatch",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "cuda-primary-blocked",
            lambda case: mutate_selftest_json(
                case / "manifest.json",
                lambda data: data["correctness_lanes"].update(
                    {"m1-qwen35-4b/cuda": "blocked"}
                ),
            ),
            "correctness status matrix",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "model-resolution-sha",
            lambda case: (
                mutate_selftest_json(
                    case / "model-resolution.json",
                    lambda data: data["lanes"][0]["weight_source"].update({"revision": "2" * 40}),
                ),
                refresh_selftest_vnext_manifest(case),
            ),
            "model-resolution",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "config-sha",
            lambda case: (
                mutate_selftest_json(
                    case / "correctness/m3-qwen3-30b-a3b/cuda/effective-config.json",
                    lambda data: data.update({"tampered": True}),
                ),
                refresh_selftest_vnext_manifest(case),
            ),
            "effective_config",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "binary-sha",
            lambda case: (
                (case / "binaries/cuda/ferrum").write_text("tampered CUDA binary\n", encoding="utf-8"),
                refresh_selftest_vnext_manifest(case),
            ),
            "legacy-binaries.cuda.artifact_binary",
        )
        expect_vnext_provenance_reject(
            vnext_provenance_root,
            "model-identity",
            lambda case: (
                mutate_selftest_json(
                    case / "models.lock.json",
                    lambda data: data["models"][0]["lanes"]["cuda"].update({"revision": "3" * 40}),
                ),
                refresh_selftest_vnext_manifest(case, sync_models_lock=True),
            ),
            "weight identity differs",
        )
        try:
            verify_child_pass_line(
                LaneCommand(["selftest"], expected_child_pass_line="SELFTEST PASS"),
                "no pass line here\n",
            )
            raise AssertionError("missing delegated PASS line unexpectedly passed")
        except GateError as exc:
            require_selftest("SELFTEST PASS" in str(exc), str(exc))

        delegated_manifest_path = root / "delegated-manifest.json"
        expected_delegated_pass = "DELEGATED GATE PASS: selftest"
        expected_source_sha = "1" * 40
        delegated_lane = LaneCommand(
            ["selftest"],
            expected_child_pass_line=expected_delegated_pass,
            child_manifest_path=delegated_manifest_path,
            expected_source_git_sha=expected_source_sha,
        )
        valid_delegated_manifest = {
            "status": "pass",
            "pass_line": expected_delegated_pass,
            "source_git_sha": expected_source_sha,
        }
        write_selftest_json(delegated_manifest_path, valid_delegated_manifest)
        verify_child_pass_line(delegated_lane, expected_delegated_pass + "\n")
        for field, value, marker in [
            ("status", "fail", "status is not pass"),
            ("pass_line", "WRONG PASS", "pass_line mismatch"),
            ("source_git_sha", "2" * 40, "source_git_sha mismatch"),
        ]:
            bad_manifest = dict(valid_delegated_manifest)
            bad_manifest[field] = value
            write_selftest_json(delegated_manifest_path, bad_manifest)
            try:
                verify_child_pass_line(delegated_lane, expected_delegated_pass + "\n")
                raise AssertionError(f"bad delegated manifest field {field} unexpectedly passed")
            except GateError as exc:
                require_selftest(marker in str(exc), str(exc))

        release_root = root / "release-root"
        make_selftest_release_summary_artifact(release_root)
        summary_out = root / "release-summary"
        summary = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "release-summary",
                "--release-root",
                str(release_root),
                "--out",
                str(summary_out),
            ]
        )
        require_selftest(summary.returncode == 0, summary.stderr or summary.stdout)
        require_selftest(
            f"FERRUM GATE release-summary PASS: {summary_out}" in summary.stdout,
            summary.stdout,
        )
        summary_manifest = json.loads((summary_out / "gate.manifest.json").read_text())
        require_selftest(summary_manifest["status"] == "pass", summary_manifest)
        require_selftest(summary_manifest["pass_line"], summary_manifest)
        execution_paths = {row["path"] for row in summary_manifest["child_execution_artifacts"]}
        require_selftest(
            execution_paths
            == {
                "run_gate.child.command.json",
                "run_gate.child.stdout",
                "run_gate.child.stderr",
            },
            summary_manifest,
        )
        for row in summary_manifest["child_execution_artifacts"]:
            require_selftest(row["sha256"] == sha256(summary_out / row["path"]), row)
        require_selftest(
            summary_manifest["child_pass_line"] == f"G0 RELEASE PASS: {release_root}",
            summary_manifest,
        )

        completion_manifest_path = root / "completion-manifest.json"
        make_selftest_completion_manifest(completion_manifest_path)
        completion_out = root / "release-complete"
        complete = run_selftest_command(
            [
                sys.executable,
                str(this_script),
                "release-complete",
                "--completion-manifest",
                str(completion_manifest_path),
                "--out",
                str(completion_out),
            ]
        )
        require_selftest(complete.returncode == 0, complete.stderr or complete.stdout)
        require_selftest(
            f"FERRUM GATE release-complete PASS: {completion_out}" in complete.stdout,
            complete.stdout,
        )
        complete_manifest = json.loads((completion_out / "gate.manifest.json").read_text())
        require_selftest(complete_manifest["status"] == "pass", complete_manifest)
        require_selftest(
            complete_manifest["child_pass_line"]
            == f"FERRUM RELEASE COMPLETION PASS: {completion_out}",
            complete_manifest,
        )
        require_selftest(
            (completion_out / "release_completion_gate.json").is_file(),
            "missing completion validator artifact",
        )
    print("FERRUM RUN GATE SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lane", nargs="?", choices=LANES)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list-lanes", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--version")
    parser.add_argument("--asset-path", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--model")
    parser.add_argument("--model-name")
    parser.add_argument("--port", type=int)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--completion-manifest", type=Path)
    parser.add_argument("--coupling-inventory", type=Path)
    parser.add_argument("--model-resolution", type=Path)
    parser.add_argument("--g00a", type=Path)
    parser.add_argument("--g00f", type=Path)
    parser.add_argument("--s1-artifact", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.list_lanes:
        list_lanes()
        return 0
    if args.lane is None:
        parser.error("lane is required unless --list-lanes is set")
    if args.out is None:
        parser.error("--out is required")

    out_dir = args.out.resolve() if args.lane.startswith("vnext-") else args.out
    if args.lane in {"vnext-g00", "vnext-g00f", "vnext-g01a"}:
        try:
            require_external_vnext_g00_output(out_dir)
        except GateError as exc:
            print(f"FERRUM GATE {args.lane} FAIL: {out_dir}: {exc}", file=sys.stderr)
            return 1
    started_at = iso_now()
    start = time.monotonic()
    lane_command: LaneCommand | None = None
    child_returncode: int | None = None
    child_pass_line: str | None = None
    child_artifacts: dict[str, Any] | None = None
    status = "fail"
    error: str | None = None
    try:
        lane_command = build_lane_command(args, out_dir)
        if args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                out_dir / "gate.manifest.json",
                manifest(
                    args=args,
                    out_dir=out_dir,
                    lane_command=lane_command,
                    status="dry-run",
                    started_at=started_at,
                    finished_at=iso_now(),
                    duration_sec=time.monotonic() - start,
                    child_returncode=None,
                    child_pass_line=lane_command.expected_child_pass_line,
                    child_artifacts=None,
                    error=None,
                ),
            )
            print(" ".join(shlex.quote(part) for part in lane_command.cmd))
            return 0
        proc = run_child(
            lane_command.cmd,
            out_dir,
            args.timeout,
            prepare_out_dir=lane_command.provenance_kind != "vnext-g00a",
        )
        child_returncode = proc.returncode
        if proc.returncode != 0:
            error = f"delegated command failed rc={proc.returncode}"
            status = "fail"
        else:
            child_artifacts = verify_child_pass_line(lane_command, proc.stdout)
            child_pass_line = lane_command.expected_child_pass_line
            status = "pass"
    except (GateError, subprocess.TimeoutExpired) as exc:
        error = str(exc)
        status = "fail"
    finished_at = iso_now()
    doc = manifest(
        args=args,
        out_dir=out_dir,
        lane_command=lane_command,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=time.monotonic() - start,
        child_returncode=child_returncode,
        child_pass_line=child_pass_line,
        child_artifacts=child_artifacts,
        error=error,
    )
    write_json(out_dir / "gate.manifest.json", doc)
    if status == "pass":
        if args.lane.startswith("vnext-") and child_pass_line is not None:
            print(child_pass_line)
        print(doc["pass_line"])
        return 0
    print(f"FERRUM GATE {args.lane} FAIL: {out_dir}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
