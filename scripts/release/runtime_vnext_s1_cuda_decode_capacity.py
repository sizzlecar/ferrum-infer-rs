#!/usr/bin/env python3
"""Collect and validate bounded CUDA decode-capacity pressure evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import runtime_vnext_s1_cuda_capacity as common


PASS_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY PASS"
COLLECT_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY COLLECTED"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY FAIL"
CALIBRATION_TOKEN_BUDGET = 3
TARGET_TOKEN_BUDGET = 1024
MAX_NUM_SEQS = 3
MAX_MODEL_LEN = 512
PREFILL_FIRST_UNTIL_ACTIVE = 3
DECODE_SEQUENCE_FIT_POLICY = "immediate-only"
CALIBRATION_MAX_TOKENS = {"A": 128, "B": 1, "C": 16}
TARGET_MAX_TOKENS = {"A": 128, "B": 384, "C": 384}
PRESSURE_DECODE_SLOTS = ("A", "B", "C")
REBALANCE_PRIME_MAX_TOKENS = CALIBRATION_MAX_TOKENS
REBALANCE_PROBE_MAX_TOKENS = 1
REBALANCE_PROBE_WORD_COUNT = 256
REBALANCE_PROBE_WORKLOAD_SLOT = "rebalance-probe"
REBALANCE_PROBE_PROMPT = (
    "Cross-pool capacity probe. Read every word before answering."
    + " token" * REBALANCE_PROBE_WORD_COUNT
    + " Answer with one word."
)
DECODE_PROMPTS = {
    slot: (
        f"Capacity lane slot {slot}. Emit deterministic short words until the token "
        "limit; do not explain the task."
    )
    for slot in ("A", "B", "C")
}
DECODE_PROMPT_SHA256_BY_SLOT = {
    slot: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    for slot, prompt in DECODE_PROMPTS.items()
}
CANONICAL_DECODE_PROMPT_SHA256_BY_SLOT = {
    "A": "e979f692dd73f3d2469ec1152bc012476f205098ca6748b309574cf3f0fb2dc1",
    "B": "5f651c2e23b18b21ff42f68f66b208210d9423cf9d6f434c7abf844c0912456a",
    "C": "e3135728e0cc1a68b6c7af061931d5be5fe9dd4bb1ae40e1e778ea2b0fac325c",
}
MAX_DECODE_CAPACITY_EVENTS = 2048
ALLOWED_EXECUTION_STAGES = {
    "sequence_extension",
    "step_admission",
    "submission_wave",
}
ALLOWED_PRESSURE_YIELD_KINDS = {"peer_handoff", "self_recompute"}
SERVER_POLICY = {
    "max_model_len": MAX_MODEL_LEN,
    "max_num_seqs": MAX_NUM_SEQS,
    "sequence_fit_policy": DECODE_SEQUENCE_FIT_POLICY,
    "prefill_first_until_active": PREFILL_FIRST_UNTIL_ACTIVE,
    "calibration_max_num_batched_tokens": CALIBRATION_TOKEN_BUDGET,
    "target_max_num_batched_tokens": TARGET_TOKEN_BUDGET,
    "calibration_max_tokens": CALIBRATION_MAX_TOKENS,
    "target_sizing_max_tokens": CALIBRATION_MAX_TOKENS,
    "target_budget_derivation": "typed_initial_bundle_plus_sizing_residency",
    "target_rebalance_prime_max_tokens": REBALANCE_PRIME_MAX_TOKENS,
    "target_rebalance_probe_max_tokens": REBALANCE_PROBE_MAX_TOKENS,
    "target_rebalance_probe_prompt_sha256": hashlib.sha256(
        REBALANCE_PROBE_PROMPT.encode("utf-8")
    ).hexdigest(),
    "target_rebalance_probe_word_count": REBALANCE_PROBE_WORD_COUNT,
    "target_max_tokens": TARGET_MAX_TOKENS,
    "decode_prompt_sha256_by_slot": DECODE_PROMPT_SHA256_BY_SLOT,
}
STOP_POLICY = {
    "no_progress_timeout_seconds": common.MAX_PRESSURE_NO_PROGRESS_SECONDS,
    "joint_stream_timeout_seconds": common.MAX_PRESSURE_JOINT_STREAM_SECONDS,
    "max_trace_bytes": common.MAX_PRESSURE_TRACE_BYTES,
    "max_decode_capacity_events": MAX_DECODE_CAPACITY_EVENTS,
}
STABLE_EXECUTOR_IDENTITY_FIELDS = (
    "model_id",
    "family_fingerprint",
    "program_fingerprint",
    "runtime_fingerprint",
    "policy_id",
    "device_id",
)
EXECUTION_PLAN_IDENTITY_FIELDS = (
    "plan_id",
    "plan_hash",
    "policy_fingerprint",
    "runtime_memory_policy",
)


class DecodeCapacityGateError(common.CapacityGateError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DecodeCapacityGateError(message)


def require_executor_identity_shape(executor: dict[str, Any], label: str) -> None:
    require(
        common.GIT_SHA_RE.fullmatch(str(executor.get("model_id"))) is not None,
        f"{label}: invalid model id",
    )
    for field in (
        "family_fingerprint",
        "program_fingerprint",
        "runtime_fingerprint",
        "plan_hash",
        "policy_fingerprint",
    ):
        require(
            common.SHA256_RE.fullmatch(str(executor.get(field))) is not None,
            f"{label}: invalid {field}",
        )
    plan_hash = executor["plan_hash"]
    require(
        executor.get("plan_id") == f"plan/sha256/{plan_hash}",
        f"{label}: plan id does not identify plan hash",
    )
    require(
        executor.get("policy_id") == "policy.ferrum.product.vnext.default",
        f"{label}: non-canonical product policy id",
    )
    require(executor.get("device_id") == "device.cuda.0", f"{label}: non-CUDA-0 device id")
    require(
        isinstance(executor.get("runtime_memory_policy"), dict),
        f"{label}: runtime memory policy is missing",
    )
    admission = executor.get("runtime_admission_policy")
    require(
        isinstance(admission, dict)
        and admission.get("sequence_fit_policy") == "immediate_only",
        f"{label}: decode lane did not use ImmediateOnly sequence fit",
    )


def require_same_fields(
    snapshots: list[tuple[str, dict[str, Any]]],
    fields: tuple[str, ...],
    label: str,
) -> None:
    require(bool(snapshots), f"{label}: no executor snapshots")
    reference_name, reference = snapshots[0]
    for snapshot_name, snapshot in snapshots[1:]:
        for field in fields:
            require(
                snapshot.get(field) == reference.get(field),
                f"{label}: {snapshot_name} changed {field} from {reference_name}",
            )


def validate_executor_identity_contract(
    phase_snapshots: dict[str, list[tuple[str, dict[str, Any]]]],
) -> None:
    all_snapshots: list[tuple[str, dict[str, Any]]] = []
    for phase, snapshots in phase_snapshots.items():
        require(bool(snapshots), f"{phase}: no executor snapshots")
        for snapshot_name, snapshot in snapshots:
            require_executor_identity_shape(snapshot, snapshot_name)
            all_snapshots.append((snapshot_name, snapshot))
        require_same_fields(
            snapshots,
            STABLE_EXECUTOR_IDENTITY_FIELDS + EXECUTION_PLAN_IDENTITY_FIELDS,
            f"{phase} process identity",
        )
    require_same_fields(
        all_snapshots,
        STABLE_EXECUTOR_IDENTITY_FIELDS,
        "cross-process executor identity",
    )


def argv_option(argv: list[str], flag: str, label: str) -> str | None:
    positions = [index for index, value in enumerate(argv) if value == flag]
    require(len(positions) <= 1, f"{label}: duplicate {flag}")
    if not positions:
        return None
    position = positions[0]
    require(position + 1 < len(argv), f"{label}: {flag} has no value")
    return argv[position + 1]


def validate_canonical_server_argv(
    argv: list[str],
    *,
    label: str,
    token_budget: int,
    runtime_budget: int | None,
) -> None:
    require(
        len(argv) >= 3 and argv[1] == "serve",
        f"{label}: command is not the ferrum serve product entrypoint",
    )
    expected_options = {
        "--backend": "cuda",
        "--max-model-len": str(MAX_MODEL_LEN),
        "--max-num-seqs": str(MAX_NUM_SEQS),
        "--max-num-batched-tokens": str(token_budget),
        "--sequence-fit-policy": DECODE_SEQUENCE_FIT_POLICY,
        "--scheduler-prefill-first-until-active": str(PREFILL_FIRST_UNTIL_ACTIVE),
    }
    for flag, expected in expected_options.items():
        require(
            argv_option(argv, flag, label) == expected,
            f"{label}: {flag} differs from canonical value {expected}",
        )
    observed_runtime_budget = argv_option(argv, "--runtime-memory-budget-bytes", label)
    if runtime_budget is None:
        require(
            observed_runtime_budget is None,
            f"{label}: calibration unexpectedly overrides runtime memory budget",
        )
    else:
        require(
            observed_runtime_budget == str(runtime_budget),
            f"{label}: runtime memory budget differs from derived value",
        )


def read_server_argv(root: Path, phase: str) -> list[str]:
    command = common.read_json(root / phase / "server.command.json")
    argv = command.get("argv")
    require(
        isinstance(argv, list) and all(isinstance(value, str) for value in argv),
        f"{phase}: server command argv is invalid",
    )
    return argv


def demand_is_token_scaled(demand: Any) -> bool:
    if not isinstance(demand, dict) or len(demand) != 1:
        return False
    kind, value = next(iter(demand.items()))
    if kind in {"tokens", "pages", "bounded_shape_buckets"}:
        return True
    return (
        kind == "affine"
        and isinstance(value, dict)
        and isinstance(value.get("bytes_per_token"), int)
        and value["bytes_per_token"] > 0
    )


def phase_stable_demand_contract(demand: dict[str, Any]) -> dict[str, Any]:
    kind, value = next(iter(demand.items()))
    if kind not in {"tokens", "affine", "pages"} or not isinstance(value, dict):
        return demand
    stable = dict(value)
    stable.pop("maximum_tokens", None)
    stable.pop("maximum_pages", None)
    return {kind: stable}


def phase_stable_pool_contract(contract: dict[str, Any]) -> dict[str, Any]:
    provisioning = dict(contract["provisioning"])
    provisioning.pop("maximum_resident_bytes", None)
    return {
        **contract,
        "resources": [
            {
                **resource,
                "demand": phase_stable_demand_contract(resource["demand"]),
            }
            for resource in contract["resources"]
        ],
        "provisioning": provisioning,
    }


def validate_typed_pool_contract(
    pool_id: str, envelope: dict[str, Any], label: str
) -> dict[str, Any]:
    contract = envelope.get("contract")
    require(isinstance(contract, dict), f"{label}: {pool_id} has no typed pool contract")
    compatibility = contract.get("compatibility")
    require(
        isinstance(compatibility, dict)
        and compatibility.get("profile") == envelope.get("storage_profile"),
        f"{label}: {pool_id} compatibility differs from its storage profile",
    )
    resources = contract.get("resources")
    require(
        isinstance(resources, list) and resources,
        f"{label}: {pool_id} typed resources are missing",
    )
    seen_resources: set[str] = set()
    for resource in resources:
        require(
            isinstance(resource, dict),
            f"{label}: {pool_id} has an invalid typed resource",
        )
        resource_id = resource.get("resource_id")
        lifetime = resource.get("lifetime")
        quantum = resource.get("physical_allocation_quantum_bytes")
        require(
            isinstance(resource_id, str)
            and resource_id
            and resource_id not in seen_resources,
            f"{label}: {pool_id} has an invalid or duplicate resource id",
        )
        seen_resources.add(resource_id)
        require(
            lifetime in {"request", "sequence", "step", "invocation"},
            f"{label}: {pool_id}/{resource_id} has an invalid dynamic lifetime",
        )
        require(
            isinstance(resource.get("demand"), dict)
            and len(resource["demand"]) == 1,
            f"{label}: {pool_id}/{resource_id} has no typed demand",
        )
        require(
            isinstance(quantum, int) and quantum > 0,
            f"{label}: {pool_id}/{resource_id} has an invalid allocation quantum",
        )
    minima = (
        "minimum_request_bytes",
        "minimum_sequence_bytes",
        "minimum_step_bytes",
        "minimum_invocation_peak_bytes",
    )
    for key in minima:
        require(
            isinstance(contract.get(key), int) and contract[key] >= 0,
            f"{label}: {pool_id} has invalid {key}",
        )
    provisioning = contract.get("provisioning")
    minimum_resident = sum(contract[key] for key in minima)
    require(
        isinstance(provisioning, dict)
        and provisioning.get("mode") == "demand_driven_elastic"
        and provisioning.get("minimum_resident_bytes") == minimum_resident
        and isinstance(provisioning.get("maximum_resident_bytes"), int)
        and provisioning["maximum_resident_bytes"] >= minimum_resident,
        f"{label}: {pool_id} has an invalid typed provisioning contract",
    )
    return contract


def derive_target_budget_envelope(
    calibration: dict[str, Any], target_sizing: dict[str, Any]
) -> dict[str, Any]:
    require(
        target_sizing.get("static_bytes") == calibration.get("static_bytes"),
        "target sizing static bytes differ from calibration",
    )
    calibration_pools = calibration.get("pool_resident_bytes")
    sizing_pools = target_sizing.get("pool_resident_bytes")
    calibration_envelopes = calibration.get("pool_envelopes")
    sizing_envelopes = target_sizing.get("pool_envelopes")
    require(
        isinstance(calibration_pools, dict)
        and isinstance(sizing_pools, dict)
        and calibration_pools.keys() == sizing_pools.keys(),
        "target sizing pool identities differ from calibration",
    )
    require(
        isinstance(calibration_envelopes, dict)
        and isinstance(sizing_envelopes, dict)
        and calibration_envelopes.keys() == sizing_envelopes.keys(),
        "target sizing pool envelopes differ from calibration",
    )
    maximum_active_sequences = target_sizing.get("maximum_active_sequences")
    require(
        maximum_active_sequences == calibration.get("maximum_active_sequences") == MAX_NUM_SEQS,
        "target sizing typed sequence ceiling differs from the canonical workload",
    )

    sizing_observed_pool_resident_bytes: dict[str, int] = {}
    observed_or_floor_pool_bytes: dict[str, int] = {}
    initial_bundle_floor_bytes_by_pool: dict[str, int] = {}
    pool_storage_profiles: dict[str, Any] = {}
    pool_contracts: dict[str, Any] = {}
    token_scaled_sequence_pool_ids: list[str] = []
    for pool_id in sorted(calibration_pools):
        calibration_bytes = calibration_pools[pool_id]
        sizing_bytes = sizing_pools[pool_id]
        require(
            isinstance(calibration_bytes, int)
            and calibration_bytes >= 0
            and isinstance(sizing_bytes, int)
            and sizing_bytes >= 0,
            f"invalid sizing residency for {pool_id}",
        )
        calibration_profile = calibration_envelopes[pool_id].get("storage_profile")
        sizing_profile = sizing_envelopes[pool_id].get("storage_profile")
        require(
            calibration_profile == sizing_profile,
            f"target sizing storage profile differs for {pool_id}",
        )
        calibration_contract = validate_typed_pool_contract(
            pool_id, calibration_envelopes[pool_id], "calibration"
        )
        sizing_contract = validate_typed_pool_contract(
            pool_id, sizing_envelopes[pool_id], "target sizing"
        )
        require(
            phase_stable_pool_contract(calibration_contract)
            == phase_stable_pool_contract(sizing_contract),
            f"target sizing phase-stable typed contract differs for {pool_id}",
        )
        initial_bundle_floor = (
            sizing_contract["minimum_request_bytes"]
            + sizing_contract["minimum_sequence_bytes"]
        ) * maximum_active_sequences
        initial_bundle_floor += (
            sizing_contract["minimum_step_bytes"]
            + sizing_contract["minimum_invocation_peak_bytes"]
        )
        require(
            calibration_bytes >= initial_bundle_floor,
            f"calibration did not provision the typed initial bundle floor for {pool_id}",
        )
        observed_or_floor_bytes = max(sizing_bytes, initial_bundle_floor)
        require(
            initial_bundle_floor
            <= sizing_contract["provisioning"]["maximum_resident_bytes"],
            f"typed initial bundle floor exceeds the pool ceiling for {pool_id}",
        )
        if any(
            resource.get("lifetime") == "sequence"
            and demand_is_token_scaled(resource.get("demand"))
            for resource in sizing_contract["resources"]
        ):
            token_scaled_sequence_pool_ids.append(pool_id)
        sizing_observed_pool_resident_bytes[pool_id] = sizing_bytes
        observed_or_floor_pool_bytes[pool_id] = observed_or_floor_bytes
        initial_bundle_floor_bytes_by_pool[pool_id] = initial_bundle_floor
        pool_storage_profiles[pool_id] = calibration_profile
        pool_contracts[pool_id] = sizing_contract
    require(
        token_scaled_sequence_pool_ids,
        "typed target sizing contains no token-scaled sequence resource",
    )

    static_bytes = target_sizing.get("static_bytes")
    require(isinstance(static_bytes, int) and static_bytes > 0, "invalid sizing static bytes")
    sizing_resident_bytes = target_sizing.get("resident_bytes")
    require(
        isinstance(sizing_resident_bytes, int)
        and sizing_resident_bytes == sum(sizing_observed_pool_resident_bytes.values()),
        "target sizing resident total differs from its pool receipts",
    )
    calibration_budget = calibration.get("budget_claimed_bytes")
    calibration_resident_bytes = calibration.get("resident_bytes")
    require(
        target_sizing.get("budget_claimed_bytes") == static_bytes + sizing_resident_bytes,
        "target sizing budget differs from installed backing",
    )
    require(
        isinstance(calibration_budget, int)
        and isinstance(calibration_resident_bytes, int)
        and calibration_resident_bytes == sum(calibration_pools.values())
        and calibration_budget == static_bytes + calibration_resident_bytes,
        "calibration budget differs from its installed backing",
    )
    require(
        sizing_resident_bytes <= calibration_resident_bytes,
        "target sizing backing exceeds the calibrated resident budget",
    )
    minimum_initial_bundle_resident_bytes = sum(
        initial_bundle_floor_bytes_by_pool.values()
    )
    require(
        minimum_initial_bundle_resident_bytes <= calibration_resident_bytes,
        "typed initial bundles do not fit the calibrated resident budget",
    )
    observed_or_floor_resident_bytes = sum(observed_or_floor_pool_bytes.values())
    observed_or_floor_budget_gap_bytes = max(
        0, observed_or_floor_resident_bytes - calibration_resident_bytes
    )
    resident_bytes = calibration_resident_bytes
    exact_budget = calibration_budget
    return {
        "static_bytes": static_bytes,
        "resident_bytes": resident_bytes,
        "budget_claimed_bytes": exact_budget,
        "maximum_active_sequences": maximum_active_sequences,
        "sizing_observed_resident_bytes": sizing_resident_bytes,
        "sizing_observed_pool_resident_bytes": sizing_observed_pool_resident_bytes,
        "observed_or_floor_pool_bytes": observed_or_floor_pool_bytes,
        "observed_or_floor_resident_bytes": observed_or_floor_resident_bytes,
        "observed_or_floor_budget_gap_bytes": observed_or_floor_budget_gap_bytes,
        "requires_cross_pool_rebalance": observed_or_floor_budget_gap_bytes > 0,
        "initial_bundle_floor_bytes_by_pool": initial_bundle_floor_bytes_by_pool,
        "minimum_initial_bundle_resident_bytes": minimum_initial_bundle_resident_bytes,
        "initial_bundle_headroom_bytes": (
            calibration_resident_bytes - minimum_initial_bundle_resident_bytes
        ),
        "token_scaled_sequence_pool_ids": token_scaled_sequence_pool_ids,
        "pool_storage_profiles": pool_storage_profiles,
        "pool_contracts": pool_contracts,
        "calibration_budget_claimed_bytes": calibration_budget,
        "bootstrap_headroom_bytes": calibration_budget
        - target_sizing["budget_claimed_bytes"],
    }


def require_target_pool_within_budget_contract(
    target: dict[str, Any], envelope: dict[str, Any], exact_budget: int
) -> None:
    require(
        target.get("static_bytes") == envelope.get("static_bytes"),
        "target static bytes differ from its sizing envelope",
    )
    target_pools = target.get("pool_resident_bytes")
    target_envelopes = target.get("pool_envelopes")
    floor_pools = envelope.get("initial_bundle_floor_bytes_by_pool")
    profiles = envelope.get("pool_storage_profiles")
    contracts = envelope.get("pool_contracts")
    require(
        isinstance(target_pools, dict)
        and isinstance(floor_pools, dict)
        and target_pools.keys() == floor_pools.keys(),
        "target pool identities differ from its sizing envelope",
    )
    require(
        isinstance(target_envelopes, dict)
        and isinstance(profiles, dict)
        and isinstance(contracts, dict)
        and target_envelopes.keys() == target_pools.keys()
        and profiles.keys() == target_pools.keys()
        and contracts.keys() == target_pools.keys(),
        "target sizing profiles are missing",
    )
    require(
        target.get("maximum_active_sequences") == envelope.get("maximum_active_sequences"),
        "target dynamic pool sequence ceiling differs from its sizing envelope",
    )
    for pool_id, resident_bytes in target_pools.items():
        require(
            isinstance(resident_bytes, int) and resident_bytes >= 0,
            f"target pool {pool_id} has invalid residency",
        )
        require(
            target_envelopes[pool_id].get("storage_profile") == profiles[pool_id],
            f"target pool {pool_id} changed storage profile",
        )
        require(
            target_envelopes[pool_id].get("contract") == contracts[pool_id],
            f"target pool {pool_id} changed typed contract",
        )
    target_resident_bytes = target.get("resident_bytes")
    require(
        isinstance(target_resident_bytes, int)
        and target_resident_bytes == sum(target_pools.values()),
        "target resident total differs from its pool receipts",
    )
    require(
        target_resident_bytes <= envelope.get("resident_bytes", -1),
        "target installed dynamic backing exceeded the calibrated resident budget",
    )
    require(
        target.get("budget_claimed_bytes")
        == target.get("static_bytes", exact_budget + 1) + target_resident_bytes
        and target.get("budget_claimed_bytes", exact_budget + 1) <= exact_budget,
        "target installed backing exceeded the derived exact budget",
    )


def rebalance_prime_budget_receipt(
    prime: dict[str, Any],
    envelope: dict[str, Any],
    exact_budget: int,
) -> dict[str, int]:
    require_target_pool_within_budget_contract(prime, envelope, exact_budget)
    claimed_bytes = prime.get("budget_claimed_bytes")
    resident_bytes = prime.get("resident_bytes")
    resident_ceiling_bytes = envelope.get("resident_bytes")
    require(
        isinstance(claimed_bytes, int)
        and isinstance(resident_bytes, int)
        and isinstance(resident_ceiling_bytes, int),
        "rebalance prime budget receipt is incomplete",
    )
    headroom_bytes = exact_budget - claimed_bytes
    require(
        0 <= headroom_bytes <= exact_budget,
        "rebalance prime headroom is outside the exact budget",
    )
    return {
        "budget_ceiling_bytes": exact_budget,
        "claimed_bytes": claimed_bytes,
        "headroom_bytes": headroom_bytes,
        "resident_ceiling_bytes": resident_ceiling_bytes,
        "resident_bytes": resident_bytes,
    }


def source_key(source: Any) -> str:
    return json.dumps(source, sort_keys=True, separators=(",", ":"))


def wait_source_epochs(value: Any, label: str) -> dict[str, int]:
    require(isinstance(value, list) and value, f"{label}: current wait sources are missing")
    epochs: dict[str, int] = {}
    for entry in value:
        require(isinstance(entry, dict), f"{label}: current wait source is invalid")
        source = entry.get("source")
        key = source_key(source)
        require(key not in epochs, f"{label}: current wait source is duplicated")
        epoch = entry.get("epoch")
        require(isinstance(epoch, int) and epoch > 0, f"{label}: current wait epoch is invalid")
        epochs[key] = epoch
    return epochs


def compare_wait_sources(
    wait_condition: dict[str, Any],
    current_wait_sources: Any,
    *,
    label: str,
    expect_changed: bool,
) -> None:
    observed = {
        source_key(entry["source"]): entry["epoch"]
        for entry in wait_condition["observed"]
    }
    current = wait_source_epochs(current_wait_sources, label)
    require(current.keys() == observed.keys(), f"{label}: exact wait source set changed")
    require(
        all(current[key] >= observed[key] for key in observed),
        f"{label}: exact wait source generation regressed",
    )
    changed = any(current[key] > observed[key] for key in observed)
    require(changed == expect_changed, f"{label}: exact wait source change evidence is inconsistent")


def event_request_ids(row: dict[str, Any], label: str) -> list[str]:
    attributes = row.get("attributes")
    require(isinstance(attributes, dict), f"{label}: trace attributes are missing")
    request_ids = attributes.get("request_ids")
    require(isinstance(request_ids, list) and request_ids, f"{label}: request IDs are missing")
    require(
        all(isinstance(request_id, str) and request_id for request_id in request_ids),
        f"{label}: request identity is invalid",
    )
    require(len(set(request_ids)) == len(request_ids), f"{label}: request IDs are duplicated")
    return request_ids


def validate_decode_deferral(row: dict[str, Any], label: str) -> dict[str, Any]:
    require(row.get("status") == "ok" and row.get("error") is None, f"{label}: event failed")
    shape = row.get("shape")
    require(isinstance(shape, dict), f"{label}: shape is missing")
    decision = shape.get("decision")
    require(
        decision in {"split_cohort", "wait_for_release", "pressure_yield_planned"},
        f"{label}: decision is invalid",
    )
    width = shape.get("attempted_decode_width")
    require(isinstance(width, int) and width > 0, f"{label}: attempted width is invalid")
    stage = shape.get("execution_stage")
    require(stage in ALLOWED_EXECUTION_STAGES, f"{label}: execution stage is invalid")
    require(shape.get("decode_submit_observed") is False, f"{label}: decode submit preceded defer")
    request_ids = event_request_ids(row, label)
    require(len(request_ids) == width, f"{label}: attempted width/request count mismatch")
    attributes = row.get("attributes", {})
    victim_request_id = attributes.get("victim_request_id")
    progress_owner_id = attributes.get("progress_owner_id")
    progress_baseline = attributes.get("progress_baseline")
    episode_id = attributes.get("episode_id")
    planned_transition_ordinal = attributes.get("planned_transition_ordinal")
    yield_kind = attributes.get("yield_kind")
    if decision == "split_cohort":
        require(width >= 2, f"{label}: split cohort is not wide")
        require(victim_request_id is None, f"{label}: split cohort named a victim")
        require(progress_owner_id is None, f"{label}: split cohort named a progress owner")
        require(progress_baseline is None, f"{label}: split cohort named a progress baseline")
        require(yield_kind is None, f"{label}: split cohort named a yield kind")
    elif decision == "pressure_yield_planned":
        require(width == 1, f"{label}: pressure-yield cohort is not exact")
        require(
            isinstance(victim_request_id, str) and victim_request_id,
            f"{label}: pressure-yield victim is missing",
        )
        require(
            common.request_identity_matches(victim_request_id, request_ids[0]),
            f"{label}: pressure-yield victim does not match the failing cohort",
        )
        require(yield_kind in ALLOWED_PRESSURE_YIELD_KINDS, f"{label}: yield kind is invalid")
        require(isinstance(progress_owner_id, str), f"{label}: progress owner is missing")
        same_frontier = common.request_identity_matches(progress_owner_id, victim_request_id)
        require(
            (yield_kind == "self_recompute" and same_frontier)
            or (yield_kind == "peer_handoff" and not same_frontier),
            f"{label}: yield kind does not match its frontier identities",
        )
        require(
            isinstance(progress_baseline, int) and progress_baseline >= 0,
            f"{label}: logical progress baseline is invalid",
        )
        require(isinstance(episode_id, int) and episode_id > 0, f"{label}: episode id is invalid")
        require(
            isinstance(planned_transition_ordinal, int)
            and planned_transition_ordinal > 0,
            f"{label}: planned transition ordinal is invalid",
        )
    else:
        require(width == 1, f"{label}: a non-exact cohort was parked")
        require(victim_request_id is None, f"{label}: parked decode named a victim")
        require(progress_owner_id is None, f"{label}: parked decode named a progress owner")
        require(progress_baseline is None, f"{label}: parked decode named a progress baseline")
        require(yield_kind is None, f"{label}: parked decode named a yield kind")

    evidence = attributes.get("capacity_evidence")
    require(isinstance(evidence, dict), f"{label}: capacity evidence is missing")
    observed = common.validate_admission_epochs(evidence.get("observed"), label)
    wait_condition = common.validate_capacity_wait_condition(
        evidence.get("wait_condition"),
        coordinator_id=observed["coordinator_id"],
        label=label,
    )
    scheduler_snapshot = attributes.get("scheduler_snapshot")
    require(isinstance(scheduler_snapshot, dict), f"{label}: scheduler snapshot is missing")
    return {
        "ts_unix_nanos": common.event_wall_ns(row),
        "decision": decision,
        "width": width,
        "stage": stage,
        "request_ids": request_ids,
        "victim_request_id": victim_request_id,
        "progress_owner_id": progress_owner_id,
        "progress_baseline": progress_baseline,
        "episode_id": episode_id,
        "planned_transition_ordinal": planned_transition_ordinal,
        "yield_kind": yield_kind,
        "observed": observed,
        "wait_condition": wait_condition,
    }


def validate_decode_queue_transition(
    row: dict[str, Any], label: str, *, resumed: bool
) -> dict[str, Any]:
    require(row.get("status") == "ok" and row.get("error") is None, f"{label}: event failed")
    request_id = row.get("request_id")
    require(isinstance(request_id, str) and request_id, f"{label}: request identity is missing")
    shape = row.get("shape")
    attributes = row.get("attributes")
    require(isinstance(shape, dict) and isinstance(attributes, dict), f"{label}: trace payload is missing")
    require(shape.get("decode_submit_observed") is False, f"{label}: retry happened after submit")
    require(shape.get("probe_performed") is False, f"{label}: scheduler performed an admission probe")
    evidence = attributes.get("deferral_evidence")
    require(isinstance(evidence, dict), f"{label}: deferral evidence is missing")
    require(evidence.get("action") == "wait_for_release", f"{label}: deferred action is invalid")
    observed = common.validate_admission_epochs(evidence.get("observed"), label)
    current = common.validate_admission_epochs(evidence.get("current"), label)
    require(
        current["coordinator_id"] == observed["coordinator_id"],
        f"{label}: admission coordinator changed",
    )
    require(
        current["release_epoch"] >= observed["release_epoch"]
        and current["capacity_epoch"] >= observed["capacity_epoch"],
        f"{label}: global audit epoch regressed",
    )
    wait_condition = common.validate_capacity_wait_condition(
        evidence.get("wait_condition"),
        coordinator_id=observed["coordinator_id"],
        label=label,
    )
    if resumed:
        exact_changed = shape.get("exact_source_changed")
        policy_changed = shape.get("policy_epoch_changed")
        require(isinstance(exact_changed, bool), f"{label}: exact-source flag is missing")
        require(isinstance(policy_changed, bool), f"{label}: policy flag is missing")
        require(exact_changed or policy_changed, f"{label}: resume has no wake reason")
        expected_decision = "exact_source_changed" if exact_changed else "policy_epoch_changed"
        require(shape.get("decision") == expected_decision, f"{label}: resume reason is inconsistent")
        compare_wait_sources(
            wait_condition,
            evidence.get("current_wait_sources"),
            label=label,
            expect_changed=exact_changed,
        )
    else:
        require(shape.get("decision") == "skipped_unchanged", f"{label}: skip decision is invalid")
        compare_wait_sources(
            wait_condition,
            evidence.get("current_wait_sources"),
            label=label,
            expect_changed=False,
        )
    return {
        "ts_unix_nanos": common.event_wall_ns(row),
        "request_id": request_id,
        "exact_source_changed": shape.get("exact_source_changed", False),
    }


def validate_pressure_hold(row: dict[str, Any], label: str) -> dict[str, Any]:
    require(row.get("status") == "ok" and row.get("error") is None, f"{label}: event failed")
    request_id = row.get("request_id")
    shape = row.get("shape")
    require(isinstance(request_id, str) and request_id, f"{label}: victim identity is missing")
    require(isinstance(shape, dict), f"{label}: shape is missing")
    progress_owner_id = shape.get("progress_owner_id")
    require(
        isinstance(progress_owner_id, str) and progress_owner_id,
        f"{label}: progress owner identity is missing",
    )
    require(
        not common.request_identity_matches(request_id, progress_owner_id),
        f"{label}: pressure victim cannot own the progress role",
    )
    require(shape.get("decision") == "held_for_owner_progress", f"{label}: decision is invalid")
    require(shape.get("prefill_submit_observed") is False, f"{label}: held victim reached submit")
    require(shape.get("probe_performed") is False, f"{label}: held victim reached admission probe")
    progress_baseline = shape.get("progress_baseline")
    progress_current = shape.get("progress_current")
    require(
        isinstance(progress_baseline, int) and progress_baseline >= 0,
        f"{label}: progress baseline is invalid",
    )
    require(
        isinstance(progress_current, int) and progress_current >= progress_baseline,
        f"{label}: pressure owner logical progress regressed while its peer was held",
    )
    episode_id = shape.get("episode_id")
    require(isinstance(episode_id, int) and episode_id > 0, f"{label}: episode id is invalid")
    ticket = shape.get("waiting_ticket")
    require(isinstance(ticket, int) and ticket > 0, f"{label}: waiting ticket is invalid")
    return {
        "ts_unix_nanos": common.event_wall_ns(row),
        "victim_request_id": request_id,
        "progress_owner_id": progress_owner_id,
        "progress_baseline": progress_baseline,
        "progress_current": progress_current,
        "waiting_ticket": ticket,
        "episode_id": episode_id,
    }


def validate_pressure_hold_release(row: dict[str, Any], label: str) -> dict[str, Any]:
    require(row.get("status") == "ok" and row.get("error") is None, f"{label}: event failed")
    request_id = row.get("request_id")
    shape = row.get("shape")
    require(isinstance(request_id, str) and request_id, f"{label}: victim identity is missing")
    require(isinstance(shape, dict), f"{label}: shape is missing")
    progress_owner_id = shape.get("progress_owner_id")
    require(
        isinstance(progress_owner_id, str) and progress_owner_id,
        f"{label}: progress owner identity is missing",
    )
    require(
        not common.request_identity_matches(request_id, progress_owner_id),
        f"{label}: pressure victim cannot own the progress role",
    )
    decision = shape.get("decision")
    require(
        decision == "owner_terminal",
        f"{label}: release reason is invalid",
    )
    progress_baseline = shape.get("progress_baseline")
    progress_current = shape.get("progress_current")
    require(
        isinstance(progress_baseline, int) and progress_baseline >= 0,
        f"{label}: progress baseline is invalid",
    )
    require(
        isinstance(progress_current, int) and progress_current >= progress_baseline,
        f"{label}: logical progress generation regressed",
    )
    previous_wait_condition = shape.get("previous_wait_condition")
    current_wait_condition = shape.get("current_wait_condition")
    require(
        previous_wait_condition is None and current_wait_condition is None,
        f"{label}: terminal release carries obsolete source-retarget evidence",
    )
    require(
        shape.get("admission_eligible") is True,
        f"{label}: released victim did not regain dynamic admission eligibility",
    )
    require(
        shape.get("probe_performed") is False,
        f"{label}: pressure-hold release was incorrectly coupled to an admission probe",
    )
    require(
        shape.get("prefill_submit_observed") is False,
        f"{label}: release observation happened after prefill submit",
    )
    ticket = shape.get("waiting_ticket")
    require(isinstance(ticket, int) and ticket > 0, f"{label}: waiting ticket is invalid")
    episode_id = shape.get("episode_id")
    transition_ordinal = shape.get("transition_ordinal")
    require(isinstance(episode_id, int) and episode_id > 0, f"{label}: episode id is invalid")
    require(
        isinstance(transition_ordinal, int) and transition_ordinal > 0,
        f"{label}: release transition ordinal is invalid",
    )
    return {
        "ts_unix_nanos": common.event_wall_ns(row),
        "victim_request_id": request_id,
        "progress_owner_id": progress_owner_id,
        "progress_baseline": progress_baseline,
        "progress_current": progress_current,
        "decision": decision,
        "waiting_ticket": ticket,
        "episode_id": episode_id,
        "transition_ordinal": transition_ordinal,
        "previous_wait_condition": previous_wait_condition,
        "current_wait_condition": current_wait_condition,
    }


def validate_pressure_fence_armed(row: dict[str, Any], label: str) -> dict[str, Any]:
    require(row.get("status") == "ok" and row.get("error") is None, f"{label}: event failed")
    request_id = row.get("request_id")
    shape = row.get("shape")
    attributes = row.get("attributes")
    require(isinstance(request_id, str) and request_id, f"{label}: victim identity is missing")
    require(isinstance(shape, dict) and isinstance(attributes, dict), f"{label}: payload is missing")
    episode_id = shape.get("episode_id")
    planned = shape.get("planned_transition_ordinal")
    armed = shape.get("transition_ordinal")
    yield_kind = shape.get("yield_kind")
    require(isinstance(episode_id, int) and episode_id > 0, f"{label}: episode id is invalid")
    require(
        isinstance(planned, int) and isinstance(armed, int) and 0 < planned < armed,
        f"{label}: planned/armed ordinal order is invalid",
    )
    require(
        shape.get("physical_release_completed") is False,
        f"{label}: armed fence already claims physical release",
    )
    progress_owner_id = attributes.get("progress_owner_id")
    require(yield_kind in ALLOWED_PRESSURE_YIELD_KINDS, f"{label}: yield kind is invalid")
    require(isinstance(progress_owner_id, str), f"{label}: progress owner identity is missing")
    same_frontier = common.request_identity_matches(progress_owner_id, request_id)
    require(
        (yield_kind == "self_recompute" and same_frontier)
        or (yield_kind == "peer_handoff" and not same_frontier),
        f"{label}: yield kind does not match its frontier identities",
    )
    return {
        "ts_unix_nanos": common.event_wall_ns(row),
        "episode_id": episode_id,
        "victim_request_id": request_id,
        "progress_owner_id": progress_owner_id,
        "yield_kind": yield_kind,
        "planned_transition_ordinal": planned,
        "armed_transition_ordinal": armed,
    }


def validate_pressure_fence_completed(row: dict[str, Any], label: str) -> dict[str, Any]:
    require(row.get("status") == "ok" and row.get("error") is None, f"{label}: event failed")
    request_id = row.get("request_id")
    shape = row.get("shape")
    attributes = row.get("attributes")
    require(isinstance(request_id, str) and request_id, f"{label}: victim identity is missing")
    require(isinstance(shape, dict) and isinstance(attributes, dict), f"{label}: payload is missing")
    episode_id = shape.get("episode_id")
    released = shape.get("release_transition_ordinal")
    resumable = shape.get("resumable_transition_ordinal")
    owner_admission_pending = shape.get(
        "owner_admission_pending_transition_ordinal"
    )
    closed = shape.get("closed_transition_ordinal")
    closed_reason = shape.get("closed_reason")
    disposition = shape.get("completion_disposition")
    yield_kind = shape.get("yield_kind")
    require(isinstance(episode_id, int) and episode_id > 0, f"{label}: episode id is invalid")
    require(isinstance(released, int) and released > 0, f"{label}: release ordinal is invalid")
    require(
        shape.get("physical_release_completed") is True,
        f"{label}: completed fence has no physical release evidence",
    )
    require(
        shape.get("exact_source_advanced") is True,
        f"{label}: completed fence did not advance its exact failed source",
    )
    require(
        shape.get("transaction_wait_condition_advanced") is True,
        f"{label}: completed fence did not identify the advanced transaction predicate",
    )
    progress_owner_resumable = shape.get("progress_owner_resumable")
    if yield_kind == "self_recompute":
        require(progress_owner_resumable is False, f"{label}: self recompute resumed stale work")
        if disposition == "progress_owner_admission_pending":
            require(
                resumable is None
                and closed is None
                and closed_reason is None
                and isinstance(owner_admission_pending, int)
                and released < owner_admission_pending,
                f"{label}: owner-admission-pending ordinal order is invalid",
            )
            completion_ordinal = owner_admission_pending
        else:
            require(
                resumable is None
                and owner_admission_pending is None
                and isinstance(closed, int)
                and released < closed,
                f"{label}: standalone self-recompute release/closed ordinal order is invalid",
            )
            require(
                closed_reason is None and disposition == "self_recompute_queued",
                f"{label}: self-recompute completion disposition is invalid",
            )
            completion_ordinal = closed
    elif progress_owner_resumable is True:
        require(
            isinstance(resumable, int) and released < resumable,
            f"{label}: release/resumable ordinal order is invalid",
        )
        require(
            owner_admission_pending is None
            and closed is None
            and closed_reason is None
            and disposition == "progress_owner_resumable",
            f"{label}: resumable completion carries a closed disposition",
        )
        completion_ordinal = resumable
    else:
        require(progress_owner_resumable is False, f"{label}: resumable state is not typed")
        require(
            resumable is None
            and owner_admission_pending is None
            and isinstance(closed, int)
            and released < closed,
            f"{label}: release/closed ordinal order is invalid",
        )
        require(
            closed_reason == "owner_terminal" and disposition == "owner_terminal",
            f"{label}: closed completion reason is invalid",
        )
        completion_ordinal = closed
    release_authority = shape.get("release_authority")
    require(
        release_authority in {"retained_prefill", "active_sequence"},
        f"{label}: completed fence has no typed release authority",
    )
    current_availability = attributes.get("current_capacity_availability")
    require(
        isinstance(current_availability, list) and current_availability,
        f"{label}: completed fence has no current capacity snapshot",
    )
    require(shape.get("victim_requeued") is True, f"{label}: victim was not requeued")
    progress_owner_id = attributes.get("progress_owner_id")
    require(yield_kind in ALLOWED_PRESSURE_YIELD_KINDS, f"{label}: yield kind is invalid")
    require(isinstance(progress_owner_id, str), f"{label}: progress owner identity is missing")
    same_frontier = common.request_identity_matches(progress_owner_id, request_id)
    require(
        (yield_kind == "self_recompute" and same_frontier)
        or (yield_kind == "peer_handoff" and not same_frontier),
        f"{label}: yield kind does not match its frontier identities",
    )
    return {
        "ts_unix_nanos": common.event_wall_ns(row),
        "episode_id": episode_id,
        "victim_request_id": request_id,
        "progress_owner_id": progress_owner_id,
        "yield_kind": yield_kind,
        "release_transition_ordinal": released,
        "resumable_transition_ordinal": resumable,
        "owner_admission_pending_transition_ordinal": owner_admission_pending,
        "closed_transition_ordinal": closed,
        "closed_reason": closed_reason,
        "completion_disposition": disposition,
        "completion_transition_ordinal": completion_ordinal,
        "release_authority": release_authority,
    }


def validate_decode_trace(
    rows: list[dict[str, Any]], *, started_wall_ns: int, finished_wall_ns: int
) -> dict[str, Any]:
    require(started_wall_ns > 0 and finished_wall_ns >= started_wall_ns, "invalid trace window")
    window = [
        row
        for row in rows
        if isinstance(row.get("ts_unix_nanos"), int)
        and started_wall_ns <= row["ts_unix_nanos"] <= finished_wall_ns
    ]
    deferral_rows = [row for row in window if row.get("phase") == "vnext.decode_capacity_deferred"]
    require(deferral_rows, "target produced no typed decode-capacity deferral")
    require(
        len(deferral_rows) <= MAX_DECODE_CAPACITY_EVENTS,
        "decode-capacity deferrals exceeded the bounded event ceiling",
    )
    deferrals = [
        validate_decode_deferral(row, f"decode deferral {index}")
        for index, row in enumerate(deferral_rows)
    ]
    splits = [event for event in deferrals if event["decision"] == "split_cohort"]
    parks = [event for event in deferrals if event["decision"] == "wait_for_release"]
    yields = [
        event for event in deferrals if event["decision"] == "pressure_yield_planned"
    ]
    require(splits, "target never adaptively split a capacity-blocked decode cohort")
    require(yields, "target never planned a typed execution-capacity yield")

    hold_rows = [
        row
        for row in window
        if row.get("phase") == "vnext.execution_capacity_pressure_hold_active"
    ]
    require(
        len(hold_rows) <= MAX_DECODE_CAPACITY_EVENTS,
        "execution-capacity pressure holds exceeded the bounded event ceiling",
    )
    holds = [
        validate_pressure_hold(row, f"pressure hold {index}")
        for index, row in enumerate(hold_rows)
    ]
    release_rows = [
        row
        for row in window
        if row.get("phase") == "vnext.execution_capacity_pressure_hold_released"
    ]
    require(
        len(release_rows) <= MAX_DECODE_CAPACITY_EVENTS,
        "execution-capacity pressure-hold releases exceeded the bounded event ceiling",
    )
    releases = [
        validate_pressure_hold_release(row, f"pressure hold release {index}")
        for index, row in enumerate(release_rows)
    ]
    armed_rows = [
        row
        for row in window
        if row.get("phase") == "vnext.execution_capacity_pressure_release_fence_armed"
    ]
    completed_fence_rows = [
        row
        for row in window
        if row.get("phase") == "vnext.execution_capacity_pressure_release_fence_completed"
    ]
    require(
        len(armed_rows) + len(completed_fence_rows) <= 2 * MAX_DECODE_CAPACITY_EVENTS,
        "execution-capacity release fences exceeded the bounded event ceiling",
    )
    armed_fences = [
        validate_pressure_fence_armed(row, f"pressure fence armed {index}")
        for index, row in enumerate(armed_rows)
    ]
    completed_fences = [
        validate_pressure_fence_completed(row, f"pressure fence completed {index}")
        for index, row in enumerate(completed_fence_rows)
    ]

    skip_rows = [row for row in window if row.get("phase") == "vnext.decode_capacity_skipped_unchanged"]
    resume_rows = [row for row in window if row.get("phase") == "vnext.decode_capacity_resumed"]
    require(
        len(skip_rows) + len(resume_rows) <= MAX_DECODE_CAPACITY_EVENTS,
        "decode-capacity wake observations exceeded the bounded event ceiling",
    )
    skips = [
        validate_decode_queue_transition(row, f"decode skip {index}", resumed=False)
        for index, row in enumerate(skip_rows)
    ]
    resumes = [
        validate_decode_queue_transition(row, f"decode resume {index}", resumed=True)
        for index, row in enumerate(resume_rows)
    ]
    for park in parks:
        request_id = park["request_ids"][0]
        matching_resume = [
            resume
            for resume in resumes
            if resume["ts_unix_nanos"] > park["ts_unix_nanos"]
            and common.request_identity_matches(resume["request_id"], request_id)
            and resume["exact_source_changed"]
        ]
        matching_yield = [
            pressure_yield
            for pressure_yield in yields
            if pressure_yield["ts_unix_nanos"] > park["ts_unix_nanos"]
            and common.request_identity_matches(
                pressure_yield["progress_owner_id"], request_id
            )
        ]
        require(
            matching_resume or matching_yield,
            f"parked decode {request_id} neither resumed after an exact-source change nor received a released progress role",
        )
    completed_rows = [
        row for row in rows if str(row.get("phase", "")).endswith("request_completed")
    ]
    admitted_rows = [
        row
        for row in rows
        if row.get("phase") == "vnext.prefill_admission"
        and row.get("shape", {}).get("decision") == "admitted"
    ]
    progress_owner_by_episode: dict[int, str] = {}
    for pressure_yield in yields:
        episode_id = pressure_yield["episode_id"]
        progress_owner_id = pressure_yield["progress_owner_id"]
        prior_owner = progress_owner_by_episode.setdefault(
            episode_id, progress_owner_id
        )
        require(
            common.request_identity_matches(prior_owner, progress_owner_id),
            f"pressure episode {episode_id} transferred its stable progress owner",
        )
    for hold in holds:
        require(
            hold["episode_id"] in progress_owner_by_episode,
            f"pressure episode {hold['episode_id']} hold has no planned yield",
        )
        require(
            common.request_identity_matches(
                progress_owner_by_episode[hold["episode_id"]],
                hold["progress_owner_id"],
            ),
            f"pressure episode {hold['episode_id']} hold has a foreign progress owner",
        )
        require(
            not any(
                pressure_yield["episode_id"] == hold["episode_id"]
                and common.request_identity_matches(
                    pressure_yield["progress_owner_id"], hold["victim_request_id"]
                )
                for pressure_yield in yields
            ),
            f"pressure episode {hold['episode_id']} promoted a held victim to owner",
        )
    for pressure_yield in yields:
        episode_id = pressure_yield["episode_id"]
        victim_request_id = pressure_yield["victim_request_id"]
        progress_owner_id = pressure_yield["progress_owner_id"]
        progress_baseline = pressure_yield["progress_baseline"]
        yield_kind = pressure_yield["yield_kind"]
        if yield_kind == "peer_handoff":
            require(
                any(
                    park["ts_unix_nanos"] < pressure_yield["ts_unix_nanos"]
                    and common.request_identity_matches(
                        park["request_ids"][0], progress_owner_id
                    )
                    for park in parks
                ),
                f"pressure progress owner {progress_owner_id} was not previously parked",
            )
        matching_armed = [
            fence
            for fence in armed_fences
            if fence["episode_id"] == episode_id
            and common.request_identity_matches(fence["victim_request_id"], victim_request_id)
            and common.request_identity_matches(fence["progress_owner_id"], progress_owner_id)
            and fence["yield_kind"] == yield_kind
        ]
        matching_completed = [
            fence
            for fence in completed_fences
            if fence["episode_id"] == episode_id
            and common.request_identity_matches(fence["victim_request_id"], victim_request_id)
            and common.request_identity_matches(fence["progress_owner_id"], progress_owner_id)
            and fence["yield_kind"] == yield_kind
        ]
        require(len(matching_armed) == 1, f"pressure episode {episode_id} has no unique armed fence")
        require(
            len(matching_completed) == 1,
            f"pressure episode {episode_id} has no unique completed fence",
        )
        armed = matching_armed[0]
        completed = matching_completed[0]
        require(
            pressure_yield["planned_transition_ordinal"]
            == armed["planned_transition_ordinal"]
            < armed["armed_transition_ordinal"]
            < completed["release_transition_ordinal"]
            < completed["completion_transition_ordinal"],
            f"pressure episode {episode_id} violated release-fence ordinal order",
        )
        matching_holds = [
            hold
            for hold in holds
            if hold["episode_id"] == episode_id
            and common.request_identity_matches(hold["victim_request_id"], victim_request_id)
            and common.request_identity_matches(hold["progress_owner_id"], progress_owner_id)
            and hold["progress_baseline"] == progress_baseline
        ]
        if yield_kind == "self_recompute":
            require(
                completed["completion_disposition"]
                in {
                    "self_recompute_queued",
                    "progress_owner_admission_pending",
                },
                f"self-recompute episode {episode_id} has an invalid reconstruction state",
            )
            require(
                not matching_holds,
                f"self-recompute episode {episode_id} incorrectly published a peer hold",
            )
            if (
                completed["completion_disposition"]
                == "progress_owner_admission_pending"
            ):
                require(
                    any(
                        hold["episode_id"] == episode_id
                        and common.request_identity_matches(
                            hold["progress_owner_id"], progress_owner_id
                        )
                        and not common.request_identity_matches(
                            hold["victim_request_id"], progress_owner_id
                        )
                        for hold in holds
                    ),
                    f"pressure owner {progress_owner_id} entered admission-pending without a held peer",
                )
        elif completed["completion_disposition"] == "progress_owner_resumable":
            require(
                matching_holds,
                f"pressure victim {victim_request_id} was not held for owner {progress_owner_id}",
            )
        matching_releases = [
            release
            for release in releases
            if release["episode_id"] == episode_id
            and common.request_identity_matches(release["victim_request_id"], victim_request_id)
            and common.request_identity_matches(release["progress_owner_id"], progress_owner_id)
            and release["progress_baseline"] == progress_baseline
        ]
        handoff_releases = [
            release
            for release in matching_releases
            if release["decision"] == "owner_terminal"
        ]
        if yield_kind == "self_recompute":
            require(
                not matching_releases,
                f"self-recompute episode {episode_id} incorrectly released a peer hold",
            )
            continue
        require(
            handoff_releases,
            f"pressure hold for {victim_request_id} has no owner-terminal release",
        )
        first_handoff_release = min(
            release["ts_unix_nanos"]
            for release in handoff_releases
        )
        require(
            all(hold["ts_unix_nanos"] < first_handoff_release for hold in matching_holds),
            f"pressure victim {victim_request_id} remained held after handoff completion",
        )
        require(
            not any(
                common.request_identity_matches(row.get("request_id"), victim_request_id)
                and completed["ts_unix_nanos"]
                < common.event_wall_ns(row)
                < first_handoff_release
                for row in admitted_rows
            ),
            f"pressure victim {victim_request_id} was re-admitted before its owner released capacity",
        )
        for release in handoff_releases:
            if release["decision"] == "owner_terminal":
                require(
                    any(
                        common.request_identity_matches(
                            row.get("request_id"), progress_owner_id
                        )
                        and common.event_wall_ns(row) <= release["ts_unix_nanos"]
                        for row in completed_rows
                    ),
                    f"pressure owner {progress_owner_id} did not terminate before victim release",
                )
        require(
            all(
                release["transition_ordinal"]
                >= completed["completion_transition_ordinal"]
                for release in matching_releases
            ),
            f"pressure episode {episode_id} released a hold before fence completion",
        )
        if matching_holds:
            require(
                any(
                    release["waiting_ticket"] == hold["waiting_ticket"]
                    for release in matching_releases
                    for hold in matching_holds
                ),
                f"pressure hold for {victim_request_id} changed waiting identity",
            )

    for hold in holds:
        require(
            any(
                pressure_yield["episode_id"] == hold["episode_id"]
                and common.request_identity_matches(
                    pressure_yield["victim_request_id"], hold["victim_request_id"]
                )
                for pressure_yield in yields
            ),
            f"pressure hold for {hold['victim_request_id']} has no typed yield decision",
        )

    for release in releases:
        require(
            any(
                pressure_yield["episode_id"] == release["episode_id"]
                and common.request_identity_matches(
                    pressure_yield["victim_request_id"], release["victim_request_id"]
                )
                for pressure_yield in yields
            ),
            f"pressure-hold release for {release['victim_request_id']} has no typed yield decision",
        )

    deferred_request_ids = sorted(
        {
            request_id
            for event in deferrals
            for request_id in [
                *event["request_ids"],
                *(
                    [event["victim_request_id"]]
                    if event["victim_request_id"] is not None
                    else []
                ),
            ]
        }
    )
    for pressure_yield in yields:
        victim_request_id = pressure_yield["victim_request_id"]
        progress_owner_id = pressure_yield["progress_owner_id"]
        progress_baseline = pressure_yield["progress_baseline"]
        if pressure_yield["yield_kind"] == "self_recompute":
            completed = next(
                fence
                for fence in completed_fences
                if fence["episode_id"] == pressure_yield["episode_id"]
                and fence["yield_kind"] == "self_recompute"
            )
            require(
                any(
                    common.request_identity_matches(row.get("request_id"), victim_request_id)
                    and common.event_wall_ns(row) > completed["ts_unix_nanos"]
                    for row in admitted_rows
                ),
                f"self-recompute frontier {victim_request_id} was not re-admitted after its fence",
            )
            continue
        handoff_releases = [
            release
            for release in releases
            if release["decision"] == "owner_terminal"
            and release["episode_id"] == pressure_yield["episode_id"]
            and common.request_identity_matches(
                release["victim_request_id"], victim_request_id
            )
            and common.request_identity_matches(
                release["progress_owner_id"], progress_owner_id
            )
            and release["progress_baseline"] == progress_baseline
        ]
        require(
            handoff_releases,
            f"decode progress owner {progress_owner_id} did not terminate before releasing its peer",
        )
        handoff_released_at = min(
            release["ts_unix_nanos"] for release in handoff_releases
        )
        require(
            any(
                common.request_identity_matches(row.get("request_id"), victim_request_id)
                and common.event_wall_ns(row) > handoff_released_at
                for row in admitted_rows
            ),
            f"pressure victim {victim_request_id} was not re-admitted after handoff completion",
        )
    for request_id in deferred_request_ids:
        require(
            any(common.request_identity_matches(row.get("request_id"), request_id) for row in completed_rows),
            f"decode-capacity request {request_id} has no completion event",
        )
    return {
        "deferral_events": len(deferrals),
        "split_events": len(splits),
        "park_events": len(parks),
        "pressure_yield_events": len(yields),
        "pressure_yield_kinds": sorted({event["yield_kind"] for event in yields}),
        "pressure_fence_armed_events": len(armed_fences),
        "pressure_fence_completed_events": len(completed_fences),
        "pressure_hold_events": len(holds),
        "pressure_hold_release_events": len(releases),
        "skip_events": len(skips),
        "resume_events": len(resumes),
        "stages": sorted({event["stage"] for event in deferrals}),
        "max_attempted_decode_width": max(event["width"] for event in deferrals),
        "deferred_request_ids": deferred_request_ids,
        "pressure_victim_request_ids": sorted(
            {event["victim_request_id"] for event in yields}
        ),
        "pressure_episode_pairs": [
            list(pair)
            for pair in sorted(
                {
                    (hold["victim_request_id"], hold["progress_owner_id"])
                    for hold in holds
                }
            )
        ],
    }


def validate_decode_counter_provenance(
    rows: list[dict[str, Any]],
    *,
    started_wall_ns: int,
    finished_wall_ns: int,
    counters: dict[str, Any],
) -> dict[str, Any]:
    counter_by_stage = {
        "sequence_extension": "extension_deferrals",
        "step_admission": "step_deferrals",
        "submission_wave": "wave_deferrals",
    }
    for counter in (*counter_by_stage.values(), "backing_deferrals"):
        require(
            isinstance(counters.get(counter), int) and counters[counter] >= 0,
            f"target counter {counter} is invalid",
        )
    raw_deferrals = [
        row
        for row in rows
        if row.get("phase") == "vnext.decode_capacity_deferred"
        and isinstance(row.get("ts_unix_nanos"), int)
        and started_wall_ns <= row["ts_unix_nanos"] <= finished_wall_ns
    ]
    deferrals = [
        validate_decode_deferral(row, f"counter provenance deferral {index}")
        for index, row in enumerate(raw_deferrals)
    ]
    direct_by_stage = {stage: 0 for stage in counter_by_stage}
    backing_events = 0
    for event in deferrals:
        sources = {
            entry["source"]
            for entry in event["wait_condition"]["observed"]
            if isinstance(entry.get("source"), str)
        }
        if sources & {"plan_device_budget", "process_device_capacity"}:
            backing_events += 1
        else:
            direct_by_stage[event["stage"]] += 1
    for stage, event_count in direct_by_stage.items():
        if event_count == 0:
            continue
        counter = counter_by_stage[stage]
        require(
            counters[counter] > 0,
            f"target counter {counter} did not record direct {stage} deferral evidence",
        )
    require(
        backing_events == 0 or counters["backing_deferrals"] >= backing_events,
        "target backing_deferrals did not cover device-backing trace evidence",
    )
    return {
        "direct_trace_events_by_stage": direct_by_stage,
        "device_backing_trace_events": backing_events,
        "counters": {
            counter: counters[counter]
            for counter in (*counter_by_stage.values(), "backing_deferrals")
        },
    }


def validate_maintenance_trace(
    rows: list[dict[str, Any]],
    *,
    started_wall_ns: int,
    finished_wall_ns: int,
    label: str,
) -> dict[str, Any]:
    require(
        started_wall_ns > 0 and finished_wall_ns >= started_wall_ns,
        f"{label}: invalid maintenance window",
    )
    maintenance_rows = [
        row
        for row in rows
        if row.get("phase") == "vnext.prefill_backing_maintenance"
        and isinstance(row.get("ts_unix_nanos"), int)
        and started_wall_ns <= row["ts_unix_nanos"] <= finished_wall_ns
    ]
    require(maintenance_rows, f"{label}: no typed prefill backing maintenance")
    growths: list[dict[str, Any]] = []
    rebalances: list[dict[str, Any]] = []
    for index, row in enumerate(maintenance_rows):
        require(
            row.get("status") == "ok" and row.get("error") is None,
            f"{label} maintenance {index}: event failed",
        )
        attributes = row.get("attributes")
        require(
            isinstance(attributes, dict),
            f"{label} maintenance {index}: attributes are missing",
        )
        evidence = attributes.get("maintenance_evidence")
        require(
            isinstance(evidence, dict),
            f"{label} maintenance {index}: evidence is missing",
        )
        if evidence.get("outcome") != "maintained":
            continue
        pools_grown = evidence.get("pools_grown")
        allocated_bytes = evidence.get("allocated_bytes")
        require(
            isinstance(pools_grown, int)
            and pools_grown > 0
            and isinstance(allocated_bytes, int)
            and allocated_bytes > 0,
            f"{label} maintenance {index}: maintained growth is invalid",
        )
        exact = common.validate_maintenance_rebalance_evidence(
            evidence,
            f"{label} maintenance {index}",
        )
        growth = {
            "pools_grown": pools_grown,
            "allocated_bytes": allocated_bytes,
            **exact,
        }
        growths.append(growth)
        if exact["pools_reclaimed"] > 0:
            rebalances.append(growth)
    require(growths, f"{label}: no successful typed backing growth")
    return {
        "maintenance_events": len(maintenance_rows),
        "maintained_events": len(growths),
        "allocated_bytes": sum(event["allocated_bytes"] for event in growths),
        "rebalance_events": len(rebalances),
        "pools_reclaimed": sum(event["pools_reclaimed"] for event in rebalances),
        "chunks_reclaimed": sum(event["chunks_reclaimed"] for event in rebalances),
        "reclaimed_bytes": sum(event["reclaimed_bytes"] for event in rebalances),
        "allocated_bytes_after_rebalance": sum(
            event["allocated_bytes"] for event in rebalances
        ),
        "exact_receipt": bool(rebalances),
        "evidence_owner": (
            common.CROSS_POOL_REBALANCE_EVIDENCE_OWNER if rebalances else None
        ),
        "receipts": [
            {
                "pool_ids": event["pool_ids"],
                "chunk_identities": event["chunk_identities"],
                "capacity_epochs": event["capacity_epochs"],
            }
            for event in rebalances
        ],
    }


def validate_rebalance_trace(
    rows: list[dict[str, Any]],
    *,
    started_wall_ns: int,
    finished_wall_ns: int,
    label: str = "cross-pool evidence",
) -> dict[str, Any]:
    summary = validate_maintenance_trace(
        rows,
        started_wall_ns=started_wall_ns,
        finished_wall_ns=finished_wall_ns,
        label=label,
    )
    require(
        summary["rebalance_events"] > 0,
        f"{label} produced no typed rebalance",
    )
    return summary


def require_decode_prompt(result: dict[str, Any], slot: str) -> None:
    require(slot in DECODE_PROMPT_SHA256_BY_SLOT, f"invalid decode workload slot {slot}")
    require(
        result.get("workload_slot") == slot
        and result.get("prompt_sha256") == DECODE_PROMPT_SHA256_BY_SLOT[slot],
        f"{slot}: decode prompt differs from the canonical workload",
    )


def require_pressure_decode_live_overlap(
    results: dict[str, dict[str, Any]], label: str
) -> dict[str, int]:
    require(
        isinstance(results, dict) and set(results) == {"A", "B", "C"},
        f"{label}: invalid decode client set",
    )
    first_content: list[int] = []
    finished: list[int] = []
    for slot in PRESSURE_DECODE_SLOTS:
        result = results[slot]
        first = result.get("first_content_wall_ns")
        end = result.get("finished_wall_ns")
        require(
            isinstance(first, int) and first > 0,
            f"{label}-{slot}: first-content timestamp is missing",
        )
        require(
            isinstance(end, int) and end >= first,
            f"{label}-{slot}: completion timestamp is invalid",
        )
        first_content.append(first)
        finished.append(end)
    latest_first_content = max(first_content)
    earliest_completion = min(finished)
    require(
        latest_first_content < earliest_completion,
        (
            f"{label}: workload calibration invalid: pressure streams did not overlap "
            "in decode"
        ),
    )
    return {
        "latest_pressure_first_content_wall_ns": latest_first_content,
        "earliest_pressure_completion_wall_ns": earliest_completion,
        "overlap_wall_ns": earliest_completion - latest_first_content,
    }


def start_stream_group(
    server: common.ServerSession,
    *,
    out_dir: Path,
    prefix: str,
    max_tokens_by_slot: dict[str, int],
    timeout: float,
) -> dict[str, common.StreamTask]:
    require(
        set(max_tokens_by_slot) == {"A", "B", "C"},
        "stream token policy must contain A/B/C",
    )
    tasks = {
        slot: common.StreamTask(
            port=server.port,
            model=server.model_id,
            role=f"{prefix}-{slot}",
            workload_slot=slot,
            max_tokens=max_tokens_by_slot[slot],
            out_dir=out_dir,
            timeout=timeout,
            prompt=DECODE_PROMPTS[slot],
        )
        for slot in ("A", "B", "C")
    }
    tasks["A"].start()
    tasks["A"].wait_first(timeout)
    require(tasks["A"].is_alive(), f"{prefix}-A completed before B/C started")
    tasks["B"].start()
    tasks["C"].start()
    return tasks


def wait_stream_group(
    tasks: dict[str, common.StreamTask],
    *,
    trace_path: Path,
    trace_baseline_bytes: int,
    timeout: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    require(set(tasks) == {"A", "B", "C"}, "stream group must contain A/B/C")
    started = time.monotonic()
    deadline = started + timeout
    guard = common.PressureStopGuard(
        initial_progress={role: task.live_content_chunks() for role, task in tasks.items()},
        started_monotonic=started,
        no_progress_timeout=STOP_POLICY["no_progress_timeout_seconds"],
        max_unchanged_skips=MAX_DECODE_CAPACITY_EVENTS,
        max_trace_bytes=STOP_POLICY["max_trace_bytes"],
    )
    while True:
        now = time.monotonic()
        active_roles = {role for role, task in tasks.items() if task.is_alive()}
        for role, task in tasks.items():
            if role not in active_roles and isinstance(task.result, dict) and task.result.get("error"):
                raise DecodeCapacityGateError(f"{role} stream failed: {task.result['error']}")
        trace_bytes = trace_path.stat().st_size if trace_path.is_file() else 0
        require(trace_bytes >= trace_baseline_bytes, "scheduler trace was truncated")
        guard.observe(
            progress={role: task.live_content_chunks() for role, task in tasks.items()},
            unchanged_skips=0,
            trace_bytes=trace_bytes - trace_baseline_bytes,
            now_monotonic=now,
            active_roles=active_roles,
        )
        if not active_roles:
            break
        require(now < deadline, "stream group exceeded the bounded joint timeout")
        time.sleep(min(0.05, max(0.0, deadline - now)))

    results = {role: task.join(0) for role, task in tasks.items()}
    for role, result in results.items():
        require_decode_prompt(result, role)
        common.validate_stream(result, role)
    require(
        max(result["started_wall_ns"] for result in results.values())
        < min(result["finished_wall_ns"] for result in results.values()),
        "three streams were not concurrently live",
    )
    return results, {
        "duration_seconds": time.monotonic() - started,
        "content_chunks_by_role": {
            role: task.live_content_chunks() for role, task in tasks.items()
        },
        "max_stall_seconds_by_role": guard.max_stall_seconds_by_role,
        "trace_baseline_bytes": trace_baseline_bytes,
        "trace_bytes": (
            trace_path.stat().st_size - trace_baseline_bytes if trace_path.is_file() else 0
        ),
    }


def server_session(
    *,
    repo: Path,
    binary: Path,
    model: Path,
    port: int,
    out_dir: Path,
    runtime_budget: int | None,
    max_num_batched_tokens: int,
    startup_timeout: float,
) -> common.ServerSession:
    return common.ServerSession(
        repo=repo,
        binary=binary,
        model=model,
        port=port,
        out_dir=out_dir,
        runtime_budget=runtime_budget,
        startup_timeout=startup_timeout,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=max_num_batched_tokens,
        prefill_first_until_active=PREFILL_FIRST_UNTIL_ACTIVE,
        sequence_fit_policy=DECODE_SEQUENCE_FIT_POLICY,
    )


def collect(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    binary = args.binary.resolve()
    model = args.model.resolve()
    out = args.out.resolve()
    require(not out.exists(), f"collection output already exists: {out}")
    require(binary.is_file(), f"missing binary: {binary}")
    require(model.is_dir(), f"missing model directory: {model}")
    require(args.port < 65534, "base port leaves no sizing/target-server ports")
    require(
        0 < args.request_timeout <= STOP_POLICY["joint_stream_timeout_seconds"],
        "request timeout exceeds the bounded stop policy",
    )
    out.mkdir(parents=True)
    provenance = {
        "schema_version": 1,
        "command_line": sys.argv,
        "git_sha": common.command_output(["git", "rev-parse", "HEAD"], repo),
        "dirty_status": common.command_output(["git", "status", "--short"], repo),
        "binary_path": str(binary),
        "binary_sha256": common.sha256(binary),
        "model_path": str(model),
        "nvidia_smi": common.command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            repo,
        ).splitlines(),
        "sanitized_env": {
            key: os.environ[key]
            for key in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "LD_LIBRARY_PATH", "RUST_LOG")
            if key in os.environ
        },
        "started_wall_ns": time.time_ns(),
    }
    common.write_json(out / "provenance.json", provenance)
    require(common.GIT_SHA_RE.fullmatch(provenance["git_sha"]) is not None, "invalid git SHA")
    require(not provenance["dirty_status"], "CUDA evidence requires a clean checkout")
    sessions: list[common.ServerSession] = []
    tasks: dict[str, common.StreamTask] = {}
    collection: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s1_cuda_decode_capacity_collection",
        "status": "reject",
        "source_git_sha": provenance["git_sha"],
        "binary_sha256": provenance["binary_sha256"],
        "model_path": str(model),
        "server_policy": SERVER_POLICY,
        "stop_policy": STOP_POLICY,
        "error": None,
    }
    try:
        run_summary = common.collect_run_smoke(
            repo=repo,
            binary=binary,
            model=model,
            out_dir=out / "run",
            timeout=args.request_timeout,
        )
        collection["run"] = run_summary
        calibration = server_session(
            repo=repo,
            binary=binary,
            model=model,
            port=args.port,
            out_dir=out / "calibration",
            runtime_budget=None,
            max_num_batched_tokens=CALIBRATION_TOKEN_BUDGET,
            startup_timeout=args.startup_timeout,
        )
        sessions.append(calibration)
        calibration_trace_baseline = (
            calibration.trace_path.stat().st_size if calibration.trace_path.is_file() else 0
        )
        tasks = start_stream_group(
            calibration,
            out_dir=out / "calibration" / "clients",
            prefix="calibration",
            max_tokens_by_slot=CALIBRATION_MAX_TOKENS,
            timeout=args.request_timeout,
        )
        calibration_clients, calibration_monitor = wait_stream_group(
            tasks,
            trace_path=calibration.trace_path,
            trace_baseline_bytes=calibration_trace_baseline,
            timeout=args.request_timeout,
        )
        tasks = {}
        calibration_health = calibration.health("health.final.json")
        calibration_executor = common.find_executor_snapshot(calibration_health)
        require(calibration_executor is not None, "calibration health has no vNext executor")
        calibration_start_executor = common.executor_snapshot_from_health(
            calibration.out_dir / "health.start.json", "decode calibration startup"
        )
        calibration_pool = common.quiescent_pool_snapshot(
            calibration_executor,
            "decode calibration",
            baseline_executor=calibration_start_executor,
        )
        calibration_budget = calibration_pool["budget_claimed_bytes"]
        collection["calibration"] = {
            "clients": calibration_clients,
            "monitor": calibration_monitor,
            "pool_snapshot": calibration_pool,
            "health_final": "calibration/health.final.json",
            "trace": "calibration/scheduler-trace.jsonl",
        }
        collection["calibration_budget_bytes"] = calibration_budget
        calibration.stop()

        target_sizing = server_session(
            repo=repo,
            binary=binary,
            model=model,
            port=args.port + 1,
            out_dir=out / "target-sizing",
            runtime_budget=calibration_budget,
            max_num_batched_tokens=TARGET_TOKEN_BUDGET,
            startup_timeout=args.startup_timeout,
        )
        sessions.append(target_sizing)
        sizing_trace_baseline = (
            target_sizing.trace_path.stat().st_size
            if target_sizing.trace_path.is_file()
            else 0
        )
        tasks = start_stream_group(
            target_sizing,
            out_dir=out / "target-sizing" / "clients",
            prefix="target-sizing",
            max_tokens_by_slot=CALIBRATION_MAX_TOKENS,
            timeout=args.request_timeout,
        )
        sizing_clients, sizing_monitor = wait_stream_group(
            tasks,
            trace_path=target_sizing.trace_path,
            trace_baseline_bytes=sizing_trace_baseline,
            timeout=args.request_timeout,
        )
        tasks = {}
        sizing_health = target_sizing.health("health.final.json")
        sizing_executor = common.find_executor_snapshot(sizing_health)
        require(sizing_executor is not None, "target sizing health has no vNext executor")
        sizing_start_executor = common.executor_snapshot_from_health(
            target_sizing.out_dir / "health.start.json",
            "decode target sizing startup",
        )
        sizing_pool = common.quiescent_pool_snapshot(
            sizing_executor,
            "decode target sizing",
            baseline_executor=sizing_start_executor,
        )
        sizing_started = min(
            result["started_wall_ns"] for result in sizing_clients.values()
        )
        sizing_finished = max(
            result["finished_wall_ns"] for result in sizing_clients.values()
        )
        sizing_maintenance_summary = validate_maintenance_trace(
            common.read_trace(target_sizing.trace_path),
            started_wall_ns=sizing_started,
            finished_wall_ns=sizing_finished,
            label="target sizing",
        )
        collection["target_sizing"] = {
            "clients": sizing_clients,
            "monitor": sizing_monitor,
            "pool_snapshot": sizing_pool,
            "maintenance_summary": sizing_maintenance_summary,
            "health_final": "target-sizing/health.final.json",
            "trace": "target-sizing/scheduler-trace.jsonl",
        }
        target_budget_envelope = derive_target_budget_envelope(
            calibration_pool, sizing_pool
        )
        require(
            target_budget_envelope["requires_cross_pool_rebalance"] is True,
            "target sizing envelope does not require the rebalance probe",
        )
        exact_budget = target_budget_envelope["budget_claimed_bytes"]
        collection["target_budget_envelope"] = target_budget_envelope
        collection["exact_budget_bytes"] = exact_budget
        target_sizing.stop()

        target = server_session(
            repo=repo,
            binary=binary,
            model=model,
            port=args.port + 2,
            out_dir=out / "target",
            runtime_budget=exact_budget,
            max_num_batched_tokens=TARGET_TOKEN_BUDGET,
            startup_timeout=args.startup_timeout,
        )
        sessions.append(target)

        prime_trace_baseline = target.trace_path.stat().st_size if target.trace_path.is_file() else 0
        tasks = start_stream_group(
            target,
            out_dir=out / "target-rebalance-prime" / "clients",
            prefix="target-rebalance-prime",
            max_tokens_by_slot=REBALANCE_PRIME_MAX_TOKENS,
            timeout=args.request_timeout,
        )
        prime_clients, prime_monitor = wait_stream_group(
            tasks,
            trace_path=target.trace_path,
            trace_baseline_bytes=prime_trace_baseline,
            timeout=args.request_timeout,
        )
        tasks = {}
        prime_health = target.health("health.rebalance-prime.json")
        prime_executor = common.find_executor_snapshot(prime_health)
        require(prime_executor is not None, "rebalance prime health has no vNext executor")
        target_start_executor = common.executor_snapshot_from_health(
            target.out_dir / "health.start.json", "decode target startup"
        )
        prime_pool = common.quiescent_pool_snapshot(
            prime_executor,
            "decode target rebalance prime",
            baseline_executor=target_start_executor,
        )
        prime_budget_receipt = rebalance_prime_budget_receipt(
            prime_pool,
            target_budget_envelope,
            exact_budget,
        )

        probe_task = common.StreamTask(
            port=target.port,
            model=target.model_id,
            role="target-rebalance-probe",
            workload_slot=REBALANCE_PROBE_WORKLOAD_SLOT,
            max_tokens=REBALANCE_PROBE_MAX_TOKENS,
            out_dir=out / "target-rebalance-probe" / "clients",
            timeout=args.request_timeout,
            prompt=REBALANCE_PROBE_PROMPT,
        )
        tasks = {REBALANCE_PROBE_WORKLOAD_SLOT: probe_task}
        probe_deadline = time.monotonic() + args.request_timeout
        probe_task.start()
        probe_task.wait_first(
            min(args.request_timeout, STOP_POLICY["no_progress_timeout_seconds"])
        )
        probe_client = probe_task.join(max(0.0, probe_deadline - time.monotonic()))
        common.validate_stream(probe_client, "target-rebalance-probe")
        tasks = {}
        probe_health = target.health("health.rebalance-probe.json")
        probe_executor = common.find_executor_snapshot(probe_health)
        require(probe_executor is not None, "rebalance probe health has no vNext executor")
        probe_pool = common.quiescent_pool_snapshot(
            probe_executor,
            "decode target rebalance probe",
            baseline_executor=target_start_executor,
        )
        target_rows = common.read_trace(target.trace_path)
        probe_maintenance_summary = validate_rebalance_trace(
            target_rows,
            started_wall_ns=probe_client["started_wall_ns"],
            finished_wall_ns=probe_client["finished_wall_ns"],
            label="target rebalance probe",
        )

        target_trace_baseline = target.trace_path.stat().st_size if target.trace_path.is_file() else 0
        tasks = start_stream_group(
            target,
            out_dir=out / "target" / "clients",
            prefix="target",
            max_tokens_by_slot=TARGET_MAX_TOKENS,
            timeout=args.request_timeout,
        )
        target_clients, target_monitor = wait_stream_group(
            tasks,
            trace_path=target.trace_path,
            trace_baseline_bytes=target_trace_baseline,
            timeout=args.request_timeout,
        )
        tasks = {}
        target_started = min(result["started_wall_ns"] for result in target_clients.values())
        target_finished = max(result["finished_wall_ns"] for result in target_clients.values())
        target_health = target.health("health.final.json")
        target_executor = common.find_executor_snapshot(target_health)
        require(target_executor is not None, "target health has no vNext executor")
        target_pool = common.quiescent_pool_snapshot(
            target_executor,
            "decode target",
            baseline_executor=target_start_executor,
        )
        target.stop()
        collection["target"] = {
            "rebalance_prime": {
                "clients": prime_clients,
                "monitor": prime_monitor,
                "pool_snapshot": prime_pool,
                "budget_receipt": prime_budget_receipt,
                "health": "target/health.rebalance-prime.json",
            },
            "rebalance_probe": {
                "client": probe_client,
                "pool_snapshot": probe_pool,
                "health": "target/health.rebalance-probe.json",
                "maintenance_summary": probe_maintenance_summary,
            },
            "clients": target_clients,
            "monitor": target_monitor,
            "pool_snapshot": target_pool,
            "health_final": "target/health.final.json",
            "trace": "target/scheduler-trace.jsonl",
        }
        collection["target"]["pressure_decode_live_overlap"] = (
            require_pressure_decode_live_overlap(target_clients, "target")
        )
        require_target_pool_within_budget_contract(
            target_pool, target_budget_envelope, exact_budget
        )
        target_rows = common.read_trace(target.trace_path)
        decode_summary = validate_decode_trace(
            target_rows,
            started_wall_ns=target_started,
            finished_wall_ns=target_finished,
        )
        collection["target"]["decode_summary"] = decode_summary

        collection.update(
            {
                "status": "collected",
                "finished_wall_ns": time.time_ns(),
                "error": None,
            }
        )
        common.write_json(out / "collection.json", collection)
        print(f"{COLLECT_PREFIX}: {out}")
        return 0
    except Exception as error:
        for session in reversed(sessions):
            try:
                session.stop()
            except Exception:
                pass
        unsettled = common.settle_stream_tasks(tasks, timeout=10.0)
        collection["failure_cleanup"] = {
            "unsettled_client_roles": unsettled,
            "finished_wall_ns": time.time_ns(),
        }
        collection["error"] = str(error)
        collection["finished_wall_ns"] = time.time_ns()
        common.write_json(out / "collection.json", collection)
        print(f"{FAIL_PREFIX}: {out}: {error}", file=sys.stderr)
        return 1
    finally:
        for session in reversed(sessions):
            try:
                session.stop()
            except Exception as error:
                print(f"decode capacity session cleanup failed: {error}", file=sys.stderr)


def validate_stream_group(
    root: Path,
    phase: str,
    results: dict[str, Any],
    max_tokens_by_slot: dict[str, int],
) -> tuple[int, int, dict[str, float]]:
    require(isinstance(results, dict) and set(results) == {"A", "B", "C"}, f"{phase}: invalid client set")
    for role, result in results.items():
        require(isinstance(result, dict), f"{phase}-{role}: result is invalid")
        require_decode_prompt(result, role)
        require(
            result.get("max_tokens") == max_tokens_by_slot[role],
            f"{phase}-{role}: max_tokens differs from the canonical workload",
        )
        common.validate_stream(result, f"{phase}-{role}")
    started = min(result["started_wall_ns"] for result in results.values())
    finished = max(result["finished_wall_ns"] for result in results.values())
    require(
        max(result["started_wall_ns"] for result in results.values())
        < min(result["finished_wall_ns"] for result in results.values()),
        f"{phase}: streams did not overlap",
    )
    silences: dict[str, float] = {}
    for role, result in results.items():
        events = root / phase / "clients" / f"{phase}-{role}.events.jsonl"
        silences[role] = common.max_stream_silence_seconds(
            result,
            common.read_stream_content_times(events),
            monitored_from_wall_ns=started,
        )
        require(
            silences[role] < STOP_POLICY["no_progress_timeout_seconds"],
            f"{phase}-{role}: token progress stalled for {silences[role]:.3f}s",
        )
    return started, finished, silences


def validate_rebalance_probe(root: Path, result: dict[str, Any]) -> tuple[int, int, float]:
    require(isinstance(result, dict), "target-rebalance-probe: result is invalid")
    require(
        result.get("role") == "target-rebalance-probe"
        and result.get("workload_slot") == REBALANCE_PROBE_WORKLOAD_SLOT
        and result.get("max_tokens") == REBALANCE_PROBE_MAX_TOKENS,
        "target-rebalance-probe: workload identity changed",
    )
    require(
        result.get("prompt_sha256")
        == hashlib.sha256(REBALANCE_PROBE_PROMPT.encode("utf-8")).hexdigest(),
        "target-rebalance-probe: prompt changed",
    )
    prompt_tokens = result.get("prompt_tokens")
    require(
        isinstance(prompt_tokens, int)
        and REBALANCE_PROBE_MAX_TOKENS < prompt_tokens < MAX_MODEL_LEN,
        "target-rebalance-probe: tokenized prompt is outside the product model limit",
    )
    common.validate_stream(result, "target-rebalance-probe")
    started = result["started_wall_ns"]
    finished = result["finished_wall_ns"]
    events = (
        root
        / "target-rebalance-probe"
        / "clients"
        / "target-rebalance-probe.events.jsonl"
    )
    silence = common.max_stream_silence_seconds(
        result,
        common.read_stream_content_times(events),
        monitored_from_wall_ns=started,
    )
    require(
        silence < STOP_POLICY["no_progress_timeout_seconds"],
        f"target-rebalance-probe: token progress stalled for {silence:.3f}s",
    )
    return started, finished, silence


def validate(root: Path, out: Path) -> int:
    root = root.resolve()
    out = out.resolve()
    require(root.is_dir(), f"missing collection directory: {root}")
    out.mkdir(parents=True, exist_ok=True)
    collection = common.read_json(root / "collection.json")
    provenance = common.read_json(root / "provenance.json")
    require(collection.get("status") == "collected", f"collection is unusable: {collection.get('error')}")
    require(collection.get("server_policy") == SERVER_POLICY, "collection used a non-canonical server policy")
    require(collection.get("stop_policy") == STOP_POLICY, "collection used a non-canonical stop policy")
    source_git_sha = collection.get("source_git_sha")
    require(common.GIT_SHA_RE.fullmatch(str(source_git_sha)) is not None, "invalid source git SHA")
    require(source_git_sha == provenance.get("git_sha"), "collection/provenance SHA mismatch")
    require(not provenance.get("dirty_status"), "CUDA evidence used a dirty checkout")
    binary_sha256 = collection.get("binary_sha256")
    require(common.SHA256_RE.fullmatch(str(binary_sha256)) is not None, "invalid binary SHA256")
    require(binary_sha256 == provenance.get("binary_sha256"), "binary SHA mismatch")
    gpu_rows = provenance.get("nvidia_smi")
    require(
        isinstance(gpu_rows, list) and len(gpu_rows) == 1 and "RTX 4090" in gpu_rows[0],
        "artifact is not from exactly one RTX 4090",
    )
    require("Qwen3.5-4B" in str(collection.get("model_path")), "artifact model is not Qwen3.5-4B")

    calibration = collection.get("calibration")
    target_sizing = collection.get("target_sizing")
    target = collection.get("target")
    require(
        isinstance(calibration, dict)
        and isinstance(target_sizing, dict)
        and isinstance(target, dict),
        "scenario summaries are missing",
    )
    rebalance_prime = target.get("rebalance_prime")
    rebalance_probe = target.get("rebalance_probe")
    require(
        isinstance(rebalance_prime, dict) and isinstance(rebalance_probe, dict),
        "target rebalance phases are missing",
    )
    calibration_start_health = common.read_json(root / "calibration/health.start.json")
    sizing_start_health = common.read_json(root / "target-sizing/health.start.json")
    target_start_health = common.read_json(root / "target/health.start.json")
    calibration_health = common.read_json(root / str(calibration.get("health_final")))
    sizing_health = common.read_json(root / str(target_sizing.get("health_final")))
    target_health = common.read_json(root / str(target.get("health_final")))
    prime_health = common.read_json(root / str(rebalance_prime.get("health")))
    probe_health = common.read_json(root / str(rebalance_probe.get("health")))
    calibration_start_executor = common.find_executor_snapshot(calibration_start_health)
    sizing_start_executor = common.find_executor_snapshot(sizing_start_health)
    target_start_executor = common.find_executor_snapshot(target_start_health)
    calibration_executor = common.find_executor_snapshot(calibration_health)
    sizing_executor = common.find_executor_snapshot(sizing_health)
    target_executor = common.find_executor_snapshot(target_health)
    prime_executor = common.find_executor_snapshot(prime_health)
    probe_executor = common.find_executor_snapshot(probe_health)
    require(
        calibration_start_executor is not None
        and sizing_start_executor is not None
        and target_start_executor is not None
        and calibration_executor is not None
        and sizing_executor is not None
        and target_executor is not None
        and prime_executor is not None
        and probe_executor is not None,
        "raw executor snapshots are missing",
    )
    validate_executor_identity_contract(
        {
            "calibration": [
                ("calibration start", calibration_start_executor),
                ("calibration final", calibration_executor),
            ],
            "target sizing": [
                ("target sizing start", sizing_start_executor),
                ("target sizing final", sizing_executor),
            ],
            "target": [
                ("target start", target_start_executor),
                ("target rebalance prime", prime_executor),
                ("target rebalance probe", probe_executor),
                ("target final", target_executor),
            ],
        }
    )
    require(
        calibration_executor.get("model_id") in str(collection.get("model_path")),
        "executor model id is absent from immutable model path",
    )
    calibration_pool = common.quiescent_pool_snapshot(
        calibration_executor,
        "raw decode calibration",
        baseline_executor=calibration_start_executor,
    )
    sizing_pool = common.quiescent_pool_snapshot(
        sizing_executor,
        "raw decode target sizing",
        baseline_executor=sizing_start_executor,
    )
    target_pool = common.quiescent_pool_snapshot(
        target_executor,
        "raw decode target",
        baseline_executor=target_start_executor,
    )
    prime_pool = common.quiescent_pool_snapshot(
        prime_executor,
        "raw decode target rebalance prime",
        baseline_executor=target_start_executor,
    )
    probe_pool = common.quiescent_pool_snapshot(
        probe_executor,
        "raw decode target rebalance probe",
        baseline_executor=target_start_executor,
    )
    calibration_budget = collection.get("calibration_budget_bytes")
    exact_budget = collection.get("exact_budget_bytes")
    require(
        isinstance(calibration_budget, int)
        and calibration_budget == calibration_pool["budget_claimed_bytes"],
        "calibration budget does not match its installed backing",
    )
    require(calibration.get("pool_snapshot") == calibration_pool, "calibration summary differs from raw health")
    require(
        target_sizing.get("pool_snapshot") == sizing_pool,
        "target sizing summary differs from raw health",
    )
    require(target.get("pool_snapshot") == target_pool, "target summary differs from raw health")
    require(
        rebalance_prime.get("pool_snapshot") == prime_pool,
        "rebalance prime summary differs from raw health",
    )
    require(
        rebalance_probe.get("pool_snapshot") == probe_pool,
        "rebalance probe summary differs from raw health",
    )
    target_budget_envelope = derive_target_budget_envelope(calibration_pool, sizing_pool)
    require(
        collection.get("target_budget_envelope") == target_budget_envelope,
        "target budget envelope differs from raw sizing receipts",
    )
    require(
        target_budget_envelope["requires_cross_pool_rebalance"] is True,
        "target sizing envelope does not require the rebalance probe",
    )
    require(
        isinstance(exact_budget, int)
        and exact_budget == target_budget_envelope["budget_claimed_bytes"],
        "exact budget does not match the target-compatible sizing envelope",
    )
    validate_canonical_server_argv(
        read_server_argv(root, "calibration"),
        label="calibration",
        token_budget=CALIBRATION_TOKEN_BUDGET,
        runtime_budget=None,
    )
    validate_canonical_server_argv(
        read_server_argv(root, "target-sizing"),
        label="target sizing",
        token_budget=TARGET_TOKEN_BUDGET,
        runtime_budget=calibration_budget,
    )
    validate_canonical_server_argv(
        read_server_argv(root, "target"),
        label="target",
        token_budget=TARGET_TOKEN_BUDGET,
        runtime_budget=exact_budget,
    )
    require_target_pool_within_budget_contract(
        target_pool, target_budget_envelope, exact_budget
    )
    require_target_pool_within_budget_contract(
        prime_pool, target_budget_envelope, exact_budget
    )
    require_target_pool_within_budget_contract(
        probe_pool, target_budget_envelope, exact_budget
    )
    prime_budget_receipt = rebalance_prime_budget_receipt(
        prime_pool,
        target_budget_envelope,
        exact_budget,
    )
    require(
        rebalance_prime.get("budget_receipt") == prime_budget_receipt,
        "rebalance prime budget receipt differs from raw backing",
    )
    sizing_policy = sizing_executor.get("runtime_memory_policy")
    require(isinstance(sizing_policy, dict), "target sizing runtime memory policy is missing")
    require(
        sizing_policy.get("capacity_bytes", 0) - sizing_policy.get("reserve_bytes", 0)
        == calibration_budget,
        "target sizing runtime did not use the narrow calibration budget",
    )
    policy = target_executor.get("runtime_memory_policy")
    require(isinstance(policy, dict), "target runtime memory policy is missing")
    require(
        policy.get("capacity_bytes", 0) - policy.get("reserve_bytes", 0) == exact_budget,
        "target runtime did not use the calibrated exact budget",
    )

    calibration_started, calibration_finished, calibration_silence = validate_stream_group(
        root, "calibration", calibration.get("clients"), CALIBRATION_MAX_TOKENS
    )
    sizing_started, sizing_finished, sizing_silence = validate_stream_group(
        root, "target-sizing", target_sizing.get("clients"), CALIBRATION_MAX_TOKENS
    )
    prime_started, prime_finished, prime_silence = validate_stream_group(
        root,
        "target-rebalance-prime",
        rebalance_prime.get("clients"),
        REBALANCE_PRIME_MAX_TOKENS,
    )
    probe_started, probe_finished, probe_silence = validate_rebalance_probe(
        root, rebalance_probe.get("client")
    )
    target_started, target_finished, target_silence = validate_stream_group(
        root, "target", target.get("clients"), TARGET_MAX_TOKENS
    )
    decode_live_overlap = require_pressure_decode_live_overlap(target["clients"], "target")
    require(
        target.get("pressure_decode_live_overlap") == decode_live_overlap,
        "target pressure decode-live overlap receipt differs from raw clients",
    )
    for slot in ("A", "B", "C"):
        common.validate_replayed_workload(
            slot,
            {
                "calibration": calibration["clients"][slot],
                "target-sizing": target_sizing["clients"][slot],
                "target-rebalance-prime": rebalance_prime["clients"][slot],
                "target": target["clients"][slot],
            },
        )
    calibration_rows = common.read_trace(root / str(calibration.get("trace")))
    require(
        not any(
            row.get("phase") == "vnext.decode_capacity_deferred"
            and isinstance(row.get("ts_unix_nanos"), int)
            and calibration_started <= row["ts_unix_nanos"] <= calibration_finished
            for row in calibration_rows
        ),
        "single-token calibration unexpectedly hit decode capacity pressure",
    )
    sizing_rows = common.read_trace(root / str(target_sizing.get("trace")))
    require(
        not any(
            row.get("phase") == "vnext.decode_capacity_deferred"
            and isinstance(row.get("ts_unix_nanos"), int)
            and sizing_started <= row["ts_unix_nanos"] <= sizing_finished
            for row in sizing_rows
        ),
        "target sizing unexpectedly hit decode capacity pressure",
    )
    sizing_maintenance_summary = validate_maintenance_trace(
        sizing_rows,
        started_wall_ns=sizing_started,
        finished_wall_ns=sizing_finished,
        label="target sizing",
    )
    require(
        target_sizing.get("maintenance_summary") == sizing_maintenance_summary,
        "target sizing maintenance summary differs from raw trace",
    )
    target_rows = common.read_trace(root / str(target.get("trace")))
    target_trace_path = root / str(target.get("trace"))
    target_trace_bytes = target_trace_path.stat().st_size
    require(target_trace_bytes <= STOP_POLICY["max_trace_bytes"], "target trace exceeds its byte ceiling")
    require(
        rebalance_probe["client"]["prompt_tokens"]
        > max(result["prompt_tokens"] for result in target["clients"].values()),
        "rebalance probe did not increase request-shaped token demand",
    )
    for phase, started, finished in (
        ("target-rebalance-prime", prime_started, prime_finished),
        ("target-rebalance-probe", probe_started, probe_finished),
    ):
        require(
            not any(
                row.get("phase") == "vnext.decode_capacity_deferred"
                and isinstance(row.get("ts_unix_nanos"), int)
                and started <= row["ts_unix_nanos"] <= finished
                for row in target_rows
            ),
            f"{phase} unexpectedly hit decode capacity pressure",
        )
    decode_summary = validate_decode_trace(
        target_rows,
        started_wall_ns=target_started,
        finished_wall_ns=target_finished,
    )
    require(target.get("decode_summary") == decode_summary, "decode summary differs from raw trace")
    probe_maintenance_summary = validate_rebalance_trace(
        target_rows,
        started_wall_ns=probe_started,
        finished_wall_ns=probe_finished,
        label="target rebalance probe",
    )
    require(
        rebalance_probe.get("maintenance_summary") == probe_maintenance_summary,
        "target rebalance probe maintenance summary differs from raw trace",
    )

    counters = target_executor.get("counters")
    require(isinstance(counters, dict), "target executor counters are missing")
    counter_provenance = validate_decode_counter_provenance(
        target_rows,
        started_wall_ns=target_started,
        finished_wall_ns=target_finished,
        counters=counters,
    )
    require(target_executor.get("active_sequences") == 0, "target still has active sequences")
    require(target_executor.get("pending_sequences") == 0, "target still has pending sequences")
    require(target_executor.get("pending_prefill_maintenance") == 0, "target still has prefill maintenance")
    require(target_executor.get("executing_prefills") == 0, "target still has executing prefills")
    require(target_executor.get("staged_prefill_requests") == 0, "target still has staged prefill requests")
    require(target_executor.get("staged_prefill_sequences") == 0, "target still has staged prefill sequences")
    require(target_health.get("engine", {}).get("active_requests") == 0, "target engine still has active requests")
    require(target_health.get("engine", {}).get("queued_requests") == 0, "target engine still has queued requests")

    run_result = common.read_json(root / "run" / "result.json")
    require(run_result.get("returncode") == 0, "ferrum run smoke failed")
    run_rows = [
        json.loads(line)
        for line in (root / "run" / "stdout.jsonl").read_text().splitlines()
        if line.strip()
    ]
    require(any("paris" in str(row.get("content", "")).lower() for row in run_rows), "run smoke is not Paris")
    run_phases = {
        row.get("phase") for row in common.read_trace(root / "run" / "scheduler-trace.jsonl")
    }
    require("vnext.operation_submitted" in run_phases, "run smoke has no vNext submission")
    require("vnext.request_completed" in run_phases, "run smoke has no vNext completion")

    for path in [*root.rglob("*.log"), *root.rglob("*.sse")]:
        text = path.read_text(errors="replace").lower()
        require("\ufffd" not in text, f"Unicode replacement character in {path}")
        for pattern in common.FORBIDDEN_PATTERNS:
            require(pattern not in text, f"forbidden pattern {pattern!r} in {path}")

    pass_line = f"{PASS_PREFIX}: {out}"
    manifest = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s1_cuda_decode_capacity_validation",
        "status": "pass",
        "source_git_sha": source_git_sha,
        "binary_sha256": binary_sha256,
        "model_path": collection["model_path"],
        "source_artifact": str(root),
        "source_collection_sha256": common.sha256(root / "collection.json"),
        "exact_budget_bytes": exact_budget,
        "server_policy": SERVER_POLICY,
        "stop_policy": STOP_POLICY,
        "decode_summary": decode_summary,
        "rebalance_summary": probe_maintenance_summary,
        "rebalance_evidence_phase": "target-rebalance-probe",
        "sizing_maintenance_summary": sizing_maintenance_summary,
        "probe_maintenance_summary": probe_maintenance_summary,
        "cross_pool_rebalance_evidence_owner": (
            common.CROSS_POOL_REBALANCE_EVIDENCE_OWNER
        ),
        "target_trace_bytes": target_trace_bytes,
        "calibration_window_ns": [calibration_started, calibration_finished],
        "target_sizing_window_ns": [sizing_started, sizing_finished],
        "target_rebalance_prime_window_ns": [prime_started, prime_finished],
        "target_rebalance_probe_window_ns": [probe_started, probe_finished],
        "target_window_ns": [target_started, target_finished],
        "decode_counter_provenance": counter_provenance,
        "max_silence_seconds": {
            "calibration": calibration_silence,
            "target_sizing": sizing_silence,
            "target_rebalance_prime": prime_silence,
            "target_rebalance_probe": probe_silence,
            "target": target_silence,
        },
        "does_not_prove": ["S1", "G04", "performance", "release"],
        "pass_line": pass_line,
    }
    common.write_json(out / "validation.json", manifest)
    common.write_json(out / "manifest.json", manifest)
    print(pass_line)
    return 0


def self_test() -> int:
    require(
        CALIBRATION_TOKEN_BUDGET >= MAX_NUM_SEQS,
        "calibration token budget must be a valid product configuration",
    )
    require(
        CALIBRATION_MAX_TOKENS["A"] == TARGET_MAX_TOKENS["A"]
        and CALIBRATION_MAX_TOKENS["B"] < TARGET_MAX_TOKENS["B"]
        and CALIBRATION_MAX_TOKENS["C"] < TARGET_MAX_TOKENS["C"]
        and TARGET_MAX_TOKENS["B"] == TARGET_MAX_TOKENS["C"],
        "target must retain A while extending both pressure-victim lifetimes",
    )
    require(
        SERVER_POLICY["target_sizing_max_tokens"] == CALIBRATION_MAX_TOKENS,
        "target sizing must replay the narrow calibration workload",
    )
    require(
        SERVER_POLICY["sequence_fit_policy"] == DECODE_SEQUENCE_FIT_POLICY
        and DECODE_SEQUENCE_FIT_POLICY == "immediate-only",
        "decode capacity must exercise dynamic sequence extension",
    )
    require(
        SERVER_POLICY["target_rebalance_prime_max_tokens"] == CALIBRATION_MAX_TOKENS
        and SERVER_POLICY["target_rebalance_probe_max_tokens"] == 1
        and SERVER_POLICY["target_rebalance_probe_prompt_sha256"]
        == hashlib.sha256(REBALANCE_PROBE_PROMPT.encode("utf-8")).hexdigest(),
        "rebalance phases are not pinned to the canonical product workload",
    )
    require(
        SERVER_POLICY["decode_prompt_sha256_by_slot"]
        == CANONICAL_DECODE_PROMPT_SHA256_BY_SLOT
        == DECODE_PROMPT_SHA256_BY_SLOT,
        "decode prompts are not pinned to the decode-capacity lane",
    )
    require(
        DECODE_PROMPTS["B"] != common.capacity_prompt("B"),
        "decode capacity silently inherited the prefill capacity B prompt",
    )

    def expect_reject(action: Callable[[], None], label: str) -> None:
        try:
            action()
            raise AssertionError(f"self-test unexpectedly accepted {label}")
        except DecodeCapacityGateError:
            pass

    pressure_shape_clients = {
        "A": {"first_content_wall_ns": 100, "finished_wall_ns": 600},
        "B": {"first_content_wall_ns": 200, "finished_wall_ns": 700},
        "C": {"first_content_wall_ns": 300, "finished_wall_ns": 800},
    }
    require(
        require_pressure_decode_live_overlap(pressure_shape_clients, "self-test")
        == {
            "latest_pressure_first_content_wall_ns": 300,
            "earliest_pressure_completion_wall_ns": 600,
            "overlap_wall_ns": 300,
        },
        "pressure decode-live overlap receipt changed",
    )
    expect_reject(
        lambda: require_pressure_decode_live_overlap(
            {
                **pressure_shape_clients,
                "C": {"first_content_wall_ns": 650, "finished_wall_ns": 800},
            },
            "self-test non-overlap",
        ),
        "sequential decode workload",
    )
    prompt_result = {
        "workload_slot": "B",
        "prompt_sha256": DECODE_PROMPT_SHA256_BY_SLOT["B"],
    }
    require_decode_prompt(prompt_result, "B")
    expect_reject(
        lambda: require_decode_prompt(
            {**prompt_result, "prompt_sha256": "0" * 64}, "B"
        ),
        "decode prompt drift",
    )

    def executor_fixture(
        plan_digit: str, policy_digit: str, reserve_bytes: int
    ) -> dict[str, Any]:
        plan_hash = plan_digit * 64
        return {
            "model_id": "1" * 40,
            "family_fingerprint": "2" * 64,
            "program_fingerprint": "3" * 64,
            "runtime_fingerprint": "4" * 64,
            "policy_id": "policy.ferrum.product.vnext.default",
            "device_id": "device.cuda.0",
            "plan_id": f"plan/sha256/{plan_hash}",
            "plan_hash": plan_hash,
            "policy_fingerprint": policy_digit * 64,
            "runtime_memory_policy": {
                "capacity_bytes": 1000,
                "reserve_bytes": reserve_bytes,
                "maximum_active_sequences": MAX_NUM_SEQS,
            },
            "runtime_admission_policy": {
                "sequence_fit_policy": "immediate_only",
            },
        }

    calibration_executor = executor_fixture("a", "5", 100)
    sizing_executor = executor_fixture("b", "6", 200)
    target_executor = executor_fixture("c", "7", 300)
    validate_executor_identity_contract(
        {
            "calibration": [
                ("calibration start", calibration_executor),
                ("calibration final", dict(calibration_executor)),
            ],
            "target sizing": [
                ("target sizing start", sizing_executor),
                ("target sizing final", dict(sizing_executor)),
            ],
            "target": [
                ("target start", target_executor),
                ("target final", dict(target_executor)),
            ],
        }
    )
    drifted_target = dict(target_executor)
    drifted_target["plan_hash"] = "d" * 64
    drifted_target["plan_id"] = f"plan/sha256/{drifted_target['plan_hash']}"
    expect_reject(
        lambda: validate_executor_identity_contract(
            {
                "target": [
                    ("target start", target_executor),
                    ("target final", drifted_target),
                ]
            }
        ),
        "within-process plan drift",
    )
    malformed_plan = dict(target_executor)
    malformed_plan["plan_id"] = "plan/sha256/not-the-plan-hash"
    expect_reject(
        lambda: require_executor_identity_shape(malformed_plan, "malformed plan"),
        "plan id/hash mismatch",
    )
    wrong_fit_policy = dict(target_executor)
    wrong_fit_policy["runtime_admission_policy"] = {
        "sequence_fit_policy": "full_input_must_fit"
    }
    expect_reject(
        lambda: require_executor_identity_shape(
            wrong_fit_policy, "wrong decode fit policy"
        ),
        "prefill fit policy in decode lane",
    )

    canonical_argv = [
        "/tmp/ferrum",
        "serve",
        "/tmp/model",
        "--backend",
        "cuda",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-num-batched-tokens",
        str(CALIBRATION_TOKEN_BUDGET),
        "--sequence-fit-policy",
        DECODE_SEQUENCE_FIT_POLICY,
        "--scheduler-prefill-first-until-active",
        str(PREFILL_FIRST_UNTIL_ACTIVE),
    ]
    validate_canonical_server_argv(
        canonical_argv,
        label="self-test calibration",
        token_budget=CALIBRATION_TOKEN_BUDGET,
        runtime_budget=None,
    )
    expect_reject(
        lambda: validate_canonical_server_argv(
            canonical_argv + ["--max-num-seqs", str(MAX_NUM_SEQS)],
            label="self-test duplicate flag",
            token_budget=CALIBRATION_TOKEN_BUDGET,
            runtime_budget=None,
        ),
        "duplicate canonical option",
    )

    storage_profiles = {
        "sequence": {
            "allocator": {"fixed_block_arena": {"block_bytes": 1}},
            "view": {"paged_regions": {"block_bytes": 1}},
        },
        "workspace": {"allocator": "linear_arena", "view": "contiguous"},
    }
    pool_contracts = {
        "sequence": {
            "compatibility": {
                "version": {"major": 1, "minor": 0},
                "profile": storage_profiles["sequence"],
                "usage": "state",
                "element_type": "u8",
                "logical_layout_fingerprint": "a" * 64,
                "alignment_bytes": 1,
            },
            "resources": [
                {
                    "resource_id": "resource/sequence",
                    "demand": {
                        "tokens": {"bytes_per_token": 10, "maximum_tokens": 10}
                    },
                    "lifetime": "sequence",
                    "kind": "value",
                    "physical_allocation_quantum_bytes": 1,
                    "initialization": "none",
                }
            ],
            "minimum_request_bytes": 0,
            "minimum_sequence_bytes": 10,
            "minimum_step_bytes": 0,
            "minimum_invocation_peak_bytes": 0,
            "reusable_workspace_ceiling_bytes": 0,
            "provisioning": {
                "mode": "demand_driven_elastic",
                "minimum_resident_bytes": 10,
                "maximum_resident_bytes": 100,
            },
            "invocation_liveness_mode": "no_invocation_resources",
        },
        "workspace": {
            "compatibility": {
                "version": {"major": 1, "minor": 0},
                "profile": storage_profiles["workspace"],
                "usage": "scratch",
                "element_type": "u8",
                "logical_layout_fingerprint": "b" * 64,
                "alignment_bytes": 1,
            },
            "resources": [
                {
                    "resource_id": "resource/workspace",
                    "demand": {"fixed": {"bytes": 4}},
                    "lifetime": "invocation",
                    "kind": {"scratch": {"node_id": "node/workspace"}},
                    "physical_allocation_quantum_bytes": 1,
                    "initialization": "none",
                }
            ],
            "minimum_request_bytes": 0,
            "minimum_sequence_bytes": 0,
            "minimum_step_bytes": 0,
            "minimum_invocation_peak_bytes": 4,
            "reusable_workspace_ceiling_bytes": 0,
            "provisioning": {
                "mode": "demand_driven_elastic",
                "minimum_resident_bytes": 4,
                "maximum_resident_bytes": 100,
            },
            "invocation_liveness_mode": "total_order_reuse",
        },
    }
    calibration_contracts = json.loads(json.dumps(pool_contracts))
    calibration_contracts["sequence"]["resources"][0]["demand"]["tokens"][
        "maximum_tokens"
    ] = 3
    calibration_contracts["sequence"]["provisioning"]["maximum_resident_bytes"] = 90

    def pool_snapshot(
        pools: dict[str, int],
        contracts: dict[str, dict[str, Any]] = pool_contracts,
    ) -> dict[str, Any]:
        resident_bytes = sum(pools.values())
        return {
            "static_bytes": 100,
            "resident_bytes": resident_bytes,
            "budget_claimed_bytes": 100 + resident_bytes,
            "maximum_active_sequences": MAX_NUM_SEQS,
            "pool_resident_bytes": pools,
            "pool_envelopes": {
                pool_id: {
                    "resident_bytes": value,
                    "resident_chunks": 1,
                    "largest_contiguous_bytes": value,
                    "storage_profile": storage_profiles[pool_id],
                    "contract": contracts[pool_id],
                }
                for pool_id, value in pools.items()
            },
        }

    calibration_pool = pool_snapshot(
        {"sequence": 30, "workspace": 6}, calibration_contracts
    )
    sizing_pool = pool_snapshot({"sequence": 20, "workspace": 7})
    target_envelope = derive_target_budget_envelope(calibration_pool, sizing_pool)
    require(
        target_envelope["budget_claimed_bytes"] == 136
        and target_envelope["resident_bytes"] == 36,
        "self-test lost typed target budget derivation",
    )
    require(
        target_envelope["sizing_observed_pool_resident_bytes"]
        == {"sequence": 20, "workspace": 7}
        and target_envelope["observed_or_floor_pool_bytes"]
        == {"sequence": 30, "workspace": 7}
        and target_envelope["observed_or_floor_resident_bytes"] == 37
        and target_envelope["observed_or_floor_budget_gap_bytes"] == 1
        and target_envelope["requires_cross_pool_rebalance"] is True
        and target_envelope["initial_bundle_floor_bytes_by_pool"]
        == {"sequence": 30, "workspace": 4}
        and target_envelope["minimum_initial_bundle_resident_bytes"] == 34
        and target_envelope["initial_bundle_headroom_bytes"] == 2
        and target_envelope["token_scaled_sequence_pool_ids"] == ["sequence"]
        and target_envelope["pool_contracts"] == pool_contracts
        and target_envelope["bootstrap_headroom_bytes"] == 9,
        "self-test lost typed target-sizing provenance",
    )
    require_target_pool_within_budget_contract(
        pool_snapshot({"sequence": 32, "workspace": 4}), target_envelope, 136
    )
    prime_receipt = rebalance_prime_budget_receipt(
        pool_snapshot({"sequence": 31, "workspace": 4}),
        target_envelope,
        136,
    )
    require(
        prime_receipt
        == {
            "budget_ceiling_bytes": 136,
            "claimed_bytes": 135,
            "headroom_bytes": 1,
            "resident_ceiling_bytes": 36,
            "resident_bytes": 35,
        },
        "self-test lost bounded rebalance-prime headroom evidence",
    )
    try:
        require_target_pool_within_budget_contract(
            pool_snapshot({"sequence": 33, "workspace": 4}), target_envelope, 136
        )
        raise AssertionError("oversized global target residency unexpectedly fit its sizing envelope")
    except common.CapacityGateError:
        pass
    opaque_sizing = json.loads(json.dumps(sizing_pool))
    del opaque_sizing["pool_envelopes"]["sequence"]["contract"]
    expect_reject(
        lambda: derive_target_budget_envelope(calibration_pool, opaque_sizing),
        "opaque target sizing pool",
    )
    coefficient_drift = json.loads(json.dumps(sizing_pool))
    coefficient_drift["pool_envelopes"]["sequence"]["contract"]["resources"][0][
        "demand"
    ]["tokens"]["bytes_per_token"] = 11
    expect_reject(
        lambda: derive_target_budget_envelope(calibration_pool, coefficient_drift),
        "phase-stable token coefficient drift",
    )
    target_bound_drift = json.loads(
        json.dumps(pool_snapshot({"sequence": 32, "workspace": 4}))
    )
    target_bound_drift["pool_envelopes"]["sequence"]["contract"]["resources"][0][
        "demand"
    ]["tokens"]["maximum_tokens"] = 11
    expect_reject(
        lambda: require_target_pool_within_budget_contract(
            target_bound_drift, target_envelope, 136
        ),
        "target runtime-bound contract drift",
    )

    wait_condition = {
        "coordinator_id": 7,
        "observed": [{"source": {"domain": 5}, "epoch": 3}],
    }
    capacity_evidence = {
        "observed": {"coordinator_id": 7, "release_epoch": 11, "capacity_epoch": 13},
        "wait_condition": wait_condition,
    }

    def deferral(
        ts: int,
        decision: str,
        request_ids: list[str],
        *,
        victim_request_id: str | None = None,
        progress_owner_id: str | None = None,
        progress_baseline: int | None = None,
        episode_id: int | None = None,
        planned_transition_ordinal: int | None = None,
        yield_kind: str | None = None,
    ) -> dict[str, Any]:
        attributes = {
            "request_ids": request_ids,
            "capacity_evidence": capacity_evidence,
            "scheduler_snapshot": {},
        }
        if victim_request_id is not None:
            attributes["victim_request_id"] = victim_request_id
            attributes["progress_owner_id"] = progress_owner_id
            attributes["progress_baseline"] = progress_baseline
            attributes["episode_id"] = episode_id
            attributes["planned_transition_ordinal"] = planned_transition_ordinal
            attributes["yield_kind"] = yield_kind
        return {
            "ts_unix_nanos": ts,
            "phase": "vnext.decode_capacity_deferred",
            "status": "ok",
            "error": None,
            "shape": {
                "decision": decision,
                "attempted_decode_width": len(request_ids),
                "execution_stage": "step_admission",
                "decode_submit_observed": False,
            },
            "attributes": attributes,
        }

    rows = [
        deferral(100, "split_cohort", ["A", "B", "C"]),
        deferral(110, "wait_for_release", ["B"]),
        {
            "ts_unix_nanos": 120,
            "phase": "vnext.decode_capacity_skipped_unchanged",
            "status": "ok",
            "error": None,
            "request_id": "B",
            "shape": {
                "decision": "skipped_unchanged",
                "decode_submit_observed": False,
                "probe_performed": False,
            },
            "attributes": {
                "deferral_evidence": {
                    "action": "wait_for_release",
                    "observed": {"coordinator_id": 7, "release_epoch": 11, "capacity_epoch": 13},
                    "current": {"coordinator_id": 7, "release_epoch": 12, "capacity_epoch": 14},
                    "wait_condition": wait_condition,
                    "current_wait_sources": [{"source": {"domain": 5}, "epoch": 3}],
                }
            },
        },
        {
            "ts_unix_nanos": 130,
            "phase": "vnext.decode_capacity_resumed",
            "status": "ok",
            "error": None,
            "request_id": "request.product.B",
            "shape": {
                "decision": "exact_source_changed",
                "decode_submit_observed": False,
                "probe_performed": False,
                "exact_source_changed": True,
                "policy_epoch_changed": False,
            },
            "attributes": {
                "deferral_evidence": {
                    "action": "wait_for_release",
                    "observed": {"coordinator_id": 7, "release_epoch": 11, "capacity_epoch": 13},
                    "current": {"coordinator_id": 7, "release_epoch": 13, "capacity_epoch": 14},
                    "wait_condition": wait_condition,
                    "current_wait_sources": [{"source": {"domain": 5}, "epoch": 4}],
                }
            },
        },
        deferral(135, "wait_for_release", ["A"]),
        deferral(
            140,
            "pressure_yield_planned",
            ["C"],
            victim_request_id="C",
            progress_owner_id="A",
            progress_baseline=53,
            episode_id=1,
            planned_transition_ordinal=3,
            yield_kind="peer_handoff",
        ),
        {
            "ts_unix_nanos": 141,
            "phase": "vnext.execution_capacity_pressure_release_fence_armed",
            "status": "ok",
            "error": None,
            "request_id": "C",
            "shape": {
                "episode_id": 1,
                "planned_transition_ordinal": 3,
                "transition_ordinal": 4,
                "yield_kind": "peer_handoff",
                "physical_release_completed": False,
            },
            "attributes": {"progress_owner_id": "A"},
        },
        {
            "ts_unix_nanos": 142,
            "phase": "vnext.execution_capacity_pressure_release_fence_completed",
            "status": "ok",
            "error": None,
            "request_id": "C",
            "shape": {
                "episode_id": 1,
                "release_transition_ordinal": 5,
                "resumable_transition_ordinal": 6,
                "owner_admission_pending_transition_ordinal": None,
                "yield_kind": "peer_handoff",
                "physical_release_completed": True,
                "exact_source_advanced": True,
                "transaction_wait_condition_advanced": True,
                "release_authority": "active_sequence",
                "progress_owner_resumable": True,
                "closed_transition_ordinal": None,
                "closed_reason": None,
                "completion_disposition": "progress_owner_resumable",
                "victim_requeued": True,
            },
            "attributes": {
                "progress_owner_id": "A",
                "current_capacity_availability": [
                    {"source": {"domain": 5}, "epoch": 4}
                ],
            },
        },
        {
            "ts_unix_nanos": 145,
            "phase": "vnext.execution_capacity_pressure_hold_active",
            "status": "ok",
            "error": None,
            "request_id": "C",
            "shape": {
                "decision": "held_for_owner_progress",
                "episode_id": 1,
                "waiting_ticket": 1,
                "progress_owner_id": "A",
                "progress_baseline": 53,
                "progress_current": 53,
                "prefill_submit_observed": False,
                "probe_performed": False,
            },
        },
        deferral(
            146,
            "pressure_yield_planned",
            ["A"],
            victim_request_id="A",
            progress_owner_id="A",
            progress_baseline=54,
            episode_id=1,
            planned_transition_ordinal=7,
            yield_kind="self_recompute",
        ),
        {
            "ts_unix_nanos": 147,
            "phase": "vnext.execution_capacity_pressure_release_fence_armed",
            "status": "ok",
            "error": None,
            "request_id": "A",
            "shape": {
                "episode_id": 1,
                "planned_transition_ordinal": 7,
                "transition_ordinal": 8,
                "yield_kind": "self_recompute",
                "physical_release_completed": False,
            },
            "attributes": {"progress_owner_id": "A"},
        },
        {
            "ts_unix_nanos": 148,
            "phase": "vnext.execution_capacity_pressure_release_fence_completed",
            "status": "ok",
            "error": None,
            "request_id": "A",
            "shape": {
                "episode_id": 1,
                "release_transition_ordinal": 9,
                "resumable_transition_ordinal": None,
                "owner_admission_pending_transition_ordinal": 10,
                "closed_transition_ordinal": None,
                "closed_reason": None,
                "yield_kind": "self_recompute",
                "physical_release_completed": True,
                "exact_source_advanced": True,
                "transaction_wait_condition_advanced": True,
                "release_authority": "active_sequence",
                "progress_owner_resumable": False,
                "completion_disposition": "progress_owner_admission_pending",
                "victim_requeued": True,
            },
            "attributes": {
                "progress_owner_id": "A",
                "current_capacity_availability": [
                    {"source": {"domain": 5}, "epoch": 5}
                ],
            },
        },
        {
            "ts_unix_nanos": 149,
            "phase": "vnext.prefill_admission",
            "request_id": "A",
            "shape": {"decision": "admitted"},
        },
        {"ts_unix_nanos": 150, "phase": "vnext.request_completed", "request_id": "A"},
        {
            "ts_unix_nanos": 151,
            "phase": "vnext.execution_capacity_pressure_hold_released",
            "status": "ok",
            "error": None,
            "request_id": "C",
            "shape": {
                "decision": "owner_terminal",
                "episode_id": 1,
                "transition_ordinal": 12,
                "waiting_ticket": 1,
                "progress_owner_id": "A",
                "progress_baseline": 53,
                "progress_current": 53,
                "admission_eligible": True,
                "probe_performed": False,
                "prefill_submit_observed": False,
            },
        },
        {
            "ts_unix_nanos": 152,
            "phase": "vnext.prefill_admission",
            "request_id": "C",
            "shape": {"decision": "admitted"},
        },
        {"ts_unix_nanos": 153, "phase": "vnext.request_completed", "request_id": "C"},
        {"ts_unix_nanos": 154, "phase": "vnext.request_completed", "request_id": "B"},
    ]
    rows.extend(
        [
            deferral(
                160,
                "pressure_yield_planned",
                ["D"],
                victim_request_id="D",
                progress_owner_id="D",
                progress_baseline=21,
                episode_id=2,
                planned_transition_ordinal=13,
                yield_kind="self_recompute",
            ),
            {
                "ts_unix_nanos": 161,
                "phase": "vnext.execution_capacity_pressure_release_fence_armed",
                "status": "ok",
                "error": None,
                "request_id": "D",
                "shape": {
                    "episode_id": 2,
                    "planned_transition_ordinal": 13,
                    "transition_ordinal": 14,
                    "yield_kind": "self_recompute",
                    "physical_release_completed": False,
                },
                "attributes": {"progress_owner_id": "D"},
            },
            {
                "ts_unix_nanos": 162,
                "phase": "vnext.execution_capacity_pressure_release_fence_completed",
                "status": "ok",
                "error": None,
                "request_id": "D",
                "shape": {
                    "episode_id": 2,
                    "release_transition_ordinal": 15,
                    "resumable_transition_ordinal": None,
                    "owner_admission_pending_transition_ordinal": None,
                    "closed_transition_ordinal": 16,
                    "closed_reason": None,
                    "yield_kind": "self_recompute",
                    "physical_release_completed": True,
                    "exact_source_advanced": True,
                    "transaction_wait_condition_advanced": True,
                    "release_authority": "active_sequence",
                    "progress_owner_resumable": False,
                    "completion_disposition": "self_recompute_queued",
                    "victim_requeued": True,
                },
                "attributes": {
                    "progress_owner_id": "D",
                    "current_capacity_availability": [
                        {"source": {"domain": 5}, "epoch": 4}
                    ],
                },
            },
            {
                "ts_unix_nanos": 163,
                "phase": "vnext.prefill_admission",
                "request_id": "D",
                "shape": {"decision": "admitted"},
            },
            {"ts_unix_nanos": 164, "phase": "vnext.request_completed", "request_id": "D"},
        ]
    )
    rebalance_rows = [
        {
            "ts_unix_nanos": 85,
            "phase": "vnext.prefill_backing_maintenance",
            "status": "ok",
            "error": None,
            "shape": {"outcome": "maintained"},
            "attributes": {
                "maintenance_evidence": {
                    "outcome": "maintained",
                    "current": {
                        "coordinator_id": 7,
                        "release_epoch": 15,
                        "capacity_epoch": 17,
                    },
                    "pools_grown": 1,
                    "allocated_bytes": 32,
                    "pools_reclaimed": 1,
                    "chunks_reclaimed": 1,
                    "reclaimed_bytes": 64,
                    "rebalance": {
                        "pools": [
                            {
                                "pool_id": "dynamic-pool/sha256/" + "a" * 64,
                                "chunks": [
                                    {
                                        "pool_id": (
                                            "dynamic-pool/sha256/" + "a" * 64
                                        ),
                                        "ordinal": 1,
                                        "generation": 2,
                                    }
                                ],
                                "reclaimed_bytes": 64,
                                "published_capacity_bytes": 128,
                            }
                        ],
                        "reclaimed_chunks": 1,
                        "reclaimed_bytes": 64,
                        "logical_capacity_epoch": 15,
                        "plan_device_capacity_epoch": 16,
                        "process_device_capacity_epoch": 17,
                    },
                }
            },
        }
    ]
    summary = validate_decode_trace(rows, started_wall_ns=90, finished_wall_ns=170)
    direct_counter_provenance = validate_decode_counter_provenance(
        rows,
        started_wall_ns=90,
        finished_wall_ns=170,
        counters={
            "extension_deferrals": 0,
            "step_deferrals": summary["deferral_events"],
            "wave_deferrals": 0,
            "backing_deferrals": 0,
        },
    )
    require(
        direct_counter_provenance["direct_trace_events_by_stage"]["step_admission"]
        == summary["deferral_events"],
        "self-test lost direct step-admission counter provenance",
    )
    backing_rows = json.loads(json.dumps(rows))
    for row in backing_rows:
        if row.get("phase") != "vnext.decode_capacity_deferred":
            continue
        row["attributes"]["capacity_evidence"]["wait_condition"]["observed"].append(
            {"source": "plan_device_budget", "epoch": 9}
        )
    backing_counter_provenance = validate_decode_counter_provenance(
        backing_rows,
        started_wall_ns=90,
        finished_wall_ns=170,
        counters={
            "extension_deferrals": 0,
            "step_deferrals": 0,
            "wave_deferrals": 0,
            "backing_deferrals": summary["deferral_events"],
        },
    )
    require(
        backing_counter_provenance["device_backing_trace_events"]
        == summary["deferral_events"],
        "self-test lost device-backing counter provenance",
    )
    expect_reject(
        lambda: validate_decode_counter_provenance(
            backing_rows,
            started_wall_ns=90,
            finished_wall_ns=170,
            counters={
                "extension_deferrals": 0,
                "step_deferrals": 0,
                "wave_deferrals": 0,
                "backing_deferrals": 0,
            },
        ),
        "missing backing deferral counter provenance",
    )
    rebalance_summary = validate_rebalance_trace(
        rebalance_rows, started_wall_ns=80, finished_wall_ns=89
    )
    require(
        rebalance_summary["rebalance_events"] == 1
        and rebalance_summary["reclaimed_bytes"] == 64,
        "self-test lost typed cross-pool rebalance evidence",
    )
    require(
        rebalance_summary["exact_receipt"] is True
        and rebalance_summary["evidence_owner"]
        == common.CROSS_POOL_REBALANCE_EVIDENCE_OWNER
        and rebalance_summary["receipts"]
        == [
            {
                "pool_ids": ["dynamic-pool/sha256/" + "a" * 64],
                "chunk_identities": [
                    ["dynamic-pool/sha256/" + "a" * 64, 1, 2]
                ],
                "capacity_epochs": {
                    "logical_capacity_epoch": 15,
                    "plan_device_capacity_epoch": 16,
                    "process_device_capacity_epoch": 17,
                },
            }
        ],
        "self-test lost the exact pool, chunk, or epoch receipt",
    )
    missing_rebalance = json.loads(json.dumps(rebalance_rows))
    missing_rebalance[-1]["attributes"]["maintenance_evidence"].update(
        {
            "pools_reclaimed": 0,
            "chunks_reclaimed": 0,
            "reclaimed_bytes": 0,
            "rebalance": None,
        }
    )
    growth_only_summary = validate_maintenance_trace(
        missing_rebalance,
        started_wall_ns=80,
        finished_wall_ns=89,
        label="self-test growth-only probe",
    )
    require(
        growth_only_summary["maintained_events"] == 1
        and growth_only_summary["allocated_bytes"] == 32
        and growth_only_summary["rebalance_events"] == 0
        and growth_only_summary["exact_receipt"] is False
        and growth_only_summary["evidence_owner"] is None
        and growth_only_summary["receipts"] == [],
        "self-test lost growth-only maintenance evidence",
    )
    try:
        validate_rebalance_trace(
            missing_rebalance, started_wall_ns=80, finished_wall_ns=89
        )
        raise AssertionError("trace without cross-pool reclaim unexpectedly passed")
    except common.CapacityGateError:
        pass
    missing_exact_receipt = json.loads(json.dumps(rebalance_rows))
    missing_exact_receipt[-1]["attributes"]["maintenance_evidence"][
        "rebalance"
    ] = None
    try:
        validate_rebalance_trace(
            missing_exact_receipt, started_wall_ns=80, finished_wall_ns=89
        )
        raise AssertionError("aggregate-only cross-pool reclaim unexpectedly passed")
    except common.CapacityGateError:
        pass
    invalid_rebalance = json.loads(json.dumps(rebalance_rows))
    invalid_rebalance[-1]["attributes"]["maintenance_evidence"]["chunks_reclaimed"] = 0
    try:
        validate_rebalance_trace(
            invalid_rebalance, started_wall_ns=80, finished_wall_ns=89
        )
        raise AssertionError("inconsistent cross-pool reclaim unexpectedly passed")
    except common.CapacityGateError:
        pass
    invalid_chunk_identity = json.loads(json.dumps(rebalance_rows))
    invalid_chunk_identity[-1]["attributes"]["maintenance_evidence"]["rebalance"][
        "pools"
    ][0]["chunks"][0]["pool_id"] = "dynamic-pool/sha256/" + "b" * 64
    try:
        validate_rebalance_trace(
            invalid_chunk_identity, started_wall_ns=80, finished_wall_ns=89
        )
        raise AssertionError("mismatched exact chunk identity unexpectedly passed")
    except common.CapacityGateError:
        pass
    require(summary["split_events"] == 1, "self-test lost split evidence")
    require(summary["park_events"] == 2, "self-test lost park evidence")
    require(summary["resume_events"] == 1, "self-test lost resume evidence")
    require(summary["pressure_yield_events"] == 3, "self-test lost pressure-yield evidence")
    require(
        summary["pressure_yield_kinds"] == ["peer_handoff", "self_recompute"],
        "self-test lost typed yield strategies",
    )
    require(
        summary["pressure_fence_armed_events"] == 3
        and summary["pressure_fence_completed_events"] == 3,
        "self-test lost release-fence evidence",
    )
    require(
        summary["pressure_hold_events"] == 1
        and summary["pressure_hold_release_events"] == 1,
        "self-test lost pressure-hold evidence",
    )
    require(
        summary["pressure_victim_request_ids"] == ["A", "C", "D"],
        "self-test lost pressure-victim identity",
    )

    retargeted_completion = json.loads(json.dumps(rows[7]))
    retargeted_completion["shape"].update(
        {
            "resumable_transition_ordinal": None,
            "progress_owner_resumable": False,
            "closed_transition_ordinal": 6,
            "closed_reason": "source_retargeted",
            "completion_disposition": "source_retargeted",
        }
    )
    try:
        validate_pressure_fence_completed(
            retargeted_completion, "obsolete retargeted fence self-test"
        )
        raise AssertionError("source-retargeted completion unexpectedly passed")
    except common.CapacityGateError:
        pass

    unchanged_resume = json.loads(json.dumps(rows))
    unchanged_resume[3]["attributes"]["deferral_evidence"]["current_wait_sources"][0]["epoch"] = 3
    try:
        validate_decode_trace(unchanged_resume, started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("unchanged exact source unexpectedly resumed")
    except common.CapacityGateError:
        pass
    try:
        validate_decode_trace(rows[1:], started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("trace without adaptive split unexpectedly passed")
    except common.CapacityGateError:
        pass
    missing_victim = json.loads(json.dumps(rows))
    del missing_victim[5]["attributes"]["victim_request_id"]
    try:
        validate_decode_trace(missing_victim, started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("pressure yield without a victim unexpectedly passed")
    except common.CapacityGateError:
        pass
    missing_hold = json.loads(json.dumps(rows))
    del missing_hold[8]
    try:
        validate_decode_trace(missing_hold, started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("pressure yield without a hold unexpectedly passed")
    except common.CapacityGateError:
        pass
    missing_release = json.loads(json.dumps(rows))
    del missing_release[14]
    try:
        validate_decode_trace(missing_release, started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("pressure hold without a release unexpectedly passed")
    except common.CapacityGateError:
        pass
    missing_owner_terminal = json.loads(json.dumps(rows))
    del missing_owner_terminal[13]
    try:
        validate_decode_trace(
            missing_owner_terminal, started_wall_ns=90, finished_wall_ns=170
        )
        raise AssertionError("live owner unexpectedly released a pressure hold")
    except common.CapacityGateError:
        pass
    stale_hold = json.loads(json.dumps(rows))
    stale_hold_event = json.loads(json.dumps(stale_hold[8]))
    stale_hold_event["ts_unix_nanos"] = 152
    stale_hold.append(stale_hold_event)
    try:
        validate_decode_trace(stale_hold, started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("victim remained held after owner terminal")
    except common.CapacityGateError:
        pass
    premature_readmission = json.loads(json.dumps(rows))
    premature_readmission[15]["ts_unix_nanos"] = 149
    try:
        validate_decode_trace(premature_readmission, started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("victim re-admitted before owner terminal")
    except common.CapacityGateError:
        pass
    rotated_owner = json.loads(json.dumps(rows))
    rotated_owner[9]["shape"]["attempted_decode_width"] = 1
    rotated_owner[9]["attributes"].update(
        {
            "request_ids": ["C"],
            "victim_request_id": "C",
            "progress_owner_id": "C",
        }
    )
    rotated_owner[10]["request_id"] = "C"
    rotated_owner[10]["attributes"]["progress_owner_id"] = "C"
    rotated_owner[11]["request_id"] = "C"
    rotated_owner[11]["attributes"]["progress_owner_id"] = "C"
    try:
        validate_decode_trace(rotated_owner, started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("held victim unexpectedly became the pressure owner")
    except common.CapacityGateError:
        pass
    for obsolete_reason in ["role_transferred", "source_retargeted"]:
        obsolete_release = json.loads(json.dumps(rows[14]))
        obsolete_release["shape"]["decision"] = obsolete_reason
        try:
            validate_pressure_hold_release(
                obsolete_release, f"obsolete {obsolete_reason} hold release"
            )
            raise AssertionError(f"{obsolete_reason} hold release unexpectedly passed")
        except common.CapacityGateError:
            pass
    print("FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--repo", type=Path, default=Path.cwd())
    collect_parser.add_argument("--binary", type=Path, required=True)
    collect_parser.add_argument("--model", type=Path, required=True)
    collect_parser.add_argument("--out", type=Path, required=True)
    collect_parser.add_argument("--port", type=int, default=18130)
    collect_parser.add_argument("--startup-timeout", type=float, default=600)
    collect_parser.add_argument("--request-timeout", type=float, default=300)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("artifact", type=Path)
    validate_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test()
        if args.command == "collect":
            return collect(args)
        if args.command == "validate":
            return validate(args.artifact, args.out)
        parser.error("a command is required")
    except common.CapacityGateError as error:
        target = getattr(args, "out", Path("."))
        print(f"{FAIL_PREFIX}: {target}: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
