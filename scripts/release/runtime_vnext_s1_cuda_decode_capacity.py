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
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_CACHE_COMPONENT = "models--Qwen--Qwen3.5-4B"
CALIBRATION_TOKEN_BUDGET = 3
TARGET_TOKEN_BUDGET = 1024
MAX_NUM_SEQS = 3
MAX_MODEL_LEN = 2048
PREFILL_FIRST_UNTIL_ACTIVE = 3
DECODE_SEQUENCE_FIT_POLICY = "immediate-only"
CALIBRATION_MAX_TOKENS = {"A": 128, "B": 1, "C": 16}
TARGET_MAX_TOKENS = {"A": 1536, "B": 1536, "C": 1536}
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
PREFILL_MAINTENANCE_PHASE = "vnext.prefill_backing_maintenance"
EXECUTION_MAINTENANCE_PHASE = "vnext.execution_backing_maintenance"
EXECUTION_MAINTENANCE_SCHEMA_VERSION = 2
MAINTENANCE_BOUNDARY_SCHEMA_VERSION = 1
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
    "target_sizing_runtime_budget": "product_default",
    "target_budget_derivation": "unpressured_typed_prime_probe_growth_with_exact_target_rebalance",
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


def model_revision_from_snapshot(path: object) -> str:
    parts = Path(str(path)).parts
    require(
        MODEL_CACHE_COMPONENT in parts,
        f"model snapshot is not {MODEL_ID}: {path}",
    )
    require("snapshots" in parts, f"model path is not a Hugging Face snapshot: {path}")
    snapshot_index = parts.index("snapshots")
    require(
        snapshot_index + 1 < len(parts),
        f"model snapshot revision is missing: {path}",
    )
    revision = parts[snapshot_index + 1]
    require(
        common.GIT_SHA_RE.fullmatch(revision) is not None,
        f"invalid model snapshot revision: {revision}",
    )
    return revision


def require_executor_identity_shape(executor: dict[str, Any], label: str) -> None:
    require(
        executor.get("model_id") == MODEL_ID,
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


def runtime_memory_usable_bytes(policy: Any, label: str) -> int:
    require(isinstance(policy, dict), f"{label}: runtime memory policy is missing")
    capacity = policy.get("capacity_bytes")
    reserve = policy.get("reserve_bytes")
    require(
        isinstance(capacity, int)
        and not isinstance(capacity, bool)
        and capacity > 0
        and isinstance(reserve, int)
        and not isinstance(reserve, bool)
        and 0 <= reserve < capacity,
        f"{label}: runtime memory capacity/reserve is invalid",
    )
    maximum_active_sequences = policy.get("maximum_active_sequences")
    require(
        maximum_active_sequences == MAX_NUM_SEQS,
        f"{label}: runtime memory sequence ceiling changed",
    )
    profiles = policy.get("dynamic_storage_profile_order")
    require(
        isinstance(profiles, list) and profiles,
        f"{label}: runtime memory storage-profile order is missing",
    )
    return capacity - reserve


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
            f"{label}: unexpectedly overrides runtime memory budget",
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


def budget_stable_pool_contract(contract: dict[str, Any]) -> dict[str, Any]:
    provisioning = dict(contract["provisioning"])
    provisioning.pop("maximum_resident_bytes", None)
    return {**contract, "provisioning": provisioning}


def phase_stable_pool_contract(contract: dict[str, Any]) -> dict[str, Any]:
    stable = budget_stable_pool_contract(contract)
    return {
        **stable,
        "resources": [
            {
                **resource,
                "demand": phase_stable_demand_contract(resource["demand"]),
            }
            for resource in contract["resources"]
        ],
    }


def growth_replay_signature(
    receipts: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for receipt_index, receipt in enumerate(receipts):
        growths = receipt.get("growths") if isinstance(receipt, dict) else None
        require(
            receipt.get("stage") in ALLOWED_EXECUTION_STAGES
            and isinstance(receipt.get("allocated_bytes"), int)
            and not isinstance(receipt.get("allocated_bytes"), bool)
            and isinstance(growths, list)
            and growths,
            f"{label}: growth receipt {receipt_index} is incomplete",
        )
        normalized_growths = [
            {
                "pool_id": growth.get("pool_id") if isinstance(growth, dict) else None,
                "chunk_bytes": (
                    growth.get("chunk_bytes") if isinstance(growth, dict) else None
                ),
            }
            for growth in growths
        ]
        require(
            all(
                isinstance(growth["pool_id"], str)
                and isinstance(growth["chunk_bytes"], int)
                and not isinstance(growth["chunk_bytes"], bool)
                and growth["chunk_bytes"] > 0
                for growth in normalized_growths
            )
            and sum(growth["chunk_bytes"] for growth in normalized_growths)
            == receipt["allocated_bytes"],
            f"{label}: growth receipt {receipt_index} byte signature drifted",
        )
        signature.append(
            {
                "stage": receipt["stage"],
                "allocated_bytes": receipt["allocated_bytes"],
                "growths": normalized_growths,
            }
        )
    return signature


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
    calibration: dict[str, Any],
    target_sizing: dict[str, Any],
    target_probe: dict[str, Any],
    probe_maintenance: dict[str, Any],
) -> dict[str, Any]:
    static_bytes = target_sizing.get("static_bytes")
    require(isinstance(static_bytes, int) and static_bytes > 0, "invalid sizing static bytes")
    require(
        static_bytes == calibration.get("static_bytes") == target_probe.get("static_bytes"),
        "target sizing static bytes differ across calibration, prime, and probe",
    )
    calibration_pools = calibration.get("pool_resident_bytes")
    sizing_pools = target_sizing.get("pool_resident_bytes")
    probe_pools = target_probe.get("pool_resident_bytes")
    calibration_envelopes = calibration.get("pool_envelopes")
    sizing_envelopes = target_sizing.get("pool_envelopes")
    probe_envelopes = target_probe.get("pool_envelopes")
    require(
        isinstance(calibration_pools, dict)
        and isinstance(sizing_pools, dict)
        and isinstance(probe_pools, dict)
        and calibration_pools.keys() == sizing_pools.keys() == probe_pools.keys(),
        "target sizing pool identities differ across calibration, prime, and probe",
    )
    require(
        isinstance(calibration_envelopes, dict)
        and isinstance(sizing_envelopes, dict)
        and isinstance(probe_envelopes, dict)
        and calibration_envelopes.keys()
        == sizing_envelopes.keys()
        == probe_envelopes.keys(),
        "target sizing pool envelopes differ across calibration, prime, and probe",
    )
    maximum_active_sequences = target_sizing.get("maximum_active_sequences")
    require(
        maximum_active_sequences
        == calibration.get("maximum_active_sequences")
        == target_probe.get("maximum_active_sequences")
        == MAX_NUM_SEQS,
        "target sizing typed sequence ceiling differs from the canonical workload",
    )

    growth_receipts = probe_maintenance.get("growth_receipts")
    require(
        isinstance(growth_receipts, list)
        and growth_receipts
        and probe_maintenance.get("maintained_events") == len(growth_receipts),
        "target sizing probe has no exact typed growth receipts",
    )
    require(
        probe_maintenance.get("rebalance_events") == 0
        and probe_maintenance.get("pools_reclaimed") == 0
        and probe_maintenance.get("chunks_reclaimed") == 0
        and probe_maintenance.get("reclaimed_bytes") == 0,
        "target sizing probe is not an unpressured physical-growth baseline",
    )
    trace_growth_bytes_by_pool: dict[str, int] = {}
    sequence_extension_growth_bytes_by_pool: dict[str, int] = {}
    sequence_extension_growth_chunks_by_pool: dict[str, list[int]] = {}
    trace_reclaimed_bytes_by_pool: dict[str, int] = {}
    trace_growth_chunks: list[dict[str, Any]] = []
    for receipt_index, receipt in enumerate(growth_receipts):
        growths = receipt.get("growths") if isinstance(receipt, dict) else None
        reclaims = receipt.get("reclaims") if isinstance(receipt, dict) else None
        stage = receipt.get("stage") if isinstance(receipt, dict) else None
        require(
            stage in ALLOWED_EXECUTION_STAGES
            and isinstance(growths, list)
            and growths
            and isinstance(reclaims, list),
            f"target sizing probe growth receipt {receipt_index} is empty",
        )
        receipt_allocated_bytes = 0
        for growth_index, growth in enumerate(growths):
            pool_id = growth.get("pool_id") if isinstance(growth, dict) else None
            chunk_bytes = growth.get("chunk_bytes") if isinstance(growth, dict) else None
            require(
                isinstance(pool_id, str)
                and pool_id in sizing_pools
                and isinstance(chunk_bytes, int)
                and not isinstance(chunk_bytes, bool)
                and chunk_bytes > 0,
                f"target sizing probe growth {receipt_index}/{growth_index} is invalid",
            )
            trace_growth_bytes_by_pool[pool_id] = (
                trace_growth_bytes_by_pool.get(pool_id, 0) + chunk_bytes
            )
            if stage == "sequence_extension":
                sequence_extension_growth_bytes_by_pool[pool_id] = (
                    sequence_extension_growth_bytes_by_pool.get(pool_id, 0)
                    + chunk_bytes
                )
                sequence_extension_growth_chunks_by_pool.setdefault(pool_id, []).append(
                    chunk_bytes
                )
            receipt_allocated_bytes += chunk_bytes
            trace_growth_chunks.append(growth)
        require(
            receipt.get("allocated_bytes") == receipt_allocated_bytes,
            f"target sizing probe growth receipt {receipt_index} byte total drifted",
        )
        receipt_reclaimed_bytes = 0
        for reclaim_index, reclaim in enumerate(reclaims):
            pool_id = reclaim.get("pool_id") if isinstance(reclaim, dict) else None
            reclaimed_bytes = (
                reclaim.get("reclaimed_bytes") if isinstance(reclaim, dict) else None
            )
            require(
                isinstance(pool_id, str)
                and pool_id in sizing_pools
                and isinstance(reclaimed_bytes, int)
                and not isinstance(reclaimed_bytes, bool)
                and reclaimed_bytes > 0,
                f"target sizing probe reclaim {receipt_index}/{reclaim_index} is invalid",
            )
            trace_reclaimed_bytes_by_pool[pool_id] = (
                trace_reclaimed_bytes_by_pool.get(pool_id, 0) + reclaimed_bytes
            )
            receipt_reclaimed_bytes += reclaimed_bytes
        require(
            receipt.get("reclaimed_bytes") == receipt_reclaimed_bytes,
            f"target sizing probe reclaim receipt {receipt_index} byte total drifted",
        )
    require(
        probe_maintenance.get("allocated_bytes")
        == sum(trace_growth_bytes_by_pool.values()),
        "target sizing probe aggregate growth bytes differ from exact receipts",
    )
    require(
        probe_maintenance.get("reclaimed_bytes")
        == sum(trace_reclaimed_bytes_by_pool.values()),
        "target sizing probe aggregate reclaimed bytes differ from exact receipts",
    )

    sizing_pool_bytes: dict[str, int] = {}
    probe_pool_bytes: dict[str, int] = {}
    probe_growth_bytes_by_pool: dict[str, int] = {}
    probe_shrink_bytes_by_pool: dict[str, int] = {}
    initial_bundle_floor_bytes_by_pool: dict[str, int] = {}
    pool_storage_profiles: dict[str, Any] = {}
    pool_contracts: dict[str, Any] = {}
    token_scaled_sequence_pool_ids: list[str] = []
    token_scaled_sequence_quanta: dict[str, int] = {}
    for pool_id in sorted(calibration_pools):
        calibration_bytes = calibration_pools[pool_id]
        sizing_bytes = sizing_pools[pool_id]
        probe_bytes = probe_pools[pool_id]
        require(
            all(
                isinstance(value, int) and value >= 0
                for value in (calibration_bytes, sizing_bytes, probe_bytes)
            ),
            f"invalid sizing residency for {pool_id}",
        )
        calibration_profile = calibration_envelopes[pool_id].get("storage_profile")
        sizing_profile = sizing_envelopes[pool_id].get("storage_profile")
        probe_profile = probe_envelopes[pool_id].get("storage_profile")
        require(
            calibration_profile == sizing_profile == probe_profile,
            f"target sizing storage profile differs for {pool_id}",
        )
        calibration_contract = validate_typed_pool_contract(
            pool_id, calibration_envelopes[pool_id], "calibration"
        )
        sizing_contract = validate_typed_pool_contract(
            pool_id, sizing_envelopes[pool_id], "target sizing prime"
        )
        probe_contract = validate_typed_pool_contract(
            pool_id, probe_envelopes[pool_id], "target sizing probe"
        )
        stable_contract = phase_stable_pool_contract(sizing_contract)
        require(
            phase_stable_pool_contract(calibration_contract) == stable_contract,
            f"calibration and target sizing phase-stable contract differ for {pool_id}",
        )
        require(
            probe_contract == sizing_contract,
            f"target sizing prime/probe typed contract differs for {pool_id}",
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
        require(
            initial_bundle_floor
            <= sizing_contract["provisioning"]["maximum_resident_bytes"],
            f"typed initial bundle floor exceeds the pool ceiling for {pool_id}",
        )
        sequence_resources = [
            resource
            for resource in sizing_contract["resources"]
            if resource.get("lifetime") == "sequence"
        ]
        token_scaled_sequence_resources = [
            resource
            for resource in sequence_resources
            if demand_is_token_scaled(resource.get("demand"))
        ]
        if token_scaled_sequence_resources:
            require(
                len(token_scaled_sequence_resources) == len(sequence_resources),
                f"target sizing cannot attribute mixed sequence demand in {pool_id}",
            )
            quanta = {
                resource["physical_allocation_quantum_bytes"]
                for resource in token_scaled_sequence_resources
            }
            require(
                len(quanta) == 1,
                f"target sizing cannot attribute mixed sequence quanta in {pool_id}",
            )
            token_scaled_sequence_pool_ids.append(pool_id)
            token_scaled_sequence_quanta[pool_id] = next(iter(quanta))

        growth_bytes = max(0, probe_bytes - sizing_bytes)
        shrink_bytes = max(0, sizing_bytes - probe_bytes)
        sizing_pool_bytes[pool_id] = sizing_bytes
        probe_pool_bytes[pool_id] = probe_bytes
        probe_growth_bytes_by_pool[pool_id] = growth_bytes
        probe_shrink_bytes_by_pool[pool_id] = shrink_bytes
        initial_bundle_floor_bytes_by_pool[pool_id] = initial_bundle_floor
        pool_storage_profiles[pool_id] = sizing_profile
        pool_contracts[pool_id] = sizing_contract
    require(
        token_scaled_sequence_pool_ids,
        "typed target sizing contains no token-scaled sequence resource",
    )

    for pool_id in sizing_pool_bytes:
        require(
            probe_pool_bytes[pool_id]
            == sizing_pool_bytes[pool_id]
            + trace_growth_bytes_by_pool.get(pool_id, 0)
            - trace_reclaimed_bytes_by_pool.get(pool_id, 0),
            f"target sizing probe pool conservation failed for {pool_id}",
        )

    sizing_resident_bytes = target_sizing.get("resident_bytes")
    probe_resident_bytes = target_probe.get("resident_bytes")
    calibration_budget = calibration.get("budget_claimed_bytes")
    calibration_resident_bytes = calibration.get("resident_bytes")
    require(
        isinstance(sizing_resident_bytes, int)
        and sizing_resident_bytes == sum(sizing_pool_bytes.values())
        and target_sizing.get("budget_claimed_bytes")
        == static_bytes + sizing_resident_bytes,
        "target sizing prime budget differs from installed backing",
    )
    require(
        isinstance(probe_resident_bytes, int)
        and probe_resident_bytes == sum(probe_pool_bytes.values())
        and target_probe.get("budget_claimed_bytes")
        == static_bytes + probe_resident_bytes,
        "target sizing probe budget differs from installed backing",
    )
    require(
        probe_resident_bytes
        == sizing_resident_bytes
        + sum(trace_growth_bytes_by_pool.values())
        - sum(trace_reclaimed_bytes_by_pool.values()),
        "target sizing probe total residency does not conserve exact maintenance bytes",
    )
    require(
        isinstance(calibration_budget, int)
        and isinstance(calibration_resident_bytes, int)
        and calibration_resident_bytes == sum(calibration_pools.values())
        and calibration_budget == static_bytes + calibration_resident_bytes,
        "calibration budget differs from its installed backing",
    )
    minimum_initial_bundle_resident_bytes = sum(
        initial_bundle_floor_bytes_by_pool.values()
    )
    require(
        minimum_initial_bundle_resident_bytes <= calibration_resident_bytes,
        "typed initial bundles do not fit the calibrated resident budget",
    )

    sizing_growth_replay_signature = growth_replay_signature(
        growth_receipts, "target sizing probe"
    )
    selected_pressure_event_ordinal: int | None = None
    selected_pressure_growth_pool_ids: list[str] = []
    selected_pressure_growth_end_bytes = 0
    growth_prefix_bytes = 0
    for receipt_ordinal, signature in enumerate(sizing_growth_replay_signature):
        all_event_growth_pool_ids = {
            growth["pool_id"] for growth in signature["growths"]
        }
        event_growth_pool_ids = sorted(
            {
                growth["pool_id"]
                for growth in signature["growths"]
                if growth["pool_id"] in token_scaled_sequence_pool_ids
            }
        )
        if signature["stage"] == "sequence_extension" and event_growth_pool_ids:
            require(
                all_event_growth_pool_ids.issubset(token_scaled_sequence_pool_ids),
                "target sizing sequence pressure event contains unattributed pool growth",
            )
            for pool_id in event_growth_pool_ids:
                quantum = token_scaled_sequence_quanta[pool_id]
                pool_growths = [
                    growth["chunk_bytes"]
                    for growth in signature["growths"]
                    if growth["pool_id"] == pool_id
                ]
                require(
                    sum(pool_growths) >= quantum
                    and all(chunk_bytes % quantum == 0 for chunk_bytes in pool_growths),
                    f"target sizing pressure event has non-quantized growth for {pool_id}",
                )
            selected_pressure_event_ordinal = receipt_ordinal
            selected_pressure_growth_pool_ids = event_growth_pool_ids
            selected_pressure_growth_end_bytes = (
                growth_prefix_bytes + signature["allocated_bytes"]
            )
            break
        growth_prefix_bytes += signature["allocated_bytes"]
    require(
        selected_pressure_event_ordinal is not None
        and selected_pressure_growth_pool_ids,
        "long probe has no attributable sequence-extension growth event",
    )
    pressure_quantum_bytes = max(
        token_scaled_sequence_quanta[pool_id]
        for pool_id in selected_pressure_growth_pool_ids
    )
    probe_positive_growth_bytes = sum(trace_growth_bytes_by_pool.values())
    token_scaled_sequence_growth_bytes = sum(
        growth["chunk_bytes"]
        for growth in sizing_growth_replay_signature[selected_pressure_event_ordinal][
            "growths"
        ]
        if growth["pool_id"] in selected_pressure_growth_pool_ids
    )
    require(
        probe_positive_growth_bytes >= pressure_quantum_bytes
        and token_scaled_sequence_growth_bytes >= pressure_quantum_bytes,
        "long probe growth is smaller than its typed sequence allocation quantum",
    )
    pressure_budget_candidate_resident_bytes = (
        sizing_resident_bytes
        + selected_pressure_growth_end_bytes
        - pressure_quantum_bytes
    )
    resident_bytes = max(
        sizing_resident_bytes,
        minimum_initial_bundle_resident_bytes,
        pressure_budget_candidate_resident_bytes,
    )
    require(
        resident_bytes >= sizing_resident_bytes,
        "decode pressure budget cannot replay the measured prime workload",
    )
    require(
        resident_bytes >= minimum_initial_bundle_resident_bytes,
        "typed initial bundles do not fit the decode pressure budget",
    )
    probe_growth_headroom_bytes = resident_bytes - sizing_resident_bytes
    selected_pressure_forced_deficit_bytes = (
        selected_pressure_growth_end_bytes - probe_growth_headroom_bytes
    )
    probe_total_growth_budget_gap_bytes = (
        probe_positive_growth_bytes - probe_growth_headroom_bytes
    )
    pressure_budget_reduction_from_sizing_probe_bytes = (
        probe_resident_bytes - resident_bytes
    )
    require(
        selected_pressure_forced_deficit_bytes >= pressure_quantum_bytes,
        "decode pressure budget does not force the selected sequence event",
    )
    require(
        pressure_budget_reduction_from_sizing_probe_bytes
        >= pressure_quantum_bytes,
        "decode pressure budget is not below the unpressured sizing probe",
    )
    exact_budget = static_bytes + resident_bytes
    return {
        "static_bytes": static_bytes,
        "resident_bytes": resident_bytes,
        "budget_claimed_bytes": exact_budget,
        "calibration_resident_bytes": calibration_resident_bytes,
        "maximum_active_sequences": maximum_active_sequences,
        "sizing_observed_resident_bytes": sizing_resident_bytes,
        "sizing_observed_pool_resident_bytes": sizing_pool_bytes,
        "probe_observed_resident_bytes": probe_resident_bytes,
        "probe_observed_pool_resident_bytes": probe_pool_bytes,
        "probe_growth_bytes_by_pool": probe_growth_bytes_by_pool,
        "probe_shrink_bytes_by_pool": probe_shrink_bytes_by_pool,
        "trace_growth_bytes_by_pool": dict(sorted(trace_growth_bytes_by_pool.items())),
        "trace_growth_chunks": trace_growth_chunks,
        "probe_positive_growth_bytes": probe_positive_growth_bytes,
        "sequence_extension_growth_bytes_by_pool": dict(
            sorted(sequence_extension_growth_bytes_by_pool.items())
        ),
        "sequence_extension_growth_chunks_by_pool": {
            pool_id: list(chunk_bytes)
            for pool_id, chunk_bytes in sorted(
                sequence_extension_growth_chunks_by_pool.items()
            )
        },
        "trace_reclaimed_bytes_by_pool": dict(
            sorted(trace_reclaimed_bytes_by_pool.items())
        ),
        "probe_growth_headroom_bytes": probe_growth_headroom_bytes,
        "probe_growth_budget_gap_bytes": selected_pressure_forced_deficit_bytes,
        "probe_total_growth_budget_gap_bytes": probe_total_growth_budget_gap_bytes,
        "requires_cross_pool_rebalance": selected_pressure_forced_deficit_bytes > 0,
        "pressure_quantum_bytes": pressure_quantum_bytes,
        "pressure_quantum_bytes_by_pool": token_scaled_sequence_quanta,
        "growth_replay_signature": sizing_growth_replay_signature,
        "selected_pressure_event_ordinal": selected_pressure_event_ordinal,
        "selected_pressure_event_signature": sizing_growth_replay_signature[
            selected_pressure_event_ordinal
        ],
        "selected_pressure_growth_end_bytes": selected_pressure_growth_end_bytes,
        "selected_pressure_forced_deficit_bytes": (
            selected_pressure_forced_deficit_bytes
        ),
        "pressure_budget_candidate_resident_bytes": (
            pressure_budget_candidate_resident_bytes
        ),
        "pressure_budget_reduction_from_sizing_probe_bytes": (
            pressure_budget_reduction_from_sizing_probe_bytes
        ),
        "initial_bundle_floor_bytes_by_pool": initial_bundle_floor_bytes_by_pool,
        "minimum_initial_bundle_resident_bytes": minimum_initial_bundle_resident_bytes,
        "initial_bundle_headroom_bytes": (
            resident_bytes - minimum_initial_bundle_resident_bytes
        ),
        "donor_evidence_kind": "target_event_exact_receipt_only",
        "token_scaled_sequence_pool_ids": token_scaled_sequence_pool_ids,
        "token_scaled_sequence_growth_pool_ids": selected_pressure_growth_pool_ids,
        "token_scaled_sequence_growth_bytes": token_scaled_sequence_growth_bytes,
        "pool_storage_profiles": pool_storage_profiles,
        "pool_contracts": pool_contracts,
        "calibration_budget_claimed_bytes": calibration_budget,
        "calibration_sizing_delta_bytes": (
            calibration_budget - target_sizing["budget_claimed_bytes"]
        ),
        "bootstrap_headroom_bytes": (
            exact_budget - target_sizing["budget_claimed_bytes"]
        ),
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
        target_contract = validate_typed_pool_contract(
            pool_id, target_envelopes[pool_id], "target"
        )
        sizing_contract = contracts[pool_id]
        require(
            budget_stable_pool_contract(target_contract)
            == budget_stable_pool_contract(sizing_contract),
            f"target pool {pool_id} changed budget-stable typed contract",
        )
        target_maximum_resident_bytes = target_contract["provisioning"][
            "maximum_resident_bytes"
        ]
        sizing_maximum_resident_bytes = sizing_contract["provisioning"][
            "maximum_resident_bytes"
        ]
        require(
            resident_bytes <= target_maximum_resident_bytes
            <= sizing_maximum_resident_bytes,
            f"target pool {pool_id} has an invalid exact-budget resident ceiling",
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


def replayable_prime_layout(snapshot: dict[str, Any], label: str) -> dict[str, Any]:
    pool_resident_bytes = snapshot.get("pool_resident_bytes")
    pool_used_bytes = snapshot.get("pool_used_bytes")
    pool_live_segments = snapshot.get("pool_live_segments")
    pool_transient_occupancy = snapshot.get("pool_transient_occupancy")
    pool_lane_stable_occupancy = snapshot.get("pool_lane_stable_occupancy")
    pool_envelopes = snapshot.get("pool_envelopes")
    require(
        all(
            isinstance(value, dict)
            for value in (
                pool_resident_bytes,
                pool_used_bytes,
                pool_live_segments,
                pool_transient_occupancy,
                pool_lane_stable_occupancy,
                pool_envelopes,
            )
        ),
        f"{label}: replayable pool layout is incomplete",
    )
    pool_ids = set(pool_resident_bytes)
    require(
        pool_ids
        and all(
            set(value) == pool_ids
            for value in (
                pool_used_bytes,
                pool_live_segments,
                pool_transient_occupancy,
                pool_lane_stable_occupancy,
                pool_envelopes,
            )
        ),
        f"{label}: replayable pool identities differ",
    )
    replayable_pool_envelopes: dict[str, Any] = {}
    for pool_id in sorted(pool_ids):
        pool_envelope = pool_envelopes[pool_id]
        require(
            isinstance(pool_envelope, dict),
            f"{label}: {pool_id} replayable pool envelope is invalid",
        )
        contract = validate_typed_pool_contract(pool_id, pool_envelope, label)
        replayable_pool_envelopes[pool_id] = {
            **pool_envelope,
            "contract": budget_stable_pool_contract(contract),
        }
    return {
        "static_bytes": snapshot.get("static_bytes"),
        "resident_bytes": snapshot.get("resident_bytes"),
        "maximum_active_sequences": snapshot.get("maximum_active_sequences"),
        "pool_resident_bytes": pool_resident_bytes,
        "pool_used_bytes": pool_used_bytes,
        "pool_live_segments": pool_live_segments,
        "pool_transient_occupancy": pool_transient_occupancy,
        "pool_lane_stable_occupancy": pool_lane_stable_occupancy,
        "pool_envelopes": replayable_pool_envelopes,
    }


def require_replayed_prime_layout(
    sizing_prime: dict[str, Any], target_prime: dict[str, Any]
) -> dict[str, Any]:
    sizing_layout = replayable_prime_layout(sizing_prime, "target sizing prime")
    target_layout = replayable_prime_layout(target_prime, "fresh target prime")
    require(
        target_layout == sizing_layout,
        "fresh target prime did not replay the sizing-prime physical layout",
    )
    encoded = json.dumps(
        sizing_layout, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "layout_sha256": hashlib.sha256(encoded).hexdigest(),
        "pool_resident_bytes": sizing_layout["pool_resident_bytes"],
        "pool_used_bytes": sizing_layout["pool_used_bytes"],
        "pool_live_segments": sizing_layout["pool_live_segments"],
    }


def rebalance_prime_budget_receipt(
    prime: dict[str, Any],
    sizing_prime: dict[str, Any],
    envelope: dict[str, Any],
    exact_budget: int,
) -> dict[str, Any]:
    require_target_pool_within_budget_contract(prime, envelope, exact_budget)
    layout_receipt = require_replayed_prime_layout(sizing_prime, prime)
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
    replayed_probe_growth_bytes = envelope.get("probe_positive_growth_bytes")
    selected_pressure_growth_end_bytes = envelope.get(
        "selected_pressure_growth_end_bytes"
    )
    selected_pressure_event_ordinal = envelope.get("selected_pressure_event_ordinal")
    pressure_quantum_bytes = envelope.get("pressure_quantum_bytes")
    require(
        0 <= headroom_bytes <= exact_budget,
        "rebalance prime headroom is outside the exact budget",
    )
    require(
        isinstance(replayed_probe_growth_bytes, int)
        and replayed_probe_growth_bytes > 0
        and isinstance(selected_pressure_growth_end_bytes, int)
        and selected_pressure_growth_end_bytes > 0
        and isinstance(selected_pressure_event_ordinal, int)
        and selected_pressure_event_ordinal >= 0
        and isinstance(pressure_quantum_bytes, int)
        and pressure_quantum_bytes > 0,
        "rebalance prime has no typed sizing-probe growth contract",
    )
    forced_deficit_bytes = selected_pressure_growth_end_bytes - headroom_bytes
    require(
        forced_deficit_bytes >= pressure_quantum_bytes,
        "rebalance prime headroom can absorb the replayed sequence growth",
    )
    return {
        "budget_ceiling_bytes": exact_budget,
        "claimed_bytes": claimed_bytes,
        "headroom_bytes": headroom_bytes,
        "resident_ceiling_bytes": resident_ceiling_bytes,
        "resident_bytes": resident_bytes,
        "replayed_probe_growth_bytes": replayed_probe_growth_bytes,
        "selected_pressure_growth_end_bytes": selected_pressure_growth_end_bytes,
        "selected_pressure_event_ordinal": selected_pressure_event_ordinal,
        "forced_deficit_bytes": forced_deficit_bytes,
        "pressure_quantum_bytes": pressure_quantum_bytes,
        "replayed_prime_layout": layout_receipt,
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
        require(yield_kind in ALLOWED_PRESSURE_YIELD_KINDS, f"{label}: yield kind is invalid")
        require(isinstance(progress_owner_id, str), f"{label}: progress owner is missing")
        same_frontier = common.request_identity_matches(progress_owner_id, victim_request_id)
        victim_is_cohort = common.request_identity_matches(
            victim_request_id, request_ids[0]
        )
        owner_is_cohort = common.request_identity_matches(
            progress_owner_id, request_ids[0]
        )
        if yield_kind == "self_recompute":
            require(
                same_frontier and victim_is_cohort and owner_is_cohort,
                f"{label}: self-recompute does not match the failing cohort",
            )
        else:
            require(
                not same_frontier and victim_is_cohort != owner_is_cohort,
                f"{label}: peer handoff does not contain the failing cohort exactly once",
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
    shortfalls = evidence.get("shortfalls")
    backing_blockers = evidence.get("backing_blockers")
    require(isinstance(shortfalls, list), f"{label}: logical shortfalls are missing")
    require(
        isinstance(backing_blockers, list),
        f"{label}: physical backing blockers are missing",
    )
    typed_evidence = evidence.get("typed_evidence")
    require(isinstance(typed_evidence, dict), f"{label}: typed capacity evidence is missing")
    evidence_owner = typed_evidence.get("owner")
    evidence_kind = typed_evidence.get("kind")
    require(
        evidence_owner in {"logical", "backing"},
        f"{label}: typed capacity evidence owner is invalid",
    )
    if evidence_owner == "logical":
        require(evidence_kind == "logical", f"{label}: logical evidence kind is invalid")
        require(bool(shortfalls), f"{label}: logical evidence has no shortfall")
        require(not backing_blockers, f"{label}: logical evidence also owns backing blockers")
        require(
            typed_evidence.get("shortfalls") == shortfalls,
            f"{label}: typed and compatibility logical evidence differ",
        )
    elif evidence_kind == "backing_deferred":
        require(not shortfalls, f"{label}: backing evidence also owns logical shortfalls")
        require(bool(backing_blockers), f"{label}: backing evidence has no blocker")
        require(
            typed_evidence.get("blockers") == backing_blockers,
            f"{label}: typed and compatibility backing evidence differ",
        )
    else:
        require(
            evidence_kind == "backing_pressure",
            f"{label}: backing evidence kind is invalid",
        )
        require(
            not shortfalls and not backing_blockers,
            f"{label}: direct backing pressure also owns compatibility blockers",
        )
    maintenance_boundary = None
    if evidence_kind in {"logical", "backing_deferred"}:
        pressure = typed_evidence.get("pressure")
        require(
            pressure is None or isinstance(pressure, dict),
            f"{label}: maintenance pressure evidence is invalid",
        )
    else:
        require(
            isinstance(typed_evidence.get("pressure"), dict),
            f"{label}: direct backing pressure evidence is missing",
        )
        pressure = typed_evidence["pressure"]
    if isinstance(pressure, dict) and pressure.get("kind") == "device_capacity":
        boundary = typed_evidence.get("maintenance_boundary")
        if evidence_kind in {"logical", "backing_deferred"}:
            require(
                isinstance(boundary, dict),
                f"{label}: device maintenance deferral lost its boundary receipt",
            )
        if boundary is not None:
            maintenance_boundary = validate_maintenance_boundary(
                boundary,
                [],
                coordinator_id=observed["coordinator_id"],
                label=label,
                expect_sufficient=False,
            )
            require(
                maintenance_boundary["pressure"] == pressure.get("evidence"),
                f"{label}: deferred maintenance boundary differs from typed pressure",
            )
    else:
        require(
            typed_evidence.get("maintenance_boundary") is None,
            f"{label}: non-device pressure contains a maintenance boundary",
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
        "evidence_owner": evidence_owner,
        "maintenance_boundary": maintenance_boundary,
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
    hold_transition_ordinal = shape.get("hold_transition_ordinal")
    require(
        isinstance(hold_transition_ordinal, int) and hold_transition_ordinal > 0,
        f"{label}: hold transition ordinal is invalid",
    )
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
        "hold_transition_ordinal": hold_transition_ordinal,
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
    rows: list[dict[str, Any]],
    *,
    started_wall_ns: int,
    finished_wall_ns: int,
    require_maintenance_boundary: bool = False,
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
    boundary_deferrals = [
        event for event in deferrals if event["maintenance_boundary"] is not None
    ]
    if require_maintenance_boundary:
        require(
            boundary_deferrals,
            "target decode pressure has no event-bound resource maintenance receipt",
        )
    splits = [event for event in deferrals if event["decision"] == "split_cohort"]
    parks = [event for event in deferrals if event["decision"] == "wait_for_release"]
    yields = [
        event for event in deferrals if event["decision"] == "pressure_yield_planned"
    ]
    sequence_yields = [
        event for event in yields if event["stage"] == "sequence_extension"
    ]
    require(splits, "target never adaptively split a capacity-blocked decode cohort")
    require(yields, "target never planned a typed execution-capacity yield")
    require(
        sequence_yields,
        "target never planned a typed yield for sequence-extension capacity",
    )

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
    progress_baseline_by_episode: dict[int, int] = {}
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
        prior_baseline = progress_baseline_by_episode.setdefault(
            episode_id, pressure_yield["progress_baseline"]
        )
        require(
            prior_baseline == pressure_yield["progress_baseline"],
            f"pressure episode {episode_id} changed its logical progress baseline",
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
            and hold["hold_transition_ordinal"]
            == completed["release_transition_ordinal"]
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
        "maintenance_boundary_events": len(boundary_deferrals),
        "maintenance_boundary_deficits": [
            event["maintenance_boundary"]["deficit_bytes"]
            for event in boundary_deferrals
        ],
        "split_events": len(splits),
        "park_events": len(parks),
        "pressure_yield_events": len(yields),
        "pressure_yield_kinds": sorted({event["yield_kind"] for event in yields}),
        "pressure_yield_stages": sorted({event["stage"] for event in yields}),
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
        if event["evidence_owner"] == "backing":
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
    phase: str,
) -> dict[str, Any]:
    require(
        phase in {PREFILL_MAINTENANCE_PHASE, EXECUTION_MAINTENANCE_PHASE},
        f"{label}: unsupported maintenance phase {phase}",
    )
    require(
        started_wall_ns > 0 and finished_wall_ns >= started_wall_ns,
        f"{label}: invalid maintenance window",
    )
    maintenance_rows = [
        row
        for row in rows
        if row.get("phase") == phase
        and isinstance(row.get("ts_unix_nanos"), int)
        and started_wall_ns <= row["ts_unix_nanos"] <= finished_wall_ns
    ]
    require(maintenance_rows, f"{label}: no typed {phase} maintenance")
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
            and not isinstance(pools_grown, bool)
            and pools_grown > 0
            and isinstance(allocated_bytes, int)
            and not isinstance(allocated_bytes, bool)
            and allocated_bytes > 0,
            f"{label} maintenance {index}: maintained growth is invalid",
        )
        exact = common.validate_maintenance_rebalance_evidence(
            evidence,
            f"{label} maintenance {index}",
        )
        execution_receipt = None
        if phase == EXECUTION_MAINTENANCE_PHASE:
            execution_receipt = validate_execution_maintenance_receipt(
                row,
                evidence,
                exact,
                f"{label} maintenance {index}",
            )
        growth = {
            "pools_grown": pools_grown,
            "allocated_bytes": allocated_bytes,
            "execution_receipt": execution_receipt,
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
        "growth_receipts": [
            event["execution_receipt"]
            for event in growths
            if event["execution_receipt"] is not None
        ],
    }


def validate_execution_maintenance_receipt(
    row: dict[str, Any],
    evidence: dict[str, Any],
    exact_rebalance: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    require(
        evidence.get("schema_version") == EXECUTION_MAINTENANCE_SCHEMA_VERSION,
        f"{label}: execution maintenance schema version is invalid",
    )
    stage = evidence.get("stage")
    require(stage in ALLOWED_EXECUTION_STAGES, f"{label}: execution stage is invalid")
    shape = row.get("shape")
    require(isinstance(shape, dict), f"{label}: execution shape is missing")
    require(
        shape.get("stage") == stage
        and shape.get("pools_grown") == evidence.get("pools_grown")
        and shape.get("allocated_bytes") == evidence.get("allocated_bytes")
        and shape.get("pools_reclaimed") == evidence.get("pools_reclaimed"),
        f"{label}: execution shape does not reconcile with evidence",
    )
    coordinator_id = evidence.get("coordinator_id")
    require(
        isinstance(coordinator_id, int)
        and not isinstance(coordinator_id, bool)
        and coordinator_id > 0,
        f"{label}: execution coordinator is invalid",
    )
    receipt = evidence.get("receipt")
    require(isinstance(receipt, dict), f"{label}: allocator growth receipt is missing")
    capacity_epoch = receipt.get("capacity_epoch")
    growth_receipts = receipt.get("growths")
    require(
        receipt.get("coordinator_id") == coordinator_id
        and isinstance(capacity_epoch, int)
        and not isinstance(capacity_epoch, bool)
        and capacity_epoch > 0,
        f"{label}: allocator receipt authority or epoch is invalid",
    )
    require(
        isinstance(growth_receipts, list)
        and len(growth_receipts) == evidence.get("pools_grown")
        and growth_receipts,
        f"{label}: allocator growth receipts do not match the aggregate count",
    )
    growth_identities: list[list[Any]] = []
    normalized_growths: list[dict[str, Any]] = []
    allocated_bytes = 0
    for growth_index, growth in enumerate(growth_receipts):
        require(
            isinstance(growth, dict),
            f"{label}: growth receipt {growth_index} is invalid",
        )
        pool_id = growth.get("pool_id")
        chunk = growth.get("chunk")
        chunk_bytes = growth.get("chunk_bytes")
        published_capacity_bytes = growth.get("published_capacity_bytes")
        require(
            isinstance(pool_id, str)
            and pool_id.startswith("dynamic-pool/sha256/")
            and common.SHA256_RE.fullmatch(
                pool_id.removeprefix("dynamic-pool/sha256/")
            )
            is not None
            and isinstance(chunk, dict)
            and chunk.get("pool_id") == pool_id
            and isinstance(chunk.get("ordinal"), int)
            and not isinstance(chunk.get("ordinal"), bool)
            and chunk["ordinal"] > 0
            and isinstance(chunk.get("generation"), int)
            and not isinstance(chunk.get("generation"), bool)
            and chunk["generation"] > 0,
            f"{label}: growth receipt {growth_index} has an invalid chunk identity",
        )
        identity = [pool_id, chunk["ordinal"], chunk["generation"]]
        require(
            identity not in growth_identities,
            f"{label}: allocator growth receipt repeats a chunk identity",
        )
        require(
            isinstance(chunk_bytes, int)
            and not isinstance(chunk_bytes, bool)
            and chunk_bytes > 0
            and isinstance(published_capacity_bytes, int)
            and not isinstance(published_capacity_bytes, bool)
            and published_capacity_bytes >= chunk_bytes
            and growth.get("capacity_epoch") == capacity_epoch,
            f"{label}: growth receipt {growth_index} has invalid bytes or epoch",
        )
        growth_identities.append(identity)
        normalized_growths.append(
            {
                "pool_id": pool_id,
                "chunk_identity": identity,
                "chunk_bytes": chunk_bytes,
                "published_capacity_bytes": published_capacity_bytes,
            }
        )
        allocated_bytes += chunk_bytes
    require(
        allocated_bytes == evidence.get("allocated_bytes"),
        f"{label}: exact growth bytes do not match the aggregate count",
    )
    require(
        receipt.get("rebalance") == evidence.get("rebalance"),
        f"{label}: allocator and event rebalance receipts differ",
    )
    require(
        receipt.get("maintenance_boundary") == evidence.get("maintenance_boundary"),
        f"{label}: allocator and event maintenance boundaries differ",
    )
    normalized_reclaims: list[dict[str, Any]] = []
    rebalance = evidence.get("rebalance")
    if exact_rebalance["pools_reclaimed"] == 0:
        require(
            rebalance is None and evidence.get("maintenance_boundary") is None,
            f"{label}: unpressured growth has reclaim-boundary evidence",
        )
        maintenance_boundary = None
    else:
        require(isinstance(rebalance, dict), f"{label}: exact reclaim receipt is missing")
        reclaim_pools = rebalance.get("pools")
        require(
            isinstance(reclaim_pools, list)
            and len(reclaim_pools) == exact_rebalance["pools_reclaimed"],
            f"{label}: exact reclaim pool count drifted",
        )
        for reclaim_index, reclaim in enumerate(reclaim_pools):
            pool_id = reclaim.get("pool_id") if isinstance(reclaim, dict) else None
            chunks = reclaim.get("chunks") if isinstance(reclaim, dict) else None
            reclaimed_bytes = (
                reclaim.get("reclaimed_bytes") if isinstance(reclaim, dict) else None
            )
            published_capacity_bytes = (
                reclaim.get("published_capacity_bytes")
                if isinstance(reclaim, dict)
                else None
            )
            require(
                isinstance(pool_id, str)
                and isinstance(chunks, list)
                and chunks
                and isinstance(reclaimed_bytes, int)
                and not isinstance(reclaimed_bytes, bool)
                and reclaimed_bytes > 0
                and isinstance(published_capacity_bytes, int)
                and not isinstance(published_capacity_bytes, bool)
                and published_capacity_bytes >= 0,
                f"{label}: reclaim receipt {reclaim_index} is invalid",
            )
            chunk_identities = [
                [chunk.get("pool_id"), chunk.get("ordinal"), chunk.get("generation")]
                for chunk in chunks
            ]
            normalized_reclaims.append(
                {
                    "pool_id": pool_id,
                    "reclaimed_bytes": reclaimed_bytes,
                    "chunk_identities": chunk_identities,
                    "published_capacity_bytes": published_capacity_bytes,
                }
            )
        require(
            [reclaim["pool_id"] for reclaim in normalized_reclaims]
            == exact_rebalance["pool_ids"]
            and [
                identity
                for reclaim in normalized_reclaims
                for identity in reclaim["chunk_identities"]
            ]
            == exact_rebalance["chunk_identities"]
            and sum(
                reclaim["reclaimed_bytes"] for reclaim in normalized_reclaims
            )
            == exact_rebalance["reclaimed_bytes"],
            f"{label}: normalized reclaim receipt differs from exact evidence",
        )
        maintenance_boundary = validate_maintenance_boundary(
            evidence.get("maintenance_boundary"),
            normalized_reclaims,
            coordinator_id=coordinator_id,
            label=label,
        )
    require(
        {growth["pool_id"] for growth in normalized_growths}.isdisjoint(
            reclaim["pool_id"] for reclaim in normalized_reclaims
        ),
        f"{label}: one event grows and reclaims the same pool",
    )
    participants = evidence.get("participants")
    require(
        isinstance(participants, list)
        and participants
        and shape.get("participant_count") == len(participants),
        f"{label}: execution participants are missing or miscounted",
    )
    participant_identities: list[list[Any]] = []
    for participant_index, participant in enumerate(participants):
        authority = participant.get("sequence_authority") if isinstance(participant, dict) else None
        identity = [
            participant.get("run_id") if isinstance(participant, dict) else None,
            participant.get("request_id") if isinstance(participant, dict) else None,
            authority.get("sparse_id") if isinstance(authority, dict) else None,
            authority.get("generation") if isinstance(authority, dict) else None,
        ]
        require(
            all(isinstance(value, str) and value for value in identity[:2])
            and isinstance(identity[2], int)
            and not isinstance(identity[2], bool)
            and identity[2] >= 0
            and isinstance(identity[3], int)
            and not isinstance(identity[3], bool)
            and identity[3] > 0
            and common.SHA256_RE.fullmatch(
                str(participant.get("active_sequence_fingerprint"))
            )
            is not None
            and identity not in participant_identities,
            f"{label}: participant {participant_index} is invalid or duplicated",
        )
        participant_identities.append(identity)
    fingerprint = evidence.get("event_fingerprint")
    require(
        common.SHA256_RE.fullmatch(str(fingerprint)) is not None
        and row.get("correlation_id") == fingerprint
        and row.get("event_id")
        == f"evt-vnext-execution-resource-maintenance-{fingerprint}"
        and row.get("request_id") == participants[0]["request_id"],
        f"{label}: execution event identity does not bind to its evidence",
    )
    attributes = row.get("attributes")
    require(
        attributes.get("execution_trace_source") == "vnext_resource_maintenance"
        and isinstance(attributes.get("plan_id"), str)
        and attributes["plan_id"]
        and common.SHA256_RE.fullmatch(str(attributes.get("plan_hash"))) is not None
        and attributes.get("run_id") == participants[0]["run_id"],
        f"{label}: plan or run authority is missing",
    )
    return {
        "stage": stage,
        "coordinator_id": coordinator_id,
        "capacity_epoch": capacity_epoch,
        "allocated_bytes": allocated_bytes,
        "growths": normalized_growths,
        "reclaimed_bytes": exact_rebalance["reclaimed_bytes"],
        "reclaims": normalized_reclaims,
        "maintenance_boundary": maintenance_boundary,
        "growth_chunk_identities": growth_identities,
        "participant_identities": participant_identities,
        "event_fingerprint": fingerprint,
    }


def validate_maintenance_boundary(
    boundary: Any,
    reclaims: list[dict[str, Any]],
    *,
    coordinator_id: int,
    label: str,
    expect_sufficient: bool = True,
) -> dict[str, Any]:
    require(isinstance(boundary, dict), f"{label}: maintenance boundary is missing")
    require(
        boundary.get("schema_version") == MAINTENANCE_BOUNDARY_SCHEMA_VERSION
        and boundary.get("coordinator_id") == coordinator_id,
        f"{label}: maintenance boundary schema or coordinator is invalid",
    )
    for key in (
        "logical_release_epoch",
        "logical_capacity_epoch",
        "plan_device_capacity_epoch",
        "process_device_capacity_epoch",
    ):
        require(
            isinstance(boundary.get(key), int)
            and not isinstance(boundary[key], bool)
            and boundary[key] > 0,
            f"{label}: maintenance boundary {key} is invalid",
        )
    pressure = boundary.get("pressure")
    require(isinstance(pressure, dict), f"{label}: maintenance pressure is missing")
    require(
        pressure.get("scope") in {"plan_budget", "process_wide"}
        and isinstance(pressure.get("device_id"), str)
        and pressure["device_id"],
        f"{label}: maintenance pressure authority is invalid",
    )
    pressure_bytes: dict[str, int] = {}
    for key in (
        "requested_bytes",
        "plan_claimed_bytes",
        "plan_usable_bytes",
        "process_claimed_bytes",
        "process_usable_bytes",
    ):
        value = pressure.get(key)
        require(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0,
            f"{label}: maintenance pressure {key} is invalid",
        )
        pressure_bytes[key] = value
    require(
        pressure_bytes["requested_bytes"] > 0
        and pressure_bytes["plan_claimed_bytes"]
        <= pressure_bytes["plan_usable_bytes"]
        and pressure_bytes["process_claimed_bytes"]
        <= pressure_bytes["process_usable_bytes"],
        f"{label}: maintenance pressure byte envelope is inconsistent",
    )
    available_bytes = min(
        pressure_bytes["plan_usable_bytes"]
        - pressure_bytes["plan_claimed_bytes"],
        pressure_bytes["process_usable_bytes"]
        - pressure_bytes["process_claimed_bytes"],
    )
    require(
        available_bytes < pressure_bytes["requested_bytes"],
        f"{label}: maintenance boundary does not contain a capacity deficit",
    )
    deficit_bytes = pressure_bytes["requested_bytes"] - available_bytes

    planned_domains = boundary.get("planned_domains")
    require(
        isinstance(planned_domains, list)
        and planned_domains
        and all(
            isinstance(domain, int) and not isinstance(domain, bool) and domain > 0
            for domain in planned_domains
        )
        and planned_domains == sorted(set(planned_domains)),
        f"{label}: maintenance planned domains are invalid or non-canonical",
    )
    require(
        isinstance(boundary.get("protected_immediate"), list)
        and isinstance(boundary.get("protected_packing_envelopes"), list),
        f"{label}: maintenance protection evidence is missing",
    )
    pools = boundary.get("pools")
    require(isinstance(pools, list) and pools, f"{label}: maintenance pools are missing")
    pool_ids: list[str] = []
    candidates: dict[tuple[str, int, int], int] = {}
    for pool_index, pool in enumerate(pools):
        pool_id = pool.get("pool_id") if isinstance(pool, dict) else None
        domain_id = pool.get("domain_id") if isinstance(pool, dict) else None
        require(
            isinstance(pool_id, str)
            and pool_id.startswith("dynamic-pool/sha256/")
            and common.SHA256_RE.fullmatch(
                pool_id.removeprefix("dynamic-pool/sha256/")
            )
            is not None
            and isinstance(domain_id, int)
            and not isinstance(domain_id, bool)
            and domain_id > 0,
            f"{label}: maintenance pool {pool_index} identity is invalid",
        )
        pool_ids.append(pool_id)
        require(
            pool.get("excluded_from_reclaim") == (domain_id in planned_domains),
            f"{label}: maintenance pool {pool_id} exclusion differs from planned domains",
        )
        byte_fields: dict[str, int] = {}
        for key in (
            "resident_bytes",
            "pending_growth_bytes",
            "free_bytes",
            "largest_contiguous_bytes",
            "logical_used_bytes",
            "minimum_resident_bytes",
            "maximum_resident_bytes",
            "protected_immediate_bytes",
            "coherent_runnable_floor_bytes",
            "resident_floor_bytes",
            "reclaimable_bytes",
        ):
            value = pool.get(key)
            require(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0,
                f"{label}: maintenance pool {pool_id} {key} is invalid",
            )
            byte_fields[key] = value
        live = pool.get("live_occupancy")
        live_total = live.get("total") if isinstance(live, dict) else None
        live_physical = (
            live_total.get("physical_bytes") if isinstance(live_total, dict) else None
        )
        require(
            isinstance(live_physical, int)
            and not isinstance(live_physical, bool)
            and live_physical >= 0
            and byte_fields["free_bytes"] <= byte_fields["resident_bytes"]
            and byte_fields["resident_bytes"] - byte_fields["free_bytes"]
            == live_physical,
            f"{label}: maintenance pool {pool_id} live occupancy is inconsistent",
        )
        coherent_floor = max(byte_fields["logical_used_bytes"], live_physical) + byte_fields[
            "protected_immediate_bytes"
        ]
        resident_floor = max(byte_fields["minimum_resident_bytes"], coherent_floor)
        require(
            byte_fields["coherent_runnable_floor_bytes"] == coherent_floor
            and byte_fields["resident_floor_bytes"] == resident_floor
            and byte_fields["reclaimable_bytes"]
            == max(0, byte_fields["resident_bytes"] - resident_floor)
            and byte_fields["minimum_resident_bytes"]
            <= byte_fields["maximum_resident_bytes"],
            f"{label}: maintenance pool {pool_id} runnable floor is inconsistent",
        )
        fingerprint = pool.get("free_extent_layout_fingerprint")
        require(
            isinstance(fingerprint, str)
            and fingerprint.startswith("sha256/")
            and common.SHA256_RE.fullmatch(fingerprint.removeprefix("sha256/"))
            is not None
            and isinstance(pool.get("protected_packing_satisfied"), bool),
            f"{label}: maintenance pool {pool_id} layout evidence is invalid",
        )
        chunks = pool.get("chunks")
        require(isinstance(chunks, list), f"{label}: maintenance chunks are missing")
        chunk_bytes_total = 0
        chunk_identities: set[tuple[str, int, int]] = set()
        for chunk_index, chunk in enumerate(chunks):
            identity = chunk.get("identity") if isinstance(chunk, dict) else None
            chunk_bytes = chunk.get("bytes") if isinstance(chunk, dict) else None
            chunk_identity = (
                identity.get("pool_id") if isinstance(identity, dict) else None,
                identity.get("ordinal") if isinstance(identity, dict) else None,
                identity.get("generation") if isinstance(identity, dict) else None,
            )
            require(
                chunk_identity[0] == pool_id
                and isinstance(chunk_identity[1], int)
                and not isinstance(chunk_identity[1], bool)
                and chunk_identity[1] > 0
                and isinstance(chunk_identity[2], int)
                and not isinstance(chunk_identity[2], bool)
                and chunk_identity[2] > 0
                and chunk_identity not in chunk_identities
                and isinstance(chunk_bytes, int)
                and not isinstance(chunk_bytes, bool)
                and chunk_bytes > 0,
                f"{label}: maintenance chunk {pool_id}/{chunk_index} is invalid",
            )
            chunk_identities.add(chunk_identity)
            chunk_bytes_total += chunk_bytes
            live_segments = chunk.get("live_segments")
            external_references = chunk.get("external_references")
            require(
                isinstance(live_segments, int)
                and not isinstance(live_segments, bool)
                and live_segments >= 0
                and isinstance(external_references, int)
                and not isinstance(external_references, bool)
                and external_references >= 0
                and all(
                    isinstance(chunk.get(key), bool)
                    for key in (
                        "protected_packing",
                        "full_extent_available",
                        "resident_floor_allows_reclaim",
                        "reclaim_candidate",
                    )
                ),
                f"{label}: maintenance chunk {pool_id}/{chunk_index} state is invalid",
            )
            expected_candidate = (
                not pool["excluded_from_reclaim"]
                and byte_fields["pending_growth_bytes"] == 0
                and pool["protected_packing_satisfied"]
                and live_segments == 0
                and external_references == 0
                and not chunk["protected_packing"]
                and chunk["full_extent_available"]
                and chunk["resident_floor_allows_reclaim"]
            )
            require(
                chunk["resident_floor_allows_reclaim"]
                == (chunk_bytes <= byte_fields["reclaimable_bytes"])
                and chunk["reclaim_candidate"] == expected_candidate,
                f"{label}: maintenance chunk {pool_id}/{chunk_index} eligibility drifted",
            )
            if expected_candidate:
                candidates[chunk_identity] = chunk_bytes
        require(
            chunk_bytes_total == byte_fields["resident_bytes"],
            f"{label}: maintenance pool {pool_id} chunk residency does not reconcile",
        )
    require(
        pool_ids == sorted(set(pool_ids)),
        f"{label}: maintenance pool order is not canonical",
    )
    require(
        boundary.get("reclaim_candidate_chunks") == len(candidates)
        and boundary.get("reclaim_candidate_bytes") == sum(candidates.values()),
        f"{label}: maintenance candidate aggregate is inconsistent",
    )
    selected = boundary.get("selected_chunks")
    require(isinstance(selected, list), f"{label}: selected reclaim chunks are missing")
    selected_identities = [
        (chunk.get("pool_id"), chunk.get("ordinal"), chunk.get("generation"))
        for chunk in selected
        if isinstance(chunk, dict)
    ]
    require(
        len(selected_identities) == len(selected)
        and len(set(selected_identities)) == len(selected_identities)
        and all(identity in candidates for identity in selected_identities),
        f"{label}: selected reclaim chunks are invalid",
    )
    selected_bytes = sum(candidates[identity] for identity in selected_identities)
    require(
        boundary.get("selected_bytes") == selected_bytes,
        f"{label}: maintenance selected-byte aggregate is inconsistent",
    )
    if expect_sufficient:
        reclaimed_identities = [
            tuple(identity)
            for reclaim in reclaims
            for identity in reclaim["chunk_identities"]
        ]
        require(
            boundary.get("reclaim_sufficient") is True
            and selected_bytes >= deficit_bytes
            and selected_identities == reclaimed_identities
            and selected_bytes
            == sum(reclaim["reclaimed_bytes"] for reclaim in reclaims),
            f"{label}: maintenance selection differs from its published rebalance",
        )
    else:
        require(
            not reclaims
            and boundary.get("reclaim_sufficient") is False
            and selected_bytes < deficit_bytes,
            f"{label}: blocked maintenance boundary incorrectly claims sufficient reclaim",
        )
    return {
        "schema_version": boundary["schema_version"],
        "coordinator_id": coordinator_id,
        "logical_release_epoch": boundary["logical_release_epoch"],
        "logical_capacity_epoch": boundary["logical_capacity_epoch"],
        "plan_device_capacity_epoch": boundary["plan_device_capacity_epoch"],
        "process_device_capacity_epoch": boundary["process_device_capacity_epoch"],
        "pressure": pressure,
        "deficit_bytes": deficit_bytes,
        "planned_domains": planned_domains,
        "pool_ids": pool_ids,
        "reclaim_candidate_chunks": len(candidates),
        "reclaim_candidate_bytes": sum(candidates.values()),
        "selected_chunk_identities": [list(identity) for identity in selected_identities],
        "selected_bytes": selected_bytes,
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
        phase=EXECUTION_MAINTENANCE_PHASE,
    )
    require(
        summary["rebalance_events"] > 0,
        f"{label} produced no typed rebalance",
    )
    return summary


def validate_target_rebalance_witness(
    summary: dict[str, Any],
    envelope: dict[str, Any],
    prime_budget_receipt: dict[str, Any],
    prime_pool: dict[str, Any],
    probe_pool: dict[str, Any],
) -> dict[str, Any]:
    receipts = summary.get("growth_receipts")
    require(
        isinstance(receipts, list) and receipts,
        "target rebalance probe has no exact execution receipts",
    )
    actual_growth_replay_signature = growth_replay_signature(
        receipts, "target rebalance probe"
    )
    require(
        actual_growth_replay_signature == envelope.get("growth_replay_signature"),
        "target probe did not replay the ordered sizing growth events",
    )
    selected_growth_pools = set(envelope["token_scaled_sequence_growth_pool_ids"])
    pressure_quanta = envelope["pressure_quantum_bytes_by_pool"]
    actual_growth_bytes_by_pool: dict[str, int] = {}
    actual_reclaimed_bytes_by_pool: dict[str, int] = {}
    for receipt_index, receipt in enumerate(receipts):
        growths = receipt.get("growths") if isinstance(receipt, dict) else None
        reclaims = receipt.get("reclaims") if isinstance(receipt, dict) else None
        require(
            isinstance(growths, list)
            and growths
            and isinstance(reclaims, list),
            f"target rebalance receipt {receipt_index} is incomplete",
        )
        for growth in growths:
            pool_id = growth["pool_id"]
            actual_growth_bytes_by_pool[pool_id] = (
                actual_growth_bytes_by_pool.get(pool_id, 0) + growth["chunk_bytes"]
            )
        for reclaim in reclaims:
            pool_id = reclaim["pool_id"]
            actual_reclaimed_bytes_by_pool[pool_id] = (
                actual_reclaimed_bytes_by_pool.get(pool_id, 0)
                + reclaim["reclaimed_bytes"]
            )
        require(
            {growth["pool_id"] for growth in growths}.isdisjoint(
                reclaim["pool_id"] for reclaim in reclaims
            ),
            f"target rebalance receipt {receipt_index} grows and reclaims one pool",
        )
    selected_event_ordinal = envelope.get("selected_pressure_event_ordinal")
    require(
        isinstance(selected_event_ordinal, int)
        and 0 <= selected_event_ordinal < len(receipts)
        and prime_budget_receipt.get("selected_pressure_event_ordinal")
        == selected_event_ordinal,
        "target pressure event ordinal is invalid",
    )
    selected_receipt = receipts[selected_event_ordinal]
    require(
        all(
            receipt.get("reclaimed_bytes") == 0 and not receipt.get("reclaims")
            for receipt in receipts[:selected_event_ordinal]
        ),
        "target probe reclaimed backing before the selected pressure event",
    )
    selected_growths = selected_receipt["growths"]
    selected_reclaims = selected_receipt["reclaims"]
    event_sequence_growths = [
        growth
        for growth in selected_growths
        if growth["pool_id"] in selected_growth_pools
    ]
    require(
        selected_receipt.get("stage") == "sequence_extension"
        and event_sequence_growths
        and all(
            growth["chunk_bytes"] % pressure_quanta[growth["pool_id"]] == 0
            for growth in event_sequence_growths
        ),
        "selected target pressure event is not attributable sequence growth",
    )
    require(
        selected_reclaims
        and selected_receipt.get("reclaimed_bytes", 0)
        >= prime_budget_receipt["forced_deficit_bytes"],
        "selected target pressure event did not reclaim the forced deficit",
    )
    qualifying_events = [
        {
            "event_ordinal": selected_event_ordinal,
            "event_fingerprint": selected_receipt["event_fingerprint"],
            "growth_pool_ids": sorted(
                {growth["pool_id"] for growth in event_sequence_growths}
            ),
            "reclaim_pool_ids": sorted(
                {reclaim["pool_id"] for reclaim in selected_reclaims}
            ),
            "allocated_bytes": selected_receipt["allocated_bytes"],
            "reclaimed_bytes": selected_receipt["reclaimed_bytes"],
        }
    ]
    require(
        actual_growth_bytes_by_pool == envelope["trace_growth_bytes_by_pool"],
        "target probe did not replay the sizing probe physical growth",
    )
    require(
        summary.get("allocated_bytes") == sum(actual_growth_bytes_by_pool.values())
        and summary.get("reclaimed_bytes")
        == sum(actual_reclaimed_bytes_by_pool.values()),
        "target probe aggregate bytes differ from its exact receipts",
    )
    prime_pools = prime_pool.get("pool_resident_bytes")
    probe_pools = probe_pool.get("pool_resident_bytes")
    require(
        isinstance(prime_pools, dict)
        and isinstance(probe_pools, dict)
        and prime_pools.keys() == probe_pools.keys(),
        "target prime/probe pool identities differ",
    )
    for pool_id in prime_pools:
        require(
            probe_pools[pool_id]
            == prime_pools[pool_id]
            + actual_growth_bytes_by_pool.get(pool_id, 0)
            - actual_reclaimed_bytes_by_pool.get(pool_id, 0),
            f"target probe pool conservation failed for {pool_id}",
        )
    require(
        probe_pool.get("resident_bytes")
        == prime_pool.get("resident_bytes")
        + sum(actual_growth_bytes_by_pool.values())
        - sum(actual_reclaimed_bytes_by_pool.values()),
        "target probe total residency does not conserve exact maintenance bytes",
    )
    return {
        "growth_replay_signature": actual_growth_replay_signature,
        "sizing_growth_bytes_by_pool": envelope["trace_growth_bytes_by_pool"],
        "actual_growth_bytes_by_pool": dict(sorted(actual_growth_bytes_by_pool.items())),
        "actual_reclaimed_bytes_by_pool": dict(
            sorted(actual_reclaimed_bytes_by_pool.items())
        ),
        "forced_deficit_bytes": prime_budget_receipt["forced_deficit_bytes"],
        "qualifying_events": qualifying_events,
    }


def require_decode_prompt(result: dict[str, Any], slot: str) -> None:
    require(slot in DECODE_PROMPT_SHA256_BY_SLOT, f"invalid decode workload slot {slot}")
    require(
        result.get("workload_slot") == slot
        and result.get("prompt_sha256") == DECODE_PROMPT_SHA256_BY_SLOT[slot],
        f"{slot}: decode prompt differs from the canonical workload",
    )


def validate_replayed_decode_workload(
    slot: str, labeled_results: dict[str, dict[str, Any]]
) -> None:
    expected_max_tokens = {
        "calibration": CALIBRATION_MAX_TOKENS[slot],
        "target-sizing": CALIBRATION_MAX_TOKENS[slot],
        "target-rebalance-prime": REBALANCE_PRIME_MAX_TOKENS[slot],
        "target": TARGET_MAX_TOKENS[slot],
    }
    require(
        labeled_results.keys() == expected_max_tokens.keys(),
        f"workload slot {slot} has an invalid replay phase set",
    )
    prompt_token_counts: set[int] = set()
    for phase, result in labeled_results.items():
        require_decode_prompt(result, slot)
        prompt_tokens = result.get("prompt_tokens")
        require(
            isinstance(prompt_tokens, int) and prompt_tokens > 0,
            f"{phase}: prompt token count is missing",
        )
        max_tokens = expected_max_tokens[phase]
        require(
            result.get("max_tokens") == max_tokens,
            f"{phase}: max_tokens does not match decode workload slot {slot}",
        )
        require(
            prompt_tokens + max_tokens <= MAX_MODEL_LEN,
            f"{phase}: decode workload slot {slot} exceeds the model-length ceiling",
        )
        prompt_token_counts.add(prompt_tokens)
    require(
        len(prompt_token_counts) == 1,
        f"workload slot {slot} changed its tokenized prompt length across phases",
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
            runtime_budget=None,
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
        sizing_health = target_sizing.health("health.prime.json")
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
            phase=PREFILL_MAINTENANCE_PHASE,
        )
        sizing_probe_task = common.StreamTask(
            port=target_sizing.port,
            model=target_sizing.model_id,
            role="target-sizing-rebalance-probe",
            workload_slot=REBALANCE_PROBE_WORKLOAD_SLOT,
            max_tokens=REBALANCE_PROBE_MAX_TOKENS,
            out_dir=out / "target-sizing-rebalance-probe" / "clients",
            timeout=args.request_timeout,
            prompt=REBALANCE_PROBE_PROMPT,
        )
        tasks = {REBALANCE_PROBE_WORKLOAD_SLOT: sizing_probe_task}
        sizing_probe_deadline = time.monotonic() + args.request_timeout
        sizing_probe_task.start()
        sizing_probe_task.wait_first(
            min(args.request_timeout, STOP_POLICY["no_progress_timeout_seconds"])
        )
        sizing_probe_client = sizing_probe_task.join(
            max(0.0, sizing_probe_deadline - time.monotonic())
        )
        common.validate_stream(
            sizing_probe_client, "target-sizing-rebalance-probe"
        )
        tasks = {}
        sizing_probe_health = target_sizing.health("health.rebalance-probe.json")
        sizing_probe_executor = common.find_executor_snapshot(sizing_probe_health)
        require(
            sizing_probe_executor is not None,
            "target sizing rebalance probe health has no vNext executor",
        )
        sizing_probe_pool = common.quiescent_pool_snapshot(
            sizing_probe_executor,
            "decode target sizing rebalance probe",
            baseline_executor=sizing_start_executor,
        )
        sizing_rows = common.read_trace(target_sizing.trace_path)
        sizing_probe_maintenance_summary = validate_maintenance_trace(
            sizing_rows,
            started_wall_ns=sizing_probe_client["started_wall_ns"],
            finished_wall_ns=sizing_probe_client["finished_wall_ns"],
            label="target sizing rebalance probe",
            phase=EXECUTION_MAINTENANCE_PHASE,
        )
        collection["target_sizing"] = {
            "clients": sizing_clients,
            "monitor": sizing_monitor,
            "pool_snapshot": sizing_pool,
            "maintenance_summary": sizing_maintenance_summary,
            "health_prime": "target-sizing/health.prime.json",
            "rebalance_probe": {
                "client": sizing_probe_client,
                "pool_snapshot": sizing_probe_pool,
                "health": "target-sizing/health.rebalance-probe.json",
                "maintenance_summary": sizing_probe_maintenance_summary,
            },
            "trace": "target-sizing/scheduler-trace.jsonl",
        }
        target_budget_envelope = derive_target_budget_envelope(
            calibration_pool,
            sizing_pool,
            sizing_probe_pool,
            sizing_probe_maintenance_summary,
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
            sizing_pool,
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
        rebalance_witness = validate_target_rebalance_witness(
            probe_maintenance_summary,
            target_budget_envelope,
            prime_budget_receipt,
            prime_pool,
            probe_pool,
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
                "rebalance_witness": rebalance_witness,
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
            require_maintenance_boundary=True,
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


def validate_rebalance_probe(
    root: Path,
    result: dict[str, Any],
    *,
    artifact_phase: str,
    role: str,
) -> tuple[int, int, float]:
    require(isinstance(result, dict), f"{role}: result is invalid")
    require(
        result.get("role") == role
        and result.get("workload_slot") == REBALANCE_PROBE_WORKLOAD_SLOT
        and result.get("max_tokens") == REBALANCE_PROBE_MAX_TOKENS,
        f"{role}: workload identity changed",
    )
    require(
        result.get("prompt_sha256")
        == hashlib.sha256(REBALANCE_PROBE_PROMPT.encode("utf-8")).hexdigest(),
        f"{role}: prompt changed",
    )
    prompt_tokens = result.get("prompt_tokens")
    require(
        isinstance(prompt_tokens, int)
        and REBALANCE_PROBE_MAX_TOKENS < prompt_tokens < MAX_MODEL_LEN,
        f"{role}: tokenized prompt is outside the product model limit",
    )
    common.validate_stream(result, role)
    started = result["started_wall_ns"]
    finished = result["finished_wall_ns"]
    events = (
        root
        / artifact_phase
        / "clients"
        / f"{role}.events.jsonl"
    )
    silence = common.max_stream_silence_seconds(
        result,
        common.read_stream_content_times(events),
        monitored_from_wall_ns=started,
    )
    require(
        silence < STOP_POLICY["no_progress_timeout_seconds"],
        f"{role}: token progress stalled for {silence:.3f}s",
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
    model_revision_from_snapshot(collection.get("model_path"))

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
    sizing_rebalance_probe = target_sizing.get("rebalance_probe")
    require(
        isinstance(sizing_rebalance_probe, dict)
        and isinstance(rebalance_prime, dict)
        and isinstance(rebalance_probe, dict),
        "sizing or target rebalance phases are missing",
    )
    calibration_start_health = common.read_json(root / "calibration/health.start.json")
    sizing_start_health = common.read_json(root / "target-sizing/health.start.json")
    target_start_health = common.read_json(root / "target/health.start.json")
    calibration_health = common.read_json(root / str(calibration.get("health_final")))
    sizing_health = common.read_json(root / str(target_sizing.get("health_prime")))
    sizing_probe_health = common.read_json(
        root / str(sizing_rebalance_probe.get("health"))
    )
    target_health = common.read_json(root / str(target.get("health_final")))
    prime_health = common.read_json(root / str(rebalance_prime.get("health")))
    probe_health = common.read_json(root / str(rebalance_probe.get("health")))
    calibration_start_executor = common.find_executor_snapshot(calibration_start_health)
    sizing_start_executor = common.find_executor_snapshot(sizing_start_health)
    target_start_executor = common.find_executor_snapshot(target_start_health)
    calibration_executor = common.find_executor_snapshot(calibration_health)
    sizing_executor = common.find_executor_snapshot(sizing_health)
    sizing_probe_executor = common.find_executor_snapshot(sizing_probe_health)
    target_executor = common.find_executor_snapshot(target_health)
    prime_executor = common.find_executor_snapshot(prime_health)
    probe_executor = common.find_executor_snapshot(probe_health)
    require(
        calibration_start_executor is not None
        and sizing_start_executor is not None
        and target_start_executor is not None
        and calibration_executor is not None
        and sizing_executor is not None
        and sizing_probe_executor is not None
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
                ("target sizing prime", sizing_executor),
                ("target sizing rebalance probe", sizing_probe_executor),
            ],
            "target": [
                ("target start", target_start_executor),
                ("target rebalance prime", prime_executor),
                ("target rebalance probe", probe_executor),
                ("target final", target_executor),
            ],
        }
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
    sizing_probe_pool = common.quiescent_pool_snapshot(
        sizing_probe_executor,
        "raw decode target sizing rebalance probe",
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
    require(
        sizing_rebalance_probe.get("pool_snapshot") == sizing_probe_pool,
        "target sizing rebalance probe summary differs from raw health",
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
    sizing_rows = common.read_trace(root / str(target_sizing.get("trace")))
    sizing_probe_client = sizing_rebalance_probe.get("client")
    require(
        isinstance(sizing_probe_client, dict)
        and isinstance(sizing_probe_client.get("started_wall_ns"), int)
        and isinstance(sizing_probe_client.get("finished_wall_ns"), int),
        "target sizing rebalance probe client window is missing",
    )
    sizing_probe_maintenance_summary = validate_maintenance_trace(
        sizing_rows,
        started_wall_ns=sizing_probe_client["started_wall_ns"],
        finished_wall_ns=sizing_probe_client["finished_wall_ns"],
        label="target sizing rebalance probe",
        phase=EXECUTION_MAINTENANCE_PHASE,
    )
    require(
        sizing_rebalance_probe.get("maintenance_summary")
        == sizing_probe_maintenance_summary,
        "target sizing rebalance probe summary differs from raw trace",
    )
    target_budget_envelope = derive_target_budget_envelope(
        calibration_pool,
        sizing_pool,
        sizing_probe_pool,
        sizing_probe_maintenance_summary,
    )
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
        runtime_budget=None,
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
        sizing_pool,
        target_budget_envelope,
        exact_budget,
    )
    require(
        rebalance_prime.get("budget_receipt") == prime_budget_receipt,
        "rebalance prime budget receipt differs from raw backing",
    )
    calibration_policy = calibration_executor.get("runtime_memory_policy")
    sizing_policy = sizing_executor.get("runtime_memory_policy")
    calibration_usable = runtime_memory_usable_bytes(
        calibration_policy, "calibration"
    )
    sizing_usable = runtime_memory_usable_bytes(sizing_policy, "target sizing")
    require(
        calibration_policy == sizing_policy
        and calibration_usable == sizing_usable
        and calibration_usable >= calibration_pool["budget_claimed_bytes"]
        and sizing_usable >= sizing_probe_pool["budget_claimed_bytes"],
        "calibration and target sizing did not share a sufficient product-default runtime budget",
    )
    policy = target_executor.get("runtime_memory_policy")
    require(
        runtime_memory_usable_bytes(policy, "target") == exact_budget,
        "target runtime did not use the calibrated exact budget",
    )

    calibration_started, calibration_finished, calibration_silence = validate_stream_group(
        root, "calibration", calibration.get("clients"), CALIBRATION_MAX_TOKENS
    )
    sizing_started, sizing_finished, sizing_silence = validate_stream_group(
        root, "target-sizing", target_sizing.get("clients"), CALIBRATION_MAX_TOKENS
    )
    (
        sizing_probe_started,
        sizing_probe_finished,
        sizing_probe_silence,
    ) = validate_rebalance_probe(
        root,
        sizing_probe_client,
        artifact_phase="target-sizing-rebalance-probe",
        role="target-sizing-rebalance-probe",
    )
    prime_started, prime_finished, prime_silence = validate_stream_group(
        root,
        "target-rebalance-prime",
        rebalance_prime.get("clients"),
        REBALANCE_PRIME_MAX_TOKENS,
    )
    probe_started, probe_finished, probe_silence = validate_rebalance_probe(
        root,
        rebalance_probe.get("client"),
        artifact_phase="target-rebalance-probe",
        role="target-rebalance-probe",
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
        validate_replayed_decode_workload(
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
    require(
        not any(
            row.get("phase") == "vnext.decode_capacity_deferred"
            and isinstance(row.get("ts_unix_nanos"), int)
            and sizing_probe_started
            <= row["ts_unix_nanos"]
            <= sizing_probe_finished
            for row in sizing_rows
        ),
        "target sizing rebalance probe unexpectedly hit logical decode pressure",
    )
    sizing_maintenance_summary = validate_maintenance_trace(
        sizing_rows,
        started_wall_ns=sizing_started,
        finished_wall_ns=sizing_finished,
        label="target sizing",
        phase=PREFILL_MAINTENANCE_PHASE,
    )
    require(
        target_sizing.get("maintenance_summary") == sizing_maintenance_summary,
        "target sizing maintenance summary differs from raw trace",
    )
    target_rows = common.read_trace(root / str(target.get("trace")))
    sizing_trace_path = root / str(target_sizing.get("trace"))
    sizing_trace_bytes = sizing_trace_path.stat().st_size
    require(
        sizing_trace_bytes <= STOP_POLICY["max_trace_bytes"],
        "target sizing trace exceeds its byte ceiling",
    )
    target_trace_path = root / str(target.get("trace"))
    target_trace_bytes = target_trace_path.stat().st_size
    require(target_trace_bytes <= STOP_POLICY["max_trace_bytes"], "target trace exceeds its byte ceiling")
    require(
        rebalance_probe["client"]["prompt_tokens"]
        > max(result["prompt_tokens"] for result in target["clients"].values()),
        "rebalance probe did not increase request-shaped token demand",
    )
    require(
        {
            key: sizing_probe_client.get(key)
            for key in ("workload_slot", "prompt_sha256", "prompt_tokens", "max_tokens")
        }
        == {
            key: rebalance_probe["client"].get(key)
            for key in ("workload_slot", "prompt_sha256", "prompt_tokens", "max_tokens")
        },
        "sizing and target rebalance probes did not replay the same workload",
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
        require_maintenance_boundary=True,
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
    rebalance_witness = validate_target_rebalance_witness(
        probe_maintenance_summary,
        target_budget_envelope,
        prime_budget_receipt,
        prime_pool,
        probe_pool,
    )
    require(
        rebalance_probe.get("rebalance_witness") == rebalance_witness,
        "target rebalance witness differs from raw trace and pool receipts",
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
        "rebalance_witness": rebalance_witness,
        "rebalance_evidence_phase": "target-rebalance-probe",
        "sizing_maintenance_summary": sizing_maintenance_summary,
        "sizing_probe_maintenance_summary": sizing_probe_maintenance_summary,
        "probe_maintenance_summary": probe_maintenance_summary,
        "cross_pool_rebalance_evidence_owner": (
            common.CROSS_POOL_REBALANCE_EVIDENCE_OWNER
        ),
        "target_trace_bytes": target_trace_bytes,
        "target_sizing_trace_bytes": sizing_trace_bytes,
        "calibration_window_ns": [calibration_started, calibration_finished],
        "target_sizing_window_ns": [sizing_started, sizing_finished],
        "target_sizing_rebalance_probe_window_ns": [
            sizing_probe_started,
            sizing_probe_finished,
        ],
        "target_rebalance_prime_window_ns": [prime_started, prime_finished],
        "target_rebalance_probe_window_ns": [probe_started, probe_finished],
        "target_window_ns": [target_started, target_finished],
        "decode_counter_provenance": counter_provenance,
        "max_silence_seconds": {
            "calibration": calibration_silence,
            "target_sizing": sizing_silence,
            "target_sizing_rebalance_probe": sizing_probe_silence,
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
        all(
            CALIBRATION_MAX_TOKENS[slot] < TARGET_MAX_TOKENS[slot]
            for slot in PRESSURE_DECODE_SLOTS
        )
        and len(set(TARGET_MAX_TOKENS.values())) == 1,
        "target frontiers must outlive calibration with one long decode horizon",
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
    replay_results = {
        phase: {
            **prompt_result,
            "prompt_tokens": 8,
            "max_tokens": max_tokens,
        }
        for phase, max_tokens in {
            "calibration": CALIBRATION_MAX_TOKENS["B"],
            "target-sizing": CALIBRATION_MAX_TOKENS["B"],
            "target-rebalance-prime": REBALANCE_PRIME_MAX_TOKENS["B"],
            "target": TARGET_MAX_TOKENS["B"],
        }.items()
    }
    validate_replayed_decode_workload("B", replay_results)
    wrong_replay_tokens = json.loads(json.dumps(replay_results))
    wrong_replay_tokens["target"]["max_tokens"] -= 1
    expect_reject(
        lambda: validate_replayed_decode_workload("B", wrong_replay_tokens),
        "decode replay token-budget drift",
    )

    def executor_fixture(
        plan_digit: str, policy_digit: str, reserve_bytes: int
    ) -> dict[str, Any]:
        plan_hash = plan_digit * 64
        return {
            "model_id": MODEL_ID,
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
                "dynamic_storage_profile_order": [
                    {"allocator": "linear_arena", "view": "contiguous"}
                ],
            },
            "runtime_admission_policy": {
                "sequence_fit_policy": "immediate_only",
            },
        }

    calibration_executor = executor_fixture("a", "5", 100)
    sizing_executor = executor_fixture("b", "6", 200)
    target_executor = executor_fixture("c", "7", 300)
    require(
        runtime_memory_usable_bytes(
            calibration_executor["runtime_memory_policy"], "self-test calibration"
        )
        == 900,
        "self-test runtime usable budget drifted",
    )
    invalid_runtime_policy = json.loads(
        json.dumps(calibration_executor["runtime_memory_policy"])
    )
    invalid_runtime_policy["reserve_bytes"] = invalid_runtime_policy["capacity_bytes"]
    expect_reject(
        lambda: runtime_memory_usable_bytes(
            invalid_runtime_policy, "self-test invalid runtime policy"
        ),
        "invalid runtime reserve",
    )
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
    wrong_model = dict(target_executor)
    wrong_model["model_id"] = "Qwen/not-the-collected-model"
    expect_reject(
        lambda: require_executor_identity_shape(wrong_model, "wrong model"),
        "non-canonical model identity",
    )
    valid_snapshot = (
        f"/tmp/hub/{MODEL_CACHE_COMPONENT}/snapshots/" + "e" * 40
    )
    require(
        model_revision_from_snapshot(valid_snapshot) == "e" * 40,
        "canonical model snapshot revision changed",
    )
    expect_reject(
        lambda: model_revision_from_snapshot(
            "/tmp/hub/models--Qwen--another-model/snapshots/" + "e" * 40
        ),
        "wrong model snapshot",
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
            "allocator": {"fixed_block_arena": {"block_bytes": 4}},
            "view": {"paged_regions": {"block_bytes": 4}},
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
                    "physical_allocation_quantum_bytes": 4,
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
        *,
        free_bytes_by_pool: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        resident_bytes = sum(pools.values())
        free_bytes_by_pool = free_bytes_by_pool or pools
        used_bytes_by_pool = {
            pool_id: value - free_bytes_by_pool[pool_id]
            for pool_id, value in pools.items()
        }
        live_segments_by_pool = {
            pool_id: int(used_bytes_by_pool[pool_id] > 0) for pool_id in pools
        }
        transient_occupancy = {pool_id: {} for pool_id in pools}
        lane_stable_occupancy = {pool_id: {} for pool_id in pools}
        return {
            "static_bytes": 100,
            "resident_bytes": resident_bytes,
            "budget_claimed_bytes": 100 + resident_bytes,
            "maximum_active_sequences": MAX_NUM_SEQS,
            "pool_resident_bytes": pools,
            "pool_used_bytes": used_bytes_by_pool,
            "pool_live_segments": live_segments_by_pool,
            "pool_transient_occupancy": transient_occupancy,
            "pool_lane_stable_occupancy": lane_stable_occupancy,
            "startup_baseline": {
                "pool_used_bytes": {pool_id: 0 for pool_id in pools},
                "pool_live_segments": {pool_id: 0 for pool_id in pools},
                "pool_transient_occupancy": transient_occupancy,
                "pool_lane_stable_occupancy": lane_stable_occupancy,
            },
            "pool_envelopes": {
                pool_id: {
                    "domain_id": index + 1,
                    "resident_bytes": value,
                    "resident_chunks": 1,
                    "free_bytes": free_bytes_by_pool[pool_id],
                    "largest_contiguous_bytes": free_bytes_by_pool[pool_id],
                    "live_segments": live_segments_by_pool[pool_id],
                    "live_occupancy": {
                        "transient": transient_occupancy[pool_id],
                        "lane_stable": lane_stable_occupancy[pool_id],
                    },
                    "storage_profile": storage_profiles[pool_id],
                    "contract": contracts[pool_id],
                }
                for index, (pool_id, value) in enumerate(pools.items())
            },
        }

    probe_maintenance = {
        "maintenance_events": 1,
        "maintained_events": 1,
        "allocated_bytes": 4,
        "rebalance_events": 0,
        "pools_reclaimed": 0,
        "chunks_reclaimed": 0,
        "reclaimed_bytes": 0,
        "growth_receipts": [
            {
                "stage": "sequence_extension",
                "allocated_bytes": 4,
                "reclaimed_bytes": 0,
                "reclaims": [],
                "growths": [
                    {
                        "pool_id": "sequence",
                        "chunk_identity": ["sequence", 1, 1],
                        "chunk_bytes": 4,
                        "published_capacity_bytes": 34,
                    }
                ],
            }
        ],
    }
    calibration_pool = pool_snapshot(
        {"sequence": 50, "workspace": 20}, calibration_contracts
    )
    sizing_pool = pool_snapshot({"sequence": 30, "workspace": 10})
    sizing_probe_pool = pool_snapshot({"sequence": 34, "workspace": 10})
    target_envelope = derive_target_budget_envelope(
        calibration_pool,
        sizing_pool,
        sizing_probe_pool,
        probe_maintenance,
    )
    require(
        target_envelope["budget_claimed_bytes"] == 140
        and target_envelope["resident_bytes"] == 40,
        "self-test lost typed target budget derivation",
    )
    require(
        target_envelope["sizing_observed_pool_resident_bytes"]
        == {"sequence": 30, "workspace": 10}
        and target_envelope["probe_observed_pool_resident_bytes"]
        == {"sequence": 34, "workspace": 10}
        and target_envelope["probe_growth_bytes_by_pool"]
        == {"sequence": 4, "workspace": 0}
        and target_envelope["probe_shrink_bytes_by_pool"]
        == {"sequence": 0, "workspace": 0}
        and target_envelope["trace_growth_bytes_by_pool"] == {"sequence": 4}
        and target_envelope["sequence_extension_growth_chunks_by_pool"]
        == {"sequence": [4]}
        and target_envelope["probe_growth_budget_gap_bytes"] == 4
        and target_envelope["requires_cross_pool_rebalance"] is True
        and target_envelope["pressure_quantum_bytes"] == 4
        and target_envelope["pressure_quantum_bytes_by_pool"]
        == {"sequence": 4}
        and target_envelope["pressure_budget_candidate_resident_bytes"] == 40
        and target_envelope["pressure_budget_reduction_from_sizing_probe_bytes"] == 4
        and target_envelope["initial_bundle_floor_bytes_by_pool"]
        == {"sequence": 30, "workspace": 4}
        and target_envelope["minimum_initial_bundle_resident_bytes"] == 34
        and target_envelope["initial_bundle_headroom_bytes"] == 6
        and target_envelope["donor_evidence_kind"]
        == "target_event_exact_receipt_only"
        and target_envelope["token_scaled_sequence_pool_ids"] == ["sequence"]
        and target_envelope["token_scaled_sequence_growth_pool_ids"]
        == ["sequence"]
        and target_envelope["pool_contracts"] == pool_contracts
        and target_envelope["calibration_resident_bytes"] == 70
        and target_envelope["calibration_sizing_delta_bytes"] == 30
        and target_envelope["bootstrap_headroom_bytes"] == 0,
        "self-test lost typed target-sizing provenance",
    )
    wider_calibration_pool = pool_snapshot(
        {"sequence": 60, "workspace": 20}, calibration_contracts
    )
    pressure_envelope = derive_target_budget_envelope(
        wider_calibration_pool,
        sizing_pool,
        sizing_probe_pool,
        probe_maintenance,
    )
    require(
        pressure_envelope["calibration_resident_bytes"] == 80
        and pressure_envelope["pressure_budget_candidate_resident_bytes"] == 40
        and pressure_envelope["resident_bytes"] == 40
        and pressure_envelope["budget_claimed_bytes"] == 140
        and pressure_envelope["pressure_budget_reduction_from_sizing_probe_bytes"] == 4
        and pressure_envelope["probe_growth_budget_gap_bytes"] == 4
        and pressure_envelope["requires_cross_pool_rebalance"] is True,
        "self-test let calibration headroom erase typed decode pressure",
    )
    require_target_pool_within_budget_contract(
        pool_snapshot({"sequence": 34, "workspace": 6}), target_envelope, 140
    )
    exact_budget_contracts = json.loads(json.dumps(pool_contracts))
    exact_budget_contracts["sequence"]["provisioning"][
        "maximum_resident_bytes"
    ] = 34
    exact_budget_contracts["workspace"]["provisioning"][
        "maximum_resident_bytes"
    ] = 6
    exact_budget_target = pool_snapshot(
        {"sequence": 34, "workspace": 6}, exact_budget_contracts
    )
    require_target_pool_within_budget_contract(
        exact_budget_target, target_envelope, 140
    )
    expanded_budget_contracts = json.loads(json.dumps(exact_budget_contracts))
    expanded_budget_contracts["sequence"]["provisioning"][
        "maximum_resident_bytes"
    ] = 101
    expect_reject(
        lambda: require_target_pool_within_budget_contract(
            pool_snapshot(
                {"sequence": 34, "workspace": 6}, expanded_budget_contracts
            ),
            target_envelope,
            140,
        ),
        "target exact-budget pool expanded its sizing resident ceiling",
    )
    undersized_budget_contracts = json.loads(json.dumps(exact_budget_contracts))
    undersized_budget_contracts["sequence"]["provisioning"][
        "maximum_resident_bytes"
    ] = 33
    expect_reject(
        lambda: require_target_pool_within_budget_contract(
            pool_snapshot(
                {"sequence": 34, "workspace": 6}, undersized_budget_contracts
            ),
            target_envelope,
            140,
        ),
        "target exact-budget pool ceiling fell below installed residency",
    )
    prime_receipt = rebalance_prime_budget_receipt(
        sizing_pool,
        sizing_pool,
        target_envelope,
        140,
    )
    expected_layout_receipt = require_replayed_prime_layout(sizing_pool, sizing_pool)
    require(
        prime_receipt
        == {
            "budget_ceiling_bytes": 140,
            "claimed_bytes": 140,
            "headroom_bytes": 0,
            "resident_ceiling_bytes": 40,
            "resident_bytes": 40,
            "replayed_probe_growth_bytes": 4,
            "selected_pressure_growth_end_bytes": 4,
            "selected_pressure_event_ordinal": 0,
            "forced_deficit_bytes": 4,
            "pressure_quantum_bytes": 4,
            "replayed_prime_layout": expected_layout_receipt,
        },
        "self-test lost bounded rebalance-prime headroom evidence",
    )
    exact_budget_prime_contracts = json.loads(json.dumps(pool_contracts))
    exact_budget_prime_contracts["sequence"]["provisioning"][
        "maximum_resident_bytes"
    ] = 40
    exact_budget_prime_contracts["workspace"]["provisioning"][
        "maximum_resident_bytes"
    ] = 40
    exact_budget_prime = pool_snapshot(
        {"sequence": 30, "workspace": 10}, exact_budget_prime_contracts
    )
    require(
        rebalance_prime_budget_receipt(
            exact_budget_prime,
            sizing_pool,
            target_envelope,
            140,
        )
        == prime_receipt,
        "self-test treated the exact-budget pool ceiling as physical layout drift",
    )
    expect_reject(
        lambda: rebalance_prime_budget_receipt(
            pool_snapshot({"sequence": 28, "workspace": 8}),
            sizing_pool,
            target_envelope,
            140,
        ),
        "fresh target prime physical layout drift",
    )
    swapped_prime_layout = pool_snapshot({"sequence": 26, "workspace": 14})
    expect_reject(
        lambda: rebalance_prime_budget_receipt(
            swapped_prime_layout,
            sizing_pool,
            target_envelope,
            140,
        ),
        "same-total fresh target prime with swapped pool distribution",
    )
    target_probe_pool = pool_snapshot({"sequence": 34, "workspace": 6})
    target_witness_summary = {
        "allocated_bytes": 4,
        "reclaimed_bytes": 4,
        "growth_receipts": [
            {
                "stage": "sequence_extension",
                "allocated_bytes": 4,
                "reclaimed_bytes": 4,
                "growths": [
                    {
                        "pool_id": "sequence",
                        "chunk_identity": ["sequence", 2, 1],
                        "chunk_bytes": 4,
                        "published_capacity_bytes": 34,
                    }
                ],
                "reclaims": [
                    {
                        "pool_id": "workspace",
                        "reclaimed_bytes": 4,
                        "chunk_identities": [["workspace", 1, 1]],
                        "published_capacity_bytes": 6,
                    }
                ],
                "event_fingerprint": "f" * 64,
            }
        ],
    }
    witness_receipt = validate_target_rebalance_witness(
        target_witness_summary,
        target_envelope,
        prime_receipt,
        sizing_pool,
        target_probe_pool,
    )
    require(
        witness_receipt["actual_growth_bytes_by_pool"] == {"sequence": 4}
        and witness_receipt["actual_reclaimed_bytes_by_pool"] == {"workspace": 4}
        and witness_receipt["forced_deficit_bytes"] == 4
        and witness_receipt["qualifying_events"][0]["growth_pool_ids"]
        == ["sequence"]
        and witness_receipt["qualifying_events"][0]["reclaim_pool_ids"]
        == ["workspace"],
        "self-test lost the exact target rebalance witness",
    )
    prefixed_maintenance = json.loads(json.dumps(probe_maintenance))
    prefixed_maintenance.update({"maintenance_events": 2, "maintained_events": 2, "allocated_bytes": 8})
    prefixed_maintenance["growth_receipts"] = [
        {
            "stage": "step_admission",
            "allocated_bytes": 4,
            "reclaimed_bytes": 0,
            "growths": [
                {
                    "pool_id": "workspace",
                    "chunk_identity": ["workspace", 2, 1],
                    "chunk_bytes": 4,
                    "published_capacity_bytes": 14,
                }
            ],
            "reclaims": [],
            "event_fingerprint": "d" * 64,
        },
        {
            "stage": "sequence_extension",
            "allocated_bytes": 4,
            "reclaimed_bytes": 0,
            "growths": [
                {
                    "pool_id": "sequence",
                    "chunk_identity": ["sequence", 2, 1],
                    "chunk_bytes": 4,
                    "published_capacity_bytes": 34,
                }
            ],
            "reclaims": [],
            "event_fingerprint": "e" * 64,
        },
    ]
    prefixed_envelope = derive_target_budget_envelope(
        calibration_pool,
        sizing_pool,
        pool_snapshot({"sequence": 34, "workspace": 14}),
        prefixed_maintenance,
    )
    require(
        prefixed_envelope["budget_claimed_bytes"] == 144
        and prefixed_envelope["selected_pressure_event_ordinal"] == 1
        and prefixed_envelope["selected_pressure_growth_end_bytes"] == 8
        and prefixed_envelope["probe_growth_headroom_bytes"] == 4
        and prefixed_envelope["selected_pressure_forced_deficit_bytes"] == 4,
        "event-specific pressure budget ignored preceding growth",
    )
    prefixed_prime_receipt = rebalance_prime_budget_receipt(
        sizing_pool,
        sizing_pool,
        prefixed_envelope,
        144,
    )
    prefixed_target_summary = json.loads(json.dumps(prefixed_maintenance))
    prefixed_target_summary.update(
        {
            "rebalance_events": 1,
            "pools_reclaimed": 1,
            "chunks_reclaimed": 1,
            "reclaimed_bytes": 4,
        }
    )
    prefixed_target_summary["growth_receipts"][1]["reclaimed_bytes"] = 4
    prefixed_target_summary["growth_receipts"][1]["reclaims"] = [
        {
            "pool_id": "workspace",
            "reclaimed_bytes": 4,
            "chunk_identities": [["workspace", 2, 1]],
            "published_capacity_bytes": 10,
        }
    ]
    prefixed_witness = validate_target_rebalance_witness(
        prefixed_target_summary,
        prefixed_envelope,
        prefixed_prime_receipt,
        sizing_pool,
        pool_snapshot({"sequence": 34, "workspace": 10}),
    )
    require(
        prefixed_witness["qualifying_events"][0]["event_ordinal"] == 1
        and prefixed_witness["forced_deficit_bytes"] == 4,
        "target witness did not bind reclaim to the selected pressure event",
    )
    early_reclaim_summary = json.loads(json.dumps(prefixed_target_summary))
    early_reclaim_summary["reclaimed_bytes"] = 8
    early_reclaim_summary["growth_receipts"][0]["reclaimed_bytes"] = 4
    early_reclaim_summary["growth_receipts"][0]["reclaims"] = [
        {
            "pool_id": "sequence",
            "reclaimed_bytes": 4,
            "chunk_identities": [["sequence", 1, 1]],
            "published_capacity_bytes": 26,
        }
    ]
    expect_reject(
        lambda: validate_target_rebalance_witness(
            early_reclaim_summary,
            prefixed_envelope,
            prefixed_prime_receipt,
            sizing_pool,
            pool_snapshot({"sequence": 30, "workspace": 10}),
        ),
        "target reclaimed headroom before the selected pressure event",
    )
    wrong_target_stage = json.loads(json.dumps(target_witness_summary))
    wrong_target_stage["growth_receipts"][0]["stage"] = "step_admission"
    expect_reject(
        lambda: validate_target_rebalance_witness(
            wrong_target_stage,
            target_envelope,
            prime_receipt,
            sizing_pool,
            target_probe_pool,
        ),
        "target reclaim not bound to sequence-extension growth",
    )
    same_pool_target_reclaim = json.loads(json.dumps(target_witness_summary))
    same_pool_target_reclaim["growth_receipts"][0]["reclaims"][0]["pool_id"] = (
        "sequence"
    )
    expect_reject(
        lambda: validate_target_rebalance_witness(
            same_pool_target_reclaim,
            target_envelope,
            prime_receipt,
            sizing_pool,
            target_probe_pool,
        ),
        "target event reused its growth pool as donor",
    )
    insufficient_target_reclaim = json.loads(json.dumps(target_witness_summary))
    insufficient_target_reclaim["reclaimed_bytes"] = 3
    insufficient_target_reclaim["growth_receipts"][0]["reclaimed_bytes"] = 3
    insufficient_target_reclaim["growth_receipts"][0]["reclaims"][0][
        "reclaimed_bytes"
    ] = 3
    expect_reject(
        lambda: validate_target_rebalance_witness(
            insufficient_target_reclaim,
            target_envelope,
            prime_receipt,
            sizing_pool,
            pool_snapshot({"sequence": 34, "workspace": 7}),
        ),
        "target event reclaimed less than the actual forced deficit",
    )
    expect_reject(
        lambda: validate_target_rebalance_witness(
            target_witness_summary,
            target_envelope,
            prime_receipt,
            sizing_pool,
            pool_snapshot({"sequence": 34, "workspace": 7}),
        ),
        "target probe snapshot broke exact growth/reclaim conservation",
    )
    try:
        require_target_pool_within_budget_contract(
            pool_snapshot({"sequence": 35, "workspace": 10}), target_envelope, 140
        )
        raise AssertionError("oversized global target residency unexpectedly fit its sizing envelope")
    except common.CapacityGateError:
        pass
    opaque_sizing = json.loads(json.dumps(sizing_pool))
    del opaque_sizing["pool_envelopes"]["sequence"]["contract"]
    expect_reject(
        lambda: derive_target_budget_envelope(
            calibration_pool,
            opaque_sizing,
            sizing_probe_pool,
            probe_maintenance,
        ),
        "opaque target sizing pool",
    )
    coefficient_drift = json.loads(json.dumps(sizing_pool))
    coefficient_drift["pool_envelopes"]["sequence"]["contract"]["resources"][0][
        "demand"
    ]["tokens"]["bytes_per_token"] = 11
    expect_reject(
        lambda: derive_target_budget_envelope(
            calibration_pool,
            coefficient_drift,
            sizing_probe_pool,
            probe_maintenance,
        ),
        "phase-stable token coefficient drift",
    )
    mixed_sequence_contracts = json.loads(json.dumps(pool_contracts))
    mixed_sequence_contracts["sequence"]["resources"].append(
        {
            "resource_id": "resource/sequence-fixed",
            "demand": {"fixed": {"bytes": 4}},
            "lifetime": "sequence",
            "kind": "value",
            "physical_allocation_quantum_bytes": 4,
            "initialization": "none",
        }
    )
    mixed_calibration_contracts = json.loads(json.dumps(mixed_sequence_contracts))
    mixed_calibration_contracts["sequence"]["resources"][0]["demand"]["tokens"][
        "maximum_tokens"
    ] = 3
    mixed_calibration_contracts["sequence"]["provisioning"][
        "maximum_resident_bytes"
    ] = 90
    expect_reject(
        lambda: derive_target_budget_envelope(
            pool_snapshot(
                {"sequence": 50, "workspace": 20}, mixed_calibration_contracts
            ),
            pool_snapshot({"sequence": 30, "workspace": 10}, mixed_sequence_contracts),
            pool_snapshot({"sequence": 34, "workspace": 10}, mixed_sequence_contracts),
            probe_maintenance,
        ),
        "ambiguous mixed token/fixed sequence pool",
    )
    probe_contract_drift = json.loads(json.dumps(sizing_probe_pool))
    probe_contract_drift["pool_envelopes"]["sequence"]["contract"]["resources"][0][
        "demand"
    ]["tokens"]["maximum_tokens"] = 11
    expect_reject(
        lambda: derive_target_budget_envelope(
            calibration_pool,
            sizing_pool,
            probe_contract_drift,
            probe_maintenance,
        ),
        "same-process sizing prime/probe contract drift",
    )
    non_sequence_growth = json.loads(json.dumps(probe_maintenance))
    non_sequence_growth["growth_receipts"][0]["growths"][0].update(
        {
            "pool_id": "workspace",
            "chunk_identity": ["workspace", 1, 1],
            "published_capacity_bytes": 14,
        }
    )
    non_sequence_probe = pool_snapshot({"sequence": 30, "workspace": 14})
    expect_reject(
        lambda: derive_target_budget_envelope(
            calibration_pool,
            sizing_pool,
            non_sequence_probe,
            non_sequence_growth,
        ),
        "probe without token-scaled sequence growth",
    )
    subquantum_maintenance = json.loads(json.dumps(probe_maintenance))
    subquantum_maintenance["allocated_bytes"] = 3
    subquantum_maintenance["growth_receipts"][0]["allocated_bytes"] = 3
    subquantum_maintenance["growth_receipts"][0]["growths"][0]["chunk_bytes"] = 3
    subquantum_maintenance["growth_receipts"][0]["growths"][0][
        "published_capacity_bytes"
    ] = 33
    subquantum_probe = pool_snapshot({"sequence": 33, "workspace": 10})
    expect_reject(
        lambda: derive_target_budget_envelope(
            calibration_pool,
            sizing_pool,
            subquantum_probe,
            subquantum_maintenance,
        ),
        "sub-quantum persistent sequence growth",
    )
    misaligned_growth = json.loads(json.dumps(probe_maintenance))
    misaligned_growth["allocated_bytes"] = 5
    misaligned_growth["growth_receipts"][0]["allocated_bytes"] = 5
    misaligned_growth["growth_receipts"][0]["growths"][0]["chunk_bytes"] = 5
    misaligned_growth["growth_receipts"][0]["growths"][0][
        "published_capacity_bytes"
    ] = 35
    misaligned_probe = pool_snapshot({"sequence": 35, "workspace": 10})
    expect_reject(
        lambda: derive_target_budget_envelope(
            calibration_pool,
            sizing_pool,
            misaligned_probe,
            misaligned_growth,
        ),
        "non-quantized sequence growth receipt",
    )
    split_misaligned_growth = json.loads(json.dumps(probe_maintenance))
    split_misaligned_growth["growth_receipts"][0]["growths"] = [
        {
            "pool_id": "sequence",
            "chunk_identity": ["sequence", 1, 1],
            "chunk_bytes": 3,
            "published_capacity_bytes": 33,
        },
        {
            "pool_id": "sequence",
            "chunk_identity": ["sequence", 2, 1],
            "chunk_bytes": 1,
            "published_capacity_bytes": 34,
        },
    ]
    expect_reject(
        lambda: derive_target_budget_envelope(
            calibration_pool,
            sizing_pool,
            sizing_probe_pool,
            split_misaligned_growth,
        ),
        "misaligned growth chunks hidden by an aligned aggregate",
    )
    mixed_event_growth = json.loads(json.dumps(probe_maintenance))
    mixed_event_growth["allocated_bytes"] = 5
    mixed_event_growth["growth_receipts"][0]["allocated_bytes"] = 5
    mixed_event_growth["growth_receipts"][0]["growths"].append(
        {
            "pool_id": "workspace",
            "chunk_identity": ["workspace", 2, 1],
            "chunk_bytes": 1,
            "published_capacity_bytes": 11,
        }
    )
    expect_reject(
        lambda: derive_target_budget_envelope(
            calibration_pool,
            sizing_pool,
            pool_snapshot({"sequence": 34, "workspace": 11}),
            mixed_event_growth,
        ),
        "sequence pressure event with unattributed pool growth",
    )
    sizing_rebalanced = json.loads(json.dumps(probe_maintenance))
    sizing_rebalanced.update(
        {
            "rebalance_events": 1,
            "pools_reclaimed": 1,
            "chunks_reclaimed": 1,
            "reclaimed_bytes": 4,
        }
    )
    sizing_rebalanced["growth_receipts"][0]["reclaimed_bytes"] = 4
    sizing_rebalanced["growth_receipts"][0]["reclaims"] = [
        {
            "pool_id": "workspace",
            "reclaimed_bytes": 4,
            "chunk_identities": [["workspace", 1, 1]],
            "published_capacity_bytes": 6,
        }
    ]
    expect_reject(
        lambda: derive_target_budget_envelope(
            calibration_pool,
            sizing_pool,
            pool_snapshot({"sequence": 34, "workspace": 6}),
            sizing_rebalanced,
        ),
        "sizing probe contaminated by cross-pool rebalance",
    )
    low_prime = pool_snapshot({"sequence": 24, "workspace": 6})
    low_prime_probe = pool_snapshot({"sequence": 28, "workspace": 6})
    expect_reject(
        lambda: derive_target_budget_envelope(
            calibration_pool,
            low_prime,
            low_prime_probe,
            probe_maintenance,
        ),
        "initial floor consumed the full observed growth",
    )
    narrower_calibration = pool_snapshot(
        {"sequence": 30, "workspace": 9}, calibration_contracts
    )
    narrower_calibration_envelope = derive_target_budget_envelope(
        narrower_calibration,
        sizing_pool,
        sizing_probe_pool,
        probe_maintenance,
    )
    require(
        narrower_calibration_envelope["budget_claimed_bytes"] == 140
        and narrower_calibration_envelope["resident_bytes"] == 40
        and narrower_calibration_envelope["calibration_sizing_delta_bytes"] == -1
        and narrower_calibration_envelope[
            "pressure_budget_reduction_from_sizing_probe_bytes"
        ]
        == 4,
        "calibration footprint incorrectly capped the unpressured sizing baseline",
    )
    target_bound_drift = json.loads(
        json.dumps(pool_snapshot({"sequence": 34, "workspace": 6}))
    )
    target_bound_drift["pool_envelopes"]["sequence"]["contract"]["resources"][0][
        "demand"
    ]["tokens"]["maximum_tokens"] = 11
    expect_reject(
        lambda: require_target_pool_within_budget_contract(
            target_bound_drift, target_envelope, 140
        ),
        "target runtime-bound contract drift",
    )

    wait_condition = {
        "coordinator_id": 7,
        "observed": [{"source": {"domain": 5}, "epoch": 3}],
    }
    logical_shortfalls = [
        {
            "domain": 5,
            "kind": "fit_availability",
            "requested": 2,
            "available": 1,
            "current_total": 1,
            "maximum_total": 2,
        }
    ]
    capacity_evidence = {
        "observed": {"coordinator_id": 7, "release_epoch": 11, "capacity_epoch": 13},
        "wait_condition": wait_condition,
        "shortfalls": logical_shortfalls,
        "backing_blockers": [],
        "typed_evidence": {
            "owner": "logical",
            "kind": "logical",
            "shortfalls": logical_shortfalls,
        },
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
        execution_stage: str = "step_admission",
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
                "execution_stage": execution_stage,
                "decode_submit_observed": False,
            },
            "attributes": attributes,
        }

    owner_cohort_handoff = validate_decode_deferral(
        deferral(
            90,
            "pressure_yield_planned",
            ["A"],
            victim_request_id="C",
            progress_owner_id="A",
            progress_baseline=52,
            episode_id=1,
            planned_transition_ordinal=1,
            yield_kind="peer_handoff",
            execution_stage="sequence_extension",
        ),
        "owner-cohort peer handoff",
    )
    require(
        owner_cohort_handoff["victim_request_id"] == "C"
        and owner_cohort_handoff["progress_owner_id"] == "A",
        "self-test lost a peer handoff selected from the progress-owner cohort",
    )
    expect_reject(
        lambda: validate_decode_deferral(
            deferral(
                91,
                "pressure_yield_planned",
                ["B"],
                victim_request_id="C",
                progress_owner_id="A",
                progress_baseline=52,
                episode_id=1,
                planned_transition_ordinal=2,
                yield_kind="peer_handoff",
                execution_stage="sequence_extension",
            ),
            "foreign-cohort peer handoff",
        ),
        "peer handoff whose victim and owner both differ from the failing cohort",
    )

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
            execution_stage="sequence_extension",
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
                "hold_transition_ordinal": 5,
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
            progress_baseline=53,
            episode_id=1,
            planned_transition_ordinal=7,
            yield_kind="self_recompute",
            execution_stage="sequence_extension",
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
                execution_stage="sequence_extension",
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
    donor_pool_id = "dynamic-pool/sha256/" + "a" * 64
    growth_pool_id = "dynamic-pool/sha256/" + "b" * 64
    event_fingerprint = "c" * 64
    active_fingerprint = "d" * 64
    exact_rebalance_receipt = {
        "pools": [
            {
                "pool_id": donor_pool_id,
                "chunks": [
                    {
                        "pool_id": donor_pool_id,
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
    }
    maintenance_boundary = {
        "schema_version": MAINTENANCE_BOUNDARY_SCHEMA_VERSION,
        "coordinator_id": 7,
        "logical_release_epoch": 13,
        "logical_capacity_epoch": 14,
        "plan_device_capacity_epoch": 15,
        "process_device_capacity_epoch": 15,
        "pressure": {
            "scope": "plan_budget",
            "device_id": "device.self-test",
            "requested_bytes": 32,
            "plan_claimed_bytes": 192,
            "plan_usable_bytes": 192,
            "process_claimed_bytes": 192,
            "process_usable_bytes": 192,
        },
        "planned_domains": [2],
        "protected_immediate": [],
        "protected_packing_envelopes": [],
        "pools": [
            {
                "pool_id": donor_pool_id,
                "domain_id": 1,
                "excluded_from_reclaim": False,
                "resident_bytes": 192,
                "pending_growth_bytes": 0,
                "free_bytes": 192,
                "largest_contiguous_bytes": 128,
                "free_extent_layout_fingerprint": "sha256/" + "1" * 64,
                "logical_used_bytes": 0,
                "live_occupancy": {"total": {"physical_bytes": 0}},
                "minimum_resident_bytes": 128,
                "maximum_resident_bytes": 192,
                "protected_immediate_bytes": 0,
                "protected_packing_satisfied": True,
                "coherent_runnable_floor_bytes": 0,
                "resident_floor_bytes": 128,
                "reclaimable_bytes": 64,
                "chunks": [
                    {
                        "identity": {
                            "pool_id": donor_pool_id,
                            "ordinal": 1,
                            "generation": 2,
                        },
                        "bytes": 64,
                        "live_segments": 0,
                        "external_references": 0,
                        "protected_packing": False,
                        "full_extent_available": True,
                        "resident_floor_allows_reclaim": True,
                        "reclaim_candidate": True,
                    },
                    {
                        "identity": {
                            "pool_id": donor_pool_id,
                            "ordinal": 2,
                            "generation": 3,
                        },
                        "bytes": 128,
                        "live_segments": 0,
                        "external_references": 0,
                        "protected_packing": False,
                        "full_extent_available": True,
                        "resident_floor_allows_reclaim": False,
                        "reclaim_candidate": False,
                    },
                ],
            },
            {
                "pool_id": growth_pool_id,
                "domain_id": 2,
                "excluded_from_reclaim": True,
                "resident_bytes": 0,
                "pending_growth_bytes": 0,
                "free_bytes": 0,
                "largest_contiguous_bytes": 0,
                "free_extent_layout_fingerprint": "sha256/" + "2" * 64,
                "logical_used_bytes": 0,
                "live_occupancy": {"total": {"physical_bytes": 0}},
                "minimum_resident_bytes": 0,
                "maximum_resident_bytes": 32,
                "protected_immediate_bytes": 0,
                "protected_packing_satisfied": True,
                "coherent_runnable_floor_bytes": 0,
                "resident_floor_bytes": 0,
                "reclaimable_bytes": 0,
                "chunks": [],
            },
        ],
        "reclaim_candidate_chunks": 1,
        "reclaim_candidate_bytes": 64,
        "selected_chunks": [
            {
                "pool_id": donor_pool_id,
                "ordinal": 1,
                "generation": 2,
            }
        ],
        "selected_bytes": 64,
        "reclaim_sufficient": True,
    }
    rebalance_rows = [
        {
            "ts_unix_nanos": 85,
            "event_id": (
                "evt-vnext-execution-resource-maintenance-" + event_fingerprint
            ),
            "correlation_id": event_fingerprint,
            "request_id": "request.self-test",
            "phase": EXECUTION_MAINTENANCE_PHASE,
            "status": "ok",
            "error": None,
            "shape": {
                "allocated_bytes": 32,
                "participant_count": 1,
                "pools_grown": 1,
                "pools_reclaimed": 1,
                "stage": "step_admission",
            },
            "attributes": {
                "execution_trace_source": "vnext_resource_maintenance",
                "plan_hash": "e" * 64,
                "plan_id": "plan.self-test",
                "run_id": "run.self-test",
                "maintenance_evidence": {
                    "schema_version": EXECUTION_MAINTENANCE_SCHEMA_VERSION,
                    "outcome": "maintained",
                    "stage": "step_admission",
                    "coordinator_id": 7,
                    "pools_grown": 1,
                    "allocated_bytes": 32,
                    "pools_reclaimed": 1,
                    "chunks_reclaimed": 1,
                    "reclaimed_bytes": 64,
                    "rebalance": exact_rebalance_receipt,
                    "maintenance_boundary": maintenance_boundary,
                    "receipt": {
                        "coordinator_id": 7,
                        "growths": [
                            {
                                "pool_id": growth_pool_id,
                                "chunk": {
                                    "pool_id": growth_pool_id,
                                    "ordinal": 2,
                                    "generation": 3,
                                },
                                "chunk_bytes": 32,
                                "published_capacity_bytes": 32,
                                "capacity_epoch": 18,
                            }
                        ],
                        "capacity_epoch": 18,
                        "rebalance": exact_rebalance_receipt,
                        "maintenance_boundary": maintenance_boundary,
                    },
                    "event_fingerprint": event_fingerprint,
                    "participants": [
                        {
                            "run_id": "run.self-test",
                            "request_id": "request.self-test",
                            "sequence_authority": {
                                "sparse_id": 0,
                                "generation": 1,
                            },
                            "active_sequence_fingerprint": active_fingerprint,
                        }
                    ],
                },
            },
        }
    ]
    summary = validate_decode_trace(rows, started_wall_ns=90, finished_wall_ns=170)
    insufficient_boundary = json.loads(json.dumps(maintenance_boundary))
    insufficient_boundary["pressure"]["requested_bytes"] = 96
    insufficient_boundary["reclaim_sufficient"] = False
    boundary_rows = json.loads(json.dumps(rows))
    boundary_evidence = boundary_rows[0]["attributes"]["capacity_evidence"]
    boundary_evidence["typed_evidence"]["pressure"] = {
        "kind": "device_capacity",
        "evidence": insufficient_boundary["pressure"],
    }
    boundary_evidence["typed_evidence"][
        "maintenance_boundary"
    ] = insufficient_boundary
    boundary_summary = validate_decode_trace(
        boundary_rows,
        started_wall_ns=90,
        finished_wall_ns=170,
        require_maintenance_boundary=True,
    )
    require(
        boundary_summary["maintenance_boundary_events"] == 1
        and boundary_summary["maintenance_boundary_deficits"] == [96],
        "self-test lost the event-bound insufficient-reclaim receipt",
    )
    missing_decode_boundary = json.loads(json.dumps(boundary_rows))
    missing_decode_boundary[0]["attributes"]["capacity_evidence"][
        "typed_evidence"
    ].pop("maintenance_boundary")
    expect_reject(
        lambda: validate_decode_trace(
            missing_decode_boundary,
            started_wall_ns=90,
            finished_wall_ns=170,
            require_maintenance_boundary=True,
        ),
        "decode pressure without an event-bound maintenance receipt",
    )
    direct_counter_provenance = validate_decode_counter_provenance(
        rows,
        started_wall_ns=90,
        finished_wall_ns=170,
        counters={
            "extension_deferrals": summary["pressure_yield_events"],
            "step_deferrals": (
                summary["deferral_events"] - summary["pressure_yield_events"]
            ),
            "wave_deferrals": 0,
            "backing_deferrals": 0,
        },
    )
    require(
        direct_counter_provenance["direct_trace_events_by_stage"]
        == {
            "sequence_extension": summary["pressure_yield_events"],
            "step_admission": (
                summary["deferral_events"] - summary["pressure_yield_events"]
            ),
            "submission_wave": 0,
        },
        "self-test lost direct per-stage counter provenance",
    )
    backing_rows = json.loads(json.dumps(rows))
    for row in backing_rows:
        if row.get("phase") != "vnext.decode_capacity_deferred":
            continue
        evidence = row["attributes"]["capacity_evidence"]
        evidence["shortfalls"] = []
        blockers = [
            {
                "pool_id": "dynamic-pool/sha256/" + "a" * 64,
                "domain_id": 5,
                "reason": "growth_required",
                "requested_bytes": 64,
                "free_bytes": 0,
                "largest_contiguous_bytes": 0,
                "free_extent_layout_fingerprint": "sha256/" + "b" * 64,
            }
        ]
        evidence["backing_blockers"] = blockers
        evidence["typed_evidence"] = {
            "owner": "backing",
            "kind": "backing_deferred",
            "blockers": blockers,
        }
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
    direct_pressure_rows = json.loads(json.dumps(rows))
    for row in direct_pressure_rows:
        if row.get("phase") != "vnext.decode_capacity_deferred":
            continue
        evidence = row["attributes"]["capacity_evidence"]
        evidence["shortfalls"] = []
        evidence["backing_blockers"] = []
        evidence["typed_evidence"] = {
            "owner": "backing",
            "kind": "backing_pressure",
            "pressure": {
                "kind": "device_capacity",
                "evidence": {
                    "scope": "plan_budget",
                    "device_id": "device.self-test",
                    "requested_bytes": 1,
                    "plan_claimed_bytes": 1,
                    "plan_usable_bytes": 1,
                    "process_claimed_bytes": 1,
                    "process_usable_bytes": 1,
                },
            },
        }
    direct_pressure_provenance = validate_decode_counter_provenance(
        direct_pressure_rows,
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
        direct_pressure_provenance["device_backing_trace_events"]
        == summary["deferral_events"],
        "self-test lost direct backing-pressure provenance",
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
    ambiguous_rows = json.loads(json.dumps(backing_rows))
    ambiguous_rows[0]["attributes"]["capacity_evidence"]["shortfalls"] = [
        {
            "domain": 5,
            "kind": "fit_availability",
            "requested": 2,
            "available": 1,
            "current_total": 1,
            "maximum_total": 2,
        }
    ]
    expect_reject(
        lambda: validate_decode_counter_provenance(
            ambiguous_rows,
            started_wall_ns=90,
            finished_wall_ns=170,
            counters={
                "extension_deferrals": 1,
                "step_deferrals": 1,
                "wave_deferrals": 0,
                "backing_deferrals": summary["deferral_events"],
            },
        ),
        "ambiguous logical/backing deferral ownership",
    )
    untyped_rows = json.loads(json.dumps(rows))
    untyped_rows[0]["attributes"]["capacity_evidence"].pop("typed_evidence")
    expect_reject(
        lambda: validate_decode_counter_provenance(
            untyped_rows,
            started_wall_ns=90,
            finished_wall_ns=170,
            counters={
                "extension_deferrals": summary["pressure_yield_events"],
                "step_deferrals": (
                    summary["deferral_events"] - summary["pressure_yield_events"]
                ),
                "wave_deferrals": 0,
                "backing_deferrals": 0,
            },
        ),
        "missing typed deferral ownership",
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
    require(
        rebalance_summary["growth_receipts"]
        == [
            {
                "stage": "step_admission",
                "coordinator_id": 7,
                "capacity_epoch": 18,
                "allocated_bytes": 32,
                "reclaimed_bytes": 64,
                "growths": [
                    {
                        "pool_id": growth_pool_id,
                        "chunk_identity": [growth_pool_id, 2, 3],
                        "chunk_bytes": 32,
                        "published_capacity_bytes": 32,
                    }
                ],
                "reclaims": [
                    {
                        "pool_id": donor_pool_id,
                        "reclaimed_bytes": 64,
                        "chunk_identities": [[donor_pool_id, 1, 2]],
                        "published_capacity_bytes": 128,
                    }
                ],
                "maintenance_boundary": {
                    "schema_version": MAINTENANCE_BOUNDARY_SCHEMA_VERSION,
                    "coordinator_id": 7,
                    "logical_release_epoch": 13,
                    "logical_capacity_epoch": 14,
                    "plan_device_capacity_epoch": 15,
                    "process_device_capacity_epoch": 15,
                    "pressure": maintenance_boundary["pressure"],
                    "deficit_bytes": 32,
                    "planned_domains": [2],
                    "pool_ids": [donor_pool_id, growth_pool_id],
                    "reclaim_candidate_chunks": 1,
                    "reclaim_candidate_bytes": 64,
                    "selected_chunk_identities": [[donor_pool_id, 1, 2]],
                    "selected_bytes": 64,
                },
                "growth_chunk_identities": [[growth_pool_id, 2, 3]],
                "participant_identities": [
                    ["run.self-test", "request.self-test", 0, 1]
                ],
                "event_fingerprint": event_fingerprint,
            }
        ],
        "self-test lost the exact growth, participant, or event identity",
    )
    same_pool_growth_reclaim = json.loads(json.dumps(rebalance_rows))
    same_pool_growth = same_pool_growth_reclaim[-1]["attributes"][
        "maintenance_evidence"
    ]["receipt"]["growths"][0]
    same_pool_growth["pool_id"] = donor_pool_id
    same_pool_growth["chunk"]["pool_id"] = donor_pool_id
    expect_reject(
        lambda: validate_rebalance_trace(
            same_pool_growth_reclaim, started_wall_ns=80, finished_wall_ns=89
        ),
        "same-event growth and reclaim from one pool",
    )
    prefill_substitution = json.loads(json.dumps(rebalance_rows))
    prefill_substitution[-1]["phase"] = PREFILL_MAINTENANCE_PHASE
    try:
        validate_rebalance_trace(
            prefill_substitution, started_wall_ns=80, finished_wall_ns=89
        )
        raise AssertionError("prefill maintenance substituted for execution evidence")
    except common.CapacityGateError:
        pass
    missing_growth_receipt = json.loads(json.dumps(rebalance_rows))
    del missing_growth_receipt[-1]["attributes"]["maintenance_evidence"]["receipt"]
    try:
        validate_rebalance_trace(
            missing_growth_receipt, started_wall_ns=80, finished_wall_ns=89
        )
        raise AssertionError("execution event without allocator receipt unexpectedly passed")
    except common.CapacityGateError:
        pass
    invalid_growth_epoch = json.loads(json.dumps(rebalance_rows))
    invalid_growth_epoch[-1]["attributes"]["maintenance_evidence"]["receipt"][
        "growths"
    ][0]["capacity_epoch"] = 19
    try:
        validate_rebalance_trace(
            invalid_growth_epoch, started_wall_ns=80, finished_wall_ns=89
        )
        raise AssertionError("growth with a mismatched capacity epoch unexpectedly passed")
    except common.CapacityGateError:
        pass
    missing_boundary = json.loads(json.dumps(rebalance_rows))
    missing_boundary[-1]["attributes"]["maintenance_evidence"].pop(
        "maintenance_boundary"
    )
    missing_boundary[-1]["attributes"]["maintenance_evidence"]["receipt"].pop(
        "maintenance_boundary"
    )
    expect_reject(
        lambda: validate_rebalance_trace(
            missing_boundary, started_wall_ns=80, finished_wall_ns=89
        ),
        "rebalance without maintenance boundary",
    )
    invalid_boundary_candidate = json.loads(json.dumps(rebalance_rows))
    invalid_boundary_candidate[-1]["attributes"]["maintenance_evidence"][
        "maintenance_boundary"
    ]["pools"][0]["chunks"][0]["reclaim_candidate"] = False
    invalid_boundary_candidate[-1]["attributes"]["maintenance_evidence"][
        "receipt"
    ]["maintenance_boundary"]["pools"][0]["chunks"][0][
        "reclaim_candidate"
    ] = False
    expect_reject(
        lambda: validate_rebalance_trace(
            invalid_boundary_candidate, started_wall_ns=80, finished_wall_ns=89
        ),
        "maintenance boundary candidate drift",
    )
    mismatched_receipt_rebalance = json.loads(json.dumps(rebalance_rows))
    mismatched_receipt_rebalance[-1]["attributes"]["maintenance_evidence"][
        "receipt"
    ]["rebalance"]["reclaimed_bytes"] = 63
    try:
        validate_rebalance_trace(
            mismatched_receipt_rebalance, started_wall_ns=80, finished_wall_ns=89
        )
        raise AssertionError("divergent event and allocator receipts unexpectedly passed")
    except common.CapacityGateError:
        pass
    missing_rebalance = json.loads(json.dumps(rebalance_rows))
    missing_rebalance[-1]["shape"]["pools_reclaimed"] = 0
    missing_rebalance[-1]["attributes"]["maintenance_evidence"].update(
        {
            "pools_reclaimed": 0,
            "chunks_reclaimed": 0,
            "reclaimed_bytes": 0,
            "rebalance": None,
            "maintenance_boundary": None,
        }
    )
    missing_rebalance[-1]["attributes"]["maintenance_evidence"]["receipt"].pop(
        "rebalance"
    )
    missing_rebalance[-1]["attributes"]["maintenance_evidence"]["receipt"].pop(
        "maintenance_boundary"
    )
    growth_only_summary = validate_maintenance_trace(
        missing_rebalance,
        started_wall_ns=80,
        finished_wall_ns=89,
        label="self-test growth-only probe",
        phase=EXECUTION_MAINTENANCE_PHASE,
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
        summary["pressure_yield_stages"] == ["sequence_extension"],
        "self-test lost sequence-extension pressure provenance",
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
    non_sequence_yield = json.loads(json.dumps(rows))
    for row in non_sequence_yield:
        if (
            row.get("phase") == "vnext.decode_capacity_deferred"
            and row.get("shape", {}).get("decision") == "pressure_yield_planned"
        ):
            row["shape"]["execution_stage"] = "submission_wave"
    try:
        validate_decode_trace(
            non_sequence_yield, started_wall_ns=90, finished_wall_ns=170
        )
        raise AssertionError("non-sequence pressure yield unexpectedly passed")
    except common.CapacityGateError:
        pass
    missing_victim = json.loads(json.dumps(rows))
    del missing_victim[5]["attributes"]["victim_request_id"]
    try:
        validate_decode_trace(missing_victim, started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("pressure yield without a victim unexpectedly passed")
    except common.CapacityGateError:
        pass
    drifting_episode_baseline = json.loads(json.dumps(rows))
    drifting_episode_baseline[9]["attributes"]["progress_baseline"] = 54
    try:
        validate_decode_trace(
            drifting_episode_baseline, started_wall_ns=90, finished_wall_ns=170
        )
        raise AssertionError("pressure episode with a drifting baseline unexpectedly passed")
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
