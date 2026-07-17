#!/usr/bin/env python3
"""Collect and validate bounded CUDA decode-capacity pressure evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import runtime_vnext_s1_cuda_capacity as common


PASS_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY PASS"
COLLECT_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY COLLECTED"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY FAIL"
CALIBRATION_TOKEN_BUDGET = 3
TARGET_TOKEN_BUDGET = 1024
MAX_NUM_SEQS = 3
MAX_MODEL_LEN = 512
PREFILL_FIRST_UNTIL_ACTIVE = 3
CALIBRATION_MAX_TOKENS = {"A": 128, "B": 1, "C": 16}
TARGET_MAX_TOKENS = {"A": 128, "B": 128, "C": 16}
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
    "prefill_first_until_active": PREFILL_FIRST_UNTIL_ACTIVE,
    "calibration_max_num_batched_tokens": CALIBRATION_TOKEN_BUDGET,
    "target_max_num_batched_tokens": TARGET_TOKEN_BUDGET,
    "calibration_max_tokens": CALIBRATION_MAX_TOKENS,
    "target_sizing_max_tokens": CALIBRATION_MAX_TOKENS,
    "target_budget_derivation": "per_pool_maximum",
    "target_max_tokens": TARGET_MAX_TOKENS,
}
STOP_POLICY = {
    "no_progress_timeout_seconds": common.MAX_PRESSURE_NO_PROGRESS_SECONDS,
    "joint_stream_timeout_seconds": common.MAX_PRESSURE_JOINT_STREAM_SECONDS,
    "max_trace_bytes": common.MAX_PRESSURE_TRACE_BYTES,
    "max_decode_capacity_events": MAX_DECODE_CAPACITY_EVENTS,
}


class DecodeCapacityGateError(common.CapacityGateError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DecodeCapacityGateError(message)


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

    pool_resident_bytes: dict[str, int] = {}
    pool_sources: dict[str, str] = {}
    pool_storage_profiles: dict[str, Any] = {}
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
        pool_resident_bytes[pool_id] = max(calibration_bytes, sizing_bytes)
        if calibration_bytes > sizing_bytes:
            pool_sources[pool_id] = "calibration"
        elif sizing_bytes > calibration_bytes:
            pool_sources[pool_id] = "target_sizing"
        else:
            pool_sources[pool_id] = "equal"
        pool_storage_profiles[pool_id] = calibration_profile

    static_bytes = calibration.get("static_bytes")
    require(isinstance(static_bytes, int) and static_bytes > 0, "invalid sizing static bytes")
    resident_bytes = sum(pool_resident_bytes.values())
    exact_budget = static_bytes + resident_bytes
    require(
        exact_budget >= calibration.get("budget_claimed_bytes", 0)
        and exact_budget >= target_sizing.get("budget_claimed_bytes", 0),
        "derived target budget is smaller than a sizing receipt",
    )
    return {
        "static_bytes": static_bytes,
        "resident_bytes": resident_bytes,
        "budget_claimed_bytes": exact_budget,
        "pool_resident_bytes": pool_resident_bytes,
        "pool_sources": pool_sources,
        "pool_storage_profiles": pool_storage_profiles,
    }


def require_target_pool_within_envelope(
    target: dict[str, Any], envelope: dict[str, Any], exact_budget: int
) -> None:
    require(
        target.get("static_bytes") == envelope.get("static_bytes"),
        "target static bytes differ from its sizing envelope",
    )
    target_pools = target.get("pool_resident_bytes")
    target_envelopes = target.get("pool_envelopes")
    limits = envelope.get("pool_resident_bytes")
    profiles = envelope.get("pool_storage_profiles")
    require(
        isinstance(target_pools, dict)
        and isinstance(limits, dict)
        and target_pools.keys() == limits.keys(),
        "target pool identities differ from its sizing envelope",
    )
    require(
        isinstance(target_envelopes, dict)
        and isinstance(profiles, dict)
        and target_envelopes.keys() == target_pools.keys()
        and profiles.keys() == target_pools.keys(),
        "target sizing profiles are missing",
    )
    for pool_id, resident_bytes in target_pools.items():
        require(
            isinstance(resident_bytes, int) and 0 <= resident_bytes <= limits[pool_id],
            f"target pool {pool_id} exceeded its sizing envelope",
        )
        require(
            target_envelopes[pool_id].get("storage_profile") == profiles[pool_id],
            f"target pool {pool_id} changed storage profile",
        )
    require(
        target.get("budget_claimed_bytes", exact_budget + 1) <= exact_budget,
        "target installed backing exceeded the derived exact budget",
    )


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


def validate_source_retarget(
    previous: Any, current: Any, *, label: str
) -> tuple[dict[str, int], dict[str, int]]:
    require(isinstance(previous, dict), f"{label}: previous wait condition is missing")
    require(isinstance(current, dict), f"{label}: current wait condition is missing")
    previous_coordinator = previous.get("coordinator_id")
    current_coordinator = current.get("coordinator_id")
    require(
        isinstance(previous_coordinator, int) and previous_coordinator > 0,
        f"{label}: previous coordinator id is invalid",
    )
    require(
        isinstance(current_coordinator, int) and current_coordinator > 0,
        f"{label}: current coordinator id is invalid",
    )
    previous_sources = wait_source_epochs(
        previous.get("observed"), f"{label} previous condition"
    )
    current_sources = wait_source_epochs(
        current.get("observed"), f"{label} current condition"
    )
    require(
        previous_coordinator != current_coordinator
        or previous_sources.keys() != current_sources.keys(),
        f"{label}: source-retarget release preserved the old exact topology",
    )
    require(
        all(
            current_sources[source] >= previous_sources[source]
            for source in previous_sources.keys() & current_sources.keys()
        ),
        f"{label}: shared source generation regressed during retarget",
    )
    return previous_sources, current_sources


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
        isinstance(progress_current, int) and progress_current == progress_baseline,
        f"{label}: pressure hold did not preserve its exact progress baseline",
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
        decision
        in {
            "owner_advanced",
            "owner_terminal",
            "role_transferred",
            "source_retargeted",
        },
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
    if decision == "owner_advanced":
        require(
            progress_current > progress_baseline,
            f"{label}: owner-advanced release has no committed progress",
        )
    previous_wait_condition = shape.get("previous_wait_condition")
    current_wait_condition = shape.get("current_wait_condition")
    if decision == "source_retargeted":
        validate_source_retarget(
            previous_wait_condition,
            current_wait_condition,
            label=label,
        )
    else:
        require(
            previous_wait_condition is None and current_wait_condition is None,
            f"{label}: non-retarget release carries source-retarget evidence",
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
        require(
            resumable is None and isinstance(closed, int) and released < closed,
            f"{label}: self-recompute release/closed ordinal order is invalid",
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
            closed is None
            and closed_reason is None
            and disposition == "progress_owner_resumable",
            f"{label}: resumable completion carries a closed disposition",
        )
        completion_ordinal = resumable
    else:
        require(progress_owner_resumable is False, f"{label}: resumable state is not typed")
        require(
            resumable is None and isinstance(closed, int) and released < closed,
            f"{label}: release/closed ordinal order is invalid",
        )
        require(
            closed_reason in {"owner_terminal", "source_retargeted"}
            and disposition == closed_reason,
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
                completed["completion_disposition"] == "self_recompute_queued",
                f"self-recompute episode {episode_id} did not queue reconstruction",
            )
            require(
                not matching_holds,
                f"self-recompute episode {episode_id} incorrectly published a peer hold",
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
            if release["decision"] in {"owner_advanced", "owner_terminal", "source_retargeted"}
        ]
        if yield_kind == "self_recompute":
            require(
                not matching_releases,
                f"self-recompute episode {episode_id} incorrectly released a peer hold",
            )
            continue
        require(
            handoff_releases,
            f"pressure hold for {victim_request_id} has no proven progress or source retarget",
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
    completed_rows = [
        row for row in rows if str(row.get("phase", "")).endswith("request_completed")
    ]
    admitted_rows = [
        row
        for row in rows
        if row.get("phase") == "vnext.prefill_admission"
        and row.get("shape", {}).get("decision") == "admitted"
    ]
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
            if release["decision"] in {"owner_advanced", "source_retargeted"}
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
            f"decode progress owner {progress_owner_id} neither advanced nor retargeted its exact source",
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
        calibration_pool = common.quiescent_pool_snapshot(
            calibration_executor, "decode calibration"
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
        sizing_pool = common.quiescent_pool_snapshot(
            sizing_executor, "decode target sizing"
        )
        collection["target_sizing"] = {
            "clients": sizing_clients,
            "monitor": sizing_monitor,
            "pool_snapshot": sizing_pool,
            "health_final": "target-sizing/health.final.json",
            "trace": "target-sizing/scheduler-trace.jsonl",
        }
        target_budget_envelope = derive_target_budget_envelope(
            calibration_pool, sizing_pool
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
        target_pool = common.quiescent_pool_snapshot(target_executor, "decode target")
        target.stop()
        collection["target"] = {
            "clients": target_clients,
            "monitor": target_monitor,
            "pool_snapshot": target_pool,
            "health_final": "target/health.final.json",
            "trace": "target/scheduler-trace.jsonl",
        }
        require_target_pool_within_envelope(
            target_pool, target_budget_envelope, exact_budget
        )
        decode_summary = validate_decode_trace(
            common.read_trace(target.trace_path),
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
    calibration_health = common.read_json(root / str(calibration.get("health_final")))
    sizing_health = common.read_json(root / str(target_sizing.get("health_final")))
    target_health = common.read_json(root / str(target.get("health_final")))
    calibration_executor = common.find_executor_snapshot(calibration_health)
    sizing_executor = common.find_executor_snapshot(sizing_health)
    target_executor = common.find_executor_snapshot(target_health)
    require(
        calibration_executor is not None
        and sizing_executor is not None
        and target_executor is not None,
        "raw executor snapshots are missing",
    )
    for identity in ("model_id", "family_fingerprint", "program_fingerprint", "plan_hash"):
        require(
            calibration_executor.get(identity)
            == sizing_executor.get(identity)
            == target_executor.get(identity),
            f"calibration/sizing/target changed {identity}",
        )
    calibration_pool = common.quiescent_pool_snapshot(calibration_executor, "raw decode calibration")
    sizing_pool = common.quiescent_pool_snapshot(sizing_executor, "raw decode target sizing")
    target_pool = common.quiescent_pool_snapshot(target_executor, "raw decode target")
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
    target_budget_envelope = derive_target_budget_envelope(calibration_pool, sizing_pool)
    require(
        collection.get("target_budget_envelope") == target_budget_envelope,
        "target budget envelope differs from raw sizing receipts",
    )
    require(
        isinstance(exact_budget, int)
        and exact_budget == target_budget_envelope["budget_claimed_bytes"],
        "exact budget does not match the target-compatible sizing envelope",
    )
    require_target_pool_within_envelope(
        target_pool, target_budget_envelope, exact_budget
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
    target_started, target_finished, target_silence = validate_stream_group(
        root, "target", target.get("clients"), TARGET_MAX_TOKENS
    )
    for slot in ("A", "B", "C"):
        common.validate_replayed_workload(
            slot,
            {
                "calibration": calibration["clients"][slot],
                "target-sizing": target_sizing["clients"][slot],
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
    target_rows = common.read_trace(root / str(target.get("trace")))
    target_trace_path = root / str(target.get("trace"))
    target_trace_bytes = target_trace_path.stat().st_size
    require(target_trace_bytes <= STOP_POLICY["max_trace_bytes"], "target trace exceeds its byte ceiling")
    decode_summary = validate_decode_trace(
        target_rows,
        started_wall_ns=target_started,
        finished_wall_ns=target_finished,
    )
    require(target.get("decode_summary") == decode_summary, "decode summary differs from raw trace")

    counters = target_executor.get("counters")
    require(isinstance(counters, dict), "target executor counters are missing")
    counter_by_stage = {
        "sequence_extension": "extension_deferrals",
        "step_admission": "step_deferrals",
        "submission_wave": "wave_deferrals",
    }
    for stage in decode_summary["stages"]:
        counter = counter_by_stage[stage]
        require(counters.get(counter, 0) > 0, f"target counter {counter} did not record its trace stage")
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
        "target_trace_bytes": target_trace_bytes,
        "calibration_window_ns": [calibration_started, calibration_finished],
        "target_sizing_window_ns": [sizing_started, sizing_finished],
        "target_window_ns": [target_started, target_finished],
        "max_silence_seconds": {
            "calibration": calibration_silence,
            "target_sizing": sizing_silence,
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
        and CALIBRATION_MAX_TOKENS["C"] == TARGET_MAX_TOKENS["C"],
        "calibration must replace only B long-decode demand with a short request",
    )
    require(
        SERVER_POLICY["target_sizing_max_tokens"] == CALIBRATION_MAX_TOKENS,
        "target sizing must replay the narrow calibration workload",
    )

    storage_profile = {"allocator": "linear_arena", "view": "contiguous"}

    def pool_snapshot(pools: dict[str, int]) -> dict[str, Any]:
        resident_bytes = sum(pools.values())
        return {
            "static_bytes": 100,
            "resident_bytes": resident_bytes,
            "budget_claimed_bytes": 100 + resident_bytes,
            "pool_resident_bytes": pools,
            "pool_envelopes": {
                pool_id: {
                    "resident_bytes": value,
                    "resident_chunks": 1,
                    "largest_contiguous_bytes": value,
                    "storage_profile": storage_profile,
                }
                for pool_id, value in pools.items()
            },
        }

    calibration_pool = pool_snapshot({"sequence": 30, "workspace": 4})
    sizing_pool = pool_snapshot({"sequence": 20, "workspace": 7})
    target_envelope = derive_target_budget_envelope(calibration_pool, sizing_pool)
    require(
        target_envelope["budget_claimed_bytes"] == 137,
        "self-test lost target-compatible budget derivation",
    )
    require(
        target_envelope["pool_sources"]
        == {"sequence": "calibration", "workspace": "target_sizing"},
        "self-test lost per-pool sizing provenance",
    )
    require_target_pool_within_envelope(
        pool_snapshot({"sequence": 30, "workspace": 7}), target_envelope, 137
    )
    try:
        require_target_pool_within_envelope(
            pool_snapshot({"sequence": 31, "workspace": 7}), target_envelope, 137
        )
        raise AssertionError("oversized target pool unexpectedly fit its sizing envelope")
    except common.CapacityGateError:
        pass

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
        {
            "ts_unix_nanos": 150,
            "phase": "vnext.execution_capacity_pressure_hold_released",
            "status": "ok",
            "error": None,
            "request_id": "C",
            "shape": {
                "decision": "owner_advanced",
                "episode_id": 1,
                "transition_ordinal": 8,
                "waiting_ticket": 1,
                "progress_owner_id": "A",
                "progress_baseline": 53,
                "progress_current": 54,
                "admission_eligible": True,
                "probe_performed": False,
                "prefill_submit_observed": False,
            },
        },
        {
            "ts_unix_nanos": 151,
            "phase": "vnext.prefill_admission",
            "request_id": "C",
            "shape": {"decision": "admitted"},
        },
        {"ts_unix_nanos": 152, "phase": "vnext.request_completed", "request_id": "C"},
        {"ts_unix_nanos": 153, "phase": "vnext.request_completed", "request_id": "A"},
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
                planned_transition_ordinal=9,
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
                    "planned_transition_ordinal": 9,
                    "transition_ordinal": 10,
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
                    "release_transition_ordinal": 11,
                    "resumable_transition_ordinal": None,
                    "closed_transition_ordinal": 12,
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
    summary = validate_decode_trace(rows, started_wall_ns=90, finished_wall_ns=170)
    require(summary["split_events"] == 1, "self-test lost split evidence")
    require(summary["park_events"] == 2, "self-test lost park evidence")
    require(summary["resume_events"] == 1, "self-test lost resume evidence")
    require(summary["pressure_yield_events"] == 2, "self-test lost pressure-yield evidence")
    require(
        summary["pressure_yield_kinds"] == ["peer_handoff", "self_recompute"],
        "self-test lost typed yield strategies",
    )
    require(
        summary["pressure_fence_armed_events"] == 2
        and summary["pressure_fence_completed_events"] == 2,
        "self-test lost release-fence evidence",
    )
    require(
        summary["pressure_hold_events"] == 1
        and summary["pressure_hold_release_events"] == 1,
        "self-test lost pressure-hold evidence",
    )
    require(
        summary["pressure_victim_request_ids"] == ["C", "D"],
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
    retargeted = validate_pressure_fence_completed(
        retargeted_completion, "retargeted fence self-test"
    )
    require(
        retargeted["completion_transition_ordinal"] == 6
        and retargeted["completion_disposition"] == "source_retargeted",
        "self-test lost source-retargeted fence completion",
    )

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
    del missing_release[9]
    try:
        validate_decode_trace(missing_release, started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("pressure hold without a release unexpectedly passed")
    except common.CapacityGateError:
        pass
    unchanged_progress = json.loads(json.dumps(rows))
    unchanged_progress[9]["shape"]["progress_current"] = 53
    try:
        validate_decode_trace(unchanged_progress, started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("unchanged owner progress unexpectedly released a pressure hold")
    except common.CapacityGateError:
        pass
    stale_hold = json.loads(json.dumps(rows))
    stale_hold_event = json.loads(json.dumps(stale_hold[8]))
    stale_hold_event["ts_unix_nanos"] = 151
    stale_hold.append(stale_hold_event)
    try:
        validate_decode_trace(stale_hold, started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("victim remained held after owner progress")
    except common.CapacityGateError:
        pass
    premature_readmission = json.loads(json.dumps(rows))
    premature_readmission[10]["ts_unix_nanos"] = 149
    try:
        validate_decode_trace(premature_readmission, started_wall_ns=90, finished_wall_ns=170)
        raise AssertionError("victim re-admitted before owner progress")
    except common.CapacityGateError:
        pass
    source_retarget = json.loads(json.dumps(rows[9]))
    source_retarget["shape"].update(
        {
            "decision": "source_retargeted",
            "progress_current": 53,
            "previous_wait_condition": {
                "coordinator_id": 7,
                "observed": [
                    {"source": {"domain": 4}, "epoch": 109},
                    {"source": "plan_device_budget", "epoch": 1},
                ],
            },
            "current_wait_condition": {
                "coordinator_id": 7,
                "observed": [
                    {"source": {"domain": 2}, "epoch": 216},
                    {"source": "plan_device_budget", "epoch": 1},
                ],
            },
        }
    )
    validate_pressure_hold_release(source_retarget, "valid source retarget")
    unchanged_retarget = json.loads(json.dumps(source_retarget))
    unchanged_retarget["shape"]["current_wait_condition"] = json.loads(
        json.dumps(unchanged_retarget["shape"]["previous_wait_condition"])
    )
    try:
        validate_pressure_hold_release(unchanged_retarget, "unchanged source retarget")
        raise AssertionError("unchanged exact topology unexpectedly released a pressure hold")
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
