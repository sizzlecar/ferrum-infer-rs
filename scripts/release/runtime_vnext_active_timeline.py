#!/usr/bin/env python3
"""Derive formal Runtime vNext active-concurrency evidence from scheduler traces."""

from __future__ import annotations

import argparse
import bisect
import json
from typing import Any


SCHEMA_VERSION = 1
PASS_LINE = "FERRUM RUNTIME VNEXT ACTIVE TIMELINE SELFTEST PASS"
REQUEST_OPEN_PHASE = "engine_request_open"
REQUEST_CLOSE_PHASE = "engine_request_close"


class ActiveTimelineError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActiveTimelineError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return value


def _positive_int(value: Any, label: str) -> int:
    value = _nonnegative_int(value, label)
    _require(value > 0, f"{label} must be positive")
    return value


def _raw_event(row: Any, label: str) -> dict[str, Any]:
    wrapper = _object(row, label)
    raw = wrapper.get("raw", wrapper)
    return _object(raw, f"{label}.raw")


def _activity_point(event: dict[str, Any], index: int, typed_active_cap: int) -> dict[str, Any] | None:
    attributes = event.get("attributes")
    if not isinstance(attributes, dict):
        return None
    timestamp = attributes.get("monotonic_nanos")
    active = attributes.get("active_sequence_count")
    if timestamp is None and active is None:
        return None
    timestamp = _nonnegative_int(timestamp, f"trace[{index}].attributes.monotonic_nanos")
    active = _nonnegative_int(active, f"trace[{index}].attributes.active_sequence_count")
    _require(active <= typed_active_cap, f"trace[{index}] active count exceeds typed cap")
    snapshot = _object(attributes.get("scheduler_snapshot"), f"trace[{index}].attributes.scheduler_snapshot")
    snapshot_active = _nonnegative_int(snapshot.get("active_len"), f"trace[{index}].scheduler_snapshot.active_len")
    _require(snapshot_active == active, f"trace[{index}] active count differs from scheduler snapshot")
    phase = event.get("phase")
    _require(isinstance(phase, str) and phase, f"trace[{index}].phase is missing")
    request_id = event.get("request_id")
    _require(isinstance(request_id, str) and request_id, f"trace[{index}].request_id is missing")
    return {
        "monotonic_nanos": timestamp,
        "active_sequence_count": active,
        "phase": phase,
        "request_id": request_id,
        "event_id": event.get("event_id"),
        "input_ordinal": index,
    }


def _request_transitions(events: list[dict[str, Any]], expected_request_count: int) -> list[dict[str, Any]]:
    by_request: dict[str, dict[str, dict[str, Any]]] = {}
    transitions: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        phase = event.get("phase")
        if phase not in {REQUEST_OPEN_PHASE, REQUEST_CLOSE_PHASE}:
            continue
        attributes = _object(event.get("attributes"), f"trace[{index}].attributes")
        timestamp = _nonnegative_int(
            attributes.get("monotonic_nanos"),
            f"trace[{index}].attributes.monotonic_nanos",
        )
        active = _nonnegative_int(
            attributes.get("active_sequence_count"),
            f"trace[{index}].attributes.active_sequence_count",
        )
        snapshot = _object(attributes.get("scheduler_snapshot"), f"trace[{index}].scheduler_snapshot")
        _require(snapshot.get("active_len") == active, f"trace[{index}] transition active count mismatch")
        request_id = event.get("request_id")
        _require(isinstance(request_id, str) and request_id, f"trace[{index}].request_id is missing")
        kind = "open" if phase == REQUEST_OPEN_PHASE else "close"
        slots = by_request.setdefault(request_id, {})
        _require(kind not in slots, f"request {request_id} has duplicate {kind} transition")
        transition = {
            "monotonic_nanos": timestamp,
            "kind": kind,
            "request_id": request_id,
            "active_sequence_count": active,
            "input_ordinal": index,
        }
        slots[kind] = transition
        transitions.append(transition)
    _require(
        len(by_request) == expected_request_count,
        f"request transition count {len(by_request)} differs from expected {expected_request_count}",
    )
    for request_id, slots in by_request.items():
        _require(set(slots) == {"open", "close"}, f"request {request_id} lacks an open/close pair")
        _require(
            slots["close"]["monotonic_nanos"] > slots["open"]["monotonic_nanos"],
            f"request {request_id} close does not follow open",
        )
    transitions.sort(key=lambda item: (item["monotonic_nanos"], item["input_ordinal"]))
    return transitions


def _eligible_intervals(transitions: list[dict[str, Any]], active_floor: int) -> list[dict[str, int]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for transition in transitions:
        grouped.setdefault(transition["monotonic_nanos"], []).append(transition)
    timestamps = sorted(grouped)
    outstanding: set[str] = set()
    intervals: list[dict[str, int]] = []
    previous: int | None = None
    for timestamp in timestamps:
        if previous is not None and timestamp > previous and len(outstanding) >= active_floor:
            intervals.append(
                {
                    "start_monotonic_nanos": previous,
                    "end_monotonic_nanos": timestamp,
                    "outstanding_request_count": len(outstanding),
                }
            )
        for transition in grouped[timestamp]:
            request_id = transition["request_id"]
            if transition["kind"] == "open":
                _require(request_id not in outstanding, f"request {request_id} opened while already outstanding")
                outstanding.add(request_id)
            else:
                _require(request_id in outstanding, f"request {request_id} closed while not outstanding")
                outstanding.remove(request_id)
        previous = timestamp
    _require(not outstanding, "request transitions do not return to zero outstanding requests")
    merged: list[dict[str, int]] = []
    for interval in intervals:
        if (
            merged
            and merged[-1]["end_monotonic_nanos"] == interval["start_monotonic_nanos"]
            and merged[-1]["outstanding_request_count"] == interval["outstanding_request_count"]
        ):
            merged[-1]["end_monotonic_nanos"] = interval["end_monotonic_nanos"]
        else:
            merged.append(interval)
    _require(merged, "no eligible interval has enough outstanding requests")
    return merged


def _active_duration(
    intervals: list[dict[str, int]],
    activity: list[dict[str, Any]],
    active_floor: int,
) -> tuple[int, int]:
    timestamps = [point["monotonic_nanos"] for point in activity]
    eligible_ns = 0
    active_ns = 0
    for interval in intervals:
        start = interval["start_monotonic_nanos"]
        end = interval["end_monotonic_nanos"]
        eligible_ns += end - start
        point_index = bisect.bisect_right(timestamps, start) - 1
        _require(point_index >= 0, "active timeline does not cover eligible interval start")
        boundaries = [
            start,
            *[timestamp for timestamp in timestamps[point_index + 1 :] if start < timestamp < end],
            end,
        ]
        current_index = point_index
        for left, right in zip(boundaries, boundaries[1:]):
            while current_index + 1 < len(activity) and activity[current_index + 1]["monotonic_nanos"] <= left:
                current_index += 1
            if activity[current_index]["active_sequence_count"] >= active_floor:
                active_ns += right - left
    _require(eligible_ns > 0, "eligible interval duration must be positive")
    return eligible_ns, active_ns


def derive_active_timeline(
    trace_rows: list[Any],
    *,
    requested_concurrency: int,
    typed_active_cap: int,
    active_floor: int,
    expected_request_count: int,
) -> dict[str, Any]:
    """Validate and deterministically derive active-duty evidence."""

    requested_concurrency = _positive_int(requested_concurrency, "requested_concurrency")
    typed_active_cap = _positive_int(typed_active_cap, "typed_active_cap")
    active_floor = _positive_int(active_floor, "active_floor")
    expected_request_count = _positive_int(expected_request_count, "expected_request_count")
    _require(expected_request_count == requested_concurrency, "request count must equal client concurrency")
    _require(active_floor <= requested_concurrency, "active floor exceeds requested concurrency")
    _require(active_floor <= typed_active_cap, "typed active cap is below the required active floor")
    events = [_raw_event(row, f"trace[{index}]") for index, row in enumerate(trace_rows)]
    _require(events, "active timeline trace is empty")
    transitions = _request_transitions(events, expected_request_count)
    activity = [
        point
        for index, event in enumerate(events)
        if (point := _activity_point(event, index, typed_active_cap)) is not None
    ]
    _require(activity, "scheduler trace contains no active-count observations")
    activity.sort(key=lambda item: (item["monotonic_nanos"], item["input_ordinal"]))
    intervals = _eligible_intervals(transitions, active_floor)
    first_transition = transitions[0]["monotonic_nanos"]
    last_transition = transitions[-1]["monotonic_nanos"]
    request_ids = {transition["request_id"] for transition in transitions}
    foreign_activity = [
        point
        for point in activity
        if first_transition <= point["monotonic_nanos"] <= last_transition
        and point["request_id"] not in request_ids
    ]
    _require(
        not foreign_activity,
        "active timeline contains a foreign request inside the measured window",
    )
    covered_activity = [
        point
        for point in activity
        if first_transition <= point["monotonic_nanos"] <= last_transition
        and point["request_id"] in request_ids
    ]
    _require(covered_activity, "active timeline does not overlap request lifetime")
    eligible_ns, active_ns = _active_duration(intervals, covered_activity, active_floor)
    return {
        "schema_version": SCHEMA_VERSION,
        "requested_concurrency": requested_concurrency,
        "typed_active_cap": typed_active_cap,
        "active_floor": active_floor,
        "observed_max_active": max(point["active_sequence_count"] for point in covered_activity),
        "eligible_intervals": intervals,
        "eligible_duration_ns": eligible_ns,
        "active_at_or_above_floor_duration_ns": active_ns,
        "active_duty_cycle": active_ns / eligible_ns,
        "request_transition_coverage": {
            "expected": expected_request_count,
            "open": sum(item["kind"] == "open" for item in transitions),
            "close": sum(item["kind"] == "close" for item in transitions),
            "active_count": len(transitions),
        },
        "activity_observation_count": len(covered_activity),
        "activity_timeline": [
            {
                key: point[key]
                for key in ("monotonic_nanos", "active_sequence_count", "phase", "request_id", "event_id")
            }
            for point in covered_activity
        ],
    }


def _fixture_event(
    ordinal: int,
    timestamp: int,
    active: int,
    phase: str,
    request_id: str,
) -> dict[str, Any]:
    return {
        "event_id": f"evt-{ordinal}",
        "phase": phase,
        "request_id": request_id,
        "attributes": {
            "monotonic_nanos": timestamp,
            "active_sequence_count": active,
            "scheduler_snapshot": {"active_len": active},
        },
    }


def self_test() -> int:
    rows = [
        _fixture_event(1, 10, 0, REQUEST_OPEN_PHASE, "r1"),
        _fixture_event(2, 20, 1, "vnext.prefill_admission", "r1"),
        _fixture_event(3, 30, 1, REQUEST_OPEN_PHASE, "r2"),
        _fixture_event(4, 40, 2, "vnext.prefill_admission", "r2"),
        _fixture_event(5, 90, 1, REQUEST_CLOSE_PHASE, "r1"),
        _fixture_event(6, 100, 0, REQUEST_CLOSE_PHASE, "r2"),
    ]
    summary = derive_active_timeline(
        rows,
        requested_concurrency=2,
        typed_active_cap=2,
        active_floor=2,
        expected_request_count=2,
    )
    _require(summary["eligible_duration_ns"] == 60, "self-test eligible duration mismatch")
    _require(summary["active_at_or_above_floor_duration_ns"] == 50, "self-test active duration mismatch")
    _require(abs(summary["active_duty_cycle"] - (5 / 6)) < 1e-12, "self-test duty mismatch")
    hostile = json.loads(json.dumps(rows))
    hostile[-1]["attributes"].pop("active_sequence_count")
    try:
        derive_active_timeline(
            hostile,
            requested_concurrency=2,
            typed_active_cap=2,
            active_floor=2,
            expected_request_count=2,
        )
    except ActiveTimelineError as exc:
        _require("active_sequence_count" in str(exc), "self-test rejected hostile trace for wrong reason")
    else:
        raise ActiveTimelineError("self-test accepted a close transition without active count")
    hostile = json.loads(json.dumps(rows))
    hostile.insert(4, _fixture_event(7, 50, 2, "vnext.prefill_admission", "foreign"))
    try:
        derive_active_timeline(
            hostile,
            requested_concurrency=2,
            typed_active_cap=2,
            active_floor=2,
            expected_request_count=2,
        )
    except ActiveTimelineError as exc:
        _require("foreign request" in str(exc), "self-test rejected foreign activity for wrong reason")
    else:
        raise ActiveTimelineError("self-test accepted foreign activity inside the measured window")
    print(PASS_LINE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    parser.error("--self-test is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
