#!/usr/bin/env python3
"""Validate Ferrum observability profile and native-op manifest fixtures."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_FIXTURES = REPO_ROOT / "scripts/release/fixtures/observability_profile"
DEFAULT_NATIVE_FIXTURES = REPO_ROOT / "scripts/release/fixtures/native_operator"
PASS_LINE = "FERRUM PROFILE ANALYZER PASS"
SELFTEST_PASS_LINE = "FERRUM PROFILE ANALYZER SELFTEST PASS"
ENGINE_TIMING_PROFILE_DETAILS = {"latency", "kernel", "replay", "verify", "full"}
ENGINE_TIMING_ATTRIBUTE_KEYS = {
    "engine_token_clock_source",
    "engine_token_wall_anchor_unix_nanos",
    "clock_conversion_max_error_nanos",
    "engine_token_commit_nanos_since_request_start",
    "engine_token_commit_count",
    "itl_source",
    "itl_interval_count",
    "itl_nanos",
    "engine_decode_ready_nanos_since_request_start",
    "engine_decode_wall_nanos",
    "clock_conversion_error_ppm",
    "decode_wall_timing_eligible",
    "decode_wall_timing_unavailable_reason",
}


class ValidationError(RuntimeError):
    pass


def is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def require_non_empty_string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context}.{key} must be a non-empty string")
    return value


def validate_resource(resource: Any, context: str) -> None:
    if not isinstance(resource, dict):
        raise ValidationError(f"{context}.resource must be an object")
    require_non_empty_string(resource, "owner_kind", f"{context}.resource")
    require_non_empty_string(resource, "owner_id", f"{context}.resource")
    require_non_empty_string(resource, "resource_kind", f"{context}.resource")
    action = require_non_empty_string(resource, "action", f"{context}.resource")
    if action in {"reserve", "commit", "release", "rollback"}:
        for key in ("amount", "before", "after"):
            if key not in resource or not isinstance(resource[key], int):
                raise ValidationError(f"{context}.resource.{key} must be an integer")
    if action in {"defer", "reject"}:
        require_non_empty_string(resource, "reason", f"{context}.resource")
    if action == "capacity_snapshot" and not isinstance(resource.get("capacity"), int):
        raise ValidationError(f"{context}.resource.capacity must be an integer")


def validate_memory(memory: Any, context: str) -> None:
    if not isinstance(memory, dict):
        raise ValidationError(f"{context}.memory must be an object")
    require_non_empty_string(memory, "scope", f"{context}.memory")
    for key in ("before_bytes", "after_bytes", "current_bytes", "high_water_bytes"):
        if key not in memory or not isinstance(memory[key], int) or memory[key] < 0:
            raise ValidationError(f"{context}.memory.{key} must be a non-negative integer")
    available = memory.get("available_bytes")
    if available is not None and (not isinstance(available, int) or available < 0):
        raise ValidationError(f"{context}.memory.available_bytes must be a non-negative integer")


def validate_profile_event(event: Any, context: str) -> None:
    if not isinstance(event, dict):
        raise ValidationError(f"{context} must be a JSON object")
    allowed_keys = {
        "schema_version",
        "ts_unix_nanos",
        "event_id",
        "request_id",
        "correlation_id",
        "entrypoint",
        "backend",
        "runtime_preset_hash",
        "phase",
        "event_kind",
        "timestamp",
        "status",
        "model",
        "duration_us",
        "memory",
        "resource",
        "error",
        "replay",
        "shape",
        "backend_detail",
        "attributes",
    }
    unknown = set(event) - allowed_keys
    if unknown:
        raise ValidationError(f"{context} has unknown top-level fields: {sorted(unknown)}")
    if event.get("schema_version") != 1:
        raise ValidationError(f"{context}.schema_version must be 1")
    if not isinstance(event.get("ts_unix_nanos"), int) or event["ts_unix_nanos"] <= 0:
        raise ValidationError(f"{context}.ts_unix_nanos must be a positive integer")
    require_non_empty_string(event, "event_id", context)
    require_non_empty_string(event, "request_id", context)
    require_non_empty_string(event, "correlation_id", context)
    require_non_empty_string(event, "backend", context)
    require_non_empty_string(event, "runtime_preset_hash", context)
    require_non_empty_string(event, "phase", context)
    require_non_empty_string(event, "timestamp", context)
    shape = event.get("shape")
    if not isinstance(shape, dict) or not shape:
        raise ValidationError(f"{context}.shape must be a non-empty object")
    backend_detail = event.get("backend_detail")
    if backend_detail is not None and not isinstance(backend_detail, dict):
        raise ValidationError(f"{context}.backend_detail must be an object when set")
    event_kind = require_non_empty_string(event, "event_kind", context)
    status = require_non_empty_string(event, "status", context)
    if require_non_empty_string(event, "entrypoint", context) not in {
        "run",
        "serve",
        "bench_serve",
        "synthetic",
    }:
        raise ValidationError(f"{context}.entrypoint is invalid")
    if event_kind == "timed_span" and not isinstance(event.get("duration_us"), int):
        raise ValidationError(f"{context}.duration_us is required for timed_span")
    if "memory" in event:
        validate_memory(event["memory"], context)
    if "resource" in event:
        validate_resource(event["resource"], context)
    if status == "failure":
        error = event.get("error")
        if not isinstance(error, dict):
            raise ValidationError(f"{context}.error is required for failure status")
        require_non_empty_string(error, "kind", f"{context}.error")
        require_non_empty_string(error, "message", f"{context}.error")
        blocking = error.get("blocking")
        if not isinstance(blocking, bool):
            raise ValidationError(f"{context}.error.blocking must be a boolean")
    if "replay" in event:
        replay = event["replay"]
        if not isinstance(replay, dict):
            raise ValidationError(f"{context}.replay must be an object")
        require_non_empty_string(replay, "command", f"{context}.replay")


def event_attributes(event: dict[str, Any]) -> dict[str, Any]:
    attributes = event.get("attributes", {})
    if attributes is None:
        return {}
    if not isinstance(attributes, dict):
        raise ValidationError(f"{event.get('event_id', '<unknown>')}.attributes must be an object")
    return attributes


def error_kind(event: dict[str, Any]) -> str:
    error = event.get("error")
    if not isinstance(error, dict):
        return ""
    kind = error.get("kind")
    return kind if isinstance(kind, str) else ""


def is_non_negative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def event_order_key(event: dict[str, Any]) -> tuple[int, int, int, int, str]:
    attributes = event_attributes(event)
    monotonic = attributes.get("monotonic_nanos_since_run_start")
    sequence = attributes.get("execution_sequence")
    if not is_non_negative_int(sequence):
        shape = event.get("shape")
        sequence = shape.get("execution_sequence") if isinstance(shape, dict) else None
    sequence_value = sequence if is_non_negative_int(sequence) else (1 << 63) - 1
    timestamp = event.get("ts_unix_nanos")
    timestamp_value = timestamp if is_non_negative_int(timestamp) else (1 << 63) - 1
    event_id = str(event.get("event_id", ""))
    if is_non_negative_int(monotonic):
        return (0, monotonic, sequence_value, timestamp_value, event_id)
    if is_non_negative_int(sequence):
        return (1, sequence, sequence_value, timestamp_value, event_id)
    return (2, timestamp_value, sequence_value, timestamp_value, event_id)


def event_strictly_after(
    later: dict[str, Any], earlier: dict[str, Any]
) -> bool:
    later_attributes = event_attributes(later)
    earlier_attributes = event_attributes(earlier)
    later_run_id = later_attributes.get("run_id")
    earlier_run_id = earlier_attributes.get("run_id")
    later_monotonic = later_attributes.get("monotonic_nanos_since_run_start")
    earlier_monotonic = earlier_attributes.get("monotonic_nanos_since_run_start")
    if (
        isinstance(later_run_id, str)
        and later_run_id
        and later_run_id == earlier_run_id
        and is_non_negative_int(later_monotonic)
        and is_non_negative_int(earlier_monotonic)
    ):
        return later_monotonic > earlier_monotonic

    later_timestamp = later.get("ts_unix_nanos")
    earlier_timestamp = earlier.get("ts_unix_nanos")
    if is_non_negative_int(later_timestamp) and is_non_negative_int(earlier_timestamp):
        return later_timestamp > earlier_timestamp
    return False


def deduplicate_profile_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed: dict[tuple[str, str, str, str], str] = {}
    unique: list[dict[str, Any]] = []
    for event in events:
        key = (
            str(event.get("event_id", "")),
            str(event.get("request_id", "")),
            str(event.get("entrypoint", "")),
            str(event.get("phase", "")),
        )
        payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
        previous = observed.get(key)
        if previous is not None:
            if previous != payload:
                raise ValidationError(
                    "profile aliases contain conflicting payloads for "
                    f"event identity {key}"
                )
            continue
        observed[key] = payload
        unique.append(event)
    return unique


def logical_request_groups(
    events: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    unique = deduplicate_profile_events(events)
    candidates: dict[str, set[str]] = {}
    for event in unique:
        entrypoint = str(event.get("entrypoint", ""))
        attributes = event_attributes(event)
        request_id = str(event.get("request_id", ""))
        execution_id = attributes.get("execution_request_id")
        if isinstance(execution_id, str) and execution_id.strip():
            execution_id = execution_id.strip()
            if request_id != execution_id and not execution_id.endswith(f".{request_id}"):
                raise ValidationError(
                    f"{event.get('event_id', '<unknown>')} execution_request_id "
                    f"{execution_id!r} does not map request_id {request_id!r}"
                )
            candidates.setdefault(entrypoint, set()).add(execution_id)
        elif request_id.startswith("request."):
            candidates.setdefault(entrypoint, set()).add(request_id)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in unique:
        entrypoint = str(event.get("entrypoint", ""))
        attributes = event_attributes(event)
        request_id = str(event.get("request_id", ""))
        explicit = attributes.get("execution_request_id")
        if isinstance(explicit, str) and explicit.strip():
            logical_id = explicit.strip()
        elif request_id.startswith("request."):
            logical_id = request_id
        else:
            matches = sorted(
                candidate
                for candidate in candidates.get(entrypoint, set())
                if candidate.endswith(f".{request_id}")
            )
            if len(matches) > 1:
                raise ValidationError(
                    f"{event.get('event_id', '<unknown>')} request_id {request_id!r} "
                    f"has ambiguous execution identities {matches}"
                )
            if matches:
                logical_id = matches[0]
            elif entrypoint in {"run", "serve"} and event.get("backend") == "actual":
                logical_id = f"request.product.{request_id}"
            else:
                logical_id = request_id
        groups.setdefault((entrypoint, logical_id), []).append(event)

    for key, group in groups.items():
        run_ids = {
            event_attributes(event).get("run_id")
            for event in group
            if isinstance(event_attributes(event).get("run_id"), str)
            and event_attributes(event)["run_id"].strip()
        }
        if len(run_ids) > 1:
            raise ValidationError(f"logical request {key} mixes run_id values {sorted(run_ids)}")
    return groups


def validate_engine_token_timing_attributes(
    attributes: dict[str, Any], context: str
) -> None:
    if attributes.get("engine_token_clock_source") != "rust_std_instant":
        raise ValidationError(
            f"{context} attributes.engine_token_clock_source must be rust_std_instant"
        )
    wall_anchor = attributes.get("engine_token_wall_anchor_unix_nanos")
    if type(wall_anchor) is not int or wall_anchor <= 0:
        raise ValidationError(
            f"{context} attributes.engine_token_wall_anchor_unix_nanos must be positive"
        )
    max_error = attributes.get("clock_conversion_max_error_nanos")
    if not is_non_negative_int(max_error):
        raise ValidationError(
            f"{context} attributes.clock_conversion_max_error_nanos must be non-negative"
        )
    output_tokens = attributes.get("output_token_count")
    completion_tokens = attributes.get("completion_token_count")
    if not is_non_negative_int(output_tokens):
        raise ValidationError(f"{context} attributes.output_token_count must be non-negative")
    if completion_tokens != output_tokens:
        raise ValidationError(
            f"{context} completion_token_count={completion_tokens!r} does not match "
            f"output_token_count={output_tokens}"
        )
    commits = attributes.get("engine_token_commit_nanos_since_request_start")
    if not isinstance(commits, list) or not all(is_non_negative_int(value) for value in commits):
        raise ValidationError(f"{context} engine token commits must be non-negative integers")
    if any(right < left for left, right in zip(commits, commits[1:])):
        raise ValidationError(f"{context} engine token commits must be monotonic")
    commit_count = attributes.get("engine_token_commit_count")
    if commit_count != len(commits) or commit_count != output_tokens:
        raise ValidationError(
            f"{context} engine token commit count must equal raw commits and output tokens"
        )

    intervals = [right - left for left, right in zip(commits, commits[1:])]
    if attributes.get("itl_source") != "engine_token_commit":
        raise ValidationError(f"{context} attributes.itl_source must be engine_token_commit")
    if attributes.get("itl_nanos") != intervals:
        raise ValidationError(f"{context} attributes.itl_nanos does not match commit deltas")
    if attributes.get("itl_interval_count") != max(output_tokens - 1, 0):
        raise ValidationError(f"{context} attributes.itl_interval_count is inconsistent")
    expected_itl_us = (sum(intervals) // len(intervals) // 1_000) if intervals else 0
    if attributes.get("itl_us_avg") != expected_itl_us:
        raise ValidationError(f"{context} attributes.itl_us_avg is not commit-derived")
    if commits:
        if attributes.get("ttft_us") != commits[0] // 1_000:
            raise ValidationError(f"{context} attributes.ttft_us is not commit-derived")
    elif "ttft_us" in attributes:
        raise ValidationError(f"{context} zero-token timing must not contain ttft_us")

    decode_ready = attributes.get("engine_decode_ready_nanos_since_request_start")
    expected_reason: str | None = None
    decode_wall: int | None = None
    if not commits:
        expected_reason = "no_token_commits"
    elif decode_ready is None:
        expected_reason = "decode_not_entered"
    elif not is_non_negative_int(decode_ready):
        raise ValidationError(f"{context} engine decode-ready timestamp must be non-negative")
    elif commits[-1] <= decode_ready:
        expected_reason = "no_positive_decode_commit_interval"
    else:
        decode_wall = commits[-1] - decode_ready

    eligible = attributes.get("decode_wall_timing_eligible")
    if decode_wall is None:
        if eligible is not False:
            raise ValidationError(f"{context} decode wall timing must be marked ineligible")
        if attributes.get("decode_wall_timing_unavailable_reason") != expected_reason:
            raise ValidationError(f"{context} decode wall unavailable reason is inconsistent")
        for key in ("engine_decode_wall_nanos", "clock_conversion_error_ppm"):
            if key in attributes:
                raise ValidationError(f"{context} ineligible decode timing contains attributes.{key}")
        return

    if eligible is not True:
        raise ValidationError(f"{context} decode wall timing must be marked eligible")
    if attributes.get("engine_decode_wall_nanos") != decode_wall:
        raise ValidationError(f"{context} attributes.engine_decode_wall_nanos is inconsistent")
    expected_ppm = max_error * 1_000_000 // decode_wall
    if attributes.get("clock_conversion_error_ppm") != expected_ppm:
        raise ValidationError(f"{context} clock conversion ppm is inconsistent")
    if max_error * 200 > decode_wall:
        raise ValidationError(f"{context} clock conversion error exceeds 0.5% decode wall")
    if "decode_wall_timing_unavailable_reason" in attributes:
        raise ValidationError(f"{context} eligible decode timing contains an unavailable reason")


def has_engine_token_timing_attributes(attributes: dict[str, Any]) -> bool:
    return any(key in attributes for key in ENGINE_TIMING_ATTRIBUTE_KEYS)


def validate_failure_semantics(events: list[dict[str, Any]], context: str) -> None:
    for logical_key, group in logical_request_groups(events).items():
        failures = [event for event in group if event.get("status") == "failure"]
        if not failures:
            continue
        first_failures: list[dict[str, Any]] = []
        terminal_failures: list[dict[str, Any]] = []
        for event in failures:
            event_context = f"{context}:{event.get('event_id', '<unknown>')}"
            attributes = event_attributes(event)
            error = event.get("error") if isinstance(event.get("error"), dict) else {}
            blocking = error.get("blocking")
            first = attributes.get("first_failure_event") is True
            terminal = attributes.get("terminal_failure_event") is True
            if blocking is True:
                if not first:
                    raise ValidationError(
                        f"{event_context} blocking failure requires first_failure_event=true"
                    )
                if terminal:
                    raise ValidationError(
                        f"{event_context} blocking failure cannot be terminal_failure_event"
                    )
                first_failures.append(event)
            else:
                if first:
                    raise ValidationError(
                        f"{event_context} nonblocking failure cannot be first_failure_event"
                    )
                if terminal:
                    terminal_failures.append(event)

        if len(first_failures) != 1:
            raise ValidationError(
                f"{context} logical request {logical_key} requires exactly one first failure; "
                f"observed {len(first_failures)}"
            )
        first = first_failures[0]
        snapshots = [event for event in group if "memory" in event or "resource" in event]
        if not snapshots:
            raise ValidationError(
                f"{context}:{first.get('event_id')} first failure requires a same-request "
                "memory or resource snapshot"
            )
        first_fingerprint = event_attributes(first).get("first_failure_fingerprint")
        for terminal in terminal_failures:
            terminal_fingerprint = event_attributes(terminal).get("first_failure_fingerprint")
            if (
                terminal_fingerprint is not None
                and terminal_fingerprint != first_fingerprint
            ):
                raise ValidationError(
                    f"{context}:{terminal.get('event_id')} terminal failure fingerprint "
                    "does not match first failure"
                )
            if not event_strictly_after(terminal, first):
                raise ValidationError(
                    f"{context}:{terminal.get('event_id')} terminal failure precedes first failure"
                )

        if any(error_kind(event) in {"cuda_oom", "metal_oom", "oom", "silent_oom"} for event in failures):
            if not any(has_capacity_explanation(event) for event in group):
                raise ValidationError(
                    f"{context} logical request {logical_key} OOM requires capacity evidence"
                )


def has_capacity_explanation(event: dict[str, Any]) -> bool:
    resource = event.get("resource")
    if not isinstance(resource, dict):
        return False
    action = resource.get("action")
    if action not in {"defer", "reject", "capacity_snapshot"}:
        return False
    if action in {"defer", "reject"}:
        return isinstance(resource.get("capacity"), int) and bool(str(resource.get("reason", "")).strip())
    return isinstance(resource.get("capacity"), int)


def validate_chat_completion_profile_event(event_attrs: dict[str, Any], context: str) -> None:
    validate_request_token_latency_summary(
        event_attrs,
        context,
        allowed_token_sources={"usage", "generated_tokens"},
    )
    if event_attrs.get("stream") is True:
        for key in ("ttft_us", "itl_us_avg"):
            value = event_attrs.get(key)
            if not isinstance(value, int) or value < 0:
                raise ValidationError(
                    f"{context} streaming chat completion requires attributes.{key}"
                )


def validate_run_generation_profile_event(event_attrs: dict[str, Any], context: str) -> None:
    validate_request_token_latency_summary(
        event_attrs,
        context,
        allowed_token_sources={"rendered_prompt_and_generated_tokens"},
    )


def validate_request_token_latency_summary(
    event_attrs: dict[str, Any],
    context: str,
    *,
    allowed_token_sources: set[str],
) -> None:
    for key in (
        "completion_token_count",
        "e2e_duration_us",
        "output_token_count",
        "prompt_token_count",
        "total_token_count",
    ):
        value = event_attrs.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValidationError(f"{context} requires non-negative integer attributes.{key}")
    if event_attrs["e2e_duration_us"] <= 0:
        raise ValidationError(f"{context} attributes.e2e_duration_us must be positive")
    if event_attrs["total_token_count"] < event_attrs["completion_token_count"]:
        raise ValidationError(
            f"{context} attributes.total_token_count must cover completion_token_count"
        )
    token_count_source = event_attrs.get("token_count_source")
    if token_count_source not in allowed_token_sources:
        allowed = ", ".join(sorted(allowed_token_sources))
        raise ValidationError(
            f"{context} attributes.token_count_source must be one of: {allowed}"
        )
    if event_attrs.get("profile_detail") in ENGINE_TIMING_PROFILE_DETAILS:
        validate_engine_token_timing_attributes(event_attrs, context)


def validate_profile_semantics(path: Path, events: list[dict[str, Any]]) -> None:
    schema_fingerprints = {
        event_attributes(event).get("profile_schema_fingerprint")
        for event in events
        if event_attributes(event).get("profile_schema_fingerprint") is not None
    }
    if len(schema_fingerprints) > 1:
        raise ValidationError(f"{path} mixes profile schema fingerprints: {sorted(schema_fingerprints)}")
    has_latency_sample = any(
        event.get("event_kind") == "timed_span" and isinstance(event.get("duration_us"), int)
        for event in events
    )
    if not has_latency_sample:
        raise ValidationError(f"{path} requires at least one duration_us latency sample")

    for event in events:
        attrs = event_attributes(event)
        context = f"{path}:{event.get('event_id', '<unknown>')}"
        resource = event.get("resource")
        if isinstance(resource, dict) and resource.get("action") in {"defer", "reject"}:
            if not has_capacity_explanation(event):
                raise ValidationError(f"{context} defer/reject requires capacity and reason")
        if isinstance(attrs.get("resource_leak_count"), int) and attrs["resource_leak_count"] > 0:
            raise ValidationError(f"{context} reports resource_leak_count={attrs['resource_leak_count']}")
        profile_detail = attrs.get("profile_detail")
        diagnostic_only = attrs.get("diagnostic_only")
        if has_engine_token_timing_attributes(attrs):
            validate_engine_token_timing_attributes(attrs, context)
        if attrs.get("performance_claim") is True and (
            profile_detail in {"kernel", "debug", "replay", "verify", "full"}
            or diagnostic_only is True
        ):
            raise ValidationError(f"{context} uses diagnostic profile as performance claim")
        profile_count = attrs.get("profile_completed_requests")
        prometheus_count = attrs.get("prometheus_completed_requests")
        if profile_count is not None or prometheus_count is not None:
            if profile_count != prometheus_count:
                raise ValidationError(
                    f"{context} profile/prometheus completed request mismatch: "
                    f"{profile_count} != {prometheus_count}"
                )
        if (
            event.get("entrypoint") == "serve"
            and event.get("status") == "ok"
            and str(event.get("phase", "")).startswith("chat_completions_")
            and str(event.get("phase", "")).endswith("_complete")
        ):
            validate_chat_completion_profile_event(attrs, context)
        if (
            event.get("entrypoint") == "run"
            and event.get("status") == "ok"
            and event.get("phase") == "actual_run_generation"
        ):
            validate_run_generation_profile_event(attrs, context)

    for event in events:
        if event.get("status") != "failure":
            continue
        kind = error_kind(event)
        context = f"{path}:{event.get('event_id', '<unknown>')}"
        if kind in {"bad_output", "bad_text", "missing_done", "duplicate_done", "malformed_sse"}:
            if "replay" not in event:
                raise ValidationError(f"{context} correctness failure requires replay command")
    validate_failure_semantics(events, str(path))


def validate_profile_jsonl(path: Path) -> dict[str, Any]:
    events = 0
    payloads: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_no} invalid JSON: {exc}") from exc
            validate_profile_event(payload, f"{path}:{line_no}")
            events += 1
            payloads.append(payload)
    if events == 0:
        raise ValidationError(f"{path} must contain at least one profile event")
    validate_profile_semantics(path, payloads)
    return {"path": str(path), "events": events}


def validate_native_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValidationError(f"{path} must be a JSON object")
    if data.get("schema_version") != 1:
        raise ValidationError(f"{path}.schema_version must be 1")
    for key in ("operator", "operator_abi_version", "ferrum_native_abi_version", "backend"):
        require_non_empty_string(data, key, str(path))
    backend = data["backend"]
    if backend not in {"cuda", "metal", "cpu"}:
        raise ValidationError(f"{path}.backend is invalid")
    for key in ("inputs_sha256", "binary_sha256"):
        if not is_sha256_digest(data.get(key)):
            raise ValidationError(f"{path}.{key} must be a sha256 digest")
    source = data.get("source_package")
    if not isinstance(source, dict):
        raise ValidationError(f"{path}.source_package must be an object")
    for key in ("kind", "revision"):
        require_non_empty_string(source, key, f"{path}.source_package")
    if not is_sha256_digest(source.get("sha256")):
        raise ValidationError(f"{path}.source_package.sha256 must be a sha256 digest")
    if backend == "cuda":
        caps = data.get("compute_capabilities")
        if not isinstance(caps, list) or not caps:
            raise ValidationError(f"{path}.compute_capabilities must be a non-empty list")
        if not all(isinstance(cap, str) and cap.startswith("sm_") for cap in caps):
            raise ValidationError(f"{path}.compute_capabilities entries must use sm_xx")
    exports = data.get("exports")
    if not isinstance(exports, list) or "ferrum_native_op_init" not in exports:
        raise ValidationError(f"{path}.exports must include ferrum_native_op_init")
    if "ferrum_native_op_descriptor" not in exports:
        raise ValidationError(f"{path}.exports must include ferrum_native_op_descriptor")
    return {"path": str(path), "operator": data["operator"], "backend": backend}


def expect_pass(path: Path, validator) -> dict[str, Any]:
    return validator(path)


def expect_fail(path: Path, validator) -> dict[str, Any]:
    try:
        validator(path)
    except ValidationError as exc:
        return {"path": str(path), "error": str(exc)}
    raise ValidationError(f"{path} unexpectedly passed")


def fixture_files(root: Path, suffix: str) -> tuple[list[Path], list[Path]]:
    pass_files = sorted((root / "pass").glob(f"*{suffix}"))
    fail_files = sorted((root / "fail").glob(f"*{suffix}"))
    if not pass_files:
        raise ValidationError(f"{root / 'pass'} has no *{suffix} fixtures")
    if not fail_files:
        raise ValidationError(f"{root / 'fail'} has no *{suffix} fixtures")
    return pass_files, fail_files


def run_semantic_contract_selftest(profile_root: Path) -> dict[str, Any]:
    base_path = profile_root / "pass/basic_profile.jsonl"
    base_events = [
        json.loads(line)
        for line in base_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    timed = next(event for event in base_events if event.get("event_kind") == "timed_span")
    execution_id = "request.product.semantic-selftest"

    snapshot = copy.deepcopy(timed)
    snapshot.update(
        {
            "event_id": "evt-semantic-snapshot",
            "request_id": "semantic-selftest",
            "phase": "vnext.capacity_snapshot",
            "ts_unix_nanos": 30,
        }
    )
    snapshot["attributes"] = {
        "execution_request_id": execution_id,
        "monotonic_nanos_since_run_start": 10,
        "run_id": "run.semantic-selftest",
    }

    first = copy.deepcopy(timed)
    first.pop("memory", None)
    first.update(
        {
            "event_id": "evt-semantic-first",
            "request_id": execution_id,
            "phase": "vnext.failure_observed",
            "event_kind": "error",
            "status": "failure",
            "error": {"blocking": True, "kind": "resource_exhausted", "message": "blocked"},
            "ts_unix_nanos": 40,
        }
    )
    first["attributes"] = {
        "execution_request_id": execution_id,
        "first_failure_event": True,
        "first_failure_fingerprint": "failure.semantic",
        "monotonic_nanos_since_run_start": 20,
        "run_id": "run.semantic-selftest",
    }

    terminal = copy.deepcopy(first)
    terminal.update(
        {
            "event_id": "evt-semantic-terminal",
            "request_id": "semantic-selftest",
            "phase": "actual_run_generation_failed",
            "error": {"blocking": False, "kind": "request_failed", "message": "terminated"},
            "ts_unix_nanos": 50,
        }
    )
    terminal["attributes"] = {
        "execution_request_id": execution_id,
        "terminal_failure_event": True,
    }
    validate_profile_semantics(
        Path("semantic-first-failure-pass"),
        [terminal, snapshot, first],
    )

    duplicate = copy.deepcopy(first)
    duplicate["event_id"] = "evt-semantic-first-duplicate"
    duplicate["attributes"]["monotonic_nanos_since_run_start"] = 21
    try:
        validate_failure_semantics(
            [snapshot, first, duplicate, terminal],
            "semantic-duplicate-first",
        )
    except ValidationError:
        duplicate_rejected = True
    else:
        raise ValidationError("duplicate first-failure semantic self-test unexpectedly passed")

    reversed_terminal = copy.deepcopy(terminal)
    reversed_terminal["event_id"] = "evt-semantic-terminal-reversed"
    reversed_terminal["ts_unix_nanos"] = 35
    try:
        validate_failure_semantics(
            [snapshot, first, reversed_terminal],
            "semantic-terminal-order",
        )
    except ValidationError:
        reversed_terminal_rejected = True
    else:
        raise ValidationError("reversed terminal semantic self-test unexpectedly passed")

    timing = {
        "profile_detail": "latency",
        "output_token_count": 3,
        "completion_token_count": 3,
        "engine_token_clock_source": "rust_std_instant",
        "engine_token_wall_anchor_unix_nanos": 1_700_000_000_000_000_000,
        "clock_conversion_max_error_nanos": 400,
        "engine_token_commit_nanos_since_request_start": [1_000_000, 2_500_000, 5_000_000],
        "engine_token_commit_count": 3,
        "itl_source": "engine_token_commit",
        "itl_nanos": [1_500_000, 2_500_000],
        "itl_interval_count": 2,
        "itl_us_avg": 2_000,
        "ttft_us": 1_000,
        "engine_decode_ready_nanos_since_request_start": 2_000_000,
        "engine_decode_wall_nanos": 3_000_000,
        "clock_conversion_error_ppm": 133,
        "decode_wall_timing_eligible": True,
    }
    validate_engine_token_timing_attributes(timing, "semantic-valid-timing")
    forged = copy.deepcopy(timing)
    forged["itl_us_avg"] = 2_001
    try:
        validate_engine_token_timing_attributes(forged, "semantic-forged-timing")
    except ValidationError:
        forged_timing_rejected = True
    else:
        raise ValidationError("forged engine timing semantic self-test unexpectedly passed")

    return {
        "out_of_order_terminal_join": "pass",
        "separate_snapshot_join": "pass",
        "duplicate_first_rejected": duplicate_rejected,
        "reversed_terminal_rejected": reversed_terminal_rejected,
        "engine_timing_recomputed": "pass",
        "forged_timing_rejected": forged_timing_rejected,
    }


def run_fixture_selftest(profile_root: Path, native_root: Path) -> dict[str, Any]:
    profile_pass, profile_fail = fixture_files(profile_root, ".jsonl")
    native_pass, native_fail = fixture_files(native_root, ".json")
    return {
        "profile_pass": [expect_pass(path, validate_profile_jsonl) for path in profile_pass],
        "profile_fail": [expect_fail(path, validate_profile_jsonl) for path in profile_fail],
        "native_pass": [expect_pass(path, validate_native_manifest) for path in native_pass],
        "native_fail": [expect_fail(path, validate_native_manifest) for path in native_fail],
        "semantic_contracts": run_semantic_contract_selftest(profile_root),
    }


def write_summary(out: Path | None, summary: dict[str, Any]) -> None:
    if out is None:
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "ferrum_profile_analyzer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--native-manifest", action="append", type=Path, default=[])
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_PROFILE_FIXTURES)
    parser.add_argument("--native-fixtures", type=Path, default=DEFAULT_NATIVE_FIXTURES)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        summary = run_fixture_selftest(args.fixtures, args.native_fixtures)
        write_summary(args.out, summary)
        print(SELFTEST_PASS_LINE)
        return 0

    summary: dict[str, Any] = {"profiles": [], "native_manifests": []}
    for path in args.profile_jsonl:
        summary["profiles"].append(validate_profile_jsonl(path))
    for path in args.native_manifest:
        summary["native_manifests"].append(validate_native_manifest(path))
    if not summary["profiles"] and not summary["native_manifests"]:
        raise ValidationError("provide --profile-jsonl, --native-manifest, or --self-test")
    write_summary(args.out, summary)
    suffix = f": {args.out}" if args.out else ""
    print(f"{PASS_LINE}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
