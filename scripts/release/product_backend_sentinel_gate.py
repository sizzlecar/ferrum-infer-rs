#!/usr/bin/env python3
"""WP11 product/backend sentinel gate.

This gate consumes the artifact formats produced by the profile, resource, replay,
and native-operator gates. It intentionally does not add another benchmark client.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from native_operator_artifact_gate import (
    GateError as NativeGateError,
    ResolverRequirement,
    read_json as read_native_json,
    resolve_manifest,
)
from observability_profile_gate import (
    GateError as ProfileGateError,
    load_and_validate_profiles,
    summarize_events,
    validate_replay_bundles,
)
from request_replay_bundle_gate import BundleError, make_bundle, validate_bundle_root
from resource_invariant_gate import GateError as ResourceGateError, check_trace, load_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "scripts/release/scenarios/product_backend_sentinel.json"
PASS_LINE = "PRODUCT BACKEND SENTINEL PASS"
SELFTEST_PASS_LINE = "PRODUCT BACKEND SENTINEL SELFTEST PASS"
SCHEMA_VERSION = 1
REQUIRED_STAGE2_FIXTURES = 12
REQUIRED_PRODUCT_SCENARIOS = {
    "run_first_token",
    "run_multiturn",
    "serve_chat",
    "serve_concurrency_quality",
    "serve_multiturn",
    "serve_stream",
    "serve_structured_output",
    "serve_tool_call",
}


class SentinelError(RuntimeError):
    pass


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SentinelError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SentinelError(f"{path}: expected JSON object")
    return data


def resolve_path(raw: str, *, base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = (base / path).resolve()
    if candidate.exists():
        return candidate
    return (REPO_ROOT / path).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SentinelError(message)


def require_non_empty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.strip(), f"{label} must be a non-empty string")
    return str(value)


def require_string_list(value: Any, label: str) -> list[str]:
    require(isinstance(value, list), f"{label} must be a list")
    require(all(isinstance(item, str) for item in value), f"{label} entries must be strings")
    return list(value)


def require_pass_line_for_dir(value: Any, *, expected_prefix: str, expected_dir: Path, label: str) -> str:
    pass_line = require_non_empty_string(value, label)
    require(pass_line.startswith(f"{expected_prefix}: "), f"{label} must start with {expected_prefix}:")
    raw_path = pass_line.split(":", 1)[1].strip()
    require(raw_path, f"{label} must include an artifact directory")
    actual = Path(raw_path)
    if not actual.is_absolute():
        actual = (REPO_ROOT / actual).resolve()
    else:
        actual = actual.resolve()
    require(
        actual == expected_dir.resolve(),
        f"{label} artifact dir {actual} does not match {expected_dir.resolve()}",
    )
    return pass_line


def manifest_scenarios(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SentinelError(f"manifest.schema_version must be {SCHEMA_VERSION}")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise SentinelError("manifest.scenarios must be a non-empty list")
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise SentinelError(f"manifest.scenarios[{index}] must be an object")
        if not isinstance(scenario.get("name"), str) or not scenario["name"].strip():
            raise SentinelError(f"manifest.scenarios[{index}].name must be non-empty")
        if not isinstance(scenario.get("type"), str) or not scenario["type"].strip():
            raise SentinelError(f"manifest.scenarios[{index}].type must be non-empty")
        product_scenarios = scenario.get("product_scenarios", [])
        if product_scenarios is not None:
            if not isinstance(product_scenarios, list) or not all(
                isinstance(item, str) and item.strip() for item in product_scenarios
            ):
                raise SentinelError(f"manifest.scenarios[{index}].product_scenarios must be a string array")
    return scenarios


def scenario_product_scenarios(scenario: dict[str, Any]) -> list[str]:
    values = scenario.get("product_scenarios", [])
    if values is None:
        return []
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, str) and item.strip()]


def validate_product_scenario_coverage(scenarios: list[dict[str, Any]]) -> list[str]:
    covered: set[str] = set()
    for scenario in scenarios:
        covered.update(scenario_product_scenarios(scenario))
    missing = sorted(REQUIRED_PRODUCT_SCENARIOS - covered)
    require(not missing, f"manifest product_scenarios missing required product scenarios: {missing}")
    return sorted(covered)


def git_value(args: list[str], default: str = "unknown") -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return default
    if proc.returncode != 0:
        return default
    return proc.stdout.strip() or default


def profile_artifact_result(scenario: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    raw_paths = scenario.get("profile_jsonl")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise SentinelError("profile_artifact.profile_jsonl must be a non-empty list")
    paths = [resolve_path(str(path), base=manifest_dir) for path in raw_paths]
    events_by_path = load_and_validate_profiles(paths)
    summary = summarize_events(paths, events_by_path)
    expected_entrypoints = scenario.get("expected_entrypoints") or []
    if not isinstance(expected_entrypoints, list):
        raise SentinelError("profile_artifact.expected_entrypoints must be a list")
    for entrypoint in expected_entrypoints:
        require(
            str(entrypoint) in summary["entrypoints"],
            f"profile artifact missing entrypoint {entrypoint!r}",
        )
    min_replay_commands = int(scenario.get("min_replay_commands", 0))
    require(
        len(summary.get("replay_commands", [])) >= min_replay_commands,
        f"profile artifact replay command count < {min_replay_commands}",
    )
    return {
        "status": "pass",
        "profile_paths": [str(path) for path in paths],
        "entrypoints": summary["entrypoints"],
        "replay_command_count": len(summary.get("replay_commands", [])),
    }


def profile_replay_link_result(scenario_dir: Path) -> dict[str, Any]:
    bundle_root = scenario_dir / "request_dump"
    make_bundle(bundle_root)
    profile = scenario_dir / "profile.jsonl"
    event = {
        "schema_version": 1,
        "event_id": "evt-profile-replay-link",
        "request_id": "req-fixture",
        "correlation_id": "corr-fixture",
        "entrypoint": "run",
        "backend": "synthetic",
        "phase": "request_complete",
        "event_kind": "instant",
        "timestamp": "2026-07-02T00:00:00Z",
        "status": "ok",
        "model": "synthetic/no-weight",
        "replay": {
            "command": "ferrum run synthetic/no-weight",
            "bundle_dir": "request_dump",
        },
        "attributes": {
            "profile_detail": "basic",
            "profile_schema_fingerprint": "obs-v1",
        },
    }
    profile.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    events_by_path = load_and_validate_profiles([profile])
    summary = summarize_events([profile], events_by_path)
    replay_bundles = validate_replay_bundles(events_by_path)
    commands = summary.get("replay_commands", [])
    require(commands, "profile replay link fixture did not produce replay command summary")
    command = commands[0]
    require(command.get("event_id"), "replay command summary missing event_id")
    require(command.get("request_id") == "req-fixture", "replay command request_id mismatch")
    require(command.get("bundle_dir") == "request_dump", "replay command bundle_dir mismatch")
    return {
        "status": "pass",
        "profile": str(profile),
        "replay_command": command,
        "replay_bundle_count": len(replay_bundles),
    }


def error_kind_for_fixture(fixture_kind: str) -> str:
    if fixture_kind == "bad_output":
        return "bad_text"
    if fixture_kind == "oom_admission":
        return "oom"
    if fixture_kind == "panic_error":
        return "panic"
    return "none"


def write_blocker_profile_fixture(
    profile: Path,
    *,
    fixture_kind: str,
    failure_kind: str,
    bundle_dir: str,
) -> None:
    events = []
    if fixture_kind == "oom_admission":
        events.append(
            {
                "schema_version": 1,
                "event_id": "evt-oom-admission-capacity",
                "request_id": "req-fixture",
                "correlation_id": "corr-fixture",
                "entrypoint": "run",
                "backend": "synthetic",
                "phase": "admission",
                "event_kind": "instant",
                "timestamp": "2026-07-02T00:00:00Z",
                "status": "ok",
                "model": "synthetic/no-weight",
                "resource": {
                    "owner_kind": "request",
                    "owner_id": "req-fixture",
                    "resource_kind": "kv_block",
                    "action": "reject",
                    "capacity": 8,
                    "reason": "insufficient_kv_capacity",
                },
                "attributes": {
                    "profile_detail": "basic",
                    "profile_schema_fingerprint": "obs-v1",
                },
            }
        )
    failure_event = {
        "schema_version": 1,
        "event_id": f"evt-{fixture_kind.replace('_', '-')}-blocker",
        "request_id": "req-fixture",
        "correlation_id": "corr-fixture",
        "entrypoint": "run",
        "backend": "synthetic",
        "phase": "decode" if fixture_kind != "oom_admission" else "admission",
        "event_kind": "error",
        "timestamp": "2026-07-02T00:00:01Z",
        "status": "failure",
        "model": "synthetic/no-weight",
        "error": {
            "kind": error_kind_for_fixture(fixture_kind),
            "message": f"synthetic {failure_kind} blocker fixture",
            "blocking": fixture_kind == "panic_error",
        },
        "replay": {
            "command": "ferrum run synthetic/no-weight",
            "bundle_dir": bundle_dir,
        },
        "attributes": {
            "first_failure_event": True,
            "profile_detail": "basic",
            "profile_schema_fingerprint": "obs-v1",
        },
    }
    if fixture_kind == "panic_error":
        failure_event["memory"] = {
            "scope": "process",
            "backend": "synthetic",
            "before_bytes": 2048,
            "after_bytes": 2048,
            "current_bytes": 2048,
            "high_water_bytes": 4096,
        }
    events.append(failure_event)
    profile.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def failure_link_from_blocker(
    *,
    scenario_name: str | None,
    fixture_kind: str,
    profile: Path,
    command: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scenario_name": scenario_name,
        "fixture_kind": fixture_kind,
        "failure_kind": bundle.get("failure_kind"),
        "profile_path": str(profile),
        "profile_event_id": command.get("event_id"),
        "request_id": command.get("request_id"),
        "replay_command": command.get("command"),
        "bundle_dir": str((profile.parent / str(command.get("bundle_dir") or "")).resolve()),
        "failure_diagnostics": bundle.get("failure_diagnostics"),
    }


def resource_trace_result(scenario: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    path_value = scenario.get("trace_jsonl")
    if not isinstance(path_value, str) or not path_value.strip():
        raise SentinelError("resource_trace.trace_jsonl must be a non-empty string")
    path = resolve_path(path_value, base=manifest_dir)
    report = check_trace(load_jsonl(path), source=str(path))
    require(not report.get("failures"), f"resource trace failures: {report.get('failures')}")
    require(int(report.get("leaked_resources") or 0) == 0, "resource trace leaked resources")
    require(int(report.get("underflow_count") or 0) == 0, "resource trace underflow")
    require(int(report.get("silent_oom_count") or 0) == 0, "resource trace silent OOM")
    require(int(report.get("panic_count") or 0) == 0, "resource trace panic")
    return {
        "status": "pass",
        "trace_jsonl": str(path),
        "scenario_count": len(report.get("scenarios", [])),
        "leaked_resources": report.get("leaked_resources", 0),
        "underflow_count": report.get("underflow_count", 0),
    }


def replay_bundle_result(scenario: dict[str, Any], scenario_dir: Path) -> dict[str, Any]:
    fixture_kind = str(scenario.get("fixture_kind") or "normal")
    kwargs: dict[str, Any] = {}
    if fixture_kind == "normal":
        pass
    elif fixture_kind == "bad_output":
        kwargs.update({"bad_output": True, "failure_kind": "bad_output"})
    elif fixture_kind == "oom_admission":
        kwargs.update({"failure_kind": "oom_admission"})
    elif fixture_kind == "panic_error":
        kwargs.update({"failure_kind": "panic_error"})
    else:
        raise SentinelError(f"unknown replay fixture_kind: {fixture_kind}")
    bundle_root = scenario_dir / "bundle"
    make_bundle(bundle_root, **kwargs)
    bundles = validate_bundle_root(bundle_root)
    require(bundles, "replay fixture produced no bundles")
    failure_link = None
    if fixture_kind != "normal":
        profile = scenario_dir / "profile.jsonl"
        write_blocker_profile_fixture(
            profile,
            fixture_kind=fixture_kind,
            failure_kind=fixture_kind,
            bundle_dir="bundle",
        )
        events_by_path = load_and_validate_profiles([profile])
        profile_summary = summarize_events([profile], events_by_path)
        replay_bundles = validate_replay_bundles(events_by_path)
        commands = profile_summary.get("replay_commands", [])
        require(commands, f"{fixture_kind} blocker profile did not produce a replay command")
        require(replay_bundles, f"{fixture_kind} blocker profile did not validate replay bundle")
        failure_link = failure_link_from_blocker(
            scenario_name=scenario.get("name"),
            fixture_kind=fixture_kind,
            profile=profile,
            command=commands[0],
            bundle=bundles[0],
        )
    return {
        "status": "pass",
        "fixture_kind": fixture_kind,
        "bundle_root": str(bundle_root),
        "bundle_count": len(bundles),
        "failure_kinds": sorted(
            {str(bundle.get("failure_kind")) for bundle in bundles if bundle.get("failure_kind")}
        ),
        "failure_link": failure_link,
    }


def parse_sse(body: str) -> dict[str, Any]:
    done_count = 0
    malformed_json = 0
    content_delta_count = 0
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            done_count += 1
            continue
        if not data:
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            malformed_json += 1
            continue
        choices = parsed.get("choices", [])
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and str(delta.get("content") or ""):
                content_delta_count += 1
    return {
        "done_count": done_count,
        "malformed_json": malformed_json,
        "content_delta_count": content_delta_count,
    }


def sse_fixture_body(fixture_kind: str) -> str:
    content = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
    if fixture_kind == "valid":
        return content + "data: [DONE]\n\n"
    if fixture_kind == "missing_done":
        return content
    if fixture_kind == "duplicate_done":
        return content + "data: [DONE]\n\ndata: [DONE]\n\n"
    if fixture_kind == "malformed_json":
        return "data: {not-json}\n\ndata: [DONE]\n\n"
    raise SentinelError(f"unknown sse fixture_kind: {fixture_kind}")


def sse_fixture_result(scenario: dict[str, Any]) -> dict[str, Any]:
    fixture_kind = str(scenario.get("fixture_kind") or "")
    expected_error = str(scenario.get("expected_error") or "")
    parsed = parse_sse(sse_fixture_body(fixture_kind))
    detected_errors = []
    if parsed["done_count"] == 0:
        detected_errors.append("missing_done")
    if parsed["done_count"] > 1:
        detected_errors.append("duplicate_done")
    if parsed["malformed_json"] > 0:
        detected_errors.append("malformed_sse")
    if fixture_kind == "valid":
        require(not detected_errors, f"valid SSE fixture reported errors: {detected_errors}")
        require(parsed["content_delta_count"] > 0, "valid SSE fixture had no content delta")
    else:
        require(expected_error in detected_errors, f"expected {expected_error}, got {detected_errors}")
    return {
        "status": "pass",
        "fixture_kind": fixture_kind,
        "expected_error": expected_error or None,
        "detected_errors": detected_errors,
        **parsed,
    }


def native_op_manifest_result(scenario: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    path_value = scenario.get("manifest")
    if not isinstance(path_value, str) or not path_value.strip():
        raise SentinelError("native_op_manifest.manifest must be a non-empty string")
    path = resolve_path(path_value, base=manifest_dir)
    manifest = read_native_json(path)
    requirement = ResolverRequirement(
        operator=str(scenario.get("operator") or manifest.get("operator") or ""),
        backend=str(scenario.get("backend") or manifest.get("backend") or ""),
        compute_capability=scenario.get("compute_capability"),
        source_package_sha256=scenario.get("source_package_sha256"),
        inputs_sha256=scenario.get("inputs_sha256"),
        binary_sha256=scenario.get("binary_sha256"),
    )
    resolution = resolve_manifest(manifest, requirement, context=str(path))
    return {"status": "pass", "manifest": str(path), "resolution": resolution}


def validate_actual_smoke_offline_replay_summary(
    summary: dict[str, Any],
    *,
    label: str,
) -> dict[str, int]:
    require(summary.get("status") == "pass", f"{label}.status must be pass")
    counts: dict[str, int] = {}
    for key in ("bundle_count", "replay_execution_count", "replay_execution_skipped_count"):
        value = summary.get(key)
        require(isinstance(value, int) and value >= 0, f"{label}.{key} must be a non-negative integer")
        counts[key] = value
    require(counts["bundle_count"] > 0, f"{label}.bundle_count must be positive")
    require(
        counts["replay_execution_count"] > 0,
        f"{label} did not execute any offline replay",
    )
    require(
        counts["replay_execution_skipped_count"] < counts["bundle_count"],
        f"{label} skipped all replay bundles",
    )
    replay_executions = summary.get("replay_executions")
    require(isinstance(replay_executions, list), f"{label}.replay_executions must be a list")
    skipped = [
        item
        for item in replay_executions
        if isinstance(item, dict) and item.get("status") == "skipped_requires_running_server"
    ]
    require(
        len(skipped) == counts["replay_execution_skipped_count"],
        f"{label}.replay_execution_skipped_count does not match skipped replay records",
    )
    for item in skipped:
        source = str(item.get("source_bundle_dir") or "").replace("\\", "/")
        require(source, f"{label}.skipped source_bundle_dir is required")
        require(
            "/run/request_dump/" not in source and not source.endswith("/run/request_dump"),
            f"{label} skipped run replay bundle: {source}",
        )
    return counts


def actual_smoke_result(actual_smoke: Path) -> dict[str, Any]:
    require(actual_smoke.is_dir(), f"actual smoke artifact must be a directory: {actual_smoke}")
    manifest_path = actual_smoke / "gate.manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "pass", f"{manifest_path}.status must be pass")
    require(
        manifest.get("phase") == "product_observability_l1_smoke",
        f"{manifest_path}.phase must be product_observability_l1_smoke",
    )
    pass_line = require_pass_line_for_dir(
        manifest.get("pass_line"),
        expected_prefix="PRODUCT OBSERVABILITY L1 SMOKE PASS",
        expected_dir=actual_smoke,
        label=f"{manifest_path}.pass_line",
    )
    expected_sha = git_value(["rev-parse", "HEAD"])
    git_sha = require_non_empty_string(manifest.get("git_sha"), f"{manifest_path}.git_sha")
    require(git_sha == expected_sha, f"{manifest_path}.git_sha {git_sha} is stale vs HEAD {expected_sha}")
    require(
        manifest.get("git_dirty") is False,
        f"{manifest_path}.git_dirty must be false for actual smoke evidence",
    )
    dirty_files = require_string_list(manifest.get("dirty_files", []), f"{manifest_path}.dirty_files")
    require(not dirty_files, f"{manifest_path}.dirty_files must be empty for actual smoke evidence")

    summary_path = actual_smoke / "product_observability_l1_smoke_summary.json"
    summary = read_json(summary_path)
    require(summary.get("status") == "pass", f"{summary_path} is not pass")
    require(
        summary.get("pass_line") == pass_line,
        f"{summary_path}.pass_line must match manifest pass_line",
    )
    require(
        summary.get("git_sha") == git_sha,
        f"{summary_path}.git_sha must match manifest git_sha",
    )
    require(
        summary.get("git_dirty") is False,
        f"{summary_path}.git_dirty must be false for actual smoke evidence",
    )
    require(
        require_string_list(summary.get("dirty_files", []), f"{summary_path}.dirty_files") == [],
        f"{summary_path}.dirty_files must be empty for actual smoke evidence",
    )
    summary_artifact_dir = Path(
        require_non_empty_string(summary.get("artifact_dir"), f"{summary_path}.artifact_dir")
    )
    if not summary_artifact_dir.is_absolute():
        summary_artifact_dir = (REPO_ROOT / summary_artifact_dir).resolve()
    else:
        summary_artifact_dir = summary_artifact_dir.resolve()
    require(
        summary_artifact_dir == actual_smoke.resolve(),
        f"{summary_path}.artifact_dir must match actual smoke directory",
    )
    require(
        summary.get("actual_model_smoke") is True,
        f"{summary_path}.actual_model_smoke must be true",
    )
    profile_paths = [
        actual_smoke / "run/profile.jsonl",
        actual_smoke / "run/memory_profile.jsonl",
        actual_smoke / "run/scheduler_trace.jsonl",
        actual_smoke / "serve/profile.jsonl",
        actual_smoke / "serve/memory_profile.jsonl",
        actual_smoke / "serve/scheduler_trace.jsonl",
    ]
    events_by_path = load_and_validate_profiles(profile_paths)
    profile_summary = summarize_events(profile_paths, events_by_path)
    require("run" in profile_summary["entrypoints"], "actual smoke missing run profile events")
    require("serve" in profile_summary["entrypoints"], "actual smoke missing serve profile events")
    replay_bundle_summary = []
    for bundle_root in [actual_smoke / "run/request_dump", actual_smoke / "serve/request_dump"]:
        replay_bundle_summary.extend(validate_bundle_root(bundle_root))
    for trace in [actual_smoke / "run/scheduler_trace.jsonl", actual_smoke / "serve/scheduler_trace.jsonl"]:
        report = check_trace(load_jsonl(trace), source=str(trace))
        require(not report.get("failures"), f"actual smoke resource trace failed: {trace}")
        require(int(report.get("leaked_resources") or 0) == 0, f"actual smoke resource leak: {trace}")
        require(int(report.get("underflow_count") or 0) == 0, f"actual smoke resource underflow: {trace}")
    live_summary_path = actual_smoke / "serve_live_replay_bundle/request_replay_bundle_summary.json"
    live_summary = read_json(live_summary_path)
    require(live_summary.get("status") == "pass", f"{live_summary_path} is not pass")
    require(
        int(live_summary.get("replay_execution_count") or 0) > 0,
        "actual smoke live replay did not execute any request",
    )
    require(
        int(live_summary.get("replay_execution_skipped_count") or 0) == 0,
        "actual smoke live replay skipped requests",
    )
    offline_summary_path = actual_smoke / "request_replay_bundle/request_replay_bundle_summary.json"
    offline_summary = read_json(offline_summary_path)
    offline_replay = validate_actual_smoke_offline_replay_summary(
        offline_summary,
        label=str(offline_summary_path),
    )
    summary_replay = summary.get("request_replay_bundle")
    require(
        isinstance(summary_replay, dict),
        f"{summary_path}.request_replay_bundle must be an object",
    )
    summary_replay_path = Path(
        require_non_empty_string(
            summary_replay.get("summary"),
            f"{summary_path}.request_replay_bundle.summary",
        )
    )
    if not summary_replay_path.is_absolute():
        summary_replay_path = (actual_smoke / summary_replay_path).resolve()
    else:
        summary_replay_path = summary_replay_path.resolve()
    require(
        summary_replay_path == offline_summary_path.resolve(),
        f"{summary_path}.request_replay_bundle.summary must match offline replay summary",
    )
    for key, value in offline_replay.items():
        require(
            summary_replay.get(key) == value,
            f"{summary_path}.request_replay_bundle.{key} must match offline summary",
        )
    return {
        "status": "pass",
        "actual_smoke": str(actual_smoke),
        "git_sha": git_sha,
        "git_dirty": False,
        "pass_line": pass_line,
        "model": summary.get("model"),
        "requested_backend": summary.get("requested_backend"),
        "effective_backend": summary.get("effective_backend"),
        "profile_detail": summary.get("profile_detail"),
        "entrypoints": profile_summary["entrypoints"],
        "replay_bundle_count": len(replay_bundle_summary),
        "offline_replay_execution_count": offline_replay["replay_execution_count"],
        "offline_replay_skipped_count": offline_replay["replay_execution_skipped_count"],
        "live_replay_execution_count": live_summary.get("replay_execution_count"),
    }


def scenario_summary_result(summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    require(summary.get("status") == "pass", f"{summary_path} is not pass")
    observability = summary.get("observability")
    if not isinstance(observability, dict) or observability.get("enabled") is not True:
        raise SentinelError(f"{summary_path}: observability.enabled=true is required")
    profile_paths = [
        Path(path)
        for path in observability.get("profile_paths", [])
        if isinstance(path, str) and path.strip()
    ]
    require(profile_paths, f"{summary_path}: observability.profile_paths is empty")
    events_by_path = load_and_validate_profiles(profile_paths)
    profile_summary = summarize_events(profile_paths, events_by_path)
    require("run" in profile_summary["entrypoints"], "scenario summary missing run profile events")
    require("serve" in profile_summary["entrypoints"], "scenario summary missing serve profile events")
    require(
        len(profile_summary.get("replay_commands", [])) > 0,
        "scenario summary profile events do not link replay commands",
    )
    replay_bundle_summary = validate_replay_bundles(events_by_path)
    request_dump_dirs = [
        Path(path)
        for path in observability.get("request_dump_dirs", [])
        if isinstance(path, str) and path.strip()
    ]
    for request_dump_dir in request_dump_dirs:
        replay_bundle_summary.extend(validate_bundle_root(request_dump_dir))
    scheduler_trace_paths = [
        Path(path)
        for path in observability.get("scheduler_trace_paths", [])
        if isinstance(path, str) and path.strip()
    ]
    require(scheduler_trace_paths, "scenario summary has no scheduler_trace_paths")
    for trace in scheduler_trace_paths:
        report = check_trace(load_jsonl(trace), source=str(trace))
        require(not report.get("failures"), f"scenario summary resource trace failed: {trace}")
        require(int(report.get("leaked_resources") or 0) == 0, f"scenario summary resource leak: {trace}")
        require(int(report.get("underflow_count") or 0) == 0, f"scenario summary resource underflow: {trace}")
    return {
        "status": "pass",
        "scenario_summary": str(summary_path),
        "entrypoints": profile_summary["entrypoints"],
        "replay_command_count": len(profile_summary.get("replay_commands", [])),
        "replay_bundle_count": len(replay_bundle_summary),
        "scheduler_trace_count": len(scheduler_trace_paths),
    }


def collect_failure_links(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    links = []
    for result in results:
        link = result.get("failure_link")
        if isinstance(link, dict):
            links.append(link)
    return links


def validate_failure_links(failure_links: list[dict[str, Any]]) -> None:
    required = {"bad_output", "oom_admission", "panic_error"}
    seen: set[str] = set()
    for index, link in enumerate(failure_links):
        context = f"failure_links[{index}]"
        failure_kind = link.get("failure_kind")
        require(isinstance(failure_kind, str) and failure_kind.strip(), f"{context}.failure_kind is required")
        seen.add(failure_kind)
        for key in ("profile_event_id", "request_id", "replay_command", "bundle_dir"):
            require(isinstance(link.get(key), str) and link[key].strip(), f"{context}.{key} is required")
        if failure_kind != "bad_output":
            require(
                isinstance(link.get("failure_diagnostics"), str) and link["failure_diagnostics"].strip(),
                f"{context}.failure_diagnostics is required for {failure_kind}",
            )
    missing = sorted(required - seen)
    require(not missing, f"failure_links missing required blocker classes: {missing}")


def run_scenario(scenario: dict[str, Any], manifest_dir: Path, scenario_dir: Path) -> dict[str, Any]:
    typ = str(scenario["type"])
    if typ == "profile_artifact":
        return profile_artifact_result(scenario, manifest_dir)
    if typ == "profile_replay_link":
        return profile_replay_link_result(scenario_dir)
    if typ == "resource_trace":
        return resource_trace_result(scenario, manifest_dir)
    if typ == "replay_bundle":
        return replay_bundle_result(scenario, scenario_dir)
    if typ == "sse_fixture":
        return sse_fixture_result(scenario)
    if typ == "native_op_manifest":
        return native_op_manifest_result(scenario, manifest_dir)
    raise SentinelError(f"unknown sentinel scenario type: {typ}")


def scenario_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in name).strip("-") or "scenario"


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    started_at = int(time.time())
    manifest = read_json(args.manifest)
    scenarios = manifest_scenarios(manifest)
    required_count = int(manifest.get("required_stage2_fixture_count") or REQUIRED_STAGE2_FIXTURES)
    require(
        len(scenarios) >= required_count,
        f"manifest must include at least {required_count} stage 2 fixtures",
    )
    covered_product_scenarios = validate_product_scenario_coverage(scenarios)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.manifest.parent
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for scenario in scenarios:
        name = str(scenario["name"])
        scenario_dir = out / scenario_slug(name)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_scenario(scenario, manifest_dir, scenario_dir)
            result.update(
                {
                    "name": name,
                    "type": scenario["type"],
                    "product_scenarios": scenario_product_scenarios(scenario),
                    "status": "pass",
                }
            )
        except Exception as exc:
            result = {
                "name": name,
                "type": scenario["type"],
                "product_scenarios": scenario_product_scenarios(scenario),
                "status": "fail",
                "error": str(exc),
            }
            failures.append(result)
        write_json(scenario_dir / "result.json", result)
        results.append(result)
    actual_smoke = None
    if args.actual_smoke is not None:
        actual_smoke = actual_smoke_result(args.actual_smoke)
        write_json(out / "actual_smoke_result.json", actual_smoke)
    scenario_summary = None
    if args.scenario_summary is not None:
        scenario_summary = scenario_summary_result(args.scenario_summary)
        write_json(out / "scenario_summary_result.json", scenario_summary)
    failure_links = collect_failure_links(results)
    if not failures:
        validate_failure_links(failure_links)
    status = "fail" if failures else "pass"
    ended_at = int(time.time())
    dirty_files = git_value(["status", "--short"], default="").splitlines()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "gate": "product_backend_sentinel",
        "status": status,
        "manifest": str(args.manifest),
        "artifact_dir": str(out),
        "scenario_count": len(results),
        "required_stage2_fixture_count": required_count,
        "required_product_scenarios": sorted(REQUIRED_PRODUCT_SCENARIOS),
        "product_scenarios": covered_product_scenarios,
        "failed": len(failures),
        "scenarios": results,
        "failure_links": failure_links,
        "actual_smoke": actual_smoke,
        "scenario_summary": scenario_summary,
        "git_sha": git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "started_at_unix": started_at,
        "ended_at_unix": ended_at,
        "duration_sec": ended_at - started_at,
        "pass_line": f"{PASS_LINE}: {out}" if status == "pass" else None,
    }
    write_json(out / "product_backend_sentinel_summary.json", summary)
    write_json(
        out / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "gate": "product_backend_sentinel",
            "status": status,
            "artifact_dir": str(out),
            "summary": str(out / "product_backend_sentinel_summary.json"),
            "pass_line": summary["pass_line"],
            "git_sha": summary["git_sha"],
            "git_dirty": summary["git_dirty"],
            "dirty_files": summary["dirty_files"],
        },
    )
    if failures:
        raise SentinelError(f"{len(failures)} sentinel scenarios failed")
    (out / "pass_line.txt").write_text(f"{PASS_LINE}: {out}\n", encoding="utf-8")
    return summary


def write_profile_link_fixture(path: Path, *, entrypoint: str, request_id: str, bundle_dir: Path) -> None:
    event = {
        "schema_version": 1,
        "event_id": f"evt-{entrypoint}-scenario-summary",
        "request_id": request_id,
        "correlation_id": f"corr-{entrypoint}",
        "entrypoint": entrypoint,
        "backend": "synthetic",
        "phase": "request_complete",
        "event_kind": "instant",
        "timestamp": "2026-07-02T00:00:00Z",
        "status": "ok",
        "model": "synthetic/no-weight",
        "replay": {
            "command": f"ferrum {entrypoint} synthetic/no-weight",
            "bundle_dir": str(bundle_dir),
        },
        "attributes": {
            "profile_detail": "basic",
            "profile_schema_fingerprint": "obs-v1",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")


def make_scenario_summary_fixture(root: Path) -> Path:
    run_root = root / "scenario-run"
    serve_root = root / "scenario-serve"
    make_bundle(run_root / "request_dump")
    make_bundle(serve_root / "request_dump")
    write_profile_link_fixture(
        run_root / "profile.jsonl",
        entrypoint="run",
        request_id="req-fixture",
        bundle_dir=run_root / "request_dump",
    )
    write_profile_link_fixture(
        serve_root / "profile.jsonl",
        entrypoint="serve",
        request_id="req-fixture",
        bundle_dir=serve_root / "request_dump",
    )
    trace = REPO_ROOT / "scripts/release/fixtures/resource_invariant/pass/oom_prevented_by_admission.jsonl"
    summary = {
        "schema_version": 1,
        "status": "pass",
        "observability": {
            "enabled": True,
            "profile_paths": [str(run_root / "profile.jsonl"), str(serve_root / "profile.jsonl")],
            "scheduler_trace_paths": [str(trace)],
            "request_dump_dirs": [str(run_root / "request_dump"), str(serve_root / "request_dump")],
        },
    }
    path = root / "scenario-summary.json"
    write_json(path, summary)
    return path


def actual_smoke_profile_event(
    *,
    entrypoint: str,
    request_id: str,
    event_id: str,
    phase: str,
    event_kind: str = "instant",
    resource: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
    replay_bundle_dir: Path | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "request_id": request_id,
        "correlation_id": f"corr-{entrypoint}",
        "entrypoint": entrypoint,
        "backend": "metal",
        "phase": phase,
        "event_kind": event_kind,
        "timestamp": "2026-07-02T00:00:00Z",
        "status": "ok",
        "model": "fixture/actual-model",
        "attributes": {
            "actual_model_smoke": True,
            "profile_detail": "basic",
            "profile_schema_fingerprint": "obs-v1",
        },
    }
    if resource is not None:
        event["resource"] = resource
    if memory is not None:
        event["memory"] = memory
        event["duration_us"] = 100
        event["attributes"]["memory_measurement"] = "process_rss"
    if replay_bundle_dir is not None:
        event["replay"] = {
            "command": f"ferrum {entrypoint} fixture/actual-model",
            "bundle_dir": str(replay_bundle_dir),
        }
    return event


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def write_actual_smoke_entrypoint(root: Path, entrypoint: str) -> None:
    request_id = f"req-{entrypoint}"
    request_dump = root / "request_dump"
    make_bundle(request_dump)
    profile = actual_smoke_profile_event(
        entrypoint=entrypoint,
        request_id=request_id,
        event_id=f"evt-{entrypoint}-complete",
        phase="request_complete",
        replay_bundle_dir=request_dump,
    )
    memory = actual_smoke_profile_event(
        entrypoint=entrypoint,
        request_id=request_id,
        event_id=f"evt-{entrypoint}-memory",
        phase="first_request_done",
        event_kind="timed_span",
        memory={
            "scope": "process",
            "backend": "metal",
            "before_bytes": 1024,
            "after_bytes": 2048,
            "current_bytes": 2048,
            "high_water_bytes": 4096,
            "available_bytes": 8192,
        },
    )
    resource_events = [
        actual_smoke_profile_event(
            entrypoint=entrypoint,
            request_id=request_id,
            event_id=f"evt-{entrypoint}-open",
            phase="request_open",
            resource={
                "owner_kind": "request",
                "owner_id": request_id,
                "resource_kind": "request_slot",
                "action": "request_open",
                "capacity": 1,
            },
        ),
        actual_smoke_profile_event(
            entrypoint=entrypoint,
            request_id=request_id,
            event_id=f"evt-{entrypoint}-kv-reserve",
            phase="kv_reserve",
            resource={
                "owner_kind": "request",
                "owner_id": request_id,
                "resource_kind": "kv_block",
                "action": "reserve",
                "amount": 1,
                "before": 1,
                "after": 0,
                "capacity": 1,
            },
        ),
        actual_smoke_profile_event(
            entrypoint=entrypoint,
            request_id=request_id,
            event_id=f"evt-{entrypoint}-kv-commit",
            phase="kv_commit",
            resource={
                "owner_kind": "request",
                "owner_id": request_id,
                "resource_kind": "kv_block",
                "action": "commit",
                "amount": 1,
                "before": 0,
                "after": 1,
                "capacity": 1,
            },
        ),
        actual_smoke_profile_event(
            entrypoint=entrypoint,
            request_id=request_id,
            event_id=f"evt-{entrypoint}-kv-release",
            phase="kv_release",
            resource={
                "owner_kind": "request",
                "owner_id": request_id,
                "resource_kind": "kv_block",
                "action": "release",
                "amount": 1,
                "before": 0,
                "after": 1,
                "capacity": 1,
            },
        ),
        actual_smoke_profile_event(
            entrypoint=entrypoint,
            request_id=request_id,
            event_id=f"evt-{entrypoint}-close",
            phase="request_close",
            resource={
                "owner_kind": "request",
                "owner_id": request_id,
                "resource_kind": "request_slot",
                "action": "request_close",
                "capacity": 1,
            },
        ),
    ]
    for event in resource_events:
        event["attributes"]["resource_trace_source"] = "engine"
    write_jsonl(root / "profile.jsonl", [profile])
    write_jsonl(root / "memory_profile.jsonl", [memory])
    write_jsonl(root / "scheduler_trace.jsonl", resource_events)


def make_actual_smoke_fixture(root: Path, *, sha: str, dirty: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_actual_smoke_entrypoint(root / "run", "run")
    write_actual_smoke_entrypoint(root / "serve", "serve")
    offline_replay = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "bundle_count": 2,
        "execute_replay": True,
        "replay_execution_count": 1,
        "replay_execution_skipped_count": 1,
        "replay_executions": [
            {
                "source_bundle_dir": str(root / "run/request_dump/req-fixture"),
                "status": "executed_synthetic",
                "artifact_dir": str(root / "request_replay_bundle/executed_replays/req-run"),
            },
            {
                "source_bundle_dir": str(root / "serve/request_dump/req-fixture"),
                "status": "skipped_requires_running_server",
                "reason": "offline replay execution only runs synthetic/no-weight commands",
            },
        ],
    }
    write_json(root / "request_replay_bundle/request_replay_bundle_summary.json", offline_replay)
    live_replay = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "replay_execution_count": 1,
        "replay_execution_skipped_count": 0,
    }
    write_json(root / "serve_live_replay_bundle/request_replay_bundle_summary.json", live_replay)
    pass_line = f"PRODUCT OBSERVABILITY L1 SMOKE PASS: {root}"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "gate": "product_observability_l1_smoke",
        "artifact_dir": str(root),
        "pass_line": pass_line,
        "git_sha": sha,
        "git_dirty": dirty,
        "dirty_files": ["dirty-fixture"] if dirty else [],
        "model": "fixture/actual-model",
        "requested_backend": "metal",
        "backend": "metal",
        "effective_backend": "metal",
        "profile_detail": "basic",
        "actual_model_smoke": True,
        "request_replay_bundle": {
            "out": str(root / "request_replay_bundle"),
            "summary": str(root / "request_replay_bundle/request_replay_bundle_summary.json"),
            "bundle_count": offline_replay["bundle_count"],
            "replay_execution_count": offline_replay["replay_execution_count"],
            "replay_execution_skipped_count": offline_replay["replay_execution_skipped_count"],
        },
    }
    write_json(root / "product_observability_l1_smoke_summary.json", summary)
    write_json(
        root / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "goal": "release-regression-hardening-2026-06-28",
            "phase": "product_observability_l1_smoke",
            "status": "pass",
            "git_sha": sha,
            "git_dirty": dirty,
            "dirty_files": ["dirty-fixture"] if dirty else [],
            "artifact_dir": str(root),
            "pass_line": pass_line,
            "summary": str(root / "product_observability_l1_smoke_summary.json"),
        },
    )
    return root


def run_selftest() -> None:
    temp = Path(tempfile.mkdtemp(prefix="ferrum-product-backend-sentinel-"))
    try:
        out = temp / "out"
        summary = run_gate(
            argparse.Namespace(
                manifest=DEFAULT_MANIFEST,
                out=out,
                actual_smoke=None,
                scenario_summary=None,
            )
        )
        if summary.get("status") != "pass":
            raise AssertionError(summary)
        if summary.get("scenario_count") != REQUIRED_STAGE2_FIXTURES:
            raise AssertionError(summary)
        validate_failure_links(summary.get("failure_links") or [])
        bad_manifest_path = temp / "missing-product-scenario.json"
        bad_manifest = read_json(DEFAULT_MANIFEST)
        for scenario in bad_manifest["scenarios"]:
            if isinstance(scenario, dict):
                scenario["product_scenarios"] = [
                    item
                    for item in scenario.get("product_scenarios", [])
                    if item != "serve_tool_call"
                ]
        write_json(bad_manifest_path, bad_manifest)
        try:
            run_gate(
                argparse.Namespace(
                    manifest=bad_manifest_path,
                    out=temp / "bad-missing-product-scenario",
                    actual_smoke=None,
                    scenario_summary=None,
                )
            )
            raise AssertionError("missing product scenario coverage unexpectedly passed")
        except SentinelError as exc:
            require("product_scenarios missing" in str(exc), f"unexpected product scenario error: {exc}")
        scenario_summary = make_scenario_summary_fixture(temp / "scenario-summary-fixture")
        scenario_out = temp / "scenario-out"
        scenario_gate_summary = run_gate(
            argparse.Namespace(
                manifest=DEFAULT_MANIFEST,
                out=scenario_out,
                actual_smoke=None,
                scenario_summary=scenario_summary,
            )
        )
        if scenario_gate_summary.get("scenario_summary", {}).get("status") != "pass":
            raise AssertionError(scenario_gate_summary)
        validate_failure_links(scenario_gate_summary.get("failure_links") or [])
        head = git_value(["rev-parse", "HEAD"])
        actual_smoke = make_actual_smoke_fixture(temp / "actual-smoke", sha=head)
        actual_out = temp / "actual-out"
        actual_gate_summary = run_gate(
            argparse.Namespace(
                manifest=DEFAULT_MANIFEST,
                out=actual_out,
                actual_smoke=actual_smoke,
                scenario_summary=None,
            )
        )
        if actual_gate_summary.get("actual_smoke", {}).get("status") != "pass":
            raise AssertionError(actual_gate_summary)
        stale_smoke = make_actual_smoke_fixture(temp / "stale-smoke", sha="0" * 40)
        try:
            run_gate(
                argparse.Namespace(
                    manifest=DEFAULT_MANIFEST,
                    out=temp / "bad-stale-smoke",
                    actual_smoke=stale_smoke,
                    scenario_summary=None,
                )
            )
            raise AssertionError("stale actual smoke unexpectedly passed")
        except SentinelError as exc:
            require("stale vs HEAD" in str(exc), f"unexpected stale actual smoke error: {exc}")
        dirty_smoke = make_actual_smoke_fixture(temp / "dirty-smoke", sha=head, dirty=True)
        try:
            run_gate(
                argparse.Namespace(
                    manifest=DEFAULT_MANIFEST,
                    out=temp / "bad-dirty-smoke",
                    actual_smoke=dirty_smoke,
                    scenario_summary=None,
                )
            )
            raise AssertionError("dirty actual smoke unexpectedly passed")
        except SentinelError as exc:
            require("git_dirty" in str(exc), f"unexpected dirty actual smoke error: {exc}")
        skipped_smoke = make_actual_smoke_fixture(temp / "skipped-smoke", sha=head)
        skipped_summary_path = skipped_smoke / "request_replay_bundle/request_replay_bundle_summary.json"
        skipped_summary = read_json(skipped_summary_path)
        skipped_summary["replay_execution_count"] = 0
        skipped_summary["replay_execution_skipped_count"] = skipped_summary["bundle_count"]
        skipped_summary["replay_executions"] = [
            {
                "source_bundle_dir": str(skipped_smoke / "run/request_dump/req-fixture"),
                "status": "skipped_requires_running_server",
            },
            {
                "source_bundle_dir": str(skipped_smoke / "serve/request_dump/req-fixture"),
                "status": "skipped_requires_running_server",
            },
        ]
        write_json(skipped_summary_path, skipped_summary)
        try:
            run_gate(
                argparse.Namespace(
                    manifest=DEFAULT_MANIFEST,
                    out=temp / "bad-skipped-smoke",
                    actual_smoke=skipped_smoke,
                    scenario_summary=None,
                )
            )
            raise AssertionError("all-skipped actual smoke replay unexpectedly passed")
        except SentinelError as exc:
            require("did not execute any offline replay" in str(exc), f"unexpected replay error: {exc}")
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--actual-smoke", type=Path)
    parser.add_argument("--scenario-summary", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_selftest()
            print(SELFTEST_PASS_LINE)
            return 0
        if args.out is None:
            raise SentinelError("--out is required unless --self-test is set")
        summary = run_gate(args)
    except (
        SentinelError,
        ProfileGateError,
        BundleError,
        ResourceGateError,
        NativeGateError,
    ) as exc:
        out = args.out or Path("<unset>")
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            summary_path = args.out / "product_backend_sentinel_summary.json"
            if not summary_path.exists():
                write_json(summary_path, {"status": "fail", "error": str(exc)})
        print(f"PRODUCT BACKEND SENTINEL FAIL: {out}: {exc}", file=sys.stderr)
        return 1
    print(summary["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
