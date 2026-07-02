#!/usr/bin/env python3
"""Offline gate for product observability profile summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from analyze_ferrum_profile import ValidationError, validate_profile_event


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES = REPO_ROOT / "scripts/release/fixtures/observability_profile"
PASS_LINE = "OBSERVABILITY PROFILE GATE PASS"
SELFTEST_PASS_LINE = "OBSERVABILITY PROFILE GATE SELFTEST PASS"
GOAL = "release-regression-hardening-2026-06-28"
SCHEMA_VERSION = 1
BAD_TEXT_KINDS = {
    "bad_output",
    "bad_text",
    "reserved_token",
    "invalid_utf8",
    "mojibake",
    "missing_done",
    "duplicate_done",
    "malformed_sse",
    "strict_schema_failure",
    "required_tool_failure",
}
OOM_KINDS = {"cuda_oom", "metal_oom", "oom", "silent_oom"}


class GateError(RuntimeError):
    pass


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"{path}:{line_no} invalid JSON: {exc}") from exc
            try:
                validate_profile_event(event, f"{path}:{line_no}")
            except ValidationError as exc:
                raise GateError(str(exc)) from exc
            events.append(event)
    if not events:
        raise GateError(f"{path} must contain at least one profile event")
    return events


def attrs(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("attributes", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise GateError(f"{event.get('event_id', '<unknown>')}.attributes must be an object")
    return value


def err_kind(event: dict[str, Any]) -> str:
    error = event.get("error")
    if not isinstance(error, dict):
        return ""
    kind = error.get("kind")
    return kind if isinstance(kind, str) else ""


def percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = round((len(sorted_values) - 1) * pct)
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def capacity_events(events: list[dict[str, Any]]) -> set[str]:
    requests: set[str] = set()
    for event in events:
        resource = event.get("resource")
        if not isinstance(resource, dict):
            continue
        action = resource.get("action")
        if action in {"defer", "reject", "capacity_snapshot"} and isinstance(
            resource.get("capacity"), int
        ):
            requests.add(str(event.get("request_id", "")))
    return requests


def validate_gate_semantics(path: Path, events: list[dict[str, Any]]) -> None:
    fingerprints = {
        attrs(event).get("profile_schema_fingerprint")
        for event in events
        if attrs(event).get("profile_schema_fingerprint") is not None
    }
    if len(fingerprints) > 1:
        raise GateError(f"{path} mixes profile schema fingerprints: {sorted(fingerprints)}")

    for event in events:
        event_attrs = attrs(event)
        context = f"{path}:{event.get('event_id', '<unknown>')}"
        resource = event.get("resource")
        if isinstance(resource, dict) and resource.get("action") in {"defer", "reject"}:
            reason = str(resource.get("reason", "")).strip()
            if not isinstance(resource.get("capacity"), int) or not reason:
                raise GateError(f"{context} defer/reject requires capacity and reason")
        available = (event.get("memory") or {}).get("available_bytes")
        if available is not None and available < 0:
            raise GateError(f"{context} memory.available_bytes must not be negative")
        if isinstance(event_attrs.get("resource_leak_count"), int) and event_attrs[
            "resource_leak_count"
        ] > 0:
            raise GateError(f"{context} reports resource_leak_count={event_attrs['resource_leak_count']}")
        if event_attrs.get("performance_claim") is True and (
            event_attrs.get("profile_detail") in {"debug", "full"}
            or event_attrs.get("diagnostic_only") is True
        ):
            raise GateError(f"{context} uses diagnostic profile as performance claim")
        profile_count = event_attrs.get("profile_completed_requests")
        prometheus_count = event_attrs.get("prometheus_completed_requests")
        if profile_count is not None or prometheus_count is not None:
            if profile_count != prometheus_count:
                raise GateError(
                    f"{context} profile/prometheus completed request mismatch: "
                    f"{profile_count} != {prometheus_count}"
                )

    capacity_by_request = capacity_events(events)
    for event in events:
        if event.get("status") != "failure":
            continue
        kind = err_kind(event)
        event_attrs = attrs(event)
        context = f"{path}:{event.get('event_id', '<unknown>')}"
        error = event.get("error") if isinstance(event.get("error"), dict) else {}
        if error.get("blocking") is True and event_attrs.get("first_failure_event") is not True:
            raise GateError(f"{context} blocking failure requires first_failure_event=true")
        if kind in BAD_TEXT_KINDS and "replay" not in event:
            raise GateError(f"{context} correctness failure requires replay command")
        if kind in OOM_KINDS and str(event.get("request_id", "")) not in capacity_by_request:
            raise GateError(f"{context} OOM failure requires admission/defer/reject evidence")


def event_summary(event: dict[str, Any]) -> dict[str, Any]:
    error = event.get("error") if isinstance(event.get("error"), dict) else None
    return {
        "event_id": event.get("event_id"),
        "request_id": event.get("request_id"),
        "entrypoint": event.get("entrypoint"),
        "backend": event.get("backend"),
        "phase": event.get("phase"),
        "event_kind": event.get("event_kind"),
        "status": event.get("status"),
        "error": error,
    }


def summarize_events(profile_paths: list[Path], events_by_path: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    seen_event_keys: set[tuple[str, str, str, str]] = set()
    all_events: list[dict[str, Any]] = []
    for events in events_by_path.values():
        for event in events:
            key = (
                str(event.get("event_id", "")),
                str(event.get("request_id", "")),
                str(event.get("entrypoint", "")),
                str(event.get("phase", "")),
            )
            if key in seen_event_keys:
                continue
            seen_event_keys.add(key)
            all_events.append(event)
    request_ids = {str(event.get("request_id")) for event in all_events if event.get("request_id")}
    failed_events = [event for event in all_events if event.get("status") == "failure"]
    durations = sorted(
        int(event["duration_us"])
        for event in all_events
        if event.get("event_kind") == "timed_span" and isinstance(event.get("duration_us"), int)
    )
    memory_high_by_backend_scope: dict[str, int] = {}
    for event in all_events:
        memory = event.get("memory")
        if not isinstance(memory, dict):
            continue
        backend = event.get("backend", "unknown")
        scope = memory.get("scope", "unknown")
        key = f"{backend}:{scope}"
        memory_high_by_backend_scope[key] = max(
            memory_high_by_backend_scope.get(key, 0),
            int(memory.get("high_water_bytes", 0)),
        )
    slow_events = sorted(
        (
            event
            for event in all_events
            if event.get("event_kind") == "timed_span" and isinstance(event.get("duration_us"), int)
        ),
        key=lambda event: int(event["duration_us"]),
        reverse=True,
    )[:5]
    first_failure = event_summary(failed_events[0]) if failed_events else None
    replay_commands = []
    for event in all_events:
        replay = event.get("replay")
        if isinstance(replay, dict) and isinstance(replay.get("command"), str):
            replay_commands.append(
                {
                    "event_id": event.get("event_id"),
                    "request_id": event.get("request_id"),
                    "command": replay["command"],
                    "bundle_dir": replay.get("bundle_dir"),
                }
            )
    bad_text_count = sum(1 for event in failed_events if err_kind(event) in BAD_TEXT_KINDS)
    silent_oom_count = sum(1 for event in failed_events if err_kind(event) in {"silent_oom"})
    oom_prevented_count = 0
    for event in all_events:
        resource = event.get("resource")
        reason = ""
        if isinstance(resource, dict):
            reason = str(resource.get("reason", "")).lower()
        if attrs(event).get("oom_prevented") is True or "oom" in reason:
            oom_prevented_count += 1
    resource_leak_count = sum(
        int(attrs(event).get("resource_leak_count", 0))
        for event in all_events
        if isinstance(attrs(event).get("resource_leak_count", 0), int)
    )
    corrupted_count = sum(
        1
        for event in all_events
        if attrs(event).get("corrupted") is True or err_kind(event) in BAD_TEXT_KINDS
    )
    entrypoints: dict[str, int] = {}
    for event in all_events:
        entry = str(event.get("entrypoint", "unknown"))
        entrypoints[entry] = entrypoints.get(entry, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "request_count": len(request_ids),
        "failed_count": len(failed_events),
        "corrupted_count": corrupted_count,
        "bad_text_count": bad_text_count,
        "oom_prevented_count": oom_prevented_count,
        "silent_oom_count": silent_oom_count,
        "latency_p50_p95_p99": {
            "duration_us": {
                "p50": percentile(durations, 0.50),
                "p95": percentile(durations, 0.95),
                "p99": percentile(durations, 0.99),
                "sample_count": len(durations),
            }
        },
        "memory_high_water_bytes": {
            "max": max(memory_high_by_backend_scope.values(), default=0),
            "by_backend_scope": memory_high_by_backend_scope,
        },
        "resource_leak_count": resource_leak_count,
        "top_slow_phases": [
            {
                "event_id": event.get("event_id"),
                "request_id": event.get("request_id"),
                "entrypoint": event.get("entrypoint"),
                "backend": event.get("backend"),
                "phase": event.get("phase"),
                "duration_us": event.get("duration_us"),
            }
            for event in slow_events
        ],
        "first_failure_event": first_failure,
        "replay_commands": replay_commands,
        "entrypoints": entrypoints,
        "profile_paths": [str(path) for path in profile_paths],
    }


def load_and_validate_profiles(profile_paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    events_by_path: dict[str, list[dict[str, Any]]] = {}
    for path in profile_paths:
        events = read_jsonl(path)
        validate_gate_semantics(path, events)
        events_by_path[str(path)] = events
    return events_by_path


def fixture_files(root: Path, kind: str) -> list[Path]:
    files = sorted((root / kind).glob("*.jsonl"))
    if not files:
        raise GateError(f"{root / kind} has no .jsonl fixtures")
    return files


def run_fixture_selftest(root: Path) -> dict[str, Any]:
    pass_files = fixture_files(root, "pass")
    fail_files = fixture_files(root, "fail")
    pass_results = []
    for path in pass_files:
        events = load_and_validate_profiles([path])
        pass_results.append(summarize_events([path], events))
    fail_results = []
    for path in fail_files:
        try:
            events = load_and_validate_profiles([path])
            summarize_events([path], events)
        except (GateError, ValidationError) as exc:
            fail_results.append({"path": str(path), "error": str(exc)})
            continue
        raise GateError(f"{path} unexpectedly passed")
    return {
        "pass_fixture_count": len(pass_results),
        "fail_fixture_count": len(fail_results),
        "pass_fixtures": [str(path) for path in pass_files],
        "fail_fixtures": fail_results,
    }


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def build_manifest(
    out: Path,
    args: argparse.Namespace,
    started_at: int,
    summary: dict[str, Any],
    fixture_summary: dict[str, Any],
) -> dict[str, Any]:
    ended_at = int(time.time())
    dirty_files = git_value(["status", "--short"]).splitlines()
    return {
        "schema_version": SCHEMA_VERSION,
        "goal": GOAL,
        "phase": "observability_profile",
        "status": "pass",
        "started_at_unix": started_at,
        "ended_at_unix": ended_at,
        "duration_sec": ended_at - started_at,
        "repo_root": str(REPO_ROOT),
        "git_sha": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "git_dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "command": " ".join(sys.argv),
        "artifact_dir": str(out),
        "pass_line": f"{PASS_LINE}: {out}",
        "inputs": {
            "fixtures": str(args.fixtures),
            "profile_jsonl": [str(path) for path in args.profile_jsonl],
            "vertical_slice": str(args.vertical_slice) if args.vertical_slice else None,
            "actual_smoke": str(args.actual_smoke) if args.actual_smoke else None,
        },
        "outputs": {
            "summary": str(out / "observability_profile_summary.json"),
            "fixture_report": str(out / "fixture_report.json"),
        },
        "validation_summary": {
            "request_count": summary["request_count"],
            "failed_count": summary["failed_count"],
            "bad_text_count": summary["bad_text_count"],
            "silent_oom_count": summary["silent_oom_count"],
            "resource_leak_count": summary["resource_leak_count"],
            "fixture_pass_count": fixture_summary["pass_fixture_count"],
            "fixture_fail_count": fixture_summary["fail_fixture_count"],
        },
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    started_at = int(time.time())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    fixture_summary = run_fixture_selftest(args.fixtures)
    if len(args.profile_jsonl) < 2:
        raise GateError("provide at least two --profile-jsonl paths covering run and serve artifacts")
    events_by_path = load_and_validate_profiles(args.profile_jsonl)
    summary = summarize_events(args.profile_jsonl, events_by_path)
    if "run" not in summary["entrypoints"] or "serve" not in summary["entrypoints"]:
        raise GateError("observability profile gate requires both run and serve entrypoint artifacts")
    if summary["bad_text_count"] != 0:
        raise GateError("bad_text_count must be 0 for passing product profiles")
    if summary["silent_oom_count"] != 0:
        raise GateError("silent_oom_count must be 0 for passing product profiles")
    if summary["resource_leak_count"] != 0:
        raise GateError("resource_leak_count must be 0 for passing product profiles")
    summary["gate"] = "observability_profile"
    summary["fixture_summary"] = fixture_summary
    summary["vertical_slice_artifact"] = str(args.vertical_slice) if args.vertical_slice else None
    summary["actual_smoke_artifact"] = str(args.actual_smoke) if args.actual_smoke else None
    write_json(out / "observability_profile_summary.json", summary)
    write_json(out / "fixture_report.json", fixture_summary)
    manifest = build_manifest(out, args, started_at, summary, fixture_summary)
    write_json(out / "gate.manifest.json", manifest)
    (out / "pass_line.txt").write_text(f"{PASS_LINE}: {out}\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--profile-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--vertical-slice", type=Path)
    parser.add_argument("--actual-smoke", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            fixture_summary = run_fixture_selftest(args.fixtures)
            if args.out:
                write_json(args.out / "fixture_report.json", fixture_summary)
            print(SELFTEST_PASS_LINE)
            return 0
        if args.out is None:
            raise GateError("--out is required unless --self-test is used")
        run_gate(args)
        print(f"{PASS_LINE}: {args.out}")
        return 0
    except (GateError, ValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
