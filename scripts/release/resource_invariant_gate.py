#!/usr/bin/env python3
"""Validate ResourceTraceEvent lifecycle invariants from JSONL traces."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES = REPO_ROOT / "scripts/release/fixtures/resource_invariant"
PASS_LINE = "RESOURCE INVARIANT GATE PASS"
SELFTEST_PASS_LINE = "RESOURCE INVARIANT GATE SELFTEST PASS"

REQUIRED_PASS_SCENARIOS = {
    "kv_allocate_release_success",
    "kv_capacity_reject",
    "recurrent_slot_limit_reject",
    "scheduler_defer_reopen_after_capacity",
    "scheduler_cancel_releases_capacity",
    "engine_prefill_failure_rolls_back",
    "engine_decode_failure_rolls_back",
    "serve_client_disconnect_cleans_up",
    "run_multiturn_resource_balance",
    "mixed_batch_partial_failure",
    "oom_prevented_by_admission",
    "trace_replay_selftest",
}

EXPECTED_FAIL_KINDS = {
    "resource_leak",
    "release_underflow",
    "capacity_overcommit",
    "defer_with_committed_resource",
    "rollback_incomplete",
    "silent_cuda_oom",
    "panic_after_resource_error",
    "transition_mismatch",
}

LIFECYCLE_ACTIONS = {"reserve", "commit", "release", "rollback"}
TERMINAL_EXPLICIT_ACTIONS = {"reject", "defer"}
OOM_ERROR_KINDS = {"cuda_oom", "metal_oom", "oom", "silent_oom"}
PANIC_ERROR_KINDS = {"panic", "panic_error", "panic_after_resource_error"}


class GateError(RuntimeError):
    pass


@dataclass
class ResourceState:
    reserved: int = 0
    committed: int = 0
    released: int = 0
    rolled_back: int = 0
    capacity: int | None = None
    rollback_seen: bool = False
    actions: list[str] = field(default_factory=list)

    @property
    def cleaned(self) -> int:
        return self.released + self.rolled_back

    @property
    def outstanding(self) -> int:
        return self.reserved - self.cleaned


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateError(f"{path}:{line_no} invalid JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise GateError(f"{path}:{line_no} must be a JSON object")
        if "resource" in event:
            resource = event["resource"]
            if not isinstance(resource, dict):
                raise GateError(f"{path}:{line_no}.resource must be an object")
            merged = dict(resource)
            merged.setdefault("scenario", event.get("scenario"))
            events.append(merged)
        else:
            events.append(event)
    if not events:
        raise GateError(f"{path} must contain at least one event")
    return events


def non_empty_string(event: dict[str, Any], key: str, context: str) -> str:
    value = event.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"{context}.{key} must be a non-empty string")
    return value


def optional_int(event: dict[str, Any], key: str, context: str) -> int | None:
    if key not in event or event[key] is None:
        return None
    value = event[key]
    if not isinstance(value, int):
        raise GateError(f"{context}.{key} must be an integer")
    return value


def optional_non_empty_string(event: dict[str, Any], key: str, context: str) -> str | None:
    if key not in event or event[key] is None:
        return None
    value = event[key]
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"{context}.{key} must be a non-empty string when set")
    return value


def normalize_event(event: dict[str, Any], context: str) -> dict[str, Any]:
    action = non_empty_string(event, "action", context)
    if action not in {
        "request_open",
        "reserve",
        "commit",
        "defer",
        "reject",
        "release",
        "rollback",
        "request_close",
        "capacity_snapshot",
    }:
        raise GateError(f"{context}.action is invalid: {action}")
    owner_kind = non_empty_string(event, "owner_kind", context)
    owner_id = non_empty_string(event, "owner_id", context)
    resource_kind = non_empty_string(event, "resource_kind", context)
    amount = optional_int(event, "amount", context)
    before = optional_int(event, "before", context)
    after = optional_int(event, "after", context)
    capacity = optional_int(event, "capacity", context)
    error_kind = optional_non_empty_string(event, "error_kind", context)
    message = optional_non_empty_string(event, "message", context)
    resource_error_kind = optional_non_empty_string(event, "resource_error_kind", context)
    if action in LIFECYCLE_ACTIONS:
        if amount is None or before is None or after is None:
            raise GateError(f"{context}.{action} requires amount, before, and after")
        if amount < 0:
            raise GateError(f"{context}.{action}.amount must be non-negative")
    if action in TERMINAL_EXPLICIT_ACTIONS:
        non_empty_string(event, "reason", context)
    if action == "capacity_snapshot" and capacity is None:
        raise GateError(f"{context}.capacity_snapshot requires capacity")
    return {
        **event,
        "action": action,
        "owner_kind": owner_kind,
        "owner_id": owner_id,
        "resource_kind": resource_kind,
        "amount": amount,
        "before": before,
        "after": after,
        "capacity": capacity,
        "error_kind": error_kind,
        "message": message,
        "resource_error_kind": resource_error_kind,
    }


def state_key(event: dict[str, Any]) -> tuple[str, str, str]:
    return (event["owner_kind"], event["owner_id"], event["resource_kind"])


def owner_key(event: dict[str, Any]) -> tuple[str, str]:
    return (event["owner_kind"], event["owner_id"])


def failure(kind: str, event: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "scenario": event.get("scenario"),
        "owner_kind": event.get("owner_kind"),
        "owner_id": event.get("owner_id"),
        "resource_kind": event.get("resource_kind"),
        "action": event.get("action"),
        "message": message,
    }


def outstanding_by_resource(states: dict[tuple[str, str, str], ResourceState], kind: str) -> int:
    return sum(state.outstanding for (_, _, resource_kind), state in states.items() if resource_kind == kind)


def committed_outstanding(state: ResourceState) -> int:
    return state.committed - state.cleaned


def owner_has_explicit_terminal(
    states: dict[tuple[str, str, str], ResourceState],
    owner: tuple[str, str],
) -> bool:
    return any(
        any(action in TERMINAL_EXPLICIT_ACTIONS for action in state.actions)
        for (owner_kind, owner_id, _resource_kind), state in states.items()
        if (owner_kind, owner_id) == owner
    )


def expected_transition(state: ResourceState, action: str, amount: int) -> tuple[int, int] | None:
    if action == "reserve":
        before = state.outstanding
        return before, before + amount
    if action == "commit":
        before = committed_outstanding(state)
        return before, before + amount
    if action == "release":
        before = committed_outstanding(state)
        return before, before - amount
    if action == "rollback":
        before = state.outstanding
        return before, before - amount
    return None


def check_transition(
    failures: list[dict[str, Any]], event: dict[str, Any], state: ResourceState, amount: int
) -> None:
    expected = expected_transition(state, event["action"], amount)
    if expected is None:
        return
    expected_before, expected_after = expected
    if event["before"] != expected_before or event["after"] != expected_after:
        failures.append(
            failure(
                "transition_mismatch",
                event,
                f"expected before={expected_before} after={expected_after}, "
                f"got before={event['before']} after={event['after']}",
            )
        )


def check_trace(events: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    states: dict[tuple[str, str, str], ResourceState] = defaultdict(ResourceState)
    owner_open: set[tuple[str, str]] = set()
    owner_explicit_terminal: set[tuple[str, str]] = set()
    failures: list[dict[str, Any]] = []
    scenarios: set[str] = set()

    for idx, raw in enumerate(events, start=1):
        event = normalize_event(raw, f"{source}:{idx}")
        scenario = event.get("scenario")
        if isinstance(scenario, str) and scenario:
            scenarios.add(scenario)
        key = state_key(event)
        owner = owner_key(event)
        state = states[key]
        state.actions.append(event["action"])
        if event["capacity"] is not None:
            state.capacity = event["capacity"]

        action = event["action"]
        amount = event["amount"] or 0
        check_transition(failures, event, state, amount)
        if action == "request_open":
            owner_open.add(owner)
        elif action == "reserve":
            state.reserved += amount
            if state.committed > state.reserved:
                failures.append(
                    failure(
                        "committed_exceeds_reserved",
                        event,
                        f"committed={state.committed} reserved={state.reserved}",
                    )
                )
        elif action == "commit":
            state.committed += amount
            if state.committed > state.reserved:
                failures.append(
                    failure(
                        "committed_exceeds_reserved",
                        event,
                        f"committed={state.committed} reserved={state.reserved}",
                    )
                )
        elif action == "release":
            state.released += amount
            if state.released > state.committed:
                failures.append(
                    failure(
                        "release_underflow",
                        event,
                        f"released={state.released} committed={state.committed}",
                    )
                )
        elif action == "rollback":
            state.rollback_seen = True
            state.rolled_back += amount
        elif action == "defer":
            owner_explicit_terminal.add(owner)
            committed_outstanding = sum(
                s.committed - s.released - s.rolled_back
                for (owner_kind, owner_id, _), s in states.items()
                if (owner_kind, owner_id) == owner
            )
            if committed_outstanding > 0:
                failures.append(
                    failure(
                        "defer_with_committed_resource",
                        event,
                        f"defer left committed outstanding={committed_outstanding}",
                    )
                )
        elif action == "reject":
            owner_explicit_terminal.add(owner)
        elif action == "request_close":
            for (owner_kind, owner_id, _resource_kind), close_state in states.items():
                if (owner_kind, owner_id) != owner:
                    continue
                if close_state.outstanding > 0:
                    kind = "rollback_incomplete" if close_state.rollback_seen else "resource_leak"
                    failures.append(
                        failure(
                            kind,
                            event,
                            f"outstanding={close_state.outstanding} reserved={close_state.reserved} "
                            f"released={close_state.released} rolled_back={close_state.rolled_back}",
                        )
                    )
            owner_open.discard(owner)
        elif action == "capacity_snapshot":
            pass

        error_kind = event.get("error_kind")
        if error_kind in OOM_ERROR_KINDS and not owner_has_explicit_terminal(states, owner):
            failures.append(
                failure(
                    "silent_cuda_oom",
                    event,
                    "OOM surfaced without an earlier explicit defer/reject admission event",
                )
            )
        if error_kind in PANIC_ERROR_KINDS and event.get("resource_error_kind"):
            failures.append(
                failure(
                    "panic_after_resource_error",
                    event,
                    f"panic surfaced after resource error {event['resource_error_kind']}",
                )
            )

        capacity = state.capacity
        if capacity is not None:
            outstanding = outstanding_by_resource(states, event["resource_kind"])
            if outstanding > capacity:
                failures.append(
                    failure(
                        "capacity_overcommit",
                        event,
                        f"outstanding={outstanding} capacity={capacity}",
                    )
                )

    for (owner_kind, owner_id, resource_kind), state in states.items():
        if state.outstanding > 0:
            event = {
                "owner_kind": owner_kind,
                "owner_id": owner_id,
                "resource_kind": resource_kind,
                "action": "end_of_trace",
                "scenario": None,
            }
            kind = "rollback_incomplete" if state.rollback_seen else "resource_leak"
            failures.append(
                failure(
                    kind,
                    event,
                    f"trace ended with outstanding={state.outstanding}",
                )
            )

    resource_summary: dict[str, dict[str, int]] = {}
    for (_owner_kind, _owner_id, resource_kind), state in states.items():
        bucket = resource_summary.setdefault(
            resource_kind,
            {
                "capacity": 0,
                "reserved": 0,
                "committed": 0,
                "released": 0,
                "rolled_back": 0,
                "leaked": 0,
            },
        )
        if state.capacity is not None:
            bucket["capacity"] = max(bucket["capacity"], state.capacity)
        bucket["reserved"] += state.reserved
        bucket["committed"] += state.committed
        bucket["released"] += state.released
        bucket["rolled_back"] += state.rolled_back
        bucket["leaked"] += max(state.outstanding, 0)

    failure_counts: dict[str, int] = defaultdict(int)
    for item in failures:
        failure_counts[item["kind"]] += 1

    return {
        "source": source,
        "events": len(events),
        "scenarios": sorted(scenarios),
        "failures": failures,
        "failure_counts": dict(sorted(failure_counts.items())),
        "leaked_resources": sum(bucket["leaked"] for bucket in resource_summary.values()),
        "underflow_count": failure_counts.get("release_underflow", 0),
        "silent_oom_count": failure_counts.get("silent_cuda_oom", 0),
        "panic_count": failure_counts.get("panic_after_resource_error", 0),
        "resource_summary": resource_summary,
        "explicit_terminal_owner_count": len(owner_explicit_terminal),
        "unclosed_owner_count": len(owner_open),
    }


def validate_trace_file(path: Path) -> dict[str, Any]:
    return check_trace(load_jsonl(path), source=str(path))


def fixture_files(root: Path, kind: str) -> list[Path]:
    files = sorted((root / kind).glob("*.jsonl"))
    if not files:
        raise GateError(f"{root / kind} has no .jsonl fixtures")
    return files


def run_fixture_selftest(root: Path) -> dict[str, Any]:
    pass_results = []
    fail_results = []
    seen_scenarios: set[str] = set()
    for path in fixture_files(root, "pass"):
        result = validate_trace_file(path)
        if result["failures"]:
            raise GateError(f"pass fixture failed: {path}: {result['failures']}")
        seen_scenarios.update(result["scenarios"])
        pass_results.append(result)
    missing = sorted(REQUIRED_PASS_SCENARIOS - seen_scenarios)
    if missing:
        raise GateError(f"missing pass scenarios: {', '.join(missing)}")

    for path in fixture_files(root, "fail"):
        result = validate_trace_file(path)
        expected = path.stem
        if expected not in EXPECTED_FAIL_KINDS:
            raise GateError(f"unexpected fail fixture name: {path.name}")
        if result["failure_counts"].get(expected, 0) == 0:
            raise GateError(f"fail fixture {path} did not trigger {expected}: {result['failures']}")
        fail_results.append(result)

    return {
        "pass_fixtures": pass_results,
        "fail_fixtures": fail_results,
        "scenario_count": len(seen_scenarios),
        "required_scenarios": sorted(REQUIRED_PASS_SCENARIOS),
    }


def git_value(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return f"git {' '.join(args)} failed: {proc.stderr.strip()}"
    return proc.stdout.strip()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_combined_trace(out: Path, fixture_root: Path) -> Path:
    trace_path = out / "resource_trace.jsonl"
    lines: list[str] = []
    for path in fixture_files(fixture_root, "pass"):
        lines.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return trace_path


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    fixture_summary = run_fixture_selftest(args.fixtures)
    trace_path = write_combined_trace(out, args.fixtures)
    trace_report = validate_trace_file(trace_path)
    if trace_report["failures"]:
        raise GateError(f"combined pass trace failed: {trace_report['failures']}")
    external_reports = []
    for path in args.trace_jsonl:
        external_report = validate_trace_file(path)
        if external_report["failures"]:
            raise GateError(f"external trace failed: {path}: {external_report['failures']}")
        if external_report["leaked_resources"] != 0:
            raise GateError(f"external trace leaked resources: {path}")
        if external_report["underflow_count"] != 0:
            raise GateError(f"external trace has release underflow: {path}")
        if external_report["unclosed_owner_count"] != 0:
            raise GateError(f"external trace has unclosed owners: {path}")
        external_reports.append(external_report)
    if fixture_summary["scenario_count"] != len(REQUIRED_PASS_SCENARIOS):
        raise GateError("not all required pass scenarios were covered")
    report = {
        "schema_version": 1,
        "status": "pass",
        "trace": trace_report,
        "external_traces": external_reports,
        "fixture_summary": fixture_summary,
        "leaked_resources": trace_report["leaked_resources"],
        "underflow_count": trace_report["underflow_count"],
        "silent_oom_count": trace_report["silent_oom_count"],
        "panic_count": trace_report["panic_count"],
        "resource_summary": trace_report["resource_summary"],
    }
    if report["leaked_resources"] != 0:
        raise GateError("leaked_resources must be 0")
    if report["underflow_count"] != 0:
        raise GateError("underflow_count must be 0")
    if report["silent_oom_count"] != 0:
        raise GateError("silent_oom_count must be 0")
    if report["panic_count"] != 0:
        raise GateError("panic_count must be 0")
    report_path = out / "invariant_report.json"
    write_json(report_path, report)
    dirty_files = git_value(["status", "--short"]).splitlines()
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "artifact_dir": str(out),
        "pass_line": f"{PASS_LINE}: {out}",
        "git_sha": git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "dirty_status": git_value(["status", "--short"]),
        "commands": [
            f"{sys.executable} scripts/release/resource_invariant_gate.py --out {out}",
            f"{sys.executable} scripts/release/resource_invariant_gate.py --self-test",
        ],
        "resource_trace": str(trace_path),
        "external_trace_jsonl": [str(path) for path in args.trace_jsonl],
        "invariant_report": str(report_path),
    }
    write_json(out / "gate.manifest.json", manifest)
    return report


def run_self_test(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-resource-invariant-selftest-") as tmp:
        args = argparse.Namespace(out=Path(tmp), fixtures=root, trace_jsonl=[])
        run_gate(args)
    print(SELFTEST_PASS_LINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--trace-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and args.out is None:
        parser.error("--out is required unless --self-test is set")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_self_test(args.fixtures)
        else:
            run_gate(args)
            print(f"{PASS_LINE}: {args.out}")
        return 0
    except GateError as exc:
        print(f"RESOURCE INVARIANT GATE FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
