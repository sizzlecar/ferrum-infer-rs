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
    return scenarios


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
    return {
        "status": "pass",
        "fixture_kind": fixture_kind,
        "bundle_root": str(bundle_root),
        "bundle_count": len(bundles),
        "failure_kinds": sorted(
            {str(bundle.get("failure_kind")) for bundle in bundles if bundle.get("failure_kind")}
        ),
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


def actual_smoke_result(actual_smoke: Path) -> dict[str, Any]:
    summary_path = actual_smoke / "product_observability_l1_smoke_summary.json"
    summary = read_json(summary_path)
    require(summary.get("status") == "pass", f"{summary_path} is not pass")
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
    return {
        "status": "pass",
        "actual_smoke": str(actual_smoke),
        "entrypoints": profile_summary["entrypoints"],
        "replay_bundle_count": len(replay_bundle_summary),
        "live_replay_execution_count": live_summary.get("replay_execution_count"),
    }


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
            result.update({"name": name, "type": scenario["type"], "status": "pass"})
        except Exception as exc:
            result = {
                "name": name,
                "type": scenario["type"],
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
        "failed": len(failures),
        "scenarios": results,
        "actual_smoke": actual_smoke,
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


def run_selftest() -> None:
    temp = Path(tempfile.mkdtemp(prefix="ferrum-product-backend-sentinel-"))
    try:
        out = temp / "out"
        summary = run_gate(
            argparse.Namespace(
                manifest=DEFAULT_MANIFEST,
                out=out,
                actual_smoke=None,
            )
        )
        if summary.get("status") != "pass":
            raise AssertionError(summary)
        if summary.get("scenario_count") != REQUIRED_STAGE2_FIXTURES:
            raise AssertionError(summary)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--actual-smoke", type=Path)
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
