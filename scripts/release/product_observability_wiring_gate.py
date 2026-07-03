#!/usr/bin/env python3
"""Validate typed product observability wiring for run and serve."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL = "release-regression-hardening-2026-06-28"
PASS_LINE = "PRODUCT OBSERVABILITY WIRING PASS"
SELFTEST_PASS_LINE = "PRODUCT OBSERVABILITY WIRING SELFTEST PASS"
SCHEMA_VERSION = 1
SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTH", "CREDENTIAL", "KEY")
SAFE_ENV_NAMES = {"CI", "CARGO_HOME", "HF_HOME", "HOME", "PATH", "RUSTFLAGS", "RUST_BACKTRACE", "RUST_LOG"}
SAFE_ENV_PREFIXES = ("CARGO_", "FERRUM_", "HF_", "RUST_")


class GateError(RuntimeError):
    pass


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GateError(f"{path} must contain a JSON object")
    return data


def git_value(args: list[str], default: str = "unknown") -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return default
    return proc.stdout.strip() or default


def git_status_short() -> list[str]:
    return [line for line in git_value(["status", "--short"], default="").splitlines() if line.strip()]


def sanitized_env() -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in os.environ.items():
        if any(marker in key.upper() for marker in SECRET_ENV_MARKERS):
            continue
        if key in SAFE_ENV_NAMES or any(key.startswith(prefix) for prefix in SAFE_ENV_PREFIXES):
            safe[key] = value
    return dict(sorted(safe.items()))


def run_checked(cmd: list[str], *, log_path: Path, timeout: int) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    log = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    write_json(log_path, log)
    if proc.returncode != 0:
        raise GateError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}")
    return log


def ferrum_command(args: argparse.Namespace, entrypoint: str, root: Path) -> list[str]:
    cli_args = [
        entrypoint,
        "synthetic/no-weight",
        "--profile-jsonl",
        str(root / "profile.jsonl"),
        "--profile-detail",
        args.profile_detail,
        "--memory-profile-jsonl",
        str(root / "memory_profile.jsonl"),
        "--scheduler-trace-jsonl",
        str(root / "scheduler_trace.jsonl"),
        "--request-dump-dir",
        str(root / "request_dump"),
        "--profile-sample-rate",
        "1.0",
    ]
    if args.ferrum_bin:
        return [str(args.ferrum_bin), *cli_args]
    return ["cargo", "run", "--quiet", "-p", "ferrum-cli", "--", *cli_args]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateError(f"{path}:{line_no} invalid JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise GateError(f"{path}:{line_no} must be a JSON object")
        events.append(event)
    if not events:
        raise GateError(f"{path} must contain at least one event")
    return events


def validate_entrypoint(root: Path, entrypoint: str, detail: str) -> dict[str, Any]:
    profile = root / "profile.jsonl"
    memory = root / "memory_profile.jsonl"
    scheduler = root / "scheduler_trace.jsonl"
    request_dump = root / "request_dump/request.json"
    replay = root / "request_dump/replay_command.txt"
    for path in (profile, memory, scheduler, request_dump, replay):
        if not path.is_file():
            raise GateError(f"missing {entrypoint} artifact: {path}")
    summaries = {}
    for label, path in (
        ("profile", profile),
        ("memory_profile", memory),
        ("scheduler_trace", scheduler),
    ):
        events = read_jsonl(path)
        schema_versions = {event.get("schema_version") for event in events}
        entrypoints = {event.get("entrypoint") for event in events}
        details = {
            (event.get("attributes") or {}).get("profile_detail")
            for event in events
            if isinstance(event.get("attributes"), dict)
        }
        if schema_versions != {SCHEMA_VERSION}:
            raise GateError(f"{path} schema versions are {schema_versions}")
        if entrypoints != {entrypoint}:
            raise GateError(f"{path} entrypoints are {entrypoints}")
        if details != {detail}:
            raise GateError(f"{path} profile_detail values are {details}")
        summaries[label] = {"path": str(path), "event_count": len(events)}
    request = load_json(request_dump)
    if request.get("entrypoint") != entrypoint:
        raise GateError(f"{request_dump}.entrypoint must be {entrypoint}")
    if request.get("profile_detail") != detail:
        raise GateError(f"{request_dump}.profile_detail must be {detail}")
    if not replay.read_text(encoding="utf-8").strip():
        raise GateError(f"{replay} must be non-empty")
    return {
        "entrypoint": entrypoint,
        "schema_version": SCHEMA_VERSION,
        "profile_detail": detail,
        "profile_sample_rate": 1.0,
        "artifacts": summaries,
        "request_dump": str(request_dump),
        "replay_command": replay.read_text(encoding="utf-8").strip(),
    }


def run_analyzer(out: Path, timeout: int) -> dict[str, Any]:
    profile_paths = [
        out / "run/profile.jsonl",
        out / "run/memory_profile.jsonl",
        out / "serve/profile.jsonl",
        out / "serve/memory_profile.jsonl",
    ]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/release/analyze_ferrum_profile.py"),
    ]
    for path in profile_paths:
        cmd.extend(["--profile-jsonl", str(path)])
    cmd.extend(["--out", str(out / "analyzer")])
    log = run_checked(cmd, log_path=out / "logs/analyzer.json", timeout=timeout)
    if "FERRUM PROFILE ANALYZER PASS" not in log["stdout"]:
        raise GateError("profile analyzer did not print PASS")
    return {"out": str(out / "analyzer"), "profiles": [str(path) for path in profile_paths]}


def run_resource_invariant(out: Path, timeout: int) -> dict[str, Any]:
    trace_paths = [
        out / "run/scheduler_trace.jsonl",
        out / "serve/scheduler_trace.jsonl",
    ]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/release/resource_invariant_gate.py"),
        "--out",
        str(out / "resource_invariant"),
    ]
    for path in trace_paths:
        cmd.extend(["--trace-jsonl", str(path)])
    log = run_checked(cmd, log_path=out / "logs/resource_invariant.json", timeout=timeout)
    if "RESOURCE INVARIANT GATE PASS" not in log["stdout"]:
        raise GateError("resource invariant gate did not print PASS")
    return {
        "out": str(out / "resource_invariant"),
        "traces": [str(path) for path in trace_paths],
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    started_at = int(time.time())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "failures").mkdir(exist_ok=True)
    (out / "diagnostics").mkdir(exist_ok=True)
    run_checked(
        ferrum_command(args, "run", out / "run"),
        log_path=out / "logs/run_cli.json",
        timeout=args.timeout,
    )
    run_checked(
        ferrum_command(args, "serve", out / "serve"),
        log_path=out / "logs/serve_cli.json",
        timeout=args.timeout,
    )
    run_summary = validate_entrypoint(out / "run", "run", args.profile_detail)
    serve_summary = validate_entrypoint(out / "serve", "serve", args.profile_detail)
    analyzer = run_analyzer(out, args.timeout)
    resource_invariant = run_resource_invariant(out, args.timeout)
    dirty_files = git_status_short()
    pass_line = f"{PASS_LINE}: {out}"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "goal": GOAL,
        "status": "pass",
        "gate": "product_observability_wiring",
        "l0_only": True,
        "artifact_dir": str(out),
        "pass_line": pass_line,
        "git_sha": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "command": sys.argv,
        "entrypoints": {
            "run": run_summary,
            "serve": serve_summary,
        },
        "analyzer": analyzer,
        "resource_invariant": resource_invariant,
        "l1_actual_smoke": {
            "status": "not_run_in_l0_gate",
            "reason": "WP6 typed wiring slice only; actual model smoke remains required before WP6 completion",
        },
    }
    write_json(out / "product_observability_wiring_summary.json", summary)
    ended_at = int(time.time())
    write_json(
        out / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "goal": GOAL,
            "phase": "product_observability_wiring",
            "status": "pass",
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_sec": ended_at - started_at,
            "repo_root": str(REPO_ROOT),
            "git_sha": summary["git_sha"],
            "git_branch": summary["git_branch"],
            "git_dirty": summary["git_dirty"],
            "dirty_files": dirty_files,
            "command": sys.argv,
            "artifact_dir": str(out),
            "pass_line": pass_line,
            "profile_detail": args.profile_detail,
            "profile_paths": analyzer["profiles"],
            "resource_trace_paths": resource_invariant["traces"],
            "inputs": {
                "ferrum_bin": str(args.ferrum_bin) if args.ferrum_bin else None,
                "profile_detail": args.profile_detail,
                "timeout": args.timeout,
            },
            "outputs": {
                "summary": str(out / "product_observability_wiring_summary.json"),
                "run_profile": str(out / "run/profile.jsonl"),
                "run_memory": str(out / "run/memory_profile.jsonl"),
                "run_scheduler": str(out / "run/scheduler_trace.jsonl"),
                "serve_profile": str(out / "serve/profile.jsonl"),
                "serve_memory": str(out / "serve/memory_profile.jsonl"),
                "serve_scheduler": str(out / "serve/scheduler_trace.jsonl"),
                "resource_invariant": resource_invariant["out"],
            },
            "summary": str(out / "product_observability_wiring_summary.json"),
            "validation_summary": {
                "l0_only": True,
                "entrypoints": ["run", "serve"],
                "profile_detail": args.profile_detail,
                "run_profile_event_count": run_summary["artifacts"]["profile"]["event_count"],
                "serve_profile_event_count": serve_summary["artifacts"]["profile"]["event_count"],
                "analyzer_out": analyzer["out"],
                "resource_invariant_out": resource_invariant["out"],
            },
        },
    )
    (out / "pass_line.txt").write_text(pass_line + "\n", encoding="utf-8")
    (out / "command.log").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (out / "git_status.txt").write_text(
        "\n".join(dirty_files) + ("\n" if dirty_files else ""),
        encoding="utf-8",
    )
    write_json(out / "sanitized_env.json", sanitized_env())
    return summary


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-product-observability-selftest-") as tmp:
        root = Path(tmp)
        entry = root / "run"
        entry.mkdir(parents=True)
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_id": "evt-run",
            "request_id": "req-run",
            "entrypoint": "run",
            "backend": "synthetic",
            "phase": "request_complete",
            "event_kind": "instant",
            "timestamp": "2026-07-02T00:00:00Z",
            "status": "ok",
            "model": "synthetic/no-weight",
            "attributes": {"profile_detail": "basic", "profile_sample_rate": 1.0},
            "replay": {"command": "ferrum run synthetic/no-weight"},
        }
        for name in ("profile.jsonl", "memory_profile.jsonl", "scheduler_trace.jsonl"):
            (entry / name).write_text(json.dumps(event) + "\n", encoding="utf-8")
        (entry / "request_dump").mkdir()
        write_json(
            entry / "request_dump/request.json",
            {"schema_version": 1, "entrypoint": "run", "profile_detail": "basic"},
        )
        (entry / "request_dump/replay_command.txt").write_text(
            "ferrum run synthetic/no-weight\n",
            encoding="utf-8",
        )
        validate_entrypoint(entry, "run", "basic")
    print(SELFTEST_PASS_LINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ferrum-bin", type=Path)
    parser.add_argument("--profile-detail", default="basic", choices=["basic", "debug", "full"])
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and args.out is None:
        parser.error("--out is required unless --self-test is set")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_self_test()
        else:
            run_gate(args)
            print(f"{PASS_LINE}: {args.out}")
        return 0
    except GateError as exc:
        print(f"PRODUCT OBSERVABILITY WIRING FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
