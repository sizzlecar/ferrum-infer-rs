#!/usr/bin/env python3
"""Run and validate Ferrum's L0 observability vertical slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from request_replay_bundle_gate import BundleError, validate_bundle_root

REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_LINE = "OBSERVABILITY VERTICAL SLICE PASS"
SELFTEST_PASS_LINE = "OBSERVABILITY VERTICAL SLICE SELFTEST PASS"
SCHEMA_VERSION = 1


class GateError(RuntimeError):
    pass


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


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise GateError(f"{path} must be a JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_checked(cmd: list[str], *, cwd: Path, log_path: Path, timeout: int) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def ferrum_command(args: argparse.Namespace, subcommand: str, out_dir: Path) -> list[str]:
    cli_args = [
        subcommand,
        "synthetic/no-weight",
        "--observability-vertical-slice-out",
        str(out_dir),
    ]
    if args.ferrum_bin:
        return [str(args.ferrum_bin), *cli_args]
    return ["cargo", "run", "--quiet", "-p", "ferrum-cli", "--", *cli_args]


def read_profile_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateError(f"{path}:{line_no} invalid JSONL event: {exc}") from exc
        if not isinstance(event, dict):
            raise GateError(f"{path}:{line_no} must be a JSON object")
        events.append(event)
    if not events:
        raise GateError(f"{path} must contain profile events")
    return events


def validate_entrypoint_artifact(root: Path, entrypoint: str) -> dict[str, Any]:
    profile = root / "profile.jsonl"
    request_dump_root = root / "request_dump"
    request_dump = request_dump_root / "request.json"
    replay_command = root / "replay_command.txt"
    summary_path = root / "observability_profile_summary.json"
    required = [profile, request_dump, replay_command, summary_path]
    for path in required:
        if not path.is_file():
            raise GateError(f"missing {entrypoint} artifact: {path}")

    events = read_profile_events(profile)
    schema_versions = {event.get("schema_version") for event in events}
    entrypoints = {event.get("entrypoint") for event in events}
    if schema_versions != {SCHEMA_VERSION}:
        raise GateError(f"{profile} has schema versions {sorted(schema_versions)}")
    if entrypoints != {entrypoint}:
        raise GateError(f"{profile} has entrypoints {sorted(entrypoints)}")
    if not any("replay" in event for event in events):
        raise GateError(f"{profile} must contain a replay reference")

    request = load_json(request_dump)
    summary = load_json(summary_path)
    if request.get("schema_version") != SCHEMA_VERSION:
        raise GateError(f"{request_dump}.schema_version must be {SCHEMA_VERSION}")
    if request.get("entrypoint") != entrypoint:
        raise GateError(f"{request_dump}.entrypoint must be {entrypoint}")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise GateError(f"{summary_path}.schema_version must be {SCHEMA_VERSION}")
    if summary.get("entrypoint") != entrypoint:
        raise GateError(f"{summary_path}.entrypoint must be {entrypoint}")
    if not summary.get("l0_only"):
        raise GateError(f"{summary_path}.l0_only must be true")
    if not replay_command.read_text(encoding="utf-8").strip():
        raise GateError(f"{replay_command} must be non-empty")
    replay_refs = [
        event["replay"]
        for event in events
        if isinstance(event.get("replay"), dict)
    ]
    for index, replay in enumerate(replay_refs):
        bundle_dir = replay.get("bundle_dir")
        if not isinstance(bundle_dir, str) or not bundle_dir.strip():
            raise GateError(f"{profile} replay[{index}].bundle_dir must be non-empty")
    try:
        replay_bundles = validate_bundle_root(request_dump_root)
    except BundleError as exc:
        raise GateError(f"{entrypoint} replay bundle validation failed: {exc}") from exc
    if not replay_bundles:
        raise GateError(f"{request_dump_root} must contain at least one replay bundle")

    return {
        "entrypoint": entrypoint,
        "schema_version": SCHEMA_VERSION,
        "profile_jsonl": str(profile),
        "event_count": len(events),
        "request_dump": str(request_dump),
        "request_dump_dir": str(request_dump_root),
        "replay_command": replay_command.read_text(encoding="utf-8").strip(),
        "replay_bundles": replay_bundles,
        "summary": str(summary_path),
    }


def run_analyzer(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    analyzer_out = out / "analyzer"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/release/analyze_ferrum_profile.py"),
        "--profile-jsonl",
        str(out / "run/profile.jsonl"),
        "--profile-jsonl",
        str(out / "serve/profile.jsonl"),
        "--out",
        str(analyzer_out),
    ]
    log = run_checked(
        cmd,
        cwd=REPO_ROOT,
        log_path=out / "logs/analyzer.json",
        timeout=args.timeout,
    )
    if "FERRUM PROFILE ANALYZER PASS" not in log["stdout"]:
        raise GateError("profile analyzer did not print its PASS line")
    return {"out": str(analyzer_out), "stdout": log["stdout"]}


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    run_checked(
        ferrum_command(args, "run", out / "run"),
        cwd=REPO_ROOT,
        log_path=out / "logs/run_cli.json",
        timeout=args.timeout,
    )
    run_checked(
        ferrum_command(args, "serve", out / "serve"),
        cwd=REPO_ROOT,
        log_path=out / "logs/serve_cli.json",
        timeout=args.timeout,
    )
    run_summary = validate_entrypoint_artifact(out / "run", "run")
    serve_summary = validate_entrypoint_artifact(out / "serve", "serve")
    analyzer = run_analyzer(args, out)
    if run_summary["schema_version"] != serve_summary["schema_version"]:
        raise GateError("run and serve schema versions differ")
    dirty_files = git_value(["status", "--short"], default="").splitlines()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "gate": "observability_vertical_slice",
        "status": "pass",
        "l0_only": True,
        "same_schema_version": True,
        "entrypoints": {
            "run": run_summary,
            "serve": serve_summary,
        },
        "analyzer": analyzer,
    }
    write_json(out / "observability_profile_summary.json", summary)
    write_json(
        out / "observability_vertical_slice_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "artifact_dir": str(out),
            "pass_line": f"{PASS_LINE}: {out}",
            "git_sha": git_value(["rev-parse", "HEAD"]),
            "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_dirty": bool(dirty_files),
            "dirty_files": dirty_files,
            "summary": str(out / "observability_profile_summary.json"),
        },
    )
    return summary


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-observability-gate-selftest-") as tmp:
        root = Path(tmp)
        for entrypoint in ("run", "serve"):
            entry_dir = root / entrypoint
            (entry_dir / "request_dump").mkdir(parents=True)
            (entry_dir / "profile.jsonl").write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "ts_unix_nanos": 1782950400000000000,
                        "event_id": f"evt-{entrypoint}",
                        "request_id": f"req-{entrypoint}",
                        "correlation_id": f"corr-{entrypoint}",
                        "entrypoint": entrypoint,
                        "backend": "synthetic",
                        "runtime_preset_hash": "sha256:6c3b8d2c431c47cf612289b02a8c631c894f34f532508fc58841e572aedaa7bc",
                        "phase": "request_complete",
                        "event_kind": "instant",
                        "timestamp": "2026-07-02T00:00:00Z",
                        "status": "diagnostic_only",
                        "model": "synthetic/no-weight",
                        "replay": {
                            "command": f"ferrum {entrypoint} synthetic/no-weight",
                            "bundle_dir": str(entry_dir / "request_dump"),
                        },
                        "shape": {"batch_size": 1},
                        "attributes": {},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            write_json(
                entry_dir / "request_dump/request.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "entrypoint": entrypoint,
                    "request_id": f"req-{entrypoint}",
                    "l0_only": True,
                    "model": "synthetic/no-weight",
                    "backend": "synthetic",
                    "sanitized": True,
                },
            )
            (entry_dir / "replay_command.txt").write_text(
                f"ferrum {entrypoint} synthetic/no-weight\n",
                encoding="utf-8",
            )
            write_selftest_replay_bundle(entry_dir, entrypoint)
            write_json(
                entry_dir / "observability_profile_summary.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "entrypoint": entrypoint,
                    "l0_only": True,
                    "status": "pass",
                },
            )
            validate_entrypoint_artifact(entry_dir, entrypoint)
    print(SELFTEST_PASS_LINE)


def write_selftest_replay_bundle(entry_dir: Path, entrypoint: str) -> None:
    request_id = f"req-{entrypoint}"
    request_dump = entry_dir / "request_dump"
    bundle = request_dump / request_id
    output_text = "synthetic ok"
    replay_argv = [
        "cargo",
        "run",
        "-p",
        "ferrum-cli",
        "--",
        entrypoint,
        "synthetic/no-weight",
        "--profile-detail",
        "basic",
        "--profile-jsonl",
        str(entry_dir / "profile.jsonl"),
        "--request-dump-dir",
        str(request_dump),
    ]
    request = {
        "schema_version": SCHEMA_VERSION,
        "entrypoint": entrypoint,
        "request_id": request_id,
        "model": "synthetic/no-weight",
        "backend": "synthetic",
        "l0_only": True,
        "sanitized": True,
    }
    files: dict[str, Any] = {
        "request.json": request,
        "prompt_token_ids.json": {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "token_ids": [101, 202],
            "token_count": 2,
            "sanitized": True,
        },
        "sampling_params.json": {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "sampling_params": {"max_tokens": 4, "temperature": 0.0},
        },
        "runtime_effective_config.json": {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "entrypoint": entrypoint,
            "request_dump_dir": str(request_dump),
            "sanitized": True,
        },
        "backend_selection.json": {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "backend": "synthetic",
            "model": "synthetic/no-weight",
        },
        "output_token_ids.json": {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "token_ids": [909, 808],
            "token_count": 2,
            "finish_reason": "stop",
        },
        "bad_output_scan.json": {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "bad_output": False,
            "bad_text_count": 0,
            "reasons": [],
            "first_bad_text_span": None,
            "failure_kind": None,
            "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        },
        "replay.command.json": {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "entrypoint": entrypoint,
            "command": " ".join(replay_argv),
            "argv": replay_argv,
            "bundle_dir": str(bundle),
            "sanitized": True,
        },
    }
    for name, data in files.items():
        write_json(bundle / name, data)
    (bundle / "output_text.txt").write_text(output_text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ferrum-bin", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and args.out is None:
        parser.error("--out is required unless --self-test is set")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run_gate(args)
    print(f"{PASS_LINE}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
