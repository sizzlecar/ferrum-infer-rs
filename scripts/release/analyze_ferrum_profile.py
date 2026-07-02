#!/usr/bin/env python3
"""Validate Ferrum observability profile and native-op manifest fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_FIXTURES = REPO_ROOT / "scripts/release/fixtures/observability_profile"
DEFAULT_NATIVE_FIXTURES = REPO_ROOT / "scripts/release/fixtures/native_operator"
PASS_LINE = "FERRUM PROFILE ANALYZER PASS"
SELFTEST_PASS_LINE = "FERRUM PROFILE ANALYZER SELFTEST PASS"


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
    for key in ("before_bytes", "after_bytes"):
        if key not in memory or not isinstance(memory[key], int) or memory[key] < 0:
            raise ValidationError(f"{context}.memory.{key} must be a non-negative integer")


def validate_profile_event(event: Any, context: str) -> None:
    if not isinstance(event, dict):
        raise ValidationError(f"{context} must be a JSON object")
    if event.get("schema_version") != 1:
        raise ValidationError(f"{context}.schema_version must be 1")
    require_non_empty_string(event, "event_id", context)
    require_non_empty_string(event, "request_id", context)
    require_non_empty_string(event, "backend", context)
    require_non_empty_string(event, "phase", context)
    require_non_empty_string(event, "timestamp", context)
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
    if "replay" in event:
        replay = event["replay"]
        if not isinstance(replay, dict):
            raise ValidationError(f"{context}.replay must be an object")
        require_non_empty_string(replay, "command", f"{context}.replay")


def validate_profile_jsonl(path: Path) -> dict[str, Any]:
    events = 0
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
    if events == 0:
        raise ValidationError(f"{path} must contain at least one profile event")
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


def run_fixture_selftest(profile_root: Path, native_root: Path) -> dict[str, Any]:
    profile_pass, profile_fail = fixture_files(profile_root, ".jsonl")
    native_pass, native_fail = fixture_files(native_root, ".json")
    return {
        "profile_pass": [expect_pass(path, validate_profile_jsonl) for path in profile_pass],
        "profile_fail": [expect_fail(path, validate_profile_jsonl) for path in profile_fail],
        "native_pass": [expect_pass(path, validate_native_manifest) for path in native_pass],
        "native_fail": [expect_fail(path, validate_native_manifest) for path in native_fail],
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
