#!/usr/bin/env python3
"""Validate request replay bundles used by product observability artifacts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PASS_LINE = "REQUEST REPLAY BUNDLE PASS"
SELFTEST_PASS_LINE = "REQUEST REPLAY BUNDLE SELFTEST PASS"
SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = {
    "request.json",
    "prompt_token_ids.json",
    "sampling_params.json",
    "runtime_effective_config.json",
    "backend_selection.json",
    "output_token_ids.json",
    "output_text.txt",
    "bad_output_scan.json",
    "replay.command.json",
}
SECRET_KEY_RE = re.compile(
    r"(authorization|cookie|secret|api[_-]?key|password|access[_-]?token|refresh[_-]?token|id[_-]?token)",
    re.I,
)
SECRET_VALUE_RE = re.compile(r"(sk-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._-]{16,})")
REPLAY_PATH_FLAGS = {
    "--profile-jsonl": "profile.jsonl",
    "--memory-profile-jsonl": "memory_profile.jsonl",
    "--scheduler-trace-jsonl": "scheduler_trace.jsonl",
}
BAD_OUTPUT_FAILURE_KINDS = {"bad_output"}
RESOURCE_FAILURE_KINDS = {
    "oom",
    "prevented_oom",
    "admission",
    "admission_reject",
    "oom_admission",
}
PANIC_FAILURE_KINDS = {"panic", "error", "panic_error"}
KNOWN_FAILURE_KINDS = BAD_OUTPUT_FAILURE_KINDS | RESOURCE_FAILURE_KINDS | PANIC_FAILURE_KINDS


class BundleError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BundleError(f"{path}: expected JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def non_empty_string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BundleError(f"{context}.{key} must be a non-empty string")
    return value


def scan_secrets(value: Any, context: str) -> list[str]:
    problems: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_context = f"{context}.{key_text}"
            if SECRET_KEY_RE.search(key_text) and child not in (None, "", "[redacted]", "redacted"):
                problems.append(f"{child_context} looks secret-bearing and is not redacted")
            problems.extend(scan_secrets(child, child_context))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            problems.extend(scan_secrets(child, f"{context}[{index}]"))
    elif isinstance(value, str) and SECRET_VALUE_RE.search(value):
        problems.append(f"{context} contains token-looking value")
    return problems


def candidate_bundle_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise BundleError(f"{root}: does not exist")
    if not root.is_dir():
        raise BundleError(f"{root}: must be a directory")
    if (root / "request.json").is_file() and REQUIRED_FILES <= {path.name for path in root.iterdir() if path.is_file()}:
        return [root]
    bundles = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "request.json").is_file()
    ]
    if not bundles:
        raise BundleError(f"{root}: no replay bundle directories found")
    return bundles


def validate_token_dump(path: Path, *, allow_unavailable: bool) -> dict[str, Any]:
    data = read_json(path)
    token_ids = data.get("token_ids")
    token_count = data.get("token_count")
    unavailable = data.get("unavailable_reason")
    if token_ids is None:
        if not allow_unavailable:
            raise BundleError(f"{path}.token_ids must not be null")
        if not isinstance(unavailable, str) or not unavailable.strip():
            raise BundleError(f"{path}.unavailable_reason is required when token_ids is null")
    elif not isinstance(token_ids, list) or not all(isinstance(item, int) and item >= 0 for item in token_ids):
        raise BundleError(f"{path}.token_ids must be an array of non-negative integers or null")
    if token_count is not None and (not isinstance(token_count, int) or token_count < 0):
        raise BundleError(f"{path}.token_count must be a non-negative integer or null")
    if isinstance(token_ids, list) and token_count != len(token_ids):
        raise BundleError(f"{path}.token_count must match token_ids length")
    return data


def validate_bad_output_scan(path: Path) -> dict[str, Any]:
    data = read_json(path)
    bad_output = data.get("bad_output")
    if not isinstance(bad_output, bool):
        raise BundleError(f"{path}.bad_output must be boolean")
    reasons = data.get("reasons")
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise BundleError(f"{path}.reasons must be a string array")
    if bad_output:
        if not reasons:
            raise BundleError(f"{path}.reasons must be non-empty for bad_output=true")
        if not isinstance(data.get("first_bad_text_span"), dict):
            raise BundleError(f"{path}.first_bad_text_span is required for bad_output=true")
    if not isinstance(data.get("output_sha256"), str) or len(data["output_sha256"]) != 64:
        raise BundleError(f"{path}.output_sha256 must be a sha256 hex digest")
    failure_kind = data.get("failure_kind")
    if failure_kind is not None and (
        not isinstance(failure_kind, str) or failure_kind not in KNOWN_FAILURE_KINDS
    ):
        raise BundleError(f"{path}.failure_kind is unknown: {failure_kind!r}")
    return data


def validate_failure_diagnostics(bundle: Path, request_id: str, bad_scan: dict[str, Any]) -> dict[str, Any] | None:
    failure_kind = bad_scan.get("failure_kind")
    if failure_kind is None or failure_kind in BAD_OUTPUT_FAILURE_KINDS:
        return None
    path = bundle / "failure_diagnostics.json"
    if not path.is_file():
        raise BundleError(f"{path} is required for failure_kind={failure_kind}")
    data = read_json(path)
    if data.get("request_id") != request_id:
        raise BundleError(f"{path}.request_id mismatch")
    if data.get("failure_kind") != failure_kind:
        raise BundleError(f"{path}.failure_kind must match bad_output_scan.failure_kind")
    problems = scan_secrets(data, str(path))
    if problems:
        raise BundleError("; ".join(problems))

    if failure_kind in RESOURCE_FAILURE_KINDS:
        capacity = data.get("capacity")
        if not isinstance(capacity, dict):
            raise BundleError(f"{path}.capacity is required for resource failure")
        for key in ("resource_kind", "needed", "available", "capacity", "reason"):
            if key not in capacity:
                raise BundleError(f"{path}.capacity.{key} is required")
        for key in ("needed", "available", "capacity"):
            if not isinstance(capacity.get(key), int):
                raise BundleError(f"{path}.capacity.{key} must be integer")
        if not isinstance(capacity.get("reason"), str) or not capacity["reason"].strip():
            raise BundleError(f"{path}.capacity.reason must be non-empty")
        if not isinstance(data.get("nearest_resource_event"), dict):
            raise BundleError(f"{path}.nearest_resource_event is required for resource failure")
        memory = data.get("nearest_memory_snapshot")
        if not isinstance(memory, dict):
            raise BundleError(f"{path}.nearest_memory_snapshot is required for resource failure")
        for key in ("current_bytes", "high_water_bytes"):
            if not isinstance(memory.get(key), int) or memory[key] < 0:
                raise BundleError(f"{path}.nearest_memory_snapshot.{key} must be non-negative integer")
    elif failure_kind in PANIC_FAILURE_KINDS:
        first_failure = data.get("first_failure_event")
        if not isinstance(first_failure, dict):
            raise BundleError(f"{path}.first_failure_event is required for panic/error failure")
        for key in ("phase", "error_kind"):
            if not isinstance(first_failure.get(key), str) or not first_failure[key].strip():
                raise BundleError(f"{path}.first_failure_event.{key} must be non-empty")
        if not (
            isinstance(data.get("backtrace_excerpt"), str) and data["backtrace_excerpt"].strip()
        ) and not (isinstance(data.get("log_excerpt"), str) and data["log_excerpt"].strip()):
            raise BundleError(f"{path} requires backtrace_excerpt or log_excerpt")
        if not (
            isinstance(data.get("nearest_request_id"), str) and data["nearest_request_id"].strip()
        ) and not (isinstance(data.get("global_failure_id"), str) and data["global_failure_id"].strip()):
            raise BundleError(f"{path} requires nearest_request_id or global_failure_id")
    else:
        raise BundleError(f"{path}: unsupported failure_kind={failure_kind}")
    return data


def validate_bundle_dir(bundle: Path) -> dict[str, Any]:
    missing = sorted(REQUIRED_FILES - {path.name for path in bundle.iterdir() if path.is_file()})
    if missing:
        raise BundleError(f"{bundle}: missing required files: {missing}")

    request = read_json(bundle / "request.json")
    request_id = non_empty_string(request, "request_id", str(bundle / "request.json"))
    if request.get("schema_version") != SCHEMA_VERSION:
        raise BundleError(f"{bundle / 'request.json'}.schema_version must be {SCHEMA_VERSION}")
    if request.get("sanitized") is not True:
        raise BundleError(f"{bundle / 'request.json'}.sanitized must be true")
    secret_problems = scan_secrets(request, str(bundle / "request.json"))
    if secret_problems:
        raise BundleError("; ".join(secret_problems))

    prompt_tokens = validate_token_dump(bundle / "prompt_token_ids.json", allow_unavailable=True)
    output_tokens = validate_token_dump(bundle / "output_token_ids.json", allow_unavailable=False)
    sampling = read_json(bundle / "sampling_params.json")
    runtime_config = read_json(bundle / "runtime_effective_config.json")
    backend = read_json(bundle / "backend_selection.json")
    replay = read_json(bundle / "replay.command.json")
    bad_scan = validate_bad_output_scan(bundle / "bad_output_scan.json")
    failure_diagnostics = validate_failure_diagnostics(bundle, request_id, bad_scan)

    for label, data in [
        ("prompt_token_ids", prompt_tokens),
        ("output_token_ids", output_tokens),
        ("sampling_params", sampling),
        ("runtime_effective_config", runtime_config),
        ("backend_selection", backend),
        ("replay.command", replay),
        ("bad_output_scan", bad_scan),
    ]:
        if data.get("request_id") != request_id:
            raise BundleError(f"{bundle}/{label}.json request_id mismatch")
        problems = scan_secrets(data, f"{bundle}/{label}.json")
        if problems:
            raise BundleError("; ".join(problems))

    argv = replay.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise BundleError(f"{bundle / 'replay.command.json'}.argv must be a non-empty string array")
    non_empty_string(replay, "command", str(bundle / "replay.command.json"))
    output_text = (bundle / "output_text.txt").read_text(encoding="utf-8", errors="replace")
    if SECRET_VALUE_RE.search(output_text):
        raise BundleError(f"{bundle / 'output_text.txt'} contains token-looking value")

    return {
        "bundle_dir": str(bundle),
        "request_id": request_id,
        "entrypoint": request.get("entrypoint"),
        "backend": backend.get("backend"),
        "bad_output": bad_scan["bad_output"],
        "failure_kind": bad_scan.get("failure_kind"),
        "failure_diagnostics": str(bundle / "failure_diagnostics.json")
        if failure_diagnostics is not None
        else None,
        "prompt_token_count": prompt_tokens.get("token_count"),
        "output_token_count": output_tokens.get("token_count"),
    }


def validate_bundle_root(root: Path) -> list[dict[str, Any]]:
    return [validate_bundle_dir(bundle) for bundle in candidate_bundle_dirs(root)]


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "bundle"


def rewrite_replay_argv(argv: list[str], replay_out: Path) -> list[str]:
    if "synthetic/no-weight" not in argv:
        raise BundleError("executable replay argv must target synthetic/no-weight")
    rewritten = list(argv)
    saw_request_dump = False
    index = 0
    while index < len(rewritten):
        flag = rewritten[index]
        if flag in REPLAY_PATH_FLAGS:
            if index + 1 >= len(rewritten):
                raise BundleError(f"{flag} missing path value in replay argv")
            rewritten[index + 1] = str(replay_out / REPLAY_PATH_FLAGS[flag])
            index += 2
            continue
        if flag == "--request-dump-dir":
            if index + 1 >= len(rewritten):
                raise BundleError("--request-dump-dir missing path value in replay argv")
            rewritten[index + 1] = str(replay_out / "request_dump")
            saw_request_dump = True
            index += 2
            continue
        index += 1
    if not saw_request_dump:
        raise BundleError("executable replay argv must include --request-dump-dir")
    return rewritten


def execute_replay(bundle: Path, *, out: Path, timeout: int) -> dict[str, Any]:
    replay = read_json(bundle / "replay.command.json")
    argv = replay.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise BundleError(f"{bundle / 'replay.command.json'}.argv must be a string array")
    if "synthetic/no-weight" not in argv and replay.get("requires_running_server") is True:
        return {
            "source_bundle_dir": str(bundle),
            "status": "skipped_requires_running_server",
            "reason": "offline replay execution only runs synthetic/no-weight commands",
        }
    replay_out = out / "executed_replays" / safe_path_part(bundle.name)
    replay_out.mkdir(parents=True, exist_ok=True)
    rewritten = rewrite_replay_argv(argv, replay_out)
    proc = subprocess.run(
        rewritten,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    log = {
        "bundle_dir": str(bundle),
        "cmd": rewritten,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    write_json(replay_out / "replay_execution.json", log)
    if proc.returncode != 0:
        raise BundleError(f"{bundle}: replay command failed with exit {proc.returncode}")
    generated_root = replay_out / "request_dump"
    generated = validate_bundle_root(generated_root)
    return {
        "source_bundle_dir": str(bundle),
        "artifact_dir": str(replay_out),
        "generated_bundle_count": len(generated),
        "generated_bundles": generated,
    }


def make_bundle(
    root: Path,
    *,
    bad_output: bool = False,
    failure_kind: str | None = None,
    secret: bool = False,
    missing: str | None = None,
    omit_failure_diagnostics: bool = False,
) -> None:
    bundle = root / "req-fixture"
    bundle.mkdir(parents=True)
    request = {
        "schema_version": 1,
        "entrypoint": "run",
        "request_id": "req-fixture",
        "model": "synthetic/no-weight",
        "backend": "synthetic",
        "sanitized": True,
        "prompt": "public fixture",
    }
    if secret:
        request["authorization"] = "Bearer sk-thisShouldFail1234567890"
    files: dict[str, Any] = {
        "request.json": request,
        "prompt_token_ids.json": {
            "schema_version": 1,
            "request_id": "req-fixture",
            "token_ids": [1, 2, 3],
            "token_count": 3,
            "sanitized": True,
        },
        "sampling_params.json": {
            "schema_version": 1,
            "request_id": "req-fixture",
            "sampling_params": {"max_tokens": 4, "temperature": 0.0},
        },
        "runtime_effective_config.json": {
            "schema_version": 1,
            "request_id": "req-fixture",
            "entrypoint": "run",
            "sanitized": True,
        },
        "backend_selection.json": {
            "schema_version": 1,
            "request_id": "req-fixture",
            "backend": "synthetic",
            "model": "synthetic/no-weight",
        },
        "output_token_ids.json": {
            "schema_version": 1,
            "request_id": "req-fixture",
            "token_ids": [4, 5],
            "token_count": 2,
            "finish_reason": "stop",
        },
        "bad_output_scan.json": {
            "schema_version": 1,
            "request_id": "req-fixture",
            "bad_output": bad_output,
            "bad_text_count": 1 if bad_output else 0,
            "reasons": ["reserved_token"] if bad_output else [],
            "first_bad_text_span": {"byte_start": 0, "byte_end": 5, "reason": "reserved_token"}
            if bad_output
            else None,
            "failure_kind": failure_kind,
            "output_sha256": "0" * 64,
        },
        "replay.command.json": {
            "schema_version": 1,
            "request_id": "req-fixture",
            "entrypoint": "run",
            "command": "ferrum run synthetic/no-weight",
            "argv": ["ferrum", "run", "synthetic/no-weight"],
            "sanitized": True,
        },
    }
    for name, data in files.items():
        if name != missing:
            write_json(bundle / name, data)
    if failure_kind and failure_kind not in BAD_OUTPUT_FAILURE_KINDS and not omit_failure_diagnostics:
        if failure_kind in RESOURCE_FAILURE_KINDS:
            diagnostics = {
                "schema_version": 1,
                "request_id": "req-fixture",
                "failure_kind": failure_kind,
                "capacity": {
                    "resource_kind": "kv_block",
                    "needed": 4,
                    "available": 1,
                    "capacity": 8,
                    "reason": "insufficient_kv_capacity",
                },
                "nearest_resource_event": {
                    "phase": "admission",
                    "action": "reject",
                    "resource_kind": "kv_block",
                    "needed": 4,
                    "available": 1,
                },
                "nearest_memory_snapshot": {
                    "scope": "device",
                    "current_bytes": 23_000_000_000,
                    "high_water_bytes": 23_500_000_000,
                    "available_bytes": 512_000_000,
                },
            }
        elif failure_kind in PANIC_FAILURE_KINDS:
            diagnostics = {
                "schema_version": 1,
                "request_id": "req-fixture",
                "failure_kind": failure_kind,
                "first_failure_event": {
                    "phase": "decode",
                    "error_kind": "panic",
                    "message": "synthetic panic fixture",
                },
                "nearest_request_id": "req-fixture",
                "log_excerpt": "thread panicked at synthetic fixture",
            }
        else:
            diagnostics = {
                "schema_version": 1,
                "request_id": "req-fixture",
                "failure_kind": failure_kind,
            }
        write_json(bundle / "failure_diagnostics.json", diagnostics)
    if missing != "output_text.txt":
        (bundle / "output_text.txt").write_text("<unk>\n" if bad_output else "ok\n", encoding="utf-8")


def run_selftest() -> dict[str, Any]:
    temp = Path(tempfile.mkdtemp(prefix="ferrum-replay-bundle-selftest-"))
    try:
        pass_root = temp / "pass"
        make_bundle(pass_root / "normal")
        make_bundle(pass_root / "bad-output", bad_output=True, failure_kind="bad_output")
        make_bundle(pass_root / "oom-admission", failure_kind="oom_admission")
        make_bundle(pass_root / "panic-error", failure_kind="panic_error")
        pass_results = []
        for root in sorted(pass_root.iterdir()):
            pass_results.extend(validate_bundle_root(root))

        fail_cases = {
            "secret": {"secret": True},
            "missing-output": {"missing": "output_token_ids.json"},
            "bad-scan": {"bad_output": True},
            "missing-failure-diagnostics": {
                "failure_kind": "oom_admission",
                "omit_failure_diagnostics": True,
            },
            "unknown-failure-kind": {"failure_kind": "unknown"},
        }
        fail_results = []
        for name, kwargs in fail_cases.items():
            root = temp / "fail" / name
            make_bundle(root, **kwargs)
            if name == "bad-scan":
                scan = root / "req-fixture" / "bad_output_scan.json"
                data = read_json(scan)
                data["first_bad_text_span"] = None
                write_json(scan, data)
            try:
                validate_bundle_root(root)
            except BundleError as exc:
                fail_results.append({"case": name, "error": str(exc)})
            else:
                raise BundleError(f"selftest fail case {name} unexpectedly passed")
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "pass_count": len(pass_results),
            "fail_count": len(fail_results),
            "pass_results": pass_results,
            "fail_results": fail_results,
        }
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bundle-dir", action="append", type=Path, default=[])
    parser.add_argument("--execute-replay", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            summary = run_selftest()
            if args.out:
                write_json(args.out / "request_replay_bundle_selftest.json", summary)
            print(SELFTEST_PASS_LINE)
            return 0
        if args.out is None:
            raise BundleError("--out is required unless --self-test is used")
        if not args.bundle_dir:
            raise BundleError("provide at least one --bundle-dir")
        results = []
        bundle_paths = []
        for bundle_dir in args.bundle_dir:
            roots = candidate_bundle_dirs(bundle_dir)
            bundle_paths.extend(roots)
            results.extend(validate_bundle_dir(bundle) for bundle in roots)
        replay_executions = []
        if args.execute_replay:
            for bundle in bundle_paths:
                replay_executions.append(
                    execute_replay(bundle, out=args.out, timeout=args.timeout)
                )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "bundle_count": len(results),
            "bundles": results,
            "execute_replay": bool(args.execute_replay),
            "replay_execution_count": sum(
                1
                for item in replay_executions
                if item.get("status") != "skipped_requires_running_server"
            ),
            "replay_execution_skipped_count": sum(
                1
                for item in replay_executions
                if item.get("status") == "skipped_requires_running_server"
            ),
            "replay_executions": replay_executions,
        }
        write_json(args.out / "request_replay_bundle_summary.json", summary)
        write_json(
            args.out / "gate.manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "phase": "request_replay_bundle",
                "status": "pass",
                "artifact_dir": str(args.out),
                "pass_line": f"{PASS_LINE}: {args.out}",
                "bundle_dirs": [str(path) for path in args.bundle_dir],
                "validation_summary": summary,
            },
        )
        print(f"{PASS_LINE}: {args.out}")
        return 0
    except BundleError as exc:
        print(f"REQUEST REPLAY BUNDLE FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
