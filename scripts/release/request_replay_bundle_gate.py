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
HTTP_BODY_FLAGS = {"--data-binary", "--data", "-d"}
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


def validate_serve_replay_command(bundle: Path, request: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any] | None:
    http = request.get("http")
    is_serve_replay = request.get("entrypoint") == "serve" and (
        replay.get("requires_running_server") is True
        or (request.get("backend") == "actual" and isinstance(http, dict))
    )
    if not is_serve_replay:
        return None
    if replay.get("requires_running_server") is not True:
        raise BundleError(f"{bundle / 'replay.command.json'}.requires_running_server must be true for serve request replay")
    replay_body_path = bundle / "replay_body.json"
    if not replay_body_path.is_file():
        raise BundleError(f"{replay_body_path} is required for serve request replay")
    replay_body = read_json(replay_body_path)
    problems = scan_secrets(replay_body, str(replay_body_path))
    if problems:
        raise BundleError("; ".join(problems))

    if not isinstance(http, dict):
        raise BundleError(f"{bundle / 'request.json'}.http is required for serve request replay")
    if http.get("path") != "/v1/chat/completions":
        raise BundleError(f"{bundle / 'request.json'}.http.path must be /v1/chat/completions")
    if http.get("method") != "POST":
        raise BundleError(f"{bundle / 'request.json'}.http.method must be POST")

    argv = replay.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise BundleError(f"{bundle / 'replay.command.json'}.argv must be a string array")
    if not any(item.endswith("/v1/chat/completions") for item in argv):
        raise BundleError(f"{bundle / 'replay.command.json'}.argv must target /v1/chat/completions")
    body_arg = None
    for index, item in enumerate(argv[:-1]):
        if item in HTTP_BODY_FLAGS:
            body_arg = argv[index + 1]
            break
    if body_arg is None:
        raise BundleError(f"{bundle / 'replay.command.json'}.argv must include a body flag")
    if not body_arg.startswith("@") or Path(body_arg[1:]).name != "replay_body.json":
        raise BundleError(f"{bundle / 'replay.command.json'}.argv body flag must reference @replay_body.json")
    return {
        "replay_body": str(replay_body_path),
        "http_path": http.get("path"),
        "body_arg": body_arg,
    }


def validate_engine_replay_command(bundle: Path, replay: dict[str, Any]) -> dict[str, Any] | None:
    engine_replay = replay.get("engine_replay")
    if engine_replay is None:
        return None
    if not isinstance(engine_replay, dict):
        raise BundleError(f"{bundle / 'replay.command.json'}.engine_replay must be an object")
    if engine_replay.get("requires_http_server") is not False:
        raise BundleError(
            f"{bundle / 'replay.command.json'}.engine_replay.requires_http_server must be false"
        )
    argv = engine_replay.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise BundleError(f"{bundle / 'replay.command.json'}.engine_replay.argv must be a non-empty string array")
    if "replay-bundle" not in argv:
        raise BundleError(f"{bundle / 'replay.command.json'}.engine_replay.argv must invoke replay-bundle")
    command = engine_replay.get("command")
    if not isinstance(command, str) or not command.strip():
        raise BundleError(f"{bundle / 'replay.command.json'}.engine_replay.command must be non-empty")
    return {
        "command": command,
        "argv": argv,
        "mode": engine_replay.get("mode"),
    }


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

    actual_run_prompt_required = (
        request.get("entrypoint") == "run"
        and request.get("backend") == "actual"
        and request.get("actual_model_smoke") is True
    )
    prompt_tokens = validate_token_dump(
        bundle / "prompt_token_ids.json",
        allow_unavailable=not actual_run_prompt_required,
    )
    output_tokens = validate_token_dump(bundle / "output_token_ids.json", allow_unavailable=False)
    sampling = read_json(bundle / "sampling_params.json")
    runtime_config = read_json(bundle / "runtime_effective_config.json")
    backend = read_json(bundle / "backend_selection.json")
    replay = read_json(bundle / "replay.command.json")
    bad_scan = validate_bad_output_scan(bundle / "bad_output_scan.json")
    serve_replay = validate_serve_replay_command(bundle, request, replay)
    engine_replay = validate_engine_replay_command(bundle, replay)
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
        "serve_replay": serve_replay,
        "engine_replay": engine_replay,
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


def rewrite_engine_replay_argv(argv: list[str], bundle: Path, replay_out: Path) -> list[str]:
    rewritten = list(argv)
    try:
        replay_index = rewritten.index("replay-bundle")
    except ValueError as exc:
        raise BundleError("engine replay argv must include replay-bundle") from exc
    if replay_index + 1 >= len(rewritten):
        raise BundleError("engine replay argv missing bundle path")
    rewritten[replay_index + 1] = str(bundle)
    index = 0
    saw_out = False
    while index < len(rewritten):
        if rewritten[index] == "--out":
            if index + 1 >= len(rewritten):
                raise BundleError("engine replay argv --out missing path value")
            rewritten[index + 1] = str(replay_out / "engine_replay")
            saw_out = True
            index += 2
            continue
        index += 1
    if not saw_out:
        rewritten.extend(["--out", str(replay_out / "engine_replay")])
    if "--json" not in rewritten:
        rewritten.append("--json")
    return rewritten


def rewrite_live_server_argv(argv: list[str], live_server_base_url: str, bundle: Path) -> list[str]:
    rewritten = list(argv)
    for index, item in enumerate(rewritten):
        for base in ("http://127.0.0.1:8000", "http://localhost:8000"):
            if item.startswith(base):
                rewritten[index] = live_server_base_url.rstrip("/") + item[len(base) :]
                break
        if index > 0 and rewritten[index - 1] in HTTP_BODY_FLAGS and item.startswith("@"):
            if Path(item[1:]).name == "replay_body.json":
                rewritten[index] = f"@{bundle / 'replay_body.json'}"
    return rewritten


def parse_sse(body: str) -> dict[str, int]:
    done_count = 0
    content_chunks = 0
    malformed = 0
    for raw in body.splitlines():
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            done_count += 1
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            malformed += 1
            continue
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta") or {}
            if isinstance(delta, dict) and delta.get("content"):
                content_chunks += 1
    return {
        "done_count": done_count,
        "content_chunks": content_chunks,
        "malformed": malformed,
    }


def validate_live_server_response(bundle: Path, stdout: str) -> dict[str, Any]:
    request = read_json(bundle / "request.json")
    stream = bool(request.get("stream"))
    if stream:
        sse = parse_sse(stdout)
        if sse["done_count"] != 1:
            raise BundleError(f"{bundle}: live replay expected one [DONE], got {sse['done_count']}")
        if sse["content_chunks"] < 1:
            raise BundleError(f"{bundle}: live replay stream emitted no content chunks")
        if sse["malformed"] != 0:
            raise BundleError(f"{bundle}: live replay stream malformed SSE count={sse['malformed']}")
        return {"mode": "stream", **sse}
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise BundleError(f"{bundle}: live replay nonstream response is invalid JSON: {exc}") from exc
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BundleError(f"{bundle}: live replay response missing choices")
    message = choices[0].get("message") or {}
    if not isinstance(message, dict) or not str(message.get("content") or "").strip():
        raise BundleError(f"{bundle}: live replay nonstream content is empty")
    return {"mode": "nonstream", "choice_count": len(choices)}


def execute_replay(
    bundle: Path,
    *,
    out: Path,
    timeout: int,
    live_server_base_url: str | None = None,
) -> dict[str, Any]:
    replay = read_json(bundle / "replay.command.json")
    argv = replay.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise BundleError(f"{bundle / 'replay.command.json'}.argv must be a string array")
    engine_replay = validate_engine_replay_command(bundle, replay)
    if engine_replay is not None and replay.get("requires_running_server") is not True:
        replay_out = out / "executed_replays" / safe_path_part(bundle.name)
        replay_out.mkdir(parents=True, exist_ok=True)
        rewritten = rewrite_engine_replay_argv(engine_replay["argv"], bundle, replay_out)
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
            raise BundleError(f"{bundle}: engine replay command failed with exit {proc.returncode}")
        return {
            "source_bundle_dir": str(bundle),
            "status": "executed_engine_replay",
            "artifact_dir": str(replay_out),
        }
    if "synthetic/no-weight" not in argv and replay.get("requires_running_server") is True:
        if live_server_base_url:
            replay_out = out / "executed_replays" / safe_path_part(bundle.name)
            replay_out.mkdir(parents=True, exist_ok=True)
            rewritten = rewrite_live_server_argv(argv, live_server_base_url, bundle)
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
                raise BundleError(f"{bundle}: live replay command failed with exit {proc.returncode}")
            response_summary = validate_live_server_response(bundle, proc.stdout)
            log["response_summary"] = response_summary
            write_json(replay_out / "replay_execution.json", log)
            return {
                "source_bundle_dir": str(bundle),
                "status": "executed_live_server",
                "artifact_dir": str(replay_out),
                "response_summary": response_summary,
            }
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
        "status": "executed_synthetic",
        "artifact_dir": str(replay_out),
        "generated_bundle_count": len(generated),
        "generated_bundles": generated,
    }


def make_bundle(
    root: Path,
    *,
    entrypoint: str = "run",
    serve_replay: bool = False,
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
        "entrypoint": entrypoint,
        "request_id": "req-fixture",
        "model": "synthetic/no-weight",
        "backend": "actual" if serve_replay else "synthetic",
        "sanitized": True,
        "prompt": "public fixture",
    }
    if serve_replay:
        request["stream"] = False
        request["http"] = {
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": {"content-type": "application/json"},
            "body": {
                "model": "synthetic/no-weight",
                "messages": [{"role": "user", "content": "[redacted]"}],
                "stream": False,
            },
        }
    if secret:
        request["authorization"] = "Bearer sk-thisShouldFail1234567890"
    replay_body = {
        "model": "synthetic/no-weight",
        "messages": [{"role": "user", "content": "[redacted]"}],
        "stream": False,
    }
    replay_argv = ["ferrum", "run", "synthetic/no-weight"]
    replay_command = "ferrum run synthetic/no-weight"
    engine_replay_argv = [
        "cargo",
        "run",
        "-p",
        "ferrum-cli",
        "--",
        "replay-bundle",
        str(root / "req-fixture"),
        "--out",
        str(root / "req-fixture" / "engine_replay"),
        "--json",
    ]
    if serve_replay:
        replay_argv = [
            "curl",
            "-sS",
            "-X",
            "POST",
            "http://127.0.0.1:8000/v1/chat/completions",
            "-H",
            "content-type: application/json",
            "--data-binary",
            f"@{root / 'req-fixture' / 'replay_body.json'}",
        ]
        replay_command = " ".join(replay_argv)
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
            "entrypoint": entrypoint,
            "command": replay_command,
            "argv": replay_argv,
            "requires_running_server": True if serve_replay else None,
            "engine_replay": {
                "mode": "bundle_offline",
                "requires_http_server": False,
                "command": " ".join(engine_replay_argv),
                "argv": engine_replay_argv,
            },
            "sanitized": True,
        },
    }
    for name, data in files.items():
        if name != missing:
            write_json(bundle / name, data)
    if serve_replay:
        write_json(bundle / "replay_body.json", replay_body)
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
        make_bundle(pass_root / "serve-live", entrypoint="serve", serve_replay=True)
        make_bundle(pass_root / "serve-startup", entrypoint="serve")
        serve_startup_request = pass_root / "serve-startup" / "req-fixture" / "request.json"
        startup_request = read_json(serve_startup_request)
        startup_request["backend"] = "actual"
        startup_request["actual_model_smoke"] = True
        startup_request["http"] = None
        write_json(serve_startup_request, startup_request)
        make_bundle(pass_root / "serve-synthetic-http", entrypoint="serve")
        serve_synthetic_request = pass_root / "serve-synthetic-http" / "req-fixture" / "request.json"
        synthetic_request = read_json(serve_synthetic_request)
        synthetic_request["backend"] = "synthetic"
        synthetic_request["http"] = {
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": {"content-type": "application/json"},
            "body": {"model": "synthetic/no-weight", "messages": []},
        }
        write_json(serve_synthetic_request, synthetic_request)
        pass_results = []
        for root in sorted(pass_root.iterdir()):
            pass_results.extend(validate_bundle_root(root))
        rewritten = rewrite_live_server_argv(
            [
                "curl",
                "http://127.0.0.1:8000/v1/chat/completions",
                "--data-binary",
                "@/stale/path/replay_body.json",
            ],
            "http://127.0.0.1:9876",
            pass_root / "serve-live" / "req-fixture",
        )
        if rewritten[1] != "http://127.0.0.1:9876/v1/chat/completions":
            raise BundleError(f"live replay base URL rewrite failed: {rewritten}")
        if rewritten[3] != f"@{pass_root / 'serve-live' / 'req-fixture' / 'replay_body.json'}":
            raise BundleError(f"live replay body path rewrite failed: {rewritten}")
        engine_rewritten = rewrite_engine_replay_argv(
            [
                "cargo",
                "run",
                "-p",
                "ferrum-cli",
                "--",
                "replay-bundle",
                "/stale/bundle",
                "--out",
                "/stale/out",
            ],
            pass_root / "normal" / "req-fixture",
            temp / "engine-replay-out",
        )
        if engine_rewritten[6] != str(pass_root / "normal" / "req-fixture"):
            raise BundleError(f"engine replay bundle path rewrite failed: {engine_rewritten}")
        if engine_rewritten[8] != str(temp / "engine-replay-out" / "engine_replay"):
            raise BundleError(f"engine replay out path rewrite failed: {engine_rewritten}")
        if "--json" not in engine_rewritten:
            raise BundleError(f"engine replay argv rewrite must force --json: {engine_rewritten}")

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

        root = temp / "fail" / "actual-run-missing-prompt-ids"
        make_bundle(root)
        bundle = root / "req-fixture"
        request = read_json(bundle / "request.json")
        request["backend"] = "actual"
        request["actual_model_smoke"] = True
        write_json(bundle / "request.json", request)
        prompt_tokens = read_json(bundle / "prompt_token_ids.json")
        prompt_tokens["token_ids"] = None
        prompt_tokens["token_count"] = None
        prompt_tokens["unavailable_reason"] = "regression fixture"
        write_json(bundle / "prompt_token_ids.json", prompt_tokens)
        try:
            validate_bundle_root(root)
        except BundleError as exc:
            fail_results.append({"case": "actual-run-missing-prompt-ids", "error": str(exc)})
        else:
            raise BundleError("selftest fail case actual-run-missing-prompt-ids unexpectedly passed")

        root = temp / "fail" / "serve-missing-replay-body"
        make_bundle(root, entrypoint="serve", serve_replay=True)
        (root / "req-fixture" / "replay_body.json").unlink()
        try:
            validate_bundle_root(root)
        except BundleError as exc:
            fail_results.append({"case": "serve-missing-replay-body", "error": str(exc)})
        else:
            raise BundleError("selftest fail case serve-missing-replay-body unexpectedly passed")

        root = temp / "fail" / "serve-missing-body-flag"
        make_bundle(root, entrypoint="serve", serve_replay=True)
        replay_path = root / "req-fixture" / "replay.command.json"
        replay = read_json(replay_path)
        replay["argv"] = [
            "curl",
            "-sS",
            "http://127.0.0.1:8000/v1/chat/completions",
        ]
        write_json(replay_path, replay)
        try:
            validate_bundle_root(root)
        except BundleError as exc:
            fail_results.append({"case": "serve-missing-body-flag", "error": str(exc)})
        else:
            raise BundleError("selftest fail case serve-missing-body-flag unexpectedly passed")
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
    parser.add_argument("--live-server-base-url")
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
                    execute_replay(
                        bundle,
                        out=args.out,
                        timeout=args.timeout,
                        live_server_base_url=args.live_server_base_url,
                    )
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
