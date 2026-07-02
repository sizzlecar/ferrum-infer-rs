#!/usr/bin/env python3
"""L1 actual-model smoke for product observability wiring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_LINE = "PRODUCT OBSERVABILITY L1 SMOKE PASS"
MODEL_DEFAULT = "Qwen/Qwen3-0.6B"
SCHEMA_VERSION = 1


class SmokeError(RuntimeError):
    pass


class BackendMismatchError(SmokeError):
    def __init__(self, requested: str, effective: str | None) -> None:
        self.requested = requested
        self.effective = effective
        super().__init__(
            f"requested backend {requested!r} but effective backend is {effective!r}; "
            "refusing silent backend fallback evidence"
        )


class BackendUnavailableError(SmokeError):
    def __init__(self, message: str) -> None:
        self.requested = requested_backend_from_message(message)
        super().__init__(message)


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


def binary_sha256(args: argparse.Namespace) -> str | None:
    if args.ferrum_bin is None:
        return None
    try:
        digest = hashlib.sha256()
        with args.ferrum_bin.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SmokeError(f"{path}:{line_no} invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise SmokeError(f"{path}:{line_no} must be an object")
        events.append(data)
    if not events:
        raise SmokeError(f"{path} must contain at least one event")
    return events


def ferrum_base_cmd(args: argparse.Namespace) -> list[str]:
    if args.ferrum_bin:
        return [str(args.ferrum_bin)]
    return ["cargo", "run", "--quiet", "-p", "ferrum-cli", "--"]


def obs_flags(root: Path) -> list[str]:
    return [
        "--profile-jsonl",
        str(root / "profile.jsonl"),
        "--profile-detail",
        "basic",
        "--memory-profile-jsonl",
        str(root / "memory_profile.jsonl"),
        "--scheduler-trace-jsonl",
        str(root / "scheduler_trace.jsonl"),
        "--request-dump-dir",
        str(root / "request_dump"),
        "--profile-sample-rate",
        "1.0",
    ]


def profile_detail_from_flags() -> str:
    return "basic"


def run_checked(cmd: list[str], *, cwd: Path, timeout: int, log_path: Path, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env={**os.environ, "NO_COLOR": "1"},
        check=False,
    )
    write_json(
        log_path,
        {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        },
    )
    if proc.returncode != 0:
        backend_error = backend_unavailable_message(f"{proc.stdout}\n{proc.stderr}")
        if backend_error:
            raise BackendUnavailableError(backend_error)
        raise SmokeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}")
    return proc


def assistant_events(stdout: str) -> list[dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("event") == "assistant":
            events.append(data)
    return events


def run_actual(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    root = out / "run"
    cmd = [
        *ferrum_base_cmd(args),
        "run",
        "--backend",
        args.backend,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        "0",
        "--output-format",
        "jsonl",
        "--prompt",
        "Reply with the word OK.",
        *obs_flags(root),
        args.model,
    ]
    proc = run_checked(cmd, cwd=REPO_ROOT, timeout=args.timeout, log_path=out / "logs/run.json")
    assistants = assistant_events(proc.stdout)
    if not assistants:
        raise SmokeError("ferrum run did not emit an assistant JSONL event")
    content = str(assistants[-1].get("content") or "").strip()
    if not content:
        raise SmokeError("ferrum run assistant content is empty")
    return {
        "status": "pass",
        "assistant_event_count": len(assistants),
        "content_preview": content[:200],
    }


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    try:
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def http_json(url: str, payload: dict[str, Any], timeout: int) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")


def wait_health(
    base_url: str,
    timeout: int,
    proc: subprocess.Popen[str] | None = None,
    server_log: Path | None = None,
) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = ""
            if server_log is not None and server_log.is_file():
                tail = "\n".join(server_log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
            backend_error = backend_unavailable_message(tail)
            if backend_error:
                raise BackendUnavailableError(backend_error)
            raise SmokeError(f"server exited before health check rc={proc.returncode}: {tail}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                if resp.status == 200:
                    return json.loads(body)
        except Exception as exc:  # noqa: BLE001 - saved for diagnostics.
            last_error = str(exc)
        time.sleep(1)
    raise SmokeError(f"server did not become healthy: {last_error}")


def parse_sse(body: str) -> dict[str, Any]:
    done_count = 0
    content_chunks = 0
    usage_chunks = 0
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
        if data.get("usage"):
            usage_chunks += 1
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta") or {}
            if isinstance(delta, dict) and delta.get("content"):
                content_chunks += 1
    return {
        "done_count": done_count,
        "content_chunks": content_chunks,
        "usage_chunks": usage_chunks,
        "malformed": malformed,
    }


def effective_backend_from_health(health: dict[str, Any]) -> str | None:
    auto_config = health.get("auto_config")
    if isinstance(auto_config, dict) and isinstance(auto_config.get("backend"), str):
        return auto_config["backend"]
    return None


def validate_requested_backend(args: argparse.Namespace, effective_backend: str | None) -> None:
    requested = args.backend
    if requested == "auto":
        return
    if effective_backend != requested:
        raise BackendMismatchError(requested, effective_backend)


def backend_unavailable_message(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if "requested backend" in line and "not built" in line:
            return line
    return None


def requested_backend_from_message(message: str) -> str | None:
    for backend in ("metal", "cuda", "cpu"):
        if f"requested backend '{backend}'" in message:
            return backend
    return None


def live_server_replay_bundle_dirs(request_dump_root: Path) -> list[Path]:
    bundles: list[Path] = []
    for path in sorted(request_dump_root.iterdir()):
        replay_path = path / "replay.command.json"
        if not replay_path.is_file():
            continue
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        if replay.get("requires_running_server") is True:
            bundles.append(path)
    return bundles


def run_live_server_replay_bundle_gate(
    base_url: str,
    request_dump_root: Path,
    out: Path,
    timeout: int,
) -> dict[str, Any]:
    bundles = live_server_replay_bundle_dirs(request_dump_root)
    if not bundles:
        raise SmokeError(f"no live-server replay bundles found under {request_dump_root}")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/release/request_replay_bundle_gate.py"),
        "--out",
        str(out / "serve_live_replay_bundle"),
        "--execute-replay",
        "--live-server-base-url",
        base_url,
        "--timeout",
        str(timeout),
    ]
    for bundle in bundles:
        cmd.extend(["--bundle-dir", str(bundle)])
    proc = run_checked(
        cmd,
        cwd=REPO_ROOT,
        timeout=timeout,
        log_path=out / "logs/serve_live_replay_bundle.json",
    )
    if "REQUEST REPLAY BUNDLE PASS" not in proc.stdout:
        raise SmokeError("live-server request replay bundle gate did not print PASS")
    summary_path = out / "serve_live_replay_bundle/request_replay_bundle_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("replay_execution_count") != len(bundles):
        raise SmokeError(
            "live-server replay execution count mismatch: "
            f"{summary.get('replay_execution_count')} != {len(bundles)}"
        )
    return {
        "out": str(out / "serve_live_replay_bundle"),
        "bundles": [str(bundle) for bundle in bundles],
        "bundle_count": len(bundles),
        "replay_execution_count": summary.get("replay_execution_count"),
    }


def serve_actual(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    root = out / "serve"
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    server_log = out / "logs/server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        *ferrum_base_cmd(args),
        "serve",
        "--backend",
        args.backend,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *obs_flags(root),
        args.model,
    ]
    with server_log.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "NO_COLOR": "1"},
        )
        try:
            health = wait_health(base_url, args.timeout, proc, server_log)
            effective_backend = effective_backend_from_health(health)
            validate_requested_backend(args, effective_backend)
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
            }
            status, body = http_json(f"{base_url}/v1/chat/completions", payload, args.timeout)
            (out / "serve_nonstream.json").write_text(body, encoding="utf-8")
            if status != 200:
                raise SmokeError(f"serve nonstream HTTP {status}: {body[:500]}")
            data = json.loads(body)
            content = str(data["choices"][0]["message"].get("content") or "").strip()
            if not content:
                raise SmokeError("serve nonstream content is empty")

            stream_payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
            stream_status, stream_body = http_json(
                f"{base_url}/v1/chat/completions",
                stream_payload,
                args.timeout,
            )
            (out / "serve_stream.sse").write_text(stream_body, encoding="utf-8")
            if stream_status != 200:
                raise SmokeError(f"serve stream HTTP {stream_status}: {stream_body[:500]}")
            sse = parse_sse(stream_body)
            if sse["done_count"] != 1:
                raise SmokeError(f"serve stream expected one [DONE], got {sse['done_count']}")
            if sse["content_chunks"] < 1:
                raise SmokeError("serve stream emitted no content")
            if sse["malformed"] != 0:
                raise SmokeError(f"serve stream malformed SSE JSON count={sse['malformed']}")
            live_replay = run_live_server_replay_bundle_gate(
                base_url,
                root / "request_dump",
                out,
                args.timeout,
            )
            return {
                "status": "pass",
                "effective_backend": effective_backend,
                "health": health,
                "nonstream_content_preview": content[:200],
                "stream": sse,
                "live_replay": live_replay,
            }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)


def validate_profile_group(root: Path, entrypoint: str) -> dict[str, Any]:
    paths = {
        "profile": root / "profile.jsonl",
        "memory": root / "memory_profile.jsonl",
        "scheduler": root / "scheduler_trace.jsonl",
    }
    result: dict[str, Any] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise SmokeError(f"missing {entrypoint} {label} profile: {path}")
        events = read_jsonl(path)
        entrypoints = {event.get("entrypoint") for event in events}
        actual = {
            (event.get("attributes") or {}).get("actual_model_smoke")
            for event in events
            if isinstance(event.get("attributes"), dict)
        }
        if entrypoints != {entrypoint}:
            raise SmokeError(f"{path} entrypoints are {entrypoints}")
        if actual != {True}:
            raise SmokeError(f"{path} actual_model_smoke values are {actual}")
        if {event.get("schema_version") for event in events} != {SCHEMA_VERSION}:
            raise SmokeError(f"{path} schema_version mismatch")
        if label == "memory":
            measurements = {
                (event.get("attributes") or {}).get("memory_measurement")
                for event in events
                if isinstance(event.get("attributes"), dict)
            }
            if measurements != {"process_rss"}:
                raise SmokeError(f"{path} memory_measurement values are {measurements}")
            for event in events:
                memory = event.get("memory")
                if not isinstance(memory, dict):
                    raise SmokeError(f"{path} memory event missing memory object")
                for key in ("current_bytes", "high_water_bytes"):
                    if not isinstance(memory.get(key), int) or memory[key] <= 0:
                        raise SmokeError(f"{path} memory.{key} must be a positive integer")
        if label == "scheduler":
            sources = {
                (event.get("attributes") or {}).get("resource_trace_source")
                for event in events
                if isinstance(event.get("attributes"), dict)
            }
            if "engine" not in sources:
                raise SmokeError(f"{path} must contain engine runtime resource trace events")
            resource_kinds = {
                (event.get("resource") or {}).get("resource_kind")
                for event in events
                if isinstance(event.get("resource"), dict)
            }
            missing = {"request_slot", "kv_block"} - resource_kinds
            if missing:
                raise SmokeError(f"{path} missing runtime resource kinds: {sorted(missing)}")
        result[label] = {"path": str(path), "event_count": len(events)}
    request_dump = root / "request_dump/request.json"
    replay = root / "request_dump/replay_command.txt"
    if not request_dump.is_file() or not replay.is_file():
        raise SmokeError(f"missing request dump/replay for {entrypoint}")
    request = json.loads(request_dump.read_text(encoding="utf-8"))
    if request.get("actual_model_smoke") is not True:
        raise SmokeError(f"{request_dump} must mark actual_model_smoke=true")
    result["request_dump"] = str(request_dump)
    result["replay_command"] = replay.read_text(encoding="utf-8").strip()
    return result


def run_analyzer(out: Path, timeout: int) -> dict[str, Any]:
    profiles = [
        out / "run/profile.jsonl",
        out / "run/memory_profile.jsonl",
        out / "run/scheduler_trace.jsonl",
        out / "serve/profile.jsonl",
        out / "serve/memory_profile.jsonl",
        out / "serve/scheduler_trace.jsonl",
    ]
    cmd = [sys.executable, str(REPO_ROOT / "scripts/release/analyze_ferrum_profile.py")]
    for profile in profiles:
        cmd.extend(["--profile-jsonl", str(profile)])
    cmd.extend(["--out", str(out / "analyzer")])
    proc = run_checked(cmd, cwd=REPO_ROOT, timeout=timeout, log_path=out / "logs/analyzer.json")
    if "FERRUM PROFILE ANALYZER PASS" not in proc.stdout:
        raise SmokeError("analyzer did not print PASS")
    return {"profiles": [str(profile) for profile in profiles], "out": str(out / "analyzer")}


def run_resource_invariant(out: Path, timeout: int) -> dict[str, Any]:
    traces = [
        out / "run/scheduler_trace.jsonl",
        out / "serve/scheduler_trace.jsonl",
    ]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/release/resource_invariant_gate.py"),
        "--out",
        str(out / "resource_invariant"),
    ]
    for trace in traces:
        cmd.extend(["--trace-jsonl", str(trace)])
    proc = run_checked(
        cmd,
        cwd=REPO_ROOT,
        timeout=timeout,
        log_path=out / "logs/resource_invariant.json",
    )
    if "RESOURCE INVARIANT GATE PASS" not in proc.stdout:
        raise SmokeError("resource invariant gate did not print PASS")
    return {"traces": [str(trace) for trace in traces], "out": str(out / "resource_invariant")}


def run_replay_bundle_gate(out: Path, timeout: int) -> dict[str, Any]:
    bundles = [
        out / "run/request_dump",
        out / "serve/request_dump",
    ]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/release/request_replay_bundle_gate.py"),
        "--out",
        str(out / "request_replay_bundle"),
        "--execute-replay",
        "--timeout",
        str(timeout),
    ]
    for bundle in bundles:
        cmd.extend(["--bundle-dir", str(bundle)])
    proc = run_checked(
        cmd,
        cwd=REPO_ROOT,
        timeout=timeout,
        log_path=out / "logs/request_replay_bundle.json",
    )
    if "REQUEST REPLAY BUNDLE PASS" not in proc.stdout:
        raise SmokeError("request replay bundle gate did not print PASS")
    return {"bundles": [str(bundle) for bundle in bundles], "out": str(out / "request_replay_bundle")}


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dirty_files = git_value(["status", "--short"], default="").splitlines()
    git_sha = git_value(["rev-parse", "HEAD"])
    git_branch = git_value(["rev-parse", "--abbrev-ref", "HEAD"])
    pass_line = f"{PASS_LINE}: {out}"
    command = sys.argv
    ferrum_cmd = ferrum_base_cmd(args)
    run_result = run_actual(args, out)
    serve_result = serve_actual(args, out)
    effective_backend = serve_result.get("effective_backend")
    run_profiles = validate_profile_group(out / "run", "run")
    serve_profiles = validate_profile_group(out / "serve", "serve")
    analyzer = run_analyzer(out, args.timeout)
    resource_invariant = run_resource_invariant(out, args.timeout)
    replay_bundle = run_replay_bundle_gate(out, args.timeout)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "gate": "product_observability_l1_smoke",
        "goal": "release-regression-hardening-2026-06-28",
        "artifact_dir": str(out),
        "pass_line": pass_line,
        "git_sha": git_sha,
        "git_branch": git_branch,
        "git_dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "command": command,
        "ferrum_command": ferrum_cmd,
        "ferrum_binary_sha256": binary_sha256(args),
        "model": args.model,
        "requested_backend": args.backend,
        "backend": effective_backend or args.backend,
        "effective_backend": effective_backend,
        "profile_detail": profile_detail_from_flags(),
        "actual_model_smoke": True,
        "entrypoints": {
            "run": {"product": run_result, "profiles": run_profiles},
            "serve": {"product": serve_result, "profiles": serve_profiles},
        },
        "analyzer": analyzer,
        "resource_invariant": resource_invariant,
        "request_replay_bundle": replay_bundle,
    }
    write_json(out / "product_observability_l1_smoke_summary.json", summary)
    write_json(
        out / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "goal": "release-regression-hardening-2026-06-28",
            "phase": "product_observability_l1_smoke",
            "status": "pass",
            "repo_root": str(REPO_ROOT),
            "git_sha": git_sha,
            "git_branch": git_branch,
            "git_dirty": bool(dirty_files),
            "dirty_files": dirty_files,
            "command": command,
            "artifact_dir": str(out),
            "pass_line": pass_line,
            "model": args.model,
            "requested_backend": args.backend,
            "backend": effective_backend or args.backend,
            "effective_backend": effective_backend,
            "profile_detail": profile_detail_from_flags(),
            "ferrum_command": ferrum_cmd,
            "ferrum_binary_sha256": summary["ferrum_binary_sha256"],
            "summary": str(out / "product_observability_l1_smoke_summary.json"),
            "outputs": {"summary": str(out / "product_observability_l1_smoke_summary.json")},
            "profile_paths": analyzer["profiles"],
            "resource_invariant": resource_invariant,
            "request_replay_bundle": replay_bundle,
        },
    )
    return summary


def failure_kind(exc: Exception) -> str:
    if isinstance(exc, BackendMismatchError):
        return "backend_mismatch"
    if isinstance(exc, BackendUnavailableError):
        return "backend_unavailable"
    message = str(exc).lower()
    if "replay" in message:
        return "replay_failure"
    if "stream" in message or "[done]" in message or "sse" in message:
        return "stream_failure"
    if "resource" in message or "invariant" in message:
        return "resource_invariant_failure"
    if "profile" in message or "analyzer" in message:
        return "profile_failure"
    if "server" in message or "health" in message:
        return "serve_failure"
    if "run" in message:
        return "run_failure"
    return "unknown"


def write_failure_artifacts(args: argparse.Namespace, exc: SmokeError) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dirty_files = git_value(["status", "--short"], default="").splitlines()
    classification: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "fail",
        "failure_kind": failure_kind(exc),
        "error": str(exc),
        "requested_backend": exc.requested if isinstance(exc, BackendUnavailableError) else args.backend,
        "effective_backend": exc.effective if isinstance(exc, BackendMismatchError) else None,
        "suspected_domain": "backend_runtime_preset"
        if isinstance(exc, BackendMismatchError)
        else "backend_compilation_or_cli_selection"
        if isinstance(exc, BackendUnavailableError)
        else "product_observability_l1_smoke",
        "next_gate": "backend_runtime_preset_snapshot"
        if isinstance(exc, BackendMismatchError)
        else "build_feature_or_backend_smoke"
        if isinstance(exc, BackendUnavailableError)
        else "inspect_product_observability_l1_artifact",
        "do_not_run": ["l2_representative_backend", "release_full"]
        if isinstance(exc, (BackendMismatchError, BackendUnavailableError))
        else ["release_full"],
    }
    write_json(out / "failures/failure_classification.json", classification)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "fail",
        "gate": "product_observability_l1_smoke",
        "goal": "release-regression-hardening-2026-06-28",
        "artifact_dir": str(out),
        "git_sha": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "command": sys.argv,
        "model": args.model,
        "requested_backend": args.backend,
        "effective_backend": classification["effective_backend"],
        "failure_classification": str(out / "failures/failure_classification.json"),
        "error": str(exc),
    }
    write_json(out / "product_observability_l1_smoke_summary.json", summary)
    write_json(
        out / "gate.manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "goal": "release-regression-hardening-2026-06-28",
            "phase": "product_observability_l1_smoke",
            "status": "fail",
            "repo_root": str(REPO_ROOT),
            "git_sha": summary["git_sha"],
            "git_branch": summary["git_branch"],
            "git_dirty": summary["git_dirty"],
            "dirty_files": dirty_files,
            "command": sys.argv,
            "artifact_dir": str(out),
            "pass_line": None,
            "model": args.model,
            "requested_backend": args.backend,
            "effective_backend": classification["effective_backend"],
            "failure_classification": str(out / "failures/failure_classification.json"),
            "outputs": {"summary": str(out / "product_observability_l1_smoke_summary.json")},
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--ferrum-bin", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_gate(args)
        print(f"{PASS_LINE}: {args.out}")
        return 0
    except SmokeError as exc:
        write_failure_artifacts(args, exc)
        print(f"PRODUCT OBSERVABILITY L1 SMOKE FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
