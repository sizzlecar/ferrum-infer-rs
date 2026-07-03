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
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from request_replay_bundle_gate import BundleError, validate_bundle_root

REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_LINE = "PRODUCT OBSERVABILITY L1 SMOKE PASS"
SELFTEST_PASS_LINE = "PRODUCT OBSERVABILITY L1 SMOKE SELFTEST PASS"
MODEL_DEFAULT = "Qwen/Qwen3-0.6B"
SCHEMA_VERSION = 1
SYNTHETIC_RUNTIME_PRESET_HASH = "sha256:6c3b8d2c431c47cf612289b02a8c631c894f34f532508fc58841e572aedaa7bc"


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


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


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


def reject_synthetic_model_for_actual_smoke(args: argparse.Namespace) -> None:
    if args.model != "synthetic/no-weight":
        return
    raise SmokeError(
        "synthetic/no-weight is an L0 product observability wiring fixture, not an actual model "
        "smoke; run scripts/release/product_observability_wiring_gate.py for synthetic run/serve "
        "wiring evidence"
    )


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


def validate_replay_execution_summary(summary: dict[str, Any], *, label: str) -> dict[str, Any]:
    if summary.get("status") != "pass":
        raise SmokeError(f"{label}.status must be pass")
    bundle_count = summary.get("bundle_count")
    replay_execution_count = summary.get("replay_execution_count")
    replay_execution_skipped_count = summary.get("replay_execution_skipped_count")
    for key, value in [
        ("bundle_count", bundle_count),
        ("replay_execution_count", replay_execution_count),
        ("replay_execution_skipped_count", replay_execution_skipped_count),
    ]:
        if not isinstance(value, int) or value < 0:
            raise SmokeError(f"{label}.{key} must be a non-negative integer")
    if bundle_count <= 0:
        raise SmokeError(f"{label}.bundle_count must be positive")
    if replay_execution_count <= 0:
        raise SmokeError(f"{label} did not execute any offline replay")
    if replay_execution_skipped_count >= bundle_count:
        raise SmokeError(f"{label} skipped all replay bundles")
    replay_executions = summary.get("replay_executions")
    if not isinstance(replay_executions, list):
        raise SmokeError(f"{label}.replay_executions must be a list")
    skipped = [
        item
        for item in replay_executions
        if isinstance(item, dict) and item.get("status") == "skipped_requires_running_server"
    ]
    if len(skipped) != replay_execution_skipped_count:
        raise SmokeError(
            f"{label}.replay_execution_skipped_count does not match skipped replay records"
        )
    for item in skipped:
        source = str(item.get("source_bundle_dir") or "").replace("\\", "/")
        if "/run/request_dump/" in source or source.endswith("/run/request_dump"):
            raise SmokeError(f"{label} skipped run replay bundle: {source}")
    return {
        "bundle_count": bundle_count,
        "replay_execution_count": replay_execution_count,
        "replay_execution_skipped_count": replay_execution_skipped_count,
    }


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
    summary_path = out / "request_replay_bundle/request_replay_bundle_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    replay_counts = validate_replay_execution_summary(summary, label=str(summary_path))
    return {
        "bundles": [str(bundle) for bundle in bundles],
        "out": str(out / "request_replay_bundle"),
        "summary": str(summary_path),
        **replay_counts,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    reject_synthetic_model_for_actual_smoke(args)
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
    if "synthetic/no-weight" in message or "not an actual model smoke" in message:
        return "not_actual_model"
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
    kind = failure_kind(exc)
    classification: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "fail",
        "failure_kind": kind,
        "error": str(exc),
        "requested_backend": exc.requested if isinstance(exc, BackendUnavailableError) else args.backend,
        "effective_backend": exc.effective if isinstance(exc, BackendMismatchError) else None,
        "suspected_domain": "goal_stage_selection"
        if kind == "not_actual_model"
        else "backend_runtime_preset"
        if isinstance(exc, BackendMismatchError)
        else "backend_compilation_or_cli_selection"
        if isinstance(exc, BackendUnavailableError)
        else "product_observability_l1_smoke",
        "next_gate": "product_observability_wiring_gate"
        if kind == "not_actual_model"
        else "backend_runtime_preset_snapshot"
        if isinstance(exc, BackendMismatchError)
        else "build_feature_or_backend_smoke"
        if isinstance(exc, BackendUnavailableError)
        else "inspect_product_observability_l1_artifact",
        "do_not_run": ["actual_model_regression", "l2_representative_backend", "release_full"]
        if kind == "not_actual_model"
        else ["l2_representative_backend", "release_full"]
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


def selftest_profile_event(
    *,
    entrypoint: str,
    event_id: str,
    phase: str,
    event_kind: str = "timed_span",
    duration_us: int | None = 100,
    memory: dict[str, Any] | None = None,
    resource: dict[str, Any] | None = None,
    replay: dict[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ts_unix_nanos": 1783036800000000000,
        "event_id": event_id,
        "request_id": f"req-{entrypoint}-selftest",
        "correlation_id": f"corr-{entrypoint}-selftest",
        "entrypoint": entrypoint,
        "backend": "synthetic",
        "runtime_preset_hash": SYNTHETIC_RUNTIME_PRESET_HASH,
        "phase": phase,
        "event_kind": event_kind,
        "timestamp": "2026-07-03T00:00:00Z",
        "status": "ok",
        "model": "synthetic/no-weight",
        "shape": {"batch_size": 1},
        "attributes": {
            "actual_model_smoke": True,
            "profile_detail": "basic",
            "profile_schema_fingerprint": "obs-v1",
            **(attributes or {}),
        },
    }
    if duration_us is not None:
        event["duration_us"] = duration_us
    if memory is not None:
        event["memory"] = memory
    if resource is not None:
        event["resource"] = resource
    if replay is not None:
        event["replay"] = replay
    return event


def selftest_resource(
    action: str,
    resource_kind: str,
    *,
    amount: int | None = None,
    before: int | None = None,
    after: int | None = None,
    capacity: int | None = None,
) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "owner_kind": "request",
        "owner_id": "req-selftest",
        "resource_kind": resource_kind,
        "action": action,
    }
    if amount is not None:
        resource["amount"] = amount
    if before is not None:
        resource["before"] = before
    if after is not None:
        resource["after"] = after
    if capacity is not None:
        resource["capacity"] = capacity
    return resource


def write_selftest_replay_bundle(root: Path, *, entrypoint: str, request_id: str) -> None:
    bundle = root / request_id
    bundle.mkdir(parents=True, exist_ok=True)
    output_text = "OK\n"
    command = f"ferrum {entrypoint} synthetic/no-weight --request-dump-dir {root}"
    common = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "sanitized": True,
    }
    write_json(
        bundle / "request.json",
        {
            **common,
            "entrypoint": entrypoint,
            "model": "synthetic/no-weight",
            "backend": "synthetic",
            "messages": [{"role": "user", "content": "Reply OK."}],
        },
    )
    write_json(
        bundle / "prompt_token_ids.json",
        {**common, "token_ids": [1, 2, 3], "token_count": 3},
    )
    write_json(
        bundle / "sampling_params.json",
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "sampling_params": {"temperature": 0.0, "max_tokens": 4},
            "sanitized": True,
        },
    )
    write_json(
        bundle / "runtime_effective_config.json",
        {**common, "entrypoint": entrypoint, "profile_detail": "basic"},
    )
    write_json(
        bundle / "backend_selection.json",
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "backend": "synthetic",
            "model": "synthetic/no-weight",
        },
    )
    write_json(
        bundle / "output_token_ids.json",
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "token_ids": [4, 5],
            "token_count": 2,
            "finish_reason": "stop",
        },
    )
    write_json(
        bundle / "bad_output_scan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "bad_output": False,
            "bad_text_count": 0,
            "reasons": [],
            "first_bad_text_span": None,
            "failure_kind": None,
            "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        },
    )
    write_json(
        bundle / "replay.command.json",
        {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "entrypoint": entrypoint,
            "command": command,
            "argv": [
                "ferrum",
                entrypoint,
                "synthetic/no-weight",
                "--request-dump-dir",
                str(root),
            ],
            "sanitized": True,
        },
    )
    (bundle / "output_text.txt").write_text(output_text, encoding="utf-8")


def write_selftest_profile_group(out: Path, entrypoint: str) -> None:
    root = out / entrypoint
    request_id = f"req-{entrypoint}-selftest"
    request_dump_root = root / "request_dump"
    replay = {"command": f"ferrum {entrypoint} synthetic/no-weight", "bundle_dir": str(request_dump_root)}
    write_jsonl(
        root / "profile.jsonl",
        [
            selftest_profile_event(
                entrypoint=entrypoint,
                event_id=f"evt-{entrypoint}-latency",
                phase="decode",
                duration_us=1234,
                replay=replay,
                attributes={"finish_reason": "stop", "output_tokens": 2},
            )
        ],
    )
    memory = {
        "scope": "process",
        "before_bytes": 1024,
        "after_bytes": 2048,
        "current_bytes": 2048,
        "high_water_bytes": 4096,
        "available_bytes": 8192,
    }
    write_jsonl(
        root / "memory_profile.jsonl",
        [
            selftest_profile_event(
                entrypoint=entrypoint,
                event_id=f"evt-{entrypoint}-memory",
                phase="memory_sample",
                duration_us=10,
                memory=memory,
                attributes={"memory_measurement": "process_rss"},
            )
        ],
    )
    scheduler_events = [
        selftest_profile_event(
            entrypoint=entrypoint,
            event_id=f"evt-{entrypoint}-request-open",
            phase="admission",
            event_kind="instant",
            duration_us=None,
            resource=selftest_resource("request_open", "request_slot"),
            attributes={"resource_trace_source": "engine"},
        ),
        selftest_profile_event(
            entrypoint=entrypoint,
            event_id=f"evt-{entrypoint}-slot-reserve",
            phase="admission",
            resource=selftest_resource(
                "reserve", "request_slot", amount=1, before=0, after=1, capacity=4
            ),
            attributes={"resource_trace_source": "engine"},
        ),
        selftest_profile_event(
            entrypoint=entrypoint,
            event_id=f"evt-{entrypoint}-slot-commit",
            phase="prefill",
            resource=selftest_resource(
                "commit", "request_slot", amount=1, before=0, after=1, capacity=4
            ),
            attributes={"resource_trace_source": "engine"},
        ),
        selftest_profile_event(
            entrypoint=entrypoint,
            event_id=f"evt-{entrypoint}-kv-reserve",
            phase="prefill",
            resource=selftest_resource(
                "reserve", "kv_block", amount=1, before=0, after=1, capacity=16
            ),
            attributes={"resource_trace_source": "engine"},
        ),
        selftest_profile_event(
            entrypoint=entrypoint,
            event_id=f"evt-{entrypoint}-kv-commit",
            phase="decode",
            resource=selftest_resource(
                "commit", "kv_block", amount=1, before=0, after=1, capacity=16
            ),
            attributes={"resource_trace_source": "engine"},
        ),
        selftest_profile_event(
            entrypoint=entrypoint,
            event_id=f"evt-{entrypoint}-kv-release",
            phase="request_complete",
            resource=selftest_resource(
                "release", "kv_block", amount=1, before=1, after=0, capacity=16
            ),
            attributes={"resource_trace_source": "engine"},
        ),
        selftest_profile_event(
            entrypoint=entrypoint,
            event_id=f"evt-{entrypoint}-slot-release",
            phase="request_complete",
            resource=selftest_resource(
                "release", "request_slot", amount=1, before=1, after=0, capacity=4
            ),
            attributes={"resource_trace_source": "engine"},
        ),
        selftest_profile_event(
            entrypoint=entrypoint,
            event_id=f"evt-{entrypoint}-request-close",
            phase="request_complete",
            event_kind="instant",
            duration_us=None,
            resource=selftest_resource("request_close", "request_slot"),
            attributes={"resource_trace_source": "engine"},
        ),
    ]
    write_jsonl(root / "scheduler_trace.jsonl", scheduler_events)
    write_json(
        request_dump_root / "request.json",
        {
            "schema_version": SCHEMA_VERSION,
            "entrypoint": entrypoint,
            "request_id": request_id,
            "model": "synthetic/no-weight",
            "backend": "synthetic",
            "actual_model_smoke": True,
            "sanitized": True,
        },
    )
    (request_dump_root / "replay_command.txt").write_text(
        f"ferrum {entrypoint} synthetic/no-weight\n",
        encoding="utf-8",
    )
    write_selftest_replay_bundle(request_dump_root, entrypoint=entrypoint, request_id=request_id)


def assert_raises(fn, expected: str) -> str:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - self-test reports exact validator failure.
        text = str(exc)
        if expected not in text:
            raise SmokeError(f"expected error containing {expected!r}, got {text!r}") from exc
        return text
    raise SmokeError(f"expected failure containing {expected!r}")


def run_selftest_in_root(root: Path, out: Path | None = None) -> dict[str, Any]:
    work = root / "artifact"
    write_selftest_profile_group(work, "run")
    write_selftest_profile_group(work, "serve")
    run_profiles = validate_profile_group(work / "run", "run")
    serve_profiles = validate_profile_group(work / "serve", "serve")
    analyzer = run_analyzer(work, 60)
    resource_invariant = run_resource_invariant(work, 60)
    replay_bundles = []
    for bundle_root in [work / "run/request_dump", work / "serve/request_dump"]:
        replay_bundles.extend(validate_bundle_root(bundle_root))

    sse = parse_sse(
        'data: {"choices":[{"delta":{"content":"O"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"K"}}],"usage":{"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    if sse != {"done_count": 1, "content_chunks": 2, "usage_chunks": 1, "malformed": 0}:
        raise SmokeError(f"unexpected SSE parse result: {sse}")
    malformed_sse = parse_sse("data: {not json}\n\ndata: [DONE]\n\n")
    if malformed_sse["malformed"] != 1 or malformed_sse["done_count"] != 1:
        raise SmokeError(f"unexpected malformed SSE parse result: {malformed_sse}")

    replay_summary_pass = validate_replay_execution_summary(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "bundle_count": 2,
            "execute_replay": True,
            "replay_execution_count": 1,
            "replay_execution_skipped_count": 1,
            "replay_executions": [
                {
                    "source_bundle_dir": str(work / "run/request_dump/req-run-selftest"),
                    "status": "executed_synthetic",
                },
                {
                    "source_bundle_dir": str(work / "serve/request_dump/req-serve-selftest"),
                    "status": "skipped_requires_running_server",
                },
            ],
        },
        label="selftest.replay_summary_pass",
    )
    all_skipped_replay_error = assert_raises(
        lambda: validate_replay_execution_summary(
            {
                "status": "pass",
                "bundle_count": 2,
                "replay_execution_count": 0,
                "replay_execution_skipped_count": 2,
                "replay_executions": [
                    {
                        "source_bundle_dir": str(work / "serve/request_dump/req-a"),
                        "status": "skipped_requires_running_server",
                    },
                    {
                        "source_bundle_dir": str(work / "serve/request_dump/req-b"),
                        "status": "skipped_requires_running_server",
                    },
                ],
            },
            label="selftest.replay_summary_all_skipped",
        ),
        "did not execute any offline replay",
    )
    skipped_run_replay_error = assert_raises(
        lambda: validate_replay_execution_summary(
            {
                "status": "pass",
                "bundle_count": 2,
                "replay_execution_count": 1,
                "replay_execution_skipped_count": 1,
                "replay_executions": [
                    {
                        "source_bundle_dir": str(work / "run/request_dump/req-run-selftest"),
                        "status": "skipped_requires_running_server",
                    },
                    {
                        "source_bundle_dir": str(work / "serve/request_dump/req-serve-selftest"),
                        "status": "executed_synthetic",
                    },
                ],
            },
            label="selftest.replay_summary_run_skipped",
        ),
        "skipped run replay bundle",
    )

    failure_cases = {
        "backend_mismatch": BackendMismatchError("cuda", "metal"),
        "backend_unavailable": BackendUnavailableError("requested backend 'cuda' is not built"),
        "replay_failure": SmokeError("replay bundle validation failed"),
        "stream_failure": SmokeError("serve stream expected one [DONE], got 0"),
        "resource_invariant_failure": SmokeError("resource invariant gate did not print PASS"),
        "profile_failure": SmokeError("profile analyzer did not print PASS"),
        "serve_failure": SmokeError("server did not become healthy"),
        "run_failure": SmokeError("ferrum run did not emit an assistant JSONL event"),
        "not_actual_model": SmokeError("synthetic/no-weight is not an actual model smoke"),
    }
    failure_kinds = {name: failure_kind(exc) for name, exc in failure_cases.items()}
    for expected, actual in failure_kinds.items():
        if actual != expected:
            raise SmokeError(f"failure_kind({expected}) returned {actual}")

    bad_profile = root / "bad-profile"
    write_selftest_profile_group(bad_profile, "run")
    events = read_jsonl(bad_profile / "run/profile.jsonl")
    events[0]["attributes"]["actual_model_smoke"] = False
    write_jsonl(bad_profile / "run/profile.jsonl", events)
    bad_profile_error = assert_raises(
        lambda: validate_profile_group(bad_profile / "run", "run"),
        "actual_model_smoke",
    )

    failure_out = root / "failure-artifact"
    failure_args = argparse.Namespace(
        out=failure_out,
        model=MODEL_DEFAULT,
        backend="cuda",
        ferrum_bin=None,
        timeout=60,
        max_tokens=8,
    )
    write_failure_artifacts(
        failure_args,
        BackendMismatchError("cuda", "metal"),
    )
    classification = json.loads(
        (failure_out / "failures/failure_classification.json").read_text(encoding="utf-8")
    )
    if classification.get("failure_kind") != "backend_mismatch":
        raise SmokeError(f"unexpected failure classification: {classification}")
    if classification.get("do_not_run") != ["l2_representative_backend", "release_full"]:
        raise SmokeError(f"unexpected do_not_run classification: {classification}")
    synthetic_failure_out = root / "synthetic-failure-artifact"
    synthetic_failure_args = argparse.Namespace(
        out=synthetic_failure_out,
        model="synthetic/no-weight",
        backend="auto",
        ferrum_bin=None,
        timeout=60,
        max_tokens=8,
    )
    write_failure_artifacts(
        synthetic_failure_args,
        SmokeError("synthetic/no-weight is not an actual model smoke"),
    )
    synthetic_classification = json.loads(
        (synthetic_failure_out / "failures/failure_classification.json").read_text(
            encoding="utf-8"
        )
    )
    if synthetic_classification.get("failure_kind") != "not_actual_model":
        raise SmokeError(f"unexpected synthetic failure classification: {synthetic_classification}")
    if synthetic_classification.get("next_gate") != "product_observability_wiring_gate":
        raise SmokeError(f"unexpected synthetic next_gate: {synthetic_classification}")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "run_profiles": run_profiles,
        "serve_profiles": serve_profiles,
        "analyzer": analyzer,
        "resource_invariant": resource_invariant,
        "replay_bundle_count": len(replay_bundles),
        "replay_summary_validation": replay_summary_pass,
        "sse": sse,
        "failure_kinds": failure_kinds,
        "negative_cases": {
            "bad_profile": bad_profile_error,
            "all_skipped_replay": all_skipped_replay_error,
            "skipped_run_replay": skipped_run_replay_error,
        },
    }
    if summary["replay_bundle_count"] != 2:
        raise SmokeError(f"expected 2 replay bundles, got {summary['replay_bundle_count']}")
    if out is not None:
        write_json(out / "product_observability_l1_smoke_selftest.json", summary)
    return summary


def run_selftest(out: Path | None = None) -> dict[str, Any]:
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="selftest-work-", dir=out))
        return run_selftest_in_root(root, out)

    with tempfile.TemporaryDirectory(prefix="ferrum-product-observability-l1-selftest-") as tmp:
        root = Path(tmp)
        return run_selftest_in_root(root, out=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--ferrum-bin", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and args.out is None:
        parser.error("--out is required unless --self-test is set")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_selftest(args.out)
            print(SELFTEST_PASS_LINE)
        else:
            run_gate(args)
            print(f"{PASS_LINE}: {args.out}")
        return 0
    except SmokeError as exc:
        if args.out is not None:
            write_failure_artifacts(args, exc)
        print(f"PRODUCT OBSERVABILITY L1 SMOKE FAIL: {exc}", file=sys.stderr)
        return 1
    except BundleError as exc:
        if args.out is not None:
            write_failure_artifacts(args, SmokeError(str(exc)))
        print(f"PRODUCT OBSERVABILITY L1 SMOKE FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
