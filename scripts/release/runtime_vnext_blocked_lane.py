#!/usr/bin/env python3
"""Collect a truthful Runtime vNext G00 blocked Metal product lane.

The collector executes a frozen ``ferrum run`` or ``ferrum serve`` product
entrypoint inside the repository's bounded process-group runner. The product receives an explicit,
sanitized environment, while the artifact binds the product PID/PGID/start
identity, resource peaks, memory/swap preflight, output logs, effective config,
model bytes, and the checked-in collector identity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import bounded_command
    import runtime_vnext_resource_sampler as resource_sampler
except ModuleNotFoundError:
    from scripts.release import bounded_command
    from scripts.release import runtime_vnext_resource_sampler as resource_sampler


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_REPO_PATH = SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()
FROZEN_LEGACY_SHA = "cff4c47765ef3259b8a04890187d99c60da86394"
SCHEMA_VERSION = 1
ATTEMPT_TYPE = "runtime_vnext_blocked_product_attempt"
PROCESS_RECEIPT_TYPE = "runtime_vnext_product_process_receipt"
PASS_PREFIX = "FERRUM RUNTIME VNEXT G00 BLOCKED LANE PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT G00 BLOCKED LANE SELFTEST PASS"
MIN_METAL_HEADROOM_BYTES = 2 * 1024**3
BLOCKED_MODELS = {
    "m1-qwen35-4b": {
        "mode": "unsupported-architecture",
        "failure_class": "legacy-model-backend-unsupported",
        "architecture_marker": "unsupported GGUF architecture 'qwen35'",
        "downstream_goal": "G08A",
        "implementation_path": "implement Qwen3.5 dense-hybrid operations and Metal providers through Runtime vNext",
        "acceptance_path": "run the complete G08A Qwen3.5-4B correctness and performance matrix",
        "pass_line": "FERRUM RUNTIME VNEXT G08A QWEN35 4B PASS: <out_dir>",
    },
    "m2-qwen35-35b-a3b": {
        "mode": "unsupported-architecture",
        "failure_class": "legacy-model-backend-unsupported",
        "architecture_marker": "unsupported GGUF architecture 'qwen35moe'",
        "downstream_goal": "G08B",
        "implementation_path": "implement Qwen3.5 hybrid-MoE operations and Metal providers through Runtime vNext",
        "acceptance_path": "run the complete G08B Qwen3.5-35B-A3B correctness and performance matrix",
        "pass_line": "FERRUM RUNTIME VNEXT G08B QWEN35 35B A3B PASS: <out_dir>",
    },
    "m3-qwen3-30b-a3b": {
        "mode": "resource-capacity",
        "failure_class": "legacy-metal-unified-memory-capacity",
        "downstream_goal": "G08C",
        "implementation_path": "retain GGUF weights with explicit Metal residency and typed unified-memory admission through Runtime vNext",
        "acceptance_path": "run the complete G08C Qwen3-30B-A3B Metal correctness and performance matrix without active swap growth",
        "pass_line": "FERRUM RUNTIME VNEXT G08C QWEN3 30B A3B PASS: <out_dir>",
    },
}
CHILD_ENV_ALLOWLIST = frozenset(
    {
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "RUST_BACKTRACE",
        "RUST_LOG",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "USER",
    }
)


class BlockedLaneError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BlockedLaneError(message)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlockedLaneError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_path(root: Path, raw: Any, label: str) -> Path:
    require(isinstance(raw, str) and raw, f"{label} must be a non-empty path")
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise BlockedLaneError(f"{label} escapes artifact root: {raw}") from exc
    return resolved


def file_ref(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise BlockedLaneError(f"artifact is outside root: {path}") from exc
    require(
        path.is_file() and not path.is_symlink(),
        f"artifact must be a regular file: {path}",
    )
    return {
        "path": relative,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def validate_file_ref(
    root: Path, raw: Any, label: str, *, nonempty: bool = False
) -> Path:
    require(isinstance(raw, dict), f"{label} must be an artifact reference")
    path = artifact_path(root, raw.get("path"), f"{label}.path")
    require(
        path.is_file() and not path.is_symlink(), f"{label} is missing or is a symlink"
    )
    require(raw.get("size_bytes") == path.stat().st_size, f"{label} size mismatch")
    require(raw.get("sha256") == file_sha256(path), f"{label} SHA256 mismatch")
    if nonempty:
        require(path.stat().st_size > 0, f"{label} must not be empty")
    return path


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}"
    )
    return result.stdout.strip()


def collector_identity(*, allow_selftest: bool = False) -> dict[str, Any]:
    status = run_git("status", "--short").splitlines()
    if not allow_selftest:
        require(
            not status, "blocked lane collection requires a clean committed worktree"
        )
    return {
        "path": SCRIPT_REPO_PATH,
        "sha256": file_sha256(SCRIPT_PATH),
        "git_sha": run_git("rev-parse", "HEAD"),
        "tree_sha": run_git("rev-parse", "HEAD^{tree}"),
        "dirty_status": {"is_dirty": bool(status), "status_short": status},
    }


def sanitized_child_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in CHILD_ENV_ALLOWLIST and isinstance(value, str)
    }
    environment.update(
        {
            "LANG": "C",
            "LC_ALL": "C",
            "NO_COLOR": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return dict(sorted(environment.items()))


def _command_output(argv: list[str]) -> str:
    result = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    require(
        result.returncode == 0,
        f"memory command failed: {' '.join(argv)}: {result.stdout.strip()}",
    )
    return result.stdout


def capture_memory_snapshot(
    root: Path, attempt_root: Path, name: str
) -> dict[str, Any]:
    system = platform.system()
    raw_refs: list[dict[str, Any]] = []
    if system == "Darwin":
        commands = [
            ("vm-stat", ["/usr/bin/vm_stat"]),
            ("swap", ["/usr/sbin/sysctl", "vm.swapusage"]),
            ("pressure", ["/usr/bin/memory_pressure", "-Q"]),
        ]
        headroom, swap = resource_sampler._mac_memory()
    elif system == "Linux":
        commands = [("meminfo", ["/bin/cat", "/proc/meminfo"])]
        headroom, swap = resource_sampler._linux_memory()
    else:
        raise BlockedLaneError(f"unsupported memory platform: {system}")
    for suffix, argv in commands:
        path = attempt_root / f"memory.{name}.{suffix}.log"
        path.write_text(_command_output(argv), encoding="utf-8")
        raw_refs.append(file_ref(path, root))
    return {
        "captured_at": utc_now(),
        "physical_headroom_bytes": headroom,
        "swap_used_bytes": swap,
        "raw": raw_refs,
    }


def memory_values() -> tuple[int, int]:
    if platform.system() == "Darwin":
        return resource_sampler._mac_memory()
    if platform.system() == "Linux":
        return resource_sampler._linux_memory()
    raise BlockedLaneError(f"unsupported memory platform: {platform.system()}")


def reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def server_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=0.2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def write_memory_samples(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def terminate_product(process: subprocess.Popen[bytes]) -> int:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    return int(process.returncode)


def capture_process_receipt(
    root: Path,
    path: Path,
    *,
    process: subprocess.Popen[bytes],
    argv: list[str],
    environment: dict[str, str],
) -> dict[str, Any]:
    pid = process.pid
    pgid = os.getpgid(pid)
    ps = subprocess.run(
        [
            "/bin/ps",
            "-ww",
            "-p",
            str(pid),
            "-o",
            "pid=",
            "-o",
            "ppid=",
            "-o",
            "pgid=",
            "-o",
            "command=",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    require(
        ps.returncode == 0 and ps.stdout.strip(),
        f"cannot capture product ps receipt: {ps.stderr.strip()}",
    )
    fields = ps.stdout.strip().split(None, 3)
    require(len(fields) == 4, "product ps receipt has an invalid shape")
    ps_pid, ppid, ps_pgid = (int(fields[index]) for index in range(3))
    require(ps_pid == pid and ps_pgid == pgid, "product ps receipt PID/PGID mismatch")
    marker, source = resource_sampler._process_start_identity(pid)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": PROCESS_RECEIPT_TYPE,
        "captured_at": utc_now(),
        "captured_monotonic_ns": time.monotonic_ns(),
        "pid": pid,
        "ppid": ppid,
        "pgid": pgid,
        "argv": argv,
        "argv_sha256": canonical_json_sha256(argv),
        "environment": environment,
        "environment_sha256": canonical_json_sha256(environment),
        "product_ferrum_keys": sorted(
            key for key in environment if key.startswith("FERRUM_")
        ),
        "process_start_marker": marker,
        "process_start_source": source,
        "ps_command": fields[3],
        "ps_stdout_sha256": hashlib.sha256(ps.stdout.encode("utf-8")).hexdigest(),
    }
    bounded_command.atomic_write_json(path, receipt)
    return file_ref(path, root)


def internal_execute(spec_path: Path) -> int:
    spec = read_json(spec_path)
    mode = str(spec["mode"])
    policy = BLOCKED_MODELS[str(spec["model_key"])]
    require(mode == policy["mode"], "blocked attempt mode differs from model policy")
    root = Path(spec["artifact_root"]).resolve()
    attempt_root = Path(spec["attempt_root"]).resolve()
    attempt_root.mkdir(parents=True, exist_ok=True)
    argv = list(spec["argv"])
    environment = dict(spec["child_environment"])
    require(
        not any(key.startswith("FERRUM_") for key in environment),
        "product environment contains FERRUM_*",
    )
    environment_path = attempt_root / "child-environment.json"
    bounded_command.atomic_write_json(environment_path, environment)
    stdout_path = attempt_root / "product.stdout.log"
    stderr_path = attempt_root / "product.stderr.log"
    effective_path = attempt_root / "effective-config.json"
    samples_path = attempt_root / "memory.samples.jsonl"
    failure_log_path = attempt_root / "failure.log"
    before = capture_memory_snapshot(root, attempt_root, "before")
    require(
        before["physical_headroom_bytes"] >= MIN_METAL_HEADROOM_BYTES,
        "blocked product preflight headroom is below 2 GiB",
    )
    started_at = utc_now()
    started_monotonic = time.monotonic()
    memory_rows: list[dict[str, Any]] = []
    violation: dict[str, Any] | None = None
    ready_observed = False
    with (
        stdout_path.open("wb") as stdout_handle,
        stderr_path.open("wb") as stderr_handle,
    ):
        process = subprocess.Popen(
            argv,
            cwd=spec["cwd"],
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=environment,
            start_new_session=False,
        )
        process_ref = capture_process_receipt(
            root,
            attempt_root / "process-receipt.json",
            process=process,
            argv=argv,
            environment=environment,
        )
        if mode == "unsupported-architecture":
            try:
                returncode = process.wait(timeout=float(spec["product_timeout_seconds"]))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                returncode = 124
        else:
            deadline = time.monotonic() + float(spec["product_timeout_seconds"])
            interval = float(spec["sample_interval_seconds"])
            health_url = str(spec["health_url"])
            sequence = 0
            while time.monotonic() < deadline:
                headroom, swap = memory_values()
                ready_observed = ready_observed or server_ready(health_url)
                row = {
                    "sequence": sequence,
                    "captured_at": utc_now(),
                    "captured_monotonic_ns": time.monotonic_ns(),
                    "physical_headroom_bytes": headroom,
                    "swap_used_bytes": swap,
                    "process_alive": process.poll() is None,
                    "ready_observed": ready_observed,
                }
                memory_rows.append(row)
                write_memory_samples(samples_path, memory_rows)
                if swap > before["swap_used_bytes"]:
                    violation = {
                        "kind": "swap_growth",
                        "sequence": sequence,
                        "initial_bytes": before["swap_used_bytes"],
                        "observed_bytes": swap,
                    }
                elif headroom < MIN_METAL_HEADROOM_BYTES:
                    violation = {
                        "kind": "physical_headroom_below_2gib",
                        "sequence": sequence,
                        "limit_bytes": MIN_METAL_HEADROOM_BYTES,
                        "observed_bytes": headroom,
                    }
                if violation is not None or process.poll() is not None:
                    break
                sequence += 1
                time.sleep(interval)
            returncode = terminate_product(process)
    finished_at = utc_now()
    duration = time.monotonic() - started_monotonic
    after = capture_memory_snapshot(root, attempt_root, "after")
    if stdout_path.stat().st_size == 0:
        stdout_path.write_text(
            "product stdout capture was empty; collector observed the product attempt directly\n",
            encoding="utf-8",
        )
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    if mode == "unsupported-architecture":
        marker = str(spec["architecture_marker"])
        failure_line = next(
            (line.strip() for line in stderr_text.splitlines() if marker in line), ""
        )
        observed_failure_class = (
            str(policy["failure_class"])
            if returncode != 0 and failure_line
            else "unexpected-product-outcome"
        )
    else:
        marker = str(violation["kind"]) if violation is not None else "resource-capacity-violation-missing"
        failure_line = (
            f"{marker}: {json.dumps(violation, sort_keys=True)}"
            if violation is not None
            else "frozen Metal serve exited or timed out without the required resource-capacity evidence"
        )
        observed_failure_class = (
            str(policy["failure_class"])
            if violation is not None and returncode != 0
            else "unexpected-product-outcome"
        )
    failure_log_path.write_text(failure_line + "\n" + stderr_text, encoding="utf-8")
    effective: dict[str, Any] | None = None
    if effective_path.is_file():
        effective = read_json(effective_path)
    attempt = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ATTEMPT_TYPE,
        "model_key": spec["model_key"],
        "backend": "metal",
        "mode": mode,
        "failure_class": observed_failure_class,
        "failure_signature": marker,
        "first_failure": failure_line,
        "argv": argv,
        "argv_sha256": canonical_json_sha256(argv),
        "child_environment": file_ref(environment_path, root),
        "child_environment_sha256": canonical_json_sha256(environment),
        "product_ferrum_keys": sorted(
            key for key in environment if key.startswith("FERRUM_")
        ),
        "process_receipt": process_ref,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration, 6),
        "returncode": returncode,
        "memory": {
            "before": before,
            "after": after,
            "samples": file_ref(samples_path, root) if samples_path.is_file() else None,
            "violation": violation,
        },
        "ready_observed": ready_observed,
        "stdout": file_ref(stdout_path, root),
        "stderr": file_ref(stderr_path, root),
        "failure_log": file_ref(failure_log_path, root),
        "effective_config": file_ref(effective_path, root)
        if effective is not None
        else None,
    }
    bounded_command.atomic_write_json(Path(spec["attempt_receipt_path"]), attempt)
    print(
        json.dumps(
            {
                "blocked_product_attempt": {
                    "model_key": spec["model_key"],
                    "returncode": returncode,
                    "failure_class": attempt["failure_class"],
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        "blocked product attempt executor finished and wrote its receipt",
        file=sys.stderr,
        flush=True,
    )
    return (
        0
        if attempt["failure_class"] == policy["failure_class"] and effective is not None
        else 1
    )


def _validate_environment(environment: dict[str, Any], label: str) -> None:
    require(environment == dict(sorted(environment.items())), f"{label} must be sorted")
    require(
        all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ),
        f"{label} must contain strings",
    )
    allowed = CHILD_ENV_ALLOWLIST | {"NO_COLOR", "PYTHONUNBUFFERED"}
    require(set(environment) <= allowed, f"{label} contains non-allowlisted keys")
    require(
        not any(key.startswith("FERRUM_") for key in environment),
        f"{label} contains product FERRUM_*",
    )
    require(environment.get("NO_COLOR") == "1", f"{label}.NO_COLOR must be 1")
    require(
        environment.get("PYTHONUNBUFFERED") == "1",
        f"{label}.PYTHONUNBUFFERED must be 1",
    )


def validate_attempt_document(
    root: Path, attempt: dict[str, Any], lane: dict[str, Any]
) -> None:
    model_key = str(lane.get("model_key"))
    require(model_key in BLOCKED_MODELS, "attempt model is not blockable")
    policy = BLOCKED_MODELS[model_key]
    mode = str(policy["mode"])
    require(attempt.get("schema_version") == SCHEMA_VERSION, "attempt schema mismatch")
    require(
        attempt.get("artifact_type") == ATTEMPT_TYPE, "attempt artifact type mismatch"
    )
    require(
        attempt.get("model_key") == lane.get("model_key")
        and attempt.get("backend") == "metal",
        "attempt lane identity mismatch",
    )
    require(
        attempt.get("mode") == mode,
        "attempt mode mismatch",
    )
    require(
        attempt.get("failure_class") == policy["failure_class"],
        "attempt failure class mismatch",
    )
    signature = attempt.get("failure_signature")
    require(
        isinstance(signature, str) and signature, "attempt failure signature missing"
    )
    require(
        signature in str(attempt.get("first_failure", "")),
        "attempt first failure does not contain signature",
    )
    require(
        attempt.get("argv") == lane.get("attempted_command"),
        "attempt argv differs from lane",
    )
    require(
        (mode == "resource-capacity" and "serve" in attempt["argv"])
        or (mode == "unsupported-architecture" and "run" in attempt["argv"]),
        "attempt product entrypoint differs from blocked mode",
    )
    require(
        attempt.get("argv_sha256") == canonical_json_sha256(attempt["argv"]),
        "attempt argv SHA mismatch",
    )
    returncode = attempt.get("returncode")
    require(
        isinstance(returncode, int)
        and not isinstance(returncode, bool)
        and returncode != 0,
        "attempt returncode must be non-zero",
    )
    require(
        returncode == lane.get("attempted_returncode"),
        "attempt returncode differs from lane",
    )
    environment_path = validate_file_ref(
        root,
        attempt.get("child_environment"),
        "attempt child environment",
        nonempty=True,
    )
    environment = read_json(environment_path)
    _validate_environment(environment, "attempt child environment")
    require(
        attempt.get("child_environment_sha256") == canonical_json_sha256(environment),
        "attempt environment SHA mismatch",
    )
    require(
        attempt.get("product_ferrum_keys") == [],
        "attempt inherited product FERRUM_* controls",
    )
    process_path = validate_file_ref(
        root, attempt.get("process_receipt"), "attempt process receipt", nonempty=True
    )
    process = read_json(process_path)
    require(
        process.get("artifact_type") == PROCESS_RECEIPT_TYPE,
        "process receipt type mismatch",
    )
    for field in ("pid", "ppid", "pgid"):
        require(
            isinstance(process.get(field), int) and process[field] > 0,
            f"process receipt {field} invalid",
        )
    require(process.get("argv") == attempt.get("argv"), "process receipt argv mismatch")
    require(
        process.get("argv_sha256") == attempt.get("argv_sha256"),
        "process receipt argv SHA mismatch",
    )
    require(
        process.get("environment") == environment,
        "process receipt environment mismatch",
    )
    require(
        process.get("environment_sha256") == attempt.get("child_environment_sha256"),
        "process receipt environment SHA mismatch",
    )
    require(
        process.get("product_ferrum_keys") == [], "process receipt inherited FERRUM_*"
    )
    require(
        isinstance(process.get("process_start_marker"), str)
        and process["process_start_marker"],
        "process start marker missing",
    )
    source = process.get("process_start_source")
    require(
        isinstance(source, dict)
        and source.get("kind") in {"linux-proc-start", "ps-lstart"},
        "process start source invalid",
    )
    require(
        resource_sampler.process_marker_from_source(process["pid"], source)
        == process["process_start_marker"],
        "process start marker mismatch",
    )
    stdout_path = validate_file_ref(root, attempt.get("stdout"), "attempt stdout")
    stderr_path = validate_file_ref(
        root, attempt.get("stderr"), "attempt stderr", nonempty=True
    )
    effective_path = validate_file_ref(
        root, attempt.get("effective_config"), "attempt effective config", nonempty=True
    )
    require(
        isinstance(read_json(effective_path), dict),
        "attempt effective config must be JSON",
    )
    failure_log_path = validate_file_ref(
        root, attempt.get("failure_log"), "attempt failure log", nonempty=True
    )
    require(
        lane.get("failure_log")
        == failure_log_path.relative_to(root.resolve()).as_posix(),
        "lane failure log differs from attempt evidence",
    )
    require(
        len({stdout_path, stderr_path, failure_log_path}) == 3,
        "attempt output paths must be distinct",
    )
    memory = attempt.get("memory")
    require(isinstance(memory, dict), "attempt memory evidence missing")
    before = memory.get("before")
    after = memory.get("after")
    require(
        isinstance(before, dict) and isinstance(after, dict),
        "attempt memory snapshots missing",
    )
    for name, snapshot in (("before", before), ("after", after)):
        require(
            isinstance(snapshot.get("physical_headroom_bytes"), int),
            f"attempt {name} headroom invalid",
        )
        require(
            isinstance(snapshot.get("swap_used_bytes"), int)
            and snapshot["swap_used_bytes"] >= 0,
            f"attempt {name} swap invalid",
        )
        raw = snapshot.get("raw")
        require(
            isinstance(raw, list) and raw, f"attempt {name} raw memory evidence missing"
        )
        for index, ref in enumerate(raw):
            validate_file_ref(
                root, ref, f"attempt {name} memory raw[{index}]", nonempty=True
            )
    require(
        before["physical_headroom_bytes"] >= MIN_METAL_HEADROOM_BYTES,
        "attempt before headroom below 2 GiB",
    )
    failure_text = failure_log_path.read_text(encoding="utf-8", errors="strict")
    require(signature in failure_text, "failure signature absent from failure log")
    if mode == "unsupported-architecture":
        require(
            signature in stderr_path.read_text(encoding="utf-8", errors="strict"),
            "failure signature absent from stderr",
        )
        require(
            after["physical_headroom_bytes"] >= MIN_METAL_HEADROOM_BYTES,
            "attempt after headroom below 2 GiB",
        )
        require(
            after["swap_used_bytes"] <= before["swap_used_bytes"],
            "attempt active swap growth detected",
        )
        require(memory.get("samples") is None and memory.get("violation") is None, "unsupported attempt fabricated resource violation")
    else:
        samples_path = validate_file_ref(
            root, memory.get("samples"), "attempt memory samples", nonempty=True
        )
        rows = [
            json.loads(line)
            for line in samples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        require(rows, "resource attempt has no memory samples")
        require(
            all(isinstance(row, dict) for row in rows),
            "resource attempt memory samples must be objects",
        )
        require(
            [row.get("sequence") for row in rows] == list(range(len(rows))),
            "resource attempt memory sample sequence mismatch",
        )
        require(
            all(
                isinstance(row.get("captured_monotonic_ns"), int)
                and row["captured_monotonic_ns"] > 0
                and isinstance(row.get("physical_headroom_bytes"), int)
                and isinstance(row.get("swap_used_bytes"), int)
                for row in rows
            ),
            "resource attempt memory sample fields invalid",
        )
        monotonic_values = [row["captured_monotonic_ns"] for row in rows]
        require(
            all(left < right for left, right in zip(monotonic_values, monotonic_values[1:])),
            "resource attempt monotonic sample order invalid",
        )
        wall_values = [
            datetime.fromisoformat(str(row.get("captured_at")).replace("Z", "+00:00"))
            for row in rows
        ]
        require(
            all(left <= right for left, right in zip(wall_values, wall_values[1:])),
            "resource attempt wall-clock sample order invalid",
        )
        violation = memory.get("violation")
        require(isinstance(violation, dict), "resource attempt violation missing")
        sequence = violation.get("sequence")
        require(
            isinstance(sequence, int) and sequence == len(rows) - 1,
            "resource attempt violation is not bound to the final sample",
        )
        row = rows[sequence]
        require(row.get("process_alive") is True, "resource violation was not observed while product was alive")
        if violation.get("kind") == "swap_growth":
            require(
                violation.get("initial_bytes") == before["swap_used_bytes"]
                and violation.get("observed_bytes") == row["swap_used_bytes"]
                and row["swap_used_bytes"] > before["swap_used_bytes"],
                "resource attempt swap-growth evidence mismatch",
            )
        elif violation.get("kind") == "physical_headroom_below_2gib":
            require(
                violation.get("limit_bytes") == MIN_METAL_HEADROOM_BYTES
                and violation.get("observed_bytes") == row["physical_headroom_bytes"]
                and row["physical_headroom_bytes"] < MIN_METAL_HEADROOM_BYTES,
                "resource attempt headroom evidence mismatch",
            )
        else:
            raise BlockedLaneError("resource attempt violation kind invalid")
        require(signature == violation["kind"], "resource attempt failure signature differs from violation")
    started = datetime.fromisoformat(
        str(attempt.get("started_at")).replace("Z", "+00:00")
    )
    finished = datetime.fromisoformat(
        str(attempt.get("finished_at")).replace("Z", "+00:00")
    )
    require(finished > started, "attempt execution window invalid")
    duration = attempt.get("duration_seconds")
    require(
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and duration > 0,
        "attempt duration invalid",
    )


def validate_bounded_receipt(receipt: dict[str, Any]) -> None:
    require(
        receipt.get("schema") == bounded_command.RECEIPT_SCHEMA,
        "bounded receipt schema mismatch",
    )
    require(
        receipt.get("status") == "pass"
        and receipt.get("reason") == "command_completed"
        and receipt.get("rc") == 0,
        "bounded attempt did not pass",
    )
    require(
        receipt.get("violation") is None, "bounded attempt has a resource violation"
    )
    require(
        receipt.get("cleanup") == {"process_group_gone": True},
        "bounded attempt process group was not reaped",
    )
    require(
        isinstance(receipt.get("successful_samples"), int)
        and receipt["successful_samples"] > 0,
        "bounded attempt lacks samples",
    )
    limits = receipt.get("limits")
    peaks = receipt.get("peaks")
    require(
        isinstance(limits, dict) and isinstance(peaks, dict),
        "bounded attempt limits/peaks missing",
    )
    require(
        limits.get("max_processes") == 4
        and limits.get("max_group_threads") == 64
        and limits.get("max_per_process_threads") == 64,
        "bounded attempt limits drifted",
    )
    require(
        peaks.get("processes", 0) <= 4
        and peaks.get("group_threads", 0) <= 64
        and peaks.get("per_process_threads", 0) <= 64,
        "bounded attempt peaks exceed limits",
    )
    command = receipt.get("command")
    require(
        isinstance(command, list) and "--internal-execute" in command,
        "bounded attempt did not run the collector executor",
    )


def validate_lane_evidence(
    root: Path, lane: dict[str, Any], *, allow_selftest: bool = False
) -> None:
    root = root.resolve()
    model_key = str(lane.get("model_key"))
    require(model_key in BLOCKED_MODELS, "blocked lane model policy missing")
    policy = BLOCKED_MODELS[model_key]
    require(
        lane.get("failure_class") == policy["failure_class"],
        "blocked lane failure class differs from model policy",
    )
    require(
        lane.get("models_lock_sha256") == file_sha256(root / "models.lock.json"),
        "blocked lane models.lock SHA mismatch",
    )
    collector = lane.get("collector")
    require(isinstance(collector, dict), "blocked lane collector identity missing")
    require(
        collector.get("path") == SCRIPT_REPO_PATH
        and collector.get("sha256") == file_sha256(SCRIPT_PATH),
        "blocked lane collector source mismatch",
    )
    if not allow_selftest:
        current = collector_identity()
        require(
            collector == current,
            "blocked lane collector git/tree/dirty identity is stale",
        )
    spec_path = validate_file_ref(
        root, lane.get("collection_spec"), "blocked lane collection spec", nonempty=True
    )
    spec = read_json(spec_path)
    require(spec.get("model_key") == model_key and spec.get("mode") == policy["mode"], "blocked lane collection spec mismatch")
    if policy["mode"] == "unsupported-architecture":
        require(
            spec.get("architecture_marker") in lane.get("first_failure", ""),
            "blocked lane architecture signature mismatch",
        )
    else:
        models_lock = read_json(root / "models.lock.json")
        locked_hardware = next(
            (
                row
                for row in models_lock.get("hardware", [])
                if isinstance(row, dict) and row.get("id") == lane.get("hardware_id")
            ),
            None,
        )
        require(
            isinstance(locked_hardware, dict)
            and locked_hardware.get("policy_id") == "metal-reference-m1-max-32gb"
            and locked_hardware.get("device_name") == "Apple M1 Max"
            and locked_hardware.get("memory_bytes") == 32 * 1024**3,
            "blocked resource models.lock hardware identity mismatch",
        )
        require(
            spec.get("minimum_physical_headroom_bytes") == MIN_METAL_HEADROOM_BYTES
            and spec.get("allow_swap_growth") is False,
            "blocked resource policy drift",
        )
        require(
            lane.get("hardware_constraint")
            == {
                "policy_id": "metal-reference-m1-max-32gb",
                "device_name": "Apple M1 Max",
                "memory_bytes": 32 * 1024**3,
                "minimum_physical_headroom_bytes": MIN_METAL_HEADROOM_BYTES,
                "allow_swap_growth": False,
            },
            "blocked resource hardware constraint drift",
        )
    attempt_path = validate_file_ref(
        root, lane.get("attempt_receipt"), "blocked lane attempt receipt", nonempty=True
    )
    attempt = read_json(attempt_path)
    validate_attempt_document(root, attempt, lane)
    bounded_path = validate_file_ref(
        root, lane.get("bounded_receipt"), "blocked lane bounded receipt", nonempty=True
    )
    validate_bounded_receipt(read_json(bounded_path))
    resolution = lane.get("resolution_evidence")
    require(isinstance(resolution, dict), "blocked lane resolution evidence missing")
    require(
        resolution.get("model_arg") in lane.get("attempted_command", []),
        "blocked lane model arg not executed",
    )
    require(
        resolution.get("weight_sha256") in lane.get("model_files", {}).values(),
        "blocked lane weight SHA is not locked",
    )


def _locked_model(models_lock: dict[str, Any], model_key: str) -> dict[str, Any]:
    rows = models_lock.get("models")
    require(isinstance(rows, list), "models.lock models must be an array")
    matches = [
        row for row in rows if isinstance(row, dict) and row.get("key") == model_key
    ]
    require(len(matches) == 1, f"models.lock does not contain exactly one {model_key}")
    return matches[0]


def _validate_source_root(root: Path, source: dict[str, Any], label: str) -> None:
    files = source.get("files")
    require(isinstance(files, list) and files, f"{label} lock is empty")
    for row in files:
        path = root / row["path"]
        require(path.is_file(), f"{label} file missing: {path}")
        require(
            path.stat().st_size == row["size_bytes"], f"{label} size mismatch: {path}"
        )
        require(file_sha256(path) == row["sha256"], f"{label} SHA mismatch: {path}")


def collect(args: argparse.Namespace) -> Path:
    root = args.artifact_root.expanduser().resolve()
    try:
        root.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise BlockedLaneError("artifact root must be outside the source worktree")
    require(
        root.is_dir() and (root / "models.lock.json").is_file(),
        "artifact root lacks models.lock.json",
    )
    models_lock = read_json(root / "models.lock.json")
    require(
        models_lock.get("source_git_sha") == FROZEN_LEGACY_SHA,
        "models.lock does not bind the frozen legacy SHA",
    )
    model = _locked_model(models_lock, args.model_key)
    policy = BLOCKED_MODELS[args.model_key]
    lane_lock = model.get("lanes", {}).get("metal")
    require(isinstance(lane_lock, dict), "models.lock Metal lane missing")
    hardware_rows = models_lock.get("hardware")
    require(isinstance(hardware_rows, list), "models.lock hardware list missing")
    locked_hardware = next(
        (
            row
            for row in hardware_rows
            if isinstance(row, dict) and row.get("id") == lane_lock.get("hardware_id")
        ),
        None,
    )
    require(isinstance(locked_hardware, dict), "blocked lane hardware lock missing")
    if policy["mode"] == "resource-capacity":
        require(platform.system() == "Darwin", "M3 resource blocker must be collected on Darwin")
        require(
            locked_hardware.get("policy_id") == "metal-reference-m1-max-32gb"
            and locked_hardware.get("device_name") == "Apple M1 Max"
            and locked_hardware.get("memory_bytes") == 32 * 1024**3,
            "M3 resource blocker is allowed only on the locked 32 GiB M1 Max",
        )
    weights = lane_lock.get("files")
    require(
        isinstance(weights, list) and len(weights) == 1,
        "blocked Metal lane must lock exactly one GGUF",
    )
    weight = weights[0]
    model_arg = args.model_arg.expanduser().resolve()
    require(
        model_arg.is_file() and model_arg.name == weight["path"],
        "model argument does not match locked GGUF path",
    )
    require(
        model_arg.stat().st_size == weight["size_bytes"]
        and file_sha256(model_arg) == weight["sha256"],
        "model argument bytes differ from models.lock",
    )
    semantic_root = args.semantic_source_root.expanduser().resolve()
    _validate_source_root(
        semantic_root, lane_lock["semantic_source"], "semantic source"
    )
    tokenizer_source = lane_lock.get("tokenizer_source") or lane_lock["semantic_source"]
    tokenizer_root = (
        (args.tokenizer_source_root or args.semantic_source_root).expanduser().resolve()
    )
    _validate_source_root(tokenizer_root, tokenizer_source, "tokenizer source")
    binary = artifact_path(root, args.binary_artifact, "binary artifact")
    require(
        binary.is_file() and not binary.is_symlink(),
        "blocked lane binary artifact missing",
    )
    lane_root = root / "correctness" / args.model_key / "metal"
    require(
        not lane_root.exists(),
        f"blocked lane artifact already exists; use a fresh G00 root: {lane_root}",
    )
    attempt_root = lane_root / "attempt"
    attempt_root.mkdir(parents=True)
    effective_path = attempt_root / "effective-config.json"
    if policy["mode"] == "unsupported-architecture":
        argv = [
            str(binary),
            "run",
            str(model_arg),
            "--backend",
            "metal",
            "--prompt",
            "Reply with Paris.",
            "--max-tokens",
            "4",
            "--output-format",
            "jsonl",
            "--tokenizer",
            str(tokenizer_root / "tokenizer.json"),
            "--effective-config-json",
            str(effective_path),
        ]
        resource_spec: dict[str, Any] = {}
    else:
        port = reserve_loopback_port()
        argv = [
            str(binary),
            "serve",
            str(model_arg),
            "--backend",
            "metal",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--effective-config-json",
            str(effective_path),
            "--scheduler-trace-jsonl",
            str(attempt_root / "scheduler-trace.jsonl"),
        ]
        resource_spec = {
            "health_url": f"http://127.0.0.1:{port}/health",
            "sample_interval_seconds": 0.1,
            "minimum_physical_headroom_bytes": MIN_METAL_HEADROOM_BYTES,
            "allow_swap_growth": False,
        }
    spec = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_blocked_lane_collection_spec",
        "artifact_root": str(root),
        "attempt_root": str(attempt_root),
        "attempt_receipt_path": str(attempt_root / "attempt-receipt.json"),
        "cwd": str(REPO_ROOT),
        "model_key": args.model_key,
        "mode": policy["mode"],
        "failure_class": policy["failure_class"],
        "product_timeout_seconds": 60,
        "argv": argv,
        "child_environment": sanitized_child_environment(),
        **(
            {"architecture_marker": policy["architecture_marker"]}
            if policy["mode"] == "unsupported-architecture"
            else resource_spec
        ),
    }
    spec_path = attempt_root / "collection-spec.json"
    bounded_command.atomic_write_json(spec_path, spec)
    _, bounded_receipt = bounded_command.run_bounded_command(
        command=[
            sys.executable,
            str(SCRIPT_PATH),
            "--internal-execute",
            "--spec",
            str(spec_path),
        ],
        cwd=REPO_ROOT,
        receipt_path=attempt_root / "bounded-receipt.json",
        stdout_path=attempt_root / "collector.stdout.log",
        stderr_path=attempt_root / "collector.stderr.log",
        limits=bounded_command.Limits(
            wall_timeout_seconds=120,
            max_processes=4,
            max_group_threads=64,
            max_per_process_threads=64,
            sample_interval_seconds=0.05,
            max_sampling_errors=3,
            term_grace_seconds=1,
        ),
    )
    require(
        bounded_receipt["status"] == "pass",
        f"blocked product attempt rejected: {bounded_receipt['reason']}",
    )
    attempt_path = attempt_root / "attempt-receipt.json"
    require(attempt_path.is_file(), "blocked product attempt did not write a receipt")
    attempt = read_json(attempt_path)
    source_tree = models_lock.get("source_tree_sha")
    lane = {
        "schema_version": SCHEMA_VERSION,
        "source_git_sha": FROZEN_LEGACY_SHA,
        "source_tree_sha": source_tree,
        "dirty_status": {"is_dirty": False, "status_short": []},
        "model_key": args.model_key,
        "backend": "metal",
        "model_revision": lane_lock["revision"],
        "model_files": {weight["path"]: weight["sha256"]},
        "hardware_id": lane_lock["hardware_id"],
        "binary_sha256": file_sha256(binary),
        "models_lock_sha256": file_sha256(root / "models.lock.json"),
        "status": "blocked",
        "current_support": False,
        "comparable": False,
        "waiver": False,
        "failure_class": policy["failure_class"],
        "reason": (
            "the frozen legacy product loader rejects this pinned Qwen3.5 GGUF architecture before inference"
            if policy["mode"] == "unsupported-architecture"
            else "the frozen legacy M3 Metal serve path violates the locked 32 GiB M1 Max headroom/swap safety gate during startup"
        ),
        "first_failure": attempt["first_failure"],
        "downstream_goal": policy["downstream_goal"],
        "implementation_path": policy["implementation_path"],
        "acceptance_path": policy["acceptance_path"],
        "downstream_acceptance_pass_line": policy["pass_line"],
        "attempted_command": argv,
        "attempted_returncode": attempt["returncode"],
        "failure_log": attempt["failure_log"]["path"],
        "collector": collector_identity(),
        "collection_spec": file_ref(spec_path, root),
        "attempt_receipt": file_ref(attempt_path, root),
        "bounded_receipt": file_ref(attempt_root / "bounded-receipt.json", root),
        "resolution_evidence": {
            "model_arg": str(model_arg),
            "weight_path": weight["path"],
            "weight_sha256": weight["sha256"],
            "weight_size_bytes": weight["size_bytes"],
            "semantic_source_root": str(semantic_root),
            "semantic_source_repo": lane_lock["semantic_source"]["repo"],
            "semantic_source_revision": lane_lock["semantic_source"]["revision"],
            "tokenizer_source_root": str(tokenizer_root),
        },
        **(
            {
                "hardware_constraint": {
                    "policy_id": locked_hardware["policy_id"],
                    "device_name": locked_hardware["device_name"],
                    "memory_bytes": locked_hardware["memory_bytes"],
                    "minimum_physical_headroom_bytes": MIN_METAL_HEADROOM_BYTES,
                    "allow_swap_growth": False,
                }
            }
            if policy["mode"] == "resource-capacity"
            else {}
        ),
    }
    lane_path = lane_root / "lane.json"
    bounded_command.atomic_write_json(lane_path, lane)
    validate_lane_evidence(root, read_json(lane_path))
    return lane_path


def _expect_reject(action: Callable[[], None], marker: str) -> None:
    try:
        action()
    except BlockedLaneError as exc:
        require(marker in str(exc), f"mutation rejected for wrong reason: {exc}")
    else:
        raise BlockedLaneError(f"mutation unexpectedly passed: {marker}")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-blocked-lane-") as temporary:
        root = Path(temporary)
        (root / "models.lock.json").write_text("{}\n", encoding="utf-8")
        fake = root / "binaries/metal/ferrum"
        fake.parent.mkdir(parents=True)
        fake.write_text(
            "#!/bin/sh\n"
            "effective=''\n"
            "while [ $# -gt 0 ]; do\n"
            '  if [ "$1" = --effective-config-json ]; then shift; effective=$1; fi\n'
            "  shift\n"
            "done\n"
            'printf \'{"schema_version":1,"parent_poison":"%s"}\\n\' "${FERRUM_BLOCKED_LANE_POISON-unset}" > "$effective"\n'
            "sleep 0.2\n"
            "echo \"Error: unsupported GGUF architecture 'qwen35'\" >&2\n"
            "exit 42\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        attempt_root = root / "correctness/m1-qwen35-4b/metal/attempt"
        attempt_root.mkdir(parents=True)
        effective = attempt_root / "effective-config.json"
        argv = [
            str(fake),
            "run",
            str(root / "model.gguf"),
            "--effective-config-json",
            str(effective),
        ]
        previous = os.environ.get("FERRUM_BLOCKED_LANE_POISON")
        os.environ["FERRUM_BLOCKED_LANE_POISON"] = "must-not-leak"
        try:
            spec = {
                "schema_version": 1,
                "artifact_type": "runtime_vnext_blocked_lane_collection_spec",
                "artifact_root": str(root),
                "attempt_root": str(attempt_root),
                "attempt_receipt_path": str(attempt_root / "attempt-receipt.json"),
                "cwd": str(root),
                "model_key": "m1-qwen35-4b",
                "mode": "unsupported-architecture",
                "failure_class": "legacy-model-backend-unsupported",
                "architecture_marker": "unsupported GGUF architecture 'qwen35'",
                "product_timeout_seconds": 5,
                "argv": argv,
                "child_environment": sanitized_child_environment(),
            }
            spec_path = attempt_root / "collection-spec.json"
            bounded_command.atomic_write_json(spec_path, spec)
            _, bounded_receipt = bounded_command.run_bounded_command(
                command=[
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--internal-execute",
                    "--spec",
                    str(spec_path),
                ],
                cwd=REPO_ROOT,
                receipt_path=attempt_root / "bounded-receipt.json",
                stdout_path=attempt_root / "collector.stdout.log",
                stderr_path=attempt_root / "collector.stderr.log",
                limits=bounded_command.Limits(120, 4, 64, 64),
            )
        finally:
            if previous is None:
                os.environ.pop("FERRUM_BLOCKED_LANE_POISON", None)
            else:
                os.environ["FERRUM_BLOCKED_LANE_POISON"] = previous
        validate_bounded_receipt(bounded_receipt)
        attempt = read_json(attempt_root / "attempt-receipt.json")
        lane = {
            "model_key": "m1-qwen35-4b",
            "attempted_command": argv,
            "attempted_returncode": 42,
            "failure_log": attempt["failure_log"]["path"],
        }
        validate_attempt_document(root, attempt, lane)
        effective_doc = read_json(effective)
        require(
            effective_doc.get("parent_poison") == "unset",
            "hostile parent FERRUM_* reached product",
        )
        poisoned = copy.deepcopy(attempt)
        poisoned["product_ferrum_keys"] = ["FERRUM_BLOCKED_LANE_POISON"]
        _expect_reject(
            lambda: validate_attempt_document(root, poisoned, lane),
            "inherited product FERRUM",
        )
        missing_start = copy.deepcopy(attempt)
        process_path = artifact_path(
            root, missing_start["process_receipt"]["path"], "process receipt"
        )
        process = read_json(process_path)
        process["process_start_marker"] = ""
        bad_process = attempt_root / "bad-process-receipt.json"
        bounded_command.atomic_write_json(bad_process, process)
        missing_start["process_receipt"] = file_ref(bad_process, root)
        _expect_reject(
            lambda: validate_attempt_document(root, missing_start, lane),
            "process start marker",
        )
        success = copy.deepcopy(attempt)
        success["returncode"] = 0
        _expect_reject(
            lambda: validate_attempt_document(root, success, lane),
            "returncode must be non-zero",
        )
        missing_signature = copy.deepcopy(attempt)
        missing_signature["failure_signature"] = "absent failure marker"
        _expect_reject(
            lambda: validate_attempt_document(root, missing_signature, lane),
            "first failure",
        )
        resource_attempt = copy.deepcopy(attempt)
        resource_argv = [str(fake), "serve", str(root / "model.gguf")]
        resource_attempt.update(
            {
                "model_key": "m3-qwen3-30b-a3b",
                "mode": "resource-capacity",
                "failure_class": "legacy-metal-unified-memory-capacity",
                "failure_signature": "swap_growth",
                "first_failure": "swap_growth: synthetic self-test resource violation",
                "argv": resource_argv,
                "argv_sha256": canonical_json_sha256(resource_argv),
            }
        )
        resource_process = read_json(
            artifact_path(root, resource_attempt["process_receipt"]["path"], "resource process receipt")
        )
        resource_process["argv"] = resource_argv
        resource_process["argv_sha256"] = canonical_json_sha256(resource_argv)
        resource_process_path = attempt_root / "resource-process-receipt.json"
        bounded_command.atomic_write_json(resource_process_path, resource_process)
        resource_attempt["process_receipt"] = file_ref(resource_process_path, root)
        before_swap = resource_attempt["memory"]["before"]["swap_used_bytes"]
        samples_path = attempt_root / "resource-memory.samples.jsonl"
        write_memory_samples(
            samples_path,
            [
                {
                    "sequence": 0,
                    "captured_at": utc_now(),
                    "captured_monotonic_ns": time.monotonic_ns(),
                    "physical_headroom_bytes": resource_attempt["memory"]["before"]["physical_headroom_bytes"],
                    "swap_used_bytes": before_swap + 4096,
                    "process_alive": True,
                    "ready_observed": False,
                }
            ],
        )
        resource_attempt["memory"].update(
            {
                "samples": file_ref(samples_path, root),
                "violation": {
                    "kind": "swap_growth",
                    "sequence": 0,
                    "initial_bytes": before_swap,
                    "observed_bytes": before_swap + 4096,
                },
            }
        )
        resource_failure_path = attempt_root / "resource-failure.log"
        resource_failure_path.write_text(resource_attempt["first_failure"] + "\n", encoding="utf-8")
        resource_attempt["failure_log"] = file_ref(resource_failure_path, root)
        resource_lane = {
            "model_key": "m3-qwen3-30b-a3b",
            "attempted_command": resource_argv,
            "attempted_returncode": 42,
            "failure_log": resource_attempt["failure_log"]["path"],
        }
        validate_attempt_document(root, resource_attempt, resource_lane)
        missing_growth = copy.deepcopy(resource_attempt)
        missing_growth["memory"]["violation"]["observed_bytes"] = before_swap
        _expect_reject(
            lambda: validate_attempt_document(root, missing_growth, resource_lane),
            "swap-growth evidence mismatch",
        )
        leaked = copy.deepcopy(bounded_receipt)
        leaked["cleanup"] = {"process_group_gone": False}
        _expect_reject(
            lambda: validate_bounded_receipt(leaked), "process group was not reaped"
        )
    print(SELFTEST_PASS_LINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--model-key", choices=sorted(BLOCKED_MODELS))
    parser.add_argument("--binary-artifact", default="binaries/metal/ferrum")
    parser.add_argument("--model-arg", type=Path)
    parser.add_argument("--semantic-source-root", type=Path)
    parser.add_argument("--tokenizer-source-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--internal-execute", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--spec", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_execute:
        if args.spec is None:
            parser.error("--internal-execute requires --spec")
        return args
    if args.self_test:
        return args
    if args.artifact_root is None or args.model_key is None:
        parser.error("collection/validation requires --artifact-root and --model-key")
    if not args.validate_only and (
        args.model_arg is None or args.semantic_source_root is None
    ):
        parser.error("collection requires --model-arg and --semantic-source-root")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.internal_execute:
            return internal_execute(args.spec)
        if args.self_test:
            self_test()
            return 0
        root = args.artifact_root.expanduser().resolve()
        lane_path = root / "correctness" / args.model_key / "metal" / "lane.json"
        if args.validate_only:
            validate_lane_evidence(root, read_json(lane_path))
        else:
            lane_path = collect(args)
    except (BlockedLaneError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"FERRUM RUNTIME VNEXT G00 BLOCKED LANE FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"{PASS_PREFIX}: {args.model_key}/metal: {lane_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
