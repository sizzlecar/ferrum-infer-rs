#!/usr/bin/env python3
"""Collect the bounded M1 profile evidence required by Runtime vNext R2.

The default mode collects one backend.  ``aggregate`` binds one CUDA and one
Metal backend manifest without re-running either lane.  This is intentionally
Ferrum-only: it never starts an external inference engine and it never uses a
hidden FERRUM_* environment variable.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_PATH = Path(__file__).resolve()
BOUNDED_COMMAND = REPO_ROOT / "scripts/release/bounded_command.py"
PROFILE_ANALYZER = REPO_ROOT / "scripts/release/analyze_ferrum_profile.py"
REPLAY_BUNDLE_GATE = REPO_ROOT / "scripts/release/request_replay_bundle_gate.py"

BACKEND_SCHEMA = "ferrum.runtime-vnext-r2-profile-backend.v1"
AGGREGATE_SCHEMA = "ferrum.runtime-vnext-r2-profile-aggregate.v1"
PASS_PREFIX = "FERRUM RUNTIME VNEXT R2 PROFILE COLLECTOR"
AGGREGATE_PASS_PREFIX = "FERRUM RUNTIME VNEXT R2 PROFILE AGGREGATE PASS:"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT R2 PROFILE COLLECTOR SELFTEST PASS"

PROFILE_REPEATS = 3
MAX_PROFILE_OVERHEAD = 0.07
HARDENING_PROFILE_OVERHEAD = 0.02
MIN_CUDA_STAGE_COVERAGE = 0.90
MIN_CUDA_DEVICE_ATTRIBUTION_COVERAGE = 0.80
MAX_CLOCK_CONVERSION_ERROR_FRACTION = 0.005

VNEXT_CLOCK_SOURCE_ATTRIBUTE = "vnext_monotonic_clock_source"
VNEXT_WALL_ANCHOR_ATTRIBUTE = "vnext_monotonic_wall_anchor_unix_nanos"
VNEXT_CLOCK_ERROR_ATTRIBUTE = "vnext_clock_conversion_max_error_nanos"
ENGINE_DECODE_STAGE_INTERVALS_ATTRIBUTE = "engine_decode_stage_intervals"
ENGINE_DECODE_STAGE_COUNT_ATTRIBUTE = "engine_decode_stage_interval_count"
ENGINE_DECODE_STAGE_KINDS = {
    "decode_scheduling",
    "decode_execution",
    "decode_postprocess",
}

# These limits are fixed independently from model size, request count, GPU
# capacity, or user concurrency.  They are applied before every child spawn.
MAX_PROCESSES = 8
MAX_GROUP_THREADS = 64
MAX_PER_PROCESS_THREADS = 64

OVERHEAD_PROMPT = (
    "Write the integers from 1 through 64, separated by single spaces, "
    "and output nothing else."
)
OVERHEAD_MAX_TOKENS = 128
DIAGNOSTIC_PROMPT = "Reply with exactly: PROFILE-OK"
DIAGNOSTIC_MAX_TOKENS = 16

BAD_OUTPUT_MARKERS = ("<unk>", "[PAD", "\ufffd", "Ã", "Â")
METAL_TIMING_UNAVAILABLE_REASONS = {
    "backendmeasurementfailed",
    "backendunsupported",
    "backend_unsupported",
}
PRODUCT_SOURCE_PATHS = (
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "ferrum.toml",
    "crates",
    "native-operators",
)


class CollectorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectorError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required artifact is missing: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CollectorError(f"cannot read JSON object {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def relative_to(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise CollectorError(f"artifact escapes output root: {path}") from error


def git_output(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise CollectorError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()[:512]}"
        )
    return result.stdout.strip()


@functools.lru_cache(maxsize=None)
def reviewed_collector_sha256s() -> frozenset[str]:
    relative_path = COLLECTOR_PATH.relative_to(REPO_ROOT).as_posix()
    commits = git_output(["log", "--format=%H", "--", relative_path]).splitlines()
    require(commits, "profile collector has no Git history")
    digests: set[str] = set()
    for commit in commits:
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            digests.add(hashlib.sha256(result.stdout).hexdigest())
    require(digests, "profile collector has no reviewed Git blobs")
    return frozenset(digests)


def require_reviewed_collector_source(source: dict[str, Any], label: str) -> None:
    relative_path = COLLECTOR_PATH.relative_to(REPO_ROOT).as_posix()
    recorded_sha256 = source.get("collector_sha256")
    require(
        source.get("collector_path") == relative_path
        and (
            recorded_sha256 == sha256_file(COLLECTOR_PATH)
            or recorded_sha256 in reviewed_collector_sha256s()
        ),
        f"{label}: collector is not a reviewed Git-history source",
    )


def source_identity() -> dict[str, Any]:
    status = git_output(["status", "--short", "--untracked-files=all"]).splitlines()
    product_listing = git_output(
        ["ls-tree", "-r", "HEAD", "--", *PRODUCT_SOURCE_PATHS]
    )
    product_entries = product_listing.splitlines() if product_listing else []
    return {
        "git_sha": git_output(["rev-parse", "HEAD"]),
        "tree_sha": git_output(["rev-parse", "HEAD^{tree}"]),
        "dirty_status": {
            "is_dirty": bool(status),
            "status_short": status,
        },
        "collector_path": "scripts/release/runtime_vnext_r2_profile_collector.py",
        "collector_sha256": sha256_file(COLLECTOR_PATH),
        "profile_analyzer_sha256": sha256_file(PROFILE_ANALYZER),
        "replay_bundle_gate_sha256": sha256_file(REPLAY_BUNDLE_GATE),
        "bounded_command_sha256": sha256_file(BOUNDED_COMMAND),
        "product_source_closure": {
            "pathspecs": list(PRODUCT_SOURCE_PATHS),
            "entry_count": len(product_entries),
            "git_tree_listing_sha256": hashlib.sha256(
                product_listing.encode("utf-8")
            ).hexdigest(),
        },
    }


def path_closure_identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    require(path.exists(), f"input path does not exist: {path}")
    if path.is_file():
        identity = file_identity(path)
        identity.update({"kind": "file", "closure_sha256": identity["sha256"]})
        return identity
    require(path.is_dir(), f"input path is neither file nor directory: {path}")
    files: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not item.is_file():
            continue
        row = {
            "path": item.relative_to(path).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        if item.is_symlink():
            row["symlink_target"] = os.readlink(item)
        files.append(row)
    require(files, f"input directory has no files: {path}")
    return {
        "kind": "directory",
        "path": str(path),
        "file_count": len(files),
        "files": files,
        "closure_sha256": canonical_sha256(files),
    }


def lexical_absolute_path(path: Path) -> Path:
    """Make a path absolute without dereferencing its final symlink."""

    return Path(os.path.abspath(os.path.expanduser(str(path))))


def sanitized_environment() -> dict[str, str]:
    environment: dict[str, str] = {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "LC_ALL": "C",
        "NO_COLOR": "1",
        "PATH": os.environ.get(
            "PATH", "/usr/local/cuda/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        ),
    }
    for key in (
        "CUDA_HOME",
        "CUDA_PATH",
        "DYLD_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TMPDIR",
    ):
        value = os.environ.get(key)
        if value:
            environment[key] = value
    require(
        not any(key.startswith("FERRUM_") for key in environment),
        "sanitized environment unexpectedly contains a FERRUM_* variable",
    )
    return environment


def with_sanitized_environment(command: Sequence[str], environment: dict[str, str]) -> list[str]:
    env_binary = Path("/usr/bin/env")
    require(env_binary.is_file(), "/usr/bin/env is required")
    return [
        str(env_binary),
        "-i",
        *[f"{key}={environment[key]}" for key in sorted(environment)],
        *command,
    ]


def validate_receipt(
    receipt_path: Path,
    *,
    expected_command: Sequence[str],
    expected_timeout: float,
) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    require(
        receipt.get("schema") == "ferrum.bounded-command-receipt.v1",
        f"{receipt_path}: bounded receipt schema mismatch",
    )
    require(receipt.get("status") == "pass", f"{receipt_path}: command did not pass")
    require(receipt.get("rc") == 0, f"{receipt_path}: command return code is not zero")
    require(
        receipt.get("command") == list(expected_command),
        f"{receipt_path}: command differs from the declared action",
    )
    limits = receipt.get("limits")
    require(isinstance(limits, dict), f"{receipt_path}: missing bounded limits")
    require(
        float(limits.get("wall_timeout_seconds", -1)) == float(expected_timeout),
        f"{receipt_path}: hard deadline differs from the declared action",
    )
    require(
        limits.get("max_processes") == MAX_PROCESSES
        and limits.get("max_group_threads") == MAX_GROUP_THREADS
        and limits.get("max_per_process_threads") == MAX_PER_PROCESS_THREADS,
        f"{receipt_path}: worker limits are not the fixed collector limits",
    )
    require(
        receipt.get("cleanup", {}).get("process_group_gone") is True,
        f"{receipt_path}: process group cleanup was not proven",
    )
    for key in ("stdout", "stderr"):
        raw = receipt.get(key)
        require(isinstance(raw, dict), f"{receipt_path}: missing {key} identity")
        artifact = Path(str(raw.get("path", "")))
        require(artifact.is_file(), f"{receipt_path}: missing {key} log {artifact}")
        require(
            raw.get("size_bytes") == artifact.stat().st_size
            and raw.get("sha256") == sha256_file(artifact),
            f"{receipt_path}: {key} log identity mismatch",
        )
    return receipt


def archive_failed_action(action_dir: Path) -> None:
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = action_dir.with_name(f"{action_dir.name}.failed-{suffix}")
    ordinal = 1
    while candidate.exists():
        candidate = action_dir.with_name(
            f"{action_dir.name}.failed-{suffix}-{ordinal:02d}"
        )
        ordinal += 1
    action_dir.rename(candidate)


def run_bounded_action(
    action_dir: Path,
    *,
    product_command: Sequence[str],
    environment: dict[str, str],
    expected_duration_seconds: int,
    hard_deadline_seconds: int,
    progress_signals: Sequence[str],
    resume: bool,
) -> dict[str, Any]:
    command = with_sanitized_environment(product_command, environment)
    receipt_path = action_dir / "bounded.receipt.json"
    plan_path = action_dir / "command.plan.json"
    if action_dir.exists():
        if not resume:
            raise CollectorError(f"action output already exists; pass --resume: {action_dir}")
        if receipt_path.is_file() and plan_path.is_file():
            plan = read_json(plan_path)
            if (
                plan.get("command") == command
                and plan.get("hard_deadline_seconds") == hard_deadline_seconds
            ):
                try:
                    receipt = validate_receipt(
                        receipt_path,
                        expected_command=command,
                        expected_timeout=float(hard_deadline_seconds),
                    )
                except CollectorError:
                    pass
                else:
                    return {
                        "action_dir": action_dir,
                        "plan": plan,
                        "receipt": receipt,
                        "reused": True,
                    }
        archive_failed_action(action_dir)

    action_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    plan = {
        "schema": "ferrum.long-command-plan.v1",
        "status": "running",
        "command": command,
        "product_command": list(product_command),
        "cwd": str(REPO_ROOT),
        "environment": environment,
        "expected_duration_seconds": expected_duration_seconds,
        "hard_deadline_seconds": hard_deadline_seconds,
        "started_at": started.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "hard_deadline_at": (started + timedelta(seconds=hard_deadline_seconds))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "observable_progress_signals": list(progress_signals),
        "stop_condition": "bounded command exits, reaches its hard deadline, or violates a fixed worker limit",
    }
    plan["command_fingerprint"] = canonical_sha256(
        {
            "command": command,
            "cwd": str(REPO_ROOT),
            "expected_duration_seconds": expected_duration_seconds,
            "hard_deadline_seconds": hard_deadline_seconds,
            "observable_progress_signals": list(progress_signals),
        }
    )
    atomic_write_json(plan_path, plan)

    wrapper_command = [
        sys.executable,
        str(BOUNDED_COMMAND),
        "--receipt",
        str(receipt_path),
        "--stdout-log",
        str(action_dir / "stdout.log"),
        "--stderr-log",
        str(action_dir / "stderr.log"),
        "--cwd",
        str(REPO_ROOT),
        "--wall-timeout-seconds",
        str(hard_deadline_seconds),
        "--max-processes",
        str(MAX_PROCESSES),
        "--max-group-threads",
        str(MAX_GROUP_THREADS),
        "--max-per-process-threads",
        str(MAX_PER_PROCESS_THREADS),
        "--sample-interval-seconds",
        "0.1",
        "--term-grace-seconds",
        "5",
        "--",
        *command,
    ]
    result = subprocess.run(wrapper_command, cwd=REPO_ROOT, check=False)
    require(receipt_path.is_file(), f"bounded command produced no receipt: {action_dir}")
    receipt = read_json(receipt_path)
    plan.update(
        {
            "status": "pass" if result.returncode == 0 else "fail",
            "finished_at": utc_now(),
            "bounded_wrapper_returncode": result.returncode,
            "receipt": relative_to(action_dir, receipt_path),
            "receipt_status": receipt.get("status"),
            "receipt_reason": receipt.get("reason"),
        }
    )
    atomic_write_json(plan_path, plan)
    if result.returncode != 0:
        raise CollectorError(
            f"bounded action failed: {action_dir} status={receipt.get('status')} "
            f"reason={receipt.get('reason')}"
        )
    receipt = validate_receipt(
        receipt_path,
        expected_command=command,
        expected_timeout=float(hard_deadline_seconds),
    )
    return {
        "action_dir": action_dir,
        "plan": plan,
        "receipt": receipt,
        "reused": False,
    }


def read_stdout_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise CollectorError(f"{path}:{line_number}: invalid JSONL: {error}") from error
        require(isinstance(value, dict), f"{path}:{line_number}: expected JSON object")
        events.append(value)
    require(events, f"{path}: no product JSONL events")
    return events


def validate_run_stdout(path: Path) -> dict[str, Any]:
    events = read_stdout_events(path)
    require(sum(event.get("event") == "ready" for event in events) == 1, f"{path}: ready count mismatch")
    require(sum(event.get("event") == "exit" for event in events) == 1, f"{path}: exit count mismatch")
    ready = next(event for event in events if event.get("event") == "ready")
    resolved_model = ready.get("resolved_model") or ready.get("model")
    normalized_model = str(resolved_model).casefold().replace("_", ".")
    require(
        re.search(r"qwen3[.-]?5.*4b", normalized_model) is not None,
        f"{path}: R2 profile collector requires M1 Qwen3.5-4B, observed {resolved_model!r}",
    )
    assistants = [event for event in events if event.get("event") == "assistant"]
    require(len(assistants) == 1, f"{path}: assistant terminal count mismatch")
    assistant = assistants[0]
    content = assistant.get("content")
    require(isinstance(content, str) and content.strip(), f"{path}: empty assistant output")
    lowered = content.casefold()
    for marker in BAD_OUTPUT_MARKERS:
        require(marker.casefold() not in lowered, f"{path}: bad output marker {marker!r}")
    token_count = assistant.get("n_tokens")
    duration_ms = assistant.get("ms")
    usage = assistant.get("usage")
    require(type(token_count) is int and token_count > 0, f"{path}: invalid output token count")
    require(
        isinstance(duration_ms, (int, float))
        and not isinstance(duration_ms, bool)
        and math.isfinite(float(duration_ms))
        and float(duration_ms) > 0,
        f"{path}: invalid generation duration",
    )
    require(isinstance(usage, dict), f"{path}: missing usage")
    require(
        usage.get("completion_tokens") == token_count,
        f"{path}: usage completion token mismatch",
    )
    require(
        assistant.get("finish_reason") in {"stop", "length"},
        f"{path}: invalid finish reason",
    )
    return {
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "raw_text_sha256": assistant.get("raw_text_sha256"),
        "completion_tokens": token_count,
        "prompt_tokens": usage.get("prompt_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "duration_ms": float(duration_ms),
        "throughput_tokens_per_second": token_count * 1000.0 / float(duration_ms),
        "finish_reason": assistant.get("finish_reason"),
        "resolved_model": resolved_model,
    }


def validate_effective_config(path: Path, backend: str) -> dict[str, Any]:
    config = read_json(path)
    require(config.get("backend") == backend, f"{path}: effective backend mismatch")
    entries = config.get("entries")
    require(isinstance(entries, list), f"{path}: effective config entries are missing")
    env_source_entries = sorted(
        [
            {
                "key": str(row.get("key")),
                "effective_value": row.get("effective_value"),
            }
            for row in entries
            if isinstance(row, dict)
            and row.get("source") == "env"
            and str(row.get("key", "")).startswith("FERRUM_")
        ],
        key=lambda row: row["key"],
    )
    # Ferrum's typed auto-sizing layer currently materializes a bounded set of
    # resolved defaults through its internal compatibility environment and
    # labels those rows ``source=env``.  The launch command is still executed
    # through ``env -i`` and contains zero FERRUM_* names, so these rows are
    # recorded as runtime-derived evidence rather than misclassified as caller
    # hidden configuration.
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "backend": backend,
        "launch_hidden_ferrum_env_count": 0,
        "runtime_derived_env_compatibility_entries": env_source_entries,
    }


def run_product_command(
    *,
    root: Path,
    name: str,
    backend: str,
    binary: Path,
    model: Path,
    semantic_model: Path,
    mode: str,
    prompt: str,
    max_tokens: int,
    environment: dict[str, str],
    resume: bool,
) -> dict[str, Any]:
    action_dir = root / "raw" / name
    profile_path = action_dir / "profile.jsonl"
    effective_config_path = action_dir / "effective-config.json"
    request_dump_dir = action_dir / "request-dump"
    product_command = [
        str(binary),
        "run",
        str(model),
        "--backend",
        backend,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--disable-thinking",
        "--temperature",
        "0",
        "--repeat-penalty",
        "1.0",
        "--output-format",
        "jsonl",
        "--semantic-source",
        str(semantic_model),
        "--tokenizer-source",
        str(semantic_model),
        "--profile-detail",
        mode,
        "--effective-config-json",
        str(effective_config_path),
    ]
    if mode != "off":
        product_command.extend(
            [
                "--profile-sample-rate",
                "1.0",
                "--profile-jsonl",
                str(profile_path),
            ]
        )
    if mode in {"replay", "full"}:
        product_command.extend(["--request-dump-dir", str(request_dump_dir)])

    if mode == "full":
        expected_seconds, hard_seconds = 600, 1800
    elif mode == "replay":
        expected_seconds, hard_seconds = 300, 900
    else:
        expected_seconds, hard_seconds = 180, 600
    progress = ["stdout.log bytes", "stderr.log bytes"]
    if mode != "off":
        progress.append("profile.jsonl bytes/events")
    if mode in {"replay", "full"}:
        progress.append("request-dump file count")
    action = run_bounded_action(
        action_dir,
        product_command=product_command,
        environment=environment,
        expected_duration_seconds=expected_seconds,
        hard_deadline_seconds=hard_seconds,
        progress_signals=progress,
        resume=resume,
    )
    stdout_path = Path(action["receipt"]["stdout"]["path"])
    summary = validate_run_stdout(stdout_path)
    effective = validate_effective_config(effective_config_path, backend)
    if mode == "off":
        require(not profile_path.exists(), f"{name}: profile-off unexpectedly created a profile")
    else:
        require(profile_path.is_file() and profile_path.stat().st_size > 0, f"{name}: profile is empty")
    if mode in {"replay", "full"}:
        require(request_dump_dir.is_dir(), f"{name}: request replay bundle is missing")
    return {
        "name": name,
        "mode": mode,
        "action_dir": relative_to(root, action_dir),
        "command": product_command,
        "sanitized_command": action["receipt"]["command"],
        "receipt": relative_to(root, action_dir / "bounded.receipt.json"),
        "stdout": relative_to(root, stdout_path),
        "stderr": relative_to(root, Path(action["receipt"]["stderr"]["path"])),
        "profile": relative_to(root, profile_path) if mode != "off" else None,
        "effective_config": effective,
        "request_dump": relative_to(root, request_dump_dir)
        if mode in {"replay", "full"}
        else None,
        "run_summary": summary,
        "reused": action["reused"],
    }


def coefficient_of_variation(values: Sequence[float]) -> float:
    require(bool(values), "cannot calculate CV of an empty sample")
    mean = statistics.fmean(values)
    require(mean > 0, "cannot calculate CV with a non-positive mean")
    return statistics.pstdev(values) / mean


def validate_overhead(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    off = [row for row in rows if row["mode"] == "off"]
    basic = [row for row in rows if row["mode"] == "basic"]
    require(len(off) == PROFILE_REPEATS, "profile-off repeat count mismatch")
    require(len(basic) == PROFILE_REPEATS, "profile-basic repeat count mismatch")
    fingerprints = {
        (
            row["run_summary"]["content_sha256"],
            row["run_summary"]["raw_text_sha256"],
            row["run_summary"]["completion_tokens"],
            row["run_summary"]["finish_reason"],
        )
        for row in [*off, *basic]
    }
    require(
        len(fingerprints) == 1,
        "profile-off/basic changed deterministic output or token count",
    )
    off_ms = [row["run_summary"]["duration_ms"] for row in off]
    basic_ms = [row["run_summary"]["duration_ms"] for row in basic]
    off_tps = [row["run_summary"]["throughput_tokens_per_second"] for row in off]
    basic_tps = [row["run_summary"]["throughput_tokens_per_second"] for row in basic]
    median_duration_overhead = statistics.median(basic_ms) / statistics.median(off_ms) - 1.0
    mean_duration_overhead = statistics.fmean(basic_ms) / statistics.fmean(off_ms) - 1.0
    require(
        median_duration_overhead <= MAX_PROFILE_OVERHEAD,
        f"basic profile median overhead {median_duration_overhead:.6f} exceeds {MAX_PROFILE_OVERHEAD:.2f}",
    )
    return {
        "status": "pass",
        "truth_mode": "off",
        "repeat_count_each": PROFILE_REPEATS,
        "metric_boundary": "ferrum_run_assistant_generation_duration",
        "off_duration_ms": off_ms,
        "basic_duration_ms": basic_ms,
        "off_throughput_tokens_per_second": off_tps,
        "basic_throughput_tokens_per_second": basic_tps,
        "off_duration_cv": coefficient_of_variation(off_ms),
        "basic_duration_cv": coefficient_of_variation(basic_ms),
        "median_duration_overhead": median_duration_overhead,
        "mean_duration_overhead": mean_duration_overhead,
        "release_limit": MAX_PROFILE_OVERHEAD,
        "hardening_target": HARDENING_PROFILE_OVERHEAD,
        "hardening_target_met": median_duration_overhead <= HARDENING_PROFILE_OVERHEAD,
        "output_fingerprint": {
            "content_sha256": next(iter(fingerprints))[0],
            "raw_text_sha256": next(iter(fingerprints))[1],
            "completion_tokens": next(iter(fingerprints))[2],
            "finish_reason": next(iter(fingerprints))[3],
        },
    }


def read_profile_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise CollectorError(f"{path}:{line_number}: malformed profile JSON: {error}") from error
            require(isinstance(event, dict), f"{path}:{line_number}: profile event is not an object")
            events.append(event)
    require(events, f"{path}: profile has no events")
    return events


def event_attributes(event: dict[str, Any]) -> dict[str, Any]:
    attributes = event.get("attributes")
    return attributes if isinstance(attributes, dict) else {}


def non_empty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def merge_interval_length(intervals: Iterable[tuple[int, int]]) -> int:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def validate_device_interval_contract(
    intervals: Sequence[tuple[int, int]], *, elapsed_ns: int, context: str
) -> int:
    """Validate the ordering and exact-duration contract emitted by Rust."""
    require(intervals, f"{context} has no device subwork intervals")
    accounted_ns = 0
    previous_end: int | None = None
    for index, (start, end) in enumerate(intervals):
        require(
            type(start) is int and type(end) is int and 0 <= start < end,
            f"{context} device subwork interval {index} has invalid bounds",
        )
        require(
            previous_end is None or previous_end <= start,
            f"{context} device subwork intervals are not ordered and non-overlapping",
        )
        accounted_ns += end - start
        previous_end = end
    require(
        accounted_ns == elapsed_ns,
        f"{context} device subwork intervals account for {accounted_ns} ns, "
        f"not elapsed {elapsed_ns} ns",
    )
    return accounted_ns


def unavailable_stage_timing(
    *,
    reason: str,
    decode_wall: int,
    pair_count: int,
    missing_fields: Sequence[str] = (),
    clock_conversion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "diagnostic_only",
        "formal_coverage_eligible": False,
        "unavailable_reason": reason,
        "missing_fields": list(missing_fields),
        "decode_wall_ns": decode_wall,
        "stage_accounted_union_ns": None,
        "unattributed_ns": None,
        "coverage": None,
        "node_interval_pairs": pair_count,
        "clock_source": "unjoined",
        "boundary_source": "engine_wall_anchor_plus_token_offsets",
        "legacy_mixed_clock_coverage_omitted": True,
        "clock_conversion": clock_conversion,
    }


def calculate_stage_coverage(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    terminal = next(
        (event for event in events if event.get("phase") == "actual_run_generation"),
        None,
    )
    require(isinstance(terminal, dict), "full profile lacks actual_run_generation")
    terminal_attributes = event_attributes(terminal)
    request_id = terminal_attributes.get("execution_request_id")
    commits = terminal_attributes.get("engine_token_commit_nanos_since_request_start")
    decode_ready = terminal_attributes.get("engine_decode_ready_nanos_since_request_start")
    require(non_empty(request_id), "full profile terminal lacks execution_request_id")
    require(
        isinstance(commits, list)
        and commits
        and all(type(value) is int and value >= 0 for value in commits),
        "full profile lacks typed engine token commits",
    )
    require(type(decode_ready) is int and decode_ready >= 0, "full profile lacks decode-ready timing")
    decode_wall = commits[-1] - decode_ready
    require(decode_wall > 0, "decode wall interval is not positive")
    raw_engine_intervals = terminal_attributes.get(ENGINE_DECODE_STAGE_INTERVALS_ATTRIBUTE)
    engine_interval_count = terminal_attributes.get(ENGINE_DECODE_STAGE_COUNT_ATTRIBUTE)
    require(
        isinstance(raw_engine_intervals, list),
        "full profile lacks typed engine decode stage intervals",
    )
    require(
        type(engine_interval_count) is int
        and engine_interval_count == len(raw_engine_intervals),
        "engine decode stage interval count differs from typed rows",
    )
    require(engine_interval_count > 0, "full profile has no engine decode stage intervals")
    engine_intervals_by_kind: dict[str, list[tuple[int, int]]] = {
        kind: [] for kind in sorted(ENGINE_DECODE_STAGE_KINDS)
    }
    previous_engine_start: int | None = None
    for index, row in enumerate(raw_engine_intervals):
        require(
            isinstance(row, dict)
            and set(row)
            == {
                "stage",
                "start_nanos_since_request_start",
                "end_nanos_since_request_start",
            },
            f"engine decode stage interval {index} has invalid shape",
        )
        stage = row.get("stage")
        start = row.get("start_nanos_since_request_start")
        end = row.get("end_nanos_since_request_start")
        require(
            stage in ENGINE_DECODE_STAGE_KINDS,
            f"engine decode stage interval {index} has unsupported stage",
        )
        require(
            type(start) is int and start >= 0 and type(end) is int and end >= start,
            f"engine decode stage interval {index} has invalid bounds",
        )
        require(
            previous_engine_start is None or start >= previous_engine_start,
            "engine decode stage intervals are not ordered by start",
        )
        previous_engine_start = start
        engine_intervals_by_kind[str(stage)].append((start, end))
    require(
        all(engine_intervals_by_kind.values()),
        "full profile does not measure every required engine decode stage",
    )
    accepted = next(
        (
            event
            for event in events
            if event.get("phase") == "vnext.request_accepted"
            and event.get("request_id") == request_id
        ),
        None,
    )
    require(isinstance(accepted, dict), "full profile cannot join request acceptance")
    accepted_attributes = event_attributes(accepted)
    request_origin = accepted_attributes.get("monotonic_nanos_since_run_start")
    require(type(request_origin) is int and request_origin >= 0, "request lacks monotonic origin")

    starts: dict[tuple[Any, Any, Any, Any], tuple[int, str]] = {}
    monotonic_intervals: list[tuple[int, int]] = []
    pair_count = 0
    for event in events:
        if event.get("phase") not in {"vnext.node_started", "vnext.node_retired"}:
            continue
        attributes = event_attributes(event)
        identity = attributes.get("execution_identity")
        require(isinstance(identity, dict), "node event lacks typed execution_identity")
        event_request_id = event.get("request_id")
        identity_request_id = identity.get("request_id")
        require(non_empty(event_request_id), "node event lacks request_id")
        require(non_empty(identity_request_id), "node identity lacks request_id")
        require(
            event_request_id == identity_request_id,
            "node event request_id differs from its execution identity",
        )
        if event_request_id != request_id:
            continue
        run_id = identity.get("run_id")
        frame_id = identity.get("frame_id")
        node_invocation_id = identity.get("node_invocation_id")
        span_id = identity.get("span_id")
        require(non_empty(run_id), "node identity lacks run_id")
        require(type(frame_id) is int and frame_id > 0, "node identity lacks frame_id")
        require(
            type(node_invocation_id) is int and node_invocation_id > 0,
            "node identity lacks node_invocation_id",
        )
        require(non_empty(span_id), "node identity lacks span_id")
        key = (
            run_id,
            identity_request_id,
            frame_id,
            node_invocation_id,
        )
        timestamp = attributes.get("monotonic_nanos_since_run_start")
        require(type(timestamp) is int and timestamp >= 0, "node event lacks monotonic timestamp")
        if event.get("phase") == "vnext.node_started":
            require(key not in starts, f"duplicate node start identity: {key}")
            starts[key] = (timestamp, str(span_id))
            continue
        require(key in starts, f"node retirement lacks start: {key}")
        start, started_span_id = starts.pop(key)
        require(
            span_id == started_span_id,
            f"node retirement span differs from start: {key}",
        )
        require(timestamp >= start, f"node interval is reversed: {key}")
        pair_count += 1
        monotonic_intervals.append((start, timestamp))
    require(not starts, "full profile has unterminated node spans")

    engine_clock_source = terminal_attributes.get("engine_token_clock_source")
    engine_wall_anchor = terminal_attributes.get("engine_token_wall_anchor_unix_nanos")
    engine_max_error = terminal_attributes.get("clock_conversion_max_error_nanos")
    vnext_clock_source = accepted_attributes.get(VNEXT_CLOCK_SOURCE_ATTRIBUTE)
    vnext_wall_anchor = accepted_attributes.get(VNEXT_WALL_ANCHOR_ATTRIBUTE)
    vnext_max_error = accepted_attributes.get(VNEXT_CLOCK_ERROR_ATTRIBUTE)
    required_clock_fields = (
        ("actual_run_generation.attributes.engine_token_clock_source", engine_clock_source),
        (
            "actual_run_generation.attributes.engine_token_wall_anchor_unix_nanos",
            engine_wall_anchor,
        ),
        (
            "actual_run_generation.attributes.clock_conversion_max_error_nanos",
            engine_max_error,
        ),
        (
            f"vnext.request_accepted.attributes.{VNEXT_CLOCK_SOURCE_ATTRIBUTE}",
            vnext_clock_source,
        ),
        (
            f"vnext.request_accepted.attributes.{VNEXT_WALL_ANCHOR_ATTRIBUTE}",
            vnext_wall_anchor,
        ),
        (
            f"vnext.request_accepted.attributes.{VNEXT_CLOCK_ERROR_ATTRIBUTE}",
            vnext_max_error,
        ),
    )
    missing_clock_fields = [name for name, value in required_clock_fields if value is None]
    if missing_clock_fields:
        return unavailable_stage_timing(
            reason="missing_bounded_clock_domain_conversion",
            decode_wall=decode_wall,
            pair_count=pair_count,
            missing_fields=missing_clock_fields,
            clock_conversion={
                "engine_clock_source": engine_clock_source,
                "vnext_clock_source": vnext_clock_source,
            },
        )

    require(engine_clock_source == "rust_std_instant", "engine token clock source is unsupported")
    require(vnext_clock_source == "rust_std_instant", "vNext event clock source is unsupported")
    require(
        type(engine_wall_anchor) is int and engine_wall_anchor > 0,
        "engine token wall anchor is invalid",
    )
    require(
        type(vnext_wall_anchor) is int and vnext_wall_anchor > 0,
        "vNext monotonic wall anchor is invalid",
    )
    require(
        type(engine_max_error) is int and engine_max_error >= 0,
        "engine clock conversion error is invalid",
    )
    require(
        type(vnext_max_error) is int and vnext_max_error >= 0,
        "vNext clock conversion error is invalid",
    )
    relative_max_error = engine_max_error + vnext_max_error
    conversion = {
        "event_source": "vnext.monotonic_nanos_since_run_start",
        "common_domain": "unix_nanos",
        "engine_clock_source": engine_clock_source,
        "engine_wall_anchor_unix_nanos": engine_wall_anchor,
        "engine_max_error_nanos": engine_max_error,
        "vnext_clock_source": vnext_clock_source,
        "vnext_wall_anchor_unix_nanos": vnext_wall_anchor,
        "vnext_max_error_nanos": vnext_max_error,
        "relative_max_error_nanos": relative_max_error,
        "relative_error_ppm": relative_max_error * 1_000_000 // decode_wall,
    }
    if relative_max_error > decode_wall * MAX_CLOCK_CONVERSION_ERROR_FRACTION:
        return unavailable_stage_timing(
            reason="clock_conversion_error_exceeds_formal_limit",
            decode_wall=decode_wall,
            pair_count=pair_count,
            clock_conversion=conversion,
        )

    decode_start = engine_wall_anchor + decode_ready
    decode_end = engine_wall_anchor + commits[-1]
    node_intervals = [
        (
            max(vnext_wall_anchor + start, decode_start),
            min(vnext_wall_anchor + end, decode_end),
        )
        for start, end in monotonic_intervals
    ]
    engine_intervals = [
        (
            max(engine_wall_anchor + start, decode_start),
            min(engine_wall_anchor + end, decode_end),
        )
        for rows in engine_intervals_by_kind.values()
        for start, end in rows
    ]
    intervals = node_intervals + engine_intervals
    accounted = merge_interval_length(intervals)
    coverage = accounted / decode_wall
    require(coverage <= 1.0 + 1e-9, "stage coverage exceeds 100%")
    return {
        "status": "measured",
        "formal_coverage_eligible": True,
        "decode_wall_ns": decode_wall,
        "stage_accounted_union_ns": accounted,
        "unattributed_ns": decode_wall - accounted,
        "coverage": coverage,
        "node_interval_pairs": pair_count,
        "engine_stage_interval_count": engine_interval_count,
        "engine_stage_kinds": sorted(engine_intervals_by_kind),
        "engine_stage_accounted_union_ns": merge_interval_length(engine_intervals),
        "clock_source": "unix_nanos_from_bounded_wall_anchors",
        "boundary_source": "engine_wall_anchor_plus_token_offsets",
        "legacy_mixed_clock_coverage_omitted": False,
        "clock_conversion": conversion,
    }


def validate_identity_and_device_contract(
    *,
    backend: str,
    basic_events: Sequence[Sequence[dict[str, Any]]],
    replay_events: Sequence[dict[str, Any]],
    full_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    all_modes: list[tuple[str, Sequence[dict[str, Any]]]] = [
        *[("basic", events) for events in basic_events],
        ("replay", replay_events),
        ("full", full_events),
    ]
    plan_sets: dict[str, list[str]] = {}
    mode_identity_rows: dict[str, int] = {}
    for mode, events in all_modes:
        observed_modes = {
            event_attributes(event).get("profile_detail")
            for event in events
            if event_attributes(event).get("profile_detail") is not None
        }
        require(observed_modes == {mode}, f"{mode} profile detail mismatch: {observed_modes}")
        require(
            not any(event.get("status") == "failure" for event in events),
            f"{mode} profile contains a failure event",
        )
        plans = sorted(
            {
                str(event_attributes(event)["plan_id"])
                for event in events
                if non_empty(event_attributes(event).get("plan_id"))
            }
        )
        require(len(plans) == 1, f"{mode} profile does not resolve exactly one plan")
        previous_plans = plan_sets.setdefault(mode, plans)
        require(previous_plans == plans, f"{mode} repeats resolved different plans")
        joined_identity_rows = 0
        for event in events:
            attributes = event_attributes(event)
            identity = attributes.get("execution_identity")
            identity = identity if isinstance(identity, dict) else {}
            node_id = attributes.get("node_id") or identity.get("node_id")
            operation_id = attributes.get("operation_id") or identity.get("operation_id")
            provider_id = attributes.get("provider_id") or identity.get("provider_id")
            resource_pool_id = attributes.get("resource_pool_id")
            if resource_pool_id is None:
                resource_pool_id = identity.get("resource_pool_id")
            if (
                non_empty(node_id)
                and non_empty(operation_id)
                and non_empty(provider_id)
                and resource_pool_id is not None
            ):
                joined_identity_rows += 1
        require(joined_identity_rows > 0, f"{mode} profile lacks node/op/resource/provider correlation")
        mode_identity_rows[mode] = mode_identity_rows.get(mode, 0) + joined_identity_rows
    unique_plans = {tuple(plans) for plans in plan_sets.values()}
    require(len(unique_plans) == 1, "basic/replay/full plan identity differs")

    node_mapping: dict[str, tuple[str, str, str]] = {}
    resource_ids: set[str] = set()
    for event in full_events:
        if event.get("phase") not in {"vnext.node_started", "vnext.node_retired"}:
            continue
        attributes = event_attributes(event)
        identity = attributes.get("execution_identity")
        require(isinstance(identity, dict), "full node event lacks execution_identity")
        node_id = identity.get("node_id")
        operation_id = identity.get("operation_id")
        provider_id = identity.get("provider_id")
        resource_pool_id = identity.get("resource_pool_id")
        require(
            all(non_empty(value) for value in (node_id, operation_id, provider_id)),
            "full node event lacks node/operation/provider identity",
        )
        require(resource_pool_id is not None, "full node event lacks resource pool identity")
        mapping = (str(operation_id), str(provider_id), str(resource_pool_id))
        previous = node_mapping.setdefault(str(node_id), mapping)
        require(previous == mapping, f"node identity changed within full profile: {node_id}")
        resource_ids.add(str(resource_pool_id))
    require(node_mapping, "full profile has no typed node mapping")
    require(resource_ids, "full profile has no resource identity")

    native_rows = [
        event for event in full_events if event.get("phase") == "vnext.device_native_work"
    ]
    require(native_rows, "full profile has no native work attribution")
    native_ids: set[str] = set()
    compute_dispatches = 0
    transfer_dispatches = 0
    native_node_rows = 0
    for event in native_rows:
        attributes = event_attributes(event)
        shape = event.get("shape")
        require(isinstance(shape, dict), "native work row lacks shape")
        native_op_id = attributes.get("native_op_id")
        require(non_empty(native_op_id), "native work row lacks native_op_id")
        native_ids.add(str(native_op_id))
        compute = shape.get("physical_compute_dispatch_count")
        transfer = shape.get("physical_transfer_command_count")
        require(type(compute) is int and compute >= 0, "native work compute dispatch count is invalid")
        require(type(transfer) is int and transfer >= 0, "native work transfer dispatch count is invalid")
        compute_dispatches += compute
        transfer_dispatches += transfer
        if attributes.get("attribution_scope") == "node":
            node_id = attributes.get("node_id")
            operation_id = attributes.get("operation_id")
            provider_id = attributes.get("provider_id")
            require(
                non_empty(node_id) and non_empty(operation_id) and non_empty(provider_id),
                "native node row lacks node/operation/provider identity",
            )
            require(str(node_id) in node_mapping, f"native node is absent from full topology: {node_id}")
            expected = node_mapping[str(node_id)]
            require(
                (str(operation_id), str(provider_id)) == expected[:2],
                f"native row operation/provider does not join node {node_id}",
            )
            native_node_rows += 1
    require(native_node_rows > 0, "native work does not join any typed node")
    require(compute_dispatches > 0, "full profile reports no physical compute dispatch")

    replay_submissions = [
        event
        for event in replay_events
        if event.get("phase") == "vnext.device_physical_submission"
    ]
    require(replay_submissions, "replay profile has no physical submission evidence")
    for event in replay_submissions:
        attributes = event_attributes(event)
        require(
            attributes.get("measurement_instrumentation_present") is True,
            "replay submission does not declare timing instrumentation",
        )
        require(
            attributes.get("production_reusable_execution_selection_preserved") is True,
            "replay submission changed product reusable execution selection",
        )
        require(
            attributes.get("execution_path_policy") == "production_selection",
            "replay submission execution path policy mismatch",
        )

    stage = calculate_stage_coverage(full_events)
    require(
        stage.get("formal_coverage_eligible") is True,
        "stage_clock_domain_unavailable: "
        f"{stage.get('unavailable_reason', 'unknown')} "
        f"missing={stage.get('missing_fields', [])}",
    )
    if backend == "metal":
        unavailable = 0
        for event in native_rows:
            attributes = event_attributes(event)
            shape = event["shape"]
            require(
                attributes.get("device_timing_status") == "unavailable",
                "Metal native work must explicitly mark physical timing unavailable",
            )
            reason = str(attributes.get("device_timing_unavailable_reason", "")).lower()
            require(
                reason in METAL_TIMING_UNAVAILABLE_REASONS,
                f"Metal timing unavailability has an unrecognized reason: {reason!r}",
            )
            require(
                shape.get("device_elapsed_ns") is None,
                "Metal unavailable timing must not contain fabricated elapsed time",
            )
            require(
                attributes.get("formal_device_busy_time_eligible") is False,
                "Metal unavailable timing must not be marked formal device-busy evidence",
            )
            unavailable += 1
        for event in replay_submissions:
            attributes = event_attributes(event)
            require(
                attributes.get("device_timing_status") == "unavailable",
                "Metal replay timing must explicitly be unavailable",
            )
            require(
                str(attributes.get("device_timing_unavailable_reason", "")).lower()
                in METAL_TIMING_UNAVAILABLE_REASONS,
                "Metal replay timing lacks the product-reported unavailable reason",
            )
        device = {
            "status": "unavailable",
            "reason": "backendmeasurementfailed",
            "fabricated_device_time_count": 0,
            "unavailable_native_rows": unavailable,
            "native_op_count": len(native_ids),
            "native_node_rows": native_node_rows,
            "physical_compute_dispatch_count": compute_dispatches,
            "physical_transfer_command_count": transfer_dispatches,
            "formal_device_busy_time_claim": False,
        }
        stage["release_threshold_applied"] = False
        stage["reason"] = "Metal physical timing is product-reported unavailable; identity/dispatch attribution remains mandatory"
    else:
        require(
            stage["coverage"] >= MIN_CUDA_STAGE_COVERAGE,
            f"CUDA decode stage coverage {stage['coverage']:.6f} is below {MIN_CUDA_STAGE_COVERAGE:.2f}",
        )
        total_device_ns = 0
        attributed_device_ns = 0
        interval_device_ns = 0
        kernel_ids: set[str] = set()
        measured_rows = 0
        timing_events = [
            event
            for event in full_events
            if event.get("phase")
            in {"vnext.device_native_work", "vnext.device_execution_span"}
        ]
        measured_span_ranges: dict[str, list[tuple[int, int]]] = {}
        for event in timing_events:
            if event.get("phase") != "vnext.device_execution_span":
                continue
            attributes = event_attributes(event)
            shape = event.get("shape")
            if attributes.get("device_timing_status") != "measured" or not isinstance(shape, dict):
                continue
            fingerprint = attributes.get("physical_submission_fingerprint")
            start = shape.get("start_command_index")
            end = shape.get("end_command_index")
            if non_empty(fingerprint) and type(start) is int and type(end) is int and 0 <= start < end:
                measured_span_ranges.setdefault(str(fingerprint), []).append((start, end))

        total_dispatches_for_timing = 0
        covered_dispatches_for_timing = 0
        for event in native_rows:
            attributes = event_attributes(event)
            shape = event.get("shape")
            require(isinstance(shape, dict), "CUDA native timing row lacks shape")
            dispatches = int(shape.get("physical_compute_dispatch_count", 0)) + int(
                shape.get("physical_transfer_command_count", 0)
            )
            if dispatches <= 0:
                continue
            total_dispatches_for_timing += dispatches
            covered = attributes.get("device_timing_status") == "measured"
            if not covered:
                fingerprint = attributes.get("physical_submission_fingerprint")
                command_index = shape.get("command_index")
                if non_empty(fingerprint) and type(command_index) is int:
                    covered = any(
                        start <= command_index < end
                        for start, end in measured_span_ranges.get(str(fingerprint), [])
                    )
            if covered:
                covered_dispatches_for_timing += dispatches
        require(total_dispatches_for_timing > 0, "CUDA full profile has no dispatches to time")
        dispatch_timing_coverage = (
            covered_dispatches_for_timing / total_dispatches_for_timing
        )
        require(
            dispatch_timing_coverage >= MIN_CUDA_DEVICE_ATTRIBUTION_COVERAGE,
            f"CUDA dispatch timing coverage {dispatch_timing_coverage:.6f} is below "
            f"{MIN_CUDA_DEVICE_ATTRIBUTION_COVERAGE:.2f}",
        )
        for event in timing_events:
            attributes = event_attributes(event)
            shape = event.get("shape")
            detail = event.get("backend_detail")
            require(isinstance(shape, dict), "CUDA device timing row lacks shape")
            if attributes.get("device_timing_status") != "measured":
                continue
            elapsed = shape.get("device_elapsed_ns")
            require(type(elapsed) is int and elapsed > 0, "measured CUDA row lacks positive elapsed time")
            measured_rows += 1
            total_device_ns += elapsed
            has_native_mapping = (
                event.get("phase") == "vnext.device_native_work"
                and non_empty(attributes.get("native_op_id"))
                and (
                    attributes.get("attribution_scope") != "node"
                    or (
                        non_empty(attributes.get("node_id"))
                        and non_empty(attributes.get("operation_id"))
                        and non_empty(attributes.get("provider_id"))
                    )
                )
            )
            intervals = detail.get("device_intervals") if isinstance(detail, dict) else None
            require(
                isinstance(intervals, list),
                "measured CUDA row lacks device subwork intervals",
            )
            local_intervals: list[tuple[int, int]] = []
            for interval in intervals:
                require(isinstance(interval, dict), "CUDA device subwork interval is not an object")
                start = interval.get("start_offset_ns")
                end = interval.get("end_offset_ns")
                subwork_id = interval.get("subwork_id")
                require(
                    type(start) is int and type(end) is int and 0 <= start < end,
                    "CUDA device subwork interval bounds are invalid",
                )
                require(non_empty(subwork_id), "CUDA device subwork interval lacks kernel/native identity")
                kernel_ids.add(str(subwork_id))
                local_intervals.append((start, end))
            interval_length = validate_device_interval_contract(
                local_intervals,
                elapsed_ns=elapsed,
                context="CUDA measured row",
            )
            interval_device_ns += interval_length
            if has_native_mapping or interval_length > 0:
                attributed_device_ns += elapsed
        require(measured_rows > 0 and total_device_ns > 0, "CUDA full profile has no measured device timing")
        require(kernel_ids, "CUDA full profile has no kernel/native subwork identity")
        attribution_coverage = attributed_device_ns / total_device_ns
        require(
            attribution_coverage >= MIN_CUDA_DEVICE_ATTRIBUTION_COVERAGE,
            f"CUDA device attribution coverage {attribution_coverage:.6f} is below "
            f"{MIN_CUDA_DEVICE_ATTRIBUTION_COVERAGE:.2f}",
        )
        device = {
            "status": "measured",
            "timing_semantics": "submission_relative_duration_only",
            "formal_device_busy_time_claim": False,
            "measured_rows": measured_rows,
            "total_device_ns": total_device_ns,
            "attributed_device_ns": attributed_device_ns,
            "kernel_interval_ns": interval_device_ns,
            "attribution_coverage": attribution_coverage,
            "minimum_attribution_coverage": MIN_CUDA_DEVICE_ATTRIBUTION_COVERAGE,
            "dispatch_timing_coverage": dispatch_timing_coverage,
            "timed_dispatch_count": covered_dispatches_for_timing,
            "total_dispatch_count": total_dispatches_for_timing,
            "native_op_count": len(native_ids),
            "kernel_or_subwork_id_count": len(kernel_ids),
            "native_node_rows": native_node_rows,
            "physical_compute_dispatch_count": compute_dispatches,
            "physical_transfer_command_count": transfer_dispatches,
        }
        stage["release_threshold_applied"] = True
        stage["minimum_coverage"] = MIN_CUDA_STAGE_COVERAGE

    return {
        "status": "pass",
        "plan_ids": next(iter(unique_plans)),
        "node_count": len(node_mapping),
        "operation_count": len({mapping[0] for mapping in node_mapping.values()}),
        "provider_count": len({mapping[1] for mapping in node_mapping.values()}),
        "resource_pool_count": len(resource_ids),
        "native_op_count": len(native_ids),
        "replay_physical_submission_count": len(replay_submissions),
        "mode_joined_identity_rows": mode_identity_rows,
        "stage_timing": stage,
        "device_timing": device,
    }


def run_validator_action(
    *,
    root: Path,
    name: str,
    command_factory: Callable[[Path], list[str]],
    environment: dict[str, str],
    resume: bool,
    pass_fragment: str,
) -> dict[str, Any]:
    action_dir = root / "validation" / name
    command = command_factory(action_dir)
    action = run_bounded_action(
        action_dir,
        product_command=command,
        environment=environment,
        expected_duration_seconds=30,
        hard_deadline_seconds=180,
        progress_signals=["stdout.log bytes", "validator output file count"],
        resume=resume,
    )
    stdout_path = Path(action["receipt"]["stdout"]["path"])
    stdout = stdout_path.read_text(encoding="utf-8")
    require(pass_fragment in stdout, f"{name}: validator PASS line is missing")
    return {
        "action_dir": relative_to(root, action_dir),
        "command": command,
        "receipt": relative_to(root, action_dir / "bounded.receipt.json"),
        "stdout": relative_to(root, stdout_path),
        "pass_line": next(line for line in stdout.splitlines() if pass_fragment in line),
        "reused": action["reused"],
    }


def metal_probe_helper() -> str:
    return (
        "import json,subprocess;"
        "p=subprocess.run(['/usr/sbin/system_profiler','SPHardwareDataType','SPDisplaysDataType','-json'],"
        "stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45,check=True);"
        "d=json.loads(p.stdout);h=d['SPHardwareDataType'][0];g=d['SPDisplaysDataType'][0];"
        "print(json.dumps({'chip':h.get('chip_type'),'memory':h.get('physical_memory'),"
        "'machine_model':h.get('machine_model'),'gpu':g.get('sppci_model') or g.get('_name'),"
        "'gpu_cores':g.get('sppci_cores')},sort_keys=True))"
    )


def probe_hardware(
    root: Path, *, backend: str, environment: dict[str, str], resume: bool
) -> dict[str, Any]:
    if backend == "cuda":
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    else:
        command = [sys.executable, "-c", metal_probe_helper()]
    action_dir = root / "hardware"
    action = run_bounded_action(
        action_dir,
        product_command=command,
        environment=environment,
        expected_duration_seconds=10,
        hard_deadline_seconds=60,
        progress_signals=["stdout.log bytes", "bounded receipt status"],
        resume=resume,
    )
    stdout_path = Path(action["receipt"]["stdout"]["path"])
    stdout = stdout_path.read_text(encoding="utf-8").strip()
    if backend == "cuda":
        rows = [line.strip() for line in stdout.splitlines() if line.strip()]
        require(len(rows) == 1, f"CUDA R2 profile requires exactly one GPU; observed {len(rows)}")
        columns = [value.strip() for value in rows[0].split(",")]
        require(len(columns) == 5, "nvidia-smi hardware row has unexpected columns")
        require("4090" in columns[1], f"CUDA R2 profile requires RTX 4090, observed {columns[1]}")
        hardware = {
            "gpu_count": 1,
            "index": columns[0],
            "name": columns[1],
            "uuid": columns[2],
            "memory_total_mib": int(columns[3]),
            "driver_version": columns[4],
        }
    else:
        try:
            hardware = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise CollectorError(f"Metal hardware probe returned invalid JSON: {error}") from error
        require(isinstance(hardware, dict), "Metal hardware probe did not return an object")
        require(hardware.get("chip") == "Apple M1 Max", f"Metal R2 profile requires Apple M1 Max: {hardware}")
        require(hardware.get("memory") == "32 GB", f"Metal R2 profile requires 32 GB: {hardware}")
        require(str(hardware.get("gpu_cores")) == "24", f"Metal R2 profile requires 24 GPU cores: {hardware}")
    return {
        "backend": backend,
        "probe": hardware,
        "platform": platform.platform(),
        "receipt": relative_to(root, action_dir / "bounded.receipt.json"),
        "stdout": relative_to(root, stdout_path),
        "reused": action["reused"],
    }


def evidence_files(root: Path) -> list[dict[str, Any]]:
    excluded = {"manifest.json", "manifest.sha256"}
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not path.is_file() or relative_to(root, path) in excluded:
            continue
        identity = file_identity(path)
        identity["path"] = relative_to(root, path)
        rows.append(identity)
    return rows


def verify_evidence(root: Path, rows: Any) -> None:
    require(isinstance(rows, list) and rows, "manifest evidence_files must be non-empty")
    observed_paths: set[str] = set()
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"evidence_files[{index}] is not an object")
        relative = row.get("path")
        require(isinstance(relative, str) and relative and not relative.startswith("/"), f"evidence_files[{index}].path is invalid")
        require(relative not in observed_paths, f"duplicate evidence path: {relative}")
        observed_paths.add(relative)
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()), f"evidence path escapes artifact root: {relative}")
        require(path.is_file(), f"evidence file is missing: {path}")
        require(
            row.get("size_bytes") == path.stat().st_size
            and row.get("sha256") == sha256_file(path),
            f"evidence identity mismatch: {path}",
        )


def backend_pass_line(backend: str, out: Path) -> str:
    return f"{PASS_PREFIX} {backend.upper()} PASS: {out}"


def validate_backend_manifest(path: Path, expected_backend: str | None = None) -> dict[str, Any]:
    manifest = read_json(path)
    root = path.parent.resolve()
    require(manifest.get("schema") == BACKEND_SCHEMA, f"{path}: backend manifest schema mismatch")
    require(manifest.get("status") == "pass", f"{path}: backend profile status is not pass")
    backend = manifest.get("backend")
    require(backend in {"cuda", "metal"}, f"{path}: invalid backend")
    if expected_backend is not None:
        require(backend == expected_backend, f"{path}: expected {expected_backend}, observed {backend}")
    require(manifest.get("model_key") == "m1", f"{path}: profile lane is not M1")
    source = manifest.get("source")
    require(isinstance(source, dict), f"{path}: source identity is missing")
    require_reviewed_collector_source(source, str(path))
    require(
        source.get("dirty_status", {}).get("is_dirty") is False,
        f"{path}: formal profile source is dirty",
    )
    product_closure = source.get("product_source_closure")
    require(
        isinstance(product_closure, dict)
        and product_closure.get("pathspecs") == list(PRODUCT_SOURCE_PATHS)
        and type(product_closure.get("entry_count")) is int
        and product_closure["entry_count"] > 0
        and re.fullmatch(
            r"[0-9a-f]{64}", str(product_closure.get("git_tree_listing_sha256", ""))
        )
        is not None,
        f"{path}: product source closure is missing or malformed",
    )
    overhead = manifest.get("overhead")
    require(isinstance(overhead, dict) and overhead.get("status") == "pass", f"{path}: overhead did not pass")
    require(
        isinstance(overhead.get("median_duration_overhead"), (int, float))
        and overhead["median_duration_overhead"] <= MAX_PROFILE_OVERHEAD,
        f"{path}: profile overhead exceeds release limit",
    )
    contract = manifest.get("profile_contract")
    require(isinstance(contract, dict) and contract.get("status") == "pass", f"{path}: profile contract did not pass")
    if backend == "cuda":
        require(
            contract.get("stage_timing", {}).get("formal_coverage_eligible") is True,
            f"{path}: CUDA stage coverage lacks bounded clock conversion",
        )
        require(
            contract.get("stage_timing", {}).get("coverage", 0) >= MIN_CUDA_STAGE_COVERAGE,
            f"{path}: CUDA stage coverage is below threshold",
        )
        require(
            contract.get("device_timing", {}).get("attribution_coverage", 0)
            >= MIN_CUDA_DEVICE_ATTRIBUTION_COVERAGE,
            f"{path}: CUDA device attribution coverage is below threshold",
        )
        require(
            contract.get("device_timing", {}).get("dispatch_timing_coverage", 0)
            >= MIN_CUDA_DEVICE_ATTRIBUTION_COVERAGE,
            f"{path}: CUDA dispatch timing coverage is below threshold",
        )
    else:
        device = contract.get("device_timing", {})
        require(device.get("status") == "unavailable", f"{path}: Metal timing status mismatch")
        require(device.get("fabricated_device_time_count") == 0, f"{path}: Metal timing was fabricated")
        require(device.get("physical_compute_dispatch_count", 0) > 0, f"{path}: Metal dispatch evidence is absent")
    recorded_artifact_dir = manifest.get("artifact_dir")
    require(
        isinstance(recorded_artifact_dir, str)
        and Path(recorded_artifact_dir).is_absolute(),
        f"{path}: recorded artifact directory is invalid",
    )
    pass_line = backend_pass_line(str(backend), Path(recorded_artifact_dir))
    require(manifest.get("pass_line") == pass_line, f"{path}: backend PASS line mismatch")
    verify_evidence(root, manifest.get("evidence_files"))
    require(
        manifest.get("evidence_closure_sha256") == canonical_sha256(manifest["evidence_files"]),
        f"{path}: evidence closure digest mismatch",
    )
    return manifest


def command_blueprint(
    backend: str, binary: Path, model: Path, semantic_model: Path
) -> dict[str, Any]:
    common = [
        str(binary),
        "run",
        str(model),
        "--backend",
        backend,
        "--disable-thinking",
        "--temperature",
        "0",
        "--repeat-penalty",
        "1.0",
        "--output-format",
        "jsonl",
        "--semantic-source",
        str(semantic_model),
        "--tokenizer-source",
        str(semantic_model),
    ]
    return {
        "overhead": {
            "common": common,
            "prompt": OVERHEAD_PROMPT,
            "max_tokens": OVERHEAD_MAX_TOKENS,
            "modes": ["off", "basic"],
            "independent_process_repeats_each": PROFILE_REPEATS,
        },
        "diagnostic": {
            "common": common,
            "prompt": DIAGNOSTIC_PROMPT,
            "max_tokens": DIAGNOSTIC_MAX_TOKENS,
            "modes": ["replay", "full"],
            "throughput_claim": False,
        },
    }


def write_collection_plan(
    *,
    out: Path,
    backend: str,
    binary: Path,
    model: Path,
    semantic_model: Path,
    source: dict[str, Any],
    environment: dict[str, str],
    plan_only: bool,
    resume: bool,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "collection.plan.json"
    core = {
        "schema": "ferrum.runtime-vnext-r2-profile-collection-plan.v1",
        "backend": backend,
        "model_key": "m1",
        "binary": str(binary),
        "model": str(model),
        "semantic_model": str(semantic_model),
        "source": source,
        "sanitized_environment": environment,
        "worker_limits": {
            "max_processes": MAX_PROCESSES,
            "max_group_threads": MAX_GROUP_THREADS,
            "max_per_process_threads": MAX_PER_PROCESS_THREADS,
        },
        "command_blueprint": command_blueprint(backend, binary, model, semantic_model),
        "actions": [
            {"name": "hardware", "expected_seconds": 10, "hard_deadline_seconds": 60},
            *[
                {"name": f"off-{index}", "expected_seconds": 180, "hard_deadline_seconds": 600}
                for index in range(1, PROFILE_REPEATS + 1)
            ],
            *[
                {"name": f"basic-{index}", "expected_seconds": 180, "hard_deadline_seconds": 600}
                for index in range(1, PROFILE_REPEATS + 1)
            ],
            {"name": "replay", "expected_seconds": 300, "hard_deadline_seconds": 900},
            {"name": "full", "expected_seconds": 600, "hard_deadline_seconds": 1800},
            {"name": "profile-analyzer", "expected_seconds": 30, "hard_deadline_seconds": 180},
            {"name": "replay-bundle-gate", "expected_seconds": 30, "hard_deadline_seconds": 180},
        ],
    }
    core["plan_fingerprint"] = canonical_sha256(core)
    plan = {**core, "status": "planned" if plan_only else "collecting", "written_at": utc_now()}
    if plan_path.exists():
        existing = read_json(plan_path)
        if resume:
            require(
                existing.get("plan_fingerprint") == core["plan_fingerprint"],
                "--resume inputs differ from the original collection plan",
            )
        elif existing != plan:
            raise CollectorError(f"collection plan already exists; pass --resume: {plan_path}")
    atomic_write_json(plan_path, plan)
    return plan


def collect_backend(args: argparse.Namespace) -> int:
    out = args.out.expanduser().resolve()
    binary = args.binary.expanduser().resolve()
    model = lexical_absolute_path(args.model)
    semantic_model = args.semantic_model.expanduser().resolve()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        require(args.resume, f"manifest already exists; pass --resume: {manifest_path}")
        manifest = validate_backend_manifest(manifest_path, args.backend)
        print(manifest["pass_line"])
        return 0
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise CollectorError(f"output directory is not empty; pass --resume: {out}")

    source = source_identity()
    environment = sanitized_environment()
    write_collection_plan(
        out=out,
        backend=args.backend,
        binary=binary,
        model=model,
        semantic_model=semantic_model,
        source=source,
        environment=environment,
        plan_only=args.plan_only,
        resume=args.resume,
    )
    if args.plan_only:
        print(f"FERRUM RUNTIME VNEXT R2 PROFILE COLLECTOR PLAN: {out}")
        return 0

    require(binary.is_file() and os.access(binary, os.X_OK), f"binary is not executable: {binary}")
    require(model.exists(), f"model path is missing: {model}")
    require(semantic_model.exists(), f"semantic model path is missing: {semantic_model}")
    require(
        source["dirty_status"]["is_dirty"] is False,
        f"formal profile collection requires clean source: {source['dirty_status']['status_short']}",
    )
    hardware = probe_hardware(
        out, backend=args.backend, environment=environment, resume=args.resume
    )

    identity_cache: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for label, path in (("binary", binary), ("model", model), ("semantic_model", semantic_model)):
        key = str(path)
        if key not in identity_cache:
            identity_cache[key] = path_closure_identity(path)
        inputs[label] = identity_cache[key]

    overhead_rows: list[dict[str, Any]] = []
    for repeat in range(1, PROFILE_REPEATS + 1):
        for mode in ("off", "basic"):
            overhead_rows.append(
                run_product_command(
                    root=out,
                    name=f"{mode}-{repeat}",
                    backend=args.backend,
                    binary=binary,
                    model=model,
                    semantic_model=semantic_model,
                    mode=mode,
                    prompt=OVERHEAD_PROMPT,
                    max_tokens=OVERHEAD_MAX_TOKENS,
                    environment=environment,
                    resume=args.resume,
                )
            )
    overhead = validate_overhead(overhead_rows)

    replay_row = run_product_command(
        root=out,
        name="replay",
        backend=args.backend,
        binary=binary,
        model=model,
        semantic_model=semantic_model,
        mode="replay",
        prompt=DIAGNOSTIC_PROMPT,
        max_tokens=DIAGNOSTIC_MAX_TOKENS,
        environment=environment,
        resume=args.resume,
    )
    full_row = run_product_command(
        root=out,
        name="full",
        backend=args.backend,
        binary=binary,
        model=model,
        semantic_model=semantic_model,
        mode="full",
        prompt=DIAGNOSTIC_PROMPT,
        max_tokens=DIAGNOSTIC_MAX_TOKENS,
        environment=environment,
        resume=args.resume,
    )
    require(
        replay_row["run_summary"]["content_sha256"]
        == full_row["run_summary"]["content_sha256"],
        "replay/full diagnostic output differs",
    )

    profile_paths = [
        out / str(row["profile"])
        for row in overhead_rows
        if row["mode"] == "basic"
    ] + [out / str(replay_row["profile"]), out / str(full_row["profile"])]
    analyzer = run_validator_action(
        root=out,
        name="profile-analyzer",
        command_factory=lambda action_dir: [
            sys.executable,
            str(PROFILE_ANALYZER),
            *sum((["--profile-jsonl", str(path)] for path in profile_paths), []),
            "--out",
            str(action_dir / "result"),
        ],
        environment=environment,
        resume=args.resume,
        pass_fragment="FERRUM PROFILE ANALYZER PASS",
    )
    replay_bundle = run_validator_action(
        root=out,
        name="replay-bundle-gate",
        command_factory=lambda action_dir: [
            sys.executable,
            str(REPLAY_BUNDLE_GATE),
            "--bundle-dir",
            str(out / str(replay_row["request_dump"])),
            "--bundle-dir",
            str(out / str(full_row["request_dump"])),
            "--out",
            str(action_dir / "result"),
        ],
        environment=environment,
        resume=args.resume,
        pass_fragment="REQUEST REPLAY BUNDLE PASS",
    )

    basic_events = [
        read_profile_events(out / str(row["profile"]))
        for row in overhead_rows
        if row["mode"] == "basic"
    ]
    replay_events = read_profile_events(out / str(replay_row["profile"]))
    full_events = read_profile_events(out / str(full_row["profile"]))
    profile_contract = validate_identity_and_device_contract(
        backend=args.backend,
        basic_events=basic_events,
        replay_events=replay_events,
        full_events=full_events,
    )
    contract_path = out / "validation" / "profile-contract.json"
    atomic_write_json(contract_path, profile_contract)

    final_source = source_identity()
    require(
        final_source == source,
        "source identity changed during profile collection; all collected evidence is stale",
    )

    plan = read_json(out / "collection.plan.json")
    plan.update({"status": "pass", "finished_at": utc_now()})
    atomic_write_json(out / "collection.plan.json", plan)

    rows = evidence_files(out)
    pass_line = backend_pass_line(args.backend, out)
    manifest = {
        "schema_version": 1,
        "schema": BACKEND_SCHEMA,
        "artifact_type": "runtime_vnext_r2_profile_backend",
        "status": "pass",
        "artifact_dir": str(out),
        "backend": args.backend,
        "model_key": "m1",
        "source": source,
        "inputs": inputs,
        "hardware": hardware,
        "sanitized_environment": environment,
        "hidden_ferrum_env_count": 0,
        "worker_limits": {
            "max_processes": MAX_PROCESSES,
            "max_group_threads": MAX_GROUP_THREADS,
            "max_per_process_threads": MAX_PER_PROCESS_THREADS,
        },
        "workload": {
            "overhead": {
                "entrypoint": "ferrum run",
                "prompt": OVERHEAD_PROMPT,
                "max_tokens": OVERHEAD_MAX_TOKENS,
                "temperature": 0.0,
                "repeat_penalty": 1.0,
                "disable_thinking": True,
                "profile_modes": ["off", "basic"],
                "independent_process_repeats_each": PROFILE_REPEATS,
            },
            "diagnostic": {
                "entrypoint": "ferrum run",
                "prompt": DIAGNOSTIC_PROMPT,
                "max_tokens": DIAGNOSTIC_MAX_TOKENS,
                "temperature": 0.0,
                "repeat_penalty": 1.0,
                "disable_thinking": True,
                "profile_modes": ["replay", "full"],
                "throughput_claim": False,
            },
        },
        "workload_contract_sha256": canonical_sha256(
            {
                "overhead_prompt": OVERHEAD_PROMPT,
                "overhead_max_tokens": OVERHEAD_MAX_TOKENS,
                "diagnostic_prompt": DIAGNOSTIC_PROMPT,
                "diagnostic_max_tokens": DIAGNOSTIC_MAX_TOKENS,
                "sampling": {"temperature": 0.0, "repeat_penalty": 1.0},
            }
        ),
        "runs": [*overhead_rows, replay_row, full_row],
        "overhead": overhead,
        "profile_contract": profile_contract,
        "validators": {
            "analyze_ferrum_profile": analyzer,
            "request_replay_bundle_gate": replay_bundle,
        },
        "evidence_files": rows,
        "evidence_closure_sha256": canonical_sha256(rows),
        "created_at": utc_now(),
        "pass_line": pass_line,
    }
    atomic_write_json(manifest_path, manifest)
    (out / "manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    validate_backend_manifest(manifest_path, args.backend)
    print(pass_line)
    return 0


def aggregate_manifests(args: argparse.Namespace) -> int:
    out = args.out.expanduser().resolve()
    cuda_path = args.cuda.expanduser().resolve()
    metal_path = args.metal.expanduser().resolve()
    aggregate_source = source_identity()
    require(
        aggregate_source["dirty_status"]["is_dirty"] is False,
        "formal profile aggregate requires clean source",
    )
    require(cuda_path != metal_path, "CUDA and Metal manifests must be distinct")
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise CollectorError(f"aggregate output is not empty; pass --resume: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and args.resume:
        existing = read_json(manifest_path)
        require(existing.get("schema") == AGGREGATE_SCHEMA, "existing aggregate schema mismatch")
        verify_evidence(out, existing.get("evidence_files"))
        print(existing["pass_line"])
        return 0
    cuda = validate_backend_manifest(cuda_path, "cuda")
    metal = validate_backend_manifest(metal_path, "metal")
    cuda_product_closure = cuda["source"].get("product_source_closure")
    metal_product_closure = metal["source"].get("product_source_closure")
    require(
        isinstance(cuda_product_closure, dict)
        and cuda_product_closure == metal_product_closure
        and cuda_product_closure == aggregate_source["product_source_closure"],
        "CUDA, Metal, and aggregate source differ in product/runtime/native closure",
    )
    require_reviewed_collector_source(aggregate_source, "profile aggregate")
    require(
        cuda["workload_contract_sha256"] == metal["workload_contract_sha256"],
        "CUDA and Metal profile workload contracts differ",
    )
    child_dir = out / "children"
    child_dir.mkdir(exist_ok=True)
    staged: dict[str, dict[str, Any]] = {}
    for backend, source_path in (("cuda", cuda_path), ("metal", metal_path)):
        destination = child_dir / f"{backend}.manifest.json"
        shutil.copyfile(source_path, destination)
        staged[backend] = {
            "source_path": str(source_path),
            "artifact_dir": str(source_path.parent),
            "manifest": relative_to(out, destination),
            "manifest_sha256": sha256_file(destination),
            "binary_sha256": (cuda if backend == "cuda" else metal)["inputs"]["binary"]["closure_sha256"],
            "pass_line": (cuda if backend == "cuda" else metal)["pass_line"],
        }
    rows = evidence_files(out)
    pass_line = f"{AGGREGATE_PASS_PREFIX} {out}"
    aggregate = {
        "schema_version": 1,
        "schema": AGGREGATE_SCHEMA,
        "artifact_type": "runtime_vnext_r2_profile_aggregate",
        "status": "pass",
        "artifact_dir": str(out),
        "model_key": "m1",
        "backends": ["cuda", "metal"],
        "source": aggregate_source,
        "backend_source_bindings": {
            "cuda": {
                "git_sha": cuda["source"]["git_sha"],
                "tree_sha": cuda["source"]["tree_sha"],
                "product_source_closure": cuda_product_closure,
            },
            "metal": {
                "git_sha": metal["source"]["git_sha"],
                "tree_sha": metal["source"]["tree_sha"],
                "product_source_closure": metal_product_closure,
            },
        },
        "workload_contract_sha256": cuda["workload_contract_sha256"],
        "children": staged,
        "summary": {
            "backend_pass_count": 2,
            "profile_off_basic_independent_processes": 12,
            "diagnostic_processes": 4,
            "cuda_stage_coverage": cuda["profile_contract"]["stage_timing"]["coverage"],
            "cuda_device_attribution_coverage": cuda["profile_contract"]["device_timing"]["attribution_coverage"],
            "cuda_dispatch_timing_coverage": cuda["profile_contract"]["device_timing"]["dispatch_timing_coverage"],
            "metal_device_timing_status": metal["profile_contract"]["device_timing"]["status"],
            "metal_fabricated_device_time_count": 0,
            "cuda_basic_overhead": cuda["overhead"]["median_duration_overhead"],
            "metal_basic_overhead": metal["overhead"]["median_duration_overhead"],
        },
        "evidence_files": rows,
        "evidence_closure_sha256": canonical_sha256(rows),
        "created_at": utc_now(),
        "pass_line": pass_line,
    }
    atomic_write_json(manifest_path, aggregate)
    (out / "manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    verify_evidence(out, aggregate["evidence_files"])
    print(pass_line)
    return 0


def fixture_profile_events(backend: str) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    plan_id = "plan/sha256/" + "a" * 64
    run_id = "run.fixture"
    request_id = "request.product.fixture"
    engine_wall_anchor = 1_700_000_000_000_000_000
    vnext_wall_anchor = engine_wall_anchor + 5_000

    def base(phase: str, mode: str, timestamp: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "ts_unix_nanos": timestamp + 1_700_000_000_000_000_000,
            "event_id": f"{mode}-{phase}-{timestamp}",
            "request_id": request_id,
            "correlation_id": request_id,
            "entrypoint": "run",
            "backend": "actual",
            "runtime_preset_hash": "sha256:" + "b" * 64,
            "phase": phase,
            "event_kind": "instant",
            "timestamp": "2026-08-09T00:00:00Z",
            "status": "ok",
            "model": "fixture/M1",
            "shape": {"execution_sequence": timestamp},
            "attributes": {
                "profile_detail": mode,
                "plan_id": plan_id,
                "plan_hash": "a" * 64,
            },
        }

    def topology(mode: str) -> list[dict[str, Any]]:
        accepted = base("vnext.request_accepted", mode, 0)
        accepted["attributes"].update(
            {
                "monotonic_nanos_since_run_start": 0,
                "run_id": run_id,
                VNEXT_CLOCK_SOURCE_ATTRIBUTE: "rust_std_instant",
                VNEXT_WALL_ANCHOR_ATTRIBUTE: vnext_wall_anchor,
                VNEXT_CLOCK_ERROR_ATTRIBUTE: 1,
            }
        )
        plan = base("vnext.plan_built", mode, 1)
        started = base("vnext.node_started", mode, 100)
        retired = base("vnext.node_retired", mode, 1_000)
        for event in (started, retired):
            event["attributes"].update(
                {
                    "monotonic_nanos_since_run_start": event["shape"]["execution_sequence"],
                    "execution_identity": {
                        "run_id": run_id,
                        "request_id": request_id,
                        "frame_id": 1,
                        "node_invocation_id": 1,
                        "node_id": "node.decode",
                        "operation_id": "operation.decode",
                        "provider_id": f"provider.{backend}.decode",
                        "resource_pool_id": 1,
                        "span_id": f"span/{request_id}/frame/1/node/1",
                    },
                }
            )
        return [accepted, plan, started, retired]

    basics = [topology("basic") for _ in range(PROFILE_REPEATS)]
    replay = topology("replay")
    submission = base("vnext.device_physical_submission", "replay", 2_000)
    submission["status"] = "diagnostic_only"
    submission["attributes"].update(
        {
            "measurement_instrumentation_present": True,
            "production_reusable_execution_selection_preserved": True,
            "execution_path_policy": "production_selection",
            "device_timing_status": "measured" if backend == "cuda" else "unavailable",
        }
    )
    if backend == "metal":
        submission["attributes"]["device_timing_unavailable_reason"] = "backendmeasurementfailed"
    replay.append(submission)

    full = topology("full")
    full[2]["attributes"]["monotonic_nanos_since_run_start"] = 300
    full[3]["attributes"]["monotonic_nanos_since_run_start"] = 800
    terminal = base("actual_run_generation", "full", 1_100)
    terminal["event_kind"] = "timed_span"
    terminal["duration_us"] = 1
    terminal["attributes"].update(
        {
            "execution_request_id": request_id,
            "engine_token_clock_source": "rust_std_instant",
            "engine_token_wall_anchor_unix_nanos": engine_wall_anchor,
            "clock_conversion_max_error_nanos": 1,
            "engine_decode_ready_nanos_since_request_start": 5_100,
            "engine_token_commit_nanos_since_request_start": [6_000],
            ENGINE_DECODE_STAGE_INTERVALS_ATTRIBUTE: [
                {
                    "stage": "decode_scheduling",
                    "start_nanos_since_request_start": 5_100,
                    "end_nanos_since_request_start": 5_300,
                },
                {
                    "stage": "decode_execution",
                    "start_nanos_since_request_start": 5_300,
                    "end_nanos_since_request_start": 5_800,
                },
                {
                    "stage": "decode_postprocess",
                    "start_nanos_since_request_start": 5_800,
                    "end_nanos_since_request_start": 6_000,
                },
            ],
            ENGINE_DECODE_STAGE_COUNT_ATTRIBUTE: 3,
        }
    )
    native = base("vnext.device_native_work", "full", 1_050)
    native["status"] = "diagnostic_only"
    native["shape"] = {
        "physical_compute_dispatch_count": 2,
        "physical_transfer_command_count": 0,
        "device_elapsed_ns": 900 if backend == "cuda" else None,
    }
    native["attributes"].update(
        {
            "attribution_scope": "node",
            "native_op_id": "native.decode",
            "node_id": "node.decode",
            "operation_id": "operation.decode",
            "provider_id": f"provider.{backend}.decode",
            "device_timing_status": "measured" if backend == "cuda" else "unavailable",
            "formal_device_busy_time_eligible": False,
        }
    )
    if backend == "cuda":
        native["backend_detail"] = {
            "device_intervals": [
                {
                    "start_offset_ns": 100,
                    "end_offset_ns": 1_000,
                    "subwork_id": "kernel.decode",
                }
            ]
        }
    else:
        native["attributes"]["device_timing_unavailable_reason"] = "backendmeasurementfailed"
    full.extend([native, terminal])
    return basics, replay, full


def run_selftest() -> int:
    current_source = {
        "collector_path": COLLECTOR_PATH.relative_to(REPO_ROOT).as_posix(),
        "collector_sha256": sha256_file(COLLECTOR_PATH),
    }
    require_reviewed_collector_source(current_source, "profile collector self-test")
    unreviewed_source = dict(current_source)
    unreviewed_source["collector_sha256"] = next(
        candidate
        for candidate in ("0" * 64, "f" * 64)
        if candidate not in reviewed_collector_sha256s()
    )
    try:
        require_reviewed_collector_source(unreviewed_source, "profile collector self-test")
        raise CollectorError("unreviewed profile collector unexpectedly passed")
    except CollectorError as error:
        require(
            "is not a reviewed Git-history source" in str(error),
            "unreviewed profile collector failed for the wrong reason",
        )
    cuda = fixture_profile_events("cuda")
    cuda_result = validate_identity_and_device_contract(
        backend="cuda", basic_events=cuda[0], replay_events=cuda[1], full_events=cuda[2]
    )
    require(cuda_result["status"] == "pass", "positive CUDA fixture did not pass")
    incomplete_device_interval = json.loads(json.dumps(cuda))
    incomplete_device_interval[2][-2]["backend_detail"]["device_intervals"][0][
        "end_offset_ns"
    ] = 101
    try:
        validate_identity_and_device_contract(
            backend="cuda",
            basic_events=incomplete_device_interval[0],
            replay_events=incomplete_device_interval[1],
            full_events=incomplete_device_interval[2],
        )
    except CollectorError:
        incomplete_device_interval_rejected = True
    else:
        incomplete_device_interval_rejected = False
    overlapping_device_intervals = json.loads(json.dumps(cuda))
    overlapping_device_intervals[2][-2]["backend_detail"]["device_intervals"] = [
        {
            "start_offset_ns": 100,
            "end_offset_ns": 550,
            "subwork_id": "kernel.decode.first",
        },
        {
            "start_offset_ns": 500,
            "end_offset_ns": 950,
            "subwork_id": "kernel.decode.second",
        },
    ]
    try:
        validate_identity_and_device_contract(
            backend="cuda",
            basic_events=overlapping_device_intervals[0],
            replay_events=overlapping_device_intervals[1],
            full_events=overlapping_device_intervals[2],
        )
    except CollectorError:
        overlapping_device_intervals_rejected = True
    else:
        overlapping_device_intervals_rejected = False
    baseline_stage = calculate_stage_coverage(cuda[2])
    require(
        baseline_stage["formal_coverage_eligible"] is True
        and baseline_stage["coverage"] == 1.0,
        "bounded wall-anchor fixture did not produce exact formal coverage",
    )
    require(
        baseline_stage["engine_stage_interval_count"] == 3
        and baseline_stage["engine_stage_accounted_union_ns"] == 900,
        "typed engine decode stages were not unioned into formal coverage",
    )
    missing_execution = json.loads(json.dumps(cuda[2]))
    missing_execution[-1]["attributes"][ENGINE_DECODE_STAGE_INTERVALS_ATTRIBUTE].pop(1)
    missing_execution[-1]["attributes"][ENGINE_DECODE_STAGE_COUNT_ATTRIBUTE] = 2
    try:
        calculate_stage_coverage(missing_execution)
    except CollectorError:
        missing_execution_stage_rejected = True
    else:
        missing_execution_stage_rejected = False
    missing_postprocess = json.loads(json.dumps(cuda[2]))
    missing_postprocess[-1]["attributes"][ENGINE_DECODE_STAGE_INTERVALS_ATTRIBUTE].pop()
    missing_postprocess[-1]["attributes"][ENGINE_DECODE_STAGE_COUNT_ATTRIBUTE] = 2
    try:
        calculate_stage_coverage(missing_postprocess)
    except CollectorError:
        missing_postprocess_stage_rejected = True
    else:
        missing_postprocess_stage_rejected = False
    malformed_engine_stage = json.loads(json.dumps(cuda[2]))
    malformed_engine_stage[-1]["attributes"][ENGINE_DECODE_STAGE_INTERVALS_ATTRIBUTE][0][
        "end_nanos_since_request_start"
    ] = 5_099
    try:
        calculate_stage_coverage(malformed_engine_stage)
    except CollectorError:
        malformed_engine_stage_rejected = True
    else:
        malformed_engine_stage_rejected = False
    concurrent = json.loads(json.dumps(cuda[2]))
    foreign_request_id = "request.startup.concurrent"
    foreign_started = json.loads(json.dumps(concurrent[2]))
    foreign_retired = json.loads(json.dumps(concurrent[3]))
    for event, timestamp in ((foreign_started, 120), (foreign_retired, 980)):
        event["request_id"] = foreign_request_id
        event["correlation_id"] = foreign_request_id
        event["attributes"]["monotonic_nanos_since_run_start"] = timestamp
        identity = event["attributes"]["execution_identity"]
        identity["request_id"] = foreign_request_id
        identity["span_id"] = f"span/{foreign_request_id}/frame/1/node/1"
    concurrent[2:4] = [concurrent[2], foreign_started, concurrent[3], foreign_retired]
    concurrent_stage = calculate_stage_coverage(concurrent)
    require(
        concurrent_stage == baseline_stage,
        "foreign concurrent request changed target stage coverage",
    )
    duplicate = json.loads(json.dumps(cuda[2]))
    duplicate.insert(3, json.loads(json.dumps(duplicate[2])))
    try:
        calculate_stage_coverage(duplicate)
    except CollectorError:
        target_duplicate_rejected = True
    else:
        raise CollectorError("target-request duplicate node start unexpectedly passed")
    missing_anchor = json.loads(json.dumps(cuda[2]))
    missing_anchor[0]["attributes"].pop(VNEXT_WALL_ANCHOR_ATTRIBUTE)
    unavailable_stage = calculate_stage_coverage(missing_anchor)
    require(
        unavailable_stage["formal_coverage_eligible"] is False
        and unavailable_stage["coverage"] is None
        and unavailable_stage["legacy_mixed_clock_coverage_omitted"] is True,
        "missing vNext wall anchor produced overstated formal coverage",
    )
    try:
        validate_identity_and_device_contract(
            backend="cuda",
            basic_events=cuda[0],
            replay_events=cuda[1],
            full_events=missing_anchor,
        )
    except CollectorError as error:
        missing_clock_rejected = str(error).startswith("stage_clock_domain_unavailable:")
    else:
        missing_clock_rejected = False
    excessive_error = json.loads(json.dumps(cuda[2]))
    excessive_error[0]["attributes"][VNEXT_CLOCK_ERROR_ATTRIBUTE] = 10
    excessive_error_stage = calculate_stage_coverage(excessive_error)
    require(
        excessive_error_stage["formal_coverage_eligible"] is False
        and excessive_error_stage["coverage"] is None,
        "excessive clock conversion error produced formal coverage",
    )
    metal = fixture_profile_events("metal")
    metal_result = validate_identity_and_device_contract(
        backend="metal", basic_events=metal[0], replay_events=metal[1], full_events=metal[2]
    )
    require(metal_result["status"] == "pass", "positive Metal fixture did not pass")

    broken = fixture_profile_events("cuda")
    broken[2][-2]["attributes"].pop("provider_id")
    try:
        validate_identity_and_device_contract(
            backend="cuda", basic_events=broken[0], replay_events=broken[1], full_events=broken[2]
        )
    except CollectorError:
        negative_identity_rejected = True
    else:
        raise CollectorError("missing provider negative fixture unexpectedly passed")

    rows = []
    for mode, values in (("off", [100.0, 101.0, 99.0]), ("basic", [106.0, 107.0, 105.0])):
        for index, duration in enumerate(values, 1):
            rows.append(
                {
                    "mode": mode,
                    "run_summary": {
                        "content_sha256": "c" * 64,
                        "raw_text_sha256": "d" * 64,
                        "completion_tokens": 10,
                        "finish_reason": "length",
                        "duration_ms": duration,
                        "throughput_tokens_per_second": 10_000.0 / duration,
                    },
                    "name": f"{mode}-{index}",
                }
            )
    overhead = validate_overhead(rows)
    require(overhead["status"] == "pass", "positive overhead fixture did not pass")
    for row in rows:
        if row["mode"] == "basic":
            row["run_summary"]["duration_ms"] *= 1.2
            row["run_summary"]["throughput_tokens_per_second"] /= 1.2
    try:
        validate_overhead(rows)
    except CollectorError:
        negative_overhead_rejected = True
    else:
        raise CollectorError("over-limit profile overhead unexpectedly passed")
    require(
        negative_identity_rejected
        and negative_overhead_rejected
        and target_duplicate_rejected
        and missing_clock_rejected,
        "negative fixtures did not reject",
    )
    require(
        incomplete_device_interval_rejected
        and overlapping_device_intervals_rejected,
        "invalid device interval contract unexpectedly passed",
    )
    require(
        missing_execution_stage_rejected
        and missing_postprocess_stage_rejected
        and malformed_engine_stage_rejected,
        "invalid engine decode stage unexpectedly passed",
    )
    with tempfile.TemporaryDirectory(prefix="runtime-vnext-r2-profile-selftest-") as temporary:
        root = Path(temporary)
        blob = root / "model-blob"
        blob.write_bytes(b"locked-profile-model")
        logical_model = root / "Qwen3.5-4B-Q4_K_M.gguf"
        logical_model.symlink_to(blob)
        lexical_model = lexical_absolute_path(logical_model)
        require(
            lexical_model.name == logical_model.name and lexical_model.is_symlink(),
            "profile model path dereferenced the logical GGUF filename",
        )
    print(SELFTEST_PASS_LINE)
    return 0


def parse_collect_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cuda", "metal"))
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--semantic-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def parse_aggregate_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CUDA and Metal R2 profile manifests")
    parser.add_argument("--cuda", required=True, type=Path)
    parser.add_argument("--metal", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def write_reject_artifact(arguments: Sequence[str], error: CollectorError) -> Path | None:
    try:
        out_index = list(arguments).index("--out")
        out = Path(arguments[out_index + 1]).expanduser().resolve()
    except (ValueError, IndexError):
        return None
    if (out / "manifest.json").is_file():
        return None
    try:
        out.mkdir(parents=True, exist_ok=True)
        path = out / "reject.json"
        atomic_write_json(
            path,
            {
                "schema": "ferrum.runtime-vnext-r2-profile-reject.v1",
                "status": "reject",
                "failure_class": str(error).split(":", 1)[0][:256],
                "error": str(error),
                "argv": list(arguments),
                "created_at": utc_now(),
                "stop_condition_reached": True,
                "performance_claim": False,
            },
        )
        return path
    except OSError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments == ["--self-test"]:
            return run_selftest()
        if arguments[:1] == ["aggregate"]:
            return aggregate_manifests(parse_aggregate_args(arguments[1:]))
        return collect_backend(parse_collect_args(arguments))
    except CollectorError as error:
        reject = write_reject_artifact(arguments, error)
        print(f"FERRUM RUNTIME VNEXT R2 PROFILE COLLECTOR FAIL: {error}", file=sys.stderr)
        if reject is not None:
            print(f"FERRUM RUNTIME VNEXT R2 PROFILE COLLECTOR REJECT: {reject}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
