#!/usr/bin/env python3
"""Release binary gates for official Ferrum assets and Homebrew formulae."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = "https://github.com/sizzlecar/ferrum-infer-rs"
SCHEMA_VERSION = 2
FORMULA_TAP = "sizzlecar/ferrum"
DEFAULT_ASSET_DOWNLOAD_TIMEOUT_SECONDS = 7200
DEFAULT_GATE_TIMEOUT_SECONDS = 14400
DEFAULT_EXTRACTION_TIMEOUT_SECONDS = 3600
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES_INTERVAL = 64 * 1024 * 1024
PROGRESS_SECONDS_INTERVAL = 30.0
FORMULAE = {
    "homebrew-metal": {
        "formula": "sizzlecar/ferrum/ferrum",
        "name": "ferrum",
        "asset": "ferrum-macos-aarch64.tar.gz",
    },
    "homebrew-cuda-fetch": {
        "formula": "sizzlecar/ferrum/ferrum-cuda",
        "name": "ferrum-cuda",
        "asset": "ferrum-linux-x86_64-cuda-sm89.tar.gz",
    },
}
BAD_LOG_PATTERNS = [
    "panicked",
    "panic",
    "KV cache overflow",
    "failed to render model chat template",
    "command encoder",
    "failed assertion",
    "<unk>",
    "[PAD]",
]


class GateError(RuntimeError):
    """A release-evidence validation failure."""


def utc_timestamp(epoch: float | None = None) -> str:
    value = datetime.fromtimestamp(epoch if epoch is not None else time.time(), timezone.utc)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_timestamp(value: Any, label: str) -> float:
    if not isinstance(value, str) or not value:
        raise GateError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise GateError(f"{label} is invalid") from error
    if parsed.tzinfo is None:
        raise GateError(f"{label} lacks timezone")
    return parsed.timestamp()


def timing_receipt(started: float, deadline: float, finished: float, rc: int) -> dict[str, Any]:
    return {
        "started_at": utc_timestamp(started),
        "finished_at": utc_timestamp(finished),
        "deadline_at": utc_timestamp(deadline),
        "duration_sec": round(max(0.0, finished - started), 6),
        "rc": rc,
    }


def validate_timing(receipt: Any, label: str, *, expected_rc: int | None = None) -> None:
    if not isinstance(receipt, dict):
        raise GateError(f"{label} is not an object")
    started = parse_timestamp(receipt.get("started_at"), f"{label}.started_at")
    finished = parse_timestamp(receipt.get("finished_at"), f"{label}.finished_at")
    deadline = parse_timestamp(receipt.get("deadline_at"), f"{label}.deadline_at")
    duration = receipt.get("duration_sec")
    rc = receipt.get("rc")
    if not isinstance(duration, (int, float)) or duration < 0:
        raise GateError(f"{label}.duration_sec is invalid")
    if type(rc) is not int:
        raise GateError(f"{label}.rc is invalid")
    if finished + 0.001 < started or finished > deadline + 1.0:
        raise GateError(f"{label} timing is outside its deadline")
    if expected_rc is not None and rc != expected_rc:
        raise GateError(f"{label}.rc must be {expected_rc}, got {rc}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise GateError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise GateError(f"{label} is not a regular file: {path}")
    return path


def evidence_ref(root: Path, path: Path) -> dict[str, Any]:
    root_resolved = root.resolve(strict=True)
    path = require_regular_file(path, "evidence file")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise GateError(f"evidence file escapes artifact directory: {path}") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise GateError(f"invalid evidence path: {relative}")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise GateError(f"evidence path contains a symlink: {relative}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise GateError(f"evidence path resolves outside artifact directory: {relative}") from error
    return {
        "path": relative.as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def resolve_evidence_ref(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "size_bytes"}:
        raise GateError(f"{label} reference schema differs")
    relative_raw = raw.get("path")
    digest = raw.get("sha256")
    size = raw.get("size_bytes")
    if not isinstance(relative_raw, str) or not relative_raw:
        raise GateError(f"{label} path is invalid")
    relative = Path(relative_raw)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != relative_raw:
        raise GateError(f"{label} path must be a normalized relative path")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise GateError(f"{label} SHA256 is invalid")
    if type(size) is not int or size < 0:
        raise GateError(f"{label} size is invalid")
    candidate = root / relative
    require_regular_file(candidate, label)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise GateError(f"{label} path contains a symlink")
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise GateError(f"{label} resolves outside artifact directory") from error
    if candidate.stat().st_size != size or sha256(candidate) != digest:
        raise GateError(f"{label} byte binding differs")
    return candidate


def sanitize_effective_url(url: str) -> tuple[str, str | None]:
    """Keep redirect identity without persisting a temporary signed query string."""
    parsed = urllib.parse.urlsplit(url)
    if not parsed.query:
        return url, None
    sanitized = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "<redacted>", parsed.fragment))
    return sanitized, hashlib.sha256(url.encode()).hexdigest()


def remaining_seconds(deadline_monotonic: float | None, configured_timeout: float) -> float:
    if configured_timeout <= 0:
        raise GateError("timeout must be positive")
    if deadline_monotonic is None:
        return configured_timeout
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise GateError("gate hard deadline expired")
    return min(configured_timeout, remaining)


def assert_no_bad_patterns(label: str, text: str) -> None:
    lower = text.lower()
    for pat in BAD_LOG_PATTERNS:
        if pat.lower() in lower:
            raise RuntimeError(f"forbidden pattern {pat!r} in {label}")


def run(cmd: list[str], *, cwd: Path | None = None, input: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, input=input, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


def run_evidenced(
    cmd: list[str],
    *,
    out: Path,
    label: str,
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout: int = 120,
    stdout_name: str | None = None,
    gate_deadline_monotonic: float | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    effective_timeout = remaining_seconds(gate_deadline_monotonic, float(timeout))
    deadline = started + effective_timeout
    timed_out = False
    launch_error: str | None = None
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=effective_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout.decode("utf-8", "replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode("utf-8", "replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        completed = subprocess.CompletedProcess(cmd, 124, stdout, stderr)
    except OSError as error:
        launch_error = str(error)
        completed = subprocess.CompletedProcess(cmd, 127, "", str(error))
    finished = time.time()
    stdout_path = out / (stdout_name or f"{label}.stdout")
    stderr_path = out / f"{label}.stderr"
    command_path = out / f"{label}.command.json"
    stdout_path.write_text(completed.stdout or "", errors="replace")
    stderr_path.write_text(completed.stderr or "", errors="replace")
    receipt = {
        "command": cmd,
        "cwd": str(cwd.resolve()) if cwd is not None else None,
        "timeout_sec": round(effective_timeout, 6),
        "timed_out": timed_out,
        "launch_error": launch_error,
        "stdin_sha256": hashlib.sha256(input_text.encode()).hexdigest() if input_text is not None else None,
        "stdin_size_bytes": len(input_text.encode()) if input_text is not None else 0,
        **timing_receipt(started, deadline, finished, completed.returncode),
    }
    write_json(command_path, receipt)
    bundle = {
        "receipt": evidence_ref(out, command_path),
        "stdout": evidence_ref(out, stdout_path),
        "stderr": evidence_ref(out, stderr_path),
    }
    return completed, bundle


def write_progress_row(handle, *, byte_count: int, started_monotonic: float, complete: bool, attempt: int) -> dict[str, Any]:
    row = {
        "bytes": byte_count,
        "elapsed_sec": round(max(0.0, time.monotonic() - started_monotonic), 6),
        "complete": complete,
        "attempt": attempt,
        "timestamp": utc_timestamp(),
    }
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    return row


def download_with_receipt(
    url: str,
    path: Path,
    *,
    retries: int = 5,
    timeout: int = 60,
    total_timeout_seconds: float = DEFAULT_ASSET_DOWNLOAD_TIMEOUT_SECONDS,
    progress_path: Path | None = None,
    evidence_root: Path | None = None,
    hard_deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    if retries < 1:
        raise GateError("download retries must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = progress_path or path.with_name(f"{path.name}.progress.jsonl")
    evidence_root = evidence_root or path.parent
    overall_started = time.time()
    overall_started_monotonic = time.monotonic()
    allowed_seconds = remaining_seconds(hard_deadline_monotonic, total_timeout_seconds)
    overall_deadline_monotonic = overall_started_monotonic + allowed_seconds
    overall_deadline = overall_started + allowed_seconds
    attempts: list[dict[str, Any]] = []
    last: Exception | None = None
    partial_path = path.with_name(f".{path.name}.partial")
    partial_path.unlink(missing_ok=True)
    for index in range(1, retries + 1):
        if time.monotonic() >= overall_deadline_monotonic:
            last = TimeoutError(f"download hard deadline exceeded after {allowed_seconds:.3f}s")
            break
        started = time.time()
        started_monotonic = time.monotonic()
        http_status: int | None = None
        effective_url: str | None = None
        effective_url_sha256: str | None = None
        headers: dict[str, str] = {}
        byte_count = 0
        try:
            progress_path.unlink(missing_ok=True)
            socket_timeout = max(0.001, min(float(timeout), overall_deadline_monotonic - time.monotonic()))
            with urllib.request.urlopen(url, timeout=socket_timeout) as r, partial_path.open("wb") as output, progress_path.open("w", buffering=1) as progress:
                http_status = getattr(r, "status", None)
                raw_effective_url = r.geturl()
                effective_url, effective_url_sha256 = sanitize_effective_url(raw_effective_url)
                for name in ("Content-Length", "Content-Type", "ETag", "Last-Modified"):
                    value = r.headers.get(name)
                    if value is not None:
                        headers[name.lower()] = value
                write_progress_row(progress, byte_count=0, started_monotonic=started_monotonic, complete=False, attempt=index)
                last_report_bytes = 0
                last_report_monotonic = started_monotonic
                while True:
                    if time.monotonic() >= overall_deadline_monotonic:
                        raise TimeoutError(f"download hard deadline exceeded after {allowed_seconds:.3f}s")
                    chunk = r.read(DOWNLOAD_CHUNK_BYTES)
                    if time.monotonic() > overall_deadline_monotonic:
                        raise TimeoutError(f"download hard deadline exceeded after {allowed_seconds:.3f}s")
                    if not chunk:
                        break
                    output.write(chunk)
                    byte_count += len(chunk)
                    now_monotonic = time.monotonic()
                    if byte_count - last_report_bytes >= PROGRESS_BYTES_INTERVAL or now_monotonic - last_report_monotonic >= PROGRESS_SECONDS_INTERVAL:
                        row = write_progress_row(progress, byte_count=byte_count, started_monotonic=started_monotonic, complete=False, attempt=index)
                        print(
                            f"FERRUM DOWNLOAD PROGRESS: {path.name} bytes={row['bytes']} elapsed_sec={row['elapsed_sec']}",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_report_bytes = byte_count
                        last_report_monotonic = now_monotonic
                output.flush()
                os.fsync(output.fileno())
                os.replace(partial_path, path)
                final_row = write_progress_row(progress, byte_count=byte_count, started_monotonic=started_monotonic, complete=True, attempt=index)
                print(
                    f"FERRUM DOWNLOAD COMPLETE: {path.name} bytes={final_row['bytes']} elapsed_sec={final_row['elapsed_sec']}",
                    file=sys.stderr,
                    flush=True,
                )
            finished = time.time()
            attempt = {
                "attempt": index,
                "requested_url": url,
                "effective_url": effective_url,
                "effective_url_sha256": effective_url_sha256,
                "http_status": http_status,
                "response_headers": headers,
                "received_size_bytes": byte_count,
                "error": None,
                **timing_receipt(started, overall_deadline, finished, 0),
            }
            attempts.append(attempt)
            return {
                "source": "public-url",
                "http_performed": True,
                "requested_url": url,
                "requested_path": None,
                "effective_url": effective_url,
                "effective_url_sha256": effective_url_sha256,
                "http_status": http_status,
                "response_headers": headers,
                "received_size_bytes": path.stat().st_size,
                "attempts": attempts,
                "progress": evidence_ref(evidence_root, progress_path),
                **timing_receipt(overall_started, overall_deadline, finished, 0),
            }
        except Exception as e:
            last = e
            finished = time.time()
            attempts.append(
                {
                    "attempt": index,
                    "requested_url": url,
                    "effective_url": effective_url,
                    "effective_url_sha256": effective_url_sha256,
                    "http_status": http_status,
                    "response_headers": headers,
                    "received_size_bytes": byte_count,
                    "error": f"{type(e).__name__}: {e}",
                    **timing_receipt(started, overall_deadline, finished, 1),
                }
            )
            partial_path.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            if time.monotonic() >= overall_deadline_monotonic:
                break
            if index < retries:
                time.sleep(min(2.0, max(0.0, overall_deadline_monotonic - time.monotonic())))
    finished = time.time()
    error = RuntimeError(f"download failed: {url}: {last}")
    progress_reference = evidence_ref(evidence_root, progress_path) if progress_path.is_file() and not progress_path.is_symlink() else None
    setattr(
        error,
        "download_receipt",
        {
            "source": "public-url",
            "http_performed": True,
            "requested_url": url,
            "requested_path": None,
            "effective_url": None,
            "effective_url_sha256": None,
            "http_status": None,
            "response_headers": {},
            "received_size_bytes": 0,
            "attempts": attempts,
            "progress": progress_reference,
            **timing_receipt(overall_started, overall_deadline, finished, 1),
        },
    )
    raise error


def download(url: str, path: Path) -> None:
    """Backward-compatible download helper; release paths retain the receipt."""
    download_with_receipt(url, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def official_asset(version: str, asset: str) -> str:
    return f"{REPO}/releases/download/v{version}/{asset}"


def parse_checksum_file(path: Path, *, expected_name: str) -> str:
    require_regular_file(path, "checksum file")
    fields = path.read_text(errors="strict").strip().split()
    if not fields or re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]) is None:
        raise GateError(f"invalid SHA256 sidecar: {path}")
    if len(fields) > 1 and fields[-1].lstrip("*") != expected_name:
        raise GateError(f"SHA256 sidecar names a different asset: {fields[-1]}")
    return fields[0].lower()


def safe_extract_tarball(
    archive: Path,
    destination: Path,
    *,
    timeout_seconds: float = DEFAULT_EXTRACTION_TIMEOUT_SECONDS,
    progress_path: Path | None = None,
    hard_deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    started = time.time()
    started_monotonic = time.monotonic()
    allowed_seconds = remaining_seconds(hard_deadline_monotonic, timeout_seconds)
    deadline_monotonic = started_monotonic + allowed_seconds
    deadline = started + allowed_seconds
    destination_resolved = destination.resolve(strict=True)
    progress_path = progress_path or destination.parent / "asset.extraction.progress.jsonl"
    byte_count = 0
    members_completed = 0
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        if not members:
            raise GateError(f"empty release archive: {archive}")
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise GateError(f"unsafe archive path: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise GateError(f"release archive contains a link/device: {member.name}")
            target = destination / member_path
            try:
                target.resolve(strict=False).relative_to(destination_resolved)
            except ValueError as error:
                raise GateError(f"archive member escapes destination: {member.name}") from error
            if not member.isdir() and not member.isfile():
                raise GateError(f"release archive contains an unsupported member: {member.name}")
        with progress_path.open("w", buffering=1) as progress:
            write_progress_row(progress, byte_count=0, started_monotonic=started_monotonic, complete=False, attempt=1)
            last_report_bytes = 0
            last_report_monotonic = started_monotonic
            for member in members:
                if time.monotonic() >= deadline_monotonic:
                    raise TimeoutError(f"archive extraction hard deadline exceeded after {allowed_seconds:.3f}s")
                target = destination / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    members_completed += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tf.extractfile(member)
                if source is None:
                    raise GateError(f"could not read archive member: {member.name}")
                with source, target.open("wb") as output:
                    while True:
                        if time.monotonic() >= deadline_monotonic:
                            raise TimeoutError(f"archive extraction hard deadline exceeded after {allowed_seconds:.3f}s")
                        chunk = source.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        output.write(chunk)
                        byte_count += len(chunk)
                        now_monotonic = time.monotonic()
                        if byte_count - last_report_bytes >= PROGRESS_BYTES_INTERVAL or now_monotonic - last_report_monotonic >= PROGRESS_SECONDS_INTERVAL:
                            row = write_progress_row(progress, byte_count=byte_count, started_monotonic=started_monotonic, complete=False, attempt=1)
                            print(
                                f"FERRUM EXTRACTION PROGRESS: {archive.name} bytes={row['bytes']} elapsed_sec={row['elapsed_sec']}",
                                file=sys.stderr,
                                flush=True,
                            )
                            last_report_bytes = byte_count
                            last_report_monotonic = now_monotonic
                    output.flush()
                    os.fsync(output.fileno())
                target.chmod(member.mode & 0o777)
                members_completed += 1
            final_row = write_progress_row(progress, byte_count=byte_count, started_monotonic=started_monotonic, complete=True, attempt=1)
            print(
                f"FERRUM EXTRACTION COMPLETE: {archive.name} bytes={final_row['bytes']} elapsed_sec={final_row['elapsed_sec']}",
                file=sys.stderr,
                flush=True,
            )
    finished = time.time()
    return {
        "operation": "bounded-safe-tar-extract",
        "archive": archive.name,
        "timeout_seconds": allowed_seconds,
        "extracted_size_bytes": byte_count,
        "members_total": len(members),
        "members_completed": members_completed,
        **timing_receipt(started, deadline, finished, 0),
    }


def local_source_receipt(
    source: Path,
    *,
    source_sha256: str,
    source_size_bytes: int,
    copied_sha256: str,
    copied_size_bytes: int,
    started: float,
    deadline: float,
    finished: float,
    progress: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "asset-path",
        "classification": "local-prepublication",
        "http_performed": False,
        "requested_url": None,
        "effective_url": None,
        "effective_url_sha256": None,
        "http_status": None,
        "requested_path": str(source),
        "resolved_path": str(source.resolve(strict=True)),
        "source_sha256": source_sha256,
        "source_size_bytes": source_size_bytes,
        "copied_sha256": copied_sha256,
        "copied_size_bytes": copied_size_bytes,
        "progress": progress,
        **timing_receipt(started, deadline, finished, 0),
    }


def require_safe_output_path(root: Path, path: Path, label: str) -> Path:
    """Reject output paths that could follow a symlink or escape their evidence root."""
    if not root.is_dir() or root.is_symlink():
        raise GateError(f"{label} root is invalid: {root}")
    root_resolved = root.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise GateError(f"{label} escapes artifact directory: {path}") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise GateError(f"{label} is not a normalized artifact path: {path}")
    cursor = root
    for part in relative.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise GateError(f"{label} parent contains a symlink: {cursor}")
        if not cursor.is_dir():
            raise GateError(f"{label} parent is not a directory: {cursor}")
    if path.is_symlink():
        raise GateError(f"{label} must not be a symlink: {path}")
    if path.exists() and not path.is_file():
        raise GateError(f"{label} is not a regular file: {path}")
    try:
        path.parent.resolve(strict=True).relative_to(root_resolved)
    except ValueError as error:
        raise GateError(f"{label} parent resolves outside artifact directory: {path}") from error
    return path


def sha256_before_deadline(path: Path, *, deadline_monotonic: float, label: str) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if time.monotonic() >= deadline_monotonic:
                raise TimeoutError(f"{label} hard deadline exceeded")
            chunk = handle.read(DOWNLOAD_CHUNK_BYTES)
            if time.monotonic() > deadline_monotonic:
                raise TimeoutError(f"{label} hard deadline exceeded")
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def copy_local_asset_with_receipt(
    source: Path,
    destination: Path,
    *,
    progress_path: Path,
    evidence_root: Path,
    total_timeout_seconds: float,
    hard_deadline_monotonic: float | None,
) -> dict[str, Any]:
    """Copy a staged asset atomically while recording real byte progress."""
    require_regular_file(source, "asset path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require_safe_output_path(evidence_root, destination, "local asset destination")
    require_safe_output_path(evidence_root, progress_path, "local asset copy progress")
    partial_path = destination.with_name(f".{destination.name}.partial")
    require_safe_output_path(evidence_root, partial_path, "local asset partial")
    # Cleanup is deliberately before deadline evaluation so an expired retry
    # cannot leave a stale final/partial file looking like fresh evidence.
    partial_path.unlink(missing_ok=True)
    destination.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    started = time.time()
    started_monotonic = time.monotonic()
    allowed_seconds = remaining_seconds(hard_deadline_monotonic, total_timeout_seconds)
    deadline_monotonic = started_monotonic + allowed_seconds
    deadline = started + allowed_seconds
    byte_count = 0
    source_digest = hashlib.sha256()
    source_size_bytes = source.stat().st_size
    try:
        with source.open("rb") as input_file, partial_path.open("wb") as output, progress_path.open(
            "w", buffering=1
        ) as progress:
            write_progress_row(
                progress,
                byte_count=0,
                started_monotonic=started_monotonic,
                complete=False,
                attempt=1,
            )
            last_report_bytes = 0
            last_report_monotonic = started_monotonic
            while True:
                if time.monotonic() >= deadline_monotonic:
                    raise TimeoutError(
                        f"local asset copy hard deadline exceeded after {allowed_seconds:.3f}s"
                    )
                chunk = input_file.read(DOWNLOAD_CHUNK_BYTES)
                if time.monotonic() > deadline_monotonic:
                    raise TimeoutError(
                        f"local asset copy hard deadline exceeded after {allowed_seconds:.3f}s"
                    )
                if not chunk:
                    break
                output.write(chunk)
                source_digest.update(chunk)
                byte_count += len(chunk)
                now_monotonic = time.monotonic()
                if (
                    byte_count - last_report_bytes >= PROGRESS_BYTES_INTERVAL
                    or now_monotonic - last_report_monotonic >= PROGRESS_SECONDS_INTERVAL
                ):
                    row = write_progress_row(
                        progress,
                        byte_count=byte_count,
                        started_monotonic=started_monotonic,
                        complete=False,
                        attempt=1,
                    )
                    print(
                        f"FERRUM LOCAL COPY PROGRESS: {destination.name} "
                        f"bytes={row['bytes']} elapsed_sec={row['elapsed_sec']}",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_report_bytes = byte_count
                    last_report_monotonic = now_monotonic
            output.flush()
            os.fsync(output.fileno())
            if time.monotonic() > deadline_monotonic:
                raise TimeoutError(
                    f"local asset copy hard deadline exceeded after {allowed_seconds:.3f}s"
                )
            shutil.copystat(source, partial_path, follow_symlinks=False)
            os.replace(partial_path, destination)
            if byte_count != source_size_bytes or destination.stat().st_size != byte_count:
                raise GateError("local asset size changed during copy")
            copied_sha256 = sha256_before_deadline(
                destination,
                deadline_monotonic=deadline_monotonic,
                label="local copied asset SHA256",
            )
            source_sha256 = source_digest.hexdigest()
            if copied_sha256 != source_sha256:
                raise GateError("local copied asset SHA256 differs from copied source bytes")
            if time.monotonic() > deadline_monotonic:
                raise TimeoutError(
                    f"local asset copy hard deadline exceeded after {allowed_seconds:.3f}s"
                )
            final_row = write_progress_row(
                progress,
                byte_count=byte_count,
                started_monotonic=started_monotonic,
                complete=True,
                attempt=1,
            )
            print(
                f"FERRUM LOCAL COPY COMPLETE: {destination.name} "
                f"bytes={final_row['bytes']} elapsed_sec={final_row['elapsed_sec']}",
                file=sys.stderr,
                flush=True,
            )
        progress_ref = evidence_ref(evidence_root, progress_path)
        if time.monotonic() > deadline_monotonic:
            raise TimeoutError(
                f"local asset copy hard deadline exceeded after {allowed_seconds:.3f}s"
            )
        finished = time.time()
        receipt = local_source_receipt(
            source,
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            copied_sha256=copied_sha256,
            copied_size_bytes=byte_count,
            started=started,
            deadline=deadline,
            finished=finished,
            progress=progress_ref,
        )
        if time.monotonic() > deadline_monotonic:
            raise TimeoutError(
                f"local asset copy hard deadline exceeded after {allowed_seconds:.3f}s"
            )
        return receipt
    except Exception:
        partial_path.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise


def prepare_tarball(
    version: str,
    asset: str,
    out: Path,
    expected_sha: str | None,
    asset_path: Path | None,
    metadata: dict[str, Any] | None = None,
    *,
    asset_download_timeout_seconds: float = DEFAULT_ASSET_DOWNLOAD_TIMEOUT_SECONDS,
    extraction_timeout_seconds: float = DEFAULT_EXTRACTION_TIMEOUT_SECONDS,
    gate_deadline_monotonic: float | None = None,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / asset
    sha_path = out / f"{asset}.sha256"
    source_receipt_path = out / "asset.source.receipt.json"
    checksum_receipt_path = out / "asset.checksum.receipt.json"
    requested_url: str | None = None
    requested_path: str | None = None
    checksum_kind: str
    if asset_path is not None:
        require_regular_file(asset_path, "asset path")
        if asset_path.resolve(strict=True) == tar_path.resolve(strict=False):
            raise GateError("asset path must not be the gate output archive")
        write_json(
            source_receipt_path,
            copy_local_asset_with_receipt(
                asset_path,
                tar_path,
                progress_path=out / "asset.local-copy.progress.jsonl",
                evidence_root=out,
                total_timeout_seconds=asset_download_timeout_seconds,
                hard_deadline_monotonic=gate_deadline_monotonic,
            ),
        )
        requested_path = str(asset_path)
        local_sha_path = asset_path.with_name(f"{asset_path.name}.sha256")
        if local_sha_path.is_symlink():
            raise GateError(f"adjacent SHA256 sidecar must not be a symlink: {local_sha_path}")
        if local_sha_path.is_file():
            shutil.copy2(local_sha_path, sha_path)
            adjacent_sha = parse_checksum_file(sha_path, expected_name=asset)
            if expected_sha is not None and adjacent_sha != expected_sha.lower():
                raise GateError("sha256 mismatch: --sha256 differs from adjacent SHA256 sidecar")
            expected_sha = adjacent_sha
            checksum_kind = "adjacent-file"
        elif expected_sha is not None:
            if re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha) is None:
                raise GateError("--sha256 must contain 64 hexadecimal characters")
            expected_sha = expected_sha.lower()
            sha_path.write_text(f"{expected_sha}  {asset}\n")
            checksum_kind = "argument"
        else:
            raise RuntimeError(f"missing sha256 for local asset: pass --sha256 or provide {sha_path.name}")
        now = time.time()
        checksum_receipt = {
            "source": checksum_kind,
            "http_performed": False,
            "requested_url": None,
            "effective_url": None,
            "effective_url_sha256": None,
            "http_status": None,
            "requested_path": str(local_sha_path) if checksum_kind == "adjacent-file" else None,
            "sha256": expected_sha,
            **timing_receipt(now, now + 1, now, 0),
        }
        write_json(checksum_receipt_path, checksum_receipt)
    else:
        requested_url = official_asset(version, asset)
        asset_receipt = download_with_receipt(
            requested_url,
            tar_path,
            total_timeout_seconds=asset_download_timeout_seconds,
            progress_path=out / "asset.download.progress.jsonl",
            evidence_root=out,
            hard_deadline_monotonic=gate_deadline_monotonic,
        )
        write_json(source_receipt_path, asset_receipt)
        checksum_url = official_asset(version, f"{asset}.sha256")
        checksum_receipt = download_with_receipt(
            checksum_url,
            sha_path,
            total_timeout_seconds=asset_download_timeout_seconds,
            progress_path=out / "asset.checksum.download.progress.jsonl",
            evidence_root=out,
            hard_deadline_monotonic=gate_deadline_monotonic,
        )
        write_json(checksum_receipt_path, checksum_receipt)
        downloaded_sha = parse_checksum_file(sha_path, expected_name=asset)
        if expected_sha is not None and downloaded_sha != expected_sha.lower():
            raise GateError("--sha256 differs from canonical public SHA256 sidecar")
        expected_sha = downloaded_sha
        checksum_kind = "public-url"
    if expected_sha is None:  # Defensive: every branch above establishes it.
        raise GateError("expected SHA256 was not established")
    actual = sha256(tar_path)
    if actual != expected_sha:
        raise RuntimeError(f"sha256 mismatch for {asset}: actual={actual} expected={expected_sha}")
    extraction_dir = out / "unpacked"
    extraction_dir.mkdir()
    extraction_progress_path = out / "asset.extraction.progress.jsonl"
    extraction_receipt = safe_extract_tarball(
        tar_path,
        extraction_dir,
        timeout_seconds=extraction_timeout_seconds,
        progress_path=extraction_progress_path,
        hard_deadline_monotonic=gate_deadline_monotonic,
    )
    extraction_receipt_path = out / "asset.extraction.receipt.json"
    extraction_receipt["progress"] = evidence_ref(out, extraction_progress_path)
    write_json(extraction_receipt_path, extraction_receipt)
    bin_path = extraction_dir / "ferrum"
    if not bin_path.is_file() or bin_path.is_symlink():
        matches = [candidate for candidate in extraction_dir.rglob("ferrum") if candidate.is_file() and not candidate.is_symlink()]
        if len(matches) == 1:
            bin_path = matches[0]
        elif len(matches) > 1:
            raise GateError(f"multiple ferrum binaries found after extracting {asset}")
    if not bin_path.is_file() or bin_path.is_symlink():
        raise RuntimeError(f"ferrum binary not found after extracting {asset}")
    bin_path.chmod(bin_path.stat().st_mode | 0o111)
    if metadata is not None:
        metadata.update(
            {
                "source": "asset-path" if asset_path is not None else "public-url",
                "classification": "local-prepublication" if asset_path is not None else "canonical-public-release",
                "name": asset,
                "requested_url": requested_url,
                "requested_path": requested_path,
                "canonical_public_url": official_asset(version, asset),
                "sha256": actual,
                "size_bytes": tar_path.stat().st_size,
                "archive": evidence_ref(out, tar_path),
                "unpacked_binary": evidence_ref(out, bin_path),
                "source_receipt": evidence_ref(out, source_receipt_path),
                "checksum": {
                    "source": checksum_kind,
                    "sha256": expected_sha,
                    "sidecar": evidence_ref(out, sha_path),
                    "receipt": evidence_ref(out, checksum_receipt_path),
                },
                "extraction_receipt": evidence_ref(out, extraction_receipt_path),
            }
        )
    return bin_path


def assert_version(
    bin_path: Path,
    version: str,
    out: Path | None = None,
    *,
    label: str = "version",
    gate_deadline_monotonic: float | None = None,
) -> dict[str, Any] | None:
    if out is None:
        p = run([str(bin_path), "--version"], timeout=20)
        bundle = None
    else:
        p, bundle = run_evidenced(
            [str(bin_path), "--version"],
            out=out,
            label=label,
            timeout=20,
            gate_deadline_monotonic=gate_deadline_monotonic,
        )
    if p.returncode != 0 or f"ferrum {version}" not in (p.stdout + p.stderr):
        raise RuntimeError(f"version check failed: rc={p.returncode} out={p.stdout} err={p.stderr}")
    return bundle


def assert_help(
    bin_path: Path,
    out: Path,
    *,
    label: str = "help",
    gate_deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    p, bundle = run_evidenced(
        [str(bin_path), "--help"],
        out=out,
        label=label,
        timeout=20,
        gate_deadline_monotonic=gate_deadline_monotonic,
    )
    combined = p.stdout + "\n" + p.stderr
    if p.returncode != 0 or "ferrum" not in combined.lower() or "usage" not in combined.lower():
        raise RuntimeError(f"help check failed: rc={p.returncode}")
    return bundle


def cli_gate(
    bin_path: Path,
    model: str,
    out: Path,
    *,
    gate_deadline_monotonic: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    text = "\n".join([
        "本轮句子是 ferrum-blue。只回答 OK",
        "上一条用户消息里的 ferrum 开头短语是什么？只输出短语，不要输出 OK",
        "/clear",
        "123+456 等于多少？只输出数字",
        "/bye",
        "",
    ])
    cmd = [str(bin_path), "run", model, "--disable-thinking"]
    p, command_evidence = run_evidenced(
        cmd,
        out=out,
        label="cli",
        input_text=text,
        timeout=180,
        gate_deadline_monotonic=gate_deadline_monotonic,
    )
    assert_no_bad_patterns("cli output", p.stdout + "\n" + p.stderr)
    combined = re.sub(r"<think>.*?</think>", "", p.stdout + "\n" + p.stderr, flags=re.S)
    ok = p.returncode == 0 and "ferrum-blue" in combined and "579" in combined
    if not ok:
        raise RuntimeError("CLI gate failed: expected ferrum-blue and 579")
    return (
        {
            "passed": True,
            "has_context": True,
            "has_math": True,
            "disable_thinking": True,
        },
        command_evidence,
    )


def post(base: str, payload: dict, timeout: int = 120) -> tuple[int, str]:
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def post_evidenced(
    base: str,
    payload: dict,
    *,
    out: Path,
    label: str,
    timeout: int = 120,
    gate_deadline_monotonic: float | None = None,
) -> tuple[int, str, dict[str, Any]]:
    started = time.time()
    effective_timeout = remaining_seconds(gate_deadline_monotonic, float(timeout))
    deadline = started + effective_timeout
    status, body = post(base, payload, timeout=effective_timeout)
    finished = time.time()
    body_path = out / f"{label}.response"
    receipt_path = out / f"{label}.receipt.json"
    body_path.write_text(body, errors="replace")
    receipt = {
        "method": "POST",
        "requested_url": base + "/v1/chat/completions",
        "effective_url": base + "/v1/chat/completions",
        "http_status": status,
        "request_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "response_size_bytes": len(body.encode()),
        **timing_receipt(started, deadline, finished, 0),
    }
    write_json(receipt_path, receipt)
    return status, body, {
        "receipt": evidence_ref(out, receipt_path),
        "response": evidence_ref(out, body_path),
    }


def wait_health(port: int) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not become healthy")


def wait_health_evidenced(
    port: int,
    *,
    out: Path,
    timeout: int = 180,
    gate_deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    started = time.time()
    started_monotonic = time.monotonic()
    effective_timeout = remaining_seconds(gate_deadline_monotonic, float(timeout))
    deadline_monotonic = started_monotonic + effective_timeout
    deadline = started + effective_timeout
    attempts: list[dict[str, Any]] = []
    status: int | None = None
    while time.monotonic() < deadline_monotonic:
        attempt_started = time.time()
        attempt_timeout = max(0.001, min(2.0, deadline_monotonic - time.monotonic()))
        attempt_deadline = min(deadline, attempt_started + attempt_timeout)
        error: str | None = None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=attempt_timeout) as response:
                status = response.status
                response.read()
            rc = 0
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            rc = 1
        attempt_finished = time.time()
        attempts.append(
            {
                "requested_url": f"http://127.0.0.1:{port}/health",
                "effective_url": f"http://127.0.0.1:{port}/health" if rc == 0 else None,
                "http_status": status if rc == 0 else None,
                "error": error,
                **timing_receipt(attempt_started, attempt_deadline, attempt_finished, rc),
            }
        )
        if rc == 0 and status == 200:
            receipt = {
                "method": "GET",
                "requested_url": f"http://127.0.0.1:{port}/health",
                "effective_url": f"http://127.0.0.1:{port}/health",
                "http_status": 200,
                "attempts": attempts,
                **timing_receipt(started, deadline, attempt_finished, 0),
            }
            path = out / "serve.health.receipt.json"
            write_json(path, receipt)
            return evidence_ref(out, path)
        time.sleep(min(1.0, max(0.0, deadline_monotonic - time.monotonic())))
    finished = time.time()
    receipt = {
        "method": "GET",
        "requested_url": f"http://127.0.0.1:{port}/health",
        "effective_url": None,
        "http_status": None,
        "attempts": attempts,
        **timing_receipt(started, deadline, finished, 1),
    }
    path = out / "serve.health.receipt.json"
    write_json(path, receipt)
    raise RuntimeError("server did not become healthy")


def serve_gate(
    bin_path: Path,
    model_path: str,
    model_name: str,
    out: Path,
    port: int,
    api_extra: bool,
    *,
    gate_deadline_monotonic: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    log = out / "serve.log"
    serve_cmd = [
        str(bin_path),
        "serve",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--disable-thinking",
    ]
    started = time.time()
    allowed_seconds = remaining_seconds(gate_deadline_monotonic, 1200.0)
    deadline = started + allowed_seconds
    with log.open("wb") as f:
        proc = subprocess.Popen(serve_cmd, stdout=f, stderr=subprocess.STDOUT)
    http_evidence: dict[str, Any] = {}
    result: dict[str, Any] | None = None
    try:
        http_evidence["health"] = wait_health_evidenced(
            port,
            out=out,
            gate_deadline_monotonic=gate_deadline_monotonic,
        )
        common = {
            "model": model_name,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        s1, b1, http_evidence["math"] = post_evidenced(f"http://127.0.0.1:{port}", {**common, "messages": [{"role": "user", "content": "123+456 等于多少？只输出数字"}], "max_tokens": 256}, out=out, label="serve.math", gate_deadline_monotonic=gate_deadline_monotonic)
        c1 = json.loads(b1)["choices"][0]["message"].get("content", "") if s1 == 200 else b1
        s2, b2, http_evidence["multiturn"] = post_evidenced(f"http://127.0.0.1:{port}", {**common, "messages": [{"role": "user", "content": "本轮短语是 ferrum-blue。只回答 OK"}, {"role": "assistant", "content": "OK"}, {"role": "user", "content": "第一条用户消息里的 ferrum 开头短语是什么？只输出短语，不要输出 OK"}], "max_tokens": 256}, out=out, label="serve.multiturn", gate_deadline_monotonic=gate_deadline_monotonic)
        c2 = json.loads(b2)["choices"][0]["message"].get("content", "") if s2 == 200 else b2
        s3, b3, http_evidence["boundary"] = post_evidenced(f"http://127.0.0.1:{port}", {**common, "messages": [{"role": "user", "content": "写一个一万字介绍"}], "max_tokens": 1000000}, out=out, label="serve.boundary", gate_deadline_monotonic=gate_deadline_monotonic)
        assert_no_bad_patterns("serve math response", c1)
        assert_no_bad_patterns("serve multiturn response", c2)
        assert_no_bad_patterns("serve boundary response", b3)
        result = {
            "math": [s1, c1],
            "multiturn": [s2, c2],
            "boundary_status": s3,
            "disable_thinking": True,
        }
        if s1 != 200 or "579" not in c1:
            raise RuntimeError("serve math gate failed")
        if s2 != 200 or "ferrum-blue" not in c2:
            raise RuntimeError("serve multi-turn gate failed")
        if s3 != 400:
            raise RuntimeError("serve boundary gate did not return 400")
        if api_extra:
            schema = {"type": "json_schema", "json_schema": {"name": "Answer", "strict": True, "schema": {"type": "object", "additionalProperties": False, "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}}}
            s4, b4, http_evidence["strict_json"] = post_evidenced(f"http://127.0.0.1:{port}", {**common, "messages": [{"role": "user", "content": "计算 123+456。最终答案必须只用 JSON 对象表示，格式为 {\"answer\":579}，不要 Markdown。"}], "response_format": schema, "max_tokens": 256}, out=out, label="serve.strict_json", gate_deadline_monotonic=gate_deadline_monotonic)
            msg = json.loads(b4)["choices"][0]["message"].get("content", "") if s4 == 200 else b4
            assert_no_bad_patterns("serve strict-json response", msg)
            if s4 != 200 or json.loads(msg).get("answer") != 579:
                raise RuntimeError("strict JSON gate failed")
            tools = [{"type": "function", "function": {"name": "calc", "description": "calculate expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
            s5, b5, http_evidence["tool_call"] = post_evidenced(f"http://127.0.0.1:{port}", {**common, "messages": [{"role": "user", "content": "调用工具 calc 计算 123+456"}], "tools": tools, "tool_choice": {"type": "function", "function": {"name": "calc"}}, "max_tokens": 256}, out=out, label="serve.tool_call", gate_deadline_monotonic=gate_deadline_monotonic)
            choice = json.loads(b5)["choices"][0] if s5 == 200 else {}
            assert_no_bad_patterns("serve tool-call response", b5)
            if s5 != 200 or choice.get("finish_reason") != "tool_calls" or "123+456" not in json.dumps(choice, ensure_ascii=False):
                raise RuntimeError("tool call gate failed")
            s6, b6, http_evidence["stream"] = post_evidenced(f"http://127.0.0.1:{port}", {**common, "messages": [{"role": "user", "content": "请用一句话解释 String::from"}], "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 256}, out=out, label="serve.stream", gate_deadline_monotonic=gate_deadline_monotonic)
            assert_no_bad_patterns("serve stream response", b6)
            if s6 != 200 or b6.count("data: [DONE]") != 1 or '"content"' not in b6 or '"usage"' not in b6:
                raise RuntimeError("stream gate failed")
            result.update({"strict_json": [s4, msg], "tool_call": [s5, choice.get("finish_reason")], "stream": [s6, b6.count("data: [DONE]")]})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        finished = time.time()
        command_receipt_path = out / "serve.command.json"
        write_json(
            command_receipt_path,
            {
                "command": serve_cmd,
                "cwd": None,
                "timeout_sec": allowed_seconds,
                "timed_out": finished > deadline,
                "launch_error": None,
                "stdin_sha256": None,
                "stdin_size_bytes": 0,
                "termination_reason": "gate-cleanup",
                **timing_receipt(started, deadline, finished, proc.returncode if proc.returncode is not None else 1),
            },
        )
        text = log.read_text(errors="replace") if log.exists() else ""
        assert_no_bad_patterns(str(log), text)
    if result is None:
        raise RuntimeError("serve gate produced no result")
    return result, {
        "command": {
            "receipt": evidence_ref(out, out / "serve.command.json"),
            "combined_log": evidence_ref(out, log),
        },
        "http": http_evidence,
    }


def check_ldd(
    bin_path: Path,
    out: Path,
    *,
    gate_deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    p, bundle = run_evidenced(
        ["ldd", str(bin_path)],
        out=out,
        label="ldd",
        timeout=30,
        stdout_name="ldd.txt",
        gate_deadline_monotonic=gate_deadline_monotonic,
    )
    text = p.stdout + p.stderr
    if p.returncode != 0 or "not found" in text or re.search(r"torch|python|vllm", text, re.I):
        raise RuntimeError("ldd dependency gate failed")
    return bundle


PASS_PREFIXES = {
    "metal-tarball": "METAL TARBALL GATE PASS: ",
    "cuda-tarball": "CUDA TARBALL GATE PASS: ",
    "homebrew-metal": "HOMEBREW METAL GATE PASS: ",
    "homebrew-cuda-fetch": "HOMEBREW CUDA FETCH GATE PASS: ",
}


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is not valid JSON: {path}") from error


def parse_formula_info(document: Any, *, mode: str, version: str) -> dict[str, str]:
    spec = FORMULAE[mode]
    if not isinstance(document, dict) or set(document).isdisjoint({"formulae"}):
        raise GateError("brew info --json=v2 document is invalid")
    formulae = document.get("formulae")
    if not isinstance(formulae, list) or len(formulae) != 1 or not isinstance(formulae[0], dict):
        raise GateError("brew info must contain exactly one formula")
    formula = formulae[0]
    versions = formula.get("versions")
    urls = formula.get("urls")
    stable = urls.get("stable") if isinstance(urls, dict) else None
    if not isinstance(versions, dict) or not isinstance(stable, dict):
        raise GateError("brew formula stable version/URL is missing")
    checksum = stable.get("checksum")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-fA-F]{64}", checksum) is None:
        raise GateError("brew formula stable checksum is invalid")
    identity = {
        "formula": formula.get("full_name"),
        "name": formula.get("name"),
        "tap": formula.get("tap"),
        "version": str(versions.get("stable")),
        "stable_url": stable.get("url"),
        "stable_checksum": checksum.lower(),
        "asset": spec["asset"],
    }
    expected = {
        "formula": spec["formula"],
        "name": spec["name"],
        "tap": FORMULA_TAP,
        "version": version,
        "stable_url": official_asset(version, spec["asset"]),
        "asset": spec["asset"],
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise GateError(f"brew formula {key} differs: expected={value!r} actual={identity.get(key)!r}")
    return identity


def validate_command_bundle(root: Path, raw: Any, label: str, *, expected_rc: int = 0) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"receipt", "stdout", "stderr"}:
        raise GateError(f"{label} command evidence schema differs")
    receipt_path = resolve_evidence_ref(root, raw["receipt"], f"{label} command receipt")
    resolve_evidence_ref(root, raw["stdout"], f"{label} stdout")
    resolve_evidence_ref(root, raw["stderr"], f"{label} stderr")
    receipt = read_json(receipt_path, f"{label} command receipt")
    validate_timing(receipt, f"{label} command receipt", expected_rc=expected_rc)
    command = receipt.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        raise GateError(f"{label} command line is invalid")
    return receipt


def referenced_text(root: Path, raw: Any, label: str) -> str:
    path = resolve_evidence_ref(root, raw, label)
    try:
        return path.read_text(errors="strict")
    except (OSError, UnicodeError) as error:
        raise GateError(f"{label} is not valid UTF-8 text") from error


def validate_progress_jsonl(root: Path, raw: Any, label: str, *, expected_size: int) -> list[dict[str, Any]]:
    path = resolve_evidence_ref(root, raw, label)
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(errors="strict").splitlines()
    except (OSError, UnicodeError) as error:
        raise GateError(f"{label} is not valid UTF-8 JSONL") from error
    if not lines:
        raise GateError(f"{label} is empty")
    previous_bytes = -1
    previous_elapsed = -1.0
    attempt: int | None = None
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise GateError(f"{label} line {index + 1} is invalid JSON") from error
        if not isinstance(row, dict) or set(row) != {"bytes", "elapsed_sec", "complete", "attempt", "timestamp"}:
            raise GateError(f"{label} line {index + 1} schema differs")
        byte_count = row.get("bytes")
        elapsed = row.get("elapsed_sec")
        current_attempt = row.get("attempt")
        if type(byte_count) is not int or byte_count < 0 or byte_count < previous_bytes:
            raise GateError(f"{label} bytes are not monotonic")
        if not isinstance(elapsed, (int, float)) or elapsed < 0 or elapsed < previous_elapsed:
            raise GateError(f"{label} elapsed time is not monotonic")
        if type(current_attempt) is not int or current_attempt < 1:
            raise GateError(f"{label} attempt is invalid")
        if attempt is None:
            attempt = current_attempt
        elif current_attempt != attempt:
            raise GateError(f"{label} mixes retry attempts")
        parse_timestamp(row.get("timestamp"), f"{label} line {index + 1} timestamp")
        if type(row.get("complete")) is not bool:
            raise GateError(f"{label} complete flag is invalid")
        if index < len(lines) - 1 and row["complete"]:
            raise GateError(f"{label} completed before its terminal row")
        previous_bytes = byte_count
        previous_elapsed = float(elapsed)
        rows.append(row)
    if rows[0]["bytes"] != 0 or rows[0]["complete"]:
        raise GateError(f"{label} initial row differs")
    if rows[-1]["complete"] is not True or rows[-1]["bytes"] != expected_size:
        raise GateError(f"{label} terminal complete/size differs")
    return rows


def validate_http_receipt(root: Path, raw: Any, label: str, *, expected_status: int | None = None) -> dict[str, Any]:
    path = resolve_evidence_ref(root, raw, label)
    receipt = read_json(path, label)
    validate_timing(receipt, label, expected_rc=0)
    if expected_status is not None and receipt.get("http_status") != expected_status:
        raise GateError(f"{label} HTTP status differs")
    requested = receipt.get("requested_url")
    effective = receipt.get("effective_url")
    if not isinstance(requested, str) or not requested or not isinstance(effective, str) or not effective:
        raise GateError(f"{label} requested/effective URL is missing")
    attempts = receipt.get("attempts")
    if attempts is not None:
        if not isinstance(attempts, list) or not attempts:
            raise GateError(f"{label} attempts are invalid")
        for index, attempt in enumerate(attempts):
            validate_timing(attempt, f"{label}.attempts[{index}]")
    return receipt


def validate_serve_evidence(root: Path, raw: Any) -> None:
    if not isinstance(raw, dict) or set(raw) != {"command", "http"}:
        raise GateError("serve evidence schema differs")
    command = raw["command"]
    if not isinstance(command, dict) or set(command) != {"receipt", "combined_log"}:
        raise GateError("serve command evidence schema differs")
    receipt_path = resolve_evidence_ref(root, command["receipt"], "serve command receipt")
    resolve_evidence_ref(root, command["combined_log"], "serve combined log")
    receipt = read_json(receipt_path, "serve command receipt")
    validate_timing(receipt, "serve command receipt")
    if receipt.get("termination_reason") != "gate-cleanup":
        raise GateError("serve command was not terminated by gate cleanup")
    http = raw["http"]
    expected = {"health", "math", "multiturn", "boundary", "strict_json", "tool_call", "stream"}
    if not isinstance(http, dict) or set(http) != expected:
        raise GateError("serve HTTP evidence set differs")
    validate_http_receipt(root, http["health"], "serve health receipt", expected_status=200)
    statuses = {
        "math": 200,
        "multiturn": 200,
        "boundary": 400,
        "strict_json": 200,
        "tool_call": 200,
        "stream": 200,
    }
    for name, expected_status in statuses.items():
        bundle = http[name]
        if not isinstance(bundle, dict) or set(bundle) != {"receipt", "response"}:
            raise GateError(f"serve {name} HTTP evidence schema differs")
        validate_http_receipt(root, bundle["receipt"], f"serve {name} receipt", expected_status=expected_status)
        resolve_evidence_ref(root, bundle["response"], f"serve {name} response")


def validate_asset_evidence(root: Path, raw: Any, *, version: str, mode: str) -> None:
    required = {
        "source",
        "classification",
        "name",
        "requested_url",
        "requested_path",
        "canonical_public_url",
        "sha256",
        "size_bytes",
        "archive",
        "unpacked_binary",
        "source_receipt",
        "checksum",
        "extraction_receipt",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise GateError("asset evidence schema differs")
    expected_asset = "ferrum-macos-aarch64.tar.gz" if mode == "metal-tarball" else "ferrum-linux-x86_64-cuda-sm89.tar.gz"
    canonical = official_asset(version, expected_asset)
    if raw.get("name") != expected_asset or raw.get("canonical_public_url") != canonical:
        raise GateError("release asset name/canonical URL differs")
    archive = resolve_evidence_ref(root, raw["archive"], "release archive")
    binary = resolve_evidence_ref(root, raw["unpacked_binary"], "unpacked ferrum binary")
    if binary.name != "ferrum":
        raise GateError("unpacked binary is not named ferrum")
    if raw.get("sha256") != sha256(archive) or raw.get("size_bytes") != archive.stat().st_size:
        raise GateError("release archive SHA256/size differs")
    source_path = resolve_evidence_ref(root, raw["source_receipt"], "asset source receipt")
    source_receipt = read_json(source_path, "asset source receipt")
    validate_timing(source_receipt, "asset source receipt", expected_rc=0)
    source = raw.get("source")
    if source == "public-url":
        if raw.get("classification") != "canonical-public-release":
            raise GateError("public asset classification differs")
        if raw.get("requested_url") != canonical or raw.get("requested_path") is not None:
            raise GateError("public asset request is not the canonical GitHub URL")
        if source_receipt.get("source") != "public-url" or source_receipt.get("http_performed") is not True:
            raise GateError("public asset lacks an HTTP download receipt")
        if source_receipt.get("requested_url") != canonical or source_receipt.get("requested_path") is not None:
            raise GateError("public download receipt requested a different URL")
        if source_receipt.get("http_status") != 200 or not isinstance(source_receipt.get("effective_url"), str):
            raise GateError("public download did not record a successful effective URL")
        if source_receipt.get("received_size_bytes") != archive.stat().st_size:
            raise GateError("public download receipt size differs")
        validate_progress_jsonl(
            root,
            source_receipt.get("progress"),
            "public asset download progress",
            expected_size=archive.stat().st_size,
        )
        attempts = source_receipt.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise GateError("public download attempts are missing")
        for index, attempt in enumerate(attempts):
            validate_timing(attempt, f"public download attempt {index}")
        if attempts[-1].get("rc") != 0 or attempts[-1].get("http_status") != 200 or attempts[-1].get("requested_url") != canonical:
            raise GateError("public download terminal attempt differs")
    elif source == "asset-path":
        if raw.get("classification") != "local-prepublication":
            raise GateError("local asset classification differs")
        if raw.get("requested_url") is not None or not isinstance(raw.get("requested_path"), str):
            raise GateError("local asset requested URL/path differs")
        expected_nulls = {
            "http_performed": False,
            "requested_url": None,
            "effective_url": None,
            "effective_url_sha256": None,
            "http_status": None,
        }
        if source_receipt.get("source") != "asset-path" or source_receipt.get("classification") != "local-prepublication":
            raise GateError("local source receipt classification differs")
        if any(source_receipt.get(key) != value for key, value in expected_nulls.items()):
            raise GateError("local source receipt must explicitly record that no HTTP occurred")
        if source_receipt.get("requested_path") != raw.get("requested_path"):
            raise GateError("local source receipt path differs")
        if (
            source_receipt.get("source_sha256") != sha256(archive)
            or source_receipt.get("source_size_bytes") != archive.stat().st_size
            or source_receipt.get("copied_sha256") != sha256(archive)
            or source_receipt.get("copied_size_bytes") != archive.stat().st_size
        ):
            raise GateError("local copied asset bytes differ")
        validate_progress_jsonl(
            root,
            source_receipt.get("progress"),
            "local asset copy progress",
            expected_size=archive.stat().st_size,
        )
    else:
        raise GateError("asset source must be public-url or asset-path")
    checksum = raw.get("checksum")
    if not isinstance(checksum, dict) or set(checksum) != {"source", "sha256", "sidecar", "receipt"}:
        raise GateError("asset checksum evidence schema differs")
    sidecar = resolve_evidence_ref(root, checksum["sidecar"], "asset SHA256 sidecar")
    checksum_receipt = resolve_evidence_ref(root, checksum["receipt"], "asset checksum receipt")
    checksum_document = read_json(checksum_receipt, "asset checksum receipt")
    validate_timing(checksum_document, "asset checksum receipt", expected_rc=0)
    parsed = parse_checksum_file(sidecar, expected_name=expected_asset)
    if parsed != sha256(archive) or checksum.get("sha256") != parsed or raw.get("sha256") != parsed:
        raise GateError("asset/sidecar SHA256 binding differs")
    if source == "public-url":
        checksum_url = official_asset(version, f"{expected_asset}.sha256")
        if checksum.get("source") != "public-url" or checksum_document.get("requested_url") != checksum_url:
            raise GateError("public checksum did not use the canonical sidecar URL")
        if checksum_document.get("http_performed") is not True or checksum_document.get("http_status") != 200:
            raise GateError("public checksum HTTP receipt differs")
        validate_progress_jsonl(
            root,
            checksum_document.get("progress"),
            "public checksum download progress",
            expected_size=sidecar.stat().st_size,
        )
        attempts = checksum_document.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise GateError("public checksum HTTP attempts are missing")
        for index, attempt in enumerate(attempts):
            validate_timing(attempt, f"public checksum attempt {index}")
        if attempts[-1].get("rc") != 0 or attempts[-1].get("http_status") != 200 or attempts[-1].get("requested_url") != checksum_url:
            raise GateError("public checksum terminal attempt differs")
    else:
        local_checksum_source = checksum.get("source")
        expected_checksum_fields = {
            "source",
            "http_performed",
            "requested_url",
            "effective_url",
            "effective_url_sha256",
            "http_status",
            "requested_path",
            "sha256",
            "started_at",
            "finished_at",
            "deadline_at",
            "duration_sec",
            "rc",
        }
        if set(checksum_document) != expected_checksum_fields:
            raise GateError("local checksum receipt schema differs")
        if (
            local_checksum_source not in {"argument", "adjacent-file"}
            or checksum_document.get("source") != local_checksum_source
            or checksum_document.get("sha256") != parsed
            or checksum_document.get("http_performed") is not False
            or any(
                checksum_document.get(key) is not None
                for key in (
                    "requested_url",
                    "effective_url",
                    "effective_url_sha256",
                    "http_status",
                )
            )
        ):
            raise GateError("local checksum provenance differs")
        requested_checksum_path = checksum_document.get("requested_path")
        if local_checksum_source == "argument":
            if requested_checksum_path is not None:
                raise GateError("argument checksum receipt unexpectedly records a source path")
        elif (
            not isinstance(requested_checksum_path, str)
            or not requested_checksum_path
            or Path(requested_checksum_path).name != f"{expected_asset}.sha256"
        ):
            raise GateError("adjacent checksum receipt path differs")
    extraction = resolve_evidence_ref(root, raw["extraction_receipt"], "asset extraction receipt")
    extraction_document = read_json(extraction, "asset extraction receipt")
    validate_timing(extraction_document, "asset extraction receipt", expected_rc=0)
    extracted_size = extraction_document.get("extracted_size_bytes")
    if type(extracted_size) is not int or extracted_size < 0:
        raise GateError("asset extraction size is invalid")
    if extraction_document.get("operation") != "bounded-safe-tar-extract":
        raise GateError("asset extraction operation differs")
    if extraction_document.get("members_completed") != extraction_document.get("members_total"):
        raise GateError("asset extraction did not complete every member")
    validate_progress_jsonl(
        root,
        extraction_document.get("progress"),
        "asset extraction progress",
        expected_size=extracted_size,
    )


def validate_formula_evidence(root: Path, raw: Any, *, mode: str, version: str) -> dict[str, str]:
    if not isinstance(raw, dict) or set(raw) != {"brew_info", "identity"}:
        raise GateError("formula evidence schema differs")
    info_path = resolve_evidence_ref(root, raw["brew_info"], "brew info JSON")
    parsed = parse_formula_info(read_json(info_path, "brew info JSON"), mode=mode, version=version)
    if raw.get("identity") != parsed:
        raise GateError("recorded formula identity differs from raw brew info JSON")
    return parsed


def validate_product_checks(checks: dict[str, Any]) -> None:
    cli = checks.get("cli")
    expected_cli = {
        "passed": True,
        "has_context": True,
        "has_math": True,
        "disable_thinking": True,
    }
    if cli != expected_cli:
        raise GateError("CLI correctness checks differ")
    serve = checks.get("serve")
    expected_fields = {
        "math",
        "multiturn",
        "boundary_status",
        "disable_thinking",
        "strict_json",
        "tool_call",
        "stream",
    }
    if not isinstance(serve, dict) or set(serve) != expected_fields:
        raise GateError("serve correctness check set differs")
    math = serve.get("math")
    multiturn = serve.get("multiturn")
    strict_json = serve.get("strict_json")
    try:
        strict_answer = json.loads(strict_json[1]).get("answer")
    except (IndexError, TypeError, json.JSONDecodeError, AttributeError):
        strict_answer = None
    if not (
        isinstance(math, list)
        and len(math) == 2
        and math[0] == 200
        and "579" in str(math[1])
        and isinstance(multiturn, list)
        and len(multiturn) == 2
        and multiturn[0] == 200
        and "ferrum-blue" in str(multiturn[1])
        and serve.get("boundary_status") == 400
        and serve.get("disable_thinking") is True
        and isinstance(strict_json, list)
        and len(strict_json) == 2
        and strict_json[0] == 200
        and strict_answer == 579
        and serve.get("tool_call") == [200, "tool_calls"]
        and serve.get("stream") == [200, 1]
    ):
        raise GateError("serve correctness semantics differ")


def validate_gate_data(data: Any, *, root: Path) -> None:
    fields = {
        "schema_version",
        "artifact_type",
        "status",
        "mode",
        "version",
        "artifact_dir",
        "started_at",
        "finished_at",
        "deadline_at",
        "duration_sec",
        "rc",
        "pass_line",
        "checks",
        "evidence",
    }
    if not isinstance(data, dict) or set(data) != fields:
        raise GateError("release binary gate schema differs")
    if data.get("schema_version") != SCHEMA_VERSION or data.get("artifact_type") != "ferrum_release_binary_gate":
        raise GateError("release binary gate schema/type differs")
    mode = data.get("mode")
    version = data.get("version")
    if data.get("status") != "pass" or mode not in PASS_PREFIXES or not isinstance(version, str) or not version:
        raise GateError("release binary gate status/mode/version differs")
    validate_timing(data, "release binary gate", expected_rc=0)
    artifact_dir = data.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise GateError("release binary gate artifact_dir is missing")
    if data.get("pass_line") != PASS_PREFIXES[mode] + artifact_dir:
        raise GateError("release binary gate exact PASS line differs")
    checks = data.get("checks")
    evidence = data.get("evidence")
    if not isinstance(checks, dict) or not isinstance(evidence, dict):
        raise GateError("release binary checks/evidence is invalid")
    if mode in {"metal-tarball", "cuda-tarball"}:
        if set(checks) != {"version", "cli", "serve"} or checks.get("version") is not True:
            raise GateError("tarball check set differs")
        validate_product_checks(checks)
        command_names = {"version", "cli", "serve"} | ({"ldd"} if mode == "cuda-tarball" else set())
        if set(evidence) != {"asset", "commands"} or not isinstance(evidence["commands"], dict) or set(evidence["commands"]) != command_names:
            raise GateError("tarball evidence command set differs")
        validate_asset_evidence(root, evidence["asset"], version=version, mode=mode)
        version_command = validate_command_bundle(root, evidence["commands"]["version"], "version")
        cli_command = validate_command_bundle(root, evidence["commands"]["cli"], "CLI")
        binary = resolve_evidence_ref(root, evidence["asset"]["unpacked_binary"], "unpacked ferrum binary")
        try:
            version_binary = Path(version_command["command"][0]).resolve(strict=True)
            cli_binary = Path(cli_command["command"][0]).resolve(strict=True)
        except (IndexError, OSError) as error:
            raise GateError("tarball command binary path is invalid") from error
        if version_binary != binary.resolve(strict=True) or version_command["command"][1:] != ["--version"]:
            raise GateError("version command did not execute the unpacked binary")
        if cli_binary != binary.resolve(strict=True) or "run" not in cli_command["command"] or "--disable-thinking" not in cli_command["command"]:
            raise GateError("CLI command did not execute the unpacked binary with documented thinking behavior")
        version_output = referenced_text(root, evidence["commands"]["version"]["stdout"], "version stdout") + referenced_text(root, evidence["commands"]["version"]["stderr"], "version stderr")
        if f"ferrum {version}" not in version_output:
            raise GateError("version command output differs")
        if mode == "cuda-tarball":
            ldd_command = validate_command_bundle(root, evidence["commands"]["ldd"], "ldd")
            if len(ldd_command["command"]) != 2 or ldd_command["command"][0] != "ldd" or Path(ldd_command["command"][1]).resolve(strict=True) != binary.resolve(strict=True):
                raise GateError("ldd did not inspect the unpacked CUDA binary")
        validate_serve_evidence(root, evidence["commands"]["serve"])
        serve_receipt_path = resolve_evidence_ref(root, evidence["commands"]["serve"]["command"]["receipt"], "serve command receipt")
        serve_command = read_json(serve_receipt_path, "serve command receipt").get("command")
        if not isinstance(serve_command, list) or len(serve_command) < 2 or Path(serve_command[0]).resolve(strict=True) != binary.resolve(strict=True) or serve_command[1] != "serve" or "--disable-thinking" not in serve_command:
            raise GateError("serve command did not execute the unpacked binary")
    elif mode == "homebrew-metal":
        if set(checks) != {"version", "help", "cli", "serve"} or checks.get("version") is not True or checks.get("help") is not True:
            raise GateError("Homebrew Metal check set differs")
        validate_product_checks(checks)
        if set(evidence) != {"formula", "installed_binary", "commands"}:
            raise GateError("Homebrew Metal evidence set differs")
        formula = validate_formula_evidence(root, evidence["formula"], mode=mode, version=version)
        commands = evidence["commands"]
        expected_commands = {"reinstall", "brew_info", "command_v", "version", "help", "cli", "serve"}
        if not isinstance(commands, dict) or set(commands) != expected_commands:
            raise GateError("Homebrew Metal command evidence set differs")
        for name in ("reinstall", "brew_info", "command_v", "version", "help", "cli"):
            validate_command_bundle(root, commands[name], f"Homebrew Metal {name}")
        validate_serve_evidence(root, commands["serve"])
        formula_spec = FORMULAE[mode]
        command_receipts = {
            name: read_json(
                resolve_evidence_ref(root, commands[name]["receipt"], f"Homebrew Metal {name} receipt"),
                f"Homebrew Metal {name} receipt",
            )
            for name in ("reinstall", "brew_info", "command_v", "version", "help", "cli")
        }
        if command_receipts["reinstall"].get("command") != ["brew", "reinstall", formula_spec["formula"]]:
            raise GateError("Homebrew Metal reinstall command differs")
        if command_receipts["brew_info"].get("command") != ["brew", "info", "--json=v2", formula_spec["formula"]]:
            raise GateError("Homebrew Metal brew info command differs")
        if commands["brew_info"]["stdout"] != evidence["formula"]["brew_info"]:
            raise GateError("Homebrew Metal raw brew info JSON is not the command stdout")
        if command_receipts["command_v"].get("command") != ["/bin/sh", "-c", "command -v ferrum"]:
            raise GateError("Homebrew Metal command -v receipt differs")
        installed_path = resolve_evidence_ref(root, evidence["installed_binary"], "installed binary identity")
        installed = read_json(installed_path, "installed binary identity")
        required = {"command_v_path", "resolved_path", "command_v_is_symlink", "sha256", "size_bytes", "captured_binary"}
        if not isinstance(installed, dict) or set(installed) != required:
            raise GateError("installed binary identity schema differs")
        captured = resolve_evidence_ref(root, installed["captured_binary"], "captured installed binary")
        if installed.get("sha256") != sha256(captured) or installed.get("size_bytes") != captured.stat().st_size:
            raise GateError("installed binary SHA256/size binding differs")
        command_v_output = referenced_text(root, commands["command_v"]["stdout"], "command -v stdout").strip()
        command_v_path = installed.get("command_v_path")
        if command_v_output != command_v_path or not isinstance(command_v_path, str) or not Path(command_v_path).is_absolute():
            raise GateError("installed binary path differs from command -v output")
        if not isinstance(installed.get("resolved_path"), str) or not Path(installed["resolved_path"]).is_absolute() or type(installed.get("command_v_is_symlink")) is not bool:
            raise GateError("installed binary resolved path metadata differs")
        if command_receipts["version"].get("command") != [command_v_path, "--version"]:
            raise GateError("Homebrew Metal version command did not use command -v path")
        if command_receipts["help"].get("command") != [command_v_path, "--help"]:
            raise GateError("Homebrew Metal help command did not use command -v path")
        version_output = referenced_text(root, commands["version"]["stdout"], "Homebrew version stdout") + referenced_text(root, commands["version"]["stderr"], "Homebrew version stderr")
        help_output = referenced_text(root, commands["help"]["stdout"], "Homebrew help stdout") + referenced_text(root, commands["help"]["stderr"], "Homebrew help stderr")
        if f"ferrum {version}" not in version_output or "usage" not in help_output.lower() or "ferrum" not in help_output.lower():
            raise GateError("Homebrew Metal version/help output differs")
        cli_line = command_receipts["cli"].get("command")
        if not isinstance(cli_line, list) or not cli_line or cli_line[0] != command_v_path or "run" not in cli_line or "--disable-thinking" not in cli_line:
            raise GateError("Homebrew Metal CLI did not use command -v path")
        serve_receipt = read_json(resolve_evidence_ref(root, commands["serve"]["command"]["receipt"], "Homebrew serve receipt"), "Homebrew serve receipt")
        serve_line = serve_receipt.get("command")
        if not isinstance(serve_line, list) or len(serve_line) < 2 or serve_line[0] != command_v_path or serve_line[1] != "serve" or "--disable-thinking" not in serve_line:
            raise GateError("Homebrew Metal serve did not use command -v path")
        if formula["stable_checksum"] == "0" * 64:
            raise GateError("Homebrew Metal formula checksum is a placeholder")
    else:
        if set(checks) != {"fetch", "formula_version"} or checks.get("fetch") is not True or checks.get("formula_version") != version:
            raise GateError("Homebrew CUDA fetch check set differs")
        if set(evidence) != {"formula", "fetched_archive", "commands"}:
            raise GateError("Homebrew CUDA fetch evidence set differs")
        formula = validate_formula_evidence(root, evidence["formula"], mode=mode, version=version)
        commands = evidence["commands"]
        if not isinstance(commands, dict) or set(commands) != {"fetch", "brew_info", "brew_cache"}:
            raise GateError("Homebrew CUDA command evidence set differs")
        command_receipts = {
            name: validate_command_bundle(root, commands[name], f"Homebrew CUDA {name}")
            for name in ("fetch", "brew_info", "brew_cache")
        }
        formula_spec = FORMULAE[mode]
        if command_receipts["fetch"].get("command") != ["brew", "fetch", "--force", formula_spec["formula"]]:
            raise GateError("Homebrew CUDA fetch command differs")
        if command_receipts["brew_info"].get("command") != ["brew", "info", "--json=v2", formula_spec["formula"]]:
            raise GateError("Homebrew CUDA brew info command differs")
        if commands["brew_info"]["stdout"] != evidence["formula"]["brew_info"]:
            raise GateError("Homebrew CUDA raw brew info JSON is not the command stdout")
        if command_receipts["brew_cache"].get("command") != ["brew", "--cache", formula_spec["formula"]]:
            raise GateError("Homebrew CUDA brew cache command differs")
        archive_identity_path = resolve_evidence_ref(root, evidence["fetched_archive"], "fetched archive identity")
        archive_identity = read_json(archive_identity_path, "fetched archive identity")
        required = {"reported_path", "resolved_path", "sha256", "size_bytes", "captured_archive"}
        if not isinstance(archive_identity, dict) or set(archive_identity) != required:
            raise GateError("fetched archive identity schema differs")
        captured = resolve_evidence_ref(root, archive_identity["captured_archive"], "captured fetched archive")
        if archive_identity.get("sha256") != sha256(captured) or archive_identity.get("size_bytes") != captured.stat().st_size:
            raise GateError("fetched archive SHA256/size binding differs")
        if formula["stable_checksum"] != sha256(captured):
            raise GateError("fetched archive differs from the formula checksum")
        if captured.name != formula_spec["asset"]:
            raise GateError("captured fetched archive name differs")
        reported_path = archive_identity.get("reported_path")
        resolved_path = archive_identity.get("resolved_path")
        if not isinstance(reported_path, str) or not Path(reported_path).is_absolute() or not isinstance(resolved_path, str) or not Path(resolved_path).is_absolute():
            raise GateError("fetched archive reported/resolved path is invalid")
        cache_lines = [line.strip() for line in referenced_text(root, commands["brew_cache"]["stdout"], "brew cache stdout").splitlines() if line.strip()]
        normalized_cache_lines = []
        for line in cache_lines:
            if line.startswith("file://"):
                normalized_cache_lines.append(urllib.parse.unquote(urllib.parse.urlsplit(line).path))
            else:
                normalized_cache_lines.append(line)
        if normalized_cache_lines != [reported_path]:
            raise GateError("fetched archive reported path differs from brew --cache output")


def write_gate(
    out: Path,
    mode: str,
    version: str,
    checks: dict[str, Any],
    *,
    evidence: dict[str, Any],
    started: float,
    deadline: float,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    finished = time.time()
    data = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "ferrum_release_binary_gate",
        "status": "pass",
        "mode": mode,
        "version": version,
        "artifact_dir": str(out),
        **timing_receipt(started, deadline, finished, 0),
        "pass_line": PASS_PREFIXES[mode] + str(out),
        "checks": checks,
        "evidence": evidence,
    }
    validate_gate_data(data, root=out)
    write_json(out / "gate.json", data)
    validate_gate_data(read_json(out / "gate.json", "release binary gate"), root=out)
    return data


def gate_tarball(args, *, asset: str, default_model: str, model_name: str, cuda: bool) -> None:
    mode = "cuda-tarball" if cuda else "metal-tarball"
    asset_evidence: dict[str, Any] = {}
    bin_path = prepare_tarball(
        args.version,
        asset,
        args.out,
        args.sha256,
        args.asset_path,
        asset_evidence,
        asset_download_timeout_seconds=args.asset_download_timeout_seconds,
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    version_evidence = assert_version(
        bin_path,
        args.version,
        args.out,
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    if version_evidence is None:
        raise GateError("version evidence was not recorded")
    command_evidence: dict[str, Any] = {"version": version_evidence}
    if cuda:
        command_evidence["ldd"] = check_ldd(
            bin_path,
            args.out,
            gate_deadline_monotonic=args.gate_deadline_monotonic,
        )
    model = args.model or default_model
    cli_checks, command_evidence["cli"] = cli_gate(
        bin_path,
        model,
        args.out,
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    serve_checks, command_evidence["serve"] = serve_gate(
        bin_path,
        model,
        args.model_name or model_name,
        args.out,
        args.port,
        True,
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    checks = {"version": True, "cli": cli_checks, "serve": serve_checks}
    write_gate(
        args.out,
        mode,
        args.version,
        checks,
        evidence={"asset": asset_evidence, "commands": command_evidence},
        started=args.gate_started,
        deadline=args.gate_deadline,
    )
    print(PASS_PREFIXES[mode] + str(args.out))


def capture_external_file(source: Path, *, out: Path, relative: Path) -> Path:
    require_regular_file(source, "captured source file")
    target = out / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise GateError(f"captured evidence target already exists: {target}")
    shutil.copy2(source, target)
    require_regular_file(target, "captured evidence file")
    if sha256(target) != sha256(source) or target.stat().st_size != source.stat().st_size:
        raise GateError(f"captured evidence bytes differ: {source}")
    return target


def brew_info(args, *, mode: str) -> tuple[dict[str, str], dict[str, Any], Path]:
    formula = FORMULAE[mode]["formula"]
    completed, command_evidence = run_evidenced(
        ["brew", "info", "--json=v2", formula],
        out=args.out,
        label="brew_info",
        timeout=60,
        stdout_name="brew_info.json",
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"brew info failed for {formula}")
    info_path = args.out / "brew_info.json"
    identity = parse_formula_info(read_json(info_path, "brew info JSON"), mode=mode, version=args.version)
    if args.sha256 is not None:
        if re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256) is None:
            raise GateError("--sha256 must contain 64 hexadecimal characters")
        if identity["stable_checksum"] != args.sha256.lower():
            raise GateError("Homebrew formula checksum differs from --sha256")
    return identity, command_evidence, info_path


def homebrew_metal(args) -> None:
    mode = "homebrew-metal"
    formula = FORMULAE[mode]["formula"]
    reinstall, reinstall_evidence = run_evidenced(
        ["brew", "reinstall", formula],
        out=args.out,
        label="brew_reinstall",
        timeout=600,
        stdout_name="brew_reinstall.log",
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    if reinstall.returncode != 0:
        raise RuntimeError("brew reinstall failed")
    formula_identity, info_evidence, info_path = brew_info(args, mode=mode)
    command_v, command_v_evidence = run_evidenced(
        ["/bin/sh", "-c", "command -v ferrum"],
        out=args.out,
        label="command_v_ferrum",
        timeout=20,
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    command_v_lines = [line.strip() for line in command_v.stdout.splitlines() if line.strip()]
    if command_v.returncode != 0 or len(command_v_lines) != 1:
        raise GateError("command -v ferrum did not return exactly one path")
    command_path = Path(command_v_lines[0])
    if not command_path.is_absolute() or not command_path.exists():
        raise GateError("command -v ferrum returned a missing/non-absolute path")
    resolved_path = command_path.resolve(strict=True)
    require_regular_file(resolved_path, "resolved installed ferrum binary")
    version_evidence = assert_version(
        command_path,
        args.version,
        args.out,
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    if version_evidence is None:
        raise GateError("Homebrew version evidence was not recorded")
    help_evidence = assert_help(
        command_path,
        args.out,
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    model = args.model or "/Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf"
    cli_checks, cli_evidence = cli_gate(
        command_path,
        model,
        args.out,
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    serve_checks, serve_evidence = serve_gate(
        command_path,
        model,
        args.model_name or "Qwen3-30B-A3B-Q4_K_M",
        args.out,
        args.port,
        True,
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    captured = capture_external_file(resolved_path, out=args.out, relative=Path("installed/ferrum"))
    installed_identity_path = args.out / "installed_binary.json"
    write_json(
        installed_identity_path,
        {
            "command_v_path": str(command_path),
            "resolved_path": str(resolved_path),
            "command_v_is_symlink": command_path.is_symlink(),
            "sha256": sha256(resolved_path),
            "size_bytes": resolved_path.stat().st_size,
            "captured_binary": evidence_ref(args.out, captured),
        },
    )
    checks = {"version": True, "help": True, "cli": cli_checks, "serve": serve_checks}
    evidence = {
        "formula": {
            "brew_info": evidence_ref(args.out, info_path),
            "identity": formula_identity,
        },
        "installed_binary": evidence_ref(args.out, installed_identity_path),
        "commands": {
            "reinstall": reinstall_evidence,
            "brew_info": info_evidence,
            "command_v": command_v_evidence,
            "version": version_evidence,
            "help": help_evidence,
            "cli": cli_evidence,
            "serve": serve_evidence,
        },
    }
    write_gate(args.out, mode, args.version, checks, evidence=evidence, started=args.gate_started, deadline=args.gate_deadline)
    print(PASS_PREFIXES[mode] + str(args.out))


def parse_brew_cache_path(stdout: str) -> Path:
    candidates: list[Path] = []
    for line in stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        if value.startswith("file://"):
            parsed = urllib.parse.urlsplit(value)
            value = urllib.parse.unquote(parsed.path)
        candidate = Path(value)
        if candidate.is_absolute() and candidate.exists():
            candidates.append(candidate)
    if len(candidates) != 1:
        raise GateError("brew --cache did not identify exactly one existing archive")
    require_regular_file(candidates[0], "Homebrew cached archive")
    return candidates[0]


def homebrew_cuda_fetch(args) -> None:
    mode = "homebrew-cuda-fetch"
    formula = FORMULAE[mode]["formula"]
    fetch, fetch_evidence = run_evidenced(
        ["brew", "fetch", "--force", formula],
        out=args.out,
        label="brew_fetch",
        timeout=600,
        stdout_name="brew_fetch.log",
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    if fetch.returncode != 0:
        raise RuntimeError("brew fetch ferrum-cuda failed")
    formula_identity, info_evidence, info_path = brew_info(args, mode=mode)
    cache, cache_evidence = run_evidenced(
        ["brew", "--cache", formula],
        out=args.out,
        label="brew_cache",
        timeout=60,
        gate_deadline_monotonic=args.gate_deadline_monotonic,
    )
    if cache.returncode != 0:
        raise RuntimeError("brew --cache ferrum-cuda failed")
    cached_path = parse_brew_cache_path(cache.stdout)
    resolved_path = cached_path.resolve(strict=True)
    require_regular_file(resolved_path, "resolved Homebrew cached archive")
    if sha256(resolved_path) != formula_identity["stable_checksum"]:
        raise GateError("Homebrew cached CUDA archive differs from formula checksum")
    captured = capture_external_file(
        resolved_path,
        out=args.out,
        relative=Path("fetched") / FORMULAE[mode]["asset"],
    )
    archive_identity_path = args.out / "fetched_archive.json"
    write_json(
        archive_identity_path,
        {
            "reported_path": str(cached_path),
            "resolved_path": str(resolved_path),
            "sha256": sha256(resolved_path),
            "size_bytes": resolved_path.stat().st_size,
            "captured_archive": evidence_ref(args.out, captured),
        },
    )
    evidence = {
        "formula": {
            "brew_info": evidence_ref(args.out, info_path),
            "identity": formula_identity,
        },
        "fetched_archive": evidence_ref(args.out, archive_identity_path),
        "commands": {
            "fetch": fetch_evidence,
            "brew_info": info_evidence,
            "brew_cache": cache_evidence,
        },
    }
    checks = {"fetch": True, "formula_version": args.version}
    write_gate(args.out, mode, args.version, checks, evidence=evidence, started=args.gate_started, deadline=args.gate_deadline)
    print(PASS_PREFIXES[mode] + str(args.out))


def selftest_command_bundle(
    out: Path,
    label: str,
    *,
    stdout: str = "selftest\n",
    command: list[str] | None = None,
    stdout_path: Path | None = None,
) -> dict[str, Any]:
    now = time.time()
    stdout_path = stdout_path or out / f"{label}.stdout"
    stderr_path = out / f"{label}.stderr"
    receipt_path = out / f"{label}.command.json"
    stdout_path.write_text(stdout)
    stderr_path.write_text("")
    write_json(
        receipt_path,
        {
            "command": command or ["/bin/sh", "-c", "true"],
            "cwd": None,
            "timeout_sec": 10,
            "timed_out": False,
            "launch_error": None,
            "stdin_sha256": None,
            "stdin_size_bytes": 0,
            **timing_receipt(now, now + 10, now, 0),
        },
    )
    return {
        "receipt": evidence_ref(out, receipt_path),
        "stdout": evidence_ref(out, stdout_path),
        "stderr": evidence_ref(out, stderr_path),
    }


def selftest_progress_ref(out: Path, name: str, size: int) -> dict[str, Any]:
    path = out / name
    now = time.time()
    rows = [
        {"bytes": 0, "elapsed_sec": 0.0, "complete": False, "attempt": 1, "timestamp": utc_timestamp(now)},
        {"bytes": size, "elapsed_sec": 0.1, "complete": True, "attempt": 1, "timestamp": utc_timestamp(now + 0.1)},
    ]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return evidence_ref(out, path)


def selftest_serve_evidence(out: Path, prefix: str, *, binary_path: str = "/selftest/ferrum") -> dict[str, Any]:
    now = time.time()
    log = out / f"{prefix}.serve.log"
    log.write_text("selftest server log\n")
    command_path = out / f"{prefix}.serve.command.json"
    write_json(
        command_path,
        {
            "command": [binary_path, "serve", "selftest-model", "--disable-thinking"],
            "cwd": None,
            "timeout_sec": 10,
            "timed_out": False,
            "launch_error": None,
            "stdin_sha256": None,
            "stdin_size_bytes": 0,
            "termination_reason": "gate-cleanup",
            **timing_receipt(now, now + 10, now, -15),
        },
    )
    http: dict[str, Any] = {}
    health_path = out / f"{prefix}.serve.health.receipt.json"
    write_json(
        health_path,
        {
            "method": "GET",
            "requested_url": "http://127.0.0.1:18080/health",
            "effective_url": "http://127.0.0.1:18080/health",
            "http_status": 200,
            "attempts": [
                {
                    "requested_url": "http://127.0.0.1:18080/health",
                    "effective_url": "http://127.0.0.1:18080/health",
                    "http_status": 200,
                    "error": None,
                    **timing_receipt(now, now + 10, now, 0),
                }
            ],
            **timing_receipt(now, now + 10, now, 0),
        },
    )
    http["health"] = evidence_ref(out, health_path)
    for name, status in {
        "math": 200,
        "multiturn": 200,
        "boundary": 400,
        "strict_json": 200,
        "tool_call": 200,
        "stream": 200,
    }.items():
        body = out / f"{prefix}.serve.{name}.response"
        receipt_path = out / f"{prefix}.serve.{name}.receipt.json"
        body.write_text(f"selftest {name}\n")
        write_json(
            receipt_path,
            {
                "method": "POST",
                "requested_url": "http://127.0.0.1:18080/v1/chat/completions",
                "effective_url": "http://127.0.0.1:18080/v1/chat/completions",
                "http_status": status,
                "request_sha256": "1" * 64,
                "response_size_bytes": body.stat().st_size,
                **timing_receipt(now, now + 10, now, 0),
            },
        )
        http[name] = {
            "receipt": evidence_ref(out, receipt_path),
            "response": evidence_ref(out, body),
        }
    return {
        "command": {
            "receipt": evidence_ref(out, command_path),
            "combined_log": evidence_ref(out, log),
        },
        "http": http,
    }


def selftest_checks(*, homebrew: bool = False) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "version": True,
        "cli": {
            "passed": True,
            "has_context": True,
            "has_math": True,
            "disable_thinking": True,
        },
        "serve": {
            "math": [200, "579"],
            "multiturn": [200, "ferrum-blue"],
            "boundary_status": 400,
            "disable_thinking": True,
            "strict_json": [200, '{"answer":579}'],
            "tool_call": [200, "tool_calls"],
            "stream": [200, 1],
        },
    }
    if homebrew:
        checks["help"] = True
    return checks


def selftest_formula_document(mode: str, version: str, digest: str) -> dict[str, Any]:
    spec = FORMULAE[mode]
    return {
        "formulae": [
            {
                "name": spec["name"],
                "full_name": spec["formula"],
                "tap": FORMULA_TAP,
                "versions": {"stable": version},
                "urls": {
                    "stable": {
                        "url": official_asset(version, spec["asset"]),
                        "checksum": digest,
                    }
                },
            }
        ]
    }


def expect_selftest_failure(callback, expected: str) -> None:
    try:
        callback()
    except (GateError, RuntimeError, TimeoutError) as error:
        if expected not in str(error):
            raise AssertionError(f"unexpected self-test failure: {error}") from error
    else:
        raise AssertionError(f"negative self-test unexpectedly passed: {expected}")


def positive_timeout_seconds(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be an integer number of seconds") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("timeout must be greater than zero")
    return parsed


def self_test() -> int:
    global sha256_before_deadline

    version = "0.8.4"
    if DEFAULT_ASSET_DOWNLOAD_TIMEOUT_SECONDS < 7200 or DEFAULT_GATE_TIMEOUT_SECONDS < 14400:
        raise AssertionError("release binary default timeouts are below the user-download floor")
    with tempfile.TemporaryDirectory(prefix="ferrum-release-binary-selftest-") as temporary:
        root = Path(temporary)

        # A deliberately slow, fully in-memory response must pass under one shared
        # total deadline; a response that crosses that deadline must clean partials.
        class SlowResponse:
            def __init__(self, payload: bytes, delay_seconds: float, url: str):
                self._payload = payload
                self._offset = 0
                self._delay_seconds = delay_seconds
                self._url = url
                self.status = 200
                self.headers = {"Content-Length": str(len(payload)), "Content-Type": "application/octet-stream"}

            def __enter__(self):
                return self

            def __exit__(self, _kind, _value, _traceback):
                return False

            def geturl(self) -> str:
                return self._url

            def read(self, size: int) -> bytes:
                time.sleep(self._delay_seconds)
                if self._offset >= len(self._payload):
                    return b""
                end = min(len(self._payload), self._offset + max(1, min(size, 3)))
                chunk = self._payload[self._offset:end]
                self._offset = end
                return chunk

        original_urlopen = urllib.request.urlopen
        slow_url = "https://github.com/sizzlecar/ferrum-infer-rs/releases/download/v0.8.4/selftest.bin"
        slow_payload = b"slow-download-payload"
        try:
            urllib.request.urlopen = lambda url, timeout=0: SlowResponse(slow_payload, 0.002, str(url))
            slow_out = root / "slow-download"
            slow_out.mkdir()
            slow_path = slow_out / "selftest.bin"
            slow_progress = slow_out / "progress.jsonl"
            slow_receipt = download_with_receipt(
                slow_url,
                slow_path,
                retries=3,
                total_timeout_seconds=1.0,
                progress_path=slow_progress,
                evidence_root=slow_out,
            )
            if slow_path.read_bytes() != slow_payload or slow_receipt.get("http_status") != 200:
                raise AssertionError("slow download positive self-test bytes/status differ")
            validate_progress_jsonl(
                slow_out,
                slow_receipt["progress"],
                "slow download self-test progress",
                expected_size=len(slow_payload),
            )
            original_progress = slow_progress.read_bytes()
            progress_rows = [json.loads(line) for line in slow_progress.read_text().splitlines()]
            progress_rows.insert(
                -1,
                {
                    "bytes": len(slow_payload) + 1,
                    "elapsed_sec": progress_rows[-1]["elapsed_sec"],
                    "complete": False,
                    "attempt": progress_rows[-1]["attempt"],
                    "timestamp": progress_rows[-1]["timestamp"],
                },
            )
            slow_progress.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in progress_rows))
            expect_selftest_failure(
                lambda: validate_progress_jsonl(
                    slow_out,
                    evidence_ref(slow_out, slow_progress),
                    "non-monotonic progress self-test",
                    expected_size=len(slow_payload),
                ),
                "bytes are not monotonic",
            )
            slow_progress.write_bytes(original_progress)

            urllib.request.urlopen = lambda url, timeout=0: SlowResponse(b"will-time-out", 0.05, str(url))
            timeout_out = root / "timeout-download"
            timeout_out.mkdir()
            timeout_path = timeout_out / "timeout.bin"
            try:
                download_with_receipt(
                    slow_url,
                    timeout_path,
                    retries=3,
                    total_timeout_seconds=0.01,
                    progress_path=timeout_out / "progress.jsonl",
                    evidence_root=timeout_out,
                )
            except RuntimeError as error:
                if "download failed" not in str(error):
                    raise AssertionError(f"unexpected timeout failure: {error}") from error
            else:
                raise AssertionError("download hard-deadline negative self-test unexpectedly passed")
            if timeout_path.exists() or timeout_path.with_name(f".{timeout_path.name}.partial").exists():
                raise AssertionError("timed-out download left a final or partial asset")
        finally:
            urllib.request.urlopen = original_urlopen

        # Local-copy output paths must be safe before any byte is opened for
        # writing, and an already-expired retry must still remove stale output.
        copy_source = root / "local-copy-source.bin"
        copy_source.write_bytes(b"bounded-local-copy")
        symlink_out = root / "local-copy-symlink"
        symlink_out.mkdir()
        outside = root / "outside-progress-sentinel"
        outside.write_text("do-not-overwrite\n")
        symlink_progress = symlink_out / "progress.jsonl"
        symlink_progress.symlink_to(outside)
        expect_selftest_failure(
            lambda: copy_local_asset_with_receipt(
                copy_source,
                symlink_out / "asset.bin",
                progress_path=symlink_progress,
                evidence_root=symlink_out,
                total_timeout_seconds=10,
                hard_deadline_monotonic=None,
            ),
            "must not be a symlink",
        )
        if outside.read_text() != "do-not-overwrite\n":
            raise AssertionError("unsafe progress symlink overwrote an outside file")

        expired_out = root / "local-copy-expired"
        expired_out.mkdir()
        expired_destination = expired_out / "asset.bin"
        expired_partial = expired_out / ".asset.bin.partial"
        expired_progress = expired_out / "progress.jsonl"
        expired_destination.write_bytes(b"stale-final")
        expired_partial.write_bytes(b"stale-partial")
        expired_progress.write_text("stale-progress\n")
        expect_selftest_failure(
            lambda: copy_local_asset_with_receipt(
                copy_source,
                expired_destination,
                progress_path=expired_progress,
                evidence_root=expired_out,
                total_timeout_seconds=10,
                hard_deadline_monotonic=time.monotonic() - 1,
            ),
            "gate hard deadline expired",
        )
        if expired_destination.exists() or expired_partial.exists() or expired_progress.exists():
            raise AssertionError("expired local copy left stale final/partial/progress output")

        hash_timeout_out = root / "local-copy-hash-timeout"
        hash_timeout_out.mkdir()
        original_bounded_hash = sha256_before_deadline

        def delayed_bounded_hash(path: Path, *, deadline_monotonic: float, label: str) -> str:
            time.sleep(0.03)
            return original_bounded_hash(
                path,
                deadline_monotonic=deadline_monotonic,
                label=label,
            )

        try:
            sha256_before_deadline = delayed_bounded_hash
            expect_selftest_failure(
                lambda: copy_local_asset_with_receipt(
                    copy_source,
                    hash_timeout_out / "asset.bin",
                    progress_path=hash_timeout_out / "progress.jsonl",
                    evidence_root=hash_timeout_out,
                    total_timeout_seconds=0.01,
                    hard_deadline_monotonic=None,
                ),
                "hard deadline exceeded",
            )
            if (hash_timeout_out / "asset.bin").exists() or (
                hash_timeout_out / ".asset.bin.partial"
            ).exists():
                raise AssertionError("timed-out local SHA256 left final/partial output")
        finally:
            sha256_before_deadline = original_bounded_hash

        # Local/prepublication tarball: no network, explicit no-HTTP receipt.
        payload = root / "payload"
        payload.mkdir()
        ferrum = payload / "ferrum"
        ferrum.write_text("#!/bin/sh\necho ferrum 0.8.4\n")
        ferrum.chmod(0o755)
        asset_name = "ferrum-macos-aarch64.tar.gz"
        staged = root / "staged" / asset_name
        staged.parent.mkdir()
        with tarfile.open(staged, "w:gz") as archive:
            archive.add(ferrum, arcname="ferrum")
        staged_digest = sha256(staged)
        staged.with_name(f"{asset_name}.sha256").write_text(f"{staged_digest}  {asset_name}\n")
        tar_out = root / "tar-local"
        asset_evidence: dict[str, Any] = {}
        unpacked_binary = prepare_tarball(version, asset_name, tar_out, None, staged, asset_evidence)
        commands = {
            "version": selftest_command_bundle(
                tar_out,
                "version",
                stdout="ferrum 0.8.4\n",
                command=[str(unpacked_binary), "--version"],
            ),
            "cli": selftest_command_bundle(
                tar_out,
                "cli",
                command=[str(unpacked_binary), "run", "selftest-model", "--disable-thinking"],
            ),
            "serve": selftest_serve_evidence(tar_out, "tar", binary_path=str(unpacked_binary)),
        }
        started = time.time()
        local_gate = write_gate(
            tar_out,
            "metal-tarball",
            version,
            selftest_checks(),
            evidence={"asset": asset_evidence, "commands": commands},
            started=started,
            deadline=started + 60,
        )
        validate_gate_data(local_gate, root=tar_out)

        tampered_checksum_evidence = json.loads(json.dumps(asset_evidence))
        original_checksum_receipt = resolve_evidence_ref(
            tar_out,
            asset_evidence["checksum"]["receipt"],
            "self-test local checksum receipt",
        )
        tampered_checksum_receipt = read_json(
            original_checksum_receipt, "self-test local checksum receipt"
        )
        tampered_checksum_receipt.update(
            {
                "source": "argument",
                "sha256": "0" * 64,
                "requested_url": "https://example.invalid/checksum",
                "effective_url": "https://example.invalid/checksum",
                "http_status": 200,
                "requested_path": "/tmp/fake.sha256",
            }
        )
        tampered_checksum_path = tar_out / "selftest.tampered.local-checksum.receipt.json"
        write_json(tampered_checksum_path, tampered_checksum_receipt)
        tampered_checksum_evidence["checksum"]["receipt"] = evidence_ref(
            tar_out, tampered_checksum_path
        )
        expect_selftest_failure(
            lambda: validate_asset_evidence(
                tar_out,
                tampered_checksum_evidence,
                version=version,
                mode="metal-tarball",
            ),
            "local checksum provenance differs",
        )

        original_archive = (tar_out / asset_name).read_bytes()
        (tar_out / asset_name).write_bytes(original_archive + b"tamper")
        expect_selftest_failure(lambda: validate_gate_data(local_gate, root=tar_out), "byte binding differs")
        (tar_out / asset_name).write_bytes(original_archive)

        # Simulated canonical public receipts exercise URL/effective URL/status binding without I/O.
        public_evidence = json.loads(json.dumps(asset_evidence))
        canonical = official_asset(version, asset_name)
        now = time.time()
        source_receipt_path = tar_out / "selftest.public.asset.receipt.json"
        public_asset_progress = selftest_progress_ref(
            tar_out,
            "selftest.public.asset.progress.jsonl",
            (tar_out / asset_name).stat().st_size,
        )
        source_attempt = {
            "attempt": 1,
            "requested_url": canonical,
            "effective_url": canonical,
            "effective_url_sha256": None,
            "http_status": 200,
            "response_headers": {},
            "error": None,
            **timing_receipt(now, now + 10, now, 0),
        }
        write_json(
            source_receipt_path,
            {
                "source": "public-url",
                "http_performed": True,
                "requested_url": canonical,
                "requested_path": None,
                "effective_url": canonical,
                "effective_url_sha256": None,
                "http_status": 200,
                "response_headers": {},
                "received_size_bytes": (tar_out / asset_name).stat().st_size,
                "attempts": [source_attempt],
                "progress": public_asset_progress,
                **timing_receipt(now, now + 10, now, 0),
            },
        )
        checksum_url = official_asset(version, f"{asset_name}.sha256")
        checksum_receipt_path = tar_out / "selftest.public.checksum.receipt.json"
        public_checksum_progress = selftest_progress_ref(
            tar_out,
            "selftest.public.checksum.progress.jsonl",
            (tar_out / f"{asset_name}.sha256").stat().st_size,
        )
        checksum_attempt = dict(source_attempt)
        checksum_attempt["requested_url"] = checksum_url
        checksum_attempt["effective_url"] = checksum_url
        write_json(
            checksum_receipt_path,
            {
                "source": "public-url",
                "http_performed": True,
                "requested_url": checksum_url,
                "requested_path": None,
                "effective_url": checksum_url,
                "effective_url_sha256": None,
                "http_status": 200,
                "response_headers": {},
                "received_size_bytes": (tar_out / f"{asset_name}.sha256").stat().st_size,
                "attempts": [checksum_attempt],
                "progress": public_checksum_progress,
                **timing_receipt(now, now + 10, now, 0),
            },
        )
        public_evidence.update(
            {
                "source": "public-url",
                "classification": "canonical-public-release",
                "requested_url": canonical,
                "requested_path": None,
                "source_receipt": evidence_ref(tar_out, source_receipt_path),
            }
        )
        public_evidence["checksum"]["source"] = "public-url"
        public_evidence["checksum"]["receipt"] = evidence_ref(tar_out, checksum_receipt_path)
        validate_asset_evidence(tar_out, public_evidence, version=version, mode="metal-tarball")
        wrong_url = json.loads(json.dumps(public_evidence))
        wrong_url["requested_url"] = "https://example.invalid/ferrum.tar.gz"
        expect_selftest_failure(
            lambda: validate_asset_evidence(tar_out, wrong_url, version=version, mode="metal-tarball"),
            "canonical GitHub URL",
        )

        # Full synthetic Homebrew Metal receipt, including actual-path metadata and captured bytes.
        metal_out = root / "homebrew-metal"
        metal_out.mkdir()
        captured_binary = metal_out / "installed" / "ferrum"
        captured_binary.parent.mkdir()
        captured_binary.write_bytes(b"selftest-installed-ferrum")
        formula_digest = "a" * 64
        metal_info_path = metal_out / "brew_info.json"
        write_json(metal_info_path, selftest_formula_document("homebrew-metal", version, formula_digest))
        metal_identity = parse_formula_info(read_json(metal_info_path, "metal formula"), mode="homebrew-metal", version=version)
        installed_identity_path = metal_out / "installed_binary.json"
        write_json(
            installed_identity_path,
            {
                "command_v_path": "/opt/homebrew/bin/ferrum",
                "resolved_path": "/opt/homebrew/Cellar/ferrum/0.8.4/bin/ferrum",
                "command_v_is_symlink": True,
                "sha256": sha256(captured_binary),
                "size_bytes": captured_binary.stat().st_size,
                "captured_binary": evidence_ref(metal_out, captured_binary),
            },
        )
        metal_command_path = "/opt/homebrew/bin/ferrum"
        metal_formula = FORMULAE["homebrew-metal"]["formula"]
        metal_commands = {
            "reinstall": selftest_command_bundle(
                metal_out,
                "metal.reinstall",
                command=["brew", "reinstall", metal_formula],
            ),
            "brew_info": selftest_command_bundle(
                metal_out,
                "metal.brew_info",
                stdout=json.dumps(selftest_formula_document("homebrew-metal", version, formula_digest)) + "\n",
                stdout_path=metal_info_path,
                command=["brew", "info", "--json=v2", metal_formula],
            ),
            "command_v": selftest_command_bundle(
                metal_out,
                "metal.command_v",
                stdout=metal_command_path + "\n",
                command=["/bin/sh", "-c", "command -v ferrum"],
            ),
            "version": selftest_command_bundle(
                metal_out,
                "metal.version",
                stdout="ferrum 0.8.4\n",
                command=[metal_command_path, "--version"],
            ),
            "help": selftest_command_bundle(
                metal_out,
                "metal.help",
                stdout="Usage: ferrum [COMMAND]\n",
                command=[metal_command_path, "--help"],
            ),
            "cli": selftest_command_bundle(
                metal_out,
                "metal.cli",
                command=[metal_command_path, "run", "selftest-model", "--disable-thinking"],
            ),
        }
        metal_commands["serve"] = selftest_serve_evidence(metal_out, "metal", binary_path=metal_command_path)
        started = time.time()
        metal_gate = write_gate(
            metal_out,
            "homebrew-metal",
            version,
            selftest_checks(homebrew=True),
            evidence={
                "formula": {"brew_info": evidence_ref(metal_out, metal_info_path), "identity": metal_identity},
                "installed_binary": evidence_ref(metal_out, installed_identity_path),
                "commands": metal_commands,
            },
            started=started,
            deadline=started + 60,
        )
        validate_gate_data(metal_gate, root=metal_out)

        # Full synthetic CUDA fetch receipt binds raw formula JSON to captured cache bytes.
        cuda_out = root / "homebrew-cuda"
        cuda_out.mkdir()
        captured_archive = cuda_out / "fetched" / FORMULAE["homebrew-cuda-fetch"]["asset"]
        captured_archive.parent.mkdir()
        captured_archive.write_bytes(b"selftest-cuda-archive")
        cuda_digest = sha256(captured_archive)
        cuda_info_path = cuda_out / "brew_info.json"
        write_json(cuda_info_path, selftest_formula_document("homebrew-cuda-fetch", version, cuda_digest))
        cuda_identity = parse_formula_info(read_json(cuda_info_path, "CUDA formula"), mode="homebrew-cuda-fetch", version=version)
        fetched_identity_path = cuda_out / "fetched_archive.json"
        write_json(
            fetched_identity_path,
            {
                "reported_path": "/selftest/cache/ferrum-cuda.tar.gz",
                "resolved_path": "/selftest/cache/ferrum-cuda.tar.gz",
                "sha256": cuda_digest,
                "size_bytes": captured_archive.stat().st_size,
                "captured_archive": evidence_ref(cuda_out, captured_archive),
            },
        )
        cuda_formula = FORMULAE["homebrew-cuda-fetch"]["formula"]
        cuda_reported_path = "/selftest/cache/ferrum-cuda.tar.gz"
        cuda_commands = {
            "fetch": selftest_command_bundle(
                cuda_out,
                "cuda.fetch",
                command=["brew", "fetch", "--force", cuda_formula],
            ),
            "brew_info": selftest_command_bundle(
                cuda_out,
                "cuda.brew_info",
                stdout=json.dumps(selftest_formula_document("homebrew-cuda-fetch", version, cuda_digest)) + "\n",
                stdout_path=cuda_info_path,
                command=["brew", "info", "--json=v2", cuda_formula],
            ),
            "brew_cache": selftest_command_bundle(
                cuda_out,
                "cuda.brew_cache",
                stdout=cuda_reported_path + "\n",
                command=["brew", "--cache", cuda_formula],
            ),
        }
        started = time.time()
        cuda_gate = write_gate(
            cuda_out,
            "homebrew-cuda-fetch",
            version,
            {"fetch": True, "formula_version": version},
            evidence={
                "formula": {"brew_info": evidence_ref(cuda_out, cuda_info_path), "identity": cuda_identity},
                "fetched_archive": evidence_ref(cuda_out, fetched_identity_path),
                "commands": cuda_commands,
            },
            started=started,
            deadline=started + 60,
        )
        validate_gate_data(cuda_gate, root=cuda_out)
        wrong_formula = json.loads(json.dumps(cuda_gate))
        formula_doc = read_json(cuda_info_path, "CUDA formula")
        formula_doc["formulae"][0]["urls"]["stable"]["url"] = "https://example.invalid/not-canonical.tar.gz"
        write_json(cuda_info_path, formula_doc)
        wrong_formula["evidence"]["formula"]["brew_info"] = evidence_ref(cuda_out, cuda_info_path)
        expect_selftest_failure(lambda: validate_gate_data(wrong_formula, root=cuda_out), "stable_url differs")

        # A referenced symlink or path escape is never accepted as evidence.
        target = root / "target.txt"
        target.write_text("target")
        link = root / "link.txt"
        link.symlink_to(target)
        link_ref = {"path": "link.txt", "sha256": sha256(target), "size_bytes": target.stat().st_size}
        expect_selftest_failure(lambda: resolve_evidence_ref(root, link_ref, "symlink self-test"), "symlink")
        escape_ref = {"path": "../target.txt", "sha256": sha256(target), "size_bytes": target.stat().st_size}
        expect_selftest_failure(lambda: resolve_evidence_ref(root, escape_ref, "escape self-test"), "normalized relative path")

    print("FERRUM RELEASE BINARY GATE SELFTEST PASS")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ["metal-tarball", "cuda-tarball", "homebrew-metal", "homebrew-cuda-fetch"]:
        p = sub.add_parser(mode)
        p.add_argument("--version", required=True)
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--sha256")
        p.add_argument("--asset-path", type=Path)
        p.add_argument("--model")
        p.add_argument("--model-name")
        p.add_argument("--port", type=int, default=18080)
        p.add_argument(
            "--asset-download-timeout-seconds",
            type=positive_timeout_seconds,
            default=DEFAULT_ASSET_DOWNLOAD_TIMEOUT_SECONDS,
            help="single hard deadline shared by all retry attempts for each release-asset download",
        )
        p.add_argument(
            "--gate-timeout-seconds",
            type=positive_timeout_seconds,
            default=DEFAULT_GATE_TIMEOUT_SECONDS,
            help="overall hard deadline covering download, extraction, run, serve, and receipt validation",
        )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.gate_started = time.time()
    args.gate_started_monotonic = time.monotonic()
    args.gate_deadline = args.gate_started + args.gate_timeout_seconds
    args.gate_deadline_monotonic = args.gate_started_monotonic + args.gate_timeout_seconds
    try:
        if args.mode == "metal-tarball":
            gate_tarball(args, asset="ferrum-macos-aarch64.tar.gz", default_model="/Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf", model_name="Qwen3-30B-A3B-Q4_K_M", cuda=False)
        elif args.mode == "cuda-tarball":
            gate_tarball(args, asset="ferrum-linux-x86_64-cuda-sm89.tar.gz", default_model="/workspace/hf-cache/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441", model_name="Qwen3-30B-A3B-GPTQ-Int4", cuda=True)
        elif args.mode == "homebrew-metal":
            homebrew_metal(args)
        elif args.mode == "homebrew-cuda-fetch":
            homebrew_cuda_fetch(args)
        return 0
    except Exception as e:
        failed = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "ferrum_release_binary_gate",
            "status": "fail",
            "mode": args.mode,
            "version": args.version,
            "artifact_dir": str(args.out),
            **timing_receipt(args.gate_started, args.gate_deadline, time.time(), 1),
            "error": str(e),
        }
        write_json(args.out / "gate.json", failed)
        print(f"RELEASE BINARY GATE FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
