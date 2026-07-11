#!/usr/bin/env python3
"""Run one command inside a fail-closed process-group resource envelope.

The runner intentionally records no environment variables.  The child inherits
the caller's environment, but receipts contain only the explicit command, cwd,
resource observations, termination result, and output-log identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


RECEIPT_SCHEMA = "ferrum.bounded-command-receipt.v1"
PS_SAMPLE_TIMEOUT_SECONDS = 1.0
KILL_POLL_SECONDS = 0.01
EXIT_PASS = 0
EXIT_COMMAND_FAILED = 1
EXIT_TIMEOUT = 124
EXIT_REJECTED = 125
EXIT_INTERNAL_ERROR = 126
EXIT_SPAWN_ERROR = 127


class SamplingError(RuntimeError):
    """A process-group observation could not be trusted."""


class RunnerInterrupted(RuntimeError):
    def __init__(self, signum: int) -> None:
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


@dataclass(frozen=True)
class Limits:
    wall_timeout_seconds: float
    max_processes: int
    max_group_threads: int
    max_per_process_threads: int
    sample_interval_seconds: float = 0.05
    max_sampling_errors: int = 3
    term_grace_seconds: float = 1.0

    def validate(self) -> None:
        if self.wall_timeout_seconds <= 0:
            raise ValueError("wall_timeout_seconds must be > 0")
        if self.max_processes < 1:
            raise ValueError("max_processes must be >= 1")
        if self.max_group_threads < 1:
            raise ValueError("max_group_threads must be >= 1")
        if self.max_per_process_threads < 1:
            raise ValueError("max_per_process_threads must be >= 1")
        if self.sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be > 0")
        if self.max_sampling_errors < 1:
            raise ValueError("max_sampling_errors must be >= 1")
        if self.term_grace_seconds < 0:
            raise ValueError("term_grace_seconds must be >= 0")

    def to_json(self) -> dict[str, int | float]:
        return {
            "wall_timeout_seconds": self.wall_timeout_seconds,
            "max_processes": self.max_processes,
            "max_group_threads": self.max_group_threads,
            "max_per_process_threads": self.max_per_process_threads,
            "sample_interval_seconds": self.sample_interval_seconds,
            "max_sampling_errors": self.max_sampling_errors,
            "term_grace_seconds": self.term_grace_seconds,
        }


@dataclass(frozen=True)
class ProcessGroupSnapshot:
    process_threads: dict[int, int]

    @property
    def processes(self) -> int:
        return len(self.process_threads)

    @property
    def group_threads(self) -> int:
        return sum(self.process_threads.values())

    @property
    def max_per_process_threads(self) -> int:
        return max(self.process_threads.values(), default=0)

    @property
    def max_per_process_pid(self) -> int | None:
        if not self.process_threads:
            return None
        return max(self.process_threads, key=lambda pid: self.process_threads[pid])


Sampler = Callable[[int], ProcessGroupSnapshot]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def file_identity(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _sample_darwin_process_group(pgid: int) -> ProcessGroupSnapshot:
    try:
        result = subprocess.run(
            ["/bin/ps", "-M", "-g", str(pgid)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=PS_SAMPLE_TIMEOUT_SECONDS,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SamplingError(f"/bin/ps -M failed: {error}") from error
    # Darwin `ps -M -g` emits exactly one row per thread.  The first row for a
    # task has USER in column zero and PID at tokens[1]; continuation rows leave
    # USER blank and put PID at tokens[0].  Count every data row as one thread,
    # while deduplicating PID for the process limit.
    process_threads: dict[int, int] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "USER":
            continue
        pid_index = 0 if line[0].isspace() else 1
        if len(fields) <= pid_index:
            raise SamplingError(f"malformed /bin/ps -M row: {line[:128]!r}")
        try:
            pid = int(fields[pid_index])
        except ValueError as error:
            raise SamplingError(f"non-integer /bin/ps -M row: {line[:128]!r}") from error
        process_threads[pid] = process_threads.get(pid, 0) + 1
    if result.returncode != 0 and process_threads:
        detail = result.stderr.strip().replace("\n", " ")[:256]
        raise SamplingError(
            f"/bin/ps -M returned {result.returncode}: {detail or 'no stderr'}"
        )
    return ProcessGroupSnapshot(process_threads)


def _parse_linux_stat(raw: str) -> tuple[int, int, int]:
    left = raw.find("(")
    right = raw.rfind(")")
    if left <= 0 or right <= left or right + 2 > len(raw):
        raise SamplingError("malformed /proc PID stat record")
    try:
        pid = int(raw[:left].strip())
        fields = raw[right + 2 :].split()
        # fields starts at proc(5) field 3: state. pgrp is field 5 and
        # num_threads is field 20.
        pgid = int(fields[2])
        threads = int(fields[17])
    except (IndexError, ValueError) as error:
        raise SamplingError("malformed /proc PID stat fields") from error
    if threads < 1:
        raise SamplingError(f"invalid /proc thread count for pid {pid}: {threads}")
    return pid, pgid, threads


def _sample_linux_process_group(pgid: int) -> ProcessGroupSnapshot:
    process_threads: dict[int, int] = {}
    try:
        entries = os.scandir("/proc")
    except OSError as error:
        raise SamplingError(f"cannot scan /proc: {error}") from error
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                raw = Path(entry.path, "stat").read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            except OSError as error:
                raise SamplingError(f"cannot read {entry.path}/stat: {error}") from error
            pid, row_pgid, threads = _parse_linux_stat(raw)
            if row_pgid == pgid:
                process_threads[pid] = threads
    return ProcessGroupSnapshot(process_threads)


def sample_process_group(pgid: int) -> ProcessGroupSnapshot:
    system = platform.system()
    if system == "Darwin":
        return _sample_darwin_process_group(pgid)
    if system == "Linux":
        return _sample_linux_process_group(pgid)
    raise SamplingError(f"unsupported operating system: {system}")


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _record_signal(
    pgid: int, signum: signal.Signals, termination: dict[str, Any]
) -> bool:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        return False
    except OSError as error:
        termination["errors"].append(
            {"at": utc_now(), "operation": signum.name, "error": str(error)[:512]}
        )
        return False
    termination["signals"].append({"signal": signum.name, "at": utc_now()})
    return True


def _terminate_group(
    process: subprocess.Popen[bytes],
    pgid: int,
    grace_seconds: float,
    termination: dict[str, Any],
) -> None:
    _record_signal(pgid, signal.SIGTERM, termination)
    grace_deadline = time.monotonic() + grace_seconds
    while _group_exists(pgid) and time.monotonic() < grace_deadline:
        process.poll()
        time.sleep(KILL_POLL_SECONDS)
    if _group_exists(pgid):
        _record_signal(pgid, signal.SIGKILL, termination)
    # The direct child must always be reaped.  Once SIGKILL is delivered this
    # wait should be immediate; keep it bounded so the safety runner itself
    # cannot hang forever in an unexpected kernel/process state.
    try:
        process.wait(timeout=max(1.0, grace_seconds + 1.0))
    except subprocess.TimeoutExpired:
        _record_signal(pgid, signal.SIGKILL, termination)
        process.kill()
        process.wait(timeout=1.0)


def _wait_for_group_exit(pgid: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(KILL_POLL_SECONDS)
    return not _group_exists(pgid)


def _update_peaks(peaks: dict[str, int | None], sample: ProcessGroupSnapshot) -> None:
    peaks["processes"] = max(int(peaks["processes"] or 0), sample.processes)
    peaks["group_threads"] = max(
        int(peaks["group_threads"] or 0), sample.group_threads
    )
    if sample.max_per_process_threads > int(peaks["per_process_threads"] or 0):
        peaks["per_process_threads"] = sample.max_per_process_threads
        peaks["per_process_threads_pid"] = sample.max_per_process_pid


def _resource_violation(
    sample: ProcessGroupSnapshot, limits: Limits
) -> tuple[str, dict[str, int]] | None:
    if sample.processes > limits.max_processes:
        return (
            "max_processes_exceeded",
            {"observed": sample.processes, "limit": limits.max_processes},
        )
    if sample.group_threads > limits.max_group_threads:
        return (
            "max_group_threads_exceeded",
            {"observed": sample.group_threads, "limit": limits.max_group_threads},
        )
    if sample.max_per_process_threads > limits.max_per_process_threads:
        return (
            "max_per_process_threads_exceeded",
            {
                "observed": sample.max_per_process_threads,
                "limit": limits.max_per_process_threads,
                "pid": sample.max_per_process_pid or -1,
            },
        )
    return None


def _wrapper_exit_code(status: str) -> int:
    return {
        "pass": EXIT_PASS,
        "failed": EXIT_COMMAND_FAILED,
        "timeout": EXIT_TIMEOUT,
        "rejected": EXIT_REJECTED,
        "interrupted": EXIT_REJECTED,
        "internal_error": EXIT_INTERNAL_ERROR,
        "spawn_error": EXIT_SPAWN_ERROR,
    }[status]


def run_bounded_command(
    *,
    command: Sequence[str],
    cwd: Path,
    receipt_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    limits: Limits,
    sampler: Sampler = sample_process_group,
) -> tuple[int, dict[str, Any]]:
    limits.validate()
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError("command must contain non-empty argv strings")

    cwd = cwd.resolve()
    receipt_path = receipt_path.resolve()
    stdout_path = stdout_path.resolve()
    stderr_path = stderr_path.resolve()
    if len({receipt_path, stdout_path, stderr_path}) != 3:
        raise ValueError("receipt, stdout log, and stderr log paths must be distinct")

    for path in (receipt_path, stdout_path, stderr_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    started_monotonic = time.monotonic()
    peaks: dict[str, int | None] = {
        "processes": 0,
        "group_threads": 0,
        "per_process_threads": 0,
        "per_process_threads_pid": None,
    }
    sampling_errors: list[dict[str, str]] = []
    termination: dict[str, Any] = {"signals": [], "errors": []}
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "command": list(command),
        "cwd": str(cwd),
        "pid": None,
        "pgid": None,
        "limits": limits.to_json(),
        "peaks": peaks,
        "started_at": started_at,
        "ended_at": None,
        "duration_seconds": None,
        "reason": None,
        "rc": None,
        "status": "running",
        "successful_samples": 0,
        "sampling_error_count": 0,
        "sampling_errors": sampling_errors,
        "violation": None,
        "termination": termination,
        "cleanup": {"process_group_gone": None},
        "stdout": {"path": str(stdout_path), "sha256": None, "size_bytes": None},
        "stderr": {"path": str(stderr_path), "sha256": None, "size_bytes": None},
    }

    process: subprocess.Popen[bytes] | None = None
    pgid: int | None = None
    status = "internal_error"
    reason = "runner_internal_error"
    violation: dict[str, int] | None = None
    prior_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        raise RunnerInterrupted(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            prior_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)

        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            try:
                process = subprocess.Popen(
                    list(command),
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
            except OSError as error:
                status = "spawn_error"
                reason = "spawn_failed"
                termination["errors"].append(
                    {"at": utc_now(), "operation": "spawn", "error": str(error)[:512]}
                )
            else:
                pgid = process.pid
                receipt["pid"] = process.pid
                receipt["pgid"] = pgid
                deadline = started_monotonic + limits.wall_timeout_seconds

                while True:
                    now = time.monotonic()
                    if now >= deadline:
                        status = "timeout"
                        reason = "wall_timeout_exceeded"
                        break

                    rc_before_sample = process.poll()
                    try:
                        sample = sampler(pgid)
                        receipt["successful_samples"] += 1
                        _update_peaks(peaks, sample)
                    except RunnerInterrupted:
                        raise
                    except Exception as error:
                        sampling_errors.append(
                            {
                                "at": utc_now(),
                                "type": type(error).__name__,
                                "error": str(error)[:512],
                            }
                        )
                        receipt["sampling_error_count"] = len(sampling_errors)
                        if len(sampling_errors) >= limits.max_sampling_errors:
                            status = "rejected"
                            reason = "sampling_error_limit_exceeded"
                            break
                        sample = None

                    rc = process.poll()
                    # If the root exited while the sampler was running, sample
                    # again after poll() has reaped it.  Otherwise the prior row
                    # for the just-reaped root could look like a leaked group.
                    if rc_before_sample is None and rc is not None:
                        continue
                    if sample is not None:
                        if rc is None and process.pid not in sample.process_threads:
                            sampling_errors.append(
                                {
                                    "at": utc_now(),
                                    "type": "MissingRootProcess",
                                    "error": f"live root pid {process.pid} absent from group sample",
                                }
                            )
                            receipt["sampling_error_count"] = len(sampling_errors)
                            if len(sampling_errors) >= limits.max_sampling_errors:
                                status = "rejected"
                                reason = "sampling_error_limit_exceeded"
                                break
                        resource_violation = _resource_violation(sample, limits)
                        if resource_violation is not None:
                            reason, violation = resource_violation
                            status = "rejected"
                            break

                    if rc is not None:
                        if sample is not None and sample.processes > 0:
                            status = "rejected"
                            reason = "root_exited_with_live_process_group"
                        elif rc == 0:
                            status = "pass"
                            reason = "command_completed"
                        else:
                            status = "failed"
                            reason = "command_exit_nonzero"
                        break

                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(limits.sample_interval_seconds, remaining))
    except RunnerInterrupted as error:
        status = "interrupted"
        reason = f"runner_interrupted_{signal.Signals(error.signum).name}"
    except KeyboardInterrupt:
        status = "interrupted"
        reason = "runner_interrupted_SIGINT"
    except Exception as error:
        status = "internal_error"
        reason = "runner_internal_error"
        termination["errors"].append(
            {
                "at": utc_now(),
                "operation": "runner",
                "error": f"{type(error).__name__}: {str(error)[:512]}",
            }
        )
    finally:
        # A second SIGINT/SIGTERM must not interrupt group termination, direct
        # child reaping, or the atomic receipt write.
        for signum in prior_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if process is not None:
                try:
                    if pgid is not None and _group_exists(pgid):
                        _terminate_group(
                            process, pgid, limits.term_grace_seconds, termination
                        )
                    else:
                        process.wait(timeout=1.0)
                except Exception as error:
                    termination["errors"].append(
                        {
                            "at": utc_now(),
                            "operation": "cleanup",
                            "error": f"{type(error).__name__}: {str(error)[:512]}",
                        }
                    )
                    status = "internal_error"
                    reason = "process_cleanup_failed"
                    if pgid is not None:
                        _record_signal(pgid, signal.SIGKILL, termination)
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired as fallback_error:
                        termination["errors"].append(
                            {
                                "at": utc_now(),
                                "operation": "cleanup_fallback",
                                "error": str(fallback_error)[:512],
                            }
                        )
                receipt["rc"] = process.poll()
                if pgid is not None:
                    process_group_gone = _wait_for_group_exit(
                        pgid, max(1.0, limits.term_grace_seconds)
                    )
                    receipt["cleanup"]["process_group_gone"] = process_group_gone
                    if not process_group_gone:
                        termination["errors"].append(
                            {
                                "at": utc_now(),
                                "operation": "cleanup",
                                "error": f"process group {pgid} remains after termination",
                            }
                        )
                        status = "internal_error"
                        reason = "process_group_cleanup_failed"

            receipt["status"] = status
            receipt["reason"] = reason
            receipt["violation"] = violation
            receipt["sampling_error_count"] = len(sampling_errors)
            receipt["ended_at"] = utc_now()
            receipt["duration_seconds"] = round(time.monotonic() - started_monotonic, 6)
            receipt["stdout"] = file_identity(stdout_path)
            receipt["stderr"] = file_identity(stderr_path)
            atomic_write_json(receipt_path, receipt)
        finally:
            for signum, previous in prior_handlers.items():
                signal.signal(signum, previous)

    return _wrapper_exit_code(status), receipt


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _assert_pid_gone(pid: int, label: str) -> None:
    deadline = time.monotonic() + 2.0
    while _pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _pid_exists(pid), f"{label}: leftover child pid {pid}"


def _self_test_case(
    root: Path,
    name: str,
    code: str,
    limits: Limits,
    sampler: Sampler = sample_process_group,
) -> tuple[int, dict[str, Any], Path]:
    case = root / name
    child_pid_path = case / "child.pid"
    exit_code, receipt = run_bounded_command(
        command=[sys.executable, "-c", code, str(child_pid_path)],
        cwd=root,
        receipt_path=case / "receipt.json",
        stdout_path=case / "stdout.log",
        stderr_path=case / "stderr.log",
        limits=limits,
        sampler=sampler,
    )
    loaded = json.loads((case / "receipt.json").read_text(encoding="utf-8"))
    assert loaded == receipt, f"{name}: receipt did not round trip"
    assert loaded["schema"] == RECEIPT_SCHEMA
    assert loaded["command"] == [sys.executable, "-c", code, str(child_pid_path)]
    assert loaded["cwd"] == str(root.resolve())
    assert loaded["pid"] == loaded["pgid"]
    assert loaded["cleanup"] == {"process_group_gone": True}
    assert loaded["duration_seconds"] >= 0
    assert len(loaded["stdout"]["sha256"]) == 64
    assert len(loaded["stderr"]["sha256"]) == 64
    assert not list(case.glob(".receipt.json.*.tmp"))
    return exit_code, loaded, child_pid_path


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="bounded-command-selftest-") as temporary:
        root = Path(temporary).resolve()
        fast = {
            "sample_interval_seconds": 0.01,
            "max_sampling_errors": 3,
            "term_grace_seconds": 0.1,
        }

        code = (
            "import sys,time; print('bounded-pass'); "
            "print('bounded-stderr', file=sys.stderr); time.sleep(0.08)"
        )
        exit_code, receipt, _ = _self_test_case(
            root,
            "pass",
            code,
            Limits(2.0, 2, 8, 8, **fast),
        )
        assert exit_code == EXIT_PASS
        assert receipt["status"] == "pass"
        assert receipt["reason"] == "command_completed"
        assert receipt["rc"] == 0
        assert receipt["stdout"]["size_bytes"] > 0
        assert receipt["stderr"]["size_bytes"] > 0

        code = (
            "import threading,time; "
            "ts=[threading.Thread(target=time.sleep,args=(5,)) for _ in range(3)]; "
            "[t.start() for t in ts]; time.sleep(5)"
        )
        exit_code, receipt, _ = _self_test_case(
            root,
            "per-process-thread-reject",
            code,
            Limits(2.0, 2, 16, 2, **fast),
        )
        assert exit_code == EXIT_REJECTED
        assert receipt["status"] == "rejected"
        assert receipt["reason"] == "max_per_process_threads_exceeded"
        assert receipt["peaks"]["per_process_threads"] > 2
        assert receipt["termination"]["signals"]

        code = (
            "import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c',"
            "'import threading,time; ts=[threading.Thread(target=time.sleep,args=(5,)) "
            "for _ in range(2)]; [t.start() for t in ts]; time.sleep(5)']); "
            "open(sys.argv[1],'w').write(str(p.pid)); time.sleep(5)"
        )
        exit_code, receipt, child_pid_path = _self_test_case(
            root,
            "group-thread-reject",
            code,
            Limits(2.0, 3, 3, 4, **fast),
        )
        assert exit_code == EXIT_REJECTED
        assert receipt["reason"] == "max_group_threads_exceeded"
        _assert_pid_gone(int(child_pid_path.read_text(encoding="utf-8")), "group thread")

        code = (
            "import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)']); "
            "open(sys.argv[1],'w').write(str(p.pid)); time.sleep(5)"
        )
        exit_code, receipt, child_pid_path = _self_test_case(
            root,
            "process-reject",
            code,
            Limits(2.0, 1, 16, 8, **fast),
        )
        assert exit_code == EXIT_REJECTED
        assert receipt["reason"] == "max_processes_exceeded"
        assert receipt["peaks"]["processes"] > 1
        _assert_pid_gone(int(child_pid_path.read_text(encoding="utf-8")), "process")

        exit_code, receipt, child_pid_path = _self_test_case(
            root,
            "timeout",
            code,
            Limits(0.15, 4, 16, 8, **fast),
        )
        assert exit_code == EXIT_TIMEOUT
        assert receipt["status"] == "timeout"
        assert receipt["reason"] == "wall_timeout_exceeded"
        _assert_pid_gone(int(child_pid_path.read_text(encoding="utf-8")), "timeout")

        def fail_sampling(_pgid: int) -> ProcessGroupSnapshot:
            raise SamplingError("synthetic self-test sampling failure")

        exit_code, receipt, _ = _self_test_case(
            root,
            "sampling-reject",
            "import time; time.sleep(5)",
            Limits(2.0, 2, 4, 4, **fast),
            sampler=fail_sampling,
        )
        assert exit_code == EXIT_REJECTED
        assert receipt["status"] == "rejected"
        assert receipt["reason"] == "sampling_error_limit_exceeded"
        assert receipt["sampling_error_count"] == fast["max_sampling_errors"]
        assert len(receipt["sampling_errors"]) == fast["max_sampling_errors"]

    print("BOUNDED COMMAND SELFTEST PASS")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--wall-timeout-seconds", type=float)
    parser.add_argument("--max-processes", type=int)
    parser.add_argument("--max-group-threads", type=int)
    parser.add_argument("--max-per-process-threads", type=int)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.05)
    parser.add_argument("--max-sampling-errors", type=int, default=3)
    parser.add_argument("--term-grace-seconds", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.self_test:
        if len(argv) != 1:
            parser.error("--self-test cannot be combined with other arguments")
        return args
    required = {
        "--receipt": args.receipt,
        "--stdout-log": args.stdout_log,
        "--stderr-log": args.stderr_log,
        "--wall-timeout-seconds": args.wall_timeout_seconds,
        "--max-processes": args.max_processes,
        "--max-group-threads": args.max_group_threads,
        "--max-per-process-threads": args.max_per_process_threads,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        return self_test()
    limits = Limits(
        wall_timeout_seconds=args.wall_timeout_seconds,
        max_processes=args.max_processes,
        max_group_threads=args.max_group_threads,
        max_per_process_threads=args.max_per_process_threads,
        sample_interval_seconds=args.sample_interval_seconds,
        max_sampling_errors=args.max_sampling_errors,
        term_grace_seconds=args.term_grace_seconds,
    )
    try:
        exit_code, receipt = run_bounded_command(
            command=args.command,
            cwd=args.cwd,
            receipt_path=args.receipt,
            stdout_path=args.stdout_log,
            stderr_path=args.stderr_log,
            limits=limits,
        )
    except ValueError as error:
        print(f"bounded_command.py: error: {error}", file=sys.stderr)
        return 2
    label = "PASS" if exit_code == 0 else "REJECT"
    print(f"BOUNDED COMMAND {label}: {receipt['reason']}: {args.receipt.resolve()}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
