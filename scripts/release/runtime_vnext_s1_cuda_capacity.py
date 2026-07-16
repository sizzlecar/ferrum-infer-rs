#!/usr/bin/env python3
"""Collect and validate the bounded S1 CUDA capacity-pressure product slice."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


PASS_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA CAPACITY PRESSURE PASS"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_PRESSURE_NO_PROGRESS_SECONDS = 30.0
MAX_PRESSURE_UNCHANGED_SKIPS = 512
MAX_PRESSURE_TRACE_BYTES = 16 * 1024 * 1024
MAX_PRESSURE_JOINT_STREAM_SECONDS = 300.0
PRESSURE_STOP_POLICY = {
    "no_progress_timeout_seconds": MAX_PRESSURE_NO_PROGRESS_SECONDS,
    "max_unchanged_epoch_skips": MAX_PRESSURE_UNCHANGED_SKIPS,
    "max_pressure_trace_bytes": MAX_PRESSURE_TRACE_BYTES,
    "joint_stream_timeout_seconds": MAX_PRESSURE_JOINT_STREAM_SECONDS,
}
FORBIDDEN_PATTERNS = (
    "panic",
    "panicked",
    "segmentation fault",
    "cuda out of memory",
    "invalid utf-8",
    "<unk>",
    "[pad",
    "missing data: [done]",
)


class CapacityGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CapacityGateError(message)


def bounded_remaining_timeout(
    *,
    deadline_monotonic: float,
    requested_timeout: float,
    now_monotonic: float,
    label: str,
) -> float:
    require(requested_timeout > 0, f"{label} timeout must be positive")
    remaining = deadline_monotonic - now_monotonic
    require(remaining > 0, f"{label} exceeded the joint pressure lifecycle timeout")
    return min(requested_timeout, remaining)


def max_stream_silence_seconds(
    result: dict[str, Any],
    content_wall_ns: list[int],
    *,
    monitored_from_wall_ns: int,
) -> float:
    started_wall_ns = result.get("started_wall_ns")
    finished_wall_ns = result.get("finished_wall_ns")
    require(
        isinstance(started_wall_ns, int) and started_wall_ns > 0,
        "stream result has no start timestamp",
    )
    require(
        isinstance(finished_wall_ns, int) and finished_wall_ns >= started_wall_ns,
        "stream result has no valid finish timestamp",
    )
    require(monitored_from_wall_ns > 0, "stream monitor has no start timestamp")
    interval_start = max(started_wall_ns, monitored_from_wall_ns)
    if finished_wall_ns <= interval_start:
        return 0.0
    progress_points = sorted(
        wall_ns
        for wall_ns in content_wall_ns
        if interval_start <= wall_ns <= finished_wall_ns
    )
    boundaries = [interval_start, *progress_points, finished_wall_ns]
    return max(
        (right - left) / 1_000_000_000
        for left, right in zip(boundaries, boundaries[1:])
    )


def read_stream_content_times(path: Path) -> list[int]:
    require(path.is_file(), f"missing stream event file: {path}")
    content_times: list[int] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CapacityGateError(f"invalid stream event JSON {path}: {error}") from error
        if isinstance(row, dict) and row.get("kind") == "content":
            wall_ns = row.get("wall_ns")
            require(
                isinstance(wall_ns, int) and wall_ns > 0,
                f"stream content event has no timestamp: {path}",
            )
            content_times.append(wall_ns)
    return content_times


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CapacityGateError(f"invalid JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(argv: list[str], cwd: Path) -> str:
    result = subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(result.returncode == 0, f"command failed ({argv!r}): {result.stdout}")
    return result.stdout.strip()


def http_json(port: int, path: str, timeout: float = 5.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{path}", timeout=timeout
        ) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, urllib.error.URLError) as error:
        raise CapacityGateError(f"GET {path} failed: {error}") from error
    require(isinstance(value, dict), f"GET {path} did not return an object")
    return value


def find_executor_snapshot(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if value.get("schema") == "ferrum.runtime-vnext.executor-trace.v1":
            return value
        for child in value.values():
            found = find_executor_snapshot(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_executor_snapshot(child)
            if found is not None:
                return found
    return None


def quiescent_pool_snapshot(executor: dict[str, Any], label: str) -> dict[str, Any]:
    static_bytes = executor.get("static_bytes")
    dynamic_pools = executor.get("dynamic_pools")
    require(isinstance(static_bytes, int) and static_bytes > 0, f"{label}: invalid static bytes")
    require(isinstance(dynamic_pools, dict), f"{label}: dynamic pools are missing")
    pools = dynamic_pools.get("pools")
    budget_claimed_bytes = dynamic_pools.get("budget_claimed_bytes")
    require(isinstance(pools, list) and pools, f"{label}: dynamic pool list is empty")
    require(
        isinstance(budget_claimed_bytes, int) and budget_claimed_bytes >= static_bytes,
        f"{label}: invalid budget claimed bytes",
    )
    residency: dict[str, int] = {}
    envelopes: dict[str, dict[str, Any]] = {}
    for pool in pools:
        require(isinstance(pool, dict), f"{label}: invalid dynamic pool status")
        pool_id = pool.get("pool_id")
        resident_bytes = pool.get("resident_bytes")
        resident_chunks = pool.get("resident_chunks")
        largest_contiguous_bytes = pool.get("largest_contiguous_bytes")
        storage_profile = pool.get("storage_profile")
        require(isinstance(pool_id, str) and pool_id, f"{label}: pool id is missing")
        require(pool_id not in residency, f"{label}: duplicate pool id {pool_id}")
        require(
            isinstance(resident_bytes, int) and resident_bytes >= 0,
            f"{label}: invalid resident bytes for {pool_id}",
        )
        require(
            isinstance(resident_chunks, int)
            and resident_chunks >= 0
            and (resident_bytes == 0) == (resident_chunks == 0),
            f"{label}: invalid resident chunk count for {pool_id}",
        )
        require(
            isinstance(largest_contiguous_bytes, int)
            and 0 <= largest_contiguous_bytes <= resident_bytes,
            f"{label}: invalid contiguous capacity for {pool_id}",
        )
        require(isinstance(storage_profile, dict), f"{label}: storage profile is missing for {pool_id}")
        require(pool.get("pending_growth_bytes") == 0, f"{label}: pending growth remains in {pool_id}")
        require(pool.get("live_segments") == 0, f"{label}: live segments remain in {pool_id}")
        require(
            pool.get("free_bytes") == resident_bytes,
            f"{label}: quiescent pool {pool_id} is not fully free",
        )
        require(pool.get("quarantined_bytes") == 0, f"{label}: quarantined bytes remain in {pool_id}")
        require(pool.get("quarantined_chunks") == 0, f"{label}: quarantined chunks remain in {pool_id}")
        require(
            pool.get("descriptor_mismatch_chunks") == 0,
            f"{label}: descriptor mismatch remains in {pool_id}",
        )
        require(
            pool.get("publication_rejected_chunks") == 0,
            f"{label}: rejected publication remains in {pool_id}",
        )
        require(pool.get("poisoned") is False, f"{label}: pool {pool_id} is poisoned")
        residency[pool_id] = resident_bytes
        envelopes[pool_id] = {
            "resident_bytes": resident_bytes,
            "resident_chunks": resident_chunks,
            "largest_contiguous_bytes": largest_contiguous_bytes,
            "storage_profile": storage_profile,
        }
    resident_bytes = sum(residency.values())
    require(resident_bytes > 0, f"{label}: no dynamic backing was installed")
    require(
        budget_claimed_bytes == static_bytes + resident_bytes,
        f"{label}: budget claim is not static plus installed backing",
    )
    return {
        "static_bytes": static_bytes,
        "resident_bytes": resident_bytes,
        "budget_claimed_bytes": budget_claimed_bytes,
        "pool_resident_bytes": dict(sorted(residency.items())),
        "pool_envelopes": dict(sorted(envelopes.items())),
    }


def require_replayed_pool_snapshot(
    calibration: dict[str, Any], replay: dict[str, Any], exact_budget: int
) -> None:
    require(
        replay.get("static_bytes") == calibration.get("static_bytes"),
        "replay static bytes differ from calibration",
    )
    require(
        replay.get("resident_bytes") == calibration.get("resident_bytes"),
        "replay aggregate resident bytes differ from calibration",
    )
    require(
        replay.get("pool_resident_bytes") == calibration.get("pool_resident_bytes"),
        "replay per-pool residency differs from calibration",
    )
    require(
        replay.get("pool_envelopes") == calibration.get("pool_envelopes"),
        "replay per-pool allocation envelope differs from calibration",
    )
    require(
        replay.get("budget_claimed_bytes") == exact_budget,
        "replay budget claim differs from the calibrated exact budget",
    )


def read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def trace_bytes_at_or_after(path: Path, started_wall_ns: int) -> int:
    require(path.is_file(), f"missing trace file: {path}")
    require(started_wall_ns > 0, "pressure trace boundary has no wall timestamp")
    total_bytes = path.stat().st_size
    offset = 0
    with path.open("rb") as handle:
        for raw in handle:
            try:
                row = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                offset += len(raw)
                continue
            if (
                isinstance(row, dict)
                and isinstance(row.get("ts_unix_nanos"), int)
                and row["ts_unix_nanos"] >= started_wall_ns
            ):
                return total_bytes - offset
            offset += len(raw)
    return 0


class PressureStopGuard:
    def __init__(
        self,
        *,
        initial_progress: dict[str, int],
        started_monotonic: float,
        no_progress_timeout: float,
        max_unchanged_skips: int,
        max_trace_bytes: int,
    ) -> None:
        self.last_progress = dict(initial_progress)
        self.last_progress_monotonic_by_role = {
            role: started_monotonic for role in initial_progress
        }
        self.max_stall_seconds_by_role = {role: 0.0 for role in initial_progress}
        self.no_progress_timeout = no_progress_timeout
        self.max_unchanged_skips = max_unchanged_skips
        self.max_trace_bytes = max_trace_bytes

    def observe(
        self,
        *,
        progress: dict[str, int],
        unchanged_skips: int,
        trace_bytes: int,
        now_monotonic: float,
        active_roles: set[str],
    ) -> None:
        require(
            unchanged_skips <= self.max_unchanged_skips,
            "unchanged-epoch skip limit exceeded: "
            f"{unchanged_skips} > {self.max_unchanged_skips}",
        )
        require(
            trace_bytes <= self.max_trace_bytes,
            f"pressure trace byte limit exceeded: {trace_bytes} > {self.max_trace_bytes}",
        )
        require(set(progress) == set(self.last_progress), "stream progress roles changed")
        require(active_roles.issubset(progress), "active stream role has no progress counter")
        for role, content_chunks in progress.items():
            require(
                content_chunks >= self.last_progress[role],
                f"stream token progress moved backwards for role {role}",
            )
            if content_chunks > self.last_progress[role]:
                self.last_progress[role] = content_chunks
                self.last_progress_monotonic_by_role[role] = now_monotonic
            if role in active_roles:
                stalled_seconds = (
                    now_monotonic - self.last_progress_monotonic_by_role[role]
                )
                self.max_stall_seconds_by_role[role] = max(
                    self.max_stall_seconds_by_role[role], stalled_seconds
                )
                require(
                    stalled_seconds < self.no_progress_timeout,
                    "stream token progress timeout exceeded for role "
                    f"{role}: {stalled_seconds:.3f}s >= "
                    f"{self.no_progress_timeout:.3f}s",
                )


class IncrementalTracePhaseCounter:
    def __init__(self, path: Path, *, phase: str, request_id: str) -> None:
        self.path = path
        self.phase = phase
        self.request_id = request_id
        self.offset = 0
        self.partial = b""
        self.count = 0

    def poll(self) -> int:
        if not self.path.is_file():
            return self.count
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.partial = b""
            self.count = 0
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            appended = handle.read()
        self.offset += len(appended)
        if not appended:
            return self.count
        rows = (self.partial + appended).split(b"\n")
        self.partial = rows.pop()
        for raw in rows:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if (
                isinstance(row, dict)
                and row.get("phase") == self.phase
                and request_identity_matches(row.get("request_id"), self.request_id)
            ):
                self.count += 1
        return self.count


def request_identity_matches(observed: Any, request_id: str) -> bool:
    return observed == request_id or observed == f"request.product.{request_id}"


def capacity_prompt(workload_slot: str) -> str:
    require(workload_slot in {"A", "B", "C"}, "invalid capacity workload slot")
    return (
        f"Capacity lane slot {workload_slot}. Emit deterministic short words until the token "
        "limit; do not explain the task."
    )


def collect_run_smoke(
    *, repo: Path, binary: Path, model: Path, out_dir: Path, timeout: float
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / "scheduler-trace.jsonl"
    profile = out_dir / "profile.jsonl"
    command = [
        str(binary),
        "run",
        str(model),
        "--backend",
        "cuda",
        "--prompt",
        "Respond with only the city name: What is the capital of France?",
        "--max-tokens",
        "16",
        "--temperature",
        "0",
        "--seed",
        "9271",
        "--output-format",
        "jsonl",
        "--profile-detail",
        "basic",
        "--profile-sample-rate",
        "1.0",
        "--profile-jsonl",
        str(profile),
        "--scheduler-trace-jsonl",
        str(trace),
    ]
    write_json(out_dir / "command.json", {"argv": command})
    started = time.time_ns()
    try:
        result = subprocess.run(
            command,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        summary = {
            "returncode": result.returncode,
            "started_wall_ns": started,
            "finished_wall_ns": time.time_ns(),
            "timed_out": False,
        }
        (out_dir / "stdout.jsonl").write_text(result.stdout)
        (out_dir / "stderr.log").write_text(result.stderr)
    except subprocess.TimeoutExpired as error:
        summary = {
            "returncode": None,
            "started_wall_ns": started,
            "finished_wall_ns": time.time_ns(),
            "timed_out": True,
        }
        (out_dir / "stdout.jsonl").write_text(error.stdout or "")
        (out_dir / "stderr.log").write_text(error.stderr or "")
    write_json(out_dir / "result.json", summary)
    require(summary["returncode"] == 0, f"ferrum run failed: {summary}")
    rows = [
        json.loads(line)
        for line in (out_dir / "stdout.jsonl").read_text().splitlines()
        if line.strip()
    ]
    require(rows and all(isinstance(row, dict) for row in rows), "ferrum run JSONL is empty")
    assistant = next((row for row in rows if row.get("event") == "assistant"), None)
    require(isinstance(assistant, dict), "ferrum run emitted no assistant row")
    require("paris" in str(assistant.get("content", "")).lower(), "ferrum run answer is not Paris")
    require(assistant.get("n_tokens", 0) > 0, "ferrum run emitted no output tokens")
    phases = {row.get("phase") for row in read_trace(trace)}
    require("vnext.operation_submitted" in phases, "ferrum run has no vNext submission trace")
    require("vnext.request_completed" in phases, "ferrum run has no vNext completion trace")
    return summary


def stream_request(
    *,
    port: int,
    model: str,
    role: str,
    workload_slot: str,
    max_tokens: int,
    out_dir: Path,
    first_content: threading.Event,
    timeout: float,
    on_content: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    started_wall_ns = time.time_ns()
    prompt = capacity_prompt(workload_slot)
    result: dict[str, Any] = {
        "role": role,
        "workload_slot": workload_slot,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "max_tokens": max_tokens,
        "started_wall_ns": started_wall_ns,
        "first_content_wall_ns": None,
        "finished_wall_ns": None,
        "http_status": None,
        "stream_id": None,
        "content_chunks": 0,
        "done_count": 0,
        "usage_count": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "error": None,
    }
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 9271,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
            "user": f"runtime-vnext-capacity-{role}",
        }
    ).encode("utf-8")
    raw_path = out_dir / f"{role}.sse"
    events_path = out_dir / f"{role}.events.jsonl"
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        result["http_status"] = response.status
        with raw_path.open("w") as raw_file, events_path.open("w") as events_file:
            while True:
                line_bytes = response.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace")
                raw_file.write(line)
                stripped = line.strip()
                if not stripped.startswith("data:"):
                    continue
                payload = stripped[5:].strip()
                now_ns = time.time_ns()
                if payload == "[DONE]":
                    result["done_count"] += 1
                    events_file.write(
                        json.dumps({"wall_ns": now_ns, "kind": "done"}) + "\n"
                    )
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError as error:
                    raise CapacityGateError(
                        f"{role} returned malformed SSE JSON: {error}"
                    ) from error
                if isinstance(chunk.get("id"), str):
                    result["stream_id"] = chunk["id"]
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    result["usage_count"] += 1
                    result["prompt_tokens"] = usage.get("prompt_tokens")
                    result["completion_tokens"] = usage.get("completion_tokens")
                    result["total_tokens"] = usage.get("total_tokens")
                content = ""
                choices = chunk.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                            content += delta["content"]
                if content:
                    result["content_chunks"] += 1
                    if on_content is not None:
                        on_content(now_ns)
                    if result["first_content_wall_ns"] is None:
                        result["first_content_wall_ns"] = now_ns
                        first_content.set()
                    events_file.write(
                        json.dumps(
                            {
                                "wall_ns": now_ns,
                                "kind": "content",
                                "chars": len(content),
                            }
                        )
                        + "\n"
                    )
        if response.status != 200:
            raise CapacityGateError(f"{role} returned HTTP {response.status}")
    except Exception as error:  # The artifact must retain client-side failures.
        result["error"] = str(error)
    finally:
        result["finished_wall_ns"] = time.time_ns()
        first_content.set()
        connection.close()
        write_json(out_dir / f"{role}.result.json", result)
    return result


class StreamTask:
    def __init__(
        self,
        *,
        port: int,
        model: str,
        role: str,
        workload_slot: str,
        max_tokens: int,
        out_dir: Path,
        timeout: float,
    ) -> None:
        self.first_content = threading.Event()
        self.result: dict[str, Any] | None = None
        self._progress_lock = threading.Lock()
        self._content_chunks = 0
        self.thread = threading.Thread(
            target=self._run,
            kwargs={
                "port": port,
                "model": model,
                "role": role,
                "workload_slot": workload_slot,
                "max_tokens": max_tokens,
                "out_dir": out_dir,
                "first_content": self.first_content,
                "timeout": timeout,
                "on_content": self._record_content,
            },
            daemon=True,
        )

    def _record_content(self, _wall_ns: int) -> None:
        with self._progress_lock:
            self._content_chunks += 1

    def live_content_chunks(self) -> int:
        with self._progress_lock:
            return self._content_chunks

    def _run(self, **kwargs: Any) -> None:
        self.result = stream_request(**kwargs)

    def start(self) -> None:
        self.thread.start()

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def wait_first(self, timeout: float) -> None:
        require(self.first_content.wait(timeout), "stream did not produce a first-content signal")
        if self.result is not None and self.result.get("error"):
            raise CapacityGateError(str(self.result["error"]))

    def join(self, timeout: float) -> dict[str, Any]:
        self.thread.join(timeout)
        require(not self.thread.is_alive(), "stream request exceeded the bounded timeout")
        require(self.result is not None, "stream request produced no result")
        return self.result

    def settle(self, timeout: float) -> bool:
        self.thread.join(max(0.0, timeout))
        return not self.thread.is_alive()


class ServerSession:
    def __init__(
        self,
        *,
        repo: Path,
        binary: Path,
        model: Path,
        port: int,
        out_dir: Path,
        runtime_budget: int | None,
        startup_timeout: float,
    ) -> None:
        self.repo = repo
        self.port = port
        self.out_dir = out_dir
        self.trace_path = out_dir / "scheduler-trace.jsonl"
        self.profile_path = out_dir / "profile.jsonl"
        self.proc: subprocess.Popen[str] | None = None
        self.stdout_file: Any = None
        self.stderr_file: Any = None
        command = [
            str(binary),
            "serve",
            str(model),
            "--backend",
            "cuda",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-model-len",
            "512",
            "--max-num-seqs",
            "4",
            "--max-num-batched-tokens",
            "1024",
            "--scheduler-prefill-first-until-active",
            "4",
            "--profile-detail",
            "basic",
            "--profile-sample-rate",
            "1.0",
            "--profile-jsonl",
            str(self.profile_path),
            "--scheduler-trace-jsonl",
            str(self.trace_path),
        ]
        if runtime_budget is not None:
            command.extend(["--runtime-memory-budget-bytes", str(runtime_budget)])
        self.command = command
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "server.command.json", {"argv": command})
        self.stdout_file = (out_dir / "server.stdout.log").open("w")
        self.stderr_file = (out_dir / "server.stderr.log").open("w")
        self.proc = subprocess.Popen(
            command,
            cwd=repo,
            text=True,
            stdout=self.stdout_file,
            stderr=self.stderr_file,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + startup_timeout
            last_error = "server did not answer"
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise CapacityGateError(
                        f"server exited during startup with {self.proc.returncode}"
                    )
                try:
                    health = http_json(port, "/health")
                    if health.get("status") == "healthy":
                        models = http_json(port, "/v1/models")
                        write_json(out_dir / "health.start.json", health)
                        write_json(out_dir / "models.json", models)
                        data = models.get("data")
                        require(isinstance(data, list) and data, "model list is empty")
                        model_id = data[0].get("id") if isinstance(data[0], dict) else None
                        require(isinstance(model_id, str) and model_id, "model id is missing")
                        self.model_id = model_id
                        return
                except CapacityGateError as error:
                    last_error = str(error)
                time.sleep(1)
            raise CapacityGateError(f"server startup timed out: {last_error}")
        except Exception:
            self.stop()
            raise

    def health(self, name: str) -> dict[str, Any]:
        value = http_json(self.port, "/health")
        write_json(self.out_dir / name, value)
        return value

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if self.proc.poll() is None:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=10)
        write_json(
            self.out_dir / "server.exit.json",
            {"returncode": self.proc.returncode, "stopped_wall_ns": time.time_ns()},
        )
        self.stdout_file.close()
        self.stderr_file.close()
        self.proc = None


def require_stream_success(result: dict[str, Any], label: str) -> None:
    require(result.get("error") is None, f"{label}: {result.get('error')}")
    require(result.get("http_status") == 200, f"{label}: HTTP status is not 200")
    require(result.get("done_count") == 1, f"{label}: expected exactly one [DONE]")
    require(result.get("usage_count") == 1, f"{label}: expected exactly one usage chunk")
    require(
        isinstance(result.get("prompt_tokens"), int) and result["prompt_tokens"] > 0,
        f"{label}: missing positive prompt token count",
    )
    require(
        isinstance(result.get("completion_tokens"), int)
        and result["completion_tokens"] > 0,
        f"{label}: missing positive completion token count",
    )
    require(result.get("content_chunks", 0) > 0, f"{label}: no content chunks")
    require(
        result.get("total_tokens") == result["prompt_tokens"] + result["completion_tokens"],
        f"{label}: usage token counts do not reconcile",
    )


def run_capacity_pair(
    server: ServerSession,
    out_dir: Path,
    timeout: float,
    prefix: str,
) -> dict[str, Any]:
    a = StreamTask(
        port=server.port,
        model=server.model_id,
        role=f"{prefix}-A",
        workload_slot="A",
        max_tokens=128,
        out_dir=out_dir,
        timeout=timeout,
    )
    a.start()
    a.wait_first(timeout)
    c = StreamTask(
        port=server.port,
        model=server.model_id,
        role=f"{prefix}-C",
        workload_slot="C",
        max_tokens=16,
        out_dir=out_dir,
        timeout=timeout,
    )
    c.start()
    c_result = c.join(timeout)
    a_result = a.join(timeout)
    require_stream_success(a_result, f"{prefix}-A")
    require_stream_success(c_result, f"{prefix}-C")
    require(
        c_result["started_wall_ns"] < a_result["finished_wall_ns"],
        f"{prefix}: A and C did not overlap",
    )
    return {"A": a_result, "C": c_result}


def wait_for_deferral(
    trace_path: Path,
    baseline_rows: int,
    started_wall_ns: int,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = read_trace(trace_path)
        for row in rows[baseline_rows:]:
            shape = row.get("shape")
            decision = shape.get("decision") if isinstance(shape, dict) else None
            if (
                row.get("phase") == "vnext.prefill_admission"
                and decision in {"deferred", "maintenance_deferred"}
                and isinstance(row.get("request_id"), str)
                and row.get("ts_unix_nanos", 0) >= started_wall_ns
            ):
                return row["request_id"], row
        time.sleep(0.02)
    raise CapacityGateError("B did not produce a typed admission deferral before timeout")


def wait_for_pressure_streams(
    *,
    tasks: dict[str, StreamTask],
    trace_path: Path,
    trace_baseline_bytes: int,
    deferred_request_id: str,
    timeout: float,
    no_progress_timeout: float,
    max_unchanged_skips: int,
    max_trace_bytes: int,
    poll_interval: float = 0.05,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    require(set(tasks) == {"A", "B", "C"}, "pressure wait requires exactly A/B/C tasks")
    require(timeout > 0, "pressure stream timeout must be positive")
    require(no_progress_timeout > 0, "pressure progress timeout must be positive")
    require(max_unchanged_skips > 0, "pressure unchanged-skip limit must be positive")
    require(max_trace_bytes > 0, "pressure trace byte limit must be positive")
    require(poll_interval > 0, "pressure poll interval must be positive")
    require(trace_baseline_bytes >= 0, "pressure trace baseline must not be negative")

    started = time.monotonic()
    started_wall_ns = time.time_ns()
    deadline = started + timeout
    initial_progress = {
        role: task.live_content_chunks() for role, task in tasks.items()
    }
    guard = PressureStopGuard(
        initial_progress=initial_progress,
        started_monotonic=started,
        no_progress_timeout=no_progress_timeout,
        max_unchanged_skips=max_unchanged_skips,
        max_trace_bytes=max_trace_bytes,
    )
    skipped = IncrementalTracePhaseCounter(
        trace_path,
        phase="vnext.prefill_admission_skipped_unchanged",
        request_id=deferred_request_id,
    )

    while True:
        now = time.monotonic()
        active_roles = {role for role, task in tasks.items() if task.is_alive()}
        for role, task in tasks.items():
            result = getattr(task, "result", None)
            if role not in active_roles and isinstance(result, dict) and result.get("error"):
                raise CapacityGateError(f"pressure-{role}: {result['error']}")
        progress = {
            role: task.live_content_chunks() for role, task in tasks.items()
        }
        target_trace_bytes = trace_path.stat().st_size if trace_path.is_file() else 0
        require(
            target_trace_bytes >= trace_baseline_bytes,
            "target trace was truncated during pressure collection",
        )
        pressure_trace_bytes = target_trace_bytes - trace_baseline_bytes
        unchanged_skips = skipped.poll()
        guard.observe(
            progress=progress,
            unchanged_skips=unchanged_skips,
            trace_bytes=pressure_trace_bytes,
            now_monotonic=now,
            active_roles=active_roles,
        )
        if not active_roles:
            break
        require(
            now < deadline,
            f"pressure streams exceeded the joint bounded timeout: {timeout:.3f}s",
        )
        time.sleep(min(poll_interval, max(0.0, deadline - now)))

    results = {role: task.join(0) for role, task in tasks.items()}
    return results, {
        "duration_seconds": time.monotonic() - started,
        "monitor_started_wall_ns": started_wall_ns,
        "content_chunks_by_role": {
            role: task.live_content_chunks() for role, task in tasks.items()
        },
        "max_stall_seconds_by_role": guard.max_stall_seconds_by_role,
        "unchanged_epoch_skips": skipped.poll(),
        "pressure_trace_bytes": (
            trace_path.stat().st_size - trace_baseline_bytes
            if trace_path.is_file()
            else 0
        ),
        "target_trace_bytes": trace_path.stat().st_size if trace_path.is_file() else 0,
    }


def settle_stream_tasks(tasks: dict[str, StreamTask], *, timeout: float) -> list[str]:
    deadline = time.monotonic() + max(0.0, timeout)
    unsettled = []
    for role, task in tasks.items():
        if not task.settle(deadline - time.monotonic()):
            unsettled.append(role)
    return unsettled


def collect(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    binary = args.binary.resolve()
    model = args.model.resolve()
    out = args.out.resolve()
    require(not out.exists(), f"collection output already exists: {out}")
    require(binary.is_file(), f"missing binary: {binary}")
    require(model.is_dir(), f"missing model directory: {model}")
    require(args.port < 65535, "base port leaves no target-server port")
    require(
        0 < args.request_timeout <= MAX_PRESSURE_JOINT_STREAM_SECONDS,
        "request timeout must be within the canonical 300-second ceiling",
    )
    require(
        0 < args.deferral_timeout <= MAX_PRESSURE_NO_PROGRESS_SECONDS,
        "deferral timeout must be within the canonical 30-second ceiling",
    )
    pressure_stop_policy = dict(PRESSURE_STOP_POLICY)
    out.mkdir(parents=True)
    provenance = {
        "schema_version": 1,
        "command_line": sys.argv,
        "git_sha": command_output(["git", "rev-parse", "HEAD"], repo),
        "dirty_status": command_output(["git", "status", "--short"], repo),
        "binary_path": str(binary),
        "binary_sha256": sha256(binary),
        "model_path": str(model),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            repo,
        ).splitlines(),
        "sanitized_env": {
            key: os.environ[key]
            for key in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "LD_LIBRARY_PATH", "RUST_LOG")
            if key in os.environ
        },
        "started_wall_ns": time.time_ns(),
    }
    write_json(out / "provenance.json", provenance)
    require(GIT_SHA_RE.fullmatch(provenance["git_sha"]) is not None, "invalid git SHA")
    require(not provenance["dirty_status"], "CUDA capacity evidence requires a clean checkout")
    sessions: list[ServerSession] = []
    pressure_tasks: dict[str, StreamTask] = {}
    collection: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s1_cuda_capacity_pressure_collection",
        "status": "reject",
        "provenance": "provenance.json",
        "pressure_stop_policy": pressure_stop_policy,
        "error": None,
    }
    try:
        run_summary = collect_run_smoke(
            repo=repo,
            binary=binary,
            model=model,
            out_dir=out / "run",
            timeout=args.request_timeout,
        )
        calibration = ServerSession(
            repo=repo,
            binary=binary,
            model=model,
            port=args.port,
            out_dir=out / "calibration",
            runtime_budget=None,
            startup_timeout=args.startup_timeout,
        )
        sessions.append(calibration)
        calibration_clients = run_capacity_pair(
            calibration, out / "calibration" / "clients", args.request_timeout, "calibration"
        )
        calibration_health = calibration.health("health.final.json")
        executor = find_executor_snapshot(calibration_health)
        require(executor is not None, "calibration health has no vNext executor snapshot")
        calibration_pool_snapshot = quiescent_pool_snapshot(executor, "calibration")
        static_bytes = calibration_pool_snapshot["static_bytes"]
        resident_bytes = calibration_pool_snapshot["resident_bytes"]
        exact_budget = static_bytes + resident_bytes
        calibration.stop()

        target = ServerSession(
            repo=repo,
            binary=binary,
            model=model,
            port=args.port + 1,
            out_dir=out / "target",
            runtime_budget=exact_budget,
            startup_timeout=args.startup_timeout,
        )
        sessions.append(target)
        warmup_clients = run_capacity_pair(
            target, out / "target" / "warmup", args.request_timeout, "warmup"
        )
        warmup_health = target.health("health.warmup.json")
        warmup_executor = find_executor_snapshot(warmup_health)
        require(warmup_executor is not None, "warmup health has no vNext executor snapshot")
        warmup_pool_snapshot = quiescent_pool_snapshot(warmup_executor, "warmup replay")
        require_replayed_pool_snapshot(
            calibration_pool_snapshot, warmup_pool_snapshot, exact_budget
        )

        pressure_dir = out / "target" / "pressure"
        pressure_trace_start_bytes = (
            target.trace_path.stat().st_size if target.trace_path.is_file() else 0
        )
        pressure_started_monotonic = time.monotonic()
        pressure_started_wall_ns = time.time_ns()
        pressure_deadline = (
            pressure_started_monotonic
            + pressure_stop_policy["joint_stream_timeout_seconds"]
        )
        a = StreamTask(
            port=target.port,
            model=target.model_id,
            role="pressure-A",
            workload_slot="A",
            max_tokens=128,
            out_dir=pressure_dir,
            timeout=args.request_timeout,
        )
        a.start()
        pressure_tasks["A"] = a
        a.wait_first(
            bounded_remaining_timeout(
                deadline_monotonic=pressure_deadline,
                requested_timeout=args.request_timeout,
                now_monotonic=time.monotonic(),
                label="pressure A first content",
            )
        )
        baseline_rows = len(read_trace(target.trace_path))
        b = StreamTask(
            port=target.port,
            model=target.model_id,
            role="pressure-B",
            workload_slot="B",
            max_tokens=128,
            out_dir=pressure_dir,
            timeout=args.request_timeout,
        )
        b_started = time.time_ns()
        b.start()
        pressure_tasks["B"] = b
        b_request_id, b_deferral = wait_for_deferral(
            target.trace_path,
            baseline_rows,
            b_started,
            bounded_remaining_timeout(
                deadline_monotonic=pressure_deadline,
                requested_timeout=args.deferral_timeout,
                now_monotonic=time.monotonic(),
                label="pressure B deferral",
            ),
        )
        c = StreamTask(
            port=target.port,
            model=target.model_id,
            role="pressure-C",
            workload_slot="C",
            max_tokens=16,
            out_dir=pressure_dir,
            timeout=args.request_timeout,
        )
        c.start()
        pressure_tasks["C"] = c
        pressure_results, pressure_stop_observed = wait_for_pressure_streams(
            tasks={"A": a, "B": b, "C": c},
            trace_path=target.trace_path,
            trace_baseline_bytes=pressure_trace_start_bytes,
            deferred_request_id=b_request_id,
            timeout=bounded_remaining_timeout(
                deadline_monotonic=pressure_deadline,
                requested_timeout=pressure_stop_policy[
                    "joint_stream_timeout_seconds"
                ],
                now_monotonic=time.monotonic(),
                label="pressure A/B/C streams",
            ),
            no_progress_timeout=pressure_stop_policy["no_progress_timeout_seconds"],
            max_unchanged_skips=pressure_stop_policy["max_unchanged_epoch_skips"],
            max_trace_bytes=pressure_stop_policy["max_pressure_trace_bytes"],
        )
        pressure_stop_observed.update(
            {
                "lifecycle_duration_seconds": (
                    time.monotonic() - pressure_started_monotonic
                ),
                "lifecycle_started_wall_ns": pressure_started_wall_ns,
                "trace_baseline_bytes": pressure_trace_start_bytes,
            }
        )
        a_result = pressure_results["A"]
        b_result = pressure_results["B"]
        c_result = pressure_results["C"]
        for label, result in (
            ("pressure-A", a_result),
            ("pressure-B", b_result),
            ("pressure-C", c_result),
        ):
            require_stream_success(result, label)
        raw_max_silence_seconds_by_role = {}
        for role, result in pressure_results.items():
            content_times = read_stream_content_times(
                pressure_dir / f"pressure-{role}.events.jsonl"
            )
            raw_max_silence_seconds_by_role[role] = max_stream_silence_seconds(
                result,
                content_times,
                monitored_from_wall_ns=pressure_stop_observed[
                    "monitor_started_wall_ns"
                ],
            )
            require(
                raw_max_silence_seconds_by_role[role]
                < pressure_stop_policy["no_progress_timeout_seconds"],
                "raw stream token progress timeout exceeded for role "
                f"{role}: {raw_max_silence_seconds_by_role[role]:.3f}s",
            )
            require(
                pressure_stop_observed["content_chunks_by_role"][role]
                == result["content_chunks"],
                f"live content progress does not match result for role {role}",
            )
        pressure_stop_observed["raw_max_silence_seconds_by_role"] = (
            raw_max_silence_seconds_by_role
        )
        require(
            pressure_stop_observed["lifecycle_duration_seconds"]
            < pressure_stop_policy["joint_stream_timeout_seconds"],
            "pressure lifecycle exceeded the joint bounded timeout",
        )
        final_health = target.health("health.final.json")
        target.stop()
        collection.update(
            {
                "status": "collected",
                "error": None,
                "source_git_sha": provenance["git_sha"],
                "binary_sha256": provenance["binary_sha256"],
                "model_path": str(model),
                "run": run_summary,
                "calibration": {
                    "static_bytes": static_bytes,
                    "resident_bytes": resident_bytes,
                    "exact_budget_bytes": exact_budget,
                    "pool_snapshot": calibration_pool_snapshot,
                    "clients": calibration_clients,
                },
                "target": {
                    "warmup_clients": warmup_clients,
                    "warmup_pool_snapshot": warmup_pool_snapshot,
                    "pressure_clients": {
                        "A": a_result,
                        "B": b_result,
                        "C": c_result,
                    },
                    "b_request_id": b_request_id,
                    "b_deferral": b_deferral,
                    "pressure_stop_observed": pressure_stop_observed,
                    "health_warmup": "target/health.warmup.json",
                    "health_final": "target/health.final.json",
                    "trace": "target/scheduler-trace.jsonl",
                },
                "finished_wall_ns": time.time_ns(),
            }
        )
        write_json(out / "collection.json", collection)
        print(f"FERRUM RUNTIME VNEXT S1 CUDA CAPACITY PRESSURE COLLECTED: {out}")
        return 0
    except Exception as error:
        cleanup_errors = []
        for session in reversed(sessions):
            try:
                session.stop()
            except Exception as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        unsettled_roles = settle_stream_tasks(pressure_tasks, timeout=10.0)
        collection["pressure_failure_cleanup"] = {
            "client_roles": sorted(pressure_tasks),
            "unsettled_client_roles": unsettled_roles,
            "server_cleanup_errors": cleanup_errors,
            "finished_wall_ns": time.time_ns(),
        }
        collection["error"] = str(error)
        collection["finished_wall_ns"] = time.time_ns()
        write_json(out / "collection.json", collection)
        print(f"FERRUM RUNTIME VNEXT S1 CUDA CAPACITY PRESSURE REJECT: {out}: {error}", file=sys.stderr)
        return 1
    finally:
        for session in reversed(sessions):
            try:
                session.stop()
            except Exception as cleanup_error:
                print(
                    f"capacity session cleanup failed: {cleanup_error}",
                    file=sys.stderr,
                )


def event_wall_ns(row: dict[str, Any]) -> int:
    value = row.get("ts_unix_nanos")
    require(isinstance(value, int) and value > 0, "trace event has no wall timestamp")
    return value


def validate_admission_epochs(value: Any, label: str) -> dict[str, int]:
    require(isinstance(value, dict), f"{label} epochs are missing")
    coordinator_id = value.get("coordinator_id")
    release_epoch = value.get("release_epoch")
    capacity_epoch = value.get("capacity_epoch")
    require(
        isinstance(coordinator_id, int) and coordinator_id > 0,
        f"{label} coordinator is invalid",
    )
    require(
        isinstance(release_epoch, int) and release_epoch >= 0,
        f"{label} release epoch is invalid",
    )
    require(
        isinstance(capacity_epoch, int) and capacity_epoch >= 0,
        f"{label} capacity epoch is invalid",
    )
    return {
        "coordinator_id": coordinator_id,
        "release_epoch": release_epoch,
        "capacity_epoch": capacity_epoch,
    }


def validate_capacity_wait_condition(
    value: Any, *, coordinator_id: int, label: str
) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} wait condition is missing")
    require(
        value.get("coordinator_id") == coordinator_id,
        f"{label} wait condition belongs to a different coordinator",
    )
    observed = value.get("observed")
    require(
        isinstance(observed, list) and observed,
        f"{label} wait condition has no exact availability source",
    )
    seen_sources: set[str] = set()
    for entry in observed:
        require(isinstance(entry, dict), f"{label} wait source is invalid")
        source = entry.get("source")
        if isinstance(source, dict):
            require(
                set(source) == {"domain"}
                and isinstance(source.get("domain"), int)
                and source["domain"] > 0,
                f"{label} domain wait source is invalid",
            )
        else:
            require(
                source
                in {
                    "active_sequence_slots",
                    "plan_device_budget",
                    "process_device_capacity",
                },
                f"{label} wait source is unknown",
            )
        source_key = json.dumps(source, sort_keys=True, separators=(",", ":"))
        require(source_key not in seen_sources, f"{label} wait source is duplicated")
        seen_sources.add(source_key)
        require(
            isinstance(entry.get("epoch"), int) and entry["epoch"] > 0,
            f"{label} wait source epoch is invalid",
        )
    return value


def validate_device_capacity_pressure(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} pressure evidence is missing")
    scope = value.get("scope")
    require(
        scope in {"plan_budget", "process_wide"},
        f"{label} pressure scope is invalid",
    )
    require(
        isinstance(value.get("device_id"), str) and value["device_id"].strip(),
        f"{label} pressure device is invalid",
    )
    requested = value.get("requested_bytes")
    plan_claimed = value.get("plan_claimed_bytes")
    plan_usable = value.get("plan_usable_bytes")
    process_claimed = value.get("process_claimed_bytes")
    process_usable = value.get("process_usable_bytes")
    require(
        isinstance(requested, int) and requested > 0,
        f"{label} requested bytes are invalid",
    )
    require(
        all(
            isinstance(item, int) and item >= 0
            for item in (plan_claimed, plan_usable, process_claimed, process_usable)
        ),
        f"{label} capacity totals are invalid",
    )
    require(
        plan_claimed <= plan_usable and process_claimed <= process_usable,
        f"{label} claimed capacity exceeds usable capacity",
    )
    plan_available = plan_usable - plan_claimed
    process_available = process_usable - process_claimed
    if scope == "plan_budget":
        require(
            plan_available < requested,
            f"{label} plan pressure does not block the requested bytes",
        )
    else:
        require(
            plan_available >= requested and process_available < requested,
            f"{label} process pressure does not match the capacity totals",
        )
    return value


def typed_wait_for_release_transitions(
    rows: list[dict[str, Any]], request_id: str
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    pending_maintenance: dict[str, Any] | None = None
    for row in rows:
        phase = row.get("phase")
        shape = row.get("shape")
        if not isinstance(shape, dict):
            continue
        if phase == "vnext.prefill_admission":
            decision = shape.get("decision")
            if decision == "maintenance_deferred":
                require(
                    pending_maintenance is None,
                    "B produced a second maintenance deferral before completing the first",
                )
                require(
                    shape.get("maintenance_required") is True
                    and shape.get("execution_authority_retained") is False
                    and shape.get("prefill_submit_observed") is False,
                    "B maintenance deferral has inconsistent authority evidence",
                )
                evidence = row.get("attributes", {}).get("admission_evidence")
                require(
                    isinstance(evidence, dict),
                    "B maintenance deferral has no typed admission evidence",
                )
                require(
                    evidence.get("request_id") == request_id,
                    "B maintenance deferral evidence has the wrong request identity",
                )
                require(
                    evidence.get("stage") in {"logical_capacity", "physical_backing"},
                    "B maintenance deferral stage is invalid",
                )
                observed = validate_admission_epochs(
                    evidence.get("observed"), "B maintenance deferral"
                )
                validate_capacity_wait_condition(
                    evidence.get("wait_condition"),
                    coordinator_id=observed["coordinator_id"],
                    label="B maintenance deferral",
                )
                require(
                    isinstance(evidence.get("blockers"), list) and evidence["blockers"],
                    "B maintenance deferral has no blockers",
                )
                pending_maintenance = {
                    "row": row,
                    "observed": observed,
                    "wall_ns": event_wall_ns(row),
                }
            elif decision == "deferred":
                require(
                    pending_maintenance is None,
                    "B direct deferral raced an unfinished maintenance deferral",
                )
                require(
                    shape.get("maintenance_required") is False
                    and shape.get("execution_authority_retained") is False
                    and shape.get("prefill_submit_observed") is False,
                    "B direct deferral has inconsistent authority evidence",
                )
                evidence = row.get("attributes", {}).get("admission_evidence")
                require(isinstance(evidence, dict), "B direct deferral evidence is missing")
                require(
                    evidence.get("action") == "wait_for_release",
                    "B direct deferral is not WaitForRelease",
                )
                available = evidence.get("available")
                require(
                    isinstance(available, dict),
                    "B direct deferral has no capacity snapshot",
                )
                epochs = validate_admission_epochs(
                    {
                        "coordinator_id": available.get("coordinator_id"),
                        "release_epoch": evidence.get("release_epoch"),
                        "capacity_epoch": evidence.get("capacity_epoch"),
                    },
                    "B direct deferral",
                )
                wait_condition = validate_capacity_wait_condition(
                    evidence.get("wait_condition"),
                    coordinator_id=epochs["coordinator_id"],
                    label="B direct deferral",
                )
                transitions.append(
                    {
                        "source": "direct_admission",
                        "wall_ns": event_wall_ns(row),
                        "epochs": epochs,
                        "wait_condition": wait_condition,
                    }
                )
            elif decision in {"admitted", "permanent_rejected", "faulted"}:
                require(
                    pending_maintenance is None,
                    "B admission completed before retained maintenance produced an outcome",
                )
        elif phase == "vnext.prefill_backing_maintenance":
            require(
                pending_maintenance is not None,
                "B backing maintenance has no preceding typed maintenance deferral",
            )
            maintenance_ns = event_wall_ns(row)
            require(
                maintenance_ns > pending_maintenance["wall_ns"],
                "B backing maintenance did not follow its typed deferral",
            )
            outcome = shape.get("outcome")
            if outcome == "wait_for_release":
                require(
                    row.get("status") == "ok" and row.get("error") is None,
                    "B WaitForRelease maintenance event is not successful",
                )
                evidence = row.get("attributes", {}).get("maintenance_evidence")
                require(
                    isinstance(evidence, dict),
                    "B WaitForRelease maintenance evidence is missing",
                )
                require(
                    evidence.get("outcome") == "wait_for_release",
                    "B maintenance shape/evidence outcome mismatch",
                )
                current = validate_admission_epochs(
                    evidence.get("current"), "B WaitForRelease maintenance"
                )
                observed = pending_maintenance["observed"]
                require(
                    current["coordinator_id"] == observed["coordinator_id"],
                    "B maintenance changed admission coordinator",
                )
                require(
                    current["release_epoch"] >= observed["release_epoch"]
                    and current["capacity_epoch"] >= observed["capacity_epoch"],
                    "B maintenance regressed admission epochs",
                )
                wait_condition = validate_capacity_wait_condition(
                    evidence.get("wait_condition"),
                    coordinator_id=current["coordinator_id"],
                    label="B WaitForRelease maintenance",
                )
                pressure = validate_device_capacity_pressure(
                    evidence.get("pressure"), "B WaitForRelease maintenance"
                )
                required_source = (
                    "plan_device_budget"
                    if pressure["scope"] == "plan_budget"
                    else "process_device_capacity"
                )
                require(
                    any(
                        entry.get("source") == required_source
                        for entry in wait_condition["observed"]
                    ),
                    "B maintenance wait condition omits its pressure source",
                )
                transitions.append(
                    {
                        "source": "backing_maintenance",
                        "wall_ns": maintenance_ns,
                        "epochs": current,
                        "wait_condition": wait_condition,
                    }
                )
            pending_maintenance = None
    require(
        pending_maintenance is None,
        "B trace ended with unfinished typed backing maintenance",
    )
    require(transitions, "B never entered typed WaitForRelease admission")
    return transitions


def validate_stream(result: dict[str, Any], label: str) -> None:
    require_stream_success(result, label)
    require(
        result["completion_tokens"] == result["max_tokens"],
        f"{label}: ignore_eos request did not reach its exact token limit",
    )
    require(
        isinstance(result.get("first_content_wall_ns"), int),
        f"{label}: first-content timestamp is missing",
    )


def validate_replayed_workload(
    workload_slot: str, labeled_results: dict[str, dict[str, Any]]
) -> None:
    prompt_hashes: set[str] = set()
    prompt_token_counts: set[int] = set()
    for label, result in labeled_results.items():
        require(
            result.get("workload_slot") == workload_slot,
            f"{label}: workload slot does not match {workload_slot}",
        )
        prompt_hash = result.get("prompt_sha256")
        require(
            isinstance(prompt_hash, str) and SHA256_RE.fullmatch(prompt_hash) is not None,
            f"{label}: prompt SHA256 is missing",
        )
        prompt_hashes.add(prompt_hash)
        prompt_tokens = result.get("prompt_tokens")
        require(
            isinstance(prompt_tokens, int) and prompt_tokens > 0,
            f"{label}: prompt token count is missing",
        )
        prompt_token_counts.add(prompt_tokens)
    require(
        len(prompt_hashes) == 1,
        f"workload slot {workload_slot} changed its model-visible prompt across phases",
    )
    require(
        len(prompt_token_counts) == 1,
        f"workload slot {workload_slot} changed its tokenized prompt length across phases",
    )


def validate(root: Path, out: Path) -> int:
    root = root.resolve()
    out = out.resolve()
    require(root.is_dir(), f"missing collection directory: {root}")
    out.mkdir(parents=True, exist_ok=True)
    collection = read_json(root / "collection.json")
    provenance = read_json(root / "provenance.json")
    require(collection.get("status") == "collected", f"collection is not usable: {collection.get('error')}")
    pressure_stop_policy = collection.get("pressure_stop_policy")
    require(
        pressure_stop_policy == PRESSURE_STOP_POLICY,
        "collection did not use the canonical pressure stop policy",
    )
    source_git_sha = collection.get("source_git_sha")
    require(GIT_SHA_RE.fullmatch(str(source_git_sha)) is not None, "invalid source git SHA")
    require(source_git_sha == provenance.get("git_sha"), "collection/provenance git SHA mismatch")
    require(not provenance.get("dirty_status"), "capacity artifact used a dirty checkout")
    require(SHA256_RE.fullmatch(str(collection.get("binary_sha256"))) is not None, "invalid binary SHA256")
    gpu_rows = provenance.get("nvidia_smi", [])
    require(
        isinstance(gpu_rows, list) and len(gpu_rows) == 1 and "RTX 4090" in gpu_rows[0],
        "artifact is not from exactly one RTX 4090",
    )
    require("Qwen3.5-4B" in str(collection.get("model_path")), "artifact model is not Qwen3.5-4B")

    calibration = collection.get("calibration")
    target = collection.get("target")
    require(isinstance(calibration, dict) and isinstance(target, dict), "missing scenario summaries")
    exact_budget = calibration.get("exact_budget_bytes")
    require(
        exact_budget == calibration.get("static_bytes", 0) + calibration.get("resident_bytes", 0),
        "exact budget is not static plus calibrated concurrent backing",
    )
    require(calibration.get("resident_bytes", 0) > 0, "calibrated backing is empty")
    calibration_health = read_json(root / "calibration" / "health.final.json")
    warmup_health = read_json(root / "target" / "health.warmup.json")
    calibration_executor = find_executor_snapshot(calibration_health)
    warmup_executor = find_executor_snapshot(warmup_health)
    require(
        calibration_executor is not None and warmup_executor is not None,
        "raw calibration/warmup executor snapshots are missing",
    )
    calibration_pool_snapshot = quiescent_pool_snapshot(
        calibration_executor, "raw calibration"
    )
    warmup_pool_snapshot = quiescent_pool_snapshot(warmup_executor, "raw warmup replay")
    require(
        calibration_pool_snapshot.get("static_bytes") == calibration.get("static_bytes")
        and calibration_pool_snapshot.get("resident_bytes") == calibration.get("resident_bytes")
        and calibration_pool_snapshot.get("budget_claimed_bytes") == exact_budget,
        "calibration pool snapshot does not reconcile with the exact budget",
    )
    require(
        calibration.get("pool_snapshot") == calibration_pool_snapshot
        and target.get("warmup_pool_snapshot") == warmup_pool_snapshot,
        "collection pool summaries do not match raw health evidence",
    )
    require_replayed_pool_snapshot(
        calibration_pool_snapshot, warmup_pool_snapshot, exact_budget
    )
    for label, result in calibration.get("clients", {}).items():
        validate_stream(result, f"calibration-{label}")
    for label, result in target.get("warmup_clients", {}).items():
        validate_stream(result, f"warmup-{label}")
    pressure = target.get("pressure_clients")
    require(isinstance(pressure, dict) and set(pressure) == {"A", "B", "C"}, "invalid pressure client set")
    for label, result in pressure.items():
        validate_stream(result, f"pressure-{label}")
    pressure_stop_observed = target.get("pressure_stop_observed")
    require(
        isinstance(pressure_stop_observed, dict),
        "pressure stop observations are missing",
    )
    observed_lifecycle_seconds = pressure_stop_observed.get(
        "lifecycle_duration_seconds"
    )
    require(
        isinstance(observed_lifecycle_seconds, (int, float))
        and not isinstance(observed_lifecycle_seconds, bool)
        and 0 <= observed_lifecycle_seconds
        < pressure_stop_policy["joint_stream_timeout_seconds"],
        "observed pressure lifecycle exceeds the joint timeout",
    )
    observed_content_chunks = pressure_stop_observed.get("content_chunks_by_role")
    require(
        isinstance(observed_content_chunks, dict)
        and set(observed_content_chunks) == {"A", "B", "C"},
        "observed pressure content progress is incomplete",
    )
    for role, result in pressure.items():
        require(
            observed_content_chunks.get(role) == result.get("content_chunks"),
            f"observed pressure content progress differs for role {role}",
        )

    calibration_clients = calibration.get("clients", {})
    warmup_clients = target.get("warmup_clients", {})
    require(
        isinstance(calibration_clients, dict)
        and isinstance(warmup_clients, dict)
        and {"A", "C"}.issubset(calibration_clients)
        and {"A", "C"}.issubset(warmup_clients),
        "calibration/warmup replay client set is incomplete",
    )
    for workload_slot in ("A", "C"):
        validate_replayed_workload(
            workload_slot,
            {
                "calibration": calibration_clients[workload_slot],
                "warmup": warmup_clients[workload_slot],
                "pressure": pressure[workload_slot],
            },
        )
    validate_replayed_workload("B", {"pressure": pressure["B"]})
    require(
        len({result["prompt_sha256"] for result in pressure.values()}) == 3,
        "capacity slots do not have distinct model-visible prompts",
    )

    a, b, c = pressure["A"], pressure["B"], pressure["C"]
    b_request_id = target.get("b_request_id")
    require(isinstance(b_request_id, str) and b_request_id, "B request identity is missing")
    target_trace_path = root / str(target.get("trace"))
    rows = read_trace(target_trace_path)
    require(rows, "target scheduler trace is empty")
    target_trace_bytes = target_trace_path.stat().st_size
    pressure_trace_bytes = trace_bytes_at_or_after(
        target_trace_path, a["started_wall_ns"]
    )
    require(
        pressure_trace_bytes <= pressure_stop_policy["max_pressure_trace_bytes"],
        "pressure trace exceeds the artifact stop-policy ceiling",
    )
    observed_pressure_trace_bytes = pressure_stop_observed.get(
        "pressure_trace_bytes"
    )
    require(
        isinstance(observed_pressure_trace_bytes, int)
        and 0 <= observed_pressure_trace_bytes
        <= pressure_stop_policy["max_pressure_trace_bytes"],
        "observed pressure trace bytes are invalid",
    )
    b_rows = [
        row
        for row in rows
        if request_identity_matches(row.get("request_id"), b_request_id)
    ]
    wait_transitions = typed_wait_for_release_transitions(b_rows, b_request_id)
    defer_ns = min(transition["wall_ns"] for transition in wait_transitions)
    skipped = [
        row
        for row in b_rows
        if row.get("phase") == "vnext.prefill_admission_skipped_unchanged"
        and event_wall_ns(row) > defer_ns
    ]
    require(skipped, "B has no unchanged-epoch skip evidence")
    require(
        len(skipped) <= pressure_stop_policy["max_unchanged_epoch_skips"],
        "B unchanged-epoch skips exceed the artifact stop-policy ceiling",
    )
    require(
        pressure_stop_observed.get("unchanged_epoch_skips") == len(skipped),
        "observed B unchanged-epoch skips differ from raw trace",
    )
    require(
        all(row.get("shape", {}).get("probe_performed") is False for row in skipped),
        "unchanged-epoch skip performed a probe",
    )
    admitted = [
        row
        for row in b_rows
        if row.get("phase") == "vnext.prefill_admission"
        and row.get("shape", {}).get("decision") == "admitted"
        and event_wall_ns(row) > defer_ns
    ]
    require(admitted, "B was not admitted after a release epoch")
    admitted_ns = event_wall_ns(admitted[0])
    submitted = [row for row in b_rows if str(row.get("phase", "")).endswith("operation_submitted")]
    require(submitted, "B has no operation submission evidence")
    require(
        min(event_wall_ns(row) for row in submitted) >= admitted_ns,
        "B submitted work before typed admission",
    )
    require(
        any(str(row.get("phase", "")).endswith("request_completed") for row in b_rows),
        "B has no vNext request completion",
    )

    require(a["started_wall_ns"] < b["started_wall_ns"] < c["started_wall_ns"], "client arrival order is not A/B/C")
    require(b["started_wall_ns"] <= defer_ns <= c["started_wall_ns"], "C did not arrive after B was deferred")
    raw_max_silence_seconds_by_role = {}
    for role, result in pressure.items():
        content_times = read_stream_content_times(
            root / "target" / "pressure" / f"pressure-{role}.events.jsonl"
        )
        raw_max_silence_seconds_by_role[role] = max_stream_silence_seconds(
            result,
            content_times,
            monitored_from_wall_ns=c["started_wall_ns"],
        )
        require(
            raw_max_silence_seconds_by_role[role]
            < pressure_stop_policy["no_progress_timeout_seconds"],
            "raw stream token progress timeout exceeded for role "
            f"{role}: {raw_max_silence_seconds_by_role[role]:.3f}s",
        )
    raw_lifecycle_seconds = (
        max(result["finished_wall_ns"] for result in pressure.values())
        - a["started_wall_ns"]
    ) / 1_000_000_000
    require(
        raw_lifecycle_seconds
        < pressure_stop_policy["joint_stream_timeout_seconds"],
        "raw pressure lifecycle exceeds the joint timeout",
    )
    a_events = root / "target" / "pressure" / "pressure-A.events.jsonl"
    a_content_times = read_stream_content_times(a_events)
    require(any(defer_ns < value < admitted_ns for value in a_content_times), "active A decode made no progress while B waited")
    require(c["finished_wall_ns"] < b["first_content_wall_ns"], "eligible C did not bypass deferred B")

    final_health = read_json(root / str(target.get("health_final")))
    final_executor = find_executor_snapshot(final_health)
    require(final_executor is not None, "final health has no vNext executor snapshot")
    final_pools = final_executor.get("dynamic_pools", {})
    final_epochs = final_pools.get("epochs", {})
    require(
        final_epochs.get("release_epoch", 0)
        > max(
            transition["epochs"]["release_epoch"]
            for transition in wait_transitions
        ),
        "final capacity release epoch did not advance beyond B's observation",
    )
    require(final_executor.get("active_sequences") == 0, "final executor still has active sequences")
    require(final_executor.get("pending_sequences") == 0, "final executor still has pending sequences")
    require(final_executor.get("pending_prefill_maintenance") == 0, "final executor still has maintenance state")
    require(final_health.get("engine", {}).get("active_requests") == 0, "final engine has active requests")
    require(final_health.get("engine", {}).get("queued_requests") == 0, "final engine has queued requests")
    policy = final_executor.get("runtime_memory_policy", {})
    require(
        policy.get("capacity_bytes", 0) - policy.get("reserve_bytes", 0) == exact_budget,
        "target runtime did not use the calibrated exact memory budget",
    )

    run_result = read_json(root / "run" / "result.json")
    require(run_result.get("returncode") == 0, "ferrum run smoke failed")
    run_rows = [
        json.loads(line)
        for line in (root / "run" / "stdout.jsonl").read_text().splitlines()
        if line.strip()
    ]
    require(
        any("paris" in str(row.get("content", "")).lower() for row in run_rows),
        "ferrum run smoke did not return Paris",
    )
    run_phases = {
        row.get("phase") for row in read_trace(root / "run" / "scheduler-trace.jsonl")
    }
    require("vnext.operation_submitted" in run_phases, "ferrum run did not use vNext execution")
    require("vnext.request_completed" in run_phases, "ferrum run did not complete vNext execution")

    for path in [*root.rglob("*.log"), *root.rglob("*.sse")]:
        text = path.read_text(errors="replace").lower()
        require("\ufffd" not in text, f"Unicode replacement character in {path}")
        for pattern in FORBIDDEN_PATTERNS:
            require(pattern not in text, f"forbidden pattern {pattern!r} in {path}")

    pass_line = f"{PASS_PREFIX}: {out}"
    manifest = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s1_cuda_capacity_pressure_validation",
        "status": "pass",
        "source_git_sha": source_git_sha,
        "binary_sha256": collection["binary_sha256"],
        "model_path": collection["model_path"],
        "source_artifact": str(root),
        "source_collection_sha256": sha256(root / "collection.json"),
        "exact_budget_bytes": exact_budget,
        "b_request_id": b_request_id,
        "b_wait_for_release_events": len(wait_transitions),
        "b_wait_for_release_sources": sorted(
            {transition["source"] for transition in wait_transitions}
        ),
        "b_unchanged_epoch_skips": len(skipped),
        "target_trace_bytes": target_trace_bytes,
        "pressure_trace_bytes": pressure_trace_bytes,
        "pressure_stop_policy": pressure_stop_policy,
        "pressure_stop_observed": pressure_stop_observed,
        "pressure_stop_recomputed": {
            "lifecycle_duration_seconds": raw_lifecycle_seconds,
            "max_silence_seconds_by_role": raw_max_silence_seconds_by_role,
            "unchanged_epoch_skips": len(skipped),
            "pressure_trace_bytes": pressure_trace_bytes,
        },
        "capacity_release_epoch_before": max(
            transition["epochs"]["release_epoch"]
            for transition in wait_transitions
        ),
        "capacity_release_epoch_after": final_epochs["release_epoch"],
        "does_not_prove": ["G01B", "S1", "performance", "release"],
        "pass_line": pass_line,
    }
    write_json(out / "validation.json", manifest)
    write_json(out / "manifest.json", manifest)
    print(pass_line)
    return 0


def self_test() -> int:
    snapshot = {
        "cache": {
            "prefix_cache": {
                "schema": "ferrum.runtime-vnext.executor-trace.v1",
                "static_bytes": 7,
            }
        }
    }
    require(find_executor_snapshot(snapshot) == snapshot["cache"]["prefix_cache"], "snapshot discovery failed")
    require(request_identity_matches("request.product.abc", "abc"), "request identity mapping failed")
    executor = {
        "static_bytes": 7,
        "dynamic_pools": {
            "budget_claimed_bytes": 19,
            "pools": [
                {
                    "pool_id": "pool-a",
                    "resident_bytes": 5,
                    "resident_chunks": 1,
                    "free_bytes": 5,
                    "largest_contiguous_bytes": 5,
                    "pending_growth_bytes": 0,
                    "live_segments": 0,
                    "quarantined_bytes": 0,
                    "quarantined_chunks": 0,
                    "descriptor_mismatch_chunks": 0,
                    "publication_rejected_chunks": 0,
                    "poisoned": False,
                    "storage_profile": {"allocator": "linear", "view": "contiguous"},
                },
                {
                    "pool_id": "pool-b",
                    "resident_bytes": 7,
                    "resident_chunks": 1,
                    "free_bytes": 7,
                    "largest_contiguous_bytes": 7,
                    "pending_growth_bytes": 0,
                    "live_segments": 0,
                    "quarantined_bytes": 0,
                    "quarantined_chunks": 0,
                    "descriptor_mismatch_chunks": 0,
                    "publication_rejected_chunks": 0,
                    "poisoned": False,
                    "storage_profile": {"allocator": "linear", "view": "contiguous"},
                },
            ],
        },
    }
    pool_snapshot = quiescent_pool_snapshot(executor, "self-test")
    require_replayed_pool_snapshot(pool_snapshot, pool_snapshot, 19)
    drifted = json.loads(json.dumps(pool_snapshot))
    drifted["pool_resident_bytes"] = {"pool-a": 6, "pool-b": 6}
    try:
        require_replayed_pool_snapshot(pool_snapshot, drifted, 19)
        raise AssertionError("per-pool replay drift unexpectedly passed")
    except CapacityGateError:
        pass
    prompt_hashes = {
        slot: hashlib.sha256(capacity_prompt(slot).encode("utf-8")).hexdigest()
        for slot in ("A", "B", "C")
    }
    require(len(set(prompt_hashes.values())) == 3, "capacity workload prompts are not slot-specific")
    validate_replayed_workload(
        "A",
        {
            "calibration": {
                "workload_slot": "A",
                "prompt_sha256": prompt_hashes["A"],
                "prompt_tokens": 32,
            },
            "warmup": {
                "workload_slot": "A",
                "prompt_sha256": prompt_hashes["A"],
                "prompt_tokens": 32,
            },
            "pressure": {
                "workload_slot": "A",
                "prompt_sha256": prompt_hashes["A"],
                "prompt_tokens": 32,
            },
        },
    )
    try:
        validate_replayed_workload(
            "A",
            {
                "calibration": {
                    "workload_slot": "A",
                    "prompt_sha256": prompt_hashes["A"],
                    "prompt_tokens": 32,
                },
                "warmup": {
                    "workload_slot": "A",
                    "prompt_sha256": prompt_hashes["B"],
                    "prompt_tokens": 32,
                },
            },
        )
        raise AssertionError("mismatched replay prompt unexpectedly passed")
    except CapacityGateError:
        pass
    def expect_gate_error(action: Callable[[], Any], expected: str) -> None:
        try:
            action()
        except CapacityGateError as error:
            require(expected in str(error), f"unexpected gate error: {error}")
            return
        raise AssertionError(f"expected gate error containing {expected!r}")

    wait_condition = {
        "coordinator_id": 7,
        "observed": [
            {"source": {"domain": 5}, "epoch": 3},
            {"source": "plan_device_budget", "epoch": 2},
        ],
    }
    direct_wait = {
        "ts_unix_nanos": 10,
        "phase": "vnext.prefill_admission",
        "shape": {
            "decision": "deferred",
            "execution_authority_retained": False,
            "maintenance_required": False,
            "prefill_submit_observed": False,
        },
        "attributes": {
            "admission_evidence": {
                "action": "wait_for_release",
                "available": {"coordinator_id": 7},
                "release_epoch": 11,
                "capacity_epoch": 13,
                "wait_condition": wait_condition,
            }
        },
    }
    direct_transitions = typed_wait_for_release_transitions([direct_wait], "B")
    require(
        len(direct_transitions) == 1
        and direct_transitions[0]["source"] == "direct_admission",
        "direct WaitForRelease transition was not recognized",
    )

    maintenance_deferred = {
        "ts_unix_nanos": 20,
        "phase": "vnext.prefill_admission",
        "shape": {
            "decision": "maintenance_deferred",
            "execution_authority_retained": False,
            "maintenance_required": True,
            "prefill_submit_observed": False,
        },
        "attributes": {
            "admission_evidence": {
                "request_id": "B",
                "observed": {
                    "coordinator_id": 7,
                    "release_epoch": 11,
                    "capacity_epoch": 13,
                },
                "wait_condition": {
                    "coordinator_id": 7,
                    "observed": [{"source": {"domain": 5}, "epoch": 3}],
                },
                "stage": "physical_backing",
                "blockers": [{"source": "backing"}],
            }
        },
    }
    maintenance_wait = {
        "ts_unix_nanos": 30,
        "phase": "vnext.prefill_backing_maintenance",
        "status": "ok",
        "error": None,
        "shape": {"outcome": "wait_for_release"},
        "attributes": {
            "maintenance_evidence": {
                "outcome": "wait_for_release",
                "current": {
                    "coordinator_id": 7,
                    "release_epoch": 12,
                    "capacity_epoch": 14,
                },
                "wait_condition": wait_condition,
                "pressure": {
                    "scope": "plan_budget",
                    "device_id": "device.cuda.0",
                    "requested_bytes": 8,
                    "plan_claimed_bytes": 100,
                    "plan_usable_bytes": 100,
                    "process_claimed_bytes": 100,
                    "process_usable_bytes": 200,
                },
            }
        },
    }
    maintenance_transitions = typed_wait_for_release_transitions(
        [maintenance_deferred, maintenance_wait], "B"
    )
    require(
        len(maintenance_transitions) == 1
        and maintenance_transitions[0]["source"] == "backing_maintenance",
        "two-stage WaitForRelease transition was not recognized",
    )
    missing_evidence = json.loads(json.dumps(maintenance_wait))
    missing_evidence["attributes"] = {}
    expect_gate_error(
        lambda: typed_wait_for_release_transitions(
            [maintenance_deferred, missing_evidence], "B"
        ),
        "maintenance evidence is missing",
    )
    expect_gate_error(
        lambda: typed_wait_for_release_transitions(
            [maintenance_wait, maintenance_deferred], "B"
        ),
        "no preceding typed maintenance deferral",
    )
    mismatched_outcome = json.loads(json.dumps(maintenance_wait))
    mismatched_outcome["attributes"]["maintenance_evidence"]["outcome"] = (
        "retry_admission"
    )
    expect_gate_error(
        lambda: typed_wait_for_release_transitions(
            [maintenance_deferred, mismatched_outcome], "B"
        ),
        "shape/evidence outcome mismatch",
    )

    with __import__("tempfile").TemporaryDirectory() as temp:
        trace = Path(temp) / "trace.jsonl"
        trace.write_text('{"phase":"complete"}\n{"phase":')
        require(read_trace(trace) == [{"phase": "complete"}], "partial trace handling failed")

        phase = "vnext.prefill_admission_skipped_unchanged"
        warmup_row = json.dumps({"phase": "warmup"}) + "\n"
        trace.write_text(
            warmup_row
            + json.dumps({"phase": phase, "request_id": "B"})
            + "\n"
            + '{"phase":"vnext.prefill_admission_skipped_unchanged","request_id":"'
        )
        baseline = len(warmup_row.encode())
        counter = IncrementalTracePhaseCounter(trace, phase=phase, request_id="B")
        require(counter.poll() == 1, "incremental trace counter missed a row")
        with trace.open("a") as handle:
            handle.write('B"}\n' + json.dumps({"phase": phase, "request_id": "C"}) + "\n")
        require(counter.poll() == 2, "incremental trace counter lost a partial row")

        timed_trace = Path(temp) / "timed-trace.jsonl"
        warmup = json.dumps({"ts_unix_nanos": 100, "phase": "warmup"}) + "\n"
        pressure = json.dumps({"ts_unix_nanos": 200, "phase": "pressure"}) + "\n"
        timed_trace.write_text(warmup + pressure)
        require(
            trace_bytes_at_or_after(timed_trace, 150) == len(pressure.encode()),
            "pressure trace bytes included warmup",
        )

        task = StreamTask(
            port=1,
            model="self-test",
            role="self-test",
            workload_slot="A",
            max_tokens=1,
            out_dir=Path(temp),
            timeout=1.0,
        )
        task._record_content(123)
        require(task.live_content_chunks() == 1, "live stream progress was not published")

        class FinishedTask:
            result = None

            def __init__(self, role: str) -> None:
                self.role = role

            def is_alive(self) -> bool:
                return False

            def live_content_chunks(self) -> int:
                return 1

            def join(self, timeout: float) -> dict[str, Any]:
                return {"role": self.role}

        results, observed = wait_for_pressure_streams(
            tasks={role: FinishedTask(role) for role in ("A", "B", "C")},
            trace_path=trace,
            trace_baseline_bytes=baseline,
            deferred_request_id="B",
            timeout=1.0,
            no_progress_timeout=0.5,
            max_unchanged_skips=2,
            max_trace_bytes=trace.stat().st_size - baseline,
            poll_interval=0.001,
        )
        require(set(results) == {"A", "B", "C"}, "bounded wait lost a result")
        require(observed["unchanged_epoch_skips"] == 2, "bounded wait lost skip evidence")
        require(
            observed["pressure_trace_bytes"] == trace.stat().st_size - baseline,
            "bounded wait counted warmup trace bytes",
        )

    require(
        bounded_remaining_timeout(
            deadline_monotonic=150.0,
            requested_timeout=300.0,
            now_monotonic=100.0,
            label="self-test",
        )
        == 50.0,
        "joint deadline did not cap a phase timeout",
    )
    expect_gate_error(
        lambda: bounded_remaining_timeout(
            deadline_monotonic=100.0,
            requested_timeout=30.0,
            now_monotonic=100.0,
            label="self-test",
        ),
        "joint pressure lifecycle timeout",
    )
    stream_result = {
        "started_wall_ns": 90_000_000_000,
        "finished_wall_ns": 140_000_000_000,
    }
    require(
        max_stream_silence_seconds(
            stream_result,
            [110_000_000_000, 125_000_000_000],
            monitored_from_wall_ns=100_000_000_000,
        )
        == 15.0,
        "stream silence used the wrong interval",
    )

    def new_guard() -> Any:
        return PressureStopGuard(
            initial_progress={"A": 1, "B": 0},
            started_monotonic=100.0,
            no_progress_timeout=30.0,
            max_unchanged_skips=512,
            max_trace_bytes=16 * 1024 * 1024,
        )

    guard = new_guard()
    guard.observe(
        progress={"A": 2, "B": 0},
        unchanged_skips=512,
        trace_bytes=16 * 1024 * 1024,
        now_monotonic=129.5,
        active_roles={"A", "B"},
    )
    expect_gate_error(
        lambda: guard.observe(
            progress={"A": 3, "B": 0},
            unchanged_skips=512,
            trace_bytes=16 * 1024 * 1024,
            now_monotonic=130.0,
            active_roles={"A", "B"},
        ),
        "role B",
    )
    expect_gate_error(
        lambda: new_guard().observe(
            progress={"A": 1, "B": 0},
            unchanged_skips=513,
            trace_bytes=0,
            now_monotonic=100.0,
            active_roles=set(),
        ),
        "skip limit exceeded",
    )
    expect_gate_error(
        lambda: new_guard().observe(
            progress={"A": 1, "B": 0},
            unchanged_skips=0,
            trace_bytes=MAX_PRESSURE_TRACE_BYTES + 1,
            now_monotonic=100.0,
            active_roles=set(),
        ),
        "trace byte limit exceeded",
    )
    print("FERRUM RUNTIME VNEXT S1 CUDA CAPACITY SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--repo", type=Path, default=Path.cwd())
    collect_parser.add_argument("--binary", type=Path, required=True)
    collect_parser.add_argument("--model", type=Path, required=True)
    collect_parser.add_argument("--out", type=Path, required=True)
    collect_parser.add_argument("--port", type=int, default=18120)
    collect_parser.add_argument("--startup-timeout", type=float, default=600)
    collect_parser.add_argument("--request-timeout", type=float, default=300)
    collect_parser.add_argument("--deferral-timeout", type=float, default=30)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("artifact", type=Path)
    validate_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test()
        if args.command == "collect":
            return collect(args)
        if args.command == "validate":
            return validate(args.artifact, args.out)
        parser.error("a command is required")
    except CapacityGateError as error:
        target = getattr(args, "out", Path("."))
        print(f"FERRUM RUNTIME VNEXT S1 CUDA CAPACITY PRESSURE FAIL: {target}: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
