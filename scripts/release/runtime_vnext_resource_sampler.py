#!/usr/bin/env python3
"""Collect and validate process-bound resource evidence for Runtime vNext G00.

The collector writes append-only JSONL: one header, timestamped observations,
and one footer.  The G00 validator imports :func:`derive_summary` and rebuilds
all resource summaries from these observations instead of trusting aggregate
fields supplied by a runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
COLLECTOR_RELATIVE_PATH = "scripts/release/runtime_vnext_resource_sampler.py"
PASS_PREFIX = "FERRUM RUNTIME VNEXT RESOURCE SAMPLER PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT RESOURCE SAMPLER SELFTEST PASS"
OOM_RE = re.compile(
    r"(?:out[ -]of[ -]memory|\bcuda\s+oom\b|\bmetal\s+oom\b|"
    r"memory allocation (?:failed|failure)|allocator exhaustion)",
    re.IGNORECASE,
)
ADMISSION_ERROR_RE = re.compile(
    r"(?:admission (?:failed|failure|rejected|error)|kv (?:admission|capacity) (?:failed|overflow)|"
    r"resource exhausted)",
    re.IGNORECASE,
)


class ResourceEvidenceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResourceEvidenceError(message)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any, label: str) -> datetime:
    _require(isinstance(value, str) and value, f"{label} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResourceEvidenceError(f"{label} is not ISO-8601") from exc
    _require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collector_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"resource observations missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            _require(line.endswith("\n"), f"resource observations line {line_number} is not newline terminated")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResourceEvidenceError(
                    f"resource observations line {line_number} is not JSON"
                ) from exc
            _require(isinstance(row, dict), f"resource observations line {line_number} must be an object")
            rows.append(row)
    _require(len(rows) >= 5, "resource observations require a header, at least three samples, and a footer")
    return rows


def _positive_int(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{label} must be a non-negative integer")
    return value


def process_marker_from_source(pid: int, raw: Any) -> str:
    source = raw if isinstance(raw, dict) else {}
    kind = source.get("kind")
    if kind == "linux-proc-start":
        boot_id = source.get("boot_id")
        start_ticks = source.get("start_ticks")
        _require(isinstance(boot_id, str) and bool(boot_id), "process start source boot_id is invalid")
        _require(isinstance(start_ticks, str) and start_ticks.isdigit(), "process start source start_ticks is invalid")
        return f"linux:{boot_id}:{start_ticks}"
    if kind == "ps-lstart":
        started = source.get("raw_output")
        _require(isinstance(started, str) and bool(started.strip()), "process start source ps output is invalid")
        return f"darwin:{pid}:{' '.join(started.split())}"
    raise ResourceEvidenceError("process start source kind is invalid")


def require_active_coverage(values: list[int], *, process_probe: bool) -> None:
    _require(bool(values), "resource active coverage is empty")
    if process_probe:
        _require(all(value == 1 for value in values), "one-shot process probe must observe active=1")
    else:
        _require(max(values) > 0, "HTTP scheduler probe never observed active work in the measured window")


def derive_summary(
    path: Path,
    *,
    session_id: str,
    cell_id: str,
    backend: str,
    hardware_id: str,
    pid: int,
    pgid: int,
    process_start_marker: str,
    base_url: str,
    session_started_at: str,
    session_finished_at: str,
    measurement_started_at: str,
    measurement_finished_at: str,
    memory_budget_bytes: int,
    requested_concurrency: int,
    typed_active_cap: int,
    runtime_log_path: str,
) -> dict[str, Any]:
    """Validate raw observations and derive the only accepted resource summary."""

    rows = _read_jsonl(path)
    header = rows[0]
    footer = rows[-1]
    samples = rows[1:-1]
    _require(header.get("record_type") == "header", "resource observations must start with a header")
    _require(footer.get("record_type") == "footer", "resource observations must end with a footer")
    _require(header.get("schema_version") == SCHEMA_VERSION, "resource observation schema mismatch")
    _require(header.get("collector_path") == COLLECTOR_RELATIVE_PATH, "resource collector path mismatch")
    _require(header.get("collector_sha256") == collector_sha256(), "resource collector SHA256 mismatch")
    expected_header = {
        "session_id": session_id,
        "cell_id": cell_id,
        "backend": backend,
        "hardware_id": hardware_id,
        "pid": pid,
        "pgid": pgid,
        "process_start_marker": process_start_marker,
        "base_url": base_url,
    }
    for field, expected in expected_header.items():
        _require(header.get(field) == expected, f"resource header {field} mismatch")
    _require(
        process_marker_from_source(pid, header.get("process_start_source")) == process_start_marker,
        "resource process start marker is not derived from raw OS identity evidence",
    )
    interval_ms = _positive_int(header.get("interval_ms"), "resource header interval_ms")
    _require(50 <= interval_ms <= 1000, "resource sampling interval must be 50..1000 ms")
    probe = header.get("active_probe")
    _require(isinstance(probe, dict), "resource header active_probe must be an object")
    _require(probe.get("format") in {"json", "prometheus", "process"}, "resource active probe format is invalid")
    expected_probe_url = "" if probe.get("format") == "process" else f"{base_url.rstrip('/')}{probe.get('path', '')}"
    _require(probe.get("url") == expected_probe_url, "resource active probe URL/base mismatch")
    _require(isinstance(probe.get("selector"), str) and probe["selector"], "resource active probe selector is missing")
    expected_semantics = "process-alive" if probe.get("format") == "process" else "scheduler-active-high-water"
    _require(probe.get("semantics") == expected_semantics, "resource active probe semantics mismatch")
    _require(header.get("runtime_log_path") == runtime_log_path, "resource runtime log path mismatch")

    session_start = _parse_timestamp(session_started_at, "session_started_at")
    session_finish = _parse_timestamp(session_finished_at, "session_finished_at")
    measure_start = _parse_timestamp(measurement_started_at, "measurement_started_at")
    measure_finish = _parse_timestamp(measurement_finished_at, "measurement_finished_at")
    _require(session_start < measure_start < measure_finish < session_finish, "resource measurement window is outside session")
    header_started = _parse_timestamp(header.get("started_at"), "resource header started_at")
    footer_finished = _parse_timestamp(footer.get("finished_at"), "resource footer finished_at")
    _require(session_start <= header_started < measure_start, "resource sampler must start after process start and before measurement")
    _require(measure_finish < footer_finished <= session_finish, "resource sampler must finish after measurement and before session finish")
    _require(footer.get("exit_reason") in {"stop-file", "signal", "duration"}, "resource footer exit_reason is invalid")
    _require(footer.get("sample_count") == len(samples), "resource footer sample_count mismatch")

    timestamps: list[datetime] = []
    normalized: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        label = f"resource sample[{index}]"
        _require(sample.get("record_type") == "sample", f"{label}.record_type must be sample")
        _require(sample.get("sequence") == index, f"{label}.sequence must be contiguous")
        timestamp = _parse_timestamp(sample.get("sampled_at"), f"{label}.sampled_at")
        if timestamps:
            _require(timestamp > timestamps[-1], f"{label}.sampled_at must be strictly increasing")
        _require(header_started <= timestamp <= footer_finished, f"{label} is outside sampler lifetime")
        _require(sample.get("pid") == pid and sample.get("pgid") == pgid, f"{label} pid/pgid mismatch")
        _require(sample.get("process_start_marker") == process_start_marker, f"{label} process start marker mismatch")
        _require(sample.get("process_alive") is True, f"{label} did not observe the server process alive")
        process_rss = _positive_int(sample.get("process_rss_bytes"), f"{label}.process_rss_bytes")
        memory_used = _positive_int(sample.get("memory_used_bytes"), f"{label}.memory_used_bytes")
        headroom = _nonnegative_int(sample.get("physical_headroom_bytes"), f"{label}.physical_headroom_bytes")
        swap = _nonnegative_int(sample.get("swap_used_bytes"), f"{label}.swap_used_bytes")
        active = _nonnegative_int(sample.get("active_requests"), f"{label}.active_requests")
        oom_count = _nonnegative_int(sample.get("oom_count"), f"{label}.oom_count")
        admission_errors = _nonnegative_int(sample.get("admission_error_count"), f"{label}.admission_error_count")
        if backend == "metal":
            _require(memory_used == process_rss, f"{label}.memory_used_bytes must equal process RSS on Metal")
            _require(sample.get("thermal_state") in {"nominal", "throttled"}, f"{label}.thermal_state is invalid")
            _require(sample.get("power_mode") in {"normal", "low-power"}, f"{label}.power_mode is invalid")
        else:
            _require(backend == "cuda", f"{label}.backend is invalid")
            _positive_int(sample.get("device_memory_bytes"), f"{label}.device_memory_bytes")
            _require(memory_used == sample["device_memory_bytes"], f"{label}.memory_used_bytes must equal device memory on CUDA")
        timestamps.append(timestamp)
        normalized.append(
            {
                **sample,
                "_timestamp": timestamp,
                "_memory_used": memory_used,
                "_headroom": headroom,
                "_swap": swap,
                "_active": active,
                "_oom": oom_count,
                "_admission": admission_errors,
            }
        )

    before = [index for index, row in enumerate(normalized) if row["_timestamp"] <= measure_start]
    after = [index for index, row in enumerate(normalized) if row["_timestamp"] >= measure_finish]
    _require(before and after, "resource observations do not bracket the measurement window")
    start_index = before[-1]
    finish_index = after[0]
    covered = normalized[start_index : finish_index + 1]
    in_window = [row for row in normalized if measure_start <= row["_timestamp"] <= measure_finish]
    _require(len(in_window) >= 3, "resource observations need at least three samples inside measurement window")
    max_gap = max(2.0, interval_ms / 1000.0 * 4.0)
    for left, right in zip(covered, covered[1:]):
        gap = (right["_timestamp"] - left["_timestamp"]).total_seconds()
        _require(gap <= max_gap, f"resource observation gap {gap:.3f}s exceeds {max_gap:.3f}s")
    _require(all(row["_active"] <= requested_concurrency for row in covered), "raw active requests exceed requested concurrency")
    _require(all(row["_active"] <= typed_active_cap for row in covered), "raw active requests exceed typed active cap")
    require_active_coverage(
        [row["_active"] for row in covered],
        process_probe=probe.get("format") == "process",
    )
    _require(all(row["_oom"] == 0 for row in covered), "raw resource observations contain OOM events")
    _require(all(row["_admission"] == 0 for row in covered), "raw resource observations contain admission errors")

    summary: dict[str, Any] = {
        "sample_count": len(covered),
        "first_sample_at": covered[0]["sampled_at"],
        "last_sample_at": covered[-1]["sampled_at"],
        "peak_memory_bytes": max(row["_memory_used"] for row in covered),
        "memory_budget_bytes": memory_budget_bytes,
        "physical_headroom_bytes": min(row["_headroom"] for row in covered),
        "swap_start_bytes": covered[0]["_swap"],
        "swap_end_bytes": covered[-1]["_swap"],
        "oom_count": max(row["_oom"] for row in covered),
        "admission_error_count": max(row["_admission"] for row in covered),
        "observed_max_active": max(row["_active"] for row in covered),
    }
    if backend == "metal":
        summary.update(
            {
                "thermal_start": covered[0]["thermal_state"],
                "thermal_end": covered[-1]["thermal_state"],
                "power_mode_start": covered[0]["power_mode"],
                "power_mode_end": covered[-1]["power_mode"],
                "thermal_throttling_count": sum(row["thermal_state"] != "nominal" for row in covered),
            }
        )
    return summary


def _run(argv: list[str]) -> str:
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    return result.stdout


def _process_group_rows(pgid: int) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for raw in _run(["ps", "-axo", "pid=,pgid=,rss="]).splitlines():
        fields = raw.split()
        if len(fields) != 3:
            continue
        row_pid, row_pgid, rss_kib = map(int, fields)
        if row_pgid == pgid:
            rows.append((row_pid, rss_kib * 1024))
    return rows


def _process_start_identity(pid: int) -> tuple[str, dict[str, str]]:
    if platform.system() == "Linux":
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = stat.rfind(")")
        _require(close > 0, f"cannot parse /proc/{pid}/stat")
        fields = stat[close + 2 :].split()
        _require(len(fields) >= 20, f"/proc/{pid}/stat is incomplete")
        start_ticks = fields[19]
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        source = {"kind": "linux-proc-start", "boot_id": boot_id, "start_ticks": start_ticks}
        return process_marker_from_source(pid, source), source
    started = _run(["ps", "-o", "lstart=", "-p", str(pid)]).strip()
    _require(bool(started), f"ps did not report process start time for pid {pid}")
    source = {"kind": "ps-lstart", "raw_output": started}
    return process_marker_from_source(pid, source), source


def _linux_memory() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        fields = raw.split()
        if fields:
            values[key] = int(fields[0]) * 1024
    return values["MemAvailable"], values.get("SwapTotal", 0) - values.get("SwapFree", 0)


def _mac_memory() -> tuple[int, int]:
    output = _run(["vm_stat"])
    page_match = re.search(r"page size of (\d+) bytes", output)
    _require(page_match is not None, "vm_stat did not report page size")
    page_size = int(page_match.group(1))
    pages: dict[str, int] = {}
    for line in output.splitlines()[1:]:
        match = re.match(r"([^:]+):\s+(\d+)\.", line)
        if match:
            pages[match.group(1)] = int(match.group(2))
    available_keys = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    available = sum(pages.get(key, 0) for key in available_keys) * page_size
    swap_raw = _run(["sysctl", "-n", "vm.swapusage"])
    match = re.search(r"used\s*=\s*([0-9.]+)([MG])", swap_raw)
    _require(match is not None, "vm.swapusage did not report used swap")
    multiplier = 1024**2 if match.group(2) == "M" else 1024**3
    return available, int(float(match.group(1)) * multiplier)


def _mac_thermal_power() -> tuple[str, str]:
    thermal = _run(["pmset", "-g", "therm"]).lower()
    thermal_state = "nominal" if "no thermal warning" in thermal and "no performance warning" in thermal else "throttled"
    power = _run(["pmset", "-g"])
    match = re.search(r"^\s*lowpowermode\s+(\d+)\s*$", power, re.MULTILINE)
    _require(match is not None, "pmset did not report lowpowermode")
    return thermal_state, "low-power" if int(match.group(1)) else "normal"


def _cuda_memory(group_pids: set[int]) -> tuple[int, int]:
    used = 0
    apps = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in apps.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit() and int(fields[0]) in group_pids:
            used += int(fields[1]) * 1024**2
    free_rows = _run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    ).splitlines()
    free = sum(int(row.strip()) * 1024**2 for row in free_rows if row.strip().isdigit())
    _require(used > 0 and free >= 0, "nvidia-smi did not bind device memory to the server process group")
    return used, free


def _json_path(value: Any, selector: str) -> Any:
    current = value
    for part in selector.split("."):
        _require(isinstance(current, dict) and part in current, f"active JSON selector {selector!r} is absent")
        current = current[part]
    return current


def _active_requests(url: str, probe_format: str, selector: str) -> int:
    with urllib.request.urlopen(url, timeout=2.0) as response:
        _require(response.status == 200, f"active probe returned HTTP {response.status}")
        payload = response.read().decode("utf-8")
    if probe_format == "json":
        value = _json_path(json.loads(payload), selector)
        return _nonnegative_int(value, "active probe value")
    values: list[float] = []
    metric_re = re.compile(rf"^{re.escape(selector)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)(?:\s+\d+)?$")
    for line in payload.splitlines():
        match = metric_re.match(line.strip())
        if match:
            values.append(float(match.group(1)))
    _require(values and all(math.isfinite(value) and value >= 0 for value in values), "active Prometheus metric is absent or invalid")
    total = sum(values)
    _require(total.is_integer(), "active Prometheus metric must be integral")
    return int(total)


def _scan_log_increment(
    path: Path,
    offset: int,
    oom_count: int,
    admission_error_count: int,
) -> tuple[int, int, int]:
    if not path.exists():
        return offset, oom_count, admission_error_count
    size = path.stat().st_size
    if size < offset:
        offset = 0
        oom_count = 0
        admission_error_count = 0
    with path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
        offset = handle.tell()
    text = chunk.decode("utf-8", errors="replace")
    return (
        offset,
        oom_count + len(OOM_RE.findall(text)),
        admission_error_count + len(ADMISSION_ERROR_RE.findall(text)),
    )


def _append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def probe_endpoint(args: argparse.Namespace) -> None:
    body_out = args.probe_body_out.resolve()
    receipt_out = args.probe_receipt_out.resolve()
    _require(not body_out.exists() and not receipt_out.exists(), "refusing to overwrite HTTP probe evidence")
    body_out.parent.mkdir(parents=True, exist_ok=True)
    receipt_out.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    with urllib.request.urlopen(args.probe_url, timeout=args.probe_timeout_sec) as response:
        status = response.status
        body = response.read()
    finished = datetime.now(timezone.utc)
    _require(status == 200, f"HTTP probe returned {status}")
    _require(bool(body), "HTTP probe returned an empty body")
    body_out.write_bytes(body)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "collector_path": COLLECTOR_RELATIVE_PATH,
        "collector_sha256": collector_sha256(),
        "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "finished_at": finished.isoformat().replace("+00:00", "Z"),
        "duration_sec": (finished - started).total_seconds(),
        "returncode": 0,
        "url": args.probe_url,
        "http_status": status,
        "body_origin_path": str(args.probe_body_out),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "body_size_bytes": len(body),
    }
    receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _require(not out.exists(), f"refusing to overwrite resource artifact: {out}")
    _require(os.getpgid(args.pid) == args.pgid, "server pid is not in the declared process group")
    process_start_marker, process_start_source = _process_start_identity(args.pid)
    stop = False

    def stop_handler(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    probe_path = "" if args.active_probe_format == "process" else (
        args.active_path if args.active_path.startswith("/") else f"/{args.active_path}"
    )
    probe_url = "" if args.active_probe_format == "process" else f"{args.base_url.rstrip('/')}{probe_path}"
    started_monotonic = time.monotonic()
    with out.open("x", encoding="utf-8") as handle:
        _append_jsonl(
            handle,
            {
                "record_type": "header",
                "schema_version": SCHEMA_VERSION,
                "collector_path": COLLECTOR_RELATIVE_PATH,
                "collector_sha256": collector_sha256(),
                "session_id": args.session_id,
                "cell_id": args.cell_id,
                "backend": args.backend,
                "hardware_id": args.hardware_id,
                "pid": args.pid,
                "pgid": args.pgid,
                "process_start_marker": process_start_marker,
                "process_start_source": process_start_source,
                "base_url": args.base_url,
                "started_at": _now_iso(),
                "interval_ms": args.interval_ms,
                "runtime_log_path": str(args.runtime_log),
                "active_probe": {
                    "format": args.active_probe_format,
                    "path": probe_path,
                    "url": probe_url,
                    "selector": args.active_selector,
                    "semantics": args.active_semantics,
                },
            },
        )
        sequence = 0
        exit_reason = "duration"
        log_offset = 0
        oom_count = 0
        admission_errors = 0
        while True:
            if stop:
                exit_reason = "signal"
                break
            if args.stop_file.exists():
                exit_reason = "stop-file"
                break
            if time.monotonic() - started_monotonic >= args.max_duration_sec:
                break
            group_rows = _process_group_rows(args.pgid)
            group_pids = {row[0] for row in group_rows}
            process_alive = args.pid in group_pids
            _require(process_alive, "server process exited while resource sampler was active")
            rss = sum(row[1] for row in group_rows)
            if args.backend == "cuda":
                memory_used, headroom = _cuda_memory(group_pids)
                _, swap = _linux_memory()
                extra = {"device_memory_bytes": memory_used}
            else:
                _require(platform.system() == "Darwin", "Metal resource sampling requires macOS")
                headroom, swap = _mac_memory()
                memory_used = rss
                thermal, power = _mac_thermal_power()
                extra = {"thermal_state": thermal, "power_mode": power}
            active = 1 if args.active_probe_format == "process" else _active_requests(
                probe_url, args.active_probe_format, args.active_selector
            )
            log_offset, oom_count, admission_errors = _scan_log_increment(
                args.runtime_log,
                log_offset,
                oom_count,
                admission_errors,
            )
            _append_jsonl(
                handle,
                {
                    "record_type": "sample",
                    "sequence": sequence,
                    "sampled_at": _now_iso(),
                    "pid": args.pid,
                    "pgid": args.pgid,
                    "process_start_marker": process_start_marker,
                    "process_alive": process_alive,
                    "process_rss_bytes": rss,
                    "memory_used_bytes": memory_used,
                    "physical_headroom_bytes": headroom,
                    "swap_used_bytes": swap,
                    "active_requests": active,
                    "oom_count": oom_count,
                    "admission_error_count": admission_errors,
                    **extra,
                },
            )
            sequence += 1
            time.sleep(args.interval_ms / 1000.0)
        _append_jsonl(
            handle,
            {
                "record_type": "footer",
                "finished_at": _now_iso(),
                "sample_count": sequence,
                "exit_reason": exit_reason,
            },
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--probe-url")
    parser.add_argument("--probe-body-out", type=Path)
    parser.add_argument("--probe-receipt-out", type=Path)
    parser.add_argument("--probe-timeout-sec", type=float, default=10.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--pgid", type=int)
    parser.add_argument("--session-id")
    parser.add_argument("--cell-id")
    parser.add_argument("--backend", choices=("cuda", "metal"))
    parser.add_argument("--hardware-id")
    parser.add_argument("--base-url")
    parser.add_argument("--active-probe-format", choices=("json", "prometheus", "process"))
    parser.add_argument("--active-path")
    parser.add_argument("--active-selector")
    parser.add_argument(
        "--active-semantics",
        choices=("scheduler-active-high-water", "process-alive"),
    )
    parser.add_argument("--runtime-log", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--max-duration-sec", type=float, default=7200.0)
    return parser


def self_test() -> int:
    _require(_json_path({"engine": {"active": 3}}, "engine.active") == 3, "JSON path self-test failed")
    _require(bool(OOM_RE.search("CUDA out of memory")), "OOM classifier self-test failed")
    _require(bool(ADMISSION_ERROR_RE.search("KV admission failed")), "admission classifier self-test failed")
    marker, source = _process_start_identity(os.getpid())
    _require(process_marker_from_source(os.getpid(), source) == marker, "process start identity self-test failed")
    require_active_coverage([0, 1, 0], process_probe=False)
    require_active_coverage([1, 1, 1], process_probe=True)
    try:
        require_active_coverage([0, 0, 0], process_probe=False)
        raise ResourceEvidenceError("zero-active HTTP coverage unexpectedly passed")
    except ResourceEvidenceError as exc:
        _require("never observed active work" in str(exc), "zero-active HTTP self-test failed unexpectedly")
    print(SELFTEST_PASS_LINE)
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    if args.self_test:
        return self_test()
    if args.probe_url is not None:
        _require(args.probe_body_out is not None, "--probe-body-out is required with --probe-url")
        _require(args.probe_receipt_out is not None, "--probe-receipt-out is required with --probe-url")
        _require(
            math.isfinite(args.probe_timeout_sec) and args.probe_timeout_sec > 0,
            "--probe-timeout-sec must be positive",
        )
        probe_endpoint(args)
        print(f"{PASS_PREFIX}: {args.probe_receipt_out}")
        return 0
    required = (
        "out",
        "pid",
        "pgid",
        "session_id",
        "cell_id",
        "backend",
        "hardware_id",
        "base_url",
        "active_probe_format",
        "active_selector",
        "active_semantics",
        "runtime_log",
        "stop_file",
    )
    missing = [field for field in required if getattr(args, field) is None]
    if missing:
        raise ResourceEvidenceError(f"missing required collector arguments: {missing}")
    expected_semantics = "process-alive" if args.active_probe_format == "process" else "scheduler-active-high-water"
    _require(args.active_semantics == expected_semantics, "active probe format/semantics mismatch")
    if args.active_probe_format != "process":
        _require(bool(args.active_path), "--active-path is required for HTTP active probes")
    _require(50 <= args.interval_ms <= 1000, "--interval-ms must be 50..1000")
    _require(math.isfinite(args.max_duration_sec) and args.max_duration_sec > 0, "--max-duration-sec must be positive")
    collect(args)
    print(f"{PASS_PREFIX}: {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ResourceEvidenceError, subprocess.CalledProcessError) as exc:
        print(f"resource sampler failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
