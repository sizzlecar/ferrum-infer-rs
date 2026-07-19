#!/usr/bin/env python3
"""Collect the canonical Qwen3.5-4B CUDA S1 basic-slice raw artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_PATH = Path(__file__).resolve()
COLLECTOR_RELATIVE_PATH = COLLECTOR_PATH.relative_to(REPO_ROOT).as_posix()
SCHEMA_VERSION = 2
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT S1 CUDA BASIC COLLECTOR SELFTEST PASS"
COLLECTED_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA BASIC COLLECTED"
SLOT_ORDER = (
    "off1",
    "basic1",
    "basic2",
    "off2",
    "basic3",
    "off3",
    "off4",
    "basic4",
)
FIRST_HALF_SLOT_ORDER = SLOT_ORDER[:4]
GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "pstate",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
)
BUILD_ARGV = (
    "cargo",
    "build",
    "--release",
    "-p",
    "ferrum-cli",
    "--bin",
    "ferrum",
    "--features",
    "cuda,vllm-moe-marlin,vllm-paged-attn-v2",
)
PROFILE_REQUESTS_PER_REPEAT = 4
PROFILE_REPEAT_COUNT = 3
PROFILE_WARMUP_REQUESTS = 1
PROFILE_MAX_OVERHEAD = 0.02
PROFILE_MAX_CV = 0.05
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CollectionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectionError(message)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def file_sha256(path: Path) -> str:
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
        stderr=subprocess.PIPE,
        check=False,
    )
    require(result.returncode == 0, f"command failed {argv!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def product_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("FERRUM_")
    }
    require(
        not any(key.startswith("FERRUM_") for key in environment),
        "hidden Ferrum environment survived sanitization",
    )
    return environment


def write_command(path: Path, argv: list[str], env_prefix: dict[str, str] | None = None) -> None:
    tokens = [f"{key}={value}" for key, value in sorted((env_prefix or {}).items())]
    tokens.extend(argv)
    write_text(path, shlex.join(tokens) + "\n")


def run_command(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    directory: Path,
    name: str,
    stdout_name: str,
    stderr_name: str,
    timeout_seconds: float,
) -> int:
    prefix = "" if name == "command" else f"{name}."
    write_command(directory / f"{prefix}command", argv)
    write_text(directory / f"{prefix}started", now_iso() + "\n")
    with (directory / stdout_name).open("w", encoding="utf-8") as stdout, (
        directory / stderr_name
    ).open("w", encoding="utf-8") as stderr:
        try:
            result = subprocess.run(
                argv,
                cwd=cwd,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=timeout_seconds,
                check=False,
                start_new_session=True,
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
    write_text(directory / f"{prefix}exit", f"{returncode}\n")
    write_text(directory / f"{prefix}finished", now_iso() + "\n")
    require(returncode == 0, f"{name} failed with exit {returncode}: {directory}")
    return returncode


class Server:
    def __init__(
        self,
        argv: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
        directory: Path,
        port: int,
        log_prefix: str = "server.",
        receipt_prefix: str = "server.",
    ) -> None:
        self.argv = argv
        self.cwd = cwd
        self.environment = environment
        self.directory = directory
        self.port = port
        self.log_prefix = log_prefix
        self.receipt_prefix = receipt_prefix
        self.process: subprocess.Popen[str] | None = None
        self.stdout: TextIO | None = None
        self.stderr: TextIO | None = None

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        write_command(self.directory / f"{self.receipt_prefix}command", self.argv)
        write_text(self.directory / f"{self.receipt_prefix}started", now_iso() + "\n")
        self.stdout = (self.directory / f"{self.log_prefix}stdout.log").open(
            "w", encoding="utf-8"
        )
        self.stderr = (self.directory / f"{self.log_prefix}stderr.log").open(
            "w", encoding="utf-8"
        )
        self.process = subprocess.Popen(
            self.argv,
            cwd=self.cwd,
            env=self.environment,
            stdout=self.stdout,
            stderr=self.stderr,
            text=True,
            start_new_session=True,
        )
        write_text(self.directory / "server.pid", f"{self.process.pid}\n")
        deadline = time.monotonic() + 180.0
        last_error = "server did not answer"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise CollectionError(
                    f"server exited before readiness with {self.process.returncode}: {self.directory}"
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=2.0
                ) as response:
                    body = response.read()
                health = json.loads(body)
                if response.status == 200 and health.get("status") == "healthy":
                    write_json(self.directory / "health.json", health)
                    return
                last_error = f"unexpected health response: {health!r}"
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                last_error = type(error).__name__
            time.sleep(1.0)
        raise CollectionError(f"server readiness timeout ({last_error}): {self.directory}")

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5.0)
        returncode = process.returncode
        write_text(self.directory / "server.exit", f"{returncode}\n")
        write_text(self.directory / "server.finished", now_iso() + "\n")
        if self.stdout is not None:
            self.stdout.close()
        if self.stderr is not None:
            self.stderr.close()
        self.process = None
        require(returncode == 0, f"server failed with exit {returncode}: {self.directory}")


def parse_number(value: str, label: str, *, integer: bool = False) -> int | float:
    stripped = value.strip()
    require(stripped and stripped.upper() != "N/A", f"{label} is unavailable")
    try:
        return int(stripped) if integer else float(stripped)
    except ValueError as error:
        raise CollectionError(f"{label} is not numeric: {value!r}") from error


def parse_gpu_row(text: str) -> dict[str, Any]:
    rows = list(csv.reader(text.strip().splitlines()))
    require(len(rows) == 1, "telemetry requires exactly one visible GPU")
    values = [value.strip() for value in rows[0]]
    require(len(values) == len(GPU_QUERY_FIELDS), "unexpected nvidia-smi telemetry row")
    return {
        "index": parse_number(values[0], "GPU index", integer=True),
        "uuid": values[1],
        "pstate": values[2],
        "graphics_clock_mhz": parse_number(values[3], "graphics clock", integer=True),
        "sm_clock_mhz": parse_number(values[4], "SM clock", integer=True),
        "memory_clock_mhz": parse_number(values[5], "memory clock", integer=True),
        "power_draw_w": parse_number(values[6], "power draw"),
        "power_limit_w": parse_number(values[7], "power limit"),
        "temperature_c": parse_number(values[8], "GPU temperature", integer=True),
        "gpu_utilization_percent": parse_number(values[9], "GPU utilization", integer=True),
        "memory_utilization_percent": parse_number(values[10], "memory utilization", integer=True),
        "memory_used_mib": parse_number(values[11], "GPU memory used", integer=True),
        "memory_total_mib": parse_number(values[12], "GPU memory total", integer=True),
    }


def host_state() -> dict[str, Any]:
    load_parts = Path("/proc/loadavg").read_text(encoding="utf-8").split()
    stat_parts = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    require(stat_parts[0] == "cpu" and len(stat_parts) >= 6, "invalid /proc/stat CPU row")
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, raw = line.partition(":")
        parts = raw.split()
        if parts and parts[0].isdigit():
            memory[key] = int(parts[0]) * 1024
    require("MemAvailable" in memory and "SwapTotal" in memory and "SwapFree" in memory, "invalid /proc/meminfo")
    return {
        "load_1": float(load_parts[0]),
        "load_5": float(load_parts[1]),
        "load_15": float(load_parts[2]),
        "cpu_count": os.cpu_count() or 1,
        "cpu_ticks": {
            "user": int(stat_parts[1]),
            "nice": int(stat_parts[2]),
            "system": int(stat_parts[3]),
            "idle": int(stat_parts[4]),
            "iowait": int(stat_parts[5]),
        },
        "mem_available_bytes": memory["MemAvailable"],
        "swap_used_bytes": memory["SwapTotal"] - memory["SwapFree"],
    }


def telemetry_sample(slot: str, mode: str, phase: str) -> dict[str, Any]:
    query = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(query, text=True, capture_output=True, check=False)
    require(result.returncode == 0, f"nvidia-smi telemetry failed: {result.stderr.strip()}")
    wall_time_ns = time.time_ns()
    return {
        "schema_version": 1,
        "slot": slot,
        "mode": mode,
        "phase": phase,
        "sampled_at": now_iso(),
        "wall_time_ns": wall_time_ns,
        "monotonic_ns": time.monotonic_ns(),
        "gpu": parse_gpu_row(result.stdout),
        "host": host_state(),
    }


class Telemetry:
    def __init__(self, directory: Path, slot: str, mode: str, interval_ms: int) -> None:
        self.directory = directory
        self.slot = slot
        self.mode = mode
        self.interval_ms = interval_ms
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def prepare(self) -> None:
        write_json(
            self.directory / "telemetry.command.json",
            {
                "schema_version": 1,
                "collector_path": COLLECTOR_RELATIVE_PATH,
                "collector_sha256": file_sha256(COLLECTOR_PATH),
                "interval_ms": self.interval_ms,
                "gpu_query_fields": list(GPU_QUERY_FIELDS),
                "host_sources": ["/proc/loadavg", "/proc/stat", "/proc/meminfo"],
                "mutates_gpu_state": False,
            },
        )
        write_json(
            self.directory / "telemetry.before.json",
            telemetry_sample(self.slot, self.mode, "before"),
        )

    def start(self) -> None:
        samples = self.directory / "telemetry.samples.jsonl"

        def collect() -> None:
            try:
                while not self.stop_event.is_set():
                    append_jsonl(samples, telemetry_sample(self.slot, self.mode, "during"))
                    self.stop_event.wait(self.interval_ms / 1000.0)
            except BaseException as error:  # Captured and re-raised on the owner thread.
                self.error = error
                self.stop_event.set()

        self.thread = threading.Thread(target=collect, name=f"telemetry-{self.slot}")
        self.thread.start()

    def stop_during(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10.0)
            require(not self.thread.is_alive(), f"telemetry thread did not stop: {self.slot}")
        if self.error is not None:
            raise CollectionError(f"telemetry failed for {self.slot}: {self.error}")

    def finish(self) -> None:
        self.stop_during()
        samples_path = self.directory / "telemetry.samples.jsonl"
        samples = [
            json.loads(line)
            for line in samples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        require(len(samples) >= 3, f"telemetry captured fewer than three during samples: {self.slot}")
        after = telemetry_sample(self.slot, self.mode, "after")
        write_json(self.directory / "telemetry.after.json", after)
        uuids = {row["gpu"]["uuid"] for row in samples}
        uuids.add(after["gpu"]["uuid"])
        before = json.loads((self.directory / "telemetry.before.json").read_text())
        uuids.add(before["gpu"]["uuid"])
        require(len(uuids) == 1, f"GPU UUID changed during slot {self.slot}")
        write_json(
            self.directory / "telemetry.summary.json",
            {
                "schema_version": 1,
                "slot": self.slot,
                "mode": self.mode,
                "gpu_uuid": next(iter(uuids)),
                "during_sample_count": len(samples),
                "before_sampled_at": before["sampled_at"],
                "after_sampled_at": after["sampled_at"],
            },
        )
        write_text(self.directory / "telemetry.exit", "0\n")


def fetch_json(port: int, path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10.0) as response:
        value = json.loads(response.read())
    require(response.status == 200 and isinstance(value, dict), f"GET {path} failed")
    return value


def collect_stream(port: int, model_id: str, directory: Path) -> None:
    request = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Answer in one word.",
            }
        ],
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    write_json(directory / "request.json", request)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300.0)
    body = json.dumps(request, separators=(",", ":"))
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    payload = response.read().decode("utf-8", errors="strict")
    connection.close()
    write_text(directory / "http.status", f"{response.status}\n")
    write_text(directory / "stream.sse", payload)
    data_rows: list[str] = []
    for line in payload.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data != "[DONE]":
            parsed = json.loads(data)
            data_rows.append(json.dumps(parsed, separators=(",", ":")))
    write_text(directory / "stream.data.jsonl", "\n".join(data_rows) + "\n")
    require(response.status == 200, f"stream request returned HTTP {response.status}")


def scalar_stats(values: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values)
    return {
        "values": values,
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "sample_stddev": deviation,
        "cv": deviation / mean,
    }


def first_half_receipt(performance: Path) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    for slot in FIRST_HALF_SLOT_ORDER:
        report = json.loads((performance / slot / "bench.json").read_text(encoding="utf-8"))
        values[slot] = [float(row["output_throughput_tps"]) for row in report["repeat_metrics"]]
    off = scalar_stats([value for slot in ("off1", "off2") for value in values[slot]])
    basic = scalar_stats([value for slot in ("basic1", "basic2") for value in values[slot]])
    mean_overhead = (off["mean"] - basic["mean"]) / off["mean"]
    median_overhead = (off["median"] - basic["median"]) / off["median"]
    return {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s1_profile_overhead_first_half",
        "slot_order": list(FIRST_HALF_SLOT_ORDER),
        "off": off,
        "basic": basic,
        "mean_overhead": mean_overhead,
        "median_overhead": median_overhead,
        "max_overhead": PROFILE_MAX_OVERHEAD,
        "max_cv": PROFILE_MAX_CV,
        "status": (
            "pass"
            if off["cv"] <= PROFILE_MAX_CV
            and basic["cv"] <= PROFILE_MAX_CV
            and mean_overhead <= PROFILE_MAX_OVERHEAD
            and median_overhead <= PROFILE_MAX_OVERHEAD
            else "reject"
        ),
    }


def collect_correctness(
    repo: Path,
    raw: Path,
    model: Path,
    model_id: str,
    environment: dict[str, str],
    batched_graph: bool = False,
) -> None:
    binary = "target/release/ferrum"
    run = raw / "run"
    run.mkdir(parents=True)
    run_argv = [
        binary,
        "run",
        str(model),
        "--backend",
        "cuda",
        "--prompt",
        "Respond with only the city name: What is the capital of France?",
        "--max-tokens",
        "16",
        "--output-format",
        "jsonl",
        "--profile-jsonl",
        str(run / "profile.jsonl"),
        "--profile-detail",
        "basic",
        "--profile-sample-rate",
        "1.0",
        "--scheduler-trace-jsonl",
        str(run / "scheduler-trace.jsonl"),
    ]
    if batched_graph:
        run_argv.append("--batched-graph")
    run_command(
        run_argv,
        cwd=repo,
        environment=environment,
        directory=run,
        name="command",
        stdout_name="stdout.jsonl",
        stderr_name="stderr.log",
        timeout_seconds=300.0,
    )

    serve = raw / "serve"
    port = 18080
    serve_argv = [
        binary,
        "serve",
        str(model),
        "--backend",
        "cuda",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--profile-detail",
        "basic",
        "--profile-sample-rate",
        "1.0",
        "--profile-jsonl",
        str(serve / "profile.jsonl"),
        "--scheduler-trace-jsonl",
        str(serve / "scheduler-trace.jsonl"),
    ]
    if batched_graph:
        serve_argv.append("--batched-graph")
    server = Server(
        serve_argv,
        cwd=repo,
        environment=environment,
        directory=serve,
        port=port,
        log_prefix="",
        receipt_prefix="",
    )
    try:
        server.start()
        write_json(serve / "models.json", fetch_json(port, "/v1/models"))
        collect_stream(port, model_id, serve)
        bench_argv = [
            binary,
            "bench-serve",
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--model",
            model_id,
            "--tokenizer",
            str(model),
            "--target-backend",
            "cuda",
            "--concurrency",
            "1",
            "--dataset",
            "random",
            "--random-input-len",
            "128",
            "--random-output-len",
            "16",
            "--num-prompts",
            "4",
            "--warmup-requests",
            "1",
            "--n-repeats",
            "1",
            "--fail-on-error",
            "--seed",
            "9271",
            "--output",
            "json",
            "--out",
            str(serve / "bench-smoke.json"),
        ]
        run_command(
            bench_argv,
            cwd=repo,
            environment=environment,
            directory=serve,
            name="bench",
            stdout_name="bench.stdout.log",
            stderr_name="bench.stderr.log",
            timeout_seconds=600.0,
        )
        write_json(serve / "health.after-bench.json", fetch_json(port, "/health"))
    finally:
        server.stop()
    write_json(serve / "server-stop.json", {"stopped": True, "signal": "SIGINT", "exit_code": 0})


def collect_profile_slots(
    repo: Path,
    raw: Path,
    model: Path,
    model_id: str,
    environment: dict[str, str],
    port_base: int,
    telemetry_interval_ms: int,
    batched_graph: bool = False,
) -> None:
    binary = "target/release/ferrum"
    performance = raw / "profile-overhead"
    performance.mkdir(parents=True)
    write_text(performance / "slot-order", " ".join(SLOT_ORDER) + "\n")
    write_text(performance / "started", now_iso() + "\n")
    for index, slot in enumerate(SLOT_ORDER):
        mode = "basic" if slot.startswith("basic") else "off"
        port = port_base + index
        directory = performance / slot
        server_argv = [
            binary,
            "serve",
            str(model),
            "--backend",
            "cuda",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--profile-detail",
            mode,
        ]
        if batched_graph:
            server_argv.append("--batched-graph")
        if mode == "basic":
            server_argv.extend(
                [
                    "--profile-sample-rate",
                    "1.0",
                    "--profile-jsonl",
                    str(directory / "profile.jsonl"),
                    "--scheduler-trace-jsonl",
                    str(directory / "scheduler-trace.jsonl"),
                ]
            )
        server = Server(
            server_argv,
            cwd=repo,
            environment=environment,
            directory=directory,
            port=port,
        )
        telemetry = Telemetry(directory, slot, mode, telemetry_interval_ms)
        telemetry_started = False
        try:
            server.start()
            telemetry.prepare()
            bench_argv = [
                binary,
                "bench-serve",
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--model",
                model_id,
                "--tokenizer",
                str(model),
                "--target-backend",
                "cuda",
                "--concurrency",
                "1",
                "--dataset",
                "random",
                "--random-input-len",
                "128",
                "--random-output-len",
                "64",
                "--num-prompts",
                str(PROFILE_REQUESTS_PER_REPEAT),
                "--warmup-requests",
                str(PROFILE_WARMUP_REQUESTS),
                "--n-repeats",
                str(PROFILE_REPEAT_COUNT),
                "--require-ci",
                "--fail-on-error",
                "--seed",
                "9271",
                "--output",
                "json",
                "--out",
                str(directory / "bench.json"),
            ]
            write_command(directory / "bench.command", bench_argv)
            write_text(directory / "bench.started", now_iso() + "\n")
            telemetry.start()
            telemetry_started = True
            with (directory / "bench.stdout.log").open("w", encoding="utf-8") as stdout, (
                directory / "bench.stderr.log"
            ).open("w", encoding="utf-8") as stderr:
                try:
                    result = subprocess.run(
                        bench_argv,
                        cwd=repo,
                        env=environment,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        timeout=900.0,
                        check=False,
                        start_new_session=True,
                    )
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    returncode = 124
            telemetry.stop_during()
            telemetry_started = False
            write_text(directory / "bench.exit", f"{returncode}\n")
            write_text(directory / "bench.finished", now_iso() + "\n")
            telemetry.finish()
            write_json(directory / "health.after-bench.json", fetch_json(port, "/health"))
            require(returncode == 0, f"profile benchmark failed with exit {returncode}: {slot}")
        finally:
            if telemetry_started:
                telemetry.stop_during()
            server.stop()
        if mode == "basic":
            size = (directory / "scheduler-trace.jsonl").stat().st_size
            write_text(directory / "scheduler-trace.size", f"{size}\n")
        if index == 3:
            write_json(performance / "overhead.first-half.json", first_half_receipt(performance))
        print(f"S1 CUDA BASIC SLOT COLLECTED: {slot}", flush=True)
    write_text(performance / "finished", now_iso() + "\n")


def collect(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    model = args.model.resolve()
    out = args.out.resolve()
    require(repo.is_dir() and (repo / "Cargo.toml").is_file(), "invalid repository root")
    require(model.is_dir(), f"model snapshot does not exist: {model}")
    require("models--Qwen--Qwen3.5-4B/snapshots/" in str(model), "collector requires Qwen3.5-4B HF snapshot")
    require(not out.exists(), f"refusing to overwrite artifact directory: {out}")
    require(1024 <= args.port_base <= 65528, "--port-base leaves the valid port range")
    require(250 <= args.telemetry_interval_ms <= 1000, "telemetry interval must be 250..1000 ms")
    source_sha = command_output(["git", "rev-parse", "HEAD"], repo)
    dirty = command_output(["git", "status", "--porcelain"], repo)
    require(not dirty, "collector requires a clean source worktree")
    out.mkdir(parents=True)
    environment = product_environment()
    write_text(out / "git.sha", source_sha + "\n")
    write_text(out / "git.status", "")
    write_json(
        out / "runtime-env.json",
        {
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            "ld_library_path": environment.get("LD_LIBRARY_PATH"),
            "hidden_ferrum_environment_overrides": [],
        },
    )
    write_json(
        out / "collection.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_s1_cuda_basic_raw_collection",
            "source_git_sha": source_sha,
            "collector": {
                "path": COLLECTOR_RELATIVE_PATH,
                "sha256": file_sha256(COLLECTOR_PATH),
            },
            "model_snapshot_path": str(model),
            "protocol": {
                "slot_order": list(SLOT_ORDER),
                "comparison": "ABBA-BAAB",
                "concurrency": 1,
                "random_input_len": 128,
                "random_output_len": 64,
                "prompts_per_repeat": PROFILE_REQUESTS_PER_REPEAT,
                "warmup_requests": PROFILE_WARMUP_REQUESTS,
                "repeats_per_slot": PROFILE_REPEAT_COUNT,
                "seed": 9271,
                "require_ci": True,
                "fail_on_error": True,
                "batched_graph": args.batched_graph,
                "telemetry_interval_ms": args.telemetry_interval_ms,
                "profile_health_after_bench": True,
            },
        },
    )
    write_text(out / "started", now_iso() + "\n")

    build_environment = dict(environment)
    build_environment["CARGO_BUILD_JOBS"] = "8"
    write_command(out / "build.command", list(BUILD_ARGV), {"CARGO_BUILD_JOBS": "8"})
    write_text(out / "build.started", now_iso() + "\n")
    with (out / "build.log").open("w", encoding="utf-8") as build_log:
        build = subprocess.run(
            list(BUILD_ARGV),
            cwd=repo,
            env=build_environment,
            stdout=build_log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    write_text(out / "build.exit", f"{build.returncode}\n")
    write_text(out / "build.finished", now_iso() + "\n")
    require(build.returncode == 0, f"CUDA release build failed: {out / 'build.log'}")
    binary = repo / "target/release/ferrum"
    require(binary.is_file(), "CUDA release binary is missing after build")
    write_text(out / "binary.sha256", f"{file_sha256(binary)}  {binary}\n")
    hardware_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    require(hardware_query.returncode == 0, "nvidia-smi hardware identity query failed")
    require(len(hardware_query.stdout.strip().splitlines()) == 1, "collector requires exactly one RTX 4090")
    require("RTX 4090" in hardware_query.stdout, "collector requires one RTX 4090")
    write_text(out / "hardware.csv", hardware_query.stdout)

    model_id = model.name
    collect_correctness(
        repo,
        out,
        model,
        model_id,
        environment,
        batched_graph=args.batched_graph,
    )
    correctness = subprocess.run(
        [
            sys.executable,
            "scripts/release/runtime_vnext_s1_cuda_checkpoint.py",
            str(out),
            "--expected-git-sha",
            source_sha,
        ],
        cwd=repo,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    write_command(
        out / "correctness-gate.command",
        [
            sys.executable,
            "scripts/release/runtime_vnext_s1_cuda_checkpoint.py",
            str(out),
            "--expected-git-sha",
            source_sha,
        ],
    )
    write_text(out / "correctness-gate.log", correctness.stdout)
    write_text(out / "correctness-gate.exit", f"{correctness.returncode}\n")
    require(
        correctness.returncode == 0
        and f"FERRUM RUNTIME VNEXT S1 CUDA TRACE CHECKPOINT PASS: {out}" in correctness.stdout,
        f"correctness gate failed before performance collection: {out / 'correctness-gate.log'}",
    )
    collect_profile_slots(
        repo,
        out,
        model,
        model_id,
        environment,
        args.port_base,
        args.telemetry_interval_ms,
        batched_graph=args.batched_graph,
    )
    write_text(out / "finished", now_iso() + "\n")
    print(f"{COLLECTED_PREFIX}: {out}")
    return 0


def self_test() -> int:
    sample = (
        "0, GPU-1234, P2, 2520, 2520, 10501, 241.50, 450.00, 61, 97, 42, 8123, 24564\n"
    )
    parsed = parse_gpu_row(sample)
    require(parsed["uuid"] == "GPU-1234", "GPU row parser lost UUID")
    require(parsed["sm_clock_mhz"] == 2520, "GPU row parser lost SM clock")
    require(parsed["power_draw_w"] == 241.5, "GPU row parser lost power")
    try:
        parse_gpu_row(sample.replace("241.50", "N/A"))
    except CollectionError as error:
        require("unavailable" in str(error), "N/A GPU mutation failed for the wrong reason")
    else:
        raise CollectionError("N/A GPU telemetry unexpectedly passed")
    original = os.environ.get("FERRUM_SELFTEST_SECRET")
    os.environ["FERRUM_SELFTEST_SECRET"] = "forbidden"
    try:
        require("FERRUM_SELFTEST_SECRET" not in product_environment(), "hidden Ferrum env leaked")
    finally:
        if original is None:
            os.environ.pop("FERRUM_SELFTEST_SECRET", None)
        else:
            os.environ["FERRUM_SELFTEST_SECRET"] = original
    require(SLOT_ORDER == ("off1", "basic1", "basic2", "off2", "basic3", "off3", "off4", "basic4"), "slot order drift")
    require(SHA256_RE.fullmatch(file_sha256(COLLECTOR_PATH)) is not None, "collector SHA is malformed")
    print(SELFTEST_PASS_LINE)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    collect_parser.add_argument("--model", type=Path, required=True)
    collect_parser.add_argument("--out", type=Path, required=True)
    collect_parser.add_argument("--port-base", type=int, default=18101)
    collect_parser.add_argument("--telemetry-interval-ms", type=int, default=500)
    collect_parser.add_argument(
        "--batched-graph",
        action="store_true",
        help="exercise the user-visible typed CUDA graph preset",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    require(args.command == "collect", "collect subcommand or --self-test is required")
    return collect(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CollectionError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"FERRUM RUNTIME VNEXT S1 CUDA BASIC COLLECTOR FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
