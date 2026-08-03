#!/usr/bin/env python3
"""Collect the Qwen3.5-4B CUDA S2 latency/first-failure artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, TextIO

import runtime_vnext_s2_latency_failure_checkpoint as checkpoint


COLLECTED_PREFIX = "FERRUM RUNTIME VNEXT S2 LATENCY FIRST FAILURE COLLECTED"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT S2 LATENCY FIRST FAILURE COLLECTOR SELFTEST PASS"
RUN_TIMEOUT_SECONDS = 600.0
SERVER_READY_TIMEOUT_SECONDS = 300.0
BENCH_TIMEOUT_SECONDS = 900.0
STOP_TIMEOUT_SECONDS = 45.0
PROMPT = "What is the capital of France? Answer with only the city name."


class CollectionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectionError(message)


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


def product_environment() -> tuple[dict[str, str], list[str]]:
    removed = sorted(key for key in os.environ if key.startswith("FERRUM_"))
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("FERRUM_")
    }
    require(
        not any(key.startswith("FERRUM_") for key in environment),
        "hidden Ferrum environment survived sanitization",
    )
    return environment, removed


def write_command(path: Path, argv: list[str]) -> None:
    checkpoint.write_text(path, shlex.join(argv) + "\n")


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    for sig, timeout in (
        (signal.SIGINT, 20.0),
        (signal.SIGTERM, 10.0),
        (signal.SIGKILL, 5.0),
    ):
        os.killpg(process.pid, sig)
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
    raise CollectionError(f"process group {process.pid} did not terminate")


def run_product(
    argv: list[str],
    *,
    repo: Path,
    environment: dict[str, str],
    directory: Path,
    expect_success: bool,
    timeout_seconds: float,
) -> int:
    directory.mkdir(parents=True, exist_ok=False)
    write_command(directory / "command.txt", argv)
    checkpoint.write_text(directory / "started_at", checkpoint.iso_now() + "\n")
    stdout_handle = (directory / "stdout.log").open("w", encoding="utf-8")
    stderr_handle = (directory / "stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        argv,
        cwd=repo,
        env=environment,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        start_new_session=True,
    )
    try:
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            terminate_process_group(process)
            returncode = 124
            raise CollectionError(
                f"product command exceeded {timeout_seconds:.0f}s: {directory}"
            ) from error
    finally:
        stdout_handle.close()
        stderr_handle.close()
        checkpoint.write_text(directory / "exit_code", f"{process.returncode if process.returncode is not None else 124}\n")
        checkpoint.write_text(directory / "finished_at", checkpoint.iso_now() + "\n")
    require(
        (returncode == 0) is expect_success,
        f"product command exit {returncode} did not match expected success={expect_success}: {directory}",
    )
    return returncode


class Server:
    def __init__(
        self,
        argv: list[str],
        *,
        repo: Path,
        environment: dict[str, str],
        directory: Path,
        port: int,
    ) -> None:
        self.argv = argv
        self.repo = repo
        self.environment = environment
        self.directory = directory
        self.port = port
        self.process: subprocess.Popen[str] | None = None
        self.stdout: TextIO | None = None
        self.stderr: TextIO | None = None

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=False)
        write_command(self.directory / "command.txt", self.argv)
        checkpoint.write_text(self.directory / "started_at", checkpoint.iso_now() + "\n")
        self.stdout = (self.directory / "stdout.log").open("w", encoding="utf-8")
        self.stderr = (self.directory / "stderr.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            self.argv,
            cwd=self.repo,
            env=self.environment,
            stdout=self.stdout,
            stderr=self.stderr,
            text=True,
            start_new_session=True,
        )
        checkpoint.write_text(self.directory / "pid", f"{self.process.pid}\n")
        deadline = time.monotonic() + SERVER_READY_TIMEOUT_SECONDS
        last_error = "health endpoint did not answer"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise CollectionError(
                    f"server exited before readiness with {self.process.returncode}: {self.directory}"
                )
            try:
                health = http_get_json(self.port, "/health", timeout=3.0)
                if health.get("status") == "healthy":
                    checkpoint.write_json(self.directory / "health.json", health)
                    return
                last_error = f"unexpected health response {health!r}"
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                last_error = type(error).__name__
            time.sleep(1.0)
        raise CollectionError(f"server readiness timeout ({last_error}): {self.directory}")

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        terminate_process_group(process)
        checkpoint.write_text(self.directory / "exit_code", f"{process.returncode}\n")
        checkpoint.write_text(self.directory / "finished_at", checkpoint.iso_now() + "\n")
        if self.stdout is not None:
            self.stdout.close()
        if self.stderr is not None:
            self.stderr.close()
        self.process = None
        require(process.returncode == 0, f"server exit was {process.returncode}: {self.directory}")


def http_get_json(port: int, path: str, *, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{path}", timeout=timeout
    ) as response:
        body = response.read().decode("utf-8", errors="strict")
        status = response.status
    value = json.loads(body)
    require(status == 200 and isinstance(value, dict), f"GET {path} failed with HTTP {status}")
    return value


def served_model_id(models: dict[str, Any]) -> str:
    data = models.get("data")
    require(isinstance(data, list) and len(data) == 1, "server must expose exactly one model")
    row = data[0]
    require(isinstance(row, dict), "server model row is not an object")
    model_id = row.get("id")
    require(isinstance(model_id, str) and model_id.strip() == model_id and model_id, "served model id is invalid")
    return model_id


def stream_request(model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def post_stream(
    port: int,
    request: dict[str, Any],
    *,
    directory: Path,
    prefix: str = "",
) -> None:
    import http.client

    request_name = f"{prefix}request.json"
    status_name = f"{prefix}http_status"
    response_name = f"{prefix}response.sse"
    checkpoint.write_json(directory / request_name, request)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=RUN_TIMEOUT_SECONDS)
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
    checkpoint.write_text(directory / status_name, f"{response.status}\n")
    checkpoint.write_text(directory / response_name, payload)


def scenario_observability_args(directory: Path) -> list[str]:
    return [
        "--profile-detail",
        "latency",
        "--profile-sample-rate",
        "1.0",
        "--profile-jsonl",
        str(directory / "profile.jsonl"),
        "--effective-config-json",
        str(directory / "effective-config.json"),
        "--decision-trace-jsonl",
        str(directory / "decision-trace.jsonl"),
        "--scheduler-trace-jsonl",
        str(directory / "scheduler-trace.jsonl"),
        "--request-dump-dir",
        str(directory / "request-dump"),
    ]


def collect_run_scenario(
    *,
    repo: Path,
    binary: Path,
    model: Path,
    environment: dict[str, str],
    root: Path,
    failure: bool,
) -> None:
    name = "run-failure" if failure else "run-success"
    directory = root / name
    argv = [
        str(binary),
        "run",
        str(model),
        "--backend",
        "cuda",
        "--prompt",
        PROMPT,
        "--max-tokens",
        "16",
        "--disable-thinking",
        "--output-format",
        "jsonl",
        *scenario_observability_args(directory),
    ]
    if failure:
        argv.extend(["--vnext-diagnostic-fault", checkpoint.FAULT_VALUE])
    run_product(
        argv,
        repo=repo,
        environment=environment,
        directory=directory,
        expect_success=not failure,
        timeout_seconds=RUN_TIMEOUT_SECONDS,
    )
    print(f"S2 LATENCY SCENARIO COLLECTED: {name}", flush=True)


def collect_serve_scenario(
    *,
    repo: Path,
    binary: Path,
    model: Path,
    environment: dict[str, str],
    root: Path,
    port: int,
    failure: bool,
) -> None:
    name = "serve-failure" if failure else "serve-success"
    directory = root / name
    argv = [
        str(binary),
        "serve",
        str(model),
        "--backend",
        "cuda",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *scenario_observability_args(directory),
    ]
    if failure:
        argv.extend(["--vnext-diagnostic-fault", checkpoint.FAULT_VALUE])
    server = Server(
        argv,
        repo=repo,
        environment=environment,
        directory=directory,
        port=port,
    )
    try:
        server.start()
        models = http_get_json(port, "/v1/models")
        checkpoint.write_json(directory / "models.json", models)
        request = stream_request(served_model_id(models))
        post_stream(port, request, directory=directory)
        parsed = checkpoint.parse_sse(directory / "response.sse")
        if failure:
            require(
                any(isinstance(row.get("error"), dict) for row in parsed["payloads"]),
                "typed serve fault did not reach the HTTP client",
            )
            post_stream(port, request, directory=directory, prefix="recovery.")
            recovery = checkpoint.parse_sse(directory / "recovery.response.sse")
            require(
                not any(isinstance(row.get("error"), dict) for row in recovery["payloads"]),
                "serve recovery request failed",
            )
            require(
                any(isinstance(row.get("usage"), dict) for row in recovery["payloads"]),
                "serve recovery response lacks usage",
            )
        else:
            require(
                any(isinstance(row.get("usage"), dict) for row in parsed["payloads"]),
                "serve success response lacks usage",
            )
        checkpoint.write_json(directory / "health.after.json", http_get_json(port, "/health"))
    finally:
        server.stop()
    print(f"S2 LATENCY SCENARIO COLLECTED: {name}", flush=True)


def parse_hardware_csv(text: str) -> dict[str, Any]:
    rows = list(csv.reader(text.strip().splitlines()))
    require(len(rows) == 1, "collector requires exactly one visible GPU")
    values = [value.strip() for value in rows[0]]
    require(len(values) == 4, "unexpected nvidia-smi hardware row")
    name, uuid, memory, driver = values
    require("RTX 4090" in name, "collector requires one RTX 4090")
    require(uuid.startswith("GPU-"), "nvidia-smi UUID is invalid")
    try:
        memory_total_mib = int(memory)
    except ValueError as error:
        raise CollectionError(f"nvidia-smi memory is not an integer: {memory!r}") from error
    require(memory_total_mib >= 24000, "RTX 4090 memory total is unexpectedly small")
    return {
        "gpu_count": 1,
        "name": name,
        "uuid": uuid,
        "memory_total_mib": memory_total_mib,
        "driver_version": driver,
    }


def hardware_identity(repo: Path, out: Path) -> dict[str, Any]:
    argv = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        argv,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(result.returncode == 0, f"nvidia-smi failed: {result.stderr.strip()}")
    checkpoint.write_text(out / "hardware.command", shlex.join(argv) + "\n")
    checkpoint.write_text(out / "hardware.csv", result.stdout)
    checkpoint.write_text(out / "hardware.stderr.log", result.stderr)
    return parse_hardware_csv(result.stdout)


def model_closure(model: Path) -> dict[str, Any]:
    match = checkpoint.SNAPSHOT_RE.search(str(model))
    require(match is not None, "model must be a Qwen3.5-4B Hugging Face snapshot")
    files: list[dict[str, Any]] = []
    for path in sorted(model.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(model).as_posix()
        size = path.stat().st_size
        require(size > 0, f"model snapshot contains empty file: {relative}")
        print(f"MODEL CLOSURE HASH: {relative} ({size} bytes)", flush=True)
        files.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": checkpoint.file_sha256(path),
            }
        )
    require(files, "model snapshot contains no regular files")
    names = {row["path"] for row in files}
    require("config.json" in names, "model snapshot lacks config.json")
    require("tokenizer_config.json" in names, "model snapshot lacks tokenizer_config.json")
    require(any(name.endswith(".safetensors") for name in names), "model snapshot lacks safetensors weights")
    return {
        "id": "Qwen/Qwen3.5-4B",
        "snapshot_path": str(model),
        "revision": match.group(1),
        "files": files,
        "closure_sha256": checkpoint.canonical_sha256(files),
    }


def bench_argv(binary: Path, model: Path, model_id: str, port: int, out: Path) -> list[str]:
    return [
        str(binary),
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
        "32",
        "--num-prompts",
        "4",
        "--warmup-requests",
        "1",
        "--n-repeats",
        "3",
        "--require-ci",
        "--fail-on-error",
        "--seed",
        "9271",
        "--enable-thinking",
        "false",
        "--output",
        "json",
        "--out",
        str(out),
    ]


def slot_throughput(path: Path) -> float:
    report = checkpoint.read_json(path)
    repeats = report.get("repeat_metrics")
    require(isinstance(repeats, list) and len(repeats) >= 3, f"bench report lacks repeats: {path}")
    values = [row.get("output_throughput_tps") for row in repeats if isinstance(row, dict)]
    require(
        len(values) == len(repeats)
        and all(type(value) in (int, float) and math.isfinite(float(value)) and float(value) > 0 for value in values),
        f"bench report has invalid throughput: {path}",
    )
    return statistics.fmean(float(value) for value in values)


def scalar_stats(values: list[float]) -> dict[str, float | int | list[float]]:
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


def overhead_report(slots: list[dict[str, Any]]) -> dict[str, Any]:
    off_values = [float(row["output_throughput_tps"]) for row in slots if row["mode"] == "off"]
    latency_values = [float(row["output_throughput_tps"]) for row in slots if row["mode"] == "latency"]
    require(len(off_values) == len(latency_values) == 4, "ABBA-BAAB requires four samples per mode")
    off = scalar_stats(off_values)
    latency = scalar_stats(latency_values)
    mean_overhead = (float(off["mean"]) - float(latency["mean"])) / float(off["mean"])
    median_overhead = (float(off["median"]) - float(latency["median"])) / float(off["median"])
    stable = float(off["cv"]) <= 0.05 and float(latency["cv"]) <= 0.05
    target_met = mean_overhead <= 0.05 and median_overhead <= 0.05
    return {
        "schema_version": 1,
        "comparison": "ABBA-BAAB",
        "slot_order": list(checkpoint.OVERHEAD_SLOT_ORDER),
        "slots": slots,
        "off": off,
        "latency": latency,
        "mean_overhead_fraction": mean_overhead,
        "median_overhead_fraction": median_overhead,
        "classification": (
            "stable_target_met"
            if stable and target_met
            else ("target_miss" if stable else "noisy")
        ),
        "blocking": False,
    }


def collect_overhead(
    *,
    repo: Path,
    binary: Path,
    model: Path,
    environment: dict[str, str],
    root: Path,
    port_base: int,
) -> None:
    overhead = root / "profile-overhead"
    overhead.mkdir()
    slots: list[dict[str, Any]] = []
    for index, slot in enumerate(checkpoint.OVERHEAD_SLOT_ORDER):
        mode = "latency" if slot.startswith("latency") else "off"
        port = port_base + index
        directory = overhead / slot
        argv = [
            str(binary),
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
            "--effective-config-json",
            str(directory / "effective-config.json"),
            "--decision-trace-jsonl",
            str(directory / "decision-trace.jsonl"),
        ]
        if mode == "latency":
            argv.extend(
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
            argv,
            repo=repo,
            environment=environment,
            directory=directory,
            port=port,
        )
        try:
            server.start()
            models = http_get_json(port, "/v1/models")
            checkpoint.write_json(directory / "models.json", models)
            command = bench_argv(
                binary,
                model,
                served_model_id(models),
                port,
                directory / "bench.json",
            )
            write_command(directory / "bench.command", command)
            checkpoint.write_text(directory / "bench.started_at", checkpoint.iso_now() + "\n")
            with (directory / "bench.stdout.log").open("w", encoding="utf-8") as stdout, (
                directory / "bench.stderr.log"
            ).open("w", encoding="utf-8") as stderr:
                result = subprocess.run(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=BENCH_TIMEOUT_SECONDS,
                    check=False,
                    start_new_session=True,
                )
            checkpoint.write_text(directory / "bench.exit_code", f"{result.returncode}\n")
            checkpoint.write_text(directory / "bench.finished_at", checkpoint.iso_now() + "\n")
            require(result.returncode == 0, f"overhead bench failed: {slot}")
            checkpoint.write_json(directory / "health.after.json", http_get_json(port, "/health"))
        except subprocess.TimeoutExpired as error:
            raise CollectionError(f"overhead bench exceeded {BENCH_TIMEOUT_SECONDS:.0f}s: {slot}") from error
        finally:
            server.stop()
        throughput = slot_throughput(directory / "bench.json")
        slots.append(
            {
                "slot": slot,
                "mode": mode,
                "output_throughput_tps": throughput,
            }
        )
        print(f"S2 LATENCY OVERHEAD SLOT COLLECTED: {slot}", flush=True)
    checkpoint.write_json(overhead / "report.json", overhead_report(slots))


def write_artifact_tree(root: Path) -> None:
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": checkpoint.file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != "artifact_tree.json"
    ]
    tree = {
        "schema_version": 1,
        "artifact_type": checkpoint.CHECKPOINT_ID,
        "file_count": len(entries),
        "files": entries,
    }
    tree["canonical_sha256"] = checkpoint.canonical_sha256(tree)
    checkpoint.write_json(root / "artifact_tree.json", tree)


def collect(args: argparse.Namespace) -> int:
    repo = args.repo.resolve(strict=True)
    model = args.model.resolve(strict=True)
    source_binary = args.binary.resolve(strict=True)
    out = args.out.resolve(strict=False)
    require((repo / "Cargo.toml").is_file(), "--repo is not the Ferrum workspace")
    require(model.is_dir(), "--model is not a snapshot directory")
    require(
        source_binary.is_file() and os.access(source_binary, os.X_OK),
        "--binary is not executable",
    )
    require(not out.exists(), f"refusing to overwrite artifact directory: {out}")
    require(1024 <= args.port_base <= 65525, "--port-base leaves insufficient port range")
    git_sha = command_output(["git", "rev-parse", "HEAD"], repo)
    git_tree = command_output(["git", "rev-parse", "HEAD^{tree}"], repo)
    dirty = command_output(["git", "status", "--porcelain"], repo)
    require(not dirty, "collector requires a clean source checkout")
    environment, removed_hidden = product_environment()
    out.mkdir(parents=True)
    checkpoint.write_text(out / "started_at", checkpoint.iso_now() + "\n")
    source_receipts = out / "source"
    source_receipts.mkdir()
    checkpoint.write_text(source_receipts / "git.sha", git_sha + "\n")
    checkpoint.write_text(source_receipts / "git.tree", git_tree + "\n")
    checkpoint.write_text(source_receipts / "git.status", "")
    staged_binary = out / "binary" / "ferrum"
    staged_binary.parent.mkdir()
    shutil.copy2(source_binary, staged_binary)
    staged_binary.chmod(staged_binary.stat().st_mode | 0o111)
    hardware = hardware_identity(repo, out)
    model_evidence = model_closure(model)
    checkpoint.write_json(out / "model-closure.json", model_evidence)
    inputs = out / "inputs"
    inputs.mkdir()
    source_inputs = {
        "collector": Path(__file__).resolve(),
        "collector_validator": Path(checkpoint.__file__).resolve(),
        "profile_analyzer": checkpoint.ANALYZER_PATH,
    }
    input_receipts: dict[str, dict[str, str]] = {}
    for key, source in source_inputs.items():
        destination = inputs / source.name
        destination.write_bytes(source.read_bytes())
        input_receipts[key] = {
            "path": destination.relative_to(out).as_posix(),
            "sha256": checkpoint.file_sha256(destination),
        }
    collection = {
        "schema_version": 1,
        "artifact_type": checkpoint.CHECKPOINT_ID,
        "artifact_root": str(out),
        "git_sha": git_sha,
        "git_tree": git_tree,
        "dirty_status": {"is_dirty": False, "status_short": []},
        "binary": {
            "path": staged_binary.relative_to(out).as_posix(),
            "source_path": str(source_binary),
            "sha256": checkpoint.file_sha256(staged_binary),
        },
        "hardware": hardware,
        "model": model_evidence,
        "inputs": input_receipts,
        "scenarios": list(checkpoint.SCENARIOS),
        "environment": {
            "removed_hidden_ferrum_env_names": removed_hidden,
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            "ld_library_path": environment.get("LD_LIBRARY_PATH"),
        },
        "protocol": {
            "profile_detail": "latency",
            "profile_sample_rate": 1.0,
            "diagnostic_fault": checkpoint.FAULT_VALUE,
            "overhead_comparison": "ABBA-BAAB",
            "overhead_target_fraction": 0.05,
            "overhead_blocking": False,
            "bench_concurrency": 1,
            "bench_repeats": 3,
            "bench_seed": 9271,
        },
    }
    checkpoint.write_json(out / "collection.json", collection)
    collect_run_scenario(
        repo=repo,
        binary=staged_binary,
        model=model,
        environment=environment,
        root=out,
        failure=False,
    )
    collect_serve_scenario(
        repo=repo,
        binary=staged_binary,
        model=model,
        environment=environment,
        root=out,
        port=args.port_base,
        failure=False,
    )
    collect_run_scenario(
        repo=repo,
        binary=staged_binary,
        model=model,
        environment=environment,
        root=out,
        failure=True,
    )
    collect_serve_scenario(
        repo=repo,
        binary=staged_binary,
        model=model,
        environment=environment,
        root=out,
        port=args.port_base + 1,
        failure=True,
    )
    print("S2 LATENCY CORRECTNESS COLLECTION COMPLETE; STARTING REPORT-ONLY OVERHEAD", flush=True)
    collect_overhead(
        repo=repo,
        binary=staged_binary,
        model=model,
        environment=environment,
        root=out,
        port_base=args.port_base + 2,
    )
    checkpoint.write_text(out / "finished_at", checkpoint.iso_now() + "\n")
    write_artifact_tree(out)
    print(f"{COLLECTED_PREFIX}: {out}")
    return 0


def self_test() -> int:
    hardware = parse_hardware_csv(
        "NVIDIA GeForce RTX 4090, GPU-fixture, 24564, 570.00\n"
    )
    require(hardware["gpu_count"] == 1 and hardware["memory_total_mib"] == 24564, "hardware parser drifted")
    slots = []
    for index, slot in enumerate(checkpoint.OVERHEAD_SLOT_ORDER):
        mode = "latency" if slot.startswith("latency") else "off"
        slots.append(
            {
                "slot": slot,
                "mode": mode,
                "output_throughput_tps": (98.0 if mode == "latency" else 100.0) + index * 0.01,
            }
        )
    report = overhead_report(slots)
    require(report["classification"] == "stable_target_met" and report["blocking"] is False, "overhead report contract drifted")
    with tempfile.TemporaryDirectory(prefix="ferrum-s2-latency-model-") as temporary:
        snapshot = (
            Path(temporary)
            / "models--Qwen--Qwen3.5-4B"
            / "snapshots"
            / ("a" * 40)
        )
        snapshot.mkdir(parents=True)
        checkpoint.write_text(snapshot / "config.json", "{}\n")
        checkpoint.write_text(snapshot / "tokenizer_config.json", "{}\n")
        checkpoint.write_text(snapshot / "model.safetensors", "fixture\n")
        closure = model_closure(snapshot)
        require(closure["revision"] == "a" * 40 and len(closure["files"]) == 3, "model closure drifted")
        require(checkpoint.SHA256_RE.fullmatch(closure["closure_sha256"]) is not None, "model closure SHA invalid")
    print(SELFTEST_PASS_LINE)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--repo", type=Path, default=checkpoint.REPO_ROOT)
    collect_parser.add_argument("--model", type=Path, required=True)
    collect_parser.add_argument("--binary", type=Path, required=True)
    collect_parser.add_argument("--out", type=Path, required=True)
    collect_parser.add_argument("--port-base", type=int, default=18480)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.command is not None:
            parser.error("--self-test cannot be combined with collect")
    elif args.command != "collect":
        parser.error("collect subcommand or --self-test is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        return self_test()
    return collect(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        CollectionError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        checkpoint.ValidationError,
    ) as error:
        print(f"FERRUM RUNTIME VNEXT S2 LATENCY FIRST FAILURE COLLECTOR FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
