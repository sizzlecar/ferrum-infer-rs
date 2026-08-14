#!/usr/bin/env python3
"""Collect the exact-staged G0 Llama sample used by Runtime vNext R3.

This is intentionally one small Metal/CUDA runner.  It never builds Ferrum:
the executable is recovered from a validated staged-asset build receipt.  One
server process supplies correctness, three same-prompt serve parity samples,
and the c=1 benchmark.  Three independent ``ferrum run`` processes supply the
other side of the parity ratio.  The resulting receipt is consumed (and fully
replayed) by ``runtime_vnext_sampled_final.py``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import runtime_vnext_sampled_final as sampled


RUN_INPUT = "\n".join(
    [
        "Remember the exact phrase ferrum-blue. Reply only OK.",
        "What was the ferrum phrase in the first user message? Reply only with it.",
        "/bye",
        "",
    ]
)
PARITY_MESSAGES = [
    {"role": "user", "content": "Remember the exact phrase ferrum-blue. Reply only OK."},
    {"role": "assistant", "content": "OK"},
    {
        "role": "user",
        "content": "What was the ferrum phrase in the first user message? Reply only with it.",
    },
]
MAX_TOKENS = 128


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    sampled.write_json(path, value)


def sanitized_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key in sampled.LLAMA_ENV_ALLOWLIST
    }


def argv_sha(argv: list[str]) -> str:
    return sampled.canonical_json_sha256(argv)


def prompt_sha() -> str:
    return sampled.canonical_json_sha256(
        {"messages": PARITY_MESSAGES, "max_tokens": MAX_TOKENS, "temperature": 0}
    )


def command_for_run(
    binary: Path, model: str, backend: str, effective: Path
) -> list[str]:
    return [
        str(binary),
        "run",
        model,
        "--backend",
        backend,
        "--temperature",
        "0",
        "--max-tokens",
        str(MAX_TOKENS),
        "--output-format",
        "jsonl",
        "--effective-config-json",
        str(effective),
    ]


def command_for_serve(
    binary: Path,
    model: str,
    backend: str,
    port: int,
    effective: Path,
    extra: list[str],
) -> list[str]:
    return [
        str(binary),
        "serve",
        "--model",
        model,
        "--backend",
        backend,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--effective-config-json",
        str(effective),
        *extra,
    ]


def command_for_bench(
    binary: Path,
    model: str,
    tokenizer: Path,
    port: int,
    report: Path,
    backend: str,
) -> list[str]:
    return [
        str(binary),
        "bench-serve",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        model,
        "--target-backend",
        backend,
        "--tokenizer",
        str(tokenizer),
        "--dataset",
        "random",
        "--random-input-len",
        "64",
        "--random-output-len",
        "128",
        "--num-prompts",
        "8",
        "--warmup-requests",
        "1",
        "--n-repeats",
        "3",
        "--concurrency",
        "1",
        "--fail-on-error",
        "--require-ci",
        "--seed",
        "9271",
        "--output",
        "json",
        "--out",
        str(report),
    ]


def wait_health(port: int, process: subprocess.Popen[Any], timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    last = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise sampled.SampledFinalError(
                f"staged Llama server exited during startup with {process.returncode}"
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
                last = f"HTTP {response.status}"
        except Exception as error:
            last = str(error)
        time.sleep(0.25)
    raise sampled.SampledFinalError(f"staged Llama server readiness timeout: {last}")


def run_process(
    argv: list[str],
    *,
    root: Path,
    name: str,
    environment: dict[str, str],
    timeout_sec: float,
    stdin_text: str | None = None,
) -> dict[str, Any]:
    stdout_path = root / f"{name}.stdout"
    stderr_path = root / f"{name}.stderr"
    started_at = now_iso()
    started = time.monotonic()
    process = subprocess.Popen(
        argv,
        cwd=sampled.REPO_ROOT,
        env=environment,
        text=True,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=stdin_text, timeout=timeout_sec)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        raise sampled.SampledFinalError(f"{name} timed out after {timeout_sec}s") from error
    finished_at = now_iso()
    elapsed_ms = (time.monotonic() - started) * 1000.0
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
    sampled.require(process.returncode == 0, f"{name} failed with {process.returncode}")
    sampled.require(
        sampled.BLOCKER_RE.search(stdout + "\n" + stderr) is None,
        f"{name} emitted a release blocker marker",
    )
    return {
        "argv": argv,
        "pid": process.pid,
        "pgid": process.pid,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_ms": elapsed_ms,
        "returncode": process.returncode,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "stdout": stdout,
        "stderr": stderr,
    }


def http_completion(port: int, payload: dict[str, Any], path: Path) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        status = error.code
        body = error.read().decode("utf-8", errors="replace")
    elapsed_ms = (time.monotonic() - started) * 1000.0
    path.write_text(body, encoding="utf-8", errors="replace")
    sampled.require(status == 200, f"serve request failed with HTTP {status}")
    sampled.require(sampled.BLOCKER_RE.search(body) is None, "serve response has blocker marker")
    parsed = sampled.require_object(json.loads(body), "serve response")
    usage = sampled.require_object(parsed.get("usage"), "serve response usage")
    output_tokens = sampled.positive_int(
        usage.get("completion_tokens"), "serve response completion tokens"
    )
    content = str(parsed.get("choices", [{}])[0].get("message", {}).get("content", ""))
    return {
        "elapsed_ms": elapsed_ms,
        "output_tokens": output_tokens,
        "content": content,
        "finish_reason": str(parsed.get("choices", [{}])[0].get("finish_reason", "stop")),
        "path": path,
    }


def http_stream(port: int, model: str, path: Path) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with a short hello."}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    lines: list[str] = []
    usage_tokens = 0
    done = 0
    with urllib.request.urlopen(request, timeout=300) as response:
        sampled.require(response.status == 200, "stream request failed")
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            lines.append(line)
            body = line[6:]
            if body == "[DONE]":
                done += 1
                continue
            parsed = json.loads(body)
            usage = parsed.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                usage_tokens = usage["completion_tokens"]
    elapsed_ms = (time.monotonic() - started) * 1000.0
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    sampled.require(done == 1 and usage_tokens > 0, "stream lacks one DONE/usage tokens")
    sampled.require(sampled.BLOCKER_RE.search(text) is None, "stream has blocker marker")
    return {
        "elapsed_ms": elapsed_ms,
        "output_tokens": usage_tokens,
        "finish_reason": "stop",
        "done": done,
        "path": path,
        "prompt_sha256": sampled.canonical_json_sha256(payload),
    }


def process_receipt(
    path: Path,
    *,
    observation: dict[str, Any],
    role: str,
    source: dict[str, Any],
    binary_sha256: str,
    model_id: str,
    typed_config_sha256: str,
    environment: dict[str, str],
    artifacts: dict[str, Any],
    server_sha256: str | None,
    shutdown_clean: bool = True,
) -> dict[str, Any]:
    document = {
        "schema_version": sampled.SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g0_process_receipt",
        "role": role,
        "argv": observation["argv"],
        "argv_sha256": argv_sha(observation["argv"]),
        "environment": environment,
        "environment_sha256": sampled.canonical_json_sha256(environment),
        "source": source,
        "binary_sha256": binary_sha256,
        "model_id": model_id,
        "typed_config_sha256": typed_config_sha256,
        "pid": observation["pid"],
        "pgid": observation["pgid"],
        "started_at": observation["started_at"],
        "finished_at": observation["finished_at"],
        "returncode": observation["returncode"],
        "shutdown_clean": shutdown_clean,
        "stdout": sampled.artifact_ref(observation["stdout_path"]),
        "stderr": sampled.artifact_ref(observation["stderr_path"]),
        "artifacts": artifacts,
        "server_process_receipt_sha256": server_sha256,
    }
    write_json(path, document)
    return sampled.artifact_ref(path)


def transcript(
    path: Path,
    *,
    scenario_id: str,
    entrypoint: str,
    source: dict[str, Any],
    binary_sha256: str,
    model_id: str,
    typed_config_sha256: str,
    process_sha256: str,
    prompt_sha256: str,
    output_tokens: int,
    elapsed_ms: float,
    finish_reason: str,
    done: int,
    stdout: Path,
    stderr: Path,
) -> dict[str, Any]:
    write_json(
        path,
        {
            "schema_version": sampled.SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_g0_generation_transcript",
            "status": "pass",
            "scenario_id": scenario_id,
            "entrypoint": entrypoint,
            "source": source,
            "binary_sha256": binary_sha256,
            "model_id": model_id,
            "typed_config_sha256": typed_config_sha256,
            "process_receipt_sha256": process_sha256,
            "prompt_sha256": prompt_sha256,
            "max_tokens": MAX_TOKENS,
            "output_token_count": output_tokens,
            "elapsed_ms": elapsed_ms,
            "finish_reason": finish_reason,
            "error_count": 0,
            "stream_done_count": done,
            "stdout": sampled.artifact_ref(stdout),
            "stderr": sampled.artifact_ref(stderr),
        },
    )
    return sampled.artifact_ref(path)


def last_assistant_metrics(stdout: str) -> tuple[int, float, str]:
    rows = []
    for line in stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "assistant":
            rows.append(row)
    sampled.require(len(rows) >= 2, "ferrum run did not emit two assistant turns")
    last = rows[-1]
    sampled.require("ferrum-blue" in str(last.get("content", "")).lower(), "run lost multi-turn context")
    return (
        sampled.positive_int(last.get("n_tokens"), "run output tokens"),
        sampled.finite_positive(last.get("ms"), "run generation ms"),
        str(last.get("finish_reason", "stop")),
    )


def collect(args: argparse.Namespace) -> Path:
    out = sampled.ensure_fresh_out(args.out)
    staged = sampled.staged_context(args.staged_assets, args.backend)
    build = sampled.validate_receipt(
        args.binary_build_receipt, staged=staged, backend=args.backend
    )
    source = staged["release_candidate"]
    staged_row = staged["assets"][args.backend]
    binary_path = out / "inputs/ferrum"
    binary_path.parent.mkdir(parents=True)
    binary_path.write_bytes(build["binary_path"].read_bytes())
    binary_path.chmod(0o755)
    sampled.require(
        sampled.file_sha256(binary_path) == staged_row["binary"]["sha256"],
        "copied execution binary differs from staged bytes",
    )
    model_files = sampled.require_object(
        sampled.read_json(args.model_files, "Llama model files"), "Llama model files"
    )
    tokenizer_path = out / "inputs/tokenizer"
    tokenizer_path.mkdir()
    shutil.copy2(args.tokenizer / "tokenizer.json", tokenizer_path / "tokenizer.json")
    sampled.require(
        model_files.get("tokenizer.json")
        == sampled.file_sha256(tokenizer_path / "tokenizer.json"),
        "model-files receipt does not bind the tokenizer.json used by bench-serve",
    )
    hardware = sampled.require_object(
        sampled.read_json(args.hardware, "Llama hardware"), "Llama hardware"
    )
    plan = sampled.checked_sample_plan()
    planned = plan["manifest"]["performance"][sampled.LLAMA_MODEL_KEY][args.backend]
    sampled.validate_llama_hardware(
        hardware, args.backend, planned["floor"]["hardware_contract"]
    )
    environment = sanitized_environment()
    gate_started = now_iso()
    effective_raw = out / "serve.effective-config.raw.json"
    server_stdout = out / "server.stdout"
    server_stderr = out / "server.stderr"
    serve_argv = command_for_serve(
        binary_path,
        args.model_id,
        args.backend,
        args.port,
        effective_raw,
        args.serve_extra_arg,
    )
    server_started_at = now_iso()
    server_stdout_handle = server_stdout.open("x", encoding="utf-8")
    server_stderr_handle = server_stderr.open("x", encoding="utf-8")
    try:
        server_process = subprocess.Popen(
            serve_argv,
            cwd=sampled.REPO_ROOT,
            env=environment,
            text=True,
            stdout=server_stdout_handle,
            stderr=server_stderr_handle,
            start_new_session=True,
        )
    finally:
        server_stdout_handle.close()
        server_stderr_handle.close()
    bench_observation: dict[str, Any] | None = None
    serve_samples: list[dict[str, Any]] = []
    stream_sample: dict[str, Any] | None = None
    try:
        wait_health(args.port, server_process, args.ready_timeout_sec)
        sampled.require(effective_raw.is_file(), "serve did not emit typed effective config")
        raw_typed = sampled.require_object(
            sampled.read_json(effective_raw, "raw product effective config"),
            "raw product effective config",
        )
        effective_path = out / "typed-effective-config.json"
        write_json(
            effective_path,
            {
                "source": source,
                "model_key": sampled.LLAMA_MODEL_KEY,
                "model_id": args.model_id,
                "backend": args.backend,
                "binary_sha256": staged_row["binary"]["sha256"],
                "model_files": model_files,
                "typed_effective_config": raw_typed,
            },
        )
        parity_payload = {
            "model": args.model_id,
            "messages": PARITY_MESSAGES,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }
        for ordinal in range(1, 4):
            sample = http_completion(
                args.port, parity_payload, out / f"serve-parity-{ordinal}.json"
            )
            sampled.require("ferrum-blue" in sample["content"].lower(), "serve lost multi-turn context")
            serve_samples.append(sample)
        stream_sample = http_stream(args.port, args.model_id, out / "serve-stream.sse")
        bench_report = out / "bench-c1.json"
        bench_argv = command_for_bench(
            binary_path,
            args.model_id,
            tokenizer_path,
            args.port,
            bench_report,
            args.backend,
        )
        bench_observation = run_process(
            bench_argv,
            root=out,
            name="bench-c1",
            environment=environment,
            timeout_sec=args.bench_timeout_sec,
        )
    finally:
        if server_process.poll() is None:
            try:
                os.killpg(server_process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(server_process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                server_process.wait(timeout=10)
    server_finished_at = now_iso()
    sampled.require(bench_observation is not None and stream_sample is not None, "server lane did not finish")
    sampled.require(
        server_process.returncode in {0, -signal.SIGINT, -signal.SIGTERM},
        f"server shutdown failed with {server_process.returncode}",
    )
    server_observation = {
        "argv": serve_argv,
        "pid": server_process.pid,
        "pgid": server_process.pid,
        "started_at": server_started_at,
        "finished_at": server_finished_at,
        "returncode": server_process.returncode,
        "stdout_path": server_stdout,
        "stderr_path": server_stderr,
    }
    typed_ref = sampled.artifact_ref(effective_path)
    server_receipt = process_receipt(
        out / "server-process-receipt.json",
        observation=server_observation,
        role="ferrum-serve",
        source=source,
        binary_sha256=staged_row["binary"]["sha256"],
        model_id=args.model_id,
        typed_config_sha256=typed_ref["sha256"],
        environment=environment,
        artifacts={"effective_config": sampled.artifact_ref(effective_raw)},
        server_sha256=None,
    )
    bench_receipt = process_receipt(
        out / "bench-process-receipt.json",
        observation=bench_observation,
        role="ferrum-bench-serve",
        source=source,
        binary_sha256=staged_row["binary"]["sha256"],
        model_id=args.model_id,
        typed_config_sha256=typed_ref["sha256"],
        environment=environment,
        artifacts={"report": sampled.artifact_ref(out / "bench-c1.json")},
        server_sha256=server_receipt["sha256"],
    )
    run_observations: list[dict[str, Any]] = []
    run_raw_config_refs: list[dict[str, Any]] = []
    run_typed_config_refs: list[dict[str, Any]] = []
    for ordinal in range(1, 4):
        run_raw_path = out / f"run-effective-config-{ordinal}.raw.json"
        observation = run_process(
            command_for_run(
                binary_path, args.model_id, args.backend, run_raw_path
            ),
            root=out,
            name=f"run-parity-{ordinal}",
            environment=environment,
            timeout_sec=args.run_timeout_sec,
            stdin_text=RUN_INPUT,
        )
        sampled.require(
            run_raw_path.is_file() and not run_raw_path.is_symlink(),
            f"ferrum run {ordinal} did not emit an independent effective config",
        )
        run_raw = sampled.require_object(
            sampled.read_json(run_raw_path, f"run {ordinal} raw effective config"),
            f"run {ordinal} raw effective config",
        )
        run_raw_ref = sampled.artifact_ref(run_raw_path)
        run_typed_path = out / f"run-typed-effective-config-{ordinal}.json"
        write_json(
            run_typed_path,
            {
                "source": source,
                "model_key": sampled.LLAMA_MODEL_KEY,
                "model_id": args.model_id,
                "backend": args.backend,
                "binary_sha256": staged_row["binary"]["sha256"],
                "model_files": model_files,
                "entrypoint": "run",
                "raw_effective_config": run_raw_ref,
                "typed_effective_config": run_raw,
            },
        )
        run_observations.append(observation)
        run_raw_config_refs.append(run_raw_ref)
        run_typed_config_refs.append(sampled.artifact_ref(run_typed_path))
    run_receipts = [
        process_receipt(
            out / f"run-process-receipt-{ordinal}.json",
            observation=observation,
            role="ferrum-run",
            source=source,
            binary_sha256=staged_row["binary"]["sha256"],
            model_id=args.model_id,
            typed_config_sha256=run_typed_config_refs[ordinal - 1]["sha256"],
            environment=environment,
            artifacts={
                "raw_effective_config": run_raw_config_refs[ordinal - 1],
                "typed_effective_config": run_typed_config_refs[ordinal - 1],
            },
            server_sha256=None,
        )
        for ordinal, observation in enumerate(run_observations, start=1)
    ]
    run_metrics = [last_assistant_metrics(item["stdout"]) for item in run_observations]
    run_transcripts = [
        transcript(
            out / f"run-parity-transcript-{ordinal}.json",
            scenario_id="run-parity",
            entrypoint="run",
            source=source,
            binary_sha256=staged_row["binary"]["sha256"],
            model_id=args.model_id,
            typed_config_sha256=run_typed_config_refs[ordinal - 1]["sha256"],
            process_sha256=run_receipts[ordinal - 1]["sha256"],
            prompt_sha256=prompt_sha(),
            output_tokens=metrics[0],
            elapsed_ms=metrics[1],
            finish_reason=metrics[2],
            done=0,
            stdout=run_observations[ordinal - 1]["stdout_path"],
            stderr=run_observations[ordinal - 1]["stderr_path"],
        )
        for ordinal, metrics in enumerate(run_metrics, start=1)
    ]
    serve_transcripts = [
        transcript(
            out / f"serve-parity-transcript-{ordinal}.json",
            scenario_id="serve-parity",
            entrypoint="serve",
            source=source,
            binary_sha256=staged_row["binary"]["sha256"],
            model_id=args.model_id,
            typed_config_sha256=typed_ref["sha256"],
            process_sha256=server_receipt["sha256"],
            prompt_sha256=prompt_sha(),
            output_tokens=sample["output_tokens"],
            elapsed_ms=sample["elapsed_ms"],
            finish_reason=sample["finish_reason"],
            done=0,
            stdout=sample["path"],
            stderr=server_stderr,
        )
        for ordinal, sample in enumerate(serve_samples, start=1)
    ]
    correctness = {
        "run-multiturn": transcript(
            out / "correctness-run-multiturn.json",
            scenario_id="run-multiturn",
            entrypoint="run",
            source=source,
            binary_sha256=staged_row["binary"]["sha256"],
            model_id=args.model_id,
            typed_config_sha256=run_typed_config_refs[0]["sha256"],
            process_sha256=run_receipts[0]["sha256"],
            prompt_sha256=prompt_sha(),
            output_tokens=run_metrics[0][0],
            elapsed_ms=run_metrics[0][1],
            finish_reason=run_metrics[0][2],
            done=0,
            stdout=run_observations[0]["stdout_path"],
            stderr=run_observations[0]["stderr_path"],
        ),
        "serve-multiturn": transcript(
            out / "correctness-serve-multiturn.json",
            scenario_id="serve-multiturn",
            entrypoint="serve",
            source=source,
            binary_sha256=staged_row["binary"]["sha256"],
            model_id=args.model_id,
            typed_config_sha256=typed_ref["sha256"],
            process_sha256=server_receipt["sha256"],
            prompt_sha256=prompt_sha(),
            output_tokens=serve_samples[0]["output_tokens"],
            elapsed_ms=serve_samples[0]["elapsed_ms"],
            finish_reason=serve_samples[0]["finish_reason"],
            done=0,
            stdout=serve_samples[0]["path"],
            stderr=server_stderr,
        ),
        "serve-stream": transcript(
            out / "correctness-serve-stream.json",
            scenario_id="serve-stream",
            entrypoint="serve",
            source=source,
            binary_sha256=staged_row["binary"]["sha256"],
            model_id=args.model_id,
            typed_config_sha256=typed_ref["sha256"],
            process_sha256=server_receipt["sha256"],
            prompt_sha256=stream_sample["prompt_sha256"],
            output_tokens=stream_sample["output_tokens"],
            elapsed_ms=stream_sample["elapsed_ms"],
            finish_reason=stream_sample["finish_reason"],
            done=stream_sample["done"],
            stdout=stream_sample["path"],
            stderr=server_stderr,
        ),
    }
    gate_finished = now_iso()
    gate_path = out / "gate.manifest.json"
    gate = {
        "schema_version": sampled.SCHEMA_VERSION,
        "status": "pass",
        "lane": f"g0-{args.backend}-llama-dense-sampled",
        "git_sha": source["git_sha"],
        "binary": {"path": str(binary_path), "sha256": staged_row["binary"]["sha256"]},
        "model": args.model_id,
        "started_at": gate_started,
        "finished_at": gate_finished,
        "child_returncode": 0,
        "pass_line": f"FERRUM GATE {args.backend}-llama-dense-sampled PASS: {out}",
    }
    write_json(gate_path, gate)
    receipt_path = out / "manifest.json"
    write_json(
        receipt_path,
        {
            "schema_version": sampled.SCHEMA_VERSION,
            "artifact_type": "runtime_vnext_g0_llama_dense_execution_receipt",
            "status": "pass",
            "producer": "g0-llama-dense-execution-binding-v1",
            "source": source,
            "model_key": sampled.LLAMA_MODEL_KEY,
            "model_id": args.model_id,
            "backend": args.backend,
            "hardware": hardware,
            "binary_artifact": sampled.artifact_ref(binary_path),
            "binary_sha256": staged_row["binary"]["sha256"],
            "model_files": model_files,
            "model_files_sha256": sampled.canonical_json_sha256(model_files),
            "typed_effective_config": typed_ref,
            "staged_assets_manifest": staged["ref"],
            "sample_plan": plan["ref"],
            "g0_gate_manifest": sampled.artifact_ref(gate_path),
            "server_process": server_receipt,
            "bench_process": bench_receipt,
            "run_processes": run_receipts,
            "run_parity_transcripts": run_transcripts,
            "correctness_transcripts": correctness,
            "serve_parity_transcripts": serve_transcripts,
            "bench_report": sampled.artifact_ref(out / "bench-c1.json"),
        },
    )
    sampled.validate_g0_llama_execution_receipt(
        receipt_path,
        backend=args.backend,
        staged=staged,
        expected_sample_plan_sha256=plan["ref"]["sha256"],
    )
    print(f"FERRUM G0 LLAMA SAMPLED EXECUTION PASS: {out}")
    return out


def self_test() -> int:
    binary = Path("/tmp/ferrum")
    for backend in ("cuda", "metal"):
        effective = Path(f"/tmp/run-{backend}-effective.json")
        run = command_for_run(binary, "model", backend, effective)
        assert run[:3] == [str(binary), "run", "model"]
        assert run[run.index("--backend") + 1] == backend
        assert run[run.index("--temperature") + 1] == "0"
        assert run[run.index("--effective-config-json") + 1] == str(effective)
        serve = command_for_serve(
            binary, "model", backend, 18080, Path("/tmp/effective.json"), []
        )
        assert serve[1] == "serve"
        assert serve[serve.index("--backend") + 1] == backend
        bench = command_for_bench(
            binary,
            "model",
            Path("/tmp/tokenizer"),
            18080,
            Path("/tmp/report"),
            backend,
        )
        assert bench[1] == "bench-serve"
        assert bench[bench.index("--n-repeats") + 1] == "3"
        assert bench[bench.index("--concurrency") + 1] == "1"
        assert bench[bench.index("--target-backend") + 1] == backend
        assert "--fail-on-error" in bench and "--require-ci" in bench
    print("FERRUM G0 LLAMA SAMPLED EXECUTION SELFTEST PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--backend", choices=sorted(sampled.BACKENDS))
    parser.add_argument("--staged-assets", type=Path)
    parser.add_argument("--binary-build-receipt", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-files", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--port", type=int, default=19300)
    parser.add_argument("--serve-extra-arg", action="append", default=[])
    parser.add_argument("--ready-timeout-sec", type=float, default=900)
    parser.add_argument("--bench-timeout-sec", type=float, default=3600)
    parser.add_argument("--run-timeout-sec", type=float, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    for name in (
        "out",
        "backend",
        "staged_assets",
        "binary_build_receipt",
        "model_id",
        "model_files",
        "tokenizer",
        "hardware",
    ):
        sampled.require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required")
    sampled.require(1024 <= args.port <= 65535, "--port must be 1024..65535")
    sampled.require(args.tokenizer.is_dir(), "--tokenizer must be a directory containing tokenizer.json")
    sampled.require((args.tokenizer / "tokenizer.json").is_file(), "tokenizer.json is missing")
    collect(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except sampled.SampledFinalError as error:
        print(f"FERRUM G0 LLAMA SAMPLED EXECUTION FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
