#!/usr/bin/env python3
"""Validate the actual Qwen3.5-4B CUDA S1 run/serve trace checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PASS_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA TRACE CHECKPOINT PASS"
BASIC_SLICE_PASS_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA BASIC SLICE PASS"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_EVENTS = {
    "request_accepted",
    "plan_built",
    "frame_started",
    "node_started",
    "operation_submitted",
    "node_retired",
    "frame_completed",
    "sequence_completed",
    "request_completed",
}
OPERATION_IDENTITY_FIELDS = {
    "plan_id",
    "plan_hash",
    "node_id",
    "operation_id",
    "provider_id",
    "device_id",
}
FORBIDDEN_PATTERNS = (
    "panic",
    "panicked",
    "<unk>",
    "[pad",
    "invalid utf-8",
    "cuda out of memory",
    "kv cache overflow",
    "differs from its value binding",
    "segmentation fault",
)
PROFILE_SLOT_ORDER = (
    "off1",
    "basic1",
    "basic2",
    "off2",
    "basic3",
    "off3",
    "off4",
    "basic4",
)
OFF_PROFILE_SLOTS = ("off1", "off2", "off3", "off4")
BASIC_PROFILE_SLOTS = ("basic1", "basic2", "basic3", "basic4")
PROFILE_REPEAT_COUNT = 3
PROFILE_REQUESTS_PER_REPEAT = 4
PROFILE_WARMUP_REQUESTS = 1
PROFILE_EXPECTED_REQUESTS_PER_SLOT = 15
PROFILE_EXPECTED_NODES_PER_REQUEST = 131
PROFILE_EXPECTED_EVENTS_PER_REQUEST = 399
PROFILE_MAX_BYTES_PER_REQUEST = 1024 * 1024
PROFILE_MAX_OVERHEAD = 0.02
PROFILE_MAX_CV = 0.05


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def read_text(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    return path.read_text(errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(read_text(path))
    except json.JSONDecodeError as error:
        raise ValidationError(f"malformed JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(read_text(path).splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValidationError(f"malformed JSONL {path}:{line_number}: {error}") from error
        require(isinstance(value, dict), f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    require(rows, f"empty JSONL artifact: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_zero_exit(path: Path) -> None:
    require(read_text(path).strip() == "0", f"non-zero or malformed exit receipt: {path}")


def validate_forbidden_logs(root: Path) -> None:
    candidates = [
        root / "build.log",
        root / "run" / "stderr.log",
        root / "run" / "stdout.jsonl",
        root / "serve" / "stderr.log",
        root / "serve" / "stdout.log",
        root / "serve" / "stream.sse",
        root / "serve" / "bench.stderr.log",
    ]
    for path in candidates:
        text = read_text(path).lower()
        for pattern in FORBIDDEN_PATTERNS:
            require(pattern not in text, f"forbidden pattern {pattern!r} in {path}")
        require("\ufffd" not in text, f"Unicode replacement character in {path}")


def validate_resource_balance(rows: list[dict[str, Any]], label: str) -> dict[str, int]:
    amounts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    request_actions: Counter[str] = Counter()
    resource_rows = 0
    for row in rows:
        resource = row.get("resource")
        if not isinstance(resource, dict):
            continue
        action = resource.get("action")
        if not isinstance(action, str):
            continue
        resource_rows += 1
        if action in {"request_open", "request_close"}:
            request_actions[action] += 1
            continue
        if action not in {"reserve", "commit", "release"}:
            continue
        key = (
            str(resource.get("owner_kind")),
            str(resource.get("owner_id")),
            str(resource.get("resource_kind")),
        )
        amount = resource.get("amount", 1)
        require(isinstance(amount, int) and amount > 0, f"{label}: invalid resource amount {key}")
        amounts[key][action] += amount
    require(resource_rows > 0, f"{label}: no resource trace rows")
    require(
        request_actions["request_open"] == request_actions["request_close"] > 0,
        f"{label}: request open/close imbalance {dict(request_actions)}",
    )
    for key, actions in amounts.items():
        require(actions["reserve"] > 0, f"{label}: release/commit without reserve for {key}")
        require(
            actions["reserve"] == actions["commit"] == actions["release"],
            f"{label}: resource imbalance for {key}: {dict(actions)}",
        )
    return {
        "resource_rows": resource_rows,
        "balanced_resource_owners": len(amounts),
        "request_lifecycles": request_actions["request_open"],
    }


def validate_vnext_trace(rows: list[dict[str, Any]], entrypoint: str) -> dict[str, Any]:
    events = [
        row
        for row in rows
        if row.get("phase", "").startswith("vnext.")
        and (row.get("attributes") or {}).get("execution_trace_source") == "vnext"
    ]
    require(events, f"{entrypoint}: no typed vNext execution events")
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    kinds: Counter[str] = Counter()
    for row in events:
        require(row.get("schema_version") == 1, f"{entrypoint}: wrong profile schema")
        require(row.get("entrypoint") == entrypoint, f"{entrypoint}: entrypoint mismatch")
        require(row.get("status") == "ok", f"{entrypoint}: non-ok vNext event")
        attributes = row.get("attributes")
        shape = row.get("shape")
        require(isinstance(attributes, dict), f"{entrypoint}: missing event attributes")
        require(isinstance(shape, dict), f"{entrypoint}: missing event shape")
        kind = attributes.get("execution_event_kind")
        sequence = shape.get("execution_sequence")
        request_id = row.get("request_id")
        require(isinstance(kind, str), f"{entrypoint}: missing execution event kind")
        require(isinstance(sequence, int) and sequence > 0, f"{entrypoint}: bad sequence")
        require(isinstance(request_id, str) and request_id, f"{entrypoint}: missing request id")
        require(attributes.get("run_id"), f"{entrypoint}: missing run id")
        require(attributes.get("span_id"), f"{entrypoint}: missing span id")
        if kind == "operation_submitted":
            missing = sorted(field for field in OPERATION_IDENTITY_FIELDS if not attributes.get(field))
            require(not missing, f"{entrypoint}: operation identity missing {missing}")
        kinds[kind] += 1
        by_request[request_id].append(row)
    missing_events = sorted(REQUIRED_EVENTS - set(kinds))
    require(not missing_events, f"{entrypoint}: missing required events {missing_events}")
    require(kinds["failure_observed"] == 0, f"{entrypoint}: failure event present")
    require(kinds["request_failed"] == 0, f"{entrypoint}: failed request event present")
    require(
        kinds["frame_started"] == kinds["frame_completed"] > 0,
        f"{entrypoint}: frame lifecycle imbalance",
    )
    require(
        kinds["node_started"] == kinds["operation_submitted"] == kinds["node_retired"] > 0,
        f"{entrypoint}: node/operation lifecycle imbalance",
    )
    for request_id, request_rows in by_request.items():
        sequences = [row["shape"]["execution_sequence"] for row in request_rows]
        require(
            sequences == list(range(1, len(sequences) + 1)),
            f"{entrypoint}: non-contiguous execution sequence for {request_id}",
        )
        request_kinds = [(row.get("attributes") or {}).get("execution_event_kind") for row in request_rows]
        require(request_kinds[0] == "request_accepted", f"{entrypoint}: request does not start accepted")
        require(request_kinds[-1] == "request_completed", f"{entrypoint}: request does not complete")
    return {
        "typed_event_rows": len(events),
        "typed_request_count": len(by_request),
        "event_counts": dict(sorted(kinds.items())),
    }


def validate_run(root: Path) -> dict[str, Any]:
    run = root / "run"
    require_zero_exit(run / "exit")
    output = read_jsonl(run / "stdout.jsonl")
    assistants = [row for row in output if row.get("event") == "assistant"]
    require(len(assistants) == 1, "run: expected exactly one assistant row")
    assistant = assistants[0]
    require(assistant.get("content", "").strip() == "Paris", "run: answer is not Paris")
    require(assistant.get("finish_reason") == "stop", "run: finish_reason is not stop")
    require(isinstance(assistant.get("n_tokens"), int) and assistant["n_tokens"] > 0, "run: no tokens")
    profile_rows = read_jsonl(run / "profile.jsonl")
    trace_rows = read_jsonl(run / "scheduler-trace.jsonl")
    return {
        "assistant": assistant["content"].strip(),
        "tokens": assistant["n_tokens"],
        "profile_rows": len(profile_rows),
        **validate_resource_balance(trace_rows, "run"),
        **validate_vnext_trace(trace_rows, "run"),
    }


def validate_serve(root: Path) -> dict[str, Any]:
    serve = root / "serve"
    require_zero_exit(serve / "server.exit")
    stop = read_json(serve / "server-stop.json")
    require(stop.get("stopped") is True, "serve: server process was not stopped")
    require(read_text(serve / "http.status").strip() == "200", "serve: HTTP status is not 200")
    require(read_json(serve / "health.json").get("status") == "healthy", "serve: health failed")
    models = read_json(serve / "models.json").get("data")
    require(isinstance(models, list) and models, "serve: model list is empty")
    sse = read_text(serve / "stream.sse")
    require(sse.count("data: [DONE]") == 1, "serve: expected exactly one [DONE]")
    stream_rows = read_jsonl(serve / "stream.data.jsonl")
    content = "".join(
        str(choice.get("delta", {}).get("content") or "")
        for row in stream_rows
        for choice in row.get("choices", [])
        if isinstance(choice, dict)
    )
    require(content.strip() == "Paris", "serve: streamed answer is not Paris")
    usage = [row.get("usage") for row in stream_rows if row.get("usage") is not None]
    require(len(usage) == 1, "serve: expected exactly one usage row")
    require(
        isinstance(usage[0], dict) and int(usage[0].get("completion_tokens", 0)) > 0,
        "serve: missing positive completion usage",
    )
    profile_rows = read_jsonl(serve / "profile.jsonl")
    trace_rows = read_jsonl(serve / "scheduler-trace.jsonl")
    summary = {
        "assistant": content.strip(),
        "completion_tokens": usage[0]["completion_tokens"],
        "profile_rows": len(profile_rows),
        **validate_resource_balance(trace_rows, "serve"),
        **validate_vnext_trace(trace_rows, "serve"),
    }
    require_zero_exit(serve / "bench.exit")
    report = read_json(serve / "bench-smoke.json")
    expected = report.get("n_requests_per_run")
    completed = report.get("completed_per_run")
    errored = report.get("errored_per_run")
    require(isinstance(expected, int) and expected > 0, "serve: bad bench request count")
    require(completed == [expected], "serve: bench did not complete every request")
    require(errored == [0], "serve: bench smoke has request errors")
    for key in (
        "bad_output_per_run",
        "duplicate_done_per_run",
        "http_500_per_run",
        "malformed_stream_per_run",
        "missing_done_per_run",
        "panic_per_run",
        "stream_bulk_flush_per_run",
        "zero_output_tokens_per_run",
    ):
        require(report.get(key) == [0], f"serve: bench protocol failure in {key}")
    quality_issues = report.get("quality_issues_per_run")
    require(
        isinstance(quality_issues, list)
        and len(quality_issues) == 1
        and isinstance(quality_issues[0], dict)
        and quality_issues[0]
        and all(value == 0 for value in quality_issues[0].values()),
        "serve: bench quality issue counters are non-zero or malformed",
    )
    require(report.get("output_token_count_source") == "usage", "serve: bench token source is not usage")
    summary["bench_smoke"] = {
        "completed_requests": expected,
        "failed_requests": 0,
        "request_throughput_rps": (report.get("request_throughput_rps") or {}).get("mean"),
        "output_throughput_tps": (report.get("output_throughput_tps") or {}).get("mean"),
    }
    return summary


def validate(root: Path, expected_git_sha: str | None) -> dict[str, Any]:
    root = root.resolve()
    require(root.is_dir(), f"artifact directory does not exist: {root}")
    source_sha = read_text(root / "git.sha").strip()
    require(GIT_SHA_RE.fullmatch(source_sha) is not None, "invalid source git SHA")
    if expected_git_sha is not None:
        require(source_sha == expected_git_sha, "artifact source SHA differs from expected SHA")
    require(not read_text(root / "git.status").strip(), "remote worktree was dirty")
    require_zero_exit(root / "build.exit")
    binary_line = read_text(root / "binary.sha256").strip().split()
    require(binary_line and SHA256_RE.fullmatch(binary_line[0]) is not None, "invalid binary SHA256")
    require("RTX 4090" in read_text(root / "hardware.csv"), "artifact is not from RTX 4090")
    validate_forbidden_logs(root)
    summary = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s1_cuda_trace_checkpoint",
        "status": "pass",
        "source_git_sha": source_sha,
        "binary_sha256": binary_line[0],
        "hardware": read_text(root / "hardware.csv").strip(),
        "run": validate_run(root),
        "serve": validate_serve(root),
    }
    artifact_files = [path for path in root.rglob("*") if path.is_file() and path.name != "validation.json"]
    summary["artifact_files"] = {
        str(path.relative_to(root)): sha256(path) for path in sorted(artifact_files)
    }
    return summary


def command_tokens(path: Path) -> list[str]:
    try:
        tokens = shlex.split(read_text(path))
    except ValueError as error:
        raise ValidationError(f"malformed command receipt {path}: {error}") from error
    require(tokens, f"empty command receipt: {path}")
    require(
        not any(token.startswith("FERRUM_") and "=" in token for token in tokens),
        f"hidden FERRUM environment override in command receipt: {path}",
    )
    return tokens


def require_option(tokens: list[str], option: str, expected: str, label: str) -> None:
    positions = [index for index, token in enumerate(tokens) if token == option]
    require(len(positions) == 1, f"{label}: expected exactly one {option}")
    index = positions[0]
    require(index + 1 < len(tokens), f"{label}: {option} lacks a value")
    require(tokens[index + 1] == expected, f"{label}: {option} is not {expected!r}")


def require_zero_quality(report: dict[str, Any], label: str) -> None:
    expected = [0] * PROFILE_REPEAT_COUNT
    for key in (
        "bad_output_per_run",
        "duplicate_done_per_run",
        "http_500_per_run",
        "malformed_stream_per_run",
        "missing_done_per_run",
        "panic_per_run",
        "stream_bulk_flush_per_run",
        "zero_output_tokens_per_run",
    ):
        require(report.get(key) == expected, f"{label}: non-zero or malformed {key}")
    quality = report.get("quality_issues_per_run")
    require(
        isinstance(quality, list)
        and len(quality) == PROFILE_REPEAT_COUNT
        and all(
            isinstance(row, dict)
            and row
            and all(isinstance(value, int) and value == 0 for value in row.values())
            for row in quality
        ),
        f"{label}: non-zero or malformed quality_issues_per_run",
    )


def validate_profile_report(directory: Path, slot: str, mode: str) -> dict[str, Any]:
    label = f"profile-overhead/{slot}"
    require_zero_exit(directory / "server.exit")
    require_zero_exit(directory / "bench.exit")
    server_tokens = command_tokens(directory / "server.command")
    require("serve" in server_tokens, f"{label}: server command is not ferrum serve")
    serve_index = server_tokens.index("serve")
    require(serve_index + 1 < len(server_tokens), f"{label}: serve command lacks a model")
    model_path = server_tokens[serve_index + 1]
    require(
        "models--Qwen--Qwen3.5-4B/snapshots/" in model_path,
        f"{label}: model is not Qwen3.5-4B",
    )
    require_option(server_tokens, "--backend", "cuda", label)
    require_option(server_tokens, "--profile-detail", mode, label)
    if mode == "off":
        require("--profile-jsonl" not in server_tokens, f"{label}: off command writes profile JSONL")
        require(
            "--scheduler-trace-jsonl" not in server_tokens,
            f"{label}: off command writes scheduler trace",
        )
    else:
        require_option(server_tokens, "--profile-sample-rate", "1.0", label)
        require("--profile-jsonl" in server_tokens, f"{label}: basic command lacks profile JSONL")
        require(
            "--scheduler-trace-jsonl" in server_tokens,
            f"{label}: basic command lacks scheduler trace",
        )

    bench_tokens = command_tokens(directory / "bench.command")
    require("bench-serve" in bench_tokens, f"{label}: benchmark is not ferrum bench-serve")
    require("--require-ci" in bench_tokens, f"{label}: benchmark lacks --require-ci")
    require("--fail-on-error" in bench_tokens, f"{label}: benchmark lacks --fail-on-error")
    for option, expected in (
        ("--target-backend", "cuda"),
        ("--concurrency", "1"),
        ("--dataset", "random"),
        ("--random-input-len", "128"),
        ("--random-output-len", "64"),
        ("--num-prompts", str(PROFILE_REQUESTS_PER_REPEAT)),
        ("--warmup-requests", str(PROFILE_WARMUP_REQUESTS)),
        ("--n-repeats", str(PROFILE_REPEAT_COUNT)),
        ("--seed", "9271"),
    ):
        require_option(bench_tokens, option, expected, label)

    report = read_json(directory / "bench.json")
    require(report.get("n_repeats") == PROFILE_REPEAT_COUNT, f"{label}: wrong repeat count")
    require(
        report.get("n_requests_per_run") == PROFILE_REQUESTS_PER_REPEAT,
        f"{label}: wrong request count",
    )
    require(
        report.get("warmup_requests") == PROFILE_WARMUP_REQUESTS,
        f"{label}: wrong warmup count",
    )
    require(
        report.get("completed_per_run") == [PROFILE_REQUESTS_PER_REPEAT] * PROFILE_REPEAT_COUNT,
        f"{label}: incomplete measured requests",
    )
    require(report.get("errored_per_run") == [0] * PROFILE_REPEAT_COUNT, f"{label}: request errors")
    require(report.get("output_token_count_source") == "usage", f"{label}: token source is not usage")
    require_zero_quality(report, label)
    repeats = report.get("repeat_metrics")
    require(
        isinstance(repeats, list) and len(repeats) == PROFILE_REPEAT_COUNT,
        f"{label}: malformed repeat metrics",
    )
    throughput: list[float] = []
    for index, row in enumerate(repeats, 1):
        require(isinstance(row, dict), f"{label}: repeat {index} is not an object")
        require(row.get("repeat") == index, f"{label}: repeat ordinal mismatch")
        require(row.get("completed_requests") == PROFILE_REQUESTS_PER_REPEAT, f"{label}: repeat incomplete")
        require(row.get("errored_requests") == 0, f"{label}: repeat has errors")
        require(row.get("warmup_completed") == PROFILE_WARMUP_REQUESTS, f"{label}: warmup incomplete")
        require(row.get("warmup_errored") == 0, f"{label}: warmup errors")
        require(row.get("output_token_count_source") == "usage", f"{label}: repeat token source")
        value = row.get("output_throughput_tps")
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0,
            f"{label}: invalid output throughput",
        )
        throughput.append(float(value))
    return {"model_path": model_path, "report": report, "throughput": throughput}


def validate_product_commands(root: Path) -> dict[str, Any]:
    run_tokens = command_tokens(root / "run" / "command")
    require("run" in run_tokens, "run command is not ferrum run")
    run_index = run_tokens.index("run")
    require(run_index + 1 < len(run_tokens), "run command lacks a model")
    run_model = run_tokens[run_index + 1]
    require(
        "models--Qwen--Qwen3.5-4B/snapshots/" in run_model,
        "run model is not Qwen3.5-4B",
    )
    require_option(run_tokens, "--backend", "cuda", "run")
    require_option(run_tokens, "--profile-detail", "basic", "run")
    require_option(run_tokens, "--profile-sample-rate", "1.0", "run")
    require("--profile-jsonl" in run_tokens, "run command lacks profile JSONL")
    require("--scheduler-trace-jsonl" in run_tokens, "run command lacks scheduler trace")

    serve_tokens = command_tokens(root / "serve" / "command")
    require("serve" in serve_tokens, "serve command is not ferrum serve")
    serve_index = serve_tokens.index("serve")
    require(serve_index + 1 < len(serve_tokens), "serve command lacks a model")
    serve_model = serve_tokens[serve_index + 1]
    require(serve_model == run_model, "run and serve use different model snapshots")
    require_option(serve_tokens, "--backend", "cuda", "serve")
    require_option(serve_tokens, "--profile-detail", "basic", "serve")
    require_option(serve_tokens, "--profile-sample-rate", "1.0", "serve")
    require("--profile-jsonl" in serve_tokens, "serve command lacks profile JSONL")
    require("--scheduler-trace-jsonl" in serve_tokens, "serve command lacks scheduler trace")

    bench_tokens = command_tokens(root / "serve" / "bench.command")
    require("bench-serve" in bench_tokens, "serve smoke is not ferrum bench-serve")
    require("--fail-on-error" in bench_tokens, "serve smoke lacks --fail-on-error")
    require_option(bench_tokens, "--target-backend", "cuda", "serve smoke")
    require_option(bench_tokens, "--seed", "9271", "serve smoke")
    return {
        "model_id": "Qwen/Qwen3.5-4B",
        "model_snapshot_path": run_model,
        "backend": "cuda",
        "entrypoints": ["ferrum run", "ferrum serve"],
        "hidden_ferrum_environment_overrides": 0,
    }


def validate_bounded_profile_trace(
    directory: Path,
    slot: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    label = f"profile-overhead/{slot}"
    path = directory / "scheduler-trace.jsonl"
    rows = read_jsonl(path)
    events = [
        row
        for row in rows
        if row.get("phase", "").startswith("vnext.")
        and (row.get("attributes") or {}).get("execution_trace_source") == "vnext"
    ]
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    terminals: list[int] = []
    counts: Counter[str] = Counter()
    operation_identity_omissions = 0
    for event in events:
        attributes = event.get("attributes")
        shape = event.get("shape")
        request_id = event.get("request_id")
        require(isinstance(attributes, dict), f"{label}: event lacks attributes")
        require(isinstance(shape, dict), f"{label}: event lacks shape")
        require(isinstance(request_id, str) and request_id, f"{label}: event lacks request id")
        require(
            attributes.get("execution_capture_policy") == "first_frame_per_request",
            f"{label}: event is not bounded first-frame capture",
        )
        kind = attributes.get("execution_event_kind")
        require(isinstance(kind, str), f"{label}: event lacks kind")
        counts[kind] += 1
        by_request[request_id].append(event)
        if kind == "operation_submitted":
            operation_identity_omissions += int(
                any(not attributes.get(field) for field in OPERATION_IDENTITY_FIELDS)
            )
        if kind == "request_completed":
            output = shape.get("event_output_count")
            require(
                isinstance(output, int) and 0 < output <= 64,
                f"{label}: terminal output count is outside 1..64",
            )
            terminals.append(output)

    require(
        len(by_request) == PROFILE_EXPECTED_REQUESTS_PER_SLOT,
        f"{label}: expected {PROFILE_EXPECTED_REQUESTS_PER_SLOT} typed requests",
    )
    for request_id, request_events in by_request.items():
        sequences = [event["shape"].get("execution_sequence") for event in request_events]
        require(
            sequences == list(range(1, PROFILE_EXPECTED_EVENTS_PER_REQUEST + 1)),
            f"{label}: non-contiguous or wrong event count for {request_id}",
        )
        request_counts = Counter(
            str(event["attributes"].get("execution_event_kind")) for event in request_events
        )
        for kind in (
            "request_accepted",
            "plan_built",
            "frame_started",
            "frame_completed",
            "sequence_completed",
            "request_completed",
        ):
            require(request_counts[kind] == 1, f"{label}: {request_id} has wrong {kind} count")
        for kind in ("node_started", "operation_submitted", "node_retired"):
            require(
                request_counts[kind] == PROFILE_EXPECTED_NODES_PER_REQUEST,
                f"{label}: {request_id} has wrong {kind} count",
            )
    require(operation_identity_omissions == 0, f"{label}: operation identity omission")
    require(
        len(events) == PROFILE_EXPECTED_EVENTS_PER_REQUEST * PROFILE_EXPECTED_REQUESTS_PER_SLOT,
        f"{label}: wrong total typed event count",
    )
    require(
        counts["frame_started"] == counts["frame_completed"] == PROFILE_EXPECTED_REQUESTS_PER_SLOT,
        f"{label}: repeated or imbalanced frame capture",
    )
    stride = PROFILE_WARMUP_REQUESTS + PROFILE_REQUESTS_PER_REPEAT
    require(len(terminals) == PROFILE_REPEAT_COUNT * stride, f"{label}: terminal count mismatch")
    repeats = report["repeat_metrics"]
    for repeat in range(PROFILE_REPEAT_COUNT):
        start = repeat * stride + PROFILE_WARMUP_REQUESTS
        measured = terminals[start : start + PROFILE_REQUESTS_PER_REPEAT]
        require(
            sum(measured) == repeats[repeat].get("output_tokens"),
            f"{label}: terminal counts differ from usage for repeat {repeat + 1}",
        )
    bytes_per_request = path.stat().st_size / len(by_request)
    require(
        bytes_per_request <= PROFILE_MAX_BYTES_PER_REQUEST,
        f"{label}: trace bytes/request exceeds {PROFILE_MAX_BYTES_PER_REQUEST}",
    )
    return {
        "typed_requests": len(by_request),
        "typed_events": len(events),
        "nodes_per_request": PROFILE_EXPECTED_NODES_PER_REQUEST,
        "captured_frames_per_request": 1,
        "terminal_output_counts": terminals,
        "usage_reconciled": True,
        "operation_identity_omissions": operation_identity_omissions,
        "trace_bytes": path.stat().st_size,
        "bytes_per_request": bytes_per_request,
    }


def scalar_stats(values: list[float]) -> dict[str, Any]:
    require(len(values) >= 2, "profile overhead needs at least two samples")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    return {
        "values": values,
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "sample_stddev": standard_deviation,
        "cv": standard_deviation / mean,
    }


def validate_profile_overhead(root: Path) -> dict[str, Any]:
    performance = root / "profile-overhead"
    require(
        read_text(performance / "slot-order").split() == list(PROFILE_SLOT_ORDER),
        "profile overhead slot order is not ABBA-BAAB",
    )
    reports: dict[str, dict[str, Any]] = {}
    traces: dict[str, dict[str, Any]] = {}
    for slot in PROFILE_SLOT_ORDER:
        mode = "basic" if slot in BASIC_PROFILE_SLOTS else "off"
        result = validate_profile_report(performance / slot, slot, mode)
        reports[slot] = result
        if mode == "basic":
            traces[slot] = validate_bounded_profile_trace(
                performance / slot,
                slot,
                result["report"],
            )
    model_paths = {result["model_path"] for result in reports.values()}
    require(len(model_paths) == 1, "profile slots use different model snapshots")

    off = scalar_stats(
        [value for slot in OFF_PROFILE_SLOTS for value in reports[slot]["throughput"]]
    )
    basic = scalar_stats(
        [value for slot in BASIC_PROFILE_SLOTS for value in reports[slot]["throughput"]]
    )
    mean_overhead = (off["mean"] - basic["mean"]) / off["mean"]
    median_overhead = (off["median"] - basic["median"]) / off["median"]
    require(off["cv"] <= PROFILE_MAX_CV, f"profile off CV {off['cv']:.6f} exceeds 0.05")
    require(basic["cv"] <= PROFILE_MAX_CV, f"profile basic CV {basic['cv']:.6f} exceeds 0.05")
    require(
        mean_overhead <= PROFILE_MAX_OVERHEAD,
        f"basic mean overhead {mean_overhead:.6f} exceeds 0.02",
    )
    require(
        median_overhead <= PROFILE_MAX_OVERHEAD,
        f"basic median overhead {median_overhead:.6f} exceeds 0.02",
    )
    require(read_json(performance / "overhead.json").get("status") == "pass", "overhead summary rejects")
    require(
        read_json(performance / "trace-boundedness.json").get("status") == "pass",
        "trace boundedness summary rejects",
    )
    initial = read_json(performance / "overhead.abba.reject.json")
    require(
        initial.get("status") == "reject" and initial.get("noise_pass") is False,
        "initial noisy ABBA rejection was not preserved",
    )
    return {
        "status": "pass",
        "protocol": {
            "comparison": "ABBA-BAAB",
            "slot_order": list(PROFILE_SLOT_ORDER),
            "concurrency": 1,
            "random_input_len": 128,
            "random_output_len": 64,
            "requests_per_repeat": PROFILE_REQUESTS_PER_REPEAT,
            "repeats_per_slot": PROFILE_REPEAT_COUNT,
            "seed": 9271,
            "require_ci": True,
            "fail_on_error": True,
            "model_snapshot_path": next(iter(model_paths)),
        },
        "off": off,
        "basic": basic,
        "mean_overhead": mean_overhead,
        "median_overhead": median_overhead,
        "max_overhead": PROFILE_MAX_OVERHEAD,
        "max_cv": PROFILE_MAX_CV,
        "completed_requests": {"off": 48, "basic": 48},
        "failed_requests": {"off": 0, "basic": 0},
        "bounded_traces": traces,
    }


def artifact_index(root: Path) -> dict[str, dict[str, Any]]:
    excluded = {"validation.json"}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded and not path.is_symlink()
    ]
    return {
        str(path.relative_to(root)): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(files)
    }


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_basic_slice_evidence(
    root: Path,
    out: Path,
    correctness: dict[str, Any],
    performance: dict[str, Any],
) -> None:
    root = root.resolve()
    out = out.resolve()
    require(out != root, "bounded slice output must not overwrite the raw artifact")
    out.mkdir(parents=True, exist_ok=True)
    validation_path = out / "validation.json"
    raw_index = artifact_index(root)
    product = validate_product_commands(root)
    validation = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s1_cuda_basic_slice_validation",
        "status": "pass",
        "raw_artifact_dir": str(root),
        "source_git_sha": correctness["source_git_sha"],
        "binary_sha256": correctness["binary_sha256"],
        "hardware": correctness["hardware"],
        "product": product,
        "correctness": {
            "run": correctness["run"],
            "serve": correctness["serve"],
        },
        "profile_overhead": performance,
        "raw_artifact_count": len(raw_index),
        "raw_artifact_index_sha256": canonical_json_sha256(raw_index),
        "raw_artifact_index": raw_index,
    }
    write_json(validation_path, validation)
    pass_line = f"{BASIC_SLICE_PASS_PREFIX}: {out}"
    manifest = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s1_cuda_basic_slice_manifest",
        "checkpoint_id": "S1-CUDA-basic",
        "lane": "runtime-vnext-s1-cuda",
        "status": "pass",
        "pass_line": pass_line,
        "artifact_dir": str(out),
        "raw_artifact_dir": str(root),
        "source_git_sha": correctness["source_git_sha"],
        "binary_sha256": correctness["binary_sha256"],
        "hardware": correctness["hardware"],
        "model_id": product["model_id"],
        "backend": product["backend"],
        "entrypoints": product["entrypoints"],
        "validation": {
            "path": "validation.json",
            "sha256": sha256(validation_path),
            "size_bytes": validation_path.stat().st_size,
        },
        "acceptance": {
            "run_correctness": True,
            "serve_correctness": True,
            "stream_correctness": True,
            "bench_serve_correctness": True,
            "bounded_basic_trace": True,
            "profile_mean_overhead_lte_2pct": True,
            "profile_median_overhead_lte_2pct": True,
            "profile_cv_lte_5pct": True,
        },
        "metrics": {
            "mean_overhead_fraction": performance["mean_overhead"],
            "median_overhead_fraction": performance["median_overhead"],
            "off_cv": performance["off"]["cv"],
            "basic_cv": performance["basic"]["cv"],
            "off_completed_requests": performance["completed_requests"]["off"],
            "basic_completed_requests": performance["completed_requests"]["basic"],
            "failed_requests": 0,
            "max_trace_bytes_per_request": max(
                trace["bytes_per_request"]
                for trace in performance["bounded_traces"].values()
            ),
        },
        "unlocks": ["G01B"],
        "does_not_prove": [
            "S1_milestone",
            "G01B",
            "G01",
            "G06",
            "full_model_migration",
            "release",
        ],
    }
    write_json(out / "manifest.json", manifest)
    print(pass_line)


def profile_event(sequence: int, kind: str, entrypoint: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "execution_trace_source": "vnext",
        "execution_event_kind": kind,
        "run_id": "run.selftest",
        "span_id": f"span.{sequence}",
    }
    if kind == "operation_submitted":
        attrs.update({field: f"{field}.selftest" for field in OPERATION_IDENTITY_FIELDS})
    return {
        "schema_version": 1,
        "request_id": "request.selftest",
        "entrypoint": entrypoint,
        "phase": f"vnext.{kind}",
        "status": "ok",
        "shape": {"execution_sequence": sequence},
        "attributes": attrs,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def create_selftest_fixture(root: Path) -> None:
    (root / "run").mkdir(parents=True)
    (root / "serve").mkdir()
    (root / "git.sha").write_text("a" * 40 + "\n")
    (root / "git.status").write_text("")
    (root / "build.exit").write_text("0")
    (root / "build.log").write_text("Finished release build\n")
    (root / "binary.sha256").write_text("b" * 64 + "  target/release/ferrum\n")
    (root / "hardware.csv").write_text("NVIDIA GeForce RTX 4090, 24564 MiB\n")
    event_names = [
        "request_accepted",
        "plan_built",
        "frame_started",
        "node_started",
        "operation_submitted",
        "node_retired",
        "frame_completed",
        "sequence_completed",
        "request_completed",
    ]
    resource_rows = [
        {"resource": {"action": "request_open"}},
        {"resource": {"action": "reserve", "owner_kind": "request", "owner_id": "r", "resource_kind": "slot", "amount": 1}},
        {"resource": {"action": "commit", "owner_kind": "request", "owner_id": "r", "resource_kind": "slot", "amount": 1}},
        {"resource": {"action": "release", "owner_kind": "request", "owner_id": "r", "resource_kind": "slot", "amount": 1}},
        {"resource": {"action": "request_close"}},
    ]
    for entrypoint in ("run", "serve"):
        directory = root / entrypoint
        write_jsonl(directory / "profile.jsonl", [{"profile": "selftest"}])
        write_jsonl(
            directory / "scheduler-trace.jsonl",
            resource_rows + [profile_event(index, kind, entrypoint) for index, kind in enumerate(event_names, 1)],
        )
        (directory / "stderr.log").write_text("")
    (root / "run" / "exit").write_text("0")
    write_jsonl(root / "run" / "stdout.jsonl", [{"event": "assistant", "content": "Paris", "finish_reason": "stop", "n_tokens": 2}])
    (root / "serve" / "server.exit").write_text("0")
    (root / "serve" / "server-stop.json").write_text('{"stopped":true}\n')
    (root / "serve" / "stdout.log").write_text("server stopped\n")
    (root / "serve" / "http.status").write_text("200")
    (root / "serve" / "health.json").write_text('{"status":"healthy"}\n')
    (root / "serve" / "models.json").write_text('{"data":[{"id":"model"}]}\n')
    (root / "serve" / "stream.sse").write_text("data: chunk\n\ndata: [DONE]\n\n")
    write_jsonl(
        root / "serve" / "stream.data.jsonl",
        [
            {"choices": [{"delta": {"content": "Paris"}}]},
            {"choices": [], "usage": {"completion_tokens": 2}},
        ],
    )
    (root / "serve" / "bench.exit").write_text("0")
    (root / "serve" / "bench.stderr.log").write_text("")
    (root / "serve" / "bench-smoke.json").write_text(
        json.dumps(
            {
                "n_requests_per_run": 4,
                "completed_per_run": [4],
                "errored_per_run": [0],
                "bad_output_per_run": [0],
                "duplicate_done_per_run": [0],
                "http_500_per_run": [0],
                "malformed_stream_per_run": [0],
                "missing_done_per_run": [0],
                "panic_per_run": [0],
                "quality_issues_per_run": [
                    {
                        "bad_output": 0,
                        "malformed_stream": 0,
                        "missing_done": 0,
                        "duplicate_done": 0,
                        "zero_output_tokens": 0,
                        "stream_bulk_flush": 0,
                        "http_500": 0,
                        "panic": 0,
                    }
                ],
                "stream_bulk_flush_per_run": [0],
                "zero_output_tokens_per_run": [0],
                "output_token_count_source": "usage",
                "request_throughput_rps": {"mean": 1.0},
                "output_throughput_tps": {"mean": 16.0},
            }
        )
        + "\n"
    )


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="runtime-vnext-s1-") as temp:
        root = Path(temp)
        create_selftest_fixture(root)
        validate(root, "a" * 40)
        rows = read_jsonl(root / "run" / "scheduler-trace.jsonl")
        operation = next(row for row in rows if row.get("phase") == "vnext.operation_submitted")
        del operation["attributes"]["provider_id"]
        write_jsonl(root / "run" / "scheduler-trace.jsonl", rows)
        try:
            validate(root, "a" * 40)
        except ValidationError as error:
            require("provider_id" in str(error), "self-test rejected mutation for wrong reason")
        else:
            raise ValidationError("self-test missing-provider mutation unexpectedly passed")
    print("FERRUM RUNTIME VNEXT S1 CUDA TRACE CHECKPOINT SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", nargs="?", type=Path)
    parser.add_argument("--expected-git-sha")
    parser.add_argument("--require-bounded-overhead", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test()
        require(args.artifact_dir is not None, "artifact_dir is required")
        summary = validate(args.artifact_dir, args.expected_git_sha)
        if args.require_bounded_overhead:
            require(args.out is not None, "--out is required with --require-bounded-overhead")
            performance = validate_profile_overhead(args.artifact_dir.resolve())
            write_basic_slice_evidence(
                args.artifact_dir,
                args.out,
                summary,
                performance,
            )
            return 0
        require(args.out is None, "--out requires --require-bounded-overhead")
        output = args.artifact_dir.resolve() / "validation.json"
        write_json(output, summary)
        print(f"{PASS_PREFIX}: {args.artifact_dir.resolve()}")
        return 0
    except (OSError, ValidationError) as error:
        fail_prefix = (
            "FERRUM RUNTIME VNEXT S1 CUDA BASIC SLICE FAIL"
            if args.require_bounded_overhead
            else "FERRUM RUNTIME VNEXT S1 CUDA TRACE CHECKPOINT FAIL"
        )
        print(f"{fail_prefix}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
