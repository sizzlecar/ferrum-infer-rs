#!/usr/bin/env python3
"""Validate the actual Qwen3.5-4B CUDA S1 run/serve trace checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PASS_PREFIX = "FERRUM RUNTIME VNEXT S1 CUDA TRACE CHECKPOINT PASS"
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
        "malformed_stream_per_run",
        "missing_done_per_run",
        "panic_per_run",
        "quality_issues_per_run",
        "stream_bulk_flush_per_run",
        "zero_output_tokens_per_run",
    ):
        require(report.get(key) == [0], f"serve: bench protocol failure in {key}")
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
                "malformed_stream_per_run": [0],
                "missing_done_per_run": [0],
                "panic_per_run": [0],
                "quality_issues_per_run": [0],
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
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test()
        require(args.artifact_dir is not None, "artifact_dir is required")
        summary = validate(args.artifact_dir, args.expected_git_sha)
        output = args.artifact_dir.resolve() / "validation.json"
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"{PASS_PREFIX}: {args.artifact_dir.resolve()}")
        return 0
    except (OSError, ValidationError) as error:
        print(f"FERRUM RUNTIME VNEXT S1 CUDA TRACE CHECKPOINT FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
