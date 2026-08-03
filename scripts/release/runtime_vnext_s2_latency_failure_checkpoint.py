#!/usr/bin/env python3
"""Validate the Runtime vNext S2 CUDA latency and first-failure slice."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import analyze_ferrum_profile as profile_analyzer


PASS_PREFIX = "FERRUM RUNTIME VNEXT S2 LATENCY FIRST FAILURE PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT S2 LATENCY FIRST FAILURE FAIL"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT S2 LATENCY FIRST FAILURE SELFTEST PASS"
CHECKPOINT_ID = "runtime-vnext-s2-latency-first-failure"
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = Path(profile_analyzer.__file__).resolve()
SCENARIOS = {
    "run-success": ("run", False),
    "serve-success": ("serve", False),
    "run-failure": ("run", True),
    "serve-failure": ("serve", True),
}
FAULT_VALUE = "prefill-resource-after-submit-once"
FAULT_ERROR_KIND = "diagnostic_resource_after_submit"
OVERHEAD_SLOT_ORDER = (
    "off1",
    "latency1",
    "latency2",
    "off2",
    "latency3",
    "off3",
    "off4",
    "latency4",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SNAPSHOT_RE = re.compile(
    r"(?:^|/)models--Qwen--Qwen3\.5-4B/snapshots/([0-9a-f]{40})/?$"
)
BAD_TEXT = (
    "\ufffd",
    "\x00",
    "<unk>",
    "[pad",
    "invalid utf-8",
    "mojibake",
    "panicked at",
    "thread 'main' panicked",
)


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_text(path: Path, *, allow_failure_text: bool = False) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationError(f"invalid UTF-8 in {path}: {error}") from error
    lowered = text.lower()
    for token in BAD_TEXT:
        if allow_failure_text and token in {"panicked at", "thread 'main' panicked"}:
            continue
        require(token not in lowered, f"{path}: forbidden text {token!r}")
    return text


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(read_text(path))
    except json.JSONDecodeError as error:
        raise ValidationError(f"malformed JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(read_text(path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValidationError(f"malformed JSONL {path}:{line_no}: {error}") from error
        require(isinstance(value, dict), f"JSONL row is not an object: {path}:{line_no}")
        rows.append(value)
    require(rows, f"empty JSONL artifact: {path}")
    return rows


def validate_artifact_tree(root: Path) -> dict[str, Any]:
    tree = read_json(root / "artifact_tree.json")
    require(tree.get("schema_version") == 1, "artifact_tree schema_version must be 1")
    require(tree.get("artifact_type") == CHECKPOINT_ID, "artifact_tree type mismatch")
    entries = tree.get("files")
    require(isinstance(entries, list), "artifact_tree.files must be a list")
    require(tree.get("file_count") == len(entries), "artifact_tree.file_count mismatch")
    recorded: set[str] = set()
    for index, entry in enumerate(entries):
        require(isinstance(entry, dict), f"artifact_tree.files[{index}] must be an object")
        relative = entry.get("path")
        require(isinstance(relative, str) and relative, f"artifact_tree.files[{index}] path missing")
        relative_path = Path(relative)
        require(not relative_path.is_absolute(), f"artifact_tree path is absolute: {relative}")
        require(".." not in relative_path.parts, f"artifact_tree path escapes root: {relative}")
        require(relative != "artifact_tree.json", "artifact_tree must not hash itself")
        require(relative not in recorded, f"artifact_tree duplicate path: {relative}")
        recorded.add(relative)
        path = (root / relative_path).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValidationError(f"artifact_tree path escapes source root: {relative}") from error
        require(path.is_file() and not path.is_symlink(), f"artifact_tree missing file: {relative}")
        require(entry.get("size_bytes") == path.stat().st_size, f"artifact_tree size drift: {relative}")
        expected = entry.get("sha256")
        require(isinstance(expected, str) and SHA256_RE.fullmatch(expected), f"bad SHA256: {relative}")
        require(file_sha256(path) == expected, f"artifact_tree SHA256 drift: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "artifact_tree.json"
    }
    require(recorded == actual, "artifact_tree does not exactly cover regular files")
    unsigned = dict(tree)
    fingerprint = unsigned.pop("canonical_sha256", None)
    require(isinstance(fingerprint, str) and SHA256_RE.fullmatch(fingerprint), "artifact_tree canonical SHA256 missing")
    require(canonical_sha256(unsigned) == fingerprint, "artifact_tree canonical SHA256 mismatch")
    return {"file_count": len(entries), "canonical_sha256": fingerprint}


def validate_bound_inputs(root: Path, collection: dict[str, Any]) -> dict[str, str]:
    inputs = collection.get("inputs")
    require(isinstance(inputs, dict), "collection.inputs must be an object")
    current = {
        "collector_validator": Path(__file__).resolve(),
        "profile_analyzer": ANALYZER_PATH,
    }
    result: dict[str, str] = {}
    for key, current_path in current.items():
        receipt = inputs.get(key)
        require(isinstance(receipt, dict), f"collection.inputs.{key} missing")
        relative = receipt.get("path")
        require(isinstance(relative, str) and relative, f"collection.inputs.{key}.path missing")
        path = (root / relative).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValidationError(f"collection.inputs.{key}.path escapes source") from error
        expected = receipt.get("sha256")
        require(isinstance(expected, str) and SHA256_RE.fullmatch(expected), f"collection.inputs.{key}.sha256 invalid")
        require(file_sha256(path) == expected, f"bound {key} artifact SHA256 mismatch")
        require(file_sha256(current_path) == expected, f"bound {key} differs from current checked-in source")
        result[key] = expected
    return result


def validate_model(collection: dict[str, Any]) -> dict[str, Any]:
    model = collection.get("model")
    require(isinstance(model, dict), "collection.model must be an object")
    require(model.get("id") == "Qwen/Qwen3.5-4B", "S2 latency lane requires Qwen/Qwen3.5-4B")
    snapshot = model.get("snapshot_path")
    require(isinstance(snapshot, str), "model.snapshot_path missing")
    match = SNAPSHOT_RE.search(snapshot)
    require(match is not None, "model snapshot is not the Qwen3.5-4B HF cache layout")
    require(model.get("revision") == match.group(1), "model revision differs from snapshot path")
    files = model.get("files")
    require(isinstance(files, list) and files, "model.files must be non-empty")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, row in enumerate(files):
        require(isinstance(row, dict), f"model.files[{index}] must be an object")
        name = row.get("path")
        require(isinstance(name, str) and name, f"model.files[{index}].path missing")
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, f"model file path invalid: {name}")
        require(name not in names, f"duplicate model file: {name}")
        names.add(name)
        size = row.get("size_bytes")
        digest = row.get("sha256")
        require(type(size) is int and size > 0, f"model file size invalid: {name}")
        require(isinstance(digest, str) and SHA256_RE.fullmatch(digest), f"model file SHA256 invalid: {name}")
        normalized.append({"path": name, "size_bytes": size, "sha256": digest})
    require("config.json" in names, "model closure lacks config.json")
    require("tokenizer_config.json" in names, "model closure lacks tokenizer_config.json")
    require(any(name.endswith(".safetensors") for name in names), "model closure lacks safetensors weights")
    require(model.get("closure_sha256") == canonical_sha256(normalized), "model closure SHA256 mismatch")
    return {"id": model["id"], "revision": model["revision"], "file_count": len(files), "closure_sha256": model["closure_sha256"]}


def config_entries(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = config.get("entries")
    require(isinstance(entries, list), "effective config entries missing")
    result: dict[str, dict[str, Any]] = {}
    for row in entries:
        require(isinstance(row, dict), "effective config entry is not an object")
        key = row.get("key")
        require(isinstance(key, str) and key and key not in result, f"invalid/duplicate effective config key: {key!r}")
        result[key] = row
    return result


def command_value(tokens: list[str], flag: str) -> str | None:
    indexes = [index for index, token in enumerate(tokens) if token == flag]
    require(len(indexes) <= 1, f"command repeats {flag}")
    if not indexes:
        return None
    index = indexes[0]
    require(index + 1 < len(tokens), f"command {flag} has no value")
    return tokens[index + 1]


def validate_command(directory: Path, entrypoint: str, failure: bool) -> list[str]:
    command_text = read_text(directory / "command.txt").strip()
    tokens = shlex.split(command_text)
    require(tokens and Path(tokens[0]).name == "ferrum", "scenario command does not invoke ferrum")
    require(len(tokens) > 2 and tokens[1] == entrypoint, f"scenario command does not invoke ferrum {entrypoint}")
    require(not any(token.startswith("FERRUM_") for token in tokens), "scenario command uses hidden FERRUM environment")
    require(command_value(tokens, "--backend") == "cuda", "scenario command backend must be cuda")
    require(command_value(tokens, "--profile-detail") == "latency", "scenario command profile detail must be latency")
    require(command_value(tokens, "--profile-sample-rate") == "1.0", "scenario profile sample rate must be 1.0")
    fault = command_value(tokens, "--vnext-diagnostic-fault")
    require(fault == (FAULT_VALUE if failure else None), "scenario diagnostic fault command mismatch")
    for flag, filename in (
        ("--profile-jsonl", "profile.jsonl"),
        ("--effective-config-json", "effective-config.json"),
        ("--decision-trace-jsonl", "decision-trace.jsonl"),
    ):
        value = command_value(tokens, flag)
        require(isinstance(value, str) and Path(value).name == filename, f"scenario {flag} must name {filename}")
    return tokens


def validate_effective_config(path: Path, *, failure: bool) -> dict[str, Any]:
    config = read_json(path)
    require(config.get("backend") == "cuda", f"{path}: effective backend must be cuda")
    entries = config_entries(config)
    detail = entries.get("FERRUM_PROFILE_DETAIL")
    require(isinstance(detail, dict), f"{path}: FERRUM_PROFILE_DETAIL missing")
    require(detail.get("effective_value") == "latency" and detail.get("source") == "cli", f"{path}: latency must have CLI authority")
    diagnostic = entries.get("FERRUM_VNEXT_DIAGNOSTIC_FAULT")
    if failure:
        require(isinstance(diagnostic, dict), f"{path}: diagnostic fault entry missing")
        require(diagnostic.get("effective_value") == FAULT_VALUE and diagnostic.get("source") == "cli", f"{path}: diagnostic fault must have CLI authority")
    else:
        require(diagnostic is None, f"{path}: success scenario unexpectedly enables diagnostic fault")
    return {"env_hash": config.get("env_hash"), "entry_count": len(entries)}


def event_attributes(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("attributes")
    require(isinstance(value, dict), f"event {event.get('event_id')} attributes missing")
    return value


def event_order(event: dict[str, Any]) -> tuple[int, int]:
    attrs = event_attributes(event)
    monotonic = attrs.get("monotonic_nanos_since_run_start")
    return (
        monotonic if type(monotonic) is int and monotonic >= 0 else 2**63 - 1,
        int(event.get("ts_unix_nanos", 2**63 - 1)),
    )


def validate_success_profile(path: Path, entrypoint: str) -> dict[str, Any]:
    try:
        analyzer = profile_analyzer.validate_profile_jsonl(path)
    except profile_analyzer.ValidationError as error:
        raise ValidationError(f"profile analyzer rejected {path}: {error}") from error
    events = read_jsonl(path)
    phase = "actual_run_generation" if entrypoint == "run" else "chat_completions_stream_complete"
    matches = [event for event in events if event.get("entrypoint") == entrypoint and event.get("phase") == phase and event.get("status") == "ok"]
    require(len(matches) == 1, f"{path}: requires exactly one successful {phase} event")
    event = matches[0]
    attrs = event_attributes(event)
    prompt = attrs.get("prompt_token_count")
    completion = attrs.get("completion_token_count")
    total = attrs.get("total_token_count")
    require(type(prompt) is int and prompt > 0, f"{path}: prompt token count missing")
    require(type(completion) is int and completion > 0, f"{path}: completion token count missing")
    require(total == prompt + completion, f"{path}: total token count is not prompt + completion")
    expected_source = "rendered_prompt_and_generated_tokens" if entrypoint == "run" else "usage"
    require(attrs.get("token_count_source") == expected_source, f"{path}: token source mismatch")
    require(attrs.get("profile_detail") == "latency", f"{path}: product event is not latency detail")
    require(type(attrs.get("e2e_duration_us")) is int and attrs["e2e_duration_us"] > 0, f"{path}: E2E duration missing")
    if entrypoint == "serve":
        require(attrs.get("stream") is True, f"{path}: serve sample must be streaming")
    return {
        "analyzer": analyzer,
        "request_id": event.get("request_id"),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "token_source": expected_source,
        "e2e_duration_us": attrs["e2e_duration_us"],
        "ttft_us": attrs.get("ttft_us"),
        "itl_us_avg": attrs.get("itl_us_avg"),
        "clock_conversion_error_ppm": attrs.get("clock_conversion_error_ppm"),
    }


def validate_snapshot(snapshot: Any, context: str) -> dict[str, int]:
    require(isinstance(snapshot, dict), f"{context}: resource snapshot missing")
    keys = (
        "device_capacity_bytes",
        "usable_capacity_bytes",
        "process_claimed_bytes",
        "plan_claimed_bytes",
        "static_bytes",
        "dynamic_resident_bytes",
        "dynamic_free_bytes",
        "pending_growth_bytes",
        "quarantined_bytes",
    )
    for key in keys:
        require(type(snapshot.get(key)) is int and snapshot[key] >= 0, f"{context}: snapshot.{key} invalid")
    require(snapshot["usable_capacity_bytes"] <= snapshot["device_capacity_bytes"], f"{context}: usable capacity exceeds device")
    require(snapshot["process_claimed_bytes"] <= snapshot["usable_capacity_bytes"], f"{context}: process claims exceed usable capacity")
    require(snapshot["plan_claimed_bytes"] <= snapshot["process_claimed_bytes"], f"{context}: plan claims exceed process claims")
    require(snapshot["dynamic_free_bytes"] <= snapshot["dynamic_resident_bytes"], f"{context}: dynamic free exceeds resident")
    minimum_claim = snapshot["static_bytes"] + snapshot["dynamic_resident_bytes"] + snapshot["quarantined_bytes"]
    require(minimum_claim <= snapshot["plan_claimed_bytes"], f"{context}: plan accounting exceeds claims")
    available = snapshot["usable_capacity_bytes"] - snapshot["process_claimed_bytes"] + snapshot["dynamic_free_bytes"]
    require(0 < snapshot["device_capacity_bytes"] and 0 <= available <= snapshot["usable_capacity_bytes"], f"{context}: snapshot availability invalid")
    return {key: int(snapshot[key]) for key in keys} | {"available_bytes": available}


def validate_failure_profile(path: Path, entrypoint: str) -> dict[str, Any]:
    try:
        analyzer = profile_analyzer.validate_profile_jsonl(path)
    except profile_analyzer.ValidationError as error:
        raise ValidationError(f"profile analyzer rejected {path}: {error}") from error
    events = read_jsonl(path)
    first = [
        event
        for event in events
        if event.get("phase") == "vnext.failure_observed"
        and event.get("status") == "failure"
        and isinstance(event.get("error"), dict)
        and event["error"].get("blocking") is True
    ]
    require(len(first) == 1, f"{path}: requires exactly one blocking first failure")
    failure = first[0]
    attrs = event_attributes(failure)
    request_id = failure.get("request_id")
    fingerprint = attrs.get("first_failure_fingerprint")
    require(isinstance(request_id, str) and request_id, f"{path}: first failure request id missing")
    require(isinstance(fingerprint, str) and SHA256_RE.fullmatch(fingerprint), f"{path}: first failure fingerprint invalid")
    require(attrs.get("first_failure_event") is True, f"{path}: first failure marker missing")
    require(attrs.get("failure_domain") == "resource", f"{path}: first failure domain is not resource")
    require(failure["error"].get("kind") == FAULT_ERROR_KIND, f"{path}: diagnostic error kind mismatch")
    identity = attrs.get("execution_identity")
    require(isinstance(identity, dict), f"{path}: canonical execution identity missing")
    require(identity.get("request_id") == request_id, f"{path}: identity request mismatch")
    for key in ("run_id", "plan_id", "plan_hash", "node_id", "operation_id", "provider_id", "device_id", "span_id"):
        require(isinstance(identity.get(key), str) and identity[key], f"{path}: identity.{key} missing")
    for key in ("sequence", "resource_pool_id", "active_sequence_slot", "admission_generation", "activation_epoch"):
        require(type(identity.get(key)) is int and identity[key] >= 0, f"{path}: identity.{key} missing")
    require(isinstance(identity.get("resource_pool_identity_fingerprint"), str) and SHA256_RE.fullmatch(identity["resource_pool_identity_fingerprint"]), f"{path}: resource pool fingerprint invalid")
    require(isinstance(identity.get("runtime_implementation_fingerprint"), str) and SHA256_RE.fullmatch(identity["runtime_implementation_fingerprint"]), f"{path}: runtime fingerprint invalid")
    backend = failure.get("backend_detail")
    require(isinstance(backend, dict) and backend.get("backend_type") == "cuda", f"{path}: CUDA backend identity missing")
    snapshot = validate_snapshot(attrs.get("plan_runtime_resource_snapshot"), str(path))
    resource = failure.get("resource")
    require(isinstance(resource, dict), f"{path}: first failure resource event missing")
    require(resource.get("action") == "reject", f"{path}: first failure resource action is not reject")
    require(resource.get("owner_kind") == "resource_pool", f"{path}: first failure resource owner kind mismatch")
    require(resource.get("owner_id") == f"resource-pool:{identity['resource_pool_id']}", f"{path}: first failure resource owner id mismatch")
    require(resource.get("capacity") == snapshot["usable_capacity_bytes"], f"{path}: resource capacity differs from snapshot")
    require(resource.get("before") == resource.get("after") == snapshot["available_bytes"], f"{path}: reject mutated or forged available capacity")
    same_request = [event for event in events if event.get("request_id") == request_id]
    submitted = [event for event in same_request if event.get("phase") == "vnext.operation_submitted"]
    require(submitted, f"{path}: no real operation submission precedes failure")
    matching_submission = None
    for event in submitted:
        submitted_identity = event_attributes(event).get("execution_identity")
        if not isinstance(submitted_identity, dict):
            continue
        if all(submitted_identity.get(key) == identity.get(key) for key in ("run_id", "request_id", "plan_id", "plan_hash", "node_id", "operation_id", "provider_id", "device_id", "resource_pool_id")):
            matching_submission = event
            break
    require(matching_submission is not None, f"{path}: first failure identity is not owned by a submitted operation")
    require(event_order(matching_submission) < event_order(failure), f"{path}: failure precedes operation submission")
    terminals: dict[str, dict[str, Any]] = {}
    for phase in ("vnext.sequence_aborted", "vnext.request_failed"):
        matches = [event for event in same_request if event.get("phase") == phase]
        require(len(matches) == 1, f"{path}: requires exactly one {phase}")
        terminal = matches[0]
        terminal_attrs = event_attributes(terminal)
        require(terminal_attrs.get("terminal_failure_event") is True, f"{path}: {phase} terminal marker missing")
        require(terminal_attrs.get("first_failure_fingerprint") == fingerprint, f"{path}: {phase} fingerprint mismatch")
        require(event_order(terminal) > event_order(failure), f"{path}: {phase} precedes first failure")
        terminals[phase] = terminal
    require(event_order(terminals["vnext.sequence_aborted"]) < event_order(terminals["vnext.request_failed"]), f"{path}: request failed before sequence aborted")
    for event in events:
        event_attrs = event.get("attributes")
        if not isinstance(event_attrs, dict):
            continue
        for key in ("resource_leak_count", "resource_underflow_count", "resource_double_release_count"):
            require(event_attrs.get(key, 0) == 0, f"{path}: {key} is nonzero")
    return {
        "analyzer": analyzer,
        "request_id": request_id,
        "first_failure_phase": failure.get("phase"),
        "first_failure_kind": failure["error"]["kind"],
        "first_failure_status": failure.get("status"),
        "first_failure_fingerprint": fingerprint,
        "identity": {key: identity.get(key) for key in ("request_id", "sequence", "plan_id", "node_id", "operation_id", "resource_pool_id", "provider_id", "device_id")},
        "snapshot": snapshot,
    }


def parse_sse(path: Path) -> dict[str, Any]:
    text = read_text(path)
    payloads: list[dict[str, Any]] = []
    done = 0
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        value = line.removeprefix("data: ")
        if value == "[DONE]":
            done += 1
        else:
            try:
                payload = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValidationError(f"malformed SSE JSON in {path}: {error}") from error
            require(isinstance(payload, dict), f"non-object SSE payload in {path}")
            payloads.append(payload)
    require(done == 1, f"{path}: expected exactly one [DONE], observed {done}")
    return {"payloads": payloads, "done_count": done}


def validate_scenario(root: Path, name: str) -> dict[str, Any]:
    entrypoint, failure = SCENARIOS[name]
    directory = root / name
    require(directory.is_dir() and not directory.is_symlink(), f"missing scenario directory: {name}")
    exit_code = int(read_text(directory / "exit_code").strip())
    if entrypoint == "run":
        require((exit_code != 0) is failure, f"{name}: process exit does not match expected outcome")
    else:
        require(exit_code == 0, f"{name}: server did not shut down cleanly")
    validate_command(directory, entrypoint, failure)
    config = validate_effective_config(directory / "effective-config.json", failure=failure)
    read_jsonl(directory / "decision-trace.jsonl")
    read_text(directory / "stderr.log")
    if entrypoint == "run":
        stdout = read_text(directory / "stdout.log", allow_failure_text=False)
        require(bool(stdout.strip()), f"{name}: stdout is empty")
    result: dict[str, Any] = {"entrypoint": entrypoint, "failure_injected": failure, "effective_config": config}
    if failure:
        result["failure"] = validate_failure_profile(directory / "profile.jsonl", entrypoint)
    else:
        result["success"] = validate_success_profile(directory / "profile.jsonl", entrypoint)
    if entrypoint == "serve":
        request = read_json(directory / "request.json")
        require(request.get("stream") is True, f"{name}: request must stream")
        options = request.get("stream_options")
        require(isinstance(options, dict) and options.get("include_usage") is True, f"{name}: stream usage missing")
        status = int(read_text(directory / "http_status").strip())
        require(status == 200, f"{name}: HTTP status must be 200")
        sse = parse_sse(directory / "response.sse")
        if failure:
            require(any(isinstance(row.get("error"), dict) for row in sse["payloads"]), f"{name}: diagnostic fault did not reach the HTTP client")
            recovery_request = read_json(directory / "recovery.request.json")
            require(recovery_request.get("stream") is True, f"{name}: recovery request must stream")
            require(int(read_text(directory / "recovery.http_status").strip()) == 200, f"{name}: recovery HTTP status must be 200")
            recovery = parse_sse(directory / "recovery.response.sse")
            require(not any(isinstance(row.get("error"), dict) for row in recovery["payloads"]), f"{name}: recovery request failed")
            require(any(isinstance(row.get("usage"), dict) for row in recovery["payloads"]), f"{name}: recovery stream lacks usage")
            success = validate_success_profile(directory / "profile.jsonl", entrypoint)
            require(success["request_id"] != result["failure"]["request_id"], f"{name}: recovery reused failed request identity")
            result["recovery"] = success
        else:
            usage = [row["usage"] for row in sse["payloads"] if isinstance(row.get("usage"), dict)]
            require(len(usage) == 1, f"{name}: stream must contain exactly one usage payload")
            measured = result["success"]
            require(usage[0].get("prompt_tokens") == measured["prompt_tokens"], f"{name}: profile/usage prompt tokens differ")
            require(usage[0].get("completion_tokens") == measured["completion_tokens"], f"{name}: profile/usage completion tokens differ")
            require(usage[0].get("total_tokens") == measured["total_tokens"], f"{name}: profile/usage total tokens differ")
    return result


def scalar_stats(values: list[float]) -> dict[str, Any]:
    require(len(values) >= 2 and all(math.isfinite(value) and value > 0 for value in values), "overhead samples must contain at least two positive finite values")
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values)
    return {"values": values, "n": len(values), "mean": mean, "median": statistics.median(values), "sample_stddev": deviation, "cv": deviation / mean}


def validate_overhead(root: Path) -> dict[str, Any]:
    directory = root / "profile-overhead"
    report = read_json(directory / "report.json")
    require(report.get("schema_version") == 1, "overhead report schema_version must be 1")
    require(report.get("comparison") == "ABBA-BAAB", "overhead comparison must be ABBA-BAAB")
    require(report.get("slot_order") == list(OVERHEAD_SLOT_ORDER), "overhead slot order mismatch")
    grouped: dict[str, list[float]] = {"off": [], "latency": []}
    slots = report.get("slots")
    require(isinstance(slots, list) and len(slots) == len(OVERHEAD_SLOT_ORDER), "overhead slots missing")
    for expected, row in zip(OVERHEAD_SLOT_ORDER, slots):
        require(isinstance(row, dict) and row.get("slot") == expected, f"overhead slot identity mismatch: {expected}")
        expected_mode = "latency" if expected.startswith("latency") else "off"
        require(row.get("mode") == expected_mode, f"overhead slot mode mismatch: {expected}")
        bench = read_json(directory / expected / "bench.json")
        repeats = bench.get("repeat_metrics")
        require(isinstance(repeats, list) and len(repeats) >= 3, f"overhead {expected} requires at least three repeats")
        values = [repeat.get("output_throughput_tps") for repeat in repeats if isinstance(repeat, dict)]
        require(len(values) == len(repeats) and all(type(value) in (int, float) for value in values), f"overhead {expected} throughput missing")
        throughput = statistics.fmean(float(value) for value in values)
        require(math.isclose(float(row.get("output_throughput_tps", -1)), throughput, rel_tol=1e-12, abs_tol=1e-12), f"overhead {expected} report drifted from bench")
        grouped[expected_mode].append(throughput)
    off = scalar_stats(grouped["off"])
    latency = scalar_stats(grouped["latency"])
    mean_overhead = (off["mean"] - latency["mean"]) / off["mean"]
    median_overhead = (off["median"] - latency["median"]) / off["median"]
    require(math.isclose(report.get("mean_overhead_fraction", math.inf), mean_overhead, rel_tol=1e-12, abs_tol=1e-12), "overhead mean fraction mismatch")
    require(math.isclose(report.get("median_overhead_fraction", math.inf), median_overhead, rel_tol=1e-12, abs_tol=1e-12), "overhead median fraction mismatch")
    stable = off["cv"] <= 0.05 and latency["cv"] <= 0.05
    target_met = mean_overhead <= 0.05 and median_overhead <= 0.05
    classification = "stable_target_met" if stable and target_met else ("target_miss" if stable else "noisy")
    require(report.get("classification") == classification, "overhead classification mismatch")
    require(report.get("blocking") is False, "latency overhead is report-only for S2")
    return {"off": off, "latency": latency, "mean_overhead_fraction": mean_overhead, "median_overhead_fraction": median_overhead, "target_met": target_met, "classification": classification, "blocking": False}


def validate_source(source: Path, expected_git_sha: str | None) -> dict[str, Any]:
    require(source.is_dir() and not source.is_symlink(), f"source root is not a real directory: {source}")
    root = source.resolve(strict=True)
    artifact_tree = validate_artifact_tree(root)
    collection = read_json(root / "collection.json")
    require(collection.get("schema_version") == 1, "collection schema_version must be 1")
    require(collection.get("artifact_type") == CHECKPOINT_ID, "collection artifact type mismatch")
    git_sha = collection.get("git_sha")
    git_tree = collection.get("git_tree")
    require(isinstance(git_sha, str) and GIT_SHA_RE.fullmatch(git_sha), "collection git_sha invalid")
    require(isinstance(git_tree, str) and GIT_SHA_RE.fullmatch(git_tree), "collection git_tree invalid")
    require(collection.get("dirty_status") == {"is_dirty": False, "status_short": []}, "collection source checkout is dirty")
    if expected_git_sha is not None:
        require(git_sha == expected_git_sha, "artifact git SHA differs from validation candidate")
    binary = collection.get("binary")
    require(isinstance(binary, dict), "collection.binary missing")
    require(isinstance(binary.get("sha256"), str) and SHA256_RE.fullmatch(binary["sha256"]), "binary SHA256 invalid")
    hardware = collection.get("hardware")
    require(isinstance(hardware, dict), "collection.hardware missing")
    require(hardware.get("gpu_count") == 1, "S2 latency lane requires exactly one GPU")
    require("RTX 4090" in str(hardware.get("name", "")), "S2 latency lane requires RTX 4090")
    require(isinstance(hardware.get("uuid"), str) and hardware["uuid"].startswith("GPU-"), "GPU UUID missing")
    require(type(hardware.get("memory_total_mib")) is int and hardware["memory_total_mib"] >= 24000, "GPU memory identity invalid")
    inputs = validate_bound_inputs(root, collection)
    model = validate_model(collection)
    expected_scenarios = collection.get("scenarios")
    require(expected_scenarios == list(SCENARIOS), "collection scenario set/order mismatch")
    scenarios = {name: validate_scenario(root, name) for name in SCENARIOS}
    overhead = validate_overhead(root)
    return {
        "git_sha": git_sha,
        "git_tree": git_tree,
        "binary_sha256": binary["sha256"],
        "hardware": hardware,
        "model": model,
        "inputs": inputs,
        "artifact_tree": artifact_tree,
        "scenarios": scenarios,
        "overhead": overhead,
    }


def run_checkpoint(source: Path, out: Path, expected_git_sha: str | None) -> int:
    started_at = iso_now()
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_id": CHECKPOINT_ID,
        "scope": ["S2/M1/CUDA/latency", "S2/M1/CUDA/first-failure"],
        "full_s2": False,
        "source_root": str(source.resolve(strict=False)),
        "artifact_dir": str(out.resolve(strict=False)),
        "started_at": started_at,
    }
    try:
        evidence = validate_source(source, expected_git_sha)
    except (OSError, ValueError, ValidationError) as error:
        manifest.update({"status": "fail", "finished_at": iso_now(), "duration_sec": time.monotonic() - started, "pass_line": None, "evidence": None, "error": str(error)})
        write_json(out / "manifest.json", manifest)
        print(f"{FAIL_PREFIX}: {out}: {error}", file=sys.stderr)
        return 1
    pass_line = f"{PASS_PREFIX}: {out}"
    manifest.update({"status": "pass", "finished_at": iso_now(), "duration_sec": time.monotonic() - started, "pass_line": pass_line, "evidence": evidence, "error": None})
    write_json(out / "manifest.json", manifest)
    print(pass_line)
    return 0


def timing_attributes(entrypoint: str) -> dict[str, Any]:
    return {
        "profile_detail": "latency",
        "diagnostic_only": False,
        "prompt_token_count": 5,
        "completion_token_count": 2,
        "output_token_count": 2,
        "total_token_count": 7,
        "token_count_source": "rendered_prompt_and_generated_tokens" if entrypoint == "run" else "usage",
        "e2e_duration_us": 5_000,
        "stream": entrypoint == "serve",
        "engine_token_clock_source": "rust_std_instant",
        "engine_token_wall_anchor_unix_nanos": 1_000_000_000,
        "clock_conversion_max_error_nanos": 1_000,
        "engine_token_commit_nanos_since_request_start": [1_000_000, 2_000_000],
        "engine_token_commit_count": 2,
        "itl_source": "engine_token_commit",
        "itl_interval_count": 1,
        "itl_nanos": [1_000_000],
        "itl_us_avg": 1_000,
        "ttft_us": 1_000,
        "engine_decode_ready_nanos_since_request_start": 1_200_000,
        "engine_decode_wall_nanos": 800_000,
        "clock_conversion_error_ppm": 1_250,
        "decode_wall_timing_eligible": True,
    }


def base_event(request_id: str, entrypoint: str, phase: str, sequence: int, *, kind: str = "instant", status: str = "ok", duration_us: int | None = None, attributes: dict[str, Any] | None = None, error: dict[str, Any] | None = None, resource: dict[str, Any] | None = None) -> dict[str, Any]:
    event = {
        "schema_version": 1,
        "ts_unix_nanos": 1_000_000_000 + sequence,
        "event_id": f"evt-{request_id}-{sequence}",
        "request_id": request_id,
        "correlation_id": request_id,
        "entrypoint": entrypoint,
        "backend": "actual",
        "runtime_preset_hash": "preset.fixture",
        "phase": phase,
        "event_kind": kind,
        "timestamp": "2026-08-03T00:00:00Z",
        "status": status,
        "shape": {"execution_sequence": sequence},
        "backend_detail": {"backend_type": "cuda", "backend_device": "cuda:0"},
        "attributes": {"profile_detail": "latency", "diagnostic_only": False, "run_id": f"run.{entrypoint}.fixture", "monotonic_nanos_since_run_start": sequence, **(attributes or {})},
    }
    if duration_us is not None:
        event["duration_us"] = duration_us
    if error is not None:
        event["error"] = error
    if resource is not None:
        event["resource"] = resource
    return event


def identity(request_id: str, sequence: int) -> dict[str, Any]:
    return {
        "version": {"major": 1, "minor": 0},
        "run_id": "run.fixture",
        "request_id": request_id,
        "sequence": sequence,
        "plan_id": "plan/fixture",
        "plan_hash": "1" * 64,
        "frame_id": 1,
        "node_invocation_id": 1,
        "node_id": "node/fixture",
        "operation_id": "operation/fixture",
        "provider_id": "provider/cuda/fixture",
        "device_id": "cuda:0",
        "resource_pool_id": 7,
        "resource_pool_identity_fingerprint": "2" * 64,
        "provisioning_run_id": "run.pool.fixture",
        "provisioning_request_id": "request.pool.fixture",
        "transaction_id": "transaction.pool.fixture",
        "active_sequence_slot": 0,
        "admission_generation": 1,
        "activation_epoch": 1,
        "runtime_implementation_fingerprint": "3" * 64,
        "active_sequence_fingerprint": "4" * 64,
        "completed_sequence_fingerprint": None,
        "aborted_sequence_fingerprint": None,
        "resource_id": None,
        "resource_generation": None,
        "resource_batch_fingerprint": None,
        "span_id": f"span/{sequence}",
        "parent_span_id": "span/parent",
        "async_links": [],
    }


def fixture_success_profile(entrypoint: str, request_id: str) -> list[dict[str, Any]]:
    phase = "actual_run_generation" if entrypoint == "run" else "chat_completions_stream_complete"
    return [base_event(request_id, entrypoint, phase, 1, kind="timed_span", duration_us=5_000, attributes=timing_attributes(entrypoint))]


def fixture_failure_events(entrypoint: str, request_id: str, *, recovery: bool) -> list[dict[str, Any]]:
    fingerprint = "5" * 64
    snapshot = {
        "device_capacity_bytes": 1_000,
        "usable_capacity_bytes": 900,
        "process_claimed_bytes": 700,
        "plan_claimed_bytes": 700,
        "static_bytes": 400,
        "dynamic_resident_bytes": 300,
        "dynamic_free_bytes": 200,
        "pending_growth_bytes": 0,
        "quarantined_bytes": 0,
    }
    submitted_identity = identity(request_id, 2)
    failure_identity = identity(request_id, 3)
    common = {"execution_trace_source": "vnext", "execution_request_id": request_id}
    submitted = base_event(request_id, entrypoint, "vnext.operation_submitted", 2, attributes={**common, "execution_identity": submitted_identity})
    failure = base_event(
        request_id,
        entrypoint,
        "vnext.failure_observed",
        3,
        kind="error",
        status="failure",
        attributes={**common, "execution_identity": failure_identity, "first_failure_event": True, "first_failure_fingerprint": fingerprint, "failure_domain": "resource", "plan_runtime_resource_snapshot": snapshot},
        error={"kind": FAULT_ERROR_KIND, "message": "injected resource failure after operation submission", "blocking": True},
        resource={"owner_kind": "resource_pool", "owner_id": "resource-pool:7", "resource_kind": "plan_runtime_memory", "action": "reject", "before": 400, "after": 400, "capacity": 900, "reason": "injected resource failure after operation submission"},
    )
    terminals = []
    for sequence, phase in ((4, "vnext.sequence_aborted"), (5, "vnext.request_failed")):
        terminals.append(base_event(request_id, entrypoint, phase, sequence, kind="error", status="failure", attributes={**common, "execution_identity": identity(request_id, sequence), "terminal_failure_event": True, "first_failure_fingerprint": fingerprint}, error={"kind": "request_aborted_after_failure", "message": f"request terminated after failure {fingerprint}", "blocking": False}))
    events = [base_event(request_id, entrypoint, "request_execution_started", 1, kind="timed_span", duration_us=1), submitted, failure, *terminals]
    if recovery:
        events.extend(fixture_success_profile(entrypoint, f"{request_id}.recovery"))
    return events


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def fixture_effective_config(failure: bool) -> dict[str, Any]:
    entries = [{"key": "FERRUM_PROFILE_DETAIL", "effective_value": "latency", "source": "cli", "affects": ["diagnostics"]}]
    if failure:
        entries.append({"key": "FERRUM_VNEXT_DIAGNOSTIC_FAULT", "effective_value": FAULT_VALUE, "source": "cli", "affects": ["diagnostics"]})
    return {"schema_version": 1, "backend": "cuda", "env_hash": "sha256:" + "6" * 64, "entries": entries}


def write_fixture_tree(root: Path) -> None:
    entries = [
        {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != "artifact_tree.json"
    ]
    tree = {"schema_version": 1, "artifact_type": CHECKPOINT_ID, "file_count": len(entries), "files": entries}
    tree["canonical_sha256"] = canonical_sha256(tree)
    write_json(root / "artifact_tree.json", tree)


def create_fixture(root: Path, *, overhead_scale: float = 0.98) -> None:
    root.mkdir(parents=True)
    inputs = root / "inputs"
    inputs.mkdir()
    validator_copy = inputs / Path(__file__).name
    analyzer_copy = inputs / ANALYZER_PATH.name
    shutil.copyfile(Path(__file__).resolve(), validator_copy)
    shutil.copyfile(ANALYZER_PATH, analyzer_copy)
    for name, (entrypoint, failure) in SCENARIOS.items():
        directory = root / name
        directory.mkdir()
        argv = ["/workspace/target/release/ferrum", entrypoint, "/workspace/hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/" + "a" * 40, "--backend", "cuda", "--profile-detail", "latency", "--profile-sample-rate", "1.0", "--profile-jsonl", str(directory / "profile.jsonl"), "--effective-config-json", str(directory / "effective-config.json"), "--decision-trace-jsonl", str(directory / "decision-trace.jsonl")]
        if failure:
            argv.extend(["--vnext-diagnostic-fault", FAULT_VALUE])
        write_text(directory / "command.txt", shlex.join(argv) + "\n")
        write_json(directory / "effective-config.json", fixture_effective_config(failure))
        write_jsonl(directory / "decision-trace.jsonl", [{"selected": "cuda", "source": "cli"}])
        write_text(directory / "stderr.log", "fixture completed without panic\n")
        if entrypoint == "run":
            write_text(directory / "stdout.log", '{"type":"completion","text":"Paris"}\n' if not failure else '{"type":"error","kind":"resource_exhausted"}\n')
            write_text(directory / "exit_code", "1\n" if failure else "0\n")
            events = fixture_failure_events(entrypoint, f"request.{name}", recovery=False) if failure else fixture_success_profile(entrypoint, f"request.{name}")
        else:
            write_text(directory / "stdout.log", "server fixture\n")
            write_text(directory / "exit_code", "0\n")
            request = {"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "Capital of France?"}], "max_tokens": 2, "stream": True, "stream_options": {"include_usage": True}}
            write_json(directory / "request.json", request)
            write_text(directory / "http_status", "200\n")
            if failure:
                write_text(directory / "response.sse", 'data: {"error":{"message":"injected resource failure"}}\n\ndata: [DONE]\n\n')
                write_json(directory / "recovery.request.json", request)
                write_text(directory / "recovery.http_status", "200\n")
                write_text(directory / "recovery.response.sse", 'data: {"choices":[{"delta":{"content":"Paris"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\ndata: [DONE]\n\n')
                events = fixture_failure_events(entrypoint, f"request.{name}", recovery=True)
            else:
                write_text(directory / "response.sse", 'data: {"choices":[{"delta":{"content":"Paris"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\ndata: [DONE]\n\n')
                events = fixture_success_profile(entrypoint, f"request.{name}")
        write_jsonl(directory / "profile.jsonl", events)
    overhead = root / "profile-overhead"
    slots = []
    for index, slot in enumerate(OVERHEAD_SLOT_ORDER):
        mode = "latency" if slot.startswith("latency") else "off"
        base = 100.0 + (index % 2) * 0.2
        throughput = base * (overhead_scale if mode == "latency" else 1.0)
        repeats = [throughput - 0.1, throughput, throughput + 0.1]
        write_json(overhead / slot / "bench.json", {"repeat_metrics": [{"output_throughput_tps": value} for value in repeats]})
        slots.append({"slot": slot, "mode": mode, "output_throughput_tps": statistics.fmean(repeats)})
    off_values = [row["output_throughput_tps"] for row in slots if row["mode"] == "off"]
    latency_values = [row["output_throughput_tps"] for row in slots if row["mode"] == "latency"]
    off = scalar_stats(off_values)
    latency = scalar_stats(latency_values)
    mean_overhead = (off["mean"] - latency["mean"]) / off["mean"]
    median_overhead = (off["median"] - latency["median"]) / off["median"]
    stable = off["cv"] <= 0.05 and latency["cv"] <= 0.05
    target = mean_overhead <= 0.05 and median_overhead <= 0.05
    write_json(overhead / "report.json", {"schema_version": 1, "comparison": "ABBA-BAAB", "slot_order": list(OVERHEAD_SLOT_ORDER), "slots": slots, "mean_overhead_fraction": mean_overhead, "median_overhead_fraction": median_overhead, "classification": "stable_target_met" if stable and target else ("target_miss" if stable else "noisy"), "blocking": False})
    model_files = [
        {"path": "config.json", "size_bytes": 100, "sha256": "7" * 64},
        {"path": "tokenizer_config.json", "size_bytes": 200, "sha256": "8" * 64},
        {"path": "model-00001-of-00001.safetensors", "size_bytes": 1_000, "sha256": "9" * 64},
    ]
    collection = {
        "schema_version": 1,
        "artifact_type": CHECKPOINT_ID,
        "git_sha": "a" * 40,
        "git_tree": "b" * 40,
        "dirty_status": {"is_dirty": False, "status_short": []},
        "binary": {"path": "/workspace/target/release/ferrum", "sha256": "c" * 64},
        "hardware": {"gpu_count": 1, "name": "NVIDIA GeForce RTX 4090", "uuid": "GPU-fixture", "memory_total_mib": 24564, "driver_version": "fixture"},
        "model": {"id": "Qwen/Qwen3.5-4B", "snapshot_path": "/workspace/hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/" + "a" * 40, "revision": "a" * 40, "files": model_files, "closure_sha256": canonical_sha256(model_files)},
        "inputs": {
            "collector_validator": {"path": validator_copy.relative_to(root).as_posix(), "sha256": file_sha256(validator_copy)},
            "profile_analyzer": {"path": analyzer_copy.relative_to(root).as_posix(), "sha256": file_sha256(analyzer_copy)},
        },
        "scenarios": list(SCENARIOS),
    }
    write_json(root / "collection.json", collection)
    write_fixture_tree(root)


def mutate_jsonl(path: Path, predicate: Callable[[dict[str, Any]], bool], mutate: Callable[[dict[str, Any]], None]) -> None:
    rows = read_jsonl(path)
    matches = [row for row in rows if predicate(row)]
    require(len(matches) == 1, f"self-test mutation target is not unique: {path}")
    mutate(matches[0])
    write_jsonl(path, rows)


def self_test_process(source: Path, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(Path(__file__).resolve()), "--source", str(source), "--expected-git-sha", "a" * 40, "--out", str(out)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def self_test() -> int:
    mutations: list[tuple[str, Callable[[Path], None]]] = [
        ("clock-error", lambda root: mutate_jsonl(root / "run-success/profile.jsonl", lambda row: row.get("phase") == "actual_run_generation", lambda row: row["attributes"].update({"clock_conversion_max_error_nanos": 10_000, "clock_conversion_error_ppm": 12_500}))),
        ("missing-operation-identity", lambda root: mutate_jsonl(root / "run-failure/profile.jsonl", lambda row: row.get("phase") == "vnext.failure_observed", lambda row: row["attributes"]["execution_identity"].pop("operation_id"))),
        ("forged-snapshot", lambda root: mutate_jsonl(root / "serve-failure/profile.jsonl", lambda row: row.get("phase") == "vnext.failure_observed", lambda row: row["attributes"]["plan_runtime_resource_snapshot"].update({"process_claimed_bytes": 901}))),
        ("duplicate-first-failure", lambda root: write_jsonl(root / "run-failure/profile.jsonl", read_jsonl(root / "run-failure/profile.jsonl") + [copy.deepcopy(next(row for row in read_jsonl(root / "run-failure/profile.jsonl") if row.get("phase") == "vnext.failure_observed"))])),
        ("stale-analyzer", lambda root: write_text(root / "inputs/analyze_ferrum_profile.py", "# stale\n")),
    ]
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-s2-latency-") as temporary:
        temporary_root = Path(temporary)
        baseline = temporary_root / "baseline"
        create_fixture(baseline)
        baseline_out = temporary_root / "baseline-out"
        proc = self_test_process(baseline, baseline_out)
        require(proc.returncode == 0, proc.stderr or proc.stdout)
        require(f"{PASS_PREFIX}: {baseline_out}" in proc.stdout.splitlines(), "baseline missing exact PASS line")
        target_miss = temporary_root / "target-miss"
        create_fixture(target_miss, overhead_scale=0.90)
        target_proc = self_test_process(target_miss, temporary_root / "target-miss-out")
        require(target_proc.returncode == 0, target_proc.stderr or target_proc.stdout)
        target_manifest = read_json(temporary_root / "target-miss-out/manifest.json")
        require(target_manifest["evidence"]["overhead"]["classification"] == "target_miss", "report-only target miss did not survive gate")
        for name, mutate in mutations:
            source = temporary_root / name
            create_fixture(source)
            mutate(source)
            write_fixture_tree(source)
            out = temporary_root / f"{name}-out"
            rejected = self_test_process(source, out)
            require(rejected.returncode != 0, f"mutation unexpectedly passed: {name}")
            require(read_json(out / "manifest.json").get("status") == "fail", f"mutation lacks fail manifest: {name}")
    print(SELFTEST_PASS_LINE)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "--artifact-root", dest="source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-git-sha")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.source is not None or args.out is not None or args.expected_git_sha is not None:
            parser.error("--self-test cannot be combined with source/out/SHA")
    elif args.source is None or args.out is None:
        parser.error("--source and --out are required")
    elif args.expected_git_sha is not None and GIT_SHA_RE.fullmatch(args.expected_git_sha) is None:
        parser.error("--expected-git-sha must be 40 lowercase hex characters")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        try:
            return self_test()
        except (OSError, ValueError, ValidationError) as error:
            print(f"{SELFTEST_PASS_LINE.replace(' PASS', ' FAIL')}: {error}", file=sys.stderr)
            return 1
    return run_checkpoint(args.source, args.out, args.expected_git_sha)


if __name__ == "__main__":
    raise SystemExit(main())
