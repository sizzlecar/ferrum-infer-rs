#!/usr/bin/env python3
"""Validate the Qwen3.5-4B CUDA S2 Unicode-stream/disconnect sentinel."""

from __future__ import annotations

import argparse
import codecs
import copy
import hashlib
import json
import math
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_PREFIX = "FERRUM RUNTIME VNEXT S2 STREAM DISCONNECT PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT S2 STREAM DISCONNECT FAIL"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT S2 STREAM DISCONNECT SELFTEST PASS"
CHECKPOINT_ID = "runtime-vnext-s2-stream-disconnect-sentinel"
MODEL = "Qwen/Qwen3.5-4B"
UNICODE_NAME = "m1_s2_stream_equivalence_unicode"
DISCONNECT_NAME = "m1_s2_disconnect_release"
SCENARIO_NAMES = (UNICODE_NAME, DISCONNECT_NAME)
SCENARIO_TYPES = ("serve_stream_equivalence_unicode", "serve_disconnect_release")
CATEGORIES = ("chinese", "emoji", "combining")
EXPECTED_CATEGORY_COUNTS = {"chinese": 7, "emoji": 7, "combining": 6}
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DETERMINISTIC_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": 9271,
    "stop": [],
}
BAD_TEXT = (
    "<unk>",
    "[pad",
    "invalid utf-8",
    "mojibake",
    "cuda out of memory",
    "kv cache overflow",
    "segmentation fault",
    "panicked at",
)
QUIESCENT_FIELDS = ("queue_depth", "active_prefill", "active_decode", "current_batch_size")


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_text(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationError(f"invalid UTF-8 in {path}: {error}") from error
    require("\x00" not in text, f"NUL byte in {path}")
    require("\ufffd" not in text, f"Unicode replacement character in {path}")
    return text


def assert_clean_output(label: str, text: str) -> None:
    require("\x00" not in text, f"{label}: NUL byte")
    require("\ufffd" not in text, f"{label}: Unicode replacement character")
    lowered = text.lower()
    for token in BAD_TEXT:
        require(token not in lowered, f"{label}: forbidden text {token!r}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(read_text(path))
    except json.JSONDecodeError as error:
        raise ValidationError(f"malformed JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(read_text(path).splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValidationError(f"malformed JSONL {path}:{line_no}: {error}") from error
        require(isinstance(row, dict), f"JSONL row is not an object: {path}:{line_no}")
        rows.append(row)
    require(rows, f"empty JSONL artifact: {path}")
    return rows


def validate_self_hash(value: dict[str, Any], label: str) -> None:
    require(
        value.get("canonical_sha256_scope") == "document_without_canonical_sha256_fields",
        f"{label}: canonical SHA scope mismatch",
    )
    saved = value.get("canonical_sha256")
    require(isinstance(saved, str) and SHA256_RE.fullmatch(saved) is not None, f"{label}: bad SHA")
    unhashed = copy.deepcopy(value)
    unhashed.pop("canonical_sha256", None)
    require(json_sha256(unhashed) == saved, f"{label}: canonical self hash mismatch")


def resolve_member(source: Path, recorded: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label}: missing path")
    raw = Path(value)
    require(".." not in raw.parts, f"{label}: parent traversal")
    if raw.is_absolute():
        try:
            relative = raw.relative_to(recorded)
        except ValueError as error:
            raise ValidationError(f"{label}: path is outside recorded root: {raw}") from error
    else:
        relative = raw
    candidate = (source / relative).resolve(strict=False)
    try:
        candidate.relative_to(source)
    except ValueError as error:
        raise ValidationError(f"{label}: path escapes source root") from error
    require(candidate.exists(), f"{label}: path does not exist: {candidate}")
    return candidate


def expected_tree_paths(source: Path) -> set[str]:
    return {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "artifact_tree.json"
    }


def validate_artifact_tree(source: Path, recorded: Path) -> dict[str, Any]:
    for path in source.rglob("*"):
        require(not path.is_symlink(), f"artifact contains symlink: {path}")
    tree = read_json(source / "artifact_tree.json")
    validate_self_hash(tree, "artifact_tree.json")
    require(tree.get("schema_version") == 1, "artifact tree schema mismatch")
    require(tree.get("artifact_root") == str(recorded), "artifact tree recorded root mismatch")
    rows = tree.get("files")
    require(isinstance(rows, list), "artifact tree files must be a list")
    require(tree.get("file_count") == len(rows), "artifact tree file_count mismatch")
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"artifact tree row {index} is not an object")
        require(set(row) == {"path", "size", "sha256"}, f"artifact tree row {index} fields")
        relative = row.get("path")
        require(isinstance(relative, str) and relative, f"artifact tree row {index} path")
        raw = Path(relative)
        require(not raw.is_absolute() and ".." not in raw.parts, f"artifact tree escape: {relative}")
        require(relative not in indexed and relative != "artifact_tree.json", f"duplicate tree path: {relative}")
        candidate = (source / raw).resolve(strict=False)
        try:
            candidate.relative_to(source)
        except ValueError as error:
            raise ValidationError(f"artifact tree path escapes: {relative}") from error
        require(candidate.is_file() and not candidate.is_symlink(), f"tree file missing: {relative}")
        require(row.get("size") == candidate.stat().st_size, f"tree size mismatch: {relative}")
        require(row.get("sha256") == file_sha256(candidate), f"tree SHA mismatch: {relative}")
        if candidate.suffix.lower() in {".json", ".jsonl", ".log", ".sse", ".txt"}:
            read_text(candidate)
        indexed[relative] = row
    actual = expected_tree_paths(source)
    require(set(indexed) == actual, f"artifact tree coverage mismatch: missing={sorted(actual-set(indexed))}, extra={sorted(set(indexed)-actual)}")
    return {"file_count": len(indexed), "sha256": file_sha256(source / "artifact_tree.json")}


def finite_number(value: Any, label: str, minimum: float = 0.0) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= minimum,
        f"{label}: invalid number {value!r}",
    )
    return float(value)


def validate_usage(value: Any, label: str) -> dict[str, int]:
    require(isinstance(value, dict), f"{label}: usage missing")
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        require(isinstance(item, int) and not isinstance(item, bool) and item >= 0, f"{label}.{key}")
        result[key] = item
    require(result["completion_tokens"] > 0, f"{label}: no completion tokens")
    require(result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"], f"{label}: inconsistent usage")
    return result


def parse_sync_response(path: Path, expected_text: str) -> tuple[dict[str, int], str]:
    response = read_json(path)
    require(response.get("object") == "chat.completion", f"{path}: object mismatch")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, f"{path}: choices mismatch")
    choice = choices[0]
    require(isinstance(choice, dict) and choice.get("finish_reason") == "stop", f"{path}: finish mismatch")
    message = choice.get("message")
    require(isinstance(message, dict), f"{path}: message missing")
    content = message.get("content")
    require(content == expected_text, f"{path}: assistant bytes differ from expected")
    assert_clean_output(str(path), content)
    response_id = response.get("id")
    require(isinstance(response_id, str) and response_id, f"{path}: response id missing")
    return validate_usage(response.get("usage"), str(path)), response_id


def incremental_wire_evidence(raw: bytes) -> tuple[str, int]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    fragments: list[str] = []
    splits = 0
    try:
        for byte in raw:
            fragment = decoder.decode(bytes([byte]), final=False)
            buffered, _ = decoder.getstate()
            if buffered:
                splits += 1
            if fragment:
                fragments.append(fragment)
        tail = decoder.decode(b"", final=True)
    except UnicodeDecodeError as error:
        raise ValidationError(f"SSE wire is not incrementally valid UTF-8: {error}") from error
    if tail:
        fragments.append(tail)
    decoded = "".join(fragments)
    require(decoded.encode("utf-8") == raw, "incremental SSE round trip mismatch")
    return decoded, splits


def parse_sse(raw: bytes, label: str) -> dict[str, Any]:
    body, splits = incremental_wire_evidence(raw)
    assert_clean_output(label, body)
    done = 0
    malformed = 0
    output: list[str] = []
    finishes: list[str] = []
    usages: list[dict[str, int]] = []
    content_deltas = 0
    response_ids: set[str] = set()
    errors: list[Any] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ").strip()
        if payload == "[DONE]":
            done += 1
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            malformed += 1
            continue
        require(isinstance(event, dict), f"{label}: SSE event is not an object")
        if isinstance(event.get("id"), str):
            response_ids.add(event["id"])
        if event.get("error") is not None:
            errors.append(event["error"])
        if event.get("usage") is not None:
            usages.append(validate_usage(event["usage"], f"{label}.usage"))
        choices = event.get("choices", [])
        require(isinstance(choices, list), f"{label}: SSE choices is not a list")
        for choice in choices:
            require(isinstance(choice, dict), f"{label}: SSE choice is not an object")
            finish = choice.get("finish_reason")
            if finish is not None:
                finishes.append(str(finish))
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            fragment = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
            if fragment:
                require(isinstance(fragment, str), f"{label}: delta is not text")
                output.append(fragment)
                content_deltas += 1
    return {
        "body": body,
        "split_boundary_count": splits,
        "done_count": done,
        "malformed_json": malformed,
        "output_text": "".join(output),
        "finish_reasons": finishes,
        "usages": usages,
        "content_delta_count": content_deltas,
        "response_ids": response_ids,
        "errors": errors,
    }


def unicode_expected(ordinal: int) -> tuple[str, dict[str, Any], str]:
    category = CATEGORIES[ordinal % len(CATEGORIES)]
    if category == "chinese":
        value = f"\u4f60\u597d\uff0cFerrum\uff0c\u7f16\u53f7 {ordinal:03d}"
    elif category == "emoji":
        value = f"Ferrum \U0001f525\U0001f680 {ordinal:03d}"
    else:
        value = f"Cafe\u0301 Ferrum {ordinal:03d}"
    expected = {"category": category, "ordinal": ordinal, "value": value}
    return category, expected, json.dumps(expected, ensure_ascii=False, separators=(",", ":"))


def expected_unicode_payload(ordinal: int) -> tuple[dict[str, Any], dict[str, Any]]:
    category, expected, expected_text = unicode_expected(ordinal)
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return the following JSON object exactly, with no markdown or extra text. "
                    f"EXACT_JSON:{expected_text}"
                ),
            }
        ],
        "max_tokens": 1024,
        **DETERMINISTIC_SAMPLING,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"unicode_equivalence_{ordinal:03d}",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "const": category},
                        "ordinal": {"type": "integer", "const": ordinal},
                        "value": {"type": "string", "const": expected["value"]},
                    },
                    "required": ["category", "ordinal", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "metadata": {
            "ferrum_scenario": "stream_equivalence_unicode",
            "ferrum_case": f"{category}-{ordinal:03d}",
        },
    }
    return payload, expected


def validate_unicode(source: Path, recorded: Path, summary_row: dict[str, Any]) -> dict[str, Any]:
    root = source / UNICODE_NAME
    result = read_json(root / "result.json")
    require(result == summary_row, "Unicode summary/result mismatch")
    require(result.get("status") == "pass" and result.get("type") == SCENARIO_TYPES[0], "Unicode scenario failed")
    require(resolve_member(source, recorded, result.get("artifact"), "Unicode artifact") == root / "result.json", "Unicode artifact path mismatch")
    rows = result.get("cases")
    require(isinstance(rows, list) and len(rows) == 20, "Unicode result must contain 20 cases")
    categories: Counter[str] = Counter()
    pair_hashes: set[str] = set()
    response_ids: set[str] = set()
    for ordinal, aggregate in enumerate(rows):
        require(isinstance(aggregate, dict), f"Unicode aggregate {ordinal} is not an object")
        category, expected, expected_text = unicode_expected(ordinal)
        case_root = root / f"{ordinal:03d}-{category}"
        expected_files = {
            "sync.request.json",
            "sync.response.json",
            "stream.request.json",
            "stream.response.sse",
            "stream.wire.json",
            "result.json",
        }
        actual_files = {path.name for path in case_root.iterdir() if path.is_file() and not path.is_symlink()}
        require(actual_files == expected_files, f"Unicode case {ordinal} file set mismatch")
        case = read_json(case_root / "result.json")
        validate_self_hash(case, f"Unicode case {ordinal}")
        require(case == aggregate, f"Unicode aggregate {ordinal} differs from case result")
        sync_payload, expected_object = expected_unicode_payload(ordinal)
        stream_payload = copy.deepcopy(sync_payload)
        stream_payload["stream"] = True
        stream_payload["stream_options"] = {"include_usage": True}
        require(read_json(case_root / "sync.request.json") == sync_payload, f"Unicode sync request {ordinal} mismatch")
        require(read_json(case_root / "stream.request.json") == stream_payload, f"Unicode stream request {ordinal} mismatch")
        require(case.get("expected") == expected_object, f"Unicode expected object {ordinal} mismatch")
        require(case.get("expected_sha256") == json_sha256(expected_object), f"Unicode expected SHA {ordinal}")
        require(case.get("pair_payload_sha256") == json_sha256(sync_payload), f"Unicode pair SHA {ordinal}")
        require(case.get("sync_request_sha256") == json_sha256(sync_payload), f"Unicode sync request SHA {ordinal}")
        require(case.get("stream_request_sha256") == json_sha256(stream_payload), f"Unicode stream request SHA {ordinal}")
        pair_hashes.add(case["pair_payload_sha256"])
        sync_raw = (case_root / "sync.response.json").read_bytes()
        try:
            sync_text_raw = sync_raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValidationError(f"Unicode sync response {ordinal} invalid UTF-8: {error}") from error
        assert_clean_output(f"Unicode sync response {ordinal}", sync_text_raw)
        require(case.get("sync_response_sha256") == hashlib.sha256(sync_raw).hexdigest(), f"Unicode sync response SHA {ordinal}")
        sync_usage, sync_id = parse_sync_response(case_root / "sync.response.json", expected_text)
        response_ids.add(sync_id)
        wire_raw = (case_root / "stream.response.sse").read_bytes()
        parsed = parse_sse(wire_raw, f"Unicode stream {ordinal}")
        wire = read_json(case_root / "stream.wire.json")
        wire_sha = hashlib.sha256(wire_raw).hexdigest()
        require(wire == case.get("wire"), f"Unicode wire aggregate {ordinal} mismatch")
        require(wire.get("read_size") == 1 and wire.get("read_count") == len(wire_raw), f"Unicode wire reads {ordinal}")
        require(wire.get("wire_size_bytes") == len(wire_raw), f"Unicode wire size {ordinal}")
        require(wire.get("wire_sha256") == wire_sha, f"Unicode wire SHA {ordinal}")
        require(wire.get("decoded_sha256") == wire_sha, f"Unicode decoded SHA {ordinal}")
        require(wire.get("split_boundary_count") == parsed["split_boundary_count"] > 0, f"Unicode split evidence {ordinal}")
        require(case.get("stream_response_sha256") == wire_sha, f"Unicode case wire SHA {ordinal}")
        require(parsed["done_count"] == 1 and parsed["malformed_json"] == 0, f"Unicode SSE framing {ordinal}")
        require(parsed["content_delta_count"] > 0 and not parsed["errors"], f"Unicode SSE content {ordinal}")
        require(parsed["output_text"] == expected_text, f"Unicode stream bytes {ordinal} differ")
        require(parsed["finish_reasons"] == ["stop"], f"Unicode stream finish {ordinal}")
        require(parsed["usages"] == [sync_usage] == [case.get("usage")], f"Unicode stream usage {ordinal}")
        require(len(parsed["response_ids"]) == 1, f"Unicode stream response id {ordinal}")
        response_ids.update(parsed["response_ids"])
        require(case.get("ordinal") == ordinal and case.get("category") == category, f"Unicode identity {ordinal}")
        require(case.get("finish_reason") == "stop", f"Unicode finish aggregate {ordinal}")
        categories[category] += 1
    require(dict(categories) == EXPECTED_CATEGORY_COUNTS, "Unicode category counts mismatch")
    require(len(pair_hashes) == 20, "Unicode payloads are not unique")
    require(result.get("case_count") == 20, "Unicode case_count mismatch")
    require(result.get("category_counts") == EXPECTED_CATEGORY_COUNTS, "Unicode summary categories mismatch")
    for key in ("exact_content_matches", "exact_finish_matches", "exact_usage_matches", "multibyte_split_cases"):
        require(result.get(key) == 20, f"Unicode summary {key} mismatch")
    return {"case_count": 20, "category_counts": dict(categories), "response_id_count": len(response_ids)}


def validate_quiescent(value: dict[str, Any], label: str) -> None:
    terminal = value.get("terminal")
    require(isinstance(terminal, dict), f"{label}: terminal missing")
    admission = terminal.get("admission")
    require(isinstance(admission, dict), f"{label}: admission missing")
    require(admission.get("effective_max_concurrent") == 1, f"{label}: capacity mismatch")
    for key in QUIESCENT_FIELDS:
        require(admission.get(key) == 0, f"{label}: {key} is not quiescent")


def validate_resource_balance(rows: list[dict[str, Any]], request_id: str) -> None:
    states: dict[tuple[str, str, str], dict[str, int]] = {}
    opened = 0
    closed = 0
    for event in rows:
        if event.get("request_id") != request_id or not isinstance(event.get("resource"), dict):
            continue
        resource = event["resource"]
        key = (str(resource.get("owner_kind")), str(resource.get("owner_id")), str(resource.get("resource_kind")))
        action = resource.get("action")
        state = states.setdefault(key, {"reserved": 0, "committed": 0, "released": 0, "rolled_back": 0})
        if action == "request_open":
            opened += 1
        elif action == "request_close":
            closed += 1
            for state_key, current in states.items():
                if state_key[:2] == key[:2]:
                    require(current["reserved"] - current["released"] - current["rolled_back"] == 0, f"resource reserve leak at close: {state_key}")
                    require(current["committed"] - current["released"] - current["rolled_back"] == 0, f"resource commit leak at close: {state_key}")
        elif action in {"reserve", "commit", "release", "rollback"}:
            amount = resource.get("amount")
            require(isinstance(amount, int) and amount > 0, f"resource amount invalid: {resource}")
            if action == "reserve":
                before = state["reserved"] - state["released"] - state["rolled_back"]
                require(resource.get("before") == before, f"resource reserve before mismatch: {resource}")
                state["reserved"] += amount
                after = state["reserved"] - state["released"] - state["rolled_back"]
            elif action == "commit":
                before = state["committed"] - state["released"] - state["rolled_back"]
                require(resource.get("before") == before, f"resource commit before mismatch: {resource}")
                state["committed"] += amount
                after = state["committed"] - state["released"] - state["rolled_back"]
            elif action == "release":
                before = state["committed"] - state["released"] - state["rolled_back"]
                require(amount <= before and resource.get("before") == before, f"resource release underflow: {resource}")
                state["released"] += amount
                after = state["committed"] - state["released"] - state["rolled_back"]
            else:
                before = state["reserved"] - state["released"] - state["rolled_back"]
                require(amount <= before and resource.get("before") == before, f"resource rollback underflow: {resource}")
                state["rolled_back"] += amount
                after = state["reserved"] - state["released"] - state["rolled_back"]
            require(resource.get("after") == after, f"resource after mismatch: {resource}")
    require(opened == 1 and closed == 1, f"disconnect request open/close count mismatch: {opened}/{closed}")
    for key, state in states.items():
        require(state["reserved"] - state["released"] - state["rolled_back"] == 0, f"resource reserve leak: {key}")
        require(state["committed"] - state["released"] - state["rolled_back"] == 0, f"resource commit leak: {key}")


def validate_trace(rows: list[dict[str, Any]], request_ids: set[str], disconnect_id: str) -> dict[str, Any]:
    for request_id in request_ids:
        require(any(row.get("request_id") == request_id for row in rows), f"request dump has no scheduler trace: {request_id}")
    detected = [(index, row) for index, row in enumerate(rows) if row.get("request_id") == disconnect_id and row.get("phase") == "engine_client_disconnect_detected"]
    released = [(index, row) for index, row in enumerate(rows) if row.get("request_id") == disconnect_id and row.get("phase") == "engine_client_disconnect_released"]
    failed = [row for row in rows if row.get("request_id") == disconnect_id and row.get("phase") == "engine_client_disconnect_release_failed"]
    require(len(detected) == 1 and len(released) == 1 and not failed, "disconnect terminal trace cardinality mismatch")
    detected_index, detected_event = detected[0]
    released_index, released_event = released[0]
    require(detected_index < released_index, "disconnect release precedes detection")
    require(detected_event.get("status") == "ok" and released_event.get("status") == "ok", "disconnect trace status")
    require(released_event.get("error") is None, "disconnect release trace contains error")
    require(detected_event.get("correlation_id") == disconnect_id and released_event.get("correlation_id") == disconnect_id, "disconnect trace correlation mismatch")
    detected_shape = detected_event.get("shape")
    released_shape = released_event.get("shape")
    released_attrs = released_event.get("attributes")
    require(isinstance(detected_shape, dict) and isinstance(released_shape, dict) and isinstance(released_attrs, dict), "disconnect trace fields missing")
    start_iteration = detected_shape.get("scheduler_iteration")
    require(released_shape.get("detected_scheduler_iteration") == start_iteration, "disconnect detected iteration mismatch")
    terminal_iteration = released_shape.get("terminal_scheduler_iteration")
    tick_delta = released_shape.get("scheduler_tick_delta")
    require(isinstance(start_iteration, int) and isinstance(terminal_iteration, int) and isinstance(tick_delta, int), "disconnect tick values invalid")
    require(terminal_iteration - start_iteration == tick_delta and 0 <= tick_delta <= 2, "disconnect exceeded two scheduler ticks")
    require(released_attrs.get("terminal_state") == "released", "disconnect terminal state mismatch")
    require(released_attrs.get("scheduler_cancel_result") == "cancelled", "disconnect scheduler cancel mismatch")
    start_nanos = detected_event.get("ts_unix_nanos")
    terminal_nanos = released_event.get("ts_unix_nanos")
    require(isinstance(start_nanos, int) and isinstance(terminal_nanos, int), "disconnect timestamps invalid")
    wall_sec = (terminal_nanos - start_nanos) / 1_000_000_000
    require(0 <= wall_sec <= 5.0, "disconnect trace exceeded five seconds")
    validate_resource_balance(rows, disconnect_id)
    return {"scheduler_tick_delta": tick_delta, "trace_release_wall_sec": wall_sec, "released_row": released_index}


def partition_request_dump_bundles(root: Path) -> tuple[Path, list[Path]]:
    all_bundles = sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink())
    startup_bundles = [path for path in all_bundles if path.name.startswith("serve-startup-")]
    request_bundles = [path for path in all_bundles if not path.name.startswith("serve-startup-")]
    require(len(startup_bundles) == 1, f"expected one serve startup bundle, found {len(startup_bundles)}")
    require(len(request_bundles) == 42, f"expected 42 scenario request bundles, found {len(request_bundles)}")
    return startup_bundles[0], request_bundles


def validate_startup_bundle(bundle: Path) -> str:
    request = read_json(bundle / "request.json")
    request_id = request.get("request_id")
    require(request_id == bundle.name, f"startup bundle identity mismatch: {bundle}")
    require(request.get("schema_version") == 1 and request.get("entrypoint") == "serve", "startup bundle schema/entrypoint mismatch")
    require(request.get("model") == MODEL and request.get("backend") == "actual", "startup bundle model/backend mismatch")
    require(request.get("actual_model_smoke") is True and request.get("sanitized") is True, "startup bundle provenance mismatch")
    require("http" not in request, "startup bundle must not masquerade as an HTTP scenario request")
    for json_path in bundle.glob("*.json"):
        value = read_json(json_path)
        if value.get("request_id") is not None:
            require(value.get("request_id") == request_id, f"startup bundle member request_id mismatch: {json_path}")
    backend = read_json(bundle / "backend_selection.json")
    require(backend.get("backend") == "actual" and backend.get("model") == MODEL, "startup bundle backend mismatch")
    bad_output = read_json(bundle / "bad_output_scan.json")
    require(bad_output.get("bad_output") is False and bad_output.get("bad_text_count") == 0, "startup bundle bad-output scan failed")
    return request_id


def validate_request_dumps(source: Path, trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    root = source / "observability" / "serve" / "request_dump"
    startup_bundle, bundles = partition_request_dump_bundles(root)
    startup_request_id = validate_startup_bundle(startup_bundle)
    keys: dict[tuple[Any, ...], str] = {}
    request_ids: set[str] = set()
    for bundle in bundles:
        request = read_json(bundle / "request.json")
        request_id = request.get("request_id")
        require(request_id == bundle.name and request_id not in request_ids, f"request bundle identity mismatch: {bundle}")
        request_ids.add(request_id)
        require(request.get("backend") == "actual" and request.get("actual_model_smoke") is True, f"request bundle is not actual: {request_id}")
        body = request.get("http", {}).get("body")
        require(isinstance(body, dict) and body.get("model") == MODEL, f"request bundle body mismatch: {request_id}")
        metadata = body.get("metadata")
        require(isinstance(metadata, dict), f"request bundle metadata missing: {request_id}")
        scenario = metadata.get("ferrum_scenario")
        if scenario == "stream_equivalence_unicode":
            case = metadata.get("ferrum_case")
            require(isinstance(case, str), f"Unicode bundle case missing: {request_id}")
            key = (scenario, case, body.get("stream") is True)
        elif scenario == "disconnect_release":
            require(metadata.get("ferrum_disconnect_probe") is True, "disconnect bundle marker missing")
            key = (scenario, metadata.get("ferrum_marker"))
        elif scenario == "disconnect_release_followup":
            key = (scenario, metadata.get("ferrum_marker"))
        else:
            raise ValidationError(f"unexpected request bundle scenario: {scenario!r}")
        require(key not in keys, f"duplicate request bundle key: {key}")
        keys[key] = request_id
        for json_path in bundle.glob("*.json"):
            value = read_json(json_path)
            if value.get("request_id") is not None:
                require(value.get("request_id") == request_id, f"bundle member request_id mismatch: {json_path}")
        backend = read_json(bundle / "backend_selection.json")
        require(backend.get("backend") == "actual" and backend.get("model") == MODEL, f"bundle backend mismatch: {request_id}")
    expected_unicode_keys = {
        ("stream_equivalence_unicode", f"{CATEGORIES[ordinal % 3]}-{ordinal:03d}", stream)
        for ordinal in range(20)
        for stream in (False, True)
    }
    require(expected_unicode_keys <= set(keys), "Unicode request dump matrix incomplete")
    marker = "m1-s2-disconnect-release"
    disconnect_key = ("disconnect_release", marker)
    followup_key = ("disconnect_release_followup", marker)
    require(set(keys) == expected_unicode_keys | {disconnect_key, followup_key}, "request dump key set mismatch")
    disconnect_id = keys[disconnect_key]
    followup_id = keys[followup_key]
    require(disconnect_id != followup_id, "disconnect and followup share request id")
    trace = validate_trace(trace_rows, request_ids, disconnect_id)
    followup_rows = [index for index, row in enumerate(trace_rows) if row.get("request_id") == followup_id]
    require(followup_rows and min(followup_rows) > trace["released_row"], "followup began before disconnect release")
    return {
        "request_count": len(request_ids),
        "startup_request_id": startup_request_id,
        "disconnect_id": disconnect_id,
        "followup_id": followup_id,
        **trace,
    }


def validate_disconnect(source: Path, recorded: Path, summary_row: dict[str, Any]) -> dict[str, Any]:
    root = source / DISCONNECT_NAME
    result = read_json(root / "result.json")
    require(result == summary_row, "disconnect summary/result mismatch")
    require(result.get("status") == "pass" and result.get("type") == SCENARIO_TYPES[1], "disconnect scenario failed")
    require(resolve_member(source, recorded, result.get("artifact"), "disconnect artifact") == root / "result.json", "disconnect artifact path mismatch")
    expected_files = {
        "disconnect.request.json",
        "disconnect.partial.sse",
        "disconnect.observed.json",
        "followup.request.json",
        "followup.response.json",
        "health.before.json",
        "health.released.json",
        "health.after.json",
        "result.json",
    }
    require({path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()} == expected_files, "disconnect file set mismatch")
    marker = "m1-s2-disconnect-release"
    disconnect_request = read_json(root / "disconnect.request.json")
    followup_request = read_json(root / "followup.request.json")
    require(disconnect_request.get("model") == MODEL and followup_request.get("model") == MODEL, "disconnect model mismatch")
    require(disconnect_request.get("max_tokens") == followup_request.get("max_tokens") == result.get("max_tokens") == 1024, "disconnect capacity mismatch")
    require(disconnect_request.get("stream") is True and disconnect_request.get("stream_options") == {"include_usage": True}, "disconnect stream contract mismatch")
    require(disconnect_request.get("metadata") == {"ferrum_scenario": "disconnect_release", "ferrum_disconnect_probe": True, "ferrum_marker": marker}, "disconnect metadata mismatch")
    require(followup_request.get("metadata") == {"ferrum_scenario": "disconnect_release_followup", "ferrum_marker": marker}, "followup metadata mismatch")
    require(followup_request.get("seed") == 9271 and followup_request.get("chat_template_kwargs") == {"enable_thinking": False}, "followup deterministic contract mismatch")
    require(result.get("request_fingerprints") == {"disconnect": json_sha256(disconnect_request), "followup": json_sha256(followup_request)}, "disconnect request fingerprints mismatch")
    before = read_json(root / "health.before.json")
    released = read_json(root / "health.released.json")
    after = read_json(root / "health.after.json")
    validate_quiescent(before, "health.before")
    validate_quiescent(released, "health.released")
    validate_quiescent(after, "health.after")
    release_elapsed = finite_number(released.get("elapsed_sec"), "release elapsed")
    require(release_elapsed <= 5.0 and result.get("release_elapsed_sec") == released.get("elapsed_sec"), "disconnect wall release mismatch")
    require(result.get("release_timeout_sec") == 5.0 and result.get("scheduler_tick_limit") == 2, "disconnect limits mismatch")
    require(result.get("scheduler_trace_required") is True and result.get("effective_max_concurrent") == 1, "disconnect trace/capacity contract mismatch")
    observed = read_json(root / "disconnect.observed.json")
    require(observed.get("http_status") == 200 and isinstance(observed.get("first_output_event"), dict), "disconnect did not observe output")
    active = observed.get("active_health", {}).get("admission")
    require(isinstance(active, dict) and active.get("effective_max_concurrent") == 1, "disconnect active health missing")
    require(sum(int(active.get(key, 0)) for key in QUIESCENT_FIELDS) > 0, "disconnect was not active at first output")
    partial = parse_sse((root / "disconnect.partial.sse").read_bytes(), "disconnect partial SSE")
    require(partial["content_delta_count"] > 0 and partial["done_count"] == 0, "disconnect partial SSE is not an interrupted stream")
    expected_followup = json.dumps({"marker": marker, "status": "released"}, ensure_ascii=False, separators=(",", ":"))
    followup_usage, _ = parse_sync_response(root / "followup.response.json", expected_followup)
    require(result.get("same_capacity_followup") is True and result.get("followup_usage") == followup_usage, "disconnect followup result mismatch")
    return {"release_elapsed_sec": release_elapsed, "followup_usage": followup_usage}


def validate_identity(source: Path, expected_git_sha: str | None) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    summary = read_json(source / "summary.json")
    require(summary.get("schema_version") == 1 and summary.get("status") == "pass", "summary status mismatch")
    require(summary.get("backend") == "cuda" and summary.get("model") == MODEL, "summary model/backend mismatch")
    git_sha = summary.get("git_sha")
    require(isinstance(git_sha, str) and GIT_SHA_RE.fullmatch(git_sha), "summary git SHA invalid")
    if expected_git_sha is not None:
        require(git_sha == expected_git_sha, "artifact git SHA differs from expected SHA")
    require(summary.get("dirty_status") == {"is_dirty": False, "status_short": []}, "artifact source was dirty")
    require(summary.get("failed") == 0 and summary.get("skipped") == 0, "summary contains failed/skipped scenarios")
    require(summary.get("scenario_count") == summary.get("manifest_scenario_count") == 2, "summary scenario count mismatch")
    require(summary.get("requested_scenarios") == [], "checkpoint must run full manifest")
    require(summary.get("selected_scenarios") == list(SCENARIO_NAMES), "selected scenarios mismatch")
    artifact_dir = summary.get("artifact_dir")
    require(isinstance(artifact_dir, str) and Path(artifact_dir).is_absolute(), "recorded artifact root invalid")
    recorded = Path(artifact_dir)
    require(summary.get("pass_line") == f"BACKEND REGRESSION SMOKE PASS: {recorded}", "runner PASS line mismatch")
    receipt = read_json(source / "execution_receipt.json")
    validate_self_hash(receipt, "execution_receipt.json")
    require(receipt.get("mode") == "start" and receipt.get("git_sha") == git_sha, "execution receipt identity mismatch")
    require(receipt.get("dirty_status") == summary.get("dirty_status"), "receipt dirty status mismatch")
    require(receipt.get("backend") == "cuda" and receipt.get("model") == MODEL, "receipt model/backend mismatch")
    require(receipt.get("selected_scenarios") == list(SCENARIO_NAMES), "receipt scenarios mismatch")
    require(receipt.get("scenario_count") == 2 and receipt.get("failed") == receipt.get("skipped") == 0, "receipt outcome mismatch")
    binary_sha = receipt.get("binary_sha256")
    require(isinstance(binary_sha, str) and SHA256_RE.fullmatch(binary_sha), "binary SHA invalid")
    hardware = receipt.get("hardware")
    require(isinstance(hardware, dict) and hardware.get("returncode") == 0, "hardware receipt failed")
    gpu_rows = [line for line in str(hardware.get("stdout", "")).splitlines() if line.strip()]
    require(len(gpu_rows) == 1 and "RTX 4090" in gpu_rows[0], "checkpoint requires one RTX 4090")
    argv = receipt.get("server_argv")
    require(isinstance(argv, list), "server argv missing")
    for flag, value in (("--backend", "cuda"), ("--max-num-seqs", "1")):
        require(flag in argv and argv.index(flag) + 1 < len(argv) and argv[argv.index(flag) + 1] == value, f"server argv missing {flag} {value}")
    for flag in ("--profile-jsonl", "--memory-profile-jsonl", "--scheduler-trace-jsonl", "--request-dump-dir"):
        require(flag in argv, f"server argv missing observability flag {flag}")
    require(receipt.get("server_returncode") in (0, -15), "server return code mismatch")
    child_env = receipt.get("child_env")
    require(isinstance(child_env, dict) and not any(str(key).startswith("FERRUM_") for key in child_env), "hidden FERRUM env reached product server")
    summary_receipt = summary.get("execution_receipt")
    require(isinstance(summary_receipt, dict), "summary receipt binding missing")
    require(resolve_member(source, recorded, summary_receipt.get("artifact"), "summary receipt") == source / "execution_receipt.json", "summary receipt path mismatch")
    require(summary_receipt.get("artifact_sha256") == file_sha256(source / "execution_receipt.json"), "summary receipt file SHA mismatch")
    for key in ("canonical_sha256", "runner_sha256", "manifest_sha256", "binary_sha256"):
        require(summary_receipt.get(key) == receipt.get(key), f"summary/receipt {key} mismatch")
    inputs = receipt.get("input_artifacts")
    require(isinstance(inputs, dict), "receipt input artifacts missing")
    for name, filename, sha_key in (("runner", "run_scenarios.py", "runner_sha256"), ("manifest", "scenario_manifest.json", "manifest_sha256")):
        item = inputs.get(name)
        require(isinstance(item, dict), f"receipt input {name} missing")
        path = resolve_member(source, recorded, item.get("path"), f"receipt input {name}")
        require(path == source / "inputs" / filename, f"receipt input {name} path mismatch")
        require(item.get("sha256") == file_sha256(path) == receipt.get(sha_key), f"receipt input {name} SHA mismatch")
    manifest = read_json(source / "inputs" / "scenario_manifest.json")
    require(manifest.get("model") == MODEL and manifest.get("backend") == "cuda", "input manifest model/backend mismatch")
    require(manifest.get("goal_scope") == {"full_s2": False, "model_matrix_c09_complete": False, "model_matrix_c17_complete": False}, "input manifest overclaims S2/C09/C17")
    manifest_scenarios = manifest.get("scenarios")
    require(isinstance(manifest_scenarios, list) and [(row.get("name"), row.get("type")) for row in manifest_scenarios if isinstance(row, dict)] == list(zip(SCENARIO_NAMES, SCENARIO_TYPES)), "input manifest scenario matrix mismatch")
    evidence = receipt.get("evidence_files")
    require(isinstance(evidence, dict), "execution evidence bindings missing")
    for label, filename in (("effective_config", "server.effective_config.json"), ("decision_trace", "server.decision_trace.jsonl"), ("server_log", "server.log"), ("health_before", "server.health.json"), ("health_after", "server.health.after.json")):
        item = evidence.get(label)
        require(isinstance(item, dict), f"execution evidence missing: {label}")
        path = resolve_member(source, recorded, item.get("path"), f"execution evidence {label}")
        require(path == source / filename and item.get("size") == path.stat().st_size and item.get("sha256") == file_sha256(path), f"execution evidence binding mismatch: {label}")
    assert_clean_output("server.log", read_text(source / "server.log"))
    return summary, recorded, receipt


def validate_observability(source: Path, recorded: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    obs = summary.get("observability")
    require(isinstance(obs, dict) and obs.get("enabled") is True, "observability summary missing")
    scheduler_paths = obs.get("scheduler_trace_paths")
    dump_paths = obs.get("request_dump_dirs")
    require(isinstance(scheduler_paths, list) and len(scheduler_paths) == 1, "scheduler trace path cardinality")
    require(isinstance(dump_paths, list) and len(dump_paths) == 1, "request dump path cardinality")
    scheduler_path = resolve_member(source, recorded, scheduler_paths[0], "scheduler trace")
    dump_path = resolve_member(source, recorded, dump_paths[0], "request dump")
    require(scheduler_path == source / "observability" / "serve" / "scheduler_trace.jsonl", "scheduler trace path mismatch")
    require(dump_path == source / "observability" / "serve" / "request_dump", "request dump path mismatch")
    rows = read_jsonl(scheduler_path)
    require(any(str(row.get("phase", "")).startswith("vnext.") for row in rows), "trace does not prove vNext execution")
    return rows


def validate_source(source: Path, expected_git_sha: str | None) -> dict[str, Any]:
    source = source.resolve(strict=True)
    summary, recorded, receipt = validate_identity(source, expected_git_sha)
    scenarios = summary.get("scenarios")
    require(isinstance(scenarios, list) and len(scenarios) == 2, "summary scenarios missing")
    require([row.get("name") for row in scenarios if isinstance(row, dict)] == list(SCENARIO_NAMES), "summary scenario order mismatch")
    unicode = validate_unicode(source, recorded, scenarios[0])
    disconnect = validate_disconnect(source, recorded, scenarios[1])
    trace_rows = validate_observability(source, recorded, summary)
    request_trace = validate_request_dumps(source, trace_rows)
    tree = validate_artifact_tree(source, recorded)
    return {
        "git_sha": summary["git_sha"],
        "binary_sha256": receipt["binary_sha256"],
        "backend": "cuda",
        "model": MODEL,
        "hardware": receipt["hardware"]["stdout"].strip(),
        "full_s2": False,
        "model_matrix_c09_complete": False,
        "model_matrix_c17_complete": False,
        "unicode": unicode,
        "disconnect": disconnect,
        "request_trace": request_trace,
        "artifact_tree": tree,
        "source_sha256": {"summary.json": file_sha256(source / "summary.json")},
    }


def self_test() -> None:
    expected = {"category": "emoji", "ordinal": 1, "value": "Ferrum \U0001f525\U0001f680 001"}
    text = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
    usage = {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    chunks = [
        {"id": "stream-1", "choices": [{"delta": {"content": text}, "finish_reason": None}]},
        {"id": "stream-1", "choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"id": "stream-1", "choices": [], "usage": usage},
    ]
    raw = ("".join(f"data: {json.dumps(row, ensure_ascii=False)}\n\n" for row in chunks) + "data: [DONE]\n\n").encode("utf-8")
    parsed = parse_sse(raw, "selftest SSE")
    require(parsed["output_text"] == text, "selftest SSE output mismatch")
    require(parsed["done_count"] == 1 and parsed["usages"] == [usage], "selftest SSE terminal mismatch")
    require(parsed["split_boundary_count"] > 0, "selftest did not split multibyte UTF-8")
    bad = raw[:-3] + b"\xf0\x9f"
    try:
        parse_sse(bad, "selftest truncated SSE")
    except ValidationError as error:
        require("UTF-8" in str(error), "selftest UTF-8 mutation failed for wrong reason")
    else:
        raise ValidationError("selftest truncated UTF-8 unexpectedly passed")
    request_id = "11111111-1111-1111-1111-111111111111"
    now = 1_000_000_000
    rows = [
        {"request_id": request_id, "correlation_id": request_id, "phase": "engine_request_open", "resource": {"owner_kind": "request", "owner_id": request_id, "resource_kind": "request_slot", "action": "request_open"}},
        {"request_id": request_id, "correlation_id": request_id, "phase": "engine_request_slot_reserve", "resource": {"owner_kind": "request", "owner_id": request_id, "resource_kind": "request_slot", "action": "reserve", "amount": 1, "before": 0, "after": 1}},
        {"request_id": request_id, "correlation_id": request_id, "phase": "engine_request_slot_commit", "resource": {"owner_kind": "request", "owner_id": request_id, "resource_kind": "request_slot", "action": "commit", "amount": 1, "before": 0, "after": 1}},
        {"request_id": request_id, "correlation_id": request_id, "phase": "engine_client_disconnect_detected", "status": "ok", "ts_unix_nanos": now, "shape": {"scheduler_iteration": 7}, "attributes": {}},
        {"request_id": request_id, "correlation_id": request_id, "phase": "engine_request_slot_release", "resource": {"owner_kind": "request", "owner_id": request_id, "resource_kind": "request_slot", "action": "release", "amount": 1, "before": 1, "after": 0}},
        {"request_id": request_id, "correlation_id": request_id, "phase": "engine_request_close", "resource": {"owner_kind": "request", "owner_id": request_id, "resource_kind": "request_slot", "action": "request_close"}},
        {"request_id": request_id, "correlation_id": request_id, "phase": "engine_client_disconnect_released", "status": "ok", "error": None, "ts_unix_nanos": now + 1_000_000, "shape": {"detected_scheduler_iteration": 7, "terminal_scheduler_iteration": 9, "scheduler_tick_delta": 2}, "attributes": {"terminal_state": "released", "scheduler_cancel_result": "cancelled"}},
    ]
    validated = validate_trace(rows, {request_id}, request_id)
    require(validated["scheduler_tick_delta"] == 2, "selftest trace tick mismatch")
    mutated = copy.deepcopy(rows)
    mutated[-1]["shape"]["terminal_scheduler_iteration"] = 10
    mutated[-1]["shape"]["scheduler_tick_delta"] = 3
    try:
        validate_trace(mutated, {request_id}, request_id)
    except ValidationError as error:
        require("two scheduler ticks" in str(error), "selftest tick mutation failed for wrong reason")
    else:
        raise ValidationError("selftest three-tick disconnect unexpectedly passed")
    with tempfile.TemporaryDirectory(prefix="ferrum-s2-request-dump-selftest-") as tmp:
        root = Path(tmp)
        startup = root / "serve-startup-fixture"
        startup.mkdir()
        for index in range(42):
            (root / f"request-{index:02d}").mkdir()
        observed_startup, observed_requests = partition_request_dump_bundles(root)
        require(observed_startup == startup and len(observed_requests) == 42, "selftest request bundle partition mismatch")
        duplicate_startup = root / "serve-startup-duplicate"
        duplicate_startup.mkdir()
        try:
            partition_request_dump_bundles(root)
        except ValidationError as error:
            require("one serve startup bundle" in str(error), "selftest duplicate startup failed for wrong reason")
        else:
            raise ValidationError("selftest duplicate startup bundle unexpectedly passed")
        duplicate_startup.rmdir()
        extra_request = root / "request-extra"
        extra_request.mkdir()
        try:
            partition_request_dump_bundles(root)
        except ValidationError as error:
            require("42 scenario request bundles" in str(error), "selftest extra request failed for wrong reason")
        else:
            raise ValidationError("selftest extra scenario request bundle unexpectedly passed")
    with tempfile.TemporaryDirectory(prefix="ferrum-s2-stream-disconnect-selftest-") as tmp:
        path = Path(tmp) / "event.json"
        value = {"canonical_sha256_scope": "document_without_canonical_sha256_fields", "value": 1}
        value["canonical_sha256"] = json_sha256(value)
        write_json(path, value)
        validate_self_hash(read_json(path), "selftest hash")
        value["value"] = 2
        write_json(path, value)
        try:
            validate_self_hash(read_json(path), "selftest hash mutation")
        except ValidationError as error:
            require("self hash mismatch" in str(error), "selftest hash mutation failed for wrong reason")
        else:
            raise ValidationError("selftest hash mutation unexpectedly passed")
    print(SELFTEST_PASS_LINE)


def run_checkpoint(source: Path, out: Path, expected_git_sha: str | None) -> int:
    started_at = iso_now()
    started = time.monotonic()
    out = out.resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_id": CHECKPOINT_ID,
        "scope": ["S2/Unicode-stream-sentinel", "S2/disconnect-release-sentinel"],
        "full_s2": False,
        "model_matrix_c09_complete": False,
        "model_matrix_c17_complete": False,
        "source_root": str(source.resolve(strict=False)),
        "artifact_dir": str(out),
        "started_at": started_at,
    }
    try:
        evidence = validate_source(source, expected_git_sha)
    except (OSError, ValidationError) as error:
        manifest.update({"status": "fail", "finished_at": iso_now(), "duration_sec": time.monotonic() - started, "evidence": None, "error": str(error), "pass_line": None})
        write_json(out / "manifest.json", manifest)
        print(f"{FAIL_PREFIX}: {out}: {error}", file=sys.stderr)
        return 1
    pass_line = f"{PASS_PREFIX}: {out}"
    manifest.update({"status": "pass", "finished_at": iso_now(), "duration_sec": time.monotonic() - started, "evidence": evidence, "error": None, "pass_line": pass_line})
    write_json(out / "manifest.json", manifest)
    print(pass_line)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-git-sha")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    require(args.artifact_dir is not None, "--artifact-dir is required")
    require(args.out is not None, "--out is required")
    if args.expected_git_sha is not None:
        require(GIT_SHA_RE.fullmatch(args.expected_git_sha) is not None, "--expected-git-sha must be 40 lowercase hex characters")
    return run_checkpoint(args.artifact_dir, args.out, args.expected_git_sha)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"{FAIL_PREFIX}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
