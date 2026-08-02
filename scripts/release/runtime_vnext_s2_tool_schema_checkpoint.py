#!/usr/bin/env python3
"""Validate the CUDA S2 required-tool/strict-schema priority sentinel."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from runtime_vnext_s2_stream_disconnect_checkpoint import (
    GIT_SHA_RE,
    SHA256_RE,
    ValidationError,
    assert_clean_output,
    file_sha256,
    iso_now,
    json_sha256,
    read_jsonl,
    read_text,
    require,
    resolve_member,
    validate_artifact_tree,
    validate_resource_balance,
    validate_self_hash,
    write_json,
)


PASS_PREFIX = "FERRUM RUNTIME VNEXT S2 TOOL SCHEMA PRIORITY PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT S2 TOOL SCHEMA PRIORITY FAIL"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT S2 TOOL SCHEMA PRIORITY SELFTEST PASS"
CHECKPOINT_ID = "runtime-vnext-s2-tool-schema-priority-sentinel"
SCOPE = "S2/C21-required-tool-strict-response-format-priority-sentinel"
SCENARIO_NAME = "c21_required_tool_schema_priority"
SCENARIO_TYPE = "serve_tool_schema_priority"
MODEL = "Qwen/Qwen3.5-4B"
CASE_COUNT = 4
TRANSPORT_COUNT = CASE_COUNT * 2
RUNNER_PATH = Path(__file__).resolve().parent / "run_scenarios.py"
SCENARIO_MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "scenarios/runtime_vnext_s2_c21_cuda_smoke.json"
)
QWEN35_CACHE_RE = re.compile(
    r"(?:^|/)models--Qwen--Qwen3\.5-4B/snapshots/[0-9a-f]{40}/?$"
)
CASE_DIR_RE = re.compile(r"^[0-9]{3}-(?:sync|stream)$")
EXECUTION_PHASES = {
    "vnext.request_accepted",
    "vnext.plan_built",
    "vnext.frame_started",
    "vnext.operation_submitted",
    "vnext.frame_completed",
    "vnext.request_completed",
}
DOES_NOT_PROVE = [
    "P_OFFICIAL_DEFAULT/default-thinking C21",
    "run-plain",
    "generic serve-stream",
    "standalone strict-schema",
    "json_object",
    "auto-tool",
    "tool-result continuation",
    "full S2",
    "full MODEL_MATRIX C21",
]


def strict_json_loads(text: str, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise ValidationError(f"{label}: malformed JSON: {error}") from error


def read_json(path: Path) -> dict[str, Any]:
    text = read_text(path)
    assert_clean_output(str(path), text)
    value = strict_json_loads(text, str(path))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def expected_tool(marker: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "echo_value",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string", "const": marker}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }


def expected_response_format(marker: str, ordinal: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"conflict_{ordinal:03d}",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"result": {"type": "string", "const": marker}},
                "required": ["result"],
                "additionalProperties": False,
            },
        },
    }


def expected_request(marker: str, ordinal: int, stream: bool) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Call echo_value for {marker}; tool choice has priority "
                    "over the simultaneous strict response format."
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": 256,
        "tools": [expected_tool(marker)],
        "tool_choice": "required",
        "response_format": expected_response_format(marker, ordinal),
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": stream,
    }
    if stream:
        request["stream_options"] = {"include_usage": True}
    return request


def validate_usage(value: Any, label: str) -> dict[str, int]:
    require(isinstance(value, dict), f"{label}: usage missing")
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        require(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0,
            f"{label}.{key}: invalid token count",
        )
        result[key] = item
    require(result["completion_tokens"] > 0, f"{label}: completion_tokens must be positive")
    require(
        result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"],
        f"{label}: inconsistent usage",
    )
    return result


def validate_tool_call(call: Any, marker: str, label: str) -> dict[str, Any]:
    require(isinstance(call, dict), f"{label}: tool call must be an object")
    call_id = call.get("id")
    require(isinstance(call_id, str) and call_id, f"{label}: tool call id missing")
    require(call.get("type") == "function", f"{label}: tool call type mismatch")
    function = call.get("function")
    require(isinstance(function, dict), f"{label}: function missing")
    require(function.get("name") == "echo_value", f"{label}: tool name mismatch")
    arguments_text = function.get("arguments")
    require(isinstance(arguments_text, str), f"{label}: arguments must be a JSON string")
    arguments = strict_json_loads(arguments_text, f"{label}.arguments")
    require(arguments == {"value": marker}, f"{label}: tool arguments differ from marker")
    return {"name": "echo_value", "arguments": arguments}


def validate_sync_response(path: Path, marker: str) -> dict[str, Any]:
    response = read_json(path)
    require(response.get("object") == "chat.completion", f"{path}: object mismatch")
    require(response.get("model") == MODEL, f"{path}: model mismatch")
    response_id = response.get("id")
    require(isinstance(response_id, str) and response_id, f"{path}: response id missing")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, f"{path}: expected one choice")
    choice = choices[0]
    require(isinstance(choice, dict), f"{path}: choice must be an object")
    require(choice.get("index") == 0, f"{path}: choice index mismatch")
    require(choice.get("finish_reason") == "tool_calls", f"{path}: finish_reason mismatch")
    message = choice.get("message")
    require(isinstance(message, dict), f"{path}: assistant message missing")
    require(message.get("role") == "assistant", f"{path}: assistant role mismatch")
    require(message.get("content") in (None, ""), f"{path}: assistant content leaked")
    for key in ("reasoning", "reasoning_content"):
        require(message.get(key) in (None, ""), f"{path}: assistant reasoning leaked")
    calls = message.get("tool_calls")
    require(isinstance(calls, list) and len(calls) == 1, f"{path}: expected one tool call")
    return {
        "tool_call": validate_tool_call(calls[0], marker, str(path)),
        "usage": validate_usage(response.get("usage"), str(path)),
        "response_id": response_id,
    }


def parse_stream(path: Path, marker: str) -> dict[str, Any]:
    text = read_text(path)
    assert_clean_output(str(path), text)
    done_count = 0
    malformed_json = 0
    content_delta_count = 0
    usage_chunks = 0
    chunks = 0
    finish_reasons: list[str] = []
    usage_payloads: list[dict[str, int]] = []
    tool_call_deltas: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    response_ids: set[str] = set()
    saw_done = False
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        require(line.startswith("data: "), f"{path}:{line_no}: non-SSE line")
        require(not saw_done, f"{path}:{line_no}: data after [DONE]")
        payload = line.removeprefix("data: ").strip()
        if payload == "[DONE]":
            done_count += 1
            saw_done = True
            continue
        value = strict_json_loads(payload, f"{path}:{line_no}")
        require(isinstance(value, dict), f"{path}:{line_no}: SSE payload must be an object")
        chunks += 1
        require(value.get("object") == "chat.completion.chunk", f"{path}:{line_no}: object mismatch")
        require(value.get("model") == MODEL, f"{path}:{line_no}: model mismatch")
        response_id = value.get("id")
        require(isinstance(response_id, str) and response_id, f"{path}:{line_no}: response id missing")
        response_ids.add(response_id)
        error = value.get("error")
        if isinstance(error, dict):
            errors.append(error)
        usage = value.get("usage")
        if usage is not None:
            usage_chunks += 1
            usage_payloads.append(validate_usage(usage, f"{path}:{line_no}"))
        choices = value.get("choices")
        require(isinstance(choices, list), f"{path}:{line_no}: choices must be a list")
        for choice in choices:
            require(isinstance(choice, dict), f"{path}:{line_no}: choice must be an object")
            require(choice.get("index") == 0, f"{path}:{line_no}: choice index mismatch")
            reason = choice.get("finish_reason")
            if reason is not None:
                finish_reasons.append(str(reason))
            delta = choice.get("delta")
            require(isinstance(delta, dict), f"{path}:{line_no}: delta must be an object")
            for key in ("content", "reasoning", "reasoning_content"):
                value_text = delta.get(key)
                require(value_text in (None, ""), f"{path}:{line_no}: {key} leaked")
                if value_text:
                    content_delta_count += 1
            calls = delta.get("tool_calls")
            if calls is not None:
                require(isinstance(calls, list), f"{path}:{line_no}: tool_calls must be a list")
                for call in calls:
                    require(isinstance(call, dict), f"{path}:{line_no}: tool delta must be an object")
                    tool_call_deltas.append(call)
    require(done_count == 1, f"{path}: [DONE] count mismatch")
    require(malformed_json == 0, f"{path}: malformed SSE JSON")
    require(usage_chunks == 1, f"{path}: usage chunk count mismatch")
    require(content_delta_count == 0, f"{path}: content delta leaked")
    require(not errors, f"{path}: stream error payload")
    require(finish_reasons == ["tool_calls"], f"{path}: finish reasons mismatch")
    require(len(response_ids) == 1, f"{path}: response id changed across chunks")
    calls: dict[int, dict[str, Any]] = {}
    for delta in tool_call_deltas:
        index = delta.get("index", 0)
        require(isinstance(index, int) and not isinstance(index, bool), f"{path}: tool index invalid")
        call = calls.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        delta_id = delta.get("id")
        if delta_id:
            require(call["id"] in ("", delta_id), f"{path}: streamed tool id changed")
            call["id"] = str(delta_id)
        delta_type = delta.get("type")
        if delta_type:
            require(delta_type == "function", f"{path}: streamed tool type mismatch")
        function = delta.get("function")
        if function is not None:
            require(isinstance(function, dict), f"{path}: streamed function delta invalid")
            if function.get("name"):
                call["function"]["name"] += str(function["name"])
            if function.get("arguments"):
                call["function"]["arguments"] += str(function["arguments"])
    require(set(calls) == {0}, f"{path}: expected one tool call at index zero")
    tool_call = validate_tool_call(calls[0], marker, str(path))
    runner_protocol = {
        "done_count": done_count,
        "malformed_json": malformed_json,
        "content_delta_count": content_delta_count,
        "usage_chunks": usage_chunks,
        "chunk_count": chunks,
        "output_text": "",
        "finish_reasons": finish_reasons,
        "usage_payloads": usage_payloads,
        "tool_call_deltas": tool_call_deltas,
        "errors": errors,
    }
    return {
        "tool_call": tool_call,
        "usage": usage_payloads[0],
        "event_count": chunks + done_count,
        "protocol": runner_protocol,
        "response_id": next(iter(response_ids)),
    }


def validate_case(
    scenario_root: Path,
    ordinal: int,
    mode: str,
    aggregate: Any,
) -> dict[str, Any]:
    marker = f"vnext-c21-{ordinal:03d}"
    stream = mode == "stream"
    root = scenario_root / f"{ordinal:03d}-{mode}"
    require(root.is_dir() and not root.is_symlink(), f"missing case directory: {root}")
    expected_files = {"request.json", "result.json", "response.sse" if stream else "response.json"}
    actual_files = {
        path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()
    }
    require(actual_files == expected_files, f"{root}: case file set mismatch")
    request = read_json(root / "request.json")
    require(request == expected_request(marker, ordinal, stream), f"{root}: request contract mismatch")
    observed = (
        parse_stream(root / "response.sse", marker)
        if stream
        else validate_sync_response(root / "response.json", marker)
    )
    expected_result: dict[str, Any] = {
        "status": "pass",
        "marker": marker,
        "mode": mode,
        "http_status": 200,
    }
    if stream:
        expected_result.update(
            {
                "event_count": observed["event_count"],
                "protocol": observed["protocol"],
                "tool_call": observed["tool_call"],
            }
        )
    else:
        expected_result["tool_call"] = observed["tool_call"]
    result = read_json(root / "result.json")
    require(result == expected_result, f"{root}: result differs from raw request/response")
    require(isinstance(aggregate, dict), f"{root}: aggregate row missing")
    require(aggregate == {key: value for key, value in expected_result.items() if key != "status"}, f"{root}: scenario aggregate mismatch")
    return {
        "ordinal": ordinal,
        "mode": mode,
        "marker": marker,
        "request_sha256": json_sha256(request),
        "response_id": observed["response_id"],
        "usage": observed["usage"],
    }


def validate_scenario(
    source: Path,
    recorded: Path,
    summary_row: dict[str, Any],
) -> list[dict[str, Any]]:
    root = source / SCENARIO_NAME
    require(root.is_dir() and not root.is_symlink(), f"missing scenario directory: {root}")
    result = read_json(root / "result.json")
    require(result == summary_row, "summary scenario differs from scenario result")
    require(result.get("name") == SCENARIO_NAME, "scenario name mismatch")
    require(result.get("type") == SCENARIO_TYPE, "scenario type mismatch")
    require(result.get("status") == "pass", "scenario status mismatch")
    require(result.get("case_count") == TRANSPORT_COUNT, "scenario transport count must be eight")
    require(
        resolve_member(source, recorded, result.get("artifact"), "scenario artifact")
        == root / "result.json",
        "scenario artifact path mismatch",
    )
    cases = result.get("cases")
    require(isinstance(cases, list) and len(cases) == TRANSPORT_COUNT, "scenario case cardinality mismatch")
    expected_dirs = {
        f"{ordinal:03d}-{mode}"
        for ordinal in range(CASE_COUNT)
        for mode in ("sync", "stream")
    }
    actual_dirs = {
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and not entry.is_symlink() and CASE_DIR_RE.fullmatch(entry.name)
    }
    require(actual_dirs == expected_dirs, "scenario case directories are incomplete or duplicated")
    rows: list[dict[str, Any]] = []
    index = 0
    for ordinal in range(CASE_COUNT):
        for mode in ("sync", "stream"):
            rows.append(validate_case(root, ordinal, mode, cases[index]))
            index += 1
    require(len({row["marker"] for row in rows}) == CASE_COUNT, "logical marker cardinality mismatch")
    require(len({row["request_sha256"] for row in rows}) == TRANSPORT_COUNT, "transport requests are not unique")
    require(len({row["response_id"] for row in rows}) == TRANSPORT_COUNT, "response ids are not unique")
    return rows


def expected_matrix_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pass",
        "matrix_scenario_count": 0,
        "case_counts": {"json_schema": 0, "json_object": 0},
        "unique_expected_object_counts": {"json_schema": 0, "json_object": 0},
        "unique_json_schema_count": 0,
        "json_schema_category_counts": {
            "required": 0,
            "type": 0,
            "additionalProperties": 0,
            "enum": 0,
        },
        "cases": [],
    }


def is_qwen35_model(value: Any) -> bool:
    return value == MODEL or (
        isinstance(value, str) and QWEN35_CACHE_RE.search(value) is not None
    )


def validate_identity(
    source: Path,
    expected_git_sha: str | None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    summary = read_json(source / "summary.json")
    require(summary.get("schema_version") == 1 and summary.get("status") == "pass", "summary status mismatch")
    require(summary.get("backend") == "cuda" and summary.get("model") == MODEL, "summary model/backend mismatch")
    git_sha = summary.get("git_sha")
    require(isinstance(git_sha, str) and GIT_SHA_RE.fullmatch(git_sha), "summary git SHA invalid")
    if expected_git_sha is not None:
        require(git_sha == expected_git_sha, "artifact git SHA differs from expected SHA")
    require(summary.get("dirty_status") == {"is_dirty": False, "status_short": []}, "artifact source was dirty")
    require(summary.get("failed") == 0 and summary.get("skipped") == 0, "summary contains failed/skipped scenarios")
    require(summary.get("scenario_count") == summary.get("manifest_scenario_count") == 1, "summary scenario count mismatch")
    require(summary.get("requested_scenarios") == [], "checkpoint must run the full manifest")
    require(summary.get("selected_scenarios") == [SCENARIO_NAME], "selected scenario mismatch")
    artifact_dir = summary.get("artifact_dir")
    require(isinstance(artifact_dir, str) and Path(artifact_dir).is_absolute(), "recorded artifact root invalid")
    recorded = Path(artifact_dir)
    require(summary.get("pass_line") == f"BACKEND REGRESSION SMOKE PASS: {recorded}", "runner PASS line mismatch")
    matrix = read_json(source / "response_format_matrix_contract.json")
    require(matrix == expected_matrix_contract(), "response-format matrix must remain empty for this sentinel")
    summary_matrix = summary.get("response_format_matrix_contract")
    require(isinstance(summary_matrix, dict), "summary response-format contract binding missing")
    require(resolve_member(source, recorded, summary_matrix.get("artifact"), "matrix artifact") == source / "response_format_matrix_contract.json", "matrix artifact path mismatch")
    require(summary_matrix.get("case_counts") == matrix["case_counts"], "matrix case counts mismatch")
    require(summary_matrix.get("unique_json_schema_count") == 0, "matrix schema count mismatch")

    receipt = read_json(source / "execution_receipt.json")
    validate_self_hash(receipt, "execution_receipt.json")
    require(receipt.get("schema_version") == 1, "execution receipt schema mismatch")
    require(receipt.get("mode") == "start" and receipt.get("git_sha") == git_sha, "execution receipt identity mismatch")
    require(receipt.get("dirty_status") == summary.get("dirty_status"), "receipt dirty status mismatch")
    require(receipt.get("backend") == "cuda" and receipt.get("model") == MODEL, "receipt model/backend mismatch")
    require(receipt.get("selected_scenarios") == [SCENARIO_NAME], "receipt scenarios mismatch")
    require(receipt.get("scenario_count") == 1 and receipt.get("failed") == receipt.get("skipped") == 0, "receipt outcome mismatch")
    binary_sha = receipt.get("binary_sha256")
    require(isinstance(binary_sha, str) and SHA256_RE.fullmatch(binary_sha), "binary SHA invalid")
    binary_path = receipt.get("binary_path")
    require(isinstance(binary_path, str) and Path(binary_path).is_absolute(), "binary path invalid")
    hardware = receipt.get("hardware")
    require(isinstance(hardware, dict), "hardware receipt missing")
    require(hardware.get("argv") == ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"], "hardware probe argv mismatch")
    require(hardware.get("returncode") == 0, "hardware probe failed")
    gpu_rows = [line for line in str(hardware.get("stdout", "")).splitlines() if line.strip()]
    require(len(gpu_rows) == 1 and "RTX 4090" in gpu_rows[0], "checkpoint requires one RTX 4090")
    argv = receipt.get("server_argv")
    require(isinstance(argv, list) and len(argv) > 2, "server argv missing")
    require(argv[0] == binary_path and argv[1] == "serve" and argv[-1] == MODEL, "server argv product path mismatch")
    require("--backend" in argv and argv[argv.index("--backend") + 1] == "cuda", "server argv missing typed CUDA backend")
    expected_observability = {
        "--profile-jsonl": source / "observability/serve/profile.jsonl",
        "--memory-profile-jsonl": source / "observability/serve/memory_profile.jsonl",
        "--scheduler-trace-jsonl": source / "observability/serve/scheduler_trace.jsonl",
        "--request-dump-dir": source / "observability/serve/request_dump",
    }
    for flag, local_path in expected_observability.items():
        require(flag in argv, f"server argv missing {flag}")
        recorded_value = Path(argv[argv.index(flag) + 1])
        require(resolve_member(source, recorded, str(recorded_value), f"server argv {flag}") == local_path, f"server argv {flag} path mismatch")
    require("--profile-detail" in argv and argv[argv.index("--profile-detail") + 1] == "basic", "profile detail mismatch")
    require("--profile-sample-rate" in argv and float(argv[argv.index("--profile-sample-rate") + 1]) == 1.0, "profile sample rate mismatch")
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
    for name, filename, sha_key, current in (
        ("runner", "run_scenarios.py", "runner_sha256", RUNNER_PATH),
        ("manifest", "scenario_manifest.json", "manifest_sha256", SCENARIO_MANIFEST_PATH),
    ):
        item = inputs.get(name)
        require(isinstance(item, dict), f"receipt input {name} missing")
        path = resolve_member(source, recorded, item.get("path"), f"receipt input {name}")
        require(path == source / "inputs" / filename, f"receipt input {name} path mismatch")
        require(item.get("sha256") == file_sha256(path) == receipt.get(sha_key), f"receipt input {name} SHA mismatch")
        require(file_sha256(path) == file_sha256(current), f"artifact {name} differs from current checked-in source")
    runner_argv = receipt.get("runner_argv")
    require(isinstance(runner_argv, list) and "--manifest" in runner_argv and "--out" in runner_argv, "runner argv missing manifest/out")
    require("--only" not in runner_argv, "checkpoint runner must not use --only")
    runner_path_value = receipt.get("runner_path")
    require(
        isinstance(runner_path_value, str) and Path(runner_path_value).is_absolute(),
        "receipt runner_path must be absolute",
    )
    require(len(runner_argv) >= 2, "runner argv does not identify the executed script")
    require(
        Path(runner_argv[1]).resolve(strict=False)
        == Path(runner_path_value).resolve(strict=False),
        "runner argv script does not match receipt runner_path",
    )
    require(
        Path(runner_path_value).name == "run_scenarios.py",
        "receipt runner_path does not identify run_scenarios.py",
    )
    require(
        receipt.get("runner_sha256") == file_sha256(source / "inputs/run_scenarios.py"),
        "executed runner SHA is not bound to the copied runner",
    )
    cwd = Path(str(receipt.get("cwd") or ""))
    require(cwd.is_absolute(), "receipt cwd must be absolute")
    manifest_arg = Path(runner_argv[runner_argv.index("--manifest") + 1])
    out_arg = Path(runner_argv[runner_argv.index("--out") + 1])
    if not manifest_arg.is_absolute():
        manifest_arg = cwd / manifest_arg
    if not out_arg.is_absolute():
        out_arg = cwd / out_arg
    require(manifest_arg.resolve(strict=False) == Path(str(receipt.get("manifest_path"))).resolve(strict=False), "runner manifest argv mismatch")
    require(out_arg.resolve(strict=False) == recorded.resolve(strict=False), "runner out argv mismatch")
    manifest = read_json(source / "inputs/scenario_manifest.json")
    require(manifest.get("goal_scope") == {"full_s2": False, "model_matrix_c21_complete": False}, "input manifest overclaims S2/C21")
    require(manifest.get("backend") == "cuda" and manifest.get("model") == MODEL, "input manifest model/backend mismatch")
    require(manifest.get("observability") == {"enabled": True, "profile_detail": "basic", "profile_sample_rate": 1.0}, "input manifest observability mismatch")
    require(manifest.get("server") == {"args": ["--backend", "cuda"], "mode": "start"}, "input manifest must start typed CUDA server")
    scenarios = manifest.get("scenarios")
    require(isinstance(scenarios, list) and len(scenarios) == 1, "input manifest scenario count mismatch")
    require(scenarios[0] == {"name": SCENARIO_NAME, "type": SCENARIO_TYPE, "case_count": 4, "enable_thinking": False, "marker_prefix": "vnext-c21", "max_tokens": 256}, "input manifest scenario contract mismatch")
    evidence = receipt.get("evidence_files")
    require(isinstance(evidence, dict), "execution evidence bindings missing")
    expected_evidence = {
        "effective_config": "server.effective_config.json",
        "decision_trace": "server.decision_trace.jsonl",
        "server_log": "server.log",
        "health_before": "server.health.json",
        "health_after": "server.health.after.json",
    }
    require(set(evidence) == set(expected_evidence), "execution evidence file set mismatch")
    for label, filename in expected_evidence.items():
        item = evidence[label]
        require(isinstance(item, dict), f"execution evidence missing: {label}")
        path = resolve_member(source, recorded, item.get("path"), f"execution evidence {label}")
        require(path == source / filename, f"execution evidence path mismatch: {label}")
        require(item.get("size") == path.stat().st_size and item.get("sha256") == file_sha256(path), f"execution evidence binding mismatch: {label}")
    effective = read_json(source / "server.effective_config.json")
    require(effective.get("backend") == "cuda" and effective.get("cuda_device_count") == 1, "effective CUDA config mismatch")
    require(effective.get("selected_gpu_devices") == [0], "effective selected GPU mismatch")
    require(effective.get("model_capabilities", {}).get("architecture") == "qwen3_5", "effective architecture mismatch")
    require(effective.get("hardware_capabilities", {}).get("backend") == "cuda", "effective hardware backend mismatch")
    require(effective.get("hardware_capabilities", {}).get("compiled_features", {}).get("cuda") is True, "binary lacks compiled CUDA feature")
    for filename in ("server.health.json", "server.health.after.json"):
        health = read_json(source / filename)
        require(health.get("status") == "pass" and health.get("http_status") == 200, f"{filename}: health mismatch")
    assert_clean_output("server.log", read_text(source / "server.log"))
    assert_clean_output("server.decision_trace.jsonl", read_text(source / "server.decision_trace.jsonl"))
    return summary, recorded, receipt


def validate_observability(
    source: Path,
    recorded: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    obs = summary.get("observability")
    require(isinstance(obs, dict) and obs.get("enabled") is True, "observability summary missing")
    require(read_json(source / "observability_summary.json") == obs, "observability summary artifact mismatch")
    scheduler_paths = obs.get("scheduler_trace_paths")
    dump_paths = obs.get("request_dump_dirs")
    require(isinstance(scheduler_paths, list) and len(scheduler_paths) == 1, "scheduler trace path cardinality")
    require(isinstance(dump_paths, list) and len(dump_paths) == 1, "request dump path cardinality")
    require(resolve_member(source, recorded, scheduler_paths[0], "scheduler trace") == source / "observability/serve/scheduler_trace.jsonl", "scheduler trace path mismatch")
    require(resolve_member(source, recorded, dump_paths[0], "request dump") == source / "observability/serve/request_dump", "request dump path mismatch")
    require(any(str(row.get("phase", "")).startswith("vnext.") for row in rows), "trace does not prove vNext execution")
    return {"trace_rows": len(rows)}


def partition_bundles(root: Path) -> tuple[Path, list[Path]]:
    bundles = sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink())
    startup = [path for path in bundles if path.name.startswith("serve-startup-")]
    requests = [path for path in bundles if not path.name.startswith("serve-startup-")]
    require(len(startup) == 1, f"expected one startup request dump, found {len(startup)}")
    require(len(requests) == TRANSPORT_COUNT, f"expected eight scenario request dumps, found {len(requests)}")
    return startup[0], requests


def validate_bundle_members(bundle: Path, request_id: str) -> None:
    for path in bundle.glob("*.json"):
        value = read_json(path)
        if value.get("request_id") is not None:
            require(value.get("request_id") == request_id, f"{path}: request id mismatch")
    backend = read_json(bundle / "backend_selection.json")
    require(backend.get("backend") == "actual" and is_qwen35_model(backend.get("model")), f"{bundle}: backend selection mismatch")
    bad_output = read_json(bundle / "bad_output_scan.json")
    require(bad_output.get("bad_output") is False and bad_output.get("bad_text_count") == 0 and bad_output.get("reasons") == [], f"{bundle}: bad-output scan failed")


def redacted_request_body(marker: str, ordinal: int, stream: bool) -> dict[str, Any]:
    body = expected_request(marker, ordinal, stream)
    prompt = body["messages"][0]["content"]
    body["messages"] = [{"role": "user", "content": "[redacted]", "content_redacted": True, "content_chars": len(prompt)}]
    return body


def validate_request_dumps(
    source: Path,
    trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    root = source / "observability/serve/request_dump"
    startup, bundles = partition_bundles(root)
    startup_request = read_json(startup / "request.json")
    startup_id = startup_request.get("request_id")
    require(startup_id == startup.name, "startup request dump identity mismatch")
    require(startup_request.get("entrypoint") == "serve" and startup_request.get("backend") == "actual", "startup request dump provenance mismatch")
    require(is_qwen35_model(startup_request.get("model")), "startup request dump model mismatch")
    require("http" not in startup_request, "startup bundle masquerades as an HTTP request")
    validate_bundle_members(startup, startup.name)

    observed: dict[tuple[str, bool], str] = {}
    request_ids: set[str] = set()
    for bundle in bundles:
        request = read_json(bundle / "request.json")
        request_id = request.get("request_id")
        require(isinstance(request_id, str) and request_id == bundle.name, f"{bundle}: request id mismatch")
        require(request_id not in request_ids, f"duplicate request id: {request_id}")
        request_ids.add(request_id)
        require(request.get("schema_version") == 1 and request.get("entrypoint") == "serve", f"{bundle}: request dump schema mismatch")
        require(request.get("backend") == "actual" and request.get("actual_model_smoke") is True and request.get("sanitized") is True, f"{bundle}: request dump provenance mismatch")
        require(is_qwen35_model(request.get("model")), f"{bundle}: request dump model mismatch")
        http = request.get("http")
        require(isinstance(http, dict) and http.get("method") == "POST" and http.get("path") == "/v1/chat/completions", f"{bundle}: HTTP identity mismatch")
        body = http.get("body")
        require(isinstance(body, dict), f"{bundle}: HTTP body missing")
        response_format = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
        marker = response_format.get("properties", {}).get("result", {}).get("const")
        require(
            isinstance(marker, str) and re.fullmatch(r"vnext-c21-[0-9]{3}", marker),
            f"{bundle}: marker missing",
        )
        ordinal = int(marker.rsplit("-", 1)[1])
        require(0 <= ordinal < CASE_COUNT, f"{bundle}: marker ordinal out of range")
        stream = body.get("stream") is True
        require(body == redacted_request_body(marker, ordinal, stream), f"{bundle}: redacted request body mismatch")
        require(request.get("stream") is stream, f"{bundle}: stream identity mismatch")
        key = (marker, stream)
        require(key not in observed, f"duplicate request dump key: {key}")
        observed[key] = request_id
        validate_bundle_members(bundle, request_id)
        resource_rows = [row for row in trace_rows if row.get("request_id") == request_id]
        require(resource_rows, f"request dump has no scheduler trace: {request_id}")
        closes = [row for row in resource_rows if row.get("phase") == "engine_request_close"]
        require(len(closes) == 1, f"request close cardinality mismatch: {request_id}")
        close_shape = closes[0].get("shape")
        close_attrs = closes[0].get("attributes")
        outstanding = close_shape.get("resource_owner_outstanding_count") if isinstance(close_shape, dict) else None
        if outstanding is None and isinstance(close_attrs, dict):
            outstanding = close_attrs.get("resource_owner_outstanding_count")
        require(outstanding == 0, f"request resources outstanding at close: {request_id}")
        validate_resource_balance(trace_rows, request_id)
        execution_id = f"request.product.{request_id}"
        execution_rows = [
            row for row in trace_rows if row.get("request_id") == execution_id
        ]
        require(execution_rows, f"request has no namespaced vNext execution trace: {request_id}")
        phases = {str(row.get("phase")) for row in execution_rows}
        require(
            EXECUTION_PHASES <= phases,
            f"request vNext execution lifecycle incomplete: {request_id}",
        )
        for row in execution_rows:
            phase = str(row.get("phase"))
            if phase not in EXECUTION_PHASES:
                continue
            require(
                row.get("entrypoint") == "serve"
                and row.get("backend") == "actual"
                and row.get("status") == "ok",
                f"request vNext execution provenance mismatch: {request_id}/{phase}",
            )
            backend_detail = row.get("backend_detail")
            attributes = row.get("attributes")
            require(
                isinstance(backend_detail, dict)
                and backend_detail.get("backend_device") == "CUDA(0)",
                f"request vNext execution did not use CUDA(0): {request_id}/{phase}",
            )
            require(
                isinstance(attributes, dict)
                and attributes.get("execution_trace_source") == "vnext"
                and attributes.get("actual_model_smoke") is True
                and attributes.get("diagnostic_only") is False
                and attributes.get("l0_only") is False,
                f"request vNext execution source mismatch: {request_id}/{phase}",
            )
        operations = [
            row for row in execution_rows if row.get("phase") == "vnext.operation_submitted"
        ]
        require(operations, f"request submitted no vNext operation: {request_id}")
        for row in operations:
            attributes = row["attributes"]
            require(
                attributes.get("execution_phase") == "execution"
                and str(attributes.get("provider_id", "")).startswith("provider.cuda.")
                and attributes.get("device_id") == "device.cuda.0",
                f"request vNext operation lacks CUDA provider identity: {request_id}",
            )
    expected = {
        (f"vnext-c21-{ordinal:03d}", stream)
        for ordinal in range(CASE_COUNT)
        for stream in (False, True)
    }
    require(set(observed) == expected, "request dump tool/schema matrix incomplete")
    return {"startup_request_id": startup_id, "request_count": len(request_ids)}


def validate_source(source: Path, expected_git_sha: str | None) -> dict[str, Any]:
    source = source.resolve(strict=True)
    summary, recorded, receipt = validate_identity(source, expected_git_sha)
    scenarios = summary.get("scenarios")
    require(isinstance(scenarios, list) and len(scenarios) == 1 and isinstance(scenarios[0], dict), "summary scenario missing")
    cases = validate_scenario(source, recorded, scenarios[0])
    trace_rows = read_jsonl(source / "observability/serve/scheduler_trace.jsonl")
    observability = validate_observability(source, recorded, summary, trace_rows)
    request_dumps = validate_request_dumps(source, trace_rows)
    read_json(source / "artifact_tree.json")
    tree = validate_artifact_tree(source, recorded)
    return {
        "git_sha": summary["git_sha"],
        "binary_sha256": receipt["binary_sha256"],
        "backend": "cuda",
        "model": MODEL,
        "hardware": receipt["hardware"]["stdout"].strip(),
        "scope": SCOPE,
        "full_s2": False,
        "model_matrix_c21_complete": False,
        "logical_case_count": CASE_COUNT,
        "transport_execution_count": TRANSPORT_COUNT,
        "does_not_prove": DOES_NOT_PROVE,
        "cases": cases,
        "observability": observability,
        "request_dumps": request_dumps,
        "artifact_tree": tree,
    }


def write_fixture_tree(root: Path) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name != "artifact_tree.json":
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_root": str(root),
        "file_count": len(files),
        "files": files,
        "canonical_sha256_scope": "document_without_canonical_sha256_fields",
    }
    value["canonical_sha256"] = json_sha256(value)
    write_json(root / "artifact_tree.json", value)


def self_hash(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["canonical_sha256_scope"] = "document_without_canonical_sha256_fields"
    result["canonical_sha256"] = json_sha256(result)
    return result


def rebind_receipt_input(root: Path, name: str, filename: str) -> None:
    receipt_path = root / "execution_receipt.json"
    receipt = read_json(receipt_path)
    receipt.pop("canonical_sha256", None)
    receipt.pop("canonical_sha256_scope", None)
    digest = file_sha256(root / "inputs" / filename)
    receipt[f"{name}_sha256"] = digest
    receipt["input_artifacts"][name]["sha256"] = digest
    receipt = self_hash(receipt)
    write_json(receipt_path, receipt)
    summary_path = root / "summary.json"
    summary = read_json(summary_path)
    summary["execution_receipt"][f"{name}_sha256"] = digest
    summary["execution_receipt"]["canonical_sha256"] = receipt["canonical_sha256"]
    summary["execution_receipt"]["artifact_sha256"] = file_sha256(receipt_path)
    write_json(summary_path, summary)


def mutate_receipt(root: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    receipt_path = root / "execution_receipt.json"
    receipt = read_json(receipt_path)
    receipt.pop("canonical_sha256", None)
    receipt.pop("canonical_sha256_scope", None)
    mutation(receipt)
    receipt = self_hash(receipt)
    write_json(receipt_path, receipt)
    summary_path = root / "summary.json"
    summary = read_json(summary_path)
    summary["execution_receipt"]["canonical_sha256"] = receipt["canonical_sha256"]
    summary["execution_receipt"]["artifact_sha256"] = file_sha256(receipt_path)
    write_json(summary_path, summary)


def make_response(marker: str, response_id: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{response_id}",
                            "type": "function",
                            "function": {
                                "name": "echo_value",
                                "arguments": json.dumps({"value": marker}, separators=(",", ":")),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }


def make_sse(marker: str, response_id: str) -> str:
    rows = [
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"call-{response_id}",
                                "type": "function",
                                "function": {"name": "echo_value", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": json.dumps({"value": marker}, separators=(",", ":"))
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "model": MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "model": MODEL,
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        },
    ]
    return "".join(f"data: {json.dumps(row, separators=(',', ':'))}\n\n" for row in rows) + "data: [DONE]\n\n"


def make_fixture(root: Path, git_sha: str) -> None:
    root.mkdir(parents=True)
    (root / "inputs").mkdir()
    shutil.copyfile(RUNNER_PATH, root / "inputs/run_scenarios.py")
    shutil.copyfile(SCENARIO_MANIFEST_PATH, root / "inputs/scenario_manifest.json")
    scenario_root = root / SCENARIO_NAME
    scenario_root.mkdir()
    aggregate_cases: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    dump_root = root / "observability/serve/request_dump"
    dump_root.mkdir(parents=True)

    startup_id = "serve-startup-fixture"
    startup = dump_root / startup_id
    startup.mkdir()
    write_json(startup / "request.json", {"schema_version": 1, "entrypoint": "serve", "request_id": startup_id, "model": MODEL, "backend": "actual", "actual_model_smoke": True, "sanitized": True})
    write_json(startup / "backend_selection.json", {"schema_version": 1, "request_id": startup_id, "backend": "actual", "model": MODEL, "actual_model_smoke": True})
    write_json(startup / "bad_output_scan.json", {"schema_version": 1, "request_id": startup_id, "bad_output": False, "bad_text_count": 0, "reasons": []})

    for ordinal in range(CASE_COUNT):
        marker = f"vnext-c21-{ordinal:03d}"
        for mode in ("sync", "stream"):
            stream = mode == "stream"
            case_root = scenario_root / f"{ordinal:03d}-{mode}"
            case_root.mkdir()
            request = expected_request(marker, ordinal, stream)
            write_json(case_root / "request.json", request)
            response_id = f"response-{ordinal:03d}-{mode}"
            if stream:
                (case_root / "response.sse").write_text(make_sse(marker, response_id), encoding="utf-8")
                observed = parse_stream(case_root / "response.sse", marker)
                row = {"marker": marker, "mode": mode, "http_status": 200, "event_count": observed["event_count"], "protocol": observed["protocol"], "tool_call": observed["tool_call"]}
            else:
                write_json(case_root / "response.json", make_response(marker, response_id))
                observed = validate_sync_response(case_root / "response.json", marker)
                row = {"marker": marker, "mode": mode, "http_status": 200, "tool_call": observed["tool_call"]}
            write_json(case_root / "result.json", {"status": "pass", **row})
            aggregate_cases.append(row)

            request_id = f"request-{ordinal:03d}-{mode}"
            bundle = dump_root / request_id
            bundle.mkdir()
            write_json(bundle / "request.json", {"schema_version": 1, "entrypoint": "serve", "request_id": request_id, "model": MODEL, "backend": "actual", "endpoint": "/v1/chat/completions", "method": "POST", "stream": stream, "actual_model_smoke": True, "sanitized": True, "http": {"method": "POST", "path": "/v1/chat/completions", "body": redacted_request_body(marker, ordinal, stream)}})
            write_json(bundle / "backend_selection.json", {"schema_version": 1, "request_id": request_id, "backend": "actual", "model": MODEL, "actual_model_smoke": True})
            write_json(bundle / "bad_output_scan.json", {"schema_version": 1, "request_id": request_id, "bad_output": False, "bad_text_count": 0, "reasons": []})
            trace_rows.extend(
                [
                    {"request_id": request_id, "correlation_id": request_id, "phase": "engine_request_open", "resource": {"owner_kind": "request", "owner_id": request_id, "resource_kind": "request_slot", "action": "request_open"}},
                    {"request_id": request_id, "correlation_id": request_id, "phase": "engine_request_slot_reserve", "resource": {"owner_kind": "request", "owner_id": request_id, "resource_kind": "request_slot", "action": "reserve", "amount": 1, "before": 0, "after": 1}},
                    {"request_id": request_id, "correlation_id": request_id, "phase": "engine_request_slot_commit", "resource": {"owner_kind": "request", "owner_id": request_id, "resource_kind": "request_slot", "action": "commit", "amount": 1, "before": 0, "after": 1}},
                    {"request_id": request_id, "correlation_id": request_id, "phase": "vnext.prefill_admission", "status": "ok"},
                    {"request_id": request_id, "correlation_id": request_id, "phase": "engine_request_slot_release", "resource": {"owner_kind": "request", "owner_id": request_id, "resource_kind": "request_slot", "action": "release", "amount": 1, "before": 1, "after": 0}},
                    {"request_id": request_id, "correlation_id": request_id, "phase": "engine_request_close", "shape": {"resource_owner_outstanding_count": 0}, "attributes": {"resource_owner_outstanding_count": 0}, "resource": {"owner_kind": "request", "owner_id": request_id, "resource_kind": "request_slot", "action": "request_close"}},
                ]
            )
            execution_id = f"request.product.{request_id}"
            for phase in (
                "vnext.request_accepted",
                "vnext.plan_built",
                "vnext.frame_started",
                "vnext.operation_submitted",
                "vnext.frame_completed",
                "vnext.request_completed",
            ):
                attributes: dict[str, Any] = {
                    "actual_model_smoke": True,
                    "backend_device": "CUDA(0)",
                    "diagnostic_only": False,
                    "execution_trace_source": "vnext",
                    "l0_only": False,
                }
                if phase == "vnext.operation_submitted":
                    attributes.update(
                        {
                            "device_id": "device.cuda.0",
                            "execution_phase": "execution",
                            "provider_id": "provider.cuda.fixture.f16",
                        }
                    )
                trace_rows.append(
                    {
                        "request_id": execution_id,
                        "correlation_id": execution_id,
                        "entrypoint": "serve",
                        "backend": "actual",
                        "phase": phase,
                        "status": "ok",
                        "backend_detail": {
                            "backend_device": "CUDA(0)",
                            "backend_type": "Candle",
                        },
                        "attributes": attributes,
                    }
                )

    scenario_result = {
        "status": "pass",
        "case_count": TRANSPORT_COUNT,
        "cases": aggregate_cases,
        "name": SCENARIO_NAME,
        "type": SCENARIO_TYPE,
        "artifact": str(scenario_root / "result.json"),
        "duration_sec": 1.0,
    }
    write_json(scenario_root / "result.json", scenario_result)
    observability_root = root / "observability/serve"
    (observability_root / "profile.jsonl").write_text('{"event":"fixture"}\n', encoding="utf-8")
    (observability_root / "memory_profile.jsonl").write_text('{"event":"fixture"}\n', encoding="utf-8")
    (observability_root / "scheduler_trace.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in trace_rows), encoding="utf-8")
    observability = {
        "enabled": True,
        "roots": {"run": [], "serve": str(observability_root)},
        "profile_paths": [str(observability_root / "profile.jsonl"), str(observability_root / "memory_profile.jsonl"), str(observability_root / "scheduler_trace.jsonl")],
        "scheduler_trace_paths": [str(observability_root / "scheduler_trace.jsonl")],
        "request_dump_dirs": [str(dump_root)],
    }
    write_json(root / "observability_summary.json", observability)
    write_json(root / "response_format_matrix_contract.json", expected_matrix_contract())
    write_json(root / "server.effective_config.json", {"backend": "cuda", "cuda_device_count": 1, "selected_gpu_devices": [0], "model_capabilities": {"architecture": "qwen3_5"}, "hardware_capabilities": {"backend": "cuda", "compiled_features": {"cuda": True}}})
    (root / "server.decision_trace.jsonl").write_text('{"selection":"fixture"}\n', encoding="utf-8")
    (root / "server.log").write_text("fixture server stopped cleanly\n", encoding="utf-8")
    health = {"status": "pass", "http_status": 200}
    write_json(root / "server.health.json", health)
    write_json(root / "server.health.after.json", health)

    binary_path = "/workspace/ferrum/target/release/ferrum"
    server_argv = [
        binary_path,
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--effective-config-json",
        str(root / "server.effective_config.json"),
        "--decision-trace-jsonl",
        str(root / "server.decision_trace.jsonl"),
        "--profile-jsonl",
        str(observability_root / "profile.jsonl"),
        "--profile-detail",
        "basic",
        "--memory-profile-jsonl",
        str(observability_root / "memory_profile.jsonl"),
        "--scheduler-trace-jsonl",
        str(observability_root / "scheduler_trace.jsonl"),
        "--request-dump-dir",
        str(dump_root),
        "--profile-sample-rate",
        "1.0",
        "--backend",
        "cuda",
        MODEL,
    ]
    evidence_files: dict[str, Any] = {}
    for label, filename in {
        "effective_config": "server.effective_config.json",
        "decision_trace": "server.decision_trace.jsonl",
        "server_log": "server.log",
        "health_before": "server.health.json",
        "health_after": "server.health.after.json",
    }.items():
        path = root / filename
        evidence_files[label] = {"path": str(path), "size": path.stat().st_size, "sha256": file_sha256(path)}
    binary_sha = "b" * 64
    runner_sha = file_sha256(root / "inputs/run_scenarios.py")
    manifest_sha = file_sha256(root / "inputs/scenario_manifest.json")
    receipt = self_hash(
        {
            "schema_version": 1,
            "mode": "start",
            "runner_argv": [sys.executable, str(RUNNER_PATH), "--manifest", str(SCENARIO_MANIFEST_PATH), "--out", str(root)],
            "runner_path": str(RUNNER_PATH),
            "runner_sha256": runner_sha,
            "manifest_path": str(SCENARIO_MANIFEST_PATH),
            "manifest_sha256": manifest_sha,
            "cwd": str(Path.cwd().resolve()),
            "git_sha": git_sha,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "input_artifacts": {"runner": {"path": str(root / "inputs/run_scenarios.py"), "sha256": runner_sha}, "manifest": {"path": str(root / "inputs/scenario_manifest.json"), "sha256": manifest_sha}},
            "backend": "cuda",
            "model": MODEL,
            "selected_scenarios": [SCENARIO_NAME],
            "mode": "start",
            "server_argv": server_argv,
            "binary_path": binary_path,
            "binary_sha256": binary_sha,
            "hardware": {"argv": ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"], "returncode": 0, "stdout": "0, NVIDIA GeForce RTX 4090, GPU-fixture, 24564, 555.42\n", "stderr": ""},
            "removed_hidden_env_names": [],
            "child_env": {},
            "server_started_at": iso_now(),
            "server_finished_at": iso_now(),
            "server_returncode": -15,
            "scenario_count": 1,
            "failed": 0,
            "skipped": 0,
            "evidence_files": evidence_files,
        }
    )
    write_json(root / "execution_receipt.json", receipt)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "manifest": str(SCENARIO_MANIFEST_PATH),
        "artifact_dir": str(root),
        "model": MODEL,
        "backend": "cuda",
        "base_url": "http://127.0.0.1:8000",
        "git_sha": git_sha,
        "dirty_status": {"is_dirty": False, "status_short": []},
        "started_at": iso_now(),
        "finished_at": iso_now(),
        "scenario_count": 1,
        "manifest_scenario_count": 1,
        "requested_scenarios": [],
        "selected_scenarios": [SCENARIO_NAME],
        "failed": 0,
        "skipped": 0,
        "scenarios": [scenario_result],
        "response_format_matrix_contract": {"artifact": str(root / "response_format_matrix_contract.json"), "case_counts": {"json_schema": 0, "json_object": 0}, "unique_json_schema_count": 0},
        "observability": observability,
        "execution_receipt": {"artifact": str(root / "execution_receipt.json"), "artifact_sha256": file_sha256(root / "execution_receipt.json"), "canonical_sha256": receipt["canonical_sha256"], "mode": "start", "runner_sha256": runner_sha, "manifest_sha256": manifest_sha, "binary_sha256": binary_sha},
        "pass_line": f"BACKEND REGRESSION SMOKE PASS: {root}",
    }
    write_json(root / "summary.json", summary)
    write_fixture_tree(root)


def expect_reject(root: Path, mutate: Callable[[Path], None], needle: str) -> None:
    candidate = root.parent / f"mutation-{needle.replace(' ', '-').replace('/', '-')}"
    shutil.copytree(root, candidate)
    mutate(candidate)
    write_fixture_tree(candidate)
    try:
        validate_source(candidate, "a" * 40)
    except ValidationError as error:
        require(needle in str(error), f"mutation failed for wrong reason: {error}")
    else:
        raise ValidationError(f"mutation unexpectedly passed: {needle}")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-s2-tool-schema-") as temporary:
        root = Path(temporary) / "fixture"
        make_fixture(root, "a" * 40)
        evidence = validate_source(root, "a" * 40)
        require(evidence["logical_case_count"] == 4 and evidence["transport_execution_count"] == 8, "selftest evidence denominator mismatch")
        try:
            validate_source(root, "c" * 40)
        except ValidationError as error:
            require("differs from expected SHA" in str(error), "stale SHA mutation failed for wrong reason")
        else:
            raise ValidationError("stale SHA unexpectedly passed")

        def mutate_request(path: Path) -> None:
            request_path = path / SCENARIO_NAME / "000-sync/request.json"
            request = read_json(request_path)
            request.pop("tool_choice")
            write_json(request_path, request)

        def mutate_stream(path: Path) -> None:
            stream_path = path / SCENARIO_NAME / "000-stream/response.sse"
            stream_path.write_text(read_text(stream_path).replace("data: [DONE]\n\n", ""), encoding="utf-8")

        def mutate_trace(path: Path) -> None:
            trace_path = path / "observability/serve/scheduler_trace.jsonl"
            rows = read_jsonl(trace_path)
            for row in rows:
                if row.get("phase") == "engine_request_close":
                    row["shape"]["resource_owner_outstanding_count"] = 1
                    break
            trace_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")

        def mutate_execution_source(path: Path) -> None:
            trace_path = path / "observability/serve/scheduler_trace.jsonl"
            rows = read_jsonl(trace_path)
            for row in rows:
                if row.get("phase") == "vnext.operation_submitted":
                    row["attributes"]["execution_trace_source"] = "legacy"
                    break
            trace_path.write_text(
                "".join(
                    json.dumps(row, separators=(",", ":")) + "\n" for row in rows
                ),
                encoding="utf-8",
            )

        def mutate_runner(path: Path) -> None:
            runner = path / "inputs/run_scenarios.py"
            runner.write_text(read_text(runner) + "\n# stale fixture\n", encoding="utf-8")
            rebind_receipt_input(path, "runner", "run_scenarios.py")

        def mutate_runner_identity(path: Path) -> None:
            mutate_receipt(
                path,
                lambda receipt: receipt.update(
                    {"runner_path": "/workspace/ferrum/scripts/release/forged.py"}
                ),
            )

        expect_reject(root, mutate_request, "request contract mismatch")
        expect_reject(root, mutate_stream, "[DONE] count mismatch")
        expect_reject(root, mutate_trace, "resources outstanding at close")
        expect_reject(root, mutate_execution_source, "vNext execution source mismatch")
        expect_reject(root, mutate_runner, "artifact runner differs from current checked-in source")
        expect_reject(
            root,
            mutate_runner_identity,
            "runner argv script does not match receipt runner_path",
        )
    print(SELFTEST_PASS_LINE)


def run_checkpoint(source: Path, out: Path, expected_git_sha: str | None) -> int:
    started_at = iso_now()
    started = time.monotonic()
    out = out.resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_id": CHECKPOINT_ID,
        "scope": [SCOPE],
        "full_s2": False,
        "model_matrix_c21_complete": False,
        "does_not_prove": DOES_NOT_PROVE,
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
