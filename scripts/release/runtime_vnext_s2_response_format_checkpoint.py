#!/usr/bin/env python3
"""Validate only the Runtime vNext S2 C14/C15 CUDA response-format slice."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PASS_PREFIX = "FERRUM RUNTIME VNEXT S2 RESPONSE FORMAT PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT S2 RESPONSE FORMAT FAIL"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT S2 RESPONSE FORMAT SELFTEST PASS"
CHECKPOINT_ID = "runtime-vnext-s2-response-format-c14-c15"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QWEN35_4B_CACHE_RE = re.compile(
    r"(?:^|/)models--Qwen--Qwen3\.5-4B/snapshots/[0-9a-f]{40}/?$"
)
CASE_DIR_RE = re.compile(r"^[0-9]{3}$")
SCENARIO_TYPE = "serve_response_format_matrix"
BAD_TEXT = (
    "<unk>",
    "[pad",
    "<|assistant|>",
    "<|tool|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_",
    "classname=",
    "invalid utf-8",
    "mojibake",
)
SCENARIOS: dict[str, dict[str, Any]] = {
    "m1_c14_no_thinking": {
        "format": "json_schema",
        "count": 50,
        "thinking": False,
        "preset": "P_NO_THINKING",
        "categories": {
            "required": 13,
            "type": 13,
            "additionalProperties": 12,
            "enum": 12,
        },
    },
    "m1_c14_thinking": {
        "format": "json_schema",
        "count": 20,
        "thinking": True,
        "preset": "P_THINKING",
        "categories": {
            "required": 5,
            "type": 5,
            "additionalProperties": 5,
            "enum": 5,
        },
    },
    "m1_c15_no_thinking": {
        "format": "json_object",
        "count": 50,
        "thinking": False,
        "preset": "P_NO_THINKING",
        "categories": {},
    },
    "m1_c15_thinking": {
        "format": "json_object",
        "count": 20,
        "thinking": True,
        "preset": "P_THINKING",
        "categories": {},
    },
}
SAMPLING = {
    False: {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "seed": 9271,
        "stop": [],
    },
    True: {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "seed": 9271,
        "stop": [],
    },
}


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


def read_text(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationError(f"invalid UTF-8 in {path}: {error}") from error


def read_json(path: Path) -> dict[str, Any]:
    text = read_text(path)
    assert_clean_text(str(path), text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValidationError(f"malformed JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def with_canonical_sha256(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["canonical_sha256_scope"] = "document_without_canonical_sha256_fields"
    result["canonical_sha256"] = json_fingerprint(result)
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_clean_text(label: str, text: str) -> None:
    require("\ufffd" not in text, f"{label}: Unicode replacement character")
    require("\x00" not in text, f"{label}: NUL byte")
    lowered = text.lower()
    for token in BAD_TEXT:
        require(token not in lowered, f"{label}: forbidden text {token!r}")


def resolve_source_member(
    source_root: Path,
    value: Any,
    label: str,
    *,
    recorded_root: Path | None = None,
    expected: Path | None = None,
    must_exist: bool = True,
) -> Path:
    require(isinstance(value, str) and value, f"{label} must be a non-empty path string")
    raw = Path(value)
    require(".." not in raw.parts, f"{label} contains a parent traversal: {value}")
    if raw.is_absolute() and recorded_root is not None and raw != source_root:
        require(recorded_root.is_absolute(), f"{label}: recorded artifact root must be absolute")
        try:
            relative = raw.relative_to(recorded_root)
        except ValueError as error:
            raise ValidationError(
                f"{label} is outside recorded artifact root {recorded_root}: {value}"
            ) from error
        candidate = source_root / relative
    else:
        candidate = raw if raw.is_absolute() else source_root / raw
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(source_root)
    except ValueError as error:
        raise ValidationError(f"{label} escapes source root: {value}") from error
    if expected is not None:
        require(resolved == expected.resolve(strict=False), f"{label} does not name {expected}")
    if must_exist:
        require(resolved.exists(), f"{label} does not exist: {resolved}")
    return resolved


def validate_canonical_self_hash(value: dict[str, Any], label: str) -> str:
    require(
        value.get("canonical_sha256_scope")
        == "document_without_canonical_sha256_fields",
        f"{label}.canonical_sha256_scope mismatch",
    )
    actual = value.get("canonical_sha256")
    require(
        isinstance(actual, str) and SHA256_RE.fullmatch(actual),
        f"{label}.canonical_sha256 must be lowercase SHA256",
    )
    unsigned = dict(value)
    unsigned.pop("canonical_sha256", None)
    require(json_fingerprint(unsigned) == actual, f"{label} canonical SHA256 mismatch")
    return actual


def validate_artifact_tree(source_root: Path) -> dict[str, Any]:
    tree_path = source_root / "artifact_tree.json"
    tree = read_json(tree_path)
    tree_canonical_sha256 = validate_canonical_self_hash(tree, "artifact_tree")
    require(tree.get("schema_version") == 1, "artifact_tree schema_version must be 1")
    recorded_root = tree.get("artifact_root")
    require(
        isinstance(recorded_root, str) and Path(recorded_root).is_absolute(),
        "artifact_tree.artifact_root must be absolute",
    )
    entries = tree.get("files")
    require(isinstance(entries, list), "artifact_tree.files must be a list")
    require(tree.get("file_count") == len(entries), "artifact_tree.file_count mismatch")
    recorded: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        require(isinstance(entry, dict), f"artifact_tree.files[{index}] must be an object")
        relative = entry.get("path")
        require(isinstance(relative, str) and relative, f"artifact_tree.files[{index}] missing path")
        relative_path = Path(relative)
        require(not relative_path.is_absolute(), f"artifact_tree path must be relative: {relative}")
        require(".." not in relative_path.parts, f"artifact_tree path escapes root: {relative}")
        require(relative != "artifact_tree.json", "artifact_tree must not hash itself")
        require(relative not in recorded, f"artifact_tree duplicate path: {relative}")
        path = (source_root / relative_path).resolve(strict=False)
        try:
            path.relative_to(source_root)
        except ValueError as error:
            raise ValidationError(f"artifact_tree path escapes source root: {relative}") from error
        require(path.is_file() and not path.is_symlink(), f"artifact_tree missing regular file: {relative}")
        require(entry.get("size") == path.stat().st_size, f"artifact_tree size mismatch: {relative}")
        sha256 = entry.get("sha256")
        require(
            isinstance(sha256, str) and SHA256_RE.fullmatch(sha256),
            f"artifact_tree bad SHA256: {relative}",
        )
        require(file_sha256(path) == sha256, f"artifact_tree SHA256 mismatch: {relative}")
        recorded[relative] = entry
    actual: set[str] = set()
    for path in source_root.rglob("*"):
        require(not path.is_symlink(), f"artifact source contains symlink: {path}")
        if path.is_file() and path.name != "artifact_tree.json":
            actual.add(path.relative_to(source_root).as_posix())
    require(set(recorded) == actual, "artifact_tree does not exactly cover source regular files")
    return {
        "path": str(tree_path),
        "file_sha256": file_sha256(tree_path),
        "canonical_sha256": tree_canonical_sha256,
        "file_count": len(entries),
    }


def validate_execution_receipt(
    source_root: Path,
    recorded_root: Path,
    summary: dict[str, Any],
    *,
    git_sha: str,
    model: str,
) -> dict[str, Any]:
    summary_receipt = summary.get("execution_receipt")
    require(isinstance(summary_receipt, dict), "summary missing execution_receipt")
    receipt_path = resolve_source_member(
        source_root,
        summary_receipt.get("artifact"),
        "summary.execution_receipt.artifact",
        recorded_root=recorded_root,
        expected=source_root / "execution_receipt.json",
    )
    receipt = read_json(receipt_path)
    canonical_sha256 = validate_canonical_self_hash(receipt, "execution_receipt")
    require(receipt.get("schema_version") == 1, "execution_receipt schema_version must be 1")
    require(receipt.get("mode") == "start", "execution_receipt must describe a started server")
    require(receipt.get("git_sha") == git_sha, "execution_receipt git_sha mismatch")
    require(receipt.get("dirty_status") == summary.get("dirty_status"), "receipt dirty status mismatch")
    require(receipt.get("backend") == "cuda", "execution_receipt backend must be cuda")
    require(receipt.get("model") == model, "execution_receipt model mismatch")
    require(
        receipt.get("selected_scenarios") == list(SCENARIOS),
        "execution_receipt selected scenario set/order mismatch",
    )
    require(receipt.get("scenario_count") == 4, "execution_receipt scenario_count must be four")
    require(receipt.get("failed") == 0, "execution_receipt failed must be zero")
    require(receipt.get("skipped") == 0, "execution_receipt skipped must be zero")
    require(
        summary_receipt.get("mode") == "start",
        "summary execution receipt mode must be start",
    )
    for key in ("runner_sha256", "manifest_sha256", "binary_sha256"):
        value = receipt.get(key)
        require(
            isinstance(value, str) and SHA256_RE.fullmatch(value),
            f"execution_receipt.{key} must be lowercase SHA256",
        )
        require(summary_receipt.get(key) == value, f"summary execution receipt {key} mismatch")
    require(
        summary_receipt.get("canonical_sha256") == canonical_sha256,
        "summary execution receipt canonical SHA mismatch",
    )
    require(
        summary_receipt.get("artifact_sha256") == file_sha256(receipt_path),
        "summary execution receipt file SHA mismatch",
    )
    inputs = receipt.get("input_artifacts")
    require(isinstance(inputs, dict), "execution_receipt.input_artifacts missing")
    for key, expected_name, sha_key in (
        ("runner", "inputs/run_scenarios.py", "runner_sha256"),
        ("manifest", "inputs/scenario_manifest.json", "manifest_sha256"),
    ):
        item = inputs.get(key)
        require(isinstance(item, dict), f"execution_receipt input {key} missing")
        input_path = resolve_source_member(
            source_root,
            item.get("path"),
            f"execution_receipt.input_artifacts.{key}.path",
            recorded_root=recorded_root,
            expected=source_root / expected_name,
        )
        require(item.get("sha256") == file_sha256(input_path), f"input {key} file SHA mismatch")
        require(item.get("sha256") == receipt.get(sha_key), f"input {key} receipt SHA mismatch")
    runner_argv = receipt.get("runner_argv")
    require(isinstance(runner_argv, list) and runner_argv, "execution_receipt runner_argv missing")
    require("--manifest" in runner_argv and "--out" in runner_argv, "runner argv missing manifest/out")
    cwd = Path(str(receipt.get("cwd") or ""))
    require(cwd.is_absolute(), "execution_receipt cwd must be absolute")
    manifest_arg = Path(runner_argv[runner_argv.index("--manifest") + 1])
    out_arg = Path(runner_argv[runner_argv.index("--out") + 1])
    if not manifest_arg.is_absolute():
        manifest_arg = cwd / manifest_arg
    if not out_arg.is_absolute():
        out_arg = cwd / out_arg
    require(
        manifest_arg.resolve(strict=False) == Path(str(receipt.get("manifest_path"))).resolve(strict=False),
        "runner argv manifest path mismatch",
    )
    require(
        out_arg.resolve(strict=False) == recorded_root.resolve(strict=False),
        "runner argv artifact root mismatch",
    )
    server_argv = receipt.get("server_argv")
    require(isinstance(server_argv, list) and len(server_argv) >= 2, "server argv missing")
    require(server_argv[0] == receipt.get("binary_path"), "server argv binary path mismatch")
    require(server_argv[1] == "serve", "server argv must invoke ferrum serve")
    require(model == server_argv[-1], "server argv model mismatch")
    require("--backend" in server_argv, "server argv missing typed backend")
    backend_index = server_argv.index("--backend")
    require(server_argv[backend_index + 1] == "cuda", "server argv backend must be cuda")
    require("--effective-config-json" in server_argv, "server argv missing effective config")
    require("--decision-trace-jsonl" in server_argv, "server argv missing decision trace")
    require(
        server_argv[server_argv.index("--effective-config-json") + 1]
        == str(recorded_root / "server.effective_config.json"),
        "server argv effective config path mismatch",
    )
    require(
        server_argv[server_argv.index("--decision-trace-jsonl") + 1]
        == str(recorded_root / "server.decision_trace.jsonl"),
        "server argv decision trace path mismatch",
    )
    hardware = receipt.get("hardware")
    require(isinstance(hardware, dict), "execution_receipt hardware missing")
    require(
        hardware.get("argv")
        == [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        "hardware probe argv mismatch",
    )
    require(hardware.get("returncode") == 0, "nvidia-smi hardware probe failed")
    hardware_rows = [line.strip() for line in str(hardware.get("stdout") or "").splitlines() if line.strip()]
    require(len(hardware_rows) == 1, "hardware receipt must contain exactly one GPU")
    require("RTX 4090" in hardware_rows[0], "hardware receipt is not RTX 4090")
    require(receipt.get("server_returncode") in (0, -15), "unexpected ferrum serve return code")
    for key in ("server_started_at", "server_finished_at"):
        require(isinstance(receipt.get(key), str) and receipt[key], f"execution_receipt {key} missing")
    child_env = receipt.get("child_env")
    require(isinstance(child_env, dict), "execution_receipt child_env must be an object")
    require(
        not any(str(key).startswith("FERRUM_") for key in child_env),
        "hidden FERRUM_* env leaked into server child",
    )
    evidence = receipt.get("evidence_files")
    require(isinstance(evidence, dict), "execution_receipt evidence_files missing")
    expected_files = {
        "effective_config": "server.effective_config.json",
        "decision_trace": "server.decision_trace.jsonl",
        "server_log": "server.log",
        "health_before": "server.health.json",
        "health_after": "server.health.after.json",
    }
    require(set(evidence) == set(expected_files), "execution receipt evidence file set mismatch")
    evidence_paths: dict[str, Path] = {}
    for key, relative in expected_files.items():
        item = evidence[key]
        require(isinstance(item, dict), f"execution evidence {key} must be an object")
        path = resolve_source_member(
            source_root,
            item.get("path"),
            f"execution_receipt.evidence_files.{key}.path",
            recorded_root=recorded_root,
            expected=source_root / relative,
        )
        require(path.stat().st_size > 0, f"execution evidence is empty: {relative}")
        require(item.get("size") == path.stat().st_size, f"execution evidence size mismatch: {relative}")
        require(item.get("sha256") == file_sha256(path), f"execution evidence SHA mismatch: {relative}")
        evidence_paths[key] = path
    effective = read_json(evidence_paths["effective_config"])
    require(effective.get("backend") == "cuda", "effective config backend must be cuda")
    require(effective.get("cuda_device_count") == 1, "effective config CUDA device count must be one")
    require(effective.get("selected_gpu_devices") == [0], "effective config selected GPU mismatch")
    capabilities = effective.get("model_capabilities")
    require(isinstance(capabilities, dict), "effective config model capabilities missing")
    require(capabilities.get("architecture") == "qwen3_5", "effective config architecture mismatch")
    hardware_capabilities = effective.get("hardware_capabilities")
    require(isinstance(hardware_capabilities, dict), "effective hardware capabilities missing")
    require(hardware_capabilities.get("backend") == "cuda", "effective hardware backend mismatch")
    compiled = hardware_capabilities.get("compiled_features")
    require(isinstance(compiled, dict) and compiled.get("cuda") is True, "binary lacks CUDA feature")
    for key in ("health_before", "health_after"):
        health = read_json(evidence_paths[key])
        require(health.get("status") == "pass", f"{key} status is not pass")
        require(health.get("http_status") == 200, f"{key} HTTP status is not 200")
    assert_clean_text("server.log", read_text(evidence_paths["server_log"]))
    assert_clean_text("server.decision_trace.jsonl", read_text(evidence_paths["decision_trace"]))
    return {
        "path": str(receipt_path),
        "file_sha256": file_sha256(receipt_path),
        "canonical_sha256": canonical_sha256,
        "binary_sha256": receipt["binary_sha256"],
        "hardware": hardware_rows,
        "effective_config_sha256": file_sha256(evidence_paths["effective_config"]),
    }


def validate_observability_paths(
    source_root: Path,
    recorded_root: Path,
    observability: Any,
) -> None:
    if observability is None:
        return
    require(isinstance(observability, dict), "summary.observability must be an object or null")
    path_lists = ("profile_paths", "scheduler_trace_paths", "request_dump_dirs")
    for key in path_lists:
        values = observability.get(key, [])
        require(isinstance(values, list), f"summary.observability.{key} must be a list")
        for index, value in enumerate(values):
            resolve_source_member(
                source_root,
                value,
                f"summary.observability.{key}[{index}]",
                recorded_root=recorded_root,
            )
    roots = observability.get("roots")
    if roots is None:
        return
    require(isinstance(roots, dict), "summary.observability.roots must be an object")
    for key, value in roots.items():
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        for index, item in enumerate(values):
            resolve_source_member(
                source_root,
                item,
                f"summary.observability.roots.{key}[{index}]",
                recorded_root=recorded_root,
            )


def is_qwen35_4b_model(value: Any) -> bool:
    if value == "Qwen/Qwen3.5-4B":
        return True
    return isinstance(value, str) and QWEN35_4B_CACHE_RE.search(value) is not None


def strict_schema_case(
    category: str, marker: str, ordinal: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    if category == "required":
        expected = {"required_value": marker}
        schema = {
            "type": "object",
            "properties": {"required_value": {"type": "string", "const": marker}},
            "required": ["required_value"],
            "additionalProperties": False,
        }
    elif category == "type":
        expected = {"case": marker, "ordinal": ordinal}
        schema = {
            "type": "object",
            "properties": {
                "case": {"type": "string", "const": marker},
                "ordinal": {"type": "integer", "const": ordinal},
            },
            "required": ["case", "ordinal"],
            "additionalProperties": False,
        }
    elif category == "additionalProperties":
        expected = {"marker": marker}
        schema = {
            "type": "object",
            "properties": {"marker": {"type": "string", "const": marker}},
            "required": ["marker"],
            "additionalProperties": False,
        }
    elif category == "enum":
        value = f"accepted-{marker}"
        expected = {"status": value}
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [value, f"rejected-{marker}"],
                }
            },
            "required": ["status"],
            "additionalProperties": False,
        }
    else:
        raise ValidationError(f"unsupported C14 category: {category}")
    return expected, schema


def expected_case(
    scenario_name: str, ordinal: int
) -> tuple[str, str | None, dict[str, Any], dict[str, Any] | None]:
    spec = SCENARIOS[scenario_name]
    marker = f"{scenario_name}-{ordinal:03d}"
    if spec["format"] == "json_schema":
        category = ("required", "type", "additionalProperties", "enum")[ordinal % 4]
        expected, schema = strict_schema_case(category, marker, ordinal)
        return marker, category, expected, schema
    return marker, None, {"marker": marker, "ordinal": ordinal}, None


def expected_response_format(
    scenario_name: str,
    ordinal: int,
    category: str | None,
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    if schema is None:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{scenario_name}-{category}-{ordinal:03d}",
            "strict": True,
            "schema": schema,
        },
    }


def reasoning_text(message: dict[str, Any], label: str) -> str:
    values: list[str] = []
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if value is None:
            continue
        require(isinstance(value, str), f"{label}.{key} must be a string or null")
        if value:
            values.append(value)
    require(len(values) <= 1, f"{label}: duplicate reasoning fields")
    return values[0] if values else ""


def validate_request(
    request: dict[str, Any],
    *,
    scenario_name: str,
    ordinal: int,
    model: str,
    marker: str,
    category: str | None,
    expected: dict[str, Any],
    schema: dict[str, Any] | None,
) -> None:
    spec = SCENARIOS[scenario_name]
    label = f"{scenario_name}/{ordinal:03d}/request.json"
    require(request.get("model") == model, f"{label}: model differs from summary")
    require(request.get("max_tokens") == 1024, f"{label}: max_tokens must be 1024")
    for key, value in SAMPLING[bool(spec["thinking"])].items():
        require(request.get(key) == value, f"{label}: sampling field {key} mismatch")
    require(
        request.get("chat_template_kwargs") == {"enable_thinking": spec["thinking"]},
        f"{label}: typed thinking setting mismatch",
    )
    prompt = "Return one JSON object and no other text. EXACT_JSON:" + json.dumps(
        expected, ensure_ascii=False, separators=(",", ":")
    )
    require(
        request.get("messages") == [{"role": "user", "content": prompt}],
        f"{label}: prompt does not bind the expected object",
    )
    require(marker in prompt, f"{label}: marker absent from prompt")
    require(
        request.get("response_format")
        == expected_response_format(scenario_name, ordinal, category, schema),
        f"{label}: response_format contract mismatch",
    )


def validate_case(
    source_root: Path,
    scenario_name: str,
    ordinal: int,
    model: str,
    scenario_case: Any,
) -> dict[str, Any]:
    spec = SCENARIOS[scenario_name]
    case_root = source_root / scenario_name / f"{ordinal:03d}"
    require(case_root.is_dir() and not case_root.is_symlink(), f"missing case directory: {case_root}")
    request_path = case_root / "request.json"
    response_path = case_root / "response.json"
    result_path = case_root / "result.json"
    request = read_json(request_path)
    response_text = read_text(response_path)
    assert_clean_text(f"{scenario_name}/{ordinal:03d}/response.json", response_text)
    try:
        response = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValidationError(f"malformed JSON {response_path}: {error}") from error
    require(isinstance(response, dict), f"response root is not an object: {response_path}")
    result = read_json(result_path)
    marker, category, expected, schema = expected_case(scenario_name, ordinal)
    validate_request(
        request,
        scenario_name=scenario_name,
        ordinal=ordinal,
        model=model,
        marker=marker,
        category=category,
        expected=expected,
        schema=schema,
    )
    require(response.get("object") == "chat.completion", f"{response_path}: response object mismatch")
    require(response.get("model") == model, f"{response_path}: model differs from request/summary")

    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, f"{response_path}: expected one choice")
    choice = choices[0]
    require(isinstance(choice, dict), f"{response_path}: choice must be an object")
    require(choice.get("index") == 0, f"{response_path}: choice index must be zero")
    require(choice.get("finish_reason") == "stop", f"{response_path}: finish_reason must be stop")
    message = choice.get("message")
    require(isinstance(message, dict), f"{response_path}: missing assistant message")
    require(message.get("role") == "assistant", f"{response_path}: assistant role mismatch")
    content = message.get("content")
    require(isinstance(content, str), f"{response_path}: content must be a string")
    require("```" not in content, f"{response_path}: markdown fence in final content")
    require(
        "<think>" not in content.lower() and "</think>" not in content.lower(),
        f"{response_path}: reasoning tag leaked into final content",
    )
    assert_clean_text(f"{scenario_name}/{ordinal:03d}/content", content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValidationError(f"{response_path}: final content is not one JSON value: {error}") from error
    require(isinstance(parsed, dict), f"{response_path}: final JSON root must be an object")
    require(parsed == expected, f"{response_path}: final object differs from expected")
    reasoning = reasoning_text(message, f"{scenario_name}/{ordinal:03d}/message")
    assert_clean_text(f"{scenario_name}/{ordinal:03d}/reasoning", reasoning)
    if spec["thinking"]:
        require(reasoning, f"{response_path}: thinking case has no separated reasoning")
    else:
        require(not reasoning, f"{response_path}: no-thinking case emitted reasoning")

    expected_result = {
        "status": "pass",
        "ordinal": ordinal,
        "format": spec["format"],
        "category": category,
        "preset": spec["preset"],
        "enable_thinking": spec["thinking"],
        "http_status": 200,
        "finish_reason": "stop",
        "reasoning_chars": len(reasoning),
        "object": expected,
    }
    require(result == expected_result, f"{result_path}: result does not match request/response")
    require(isinstance(scenario_case, dict), f"{scenario_name}: case {ordinal} summary is not an object")
    require(
        scenario_case == {key: value for key, value in expected_result.items() if key != "status"},
        f"{scenario_name}: case {ordinal} aggregate differs from case result",
    )
    return {
        "scenario": scenario_name,
        "ordinal": ordinal,
        "format": spec["format"],
        "category": category,
        "marker": marker,
        "expected_fingerprint": json_fingerprint(expected),
        "schema_fingerprint": json_fingerprint(schema) if schema is not None else None,
        "request_fingerprint": json_fingerprint(request),
    }


def validate_scenario(
    source_root: Path,
    recorded_root: Path,
    scenario_name: str,
    model: str,
    summary_scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    spec = SCENARIOS[scenario_name]
    scenario_root = source_root / scenario_name
    require(
        scenario_root.is_dir() and not scenario_root.is_symlink(),
        f"missing scenario directory: {scenario_root}",
    )
    result_path = scenario_root / "result.json"
    scenario_result = read_json(result_path)
    require(summary_scenario == scenario_result, f"summary scenario differs from {result_path}")
    expected_artifact = result_path.resolve(strict=False)
    resolve_source_member(
        source_root,
        scenario_result.get("artifact"),
        f"{scenario_name}.artifact",
        recorded_root=recorded_root,
        expected=expected_artifact,
    )
    require(scenario_result.get("name") == scenario_name, f"{scenario_name}: name mismatch")
    require(scenario_result.get("type") == SCENARIO_TYPE, f"{scenario_name}: type mismatch")
    require(scenario_result.get("status") == "pass", f"{scenario_name}: status is not pass")
    require(scenario_result.get("format") == spec["format"], f"{scenario_name}: format mismatch")
    require(scenario_result.get("preset") == spec["preset"], f"{scenario_name}: preset mismatch")
    require(
        scenario_result.get("enable_thinking") is spec["thinking"],
        f"{scenario_name}: thinking mode mismatch",
    )
    require(scenario_result.get("case_count") == spec["count"], f"{scenario_name}: case_count mismatch")
    require(
        scenario_result.get("category_counts") == spec["categories"],
        f"{scenario_name}: category partition mismatch",
    )
    cases = scenario_result.get("cases")
    require(isinstance(cases, list), f"{scenario_name}: cases must be a list")
    require(len(cases) == spec["count"], f"{scenario_name}: aggregate case cardinality mismatch")
    require(
        [case.get("ordinal") if isinstance(case, dict) else None for case in cases]
        == list(range(spec["count"])),
        f"{scenario_name}: aggregate ordinals are not exact and contiguous",
    )
    actual_case_dirs = {
        entry.name
        for entry in scenario_root.iterdir()
        if CASE_DIR_RE.fullmatch(entry.name) and entry.is_dir() and not entry.is_symlink()
    }
    expected_case_dirs = {f"{ordinal:03d}" for ordinal in range(spec["count"])}
    require(
        actual_case_dirs == expected_case_dirs,
        f"{scenario_name}: case directories differ; missing={sorted(expected_case_dirs - actual_case_dirs)}, "
        f"extra={sorted(actual_case_dirs - expected_case_dirs)}",
    )
    return [
        validate_case(source_root, scenario_name, ordinal, model, cases[ordinal])
        for ordinal in range(spec["count"])
    ]


def validate_contract(contract: dict[str, Any], case_rows: list[dict[str, Any]]) -> None:
    require(contract.get("schema_version") == 1, "matrix contract schema_version must be 1")
    require(contract.get("status") == "pass", "matrix contract status is not pass")
    require(contract.get("matrix_scenario_count") == 4, "matrix contract must contain four scenarios")
    require(
        contract.get("case_counts") == {"json_schema": 70, "json_object": 70},
        "matrix contract case_counts must be 70/70",
    )
    require(
        contract.get("unique_expected_object_counts")
        == {"json_schema": 70, "json_object": 70},
        "matrix contract must report 70 unique expected objects per format",
    )
    require(
        contract.get("unique_json_schema_count") == 70,
        "matrix contract must report 70 unique schemas",
    )
    expected_category_totals = {
        "required": 18,
        "type": 18,
        "additionalProperties": 17,
        "enum": 17,
    }
    require(
        contract.get("json_schema_category_counts") == expected_category_totals,
        "matrix contract C14 category totals mismatch",
    )
    contract_cases = contract.get("cases")
    require(isinstance(contract_cases, list), "matrix contract cases must be a list")
    require(len(contract_cases) == 140, "matrix contract must contain exactly 140 cases")
    expected_by_key = {(row["scenario"], row["ordinal"]): row for row in case_rows}
    actual_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for index, row in enumerate(contract_cases):
        require(isinstance(row, dict), f"matrix contract case {index} is not an object")
        scenario = row.get("scenario")
        ordinal = row.get("ordinal")
        require(isinstance(scenario, str), f"matrix contract case {index} missing scenario")
        require(isinstance(ordinal, int) and not isinstance(ordinal, bool), f"matrix contract case {index} bad ordinal")
        key = (scenario, ordinal)
        require(key not in actual_by_key, f"matrix contract duplicate case owner: {key}")
        actual_by_key[key] = row
    require(set(actual_by_key) == set(expected_by_key), "matrix contract owner set mismatch")
    contract_keys = {
        "scenario",
        "ordinal",
        "format",
        "category",
        "marker",
        "expected_fingerprint",
        "schema_fingerprint",
    }
    for key, expected in expected_by_key.items():
        actual = actual_by_key[key]
        require(
            {field: actual.get(field) for field in contract_keys}
            == {field: expected.get(field) for field in contract_keys},
            f"matrix contract evidence mismatch for {key}",
        )


def validate_summary(
    source_root: Path,
    summary: dict[str, Any],
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    require(summary.get("schema_version") == 1, "summary schema_version must be 1")
    require(summary.get("status") == "pass", "summary status is not pass")
    require(summary.get("backend") == "cuda", "summary backend must be cuda")
    model = summary.get("model")
    require(is_qwen35_4b_model(model), "summary model must be Qwen/Qwen3.5-4B")
    require(isinstance(model, str), "summary model must be a string")
    git_sha = summary.get("git_sha")
    require(
        isinstance(git_sha, str) and GIT_SHA_RE.fullmatch(git_sha),
        "summary git_sha must be 40 lowercase hex characters",
    )
    dirty = summary.get("dirty_status")
    require(isinstance(dirty, dict), "summary dirty_status must be an object")
    require(dirty.get("is_dirty") is False, "summary source checkout is dirty")
    require(dirty.get("status_short") == [], "summary dirty_status.status_short must be empty")
    require(summary.get("failed") == 0, "summary failed must be zero")
    require(summary.get("skipped") == 0, "summary skipped must be zero")
    require(summary.get("scenario_count") == 4, "summary scenario_count must be four")
    require(summary.get("manifest_scenario_count") == 4, "manifest must contain four scenarios")
    require(summary.get("requested_scenarios") == [], "official checkpoint forbids partial --only")
    require(
        summary.get("selected_scenarios") == list(SCENARIOS),
        "summary selected scenario set/order mismatch",
    )
    artifact_dir = summary.get("artifact_dir")
    require(
        isinstance(artifact_dir, str) and artifact_dir,
        "summary.artifact_dir must be a non-empty path string",
    )
    recorded_root = Path(artifact_dir)
    require(recorded_root.is_absolute(), "summary.artifact_dir must be absolute")
    require(".." not in recorded_root.parts, "summary.artifact_dir contains parent traversal")
    require(
        summary.get("pass_line") == f"BACKEND REGRESSION SMOKE PASS: {artifact_dir}",
        "summary generic runner PASS line mismatch",
    )
    execution_evidence = validate_execution_receipt(
        source_root,
        recorded_root,
        summary,
        git_sha=git_sha,
        model=model,
    )
    validate_observability_paths(source_root, recorded_root, summary.get("observability"))
    matrix_summary = summary.get("response_format_matrix_contract")
    require(isinstance(matrix_summary, dict), "summary missing response_format_matrix_contract")
    resolve_source_member(
        source_root,
        matrix_summary.get("artifact"),
        "summary.response_format_matrix_contract.artifact",
        recorded_root=recorded_root,
        expected=source_root / "response_format_matrix_contract.json",
    )
    require(
        matrix_summary.get("case_counts") == {"json_schema": 70, "json_object": 70},
        "summary matrix case counts mismatch",
    )
    require(
        matrix_summary.get("unique_json_schema_count") == 70,
        "summary unique schema count mismatch",
    )
    scenarios = summary.get("scenarios")
    require(isinstance(scenarios, list) and len(scenarios) == 4, "summary must contain four scenarios")
    names = [scenario.get("name") if isinstance(scenario, dict) else None for scenario in scenarios]
    require(names == list(SCENARIOS), f"summary scenario order/identity mismatch: {names}")
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        require(isinstance(scenario, dict), "summary scenario is not an object")
        name = str(scenario["name"])
        resolve_source_member(
            source_root,
            scenario.get("artifact"),
            f"summary.scenarios[{name}].artifact",
            recorded_root=recorded_root,
            expected=source_root / name / "result.json",
        )
        rows.extend(validate_scenario(source_root, recorded_root, name, model, scenario))
    return git_sha, model, rows, execution_evidence


def validate_source(source: Path) -> dict[str, Any]:
    require(source.exists() and source.is_dir(), f"source root is not a directory: {source}")
    require(not source.is_symlink(), f"source root must not be a symlink: {source}")
    source_root = source.resolve(strict=True)
    artifact_tree = validate_artifact_tree(source_root)
    summary_path = source_root / "summary.json"
    contract_path = source_root / "response_format_matrix_contract.json"
    summary = read_json(summary_path)
    git_sha, model, rows, execution_evidence = validate_summary(source_root, summary)
    require(len(rows) == 140, "validated case count must be 140")
    request_fingerprints = [row["request_fingerprint"] for row in rows]
    require(
        len(set(request_fingerprints)) == 140,
        f"request payload fingerprints are not unique: {len(set(request_fingerprints))}/140",
    )
    schema_fingerprints = [
        row["schema_fingerprint"] for row in rows if row["format"] == "json_schema"
    ]
    require(
        len(schema_fingerprints) == len(set(schema_fingerprints)) == 70,
        f"C14 inner schema fingerprints are not unique: {len(set(schema_fingerprints))}/70",
    )
    expected_counts = {
        format_type: len(
            {
                row["expected_fingerprint"]
                for row in rows
                if row["format"] == format_type
            }
        )
        for format_type in ("json_schema", "json_object")
    }
    require(
        expected_counts == {"json_schema": 70, "json_object": 70},
        f"expected object fingerprints are not unique per format: {expected_counts}",
    )
    contract = read_json(contract_path)
    validate_contract(contract, rows)
    category_counts = Counter(
        row["category"] for row in rows if row["format"] == "json_schema"
    )
    return {
        "git_sha": git_sha,
        "backend": "cuda",
        "model": model,
        "scenario_counts": {name: spec["count"] for name, spec in SCENARIOS.items()},
        "case_count": len(rows),
        "request_payload_unique_count": len(set(request_fingerprints)),
        "expected_object_unique_counts": expected_counts,
        "json_schema_unique_count": len(set(schema_fingerprints)),
        "json_schema_category_counts": dict(category_counts),
        "execution_receipt": execution_evidence,
        "artifact_tree": artifact_tree,
        "source_sha256": {
            "summary.json": file_sha256(summary_path),
            "response_format_matrix_contract.json": file_sha256(contract_path),
            "execution_receipt.json": file_sha256(source_root / "execution_receipt.json"),
            "artifact_tree.json": file_sha256(source_root / "artifact_tree.json"),
        },
    }


def base_manifest(source: Path, out: Path, started_at: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_id": CHECKPOINT_ID,
        "scope": ["S2/C14", "S2/C15"],
        "full_s2": False,
        "source_root": str(source.resolve(strict=False)),
        "artifact_dir": str(out.resolve(strict=False)),
        "started_at": started_at,
    }


def run_checkpoint(source: Path, out: Path) -> int:
    started_at = iso_now()
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    manifest = base_manifest(source, out, started_at)
    try:
        evidence = validate_source(source)
    except (OSError, ValidationError) as error:
        manifest.update(
            {
                "status": "fail",
                "finished_at": iso_now(),
                "duration_sec": time.monotonic() - started,
                "evidence": None,
                "pass_line": None,
                "error": str(error),
            }
        )
        write_json(out / "manifest.json", manifest)
        print(f"{FAIL_PREFIX}: {out}: {error}", file=sys.stderr)
        return 1
    pass_line = f"{PASS_PREFIX}: {out}"
    manifest.update(
        {
            "status": "pass",
            "finished_at": iso_now(),
            "duration_sec": time.monotonic() - started,
            "evidence": evidence,
            "pass_line": pass_line,
            "error": None,
        }
    )
    write_json(out / "manifest.json", manifest)
    print(pass_line)
    return 0


def fixture_scenario_result(
    source_root: Path,
    scenario_name: str,
    model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = SCENARIOS[scenario_name]
    cases: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    for ordinal in range(spec["count"]):
        marker, category, expected, schema = expected_case(scenario_name, ordinal)
        request = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Return one JSON object and no other text. EXACT_JSON:"
                    + json.dumps(expected, ensure_ascii=False, separators=(",", ":")),
                }
            ],
            "max_tokens": 1024,
            **SAMPLING[bool(spec["thinking"])],
            "chat_template_kwargs": {"enable_thinking": spec["thinking"]},
            "response_format": expected_response_format(
                scenario_name, ordinal, category, schema
            ),
        }
        reasoning = f"internal reasoning for {marker}" if spec["thinking"] else ""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": json.dumps(expected, ensure_ascii=False, separators=(",", ":")),
        }
        if reasoning:
            message["reasoning"] = reasoning
        response = {
            "id": f"fixture-{scenario_name}-{ordinal:03d}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        case_without_status = {
            "ordinal": ordinal,
            "format": spec["format"],
            "category": category,
            "preset": spec["preset"],
            "enable_thinking": spec["thinking"],
            "http_status": 200,
            "finish_reason": "stop",
            "reasoning_chars": len(reasoning),
            "object": expected,
        }
        case_root = source_root / scenario_name / f"{ordinal:03d}"
        write_json(case_root / "request.json", request)
        write_json(case_root / "response.json", response)
        write_json(case_root / "result.json", {"status": "pass", **case_without_status})
        cases.append(case_without_status)
        contract_rows.append(
            {
                "scenario": scenario_name,
                "ordinal": ordinal,
                "format": spec["format"],
                "category": category,
                "marker": marker,
                "expected_fingerprint": json_fingerprint(expected),
                "schema_fingerprint": json_fingerprint(schema) if schema is not None else None,
            }
        )
    result = {
        "status": "pass",
        "format": spec["format"],
        "preset": spec["preset"],
        "enable_thinking": spec["thinking"],
        "case_count": spec["count"],
        "category_counts": spec["categories"],
        "cases": cases,
        "name": scenario_name,
        "type": SCENARIO_TYPE,
        "artifact": str(source_root / scenario_name / "result.json"),
        "duration_sec": 1.0,
    }
    write_json(source_root / scenario_name / "result.json", result)
    return result, contract_rows


def fixture_execution_receipt(source_root: Path, model: str) -> dict[str, Any]:
    input_root = source_root / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    runner_path = input_root / "run_scenarios.py"
    manifest_path = input_root / "scenario_manifest.json"
    runner_path.write_text("# frozen fixture runner input\n", encoding="utf-8")
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "backend": "cuda",
            "model": model,
            "scenarios": [{"name": name} for name in SCENARIOS],
        },
    )
    effective_path = source_root / "server.effective_config.json"
    write_json(
        effective_path,
        {
            "schema_version": 1,
            "backend": "cuda",
            "cuda_device_count": 1,
            "requested_gpu_devices": [0],
            "selected_gpu_devices": [0],
            "model_capabilities": {"architecture": "qwen3_5"},
            "hardware_capabilities": {
                "backend": "cuda",
                "compiled_features": {"cuda": True},
            },
        },
    )
    decision_path = source_root / "server.decision_trace.jsonl"
    decision_path.write_text('{"selection":"backend","selected":"cuda"}\n', encoding="utf-8")
    log_path = source_root / "server.log"
    log_path.write_text("ferrum serve fixture process receipt\n", encoding="utf-8")
    health_before = source_root / "server.health.json"
    health_after = source_root / "server.health.after.json"
    write_json(health_before, {"status": "pass", "http_status": 200})
    write_json(health_after, {"status": "pass", "http_status": 200})
    runner_sha256 = file_sha256(runner_path)
    manifest_sha256 = file_sha256(manifest_path)
    binary_sha256 = "b" * 64
    evidence_paths = {
        "effective_config": effective_path,
        "decision_trace": decision_path,
        "server_log": log_path,
        "health_before": health_before,
        "health_after": health_after,
    }
    receipt = with_canonical_sha256(
        {
            "schema_version": 1,
            "mode": "start",
            "runner_argv": [
                sys.executable,
                str(runner_path),
                "--manifest",
                str(manifest_path),
                "--out",
                str(source_root),
            ],
            "runner_path": str(runner_path),
            "runner_sha256": runner_sha256,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "cwd": "/workspace/ferrum-infer-rs",
            "git_sha": "a" * 40,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "input_artifacts": {
                "runner": {"path": str(runner_path), "sha256": runner_sha256},
                "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
            },
            "backend": "cuda",
            "model": model,
            "selected_scenarios": list(SCENARIOS),
            "scenario_count": 4,
            "failed": 0,
            "skipped": 0,
            "server_argv": [
                "/workspace/target/release/ferrum",
                "serve",
                "--effective-config-json",
                str(effective_path),
                "--decision-trace-jsonl",
                str(decision_path),
                "--backend",
                "cuda",
                model,
            ],
            "binary_path": "/workspace/target/release/ferrum",
            "binary_sha256": binary_sha256,
            "hardware": {
                "argv": [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                "returncode": 0,
                "stdout": "0, NVIDIA GeForce RTX 4090, GPU-fixture, 24564, 570.00\n",
                "stderr": "",
            },
            "removed_hidden_env_names": [],
            "child_env": {"CUDA_VISIBLE_DEVICES": "0"},
            "server_started_at": "2026-07-20T00:00:00+00:00",
            "server_finished_at": "2026-07-20T00:01:00+00:00",
            "server_returncode": -15,
            "evidence_files": {
                key: {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for key, path in evidence_paths.items()
            },
        }
    )
    receipt_path = source_root / "execution_receipt.json"
    write_json(receipt_path, receipt)
    return {
        "artifact": str(receipt_path),
        "artifact_sha256": file_sha256(receipt_path),
        "canonical_sha256": receipt["canonical_sha256"],
        "mode": "start",
        "runner_sha256": runner_sha256,
        "manifest_sha256": manifest_sha256,
        "binary_sha256": binary_sha256,
    }


def write_fixture_artifact_tree(source_root: Path) -> None:
    entries = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name == "artifact_tree.json":
            continue
        entries.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    write_json(
        source_root / "artifact_tree.json",
        with_canonical_sha256(
            {
                "schema_version": 1,
                "artifact_root": str(source_root),
                "file_count": len(entries),
                "files": entries,
            }
        ),
    )


def create_fixture(source_root: Path) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    model = "Qwen/Qwen3.5-4B"
    scenario_results: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    for scenario_name in SCENARIOS:
        result, rows = fixture_scenario_result(source_root, scenario_name, model)
        scenario_results.append(copy.deepcopy(result))
        contract_rows.extend(rows)
    contract = {
        "schema_version": 1,
        "status": "pass",
        "matrix_scenario_count": 4,
        "case_counts": {"json_schema": 70, "json_object": 70},
        "unique_expected_object_counts": {"json_schema": 70, "json_object": 70},
        "unique_json_schema_count": 70,
        "json_schema_category_counts": {
            "required": 18,
            "type": 18,
            "additionalProperties": 17,
            "enum": 17,
        },
        "cases": contract_rows,
    }
    write_json(source_root / "response_format_matrix_contract.json", contract)
    execution_receipt = fixture_execution_receipt(source_root, model)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "manifest": "scripts/release/scenarios/runtime_vnext_s2_c14_c15_cuda.json",
        "artifact_dir": str(source_root),
        "model": model,
        "backend": "cuda",
        "base_url": "http://127.0.0.1:1",
        "git_sha": "a" * 40,
        "dirty_status": {"is_dirty": False, "status_short": []},
        "started_at": "2026-07-20T00:00:00+00:00",
        "finished_at": "2026-07-20T00:01:00+00:00",
        "scenario_count": 4,
        "manifest_scenario_count": 4,
        "requested_scenarios": [],
        "selected_scenarios": list(SCENARIOS),
        "failed": 0,
        "skipped": 0,
        "scenarios": scenario_results,
        "response_format_matrix_contract": {
            "artifact": str(source_root / "response_format_matrix_contract.json"),
            "case_counts": {"json_schema": 70, "json_object": 70},
            "unique_json_schema_count": 70,
        },
        "observability": None,
        "execution_receipt": execution_receipt,
        "pass_line": f"BACKEND REGRESSION SMOKE PASS: {source_root}",
    }
    write_json(source_root / "summary.json", summary)
    write_fixture_artifact_tree(source_root)


def mutate_duplicate_schema(source_root: Path) -> None:
    first = read_json(source_root / "m1_c14_no_thinking/000/request.json")
    second_path = source_root / "m1_c14_thinking/000/request.json"
    second = read_json(second_path)
    second["response_format"]["json_schema"]["schema"] = first["response_format"]["json_schema"]["schema"]
    write_json(second_path, second)


def mutate_missing_case(source_root: Path) -> None:
    shutil.rmtree(source_root / "m1_c15_no_thinking/049")


def mutate_length_finish(source_root: Path) -> None:
    scenario_name = "m1_c15_thinking"
    ordinal = 0
    response_path = source_root / scenario_name / "000/response.json"
    response = read_json(response_path)
    response["choices"][0]["finish_reason"] = "length"
    write_json(response_path, response)
    case_result_path = source_root / scenario_name / "000/result.json"
    case_result = read_json(case_result_path)
    case_result["finish_reason"] = "length"
    write_json(case_result_path, case_result)
    scenario_result_path = source_root / scenario_name / "result.json"
    scenario_result = read_json(scenario_result_path)
    scenario_result["cases"][ordinal]["finish_reason"] = "length"
    write_json(scenario_result_path, scenario_result)
    summary_path = source_root / "summary.json"
    summary = read_json(summary_path)
    for scenario in summary["scenarios"]:
        if scenario["name"] == scenario_name:
            scenario["cases"][ordinal]["finish_reason"] = "length"
    write_json(summary_path, summary)


def mutate_dirty_summary(source_root: Path) -> None:
    summary_path = source_root / "summary.json"
    summary = read_json(summary_path)
    summary["dirty_status"] = {"is_dirty": True, "status_short": [" M source.rs"]}
    write_json(summary_path, summary)


def mutate_empty_log(source_root: Path) -> None:
    (source_root / "server.log").write_text("", encoding="utf-8")


def mutate_wrong_binary_hash(source_root: Path) -> None:
    summary_path = source_root / "summary.json"
    summary = read_json(summary_path)
    summary["execution_receipt"]["binary_sha256"] = "c" * 64
    write_json(summary_path, summary)


def run_selftest_process(source: Path, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--source",
            str(source),
            "--out",
            str(out),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def self_test() -> int:
    mutations: list[tuple[str, Callable[[Path], None]]] = [
        ("duplicate-schema", mutate_duplicate_schema),
        ("missing-case", mutate_missing_case),
        ("length-finish", mutate_length_finish),
        ("dirty-summary", mutate_dirty_summary),
        ("empty-log", mutate_empty_log),
        ("wrong-binary-hash", mutate_wrong_binary_hash),
    ]
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-s2-response-format-") as temporary:
        root = Path(temporary)
        baseline = root / "baseline"
        create_fixture(baseline)
        baseline_out = root / "baseline-out"
        baseline_proc = run_selftest_process(baseline, baseline_out)
        expected_pass = f"{PASS_PREFIX}: {baseline_out}"
        require(baseline_proc.returncode == 0, baseline_proc.stderr or baseline_proc.stdout)
        require(expected_pass in baseline_proc.stdout.splitlines(), "self-test baseline missing exact PASS line")
        baseline_manifest = read_json(baseline_out / "manifest.json")
        require(baseline_manifest.get("status") == "pass", "self-test baseline manifest did not pass")
        require(baseline_manifest.get("full_s2") is False, "checkpoint must not claim full S2")
        for name, mutate in mutations:
            source = root / name
            create_fixture(source)
            mutate(source)
            write_fixture_artifact_tree(source)
            out = root / f"{name}-out"
            proc = run_selftest_process(source, out)
            require(proc.returncode != 0, f"mutation unexpectedly passed: {name}")
            manifest = read_json(out / "manifest.json")
            require(manifest.get("status") == "fail", f"mutation missing fail manifest: {name}")
            require(manifest.get("pass_line") is None, f"mutation retained PASS line: {name}")
    print(SELFTEST_PASS_LINE)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "--artifact-root", dest="source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.source is not None or args.out is not None:
            parser.error("--self-test cannot be combined with --source/--out")
    elif args.source is None or args.out is None:
        parser.error("--source and --out are required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        try:
            return self_test()
        except (OSError, ValidationError) as error:
            print(f"{SELFTEST_PASS_LINE.replace(' PASS', ' FAIL')}: {error}", file=sys.stderr)
            return 1
    return run_checkpoint(args.source, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
