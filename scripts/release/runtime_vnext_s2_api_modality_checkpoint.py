#!/usr/bin/env python3
"""Validate only the Runtime vNext S2 C16/C20 CUDA API/modality slice."""

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


PASS_PREFIX = "FERRUM RUNTIME VNEXT S2 API MODALITY PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT S2 API MODALITY FAIL"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT S2 API MODALITY SELFTEST PASS"
CHECKPOINT_ID = "runtime-vnext-s2-api-modality-c16-c20"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QWEN35_4B_CACHE_RE = re.compile(
    r"(?:^|/)models--Qwen--Qwen3\.5-4B/snapshots/[0-9a-f]{40}/?$"
)
C16_NAME = "m1_c16_negative_api_matrix"
C20_NAME = "m1_c20_text_only_modality_matrix"
SCENARIO_NAMES = (C16_NAME, C20_NAME)
C16_TYPE = "serve_negative_api_matrix"
C20_TYPE = "serve_text_only_modality_matrix"
C16_CATEGORIES = (
    "invalid-tool",
    "invalid-schema",
    "invalid-stream-option",
    "invalid-model",
    "invalid-context",
)
C20_CATEGORIES = (
    "image-url",
    "data-url",
    "video-url",
    "mixed-text-media",
    "text-array",
)
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
DETERMINISTIC_FIELDS = {
    "max_tokens",
    *DETERMINISTIC_SAMPLING.keys(),
    "chat_template_kwargs",
}
C20_REMOTE_MEDIA_URL = (
    "https://raw.githubusercontent.com/sizzlecar/ferrum-infer-rs/"
    "cff4c47765ef3259b8a04890187d99c60da86394/"
    "docs/bench/framework-validation-2026-05-25/m3_layerwise.png"
)
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
EXECUTION_EVIDENCE = {
    "effective_config": "server.effective_config.json",
    "decision_trace": "server.decision_trace.jsonl",
    "server_log": "server.log",
    "health_before": "server.health.json",
    "health_after": "server.health.after.json",
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


def assert_clean_text(label: str, text: str) -> None:
    require("\ufffd" not in text, f"{label}: Unicode replacement character")
    require("\x00" not in text, f"{label}: NUL byte")
    lowered = text.lower()
    for token in BAD_TEXT:
        require(token not in lowered, f"{label}: forbidden text {token!r}")


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def with_canonical_sha256(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["canonical_sha256_scope"] = "document_without_canonical_sha256_fields"
    result["canonical_sha256"] = json_fingerprint(result)
    return result


def validate_result_self_hash(result: dict[str, Any], label: str) -> None:
    require(
        result.get("canonical_sha256_scope")
        == "document_without_canonical_sha256_fields",
        f"{label}: canonical SHA scope mismatch",
    )
    saved = result.get("canonical_sha256")
    require(isinstance(saved, str) and SHA256_RE.fullmatch(saved), f"{label}: bad canonical SHA")
    unhashed = copy.deepcopy(result)
    unhashed.pop("canonical_sha256", None)
    require(json_fingerprint(unhashed) == saved, f"{label}: canonical self hash mismatch")


def resolve_source_member(
    source_root: Path,
    value: Any,
    label: str,
    *,
    recorded_root: Path,
    expected: Path | None = None,
) -> Path:
    require(isinstance(value, str) and value, f"{label} must be a non-empty path string")
    raw = Path(value)
    require(".." not in raw.parts, f"{label} contains parent traversal: {value}")
    if raw.is_absolute():
        require(recorded_root.is_absolute(), f"{label}: recorded root must be absolute")
        try:
            relative = raw.relative_to(recorded_root)
        except ValueError as error:
            raise ValidationError(
                f"{label} is outside recorded artifact root {recorded_root}: {value}"
            ) from error
        candidate = source_root / relative
    else:
        candidate = source_root / raw
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(source_root)
    except ValueError as error:
        raise ValidationError(f"{label} escapes source root: {value}") from error
    if expected is not None:
        require(resolved == expected.resolve(strict=False), f"{label} does not name {expected}")
    require(resolved.exists(), f"{label} does not exist: {resolved}")
    return resolved


def is_qwen35_4b_model(value: Any) -> bool:
    if value == "Qwen/Qwen3.5-4B":
        return True
    return isinstance(value, str) and QWEN35_4B_CACHE_RE.search(value) is not None


def function_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


def c16_case_spec(
    category: str, ordinal: int, model: str
) -> tuple[str, dict[str, Any], str | None, str]:
    if category == "invalid-tool":
        variants: list[tuple[str, dict[str, Any], str | None, str]] = [
            (
                "required-without-tools",
                {"tool_choice": "required"},
                "tool_choice",
                "requires at least one function tool",
            ),
            (
                "unsupported-choice-mode",
                {"tool_choice": "forced"},
                "tool_choice",
                "unsupported tool_choice mode",
            ),
            (
                "unsupported-tool-type",
                {"tools": [{**function_tool("search"), "type": "retrieval"}]},
                "tools",
                "only function tools are supported",
            ),
            (
                "unsupported-choice-type",
                {
                    "tools": [function_tool("weather")],
                    "tool_choice": {
                        "type": "retrieval",
                        "function": {"name": "weather"},
                    },
                },
                "tool_choice",
                "only function tool_choice is supported",
            ),
            (
                "undeclared-tool-choice",
                {
                    "tools": [function_tool("weather")],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "calendar"},
                    },
                },
                "tool_choice",
                "not declared in tools",
            ),
            (
                "undeclared-legacy-function",
                {
                    "functions": [function_tool("weather")["function"]],
                    "function_call": {"name": "calendar"},
                },
                "function_call",
                "not declared in functions",
            ),
        ]
    elif category == "invalid-schema":
        variants = [
            (
                "unsupported-format-type",
                {"response_format": {"type": "xml"}},
                "response_format.type",
                "unsupported response_format.type",
            ),
            (
                "missing-json-schema-config",
                {"response_format": {"type": "json_schema"}},
                "response_format.json_schema",
                "schema is required",
            ),
            (
                "null-json-schema-config",
                {"response_format": {"type": "json_schema", "json_schema": None}},
                "response_format.json_schema",
                "schema is required",
            ),
            (
                "missing-schema-body",
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "missing", "strict": True},
                    }
                },
                "response_format.json_schema",
                "schema is required",
            ),
            (
                "invalid-schema-type",
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "bad-type",
                            "strict": True,
                            "schema": {"type": "definitely-not-a-json-type"},
                        },
                    }
                },
                "response_format.json_schema",
                "unsupported strict json_schema",
            ),
            (
                "invalid-required-keyword",
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "bad-required",
                            "strict": True,
                            "schema": {"type": "object", "required": "name"},
                        },
                    }
                },
                "response_format.json_schema",
                "unsupported strict json_schema",
            ),
        ]
    elif category == "invalid-stream-option":
        variants = [
            (
                "options-with-stream-false",
                {"stream": False, "stream_options": {"include_usage": True}},
                "stream_options",
                "only valid when stream=true",
            ),
            (
                "disabled-usage-with-stream-false",
                {"stream": False, "stream_options": {"include_usage": False}},
                "stream_options",
                "only valid when stream=true",
            ),
            (
                "empty-options-with-stream-false",
                {"stream": False, "stream_options": {}},
                "stream_options",
                "only valid when stream=true",
            ),
            (
                "unknown-stream-option",
                {"stream": True, "stream_options": {"continuous_usage_stats": True}},
                None,
                "invalid chat completions request",
            ),
            (
                "non-boolean-include-usage",
                {"stream": True, "stream_options": {"include_usage": "yes"}},
                None,
                "invalid chat completions request",
            ),
            (
                "non-object-stream-options",
                {"stream": True, "stream_options": []},
                None,
                "invalid chat completions request",
            ),
        ]
    elif category == "invalid-model":
        case_variant = model.swapcase()
        if case_variant == model:
            case_variant = model + "-CASE-MISMATCH"
        variants = [
            ("unknown-name", {"model": "not-a-loaded-model"}, "model", "unknown model"),
            ("empty-name", {"model": ""}, "model", "unknown model"),
            ("whitespace-name", {"model": " "}, "model", "unknown model"),
            (
                "leading-whitespace",
                {"model": " " + model},
                "model",
                "unknown model",
            ),
            ("case-mismatch", {"model": case_variant}, "model", "unknown model"),
            (
                "missing-adapter",
                {"model": model + ":missing-adapter"},
                "model",
                "unknown model",
            ),
        ]
    elif category == "invalid-context":
        variants = [
            (
                "sync-max-tokens",
                {"max_tokens": 1_000_000_000},
                None,
                "This model context is limited to",
            ),
            (
                "max-completion-tokens",
                {"max_tokens": None, "max_completion_tokens": 1_000_000_001},
                None,
                "This model context is limited to",
            ),
            (
                "max-completion-precedence",
                {"max_tokens": 1, "max_completion_tokens": 1_000_000_002},
                None,
                "This model context is limited to",
            ),
            (
                "stream-context-budget",
                {"stream": True, "max_tokens": 1_000_000_003},
                None,
                "This model context is limited to",
            ),
            (
                "text-array-context-budget",
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "x"}]}
                    ],
                    "max_tokens": 1_000_000_004,
                },
                None,
                "This model context is limited to",
            ),
            (
                "tool-template-context-budget",
                {
                    "tools": [function_tool("weather")],
                    "tool_choice": "auto",
                    "max_tokens": 1_000_000_005,
                },
                None,
                "This model context is limited to",
            ),
        ]
    else:
        raise ValidationError(f"unsupported C16 category: {category}")
    require(0 <= ordinal < len(variants), f"bad C16 ordinal: {ordinal}")
    return variants[ordinal]


def apply_patch_values(payload: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = copy.deepcopy(value)


def expected_c16_request(
    category: str, ordinal: int, model: str
) -> tuple[str, dict[str, Any], str | None, str, dict[str, Any]]:
    marker = f"c16-{category}-{ordinal:03d}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"C16 boundary case {marker}; reply with the marker.",
            }
        ],
        "max_tokens": 1024,
        **DETERMINISTIC_SAMPLING,
        "chat_template_kwargs": {"enable_thinking": False},
        "metadata": {
            "ferrum_regression_case": marker,
            "ferrum_regression_category": category,
        },
    }
    variant, patch, expected_param, expected_message = c16_case_spec(category, ordinal, model)
    apply_patch_values(payload, patch)
    failure_contract = {"category": category, "request_patch": patch}
    return marker, payload, expected_param, expected_message, {
        "variant": variant,
        "failure_contract": failure_contract,
    }


def expected_c20_request(category: str, ordinal: int, model: str) -> tuple[str, dict[str, Any]]:
    marker = f"c20-{category}-{ordinal:03d}"
    prompt = f"Reply with exactly this marker and no other text: {marker}"
    if category == "text-array":
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Reply with exactly this marker and no other text:"},
            {"type": "text", "text": marker},
        ]
    elif category == "data-url":
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,"
                    + hashlib.sha256(marker.encode("utf-8")).hexdigest()
                },
            }
        ]
    elif category == "video-url":
        content = [
            {
                "type": "video_url",
                "video_url": {"url": f"{C20_REMOTE_MEDIA_URL}?ferrum_case={marker}"},
            }
        ]
    else:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"{C20_REMOTE_MEDIA_URL}?ferrum_case={marker}"},
            }
        ]
        if category == "mixed-text-media":
            content.insert(0, {"type": "text", "text": prompt})
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if category == "text-array":
        payload.update({"max_tokens": 1024, **DETERMINISTIC_SAMPLING})
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    payload["metadata"] = {
        "ferrum_regression_case": marker,
        "ferrum_regression_category": category,
        "ferrum_expected_marker": marker,
    }
    return marker, payload


def parse_openai_error(response: dict[str, Any], label: str) -> dict[str, Any]:
    error = response.get("error")
    require(isinstance(error, dict), f"{label}: missing OpenAI error object")
    require(error.get("type") == "invalid_request_error", f"{label}: bad error type")
    message = error.get("message")
    require(isinstance(message, str) and message, f"{label}: missing error message")
    assert_clean_text(f"{label}.message", message)
    require("param" in error, f"{label}: missing error param")
    require(
        error.get("param") is None or isinstance(error.get("param"), str),
        f"{label}: invalid error param",
    )
    if "code" in error:
        require(error.get("code") is None, f"{label}: unexpected error code")
    return error


def choice_message(response: dict[str, Any], label: str) -> tuple[dict[str, Any], str | None]:
    require(response.get("object") == "chat.completion", f"{label}: response object mismatch")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, f"{label}: expected one choice")
    choice = choices[0]
    require(isinstance(choice, dict), f"{label}: choice is not an object")
    message = choice.get("message")
    require(isinstance(message, dict), f"{label}: missing assistant message")
    return message, choice.get("finish_reason")


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def response_message_text(response: dict[str, Any], label: str) -> tuple[str, str | None]:
    message, finish = choice_message(response, label)
    value = message.get("content") or message.get("reasoning") or message.get("reasoning_content") or ""
    require(isinstance(value, str), f"{label}: assistant text must be a string")
    text = strip_think(value)
    assert_clean_text(f"{label}.assistant_text", text)
    return text, finish


def expected_tree_paths(source_root: Path) -> set[str]:
    return {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "artifact_tree.json"
    }


def validate_artifact_tree(source_root: Path, recorded_root: Path) -> dict[str, Any]:
    for path in source_root.rglob("*"):
        require(not path.is_symlink(), f"artifact tree contains symlink: {path}")
    tree_path = source_root / "artifact_tree.json"
    tree = read_json(tree_path)
    validate_result_self_hash(tree, "artifact_tree.json")
    require(tree.get("schema_version") == 1, "artifact_tree schema_version mismatch")
    require(tree.get("artifact_root") == str(recorded_root), "artifact_tree artifact_root mismatch")
    files = tree.get("files")
    require(isinstance(files, list), "artifact_tree.files must be a list")
    require(tree.get("file_count") == len(files), "artifact_tree file_count mismatch")
    actual_paths = expected_tree_paths(source_root)
    entries: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(files):
        require(isinstance(raw_entry, dict), f"artifact_tree.files[{index}] is not an object")
        require(
            set(raw_entry) == {"path", "size", "sha256"},
            f"artifact_tree.files[{index}] field set mismatch",
        )
        relative = raw_entry.get("path")
        require(isinstance(relative, str) and relative, f"artifact_tree.files[{index}] bad path")
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, f"artifact_tree path escapes: {relative}")
        require(relative not in entries, f"artifact_tree duplicate path: {relative}")
        require(relative != "artifact_tree.json", "artifact_tree must not hash itself")
        candidate = (source_root / path).resolve(strict=False)
        try:
            candidate.relative_to(source_root)
        except ValueError as error:
            raise ValidationError(f"artifact_tree path escapes source root: {relative}") from error
        require(candidate.is_file() and not candidate.is_symlink(), f"artifact_tree file missing: {relative}")
        size = raw_entry.get("size")
        sha = raw_entry.get("sha256")
        require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, f"bad size: {relative}")
        require(isinstance(sha, str) and SHA256_RE.fullmatch(sha), f"bad SHA256: {relative}")
        require(candidate.stat().st_size == size, f"artifact_tree size mismatch: {relative}")
        require(file_sha256(candidate) == sha, f"artifact_tree SHA256 mismatch: {relative}")
        if candidate.suffix.lower() in {".json", ".jsonl", ".log", ".sse", ".txt"}:
            assert_clean_text(f"artifact_tree textual file {relative}", read_text(candidate))
        entries[relative] = raw_entry
    require(
        set(entries) == actual_paths,
        "artifact_tree does not exactly cover source files; "
        f"missing={sorted(actual_paths - set(entries))}, extra={sorted(set(entries) - actual_paths)}",
    )
    required_paths = {
        "summary.json",
        "execution_receipt.json",
        "response_format_matrix_contract.json",
        "inputs/run_scenarios.py",
        "inputs/scenario_manifest.json",
        *(f"{name}/result.json" for name in SCENARIO_NAMES),
        *(EXECUTION_EVIDENCE.values()),
    }
    require(required_paths <= set(entries), "artifact_tree omits checkpoint source evidence")
    return {"file_count": len(entries), "sha256": file_sha256(tree_path)}


def validate_execution_receipt(
    source_root: Path,
    recorded_root: Path,
    summary: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    receipt_path = source_root / "execution_receipt.json"
    receipt = read_json(receipt_path)
    validate_result_self_hash(receipt, "execution_receipt.json")
    require(receipt.get("schema_version") == 1, "execution receipt schema_version mismatch")
    require(receipt.get("mode") == "start", "execution receipt must prove start mode")
    require(receipt.get("git_sha") == summary.get("git_sha"), "execution receipt git SHA mismatch")
    require(
        receipt.get("dirty_status") == summary.get("dirty_status"),
        "execution receipt dirty status mismatch",
    )
    require(receipt.get("backend") == "cuda", "execution receipt backend mismatch")
    require(receipt.get("model") == model, "execution receipt model mismatch")
    require(
        receipt.get("selected_scenarios") == list(SCENARIO_NAMES),
        "execution receipt selected scenario set/order mismatch",
    )
    require(receipt.get("scenario_count") == 2, "execution receipt scenario_count mismatch")
    require(receipt.get("failed") == 0, "execution receipt failed must be zero")
    require(receipt.get("skipped") == 0, "execution receipt skipped must be zero")
    runner_path = receipt.get("runner_path")
    require(
        isinstance(runner_path, str) and Path(runner_path).name == "run_scenarios.py",
        "execution receipt runner_path mismatch",
    )
    runner_sha = receipt.get("runner_sha256")
    manifest_sha = receipt.get("manifest_sha256")
    binary_sha = receipt.get("binary_sha256")
    for label, value in (
        ("runner_sha256", runner_sha),
        ("manifest_sha256", manifest_sha),
        ("binary_sha256", binary_sha),
    ):
        require(isinstance(value, str) and SHA256_RE.fullmatch(value), f"execution receipt {label} invalid")
    manifest_path = receipt.get("manifest_path")
    require(
        isinstance(manifest_path, str)
        and Path(manifest_path).name == "runtime_vnext_s2_c16_c20_cuda.json",
        "execution receipt manifest_path mismatch",
    )
    runner_argv = receipt.get("runner_argv")
    require(
        isinstance(runner_argv, list)
        and all(isinstance(item, str) for item in runner_argv)
        and runner_path in runner_argv,
        "execution receipt runner_argv mismatch",
    )
    cwd = receipt.get("cwd")
    require(isinstance(cwd, str) and Path(cwd).is_absolute(), "execution receipt cwd must be absolute")
    require("--manifest" in runner_argv, "execution receipt argv omits --manifest")
    manifest_index = runner_argv.index("--manifest")
    require(manifest_index + 1 < len(runner_argv), "execution receipt --manifest has no value")
    argv_manifest = Path(runner_argv[manifest_index + 1])
    if not argv_manifest.is_absolute():
        argv_manifest = Path(cwd) / argv_manifest
    require(
        argv_manifest.resolve(strict=False) == Path(manifest_path).resolve(strict=False),
        "execution receipt argv does not bind manifest_path",
    )
    summary_manifest = summary.get("manifest")
    require(isinstance(summary_manifest, str) and summary_manifest, "summary manifest path missing")
    summary_manifest_path = Path(summary_manifest)
    if not summary_manifest_path.is_absolute():
        summary_manifest_path = Path(cwd) / summary_manifest_path
    require(
        summary_manifest_path.resolve(strict=False) == Path(manifest_path).resolve(strict=False),
        "summary manifest does not bind execution receipt manifest",
    )
    require("--out" in runner_argv, "execution receipt argv omits --out")
    out_index = runner_argv.index("--out")
    require(out_index + 1 < len(runner_argv), "execution receipt --out has no value")
    argv_out = Path(runner_argv[out_index + 1])
    if not argv_out.is_absolute():
        argv_out = Path(cwd) / argv_out
    require(argv_out == recorded_root, "execution receipt argv does not bind artifact root")
    input_artifacts = receipt.get("input_artifacts")
    require(isinstance(input_artifacts, dict), "execution receipt input_artifacts missing")
    require(set(input_artifacts) == {"runner", "manifest"}, "execution input artifact set mismatch")
    for key, relative, expected_sha in (
        ("runner", "inputs/run_scenarios.py", runner_sha),
        ("manifest", "inputs/scenario_manifest.json", manifest_sha),
    ):
        row = input_artifacts[key]
        require(isinstance(row, dict), f"execution input artifact {key} is not an object")
        require(set(row) == {"path", "sha256"}, f"execution input artifact {key} field set mismatch")
        input_path = resolve_source_member(
            source_root,
            row.get("path"),
            f"execution input artifact {key}.path",
            recorded_root=recorded_root,
            expected=source_root / relative,
        )
        require(row.get("sha256") == expected_sha, f"execution input artifact {key} SHA mismatch")
        require(file_sha256(input_path) == expected_sha, f"execution input artifact {key} file mismatch")
    binary_path = receipt.get("binary_path")
    require(isinstance(binary_path, str) and Path(binary_path).is_absolute(), "binary_path must be absolute")
    server_argv = receipt.get("server_argv")
    require(
        isinstance(server_argv, list)
        and all(isinstance(item, str) for item in server_argv)
        and server_argv
        and server_argv[0] == binary_path,
        "execution receipt server_argv mismatch",
    )
    require("serve" in server_argv, "execution receipt did not run ferrum serve")
    require(model in server_argv, "execution receipt server command omits model")
    require(
        any(server_argv[index : index + 2] == ["--backend", "cuda"] for index in range(len(server_argv) - 1)),
        "execution receipt server command omits typed CUDA backend",
    )
    for flag, relative in (
        ("--effective-config-json", "server.effective_config.json"),
        ("--decision-trace-jsonl", "server.decision_trace.jsonl"),
    ):
        require(flag in server_argv, f"execution receipt server command omits {flag}")
        index = server_argv.index(flag)
        require(index + 1 < len(server_argv), f"execution receipt {flag} has no value")
        require(Path(server_argv[index + 1]) == recorded_root / relative, f"server {flag} path mismatch")
    hardware = receipt.get("hardware")
    require(isinstance(hardware, dict), "execution receipt hardware probe missing")
    require(
        hardware.get("argv")
        == [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        "hardware probe command mismatch",
    )
    require(hardware.get("returncode") == 0, "hardware probe did not pass")
    hardware_stdout = hardware.get("stdout")
    hardware_stderr = hardware.get("stderr")
    require(isinstance(hardware_stdout, str), "hardware stdout missing")
    require(isinstance(hardware_stderr, str), "hardware stderr missing")
    assert_clean_text("execution receipt hardware stdout", hardware_stdout)
    assert_clean_text("execution receipt hardware stderr", hardware_stderr)
    gpu_rows = [row for row in hardware_stdout.splitlines() if row.strip()]
    require(len(gpu_rows) == 1 and "RTX 4090" in gpu_rows[0], "hardware receipt must prove one RTX 4090")
    removed = receipt.get("removed_hidden_env_names")
    require(
        isinstance(removed, list)
        and all(isinstance(key, str) and key.startswith("FERRUM_") for key in removed)
        and len(removed) == len(set(removed)),
        "removed hidden env names receipt invalid",
    )
    child_env = receipt.get("child_env")
    require(isinstance(child_env, dict), "execution receipt child_env missing")
    require(not any(str(key).startswith("FERRUM_") for key in child_env), "hidden FERRUM env reached child")
    for key, value in child_env.items():
        require(isinstance(key, str) and isinstance(value, str), "child_env must contain strings")
        assert_clean_text(f"execution receipt child_env.{key}", value)
    for field in ("server_started_at", "server_finished_at"):
        require(isinstance(receipt.get(field), str) and receipt[field], f"execution receipt {field} missing")
    require(receipt.get("server_returncode") in (0, -15), "unexpected server return code")
    evidence = receipt.get("evidence_files")
    require(isinstance(evidence, dict) and set(evidence) == set(EXECUTION_EVIDENCE), "execution evidence set mismatch")
    for label, relative in EXECUTION_EVIDENCE.items():
        row = evidence[label]
        require(isinstance(row, dict), f"execution evidence {label} is not an object")
        require(set(row) == {"path", "size", "sha256"}, f"execution evidence {label} field set mismatch")
        path = resolve_source_member(
            source_root,
            row.get("path"),
            f"execution evidence {label}.path",
            recorded_root=recorded_root,
            expected=source_root / relative,
        )
        require(row.get("size") == path.stat().st_size and row["size"] > 0, f"execution evidence {label} size mismatch")
        require(row.get("sha256") == file_sha256(path), f"execution evidence {label} SHA mismatch")
    summary_receipt = summary.get("execution_receipt")
    require(isinstance(summary_receipt, dict), "summary execution_receipt missing")
    resolve_source_member(
        source_root,
        summary_receipt.get("artifact"),
        "summary.execution_receipt.artifact",
        recorded_root=recorded_root,
        expected=receipt_path,
    )
    require(summary_receipt.get("mode") == "start", "summary execution mode mismatch")
    require(summary_receipt.get("runner_sha256") == runner_sha, "summary runner SHA mismatch")
    require(summary_receipt.get("manifest_sha256") == manifest_sha, "summary manifest SHA mismatch")
    require(summary_receipt.get("binary_sha256") == binary_sha, "summary binary SHA mismatch")
    require(
        summary_receipt.get("canonical_sha256") == receipt.get("canonical_sha256"),
        "summary execution receipt canonical SHA mismatch",
    )
    require(
        summary_receipt.get("artifact_sha256") == file_sha256(receipt_path),
        "summary execution receipt file SHA mismatch",
    )
    effective = read_json(source_root / EXECUTION_EVIDENCE["effective_config"])
    require(effective.get("backend") == "cuda", "effective config backend must be cuda")
    require(effective.get("cuda_device_count") == 1, "effective config CUDA device count mismatch")
    require(effective.get("selected_gpu_devices") == [0], "effective config selected GPU mismatch")
    model_capabilities = effective.get("model_capabilities")
    require(isinstance(model_capabilities, dict), "effective config model capabilities missing")
    require(model_capabilities.get("architecture") == "qwen3_5", "effective architecture mismatch")
    hardware_capabilities = effective.get("hardware_capabilities")
    require(isinstance(hardware_capabilities, dict), "effective hardware capabilities missing")
    require(hardware_capabilities.get("backend") == "cuda", "effective hardware backend mismatch")
    compiled = hardware_capabilities.get("compiled_features")
    require(isinstance(compiled, dict) and compiled.get("cuda") is True, "effective CUDA feature missing")
    for relative in ("server.health.json", "server.health.after.json"):
        health = read_json(source_root / relative)
        require(health.get("status") == "pass", f"{relative} status is not pass")
        require(health.get("http_status") == 200, f"{relative} HTTP status mismatch")
    assert_clean_text("server.log", read_text(source_root / "server.log"))
    assert_clean_text(
        "server.decision_trace.jsonl",
        read_text(source_root / "server.decision_trace.jsonl"),
    )
    return {
        "mode": "start",
        "runner_sha256": runner_sha,
        "manifest_sha256": manifest_sha,
        "binary_sha256": binary_sha,
        "receipt_sha256": file_sha256(receipt_path),
    }


def validate_case_artifact_paths(
    source_root: Path,
    recorded_root: Path,
    case_root: Path,
    result: dict[str, Any],
    label: str,
) -> None:
    resolve_source_member(
        source_root,
        result.get("request_artifact"),
        f"{label}.request_artifact",
        recorded_root=recorded_root,
        expected=case_root / "request.json",
    )
    resolve_source_member(
        source_root,
        result.get("response_artifact"),
        f"{label}.response_artifact",
        recorded_root=recorded_root,
        expected=case_root / "response.json",
    )


def require_exact_regular_files(directory: Path, names: set[str], label: str) -> None:
    actual = {
        entry.name
        for entry in directory.iterdir()
        if entry.is_file() and not entry.is_symlink()
    }
    require(actual == names, f"{label}: regular file set mismatch: {sorted(actual)}")


def validate_c16_case(
    source_root: Path,
    recorded_root: Path,
    category: str,
    ordinal: int,
    model: str,
    aggregate_case: Any,
) -> dict[str, Any]:
    label = f"{C16_NAME}/{category}/{ordinal:03d}"
    case_root = source_root / C16_NAME / category / f"{ordinal:03d}"
    require(case_root.is_dir() and not case_root.is_symlink(), f"missing case directory: {case_root}")
    require_exact_regular_files(
        case_root, {"request.json", "response.json", "result.json"}, label
    )
    request = read_json(case_root / "request.json")
    response = read_json(case_root / "response.json")
    result = read_json(case_root / "result.json")
    validate_result_self_hash(result, f"{label}/result.json")
    require(aggregate_case == result, f"{label}: aggregate differs from result.json")
    validate_case_artifact_paths(source_root, recorded_root, case_root, result, label)
    marker, expected_request, expected_param, expected_message, spec = expected_c16_request(
        category, ordinal, model
    )
    require(request == expected_request, f"{label}: request contract mismatch")
    require(result.get("schema_version") == 1, f"{label}: result schema_version mismatch")
    require(result.get("status") == "pass", f"{label}: status is not pass")
    require(result.get("case_id") == marker, f"{label}: case_id mismatch")
    require(result.get("category") == category, f"{label}: category mismatch")
    require(result.get("variant") == spec["variant"], f"{label}: variant mismatch")
    require(result.get("preset") == "P_DETERMINISTIC", f"{label}: preset mismatch")
    require(result.get("ordinal") == ordinal, f"{label}: ordinal mismatch")
    require(result.get("http_status") == 400, f"{label}: HTTP status must be 400")
    require(result.get("expected_error_param") == expected_param, f"{label}: expected param mismatch")
    require(
        result.get("expected_error_message_substring") == expected_message,
        f"{label}: expected message contract mismatch",
    )
    request_sha = json_fingerprint(request)
    response_sha = json_fingerprint(response)
    contract_sha = json_fingerprint(spec["failure_contract"])
    require(result.get("request_canonical_sha256") == request_sha, f"{label}: request SHA mismatch")
    require(result.get("response_canonical_sha256") == response_sha, f"{label}: response SHA mismatch")
    require(
        result.get("failure_contract_canonical_sha256") == contract_sha,
        f"{label}: failure contract SHA mismatch",
    )
    error = parse_openai_error(response, label)
    require(error.get("param") == expected_param, f"{label}: observed error param mismatch")
    require(expected_message in error["message"], f"{label}: observed error message mismatch")
    require(result.get("observed_error") == error, f"{label}: observed_error is not response.error")
    return {
        "category": category,
        "variant": spec["variant"],
        "request_sha": request_sha,
        "contract_sha": contract_sha,
    }


def validate_c16(
    source_root: Path,
    recorded_root: Path,
    model: str,
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    scenario_root = source_root / C16_NAME
    result_path = scenario_root / "result.json"
    persisted = read_json(result_path)
    require(scenario == persisted, f"{C16_NAME}: summary differs from result.json")
    resolve_source_member(
        source_root,
        persisted.get("artifact"),
        f"{C16_NAME}.artifact",
        recorded_root=recorded_root,
        expected=result_path,
    )
    require(persisted.get("name") == C16_NAME, "C16 name mismatch")
    require(persisted.get("type") == C16_TYPE, "C16 type mismatch")
    require(persisted.get("status") == "pass", "C16 status is not pass")
    require(persisted.get("case_count") == 30, "C16 case_count must be 30")
    require(persisted.get("passed_count") == 30, "C16 passed_count must be 30")
    require(persisted.get("cases_per_category") == 6, "C16 cases_per_category must be six")
    require(persisted.get("preset") == "P_DETERMINISTIC", "C16 preset mismatch")
    exact_counts = {category: 6 for category in C16_CATEGORIES}
    require(persisted.get("category_counts") == exact_counts, "C16 category_counts mismatch")
    require(persisted.get("unique_payload_count") == 30, "C16 unique payload count mismatch")
    require(
        persisted.get("unique_payload_count_by_category") == exact_counts,
        "C16 per-category payload uniqueness mismatch",
    )
    require(
        persisted.get("unique_failure_contract_count") == 30,
        "C16 unique failure contract count mismatch",
    )
    require(
        persisted.get("unique_failure_contract_count_by_category") == exact_counts,
        "C16 per-category failure contract uniqueness mismatch",
    )
    cases = persisted.get("cases")
    require(isinstance(cases, list) and len(cases) == 30, "C16 cases must contain 30 entries")
    require_exact_regular_files(scenario_root, {"result.json"}, C16_NAME)
    expected_dirs = {category for category in C16_CATEGORIES}
    actual_dirs = {
        entry.name for entry in scenario_root.iterdir() if entry.is_dir() and not entry.is_symlink()
    }
    require(actual_dirs == expected_dirs, f"C16 category directories mismatch: {actual_dirs}")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for case in cases:
        require(isinstance(case, dict), "C16 aggregate case is not an object")
        key = (case.get("category"), case.get("ordinal"))
        require(key not in by_key, f"C16 duplicate aggregate owner: {key}")
        by_key[key] = case
    require(
        list(by_key) == [
            (category, ordinal)
            for category in C16_CATEGORIES
            for ordinal in range(6)
        ],
        "C16 aggregate case order/owner set mismatch",
    )
    rows: list[dict[str, Any]] = []
    for category in C16_CATEGORIES:
        category_root = scenario_root / category
        actual_case_dirs = {
            entry.name
            for entry in category_root.iterdir()
            if entry.is_dir() and not entry.is_symlink()
        }
        require(actual_case_dirs == {f"{ordinal:03d}" for ordinal in range(6)}, f"C16 {category} dirs mismatch")
        require_exact_regular_files(category_root, set(), f"C16 {category}")
        for ordinal in range(6):
            key = (category, ordinal)
            require(key in by_key, f"C16 missing aggregate case {key}")
            rows.append(
                validate_c16_case(
                    source_root, recorded_root, category, ordinal, model, by_key[key]
                )
            )
    request_shas = [row["request_sha"] for row in rows]
    contract_shas = [row["contract_sha"] for row in rows]
    require(len(set(request_shas)) == 30, "C16 request payloads are not globally unique")
    require(len(set(contract_shas)) == 30, "C16 failure contracts are not globally unique")
    for category in C16_CATEGORIES:
        require(
            len({row["request_sha"] for row in rows if row["category"] == category}) == 6,
            f"C16 {category} payloads are not unique",
        )
        require(
            len({row["contract_sha"] for row in rows if row["category"] == category}) == 6,
            f"C16 {category} failure contracts are not unique",
        )
        require(
            {row["variant"] for row in rows if row["category"] == category}
            == {c16_case_spec(category, ordinal, model)[0] for ordinal in range(6)},
            f"C16 {category} variant set mismatch",
        )
    return rows


def validate_models_artifact(
    source_root: Path,
    recorded_root: Path,
    model: str,
    scenario: dict[str, Any],
) -> None:
    models_root = source_root / C20_NAME / "models"
    require_exact_regular_files(
        models_root,
        {"request.json", "response.json", "result.json"},
        "C20 models",
    )
    request = read_json(models_root / "request.json")
    response = read_json(models_root / "response.json")
    result = read_json(models_root / "result.json")
    validate_result_self_hash(result, f"{C20_NAME}/models/result.json")
    require(
        request == {"method": "GET", "path": "/v1/models", "target_model": model},
        "C20 models request mismatch",
    )
    require(response.get("object") == "list", "C20 /v1/models object must be list")
    models = response.get("data")
    require(isinstance(models, list) and models, "C20 /v1/models data must be non-empty")
    for index, entry in enumerate(models):
        require(isinstance(entry, dict), f"C20 model entry {index} is not an object")
        require(entry.get("modalities") == ["text"], f"C20 model entry {index} is not text-only")
    matches = [entry for entry in models if entry.get("id") == model]
    require(len(matches) == 1, f"C20 target model must occur exactly once, got {len(matches)}")
    require(result.get("schema_version") == 1, "C20 models result schema mismatch")
    require(result.get("status") == "pass", "C20 models result status mismatch")
    require(result.get("http_status") == 200, "C20 models HTTP status must be 200")
    require(result.get("model") == model, "C20 models result model mismatch")
    require(result.get("declared_modalities") == ["text"], "C20 models declared modalities mismatch")
    require(result.get("request_canonical_sha256") == json_fingerprint(request), "C20 models request SHA mismatch")
    require(result.get("response_canonical_sha256") == json_fingerprint(response), "C20 models response SHA mismatch")
    resolve_source_member(
        source_root,
        result.get("request_artifact"),
        "C20 models request_artifact",
        recorded_root=recorded_root,
        expected=models_root / "request.json",
    )
    resolve_source_member(
        source_root,
        result.get("response_artifact"),
        "C20 models response_artifact",
        recorded_root=recorded_root,
        expected=models_root / "response.json",
    )
    resolve_source_member(
        source_root,
        scenario.get("models_artifact"),
        "C20 models_artifact",
        recorded_root=recorded_root,
        expected=models_root / "result.json",
    )


def validate_c20_case(
    source_root: Path,
    recorded_root: Path,
    category: str,
    ordinal: int,
    model: str,
    aggregate_case: Any,
) -> dict[str, Any]:
    label = f"{C20_NAME}/{category}/{ordinal:03d}"
    case_root = source_root / C20_NAME / category / f"{ordinal:03d}"
    require(case_root.is_dir() and not case_root.is_symlink(), f"missing case directory: {case_root}")
    require_exact_regular_files(
        case_root, {"request.json", "response.json", "result.json"}, label
    )
    request = read_json(case_root / "request.json")
    response = read_json(case_root / "response.json")
    result = read_json(case_root / "result.json")
    validate_result_self_hash(result, f"{label}/result.json")
    require(aggregate_case == result, f"{label}: aggregate differs from result.json")
    validate_case_artifact_paths(source_root, recorded_root, case_root, result, label)
    marker, expected_request = expected_c20_request(category, ordinal, model)
    require(request == expected_request, f"{label}: request contract mismatch")
    require(result.get("schema_version") == 1, f"{label}: result schema mismatch")
    require(result.get("status") == "pass", f"{label}: status is not pass")
    require(result.get("case_id") == marker, f"{label}: case_id mismatch")
    require(result.get("category") == category, f"{label}: category mismatch")
    require(result.get("ordinal") == ordinal, f"{label}: ordinal mismatch")
    require(result.get("declared_modalities") == ["text"], f"{label}: declared modalities mismatch")
    require(result.get("request_canonical_sha256") == json_fingerprint(request), f"{label}: request SHA mismatch")
    require(result.get("response_canonical_sha256") == json_fingerprint(response), f"{label}: response SHA mismatch")
    if category == "text-array":
        require(result.get("preset") == "P_DETERMINISTIC", f"{label}: deterministic preset missing")
        require(result.get("http_status") == 200, f"{label}: HTTP status must be 200")
        text, finish = response_message_text(response, label)
        require(marker in text, f"{label}: response marker missing")
        require(finish != "length", f"{label}: finish_reason is length")
        require(
            result.get("observed") == {"finish_reason": finish, "content": text},
            f"{label}: observed success result mismatch",
        )
    else:
        require(result.get("preset") is None, f"{label}: media request must be unpreset")
        require(not (set(request) & DETERMINISTIC_FIELDS), f"{label}: media request leaked sampling")
        require(result.get("http_status") == 400, f"{label}: HTTP status must be 400")
        error = parse_openai_error(response, label)
        message = error["message"]
        require(error.get("param") is None, f"{label}: media error param must be null")
        require("invalid chat completions request" in message, f"{label}: not a request deserialization error")
        require(
            "unsupported message content part type" in message,
            f"{label}: error does not prove non-text deserialization rejection",
        )
        require(result.get("observed") == {"error": error}, f"{label}: observed media error mismatch")
    return {"category": category, "request_sha": json_fingerprint(request)}


def validate_c20(
    source_root: Path,
    recorded_root: Path,
    model: str,
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    scenario_root = source_root / C20_NAME
    result_path = scenario_root / "result.json"
    persisted = read_json(result_path)
    require(scenario == persisted, f"{C20_NAME}: summary differs from result.json")
    resolve_source_member(
        source_root,
        persisted.get("artifact"),
        f"{C20_NAME}.artifact",
        recorded_root=recorded_root,
        expected=result_path,
    )
    require(persisted.get("name") == C20_NAME, "C20 name mismatch")
    require(persisted.get("type") == C20_TYPE, "C20 type mismatch")
    require(persisted.get("status") == "pass", "C20 status is not pass")
    require(persisted.get("case_count") == 50, "C20 case_count must be 50")
    require(persisted.get("passed_count") == 50, "C20 passed_count must be 50")
    require(persisted.get("cases_per_category") == 10, "C20 cases_per_category must be ten")
    exact_counts = {category: 10 for category in C20_CATEGORIES}
    require(persisted.get("category_counts") == exact_counts, "C20 category_counts mismatch")
    require(persisted.get("unique_payload_count") == 50, "C20 unique payload count mismatch")
    require(
        persisted.get("unique_payload_count_by_category") == exact_counts,
        "C20 per-category payload uniqueness mismatch",
    )
    require(
        persisted.get("preset_counts") == {"P_DETERMINISTIC": 10, "unpreset": 40},
        "C20 preset partition mismatch",
    )
    require(persisted.get("rejected_media_count") == 40, "C20 rejected media count mismatch")
    require(persisted.get("text_array_success_count") == 10, "C20 text-array success count mismatch")
    require(persisted.get("declared_modalities") == ["text"], "C20 declared modalities mismatch")
    validate_models_artifact(source_root, recorded_root, model, persisted)
    cases = persisted.get("cases")
    require(isinstance(cases, list) and len(cases) == 50, "C20 cases must contain 50 entries")
    require_exact_regular_files(scenario_root, {"result.json"}, C20_NAME)
    expected_dirs = {*C20_CATEGORIES, "models"}
    actual_dirs = {
        entry.name for entry in scenario_root.iterdir() if entry.is_dir() and not entry.is_symlink()
    }
    require(actual_dirs == expected_dirs, f"C20 category directories mismatch: {actual_dirs}")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for case in cases:
        require(isinstance(case, dict), "C20 aggregate case is not an object")
        key = (case.get("category"), case.get("ordinal"))
        require(key not in by_key, f"C20 duplicate aggregate owner: {key}")
        by_key[key] = case
    require(
        list(by_key) == [
            (category, ordinal)
            for category in C20_CATEGORIES
            for ordinal in range(10)
        ],
        "C20 aggregate case order/owner set mismatch",
    )
    rows: list[dict[str, Any]] = []
    for category in C20_CATEGORIES:
        category_root = scenario_root / category
        actual_case_dirs = {
            entry.name
            for entry in category_root.iterdir()
            if entry.is_dir() and not entry.is_symlink()
        }
        require(actual_case_dirs == {f"{ordinal:03d}" for ordinal in range(10)}, f"C20 {category} dirs mismatch")
        require_exact_regular_files(category_root, set(), f"C20 {category}")
        for ordinal in range(10):
            key = (category, ordinal)
            require(key in by_key, f"C20 missing aggregate case {key}")
            rows.append(
                validate_c20_case(
                    source_root, recorded_root, category, ordinal, model, by_key[key]
                )
            )
    request_shas = [row["request_sha"] for row in rows]
    require(len(set(request_shas)) == 50, "C20 request payloads are not globally unique")
    for category in C20_CATEGORIES:
        require(
            len({row["request_sha"] for row in rows if row["category"] == category}) == 10,
            f"C20 {category} payloads are not unique",
        )
    return rows


def validate_observability_paths(
    source_root: Path, recorded_root: Path, observability: Any
) -> None:
    if observability is None:
        return
    require(isinstance(observability, dict), "summary.observability must be an object or null")
    require(observability.get("enabled") is True, "observability object must be enabled")
    for key in ("profile_paths", "scheduler_trace_paths", "request_dump_dirs"):
        values = observability.get(key, [])
        require(isinstance(values, list), f"observability.{key} must be a list")
        for index, value in enumerate(values):
            resolve_source_member(
                source_root,
                value,
                f"observability.{key}[{index}]",
                recorded_root=recorded_root,
            )
    roots = observability.get("roots")
    require(isinstance(roots, dict), "observability.roots must be an object")
    for key, value in roots.items():
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        for index, item in enumerate(values):
            resolve_source_member(
                source_root,
                item,
                f"observability.roots.{key}[{index}]",
                recorded_root=recorded_root,
            )


def validate_source(source: Path) -> dict[str, Any]:
    require(source.exists() and source.is_dir(), f"source root is not a directory: {source}")
    require(not source.is_symlink(), f"source root must not be a symlink: {source}")
    source_root = source.resolve(strict=True)
    summary_path = source_root / "summary.json"
    summary = read_json(summary_path)
    require(summary.get("schema_version") == 1, "summary schema_version mismatch")
    require(summary.get("status") == "pass", "summary status is not pass")
    require(summary.get("backend") == "cuda", "summary backend must be cuda")
    model = summary.get("model")
    require(isinstance(model, str) and is_qwen35_4b_model(model), "summary model must be Qwen3.5-4B")
    git_sha = summary.get("git_sha")
    require(isinstance(git_sha, str) and GIT_SHA_RE.fullmatch(git_sha), "summary git_sha invalid")
    dirty = summary.get("dirty_status")
    require(
        dirty == {"is_dirty": False, "status_short": []},
        "summary source checkout must be exactly clean",
    )
    require(summary.get("failed") == 0, "summary failed must be zero")
    require(summary.get("skipped") == 0, "summary skipped must be zero")
    require(summary.get("scenario_count") == 2, "summary scenario_count must be two")
    require(summary.get("manifest_scenario_count") == 2, "manifest scenario count must be two")
    require(summary.get("requested_scenarios") == [], "checkpoint must run the full two-scenario manifest")
    require(summary.get("selected_scenarios") == list(SCENARIO_NAMES), "selected scenario set/order mismatch")
    artifact_dir = summary.get("artifact_dir")
    require(isinstance(artifact_dir, str) and artifact_dir, "summary artifact_dir missing")
    recorded_root = Path(artifact_dir)
    require(recorded_root.is_absolute() and ".." not in recorded_root.parts, "recorded artifact root invalid")
    require(
        summary.get("pass_line") == f"BACKEND REGRESSION SMOKE PASS: {artifact_dir}",
        "generic runner PASS line mismatch",
    )
    scenarios = summary.get("scenarios")
    require(isinstance(scenarios, list) and len(scenarios) == 2, "summary scenarios must contain two entries")
    require(
        [item.get("name") if isinstance(item, dict) else None for item in scenarios]
        == list(SCENARIO_NAMES),
        "summary scenario identity/order mismatch",
    )
    scenario_by_name = {item["name"]: item for item in scenarios if isinstance(item, dict)}
    scenario_result_dirs = {
        path.parent.name
        for path in source_root.glob("*/result.json")
        if path.is_file() and not path.is_symlink()
    }
    require(
        scenario_result_dirs == set(SCENARIO_NAMES),
        f"artifact contains unexpected top-level scenario results: {sorted(scenario_result_dirs)}",
    )
    c16_rows = validate_c16(source_root, recorded_root, model, scenario_by_name[C16_NAME])
    c20_rows = validate_c20(source_root, recorded_root, model, scenario_by_name[C20_NAME])
    validate_observability_paths(source_root, recorded_root, summary.get("observability"))
    receipt = validate_execution_receipt(source_root, recorded_root, summary, model)
    tree = validate_artifact_tree(source_root, recorded_root)
    server_log = source_root / "server.log"
    assert_clean_text("server.log", read_text(server_log))
    return {
        "git_sha": git_sha,
        "backend": "cuda",
        "model": model,
        "scenario_counts": {C16_NAME: 30, C20_NAME: 50},
        "case_count": len(c16_rows) + len(c20_rows),
        "c16": {
            "case_count": len(c16_rows),
            "category_counts": dict(Counter(row["category"] for row in c16_rows)),
            "unique_payload_count": len({row["request_sha"] for row in c16_rows}),
            "unique_failure_contract_count": len({row["contract_sha"] for row in c16_rows}),
        },
        "c20": {
            "case_count": len(c20_rows),
            "category_counts": dict(Counter(row["category"] for row in c20_rows)),
            "unique_payload_count": len({row["request_sha"] for row in c20_rows}),
            "rejected_media_count": 40,
            "text_array_success_count": 10,
        },
        "execution_receipt": receipt,
        "artifact_tree": tree,
        "source_sha256": {"summary.json": file_sha256(summary_path)},
    }


def base_manifest(source: Path, out: Path, started_at: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_id": CHECKPOINT_ID,
        "scope": ["S2/C16", "S2/C20"],
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


def fixture_c16(source_root: Path, model: str) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for category in C16_CATEGORIES:
        for ordinal in range(6):
            marker, request, expected_param, expected_message, spec = expected_c16_request(
                category, ordinal, model
            )
            response = {
                "error": {
                    "message": f"fixture: {expected_message}",
                    "type": "invalid_request_error",
                    "param": expected_param,
                    "code": None,
                }
            }
            case_root = source_root / C16_NAME / category / f"{ordinal:03d}"
            write_json(case_root / "request.json", request)
            write_json(case_root / "response.json", response)
            result = with_canonical_sha256(
                {
                    "schema_version": 1,
                    "status": "pass",
                    "case_id": marker,
                    "category": category,
                    "variant": spec["variant"],
                    "preset": "P_DETERMINISTIC",
                    "ordinal": ordinal,
                    "http_status": 400,
                    "expected_error_param": expected_param,
                    "expected_error_message_substring": expected_message,
                    "observed_error": response["error"],
                    "request_canonical_sha256": json_fingerprint(request),
                    "failure_contract_canonical_sha256": json_fingerprint(
                        spec["failure_contract"]
                    ),
                    "response_canonical_sha256": json_fingerprint(response),
                    "request_artifact": str(case_root / "request.json"),
                    "response_artifact": str(case_root / "response.json"),
                }
            )
            write_json(case_root / "result.json", result)
            cases.append(result)
    exact_counts = {category: 6 for category in C16_CATEGORIES}
    scenario = {
        "status": "pass",
        "case_count": 30,
        "passed_count": 30,
        "cases_per_category": 6,
        "preset": "P_DETERMINISTIC",
        "category_counts": exact_counts,
        "unique_payload_count": 30,
        "unique_payload_count_by_category": exact_counts,
        "unique_failure_contract_count": 30,
        "unique_failure_contract_count_by_category": exact_counts,
        "cases": cases,
        "name": C16_NAME,
        "type": C16_TYPE,
        "artifact": str(source_root / C16_NAME / "result.json"),
        "duration_sec": 1.0,
    }
    write_json(source_root / C16_NAME / "result.json", scenario)
    return scenario


def fixture_chat_response(model: str, marker: str) -> dict[str, Any]:
    return {
        "id": f"fixture-{marker}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": marker},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def fixture_c20(source_root: Path, model: str) -> dict[str, Any]:
    models_root = source_root / C20_NAME / "models"
    models_request = {"method": "GET", "path": "/v1/models", "target_model": model}
    models_response = {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 1,
                "owned_by": "ferrum",
                "modalities": ["text"],
            }
        ],
    }
    write_json(models_root / "request.json", models_request)
    write_json(models_root / "response.json", models_response)
    models_result = with_canonical_sha256(
        {
            "schema_version": 1,
            "status": "pass",
            "http_status": 200,
            "model": model,
            "declared_modalities": ["text"],
            "request_canonical_sha256": json_fingerprint(models_request),
            "response_canonical_sha256": json_fingerprint(models_response),
            "request_artifact": str(models_root / "request.json"),
            "response_artifact": str(models_root / "response.json"),
        }
    )
    write_json(models_root / "result.json", models_result)
    cases: list[dict[str, Any]] = []
    for category in C20_CATEGORIES:
        for ordinal in range(10):
            marker, request = expected_c20_request(category, ordinal, model)
            if category == "text-array":
                response = fixture_chat_response(model, marker)
                observed = {"finish_reason": "stop", "content": marker}
                preset: str | None = "P_DETERMINISTIC"
                http_status = 200
            else:
                response = {
                    "error": {
                        "message": "invalid chat completions request: "
                        "unsupported message content part type `fixture-media`",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None,
                    }
                }
                observed = {"error": response["error"]}
                preset = None
                http_status = 400
            case_root = source_root / C20_NAME / category / f"{ordinal:03d}"
            write_json(case_root / "request.json", request)
            write_json(case_root / "response.json", response)
            result = with_canonical_sha256(
                {
                    "schema_version": 1,
                    "status": "pass",
                    "case_id": marker,
                    "category": category,
                    "preset": preset,
                    "ordinal": ordinal,
                    "http_status": http_status,
                    "declared_modalities": ["text"],
                    "observed": observed,
                    "request_canonical_sha256": json_fingerprint(request),
                    "response_canonical_sha256": json_fingerprint(response),
                    "request_artifact": str(case_root / "request.json"),
                    "response_artifact": str(case_root / "response.json"),
                }
            )
            write_json(case_root / "result.json", result)
            cases.append(result)
    exact_counts = {category: 10 for category in C20_CATEGORIES}
    scenario = {
        "status": "pass",
        "case_count": 50,
        "passed_count": 50,
        "cases_per_category": 10,
        "preset_counts": {"P_DETERMINISTIC": 10, "unpreset": 40},
        "category_counts": exact_counts,
        "unique_payload_count": 50,
        "unique_payload_count_by_category": exact_counts,
        "rejected_media_count": 40,
        "text_array_success_count": 10,
        "declared_modalities": ["text"],
        "models_artifact": str(models_root / "result.json"),
        "cases": cases,
        "name": C20_NAME,
        "type": C20_TYPE,
        "artifact": str(source_root / C20_NAME / "result.json"),
        "duration_sec": 1.0,
    }
    write_json(source_root / C20_NAME / "result.json", scenario)
    return scenario


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
        with_canonical_sha256({
            "schema_version": 1,
            "artifact_root": str(source_root),
            "file_count": len(entries),
            "files": entries,
        }),
    )


def create_fixture(source_root: Path) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    model = "Qwen/Qwen3.5-4B"
    c16 = fixture_c16(source_root, model)
    c20 = fixture_c20(source_root, model)
    write_json(
        source_root / "response_format_matrix_contract.json",
        {
            "schema_version": 1,
            "status": "pass",
            "matrix_scenario_count": 0,
            "case_counts": {"json_schema": 0, "json_object": 0},
            "unique_expected_object_counts": {"json_schema": 0, "json_object": 0},
            "unique_json_schema_count": 0,
            "json_schema_category_counts": {},
            "cases": [],
        },
    )
    evidence_payloads = {
        "server.effective_config.json": json.dumps(
            {
                "schema_version": 1,
                "backend": "cuda",
                "cuda_device_count": 1,
                "selected_gpu_devices": [0],
                "model_capabilities": {"architecture": "qwen3_5"},
                "hardware_capabilities": {
                    "backend": "cuda",
                    "compiled_features": {"cuda": True},
                },
            }
        )
        + "\n",
        "server.decision_trace.jsonl": '{"event":"server_started"}\n',
        "server.log": "fixture server started cleanly\n",
        "server.health.json": '{"status":"pass","http_status":200}\n',
        "server.health.after.json": '{"status":"pass","http_status":200}\n',
    }
    for relative, payload in evidence_payloads.items():
        path = source_root / relative
        path.write_text(payload, encoding="utf-8")
    receipt_evidence = {
        label: {
            "path": str(source_root / relative),
            "size": (source_root / relative).stat().st_size,
            "sha256": file_sha256(source_root / relative),
        }
        for label, relative in EXECUTION_EVIDENCE.items()
    }
    manifest_path = "/workspace/ferrum/scripts/release/scenarios/runtime_vnext_s2_c16_c20_cuda.json"
    runner_path = "/workspace/ferrum/scripts/release/run_scenarios.py"
    binary_path = "/workspace/ferrum/target/release/ferrum"
    input_root = source_root / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    runner_input = input_root / "run_scenarios.py"
    manifest_input = input_root / "scenario_manifest.json"
    runner_input.write_text("# frozen fixture runner\n", encoding="utf-8")
    write_json(
        manifest_input,
        {
            "schema_version": 1,
            "backend": "cuda",
            "model": model,
            "scenarios": [{"name": name} for name in SCENARIO_NAMES],
        },
    )
    runner_sha = file_sha256(runner_input)
    manifest_sha = file_sha256(manifest_input)
    receipt = with_canonical_sha256({
        "schema_version": 1,
        "mode": "start",
        "runner_argv": [
            "/usr/bin/python3",
            runner_path,
            "--manifest",
            manifest_path,
            "--out",
            str(source_root),
        ],
        "runner_path": runner_path,
        "runner_sha256": runner_sha,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "cwd": "/workspace/ferrum",
        "git_sha": "a" * 40,
        "dirty_status": {"is_dirty": False, "status_short": []},
        "input_artifacts": {
            "runner": {"path": str(runner_input), "sha256": runner_sha},
            "manifest": {"path": str(manifest_input), "sha256": manifest_sha},
        },
        "backend": "cuda",
        "model": model,
        "selected_scenarios": list(SCENARIO_NAMES),
        "scenario_count": 2,
        "failed": 0,
        "skipped": 0,
        "server_argv": [
            binary_path,
            "serve",
            "--host",
            "127.0.0.1",
            "--effective-config-json",
            str(source_root / "server.effective_config.json"),
            "--decision-trace-jsonl",
            str(source_root / "server.decision_trace.jsonl"),
            "--backend",
            "cuda",
            model,
        ],
        "binary_path": binary_path,
        "binary_sha256": "3" * 64,
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
        "child_env": {"HF_HOME": "/workspace/hf-cache"},
        "server_started_at": "2026-07-20T00:00:00+00:00",
        "server_finished_at": "2026-07-20T00:01:00+00:00",
        "server_returncode": -15,
        "evidence_files": receipt_evidence,
    })
    receipt_path = source_root / "execution_receipt.json"
    write_json(receipt_path, receipt)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "manifest": manifest_path,
        "artifact_dir": str(source_root),
        "model": model,
        "backend": "cuda",
        "base_url": "http://127.0.0.1:8000/v1/chat/completions",
        "git_sha": "a" * 40,
        "dirty_status": {"is_dirty": False, "status_short": []},
        "started_at": "2026-07-20T00:00:00+00:00",
        "finished_at": "2026-07-20T00:01:00+00:00",
        "scenario_count": 2,
        "manifest_scenario_count": 2,
        "requested_scenarios": [],
        "selected_scenarios": list(SCENARIO_NAMES),
        "failed": 0,
        "skipped": 0,
        "scenarios": [copy.deepcopy(c16), copy.deepcopy(c20)],
        "response_format_matrix_contract": {
            "artifact": str(source_root / "response_format_matrix_contract.json"),
            "case_counts": {"json_schema": 0, "json_object": 0},
            "unique_json_schema_count": 0,
        },
        "observability": None,
        "execution_receipt": {
            "artifact": str(receipt_path),
            "artifact_sha256": file_sha256(receipt_path),
            "canonical_sha256": receipt["canonical_sha256"],
            "mode": "start",
            "runner_sha256": runner_sha,
            "manifest_sha256": manifest_sha,
            "binary_sha256": "3" * 64,
        },
        "pass_line": f"BACKEND REGRESSION SMOKE PASS: {source_root}",
    }
    write_json(source_root / "summary.json", summary)
    write_fixture_artifact_tree(source_root)


def rewrite_case_result(
    source_root: Path,
    scenario_name: str,
    category: str,
    ordinal: int,
    update: Callable[[dict[str, Any]], None],
) -> None:
    case_path = source_root / scenario_name / category / f"{ordinal:03d}" / "result.json"
    result = read_json(case_path)
    update(result)
    result.pop("canonical_sha256", None)
    result["canonical_sha256_scope"] = "document_without_canonical_sha256_fields"
    result["canonical_sha256"] = json_fingerprint(result)
    write_json(case_path, result)
    scenario_path = source_root / scenario_name / "result.json"
    scenario = read_json(scenario_path)
    for index, case in enumerate(scenario["cases"]):
        if case.get("category") == category and case.get("ordinal") == ordinal:
            scenario["cases"][index] = copy.deepcopy(result)
            break
    else:
        raise ValidationError(f"fixture case missing from scenario aggregate: {scenario_name}/{category}/{ordinal}")
    write_json(scenario_path, scenario)
    summary_path = source_root / "summary.json"
    summary = read_json(summary_path)
    for index, candidate in enumerate(summary["scenarios"]):
        if candidate.get("name") == scenario_name:
            summary["scenarios"][index] = copy.deepcopy(scenario)
            break
    write_json(summary_path, summary)


def refresh_case_hashes(
    source_root: Path, scenario_name: str, category: str, ordinal: int
) -> None:
    case_root = source_root / scenario_name / category / f"{ordinal:03d}"

    def update(result: dict[str, Any]) -> None:
        request = read_json(case_root / "request.json")
        response = read_json(case_root / "response.json")
        result["request_canonical_sha256"] = json_fingerprint(request)
        result["response_canonical_sha256"] = json_fingerprint(response)
        if scenario_name == C16_NAME:
            result["observed_error"] = response["error"]
        elif category == "text-array":
            text, finish = response_message_text(response, "fixture mutation")
            result["observed"] = {"finish_reason": finish, "content": text}
        else:
            result["observed"] = {"error": response["error"]}

    rewrite_case_result(source_root, scenario_name, category, ordinal, update)


def mutate_missing_case(source_root: Path) -> None:
    shutil.rmtree(source_root / C16_NAME / "invalid-tool" / "005")
    write_fixture_artifact_tree(source_root)


def mutate_duplicate_payload_and_contract(source_root: Path) -> None:
    first_root = source_root / C16_NAME / "invalid-tool" / "000"
    second_root = source_root / C16_NAME / "invalid-tool" / "001"
    shutil.copy2(first_root / "request.json", second_root / "request.json")
    first_result = read_json(first_root / "result.json")

    def update(result: dict[str, Any]) -> None:
        result["request_canonical_sha256"] = first_result["request_canonical_sha256"]
        result["failure_contract_canonical_sha256"] = first_result[
            "failure_contract_canonical_sha256"
        ]

    rewrite_case_result(source_root, C16_NAME, "invalid-tool", 1, update)
    write_fixture_artifact_tree(source_root)


def mutate_wrong_preset(source_root: Path) -> None:
    rewrite_case_result(
        source_root,
        C16_NAME,
        "invalid-schema",
        0,
        lambda result: result.update({"preset": "P_NO_THINKING"}),
    )
    write_fixture_artifact_tree(source_root)


def mutate_media_sampling_leak(source_root: Path) -> None:
    request_path = source_root / C20_NAME / "image-url" / "000" / "request.json"
    request = read_json(request_path)
    request["temperature"] = 0.0
    write_json(request_path, request)
    refresh_case_hashes(source_root, C20_NAME, "image-url", 0)
    write_fixture_artifact_tree(source_root)


def mutate_text_sampling_missing(source_root: Path) -> None:
    request_path = source_root / C20_NAME / "text-array" / "000" / "request.json"
    request = read_json(request_path)
    request.pop("seed")
    write_json(request_path, request)
    refresh_case_hashes(source_root, C20_NAME, "text-array", 0)
    write_fixture_artifact_tree(source_root)


def mutate_text_marker_missing(source_root: Path) -> None:
    response_path = source_root / C20_NAME / "text-array" / "000" / "response.json"
    response = read_json(response_path)
    response["choices"][0]["message"]["content"] = "wrong-marker"
    write_json(response_path, response)
    refresh_case_hashes(source_root, C20_NAME, "text-array", 0)
    write_fixture_artifact_tree(source_root)


def mutate_bad_modality(source_root: Path) -> None:
    models_root = source_root / C20_NAME / "models"
    response_path = models_root / "response.json"
    response = read_json(response_path)
    response["data"][0]["modalities"] = ["text", "image"]
    write_json(response_path, response)
    result_path = models_root / "result.json"
    result = read_json(result_path)
    result["response_canonical_sha256"] = json_fingerprint(response)
    result["declared_modalities"] = ["text", "image"]
    result.pop("canonical_sha256", None)
    result["canonical_sha256"] = json_fingerprint(result)
    write_json(result_path, result)
    write_fixture_artifact_tree(source_root)


def mutate_path_escape(source_root: Path) -> None:
    rewrite_case_result(
        source_root,
        C16_NAME,
        "invalid-model",
        0,
        lambda result: result.update({"request_artifact": "../escape.json"}),
    )
    write_fixture_artifact_tree(source_root)


def mutate_bad_text(source_root: Path) -> None:
    response_path = source_root / C16_NAME / "invalid-context" / "000" / "response.json"
    response = read_json(response_path)
    response["error"]["message"] += " \ufffd"
    write_json(response_path, response)

    def update(result: dict[str, Any]) -> None:
        result["response_canonical_sha256"] = json_fingerprint(response)
        result["observed_error"] = response["error"]

    rewrite_case_result(source_root, C16_NAME, "invalid-context", 0, update)
    write_fixture_artifact_tree(source_root)


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
        ("missing-case", mutate_missing_case),
        ("duplicate-payload-contract", mutate_duplicate_payload_and_contract),
        ("wrong-preset", mutate_wrong_preset),
        ("media-sampling-leak", mutate_media_sampling_leak),
        ("text-sampling-missing", mutate_text_sampling_missing),
        ("text-marker-missing", mutate_text_marker_missing),
        ("bad-modality", mutate_bad_modality),
        ("path-escape", mutate_path_escape),
        ("bad-text", mutate_bad_text),
    ]
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-s2-api-modality-") as temporary:
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
        require(
            baseline_manifest.get("evidence", {}).get("case_count") == 80,
            "self-test baseline did not validate 80 cases",
        )
        relocated = root / "relocated"
        shutil.copytree(baseline, relocated)
        relocated_out = root / "relocated-out"
        relocated_proc = run_selftest_process(relocated, relocated_out)
        require(
            relocated_proc.returncode == 0,
            "remote-root relocation fixture failed: "
            + (relocated_proc.stderr or relocated_proc.stdout),
        )
        require(
            f"{PASS_PREFIX}: {relocated_out}" in relocated_proc.stdout.splitlines(),
            "relocation fixture missing exact PASS line",
        )
        for name, mutate in mutations:
            source = root / name
            create_fixture(source)
            mutate(source)
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
