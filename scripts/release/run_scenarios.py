#!/usr/bin/env python3
"""Manifest-driven product regression scenario runner.

The runner is intentionally small and explicit: a scenario manifest names the
product path (`ferrum run`, external `ferrum serve`, or a server started by this
script), each scenario writes its own JSON artifact, and the runner writes a
summary artifact plus one PASS line.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import http.server
import json
import os
import pty
import re
import select
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from openai_concurrency_quality_regression import run_concurrency_quality_regression
from openai_tool_call_regression import run_tool_call_regression


BAD_TEXT = [
    "<unk>",
    "[PAD]",
    "<|assistant|>",
    "<|tool|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_",
    "classname=",
    "invalid utf-8",
    "mojibake",
]
SERVE_TYPES = {
    "serve_chat",
    "serve_multiturn_recall",
    "serve_stateful_loop",
    "serve_stream",
    "serve_stream_equivalence_unicode",
    "serve_disconnect_release",
    "serve_structured_output",
    "serve_response_format_matrix",
    "serve_negative_api_matrix",
    "serve_text_only_modality_matrix",
    "serve_context_limit",
    "serve_concurrency_quality",
    "serve_tool_schema_priority",
    "serve_tool_call",
    "serve_python_openai_sdk",
}
RUN_TYPES = {
    "run_multiturn",
    "run_first_token_ux",
}
C20_REMOTE_MEDIA_URL = (
    "https://raw.githubusercontent.com/sizzlecar/ferrum-infer-rs/"
    "cff4c47765ef3259b8a04890187d99c60da86394/"
    "docs/bench/framework-validation-2026-05-25/m3_layerwise.png"
)


class ScenarioError(Exception):
    pass


class StartedServer:
    def __init__(self, cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.cmd = list(cmd)
        self.log_path = log_path
        self.started_at = iso_now()
        self.finished_at: str | None = None
        self.returncode: int | None = None
        self.file = log_path.open("wb")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.file,
            stderr=subprocess.STDOUT,
            env=env,
        )

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.file.close()
        self.finished_at = iso_now()
        self.returncode = self.proc.returncode
        assert_no_bad_text(self.log_path.name, self.log_path.read_text(encoding="utf-8"))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def git_output(args: list[str], default: str = "unknown") -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return default
    if proc.returncode != 0:
        return default
    return proc.stdout.strip() or default


def git_dirty_status() -> dict[str, Any]:
    text = git_output(["status", "--short"], default="")
    lines = [line for line in text.splitlines() if line.strip()]
    return {"is_dirty": bool(lines), "status_short": lines}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_receipt(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        return {"argv": cmd, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "argv": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def write_artifact_tree(root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name == "artifact_tree.json":
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest = with_canonical_sha256({
        "schema_version": 1,
        "artifact_root": str(root),
        "file_count": len(entries),
        "files": entries,
    })
    write_json(root / "artifact_tree.json", manifest)
    return manifest


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ScenarioError(f"{path} must contain a JSON object")
    return data


def scenario_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-")
    return slug or "scenario"


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)


def assert_no_bad_text(label: str, text: str, extra: list[str] | None = None) -> None:
    if "\ufffd" in text:
        raise ScenarioError(f"{label}: Unicode replacement character")
    if "\x00" in text:
        raise ScenarioError(f"{label}: NUL byte")
    lowered = text.lower()
    for token in [*BAD_TEXT, *(extra or [])]:
        if token.lower() in lowered:
            raise ScenarioError(f"{label}: forbidden text {token!r}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioError(message)


def json_fingerprint(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def post_json(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    headers: dict[str, str] | None = None,
) -> tuple[int, str]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def get_url(url: str, *, timeout: int) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def request_sse(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout: int,
) -> tuple[int, str, list[float]]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8"), []
    chunks: list[str] = []
    event_times: list[float] = []
    with response:
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8")
            chunks.append(line)
            if line.startswith("data: "):
                event_times.append(time.time())
    return response.status, "".join(chunks), event_times


def request_sse_incremental(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    read_size: int = 1,
) -> tuple[int, bytes, str, dict[str, Any]]:
    require(read_size > 0, "incremental SSE read_size must be positive")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ScenarioError(f"SSE error response is not valid UTF-8: {error}") from error
        return exc.code, raw, decoded, {
            "read_size": read_size,
            "read_count": 1 if raw else 0,
            "wire_size_bytes": len(raw),
            "wire_sha256": hashlib.sha256(raw).hexdigest(),
            "decoded_sha256": hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
            "split_boundary_count": 0,
        }

    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    raw_chunks: list[bytes] = []
    decoded_fragments: list[str] = []
    split_boundary_count = 0
    with response:
        status = response.status
        while True:
            raw = response.read(read_size)
            if not raw:
                break
            raw_chunks.append(raw)
            try:
                fragment = decoder.decode(raw, final=False)
            except UnicodeDecodeError as error:
                raise ScenarioError(f"SSE wire is not valid incremental UTF-8: {error}") from error
            buffered, _ = decoder.getstate()
            if buffered:
                split_boundary_count += 1
            if fragment:
                decoded_fragments.append(fragment)
        try:
            tail = decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            raise ScenarioError(f"SSE wire ended inside a UTF-8 sequence: {error}") from error
        if tail:
            decoded_fragments.append(tail)

    raw_body = b"".join(raw_chunks)
    decoded_body = "".join(decoded_fragments)
    require(
        decoded_body.encode("utf-8") == raw_body,
        "incremental SSE decoder did not reproduce the exact wire bytes",
    )
    return status, raw_body, decoded_body, {
        "read_size": read_size,
        "read_count": len(raw_chunks),
        "wire_size_bytes": len(raw_body),
        "wire_sha256": hashlib.sha256(raw_body).hexdigest(),
        "decoded_sha256": hashlib.sha256(decoded_body.encode("utf-8")).hexdigest(),
        "split_boundary_count": split_boundary_count,
    }


def request_sse_until_output_then_disconnect(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    on_first_output: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        return {
            "http_status": exc.code,
            "partial_wire": body.encode("utf-8"),
            "first_output_event": None,
            "active_health": None,
            "time_to_first_output_sec": None,
        }

    started = time.monotonic()
    raw_lines: list[bytes] = []
    first_output_event: dict[str, Any] | None = None
    active_health: dict[str, Any] | None = None
    with response:
        status = response.status
        while True:
            raw = response.readline()
            if not raw:
                break
            raw_lines.append(raw)
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ScenarioError(f"disconnect probe received invalid UTF-8: {error}") from error
            if not line.startswith("data: "):
                continue
            event_text = line.removeprefix("data: ").strip()
            if not event_text or event_text == "[DONE]":
                continue
            try:
                event = json.loads(event_text)
            except json.JSONDecodeError as error:
                raise ScenarioError(
                    f"disconnect probe received malformed SSE JSON: {event_text[:500]}"
                ) from error
            for choice in event.get("choices", []):
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                output = (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                )
                if output:
                    first_output_event = event
                    active_health = on_first_output()
                    break
            if first_output_event is not None:
                break

    return {
        "http_status": status,
        "partial_wire": b"".join(raw_lines),
        "first_output_event": first_output_event,
        "active_health": active_health,
        "time_to_first_output_sec": time.monotonic() - started,
    }


def parse_json_response(label: str, status: int, body: str, expected_status: int = 200) -> dict[str, Any]:
    if status != expected_status:
        raise ScenarioError(f"{label}: expected HTTP {expected_status}, got {status}: {body[:500]}")
    assert_no_bad_text(label, body)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"{label}: invalid JSON response: {exc}: {body[:500]}") from exc
    if not isinstance(data, dict):
        raise ScenarioError(f"{label}: response must be JSON object")
    return data


def parse_openai_error(label: str, status: int, body: str) -> dict[str, Any]:
    require(400 <= status < 500, f"{label}: expected HTTP 4xx, got {status}: {body[:500]}")
    assert_no_bad_text(label, body)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"{label}: invalid JSON error response: {exc}: {body[:500]}") from exc
    require(isinstance(data, dict), f"{label}: error response must be a JSON object")
    error = data.get("error")
    require(isinstance(error, dict), f"{label}: missing OpenAI error object")
    require(error.get("type") == "invalid_request_error", f"{label}: bad error type: {error}")
    require(isinstance(error.get("message"), str) and error["message"], f"{label}: empty error message")
    require("param" in error, f"{label}: OpenAI error is missing param: {error}")
    require(
        error["param"] is None or isinstance(error["param"], str),
        f"{label}: OpenAI error param must be a string or null: {error}",
    )
    return error


def with_canonical_sha256(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["canonical_sha256_scope"] = "document_without_canonical_sha256_fields"
    result["canonical_sha256"] = json_fingerprint(result)
    return result


def register_unique_payload(
    payload: dict[str, Any],
    *,
    owner: str,
    category: str,
    global_owners: dict[str, str],
    category_owners: dict[str, dict[str, str]],
) -> str:
    fingerprint = json_fingerprint(payload)
    previous = global_owners.get(fingerprint)
    require(previous is None, f"duplicate matrix payload: {previous} and {owner}")
    per_category = category_owners.setdefault(category, {})
    previous_category = per_category.get(fingerprint)
    require(
        previous_category is None,
        f"duplicate {category} payload: {previous_category} and {owner}",
    )
    global_owners[fingerprint] = owner
    per_category[fingerprint] = owner
    return fingerprint


def parse_text_only_model_declaration(
    label: str, status: int, body: str, model: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = parse_json_response(label, status, body)
    require(data.get("object") == "list", f"{label}: /v1/models object must be list")
    models = data.get("data")
    require(isinstance(models, list), f"{label}: /v1/models data must be an array")
    require(models, f"{label}: /v1/models must contain at least one model")
    for candidate in models:
        require(isinstance(candidate, dict), f"{label}: every model entry must be an object")
        require(
            candidate.get("modalities") == ["text"],
            f"{label}: every served model must declare modalities exactly ['text']: {candidate}",
        )
    matches = [entry for entry in models if isinstance(entry, dict) and entry.get("id") == model]
    require(len(matches) == 1, f"{label}: expected one /v1/models entry for {model!r}, got {len(matches)}")
    entry = matches[0]
    require(
        entry.get("modalities") == ["text"],
        f"{label}: model {model!r} must declare modalities exactly ['text']: {entry}",
    )
    return data, entry


def function_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object"},
        },
    }


def deterministic_sampling_plan(scenario: dict[str, Any]) -> dict[str, Any]:
    require(
        scenario.get("preset") == "P_DETERMINISTIC",
        "scenario preset must be P_DETERMINISTIC",
    )
    require(
        scenario.get("enable_thinking") is False,
        "P_DETERMINISTIC requires enable_thinking=false",
    )
    sampling = scenario.get("sampling")
    require(isinstance(sampling, dict), "P_DETERMINISTIC sampling is required")
    expected = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": 9271,
        "stop": [],
    }
    require(sampling == expected, f"P_DETERMINISTIC sampling mismatch: {sampling!r}")
    require(
        int(scenario.get("max_tokens", 0)) == 1024,
        "P_DETERMINISTIC max_tokens must be 1024",
    )
    return sampling


def apply_deterministic_sampling(
    payload: dict[str, Any], scenario: dict[str, Any]
) -> None:
    sampling = deterministic_sampling_plan(scenario)
    payload["max_tokens"] = int(scenario["max_tokens"])
    payload.update(sampling)
    payload["chat_template_kwargs"] = {"enable_thinking": False}


def negative_api_case(
    category: str,
    ordinal: int,
    payload: dict[str, Any],
    valid_model: str,
) -> tuple[str, str | None, str, dict[str, Any]]:
    require(0 <= ordinal < 6, f"unsupported C16 ordinal: {ordinal}")

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
        case_variant = valid_model.swapcase()
        if case_variant == valid_model:
            case_variant = valid_model + "-CASE-MISMATCH"
        variants = [
            ("unknown-name", {"model": "not-a-loaded-model"}, "model", "unknown model"),
            ("empty-name", {"model": ""}, "model", "unknown model"),
            ("whitespace-name", {"model": " "}, "model", "unknown model"),
            (
                "leading-whitespace",
                {"model": " " + valid_model},
                "model",
                "unknown model",
            ),
            ("case-mismatch", {"model": case_variant}, "model", "unknown model"),
            (
                "missing-adapter",
                {"model": valid_model + ":missing-adapter"},
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
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
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
        raise ScenarioError(f"unsupported C16 category: {category}")

    variant, patch, expected_param, message_substring = variants[ordinal]
    for key, value in patch.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    failure_contract = {"category": category, "request_patch": patch}
    return variant, expected_param, message_substring, failure_contract


def first_choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ScenarioError(f"missing first choice: {data}")
    return choices[0]


def message_text(data: dict[str, Any]) -> str:
    msg = first_choice(data).get("message")
    if not isinstance(msg, dict):
        raise ScenarioError(f"missing message: {data}")
    text = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
    return strip_think(str(text))


def finish_reason(data: dict[str, Any]) -> str | None:
    value = first_choice(data).get("finish_reason")
    return str(value) if value is not None else None


def response_usage(label: str, data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usage")
    require(isinstance(usage, dict), f"{label}: missing usage object")
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{label}: usage.{key} must be a non-negative integer: {usage}",
        )
        result[key] = value
    require(result["completion_tokens"] > 0, f"{label}: completion_tokens must be positive")
    require(
        result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"],
        f"{label}: usage total is inconsistent: {result}",
    )
    return result


def chat_payload(
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


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
        raise ScenarioError(f"unsupported strict schema category: {category}")
    return expected, schema


def response_format_matrix_plan(scenario: dict[str, Any]) -> dict[str, Any]:
    name = str(scenario.get("name") or "")
    require(name != "", "serve_response_format_matrix.name is required")
    format_type = str(scenario.get("format") or "")
    require(
        format_type in {"json_object", "json_schema"},
        "serve_response_format_matrix.format must be json_object or json_schema",
    )
    case_count = int(scenario.get("case_count", 1))
    require(1 <= case_count <= 70, "serve_response_format_matrix.case_count must be 1..70")
    thinking = scenario.get("enable_thinking")
    require(isinstance(thinking, bool), "enable_thinking must be a boolean")
    preset = str(scenario.get("preset") or "")
    expected_preset = "P_THINKING" if thinking else "P_NO_THINKING"
    require(preset == expected_preset, f"preset must be {expected_preset}")
    sampling = scenario.get("sampling")
    require(isinstance(sampling, dict), "serve_response_format_matrix.sampling is required")
    required_sampling = {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "seed",
        "stop",
    }
    missing_sampling = sorted(required_sampling - sampling.keys())
    require(not missing_sampling, f"sampling fields missing: {missing_sampling}")

    categories = ("required", "type", "additionalProperties", "enum")
    cases: list[dict[str, Any]] = []
    for ordinal in range(case_count):
        marker = f"{scenario_slug(name)}-{ordinal:03d}"
        category = categories[ordinal % len(categories)]
        if format_type == "json_schema":
            expected, schema = strict_schema_case(category, marker, ordinal)
        else:
            expected = {"marker": marker, "ordinal": ordinal}
            schema = None
        cases.append(
            {
                "ordinal": ordinal,
                "marker": marker,
                "category": category if format_type == "json_schema" else None,
                "expected": expected,
                "schema": schema,
            }
        )
    return {
        "name": name,
        "format": format_type,
        "case_count": case_count,
        "enable_thinking": thinking,
        "preset": preset,
        "sampling": sampling,
        "required_sampling": required_sampling,
        "cases": cases,
    }


def response_format_matrix_contract(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    seen_names: set[str] = set()
    expected_owners: dict[str, dict[str, str]] = {"json_schema": {}, "json_object": {}}
    schema_owners: dict[str, str] = {}
    category_counts = {
        "required": 0,
        "type": 0,
        "additionalProperties": 0,
        "enum": 0,
    }
    rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        if str(scenario.get("type")) != "serve_response_format_matrix":
            continue
        plan = response_format_matrix_plan(scenario)
        name = str(plan["name"])
        require(name not in seen_names, f"duplicate response-format matrix scenario name: {name}")
        seen_names.add(name)
        format_type = str(plan["format"])
        for case in plan["cases"]:
            owner = f"{name}/{int(case['ordinal']):03d}"
            expected_fingerprint = json_fingerprint(case["expected"])
            previous_expected = expected_owners[format_type].get(expected_fingerprint)
            require(
                previous_expected is None,
                f"duplicate {format_type} expected object: {previous_expected} and {owner}",
            )
            expected_owners[format_type][expected_fingerprint] = owner
            schema_fingerprint = None
            category = case["category"]
            if format_type == "json_schema":
                schema_fingerprint = json_fingerprint(case["schema"])
                previous_schema = schema_owners.get(schema_fingerprint)
                require(
                    previous_schema is None,
                    f"duplicate json_schema schema: {previous_schema} and {owner}",
                )
                schema_owners[schema_fingerprint] = owner
                category_counts[str(category)] += 1
            rows.append(
                {
                    "scenario": name,
                    "ordinal": case["ordinal"],
                    "format": format_type,
                    "category": category,
                    "marker": case["marker"],
                    "expected_fingerprint": expected_fingerprint,
                    "schema_fingerprint": schema_fingerprint,
                }
            )

    counts = {
        format_type: sum(row["format"] == format_type for row in rows)
        for format_type in ("json_schema", "json_object")
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "matrix_scenario_count": len(seen_names),
        "case_counts": counts,
        "unique_expected_object_counts": {
            key: len(value) for key, value in expected_owners.items()
        },
        "unique_json_schema_count": len(schema_owners),
        "json_schema_category_counts": category_counts,
        "cases": rows,
    }


def parse_sse(body: str) -> dict[str, Any]:
    done_count = 0
    malformed_json = 0
    content_delta_count = 0
    output_text = ""
    usage_chunks = 0
    chunks = 0
    finish_reasons: list[str] = []
    usage_payloads: list[dict[str, Any]] = []
    tool_call_deltas: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            done_count += 1
            continue
        if not data:
            continue
        chunks += 1
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            malformed_json += 1
            continue
        error = parsed.get("error")
        if isinstance(error, dict):
            errors.append(error)
        usage = parsed.get("usage")
        if usage:
            require(isinstance(usage, dict), f"SSE usage must be an object: {usage!r}")
            usage_chunks += 1
            usage_payloads.append(usage)
        for choice in parsed.get("choices", []):
            if not isinstance(choice, dict):
                continue
            reason = choice.get("finish_reason")
            if reason is not None:
                finish_reasons.append(str(reason))
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            calls = delta.get("tool_calls") or []
            if isinstance(calls, list):
                tool_call_deltas.extend(call for call in calls if isinstance(call, dict))
            text = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or ""
            if str(text):
                output_text += str(text)
                content_delta_count += 1
    return {
        "done_count": done_count,
        "malformed_json": malformed_json,
        "content_delta_count": content_delta_count,
        "usage_chunks": usage_chunks,
        "chunk_count": chunks,
        "output_text": output_text,
        "finish_reasons": finish_reasons,
        "usage_payloads": usage_payloads,
        "tool_call_deltas": tool_call_deltas,
        "errors": errors,
    }


def validate_sync_tool_schema_priority(data: dict[str, Any], marker: str) -> dict[str, Any]:
    choice = first_choice(data)
    require(choice.get("finish_reason") == "tool_calls", f"bad finish_reason: {choice}")
    message = choice.get("message")
    require(isinstance(message, dict), f"missing assistant message: {choice}")
    require(message.get("content") in (None, ""), f"tool response leaked content: {message}")
    calls = message.get("tool_calls")
    require(isinstance(calls, list) and len(calls) == 1, f"expected one tool call: {message}")
    return validate_tool_call(calls[0], marker)


def validate_stream_tool_schema_priority(parsed: dict[str, Any], marker: str) -> dict[str, Any]:
    require(parsed["done_count"] == 1, f"stream [DONE] count {parsed['done_count']} != 1")
    require(parsed["malformed_json"] == 0, f"stream malformed_json={parsed['malformed_json']}")
    require(parsed["usage_chunks"] == 1, f"stream usage_chunks={parsed['usage_chunks']} != 1")
    require(parsed["content_delta_count"] == 0, "required tool stream emitted assistant content")
    require(not parsed["errors"], f"stream emitted errors: {parsed['errors']}")
    require(
        parsed["finish_reasons"] == ["tool_calls"],
        f"stream finish reasons were {parsed['finish_reasons']}",
    )
    calls = reconstruct_stream_tool_calls(parsed["tool_call_deltas"])
    require(len(calls) == 1, f"expected one streamed tool call: {calls}")
    return validate_tool_call(calls[0], marker)


def reconstruct_stream_tool_calls(deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: dict[int, dict[str, Any]] = {}
    for delta in deltas:
        index = int(delta.get("index") or 0)
        call = calls.setdefault(
            index,
            {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        if delta.get("id"):
            call["id"] = str(delta["id"])
        if delta.get("type"):
            call["type"] = str(delta["type"])
        function = delta.get("function")
        if isinstance(function, dict):
            if function.get("name"):
                call["function"]["name"] += str(function["name"])
            if function.get("arguments"):
                call["function"]["arguments"] += str(function["arguments"])
    return [calls[index] for index in sorted(calls)]


def validate_tool_call(call: Any, marker: str) -> dict[str, Any]:
    require(isinstance(call, dict), f"tool call must be an object: {call!r}")
    require(call.get("type") == "function", f"tool call type must be function: {call}")
    function = call.get("function")
    require(isinstance(function, dict), f"tool call missing function: {call}")
    require(function.get("name") == "echo_value", f"wrong tool name: {function}")
    raw_arguments = function.get("arguments")
    require(isinstance(raw_arguments, str), f"tool arguments must be a JSON string: {function}")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"tool arguments are invalid JSON: {raw_arguments!r}") from exc
    require(arguments == {"value": marker}, f"tool arguments {arguments!r} != marker {marker!r}")
    return {"name": "echo_value", "arguments": arguments}


def parse_json_events(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events


def assistant_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") == "assistant"]


def exact_text_matches(actual: Any, expected: Any) -> bool:
    expected_text = str(expected).strip()
    return bool(expected_text) and str(actual).strip() == expected_text


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_health(base_url: str, timeout: int) -> dict[str, Any]:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            status, body = get_url(base_url.rstrip("/") + "/health", timeout=2)
            if status == 200:
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise ScenarioError(f"health endpoint returned invalid JSON: {exc}") from exc
                if not isinstance(data, dict):
                    raise ScenarioError("health endpoint must return a JSON object")
                return {"status": "pass", "http_status": status, "response": data}
            last = f"status={status} body={body[:200]}"
        except Exception as exc:
            last = repr(exc)
        time.sleep(0.5)
    raise ScenarioError(f"server did not become healthy within {timeout}s; last={last}")


ADMISSION_QUIESCENT_FIELDS = (
    "queue_depth",
    "active_prefill",
    "active_decode",
    "current_batch_size",
)


def admission_health_snapshot(base_url: str, timeout: float) -> dict[str, Any]:
    status, body = get_url(
        base_url.rstrip("/") + "/health",
        timeout=max(0.1, timeout),
    )
    data = parse_json_response("admission health", status, body)
    admission = data.get("admission")
    require(isinstance(admission, dict), "health response is missing admission object")
    counters: dict[str, int] = {}
    for key in ("effective_max_concurrent", *ADMISSION_QUIESCENT_FIELDS):
        value = admission.get(key)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"health admission.{key} must be a non-negative integer: {admission}",
        )
        counters[key] = value
    return {
        "captured_at": iso_now(),
        "http_status": status,
        "admission": counters,
        "response": data,
    }


def wait_admission_quiescent(
    base_url: str,
    *,
    timeout: float,
    poll_interval: float = 0.05,
) -> dict[str, Any]:
    require(timeout > 0, "admission quiescence timeout must be positive")
    require(poll_interval > 0, "admission poll interval must be positive")
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    last: dict[str, Any] | None = None
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            break
        snapshot = admission_health_snapshot(base_url, min(2.0, max(0.1, timeout - elapsed)))
        snapshot["elapsed_sec"] = time.monotonic() - started
        samples.append(snapshot)
        last = snapshot
        if all(snapshot["admission"][key] == 0 for key in ADMISSION_QUIESCENT_FIELDS):
            return {
                "status": "pass",
                "elapsed_sec": snapshot["elapsed_sec"],
                "sample_count": len(samples),
                "samples": samples,
                "terminal": snapshot,
            }
        time.sleep(poll_interval)
    raise ScenarioError(
        f"admission did not become quiescent within {timeout:.3f}s; "
        f"last={last['admission'] if last else None}"
    )


def capture_health(
    base_url: str, out: Path, timeout: int, filename: str = "server.health.json"
) -> dict[str, Any]:
    result = wait_health(base_url, timeout)
    response = result["response"]
    artifact = {**response, "status": "pass", "http_status": result["http_status"]}
    write_json(out / filename, artifact)
    return {"status": "pass", "http_status": result["http_status"]}


def selected_scenarios(scenarios: list[dict[str, Any]], only: list[str]) -> list[dict[str, Any]]:
    if not only:
        return scenarios
    require(
        len(only) == len(set(only)),
        f"--only contains duplicate scenario names: {only!r}",
    )
    allowed = set(only)
    available = {str(scenario.get("name")) for scenario in scenarios}
    missing = sorted(allowed - available)
    require(not missing, f"--only did not match manifest scenarios: {missing!r}")
    selected = [scenario for scenario in scenarios if str(scenario.get("name")) in allowed]
    require(selected, "--only selected zero scenarios")
    return selected


class ScenarioRunner:
    def __init__(self, args: argparse.Namespace, manifest: dict[str, Any]) -> None:
        self.args = args
        self.manifest = manifest
        self.out = args.out.resolve()
        self.model = args.model or str(manifest.get("model") or "")
        self.backend = args.backend or str(manifest.get("backend") or "auto")
        self.ferrum_bin = Path(args.ferrum_bin or manifest.get("ferrum_bin") or "target/release/ferrum")
        self.timeout = int(args.timeout or manifest.get("timeout_sec") or 180)
        self.base_url = args.base_url or self.manifest_base_url()
        self.started_server: StartedServer | None = None
        self.run_observability_roots: list[Path] = []
        self.git_sha = git_output(["rev-parse", "HEAD"])
        self.dirty_status = git_dirty_status()
        self.execution_receipt: dict[str, Any] = {
            "schema_version": 1,
            "mode": "external",
            "runner_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": file_sha256(args.manifest.resolve()),
            "cwd": str(Path.cwd().resolve()),
            "git_sha": self.git_sha,
            "dirty_status": self.dirty_status,
        }

    def observability_config(self) -> dict[str, Any]:
        raw = self.manifest.get("observability")
        config = raw if isinstance(raw, dict) else {}
        enabled = bool(self.args.observability or config.get("enabled") is True)
        return {
            "enabled": enabled,
            "profile_detail": str(self.args.profile_detail or config.get("profile_detail") or "basic"),
            "profile_sample_rate": float(
                self.args.profile_sample_rate
                if self.args.profile_sample_rate is not None
                else config.get("profile_sample_rate", 1.0)
            ),
        }

    def observability_enabled(self) -> bool:
        return bool(self.observability_config()["enabled"])

    def observability_args(self, root: Path) -> list[str]:
        config = self.observability_config()
        root.mkdir(parents=True, exist_ok=True)
        return [
            "--profile-jsonl",
            str(root / "profile.jsonl"),
            "--profile-detail",
            str(config["profile_detail"]),
            "--memory-profile-jsonl",
            str(root / "memory_profile.jsonl"),
            "--scheduler-trace-jsonl",
            str(root / "scheduler_trace.jsonl"),
            "--request-dump-dir",
            str(root / "request_dump"),
            "--profile-sample-rate",
            str(config["profile_sample_rate"]),
        ]

    def manifest_base_url(self) -> str | None:
        server = self.manifest.get("server")
        if isinstance(server, dict):
            value = server.get("base_url")
            if isinstance(value, str) and value:
                return value
        return None

    def scenario_types(self) -> set[str]:
        return {str(scenario.get("type")) for scenario in self.scenarios()}

    def scenarios(self) -> list[dict[str, Any]]:
        scenarios = self.manifest.get("scenarios")
        if not isinstance(scenarios, list) or not scenarios:
            raise ScenarioError("manifest.scenarios must be a non-empty list")
        for idx, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict):
                raise ScenarioError(f"manifest.scenarios[{idx}] must be an object")
            if not scenario.get("name") or not scenario.get("type"):
                raise ScenarioError(f"manifest.scenarios[{idx}] must include name and type")
        names = [str(scenario["name"]) for scenario in scenarios]
        require(
            len(names) == len(set(names)),
            "manifest.scenarios names must be unique",
        )
        return scenarios

    def needs_serve(self) -> bool:
        return bool(self.scenario_types() & SERVE_TYPES)

    def should_start_server(self) -> bool:
        if self.args.start_server:
            return True
        server = self.manifest.get("server")
        return isinstance(server, dict) and server.get("mode") == "start"

    def ensure_server(self) -> None:
        if not self.needs_serve():
            return
        if self.base_url and not self.should_start_server():
            if self.observability_enabled():
                raise ScenarioError(
                    "run_scenarios observability requires manifest.server.mode=start "
                    "or --start-server so serve flags can be passed through"
                )
            capture_health(self.base_url, self.out, self.timeout)
            return
        if not self.model:
            raise ScenarioError("serve scenarios require --model or manifest.model")
        port = int(self.args.port or self.manifest.get("port") or free_port())
        self.base_url = f"http://127.0.0.1:{port}"
        effective = self.out / "server.effective_config.json"
        decision_trace = self.out / "server.decision_trace.jsonl"
        cmd = [
            str(self.ferrum_bin),
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--effective-config-json",
            str(effective),
            "--decision-trace-jsonl",
            str(decision_trace),
        ]
        if self.observability_enabled():
            cmd.extend(self.observability_args(self.out / "observability" / "serve"))
        server = self.manifest.get("server")
        if isinstance(server, dict):
            extra_args = server.get("args")
            if isinstance(extra_args, list):
                cmd.extend(str(part) for part in extra_args)
        cmd.append(self.model)
        binary_path = self.ferrum_bin
        if not binary_path.is_absolute():
            binary_path = repo_root() / binary_path
        binary_path = binary_path.resolve(strict=True)
        cmd[0] = str(binary_path)
        binary_sha256 = file_sha256(binary_path)
        hardware = command_receipt(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        if self.backend == "cuda":
            require(hardware["returncode"] == 0, f"CUDA hardware probe failed: {hardware}")
            gpu_rows = [row.strip() for row in hardware["stdout"].splitlines() if row.strip()]
            require(len(gpu_rows) == 1, f"CUDA S2 requires exactly one GPU: {gpu_rows!r}")
            require("RTX 4090" in gpu_rows[0], f"CUDA S2 requires RTX 4090: {gpu_rows[0]!r}")
        child_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("FERRUM_")
        }
        child_env["NO_COLOR"] = "1"
        self.execution_receipt.update(
            {
                "mode": "start",
                "server_argv": cmd,
                "binary_path": str(binary_path),
                "binary_sha256": binary_sha256,
                "hardware": hardware,
                "removed_hidden_env_names": sorted(
                    key for key in os.environ if key.startswith("FERRUM_")
                ),
                "child_env": {
                    key: child_env[key]
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "HF_HOME",
                        "HF_HUB_CACHE",
                        "RUST_BACKTRACE",
                        "RUST_LOG",
                    )
                    if key in child_env
                },
            }
        )
        self.started_server = StartedServer(cmd, self.out / "server.log", child_env)
        capture_health(self.base_url, self.out, self.timeout)

    def finalize_execution_receipt(self) -> dict[str, Any]:
        receipt = dict(self.execution_receipt)
        if self.started_server is not None:
            server = self.started_server
            receipt.update(
                {
                    "server_started_at": server.started_at,
                    "server_finished_at": server.finished_at,
                    "server_returncode": server.returncode,
                }
            )
            evidence_files = {
                "effective_config": self.out / "server.effective_config.json",
                "decision_trace": self.out / "server.decision_trace.jsonl",
                "server_log": self.out / "server.log",
                "health_before": self.out / "server.health.json",
                "health_after": self.out / "server.health.after.json",
            }
            evidence: dict[str, Any] = {}
            for label, path in evidence_files.items():
                require(path.is_file() and not path.is_symlink(), f"missing execution evidence: {path}")
                require(path.stat().st_size > 0, f"empty execution evidence: {path}")
                evidence[label] = {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            receipt["evidence_files"] = evidence
            require(
                server.returncode in (0, -signal.SIGTERM),
                f"unexpected ferrum serve return code: {server.returncode}",
            )
        receipt = with_canonical_sha256(receipt)
        write_json(self.out / "execution_receipt.json", receipt)
        return receipt

    def run_all(self) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        inputs = self.out / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve(), inputs / "run_scenarios.py")
        shutil.copyfile(self.args.manifest.resolve(), inputs / "scenario_manifest.json")
        self.execution_receipt["input_artifacts"] = {
            "runner": {
                "path": str(inputs / "run_scenarios.py"),
                "sha256": file_sha256(inputs / "run_scenarios.py"),
            },
            "manifest": {
                "path": str(inputs / "scenario_manifest.json"),
                "sha256": file_sha256(inputs / "scenario_manifest.json"),
            },
        }
        started_at = iso_now()
        scenarios = self.scenarios()
        selected = selected_scenarios(scenarios, self.args.only)
        self.execution_receipt.update(
            {
                "backend": self.backend,
                "model": self.model,
                "selected_scenarios": [str(scenario["name"]) for scenario in selected],
            }
        )
        matrix_contract = response_format_matrix_contract(selected)
        write_json(self.out / "response_format_matrix_contract.json", matrix_contract)
        results: list[dict[str, Any]] = []
        failures = 0
        skipped = 0
        observability: dict[str, Any] | None = None
        try:
            self.ensure_server()
            for scenario in selected:
                result = self.run_one(scenario)
                if result["status"] == "fail":
                    failures += 1
                elif result["status"] == "skipped":
                    skipped += 1
                results.append(result)
            if self.needs_serve() and self.base_url:
                capture_health(
                    self.base_url,
                    self.out,
                    self.timeout,
                    filename="server.health.after.json",
                )
            observability = self.observability_summary()
        finally:
            if self.started_server is not None:
                self.started_server.stop()
        self.execution_receipt.update(
            {
                "scenario_count": len(results),
                "failed": failures,
                "skipped": skipped,
            }
        )
        execution_receipt = self.finalize_execution_receipt()
        status = "fail" if failures or skipped else "pass"
        summary = {
            "schema_version": 1,
            "status": status,
            "manifest": str(self.args.manifest),
            "artifact_dir": str(self.out),
            "model": self.model,
            "backend": self.backend,
            "base_url": self.base_url,
            "git_sha": self.git_sha,
            "dirty_status": self.dirty_status,
            "started_at": started_at,
            "finished_at": iso_now(),
            "scenario_count": len(results),
            "manifest_scenario_count": len(scenarios),
            "requested_scenarios": list(self.args.only),
            "selected_scenarios": [str(scenario["name"]) for scenario in selected],
            "failed": failures,
            "skipped": skipped,
            "scenarios": results,
            "response_format_matrix_contract": {
                "artifact": str(self.out / "response_format_matrix_contract.json"),
                "case_counts": matrix_contract["case_counts"],
                "unique_json_schema_count": matrix_contract["unique_json_schema_count"],
            },
            "observability": observability,
            "execution_receipt": {
                "artifact": str(self.out / "execution_receipt.json"),
                "artifact_sha256": file_sha256(self.out / "execution_receipt.json"),
                "canonical_sha256": execution_receipt["canonical_sha256"],
                "mode": execution_receipt["mode"],
                "runner_sha256": execution_receipt["runner_sha256"],
                "manifest_sha256": execution_receipt["manifest_sha256"],
                "binary_sha256": execution_receipt.get("binary_sha256"),
            },
            "pass_line": f"BACKEND REGRESSION SMOKE PASS: {self.out}" if status == "pass" else None,
        }
        write_json(self.out / "summary.json", summary)
        write_artifact_tree(self.out)
        return summary

    def observability_summary(self) -> dict[str, Any] | None:
        if not self.observability_enabled():
            return None
        roots: dict[str, Any] = {
            "run": [str(root) for root in self.run_observability_roots],
            "serve": None,
        }
        serve_root = self.out / "observability" / "serve"
        if serve_root.exists():
            roots["serve"] = str(serve_root)
        profile_paths: list[str] = []
        scheduler_trace_paths: list[str] = []
        request_dump_dirs: list[str] = []
        for root in [*self.run_observability_roots, serve_root]:
            if not root.exists():
                continue
            for filename, bucket in [
                ("profile.jsonl", profile_paths),
                ("memory_profile.jsonl", profile_paths),
                ("scheduler_trace.jsonl", profile_paths),
            ]:
                path = root / filename
                if path.is_file():
                    bucket.append(str(path))
                    if filename == "scheduler_trace.jsonl":
                        scheduler_trace_paths.append(str(path))
            request_dump = root / "request_dump"
            if request_dump.is_dir():
                request_dump_dirs.append(str(request_dump))
        summary = {
            "enabled": True,
            "roots": roots,
            "profile_paths": profile_paths,
            "scheduler_trace_paths": scheduler_trace_paths,
            "request_dump_dirs": request_dump_dirs,
        }
        write_json(self.out / "observability_summary.json", summary)
        return summary

    def run_one(self, scenario: dict[str, Any]) -> dict[str, Any]:
        name = str(scenario["name"])
        typ = str(scenario["type"])
        scenario_dir = self.out / scenario_slug(name)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            result = self.dispatch(typ, scenario, scenario_dir)
            result.update(
                {
                    "name": name,
                    "type": typ,
                    "status": result.get("status", "pass"),
                    "artifact": str(scenario_dir / "result.json"),
                    "duration_sec": time.monotonic() - started,
                }
            )
        except Exception as exc:
            if scenario.get("optional") is True and is_optional_skip(exc):
                result = {
                    "name": name,
                    "type": typ,
                    "status": "skipped",
                    "reason": str(exc),
                    "artifact": str(scenario_dir / "result.json"),
                    "duration_sec": time.monotonic() - started,
                }
            else:
                result = {
                    "name": name,
                    "type": typ,
                    "status": "fail",
                    "error": str(exc),
                    "artifact": str(scenario_dir / "result.json"),
                    "duration_sec": time.monotonic() - started,
                }
        write_json(scenario_dir / "result.json", result)
        return result

    def dispatch(self, typ: str, scenario: dict[str, Any], out: Path) -> dict[str, Any]:
        if typ == "serve_chat":
            return self.serve_chat(scenario, out)
        if typ == "serve_multiturn_recall":
            return self.serve_multiturn_recall(scenario, out)
        if typ == "serve_stateful_loop":
            return self.serve_stateful_loop(scenario, out)
        if typ == "serve_stream":
            return self.serve_stream(scenario, out)
        if typ == "serve_stream_equivalence_unicode":
            return self.serve_stream_equivalence_unicode(scenario, out)
        if typ == "serve_disconnect_release":
            return self.serve_disconnect_release(scenario, out)
        if typ == "serve_structured_output":
            return self.serve_structured_output(scenario, out)
        if typ == "serve_response_format_matrix":
            return self.serve_response_format_matrix(scenario, out)
        if typ == "serve_negative_api_matrix":
            return self.serve_negative_api_matrix(scenario, out)
        if typ == "serve_text_only_modality_matrix":
            return self.serve_text_only_modality_matrix(scenario, out)
        if typ == "serve_context_limit":
            return self.serve_context_limit(scenario, out)
        if typ == "serve_concurrency_quality":
            return self.serve_concurrency_quality(scenario, out)
        if typ == "serve_tool_schema_priority":
            return self.serve_tool_schema_priority(scenario, out)
        if typ == "serve_tool_call":
            return self.serve_tool_call(out)
        if typ == "serve_python_openai_sdk":
            return self.serve_python_openai_sdk(scenario, out)
        if typ == "run_multiturn":
            return self.run_multiturn(scenario, out)
        if typ == "run_first_token_ux":
            return self.run_first_token_ux(scenario, out)
        raise ScenarioError(f"unknown scenario type: {typ}")

    def require_base_url(self) -> str:
        if not self.base_url:
            raise ScenarioError("serve scenario requires --base-url or manifest.server.mode=start")
        return self.base_url

    def serve_chat(self, scenario: dict[str, Any], out: Path) -> dict[str, Any]:
        prompt = str(scenario.get("prompt") or "Say exactly: ferrum-ok")
        expected = [str(item) for item in scenario.get("expected_contains", ["ferrum-ok"])]
        payload = chat_payload(
            self.model,
            [{"role": "user", "content": prompt}],
            max_tokens=int(scenario.get("max_tokens", 128)),
            temperature=float(scenario.get("temperature", 0)),
        )
        status, body = post_json(self.require_base_url(), payload, timeout=self.timeout)
        (out / "response.json").write_text(body, errors="replace")
        data = parse_json_response("serve_chat", status, body)
        text = message_text(data)
        for needle in expected:
            require(needle in text, f"serve_chat missing expected text {needle!r}: {text[:500]}")
        require(finish_reason(data) != "length", "serve_chat finish_reason is length")
        return {"status": "pass", "http_status": status, "content": text[:1000]}

    def serve_multiturn_recall(self, scenario: dict[str, Any], out: Path) -> dict[str, Any]:
        secret = str(scenario.get("secret") or "ferrum-blue")
        first = str(scenario.get("first_prompt") or f"Remember this secret code: {secret}. Reply OK.")
        second = str(scenario.get("second_prompt") or "Reply with only the secret code.")
        messages = [{"role": "user", "content": first}]
        first_data = self.post_chat_messages("serve_multiturn_1", messages, out / "turn1.json", scenario)
        messages.append({"role": "assistant", "content": message_text(first_data)})
        messages.append({"role": "user", "content": second})
        second_data = self.post_chat_messages("serve_multiturn_2", messages, out / "turn2.json", scenario)
        text = message_text(second_data)
        require(secret in text, f"serve multiturn did not recall {secret!r}: {text[:500]}")
        require(finish_reason(second_data) != "length", "serve multiturn finish_reason is length")
        return {"status": "pass", "assistant_turns": 2, "recalled_secret": True}

    def serve_stateful_loop(self, scenario: dict[str, Any], out: Path) -> dict[str, Any]:
        prompts = scenario.get("prompts")
        if not isinstance(prompts, list) or len(prompts) < 5:
            prompts = [
                "Remember code ferrum-loop-blue. Reply OK.",
                "Say one short word about Paris.",
                "Reply with only the remembered code.",
                "Say one short word about Rust.",
                "Again reply with only the remembered code.",
            ]
        messages: list[dict[str, Any]] = []
        responses: list[str] = []
        length_finishes = 0
        isolated_think_close = 0
        repeated_prefixes = 0
        previous_prefix = ""
        for idx, prompt in enumerate(prompts):
            messages.append({"role": "user", "content": str(prompt)})
            data = self.post_chat_messages(
                f"serve_stateful_loop_{idx + 1}",
                messages,
                out / f"turn{idx + 1}.json",
                scenario,
            )
            text = message_text(data)
            if finish_reason(data) == "length":
                length_finishes += 1
            if text.strip() == "</think>":
                isolated_think_close += 1
            prefix = re.sub(r"\s+", "", text[:64])
            if prefix and prefix == previous_prefix:
                repeated_prefixes += 1
            previous_prefix = prefix
            if text.strip():
                responses.append(text)
                messages.append({"role": "assistant", "content": text})
        require(len(responses) >= int(scenario.get("min_non_empty_assistant", 4)), "too few responses")
        require(length_finishes == 0, f"stateful loop length_finishes={length_finishes}")
        require(isolated_think_close == 0, "stateful loop emitted isolated </think>")
        require(repeated_prefixes == 0, f"stateful loop repeated_prefixes={repeated_prefixes}")
        return {
            "status": "pass",
            "assistant_responses": len(responses),
            "length_finishes": length_finishes,
            "repeated_prefixes": repeated_prefixes,
        }

    def post_chat_messages(
        self,
        label: str,
        messages: list[dict[str, Any]],
        path: Path,
        scenario: dict[str, Any],
    ) -> dict[str, Any]:
        payload = chat_payload(
            self.model,
            messages,
            max_tokens=int(scenario.get("max_tokens", 192)),
            temperature=float(scenario.get("temperature", 0)),
        )
        status, body = post_json(self.require_base_url(), payload, timeout=self.timeout)
        path.write_text(body, errors="replace")
        return parse_json_response(label, status, body)

    def serve_stream(self, scenario: dict[str, Any], out: Path) -> dict[str, Any]:
        payload = chat_payload(
            self.model,
            [{"role": "user", "content": str(scenario.get("prompt") or "Say hello briefly.")}],
            max_tokens=int(scenario.get("max_tokens", 128)),
            temperature=float(scenario.get("temperature", 0)),
        )
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        status, body, event_times = request_sse(self.require_base_url(), payload, timeout=self.timeout)
        (out / "stream.sse").write_text(body, errors="replace")
        require(status == 200, f"stream expected HTTP 200, got {status}: {body[:500]}")
        parsed = parse_sse(body)
        assert_no_bad_text("serve_stream", parsed["output_text"])
        require(parsed["done_count"] == 1, f"stream [DONE] count {parsed['done_count']} != 1")
        require(parsed["content_delta_count"] > 0, "stream emitted no content delta")
        require(parsed["malformed_json"] == 0, f"stream malformed_json={parsed['malformed_json']}")
        require(parsed["usage_chunks"] == 1, f"stream usage_chunks={parsed['usage_chunks']} != 1")
        return {"status": "pass", **parsed, "event_count": len(event_times)}

    def serve_stream_equivalence_unicode(
        self, scenario: dict[str, Any], out: Path
    ) -> dict[str, Any]:
        case_count = int(scenario.get("case_count", 20))
        require(case_count >= 3, "Unicode stream equivalence requires at least three cases")
        categories = ("chinese", "emoji", "combining")
        category_counts = {category: 0 for category in categories}
        results: list[dict[str, Any]] = []

        for ordinal in range(case_count):
            category = categories[ordinal % len(categories)]
            if category == "chinese":
                value = f"你好，Ferrum，编号 {ordinal:03d}"
            elif category == "emoji":
                value = f"Ferrum 🔥🚀 {ordinal:03d}"
            else:
                value = f"Cafe\u0301 Ferrum {ordinal:03d}"
            expected = {
                "category": category,
                "ordinal": ordinal,
                "value": value,
            }
            expected_text = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
            prompt = (
                "Return the following JSON object exactly, with no markdown or extra text. "
                f"EXACT_JSON:{expected_text}"
            )
            common_payload = chat_payload(
                self.model,
                [{"role": "user", "content": prompt}],
                max_tokens=int(scenario.get("max_tokens", 1024)),
                temperature=0,
            )
            if scenario.get("preset") == "P_DETERMINISTIC":
                apply_deterministic_sampling(common_payload, scenario)
            else:
                common_payload["seed"] = int(scenario.get("seed", 9271))
                common_payload["chat_template_kwargs"] = {"enable_thinking": False}
            common_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": f"unicode_equivalence_{ordinal:03d}",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "const": category},
                            "ordinal": {"type": "integer", "const": ordinal},
                            "value": {"type": "string", "const": value},
                        },
                        "required": ["category", "ordinal", "value"],
                        "additionalProperties": False,
                    },
                },
            }
            common_payload["metadata"] = {
                "ferrum_scenario": "stream_equivalence_unicode",
                "ferrum_case": f"{category}-{ordinal:03d}",
            }
            pair_fingerprint = json_fingerprint(common_payload)
            case_out = out / f"{ordinal:03d}-{category}"
            case_out.mkdir(parents=True, exist_ok=True)

            sync_payload = dict(common_payload)
            write_json(case_out / "sync.request.json", sync_payload)
            sync_status, sync_body = post_json(
                self.require_base_url(), sync_payload, timeout=self.timeout
            )
            (case_out / "sync.response.json").write_text(sync_body, encoding="utf-8")
            sync_data = parse_json_response(
                f"Unicode equivalence {ordinal} sync", sync_status, sync_body
            )
            sync_text = message_text(sync_data)
            assert_no_bad_text(f"Unicode equivalence {ordinal} sync", sync_text)
            try:
                sync_object = json.loads(sync_text)
            except json.JSONDecodeError as error:
                raise ScenarioError(
                    f"Unicode equivalence {ordinal} sync content is not JSON: {sync_text!r}"
                ) from error
            require(
                sync_object == expected,
                f"Unicode equivalence {ordinal} sync object mismatch: {sync_object!r}",
            )
            sync_finish = finish_reason(sync_data)
            require(sync_finish == "stop", f"Unicode equivalence {ordinal} sync finish={sync_finish}")
            sync_usage = response_usage(f"Unicode equivalence {ordinal} sync", sync_data)

            stream_payload = dict(common_payload)
            stream_payload["stream"] = True
            stream_payload["stream_options"] = {"include_usage": True}
            write_json(case_out / "stream.request.json", stream_payload)
            stream_status, stream_wire, stream_body, wire_evidence = request_sse_incremental(
                self.require_base_url(),
                stream_payload,
                timeout=self.timeout,
                read_size=1,
            )
            (case_out / "stream.response.sse").write_bytes(stream_wire)
            write_json(case_out / "stream.wire.json", wire_evidence)
            require(
                stream_status == 200,
                f"Unicode equivalence {ordinal} stream HTTP {stream_status}: {stream_body[:500]}",
            )
            assert_no_bad_text(f"Unicode equivalence {ordinal} stream wire", stream_body)
            parsed = parse_sse(stream_body)
            assert_no_bad_text(f"Unicode equivalence {ordinal} stream", parsed["output_text"])
            require(parsed["done_count"] == 1, f"Unicode equivalence {ordinal} DONE mismatch")
            require(parsed["malformed_json"] == 0, f"Unicode equivalence {ordinal} malformed SSE")
            require(parsed["usage_chunks"] == 1, f"Unicode equivalence {ordinal} usage mismatch")
            require(parsed["content_delta_count"] > 0, f"Unicode equivalence {ordinal} no content")
            require(not parsed["errors"], f"Unicode equivalence {ordinal} stream errors")
            require(
                parsed["finish_reasons"] == [sync_finish],
                f"Unicode equivalence {ordinal} finish mismatch: {parsed['finish_reasons']}",
            )
            require(
                parsed["output_text"] == sync_text,
                f"Unicode equivalence {ordinal} stream/non-stream content differs",
            )
            stream_usage = response_usage(
                f"Unicode equivalence {ordinal} stream",
                {"usage": parsed["usage_payloads"][0]},
            )
            require(
                stream_usage == sync_usage,
                f"Unicode equivalence {ordinal} usage differs: {stream_usage} != {sync_usage}",
            )
            try:
                stream_object = json.loads(parsed["output_text"])
            except json.JSONDecodeError as error:
                raise ScenarioError(
                    f"Unicode equivalence {ordinal} stream content is not JSON"
                ) from error
            require(stream_object == expected, f"Unicode equivalence {ordinal} stream object mismatch")
            require(
                wire_evidence["split_boundary_count"] > 0,
                f"Unicode equivalence {ordinal} did not exercise a multibyte wire split",
            )

            case_result = with_canonical_sha256(
                {
                    "status": "pass",
                    "ordinal": ordinal,
                    "category": category,
                    "expected": expected,
                    "expected_sha256": json_fingerprint(expected),
                    "pair_payload_sha256": pair_fingerprint,
                    "sync_request_sha256": json_fingerprint(sync_payload),
                    "stream_request_sha256": json_fingerprint(stream_payload),
                    "sync_response_sha256": hashlib.sha256(sync_body.encode("utf-8")).hexdigest(),
                    "stream_response_sha256": wire_evidence["wire_sha256"],
                    "finish_reason": sync_finish,
                    "usage": sync_usage,
                    "wire": wire_evidence,
                }
            )
            write_json(case_out / "result.json", case_result)
            results.append(case_result)
            category_counts[category] += 1

        require(all(count > 0 for count in category_counts.values()), "Unicode category missing")
        return {
            "status": "pass",
            "case_count": len(results),
            "category_counts": category_counts,
            "exact_content_matches": len(results),
            "exact_finish_matches": len(results),
            "exact_usage_matches": len(results),
            "multibyte_split_cases": sum(
                int(case["wire"]["split_boundary_count"] > 0) for case in results
            ),
            "cases": results,
        }

    def serve_disconnect_release(
        self, scenario: dict[str, Any], out: Path
    ) -> dict[str, Any]:
        release_timeout = float(scenario.get("release_timeout_sec", 5.0))
        poll_interval = float(scenario.get("poll_interval_sec", 0.05))
        max_tokens = int(scenario.get("max_tokens", 1024))
        expected_cap = scenario.get("expected_effective_max_concurrent")
        require(max_tokens > 0, "disconnect release max_tokens must be positive")
        if scenario.get("require_scheduler_trace") is True:
            require(
                self.observability_enabled(),
                "disconnect release scheduler-tick validation requires observability",
            )

        before = wait_admission_quiescent(
            self.require_base_url(), timeout=release_timeout, poll_interval=poll_interval
        )
        write_json(out / "health.before.json", before)
        if expected_cap is not None:
            require(
                before["terminal"]["admission"]["effective_max_concurrent"] == int(expected_cap),
                "disconnect release effective capacity does not match the scenario contract",
            )

        marker = str(scenario.get("marker") or "m1-s2-disconnect-release")
        disconnect_payload = chat_payload(
            self.model,
            [
                {
                    "role": "user",
                    "content": (
                        "Write the integers from 1 through 1000, one per line, and do not "
                        "summarize or stop before the requested sequence is complete."
                    ),
                }
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        disconnect_payload["stream"] = True
        disconnect_payload["stream_options"] = {"include_usage": True}
        disconnect_payload["chat_template_kwargs"] = {"enable_thinking": False}
        disconnect_payload["metadata"] = {
            "ferrum_scenario": "disconnect_release",
            "ferrum_disconnect_probe": True,
            "ferrum_marker": marker,
        }
        write_json(out / "disconnect.request.json", disconnect_payload)

        def capture_active_health() -> dict[str, Any]:
            snapshot = admission_health_snapshot(self.require_base_url(), min(2.0, release_timeout))
            active = sum(snapshot["admission"][key] for key in ADMISSION_QUIESCENT_FIELDS)
            require(active > 0, f"disconnect probe was not active at first output: {snapshot}")
            if expected_cap is not None:
                require(
                    snapshot["admission"]["effective_max_concurrent"] == int(expected_cap),
                    "disconnect probe changed effective capacity",
                )
            return snapshot

        disconnected = request_sse_until_output_then_disconnect(
            self.require_base_url(),
            disconnect_payload,
            timeout=self.timeout,
            on_first_output=capture_active_health,
        )
        partial_wire = disconnected.pop("partial_wire")
        require(isinstance(partial_wire, bytes), "disconnect partial wire must be bytes")
        (out / "disconnect.partial.sse").write_bytes(partial_wire)
        require(
            disconnected["http_status"] == 200,
            f"disconnect probe HTTP {disconnected['http_status']}",
        )
        require(
            isinstance(disconnected["first_output_event"], dict),
            "disconnect probe received no output event before closing",
        )
        require(
            isinstance(disconnected["active_health"], dict),
            "disconnect probe did not capture active health",
        )
        write_json(out / "disconnect.observed.json", disconnected)

        released = wait_admission_quiescent(
            self.require_base_url(), timeout=release_timeout, poll_interval=poll_interval
        )
        require(
            released["elapsed_sec"] <= release_timeout,
            f"disconnect release exceeded {release_timeout:.3f}s",
        )
        write_json(out / "health.released.json", released)

        expected = {"marker": marker, "status": "released"}
        expected_text = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        followup_payload = chat_payload(
            self.model,
            [
                {
                    "role": "user",
                    "content": (
                        "Return the following JSON object exactly, with no extra text. "
                        f"EXACT_JSON:{expected_text}"
                    ),
                }
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        followup_payload["seed"] = int(scenario.get("seed", 9271))
        followup_payload["chat_template_kwargs"] = {"enable_thinking": False}
        followup_payload["metadata"] = {
            "ferrum_scenario": "disconnect_release_followup",
            "ferrum_marker": marker,
        }
        followup_payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "disconnect_release_followup",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "marker": {"type": "string", "const": marker},
                        "status": {"type": "string", "const": "released"},
                    },
                    "required": ["marker", "status"],
                    "additionalProperties": False,
                },
            },
        }
        require(
            followup_payload["max_tokens"] == disconnect_payload["max_tokens"],
            "disconnect follow-up must request the same token capacity",
        )
        write_json(out / "followup.request.json", followup_payload)
        followup_status, followup_body = post_json(
            self.require_base_url(), followup_payload, timeout=self.timeout
        )
        (out / "followup.response.json").write_text(followup_body, encoding="utf-8")
        followup = parse_json_response(
            "disconnect same-capacity follow-up", followup_status, followup_body
        )
        followup_text = message_text(followup)
        assert_no_bad_text("disconnect same-capacity follow-up", followup_text)
        try:
            followup_object = json.loads(followup_text)
        except json.JSONDecodeError as error:
            raise ScenarioError("disconnect follow-up content is not JSON") from error
        require(followup_object == expected, f"disconnect follow-up mismatch: {followup_object!r}")
        require(finish_reason(followup) == "stop", "disconnect follow-up did not finish normally")
        followup_usage = response_usage("disconnect same-capacity follow-up", followup)

        after = wait_admission_quiescent(
            self.require_base_url(), timeout=release_timeout, poll_interval=poll_interval
        )
        write_json(out / "health.after.json", after)
        return {
            "status": "pass",
            "marker": marker,
            "max_tokens": max_tokens,
            "effective_max_concurrent": before["terminal"]["admission"][
                "effective_max_concurrent"
            ],
            "time_to_first_output_sec": disconnected["time_to_first_output_sec"],
            "release_elapsed_sec": released["elapsed_sec"],
            "release_timeout_sec": release_timeout,
            "same_capacity_followup": True,
            "followup_usage": followup_usage,
            "scheduler_tick_limit": 2,
            "scheduler_trace_required": scenario.get("require_scheduler_trace") is True,
            "request_fingerprints": {
                "disconnect": json_fingerprint(disconnect_payload),
                "followup": json_fingerprint(followup_payload),
            },
        }

    def serve_structured_output(self, scenario: dict[str, Any], out: Path) -> dict[str, Any]:
        expected = scenario.get("expected_object") or {"answer": "scenario-ok"}
        payload = chat_payload(
            self.model,
            [
                {
                    "role": "user",
                    "content": "Return exactly this JSON object and nothing else: "
                    + json.dumps(expected, ensure_ascii=False),
                }
            ],
            max_tokens=int(scenario.get("max_tokens", 384)),
            temperature=float(scenario.get("temperature", 0)),
        )
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ScenarioObject",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {key: {"type": "string"} for key in expected},
                    "required": sorted(expected),
                },
            },
        }
        status, body = post_json(self.require_base_url(), payload, timeout=self.timeout)
        (out / "response.json").write_text(body, errors="replace")
        data = parse_json_response("serve_structured_output", status, body)
        text = message_text(data)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ScenarioError(f"structured output content is not JSON: {text[:500]}") from exc
        require(parsed == expected, f"structured output {parsed!r} != expected {expected!r}")
        return {"status": "pass", "object": parsed}

    def serve_response_format_matrix(
        self, scenario: dict[str, Any], out: Path
    ) -> dict[str, Any]:
        plan = response_format_matrix_plan(scenario)
        format_type = str(plan["format"])
        thinking = bool(plan["enable_thinking"])
        preset = str(plan["preset"])
        sampling = plan["sampling"]
        required_sampling = plan["required_sampling"]
        category_counts = {
            "required": 0,
            "type": 0,
            "additionalProperties": 0,
            "enum": 0,
        }
        results: list[dict[str, Any]] = []
        for case in plan["cases"]:
            ordinal = int(case["ordinal"])
            marker = str(case["marker"])
            category = case["category"]
            expected = case["expected"]
            if format_type == "json_schema":
                response_format: dict[str, Any] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{scenario_slug(str(plan['name']))}-{category}-{ordinal:03d}",
                        "strict": True,
                        "schema": case["schema"],
                    },
                }
                category_counts[str(category)] += 1
            else:
                response_format = {"type": "json_object"}

            payload = chat_payload(
                self.model,
                [
                    {
                        "role": "user",
                        "content": (
                            "Return one JSON object and no other text. EXACT_JSON:"
                            + json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
                        ),
                    }
                ],
                max_tokens=int(scenario.get("max_tokens", 256)),
                temperature=float(sampling["temperature"]),
            )
            for field in sorted(required_sampling - {"temperature"}):
                payload[field] = sampling[field]
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
            payload["response_format"] = response_format

            case_out = out / f"{ordinal:03d}"
            case_out.mkdir(parents=True, exist_ok=True)
            write_json(case_out / "request.json", payload)
            status, body = post_json(self.require_base_url(), payload, timeout=self.timeout)
            (case_out / "response.json").write_text(body, errors="replace")
            data = parse_json_response(f"{format_type}-{ordinal:03d}", status, body)
            choice = first_choice(data)
            message = choice.get("message")
            require(isinstance(message, dict), f"case {ordinal}: missing assistant message")
            content = message.get("content")
            require(isinstance(content, str), f"case {ordinal}: structured content is not a string")
            require("```" not in content, f"case {ordinal}: markdown fence in structured content")
            require(
                "<think>" not in content and "</think>" not in content,
                f"case {ordinal}: reasoning leaked into structured content",
            )
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ScenarioError(f"case {ordinal}: invalid JSON object: {content[:500]}") from exc
            require(isinstance(parsed, dict), f"case {ordinal}: JSON root is not an object")
            if format_type == "json_schema":
                require(parsed == expected, f"case {ordinal}: {parsed!r} != {expected!r}")
            require(choice.get("finish_reason") != "length", f"case {ordinal}: length finish")
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            require(isinstance(reasoning, str), f"case {ordinal}: reasoning is not a string")
            result = {
                "ordinal": ordinal,
                "format": format_type,
                "category": category if format_type == "json_schema" else None,
                "preset": preset,
                "enable_thinking": thinking,
                "http_status": status,
                "finish_reason": choice.get("finish_reason"),
                "reasoning_chars": len(reasoning),
                "object": parsed,
            }
            write_json(case_out / "result.json", {"status": "pass", **result})
            results.append(result)

        return {
            "status": "pass",
            "format": format_type,
            "preset": preset,
            "enable_thinking": thinking,
            "case_count": len(results),
            "category_counts": category_counts if format_type == "json_schema" else {},
            "cases": results,
        }

    def serve_negative_api_matrix(
        self, scenario: dict[str, Any], out: Path
    ) -> dict[str, Any]:
        categories = (
            "invalid-tool",
            "invalid-schema",
            "invalid-stream-option",
            "invalid-model",
            "invalid-context",
        )
        cases_per_category = int(scenario.get("cases_per_category", 6))
        require(
            1 <= cases_per_category <= 6,
            "serve_negative_api_matrix.cases_per_category must be 1..6",
        )
        global_payload_owners: dict[str, str] = {}
        category_payload_owners: dict[str, dict[str, str]] = {}
        global_contract_owners: dict[str, str] = {}
        category_contract_owners: dict[str, dict[str, str]] = {}
        category_counts = {category: 0 for category in categories}
        results: list[dict[str, Any]] = []
        deterministic_sampling_plan(scenario)

        for category in categories:
            for ordinal in range(cases_per_category):
                marker = f"c16-{category}-{ordinal:03d}"
                payload = chat_payload(
                    self.model,
                    [
                        {
                            "role": "user",
                            "content": f"C16 boundary case {marker}; reply with the marker.",
                        }
                    ],
                    max_tokens=int(scenario["max_tokens"]),
                    temperature=0.0,
                )
                apply_deterministic_sampling(payload, scenario)
                payload["metadata"] = {
                    "ferrum_regression_case": marker,
                    "ferrum_regression_category": category,
                }
                variant, expected_param, expected_message, failure_contract = negative_api_case(
                    category,
                    ordinal,
                    payload,
                    self.model,
                )

                owner = f"{category}/{ordinal:03d}"
                request_sha256 = register_unique_payload(
                    payload,
                    owner=owner,
                    category=category,
                    global_owners=global_payload_owners,
                    category_owners=category_payload_owners,
                )
                failure_contract_sha256 = register_unique_payload(
                    failure_contract,
                    owner=owner,
                    category=category,
                    global_owners=global_contract_owners,
                    category_owners=category_contract_owners,
                )
                case_out = out / category / f"{ordinal:03d}"
                case_out.mkdir(parents=True, exist_ok=True)
                write_json(case_out / "request.json", payload)

                status, body = post_json(self.require_base_url(), payload, timeout=self.timeout)
                error = parse_openai_error(owner, status, body)
                require(
                    error["param"] == expected_param,
                    f"{owner}: error param {error['param']!r} != {expected_param!r}",
                )
                require(
                    expected_message in error["message"],
                    f"{owner}: error message does not identify {variant!r}: {error['message']!r}",
                )
                response = json.loads(body)
                require(isinstance(response, dict), f"{owner}: response must be an object")
                write_json(case_out / "response.json", response)
                response_sha256 = json_fingerprint(response)
                case_result = with_canonical_sha256(
                    {
                        "schema_version": 1,
                        "status": "pass",
                        "case_id": marker,
                        "category": category,
                        "variant": variant,
                        "preset": "P_DETERMINISTIC",
                        "ordinal": ordinal,
                        "http_status": status,
                        "expected_error_param": expected_param,
                        "expected_error_message_substring": expected_message,
                        "observed_error": error,
                        "request_canonical_sha256": request_sha256,
                        "failure_contract_canonical_sha256": failure_contract_sha256,
                        "response_canonical_sha256": response_sha256,
                        "request_artifact": str(case_out / "request.json"),
                        "response_artifact": str(case_out / "response.json"),
                    }
                )
                write_json(case_out / "result.json", case_result)
                category_counts[category] += 1
                results.append(case_result)

        expected_count = len(categories) * cases_per_category
        require(len(results) == expected_count, "C16 matrix case count mismatch")
        require(
            len(global_payload_owners) == expected_count,
            "C16 matrix global payloads are not unique",
        )
        require(
            all(len(category_payload_owners[category]) == cases_per_category for category in categories),
            "C16 matrix category payloads are not unique",
        )
        require(
            len(global_contract_owners) == expected_count,
            "C16 matrix failure contracts are not unique",
        )
        require(
            all(len(category_contract_owners[category]) == cases_per_category for category in categories),
            "C16 matrix category failure contracts are not unique",
        )
        return {
            "status": "pass",
            "case_count": len(results),
            "passed_count": len(results),
            "cases_per_category": cases_per_category,
            "preset": "P_DETERMINISTIC",
            "category_counts": category_counts,
            "unique_payload_count": len(global_payload_owners),
            "unique_payload_count_by_category": {
                category: len(category_payload_owners[category]) for category in categories
            },
            "unique_failure_contract_count": len(global_contract_owners),
            "unique_failure_contract_count_by_category": {
                category: len(category_contract_owners[category]) for category in categories
            },
            "cases": results,
        }

    def serve_text_only_modality_matrix(
        self, scenario: dict[str, Any], out: Path
    ) -> dict[str, Any]:
        categories = (
            "image-url",
            "data-url",
            "video-url",
            "mixed-text-media",
            "text-array",
        )
        cases_per_category = int(scenario.get("cases_per_category", 10))
        require(
            1 <= cases_per_category <= 10,
            "serve_text_only_modality_matrix.cases_per_category must be 1..10",
        )
        deterministic_sampling_plan(scenario)

        models_out = out / "models"
        models_out.mkdir(parents=True, exist_ok=True)
        models_request = {
            "method": "GET",
            "path": "/v1/models",
            "target_model": self.model,
        }
        write_json(models_out / "request.json", models_request)
        models_status, models_body = get_url(
            self.require_base_url().rstrip("/") + "/v1/models",
            timeout=self.timeout,
        )
        models_response, model_entry = parse_text_only_model_declaration(
            "serve_text_only_modality_matrix.models",
            models_status,
            models_body,
            self.model,
        )
        write_json(models_out / "response.json", models_response)
        models_result = with_canonical_sha256(
            {
                "schema_version": 1,
                "status": "pass",
                "http_status": models_status,
                "model": self.model,
                "declared_modalities": model_entry["modalities"],
                "request_canonical_sha256": json_fingerprint(models_request),
                "response_canonical_sha256": json_fingerprint(models_response),
                "request_artifact": str(models_out / "request.json"),
                "response_artifact": str(models_out / "response.json"),
            }
        )
        write_json(models_out / "result.json", models_result)

        global_payload_owners: dict[str, str] = {}
        category_payload_owners: dict[str, dict[str, str]] = {}
        category_counts = {category: 0 for category in categories}
        results: list[dict[str, Any]] = []
        rejected_media_count = 0
        text_array_success_count = 0

        for category in categories:
            for ordinal in range(cases_per_category):
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
                            "video_url": {
                                "url": f"{C20_REMOTE_MEDIA_URL}?ferrum_case={marker}"
                            },
                        }
                    ]
                else:
                    content = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"{C20_REMOTE_MEDIA_URL}?ferrum_case={marker}"
                            },
                        }
                    ]
                    if category == "mixed-text-media":
                        content.insert(0, {"type": "text", "text": prompt})

                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": content}],
                }
                case_preset: str | None = None
                if category == "text-array":
                    apply_deterministic_sampling(payload, scenario)
                    case_preset = "P_DETERMINISTIC"
                payload["metadata"] = {
                    "ferrum_regression_case": marker,
                    "ferrum_regression_category": category,
                    "ferrum_expected_marker": marker,
                }

                owner = f"{category}/{ordinal:03d}"
                request_sha256 = register_unique_payload(
                    payload,
                    owner=owner,
                    category=category,
                    global_owners=global_payload_owners,
                    category_owners=category_payload_owners,
                )
                case_out = out / category / f"{ordinal:03d}"
                case_out.mkdir(parents=True, exist_ok=True)
                write_json(case_out / "request.json", payload)
                status, body = post_json(self.require_base_url(), payload, timeout=self.timeout)

                if category == "text-array":
                    response = parse_json_response(owner, status, body)
                    require(
                        response.get("object") == "chat.completion",
                        f"{owner}: response object must be chat.completion",
                    )
                    text = message_text(response)
                    require(
                        text.strip() == marker,
                        f"{owner}: text-array response must equal {marker!r}: {text[:500]}",
                    )
                    require(finish_reason(response) != "length", f"{owner}: text-array length finish")
                    observed: dict[str, Any] = {
                        "finish_reason": finish_reason(response),
                        "content": text,
                    }
                    text_array_success_count += 1
                else:
                    error = parse_openai_error(owner, status, body)
                    require(
                        error["param"] is None,
                        f"{owner}: media boundary error param must be null: {error}",
                    )
                    require(
                        "unsupported message content part type" in error["message"],
                        f"{owner}: media must fail during request deserialization: {error}",
                    )
                    response = json.loads(body)
                    require(isinstance(response, dict), f"{owner}: response must be an object")
                    observed = {
                        "error": error,
                        "rejection_stage": "request-deserialization",
                    }
                    rejected_media_count += 1

                write_json(case_out / "response.json", response)
                response_sha256 = json_fingerprint(response)
                case_result = with_canonical_sha256(
                    {
                        "schema_version": 1,
                        "status": "pass",
                        "case_id": marker,
                        "category": category,
                        "preset": case_preset,
                        "ordinal": ordinal,
                        "http_status": status,
                        "declared_modalities": model_entry["modalities"],
                        "observed": observed,
                        "request_canonical_sha256": request_sha256,
                        "response_canonical_sha256": response_sha256,
                        "request_artifact": str(case_out / "request.json"),
                        "response_artifact": str(case_out / "response.json"),
                    }
                )
                write_json(case_out / "result.json", case_result)
                category_counts[category] += 1
                results.append(case_result)

        expected_count = len(categories) * cases_per_category
        require(len(results) == expected_count, "C20 matrix case count mismatch")
        require(
            len(global_payload_owners) == expected_count,
            "C20 matrix global payloads are not unique",
        )
        require(
            all(len(category_payload_owners[category]) == cases_per_category for category in categories),
            "C20 matrix category payloads are not unique",
        )
        require(
            rejected_media_count == 4 * cases_per_category,
            "C20 media rejection count mismatch",
        )
        require(
            text_array_success_count == cases_per_category,
            "C20 text-array success count mismatch",
        )
        return {
            "status": "pass",
            "case_count": len(results),
            "passed_count": len(results),
            "cases_per_category": cases_per_category,
            "preset_counts": {
                "P_DETERMINISTIC": text_array_success_count,
                "unpreset": rejected_media_count,
            },
            "category_counts": category_counts,
            "unique_payload_count": len(global_payload_owners),
            "unique_payload_count_by_category": {
                category: len(category_payload_owners[category]) for category in categories
            },
            "rejected_media_count": rejected_media_count,
            "text_array_success_count": text_array_success_count,
            "declared_modalities": model_entry["modalities"],
            "models_artifact": str(models_out / "result.json"),
            "cases": results,
        }

    def serve_context_limit(self, scenario: dict[str, Any], out: Path) -> dict[str, Any]:
        expected_status = int(scenario.get("expected_status", 400))
        payload = chat_payload(
            self.model,
            [{"role": "user", "content": str(scenario.get("prompt") or "Say one word.")}],
            max_tokens=int(scenario.get("max_tokens", 4096)),
            temperature=float(scenario.get("temperature", 0)),
        )
        status, body = post_json(self.require_base_url(), payload, timeout=self.timeout)
        (out / "response.json").write_text(body, errors="replace")
        data = parse_json_response("serve_context_limit", status, body, expected_status)
        if expected_status == 400:
            error = data.get("error")
            require(isinstance(error, dict), "context limit response missing error object")
            require(error.get("type") == "invalid_request_error", f"bad context error: {data}")
        return {"status": "pass", "http_status": status}

    def serve_concurrency_quality(self, scenario: dict[str, Any], out: Path) -> dict[str, Any]:
        cells = scenario.get("concurrency_cells") or [1, 4, 16, 32]
        if not isinstance(cells, list) or not cells:
            raise ScenarioError("serve_concurrency_quality.concurrency_cells must be a non-empty list")
        result = run_concurrency_quality_regression(
            self.require_base_url(),
            self.model,
            out,
            [int(cell) for cell in cells],
            timeout=self.timeout,
        )
        return {"status": "pass", "cells": result.get("cells", [])}

    def serve_tool_schema_priority(
        self, scenario: dict[str, Any], out: Path
    ) -> dict[str, Any]:
        case_count = int(scenario.get("case_count", 4))
        require(1 <= case_count <= 20, "serve_tool_schema_priority.case_count must be 1..20")
        max_tokens = int(scenario.get("max_tokens", 256))
        prefix = str(scenario.get("marker_prefix") or "c21-priority")
        results: list[dict[str, Any]] = []

        for ordinal in range(case_count):
            marker = f"{prefix}-{ordinal:03d}"
            for stream in (False, True):
                mode = "stream" if stream else "sync"
                case_out = out / f"{ordinal:03d}-{mode}"
                case_out.mkdir(parents=True, exist_ok=True)
                payload = chat_payload(
                    self.model,
                    [
                        {
                            "role": "user",
                            "content": (
                                f"Call echo_value for {marker}; tool choice has priority "
                                "over the simultaneous strict response format."
                            ),
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0,
                )
                payload["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": "echo_value",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string", "const": marker}
                                },
                                "required": ["value"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ]
                payload["tool_choice"] = "required"
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"conflict_{ordinal:03d}",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "result": {"type": "string", "const": marker}
                            },
                            "required": ["result"],
                            "additionalProperties": False,
                        },
                    },
                }
                thinking = scenario.get("enable_thinking")
                if isinstance(thinking, bool):
                    payload["chat_template_kwargs"] = {"enable_thinking": thinking}
                payload["stream"] = stream
                if stream:
                    payload["stream_options"] = {"include_usage": True}
                write_json(case_out / "request.json", payload)

                if stream:
                    status, body, event_times = request_sse(
                        self.require_base_url(), payload, timeout=self.timeout
                    )
                    (case_out / "response.sse").write_text(body, errors="replace")
                    require(status == 200, f"{marker} stream expected HTTP 200, got {status}")
                    parsed = parse_sse(body)
                    validated = validate_stream_tool_schema_priority(parsed, marker)
                    result = {
                        "marker": marker,
                        "mode": mode,
                        "http_status": status,
                        "event_count": len(event_times),
                        "protocol": parsed,
                        "tool_call": validated,
                    }
                else:
                    status, body = post_json(
                        self.require_base_url(), payload, timeout=self.timeout
                    )
                    (case_out / "response.json").write_text(body, errors="replace")
                    data = parse_json_response(f"{marker} sync", status, body)
                    result = {
                        "marker": marker,
                        "mode": mode,
                        "http_status": status,
                        "tool_call": validate_sync_tool_schema_priority(data, marker),
                    }
                write_json(case_out / "result.json", {"status": "pass", **result})
                results.append(result)

        return {"status": "pass", "case_count": len(results), "cases": results}

    def serve_tool_call(self, out: Path) -> dict[str, Any]:
        result = run_tool_call_regression(self.require_base_url(), self.model, out)
        return {"status": "pass", "checks": result.get("checks", {})}

    def serve_python_openai_sdk(self, scenario: dict[str, Any], out: Path) -> dict[str, Any]:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise OptionalSkip(f"openai SDK unavailable: {exc}") from exc
        client = OpenAI(base_url=self.require_base_url().rstrip("/") + "/v1", api_key="not-needed")
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": str(scenario.get("prompt") or "Say hi.")}],
            temperature=0,
            max_tokens=int(scenario.get("max_tokens", 64)),
        )
        data = response.model_dump()
        write_json(out / "openai_sdk_response.json", data)
        text = data["choices"][0]["message"].get("content") or ""
        require(str(text).strip(), "OpenAI SDK response content is empty")
        assert_no_bad_text("serve_python_openai_sdk", str(text))
        return {"status": "pass", "content": str(text)[:1000]}

    def run_multiturn(self, scenario: dict[str, Any], out: Path) -> dict[str, Any]:
        prompts = scenario.get("prompts")
        if not isinstance(prompts, list) or len(prompts) < 2:
            secret = str(scenario.get("secret") or "ferrum-blue")
            prompts = [f"Code: {secret}.", "Reply with only the code."]
        input_text = "\n".join(str(prompt) for prompt in prompts) + "\n/bye\n"
        cmd = [
            str(self.ferrum_bin),
            "run",
            "--backend",
            str(scenario.get("backend") or self.backend),
            "--max-tokens",
            str(scenario.get("max_tokens", 192)),
            "--temperature",
            str(scenario.get("temperature", 0)),
            "--output-format",
            "jsonl",
            "--effective-config-json",
            str(out / "effective_config.json"),
            "--decision-trace-jsonl",
            str(out / "decision_trace.jsonl"),
        ]
        if self.observability_enabled():
            root = out / "observability"
            cmd.extend(self.observability_args(root))
            self.run_observability_roots.append(root)
        cmd.append(self.model)
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(scenario.get("timeout_sec", self.timeout * 3)),
            check=False,
            env={**os.environ, "NO_COLOR": "1"},
        )
        (out / "stdout.jsonl").write_text(proc.stdout, errors="replace")
        (out / "stderr.log").write_text(proc.stderr, errors="replace")
        require(proc.returncode == 0, f"ferrum run failed rc={proc.returncode}: {proc.stderr[-1000:]}")
        assert_no_bad_text("run_multiturn.stdout", proc.stdout)
        assert_no_bad_text("run_multiturn.stderr", proc.stderr)
        events = parse_json_events(proc.stdout)
        assistants = assistant_events(events)
        min_turns = int(scenario.get("min_assistant_turns", 2))
        require(len(assistants) >= min_turns, f"assistant turns {len(assistants)} < {min_turns}")
        length_finishes = sum(1 for event in assistants if event.get("finish_reason") == "length")
        require(length_finishes == 0, f"run_multiturn length_finishes={length_finishes}")
        expected = scenario.get("expected_recall") or scenario.get("secret")
        if expected:
            last = str(assistants[-1].get("content") or "")
            require(
                exact_text_matches(last, expected),
                f"run_multiturn did not exactly recall {expected!r}: {last[:500]}",
            )
        isolated = [event for event in assistants if str(event.get("content") or "").strip() == "</think>"]
        require(not isolated, "run_multiturn emitted isolated </think>")
        return {"status": "pass", "assistant_turns": len(assistants), "length_finishes": 0}

    def run_first_token_ux(self, scenario: dict[str, Any], out: Path) -> dict[str, Any]:
        if os.name != "posix":
            raise OptionalSkip("run_first_token_ux requires a POSIX pty")
        prompt = str(scenario.get("prompt") or "Say hello briefly.")
        hint_timeout_ms = int(scenario.get("hint_timeout_ms", 1000))
        timeout_sec = int(scenario.get("timeout_sec", self.timeout * 3))
        cmd = [
            str(self.ferrum_bin),
            "run",
            "--backend",
            str(scenario.get("backend") or self.backend),
            "--max-tokens",
            str(scenario.get("max_tokens", 64)),
            "--temperature",
            str(scenario.get("temperature", 0)),
            self.model,
        ]
        master, slave = pty.openpty()
        proc = subprocess.Popen(
            cmd,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env={**os.environ, "NO_COLOR": "1"},
            close_fds=True,
        )
        os.close(slave)
        raw = bytearray()
        hint_ms: float | None = None
        prompt_sent_at: float | None = None
        deadline = time.time() + timeout_sec
        try:
            while time.time() < deadline:
                ready, _, _ = select.select([master], [], [], 0.05)
                if ready:
                    chunk = os.read(master, 4096)
                    raw.extend(chunk)
                text = raw.decode("utf-8", "replace")
                if prompt_sent_at is None and (">>>" in text or "Ready." in text):
                    os.write(master, (prompt + "\n").encode())
                    prompt_sent_at = time.time()
                if prompt_sent_at is not None:
                    after_prompt = raw.decode("utf-8", "replace")
                    if hint_ms is None and "Working (" in strip_ansi(after_prompt):
                        hint_ms = (time.time() - prompt_sent_at) * 1000.0
                    if re.search(r"\[\d+ tokens,", after_prompt):
                        os.write(master, b"/bye\n")
                        break
            time.sleep(0.2)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            os.close(master)
        transcript = raw.decode("utf-8", "replace")
        (out / "transcript.txt").write_text(transcript, errors="replace")
        clean = strip_ansi(transcript)
        assert_no_bad_text("run_first_token_ux", clean)
        require("waiting for first token" not in clean.lower(), "forbidden first-token hint text")
        require(hint_ms is not None, "no first-token progress hint observed")
        require(hint_ms <= hint_timeout_ms, f"first-token hint took {hint_ms:.1f} ms")
        hint_lines = [line.strip() for line in clean.replace("\r", "\n").splitlines() if "Working (" in line]
        require(len(hint_lines) >= 1, "missing Working progress line")
        visible = re.sub(r"\([^)]*\)", "()", hint_lines[0])
        require(len(visible) <= int(scenario.get("max_hint_len", 32)), f"hint line too long: {visible!r}")
        return {"status": "pass", "hint_ms": hint_ms, "hint_line": visible}


class OptionalSkip(Exception):
    pass


def is_optional_skip(exc: Exception) -> bool:
    return isinstance(exc, OptionalSkip)


class MockOpenAIHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    active_lock = threading.Lock()
    active_requests = 0

    @classmethod
    def change_active_requests(cls, delta: int) -> int:
        with cls.active_lock:
            cls.active_requests += delta
            return cls.active_requests

    @classmethod
    def active_request_count(cls) -> int:
        with cls.active_lock:
            return cls.active_requests

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            active = self.active_request_count()
            self.send_json(
                200,
                {
                    "status": "ok",
                    "admission": {
                        "effective_max_concurrent": 1,
                        "queue_depth": 0,
                        "active_prefill": 0,
                        "active_decode": active,
                        "current_batch_size": active,
                    },
                },
            )
            return
        if self.path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "mock-model",
                            "object": "model",
                            "modalities": ["text"],
                        }
                    ],
                },
            )
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body)
        messages = payload.get("messages", [])
        media_part = any(
            isinstance(message, dict)
            and isinstance(message.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") in {"image_url", "video_url"}
                for part in message["content"]
            )
            for message in messages
        )
        if payload.get("model") != "mock-model":
            self.send_openai_error("unknown model", "model")
            return
        if payload.get("tool_choice") == "required" and not payload.get("tools"):
            self.send_openai_error(
                "tool_choice=required requires at least one function tool",
                "tool_choice",
            )
            return
        if isinstance(payload.get("tool_choice"), str) and payload["tool_choice"] not in {
            "auto",
            "none",
            "required",
        }:
            self.send_openai_error("unsupported tool_choice mode", "tool_choice")
            return
        tools = payload.get("tools")
        if isinstance(tools, list) and any(
            isinstance(tool, dict) and tool.get("type") != "function" for tool in tools
        ):
            self.send_openai_error("only function tools are supported", "tools")
            return
        tool_choice = payload.get("tool_choice")
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") != "function":
                self.send_openai_error("only function tool_choice is supported", "tool_choice")
                return
            selected_name = (
                tool_choice.get("function", {}).get("name")
                if isinstance(tool_choice.get("function"), dict)
                else None
            )
            declared_names = {
                tool.get("function", {}).get("name")
                for tool in tools or []
                if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
            }
            if selected_name not in declared_names:
                self.send_openai_error(
                    "tool_choice selects a function that is not declared in tools",
                    "tool_choice",
                )
                return
        function_call = payload.get("function_call")
        if isinstance(function_call, dict):
            selected_name = function_call.get("name")
            declared_names = {
                function.get("name")
                for function in payload.get("functions") or []
                if isinstance(function, dict)
            }
            if selected_name not in declared_names:
                self.send_openai_error(
                    "function_call selects a function that is not declared in functions",
                    "function_call",
                )
                return
        response_format = payload.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") not in {
            None,
            "text",
            "json_object",
            "json_schema",
        }:
            self.send_openai_error("unsupported response_format.type", "response_format.type")
            return
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"
            and (
                not isinstance(response_format.get("json_schema"), dict)
                or response_format["json_schema"].get("schema") is None
            )
        ):
            self.send_openai_error(
                "response_format.json_schema.schema is required",
                "response_format.json_schema",
            )
            return
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            schema = response_format["json_schema"].get("schema")
            if isinstance(schema, dict) and (
                schema.get("type") == "definitely-not-a-json-type"
                or ("required" in schema and not isinstance(schema["required"], list))
            ):
                self.send_openai_error(
                    "unsupported strict json_schema",
                    "response_format.json_schema",
                )
                return
        if payload.get("stream_options") is not None and payload.get("stream") is not True:
            self.send_openai_error(
                "stream_options is only valid when stream=true",
                "stream_options",
            )
            return
        stream_options = payload.get("stream_options")
        if payload.get("stream") is True and not isinstance(stream_options, (dict, type(None))):
            self.send_openai_error("invalid chat completions request: invalid stream_options type", None)
            return
        if isinstance(stream_options, dict):
            unknown_stream_options = set(stream_options) - {"include_usage"}
            if unknown_stream_options:
                self.send_openai_error(
                    "invalid chat completions request: unknown field in stream_options",
                    None,
                )
                return
            include_usage = stream_options.get("include_usage")
            if include_usage is not None and not isinstance(include_usage, bool):
                self.send_openai_error(
                    "invalid chat completions request: invalid stream_options.include_usage type",
                    None,
                )
                return
        effective_max_tokens = payload.get("max_completion_tokens", payload.get("max_tokens", 0))
        if int(effective_max_tokens or 0) >= 1_000_000_000:
            self.send_openai_error("This model context is limited to 4096 tokens", None)
            return
        if media_part:
            self.send_openai_error(
                "invalid chat completions request: unsupported message content part type",
                None,
            )
            return
        prompt = " ".join(str(msg.get("content") or "") for msg in messages)
        last_user = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user = str(msg.get("content") or "")
                break
        exact_object = self.exact_json_object(last_user)
        echo_marker = self.echo_value_marker(payload)
        if payload.get("stream"):
            metadata = payload.get("metadata")
            if isinstance(metadata, dict) and metadata.get("ferrum_disconnect_probe") is True:
                self.send_disconnect_stream()
            elif echo_marker is not None:
                self.send_stream_tool_call(echo_marker)
            elif response_format and exact_object is not None:
                self.send_stream(
                    json.dumps(exact_object, ensure_ascii=False, separators=(",", ":"))
                )
            else:
                self.send_stream()
            return
        if int(payload.get("max_tokens") or 0) >= 4096:
            self.send_json(
                400,
                {"error": {"type": "invalid_request_error", "message": "context limit"}},
            )
            return
        marker = re.search(r"\b(ferrum\d{2}\d{2})\b", prompt)
        square = re.search(r"(S\d{4})", prompt)
        response_format_type = (
            response_format.get("type") if isinstance(response_format, dict) else None
        )
        if echo_marker is not None:
            self.send_tool_call(
                "echo_value",
                json.dumps({"value": echo_marker}, separators=(",", ":")),
            )
            return
        if payload.get("tools") and marker and square:
            self.send_tool_call(
                "capture_quality_marker",
                json.dumps(
                    {"marker": marker.group(1), "checksum": square.group(1)},
                    separators=(",", ":"),
                ),
            )
            return
        if payload.get("tools") and any(msg.get("role") == "tool" for msg in payload.get("messages", [])):
            self.send_chat("北京 22 celsius 晴")
            return
        if payload.get("tools"):
            self.send_tool_call()
            return
        if marker and square:
            self.send_chat(
                json.dumps(
                    {"marker": marker.group(1), "checksum": square.group(1)},
                    separators=(",", ":"),
                )
            )
        elif response_format_type == "json_object":
            self.send_chat("{}")
        elif response_format and exact_object is not None:
            self.send_chat(
                json.dumps(exact_object, ensure_ascii=False, separators=(",", ":"))
            )
        elif response_format:
            self.send_chat('{"answer":"scenario-ok"}')
        elif "remembered code" in last_user.lower() or "secret code" in last_user.lower():
            self.send_chat("ferrum-blue ferrum-loop-blue")
        elif "remember code" in last_user.lower() or "secret" in last_user.lower():
            self.send_chat("OK")
        elif "paris" in last_user.lower():
            self.send_chat("Paris")
        elif "rust" in last_user.lower():
            self.send_chat("Rust")
        elif "code" in prompt.lower():
            self.send_chat("ferrum-blue ferrum-loop-blue")
        elif isinstance(payload.get("metadata"), dict) and isinstance(
            payload["metadata"].get("ferrum_expected_marker"), str
        ):
            self.send_chat(str(payload["metadata"]["ferrum_expected_marker"]))
        else:
            self.send_chat("scenario-ok")

    @staticmethod
    def echo_value_marker(payload: dict[str, Any]) -> str | None:
        tools = payload.get("tools")
        if not isinstance(tools, list):
            return None
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if not isinstance(function, dict) or function.get("name") != "echo_value":
                continue
            parameters = function.get("parameters")
            if not isinstance(parameters, dict):
                return None
            properties = parameters.get("properties")
            if not isinstance(properties, dict):
                return None
            value = properties.get("value")
            if not isinstance(value, dict) or not isinstance(value.get("const"), str):
                return None
            return str(value["const"])
        return None

    @staticmethod
    def exact_json_object(prompt: str) -> dict[str, Any] | None:
        marker = "EXACT_JSON:"
        if marker not in prompt:
            return None
        try:
            value = json.loads(prompt.split(marker, 1)[1])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def send_json(self, status: int, data: dict[str, Any]) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_openai_error(self, message: str, param: str | None) -> None:
        self.send_json(
            400,
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": param,
                    "code": None,
                }
            },
        )

    def send_chat(self, text: str) -> None:
        self.send_json(
            200,
            {
                "id": "chatcmpl_mock",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def send_tool_call(
        self,
        name: str = "get_weather",
        arguments: str = '{"city":"北京","unit":"celsius"}',
    ) -> None:
        self.send_json(
            200,
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_mock",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    def send_stream(self, text: str = "scenario") -> None:
        lines = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}
                ]
            },
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ]
        raw = (
            "".join(f"data: {json.dumps(line, ensure_ascii=False)}\n\n" for line in lines)
            + "data: [DONE]\n\n"
        )
        body = raw.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_disconnect_stream(self) -> None:
        first = (
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "1\n"},
                            "finish_reason": None,
                        }
                    ]
                }
            )
            + "\n\n"
        ).encode("utf-8")
        remainder = (
            "data: "
            + json.dumps(
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )
            + "\n\ndata: [DONE]\n\n"
        ).encode("utf-8")
        self.change_active_requests(1)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(first) + len(remainder)))
            self.end_headers()
            self.wfile.write(first)
            self.wfile.flush()
            time.sleep(0.2)
            self.wfile.write(remainder)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.change_active_requests(-1)

    def send_stream_tool_call(self, marker: str) -> None:
        arguments = json.dumps({"value": marker}, separators=(",", ":"))
        lines = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_mock",
                                    "type": "function",
                                    "function": {
                                        "name": "echo_value",
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            {"choices": [], "usage": {"completion_tokens": 1}},
        ]
        raw = "".join(f"data: {json.dumps(line)}\n\n" for line in lines) + "data: [DONE]\n\n"
        body = raw.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_selftest_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=repo_root(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def self_test() -> int:
    if not exact_text_matches("  ferrum-blue\n", "ferrum-blue"):
        raise AssertionError("exact text matcher rejected surrounding whitespace")
    if exact_text_matches(
        'I cannot recall the code "ferrum-blue" from a prior turn.',
        "ferrum-blue",
    ):
        raise AssertionError("exact text matcher accepted explanatory false positive")
    sampling = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "seed": 9271,
        "stop": [],
    }
    deterministic_sampling = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": 9271,
        "stop": [],
    }
    full_c14_contract = response_format_matrix_contract(
        [
            {
                "name": "c14-no-thinking",
                "type": "serve_response_format_matrix",
                "format": "json_schema",
                "case_count": 50,
                "enable_thinking": False,
                "preset": "P_NO_THINKING",
                "sampling": sampling,
            },
            {
                "name": "c14-thinking",
                "type": "serve_response_format_matrix",
                "format": "json_schema",
                "case_count": 20,
                "enable_thinking": True,
                "preset": "P_THINKING",
                "sampling": {**sampling, "temperature": 1.0},
            },
        ]
    )
    if full_c14_contract["case_counts"] != {"json_schema": 70, "json_object": 0}:
        raise AssertionError(full_c14_contract)
    if full_c14_contract["unique_json_schema_count"] != 70:
        raise AssertionError(full_c14_contract)
    if full_c14_contract["unique_expected_object_counts"]["json_schema"] != 70:
        raise AssertionError(full_c14_contract)
    if full_c14_contract["json_schema_category_counts"] != {
        "required": 18,
        "type": 18,
        "additionalProperties": 17,
        "enum": 17,
    }:
        raise AssertionError(full_c14_contract)
    try:
        response_format_matrix_contract(
            [
                {
                    "name": "duplicate-matrix",
                    "type": "serve_response_format_matrix",
                    "format": "json_schema",
                    "case_count": 1,
                    "enable_thinking": False,
                    "preset": "P_NO_THINKING",
                    "sampling": sampling,
                },
                {
                    "name": "duplicate-matrix",
                    "type": "serve_response_format_matrix",
                    "format": "json_schema",
                    "case_count": 1,
                    "enable_thinking": False,
                    "preset": "P_NO_THINKING",
                    "sampling": sampling,
                },
            ]
        )
    except ScenarioError as exc:
        if "duplicate response-format matrix scenario name" not in str(exc):
            raise AssertionError(str(exc)) from exc
    else:
        raise AssertionError("duplicate response-format matrix scenario unexpectedly passed")

    duplicate_global: dict[str, str] = {}
    duplicate_categories: dict[str, dict[str, str]] = {}
    register_unique_payload(
        {"model": "mock-model", "messages": []},
        owner="first",
        category="duplicate",
        global_owners=duplicate_global,
        category_owners=duplicate_categories,
    )
    try:
        register_unique_payload(
            {"model": "mock-model", "messages": []},
            owner="mutated-duplicate",
            category="duplicate",
            global_owners=duplicate_global,
            category_owners=duplicate_categories,
        )
    except ScenarioError as exc:
        if "duplicate matrix payload" not in str(exc):
            raise AssertionError(str(exc)) from exc
    else:
        raise AssertionError("duplicate negative/modality payload unexpectedly passed")

    try:
        parse_openai_error(
            "missing-param-mutation",
            400,
            json.dumps(
                {
                    "error": {
                        "message": "mutated error",
                        "type": "invalid_request_error",
                    }
                }
            ),
        )
    except ScenarioError as exc:
        if "missing param" not in str(exc):
            raise AssertionError(str(exc)) from exc
    else:
        raise AssertionError("OpenAI error without param unexpectedly passed")

    try:
        parse_text_only_model_declaration(
            "bad-modalities-mutation",
            200,
            json.dumps(
                {
                    "object": "list",
                    "data": [{"id": "mock-model", "modalities": ["text", "image"]}],
                }
            ),
            "mock-model",
        )
    except ScenarioError as exc:
        if "modalities exactly" not in str(exc):
            raise AssertionError(str(exc)) from exc
    else:
        raise AssertionError("non-text-only model declaration unexpectedly passed")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="ferrum-scenario-selftest-") as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            write_json(
                manifest,
                {
                    "schema_version": 1,
                    "model": "mock-model",
                    "backend": "cpu",
                    "server": {"mode": "external", "base_url": f"http://127.0.0.1:{port}"},
                    "scenarios": [
                        {"name": "chat", "type": "serve_chat", "expected_contains": ["scenario-ok"]},
                        {"name": "multiturn", "type": "serve_multiturn_recall", "secret": "ferrum-blue"},
                        {"name": "loop", "type": "serve_stateful_loop"},
                        {"name": "stream", "type": "serve_stream"},
                        {
                            "name": "stream-equivalence-unicode",
                            "type": "serve_stream_equivalence_unicode",
                            "case_count": 3,
                            "enable_thinking": False,
                            "preset": "P_DETERMINISTIC",
                            "max_tokens": 1024,
                            "sampling": dict(deterministic_sampling),
                        },
                        {
                            "name": "disconnect-release",
                            "type": "serve_disconnect_release",
                            "expected_effective_max_concurrent": 1,
                            "max_tokens": 64,
                            "release_timeout_sec": 5.0,
                            "poll_interval_sec": 0.02,
                            "require_scheduler_trace": False,
                        },
                        {
                            "name": "structured",
                            "type": "serve_structured_output",
                            "expected_object": {"answer": "scenario-ok"},
                        },
                        {
                            "name": "strict-matrix",
                            "type": "serve_response_format_matrix",
                            "format": "json_schema",
                            "case_count": 4,
                            "enable_thinking": False,
                            "preset": "P_NO_THINKING",
                            "sampling": {
                                "temperature": 0.7,
                                "top_p": 0.8,
                                "top_k": 20,
                                "min_p": 0.0,
                                "presence_penalty": 1.5,
                                "repetition_penalty": 1.0,
                                "seed": 9271,
                                "stop": [],
                            },
                        },
                        {
                            "name": "object-matrix",
                            "type": "serve_response_format_matrix",
                            "format": "json_object",
                            "case_count": 2,
                            "enable_thinking": True,
                            "preset": "P_THINKING",
                            "sampling": {
                                "temperature": 1.0,
                                "top_p": 0.95,
                                "top_k": 20,
                                "min_p": 0.0,
                                "presence_penalty": 1.5,
                                "repetition_penalty": 1.0,
                                "seed": 9271,
                                "stop": [],
                            },
                        },
                        {
                            "name": "negative-api-matrix",
                            "type": "serve_negative_api_matrix",
                            "cases_per_category": 6,
                            "enable_thinking": False,
                            "preset": "P_DETERMINISTIC",
                            "max_tokens": 1024,
                            "sampling": dict(deterministic_sampling),
                        },
                        {
                            "name": "text-only-modality-matrix",
                            "type": "serve_text_only_modality_matrix",
                            "cases_per_category": 1,
                            "enable_thinking": False,
                            "preset": "P_DETERMINISTIC",
                            "max_tokens": 1024,
                            "sampling": dict(deterministic_sampling),
                        },
                        {"name": "context", "type": "serve_context_limit"},
                        {
                            "name": "concurrency",
                            "type": "serve_concurrency_quality",
                            "concurrency_cells": [1, 4],
                        },
                        {
                            "name": "tool-schema-priority",
                            "type": "serve_tool_schema_priority",
                            "case_count": 1,
                        },
                        {"name": "tool", "type": "serve_tool_call"},
                    ],
                },
            )
            out = root / "out"
            proc = run_selftest_command(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(out),
                ]
            )
            if proc.returncode != 0:
                diagnostic = ""
                if (out / "summary.json").is_file():
                    diagnostic = (out / "summary.json").read_text(encoding="utf-8")
                raise AssertionError((proc.stderr or proc.stdout) + "\n" + diagnostic)
            if f"BACKEND REGRESSION SMOKE PASS: {out.resolve()}" not in proc.stdout:
                raise AssertionError(proc.stdout)
            summary = load_json_object(out / "summary.json")
            if summary.get("status") != "pass" or summary.get("scenario_count") != 15:
                raise AssertionError(summary)
            if (
                summary.get("manifest_scenario_count") != 15
                or summary.get("requested_scenarios") != []
                or len(summary.get("selected_scenarios", [])) != 15
            ):
                raise AssertionError(summary)
            receipt = load_json_object(out / "execution_receipt.json")
            receipt_sha = receipt.pop("canonical_sha256", None)
            if (
                receipt.get("mode") != "external"
                or json_fingerprint(receipt) != receipt_sha
                or summary.get("execution_receipt", {}).get("artifact_sha256")
                != file_sha256(out / "execution_receipt.json")
            ):
                raise AssertionError(receipt)
            tree = load_json_object(out / "artifact_tree.json")
            tree_sha = tree.pop("canonical_sha256", None)
            if json_fingerprint(tree) != tree_sha:
                raise AssertionError(tree)
            tree_entries = tree.get("files")
            if not isinstance(tree_entries, list):
                raise AssertionError(tree)
            actual_tree_paths = {
                path.relative_to(out).as_posix()
                for path in out.rglob("*")
                if path.is_file() and not path.is_symlink() and path.name != "artifact_tree.json"
            }
            recorded_tree_paths = {str(entry.get("path")) for entry in tree_entries}
            if actual_tree_paths != recorded_tree_paths:
                raise AssertionError({"actual": actual_tree_paths, "recorded": recorded_tree_paths})
            for entry in tree_entries:
                path = out / str(entry["path"])
                if entry.get("size") != path.stat().st_size or entry.get("sha256") != file_sha256(path):
                    raise AssertionError(entry)
            unmatched_out = root / "unmatched-only-out"
            unmatched = run_selftest_command(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(unmatched_out),
                    "--only",
                    "does-not-exist",
                ]
            )
            if unmatched.returncode == 0:
                raise AssertionError("unmatched --only unexpectedly passed")
            if "--only did not match manifest scenarios" not in unmatched.stderr:
                raise AssertionError(unmatched.stderr or unmatched.stdout)
            strict_result = load_json_object(out / "strict-matrix" / "result.json")
            if strict_result.get("category_counts") != {
                "required": 1,
                "type": 1,
                "additionalProperties": 1,
                "enum": 1,
            }:
                raise AssertionError(strict_result)
            object_result = load_json_object(out / "object-matrix" / "result.json")
            if object_result.get("case_count") != 2:
                raise AssertionError(object_result)
            if [case.get("object") for case in object_result.get("cases", [])] != [{}, {}]:
                raise AssertionError(object_result)
            equivalence_result = load_json_object(
                out / "stream-equivalence-unicode" / "result.json"
            )
            if (
                equivalence_result.get("case_count") != 3
                or equivalence_result.get("category_counts")
                != {"chinese": 1, "emoji": 1, "combining": 1}
                or equivalence_result.get("exact_content_matches") != 3
                or equivalence_result.get("exact_finish_matches") != 3
                or equivalence_result.get("exact_usage_matches") != 3
                or equivalence_result.get("multibyte_split_cases") != 3
            ):
                raise AssertionError(equivalence_result)
            disconnect_result = load_json_object(out / "disconnect-release" / "result.json")
            if (
                disconnect_result.get("same_capacity_followup") is not True
                or disconnect_result.get("effective_max_concurrent") != 1
                or disconnect_result.get("release_elapsed_sec", 6) > 5.0
                or disconnect_result.get("scheduler_tick_limit") != 2
            ):
                raise AssertionError(disconnect_result)
            matrix_contract = load_json_object(out / "response_format_matrix_contract.json")
            if matrix_contract.get("case_counts") != {"json_schema": 4, "json_object": 2}:
                raise AssertionError(matrix_contract)
            if matrix_contract.get("unique_json_schema_count") != 4:
                raise AssertionError(matrix_contract)
            if matrix_contract.get("unique_expected_object_counts") != {
                "json_schema": 4,
                "json_object": 2,
            }:
                raise AssertionError(matrix_contract)
            negative_result = load_json_object(out / "negative-api-matrix" / "result.json")
            if (
                negative_result.get("case_count") != 30
                or negative_result.get("unique_payload_count") != 30
                or negative_result.get("unique_failure_contract_count") != 30
            ):
                raise AssertionError(negative_result)
            if negative_result.get("category_counts") != {
                "invalid-tool": 6,
                "invalid-schema": 6,
                "invalid-stream-option": 6,
                "invalid-model": 6,
                "invalid-context": 6,
            }:
                raise AssertionError(negative_result)
            if negative_result.get("preset") != "P_DETERMINISTIC":
                raise AssertionError(negative_result)
            modality_result = load_json_object(
                out / "text-only-modality-matrix" / "result.json"
            )
            if modality_result.get("case_count") != 5 or modality_result.get(
                "unique_payload_count"
            ) != 5:
                raise AssertionError(modality_result)
            if modality_result.get("category_counts") != {
                "image-url": 1,
                "data-url": 1,
                "video-url": 1,
                "mixed-text-media": 1,
                "text-array": 1,
            }:
                raise AssertionError(modality_result)
            if (
                modality_result.get("rejected_media_count") != 4
                or modality_result.get("text_array_success_count") != 1
                or modality_result.get("declared_modalities") != ["text"]
                or modality_result.get("preset_counts")
                != {"P_DETERMINISTIC": 1, "unpreset": 4}
            ):
                raise AssertionError(modality_result)
            for scenario_name, scenario_result in (
                ("negative-api-matrix", negative_result),
                ("text-only-modality-matrix", modality_result),
            ):
                cases = scenario_result.get("cases")
                if not isinstance(cases, list) or not cases:
                    raise AssertionError(scenario_result)
                for case in cases:
                    if not isinstance(case, dict):
                        raise AssertionError(case)
                    request_path = Path(str(case.get("request_artifact") or ""))
                    response_path = Path(str(case.get("response_artifact") or ""))
                    case_path = (
                        out
                        / scenario_name
                        / str(case["category"])
                        / f"{int(case['ordinal']):03d}"
                        / "result.json"
                    )
                    if not request_path.is_file() or not response_path.is_file() or not case_path.is_file():
                        raise AssertionError(case)
                    request = load_json_object(request_path)
                    if json_fingerprint(request) != case.get("request_canonical_sha256"):
                        raise AssertionError(case)
                    if json_fingerprint(load_json_object(response_path)) != case.get(
                        "response_canonical_sha256"
                    ):
                        raise AssertionError(case)
                    persisted_case = load_json_object(case_path)
                    persisted_sha = persisted_case.pop("canonical_sha256", None)
                    if json_fingerprint(persisted_case) != persisted_sha:
                        raise AssertionError(case)
                    if scenario_name == "negative-api-matrix":
                        if case.get("preset") != "P_DETERMINISTIC":
                            raise AssertionError(case)
                        for key, value in deterministic_sampling.items():
                            if request.get(key) != value:
                                raise AssertionError({"case": case, "request": request})
                        if request.get("chat_template_kwargs") != {"enable_thinking": False}:
                            raise AssertionError({"case": case, "request": request})
                    elif case.get("category") == "text-array":
                        if case.get("preset") != "P_DETERMINISTIC":
                            raise AssertionError(case)
                        for key, value in deterministic_sampling.items():
                            if request.get(key) != value:
                                raise AssertionError({"case": case, "request": request})
                        if (
                            request.get("max_tokens") != 1024
                            or request.get("chat_template_kwargs")
                            != {"enable_thinking": False}
                        ):
                            raise AssertionError({"case": case, "request": request})
                    else:
                        forbidden = {
                            *deterministic_sampling,
                            "max_tokens",
                            "chat_template_kwargs",
                        }
                        leaked = sorted(key for key in forbidden if key in request)
                        if leaked or case.get("preset") is not None:
                            raise AssertionError(
                                {"case": case, "request": request, "leaked": leaked}
                            )
            health = load_json_object(out / "server.health.json")
            if health.get("status") != "pass" or health.get("http_status") != 200:
                raise AssertionError(health)
            health_after = load_json_object(out / "server.health.after.json")
            if health_after.get("status") != "pass" or health_after.get("http_status") != 200:
                raise AssertionError(health_after)
            bad_out = root / "bad-observability-out"
            bad = run_selftest_command(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(bad_out),
                    "--observability",
                ]
            )
            if bad.returncode == 0:
                raise AssertionError("external-server observability unexpectedly passed")
            if "observability requires manifest.server.mode=start" not in bad.stderr:
                raise AssertionError(bad.stderr or bad.stdout)
    finally:
        server.shutdown()
        server.server_close()
    print("BACKEND SCENARIO RUNNER SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ferrum-bin")
    parser.add_argument("--model")
    parser.add_argument("--backend")
    parser.add_argument("--base-url")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--observability", action="store_true")
    parser.add_argument("--profile-detail")
    parser.add_argument("--profile-sample-rate", type=float)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.manifest is None:
        parser.error("--manifest is required unless --self-test is set")
    if args.out is None:
        parser.error("--out is required unless --self-test is set")
    try:
        manifest = load_json_object(args.manifest)
        summary = ScenarioRunner(args, manifest).run_all()
    except Exception as exc:
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            write_json(args.out / "summary.json", {"status": "fail", "error": str(exc)})
        print(f"BACKEND REGRESSION SMOKE FAIL: {args.out}: {exc}", file=sys.stderr)
        return 1
    if summary["status"] == "pass":
        print(summary["pass_line"])
        return 0
    print(f"BACKEND REGRESSION SMOKE FAIL: {args.out}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
