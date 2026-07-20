#!/usr/bin/env python3
"""Manifest-driven product regression scenario runner.

The runner is intentionally small and explicit: a scenario manifest names the
product path (`ferrum run`, external `ferrum serve`, or a server started by this
script), each scenario writes its own JSON artifact, and the runner writes a
summary artifact plus one PASS line.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import pty
import re
import select
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
from typing import Any

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
    "serve_structured_output",
    "serve_response_format_matrix",
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


class ScenarioError(Exception):
    pass


class StartedServer:
    def __init__(self, cmd: list[str], log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self.file = log_path.open("wb")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "NO_COLOR": "1"},
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
        assert_no_bad_text(self.log_path.name, self.log_path.read_text(errors="replace"))


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
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def get_url(url: str, *, timeout: int) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


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
        return exc.code, exc.read().decode("utf-8", "replace"), []
    chunks: list[str] = []
    event_times: list[float] = []
    with response:
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace")
            chunks.append(line)
            if line.startswith("data: "):
                event_times.append(time.time())
    return response.status, "".join(chunks), event_times


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
        if parsed.get("usage"):
            usage_chunks += 1
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
    allowed = set(only)
    return [scenario for scenario in scenarios if str(scenario.get("name")) in allowed]


class ScenarioRunner:
    def __init__(self, args: argparse.Namespace, manifest: dict[str, Any]) -> None:
        self.args = args
        self.manifest = manifest
        self.out = args.out
        self.model = args.model or str(manifest.get("model") or "")
        self.backend = args.backend or str(manifest.get("backend") or "auto")
        self.ferrum_bin = Path(args.ferrum_bin or manifest.get("ferrum_bin") or "target/release/ferrum")
        self.timeout = int(args.timeout or manifest.get("timeout_sec") or 180)
        self.base_url = args.base_url or self.manifest_base_url()
        self.started_server: StartedServer | None = None
        self.run_observability_roots: list[Path] = []

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
        self.started_server = StartedServer(cmd, self.out / "server.log")
        capture_health(self.base_url, self.out, self.timeout)

    def run_all(self) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        started_at = iso_now()
        scenarios = self.scenarios()
        matrix_contract = response_format_matrix_contract(scenarios)
        write_json(self.out / "response_format_matrix_contract.json", matrix_contract)
        selected = selected_scenarios(scenarios, self.args.only)
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
        status = "fail" if failures else "pass"
        summary = {
            "schema_version": 1,
            "status": status,
            "manifest": str(self.args.manifest),
            "artifact_dir": str(self.out),
            "model": self.model,
            "backend": self.backend,
            "base_url": self.base_url,
            "git_sha": git_output(["rev-parse", "HEAD"]),
            "dirty_status": git_dirty_status(),
            "started_at": started_at,
            "finished_at": iso_now(),
            "scenario_count": len(results),
            "failed": failures,
            "skipped": skipped,
            "scenarios": results,
            "response_format_matrix_contract": {
                "artifact": str(self.out / "response_format_matrix_contract.json"),
                "case_counts": matrix_contract["case_counts"],
                "unique_json_schema_count": matrix_contract["unique_json_schema_count"],
            },
            "observability": observability,
            "pass_line": f"BACKEND REGRESSION SMOKE PASS: {self.out}" if status == "pass" else None,
        }
        write_json(self.out / "summary.json", summary)
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
        if typ == "serve_structured_output":
            return self.serve_structured_output(scenario, out)
        if typ == "serve_response_format_matrix":
            return self.serve_response_format_matrix(scenario, out)
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
            require(str(expected) in last, f"run_multiturn did not recall {expected!r}: {last[:500]}")
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

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body)
        echo_marker = self.echo_value_marker(payload)
        if payload.get("stream"):
            if echo_marker is not None:
                self.send_stream_tool_call(echo_marker)
            else:
                self.send_stream()
            return
        if int(payload.get("max_tokens") or 0) >= 4096:
            self.send_json(
                400,
                {"error": {"type": "invalid_request_error", "message": "context limit"}},
            )
            return
        messages = payload.get("messages", [])
        prompt = " ".join(str(msg.get("content") or "") for msg in messages)
        last_user = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user = str(msg.get("content") or "")
                break
        marker = re.search(r"\b(ferrum\d{2}\d{2})\b", prompt)
        square = re.search(r"(S\d{4})", prompt)
        exact_object = self.exact_json_object(last_user)
        response_format = payload.get("response_format")
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
            self.send_chat(json.dumps(exact_object, separators=(",", ":")))
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

    def send_stream(self) -> None:
        lines = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "scenario"}, "finish_reason": None}
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}},
        ]
        raw = "".join(f"data: {json.dumps(line)}\n\n" for line in lines) + "data: [DONE]\n\n"
        body = raw.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
                raise AssertionError(proc.stderr or proc.stdout)
            if f"BACKEND REGRESSION SMOKE PASS: {out}" not in proc.stdout:
                raise AssertionError(proc.stdout)
            summary = load_json_object(out / "summary.json")
            if summary.get("status") != "pass" or summary.get("scenario_count") != 11:
                raise AssertionError(summary)
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
