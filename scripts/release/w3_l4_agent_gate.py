#!/usr/bin/env python3
"""Run W3 L4 agent checks against a running OpenAI-compatible Ferrum server."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_NAME = "w3_l4_agent.json"
PASS_LINE_PREFIX = "W3 L4 AGENT PASS"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
FORBIDDEN_TEXT = [
    "<unk>",
    "[PAD",
    "<pad>",
    "<|im_start|>",
    "<|im_end|>",
    "<|reserved_special_token",
    "classname=",
    "auto_tool_response",
    "\ufffd",
]


class GateError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def post_json(base_url: str, payload: dict[str, Any], timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")


def parse_json(label: str, status: int, body: str) -> dict[str, Any]:
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GateError(f"{label}: invalid JSON response: {exc}: {body[:500]}") from exc
    if not isinstance(data, dict):
        raise GateError(f"{label}: response must be a JSON object")
    if status < 200 or status >= 300:
        raise GateError(f"{label}: expected HTTP 2xx, got {status}: {body[:500]}")
    return data


def first_choice(label: str, data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GateError(f"{label}: missing choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise GateError(f"{label}: invalid first choice")
    return choice


def message(label: str, choice: dict[str, Any]) -> dict[str, Any]:
    msg = choice.get("message")
    if not isinstance(msg, dict):
        raise GateError(f"{label}: missing message")
    return msg


def assert_no_forbidden(label: str, text: str) -> None:
    for token in FORBIDDEN_TEXT:
        if token in text:
            raise GateError(f"{label}: leaked forbidden text {token!r}: {text[:500]!r}")


def calc_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate one deterministic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "enum": ["123+456"]}},
                "required": ["expression"],
            },
        },
    }


def strict_answer_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "Answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    }


def tool_payload(model: str, iteration: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the calc tool. Return only JSON arguments: "
                    '{"expression":"123+456"}'
                ),
            }
        ],
        "tools": [calc_tool()],
        "tool_choice": "required",
        "temperature": 0,
        "seed": 9271 + iteration,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def strict_payload(model: str, iteration: int) -> dict[str, Any]:
    expected = f"ok-{iteration}"
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return exactly this JSON object and nothing else: "
                    f'{{"answer":"{expected}"}}'
                ),
            }
        ],
        "temperature": 0,
        "seed": 9271 + iteration,
        "max_tokens": 128,
        "response_format": strict_answer_schema(),
        "chat_template_kwargs": {"enable_thinking": False},
    }


def validate_tool_call(label: str, data: dict[str, Any]) -> dict[str, Any]:
    choice = first_choice(label, data)
    if choice.get("finish_reason") != "tool_calls":
        raise GateError(f"{label}: finish_reason must be tool_calls: {choice}")
    msg = message(label, choice)
    content = str(msg.get("content") or "")
    assert_no_forbidden(label, content)
    if content.strip():
        raise GateError(f"{label}: tool-call turn leaked content: {content[:500]!r}")
    calls = msg.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise GateError(f"{label}: missing tool_calls: {msg}")
    call = calls[0]
    if not isinstance(call, dict):
        raise GateError(f"{label}: invalid tool call: {call!r}")
    function = call.get("function")
    if not isinstance(function, dict):
        raise GateError(f"{label}: missing function object: {call}")
    if function.get("name") != "calc":
        raise GateError(f"{label}: wrong tool name: {function}")
    raw_args = function.get("arguments")
    if not isinstance(raw_args, str):
        raise GateError(f"{label}: function.arguments must be string: {function}")
    try:
        parsed_args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise GateError(f"{label}: arguments are not JSON: {raw_args[:500]!r}") from exc
    if parsed_args.get("expression") != "123+456":
        raise GateError(f"{label}: wrong expression args: {parsed_args}")
    return {"finish_reason": choice.get("finish_reason"), "arguments": parsed_args}


def validate_strict_schema(label: str, data: dict[str, Any], expected: str) -> dict[str, Any]:
    choice = first_choice(label, data)
    msg = message(label, choice)
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise GateError(f"{label}: missing strict JSON content: {msg}")
    assert_no_forbidden(label, content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GateError(f"{label}: content is not JSON: {content[:500]!r}") from exc
    if parsed != {"answer": expected}:
        raise GateError(f"{label}: strict JSON mismatch: {parsed!r} != {{'answer': {expected!r}}}")
    if choice.get("finish_reason") == "length":
        raise GateError(f"{label}: strict JSON finished by length")
    return {"finish_reason": choice.get("finish_reason"), "content": parsed}


def validate_negative_contracts(base_url: str, model: str, out: Path, timeout: float) -> dict[str, Any]:
    bad_tool = {
        "model": model,
        "messages": [{"role": "user", "content": "call missing"}],
        "tools": [calc_tool()],
        "tool_choice": {"type": "function", "function": {"name": "missing"}},
        "temperature": 0,
    }
    write_json(out / "negative_tool_choice.request.json", bad_tool)
    status, body = post_json(base_url, bad_tool, timeout)
    write_text(out / "negative_tool_choice.response.json", body)
    if status != 400:
        raise GateError(f"negative tool_choice expected HTTP 400, got {status}: {body[:500]}")

    bad_format = {
        "model": model,
        "messages": [{"role": "user", "content": "return xml"}],
        "response_format": {"type": "xml"},
        "temperature": 0,
    }
    write_json(out / "negative_response_format.request.json", bad_format)
    status, body = post_json(base_url, bad_format, timeout)
    write_text(out / "negative_response_format.response.json", body)
    if status != 400:
        raise GateError(f"negative response_format expected HTTP 400, got {status}: {body[:500]}")
    return {"tool_choice_400": True, "response_format_400": True}


def run_l4(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    negative = validate_negative_contracts(args.base_url, args.model, out / "negative", args.timeout)

    tool_results: list[dict[str, Any]] = []
    for idx in range(args.tool_total):
        label = f"tool_{idx:02d}"
        payload = tool_payload(args.model, idx)
        write_json(out / "tool_calls" / f"{label}.request.json", payload)
        status, body = post_json(args.base_url, payload, args.timeout)
        write_text(out / "tool_calls" / f"{label}.response.json", body)
        data = parse_json(label, status, body)
        tool_results.append({"id": label, "passed": True, **validate_tool_call(label, data)})

    strict_results: list[dict[str, Any]] = []
    for idx in range(args.strict_total):
        label = f"strict_schema_{idx:02d}"
        payload = strict_payload(args.model, idx)
        write_json(out / "strict_schema" / f"{label}.request.json", payload)
        status, body = post_json(args.base_url, payload, args.timeout)
        write_text(out / "strict_schema" / f"{label}.response.json", body)
        data = parse_json(label, status, body)
        strict_results.append(
            {
                "id": label,
                "passed": True,
                **validate_strict_schema(label, data, f"ok-{idx}"),
            }
        )

    artifact = {
        "schema_version": 1,
        "status": "pass",
        "level": "l4_agent",
        "model_id": args.release_model_id,
        "product_surface": "typed_cli",
        "hidden_env": [],
        "generated_at": iso_now(),
        "pass_line": f"{PASS_LINE_PREFIX}: {out}",
        "base_url": args.base_url,
        "request_model": args.model,
        "agent": {
            "real_model": True,
            "required_tool_enforced": negative["tool_choice_400"] and len(tool_results) == args.tool_total,
            "json_schema_strict": negative["response_format_400"]
            and len(strict_results) == args.strict_total,
            "tool_calls_total": len(tool_results),
            "tool_calls_passed": sum(1 for result in tool_results if result["passed"]),
            "strict_schema_total": len(strict_results),
            "strict_schema_passed": sum(1 for result in strict_results if result["passed"]),
        },
        "negative_contracts": negative,
        "tool_call_cases": tool_results,
        "strict_schema_cases": strict_results,
    }
    write_json(out / ARTIFACT_NAME, artifact)
    return artifact


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrum-w3-l4-agent-") as tmp:
        root = Path(tmp)
        args = argparse.Namespace(
            out=root,
            base_url="http://127.0.0.1:1",
            model="selftest-model",
            release_model_id=DEFAULT_MODEL_ID,
            timeout=1.0,
            tool_total=10,
            strict_total=20,
        )
        tool_results = [
            {"id": f"tool_{idx:02d}", "passed": True, "finish_reason": "tool_calls"}
            for idx in range(10)
        ]
        strict_results = [
            {"id": f"strict_schema_{idx:02d}", "passed": True, "finish_reason": "stop"}
            for idx in range(20)
        ]
        artifact = {
            "schema_version": 1,
            "status": "pass",
            "level": "l4_agent",
            "model_id": args.release_model_id,
            "product_surface": "typed_cli",
            "hidden_env": [],
            "generated_at": iso_now(),
            "pass_line": f"{PASS_LINE_PREFIX}: {root}",
            "agent": {
                "real_model": True,
                "required_tool_enforced": True,
                "json_schema_strict": True,
                "tool_calls_total": len(tool_results),
                "tool_calls_passed": len(tool_results),
                "strict_schema_total": len(strict_results),
                "strict_schema_passed": len(strict_results),
            },
            "negative_contracts": {
                "tool_choice_400": True,
                "response_format_400": True,
            },
            "tool_call_cases": tool_results,
            "strict_schema_cases": strict_results,
        }
        write_json(root / ARTIFACT_NAME, artifact)
        from model_release_grade_goal_gate import validate_w3_l0_l5_artifact  # type: ignore

        problems: list[str] = []
        validate_w3_l0_l5_artifact(
            "l4_agent",
            {"artifact": str(root / ARTIFACT_NAME)},
            root,
            problems,
        )
        if problems:
            raise AssertionError(f"L4 selftest artifact failed validator: {problems}")
    print("W3 L4 AGENT SELFTEST PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=False)
    parser.add_argument("--model", required=False)
    parser.add_argument("--release-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tool-total", type=int, default=10)
    parser.add_argument("--strict-total", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            return run_selftest()
        if not args.base_url:
            raise GateError("missing required arg: --base-url")
        if not args.model:
            raise GateError("missing required arg: --model")
        if args.out is None:
            raise GateError("missing required arg: --out")
        artifact = run_l4(args)
    except GateError as exc:
        print(f"W3 L4 AGENT FAIL: {exc}", file=sys.stderr)
        return 1
    print(artifact["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
