#!/usr/bin/env python3
"""OpenAI-compatible tool-call regression probe for a running Ferrum server."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

FORBIDDEN_TEXT = [
    "<|assistant|>",
    "<|tool|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_",
    "classname=",
    "auto_tool_response",
]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]

TOOL_USER_PROMPT = (
    "北京现在天气怎么样？请先调用 get_weather 工具查询，"
    "得到工具结果后用一句中文回答，unit 使用 celsius。"
)
TOOL_REQUIRED_PROMPT = (
    "Call get_weather exactly once with city set to Beijing and unit set to celsius. "
    "Do not output natural language."
)
TOOL_SEED = 9271


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def post(base_url: str, payload: dict[str, Any], timeout: int = 180) -> tuple[int, str]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def parsed_json(label: str, status: int, body: str) -> dict[str, Any]:
    if status != 200:
        raise RuntimeError(f"{label}: expected HTTP 200, got {status}: {body[:500]}")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{label}: invalid JSON response: {e}: {body[:500]}") from e
    return data


def first_choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"missing choices: {data}")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(f"invalid choice: {choice!r}")
    return choice


def message(choice: dict[str, Any]) -> dict[str, Any]:
    msg = choice.get("message")
    if not isinstance(msg, dict):
        raise RuntimeError(f"missing message: {choice}")
    return msg


def tool_calls_from_message(msg: dict[str, Any]) -> list[dict[str, Any]]:
    calls = msg.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise RuntimeError(f"expected non-empty tool_calls, got: {msg}")
    return [call for call in calls if isinstance(call, dict)]


def assert_no_forbidden_text(label: str, text: str) -> None:
    for token in FORBIDDEN_TEXT:
        if token in text:
            raise RuntimeError(f"{label}: leaked forbidden template text {token!r}: {text[:500]}")


def compact_text(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def assert_not_duplicate_answer(label: str, text: str) -> None:
    compact = compact_text(text)
    if len(compact) < 24 or len(compact) % 2 != 0:
        return
    half = len(compact) // 2
    if compact[:half] == compact[half:]:
        raise RuntimeError(f"{label}: answer appears duplicated: {text[:500]!r}")


def assert_weather_tool_call(label: str, choice: dict[str, Any]) -> dict[str, Any]:
    if choice.get("finish_reason") != "tool_calls":
        raise RuntimeError(f"{label}: finish_reason != tool_calls: {choice}")
    msg = message(choice)
    content = msg.get("content") or ""
    assert_no_forbidden_text(label, content)
    if content.strip():
        raise RuntimeError(f"{label}: tool-call turn leaked text content: {content[:500]!r}")
    calls = tool_calls_from_message(msg)
    call = calls[0]
    function = call.get("function")
    if not isinstance(function, dict):
        raise RuntimeError(f"{label}: missing function object: {call}")
    if function.get("name") != "get_weather":
        raise RuntimeError(f"{label}: wrong tool name: {function}")
    args_raw = function.get("arguments")
    if not isinstance(args_raw, str):
        raise RuntimeError(f"{label}: arguments is not a string: {function}")
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{label}: arguments is not JSON: {args_raw!r}") from e
    city_raw = str(args.get("city", ""))
    city = city_raw.lower()
    if len(city_raw) > 64:
        raise RuntimeError(f"{label}: city argument is unexpectedly long: {args}")
    if "北京" not in city and "beijing" not in city:
        raise RuntimeError(f"{label}: city argument does not identify Beijing: {args}")
    return call


def assert_omitted_tool_choice_auto_calls_weather(label: str, choice: dict[str, Any]) -> None:
    msg = message(choice)
    content = str(msg.get("content") or "")
    assert_no_forbidden_text(label, content)
    assert_weather_tool_call(label, choice)


def tool_payload(model: str, *, tool_choice: Any | None, prompt: str = TOOL_USER_PROMPT) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "seed": TOOL_SEED,
        "tools": TOOLS,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 256,
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


def run_tool_call_regression(base_url: str, model: str, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"model": model, "checks": {}}

    # BUG-1: Omitting tool_choice must still behave as auto, not return an empty stop
    # or a plain-text fake tool result when the user explicitly asks to call a tool.
    status, body = post(base_url, tool_payload(model, tool_choice=None))
    write(out / "01_omitted_tool_choice.response.json", body)
    omitted = first_choice(parsed_json("omitted_tool_choice", status, body))
    assert_omitted_tool_choice_auto_calls_weather("omitted_tool_choice", omitted)
    results["checks"]["omitted_tool_choice"] = {"passed": True}

    # Explicit auto must also work. Keep the prompt identical to the omitted
    # tool_choice case so this checks request handling rather than prompt wording.
    status, body = post(base_url, tool_payload(model, tool_choice="auto"))
    write(out / "01b_explicit_auto_tool_choice.response.json", body)
    explicit_auto = first_choice(parsed_json("explicit_auto_tool_choice", status, body))
    assert_omitted_tool_choice_auto_calls_weather("explicit_auto_tool_choice", explicit_auto)
    results["checks"]["explicit_auto_tool_choice"] = {"passed": True}

    # BUG-3: Some Llama templates emit {"auto":{"tool":...}} text. The server must
    # convert that to OpenAI tool_calls and must not leak reserved template tokens.
    required_choice = {"type": "function", "function": {"name": "get_weather"}}
    status, body = post(
        base_url,
        tool_payload(model, tool_choice=required_choice, prompt=TOOL_REQUIRED_PROMPT),
    )
    write(out / "02_required_tool_choice.response.json", body)
    required = first_choice(parsed_json("required_tool_choice", status, body))
    required_call = assert_weather_tool_call("required_tool_choice", required)
    results["checks"]["required_tool_choice"] = {"passed": True}

    # BUG-2: Feeding a tool result back must not leak chat-template markers such as
    # <|assistant|>, classname=..., or auto_tool_response into the final answer.
    tool_call_id = str(required_call.get("id") or "call_0")
    fill_payload = {
        "model": model,
        "temperature": 0,
        "seed": TOOL_SEED,
        "tools": TOOLS,
        "tool_choice": "none",
        "messages": [
            {
                "role": "user",
                "content": TOOL_USER_PROMPT,
            },
            message(required),
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(
                    {
                        "city": "北京",
                        "temp": 22,
                        "unit": "celsius",
                        "desc": "晴",
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "user",
                "content": (
                    "Use the tool result above to answer the original question in one short "
                    "sentence. Include the numeric temperature from the tool result."
                ),
            },
        ],
        "max_tokens": 256,
    }
    write(
        out / "03_tool_result_fill.request.json",
        json.dumps(fill_payload, ensure_ascii=False, indent=2) + "\n",
    )
    status, body = post(base_url, fill_payload)
    write(out / "03_tool_result_fill.response.json", body)
    fill = first_choice(parsed_json("tool_result_fill", status, body))
    fill_msg = message(fill)
    fill_text = str(fill_msg.get("content") or "")
    assert_no_forbidden_text("tool_result_fill", fill_text)
    assert_not_duplicate_answer("tool_result_fill", fill_text)
    if not fill_text.strip():
        raise RuntimeError("tool_result_fill: final answer is empty")
    if "22" not in fill_text:
        raise RuntimeError(f"tool_result_fill: answer did not use tool result: {fill_text[:500]}")
    if fill.get("finish_reason") not in {"stop", "length"}:
        raise RuntimeError(f"tool_result_fill: unexpected finish_reason: {fill}")
    results["checks"]["tool_result_fill"] = {
        "passed": True,
        "finish_reason": fill.get("finish_reason"),
    }

    results["status"] = "pass"
    write(out / "tool_call_regression.json", json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        run_tool_call_regression(args.base_url, args.model, args.out)
    except Exception as e:
        args.out.mkdir(parents=True, exist_ok=True)
        write(
            args.out / "tool_call_regression.json",
            json.dumps({"status": "fail", "model": args.model, "error": str(e)}, ensure_ascii=False, indent=2)
            + "\n",
        )
        print(f"OPENAI TOOL CALL REGRESSION FAIL: {e}", file=sys.stderr)
        return 1
    print(f"OPENAI TOOL CALL REGRESSION PASS: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
