#!/usr/bin/env python3
"""Run real Qwen3.5 product-path checks and write W3 release artifacts.

This script is the runner that feeds the existing W3 validators. It runs
`ferrum run`, starts `ferrum serve`, exercises OpenAI chat completions, and
writes:

- known_answer_report.json, consumed by w3_l2_quantized_gate.py
- w3_l3_behavior.json, consumed by model_release_grade_goal_gate.py
- w3_s2_whole_model_product_path.json, consumed by model_release_grade_goal_gate.py

It is intentionally real-model oriented. The self-test only verifies artifact
assembly and validation logic; it is not release evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL_DOC = "docs/goals/model-coverage-2026-06-12/W3_QWEN35_RELEASE_GRADE_GOAL.md"
PASS_LINE_PREFIX = "W3 QWEN35 REAL PRODUCT REPORT PASS"
S2_PASS_LINE_PREFIX = "W3 QWEN35 REAL PRODUCT PATH PASS"
L3_PASS_LINE_PREFIX = "W3 L3 BEHAVIOR PASS"
KNOWN_REPORT_NAME = "known_answer_report.json"
L3_ARTIFACT_NAME = "w3_l3_behavior.json"
S2_ARTIFACT_NAME = "w3_s2_whole_model_product_path.json"
SUMMARY_NAME = "w3_qwen35_real_product_report.json"
MODEL_ID = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
QUANTIZED_FORMAT = "hf-gptq-int4"
FORBIDDEN_TEXT_PATTERNS = [
    "<unk>",
    "[PAD",
    "<pad>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|reserved_special_token",
    "\ufffd",
    "KV cache overflow",
    "panicked at",
    "panic:",
    "stream error",
]


class ReportError(Exception):
    pass


@dataclass(frozen=True)
class KnownAnswerCase:
    case_id: str
    prompt: str
    expected: str
    max_tokens: int
    predicate: Callable[[str], bool]


@dataclass(frozen=True)
class BehaviorCase:
    case_id: str
    passed: bool
    artifact: str
    detail: dict[str, Any]


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def artifact_ref(path: Path, out_dir: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    try:
        return path.relative_to(out_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        input=input_text,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git_output(args: list[str], *, default: str = "unknown") -> str:
    try:
        proc = run_command(["git", *args], cwd=REPO_ROOT)
    except OSError:
        return default
    return proc.stdout.strip() if proc.returncode == 0 else default


def git_summary() -> dict[str, Any]:
    tracked = [
        line
        for line in git_output(["status", "--short", "--untracked-files=no"], default="").splitlines()
        if line.strip()
    ]
    untracked = [
        line
        for line in git_output(["ls-files", "--others", "--exclude-standard"], default="").splitlines()
        if line.strip()
    ]
    return {
        "sha": git_output(["rev-parse", "HEAD"]),
        "is_dirty": bool(tracked or untracked),
        "tracked_status_short": tracked,
        "untracked_count": len(untracked),
        "untracked_sample": untracked[:20],
    }


def hardware_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        proc = run_command(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            timeout=15,
        )
        snapshot["nvidia_smi_returncode"] = proc.returncode
        snapshot["nvidia_smi"] = proc.stdout.strip().splitlines()
        if proc.stderr.strip():
            snapshot["nvidia_smi_stderr"] = proc.stderr.strip()
    nvcc = shutil.which("nvcc")
    if nvcc:
        proc = run_command([nvcc, "--version"], timeout=15)
        snapshot["nvcc_returncode"] = proc.returncode
        snapshot["nvcc_version"] = proc.stdout.strip()
    return snapshot


def binary_info(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None}
    info: dict[str, Any] = {"path": str(path)}
    if not path.is_file():
        info["exists"] = False
        return info
    info["exists"] = True
    info["size_bytes"] = path.stat().st_size
    sha = shutil.which("shasum")
    if sha:
        proc = run_command(["shasum", "-a", "256", str(path)], timeout=30)
        if proc.returncode == 0:
            info["sha256"] = proc.stdout.split()[0]
    return info


def default_ferrum_bin() -> Path | None:
    release_bin = REPO_ROOT / "target/release/ferrum"
    if release_bin.is_file():
        return release_bin
    debug_bin = REPO_ROOT / "target/debug/ferrum"
    if debug_bin.is_file():
        return debug_bin
    return None


def resolve_ferrum_bin(arg: Path | None) -> Path | None:
    if arg is not None:
        return arg.resolve()
    return default_ferrum_bin()


def ferrum_command(args: list[str], ferrum_bin: Path | None) -> list[str]:
    if ferrum_bin is not None:
        return [str(ferrum_bin), *args]
    return ["cargo", "run", "-p", "ferrum-cli", "--bin", "ferrum", "--", *args]


def runtime_flags(args: argparse.Namespace) -> list[str]:
    flags = ["--backend", args.backend]
    optional_flags = [
        ("--gpu-devices", args.gpu_devices),
        ("--gpu-memory-utilization", args.gpu_memory_utilization),
        ("--max-model-len", args.max_model_len),
        ("--max-num-seqs", args.max_num_seqs),
        ("--max-num-batched-tokens", args.max_num_batched_tokens),
        ("--kv-capacity", args.kv_capacity),
        ("--kv-max-blocks", args.kv_max_blocks),
        ("--kv-dtype", args.kv_dtype),
    ]
    for flag, value in optional_flags:
        if value is not None:
            flags.extend([flag, str(value)])
    return flags


def serve_runtime_flags(args: argparse.Namespace) -> list[str]:
    flags = runtime_flags(args)
    if args.runtime_preset is not None:
        flags.extend(["--runtime-preset", str(args.runtime_preset)])
    return flags


def run_runtime_flags(args: argparse.Namespace) -> list[str]:
    flags = runtime_flags(args)
    if args.disable_thinking:
        flags.append("--disable-thinking")
    return flags


def product_env(args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    if args.hf_home is not None:
        env["HF_HOME"] = str(args.hf_home)
    elif "HF_HOME" not in env:
        env["HF_HOME"] = str(out_dir / "hf-cache")

    scrubbed = sorted(key for key in env if key.startswith("FERRUM_"))
    if not args.preserve_ferrum_env:
        for key in scrubbed:
            env.pop(key, None)
        scrubbed = []
    return env, scrubbed


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    try:
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def http_get(url: str, timeout: float) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")


def http_post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")


def parse_json(text: str, label: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReportError(f"{label}: invalid JSON: {exc}: {text[:500]}") from exc
    if not isinstance(data, dict):
        raise ReportError(f"{label}: response must be a JSON object")
    return data


def parse_sse(body: str, label: str) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    done_count = 0
    malformed: list[str] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :].strip()
        if data == "[DONE]":
            done_count += 1
            continue
        if not data:
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            malformed.append(data[:200])
            continue
        if isinstance(parsed, dict):
            chunks.append(parsed)
        else:
            malformed.append(data[:200])

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    usage_chunks = 0
    for chunk in chunks:
        if chunk.get("usage") is not None:
            usage_chunks += 1
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            content_parts.append(content)
        reasoning = delta.get("reasoning")
        if isinstance(reasoning, str):
            reasoning_parts.append(reasoning)

    if malformed:
        raise ReportError(f"{label}: malformed SSE JSON chunks: {malformed[:3]}")
    return {
        "chunk_count": len(chunks),
        "done_count": done_count,
        "usage_chunks": usage_chunks,
        "has_usage": usage_chunks > 0,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "finish_reason": finish_reason,
    }


def first_choice(data: dict[str, Any], label: str) -> dict[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ReportError(f"{label}: missing choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ReportError(f"{label}: invalid first choice")
    return choice


def message_content(data: dict[str, Any], label: str) -> tuple[str, str | None, str | None]:
    choice = first_choice(data, label)
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ReportError(f"{label}: missing message")
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ReportError(f"{label}: content must be string")
    reasoning = message.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ReportError(f"{label}: reasoning must be string when present")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ReportError(f"{label}: finish_reason must be string when present")
    return content, reasoning, finish_reason


def assert_no_bad_text(label: str, text: str) -> None:
    for pattern in FORBIDDEN_TEXT_PATTERNS:
        if pattern in text:
            raise ReportError(f"{label}: forbidden output pattern {pattern!r}: {text[:500]!r}")


def contains_word(word: str) -> Callable[[str], bool]:
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    return lambda text: bool(pattern.search(text))


def contains_number(number: str) -> Callable[[str], bool]:
    pattern = re.compile(rf"(?<!\d){re.escape(number)}(?!\d)")
    return lambda text: bool(pattern.search(text))


KNOWN_ANSWER_CASES = [
    KnownAnswerCase(
        "addition_2_plus_3",
        "What is 2+3? Answer with only the number.",
        "5",
        8,
        contains_number("5"),
    ),
    KnownAnswerCase(
        "subtraction_7_minus_4",
        "What is 7-4? Answer with only the number.",
        "3",
        8,
        contains_number("3"),
    ),
    KnownAnswerCase(
        "capital_france",
        "What is the capital of France? Answer with one word.",
        "Paris",
        8,
        contains_word("paris"),
    ),
    KnownAnswerCase(
        "capital_italy",
        "What is the capital of Italy? Answer with one word.",
        "Rome",
        8,
        contains_word("rome"),
    ),
    KnownAnswerCase(
        "capital_japan",
        "What is the capital of Japan? Answer with one word.",
        "Tokyo",
        8,
        contains_word("tokyo"),
    ),
    KnownAnswerCase(
        "clear_sky_color",
        "What color is the sky on a clear day? Answer with one word.",
        "blue",
        8,
        contains_word("blue"),
    ),
    KnownAnswerCase(
        "opposite_hot",
        "What is the opposite of hot? Answer with one word.",
        "cold",
        8,
        contains_word("cold"),
    ),
    KnownAnswerCase(
        "alphabet_first_letter",
        "What is the first letter of the English alphabet? Answer with one letter.",
        "A",
        8,
        contains_word("a"),
    ),
    KnownAnswerCase(
        "days_in_week",
        "How many days are in a week? Answer with only the number.",
        "7",
        8,
        contains_number("7"),
    ),
    KnownAnswerCase(
        "freezing_water_celsius",
        "Does water freeze at 0 degrees Celsius? Answer yes or no.",
        "yes",
        8,
        contains_word("yes"),
    ),
]


def chat_payload(
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    stream: bool = False,
    stop: list[str] | None = None,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 9271,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if stop is not None:
        payload["stop"] = stop
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def wait_for_server(proc: subprocess.Popen[str], base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_url = f"{base_url}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ReportError(f"ferrum serve exited early with rc={proc.returncode}")
        try:
            status, _ = http_get(health_url, timeout=2.0)
            if 200 <= status < 300:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise ReportError(f"ferrum serve did not become healthy before {timeout_seconds}s")


def server_model_id(base_url: str, timeout: float, fallback: str) -> str:
    status, body = http_get(f"{base_url}/v1/models", timeout=timeout)
    if not 200 <= status < 300:
        return fallback
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return fallback
    data = parsed.get("data") if isinstance(parsed, dict) else None
    if not isinstance(data, list) or not data:
        return fallback
    first = data[0]
    if not isinstance(first, dict):
        return fallback
    model_id = first.get("id")
    return model_id if isinstance(model_id, str) and model_id else fallback


def validate_run_stdout(stdout: str) -> dict[str, Any]:
    assistant: dict[str, Any] | None = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "assistant":
            assistant = event
    if assistant is None:
        raise ReportError(f"missing assistant JSONL event in ferrum run stdout: {stdout[:500]}")
    content = assistant.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ReportError(f"ferrum run assistant content is empty: {assistant}")
    assert_no_bad_text("ferrum run", content)
    n_tokens = assistant.get("n_tokens")
    if not isinstance(n_tokens, int) or n_tokens <= 0:
        raise ReportError(f"ferrum run assistant n_tokens must be positive: {assistant}")
    if assistant.get("finish_reason") not in {"stop", "length"}:
        raise ReportError(f"ferrum run finish_reason must be stop or length: {assistant}")
    return assistant


def run_ferrum_run(
    args: argparse.Namespace,
    out_dir: Path,
    env: dict[str, str],
    ferrum_bin: Path | None,
) -> dict[str, Any]:
    case = KNOWN_ANSWER_CASES[0]
    cmd = ferrum_command(
        [
            "run",
            args.model,
            *run_runtime_flags(args),
            "--output-format",
            "jsonl",
            "--temperature",
            "0",
            "--max-tokens",
            str(case.max_tokens),
            "--prompt",
            case.prompt,
        ],
        ferrum_bin,
    )
    proc = run_command(cmd, cwd=out_dir, env=env, timeout=args.run_timeout_seconds)
    write_json(out_dir / "run_command.json", {"command_line": cmd, "returncode": proc.returncode})
    write_text(out_dir / "run_stdout.jsonl", proc.stdout)
    write_text(out_dir / "run_stderr.txt", proc.stderr)
    if proc.returncode != 0:
        raise ReportError(f"ferrum run failed with rc={proc.returncode}: {proc.stderr[-2000:]}")
    assistant = validate_run_stdout(proc.stdout)
    if not case.predicate(str(assistant.get("content", ""))):
        raise ReportError(
            f"ferrum run known-answer case {case.case_id} failed: "
            f"expected {case.expected!r}, got {assistant.get('content')!r}"
        )
    return {
        "status": "pass",
        "case_id": case.case_id,
        "command_line": cmd,
        "stdout": artifact_ref(out_dir / "run_stdout.jsonl", out_dir),
        "stderr": artifact_ref(out_dir / "run_stderr.txt", out_dir),
        "assistant_event": assistant,
    }


def evaluate_known_answer(
    case: KnownAnswerCase,
    content: str,
    *,
    artifact: str,
    finish_reason: str | None,
    entrypoint: str,
) -> dict[str, Any]:
    assert_no_bad_text(case.case_id, content)
    passed = case.predicate(content)
    return {
        "id": case.case_id,
        "entrypoint": entrypoint,
        "prompt": case.prompt,
        "expected": case.expected,
        "content": content,
        "finish_reason": finish_reason,
        "artifact": artifact,
        "semantic_pass": passed,
        "passed": passed,
    }


def post_chat(
    *,
    base_url: str,
    payload: dict[str, Any],
    out_dir: Path,
    rel_path: str,
    timeout: float,
) -> tuple[int, str, str]:
    status, body = http_post_json(f"{base_url}/v1/chat/completions", payload, timeout=timeout)
    path = out_dir / rel_path
    write_json(path.with_suffix(".request.json"), payload)
    write_text(path, body)
    return status, body, artifact_ref(path, out_dir)


def run_known_answer_cases(
    args: argparse.Namespace,
    out_dir: Path,
    base_url: str,
    request_model: str,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for idx, case in enumerate(KNOWN_ANSWER_CASES, start=1):
        payload = chat_payload(
            request_model,
            [{"role": "user", "content": case.prompt}],
            max_tokens=case.max_tokens,
            stream=False,
            enable_thinking=False,
        )
        status, body, artifact = post_chat(
            base_url=base_url,
            payload=payload,
            out_dir=out_dir,
            rel_path=f"known_answers/{idx:02d}_{case.case_id}.response.json",
            timeout=args.request_timeout_seconds,
        )
        if not 200 <= status < 300:
            raise ReportError(f"known-answer case {case.case_id} HTTP {status}: {body[:500]}")
        parsed = parse_json(body, case.case_id)
        content, _, finish_reason = message_content(parsed, case.case_id)
        cases.append(
            evaluate_known_answer(
                case,
                content,
                artifact=artifact,
                finish_reason=finish_reason,
                entrypoint="ferrum serve",
            )
        )
    return cases


def behavior_multi_turn(
    args: argparse.Namespace,
    out_dir: Path,
    base_url: str,
    request_model: str,
) -> BehaviorCase:
    payload = chat_payload(
        request_model,
        [
            {"role": "user", "content": "Remember the code word orchid. Reply only OK."},
            {"role": "assistant", "content": "OK."},
            {
                "role": "user",
                "content": "What code word did I ask you to remember? Answer with one word.",
            },
        ],
        max_tokens=12,
        stream=False,
        enable_thinking=False,
    )
    status, body, artifact = post_chat(
        base_url=base_url,
        payload=payload,
        out_dir=out_dir,
        rel_path="behavior/01_multi_turn.response.json",
        timeout=args.request_timeout_seconds,
    )
    if not 200 <= status < 300:
        raise ReportError(f"multi-turn HTTP {status}: {body[:500]}")
    parsed = parse_json(body, "multi_turn")
    content, _, finish_reason = message_content(parsed, "multi_turn")
    assert_no_bad_text("multi_turn", content)
    passed = bool(re.search(r"\borchid\b", content, flags=re.IGNORECASE))
    return BehaviorCase(
        "multi_turn",
        passed,
        artifact,
        {"content": content, "finish_reason": finish_reason},
    )


def behavior_stream_match(
    args: argparse.Namespace,
    out_dir: Path,
    base_url: str,
    request_model: str,
) -> tuple[BehaviorCase, dict[str, Any], dict[str, Any]]:
    prompt = "What is 2+3? Answer with only the number."
    nonstream_payload = chat_payload(
        request_model,
        [{"role": "user", "content": prompt}],
        max_tokens=8,
        stream=False,
        enable_thinking=False,
    )
    status, body, nonstream_artifact = post_chat(
        base_url=base_url,
        payload=nonstream_payload,
        out_dir=out_dir,
        rel_path="behavior/02_stream_match_nonstream.response.json",
        timeout=args.request_timeout_seconds,
    )
    if not 200 <= status < 300:
        raise ReportError(f"stream-match nonstream HTTP {status}: {body[:500]}")
    nonstream = parse_json(body, "stream_match_nonstream")
    nonstream_content, _, nonstream_finish = message_content(nonstream, "stream_match_nonstream")
    assert_no_bad_text("stream_match_nonstream", nonstream_content)

    stream_payload = chat_payload(
        request_model,
        [{"role": "user", "content": prompt}],
        max_tokens=8,
        stream=True,
        enable_thinking=False,
    )
    stream_status, stream_body, stream_artifact = post_chat(
        base_url=base_url,
        payload=stream_payload,
        out_dir=out_dir,
        rel_path="behavior/02_stream_match_stream.response.sse",
        timeout=args.request_timeout_seconds,
    )
    if not 200 <= stream_status < 300:
        raise ReportError(f"stream-match stream HTTP {stream_status}: {stream_body[:500]}")
    stream = parse_sse(stream_body, "stream_match_stream")
    stream_content = str(stream["content"])
    assert_no_bad_text("stream_match_stream", stream_content)
    passed = (
        contains_number("5")(nonstream_content)
        and contains_number("5")(stream_content)
        and stream["done_count"] == 1
        and stream["has_usage"] is True
    )
    case = BehaviorCase(
        "stream_nonstream_match",
        passed,
        stream_artifact,
        {
            "nonstream_artifact": nonstream_artifact,
            "nonstream_content": nonstream_content,
            "stream_content": stream_content,
            "stream_done_count": stream["done_count"],
            "stream_usage_chunks": stream["usage_chunks"],
        },
    )
    serve_nonstream_summary = {
        "artifact": nonstream_artifact,
        "finish_reason": nonstream_finish,
        "content_len": len(nonstream_content.strip()),
    }
    serve_stream_summary = {
        "artifact": stream_artifact,
        "chunk_count": int(stream["chunk_count"]),
        "done_count": int(stream["done_count"]),
        "has_usage": bool(stream["has_usage"]),
        "usage_chunks": int(stream["usage_chunks"]),
    }
    return case, serve_nonstream_summary, serve_stream_summary


def behavior_natural_eos(
    args: argparse.Namespace,
    out_dir: Path,
    base_url: str,
    request_model: str,
) -> BehaviorCase:
    payload = chat_payload(
        request_model,
        [{"role": "user", "content": "Reply with exactly this word: done"}],
        max_tokens=12,
        stream=False,
        enable_thinking=False,
    )
    status, body, artifact = post_chat(
        base_url=base_url,
        payload=payload,
        out_dir=out_dir,
        rel_path="behavior/03_natural_eos.response.json",
        timeout=args.request_timeout_seconds,
    )
    if not 200 <= status < 300:
        raise ReportError(f"natural-eos HTTP {status}: {body[:500]}")
    parsed = parse_json(body, "natural_eos")
    content, _, finish_reason = message_content(parsed, "natural_eos")
    assert_no_bad_text("natural_eos", content)
    passed = finish_reason == "stop" and bool(re.search(r"\bdone\b", content, re.IGNORECASE))
    return BehaviorCase(
        "natural_eos",
        passed,
        artifact,
        {"content": content, "finish_reason": finish_reason},
    )


def behavior_custom_stop(
    args: argparse.Namespace,
    out_dir: Path,
    base_url: str,
    request_model: str,
) -> BehaviorCase:
    payload = chat_payload(
        request_model,
        [{"role": "user", "content": "Repeat exactly this text: alpha <END> beta"}],
        max_tokens=24,
        stream=False,
        stop=["<END>"],
        enable_thinking=False,
    )
    status, body, artifact = post_chat(
        base_url=base_url,
        payload=payload,
        out_dir=out_dir,
        rel_path="behavior/04_custom_stop.response.json",
        timeout=args.request_timeout_seconds,
    )
    if not 200 <= status < 300:
        raise ReportError(f"custom-stop HTTP {status}: {body[:500]}")
    parsed = parse_json(body, "custom_stop")
    content, _, finish_reason = message_content(parsed, "custom_stop")
    assert_no_bad_text("custom_stop", content)
    passed = "<END>" not in content and "beta" not in content.lower() and finish_reason == "stop"
    return BehaviorCase(
        "custom_stop",
        passed,
        artifact,
        {"content": content, "finish_reason": finish_reason},
    )


def behavior_reasoning_extraction(
    args: argparse.Namespace,
    out_dir: Path,
    base_url: str,
    request_model: str,
) -> BehaviorCase:
    payload = chat_payload(
        request_model,
        [{"role": "user", "content": "Think briefly, then answer: what is 1+1?"}],
        max_tokens=128,
        stream=False,
        enable_thinking=True,
    )
    status, body, artifact = post_chat(
        base_url=base_url,
        payload=payload,
        out_dir=out_dir,
        rel_path="behavior/05_reasoning_extraction.response.json",
        timeout=args.request_timeout_seconds,
    )
    if not 200 <= status < 300:
        raise ReportError(f"reasoning-extraction HTTP {status}: {body[:500]}")
    parsed = parse_json(body, "reasoning_extraction")
    content, reasoning, finish_reason = message_content(parsed, "reasoning_extraction")
    assert_no_bad_text("reasoning_extraction_content", content)
    if reasoning:
        assert_no_bad_text("reasoning_extraction_reasoning", reasoning)
    passed = bool(reasoning and reasoning.strip()) and contains_number("2")(content)
    return BehaviorCase(
        "reasoning_extraction",
        passed,
        artifact,
        {
            "content": content,
            "reasoning_len": len(reasoning or ""),
            "finish_reason": finish_reason,
        },
    )


def run_behavior_cases(
    args: argparse.Namespace,
    out_dir: Path,
    base_url: str,
    request_model: str,
) -> tuple[list[BehaviorCase], dict[str, Any], dict[str, Any]]:
    cases: list[BehaviorCase] = []
    cases.append(behavior_multi_turn(args, out_dir, base_url, request_model))
    stream_case, serve_nonstream, serve_stream = behavior_stream_match(
        args,
        out_dir,
        base_url,
        request_model,
    )
    cases.append(stream_case)
    cases.append(behavior_natural_eos(args, out_dir, base_url, request_model))
    cases.append(behavior_custom_stop(args, out_dir, base_url, request_model))
    cases.append(behavior_reasoning_extraction(args, out_dir, base_url, request_model))
    return cases, serve_nonstream, serve_stream


def run_ferrum_serve(
    args: argparse.Namespace,
    out_dir: Path,
    env: dict[str, str],
    ferrum_bin: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[BehaviorCase]]:
    port = args.port if args.port is not None else free_port()
    cmd = ferrum_command(
        [
            "serve",
            args.model,
            "--host",
            args.host,
            "--port",
            str(port),
            *serve_runtime_flags(args),
        ],
        ferrum_bin,
    )
    base_url = f"http://{args.host}:{port}"
    log_path = out_dir / "serve.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=out_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_server(proc, base_url, args.serve_startup_timeout_seconds)
            request_model = args.request_model or server_model_id(
                base_url,
                args.request_timeout_seconds,
                args.model,
            )
            write_json(
                out_dir / "serve_command.json",
                {"command_line": cmd, "port": port, "request_model": request_model},
            )
            known_cases = run_known_answer_cases(args, out_dir, base_url, request_model)
            behavior_cases, serve_nonstream, serve_stream = run_behavior_cases(
                args,
                out_dir,
                base_url,
                request_model,
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    assert_no_bad_text("serve log", log_text)
    serve_result = {
        "status": "pass",
        "command_line": cmd,
        "base_url": base_url,
        "log": artifact_ref(log_path, out_dir),
        "nonstream": serve_nonstream,
        "stream": serve_stream,
    }
    return serve_result, known_cases, behavior_cases


def require_all_passed(label: str, cases: list[dict[str, Any] | BehaviorCase]) -> None:
    failed: list[str] = []
    for case in cases:
        if isinstance(case, BehaviorCase):
            if not case.passed:
                failed.append(case.case_id)
        elif not case.get("passed"):
            failed.append(str(case.get("id", "unknown")))
    if failed:
        raise ReportError(f"{label} failed cases: {', '.join(failed)}")


def build_known_answer_report(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    run_result: dict[str, Any],
    serve_result: dict[str, Any],
    known_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    run_case = evaluate_known_answer(
        KNOWN_ANSWER_CASES[0],
        str(run_result["assistant_event"]["content"]),
        artifact=str(run_result["stdout"]),
        finish_reason=str(run_result["assistant_event"].get("finish_reason")),
        entrypoint="ferrum run",
    )
    all_cases = [run_case, *known_cases]
    require_all_passed("known-answer", all_cases)
    report = {
        "schema_version": 1,
        "model_id": args.release_model_id,
        "format": args.quantized_format,
        "real_size_model": True,
        "waived": False,
        "semantic_pass": True,
        "known_answer_cases": all_cases,
        "known_answer_total": len(all_cases),
        "known_answer_passed": len(all_cases),
        "product_surface": "typed_cli",
        "hidden_env": [],
        "commands": [
            {"entrypoint": "ferrum run", "command_line": run_result["command_line"]},
            {"entrypoint": "ferrum serve", "command_line": serve_result["command_line"]},
        ],
        "product_entrypoints": ["ferrum run", "ferrum serve"],
        "generated_at": iso_now(),
    }
    write_json(out_dir / KNOWN_REPORT_NAME, report)
    return report


def build_l3_artifact(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    behavior_cases: list[BehaviorCase],
) -> dict[str, Any]:
    require_all_passed("behavior", behavior_cases)
    by_id = {case.case_id: case for case in behavior_cases}
    stream_case = by_id["stream_nonstream_match"]
    artifact = {
        "schema_version": 1,
        "status": "pass",
        "level": "l3_behavior",
        "model_id": args.release_model_id,
        "product_surface": "typed_cli",
        "hidden_env": [],
        "generated_at": iso_now(),
        "pass_line": f"{L3_PASS_LINE_PREFIX}: {out_dir}",
        "behavior": {
            "multi_turn": by_id["multi_turn"].passed,
            "stream_nonstream_match": stream_case.passed,
            "natural_eos": by_id["natural_eos"].passed,
            "custom_stop": by_id["custom_stop"].passed,
            "reasoning_extraction": by_id["reasoning_extraction"].passed,
            "stream_done_exactly_once": stream_case.detail.get("stream_done_count") == 1,
            "stream_usage_present": int(stream_case.detail.get("stream_usage_chunks", 0)) >= 1,
            "cases_total": len(behavior_cases),
            "cases_passed": sum(1 for case in behavior_cases if case.passed),
        },
        "cases": [
            {
                "id": case.case_id,
                "passed": case.passed,
                "artifact": case.artifact,
                "detail": case.detail,
            }
            for case in behavior_cases
        ],
    }
    write_json(out_dir / L3_ARTIFACT_NAME, artifact)
    return artifact


def build_s2_artifact(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    run_result: dict[str, Any],
    serve_result: dict[str, Any],
    ferrum_bin: Path | None,
    hidden_env: list[str],
) -> dict[str, Any]:
    if hidden_env:
        raise ReportError(
            "FERRUM_* hidden env overrides are preserved; rerun without "
            "--preserve-ferrum-env for release-grade evidence"
        )
    artifact = {
        "schema_version": 1,
        "status": "pass",
        "lane": "w3_s2_whole_model_product_path",
        "goal_doc": GOAL_DOC,
        "pass_line": f"{S2_PASS_LINE_PREFIX}: {out_dir}",
        "created_at": iso_now(),
        "command_line": command_line(),
        "git": git_summary(),
        "model_id": args.release_model_id,
        "architecture": "qwen3_5_moe",
        "backend": args.backend,
        "quantization": args.quantized_format,
        "runtime_surface": "typed_cli",
        "hidden_env": [],
        "binary": binary_info(ferrum_bin),
        "hardware": hardware_snapshot(),
        "product_entrypoints": {
            "ferrum_run": run_result,
            "ferrum_serve": serve_result,
        },
    }
    write_json(out_dir / S2_ARTIFACT_NAME, artifact)
    return artifact


def run_report(args: argparse.Namespace) -> int:
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    git = git_summary()
    if args.require_clean_git and git["is_dirty"]:
        raise ReportError(
            "git worktree is dirty; commit or stash changes before collecting "
            f"release-grade evidence: {git['tracked_status_short'][:5]}"
        )
    ferrum_bin = resolve_ferrum_bin(args.ferrum_bin)
    env, hidden_env = product_env(args, out_dir)
    if hidden_env:
        raise ReportError(
            "refusing release-grade run with preserved FERRUM_* hidden env: "
            + ", ".join(hidden_env)
        )

    run_result = run_ferrum_run(args, out_dir, env, ferrum_bin)
    serve_result, known_cases, behavior_cases = run_ferrum_serve(
        args,
        out_dir,
        env,
        ferrum_bin,
    )
    known_report = build_known_answer_report(
        args=args,
        out_dir=out_dir,
        run_result=run_result,
        serve_result=serve_result,
        known_cases=known_cases,
    )
    l3_artifact = build_l3_artifact(args=args, out_dir=out_dir, behavior_cases=behavior_cases)
    s2_artifact = build_s2_artifact(
        args=args,
        out_dir=out_dir,
        run_result=run_result,
        serve_result=serve_result,
        ferrum_bin=ferrum_bin,
        hidden_env=hidden_env,
    )
    summary = {
        "schema_version": 1,
        "status": "pass",
        "goal_doc": GOAL_DOC,
        "model_id": args.release_model_id,
        "backend": args.backend,
        "format": args.quantized_format,
        "generated_at": iso_now(),
        "pass_line": f"{PASS_LINE_PREFIX}: {out_dir}",
        "artifacts": {
            "known_answer_report": artifact_ref(out_dir / KNOWN_REPORT_NAME, out_dir),
            "l3_behavior": artifact_ref(out_dir / L3_ARTIFACT_NAME, out_dir),
            "w3_s2_whole_model_product_path": artifact_ref(out_dir / S2_ARTIFACT_NAME, out_dir),
        },
        "known_answer_total": known_report["known_answer_total"],
        "behavior_cases_total": l3_artifact["behavior"]["cases_total"],
        "s2_pass_line": s2_artifact["pass_line"],
    }
    write_json(out_dir / SUMMARY_NAME, summary)
    print(summary["pass_line"])
    return 0


def selftest_args(out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        out=out_dir,
        model=MODEL_ID,
        release_model_id=MODEL_ID,
        request_model=None,
        quantized_format=QUANTIZED_FORMAT,
        backend="cuda",
        host="127.0.0.1",
        port=None,
        ferrum_bin=None,
        hf_home=None,
        gpu_devices=None,
        gpu_memory_utilization=None,
        max_model_len=None,
        max_num_seqs=None,
        max_num_batched_tokens=None,
        kv_capacity=None,
        kv_max_blocks=None,
        kv_dtype=None,
        runtime_preset=None,
        require_clean_git=True,
        disable_thinking=True,
        preserve_ferrum_env=False,
        run_timeout_seconds=1,
        serve_startup_timeout_seconds=1,
        request_timeout_seconds=1,
    )


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="ferrum-w3-qwen35-real-report-") as tmp:
        out_dir = Path(tmp) / "out"
        out_dir.mkdir(parents=True)
        args = selftest_args(out_dir)
        write_text(out_dir / "run_stdout.jsonl", '{"event":"assistant","content":"5","finish_reason":"stop","n_tokens":1}\n')
        write_text(out_dir / "run_stderr.txt", "")
        write_text(out_dir / "serve.log", "server ok\n")
        write_text(out_dir / "behavior_stream.sse", "data: {}\n\ndata: [DONE]\n")
        run_result = {
            "status": "pass",
            "case_id": "addition_2_plus_3",
            "command_line": ["ferrum", "run", MODEL_ID, "--backend", "cuda"],
            "stdout": artifact_ref(out_dir / "run_stdout.jsonl", out_dir),
            "stderr": artifact_ref(out_dir / "run_stderr.txt", out_dir),
            "assistant_event": {
                "event": "assistant",
                "content": "5",
                "finish_reason": "stop",
                "n_tokens": 1,
            },
        }
        serve_result = {
            "status": "pass",
            "command_line": ["ferrum", "serve", MODEL_ID, "--backend", "cuda"],
            "log": artifact_ref(out_dir / "serve.log", out_dir),
            "nonstream": {
                "artifact": artifact_ref(out_dir / "run_stdout.jsonl", out_dir),
                "finish_reason": "stop",
                "content_len": 1,
            },
            "stream": {
                "artifact": artifact_ref(out_dir / "behavior_stream.sse", out_dir),
                "chunk_count": 1,
                "done_count": 1,
                "has_usage": True,
                "usage_chunks": 1,
            },
        }
        known_cases = [
            {
                "id": case.case_id,
                "entrypoint": "ferrum serve",
                "prompt": case.prompt,
                "expected": case.expected,
                "content": case.expected,
                "finish_reason": "stop",
                "artifact": artifact_ref(out_dir / "run_stdout.jsonl", out_dir),
                "semantic_pass": True,
                "passed": True,
            }
            for case in KNOWN_ANSWER_CASES
        ]
        behavior_cases = [
            BehaviorCase("multi_turn", True, artifact_ref(out_dir / "run_stdout.jsonl", out_dir), {}),
            BehaviorCase(
                "stream_nonstream_match",
                True,
                artifact_ref(out_dir / "behavior_stream.sse", out_dir),
                {"stream_done_count": 1, "stream_usage_chunks": 1},
            ),
            BehaviorCase("natural_eos", True, artifact_ref(out_dir / "run_stdout.jsonl", out_dir), {}),
            BehaviorCase("custom_stop", True, artifact_ref(out_dir / "run_stdout.jsonl", out_dir), {}),
            BehaviorCase(
                "reasoning_extraction",
                True,
                artifact_ref(out_dir / "run_stdout.jsonl", out_dir),
                {"reasoning_len": 8},
            ),
        ]
        known_report = build_known_answer_report(
            args=args,
            out_dir=out_dir,
            run_result=run_result,
            serve_result=serve_result,
            known_cases=known_cases,
        )
        l3_artifact = build_l3_artifact(args=args, out_dir=out_dir, behavior_cases=behavior_cases)
        s2_artifact = build_s2_artifact(
            args=args,
            out_dir=out_dir,
            run_result=run_result,
            serve_result=serve_result,
            ferrum_bin=None,
            hidden_env=[],
        )
        if known_report["known_answer_total"] < 10:
            raise AssertionError("known-answer self-test total is too small")
        if l3_artifact["behavior"]["cases_passed"] != 5:
            raise AssertionError("L3 behavior self-test did not pass all cases")
        if s2_artifact["product_entrypoints"]["ferrum_serve"]["stream"]["done_count"] != 1:
            raise AssertionError("S2 self-test did not preserve stream done count")

        from model_release_grade_goal_gate import (  # type: ignore
            validate_w3_l0_l5_artifact,
            validate_w3_s2_product_artifact,
        )
        from w3_l2_quantized_gate import build_artifact as build_l2_artifact  # type: ignore

        build_l2_artifact(
            report_path=out_dir / KNOWN_REPORT_NAME,
            out_dir=out_dir / "l2",
            model_id_override=None,
            format_override=None,
        )
        problems: list[str] = []
        validate_w3_l0_l5_artifact(
            "l3_behavior",
            {"artifact": str(out_dir / L3_ARTIFACT_NAME)},
            out_dir,
            problems,
        )
        validate_w3_s2_product_artifact(
            {"artifact": str(out_dir / S2_ARTIFACT_NAME)},
            out_dir,
            problems,
        )
        if problems:
            raise AssertionError(f"self-test artifacts failed existing validators: {problems}")

        bad_cases = list(known_cases)
        bad_cases[0] = dict(bad_cases[0])
        bad_cases[0]["passed"] = False
        try:
            build_known_answer_report(
                args=args,
                out_dir=out_dir / "bad",
                run_result=run_result,
                serve_result=serve_result,
                known_cases=bad_cases,
            )
        except ReportError as exc:
            if "known-answer failed cases" not in str(exc):
                raise AssertionError(f"unexpected failed-case self-test error: {exc}") from exc
        else:
            raise AssertionError("known-answer failure unexpectedly passed")

    print("W3 QWEN35 REAL PRODUCT REPORT SELFTEST PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="artifact output directory")
    parser.add_argument("--model", default=MODEL_ID, help="model id or local model path")
    parser.add_argument("--release-model-id", default=MODEL_ID, help="model id written to artifacts")
    parser.add_argument("--request-model", help="OpenAI request model id; default reads /v1/models")
    parser.add_argument("--quantized-format", default=QUANTIZED_FORMAT)
    parser.add_argument("--backend", default="cuda", choices=["cuda", "metal", "cpu", "auto"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--ferrum-bin", type=Path)
    parser.add_argument("--hf-home", type=Path, help="HF_HOME for model cache lookup/download")
    parser.add_argument("--gpu-devices")
    parser.add_argument("--gpu-memory-utilization")
    parser.add_argument("--max-model-len")
    parser.add_argument("--max-num-seqs")
    parser.add_argument("--max-num-batched-tokens")
    parser.add_argument("--kv-capacity")
    parser.add_argument("--kv-max-blocks")
    parser.add_argument("--kv-dtype")
    parser.add_argument("--runtime-preset")
    parser.add_argument(
        "--require-clean-git",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="refuse real evidence collection when the worktree is dirty",
    )
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pass --disable-thinking to ferrum run; serve requests still set chat_template_kwargs",
    )
    parser.add_argument(
        "--preserve-ferrum-env",
        action="store_true",
        help="diagnostic only; release artifacts require hidden_env to stay empty",
    )
    parser.add_argument("--run-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--serve-startup-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            return run_selftest()
        if args.out is None:
            raise ReportError("missing required arg: --out")
        return run_report(args)
    except subprocess.TimeoutExpired as exc:
        print(f"W3 QWEN35 REAL PRODUCT REPORT FAIL: timeout: {exc}", file=sys.stderr)
        return 1
    except ReportError as exc:
        print(f"W3 QWEN35 REAL PRODUCT REPORT FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
