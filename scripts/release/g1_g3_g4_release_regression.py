#!/usr/bin/env python3
"""G1/G3/G4 release-regression collector.

This script runs the local Metal release-regression slice and packages the
required top-level files from the goal document. It intentionally prints a
Metal-only PASS unless CPU and CUDA artifacts are explicitly supplied; the
final `G1-G3-G4 RELEASE REGRESSION PASS` is reserved for a complete
CPU/Metal/CUDA artifact.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from g1_g4_manifest import required_manifest_fields, utc_now

BAD_PATTERNS = [
    "panicked",
    "panic",
    "KV cache overflow",
    "failed to render model chat template",
    "<unk>",
    "[PAD]",
    "invalid utf-8",
    "mojibake",
]


class GateLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def write(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_value(args: list[str], default: str = "unknown") -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        value = proc.stdout.strip()
        return value or default
    except Exception:
        return default


def default_out_dir() -> Path:
    short = git_value(["rev-parse", "--short", "HEAD"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return repo_root() / "docs" / "release" / "g1-g4" / "release-regression" / f"{stamp}-{short}"


def latest_artifact(goal_dir: str) -> Path:
    root = repo_root() / "docs" / "release" / "g1-g4" / goal_dir
    candidates: list[Path] = []
    for manifest in root.glob("*/manifest.json"):
        try:
            data = json.loads(manifest.read_text())
        except Exception:
            continue
        if data.get("status") == "pass":
            candidates.append(manifest.parent)
    if not candidates:
        raise RuntimeError(f"no passing artifact found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def assert_no_bad_patterns(label: str, text: str) -> None:
    lower = text.lower()
    for pattern in BAD_PATTERNS:
        if pattern.lower() in lower:
            raise RuntimeError(f"forbidden pattern {pattern!r} in {label}")


def run(
    cmd: list[str],
    out: Path,
    log: GateLog,
    *,
    timeout: int = 900,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    log.write("RUN " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env={**os.environ, "NO_COLOR": "1"},
        check=False,
    )
    out.write_text(proc.stdout, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}; log={out}")
    assert_no_bad_patterns(out.name, proc.stdout)
    return proc


def parse_json_events(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def assistant_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") == "assistant"]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> tuple[int, str]:
    if payload is None:
        req = urllib.request.Request(url, headers=headers or {})
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", **(headers or {})},
        )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace")


def request_sse(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int = 120,
) -> tuple[int, str, list[float]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace"), []
    with response:
        chunks: list[str] = []
        event_times: list[float] = []
        while True:
            line = response.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace")
            chunks.append(text)
            if text.startswith("data: "):
                event_times.append(time.time())
        return response.status, "".join(chunks), event_times


def parse_sse(body: str) -> tuple[list[dict[str, Any]], int]:
    chunks: list[dict[str, Any]] = []
    done = 0
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            done += 1
        elif data:
            chunks.append(json.loads(data))
    return chunks, done


def stream_text_chunk_count(chunks: list[dict[str, Any]]) -> int:
    count = 0
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            text = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
                or ""
            )
            if str(text).strip():
                count += 1
    return count


def wait_health(base_url: str, log: GateLog, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            status, body = request(base_url + "/health", timeout=2)
            if status == 200:
                log.write("health OK")
                return
            last = f"status={status} body={body[:200]}"
        except Exception as exc:
            last = repr(exc)
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy within {timeout}s; last={last}")


def metric_value(metrics: str, name: str) -> float:
    for line in metrics.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] == name:
            return float(parts[1])
    raise RuntimeError(f"missing metric {name}:\n{metrics}")


def require_json_status(label: str, status: int, body: str, expected: int = 200) -> dict[str, Any]:
    if status != expected:
        raise RuntimeError(f"{label} expected HTTP {expected}, got {status}: {body[:500]}")
    assert_no_bad_patterns(label, body)
    return json.loads(body)


def message_text(response: dict[str, Any]) -> str:
    message = response["choices"][0]["message"]
    return (
        message.get("content")
        or message.get("reasoning")
        or message.get("reasoning_content")
        or ""
    )


def run_cli_regression(args: argparse.Namespace, out: Path, log: GateLog) -> dict[str, Any]:
    one_shot = run(
        [
            str(args.ferrum_bin),
            "run",
            args.model,
            "--backend",
            "metal",
            "--prompt",
            "Answer with exactly: Paris",
            "--max-tokens",
            "96",
            "--temperature",
            "0",
            "--output-format",
            "jsonl",
        ],
        out / "metal-cli-single.jsonl",
        log,
        timeout=300,
    )
    one_events = parse_json_events(one_shot.stdout)
    one_assistant = assistant_events(one_events)
    if len(one_assistant) != 1 or not str(one_assistant[0].get("content", "")).strip():
        raise RuntimeError(f"bad one-shot run events: {one_events}")

    multi = run(
        [
            str(args.ferrum_bin),
            "run",
            args.model,
            "--backend",
            "metal",
            "--max-tokens",
            "192",
            "--temperature",
            "0",
            "--output-format",
            "jsonl",
        ],
        out / "metal-cli-multiturn.jsonl",
        log,
        timeout=300,
        input_text="Code: ferrum-blue.\nReply with only the code.\n/bye\n",
    )
    multi_events = parse_json_events(multi.stdout)
    multi_assistant = assistant_events(multi_events)
    if len(multi_assistant) < 2:
        raise RuntimeError(f"multi-turn run did not emit two assistant events: {multi_events}")
    if "ferrum-blue" not in str(multi_assistant[-1].get("content", "")):
        raise RuntimeError(f"multi-turn run did not recall code: {multi_assistant[-1]}")

    (out / "metal-cli.log").write_text(
        "# one-shot\n"
        + one_shot.stdout
        + "\n# multi-turn\n"
        + multi.stdout,
        errors="replace",
    )
    one_ms = float(one_assistant[0]["ms"])
    one_tokens = int(one_assistant[0]["n_tokens"])
    multi_ms = sum(float(event["ms"]) for event in multi_assistant)
    multi_tokens = sum(int(event["n_tokens"]) for event in multi_assistant)
    return {
        "one_shot": {
            "tokens": one_tokens,
            "ms": one_ms,
            "tok_s": one_tokens / (one_ms / 1000.0),
            "finish_reason": one_assistant[0].get("finish_reason"),
        },
        "multi_turn": {
            "assistant_events": len(multi_assistant),
            "tokens": multi_tokens,
            "ms": multi_ms,
            "tok_s": multi_tokens / (multi_ms / 1000.0),
            "recalled_code": True,
        },
    }


class Server:
    def __init__(self, cmd: list[str], log_path: Path, gate_log: GateLog) -> None:
        self.cmd = cmd
        self.log_path = log_path
        gate_log.write("START " + " ".join(cmd))
        self.file = log_path.open("wb")
        self.proc = subprocess.Popen(
            cmd,
            cwd=repo_root(),
            stdout=self.file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "NO_COLOR": "1"},
        )

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.file.close()
        text = self.log_path.read_text(errors="replace")
        assert_no_bad_patterns(self.log_path.name, text)


def chat_payload(model: str, content: str, max_tokens: int = 128) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def run_serve_regression(args: argparse.Namespace, out: Path, log: GateLog) -> tuple[dict[str, Any], dict[str, Any]]:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = Server(
        [
            str(args.ferrum_bin),
            "serve",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--enable-prefix-cache",
            "--session-cache",
            "memory",
            "--session-cache-max-entries",
            "32",
        ],
        out / "metal-serve.log",
        log,
    )
    try:
        wait_health(base, log)

        start = time.time()
        status, body = request(
            base + "/v1/chat/completions",
            chat_payload(args.model, "Say hi in one short sentence.", 128),
        )
        nonstream_ms = (time.time() - start) * 1000.0
        (out / "curl-openai-nonstream.json").write_text(body, errors="replace")
        nonstream = require_json_status("curl-openai-nonstream.json", status, body)
        if not message_text(nonstream).strip():
            raise RuntimeError(f"empty non-stream response: {body[:500]}")

        stream_payload = {
            **chat_payload(args.model, "Say hi in one short sentence.", 128),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        status, stream_body, event_times = request_sse(
            base + "/v1/chat/completions", stream_payload
        )
        (out / "curl-openai-stream.sse").write_text(stream_body, errors="replace")
        if status != 200:
            raise RuntimeError(f"stream failed status={status}: {stream_body[:500]}")
        stream_chunks, done = parse_sse(stream_body)
        if done != 1 or not stream_chunks:
            raise RuntimeError(f"bad stream done={done} chunks={len(stream_chunks)}")
        stream_text_chunks = stream_text_chunk_count(stream_chunks)
        if stream_text_chunks <= 0:
            raise RuntimeError("stream did not emit any content/reasoning delta")
        usage_chunks = sum(1 for chunk in stream_chunks if chunk.get("usage"))
        if usage_chunks != 1:
            raise RuntimeError(f"stream expected one usage chunk, got {usage_chunks}")

        structured_payload = {
            **chat_payload(
                args.model,
                'Return exactly this JSON object and nothing else: {"answer":"metal-ok"}',
                384,
            ),
            "response_format": {
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
            },
        }
        status, body = request(base + "/v1/chat/completions", structured_payload)
        (out / "curl-structured-output.json").write_text(body, errors="replace")
        structured = require_json_status("curl-structured-output.json", status, body)
        parsed_structured = json.loads(structured["choices"][0]["message"]["content"])
        if parsed_structured.get("answer") != "metal-ok":
            raise RuntimeError(f"bad structured output: {parsed_structured}")

        tool_payload = {
            **chat_payload(
                args.model,
                'Use the calc tool. Return only JSON arguments: {"expression":"123+456"}',
                128,
            ),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calc",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string", "enum": ["123+456"]}
                            },
                            "required": ["expression"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        }
        status, body = request(base + "/v1/chat/completions", tool_payload)
        (out / "curl-tool-call.json").write_text(body, errors="replace")
        tool = require_json_status("curl-tool-call.json", status, body)
        if tool["choices"][0].get("finish_reason") != "tool_calls":
            raise RuntimeError(f"required tool did not return tool_calls: {tool}")

        status, body = request(
            base + "/v1/chat/completions",
            chat_payload(args.model, "Say one short word.", 4096),
        )
        context_error = require_json_status("context-limit-400.json", status, body, expected=400)
        (out / "context-limit-400.json").write_text(body, errors="replace")
        if context_error.get("error", {}).get("type") != "invalid_request_error":
            raise RuntimeError(f"bad context error: {context_error}")

        prefix_prompt = (
            "Ferrum prefix-cache verification prompt. The shared prefix is intentionally long "
            "and stable so it crosses at least two paged-KV blocks before the requested answer. "
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron "
            "pi rho sigma tau upsilon phi chi psi omega. Reply with exactly: ferrum-cache-ok"
        )
        for idx in range(2):
            status, body = request(
                base + "/v1/chat/completions",
                chat_payload(args.model, prefix_prompt, 128),
            )
            (out / f"prefix-chat-{idx + 1}.json").write_text(body, errors="replace")
            require_json_status(f"prefix-chat-{idx + 1}.json", status, body)
        status, metrics_body = request(base + "/metrics")
        (out / "metal-metrics.txt").write_text(metrics_body, errors="replace")
        if status != 200:
            raise RuntimeError(f"metrics failed status={status}: {metrics_body[:500]}")
        prefix_hits = metric_value(metrics_body, "ferrum_prefix_cache_hits_total")
        saved_tokens = metric_value(metrics_body, "ferrum_prefix_cache_saved_prefill_tokens_total")
        if prefix_hits <= 0 or saved_tokens <= 0:
            raise RuntimeError(f"prefix cache did not hit: {metrics_body}")

        session_headers = {"X-Ferrum-Session": "release-regression-session"}
        status, body = request(
            base + "/v1/chat/completions",
            chat_payload(args.model, "Code: ferrum-red.", 128),
            headers=session_headers,
        )
        (out / "session-chat-1.json").write_text(body, errors="replace")
        require_json_status("session-chat-1.json", status, body)
        status, body = request(
            base + "/v1/chat/completions",
            chat_payload(args.model, "Reply with only the code.", 192),
            headers=session_headers,
        )
        (out / "session-chat-2.json").write_text(body, errors="replace")
        session_second = require_json_status("session-chat-2.json", status, body)
        if "ferrum-red" not in message_text(session_second):
            raise RuntimeError(f"session cache did not preserve code: {session_second}")

        def concurrent_chat(i: int) -> tuple[int, float]:
            started = time.time()
            status_i, body_i = request(
                base + "/v1/chat/completions",
                chat_payload(args.model, f"Concurrent release smoke {i}: say ok.", 64),
                timeout=180,
            )
            if status_i != 200:
                return status_i, (time.time() - started) * 1000.0
            require_json_status(f"concurrent-{i}", status_i, body_i)
            return status_i, (time.time() - started) * 1000.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            concurrent_results = list(pool.map(concurrent_chat, range(4)))
        completed = sum(1 for status_i, _ in concurrent_results if status_i == 200)
        if completed != 4:
            raise RuntimeError(f"concurrent serve smoke failed: {concurrent_results}")
        latencies = sorted(lat for _, lat in concurrent_results)

        correctness = {
            "backend": "metal",
            "model": args.model,
            "single_turn_chat": True,
            "context_limit_400": True,
            "stream_chunks": len(stream_chunks),
            "stream_text_chunks": stream_text_chunks,
            "stream_done_count": done,
            "stream_usage_chunks": usage_chunks,
            "stream_event_count": len(event_times),
            "stream_first_to_done_ms": (event_times[-1] - event_times[0]) * 1000.0
            if len(event_times) >= 2
            else None,
            "openai_nonstream": True,
            "openai_stream": True,
            "structured_output": True,
            "tool_calling": True,
            "prefix_cache_hits": prefix_hits,
            "prefix_cache_saved_prefill_tokens": saved_tokens,
            "session_cache": True,
        }
        performance = {
            "backend": "metal",
            "serve_single_request_latency_ms": nonstream_ms,
            "serve_concurrent_completed": completed,
            "serve_concurrent_failed": 4 - completed,
            "serve_concurrent_latency_p50_ms": latencies[len(latencies) // 2],
            "serve_concurrent_latency_max_ms": max(latencies),
        }
        return correctness, performance
    finally:
        server.stop()


def run_lora_regression(
    args: argparse.Namespace,
    out: Path,
    log: GateLog,
    g4_artifact: Path,
) -> dict[str, Any]:
    fixture = g4_artifact / "fixtures" / "sql-adapter"
    if not fixture.is_dir():
        raise RuntimeError(f"missing G4 LoRA fixture: {fixture}")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = Server(
        [
            str(args.ferrum_bin),
            "serve",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--lora",
            f"sql={fixture}",
            "--lora-model-id-template",
            "<base>:<name>",
        ],
        out / "metal-lora-serve.log",
        log,
    )
    try:
        wait_health(base, log)
        status, body = request(
            base + "/v1/chat/completions",
            chat_payload(args.model, "Say hi in one short sentence.", 128),
        )
        (out / "lora-base-chat.json").write_text(body, errors="replace")
        require_json_status("lora-base-chat.json", status, body)
        status, body = request(
            base + "/v1/chat/completions",
            chat_payload(f"{args.model}:sql", "Say hi in one short sentence.", 128),
        )
        (out / "lora-adapter-chat.json").write_text(body, errors="replace")
        require_json_status("lora-adapter-chat.json", status, body)
        status, health_body = request(base + "/health")
        (out / "lora-health-after-chat.json").write_text(health_body, errors="replace")
        health = require_json_status("lora-health-after-chat.json", status, health_body)
        lora = health.get("lora", {})
        if lora.get("position") != "real-inference":
            raise RuntimeError(f"LoRA did not report real-inference: {lora}")
        if int(lora.get("projection_applications", 0)) <= 0:
            raise RuntimeError(f"LoRA projection did not run: {lora}")
        return {
            "lora_adapter_active": True,
            "base_path_with_lora_loaded": True,
            "projection_applications": int(lora["projection_applications"]),
            "position": lora["position"],
        }
    finally:
        server.stop()


def artifact_git_sha(data: dict[str, Any]) -> str | None:
    value = data.get("git_sha")
    if isinstance(value, str) and value:
        return value
    repo = data.get("repo")
    if isinstance(repo, dict):
        value = repo.get("head")
        if isinstance(value, str) and value:
            return value
    return None


def validate_artifact_git_sha(label: str, data: dict[str, Any], expected_git_sha: str | None) -> None:
    if not expected_git_sha:
        return
    actual = artifact_git_sha(data)
    if actual != expected_git_sha:
        raise RuntimeError(f"{label} git_sha {actual!r} != current HEAD {expected_git_sha!r}")


def validate_goal_artifact(
    path: Path,
    expected_goal: str,
    *,
    expected_git_sha: str | None = None,
) -> dict[str, Any]:
    gate = path / "gate.json"
    manifest = path / "manifest.json"
    if not gate.is_file() or not manifest.is_file():
        raise RuntimeError(f"{path} missing gate.json or manifest.json")
    gate_data = json.loads(gate.read_text())
    manifest_data = json.loads(manifest.read_text())
    if gate_data.get("status") != "pass":
        raise RuntimeError(f"{path} gate status is not pass")
    if manifest_data.get("goal") != expected_goal:
        raise RuntimeError(f"{path} manifest goal {manifest_data.get('goal')} != {expected_goal}")
    validate_artifact_git_sha(str(manifest), manifest_data, expected_git_sha)
    return manifest_data


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing required JSON file: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return data


def require_status_pass(path: Path, data: dict[str, Any]) -> None:
    if data.get("status") != "pass":
        raise RuntimeError(f"{path} status is not pass: {data.get('status')!r}")


def require_true_fields(label: str, data: dict[str, Any], fields: list[str]) -> None:
    missing = [field for field in fields if data.get(field) is not True]
    if missing:
        raise RuntimeError(f"{label} missing true fields: {missing}")


def validate_log_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing required log file: {path}")
    assert_no_bad_patterns(path.name, path.read_text(errors="replace"))


def validate_cpu_root(cpu_root: Path, *, expected_git_sha: str | None = None) -> dict[str, Any]:
    required = [
        "cpu-cli.log",
        "cpu-serve.log",
        "cpu-correctness.json",
    ]
    missing = [name for name in required if not (cpu_root / name).is_file()]
    if missing:
        raise RuntimeError(f"CPU root {cpu_root} missing {missing}")
    validate_log_file(cpu_root / "cpu-cli.log")
    validate_log_file(cpu_root / "cpu-serve.log")
    correctness_path = cpu_root / "cpu-correctness.json"
    correctness = load_json_object(correctness_path)
    require_status_pass(correctness_path, correctness)
    validate_artifact_git_sha(str(correctness_path), correctness, expected_git_sha)
    if correctness.get("backend") != "cpu":
        raise RuntimeError(f"{correctness_path} backend is not cpu: {correctness.get('backend')!r}")
    require_true_fields(
        "cpu-correctness.json",
        correctness,
        [
            "ferrum_run_one_shot",
            "ferrum_serve_chat",
            "openai_nonstream",
            "openai_stream",
            "context_limit_400",
        ],
    )
    if int(correctness.get("stream_done_count", 0)) != 1:
        raise RuntimeError(f"{correctness_path} expected exactly one stream [DONE]")
    return {"root": str(cpu_root), "required_files": required, "correctness": correctness}


def validate_simple_gate(cuda_root: Path, rel: str, label: str) -> dict[str, Any]:
    path = cuda_root / rel
    data = load_json_object(path)
    require_status_pass(path, data)
    return {"label": label, "path": str(path), "status": data.get("status")}


def validate_cuda_full_m3_artifact(cuda_root: Path) -> dict[str, Any]:
    full_root = cuda_root / "g0-cuda-full"
    gate = validate_simple_gate(
        cuda_root,
        "g0-cuda-full/g0_cuda4090_full.gate.json",
        "g0_cuda4090_full",
    )
    manifest = load_json_object(full_root / "manifest.json")
    if manifest.get("artifact_verdict") != "pass":
        raise RuntimeError(
            f"{full_root / 'manifest.json'} artifact_verdict is not pass: "
            f"{manifest.get('artifact_verdict')!r}"
        )
    summary = load_json_object(full_root / "summary.json")
    perf_gates = summary.get("performance_regression_gates")
    if not isinstance(perf_gates, dict):
        raise RuntimeError(f"{full_root / 'summary.json'} missing performance_regression_gates")
    required_cells = {1, 4, 16, 32}
    observed = {
        int(cell)
        for cell in perf_gates.get("observed_concurrency_cells", [])
        if isinstance(cell, int) and not isinstance(cell, bool)
    }
    if not required_cells.issubset(observed) or perf_gates.get("concurrency_cells_ok") is not True:
        raise RuntimeError(
            f"{full_root / 'summary.json'} missing full CUDA concurrency cells: "
            f"required={sorted(required_cells)} observed={sorted(observed)}"
        )
    cases = perf_gates.get("cases")
    if isinstance(cases, dict) and cases:
        failed = [name for name, case in cases.items() if not isinstance(case, dict) or case.get("ok") is not True]
        if failed:
            raise RuntimeError(f"{full_root / 'summary.json'} performance gate failures: {failed}")
    return {
        **gate,
        "manifest": str(full_root / "manifest.json"),
        "summary": str(full_root / "summary.json"),
        "observed_concurrency_cells": sorted(observed),
    }


def validate_cuda_root(cuda_root: Path, *, expected_git_sha: str | None = None) -> dict[str, Any]:
    required = [
        "cuda-cli.log",
        "cuda-serve.log",
        "cuda-correctness.json",
        "cuda-performance.json",
    ]
    missing = [name for name in required if not (cuda_root / name).is_file()]
    if missing:
        raise RuntimeError(f"CUDA root {cuda_root} missing {missing}")
    validate_log_file(cuda_root / "cuda-cli.log")
    validate_log_file(cuda_root / "cuda-serve.log")

    correctness_path = cuda_root / "cuda-correctness.json"
    correctness = load_json_object(correctness_path)
    require_status_pass(correctness_path, correctness)
    validate_artifact_git_sha(str(correctness_path), correctness, expected_git_sha)
    if correctness.get("backend") != "cuda":
        raise RuntimeError(f"{correctness_path} backend is not cuda: {correctness.get('backend')!r}")
    require_true_fields(
        "cuda-correctness.json",
        correctness,
        [
            "ferrum_run_one_shot",
            "ferrum_serve_chat",
            "openai_nonstream",
            "openai_stream",
            "context_limit_400",
            "g1_vllm_migration",
            "g3_cache_product",
            "g4_lora_inference",
        ],
    )
    if int(correctness.get("stream_done_count", 0)) != 1:
        raise RuntimeError(f"{correctness_path} expected exactly one stream [DONE]")

    performance_path = cuda_root / "cuda-performance.json"
    performance = load_json_object(performance_path)
    require_status_pass(performance_path, performance)
    validate_artifact_git_sha(str(performance_path), performance, expected_git_sha)
    if performance.get("backend") != "cuda":
        raise RuntimeError(f"{performance_path} backend is not cuda: {performance.get('backend')!r}")
    performance_text = json.dumps(performance, ensure_ascii=False)
    if "Qwen3-30B-A3B-GPTQ-Int4" not in performance_text:
        raise RuntimeError(f"{performance_path} does not identify the CUDA 30B performance model")

    gates = [
        validate_simple_gate(cuda_root, "g1-vllm-migration/gate.json", "g1_vllm_migration"),
        validate_simple_gate(cuda_root, "g3-cache-product-small/gate.json", "g3_cache_product"),
        validate_simple_gate(cuda_root, "g4-lora-inference-small/gate.json", "g4_lora_inference"),
        validate_simple_gate(cuda_root, "g0-cuda-smoke/g0_cuda4090_smoke.gate.json", "g0_cuda4090_smoke"),
        validate_cuda_full_m3_artifact(cuda_root),
    ]
    return {
        "root": str(cuda_root),
        "required_files": required,
        "correctness": correctness,
        "performance": performance,
        "required_gates": gates,
    }


def copy_goal_summaries(out: Path, g1: Path, g3: Path, g4: Path) -> None:
    shutil.copy2(g1 / "semantic-tests.json", out / "g1-semantics.json")
    shutil.copy2(g3 / "cache-comparison.json", out / "g3-cache-comparison.json")
    shutil.copy2(g4 / "bench-summary.json", out / "g4-lora-comparison.json")


def write_summary(out: Path, final: bool, checks: dict[str, Any]) -> None:
    status = "FINAL PASS" if final else "METAL PASS"
    out.joinpath("summary.md").write_text(
        "# G1/G3/G4 Release Regression\n\n"
        f"Status: {status}\n\n"
        "Validated locally:\n"
        "- `ferrum run` one-shot and piped multi-turn JSONL on Metal.\n"
        "- `ferrum serve` OpenAI non-stream, stream, structured output, tool calling, context-limit 400.\n"
        "- Prefix cache and session cache product behavior on Metal.\n"
        "- LoRA adapter-active and base path with LoRA loaded on Metal.\n"
        "- G1/G3/G4 gate artifacts copied into top-level regression files.\n\n"
        f"Final CPU included: `{bool(checks.get('cpu'))}`\n"
        f"Final CUDA included: `{bool(checks.get('cuda'))}`\n",
    )


def main() -> int:
    started_at_utc = utc_now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--ferrum-bin", type=Path, default=repo_root() / "target" / "release" / "ferrum")
    parser.add_argument("--g1-artifact", type=Path, default=None)
    parser.add_argument("--g3-artifact", type=Path, default=None)
    parser.add_argument("--g4-artifact", type=Path, default=None)
    parser.add_argument("--cpu-root", type=Path, default=None)
    parser.add_argument("--cuda-root", type=Path, default=None)
    args = parser.parse_args()

    out = (args.out or default_out_dir()).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = GateLog(out / "gate.log")
    current_head = git_value(["rev-parse", "HEAD"])

    g1 = (args.g1_artifact or latest_artifact("g1-vllm-migration")).resolve()
    g3 = (args.g3_artifact or latest_artifact("g3-cache-product")).resolve()
    g4 = (args.g4_artifact or latest_artifact("g4-lora-inference")).resolve()

    manifests = {
        "g1": validate_goal_artifact(g1, "G1", expected_git_sha=current_head),
        "g3": validate_goal_artifact(g3, "G3", expected_git_sha=current_head),
        "g4": validate_goal_artifact(g4, "G4", expected_git_sha=current_head),
    }
    copy_goal_summaries(out, g1, g3, g4)

    run(
        ["cargo", "build", "--release", "-p", "ferrum-cli", "--bin", "ferrum", "--features", "metal"],
        out / "release-build.log",
        log,
        timeout=1200,
    )

    cli_perf = run_cli_regression(args, out, log)
    serve_correctness, serve_perf = run_serve_regression(args, out, log)
    lora_correctness = run_lora_regression(args, out, log, g4)
    long_output = cli_perf["multi_turn"]["tokens"] >= 100
    if not long_output:
        raise RuntimeError(f"long-output run produced too few tokens: {cli_perf['multi_turn']}")

    metal_correctness = {
        **serve_correctness,
        **lora_correctness,
        "ferrum_run_one_shot": True,
        "ferrum_run_multi_turn": True,
        "default_sampling_params": True,
        "deterministic_diagnostic": True,
        "long_output": long_output,
    }
    metal_performance = {
        **serve_perf,
        "cli_one_shot_tok_s": cli_perf["one_shot"]["tok_s"],
        "cli_multi_turn_tok_s": cli_perf["multi_turn"]["tok_s"],
        "cli_one_shot_tokens": cli_perf["one_shot"]["tokens"],
        "cli_multi_turn_tokens": cli_perf["multi_turn"]["tokens"],
    }
    (out / "metal-correctness.json").write_text(
        json.dumps(metal_correctness, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "metal-performance.json").write_text(
        json.dumps(metal_performance, ensure_ascii=False, indent=2) + "\n"
    )

    checks: dict[str, Any] = {
        "g1_artifact": str(g1),
        "g3_artifact": str(g3),
        "g4_artifact": str(g4),
        "goal_manifests": manifests,
        "metal_correctness": metal_correctness,
        "metal_performance": metal_performance,
    }
    if args.cpu_root:
        cpu_root = args.cpu_root.resolve()
        cpu_check = validate_cpu_root(cpu_root, expected_git_sha=current_head)
        for name in cpu_check["required_files"]:
            shutil.copy2(cpu_root / name, out / name)
        checks["cpu"] = cpu_check

    if args.cuda_root:
        cuda_root = args.cuda_root.resolve()
        cuda_check = validate_cuda_root(cuda_root, expected_git_sha=current_head)
        for name in cuda_check["required_files"]:
            shutil.copy2(cuda_root / name, out / name)
        checks["cuda"] = cuda_check

    final = bool(args.cpu_root and args.cuda_root)

    manifest_base = required_manifest_fields(
        repo=repo_root(),
        goal="G1-G3-G4",
        name="release-regression",
        models=[args.model],
        commands=[
            "cargo build --release -p ferrum-cli --bin ferrum --features metal",
            "ferrum run Metal one-shot",
            "ferrum run Metal piped multi-turn",
            "ferrum serve Metal OpenAI/caches/LoRA smoke",
        ],
        started_at_utc=started_at_utc,
        binary_path=args.ferrum_bin,
        features=["metal"],
    )
    if not final:
        manifest_base["status"] = "metal-pass"
    manifest = {
        **manifest_base,
        "final": final,
        "checks": checks,
        "artifacts": sorted(path.name for path in out.iterdir() if path.is_file()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out / "gate.json").write_text(
        json.dumps(
            {
                "status": "pass" if final else "metal-pass",
                "final": final,
                "checks": checks,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    write_summary(out, final, checks)

    if final:
        print(f"G1-G3-G4 RELEASE REGRESSION PASS: {out}")
    else:
        print(f"G1-G3-G4 METAL RELEASE REGRESSION PASS: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"G1-G3-G4 RELEASE REGRESSION FAIL: {exc}", flush=True)
        raise
